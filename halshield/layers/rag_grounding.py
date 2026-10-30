"""RAGGrounding layer."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple

from ..contract import ABSTAIN_MSG, Node, State, require_question
from ..interfaces.layer import Layer
from ..interfaces.llm_client import LLMClient
from ..interfaces.retriever import Retriever
from ..prompts import CITATION_RULE
from ..retrieval.hybrid_retriever import HybridRetriever
from ..retrieval.langchain_retriever import LangChainRetriever, documents_to_chunks
from ..retrieval.query_expansion_retriever import QueryExpansionRetriever
from ..retrieval.query_rewriter import QueryRewriter
from ..utils import (
    append_citations,
    cited,
    is_conclusion,
    is_missing_info,
    look_alike_names,
    numbers,
    resolve_citations,
    split_sentences,
    strip_leading_connective,
)

DEFAULT_K = 3
_CHUNK_TAG = re.compile(r"^\[([^\]]+)\]")


class CitationFilterResult(NamedTuple):
    """Answer sentences kept and dropped by the citation filter."""

    kept: list[str]
    dropped: list[str]
    wrong_subject: tuple[str, ...] = ()  # cited, but the cited chunks only name a look-alike product


def chunk_ids(context: Sequence[str]) -> set[str]:
    """Return the ids of all "[id] text" entries in `context`."""
    return {match.group(1) for entry in context if (match := _CHUNK_TAG.match(entry))}


def chunk_texts(context: Sequence[str]) -> dict[str, str]:
    """Return {id: text} for the "[id] text" entries in `context`."""
    return {match.group(1): entry[match.end() :] for entry in context if (match := _CHUNK_TAG.match(entry))}


def covered_by(chunk: str, context: Sequence[str]) -> bool:
    """Return True if a "[doc#n] text" chunk is already in `context` as part of its whole "[doc] ..." document."""
    match = _CHUNK_TAG.match(chunk)
    if not match or "#" not in match.group(1):
        return False
    whole = chunk_texts(context).get(match.group(1).split("#", 1)[0], "")
    return chunk[match.end() :].strip() in whole


def as_context_entries(items: Sequence[Any]) -> list[str]:
    """Return `items` as "[id] text" strings: strings are kept, LangChain Documents are converted after them.

    Anything else raises TypeError, since a citation can only be checked against known chunks.
    """
    strings: list[str] = []
    documents: list[Any] = []
    for item in items:
        if isinstance(item, str):
            strings.append(item)
        elif hasattr(item, "page_content"):
            documents.append(item)
        else:
            raise TypeError(f"context entries must be strings or Documents, got {type(item).__name__}")
    return [*strings, *documents_to_chunks(documents)]


def restatements_removed(kept: list[str], removed: Sequence[str]) -> tuple[list[str], list[str]]:
    """Split `kept` into sentences that survive and those restating a removed claim.

    A sentence restates one if it cites a chunk the removed sentence cited and repeats one of its
    numbers: "The budget variant carries 0.6 kg [k3-spec]" after "The K3 Lite carries 0.6 kg [k3-spec]".
    """
    claims = [(set(cited(sentence)), numbers(sentence)) for sentence in removed]
    survivors: list[str] = []
    restated: list[str] = []
    for sentence in kept:
        ids, values = set(cited(sentence)), numbers(sentence)
        repeats = any(ids & claim_ids and values & claim_values for claim_ids, claim_values in claims)
        (restated if repeats else survivors).append(sentence)
    return survivors, restated


class RAGGrounding(Layer):
    """Inject retrieved "[id] text" chunks into `context` and enforce citations.

    With `require_citations`, the writer is asked to cite [id]s inline and every factual answer
    sentence that cites no context chunk is dropped (and flagged). Two uncited kinds are kept:
    plain "X is not mentioned" statements, and non-speculative "Therefore ..." conclusions, which
    inherit the citations of the sentences before them. With `check_subjects`, a cited sentence is
    also dropped when it names a model variant ("K3 Lite") that its cited chunks never mention
    although they name the base model ("K3"): a fact carried over from a look-alike product.

    Retrieval, one of:
    - `sources`: in-memory docs searched by the built-in HybridRetriever (no infrastructure);
    - `retriever`: a halshield Retriever, or any LangChain retriever / vector store / Runnable
      returning Documents (wrapped in LangChainRetriever automatically);
    - neither: the node does its own RAG. Nothing is retrieved; citations are checked against the
      `context` the node returns (or was given), as "[id] text" strings or LangChain Documents.
    `k` caps the number of chunks; when omitted, the built-in retriever uses 3 and a LangChain
    retriever keeps its own setting. `rewrite_queries=n` (n >= 1) also searches for n LLM-rewritten,
    document-style queries plus one hypothetical answer sentence (HyDE) and fuses the
    results, adding up to n + 1 chunks (one extra LLM call).
    """

    name = "grounding"

    def __init__(
        self,
        sources: Mapping[str, str] | Sequence[str] | None = None,
        require_citations: bool = True,
        k: int | None = None,
        keyword_weight: float = 0.3,
        llm: LLMClient | None = None,
        retriever: Retriever | Any | None = None,
        rewrite_queries: int = 0,
        check_subjects: bool = True,
    ) -> None:
        if rewrite_queries < 0:
            raise ValueError(f"rewrite_queries must be >= 0, got {rewrite_queries}")
        if retriever is not None and sources is not None:
            raise ValueError("pass either `sources` or `retriever`, not both")
        self.retriever: Retriever | None = None
        self.require_citations, self.check_subjects = require_citations, check_subjects
        if retriever is None and sources is None:  # the wrapped node retrieves for itself
            if rewrite_queries or k is not None:
                raise ValueError("`k` and `rewrite_queries` need `sources` or a `retriever` to search")
            return
        if retriever is None:
            retriever = HybridRetriever(
                sources, k=k or DEFAULT_K, keyword_weight=keyword_weight, llm=llm, label=self.name
            )
        elif not isinstance(retriever, Retriever):
            retriever = LangChainRetriever(retriever, k=k)
        if rewrite_queries:
            retriever = QueryExpansionRetriever(retriever, QueryRewriter(rewrite_queries, llm=llm))
        self.retriever = retriever

    def retrieve(self, question: str) -> list[str]:
        """Return the retriever's chunks for `question` ([] when the node does its own retrieval)."""
        return self.retriever.retrieve(question) if self.retriever else []

    def filter_uncited(self, answer: str, context: Sequence[str], question: str = "") -> CitationFilterResult:
        """Split the answer's sentences into kept and dropped.

        A sentence is kept if it cites a context chunk (that names its subject, with
        `check_subjects`), states that information is missing (numbers repeated from `question`
        allowed), or is a conclusion following cited sentences (it then gets their citations appended).
        """
        texts = chunk_texts(context)
        answer = resolve_citations(answer, texts)  # "[doc06]" for "[d06]" still counts
        kept: list[str] = []
        dropped: list[str] = []
        wrong_subject: list[str] = []
        earlier_citations: dict[str, None] = {}  # insertion-ordered set
        for sentence in split_sentences(answer):
            if sentence_ids := [chunk_id for chunk_id in cited(sentence) if chunk_id in texts]:
                if self.check_subjects and look_alike_names(sentence, (texts[i] for i in sentence_ids)):
                    wrong_subject.append(sentence)
                    continue
                kept.append(sentence)
                earlier_citations.update(dict.fromkeys(sentence_ids))
            elif is_missing_info(sentence, question):
                kept.append(sentence)
            elif is_conclusion(sentence) and earlier_citations:
                kept.append(append_citations(sentence, earlier_citations))
            else:
                dropped.append(sentence)
        if wrong_subject:  # the carried-over fact must not survive in a reworded sentence
            kept, restated = restatements_removed(kept, wrong_subject)
            wrong_subject += restated
        if kept:
            kept[0] = strip_leading_connective(kept[0])
        return CitationFilterResult(kept, dropped, tuple(wrong_subject))

    def run(self, state: State, node: Node) -> State:
        """Retrieve context, run the node with the citation rule, then drop uncited sentences.

        Citations are checked against the context the answer was written from: the node's returned
        `context` if it reports one (a node with its own RAG), else the context it was given.
        """
        question = require_question(state)
        search_flags: list[str] = []
        if isinstance(self.retriever, QueryExpansionRetriever):
            retrieved, queries = self.retriever.search(question)
            search_flags = [f"grounding: also searched for {query!r}" for query in queries[1:]]
        else:
            retrieved = self.retrieve(question)
        own = state.get("context", [])
        # A chunk whose whole document the pipeline already supplied would only repeat it.
        retrieved = [chunk for chunk in retrieved if not covered_by(chunk, own)]
        given = retrieved + [entry for entry in own if entry not in retrieved]
        rules = [*state.get("instructions", []), *([CITATION_RULE] if self.require_citations else [])]
        output = node({**state, "context": given, "instructions": rules})
        output = {**output, "flags": [*output.get("flags", []), *search_flags]}
        context = as_context_entries(output.get("context") or given)
        if not self.require_citations or output.get("answer", "") == ABSTAIN_MSG:
            return {**output, "context": context}
        if not chunk_ids(context):
            note = "grounding: no [id] context to check citations against; answer not filtered"
            return {**output, "context": context, "flags": [*output.get("flags", []), note]}
        kept, dropped, wrong_subject = self.filter_uncited(output.get("answer", ""), context, question)
        flags = [
            *output.get("flags", []),
            *(f"grounding: dropped uncited '{sentence}'" for sentence in dropped),
            *(f"grounding: dropped '{sentence}' (cited chunks never name its subject)" for sentence in wrong_subject),
        ]
        if not kept:  # nothing grounded is left: say so instead of returning an empty answer
            return {**output, "context": context, "answer": ABSTAIN_MSG, "sources": [], "flags": flags}
        return {
            **output,
            "context": context,
            "answer": " ".join(kept),
            "sources": sorted({chunk_id for sentence in kept for chunk_id in cited(sentence)} & chunk_ids(context)),
            "flags": flags,
        }
