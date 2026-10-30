"""Query rewriting: rank fusion, the rewriter, and retrieval through rewritten queries."""

from __future__ import annotations

import pytest

from halshield import QueryExpansionRetriever, QueryRewriter, RAGGrounding, Retriever
from halshield.retrieval import reciprocal_rank_fusion

from fakes import FakeLLM, RecordingNode


def test_rank_fusion_rewards_agreement():
    assert reciprocal_rank_fusion([["a", "b"], ["b", "c"]], limit=3) == ["b", "a", "c"]
    assert reciprocal_rank_fusion([["a", "a", "b"]], limit=5) == ["a", "b"]  # duplicates count once
    with pytest.raises(ValueError, match="limit"):
        reciprocal_rank_fusion([["a"]], limit=0)


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ({"queries": ["Kestrel CTO", "kestrel cto", "Who leads tech?", "", 7, "CTO 24"]}, ["Kestrel CTO", "CTO 24"]),
        ({"queries": None}, []),
        ({}, []),
    ],
    ids=["dedupes-and-caps", "null", "garbage"],
)  # fmt: skip
def test_rewriter_cleans_model_output(reply, expected):
    rewriter = QueryRewriter(n=2, llm=FakeLLM(lambda prompt: reply))
    assert rewriter.rewrite("Who leads tech?") == expected


class Recording(Retriever):
    def __init__(self) -> None:
        self.queries: list[str] = []

    def retrieve(self, question: str) -> list[str]:
        self.queries.append(question)
        return [f"[{question[:3]}] {question}", "[shared] common chunk"]


def test_expansion_searches_every_query_and_fuses():
    base = Recording()
    retriever = QueryExpansionRetriever(base, QueryRewriter(n=2, llm=FakeLLM(lambda p: {"queries": ["xyz", "uvw"]})))
    chunks = retriever.retrieve("abc?")
    assert base.queries == ["abc?", "xyz", "uvw"]
    assert chunks[0] == "[shared] common chunk"  # found by all three searches
    assert len(chunks) == 4  # one search's 2 chunks + 1 per rewrite
    assert len(QueryExpansionRetriever(base, retriever.rewriter, limit=1).retrieve("abc?")) == 1
    with pytest.raises(TypeError, match="Retriever"):
        QueryExpansionRetriever(object(), QueryRewriter(llm=FakeLLM()))  # type: ignore[arg-type]


def test_grounding_finds_a_chunk_only_the_rewrite_matches():
    docs = {
        "lead": "Jonas Leht was CTO until 2020; Anneli Saar succeeded him as CTO.",
        "noise": "Kestrel leads the market in warehouse drones and technology partnerships.",
    }

    def reply(prompt: str) -> dict:
        return {"queries": ["Kestrel CTO succeeded"]} if "search queries" in prompt else {}

    llm = FakeLLM(reply)
    plain = RAGGrounding(docs, k=1, llm=llm)
    rewriting = RAGGrounding(docs, k=1, llm=llm, rewrite_queries=1)
    question = {"question": "Who leads technology at Kestrel?"}
    assert plain.retrieve(question["question"])[0].startswith("[noise]")
    node = RecordingNode(answer="Anneli Saar [lead].", sources=["lead"])
    output = rewriting.run(question, node)
    assert any(entry.startswith("[lead]") for entry in output["context"])
    assert output["sources"] == ["lead"]
    assert llm.usage.calls["rewrite"] == 1
    assert "grounding: also searched for 'Kestrel CTO succeeded'" in output["flags"]
    with pytest.raises(ValueError, match="rewrite_queries"):
        RAGGrounding(docs, llm=llm, rewrite_queries=-1)


def test_rewriter_adds_a_hypothetical_answer_sentence_last():
    reply = {"queries": ["Kestrel CTO", "chief technology officer"], "hypothetical": "Kestrel's CTO is Jane Doe."}
    rewriter = QueryRewriter(n=1, llm=FakeLLM(lambda prompt: reply))
    assert rewriter.rewrite("Who leads tech?") == ["Kestrel CTO", "Kestrel's CTO is Jane Doe."]
    plain = QueryRewriter(n=3, llm=FakeLLM(lambda prompt: reply), hypothetical=False)
    assert plain.rewrite("Who leads tech?") == ["Kestrel CTO", "chief technology officer"]
