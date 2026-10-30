"""Hybrid embedding + keyword retriever."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from types import MappingProxyType

import numpy as np

from ..interfaces.llm_client import LLMClient
from ..interfaces.retriever import Retriever
from ..llm.default import get_llm

_STOP_WORDS = frozenset({
    "the", "and", "for", "what", "which", "who", "how", "does", "did", "was", "were", "has", "have",
    "its", "with", "from", "that", "this", "are", "when", "where", "why", "many", "much",
})  # fmt: skip
_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")


def _content_words(text: str) -> frozenset[str]:
    return frozenset(
        word for word in re.findall(r"[a-z0-9]+", text.lower()) if len(word) > 2 and word not in _STOP_WORDS
    )


def _chunk_documents(docs: Mapping[str, str]) -> dict[str, str]:
    """Split docs into paragraphs; a one-paragraph doc keeps its id, others become "<id>#<n>"."""
    chunks: dict[str, str] = {}
    for doc_id, text in docs.items():
        paragraphs = [paragraph.strip() for paragraph in _PARAGRAPH_BREAK.split(text) if paragraph.strip()]
        if len(paragraphs) == 1:
            chunks[doc_id] = paragraphs[0]
        else:
            chunks.update({f"{doc_id}#{index}": paragraph for index, paragraph in enumerate(paragraphs)})
    return chunks


class HybridRetriever(Retriever):
    """Ranks chunks by embedding cosine plus `keyword_weight` x IDF-weighted keyword overlap.

    The keyword term catches chunks that never name the question's subject (e.g. a leadership
    paragraph that never says the company name). Embedding usage is recorded under `label`.
    """

    def __init__(
        self,
        sources: Mapping[str, str] | Sequence[str],
        k: int = 3,
        keyword_weight: float = 0.3,
        llm: LLMClient | None = None,
        label: str = "retrieval",
    ) -> None:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if keyword_weight < 0:
            raise ValueError(f"keyword_weight must be >= 0, got {keyword_weight}")
        docs = sources if isinstance(sources, Mapping) else {f"doc{i}": text for i, text in enumerate(sources, 1)}
        if empty := [doc_id for doc_id, text in docs.items() if not isinstance(text, str) or not text.strip()]:
            raise ValueError(f"sources must be non-empty strings; empty or invalid: {empty}")
        chunks = _chunk_documents(docs)
        if not chunks:
            raise ValueError("sources must contain at least one document")

        self.k, self.keyword_weight, self.label = k, keyword_weight, label
        self.llm = llm or get_llm()
        self.chunks: Mapping[str, str] = MappingProxyType(chunks)
        self.chunk_ids: tuple[str, ...] = tuple(chunks)
        self.chunk_words: tuple[frozenset[str], ...] = tuple(_content_words(text) for text in chunks.values())
        self.index = self.llm.embed(list(chunks.values()), label)

        document_frequency = Counter(word for words in self.chunk_words for word in words)
        chunk_count = len(self.chunk_words)
        self.idf: Mapping[str, float] = MappingProxyType(
            {word: math.log((chunk_count + 1) / (frequency + 0.5)) for word, frequency in document_frequency.items()}
        )
        self.unseen_word_idf = math.log(chunk_count + 1)

    def keyword_overlap(self, question: str) -> np.ndarray:
        """Return the share of the question's IDF weight found in each chunk (0..1)."""
        weights = {word: self.idf.get(word, self.unseen_word_idf) for word in _content_words(question)}
        total = sum(weights.values()) or 1.0
        return np.fromiter(
            (sum(weight for word, weight in weights.items() if word in words) / total for words in self.chunk_words),
            dtype=np.float64,
            count=len(self.chunk_words),
        )

    def retrieve(self, question: str) -> list[str]:
        """Return the top-k chunks for `question` as "[id] text" entries."""
        if not question.strip():
            raise ValueError("question must be a non-empty string")
        query_vector = self.llm.embed([question], self.label)[0]
        scores = self.index @ query_vector + self.keyword_weight * self.keyword_overlap(question)
        top_indices = np.argsort(scores)[::-1][: self.k]
        return [f"[{self.chunk_ids[i]}] {self.chunks[self.chunk_ids[i]]}" for i in top_indices]
