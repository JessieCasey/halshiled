"""Naive keyword retriever used as the pipeline's first node."""

from __future__ import annotations

import re
from collections.abc import Mapping
from functools import cache

from halshield import Retriever


@cache
def _long_words(text: str) -> frozenset[str]:
    return frozenset(word for word in re.findall(r"[a-z0-9]+", text.lower()) if len(word) > 3)


class KeywordRetriever(Retriever):
    """Ranks whole docs by plain word overlap. Deliberately weak: RAGGrounding adds better retrieval."""

    def __init__(self, docs: Mapping[str, str], k: int = 2) -> None:
        if not docs:
            raise ValueError("KeywordRetriever needs at least one document")
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self.docs, self.k = docs, k

    def retrieve(self, question: str) -> list[str]:
        """Return the k docs sharing the most words with `question`, as "[id] text"."""
        question_words = _long_words(question)
        ranked = sorted(self.docs, key=lambda doc_id: -len(question_words & _long_words(self.docs[doc_id])))
        return [f"[{doc_id}] {self.docs[doc_id]}" for doc_id in ranked[: self.k]]
