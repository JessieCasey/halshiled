"""Retriever that also searches for rewritten queries."""

from __future__ import annotations

from ..contract import require_question
from ..interfaces.retriever import Retriever
from .query_rewriter import QueryRewriter
from .rank_fusion import reciprocal_rank_fusion


class QueryExpansionRetriever(Retriever):
    """Searches `base` for the question and each rewritten query, then fuses the rankings (RRF).

    Fixes vocabulary mismatch ("head of technology" vs "CTO") and multi-step questions whose facts
    sit in different chunks. Costs one LLM call plus one base search per query.
    """

    def __init__(self, base: Retriever, rewriter: QueryRewriter, limit: int | None = None) -> None:
        if not isinstance(base, Retriever):
            raise TypeError(f"base must be a Retriever, got {type(base).__name__}")
        if limit is not None and limit < 1:
            raise ValueError(f"limit must be >= 1, got {limit}")
        self.base, self.rewriter, self.limit = base, rewriter, limit

    def queries(self, question: str) -> list[str]:
        """Return the question followed by its rewrites."""
        return [question, *self.rewriter.rewrite(question)]

    def retrieve(self, question: str) -> list[str]:
        """Return the fused top chunks (see `search`)."""
        return self.search(question)[0]

    def search(self, question: str) -> tuple[list[str], list[str]]:
        """Return the fused top chunks and the queries searched (the question first).

        Without `limit`: one base search's size plus one chunk per rewrite, so a rewrite that finds
        what the original question missed adds that chunk instead of only reordering the others.
        """
        queries = self.queries(require_question({"question": question}))
        rankings = [self.base.retrieve(query) for query in queries]
        if not any(rankings):
            return [], queries
        limit = self.limit or max(map(len, rankings)) + len(rankings) - 1
        return reciprocal_rank_fusion(rankings, limit), queries
