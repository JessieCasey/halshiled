"""LLM query rewriting for retrieval."""

from __future__ import annotations

from ..interfaces.llm_client import LLMClient
from ..llm.default import get_llm
from ..prompts import QUERY_REWRITE


class QueryRewriter:
    """Turns a user question into search queries in the documents' own vocabulary.

    Returns up to `n` rewritten queries (official terms and abbreviations, one per fact for multi-step
    questions) and, with `hypothetical`, one made-up document-style answer sentence (HyDE), which often
    matches the right chunk better than any question does. One LLM call either way.
    """

    def __init__(
        self, n: int = 3, llm: LLMClient | None = None, label: str = "rewrite", hypothetical: bool = True
    ) -> None:
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        self.n, self.label, self.hypothetical, self.llm = n, label, hypothetical, llm or get_llm()

    def rewrite(self, question: str) -> list[str]:
        """Return the distinct queries (excluding the question), hypothetical sentence last; [] on garbage."""
        response = self.llm.json(QUERY_REWRITE.substitute(question=question, n=self.n), self.label)
        raw_queries = response.get("queries")
        seen = {question.strip().casefold()}

        def fresh(raw: object) -> str:
            query = raw.strip() if isinstance(raw, str) else ""
            if not query or query.casefold() in seen:
                return ""
            seen.add(query.casefold())
            return query

        queries = [query for raw in (raw_queries if isinstance(raw_queries, list) else []) if (query := fresh(raw))]
        queries = queries[: self.n]
        if self.hypothetical and (sentence := fresh(response.get("hypothetical"))):
            queries.append(sentence)
        return queries
