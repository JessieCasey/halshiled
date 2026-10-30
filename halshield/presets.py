"""Ready-made layer stacks: the minimal, medium and maximum protection levels."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import Enum
from typing import Any

from .interfaces.layer import Layer
from .interfaces.llm_client import LLMClient
from .layers import Abstain, CoVe, Critic, RAGGrounding, SelfConsistency
from .llm.default import get_llm


class Protection(str, Enum):
    """A protection level; `Shield(Protection.MEDIUM, sources=docs)` stacks its layers for you.

    - MINIMAL: retrieval with citations, the citation and look-alike filter. One writer call.
    - MEDIUM: query rewriting, and 3 sampled answers with the most consistent one kept
      (2 when the first two agree). The best accuracy for its cost.
    - MAXIMUM: MEDIUM plus Chain-of-Verification, a critic and abstention on low confidence.
      The fewest hallucinations, at the highest cost.
    """

    MINIMAL = "minimal"
    MEDIUM = "medium"
    MAXIMUM = "maximum"

    def layers(
        self,
        sources: Mapping[str, str] | Sequence[str] | None = None,
        *,
        retriever: Any = None,
        k: int | None = 4,
        llm: LLMClient | None = None,
        threshold: float = 0.7,
    ) -> list[Layer]:
        """Return this level's layers in Shield order (closest to the node first).

        SelfConsistency sits inside RAGGrounding, so retrieval and query rewriting run once and every
        sample sees the same context. With neither `sources` nor `retriever`, the node's own RAG is
        checked (no retrieval, no rewriting) and `k` is ignored. `threshold` is the MAXIMUM level's
        Abstain threshold. Lowering it to 0.5 answered more questions on the bench but let through
        about twice as many hallucinations, against what this level is for.
        """
        llm = llm or get_llm()
        searches = sources is not None or retriever is not None
        search = {"sources": sources, "retriever": retriever, "k": k} if searches else {}
        if self is Protection.MINIMAL:
            return [RAGGrounding(**search, llm=llm)]
        grounding = RAGGrounding(**search, llm=llm, rewrite_queries=3 if searches else 0)
        if self is Protection.MEDIUM:
            return [SelfConsistency(llm=llm), grounding]
        return [SelfConsistency(llm=llm), grounding, CoVe(llm=llm), Critic(llm=llm), Abstain(threshold)]
