"""Critic layer."""

from __future__ import annotations

from typing import Any

from ..interfaces.llm_client import LLMClient
from ..interfaces.revision_layer import RevisionLayer
from ..llm.default import get_llm
from ..prompts import CRITIC_REVIEW


class Critic(RevisionLayer):
    """A reviewer model (can differ from the writer's) removes claims the context does not support."""

    name = "critic"

    def __init__(self, model: str | None = None, llm: LLMClient | None = None) -> None:
        base_llm = llm or get_llm()
        self.llm = base_llm.with_model(model) if model else base_llm  # shares usage counters

    def revise(self, question: str, answer: str, context: str) -> dict[str, Any]:
        """Ask the reviewer model to rewrite the answer without unsupported claims."""
        return self.llm.json(CRITIC_REVIEW.substitute(context=context, question=question, answer=answer), self.name)
