"""LangGraph state schema for the demo pipeline."""

from __future__ import annotations

from typing import TypedDict


class QAState(TypedDict, total=False):
    """Graph state: the node contract plus the optional fields layers use."""

    question: str
    context: list[str]
    instructions: list[str]  # extra prompt rules added by layers
    temperature: float  # sampling temperature set by SelfConsistency
    answer: str
    sources: list[str]
    confidence: float
    flags: list[str]  # what layers dropped/changed, for inspection
