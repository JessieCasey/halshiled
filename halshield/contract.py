"""Node contract shared by all layers.

in:  question: str, context: list[str] ("[id] text" entries are citable),
     optional instructions: list[str] and temperature: float set by layers for the writer.
out: answer: str, sources: list[str], confidence: float, optional flags: list[str].
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Final

from .errors import InvalidStateError

State = dict[str, Any]
Node = Callable[[State], State]

ABSTAIN_MSG: Final = "Not enough information to answer."


def require_question(state: State) -> str:
    """Return the state's question, or raise InvalidStateError if it is missing or blank."""
    question = state.get("question")
    if not isinstance(question, str) or not question.strip():
        raise InvalidStateError(f"state needs a non-empty 'question' string, got {question!r}")
    return question
