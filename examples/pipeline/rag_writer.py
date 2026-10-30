"""RAG writer node."""

from __future__ import annotations

from halshield import LLMClient, State
from halshield.contract import require_question
from halshield.utils import as_number, clamp_unit

from .prompts import STYLES, WRITER_PROMPT


class RAGWriter:
    """LangGraph node: answers from `context`, self-reports sources and confidence.

    Honours the optional `instructions` and `temperature` state fields that layers set.
    """

    def __init__(self, llm: LLMClient, style: str = "strict") -> None:
        if style not in STYLES:
            raise ValueError(f"unknown style {style!r}; choose from {sorted(STYLES)}")
        self.llm, self.style = llm, style

    def prompt(self, state: State) -> str:
        """Build the writer prompt; layer rules go last so they take precedence."""
        rules = "".join(f"\n- {rule}" for rule in state.get("instructions", []))
        return WRITER_PROMPT.substitute(
            context="\n".join(state.get("context", [])) or "(none)",
            question=require_question(state),
            style=STYLES[self.style],
            rules=f"\n\nRules:{rules}" if rules else "",
        )

    def __call__(self, state: State) -> State:
        """Run as a LangGraph node."""
        response = self.llm.json(self.prompt(state), "writer", temperature=state.get("temperature", 0.0))
        return {
            "answer": str(response.get("answer") or ""),
            "sources": [str(source) for source in response.get("sources") or []],
            "confidence": clamp_unit(as_number(response.get("confidence"), 0.0)),
        }
