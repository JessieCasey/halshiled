"""Abstain layer."""

from __future__ import annotations

from ..contract import ABSTAIN_MSG, Node, State
from ..interfaces.layer import Layer
from ..utils import as_number


class Abstain(Layer):
    """Replace the answer with ABSTAIN_MSG if confidence < threshold or no sources/answer remain."""

    name = "abstain"

    def __init__(self, threshold: float = 0.7) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be within [0, 1], got {threshold}")
        self.threshold = threshold

    def run(self, state: State, node: Node) -> State:
        """Run the node and abstain if its answer is unsupported or low-confidence."""
        output = node(state)
        confidence = as_number(output.get("confidence"), 0.0)
        has_answer = bool(str(output.get("answer", "")).strip())
        if confidence >= self.threshold and output.get("sources") and has_answer:
            return output
        reason = f"abstain: confidence={confidence:.2f}, sources={output.get('sources')}"
        return {**output, "answer": ABSTAIN_MSG, "sources": [], "flags": [*output.get("flags", []), reason]}
