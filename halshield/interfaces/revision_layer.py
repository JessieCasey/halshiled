"""RevisionLayer interface: a layer that post-edits the node's answer."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

from ..contract import Node, State
from ..utils import as_number, cited, clamp_unit, context_text, is_abstain_or_empty, restore_citations
from .layer import Layer


class RevisionLayer(Layer):
    """Runs the node, then lets `revise` rewrite the answer.

    Skips empty/abstained answers, keeps only sources still cited, and never raises confidence.
    With `cited_context_only`, `revise` sees only the chunks the answer cites (every chunk if it cites
    none): a claim must hold in the chunk it cites, and the prompt is several times shorter. Citations
    the rewrite drops from sentences it kept verbatim are put back.
    """

    cited_context_only: bool = True

    @abstractmethod
    def revise(self, question: str, answer: str, context: str) -> dict[str, Any]:
        """Return {"answer": str, "confidence": float} with unsupported claims removed."""

    def run(self, state: State, node: Node) -> State:
        """Run the node, revise its answer and reconcile sources and confidence."""
        output = node(state)
        original_answer = output.get("answer", "")
        if is_abstain_or_empty(original_answer):
            return output
        only_ids = cited(original_answer) if self.cited_context_only else ()
        revision = self.revise(state.get("question", ""), original_answer, context_text(output, state, only_ids))
        revised_answer = restore_citations(str(revision.get("answer", original_answer)).strip(), original_answer)
        sources = list(output.get("sources", []))
        if set(sources) & set(cited(original_answer)):  # sources are inline citations: keep survivors
            still_cited = set(cited(revised_answer))
            sources = [source for source in sources if source in still_cited]
        confidence = min(
            as_number(output.get("confidence"), 1.0), clamp_unit(as_number(revision.get("confidence"), 1.0))
        )
        revised_flag = [f"{self.name}: answer revised"] if revised_answer != original_answer else []
        return {
            **output,
            "answer": revised_answer,
            "sources": sources if revised_answer else [],
            "confidence": confidence,
            "flags": [*output.get("flags", []), *revised_flag],
        }
