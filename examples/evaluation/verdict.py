"""Judge verdict for one answer."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Verdict:
    """How one answer scored.

    hallucinated: it makes at least one claim no document supports, or a forbidden assertion.
    trap_triggered: the judge saw it fall for a known trap, or it made a forbidden assertion.
    correct: no forbidden assertion, and answerable cases satisfy the rubric without abstaining while
    unanswerable cases decline cleanly.
    """

    hallucinated: bool
    abstained: bool
    correct: bool
    trap_triggered: bool
    unsupported: tuple[str, ...] = ()
    missing_facts: tuple[str, ...] = ()
    forbidden_hits: tuple[str, ...] = ()

    @property
    def clean_and_correct(self) -> bool:
        """Return True if the answer is correct and contains no hallucination."""
        return self.correct and not self.hallucinated
