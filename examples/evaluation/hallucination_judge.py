"""LLM judge that checks an answer's claims against the knowledge base."""

from __future__ import annotations

from collections.abc import Mapping

from halshield import ABSTAIN_MSG, LLMClient

from .bench_case import BenchCase
from .prompts import JUDGE_PROMPT
from .verdict import Verdict


class HallucinationJudge:
    """Grades answers claim by claim against every document, then applies the case's rubric."""

    def __init__(self, llm: LLMClient, docs: Mapping[str, str]) -> None:
        if not docs:
            raise ValueError("the judge needs at least one document")
        self.llm = llm
        self.documents = "\n".join(f"[{doc_id}] {text}" for doc_id, text in docs.items())

    def grade(self, case: BenchCase, answer: str) -> Verdict:
        """Grade one answer to `case`."""
        if not answer.strip() or answer == ABSTAIN_MSG:
            return Verdict(hallucinated=False, abstained=True, correct=not case.answerable, trap_triggered=False)
        response = self.llm.json(
            JUDGE_PROMPT.substitute(
                documents=self.documents,
                question=case.question,
                gold=case.gold,
                traps="; ".join(case.traps) or "none",
                answer=answer,
            ),
            "judge",
        )
        unsupported = tuple(
            str(claim.get("claim"))
            for claim in response.get("claims") or []
            if isinstance(claim, dict) and not claim.get("supported", False)
        )
        abstained = bool(response.get("abstained"))
        missing, forbidden = case.missing_facts(answer), case.forbidden_hits(answer)
        if case.answerable:
            rubric_passed = not missing if case.must_include else bool(response.get("correct"))
            correct = rubric_passed and not abstained
        else:
            # Declining outright, or answering only what is known and saying the rest is not given.
            correct = (abstained or bool(response.get("correct"))) and not unsupported
        # Forbidden assertions are checked in code, so a lenient judge cannot excuse them.
        return Verdict(
            hallucinated=bool(unsupported or forbidden),
            abstained=abstained,
            correct=correct and not forbidden,
            trap_triggered=bool(response.get("trap_triggered") or forbidden),
            unsupported=unsupported,
            missing_facts=missing,
            forbidden_hits=forbidden,
        )
