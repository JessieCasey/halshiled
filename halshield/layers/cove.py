"""CoVe (Chain-of-Verification) layer."""

from __future__ import annotations

import re
from typing import Any, NamedTuple

from ..interfaces.llm_client import LLMClient
from ..interfaces.revision_layer import RevisionLayer
from ..llm.default import get_llm
from ..prompts import COVE_FINDING, COVE_PLAN, COVE_REVISE, COVE_VERIFY

# Where an echoed findings block starts, if the model copies it into the rewritten answer.
_FINDINGS_ECHO = re.compile(r"\s*(?:Independent checks\b|^\s*- Claim:)", re.MULTILINE)


def strip_findings_echo(answer: str) -> str:
    """Cut a copied "Independent checks ... - Claim: ..." block off the end of a rewritten answer."""
    match = _FINDINGS_ECHO.search(answer)
    return answer[: match.start()].rstrip() if match else answer


class VerificationCheck(NamedTuple):
    """One claim from the draft and the question that tests it."""

    claim: str
    question: str


class CoVe(RevisionLayer):
    """Chain-of-Verification over the answer's claims.

    Turns each claim into a check question, answers all of them from the context alone (one call,
    without the draft), then deletes only the claims their own check refutes. Three LLM calls in all.
    """

    name = "cove"

    def __init__(self, max_questions: int = 3, llm: LLMClient | None = None) -> None:
        if max_questions < 1:
            raise ValueError(f"max_questions must be >= 1, got {max_questions}")
        self.max_questions, self.llm = max_questions, llm or get_llm()

    def plan(self, answer: str) -> list[VerificationCheck]:
        """Return one check per factual claim actually stated in the answer."""
        response = self.llm.json(COVE_PLAN.substitute(answer=answer, max_questions=self.max_questions), self.name)
        checks = (
            VerificationCheck(str(item.get("claim", "")), str(item["question"]))
            for item in response.get("checks") or []
            if isinstance(item, dict) and item.get("question")
        )
        return list(checks)[: self.max_questions]

    def verify(self, questions: list[str], context: str) -> list[str]:
        """Answer the check questions from the context only, without seeing the draft (one call).

        A missing or non-string answer counts as NOT IN CONTEXT.
        """
        numbered = "\n".join(f"{index}. {question}" for index, question in enumerate(questions, 1))
        response = self.llm.json(COVE_VERIFY.substitute(context=context, questions=numbered), self.name)
        answers = response.get("answers")
        answers = answers if isinstance(answers, list) else []
        return [
            str(answers[index]) if index < len(answers) and isinstance(answers[index], str) else "NOT IN CONTEXT"
            for index in range(len(questions))
        ]

    def revise(self, question: str, answer: str, context: str) -> dict[str, Any]:
        """Plan, verify and rewrite the answer without refuted claims."""
        checks = self.plan(answer)
        if not checks:
            return {"answer": answer}
        verdicts = self.verify([check.question for check in checks], context)
        findings = "\n".join(
            COVE_FINDING.substitute(claim=check.claim, question=check.question, finding=finding)
            for check, finding in zip(checks, verdicts, strict=True)
        )
        revised = self.llm.json(COVE_REVISE.substitute(answer=answer, findings=findings), self.name)
        if isinstance(revised.get("answer"), str):
            revised["answer"] = strip_findings_echo(revised["answer"])
        return revised
