"""One annotated bench case with a deterministic rubric."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .bench_format_error import BenchFormatError
from .forbidden_claim import ForbiddenClaim

_CITATION = re.compile(r"\[[^\[\]]+\]")


def _mentions(text: str, phrase: str) -> bool:
    """Return True if `phrase` occurs in `text` as a whole word/number, case-insensitively."""
    return re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text, re.IGNORECASE) is not None


def _string_tuple(raw: Mapping[str, Any], key: str, case_id: str) -> tuple[str, ...]:
    value = raw.get(key, [])
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise BenchFormatError(f"case {case_id}: {key!r} must be a list of non-empty strings")
    return tuple(value)


@dataclass(frozen=True, slots=True)
class BenchCase:
    """A question, its gold answer, and how to score it.

    `must_include` is a rubric: every group must be matched by at least one of its alternatives
    (whole-word, case-insensitive, citations ignored). `forbidden` are deterministic checks for the
    wrong assertions the case baits; `traps` describe those wrong answers in words for the judge.
    """

    id: str
    category: str
    question: str
    answerable: bool
    gold: str
    must_include: tuple[tuple[str, ...], ...]
    traps: tuple[str, ...]
    forbidden: tuple[ForbiddenClaim, ...]
    evidence: tuple[str, ...]  # docs the gold answer needs (not the trap docs)
    why_hard: str

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> BenchCase:
        """Build a case from its JSON object, validating every field."""
        case_id = raw.get("id")
        if not isinstance(case_id, str) or not case_id:
            raise BenchFormatError(f"case is missing a string 'id': {dict(raw)!r:.80}")
        for key in ("category", "question", "gold", "why_hard"):
            if not isinstance(raw.get(key), str) or not raw[key].strip():
                raise BenchFormatError(f"case {case_id}: {key!r} must be a non-empty string")
        if not isinstance(raw.get("answerable"), bool):
            raise BenchFormatError(f"case {case_id}: 'answerable' must be true or false")
        groups = raw.get("must_include", [])
        if not isinstance(groups, list) or not all(
            isinstance(group, list) and group and all(isinstance(alt, str) and alt for alt in group) for group in groups
        ):
            raise BenchFormatError(f"case {case_id}: 'must_include' must be a list of non-empty string lists")
        if groups and not raw["answerable"]:
            raise BenchFormatError(f"case {case_id}: unanswerable cases cannot have a 'must_include' rubric")
        forbidden = raw.get("forbidden", [])
        if not isinstance(forbidden, list):
            raise BenchFormatError(f"case {case_id}: 'forbidden' must be a list")
        return cls(
            id=case_id,
            category=raw["category"],
            question=raw["question"],
            answerable=raw["answerable"],
            gold=raw["gold"],
            must_include=tuple(tuple(group) for group in groups),
            traps=_string_tuple(raw, "traps", case_id),
            forbidden=tuple(ForbiddenClaim.from_dict(item, f"case {case_id}") for item in forbidden),
            evidence=_string_tuple(raw, "evidence", case_id),
            why_hard=raw["why_hard"],
        )

    def missing_facts(self, answer: str) -> tuple[str, ...]:
        """Return the rubric groups the answer does not satisfy, as "alt1|alt2" labels."""
        text = _CITATION.sub("", answer)
        return tuple("|".join(group) for group in self.must_include if not any(_mentions(text, alt) for alt in group))

    def forbidden_hits(self, answer: str) -> tuple[str, ...]:
        """Return the labels of the forbidden assertions the answer makes."""
        return tuple(check.label for check in self.forbidden if check.violated_by(answer))
