"""A deterministic check for one wrong assertion a case baits."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from halshield.utils import split_sentences

from .bench_format_error import BenchFormatError

_CITATION = re.compile(r"\[[^\[\]]+\]")


def _compile(raw: Any, field: str, where: str) -> re.Pattern[str]:
    if not isinstance(raw, str) or not raw:
        raise BenchFormatError(f"{where}: {field!r} must be a non-empty regex string")
    try:
        return re.compile(raw, re.IGNORECASE)
    except re.error as exc:
        raise BenchFormatError(f"{where}: invalid {field} regex {raw!r}: {exc}") from exc


@dataclass(frozen=True, slots=True)
class ForbiddenClaim:
    """A sentence matching `pattern` is a wrong assertion, unless it also matches `unless`.

    Checked per sentence (citations ignored), so a correction such as "the 2019 press release
    promised 45 minutes" can be exempted by `unless` while "it flies 45 minutes" is caught.
    """

    label: str
    pattern: re.Pattern[str]
    unless: re.Pattern[str] | None = None

    @classmethod
    def from_dict(cls, raw: Any, where: str) -> ForbiddenClaim:
        """Build a check from its JSON object {"label", "pattern", "unless"?}."""
        if not isinstance(raw, Mapping):
            raise BenchFormatError(f"{where}: each 'forbidden' entry must be an object")
        label = raw.get("label")
        if not isinstance(label, str) or not label.strip():
            raise BenchFormatError(f"{where}: 'forbidden' entry needs a non-empty 'label'")
        unless = raw.get("unless")
        return cls(
            label=label,
            pattern=_compile(raw.get("pattern"), "pattern", where),
            unless=None if unless is None else _compile(unless, "unless", where),
        )

    def violated_by(self, answer: str) -> bool:
        """Return True if any sentence of `answer` makes the forbidden assertion."""
        return any(
            self.pattern.search(sentence) and not (self.unless and self.unless.search(sentence))
            for sentence in split_sentences(_CITATION.sub("", answer))
        )
