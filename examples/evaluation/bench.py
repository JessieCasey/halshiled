"""The bench: a document corpus plus annotated cases."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Collection, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType

from .bench_case import BenchCase
from .bench_format_error import BenchFormatError

SUPPORTED_VERSION = 3


@dataclass(frozen=True, slots=True)
class Bench:
    """Immutable, validated bench loaded from JSON."""

    docs: Mapping[str, str]
    cases: tuple[BenchCase, ...]

    @classmethod
    def load(cls, path: str | Path) -> Bench:
        """Read and validate a bench file."""
        bench_path = Path(path)
        try:
            raw = json.loads(bench_path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise BenchFormatError(f"cannot read bench file {bench_path}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise BenchFormatError(f"{bench_path} is not valid JSON: {exc}") from exc
        if not isinstance(raw, dict) or raw.get("version") != SUPPORTED_VERSION:
            raise BenchFormatError(f"{bench_path}: expected a version {SUPPORTED_VERSION} bench object")

        docs = raw.get("docs")
        if not isinstance(docs, dict) or not docs or not all(isinstance(text, str) for text in docs.values()):
            raise BenchFormatError(f"{bench_path}: 'docs' must be a non-empty object of strings")
        cases = tuple(BenchCase.from_dict(item) for item in raw.get("cases") or [])
        if not cases:
            raise BenchFormatError(f"{bench_path}: 'cases' must be a non-empty list")
        if duplicates := [case_id for case_id, count in Counter(c.id for c in cases).items() if count > 1]:
            raise BenchFormatError(f"{bench_path}: duplicate case ids {duplicates}")
        if unknown := sorted({doc_id for case in cases for doc_id in case.evidence} - docs.keys()):
            raise BenchFormatError(f"{bench_path}: evidence refers to unknown docs {unknown}")
        return cls(docs=MappingProxyType(dict(docs)), cases=cases)

    @property
    def categories(self) -> tuple[str, ...]:
        """Return the case categories in first-seen order."""
        return tuple(dict.fromkeys(case.category for case in self.cases))

    def select(self, categories: Collection[str] = (), limit: int | None = None) -> Bench:
        """Return a bench restricted to `categories` (all if empty) and at most `limit` cases."""
        if unknown := set(categories) - set(self.categories):
            raise ValueError(f"unknown categories {sorted(unknown)}; available: {list(self.categories)}")
        if limit is not None and limit < 1:
            raise ValueError(f"limit must be >= 1, got {limit}")
        chosen = tuple(case for case in self.cases if not categories or case.category in categories)
        return replace(self, cases=chosen[:limit])
