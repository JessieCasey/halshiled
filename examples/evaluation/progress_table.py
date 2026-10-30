"""A fixed-width table printed one row at a time while the evaluation runs."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from typing import TextIO


class ProgressTable:
    """Prints a header once, then aligned rows; the last column is truncated to fit."""

    def __init__(self, columns: Sequence[tuple[str, int]], stream: TextIO | None = None) -> None:
        if not columns:
            raise ValueError("a table needs at least one column")
        if any(width < 1 for _, width in columns):
            raise ValueError("column widths must be >= 1")
        self.columns = tuple(columns)
        self.stream = stream or sys.stdout
        self._header_printed = False

    @staticmethod
    def _cell(value: object, width: int) -> str:
        text = " ".join(str(value).split())  # one line: collapse newlines and runs of spaces
        return text if len(text) <= width else text[: width - 1] + "…"

    def _line(self, values: Sequence[object]) -> str:
        cells = [self._cell(value, width).ljust(width) for value, (_, width) in zip(values, self.columns, strict=True)]
        return "  ".join(cells).rstrip()

    def row(self, *values: object) -> None:
        """Print one row (and the header before the first one)."""
        if len(values) != len(self.columns):
            raise ValueError(f"expected {len(self.columns)} values, got {len(values)}")
        if not self._header_printed:
            print(self._line([name for name, _ in self.columns]), file=self.stream)
            print("  ".join("-" * width for _, width in self.columns), file=self.stream)
            self._header_printed = True
        print(self._line(values), file=self.stream, flush=True)
