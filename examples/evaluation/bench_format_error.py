"""Error raised for a malformed bench file."""

from __future__ import annotations


class BenchFormatError(ValueError):
    """The bench JSON is unreadable or violates the expected schema."""
