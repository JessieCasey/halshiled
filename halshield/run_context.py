"""Per-layer usage accounting."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from threading import Lock
from typing import NamedTuple


class UsageTotals(NamedTuple):
    """Tokens and LLM calls summed over all labels."""

    tokens: int
    calls: int


# Open measure() scopes in the current thread/task, as (owner id, scope) pairs.
_SCOPES: ContextVar[tuple[tuple[int, RunContext], ...]] = ContextVar("halshield_usage_scopes", default=())


@dataclass
class RunContext:
    """Counts LLM calls and tokens per label (writer, grounding, cove, ...). Thread-safe."""

    calls: Counter[str] = field(default_factory=Counter)
    tokens: Counter[str] = field(default_factory=Counter)
    _lock: Lock = field(default_factory=Lock, repr=False, compare=False)

    def record(self, label: str, tokens: int, call: bool = True) -> None:
        """Add `tokens` (and one call, unless `call` is False) to `label`."""
        if tokens < 0:
            raise ValueError(f"token count must be non-negative, got {tokens}")
        self._add(label, tokens, call)
        for owner, scope in _SCOPES.get():
            if owner == id(self):
                scope._add(label, tokens, call)

    def _add(self, label: str, tokens: int, call: bool) -> None:
        with self._lock:
            self.tokens[label] += tokens
            self.calls[label] += int(call)

    @contextmanager
    def measure(self) -> Iterator[RunContext]:
        """Also collect, into a fresh RunContext, what this context records in the current thread/task.

        Lets concurrent work (e.g. one thread per question) report its own usage from a shared client.
        """
        scope = RunContext()
        token = _SCOPES.set((*_SCOPES.get(), (id(self), scope)))
        try:
            yield scope
        finally:
            _SCOPES.reset(token)

    def totals(self) -> UsageTotals:
        """Return tokens and LLM calls summed over all labels."""
        with self._lock:
            return UsageTotals(sum(self.tokens.values()), sum(self.calls.values()))

    def reset(self) -> None:
        """Clear all counters."""
        with self._lock:
            self.calls.clear()
            self.tokens.clear()
