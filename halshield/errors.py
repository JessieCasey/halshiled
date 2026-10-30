"""Exception hierarchy: catch `HalshieldError` to handle any failure raised by the library."""

from __future__ import annotations


class HalshieldError(Exception):
    """Base class for all halshield errors."""


class InvalidStateError(HalshieldError, ValueError):
    """A node state is missing a required field or holds a value of the wrong type."""


class LLMRequestError(HalshieldError):
    """A request to the model backend failed (network, auth, rate limit, malformed response)."""
