"""LLMClient interface: what layers need from a model backend."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import numpy as np

from ..prompts import JSON_SYSTEM_PROMPT, SYSTEM_PROMPT
from ..run_context import RunContext


class LLMClient(ABC):
    """Chat + embeddings backend. Implementations must record usage in `self.usage` under `label`."""

    def __init__(self, model: str, usage: RunContext | None = None) -> None:
        if not model:
            raise ValueError("model name must be a non-empty string")
        self.model = model
        self.usage = usage if usage is not None else RunContext()

    @abstractmethod
    def chat(
        self,
        prompt: str,
        label: str,
        system: str = SYSTEM_PROMPT,
        json_mode: bool = False,
        temperature: float = 0.0,
    ) -> str:
        """Return a single-turn completion."""

    @abstractmethod
    def embed(self, texts: Sequence[str], label: str) -> np.ndarray:
        """Return L2-normalised embeddings, one row per text."""

    @abstractmethod
    def with_model(self, model: str) -> LLMClient:
        """Return a client for another chat model on the same backend, sharing `usage`."""

    def json(
        self,
        prompt: str,
        label: str,
        system: str = JSON_SYSTEM_PROMPT,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        """Return the completion parsed as a JSON object.

        Malformed or non-object JSON yields {}: layers treat a garbled reply as "no change"
        rather than failing the whole graph run.
        """
        try:
            parsed = json.loads(self.chat(prompt, label, system, json_mode=True, temperature=temperature))
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
