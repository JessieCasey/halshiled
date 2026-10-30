"""Offline test doubles: a scripted LLMClient and a recording node."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from halshield import LLMClient, RunContext, State

Responder = Callable[[str], "str | dict[str, Any]"]


class FakeLLM(LLMClient):
    """Replies via `respond(prompt)`; embeddings are hashed bags of words (similar text -> similar vector)."""

    def __init__(self, respond: Responder = lambda prompt: {}, model: str = "fake", usage: RunContext | None = None):
        super().__init__(model, usage)
        self.respond = respond
        self.prompts: list[tuple[str, str, float]] = []

    def chat(self, prompt: str, label: str, system: str = "", json_mode: bool = False, temperature: float = 0.0) -> str:
        self.prompts.append((label, prompt, temperature))
        self.usage.record(label, len(prompt) // 4)
        reply = self.respond(prompt)
        return json.dumps(reply) if isinstance(reply, dict) else reply

    def embed(self, texts: Sequence[str], label: str) -> np.ndarray:
        self.usage.record(label, sum(len(text) // 4 for text in texts), call=False)
        vectors = np.full((len(texts), 64), 1e-3)
        for row, text in enumerate(texts):
            for word in re.findall(r"[a-z0-9]+", text.lower()):
                vectors[row, int(hashlib.md5(word.encode()).hexdigest(), 16) % 64] += 1
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    def with_model(self, model: str) -> FakeLLM:
        return FakeLLM(self.respond, model=model, usage=self.usage)

    def labels(self) -> list[str]:
        return [label for label, _, _ in self.prompts]


class RecordingNode:
    """A node that returns a fixed output and records every state it receives."""

    def __init__(self, **output: Any) -> None:
        self.output = output
        self.calls: list[State] = []

    def __call__(self, state: State) -> State:
        self.calls.append(state)
        return dict(self.output)
