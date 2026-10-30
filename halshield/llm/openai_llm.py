"""OpenAI implementation of LLMClient."""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

from ..errors import LLMRequestError
from ..interfaces.llm_client import LLMClient
from ..prompts import SYSTEM_PROMPT
from ..run_context import RunContext


class OpenAILLM(LLMClient):
    """OpenAI chat + embeddings. Model names come from args or env (HALSHIELD_MODEL, HALSHIELD_EMBED_MODEL)."""

    def __init__(
        self,
        model: str | None = None,
        embed_model: str | None = None,
        usage: RunContext | None = None,
        client: Any = None,
    ) -> None:
        if client is None:
            load_dotenv()
            try:
                client = OpenAI()
            except OpenAIError as exc:
                raise LLMRequestError(f"cannot create OpenAI client (is OPENAI_API_KEY set?): {exc}") from exc
        super().__init__(model or os.getenv("HALSHIELD_MODEL", "gpt-4o-mini"), usage)
        self.client = client
        self.embed_model = embed_model or os.getenv("HALSHIELD_EMBED_MODEL", "text-embedding-3-small")

    def chat(
        self,
        prompt: str,
        label: str,
        system: str = SYSTEM_PROMPT,
        json_mode: bool = False,
        temperature: float = 0.0,
    ) -> str:
        """Return a single-turn completion and record its token usage."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=temperature,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                **({"response_format": {"type": "json_object"}} if json_mode else {}),
            )
        except OpenAIError as exc:
            raise LLMRequestError(f"chat request for {label!r} with model {self.model!r} failed: {exc}") from exc
        self.usage.record(label, response.usage.total_tokens)
        return response.choices[0].message.content or ""

    def embed(self, texts: Sequence[str], label: str) -> np.ndarray:
        """Return L2-normalised embeddings; tokens are counted, but not as LLM calls."""
        if not texts:
            raise ValueError("embed() needs at least one text")
        try:
            response = self.client.embeddings.create(model=self.embed_model, input=list(texts))
        except OpenAIError as exc:
            raise LLMRequestError(f"embedding request for {label!r} with {self.embed_model!r} failed: {exc}") from exc
        self.usage.record(label, response.usage.total_tokens, call=False)
        vectors = np.array([item.embedding for item in response.data], dtype=np.float32)
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    def with_model(self, model: str) -> OpenAILLM:
        """Return a client for `model` sharing this client's connection and usage counters."""
        return OpenAILLM(model=model, embed_model=self.embed_model, usage=self.usage, client=self.client)
