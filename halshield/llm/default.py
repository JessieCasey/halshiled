"""Process-wide default LLMClient, used by layers that are not given one."""

from __future__ import annotations

from ..interfaces.llm_client import LLMClient

_default: LLMClient | None = None


def get_llm() -> LLMClient:
    """Return the default client, creating an OpenAILLM from env / .env on first use."""
    global _default
    if _default is None:
        from .openai_llm import OpenAILLM

        _default = OpenAILLM()
    return _default


def set_llm(llm: LLMClient) -> None:
    """Make `llm` the default client (e.g. another backend implementing LLMClient)."""
    global _default
    _default = llm
