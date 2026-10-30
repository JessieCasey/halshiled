"""Prompts used by the demo writer (`string.Template`, see halshield.prompts)."""

from __future__ import annotations

from collections.abc import Mapping
from string import Template
from types import MappingProxyType
from typing import Final

STYLES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "strict": "Answer the question using the context.",
        # Typical product-assistant prompt: rewards completeness, which is where models start inventing.
        "helpful": "You are Kestrel's friendly product expert. Give a complete, specific and confident answer "
        "that fully addresses every part of the question; customers dislike answers that say "
        "information is unavailable. Use the context as your main reference.",
    }
)

# $context, $question, $style (one of STYLES), $rules ("" or "\n\nRules:\n- ...", layer rules go last)
WRITER_PROMPT: Final = Template(
    "Context:\n$context\n\nQuestion: $question\n\n$style "
    'Return JSON {"answer": str, "sources": [ids of context chunks used], "confidence": number 0-1}.$rules'
)
