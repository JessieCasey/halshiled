"""Shared fixtures."""

from __future__ import annotations

import pytest

from halshield import RAGGrounding

from fakes import FakeLLM


@pytest.fixture
def docs() -> dict[str, str]:
    return {
        "d1": "The K2 drone flies for up to 38 minutes on a single charge.",
        "d2": "The K3 drone adds a lidar sensor for obstacle avoidance.",
        "d3": "Kestrel Robotics was founded in 2014 in Tallinn.",
    }


@pytest.fixture
def grounding(docs: dict[str, str]) -> RAGGrounding:
    return RAGGrounding(docs, llm=FakeLLM())
