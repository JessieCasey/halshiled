"""Interfaces, Shield composition, state handling and constructor validation."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from pydantic import BaseModel

from halshield import (
    Abstain,
    CoVe,
    HybridRetriever,
    InvalidStateError,
    Layer,
    LLMClient,
    RAGGrounding,
    Retriever,
    RevisionLayer,
    RunContext,
    SelfConsistency,
    Shield,
    State,
)

from fakes import FakeLLM, RecordingNode


class Tag(Layer):
    def __init__(self, tag: str) -> None:
        self.tag = tag

    def run(self, state: State, node) -> State:
        output = node(state)
        return {**output, "trace": [*output.get("trace", []), self.tag]}


@pytest.mark.parametrize("interface", [Layer, LLMClient, RevisionLayer, Retriever])
def test_interfaces_cannot_be_instantiated(interface):
    with pytest.raises(TypeError):
        interface()


def test_shield_applies_layers_in_list_order():
    output = Shield([Tag("first"), Tag("second")]).wrap(RecordingNode(answer="a"))({"question": "q"})
    assert output["trace"] == ["first", "second"]  # layers[0] sees the raw output first


class PydanticState(BaseModel):
    question: str = "q"


@dataclass
class DataclassState:
    question: str = "q"


class PlainState:
    def __init__(self) -> None:
        self.question = "q"


@pytest.mark.parametrize("state", [{"question": "q"}, PydanticState(), DataclassState(), PlainState()], ids=type)
def test_shield_normalises_langgraph_state_objects(state):
    node = RecordingNode(answer="a")
    Shield([]).wrap(node)(state)
    assert node.calls == [{"question": "q"}]


@pytest.mark.parametrize(
    ("factory", "error"),
    [
        (lambda: Shield(["not a layer"]), TypeError),
        (lambda: Shield([]).wrap("not callable"), TypeError),
        (lambda: Abstain(threshold=1.5), ValueError),
        (lambda: SelfConsistency(n=1, llm=FakeLLM()), ValueError),
        (lambda: SelfConsistency(temperature=0, llm=FakeLLM()), ValueError),
        (lambda: CoVe(max_questions=0, llm=FakeLLM()), ValueError),
        (lambda: RAGGrounding(k=3), ValueError),
        (lambda: HybridRetriever({"a": "text"}, k=0, llm=FakeLLM()), ValueError),
        (lambda: HybridRetriever({"a": "  "}, llm=FakeLLM()), ValueError),
        (lambda: RunContext().record("x", -1), ValueError),
        (lambda: FakeLLM(model=""), ValueError),
    ],
    ids=[
        "shield-layer-type", "shield-node", "abstain-threshold", "sc-n", "sc-temperature", "cove-questions",
        "grounding-k-without-search", "retriever-k", "retriever-empty-doc", "negative-tokens", "empty-model",
    ],
)  # fmt: skip
def test_invalid_configuration_fails_fast(factory, error):
    with pytest.raises(error):
        factory()


@pytest.mark.parametrize("state", [{}, {"question": "  "}, {"question": 3}])
def test_layers_reject_state_without_question(grounding, state):
    with pytest.raises(InvalidStateError):
        grounding.run(state, RecordingNode(answer="a"))


def test_usage_accounting_and_json_fallback():
    llm = FakeLLM(lambda prompt: "not json")
    assert llm.json("x", "a") == {}  # garbled reply degrades to "no change"
    llm.chat("12345678", "b")
    assert llm.usage.calls == {"a": 1, "b": 1}
    assert llm.usage.totals() == (2, 2)
    assert llm.with_model("other").usage is llm.usage
    llm.usage.reset()
    assert llm.usage.totals() == (0, 0)
