"""Protection levels: which layers each one stacks, in Shield order, and Shield built from a level."""

from __future__ import annotations

import pytest

from halshield import Abstain, CoVe, Critic, Protection, RAGGrounding, SelfConsistency, Shield
from halshield.retrieval.query_expansion_retriever import QueryExpansionRetriever

from fakes import FakeLLM


@pytest.mark.parametrize(
    ("level", "layer_types"),
    [
        (Protection.MINIMAL, [RAGGrounding]),
        (Protection.MEDIUM, [SelfConsistency, RAGGrounding]),
        (Protection.MAXIMUM, [SelfConsistency, RAGGrounding, CoVe, Critic, Abstain]),
    ],
)
def test_levels_stack_layers_with_sampling_inside_grounding(docs, level, layer_types):
    layers = level.layers(docs, llm=FakeLLM())
    assert [type(layer) for layer in layers] == layer_types
    grounding = next(layer for layer in layers if isinstance(layer, RAGGrounding))
    assert isinstance(grounding.retriever, QueryExpansionRetriever) is (level is not Protection.MINIMAL)


def test_shield_builds_the_layers_of_a_level(docs):
    llm = FakeLLM()
    for level in (Protection.MAXIMUM, "maximum"):
        shield = Shield(level, sources=docs, llm=llm, threshold=0.4)
        assert [type(layer) for layer in shield.layers][-1] is Abstain
        assert shield.layers[-1].threshold == 0.4


def test_levels_without_sources_check_the_nodes_own_rag():
    for level in Protection:
        grounding = next(layer for layer in level.layers(llm=FakeLLM()) if isinstance(layer, RAGGrounding))
        assert grounding.retriever is None


def test_bad_level_or_options_are_rejected(docs):
    with pytest.raises(ValueError, match="unknown protection level"):
        Shield("extreme", llm=FakeLLM())
    with pytest.raises(TypeError, match="only to a Protection level"):
        Shield([RAGGrounding(docs, llm=FakeLLM())], sources=docs)
