"""Shield: composes layers around a LangGraph node."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Mapping
from functools import reduce
from typing import Any

from .contract import Node, State
from .interfaces.layer import Layer
from .presets import Protection


def _as_state_dict(state: Any) -> State:
    """Convert the state objects LangGraph may pass (dict, pydantic model, dataclass) to a dict."""
    if isinstance(state, Mapping):
        return dict(state)
    if hasattr(state, "model_dump"):
        return state.model_dump()
    if dataclasses.is_dataclass(state) and not isinstance(state, type):
        return dataclasses.asdict(state)
    if hasattr(state, "__dict__"):
        return dict(vars(state))
    raise TypeError(f"unsupported state type {type(state).__name__}; expected a mapping, model or dataclass")


class Shield:
    """Wraps a node with layers listed in processing order.

    layers[0] sits closest to the node (sees its raw output first); layers[-1] has the final say.
    Pass a Protection level instead of layers to get that level's stack; `options` go to
    `Protection.layers` (sources, retriever, k, llm, threshold).

        graph.add_node("writer", Shield(layers=[RAGGrounding(docs), CoVe(), Abstain(0.7)]).wrap(writer))
        graph.add_node("writer", Shield(Protection.MEDIUM, sources=docs).wrap(writer))
    """

    def __init__(self, layers: Iterable[Layer] | Protection | str, **options: Any) -> None:
        if isinstance(layers, str):  # a Protection member or its name, e.g. "medium"
            try:
                level = Protection(layers)
            except ValueError:
                choices = [level.value for level in Protection]
                raise ValueError(f"unknown protection level {layers!r}; choose from {choices}") from None
            layers = level.layers(**options)
        elif options:
            raise TypeError(f"options {sorted(options)} apply only to a Protection level, not to a list of layers")
        self.layers: tuple[Layer, ...] = tuple(layers)
        if invalid := [type(layer).__name__ for layer in self.layers if not isinstance(layer, Layer)]:
            raise TypeError(f"Shield layers must be Layer instances, got {invalid}")

    def wrap(self, node: Node) -> Node:
        """Return a plain `state -> dict` node usable with `graph.add_node`."""
        if not callable(node):
            raise TypeError(f"node must be callable, got {type(node).__name__}")
        inner = reduce(lambda wrapped, layer: layer.wrap(wrapped), self.layers, node)

        def shielded(state: Any) -> State:
            return inner(_as_state_dict(state))

        return shielded
