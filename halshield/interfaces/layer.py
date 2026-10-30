"""Layer interface: a wrapper around a LangGraph node."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..contract import Node, State


class Layer(ABC):
    """Base class for all layers. `name` labels this layer's LLM usage in RunContext."""

    name: str = "layer"

    @abstractmethod
    def run(self, state: State, node: Node) -> State:
        """Call `node` (the rest of the stack) and return its output.

        A layer may change the state going in, the output coming back, or both.
        """

    def wrap(self, node: Node) -> Node:
        """Return `node` wrapped by this layer."""

        def wrapped(state: State) -> State:
            return self.run(state, node)

        wrapped.__qualname__ = f"{type(self).__name__}.wrapped"
        return wrapped
