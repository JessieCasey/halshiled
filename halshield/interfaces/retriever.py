"""Retriever interface: finds context chunks for a question."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..contract import State


class Retriever(ABC):
    """Returns citable "[id] text" chunks. Instances are also LangGraph nodes that fill `context`."""

    @abstractmethod
    def retrieve(self, question: str) -> list[str]:
        """Return the top chunks for `question` as "[id] text" entries."""

    def __call__(self, state: State) -> dict:
        """Run as a LangGraph node: `{"context": retrieve(question)}`."""
        return {"context": self.retrieve(state["question"])}
