"""Two-node LangGraph pipeline: retriever -> writer."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from halshield import Node

from .qa_state import QAState


class QAPipeline:
    """Compiles retriever -> writer into a LangGraph app. Pass a shielded writer to protect it."""

    def __init__(self, retriever: Node, writer: Node) -> None:
        graph = StateGraph(QAState)
        graph.add_node("retriever", retriever)
        graph.add_node("writer", writer)
        graph.add_edge(START, "retriever")
        graph.add_edge("retriever", "writer")
        graph.add_edge("writer", END)
        self.app = graph.compile()

    def invoke(self, question: str) -> dict:
        """Run the graph for one question and return the final state."""
        return self.app.invoke({"question": question})
