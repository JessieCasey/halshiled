"""The README usage, end to end in a LangGraph graph with a pydantic state."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from halshield import Abstain, CoVe, Critic, RAGGrounding, SelfConsistency, Shield, set_llm

from fakes import FakeLLM


class GraphState(BaseModel):
    question: str
    context: list[str] = []
    answer: str = ""
    sources: list[str] = []
    confidence: float = 0.0
    flags: list[str] = []


def scripted(prompt: str) -> str | dict:
    if "Candidate answers" in prompt:
        return {"best": 0, "agreement": 0.9}
    if "List up to" in prompt:
        return {"checks": [{"claim": "flies 38 minutes", "question": "How long does it fly?"}]}
    if "Draft answer:" in prompt or "strict reviewer" in prompt:
        marker = "Draft answer:\n" if "Draft answer:" in prompt else "Answer:\n"
        return {"answer": prompt.split(marker)[1].split("\n\n")[0], "confidence": 0.85}
    if "NOT IN CONTEXT." in prompt:
        return {"answers": ["38 minutes."]}
    raise AssertionError(f"unexpected prompt: {prompt[:80]}")


def writer_node(state: dict) -> dict:
    return {"answer": "The K2 flies 38 minutes [d1]. It is loved.", "sources": ["d1"], "confidence": 0.9}


def test_full_stack_in_langgraph(docs):
    llm = FakeLLM(scripted)
    set_llm(llm)
    shield = Shield(
        [
            RAGGrounding(sources=docs, require_citations=True),
            SelfConsistency(n=3),
            CoVe(),
            Critic(model="gpt-4o-mini"),
            Abstain(threshold=0.7),
        ]
    )
    graph = StateGraph(GraphState)
    graph.add_node("writer", shield.wrap(writer_node))
    graph.add_edge(START, "writer")
    graph.add_edge("writer", END)

    output = graph.compile().invoke({"question": "How long does the K2 drone fly?"})

    assert output["answer"] == "The K2 flies 38 minutes [d1]."
    assert output["sources"] == ["d1"] and output["confidence"] == 0.85
    assert {"grounding", "cove", "critic"} <= set(llm.usage.calls)
    assert "self_consistency: first two samples agree" in output["flags"]  # identical samples: no pick call
