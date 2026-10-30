"""LangGraph demo: retriever -> writer, with the writer wrapped by a Shield.

python examples/demo.py                                   # all layers, sample questions
python examples/demo.py --layers grounding,abstain "Who is the CFO?"
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path

from halshield import Abstain, CoVe, Critic, Layer, LLMClient, RAGGrounding, SelfConsistency, Shield, get_llm

from pipeline import STYLES, KeywordRetriever, QAPipeline, RAGWriter

BENCH = Path(__file__).with_name("bench.json")

LAYERS: dict[str, Callable[[dict[str, str], LLMClient], Layer]] = {
    "grounding": lambda docs, llm: RAGGrounding(sources=docs, require_citations=True, llm=llm),
    "sc": lambda docs, llm: SelfConsistency(n=3, llm=llm),
    "cove": lambda docs, llm: CoVe(llm=llm),
    "critic": lambda docs, llm: Critic(llm=llm),
    "abstain": lambda docs, llm: Abstain(threshold=0.7),
}
DEFAULT_QUESTIONS = [
    "How long can the Kestrel K2 fly on one charge?",
    "How long does the K3 take to recharge on its docking pad?",
]


def main() -> None:
    """Answer the given questions and print answers, flags and per-layer usage."""
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--layers", default="grounding,sc,cove,critic,abstain", help=f"processing order, from {list(LAYERS)}"
    )
    ap.add_argument("--writer", default="strict", choices=sorted(STYLES), help="writer prompt style")
    ap.add_argument("questions", nargs="*", default=DEFAULT_QUESTIONS)
    args = ap.parse_args()

    docs = json.loads(BENCH.read_text())["docs"]
    llm = get_llm()
    layers = [LAYERS[n](docs, llm) for n in args.layers.split(",") if n]
    pipeline = QAPipeline(KeywordRetriever(docs), Shield(layers=layers).wrap(RAGWriter(llm, args.writer)))
    llm.usage.reset()  # exclude one-time index embedding cost
    for q in args.questions:
        out = pipeline.invoke(q)
        print(f"\nQ: {q}\nA: {out['answer']}\n   sources={out.get('sources')} confidence={out.get('confidence')}")
        for f in out.get("flags", []):
            print(f"   - {f}")
    print(f"\nusage by label: tokens={dict(llm.usage.tokens)} calls={dict(llm.usage.calls)}")


if __name__ == "__main__":
    main()
