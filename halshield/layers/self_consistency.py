"""SelfConsistency layer."""

from __future__ import annotations

import contextvars
import re
from concurrent.futures import ThreadPoolExecutor

from ..contract import Node, State, require_question
from ..interfaces.layer import Layer
from ..interfaces.llm_client import LLMClient
from ..llm.default import get_llm
from ..prompts import SELF_CONSISTENCY_PICK
from ..utils import as_number, cited, clamp_unit

_NUMBER = re.compile(r"(?<!\w)\d+(?:[.,]\d+)*")  # not the digit in a name like "K3"


def fact_signature(answer: str) -> tuple[frozenset[str], frozenset[str]]:
    """Return the numbers and cited ids of an answer: what two samples must share to agree."""
    without_citations = re.sub(r"\[[^\[\]]+\]", "", answer)
    numbers = frozenset(number.replace(",", "") for number in _NUMBER.findall(without_citations))
    return numbers, frozenset(cited(answer))


def samples_agree(first: str, second: str) -> bool:
    """Return True if two answers state the same numbers from the same chunks (or are the same text)."""
    if " ".join(first.split()).casefold() == " ".join(second.split()).casefold():
        return True
    signature = fact_signature(first)
    return any(signature) and signature == fact_signature(second)


class SelfConsistency(Layer):
    """Sample the node n times and keep the answer most consistent with the others.

    Sets state["temperature"] for each sample (the node must honour it) and caps confidence
    by the agreement between samples. Samples run in parallel. With `early_stop`, two samples are
    drawn first; if they state the same numbers from the same chunks, the first is kept and the
    remaining samples and the pick call are skipped (2 calls instead of n + 1).
    """

    name = "self_consistency"

    def __init__(
        self, n: int = 3, temperature: float = 0.8, llm: LLMClient | None = None, early_stop: bool = True
    ) -> None:
        if n < 2:
            raise ValueError(f"n must be >= 2 to compare samples, got {n}")
        if not 0.0 < temperature <= 2.0:
            raise ValueError(f"temperature must be within (0, 2], got {temperature}")
        self.n, self.temperature, self.early_stop, self.llm = n, temperature, early_stop, llm or get_llm()

    def _sample(self, state: State, node: Node, count: int) -> list[State]:
        """Run the node `count` times in parallel, each in a copy of the caller's context (usage scopes)."""
        sampled = {**state, "temperature": self.temperature}
        with ThreadPoolExecutor(max_workers=count, thread_name_prefix="halshield-sc") as pool:
            futures = [pool.submit(contextvars.copy_context().run, node, sampled) for _ in range(count)]
            return [future.result() for future in futures]

    def run(self, state: State, node: Node) -> State:
        """Sample, pick the most consistent answer and lower confidence by disagreement."""
        question = require_question(state)
        first = 2 if self.early_stop else self.n
        samples = self._sample(state, node, first)
        if self.early_stop and samples_agree(samples[0].get("answer", ""), samples[1].get("answer", "")):
            best = samples[0]
            return {
                **best,
                "confidence": clamp_unit(as_number(best.get("confidence"), 1.0)),
                "flags": [*best.get("flags", []), "self_consistency: first two samples agree"],
            }
        if self.n > first:
            samples += self._sample(state, node, self.n - first)
        candidates = "\n".join(f"[{index}] {sample.get('answer', '')}" for index, sample in enumerate(samples))
        verdict = self.llm.json(SELF_CONSISTENCY_PICK.substitute(question=question, candidates=candidates), self.name)
        best_index = min(max(int(as_number(verdict.get("best"), 0)), 0), self.n - 1)
        best = samples[best_index]
        agreement = clamp_unit(as_number(verdict.get("agreement"), 1.0))
        return {
            **best,
            "confidence": min(as_number(best.get("confidence"), 1.0), agreement),
            "flags": [*best.get("flags", []), f"self_consistency: agreement={agreement:.2f}"],
        }
