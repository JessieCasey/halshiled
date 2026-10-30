"""Runs layer configurations over the bench and grades every answer."""

from __future__ import annotations

import re
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict

from halshield import (
    Abstain,
    CoVe,
    Critic,
    Layer,
    LLMClient,
    Protection,
    RAGGrounding,
    RunContext,
    SelfConsistency,
    Shield,
)

from .bench import Bench
from .bench_case import BenchCase
from .eval_report import EvalReport, failure_reason
from .hallucination_judge import HallucinationJudge
from .progress_table import ProgressTable
from pipeline import STYLES, KeywordRetriever, QAPipeline, RAGWriter

ANSWER_COLUMNS = (
    ("config", 23), ("run", 3), ("case", 4), ("status", 6), ("time", 6), ("tokens", 6), ("calls", 5), ("missed", 8),
    ("answer", 50), ("reason", 40),
)  # fmt: skip
PROGRESS_COLUMNS = (("done", 7), *ANSWER_COLUMNS)
DEFAULT_WORKERS = 8
_CHUNK_TAG = re.compile(r"^\[([^\]]+)\]")


def retrieved_ids(context: Sequence[object]) -> list[str]:
    """Return the chunk ids of the "[id] text" entries in `context`, in order."""
    return [match.group(1) for entry in context if isinstance(entry, str) and (match := _CHUNK_TAG.match(entry))]


def missing_evidence(case: BenchCase, chunk_ids: Sequence[str]) -> list[str]:
    """Return the case's evidence docs that no retrieved chunk ("d21" or "d21#3") came from."""
    found = {chunk_id.split("#", 1)[0] for chunk_id in chunk_ids}
    return [doc_id for doc_id in case.evidence if doc_id not in found]


def answer_status(record: Mapping[str, object]) -> str:
    """Return the one-word status shown in the answer tables."""
    return "HALLUC" if record["hallucinated"] else ("ok" if record["correct"] else "wrong")


def answer_row(record: Mapping[str, object]) -> tuple[object, ...]:
    """Return one record's cells for ANSWER_COLUMNS."""
    return (
        record["config"], record["run"], record["id"], answer_status(record), f"{record['seconds']:.1f}s",
        record["tokens"], record["calls"], ",".join(record.get("missing_evidence") or []) or "-",
        record["answer"] or "(empty)", failure_reason(record),
    )  # fmt: skip


def print_answers(records: Sequence[Mapping[str, object]]) -> None:
    """Print every answer as one table (records are printed in the order given)."""
    table = ProgressTable(ANSWER_COLUMNS)
    for record in records:
        table.row(*answer_row(record))


class Evaluator:
    """Builds a retriever -> shielded-writer pipeline per configuration and judges its answers."""

    def __init__(self, llm: LLMClient, judge: HallucinationJudge, bench: Bench, writer_style: str = "helpful") -> None:
        if writer_style not in STYLES:
            raise ValueError(f"unknown writer style {writer_style!r}; choose from {sorted(STYLES)}")
        self.llm, self.judge, self.bench, self.writer_style = llm, judge, bench, writer_style

    @staticmethod
    def standard_configs(
        llm: LLMClient, docs: Mapping[str, str], threshold: float = 0.7, k: int = 4
    ) -> dict[str, list[Layer]]:
        """Build the configurations compared in docs/evaluation.md (k=4 suits multi-document cases).

        One method at a time first, then the three protection levels. SelfConsistency sits inside
        RAGGrounding, so it samples answers over one retrieval.
        """
        grounding = RAGGrounding(docs, k=k, llm=llm)
        rewriting = RAGGrounding(docs, k=k, llm=llm, rewrite_queries=3)
        self_consistency, cove, critic = SelfConsistency(llm=llm), CoVe(llm=llm), Critic(llm=llm)
        return {
            "baseline": [],
            "+grounding(rewrite)": [rewriting],
            "+citations(own RAG)": [RAGGrounding(llm=llm)],  # checks the pipeline's own retrieval, adds none
            "+grounding+cove": [grounding, cove],
            "+grounding+cove+abstain": [grounding, cove, Abstain(threshold)],
            "+grounding+sc": [self_consistency, grounding],
            "+grounding+critic": [grounding, critic],
            **{level.value: level.layers(docs, k=k, llm=llm) for level in Protection},
        }

    def _answer(self, config: str, pipeline: QAPipeline, run: int, case: BenchCase) -> tuple[dict, RunContext]:
        """Answer and grade one case; returns its record and the usage it caused (judge excluded)."""
        with self.llm.usage.measure() as used:
            started = time.perf_counter()
            output = pipeline.invoke(case.question)
            seconds = time.perf_counter() - started  # pipeline only; judging is not timed
        verdict = self.judge.grade(case, output["answer"])
        totals = used.totals()
        context_ids = retrieved_ids(output.get("context", []))
        evidence_missed = missing_evidence(case, context_ids)
        record = {
            "config": config,
            "run": run,
            "id": case.id,
            "category": case.category,
            "answerable": case.answerable,
            "question": case.question,
            "answer": output["answer"],
            "sources": output.get("sources", []),
            "flags": output.get("flags", []),
            **asdict(verdict),
            "clean_and_correct": verdict.clean_and_correct,
            "context_ids": context_ids,
            "missing_evidence": evidence_missed,
            "evidence_recall": round(1 - len(evidence_missed) / len(case.evidence), 3) if case.evidence else 1.0,
            "seconds": round(seconds, 3),
            "tokens": totals.tokens,
            "calls": totals.calls,
            "tokens_by_label": dict(used.tokens),  # writer, grounding, self_consistency, cove, critic, ...
            "calls_by_label": dict(used.calls),
        }
        return record, used

    def run(
        self,
        configs: Mapping[str, Sequence[Layer]],
        repeats: int = 1,
        workers: int = DEFAULT_WORKERS,
        verbose: bool = True,
    ) -> EvalReport:
        """Answer and grade every case `repeats` times under every configuration, `workers` answers at a time.

        Rows are printed as answers finish; the report's records are sorted by config, case and run.
        """
        if not configs:
            raise ValueError("no configurations to run")
        if repeats < 1:
            raise ValueError(f"repeats must be >= 1, got {repeats}")
        if workers < 1:
            raise ValueError(f"workers must be >= 1, got {workers}")
        pipelines = {
            name: QAPipeline(
                KeywordRetriever(self.bench.docs), Shield(layers).wrap(RAGWriter(self.llm, self.writer_style))
            )
            for name, layers in configs.items()
        }
        tasks = [(config, run, case) for config in configs for run in range(repeats) for case in self.bench.cases]
        progress = ProgressTable(PROGRESS_COLUMNS) if verbose else None
        results: list[tuple[dict, RunContext]] = []
        pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="halshield-eval")
        try:
            futures = [pool.submit(self._answer, config, pipelines[config], run, case) for config, run, case in tasks]
            for done, future in enumerate(as_completed(futures), start=1):
                results.append(future.result())
                if progress:
                    progress.row(f"{done}/{len(tasks)}", *answer_row(results[-1][0]))
        finally:
            pool.shutdown(wait=True, cancel_futures=True)  # on error or Ctrl-C, drop queued answers

        config_order = {name: index for index, name in enumerate(configs)}
        case_order = {case.id: index for index, case in enumerate(self.bench.cases)}
        results.sort(key=lambda item: (config_order[item[0]["config"]], case_order[item[0]["id"]], item[0]["run"]))
        report = EvalReport(list(configs))
        usage = {name: (Counter(), Counter()) for name in configs}
        for record, used in results:
            report.add(record)
            usage[record["config"]][0].update(used.tokens)
            usage[record["config"]][1].update(used.calls)
        for name, (tokens, calls) in usage.items():
            report.set_usage(name, tokens, calls)
        return report
