"""Evaluation output: the live progress table, timing columns and before/after comparison."""

from __future__ import annotations

import io
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from halshield import Abstain, RunContext

from evaluation import Bench, EvalReport, Evaluator, HallucinationJudge, ProgressTable, RunComparison
from evaluation.evaluator import missing_evidence, retrieved_ids
from fakes import FakeLLM

BENCH_PATH = Path(__file__).parents[1] / "examples" / "bench.json"


def test_progress_table_aligns_and_truncates():
    stream = io.StringIO()
    table = ProgressTable([("case", 4), ("answer", 10)], stream)
    table.row("c01", "short")
    table.row("c02", "a long\n\nmultiline answer")
    assert stream.getvalue().splitlines() == [
        "case  answer",
        "----  ----------",
        "c01   short",
        "c02   a long mu…",
    ]
    with pytest.raises(ValueError, match="expected 2 values"):
        table.row("c03")


def _record(config: str, case_id: str, clean: bool, seconds: float = 1.0, **extra: object) -> dict[str, object]:
    return {
        "config": config, "run": 0, "id": case_id, "category": "c", "answerable": True,
        "hallucinated": not clean, "trap_triggered": False, "correct": clean, "abstained": False,
        "clean_and_correct": clean, "seconds": seconds, "tokens": 100, **extra,
    }  # fmt: skip


def test_summary_reports_time_per_question():
    report = EvalReport(["baseline"])
    for seconds in (1.0, 3.0):
        report.add(_record("baseline", f"c{seconds}", clean=True, seconds=seconds))
    report.set_usage("baseline", {"writer": 200}, {"writer": 2})
    assert report.summary().at["baseline", "sec/q"] == 2.0
    assert report.total_seconds == 4.0


def test_comparison_shows_fixed_and_regressed_cases():
    before = [
        _record("full", "c01", clean=False, forbidden_hits=["45 minutes"]),
        _record("full", "c02", clean=True),
        _record("full", "c03", clean=True),
        _record("old_only", "c01", clean=True),
    ]
    after = [
        _record("maximum", "c01", clean=True, seconds=2.0),
        _record("maximum", "c02", clean=False, seconds=2.0, unsupported=["invented price"]),
        _record("maximum", "c03", clean=True, seconds=2.0),
    ]
    comparison = RunComparison(before, after)
    assert comparison.configs == ("maximum",)  # only configs both runs have; "full" was renamed "maximum"
    summary = comparison.summary()
    assert summary.at["maximum", "Δ c&c"] == 0.0
    assert summary.at["maximum", "Δ sec/q"] == 1.0
    changes = comparison.changes()
    assert changes[["id", "change", "reason"]].values.tolist() == [
        ["c02", "regressed", "unsupported: invented price"],
        ["c01", "fixed", "trap: 45 minutes"],
    ]
    assert "1 fixed, 1 regressed" in comparison.render()


def test_comparison_needs_a_shared_config():
    with pytest.raises(ValueError, match="no configuration in common"):
        RunComparison([_record("a", "c01", True)], [_record("b", "c01", True)])


def test_results_round_trip(tmp_path):
    report = EvalReport(["full"])
    report.add(_record("full", "c01", clean=True))
    path = tmp_path / "runs" / "before.jsonl"
    report.save(path)
    assert EvalReport.load_records(path)[0]["id"] == "c01"
    path.write_text(json.dumps({"no": "config"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no evaluation records"):
        EvalReport.load_records(path)


def test_measure_separates_concurrent_usage():
    usage = RunContext()

    def work(tokens: int) -> int:
        with usage.measure() as mine:
            for _ in range(50):
                usage.record("writer", tokens)
        return mine.totals().tokens

    with ThreadPoolExecutor(max_workers=4) as pool:
        per_thread = list(pool.map(work, [1, 2, 3, 4]))
    assert per_thread == [50, 100, 150, 200]
    assert usage.totals() == (500, 200)
    with RunContext().measure() as other:  # scopes only collect from the context they were opened on
        usage.record("writer", 7)
    assert other.totals() == (0, 0)


def test_parallel_run_sorts_records_and_splits_usage(capsys):
    bench = Bench.load(BENCH_PATH).select(limit=4)
    llm = FakeLLM(lambda prompt: {"answer": "Not enough information to answer.", "sources": [], "confidence": 0.1})
    judge = HallucinationJudge(FakeLLM(lambda prompt: {"claims": [], "abstained": True}), bench.docs)
    report = Evaluator(llm, judge, bench).run({"baseline": [], "+abstain": [Abstain(0.7)]}, repeats=2, workers=4)

    keys = [(record["config"], record["id"], record["run"]) for record in report.records]
    assert keys == sorted(keys, key=lambda key: (key[0] != "baseline", key[1], key[2]))
    assert len(keys) == 16
    assert sum(record["tokens"] for record in report.records) == llm.usage.totals().tokens
    assert all(record["calls"] == 1 for record in report.records)  # writer only: judge usage is separate
    assert "16/16" in capsys.readouterr().out
    assert all(record["context_ids"] and 0.0 <= record["evidence_recall"] <= 1.0 for record in report.records)
    assert "evidence recall" in report.summary()


def test_evidence_recall_counts_chunks_by_their_document():
    case = Bench.load(BENCH_PATH).select(limit=4).cases[3]  # c04: needs d14, d20, d05
    ids = retrieved_ids(["[d20] prices", "[d21#3] notes", "untagged", "[d14] lite"])
    assert ids == ["d20", "d21#3", "d14"]
    assert missing_evidence(case, ids) == ["d05"]


def test_before_and_after_table_compares_with_the_first_config():
    report = EvalReport(["baseline", "medium"])
    for case_id, clean in (("a", False), ("b", True)):
        report.add(_record("baseline", case_id, clean=clean, seconds=1.0))
        report.add(_record("medium", case_id, clean=True, seconds=3.0))
    report.set_usage("baseline", {"writer": 200}, {"writer": 2})
    report.set_usage("medium", {"writer": 600, "self_consistency": 200}, {"writer": 6, "self_consistency": 1})
    table = report.vs_reference()
    assert table.at["medium", "Δ c&c, pp"] == 50.0
    assert table.at["medium", "x time"] == 3.0
    assert table.at["medium", "x tokens"] == 4.0
    assert table.at["baseline", "x tokens"] == 1.0
    assert report.render().startswith("Before (baseline) and after (each config)")


def test_report_rebuilt_from_saved_records_splits_tokens_by_layer():
    records = [
        _record(
            "medium", "a", clean=True, tokens_by_label={"writer": 600, "grounding": 100}, calls_by_label={"writer": 2}
        ),
        _record("medium", "b", clean=True, tokens_by_label={"writer": 400}, calls_by_label={"writer": 2}),
        _record("old", "a", clean=True),  # saved before per-layer usage existed: only its total
    ]
    table = EvalReport.from_records(records).tokens_by_label()
    assert table.at["medium", "writer"] == 500 and table.at["medium", "grounding"] == 50
    assert table.at["old", "total"] == 100
