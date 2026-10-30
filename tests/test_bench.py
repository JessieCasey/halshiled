"""The evaluation bench: schema validation, rubric matching and judge scoring."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from halshield import ABSTAIN_MSG

from evaluation import Bench, BenchCase, BenchFormatError, EvalReport, HallucinationJudge
from fakes import FakeLLM

BENCH_PATH = Path(__file__).parents[1] / "examples" / "bench.json"


@pytest.fixture(scope="module")
def bench() -> Bench:
    return Bench.load(BENCH_PATH)


def test_shipped_bench_is_valid_and_hard(bench):
    assert len(bench.cases) == 28
    assert len(bench.categories) >= 12
    assert any(not case.answerable for case in bench.cases)
    assert all(case.forbidden for case in bench.cases)  # every case has a deterministic trap check
    assert sum(len(case.evidence) >= 3 for case in bench.cases) >= 4  # multi-hop cases


def test_gold_answers_pass_their_own_rubric(bench):
    for case in bench.cases:
        gold = case.gold.removeprefix("UNANSWERABLE: ")
        assert case.missing_facts(gold) == (), case.id
        assert case.forbidden_hits(gold) == (), case.id


@pytest.mark.parametrize(
    ("case_id", "answer", "missing"),
    [
        ("c04", "The minimum commitment is 84,000 euros [d20][d05].", ()),
        ("c04", "It is 184,000 euros.", ("84,000|84000|84 000|84k|84 thousand",)),  # whole-number match only
        ("c05", "About 2.8 miles [d16].", ()),
        ("c05", "4.5 miles.", ("2.8|2.79|2.80|2.796|2.797|2.798",)),
        ("c05", "Roughly 2.798 miles (4.5 * 0.621371) [d16].", ()),
        ("c06", "Anneli Saar, and Rasmus Kivi today [d11].", ()),
        ("c06", "Anneli Saar [d11].", ("Kivi",)),
        ("c08", "No, the K3 tops out at 52 minutes [d03].", ()),
        ("c08", "Yes, the F9 flies 60 minutes.", ("no|none|doesn't|does not", "52")),
    ],
)
def test_rubric_matching(bench, case_id, answer, missing):
    case = next(case for case in bench.cases if case.id == case_id)
    assert case.missing_facts(answer) == missing


@pytest.mark.parametrize(
    ("case_id", "answer", "hits"),
    [
        ("c01", "The K3 flies 45 minutes per charge [d13].", ("45 minutes as the K3's flight time",)),
        ("c01", "52 minutes [d03]. The 2019 press release had promised 45 [d13].", ()),
        ("c02", "Yes, it is rated for -20 °C [d03].", ("says yes",)),
        ("c03", "The K3 Lite carries up to 0.6 kg [d16].", ("gives 0.6 kg as the K3 Lite's payload",)),
        ("c03", "Only the K3's payload (0.6 kg) is documented [d16].", ()),
        ("c04", "It costs 28,000 euros.", ("drops the three-year minimum term",)),
        ("c04", "That is 28,000 euros per year, so 84,000 euros over three years.", ()),
        ("c07", "Baltic Freight Lines operates 12 drones.", ("says 12 drones as the current fleet",)),
        ("c07", "It piloted 12 K2 drones in 2019; the current size is not given.", ()),
        ("c16", "Kestrel employs about 1,200 people [d18].", ("gives Kestrel Logistics' 1,200 employees",)),
        ("c17", "Every Kestrel drone is CE certified [d07].", ("claims all drones are certified",)),
        ("c17", "Every drone has propeller guards; only the K3 is CE-marked [d07].", ()),
        ("c19", "Rasmus Kivi was CTO in 2022 [d11].", ("names Kivi as the 2022 CTO",)),
        ("c19", "Anneli Saar; Rasmus Kivi was appointed in 2024 [d11].", ()),
        ("c20", "It flies 14 minutes longer.", ("uses the full K3's 52 minutes",)),
        ("c21", "It costs 53,000 euros per year.", ("prices the full K3",)),
        ("c21", "With the 2022 prices that was 46,000 euros.", ("uses the 2022 price list",)),
        ("c22", "The K3 Lite is rated for -20 °C [d03].", ("gives the K3's -20 to 40 °C range to the K3 Lite",)),
        ("c22", "Not documented; the full K3 is rated for -20 °C to 40 °C [d03].", ()),
        ("c23", "Kestrel moved its headquarters to Tallinn to cut costs.", ("accepts the move to Tallinn",)),
        ("c23", "Kestrel did not move to Tallinn; it moved to Rotterdam in 2019 [d01].", ()),
        ("c24", "Kestrel runs 90 trucks.", ("gives Kestrel Logistics' 90 trucks",)),
        ("c24", "Kestrel Logistics GmbH, a different company, runs 90 trucks [d18].", ()),
        ("c25", "Atlas 4.2 added support for the K3 Lite.", ("names a later release",)),
        ("c25", "Atlas 4.0 added the K3 Lite [d21]; 4.2 added per-category thresholds.", ()),
        ("c26", "It took 8 years.", ("counts from the founding",)),
        ("c27", "Within 48 hours, and you can buy it outright.", ("says the drone can be bought outright",)),
        ("c27", "Within 48 hours [d10]; no, you cannot buy it, drones are leased [d05].", ()),
        ("c28", "Kestrel offers three models: K2, K3 and K3 Lite.", ("counts the K2 or K4 as available",)),
    ],
)
def test_forbidden_assertions(bench, case_id, answer, hits):
    case = next(case for case in bench.cases if case.id == case_id)
    assert case.forbidden_hits(answer) == hits


def _write_bench(tmp_path: Path, payload: object) -> Path:
    path = tmp_path / "bench.json"
    path.write_text(payload if isinstance(payload, str) else json.dumps(payload), encoding="utf-8")
    return path


VALID_CASE = {
    "id": "x1", "category": "arithmetic", "question": "q?", "answerable": True, "gold": "g",
    "must_include": [["1"]], "traps": [], "evidence": ["d1"], "why_hard": "w",
}  # fmt: skip


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ("{not json", "not valid JSON"),
        ({"version": 2, "docs": {"d1": "t"}, "cases": [VALID_CASE]}, "version 3"),
        ({"version": 3, "docs": {"d1": "t"}, "cases": []}, "non-empty list"),
        ({"version": 3, "docs": {"d1": "t"}, "cases": [VALID_CASE, VALID_CASE]}, "duplicate"),
        ({"version": 3, "docs": {"d2": "t"}, "cases": [VALID_CASE]}, "unknown docs"),
        ({"version": 3, "docs": {"d1": "t"}, "cases": [{**VALID_CASE, "answerable": "yes"}]}, "answerable"),
        ({"version": 3, "docs": {"d1": "t"}, "cases": [{**VALID_CASE, "answerable": False}]}, "unanswerable"),
        (
            {
                "version": 3,
                "docs": {"d1": "t"},
                "cases": [{**VALID_CASE, "forbidden": [{"label": "x", "pattern": "("}]}],
            },
            "invalid pattern",
        ),
        ({"version": 3, "docs": {"d1": "t"}, "cases": [{**VALID_CASE, "forbidden": [{"pattern": "x"}]}]}, "label"),
    ],
    ids=[
        "bad-json",
        "old-version",
        "no-cases",
        "duplicate-id",
        "unknown-evidence",
        "bad-flag",
        "rubric-on-unanswerable",
        "bad-regex",
        "unlabelled-check",
    ],
)
def test_malformed_bench_is_rejected(tmp_path, payload, message):
    with pytest.raises(BenchFormatError, match=message):
        Bench.load(_write_bench(tmp_path, payload))


def test_select_filters_and_validates(bench):
    assert {case.category for case in bench.select(["false_premise"]).cases} == {"false_premise"}
    assert len(bench.select(limit=3).cases) == 3
    with pytest.raises(ValueError, match="unknown categories"):
        bench.select(["nope"])


ANSWERABLE = BenchCase.from_dict({**VALID_CASE, "forbidden": [{"label": "says 9", "pattern": "\\b9\\b"}]})
UNANSWERABLE = BenchCase.from_dict({**VALID_CASE, "id": "x2", "answerable": False, "must_include": []})


def judge_replying(**reply) -> HallucinationJudge:
    return HallucinationJudge(FakeLLM(lambda prompt: {"claims": [], "abstained": False, **reply}), {"d1": "t"})


@pytest.mark.parametrize(
    ("case", "answer", "reply", "expected"),
    [
        pytest.param(ANSWERABLE, "It is 1 [d1].", {}, (False, True, True), id="rubric-met"),
        pytest.param(ANSWERABLE, "It is 2.", {"correct": True}, (False, False, False), id="rubric-beats-lenient-judge"),
        pytest.param(
            ANSWERABLE, "It is 1, made in Peru.", {"claims": [{"claim": "made in Peru", "supported": False}]},
            (True, True, False), id="correct-but-hallucinated",
        ),
        pytest.param(ANSWERABLE, "It is 1, or 9.", {"correct": True}, (True, False, False), id="forbidden-beats-judge"),
        pytest.param(UNANSWERABLE, ABSTAIN_MSG, {}, (False, True, True), id="clean-abstention"),
        pytest.param(UNANSWERABLE, "Only the old count is given; the current one is not specified.", {"correct": True},
                     (False, True, True), id="says-what-is-unknown"),
        pytest.param(UNANSWERABLE, "It is 7.", {"claims": [{"claim": "7", "supported": False}]},
                     (True, False, False), id="invented-answer"),
    ],
)  # fmt: skip
def test_judge_scoring(case, answer, reply, expected):
    verdict = judge_replying(**reply).grade(case, answer)
    assert (verdict.hallucinated, verdict.correct, verdict.clean_and_correct) == expected


def _record(config: str, run: int, case_id: str, clean: bool, **extra: object) -> dict[str, object]:
    verdict = {"hallucinated": not clean, "trap_triggered": False, "correct": clean, "abstained": False}
    return {"config": config, "run": run, "id": case_id, "category": "c", "answerable": True,
            "clean_and_correct": clean, **verdict, **extra}  # fmt: skip


def test_report_tables():
    report = EvalReport(["baseline", "full"])
    for run in range(2):
        report.add(_record("baseline", run, "a", clean=run == 0, forbidden_hits=["says 9"]))
        report.add(_record("baseline", run, "b", clean=True))
        report.add(_record("full", run, "a", clean=True))
        report.add(_record("full", run, "b", clean=True))
    report.set_usage("baseline", {"writer": 400}, {"writer": 4})
    report.set_usage("full", {"writer": 400, "cove": 800}, {"writer": 4, "cove": 8})

    summary = report.summary()
    assert report.runs == 2
    assert summary.at["baseline", "clean & correct"] == 0.75
    assert summary.at["baseline", "c&c sd"] == 0.25  # per-run rates 1.0 and 0.5
    assert summary.at["full", "Δ c&c vs baseline"] == 0.25
    assert summary.at["full", "tokens/q"] == 300
    assert list(report.by_case().index) == ["a", "b"]  # hardest first
    failures = report.failures()
    assert failures[["id", "config", "reason", "runs"]].values.tolist() == [["a", "baseline", "trap: says 9", 1]]
    assert "Failures" in report.render()
