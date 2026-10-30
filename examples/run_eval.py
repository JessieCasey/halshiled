"""Compare layer stacks on the hard bench with a rubric-aware LLM judge.

python examples/run_eval.py [--limit N] [--repeats 3] [--configs baseline,maximum] [--categories false_premise]
python examples/run_eval.py --levels --repeats 3     # baseline vs the minimal, medium and maximum levels
python examples/run_eval.py --out results/after.jsonl --compare results/before.jsonl
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from halshield import HalshieldError, OpenAILLM, Protection, RunContext, get_llm

from evaluation import Bench, BenchFormatError, EvalReport, Evaluator, HallucinationJudge, RunComparison
from evaluation.evaluator import DEFAULT_WORKERS, print_answers
from pipeline import STYLES

BENCH_PATH = Path(__file__).with_name("bench.json")


def _csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Compare halshield layer stacks on the hard bench.")
    parser.add_argument("--bench", type=Path, default=BENCH_PATH)
    parser.add_argument("--limit", type=int, default=None, help="evaluate at most N cases")
    parser.add_argument("--categories", type=_csv, default=(), help="comma-separated case categories")
    parser.add_argument("--configs", type=_csv, default=(), help="comma-separated config names (default: all)")
    parser.add_argument(
        "--levels",
        action="store_true",
        help=f"run only baseline and the protection levels ({', '.join(level.value for level in Protection)})",
    )
    parser.add_argument("--repeats", type=int, default=1, help="answer every case N times (reports spread)")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="answers evaluated in parallel")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Abstain threshold of +grounding+cove+abstain (the maximum level uses 0.7)",
    )
    parser.add_argument("--writer", default="helpful", choices=sorted(STYLES), help="writer prompt style")
    parser.add_argument("--out", type=Path, default=Path("results.jsonl"), help="per-answer details file")
    parser.add_argument("--compare", type=Path, default=None, help="earlier results file to diff against")
    args = parser.parse_args(argv)
    if args.levels and args.configs:
        parser.error("pass either --levels or --configs, not both")
    return args


def main(argv: list[str] | None = None) -> int:
    """Run the configurations, print the tables and save per-answer details."""
    args = parse_args(argv)
    started = time.perf_counter()
    try:
        before = EvalReport.load_records(args.compare) if args.compare else None
        bench = Bench.load(args.bench).select(args.categories, args.limit)
        llm = get_llm()
        # Separate usage counters so judge tokens don't count toward the configs' cost.
        judge_llm = OpenAILLM(model=os.getenv("HALSHIELD_JUDGE_MODEL"), usage=RunContext())
        judge = HallucinationJudge(judge_llm, bench.docs)
        configs = Evaluator.standard_configs(llm, bench.docs, args.threshold)
        if unknown := set(args.configs) - configs.keys():
            raise ValueError(f"unknown configs {sorted(unknown)}; available: {list(configs)}")
        chosen = ("baseline", *(level.value for level in Protection)) if args.levels else args.configs
        if chosen:
            configs = {name: layers for name, layers in configs.items() if name in chosen}
        report = Evaluator(llm, judge, bench, args.writer).run(configs, args.repeats, args.workers)
    except (BenchFormatError, HalshieldError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    report.save(args.out)
    elapsed = time.perf_counter() - started
    runs = f"{len(bench.cases)} cases x {args.repeats} run(s)"
    print(f"\n{runs}, model={llm.model}, writer={args.writer}, judge={judge_llm.model}")
    pipeline_minutes = report.total_seconds / 60
    print(f"Wall time {elapsed / 60:.1f} min with {args.workers} workers ({pipeline_minutes:.1f} min of pipeline time)")
    print("\nAll answers (by config, case, run)\n")
    print_answers(report.records)
    print()
    print(report.render())
    if before is not None:
        print(f"\n\nComparison with {args.compare}\n")
        print(RunComparison(before, report.records).render())
    print(f"\nPer-answer details: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
