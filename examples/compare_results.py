"""Diff two saved evaluation runs: which cases a change fixed or broke.

python examples/compare_results.py results/before.jsonl results/after.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from evaluation import EvalReport, RunComparison


def main(argv: list[str] | None = None) -> int:
    """Print the before/after tables for two results files."""
    parser = argparse.ArgumentParser(description="Compare two run_eval.py results files.")
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    args = parser.parse_args(argv)
    try:
        comparison = RunComparison(EvalReport.load_records(args.before), EvalReport.load_records(args.after))
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(comparison.render())
    return 0


if __name__ == "__main__":
    sys.exit(main())
