"""Print the report tables of a saved evaluation run, without calling the API.

python examples/show_results.py results/levels.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from evaluation import EvalReport


def main(argv: list[str] | None = None) -> int:
    """Print all report tables for a results file."""
    parser = argparse.ArgumentParser(description="Show the tables of a run_eval.py results file.")
    parser.add_argument("results", type=Path)
    args = parser.parse_args(argv)
    try:
        report = EvalReport.from_records(EvalReport.load_records(args.results))
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(report.render())
    return 0


if __name__ == "__main__":
    sys.exit(main())
