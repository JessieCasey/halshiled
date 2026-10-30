"""Evaluation results and their tables."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

_RATE_COLUMNS = {
    "hallucinated": "hallucination",
    "trap_triggered": "trap hit",
    "correct": "correct",
    "clean_and_correct": "clean & correct",
    "abstained": "abstain",
}
_SPREAD_COLUMNS = {"clean_and_correct": "c&c sd", "hallucinated": "halluc sd"}
_REASON_WIDTH = 90


def failure_reason(record: Mapping[str, Any]) -> str:
    """Explain in one line why a record is not clean & correct ("" if it is)."""
    if record.get("clean_and_correct"):
        return ""
    if forbidden := record.get("forbidden_hits"):
        return "trap: " + "; ".join(forbidden)
    if unsupported := record.get("unsupported"):
        return "unsupported: " + "; ".join(unsupported)
    if record.get("abstained"):
        return "abstained on an answerable question" if record.get("answerable") else "abstained"
    if missing := record.get("missing_facts"):
        return "missing: " + ", ".join(missing)
    return "judged incorrect"


class EvalReport:
    """Per-answer records plus per-config usage, rendered as summary tables."""

    def __init__(self, configs: Sequence[str]) -> None:
        if not configs:
            raise ValueError("a report needs at least one configuration")
        self.configs = tuple(configs)
        self.records: list[dict[str, Any]] = []
        self.tokens: dict[str, Counter[str]] = {}
        self.calls: dict[str, Counter[str]] = {}

    @classmethod
    def from_records(cls, records: Sequence[Mapping[str, Any]]) -> EvalReport:
        """Rebuild a report from saved records (configs in first-seen order), usage included when saved."""
        report = cls(list(dict.fromkeys(record["config"] for record in records)))
        tokens = {config: Counter() for config in report.configs}
        calls = {config: Counter() for config in report.configs}
        for record in records:
            report.add(record)
            tokens[record["config"]].update(record.get("tokens_by_label") or {"total": record.get("tokens", 0)})
            calls[record["config"]].update(record.get("calls_by_label") or {"total": record.get("calls", 0)})
        for config in report.configs:
            report.set_usage(config, tokens[config], calls[config])
        return report

    def add(self, record: Mapping[str, Any]) -> None:
        """Add one answer's record (config, run, case fields, answer, verdict fields)."""
        if record.get("config") not in self.configs:
            raise ValueError(f"record for unknown config {record.get('config')!r}")
        self.records.append({"run": 0, **record})

    def set_usage(self, config: str, tokens: Mapping[str, int], calls: Mapping[str, int]) -> None:
        """Store the usage counters measured for `config` (over all of its runs)."""
        self.tokens[config], self.calls[config] = Counter(tokens), Counter(calls)

    def _frame(self) -> pd.DataFrame:
        if not self.records:
            raise ValueError("the report has no records yet")
        return pd.DataFrame(self.records)

    @property
    def runs(self) -> int:
        """Return how many times each case was answered per config."""
        return int(self._frame()["run"].nunique())

    def summary(self) -> pd.DataFrame:
        """Return one row per config: rates, run-to-run spread, change vs the first config, cost per question."""
        frame = self._frame()
        by_config = frame.groupby("config", sort=False)
        rates = by_config[list(_RATE_COLUMNS)].mean().rename(columns=_RATE_COLUMNS).reindex(self.configs)
        if self.runs > 1:
            per_run = frame.groupby(["config", "run"])[list(_SPREAD_COLUMNS)].mean()
            spread = per_run.groupby(level="config").std(ddof=0).rename(columns=_SPREAD_COLUMNS)
            rates = rates.join(spread)
        reference = self.configs[0]
        rates[f"Δ c&c vs {reference}"] = rates["clean & correct"] - rates.at[reference, "clean & correct"]
        counts = by_config.size()
        rates["tokens/q"] = [sum(self.tokens.get(config, {}).values()) / counts[config] for config in rates.index]
        rates["LLM calls/q"] = [sum(self.calls.get(config, {}).values()) / counts[config] for config in rates.index]
        if "evidence_recall" in frame:
            rates["evidence recall"] = by_config["evidence_recall"].mean()
        if "seconds" in frame:
            rates["sec/q"] = by_config["seconds"].mean()
            rates["p95 sec"] = by_config["seconds"].quantile(0.95)
        return rates

    def vs_reference(self) -> pd.DataFrame:
        """Return each config against the first one (no protection): rates with their change, cost with its ratio.

        "Δ" columns are percentage points; "x" columns are how many times the first config's cost.
        """
        summary = self.summary()
        reference = summary.iloc[0]
        table = pd.DataFrame(index=summary.index)
        for column, label in (("clean & correct", "c&c"), ("correct", "correct"), ("hallucination", "halluc")):
            table[label] = summary[column]
            table[f"Δ {label}, pp"] = (summary[column] - reference[column]) * 100
        for column, label in (("sec/q", "time"), ("tokens/q", "tokens")):
            if column in summary:
                table[column] = summary[column]
                table[f"x {label}"] = summary[column] / reference[column] if reference[column] else float("nan")
        return table

    def by_category(self, metric: str = "clean_and_correct") -> pd.DataFrame:
        """Return `metric` rate by config (rows) and case category (columns)."""
        if metric not in _RATE_COLUMNS:
            raise ValueError(f"unknown metric {metric!r}; choose from {list(_RATE_COLUMNS)}")
        return (
            self._frame()
            .pivot_table(index="config", columns="category", values=metric, aggfunc="mean", sort=False)
            .reindex(self.configs)
        )

    def by_case(self) -> pd.DataFrame:
        """Return the clean & correct rate per case (rows, hardest first) and config (columns)."""
        table = self._frame().pivot_table(
            index="id", columns="config", values="clean_and_correct", aggfunc="mean", sort=False
        )[list(self.configs)]
        return table.loc[table.mean(axis=1).sort_values(kind="stable").index]

    def failures(self, limit: int | None = None) -> pd.DataFrame:
        """Return one row per failed answer with the reason, most frequent failures first."""
        frame = self._frame()
        frame["reason"] = [failure_reason(record) for record in frame.to_dict("records")]
        frame["evidence missed"] = [
            ", ".join(record.get("missing_evidence") or []) for record in frame.to_dict("records")
        ]
        failed = frame[frame["reason"] != ""]
        table = (
            failed.groupby(["id", "config", "reason", "evidence missed"], sort=False)
            .size()
            .rename("runs")
            .reset_index()
            .sort_values(["id", "runs"], ascending=[True, False], kind="stable")
        )
        table["reason"] = table["reason"].str.slice(0, _REASON_WIDTH)
        return table.head(limit).reset_index(drop=True) if limit else table.reset_index(drop=True)

    def tokens_by_label(self) -> pd.DataFrame:
        """Return tokens per question by config and usage label (writer, grounding, cove, ...)."""
        counts = self._frame().groupby("config").size()
        rows = {
            config: {label: n / counts[config] for label, n in self.tokens.get(config, {}).items()}
            for config in self.configs
        }
        return pd.DataFrame.from_dict(rows, orient="index").fillna(0.0)

    @property
    def total_seconds(self) -> float:
        """Return the summed pipeline time of all answers (judging excluded)."""
        frame = self._frame()
        return float(frame["seconds"].sum()) if "seconds" in frame else 0.0

    @classmethod
    def load_records(cls, path: str | Path) -> list[dict[str, Any]]:
        """Read per-answer records written by `save`."""
        source = Path(path)
        try:
            lines = source.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise ValueError(f"cannot read results file {source}: {exc}") from exc
        try:
            records = [json.loads(line) for line in lines if line.strip()]
        except json.JSONDecodeError as exc:
            raise ValueError(f"{source} is not a JSON-lines results file: {exc}") from exc
        if not records or not all(isinstance(record, dict) and "config" in record for record in records):
            raise ValueError(f"{source} has no evaluation records")
        return records

    def save(self, path: str | Path) -> None:
        """Write per-answer records as JSON lines."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with Path(path).open("w", encoding="utf-8") as handle:
            handle.writelines(json.dumps(record, ensure_ascii=False) + "\n" for record in self.records)

    def render(self, max_failures: int = 40) -> str:
        """Return all tables as text."""
        with pd.option_context(
            "display.float_format", "{:.2f}".format, "display.width", 200, "display.max_colwidth", _REASON_WIDTH
        ):
            failures = self.failures(max_failures)
            sections = {
                f"Before ({self.configs[0]}) and after (each config)": self.vs_reference(),
                f"Summary (rates over all cases, {self.runs} run(s) per case)": self.summary(),
                "Clean & correct by category": self.by_category("clean_and_correct"),
                "Hallucination by category": self.by_category("hallucinated"),
                "Clean & correct by case (hardest first)": self.by_case(),
                f"Failures (first {max_failures})": failures.to_string(index=False) if len(failures) else "none",
                "Tokens per question by label": self.tokens_by_label(),
            }
            return "\n\n".join(
                f"{title}\n\n{table if isinstance(table, str) else table.to_string()}"
                for title, table in sections.items()
            )
