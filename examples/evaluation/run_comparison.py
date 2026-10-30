"""Before/after comparison of two evaluation runs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from .eval_report import failure_reason

_METRICS = {"clean_and_correct": "c&c", "hallucinated": "halluc"}
_COSTS = {"seconds": "sec/q", "tokens": "tokens/q"}
_REASON_WIDTH = 70
# Configs renamed since earlier results files: the same layers under their protection-level name.
RENAMED = {"+grounding": "minimal", "full": "maximum"}


class RunComparison:
    """Compares the configs and cases two runs have in common (e.g. before and after a fix)."""

    def __init__(self, before: Sequence[Mapping[str, Any]], after: Sequence[Mapping[str, Any]]) -> None:
        if not before or not after:
            raise ValueError("both runs need at least one record")
        self.before, self.after = pd.DataFrame(list(before)), pd.DataFrame(list(after))
        for frame in (self.before, self.after):
            frame["config"] = frame["config"].replace(RENAMED)
        for frame in (self.before, self.after):
            frame["reason"] = [failure_reason(record) for record in frame.to_dict("records")]
        configs = [config for config in dict.fromkeys(self.after["config"]) if config in set(self.before["config"])]
        if not configs:
            raise ValueError("the two runs have no configuration in common")
        self.configs = tuple(configs)

    def _per(self, frame: pd.DataFrame, keys: list[str], columns: Mapping[str, str]) -> pd.DataFrame:
        present = [column for column in columns if column in frame]
        return frame[frame["config"].isin(self.configs)].groupby(keys, sort=False)[present].mean()

    def summary(self) -> pd.DataFrame:
        """Return per-config metrics before, after and their change."""
        columns = {**_METRICS, **_COSTS}
        before, after = self._per(self.before, ["config"], columns), self._per(self.after, ["config"], columns)
        table = pd.DataFrame(index=list(self.configs))
        for column, label in columns.items():
            if column in before and column in after:
                table[f"{label} before"] = before[column]
                table[f"{label} after"] = after[column]
                table[f"Δ {label}"] = after[column] - before[column]
        return table

    def changes(self) -> pd.DataFrame:
        """Return the cases whose clean & correct rate changed, regressions first."""
        keys = ["config", "id"]
        rate_before = self._per(self.before, keys, _METRICS)["clean_and_correct"].rename("before")
        rate_after = self._per(self.after, keys, _METRICS)["clean_and_correct"].rename("after")
        table = pd.concat([rate_before, rate_after], axis=1, join="inner")
        table = table[table["before"] != table["after"]].copy()
        table["change"] = ["fixed" if after > before else "regressed" for before, after in table.to_numpy()]
        # The failure that disappeared (fixed) or appeared (regressed).
        reasons = {
            "fixed": self.before.groupby(keys)["reason"].agg(lambda values: next((v for v in values if v), "")),
            "regressed": self.after.groupby(keys)["reason"].agg(lambda values: next((v for v in values if v), "")),
        }
        table["reason"] = [reasons[change].get(key, "")[:_REASON_WIDTH] for key, change in table["change"].items()]
        order = {"regressed": 0, "fixed": 1}
        return table.sort_values(by="change", key=lambda column: column.map(order), kind="stable").reset_index()

    def render(self) -> str:
        """Return the comparison tables as text."""
        with pd.option_context(
            "display.float_format", "{:.2f}".format, "display.width", 200, "display.max_colwidth", _REASON_WIDTH
        ):
            changes = self.changes()
            counts = changes["change"].value_counts()
            title = f"Changed cases: {counts.get('fixed', 0)} fixed, {counts.get('regressed', 0)} regressed"
            body = changes.to_string(index=False) if len(changes) else "none"
            return f"Before -> after by config\n\n{self.summary().to_string()}\n\n{title}\n\n{body}"
