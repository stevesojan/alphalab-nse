"""Trial ledger: every configuration ever evaluated, so the number of trials is known (for DSR)
and results can be revisited -- the automated version of alpha_log_fn_liab_fair_val.md."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import polars as pl


class Ledger:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.rows: list[dict] = []
        if self.path.exists():
            self.rows = pl.read_parquet(self.path).to_dicts()

    def add(self, family: str, expression: str, params: dict, period: str, metrics: dict, note: str = "") -> None:
        self.rows.append({
            "ts": datetime.now().isoformat(timespec="seconds"), "family": family, "expression": expression,
            "params": json.dumps(params, sort_keys=True, default=str), "period": period, "note": note,
            **{k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))},
        })

    def n_trials(self, period: str | None = None) -> int:
        keys = {(r["family"], r["params"]) for r in self.rows if period is None or r["period"] == period}
        return len(keys)

    def frame(self) -> pl.DataFrame:
        return pl.DataFrame(self.rows) if self.rows else pl.DataFrame()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.frame().write_parquet(self.path)
        self.path.with_suffix(".md").write_text(self.markdown(), encoding="utf-8")

    def markdown(self) -> str:
        lines = ["# Trial ledger", "", f"Unique configurations: {self.n_trials()}", ""]
        df = self.frame()
        if df.is_empty():
            return "\n".join(lines)
        for (family,), g in df.group_by("family", maintain_order=True):
            lines += [f"## {family}", "", "| period | params | Sharpe | Fitness | Returns | Turnover | Drawdown | Margin bps | note |",
                      "|---|---|---|---|---|---|---|---|---|"]
            for r in g.iter_rows(named=True):
                lines.append(f"| {r['period']} | `{r['params']}` | {r.get('sharpe', 0):.2f} | {r.get('fitness', 0):.2f} | "
                             f"{r.get('returns', 0):.2%} | {r.get('turnover', 0):.2%} | {r.get('drawdown', 0):.2%} | "
                             f"{r.get('margin_bps', 0):.1f} | {r['note']} |")
            lines.append("")
        return "\n".join(lines)
