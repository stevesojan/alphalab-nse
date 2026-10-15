"""Phase 5 research study on NSE (run: python -m alphalab.study).

PROTOCOL -- fixed in code before any hold-out number was computed:
  * Universe: top 100 Nifty-200 names by trailing 60-day median traded value (no look-ahead).
  * Settings (BRAIN-like): sector neutralization, decay 4, truncation 0.08, delay 1. Metrics are gross
    of costs (as on BRAIN); a net Sharpe at 5 bps one-way cost is reported alongside.
  * In-sample (IS): 2015-01-01 .. 2022-12-31. Hold-out (HO): 2023-01-01 .. last date, evaluated ONCE.
  * Selection rule: plateau (maximin Sharpe over each grid point and its neighbours), never the raw best.
  * Walk-forward inside IS: test years 2018-2022, expanding training window from 2015, 5-day embargo,
    parameters re-selected by the plateau rule on each training window.
  * Deflated Sharpe uses every configuration in every family as the trial set.
  * Combined alpha: equal-weight mean of cross-sectional ranks of the six IS-selected family alphas.
"""
from __future__ import annotations

import json
import time
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from .backtest import metrics as M
from .backtest.pipeline import Settings, research_panel, run
from .signals import indicators as ind
from .signals import operators as op
from .validation import stats as S
from .validation.ledger import Ledger
from .validation.splits import walk_forward
from .validation.sweep import grid, period_metrics, plateau_select

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "study"
IS_START, IS_END, HO_START = date(2015, 1, 1), date(2022, 12, 31), date(2023, 1, 1)
SETTINGS = Settings(neutralization="sector", decay=4, truncation=0.08, delay=1)
COST_BPS = 5.0

FAMILIES = {
    "ema_cross": dict(
        expr="ema(close, fast) / ema(close, slow) - 1   (trend following)",
        axes={"fast": [5, 10, 20], "slow": [50, 100, 200]},
        build=lambda p: ind.ema("close", p["fast"]) / ind.ema("close", p["slow"]) - 1),
    "rsi_reversion": dict(
        expr="50 - rsi(close, n)   (fade overbought / oversold)",
        axes={"n": [2, 3, 5, 7, 10, 14, 21, 28]},
        build=lambda p: 50 - ind.rsi("close", p["n"])),
    "breakout": dict(
        expr="ts_mean(breakout(n, vol_mult=1.5), hold)   (volume-confirmed Donchian breakout, held `hold` days)",
        axes={"n": [10, 20, 55, 100], "hold": [1, 5, 20]},
        build=lambda p: op.ts_mean(ind.breakout(p["n"], 1.5), p["hold"])),
    "sr_channel": dict(
        expr="0.5 - channel_position(n)   (revert toward the middle of the prior n-day support/resistance channel)",
        axes={"n": [10, 20, 50, 100, 200]},
        build=lambda p: 0.5 - ind.channel_position(p["n"])),
    "sr_pivot": dict(
        expr="0.5 - (close - pivot_support(k)) / (pivot_resistance(k) - pivot_support(k))   (confirmed swing levels)",
        axes={"k": [3, 5, 10, 20]},
        build=lambda p: 0.5 - (pl.col("close") - ind.pivot_support(p["k"]))
        / (ind.pivot_resistance(p["k"]) - ind.pivot_support(p["k"]))),
    "reversal": dict(
        expr="-returns(close, n)   (short-term reversal baseline)",
        axes={"n": [1, 2, 3, 5, 10, 21]},
        build=lambda p: -op.ts_returns("close", p["n"])),
}


def key(p: dict, axes: dict) -> tuple:
    return tuple(p[k] for k in axes)


def net_sharpe(d: pl.DataFrame, bps: float = COST_BPS) -> float:
    return M.sharpe((d["pnl"] - d["traded"] * bps * 1e-4).to_numpy())


def slice_(d: pl.DataFrame, a, b) -> pl.DataFrame:
    return d.filter((pl.col("date") >= a) & (pl.col("date") <= b))


def evaluate_all(panel: pl.DataFrame) -> dict:
    """Daily PnL for every configuration of every family (signals use past data only, so slicing by
    period afterwards is equivalent to re-running on that period)."""
    res = {}
    for fam, spec in FAMILIES.items():
        res[fam] = {}
        for p in grid(**spec["axes"]):
            df = panel.with_columns(alpha=spec["build"](p))
            res[fam][key(p, spec["axes"])] = run(df, "alpha", SETTINGS).daily
    return res


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)
    (OUT / "export").mkdir(exist_ok=True)
    panel = research_panel(SETTINGS.top_n)
    last = panel["date"].max()
    res = evaluate_all(panel)
    print(f"evaluated {sum(len(v) for v in res.values())} configurations in {time.time() - t0:.0f}s", flush=True)

    ledger = Ledger(OUT / "ledger.parquet")
    ledger.rows = []
    all_is_sr_daily = []
    for fam, spec in FAMILIES.items():
        for k, d in res[fam].items():
            p = dict(zip(spec["axes"], k))
            m = period_metrics(d, IS_START, IS_END)
            ledger.add(fam, spec["expr"], p, "IS 2015-2022", m)
            x = slice_(d, IS_START, IS_END)["pnl"].to_numpy()
            all_is_sr_daily.append(x.mean() / x.std(ddof=1))
    all_is_sr_daily = np.array(all_is_sr_daily)
    n_trials = len(all_is_sr_daily)

    dates = slice_(panel.select("date").unique(), IS_START, IS_END)["date"].to_list()
    folds = walk_forward(dates, 2018, 2022, embargo=5)
    summary_rows, chosen, wf_choices, plateau_tables = [], {}, {}, {}
    for fam, spec in FAMILIES.items():
        axes = spec["axes"]
        best, table = plateau_select(res[fam], axes, IS_START, IS_END)
        chosen[fam] = best
        plateau_tables[fam] = table
        d = res[fam][key(best, axes)]
        d_is, d_ho = slice_(d, IS_START, IS_END), slice_(d, HO_START, last)
        # walk-forward: reselect on each training window, stitch the test windows
        pieces, picks = [], []
        for f in folds:
            b, _ = plateau_select(res[fam], axes, f.train_start, f.train_end)
            picks.append({"test_year": f.test_start.year, **b})
            pieces.append(slice_(res[fam][key(b, axes)], f.test_start, f.test_end))
        wf = pl.concat(pieces)
        wf_choices[fam] = picks
        m_is = M.summary(d_is["pnl"].to_numpy(), d_is["traded"].to_numpy())
        m_wf = M.summary(wf["pnl"].to_numpy(), wf["traded"].to_numpy())
        m_ho = M.summary(d_ho["pnl"].to_numpy(), d_ho["traded"].to_numpy())
        sr_is, lo_is, hi_is = S.bootstrap_sharpe_ci(d_is["pnl"].to_numpy(), n_boot=1000, seed=1)
        sr_ho, lo_ho, hi_ho = S.bootstrap_sharpe_ci(d_ho["pnl"].to_numpy(), n_boot=1000, seed=1)
        dsr = S.deflated_sharpe(d_is["pnl"].to_numpy(), all_is_sr_daily)
        raw_best = max(period_metrics(v, IS_START, IS_END)["sharpe"] for v in res[fam].values())
        summary_rows.append(dict(
            family=fam, params=json.dumps(best), is_raw_best_sharpe=raw_best,
            is_sharpe=m_is["sharpe"], is_ci=f"[{lo_is:.2f}, {hi_is:.2f}]", is_fitness=m_is["fitness"],
            is_returns=m_is["returns"], is_turnover=m_is["turnover"], is_drawdown=m_is["drawdown"], is_margin=m_is["margin_bps"],
            is_net_sharpe=net_sharpe(d_is), dsr=dsr,
            wf_sharpe=m_wf["sharpe"], wf_fitness=m_wf["fitness"], wf_net_sharpe=net_sharpe(wf),
            ho_sharpe=m_ho["sharpe"], ho_ci=f"[{lo_ho:.2f}, {hi_ho:.2f}]", ho_fitness=m_ho["fitness"], ho_returns=m_ho["returns"],
            ho_turnover=m_ho["turnover"], ho_drawdown=m_ho["drawdown"], ho_margin=m_ho["margin_bps"], ho_net_sharpe=net_sharpe(d_ho)))
        ledger.add(fam, spec["expr"], best, "WF 2018-2022", m_wf, note="walk-forward, plateau re-selected per fold")
        ledger.add(fam, spec["expr"], best, "HO 2023-", m_ho, note="hold-out, evaluated once")
        d.write_parquet(OUT / "export" / f"daily_{fam}.parquet")

    # combined alpha from the IS-selected configs
    # materialise each family alpha before the cross-sectional rank (nested windows evaluate to null)
    comb = panel.with_columns([FAMILIES[f]["build"](chosen[f]).alias(f"_a_{f}") for f in FAMILIES])
    comb = comb.with_columns([pl.when(pl.col("in_universe")).then(pl.col(f"_a_{f}")).otherwise(None).alias(f"_a_{f}")
                              for f in FAMILIES])
    comb = comb.with_columns([op.rank(f"_a_{f}").alias(f"_r_{f}") for f in FAMILIES])
    comb = comb.with_columns(alpha=pl.mean_horizontal([pl.col(f"_r_{f}") for f in FAMILIES]))
    dc = run(comb, "alpha", SETTINGS).daily
    dc.write_parquet(OUT / "export" / "daily_combined.parquet")
    c_is, c_ho = slice_(dc, IS_START, IS_END), slice_(dc, HO_START, last)
    m_cis = M.summary(c_is["pnl"].to_numpy(), c_is["traded"].to_numpy())
    m_cho = M.summary(c_ho["pnl"].to_numpy(), c_ho["traded"].to_numpy())
    _, lo, hi = S.bootstrap_sharpe_ci(c_ho["pnl"].to_numpy(), n_boot=1000, seed=1)
    _, lo_i, hi_i = S.bootstrap_sharpe_ci(c_is["pnl"].to_numpy(), n_boot=1000, seed=1)
    summary_rows.append(dict(
        family="combined", params="rank-average of the six selections", is_raw_best_sharpe=float("nan"),
        is_sharpe=m_cis["sharpe"], is_ci=f"[{lo_i:.2f}, {hi_i:.2f}]", is_fitness=m_cis["fitness"], is_returns=m_cis["returns"],
        is_turnover=m_cis["turnover"], is_drawdown=m_cis["drawdown"], is_margin=m_cis["margin_bps"], is_net_sharpe=net_sharpe(c_is),
        dsr=S.deflated_sharpe(c_is["pnl"].to_numpy(), all_is_sr_daily),
        wf_sharpe=float("nan"), wf_fitness=float("nan"), wf_net_sharpe=float("nan"),
        ho_sharpe=m_cho["sharpe"], ho_ci=f"[{lo:.2f}, {hi:.2f}]", ho_fitness=m_cho["fitness"], ho_returns=m_cho["returns"],
        ho_turnover=m_cho["turnover"], ho_drawdown=m_cho["drawdown"], ho_margin=m_cho["margin_bps"], ho_net_sharpe=net_sharpe(c_ho)))
    ledger.add("combined", "mean of ranks of the six selected alphas", {"families": list(FAMILIES)}, "IS 2015-2022", m_cis)
    ledger.add("combined", "mean of ranks of the six selected alphas", {"families": list(FAMILIES)}, "HO 2023-", m_cho, note="hold-out, evaluated once")
    ledger.save()

    summary = pl.DataFrame(summary_rows)
    summary.write_parquet(OUT / "summary.parquet")
    yearly_comb = M.yearly(dc)
    write_report(summary, wf_choices, plateau_tables, yearly_comb, n_trials, last)
    figures(res, chosen, dc, plateau_tables)
    export_for_matlab(panel)
    print(f"done in {time.time() - t0:.0f}s -> {OUT}")


def write_report(summary, wf_choices, plateau_tables, yearly_comb, n_trials, last) -> None:
    L = ["# NSE signal study (Phase 5)", "",
         f"Generated by `python -m alphalab.study`. Data to {last}. Trials (configurations): **{n_trials}**.", "",
         "Protocol (fixed in code before the hold-out was computed): see the docstring of `src/alphalab/study.py`.",
         "IS = 2015-2022, walk-forward (WF) = 2018-2022 test years with expanding training and re-selection,",
         "HO = 2023 onward, evaluated once. Metrics are gross of costs (BRAIN-like); `net` = Sharpe after 5 bps one-way cost.", "",
         "## Summary", "",
         "| family | selected params | IS raw-best Sharpe | IS Sharpe [95% CI] | DSR | IS Fitness | IS Turnover | IS Margin bps | IS net | WF Sharpe | WF net | HO Sharpe [95% CI] | HO Fitness | HO Returns | HO Drawdown | HO Margin bps | HO net |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in summary.iter_rows(named=True):
        L.append(f"| {r['family']} | `{r['params']}` | {r['is_raw_best_sharpe']:.2f} | {r['is_sharpe']:.2f} {r['is_ci']} | {r['dsr']:.2f} | "
                 f"{r['is_fitness']:.2f} | {r['is_turnover']:.1%} | {r['is_margin']:.1f} | {r['is_net_sharpe']:.2f} | "
                 f"{r['wf_sharpe']:.2f} | {r['wf_net_sharpe']:.2f} | {r['ho_sharpe']:.2f} {r['ho_ci']} | {r['ho_fitness']:.2f} | "
                 f"{r['ho_returns']:.1%} | {r['ho_drawdown']:.1%} | {r['ho_margin']:.1f} | {r['ho_net_sharpe']:.2f} |")
    L += ["", "## Walk-forward parameter choices (stability of the selection)", ""]
    for fam, picks in wf_choices.items():
        L.append(f"- **{fam}**: " + "; ".join(f"{p.pop('test_year')}: {json.dumps(p)}" for p in [dict(x) for x in picks]))
    L += ["", "## IS plateau tables (score = worst Sharpe over the point and its grid neighbours)", ""]
    for fam, t in plateau_tables.items():
        L.append(f"### {fam}")
        L.append("")
        cols = t.columns
        L.append("| " + " | ".join(cols) + " |")
        L.append("|" + "---|" * len(cols))
        for r in t.iter_rows():
            L.append("| " + " | ".join(f"{v:.2f}" if isinstance(v, float) else str(v) for v in r) + " |")
        L.append("")
    L += ["## Combined alpha by year (gross)", "", "| year | Sharpe | Fitness | Returns | Turnover | Drawdown | Margin bps |", "|---|---|---|---|---|---|---|"]
    for r in yearly_comb.iter_rows(named=True):
        L.append(f"| {r['year']} | {r['sharpe']:.2f} | {r['fitness']:.2f} | {r['returns']:.1%} | {r['turnover']:.1%} | {r['drawdown']:.1%} | {r['margin_bps']:.1f} |")
    L += ["", "Figures: `figures/`. Full trial list: `ledger.md`."]
    (OUT / "STUDY.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def figures(res, chosen, dc, plateau_tables) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    for fam, spec in FAMILIES.items():
        d = res[fam][key(chosen[fam], spec["axes"])].filter(pl.col("date") >= IS_START)
        ax.plot(d["date"], np.cumsum(d["pnl"].to_numpy()) / 0.5, label=fam, lw=1)
    d = dc.filter(pl.col("date") >= IS_START)
    ax.plot(d["date"], np.cumsum(d["pnl"].to_numpy()) / 0.5, label="combined", lw=2, color="black")
    ax.axvline(HO_START, color="grey", ls="--")
    ax.text(HO_START, ax.get_ylim()[1] * 0.95, " hold-out", color="grey")
    ax.set_ylabel("cumulative PnL / half book (gross)")
    ax.set_title("IS-selected configurations, NSE top-100, sector neutral, delay 1")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "equity_curves.png", dpi=130)
    plt.close(fig)

    t = plateau_tables["ema_cross"]
    fasts, slows = FAMILIES["ema_cross"]["axes"]["fast"], FAMILIES["ema_cross"]["axes"]["slow"]
    Z = np.array([[t.filter((pl.col("fast") == f) & (pl.col("slow") == s))["sharpe"][0] for s in slows] for f in fasts])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(Z, cmap="RdBu", vmin=-abs(Z).max(), vmax=abs(Z).max())
    ax.set_xticks(range(len(slows)), slows)
    ax.set_yticks(range(len(fasts)), fasts)
    ax.set_xlabel("slow span")
    ax.set_ylabel("fast span")
    for i in range(len(fasts)):
        for j in range(len(slows)):
            ax.text(j, i, f"{Z[i, j]:.2f}", ha="center", va="center", fontsize=9)
    ax.set_title("EMA cross: IS Sharpe surface")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "ema_surface.png", dpi=130)
    plt.close(fig)


def export_for_matlab(panel: pl.DataFrame) -> None:
    """Wide matrices for the MATLAB parfor sweep (dates x symbols)."""
    p = panel.filter(pl.col("date") >= date(2014, 1, 1))
    for col in ["close", "in_universe"]:
        w = p.pivot(on="symbol", index="date", values=col, aggregate_function="first").sort("date")
        if col == "in_universe":
            w = w.with_columns(pl.exclude("date").cast(pl.Float64).fill_null(0.0))
        w.write_parquet(OUT / "export" / f"wide_{col}.parquet")


if __name__ == "__main__":
    main()
