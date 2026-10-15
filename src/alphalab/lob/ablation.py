"""Phase 6e: signal ablation for the market maker (run: python -m alphalab.lob.ablation).

PROTOCOL -- fixed in code before any test-seed result was computed:
  * Price scale calibrated to NSE (calibrate.py): 400,000 steps per session, p_eff from the median
    daily volatility of the 20 most liquid NSE names since 2023.
  * Flow: noise traders (Poisson limits, Hawkes market orders) PLUS a latent value V (random walk, 1 tick
    with prob. 0.01 per step), noise limit orders anchored on V, and informed traders (5 units) who buy when
    V is above the best ask and sell when it is below the best bid. Two toxicity regimes: informed traders
    act with prob. 0.02 ("low") or 0.10 ("high") per step. A first run without V / informed traders produced
    Sharpe ratios of several hundred for makers that fade moves -- a simulator artifact (no adverse
    selection); it is archived in reports/mm/v1_noise_only_flow/.
  * TRAIN: seeds 1-4, 125 sessions each. TEST: seeds 101-110, 250 sessions each, evaluated once.
  * Step 1 (per regime): Avellaneda-Stoikov inventory aversion gH and half-spread hs chosen jointly on TRAIN
    by the plateau rule (maximin of mean Sharpe over a grid point and its neighbours).
  * Step 2: each signal family (EMA, RSI, breakout, support/resistance, imbalance/microprice) is added to
    the chosen A-S maker on its own; its weights are chosen on TRAIN by the same plateau rule.
  * Step 3: "all" = A-S + every family's chosen weights at once (no further tuning).
  * TEST compares every strategy with the A-S baseline seed by seed (same seed = same random arrivals, so
    differences are paired) and reports mean, std and a paired t-statistic, gross and net of fees.
  * Adverse-selection probe: the low-regime A-S maker on a grid of value-jump size J (1, 4, 8, 16 ticks, with
    the jump probability scaled by 1/J^2 so volatility is unchanged) x informed probability (0, 0.05, 0.2),
    test seeds 101-102, 125 sessions.
No exchange fees, latency or competing market makers are simulated; absolute Sharpe ratios are therefore
idealised. The paired differences, the fee break-even (= Margin) and the probes are the results.
Known limitation (diagnosed with the probe): noise limit orders never cancel when V jumps, so they sit stale
in front of the maker and absorb most informed flow -- the maker is shielded from adverse selection.
"""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import polars as pl

from ..validation.sweep import grid, neighbours
from . import calibrate as C
from . import mm_metrics as MM
from . import sim
from .spec import Params

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "reports" / "mm"
DAY = 400_000
TRAIN_SEEDS, TRAIN_SESSIONS = (1, 2, 3, 4), 125
TEST_SEEDS, TEST_SESSIONS = tuple(range(101, 111)), 250
FLOW = dict(day_steps=DAY, markout_h=500, p_value=0.01, anchor_value=1, qty_informed=5)
REGIMES = {"low_toxicity": 0.02, "high_toxicity": 0.10}
JUMPS = (1, 4, 8, 16)
PROBE_P_INF = (0.0, 0.05, 0.2)

AS_AXES = {"gH": [0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6], "hs": [0.5, 1.0, 2.0, 3.0, 5.0]}
FAMILIES = {
    "ema": {"a_sig": [-4.0, -2.0, -1.0, 1.0, 2.0, 4.0]},
    "rsi": {"a_rsi": [-4.0, -2.0, -1.0, 1.0, 2.0, 4.0]},
    "breakout": {"a_brk": [-2.0, -1.0, 0.0, 1.0, 2.0], "brk_pull": [0.0, 1.5, 3.0]},
    "support_resistance": {"a_sr": [-4.0, -2.0, 2.0, 4.0], "hs_lvl": [0.0, 1.0]},
    "imbalance": {"a_imb": [0.25, 0.5, 1.0, 2.0]},
}
FIXED = {"breakout": dict(vol_mult=1.5), "support_resistance": dict(lvl_ticks=3)}
METRICS = ["sharpe", "fitness", "returns", "turnover", "drawdown", "margin_bps"]
DIAG = ["fills_per_session", "markout_ticks_per_unit", "adverse_fill_share", "mean_queue_ahead",
        "mean_time_to_fill", "mean_abs_inventory"]
NET = [f"net_sharpe_{f}bp" for f in MM.FEES_BP]


def _run(job: tuple[dict, float]) -> dict:
    kw, p_eff = job
    r = sim.run(Params(**kw))
    m = MM.evaluate(r, p_eff)
    return {**m, "equity": (r.daily["equity2"].to_numpy() / 2.0).tolist()}


def run_many(kws: list[dict], p_eff: float, pool) -> list[dict]:
    return list(pool.map(_run, [(k, p_eff) for k in kws], chunksize=1))


def plateau_choice(scores: dict[tuple, float], axes: dict) -> tuple[dict, list[dict]]:
    keys = list(axes)
    rows = []
    for k, v in scores.items():
        p = dict(zip(keys, k))
        nb = [scores[tuple(n[x] for x in keys)] for n in neighbours(p, axes)]
        rows.append({**p, "train_sharpe": v, "plateau": min([v] + nb), "nb_mean": float(np.mean([v] + nb))})
    rows.sort(key=lambda r: (r["plateau"], r["nb_mean"]), reverse=True)
    return {k: rows[0][k] for k in keys}, rows


def tune(name: str, axes: dict, base_kw: dict, p_eff: float, pool) -> tuple[dict, list[dict]]:
    pts = grid(**axes)
    jobs = [{**base_kw, **FIXED.get(name, {}), **p, "seed": s, "n_steps": TRAIN_SESSIONS * DAY}
            for p in pts for s in TRAIN_SEEDS]
    out = run_many(jobs, p_eff, pool)
    scores = {}
    for i, p in enumerate(pts):
        chunk = out[i * len(TRAIN_SEEDS):(i + 1) * len(TRAIN_SEEDS)]
        scores[tuple(p[k] for k in axes)] = float(np.mean([o["sharpe"] for o in chunk]))
    return plateau_choice(scores, axes)


def run_regime(name: str, p_inf: float, p_eff: float, pool, t0: float) -> dict:
    base = {**FLOW, "p_informed": p_inf}
    as_best, as_rows = tune("as", AS_AXES, base, p_eff, pool)
    as_kw = {**base, **as_best}
    print(f"[{name}] A-S {as_best}  ({time.time() - t0:.0f}s)", flush=True)
    chosen, tables = {}, {"as (gH x hs)": as_rows}
    for fam, axes in FAMILIES.items():
        best, rows = tune(fam, axes, as_kw, p_eff, pool)
        chosen[fam] = {**FIXED.get(fam, {}), **best}
        tables[fam] = rows
        print(f"[{name}] {fam}: {chosen[fam]}  ({time.time() - t0:.0f}s)", flush=True)
    strategies = {"naive": {**base, "gH": 0.0, "hs": 1.0}, "as": as_kw}
    for fam in FAMILIES:
        strategies[fam] = {**as_kw, **chosen[fam]}
    all_kw = dict(as_kw)
    for fam in FAMILIES:
        all_kw.update(chosen[fam])
    strategies["all"] = all_kw
    jobs, keys = [], []
    for sname, kw in strategies.items():
        for s in TEST_SEEDS:
            jobs.append({**kw, "seed": s, "n_steps": TEST_SESSIONS * DAY})
            keys.append((sname, s))
    test = run_many(jobs, p_eff, pool)
    df = pl.DataFrame([{"regime": name, "strategy": k[0], "seed": k[1], **{m: v for m, v in o.items() if m != "equity"}}
                       for k, o in zip(keys, test)])
    equities = {k[0]: o["equity"] for k, o in zip(keys, test) if k[1] == TEST_SEEDS[0]}
    return dict(df=df, strategies=strategies, tables=tables, equities=equities, base=base, as_kw=as_kw)


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)
    cal = C.calibrate(Params(**FLOW, p_informed=REGIMES["low_toxicity"]))
    p_eff = cal["p_eff"]
    print(f"calibration: sigma_session={cal['sigma_ticks']:.1f} ticks, NSE sigma={cal['nse_sigma_daily']:.4f}, "
          f"p_eff={p_eff:.0f} ticks, tick={cal['tick_bp']:.2f} bp", flush=True)
    workers = max(1, (os.cpu_count() or 2) - 1)
    results = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for name, p_inf in REGIMES.items():
            results[name] = run_regime(name, p_inf, p_eff, pool, t0)
        as_low = results["low_toxicity"]["as_kw"]
        jobs = [{**as_low, "p_informed": p, "value_jump": J, "p_value": FLOW["p_value"] / J ** 2, "seed": s,
                 "n_steps": (TEST_SESSIONS // 2) * DAY}
                for J in JUMPS for p in PROBE_P_INF for s in TEST_SEEDS[:2]]
        tox = run_many(jobs, p_eff, pool)
    tox_df = pl.DataFrame([{"value_jump": j["value_jump"], "p_informed": j["p_informed"], "seed": j["seed"],
                            **{m: v for m, v in o.items() if m != "equity"}} for j, o in zip(jobs, tox)])
    all_df = pl.concat([r["df"] for r in results.values()])
    all_df.write_parquet(OUT / "test_results.parquet")
    tox_df.write_parquet(OUT / "toxicity_sweep.parquet")
    (OUT / "calibration.json").write_text(json.dumps(cal, indent=1, default=str))
    write_report(results, tox_df, cal, time.time() - t0)
    figures(results, tox_df, p_eff)
    print(f"done in {time.time() - t0:.0f}s -> {OUT}")


def paired(df: pl.DataFrame, name: str, metric: str = "sharpe") -> tuple[float, float]:
    a = df.filter(pl.col("strategy") == name).sort("seed")[metric].to_numpy()
    b = df.filter(pl.col("strategy") == "as").sort("seed")[metric].to_numpy()
    d = a - b
    sd = d.std(ddof=1)
    return float(d.mean()), (float(d.mean() / (sd / np.sqrt(len(d)))) if sd > 0 else float("nan"))


def write_report(results, tox_df, cal, secs) -> None:
    L = ["# Market-making simulator: signal ablation (Phase 6e)", "",
         "Generated by `python -m alphalab.lob.ablation`. Protocol: module docstring of `src/alphalab/lob/ablation.py`.", "",
         "**Read this first.** The simulator has no exchange fees, no latency and no competing market makers, so absolute",
         "Sharpe ratios are idealised (a single zero-latency maker harvesting noise flow earns very steady small profits).",
         "The meaningful outputs are (1) paired differences between strategies on identical order flow, (2) the Margin,",
         "which is the break-even one-way fee in bp, and (3) how profitability falls as the informed share of flow rises.", "",
         "## Calibration", "",
         f"- NSE target: median daily log-return volatility of the 20 most liquid names since 2023 = **{cal['nse_sigma_daily']:.2%}**.",
         f"- Session = {DAY:,} steps; mid-price std per session = **{cal['sigma_ticks']:.1f} ticks** (low-toxicity flow).",
         f"- One unit is worth p_eff = {cal['p_eff']:.0f} ticks, so one tick = **{cal['tick_bp']:.2f} bp** of price.",
         f"- Capital G = inv_max x p_eff; Returns / Turnover / Drawdown are relative to G. Ablation runtime {secs:.0f} s.", ""]
    for name, r in results.items():
        df = r["df"]
        agg = df.group_by("strategy", maintain_order=True).agg(
            [pl.col(m).mean().alias(m) for m in METRICS + DIAG + NET] + [pl.col("sharpe").std().alias("sharpe_sd")])
        L += [f"## Test results: {name} (informed traders act with prob. {r['base']['p_informed']} per step)", "",
              "| strategy | params (on top of flow) | Sharpe (sd) | paired dSharpe vs A-S | t | Fitness | Returns | Turnover (x G/session) | Drawdown | Margin bps (= break-even fee) | net Sharpe @0.3bp | @1bp | @3bp |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for row in agg.iter_rows(named=True):
            s = row["strategy"]
            params = {k: v for k, v in r["strategies"][s].items() if k not in r["base"]}
            dm, tt = paired(df, s) if s != "as" else (0.0, float("nan"))
            L.append(f"| {s} | `{json.dumps(params)}` | {row['sharpe']:.1f} ({row['sharpe_sd']:.1f}) | {dm:+.1f} | {tt:.1f} | "
                     f"{row['fitness']:.2f} | {row['returns']:.0%} | {row['turnover']:.0f} | {row['drawdown']:.1%} | "
                     f"{row['margin_bps']:.3f} | {row['net_sharpe_0.3bp']:.1f} | {row['net_sharpe_1.0bp']:.1f} | {row['net_sharpe_3.0bp']:.1f} |")
        L += ["", "Market-making diagnostics (means over test seeds):", "",
              "| strategy | fills/session | markout ticks/unit (500 steps) | adverse fill share | queue ahead at post | steps to fill | mean abs inventory |",
              "|---|---|---|---|---|---|---|"]
        for row in agg.iter_rows(named=True):
            L.append(f"| {row['strategy']} | {row['fills_per_session']:.0f} | {row['markout_ticks_per_unit']:.3f} | "
                     f"{row['adverse_fill_share']:.1%} | {row['mean_queue_ahead']:.1f} | {row['mean_time_to_fill']:.0f} | {row['mean_abs_inventory']:.1f} |")
        L.append("")
    tg = tox_df.group_by("value_jump", "p_informed").agg(
        pl.col("sharpe").mean(), pl.col("margin_bps").mean(), pl.col("markout_ticks_per_unit").mean(),
        pl.col("adverse_fill_share").mean(), pl.col("net_sharpe_1.0bp").mean()).sort("value_jump", "p_informed")
    L += ["## Adverse-selection probe (low-regime A-S maker, test seeds 101-102, 125 sessions)", "",
          "Value-jump size J with jump probability scaled by 1/J^2 (same volatility) x informed-trader probability.", "",
          "| J (ticks) | informed prob. | Sharpe | Margin bps | markout ticks/unit | adverse fill share | net Sharpe @1bp |",
          "|---|---|---|---|---|---|---|"]
    for row in tg.iter_rows(named=True):
        L.append(f"| {row['value_jump']} | {row['p_informed']} | {row['sharpe']:.1f} | {row['margin_bps']:.3f} | "
                 f"{row['markout_ticks_per_unit']:.3f} | {row['adverse_fill_share']:.1%} | {row['net_sharpe_1.0bp']:.1f} |")
    L += ["", "Reading: larger jumps lower the maker's Sharpe only moderately and its margin barely moves, because the noise",
          "limit orders never cancel on a jump and absorb most of the informed flow. Making the maker bear realistic adverse",
          "selection needs value-aware cancellation by other liquidity providers or competing makers (future work)."]
    L += ["", "## Training choices (plateau rule on mean train Sharpe)", ""]
    for name, r in results.items():
        for fam, rows in r["tables"].items():
            L += [f"### {name}: {fam}", ""]
            cols = list(rows[0].keys())
            L.append("| " + " | ".join(cols) + " |")
            L.append("|" + "---|" * len(cols))
            for row in rows[:12]:
                L.append("| " + " | ".join(f"{row[c]:.2f}" if isinstance(row[c], float) else str(row[c]) for c in cols) + " |")
            if len(rows) > 12:
                L.append(f"| ... {len(rows) - 12} more rows | |")
            L.append("")
    L += ["Figures: `figures/`. Raw per-seed results: `test_results.parquet`, `toxicity_sweep.parquet`."]
    (OUT / "MM_RESULTS.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def figures(results, tox_df, p_eff) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(results), figsize=(12, 4), sharey=False)
    for ax, (name, r) in zip(np.atleast_1d(axes), results.items()):
        agg = r["df"].group_by("strategy", maintain_order=True).agg(pl.col("sharpe").mean().alias("m"), pl.col("sharpe").std().alias("s"))
        ax.bar(agg["strategy"], agg["m"], yerr=agg["s"], capsize=3, color="steelblue")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_title(name)
        ax.tick_params(axis="x", rotation=60)
    np.atleast_1d(axes)[0].set_ylabel("Sharpe, mean +/- sd over 10 test seeds (gross, idealised)")
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "mm_sharpe_by_strategy.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    for p in PROBE_P_INF:
        tg = tox_df.filter(pl.col("p_informed") == p).group_by("value_jump").agg(pl.col("sharpe").mean()).sort("value_jump")
        ax.plot(tg["value_jump"], tg["sharpe"], marker="o", label=f"informed prob. {p}")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("value jump size J (ticks), volatility held fixed")
    ax.set_ylabel("Sharpe (gross, idealised)")
    ax.set_title("A-S maker vs information arrival")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "mm_toxicity.png", dpi=130)
    plt.close(fig)

    r = results["high_toxicity"]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for s, eq in r["equities"].items():
        ax.plot(np.array(eq) / (50 * p_eff), label=s, lw=2 if s in ("as", "all") else 1)
    ax.set_xlabel("session")
    ax.set_ylabel("cumulative PnL / capital G (gross)")
    ax.set_title("High-toxicity regime, test seed 101")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "mm_equity_high_toxicity_seed101.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
