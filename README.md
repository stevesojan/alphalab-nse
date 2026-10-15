# AlphaLab-NSE

A reproducible quant research stack built around three questions a quant analyst at a market-making / systematic
trading firm has to answer:

1. **Does a signal really work, or did the backtest fool me?** — BRAIN-style alpha research on 200 NSE stocks with
   walk-forward validation, a one-shot hold-out, deflated Sharpe ratios and a ledger of every trial.
2. **What do technical signals (EMA, RSI, breakout, support/resistance) do inside a limit order book?** — a
   price-time-priority matching engine and market-making simulator in Rust, verified bit-for-bit against a Python oracle.
3. **Are the numbers right?** — every layer is cross-checked: three engine implementations, MATLAB vs Python metrics,
   105 automated tests.

Stack: Python 3.13, Polars, NumPy/SciPy, Rust (PyO3 + maturin), numba (cross-check), MATLAB R2026b. Data: NSE daily
bars (Yahoo Finance), Nifty 200 constituents and industries (niftyindices.com).

## Headline results

**Alpha research (NSE top-100 by liquidity, sector-neutral, delay 1; IS 2015-2022, hold-out 2023 onward, evaluated once).**
Details: [reports/study/STUDY.md](reports/study/STUDY.md).

| signal family (plateau-selected) | IS Sharpe [95% CI] | deflated Sharpe | walk-forward Sharpe | hold-out Sharpe [95% CI] | hold-out, net of 5 bp |
|---|---|---|---|---|---|
| RSI(2) reversion | 1.90 [1.08, 2.89] | 0.15 | 1.42 | -0.03 [-1.22, 1.12] | -1.48 |
| S/R channel reversion (10d) | 1.53 [0.84, 2.34] | 0.02 | 1.22 | -0.30 [-1.42, 0.78] | -1.24 |
| Swing-pivot S/R (k=3) | 1.00 [0.22, 1.76] | 0.00 | 1.08 | -0.15 [-1.26, 0.89] | -0.78 |
| 1-day reversal | 1.29 [0.63, 2.15] | 0.00 | 0.93 | 0.23 [-1.00, 1.44] | -1.33 |
| EMA 20/200 trend | -0.03 [-0.66, 0.61] | 0.00 | 0.03 | 1.21 [0.03, 2.39] | 1.12 |
| Volume-confirmed breakout | -0.64 [-1.29, -0.02] | 0.00 | -0.69 | 0.71 [-0.24, 1.74] | 0.31 |
| Rank-average of all six | 1.89 [1.14, 2.62] | 0.12 | — | 0.82 [-0.38, 1.96] | -0.41 |

- Every mean-reversion family looked strong in sample and failed out of sample; the trend families did the reverse.
- After deflating for 44 trials, **no configuration is statistically significant** (best DSR 0.15).
- Net of a modest 5 bp one-way cost, the high-turnover reversion signals lose money even in sample.
- This is the same lesson as the WorldQuant BRAIN exercise that started the project (in-sample Sharpe 1.74 → hold-out
  -0.44): see [reports/CASE_STUDY.md](reports/CASE_STUDY.md).

**Market-making simulator** (details and caveats: [reports/mm/MM_RESULTS.md](reports/mm/MM_RESULTS.md)).
An Avellaneda-Stoikov maker with EMA / RSI / breakout / support-resistance / order-book-imbalance overlays, tuned on
training seeds and compared **paired, on identical order flow** over 10 fresh seeds:

| overlay on the tuned A-S maker | low toxicity: dSharpe (t) | high toxicity: dSharpe (t) |
|---|---|---|
| EMA trend skew | +0.0 (identical to A-S: at the chosen weight the term never moved a quote) | -48.7 (-10.5) |
| RSI reversion skew | +1.1 (0.6) | +39.7 (10.2) |
| Breakout skew / side-pull | -12.1 (-4.1) | +4.5 (1.0) |
| Support/resistance skew | -25.6 (-11.8) | +13.4 (3.2) |
| Imbalance / microprice | -9.6 (-4.0) | +7.2 (3.2) |
| All overlays combined | -35.6 (-12.9) | -211.0 (-49.8) |

- Overlays matter more when flow is toxic; stacking individually tuned overlays **hurts** badly. A likely cause (not
  verified) is that several overlays encode the same short-term mean reversion, so their skews add up and over-skew
  the quotes; each weight was tuned alone, never jointly.
- The maker's edge is ~0.3-0.6 bp of traded value (its Margin = break-even fee). That is below plausible NSE cash-equity
  transaction costs, so every configuration is unprofitable at 1 bp of cost.
- Absolute Sharpe ratios (hundreds) are **not credible**: the simulator has no fees, no latency, no competing makers,
  and its noise liquidity shields the maker from informed flow. The report documents three iterations of the flow
  model and the diagnosis of each (noise-only artifact → latent value + informed traders → value jumps).

**Engineering.**

| check | result |
|---|---|
| Matching engine: Rust vs numba vs pure-Python oracle | bit-identical trade-tape hashes up to 1e7 steps; Rust chosen by a pre-registered benchmark (1.63x numba on identical models; [bench/RESULTS.md](bench/RESULTS.md)) |
| Simulator features (Rust vs oracle) | bit-identical fills, queue positions, markouts and daily snapshots for 11 feature sets |
| Six metrics: MATLAB vs Python | max difference 1.8e-14; EMA sweep parity 8e-16 ([reports/matlab/MATLAB_RESULTS.md](reports/matlab/MATLAB_RESULTS.md)) |
| BRAIN metric definitions | Fitness and Margin reproduce the WorldQuant platform's printed numbers for all 7 logged runs |
| Test suite | 105 tests (signals no-look-ahead, pipeline timing with "cheating" alphas, truncation, DSR on an overfit toy, order-book priority rules, ...) |

## What is in here

```
alphalab/
  src/alphalab/
    data/          nse_loader.py (Nifty 200 list, Yahoo download, holiday-bar removal, adjustment, QA), universe.py (liquidity universe, no look-ahead)
    signals/       indicators.py (EMA, Wilder RSI, ATR, support/resistance, confirmed swing pivots, volume-confirmed breakout)
                   operators.py (BRAIN-style ts_* and cross-sectional operators as Polars expressions)
    backtest/      pipeline.py (universe -> decay -> sector neutralization -> gross 1 -> exact truncation -> delay -> PnL), metrics.py (the six metrics)
    validation/    splits.py (walk-forward + embargo), sweep.py (plateau selection), stats.py (PSR, DSR, stationary bootstrap), ledger.py
    study.py       the NSE research study (protocol in its docstring)
    lob/           spec.py (model contract), engine_ref.py (Python oracle), engine_numba.py (cross-check), sim.py, mm_metrics.py,
                   calibrate.py (price scale from NSE volatility), ablation.py (the market-making study)
    matlab_parity.py  runs MATLAB and compares with Python
  lob_rs/          Rust crate: order book, flow, market maker (PyO3 module `lob_rs`)
  matlab/          crosscheck_metrics.m, bootstrap_sharpe.m, regime_analysis.m, param_surface.m, parfor_sweep.m, run_all.m
  bench/           engine benchmark harness and decision record
  tests/           105 tests
  reports/         generated results (study, mm, matlab) and the case study
  run.ps1          one-command reproduction
```

## How the pieces fit: technical signals in a limit order book

The exchange matches by **price-time priority**: best price first, then earliest order at that price. That rule is the
same for everyone, so it is not an edge. Signals change the market maker's *decisions*, which the rule turns into fills:

- **where to quote** (join the best price at the back of the queue, improve by a tick to jump it, or step back),
- **whether a fill is good** (a passive fill is adversely selected: it tends to happen just before the price moves against you),
- **inventory** (skew quotes to shed risk or lean into a predicted move).

In the simulator the reservation price is `mid + a_ema*EMA_gap + a_rsi*(50-RSI)/50 + a_brk*breakout_state + a_sr*(0.5 - channel_position) + a_imb*microprice_offset - gH*inventory`,
the half-spread widens with breakout state or near support/resistance, and a strong breakout can pull the quote on
the side being run over. Kept quotes keep their queue position; changed quotes go to the back of the queue.

## Reproduce

Requirements: Python 3.13, Rust (MSVC toolchain on Windows), optionally MATLAB R2026b.

```powershell
cd alphalab
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .[dev]
.\run.ps1               # data -> build Rust -> tests -> study -> MM ablation -> benchmark -> MATLAB
.\run.ps1 -SkipMatlab -SkipDownload   # faster, uses cached data
```

## Limitations (read before quoting any number)

- **Survivorship bias:** the universe is today's Nifty 200 membership applied to history.
- **Yahoo data:** adjusted for splits and dividends by Yahoo; four exchange-holiday dates with stale zero-volume bars
  were removed; a few hundred symbol-level zero-volume days remain.
- **Costs:** BRAIN-style metrics are gross; net figures use a flat one-way cost (5 bp in the study, 0.1-3 bp grid in
  the simulator). Indian cash-equity costs (STT, exchange, stamp duty, GST, SEBI fees) vary by product and side.
- **Simulator realism:** synthetic flow, one market maker, zero latency, no fees, no hidden orders. Its noise
  liquidity providers never cancel stale quotes, which shields the maker from informed traders. It is a correct and
  fast engine for studying mechanisms, not a calibrated model of NSE microstructure; real tick data can be replayed
  through the tape interface (`lob_rs.run_tape`).
- **Regime analysis** in MATLAB labels days with a mixture fitted on the full sample: descriptive, not tradable.
