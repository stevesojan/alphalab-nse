# Case study: when a 1.74 Sharpe becomes -0.44

## 1. The WorldQuant BRAIN episode (October 2026)

Starting from the platform's example alpha `-ts_rank(fn_liab_fair_val_l1_a, 252)` (short companies whose fair value of
Level-1 liabilities is high versus its own past year), a window sweep and a handful of variants were run on US TOP3000,
2019-2022 in-sample, with 2023 hidden (full log: [worldquant_alpha_log.md](worldquant_alpha_log.md)).

| step | in-sample Sharpe | note |
|---|---|---|
| example, window 252 | 0.95 | |
| window sweep 63 / 90 / 105 / 111 / 120 / 180 / 252 | 0.89 / 1.42 / 1.64 / 1.69 / 1.73 / 1.45 / 0.95 | a smooth inverted U: a real plateau around 105-120 |
| divide by total assets | 0.23 | normalisation destroyed the signal |
| group_rank within subindustry | **1.74** | best in-sample, passed 7 of 8 submission checks |
| + ts_decay_linear(5) | 1.63 | smoothing cost more edge than turnover saved |
| quarterly field instead of annual | -0.31 | |
| **hold-out year, group-rank version** | **-0.47** (test-window aggregate -0.44) | |
| hold-out year, plain window-111 version | 0.04 | |

Nine variants were tried on the same four years. The plateau was real, but the in-sample Sharpe was the maximum over
the variants, Sharpe had been decaying every year since 2019, and the field was used by 23,589 other alphas.

## 2. Turning the lesson into machinery

| what went wrong on BRAIN | what AlphaLab does about it |
|---|---|
| picked the single best point | `sweep.plateau_select`: maximin Sharpe over a point and its grid neighbours |
| no count of how many things were tried | `validation.ledger`: every configuration recorded; the count feeds the deflated Sharpe |
| one in-sample number | walk-forward with re-selection per fold and a 5-day embargo, plus bootstrap confidence intervals |
| peeked at the hidden year while still choosing | the study's protocol is fixed in code; the hold-out is computed once, after selection |
| gross numbers only | net Sharpe at 5 bp one-way cost alongside every gross figure |
| metric definitions taken on trust | Fitness and Margin formulas re-derived and tested against all 7 logged BRAIN runs; MATLAB re-implementation agrees to 1e-14 |

## 3. The same story on NSE

Six technical signal families (EMA trend, RSI reversion, volume-confirmed breakout, support/resistance channel,
swing pivots, short-term reversal), 44 configurations, top-100 NSE stocks, sector-neutral, delay 1:

- **Mean reversion** (RSI(2) IS Sharpe 1.90, S/R channel 1.53, reversal 1.29) held up in the 2018-2022 walk-forward
  but fell to roughly zero in the 2023-2026 hold-out.
- **Trend following** (EMA 20/200) was flat in sample (-0.03) and worked in the hold-out (1.21) — a regime change, not
  something the in-sample data could have selected.
- **Deflated Sharpe** — the probability that the best result beats what 44 lucky tries would produce — was at most 0.15.
  The study would have told us, before the hold-out, that nothing here was proven.
- **Costs** turn the high-turnover reversion signals negative even in sample at 5 bp.

## 4. What I would do next

- Treat walk-forward stability (the same parameters chosen in every fold, as happened for RSI and S/R) as necessary
  but not sufficient: the regime shift after 2022 broke signals that were stable for five years.
- Combine signals with an explicit regime model and size by out-of-sample confidence, not by in-sample Sharpe.
- For market making, the simulator shows the edge per unit traded is a fraction of a basis point: the work is in
  costs, latency and adverse selection, which need real tick data to study credibly.
