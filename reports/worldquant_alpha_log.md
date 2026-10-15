# Alpha Experiment Log: `fn_liab_fair_val_l1_a` (fair value of liabilities)

Running log of simulations for one idea. New results are appended to the matching section below; add a row, never overwrite one.

## 1. Idea

- **Hypothesis:** a rise in the fair value of a company's liabilities signals a higher future cost and weaker financial health, so sell stocks whose liabilities are unusually high versus their own history.
- **Starting alpha (platform "Example"):** `-ts_rank(fn_liab_fair_val_l1_a, 252)`
- **Data field:** `fn_liab_fair_val_l1_a` (fundamental, updates at filings, forward-filled daily).
- **Direction:** negative sign = short high-liability stocks, long low-liability stocks (after neutralization).

## 2. Simulation settings (unchanged across all runs unless noted)

| Setting | Value |
|---|---|
| Language | Fast Expression |
| Instrument type | Equity |
| Region / Universe | USA / TOP3000 |
| Delay | 1 |
| Neutralization | Subindustry |
| Decay | 4 |
| Truncation | 0.08 |
| Pasteurization | On |
| Unit handling | Verify |
| NaN handling | Off |
| Test period | 1 year (hidden) |

## 3. Reading the results (conventions)

- Compare runs only in the same view: button reads **"Show test period"** (test year hidden), so in-sample years are 2019 to 2022.
- Submission cutoffs seen on the platform: Sharpe > 1.25, Fitness > 1, turnover between 1% and 70%, sub-universe Sharpe > 0.6, plus a self-correlation check.
- Nothing has been submitted so far.

## 4. Window sweep: `-ts_rank(fn_liab_fair_val_l1_a, N)`

Aggregate in-sample metrics.

| Window N | Sharpe | Fitness | Returns | Turnover | Drawdown | Margin (bps) | Alpha ID | Notes |
|---|---|---|---|---|---|---|---|---|
| 63 | 0.89 | 0.71 | 8.02% | 8.55% | 9.26% | 18.75 | O08j9Xr1 | Noisy; highest turnover |
| 90 | 1.42 | 1.33 | 10.89% | 6.65% | 6.72% | 32.75 | 2rmxma2Z | |
| 105 | 1.64 | 1.58 | 11.58% | 5.99% | 6.80% | 38.65 | LLZw5Nna | |
| **111** | **1.69** | **1.62** | **11.54%** | **5.77%** | **6.67%** | 39.97 | LLZw9Mw2 | **Chosen baseline (middle of plateau)** |
| 120 | 1.73 | 1.63 | 11.13% | 5.45% | 6.82% | 40.82 | MP3nQ8Po | Seen with test year shown; treat as approximate |
| 180 | 1.45 | 1.10 | 7.16% | 4.03% | 6.51% | 35.51 | omWXLZYl | |
| 252 | 0.95 | 0.51 | 3.55% | 2.84% | 5.51% | 25.03 | QPKAEoag | Platform's example alpha; yearly rows cover 2019-2022 |

Excluded: a 110-day run (`blOgj7L6`) showed an aggregate Sharpe of 0.12 that did not match its yearly rows (view had the test year shown); dropped at the user's request and replaced by 111.

### Sharpe by year

| Window N | 2019 | 2020 | 2021 | 2022 |
|---|---|---|---|---|
| 63 | 1.41 | 1.19 | 0.94 | 0.07 |
| 90 | 1.97 | 1.52 | 1.38 | 0.81 |
| 105 | 2.27 | 1.74 | 1.44 | 1.13 |
| 111 | 2.33 | 1.80 | 1.49 | 1.15 |
| 120 | 2.39 | 1.94 | 1.49 | 1.14 |
| 180 | 2.10 | 1.91 | 1.64 | 0.27 |
| 252 | 1.46 | 1.76 | 1.62 | -0.55 |

(The 120 run with the test year shown had a 2023 Sharpe of -0.13, and the 110 run had +0.05.)

### Long / short counts, 111 baseline (average stocks held per side)

| 2019 | 2020 | 2021 | 2022 |
|---|---|---|---|
| 356 / 337 | 400 / 320 | 444 / 309 | 387 / 408 |

### Takeaways

- Sharpe forms an inverted U across window length. The plateau is roughly 105 to 120, where Sharpe is 1.64 to 1.73 and the differences are within noise.
- Short windows (63) give high turnover and a worse drawdown. Long windows (180, 252) give low turnover but weak 2022.
- Every window shows Sharpe decaying from 2019 to 2022. This is not a window problem and is the main weakness to address.

### Test-year (out-of-sample) check, 111-day baseline `-ts_rank(fn_liab_fair_val_l1_a, 111)` (LLZw9Mw2)

| Year | Sharpe | Fitness | Returns | Turnover | Drawdown | Margin (bps) | Long / Short |
|---|---|---|---|---|---|---|---|
| 2023 (test) | 0.04 | 0.01 | 0.24% | 5.84% | 6.53% | 0.81 | 353 / 533 |

Test-view "Aggregate" panel (test window stats): Sharpe 0.10, Fitness 0.02, Returns 0.65%, Drawdown 6.53%. Compare 5.2 group rank: 2023 Sharpe -0.47. The plain baseline is flat out-of-sample, not losing money, but also shows essentially no edge in 2023. Note: choosing the baseline over 5.2 because of its test-year result means the test year has now influenced the choice. Category on the alpha was set to "Fundamental" by the user. Self-correlation check still pending at time of logging.

## 5. Improvement experiments (window fixed at 111)

Baseline for comparison: Sharpe 1.69, Fitness 1.62, Returns 11.54%, Turnover 5.77%, Drawdown 6.67%, 2022 Sharpe 1.15.

Planned order: (1) normalise by assets, (2) group rank within subindustry, (3) `ts_decay_linear` smoothing.

### 5.1 Normalise by total assets

Field used: `assets` ("Assets - Total", Matrix, dataset fundamental6, 100% date coverage).

| Expression | Sharpe | Fitness | Returns | Turnover | Drawdown | Margin (bps) | Alpha ID |
|---|---|---|---|---|---|---|---|
| `-ts_rank(fn_liab_fair_val_l1_a/assets, 111)` | 0.23 | 0.06 | 0.88% | 5.21% | 11.84% | 3.37 | QPKAYwxw |

| Year | Sharpe | Fitness | Returns | Turnover | Drawdown | Long / Short |
|---|---|---|---|---|---|---|
| 2019 | 0.33 | 0.10 | 1.09% | 5.44% | 4.14% | 604 / 687 |
| 2020 | -1.18 | -0.78 | -5.47% | 5.17% | 7.65% | 620 / 690 |
| 2021 | 1.17 | 0.66 | 3.92% | 5.03% | 2.66% | 599 / 666 |
| 2022 | 0.83 | 0.44 | 3.45% | 5.19% | 2.17% | 683 / 697 |

**Outcome:** much worse than the baseline (Sharpe 1.69 down to 0.23, 2020 negative). Badge: "Needs Improvement". Raw liabilities worked better than the leverage ratio. Do not carry this forward as-is.

**Possible reasons (unconfirmed):**
- Dividing by assets may change what `ts_rank` measures. The ratio's trend over time can be driven by assets growing, so the signal drifts.
- The original signal may rely on the raw level of the fair-value field rather than size-adjusted leverage.
- Long/short counts roughly doubled (about 600+ per side against about 350), so many more stocks entered, probably noisier ones.

### 5.2 Group rank within subindustry

| Expression | Sharpe | Fitness | Returns | Turnover | Drawdown | Margin (bps) | Alpha ID |
|---|---|---|---|---|---|---|---|
| `-group_rank(ts_rank(fn_liab_fair_val_l1_a, 111), subindustry)` | **1.74** | **1.61** | 10.64% | 5.71% | 5.72% | 37.26 | QPKAO52K |

| Year | Sharpe | Fitness | Returns | Turnover | Drawdown | Long / Short |
|---|---|---|---|---|---|---|
| 2019 | 2.14 | 2.13 | 12.42% | 6.28% | 5.72% | 344 / 312 |
| 2020 | 1.54 | 1.35 | 9.63% | 5.63% | 3.74% | 356 / 300 |
| 2021 | 2.00 | 1.95 | 11.94% | 5.43% | 2.25% | 422 / 287 |
| 2022 | 1.31 | 1.08 | 8.49% | 5.50% | 4.42% | 366 / 368 |

**Outcome:** slightly better than the 111 baseline (Sharpe 1.69 to 1.74, drawdown 6.67% to 5.72%, 2022 Sharpe 1.15 to 1.31). The steady year-on-year decline is gone: 2021 (2.00) is above 2020 (1.54). Fitness is about the same (1.62 vs 1.61). Badge: "Average". Best version so far.

**Test-year (out-of-sample) result, revealed 2026-10-11:**

| Year | Sharpe | Fitness | Returns | Turnover | Drawdown | Margin (bps) | Long / Short |
|---|---|---|---|---|---|---|---|
| 2023 (test) | **-0.47** | -0.21 | -2.59% | 5.99% | 8.39% | -8.65 | 317 / 501 |

With the test view on, the "Aggregate" panel read Sharpe -0.44, Fitness -0.19, Returns -2.43%, Drawdown 8.39%, which looks like the test window's own stats rather than the 4-year in-sample figures (that would also explain the odd 110-day aggregate of 0.12 seen earlier). In-sample 1.74 versus out-of-sample -0.44 is a large gap. Earlier runs also had 2023 near zero or negative (120-day: -0.13, 110-day: +0.05). The test year is now seen, so it can no longer serve as a clean hold-out for tuning decisions.

**Decision (pending user):** do not submit this version on the current evidence.

### 5.3 `ts_decay_linear` smoothing

Applied on top of 5.2, with the Decay setting left at 4 (so the two smoothing steps stack).

| Expression | Sharpe | Fitness | Returns | Turnover | Drawdown | Margin (bps) | Alpha ID |
|---|---|---|---|---|---|---|---|
| `ts_decay_linear(-group_rank(ts_rank(fn_liab_fair_val_l1_a, 111), subindustry), 5)` | 1.63 | 1.45 | 9.90% | 4.70% | 6.30% | 42.11 | YPMO89pA |

| Year | Sharpe | Fitness | Returns | Turnover | Drawdown | Long / Short |
|---|---|---|---|---|---|---|
| 2019 | 1.97 | 1.88 | 11.38% | 5.21% | 6.30% | 575 / 459 |
| 2020 | 1.59 | 1.39 | 9.51% | 4.61% | 3.51% | 701 / 585 |
| 2021 | 1.89 | 1.80 | 11.30% | 4.51% | 2.47% | 718 / 582 |
| 2022 | 1.13 | 0.87 | 7.37% | 4.47% | 5.24% | 736 / 700 |

**Outcome:** turnover fell as expected (5.71% to 4.70%), but Sharpe (1.74 to 1.63), Fitness (1.61 to 1.45), Returns (10.64% to 9.90%), Drawdown (5.72% to 6.30%) and 2022 Sharpe (1.31 to 1.13) all got worse. Badge: "Needs Improvement". Not worth keeping; 5.2 remains the best version.

Long/short counts jumped (about 350 to about 650 per side). Likely because averaging over 5 days gives non-zero weight to stocks that had a recent non-zero score; unconfirmed.

### 5.4 Normalise by the matching fair-value assets field (with group rank + decay 5)

Field used: `fn_assets_fair_val_l1_a` (same family as the liabilities field, so the same filing schedule and coverage).

| Expression | Sharpe | Fitness | Returns | Turnover | Drawdown | Margin (bps) | Alpha ID |
|---|---|---|---|---|---|---|---|
| `ts_decay_linear(-group_rank(ts_rank(fn_liab_fair_val_l1_a/fn_assets_fair_val_l1_a, 111), subindustry), 5)` | 0.29 | 0.11 | 1.71% | 4.54% | 7.66% | 7.52 | O08jzzV7 |

| Year | Sharpe | Fitness | Returns | Turnover | Drawdown | Long / Short |
|---|---|---|---|---|---|---|
| 2019 | 0.67 | 0.40 | 4.35% | 5.13% | 6.98% | 360 / 285 |
| 2020 | -0.38 | -0.16 | -2.32% | 4.37% | 6.33% | 415 / 434 |
| 2021 | 0.19 | 0.06 | 1.07% | 4.31% | 3.80% | 425 / 417 |
| 2022 | 0.64 | 0.33 | 3.40% | 4.33% | 3.22% | 515 / 434 |

**Outcome:** far worse than 5.2/5.3 (Sharpe 0.29 vs 1.63 to 1.74); 2020 negative again. Badge: "Needs Improvement". Second normalisation attempt that destroyed the signal (see 5.1), even with a matching same-dataset denominator. Note it also carries the decay-5 wrapper, so compare against 5.3 (Sharpe 1.63), not 5.2.

**Reading:** dividing the field by any size-like quantity removes the signal, so the edge probably lies in the raw level or change of fair-value liabilities, not in leverage.

### 5.5 Field variant: quarterly liabilities (`_q`) with group rank, window 111

| Expression | Sharpe | Fitness | Returns | Turnover | Drawdown | Margin (bps) | Alpha ID |
|---|---|---|---|---|---|---|---|
| `-group_rank(ts_rank(fn_liab_fair_val_l1_q, 111), subindustry)` | -0.31 | -0.09 | -1.10% | 5.02% | 9.48% | -4.40 | MP3nZg36 |

| Year | Sharpe | Fitness | Returns | Turnover | Drawdown | Long / Short |
|---|---|---|---|---|---|---|
| 2019 | 0.79 | 0.33 | 2.22% | 4.95% | 2.10% | 585 / 518 |
| 2020 | -0.35 | -0.12 | -1.57% | 5.37% | 6.37% | 610 / 468 |
| 2021 | -1.22 | -0.71 | -4.20% | 4.89% | 4.50% | 575 / 505 |
| 2022 | -0.25 | -0.07 | -0.90% | 4.91% | 5.07% | 511 / 701 |

**Outcome:** negative Sharpe overall (-0.31) and in 3 of 4 years; loses money. Compare 5.2 (same structure, `_a` field): Sharpe 1.74. Badge: "Needs Improvement". The quarterly version of the same item does not carry the signal, so the edge may be specific to the annual field (or to the 111 window, which was tuned on `_a`). Not carried forward.

## Summary of best versions so far (window 111)

| Version | Sharpe | Fitness | Turnover | Drawdown | 2022 Sharpe |
|---|---|---|---|---|---|
| Base `-ts_rank(field, 111)` | 1.69 | 1.62 | 5.77% | 6.67% | 1.15 |
| 5.1 divide by assets | 0.23 | 0.06 | 5.21% | 11.84% | 0.83 |
| **5.2 group_rank (best)** | **1.74** | 1.61 | 5.71% | **5.72%** | **1.31** |
| 5.3 + ts_decay_linear 5 | 1.63 | 1.45 | 4.70% | 6.30% | 1.13 |
| 5.4 fair-value-assets ratio + decay 5 | 0.29 | 0.11 | 4.54% | 7.66% | 0.64 |
| 5.5 `_q` field, group rank | -0.31 | -0.09 | 5.02% | 9.48% | -0.25 |

## 6. Open ideas / next candidates

### Field facts (from Data tab, dataset "Report Footnotes" = fundamental2; USA, Delay 1, TOP3000)

| Field | Description | Instrument coverage | Alphas using it |
|---|---|---|---|
| `fn_liab_fair_val_l1_a` (current) | Liabilities Fair Value, Recurring, Level 1 | 36% | 23,589 (very crowded) |
| `fn_liab_fair_val_l1_q` | same, quarterly version | 36% | 196 |
| `fn_liab_fair_val_l2_a` / `_l2_q` | Level 2 | 40% | 426 / 146 |
| `fn_liab_fair_val_l3_a` / `_l3_q` | Level 3 | 26% | 173 / 474 |
| `fn_liab_fair_val_a` / `_q` | Total recurring | 33% / 32% | 669 / 264 |
| `fn_assets_fair_val_l1_a` | Asset-side equivalent (used in 5.4) | 40% | 320 |
| `assets` (dataset fundamental6) | Assets - Total (used in 5.1) | n/a | 185,648 |

Suffix `_a` = annual and `_q` = quarterly versions of the same item (inferred from the pairing). Coverage of about 36% means roughly a third of the universe has a value at all.

### Candidate list

- Field variants to test with the 5.2 structure `-group_rank(ts_rank(FIELD, 111), subindustry)`: `fn_liab_fair_val_l1_q`, `fn_liab_fair_val_l2_a`, `fn_liab_fair_val_a`, `fn_liab_fair_val_l3_a`. Less-used fields are likely less crowded than the 23,589-alpha `l1_a`.
- Quarterly fields update more often, so the best window may differ; re-sweep if one looks promising.
- Test other normalisations (for example `/ cap`, or a ratio of two liability fields) instead of assets, since both attempts so far (5.1, 5.4) hurt.
- Try `ts_delta` or `ts_rank` on the change in the field rather than its level.
- Confirm the 150-day window as a check on the right side of the plateau.
- Add a second, unrelated signal (momentum or reversal) with its own hypothesis.
- Do not tune against the hidden test year.

## 7. Other ideas (separate from the fair-value liabilities signal)

### 7.1 Short-term price reversal

Hypothesis: stocks that rose sharply over the last week tend to give back part of it, and vice versa. Data: `close` (price/volume, whole-universe coverage). Settings unchanged from section 2 (test year hidden).

| Expression | Sharpe | Fitness | Returns | Turnover | Drawdown | Margin (bps) | Alpha ID |
|---|---|---|---|---|---|---|---|
| `-rank(ts_delta(close, 5))` | 1.42 | 0.78 | 12.20% | 40.16% | 6.76% | 6.08 | kqJxroZP |

| Year | Sharpe | Fitness | Returns | Turnover | Drawdown | Long / Short |
|---|---|---|---|---|---|---|
| 2019 | 2.20 | 1.05 | 8.99% | 39.42% | 3.82% | 1532 / 1505 |
| 2020 | 2.72 | 2.07 | 23.62% | 40.89% | 3.45% | 1539 / 1495 |
| 2021 | 0.99 | 0.51 | 10.47% | 39.94% | 5.48% | 1558 / 1531 |
| 2022 | 0.69 | 0.28 | 6.64% | 40.45% | 6.76% | 1548 / 1541 |

Observations: Sharpe 1.42 but Fitness only 0.78 because turnover is about 40%, with thin margin per trade (6 bps). Holds about 3,000 stocks (1,500 per side), far broader than the fundamentals signal (about 350 per side). Sharpe fades after 2020 (2.72 to 0.69 by 2022). Badge: "Needs Improvement". Test year not yet looked at.

### 7.2 Price-volume divergence, raw 1-day version

Hypothesis: a price move on rising volume is more likely to continue than one on falling volume, so short stocks where price and volume rise together (and buy when price rises on falling volume). First attempt with `delta` failed ("unknown operator"); the platform operator is `ts_delta`. Settings unchanged (test year hidden).

| Expression | Sharpe | Fitness | Returns | Turnover | Drawdown | Margin (bps) | Alpha ID |
|---|---|---|---|---|---|---|---|
| `-(ts_delta(volume, 1))*(ts_delta(close, 1))` | 0.66 | 0.34 | 19.20% | **72.20%** | **44.63%** | 5.32 | pwvwRJZx |

| Year | Sharpe | Fitness | Returns | Turnover | Drawdown | Long / Short |
|---|---|---|---|---|---|---|
| 2019 | 1.47 | 0.76 | 19.95% | 73.92% | 4.05% | 1458 / 1579 |
| 2020 | 1.74 | 1.31 | 40.33% | 71.14% | 10.26% | 1657 / 1376 |
| 2021 | 0.34 | 0.16 | 16.14% | 70.90% | **43.45%** | 1596 / 1493 |
| 2022 | 0.10 | 0.02 | 2.09% | 72.75% | 13.80% | 1537 / 1552 |

Observations: turnover about 72% a day (nearly double the 5-day reversal at 40%), very large drawdown (43% in 2021, return-to-drawdown ratio well under 1), and Sharpe collapsing after 2020. Raw product is dominated by stocks with extreme volume or price swings (no ranking, mixed units). Badge: "Needs Improvement". Next planned: ranked 5-day version `-rank(ts_delta(volume, 5)) * rank(ts_delta(close, 5))`.

Platform tutorial formulas seen on the Results page: Fitness = Sharpe * sqrt(abs(Returns)) / max(Turnover, 0.125); Sharpe = annualised return / annualised std dev of returns (times sqrt(250)); Margin = total PnL / total traded amount, in bps.
