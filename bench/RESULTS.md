# Matching-engine benchmark: numba vs Rust (PyO3) vs pure-Python reference

Run date: 2026-10-13. Machine: the user's Windows 11 laptop (single run family, not a controlled lab; treat ratios as indicative, not exact).
Raw output: `bench/bench_output.txt`, `bench/results.json`. Harness: `bench/bench_engines.py` (each configuration in its own subprocess; best-of-N timing; peak working set via psutil).

## What was benchmarked

The same model implemented three times (spec: `src/alphalab/lob/spec.py`):

* limit-order book with price-time priority (per-level FIFO queues, array-based),
* noise order flow: Bernoulli-per-step Poisson arrivals + discrete-time Hawkes market orders (self-exciting), integer splitmix64 PRNG,
* Avellaneda-Stoikov market maker with an EMA(fast-slow) signal term, requoting every 10 steps.

Because flow and strategy are inside the loop, the "full" scenario measures what the real simulator will do. One step = 1 ms of simulated time; ~0.7 engine commands per step.

| Implementation | File | Lines |
|---|---|---|
| Pure-Python reference (oracle) | `src/alphalab/lob/engine_ref.py` | 235 |
| numba v2 (flat int64 state array) | `src/alphalab/lob/engine_numba.py` | 397 |
| Rust (PyO3 0.29 + rust-numpy 0.29, maturin) | `lob_rs/src/lib.rs` | 439 |
| numba v1 (tuple of 14 arrays), archived | `bench/archive/engine_numba_v1_tuple.py` | 423 |

## Correctness (gate before speed)

All three engines produce **bit-identical result vectors** (hash of the full trade tape, fills, volume, MM inventory/cash, final best bid/ask, counters) for full loops at 1e5 and 1e6 steps (ref included), and numba vs Rust identically at 1e7 steps; engine-only tape replays are identical at all sizes. 15 engine tests (hand-checked price priority, time priority, partial fills, cancels, crossing limits, MM accounting, 4 seeds three-way) pass.

## Results (best of N runs; N=5 for 1e5 and 1e6 steps, N=3 for 1e7)

Full loop (flow + engine + A-S strategy compiled together):

| Steps | Commands | numba v2 | Rust | Rust / numba | Reference (Python) |
|---|---|---|---|---|---|
| 1e5 | 70k | 0.0050 s (19.9 Msteps/s) | 0.0036 s (28.1) | 1.41x | 0.256 s (0.39) |
| **1e6** | 697k | **0.0332 s (30.1 Msteps/s, 21.0 Mcmd/s)** | **0.0204 s (49.1 Msteps/s, 34.2 Mcmd/s)** | **1.63x** | 2.65 s (0.38) |
| 1e7 | 6.97M | 0.318 s (31.5 Msteps/s, 22.0 Mcmd/s) | 0.191 s (52.5 Msteps/s, 36.6 Mcmd/s) | 1.67x | not run |

Engine only (replay of the recorded command tape):

| Steps (tape) | numba v2 | Rust | Rust / numba |
|---|---|---|---|
| 1e5 (70k cmds) | 29.7 Mcmd/s | 29.5 Mcmd/s | ~1.0x (too small; fixed costs dominate) |
| 1e6 (697k cmds) | 42.7 Mcmd/s | 68.1 Mcmd/s | 1.59x |
| 1e7 (6.97M cmds) | 44.7 Mcmd/s | 79.1 Mcmd/s | 1.77x |

Python calling the engine once per command (the "flow/strategy stay in Python" design):

| Engine | Calls/s | Time per call |
|---|---|---|
| Rust (PyO3 `Book.cmd`) | 14.0 M | ~72 ns |
| numba v2 (jitted `_step_cmd`) | 3.8 M | ~264 ns |
| pure-Python reference | 2.1 M | ~480 ns |

One-off and resource costs:

| | numba v2 | Rust |
|---|---|---|
| Startup cost per process | JIT compile on first call: ~20 s (full loop), ~4.7 s (tape replay); avoidable with `cache=True` (not enabled/measured) | none (compiled ahead of time) |
| Build cost | none | cold first build ~23 s `cargo` compile (release, LTO, includes compiling pyo3/numpy/ndarray), ~34 s incl. maturin wheel install. **Incremental rebuild after editing `lib.rs`: ~6 s compile, ~10 s total** (measured 2026-10-13) |
| Peak process memory, full loop | ~594 MB (LLVM/JIT runtime dominates) | 45 MB (1e6), 124 MB (1e7) |
| Peak process memory, tape replay | 239 MB (1e6), 415 MB (1e7) | 57 MB (1e6), 245 MB (1e7) |

## The numba redesign (important finding)

The first numba version (v1) kept state as a tuple of 14 arrays passed between helpers. It ran at **2.3 Msteps/s**, about 20x slower than Rust, and about as fast as pure Python for engine-only replay. Likely cause (inferred from the fix, **not profiled**): tuple element access in numba takes a new reference with an atomic refcount operation, and passing the tuple copies array descriptors; moving to one flat array removed the slowdown. Steps tried:

| Version | Full loop, 1e6 steps |
|---|---|
| v1: tuple state | 0.44 s (2.3 Msteps/s) |
| v1 + `inline='always'` on helpers | 0.23 s (4.4 Msteps/s) |
| **v2: single flat int64 array + constant offsets + inlined helpers** | **0.033 s (30.1 Msteps/s)** |

Rust needed no such tuning: the straightforward struct-of-Vecs implementation (safe code, bounds checks on) was fast first time. Lesson for later phases: **in numba, keep state in few flat arrays and avoid tuples/containers of arrays in hot paths.**

## Decision

Rule fixed before running (see plan): pick the higher throughput on engine + strategy at 1e6 steps; if the difference is under 1.5x, pick numba for simplicity.

* Full loop at 1e6 steps: Rust 1.63x faster (1.67x at 1e7), above the 1.5x threshold. Engine-only: 1.6-1.8x. Per-call from Python: ~3.7x faster. Memory ~5-13x lower. No JIT warm-up.
* **Decision: implement the market-making simulator on the Rust engine** (`lob_rs`), with the strategy logic (A-S quoting, EMA/RSI/breakout/support-resistance signals) written in Rust inside the loop, so nothing crosses the Python boundary per event.
* Keep the numba engine and the Python reference as **cross-checks**: any new Rust strategy/signal feature must reproduce the reference engine's results bit-for-bit on small runs (the tests above are the template). The numba engine is the fallback if Rust development proves too slow for an experiment.

## Addendum after Phase 6e (2026-10-13)

Two model changes happened after the decision, both applied to all three engines and re-verified bit-identical:
(1) the noise flow's limit-order anchor was rounded down, which made the price drift ~70 ticks per session; it is now
symmetric (buys k below ceil(mid), sells k above floor(mid)); (2) the Rust and reference engines gained the Phase 6e
features (RSI, breakout, support/resistance, imbalance, informed flow, fill/markout log). The numba engine was
frozen at the base model plus fix (1).

Quick re-run (`bench/bench_output_after_6e.txt`, best of 5, 1e6 steps, features off):

| | numba (base model) | Rust (extended model, features off) | ratio |
|---|---|---|---|
| Full loop | 28.96 Msteps/s | 41.23 Msteps/s | 1.42x |
| Engine-only replay | 57.9 Msteps/s | 87.1 Msteps/s | 1.50x |

The full-loop ratio is now below the original 1.5x rule. This is not like for like: with features off, Rust still
updates RSI, the volume baseline and the order-book imbalance at every requote, which the numba engine never computes.
A first version also scanned the breakout and support/resistance windows even when unused (41 -> 28 Msteps/s); those
scans are now skipped when their weights are zero, which cannot change outputs. The decision stands: it was made on
identical models, and the simulator's features now exist only in Rust (and the Python oracle).

## Caveats (read before relying on the margin)

* Single machine, no CPU pinning, Windows scheduling noise; the 1e5 numbers are small and include fixed allocation costs. The 1.6-1.7x ratio is consistent across the 1e6 and 1e7 sizes and across best/median.
* The two compiled engines are not exhaustively tuned. Rust is safe, bounds-checked code (could be faster with `get_unchecked`); numba could possibly gain from int32 arrays or restructuring. The 1.5x rule has little margin against such tuning, but v2 numba already took one major redesign, which counts as effort spent.
* Engine semantics are intentionally simple (integer ticks, owner flags, no order modification, no hidden/iceberg orders, no fees/latency). The workload is synthetic flow, not NSE tick data.
* Flexibility cost of the decision: a **new kind of signal or strategy logic** needs Rust code plus a ~10 s incremental rebuild (the ~25 s I first quoted was the cold build). numba is not free either: it recompiles on the next run after the strategy source changes (~20 s measured cold compile for the full loop; `cache=True` would only avoid it when nothing changed). Changing **parameters** (windows, weights, spreads, seeds) needs neither a rebuild nor a recompile in either engine. At these speeds a 1e7-step run takes 0.19 s (Rust) vs 0.32 s (numba), so raw speed is rarely the binding constraint; ergonomics (Python syntax, `NUMBA_DISABLE_JIT` debugging) favours numba, memory/startup/portfolio signal favours Rust. Mitigation: keep signal definitions small and test each against the Python reference.
* Memory figures are peak working set of the whole process (interpreter + libraries included).
