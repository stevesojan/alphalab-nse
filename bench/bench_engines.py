"""Benchmark: reference vs numba vs Rust (PyO3) limit-order-book engines.

Scenarios (each runs in its own subprocess for clean cold-start time and peak memory):
  full_*  : entire simulation compiled (Poisson/Hawkes flow + engine + Avellaneda-Stoikov strategy)
  tape_*  : engine only, replaying a recorded command tape
  py_*    : Python loop calling the engine once per command (call-overhead cost)

Usage:
  python bench/bench_engines.py                 # full benchmark, writes bench/results.json
  python bench/bench_engines.py --quick         # small sizes only
Child mode (internal): python bench/bench_engines.py --child MODE N REPS
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
TAPES = HERE / "tapes"
PY = sys.executable


def _peak_mb() -> float:
    import psutil
    return psutil.Process().memory_info().peak_wset / 1e6


def child(mode: str, n: int, reps: int) -> dict:
    t0 = time.perf_counter()
    from alphalab.lob.spec import Params, comparable
    engine = mode.split("_")[1]
    kind = mode.split("_")[0]
    if engine == "numba":
        from alphalab.lob import engine_numba as eng
    elif engine == "rust":
        import lob_rs
    else:
        from alphalab.lob import engine_ref as ref
    t_import = time.perf_counter() - t0
    p = Params(n_steps=n, seed=2024)
    pi, pf = p.pack()
    out = {"mode": mode, "n_steps": n, "t_import": t_import}

    if kind == "full":
        def run():
            if engine == "numba":
                return eng.run_full(p)
            if engine == "rust":
                return np.array(lob_rs.run_full(pi, pf, p.max_seq()))
            return ref.run_full(p)[0]
        if engine == "numba":      # cold compile: first call on a tiny run
            small = Params(n_steps=1000, seed=1)
            t = time.perf_counter()
            eng.run_full(small)
            out["cold_first_call"] = time.perf_counter() - t
        times, res = [], None
        for _ in range(reps):
            t = time.perf_counter()
            res = run()
            times.append(time.perf_counter() - t)
        out["n_cmds"] = int(res[8])
    elif kind == "tape":
        tape = np.load(TAPES / f"tape_{n}.npy")
        ms = int(tape.reshape(-1, 7)[:, 5].max()) + 2
        if engine == "numba":
            t = time.perf_counter()
            eng.run_tape(tape[: 7 * 100])
            out["cold_first_call"] = time.perf_counter() - t

        def run():
            if engine == "numba":
                return eng.run_tape(tape)
            if engine == "rust":
                return np.array(lob_rs.run_tape(tape, ms))
            return ref.run_tape(tape)[0]
        times, res = [], None
        for _ in range(reps):
            t = time.perf_counter()
            res = run()
            times.append(time.perf_counter() - t)
        out["n_cmds"] = int(len(tape) // 7)
    else:  # py_*: per-command Python calls over the first n commands of the 1e6-step tape
        tape = np.load(TAPES / "tape_1000000.npy")
        rows = tape.reshape(-1, 7)[:n].tolist()
        ms = int(tape.reshape(-1, 7)[:, 5].max()) + 2
        times, res = [], None
        for _ in range(reps):
            if engine == "numba":
                d = eng.PyDriven(ms)
                d.cmd(*rows[0])                     # make sure JIT is done before timing
                d = eng.PyDriven(ms)
                f = d.cmd
            elif engine == "rust":
                d = lob_rs.Book(ms)
                f = d.cmd
            else:
                d = ref.RefBook()

                def f(ts, typ, side, price, qty, seq, owner, d=d):
                    if typ == 0:
                        d.limit(ts, side, price, qty, owner, seq)
                    elif typ == 1:
                        d.market(ts, side, qty, owner)
                    else:
                        d.cancel(seq)
            t = time.perf_counter()
            for r in rows:
                f(*r)
            times.append(time.perf_counter() - t)
            res = np.array(d.result()) if engine != "ref" else d.result()
        out["n_cmds"] = len(rows)

    out["times"] = times
    out["peak_mb"] = _peak_mb()
    out["result"] = [int(x) for x in comparable(res)]
    return out


def run_child(mode: str, n: int, reps: int) -> dict:
    r = subprocess.run([PY, __file__, "--child", mode, str(n), str(reps)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{mode} {n} failed:\n{r.stderr[-2000:]}")
    return json.loads(r.stdout.strip().splitlines()[-1])


def make_tapes(sizes):
    from alphalab.lob import engine_numba as eng
    from alphalab.lob.spec import Params
    TAPES.mkdir(exist_ok=True)
    for n in sizes:
        f = TAPES / f"tape_{n}.npy"
        if f.exists():
            continue
        t = time.perf_counter()
        _, tape = eng.run_full(Params(n_steps=n, seed=2024), record=True)
        np.save(f, tape)
        print(f"  tape {n}: {len(tape)//7} commands in {time.perf_counter()-t:.1f}s", flush=True)


def summarize(rows):
    print(f"{'mode':<14}{'steps':>10}{'cmds':>10}{'best s':>9}{'median s':>10}{'Mcmd/s':>9}"
          f"{'Msteps/s':>10}{'cold s':>8}{'peak MB':>9}")
    for r in rows:
        best, med = min(r["times"]), statistics.median(r["times"])
        print(f"{r['mode']:<14}{r['n_steps']:>10}{r['n_cmds']:>10}{best:>9.3f}{med:>10.3f}"
              f"{r['n_cmds']/best/1e6:>9.2f}{r['n_steps']/best/1e6:>10.2f}"
              f"{r.get('cold_first_call', float('nan')):>8.2f}{r['peak_mb']:>9.0f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", nargs=3, metavar=("MODE", "N", "REPS"))
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.child:
        print(json.dumps(child(a.child[0], int(a.child[1]), int(a.child[2]))))
        return

    sizes = [100_000, 1_000_000] if a.quick else [100_000, 1_000_000, 10_000_000]
    print("recording tapes ...", flush=True)
    make_tapes(sizes)
    rows = []

    def go(mode, n, reps):
        print(f"  {mode} n={n} reps={reps}", flush=True)
        rows.append(run_child(mode, n, reps))

    for n in sizes:
        reps = 5 if n < 10_000_000 else 3
        for eng in ("numba", "rust"):
            go(f"full_{eng}", n, reps)
        go("full_ref", n, 1) if n <= 1_000_000 else None
        for eng in ("numba", "rust"):
            go(f"tape_{eng}", n, reps)
        go("tape_ref", n, 1) if n <= 1_000_000 else None
    for eng in ("numba", "rust", "ref"):
        go(f"py_{eng}", 300_000 if eng != "ref" else 100_000, 3)

    summarize(rows)
    # equivalence at scale: compiled engines must agree with each other (and with ref where run)
    print("\nEquivalence at scale (result vectors):")
    ok = True
    by = {}
    for r in rows:
        by.setdefault((r["mode"].split("_")[0], r["n_steps"]), {})[r["mode"].split("_")[1]] = r["result"]
    for (kind, n), d in sorted(by.items()):
        if kind == "py":
            continue
        same = len({tuple(v) for v in d.values()}) == 1
        ok &= same
        print(f"  {kind:<5} n={n:>9}: engines {sorted(d)} identical = {same}")
    print("ALL EQUIVALENT" if ok else "MISMATCH")
    (HERE / "results.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
