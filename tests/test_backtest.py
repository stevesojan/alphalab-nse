"""Phase 4 tests: metric definitions vs the WorldQuant BRAIN log, pipeline invariants and timing."""
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from alphalab.backtest import metrics as M
from alphalab.backtest.pipeline import Settings, daily_pnl, run, weights
from alphalab.signals import operators as op

# (window, Sharpe, Returns, Turnover, Fitness, Margin bps) from alpha_log_fn_liab_fair_val.md section 4
BRAIN_LOG = [
    (63, 0.89, 0.0802, 0.0855, 0.71, 18.75),
    (90, 1.42, 0.1089, 0.0665, 1.33, 32.75),
    (105, 1.64, 0.1158, 0.0599, 1.58, 38.65),
    (111, 1.69, 0.1154, 0.0577, 1.62, 39.97),
    (120, 1.73, 0.1113, 0.0545, 1.63, 40.82),
    (180, 1.45, 0.0716, 0.0403, 1.10, 35.51),
    (252, 0.95, 0.0355, 0.0284, 0.51, 25.03),
]


@pytest.mark.parametrize("w,s,r,t,f,m", BRAIN_LOG)
def test_fitness_formula_reproduces_brain(w, s, r, t, f, m):
    assert M.fitness(s, r, t) == pytest.approx(f, abs=0.015)   # inputs are rounded to 2 dp


@pytest.mark.parametrize("w,s,r,t,f,m", BRAIN_LOG)
def test_margin_consistent_with_half_book_returns(w, s, r, t, f, m):
    # Margin = PnL/traded; with Returns on half book and daily Turnover on full book:
    # Margin = Returns / (2 * 252 * Turnover). BRAIN's printed margins agree within ~1%.
    assert 1e4 * r / (2 * 252 * t) == pytest.approx(m, rel=0.015)


def test_summary_identities_exact():
    rng = np.random.default_rng(0)
    pnl, traded = rng.normal(2e-4, 3e-3, 750), rng.uniform(0.02, 0.1, 750)
    sm = M.summary(pnl, traded)
    assert sm["margin_bps"] == pytest.approx(1e4 * sm["returns"] / (2 * 252 * sm["turnover"]))
    assert sm["sharpe"] == pytest.approx(pnl.mean() / pnl.std(ddof=1) * np.sqrt(252))


def test_drawdown_hand_example():
    pnl = np.array([0.01, 0.01, -0.03, 0.005, -0.01, 0.04])   # peak 0.02 -> trough -0.015
    assert M.drawdown(pnl) == pytest.approx(0.035 / 0.5)


def synth(n_days=300, n_sym=30, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for j in range(n_sym):
        c = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n_days)))
        for i in range(n_days):
            rows.append(dict(date=date(2020, 1, 1) + timedelta(days=i), symbol=f"S{j:02d}", close=c[i],
                             sector=f"G{j % 3}", in_universe=True, alpha=rng.normal()))
    return pl.DataFrame(rows).with_columns(ret=op.ts_returns("close", 1))


def test_weights_neutral_scaled_and_capped():
    df = synth()
    w = weights(df, s=Settings(neutralization="sector", decay=1, truncation=1.0))
    g = w.group_by("date", "sector").agg(pl.col("w").sum().abs().alias("s"))
    assert g["s"].max() < 1e-9                                      # sector neutral
    gross = w.group_by("date").agg(pl.col("w").abs().sum().alias("g"))["g"]
    assert np.allclose(gross.to_numpy(), 1.0)                          # gross 1
    w2 = weights(df, s=Settings(neutralization="market", decay=1, truncation=0.05))
    assert w2["w"].abs().max() <= 0.05 + 1e-6                          # cap respected
    assert np.allclose(w2.group_by("date").agg(pl.col("w").abs().sum())["w"].to_numpy(), 1.0)


def test_decay_reduces_turnover():
    df = synth()
    t1 = run(df, s=Settings(decay=1, neutralization="market")).summary["turnover"]
    t10 = run(df, s=Settings(decay=10, neutralization="market")).summary["turnover"]
    assert t10 < 0.5 * t1


def test_universe_mask_excludes_names():
    df = synth().with_columns(in_universe=pl.col("symbol") != "S00")
    w = weights(df, s=Settings(decay=3))
    assert w.filter(pl.col("symbol") == "S00")["w"].abs().max() == 0.0


def test_delay_timing_with_cheating_alphas():
    """Alpha = return of day t+1 is only exploitable with delay 0; alpha = return of t+2 with delay 1."""
    df = synth(400, 40, seed=3)
    df = df.with_columns(
        fwd1=pl.col("ret").shift(-1).over("symbol", order_by="date"),
        fwd2=pl.col("ret").shift(-2).over("symbol", order_by="date"),
    )
    s0 = Settings(delay=0, decay=1, neutralization="market", truncation=1.0)
    s1 = Settings(delay=1, decay=1, neutralization="market", truncation=1.0)
    assert run(df, "fwd1", s0).summary["sharpe"] > 20
    assert abs(run(df, "fwd1", s1).summary["sharpe"]) < 3
    assert run(df, "fwd2", s1).summary["sharpe"] > 20


def test_pnl_matches_manual_two_names():
    d = [date(2020, 1, 1) + timedelta(days=i) for i in range(5)]
    closes = {"A": [100, 101, 103, 102, 104], "B": [50, 50, 49, 51, 52]}
    rows = [dict(date=d[i], symbol=s, close=float(c[i]), sector="X", in_universe=True,
                 alpha=1.0 if s == "A" else -1.0) for s, c in closes.items() for i in range(5)]
    df = pl.DataFrame(rows).with_columns(ret=op.ts_returns("close", 1))
    s = Settings(decay=1, neutralization="none", truncation=1.0, delay=1)
    out = daily_pnl(weights(df, s=s), s).sort("date")
    r = {k: np.array(v[1:]) / np.array(v[:-1]) - 1 for k, v in closes.items()}
    # w = +0.5 A, -0.5 B every day; first held return is day index 2 (decided day 0, entered day 1 close)
    exp = [0.0, 0.0] + [0.5 * r["A"][i - 1] - 0.5 * r["B"][i - 1] for i in range(2, 5)]
    assert out["pnl"].to_list() == pytest.approx(exp)
    assert out["traded"].to_list() == pytest.approx([0.0, 1.0, 0.0, 0.0, 0.0])


def test_costs_reduce_pnl_by_traded_times_bps():
    df = synth()
    a = run(df, s=Settings(decay=1, neutralization="market"))
    b = run(df, s=Settings(decay=1, neutralization="market", cost_bps=5))
    diff = (a.daily["pnl"] - b.daily["pnl"]).to_numpy()
    assert np.allclose(diff, a.daily["traded"].to_numpy() * 5e-4)
