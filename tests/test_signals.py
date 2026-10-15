"""Phase 3 tests: operators and indicators vs hand/NumPy references, plus a generic no-look-ahead test."""
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from alphalab.signals import indicators as ind
from alphalab.signals import operators as op


def panel(n=60, symbols=("A", "B", "C", "D"), seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for j, s in enumerate(symbols):
        c = 100 + np.cumsum(rng.normal(0, 1, n))
        hi = c + rng.uniform(0, 1, n)
        lo = c - rng.uniform(0, 1, n)
        op_ = c + rng.normal(0, 0.3, n)
        v = rng.uniform(1e5, 2e5, n)
        for i in range(n):
            rows.append(dict(date=date(2021, 1, 1) + timedelta(days=i), symbol=s, open=op_[i], high=hi[i],
                             low=lo[i], close=c[i], volume=v[i], sector="X" if j < 2 else "Y"))
    # shuffle rows: operators must not rely on input order
    return pl.DataFrame(rows).sample(fraction=1.0, shuffle=True, seed=1)


def series(df, sym, col):
    return df.filter(pl.col("symbol") == sym).sort("date")[col].to_numpy()


def test_ts_rank_matches_manual():
    df = panel().with_columns(r=op.ts_rank("close", 5))
    x, got = series(df, "A", "close"), series(df, "A", "r")
    for t in range(4, len(x)):
        w = x[t - 4:t + 1]
        exp = ((w < x[t]).sum() + 0.5 * ((w == x[t]).sum() - 1)) / 4
        assert got[t] == pytest.approx(exp)
    assert np.isnan(got[:4]).all() or all(v is None for v in got[:4])


def test_ts_decay_linear_matches_manual_and_skips_nulls():
    df = panel(20).with_columns(
        x=pl.when((pl.col("symbol") == "A") & (pl.col("date") == date(2021, 1, 10))).then(None).otherwise(pl.col("close")))
    df = df.with_columns(d=op.ts_decay_linear("x", 4))
    x, got = series(df, "A", "x"), series(df, "A", "d")
    for t in range(3, 20):
        w = np.array([4, 3, 2, 1], float)
        vals = np.array([x[t - i] for i in range(4)], float)
        m = ~np.isnan(vals)
        assert got[t] == pytest.approx((w[m] * vals[m]).sum() / w[m].sum())


def test_ts_delta_and_returns():
    df = panel().with_columns(d=op.ts_delta("close", 3), r=op.ts_returns("close", 1))
    x = series(df, "B", "close")
    assert series(df, "B", "d")[3:] == pytest.approx(x[3:] - x[:-3])
    assert series(df, "B", "r")[1:] == pytest.approx(x[1:] / x[:-1] - 1)


def test_cross_sectional_rank_in_unit_interval_and_ordered():
    df = panel().with_columns(r=op.rank("close"))
    for _, g in df.group_by("date"):
        g = g.sort("close")
        assert g["r"].to_list() == pytest.approx([0, 1 / 3, 2 / 3, 1])


def test_group_neutralize_sums_to_zero_per_group():
    df = panel().with_columns(n=op.group_neutralize("close", "sector"))
    s = df.group_by("date", "sector").agg(pl.col("n").sum().abs().alias("s"))
    assert s["s"].max() < 1e-9


def test_group_rank_within_group():
    df = panel().with_columns(r=op.group_rank("close", "sector"))
    assert set(df["r"].unique().to_list()) <= {0.0, 1.0}   # two names per sector


def test_ema_matches_recursion():
    df = panel().with_columns(e=ind.ema("close", 9))
    x, got = series(df, "C", "close"), series(df, "C", "e")
    a, e = 2 / 10, x[0]
    for t in range(len(x)):
        e = x[0] if t == 0 else a * x[t] + (1 - a) * e
        assert got[t] == pytest.approx(e)


def test_rsi_matches_wilder_recursion_and_bounds():
    df = panel(80).with_columns(r=ind.rsi("close", 14))
    x, got = series(df, "A", "close"), series(df, "A", "r")
    d = np.diff(x)
    g, lo = np.clip(d, 0, None), np.clip(-d, 0, None)
    ag, al = g[0], lo[0]
    for t in range(1, len(x)):
        if t > 1:
            ag = ag + (g[t - 1] - ag) / 14
            al = al + (lo[t - 1] - al) / 14
        if t >= 14:
            exp = 100.0 if al == 0 else 100 - 100 / (1 + ag / al)
            assert got[t] == pytest.approx(exp)
    valid = got[14:]
    assert ((valid >= 0) & (valid <= 100)).all()


def test_rsi_extremes():
    up = pl.DataFrame({"date": [date(2021, 1, 1) + timedelta(days=i) for i in range(30)],
                       "symbol": ["U"] * 30, "close": np.arange(30, dtype=float) + 1})
    assert up.with_columns(r=ind.rsi("close", 14))["r"][20] == 100.0


def test_support_resistance_exclude_today():
    df = panel().with_columns(r=ind.resistance(5), s=ind.support(5))
    hi, lo = series(df, "D", "high"), series(df, "D", "low")
    r, s = series(df, "D", "r"), series(df, "D", "s")
    for t in range(6, len(hi)):
        assert r[t] == pytest.approx(hi[t - 5:t].max())
        assert s[t] == pytest.approx(lo[t - 5:t].min())


def test_breakout_needs_volume_confirmation():
    n = 30
    c = np.full(n, 100.0)
    c[-1] = 110.0
    base = pl.DataFrame({"date": [date(2021, 1, 1) + timedelta(days=i) for i in range(n)], "symbol": ["Z"] * n,
                         "open": c, "high": c + 0.5, "low": c - 0.5, "close": c})
    low_vol = base.with_columns(volume=pl.lit(1000.0))
    hi_vol = base.with_columns(volume=pl.when(pl.int_range(pl.len()) == n - 1).then(5000.0).otherwise(1000.0))
    assert low_vol.with_columns(b=ind.breakout(20, 1.5))["b"][-1] == 0.0
    assert hi_vol.with_columns(b=ind.breakout(20, 1.5))["b"][-1] == 1.0
    assert low_vol.with_columns(b=ind.breakout(20, 0.0))["b"][-1] == 1.0


def test_pivot_levels_are_past_values():
    df = panel().with_columns(p=ind.pivot_resistance(3))
    hi, p = series(df, "A", "high"), series(df, "A", "p")
    for t in range(len(hi)):
        if p[t] == p[t]:                       # not NaN
            assert p[t] in set(hi[: t - 2])    # a high at least k=3 bars old


def single(expr):
    return lambda df: df.with_columns(v=expr)


def two_step(inner, outer):
    """Materialise a time-series expression, then apply a cross-sectional one (the safe pattern)."""
    return lambda df: df.with_columns(_i=inner).with_columns(v=outer("_i"))


EXPRS = {
    "ts_rank": single(op.ts_rank("close", 10)), "ts_decay": single(op.ts_decay_linear("close", 5)),
    "ts_zscore": single(op.ts_zscore("close", 10)), "ema": single(ind.ema("close", 10)),
    "rsi": single(ind.rsi("close", 14)), "atr": single(ind.atr(14)), "res": single(ind.resistance(10)),
    "sup": single(ind.support(10)), "piv_r": single(ind.pivot_resistance(3)), "piv_s": single(ind.pivot_support(3)),
    "chan": single(ind.channel_position(10)), "brk": single(ind.breakout(10, 1.2)),
    "rank": two_step(op.ts_delta("close", 5), op.rank),
    "gneut": two_step(ind.rsi("close", 14), lambda c: op.group_neutralize(c, "sector")),
}


@pytest.mark.parametrize("name", list(EXPRS))
def test_no_lookahead(name):
    """Values on/before a cutoff must be identical whether or not later data exists."""
    df = panel(80)
    cut = date(2021, 1, 1) + timedelta(days=50)
    f = EXPRS[name]
    full = f(df).filter(pl.col("date") <= cut).sort("symbol", "date")["v"]
    trunc = f(df.filter(pl.col("date") <= cut)).sort("symbol", "date")["v"]
    a, b = full.to_numpy().astype(float), trunc.to_numpy().astype(float)
    assert (~np.isnan(a)).sum() > len(a) // 2          # guard against vacuous all-null comparisons
    assert np.array_equal(np.isnan(a), np.isnan(b))
    assert np.allclose(a[~np.isnan(a)], b[~np.isnan(b)])


@pytest.mark.parametrize("cs", [op.rank, op.zscore, op.demean, lambda e: op.group_rank(e, "sector"),
                                lambda e: op.group_neutralize(e, "sector")])
def test_cross_sectional_refuses_nested_window(cs):
    """Polars evaluates cross-sectional-over-time-series windows to all nulls; we raise instead."""
    with pytest.raises(ValueError, match="materialise"):
        cs(op.ts_delta("close", 5))


def test_nested_window_is_all_null_in_polars():
    """Documents the Polars behaviour the guard protects against (if this starts failing, Polars changed)."""
    df = panel(30)
    raw = ((op.ts_delta("close", 5).rank() - 1)).over("date")
    assert df.with_columns(v=raw)["v"].drop_nulls().drop_nans().len() == 0
