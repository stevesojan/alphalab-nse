"""Data-layer tests: synthetic unit tests for the logic, invariants on the real panel (skipped if absent)."""
from datetime import date, timedelta

import polars as pl
import pytest

from alphalab.data import nse_loader as L
from alphalab.data import universe as U

PANEL = L.PARQUET / "nse_daily.parquet"


def synth(n_days=320, symbols=("AAA", "BBB", "CCC"), volume=lambda s, i: 1000.0 * (1 + (hash(s) % 3)) * (1 + i % 5)):
    d0 = date(2020, 1, 1)
    rows = []
    for s in symbols:
        for i in range(n_days):
            c = 100.0 + i * 0.1
            rows.append(dict(date=d0 + timedelta(days=i), symbol=s, open=c, high=c + 1, low=c - 1, close=c,
                             adj_close=c * 0.5, volume=volume(s, i)))
    return pl.DataFrame(rows)


def test_adjust_prices_rescales_ohlc_by_adj_factor():
    df = synth(5, ("AAA",))
    out = L.adjust_prices(df)
    assert out["close"].to_list() == pytest.approx((df["close"] * 0.5).to_list())
    assert out["high"].to_list() == pytest.approx((df["high"] * 0.5).to_list())
    assert out["close_raw"].to_list() == df["close"].to_list()


def test_drop_holiday_bars_removes_dates_where_most_symbols_have_zero_volume():
    df = synth(10)
    bad = date(2020, 1, 4)
    df = df.with_columns(pl.when(pl.col("date") == bad).then(0.0).otherwise(pl.col("volume")).alias("volume"))
    clean, dropped = L.drop_holiday_bars(df)
    assert dropped["date"].to_list() == [bad]
    assert bad not in clean["date"].to_list()


def test_drop_holiday_bars_keeps_isolated_zero_volume_symbol_days():
    df = synth(10)
    df = df.with_columns(pl.when((pl.col("date") == date(2020, 1, 4)) & (pl.col("symbol") == "AAA"))
                         .then(0.0).otherwise(pl.col("volume")).alias("volume"))
    clean, dropped = L.drop_holiday_bars(df)
    assert dropped.height == 0 and clean.height == df.height


def test_universe_adv_has_no_lookahead():
    """Changing volume on day t must not change adv (and membership) on day t."""
    panel = L.adjust_prices(synth(300))
    base = U.with_universe(panel.lazy(), top_n=2, window=60, min_history=100).collect()
    last = base["date"].max()
    changed = panel.with_columns(
        pl.when(pl.col("date") == last).then(pl.col("volume") * 1000).otherwise(pl.col("volume")).alias("volume"))
    alt = U.with_universe(changed.lazy(), top_n=2, window=60, min_history=100).collect()
    a = base.filter(pl.col("date") == last).sort("symbol")
    b = alt.filter(pl.col("date") == last).sort("symbol")
    assert a["adv"].to_list() == b["adv"].to_list()
    assert a["in_universe"].to_list() == b["in_universe"].to_list()


def test_universe_respects_top_n_and_min_history():
    panel = L.adjust_prices(synth(300))
    r = U.with_universe(panel.lazy(), top_n=2, window=60, min_history=100).collect()
    per_day = r.group_by("date").agg(pl.col("in_universe").sum().alias("n"))
    assert per_day["n"].max() <= 2
    assert r.filter(pl.col("n_obs") <= 100)["in_universe"].sum() == 0
    assert r["in_universe"].null_count() == 0


# ---- invariants on the real downloaded panel ------------------------------------------------
real = pytest.mark.skipif(not PANEL.exists(), reason="run `python -m alphalab.data.nse_loader` first")


@real
def test_real_panel_schema_and_uniqueness():
    df = pl.read_parquet(PANEL)
    assert df.columns == ["date", "symbol", "open", "high", "low", "close", "close_raw", "volume"]
    assert df.select(pl.struct("symbol", "date").is_duplicated().any()).item() is False
    assert df["symbol"].n_unique() >= 190


@real
def test_real_panel_price_sanity():
    df = pl.read_parquet(PANEL)
    assert df.filter((pl.col("close") <= 0) | (pl.col("open") <= 0)).height == 0
    assert df.filter(pl.col("high") < pl.col("low") - 1e-9).height == 0


@real
def test_real_panel_has_no_holiday_phantom_dates():
    df = pl.read_parquet(PANEL)
    share = df.group_by("date").agg((pl.col("volume") == 0).mean().alias("z"))
    assert share["z"].max() <= 0.5


@real
def test_real_universe_size_is_stable_after_warmup():
    r = U.with_universe(U.load_panel(), top_n=100).collect()
    n = r.filter(pl.col("date") >= date(2016, 1, 1)).group_by("date").agg(pl.col("in_universe").sum().alias("n"))
    assert 90 <= n["n"].min() and n["n"].max() <= 100
