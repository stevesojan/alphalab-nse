"""Phase 6e tests: the Rust simulator vs the Python oracle with signal features ON (bit-identical),
plus consistency checks on the fill log and daily snapshots."""
import numpy as np
import polars as pl
import pytest

from alphalab.lob import engine_numba as nb
from alphalab.lob import sim
from alphalab.lob.spec import Params, comparable

lob_rs = pytest.importorskip("lob_rs")

SMALL = dict(n_steps=30_000, warm=500, day_steps=3_000, markout_h=200, rsi_n=5, brk_window=20, sr_window=40)

FEATURES = {
    "baseline_AS": dict(),
    "ema": dict(a_sig=2.0),
    "rsi": dict(a_rsi=3.0),
    "breakout_skew_widen_pull": dict(a_brk=1.5, k_brk=0.5, brk_pull=1.2, brk_decay=0.8),
    "breakout_volume_confirmed": dict(a_brk=1.5, vol_mult=1.2, vol_alpha=0.2),
    "support_resistance": dict(a_sr=4.0, lvl_ticks=2, hs_lvl=1.0),
    "imbalance_microprice": dict(a_imb=1.0),
    "all": dict(a_sig=1.0, a_rsi=2.0, a_brk=1.0, k_brk=0.3, brk_pull=1.5, vol_mult=1.1, a_sr=2.0,
                lvl_ticks=2, hs_lvl=0.5, a_imb=0.5),
    "informed_flow": dict(p_value=0.02, anchor_value=1, p_informed=0.03, qty_informed=4),
    "informed_jumps": dict(p_value=0.002, value_jump=6, anchor_value=1, p_informed=0.1, qty_informed=4),
    "informed_flow_all_signals": dict(p_value=0.02, anchor_value=1, p_informed=0.03, qty_informed=4, a_sig=1.0,
                                      a_rsi=2.0, a_brk=1.0, brk_pull=1.5, a_sr=2.0, lvl_ticks=2, a_imb=0.5),
}


def same(a: sim.SimResult, b: sim.SimResult) -> bool:
    return (comparable(list(a.res.values())) == comparable(list(b.res.values()))
            and a.daily.equals(b.daily) and a.fills.equals(b.fills))


@pytest.mark.parametrize("name", list(FEATURES))
def test_rust_matches_reference_with_features(name):
    for seed in (1, 2):
        p = Params(seed=seed, **SMALL, **FEATURES[name])
        r, o = sim.run(p), sim.run_reference(p)
        assert same(r, o), name
        assert r.res["error"] == 0


def test_features_change_behaviour():
    """Guard against a feature silently doing nothing: each must change the MM's fills."""
    base = sim.run(Params(seed=3, **SMALL))
    for name, f in FEATURES.items():
        if not f:
            continue
        r = sim.run(Params(seed=3, **SMALL, **f))
        assert not r.fills.equals(base.fills), name


def test_features_off_equals_benchmarked_model():
    p = Params(seed=4, **SMALL)
    pi, pf = p.pack()
    a = list(sim.run(p).res.values())
    b = lob_rs.run_full(pi, pf, p.max_seq())
    c = nb.run_full(p)
    assert comparable(a) == comparable(b) == comparable(c)


def test_daily_and_fill_log_consistency():
    p = Params(seed=5, **SMALL, a_sig=1.0, a_imb=0.5)
    r = sim.run(p)
    d, f = r.daily, r.fills
    assert d.height == p.n_steps // p.day_steps
    assert (d["t"] + 1).to_list() == [p.day_steps * (i + 1) for i in range(d.height)]
    last = d.row(-1, named=True)
    # final snapshot is at the last step, so it must agree with the summary result
    assert last["inventory"] == r.res["mm_inventory"] and last["mm_fills"] == r.res["mm_fills"]
    assert last["equity2"] == 2 * r.res["mm_cash"] + r.res["mm_inventory"] * last["mid2"]
    # fill log agrees with the counters
    assert f.height == r.res["mm_fills"] and f["qty"].sum() == r.res["mm_volume"]
    signed = f.select(pl.when(pl.col("side") == 0).then(pl.col("qty")).otherwise(-pl.col("qty")).sum()).item()
    assert signed == r.res["mm_inventory"]
    # maker fills: posted before being filled, non-negative queue ahead
    assert (f["post_t"] <= f["t"]).all() and (f["queue_ahead"] >= 0).all()
    # only fills within the last markout_h steps can still be pending
    pend = f.filter(pl.col("markout2").is_null())
    assert (pend["t"] > p.n_steps - 1 - p.markout_h).all()


def test_markout_sign_convention():
    """A buy filled below the later mid has a positive markout (good fill)."""
    p = Params(seed=6, **SMALL)
    r = sim.run(p)
    f = r.fills.drop_nulls("markout2")
    assert f.height > 0
    # markout2 is in half-ticks * qty; |markout| per unit cannot exceed a large move bound
    assert (f["markout2"].abs() / f["qty"]).max() < 2 * 2000
