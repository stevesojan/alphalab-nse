"""Phase 5 tests: PSR/DSR, bootstrap CI, walk-forward splits, plateau selection, ledger."""
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest
from scipy import stats as st

from alphalab.validation import stats as S
from alphalab.validation.ledger import Ledger
from alphalab.validation.splits import walk_forward
from alphalab.validation.sweep import neighbours, plateau_select


def test_psr_matches_closed_form_for_gaussian():
    rng = np.random.default_rng(1)
    x = rng.normal(0.001, 0.01, 2000)
    sr = x.mean() / x.std(ddof=1)
    g3, g4 = st.skew(x), st.kurtosis(x, fisher=False)
    exp = st.norm.cdf(sr * np.sqrt(len(x) - 1) / np.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2))
    assert S.probabilistic_sharpe(x) == pytest.approx(exp)


def test_expected_max_sharpe_grows_with_trials():
    v = 0.05 ** 2
    assert S.expected_max_sharpe(1, v) == 0.0
    vals = [S.expected_max_sharpe(n, v) for n in (2, 10, 100, 1000)]
    assert all(a < b for a, b in zip(vals, vals[1:]))


def test_deflated_sharpe_flags_best_of_many_noise_strategies():
    """A deliberately overfit pick (best of 200 zero-skill strategies) must not pass DSR."""
    rng = np.random.default_rng(7)
    noise = rng.normal(0, 0.01, (200, 500))
    trial_sr = noise.mean(1) / noise.std(1, ddof=1)
    best = noise[np.argmax(trial_sr)]
    raw_annual = trial_sr.max() * np.sqrt(252)
    assert raw_annual > 1.5                                   # looks great in sample...
    assert S.deflated_sharpe(best, trial_sr) < 0.95           # ...but is not significant after deflation
    # after 200 trials the luck bar is ~2 annualised Sharpe, so genuine skill must clear it clearly
    good = rng.normal(0.01 * 3.5 / np.sqrt(252), 0.01, 1500)  # true annual Sharpe 3.5
    assert S.deflated_sharpe(good, trial_sr) > 0.95
    assert S.expected_max_sharpe(200, trial_sr.var(ddof=1)) * np.sqrt(252) > 1.5


def test_bootstrap_ci_brackets_estimate_and_is_reproducible():
    x = np.random.default_rng(3).normal(0.0005, 0.01, 1000)
    a = S.bootstrap_sharpe_ci(x, n_boot=500, seed=1)
    b = S.bootstrap_sharpe_ci(x, n_boot=500, seed=1)
    assert a == b
    sr, lo, hi = a
    assert lo < sr < hi and hi - lo < 3.0


def test_stationary_bootstrap_indices_in_range():
    idx = S.stationary_bootstrap_idx(50, 10, 5.0, np.random.default_rng(0))
    assert idx.min() >= 0 and idx.max() < 50 and idx.shape == (10, 50)


def test_walk_forward_folds_are_ordered_with_embargo():
    ds = [date(2015, 1, 1) + timedelta(days=i) for i in range(365 * 8) if (date(2015, 1, 1) + timedelta(days=i)).weekday() < 5]
    folds = walk_forward(ds, 2018, 2022, embargo=5)
    assert [f.test_start.year for f in folds] == [2018, 2019, 2020, 2021, 2022]
    for f in folds:
        assert f.train_end < f.test_start
        gap = sum(1 for d in ds if f.train_end < d < f.test_start)
        assert gap == 5
        assert f.train_start == ds[0]                         # expanding window
    rolling = walk_forward(ds, 2018, 2018, embargo=0, expanding=False, train_years=2)
    assert rolling[0].train_start.year == 2016


def test_plateau_prefers_broad_region_over_spike():
    axes = {"n": [1, 2, 3, 4, 5, 6, 7]}
    target = {1: -0.5, 2: 3.0, 3: -0.5, 4: 1.0, 5: 1.2, 6: 1.1, 7: 0.9}   # spike at 2, plateau 4-6
    d = [date(2020, 1, 1) + timedelta(days=i) for i in range(500)]
    rng = np.random.default_rng(0)
    res = {}
    for n, sr in target.items():
        z = rng.normal(0, 1, 500)
        z = (z - z.mean()) / z.std(ddof=1)
        pnl = 0.01 * z + 0.01 * sr / np.sqrt(252)            # exact sample Sharpe = sr
        res[(n,)] = pl.DataFrame({"date": d, "pnl": pnl, "traded": np.full(500, 0.1)})
    best, table = plateau_select(res, axes, d[0], d[-1])
    assert best == {"n": 5}
    assert table.filter(pl.col("n") == 2)["sharpe"][0] == pytest.approx(3.0, abs=1e-6)


def test_neighbours_on_grid_edges():
    axes = {"a": [1, 2, 3], "b": ["x", "y"]}
    assert neighbours({"a": 1, "b": "x"}, axes) == [{"a": 2, "b": "x"}, {"a": 1, "b": "y"}]


def test_ledger_counts_unique_trials_and_roundtrips(tmp_path):
    L = Ledger(tmp_path / "ledger.parquet")
    m = {"sharpe": 1.0, "fitness": 0.5, "returns": 0.1, "turnover": 0.2, "drawdown": 0.05, "margin_bps": 3.0}
    L.add("ema", "ema(f)/ema(s)-1", {"f": 5, "s": 50}, "IS", m)
    L.add("ema", "ema(f)/ema(s)-1", {"f": 5, "s": 50}, "OOS", m)
    L.add("ema", "ema(f)/ema(s)-1", {"f": 10, "s": 50}, "IS", m)
    assert L.n_trials() == 2 and L.n_trials("IS") == 2
    L.save()
    L2 = Ledger(tmp_path / "ledger.parquet")
    assert L2.n_trials() == 2
    assert "Unique configurations: 2" in (tmp_path / "ledger.md").read_text()
