"""Statistics for judging a backtest: probabilistic / deflated Sharpe, block-bootstrap confidence intervals.

References: Bailey & Lopez de Prado (2012) "The Sharpe Ratio Efficient Frontier" (PSR);
Bailey & Lopez de Prado (2014) "The Deflated Sharpe Ratio" (DSR); Politis & Romano (1994) stationary bootstrap.
Sharpe ratios inside these formulas are per period (daily), not annualised.
"""
from __future__ import annotations

import numpy as np
from scipy import stats as st

EULER_GAMMA = 0.5772156649015329


def _sr(x: np.ndarray) -> float:
    sd = x.std(ddof=1)
    return float(x.mean() / sd) if sd > 0 else 0.0


def probabilistic_sharpe(pnl: np.ndarray, sr_benchmark: float = 0.0) -> float:
    """P(true per-period Sharpe > sr_benchmark) given sample length, skew and kurtosis."""
    x = np.asarray(pnl, float)
    n = len(x)
    sr = _sr(x)
    g3 = st.skew(x)
    g4 = st.kurtosis(x, fisher=False)
    denom = np.sqrt(max(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2, 1e-12))
    return float(st.norm.cdf((sr - sr_benchmark) * np.sqrt(n - 1) / denom))


def expected_max_sharpe(n_trials: int, var_trial_sr: float) -> float:
    """Expected maximum per-period Sharpe among n_trials independent zero-skill trials."""
    if n_trials <= 1:
        return 0.0
    z1 = st.norm.ppf(1 - 1.0 / n_trials)
    z2 = st.norm.ppf(1 - 1.0 / (n_trials * np.e))
    return float(np.sqrt(var_trial_sr) * ((1 - EULER_GAMMA) * z1 + EULER_GAMMA * z2))


def deflated_sharpe(pnl: np.ndarray, trial_sharpes_daily: np.ndarray) -> float:
    """DSR: probability the selected strategy's Sharpe beats the best expected from pure luck."""
    trial = np.asarray(trial_sharpes_daily, float)
    sr0 = expected_max_sharpe(len(trial), float(np.var(trial, ddof=1)) if len(trial) > 1 else 0.0)
    return probabilistic_sharpe(pnl, sr0)


def stationary_bootstrap_idx(n: int, n_boot: int, mean_block: float, rng: np.random.Generator) -> np.ndarray:
    """Index matrix (n_boot x n) for the stationary bootstrap (geometric block lengths, wrap-around)."""
    p = 1.0 / mean_block
    idx = np.empty((n_boot, n), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n, n_boot)
    new_block = rng.random((n_boot, n)) < p
    starts = rng.integers(0, n, (n_boot, n))
    for t in range(1, n):
        idx[:, t] = np.where(new_block[:, t], starts[:, t], (idx[:, t - 1] + 1) % n)
    return idx


def bootstrap_sharpe_ci(pnl: np.ndarray, n_boot: int = 2000, mean_block: float = 20.0,
                        alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """Annualised Sharpe with a (1-alpha) stationary-bootstrap percentile interval: (sharpe, lo, hi)."""
    x = np.asarray(pnl, float)
    rng = np.random.default_rng(seed)
    idx = stationary_bootstrap_idx(len(x), n_boot, mean_block, rng)
    b = x[idx]
    srs = b.mean(axis=1) / b.std(axis=1, ddof=1) * np.sqrt(252)
    return float(_sr(x) * np.sqrt(252)), float(np.quantile(srs, alpha / 2)), float(np.quantile(srs, 1 - alpha / 2))
