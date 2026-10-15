"""BRAIN-style operators as Polars expressions on a long panel (one row per date x symbol).

Time-series operators run per symbol in date order (`.over(SYM, order_by=DATE)`); cross-sectional
operators run per date (optionally per date x group). Every time-series operator uses only the
current and past rows, so applying them never looks ahead.

IMPORTANT (Polars 2.0 behaviour, verified): a cross-sectional window wrapped around a time-series window,
e.g. `rank(ts_delta(...))` in one expression, silently evaluates to all nulls. Materialise the
time-series result as a column first (`df.with_columns(x=ts_delta(...)).with_columns(r=rank("x"))`).
The cross-sectional operators below raise instead of returning nulls when handed such an expression.
"""
from __future__ import annotations

import polars as pl

DATE, SYM = "date", "symbol"

IntoExpr = pl.Expr | str


def _e(x: IntoExpr) -> pl.Expr:
    return pl.col(x) if isinstance(x, str) else x


def _ts(e: pl.Expr) -> pl.Expr:
    return e.over(SYM, order_by=DATE)


# ---- time series ------------------------------------------------------------------------------
def ts_delay(x: IntoExpr, n: int) -> pl.Expr:
    return _ts(_e(x).shift(n))


def ts_delta(x: IntoExpr, n: int) -> pl.Expr:
    return _ts(_e(x) - _e(x).shift(n))


def ts_returns(x: IntoExpr, n: int = 1) -> pl.Expr:
    return _ts(_e(x) / _e(x).shift(n) - 1.0)


def ts_mean(x: IntoExpr, n: int) -> pl.Expr:
    return _ts(_e(x).rolling_mean(n))


def ts_sum(x: IntoExpr, n: int) -> pl.Expr:
    return _ts(_e(x).rolling_sum(n))


def ts_std(x: IntoExpr, n: int) -> pl.Expr:
    return _ts(_e(x).rolling_std(n))


def ts_max(x: IntoExpr, n: int) -> pl.Expr:
    return _ts(_e(x).rolling_max(n))


def ts_min(x: IntoExpr, n: int) -> pl.Expr:
    return _ts(_e(x).rolling_min(n))


def ts_zscore(x: IntoExpr, n: int) -> pl.Expr:
    e = _e(x)
    return _ts((e - e.rolling_mean(n)) / e.rolling_std(n))


def ts_rank(x: IntoExpr, n: int) -> pl.Expr:
    """Rank of today's value within the last n values, scaled to [0, 1] (1 = highest of the window)."""
    return _ts((_e(x).rolling_rank(n, method="average") - 1.0) / (n - 1))


def ts_decay_linear(x: IntoExpr, n: int) -> pl.Expr:
    """Linearly weighted mean of the last n values (weights n, n-1, ..., 1; newest heaviest).

    Missing values are skipped and the weights renormalised over what is present, so a stock that
    drops out for a day does not drag the average toward zero.
    """
    e = _e(x)
    num = sum((n - i) * e.shift(i).fill_null(0.0) for i in range(n))
    den = sum((n - i) * e.shift(i).is_not_null().cast(pl.Float64) for i in range(n))
    return _ts(pl.when(den > 0).then(num / den).otherwise(None))


def ts_ema(x: IntoExpr, span: int) -> pl.Expr:
    return _ts(_e(x).ewm_mean(span=span, adjust=False, ignore_nulls=True))


# ---- cross-sectional --------------------------------------------------------------------------
def _cs(x: IntoExpr) -> pl.Expr:
    """Input for a cross-sectional operator; refuse nested windows (they evaluate to all nulls)."""
    e = _e(x)
    if ".over(" in str(e):
        raise ValueError("cross-sectional operator applied to a windowed (time-series) expression: "
                         "materialise it as a column first, e.g. df.with_columns(x=...).with_columns(r=rank('x'))")
    return e


def rank(x: IntoExpr) -> pl.Expr:
    """Cross-sectional rank per date scaled to [0, 1]; nulls stay null."""
    e = _cs(x)
    return ((e.rank("average") - 1.0) / (e.count() - 1).clip(lower_bound=1)).over(DATE)


def zscore(x: IntoExpr) -> pl.Expr:
    e = _cs(x)
    return ((e - e.mean()) / e.std()).over(DATE)


def demean(x: IntoExpr) -> pl.Expr:
    e = _cs(x)
    return (e - e.mean()).over(DATE)


def group_rank(x: IntoExpr, group: str) -> pl.Expr:
    e = _cs(x)
    return ((e.rank("average") - 1.0) / (e.count() - 1).clip(lower_bound=1)).over(DATE, group)


def group_neutralize(x: IntoExpr, group: str) -> pl.Expr:
    """Subtract the group mean per date (BRAIN 'neutralization' for that grouping)."""
    e = _cs(x)
    return (e - e.mean()).over(DATE, group)


def sign(x: IntoExpr) -> pl.Expr:
    return _e(x).sign()
