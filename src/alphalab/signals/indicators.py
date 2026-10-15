"""Technical indicators (EMA, RSI, ATR, support/resistance, breakout) as Polars expressions.

Conventions: long panel with columns date, symbol, open, high, low, close, volume, sorted or not
(every expression orders by date within symbol). Levels that describe "the market so far" (support,
resistance, channel) are built from bars *before* today (shift(1)), so comparing today's close with
them is a legitimate same-day signal; the backtest then trades it with delay 1.
"""
from __future__ import annotations

import polars as pl

from .operators import DATE, SYM


def _ts(e: pl.Expr) -> pl.Expr:
    return e.over(SYM, order_by=DATE)


def ema(col: str = "close", span: int = 20) -> pl.Expr:
    """Exponential moving average, alpha = 2 / (span + 1), seeded with the first value."""
    return _ts(pl.col(col).ewm_mean(span=span, adjust=False, ignore_nulls=True))


def rsi(col: str = "close", n: int = 14) -> pl.Expr:
    """Wilder RSI in [0, 100]: smoothed gains / losses with alpha = 1/n."""
    d = pl.col(col).diff()
    gain = d.clip(lower_bound=0.0)
    loss = (-d).clip(lower_bound=0.0)
    avg_g = gain.ewm_mean(alpha=1.0 / n, adjust=False, ignore_nulls=True)
    avg_l = loss.ewm_mean(alpha=1.0 / n, adjust=False, ignore_nulls=True)
    rs = avg_g / avg_l
    out = pl.when(avg_l == 0).then(100.0).otherwise(100.0 - 100.0 / (1.0 + rs))
    # first n bars are warm-up: null
    idx = pl.int_range(pl.len())
    return _ts(pl.when(idx >= n).then(out).otherwise(None))


def true_range() -> pl.Expr:
    pc = pl.col("close").shift(1)
    return _ts(pl.max_horizontal(pl.col("high") - pl.col("low"),
                                 (pl.col("high") - pc).abs(), (pl.col("low") - pc).abs()))


def atr(n: int = 14) -> pl.Expr:
    """Average true range (Wilder smoothing)."""
    pc = pl.col("close").shift(1)
    tr = pl.max_horizontal(pl.col("high") - pl.col("low"), (pl.col("high") - pc).abs(), (pl.col("low") - pc).abs())
    return _ts(tr.ewm_mean(alpha=1.0 / n, adjust=False, ignore_nulls=True))


def resistance(n: int = 20) -> pl.Expr:
    """Highest high of the previous n bars (excludes today)."""
    return _ts(pl.col("high").shift(1).rolling_max(n))


def support(n: int = 20) -> pl.Expr:
    """Lowest low of the previous n bars (excludes today)."""
    return _ts(pl.col("low").shift(1).rolling_min(n))


def pivot_resistance(k: int = 5) -> pl.Expr:
    """Last *confirmed* swing high: a high that is the max of the 2k+1 bars centred on it.

    A pivot at bar t-k can only be confirmed at bar t (k bars later), so the level becomes known
    at t, never earlier (no look-ahead). The level is carried forward until the next pivot.
    """
    h = pl.col("high")
    is_pivot = h.shift(k) == h.rolling_max(2 * k + 1)
    return _ts(pl.when(is_pivot).then(h.shift(k)).otherwise(None).forward_fill())


def pivot_support(k: int = 5) -> pl.Expr:
    lo = pl.col("low")
    is_pivot = lo.shift(k) == lo.rolling_min(2 * k + 1)
    return _ts(pl.when(is_pivot).then(lo.shift(k)).otherwise(None).forward_fill())


def channel_position(n: int = 20) -> pl.Expr:
    """Where today's close sits between support (0) and resistance (1) of the previous n bars."""
    s, r = support(n), resistance(n)
    return pl.when(r > s).then((pl.col("close") - s) / (r - s)).otherwise(None)


def breakout(n: int = 20, vol_mult: float = 1.5, vol_window: int = 20) -> pl.Expr:
    """+1 close above the prior n-bar high, -1 below the prior n-bar low, 0 otherwise.

    With `vol_mult` > 0 the break must come on volume above vol_mult x the prior average volume.
    """
    up = pl.col("close") > resistance(n)
    dn = pl.col("close") < support(n)
    if vol_mult > 0:
        avg_v = _ts(pl.col("volume").shift(1).rolling_mean(vol_window))
        conf = pl.col("volume") > vol_mult * avg_v
        up, dn = up & conf, dn & conf
    return pl.when(up).then(1.0).when(dn).then(-1.0).otherwise(0.0)
