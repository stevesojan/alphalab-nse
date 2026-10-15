"""Vectorised BRAIN-style backtest on the long NSE panel (Polars).

alpha (per date x symbol) -> mask to universe -> linear decay -> neutralize (none/market/sector)
-> scale to gross 1 -> truncate per-name weight -> hold from the next close (delay) -> daily PnL.

Timing (as on BRAIN): an alpha value on date t uses data up to the close of t. The trade is executed
`delay` days later at the close, so with delay=1 the weight w_t is entered at the close of t+1 and earns
the close-to-close return of day t+2: pnl_d = sum_s w_{s, d-1-delay} * r_{s,d}. delay=0 (trade at the
same close that produced the signal) is available for comparison but is optimistic.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import polars as pl

from ..data import universe as U
from ..signals import operators as op
from . import metrics as M

DATE, SYM = "date", "symbol"


@dataclass(frozen=True)
class Settings:
    neutralization: str = "sector"   # "none" | "market" | "sector"
    decay: int = 4                   # linear decay window on the alpha (1 = off)
    truncation: float = 0.08         # max |weight| per name as a fraction of gross book
    delay: int = 1
    cost_bps: float = 0.0            # one-way cost per unit traded (bps); 0 = BRAIN-like gross numbers
    top_n: int = 100                 # universe size (liquidity rank)


def research_panel(top_n: int = 100) -> pl.DataFrame:
    """Panel with universe flag, sector and the daily close-to-close return `ret`."""
    p = U.with_universe(U.load_panel(), top_n=top_n).collect()
    p = p.join(U.load_sectors(), on=SYM, how="left").with_columns(pl.col("sector").fill_null("UNKNOWN"))
    return p.with_columns(ret=op.ts_returns("close", 1)).sort(SYM, DATE)


def weights(df: pl.DataFrame, alpha: str = "alpha", s: Settings = Settings()) -> pl.DataFrame:
    """Return df with column `w`: final portfolio weights decided at each date's close."""
    a = pl.col(alpha)
    clean = pl.when(pl.col("in_universe") & a.is_finite()).then(a).otherwise(None)
    out = df.with_columns(_a=clean)
    if s.decay > 1:
        out = out.with_columns(_a=op.ts_decay_linear("_a", s.decay))
        out = out.with_columns(_a=pl.when(pl.col("in_universe")).then(pl.col("_a")).otherwise(None))
    if s.neutralization == "market":
        out = out.with_columns(_a=op.demean("_a"))
    elif s.neutralization == "sector":
        out = out.with_columns(_a=op.group_neutralize("_a", "sector"))
    elif s.neutralization != "none":
        raise ValueError(s.neutralization)
    w = pl.col("_a") / pl.col("_a").abs().sum().over(DATE)
    out = out.with_columns(w=w.fill_nan(None))
    if s.truncation and s.truncation < 1:
        out = _truncate(out, s.truncation)
    return out.with_columns(w=pl.col("w").fill_null(0.0)).drop("_a")


def _truncate(df: pl.DataFrame, cap: float) -> pl.DataFrame:
    """Exact per-name cap with gross kept at 1 ("water-filling").

    Names whose weight would exceed `cap` are pinned at +/-cap; the remaining names are rescaled to
    fill the rest of the gross book; repeat until no new name breaches the cap (at most n iterations).
    If a date has fewer than 1/cap names, every name ends at the cap and gross is below 1.
    """
    w0 = pl.col("w")
    out = df.with_columns(_capped=pl.lit(False))
    prev = -1
    for _ in range(200):
        n_cap = pl.col("_capped").sum().over(DATE)
        free_sum = pl.when(~pl.col("_capped")).then(w0.abs()).otherwise(0.0).sum().over(DATE)
        scale = ((1.0 - cap * n_cap) / free_sum).clip(lower_bound=0.0)
        trial = pl.when(pl.col("_capped")).then(w0.sign() * cap).otherwise(w0 * scale)
        out = out.with_columns(_capped=pl.col("_capped") | (trial.abs() > cap * (1 + 1e-12)))
        n = int(out["_capped"].sum())
        if n == prev:
            out = out.with_columns(w=trial)
            break
        prev = n
    return out.drop("_capped")


def daily_pnl(df_w: pl.DataFrame, s: Settings = Settings()) -> pl.DataFrame:
    """Daily PnL / traded value / position counts from weights `w` and returns `ret`."""
    x = df_w.sort(SYM, DATE).with_columns(
        w_hold=pl.col("w").shift(s.delay + 1).over(SYM, order_by=DATE).fill_null(0.0),
        w_prev=pl.col("w").shift(1).over(SYM, order_by=DATE).fill_null(0.0),
    )
    x = x.with_columns(
        pnl_i=pl.col("w_hold") * pl.col("ret").fill_null(0.0),
        trade_i=(pl.col("w") - pl.col("w_prev")).abs(),
    )
    d = x.group_by(DATE).agg(
        pl.col("pnl_i").sum().alias("gross_pnl"),
        pl.col("trade_i").sum().alias("traded_today"),
        (pl.col("w") > 0).sum().alias("n_long"),
        (pl.col("w") < 0).sum().alias("n_short"),
        pl.col("w").abs().sum().alias("gross"),
    ).sort(DATE)
    # the change decided at t's close is executed at the close of t+delay; book it on that day
    d = d.with_columns(traded=pl.col("traded_today").shift(s.delay).fill_null(0.0))
    return d.with_columns(pnl=pl.col("gross_pnl") - pl.col("traded") * s.cost_bps * 1e-4)


@dataclass
class Result:
    settings: Settings
    daily: pl.DataFrame
    summary: dict
    yearly: pl.DataFrame

    def row(self, **extra) -> dict:
        return {**extra, **asdict(self.settings), **self.summary}


def run(df: pl.DataFrame, alpha: str = "alpha", s: Settings = Settings(),
        start=None, end=None) -> Result:
    """Backtest an alpha column. `start`/`end` (dates) restrict the evaluation window, not the history used."""
    d = daily_pnl(weights(df, alpha, s), s)
    d = d.filter(pl.col("gross") > 0)   # drop warm-up days with no positions (e.g. 2014 universe warm-up)
    if start is not None:
        d = d.filter(pl.col(DATE) >= start)
    if end is not None:
        d = d.filter(pl.col(DATE) <= end)
    summ = M.summary(d["pnl"].to_numpy(), d["traded"].to_numpy())
    return Result(s, d, summ, M.yearly(d))


def fmt(summary: dict) -> str:
    return (f"Sharpe {summary['sharpe']:.2f} | Fitness {summary['fitness']:.2f} | Returns {summary['returns']:.2%} | "
            f"Turnover {summary['turnover']:.2%} | Drawdown {summary['drawdown']:.2%} | Margin {summary['margin_bps']:.1f} bps")


def wide(df: pl.DataFrame, col: str) -> tuple[np.ndarray, list, list]:
    """Pivot a long column to a dates x symbols matrix (for MATLAB export and checks)."""
    piv = df.pivot(on=SYM, index=DATE, values=col, aggregate_function="first").sort(DATE)
    return piv.drop(DATE).to_numpy(), piv[DATE].to_list(), piv.columns[1:]
