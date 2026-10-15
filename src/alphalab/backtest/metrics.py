"""The six BRAIN-style metrics: Sharpe, Turnover, Fitness, Returns, Drawdown, Margin.

Book convention (matches WorldQuant BRAIN, checked against our platform log): positions are scaled to
gross book G = 1 (sum |w| = 1, i.e. long 0.5 + short 0.5). Returns and Drawdown are expressed relative
to half the book (G/2) -- the capital actually deployed per side; Turnover is daily traded value / G;
Margin is total PnL / total traded value. Inputs are daily series in units of G.
"""
from __future__ import annotations

import numpy as np
import polars as pl

DAYS = 252


def sharpe(pnl: np.ndarray) -> float:
    pnl = np.asarray(pnl, float)
    sd = pnl.std(ddof=1) if len(pnl) > 1 else 0.0
    return float(pnl.mean() / sd * np.sqrt(DAYS)) if sd > 0 else 0.0


def returns(pnl: np.ndarray, half_book: float = 0.5) -> float:
    return float(np.mean(pnl) * DAYS / half_book)


def turnover(traded: np.ndarray, book: float = 1.0) -> float:
    return float(np.mean(traded) / book)


def drawdown(pnl: np.ndarray, half_book: float = 0.5) -> float:
    cum = np.cumsum(pnl)
    peak = np.maximum.accumulate(np.concatenate([[0.0], cum]))[1:]
    return float(np.max(peak - cum) / half_book) if len(cum) else 0.0


def margin(pnl: np.ndarray, traded: np.ndarray) -> float:
    """PnL per unit traded, in basis points."""
    t = float(np.sum(traded))
    return float(np.sum(pnl) / t * 1e4) if t > 0 else 0.0


def fitness(sharpe_: float, returns_: float, turnover_: float) -> float:
    return float(sharpe_ * np.sqrt(abs(returns_) / max(turnover_, 0.125)))


def summary(pnl: np.ndarray, traded: np.ndarray, half_book: float = 0.5) -> dict:
    """Six metrics from daily series in units of the gross book. `half_book` is the capital base for
    Returns/Drawdown (0.5 for a BRAIN long-short book; 1.0 when the series are already per unit of capital)."""
    s, r, t = sharpe(pnl), returns(pnl, half_book), turnover(traded)
    return {"sharpe": s, "turnover": t, "fitness": fitness(s, r, t), "returns": r,
            "drawdown": drawdown(pnl, half_book), "margin_bps": margin(pnl, traded), "days": int(len(pnl))}


def yearly(daily: pl.DataFrame) -> pl.DataFrame:
    """Per-year metrics table like the BRAIN results page. `daily` needs date, pnl, traded, n_long, n_short."""
    rows = []
    for (y,), g in daily.sort("date").group_by(pl.col("date").dt.year(), maintain_order=True):
        m = summary(g["pnl"].to_numpy(), g["traded"].to_numpy())
        m.update(year=int(y), n_long=float(g["n_long"].mean()), n_short=float(g["n_short"].mean()))
        rows.append(m)
    return pl.DataFrame(rows).select("year", "sharpe", "turnover", "fitness", "returns", "drawdown",
                                     "margin_bps", "n_long", "n_short", "days")
