"""Calibrate the simulator's price scale to NSE.

The synthetic flow has its own volatility in ticks per session. We match *relative* daily volatility:
  sigma_session_ticks / p_eff = sigma_daily_NSE   =>   p_eff = sigma_session_ticks / sigma_daily_NSE
where sigma_daily_NSE is the median daily log-return volatility of the most liquid NSE names over a
recent window. One tick is then 1/p_eff of the price; the implied tick size in basis points is reported
so the reader can judge realism (NSE large caps trade with a tick of 0.05-0.10 INR on prices of
hundreds to thousands of INR, i.e. roughly 0.3-5 bp -- an order-of-magnitude reference, not a fit).
"""
from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl

from ..backtest.pipeline import research_panel
from . import sim
from .spec import Params


def nse_daily_vol(top_k: int = 20, start: date = date(2023, 1, 1)) -> tuple[float, list[str]]:
    """Median daily log-return std of the top_k most liquid names (by average traded value since `start`)."""
    p = research_panel(100).filter(pl.col("date") >= start)
    top = (p.group_by("symbol").agg(pl.col("traded_value").mean().alias("tv"))
           .sort("tv", descending=True).head(top_k)["symbol"].to_list())
    vols = (p.filter(pl.col("symbol").is_in(top)).sort("symbol", "date")
            .with_columns(lr=pl.col("close").log().diff().over("symbol"))
            .group_by("symbol").agg(pl.col("lr").std().alias("v")))
    return float(vols["v"].median()), top


def sim_session_vol(base: Params, n_sessions: int = 60, seeds=(11, 12, 13)) -> dict:
    """Std of the mid-price change per session (ticks), plus activity stats, for the baseline flow."""
    changes, spreads_proxy, fills = [], [], []
    for s in seeds:
        r = sim.run(Params(**{**base.__dict__, "seed": s, "n_steps": n_sessions * base.day_steps}))
        mid = r.daily["mid2"].to_numpy() / 2.0
        changes.append(np.diff(mid))
        fills.append(r.daily["n_fills"].to_numpy()[-1] / n_sessions)
    ch = np.concatenate(changes)
    return {"sigma_ticks": float(ch.std(ddof=1)), "trades_per_session": float(np.mean(fills)),
            "n_sessions": int(len(ch))}


def calibrate(base: Params | None = None) -> dict:
    base = base or Params()
    sigma_d, names = nse_daily_vol()
    s = sim_session_vol(base)
    p_eff = s["sigma_ticks"] / sigma_d
    return {"nse_sigma_daily": sigma_d, "nse_names": names, **s, "p_eff": p_eff,
            "tick_bp": 1e4 / p_eff, "day_steps": base.day_steps}
