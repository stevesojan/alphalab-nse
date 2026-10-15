"""Market-maker performance: the six metrics plus market-making diagnostics.

Units. The simulator works in integer ticks and units. To express PnL as a return we need the price of
one unit in ticks, `p_eff` (from calibration: chosen so a simulated session has the same relative
volatility as an NSE large-cap day). Capital G = inv_max * p_eff ticks: the value of the largest
position the market maker may hold. Then, per session:
  pnl / G, traded value / G   ->  Sharpe, Turnover, Fitness, Returns (on G), Drawdown (on G), Margin.
Unlike the BRAIN long-short book there is no "half book": Returns and Drawdown are relative to G.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from ..backtest import metrics as M
from .sim import SimResult


def session_series(r: SimResult, p_eff: float) -> pl.DataFrame:
    """Per-session PnL and traded value as fractions of capital G."""
    G = r.params.inv_max * p_eff
    d = r.daily
    eq = d["equity2"].to_numpy() / 2.0                      # ticks * units
    tv = d["mm_traded_value"].to_numpy().astype(float)
    pnl = np.diff(np.concatenate([[0.0], eq]))
    traded = np.diff(np.concatenate([[0.0], tv]))
    return pl.DataFrame({"session": np.arange(len(pnl)), "pnl": pnl / G, "traded": traded / G,
                         "inventory": d["inventory"].to_numpy(), "fills": np.diff(np.concatenate([[0], d["mm_fills"].to_numpy()]))})


FEES_BP = (0.1, 0.3, 1.0, 3.0)   # one-way cost per unit of traded value, for net-of-cost Sharpe


def evaluate(r: SimResult, p_eff: float) -> dict:
    s = session_series(r, p_eff)
    out = M.summary(s["pnl"].to_numpy(), s["traded"].to_numpy(), half_book=1.0)
    for f in FEES_BP:
        out[f"net_sharpe_{f}bp"] = M.sharpe(s["pnl"].to_numpy() - s["traded"].to_numpy() * f * 1e-4)
    f = r.fills
    done = f.drop_nulls("markout2")
    units = max(int(done["qty"].sum()), 1)
    out.update(
        fills_per_session=float(s["fills"].mean()),
        markout_ticks_per_unit=float(done["markout2"].sum()) / 2.0 / units,     # edge kept after markout_h steps
        adverse_fill_share=float((done["markout2"] < 0).mean()) if done.height else float("nan"),
        mean_queue_ahead=float(f["queue_ahead"].mean()) if f.height else float("nan"),
        mean_time_to_fill=float((f["t"] - f["post_t"]).mean()) if f.height else float("nan"),
        mean_abs_inventory=float(np.abs(s["inventory"].to_numpy()).mean()),
        pnl_ticks_total=float(r.daily["equity2"][-1] / 2.0) if r.daily.height else 0.0,
    )
    return out
