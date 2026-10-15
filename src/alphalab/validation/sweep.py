"""Parameter sweeps and plateau selection.

The WorldQuant window sweep (63/90/105/111/120/180/252) showed the right instinct: prefer a parameter
whose *neighbours* also work over the single best point. `plateau_select` scores each grid point by the
worst value of the metric over itself and its grid neighbours (maximin), tie-broken by the neighbourhood
mean, and picks the best score -- not the best raw value. (A plain neighbourhood mean was rejected: an
edge point next to a lucky spike inherits the spike.)
"""
from __future__ import annotations

import itertools

import numpy as np
import polars as pl

from ..backtest import metrics as M


def grid(**axes) -> list[dict]:
    keys = list(axes)
    return [dict(zip(keys, v)) for v in itertools.product(*(axes[k] for k in keys))]


def neighbours(params: dict, axes: dict) -> list[dict]:
    """Grid points that differ from `params` by one step along one axis."""
    out = []
    for k, vals in axes.items():
        i = list(vals).index(params[k])
        for j in (i - 1, i + 1):
            if 0 <= j < len(vals):
                out.append({**params, k: vals[j]})
    return out


def period_metrics(daily: pl.DataFrame, start, end) -> dict:
    d = daily.filter((pl.col("date") >= start) & (pl.col("date") <= end))
    return M.summary(d["pnl"].to_numpy(), d["traded"].to_numpy())


def plateau_select(results: dict[tuple, pl.DataFrame], axes: dict, start, end, metric: str = "sharpe") -> tuple[dict, pl.DataFrame]:
    """results: {tuple(param values in axes order): daily pnl frame}. Returns (chosen params, score table)."""
    keys = list(axes)
    raw = {k: period_metrics(v, start, end)[metric] for k, v in results.items()}
    rows = []
    for k, val in raw.items():
        p = dict(zip(keys, k))
        nb = [raw[tuple(n[x] for x in keys)] for n in neighbours(p, axes)]
        rows.append({**p, metric: val, "plateau": float(min([val] + nb)), "neighbourhood_mean": float(np.mean([val] + nb))})
    table = pl.DataFrame(rows).sort(["plateau", "neighbourhood_mean"], descending=True)
    best = {k: table[k][0] for k in keys}
    return best, table
