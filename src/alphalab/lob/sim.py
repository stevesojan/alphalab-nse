"""Python front-end for the Rust market-making simulator (`lob_rs.run_sim`)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from .spec import DAILY_FIELDS, FILL_FIELDS, MARKOUT_PENDING, Params, results_dict


@dataclass
class SimResult:
    params: Params
    res: dict
    daily: pl.DataFrame     # DAILY_FIELDS, one row per simulated session
    fills: pl.DataFrame     # FILL_FIELDS, one row per market-maker fill


def _frames(params, res, daily, fills) -> SimResult:
    d = pl.DataFrame(np.asarray(daily, dtype=np.int64).reshape(-1, len(DAILY_FIELDS)), schema=list(DAILY_FIELDS), orient="row")
    f = pl.DataFrame(np.asarray(fills, dtype=np.int64).reshape(-1, len(FILL_FIELDS)), schema=list(FILL_FIELDS), orient="row")
    f = f.with_columns(pl.when(pl.col("markout2") == MARKOUT_PENDING).then(None).otherwise(pl.col("markout2")).alias("markout2"))
    return SimResult(params, results_dict(res), d, f)


def run(params: Params) -> SimResult:
    """Run the simulator in Rust."""
    import lob_rs
    pi, pf = params.pack()
    res, daily, fills = lob_rs.run_sim(pi, pf, params.max_seq())
    return _frames(params, res, daily, fills)


def run_reference(params: Params) -> SimResult:
    """Same simulation in the pure-Python oracle (slow; for tests and small runs)."""
    from . import engine_ref
    out = engine_ref.simulate(params)
    return _frames(params, out["res"], out["daily"], out["fills"])
