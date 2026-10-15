"""Phase 7 checks on MATLAB outputs (skipped unless `python -m alphalab.matlab_parity` has been run)."""
from pathlib import Path

import polars as pl
import pytest

from alphalab.backtest import metrics as M

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "matlab"
EXPORT = ROOT / "reports" / "study" / "export"
pytestmark = pytest.mark.skipif(
    not (OUT / "crosscheck_metrics.csv").exists() or not list(EXPORT.glob("daily_*.parquet")),
    reason="MATLAB outputs or study exports not generated")
needs_wide = pytest.mark.skipif(not (EXPORT / "wide_close.parquet").exists(),
                                reason="wide price matrices are git-ignored; run `python -m alphalab.study`")


def test_matlab_six_metrics_match_python():
    mt = pl.read_csv(OUT / "crosscheck_metrics.csv")
    assert mt.height >= 7
    for r in mt.iter_rows(named=True):
        d = pl.read_parquet(EXPORT / f"daily_{r['family']}.parquet").filter(pl.col("gross") > 0)
        py = M.summary(d["pnl"].to_numpy(), d["traded"].to_numpy())
        for k in ("sharpe", "turnover", "fitness", "returns", "drawdown", "margin_bps"):
            assert py[k] == pytest.approx(r[k], abs=1e-6), (r["family"], k)
        assert int(r["days"]) == py["days"]


@needs_wide
def test_matlab_parfor_equals_serial_and_python_twin():
    from alphalab.matlab_parity import sweep
    ml = pl.read_csv(OUT / "parfor_sweep.csv")
    py, _ = sweep()
    j = ml.join(py, on=["fast", "slow"], suffix="_py")
    assert j.height == ml.height == py.height
    assert (j["sharpe"] - j["sharpe_py"]).abs().max() < 1e-6


def test_matlab_bootstrap_intervals_bracket_point_estimate():
    bt = pl.read_csv(OUT / "bootstrap_sharpe.csv")
    assert ((bt["iid_lo"] < bt["sharpe"]) & (bt["sharpe"] < bt["iid_hi"])).all()
    assert ((bt["block_lo"] < bt["sharpe"]) & (bt["sharpe"] < bt["block_hi"])).all()
