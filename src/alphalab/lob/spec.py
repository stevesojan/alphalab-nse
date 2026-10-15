"""Shared specification for the limit-order-book engines (reference / numba / Rust).

All three implementations must produce *bit-identical* results for the same
Params. To make that possible without relying on libm transcendental functions
(exp/ln can differ in the last ulp across languages) the simulation uses:

* an integer PRNG (splitmix64) -> every draw is the top 31 bits of a u64,
* discrete-time Poisson / Hawkes arrivals (Bernoulli per 1 ms step; the Hawkes
  excitation decays by a constant factor ``rho`` per step, so only * and + are
  needed),
* Avellaneda-Stoikov quoting with constants pre-computed in Python (``hs``,
  ``gH``), so the loop itself needs only + - * / floor ceil.

Conventions: side 0 = buy, 1 = sell. Prices are integer ticks in [0, P).
Command types: 0 limit, 1 market, 2 cancel. Owner 0 = noise flow, 1 = market maker.
Command tape layout: flat int32 array, 7 values per command:
``ts, typ, side, price, qty, seq, owner`` (for a cancel ``seq`` is the target order).
Result vector layout (int64, see RES_FIELDS).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

P = 65536                 # number of price levels (ticks)
INIT_REF = 32768          # initial reference price (ticks)
CAP = 1 << 17             # order-pool capacity (live orders)
LIVE_CAP = 1 << 20        # capacity of the flow's cancel-candidate list
N_SEED_LEVELS = 20        # initial book: 20 levels each side, qty 10
CMD_STRIDE = 7

FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3
MASK64 = (1 << 64) - 1

BUF_CAP = 512             # mid-price history ring buffer (requote samples) for breakout / S-R windows
MARKOUT_PENDING = -(1 << 63)   # fill markout not yet known (run ended before the horizon)

# Simulator outputs (Phase 6e). Daily snapshot taken at the end of every `day_steps` steps.
DAILY_FIELDS = ("t", "mid2", "equity2", "inventory", "mm_traded_value", "mm_fills", "mm_volume", "n_fills")
# One row per market-maker fill. price in ticks; mid2/equity2/markout2 in half-ticks (x2) to stay integer.
FILL_FIELDS = ("t", "side", "price", "qty", "post_t", "queue_ahead", "markout2")

RES_FIELDS = (
    "hash_hi", "hash_lo", "n_fills", "volume", "mm_fills", "mm_volume",
    "mm_inventory", "mm_cash", "n_cmds", "best_bid", "best_ask", "n_orders",
    "last_px", "live_len", "error",
)
R_LIVE_LEN = RES_FIELDS.index("live_len")


def thr31(p: float) -> int:
    """Probability -> threshold on a 31-bit uniform draw."""
    return int(p * 2147483648.0)


@dataclass
class Params:
    n_steps: int = 100_000
    seed: int = 12345
    # noise flow
    p_limit: float = 0.15        # per side, per step
    cancel_hazard: float = 0.0015  # per live order, per step
    k_range: int = 16            # limit offset 1..K ticks from mid
    qty_limit: int = 10          # limit qty 1..QL
    qty_market: int = 10         # market qty 1..QM
    # Hawkes market-order intensity (per step)
    mu: float = 0.02
    rho: float = 0.98            # excitation decay per step
    alpha_jump: float = 0.01     # excitation jump per market order (branching 0.5)
    p_same: float = 0.6          # prob. next market order has the same side
    # market maker (Avellaneda-Stoikov + EMA signal)
    req: int = 10                # requote every `req` steps
    warm: int = 1000
    qty_mm: int = 5
    inv_max: int = 50
    ema_fast: float = 0.2        # EMA alphas, applied at each requote
    ema_slow: float = 0.02
    a_sig: float = 1.0           # weight on (ema_fast - ema_slow), in ticks
    gH: float = 0.05             # gamma * sigma^2 * horizon: ticks of skew per unit inventory
    hs: float = 1.0              # half-spread in ticks
    # ---- Phase 6e signal / quoting features (all OFF by default -> identical to the benchmarked model) ----
    rsi_n: int = 14              # RSI length in requote samples (Wilder smoothing alpha = 1/rsi_n)
    a_rsi: float = 0.0           # ticks of skew per unit of (50 - RSI)/50
    brk_window: int = 50         # breakout look-back in requote samples (prior max/min of mid)
    a_brk: float = 0.0           # ticks of skew per unit of breakout state
    brk_decay: float = 0.9       # breakout state decay per requote
    vol_mult: float = 0.0        # breakout needs traded volume since last requote > vol_mult x its EMA (0 = off)
    vol_alpha: float = 0.1       # EMA alpha for that volume baseline
    k_brk: float = 0.0           # extra half-spread (ticks) per unit |breakout state|
    brk_pull: float = 0.0        # if > 0: stop quoting the side being run over when |state| >= brk_pull
    sr_window: int = 100         # support/resistance look-back in requote samples
    a_sr: float = 0.0            # ticks of skew per unit of (0.5 - channel position)
    lvl_ticks: int = 0           # "near a level" if within this many ticks of support/resistance (0 = off)
    hs_lvl: float = 0.0          # extra half-spread (ticks) near a level
    a_imb: float = 0.0           # weight on the microprice offset: imbalance * spread / 2 (1.0 = quote around microprice)
    # ---- informed flow (OFF by default) ----
    # Without these the flow is pure noise and the mid only bounces mechanically, which lets a market maker
    # that fades every move earn absurd Sharpe ratios (found in the first ablation). With them a latent
    # fundamental value V follows a +-1 tick random walk, noise limit orders are placed around V, and
    # informed traders send market orders whenever V is outside the best quotes -> real adverse selection.
    p_value: float = 0.0         # prob. per step that V jumps (up or down with equal prob.)
    value_jump: int = 1          # size of a V jump in ticks; jumps larger than the maker's half-spread create pick-off risk
    anchor_value: int = 0        # 1: noise limit orders anchor on V instead of the current mid
    p_informed: float = 0.0      # prob. per step that an informed trader acts
    qty_informed: int = 5
    # ---- outputs ----
    day_steps: int = 100_000     # steps per simulated session ("day") for daily snapshots
    markout_h: int = 500         # steps after a fill at which its markout is measured

    def pack(self) -> tuple[np.ndarray, np.ndarray]:
        pi = np.zeros(32, dtype=np.int64)
        pi[0] = self.n_steps
        pi[1] = self.seed
        pi[2] = self.k_range
        pi[3] = self.qty_limit
        pi[4] = self.qty_market
        pi[5] = thr31(self.p_limit)
        pi[6] = thr31(self.cancel_hazard)
        pi[7] = thr31(self.p_same)
        pi[8] = self.req
        pi[9] = self.warm
        pi[10] = self.qty_mm
        pi[11] = self.inv_max
        assert 1 <= self.brk_window <= BUF_CAP and 1 <= self.sr_window <= BUF_CAP and self.rsi_n >= 1
        pi[12] = self.rsi_n
        pi[13] = self.brk_window
        pi[14] = self.sr_window
        pi[15] = self.lvl_ticks
        pi[16] = self.day_steps
        pi[17] = self.markout_h
        pi[18] = self.qty_informed
        pi[19] = self.anchor_value
        pi[20] = thr31(self.p_value)
        pi[21] = thr31(self.p_informed)
        pi[22] = self.value_jump
        pf = np.zeros(32, dtype=np.float64)
        pf[0] = self.mu
        pf[1] = self.rho
        pf[2] = self.alpha_jump
        pf[3] = self.ema_fast
        pf[4] = self.ema_slow
        pf[5] = self.a_sig
        pf[6] = self.gH
        pf[7] = self.hs
        pf[8] = 1.0 / self.rsi_n
        pf[9] = self.a_rsi
        pf[10] = self.a_brk
        pf[11] = self.brk_decay
        pf[12] = self.vol_mult
        pf[13] = self.vol_alpha
        pf[14] = self.k_brk
        pf[15] = self.brk_pull
        pf[16] = self.a_sr
        pf[17] = self.hs_lvl
        pf[18] = self.a_imb
        return pi, pf

    def max_seq(self) -> int:
        """Upper bound on order sequence ids created in a run."""
        return 2 * self.n_steps + 2 * (self.n_steps // max(self.req, 1) + 2) + 2 * N_SEED_LEVELS + 64


def results_dict(res) -> dict:
    return dict(zip(RES_FIELDS, (int(x) for x in res)))


def comparable(res) -> list[int]:
    """Result fields that must match across engines (live_len is full-loop only)."""
    return [int(x) for i, x in enumerate(res) if i != R_LIVE_LEN]
