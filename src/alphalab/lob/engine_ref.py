"""Pure-Python reference engine: slow, deliberately simple, used as the correctness oracle.

Structure differs from the numba / Rust engines on purpose (dict of price -> deque,
best price found with min/max) so that a shared bug is unlikely.
"""
from __future__ import annotations

from collections import deque
from math import ceil, floor

import numpy as np

from .spec import (BUF_CAP, CMD_STRIDE, FNV_OFFSET, FNV_PRIME, INIT_REF, MARKOUT_PENDING, MASK64,
                   N_SEED_LEVELS, P, Params)

BUY, SELL = 0, 1
LIMIT, MARKET, CANCEL = 0, 1, 2


class RefBook:
    def __init__(self, markout_h: int = 0):
        self.bids: dict[int, deque] = {}   # price -> deque of seq (FIFO)
        self.asks: dict[int, deque] = {}
        self.orders: dict[int, list] = {}  # seq -> [price, qty, side, owner, post_t, queue_ahead]
        self.hash = FNV_OFFSET
        self.n_fills = self.volume = 0
        self.mm_fills = self.mm_volume = self.mm_inv = self.mm_cash = 0
        self.mm_traded_value = 0
        self.n_cmds = 0
        self.n_orders = 0
        self.last_px = INIT_REF
        self.trades: list[tuple] = []      # (t, price, qty, maker_seq, maker_owner, taker_side)
        self.markout_h = markout_h
        self.fills: list[list] = []        # market-maker fills, FILL_FIELDS layout
        self.pending: deque = deque()      # (due_t, index into fills)

    # ---- book state -------------------------------------------------------
    @property
    def best_bid(self) -> int:
        return max(self.bids) if self.bids else -1

    @property
    def best_ask(self) -> int:
        return min(self.asks) if self.asks else P

    def is_live(self, seq: int) -> bool:
        return seq in self.orders

    # ---- internals --------------------------------------------------------
    def _hash(self, v: int) -> None:
        self.hash = ((self.hash ^ v) * FNV_PRIME) & MASK64

    def _apply(self, t, side: int, price: int, qty: int, post_t: int, qahead: int) -> None:
        self.mm_fills += 1
        self.mm_volume += qty
        self.mm_traded_value += price * qty
        if side == BUY:
            self.mm_inv += qty
            self.mm_cash -= price * qty
        else:
            self.mm_inv -= qty
            self.mm_cash += price * qty
        self.fills.append([t, side, price, qty, post_t, qahead, MARKOUT_PENDING])
        self.pending.append((t + self.markout_h, len(self.fills) - 1))

    def settle_markouts(self, t: int, mid2: int) -> None:
        """Markout of each MM fill `markout_h` steps later, in half-ticks: (mid - price) * qty for a buy."""
        while self.pending and self.pending[0][0] <= t:
            _, i = self.pending.popleft()
            f = self.fills[i]
            f[6] = (mid2 - 2 * f[2]) * f[3] if f[1] == BUY else (2 * f[2] - mid2) * f[3]

    def _fill(self, t, price, qty, maker_seq, taker_side, taker_owner) -> None:
        order = self.orders[maker_seq]
        maker_owner = order[3]
        self.n_fills += 1
        self.volume += qty
        self.last_px = price
        for v in (t, price, qty, maker_seq, maker_owner, taker_side):
            self._hash(v)
        self.trades.append((t, price, qty, maker_seq, maker_owner, taker_side))
        if maker_owner == 1:
            self._apply(t, 1 - taker_side, price, qty, order[4], order[5])
        if taker_owner == 1:
            self._apply(t, taker_side, price, qty, t, -1)

    def _match(self, t, taker_side, limit_price, qty, owner) -> int:
        book = self.asks if taker_side == BUY else self.bids
        while qty > 0 and book:
            px = min(book) if taker_side == BUY else max(book)
            if taker_side == BUY and px > limit_price:
                break
            if taker_side == SELL and px < limit_price:
                break
            q = book[px]
            seq = q[0]
            order = self.orders[seq]
            fq = min(qty, order[1])
            self._fill(t, px, fq, seq, taker_side, owner)
            order[1] -= fq
            qty -= fq
            if order[1] == 0:
                q.popleft()
                del self.orders[seq]
                if not q:
                    del book[px]
        return qty

    # ---- commands ---------------------------------------------------------
    def limit(self, t, side, price, qty, owner, seq) -> None:
        self.n_cmds += 1
        self.n_orders += 1
        rem = self._match(t, side, price, qty, owner)
        if rem > 0:
            book = self.bids if side == BUY else self.asks
            q = book.setdefault(price, deque())
            ahead = sum(self.orders[s][1] for s in q)       # queue ahead at the moment we join
            q.append(seq)
            self.orders[seq] = [price, rem, side, owner, t, ahead]

    def level_qty(self, side: int, price: int) -> int:
        book = self.bids if side == BUY else self.asks
        return sum(self.orders[s][1] for s in book.get(price, ()))

    def market(self, t, side, qty, owner) -> None:
        self.n_cmds += 1
        self._match(t, side, P - 1 if side == BUY else 0, qty, owner)

    def cancel(self, seq: int) -> None:
        self.n_cmds += 1
        order = self.orders.pop(seq, None)
        if order is None:
            return
        price, side = order[0], order[2]
        book = self.bids if side == BUY else self.asks
        book[price].remove(seq)
        if not book[price]:
            del book[price]

    # ---- results ----------------------------------------------------------
    def result(self, live_len: int = 0, error: int = 0) -> np.ndarray:
        return np.array([
            self.hash >> 32, self.hash & 0xFFFFFFFF, self.n_fills, self.volume,
            self.mm_fills, self.mm_volume, self.mm_inv, self.mm_cash, self.n_cmds,
            self.best_bid, self.best_ask, self.n_orders, self.last_px, live_len, error,
        ], dtype=np.int64)


class SplitMix64:
    def __init__(self, seed: int):
        self.s = seed & MASK64

    def next(self) -> int:
        self.s = (self.s + 0x9E3779B97F4A7C15) & MASK64
        z = self.s
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        z ^= z >> 31
        return z >> 33


def run_tape(tape: np.ndarray) -> tuple[np.ndarray, RefBook]:
    """Replay a command tape (flat int32, 7 values per command)."""
    book = RefBook()
    a = tape.reshape(-1, CMD_STRIDE)
    for ts, typ, side, price, qty, seq, owner in a.tolist():
        if typ == LIMIT:
            book.limit(ts, side, price, qty, owner, seq)
        elif typ == MARKET:
            book.market(ts, side, qty, owner)
        else:
            book.cancel(seq)
    return book.result(), book


def run_full(params: Params, record: bool = False):
    """Full loop (backwards-compatible API): returns (res, book) or (res, tape, book)."""
    out = simulate(params, record)
    if record:
        return out["res"], out["tape"], out["book"]
    return out["res"], out["book"]


def simulate(params: Params, record: bool = False) -> dict:
    """Full loop: noise flow (Poisson + discrete Hawkes) + market maker (A-S + EMA/RSI/breakout/S-R/imbalance).

    Returns dict(res, daily (n_days x 8 int64), fills (n_fills x 7 int64), tape (if record), book).
    """
    pi, pf = params.pack()
    n_steps, seed, K, QL, QM = (int(x) for x in pi[:5])
    thr_lim, cx, thr_same, req, warm, qmm, inv_max = (int(x) for x in pi[5:12])
    rsi_n, W, W2, lvl_ticks, day_steps, markout_h = (int(x) for x in pi[12:18])
    qty_inf, anchor_value, thr_value, thr_inf, vjump = (int(x) for x in pi[18:23])
    value = INIT_REF
    mu, rho, alpha_j, af, as_, a_sig, gH, hs = (float(x) for x in pf[:8])
    (ra, a_rsi, a_brk, brk_decay, vol_mult, vol_alpha, k_brk, brk_pull, a_sr, hs_lvl, a_imb) = (float(x) for x in pf[8:19])
    rng = SplitMix64(seed)
    book = RefBook(markout_h)
    tape: list[int] = []
    daily: list[list[int]] = []

    def rec(t, typ, side, price, qty, seq, owner):
        if record:
            tape.extend((t, typ, side, price, qty, seq, owner))

    def do_limit(t, side, price, qty, owner):
        seq = book.n_orders
        rec(t, LIMIT, side, price, qty, seq, owner)
        book.limit(t, side, price, qty, owner, seq)
        return seq

    def do_cancel(t, seq):
        rec(t, CANCEL, 0, 0, 0, seq, 0)
        book.cancel(seq)

    live: list[int] = []
    for k in range(1, N_SEED_LEVELS + 1):
        live.append(do_limit(0, BUY, INIT_REF - k, 10, 0))
        live.append(do_limit(0, SELL, INIT_REF + k, 10, 0))

    e = 0.0
    last_side = 0
    ema_f = ema_s = 0.0
    ema_init = False
    bid_seq = ask_seq = -1
    bid_px = ask_px = 0
    buf: deque = deque(maxlen=BUF_CAP)    # mid2 at previous requotes
    prev_mid = 0.0
    n_diff = 0
    avg_g = avg_l = 0.0
    brk_state = 0.0
    vol_ema = 0.0
    vol_init = False
    last_vol = 0
    last_mid2 = 2 * INIT_REF

    for t in range(n_steps):
        e *= rho
        if thr_value > 0 and rng.next() < thr_value:          # latent value: +-vjump tick random walk
            value = min(max(value + (vjump if rng.next() % 2 == 0 else -vjump), 2), P - 3)
        for side in (BUY, SELL):
            if rng.next() < thr_lim:
                k = 1 + rng.next() % K
                q = 1 + rng.next() % QL
                bb, ba = book.best_bid, book.best_ask
                # symmetric around the mid: buys k below ceil(mid), sells k above floor(mid)
                if anchor_value:
                    lo_anchor = hi_anchor = value
                elif bb >= 0 and ba < P:
                    lo_anchor, hi_anchor = (bb + ba) // 2, (bb + ba + 1) // 2
                else:
                    lo_anchor = hi_anchor = book.last_px
                if side == BUY:
                    price = max(hi_anchor - k, 2)
                else:
                    price = min(lo_anchor + k, P - 3)
                live.append(do_limit(t, side, price, q, 0))
        n = len(live)
        if n > 0:
            thr = min(n * cx, 2147483648)
            if rng.next() < thr:
                idx = rng.next() % n
                seq = live[idx]
                live[idx] = live[n - 1]
                live.pop()
                do_cancel(t, seq)
        lam = mu + e
        if lam > 1.0:
            lam = 1.0
        if rng.next() < int(lam * 2147483648.0):
            e += alpha_j
            side = last_side if rng.next() < thr_same else 1 - last_side
            last_side = side
            q = 1 + rng.next() % QM
            rec(t, MARKET, side, 0, q, 0, 0)
            book.market(t, side, q, 0)
        if thr_inf > 0 and rng.next() < thr_inf:                # informed trader: trade toward the value
            bb, ba = book.best_bid, book.best_ask
            if ba < P and value > ba:
                rec(t, MARKET, BUY, 0, qty_inf, 0, 0)
                book.market(t, BUY, qty_inf, 0)
            elif bb >= 0 and value < bb:
                rec(t, MARKET, SELL, 0, qty_inf, 0, 0)
                book.market(t, SELL, qty_inf, 0)
        bb, ba = book.best_bid, book.best_ask
        if t % req == 0 and t >= warm and bb >= 0 and ba < P:
            mid2 = bb + ba
            mid = mid2 * 0.5
            if not ema_init:
                ema_f = ema_s = mid
                ema_init = True
                prev_mid = mid
            else:
                ema_f += af * (mid - ema_f)
                ema_s += as_ * (mid - ema_s)
                d = mid - prev_mid
                prev_mid = mid
                gain = d if d > 0.0 else 0.0
                loss = -d if d < 0.0 else 0.0
                if n_diff == 0:
                    avg_g, avg_l = gain, loss
                else:
                    avg_g += ra * (gain - avg_g)
                    avg_l += ra * (loss - avg_l)
                n_diff += 1
            rsi_sig = 0.0
            if n_diff >= rsi_n:
                rsi = 100.0 if avg_l == 0.0 else 100.0 - 100.0 / (1.0 + avg_g / avg_l)
                rsi_sig = (50.0 - rsi) / 50.0
            vol_since = book.volume - last_vol
            last_vol = book.volume
            brk = 0
            hist = list(buf)
            if len(hist) >= W:
                win = hist[-W:]
                if mid2 > max(win):
                    brk = 1
                elif mid2 < min(win):
                    brk = -1
                if brk != 0 and vol_mult > 0.0 and not (vol_init and vol_since > vol_mult * vol_ema):
                    brk = 0
            brk_state = brk_state * brk_decay + brk
            if not vol_init:
                vol_ema = float(vol_since)
                vol_init = True
            else:
                vol_ema += vol_alpha * (vol_since - vol_ema)
            sr_sig = 0.0
            near = False
            if len(hist) >= W2:
                win = hist[-W2:]
                s2, r2 = min(win), max(win)
                if r2 > s2:
                    sr_sig = 0.5 - (mid2 - s2) / (r2 - s2)
                if lvl_ticks > 0 and ((r2 - mid2) <= 2 * lvl_ticks or (mid2 - s2) <= 2 * lvl_ticks):
                    near = True
            buf.append(mid2)
            qb, qa = book.level_qty(BUY, bb), book.level_qty(SELL, ba)
            imb_sig = ((qb - qa) / (qb + qa)) * ((ba - bb) * 0.5)
            sig = ema_f - ema_s
            inv = book.mm_inv
            r = ((((mid + a_sig * sig) + a_rsi * rsi_sig) + a_brk * brk_state) + a_sr * sr_sig) + a_imb * imb_sig
            r = r - gH * inv
            hs_eff = (hs + k_brk * abs(brk_state)) + (hs_lvl if near else 0.0)
            bidp = int(floor(r - hs_eff))
            askp = int(ceil(r + hs_eff))
            bidp = min(bidp, ba - 1)
            askp = max(askp, bb + 1)
            bidp = max(bidp, 2)
            askp = min(askp, P - 3)
            pull_bid = brk_pull > 0.0 and brk_state <= -brk_pull
            pull_ask = brk_pull > 0.0 and brk_state >= brk_pull
            # bid
            live_b = bid_seq >= 0 and book.is_live(bid_seq)
            if inv < inv_max and not pull_bid:
                if not (live_b and bid_px == bidp):
                    if live_b:
                        do_cancel(t, bid_seq)
                    bid_seq = do_limit(t, BUY, bidp, qmm, 1)
                    bid_px = bidp
            elif live_b:
                do_cancel(t, bid_seq)
                bid_seq = -1
            # ask
            live_a = ask_seq >= 0 and book.is_live(ask_seq)
            if inv > -inv_max and not pull_ask:
                if not (live_a and ask_px == askp):
                    if live_a:
                        do_cancel(t, ask_seq)
                    ask_seq = do_limit(t, SELL, askp, qmm, 1)
                    ask_px = askp
            elif live_a:
                do_cancel(t, ask_seq)
                ask_seq = -1
        # end-of-step bookkeeping (no randomness, no commands)
        bb, ba = book.best_bid, book.best_ask
        if bb >= 0 and ba < P:
            last_mid2 = bb + ba
        book.settle_markouts(t, last_mid2)
        if (t + 1) % day_steps == 0:
            daily.append([t, last_mid2, 2 * book.mm_cash + book.mm_inv * last_mid2, book.mm_inv,
                          book.mm_traded_value, book.mm_fills, book.mm_volume, book.n_fills])

    out = {
        "res": book.result(live_len=len(live)),
        "daily": np.array(daily, dtype=np.int64).reshape(-1, 8),
        "fills": np.array(book.fills, dtype=np.int64).reshape(-1, 7),
        "book": book,
    }
    if record:
        out["tape"] = np.array(tape, dtype=np.int32)
    return out
