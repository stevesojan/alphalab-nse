"""Numba engine: array-based order book (per-level doubly linked FIFO), flow generator and
Avellaneda-Stoikov market maker, all compiled into one loop (`run_full`) or replayed from a
command tape (`run_tape`). `step_cmd` exposes a single command for Python-driven benchmarks.
"""
from __future__ import annotations

import numpy as np
from numba import njit

from .spec import (CAP, CMD_STRIDE, FNV_OFFSET, FNV_PRIME, INIT_REF, LIVE_CAP,
                   N_SEED_LEVELS, P, Params)

_PRIME = np.uint64(FNV_PRIME)
_OFFSET = np.uint64(FNV_OFFSET)
_M31 = 2147483648

# S indices: 0 best_bid, 1 best_ask, 2 free_top, 3 seq_counter, 4 n_fills, 5 volume,
# 6 mm_fills, 7 mm_volume, 8 mm_inv, 9 mm_cash, 10 n_cmds, 11 live_len, 12 error, 13 last_px


def new_state(max_seq: int):
    free = np.arange(CAP, dtype=np.int32)
    S = np.zeros(16, dtype=np.int64)
    S[0] = -1
    S[1] = P
    S[2] = CAP
    S[13] = INIT_REF
    H = np.array([FNV_OFFSET], dtype=np.uint64)
    return (
        np.full(P, -1, np.int32), np.full(P, -1, np.int32), np.zeros(P, np.int64),
        np.zeros(CAP, np.int32), np.zeros(CAP, np.int32), np.zeros(CAP, np.int32),
        np.zeros(CAP, np.int32), np.zeros(CAP, np.int32), np.zeros(CAP, np.int32),
        np.zeros(CAP, np.int32), free, np.full(max_seq, -1, np.int32), S, H,
    )


@njit
def _hash(H, v):
    H[0] = (H[0] ^ np.uint64(v)) * _PRIME


@njit
def _apply(S, side, price, qty):
    S[6] += 1
    S[7] += qty
    if side == 0:
        S[8] += qty
        S[9] -= price * qty
    else:
        S[8] -= qty
        S[9] += price * qty


@njit
def _fill(st, t, price, qty, slot, taker_side, taker_owner):
    o_owner = st[6]
    o_seq = st[7]
    S = st[12]
    H = st[13]
    maker_owner = o_owner[slot]
    S[4] += 1
    S[5] += qty
    S[13] = price
    _hash(H, t)
    _hash(H, price)
    _hash(H, qty)
    _hash(H, o_seq[slot])
    _hash(H, maker_owner)
    _hash(H, taker_side)
    if maker_owner == 1:
        _apply(S, 1 - taker_side, price, qty)
    if taker_owner == 1:
        _apply(S, taker_side, price, qty)


@njit
def _fix_best(st, side, price):
    lvl_vol = st[2]
    S = st[12]
    if side == 0:
        if price == S[0]:
            p = price - 1
            while p >= 0 and lvl_vol[p] == 0:
                p -= 1
            S[0] = p
    else:
        if price == S[1]:
            p = price + 1
            while p < P and lvl_vol[p] == 0:
                p += 1
            S[1] = p


@njit
def _remove(st, slot):
    lvl_head, lvl_tail, lvl_vol = st[0], st[1], st[2]
    o_price, o_qty, o_side, o_seq = st[3], st[4], st[5], st[7]
    o_prev, o_next, free, seq2slot, S = st[8], st[9], st[10], st[11], st[12]
    price = o_price[slot]
    side = o_side[slot]
    prv = o_prev[slot]
    nxt = o_next[slot]
    if prv >= 0:
        o_next[prv] = nxt
    else:
        lvl_head[price] = nxt
    if nxt >= 0:
        o_prev[nxt] = prv
    else:
        lvl_tail[price] = prv
    lvl_vol[price] -= o_qty[slot]
    o_qty[slot] = 0
    seq2slot[o_seq[slot]] = -1
    free[S[2]] = slot
    S[2] += 1
    if lvl_vol[price] == 0:
        _fix_best(st, side, price)


@njit
def _rest(st, side, price, qty, owner, seq):
    lvl_head, lvl_tail, lvl_vol = st[0], st[1], st[2]
    o_price, o_qty, o_side, o_owner, o_seq = st[3], st[4], st[5], st[6], st[7]
    o_prev, o_next, free, seq2slot, S = st[8], st[9], st[10], st[11], st[12]
    if S[2] == 0:
        S[12] = 1
        return
    S[2] -= 1
    slot = free[S[2]]
    o_price[slot] = price
    o_qty[slot] = qty
    o_side[slot] = side
    o_owner[slot] = owner
    o_seq[slot] = seq
    tail = lvl_tail[price]
    o_prev[slot] = tail
    o_next[slot] = -1
    if tail >= 0:
        o_next[tail] = slot
    else:
        lvl_head[price] = slot
    lvl_tail[price] = slot
    lvl_vol[price] += qty
    seq2slot[seq] = slot
    if side == 0:
        if price > S[0]:
            S[0] = price
    else:
        if price < S[1]:
            S[1] = price


@njit
def _match(st, t, taker_side, limit_price, qty, owner):
    lvl_head, lvl_vol, o_qty, S = st[0], st[2], st[4], st[12]
    if taker_side == 0:
        while qty > 0 and S[1] < P and S[1] <= limit_price:
            px = S[1]
            slot = lvl_head[px]
            fq = min(qty, np.int64(o_qty[slot]))
            _fill(st, t, px, fq, slot, 0, owner)
            o_qty[slot] -= fq
            lvl_vol[px] -= fq
            qty -= fq
            if o_qty[slot] == 0:
                _remove(st, slot)
    else:
        while qty > 0 and S[0] >= 0 and S[0] >= limit_price:
            px = S[0]
            slot = lvl_head[px]
            fq = min(qty, np.int64(o_qty[slot]))
            _fill(st, t, px, fq, slot, 1, owner)
            o_qty[slot] -= fq
            lvl_vol[px] -= fq
            qty -= fq
            if o_qty[slot] == 0:
                _remove(st, slot)
    return qty


@njit
def _limit(st, t, side, price, qty, owner, seq):
    S = st[12]
    S[3] += 1
    rem = _match(st, t, side, price, qty, owner)
    if rem > 0:
        _rest(st, side, price, rem, owner, seq)


@njit
def _market(st, t, side, qty, owner):
    if side == 0:
        _match(st, t, 0, P - 1, qty, owner)
    else:
        _match(st, t, 1, 0, qty, owner)


@njit
def _cancel(st, seq):
    seq2slot = st[11]
    slot = seq2slot[seq]
    if slot >= 0:
        _remove(st, slot)


@njit
def _rec(rec, S, t, typ, side, price, qty, seq, owner):
    i = S[10]
    if rec.size > 0:
        if (i + 1) * 7 <= rec.size:
            b = i * 7
            rec[b] = t
            rec[b + 1] = typ
            rec[b + 2] = side
            rec[b + 3] = price
            rec[b + 4] = qty
            rec[b + 5] = seq
            rec[b + 6] = owner
        else:
            S[12] = 2
    S[10] = i + 1


@njit
def _next(R):
    s = R[0] + np.uint64(0x9E3779B97F4A7C15)
    R[0] = s
    z = (s ^ (s >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    z = z ^ (z >> np.uint64(31))
    return np.int64(z >> np.uint64(33))


@njit
def _result(st):
    S = st[12]
    H = st[13]
    res = np.zeros(15, np.int64)
    res[0] = np.int64(H[0] >> np.uint64(32))
    res[1] = np.int64(H[0] & np.uint64(0xFFFFFFFF))
    res[2] = S[4]
    res[3] = S[5]
    res[4] = S[6]
    res[5] = S[7]
    res[6] = S[8]
    res[7] = S[9]
    res[8] = S[10]
    res[9] = S[0]
    res[10] = S[1]
    res[11] = S[3]
    res[12] = S[13]
    res[13] = S[11]
    res[14] = S[12]
    return res


@njit
def _do_limit(st, rec, t, side, price, qty, owner):
    S = st[12]
    seq = S[3]
    _rec(rec, S, t, 0, side, price, qty, seq, owner)
    _limit(st, t, side, price, qty, owner, seq)
    return seq


@njit
def _do_cancel(st, rec, t, seq):
    S = st[12]
    _rec(rec, S, t, 2, 0, 0, 0, seq, 0)
    _cancel(st, seq)


@njit
def _full(pi, pf, rec, st):
    n_steps = pi[0]
    K, QL, QM = pi[2], pi[3], pi[4]
    thr_lim, cx, thr_same = pi[5], pi[6], pi[7]
    req, warm, qmm, inv_max = pi[8], pi[9], pi[10], pi[11]
    mu, rho, alpha_j = pf[0], pf[1], pf[2]
    af, as_, a_sig, gH, hs = pf[3], pf[4], pf[5], pf[6], pf[7]
    S = st[12]
    seq2slot = st[11]
    R = np.zeros(1, np.uint64)
    R[0] = np.uint64(pi[1])
    live = np.zeros(LIVE_CAP, np.int32)
    n = 0
    for k in range(1, N_SEED_LEVELS + 1):
        live[n] = _do_limit(st, rec, 0, 0, INIT_REF - k, 10, 0)
        n += 1
        live[n] = _do_limit(st, rec, 0, 1, INIT_REF + k, 10, 0)
        n += 1
    e = 0.0
    last_side = 0
    ema_f = 0.0
    ema_s = 0.0
    ema_init = False
    bid_seq = -1
    ask_seq = -1
    bid_px = 0
    ask_px = 0
    for t in range(n_steps):
        e *= rho
        for side in range(2):
            if _next(R) < thr_lim:
                k = 1 + _next(R) % K
                q = 1 + _next(R) % QL
                bb = S[0]
                ba = S[1]
                if bb >= 0 and ba < P:
                    anchor = (bb + ba) // 2
                else:
                    anchor = S[13]
                if side == 0:
                    price = max(anchor - k, 2)
                else:
                    price = min(anchor + k, P - 3)
                if n >= LIVE_CAP:
                    S[12] = 3
                    return
                live[n] = _do_limit(st, rec, t, side, price, q, 0)
                n += 1
        if n > 0:
            thr = min(n * cx, _M31)
            if _next(R) < thr:
                idx = _next(R) % n
                seq = live[idx]
                live[idx] = live[n - 1]
                n -= 1
                _do_cancel(st, rec, t, seq)
        lam = mu + e
        if lam > 1.0:
            lam = 1.0
        if _next(R) < np.int64(lam * 2147483648.0):
            e += alpha_j
            if _next(R) < thr_same:
                side = last_side
            else:
                side = 1 - last_side
            last_side = side
            q = 1 + _next(R) % QM
            _rec(rec, S, t, 1, side, 0, q, 0, 0)
            _market(st, t, side, q, 0)
        bb = S[0]
        ba = S[1]
        if t % req == 0 and t >= warm and bb >= 0 and ba < P:
            mid = (bb + ba) * 0.5
            if not ema_init:
                ema_f = mid
                ema_s = mid
                ema_init = True
            else:
                ema_f += af * (mid - ema_f)
                ema_s += as_ * (mid - ema_s)
            sig = ema_f - ema_s
            inv = S[8]
            r = (mid + a_sig * sig) - gH * np.float64(inv)
            bidp = np.int64(np.floor(r - hs))
            askp = np.int64(np.ceil(r + hs))
            bidp = min(bidp, ba - 1)
            askp = max(askp, bb + 1)
            bidp = max(bidp, 2)
            askp = min(askp, P - 3)
            live_b = bid_seq >= 0 and seq2slot[bid_seq] >= 0
            if inv < inv_max:
                if not (live_b and bid_px == bidp):
                    if live_b:
                        _do_cancel(st, rec, t, bid_seq)
                    bid_seq = _do_limit(st, rec, t, 0, bidp, qmm, 1)
                    bid_px = bidp
            elif live_b:
                _do_cancel(st, rec, t, bid_seq)
                bid_seq = -1
            live_a = ask_seq >= 0 and seq2slot[ask_seq] >= 0
            if inv > -inv_max:
                if not (live_a and ask_px == askp):
                    if live_a:
                        _do_cancel(st, rec, t, ask_seq)
                    ask_seq = _do_limit(st, rec, t, 1, askp, qmm, 1)
                    ask_px = askp
            elif live_a:
                _do_cancel(st, rec, t, ask_seq)
                ask_seq = -1
    S[11] = n


@njit
def run_full_nb(pi, pf, rec, max_seq):
    st = _new_state_nb(max_seq)
    _full(pi, pf, rec, st)
    return _result(st)


@njit
def _new_state_nb(max_seq):
    free = np.arange(CAP).astype(np.int32)
    S = np.zeros(16, np.int64)
    S[0] = -1
    S[1] = P
    S[2] = CAP
    S[13] = INIT_REF
    H = np.zeros(1, np.uint64)
    H[0] = _OFFSET
    return (
        np.full(P, -1, np.int32), np.full(P, -1, np.int32), np.zeros(P, np.int64),
        np.zeros(CAP, np.int32), np.zeros(CAP, np.int32), np.zeros(CAP, np.int32),
        np.zeros(CAP, np.int32), np.zeros(CAP, np.int32), np.zeros(CAP, np.int32),
        np.zeros(CAP, np.int32), free, np.full(max_seq, -1, np.int32), S, H,
    )


@njit
def _replay(tape, n, st):
    for i in range(n):
        b = i * 7
        t = np.int64(tape[b])
        typ = tape[b + 1]
        if typ == 0:
            _limit(st, t, np.int64(tape[b + 2]), np.int64(tape[b + 3]), np.int64(tape[b + 4]),
                   np.int64(tape[b + 6]), np.int64(tape[b + 5]))
        elif typ == 1:
            _market(st, t, np.int64(tape[b + 2]), np.int64(tape[b + 4]), np.int64(tape[b + 6]))
        else:
            _cancel(st, np.int64(tape[b + 5]))
        st[12][10] += 1


@njit
def run_tape_nb(tape, max_seq):
    st = _new_state_nb(max_seq)
    _replay(tape, tape.size // 7, st)
    return _result(st)


# ---- Python-facing wrappers ---------------------------------------------------------------
_NO_REC = np.zeros(0, np.int32)


def run_full(params: Params, record: bool = False):
    pi, pf = params.pack()
    max_seq = params.max_seq()
    if not record:
        return run_full_nb(pi, pf, _NO_REC, max_seq)
    cap_cmds = int(1.0 * params.n_steps) + 5000
    while True:                      # grow the tape buffer until the run fits (error code 2 = overflow)
        rec = np.zeros(cap_cmds * CMD_STRIDE, np.int32)
        res = run_full_nb(pi, pf, rec, max_seq)
        if res[14] != 2:
            break
        cap_cmds *= 2
    n = int(res[8])
    return res, rec[: n * CMD_STRIDE].copy()


def run_tape(tape: np.ndarray) -> np.ndarray:
    tape = np.ascontiguousarray(tape, dtype=np.int32)
    seqs = tape.reshape(-1, CMD_STRIDE)
    max_seq = int(seqs[:, 5].max()) + 2 if len(seqs) else 2
    return run_tape_nb(tape, max_seq)


class PyDriven:
    """Python-driven engine: one jitted call per command (measures call overhead)."""

    def __init__(self, max_seq: int):
        self.st = new_state(max_seq)

    def cmd(self, ts, typ, side, price, qty, seq, owner):
        _step_cmd(self.st, ts, typ, side, price, qty, seq, owner)

    def result(self):
        return _result(self.st)


@njit
def _step_cmd(st, t, typ, side, price, qty, seq, owner):
    if typ == 0:
        _limit(st, t, side, price, qty, owner, seq)
    elif typ == 1:
        _market(st, t, side, qty, owner)
    else:
        _cancel(st, seq)
    st[12][10] += 1
