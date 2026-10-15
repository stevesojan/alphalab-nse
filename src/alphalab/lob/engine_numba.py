"""Numba engine (v2): the whole book lives in ONE flat int64 array `M` addressed by constant offsets.

Why: v1 (engine_numba_v1_tuple.py) kept state as a tuple of 14 arrays; every tuple access in numba
takes a new reference with an atomic refcount op, which made the engine ~10-30x slower than Rust.
A single array passed to helpers avoids that. Same model/semantics as the reference and Rust engines.
"""
from __future__ import annotations

import numpy as np
from numba import njit

from .spec import (CAP, CMD_STRIDE, FNV_OFFSET, FNV_PRIME, INIT_REF, LIVE_CAP,
                   N_SEED_LEVELS, P, Params)

_PRIME = np.uint64(FNV_PRIME)
_M31 = 2147483648

# scalar slots
I_BB, I_BA, I_FREE, I_SEQ, I_NFILLS, I_VOL, I_MMF, I_MMV, I_MMINV, I_MMCASH = range(10)
I_NCMDS, I_LIVE, I_ERR, I_LASTPX, I_HASH = 10, 11, 12, 13, 14
# array offsets inside M
O_HEAD = 16
O_TAIL = O_HEAD + P
O_VOL = O_TAIL + P
O_PRICE = O_VOL + P
O_QTY = O_PRICE + CAP
O_SIDE = O_QTY + CAP
O_OWNER = O_SIDE + CAP
O_SEQ = O_OWNER + CAP
O_PREV = O_SEQ + CAP
O_NEXT = O_PREV + CAP
O_FREE = O_NEXT + CAP
O_S2S = O_FREE + CAP


def new_M(max_seq: int) -> np.ndarray:
    M = np.zeros(O_S2S + max_seq, dtype=np.int64)
    M[O_HEAD:O_HEAD + 2 * P] = -1
    M[O_FREE:O_FREE + CAP] = np.arange(CAP)
    M[O_S2S:] = -1
    M[I_BB] = -1
    M[I_BA] = P
    M[I_FREE] = CAP
    M[I_LASTPX] = INIT_REF
    M[I_HASH] = np.array([FNV_OFFSET], dtype=np.uint64).view(np.int64)[0]
    return M


@njit(inline='always')
def _hash(M, v):
    M[I_HASH] = np.int64((np.uint64(M[I_HASH]) ^ np.uint64(v)) * _PRIME)


@njit(inline='always')
def _apply(M, side, price, qty):
    M[I_MMF] += 1
    M[I_MMV] += qty
    if side == 0:
        M[I_MMINV] += qty
        M[I_MMCASH] -= price * qty
    else:
        M[I_MMINV] -= qty
        M[I_MMCASH] += price * qty


@njit(inline='always')
def _fill(M, t, price, qty, slot, taker_side, taker_owner):
    maker_owner = M[O_OWNER + slot]
    M[I_NFILLS] += 1
    M[I_VOL] += qty
    M[I_LASTPX] = price
    _hash(M, t)
    _hash(M, price)
    _hash(M, qty)
    _hash(M, M[O_SEQ + slot])
    _hash(M, maker_owner)
    _hash(M, taker_side)
    if maker_owner == 1:
        _apply(M, 1 - taker_side, price, qty)
    if taker_owner == 1:
        _apply(M, taker_side, price, qty)


@njit(inline='always')
def _fix_best(M, side, price):
    if side == 0:
        if price == M[I_BB]:
            p = price - 1
            while p >= 0 and M[O_VOL + p] == 0:
                p -= 1
            M[I_BB] = p
    else:
        if price == M[I_BA]:
            p = price + 1
            while p < P and M[O_VOL + p] == 0:
                p += 1
            M[I_BA] = p


@njit(inline='always')
def _remove(M, slot):
    price = M[O_PRICE + slot]
    side = M[O_SIDE + slot]
    prv = M[O_PREV + slot]
    nxt = M[O_NEXT + slot]
    if prv >= 0:
        M[O_NEXT + prv] = nxt
    else:
        M[O_HEAD + price] = nxt
    if nxt >= 0:
        M[O_PREV + nxt] = prv
    else:
        M[O_TAIL + price] = prv
    M[O_VOL + price] -= M[O_QTY + slot]
    M[O_QTY + slot] = 0
    M[O_S2S + M[O_SEQ + slot]] = -1
    M[O_FREE + M[I_FREE]] = slot
    M[I_FREE] += 1
    if M[O_VOL + price] == 0:
        _fix_best(M, side, price)


@njit(inline='always')
def _rest(M, side, price, qty, owner, seq):
    if M[I_FREE] == 0:
        M[I_ERR] = 1
        return
    M[I_FREE] -= 1
    slot = M[O_FREE + M[I_FREE]]
    M[O_PRICE + slot] = price
    M[O_QTY + slot] = qty
    M[O_SIDE + slot] = side
    M[O_OWNER + slot] = owner
    M[O_SEQ + slot] = seq
    tail = M[O_TAIL + price]
    M[O_PREV + slot] = tail
    M[O_NEXT + slot] = -1
    if tail >= 0:
        M[O_NEXT + tail] = slot
    else:
        M[O_HEAD + price] = slot
    M[O_TAIL + price] = slot
    M[O_VOL + price] += qty
    M[O_S2S + seq] = slot
    if side == 0:
        if price > M[I_BB]:
            M[I_BB] = price
    else:
        if price < M[I_BA]:
            M[I_BA] = price


@njit(inline='always')
def _match(M, t, taker_side, limit_price, qty, owner):
    if taker_side == 0:
        while qty > 0 and M[I_BA] < P and M[I_BA] <= limit_price:
            px = M[I_BA]
            slot = M[O_HEAD + px]
            fq = min(qty, M[O_QTY + slot])
            _fill(M, t, px, fq, slot, 0, owner)
            M[O_QTY + slot] -= fq
            M[O_VOL + px] -= fq
            qty -= fq
            if M[O_QTY + slot] == 0:
                _remove(M, slot)
    else:
        while qty > 0 and M[I_BB] >= 0 and M[I_BB] >= limit_price:
            px = M[I_BB]
            slot = M[O_HEAD + px]
            fq = min(qty, M[O_QTY + slot])
            _fill(M, t, px, fq, slot, 1, owner)
            M[O_QTY + slot] -= fq
            M[O_VOL + px] -= fq
            qty -= fq
            if M[O_QTY + slot] == 0:
                _remove(M, slot)
    return qty


@njit(inline='always')
def _limit(M, t, side, price, qty, owner, seq):
    M[I_SEQ] += 1
    rem = _match(M, t, side, price, qty, owner)
    if rem > 0:
        _rest(M, side, price, rem, owner, seq)


@njit(inline='always')
def _market(M, t, side, qty, owner):
    if side == 0:
        _match(M, t, 0, P - 1, qty, owner)
    else:
        _match(M, t, 1, 0, qty, owner)


@njit(inline='always')
def _cancel(M, seq):
    slot = M[O_S2S + seq]
    if slot >= 0:
        _remove(M, slot)


@njit(inline='always')
def _rec(rec, M, t, typ, side, price, qty, seq, owner):
    i = M[I_NCMDS]
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
            M[I_ERR] = 2
    M[I_NCMDS] = i + 1


@njit(inline='always')
def _next(R):
    s = R[0] + np.uint64(0x9E3779B97F4A7C15)
    R[0] = s
    z = (s ^ (s >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    z = z ^ (z >> np.uint64(31))
    return np.int64(z >> np.uint64(33))


@njit
def _result(M):
    h = np.uint64(M[I_HASH])
    res = np.zeros(15, np.int64)
    res[0] = np.int64(h >> np.uint64(32))
    res[1] = np.int64(h & np.uint64(0xFFFFFFFF))
    res[2] = M[I_NFILLS]
    res[3] = M[I_VOL]
    res[4] = M[I_MMF]
    res[5] = M[I_MMV]
    res[6] = M[I_MMINV]
    res[7] = M[I_MMCASH]
    res[8] = M[I_NCMDS]
    res[9] = M[I_BB]
    res[10] = M[I_BA]
    res[11] = M[I_SEQ]
    res[12] = M[I_LASTPX]
    res[13] = M[I_LIVE]
    res[14] = M[I_ERR]
    return res


@njit(inline='always')
def _do_limit(M, rec, t, side, price, qty, owner):
    seq = M[I_SEQ]
    _rec(rec, M, t, 0, side, price, qty, seq, owner)
    _limit(M, t, side, price, qty, owner, seq)
    return seq


@njit(inline='always')
def _do_cancel(M, rec, t, seq):
    _rec(rec, M, t, 2, 0, 0, 0, seq, 0)
    _cancel(M, seq)


@njit
def _full(M, pi, pf, rec):
    n_steps = pi[0]
    K, QL, QM = pi[2], pi[3], pi[4]
    thr_lim, cx, thr_same = pi[5], pi[6], pi[7]
    req, warm, qmm, inv_max = pi[8], pi[9], pi[10], pi[11]
    mu, rho, alpha_j = pf[0], pf[1], pf[2]
    af, as_, a_sig, gH, hs = pf[3], pf[4], pf[5], pf[6], pf[7]
    R = np.zeros(1, np.uint64)
    R[0] = np.uint64(pi[1])
    live = np.zeros(LIVE_CAP, np.int32)
    n = 0
    for k in range(1, N_SEED_LEVELS + 1):
        live[n] = _do_limit(M, rec, 0, 0, INIT_REF - k, 10, 0)
        n += 1
        live[n] = _do_limit(M, rec, 0, 1, INIT_REF + k, 10, 0)
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
                bb = M[I_BB]
                ba = M[I_BA]
                if bb >= 0 and ba < P:
                    lo_anchor = (bb + ba) // 2
                    hi_anchor = (bb + ba + 1) // 2
                else:
                    lo_anchor = M[I_LASTPX]
                    hi_anchor = M[I_LASTPX]
                if side == 0:
                    price = max(hi_anchor - k, 2)
                else:
                    price = min(lo_anchor + k, P - 3)
                if n >= LIVE_CAP:
                    M[I_ERR] = 3
                    return
                live[n] = _do_limit(M, rec, t, side, price, q, 0)
                n += 1
        if n > 0:
            thr = min(n * cx, _M31)
            if _next(R) < thr:
                idx = _next(R) % n
                seq = live[idx]
                live[idx] = live[n - 1]
                n -= 1
                _do_cancel(M, rec, t, seq)
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
            _rec(rec, M, t, 1, side, 0, q, 0, 0)
            _market(M, t, side, q, 0)
        bb = M[I_BB]
        ba = M[I_BA]
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
            inv = M[I_MMINV]
            r = (mid + a_sig * sig) - gH * np.float64(inv)
            bidp = np.int64(np.floor(r - hs))
            askp = np.int64(np.ceil(r + hs))
            bidp = min(bidp, ba - 1)
            askp = max(askp, bb + 1)
            bidp = max(bidp, 2)
            askp = min(askp, P - 3)
            live_b = bid_seq >= 0 and M[O_S2S + bid_seq] >= 0
            if inv < inv_max:
                if not (live_b and bid_px == bidp):
                    if live_b:
                        _do_cancel(M, rec, t, bid_seq)
                    bid_seq = _do_limit(M, rec, t, 0, bidp, qmm, 1)
                    bid_px = bidp
            elif live_b:
                _do_cancel(M, rec, t, bid_seq)
                bid_seq = -1
            live_a = ask_seq >= 0 and M[O_S2S + ask_seq] >= 0
            if inv > -inv_max:
                if not (live_a and ask_px == askp):
                    if live_a:
                        _do_cancel(M, rec, t, ask_seq)
                    ask_seq = _do_limit(M, rec, t, 1, askp, qmm, 1)
                    ask_px = askp
            elif live_a:
                _do_cancel(M, rec, t, ask_seq)
                ask_seq = -1
    M[I_LIVE] = n


@njit
def run_full_nb(M, pi, pf, rec):
    _full(M, pi, pf, rec)
    return _result(M)


@njit
def _replay(M, tape):
    n = tape.size // 7
    for i in range(n):
        b = i * 7
        t = np.int64(tape[b])
        typ = tape[b + 1]
        if typ == 0:
            _limit(M, t, np.int64(tape[b + 2]), np.int64(tape[b + 3]), np.int64(tape[b + 4]),
                   np.int64(tape[b + 6]), np.int64(tape[b + 5]))
        elif typ == 1:
            _market(M, t, np.int64(tape[b + 2]), np.int64(tape[b + 4]), np.int64(tape[b + 6]))
        else:
            _cancel(M, np.int64(tape[b + 5]))
        M[I_NCMDS] += 1


@njit
def run_tape_nb(M, tape):
    _replay(M, tape)
    return _result(M)


# ---- Python-facing wrappers ---------------------------------------------------------------
_NO_REC = np.zeros(0, np.int32)


def run_full(params: Params, record: bool = False):
    pi, pf = params.pack()
    max_seq = params.max_seq()
    if not record:
        return run_full_nb(new_M(max_seq), pi, pf, _NO_REC)
    cap_cmds = int(1.0 * params.n_steps) + 5000
    while True:                      # grow the tape buffer until the run fits (error code 2 = overflow)
        rec = np.zeros(cap_cmds * CMD_STRIDE, np.int32)
        res = run_full_nb(new_M(max_seq), pi, pf, rec)
        if res[14] != 2:
            break
        cap_cmds *= 2
    n = int(res[8])
    return res, rec[: n * CMD_STRIDE].copy()


def run_tape(tape: np.ndarray) -> np.ndarray:
    tape = np.ascontiguousarray(tape, dtype=np.int32)
    seqs = tape.reshape(-1, CMD_STRIDE)
    max_seq = int(seqs[:, 5].max()) + 2 if len(seqs) else 2
    return run_tape_nb(new_M(max_seq), tape)


@njit
def _step_cmd(M, t, typ, side, price, qty, seq, owner):
    if typ == 0:
        _limit(M, t, side, price, qty, owner, seq)
    elif typ == 1:
        _market(M, t, side, qty, owner)
    else:
        _cancel(M, seq)
    M[I_NCMDS] += 1


class PyDriven:
    """Python-driven engine: one jitted call per command (measures call overhead)."""

    def __init__(self, max_seq: int):
        self.M = new_M(max_seq)

    def cmd(self, ts, typ, side, price, qty, seq, owner):
        _step_cmd(self.M, ts, typ, side, price, qty, seq, owner)

    def result(self):
        return _result(self.M)
