//! Rust limit-order-book engine + market-making simulator (PyO3 + maturin).
//! Same model as the Python reference (`engine_ref.py`): array-based book with per-level doubly linked
//! FIFO queues (price-time priority), discrete-time Poisson/Hawkes flow, Avellaneda-Stoikov market maker
//! with EMA / RSI / breakout / support-resistance / imbalance signals. Must produce bit-identical results
//! to the reference (see spec.py): only + - * / floor ceil on floats inside the loop, integer PRNG.
use std::collections::VecDeque;

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

const P: i64 = 65536;
const INIT_REF: i64 = 32768;
const CAP: usize = 1 << 17;
const LIVE_CAP: usize = 1 << 20;
const N_SEED: i64 = 20;
const BUF_CAP: usize = 512;
const FNV_OFFSET: u64 = 0xcbf29ce484222325;
const FNV_PRIME: u64 = 0x100000001b3;
const M31: i64 = 2147483648;
const MARKOUT_PENDING: i64 = i64::MIN;

struct Engine {
    lvl_head: Vec<i32>,
    lvl_tail: Vec<i32>,
    lvl_vol: Vec<i64>,
    o_price: Vec<i32>,
    o_qty: Vec<i32>,
    o_side: Vec<i32>,
    o_owner: Vec<i32>,
    o_seq: Vec<i32>,
    o_prev: Vec<i32>,
    o_next: Vec<i32>,
    o_post_t: Vec<i64>,
    o_qahead: Vec<i64>,
    free: Vec<i32>,
    free_top: usize,
    seq2slot: Vec<i32>,
    bb: i64,
    ba: i64,
    seq_counter: i64,
    n_fills: i64,
    volume: i64,
    mm_fills: i64,
    mm_volume: i64,
    mm_inv: i64,
    mm_cash: i64,
    mm_traded_value: i64,
    n_cmds: i64,
    error: i64,
    last_px: i64,
    hash: u64,
    // market-maker fill log (FILL_FIELDS: t, side, price, qty, post_t, queue_ahead, markout2)
    collect: bool,
    markout_h: i64,
    fills: Vec<i64>,
    pending: VecDeque<(i64, usize)>,
}

impl Engine {
    fn new(max_seq: usize) -> Self {
        Engine {
            lvl_head: vec![-1; P as usize],
            lvl_tail: vec![-1; P as usize],
            lvl_vol: vec![0; P as usize],
            o_price: vec![0; CAP],
            o_qty: vec![0; CAP],
            o_side: vec![0; CAP],
            o_owner: vec![0; CAP],
            o_seq: vec![0; CAP],
            o_prev: vec![0; CAP],
            o_next: vec![0; CAP],
            o_post_t: vec![0; CAP],
            o_qahead: vec![0; CAP],
            free: (0..CAP as i32).collect(),
            free_top: CAP,
            seq2slot: vec![-1; max_seq],
            bb: -1,
            ba: P,
            seq_counter: 0,
            n_fills: 0,
            volume: 0,
            mm_fills: 0,
            mm_volume: 0,
            mm_inv: 0,
            mm_cash: 0,
            mm_traded_value: 0,
            n_cmds: 0,
            error: 0,
            last_px: INIT_REF,
            hash: FNV_OFFSET,
            collect: false,
            markout_h: 0,
            fills: Vec::new(),
            pending: VecDeque::new(),
        }
    }

    #[inline(always)]
    fn hash_in(&mut self, v: i64) {
        self.hash = (self.hash ^ (v as u64)).wrapping_mul(FNV_PRIME);
    }

    #[inline(always)]
    fn apply(&mut self, t: i64, side: i64, price: i64, qty: i64, post_t: i64, qahead: i64) {
        self.mm_fills += 1;
        self.mm_volume += qty;
        self.mm_traded_value += price * qty;
        if side == 0 {
            self.mm_inv += qty;
            self.mm_cash -= price * qty;
        } else {
            self.mm_inv -= qty;
            self.mm_cash += price * qty;
        }
        if self.collect {
            self.fills.extend_from_slice(&[t, side, price, qty, post_t, qahead, MARKOUT_PENDING]);
            self.pending.push_back((t + self.markout_h, self.fills.len() / 7 - 1));
        }
    }

    fn settle_markouts(&mut self, t: i64, mid2: i64) {
        while let Some(&(due, i)) = self.pending.front() {
            if due > t {
                break;
            }
            self.pending.pop_front();
            let b = i * 7;
            let (side, price, qty) = (self.fills[b + 1], self.fills[b + 2], self.fills[b + 3]);
            self.fills[b + 6] = if side == 0 { (mid2 - 2 * price) * qty } else { (2 * price - mid2) * qty };
        }
    }

    #[inline(always)]
    fn fill(&mut self, t: i64, price: i64, qty: i64, slot: usize, taker_side: i64, taker_owner: i64) {
        let maker_owner = self.o_owner[slot] as i64;
        self.n_fills += 1;
        self.volume += qty;
        self.last_px = price;
        self.hash_in(t);
        self.hash_in(price);
        self.hash_in(qty);
        self.hash_in(self.o_seq[slot] as i64);
        self.hash_in(maker_owner);
        self.hash_in(taker_side);
        if maker_owner == 1 {
            let (pt, qa) = (self.o_post_t[slot], self.o_qahead[slot]);
            self.apply(t, 1 - taker_side, price, qty, pt, qa);
        }
        if taker_owner == 1 {
            self.apply(t, taker_side, price, qty, t, -1);
        }
    }

    #[inline(always)]
    fn fix_best(&mut self, side: i64, price: i64) {
        if side == 0 {
            if price == self.bb {
                let mut p = price - 1;
                while p >= 0 && self.lvl_vol[p as usize] == 0 {
                    p -= 1;
                }
                self.bb = p;
            }
        } else if price == self.ba {
            let mut p = price + 1;
            while p < P && self.lvl_vol[p as usize] == 0 {
                p += 1;
            }
            self.ba = p;
        }
    }

    #[inline(always)]
    fn remove(&mut self, slot: usize) {
        let price = self.o_price[slot] as usize;
        let side = self.o_side[slot] as i64;
        let prv = self.o_prev[slot];
        let nxt = self.o_next[slot];
        if prv >= 0 {
            self.o_next[prv as usize] = nxt;
        } else {
            self.lvl_head[price] = nxt;
        }
        if nxt >= 0 {
            self.o_prev[nxt as usize] = prv;
        } else {
            self.lvl_tail[price] = prv;
        }
        self.lvl_vol[price] -= self.o_qty[slot] as i64;
        self.o_qty[slot] = 0;
        self.seq2slot[self.o_seq[slot] as usize] = -1;
        self.free[self.free_top] = slot as i32;
        self.free_top += 1;
        if self.lvl_vol[price] == 0 {
            self.fix_best(side, price as i64);
        }
    }

    #[inline(always)]
    fn rest(&mut self, t: i64, side: i64, price: i64, qty: i64, owner: i64, seq: i64) {
        if self.free_top == 0 {
            self.error = 1;
            return;
        }
        self.free_top -= 1;
        let slot = self.free[self.free_top] as usize;
        let pu = price as usize;
        self.o_price[slot] = price as i32;
        self.o_qty[slot] = qty as i32;
        self.o_side[slot] = side as i32;
        self.o_owner[slot] = owner as i32;
        self.o_seq[slot] = seq as i32;
        self.o_post_t[slot] = t;
        self.o_qahead[slot] = self.lvl_vol[pu];
        let tail = self.lvl_tail[pu];
        self.o_prev[slot] = tail;
        self.o_next[slot] = -1;
        if tail >= 0 {
            self.o_next[tail as usize] = slot as i32;
        } else {
            self.lvl_head[pu] = slot as i32;
        }
        self.lvl_tail[pu] = slot as i32;
        self.lvl_vol[pu] += qty;
        self.seq2slot[seq as usize] = slot as i32;
        if side == 0 {
            if price > self.bb {
                self.bb = price;
            }
        } else if price < self.ba {
            self.ba = price;
        }
    }

    #[inline(always)]
    fn match_q(&mut self, t: i64, taker_side: i64, limit_price: i64, mut qty: i64, owner: i64) -> i64 {
        if taker_side == 0 {
            while qty > 0 && self.ba < P && self.ba <= limit_price {
                let px = self.ba;
                let slot = self.lvl_head[px as usize] as usize;
                let fq = qty.min(self.o_qty[slot] as i64);
                self.fill(t, px, fq, slot, 0, owner);
                self.o_qty[slot] -= fq as i32;
                self.lvl_vol[px as usize] -= fq;
                qty -= fq;
                if self.o_qty[slot] == 0 {
                    self.remove(slot);
                }
            }
        } else {
            while qty > 0 && self.bb >= 0 && self.bb >= limit_price {
                let px = self.bb;
                let slot = self.lvl_head[px as usize] as usize;
                let fq = qty.min(self.o_qty[slot] as i64);
                self.fill(t, px, fq, slot, 1, owner);
                self.o_qty[slot] -= fq as i32;
                self.lvl_vol[px as usize] -= fq;
                qty -= fq;
                if self.o_qty[slot] == 0 {
                    self.remove(slot);
                }
            }
        }
        qty
    }

    #[inline(always)]
    fn limit(&mut self, t: i64, side: i64, price: i64, qty: i64, owner: i64, seq: i64) {
        self.seq_counter += 1;
        let rem = self.match_q(t, side, price, qty, owner);
        if rem > 0 {
            self.rest(t, side, price, rem, owner, seq);
        }
    }

    #[inline(always)]
    fn market(&mut self, t: i64, side: i64, qty: i64, owner: i64) {
        if side == 0 {
            self.match_q(t, 0, P - 1, qty, owner);
        } else {
            self.match_q(t, 1, 0, qty, owner);
        }
    }

    #[inline(always)]
    fn cancel(&mut self, seq: i64) {
        let slot = self.seq2slot[seq as usize];
        if slot >= 0 {
            self.remove(slot as usize);
        }
    }

    #[inline(always)]
    fn do_limit(&mut self, t: i64, side: i64, price: i64, qty: i64, owner: i64) -> i64 {
        let seq = self.seq_counter;
        self.n_cmds += 1;
        self.limit(t, side, price, qty, owner, seq);
        seq
    }

    #[inline(always)]
    fn do_cancel(&mut self, seq: i64) {
        self.n_cmds += 1;
        self.cancel(seq);
    }

    fn command(&mut self, ts: i64, typ: i32, side: i64, price: i64, qty: i64, seq: i64, owner: i64) {
        match typ {
            0 => self.limit(ts, side, price, qty, owner, seq),
            1 => self.market(ts, side, qty, owner),
            _ => self.cancel(seq),
        }
        self.n_cmds += 1;
    }

    fn result(&self, live_len: i64) -> Vec<i64> {
        vec![
            (self.hash >> 32) as i64,
            (self.hash & 0xFFFF_FFFF) as i64,
            self.n_fills,
            self.volume,
            self.mm_fills,
            self.mm_volume,
            self.mm_inv,
            self.mm_cash,
            self.n_cmds,
            self.bb,
            self.ba,
            self.seq_counter,
            self.last_px,
            live_len,
            self.error,
        ]
    }
}

struct Rng(u64);

impl Rng {
    #[inline(always)]
    fn next(&mut self) -> i64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^= z >> 31;
        (z >> 33) as i64
    }
}

struct SimOut {
    res: Vec<i64>,
    daily: Vec<i64>,
    fills: Vec<i64>,
}

fn window_max_min(buf: &VecDeque<i64>, w: usize) -> (i64, i64) {
    let mut hi = i64::MIN;
    let mut lo = i64::MAX;
    for &x in buf.iter().rev().take(w) {
        hi = hi.max(x);
        lo = lo.min(x);
    }
    (hi, lo)
}

fn simulate(pi: &[i64], pf: &[f64], max_seq: usize, collect: bool) -> SimOut {
    let n_steps = pi[0];
    let (k_range, ql, qm) = (pi[2], pi[3], pi[4]);
    let (thr_lim, cx, thr_same) = (pi[5], pi[6], pi[7]);
    let (req, warm, qmm, inv_max) = (pi[8], pi[9], pi[10], pi[11]);
    let (rsi_n, w_brk, w_sr, lvl_ticks, day_steps, markout_h) =
        (pi[12], pi[13] as usize, pi[14] as usize, pi[15], pi[16], pi[17]);
    let (mu, rho, alpha_j) = (pf[0], pf[1], pf[2]);
    let (af, as_, a_sig, gh, hs) = (pf[3], pf[4], pf[5], pf[6], pf[7]);
    let (ra, a_rsi, a_brk, brk_decay, vol_mult, vol_alpha) = (pf[8], pf[9], pf[10], pf[11], pf[12], pf[13]);
    let (k_brk, brk_pull, a_sr, hs_lvl, a_imb) = (pf[14], pf[15], pf[16], pf[17], pf[18]);
    let (qty_inf, anchor_value, thr_value, thr_inf, vjump) = (pi[18], pi[19], pi[20], pi[21], pi[22]);
    let mut value: i64 = INIT_REF;
    // skip window scans whose results cannot affect quotes (pure optimisation: outputs are unchanged)
    let use_brk = a_brk != 0.0 || k_brk != 0.0 || brk_pull > 0.0;
    let use_sr = a_sr != 0.0 || lvl_ticks > 0;

    let mut eng = Engine::new(max_seq);
    eng.collect = collect;
    eng.markout_h = markout_h;
    let mut daily: Vec<i64> = Vec::new();
    let mut rng = Rng(pi[1] as u64);
    let mut live: Vec<i32> = vec![0; LIVE_CAP];
    let mut n: usize = 0;
    for k in 1..=N_SEED {
        live[n] = eng.do_limit(0, 0, INIT_REF - k, 10, 0) as i32;
        n += 1;
        live[n] = eng.do_limit(0, 1, INIT_REF + k, 10, 0) as i32;
        n += 1;
    }
    let mut e = 0.0f64;
    let mut last_side: i64 = 0;
    let (mut ema_f, mut ema_s) = (0.0f64, 0.0f64);
    let mut ema_init = false;
    let (mut bid_seq, mut ask_seq): (i64, i64) = (-1, -1);
    let (mut bid_px, mut ask_px): (i64, i64) = (0, 0);
    let mut buf: VecDeque<i64> = VecDeque::with_capacity(BUF_CAP + 1);
    let mut prev_mid = 0.0f64;
    let mut n_diff: i64 = 0;
    let (mut avg_g, mut avg_l) = (0.0f64, 0.0f64);
    let mut brk_state = 0.0f64;
    let mut vol_ema = 0.0f64;
    let mut vol_init = false;
    let mut last_vol: i64 = 0;
    let mut last_mid2: i64 = 2 * INIT_REF;

    for t in 0..n_steps {
        e *= rho;
        if thr_value > 0 && rng.next() < thr_value {
            // latent value: +-vjump tick random walk
            let step = if rng.next() % 2 == 0 { vjump } else { -vjump };
            value = (value + step).max(2).min(P - 3);
        }
        for side in 0..2i64 {
            if rng.next() < thr_lim {
                let k = 1 + rng.next() % k_range;
                let q = 1 + rng.next() % ql;
                let (bb, ba) = (eng.bb, eng.ba);
                // symmetric around the mid: buys k below ceil(mid), sells k above floor(mid)
                let (lo_anchor, hi_anchor) = if anchor_value != 0 {
                    (value, value)
                } else if bb >= 0 && ba < P {
                    ((bb + ba) / 2, (bb + ba + 1) / 2)
                } else {
                    (eng.last_px, eng.last_px)
                };
                let price = if side == 0 { (hi_anchor - k).max(2) } else { (lo_anchor + k).min(P - 3) };
                if n >= LIVE_CAP {
                    eng.error = 3;
                    return SimOut { res: eng.result(n as i64), daily, fills: eng.fills };
                }
                live[n] = eng.do_limit(t, side, price, q, 0) as i32;
                n += 1;
            }
        }
        if n > 0 {
            let thr = (n as i64 * cx).min(M31);
            if rng.next() < thr {
                let idx = (rng.next() % n as i64) as usize;
                let seq = live[idx] as i64;
                live[idx] = live[n - 1];
                n -= 1;
                eng.do_cancel(seq);
            }
        }
        let mut lam = mu + e;
        if lam > 1.0 {
            lam = 1.0;
        }
        if rng.next() < (lam * 2147483648.0) as i64 {
            e += alpha_j;
            let side = if rng.next() < thr_same { last_side } else { 1 - last_side };
            last_side = side;
            let q = 1 + rng.next() % qm;
            eng.n_cmds += 1;
            eng.market(t, side, q, 0);
        }
        if thr_inf > 0 && rng.next() < thr_inf {
            // informed trader: trade toward the value
            if eng.ba < P && value > eng.ba {
                eng.n_cmds += 1;
                eng.market(t, 0, qty_inf, 0);
            } else if eng.bb >= 0 && value < eng.bb {
                eng.n_cmds += 1;
                eng.market(t, 1, qty_inf, 0);
            }
        }
        let (bb, ba) = (eng.bb, eng.ba);
        if t % req == 0 && t >= warm && bb >= 0 && ba < P {
            let mid2 = bb + ba;
            let mid = mid2 as f64 * 0.5;
            if !ema_init {
                ema_f = mid;
                ema_s = mid;
                ema_init = true;
                prev_mid = mid;
            } else {
                ema_f += af * (mid - ema_f);
                ema_s += as_ * (mid - ema_s);
                let d = mid - prev_mid;
                prev_mid = mid;
                let gain = if d > 0.0 { d } else { 0.0 };
                let loss = if d < 0.0 { -d } else { 0.0 };
                if n_diff == 0 {
                    avg_g = gain;
                    avg_l = loss;
                } else {
                    avg_g += ra * (gain - avg_g);
                    avg_l += ra * (loss - avg_l);
                }
                n_diff += 1;
            }
            let mut rsi_sig = 0.0f64;
            if n_diff >= rsi_n {
                let rsi = if avg_l == 0.0 { 100.0 } else { 100.0 - 100.0 / (1.0 + avg_g / avg_l) };
                rsi_sig = (50.0 - rsi) / 50.0;
            }
            let vol_since = eng.volume - last_vol;
            last_vol = eng.volume;
            let mut brk: i64 = 0;
            if use_brk && buf.len() >= w_brk {
                let (hi, lo) = window_max_min(&buf, w_brk);
                if mid2 > hi {
                    brk = 1;
                } else if mid2 < lo {
                    brk = -1;
                }
                if brk != 0 && vol_mult > 0.0 && !(vol_init && (vol_since as f64) > vol_mult * vol_ema) {
                    brk = 0;
                }
            }
            brk_state = brk_state * brk_decay + brk as f64;
            if !vol_init {
                vol_ema = vol_since as f64;
                vol_init = true;
            } else {
                vol_ema += vol_alpha * (vol_since as f64 - vol_ema);
            }
            let mut sr_sig = 0.0f64;
            let mut near = false;
            if use_sr && buf.len() >= w_sr {
                let (r2, s2) = window_max_min(&buf, w_sr);
                if r2 > s2 {
                    sr_sig = 0.5 - (mid2 - s2) as f64 / (r2 - s2) as f64;
                }
                if lvl_ticks > 0 && ((r2 - mid2) <= 2 * lvl_ticks || (mid2 - s2) <= 2 * lvl_ticks) {
                    near = true;
                }
            }
            if use_brk || use_sr {
                buf.push_back(mid2);
                if buf.len() > BUF_CAP {
                    buf.pop_front();
                }
            }
            let (qb, qa) = (eng.lvl_vol[bb as usize], eng.lvl_vol[ba as usize]);
            let imb_sig = ((qb - qa) as f64 / (qb + qa) as f64) * ((ba - bb) as f64 * 0.5);
            let sig = ema_f - ema_s;
            let inv = eng.mm_inv;
            let mut r = ((((mid + a_sig * sig) + a_rsi * rsi_sig) + a_brk * brk_state) + a_sr * sr_sig) + a_imb * imb_sig;
            r = r - gh * (inv as f64);
            let hs_eff = (hs + k_brk * brk_state.abs()) + (if near { hs_lvl } else { 0.0 });
            let mut bidp = (r - hs_eff).floor() as i64;
            let mut askp = (r + hs_eff).ceil() as i64;
            bidp = bidp.min(ba - 1);
            askp = askp.max(bb + 1);
            bidp = bidp.max(2);
            askp = askp.min(P - 3);
            let pull_bid = brk_pull > 0.0 && brk_state <= -brk_pull;
            let pull_ask = brk_pull > 0.0 && brk_state >= brk_pull;
            let live_b = bid_seq >= 0 && eng.seq2slot[bid_seq as usize] >= 0;
            if inv < inv_max && !pull_bid {
                if !(live_b && bid_px == bidp) {
                    if live_b {
                        eng.do_cancel(bid_seq);
                    }
                    bid_seq = eng.do_limit(t, 0, bidp, qmm, 1);
                    bid_px = bidp;
                }
            } else if live_b {
                eng.do_cancel(bid_seq);
                bid_seq = -1;
            }
            let live_a = ask_seq >= 0 && eng.seq2slot[ask_seq as usize] >= 0;
            if inv > -inv_max && !pull_ask {
                if !(live_a && ask_px == askp) {
                    if live_a {
                        eng.do_cancel(ask_seq);
                    }
                    ask_seq = eng.do_limit(t, 1, askp, qmm, 1);
                    ask_px = askp;
                }
            } else if live_a {
                eng.do_cancel(ask_seq);
                ask_seq = -1;
            }
        }
        // end-of-step bookkeeping (no randomness, no commands)
        if eng.bb >= 0 && eng.ba < P {
            last_mid2 = eng.bb + eng.ba;
        }
        if collect {
            eng.settle_markouts(t, last_mid2);
            if (t + 1) % day_steps == 0 {
                daily.extend_from_slice(&[
                    t,
                    last_mid2,
                    2 * eng.mm_cash + eng.mm_inv * last_mid2,
                    eng.mm_inv,
                    eng.mm_traded_value,
                    eng.mm_fills,
                    eng.mm_volume,
                    eng.n_fills,
                ]);
            }
        }
    }
    let res = eng.result(n as i64);
    SimOut { res, daily, fills: eng.fills }
}

/// Full loop, summary result only (used by the benchmark).
#[pyfunction]
fn run_full(pi: PyReadonlyArray1<i64>, pf: PyReadonlyArray1<f64>, max_seq: usize) -> PyResult<Vec<i64>> {
    Ok(simulate(pi.as_slice()?, pf.as_slice()?, max_seq, false).res)
}

/// Full loop with outputs: (result vector, daily snapshots flat [n_days*8], MM fills flat [n_fills*7]).
#[pyfunction]
fn run_sim<'py>(
    py: Python<'py>,
    pi: PyReadonlyArray1<i64>,
    pf: PyReadonlyArray1<f64>,
    max_seq: usize,
) -> PyResult<(Vec<i64>, Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>)> {
    let out = simulate(pi.as_slice()?, pf.as_slice()?, max_seq, true);
    Ok((out.res, PyArray1::from_vec(py, out.daily), PyArray1::from_vec(py, out.fills)))
}

#[pyfunction]
fn run_tape(tape: PyReadonlyArray1<i32>, max_seq: usize) -> PyResult<Vec<i64>> {
    let tape = tape.as_slice()?;
    let mut eng = Engine::new(max_seq);
    for c in tape.chunks_exact(7) {
        eng.command(c[0] as i64, c[1], c[2] as i64, c[3] as i64, c[4] as i64, c[5] as i64, c[6] as i64);
    }
    Ok(eng.result(0))
}

/// Python-driven book: one PyO3 call per command (measures call overhead).
#[pyclass]
struct Book {
    eng: Engine,
}

#[pymethods]
impl Book {
    #[new]
    fn new(max_seq: usize) -> Self {
        Book { eng: Engine::new(max_seq) }
    }

    fn cmd(&mut self, ts: i64, typ: i32, side: i64, price: i64, qty: i64, seq: i64, owner: i64) {
        self.eng.command(ts, typ, side, price, qty, seq, owner);
    }

    fn result(&self) -> Vec<i64> {
        self.eng.result(0)
    }
}

#[pymodule]
fn lob_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_full, m)?)?;
    m.add_function(wrap_pyfunction!(run_sim, m)?)?;
    m.add_function(wrap_pyfunction!(run_tape, m)?)?;
    m.add_class::<Book>()?;
    Ok(())
}
