"""Order-book engine tests: hand-checked matching semantics + three-way equivalence."""
import numpy as np
import pytest

from alphalab.lob import engine_numba as nb
from alphalab.lob import engine_ref as ref
from alphalab.lob.spec import Params, comparable

lob_rs = pytest.importorskip("lob_rs")

B, S = 0, 1
LIMIT, MARKET, CANCEL = 0, 1, 2


def tape(*cmds):
    """cmds: (ts, typ, side, price, qty, seq, owner)."""
    return np.array(cmds, dtype=np.int32).reshape(-1)


def run_all(t):
    """Run a tape through all three engines, return (ref_result, ref_book) and assert agreement."""
    r, book = ref.run_tape(t)
    n = nb.run_tape(t)
    max_seq = int(t.reshape(-1, 7)[:, 5].max()) + 2
    rs = np.array(lob_rs.run_tape(t, max_seq))
    assert comparable(r) == comparable(n) == comparable(rs)
    return r, book


def limit(seq, side, price, qty, ts=0, owner=0):
    return (ts, LIMIT, side, price, qty, seq, owner)


def market(side, qty, ts=1, owner=0):
    return (ts, MARKET, side, 0, qty, 0, owner)


def cancel(seq, ts=1):
    return (ts, CANCEL, 0, 0, 0, seq, 0)


def test_price_priority_best_price_first():
    t = tape(limit(0, S, 100, 5), limit(1, S, 99, 5), market(B, 7))
    _, book = run_all(t)
    # lowest ask (99) is hit first even though it arrived later
    assert [(p, q, s) for (_, p, q, s, _, _) in book.trades] == [(99, 5, 1), (100, 2, 0)]
    assert book.best_ask == 100 and book.orders[0][1] == 3


def test_price_priority_buy_side_highest_bid_first():
    t = tape(limit(0, B, 100, 5), limit(1, B, 101, 5), market(S, 7))
    _, book = run_all(t)
    assert [(p, q, s) for (_, p, q, s, _, _) in book.trades] == [(101, 5, 1), (100, 2, 0)]


def test_time_priority_same_price_fifo():
    t = tape(limit(0, S, 100, 3), limit(1, S, 100, 3), market(B, 4))
    _, book = run_all(t)
    assert [(q, s) for (_, _, q, s, _, _) in book.trades] == [(3, 0), (1, 1)]
    assert book.orders[1][1] == 2


def test_cancel_removes_order_and_loses_queue_position():
    t = tape(limit(0, S, 100, 3), limit(1, S, 100, 3), cancel(0), market(B, 2))
    _, book = run_all(t)
    assert [(q, s) for (_, _, q, s, _, _) in book.trades] == [(2, 1)]


def test_cancel_of_missing_order_is_noop():
    t = tape(limit(0, S, 100, 3), cancel(5), cancel(0), cancel(0), market(B, 1))
    r, book = run_all(t)
    assert book.n_fills == 0 and book.best_ask == 65536  # book empty, nothing filled


def test_partial_fill_leaves_remainder_at_front_of_queue():
    t = tape(limit(0, S, 100, 10), limit(1, S, 100, 4), market(B, 3), market(B, 8))
    _, book = run_all(t)
    assert [(q, s) for (_, _, q, s, _, _) in book.trades] == [(3, 0), (7, 0), (1, 1)]


def test_crossing_limit_matches_then_rests_remainder():
    t = tape(limit(0, S, 100, 5), limit(1, B, 101, 8))
    _, book = run_all(t)
    assert [(p, q) for (_, p, q, _, _, _) in book.trades] == [(100, 5)]   # trades at the resting price
    assert book.best_bid == 101 and book.orders[1][1] == 3 and book.best_ask == 65536


def test_market_order_on_empty_side_does_nothing():
    _, book = run_all(tape(market(B, 5)))
    assert book.n_fills == 0


def test_market_maker_accounting():
    # MM owns a resting bid; a noise market sell hits it -> MM long 4 at 99
    t = tape(limit(0, B, 99, 4, owner=1), market(S, 4))
    r, book = run_all(t)
    assert (book.mm_inv, book.mm_cash, book.mm_fills) == (4, -99 * 4, 1)


@pytest.mark.parametrize("seed", [1, 2, 3, 99])
def test_full_loop_three_way_equivalence(seed):
    p = Params(n_steps=20_000, seed=seed)
    pi, pf = p.pack()
    r_ref, _ = ref.run_full(p)
    r_nb = nb.run_full(p)
    r_rs = np.array(lob_rs.run_full(pi, pf, p.max_seq()))
    assert comparable(r_ref) == comparable(r_nb) == comparable(r_rs)
    assert r_nb[14] == 0 and r_rs[14] == 0           # no error codes


def test_recorded_tape_replays_to_same_result():
    p = Params(n_steps=20_000, seed=5)
    r_ref, tape_ref, _ = ref.run_full(p, record=True)
    r_nb, tape_nb = nb.run_full(p, record=True)
    assert np.array_equal(tape_ref, tape_nb)
    r, _ = run_all(tape_nb)
    assert comparable(r) == comparable(r_ref) == comparable(r_nb)


def test_python_driven_wrappers_match_batch_result():
    p = Params(n_steps=5_000, seed=11)
    _, tp = nb.run_full(p, record=True)
    ms = int(tp.reshape(-1, 7)[:, 5].max()) + 2
    d = nb.PyDriven(ms)
    b = lob_rs.Book(ms)
    for row in tp.reshape(-1, 7).tolist():
        d.cmd(*row)
        b.cmd(*row)
    batch = nb.run_tape(tp)
    assert comparable(np.array(d.result())) == comparable(batch)
    assert comparable(np.array(b.result())) == comparable(batch)
