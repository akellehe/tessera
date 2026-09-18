# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""
Tests for the transactional :class:`tessera.FlipMove`.

(2,d) Pachner flip: removes 2 d-simplices sharing a (d-1)-face and
creates d new d-simplices sharing an edge.  ``dN0 = 0``;
``ΔN4 = d - 2`` (advertised; actual may be smaller on small lattices
due to dedupe).
"""
import math
import unittest
import tessera


def _make_st(d=4, n_simplices=200):
    sig = tessera.Signature(d, tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    st = tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0,
                           tessera.PREFERRED, tessera.Toroid())
    st.build(n_simplices)
    return st


def _top_size(st):
    # CDT top simplices are (d+1)-vertex; d is the spacetime's declared
    # (signature) dimension. getTopVertexCount() == signature.dimensions + 1,
    # the engine's single source of truth for top-cell membership -- O(1) and
    # immune to the lazily-materialized lower-dimensional facets that
    # propose()/getFacets() register into getSimplices() (where the first
    # scanned simplex may be a lower-dimensional face).
    return st.getTopVertexCount()


def _full_snapshot(st):
    dPlus1 = _top_size(st)
    return {
        "n0": st.getVertexCount(),
        "n41": st.getN41(),
        "n32": st.getN32(),
        "n4": st.getTopSimplexCount(),
        "top_fps": frozenset(
            hash(s) for s in st.getSimplices()
            if len(s.getVertices()) == dPlus1
        ),
        "edge_fps": frozenset(
            hash(e) for e in st.getEdgeList().toVector()
        ),
        "vertex_ids": frozenset(
            v.getId() for v in st.getVertexList().toVector()
        ),
    }


def _try_propose(st, seeds, move_cls):
    for s in seeds:
        m = move_cls(st, s)
        if m.propose():
            return m
    return None


# ---------------------------------------------------------------------------
# propose()
# ---------------------------------------------------------------------------


class TestFlipPropose(unittest.TestCase):

    def test_propose_does_not_mutate_state(self):
        st = _make_st()
        before = _full_snapshot(st)
        for seed in range(20):
            m = tessera.FlipMove(st, seed)
            m.propose()
            self.assertEqual(_full_snapshot(st), before)

    def test_propose_succeeds_eventually(self):
        st = _make_st()
        m = _try_propose(st, range(200), tessera.FlipMove)
        self.assertIsNotNone(m)

    def test_movetype(self):
        m = tessera.FlipMove(_make_st(), 0)
        self.assertEqual(m.moveType(), "flip")

    def test_dN0_is_zero(self):
        st = _make_st()
        m = _try_propose(st, range(200), tessera.FlipMove)
        self.assertIsNotNone(m)
        self.assertEqual(m.dN0(), 0)

    def test_dN4_advertised_is_d_minus_2(self):
        """Advertised ΔN4 = d - 2 = +2 in 4D (clean (2,d) replacement)."""
        st = _make_st(d=4)
        m = _try_propose(st, range(200), tessera.FlipMove)
        self.assertIsNotNone(m)
        # dN4 = number of new (d) - number of old (2) = d - 2
        # but advertised is dN41 + dN32, which equals (newN41+newN32) -
        # (oldN41+oldN32) = d - 2 (clean replacement).
        self.assertEqual(m.dN41() + m.dN32(), 4 - 2)

    def test_log_prefactor_matches_n4_ratio(self):
        st = _make_st()
        m = _try_propose(st, range(200), tessera.FlipMove)
        self.assertIsNotNone(m)
        # log(N4 / (N4 + d - 2)).
        n4 = st.getTopSimplexCount()
        d = 4
        self.assertAlmostEqual(
            m.metropolisLogPrefactor(),
            math.log(n4) - math.log(n4 + d - 2),
            places=8
        )

    def test_touched_vertex_ids_dPlus2(self):
        st = _make_st(d=4)
        m = _try_propose(st, range(200), tessera.FlipMove)
        self.assertIsNotNone(m)
        ids = m.touchedVertexIds()
        self.assertEqual(len(ids), 4 + 2)
        self.assertEqual(len(set(ids)), 4 + 2)


# ---------------------------------------------------------------------------
# apply()
# ---------------------------------------------------------------------------


class TestFlipApply(unittest.TestCase):

    def test_apply_commits(self):
        st = _make_st()
        before = _full_snapshot(st)
        m = _try_propose(st, range(200), tessera.FlipMove)
        self.assertIsNotNone(m)
        ok = m.apply()
        after = _full_snapshot(st)
        self.assertTrue(ok)
        self.assertNotEqual(before, after)
        self.assertTrue(m.isApplied())

    def test_apply_dN0_matches(self):
        st = _make_st()
        m = _try_propose(st, range(200), tessera.FlipMove)
        self.assertIsNotNone(m)
        n0 = st.getVertexCount()
        m.apply()
        self.assertEqual(st.getVertexCount(), n0 + m.dN0())

    def test_apply_n4_change_in_documented_range(self):
        """Advertised ΔN4 = d-2 = +2 in 4D; dedupe may reduce actual."""
        st = _make_st(d=4)
        m = _try_propose(st, range(200), tessera.FlipMove)
        self.assertIsNotNone(m)
        n4 = st.getTopSimplexCount()
        m.apply()
        actual = st.getTopSimplexCount() - n4
        # Worst case: all d new simplices already exist → 0 created;
        # 2 removed → ΔN4 = -2.  Best case: clean (2,d) → ΔN4 = d-2.
        d = 4
        self.assertGreaterEqual(actual, -2)
        self.assertLessEqual(actual, d - 2)

    def test_apply_without_propose_fails(self):
        st = _make_st()
        m = tessera.FlipMove(st, 0)
        before = _full_snapshot(st)
        self.assertFalse(m.apply())
        self.assertEqual(_full_snapshot(st), before)

    def test_double_apply_returns_false(self):
        st = _make_st()
        m = _try_propose(st, range(200), tessera.FlipMove)
        self.assertIsNotNone(m)
        m.apply()
        snap = _full_snapshot(st)
        self.assertFalse(m.apply())
        self.assertEqual(_full_snapshot(st), snap)


# ---------------------------------------------------------------------------
# rollback()
# ---------------------------------------------------------------------------


class TestFlipRollback(unittest.TestCase):

    def test_rollback_restores_state(self):
        st = _make_st()
        before = _full_snapshot(st)
        m = _try_propose(st, range(200), tessera.FlipMove)
        self.assertIsNotNone(m)
        m.apply()
        m.rollback()
        self.assertEqual(_full_snapshot(st), before)
        self.assertFalse(m.isApplied())

    def test_rollback_without_apply_is_noop(self):
        st = _make_st()
        before = _full_snapshot(st)
        m = tessera.FlipMove(st, 0)
        m.propose()
        m.rollback()
        self.assertEqual(_full_snapshot(st), before)

    def test_double_rollback_is_noop(self):
        st = _make_st()
        before = _full_snapshot(st)
        m = _try_propose(st, range(200), tessera.FlipMove)
        self.assertIsNotNone(m)
        m.apply()
        m.rollback()
        m.rollback()
        self.assertEqual(_full_snapshot(st), before)


# ---------------------------------------------------------------------------
# Stress
# ---------------------------------------------------------------------------


class TestFlipStress(unittest.TestCase):

    def test_repeated_apply_rollback(self):
        st = _make_st()
        before = _full_snapshot(st)
        cycles = 0
        for seed in range(2000):
            m = tessera.FlipMove(st, seed)
            if not m.propose():
                continue
            m.apply()
            m.rollback()
            self.assertEqual(_full_snapshot(st), before,
                             f"cycle {cycles} (seed={seed}) diverged")
            cycles += 1
            if cycles >= 50:
                break
        self.assertGreaterEqual(cycles, 20,
                                f"only {cycles} successful flip cycles")

    def test_chained_flips_roll_back_in_lifo(self):
        st = _make_st()
        before = _full_snapshot(st)
        m1 = m2 = None
        for seed in range(500):
            m1 = tessera.FlipMove(st, seed)
            if not m1.propose():
                continue
            if not m1.apply():
                m1 = None
                continue
            for seed2 in range(seed + 1, seed + 200):
                m2 = tessera.FlipMove(st, seed2)
                if m2.propose() and m2.apply():
                    break
                m2 = None
            if m2 is not None:
                break
            m1.rollback()
            m1 = None
        if m1 is None or m2 is None:
            self.skipTest("Could not chain two flips")
        m2.rollback()
        m1.rollback()
        self.assertEqual(_full_snapshot(st), before)


if __name__ == "__main__":
    unittest.main()
