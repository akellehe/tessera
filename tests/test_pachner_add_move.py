# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""
Tests for the transactional :class:`tessera.AddMove`.

(2, 2d) Pachner add: insert a new vertex at the spatial face shared
by two opposite-orientation N41 simplices.  ``dN0 = +1``;
``dN41 = +(2d-2) = +6`` in 4D; ``dN32 = 0``.

The trickiest move to roll back: it both creates a vertex and
optionally relabels it (swapping IDs with an existing vertex).
The rollback un-swaps before removing the new vertex.
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


def _try_propose(st, seeds, **kwargs):
    for s in seeds:
        m = tessera.AddMove(st, s, **kwargs)
        if m.propose():
            return m
    return None


# ---------------------------------------------------------------------------
# propose()
# ---------------------------------------------------------------------------


class TestAddPropose(unittest.TestCase):

    def test_propose_does_not_mutate_state(self):
        st = _make_st()
        before = _full_snapshot(st)
        for seed in range(20):
            m = tessera.AddMove(st, seed)
            m.propose()
            self.assertEqual(_full_snapshot(st), before,
                             f"propose(seed={seed}) mutated state")

    def test_propose_succeeds_eventually(self):
        st = _make_st()
        m = _try_propose(st, range(200))
        self.assertIsNotNone(m)

    def test_movetype(self):
        m = tessera.AddMove(_make_st(), 0)
        self.assertEqual(m.moveType(), "add")

    def test_dN0_is_one(self):
        st = _make_st()
        m = _try_propose(st, range(200))
        self.assertIsNotNone(m)
        self.assertEqual(m.dN0(), 1)

    def test_dN41_is_2d_minus_2(self):
        """In 4D: dN41 = 2*4 - 2 = 6."""
        st = _make_st(d=4)
        m = _try_propose(st, range(200))
        self.assertIsNotNone(m)
        self.assertEqual(m.dN41(), 6)

    def test_dN32_is_zero(self):
        st = _make_st()
        m = _try_propose(st, range(200))
        self.assertIsNotNone(m)
        self.assertEqual(m.dN32(), 0)

    def test_log_prefactor_matches_formula(self):
        st = _make_st()
        m = _try_propose(st, range(200))
        self.assertIsNotNone(m)
        # log(N41 / (N0 + 1))
        n41 = st.getN41()
        n0 = st.getVertexCount()
        self.assertAlmostEqual(
            m.metropolisLogPrefactor(),
            math.log(n41) - math.log(n0 + 1.0),
            places=8
        )

    def test_touched_vertex_ids_dPlus2(self):
        """Touched vertices: d spatial + vertA + vertB = d+2."""
        st = _make_st(d=4)
        m = _try_propose(st, range(200))
        self.assertIsNotNone(m)
        ids = m.touchedVertexIds()
        self.assertEqual(len(ids), 6)
        self.assertEqual(len(set(ids)), 6)


# ---------------------------------------------------------------------------
# apply()
# ---------------------------------------------------------------------------


class TestAddApply(unittest.TestCase):

    def test_apply_increments_n0_and_n41(self):
        st = _make_st()
        m = _try_propose(st, range(200))
        self.assertIsNotNone(m)
        n0_b, n41_b, n32_b = (st.getVertexCount(), st.getN41(),
                              st.getN32())
        m.apply()
        self.assertEqual(st.getVertexCount(), n0_b + 1)
        self.assertEqual(st.getN41(), n41_b + 6)
        self.assertEqual(st.getN32(), n32_b)

    def test_apply_with_relabel_disabled_keeps_max_id(self):
        """With relabel disabled, the new vertex gets the next
        available auto ID (max existing + 1)."""
        st = _make_st()
        max_id_before = max(v.getId()
                            for v in st.getVertexList().toVector())
        # Find a successful add proposal with relabel=False.
        m = None
        for seed in range(200):
            m = tessera.AddMove(st, seed, relabel=False)
            if m.propose():
                break
        self.assertIsNotNone(m)
        m.apply()
        # New vertex id = max_id_before + 1
        max_id_after = max(v.getId()
                           for v in st.getVertexList().toVector())
        self.assertEqual(max_id_after, max_id_before + 1)

    def test_apply_with_relabel_enabled_swaps_ids(self):
        """With relabel enabled, the auto-assigned ID is swapped with
        a random existing vertex's ID."""
        st = _make_st()
        ids_before = sorted(v.getId() for v in
                            st.getVertexList().toVector())
        m = None
        for seed in range(200):
            m = tessera.AddMove(st, seed, relabel=True)
            if m.propose():
                break
        self.assertIsNotNone(m)
        m.apply()
        ids_after = sorted(v.getId() for v in
                           st.getVertexList().toVector())
        # Same set of IDs plus one new one (the auto-assigned).  The
        # set difference: |ids_after| - |ids_before| = 1.
        self.assertEqual(len(ids_after), len(ids_before) + 1)

    def test_double_apply_returns_false(self):
        st = _make_st()
        m = _try_propose(st, range(200))
        self.assertIsNotNone(m)
        m.apply()
        snap = _full_snapshot(st)
        self.assertFalse(m.apply())
        self.assertEqual(_full_snapshot(st), snap)

    def test_apply_without_propose_fails(self):
        st = _make_st()
        m = tessera.AddMove(st, 0)
        before = _full_snapshot(st)
        self.assertFalse(m.apply())
        self.assertEqual(_full_snapshot(st), before)


# ---------------------------------------------------------------------------
# rollback()
# ---------------------------------------------------------------------------


class TestAddRollback(unittest.TestCase):

    def test_rollback_restores_state_no_relabel(self):
        st = _make_st()
        before = _full_snapshot(st)
        m = None
        for seed in range(200):
            m = tessera.AddMove(st, seed, relabel=False)
            if m.propose():
                break
        self.assertIsNotNone(m)
        m.apply()
        self.assertNotEqual(_full_snapshot(st), before)
        m.rollback()
        self.assertEqual(_full_snapshot(st), before,
                         "rollback() must restore byte-identical state "
                         "(relabel disabled)")

    def test_rollback_restores_state_with_relabel(self):
        """Rollback must un-swap before removing the new vertex."""
        st = _make_st()
        before = _full_snapshot(st)
        m = None
        for seed in range(200):
            m = tessera.AddMove(st, seed, relabel=True)
            if m.propose():
                break
        self.assertIsNotNone(m)
        m.apply()
        self.assertNotEqual(_full_snapshot(st), before)
        m.rollback()
        self.assertEqual(_full_snapshot(st), before,
                         "rollback() must restore byte-identical state "
                         "(relabel enabled)")

    def test_rollback_without_apply_is_noop(self):
        st = _make_st()
        before = _full_snapshot(st)
        m = tessera.AddMove(st, 0)
        m.propose()
        m.rollback()
        self.assertEqual(_full_snapshot(st), before)

    def test_double_rollback_is_noop(self):
        st = _make_st()
        before = _full_snapshot(st)
        m = _try_propose(st, range(200))
        self.assertIsNotNone(m)
        m.apply()
        m.rollback()
        m.rollback()
        self.assertEqual(_full_snapshot(st), before)


# ---------------------------------------------------------------------------
# Stress
# ---------------------------------------------------------------------------


class TestAddStress(unittest.TestCase):

    def test_repeated_apply_rollback_no_relabel(self):
        st = _make_st()
        before = _full_snapshot(st)
        cycles = 0
        for seed in range(2000):
            m = tessera.AddMove(st, seed, relabel=False)
            if not m.propose():
                continue
            m.apply()
            m.rollback()
            self.assertEqual(_full_snapshot(st), before,
                             f"diverged after cycle {cycles} "
                             f"(seed={seed})")
            cycles += 1
            if cycles >= 50:
                break
        self.assertGreaterEqual(cycles, 30)

    def test_repeated_apply_rollback_with_relabel(self):
        st = _make_st()
        before = _full_snapshot(st)
        cycles = 0
        for seed in range(2000):
            m = tessera.AddMove(st, seed, relabel=True)
            if not m.propose():
                continue
            m.apply()
            m.rollback()
            self.assertEqual(_full_snapshot(st), before,
                             f"diverged after relabel cycle {cycles} "
                             f"(seed={seed})")
            cycles += 1
            if cycles >= 30:
                break
        self.assertGreaterEqual(cycles, 10)

    def test_chain_of_adds_lifo_rollback(self):
        """Apply 3 adds, rollback in LIFO order: state must equal initial."""
        st = _make_st()
        before = _full_snapshot(st)
        applied = []
        for seed in range(2000):
            if len(applied) == 3:
                break
            m = tessera.AddMove(st, seed, relabel=False)
            if not m.propose():
                continue
            if m.apply():
                applied.append(m)
        if len(applied) < 3:
            self.skipTest("Could not chain 3 adds")
        # Rollback in REVERSE order.
        for m in reversed(applied):
            m.rollback()
        self.assertEqual(_full_snapshot(st), before)


if __name__ == "__main__":
    unittest.main()
