# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""
Tests for the transactional :class:`tessera.RemoveMove`.

(2d, 2) Pachner remove: pick a vertex with order 2d, delete it along
with its 2d incident N41-type simplices, and create 2 replacement
simplices.  ``dN0 = -1``; ``dN41 = -(2d-2) = -6`` in 4D; ``dN32 = 0``.

The trickiest move to roll back: rollback has to recreate a deleted
vertex (with the same ID and coordinates), reinsert its incident
edges (with original squared lengths), and recreate 2d top simplices.
"""
import math
import unittest
import tessera


def _make_st_with_addgrowth(d=4, n_simplices=200,
                              batch_size=50, max_batches=200):
    """Build a lattice and grow it via cdt.add() until at least one
    order-(2d) vertex exists that satisfies RemoveMove's structural
    prerequisites.

    The CDT growth is RNG-driven. ``cdt.add()`` proposes a Pachner
    add and accepts it via Metropolis; most attempts reject. Even
    when many accept, an accepted (1, d+1) add only creates an
    order-(d+1) vertex — but ``RemoveMove`` needs order-(2d), which
    arises only when multiple growth moves stack around the same
    vertex with the right typing. There is therefore no fixed number
    of ``cdt.add()`` calls that guarantees a removable vertex exists.

    To make the test robust regardless of RNG state we grow in
    batches and probe between batches; we return as soon as a
    removable vertex exists. Worst case: ``batch_size × max_batches``
    add attempts before raising.
    """
    sig = tessera.Signature(d, tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    st = tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0,
                           tessera.PREFERRED, tessera.Toroid())
    st.build(n_simplices)
    cdt = tessera.CDTSimulation(st, 2.2, 0.5, 0.6, 0.02, st.getN41())

    for _ in range(max_batches):
        for _ in range(batch_size):
            cdt.add()
        # Probe: try a fistful of RemoveMove.propose() calls. propose()
        # doesn't mutate the topology — only advances the Spacetime
        # RNG — so repeated probing is cheap and safe.
        for s in range(200):
            if tessera.RemoveMove(st, s).propose():
                return st

    raise RuntimeError(
        f"_make_st_with_addgrowth: no order-{2*d} vertex after "
        f"{batch_size * max_batches} add attempts."
    )


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


def _try_propose(st, seeds):
    for s in seeds:
        m = tessera.RemoveMove(st, s)
        if m.propose():
            return m
    return None


# ---------------------------------------------------------------------------
# propose()
# ---------------------------------------------------------------------------


class TestRemovePropose(unittest.TestCase):

    def test_propose_does_not_mutate_state(self):
        st = _make_st_with_addgrowth()
        before = _full_snapshot(st)
        for seed in range(20):
            m = tessera.RemoveMove(st, seed)
            m.propose()
            self.assertEqual(_full_snapshot(st), before)

    def test_propose_succeeds_eventually(self):
        st = _make_st_with_addgrowth()
        m = _try_propose(st, range(2000))
        self.assertIsNotNone(m, "Could not find a remove target — try "
                                "more grows or larger lattice")

    def test_movetype(self):
        m = tessera.RemoveMove(_make_st_with_addgrowth(), 0)
        self.assertEqual(m.moveType(), "remove")

    def test_dN0_is_minus_one(self):
        st = _make_st_with_addgrowth()
        m = _try_propose(st, range(2000))
        self.assertIsNotNone(m)
        self.assertEqual(m.dN0(), -1)

    def test_dN41_is_minus_2d_minus_2(self):
        """In 4D: dN41 = -(2*4 - 2) = -6."""
        st = _make_st_with_addgrowth(d=4)
        m = _try_propose(st, range(2000))
        self.assertIsNotNone(m)
        self.assertEqual(m.dN41(), -6)

    def test_dN32_is_zero(self):
        st = _make_st_with_addgrowth()
        m = _try_propose(st, range(2000))
        self.assertIsNotNone(m)
        self.assertEqual(m.dN32(), 0)

    def test_log_prefactor_matches_formula(self):
        st = _make_st_with_addgrowth()
        m = _try_propose(st, range(2000))
        self.assertIsNotNone(m)
        # log(N0 / N41after) where N41after = N41 - (2d-2)
        n41 = st.getN41()
        n0 = st.getVertexCount()
        d = 4
        n41_after = n41 - (2 * d - 2)
        self.assertAlmostEqual(
            m.metropolisLogPrefactor(),
            math.log(n0) - math.log(n41_after),
            places=8
        )

    def test_touched_vertex_ids(self):
        """v + d spatial + vertA + vertB = d + 3."""
        st = _make_st_with_addgrowth(d=4)
        m = _try_propose(st, range(2000))
        self.assertIsNotNone(m)
        ids = m.touchedVertexIds()
        self.assertEqual(len(ids), 7)
        self.assertEqual(len(set(ids)), 7)


# ---------------------------------------------------------------------------
# apply()
# ---------------------------------------------------------------------------


class TestRemoveApply(unittest.TestCase):

    def test_apply_decrements_n0_and_n41(self):
        st = _make_st_with_addgrowth()
        m = _try_propose(st, range(2000))
        self.assertIsNotNone(m)
        n0_b, n41_b, n32_b = (st.getVertexCount(), st.getN41(),
                              st.getN32())
        m.apply()
        self.assertEqual(st.getVertexCount(), n0_b - 1)
        self.assertEqual(st.getN41(), n41_b - 6)
        self.assertEqual(st.getN32(), n32_b)

    def test_apply_removes_target_vertex_id_from_list(self):
        """The vertex ID in touchedVertexIds()[0] (= the v being
        removed) is no longer present after apply."""
        st = _make_st_with_addgrowth()
        m = _try_propose(st, range(2000))
        self.assertIsNotNone(m)
        target_id = m.touchedVertexIds()[0]
        self.assertIn(
            target_id,
            {v.getId() for v in st.getVertexList().toVector()}
        )
        m.apply()
        self.assertNotIn(
            target_id,
            {v.getId() for v in st.getVertexList().toVector()}
        )

    def test_double_apply_returns_false(self):
        st = _make_st_with_addgrowth()
        m = _try_propose(st, range(2000))
        self.assertIsNotNone(m)
        m.apply()
        snap = _full_snapshot(st)
        self.assertFalse(m.apply())
        self.assertEqual(_full_snapshot(st), snap)

    def test_apply_without_propose_fails(self):
        st = _make_st_with_addgrowth()
        m = tessera.RemoveMove(st, 0)
        before = _full_snapshot(st)
        self.assertFalse(m.apply())
        self.assertEqual(_full_snapshot(st), before)


# ---------------------------------------------------------------------------
# rollback()
# ---------------------------------------------------------------------------


class TestRemoveRollback(unittest.TestCase):

    def test_rollback_restores_state(self):
        st = _make_st_with_addgrowth()
        before = _full_snapshot(st)
        m = _try_propose(st, range(2000))
        self.assertIsNotNone(m)
        m.apply()
        self.assertNotEqual(_full_snapshot(st), before)
        m.rollback()
        self.assertEqual(_full_snapshot(st), before,
                         "rollback() must restore byte-identical state")

    def test_rollback_restores_vertex_id(self):
        """The removed vertex's ID returns to the vertex list after
        rollback."""
        st = _make_st_with_addgrowth()
        m = _try_propose(st, range(2000))
        self.assertIsNotNone(m)
        target_id = m.touchedVertexIds()[0]
        m.apply()
        self.assertNotIn(
            target_id,
            {v.getId() for v in st.getVertexList().toVector()}
        )
        m.rollback()
        self.assertIn(
            target_id,
            {v.getId() for v in st.getVertexList().toVector()}
        )

    def test_rollback_without_apply_is_noop(self):
        st = _make_st_with_addgrowth()
        before = _full_snapshot(st)
        m = tessera.RemoveMove(st, 0)
        m.propose()
        m.rollback()
        self.assertEqual(_full_snapshot(st), before)

    def test_double_rollback_is_noop(self):
        st = _make_st_with_addgrowth()
        before = _full_snapshot(st)
        m = _try_propose(st, range(2000))
        self.assertIsNotNone(m)
        m.apply()
        m.rollback()
        m.rollback()
        self.assertEqual(_full_snapshot(st), before)


# ---------------------------------------------------------------------------
# Stress
# ---------------------------------------------------------------------------


class TestRemoveStress(unittest.TestCase):

    def test_repeated_apply_rollback(self):
        st = _make_st_with_addgrowth()
        before = _full_snapshot(st)
        cycles = 0
        for seed in range(5000):
            m = tessera.RemoveMove(st, seed)
            if not m.propose():
                continue
            m.apply()
            m.rollback()
            self.assertEqual(_full_snapshot(st), before,
                             f"diverged after remove cycle {cycles} "
                             f"(seed={seed})")
            cycles += 1
            if cycles >= 30:
                break
        self.assertGreaterEqual(
            cycles, 5,
            f"only {cycles} successful remove cycles in 5000 seeds"
        )

    def test_chain_of_removes_lifo_rollback(self):
        """Apply 2 removes, rollback in LIFO order: state must equal
        initial."""
        # max_batches × batch_size ≥ the old n_grows = 600 so this
        # case has enough room to find two stackable order-2d vertices.
        st = _make_st_with_addgrowth(max_batches=300)
        before = _full_snapshot(st)
        applied = []
        for seed in range(5000):
            if len(applied) == 2:
                break
            m = tessera.RemoveMove(st, seed)
            if not m.propose():
                continue
            if m.apply():
                applied.append(m)
        if len(applied) < 2:
            self.skipTest("Could not chain 2 removes")
        for m in reversed(applied):
            m.rollback()
        self.assertEqual(_full_snapshot(st), before)


if __name__ == "__main__":
    unittest.main()
