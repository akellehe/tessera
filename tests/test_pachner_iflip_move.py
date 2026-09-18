# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""
Tests for the transactional :class:`tessera.IFlipMove`.

Inverse (d, 2) flip: removes d d-simplices sharing an edge and creates
2 new d-simplices sharing a (d-1)-face.  ``dN0 = 0``;
``ΔN4 = -(d - 2)``.  Includes a manifold-preservation check in
propose() that rejects if either new simplex would already exist.
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


def _grow(cdt, n=300):
    """Run n cdt.add() calls so iflip has eligible (d,2) configurations."""
    for _ in range(n):
        cdt.add()


class TestIFlipPropose(unittest.TestCase):

    def test_propose_does_not_mutate_state(self):
        # Iflip rarely accepts on a fresh lattice; grow first via add.
        sig = tessera.Signature(4, tessera.Lorentzian)
        metric = tessera.Metric(True, sig)
        st = tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0,
                               tessera.PREFERRED, tessera.Toroid())
        st.build(200)
        cdt = tessera.CDTSimulation(st, 2.2, 0.5, 0.6, 0.02, st.getN41())
        _grow(cdt, 200)
        before = _full_snapshot(st)
        for seed in range(20):
            m = tessera.IFlipMove(st, seed)
            m.propose()
            self.assertEqual(_full_snapshot(st), before)

    def test_propose_succeeds_eventually(self):
        # On a default Toroid lattice, iflip needs an edge with exactly
        # d=4 cofaces — uncommon.  Grow first.
        sig = tessera.Signature(4, tessera.Lorentzian)
        metric = tessera.Metric(True, sig)
        st = tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0,
                               tessera.PREFERRED, tessera.Toroid())
        st.build(200)
        cdt = tessera.CDTSimulation(st, 2.2, 0.5, 0.6, 0.02, st.getN41())
        _grow(cdt, 200)
        m = _try_propose(st, range(2000), tessera.IFlipMove)
        self.assertIsNotNone(m, "Could not find an iflip target in 2000 "
                                "seeds — try a larger lattice")

    def test_movetype(self):
        m = tessera.IFlipMove(_make_st(), 0)
        self.assertEqual(m.moveType(), "iflip")

    def test_dN0_is_zero(self):
        sig = tessera.Signature(4, tessera.Lorentzian)
        metric = tessera.Metric(True, sig)
        st = tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0,
                               tessera.PREFERRED, tessera.Toroid())
        st.build(200)
        cdt = tessera.CDTSimulation(st, 2.2, 0.5, 0.6, 0.02, st.getN41())
        _grow(cdt, 200)
        m = _try_propose(st, range(2000), tessera.IFlipMove)
        if m is None:
            self.skipTest("No iflip proposed")
        self.assertEqual(m.dN0(), 0)

    def test_dN4_advertised_is_minus_d_minus_2(self):
        """Advertised ΔN4 = -(d - 2) = -2 in 4D (clean (d,2) replacement)."""
        sig = tessera.Signature(4, tessera.Lorentzian)
        metric = tessera.Metric(True, sig)
        st = tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0,
                               tessera.PREFERRED, tessera.Toroid())
        st.build(200)
        cdt = tessera.CDTSimulation(st, 2.2, 0.5, 0.6, 0.02, st.getN41())
        _grow(cdt, 200)
        m = _try_propose(st, range(2000), tessera.IFlipMove)
        if m is None:
            self.skipTest("No iflip proposed")
        # 2 new - d old = -(d - 2) = -2
        self.assertEqual(m.dN41() + m.dN32(), -(4 - 2))


class TestIFlipApplyRollback(unittest.TestCase):

    def _make_grown(self, cdt_grow=200):
        sig = tessera.Signature(4, tessera.Lorentzian)
        metric = tessera.Metric(True, sig)
        st = tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0,
                               tessera.PREFERRED, tessera.Toroid())
        st.build(200)
        cdt = tessera.CDTSimulation(st, 2.2, 0.5, 0.6, 0.02, st.getN41())
        _grow(cdt, cdt_grow)
        return st

    def test_apply_then_rollback_restores_state(self):
        st = self._make_grown()
        before = _full_snapshot(st)
        m = _try_propose(st, range(3000), tessera.IFlipMove)
        if m is None:
            self.skipTest("No iflip proposed")
        m.apply()
        self.assertNotEqual(_full_snapshot(st), before)
        m.rollback()
        self.assertEqual(_full_snapshot(st), before,
                         "rollback() must restore byte-identical state")

    def test_apply_n4_change_matches_advertised(self):
        """Iflip's manifold check in propose() rejects dedupe cases, so
        the actual ΔN4 always matches the advertised one."""
        st = self._make_grown()
        m = _try_propose(st, range(3000), tessera.IFlipMove)
        if m is None:
            self.skipTest("No iflip proposed")
        n4_b = st.getTopSimplexCount()
        m.apply()
        actual_dN4 = st.getTopSimplexCount() - n4_b
        advertised = m.dN41() + m.dN32()
        self.assertEqual(actual_dN4, advertised,
                         f"actual ΔN4 ({actual_dN4}) != advertised "
                         f"({advertised}) — manifold check should "
                         f"have prevented dedupe")

    def test_apply_n0_unchanged(self):
        st = self._make_grown()
        m = _try_propose(st, range(3000), tessera.IFlipMove)
        if m is None:
            self.skipTest("No iflip proposed")
        n0_b = st.getVertexCount()
        m.apply()
        self.assertEqual(st.getVertexCount(), n0_b)

    def test_log_prefactor_matches_inverse_n4_ratio(self):
        st = self._make_grown()
        m = _try_propose(st, range(3000), tessera.IFlipMove)
        if m is None:
            self.skipTest("No iflip proposed")
        n4 = st.getTopSimplexCount()
        d = 4
        self.assertAlmostEqual(
            m.metropolisLogPrefactor(),
            math.log(n4) - math.log(n4 - d + 2),
            places=8
        )


class TestIFlipStress(unittest.TestCase):

    def test_repeated_apply_rollback(self):
        sig = tessera.Signature(4, tessera.Lorentzian)
        metric = tessera.Metric(True, sig)
        st = tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0,
                               tessera.PREFERRED, tessera.Toroid())
        st.build(200)
        cdt = tessera.CDTSimulation(st, 2.2, 0.5, 0.6, 0.02, st.getN41())
        _grow(cdt, 300)
        before = _full_snapshot(st)
        cycles = 0
        for seed in range(5000):
            m = tessera.IFlipMove(st, seed)
            if not m.propose():
                continue
            m.apply()
            m.rollback()
            self.assertEqual(_full_snapshot(st), before,
                             f"diverged after iflip cycle {cycles} "
                             f"(seed={seed})")
            cycles += 1
            if cycles >= 30:
                break
        self.assertGreater(
            cycles, 0,
            "No successful iflip cycles in 5000 seeds — try a larger "
            "lattice or more grows"
        )


if __name__ == "__main__":
    unittest.main()
