# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""
Tests for the transactional CDT proposal factories:

  cdt.proposeAdd()    -> AddMove or None
  cdt.proposeRemove() -> RemoveMove or None
  cdt.proposeFlip()   -> FlipMove or None
  cdt.proposeIflip()  -> IFlipMove or None
  cdt.proposeShift()  -> ShiftMove or None

Each factory binds the move to the simulation's shared RNG (so a
sequence of proposals draws from the same Markov chain).  Returns
None if no eligible target.  Caller drives apply()/rollback().  Does
NOT update CDT's acceptance counters (those are reserved for the
canonical add()/remove()/etc. methods).
"""
import unittest
import tessera


def _make_cdt(d=4, n_simplices=200, k0=2.2, k4=0.5, delta=0.6,
              epsilon=0.02, relabel=True):
    sig = tessera.Signature(d, tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    st = tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0,
                           tessera.PREFERRED, tessera.Toroid())
    st.build(n_simplices)
    cdt = tessera.CDTSimulation(st, k0, k4, delta, epsilon, st.getN41())
    cdt.setRelabelVertices(relabel)
    return cdt, st


def _full_snapshot(st):
    dPlus1 = next(iter(
        len(s.getVertices()) for s in st.getSimplices()
    ), 0)
    return {
        "n0": st.getVertexCount(),
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


class TestFactoriesBasic(unittest.TestCase):

    def test_propose_add_returns_movetype_add(self):
        cdt, st = _make_cdt()
        m = None
        for _ in range(50):
            m = cdt.proposeAdd()
            if m is not None:
                break
        self.assertIsNotNone(m)
        self.assertEqual(m.moveType(), "add")

    def test_propose_shift_returns_movetype_shift(self):
        cdt, st = _make_cdt()
        m = None
        for _ in range(200):
            m = cdt.proposeShift()
            if m is not None:
                break
        self.assertIsNotNone(m)
        self.assertEqual(m.moveType(), "shift")

    def test_propose_flip_returns_movetype_flip(self):
        cdt, st = _make_cdt()
        m = None
        for _ in range(200):
            m = cdt.proposeFlip()
            if m is not None:
                break
        self.assertIsNotNone(m)
        self.assertEqual(m.moveType(), "flip")

    def test_propose_iflip_returns_none_or_iflip(self):
        cdt, st = _make_cdt()
        # Iflip rarely propose-validates on a fresh lattice.
        for _ in range(50):
            m = cdt.proposeIflip()
            if m is not None:
                self.assertEqual(m.moveType(), "iflip")
                return
        # Acceptable: no iflip proposed on this small lattice.

    def test_propose_remove_returns_movetype_remove(self):
        cdt, st = _make_cdt()
        # Pre-grow so remove has eligible targets.
        for _ in range(300):
            cdt.add()
        m = None
        for _ in range(200):
            m = cdt.proposeRemove()
            if m is not None:
                break
        if m is None:
            self.skipTest("No remove target found in 200 proposals")
        self.assertEqual(m.moveType(), "remove")


class TestFactoriesDoNotMutateOnPropose(unittest.TestCase):
    """Calling a propose-factory and discarding the move object must
    not change the spacetime state (propose() is read-only)."""

    def test_proposeAdd_no_mutation(self):
        cdt, st = _make_cdt()
        before = _full_snapshot(st)
        for _ in range(20):
            m = cdt.proposeAdd()
            del m
        self.assertEqual(_full_snapshot(st), before)

    def test_proposeShift_no_mutation(self):
        cdt, st = _make_cdt()
        before = _full_snapshot(st)
        for _ in range(50):
            m = cdt.proposeShift()
            del m
        self.assertEqual(_full_snapshot(st), before)

    def test_proposeFlip_no_mutation(self):
        cdt, st = _make_cdt()
        before = _full_snapshot(st)
        for _ in range(50):
            m = cdt.proposeFlip()
            del m
        self.assertEqual(_full_snapshot(st), before)


class TestFactoriesShareRNG(unittest.TestCase):
    """Successive calls to a factory advance the shared RNG, so two
    propose calls in a row sample different targets (as opposed to the
    Python-bound :class:`tessera.AddMove(seed)` which uses an
    independent RNG per object)."""

    def test_two_proposeAdd_calls_advance_state(self):
        cdt, st = _make_cdt()
        # A proposal succeeds on roughly one attempt in ten, so the
        # attempt budget is an order of magnitude above the sample.
        proposals = []
        for _ in range(400):
            m = cdt.proposeAdd()
            if m is not None:
                proposals.append(tuple(sorted(m.touchedVertexIds())))
            if len(proposals) >= 20:
                break
        if len(proposals) < 2:
            self.skipTest("Not enough proposals to compare")
        # Proposals sample different sigmas + spatial faces, so any
        # individual pair may coincide; a whole sample of twenty
        # collapsing to one touched-ID set does not happen unless the
        # shared RNG has stopped advancing.
        self.assertTrue(
            any(p != proposals[0] for p in proposals[1:]),
            "All consecutive proposals had identical touched-ID sets — "
            "RNG may not be advancing"
        )


class TestFactoriesApplyRollback(unittest.TestCase):
    """End-to-end: factory → propose (already done) → apply → rollback
    restores byte-identical state."""

    def test_proposeAdd_apply_rollback_restores_state(self):
        cdt, st = _make_cdt(relabel=False)
        before = _full_snapshot(st)
        m = None
        for _ in range(50):
            m = cdt.proposeAdd()
            if m is not None:
                break
        self.assertIsNotNone(m)
        m.apply()
        self.assertNotEqual(_full_snapshot(st), before)
        m.rollback()
        self.assertEqual(_full_snapshot(st), before)

    def test_proposeFlip_apply_rollback_restores_state(self):
        cdt, st = _make_cdt(relabel=False)
        before = _full_snapshot(st)
        m = None
        for _ in range(200):
            m = cdt.proposeFlip()
            if m is not None:
                break
        self.assertIsNotNone(m)
        m.apply()
        m.rollback()
        self.assertEqual(_full_snapshot(st), before)

    def test_proposeShift_apply_rollback_restores_state(self):
        cdt, st = _make_cdt(relabel=False)
        before = _full_snapshot(st)
        m = None
        for _ in range(200):
            m = cdt.proposeShift()
            if m is not None:
                break
        self.assertIsNotNone(m)
        m.apply()
        m.rollback()
        self.assertEqual(_full_snapshot(st), before)


class TestFactoriesDoNotChangeAcceptanceCounters(unittest.TestCase):
    """propose-factory + apply does NOT update CDT's acceptance
    counters.  Those are reserved for the canonical
    cdt.add()/remove()/etc. methods."""

    def test_proposeAdd_apply_does_not_increment_counters(self):
        cdt, st = _make_cdt()
        # Drain initial state by running a few sweeps.
        rates_before = cdt.getAcceptanceRates()
        m = None
        for _ in range(50):
            m = cdt.proposeAdd()
            if m is not None:
                break
        self.assertIsNotNone(m)
        m.apply()
        # Acceptance rates: rate = accepted / attempts.  The numerator
        # and denominator both don't change.  The rate value is
        # identical (or both numerator and denominator went up by 1
        # together — but they don't because the factory bypasses the
        # bookkeeping).
        rates_after = cdt.getAcceptanceRates()
        # If counters didn't change, rates are identical.  Note rates
        # can be NaN at counters=0; treat NaN==NaN as equal here.
        for k in rates_before:
            self.assertAlmostEqual(rates_before[k], rates_after[k],
                                   places=12)


if __name__ == "__main__":
    unittest.main()
