# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""
Tests for the transactional :class:`tessera.ShiftMove`.

The (3,3) Pachner shift removes 3 d-simplices sharing a (d-2)-face
and creates 3 new simplices sharing the complementary (d-2)-face.
``dN0 = 0``; ``dN41 + dN32 = 0``.  Self-inverse.

Coverage
--------

* :class:`TestShiftPropose` — propose() is read-only and reports
  validity correctly.
* :class:`TestShiftApply` — apply() commits; combinatorial deltas
  match what propose() advertised.
* :class:`TestShiftRollback` — rollback() restores byte-identical
  state (top-simplex fingerprints, edge fingerprints, vertex IDs,
  counts).
* :class:`TestShiftLifecycle` — ordering rules: propose-apply-rollback
  semantics, idempotent rollback, double-apply is no-op, no-apply
  rollback is no-op.
* :class:`TestShiftStress` — many propose/apply/rollback cycles in a
  row; final state byte-identical to initial.

Reference
---------
[BGL] Brunekreef, Gorlich, Loll, *Simulating CDT quantum gravity*,
arXiv:2310.16744 (2023), Sec. 2.3.3.
"""
import unittest
import tessera


# ---------------------------------------------------------------------------
# Helpers (shared across move-class tests; kept simple & local for now)
# ---------------------------------------------------------------------------


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
    """Hashable snapshot of the spacetime: counts plus full sets of
    top-simplex, edge, and vertex fingerprints."""
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
    """Try a sequence of seeds for ``move_cls``; return the first
    move that successfully proposes, or ``None``."""
    for s in seeds:
        m = move_cls(st, s)
        if m.propose():
            return m
    return None


# ---------------------------------------------------------------------------
# propose()
# ---------------------------------------------------------------------------


class TestShiftPropose(unittest.TestCase):
    """propose() is read-only and reports validity."""

    def test_propose_does_not_mutate_state(self):
        st = _make_st()
        before = _full_snapshot(st)
        # Try several seeds; whether any succeeds or fails, state must
        # not change.
        for seed in range(20):
            m = tessera.ShiftMove(st, seed)
            m.propose()
            self.assertEqual(_full_snapshot(st), before,
                             f"propose(seed={seed}) mutated state")

    def test_propose_succeeds_eventually(self):
        """On a typical built lattice, *some* seed in [0..200) yields
        a successful propose()."""
        st = _make_st()
        ok = False
        for seed in range(200):
            m = tessera.ShiftMove(st, seed)
            if m.propose():
                ok = True
                break
        self.assertTrue(ok, "Expected at least one shift to validate "
                            "in 200 seed attempts on a 200-simplex lattice")

    def test_propose_publishes_movetype(self):
        st = _make_st()
        m = tessera.ShiftMove(st, 0)
        # Move type is fixed regardless of propose() outcome.
        self.assertEqual(m.moveType(), "shift")

    def test_dN0_is_zero(self):
        """Shift never changes vertex count."""
        st = _make_st()
        m = _try_propose(st, range(200), tessera.ShiftMove)
        self.assertIsNotNone(m)
        self.assertEqual(m.dN0(), 0)

    def test_dN41_plus_dN32_is_zero_advertised(self):
        """Shift's *advertised* deltas always sum to zero (predicts a
        clean (3,3) replacement).  Actual ΔN4 may be in [-3, 0]
        because of dedupe on very small lattices — see
        ``test_pachner_deterministic.test_single_shift_exact_delta``
        for the existing-code characterization."""
        st = _make_st()
        m = _try_propose(st, range(200), tessera.ShiftMove)
        self.assertIsNotNone(m)
        self.assertEqual(m.dN41() + m.dN32(), 0)

    def test_log_prefactor_is_zero(self):
        """Shift's combinatorial selection is symmetric."""
        st = _make_st()
        m = _try_propose(st, range(200), tessera.ShiftMove)
        self.assertIsNotNone(m)
        self.assertEqual(m.metropolisLogPrefactor(), 0.0)

    def test_touched_vertex_ids_dPlus2(self):
        """Shift touches exactly d+2 vertices."""
        st = _make_st(d=4)
        m = _try_propose(st, range(200), tessera.ShiftMove)
        self.assertIsNotNone(m)
        # d=4 → d+2 = 6 unique vertex IDs
        ids = m.touchedVertexIds()
        self.assertEqual(len(ids), 6)
        self.assertEqual(len(set(ids)), 6, "vertex IDs must be unique")


# ---------------------------------------------------------------------------
# apply()
# ---------------------------------------------------------------------------


class TestShiftApply(unittest.TestCase):
    def test_apply_commits_state_change(self):
        st = _make_st()
        m = _try_propose(st, range(200), tessera.ShiftMove)
        self.assertIsNotNone(m)
        before = _full_snapshot(st)
        ok = m.apply()
        after = _full_snapshot(st)
        self.assertTrue(ok)
        self.assertNotEqual(before, after,
                            "apply() must change state")
        self.assertTrue(m.isApplied())

    def test_apply_dN0_matches_advertised(self):
        """N0 always changes by exactly dN0 (=0 for shift)."""
        st = _make_st()
        m = _try_propose(st, range(200), tessera.ShiftMove)
        self.assertIsNotNone(m)
        n0_b = st.getVertexCount()
        m.apply()
        self.assertEqual(st.getVertexCount(), n0_b + m.dN0())

    def test_apply_n4_in_documented_range(self):
        """Shift advertises dN41+dN32 = 0, but on small lattices dedupe
        can drop the actual N4 by up to 3.  The existing CDT code has
        the same property — see ``test_pachner_deterministic.
        test_single_shift_exact_delta`` (comment ``On very small
        lattices dedup can cause dN4 in [-3, 0]``)."""
        st = _make_st()
        m = _try_propose(st, range(200), tessera.ShiftMove)
        self.assertIsNotNone(m)
        n4_b = st.getTopSimplexCount()
        m.apply()
        actual_dN4 = st.getTopSimplexCount() - n4_b
        self.assertIn(actual_dN4, (-3, -2, -1, 0),
                      f"Shift's actual ΔN4 ({actual_dN4}) outside "
                      f"documented [-3, 0] range")

    def test_apply_preserves_n0(self):
        st = _make_st()
        m = _try_propose(st, range(200), tessera.ShiftMove)
        self.assertIsNotNone(m)
        n0_b = st.getVertexCount()
        m.apply()
        self.assertEqual(st.getVertexCount(), n0_b,
                         "Shift must preserve N0 exactly (dedupe only "
                         "affects top simplices, not vertices)")

    def test_apply_without_propose_fails(self):
        st = _make_st()
        m = tessera.ShiftMove(st, 0)
        # Without propose(), apply() should refuse.
        before = _full_snapshot(st)
        self.assertFalse(m.apply())
        self.assertEqual(_full_snapshot(st), before,
                         "apply() without propose() must not mutate")

    def test_double_apply_returns_false(self):
        """Calling apply() twice on the same object: the second call
        must return False and not mutate further."""
        st = _make_st()
        m = _try_propose(st, range(200), tessera.ShiftMove)
        self.assertIsNotNone(m)
        m.apply()
        snap_after_first = _full_snapshot(st)
        result = m.apply()
        snap_after_second = _full_snapshot(st)
        self.assertFalse(result)
        self.assertEqual(snap_after_first, snap_after_second)


# ---------------------------------------------------------------------------
# rollback()
# ---------------------------------------------------------------------------


class TestShiftRollback(unittest.TestCase):
    """rollback() restores byte-identical state."""

    def test_rollback_restores_state(self):
        st = _make_st()
        before = _full_snapshot(st)
        m = _try_propose(st, range(200), tessera.ShiftMove)
        self.assertIsNotNone(m)
        m.apply()
        self.assertNotEqual(_full_snapshot(st), before)
        m.rollback()
        self.assertEqual(_full_snapshot(st), before,
                         "rollback() must restore byte-identical state")
        self.assertFalse(m.isApplied())

    def test_rollback_without_apply_is_noop(self):
        st = _make_st()
        before = _full_snapshot(st)
        m = tessera.ShiftMove(st, 0)
        m.propose()  # may succeed or fail; either way, no apply
        m.rollback()
        self.assertEqual(_full_snapshot(st), before)

    def test_double_rollback_is_noop(self):
        st = _make_st()
        before = _full_snapshot(st)
        m = _try_propose(st, range(200), tessera.ShiftMove)
        self.assertIsNotNone(m)
        m.apply()
        m.rollback()
        first = _full_snapshot(st)
        m.rollback()
        second = _full_snapshot(st)
        self.assertEqual(first, second)
        self.assertEqual(first, before)


# ---------------------------------------------------------------------------
# Lifecycle ordering
# ---------------------------------------------------------------------------


class TestShiftLifecycle(unittest.TestCase):
    """Ordering rules for propose / apply / rollback."""

    def test_propose_apply_rollback_resets_isApplied(self):
        st = _make_st()
        m = _try_propose(st, range(200), tessera.ShiftMove)
        self.assertIsNotNone(m)
        self.assertFalse(m.isApplied())
        m.apply()
        self.assertTrue(m.isApplied())
        m.rollback()
        self.assertFalse(m.isApplied())

    def test_propose_called_twice_returns_false(self):
        """propose() is one-shot per object."""
        st = _make_st()
        m = tessera.ShiftMove(st, 0)
        first = m.propose()
        second = m.propose()
        if first:
            self.assertFalse(second,
                             "Second propose() on the same object "
                             "must return False")


# ---------------------------------------------------------------------------
# Stress: many cycles
# ---------------------------------------------------------------------------


class TestShiftStress(unittest.TestCase):
    """Many propose/apply/rollback cycles preserve state."""

    def test_repeated_apply_rollback_d4(self):
        """100 successful apply+rollback cycles with various seeds:
        the final state must equal the initial state."""
        st = _make_st(d=4)
        before = _full_snapshot(st)
        cycles = 0
        for seed in range(2000):
            m = tessera.ShiftMove(st, seed)
            if not m.propose():
                continue
            m.apply()
            m.rollback()
            self.assertEqual(_full_snapshot(st), before,
                             f"After cycle (seed={seed}), state diverged")
            cycles += 1
            if cycles >= 100:
                break
        self.assertGreaterEqual(cycles, 30,
                                f"Only {cycles} successful cycles in "
                                f"2000 seeds; expected ≥30")

    # Note: shift in tessera is wired as (d-1, d-1) which only really
    # accepts in d=4 (the (3,3) case).  In d=2/d=3 the hinge structure
    # required by the implementation rarely or never matches.  See
    # ``tests/test_pachner_exact.py`` line 367 ("Shift (3 → 3) — exact
    # structure, d=4 only").

    def test_apply_apply_rollback_rollback_chain(self):
        """Apply two shifts, then rollback in reverse order; final
        state must equal initial state.

        Seeds the spacetime's internal RNG (``st.setSeed``) so the
        sigma-selection inside ``ShiftMove.propose`` is deterministic
        across processes. The previous version of this test depended
        on ``std::random_device``-seeded sigma selection and was
        intermittently flaky in CI.
        """
        st = _make_st(d=4)
        st.setSeed(0)
        before = _full_snapshot(st)
        # Find two shifts that succeed sequentially.
        m1 = m2 = None
        for seed in range(500):
            m1 = tessera.ShiftMove(st, seed)
            if not m1.propose():
                continue
            if not m1.apply():
                continue
            for seed2 in range(seed + 1, seed + 200):
                m2 = tessera.ShiftMove(st, seed2)
                if m2.propose() and m2.apply():
                    break
            if m2 is not None and m2.isApplied():
                break
            m1.rollback()
            m1 = None
        if m1 is None or m2 is None or not m2.isApplied():
            self.skipTest("Could not chain two shift applies")
        # Now rollback in REVERSE order.
        m2.rollback()
        m1.rollback()
        self.assertEqual(_full_snapshot(st), before,
                         "Stacked apply/rollback in LIFO order must "
                         "restore byte-identical initial state")

    def test_overlapping_shifts_lifo_rollback_regression(self):
        """Regression for the use-after-free in the (3,3) Pachner
        rollback when two shifts share a created cell.

        Before the fix, ``m1.createdSimplices_`` held raw
        ``SimplexPtr`` for cells m1 created.  If m2 then removed one
        of those cells (and m2.rollback recreated it as a fresh
        allocation), m1's stored pointer was dangling.
        ``m1.rollback`` would feed it to ``removeSimplex``, whose
        swap-and-pop on the stale ``vecIdx_`` would clobber an
        unrelated simplex in the spacetime — leaving the dead cell in
        place and removing the wrong one.

        With ``st.setSeed(0)`` the failure is deterministic at
        ``(m1_seed=2, m2_seed=13)``: BEFORE has
        ``(25,27,32,33,34)`` and lacks ``(25,26,27,33,34)``; AFTER had
        the opposite.  After the fix (verts-based rollback resolution
        via ``Spacetime::findSimplexByVerts``), both states match
        byte-for-byte.
        """
        st = _make_st(d=4)
        st.setSeed(0)
        before = _full_snapshot(st)

        m1 = tessera.ShiftMove(st, 2)
        if not (m1.propose() and m1.apply()):
            self.skipTest("seed-pinned m1 didn't apply in this build")
        m2 = tessera.ShiftMove(st, 13)
        if not (m2.propose() and m2.apply()):
            self.skipTest("seed-pinned m2 didn't apply in this build")
        m2.rollback()
        m1.rollback()
        self.assertEqual(
            _full_snapshot(st), before,
            "LIFO rollback of two overlapping shifts must restore "
            "byte-identical initial state (regression for the "
            "stale-SimplexPtr use-after-free).")


if __name__ == "__main__":
    unittest.main()
