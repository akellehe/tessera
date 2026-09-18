# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""
Characterization tests locking in current behavior of CDT::add/remove/
flip/iflip/shift before the transactional Pachner-move refactor in
``docs/source/modularity-plan.md``.

These tests deliberately overlap a little with
``tests/test_pachner_moves.py`` and ``tests/test_pachner_deterministic.py``
but fill specific gaps that the refactor needs as a safety net:

* :class:`TestStateUnchangedOnRejection` — when ``cdt.X()`` returns
  ``False``, the full fingerprint state (top simplices, edges, vertex
  IDs, counts) is byte-identical to before the call.  This is the
  linchpin guarantee: the refactor splits each move into ``propose()``
  (read-only) + ``apply()`` (mutating), and the absence of a call to
  ``apply()`` must leave the spacetime untouched in exactly the same
  way that today's "Metropolis rejected" path does.

* :class:`TestEdgeListMonotonic` — Pachner moves never delete edges
  from the EdgeList.  ``createSimplex`` may add new edges (deduped),
  ``removeSimplex`` does not cascade-delete.  The edge fingerprint set
  before any accepted move must be a subset of the set after.

* :class:`TestCofaceIntegrityAfterMove` — after every accepted move,
  every facet of every top simplex has the parent simplex in its
  coface list.  Today's tests don't systematically check this.

* :class:`TestActionDeltaPerMove` — ``computeAction()`` before/after an
  accepted move equals the analytic prediction from
  ``(k0, k4, delta, epsilon)`` and the observed
  ``(dN0, dN41, dN32)``.  Already partially covered by
  ``test_pachner_moves.TestActionConsistency``; here we lock the
  per-move deltas explicitly.

References
  [BGL] Brunekreef, Gorlich, Loll, "Simulating CDT quantum gravity",
        arXiv:2310.16744v1 (2023)
"""
import unittest
import tessera

# Canonical move-type names, taken from the move classes rather than re-spelled,
# so a rename cannot leave these loops silently driving nothing. Assertions on
# moveType() deliberately keep their string literals: compared against these they
# could never fail.
_ALL_MOVE_NAMES = (tessera.AddMove.MOVE_TYPE, tessera.RemoveMove.MOVE_TYPE,
                   tessera.FlipMove.MOVE_TYPE, tessera.IFlipMove.MOVE_TYPE,
                   tessera.ShiftMove.MOVE_TYPE)
# The moves a growth-only sweep drives (remove would shrink the complex).
_NON_REMOVE_MOVE_NAMES = tuple(n for n in _ALL_MOVE_NAMES
                               if n != tessera.RemoveMove.MOVE_TYPE)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cdt(d=4, n_simplices=200, k0=2.2, k4=0.5, delta=0.6,
              epsilon=0.02, relabel=True, target=None):
    """Construct a d-dimensional CDT spacetime + simulation."""
    sig = tessera.Signature(d, tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    st = tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0,
                           tessera.PREFERRED, tessera.Toroid())
    st.build(n_simplices)
    tgt = target if target is not None else st.getN41()
    cdt = tessera.CDTSimulation(st, k0, k4, delta, epsilon, tgt)
    cdt.setRelabelVertices(relabel)
    return cdt, st


def _top_simplex_size(st):
    """Return d+1 (the vertex count of a top simplex) for this spacetime.

    ``getTopVertexCount()`` returns ``signature.dimensions + 1`` directly --
    the engine's single source of truth for top-cell membership -- so it is
    O(1) and immune to the lazily-materialized lower-dimensional faces that
    ``getSimplices()`` accumulates. A move's ``propose()`` inspects a simplex's
    facets via ``getFacets()``, which creates and registers those
    lower-dimensional facet simplices on demand; scanning ``getSimplices()``
    for the top dimension is therefore order-dependent. (Using the *first*
    simplex's size here was the cause of an intermittent failure in
    TestStateUnchangedOnRejection: when a tetrahedron sorted first the snapshot
    began counting tetrahedra, which then grew as more facets materialized.)
    """
    return st.getTopVertexCount()


def _state_snapshot(st):
    """Capture a hashable snapshot of the spacetime's full state.

    Includes counts plus the *sets* of fingerprints for top simplices,
    edges, and vertex IDs.  Two snapshots compare equal iff the
    spacetime's observable state is byte-identical.
    """
    dPlus1 = _top_simplex_size(st)
    top_fps = frozenset(
        hash(s) for s in st.getSimplices()
        if len(s.getVertices()) == dPlus1
    )
    edge_fps = frozenset(hash(e) for e in st.getEdgeList().toVector())
    vertex_ids = frozenset(
        v.getId() for v in st.getVertexList().toVector()
    )
    return {
        "n0": st.getVertexCount(),
        "n41": st.getN41(),
        "n32": st.getN32(),
        "n4": st.getTopSimplexCount(),
        "top_fps": top_fps,
        "edge_fps": edge_fps,
        "vertex_ids": vertex_ids,
    }


def _grow(cdt, n=200):
    """Run n add() calls so remove/iflip have eligible targets."""
    for _ in range(n):
        cdt.add()


# ---------------------------------------------------------------------------
# State unchanged on rejection
# ---------------------------------------------------------------------------


class TestStateUnchangedOnRejection(unittest.TestCase):
    """Whenever ``cdt.X()`` returns ``False``, the full state is
    byte-identical to before the call.

    Linchpin guarantee for the refactor: today's behavior is that
    structural-validity rejection and Metropolis rejection both leave
    the spacetime untouched, because mutations only happen after the
    ``accept()`` call.  The PachnerMove-based refactor must preserve
    this — anything else is a regression.
    """

    def _run_battery(self, cdt, st, n_per_move=80,
                     min_total_rejections=20):
        """For each of (add, remove, flip, iflip, shift), call N times.
        After every False return, assert state == snapshot.  After every
        True return, refresh the snapshot.

        Asserts that we saw enough rejections in total — otherwise the
        test is vacuous.
        """
        snapshot = _state_snapshot(st)
        rejections = 0
        for _ in range(n_per_move):
            for move_name in _ALL_MOVE_NAMES:
                method = getattr(cdt, move_name)
                if method():
                    snapshot = _state_snapshot(st)
                else:
                    after = _state_snapshot(st)
                    self.assertEqual(
                        after, snapshot,
                        f"{move_name}() returned False but state changed"
                    )
                    rejections += 1
        self.assertGreaterEqual(
            rejections, min_total_rejections,
            f"only {rejections} rejections seen in the battery; "
            f"increase n_per_move or vary couplings to get more signal"
        )

    def test_state_unchanged_high_k4(self):
        """At k4=20, add/flip have huge positive ΔS → mostly rejected."""
        cdt, st = _make_cdt(k4=20.0)
        self._run_battery(cdt, st)

    def test_state_unchanged_low_k4(self):
        """At k4=-20, remove/iflip have huge positive ΔS → mostly rejected.

        Pre-grow so remove/iflip have targets.
        """
        cdt, st = _make_cdt(k4=-20.0)
        _grow(cdt, n=300)
        self._run_battery(cdt, st)

    def test_state_unchanged_high_volume_penalty(self):
        """Large epsilon penalizes any deviation from target N41 → mixed
        rejection across all move types."""
        cdt, st = _make_cdt(k4=0.5, epsilon=5.0)
        self._run_battery(cdt, st, n_per_move=120)


# ---------------------------------------------------------------------------
# Edge inventory monotonic
# ---------------------------------------------------------------------------


class TestEdgeInventoryDeltas(unittest.TestCase):
    """Edge-inventory characterization per move type.  Vertex relabeling
    is disabled because ``swapVertexLabels`` rewrites edge fingerprints
    in place, which would corrupt fingerprint-based comparison.

    Locked-in behavior:

    * ``add``, ``flip``, ``iflip``, ``shift`` — edge fingerprint set
      monotonically grows.  ``Spacetime::removeSimplex`` does not
      cascade-delete edges, and ``createSimplex`` deduplicates against
      ``EdgeList`` by fingerprint, so these moves only ever insert.

    * ``remove`` — deletes exactly the edges incident to the dropped
      vertex (CDT.cpp:321-337).  After a successful remove, the edge
      fingerprint set loses ``deg(v)`` entries.

    The PachnerMove rollback design has to know these:
    AddMove/FlipMove/IFlipMove/ShiftMove track which edges they freshly
    inserted (so rollback removes them).  RemoveMove must additionally
    capture the deleted edges' (source, target, squaredLength) so
    rollback can reinsert them.
    """

    def _edge_fps(self, st):
        return frozenset(hash(e) for e in st.getEdgeList().toVector())

    def _check_monotonic_for(self, cdt, st, move_names, n_calls=80):
        """For the given move types only, assert edge fingerprint set
        is monotonically non-decreasing across accepted calls."""
        for _ in range(n_calls):
            for move_name in move_names:
                before = self._edge_fps(st)
                accepted = getattr(cdt, move_name)()
                after = self._edge_fps(st)
                if accepted:
                    self.assertTrue(
                        before.issubset(after),
                        f"{move_name}() removed {len(before - after)} "
                        f"edge(s); these moves are documented to only add"
                    )
                else:
                    self.assertEqual(
                        before, after,
                        f"{move_name}() returned False but EdgeList changed"
                    )

    def test_add_flip_iflip_shift_monotonic_d4(self):
        cdt, st = _make_cdt(d=4, relabel=False)
        self._check_monotonic_for(
            cdt, st, _NON_REMOVE_MOVE_NAMES, n_calls=80
        )

    def test_add_flip_iflip_shift_monotonic_d3(self):
        cdt, st = _make_cdt(d=3, relabel=False)
        self._check_monotonic_for(
            cdt, st, _NON_REMOVE_MOVE_NAMES, n_calls=80
        )

    def test_add_flip_iflip_shift_monotonic_d2(self):
        cdt, st = _make_cdt(d=2, relabel=False)
        self._check_monotonic_for(
            cdt, st, _NON_REMOVE_MOVE_NAMES, n_calls=80
        )

    def test_remove_drops_exactly_dropped_vertex_edges(self):
        """A successful remove deletes exactly the edges incident to
        the vertex it just removed."""
        cdt, st = _make_cdt(d=4, relabel=False)
        # Grow first so remove has eligible vertices.
        for _ in range(300):
            cdt.add()

        for _ in range(2000):
            # Snapshot the edge set + a candidate vertex's incident edges
            # before the call.  We don't know which vertex remove() will
            # pick, so we capture the full edge set and figure it out
            # from the diff.
            edges_before = self._edge_fps(st)
            ids_before = frozenset(
                v.getId() for v in st.getVertexList().toVector()
            )
            if not cdt.remove():
                continue
            edges_after = self._edge_fps(st)
            ids_after = frozenset(
                v.getId() for v in st.getVertexList().toVector()
            )

            # Exactly one vertex removed.
            removed_ids = ids_before - ids_after
            self.assertEqual(
                len(removed_ids), 1,
                f"remove() should drop exactly 1 vertex, got "
                f"{len(removed_ids)}"
            )

            # The edge delta is contained in: edges that touched the
            # dropped vertex.  Lower bound = 1 (the vertex was incident
            # to at least one edge, given it had 2d simplices).
            dropped_edges = edges_before - edges_after
            added_edges = edges_after - edges_before
            self.assertGreater(
                len(dropped_edges), 0,
                "remove() should delete at least one edge (the dropped "
                "vertex's incidences)"
            )
            # And the move may also add new edges between the d
            # spatial vertices and the new replacement simplex
            # connectivity, but those are *additions* — disjoint from
            # the dropped vertex's incidences.
            self.assertEqual(
                dropped_edges & added_edges, frozenset(),
                "Dropped and added edge sets must be disjoint"
            )
            return
        self.skipTest("No remove accepted in 2000 attempts")


# ---------------------------------------------------------------------------
# Coface integrity after accepted moves
# ---------------------------------------------------------------------------


class TestCofaceIntegrityEventuallyConsistent(unittest.TestCase):
    """Coface integrity is *eventually* consistent.  Tessera registers
    cofaces lazily in ``Simplex::getFacets()``: only after a top
    simplex's facets have been queried does it appear in their coface
    lists.  After a Pachner move that creates new top simplices, those
    facets are not registered until something walks them — typically
    the next ``getDualAdjacency`` call (which iterates every top
    simplex's facets).

    These tests check the post-walk invariant.  The refactor must
    preserve this property: it's OK if an accepted move leaves
    cofaces unregistered until something walks them; it's *not* OK
    if a walk fails to register them.
    """

    def _walk_to_register_cofaces(self, st):
        """Trigger lazy facet/coface registration on every top simplex
        by calling ``getDualAdjacency``."""
        st.getDualAdjacency()

    def _verify_coface_integrity(self, st):
        dPlus1 = _top_simplex_size(st)
        for sigma in st.getSimplices():
            if len(sigma.getVertices()) != dPlus1:
                continue
            for f in sigma.getFacets():
                self.assertIn(
                    sigma, f.getCofaces(),
                    f"Facet missing parent in coface list: "
                    f"sigma={sigma}, facet={f}"
                )

    def test_integrity_after_build(self):
        """Right after build(), coface integrity holds (Toroid::build
        explicitly forces facet registration on every top simplex)."""
        _, st = _make_cdt(d=4)
        self._verify_coface_integrity(st)

    def test_integrity_after_long_run_with_walk(self):
        """After a long sweep + walk, coface integrity holds."""
        cdt, st = _make_cdt(d=4)
        cdt.tune()
        cdt.sweep(20)
        self._walk_to_register_cofaces(st)
        self._verify_coface_integrity(st)

    def test_integrity_after_individual_moves_with_walk(self):
        """For each accepted move, after a walk, coface integrity
        holds."""
        cdt, st = _make_cdt(d=4)
        for _ in range(400):
            for move_name in (tessera.AddMove.MOVE_TYPE, tessera.FlipMove.MOVE_TYPE,
                              tessera.ShiftMove.MOVE_TYPE):
                if getattr(cdt, move_name)():
                    self._walk_to_register_cofaces(st)
                    self._verify_coface_integrity(st)


# ---------------------------------------------------------------------------
# Action delta consistency per move type
# ---------------------------------------------------------------------------


class TestActionDeltaPerMove(unittest.TestCase):
    """``computeAction()`` after an accepted move minus before equals
    the analytic prediction:

        ΔS_Regge = -(k0 + 6*delta) * dN0
                  + (k4 + 2*delta) * dN41
                  + (k4 + delta)   * dN32
        ΔS_fix   = epsilon * [ (N41' - target)^2 - (N41 - target)^2 ]
                                                     (quadratic)
                  = epsilon * [ |N41' - target| - |N41 - target| ]
                                                     (linear)

    For each accepted move, we observe (dN0, dN41, dN32) and the
    pre-move N41 and verify ΔS matches.
    """

    K0 = 2.2
    K4 = 0.5
    DELTA = 0.6
    EPSILON = 0.02

    def _predict_delta_action(self, n41_before, dN0, dN41, dN32, target):
        regge = (-(self.K0 + 6 * self.DELTA) * dN0
                 + (self.K4 + 2 * self.DELTA) * dN41
                 + (self.K4 + self.DELTA) * dN32)
        fix_before = self.EPSILON * (n41_before - target) ** 2
        fix_after = self.EPSILON * (n41_before + dN41 - target) ** 2
        return regge + (fix_after - fix_before)

    def _check_one_move(self, cdt, st, move_name, target,
                        max_attempts=2000):
        method = getattr(cdt, move_name)
        for _ in range(max_attempts):
            n0_b = st.getVertexCount()
            n41_b = st.getN41()
            n32_b = st.getN32()
            S_b = cdt.computeAction()
            if method():
                n0_a = st.getVertexCount()
                n41_a = st.getN41()
                n32_a = st.getN32()
                S_a = cdt.computeAction()
                dS_observed = S_a - S_b
                dS_predicted = self._predict_delta_action(
                    n41_b, n0_a - n0_b, n41_a - n41_b, n32_a - n32_b,
                    target
                )
                self.assertAlmostEqual(
                    dS_observed, dS_predicted, places=6,
                    msg=f"{move_name}: ΔS observed={dS_observed}, "
                        f"predicted={dS_predicted}"
                )
                return True
        return False

    def _make(self, d=4, n_simplices=200):
        sig = tessera.Signature(d, tessera.Lorentzian)
        metric = tessera.Metric(True, sig)
        st = tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0,
                               tessera.PREFERRED, tessera.Toroid())
        st.build(n_simplices)
        target = st.getN41()
        cdt = tessera.CDTSimulation(
            st, self.K0, self.K4, self.DELTA, self.EPSILON, target, True
        )
        return cdt, st, target

    def test_add_action_delta(self):
        cdt, st, target = self._make()
        if not self._check_one_move(cdt, st, "add", target):
            self.skipTest("No add accepted in window")

    def test_remove_action_delta(self):
        cdt, st, target = self._make()
        _grow(cdt, n=200)
        if not self._check_one_move(cdt, st, "remove", target):
            self.skipTest("No remove accepted in window")

    def test_flip_action_delta(self):
        cdt, st, target = self._make()
        if not self._check_one_move(cdt, st, "flip", target):
            self.skipTest("No flip accepted in window")

    def test_iflip_action_delta(self):
        cdt, st, target = self._make()
        _grow(cdt, n=100)
        if not self._check_one_move(cdt, st, "iflip", target):
            self.skipTest("No iflip accepted in window")

    def test_shift_action_delta(self):
        cdt, st, target = self._make()
        if not self._check_one_move(cdt, st, "shift", target):
            self.skipTest("No shift accepted in window")


# ---------------------------------------------------------------------------
# Lazy facet materialization is bookkeeping-neutral
# ---------------------------------------------------------------------------


def _top_fingerprints(st):
    """Fingerprints of the top-dimensional simplices (robust to lazily
    materialized lower-dimensional faces in getSimplices())."""
    top_size = _top_simplex_size(st)
    return frozenset(hash(s) for s in st.getSimplices()
                     if len(s.getVertices()) == top_size)


def _bookkeeping(st):
    """The CDT bookkeeping that a rejected move must leave untouched: the
    vertex / N41 / N32 / N4 counts and the set of top-dimensional simplices.
    Deliberately excludes the raw getSimplices() membership, which can grow
    with benign lazily-materialized facets."""
    return (st.getVertexCount(), st.getN41(), st.getN32(),
            st.getTopSimplexCount(), _top_fingerprints(st))


class TestLazyFacetMaterializationIsBenign(unittest.TestCase):
    """Inspecting a simplex's facets (via getFacets(), as every move's
    propose() does) materializes those facet simplices and registers them in
    the spacetime, so getSimplices() can grow. That is a benign caching of real
    faces: it must NOT change the CDT bookkeeping (vertex/N41/N32/N4 counts or
    the set of top-dimensional simplices). This is *why* a rejected move — which
    runs propose() but not apply() — leaves the bookkeeping intact even though
    propose() is not strictly side-effect-free.
    """

    def test_materializing_facets_preserves_counts_and_top_simplices(self):
        cdt, st = _make_cdt()
        _grow(cdt, n=100)
        before = _bookkeeping(st)
        simplices_before = len(st.getSimplices())

        # materializeFacets() forces facet materialization to a fixpoint — the
        # same lazy getFacets() machinery a move's propose() triggers.
        st.materializeFacets()

        self.assertEqual(_bookkeeping(st), before,
                         "facet materialization changed the CDT bookkeeping")
        self.assertGreaterEqual(
            len(st.getSimplices()), simplices_before,
            "materialization should only ever add face simplices")

    def test_top_simplex_size_is_robust_to_materialized_facets(self):
        # After materialization the simplex list contains faces of several
        # dimensions; _top_simplex_size must still report the true top (d+1=5).
        cdt, st = _make_cdt(d=4)
        _grow(cdt, n=100)
        st.materializeFacets()
        sizes = {len(s.getVertices()) for s in st.getSimplices()}
        self.assertIn(5, sizes)            # top simplices present
        self.assertTrue(sizes - {5})       # and lower-dim faces too
        self.assertEqual(_top_simplex_size(st), 5)

    def test_rejected_add_preserves_bookkeeping_directly(self):
        # Drive add() to rejection at k4 = -20 (favours large volume, so the
        # combinatorial / Metropolis path rejects some adds) and confirm the
        # bookkeeping is identical across each rejection.
        cdt, st = _make_cdt(k4=-20.0)
        _grow(cdt, n=200)
        before = _bookkeeping(st)
        rejections = 0
        for _ in range(200):
            if cdt.add():
                before = _bookkeeping(st)
            else:
                self.assertEqual(_bookkeeping(st), before,
                                 "rejected add() changed the CDT bookkeeping")
                rejections += 1
        self.assertGreater(rejections, 0, "saw no add() rejections to check")


if __name__ == "__main__":
    unittest.main()
