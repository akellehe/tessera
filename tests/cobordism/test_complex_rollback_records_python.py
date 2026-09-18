"""Complex+phase-exact rollback records (#581 scope items 2 and 3).

Three of the four rollback families restored edges from Re-only records,
violating their own "restore bit-exactly" contracts on analytically continued
(``Im l^2 != 0``) or phase-carrying geometry: every REJECTED move silently
projected the complex onto the real axis.  This module covers the
``SurgicalCone`` and ``EigenstateSynthesis`` families (the ``RemoveMove``
family lives with the other Pachner tests in
``tests/test_pachner_remove_complex_rollback.py``) plus the Im-aware dW pin
gates:

* ``SurgicalCone.coneOut`` + ``rollback`` restores an ORPHANED edge's full
  complex ``l^2`` and U(1) phase bit-exactly (a window-adjacent face of the
  holed icosahedron orphans its shared rim edge — the deterministic orphan);
* a REJECTED ``coneOut`` (a pinching removal) leaves the complex-valued
  geometry bit-identical — the ``directedConeOut/In`` hot path, where probes
  roll back tens of times per iteration;
* ``coneIn`` + ``rollback`` drops the fresh edges and leaves every surviving
  edge untouched;
* ``EigenstateSynthesis.removeInteriorCell`` + ``restoreLastRemoval`` is
  bit-exact on Im+phase-carrying geometry (note: an interior cell's faces all
  carry >= 2 cofaces, so such a removal can never orphan an edge — the
  complex-valued ``Removal`` record is correctness for the reject/restore
  path, exercised here through the round trip);
* the dW pin gates hold with complex boundary values: an Im+phase-carrying
  pinned boundary is ACCEPTED unchanged through an interior attach (positive
  control — the full-complex comparison does not false-reject), and an attach
  that would take a pinned dW edge out of the boundary is REJECTED with the
  complex geometry restored bit-exactly;
* action-invariance round trips on real signed-l^2 (timelike-carrying)
  geometry — the hinge-exactness contract extended to the surgical cone on a
  mixed-causal-character host.
"""

import pytest

import tessera as T
import cmath

cob = T.cobordism

# The standard icosahedron (12 vertices, 20 faces) and its three mutually
# vertex-disjoint windows — the _holed_surface fixture's constants.
ICOSA_FACES = [
    [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
    [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
    [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
    [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
]
WINDOWS = [[0, 11, 5], [3, 2, 6], [9, 8, 1]]


def _holed_icosa():
    rm = {tuple(sorted(t)) for t in WINDOWS}
    faces = [f for f in ICOSA_FACES if tuple(sorted(f)) not in rm]
    st = T.Spacetime.fromVertexTuples(2, faces, 1.0, 0.0)
    st.materializeFacets()
    return st


def _full_icosa():
    st = T.Spacetime.fromVertexTuples(2, ICOSA_FACES, 1.0, 0.0)
    st.materializeFacets()
    return st


def _seed_complex_geometry(st):
    """Distinct synthetic Im l^2 != 0 and phase != 0 on every edge, keyed off
    the endpoint ids so the values are order-independent."""
    for e in st.getEdgeList().toVector():
        a, b = e.getSource().getId(), e.getTarget().getId()
        lo, hi = min(a, b), max(a, b)
        e.setLength(cmath.sqrt(complex(complex(1.0 + 0.01 * lo, 0.05 + 0.01 * hi))))
        e.setPhase(0.1 + 0.003 * (lo * 13 + hi))


def _seed_signed_geometry(st, timelike_keys):
    """Real signed l^2 (some timelike), zero Im, nonzero phases."""
    for e in st.getEdgeList().toVector():
        a, b = e.getSource().getId(), e.getTarget().getId()
        k = (min(a, b), max(a, b))
        e.setLength(cmath.sqrt(complex(-1.5 if k in timelike_keys else 1.0 + 0.01 * k[0])))
        e.setPhase(0.2 + 0.001 * k[1])


def _edge_state(st):
    """{(a, b): (complex l^2, phase)} over the live edge list."""
    out = {}
    for e in st.getEdgeList().toVector():
        a, b = e.getSource().getId(), e.getTarget().getId()
        out[(min(a, b), max(a, b))] = (complex(e.getLength()**2),
                                       e.getPhase())
    return out


def _tops(st):
    return sorted(tuple(sorted(v.getId() for v in s.getVertices()))
                  for s in st.getTopSimplices())


def _hinge_dual_volumes(st):
    """{vertex id: dualVolume} over the registered 0-simplices — the
    (d-2)-hinges of this 2D host. Reads the circumcentric coface walk, so it
    is sensitive to the facet/coface bookkeeping, not just to edge values."""
    out = {}
    for s in st.getSimplices():
        if len(s.getVertices()) == 1:
            out[s.getVertices()[0].getId()] = s.dualVolume()
    return out


def _registered_faces(st, n_vertices):
    """Sorted vertex-id tuples of the registered simplices carrying exactly
    ``n_vertices`` vertices."""
    return sorted(tuple(sorted(v.getId() for v in s.getVertices()))
                  for s in st.getSimplices()
                  if len(s.getVertices()) == n_vertices)


def _assert_dual_volumes_equal(before, after, tol=1e-12):
    assert set(after) == set(before), (
        f"hinge set drifted: only-before={set(before) - set(after)} "
        f"only-after={set(after) - set(before)}")
    for vid, dv in before.items():
        assert abs(after[vid] - dv) < tol, (
            f"hinge v{vid}: dualVolume {dv!r} -> {after[vid]!r}")


def _assert_state_equal(before, after, allow_missing=()):
    assert set(after) == set(before) - set(allow_missing), (
        f"edge set drifted: only-before={set(before) - set(after)} "
        f"only-after={set(after) - set(before)}")
    for k, (sq, ph) in after.items():
        sq0, ph0 = before[k]
        assert sq == sq0, f"edge {k}: l^2 {sq0!r} -> {sq!r} (not bit-exact)"
        assert ph == ph0, f"edge {k}: phase {ph0!r} -> {ph!r} (not bit-exact)"


# --------------------------------------------------------------------------- #
# SurgicalCone
# --------------------------------------------------------------------------- #
def test_cone_out_rollback_restores_orphaned_complex_edge_bitexact():
    # Face [5,11,4] is the sole surviving coface of the window rim edge
    # (5,11) (its other coface was the removed window [0,11,5]), and its
    # third vertex 4 is interior, so coning it out is accepted (the hole
    # grows to a quad disk) and ORPHANS (5,11): the rollback must re-create
    # that edge from the record — full complex l^2 AND phase, not a Re-only
    # projection.
    st = _holed_icosa()
    _seed_complex_geometry(st)
    before = _edge_state(st)
    tops_before = _tops(st)

    sc = cob.SurgicalCone(st)
    ok, reason = sc.coneOut([5, 11, 4])
    assert ok, reason
    assert (5, 11) not in _edge_state(st), "rim edge (5,11) must be orphaned"

    assert sc.rollback()
    _assert_state_equal(before, _edge_state(st))
    assert _tops(st) == tops_before


def test_rejected_cone_out_leaves_complex_geometry_bitexact():
    # Face [0,1,7] shares only vertex 0 with the window hole [0,11,5]:
    # removing it would pinch vertex 0's link into two arcs, so the gate
    # rejects and the auto-rollback must leave the complex-valued geometry
    # bit-identical (the directedConeOut hot-path contract).
    st = _holed_icosa()
    _seed_complex_geometry(st)
    before = _edge_state(st)
    tops_before = _tops(st)

    sc = cob.SurgicalCone(st)
    ok, reason = sc.coneOut([0, 1, 7])
    assert not ok, "removal pinching the hole rim must be rejected"
    assert sc.depth == 0
    _assert_state_equal(before, _edge_state(st))
    assert _tops(st) == tops_before


def test_cone_in_rollback_drops_fresh_edges_and_keeps_survivors_bitexact():
    st = _holed_icosa()
    _seed_complex_geometry(st)
    before = _edge_state(st)
    tops_before = _tops(st)

    sc = cob.SurgicalCone(st)
    ok, reason = sc.coneIn([0, 5])  # cap part of the [0,11,5] window
    if not ok:
        pytest.skip(f"cone-in rejected on this host: {reason}")
    assert set(_edge_state(st)) > set(before), "cone-in must add fresh edges"

    assert sc.rollback()
    _assert_state_equal(before, _edge_state(st))
    assert _tops(st) == tops_before


def test_cone_round_trip_preserves_complex_action_on_timelike_host():
    # Real signed l^2 with timelike edges (Im l^2 = 0, so the geometry stack
    # may evaluate): the accepted cone-out round trip must retrace the full
    # COMPLEX dual Regge action, Re and Im — the hinge-exactness contract on
    # a mixed-causal-character host.  Closed host (the full icosahedron), the
    # domain the existing hinge-exactness suites cover; the already-holed
    # host, where the rollback must additionally restore an ORPHANED edge and
    # the facet/coface bookkeeping around it (#587), is covered by
    # test_cone_round_trip_retraces_dual_volumes_on_holed_host below.
    st = _full_icosa()
    _seed_signed_geometry(st, timelike_keys={(0, 1), (7, 10)})
    s0 = complex(T.ReggeSolver(st, T.MatterConfiguration()).dualReggeAction())
    assert abs(s0.imag) > 1e-9, "fixture is not genuinely Lorentzian"
    before = _edge_state(st)

    sc = cob.SurgicalCone(st)
    ok, reason = sc.coneOut([5, 11, 4])
    assert ok, reason
    assert sc.rollback()

    _assert_state_equal(before, _edge_state(st))
    s1 = complex(T.ReggeSolver(st, T.MatterConfiguration()).dualReggeAction())
    assert abs(s1.real - s0.real) < 1e-9
    assert abs(s1.imag - s0.imag) < 1e-9


def test_cone_round_trip_retraces_dual_volumes_on_holed_host():
    # The #587 repro. Face [5,11,4] is the sole surviving coface of window
    # rim edge (5,11), so coning it out orphans that edge. Removing the edge
    # must also remove the registered (5,11) 1-simplex (a face that outlived
    # its edge would read l^2 = 0 in every later Gram-matrix computation),
    # and the rollback must restore the facet/coface lattice along with the
    # edge values — every hinge's dualVolume and the complex dual action
    # retrace IMMEDIATELY, not only after a global re-materialization.
    st = _holed_icosa()
    _seed_signed_geometry(st, timelike_keys={(0, 1), (7, 10)})
    s0 = complex(T.ReggeSolver(st, T.MatterConfiguration()).dualReggeAction())
    dv0 = _hinge_dual_volumes(st)
    before = _edge_state(st)
    faces0 = _registered_faces(st, 2)
    tops_before = _tops(st)

    sc = cob.SurgicalCone(st)
    ok, reason = sc.coneOut([5, 11, 4])
    assert ok, reason
    # No zombie: the orphaned rim edge's 1-simplex is pruned with the edge.
    assert (5, 11) not in _registered_faces(st, 2)

    assert sc.rollback()
    _assert_state_equal(before, _edge_state(st))
    assert _tops(st) == tops_before
    assert _registered_faces(st, 2) == faces0
    _assert_dual_volumes_equal(dv0, _hinge_dual_volumes(st))
    s1 = complex(T.ReggeSolver(st, T.MatterConfiguration()).dualReggeAction())
    assert abs(s1.real - s0.real) < 1e-12
    assert abs(s1.imag - s0.imag) < 1e-12
    # A later global re-materialization must not shift anything either (the
    # pre-fix failure mode: it wired a pruned-edge zombie back into the walk).
    st.materializeFacets()
    _assert_dual_volumes_equal(dv0, _hinge_dual_volumes(st))


def test_rejected_cone_out_retraces_dual_volumes():
    # The directedConeOut hot path scores dF across rejected probes on the
    # SAME spacetime; the auto-rollback must leave every hinge's dualVolume
    # unchanged or every later probe sees a phantom dual-volume delta (#587).
    # Face [0,1,7] shares only vertex 0 with the window hole [0,11,5], so the
    # gate rejects it (pinched link) after the cell was already removed.
    st = _holed_icosa()
    _seed_signed_geometry(st, timelike_keys={(0, 1), (7, 10)})
    dv0 = _hinge_dual_volumes(st)

    sc = cob.SurgicalCone(st)
    ok, _ = sc.coneOut([0, 1, 7])
    assert not ok, "removal pinching the hole rim must be rejected"
    assert sc.depth == 0
    _assert_dual_volumes_equal(dv0, _hinge_dual_volumes(st))


def test_cone_in_rollback_after_materialization_leaves_no_orphan_faces():
    # If the lattice is materialized between an accepted cone-in and its
    # rollback (e.g. a solver scoring the probe), the fresh cell's faces
    # exist by then; the rollback must drop them with the cell and its fresh
    # edges — faces shared with surviving cells stay — so the registered
    # simplex set returns exactly to the pre-move closure.
    st = _holed_icosa()
    _seed_complex_geometry(st)
    before = _edge_state(st)
    n_simplices_before = len(st.getSimplices())
    faces0 = _registered_faces(st, 2)

    sc = cob.SurgicalCone(st)
    ok, reason = sc.coneIn([0, 5])  # cap part of the [0,11,5] window
    if not ok:
        pytest.skip(f"cone-in rejected on this host: {reason}")
    st.materializeFacets()

    assert sc.rollback()
    _assert_state_equal(before, _edge_state(st))
    assert _registered_faces(st, 2) == faces0
    assert len(st.getSimplices()) == n_simplices_before


# --------------------------------------------------------------------------- #
# EigenstateSynthesis: removeInteriorCell / restoreLastRemoval
# --------------------------------------------------------------------------- #
def test_remove_interior_cell_restore_is_bitexact_on_complex_geometry():
    # The full icosahedron is closed: dW is empty, every vertex interior, so
    # any face is a removable interior cell.  The restore must bring back the
    # cell with every surviving edge's complex l^2 + phase untouched.
    st = _full_icosa()
    es = cob.EigenstateSynthesis(st, 1)
    _seed_complex_geometry(st)
    before = _edge_state(st)
    tops_before = _tops(st)

    assert es.removeInteriorCell([0, 5, 1])
    assert es.restoreLastRemoval()
    _assert_state_equal(before, _edge_state(st))
    assert _tops(st) == tops_before


def test_remove_interior_cell_boundary_pre_gate_rejects_bitexact():
    # On the holed host every face touches a hole-rim (boundary) vertex, so
    # the interiority pre-gate rejects and nothing may move — including Im.
    st = _holed_icosa()
    es = cob.EigenstateSynthesis(st, 1)
    _seed_complex_geometry(st)
    before = _edge_state(st)

    assert es.interiorTopCells() == []
    assert not es.removeInteriorCell([0, 5, 1])
    _assert_state_equal(before, _edge_state(st))


# --------------------------------------------------------------------------- #
# The Im-aware dW pin gates
# --------------------------------------------------------------------------- #
def test_dw_pin_gate_accepts_interior_attach_with_complex_boundary():
    # Positive control: the pinned dW carries Im l^2 != 0 and phase != 0; an
    # attach that only touches interior edges (the stellar fan over the fully
    # interior-edged face {0,7,10}) must be ACCEPTED — the full-complex
    # comparison agrees with itself — and detach must restore bit-exactly.
    st = _holed_icosa()
    es = cob.EigenstateSynthesis(st, 1)
    _seed_complex_geometry(st)
    before = _edge_state(st)
    n_boundary = es.numBoundaryEdges()
    assert n_boundary > 0

    assert es.attachInteriorVertex([[0, 7], [7, 10], [0, 10]])
    # dW is value-identical through the accepted attach
    after = _edge_state(st)
    for k in before:
        assert after[k] == before[k], f"edge {k} drifted through the attach"

    assert es.detachLastInteriorVertex()
    _assert_state_equal(before, _edge_state(st))


def test_dw_pin_gate_rejects_attach_touching_boundary_and_restores_im():
    # An attach onto the pinned dW edge (0,5) would pull it out of the
    # boundary (a second coface) — the gate must reject it and the rollback
    # must leave the complex-valued geometry bit-identical, Im included.
    st = _holed_icosa()
    es = cob.EigenstateSynthesis(st, 1)
    _seed_complex_geometry(st)
    before = _edge_state(st)
    boundary = {tuple(k) for k in es.boundaryEdges()}
    assert (0, 5) in boundary

    assert not es.attachInteriorVertex([[0, 5]])
    _assert_state_equal(before, _edge_state(st))
    assert {tuple(k) for k in es.boundaryEdges()} == boundary
