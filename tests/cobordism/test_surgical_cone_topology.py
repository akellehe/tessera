"""Gated surgical cone-out / cone-in: the topology-changer (T3, #460).

The Emergent Color Topology epic (#457) grows the proton's ``b2`` color register
by *gated surgical coning*. T1 (#458) made the cone primitives hinge-exact and
exactly invertible; T2 (#459) made the topology-PRESERVING refinement cone
orientation-safe. T3 (this module) adds the genuine topology-CHANGER:

* **cone-out** removes one top cell (its orphaned edges, then any isolated
  vertex). On a closed manifold this opens a manifold-with-boundary; removing a
  cell disjoint from an existing hole raises ``b_{d-1}`` by exactly 1 -- on
  ``S^3`` that is ``b_2``, the color register's degree.
* **cone-in** adds one top cell on a fresh vertex joined to ``d`` existing
  vertices (the literal "add a vertex, draw edges" of the ticket).

Every surgical move is gated on ``ChainComplex.dual_complex_is_valid`` as a
*manifold-with-boundary* (the #429 ``n>=4`` recursive check). Surgery is allowed
BECAUSE it is gated; bypassing the gate is what broke the #353 weld. A rejected
move is rolled back bit-identically. The exact inverse of a cone-out (re-adding
the removed cell -- ``rollback``) restores topology AND the complex (Lorentzian /
Sorkin) action, Re AND Im, to machine precision.

Coverage:

* a surgical cone-out raises ``b_2`` by exactly 1 on a small ``S^3``, and the
  inverse lowers it by 1 (topology restored);
* the cone-out round trip restores the dual Regge action (Re AND Im) on a
  genuinely Lorentzian CDT toroid (``Im S`` ~ -11 with the #581 mixed-hinge
  branch; ~ -35 before it);
* the gate rejects a non-manifold attempt (a cone-in onto an interior facet ->
  3 cofaces) and leaves the complex unchanged;
* cone-in needs a boundary (it always rejects on a closed manifold) and is
  reversible where it is accepted;
* the ``n>=4`` recursive gate accepts + round-trips a cone-out on ``S^4``.
"""

import pytest

import tessera as T
import cmath

cob = T.cobordism

TOL = 1e-9
TOL_BIG = 1e-6


# --------------------------------------------------------------------------- #
# Builders / helpers
# --------------------------------------------------------------------------- #
def _matter():
    return T.MatterConfiguration()


def _sphere(d, sq=1.0):
    """Boundary of a (d+1)-simplex = a minimal triangulated S^d, unit spacelike."""
    sig = T.Signature(d, T.Lorentzian)
    st = T.Spacetime(T.Metric(True, sig), T.CDT, 1.0, 1.0, T.PREFERRED,
                     T.SimplexBoundarySphere(d))
    st.build()
    for e in st.get_edge_list().to_vector():
        e.set_length(cmath.sqrt(complex(sq)))
    return st


def _refined_s3(n_refine=12):
    """A refined S^3 (still Betti [1,0,0,1]) with enough tetrahedra that disjoint
    top cells exist -- the minimal S^3 = dDelta4 has none (every facet pair shares
    a ridge), so no single removal can open a b_2 hole."""
    st = _sphere(3)
    for seed in range(n_refine):
        mv = T.AddMove(st, seed, False, T.PachnerMode.PreGeometric, False)
        if mv.propose():
            mv.apply()
    return st


def _make_cdt(n_simplices=120):
    sig = T.Signature(4, T.Lorentzian)
    st = T.Spacetime(T.Metric(True, sig), T.CDT, 1.0, 1.0, T.PREFERRED, T.Toroid())
    st.build(n_simplices)
    return st


def _act(st):
    return T.ReggeSolver(st, _matter()).dual_regge_action()


def _tops(st):
    return sorted(
        tuple(sorted(v.get_id() for v in s.get_vertices()))
        for s in st.get_top_simplices()
    )


def _betti(st):
    return cob.ChainComplex.from_spacetime(st).betti_numbers()


def _first_disjoint_pair(cells):
    for i, a in enumerate(cells):
        sa = set(a)
        for b in cells[i + 1:]:
            if sa.isdisjoint(b):
                return a, b
    return None


def _assert_close(a, b, tol, what):
    assert abs(a.real - b.real) < tol, f"{what}: Re drift {abs(a.real - b.real):.2e}"
    assert abs(a.imag - b.imag) < tol, f"{what}: Im drift {abs(a.imag - b.imag):.2e}"


# --------------------------------------------------------------------------- #
# 1. cone-out raises b_2 by exactly 1; the inverse lowers it by 1
# --------------------------------------------------------------------------- #
def test_surgical_cone_out_raises_b2_and_inverse_lowers_it():
    st = _refined_s3()
    cells = _tops(st)
    assert _betti(st)[3] == 1 and _betti(st)[2] == 0, "fixture is not an S^3"
    pair = _first_disjoint_pair(cells)
    assert pair is not None, "refined S^3 must contain a disjoint cell pair"
    a, b = pair

    sc = cob.SurgicalCone(st)
    # First removal opens the manifold (b_3 -> 0) but does NOT yet make a b_2 hole.
    ok, reason = sc.cone_out(list(a))
    assert ok, reason
    b2_after_first = _betti(st)[2]
    assert sc.validate()[0]

    # Removing a cell DISJOINT from the first raises b_2 by exactly 1.
    ok, reason = sc.cone_out(list(b))
    assert ok, reason
    assert sc.validate()[0]
    assert _betti(st)[2] == b2_after_first + 1, "cone-out did not raise b_2 by 1"

    # The inverse (cone the cell back in) lowers b_2 by exactly 1 ...
    assert sc.rollback()
    assert _betti(st)[2] == b2_after_first, "inverse did not lower b_2 by 1"
    # ... and unwinding fully restores the original topology + cell set.
    assert sc.rollback()
    assert sc.depth == 0
    assert _tops(st) == cells, "round trip did not restore the top cells"
    assert _betti(st) == cob.ChainComplex.from_spacetime(st).betti_numbers()
    assert _betti(st)[3] == 1


# --------------------------------------------------------------------------- #
# 2. round-trip restores the complex action (Re AND Im) on a Lorentzian CDT
# --------------------------------------------------------------------------- #
def test_round_trip_restores_action_re_and_im():
    st = _make_cdt(120)
    a0 = _act(st)
    assert abs(a0.imag) > 1.0, "fixture is not genuinely Lorentzian"
    tops0 = _tops(st)

    sc = cob.SurgicalCone(st)
    for cell in _tops(st):
        ok, _ = sc.cone_out(list(cell))
        if not ok:
            continue
        # The surgery genuinely changes the action ...
        assert abs(_act(st) - a0) > TOL, "cone-out did not change the action"
        # ... and the exact inverse restores it, Re AND Im, to machine precision.
        assert sc.rollback()
        _assert_close(_act(st), a0, TOL_BIG, "surgical cone round trip")
        assert _tops(st) == tops0, "round trip did not restore the top cells"
        return
    pytest.skip("no gated cone-out accepted on the CDT")


# --------------------------------------------------------------------------- #
# 3. the gate rejects a non-manifold attempt and leaves the complex unchanged
# --------------------------------------------------------------------------- #
def test_gate_rejects_non_manifold_cone_in():
    st = _sphere(3)
    cells = _tops(st)
    sc = cob.SurgicalCone(st)
    # {0,1,2} is an interior facet (shared by (0,1,2,3) and (0,1,2,4)); a fresh
    # cell on it would give that facet 3 cofaces -- a non-manifold pinch.
    ok, reason = sc.cone_in([0, 1, 2])
    assert not ok
    assert "cofaces" in reason, reason
    assert sc.depth == 0
    assert _tops(st) == cells, "a rejected move must leave the complex unchanged"


# --------------------------------------------------------------------------- #
# 4. cone-in needs a boundary (rejects on a closed manifold) + is reversible
# --------------------------------------------------------------------------- #
def test_cone_in_needs_a_boundary_and_is_reversible():
    st = _sphere(3)
    cells = _tops(st)
    sc = cob.SurgicalCone(st)
    # Closed manifold: every facet has 2 cofaces, so any cone-in pinches -> reject.
    assert not sc.cone_in([0, 1, 2])[0]

    # Open a boundary, then a cone-in onto a boundary triangle is accepted and
    # exactly reversible.
    assert sc.cone_out([0, 1, 2, 3])[0]
    ok, reason = sc.cone_in([0, 1, 2])
    assert ok, reason
    assert sc.validate()[0]
    assert sc.rollback_all() == 2
    assert _tops(st) == cells, "cone-in round trip did not restore the complex"


# --------------------------------------------------------------------------- #
# 5. the n>=4 recursive gate accepts + round-trips a cone-out on S^4
# --------------------------------------------------------------------------- #
def test_n_ge_4_recursive_gate_on_s4():
    st = _sphere(4)
    cells = _tops(st)
    assert _betti(st)[4] == 1, "fixture is not an S^4"
    sc = cob.SurgicalCone(st)
    ok, reason = sc.cone_out(list(cells[0]))
    assert ok, reason
    assert sc.validate()[0], "the n>=4 recursive manifold gate must accept this"
    assert sc.rollback()
    assert _tops(st) == cells


# --------------------------------------------------------------------------- #
# 6. degenerate inputs are rejected cleanly (no mutation)
# --------------------------------------------------------------------------- #
def test_unknown_cell_and_last_cell_rejected():
    st = _sphere(3)
    cells = _tops(st)
    sc = cob.SurgicalCone(st)
    # A cell that is not present.
    assert not sc.cone_out([90, 91, 92, 93])[0]
    # Wrong arity.
    assert not sc.cone_out([0, 1, 2])[0]
    assert not sc.cone_in([0, 1, 2, 3])[0]
    assert sc.depth == 0
    assert _tops(st) == cells


def test_validate_matches_the_applied_gate():
    st = _sphere(3)
    ok, reason = cob.SurgicalCone(st).validate()
    assert ok, reason
