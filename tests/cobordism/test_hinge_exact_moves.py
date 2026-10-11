"""Round-trip exactness of the Pachner move suite + stellar cone primitives (T1).

The Emergent Color Topology epic (#457) grows the proton's ``b2`` color register by
*gated surgical coning under relaxation*. Its load-bearing prerequisite (#458, this
module) is that the **existing** CDT Pachner moves are **hinge-exact** and **exactly
invertible**: ``move o move^-1`` must leave both the complex AND the complex
(Lorentzian / Sorkin) ``dual_regge_action`` -- real AND imaginary parts -- invariant to
machine precision, so a later greedy ``Delta F`` optimiser can trust its deltas
instead of chasing orphan-hinge / lossy-rollback artifacts (#365 / #371).

The defect being guarded: ``remove_simplex`` of a top cell leaves its lazily
materialised ``(d-1)`` / ``(d-2)`` sub-faces registered. ``collectHinges`` then summed
those *orphans* into the action with a bare ``2*pi`` deficit while the gradient ignored
them. The fix is bookkeeping (a hinge with no top coface is not part of the
triangulation -- ``Simplex.has_top_coface``), not a change to ``S = sum_h |*h|*eps_h``.

Coverage:

* every CDT move type (add / remove / flip / iflip / shift) -- action invariant (Re+Im)
  and the top-cell set restored across apply o rollback;
* the stellar ``1<->(d+1)`` refinement cone (in / out) on triangulated spheres,
  including ``S^3`` (the proton's spatial slice) and ``S^4`` (the epic host);
* orientation / ``Im S`` sign stability under coning;
* stacked topology changes inverted level-by-level -- the action retraces exactly;
* multiple topologies and sizes;
* orphans excluded from the action, and ``prune_orphaned_simplices`` restoring the raw
  registered simplex set bit-for-bit.
"""

import math

import pytest

import tessera as T
import cmath

# Action is restored by re-creating cells at identical edge lengths, so the round
# trip is bit-exact up to summation reassociation: a few ULPs on small complexes,
# scaling mildly with the hinge count on the larger CDT builds.
TOL = 1e-9
TOL_BIG = 1e-6


# --------------------------------------------------------------------------- #
# Builders / measurement helpers
# --------------------------------------------------------------------------- #
def _matter():
    return T.MatterConfiguration()


def _make_cdt(n_simplices=120):
    """A built 4D Lorentzian CDT toroid -- genuine complex action (Im S != 0)."""
    sig = T.Signature(4, T.Lorentzian)
    metric = T.Metric(True, sig)
    st = T.Spacetime(metric, T.CDT, 1.0, 1.0, T.PREFERRED, T.Toroid())
    st.build(n_simplices)
    return st


def _sphere(d, sq=1.0):
    """Boundary of a (d+1)-simplex = a minimal triangulated S^d, unit spacelike."""
    sig = T.Signature(d, T.Lorentzian)
    metric = T.Metric(True, sig)
    st = T.Spacetime(metric, T.CDT, 1.0, 1.0, T.PREFERRED, T.SimplexBoundarySphere(d))
    st.build()
    for e in st.get_edge_list().to_vector():
        e.set_length(cmath.sqrt(complex(sq)))
    return st


def _s2_cross_s1():
    sig = T.Signature(3, T.Lorentzian)
    metric = T.Metric(True, sig)
    st = T.Spacetime(metric, T.CDT, 1.0, 1.0, T.PREFERRED, T.SphereCircleProduct())
    st.build()
    for e in st.get_edge_list().to_vector():
        e.set_length(cmath.sqrt(complex(1.0)))
    return st


def _dim(st):
    # d = (top-cell vertex count) - 1, read straight off the triangulation.
    return max(len([v for v in s.get_vertices()]) for s in st.get_top_simplices()) - 1


def _act(st):
    return T.ReggeSolver(st, _matter()).dual_regge_action()


def _top_set(st):
    return sorted(
        tuple(sorted(v.get_id() for v in s.get_vertices()))
        for s in st.get_top_simplices()
    )


def _all_set(st):
    return sorted(
        tuple(sorted(v.get_id() for v in s.get_vertices()))
        for s in st.get_simplices()
    )


def _genuine_hinge_set(st):
    hsize = _dim(st)  # (d-2)-simplex has d-1 vertices
    return sorted(
        tuple(sorted(v.get_id() for v in s.get_vertices()))
        for s in st.get_simplices()
        if len([v for v in s.get_vertices()]) == hsize and s.has_top_coface()
    )


def _assert_close(a, b, tol, what):
    assert abs(a.real - b.real) < tol, f"{what}: Re drift {abs(a.real - b.real):.2e}"
    assert abs(a.imag - b.imag) < tol, f"{what}: Im drift {abs(a.imag - b.imag):.2e}"


# --------------------------------------------------------------------------- #
# 1. Every CDT move type: apply o rollback leaves the action + complex invariant
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "ctor,args",
    [
        (T.AddMove, (False,)),  # relabel=False: stable vertex ids (#365 precondition)
        (T.FlipMove, ()),
        (T.IFlipMove, ()),
        (T.ShiftMove, ()),
    ],
)
def test_cdt_move_roundtrip_action_invariant(ctor, args):
    st = _make_cdt(120)
    a0 = _act(st)
    tops0, hinges0 = _top_set(st), _genuine_hinge_set(st)

    for seed in range(64):
        mv = ctor(st, seed, *args)
        if not mv.propose():
            continue
        assert mv.apply()
        a1 = _act(st)  # materialise the new region's hinges (would orphan on rollback)
        mv.rollback()
        assert not mv.is_applied()
        a2 = _act(st)

        _assert_close(a2, a0, TOL_BIG, f"{mv.move_type()} roundtrip")
        assert _top_set(st) == tops0, "top-cell set not restored"
        assert _genuine_hinge_set(st) == hinges0, "genuine hinge set not restored"
        # The move genuinely perturbed the action (the test is not vacuous).
        assert abs(a1 - a0) > TOL, f"{mv.move_type()} did not change the action"
        return
    pytest.skip(f"{ctor.__name__} never proposed in 64 seeds")


def test_remove_move_roundtrip_action_invariant():
    """RemoveMove ((2d)->2 weld). A built CDT has no order-2d vertex, so first add one
    (the inverse config), then verify the remove o rollback round trip about it."""
    st = _make_cdt(60)
    add = T.AddMove(st, 5, False)
    assert add.propose() and add.apply()

    a0 = _act(st)
    tops0, hinges0 = _top_set(st), _genuine_hinge_set(st)

    for seed in range(400):
        rm = T.RemoveMove(st, seed)
        if not rm.propose():
            continue
        assert rm.apply()
        a1 = _act(st)
        rm.rollback()
        a2 = _act(st)
        _assert_close(a2, a0, TOL_BIG, "remove roundtrip")
        assert _top_set(st) == tops0
        assert _genuine_hinge_set(st) == hinges0
        assert abs(a1 - a0) > TOL
        return
    pytest.skip("RemoveMove never proposed in 400 seeds")


# --------------------------------------------------------------------------- #
# 2. Stellar cone-in / cone-out refinement (the 1<->(d+1) primitive)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("builder", [lambda: _sphere(3), lambda: _sphere(4),
                                      _s2_cross_s1])
def test_cone_in_out_roundtrip(builder):
    """Cone-in (pre-geometric stellar AddMove) then cone-out (its rollback) restores
    the complex and the action exactly. Cone-in is a refinement -- it genuinely
    redistributes the curvature, so a1 != a0 -- but cone-out is its exact inverse."""
    st = builder()
    a0 = _act(st)
    tops0, all0 = _top_set(st), _all_set(st)
    d = _dim(st)

    cone = T.AddMove(st, 1, False, T.PachnerMode.PreGeometric, False)
    assert cone.propose(), "cone-in did not propose"
    assert cone.apply()
    # one cell -> (d+1) cells
    assert st.get_vertex_count() == len(set(v for tup in tops0 for v in tup)) + 1
    a1 = _act(st)
    assert abs(a1 - a0) > TOL, "refinement should redistribute curvature"

    cone.rollback()
    a2 = _act(st)
    _assert_close(a2, a0, TOL, f"S^{d} cone roundtrip")
    assert _top_set(st) == tops0, "cone-out did not restore the top cells"

    # Orphans (the materialised child hinges) are excluded from the action; pruning
    # them restores the raw registered simplex set bit-for-bit.
    assert st.prune_orphaned_simplices() >= 0
    assert _all_set(st) == all0, "registered simplex set not bit-identical after prune"


def test_standalone_cone_out_then_in():
    """The cone-out move (pre-geometric RemoveMove) as a first-class primitive: weld
    an apex away, then its rollback (cone-in) re-raises it, action invariant."""
    st = _sphere(3)
    # Raise an apex so a (d+1)->1 weld target exists.
    raise_ = T.AddMove(st, 1, False, T.PachnerMode.PreGeometric, False)
    assert raise_.propose() and raise_.apply()
    a0 = _act(st)
    tops0 = _top_set(st)

    for seed in range(200):
        out = T.RemoveMove(st, seed, T.PachnerMode.PreGeometric, False)
        if not out.propose():
            continue
        assert out.apply()
        a1 = _act(st)
        out.rollback()
        a2 = _act(st)
        _assert_close(a2, a0, TOL, "cone-out roundtrip")
        assert _top_set(st) == tops0
        assert abs(a1 - a0) > TOL
        return
    pytest.skip("cone-out never proposed in 200 seeds")


# --------------------------------------------------------------------------- #
# 3. Orientation / Im S sign stability under coning (Lorentzian)
# --------------------------------------------------------------------------- #
def test_imaginary_part_preserved_under_coning():
    """A surgical cone can flip a local induced orientation -> a spurious sign in the
    causal (Im S) deficit. On a genuinely complex action (CDT toroid, Im S ~ -11 with
    the #581 mixed-hinge branch; ~ -35 before it) the cone-in/out round trip must
    restore Im S, not just |S|."""
    st = _make_cdt(120)
    a0 = _act(st)
    assert abs(a0.imag) > 1.0, "fixture is not genuinely Lorentzian"

    cone = T.AddMove(st, 2, False, T.PachnerMode.PreGeometric, False)
    assert cone.propose() and cone.apply()
    cone.rollback()
    a2 = _act(st)
    assert abs(a2.imag - a0.imag) < TOL_BIG, f"Im S sign drift {a2.imag - a0.imag:.2e}"
    assert abs(a2.real - a0.real) < TOL_BIG


def test_flip_preserves_imaginary_part():
    st = _make_cdt(120)
    a0 = _act(st)
    assert abs(a0.imag) > 1.0
    for seed in range(64):
        fl = T.FlipMove(st, seed)
        if not fl.propose():
            continue
        assert fl.apply()
        fl.rollback()
        a2 = _act(st)
        _assert_close(a2, a0, TOL_BIG, "flip Im roundtrip")
        return
    pytest.skip("FlipMove never proposed")


# --------------------------------------------------------------------------- #
# 4. Stacked topology changes, inverted level by level
# --------------------------------------------------------------------------- #
def test_stacked_cones_inverted_level_by_level():
    """Stack several stellar refinements, recording the action after each; then undo
    in LIFO order, recording the action after each undo. The reverse path must retrace
    the forward action at *every* level of the stack (the ticket's 'same dualReggeAction
    one way as the other at every level')."""
    st = _sphere(4)
    forward = [_act(st)]
    cones = []
    for k in range(5):
        c = T.AddMove(st, 100 + k, False, T.PachnerMode.PreGeometric, False)
        assert c.propose() and c.apply(), f"cone {k} failed"
        cones.append(c)
        forward.append(_act(st))

    # Undo LIFO; after undoing cone k the state must match forward[k].
    for k in reversed(range(len(cones))):
        cones[k].rollback()
        _assert_close(_act(st), forward[k], TOL, f"stack level {k}")


def test_stacked_mixed_cdt_moves_roundtrip():
    """A mixed stack of accepted CDT moves, then full LIFO rollback, restores the
    action and the top-cell set exactly."""
    st = _make_cdt(150)
    a0 = _act(st)
    tops0 = _top_set(st)

    applied = []
    ctors = [(T.AddMove, (False,)), (T.FlipMove, ()), (T.ShiftMove, ()),
             (T.AddMove, (False,)), (T.IFlipMove, ())]
    for i, (ctor, args) in enumerate(ctors):
        for seed in range(64):
            mv = ctor(st, 1000 * i + seed, *args)
            if mv.propose() and mv.apply():
                applied.append(mv)
                break
    assert len(applied) >= 3, "stack did not build up"

    a_mid = _act(st)
    assert abs(a_mid - a0) > TOL, "stack did not change the action"

    for mv in reversed(applied):
        mv.rollback()
    _assert_close(_act(st), a0, TOL_BIG, "mixed stack roundtrip")
    assert _top_set(st) == tops0


# --------------------------------------------------------------------------- #
# 5. Multiple topologies / sizes
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "builder",
    [
        lambda: _sphere(3),
        lambda: _sphere(4),
        _s2_cross_s1,
        lambda: _make_cdt(40),
        lambda: _make_cdt(120),
        lambda: _make_cdt(250),
    ],
)
def test_cone_roundtrip_across_topologies(builder):
    st = builder()
    a0 = _act(st)
    tops0 = _top_set(st)
    cone = T.AddMove(st, 7, False, T.PachnerMode.PreGeometric, False)
    if not cone.propose():
        pytest.skip("no top cell to subdivide")
    assert cone.apply()
    cone.rollback()
    _assert_close(_act(st), a0, TOL_BIG, "cross-topology cone roundtrip")
    assert _top_set(st) == tops0


# --------------------------------------------------------------------------- #
# 6. Orphan hinges are excluded from the action; pruning is bit-exact
# --------------------------------------------------------------------------- #
def test_orphans_excluded_from_action():
    """After cone-in + measure + cone-out the removed child's hinges linger as orphans.
    dualReggeAction must equal a freshly built identical sphere's action (orphans carry
    a bare 2*pi deficit but no top coface, so they are not part of the triangulation)."""
    st = _sphere(3)
    ref = _act(_sphere(3))  # independent, never-mutated reference

    cone = T.AddMove(st, 1, False, T.PachnerMode.PreGeometric, False)
    assert cone.propose() and cone.apply()
    _act(st)  # materialise child hinges -> these become orphans on rollback
    cone.rollback()

    hinge_size = _dim(st)
    registered = [s for s in st.get_simplices()
                  if len([v for v in s.get_vertices()]) == hinge_size]
    orphans = [s for s in registered if not s.has_top_coface()]
    assert len(orphans) > 0, "expected lingering orphan hinges to make the test bite"

    _assert_close(_act(st), ref, TOL, "orphans excluded from action")


def test_prune_orphans_restores_raw_simplex_set():
    st = _make_cdt(120)
    _act(st)  # materialise
    all0 = _all_set(st)

    mv = T.AddMove(st, 0, False)
    assert mv.propose() and mv.apply()
    _act(st)  # materialise child sub-faces
    mv.rollback()

    assert _all_set(st) != all0, "expected orphan registrations before pruning"
    pruned = st.prune_orphaned_simplices()
    assert pruned > 0
    assert _all_set(st) == all0, "prune did not restore the raw simplex set"


# --------------------------------------------------------------------------- #
# 7. Explicit S^3 spatial-slice coverage (the proton lives here)
# --------------------------------------------------------------------------- #
def test_s3_sphere_cone_and_move_roundtrip():
    st = _sphere(3)
    assert st.get_vertex_count() == 5  # boundary of a 4-simplex
    assert len([s for s in st.get_top_simplices()]) == 5
    a0 = _act(st)
    tops0 = _top_set(st)

    cone = T.AddMove(st, 1, False, T.PachnerMode.PreGeometric, False)
    assert cone.propose() and cone.apply()
    assert len([s for s in st.get_top_simplices()]) == 8  # 5 - 1 + 4
    cone.rollback()
    _assert_close(_act(st), a0, TOL, "S^3 cone roundtrip")
    assert _top_set(st) == tops0
    assert not math.isnan(a0.real)
