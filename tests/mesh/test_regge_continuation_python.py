# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""A loop of complex squared lengths around a branch point, walked in the mesh.

This is the acceptance criterion of the Riemann-sheet work (whitepaper Section 1,
lines 150-156 and 179-180): a loop of complex squared lengths around a branch
point returns the root on the other sheet, and the Regge action along the loop is
continuous.

The fixture is one tetrahedron with five unit squared lengths and a sixth set to

    s(phi) = 3 - exp(i*phi)/2,   phi from 0 to 2*pi.

Its Gram determinant is then det G = (s/4)(3 - s), whose zeros are at s = 0 and
s = 3. The loop is the circle of radius one half about s = 3, which encloses that
zero and not the other, so

    det G = (3 - exp(i*phi)/2)/4 * exp(i*phi)/2

winds exactly once about the origin. The content of the cell is sqrt(det G)/3!,
so one turn of the loop must return it negated -- the other sheet -- while the
geometry returns to exactly where it began. That is the first clause.

The radius is one half and not one because the two faces carrying the varying
edge have squared area s(4 - s)/16, which vanishes at s = 4. A circle of radius
one about s = 3 runs exactly through that point, where a cofactor root sits on
its own branch point, the dihedral cosine diverges and there is no continuation
to take -- a degenerate path rather than a loop around a branch point. At radius
one half nothing but det G has a zero inside or on the circle.

The second clause is about the action. The action is the hinge contents times the
deficit angles, and the deficit angles are inverse cosines of ratios of square
roots of Cayley-Menger cofactors, several of which cross their cuts on this loop.
Read at principal values the action jumps at those crossings, at parameter values
where nothing at all happens to the geometry. Read on the declared sheets it does
not, and the test asserts both: the continued action's largest step and the
principal one's, measured on the same path.
"""

from __future__ import annotations

import cmath
import math

import pytest

try:
    import tessera
    _IMPORT_OK = True
except Exception:  # pragma: no cover
    _IMPORT_OK = False

pytestmark = pytest.mark.skipif(not _IMPORT_OK, reason="tessera not built")

#: Steps around the loop. Each turns the Gram determinant by about a fifth of a
#: degree, three orders below the half turn the continuation needs, which
#: ``maxRadicandTurn`` is asserted against rather than assumed.
_STEPS = 2000

#: The centre and radius of the loop in the varying squared length: the circle
#: about the zero of det G at s = 3, small enough to keep the zero of the face
#: areas at s = 4 off the path (see the module docstring).
_CENTRE = 3.0
_RADIUS = 0.5


def _spacetime(dim, topology):
    sig = tessera.Signature(dim, tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    return tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0,
                             tessera.PREFERRED, topology)


def _edge_map(st):
    out = {}
    for e in st.getEdgeList().toVector():
        a, b = e.getSource().getId(), e.getTarget().getId()
        out[(min(a, b), max(a, b))] = e
    return out


def _squared(phi):
    return _CENTRE - _RADIUS * cmath.exp(1j * phi)


def _tetrahedron():
    """One tetrahedron with its hinges materialized, five unit squared lengths
    and the sixth at s(0). Returns the spacetime, the edge map and the key of
    the varying edge."""
    st = _spacetime(3, tessera.SolidSimplex(3))
    st.build()
    # Constructing a solver materializes the canonical sub-simplices, so the
    # (d-2)-hinges the continuation labels exist in the simplex list.
    tessera.ReggeSolver(st, tessera.MatterConfiguration())
    edges = _edge_map(st)
    assert len(edges) == 6, f"expected a tetrahedron, got {len(edges)} edges"
    varying = sorted(edges)[-1]
    for key, edge in edges.items():
        edge.setLength(cmath.sqrt(_squared(0.0)) if key == varying else 1.0 + 0.0j)
        edge.setPhase(0.0)
    return st, edges, varying


def _cell_key(st):
    tops = [s for s in st.getSimplices() if len(s.getVertices()) == 4]
    assert len(tops) == 1
    return sorted(v.getId() for v in tops[0].getVertices())


def _walk(steps=_STEPS):
    """Walk the loop, returning the continuation, the two action traces and the
    largest radicand turn seen on any step."""
    st, edges, varying = _tetrahedron()
    continuation = tessera.ReggeContinuation(st)
    declared = [continuation.action()]
    principal = [continuation.principalAction()]
    largest_turn = 0.0
    for k in range(1, steps + 1):
        phi = 2.0 * math.pi * k / steps
        edges[varying].continueLength(cmath.sqrt(_squared(phi)))
        continuation.advance()
        largest_turn = max(largest_turn, continuation.maxRadicandTurn())
        declared.append(continuation.action())
        principal.append(continuation.principalAction())
    return st, continuation, declared, principal, largest_turn


def _largest_step(trace):
    return max(abs(b - a) for a, b in zip(trace, trace[1:]))


# --------------------------------------------------------------------------- #
# The declaration reproduces the sheet-blind geometry before any path is walked
# --------------------------------------------------------------------------- #
def test_a_fresh_continuation_reproduces_the_simplex_values():
    """Declaring costs nothing: at the geometry it was declared at, every label
    is principal and every value is the one Simplex computes."""
    st, _, _ = _tetrahedron()
    continuation = tessera.ReggeContinuation(st)
    cell = _cell_key(st)
    tops = [s for s in st.getSimplices() if len(s.getVertices()) == 4]
    hinges = {tuple(sorted(v.getId() for v in s.getVertices())): s
              for s in st.getSimplices() if len(s.getVertices()) == 2}

    assert continuation.cells() == [cell]
    assert continuation.volumeSheet(cell) == 0
    assert abs(continuation.volume(cell) - tops[0].volume()) < 1e-14

    assert len(continuation.hinges()) == 6
    for hinge in continuation.hinges():
        simplex = hinges[tuple(hinge)]
        assert abs(continuation.dihedralAngle(cell, hinge)
                   - tops[0].dihedralAngle(simplex)) < 1e-14
        assert abs(continuation.deficitAngle(hinge)
                   - simplex.deficitAngle()) < 1e-14
        assert continuation.angleBranchIndex(cell, hinge) == 0
        assert continuation.angleOrientation(cell, hinge) == 1
    assert abs(continuation.action() - continuation.principalAction()) < 1e-14


# --------------------------------------------------------------------------- #
# Clause one: the root comes back on the other sheet
# --------------------------------------------------------------------------- #
def test_the_loop_returns_the_cell_content_on_the_other_sheet():
    st, continuation, _, _, largest_turn = _walk()
    cell = _cell_key(st)

    # The path was resolved: no step turned a radicand by anything near the half
    # turn that would make the continuation ambiguous, and none landed on a
    # branch point, where there is no continuation to take.
    assert largest_turn < 0.5 * math.pi
    assert not continuation.touchedBranchPoint()

    # The geometry is exactly where it started: s = 5/2, det G = (5/8)(1/2) = 5/16.
    tops = [s for s in st.getSimplices() if len(s.getVertices()) == 4]
    principal_volume = tops[0].volume()
    assert abs(principal_volume - math.sqrt(5.0 / 16.0) / 6.0) < 1e-12

    # The root is not.
    assert continuation.volumeWinding(cell) == 1
    assert continuation.volumeSheet(cell) == 1
    assert abs(continuation.volume(cell) + principal_volume) < 1e-12


def test_two_loops_return_the_declared_sheet():
    """The monodromy of the content root is an element of order two: the second
    turn undoes the first, so the label is a sheet and not a counter of how far
    the path travelled."""
    st, edges, varying = _tetrahedron()
    continuation = tessera.ReggeContinuation(st)
    for k in range(1, 2 * _STEPS + 1):
        phi = 4.0 * math.pi * k / (2 * _STEPS)
        edges[varying].continueLength(cmath.sqrt(_squared(phi)))
        continuation.advance()
    cell = _cell_key(st)
    assert continuation.volumeWinding(cell) == 2
    assert continuation.volumeSheet(cell) == 0
    tops = [s for s in st.getSimplices() if len(s.getVertices()) == 4]
    assert abs(continuation.volume(cell) - tops[0].volume()) < 1e-12


def test_a_loop_that_encloses_no_zero_moves_no_sheet():
    """The control. The same walk on a circle about s = 1, which encloses
    neither zero of det G, returns every label to the principal sheet -- so the
    sheet change above is the monodromy of the enclosed branch point and not an
    artifact of having moved at all."""
    st, edges, varying = _tetrahedron()
    for key, edge in edges.items():
        edge.setLength(cmath.sqrt(1.0 + 0.5) if key == varying else 1.0 + 0.0j)
    continuation = tessera.ReggeContinuation(st)
    for k in range(1, _STEPS + 1):
        phi = 2.0 * math.pi * k / _STEPS
        edges[varying].continueLength(cmath.sqrt(1.0 + 0.5 * cmath.exp(1j * phi)))
        continuation.advance()
    cell = _cell_key(st)
    assert continuation.volumeWinding(cell) == 0
    assert continuation.volumeSheet(cell) == 0


# --------------------------------------------------------------------------- #
# Clause two: the action along the loop is continuous
# --------------------------------------------------------------------------- #
def test_the_continued_action_is_continuous_along_the_loop():
    _, continuation, declared, principal, _ = _walk()

    declared_step = _largest_step(declared)
    principal_step = _largest_step(principal)

    # The action is continuous on the declared sheets: its largest step over
    # 2000 is of the size the geometry moves in one of them -- about 3/2000,
    # the total variation over the loop -- and not of the size of the action
    # itself.
    assert declared_step < 0.02, f"continued action jumps by {declared_step}"
    # Read at principal values the same path jumps by of order ten, at
    # parameter values where nothing happens to the geometry.
    assert principal_step > 1.0, f"principal action only moves {principal_step}"
    assert declared_step < 0.01 * principal_step


def test_the_continued_action_leaves_no_step_behind():
    """Continuity is a statement about every step, not about the largest one:
    no step of the continued action may stand out against the typical step of
    the same walk."""
    _, _, declared, _, _ = _walk()
    steps = [abs(b - a) for a, b in zip(declared, declared[1:])]
    typical = sorted(steps)[len(steps) // 2]
    # The walk is nearly uniform, so the true ratio is under two; a single
    # crossing left on a principal branch would put it in the thousands.
    assert max(steps) < 20.0 * typical, (
        f"largest step {max(steps)} against a median of {typical}")


# --------------------------------------------------------------------------- #
# The labels and the step size the continuation reports
# --------------------------------------------------------------------------- #
def test_a_fresh_declaration_labels_every_root_principal():
    """At the geometry it was declared at, every hinge content and every
    cofactor root sits on its principal sheet, and no angle has moved."""
    st, _, _ = _tetrahedron()
    continuation = tessera.ReggeContinuation(st)
    cell = _cell_key(st)
    assert continuation.maxAngleStep() == 0.0
    for hinge in continuation.hinges():
        assert continuation.hingeContentSheet(hinge) == 0
        assert tuple(continuation.angleCofactorSheets(cell, hinge)) == (0, 0)


def test_the_largest_angle_step_is_the_largest_angle_movement():
    st, edges, varying = _tetrahedron()
    continuation = tessera.ReggeContinuation(st)
    cell = _cell_key(st)
    before = {tuple(h): continuation.dihedralAngle(cell, h)
              for h in continuation.hinges()}
    edges[varying].continueLength(cmath.sqrt(_squared(0.01)))
    continuation.advance()
    moved = max(abs(continuation.dihedralAngle(cell, list(h)) - angle)
                for h, angle in before.items())
    assert moved > 0.0
    assert continuation.maxAngleStep() == pytest.approx(moved, rel=1e-12)


def test_the_cofactors_give_the_cosine_of_the_dihedral_angle():
    """cos(theta) = -C_ij / (sqrt(C_ii) sqrt(C_jj)) on the principal roots,
    the single-valued data every branch of the angle is built from."""
    st, _, _ = _tetrahedron()
    top = [s for s in st.getSimplices() if len(s.getVertices()) == 4][0]
    hinges = [s for s in st.getSimplices() if len(s.getVertices()) == 2]
    assert len(hinges) == 6
    for hinge in hinges:
        ok, cij, cii, cjj = top.dihedralCofactors(hinge)
        assert ok
        expected = cmath.cos(top.dihedralAngle(hinge))
        assert abs(-cij / (cmath.sqrt(cii) * cmath.sqrt(cjj)) - expected) \
            < 1e-12


def test_a_sheeted_inverse_cosine_reports_its_cosine():
    angle = tessera.SheetedAcos(0.3 + 0.1j)
    assert angle.cosine() == 0.3 + 0.1j
    angle.advance(0.35 + 0.1j)
    assert angle.cosine() == 0.35 + 0.1j
    assert abs(angle.value() - cmath.acos(0.35 + 0.1j)) < 1e-14
