# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The two sheet primitives carry a label and a monodromy, not just a value.

Whitepaper Section 1 (lines 150-156, 179-180): where a squared-volume
formulation cannot avoid a root, the chosen Riemann sheet and its monodromy are
carried as part of the state rather than reset to a principal branch. A sheet is
which of a multivalued function's values is meant; the monodromy is the
permutation of those values a closed loop of the argument around a branch point
induces.

``SheetedSqrt`` is the two-sheeted surface of the square root, branched over a
vanishing radicand: one turn of the radicand about the origin must return the
root with the opposite sign, two turns must return it unchanged, and a loop that
does not enclose the origin must return it unchanged however far it travels.

``SheetedAcos`` is the countably-sheeted surface of the inverse cosine, branched
at +/-1 and at infinity. Each of the two finite branch points has an involutive
monodromy -- around +1 the angle goes to minus itself, around -1 it goes to
2*pi minus itself -- so a loop taken twice returns to the principal sheet, and a
loop taken once does not.

Nothing here touches a mesh: these are the primitives every sheeted geometric
quantity is built out of, tested on their own before anything reads them.
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

#: Steps per loop. The continuation needs each step to turn its radicand by less
#: than pi; 360 steps turn a unit circle by one degree, three orders below that
#: bound, so the assertions below are not testing the sampling.
_STEPS = 360

#: Agreement asked of a closed-form value against a continued one. The
#: continuation is a sequence of exact sign choices, not an accumulation, so the
#: only error is the one std::sqrt and std::acos already carry.
_TOLERANCE = 1e-12


def _walk(root, radicands):
    """Advance ``root`` through ``radicands`` and return the values it took."""
    values = []
    for z in radicands:
        root.advance(z)
        values.append(root.value())
    return values


def _circle(centre, radius, turns=1, steps=_STEPS, phase=0.0):
    """The points of ``turns`` counter-clockwise loops about ``centre``,
    starting and ending at ``centre + radius * exp(i * phase)`` and omitting
    the start.

    The phase matters. A loop is only a test of monodromy if its closing point
    is off every cut of the function being continued: on a cut the principal
    value is discontinuous, so the label at the closing point would hang on the
    sign of a rounding-level imaginary part. For the square root, whose cut is
    the negative real axis, a loop about the origin may start at +1; for the
    inverse cosine, cut on the real axis beyond +/-1, a loop about either branch
    point must start above or below the axis, so the crossing falls mid-loop
    and the closing point is off it."""
    n = steps * turns
    return [centre + radius * cmath.exp(1j * (phase + 2.0 * math.pi * turns * k / n))
            for k in range(1, n + 1)]


#: Where the inverse-cosine loops start: the top of the circle, off the cut.
_TOP = 0.5 * math.pi


# --------------------------------------------------------------------------- #
# SheetedSqrt
# --------------------------------------------------------------------------- #
def test_declared_root_starts_principal():
    """A freshly declared root is the principal one, on sheet 0."""
    for z in (3.0 + 0.0j, -2.0 + 0.0j, 0.5 - 1.25j, 1e-8 + 1e-8j):
        root = tessera.SheetedSqrt(z)
        assert root.sheet() == 0
        assert root.winding() == 0
        assert root.isPrincipal()
        assert abs(root.value() - cmath.sqrt(z)) <= _TOLERANCE
        assert abs(root.declaredArgument() - cmath.phase(z)) <= _TOLERANCE


def test_one_turn_about_the_branch_point_returns_the_other_sheet():
    """The monodromy of the square root: one loop flips the sign."""
    root = tessera.SheetedSqrt(1.0 + 0.0j)
    _walk(root, _circle(0.0 + 0.0j, 1.0))
    assert root.winding() == 1
    assert root.sheet() == 1
    assert not root.isPrincipal()
    # The radicand is exactly where it started and the root is not.
    assert abs(root.radicand() - 1.0) <= 1e-12
    assert abs(root.value() - (-1.0)) <= _TOLERANCE
    assert abs(root.declaredArgument() - 2.0 * math.pi) <= 1e-12


def test_two_turns_return_the_principal_sheet():
    """The monodromy group of the square root is Z/2, not Z."""
    root = tessera.SheetedSqrt(1.0 + 0.0j)
    _walk(root, _circle(0.0 + 0.0j, 1.0, turns=2))
    assert root.winding() == 2
    assert root.sheet() == 0
    assert abs(root.value() - 1.0) <= _TOLERANCE


def test_the_reverse_loop_carries_the_opposite_monodromy():
    """A clockwise loop winds -1 and lands on the same sheet as a
    counter-clockwise one: the sheet is the parity, the winding the signed
    count, and both are carried because only the second composes."""
    root = tessera.SheetedSqrt(1.0 + 0.0j)
    _walk(root, [cmath.exp(-2j * math.pi * k / _STEPS)
                 for k in range(1, _STEPS + 1)])
    assert root.winding() == -1
    assert root.sheet() == 1
    assert abs(root.value() - (-1.0)) <= _TOLERANCE


def test_a_loop_that_encloses_nothing_changes_no_sheet():
    """Distance travelled is not monodromy: a large loop that does not enclose
    the branch point returns the principal root."""
    root = tessera.SheetedSqrt(6.0 + 0.0j)      # the circle's own start point
    _walk(root, _circle(5.0 + 0.0j, 1.0))
    assert root.winding() == 0
    assert root.sheet() == 0
    assert abs(root.value() - cmath.sqrt(6.0)) <= _TOLERANCE


def test_the_continued_root_is_continuous_across_the_principal_cut():
    """The point of the label. Crossing the negative real axis moves the
    principal root by its whole length; the declared one moves by a step."""
    path = _circle(0.0 + 0.0j, 1.0)
    root = tessera.SheetedSqrt(1.0 + 0.0j)
    declared = [root.value()] + _walk(root, path)
    principal = [cmath.sqrt(1.0 + 0.0j)] + [cmath.sqrt(z) for z in path]

    declared_jump = max(abs(b - a) for a, b in zip(declared, declared[1:]))
    principal_jump = max(abs(b - a) for a, b in zip(principal, principal[1:]))
    # One step of a 360-step unit circle moves the root by about pi/360.
    assert declared_jump < 0.02
    # The principal root jumps by the full diameter of the unit circle at the
    # cut, two orders above that.
    assert principal_jump > 1.9


def test_the_step_size_is_reported():
    """A caller checks its sampling rather than assuming it: each step of a
    360-step circle turns the radicand by one degree."""
    root = tessera.SheetedSqrt(1.0 + 0.0j)
    for z in _circle(0.0 + 0.0j, 1.0):
        root.advance(z)
        assert abs(root.lastStep() - 2.0 * math.pi / _STEPS) < 1e-9


def test_the_branch_point_itself_is_reported_not_invented():
    """A path through the radicand's zero determines no continuation at all.
    The label is held and the fact is recorded, rather than a sheet being made
    up for it."""
    root = tessera.SheetedSqrt(1.0 + 0.0j)
    assert not root.touchedBranchPoint()
    root.advance(0.0 + 0.0j)
    assert root.touchedBranchPoint()
    assert root.value() == 0.0
    assert root.winding() == 0


def test_a_declared_winding_needs_no_path():
    """A caller that knows the sheet from the problem declares it directly."""
    root = tessera.SheetedSqrt(1.0 + 0.0j, 1)
    assert root.sheet() == 1
    assert abs(root.value() - (-1.0)) <= _TOLERANCE
    root.advance(1.0 + 0.0j)
    assert root.winding() == 1


# --------------------------------------------------------------------------- #
# SheetedAcos
# --------------------------------------------------------------------------- #
def test_declared_angle_starts_principal():
    for r in (0.25 + 0.0j, -0.5 + 0.0j, 2.0 + 0.0j, 0.3 - 1.5j):
        angle = tessera.SheetedAcos(r)
        assert angle.branchIndex() == 0
        assert angle.orientation() == 1
        assert angle.isPrincipal()
        assert abs(angle.value() - cmath.acos(r)) <= _TOLERANCE


def test_a_loop_about_plus_one_reflects_the_angle():
    """The monodromy at cos(theta) = 1: theta goes to -theta, the sheet label
    (k, eps) to (0, -1)."""
    start = 1.0 + 0.5j                       # top of the circle, off the cut
    angle = tessera.SheetedAcos(start)
    for r in _circle(1.0 + 0.0j, 0.5, phase=_TOP):
        angle.advance(r)
    assert angle.orientation() == -1
    assert angle.branchIndex() == 0
    assert abs(angle.value() + cmath.acos(start)) <= 1e-10


def test_a_loop_about_minus_one_reflects_the_angle_about_pi():
    """The monodromy at cos(theta) = -1: theta goes to 2*pi - theta, the sheet
    label to (1, -1). The two finite branch points act differently, which is
    why the label is a pair and not a sign."""
    start = -1.0 + 0.5j                      # top of the circle, off the cut
    angle = tessera.SheetedAcos(start)
    for r in _circle(-1.0 + 0.0j, 0.5, phase=_TOP):
        angle.advance(r)
    assert angle.orientation() == -1
    assert angle.branchIndex() == 1
    assert abs(angle.value() - (2.0 * math.pi - cmath.acos(start))) <= 1e-10


def test_each_finite_branch_point_has_an_involutive_monodromy():
    """Both loops taken twice return the principal sheet."""
    for centre in (1.0 + 0.0j, -1.0 + 0.0j):
        start = centre + 0.5j
        angle = tessera.SheetedAcos(start)
        for r in _circle(centre, 0.5, turns=2, phase=_TOP):
            angle.advance(r)
        assert angle.isPrincipal(), f"loop about {centre} is not involutive"
        assert abs(angle.value() - cmath.acos(start)) <= 1e-10


def test_the_continued_angle_is_continuous_where_the_principal_one_jumps():
    path = _circle(1.0 + 0.0j, 0.5, phase=_TOP)
    angle = tessera.SheetedAcos(1.0 + 0.5j)
    declared = [angle.value()]
    for r in path:
        angle.advance(r)
        declared.append(angle.value())
    principal = [cmath.acos(1.0 + 0.5j)] + [cmath.acos(r) for r in path]

    declared_jump = max(abs(b - a) for a, b in zip(declared, declared[1:]))
    principal_jump = max(abs(b - a) for a, b in zip(principal, principal[1:]))
    assert declared_jump < 0.05
    # Mid-loop the path crosses the cut at r = 3/2, where the principal value
    # reflects from -i arccosh(3/2) to +i arccosh(3/2): a jump of about 1.9.
    assert principal_jump > 1.0
    assert declared_jump < 0.05 * principal_jump
