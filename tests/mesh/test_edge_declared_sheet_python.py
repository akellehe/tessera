# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""An edge carries which of the two roots of its squared length it is.

Whitepaper Section 1 (lines 150-156, 179-180). The map l -> l^2 is two-to-one,
so l^2 -> l is branched over l^2 = 0 and an edge transported through a family of
complex geometries has to carry which of +/-sqrt(l^2) it is. ``Edge`` holds that
as the signed number of turns w its squared length has made about the branch
point since the length was declared, and the causal predicates read the declared
argument arg(l^2) + 2*pi*w rather than re-deriving arg(l^2) from the stored
number.

Two things are asserted, and they pull in opposite directions.

Folding the declared argument back into (-pi, pi] cannot move an edge between
causal buckets: causal character is a property of l^2, and the two roots +/-l of
one l^2 share it. So every predicate answers on a never-continued edge exactly
as it did before any of this existed, and a full turn does not turn a spacelike
edge timelike.

What the declaration keeps is the datum the fold discards: a timelike edge
reached by rotating through the lower half plane has declared argument -pi and
one reached through the upper half plane has +pi, while the principal argument
reports +pi for both. That sign is the side of the cut -- the +/-i*epsilon
prescription -- which every squared-volume continuation downstream has to agree
with, and it is not recoverable from the stored length.
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

_STEPS = 360


def _edge():
    """A bare two-vertex edge, with no spacetime around it: the declaration is a
    property of the edge alone."""
    metric = tessera.Metric(True, tessera.Signature(2, tessera.Lorentzian))
    st = tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0, tessera.PREFERRED,
                           tessera.SolidSimplex(2))
    st.build()
    return st, st.get_edge_list().to_vector()[0]


def _continue_squared_along(edge, arguments, modulus=1.0):
    """Walk the edge's squared length around the arc ``modulus*exp(i*theta)`` for
    theta in ``arguments``, passing at each step the length the path continued to
    -- ``sqrt(modulus)*exp(i*theta/2)`` -- rather than a root taken fresh, which
    would jump sign at the cut and break the tie between the stored root and the
    sheet the winding names."""
    root = math.sqrt(modulus)
    for theta in arguments:
        edge.continue_length(root * cmath.exp(0.5j * theta))


def _arc(total, steps):
    """``steps`` equal steps of the arc from 0 to ``total``, omitting 0."""
    return [total * k / steps for k in range(1, steps + 1)]


# --------------------------------------------------------------------------- #
# The default is unchanged
# --------------------------------------------------------------------------- #
def test_a_fresh_edge_is_declared_on_the_principal_sheet():
    _, edge = _edge()
    edge.set_length(1.0 + 0.0j)
    assert edge.squared_winding() == 0
    assert edge.squared_sheet() == 0
    assert abs(edge.declared_squared_argument() - edge.squared_argument()) < 1e-15


def test_the_declaration_does_not_move_any_bucket():
    """Spacelike, timelike, lightlike and mixed all classify as before."""
    _, edge = _edge()
    for length, predicate in ((1.0 + 0.0j, "is_spacelike"),
                              (0.0 + 1.0j, "is_timelike"),
                              (1.0 + 1.0j, "is_null"),
                              (1.0 + 0.3j, "is_mixed")):
        edge.set_length(length)
        assert getattr(edge, predicate)(), f"{length} is not {predicate}"


def test_set_length_redeclares_the_sheet():
    """A jump to an unrelated length is not a continuation, so the declaration
    restarts: carrying a winding across it would assert a path that was never
    walked."""
    _, edge = _edge()
    edge.set_length(1.0 + 0.0j)
    _continue_squared_along(edge, _arc(2.0 * math.pi, _STEPS))
    assert edge.squared_winding() == 1
    edge.set_length(1.0 + 0.0j)
    assert edge.squared_winding() == 0


# --------------------------------------------------------------------------- #
# The monodromy
# --------------------------------------------------------------------------- #
def test_a_full_turn_of_the_squared_length_returns_the_other_root():
    """The acceptance statement at the level of one edge: a loop of l^2 about
    its branch point comes back to the same squared length and the opposite
    root, and the edge says so."""
    _, edge = _edge()
    edge.set_length(1.0 + 0.0j)
    _continue_squared_along(edge, _arc(2.0 * math.pi, _STEPS))
    assert edge.squared_winding() == 1
    assert edge.squared_sheet() == 1
    assert abs(edge.declared_squared_argument() - 2.0 * math.pi) < 1e-9
    # l^2 is back where it started, so the causal character is too.
    assert edge.is_spacelike()
    # And the edge is the other root: the length itself has turned to -1.
    assert abs(edge.get_length() - (-1.0 + 0.0j)) < 1e-9


def test_two_turns_return_the_declared_root():
    _, edge = _edge()
    edge.set_length(1.0 + 0.0j)
    _continue_squared_along(edge, _arc(4.0 * math.pi, 2 * _STEPS))
    assert edge.squared_winding() == 2
    assert edge.squared_sheet() == 0
    assert abs(edge.get_length() - (1.0 + 0.0j)) < 1e-9


def test_the_declaration_and_the_stored_root_are_one_statement():
    """The invariant: the stored l is always (-1)**w times the principal root of
    l^2, so the winding never names a root the edge is not."""
    _, edge = _edge()
    edge.set_length(1.0 + 0.0j)
    for theta in _arc(4.0 * math.pi, 4 * _STEPS):
        edge.continue_length(cmath.exp(0.5j * theta))
        squared = edge.get_length() ** 2
        expected = cmath.sqrt(squared) * (-1) ** edge.squared_sheet()
        assert abs(edge.get_length() - expected) < 1e-9, f"broken at {theta}"


def test_the_declaration_survives_the_cut_the_principal_argument_wraps_at():
    """Rotate a spacelike l^2 clockwise past the negative real axis, to an
    argument of -(pi + 1/5). The principal argument has wrapped and reports
    +(pi - 1/5), an argument the path never took; the declaration reports the
    one it did, and the winding says why the two differ."""
    overshoot = 0.2
    total = math.pi + overshoot
    _, edge = _edge()
    edge.set_length(1.0 + 0.0j)
    _continue_squared_along(edge, _arc(-total, 2 * _STEPS))
    assert edge.squared_winding() == -1
    assert abs(edge.squared_argument() - (math.pi - overshoot)) < 1e-9
    assert abs(edge.declared_squared_argument() + total) < 1e-9
    # Neither reading calls it definite: l^2 is genuinely complex here.
    assert edge.is_mixed()


def test_a_timelike_edge_declares_which_lip_of_the_cut_it_sits_on():
    """The two lips of the cut are the two imaginary roots. A timelike edge
    stored as l = +i declares +pi, one stored as l = -i declares -pi, and the
    principal argument of l^2 reports +pi for both -- it cannot tell them apart,
    because it has thrown away the root. Both are timelike, which is the point:
    the sheet is further information about the same causal edge, not a
    reclassification of it."""
    _, edge = _edge()
    edge.set_length(0.0 + 1.0j)          # l^2 = -1, reached from above
    assert edge.is_timelike()
    assert abs(edge.squared_argument() - math.pi) < 1e-12
    assert abs(edge.declared_squared_argument() - math.pi) < 1e-12
    assert edge.squared_sheet() == 0

    edge.set_length(0.0 - 1.0j)          # l^2 = -1, reached from below
    assert edge.is_timelike()
    assert abs(edge.squared_argument() - math.pi) < 1e-12
    assert abs(edge.declared_squared_argument() + math.pi) < 1e-12
    assert edge.squared_sheet() == 1


def test_a_declared_winding_needs_no_path():
    """One full turn declared without walking one: the length is untouched, the
    causal bucket is untouched, and the declared argument has advanced by 2*pi."""
    _, edge = _edge()
    edge.set_length(1.0 + 0.0j)
    edge.declare_squared_turns(1)
    assert edge.squared_winding() == 2
    assert edge.squared_sheet() == 0
    assert edge.is_spacelike()
    assert abs(edge.declared_squared_argument() - 4.0 * math.pi) < 1e-15
