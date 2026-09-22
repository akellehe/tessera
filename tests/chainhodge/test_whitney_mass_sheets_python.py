# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The Whitney volume root declares its sheet, and continues from a geometry.

Whitepaper Section 1 (lines 150-156, 179-180). ``Branch::Continuation`` fixes the
one root per top simplex, sqrt(det g_T), by continuing along the straight segment
from the unit Euclidean reference simplex. That fixed a value and recorded
nothing: two instances with identical Gram determinants and roots of opposite
sign were indistinguishable in the certificate, and each instance's root was
chosen afresh from the same reference rather than continued from the geometry the
instance came from, so a family of instances was a sequence of independent
principal-value choices instead of one continued state.

Two things are added and asserted here. The certificate now carries the sheet
label of every volume root, as the signed number of turns det g_T makes about
zero along the continuation, so that the volume is that label's sign times the
principal root. And ``volumeContinuedFrom`` continues from a declared previous
geometry and sheet, so the label composes along a path: walked around a zero of
det g, it returns one higher and the volume negated.

The fixture for the loop is the same tetrahedron the mesh-level acceptance test
uses -- five unit squared lengths and a sixth on the circle of radius one about
s = 3 -- so the two levels are walking the same loop around the same branch
point.
"""

from __future__ import annotations

import cmath
import math

import numpy as np
import pytest

from tessera import chainhodge as ch
from tessera import cobordism as cob

WM = ch.WhitneyMass

_STEPS = 360


def _gram(s):
    """The Gram matrix of a tetrahedron with five unit squared lengths and a
    sixth equal to ``s``, whose determinant is (s/4)(3 - s)."""
    x = 1.0 - 0.5 * s
    return np.array([[1.0, 0.5, 0.5],
                     [0.5, 1.0, x],
                     [0.5, x, 1.0]], dtype=complex)


def _squared(phi):
    return 3.0 - cmath.exp(1j * phi)


def test_the_determinant_of_the_fixture_is_the_one_the_loop_assumes():
    """The loop below is only a loop about a branch point if det g vanishes at
    s = 3, so that is checked before anything is read off it."""
    for s in (1.0 + 0.0j, 2.0 + 0.0j, 0.5 + 1.5j):
        assert np.linalg.det(_gram(s)) == pytest.approx(s * (3.0 - s) / 4.0)
    assert abs(np.linalg.det(_gram(3.0 + 0.0j))) < 1e-15


def test_the_certificate_carries_a_sheet_label_per_volume():
    """Every volume in the certificate is its label's sign times the principal
    root, so the label and the value are one statement."""
    K = cob.ChainComplex.fromTopCells([[0, 1, 2, 3]])
    s = [1.0 + 0.0j] * 5 + [2.0 + 0.0j]
    cert = WM.certificate(K, s, ch.Branch.Continuation)
    assert len(cert.volumeWindings) == len(cert.volumes)
    for volume, determinant, winding in zip(cert.volumes, cert.gramDeterminants,
                                            cert.volumeWindings):
        expected = ((-1) ** (winding % 2)) * cmath.sqrt(determinant) / 6.0
        assert volume == pytest.approx(expected, abs=1e-14)


def test_the_euclidean_reference_declares_the_principal_sheet():
    """A geometry the reference segment reaches without turning is on sheet 0,
    and its volume is the principal root -- the unit regular tetrahedron's
    sqrt(2)/12."""
    g = _gram(1.0 + 0.0j)
    volume, ambiguous = WM.volumeOnBranch(g, ch.Branch.Continuation)
    assert not ambiguous
    assert WM.volumeWindingOnBranch(g, ch.Branch.Continuation) == 0
    assert volume == pytest.approx(math.sqrt(0.5) / 6.0)


def test_a_zero_step_continues_nothing():
    """Continuing from a geometry to itself leaves the declared sheet alone,
    whatever sheet that was."""
    g = _gram(2.0 + 0.0j)
    for winding in (0, 1, -2):
        volume, ambiguous, out = WM.volumeContinuedFrom(g, winding, g)
        assert not ambiguous
        assert out == winding
        assert volume == pytest.approx(
            ((-1) ** (winding % 2)) * cmath.sqrt(np.linalg.det(g)) / 6.0,
            abs=1e-14)


def test_a_loop_about_the_zero_of_the_determinant_returns_the_other_sheet():
    """The acceptance statement at the level of one Gram matrix: the loop comes
    back to the geometry it started at and to the other root, and the label
    composes along the path to say so."""
    winding = 0
    gram = _gram(_squared(0.0))
    volume = None
    for k in range(1, _STEPS + 1):
        nxt = _gram(_squared(2.0 * math.pi * k / _STEPS))
        volume, ambiguous, winding = WM.volumeContinuedFrom(gram, winding, nxt)
        assert not ambiguous, f"no continuation at step {k}"
        gram = nxt

    assert winding == 1
    principal, _ = WM.volumeOnBranch(gram, ch.Branch.Continuation)
    assert principal == pytest.approx(math.sqrt(0.5) / 6.0)
    assert volume == pytest.approx(-principal, abs=1e-12)


def test_two_loops_return_the_declared_sheet():
    winding = 0
    gram = _gram(_squared(0.0))
    volume = None
    for k in range(1, 2 * _STEPS + 1):
        nxt = _gram(_squared(4.0 * math.pi * k / (2 * _STEPS)))
        volume, _, winding = WM.volumeContinuedFrom(gram, winding, nxt)
        gram = nxt
    assert winding == 2
    principal, _ = WM.volumeOnBranch(gram, ch.Branch.Continuation)
    assert volume == pytest.approx(principal, abs=1e-12)


def test_a_loop_that_encloses_no_zero_moves_no_sheet():
    """The control: the circle of radius one half about s = 1 encloses neither
    zero of det g and returns the principal sheet."""
    winding = 0
    gram = _gram(1.0 + 0.5)
    for k in range(1, _STEPS + 1):
        nxt = _gram(1.0 + 0.5 * cmath.exp(2j * math.pi * k / _STEPS))
        _, _, winding = WM.volumeContinuedFrom(gram, winding, nxt)
        gram = nxt
    assert winding == 0


def test_the_continued_volume_is_continuous_where_the_principal_one_jumps():
    """The reason the label is carried at all. Along the loop the principal
    root of det g flips sign at the cut; the continued one does not."""
    winding = 0
    gram = _gram(_squared(0.0))
    continued = [WM.volumeContinuedFrom(gram, 0, gram)[0]]
    principal = [WM.volumeOnBranch(gram, ch.Branch.Continuation)[0]]
    for k in range(1, _STEPS + 1):
        nxt = _gram(_squared(2.0 * math.pi * k / _STEPS))
        volume, _, winding = WM.volumeContinuedFrom(gram, winding, nxt)
        continued.append(volume)
        principal.append(cmath.sqrt(np.linalg.det(nxt)) / 6.0)
        gram = nxt

    continued_step = max(abs(b - a) for a, b in zip(continued, continued[1:]))
    principal_step = max(abs(b - a) for a, b in zip(principal, principal[1:]))
    assert continued_step < 0.01
    assert principal_step > 0.2
    assert continued_step < 0.1 * principal_step
