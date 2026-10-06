# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The degree-1 period gradients follow a dilation of the squared lengths
(#1427).

Terms used below:

* a *dilation* by s > 0 multiplies every squared length by s;
* the fixture is the holed icosahedron of ``_holed_surface`` (b_1 = 2, three
  hole circles) on the diagonal weights with the squared-content weight
  convention (W_1 = l^2, W_2 = det G / 4 for a triangle with 2 by 2 Gram
  matrix G), and a target that is the first carried period row with 0.5 added
  to its first entry, so that r_U and r_psi are positive.

L_1 = W_1^{-1} K_1 + K_2 W_1 goes as 1/s, so its kernel and the carried
periods do not move, r_U goes as s^{-2} and its gradient d r_U / d l^2 as
s^{-3}, and r_psi goes as s^0 and its gradient as s^{-1}.

The low-rank derivative of L_1 has a term per triangle, which inverts G and
divides by W_2. `EigenstateSynthesis` skips it exactly when det G or W_2 is
zero, and checks that W_2 is det G / 4 relative to the Gram matrix's own scale
(|G_00| |G_11| + |G_01| |G_10|) / 4. On the module of db73d0e1 it skipped every
triangle with |det G| < 1e-12, that is below s of about 1e-6 here: the r_U
gradient times s^3 departed from its value at s = 1 by 0.448 at s = 1e-6 and
1e-7. The r_psi gradient reads the same derivative only on the harmonic
vectors u, where on this fixture (real positive weights) each triangle term
vanishes (d_2^T W_1 u = 0), so it agreed there to 6.5e-15 on that module as
well.

The harmonic split itself compares each eigenvalue of L_1, which goes as 1/s,
with the absolute ``kNullTol = 1e-7``, so below s of about 1e-8 and above
about 1e8 the split, and both gradients with it, depart; that case is kept as
a strict expected failure.

Measured on the module of
``~/scratch/v18code-2026-10-05/scale-free-thresholds/build-3``: from s = 1e-7 to
1e7 the largest relative departure is 1.15e-14 for the r_U gradient and
1.14e-14 for the r_psi gradient (``AGREEMENT`` is 1e-13); at s = 1e-10 and
1e10 both depart by 1.0 or more.
"""

import cmath
import os
import sys

import numpy as np
import pytest

import tessera

sys.path.insert(0, os.path.dirname(__file__))
from _holed_surface import holed_surface  # noqa: E402

cob = tessera.cobordism
HODGE = cob.HodgeLaplacian
DIAGONAL = cob.HodgeMetricSource.DiagonalWeights

#: Every decade over which the harmonic split of L_1 holds on this fixture.
SCALES = [10.0 ** k for k in range(-7, 8)]

#: Outside it, where the absolute eigenvalue cut decides the split.
SPLIT_SCALES = [1e-10, 1e10]

#: The relative agreement asserted between a gradient times its power of s and
#: its value at s = 1.
AGREEMENT = 1e-13


@pytest.fixture(autouse=True)
def _squared_content():
    """The squared-content weight convention, restored afterwards."""
    previous = HODGE.defaultWeightConvention()
    HODGE.setDefaultWeightConvention(cob.HodgeWeightConvention.SquaredContent)
    yield
    HODGE.setDefaultWeightConvention(previous)


def _gradients(scale):
    """The r_U gradient times s^3 and the r_psi gradient times s at ``scale``."""
    spacetime, synthesis, holes, periods = holed_surface(
        degree=1, metric_source=DIAGONAL)
    for edge in spacetime.getEdgeList().toVector():
        edge.setLength(cmath.sqrt(complex(scale * (edge.getLength() ** 2).real)))
    spacetime.materializeFacets()
    target = [complex(z) for z in periods[0]]
    target[0] += 0.5
    residual = np.asarray(
        synthesis.residualForPeriodsGradient(holes, target), float)
    gap = np.asarray(
        synthesis.periodGapForPeriodsGradient(holes, target), complex)
    return residual * scale ** 3, gap * scale


_REFERENCE = {}


def _reference():
    if not _REFERENCE:
        _REFERENCE["value"] = _gradients(1.0)
    return _REFERENCE["value"]


def _departures(scale):
    residual, gap = _gradients(scale)
    residual_1, gap_1 = _reference()
    return (float(np.linalg.norm(residual - residual_1) /
                  np.linalg.norm(residual_1)),
            float(np.linalg.norm(gap - gap_1) / np.linalg.norm(gap_1)))


def test_the_reference_gradients_are_not_zero():
    residual, gap = _reference()
    assert np.linalg.norm(residual) > 1.0
    assert np.linalg.norm(gap) > 0.1


@pytest.mark.parametrize("scale", SCALES, ids=lambda s: f"s={s:.0e}")
def test_the_period_gradients_follow_the_dilation(scale):
    """The r_U gradient times s^3 and the r_psi gradient times s agree with
    their values at s = 1 to ``AGREEMENT``."""
    residual, gap = _departures(scale)
    assert residual <= AGREEMENT, f"r_U gradient departs by {residual:.3e}"
    assert gap <= AGREEMENT, f"r_psi gradient departs by {gap:.3e}"


@pytest.mark.xfail(strict=True, reason=(
    "EigenstateSynthesis::periodGradientOverLoops and "
    "periodGapForLoopsGradient split L_1 into harmonic and non-harmonic "
    "modes by |lambda| < kNullTol = 1e-7, an absolute cut on an eigenvalue "
    "that goes as 1/s, so the split changes with the unit of length."))
@pytest.mark.parametrize("scale", SPLIT_SCALES, ids=lambda s: f"s={s:.0e}")
def test_the_harmonic_split_follows_the_dilation(scale):
    residual, gap = _departures(scale)
    assert residual <= AGREEMENT and gap <= AGREEMENT
