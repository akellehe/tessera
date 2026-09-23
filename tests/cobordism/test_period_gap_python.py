# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""The hard period-pin r_psi (#377): ``EigenstateSynthesis.periodGapForPeriods``
and its exact analytic gradient ``periodGapForPeriodsGradient``.

r_psi is the realizability term r_U's selectable alternative for the merge's
state objective. Where r_U (``residualForPeriods``) holds the target periods
EXACTLY (a minimal leak) and scores the resulting state's non-harmonicity, r_psi
keeps the carried object a PURE harmonic and scores the period GAP it cannot
match, ``r_psi = ||P^T c - target||^2`` with ``c`` the least-squares fit. Both
vanish iff the target lies in the carried period span (the same realizable set);
they differ off-zero.

These tests pin r_psi's contract on the level-0 merge substrate (n1=174):
  * r_psi ~ 0 at a realizable target (a carried period row), floored when the
    geometry is perturbed so that target is no longer carried,
  * the analytic gradient matches a central finite difference of the value
    (the new envelope-theorem gradient, the key correctness check),
  * r_psi shares r_U's realizable zero set.
"""

import importlib.util
import os
import sys
import unittest

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import tessera  # noqa: E402
from _holed_surface import holed_surface  # noqa: E402
import cmath

cob = tessera.cobordism


DIAGONAL = cob.HodgeMetricSource.DiagonalWeights


def _substrate(source=DIAGONAL):
    """A holed icosahedron (b1 register): st, es, hole-circles, period matrix P,
    and the edge-cell list.

    The contract pinned here is the diagonal-weight path's, named by its
    source. On the default Whitney pencil the near-kernel representatives are
    the geometric images z = G_1 h of the harmonic chains, which are closed
    cochains (d_2^T z = 0): their periods over the hole cycles span the image
    of H^1, a topological subspace, so a target in that span stays realizable
    under every perturbation of the geometry and r_psi cannot see the
    lengths at all (see TestWhitneyPeriodGapIsTopological)."""
    st, es, holes, P = holed_surface(degree=1, metric_source=source)
    cells1 = [tuple(int(v) for v in c) for c in es.cellSimplices()]
    return st, es, holes, P, cells1


def _emap(st):
    out = {}
    for e in st.getEdgeList().toVector():
        a, b = e.getSource().getId(), e.getTarget().getId()
        out[(min(a, b), max(a, b))] = e
    return out


def _perturb(st, cells1, every, factor):
    em = _emap(st)
    for i in range(0, len(cells1), every):
        ek = (min(cells1[i]), max(cells1[i]))
        em[ek].setLength(cmath.sqrt(complex((em[ek].getLength() ** 2).real * factor)))
    st.materializeFacets()


class PeriodGapValueTest(unittest.TestCase):
    """r_psi = 0 on a realizable target; floored once the carrier is perturbed."""

    def test_realizable_target_is_zero(self):
        # target = a carried period row: it lies in the carried span by
        # construction, so the pure-harmonic fit matches it exactly (gap ~ 0).
        _st, es, holes, P, _c = _substrate()
        target = [complex(z) for z in P[0]]
        self.assertLess(es.periodGapForPeriods(holes, target), 1e-12)

    def test_perturbed_geometry_floors_above_zero(self):
        st, es, holes, P, cells1 = _substrate()
        target = [complex(z) for z in P[0]]
        _perturb(st, cells1, every=7, factor=1.3)
        self.assertGreater(es.periodGapForPeriods(holes, target), 1e-6)

    def test_shares_r_u_zero_set_at_base(self):
        # At a realizable target both r_psi and r_U vanish.
        _st, es, holes, P, _c = _substrate()
        target = [complex(z) for z in P[0]]
        self.assertLess(es.periodGapForPeriods(holes, target), 1e-12)
        self.assertLess(es.residualForPeriods(holes, target), 1e-12)


class TestWhitneyPeriodGapIsTopological(unittest.TestCase):
    """On the Whitney pencil the period gap is blind to the geometry.

    The harmonic representatives are closed cochains whose periods span the
    cohomology image, so the same perturbation that floors the diagonal
    r_psi above 1e-6 leaves the Whitney r_psi at its rounding floor and its
    gradient at zero (measured 3.2e-30 and 1.6e-15 on this substrate). r_psi
    therefore carries no geometric content on the default operator."""

    def test_the_gap_and_its_gradient_stay_at_zero_off_the_carrier(self):
        st, es, holes, P, cells1 = _substrate(cob.HodgeMetricSource.WhitneyPencil)
        target = [complex(z) for z in P[0]]
        self.assertLess(es.periodGapForPeriods(holes, target), 1e-12)
        _perturb(st, cells1, every=7, factor=1.3)
        self.assertLess(es.periodGapForPeriods(holes, target), 1e-12)
        g = np.asarray(es.periodGapForPeriodsGradient(holes, target), complex)
        self.assertLess(float(np.linalg.norm(g)), 1e-9)
        # the diagonal path floors on the same perturbation
        st_d, es_d, holes_d, P_d, cells1_d = _substrate()
        _perturb(st_d, cells1_d, every=7, factor=1.3)
        self.assertGreater(es_d.periodGapForPeriods(holes_d, [complex(z) for z in P_d[0]]), 1e-6)


class PeriodGapGradientTest(unittest.TestCase):
    """The analytic r_psi gradient matches a central finite difference."""

    def test_gradient_matches_finite_difference(self):
        st, es, holes, P, cells1 = _substrate()
        target = [complex(z) for z in P[0]]
        _perturb(st, cells1, every=7, factor=1.3)  # r_psi > 0 off the carrier

        # The gradient is COMPLEX now (#746); Re is exactly the real-locus
        # derivative this test certified before, and these fixtures sit on
        # the real-l^2 locus, so the assertions below are unchanged.
        g = np.asarray(es.periodGapForPeriodsGradient(holes, target),
                       complex).real
        self.assertEqual(len(g), len(cells1))
        self.assertTrue(np.all(np.isfinite(g)))
        self.assertGreater(float(np.linalg.norm(g)), 1e-6)  # non-trivial

        em = _emap(st)
        h = 1e-6
        n1 = len(cells1)
        probe = sorted(set(int(i) for i in np.linspace(0, n1 - 1, 12)))
        worst = 0.0
        for idx in probe:
            ek = (min(cells1[idx]), max(cells1[idx]))
            e = em[ek]
            l0 = (e.getLength() * e.getLength()).real
            e.setLength(cmath.sqrt(complex(l0 + h))); st.materializeFacets()
            rp = es.periodGapForPeriods(holes, target)
            e.setLength(cmath.sqrt(complex(l0 - h))); st.materializeFacets()
            rm = es.periodGapForPeriods(holes, target)
            e.setLength(cmath.sqrt(complex(l0))); st.materializeFacets()
            fd = (rp - rm) / (2 * h)
            worst = max(worst, abs(g[idx] - fd))
        self.assertLess(worst, 1e-4, f"worst |analytic - FD| = {worst:.2e}")

    def test_gradient_near_zero_at_realizable_base(self):
        # At the realizable minimum (r_psi ~ 0) the gradient is ~ 0. (This alone
        # can't catch an all-zero stub -- the TRUE gradient is ~0 here too; that
        # regression is caught by test_gradient_matches_finite_difference, which
        # asserts norm(g) > 1e-6 at a perturbed point.)
        _st, es, holes, P, cells1 = _substrate()
        target = [complex(z) for z in P[0]]
        # The gradient is COMPLEX now (#746); Re is exactly the real-locus
        # derivative this test certified before, and these fixtures sit on
        # the real-l^2 locus, so the assertions below are unchanged.
        g = np.asarray(es.periodGapForPeriodsGradient(holes, target),
                       complex).real
        self.assertLess(float(np.max(np.abs(g))), 1e-5)


if __name__ == "__main__":
    unittest.main()
