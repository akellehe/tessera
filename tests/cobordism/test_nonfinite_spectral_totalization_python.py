# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""The spectral residual terms are total over non-finite configurations (#699).

Stage-2 line-search trials are deliberately unbounded (#589), and the cell
weights are polynomial in the edge lengths (Content ~ l^k, SquaredContent
~ l^{2k}), so a legal extreme trial can push the assembled operator or the
harmonic period matrix out of double range. Handing non-finite input to
Eigen's BDCSVD is undefined behavior with asserts compiled out (measured: a
general protection fault inside rank() on a live run). The residual terms
instead evaluate such configurations to +inf — infinitely bad, so the line
search rejects the trial and the run continues — with finite evaluations
bitwise-unchanged.
"""
import cmath
import importlib.util
import math
import os
import unittest

import tessera

cob = tessera.cobordism

_HS = os.path.join(os.path.dirname(__file__), "_holed_surface.py")
_spec = importlib.util.spec_from_file_location("_holed_surface", _HS)
_hs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_hs)


def _overflowed_holed_surface():
    """The holed surface (b_1 = 2) with one edge pushed far enough that the
    SquaredContent weights of its cofaces leave double range."""
    st, _es, _holes, _periods = _hs.holed_surface(degree=1)
    edges = st.getEdgeList().toVector()
    edges[0].setLength(edges[0].getLength() * 1e80)
    st.materializeFacets()
    return st


class NonfiniteSpectralTotalizationTest(unittest.TestCase):

    def setUp(self):
        self._prior_convention = cob.HodgeLaplacian.defaultWeightConvention()
        cob.HodgeLaplacian.setDefaultWeightConvention(
            cob.HodgeWeightConvention.SquaredContent)

    def tearDown(self):
        cob.HodgeLaplacian.setDefaultWeightConvention(self._prior_convention)

    # The SquaredContent weights are those of the diagonal-weight operator,
    # so the two terms are read with that metric source: the default one,
    # the Whitney pencil, has no SquaredContent weight and is finite on this
    # geometry (its largest entry is 19.2), where the terms read 0.0 and a
    # finite ratio.

    def test_near_kernel_residual_totalizes(self):
        st = _overflowed_holed_surface()
        r = cob.MultiCobordism.nearKernelResidual(
            st, 1, 3, cob.HodgeMetricSource.DiagonalWeights)
        self.assertEqual(r, math.inf)

    def test_half_sum_ratio_totalizes(self):
        st = _overflowed_holed_surface()
        r = cob.MultiCobordism.singularValueHalfSumRatio(
            st, 1, cob.HodgeMetricSource.DiagonalWeights)
        self.assertEqual(r, math.inf)

    def test_r_state_survives_the_crash_geometry(self):
        # The measured GPF was BDCSVD::rank() inside the period fit. The
        # contract here is survival with a defined value: the evaluation
        # returns (+inf when the period fit sees non-finite periods, or the
        # finite full leak when the broken spectrum yields no usable
        # harmonics) — it never reaches Eigen with non-finite input.
        st = _overflowed_holed_surface()
        r = cob.MultiCobordism.r_state(st, 1, [1 + 0j, -1 + 0j])
        self.assertFalse(math.isnan(r))
        self.assertGreaterEqual(r, 0.0)

    def test_finite_geometry_unchanged(self):
        # A clean fixture still evaluates finite and sane through every term.
        st, _es, _holes, _periods = _hs.holed_surface(degree=1)
        near = cob.MultiCobordism.nearKernelResidual(st, 1, 3)
        ratio = cob.MultiCobordism.singularValueHalfSumRatio(st, 1)
        r_state = cob.MultiCobordism.r_state(st, 1, [1 + 0j, -1 + 0j])
        for value in (near, ratio, r_state):
            self.assertTrue(math.isfinite(value))
        self.assertEqual(cob.MultiCobordism.nearKernelResidual(st, 1, 1), 0.0)


if __name__ == "__main__":
    unittest.main()
