# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""The singular-value half-sum ratio — the whole-complex term of r_U in the
singular_value_ratio mode (#697).

The term is the ratio of the sum of the lower half of the singular values of
the METRIC L_k (the same signed operator nearKernelResidual reads) to the sum
of the upper half; an odd count leaves the median out of both halves. Each
lower-half value is bounded by its upper-half counterpart, so the ratio lives
in [0, 1], and L_k is homogeneous of degree -1 in l^2, so a uniform rescale
scales every sigma alike and cancels — degree 0, no conformal-inflation
channel. No target enters: the mode replaces BOTH the single-output period
residual and the near-kernel continuation, and WHAT the register comes to
carry is read out afterwards (r_state stays the verdict).
"""
import cmath
import importlib.util
import os
import unittest

import tessera

cob = tessera.cobordism

_HS = os.path.join(os.path.dirname(__file__), "_holed_surface.py")
_spec = importlib.util.spec_from_file_location("_holed_surface", _HS)
_hs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_hs)

_OMEGA = complex(cmath.exp(2j * cmath.pi / 3))
_SINGLET = [1 + 0j, _OMEGA, _OMEGA * _OMEGA]


def _seed():
    return tessera.Spacetime.fromVertexTuples(4, [[0, 1, 2, 3, 4]], 1.0, 0.0)


class SingularValueHalfSumRatioTest(unittest.TestCase):

    def test_range_on_generic_geometry(self):
        # A generic spectrum has a strictly positive, strictly dominated lower
        # half: the ratio is inside (0, 1).
        st, _es, _holes, _periods = _hs.holed_surface(degree=1)
        r = cob.MultiCobordism.singularValueHalfSumRatio(st, 1)
        self.assertGreater(r, 0.0)
        self.assertLess(r, 1.0)
        seed_ratio = cob.MultiCobordism.singularValueHalfSumRatio(_seed(), 3)
        self.assertGreater(seed_ratio, 0.0)
        self.assertLessEqual(seed_ratio, 1.0)

    def test_scale_invariant(self):
        # L_k is homogeneous of degree -1 in l^2: a uniform rescale scales
        # every singular value by the same factor and the half-sum ratio
        # cancels exactly (up to SVD round-off).
        st, _es, _holes, _periods = _hs.holed_surface(degree=1)
        r0 = cob.MultiCobordism.singularValueHalfSumRatio(st, 1)
        edges = st.getEdgeList().toVector()
        base = [e.getLength() for e in edges]
        for e, l in zip(edges, base):
            e.setLength(l * cmath.sqrt(2.0))
        st.materializeFacets()
        r1 = cob.MultiCobordism.singularValueHalfSumRatio(st, 1)
        self.assertAlmostEqual(r0, r1, places=9)

    def test_edge_counts(self):
        # One k-cell (the pentatope itself at k = 4): a single mode has no
        # pair of halves to compare — 0. Above the top dimension there are no
        # k-cells at all: the worst case 1 (an empty degree must never score
        # as a collapsed spectrum, else deleting cells beats collapsing).
        st = _seed()
        self.assertEqual(cob.MultiCobordism.singularValueHalfSumRatio(st, 4), 0.0)
        self.assertEqual(cob.MultiCobordism.singularValueHalfSumRatio(st, 5), 1.0)

    def test_mode_swaps_the_whole_complex_term_in_ru(self):
        # Before any input block is seeded, r_U is the whole-complex term
        # alone. In the ratio mode that term is exactly the half-sum ratio —
        # no period residual, no near-kernel term — while the default mode
        # keeps the period leak + near-kernel pair, so the two modes disagree
        # on the same geometry.
        inputs = [[1 + 0j], [_OMEGA], [_OMEGA * _OMEGA]]
        ratio_node = cob.MultiCobordism(_seed(), inputs, [_SINGLET],
                                        degrees=[3], gamma=1.0, seed=7,
                                        singular_value_ratio=True)
        default_node = cob.MultiCobordism(_seed(), inputs, [_SINGLET],
                                          degrees=[3], gamma=1.0, seed=7)
        st = _seed()
        self.assertEqual(ratio_node.r_u(st),
                         cob.MultiCobordism.singularValueHalfSumRatio(st, 3))
        self.assertNotEqual(ratio_node.r_u(st), default_node.r_u(st))

    def test_proton_forwards_the_flag(self):
        # The keyword reaches ProtonSynthesis and each node it constructs. direct_node
        # seeds its six input blocks, so its r_U also carries the input-block
        # residuals — identical between the two modes (same seed, same
        # anchors). The whole-complex term is the only difference, and the
        # default mode's (singlet period leak + near-kernel, >= 3 before any
        # hole) always exceeds the ratio's [0, 1]: the two must disagree.
        ratio_node = cob.ProtonSynthesis(seed=11, singular_value_ratio=True).direct_node(11)
        default_node = cob.ProtonSynthesis(seed=11).direct_node(11)
        st = _seed()
        self.assertGreater(default_node.r_u(st) - ratio_node.r_u(st), 1.0)


if __name__ == "__main__":
    unittest.main()
