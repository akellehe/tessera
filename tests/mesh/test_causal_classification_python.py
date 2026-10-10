# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""Causal type is the ARGUMENT of the squared length, not a magnitude.

Writing ``l = |l| e^{i a}``, ``l^2 = |l|^2 e^{2 i a}``, so the disposition is
fixed by ``arg(l^2)``: zero is spacelike, ``+/-pi`` timelike, ``+/-pi/2``
lightlike, and anything else genuinely mixed.

Two magnitudes have to be kept apart, and conflating them was the defect this
pins (#870). The EUCLIDEAN modulus ``sqrt(x^2 + t^2)`` vanishes only when the
edge itself vanishes, so testing it makes a null INTERVAL undetectable. The
LORENTZIAN magnitude ``Re(l^2) = x^2 - t^2`` vanishes on the light cone, which
is reachable at ``Re(l) == Im(l) > 0`` on an edge of perfectly ordinary extent.

Classifying on ``Re(l^2)`` alone would not do either: it discards ``Im(l^2)``,
which is ``2x^2 != 0`` exactly at the lightlike point. A fully null ``l^2 = 0``
does not exist non-trivially, so both components have to be accounted for --
hence the argument.
"""

import cmath
import math
import unittest

from tessera import Vertex, Edge


def _edge(length):
    return Edge(Vertex(1, [0.0, 0.0, 0.0, 0.0]),
                Vertex(2, [0.0, 0.0, 0.0, 1.0]), complex(length))


def _cases(edge):
    return (edge.is_spacelike(), edge.is_timelike(), edge.is_null(),
            edge.is_mixed(), edge.is_degenerate())


class ExactlyOneCaseTest(unittest.TestCase):
    """Whatever the length, exactly one of the five predicates holds."""

    def test_exactly_one_case_holds_across_the_whole_circle(self):
        for step in range(721):
            angle = -math.pi + step * (2.0 * math.pi / 720.0)
            edge = _edge(cmath.exp(1j * angle))
            with self.subTest(angle=angle):
                self.assertEqual(sum(bool(c) for c in _cases(edge)), 1)

    def test_exactly_one_case_holds_for_a_degenerate_edge(self):
        self.assertEqual(sum(bool(c) for c in _cases(_edge(0.0))), 1)


class DefiniteDispositionsTest(unittest.TestCase):
    """The three definite arguments, each asserted on arg(l^2)."""

    def test_a_real_length_is_spacelike(self):
        edge = _edge(1.0)
        self.assertAlmostEqual(edge.squared_argument(), 0.0, places=12)
        self.assertEqual(_cases(edge), (True, False, False, False, False))

    def test_an_imaginary_length_is_timelike(self):
        edge = _edge(1j)
        self.assertAlmostEqual(abs(edge.squared_argument()), math.pi, places=12)
        self.assertEqual(_cases(edge), (False, True, False, False, False))

    def test_equal_parts_are_lightlike_and_not_degenerate(self):
        """The case the superseded classifier could not express.

        Asserts the argument and the nonzero extent SEPARATELY, so this cannot
        pass by the edge having collapsed -- which is what "null" used to mean.
        """
        component = math.sqrt(0.5)
        edge = _edge(complex(component, component))
        self.assertAlmostEqual(edge.squared_argument(), math.pi / 2.0, places=12)
        self.assertAlmostEqual(edge.lorentzian_magnitude(), 0.0, places=15)
        self.assertGreater(abs(edge.get_length()), 0.5)   # genuine extent
        self.assertEqual(_cases(edge), (False, False, True, False, False))

    def test_the_other_light_cone_branch_is_also_lightlike(self):
        component = math.sqrt(0.5)
        edge = _edge(complex(component, -component))
        self.assertAlmostEqual(edge.squared_argument(), -math.pi / 2.0, places=12)
        self.assertEqual(_cases(edge), (False, False, True, False, False))


class MixedTest(unittest.TestCase):
    """A generic argument is mixed, never snapped to the nearest definite."""

    def test_a_generic_argument_is_mixed(self):
        for angle in (0.3, 1.0, 2.0, -0.9, 2.9):
            edge = _edge(cmath.exp(1j * angle))
            with self.subTest(angle=angle):
                self.assertEqual(_cases(edge), (False, False, False, True, False))

    def test_an_almost_definite_argument_is_still_mixed(self):
        """Just outside the half-width is mixed, not rounded to definite.

        This is the property that keeps the classification honest: a bucket
        that absorbed nearby arguments would report a definiteness the geometry
        does not have.
        """
        edge = _edge(cmath.exp(1j * 1e-5))   # arg(l^2) = 2e-5, far above 1e-9
        self.assertTrue(edge.is_mixed())
        self.assertFalse(edge.is_spacelike())

    def test_a_uniformly_drawn_argument_is_almost_always_mixed(self):
        """Measured, and the reason a random seed has no causal structure."""
        mixed = 0
        total = 400
        for step in range(total):
            angle = (step + 0.5) * (2.0 * math.pi / total)
            if _edge(cmath.exp(1j * angle)).is_mixed():
                mixed += 1
        self.assertGreater(mixed / total, 0.98)


class DegenerateTest(unittest.TestCase):
    """An absent edge is not a causal type, and is not lightlike."""

    def test_zero_length_is_degenerate_not_null(self):
        edge = _edge(0.0)
        self.assertTrue(edge.is_degenerate())
        self.assertFalse(edge.is_null())
        self.assertFalse(edge.is_spacelike())
        self.assertFalse(edge.is_timelike())
        self.assertFalse(edge.is_mixed())

    def test_a_degenerate_edge_is_not_reported_spacelike(self):
        """arg(0) is 0, which would read as spacelike if extent went unchecked.

        The degenerate test therefore has to run BEFORE the argument tests,
        and this pins that ordering.
        """
        self.assertAlmostEqual(_edge(0.0).squared_argument(), 0.0, places=12)
        self.assertFalse(_edge(0.0).is_spacelike())


class ScaleInvarianceTest(unittest.TestCase):
    """An angular tolerance is scale-free; an absolute one would not be."""

    def test_disposition_is_unchanged_by_rescaling(self):
        component = math.sqrt(0.5)
        for scale in (1e-6, 1e-3, 1.0, 1e3, 1e6):
            with self.subTest(scale=scale):
                self.assertTrue(_edge(scale).is_spacelike())
                self.assertTrue(_edge(scale * 1j).is_timelike())
                self.assertTrue(
                    _edge(complex(scale * component, scale * component)).is_null())

    def test_the_lorentzian_magnitude_scales_but_the_type_does_not(self):
        """Re(l^2) grows as the square, which is why it cannot carry an
        absolute tolerance -- the disposition must not depend on the scale."""
        small = _edge(1e-3)
        large = _edge(1e3)
        self.assertLess(small.lorentzian_magnitude(), large.lorentzian_magnitude())
        self.assertTrue(small.is_spacelike())
        self.assertTrue(large.is_spacelike())


if __name__ == "__main__":
    unittest.main()
