# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""Causal character is read from the ARGUMENT of the squared edge LENGTH: an
edge is spacelike at `arg(l^2) ~ 0`, timelike at `~ +/-pi`, lightlike at
`~ +/-pi/2`, and mixed anywhere else (#870).
`get_length()` is the metric DOF (distinct from the U(1) `get_phase()`) and the
edge's ONLY stored quantity: real for spacelike, imaginary for timelike, and it
squares to the squared length rather than being derived from one (#639)."""

import cmath
import math
import unittest

from tessera import Vertex, Edge


def _edge(sq):
    # The Edge ctor takes the complex LENGTH (#639); l^2 is never stored, so a
    # fixture specified by a squared value goes in as its principal root -- real
    # for spacelike, imaginary for timelike.
    return Edge(Vertex(1, [0.0, 0.0, 0.0, 0.0]),
                Vertex(2, [0.0, 0.0, 0.0, 1.0]), cmath.sqrt(complex(sq, 0.0)))


class EdgeComplexLengthTest(unittest.TestCase):
    def test_spacelike_has_real_length(self):
        e = _edge(25.0)
        L = e.get_length()
        self.assertAlmostEqual(L.real, 5.0, places=12)
        self.assertAlmostEqual(L.imag, 0.0, places=12)
        self.assertTrue(e.is_spacelike())
        self.assertFalse(e.is_timelike())
        self.assertFalse(e.is_null())

    def test_timelike_has_imaginary_length(self):
        e = _edge(-4.0)
        L = e.get_length()
        self.assertAlmostEqual(L.real, 0.0, places=12)
        self.assertAlmostEqual(L.imag, 2.0, places=12)   # length = i*2
        self.assertTrue(e.is_timelike())
        self.assertFalse(e.is_spacelike())
        self.assertFalse(e.is_null())

    def test_zero_length_is_degenerate_not_null(self):
        """An absent edge is DEGENERATE; a null edge is a lightlike ray.

        These were conflated while causal type was read from the Euclidean
        modulus, which can only vanish when the edge itself vanishes (#870).
        Reading `arg(l^2)` separates them -- see the lightlike test below.
        """
        e = _edge(0.0)
        self.assertAlmostEqual(abs(e.get_length()), 0.0, places=12)
        self.assertTrue(e.is_degenerate())
        self.assertFalse(e.is_null())
        self.assertFalse(e.is_timelike())
        self.assertFalse(e.is_spacelike())
        self.assertFalse(e.is_mixed())

    def test_equal_parts_are_lightlike_on_an_edge_of_nonzero_extent(self):
        """`Re(l) == Im(l) > 0` is the non-trivial lightlike case.

        Asserts the argument and the nonzero extent SEPARATELY, so the test
        cannot pass by the edge having collapsed to nothing.
        """
        component = math.sqrt(0.5)
        e = _edge(1.0)
        e.set_length(complex(component, component))
        self.assertGreater(abs(e.get_length()), 0.5)          # not degenerate
        self.assertFalse(e.is_degenerate())
        self.assertAlmostEqual(e.squared_argument(), math.pi / 2.0, places=12)
        self.assertTrue(e.is_null())
        self.assertFalse(e.is_timelike())
        self.assertFalse(e.is_spacelike())
        self.assertFalse(e.is_mixed())

    def test_a_generic_argument_is_mixed(self):
        """A generic argument is NOT snapped to the nearest definite type."""
        e = _edge(1.0)
        e.set_length(cmath.exp(0.3j))
        self.assertTrue(e.is_mixed())
        for definite in (e.is_spacelike(), e.is_timelike(), e.is_null(),
                         e.is_degenerate()):
            self.assertFalse(definite)

    def test_squaring_length_recovers_squared_length(self):
        for sq in (25.0, -4.0, 1.0, -1.0, 100.0, -0.5):
            length = _edge(sq).get_length()
            sq2 = length * length
            self.assertAlmostEqual(sq2.real, sq, places=9)
            self.assertAlmostEqual(sq2.imag, 0.0, places=12)

    def test_causal_character_is_a_length_test_not_a_magnitude_test(self):
        # tiny-but-nonzero squared lengths still resolve cleanly by imaginary part
        self.assertTrue(_edge(-1e-6).is_timelike())
        self.assertTrue(_edge(1e-6).is_spacelike())


if __name__ == "__main__":
    unittest.main()
