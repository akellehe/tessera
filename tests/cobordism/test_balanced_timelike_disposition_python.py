# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""Balanced wiring must not erase a cone-in's causal disposition (#741).

Balanced wiring writes every new edge with equal real and imaginary parts, so
`Re l^2 = 0` and the edge is born causally UNDECIDED. A timelike cone-in still
has to differ from a spacelike one, and under this convention the difference is
the branch — the sign of `Im l^2` — not the sign of `Re l^2`.

It previously did not: the balanced branch passed `-kTimelikeSquaredLength` to
`balancedLength`, and that constant is already `-1.0`, so the argument was `+1`
— exactly the spacelike auto-wiring value. Both cone kinds produced identical
edges and the disposition vanished.
"""
import cmath
import unittest

import tessera

cobordism = tessera.cobordism


def _cone(balanced, timelike):
    """Apex-edge squared lengths from one cone-in on a fresh node."""
    node = cobordism.ProtonSynthesis(seed=3, precone=0,
                            balanced_edges=balanced).direct_node(3)
    spacetime = node.st

    def keys():
        return {(min(e.getSource().getId(), e.getTarget().getId()),
                 max(e.getSource().getId(), e.getTarget().getId()))
                for e in spacetime.getEdgeList().toVector()}

    before = keys()
    cone = cobordism.SurgicalCone(spacetime)
    cell = sorted(v.getId() for v in spacetime.getTopSimplices()[0].getVertices())
    accepted, reason = cone.coneIn(cell[:-1], timelike)
    assert accepted, reason
    return [e.getLength() ** 2 for e in spacetime.getEdgeList().toVector()
            if (min(e.getSource().getId(), e.getTarget().getId()),
                max(e.getSource().getId(), e.getTarget().getId())) not in before]


class BalancedTimelikeDispositionTest(unittest.TestCase):

    def test_unbalanced_dispositions_are_real_and_opposite(self):
        for squared in _cone(balanced=False, timelike=False):
            self.assertAlmostEqual(squared.real, 1.0, places=12)   # spacelike
            self.assertAlmostEqual(squared.imag, 0.0, places=12)
        for squared in _cone(balanced=False, timelike=True):
            self.assertAlmostEqual(squared.real, -1.0, places=12)  # timelike
            self.assertAlmostEqual(squared.imag, 0.0, places=12)

    def test_balanced_dispositions_stay_undecided_but_differ_by_branch(self):
        spacelike = _cone(balanced=True, timelike=False)
        timelike = _cone(balanced=True, timelike=True)
        self.assertTrue(spacelike and timelike)
        for squared in spacelike:
            self.assertAlmostEqual(squared.real, 0.0, places=12)   # undecided
            self.assertAlmostEqual(squared.imag, 1.0, places=12)
        for squared in timelike:
            self.assertAlmostEqual(squared.real, 0.0, places=12)   # undecided
            self.assertAlmostEqual(squared.imag, -1.0, places=12)  # other branch

    def test_the_two_balanced_kinds_are_not_identical(self):
        # The regression itself: under balanced wiring these two coincided.
        self.assertNotEqual(set(_cone(balanced=True, timelike=False)),
                            set(_cone(balanced=True, timelike=True)))

    def test_magnitude_is_preserved_across_wirings(self):
        # #691's convention: the wiring changes where l^2 points, never how big
        # it is, so every one of the four cases carries |l^2| = 1.
        for balanced in (False, True):
            for timelike in (False, True):
                for squared in _cone(balanced, timelike):
                    self.assertAlmostEqual(abs(squared), 1.0, places=12)


if __name__ == "__main__":
    unittest.main()
