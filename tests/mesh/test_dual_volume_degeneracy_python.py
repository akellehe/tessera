# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The dual-volume derivatives stay finite where the geometry is not singular.

The circumcentric dual is built from square roots of circumradius differences,
the distances between successive circumcentres. Differentiating divides by those
roots, so the expression diverges where a difference vanishes -- but only where
the difference also *moves*. A difference is pinned at zero along every direction
whose edge the facet does not carry, and that is most directions; there the root
is identically zero and so is its derivative.

Evaluating the quotient without taking the zero numerator first turns those
directions into ``(1/0) * 0``, which is NaN, and one NaN contaminates every edge
the hinge contributes to. On a four-dimensional Lorentzian complex at
N_4 ~ 10,500 that produced 36 non-finite gradient entries and 1,166 non-finite
Hessian entries from 14 hinges out of 27,457.
"""

import unittest

import numpy as np
import pytest
import tessera

# The degeneracy only appears once the chain has grown the volume, so the
# fixture sweeps 2,000 times at a target of 6,000 and the file takes about
# eleven minutes. It is marked slow for that reason, not because it is optional.
pytestmark = pytest.mark.slow


def _grown_spacetime(target=6000, seed=20260909, sweeps=2000):
    """A complex large enough to contain coincident circumradii.

    The degeneracy does not appear on a freshly built lattice; it develops as
    the Monte Carlo chain grows the volume, so the fixture has to sweep.
    """
    sig = tessera.Signature(4, tessera.Lorentzian)
    spacetime = tessera.Spacetime(tessera.Metric(True, sig), tessera.CDT,
                                  1.0, 1.0, tessera.PREFERRED, tessera.Toroid())
    spacetime.setSeed(seed)
    spacetime.build(1600)
    cdt = tessera.CDTSimulation(spacetime, 2.2, 0.5, 0.6, 1.0 / target, target)
    cdt.setSeed(seed)
    cdt.tune()
    cdt.sweep(sweeps)
    return spacetime


def _solver(spacetime):
    center = max(spacetime.getVertexList().toVector(), key=lambda v: v.degree())
    matter = tessera.MatterConfiguration()
    matter.setWorldlineMass(center, 1.0, spacetime)
    return tessera.ReggeSolver(spacetime, matter)


def _hinges(spacetime):
    return [s for s in spacetime.getSimplices()
            if len(s.getVertices()) == 3 and s.hasTopCoface()]


class TestDualVolumeDegeneracy(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.spacetime = _grown_spacetime()
        cls.solver = _solver(cls.spacetime)
        # The gradient pass materializes the hinges the dual is built on.
        cls.gradient = np.asarray(cls.solver.actionGradientExact(), dtype=complex)
        cls.hinges = _hinges(cls.spacetime)

    def test_the_fixture_actually_contains_a_degenerate_hinge(self):
        """Guards the rest of the file from passing vacuously."""
        degenerate = [h for h in self.hinges if h.dualGeometryIsDegenerate()]
        self.assertGreater(len(degenerate), 0)

    def test_every_action_gradient_entry_is_finite(self):
        self.assertTrue(np.isfinite(self.gradient).all())

    def test_every_per_hinge_dual_volume_gradient_is_finite(self):
        for hinge in self.hinges:
            values = np.asarray(list(hinge.dualVolumeGradient().values()),
                                dtype=complex)
            if values.size:
                self.assertTrue(np.isfinite(values).all())

    def test_the_predicate_names_every_hinge_whose_hessian_diverges(self):
        """A caller has to be able to find the responsible geometry.

        The second derivative genuinely does not exist where a circumradius
        difference vanishes and moves: the square root has infinite slope at the
        origin. Those entries stay non-finite, so the predicate is the way to
        identify which hinges they came from without hunting for the NaN.
        """
        diverging = []
        for hinge in self.hinges:
            values = np.asarray(list(hinge.dualVolumeHessian().values()),
                                dtype=complex)
            if values.size and not np.isfinite(values).all():
                diverging.append(hinge)
        flagged = {id(h) for h in self.hinges if h.dualGeometryIsDegenerate()}
        missed = [h for h in diverging if id(h) not in flagged]
        self.assertEqual(missed, [])

    def test_the_deficit_angle_hessian_is_finite(self):
        for hinge in self.hinges:
            values = np.asarray(list(hinge.deficitAngleHessian().values()),
                                dtype=complex)
            if values.size:
                self.assertTrue(np.isfinite(values).all())

    def test_a_non_degenerate_hinge_is_not_flagged(self):
        ordinary = [h for h in self.hinges if not h.dualGeometryIsDegenerate()]
        self.assertGreater(len(ordinary), 0)
        for hinge in ordinary[:200]:
            values = np.asarray(list(hinge.dualVolumeGradient().values()),
                                dtype=complex)
            if values.size:
                self.assertTrue(np.isfinite(values).all())


if __name__ == "__main__":
    unittest.main()
