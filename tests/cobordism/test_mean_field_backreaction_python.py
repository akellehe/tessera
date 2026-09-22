# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""#1189 — self-consistent mean-field backreaction with the force tr(Gamma dh/dz).

Section 7 fixes three things about the one channel from the carried state to the
geometry, and each is asserted here.

THE FORCE IS THE HELLMANN-FEYNMAN TERM, COMPLEX AND ON BOTH FIELDS. It is
``tr(Gamma dh/dz_e)`` on an edge and ``tr(Gamma U_e dh/dU_e)`` on a link. No
adjoint of the operator is formed, no real part is selected, and the link force
is present exactly when the operator is covariant — under the Whitney metric it
is nonzero and under the diagonal one it is identically zero, because that
operator is blind to the connection at every degree.

THE FORCE OBEYS THE EULER IDENTITY EXACTLY. The metric Hodge operator is
homogeneous of degree -1 in the squared lengths, so ``sum_e z_e dh/dz_e = -h``
and therefore ``sum_e z_e tr(Gamma dh/dz_e) = -tr(Gamma h)``. For a covariance
that projects onto occupied modes the right-hand side is minus the sum of their
eigenvalues, which is the whitepaper's statement that the length-weighted force
of an occupied mode is dilating: an occupied mode pushes its support to expand.
The identity is an independent check on both the operator gradient and the
contraction, and it is asserted to rounding rather than to the four decimals the
whitepaper quotes.

THE STATIONARY PAIR RE-OCCUPIES. Gamma* is the spectral projector onto the
occupied modes of h(z*), which is the whitepaper's ``Gamma = Phi PhiTilde^T``
for a matched left/right frame pair: idempotent, commuting with the operator it
was built from, and Hermitian only when that operator happens to be normal. A
covariance that is only transported forgets which modes of the current operator
it is supposed to fill, and the solve here re-occupies at every step.
"""

import os
import sys
import unittest

import numpy as np

import tessera as T

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _joint_action_hosts import (  # noqa: E402
    kuhn_ball, kuhn_interior_vertices, sphere3)

cob = T.cobordism


def _declaration(**overrides):
    """A joint-action declaration with the named fields overridden."""
    declaration = cob.JointActionDeclaration()
    declaration.carrier_degree = 1
    declaration.gravitational_weight = 0.0
    declaration.holonomy_weight = 0.0
    declaration.matter_weight = 0.0
    declaration.metric_source = cob.HodgeMetricSource.WhitneyPencil
    for name, value in overrides.items():
        setattr(declaration, name, value)
    return declaration


def _mean_field(**overrides):
    """A mean-field declaration with the named fields overridden.

    The inner relaxation is built as its own object and assigned whole, rather
    than mutated through the outer declaration's member, so the configuration a
    test asks for is the configuration the solve runs with.
    """
    geometry = cob.HolomorphicRelaxationDeclaration()
    geometry.relax_lengths = True
    geometry.relax_links = False
    geometry.relax_multipliers = False
    geometry.maximum_iterations = overrides.pop("geometry_iterations", 12)
    geometry.tolerance = 1e-12
    # The geometric term of these solves is the dual Regge action, whose exact
    # gradient is analytic on each side of the real axis in the squared lengths
    # and not across it: the deficit angle is taken on the principal branch, so
    # an arbitrarily small positive imaginary part shifts a hinge's deficit by
    # 2 pi. A contour around a real configuration reads two sheets; the
    # real-axis rule stays on one.
    geometry.jacobian_mode = cob.HolomorphicJacobianMode.RealAxisDifference
    geometry.contour_radius = 1e-5
    declaration = cob.SelfConsistentMeanFieldDeclaration()
    declaration.occupied_modes = 1
    declaration.maximum_iterations = 12
    declaration.tolerance = 1e-9
    declaration.geometry = geometry
    for name, value in overrides.items():
        setattr(declaration, name, value)
    return declaration


def _metric(index):
    return complex(1.0 + 0.041 * (index % 5), 0.017 * (1 + index % 3))


def _flux(index):
    return complex(0.19 * ((index % 5) - 2), 0.07 * ((index % 3) - 1))


def _squared_lengths(spacetime):
    return [complex(edge.getLength()) ** 2
            for edge in spacetime.getEdgeList().toVector()]


def _matrix(flat):
    order = int(round(len(flat) ** 0.5))
    return np.array(flat, dtype=complex).reshape(order, order)


class TheForceIsTheHellmannFeynmanTermTest(unittest.TestCase):
    """What the carried state pushes with, on both edge fields."""

    def _occupied(self, spacetime, modes=2, **overrides):
        seed = cob.JointAction(spacetime, _declaration(**overrides))
        declaration = _declaration(matter_weight=1.0, **overrides)
        declaration.covariance = seed.occupation_projector(modes, True)
        return cob.JointAction(spacetime, declaration)

    def test_the_length_force_is_complex(self):
        """No imaginary part is discarded on the way out of the trace."""
        spacetime = sphere3(squared=_metric, phase=_flux)
        action = self._occupied(spacetime)
        force = action.hellmann_feynman_length_force()
        self.assertGreater(max(abs(value.imag) for value in force), 1e-8)

    def test_the_link_force_exists_under_the_covariant_operator(self):
        spacetime = sphere3(squared=_metric, phase=_flux)
        covariant = self._occupied(spacetime)
        self.assertGreater(
            max(abs(value) for value in covariant.hellmann_feynman_link_force()),
            1e-8)

    def test_the_link_force_vanishes_under_the_blind_operator(self):
        """``DiagonalWeights`` ignores the connection, so it carries no force."""
        spacetime = sphere3(squared=_metric, phase=_flux)
        blind = self._occupied(
            spacetime, metric_source=cob.HodgeMetricSource.DiagonalWeights)
        for value in blind.hellmann_feynman_link_force():
            self.assertAlmostEqual(abs(value), 0.0, places=13)

    def test_the_force_obeys_the_euler_identity(self):
        """``sum_e z_e tr(Gamma dh/dz_e) = -tr(Gamma h)``, to rounding."""
        for source in (cob.HodgeMetricSource.DiagonalWeights,
                       cob.HodgeMetricSource.WhitneyPencil):
            with self.subTest(metric_source=source):
                spacetime = sphere3(squared=_metric, phase=_flux)
                action = self._occupied(spacetime, modes=3,
                                        metric_source=source)
                force = action.hellmann_feynman_length_force()
                weighted = sum(z * f for z, f
                               in zip(_squared_lengths(spacetime), force))
                energy = action.matter_term()
                self.assertAlmostEqual(
                    abs(weighted + energy), 0.0,
                    delta=1e-8 * (1.0 + abs(energy)))

    def test_an_occupied_mode_pushes_its_support_to_expand(self):
        """The length-weighted force of an occupied band is dilating.

        On a real geometry with a positive semidefinite operator the occupied
        energy is non-negative, so the Euler identity makes the length-weighted
        force non-positive: shrinking every edge raises the occupied energy and
        the band pushes the other way.
        """
        spacetime = kuhn_ball(divisions=2)
        action = self._occupied(
            spacetime, modes=1,
            metric_source=cob.HodgeMetricSource.DiagonalWeights)
        energy = action.matter_term()
        self.assertGreater(energy.real, 0.0)
        weighted = sum(z * f for z, f
                       in zip(_squared_lengths(spacetime),
                              action.hellmann_feynman_length_force()))
        self.assertLess(weighted.real, 0.0)
        self.assertAlmostEqual(abs(weighted + energy), 0.0,
                               delta=1e-8 * (1.0 + abs(energy)))


class TheOccupationProjectorTest(unittest.TestCase):
    """Gamma = Phi PhiTilde^T, built from a matched left/right frame pair."""

    def test_the_projector_is_idempotent_and_commutes_with_the_operator(self):
        spacetime = sphere3(squared=_metric, phase=_flux)
        action = cob.JointAction(spacetime, _declaration())
        projector = _matrix(action.occupation_projector(3, True))
        carrier = _matrix(action.carrier_operator())
        self.assertLess(np.max(np.abs(projector @ projector - projector)),
                        1e-9)
        self.assertLess(np.max(np.abs(projector @ carrier
                                      - carrier @ projector)),
                        1e-8 * (1.0 + np.max(np.abs(carrier))))
        self.assertAlmostEqual(abs(np.trace(projector) - 3.0), 0.0, places=8)

    def test_the_projector_carries_the_occupied_eigenvalues(self):
        """``tr(Gamma h)`` is the sum of the modes the projector selects."""
        spacetime = sphere3(squared=_metric, phase=_flux)
        seed = cob.JointAction(spacetime, _declaration())
        ordered = seed.ordered_carrier_eigenvalues(True)
        declaration = _declaration(matter_weight=1.0)
        declaration.covariance = seed.occupation_projector(3, True)
        action = cob.JointAction(spacetime, declaration)
        self.assertAlmostEqual(abs(action.matter_term() - sum(ordered[:3])),
                               0.0, delta=1e-8)

    def test_the_two_occupation_orders_differ_on_a_complex_spectrum(self):
        """The order is a declared choice, and it is one that matters."""
        spacetime = sphere3(squared=_metric, phase=_flux)
        action = cob.JointAction(spacetime, _declaration())
        by_real = action.ordered_carrier_eigenvalues(True)
        by_modulus = action.ordered_carrier_eigenvalues(False)
        self.assertNotEqual(by_real, by_modulus)

    def test_more_occupied_modes_than_cells_is_refused(self):
        spacetime = sphere3()
        action = cob.JointAction(spacetime, _declaration())
        with self.assertRaises(ValueError):
            action.occupation_projector(1000, True)


class TheSelfConsistentPairTest(unittest.TestCase):
    """#1189's acceptance: a fixed point of (z, Gamma)."""

    def _balanced_weight(self, spacetime, modes):
        """The gravitational weight that makes this geometry stationary.

        On the equilateral boundary of the 4-simplex every edge is carried to
        every other by a symmetry of the complex, so with the covariance
        invariant under the same symmetries the Regge gradient and the
        Hellmann-Feynman force are each the same complex number on every edge.
        The two are then balanced by one coefficient, and the configuration is
        an exact stationary point of the joint action rather than a nearby one.
        """
        regge = cob.JointAction(
            spacetime, _declaration(gravitational_weight=1.0)
        ).length_stationarity()
        seed = cob.JointAction(spacetime, _declaration())
        declaration = _declaration(matter_weight=1.0)
        declaration.covariance = seed.occupation_projector(modes, True)
        force = cob.JointAction(spacetime, declaration).length_stationarity()
        self.assertLess(max(abs(value - regge[0]) for value in regge),
                        1e-9 * (1.0 + abs(regge[0])))
        self.assertLess(max(abs(value - force[0]) for value in force),
                        1e-8 * (1.0 + abs(force[0])))
        return -(force[0] / regge[0]).real

    def test_the_solve_reaches_a_fixed_point_of_the_pair(self):
        reference = sphere3()
        modes = cob.ChainComplex.fromSpacetime(reference).numSimplices(1)
        weight = self._balanced_weight(reference, modes)

        spacetime = sphere3(squared=lambda index: 1.0 + 0.015 * ((index % 3) - 1))
        declaration = _declaration(gravitational_weight=weight,
                                   matter_weight=1.0)
        action = cob.JointAction(spacetime, declaration)
        seeded = _declaration(gravitational_weight=weight, matter_weight=1.0)
        seeded.covariance = cob.JointAction(
            spacetime, _declaration()).occupation_projector(modes, True)
        self.assertGreater(
            max(abs(value) for value
                in cob.JointAction(spacetime, seeded).length_stationarity()),
            1e-6,
            "the perturbed geometry must not already be stationary")

        solver = cob.SelfConsistentMeanField(
            action, _mean_field(occupied_modes=modes))
        report = solver.solve()

        self.assertTrue(report.converged,
                        (report.force_norm, report.covariance_change))
        self.assertLess(report.force_norm, 1e-9)
        self.assertLess(report.covariance_change, 1e-9)
        self.assertLess(report.purity_defect, 1e-9)

        # Gamma* projects onto modes of h(z*).
        final = solver.action
        projector = _matrix(report.covariance)
        carrier = _matrix(final.carrier_operator())
        self.assertLess(np.max(np.abs(projector @ projector - projector)),
                        1e-9)
        self.assertLess(np.max(np.abs(projector @ carrier
                                      - carrier @ projector)),
                        1e-7 * (1.0 + np.max(np.abs(carrier))))

        # The force balances the geometric action, edge by edge.
        regge = cob.JointAction(
            spacetime, _declaration(gravitational_weight=weight)
        ).length_stationarity()
        force = final.hellmann_feynman_length_force()
        for geometric, carried in zip(regge, force):
            self.assertLess(abs(geometric + carried),
                            1e-9 * (1.0 + abs(geometric)))

    def test_every_step_leaves_the_covariance_a_projector(self):
        """The loop stays inside the Gaussian class, and the step says so."""
        spacetime = sphere3(squared=lambda index: 1.0 + 0.02 * (index % 4))
        action = cob.JointAction(
            spacetime, _declaration(gravitational_weight=90.0,
                                    matter_weight=1.0))
        solver = cob.SelfConsistentMeanField(
            action, _mean_field(occupied_modes=2, maximum_iterations=3,
                                geometry_iterations=3))
        report = solver.solve()
        self.assertGreater(len(report.steps), 0)
        for step in report.steps:
            self.assertLess(step.purity_defect, 1e-9)
            self.assertEqual(len(step.occupied_eigenvalues), 2)

    def test_the_covariance_is_re_occupied_at_every_step(self):
        """The half of the fixed point a transported covariance never takes.

        After the solve the covariance is the spectral projector of the
        operator at the geometry the solve left behind, not of the operator it
        started from. The two differ, and the difference is what re-occupation
        does.
        """
        spacetime = sphere3(squared=lambda index: 1.0 + 0.02 * (index % 4))
        start = cob.JointAction(spacetime, _declaration())
        initial = np.array(start.occupation_projector(2, True), dtype=complex)

        action = cob.JointAction(
            spacetime, _declaration(gravitational_weight=90.0,
                                    matter_weight=1.0))
        solver = cob.SelfConsistentMeanField(
            action, _mean_field(occupied_modes=2, maximum_iterations=2,
                                geometry_iterations=3))
        report = solver.solve()

        final = np.array(report.covariance, dtype=complex)
        self.assertGreater(np.max(np.abs(final - initial)), 1e-9)
        reread = np.array(
            solver.action.occupation_projector(2, True), dtype=complex)
        self.assertLess(np.max(np.abs(final - reread)), 1e-9)


class TheKuhnBallCarriesTheSection7SetupTest(unittest.TestCase):
    """The complex the Section 7 self-trapping computation runs on."""

    def test_the_kuhn_ball_is_a_ball(self):
        spacetime = kuhn_ball(divisions=2)
        complex_ = cob.ChainComplex.fromSpacetime(spacetime)
        self.assertEqual(complex_.dimension(), 3)
        self.assertEqual(complex_.bettiNumbers(), [1, 0, 0, 0])
        self.assertEqual(len(kuhn_interior_vertices(2)), 1)
        self.assertEqual(len(kuhn_interior_vertices(3)), 8)

    def test_the_occupation_readout_sums_to_the_filled_mode_count(self):
        """``n_c = Gamma_cc`` sums to the trace of the projector, which is 1."""
        spacetime = kuhn_ball(divisions=3)
        seed = cob.JointAction(
            spacetime,
            _declaration(metric_source=cob.HodgeMetricSource.DiagonalWeights))
        declaration = _declaration(
            matter_weight=1.0,
            metric_source=cob.HodgeMetricSource.DiagonalWeights)
        declaration.covariance = seed.occupation_projector(1, True)
        action = cob.JointAction(spacetime, declaration)
        occupations = np.array(action.occupation_numbers(), dtype=complex)
        cells = cob.ChainComplex.fromSpacetime(spacetime).numSimplices(1)
        self.assertEqual(len(occupations), cells)
        self.assertAlmostEqual(abs(occupations.sum() - 1.0), 0.0, places=8)
        self.assertGreater(np.max(np.abs(occupations)), 0.0)


class TheDeclaredControlsAreCheckedTest(unittest.TestCase):
    """The configuration a caller may not silently get wrong."""

    def test_a_mixing_outside_the_unit_interval_is_refused(self):
        spacetime = sphere3()
        action = cob.JointAction(spacetime, _declaration(matter_weight=1.0))
        for mixing in (0.0, -0.5, 1.5):
            with self.subTest(mixing=mixing):
                with self.assertRaises(ValueError):
                    cob.SelfConsistentMeanField(action,
                                                _mean_field(mixing=mixing))

    def test_an_empty_occupation_is_refused(self):
        spacetime = sphere3()
        action = cob.JointAction(spacetime, _declaration(matter_weight=1.0))
        with self.assertRaises(ValueError):
            cob.SelfConsistentMeanField(action, _mean_field(occupied_modes=0))


if __name__ == "__main__":
    unittest.main()
