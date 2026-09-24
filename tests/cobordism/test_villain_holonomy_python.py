# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""The Villain holonomy term of the joint action.

The holonomy term of S(z, U, Gamma) is the Villain (heat-kernel) action in its
character form,

    S_hol = -beta_V sum_tau log W(F_tau),   W(F) = sum_m exp(-m^2/(2 beta)) F^m,

with beta_V = beta / <m^2>_beta, where <m^2>_beta is the second moment of the
weights exp(-m^2/(2 beta)). The suite asserts, each as a measured number:

* the matching: at trivial holonomy the second variation is beta L_1^up, the
  Wilson form's, entry by entry, so the tetrahedron's coexact block is 4 beta;
* the quarter-turn stiffness: at F = +-i, where the Wilson curvature vanishes,
  the Villain curvature is beta_V (<m^2>_i - <m>_i^2), evaluated here from its
  closed form, and the tetrahedron's coexact block is four times it;
* gauge invariance under the complex gauge group, the symmetry F <-> 1/F, and
  stationarity at trivial holonomy;
* the link stationarity (the Ward current) and the Hessian against finite
  differences at complex face holonomies, in two complex directions each, which
  is also the holomorphy of the term;
* the logarithm branch: real on the unit circle, exp of it equal to W, and a
  refusal at a zero of W;
* the truncation: the declared term count, and a tail bound that holds against
  the untruncated series and is not loose by more than a small factor.
"""

import cmath
import math
import os
import sys
import unittest

import numpy as np

import tessera as T

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _joint_action_hosts import sphere3, tetrahedron  # noqa: E402

cob = T.cobordism


def _declaration(form=cob.HolonomyForm.Villain, beta=1.3, **overrides):
    """A joint action of the holonomy term alone, in the named form."""
    declaration = cob.JointActionDeclaration()
    declaration.carrier_degree = 1
    declaration.gravitational_weight = 0.0
    declaration.regge_form = cob.ReggeForm.Dual
    declaration.holonomy_weight = beta
    declaration.holonomy_form = form
    declaration.matter_weight = 0.0
    declaration.metric_source = cob.HodgeMetricSource.WhitneyPencil
    for name, value in overrides.items():
        setattr(declaration, name, value)
    return declaration


def _flux(index):
    """Complex link phases that are no pure gauge, so every F is complex."""
    return complex(0.23 * ((index % 5) - 2), 0.09 * ((index % 3) - 1))


def _chi(spacetime, seed=0):
    """A complex vertex function: the parameter of a C* gauge transformation."""
    values = {}
    for index, vertex in enumerate(spacetime.getVertexList().toVector()):
        step = index + seed
        values[int(vertex.getId())] = complex(0.31 * ((step % 7) - 3),
                                              0.17 * ((step % 4) - 1.5))
    return values


def _gauge(spacetime, chi):
    """U_xy -> g_x^-1 U_xy g_y with g = exp(i chi), on the stored phases."""
    for edge in spacetime.getEdgeList().toVector():
        source = int(edge.getSource().getId())
        target = int(edge.getTarget().getId())
        edge.setPhase(edge.getPhase() + chi[target] - chi[source])


def _multiply_link(spacetime, index, delta):
    """U_e -> U_e exp(delta) on the stored orientation: phi -> phi - i delta."""
    edge = spacetime.getEdgeList().toVector()[index]
    edge.setPhase(edge.getPhase() - 1j * delta)


def _signed_up_laplacian(spacetime):
    """d_2 d_2^T in getEdgeList() order on the stored orientations."""
    complex_ = cob.ChainComplex.fromSpacetime(spacetime)
    boundary = np.array(complex_.boundaryMatrix(2), dtype=float).reshape(
        complex_.numSimplices(1), complex_.numSimplices(2))
    index_of = {tuple(cell): position for position, cell
                in enumerate(complex_.kSimplexVertices(1))}
    rows = []
    for edge in spacetime.getEdgeList().toVector():
        source = int(edge.getSource().getId())
        target = int(edge.getTarget().getId())
        sign = 1.0 if source < target else -1.0
        rows.append(sign * boundary[index_of[tuple(sorted((source,
                                                           target)))]])
    signed = np.array(rows)
    return signed @ signed.T


def _second_moment(beta, terms=400):
    m = np.arange(-terms, terms + 1)
    weights = np.exp(-m * m / (2.0 * beta))
    return float(np.sum(m * m * weights) / np.sum(weights))


def _quarter_turn_stiffness(beta, terms=200):
    """kappa(i) = beta_V (<m^2>_i - <m>_i^2) from its closed form: only the even
    m = 2k enter W(i) and <m^2>_i, and only the odd m = 2j + 1 enter <m>_i."""
    k = np.arange(-terms, terms + 1)
    alternating = (-1.0) ** np.abs(k)
    even = np.exp(-2.0 * k * k / beta)
    w_i = np.sum(alternating * even)
    second = np.sum(alternating * 4.0 * k * k * even) / w_i
    j = np.arange(0, terms)
    odd = (2 * j + 1).astype(float)
    mean = 2j * np.sum((-1.0) ** j * odd * np.exp(-odd * odd / (2.0 * beta))) / w_i
    beta_v = beta / _second_moment(beta)
    return float(np.real(beta_v * (second - mean * mean)))


def _quarter_turn_tetrahedron():
    """A tetrahedron whose four face holonomies are all +-i: the unit monopole.

    The outward fluxes are pi/2, pi/2, pi/2 and -3 pi/2, which sum to zero as
    every coboundary must and give F = i on every outward face.
    """
    spacetime = tetrahedron(squared=lambda index: 8.0)
    complex_ = cob.ChainComplex.fromSpacetime(spacetime)
    edges, faces = complex_.numSimplices(1), complex_.numSimplices(2)
    d2 = np.array(complex_.boundaryMatrix(2), dtype=float).reshape(edges, faces)
    d3 = np.array(complex_.boundaryMatrix(3), dtype=float).reshape(faces, 1)
    outward = d3[:, 0]
    target = outward * np.array([0.5, 0.5, 0.5, -1.5]) * math.pi
    phases, *_ = np.linalg.lstsq(d2.T, target, rcond=None)
    index_of = {tuple(cell): position for position, cell
                in enumerate(complex_.kSimplexVertices(1))}
    for edge in spacetime.getEdgeList().toVector():
        source = int(edge.getSource().getId())
        target_id = int(edge.getTarget().getId())
        phase = phases[index_of[tuple(sorted((source, target_id)))]]
        edge.setPhase(complex(phase if source < target_id else -phase))
    return spacetime


class TheMatchingToTheWilsonFormTest(unittest.TestCase):
    """beta_V = beta / <m^2>_beta makes the trivial-holonomy expansion
    beta L_1^up exactly."""

    def test_the_matched_weight_is_beta_over_the_second_moment(self):
        for beta in (0.5, 1.0, 1.3, 2.0, 5.0):
            character = cob.VillainCharacter(beta)
            self.assertAlmostEqual(character.second_moment,
                                   _second_moment(beta), places=13)
            self.assertAlmostEqual(character.matched_weight,
                                   beta / _second_moment(beta), places=13)
            # the curvature in the real angle at F = 1 is beta, the Wilson one
            self.assertAlmostEqual(-character.second_derivative(1.0).real,
                                   beta, places=12)

    def test_the_trivial_holonomy_hessian_is_the_wilson_one_entry_by_entry(self):
        beta = 1.3
        spacetime = tetrahedron()
        villain = np.array(cob.JointAction(
            spacetime, _declaration(beta=beta)).holonomy_hessian()).reshape(6, 6)
        wilson = np.array(cob.JointAction(
            spacetime, _declaration(cob.HolonomyForm.Wilson,
                                    beta=beta)).holonomy_hessian()).reshape(6, 6)
        expected = -beta * _signed_up_laplacian(spacetime)
        self.assertLess(np.max(np.abs(wilson - expected)), 1e-13)
        self.assertLess(np.max(np.abs(villain - expected)), 1e-12)
        eigenvalues = np.sort(np.linalg.eigvalsh(-villain.real))
        self.assertLess(np.max(np.abs(eigenvalues[:3])), 1e-12)
        for value in eigenvalues[3:]:
            self.assertAlmostEqual(value, 4.0 * beta, places=11)

    def test_the_relaxation_jacobian_reads_the_same_block(self):
        """HolomorphicRelaxation's Cauchy-contour Jacobian of the link
        equations is the analytic Hessian, at a complex connection."""
        spacetime = sphere3(phase=_flux)
        action = cob.JointAction(spacetime, _declaration())
        relaxation_declaration = cob.HolomorphicRelaxationDeclaration()
        relaxation_declaration.relax_lengths = False
        relaxation_declaration.relax_links = True
        relaxation_declaration.relax_multipliers = False
        relaxation = cob.HolomorphicRelaxation(action, relaxation_declaration)
        order = relaxation.variable_count()
        jacobian = np.array(relaxation.jacobian()).reshape(order, order)
        hessian = np.array(action.holonomy_hessian()).reshape(order, order)
        self.assertLess(np.max(np.abs(jacobian - hessian)),
                        1e-9 * (1.0 + np.max(np.abs(hessian))))


class TheQuarterTurnStiffnessTest(unittest.TestCase):
    """At F = +-i the Wilson stiffness is zero and the Villain one is not."""

    def test_the_per_face_curvature_is_the_closed_form(self):
        for beta in (0.5, 1.0, 1.3, 2.0, 5.0):
            character = cob.VillainCharacter(beta)
            expected = _quarter_turn_stiffness(beta)
            for holonomy in (1j, -1j):
                got = -character.second_derivative(holonomy)
                self.assertAlmostEqual(got.real, expected, places=11)
                self.assertAlmostEqual(got.imag, 0.0, places=11)
            self.assertGreater(expected, 0.4 * beta)
        # the quoted values
        self.assertAlmostEqual(_quarter_turn_stiffness(0.5), 0.43091, places=5)
        self.assertAlmostEqual(_quarter_turn_stiffness(1.0), 0.99796, places=5)

    def test_the_monopole_tetrahedron_is_stiff_under_villain_and_not_wilson(self):
        beta = 1.0
        spacetime = _quarter_turn_tetrahedron()
        faces = np.array(cob.JointAction(
            spacetime, _declaration(beta=beta)).face_holonomies())
        self.assertLess(np.max(np.abs(faces ** 2 + 1.0)), 1e-12)

        wilson = np.array(cob.JointAction(
            spacetime, _declaration(cob.HolonomyForm.Wilson,
                                    beta=beta)).holonomy_hessian())
        self.assertLess(np.max(np.abs(wilson)), 1e-12)

        villain = np.array(cob.JointAction(
            spacetime, _declaration(beta=beta)).holonomy_hessian()).reshape(6, 6)
        kappa = _quarter_turn_stiffness(beta)
        expected = -kappa * _signed_up_laplacian(spacetime)
        self.assertLess(np.max(np.abs(villain - expected)), 1e-11)
        eigenvalues = np.sort(np.linalg.eigvalsh(-villain.real))
        self.assertLess(np.max(np.abs(eigenvalues[:3])), 1e-11)
        for value in eigenvalues[3:]:
            self.assertAlmostEqual(value, 4.0 * kappa, places=10)


class TheTermIsGaugeInvariantAndEvenTest(unittest.TestCase):

    def test_every_complex_gauge_transformation_leaves_it_unchanged(self):
        spacetime = sphere3(phase=_flux)
        before = cob.JointAction(spacetime, _declaration())
        value = before.holonomy_term()
        current = np.array(before.link_stationarity())
        hessian = np.array(before.holonomy_hessian())
        _gauge(spacetime, _chi(spacetime, seed=3))
        after = cob.JointAction(spacetime, _declaration())
        self.assertLess(abs(after.holonomy_term() - value), 1e-12)
        self.assertLess(np.max(np.abs(np.array(after.link_stationarity())
                                      - current)), 1e-12)
        self.assertLess(np.max(np.abs(np.array(after.holonomy_hessian())
                                      - hessian)), 1e-12)

    def test_the_ward_current_is_conserved(self):
        spacetime = sphere3(phase=_flux)
        action = cob.JointAction(spacetime, _declaration())
        scale = np.max(np.abs(action.canonical_ward_current()))
        self.assertGreater(scale, 1e-3)
        self.assertLess(np.max(np.abs(action.ward_current_divergence())),
                        1e-12 * (1.0 + scale))

    def test_the_potential_is_even_under_inversion(self):
        character = cob.VillainCharacter(1.3)
        for holonomy in (0.8 + 0.3j, 1.4 - 0.9j, -0.7 + 0.2j, 1j):
            inverse = 1.0 / holonomy
            self.assertLess(abs(character.potential(holonomy)
                                - character.potential(inverse)), 1e-12)
            self.assertLess(abs(character.first_derivative(holonomy)
                                + character.first_derivative(inverse)), 1e-12)
            self.assertLess(abs(character.second_derivative(holonomy)
                                - character.second_derivative(inverse)), 1e-12)

    def test_reversing_the_connection_leaves_the_term_unchanged(self):
        spacetime = sphere3(phase=_flux)
        value = cob.JointAction(spacetime, _declaration()).holonomy_term()
        for edge in spacetime.getEdgeList().toVector():
            edge.setPhase(-edge.getPhase())
        reversed_value = cob.JointAction(spacetime,
                                         _declaration()).holonomy_term()
        self.assertLess(abs(reversed_value - value), 1e-12)

    def test_trivial_holonomy_is_stationary(self):
        spacetime = sphere3()
        action = cob.JointAction(spacetime, _declaration())
        self.assertLess(np.max(np.abs(action.link_stationarity())), 1e-15)
        # the value is -beta_V times the face count times log W(1)
        character = cob.VillainCharacter(1.3)
        expected = -character.matched_weight * 10 * math.log(
            character.series(1.0).value.real)
        self.assertAlmostEqual(abs(action.holonomy_term() - expected), 0.0,
                               places=12)


class TheDerivativesAgainstFiniteDifferencesTest(unittest.TestCase):
    """At complex face holonomies, in two complex directions each."""

    STEP = 1e-5

    def _faces_are_complex(self, action):
        faces = np.array(action.face_holonomies())
        self.assertGreater(np.max(np.abs(np.abs(faces) - 1.0)), 0.05)

    def test_the_link_stationarity_is_the_derivative_of_the_value(self):
        spacetime = sphere3(phase=_flux)
        action = cob.JointAction(spacetime, _declaration())
        self._faces_are_complex(action)
        analytic = np.array(action.link_stationarity())
        for index in range(len(analytic)):
            for direction in (1.0, 1j):
                h = self.STEP * direction
                _multiply_link(spacetime, index, h)
                plus = cob.JointAction(spacetime, _declaration()).holonomy_term()
                _multiply_link(spacetime, index, -2.0 * h)
                minus = cob.JointAction(spacetime,
                                        _declaration()).holonomy_term()
                _multiply_link(spacetime, index, h)
                difference = (plus - minus) / (2.0 * h)
                self.assertLess(abs(difference - analytic[index]),
                                1e-8 * (1.0 + abs(analytic[index])))

    def test_the_hessian_is_the_derivative_of_the_link_stationarity(self):
        spacetime = sphere3(phase=_flux)
        action = cob.JointAction(spacetime, _declaration())
        order = action.edge_count()
        hessian = np.array(action.holonomy_hessian()).reshape(order, order)
        for index in range(order):
            for direction in (1.0, 1j):
                h = self.STEP * direction
                _multiply_link(spacetime, index, h)
                plus = np.array(cob.JointAction(
                    spacetime, _declaration()).link_stationarity())
                _multiply_link(spacetime, index, -2.0 * h)
                minus = np.array(cob.JointAction(
                    spacetime, _declaration()).link_stationarity())
                _multiply_link(spacetime, index, h)
                column = (plus - minus) / (2.0 * h)
                self.assertLess(np.max(np.abs(column - hessian[:, index])),
                                1e-8 * (1.0 + np.max(np.abs(hessian))))


class TheLogarithmBranchTest(unittest.TestCase):

    def test_the_logarithm_is_real_on_the_unit_circle_and_inverts_w(self):
        character = cob.VillainCharacter(0.8)
        for angle in np.linspace(-math.pi, math.pi, 13):
            # exp(i angle) has modulus one to rounding, so the radial leg of
            # the continuation is of rounding length
            value = character.logarithm(cmath.exp(1j * angle))
            self.assertLess(abs(value.imag), 1e-14)
        for holonomy in (0.5 + 0.1j, 2.5 - 1.0j, -1.1 + 0.05j, 0.3j, -0.2 - 1e-3j):
            logarithm = character.logarithm(holonomy)
            series = character.series(holonomy).value
            self.assertLess(abs(cmath.exp(logarithm) - series),
                            1e-12 * abs(series))

    def test_the_logarithm_is_continuous_across_the_negative_axis_inside(self):
        """Inside the zero-free annulus q < |F| < 1/q the branch has no cut."""
        character = cob.VillainCharacter(0.8)
        above = character.logarithm(-1.2 + 1e-9j)
        below = character.logarithm(-1.2 - 1e-9j)
        self.assertLess(abs(above - below), 1e-7)

    def test_a_zero_of_w_is_refused(self):
        beta = 0.8
        q = math.exp(-1.0 / (2.0 * beta))
        character = cob.VillainCharacter(beta)
        with self.assertRaises(ValueError):
            character.logarithm(complex(-1.0 / q, 0.0))
        with self.assertRaises(ValueError):
            character.first_derivative(complex(-q, 0.0))


class TheZeroGuardTest(unittest.TestCase):
    """Newton steps are kept a declared distance away from the zeros of W."""

    def test_the_zero_distance_is_relative_to_the_nearest_zero(self):
        beta = 0.8
        q = math.exp(-1.0 / (2.0 * beta))
        character = cob.VillainCharacter(beta)
        for zero in (-q, -1.0 / q, -q ** 3, -q ** -3):
            self.assertLess(character.zero_distance(zero), 1e-12)
            self.assertAlmostEqual(character.zero_distance(zero * 1.01), 0.01,
                                   places=12)
        # F = 1 is equally far from -q, (1 + q) / q, and from -1/q, 1 + 1/q
        self.assertAlmostEqual(character.zero_distance(1.0), 1.0 + 1.0 / q,
                               places=12)

    def test_the_path_clearance_sees_a_zero_passed_between_the_end_points(self):
        """A face swept from F to F e^{Delta} through a zero reads a clearance
        of the order of the node spacing, though both end points are far."""
        beta = 0.8
        q = math.exp(-1.0 / (2.0 * beta))
        spacetime = _quarter_turn_tetrahedron()
        action = cob.JointAction(spacetime, _declaration(beta=beta))
        self.assertGreater(action.holonomy_zero_distance(), 0.3)
        # U on the first edge scaled by exp(t delta) moves each face through
        # it by exp(+-t delta). With delta = 2 (log(1/q) + i pi/2), halfway
        # along the path a face F = i with incidence +1 reaches -1/q under
        # +delta, and one with incidence -1 reaches it under -delta (F = -i
        # likewise reaches -q), so one of the two signs passes a zero
        delta = 2.0 * (-math.log(q) + 1j * math.pi / 2)
        clearance = min(
            action.holonomy_zero_clearance([sign * delta] + [0j] * 5, 0.01)
            for sign in (1.0, -1.0))
        self.assertLess(clearance, 0.02)
        self.assertGreater(action.holonomy_zero_clearance([0j] * 6, 0.01),
                           0.3)

    def test_the_guard_damps_every_step_that_would_come_too_close(self):
        """A margin no face can keep damps every trial step, and the report
        counts the damped iterations; the equations are untouched."""
        spacetime = sphere3(phase=_flux)
        action = cob.JointAction(spacetime, _declaration())
        before = np.array(action.link_stationarity())
        relaxation_declaration = cob.HolomorphicRelaxationDeclaration()
        relaxation_declaration.relax_lengths = False
        relaxation_declaration.relax_links = True
        relaxation_declaration.relax_multipliers = False
        relaxation_declaration.holonomy_zero_margin = 1e6
        report = cob.HolomorphicRelaxation(
            action, relaxation_declaration).solve()
        self.assertFalse(report.converged)
        self.assertEqual(report.zero_guard_damped_steps, 1)
        self.assertEqual(report.steps[0].zero_guard_dampings,
                         relaxation_declaration.maximum_dampings + 1)
        after = np.array(cob.JointAction(spacetime,
                                         _declaration()).link_stationarity())
        self.assertLess(np.max(np.abs(after - before)), 1e-15)

    def test_the_default_margin_leaves_a_clear_solve_alone(self):
        spacetime = sphere3(phase=_flux)
        action = cob.JointAction(spacetime, _declaration())
        relaxation_declaration = cob.HolomorphicRelaxationDeclaration()
        relaxation_declaration.relax_lengths = False
        relaxation_declaration.relax_links = True
        relaxation_declaration.relax_multipliers = False
        relaxation_declaration.tolerance = 1e-11
        report = cob.HolomorphicRelaxation(
            action, relaxation_declaration).solve()
        self.assertTrue(report.converged)
        self.assertEqual(report.zero_guard_damped_steps, 0)
        self.assertGreater(report.steps[0].holonomy_zero_distance, 0.05)

    def test_the_wilson_form_has_no_zero_to_guard(self):
        spacetime = sphere3(phase=_flux)
        action = cob.JointAction(spacetime,
                                 _declaration(cob.HolonomyForm.Wilson))
        self.assertEqual(action.holonomy_zero_distance(), math.inf)


class TheTruncationIsDeclaredAndBoundedTest(unittest.TestCase):

    def test_the_declared_term_count_follows_the_tolerance(self):
        for beta, tolerance in ((0.5, 1e-18), (1.3, 1e-18), (5.0, 1e-18),
                                (1.3, 1e-4)):
            character = cob.VillainCharacter(beta, tolerance)
            count = character.declared_term_count
            self.assertLess(math.exp(-count * count / (2 * beta)), tolerance)
            self.assertGreaterEqual(
                math.exp(-(count - 1) ** 2 / (2 * beta)), tolerance)

    def test_the_tail_bound_holds_and_is_not_loose(self):
        beta = 1.3
        character = cob.VillainCharacter(beta, 1e-4)
        # generic holonomies: at F = 1 the omitted part of DW and at F = +-i
        # the odd terms of W cancel exactly, so the omitted part can be far
        # below its bound there
        for holonomy in (1.1 + 0.05j, 0.7 + 0.4j, 2.0 - 1.5j, 0.2 + 0.1j):
            series = character.series(holonomy)
            m = np.arange(-400, 401)
            weights = np.exp(-m * m / (2.0 * beta))
            powers = np.array([holonomy ** int(k) if abs(k) <= 60 else 0.0
                               for k in m], dtype=complex)
            full = [np.sum(weights * powers),
                    np.sum(m * weights * powers),
                    np.sum(m * m * weights * powers)]
            got = [series.value, series.first, series.second]
            bounds = [series.value_tail, series.first_tail, series.second_tail]
            for exact, truncated, bound in zip(full, got, bounds):
                omitted = abs(exact - truncated)
                self.assertLessEqual(omitted, bound * (1.0 + 1e-9) + 1e-14)
                self.assertGreater(omitted, bound / 10.0)

    def test_far_from_the_unit_circle_the_count_follows_the_terms(self):
        """At |F| far from one the terms exp(-m^2/(2 beta)) r^m peak near
        m = beta log r, so the series is carried past them until its own tail
        is below the tolerance; the value then matches a long direct sum."""
        beta = 0.5
        character = cob.VillainCharacter(beta)
        for holonomy in (-2.2e-6 - 4.3e-7j, 293.8 - 124.8j, 1e-12 + 1e-12j):
            series = character.series(holonomy)
            self.assertLessEqual(series.value_tail,
                                 1e-18 * series.magnitude)
            m = np.arange(-80, 81)
            logs = -m * m / (2.0 * beta) + m * np.log(complex(holonomy))
            direct = np.sum(np.exp(logs))
            self.assertLess(abs(direct - series.value),
                            1e-13 * series.magnitude)

    def test_the_action_reports_its_truncation(self):
        spacetime = sphere3(phase=_flux)
        read = cob.JointAction(spacetime, _declaration()).holonomy_truncation()
        self.assertEqual(read.form, cob.HolonomyForm.Villain)
        self.assertEqual(read.tolerance, 1e-18)
        self.assertGreaterEqual(read.maximum_term_count,
                                read.declared_term_count)
        self.assertLess(read.relative_value_tail, 1e-15)
        self.assertLess(read.relative_second_tail, 1e-14)
        wilson = cob.JointAction(spacetime, _declaration(
            cob.HolonomyForm.Wilson)).holonomy_truncation()
        self.assertEqual(wilson.maximum_term_count, 0)

    def test_the_declaration_is_validated(self):
        spacetime = sphere3()
        with self.assertRaises(ValueError):
            cob.JointAction(spacetime, _declaration(beta=-1.0))
        with self.assertRaises(ValueError):
            cob.JointAction(spacetime, _declaration(villain_tolerance=0.0))
        with self.assertRaises(ValueError):
            cob.VillainCharacter(0.0)


if __name__ == "__main__":
    unittest.main()
