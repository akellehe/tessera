# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""#1188 — relaxation by the holomorphic stationarity equations of the action.

The whitepaper asks for the complex stationarity equations of S(z, U, Gamma),
never for the minimization of a selected real projection of it, and it asks for
the connection to relax through face holonomies in a multiplicative variable.
Four things carry that here and each is asserted rather than argued.

THE ACTION IS HOLOMORPHIC AND ITS EQUATIONS ARE COMPLEX. ``value`` is the sum of
four complex terms and ``stationarity_residual`` is a vector of complex
equations. The solve drives every component of it to zero, real part and
imaginary part together, and the tests below start from points where both parts
are large so that driving only one of them would be visible.

THE FACE-HOLONOMY TERM IS BRANCH-FREE AND GAUGE INVARIANT. F_tau is an ordered
product of links and their inverses, so it asks for no argument and no
logarithm; it is unchanged by every complex gauge transformation, including the
non-compact ones that a U(1) reading would not survive.

ITS SECOND VARIATION IS THE UP-LAPLACIAN. On a tetrahedron with trivial holonomy
the connection block of the Jacobian is minus beta times ``d_2 d_2^T`` entry by
entry, whose nonzero eigenvalues are 4 beta on the coexact block and zero on the
pure-gauge one — the numbers the whitepaper quotes when it takes the bare
connection stiffness in Wilson-plaquette form. The sign is not a discrepancy:
the Maurer-Cartan coordinate is ``delta = i theta`` for a real angle theta, so
the delta-Hessian is minus the theta-Hessian.

THE MULTIPLIERS IMPOSE THE MOMENT EQUATION EXACTLY. With a target declared, the
solve returns p_j(h) equal to p_j* to rounding and the multiplier at the value
the stationarity equations force, and it is the equation that is solved, not a
residual norm that is made small.
"""

import cmath
import os
import sys
import unittest

import numpy as np

import tessera as T

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _joint_action_hosts import sphere3, tetrahedron  # noqa: E402

cob = T.cobordism


def _declaration(**overrides):
    """A joint-action declaration with the named fields overridden."""
    declaration = cob.JointActionDeclaration()
    declaration.carrier_degree = 1
    declaration.gravitational_weight = 0.0
    # These suites exercise the dual (Sorkin) Regge form, declared explicitly
    # now that the primal form is the default.
    declaration.regge_form = cob.ReggeForm.Dual
    declaration.holonomy_weight = 0.0
    # These suites assert the Wilson plaquette form's values (a zero term at
    # trivial holonomy, the cosine current), so they declare it.
    declaration.holonomy_form = cob.HolonomyForm.Wilson
    declaration.matter_weight = 0.0
    declaration.metric_source = cob.HodgeMetricSource.WhitneyPencil
    for name, value in overrides.items():
        setattr(declaration, name, value)
    return declaration


def _relaxation(**overrides):
    """A relaxation declaration with the named fields overridden."""
    declaration = cob.HolomorphicRelaxationDeclaration()
    declaration.relax_lengths = False
    declaration.relax_links = False
    declaration.relax_multipliers = False
    declaration.maximum_iterations = 40
    declaration.tolerance = 1e-11
    for name, value in overrides.items():
        setattr(declaration, name, value)
    return declaration


def _gauge(spacetime, chi):
    """Apply ``U_xy -> g_x^-1 U_xy g_y`` with ``g = exp(i chi)``.

    On the stored phase that is ``phi -> phi + chi_target - chi_source``, since
    the stored orientation carries ``exp(i phi)`` and the reverse its inverse.
    ``chi`` is complex, so this is the full C* gauge group and not only its
    compact U(1) factor.
    """
    for edge in spacetime.getEdgeList().toVector():
        source = int(edge.getSource().getId())
        target = int(edge.getTarget().getId())
        edge.setPhase(edge.getPhase() + chi[target] - chi[source])


def _chi(spacetime, seed=0):
    """A complex vertex function, the parameter of a C* gauge transformation."""
    values = {}
    for index, vertex in enumerate(spacetime.getVertexList().toVector()):
        step = index + seed
        values[int(vertex.getId())] = complex(0.31 * ((step % 7) - 3),
                                              0.17 * ((step % 4) - 1.5))
    return values


def _flux(index):
    """A connection with nonzero flux in both components of the phase.

    The values come from no vertex function, so some closed walk necessarily
    carries a nontrivial holonomy and the connection is not a pure gauge.
    """
    return complex(0.23 * ((index % 5) - 2), 0.09 * ((index % 3) - 1))


def _metric(index):
    """A mild deterministic non-uniform complex metric."""
    return complex(1.0 + 0.037 * (index % 5), 0.021 * (1 + index % 4))


class TheActionIsTheSumOfItsDeclaredTermsTest(unittest.TestCase):
    """``value`` reads nothing but the five terms it names."""

    def test_the_value_is_the_sum_of_the_five_terms(self):
        spacetime = sphere3(squared=_metric, phase=_flux)
        declaration = _declaration(gravitational_weight=0.6,
                                   holonomy_weight=1.3)
        action = cob.JointAction(spacetime, declaration)
        total = (action.regge_term() + action.stiffness_term()
                 + action.holonomy_term() + action.matter_term()
                 + action.spectral_term())
        self.assertAlmostEqual(abs(action.value() - total), 0.0, places=12)
        self.assertEqual(cob.JointAction.term_names(),
                         ["regge", "stiffness", "holonomy", "matter",
                          "spectral"])

    def test_the_action_is_genuinely_complex(self):
        """A complex metric gives a complex action, not a real one.

        The Regge term of a complex geometry has a nonzero imaginary part, so
        making it stationary is not the same problem as making any real
        projection of it stationary.
        """
        spacetime = sphere3(squared=_metric, phase=_flux)
        action = cob.JointAction(spacetime,
                                 _declaration(gravitational_weight=1.0))
        self.assertGreater(abs(action.value().imag), 1e-6)

    def test_the_matter_term_is_zero_without_a_carried_state(self):
        spacetime = sphere3(squared=_metric)
        action = cob.JointAction(spacetime, _declaration(matter_weight=5.0))
        self.assertEqual(action.matter_term(), 0j)


class TheFaceHolonomyIsBranchFreeTest(unittest.TestCase):
    """F_tau is an ordered product of links over the boundary incidences."""

    def _expected_holonomies(self, spacetime):
        """F_tau from the stored phases, assembled independently here."""
        complex_ = cob.ChainComplex.fromSpacetime(spacetime)
        canonical = complex_.kSimplexVertices(1)
        links = {}
        for edge in spacetime.getEdgeList().toVector():
            source = int(edge.getSource().getId())
            target = int(edge.getTarget().getId())
            link = cmath.exp(1j * complex(edge.getPhase()))
            if source > target:
                link = 1.0 / link
            links[tuple(sorted((source, target)))] = link
        boundary = np.array(complex_.boundaryMatrix(2), dtype=float).reshape(
            len(canonical), complex_.numSimplices(2))
        holonomies = []
        for column in range(complex_.numSimplices(2)):
            product = 1.0 + 0j
            for row, cell in enumerate(canonical):
                exponent = int(round(boundary[row, column]))
                if exponent == 0:
                    continue
                link = links[tuple(cell)]
                product *= link if exponent > 0 else 1.0 / link
            holonomies.append(product)
        return holonomies

    def test_the_holonomies_are_the_product_over_the_boundary_incidences(self):
        spacetime = sphere3(squared=_metric, phase=_flux)
        action = cob.JointAction(spacetime, _declaration(holonomy_weight=1.0))
        reported = action.face_holonomies()
        expected = self._expected_holonomies(spacetime)
        self.assertEqual(len(reported), len(expected))
        for got, want in zip(reported, expected):
            self.assertAlmostEqual(abs(got - want), 0.0, places=12)

    def test_the_holonomy_term_survives_every_complex_gauge_transformation(self):
        """Gauge invariance holds for C*, not only for its compact factor."""
        spacetime = sphere3(squared=_metric, phase=_flux)
        action = cob.JointAction(spacetime, _declaration(holonomy_weight=1.7))
        before_term = action.holonomy_term()
        before_holonomies = action.face_holonomies()
        _gauge(spacetime, _chi(spacetime, seed=3))
        after = cob.JointAction(spacetime, _declaration(holonomy_weight=1.7))
        self.assertAlmostEqual(abs(after.holonomy_term() - before_term), 0.0,
                               places=11)
        for got, want in zip(after.face_holonomies(), before_holonomies):
            self.assertAlmostEqual(abs(got - want), 0.0, places=11)

    def test_a_trivial_connection_is_stationary_for_the_holonomy_term(self):
        spacetime = sphere3(squared=_metric)
        action = cob.JointAction(spacetime, _declaration(holonomy_weight=2.9))
        self.assertAlmostEqual(abs(action.holonomy_term()), 0.0, places=13)
        for component in action.link_stationarity():
            self.assertAlmostEqual(abs(component), 0.0, places=12)


class TheConnectionStiffnessIsTheUpLaplacianTest(unittest.TestCase):
    """The second variation of the holonomy term, on the tetrahedron."""

    @staticmethod
    def _canonical_index_and_sign(spacetime, complex_):
        """Each mesh edge's canonical cell index and its stored sign.

        Every per-edge vector of the action is in the mesh's own edge order and
        carries the equation on the edge's STORED source-to-target orientation,
        while the boundary map's incidences are in the canonical degree-one cell
        order and on the ascending-vertex-id orientation. Both the permutation
        and the sign that relate the two are read here, so the comparison below
        is against the same matrix in the same basis.
        """
        index_of = {tuple(cell): position for position, cell
                    in enumerate(complex_.kSimplexVertices(1))}
        indices = []
        signs = []
        for edge in spacetime.getEdgeList().toVector():
            source = int(edge.getSource().getId())
            target = int(edge.getTarget().getId())
            indices.append(index_of[tuple(sorted((source, target)))])
            signs.append(1.0 if source < target else -1.0)
        return indices, signs

    def test_the_connection_block_is_minus_beta_times_the_up_laplacian(self):
        beta = 1.3
        spacetime = tetrahedron()
        action = cob.JointAction(spacetime, _declaration(holonomy_weight=beta))
        relaxation = cob.HolomorphicRelaxation(action,
                                               _relaxation(relax_links=True))
        order = relaxation.variable_count()
        self.assertEqual(order, 6)
        jacobian = np.array(relaxation.jacobian()).reshape(order, order)

        complex_ = cob.ChainComplex.fromSpacetime(spacetime)
        boundary = np.array(complex_.boundaryMatrix(2), dtype=float).reshape(
            complex_.numSimplices(1), complex_.numSimplices(2))
        up_laplacian = boundary @ boundary.T

        indices, signs = self._canonical_index_and_sign(spacetime, complex_)
        expected = np.array(
            [[-beta * signs[row] * signs[column]
              * up_laplacian[indices[row], indices[column]]
              for column in range(order)] for row in range(order)])
        self.assertLess(np.max(np.abs(jacobian - expected)), 1e-8)

        eigenvalues = np.sort(np.linalg.eigvalsh(-jacobian.real))
        self.assertLess(abs(eigenvalues[0]), 1e-8)
        self.assertLess(abs(eigenvalues[1]), 1e-8)
        self.assertLess(abs(eigenvalues[2]), 1e-8)
        for value in eigenvalues[3:]:
            self.assertAlmostEqual(value, 4.0 * beta, places=7)


class TheWardCurrentTest(unittest.TestCase):
    """The link equation read as the current of Section 13.4."""

    def test_the_ward_current_is_the_link_stationarity(self):
        spacetime = sphere3(squared=_metric, phase=_flux)
        action = cob.JointAction(spacetime,
                                 _declaration(holonomy_weight=0.8,
                                              gravitational_weight=0.4))
        for current, stationarity in zip(action.ward_current(),
                                         action.link_stationarity()):
            self.assertEqual(current, stationarity)

    def test_the_gauge_invariant_terms_give_a_conserved_current(self):
        """``d j = 0`` is the discrete Ward identity, and it is exact here."""
        spacetime = sphere3(squared=_metric, phase=_flux)
        action = cob.JointAction(spacetime,
                                 _declaration(holonomy_weight=1.1,
                                              gravitational_weight=0.5))
        scale = max(abs(component) for component in action.ward_current())
        self.assertGreater(scale, 1e-6)
        for divergence in action.ward_current_divergence():
            self.assertLess(abs(divergence), 1e-10 * (1.0 + scale))


class TheHolomorphicSolveReachesAStationaryPointTest(unittest.TestCase):
    """#1188's acceptance, on the connection."""

    def test_the_connection_relaxes_to_a_stationary_point_to_rounding(self):
        spacetime = sphere3(squared=_metric, phase=_flux)
        action = cob.JointAction(spacetime, _declaration(holonomy_weight=1.0))

        start = action.link_stationarity()
        self.assertGreater(max(abs(value.real) for value in start), 1e-3)
        self.assertGreater(max(abs(value.imag) for value in start), 1e-3)

        relaxation = cob.HolomorphicRelaxation(action,
                                               _relaxation(relax_links=True))
        report = relaxation.solve()
        self.assertTrue(report.converged, report.residual_norm)
        self.assertLess(report.residual_norm, 1e-11)

        for component in relaxation.action.stationarity_residual():
            self.assertLess(abs(component.real), 1e-11)
            self.assertLess(abs(component.imag), 1e-11)

    def test_the_connection_moves_multiplicatively_and_the_geometry_does_not(
            self):
        """The relaxation writes the connection and leaves the lengths alone."""
        spacetime = sphere3(squared=_metric, phase=_flux)
        before = [complex(edge.getLength())
                  for edge in spacetime.getEdgeList().toVector()]
        action = cob.JointAction(spacetime, _declaration(holonomy_weight=1.0))
        cob.HolomorphicRelaxation(action, _relaxation(relax_links=True)).solve()
        after = [complex(edge.getLength())
                 for edge in spacetime.getEdgeList().toVector()]
        self.assertEqual(before, after)
        moved = cob.JointAction(spacetime, _declaration(holonomy_weight=1.0))
        for holonomy in moved.face_holonomies():
            self.assertAlmostEqual(abs(holonomy - 1.0), 0.0, places=9)

    def test_the_jacobian_is_rank_deficient_by_the_gauge_directions(self):
        """Gauge invariance is visible in the rank the solve reports.

        A connected complex on ``V`` vertices has ``V - 1`` independent gauge
        directions, and the action is constant along every one of them, so the
        connection block of the Jacobian is singular on exactly that many.
        """
        spacetime = sphere3(squared=_metric, phase=_flux)
        action = cob.JointAction(spacetime, _declaration(holonomy_weight=1.0))
        relaxation = cob.HolomorphicRelaxation(action,
                                               _relaxation(relax_links=True))
        report = relaxation.solve()
        vertices = len(spacetime.getVertexList().toVector())
        self.assertGreater(len(report.steps), 0)
        self.assertEqual(report.steps[0].jacobian_rank,
                         relaxation.variable_count() - (vertices - 1))

    def test_the_present_objective_is_a_real_residual_at_the_same_point(self):
        """The contrast the ticket draws, stated as two measurements.

        The injected objective's value is a non-negative real number: every one
        of its terms is the squared norm of the gradient of a real functional,
        so what it reports is how nearly two real functionals are stationary. The
        joint action's residual is a vector of complex equations, and at the
        point the solve reaches every one of them vanishes.
        """
        spacetime = sphere3(squared=_metric, phase=_flux)
        node = cob.MultiCobordism(spacetime, [], [], degrees=[1], gamma=0.0,
                                  seed=11)
        node.set_objective(cob.JointStationarityObjective())
        node.set_regge_weight(1.0)
        objective = node.objective()
        self.assertIsInstance(objective, float)
        self.assertGreaterEqual(objective, 0.0)

        action = cob.JointAction(spacetime, _declaration(holonomy_weight=1.0))
        relaxation = cob.HolomorphicRelaxation(action,
                                               _relaxation(relax_links=True))
        report = relaxation.solve()
        self.assertTrue(report.converged)
        self.assertIsInstance(report.action, complex)


class TheMultipliersImposeTheMomentEquationTest(unittest.TestCase):
    """#1188's acceptance, on the holomorphic spectral constraints."""

    def _constrained(self, spacetime, target):
        declaration = _declaration(matter_weight=1.0)
        order = cob.ChainComplex.fromSpacetime(spacetime).numSimplices(1)
        declaration.covariance = [complex(value) for value
                                  in np.eye(order, dtype=complex).reshape(-1)]
        declaration.moment_constraints = [
            cob.SpectralMomentConstraint(1, target, 0j)]
        return cob.JointAction(spacetime, declaration)

    def test_the_solve_returns_the_target_moment_exactly(self):
        """p_1(h) is driven to p_1*, and xi to the value the equations force.

        With Gamma the identity the matter term is itself p_1(h), so the length
        equations read ``(w_M + xi) d p_1/d z_e = 0`` and the stationary point
        has ``xi = -w_M`` with the moment equation left to fix the geometry.
        The multiplier's value is therefore known in closed form and is asserted
        beside the constraint it imposes.
        """
        reference = sphere3(squared=_metric)
        target = self._constrained(reference, 0j).power_sums()[0]

        spacetime = sphere3(squared=lambda index: _metric(index) * 1.08)
        action = self._constrained(spacetime, target)
        self.assertGreater(abs(action.moment_residuals()[0]), 1e-3)

        relaxation = cob.HolomorphicRelaxation(
            action, _relaxation(relax_lengths=True, relax_multipliers=True,
                                maximum_iterations=60, tolerance=1e-11))
        report = relaxation.solve()
        self.assertTrue(report.converged, report.residual_norm)
        self.assertLess(abs(report.moment_residuals[0]), 1e-10)
        self.assertAlmostEqual(abs(report.multipliers[0] + 1.0), 0.0, places=8)

    def test_the_moment_gradient_matches_the_contour_derivative(self):
        """The analytic column and the contour row agree, as they must.

        The action is ``... + xi_j (p_j(h) - p_j*)``, so the multiplier's
        Jacobian column is ``d p_j/d z`` analytically, while the moment
        equation's Jacobian row is the same derivative taken by the contour
        rule. Second derivatives commute, so the two are one number reached two
        ways, and their agreement certifies the contour rule at the precision
        the solve relies on.
        """
        spacetime = sphere3(squared=_metric)
        action = self._constrained(spacetime, 0j)
        relaxation = cob.HolomorphicRelaxation(
            action, _relaxation(relax_lengths=True, relax_multipliers=True))
        order = relaxation.variable_count()
        jacobian = np.array(relaxation.jacobian()).reshape(order, order)
        analytic = action.moment_gradient(0)
        edges = action.edge_count()
        for index in range(edges):
            self.assertAlmostEqual(
                abs(jacobian[edges, index] - analytic[index]), 0.0,
                delta=1e-7 * (1.0 + abs(analytic[index])))
            self.assertAlmostEqual(
                abs(jacobian[index, edges] - analytic[index]), 0.0,
                delta=1e-12 * (1.0 + abs(analytic[index])))


class TheDeclaredControlsAreCheckedTest(unittest.TestCase):
    """The configuration a caller may not silently get wrong."""

    def test_a_short_contour_is_refused(self):
        spacetime = sphere3()
        action = cob.JointAction(spacetime, _declaration(holonomy_weight=1.0))
        declaration = _relaxation(relax_links=True)
        declaration.contour_nodes = 4
        with self.assertRaises(ValueError):
            cob.HolomorphicRelaxation(action, declaration)

    def test_a_negative_carrier_degree_is_refused(self):
        spacetime = sphere3()
        with self.assertRaises(ValueError):
            cob.JointAction(spacetime, _declaration(carrier_degree=-1))

    def test_a_moment_order_below_one_is_refused(self):
        spacetime = sphere3()
        declaration = _declaration()
        declaration.moment_constraints = [
            cob.SpectralMomentConstraint(0, 0j, 0j)]
        with self.assertRaises(ValueError):
            cob.JointAction(spacetime, declaration)

    def test_a_covariance_of_the_wrong_order_is_refused(self):
        spacetime = sphere3()
        declaration = _declaration(matter_weight=1.0)
        declaration.covariance = [1 + 0j, 0j, 0j, 1 + 0j]
        with self.assertRaises(ValueError):
            cob.JointAction(spacetime, declaration)

    def test_no_relaxable_field_is_refused(self):
        spacetime = sphere3()
        action = cob.JointAction(spacetime, _declaration(holonomy_weight=1.0))
        with self.assertRaises(ValueError):
            cob.HolomorphicRelaxation(action, _relaxation())


if __name__ == "__main__":
    unittest.main()
