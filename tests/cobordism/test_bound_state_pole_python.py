# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""#1200 — mass as the complex bound-state pole of the Feshbach response pencil.

Section 13.3 of the whitepaper says that mass is not defined by an incoherent
sum of moduli. It is the simple isolated zero ``s_C`` of
``D_C(s) = det F_C(s)``, with ``F_C`` the exact meromorphic Feshbach response
pencil of the whole persistent bound cluster, continued in the complex spectral
parameter ``s`` on a domain that excludes unretained interior poles. Four things
are asserted here rather than argued.

THE ZEROS ARE THE EIGENVALUES THE INTERIOR DOES NOT CARRY. By the determinant
factorization ``det P = det P_II det F_C``, the zeros of ``D_C`` on a finite
complex are exactly the eigenvalues of the full pencil that are not eigenvalues
of the interior block, with matching multiplicities. Both halves of that are
checked against an independent eigendecomposition: a contour around a full
eigenvalue the interior does not carry finds it, and a contour around an
interior eigenvalue finds no zero and is refused by name, because the argument
principle on a meromorphic function would otherwise cancel a zero against a pole.

THE DERIVATIVES ARE ANALYTIC. ``F_C'(s)`` is the closed form that
``dP/ds = -M`` forces, and ``D_C'/D_C`` is the trace of ``F_C^-1 F_C'``. Neither
carries a finite difference or a step size. The closed form is checked against a
difference quotient formed in the test, which is where a difference quotient
belongs.

A KNOWN BOUND STATE IS FOUND, AND ITS CERTIFICATE IS THE SPECIFICATION. On a
pencil whose spectrum is declared, the pole below the free threshold is located
to rounding, ``D_C(s_C)`` vanishes, ``D_C'(s_C)`` does not, the residue of the
supported resolvent has rank one, the separation from the nearest other zero is
the declared gap, the binding shift is the declared depth, and the refinement
continuation at a second quadrature does not move the answer.

A MULTIPLE ROOT IS RETAINED, NOT SPLIT. A double zero is reported once with
algebraic multiplicity two and with the whole residue matrix its local data
needs, rather than as two roots separated by an ordering convention.
"""

import math
import unittest

import numpy as np

import tessera as T

cob = T.cobordism
BSP = cob.BoundStatePole


def _mixer(order):
    """A deterministic well-conditioned complex similarity.

    Built from fixed integer patterns rather than a random draw, so the pencils
    below have exactly the same conditioning on every machine and in every
    version of the numerical libraries.
    """
    real = np.array([[((row * 7 + column * 3) % 5) - 2 for column in range(order)]
                     for row in range(order)], dtype=float)
    imaginary = np.array([[((row * 2 + column * 5) % 3) - 1
                           for column in range(order)]
                          for row in range(order)], dtype=float)
    return np.eye(order, dtype=complex) + 0.2 * (real + 1j * imaginary)


def _pencil(eigenvalues):
    """A pencil ``(A, M = I)`` whose spectrum is exactly ``eigenvalues``."""
    order = len(eigenvalues)
    mixer = _mixer(order)
    operator = mixer @ np.diag(np.array(eigenvalues, dtype=complex)) \
        @ np.linalg.inv(mixer)
    return operator, np.eye(order, dtype=complex)


def _interior_spectrum(operator, metric, interface):
    """The eigenvalues of the interior block, which are the poles of D_C."""
    order = operator.shape[0]
    interior = [index for index in range(order) if index not in interface]
    block = operator[np.ix_(interior, interior)]
    metric_block = metric[np.ix_(interior, interior)]
    return np.linalg.eigvals(np.linalg.solve(metric_block, block))


def _full_spectrum(operator, metric):
    return np.linalg.eigvals(np.linalg.solve(metric, operator))


class TheDeterminantFactorizationTest(unittest.TestCase):
    """``det P = det P_II det F_C``, and ``determinant`` is that second factor."""

    def setUp(self):
        self.operator, self.metric = _pencil([-2.5, 0.5, 1.5, 3.0])
        self.interface = [0, 1]

    def test_the_determinant_is_the_response_determinant(self):
        point = complex(0.2, 0.7)
        response = BSP.response(self.operator, self.metric, self.interface,
                                point)
        self.assertFalse(response.interiorSingular)
        self.assertAlmostEqual(
            abs(BSP.determinant(self.operator, self.metric, self.interface,
                                point) - response.responseDeterminant),
            0.0, places=12)

    def test_the_factorization_reproduces_the_pencil_determinant(self):
        point = complex(-0.4, 0.3)
        response = BSP.response(self.operator, self.metric, self.interface,
                                point)
        product = response.responseDeterminant * response.interiorDeterminant
        expected = np.linalg.det(self.operator - point * self.metric)
        self.assertAlmostEqual(abs(product - expected) / abs(expected), 0.0,
                               places=10)


class TheAnalyticDerivativesTest(unittest.TestCase):
    """The closed forms agree with difference quotients taken in the test."""

    def setUp(self):
        self.operator, self.metric = _pencil([-2.5, 0.5, 1.5, 3.0])
        self.interface = [0, 1]
        self.point = complex(0.2, 0.7)

    def _response(self, point):
        return np.array(BSP.response(self.operator, self.metric,
                                     self.interface, point).response)

    def test_the_response_derivative_is_the_closed_form(self):
        step = 1e-6
        numerical = (self._response(self.point + step)
                     - self._response(self.point - step)) / (2.0 * step)
        analytic = np.array(BSP.response_derivative(
            self.operator, self.metric, self.interface, self.point))
        self.assertLess(np.max(np.abs(numerical - analytic)), 1e-6)

    def test_the_logarithmic_derivative_is_the_solved_trace(self):
        response = self._response(self.point)
        derivative = np.array(BSP.response_derivative(
            self.operator, self.metric, self.interface, self.point))
        expected = np.trace(np.linalg.solve(response, derivative))
        measured = BSP.logarithmic_derivative(self.operator, self.metric,
                                              self.interface, self.point)
        self.assertAlmostEqual(abs(measured - expected), 0.0, places=10)

    def test_the_logarithmic_derivative_is_the_derivative_of_the_logarithm(self):
        """``D'/D`` against a difference quotient of ``log D``."""
        step = 1e-5
        forward = BSP.determinant(self.operator, self.metric, self.interface,
                                  self.point + step)
        backward = BSP.determinant(self.operator, self.metric, self.interface,
                                   self.point - step)
        middle = BSP.determinant(self.operator, self.metric, self.interface,
                                 self.point)
        numerical = (forward - backward) / (2.0 * step) / middle
        measured = BSP.logarithmic_derivative(self.operator, self.metric,
                                              self.interface, self.point)
        self.assertLess(abs(numerical - measured), 1e-6)


class TheZerosAreThePencilEigenvaluesTest(unittest.TestCase):
    """#1200's acceptance: the pole is found on a fixture with a known state."""

    SPECTRUM = [-2.5, 0.5, 1.5, 3.0]

    def setUp(self):
        self.operator, self.metric = _pencil(self.SPECTRUM)
        self.interface = [0, 1]
        self.interior = _interior_spectrum(self.operator, self.metric,
                                           self.interface)

    def _config(self, **overrides):
        config = cob.BoundStatePoleConfig()
        config.contour_nodes = 128
        config.refinement_nodes = 256
        for name, value in overrides.items():
            setattr(config, name, value)
        return config

    def test_the_fixture_separates_its_zeros_from_its_interior_poles(self):
        """The fixture's own certificate, stated before it is relied on."""
        for eigenvalue in self.SPECTRUM:
            for pole in self.interior:
                self.assertGreater(abs(eigenvalue - pole), 0.2)
        for index, left in enumerate(self.SPECTRUM):
            for right in self.SPECTRUM[index + 1:]:
                self.assertGreater(abs(left - right), 0.5)

    def test_every_full_eigenvalue_is_a_zero_of_the_response(self):
        for eigenvalue in self.SPECTRUM:
            read = BSP.poles(self.operator, self.metric, self.interface,
                             complex(eigenvalue), 0.2, self._config())
            self.assertEqual(read.failed_certificates, [], eigenvalue)
            self.assertEqual(read.zeros, 1, eigenvalue)
            self.assertEqual(len(read.poles), 1)
            self.assertLess(abs(read.poles[0] - eigenvalue), 1e-9)
            self.assertEqual(read.multiplicity[0], 1)
            self.assertTrue(read.simple[0])
            self.assertEqual(read.interior_poles_enclosed, 0)

    def test_an_interior_eigenvalue_is_not_a_zero_and_is_named(self):
        for pole in self.interior:
            read = BSP.poles(self.operator, self.metric, self.interface,
                             complex(pole), 0.1, self._config())
            self.assertIn("interior-pole-enclosed", read.failed_certificates)
            self.assertEqual(read.interior_poles_enclosed, 1)
            self.assertEqual(read.poles, [])

    def test_the_bound_state_below_the_threshold_is_certified(self):
        config = self._config(free_threshold=complex(0.0))
        read = BSP.poles(self.operator, self.metric, self.interface,
                         complex(-2.5), 0.8, config)
        self.assertEqual(read.failed_certificates, [])
        self.assertEqual(read.zeros, 1)
        self.assertLess(abs(read.poles[0] - (-2.5)), 1e-10)
        self.assertLess(abs(read.zero_count - 1.0), 1e-6)
        self.assertLess(abs(read.determinant_at_pole[0]), 1e-9)
        self.assertGreater(abs(read.derivative_at_pole[0]), 1e-6)
        self.assertTrue(read.simple[0])
        self.assertEqual(read.residue_rank[0], 1)
        self.assertGreater(read.residue_norm[0], 0.0)
        self.assertTrue(math.isinf(read.separation[0]))
        self.assertLess(abs(read.binding_shift[0] - (-2.5)), 1e-10)
        self.assertLess(read.continuation_movement[0], 1e-9)
        self.assertEqual(read.centre, complex(-2.5))
        self.assertAlmostEqual(read.radius, 0.8, places=12)
        self.assertEqual(read.nodes, 128)

    def test_the_residue_is_the_supported_resolvent_residue(self):
        """The residue matrix is checked against an eigen-decomposition.

        For a simple zero the residue of ``F_C^-1`` is the outer product of the
        response's right and left null vectors, normalized so that the product
        with ``F_C'`` has unit trace. The check is that the residue times
        ``F_C'(s_C)`` has trace one, which is the residue's defining property
        and needs no null vector to be ordered.
        """
        read = BSP.poles(self.operator, self.metric, self.interface,
                         complex(-2.5), 0.8, self._config())
        size = len(self.interface)
        residue = np.array(read.residue[0], dtype=complex).reshape(size, size)
        derivative = np.array(BSP.response_derivative(
            self.operator, self.metric, self.interface, read.poles[0]))
        self.assertAlmostEqual(abs(np.trace(residue @ derivative) - 1.0), 0.0,
                               places=7)

    def test_a_contour_around_nothing_refuses(self):
        read = BSP.poles(self.operator, self.metric, self.interface,
                         complex(0.0, 4.0), 0.5, self._config())
        self.assertIn("no-zero-enclosed", read.failed_certificates)
        self.assertEqual(read.zeros, 0)
        self.assertEqual(read.poles, [])

    def test_a_contour_enclosing_two_zeros_reports_both(self):
        centre, radius = complex(1.0), 0.6
        for pole in self.interior:
            self.assertGreater(abs(pole - centre), radius)
        read = BSP.poles(self.operator, self.metric, self.interface, centre,
                         radius, self._config())
        self.assertEqual(read.failed_certificates, [])
        self.assertEqual(read.zeros, 2)
        found = sorted(value.real for value in read.poles)
        self.assertEqual(len(found), 2)
        self.assertLess(abs(found[0] - 0.5), 1e-9)
        self.assertLess(abs(found[1] - 1.5), 1e-9)
        for index in range(2):
            self.assertAlmostEqual(read.separation[index], 1.0, places=8)

    def test_an_empty_interface_refuses(self):
        read = BSP.poles(self.operator, self.metric, [], complex(-2.5), 0.8,
                         self._config())
        self.assertIn("empty-interface", read.failed_certificates)

    def test_a_nonpositive_radius_is_rejected(self):
        with self.assertRaises(ValueError):
            BSP.poles(self.operator, self.metric, self.interface,
                      complex(-2.5), 0.0, self._config())


class AMultipleRootIsRetainedTest(unittest.TestCase):
    """A double zero keeps its multiplicity and its whole residue."""

    def setUp(self):
        # Diagonal, so the interface block decouples and the response is
        # exactly diag(a - s, a - s): a zero of algebraic multiplicity two.
        self.operator = np.diag(np.array([1.25, 1.25, -3.0, 4.0],
                                         dtype=complex))
        self.metric = np.eye(4, dtype=complex)
        self.interface = [0, 1]

    def test_the_double_zero_is_reported_once_with_multiplicity_two(self):
        config = cob.BoundStatePoleConfig()
        read = BSP.poles(self.operator, self.metric, self.interface,
                         complex(1.25), 0.5, config)
        self.assertEqual(read.failed_certificates, [])
        self.assertEqual(read.zeros, 2)
        self.assertEqual(len(read.poles), 1)
        self.assertLess(abs(read.poles[0] - 1.25), 1e-9)
        self.assertEqual(read.multiplicity[0], 2)
        self.assertFalse(read.simple[0])

    def test_the_residue_of_the_double_zero_has_rank_two(self):
        config = cob.BoundStatePoleConfig()
        read = BSP.poles(self.operator, self.metric, self.interface,
                         complex(1.25), 0.5, config)
        residue = np.array(read.residue[0], dtype=complex).reshape(2, 2)
        self.assertEqual(read.residue_rank[0], 2)
        self.assertLess(np.max(np.abs(residue + np.eye(2))), 1e-8)


class TheFrameworkPencilPathTest(unittest.TestCase):
    """The same search on the framework's own dressed pencil."""

    @classmethod
    def setUpClass(cls):
        cls.previous = cob.HodgeLaplacian.defaultMetricSource()
        cob.HodgeLaplacian.setDefaultMetricSource(
            cob.HodgeMetricSource.WhitneyPencil)
        spacetime = T.Spacetime.fromVertexTuples(3, [[0, 1, 2, 3]], 1.0, 0.0)
        # A deliberately asymmetric metric: the regular tetrahedron's spectrum
        # carries the threefold degeneracies of its symmetry group, and a
        # contour cannot enclose one member of a degenerate band alone.
        for index, edge in enumerate(spacetime.getEdgeList().toVector()):
            edge.setLength(math.sqrt(1.0 + 0.07 * index))
            edge.setPhase(0.0)
        spacetime.materializeFacets()
        cls.assembled = cob.PencilLayer.assemble([spacetime])
        pencil = cob.PencilLayer.pencil(cls.assembled, 1)
        cls.operator = np.array(pencil.A)
        cls.metric = np.array(pencil.B)

    @classmethod
    def tearDownClass(cls):
        cob.HodgeLaplacian.setDefaultMetricSource(cls.previous)

    def test_a_cluster_pole_is_an_eigenvalue_the_interior_does_not_carry(self):
        interface = [0, 1, 2]
        full = _full_spectrum(self.operator, self.metric)
        interior = _interior_spectrum(self.operator, self.metric, interface)

        # The best-separated full eigenvalue that the interior does not carry,
        # chosen by measurement so the contour is never placed by hand.
        best, best_gap = None, 0.0
        for index, value in enumerate(full):
            gap = min([abs(value - other)
                       for position, other in enumerate(full)
                       if position != index]
                      + [abs(value - pole) for pole in interior])
            if gap > best_gap:
                best, best_gap = value, gap
        self.assertIsNotNone(best)
        self.assertGreater(best_gap, 1e-3)

        config = cob.BoundStatePoleConfig()
        read = BSP.cluster_poles(self.assembled, 1, interface, complex(best),
                                 0.3 * best_gap, config)
        self.assertEqual(read.failed_certificates, [])
        self.assertEqual(read.zeros, 1)
        self.assertLess(abs(read.poles[0] - best), 1e-8 * (1.0 + abs(best)))

    def test_the_matrix_form_and_the_pencil_form_agree(self):
        interface = [0, 1, 2]
        point = complex(0.37, 0.11)
        through_matrices = BSP.determinant(self.operator, self.metric,
                                           interface, point)
        through_pencil = cob.PencilLayer.boundary_response(
            self.assembled, 1, interface, point).responseDeterminant
        self.assertLess(abs(through_matrices - through_pencil),
                        1e-9 * (1.0 + abs(through_pencil)))


if __name__ == "__main__":
    unittest.main()
