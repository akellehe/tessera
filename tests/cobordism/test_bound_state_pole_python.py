# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""#1200 and #1323 — mass as the complex bound-state pole of a Feshbach
response pencil, read exactly from the pencil's spectrum.

Section 13.3 of the whitepaper says that mass is not defined by an incoherent
sum of moduli. It is the simple isolated zero ``s_C`` of
``D_C(s) = det F_C(s)``, with ``F_C`` the exact meromorphic Feshbach response
pencil of the whole persistent bound cluster, continued in the complex spectral
parameter ``s`` on a domain that excludes unretained interior poles. Four things
are asserted here rather than argued.

THE ZEROS ARE THE EIGENVALUES THE INTERIOR DOES NOT CARRY. By the determinant
factorization ``det P = det P_II det F_C``, the zeros of ``D_C`` on a finite
complex are exactly the eigenvalues of the full pencil that are not eigenvalues
of the interior block, with matching multiplicities. The read is that statement
computed: the eigenvalues of ``M^-1 A`` from its complex Schur form, clustered
at the declared rank tolerance, less those the interior pencil carries. Both
halves are checked against an independent eigendecomposition: every full
eigenvalue the interior does not carry is a reported pole at which ``D_C``
vanishes, the interior eigenvalues are listed as the poles of ``F_C`` and none
of them is reported as a zero, and an eigenvalue the interior carries is named
rather than reported, because the domain the response is continued on excludes
it.

THE DERIVATIVE IS ANALYTIC. ``F_C'(s)`` is the closed form that ``dP/ds = -M``
forces; it carries no finite difference and no step size, and it is checked
against a difference quotient formed in the test, which is where a difference
quotient belongs. It certifies each residue through ``tr(R F_C'(s_C)) = 1`` at
a simple pole, the identity that defines the residue of ``F_C^-1``.

A KNOWN BOUND STATE IS FOUND, AND ITS CERTIFICATE IS THE SPECIFICATION. On a
pencil whose spectrum is declared, the pole below the free threshold is the
declared eigenvalue to rounding, ``D_C`` vanishes there, the residue of the
supported resolvent has rank one, the residual of the pole's invariant subspace
is the rounding of the Schur form, the separation from the nearest other zero
is the declared gap and the binding shift is the declared depth.

A MULTIPLE POLE IS RETAINED, NOT SPLIT. A repeated eigenvalue is reported once
with its algebraic multiplicity, its Jordan structure and the whole residue
matrix its local data needs, rather than as two roots separated by an ordering
convention.
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
    real = np.array([[((row * 7 + column * 3) % 5) - 2
                      for column in range(order)]
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


def _config(**overrides):
    config = cob.BoundStatePoleConfig()
    for name, value in overrides.items():
        setattr(config, name, value)
    return config


class TheDeterminantFactorizationTest(unittest.TestCase):
    """``det P = det P_II det F_C``, and ``determinant`` is the second
    factor."""

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


class TheAnalyticDerivativeTest(unittest.TestCase):
    """The closed form of ``F_C'`` agrees with a difference quotient taken in
    the test."""

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


class TheZerosAreThePencilEigenvaluesTest(unittest.TestCase):
    """#1200's acceptance on a fixture with a known state: the read reports
    exactly the declared spectrum the interior does not carry."""

    SPECTRUM = [-2.5, 0.5, 1.5, 3.0]

    def setUp(self):
        self.operator, self.metric = _pencil(self.SPECTRUM)
        self.interface = [0, 1]
        self.interior = _interior_spectrum(self.operator, self.metric,
                                           self.interface)
        self.read = BSP.poles(self.operator, self.metric, self.interface,
                              _config())

    def test_the_fixture_separates_its_zeros_from_its_interior_poles(self):
        """The fixture's own certificate, stated before it is relied on."""
        for eigenvalue in self.SPECTRUM:
            for pole in self.interior:
                self.assertGreater(abs(eigenvalue - pole), 0.2)
        for index, left in enumerate(self.SPECTRUM):
            for right in self.SPECTRUM[index + 1:]:
                self.assertGreater(abs(left - right), 0.5)

    def test_every_full_eigenvalue_is_a_pole_at_which_the_determinant_vanishes(
            self):
        """The poles are the four declared eigenvalues, ascending, each of
        multiplicity one, and ``D_C`` vanishes at each. The exact read places
        a pole at the computed eigenvalue of a similarity transform of a
        diagonal matrix, whose error is the rounding of the Schur form times
        the conditioning of the fixed mixer, far below 1e-12; ``D_C`` at a
        point that close to a simple zero is that distance times the product
        of the other factors of ``D_C``, which the fixture's gaps bound by
        about 1e2, so it is below 1e-10."""
        read = self.read
        self.assertEqual(read.failed_certificates, [])
        self.assertEqual(len(read.poles), 4)
        for pole, eigenvalue in zip(read.poles, self.SPECTRUM):
            self.assertLess(abs(pole - eigenvalue), 1e-12)
            self.assertLess(abs(BSP.determinant(
                self.operator, self.metric, self.interface, pole)), 1e-10)
        self.assertEqual(list(read.multiplicity), [1, 1, 1, 1])
        self.assertEqual(list(read.simple), [True] * 4)
        self.assertEqual(list(read.geometric_multiplicity), [1, 1, 1, 1])
        self.assertEqual([list(b) for b in read.jordan_blocks], [[1]] * 4)
        self.assertEqual(list(read.cluster_spread), [0.0] * 4)

    def test_the_interior_eigenvalues_are_the_poles_of_the_response(self):
        """The interior eigenvalues are listed, with multiplicity one each,
        as the poles of ``F_C``, and none of them is reported as a zero."""
        read = self.read
        listed = sorted((complex(p) for p in read.interior_poles),
                        key=lambda value: (value.real, value.imag))
        expected = sorted((complex(p) for p in self.interior),
                          key=lambda value: (value.real, value.imag))
        self.assertEqual(len(listed), 2)
        for value, target in zip(listed, expected):
            self.assertLess(abs(value - target), 1e-10)
        self.assertEqual(list(read.interior_multiplicity), [1, 1])
        for pole in read.poles:
            for interior in read.interior_poles:
                self.assertGreater(abs(pole - interior), 0.2)

    def test_an_eigenvalue_the_interior_carries_is_named_not_reported(self):
        """``diag(1, 1)`` onto coordinate 0 has the response ``1 - s``,
        whose zero sits at the interior eigenvalue 1, outside the domain the
        response is continued on: the read names it and reports no pole,
        while the interior pole is listed."""
        read = BSP.poles(np.diag([1.0, 1.0]).astype(complex),
                         np.eye(2, dtype=complex), [0], _config())
        self.assertEqual(read.failed_certificates,
                         ["eigenvalue-at-interior-pole"])
        self.assertEqual(read.poles, [])
        self.assertEqual(list(read.interior_poles), [1.0])
        self.assertEqual(list(read.interior_multiplicity), [1])

    def test_the_bound_state_below_the_threshold_is_certified(self):
        """The lowest pole is the declared bound state -2.5 to rounding; its
        residue has rank one, its invariant subspace's residual is the
        rounding of the Schur form (below 1e-12 of the block's scale), its
        separation is the declared gap 3.0 to the next eigenvalue 0.5 and
        its binding shift below the free threshold 0 is the declared
        depth."""
        read = BSP.poles(self.operator, self.metric, self.interface,
                         _config(free_threshold=complex(0.0)))
        self.assertEqual(read.failed_certificates, [])
        self.assertLess(abs(read.poles[0] - (-2.5)), 1e-12)
        self.assertTrue(read.simple[0])
        self.assertEqual(read.residue_rank[0], 1)
        self.assertGreater(read.residue_norm[0], 0.0)
        self.assertGreater(read.scale, 0.0)
        self.assertLess(read.subspace_residual[0], 1e-12 * read.scale)
        self.assertAlmostEqual(read.separation[0], 3.0, places=12)
        self.assertLess(abs(read.binding_shift[0] - (-2.5)), 1e-12)
        self.assertEqual(len(read.binding_shift), 4)

    def test_the_residue_is_the_supported_resolvent_residue(self):
        """Each residue matrix is checked against the analytic derivative.

        At a simple zero the residue ``R`` of ``F_C^-1`` satisfies
        ``tr(R F_C'(s_C)) = 1``, which is the residue's defining property and
        needs no null vector to be ordered. The residue is the interface block
        of minus the spectral projector and the derivative is its closed form,
        so the identity holds to rounding.
        """
        read = self.read
        size = len(self.interface)
        for index, pole in enumerate(read.poles):
            residue = np.array(read.residue[index],
                               dtype=complex).reshape(size, size)
            derivative = np.array(BSP.response_derivative(
                self.operator, self.metric, self.interface, pole))
            self.assertAlmostEqual(abs(np.trace(residue @ derivative) - 1.0),
                                   0.0, places=10)

    def test_the_separations_are_the_distances_between_distinct_poles(self):
        """The separation of each pole is its distance to the nearest other
        reported pole: 3.0, 1.0, 1.0 and 1.5 for the declared spectrum."""
        for measured, expected in zip(self.read.separation,
                                      [3.0, 1.0, 1.0, 1.5]):
            self.assertAlmostEqual(measured, expected, places=12)

    def test_the_declared_defaults(self):
        """The read has exactly two declared parameters: the rank tolerance
        and the optional free threshold."""
        config = cob.BoundStatePoleConfig()
        self.assertEqual(config.rank_tolerance, 1e-15)
        self.assertIsNone(config.free_threshold)
        self.assertEqual(
            sorted(name for name in dir(config) if not name.startswith("_")),
            ["free_threshold", "rank_tolerance"])

    def test_an_empty_interface_refuses(self):
        read = BSP.poles(self.operator, self.metric, [], _config())
        self.assertIn("empty-interface", read.failed_certificates)

    def test_a_singular_metric_is_refused_by_name(self):
        """A pencil whose metric is singular at the declared rank tolerance
        has a spectrum that is not the spectrum of a matrix; the read refuses
        by name rather than reading a substitute."""
        metric = np.diag([1.0, 0.0, 1.0, 1.0]).astype(complex)
        with self.assertRaisesRegex(ValueError, "metric block is singular"):
            BSP.poles(self.operator, metric, self.interface, _config())
        # An invertible metric whose interior block, on coordinates 2 and 3,
        # is diag(0, 1): the interior pencil then has no matrix spectrum.
        metric = np.array([[1.0, 0.0, 0.0, 0.0],
                           [0.0, 1.0, 1.0, 0.0],
                           [0.0, 1.0, 0.0, 0.0],
                           [0.0, 0.0, 0.0, 1.0]], dtype=complex)
        self.assertGreater(abs(np.linalg.det(metric)), 0.5)
        with self.assertRaisesRegex(ValueError,
                                    "interior block of the metric is "
                                    "singular"):
            BSP.poles(self.operator, metric, self.interface, _config())


class AMultiplePoleIsRetainedTest(unittest.TestCase):
    """A repeated eigenvalue keeps its multiplicity, its Jordan structure and
    its whole residue."""

    def setUp(self):
        # Diagonal, with the whole block on the interface, so the response is
        # exactly diag(1.25 - s, 1.25 - s, -3 - s, 4 - s): a zero of algebraic
        # multiplicity two at 1.25 and two simple zeros.
        self.operator = np.diag(np.array([1.25, 1.25, -3.0, 4.0],
                                         dtype=complex))
        self.metric = np.eye(4, dtype=complex)
        self.interface = [0, 1, 2, 3]

    def test_the_double_zero_is_reported_once_with_multiplicity_two(self):
        """The block is diagonal, so its Schur diagonal is exact and the
        repeated eigenvalue is exactly repeated: one pole at 1.25 of
        multiplicity two with two Jordan blocks of size one, beside the
        simple poles -3 and 4, ascending."""
        read = BSP.poles(self.operator, self.metric, self.interface,
                         _config())
        self.assertEqual(read.failed_certificates, [])
        self.assertEqual(len(read.poles), 3)
        self.assertLess(abs(read.poles[0] - (-3.0)), 1e-15)
        self.assertLess(abs(read.poles[1] - 1.25), 1e-15)
        self.assertLess(abs(read.poles[2] - 4.0), 1e-15)
        self.assertEqual(list(read.multiplicity), [1, 2, 1])
        self.assertEqual(list(read.geometric_multiplicity), [1, 2, 1])
        self.assertEqual([list(b) for b in read.jordan_blocks],
                         [[1], [1, 1], [1]])
        self.assertEqual(list(read.simple), [True, False, True])
        self.assertEqual(list(read.cluster_spread), [0.0, 0.0, 0.0])

    def test_the_residue_of_the_double_zero_is_minus_its_projector(self):
        """The residue of ``F_C^-1 = (A - s)^-1`` at 1.25 is minus the
        spectral projector onto its two-dimensional eigenspace. The
        eigenvalue 1.25 sits on coordinates 0 and 1 of the diagonal block, so
        the projector is ``diag(1, 1, 0, 0)`` and the residue, the second in
        the ascending order of the poles, is ``-diag(1, 1, 0, 0)``, of rank
        two."""
        read = BSP.poles(self.operator, self.metric, self.interface,
                         _config())
        residue = np.array(read.residue[1], dtype=complex).reshape(4, 4)
        self.assertEqual(read.residue_rank[1], 2)
        self.assertLess(np.max(np.abs(residue + np.diag([1.0, 1.0, 0.0, 0.0]))),
                        1e-14)

    def test_a_jordan_block_is_read_from_the_ranks_of_the_nilpotent_part(self):
        """An upper triangular block with the eigenvalue 2 three times, once
        in a Jordan block of size two and once alone, and the eigenvalue 5:
        its Schur diagonal is exact, the pole 2 has algebraic multiplicity
        three, geometric multiplicity two and the Jordan blocks [2, 1], and
        its residue (minus the projector onto the three-dimensional
        generalized eigenspace) has rank three."""
        operator = np.array([[2.0, 1.0, 0.0, 0.0],
                             [0.0, 2.0, 0.0, 0.0],
                             [0.0, 0.0, 2.0, 0.0],
                             [0.0, 0.0, 0.0, 5.0]], dtype=complex)
        read = BSP.poles(operator, self.metric, self.interface, _config())
        self.assertEqual(read.failed_certificates, [])
        self.assertLess(abs(read.poles[0] - 2.0), 1e-15)
        self.assertLess(abs(read.poles[1] - 5.0), 1e-15)
        self.assertEqual(list(read.multiplicity), [3, 1])
        self.assertEqual(list(read.geometric_multiplicity), [2, 1])
        self.assertEqual([list(b) for b in read.jordan_blocks], [[2, 1], [1]])
        self.assertEqual(list(read.residue_rank), [3, 1])

    def test_a_coupling_above_the_tolerance_is_a_jordan_block(self):
        """``[[2, 1e-12], [0, 2]]`` has the largest singular value 2 to one
        part in 1e-24, so its nilpotent part scaled by that value has the one
        singular value 5e-13, above the declared rank tolerance 1e-15: at
        the declared tolerance the pole 2 has algebraic multiplicity two and
        geometric multiplicity one, one Jordan block of size two. A coupling
        of 1e-16, whose scaled singular value 5e-17 is below the tolerance,
        leaves the block semisimple, with two Jordan blocks of size one."""
        operator = np.array([[2.0, 1e-12], [0.0, 2.0]], dtype=complex)
        read = BSP.poles(operator, np.eye(2, dtype=complex), [0, 1],
                         _config())
        self.assertEqual(read.failed_certificates, [])
        self.assertEqual(list(read.multiplicity), [2])
        self.assertEqual(list(read.geometric_multiplicity), [1])
        self.assertEqual([list(b) for b in read.jordan_blocks], [[2]])
        operator = np.array([[2.0, 1e-16], [0.0, 2.0]], dtype=complex)
        read = BSP.poles(operator, np.eye(2, dtype=complex), [0, 1],
                         _config())
        self.assertEqual(list(read.multiplicity), [2])
        self.assertEqual(list(read.geometric_multiplicity), [2])
        self.assertEqual([list(b) for b in read.jordan_blocks], [[1, 1]])


class TheFrameworkPencilPathTest(unittest.TestCase):
    """The same read on the framework's own dressed pencil."""

    @classmethod
    def setUpClass(cls):
        cls.previous = cob.HodgeLaplacian.defaultMetricSource()
        cob.HodgeLaplacian.setDefaultMetricSource(
            cob.HodgeMetricSource.WhitneyPencil)
        spacetime = T.Spacetime.fromVertexTuples(3, [[0, 1, 2, 3]], 1.0, 0.0)
        # A deliberately asymmetric metric: the regular tetrahedron's spectrum
        # carries the threefold degeneracies of its symmetry group, and the
        # test below compares the read with the spectrum eigenvalue by
        # eigenvalue.
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

    def test_the_cluster_poles_are_the_eigenvalues_the_interior_does_not_carry(
            self):
        """The poles of the degree-one pencil onto three edges, counted with
        multiplicity, are the eigenvalues of ``M^-1 A`` that the interior
        pencil does not carry, compared with an independent
        eigendecomposition to 1e-8 of their size. The fixture's own
        certificate, stated first, is that no full eigenvalue lies within
        1e-6 of an interior eigenvalue, so every full eigenvalue is a pole."""
        interface = [0, 1, 2]
        full = _full_spectrum(self.operator, self.metric)
        interior = _interior_spectrum(self.operator, self.metric, interface)
        for value in full:
            for pole in interior:
                self.assertGreater(abs(value - pole), 1e-6)

        read = BSP.cluster_poles(self.assembled, 1, interface, _config())
        self.assertEqual(read.failed_certificates, [])
        reported = sorted(
            (complex(pole) for pole, count in zip(read.poles,
                                                  read.multiplicity)
             for _ in range(count)),
            key=lambda value: (value.real, value.imag))
        expected = sorted((complex(v) for v in full),
                          key=lambda value: (value.real, value.imag))
        self.assertEqual(len(reported), len(expected))
        for value, target in zip(reported, expected):
            self.assertLess(abs(value - target), 1e-8 * (1.0 + abs(target)))
        listed = sorted((complex(v) for v in read.interior_poles),
                        key=lambda value: (value.real, value.imag))
        targets = sorted((complex(v) for v in interior),
                         key=lambda value: (value.real, value.imag))
        self.assertEqual(len(listed), len(targets))
        for value, target in zip(listed, targets):
            self.assertLess(abs(value - target), 1e-8 * (1.0 + abs(target)))

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
