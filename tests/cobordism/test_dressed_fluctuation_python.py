# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""#1190 — the dressed fluctuation propagator, its poles, and the exact elimination.

Section 7 of the whitepaper asks for four things about a retained fluctuation of
the geometry or of the connection, and each is asserted here against a dense
reference built independently in this file.

THE DRESSED STIFFNESS IS COMPLEX BILINEAR. ``A_eff(w) = A + D - Pi(w)`` is
assembled from matrix elements taken between the left and right modes of the
carrier, ``PhiTilde^T O Phi``, with no conjugation anywhere. The carrier of the
tests below is complex and non-normal, so a construction that took an adjoint
would give different numbers; the dense reference takes the same transpose
pairing and the two agree to rounding.

THE INDUCED STIFFNESS IS THE HESSIAN OF THE OCCUPIED ENERGY, AND IT VANISHES ON
A PURE-GAUGE DIRECTION. ``D - Pi(0)`` is held to a central-difference Hessian of
the sum of the occupied eigenvalues of ``h_1(s, U)`` on the tetrahedron, which is
the check the whitepaper reports to seven decimal places. On a pure-gauge
direction the carrier moves by a similarity transformation, so every eigenvalue
stands still and the Hessian is exactly zero there; the paramagnetic term alone
is not, which is why the identity is a statement about the two terms together.

THE POLES ARE FOUND EXACTLY. ``collective_modes`` solves a linear pencil rather
than sampling frequencies, and every frequency it returns is held to a dense
evaluation of ``det A_eff(w)``. On one particle-hole energy the whitepaper gives
the closed form ``w^2 = Delta^2 (1 - pi / (a + d))``, and the pole search
reproduces it; past ``pi = a + d`` the same formula sends the pole off the real
axis, which is the random-phase signature of self-trapping and is the case in
which the radiation rate is not zero.

THE QUARTIC IS THE EXACT ELIMINATION. ``-1/2 J^T A^-1 J`` on the three-particle
space of a cluster is held to a dense reference built from the elementary
creation and annihilation operators of the whole exterior algebra, restricted
afterwards to the three-particle sector. The split of that product into its
one-body remainder and the strictly quartic, normal-ordered interaction is
asserted to be exact, and the normal-ordered part is asserted to annihilate the
one-particle sector, which is what makes it the genuine quartic of the
whitepaper's list.
"""

import itertools
import unittest

import numpy as np

import tessera as T

cob = T.cobordism
ch = T.chainhodge

#: A single tetrahedron: four vertices, six edges, four triangles. It is the
#: complex the whitepaper quotes its connection-stiffness numbers on.
TETRAHEDRON = [[0, 1, 2, 3]]


def _flat(matrix):
    """A dense matrix as the flat row-major list the bindings take."""
    return [complex(value) for value in np.asarray(matrix).reshape(-1)]


def _square(flat, order):
    """The inverse of ``_flat`` for a square matrix of the given order."""
    return np.asarray(flat, dtype=complex).reshape(order, order)


def _covariant(seed=3, complex_data=True):
    """The tetrahedron's covariant degree-one operator and its derivatives.

    The squared lengths and the links are drawn deterministically from ``seed``.
    With ``complex_data`` the links leave the unit circle and the squared lengths
    leave the real axis, which makes ``h_1(s, U)`` genuinely non-normal and so
    makes the transpose pairing distinguishable from an adjoint one.
    """
    rng = np.random.default_rng(seed)
    K = cob.ChainComplex.fromTopCells(TETRAHEDRON)
    edges = K.numSimplices(1)
    if complex_data:
        squared = [complex(1.0 + 0.1 * rng.normal(), 0.1 * rng.normal())
                   for _ in range(edges)]
        links = [complex(rng.normal(), rng.normal()) + 1.5 for _ in range(edges)]
    else:
        squared = [complex(1.0 + 0.1 * rng.uniform(-1, 1)) for _ in range(edges)]
        links = [complex(np.exp(1j * rng.uniform(-np.pi, np.pi)))
                 for _ in range(edges)]
    base = ch.ChainHodge(K, squared)
    covariant = ch.CovariantChainHodge(base, ch.Connection(K, links), 7, False)
    return K, covariant, links, base


def _declaration(carrier, couplings, second=None, stiffness=None, occupied=1,
                 tolerance=1e-8, broadening=0.0):
    """A dressed-fluctuation declaration over dense numpy matrices."""
    declaration = cob.DressedFluctuationDeclaration()
    declaration.carrier_dimension = int(np.asarray(carrier).shape[0])
    declaration.carrier = _flat(carrier)
    declaration.couplings = [_flat(operator) for operator in couplings]
    if second is not None:
        declaration.second_derivatives = [_flat(operator) for operator in second]
    if stiffness is not None:
        declaration.bare_stiffness = _flat(stiffness)
    declaration.occupied_modes = occupied
    declaration.tolerance = tolerance
    declaration.continuum_broadening = broadening
    return declaration


def _upper_triangle(count, entry):
    """The pairs ``a <= b`` of ``count`` fluctuations in the declared order."""
    return [entry(a, b) for a in range(count) for b in range(a, count)]


def _reference_response(carrier, couplings, second, stiffness, occupied,
                        frequency):
    """A dense reference for ``D``, ``Pi(w)`` and ``A_eff(w)``.

    The modes come from a plain dense eigendecomposition, the left frame is the
    inverse of the right one, and every bracket is the bilinear one. The
    paramagnetic numerator carries both orders of the two matrix elements, which
    is what second-order perturbation theory of the mixed second derivative of
    the occupied energy produces.
    """
    values, right = np.linalg.eig(np.asarray(carrier, dtype=complex))
    order = sorted(range(len(values)),
                   key=lambda index: (values[index].real, values[index].imag))
    right = right[:, order]
    values = values[order]
    left = np.linalg.inv(right)
    count = len(couplings)
    currents = [left @ np.asarray(operator, dtype=complex) @ right
                for operator in couplings]
    diamagnetic = np.zeros((count, count), dtype=complex)
    for position, (a, b) in enumerate(
            [(a, b) for a in range(count) for b in range(a, count)]):
        transformed = left @ np.asarray(second[position], dtype=complex) @ right
        value = sum(transformed[mode, mode] for mode in range(occupied))
        diamagnetic[a, b] = diamagnetic[b, a] = value
    paramagnetic = np.zeros((count, count), dtype=complex)
    for m in range(occupied):
        for n in range(occupied, len(values)):
            gap = values[n] - values[m]
            weight = gap / (gap * gap - frequency * frequency)
            forward = np.outer([current[m, n] for current in currents],
                               [current[n, m] for current in currents])
            paramagnetic += weight * (forward + forward.T)
    dressed = diamagnetic - paramagnetic
    if stiffness is not None:
        dressed = dressed + np.asarray(stiffness, dtype=complex)
    return diamagnetic, paramagnetic, dressed


def _exterior_operators(rank):
    """The elementary creation and annihilation matrices of ``Lambda(C^rank)``.

    The basis is the ascending subsets of ``{0, ..., rank - 1}`` in lexicographic
    order within each particle number, gathered by particle number. The sign of
    ``epsilon_i`` on a basis state is minus one to the number of occupied modes
    below ``i``, and the same for ``iota_i``, which is the exterior-algebra
    convention and is built here from the subsets themselves rather than from any
    one-body lifting rule.
    """
    states = []
    for particles in range(rank + 1):
        states.extend(itertools.combinations(range(rank), particles))
    position = {state: index for index, state in enumerate(states)}
    creation = [np.zeros((len(states), len(states)), dtype=complex)
                for _ in range(rank)]
    annihilation = [np.zeros((len(states), len(states)), dtype=complex)
                    for _ in range(rank)]
    for state in states:
        for mode in range(rank):
            sign = (-1.0) ** sum(1 for member in state if member < mode)
            if mode in state:
                target = tuple(member for member in state if member != mode)
                annihilation[mode][position[target], position[state]] = sign
            else:
                target = tuple(sorted(state + (mode,)))
                creation[mode][position[target], position[state]] = sign
    return states, position, creation, annihilation


def _reference_many_body(carrier, couplings, stiffness, particles):
    """A dense reference for the one-body term and the quartic of ``S_eff``.

    Every operator is built on the whole exterior algebra from the elementary
    ``epsilon`` and ``iota`` matrices and only then restricted to the
    ``particles``-particle sector, which is a different construction from the
    one the class uses and so an independent check of it.
    """
    carrier = np.asarray(carrier, dtype=complex)
    rank = carrier.shape[0]
    states, position, creation, annihilation = _exterior_operators(rank)
    sector = [position[state] for state in states if len(state) == particles]

    def lift(operator):
        total = np.zeros((len(states), len(states)), dtype=complex)
        for i in range(rank):
            for j in range(rank):
                total += operator[i, j] * (creation[i] @ annihilation[j])
        return total

    one_body = lift(carrier)
    currents = [lift(np.asarray(operator, dtype=complex))
                for operator in couplings]
    inverse = np.linalg.inv(np.asarray(stiffness, dtype=complex))
    quartic = np.zeros((len(states), len(states)), dtype=complex)
    for a in range(len(currents)):
        for b in range(len(currents)):
            quartic -= 0.5 * inverse[a, b] * (currents[a] @ currents[b])
    return one_body[np.ix_(sector, sector)], quartic[np.ix_(sector, sector)]


class TheDressedStiffnessIsTheWhitepapersFormulaTest(unittest.TestCase):
    """A + D - Pi(w) entry by entry against a dense reference."""

    def test_the_three_terms_match_a_dense_reference_on_the_tetrahedron(self):
        """The whole assembly is held to an independent dense evaluation.

        The carrier is the tetrahedron's covariant degree-one operator on a
        complex metric and a non-unimodular connection, so it is non-normal and
        its left modes are the rows of the inverse of its right modes rather
        than their conjugate transposes.
        """
        K, cov, _, _ = _covariant()
        edges = K.numSimplices(1)
        carrier = cov.covariantOperator(1)
        couplings = [cov.covariantOperatorPhaseDerivative(1, a)
                     for a in range(edges)]
        second = _upper_triangle(
            edges, lambda a, b: cov.covariantOperatorPhaseHessian(1, a, b))
        stiffness = np.eye(edges, dtype=complex) * 0.25
        fluctuation = cob.DressedFluctuation(
            _declaration(carrier, couplings, second, stiffness, occupied=3))
        for frequency in (0.0, 0.31, 0.77 + 0.4j):
            reference = _reference_response(carrier, couplings, second,
                                            stiffness, 3, frequency)
            scale = max(1.0, np.abs(reference[2]).max())
            self.assertLess(
                np.abs(_square(fluctuation.diamagnetic(), edges)
                       - reference[0]).max(), 1e-9 * scale)
            self.assertLess(
                np.abs(_square(fluctuation.paramagnetic(frequency), edges)
                       - reference[1]).max(), 1e-9 * scale)
            self.assertLess(
                np.abs(_square(fluctuation.dressed_stiffness(frequency), edges)
                       - reference[2]).max(), 1e-9 * scale)

    def test_the_polarization_is_complex_symmetric(self):
        """Both orders of the matrix elements enter, so Pi(w) = Pi(w)^T."""
        K, cov, _, _ = _covariant()
        edges = K.numSimplices(1)
        fluctuation = cob.DressedFluctuation(_declaration(
            cov.covariantOperator(1),
            [cov.covariantOperatorPhaseDerivative(1, a) for a in range(edges)],
            occupied=3))
        polarization = _square(fluctuation.paramagnetic(0.4), edges)
        self.assertLess(np.abs(polarization - polarization.T).max(),
                        1e-10 * max(1.0, np.abs(polarization).max()))

    def test_the_polarization_refuses_its_own_pole(self):
        """At a particle-hole energy the polarization has no finite value."""
        carrier = np.diag([0.0, 2.0]).astype(complex)
        coupling = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
        fluctuation = cob.DressedFluctuation(
            _declaration(carrier, [coupling], occupied=1))
        with self.assertRaises(ValueError):
            fluctuation.paramagnetic(2.0)


class TheInducedStiffnessIsTheHessianOfTheOccupiedEnergyTest(unittest.TestCase):
    """D - Pi(0) against a central difference, and the Ward identity."""

    def test_it_matches_a_central_difference_of_the_occupied_energy(self):
        """The occupied energy is the sum of the filled eigenvalues.

        A whole isolated band is a smooth function of the connection even where
        its individual members cross, so the sum over the tetrahedron's lower
        rank-three band is differentiated by plain central differences. The
        whitepaper reports this identity to seven decimal places; the step and
        the tolerance here are the ones a four-point second difference supports.
        """
        K, cov, links, base = _covariant(seed=11, complex_data=False)
        edges = K.numSimplices(1)
        couplings = [cov.covariantOperatorPhaseDerivative(1, a)
                     for a in range(edges)]
        second = _upper_triangle(
            edges, lambda a, b: cov.covariantOperatorPhaseHessian(1, a, b))
        fluctuation = cob.DressedFluctuation(_declaration(
            cov.covariantOperator(1), couplings, second, occupied=3))
        induced = _square(fluctuation.induced_stiffness(), edges)

        def energy(shifts):
            moved = [link * np.exp(1j * shift)
                     for link, shift in zip(links, shifts)]
            operator = ch.CovariantChainHodge(
                base, ch.Connection(K, moved), 7, False).covariantOperator(1)
            values = np.linalg.eigvals(operator)
            return sum(sorted(values, key=lambda v: (v.real, v.imag))[:3])

        delta = 1e-4
        for a in range(edges):
            for b in range(a, edges):
                plus = [0.0] * edges
                minus = [0.0] * edges
                plus[a] += delta
                plus[b] += delta
                minus[a] -= delta
                minus[b] -= delta
                mixed = [0.0] * edges
                mixed[a] += delta
                mixed[b] -= delta
                crossed = [0.0] * edges
                crossed[a] -= delta
                crossed[b] += delta
                curvature = (energy(plus) - energy(mixed) - energy(crossed)
                             + energy(minus)) / (4.0 * delta * delta)
                self.assertLess(abs(curvature - induced[a, b]),
                                1e-5 * max(1.0, abs(induced[a, b])))

    def test_the_ward_identity_holds_on_every_pure_gauge_direction(self):
        """A pure-gauge direction is a similarity, so no eigenvalue moves.

        The directions are the coboundary of the vertex functions, the columns
        of ``d_2``'s predecessor: ``delta phi_{xy} = chi_y - chi_x``. On each of
        them the induced stiffness is zero to rounding while the paramagnetic
        term alone is not, which is the whole content of the identity.
        """
        K, cov, _, _ = _covariant(seed=5)
        edges = K.numSimplices(1)
        fluctuation = cob.DressedFluctuation(_declaration(
            cov.covariantOperator(1),
            [cov.covariantOperatorPhaseDerivative(1, a) for a in range(edges)],
            _upper_triangle(edges,
                            lambda a, b: cov.covariantOperatorPhaseHessian(1, a, b)),
            occupied=3))
        vertices = [int(cell[0]) for cell in K.kSimplexVertices(0)]
        directions = []
        for chosen in vertices:
            chi = {vertex: (1.0 if vertex == chosen else 0.0)
                   for vertex in vertices}
            directions.append([complex(chi[int(y)] - chi[int(x)])
                               for x, y in K.kSimplexVertices(1)])
        certificate = fluctuation.ward_certificate(directions)
        self.assertTrue(certificate.holds())
        self.assertLess(certificate.residual, 1e-8)

        polarization = _square(fluctuation.paramagnetic(0.0), edges)
        worst = max(np.abs(polarization @ np.asarray(direction, dtype=complex)).max()
                    for direction in directions)
        self.assertGreater(worst, 1e-3 * np.abs(polarization).max())

    def test_a_direction_of_the_wrong_length_is_refused(self):
        carrier = np.diag([0.0, 2.0]).astype(complex)
        coupling = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
        fluctuation = cob.DressedFluctuation(
            _declaration(carrier, [coupling], occupied=1))
        with self.assertRaises(ValueError):
            fluctuation.ward_residual([1.0, 2.0])


class ThePolesAreTheZerosOfTheDressedStiffnessTest(unittest.TestCase):
    """The pole search against the determinant and against a closed form."""

    def test_one_particle_hole_energy_reproduces_the_closed_form(self):
        """The whitepaper's ``w^2 = Delta^2 (1 - pi / (a + d))``.

        With a single particle-hole pair and one retained fluctuation, the
        polarization is ``pi Delta^2 / (Delta^2 - w^2)`` with ``pi = Pi(0)``, so
        the dressed stiffness vanishes exactly where the closed form says.
        """
        gap = 2.5
        carrier = np.diag([0.0, gap]).astype(complex)
        coupling = np.array([[0.0, 0.8], [0.8, 0.0]], dtype=complex)
        second = [np.diag([0.6, 0.0]).astype(complex)]
        stiffness = np.array([[1.4]], dtype=complex)
        fluctuation = cob.DressedFluctuation(_declaration(
            carrier, [coupling], second, stiffness, occupied=1))
        scalar = fluctuation.diamagnetic()[0] + stiffness[0, 0]
        polarization = fluctuation.paramagnetic(0.0)[0]
        expected = gap * gap * (1.0 - polarization / scalar)
        modes = fluctuation.collective_modes()
        self.assertEqual(len(modes), 2)
        for mode in modes:
            self.assertAlmostEqual(mode.frequency ** 2, expected, places=9)
            self.assertAlmostEqual(mode.radiation_rate, 0.0, places=9)
        self.assertAlmostEqual(modes[0].frequency, -modes[1].frequency,
                               places=10)

    def test_a_polarization_above_the_stiffness_sends_the_pole_off_the_axis(self):
        """``pi > a + d`` is the random-phase signature of self-trapping.

        The closed form then gives a negative squared frequency, so the pole
        leaves the real axis and the mode acquires a radiation rate. One member
        of the pair relaxes and the other grows, which is what an unstable
        direction of a stationary point is.
        """
        gap = 1.0
        carrier = np.diag([0.0, gap]).astype(complex)
        coupling = np.array([[0.0, 1.2], [1.2, 0.0]], dtype=complex)
        stiffness = np.array([[0.05]], dtype=complex)
        fluctuation = cob.DressedFluctuation(
            _declaration(carrier, [coupling], stiffness=stiffness, occupied=1))
        scalar = stiffness[0, 0]
        polarization = fluctuation.paramagnetic(0.0)[0]
        self.assertGreater(polarization.real, scalar.real)
        modes = fluctuation.collective_modes()
        self.assertEqual(len(modes), 2)
        for mode in modes:
            self.assertLess(abs(mode.frequency.real), 1e-9)
            self.assertAlmostEqual(mode.radiation_rate,
                                   -2.0 * mode.frequency.imag, places=12)
        self.assertGreater(max(mode.radiation_rate for mode in modes), 0.0)

    def test_every_reported_pole_is_a_zero_of_the_determinant(self):
        """A dense determinant of A_eff at each reported frequency.

        The determinant is scaled by the product of the singular values of the
        bare part, so the comparison is relative rather than in the units the
        stiffness happens to carry.
        """
        K, cov, _, _ = _covariant(seed=9, complex_data=False)
        edges = K.numSimplices(1)
        stiffness = np.zeros((edges, edges), dtype=complex)
        boundary = np.asarray(cob.ChainComplex.fromTopCells(TETRAHEDRON)
                              .boundaryMatrix(2), dtype=float).reshape(
                                  edges, K.numSimplices(2))
        stiffness += 0.5 * (boundary @ boundary.T).astype(complex)
        fluctuation = cob.DressedFluctuation(_declaration(
            cov.covariantOperator(1),
            [cov.covariantOperatorPhaseDerivative(1, a) for a in range(edges)],
            _upper_triangle(edges,
                            lambda a, b: cov.covariantOperatorPhaseHessian(1, a, b)),
            stiffness, occupied=3))
        modes = fluctuation.collective_modes()
        self.assertGreater(len(modes), 0)
        for mode in modes:
            dressed = _square(fluctuation.dressed_stiffness(mode.frequency),
                              edges)
            singular = np.linalg.svd(dressed, compute_uv=False)
            self.assertLess(singular[-1], 1e-6 * singular[0])
            direction = np.asarray(mode.polarization, dtype=complex)
            self.assertLess(np.abs(dressed @ direction).max(),
                            1e-6 * np.abs(dressed).max())
            self.assertAlmostEqual(np.linalg.norm(direction), 1.0, places=10)

    def test_the_poles_come_in_pairs_because_the_stiffness_is_even(self):
        """A_eff depends on w only through w^2, so -w is a pole with w."""
        K, cov, _, _ = _covariant(seed=9, complex_data=False)
        edges = K.numSimplices(1)
        fluctuation = cob.DressedFluctuation(_declaration(
            cov.covariantOperator(1),
            [cov.covariantOperatorPhaseDerivative(1, a) for a in range(edges)],
            _upper_triangle(edges,
                            lambda a, b: cov.covariantOperatorPhaseHessian(1, a, b)),
            np.eye(edges, dtype=complex) * 0.5, occupied=3))
        modes = fluctuation.collective_modes()
        frequencies = [mode.frequency for mode in modes]
        # A pole of high multiplicity at zero comes back as a cluster whose
        # spread is the root the multiplicity takes of rounding, so the partner
        # is sought within that spread rather than to rounding itself.
        for frequency in frequencies:
            self.assertTrue(any(abs(other + frequency)
                                < 1e-5 * max(1.0, abs(frequency))
                                for other in frequencies))

    def test_a_declared_broadening_gives_every_pole_a_radiation_rate(self):
        """The one approximation of the class, and it is never silent.

        A finite complex carries finitely many particle-hole energies, so a
        Hermitian carrier has no continuum for a mode to leak into and every
        pole sits on the real axis. The declared broadening gives each
        particle-hole excitation a finite lifetime, and the modes acquire the
        width that lifetime implies.
        """
        gap = 2.5
        carrier = np.diag([0.0, gap]).astype(complex)
        coupling = np.array([[0.0, 0.8], [0.8, 0.0]], dtype=complex)
        stiffness = np.array([[1.4]], dtype=complex)
        sharp = cob.DressedFluctuation(
            _declaration(carrier, [coupling], stiffness=stiffness, occupied=1))
        broad = cob.DressedFluctuation(
            _declaration(carrier, [coupling], stiffness=stiffness, occupied=1,
                         broadening=0.05))
        for mode in sharp.collective_modes():
            self.assertAlmostEqual(mode.radiation_rate, 0.0, places=9)
        self.assertGreater(
            max(abs(mode.radiation_rate) for mode in broad.collective_modes()),
            1e-3)


class TheQuarticIsTheExactEliminationTest(unittest.TestCase):
    """-1/2 J^T A^-1 J on the three-particle space of a cluster."""

    def _instance(self, rank=4, fluctuations=3, seed=17):
        rng = np.random.default_rng(seed)
        carrier = (rng.normal(size=(rank, rank))
                   + 1j * rng.normal(size=(rank, rank)))
        couplings = [rng.normal(size=(rank, rank))
                     + 1j * rng.normal(size=(rank, rank))
                     for _ in range(fluctuations)]
        stiffness = rng.normal(size=(fluctuations, fluctuations))
        stiffness = (stiffness + stiffness.T).astype(complex)
        stiffness += np.eye(fluctuations) * 3.0
        return carrier, couplings, stiffness

    def test_it_matches_a_dense_exterior_algebra_reference(self):
        """Built from the elementary epsilon and iota of the whole algebra."""
        carrier, couplings, stiffness = self._instance()
        fluctuation = cob.DressedFluctuation(
            _declaration(carrier, couplings, stiffness=stiffness, occupied=2))
        read = fluctuation.effective_action([], [], 3)
        self.assertEqual(read.particles, 3)
        self.assertEqual(read.fiber_rank, 4)
        self.assertEqual(read.dimension, 4)
        self.assertEqual([list(entry) for entry in read.basis],
                         [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
        one_body, quartic = _reference_many_body(carrier, couplings, stiffness, 3)
        scale = max(1.0, np.abs(quartic).max())
        self.assertLess(np.abs(_square(read.one_body, 4) - one_body).max(),
                        1e-10 * scale)
        self.assertLess(np.abs(_square(read.quartic, 4) - quartic).max(),
                        1e-10 * scale)
        self.assertLess(
            np.abs(_square(read.effective_action, 4) - one_body - quartic).max(),
            1e-10 * scale)
        self.assertTrue(read.certificate.holds())

    def test_the_split_into_one_body_and_normal_ordered_parts_is_exact(self):
        """The reordering identity, asserted rather than argued."""
        carrier, couplings, stiffness = self._instance()
        fluctuation = cob.DressedFluctuation(
            _declaration(carrier, couplings, stiffness=stiffness, occupied=2))
        read = fluctuation.effective_action([], [], 3)
        quartic = _square(read.quartic, read.dimension)
        induced = _square(read.induced_one_body, read.dimension)
        ordered = _square(read.normal_ordered_quartic, read.dimension)
        self.assertLess(np.abs(quartic - induced - ordered).max(),
                        1e-10 * max(1.0, np.abs(quartic).max()))

    def test_the_normal_ordered_part_annihilates_the_one_particle_sector(self):
        """It is the genuine quartic, so it needs two particles to act on."""
        carrier, couplings, stiffness = self._instance()
        fluctuation = cob.DressedFluctuation(
            _declaration(carrier, couplings, stiffness=stiffness, occupied=2))
        read = fluctuation.effective_action([], [], 1)
        ordered = _square(read.normal_ordered_quartic, read.dimension)
        self.assertLess(np.abs(ordered).max(), 1e-10)
        self.assertGreater(np.abs(_square(read.quartic, read.dimension)).max(),
                           1e-6)

    def test_a_declared_cluster_fiber_restricts_the_space(self):
        """The frames are a right frame and its algebraic dual, paired by the
        transpose. The restriction is taken with that pairing, so the reported
        pairing defect is zero and the three-particle space of a rank-three
        fiber is one-dimensional."""
        carrier, couplings, stiffness = self._instance(rank=5, seed=23)
        rng = np.random.default_rng(29)
        frame = (rng.normal(size=(5, 3)) + 1j * rng.normal(size=(5, 3)))
        dual = np.linalg.pinv(frame)
        fluctuation = cob.DressedFluctuation(
            _declaration(carrier, couplings, stiffness=stiffness, occupied=2))
        read = fluctuation.effective_action(_flat(frame), _flat(dual), 3)
        self.assertEqual(read.fiber_rank, 3)
        self.assertEqual(read.dimension, 1)
        self.assertLess(read.frame_pairing_defect, 1e-10)
        one_body, quartic = _reference_many_body(
            dual @ carrier @ frame,
            [dual @ operator @ frame for operator in couplings], stiffness, 3)
        scale = max(1.0, np.abs(quartic).max())
        self.assertLess(np.abs(_square(read.one_body, 1) - one_body).max(),
                        1e-9 * scale)
        self.assertLess(np.abs(_square(read.quartic, 1) - quartic).max(),
                        1e-9 * scale)

    def test_the_elimination_refuses_what_it_cannot_do(self):
        carrier, couplings, stiffness = self._instance()
        bare = cob.DressedFluctuation(
            _declaration(carrier, couplings, occupied=2))
        with self.assertRaises(ValueError):
            bare.effective_action([], [], 3)
        fluctuation = cob.DressedFluctuation(
            _declaration(carrier, couplings, stiffness=stiffness, occupied=2))
        with self.assertRaises(ValueError):
            fluctuation.effective_action([], [], 5)
        with self.assertRaises(ValueError):
            fluctuation.effective_action([], [], 2, 1)


class TheDeclarationIsCheckedTest(unittest.TestCase):
    """Every malformed declaration is refused by name."""

    def test_a_non_square_carrier_is_refused(self):
        declaration = cob.DressedFluctuationDeclaration()
        declaration.carrier_dimension = 2
        declaration.carrier = [1.0, 0.0, 0.0]
        with self.assertRaises(ValueError):
            cob.DressedFluctuation(declaration)

    def test_a_wrong_count_of_second_derivatives_is_refused(self):
        carrier = np.eye(2, dtype=complex)
        coupling = np.zeros((2, 2), dtype=complex)
        declaration = _declaration(carrier, [coupling, coupling])
        declaration.second_derivatives = [_flat(coupling)]
        with self.assertRaises(ValueError):
            cob.DressedFluctuation(declaration)

    def test_more_occupied_modes_than_the_carrier_has_is_refused(self):
        carrier = np.eye(2, dtype=complex)
        with self.assertRaises(ValueError):
            cob.DressedFluctuation(
                _declaration(carrier, [np.zeros((2, 2), dtype=complex)],
                             occupied=3))

    def test_a_negative_broadening_is_refused(self):
        carrier = np.eye(2, dtype=complex)
        with self.assertRaises(ValueError):
            cob.DressedFluctuation(
                _declaration(carrier, [np.zeros((2, 2), dtype=complex)],
                             occupied=1, broadening=-0.1))


if __name__ == "__main__":
    unittest.main()
