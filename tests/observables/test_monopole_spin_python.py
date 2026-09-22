# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""Exact fixtures for spin from an odd Dirac monopole (issue #1196):
MonopoleSupport (the monopole number of a closed cut, the twisted
operators, the projective representation D_k(g) and its cocycle, and the
spinor bands it protects) and SharpSpin (the two sharp-spin
eigen-equations on a superposition of determinants).

Acceptance coverage (ticket #1196):

* an odd-monopole support carries a j = 1/2 doublet under D_k(g) and an
  even one does not: on the tetrahedron at monopole number +1, -1 and 3 the
  cocycle class is nontrivial, the rotation-averaged twisted edge Laplacian
  splits into three rank-two irreducible bands, and the band at eigenvalue
  4 is exactly the coexact sector ker((delta_0^U)^dagger) -- the genuine
  j = 1/2 doublet.  At monopole number 0 and 2 the class is trivial, the
  commuting order-two rotations commute as operators, and no band is a
  spinor doublet;
* the sharp-spin read decides a determinant superposition the variance test
  cannot: a right state that is a genuine superposition of determinants and
  a j = 1/2 eigenstate passes both eigen-equations, while a right state
  contaminated by a j = 3/2 component paired with a pure left eigenvector
  has expectation exactly 3/4 and complex variance exactly zero -- so the
  variance criterion would accept it -- yet fails the right eigen-equation
  outright.

Values pinned here:

* the twisted vertex Laplacian (delta_0^U)^dagger delta_0^U on the
  tetrahedron has spectrum {0, 4 (three times)} at monopole number 0 and
  {2 (three times), 6} at monopole number 2, the 1 + 3 decomposition;
* at monopole number +-1 its four vertex modes split into two doublets, at
  3 - sqrt(3) and 3 + sqrt(3).  That pair is derived here from the operator
  the file header defines: its trace is 12, three times the vertex count,
  and the trace of its square fixes the splitting;
* the rotation-averaged twisted edge Laplacian is exactly 4 I at zero flux
  and has spectrum {4 - 2/sqrt(3), 4, 4 + 2/sqrt(3)}, each doubled, at unit
  monopole number, with the j = 1/2 doublet at 4.

Independent references: the tetrahedral connection, the twisted coboundary,
the signed permutation action and the three-carrier J^2 are each rebuilt in
numpy from the definitions and compared against the kernel, rather than the
kernel being compared against itself.

Exactness bar: algebraic identities (unit-modulus holonomies, the
commutator phase, the intertwining residual, the cocycle scalar residual)
are compared at double round-off (~1e-14).  Spectra, which pass through a
self-adjoint eigensolver, are compared at 1e-10.
"""

from __future__ import annotations

import itertools
import math
import unittest

import numpy as np

import tessera

MonopoleSupport = tessera.MonopoleSupport
SharpSpin = tessera.SharpSpin

SQRT3 = math.sqrt(3.0)
EDGES = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
EDGE_INDEX = {e: i for i, e in enumerate(EDGES)}
OUTWARD_FACES = [(1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1)]


# ─── independent references ────────────────────────────────────────────────

def reference_connection(mu):
    """U_e on the six stored tetrahedron edges, gauge-fixed on the spanning
    tree from vertex 0, with every outward face holonomy exp(2 pi i mu / 4).
    Built here from the three face equations rather than from the kernel."""
    angle = 2.0 * math.pi * mu / 4.0
    return np.array([1.0, 1.0, 1.0,
                     np.exp(-1j * angle),
                     np.exp(1j * angle),
                     np.exp(-1j * angle)], dtype=complex)


def reference_transport(connection, x, y):
    if x < y:
        return connection[EDGE_INDEX[(x, y)]]
    return 1.0 / connection[EDGE_INDEX[(y, x)]]


def reference_coboundary(connection):
    """(delta_0^U f)[x, y] = U_xy f(y) - f(x) as a 6 x 4 matrix."""
    out = np.zeros((6, 4), dtype=complex)
    for (x, y), e in EDGE_INDEX.items():
        out[e, x] = -1.0
        out[e, y] = connection[e]
    return out


def even_permutations():
    """The twelve even permutations of four letters, the rotation group T."""
    out = []
    for perm in itertools.permutations(range(4)):
        inversions = sum(1 for i in range(4) for j in range(i + 1, 4)
                         if perm[i] > perm[j])
        if inversions % 2 == 0:
            out.append(list(perm))
    return out


def reference_signed_permutation(rotation):
    """P_g on edge cochains: the geometric signed permutation, before the
    gauge dressing."""
    out = np.zeros((6, 6), dtype=complex)
    for (x, y), e in EDGE_INDEX.items():
        gx, gy = rotation[x], rotation[y]
        if gx < gy:
            out[EDGE_INDEX[(gx, gy)], e] = 1.0
        else:
            out[EDGE_INDEX[(gy, gx)], e] = -1.0
    return out


def sorted_spectrum(matrix):
    return np.sort(np.linalg.eigvalsh(matrix).real)


def group_eigenvalues(values, tolerance=1e-7):
    """Distinct eigenvalues with their multiplicities, in ascending order."""
    out = []
    for value in np.sort(values):
        if out and abs(out[-1][0] - value) <= tolerance:
            out[-1] = (out[-1][0], out[-1][1] + 1)
        else:
            out.append((value, 1))
    return out


# ─── the monopole number ───────────────────────────────────────────────────

class TestMonopoleNumber(unittest.TestCase):
    """The charge of the U(1) part of the connection through a closed cut."""

    def test_the_tetrahedral_connection_matches_the_reference(self) -> None:
        for mu in (-2, -1, 0, 1, 2, 3):
            support = MonopoleSupport.tetrahedron(mu)
            self.assertEqual([tuple(e) for e in support.edges], EDGES)
            self.assertEqual([tuple(f) for f in support.faces],
                             OUTWARD_FACES)
            self.assertLessEqual(
                np.max(np.abs(np.asarray(support.connection)
                              - reference_connection(mu))), 1e-14)

    def test_every_outward_face_carries_the_declared_holonomy(self) -> None:
        for mu in (-2, -1, 0, 1, 2, 3):
            read = MonopoleSupport.tetrahedron(mu).monopoleNumber()
            expected = np.exp(2j * math.pi * mu / 4.0)
            for holonomy in read.face_holonomies:
                self.assertLessEqual(abs(holonomy - expected), 1e-14)

    def test_the_number_and_its_parity(self) -> None:
        """mu = 3 and mu = -1 name the same connection, since
        exp(3 i pi / 2) = exp(-i pi / 2); both read as -1 and both are
        odd."""
        expected = {-2: (2, False), -1: (-1, True), 0: (0, False),
                    1: (1, True), 2: (2, False), 3: (-1, True)}
        for mu, (number, odd) in expected.items():
            read = MonopoleSupport.tetrahedron(mu).monopoleNumber()
            self.assertEqual(read.monopole_number, number, msg=f"mu={mu}")
            self.assertEqual(read.odd, odd, msg=f"mu={mu}")
            self.assertTrue(read.bundle)
            self.assertLessEqual(read.integrality_residual, 1e-14)
            self.assertLessEqual(read.unit_modulus_residual, 1e-14)
            self.assertTrue(read.certificate.holds())

    def test_the_total_flux_is_two_pi_times_the_number(self) -> None:
        for mu in (-1, 0, 1, 2):
            read = MonopoleSupport.tetrahedron(mu).monopoleNumber()
            self.assertLessEqual(
                abs(read.total_flux - 2.0 * math.pi * read.monopole_number),
                1e-12)

    def test_the_branch_cut_is_reported_where_it_is_reached(self) -> None:
        """At even monopole number every face holonomy is exactly -1, so
        every face flux sits at the end of the principal interval.  At odd
        monopole number the fluxes are +-pi/2 and stay well inside it."""
        even = MonopoleSupport.tetrahedron(2).monopoleNumber()
        self.assertTrue(even.on_branch_cut)
        self.assertLessEqual(even.branch_margin, 1e-14)
        odd = MonopoleSupport.tetrahedron(1).monopoleNumber()
        self.assertFalse(odd.on_branch_cut)
        self.assertAlmostEqual(odd.branch_margin, math.pi / 2.0, places=12)

    def test_a_connection_outside_the_u1_domain_is_refused(self) -> None:
        connection = list(reference_connection(1))
        connection[0] = 2.0 + 0.0j
        with self.assertRaises(ValueError):
            MonopoleSupport(4, [list(e) for e in EDGES],
                            [list(f) for f in OUTWARD_FACES], connection)

    def test_u1_part_is_the_explicit_way_into_the_domain(self) -> None:
        scaled = [3.0 * value for value in reference_connection(1)]
        restored = MonopoleSupport.u1Part(scaled)
        support = MonopoleSupport(4, [list(e) for e in EDGES],
                                  [list(f) for f in OUTWARD_FACES], restored)
        self.assertEqual(support.monopoleNumber().monopole_number, 1)
        with self.assertRaises(ValueError):
            MonopoleSupport.u1Part([0.0 + 0.0j])

    def test_a_face_whose_boundary_is_not_an_edge_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            MonopoleSupport(4, [[0, 1], [0, 2], [1, 2]], [[0, 1, 3]],
                            [1.0 + 0j] * 3)

    def test_an_unsorted_edge_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            MonopoleSupport(4, [[1, 0]], [], [1.0 + 0j])


# ─── the twisted operators ─────────────────────────────────────────────────

class TestTwistedOperators(unittest.TestCase):
    """delta_0^U, its Laplacians and the coexact sector."""

    def test_the_coboundary_matches_the_reference(self) -> None:
        for mu in (0, 1, 2):
            support = MonopoleSupport.tetrahedron(mu)
            self.assertLessEqual(
                np.max(np.abs(support.twistedCoboundary()
                              - reference_coboundary(
                                  reference_connection(mu)))), 1e-14)

    def test_the_vertex_laplacian_is_degree_minus_the_connection(
            self) -> None:
        for mu in (0, 1, 2):
            support = MonopoleSupport.tetrahedron(mu)
            connection = reference_connection(mu)
            expected = 3.0 * np.eye(4, dtype=complex)
            for x in range(4):
                for y in range(4):
                    if x != y:
                        expected[x, y] -= reference_transport(connection, x, y)
            self.assertLessEqual(
                np.max(np.abs(support.vertexLaplacian() - expected)), 1e-13)

    def test_vertex_spectrum_at_even_monopole_number(self) -> None:
        """The 1 + 3 decomposition the whitepaper names."""
        flat = sorted_spectrum(
            MonopoleSupport.tetrahedron(0).vertexLaplacian())
        np.testing.assert_allclose(flat, [0.0, 4.0, 4.0, 4.0], atol=1e-10)
        two = sorted_spectrum(
            MonopoleSupport.tetrahedron(2).vertexLaplacian())
        np.testing.assert_allclose(two, [2.0, 2.0, 2.0, 6.0], atol=1e-10)

    def test_vertex_spectrum_at_odd_monopole_number_is_two_doublets(
            self) -> None:
        for mu in (1, -1, 3):
            values = sorted_spectrum(
                MonopoleSupport.tetrahedron(mu).vertexLaplacian())
            np.testing.assert_allclose(
                values,
                [3.0 - SQRT3, 3.0 - SQRT3, 3.0 + SQRT3, 3.0 + SQRT3],
                atol=1e-10, err_msg=f"mu={mu}")
            # The same pair from the two traces alone: tr L = 3 * 4 = 12 and
            # tr L^2 = 9 * 4 + 12, since the off-diagonal entries all have
            # unit modulus.  Two doublets at 3 -+ s then force s^2 = 3.
            matrix = MonopoleSupport.tetrahedron(mu).vertexLaplacian()
            self.assertLessEqual(abs(np.trace(matrix) - 12.0), 1e-12)
            self.assertLessEqual(
                abs(np.trace(matrix @ matrix) - (36.0 + 12.0)), 1e-12)

    def test_the_edge_laplacian_is_four_times_the_identity_at_zero_flux(
            self) -> None:
        support = MonopoleSupport.tetrahedron(0)
        self.assertLessEqual(
            np.max(np.abs(support.edgeLaplacian() - 4.0 * np.eye(6))), 1e-12)

    def test_the_coexact_sector_is_a_projector_of_the_expected_rank(
            self) -> None:
        """At zero flux the coexact sector is the rank-three anchorable
        triplet; with flux the zero mode is lifted, the exact sector fills
        out to rank four and the coexact sector drops to rank two."""
        for mu, rank in ((0, 3), (1, 2), (2, 2), (-1, 2)):
            projector = MonopoleSupport.tetrahedron(mu).coexactProjector()
            self.assertLessEqual(
                np.max(np.abs(projector @ projector - projector)), 1e-10,
                msg=f"mu={mu}")
            self.assertLessEqual(
                np.max(np.abs(projector - projector.conj().T)), 1e-12)
            self.assertAlmostEqual(np.trace(projector).real, float(rank),
                                   places=8, msg=f"mu={mu}")


# ─── the projective representation and its cocycle ─────────────────────────

class TestProjectiveRepresentation(unittest.TestCase):
    """D_k(g) = rho_k(u_g) P_g, its intertwining, and the cocycle varpi."""

    def test_the_rotation_group_is_the_twelve_even_permutations(self) -> None:
        rotations = [list(r) for r in MonopoleSupport.tetrahedralRotations()]
        self.assertEqual(len(rotations), 12)
        self.assertEqual(rotations[0], [0, 1, 2, 3])
        self.assertEqual(sorted(rotations), sorted(even_permutations()))

    def test_the_configuration_is_symmetric_up_to_gauge(self) -> None:
        for mu in (0, 1, 2, 3):
            support = MonopoleSupport.tetrahedron(mu)
            for rotation in MonopoleSupport.tetrahedralRotations():
                read = support.gaugeCompensation(rotation)
                self.assertTrue(read.symmetric, msg=f"mu={mu} g={rotation}")
                self.assertLessEqual(read.residual, 1e-13)
                self.assertLessEqual(abs(read.gauge[0] - 1.0), 0.0)
                self.assertLessEqual(
                    np.max(np.abs(np.abs(np.asarray(read.gauge)) - 1.0)),
                    1e-13)

    def test_a_map_that_is_not_a_symmetry_is_refused(self) -> None:
        """A transposition of a tetrahedron is a reflection, not a rotation,
        but it still carries edges to edges, so the refusal that matters is
        of a map that is not a permutation at all, and of one that leaves
        the edge set."""
        support = MonopoleSupport.tetrahedron(1)
        with self.assertRaises(ValueError):
            support.gaugeCompensation([0, 0, 1, 2])
        path = MonopoleSupport(4, [[0, 1], [1, 2], [2, 3]], [],
                               [1.0 + 0j] * 3)
        with self.assertRaises(ValueError):
            path.gaugeCompensation([1, 0, 2, 3])

    def test_the_vertex_action_is_the_permutation_dressed_by_the_gauge(
            self) -> None:
        support = MonopoleSupport.tetrahedron(1)
        for rotation in MonopoleSupport.tetrahedralRotations():
            gauge = np.asarray(support.gaugeCompensation(rotation).gauge)
            expected = np.zeros((4, 4), dtype=complex)
            for x in range(4):
                expected[rotation[x], x] = 1.0 / gauge[rotation[x]]
            self.assertLessEqual(
                np.max(np.abs(support.vertexRepresentation(rotation)
                              - expected)), 1e-13)

    def test_the_edge_action_dresses_the_signed_permutation(self) -> None:
        """At zero flux the gauge is trivial and D_1(g) is exactly the
        geometric signed permutation."""
        support = MonopoleSupport.tetrahedron(0)
        for rotation in MonopoleSupport.tetrahedralRotations():
            self.assertLessEqual(
                np.max(np.abs(support.edgeRepresentation(rotation)
                              - reference_signed_permutation(rotation))),
                1e-13)

    def test_both_actions_are_unitary(self) -> None:
        for mu in (0, 1, 2, 3):
            support = MonopoleSupport.tetrahedron(mu)
            for rotation in MonopoleSupport.tetrahedralRotations():
                for action in (support.vertexRepresentation(rotation),
                               support.edgeRepresentation(rotation)):
                    identity = np.eye(action.shape[0])
                    self.assertLessEqual(
                        np.max(np.abs(action.conj().T @ action - identity)),
                        1e-13)

    def test_d1_intertwines_the_twisted_coboundary(self) -> None:
        for mu in (0, 1, 2, 3):
            support = MonopoleSupport.tetrahedron(mu)
            for rotation in MonopoleSupport.tetrahedralRotations():
                self.assertLessEqual(
                    support.intertwiningResidual(rotation), 1e-13,
                    msg=f"mu={mu} g={rotation}")

    def test_the_cocycle_class_is_nontrivial_exactly_at_odd_flux(self) -> None:
        group = MonopoleSupport.tetrahedralRotations()
        for mu, nontrivial in ((0, False), (1, True), (2, False), (3, True),
                               (-1, True), (-2, False)):
            support = MonopoleSupport.tetrahedron(mu)
            for degree in (0, 1):
                read = support.cocycle(group, degree)
                self.assertEqual(read.group_order, 12)
                self.assertLessEqual(read.scalar_residual, 1e-13,
                                     msg=f"mu={mu} degree={degree}")
                self.assertTrue(read.certificate.holds())
                self.assertEqual(read.nontrivial, nontrivial,
                                 msg=f"mu={mu} degree={degree}")

    def test_the_commutator_phase_is_the_quaternion_relation(self) -> None:
        """Two commuting order-two rotations anticommute as operators at odd
        monopole number -- D(a)D(b)D(a)^-1 D(b)^-1 = -1 -- and commute at
        even."""
        group = MonopoleSupport.tetrahedralRotations()
        for mu, phase in ((0, 1.0), (1, -1.0), (2, 1.0), (3, -1.0)):
            read = MonopoleSupport.tetrahedron(mu).cocycle(group, 1)
            self.assertLessEqual(abs(read.commutator_phase - phase), 1e-13,
                                 msg=f"mu={mu}")
            first, second = read.commutator_pair
            a, b = list(group[first]), list(group[second])
            self.assertEqual([a[b[i]] for i in range(4)],
                             [b[a[i]] for i in range(4)],
                             msg="the witnessing pair must commute")

    def test_the_cocycle_values_are_roots_of_unity(self) -> None:
        group = MonopoleSupport.tetrahedralRotations()
        trivial = MonopoleSupport.tetrahedron(0).cocycle(group, 1)
        self.assertEqual(len(trivial.values), 1)
        self.assertLessEqual(abs(trivial.values[0] - 1.0), 1e-13)
        odd = MonopoleSupport.tetrahedron(1).cocycle(group, 1)
        self.assertEqual(len(odd.values), 3)
        for value in odd.values:
            self.assertLessEqual(abs(abs(value) - 1.0), 1e-13)
            self.assertLessEqual(abs(value ** 4 - 1.0), 1e-12)

    def test_a_set_that_is_not_a_group_is_refused(self) -> None:
        support = MonopoleSupport.tetrahedron(1)
        with self.assertRaises(ValueError):
            support.cocycle([[0, 1, 2, 3], [1, 2, 0, 3]])
        with self.assertRaises(ValueError):
            support.cocycle([])


# ─── the spinor bands ──────────────────────────────────────────────────────

class TestSpinorBands(unittest.TestCase):
    """The symmetry-protected bands of the rotation-averaged twisted edge
    Laplacian, and the j = 1/2 doublet among them."""

    def averaged(self, mu):
        support = MonopoleSupport.tetrahedron(mu)
        group = MonopoleSupport.tetrahedralRotations()
        return support, group, support.rotationAveragedEdgeOperator(
            support.edgeLaplacian(), group)

    def test_the_average_is_four_times_the_identity_at_zero_flux(self) -> None:
        _, _, averaged = self.averaged(0)
        self.assertLessEqual(np.max(np.abs(averaged - 4.0 * np.eye(6))),
                             1e-11)

    def test_the_average_is_the_whitepaper_edge_spectrum_at_unit_flux(
            self) -> None:
        for mu in (1, -1, 3):
            _, _, averaged = self.averaged(mu)
            values = sorted_spectrum(averaged)
            expected = [4.0 - 2.0 / SQRT3, 4.0 - 2.0 / SQRT3, 4.0, 4.0,
                        4.0 + 2.0 / SQRT3, 4.0 + 2.0 / SQRT3]
            np.testing.assert_allclose(values, expected, atol=1e-10,
                                       err_msg=f"mu={mu}")

    def test_the_average_is_rotation_invariant(self) -> None:
        for mu in (0, 1, 2):
            support, group, averaged = self.averaged(mu)
            for rotation in group:
                action = support.edgeRepresentation(rotation)
                self.assertLessEqual(
                    np.max(np.abs(action @ averaged
                                  - averaged @ action)), 1e-11,
                    msg=f"mu={mu}")

    def test_an_odd_monopole_support_carries_a_half_integer_doublet(
            self) -> None:
        for mu in (1, -1, 3):
            support = MonopoleSupport.tetrahedron(mu)
            read = support.spinRead(MonopoleSupport.tetrahedralRotations())
            self.assertTrue(read.monopole.odd, msg=f"mu={mu}")
            self.assertTrue(read.cocycle.nontrivial, msg=f"mu={mu}")
            self.assertTrue(read.half_integer_doublet, msg=f"mu={mu}")
            self.assertTrue(read.certificate.holds())
            self.assertEqual(len(read.bands), 3, msg=f"mu={mu}")
            for band in read.bands:
                self.assertEqual(band.dimension, 2, msg=f"mu={mu}")
                self.assertLessEqual(band.invariance_residual, 1e-11)
                self.assertLessEqual(abs(band.irreducibility_score - 1.0),
                                     1e-9)
                self.assertTrue(band.spinor_doublet)
            doublet = read.bands[read.doublet_index]
            self.assertAlmostEqual(doublet.eigenvalue, 4.0, places=9)
            self.assertTrue(doublet.coexact)
            self.assertLessEqual(doublet.coexact_residual, 1e-10)
            # The other two doublets are the images of the vertex doublets
            # and lie in the exact sector, so they are NOT the j = 1/2 one.
            for index, band in enumerate(read.bands):
                if index != read.doublet_index:
                    self.assertFalse(band.coexact)

    def test_an_even_monopole_support_does_not(self) -> None:
        for mu in (0, 2, -2):
            support = MonopoleSupport.tetrahedron(mu)
            read = support.spinRead(MonopoleSupport.tetrahedralRotations())
            self.assertFalse(read.monopole.odd, msg=f"mu={mu}")
            self.assertFalse(read.cocycle.nontrivial, msg=f"mu={mu}")
            self.assertFalse(read.half_integer_doublet, msg=f"mu={mu}")
            self.assertEqual(read.doublet_index, len(read.bands))
            for band in read.bands:
                self.assertFalse(band.spinor_doublet, msg=f"mu={mu}")

    def test_the_even_case_fails_on_more_than_the_class_alone(self) -> None:
        """At monopole number two there IS a rank-two coexact band, so the
        verdict cannot rest on the band's rank: it is reducible as a
        projective representation (score two, not one), which is what a
        trivial class produces."""
        support = MonopoleSupport.tetrahedron(2)
        read = support.spinRead(MonopoleSupport.tetrahedralRotations())
        coexact = [b for b in read.bands if b.coexact]
        self.assertEqual(len(coexact), 1)
        self.assertEqual(coexact[0].dimension, 2)
        self.assertGreater(coexact[0].irreducibility_score, 1.5)

    def test_a_non_hermitian_operator_is_refused(self) -> None:
        support = MonopoleSupport.tetrahedron(1)
        group = MonopoleSupport.tetrahedralRotations()
        skewed = np.eye(6, dtype=complex)
        skewed[0, 1] = 1.0
        with self.assertRaises(ValueError):
            support.spinorBands(skewed, group, True)


# ─── the sharp-spin eigen-equations ────────────────────────────────────────

def reference_spin_matrices(carriers):
    """J_a = I_carriers (x) sigma_a / 2 on the mode order (carrier, spin),
    assembled here from the Pauli matrices directly."""
    sx = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex) / 2.0
    sy = np.array([[0.0, -1j], [1j, 0.0]], dtype=complex) / 2.0
    sz = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex) / 2.0
    return [np.kron(np.eye(carriers), s) for s in (sx, sy, sz)]


def one_per_carrier_basis(carriers):
    """The determinants with exactly one occupied mode per carrier: the
    sector in which the tetrahedron's three doublets 2, 2' and 2'' each hold
    one particle."""
    return [[2 * c + s for c, s in enumerate(spins)]
            for spins in itertools.product((0, 1), repeat=carriers)]


class TestSharpSpin(unittest.TestCase):
    """The two eigen-equations on a superposition of determinants."""

    CARRIERS = 3
    MODES = 6

    def spin_matrices(self):
        return SharpSpin.doubletSpinMatrices(self.CARRIERS)

    def test_the_spin_matrices_match_the_reference(self) -> None:
        produced = self.spin_matrices()
        for mine, theirs in zip(produced, reference_spin_matrices(
                self.CARRIERS)):
            self.assertLessEqual(np.max(np.abs(mine - theirs)), 0.0)

    def test_the_applied_action_equals_the_dense_operator(self) -> None:
        matrices = self.spin_matrices()
        dense = SharpSpin.totalSpinSquaredMatrix(matrices)
        self.assertEqual(dense.shape, (64, 64))
        self.assertLessEqual(np.max(np.abs(dense - dense.conj().T)), 1e-14)
        generator = np.random.default_rng(20261196)
        state = (generator.normal(size=64) + 1j * generator.normal(size=64))
        applied = SharpSpin.applyTotalSpinSquared(matrices, state)
        self.assertLessEqual(np.max(np.abs(applied - dense @ state)), 1e-12)

    def test_determinants_are_wedges_with_the_permutation_sign(self) -> None:
        forward = SharpSpin.determinant([0, 2, 4], self.MODES)
        swapped = SharpSpin.determinant([2, 0, 4], self.MODES)
        self.assertLessEqual(np.max(np.abs(forward + swapped)), 0.0)
        self.assertAlmostEqual(np.linalg.norm(forward), 1.0, places=14)
        self.assertEqual(int(np.count_nonzero(forward)), 1)

    def test_a_repeated_mode_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            SharpSpin.determinant([0, 0], self.MODES)
        with self.assertRaises(ValueError):
            SharpSpin.determinant([0, 99], self.MODES)

    def test_the_superposition_is_linear_in_its_amplitudes(self) -> None:
        occupations = [[0, 2, 4], [1, 3, 5]]
        amplitudes = [0.6 + 0.2j, -0.3 + 0.9j]
        state = SharpSpin.determinantSuperposition(occupations, amplitudes,
                                                   self.MODES)
        expected = sum(
            a * SharpSpin.determinant(o, self.MODES)
            for a, o in zip(amplitudes, occupations))
        self.assertLessEqual(np.max(np.abs(state - expected)), 0.0)
        with self.assertRaises(ValueError):
            SharpSpin.determinantSuperposition(occupations, amplitudes[:1],
                                               self.MODES)

    def sector(self):
        """The 8-dimensional one-per-carrier sector and the J^2 block on
        it."""
        matrices = self.spin_matrices()
        dense = SharpSpin.totalSpinSquaredMatrix(matrices)
        basis = np.column_stack([
            SharpSpin.determinant(modes, self.MODES)
            for modes in one_per_carrier_basis(self.CARRIERS)])
        block = basis.conj().T @ dense @ basis
        values, vectors = np.linalg.eigh(block)
        return matrices, basis, values, vectors

    def test_the_one_per_carrier_sector_holds_spin_half_twice(self) -> None:
        """1/2 (x) 1/2 (x) 1/2 = 3/2 + 1/2 + 1/2: four states at j(j+1) =
        15/4 and four at 3/4, the latter being two copies of the doublet."""
        _, _, values, _ = self.sector()
        self.assertEqual(
            [(round(v, 9), n) for v, n in group_eigenvalues(values)],
            [(0.75, 4), (3.75, 4)])

    def test_a_genuine_eigenstate_passes_both_equations(self) -> None:
        matrices, basis, _, vectors = self.sector()
        state = basis @ vectors[:, 0]
        read = SharpSpin.read(matrices, state, state.conj())
        self.assertTrue(read.sharp)
        self.assertLessEqual(read.right_residual, 1e-12)
        self.assertLessEqual(read.left_residual, 1e-12)
        self.assertAlmostEqual(read.expectation.real, 0.75, places=12)
        self.assertLessEqual(abs(read.variance), 1e-12)
        self.assertTrue(read.certificate.holds())
        # The acceptance clause: this is a genuine superposition of
        # determinants, not a single Slater state, so no quasi-free
        # covariance carries it.
        self.assertGreater(read.determinant_count, 1)

    def test_the_variance_test_cannot_decide_an_isotropic_cancellation(
            self) -> None:
        """A right state contaminated with a j = 3/2 component, paired with a
        pure j = 1/2 LEFT eigenvector, has expectation exactly 3/4 and
        complex variance exactly zero, because the contaminating component
        is annihilated by the left vector.  The variance criterion would
        accept it; the right eigen-equation refuses it outright."""
        matrices, basis, _, vectors = self.sector()
        half = basis @ vectors[:, 0]
        three_half = basis @ vectors[:, 7]
        contaminated = half + (0.4 + 0.3j) * three_half
        read = SharpSpin.read(matrices, contaminated, half.conj())
        self.assertAlmostEqual(read.expectation.real, 0.75, places=12)
        self.assertLessEqual(abs(read.expectation.imag), 1e-12)
        self.assertLessEqual(abs(read.variance), 1e-12)
        self.assertTrue(read.variance_would_accept)
        self.assertFalse(read.sharp)
        self.assertGreater(read.right_residual, 0.1)
        self.assertLessEqual(read.left_residual, 1e-12)
        self.assertFalse(read.certificate.holds())

    def test_a_spin_three_halves_state_is_not_sharp_at_three_quarters(
            self) -> None:
        matrices, basis, _, vectors = self.sector()
        state = basis @ vectors[:, 7]
        read = SharpSpin.read(matrices, state, state.conj())
        self.assertFalse(read.sharp)
        self.assertAlmostEqual(read.expectation.real, 3.75, places=12)
        self.assertFalse(read.variance_would_accept)
        # It IS sharp against its own eigenvalue.
        at_own = SharpSpin.read(matrices, state, state.conj(),
                                targetEigenvalue=3.75)
        self.assertTrue(at_own.sharp)

    def test_the_verdict_does_not_move_when_a_state_is_rescaled(self) -> None:
        matrices, basis, _, vectors = self.sector()
        state = basis @ vectors[:, 0]
        plain = SharpSpin.read(matrices, state, state.conj())
        scaled = SharpSpin.read(matrices, 17.0 * state,
                                (0.05 + 0.02j) * state.conj())
        self.assertEqual(plain.sharp, scaled.sharp)
        self.assertLessEqual(abs(plain.right_residual - scaled.right_residual),
                             1e-12)
        self.assertLessEqual(
            abs(plain.expectation - scaled.expectation), 1e-12)

    def test_degenerate_input_is_refused(self) -> None:
        matrices = self.spin_matrices()
        state = SharpSpin.determinant([0, 2, 4], self.MODES)
        with self.assertRaises(ValueError):
            SharpSpin.read(matrices, state, np.zeros(64, dtype=complex))
        with self.assertRaises(ValueError):
            SharpSpin.read(matrices, state, np.zeros(32, dtype=complex))
        with self.assertRaises(ValueError):
            SharpSpin.applyTotalSpinSquared(matrices,
                                            np.zeros(63, dtype=complex))

    def test_a_single_determinant_reports_one_determinant(self) -> None:
        matrices = self.spin_matrices()
        state = SharpSpin.determinant([0, 2, 4], self.MODES)
        read = SharpSpin.read(matrices, state, state.conj())
        self.assertEqual(read.determinant_count, 1)
        # |up up up> is the top of the j = 3/2 quartet, not a j = 1/2 state.
        self.assertAlmostEqual(read.expectation.real, 3.75, places=12)


if __name__ == "__main__":
    unittest.main()
