# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The complex-bilinear covariance, evolved through its two frames (#1186).

Whitepaper Section 7: for N occupied right modes and their matched algebraic
duals,

    |Xi_R> = phi_1 ^ ... ^ phi_N,   <Xi_L| = phi~_1 ^ ... ^ phi~_N,
    Phi~^T Phi = I_N,   Gamma = Phi Phi~^T,   Gamma^2 = Gamma,   tr Gamma = N,

and for an arbitrary complex one-particle generator the two frames evolve by
i dPhi/dt = h Phi and -i dPhi~^T/dt = Phi~^T h, so i dGamma/dt = [h, Gamma]
with no h^dagger.

The dense reference is the exterior algebra itself: independent numpy
Jordan-Wigner chains build |Xi_R> and |Xi_L> as Fock vectors by applying
LINEAR smeared creations to the vacuum, the left state is read by the plain
transpose (<Xi_L| X |Xi_R> = psi_L^T X psi_R, the contraction by the dual
mode, never a conjugate), and the evolution is the dense Fock exponential of
dGamma(h) on each side. Every covariance quantity is compared against it.

Acceptance (#1186): for a complex-symmetric h the evolved Gamma stays
idempotent and reproduces the dense exterior-algebra evolution; for a
Hermitian h the transpose path equals the present (Hermitian-adjoint) path.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

try:
    from tessera.quantum import CovarianceDual, CovarianceState
    HAVE_QUANTUM = True
except ImportError:
    HAVE_QUANTUM = False

import tessera

cob = tessera.cobordism

try:
    from scipy.linalg import expm
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

MACHINE = 1e-12     # exact finite sums on small, well-conditioned fixtures


# ─── independent dense exterior-algebra reference ──────────────────────────

_S_MINUS = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
_Z = np.diag([1.0, -1.0]).astype(complex)


def jw_annihilation(mode: int, n_modes: int) -> np.ndarray:
    """Dense Jordan-Wigner a_mode on the n(b) = sum b_i 2^i basis (mode 0 =
    least-significant bit; Z string strictly below). Real, so a_i^T = a_i^+."""
    op = np.eye(1, dtype=complex)
    for m in range(n_modes - 1, -1, -1):
        if m > mode:
            factor = np.eye(2, dtype=complex)
        elif m == mode:
            factor = _S_MINUS
        else:
            factor = _Z
        op = np.kron(op, factor)
    return op


class BilinearFock:
    """Dense Fock reference over M modes with the transpose pairing: a+(v) =
    sum v_i a_i^+ and a~(w) = sum w_i a_i are both LINEAR, and a left state is
    read by psi_L^T (no conjugation anywhere)."""

    def __init__(self, n_modes: int):
        self.n = n_modes
        self.a = [jw_annihilation(m, n_modes) for m in range(n_modes)]
        self.adag = [op.T.copy() for op in self.a]   # real JW: a^T = a^+

    def creation(self, v):
        return sum(v[i] * self.adag[i] for i in range(self.n))

    def contraction(self, w):
        """a~(w) = sum_i w_i a_i: the contraction by the dual mode."""
        return sum(w[i] * self.a[i] for i in range(self.n))

    def wedge(self, frame):
        """a+(f_1) ... a+(f_N) |0> for the COLUMNS of `frame`."""
        psi = np.zeros(2 ** self.n, dtype=complex)
        psi[0] = 1.0
        for k in range(frame.shape[1] - 1, -1, -1):
            psi = self.creation(frame[:, k]) @ psi
        return psi

    def dgamma(self, one_particle):
        return sum(one_particle[i, j] * (self.adag[i] @ self.a[j])
                   for i in range(self.n) for j in range(self.n))

    def parity(self):
        op = np.eye(2 ** self.n, dtype=complex)
        for m in range(self.n):
            op = op @ (np.eye(2 ** self.n) - 2 * self.adag[m] @ self.a[m])
        return op

    @staticmethod
    def pair(psi_l, op, psi_r):
        """<Xi_L| op |Xi_R> / <Xi_L|Xi_R> in the transpose pairing."""
        return complex(psi_l @ (op @ psi_r)) / complex(psi_l @ psi_r)

    def two_point(self, psi_l, psi_r):
        """Gamma_ij = <Xi_L| a_j^+ a_i |Xi_R> / <Xi_L|Xi_R>."""
        g = np.zeros((self.n, self.n), dtype=complex)
        for i in range(self.n):
            for j in range(self.n):
                g[i, j] = self.pair(psi_l, self.adag[j] @ self.a[i], psi_r)
        return g


# ─── fixtures ──────────────────────────────────────────────────────────────

def cplx(rng, *shape):
    return rng.normal(size=shape) + 1j * rng.normal(size=shape)


def biorthogonal_pair(n_modes: int, n_particles: int, seed: int):
    """A random right frame and a matched left dual, Phi~^T Phi = I (a
    genuinely non-Hermitian pair: Phi~ is not conj(Phi))."""
    rng = np.random.default_rng(seed)
    phi = cplx(rng, n_modes, n_particles)
    raw = cplx(rng, n_modes, n_particles)
    left = raw @ np.linalg.inv(phi.T @ raw)
    return phi, left


def complex_symmetric(n: int, seed: int, anti_hermitian: float = 0.3):
    """h = A + i s B with A, B real symmetric: h = h^T, h != h^dagger."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n, n))
    b = rng.normal(size=(n, n))
    return ((a + a.T) / 2 + 1j * anti_hermitian * (b + b.T) / 2).astype(complex)


def random_hermitian(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    b = cplx(rng, n, n)
    return (b + b.conj().T) / 2


def orthonormal(n_modes: int, n_particles: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(cplx(rng, n_modes, n_particles))
    return q


def relative(a, b) -> float:
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)) /
                 max(1.0, np.linalg.norm(np.asarray(b))))


# ─── construction ──────────────────────────────────────────────────────────

@unittest.skipUnless(HAVE_QUANTUM, "tessera built without the quantum subsystem")
class TestBiorthogonalConstruction(unittest.TestCase):
    """Gamma = Phi Phi~^T from a matched pair, both frames carried."""

    def test_gamma_is_the_transpose_product_and_idempotent(self) -> None:
        phi, left = biorthogonal_pair(5, 2, seed=1)
        state = CovarianceState.fromBiorthogonalFrames(phi, left)
        self.assertEqual(state.dual(), CovarianceDual.Transpose)
        gamma = np.array(state.gamma())
        np.testing.assert_allclose(gamma, phi @ left.T, rtol=0, atol=1e-14)
        np.testing.assert_array_equal(np.array(state.rightFrame()), phi)
        np.testing.assert_array_equal(np.array(state.leftFrame()), left)
        self.assertLess(state.dualityDefect(), MACHINE)
        self.assertLess(state.purityDefect(), MACHINE)
        self.assertLess(abs(state.particleNumber() - 2.0), MACHINE)
        # An algebraic covariance, not a density matrix: not Hermitian.
        self.assertGreater(state.hermiticityDefect(), 0.1)

    def test_hermitian_adjoint_states_carry_no_frames(self) -> None:
        state = CovarianceState.fromSlaterFrame(orthonormal(4, 2, seed=2))
        self.assertEqual(state.dual(), CovarianceDual.HermitianAdjoint)
        self.assertEqual(np.array(state.rightFrame()).size, 0)
        self.assertEqual(np.array(state.leftFrame()).size, 0)
        self.assertTrue(math.isnan(state.dualityDefect()))

    def test_no_occupied_duals_is_the_vacuum(self) -> None:
        empty = np.zeros((4, 0), dtype=complex)
        state = CovarianceState.fromBiorthogonalFrames(empty, empty)
        np.testing.assert_array_equal(np.array(state.gamma()),
                                      np.zeros((4, 4)))
        self.assertEqual(state.wickParity().value, 1.0 + 0j)
        self.assertEqual(state.dualityDefect(), 0.0)

    def test_shapes_are_validated(self) -> None:
        phi, left = biorthogonal_pair(4, 2, seed=3)
        with self.assertRaises(ValueError):
            CovarianceState.fromBiorthogonalFrames(phi, left[:, :1])
        with self.assertRaises(ValueError):
            CovarianceState.fromBiorthogonalFrames(phi, left[:3, :])
        with self.assertRaises(ValueError):
            CovarianceState.fromBiorthogonalFrames(
                np.zeros((0, 1), dtype=complex), np.zeros((0, 1), dtype=complex))

    def test_an_unpaired_left_frame_is_reported_never_repaired(self) -> None:
        # Negative control: Phi~^T Phi != I is adopted verbatim and the
        # defect shows up in every certificate.
        rng = np.random.default_rng(4)
        phi = cplx(rng, 4, 2)
        left = cplx(rng, 4, 2)
        state = CovarianceState.fromBiorthogonalFrames(phi, left)
        np.testing.assert_allclose(np.array(state.gamma()), phi @ left.T,
                                   rtol=0, atol=1e-14)
        self.assertGreater(state.dualityDefect(), 0.1)
        self.assertGreater(state.purityDefect(), 0.1)
        self.assertFalse(state.purityCertificate(1e-9).holds())
        read = state.wickParity()
        self.assertAlmostEqual(read.residual, state.dualityDefect(), places=14)
        self.assertFalse(read.certificate.holds())


# ─── Wick reads against the dense biorthogonal reference ───────────────────

@unittest.skipUnless(HAVE_QUANTUM, "tessera built without the quantum subsystem")
class TestBiorthogonalWick(unittest.TestCase):
    """The biorthogonal Wick theorem: every polynomial left/right matrix
    element is a contraction of Gamma, at machine precision."""

    M, N, SEED = 5, 2, 11

    def setUp(self) -> None:
        self.phi, self.left = biorthogonal_pair(self.M, self.N, self.SEED)
        self.state = CovarianceState.fromBiorthogonalFrames(self.phi,
                                                            self.left)
        self.fock = BilinearFock(self.M)
        self.psi_r = self.fock.wedge(self.phi)
        self.psi_l = self.fock.wedge(self.left)

    def test_overlap_is_the_determinant_of_the_pairing(self) -> None:
        # <Xi_L|Xi_R> = det(Phi~^T Phi) = 1: the pairing, not a norm.
        self.assertLess(abs(complex(self.psi_l @ self.psi_r) - 1.0), MACHINE)
        # The Hermitian norm of the ket is NOT one (a negative control on the
        # reference itself): the pairing never uses it.
        self.assertGreater(abs(np.vdot(self.psi_r, self.psi_r) - 1.0), 0.1)

    def test_gamma_is_the_dense_transition_two_point_function(self) -> None:
        dense_gamma = self.fock.two_point(self.psi_l, self.psi_r)
        self.assertLess(relative(self.state.gamma(), dense_gamma), MACHINE)

    def test_occupations_parity_and_subset_parity(self) -> None:
        for mode in range(self.M):
            dense_val = self.fock.pair(
                self.psi_l, self.fock.adag[mode] @ self.fock.a[mode],
                self.psi_r)
            self.assertLess(abs(self.state.wickOccupation(mode).value -
                                dense_val), MACHINE)
        parity = self.fock.pair(self.psi_l, self.fock.parity(), self.psi_r)
        self.assertLess(abs(self.state.wickParity().value - parity), MACHINE)
        dim = 2 ** self.M
        sub = np.eye(dim, dtype=complex)
        for m in (0, 3):
            sub = sub @ (np.eye(dim) - 2 * self.fock.adag[m] @ self.fock.a[m])
        self.assertLess(abs(self.state.wickSubsetParity([0, 3]).value -
                            self.fock.pair(self.psi_l, sub, self.psi_r)),
                        MACHINE)

    def test_normal_ordered_determinants(self) -> None:
        f = self.fock
        for creators, annihilators in (([0, 2], [2, 0]), ([1, 4], [3, 0]),
                                       ([2], [4]), ([0, 1], [0, 1])):
            op = np.eye(2 ** self.M, dtype=complex)
            for c in creators:
                op = op @ f.adag[c]
            for a in reversed(annihilators):
                op = op @ f.a[a]
            dense_val = f.pair(self.psi_l, op, self.psi_r)
            read = self.state.wickNormalOrdered(creators, annihilators)
            self.assertLess(abs(read.value - dense_val), MACHINE)

    def test_transpose_gram_determinant_is_the_linear_smearing(self) -> None:
        rng = np.random.default_rng(12)
        v = cplx(rng, self.M, 2)
        w = cplx(rng, self.M, 2)
        f = self.fock
        op = (f.creation(v[:, 0]) @ f.creation(v[:, 1]) @
              f.contraction(w[:, 1]) @ f.contraction(w[:, 0]))
        dense_val = f.pair(self.psi_l, op, self.psi_r)
        read = self.state.wickTransposeGramDeterminant(v, w)
        scale = max(1.0, abs(dense_val))
        self.assertLess(abs(read.value - dense_val) / scale, MACHINE)
        gamma = np.array(self.state.gamma())
        self.assertLess(abs(read.value - np.linalg.det(w.T @ gamma @ v)) /
                        scale, MACHINE)
        # It differs from the conjugate-smeared read exactly by w -> conj(w).
        self.assertLess(
            abs(self.state.wickGramDeterminant(v, w.conj()).value -
                read.value) / scale, MACHINE)
        self.assertEqual(
            self.state.wickTransposeGramDeterminant(v, w[:, :1]).value, 0j)

    def test_bilinear_moments_match_the_dense_pairing(self) -> None:
        rng = np.random.default_rng(13)
        a = cplx(rng, self.M, self.M)
        b = cplx(rng, self.M, self.M)
        f = self.fock
        dense_val = f.pair(self.psi_l, f.dgamma(a) @ f.dgamma(b), self.psi_r)
        read = self.state.wickBilinearMoment([a, b])
        self.assertLess(abs(read.value - dense_val) / max(1, abs(dense_val)),
                        MACHINE)

    def test_reads_grade_on_the_pairing_not_on_hermiticity(self) -> None:
        read = self.state.wickOccupation(1)
        # A transition amplitude is complex: that is signal, not leakage.
        self.assertGreater(abs(read.value.imag), 1e-3)
        self.assertEqual(read.residual, self.state.dualityDefect())
        self.assertTrue(read.certificate.holds())
        self.assertEqual(read.certificate.grade,
                         cob.CertificateGrade.AlgebraicallyExact)
        # The regime is verified on Gamma: a non-Hermitian covariance.
        self.assertEqual(read.certificate.regime,
                         cob.CertificateRegime.NonNormal)
        self.assertTrue(self.state.purityCertificate(1e-10).holds())


# ─── evolution through the two frames ──────────────────────────────────────

@unittest.skipUnless(HAVE_QUANTUM and HAVE_SCIPY,
                     "needs the quantum subsystem and scipy")
class TestTwoFrameEvolution(unittest.TestCase):
    """i dPhi/dt = h Phi, -i dPhi~^T/dt = Phi~^T h for complex h."""

    M, N = 6, 3

    def dense_evolution(self, phi, left, h, t):
        fock = BilinearFock(self.M)
        big_h = fock.dgamma(h)
        psi_r = expm(-1j * big_h * t) @ fock.wedge(phi)
        # <Xi_L(t)| = <Xi_L| exp(+i H t): the left state by the transpose.
        psi_l = expm(1j * big_h * t).T @ fock.wedge(left)
        return fock, psi_l, psi_r

    def test_complex_symmetric_evolution_reproduces_the_dense_evolution(
            self) -> None:
        phi, left = biorthogonal_pair(self.M, self.N, seed=21)
        h = complex_symmetric(self.M, seed=22)
        self.assertLess(np.abs(h - h.T).max(), 1e-15)
        self.assertGreater(np.abs(h - h.conj().T).max(), 0.1)
        t = 0.7
        state = CovarianceState.fromBiorthogonalFrames(phi, left)
        state.evolve(h, t)
        fock, psi_l, psi_r = self.dense_evolution(phi, left, h, t)
        self.assertLess(abs(complex(psi_l @ psi_r) - 1.0), 1e-11)
        dense_gamma = fock.two_point(psi_l, psi_r)
        self.assertLess(relative(state.gamma(), dense_gamma), 1e-11)
        # Idempotency, the pairing and the trace survive exactly.
        self.assertLess(state.purityDefect(), 1e-12)
        self.assertLess(state.dualityDefect(), 1e-12)
        self.assertLess(abs(state.particleNumber() - self.N), 1e-12)
        # ... and so do the reads.
        parity = fock.pair(psi_l, fock.parity(), psi_r)
        self.assertLess(abs(state.wickParity().value - parity), 1e-11)
        # The frames themselves moved as the whitepaper states.
        np.testing.assert_allclose(np.array(state.rightFrame()),
                                   expm(-1j * h * t) @ phi, atol=1e-12)
        np.testing.assert_allclose(np.array(state.leftFrame()),
                                   expm(1j * h * t).T @ left, atol=1e-12)

    def test_gamma_obeys_the_commutator_equation(self) -> None:
        # Gamma(t) = exp(-iht) Gamma exp(+iht), the solution of
        # i dGamma/dt = [h, Gamma]; checked against scipy directly and by a
        # centered difference of the frame evolution.
        phi, left = biorthogonal_pair(self.M, self.N, seed=23)
        h = complex_symmetric(self.M, seed=24)
        gamma0 = phi @ left.T
        state = CovarianceState.fromBiorthogonalFrames(phi, left)
        state.evolve(h, 0.4)
        expected = expm(-0.4j * h) @ gamma0 @ expm(0.4j * h)
        self.assertLess(relative(state.gamma(), expected), 1e-12)
        eps = 1e-5
        plus = CovarianceState.fromBiorthogonalFrames(phi, left)
        minus = CovarianceState.fromBiorthogonalFrames(phi, left)
        plus.evolve(h, eps)
        minus.evolve(h, -eps)
        derivative = (np.array(plus.gamma()) - np.array(minus.gamma())) / (2 * eps)
        commutator = h @ gamma0 - gamma0 @ h
        self.assertLess(relative(1j * derivative, commutator), 1e-8)

    def test_a_defective_generator_needs_no_eigendecomposition(self) -> None:
        # A Jordan block: no eigenbasis, so the Hermitian path's spectral
        # propagator cannot exist. The two-frame path evolves it exactly.
        h = np.zeros((self.M, self.M), dtype=complex)
        h[0, 1] = h[1, 2] = 1.0
        h[3, 4] = 0.5j
        for i in range(self.M):
            h[i, i] = 0.2 - 0.1j
        np.testing.assert_allclose(
            np.array(CovarianceState.complexPropagator(h, 0.9)),
            expm(-0.9j * h), rtol=0, atol=1e-14)
        phi, left = biorthogonal_pair(self.M, self.N, seed=25)
        state = CovarianceState.fromBiorthogonalFrames(phi, left)
        state.evolve(h, 0.9)
        fock, psi_l, psi_r = self.dense_evolution(phi, left, h, 0.9)
        self.assertLess(relative(state.gamma(), fock.two_point(psi_l, psi_r)),
                        1e-11)
        self.assertLess(state.purityDefect(), 1e-12)
        # The Hermitian-adjoint path still refuses it, loudly.
        with self.assertRaises(ValueError):
            CovarianceState.fromOccupations(np.ones(self.M)).evolve(h, 0.9)

    def test_group_property(self) -> None:
        phi, left = biorthogonal_pair(self.M, self.N, seed=26)
        h = complex_symmetric(self.M, seed=27)
        s_a = CovarianceState.fromBiorthogonalFrames(phi, left)
        s_b = CovarianceState.fromBiorthogonalFrames(phi, left)
        s_a.evolve(h, 0.3)
        s_a.evolve(h, 0.5)
        s_b.evolve(h, 0.8)
        self.assertLess(relative(s_a.gamma(), s_b.gamma()), 1e-13)
        # Backward evolution returns the initial covariance.
        s_a.evolve(h, -0.8)
        self.assertLess(relative(s_a.gamma(), phi @ left.T), 1e-12)

    def test_evolve_equals_transport_by_the_complex_propagator(self) -> None:
        phi, left = biorthogonal_pair(self.M, self.N, seed=28)
        h = complex_symmetric(self.M, seed=29)
        s_a = CovarianceState.fromBiorthogonalFrames(phi, left)
        s_b = CovarianceState.fromBiorthogonalFrames(phi, left)
        s_a.evolve(h, 0.37)
        s_b.applyTransport(CovarianceState.complexPropagator(h, 0.37))
        self.assertLess(relative(s_a.gamma(), s_b.gamma()), 1e-13)
        self.assertLess(relative(s_a.leftFrame(), s_b.leftFrame()), 1e-13)

    def test_idempotency_across_two_hundred_steps(self) -> None:
        # The long-evolution bar: two alternating complex-symmetric
        # generators, 200 steps. Non-unitary evolution changes the frame
        # scales, so the defects are reported relative to ||Gamma||.
        phi, left = biorthogonal_pair(self.M, self.N, seed=30)
        state = CovarianceState.fromBiorthogonalFrames(phi, left)
        h1 = complex_symmetric(self.M, seed=31)
        h2 = complex_symmetric(self.M, seed=32)
        for step in range(200):
            state.evolve(h1 if step % 2 == 0 else h2, 0.02)
        gamma = np.array(state.gamma())
        scale = max(1.0, np.linalg.norm(gamma))
        self.assertLess(state.purityDefect() / scale, 1e-11)
        self.assertLess(state.dualityDefect(), 1e-11)
        self.assertLess(abs(np.trace(gamma) - self.N), 1e-10)
        eigs = np.sort_complex(np.linalg.eigvals(gamma))
        np.testing.assert_allclose(np.sort(np.abs(eigs)),
                                   [0, 0, 0, 1, 1, 1], atol=1e-9)

    def test_transport_moves_the_left_frame_by_the_contragredient(
            self) -> None:
        phi, left = biorthogonal_pair(self.M, self.N, seed=33)
        rng = np.random.default_rng(34)
        u = np.eye(self.M) + 0.4 * cplx(rng, self.M, self.M)
        state = CovarianceState.fromBiorthogonalFrames(phi, left)
        state.applyTransport(u)
        np.testing.assert_allclose(np.array(state.rightFrame()), u @ phi,
                                   atol=1e-13)
        np.testing.assert_allclose(np.array(state.leftFrame()),
                                   np.linalg.inv(u).T @ left, atol=1e-12)
        self.assertLess(state.dualityDefect(), 1e-12)
        self.assertLess(
            relative(state.gamma(), u @ phi @ left.T @ np.linalg.inv(u)), 1e-12)
        singular = np.ones((self.M, self.M), dtype=complex)
        with self.assertRaises(ValueError):
            state.applyTransport(singular)


# ─── the Hermitian special case ────────────────────────────────────────────

@unittest.skipUnless(HAVE_QUANTUM, "tessera built without the quantum subsystem")
class TestHermitianSpecialCase(unittest.TestCase):
    """With a certified *-structure (Phi~ = conj(Phi), Phi orthonormal) and a
    Hermitian h, the transpose path IS the present Hermitian-adjoint path."""

    M, N = 6, 3

    def test_propagators_agree_for_hermitian_h(self) -> None:
        h = random_hermitian(self.M, seed=41)
        np.testing.assert_allclose(
            np.array(CovarianceState.complexPropagator(h, 0.6)),
            np.array(CovarianceState.propagator(h, 0.6)), rtol=0, atol=1e-13)

    def test_evolution_equals_the_present_path(self) -> None:
        q = orthonormal(self.M, self.N, seed=42)
        hermitian = CovarianceState.fromSlaterFrame(q)
        transpose = CovarianceState.fromBiorthogonalFrames(q, q.conj())
        np.testing.assert_allclose(np.array(transpose.gamma()),
                                   np.array(hermitian.gamma()), atol=1e-14)
        h1 = random_hermitian(self.M, seed=43)
        h2 = random_hermitian(self.M, seed=44)
        for step in range(50):
            h = h1 if step % 2 == 0 else h2
            hermitian.evolve(h, 0.05)
            transpose.evolve(h, 0.05)
        self.assertLess(relative(transpose.gamma(), hermitian.gamma()), 1e-12)
        # The pair stays the *-structure pair: Phi~ = conj(Phi).
        np.testing.assert_allclose(np.array(transpose.leftFrame()),
                                   np.array(transpose.rightFrame()).conj(),
                                   atol=1e-12)
        for read_t, read_h in (
                (transpose.wickParity(), hermitian.wickParity()),
                (transpose.wickOccupation(2), hermitian.wickOccupation(2)),
                (transpose.wickNormalOrdered([0, 1], [1, 0]),
                 hermitian.wickNormalOrdered([0, 1], [1, 0]))):
            self.assertLess(abs(read_t.value - read_h.value), 1e-12)
            # The regime is verified on Gamma, so a Hermitian pair reads as
            # the positive regime on either path.
            self.assertEqual(read_t.certificate.regime,
                             cob.CertificateRegime.PositiveSemidefinite)
            self.assertTrue(read_t.certificate.holds())

    def test_mean_field_equals_the_present_path(self) -> None:
        q = orthonormal(self.M, self.N, seed=45)
        h0 = random_hermitian(self.M, seed=46)

        def hartree(gamma):
            return h0 + 0.8 * np.diag(np.diag(gamma).real).astype(complex)

        hermitian = CovarianceState.fromSlaterFrame(q)
        transpose = CovarianceState.fromBiorthogonalFrames(q, q.conj())
        reads_h = hermitian.meanFieldEvolve(hartree, 0.05, 40)
        reads_t = transpose.meanFieldEvolve(hartree, 0.05, 40)
        self.assertLess(relative(transpose.gamma(), hermitian.gamma()), 1e-11)
        for rh, rt in zip(reads_h, reads_t):
            self.assertTrue(rt.certificate.holds())
            self.assertTrue(math.isnan(rh.dualityDefect))
            self.assertLess(rt.dualityDefect, 1e-12)


# ─── the self-consistent loop on a complex covariance ──────────────────────

@unittest.skipUnless(HAVE_QUANTUM, "tessera built without the quantum subsystem")
class TestComplexMeanField(unittest.TestCase):
    """meanFieldEvolve accepts the complex h(Gamma) of a complex covariance on
    the transpose path; the Hermitian-adjoint path still refuses it."""

    M, N = 5, 2

    def generator(self):
        h0 = complex_symmetric(self.M, seed=51)

        def h_of_gamma(gamma):
            # The complex diagonal of Gamma, with no real projection.
            return h0 + 0.6 * np.diag(np.diag(gamma))
        return h_of_gamma

    def test_loop_certifies_every_step_and_matches_manual_steps(self) -> None:
        phi, left = biorthogonal_pair(self.M, self.N, seed=52)
        looped = CovarianceState.fromBiorthogonalFrames(phi, left)
        manual = CovarianceState.fromBiorthogonalFrames(phi, left)
        h_of_gamma = self.generator()
        reads = looped.meanFieldEvolve(h_of_gamma, 0.05, 40)
        for _ in range(40):
            manual.evolve(h_of_gamma(np.array(manual.gamma())), 0.05)
        np.testing.assert_array_equal(np.array(looped.gamma()),
                                      np.array(manual.gamma()))
        self.assertEqual(len(reads), 40)
        for read in reads:
            self.assertTrue(read.certificate.holds())
            self.assertLess(read.purityDefect, 1e-11)
            self.assertLess(read.dualityDefect, 1e-11)
            # The generator's non-Hermiticity is reported, not graded.
            self.assertGreater(read.generatorHermiticityDefect, 0.01)

    def test_hermitian_path_refuses_a_complex_self_consistent_h(self) -> None:
        state = CovarianceState.fromSlaterFrame(orthonormal(self.M, self.N, 53))
        with self.assertRaises(ValueError):
            state.meanFieldEvolve(self.generator(), 0.05, 3)


# ─── checkpoint serialization of the pair ──────────────────────────────────

@unittest.skipUnless(HAVE_QUANTUM, "tessera built without the quantum subsystem")
class TestBiorthogonalRecord(unittest.TestCase):

    def test_round_trip_keeps_the_frames_and_gamma_bit_exact(self) -> None:
        phi, left = biorthogonal_pair(5, 2, seed=61)
        state = CovarianceState.fromBiorthogonalFrames(phi, left)
        state.evolve(complex_symmetric(5, seed=62), 0.5)
        record = state.toRecord()
        self.assertEqual(record["dual"], "transpose")
        self.assertEqual(record["frame_columns"], 2)
        replayed = CovarianceState.fromRecord(record)
        self.assertEqual(replayed.dual(), CovarianceDual.Transpose)
        self.assertEqual(replayed.covarianceHash(), state.covarianceHash())
        np.testing.assert_array_equal(np.array(replayed.gamma()),
                                      np.array(state.gamma()))
        np.testing.assert_array_equal(np.array(replayed.leftFrame()),
                                      np.array(state.leftFrame()))
        self.assertEqual(replayed.wickParity().value,
                         state.wickParity().value)

    def test_records_without_a_dual_are_hermitian_adjoint(self) -> None:
        state = CovarianceState.fromOccupations(np.array([1.0, 0.0]))
        record = state.toRecord()
        self.assertEqual(record["dual"], "hermitian-adjoint")
        del record["dual"]
        self.assertEqual(CovarianceState.fromRecord(record).dual(),
                         CovarianceDual.HermitianAdjoint)
        record["dual"] = "sesquilinear-ish"
        with self.assertRaises(ValueError):
            CovarianceState.fromRecord(record)

    def test_corrupt_frame_payload_is_rejected(self) -> None:
        phi, left = biorthogonal_pair(4, 2, seed=63)
        record = CovarianceState.fromBiorthogonalFrames(phi, left).toRecord()
        record["left_frame_re"] = record["left_frame_re"][:-1]
        record["left_frame_im"] = record["left_frame_im"][:-1]
        with self.assertRaises(ValueError):
            CovarianceState.fromRecord(record)


if __name__ == "__main__":
    unittest.main()
