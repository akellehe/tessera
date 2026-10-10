# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Acceptance tests for the quasi-free covariance layer
(:class:`tessera.quantum.CovarianceState`), ticket #780 / design spec
sections 5.9, 6.7, and 13 (Algorithm F).

Covers every ticket acceptance bullet (the direct #771 lazy-engine
cross-validation is deferred to post-merge integration; the dense
references here are the merged #766 machinery and independent NumPy
Jordan-Wigner chains):

* on small fixtures every Wick-evaluated certificate matches dense Fock
  references: occupations, parity, Gram/Pauli determinants, |S_ABC|^2,
  <J^2>, and Var(J^2) (quartic and octic Wick sums at machine precision);
* a single-mode spin-1/2 Slater fixture is an exact J^2 eigenstate
  (Var(J^2) = 0); a generic Slater fixture with <J^2> = 3/4 reports its
  nonzero variance;
* purity ||Gamma^2 - Gamma|| stays within tolerance across long evolutions
  (hundreds of steps) and across the mean-field loop;
* the covariance path allocates no 2^M object and runs at M = 200, where
  any dense Fock construction is impossible;
* cold replay reproduces Gamma and every Wick certificate (serialization
  round trip + AnalyticCache cached-vs-cold comparison);
* property tests: basis rotations (Wick values invariant under one-particle
  unitaries applied consistently to Gamma and the observable coefficients)
  and mode relabelings;
* negative controls: a non-idempotent Gamma reports its purity defect, a
  leaky transport degrades purity visibly, a non-Hermitian generator is
  rejected loudly.

Exactness bar: Wick values on quasi-free states are finite exact sums —
they match the dense references to double round-off (~1e-13 relative on
well-conditioned fixtures). The dense references are built two independent
ways: the merged #766 ExteriorAlgebra operators AND independent NumPy
Jordan-Wigner kron chains (never re-derivations through the bindings under
test).

Skips cleanly when tessera was built without the quantum subsystem.
"""

from __future__ import annotations

import unittest

import numpy as np

import tessera

try:
    from tessera.quantum import (
        CovarianceState,
        ExteriorAlgebra,
        WickCertificateRead,
    )
    HAVE_QUANTUM = True
except ImportError:
    HAVE_QUANTUM = False

cob = tessera.cobordism
obs = tessera.observables

MACHINE = 1e-13     # relative bar for exact finite sums on small fixtures
TIGHT = 1e-12       # accumulated round-off across long evolutions


# ─── independent dense references ──────────────────────────────────────────

def dense(coo):
    """(rows, cols, values, n) COO tuple -> dense complex ndarray."""
    rows, cols, vals, n = coo
    out = np.zeros((n, n), dtype=complex)
    for r, c, v in zip(rows, cols, vals):
        out[r, c] += v
    return out


_S_MINUS = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
_Z = np.diag([1.0, -1.0]).astype(complex)


def jw_annihilation(mode: int, n_modes: int) -> np.ndarray:
    """Independent dense Jordan-Wigner a_mode on the n(b) = sum b_i 2^i
    basis (mode 0 = least-significant bit; Z string strictly below)."""
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


class DenseFock:
    """Dense Fock reference operators over M modes, built from independent
    Jordan-Wigner chains (creation/annihilation/number/parity/dGamma) plus
    Slater vectors assembled by applying smeared creations to the vacuum."""

    def __init__(self, n_modes: int):
        self.n = n_modes
        self.a = [jw_annihilation(m, n_modes) for m in range(n_modes)]
        self.adag = [op.conj().T for op in self.a]

    def creation_smeared(self, v: np.ndarray) -> np.ndarray:
        return sum(v[i] * self.adag[i] for i in range(self.n))

    def annihilation_smeared(self, w: np.ndarray) -> np.ndarray:
        return sum(np.conj(w[i]) * self.a[i] for i in range(self.n))

    def slater(self, orbitals: np.ndarray) -> np.ndarray:
        """a+(phi_1) ... a+(phi_N) |vac> for the COLUMNS of `orbitals`."""
        psi = np.zeros(2 ** self.n, dtype=complex)
        psi[0] = 1.0
        for k in range(orbitals.shape[1] - 1, -1, -1):
            psi = self.creation_smeared(orbitals[:, k]) @ psi
        return psi

    def dgamma(self, one_particle: np.ndarray) -> np.ndarray:
        return sum(one_particle[i, j] * (self.adag[i] @ self.a[j])
                   for i in range(self.n) for j in range(self.n))

    def number(self, mode: int) -> np.ndarray:
        return self.adag[mode] @ self.a[mode]

    def parity(self, modes=None) -> np.ndarray:
        modes = range(self.n) if modes is None else modes
        op = np.eye(2 ** self.n, dtype=complex)
        for m in modes:
            op = op @ (np.eye(2 ** self.n, dtype=complex) - 2 * self.number(m))
        return op


def random_orbitals(n_modes: int, n_particles: int, seed: int) -> np.ndarray:
    """Random ORTHONORMAL occupied orbitals (thin Q of a Gaussian frame)."""
    rng = np.random.default_rng(seed)
    frame = (rng.normal(size=(n_modes, n_particles))
             + 1j * rng.normal(size=(n_modes, n_particles)))
    q, _ = np.linalg.qr(frame)
    return q


def random_hermitian(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    b = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    return (b + b.conj().T) / 2


def random_unitary(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    b = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    q, r = np.linalg.qr(b)
    return q * (np.diag(r) / np.abs(np.diag(r)))


def pauli_over_sites(n_sites: int):
    """J_alpha = direct sum over sites of sigma_alpha / 2 (modes ordered
    site-major: [site0-up, site0-down, site1-up, ...])."""
    sx = np.array([[0, 1], [1, 0]], dtype=complex) / 2
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex) / 2
    sz = np.diag([1.0, -1.0]).astype(complex) / 2
    def blocks(s):
        out = np.zeros((2 * n_sites, 2 * n_sites), dtype=complex)
        for k in range(n_sites):
            out[2 * k:2 * k + 2, 2 * k:2 * k + 2] = s
        return out
    return blocks(sx), blocks(sy), blocks(sz)


def spacetime_two_triangles():
    """Two disjoint triangles (the AnalyticCache fixture idiom)."""
    sig = tessera.Signature(4, tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    st = tessera.Spacetime(metric, tessera.HERMITIAN_WEIGHTED, 1.0, 1.0,
                           tessera.PREFERRED, tessera.Toroid())
    verts = {i: st.create_vertex(i) for i in [0, 1, 2, 10, 11, 12]}
    for a, b in [(0, 1), (1, 2), (0, 2), (10, 11), (11, 12), (10, 12)]:
        st.create_simplex([verts[a], verts[b]])
    for e in st.get_edge_list().to_vector():
        e.set_length(1.0 + 0j)
        e.set_phase(0.0)
    return st


# ─── construction and the covariance data model ────────────────────────────

@unittest.skipUnless(HAVE_QUANTUM, "tessera built without the quantum subsystem")
class TestConstructionAndDataModel(unittest.TestCase):
    """Gamma_ij = <a_j+ a_i>: constructors, defects, and the Nambu shape."""

    def test_constructor_rejects_non_square(self) -> None:
        with self.assertRaises(ValueError):
            CovarianceState(np.zeros((3, 4), dtype=complex))

    def test_from_occupations_is_diagonal(self) -> None:
        s = CovarianceState.from_occupations(np.array([1.0, 0.25, 0.0]))
        np.testing.assert_allclose(np.array(s.gamma()),
                                   np.diag([1.0, 0.25, 0.0]), atol=0)
        self.assertAlmostEqual(s.particle_number().real, 1.25, places=15)
        self.assertEqual(s.occupation_spectrum_defect(), 0.0)

    def test_out_of_range_occupation_reports_spectrum_defect(self) -> None:
        s = CovarianceState.from_occupations(np.array([1.5, -0.25]))
        self.assertAlmostEqual(s.occupation_spectrum_defect(), 0.5, places=14)

    def test_slater_frame_builds_the_projector(self) -> None:
        q = random_orbitals(5, 2, seed=3)
        s = CovarianceState.from_slater_frame(q)
        np.testing.assert_allclose(np.array(s.gamma()), q @ q.conj().T,
                                   atol=1e-14)
        self.assertLess(s.purity_defect(), 1e-13)
        self.assertLess(s.hermiticity_defect(), 1e-14)

    def test_slater_frame_is_span_gauge_invariant(self) -> None:
        # A non-orthonormal frame with the same span gives the same Gamma.
        q = random_orbitals(5, 2, seed=4)
        mix = np.array([[2.0, 1.0 - 1j], [0.5j, 3.0]])
        s_a = CovarianceState.from_slater_frame(q)
        s_b = CovarianceState.from_slater_frame(q @ mix)
        np.testing.assert_allclose(np.array(s_a.gamma()),
                                   np.array(s_b.gamma()), atol=1e-13)

    def test_slater_frame_rejects_rank_deficiency(self) -> None:
        q = random_orbitals(4, 1, seed=5)
        frame = np.hstack([q, q])  # duplicated orbital
        with self.assertRaises(ValueError):
            CovarianceState.from_slater_frame(frame)

    def test_slater_frame_empty_columns_is_the_vacuum(self) -> None:
        s = CovarianceState.from_slater_frame(np.zeros((4, 0), dtype=complex))
        np.testing.assert_allclose(np.array(s.gamma()), np.zeros((4, 4)),
                                   atol=0)
        self.assertEqual(s.wick_parity().value, 1.0 + 0j)

    def test_band_projector_is_adopted_verbatim(self) -> None:
        # An oblique (non-Hermitian) projector is NOT symmetrized: the
        # defect is measured and reported.
        p = np.array([[1.0, 0.7], [0.0, 0.0]], dtype=complex)  # P^2 = P
        s = CovarianceState.from_band_projector(p)
        np.testing.assert_allclose(np.array(s.gamma()), p, atol=0)
        self.assertLess(s.purity_defect(), 1e-15)
        self.assertGreater(s.hermiticity_defect(), 0.1)
        cert = s.wick_parity().certificate
        self.assertEqual(cert.regime, cob.CertificateRegime.NonNormal)

    def test_purity_defect_separates_pure_from_mixed(self) -> None:
        pure = CovarianceState.from_occupations(np.array([1.0, 0.0, 1.0]))
        mixed = CovarianceState.from_occupations(np.array([0.5, 0.5, 0.5]))
        self.assertEqual(pure.purity_defect(), 0.0)
        self.assertGreater(mixed.purity_defect(), 0.4)  # negative control
        self.assertEqual(mixed.occupation_spectrum_defect(), 0.0)
        self.assertTrue(pure.purity_certificate(1e-9).holds())
        self.assertFalse(mixed.purity_certificate(1e-9).holds())

    def test_nambu_covariance_shape_and_blocks(self) -> None:
        q = random_orbitals(4, 2, seed=6)
        s = CovarianceState.from_slater_frame(q)
        self.assertTrue(s.number_conserving())
        np.testing.assert_allclose(np.array(s.pairing()), np.zeros((4, 4)),
                                   atol=0)
        g = np.array(s.nambu_covariance())
        self.assertEqual(g.shape, (8, 8))
        gamma = np.array(s.gamma())
        np.testing.assert_allclose(g[:4, :4], gamma, atol=0)
        np.testing.assert_allclose(g[:4, 4:], np.zeros((4, 4)), atol=0)
        np.testing.assert_allclose(g[4:, :4], np.zeros((4, 4)), atol=0)
        np.testing.assert_allclose(g[4:, 4:], np.eye(4) - gamma.T, atol=0)
        # Idempotent exactly when Gamma is (pure Slater here).
        self.assertLess(np.linalg.norm(g @ g - g), 1e-13)

    def test_nambu_covariance_mixed_is_not_idempotent(self) -> None:
        s = CovarianceState.from_occupations(np.array([0.5, 0.5]))
        g = np.array(s.nambu_covariance())
        self.assertGreater(np.linalg.norm(g @ g - g), 0.4)

    def test_occupations_and_particle_number(self) -> None:
        q = random_orbitals(5, 3, seed=7)
        s = CovarianceState.from_slater_frame(q)
        gamma = q @ q.conj().T
        np.testing.assert_allclose(np.array(s.occupations()),
                                   np.diag(gamma), atol=1e-14)
        self.assertAlmostEqual(s.particle_number().real, 3.0, places=12)
        self.assertAlmostEqual(s.occupation(2).real, gamma[2, 2].real,
                               places=14)
        with self.assertRaises(ValueError):
            s.occupation(5)

    def test_covariance_hash_tracks_gamma_exactly(self) -> None:
        q = random_orbitals(4, 2, seed=8)
        s_a = CovarianceState.from_slater_frame(q)
        s_b = CovarianceState.from_slater_frame(q)
        self.assertEqual(s_a.covariance_hash(), s_b.covariance_hash())
        gamma = np.array(s_a.gamma())
        gamma[0, 0] += 1e-15  # one ulp-scale change flips the fingerprint
        s_c = CovarianceState(gamma)
        self.assertNotEqual(s_a.covariance_hash(), s_c.covariance_hash())


# ─── Wick reads against dense Fock references ──────────────────────────────

@unittest.skipUnless(HAVE_QUANTUM, "tessera built without the quantum subsystem")
class TestWickAgainstDenseFock(unittest.TestCase):
    """Every Wick-evaluated certificate matches the dense references built
    from the merged #766 machinery AND independent NumPy Jordan-Wigner
    chains, at machine precision."""

    M, N, SEED = 5, 2, 11

    def setUp(self) -> None:
        self.q = random_orbitals(self.M, self.N, self.SEED)
        self.state = CovarianceState.from_slater_frame(self.q)
        self.fock = DenseFock(self.M)
        self.psi = self.fock.slater(self.q)
        self.assertAlmostEqual(np.vdot(self.psi, self.psi).real, 1.0,
                               places=12)

    def expect(self, op: np.ndarray) -> complex:
        return complex(np.vdot(self.psi, op @ self.psi))

    def test_covariance_matches_dense_two_point_functions(self) -> None:
        gamma = np.array(self.state.gamma())
        for i in range(self.M):
            for j in range(self.M):
                dense_val = self.expect(self.fock.adag[j] @ self.fock.a[i])
                self.assertLess(abs(gamma[i, j] - dense_val), MACHINE)

    def test_exterior_algebra_and_jw_references_agree(self) -> None:
        # The two independent dense constructions agree with each other,
        # so either one arbitrates a disagreement.
        alg = ExteriorAlgebra(self.M)
        for m in range(self.M):
            np.testing.assert_allclose(dense(alg.creation_matrix_coo(m)),
                                       self.fock.adag[m], atol=1e-15)
        psi_alg = np.array(alg.wedge([self.q[:, k] for k in range(self.N)]))
        np.testing.assert_allclose(psi_alg, self.psi, atol=1e-13)

    def test_occupations_match_dense(self) -> None:
        for m in range(self.M):
            read = self.state.wick_occupation(m)
            self.assertLess(abs(read.value - self.expect(self.fock.number(m))),
                            MACHINE)
            self.assertLess(read.residual, 1e-13)

    def test_total_number_matches_dense(self) -> None:
        total = sum(self.fock.number(m) for m in range(self.M))
        read = self.state.wick_total_number()
        self.assertLess(abs(read.value - self.expect(total)), MACHINE)
        self.assertAlmostEqual(read.value.real, self.N, places=12)

    def test_parity_matches_dense(self) -> None:
        read = self.state.wick_parity()
        self.assertLess(abs(read.value - self.expect(self.fock.parity())),
                        MACHINE)
        # N-particle Slater state: parity is exactly (-1)^N.
        self.assertAlmostEqual(read.value.real, (-1.0) ** self.N, places=12)

    def test_subset_parity_matches_dense(self) -> None:
        for modes in ([0], [1, 3], [0, 2, 4], [0, 1, 2, 3, 4]):
            read = self.state.wick_subset_parity(modes)
            dense_val = self.expect(self.fock.parity(modes))
            self.assertLess(abs(read.value - dense_val), MACHINE,
                            msg=f"subset {modes}")

    def test_subset_parity_validates_modes(self) -> None:
        with self.assertRaises(ValueError):
            self.state.wick_subset_parity([0, 0])
        with self.assertRaises(ValueError):
            self.state.wick_subset_parity([9])

    def test_normal_ordered_joint_occupations(self) -> None:
        for modes in ([0], [1, 2], [0, 3, 4]):
            read = self.state.wick_normal_ordered(modes, modes)
            op = np.eye(2 ** self.M, dtype=complex)
            for m in modes:
                op = op @ self.fock.number(m)
            self.assertLess(abs(read.value - self.expect(op)), MACHINE,
                            msg=f"joint occupation {modes}")

    def test_normal_ordered_mixed_patterns_match_dense(self) -> None:
        # <a+_{c1} a+_{c2} a_{a2} a_{a1}>: paired-slot convention — the
        # annihilator string is applied in reversed list order.
        cases = (([0], [1]), ([0, 1], [2, 3]), ([0, 2], [2, 0]),
                 ([1, 3, 4], [0, 2, 4]))
        for creators, annihilators in cases:
            read = self.state.wick_normal_ordered(creators, annihilators)
            op = np.eye(2 ** self.M, dtype=complex)
            for c in creators:
                op = op @ self.fock.adag[c]
            for x in reversed(annihilators):
                op = op @ self.fock.a[x]
            self.assertLess(abs(read.value - self.expect(op)), MACHINE,
                            msg=f"pattern {creators} / {annihilators}")

    def test_normal_ordered_pauli_exclusion_is_exact_zero(self) -> None:
        read = self.state.wick_normal_ordered([1, 1], [0, 2])
        self.assertEqual(read.value, 0.0 + 0j)  # repeated determinant row

    def test_normal_ordered_unbalanced_is_exact_zero(self) -> None:
        read = self.state.wick_normal_ordered([0, 1], [2])
        self.assertEqual(read.value, 0.0 + 0j)
        op = self.fock.adag[0] @ self.fock.adag[1] @ self.fock.a[2]
        self.assertLess(abs(self.expect(op)), 1e-14)

    def test_gram_determinant_matches_dense_smeared_strings(self) -> None:
        rng = np.random.default_rng(21)
        for p in (1, 2, 3):
            v = rng.normal(size=(self.M, p)) + 1j * rng.normal(size=(self.M, p))
            w = rng.normal(size=(self.M, p)) + 1j * rng.normal(size=(self.M, p))
            read = self.state.wick_gram_determinant(v, w)
            op = np.eye(2 ** self.M, dtype=complex)
            for k in range(p):
                op = op @ self.fock.creation_smeared(v[:, k])
            for k in range(p - 1, -1, -1):
                op = op @ self.fock.annihilation_smeared(w[:, k])
            dense_val = self.expect(op)
            self.assertLess(abs(read.value - dense_val),
                            MACHINE * max(1.0, abs(dense_val)),
                            msg=f"gram p={p}")

    def test_gram_determinant_equals_elementary_read_on_unit_columns(self) -> None:
        eye = np.eye(self.M, dtype=complex)
        creators, annihilators = [0, 2], [1, 4]
        a = self.state.wick_normal_ordered(creators, annihilators)
        b = self.state.wick_gram_determinant(eye[:, creators],
                                           eye[:, annihilators])
        self.assertLess(abs(a.value - b.value), 1e-15)

    def test_bilinear_moments_match_dense_through_octic_order(self) -> None:
        # n = 1..4 bilinears = quadratic..octic CAR monomial sums, with
        # generic NON-Hermitian coefficient matrices: the contraction
        # combinatorics (signs, orderings, partitions) all matter.
        rng = np.random.default_rng(31)
        mats = [rng.normal(size=(self.M, self.M))
                + 1j * rng.normal(size=(self.M, self.M)) for _ in range(4)]
        ops = [self.fock.dgamma(m) for m in mats]
        for n in range(1, 5):
            read = self.state.wick_bilinear_moment(mats[:n])
            op = np.eye(2 ** self.M, dtype=complex)
            for k in range(n):
                op = op @ ops[k]
            dense_val = self.expect(op)
            self.assertLess(abs(read.value - dense_val),
                            MACHINE * max(1.0, abs(dense_val)),
                            msg=f"bilinear moment n={n}")

    def test_bilinear_moment_order_sensitivity_is_the_commutator(self) -> None:
        rng = np.random.default_rng(37)
        a = rng.normal(size=(self.M, self.M)) + 1j * rng.normal(size=(self.M, self.M))
        b = rng.normal(size=(self.M, self.M)) + 1j * rng.normal(size=(self.M, self.M))
        gamma = np.array(self.state.gamma())
        ab = self.state.wick_bilinear_moment([a, b]).value
        ba = self.state.wick_bilinear_moment([b, a]).value
        commutator = np.trace((a @ b - b @ a) @ gamma)
        self.assertLess(abs((ab - ba) - commutator), MACHINE)

    def test_bilinear_moment_validates_inputs(self) -> None:
        with self.assertRaises(ValueError):
            self.state.wick_bilinear_moment([])
        with self.assertRaises(ValueError):
            self.state.wick_bilinear_moment([np.eye(3, dtype=complex)])

    def test_spin_reads_match_dense_on_random_hermitian_j(self) -> None:
        js = [random_hermitian(self.M, seed=41 + k) for k in range(3)]
        e_read = self.state.wick_spin_squared_expectation(*js)
        v_read = self.state.wick_spin_squared_variance(*js)
        j2 = sum(self.fock.dgamma(j) @ self.fock.dgamma(j) for j in js)
        e_dense = self.expect(j2)
        v_dense = self.expect(j2 @ j2) - e_dense ** 2
        self.assertLess(abs(e_read.value - e_dense),
                        MACHINE * max(1.0, abs(e_dense)))
        self.assertLess(abs(v_read.value - v_dense),
                        1e-12 * max(1.0, abs(v_dense)))
        self.assertLess(abs(e_read.value.imag), 1e-12)
        self.assertLess(abs(v_read.value.imag), 1e-11)


# ─── the color wedge |S_ABC|^2 against ColorFiber certificates ─────────────

@unittest.skipUnless(HAVE_QUANTUM, "tessera built without the quantum subsystem")
class TestColorWedge(unittest.TestCase):
    """|S_ABC|^2 = det(C+ Gamma C), cross-checked against the #767
    ColorFiber determinant certificates on three-column Slater fixtures and
    against dense Fock references."""

    def test_slater_fixture_reproduces_color_fiber_certificates(self) -> None:
        # Gamma = Slater projector onto colspan(C) for a generic
        # (non-orthonormal) 3x3 color-column matrix: the Wick value equals
        # det(C+C) = |det C|^2 exactly — ColorFiber's singlet certificates.
        rng = np.random.default_rng(51)
        c = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
        state = CovarianceState.from_slater_frame(c)
        read = state.wick_color_wedge_squared(c)
        gram = tessera.ColorFiber.singlet_gram(c)
        wedge = tessera.ColorFiber.color_wedge(c)
        self.assertLess(abs(read.value.real - gram), MACHINE * max(1.0, gram))
        self.assertLess(abs(read.value.real - abs(wedge) ** 2),
                        MACHINE * max(1.0, gram))
        self.assertLess(abs(read.value.imag), 1e-13)

    def test_embedded_color_triad_matches_dense_and_color_fiber(self) -> None:
        # Six modes, color columns supported on modes {0, 1, 2}: nontrivial
        # embedding (Gamma is 6x6, not the identity).
        rng = np.random.default_rng(52)
        c3 = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
        c6 = np.zeros((6, 3), dtype=complex)
        c6[:3, :] = c3
        state = CovarianceState.from_slater_frame(c6)
        read = state.wick_color_wedge_squared(c6)
        gram = tessera.ColorFiber.singlet_gram(c3)
        self.assertLess(abs(read.value.real - gram), MACHINE * max(1.0, gram))
        # Dense cross-check on the 6-mode Fock space.
        fock = DenseFock(6)
        q, _ = np.linalg.qr(c6)
        psi = fock.slater(q)
        op = np.eye(2 ** 6, dtype=complex)
        for k in range(3):
            op = op @ fock.creation_smeared(c6[:, k])
        for k in range(2, -1, -1):
            op = op @ fock.annihilation_smeared(c6[:, k])
        dense_val = np.vdot(psi, op @ psi)
        self.assertLess(abs(read.value - dense_val),
                        MACHINE * max(1.0, abs(dense_val)))

    def test_orthonormal_triad_saturates_at_one(self) -> None:
        q = random_orbitals(5, 3, seed=53)
        state = CovarianceState.from_slater_frame(q)
        read = state.wick_color_wedge_squared(q)
        self.assertAlmostEqual(read.value.real, 1.0, places=12)

    def test_duplicate_color_mode_is_pauli_zero(self) -> None:
        q = random_orbitals(5, 3, seed=54)
        c = q.copy()
        c[:, 2] = c[:, 0]  # duplicate color column
        state = CovarianceState.from_slater_frame(q)
        read = state.wick_color_wedge_squared(c)
        self.assertLess(abs(read.value), 1e-13)

    def test_missing_color_direction_reads_zero(self) -> None:
        # Only two of the three color directions occupied: the top wedge is
        # empty and |S_ABC|^2 vanishes.
        eye = np.eye(4, dtype=complex)
        state = CovarianceState.from_slater_frame(eye[:, :2])
        read = state.wick_color_wedge_squared(eye[:, :3])
        self.assertLess(abs(read.value), 1e-14)

    def test_color_wedge_validates_shape(self) -> None:
        state = CovarianceState.from_occupations(np.array([1.0, 0.0, 0.0]))
        with self.assertRaises(ValueError):
            state.wick_color_wedge_squared(np.eye(3, dtype=complex)[:, :2])


# ─── the mandated spin fixtures ────────────────────────────────────────────

@unittest.skipUnless(HAVE_QUANTUM, "tessera built without the quantum subsystem")
class TestSpinFixtures(unittest.TestCase):
    """The two mandated fixtures: an exact J^2 eigenstate with zero
    variance, and a generic Slater with <J^2> = 3/4 and NONZERO variance
    (design spec 5.12: expectation alone does not certify a sharp spin)."""

    def test_single_mode_spin_half_is_exact_eigenstate(self) -> None:
        # One particle in the spin-up mode of a single spin-1/2 doublet;
        # J_alpha = sigma_alpha / 2 (caller-supplied standard spin ops).
        jx, jy, jz = pauli_over_sites(1)
        state = CovarianceState.from_occupations(np.array([1.0, 0.0]))
        e = state.wick_spin_squared_expectation(jx, jy, jz)
        v = state.wick_spin_squared_variance(jx, jy, jz)
        self.assertLess(abs(e.value - 0.75), 1e-15)
        self.assertLess(abs(v.value), 1e-13)   # exact eigenstate: Var = 0
        # Dense cross-check.
        fock = DenseFock(2)
        psi = fock.slater(np.eye(2, dtype=complex)[:, :1])
        j2 = sum(fock.dgamma(j) @ fock.dgamma(j) for j in (jx, jy, jz))
        self.assertLess(abs(np.vdot(psi, j2 @ psi) - 0.75), 1e-14)

    def test_rotated_single_particle_is_still_an_eigenstate(self) -> None:
        # Any one-particle state of one doublet is a J^2 = 3/4 eigenstate.
        jx, jy, jz = pauli_over_sites(1)
        orbital = np.array([[0.6], [0.8j]], dtype=complex)
        state = CovarianceState.from_slater_frame(orbital)
        self.assertLess(
            abs(state.wick_spin_squared_expectation(jx, jy, jz).value - 0.75),
            1e-14)
        self.assertLess(abs(state.wick_spin_squared_variance(jx, jy, jz).value),
                        1e-13)

    def test_generic_slater_reports_three_quarters_with_nonzero_variance(self) -> None:
        # Caller-supplied standard spin operators on a 4-mode set: a spin-0
        # singlet mode plus a standard spin-1 triplet (S_z basis m=1,0,-1).
        # One particle in sqrt(5/8)|singlet> + sqrt(3/8)|m=1>:
        # <J^2> = 2 * 3/8 = 3/4 EXACTLY, <(J^2)^2> = 4 * 3/8 = 3/2,
        # Var = 3/2 - 9/16 = 15/16 > 0: right expectation, not a sharp spin.
        s = 1 / np.sqrt(2)
        sx1 = np.array([[0, s, 0], [s, 0, s], [0, s, 0]], dtype=complex)
        sy1 = np.array([[0, -1j * s, 0], [1j * s, 0, -1j * s],
                        [0, 1j * s, 0]], dtype=complex)
        sz1 = np.diag([1.0, 0.0, -1.0]).astype(complex)
        def pad(m3):
            out = np.zeros((4, 4), dtype=complex)
            out[1:, 1:] = m3
            return out
        jx, jy, jz = pad(sx1), pad(sy1), pad(sz1)
        orbital = np.zeros((4, 1), dtype=complex)
        orbital[0, 0] = np.sqrt(5.0 / 8.0)
        orbital[1, 0] = np.sqrt(3.0 / 8.0)
        state = CovarianceState.from_slater_frame(orbital)
        e = state.wick_spin_squared_expectation(jx, jy, jz)
        v = state.wick_spin_squared_variance(jx, jy, jz)
        self.assertLess(abs(e.value - 0.75), 1e-14)
        self.assertLess(abs(v.value - 15.0 / 16.0), 1e-13)
        self.assertGreater(v.value.real, 0.5)  # visibly NOT an eigenstate
        # Dense cross-check of both moments.
        fock = DenseFock(4)
        psi = fock.slater(orbital)
        j2 = sum(fock.dgamma(j) @ fock.dgamma(j) for j in (jx, jy, jz))
        e_dense = np.vdot(psi, j2 @ psi)
        v_dense = np.vdot(psi, j2 @ j2 @ psi) - e_dense ** 2
        self.assertLess(abs(e.value - e_dense), 1e-13)
        self.assertLess(abs(v.value - v_dense), 1e-13)

    def test_two_site_up_down_slater_j2_one_variance_one(self) -> None:
        # |up>_A |down>_B = (singlet + triplet_0) / sqrt(2): <J^2> = 1,
        # <(J^2)^2> = 2, Var = 1 — all exact rationals.
        jx, jy, jz = pauli_over_sites(2)
        orbitals = np.zeros((4, 2), dtype=complex)
        orbitals[0, 0] = 1.0  # site A up
        orbitals[3, 1] = 1.0  # site B down
        state = CovarianceState.from_slater_frame(orbitals)
        e = state.wick_spin_squared_expectation(jx, jy, jz)
        v = state.wick_spin_squared_variance(jx, jy, jz)
        self.assertLess(abs(e.value - 1.0), 1e-14)
        self.assertLess(abs(v.value - 1.0), 1e-13)
        fock = DenseFock(4)
        psi = fock.slater(orbitals)
        j2 = sum(fock.dgamma(j) @ fock.dgamma(j) for j in (jx, jy, jz))
        self.assertLess(abs(np.vdot(psi, j2 @ psi) - 1.0), 1e-14)
        self.assertLess(abs(np.vdot(psi, j2 @ j2 @ psi) - 2.0), 1e-13)


# ─── propagation: both entry points, exactness, purity ─────────────────────

@unittest.skipUnless(HAVE_QUANTUM, "tessera built without the quantum subsystem")
class TestPropagation(unittest.TestCase):
    """i dGamma/dt = [h, Gamma] via exact conjugation, and the cobordism
    one-particle transport entry point."""

    def test_evolve_equals_transport_by_the_propagator(self) -> None:
        q = random_orbitals(4, 2, seed=61)
        h = random_hermitian(4, seed=62)
        s_a = CovarianceState.from_slater_frame(q)
        s_b = CovarianceState.from_slater_frame(q)
        s_a.evolve(h, 0.37)
        s_b.apply_transport(CovarianceState.propagator(h, 0.37))
        np.testing.assert_allclose(np.array(s_a.gamma()),
                                   np.array(s_b.gamma()), atol=1e-15)

    def test_diagonal_generator_closed_form(self) -> None:
        # For diagonal h: Gamma_ij(t) = exp(-i (lambda_i - lambda_j) t)
        # Gamma_ij(0) — the hand-solvable exact solution of iGdot = [h, G].
        q = random_orbitals(4, 2, seed=63)
        state = CovarianceState.from_slater_frame(q)
        gamma0 = np.array(state.gamma())
        lam = np.array([0.3, -1.1, 0.7, 2.0])
        t = 0.83
        state.evolve(np.diag(lam).astype(complex), t)
        expected = gamma0 * np.exp(-1j * np.subtract.outer(lam, lam) * t)
        np.testing.assert_allclose(np.array(state.gamma()), expected,
                                   atol=1e-14)

    def test_evolution_matches_dense_fock_evolution(self) -> None:
        # Schrödinger side: |psi'> = wedge of exp(-i h t) phi_k. Every Wick
        # read on the evolved covariance matches the evolved dense state.
        m, n = 4, 2
        q = random_orbitals(m, n, seed=64)
        h = random_hermitian(m, seed=65)
        t = 0.61
        state = CovarianceState.from_slater_frame(q)
        state.evolve(h, t)
        lam, vec = np.linalg.eigh(h)
        u = vec @ np.diag(np.exp(-1j * lam * t)) @ vec.conj().T
        fock = DenseFock(m)
        psi = fock.slater(u @ q)
        for mode in range(m):
            dense_val = np.vdot(psi, fock.number(mode) @ psi)
            self.assertLess(abs(state.wick_occupation(mode).value - dense_val),
                            1e-13)
        parity = np.vdot(psi, fock.parity() @ psi)
        self.assertLess(abs(state.wick_parity().value - parity), 1e-13)

    def test_purity_and_spectrum_across_four_hundred_steps(self) -> None:
        # The long-evolution acceptance bullet: hundreds of steps, purity
        # and the covariance spectrum drift only at round-off.
        m = 6
        q = random_orbitals(m, 3, seed=66)
        state = CovarianceState.from_slater_frame(q)
        h = random_hermitian(m, seed=67)
        h2 = random_hermitian(m, seed=68)
        for step in range(400):
            state.evolve(h if step % 2 == 0 else h2, 0.05)
        self.assertLess(state.purity_defect(), TIGHT)
        self.assertLess(state.hermiticity_defect(), TIGHT)
        eigs = np.linalg.eigvalsh(np.array(state.gamma()))
        np.testing.assert_allclose(np.sort(eigs),
                                   [0, 0, 0, 1, 1, 1], atol=1e-12)
        self.assertTrue(state.purity_certificate(1e-10).holds())

    def test_group_property_of_evolve(self) -> None:
        q = random_orbitals(4, 2, seed=69)
        h = random_hermitian(4, seed=70)
        s_a = CovarianceState.from_slater_frame(q)
        s_b = CovarianceState.from_slater_frame(q)
        s_a.evolve(h, 0.4)
        s_a.evolve(h, 0.6)
        s_b.evolve(h, 1.0)
        np.testing.assert_allclose(np.array(s_a.gamma()),
                                   np.array(s_b.gamma()), atol=1e-13)

    def test_non_hermitian_generator_is_rejected_loudly(self) -> None:
        state = CovarianceState.from_occupations(np.array([1.0, 0.0]))
        bad = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
        with self.assertRaises(ValueError):
            state.evolve(bad, 0.1)
        with self.assertRaises(ValueError):
            CovarianceState.propagator(bad, 0.1)

    def test_shape_mismatch_is_rejected(self) -> None:
        state = CovarianceState.from_occupations(np.array([1.0, 0.0]))
        with self.assertRaises(ValueError):
            state.evolve(np.eye(3, dtype=complex), 0.1)
        with self.assertRaises(ValueError):
            state.apply_transport(np.eye(3, dtype=complex))

    def test_leaky_transport_degrades_purity_visibly(self) -> None:
        # Negative control: a non-unitary transport takes the state off the
        # Slater manifold and the certificate REPORTS it (never repaired).
        q = random_orbitals(4, 2, seed=71)
        state = CovarianceState.from_slater_frame(q)
        leaky = 0.5 * np.eye(4, dtype=complex)
        state.apply_transport(leaky)
        self.assertGreater(state.purity_defect(), 0.1)
        self.assertFalse(state.purity_certificate(1e-9).holds())

    def test_evolution_changes_the_covariance_hash(self) -> None:
        state = CovarianceState.from_occupations(np.array([1.0, 0.0]))
        before = state.covariance_hash()
        state.evolve(random_hermitian(2, seed=72), 0.3)
        self.assertNotEqual(before, state.covariance_hash())


# ─── the mean-field self-consistency loop ──────────────────────────────────

@unittest.skipUnless(HAVE_QUANTUM, "tessera built without the quantum subsystem")
class TestMeanFieldLoop(unittest.TestCase):
    """h = h(Gamma) supplied by the caller; a purity/Gaussianity certificate
    at every iteration; the loop never leaves the Gaussian manifold."""

    M, N = 6, 3

    def hartree(self, h0, g):
        def h_of_gamma(gamma):
            return h0 + g * np.diag(np.diag(gamma).real).astype(complex)
        return h_of_gamma

    def test_purity_certificate_holds_at_every_iteration(self) -> None:
        q = random_orbitals(self.M, self.N, seed=81)
        state = CovarianceState.from_slater_frame(q)
        h0 = random_hermitian(self.M, seed=82)
        reads = state.mean_field_evolve(self.hartree(h0, 0.8), 0.05, 300)
        self.assertEqual(len(reads), 300)
        for read in reads:
            self.assertLess(read.purity_defect, 1e-10)
            self.assertLess(read.hermiticity_defect, 1e-12)
            self.assertLess(read.generator_hermiticity_defect, 1e-12)
            self.assertTrue(read.certificate.holds())
        self.assertAlmostEqual(reads[-1].time, 15.0, places=9)
        self.assertEqual([r.step for r in reads[:3]], [0, 1, 2])

    def test_nonlinearity_is_actually_engaged(self) -> None:
        q = random_orbitals(self.M, self.N, seed=83)
        h0 = random_hermitian(self.M, seed=84)
        s_mf = CovarianceState.from_slater_frame(q)
        s_lin = CovarianceState.from_slater_frame(q)
        s_mf.mean_field_evolve(self.hartree(h0, 1.5), 0.05, 40)
        for _ in range(40):
            s_lin.evolve(h0, 0.05)
        delta = np.linalg.norm(np.array(s_mf.gamma()) - np.array(s_lin.gamma()))
        self.assertGreater(delta, 1e-3)  # the Gamma-dependence matters ...

    def test_loop_stays_on_the_gaussian_manifold(self) -> None:
        # ... yet the state remains EXACTLY a Slater state: rebuild a dense
        # Slater from the evolved Gamma's occupied eigenvectors and every
        # Wick read agrees with it.
        q = random_orbitals(self.M, self.N, seed=85)
        state = CovarianceState.from_slater_frame(q)
        h0 = random_hermitian(self.M, seed=86)
        state.mean_field_evolve(self.hartree(h0, 1.2), 0.05, 60)
        self.assertLess(state.purity_defect(), 1e-11)
        gamma = np.array(state.gamma())
        lam, vec = np.linalg.eigh(gamma)
        occupied = vec[:, lam > 0.5]
        self.assertEqual(occupied.shape[1], self.N)
        fock = DenseFock(self.M)
        psi = fock.slater(occupied)
        for mode in range(self.M):
            dense_val = np.vdot(psi, fock.number(mode) @ psi)
            self.assertLess(abs(state.wick_occupation(mode).value - dense_val),
                            1e-11)
        parity = np.vdot(psi, fock.parity() @ psi)
        self.assertLess(abs(state.wick_parity().value - parity), 1e-10)

    def test_mixed_state_certifies_through_the_spectrum_constraint(self) -> None:
        state = CovarianceState.from_occupations(np.full(4, 0.5))
        h0 = random_hermitian(4, seed=87)
        reads = state.mean_field_evolve(self.hartree(h0, 0.7), 0.05, 50)
        for read in reads:
            self.assertGreater(read.purity_defect, 0.4)     # honestly mixed
            self.assertLess(read.occupation_spectrum_defect, 1e-12)
            self.assertTrue(read.certificate.holds())      # mixed-path claim

    def test_bad_callback_output_is_rejected(self) -> None:
        state = CovarianceState.from_occupations(np.array([1.0, 0.0]))
        with self.assertRaises(ValueError):
            state.mean_field_evolve(
                lambda g: np.array([[0.0, 1.0], [0.0, 0.0]]), 0.1, 3)
        with self.assertRaises(ValueError):
            state.mean_field_evolve(lambda g: np.eye(3, dtype=complex), 0.1, 3)


# ─── property tests: basis rotations and relabelings ───────────────────────

@unittest.skipUnless(HAVE_QUANTUM, "tessera built without the quantum subsystem")
class TestInvariances(unittest.TestCase):
    """Wick values are invariant under one-particle unitaries applied
    consistently to Gamma and the observable coefficients, and behave
    covariantly under mode relabelings."""

    M, N = 5, 2

    def setUp(self) -> None:
        self.q = random_orbitals(self.M, self.N, seed=91)
        self.u = random_unitary(self.M, seed=92)
        self.state = CovarianceState.from_slater_frame(self.q)
        rotated = self.u @ np.array(self.state.gamma()) @ self.u.conj().T
        self.rotated = CovarianceState(rotated)

    def test_parity_is_rotation_invariant(self) -> None:
        a = self.state.wick_parity().value
        b = self.rotated.wick_parity().value
        self.assertLess(abs(a - b), 1e-12)

    def test_gram_determinant_is_rotation_covariant(self) -> None:
        rng = np.random.default_rng(93)
        v = rng.normal(size=(self.M, 2)) + 1j * rng.normal(size=(self.M, 2))
        w = rng.normal(size=(self.M, 2)) + 1j * rng.normal(size=(self.M, 2))
        a = self.state.wick_gram_determinant(v, w).value
        b = self.rotated.wick_gram_determinant(self.u @ v, self.u @ w).value
        self.assertLess(abs(a - b), 1e-12 * max(1.0, abs(a)))

    def test_color_wedge_is_rotation_covariant(self) -> None:
        rng = np.random.default_rng(94)
        c = rng.normal(size=(self.M, 3)) + 1j * rng.normal(size=(self.M, 3))
        a = self.state.wick_color_wedge_squared(c).value
        b = self.rotated.wick_color_wedge_squared(self.u @ c).value
        self.assertLess(abs(a - b), 1e-12 * max(1.0, abs(a)))

    def test_bilinear_moment_is_rotation_covariant(self) -> None:
        rng = np.random.default_rng(95)
        mats = [rng.normal(size=(self.M, self.M))
                + 1j * rng.normal(size=(self.M, self.M)) for _ in range(3)]
        a = self.state.wick_bilinear_moment(mats).value
        b = self.rotated.wick_bilinear_moment(
            [self.u @ m @ self.u.conj().T for m in mats]).value
        self.assertLess(abs(a - b), 1e-11 * max(1.0, abs(a)))

    def test_spin_reads_are_rotation_covariant(self) -> None:
        js = [random_hermitian(self.M, seed=96 + k) for k in range(3)]
        js_rot = [self.u @ j @ self.u.conj().T for j in js]
        e_a = self.state.wick_spin_squared_expectation(*js).value
        e_b = self.rotated.wick_spin_squared_expectation(*js_rot).value
        v_a = self.state.wick_spin_squared_variance(*js).value
        v_b = self.rotated.wick_spin_squared_variance(*js_rot).value
        self.assertLess(abs(e_a - e_b), 1e-11 * max(1.0, abs(e_a)))
        self.assertLess(abs(v_a - v_b), 1e-10 * max(1.0, abs(v_a)))

    def test_relabeling_permutes_the_reads(self) -> None:
        rng = np.random.default_rng(97)
        perm = rng.permutation(self.M)
        p = np.eye(self.M, dtype=complex)[perm, :]  # x'_perm[i] = x_i
        permuted = CovarianceState(
            p @ np.array(self.state.gamma()) @ p.conj().T)
        for mode in range(self.M):
            a = self.state.wick_occupation(mode).value
            b = permuted.wick_occupation(int(np.where(perm == mode)[0][0]))
            self.assertLess(abs(a - b.value), 1e-14)
        subset = [0, 2, 3]
        mapped = [int(np.where(perm == m)[0][0]) for m in subset]
        a = self.state.wick_subset_parity(subset).value
        b = permuted.wick_subset_parity(mapped).value
        self.assertLess(abs(a - b), 1e-13)


# ─── initialization from #769 band projectors ──────────────────────────────

@unittest.skipUnless(HAVE_QUANTUM, "tessera built without the quantum subsystem")
class TestBandProjectorInitialization(unittest.TestCase):
    """Gamma = P consumes real SpectralFiberTracker projector output."""

    @staticmethod
    def _tracker(st):
        # The two-triangle fixture is vertex-transitive, so every band is
        # perfectly delocalized and fails the #808 localization conjunct.
        # The subject here is the COVARIANCE built from a band projector, so
        # the analysis declares the permissive cap, which accepts any
        # MEASURED localization.
        cfg = obs.SpectralFiberConfig()
        cfg.max_localization_excess = 1.0
        return obs.SpectralFiberTracker(st, cfg)

    def test_accepted_band_projector_is_a_slater_covariance(self) -> None:
        st = spacetime_two_triangles()
        tracker = self._tracker(st)
        read = tracker.enumerate_bands([0, 1, 2], 0)
        accepted = [f for f in read.fibers if f.accepted()]
        self.assertGreater(len(accepted), 0)
        fiber = accepted[0]
        state = CovarianceState.from_band_projector(np.array(fiber.projector()))
        self.assertEqual(state.mode_count(), 3)
        # Self-adjoint-path band projector: orthogonal, hence pure Slater.
        self.assertLess(state.purity_defect(), 1e-9)
        self.assertLess(state.hermiticity_defect(), 1e-9)
        self.assertLess(abs(state.particle_number().real - fiber.rank()), 1e-9)
        self.assertTrue(state.purity_certificate(1e-8).holds())
        # Occupations are the projector's diagonal density — within [0, 1].
        occ = np.array(state.occupations()).real
        self.assertTrue(np.all(occ > -1e-12))
        self.assertTrue(np.all(occ < 1 + 1e-12))

    def test_whole_zero_band_of_two_components_reads_two_particles(self) -> None:
        # Degree-0 zero band of the two-triangle complex restricted to all
        # six vertices: rank 2 (one harmonic per component) — Gamma = P
        # carries <N> = 2 and parity +1.
        st = spacetime_two_triangles()
        tracker = self._tracker(st)
        read = tracker.enumerate_bands([0, 1, 2, 10, 11, 12], 0)
        zero_band = read.fibers[0]
        self.assertEqual(zero_band.rank(), 2)
        state = CovarianceState.from_band_projector(
            np.array(zero_band.projector()))
        self.assertLess(abs(state.particle_number().real - 2.0), 1e-9)
        self.assertLess(abs(state.wick_parity().value - 1.0), 1e-9)


# ─── cached Wick reads under the #764 contract ─────────────────────────────

@unittest.skipUnless(HAVE_QUANTUM, "tessera built without the quantum subsystem")
class TestCachedWickReads(unittest.TestCase):
    """wickReadCached: identical to cold recomputation, served only for a
    matching (polynomialId, covarianceHash), invalidated by touched stars,
    and disabled cleanly in replay mode."""

    def setUp(self) -> None:
        self.st = spacetime_two_triangles()
        self.cache = cob.AnalyticCache(self.st)
        self.state = CovarianceState.from_slater_frame(random_orbitals(4, 2, 101))
        self.calls = 0

    def compute(self):
        self.calls += 1
        return self.state.wick_parity()

    def test_cached_equals_cold_and_serves_without_recompute(self) -> None:
        cold = self.state.wick_parity()
        a = self.state.wick_read_cached(self.cache, [0, 1, 2], "parity",
                                      self.compute)
        b = self.state.wick_read_cached(self.cache, [0, 1, 2], "parity",
                                      self.compute)
        self.assertEqual(self.calls, 1)          # second call was a hit
        for read in (a, b):
            self.assertEqual(read.value, cold.value)
            self.assertEqual(read.polynomial_id, cold.polynomial_id)
            self.assertEqual(read.covariance_hash, cold.covariance_hash)
        self.assertGreaterEqual(self.cache.hits, 1)

    def test_gamma_change_forces_recomputation(self) -> None:
        self.state.wick_read_cached(self.cache, [0, 1, 2], "parity",
                                  self.compute)
        self.state.evolve(random_hermitian(4, seed=102), 0.2)
        read = self.state.wick_read_cached(self.cache, [0, 1, 2], "parity",
                                         self.compute)
        self.assertEqual(self.calls, 2)          # state change: cold again
        self.assertEqual(read.covariance_hash, self.state.covariance_hash())

    def test_touched_star_invalidates_the_component(self) -> None:
        self.state.wick_read_cached(self.cache, [0, 1, 2], "parity",
                                  self.compute)
        # Touch the component's star (a metric change on edge (0, 1)).
        for e in self.st.get_edge_list().to_vector():
            ids = {e.get_source().get_id(), e.get_target().get_id()}
            if ids == {0, 1}:
                e.set_length(2.0 + 0j)
        star = cob.TouchedStar()
        star.add_changed_edge(0, 1)
        self.cache.publish(star)
        self.state.wick_read_cached(self.cache, [0, 1, 2], "parity",
                                  self.compute)
        self.assertEqual(self.calls, 2)

    def test_disjoint_component_survives_the_publish(self) -> None:
        self.state.wick_read_cached(self.cache, [10, 11, 12], "parity",
                                  self.compute)
        for e in self.st.get_edge_list().to_vector():
            ids = {e.get_source().get_id(), e.get_target().get_id()}
            if ids == {0, 1}:
                e.set_length(3.0 + 0j)
        star = cob.TouchedStar()
        star.add_changed_edge(0, 1)
        self.cache.publish(star)
        self.state.wick_read_cached(self.cache, [10, 11, 12], "parity",
                                  self.compute)
        self.assertEqual(self.calls, 1)          # sibling survived

    def test_replay_mode_disabled_cache_always_recomputes_identically(self) -> None:
        a = self.state.wick_read_cached(self.cache, [0, 1, 2], "parity",
                                      self.compute)
        self.cache.set_enabled(False)
        b = self.state.wick_read_cached(self.cache, [0, 1, 2], "parity",
                                      self.compute)
        self.assertEqual(self.calls, 2)
        self.assertEqual(a.value, b.value)
        self.assertEqual(a.covariance_hash, b.covariance_hash)

    def test_distinct_polynomials_do_not_collide(self) -> None:
        parity = self.state.wick_read_cached(self.cache, [0, 1, 2], "parity",
                                           self.state.wick_parity)
        number = self.state.wick_read_cached(self.cache, [0, 1, 2],
                                           "total-number",
                                           self.state.wick_total_number)
        self.assertNotEqual(parity.polynomial_id, number.polynomial_id)
        self.assertNotEqual(parity.value, number.value)


# ─── checkpoint serialization and cold replay ──────────────────────────────

@unittest.skipUnless(HAVE_QUANTUM, "tessera built without the quantum subsystem")
class TestSerializationAndReplay(unittest.TestCase):
    """toRecord/fromRecord round trips Gamma exactly; cold replay
    reproduces Gamma and every Wick certificate."""

    def test_round_trip_is_bit_exact(self) -> None:
        state = CovarianceState.from_slater_frame(random_orbitals(5, 2, 111))
        state.evolve(random_hermitian(5, seed=112), 0.7)
        record = state.to_record()
        self.assertEqual(record["schema_version"], 1)
        self.assertEqual(record["record_type"], "covariance-state")
        self.assertTrue(record["number_conserving"])
        replayed = CovarianceState.from_record(record)
        self.assertEqual(replayed.covariance_hash(), state.covariance_hash())
        self.assertEqual(
            np.max(np.abs(np.array(replayed.gamma()) - np.array(state.gamma()))),
            0.0)

    def test_unknown_schema_is_rejected(self) -> None:
        state = CovarianceState.from_occupations(np.array([1.0, 0.0]))
        record = state.to_record()
        record["schema_version"] = 999
        with self.assertRaises(ValueError):
            CovarianceState.from_record(record)

    def test_wrong_record_type_is_rejected(self) -> None:
        state = CovarianceState.from_occupations(np.array([1.0, 0.0]))
        record = state.to_record()
        record["record_type"] = "spectral-fiber"
        with self.assertRaises(ValueError):
            CovarianceState.from_record(record)

    def test_corrupt_payload_is_rejected(self) -> None:
        state = CovarianceState.from_occupations(np.array([1.0, 0.0]))
        record = state.to_record()
        record["gamma_re"] = record["gamma_re"][:-1]
        with self.assertRaises(ValueError):
            CovarianceState.from_record(record)

    def test_cold_replay_reproduces_every_wick_certificate(self) -> None:
        js = [random_hermitian(5, seed=113 + k) for k in range(3)]
        state = CovarianceState.from_slater_frame(random_orbitals(5, 3, 114))
        state.evolve(js[0], 0.4)
        replayed = CovarianceState.from_record(state.to_record())
        rng = np.random.default_rng(115)
        c = rng.normal(size=(5, 3)) + 1j * rng.normal(size=(5, 3))
        pairs = [
            (state.wick_parity(), replayed.wick_parity()),
            (state.wick_total_number(), replayed.wick_total_number()),
            (state.wick_subset_parity([0, 2]), replayed.wick_subset_parity([0, 2])),
            (state.wick_normal_ordered([0, 1], [1, 0]),
             replayed.wick_normal_ordered([0, 1], [1, 0])),
            (state.wick_color_wedge_squared(c), replayed.wick_color_wedge_squared(c)),
            (state.wick_spin_squared_expectation(*js),
             replayed.wick_spin_squared_expectation(*js)),
            (state.wick_spin_squared_variance(*js),
             replayed.wick_spin_squared_variance(*js)),
        ]
        for original, replay in pairs:
            self.assertEqual(original.value, replay.value)
            self.assertEqual(original.polynomial_id, replay.polynomial_id)
            self.assertEqual(original.covariance_hash, replay.covariance_hash)
            self.assertEqual(original.certificate.holds(),
                             replay.certificate.holds())


# ─── the read certificates themselves ──────────────────────────────────────

@unittest.skipUnless(HAVE_QUANTUM, "tessera built without the quantum subsystem")
class TestReadCertificates(unittest.TestCase):
    """Grades, domains, verified regimes, residuals, and identifiers."""

    def test_clean_fixture_reads_are_algebraically_exact_psd(self) -> None:
        state = CovarianceState.from_slater_frame(random_orbitals(4, 2, 121))
        read = state.wick_parity()
        cert = read.certificate
        self.assertEqual(cert.grade, cob.CertificateGrade.AlgebraicallyExact)
        self.assertEqual(cert.domain, cob.CertificateDomain.Static)
        self.assertEqual(cert.regime,
                         cob.CertificateRegime.PositiveSemidefinite)
        self.assertTrue(cert.holds())
        self.assertLess(read.residual, 1e-13)
        self.assertEqual(read.covariance_hash, state.covariance_hash())

    def test_non_hermitian_covariance_reads_non_normal(self) -> None:
        gamma = np.array([[0.5, 0.4], [0.0, 0.5]], dtype=complex)
        read = CovarianceState(gamma).wick_parity()
        self.assertEqual(read.certificate.regime,
                         cob.CertificateRegime.NonNormal)
        self.assertGreater(read.residual, 0.1)
        self.assertFalse(read.certificate.holds())

    def test_invalid_spectrum_reads_hermitian_indefinite(self) -> None:
        read = CovarianceState.from_occupations(
            np.array([1.5, 0.0])).wick_parity()
        self.assertEqual(read.certificate.regime,
                         cob.CertificateRegime.HermitianIndefinite)

    def test_polynomial_ids_are_distinct_and_content_addressed(self) -> None:
        state = CovarianceState.from_slater_frame(random_orbitals(4, 2, 122))
        ids = {
            state.wick_parity().polynomial_id,
            state.wick_total_number().polynomial_id,
            state.wick_occupation(1).polynomial_id,
            state.wick_subset_parity([0, 1]).polynomial_id,
            state.wick_normal_ordered([0], [1]).polynomial_id,
        }
        self.assertEqual(len(ids), 5)
        # Matrix-parametrized reads: different coefficients, different ids.
        j_a = [random_hermitian(4, seed=123 + k) for k in range(3)]
        j_b = [random_hermitian(4, seed=126 + k) for k in range(3)]
        self.assertNotEqual(
            state.wick_spin_squared_expectation(*j_a).polynomial_id,
            state.wick_spin_squared_expectation(*j_b).polynomial_id)
        # ... and the same coefficients give the same id (cache-stable).
        self.assertEqual(
            state.wick_spin_squared_expectation(*j_a).polynomial_id,
            state.wick_spin_squared_expectation(*j_a).polynomial_id)


# ─── polynomial cost: no 2^M object anywhere ───────────────────────────────

@unittest.skipUnless(HAVE_QUANTUM, "tessera built without the quantum subsystem")
class TestPolynomialScaling(unittest.TestCase):
    """M = 200: any dense Fock construction would need 2^200 amplitudes —
    running the full read battery here IS the no-exponential-allocation
    proof."""

    def test_full_read_battery_at_two_hundred_modes(self) -> None:
        m, n = 200, 40
        state = CovarianceState.from_slater_frame(random_orbitals(m, n, 131))
        self.assertEqual(state.mode_count(), m)
        self.assertLess(state.purity_defect(), 1e-11)
        self.assertLess(abs(state.particle_number().real - n), 1e-10)
        # Propagation.
        h = random_hermitian(m, seed=132)
        state.evolve(h, 0.1)
        self.assertLess(state.purity_defect(), 1e-11)
        # Parity of a 40-particle state: exactly +1.
        self.assertLess(abs(state.wick_parity().value - 1.0), 1e-9)
        self.assertTrue(np.isfinite(state.wick_subset_parity(
            list(range(0, 50))).value.real))
        # Gram determinant and color wedge on random frames.
        rng = np.random.default_rng(133)
        v = rng.normal(size=(m, 3)) + 1j * rng.normal(size=(m, 3))
        self.assertTrue(np.isfinite(state.wick_gram_determinant(v, v).value.real))
        self.assertTrue(np.isfinite(
            state.wick_color_wedge_squared(v).value.real))
        # Spin block: sigma/2 over 100 sites — quartic and octic Wick sums
        # at M = 200 (polynomially, in seconds).
        jx, jy, jz = pauli_over_sites(100)
        e = state.wick_spin_squared_expectation(jx, jy, jz)
        v_read = state.wick_spin_squared_variance(jx, jy, jz)
        self.assertTrue(np.isfinite(e.value.real))
        self.assertTrue(np.isfinite(v_read.value.real))
        self.assertLess(abs(e.value.imag), 1e-9)
        # Var(J^2) >= 0 up to round-off on any state.
        self.assertGreater(v_read.value.real, -1e-8)

    def test_mean_field_loop_at_scale(self) -> None:
        m, n = 200, 40
        state = CovarianceState.from_slater_frame(random_orbitals(m, n, 134))
        h0 = random_hermitian(m, seed=135)
        reads = state.mean_field_evolve(
            lambda g: h0 + 0.5 * np.diag(np.diag(g).real).astype(complex),
            0.05, 10)
        for read in reads:
            self.assertLess(read.purity_defect, 1e-10)
            self.assertTrue(read.certificate.holds())


if __name__ == "__main__":
    unittest.main()
