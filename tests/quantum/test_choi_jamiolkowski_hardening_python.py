"""Hardening for the C++ ChoiJamiolkowski map-state duality ("bending").

Complements tests/quantum/test_choi_jamiolkowski_python.py, which pins the core
duality on 2x2 operators. Here every claim is checked against an independent
numpy oracle on RECTANGULAR / higher-dimensional operators and Haar-ish
unitaries, and the convention / validation surface the base suite leaves open:

  * map-state duality  <psiA|U|psiB> = <vec(U_T)|vec(U)> = Tr(U_T^H.U)  on random
    rectangular U (dA != dB) across several shapes and seeds;
  * vec(|a><b|) = a (x) conj(b) — the separable, Schmidt-rank-1 structure — for
    rectangular, non-unit a, b (the kron identity directly);
  * singularValues == the numpy SVD spectrum, and the SVD of U is literally the
    Schmidt decomposition of vec(U):  vec(U) = sum_k sigma_k (u_k (x) vbar_k);
  * schmidtRank == numpy matrix rank for prescribed-rank rectangular operators,
    and the sigma_max-relative `tol` threshold gates a near-singular value;
  * choiState / choiMatrix on Haar unitaries (d = 2, 3, 5): pure unit state,
    Hermitian J, Tr J = 1, rank 1, BOTH marginals = I/d; plus the non-unitary
    convention Tr J = ||U||_F^2 / d and the marginal U U^H / d;
  * the std::invalid_argument validation surface (-> ValueError).

Conventions (locked, see the header of include/quantum/ChoiJamiolkowski.h):
operators are flat ROW-MAJOR (U[i*dB + j] = U_{ij}); vec(U) = sum_{ij} U_{ij}
|i> (x) |j> is that flatten; vec(|a><b|) = a (x) conj(b).
"""

from __future__ import annotations

import unittest

import numpy as np

try:
    from tessera.quantum import ChoiJamiolkowski
    HAVE_QUANTUM = True
except ImportError:
    HAVE_QUANTUM = False


def _rand_complex(rng: "np.random.Generator", *shape: int) -> np.ndarray:
    """A complex array of the given shape with iid standard-normal parts."""
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def _flat(matrix) -> list:
    """Row-major (C-order) flatten as a list of Python complex numbers."""
    return np.asarray(matrix, dtype=complex).flatten().tolist()


def _haar_unitary(rng: "np.random.Generator", d: int) -> np.ndarray:
    """A Haar-distributed d x d unitary (QR with the standard phase fix)."""
    q, r = np.linalg.qr(_rand_complex(rng, d, d))
    ph = np.diag(r) / np.abs(np.diag(r))
    return q * ph  # rescale each column j by ph[j] -> Haar measure


# Random rectangular and square shapes that exercise dA != dB and dim > 2.
_SHAPES = [(2, 3), (3, 2), (4, 4), (1, 4), (4, 1), (5, 3), (3, 5)]


@unittest.skipUnless(HAVE_QUANTUM, "tessera built without the quantum subsystem")
class TestRectangularDuality(unittest.TestCase):
    """C1 on rectangular / higher-dim operators: the base suite only does 2x2."""

    def test_duality_on_rectangular_operators(self) -> None:
        for dA, dB in _SHAPES:
            for seed in (0, 3, 17):
                with self.subTest(d_a=dA, d_b=dB, seed=seed):
                    rng = np.random.default_rng(seed)
                    U = _rand_complex(rng, dA, dB)
                    psiA = _unit(_rand_complex(rng, dA))
                    psiB = _unit(_rand_complex(rng, dB))

                    # Independent numpy oracle: the bare sandwich.
                    amp_ref = complex(psiA.conj() @ U @ psiB)

                    amp = ChoiJamiolkowski.transition_amplitude(
                        psiA.tolist(), _flat(U), psiB.tolist(), dA, dB)
                    UT = np.array(ChoiJamiolkowski.transition_operator(
                        psiA.tolist(), psiB.tolist(), dA, dB)).reshape(dA, dB)
                    vecU = np.array(ChoiJamiolkowski.vectorize(_flat(U), dA, dB))
                    vecUT = np.array(ChoiJamiolkowski.vectorize(_flat(UT), dA, dB))

                    # transitionAmplitude == <vec(U_T)|vec(U)> == Tr(U_T^H U) == oracle.
                    self.assertAlmostEqual(abs(amp - amp_ref), 0.0, delta=1e-10)
                    self.assertAlmostEqual(
                        abs(amp - np.vdot(vecUT, vecU)), 0.0, delta=1e-10)
                    self.assertAlmostEqual(
                        abs(amp - np.trace(UT.conj().T @ U)), 0.0, delta=1e-10)
                    # U_T = |psiA><psiB|.
                    np.testing.assert_allclose(
                        UT, np.outer(psiA, psiB.conj()), atol=1e-12)

    def test_vec_of_outer_product_is_a_kron_conj_b(self) -> None:
        # vec(|a><b|) = a (x) conj(b), for rectangular, non-unit a and b.
        for dA, dB in [(2, 3), (4, 2), (3, 3), (1, 5)]:
            with self.subTest(d_a=dA, d_b=dB):
                rng = np.random.default_rng(dA * 10 + dB)
                a = _rand_complex(rng, dA)          # deliberately not normalised
                b = _rand_complex(rng, dB)
                UT = ChoiJamiolkowski.transition_operator(
                    a.tolist(), b.tolist(), dA, dB)
                vecUT = np.array(ChoiJamiolkowski.vectorize(UT, dA, dB))
                np.testing.assert_allclose(vecUT, np.kron(a, b.conj()), atol=1e-12)
                # And rank one (separable / disconnected cobordism), rectangular.
                self.assertEqual(ChoiJamiolkowski.schmidt_rank(UT, dA, dB), 1)


@unittest.skipUnless(HAVE_QUANTUM, "tessera built without the quantum subsystem")
class TestSchmidtStructure(unittest.TestCase):
    """Schmidt rank == #nonzero singular values, beyond the base suite's 2x2."""

    def test_singular_values_match_numpy(self) -> None:
        for dA, dB in _SHAPES:
            with self.subTest(d_a=dA, d_b=dB):
                rng = np.random.default_rng(100 + dA * 7 + dB)
                U = _rand_complex(rng, dA, dB)
                sv = ChoiJamiolkowski.singular_values(_flat(U), dA, dB)
                sv_ref = np.linalg.svd(U, compute_uv=False)  # descending
                self.assertEqual(len(sv), min(dA, dB))
                np.testing.assert_allclose(sv, sv_ref, atol=1e-10)

    def test_svd_is_schmidt_decomposition_of_vec(self) -> None:
        # The header's claim: the SVD of U IS the Schmidt decomposition of vec(U).
        for dA, dB in [(3, 4), (4, 3), (5, 5), (2, 6)]:
            with self.subTest(d_a=dA, d_b=dB):
                rng = np.random.default_rng(900 + dA + dB)
                U = _rand_complex(rng, dA, dB)
                u, s, vh = np.linalg.svd(U, full_matrices=False)
                # vec(U) = sum_k sigma_k (u_k (x) conj(v_k)) with conj(v_k) = vh[k].
                recon = sum(s[k] * np.kron(u[:, k], vh[k, :])
                            for k in range(len(s)))
                vecU = np.array(ChoiJamiolkowski.vectorize(_flat(U), dA, dB))
                np.testing.assert_allclose(vecU, recon, atol=1e-10)
                np.testing.assert_allclose(vecU, U.flatten(), atol=1e-12)
                # Full-rank generic operator: Schmidt rank = min(dA, dB).
                self.assertEqual(
                    ChoiJamiolkowski.schmidt_rank(_flat(U), dA, dB), min(dA, dB))

    def test_schmidt_rank_tracks_prescribed_rank(self) -> None:
        # Build an exactly rank-r operator from orthonormal singular vectors and a
        # prescribed positive spectrum; schmidtRank and numpy agree on the rank.
        for dA, dB in [(4, 4), (5, 3), (3, 6)]:
            for r in range(1, min(dA, dB) + 1):
                with self.subTest(d_a=dA, d_b=dB, r=r):
                    rng = np.random.default_rng(7 * dA + 11 * dB + r)
                    uq, _ = np.linalg.qr(_rand_complex(rng, dA, dA))
                    vq, _ = np.linalg.qr(_rand_complex(rng, dB, dB))
                    sig = np.array([3.0 - 0.5 * k for k in range(r)])  # distinct >0
                    U = uq[:, :r] @ np.diag(sig) @ vq[:, :r].conj().T
                    self.assertEqual(
                        ChoiJamiolkowski.schmidt_rank(_flat(U), dA, dB), r)
                    self.assertEqual(np.linalg.matrix_rank(U), r)
                    sv = ChoiJamiolkowski.singular_values(_flat(U), dA, dB)
                    np.testing.assert_allclose(
                        sorted(sv, reverse=True)[:r], sorted(sig, reverse=True),
                        atol=1e-10)
                    np.testing.assert_allclose(
                        sorted(sv, reverse=True)[r:], 0.0, atol=1e-10)

    def test_tol_threshold_gates_a_near_singular_value(self) -> None:
        # diag(1, 1e-7): a tiny-but-nonzero second singular value.  schmidtRank
        # counts it iff it exceeds tol * sigma_max (sigma_max = 1 here).
        U = _flat(np.diag([1.0, 1e-7]))
        self.assertEqual(ChoiJamiolkowski.schmidt_rank(U, 2, 2), 2)            # default 1e-10
        self.assertEqual(ChoiJamiolkowski.schmidt_rank(U, 2, 2, 1e-9), 2)     # 1e-7 > 1e-9
        self.assertEqual(ChoiJamiolkowski.schmidt_rank(U, 2, 2, 1e-5), 1)     # 1e-7 < 1e-5
        self.assertEqual(ChoiJamiolkowski.schmidt_rank(U, 2, 2, 1e-3), 1)
        # The same gating embedded in a rectangular operator.
        rng = np.random.default_rng(42)
        uq, _ = np.linalg.qr(_rand_complex(rng, 3, 3))
        vq, _ = np.linalg.qr(_rand_complex(rng, 2, 2))
        Ur = uq[:, :2] @ np.diag([1.0, 1e-7]) @ vq.conj().T  # 3x2, svals (1, 1e-7)
        self.assertEqual(ChoiJamiolkowski.schmidt_rank(_flat(Ur), 3, 2), 2)
        self.assertEqual(ChoiJamiolkowski.schmidt_rank(_flat(Ur), 3, 2, 1e-5), 1)
        # The zero operator has rank 0 (no positive singular value).
        Z = _flat(np.zeros((3, 4)))
        self.assertEqual(ChoiJamiolkowski.schmidt_rank(Z, 3, 4), 0)
        np.testing.assert_allclose(
            ChoiJamiolkowski.singular_values(Z, 3, 4), 0.0, atol=1e-12)


@unittest.skipUnless(HAVE_QUANTUM, "tessera built without the quantum subsystem")
class TestChoiOnHaarUnitaries(unittest.TestCase):
    """choiState / choiMatrix on Haar unitaries across d, and the non-unitary
    scaling that pins the convention. The base suite checks these properties
    only piecemeal at single dimensions."""

    def test_state_and_matrix_properties_across_dimensions(self) -> None:
        for d in (2, 3, 5):
            n = d * d
            for seed in (1, 8):
                with self.subTest(d=d, seed=seed):
                    rng = np.random.default_rng(seed * 100 + d)
                    Q = _haar_unitary(rng, d)

                    state = np.array(ChoiJamiolkowski.choi_state(_flat(Q), d))
                    self.assertAlmostEqual(np.linalg.norm(state), 1.0, delta=1e-12)
                    np.testing.assert_allclose(state, Q.flatten() / np.sqrt(d),
                                               atol=1e-12)

                    J = np.array(ChoiJamiolkowski.choi_matrix(_flat(Q), d)).reshape(n, n)
                    np.testing.assert_allclose(J, J.conj().T, atol=1e-12)        # Hermitian
                    self.assertAlmostEqual(np.trace(J).real, 1.0, delta=1e-12)   # Tr = 1
                    np.testing.assert_allclose(
                        J, np.outer(state, state.conj()), atol=1e-12)            # = |s><s|
                    evals = np.linalg.eigvalsh(J)
                    self.assertAlmostEqual(evals[-1], 1.0, delta=1e-10)          # rank 1
                    np.testing.assert_allclose(evals[:-1], 0.0, atol=1e-10)

                    # BOTH marginals of a unitary's Choi state are maximally mixed.
                    J4 = J.reshape(d, d, d, d)
                    rho_A = np.einsum("ijkj->ik", J4)   # trace out factor B
                    rho_B = np.einsum("ijil->jl", J4)   # trace out factor A
                    np.testing.assert_allclose(rho_A, np.eye(d) / d, atol=1e-10)
                    np.testing.assert_allclose(rho_B, np.eye(d) / d, atol=1e-10)

    def test_non_unitary_choi_matrix_convention(self) -> None:
        # For a general (non-unitary) U the bend is still a pure state, so J is
        # rank 1 and Hermitian, but Tr J = ||U||_F^2 / d and the A-marginal is
        # U U^H / d (= I/d only when U is unitary) — the statement that pins the
        # row-major (1/sqrt d) vec convention.
        for d in (3, 4):
            with self.subTest(d=d):
                rng = np.random.default_rng(2024 + d)
                U = _rand_complex(rng, d, d)            # generic, not unitary
                n = d * d
                J = np.array(ChoiJamiolkowski.choi_matrix(_flat(U), d)).reshape(n, n)
                np.testing.assert_allclose(J, J.conj().T, atol=1e-12)
                self.assertAlmostEqual(
                    np.trace(J).real, np.linalg.norm(U) ** 2 / d, delta=1e-10)
                evals = np.linalg.eigvalsh(J)           # still a pure state -> rank 1
                np.testing.assert_allclose(evals[:-1], 0.0, atol=1e-10)
                rho_A = np.einsum("ijkj->ik", J.reshape(d, d, d, d))
                np.testing.assert_allclose(rho_A, U @ U.conj().T / d, atol=1e-10)


@unittest.skipUnless(HAVE_QUANTUM, "tessera built without the quantum subsystem")
class TestValidation(unittest.TestCase):
    """The std::invalid_argument surface: pybind maps it to ValueError. The base
    suite exercises none of the throw paths."""

    def test_bad_dimensions_raise_value_error(self) -> None:
        good2 = _flat(np.eye(2))
        with self.assertRaises(ValueError):
            ChoiJamiolkowski.vectorize([1 + 0j, 2 + 0j, 3 + 0j], 2, 2)   # 3 != 2*2
        with self.assertRaises(ValueError):
            ChoiJamiolkowski.vectorize([1 + 0j], 0, 1)                   # dim <= 0
        with self.assertRaises(ValueError):
            ChoiJamiolkowski.singular_values([1 + 0j, 2 + 0j], 2, 2)      # 2 != 4
        with self.assertRaises(ValueError):
            ChoiJamiolkowski.schmidt_rank([1 + 0j], 2, 2)                 # 1 != 4
        with self.assertRaises(ValueError):
            ChoiJamiolkowski.transition_operator([1 + 0j], [1 + 0j, 0j], 2, 2)  # psiA
        with self.assertRaises(ValueError):
            ChoiJamiolkowski.transition_operator([1 + 0j, 0j], [1 + 0j], 2, 2)  # psiB
        with self.assertRaises(ValueError):
            ChoiJamiolkowski.transition_amplitude(
                [1 + 0j, 0j], [1 + 0j], [1 + 0j, 0j], 2, 2)             # U length
        with self.assertRaises(ValueError):
            ChoiJamiolkowski.transition_amplitude(
                [1 + 0j], good2, [1 + 0j, 0j], 2, 2)                    # psiA length
        with self.assertRaises(ValueError):
            ChoiJamiolkowski.choi_state([1 + 0j, 2 + 0j], 2)            # 2 != 2*2
        with self.assertRaises(ValueError):
            ChoiJamiolkowski.choi_state(good2, 0)                        # d <= 0
        with self.assertRaises(ValueError):
            ChoiJamiolkowski.choi_matrix([1 + 0j, 2 + 0j, 3 + 0j], 2)    # 3 != 2*2


if __name__ == "__main__":
    unittest.main()
