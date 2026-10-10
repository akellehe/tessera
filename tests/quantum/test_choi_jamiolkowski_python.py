"""Python acceptance tests for the C++ ChoiJamiolkowski utility class.

The Choi–Jamiołkowski "bending" maps an operator U to a bipartite state
vec(U). numpy is used as the independent oracle for every claim:

* C1 — the map–state (Hilbert–Schmidt) duality identity
        ⟨psiA|U|psiB⟩ = ⟨vec(U_T)|vec(U)⟩ = Tr(U_T^H·U),  U_T = |psiA⟩⟨psiB|.
* C2 — Schmidt rank = number of nonzero singular values: the rank-one
        transition operator is separable (rank 1); the identity bends to the
        cup |00⟩+|11⟩ and σ_x to |01⟩+|10⟩, both rank 2 (entangled).

Conventions (locked): operators are flat ROW-MAJOR, so a dA×dB matrix U has
U[i*dB + j] = U_{ij}; vec(U) = Σ_{ij} U_{ij} |i⟩⊗|j⟩ is that flatten; and
vec(|a⟩⟨b|) = a ⊗ conj(b).
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


@unittest.skipUnless(HAVE_QUANTUM, "tessera built without the quantum subsystem")
class TestTransitionAmplitudeDuality(unittest.TestCase):
    """C1: ⟨psiA|U|psiB⟩ = ⟨vec(U_T)|vec(U)⟩ = Tr(U_T^H·U), vs a numpy oracle."""

    def test_duality_identity_matches_numpy(self) -> None:
        for seed in (0, 1, 7, 12345):
            with self.subTest(seed=seed):
                rng = np.random.default_rng(seed)
                U = _rand_complex(rng, 2, 2)
                psiA = _unit(_rand_complex(rng, 2))
                psiB = _unit(_rand_complex(rng, 2))

                # Independent numpy oracle ------------------------------------
                amp_ref = psiA.conj() @ U @ psiB
                U_T = np.outer(psiA, psiB.conj())          # |psiA><psiB|
                vecU = U.flatten()                          # row-major
                vecUT = U_T.flatten()
                inner_ref = np.vdot(vecUT, vecU)            # <vec(U_T)|vec(U)>
                trace_ref = np.trace(U_T.conj().T @ U)      # Tr(U_T^H U)
                # The oracle must itself be self-consistent.
                self.assertAlmostEqual(abs(amp_ref - inner_ref), 0.0, delta=1e-12)
                self.assertAlmostEqual(abs(amp_ref - trace_ref), 0.0, delta=1e-12)

                # C++ under test ----------------------------------------------
                amp = ChoiJamiolkowski.transition_amplitude(
                    psiA.tolist(), _flat(U), psiB.tolist(), 2, 2)
                UT_cpp = ChoiJamiolkowski.transition_operator(
                    psiA.tolist(), psiB.tolist(), 2, 2)
                vecU_cpp = ChoiJamiolkowski.vectorize(_flat(U), 2, 2)
                vecUT_cpp = ChoiJamiolkowski.vectorize(UT_cpp, 2, 2)

                inner_cpp = np.vdot(np.array(vecUT_cpp), np.array(vecU_cpp))
                UT_mat = np.array(UT_cpp).reshape(2, 2)
                trace_cpp = np.trace(UT_mat.conj().T @ U)

                # transitionAmplitude == <vec(U_T)|vec(U)> == Tr(U_T^H U),
                # each equal to the independent numpy amplitude, to 1e-12.
                self.assertAlmostEqual(abs(amp - amp_ref), 0.0, delta=1e-12)
                self.assertAlmostEqual(abs(amp - inner_cpp), 0.0, delta=1e-12)
                self.assertAlmostEqual(abs(amp - trace_cpp), 0.0, delta=1e-12)

                # Building blocks match their definitions.
                np.testing.assert_allclose(UT_mat, U_T, atol=1e-12)        # |psiA><psiB|
                np.testing.assert_allclose(np.array(vecU_cpp), vecU, atol=1e-12)  # flatten


@unittest.skipUnless(HAVE_QUANTUM, "tessera built without the quantum subsystem")
class TestSchmidtRankAndSingularValues(unittest.TestCase):
    """C2: Schmidt rank = #nonzero singular values; separable vs entangled."""

    def test_transition_operator_is_separable_rank_one(self) -> None:
        rng = np.random.default_rng(2024)
        psiA = _unit(_rand_complex(rng, 2))
        psiB = _unit(_rand_complex(rng, 2))

        U_T = ChoiJamiolkowski.transition_operator(
            psiA.tolist(), psiB.tolist(), 2, 2)
        sv = ChoiJamiolkowski.singular_values(U_T, 2, 2)

        # |psiA><psiB| with unit psiA, psiB has singular values (1, 0).
        self.assertEqual(len(sv), 2)
        self.assertAlmostEqual(sv[0], 1.0, delta=1e-12)
        self.assertAlmostEqual(sv[1], 0.0, delta=1e-12)
        self.assertEqual(ChoiJamiolkowski.schmidt_rank(U_T, 2, 2), 1)

        # Cross-check the spectrum against a numpy SVD of |psiA><psiB|.
        sv_ref = np.linalg.svd(np.outer(psiA, psiB.conj()), compute_uv=False)
        np.testing.assert_allclose(
            sorted(sv, reverse=True), sorted(sv_ref, reverse=True), atol=1e-12)

    def test_identity_bends_to_cup_rank_two(self) -> None:
        I2 = _flat(np.eye(2))

        # vec(I_2) = |00> + |11> (the cup / unnormalized Bell state).
        vec = ChoiJamiolkowski.vectorize(I2, 2, 2)
        np.testing.assert_allclose(
            np.array(vec), np.array([1, 0, 0, 1], dtype=complex), atol=1e-12)

        sv = ChoiJamiolkowski.singular_values(I2, 2, 2)
        np.testing.assert_allclose(
            sorted(sv, reverse=True), [1.0, 1.0], atol=1e-12)
        self.assertEqual(ChoiJamiolkowski.schmidt_rank(I2, 2, 2), 2)

    def test_pauli_x_is_entangled_rank_two(self) -> None:
        sx = _flat(np.array([[0, 1], [1, 0]]))

        # vec(σ_x) = |01> + |10>.
        vec = ChoiJamiolkowski.vectorize(sx, 2, 2)
        np.testing.assert_allclose(
            np.array(vec), np.array([0, 1, 1, 0], dtype=complex), atol=1e-12)

        sv = ChoiJamiolkowski.singular_values(sx, 2, 2)
        np.testing.assert_allclose(
            sorted(sv, reverse=True), [1.0, 1.0], atol=1e-12)
        self.assertEqual(ChoiJamiolkowski.schmidt_rank(sx, 2, 2), 2)


@unittest.skipUnless(HAVE_QUANTUM, "tessera built without the quantum subsystem")
class TestChoiStateAndMatrix(unittest.TestCase):
    """Choi state (U⊗I)|Φ⁺⟩ = (1/√d) vec(U) and matrix J(U) = |state⟩⟨state|."""

    def test_choi_state_of_identity_is_maximally_entangled(self) -> None:
        # (1/√d) vec(I_d) = (1/√d) Σ_k |k,k⟩ — the maximally entangled |Φ⁺⟩.
        for d in (2, 3, 4):
            with self.subTest(d=d):
                state = np.array(ChoiJamiolkowski.choi_state(_flat(np.eye(d)), d))
                np.testing.assert_allclose(
                    state, np.eye(d).flatten() / np.sqrt(d), atol=1e-12)
                self.assertAlmostEqual(np.linalg.norm(state), 1.0, delta=1e-12)

    def test_choi_state_is_normalised_vec(self) -> None:
        rng = np.random.default_rng(11)
        for d in (2, 3):
            with self.subTest(d=d):
                U = _rand_complex(rng, d, d)
                state = np.array(ChoiJamiolkowski.choi_state(_flat(U), d))
                np.testing.assert_allclose(
                    state, U.flatten() / np.sqrt(d), atol=1e-12)

    def test_choi_matrix_is_pure_unit_trace_for_unitary(self) -> None:
        # J(U) = |Φ_U⟩⟨Φ_U|: Hermitian, Tr = 1, rank 1 for unitary U; equals
        # the outer product of choiState with itself.
        rng = np.random.default_rng(3)
        d, n = 2, 4
        Q, _ = np.linalg.qr(_rand_complex(rng, d, d))   # a unitary
        J = np.array(ChoiJamiolkowski.choi_matrix(_flat(Q), d)).reshape(n, n)
        np.testing.assert_allclose(J, J.conj().T, atol=1e-12)        # Hermitian
        self.assertAlmostEqual(np.trace(J).real, 1.0, delta=1e-12)   # Tr = 1
        evals = np.linalg.eigvalsh(J)
        self.assertAlmostEqual(evals[-1], 1.0, delta=1e-12)          # rank 1
        np.testing.assert_allclose(evals[:-1], 0.0, atol=1e-12)
        state = np.array(ChoiJamiolkowski.choi_state(_flat(Q), d))
        np.testing.assert_allclose(J, np.outer(state, state.conj()), atol=1e-12)

    def test_choi_matrix_marginal_is_maximally_mixed_for_unitary(self) -> None:
        # Tracing out the second factor of J(unitary U) gives I/d — the
        # convention-independent fact behind InteractionSimulation's Q-bookkeeping.
        rng = np.random.default_rng(5)
        d = 4
        Q, _ = np.linalg.qr(_rand_complex(rng, d, d))
        J4 = np.array(ChoiJamiolkowski.choi_matrix(_flat(Q), d)).reshape(d, d, d, d)
        rho_A = np.einsum("ijkj->ik", J4)               # trace over factor B
        np.testing.assert_allclose(rho_A, np.eye(d) / d, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
