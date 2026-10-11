"""Python acceptance tests for the C++ MutualInformation utility class.

Skips cleanly when tessera was built without TESSERA_QUANTUM=1.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

try:
    from tessera.quantum import MutualInformation
    HAVE_QUANTUM = True
except ImportError:
    HAVE_QUANTUM = False


@unittest.skipUnless(HAVE_QUANTUM, "tessera built without TESSERA_QUANTUM=1")
class TestVonNeumannEntropy(unittest.TestCase):
    """Pure tests on the von Neumann entropy helper."""

    def test_maximally_mixed_qubit(self) -> None:
        rho = 0.5 * np.eye(2, dtype=complex)
        self.assertAlmostEqual(
            MutualInformation.von_neumann_entropy(rho), math.log(2), places=12)

    def test_pure_qubit(self) -> None:
        rho = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)
        self.assertAlmostEqual(
            MutualInformation.von_neumann_entropy(rho), 0.0, places=12)

    def test_bell_state_marginal(self) -> None:
        """Bell |Φ+⟩ on two qubits has marginal ρ_A = I/2; S(ρ_A) = ln 2."""
        bell = (np.array([1, 0, 0, 1], dtype=complex) / math.sqrt(2)).reshape(4, 1)
        rho_AB = bell @ bell.conj().T
        # Partial trace over B: reshape to (2,2,2,2) then sum over B.
        rho_tensor = rho_AB.reshape(2, 2, 2, 2)
        rho_A = np.trace(rho_tensor, axis1=1, axis2=3)
        self.assertAlmostEqual(
            MutualInformation.von_neumann_entropy(rho_A), math.log(2), places=12)

    def test_zero_eigenvalues_handled(self) -> None:
        """Eigenvalues below tol shouldn't trip the log(0) branch."""
        rho = np.diag([1.0 - 1e-15, 1e-15, 0.0, 0.0]).astype(complex)
        # Tiny eigenvalues contribute ~0; total entropy is finite.
        S = MutualInformation.von_neumann_entropy(rho)
        self.assertGreaterEqual(S, 0.0)
        self.assertLess(S, math.log(4))


@unittest.skipUnless(HAVE_QUANTUM, "tessera built without TESSERA_QUANTUM=1")
class TestEdgeLength(unittest.TestCase):
    """ℓ = -log(I) with infinity floor."""

    def test_unit_mi_gives_zero_length(self) -> None:
        self.assertAlmostEqual(MutualInformation.edge_length(1.0), 0.0)

    def test_small_mi_gives_large_length(self) -> None:
        self.assertAlmostEqual(
            MutualInformation.edge_length(0.1), -math.log(0.1), places=12)

    def test_below_cutoff_returns_infinity(self) -> None:
        self.assertTrue(math.isinf(MutualInformation.edge_length(1e-15)))
        self.assertTrue(MutualInformation.edge_length(1e-15) > 0)

    def test_cutoff_threshold(self) -> None:
        eps = 1e-6
        # Just below cutoff → inf
        self.assertTrue(math.isinf(MutualInformation.edge_length(eps / 2, eps)))
        # Just above cutoff → finite
        self.assertFalse(math.isinf(MutualInformation.edge_length(eps * 2, eps)))


@unittest.skipUnless(HAVE_QUANTUM, "tessera built without TESSERA_QUANTUM=1")
class TestVonNeumannEntropySpectrum(unittest.TestCase):
    """Spectrum overload: entropy straight from an eigenvalue list."""

    def test_matches_manual_formula(self) -> None:
        spec = [0.7, 0.2, 0.08, 0.02]
        expected = -sum(p * math.log(p) for p in spec)
        self.assertAlmostEqual(
            MutualInformation.von_neumann_entropy(spec), expected, places=12)

    def test_matches_diagonal_density_matrix(self) -> None:
        """Spectrum overload must agree with the matrix overload on diag(ρ)."""
        spec = [0.5, 0.3, 0.15, 0.05]
        from_spec = MutualInformation.von_neumann_entropy(spec)
        from_matrix = MutualInformation.von_neumann_entropy(
            np.diag(spec).astype(complex))
        self.assertAlmostEqual(from_spec, from_matrix, places=12)

    def test_uniform_spectrum(self) -> None:
        self.assertAlmostEqual(
            MutualInformation.von_neumann_entropy([0.5, 0.5]), math.log(2),
            places=12)

    def test_zero_entries_handled(self) -> None:
        self.assertAlmostEqual(
            MutualInformation.von_neumann_entropy([1.0, 0.0, 0.0]), 0.0, places=12)


@unittest.skipUnless(HAVE_QUANTUM, "tessera built without TESSERA_QUANTUM=1")
class TestEdgeLengthMatrix(unittest.TestCase):
    """Vectorised ℓ = -log(I) over a matrix of MI values."""

    def test_matches_scalar_elementwise(self) -> None:
        rng = np.random.default_rng(0)
        m = np.abs(rng.standard_normal((5, 5))) * 0.3 + 1e-3
        out = np.asarray(MutualInformation.edge_length(m))
        ref = -np.log(m)
        self.assertTrue(np.allclose(out, ref, atol=1e-12))

    def test_below_cutoff_entries_are_infinite(self) -> None:
        m = np.array([[0.5, 1e-15], [1e-15, 0.5]])
        out = np.asarray(MutualInformation.edge_length(m, 1e-10))
        self.assertTrue(math.isinf(out[0, 1]) and out[0, 1] > 0)
        self.assertFalse(math.isinf(out[0, 0]))

    def test_shape_preserved(self) -> None:
        m = np.full((3, 4), 0.5)
        out = np.asarray(MutualInformation.edge_length(m))
        self.assertEqual(out.shape, (3, 4))


if __name__ == "__main__":
    unittest.main()
