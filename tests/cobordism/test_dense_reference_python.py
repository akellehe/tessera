# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""Dense reference kernels and the configurable crossover (#764).

The dense kernels are fixture references: below the crossover they agree with
numpy to machine precision and report measured residuals; at or above the
crossover they refuse rather than becoming a silent dense fallback. The
self-adjoint eigensolver is applied only after Hermiticity is VERIFIED.
"""

import unittest

import numpy as np

import tessera

cob = tessera.cobordism


def _flat(matrix):
    return [complex(z) for z in np.asarray(matrix).reshape(-1)]


class TestCrossover(unittest.TestCase):
    def test_refusal_at_and_above_crossover(self):
        dense = cob.DenseReference(4)
        self.assertEqual(dense.crossover_dimension, 4)
        self.assertTrue(dense.below_crossover(3))
        self.assertFalse(dense.below_crossover(4))
        matrix = _flat(np.eye(4, dtype=complex))
        with self.assertRaises(ValueError):
            dense.solve(matrix, 4, [1 + 0j] * 4)
        with self.assertRaises(ValueError):
            dense.spectrum(matrix, 4, True)

    def test_crossover_is_configurable(self):
        dense = cob.DenseReference(4)
        dense.set_crossover_dimension(8)
        matrix = _flat(np.eye(4, dtype=complex))
        result = dense.solve(matrix, 4, [1 + 0j] * 4)
        np.testing.assert_allclose(result.values, np.ones(4), rtol=0,
                                   atol=1e-15)

    def test_invalid_crossover_refused(self):
        with self.assertRaises(ValueError):
            cob.DenseReference(0)


class TestDenseSolve(unittest.TestCase):
    def test_matches_numpy(self):
        rng = np.random.default_rng(61)
        dim = 12
        a = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
        a += 3 * np.eye(dim)
        b = rng.normal(size=dim) + 1j * rng.normal(size=dim)
        dense = cob.DenseReference(64)
        result = dense.solve(_flat(a), dim, [complex(z) for z in b])
        np.testing.assert_allclose(result.values, np.linalg.solve(a, b),
                                   rtol=1e-11, atol=1e-13)
        cert = result.certificate
        self.assertEqual(cert.grade, cob.CertificateGrade.StructureExact)
        self.assertTrue(cert.holds())
        self.assertGreaterEqual(cert.conditioning, 1.0)


class TestDenseSpectrum(unittest.TestCase):
    def test_hermitian_path_matches_numpy(self):
        rng = np.random.default_rng(67)
        dim = 10
        a = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
        a = a + a.conj().T
        dense = cob.DenseReference(64)
        result = dense.spectrum(_flat(a), dim, True)
        np.testing.assert_allclose(np.real(result.values),
                                   np.linalg.eigvalsh(a), rtol=0, atol=1e-12)
        self.assertEqual(result.certificate.regime,
                         cob.CertificateRegime.HermitianIndefinite)
        self.assertTrue(result.certificate.holds())
        # Verified-unitary eigenbasis: conditioning 1.
        self.assertEqual(result.certificate.conditioning, 1.0)

    def test_self_adjoint_request_on_non_hermitian_falls_to_general(self):
        """A self-adjoint REQUEST is honored only after verification: a
        non-normal operator runs the general solver and says so."""
        a = np.array([[0, 1], [0, 0]], dtype=complex)  # nilpotent, non-normal
        dense = cob.DenseReference(64)
        result = dense.spectrum(_flat(a), 2, True)
        self.assertEqual(result.certificate.regime,
                         cob.CertificateRegime.NonNormal)
        np.testing.assert_allclose(result.values, [0j, 0j], rtol=0, atol=1e-12)

    def test_general_path_matches_numpy_sorted(self):
        rng = np.random.default_rng(71)
        dim = 8
        a = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
        dense = cob.DenseReference(64)
        result = dense.spectrum(_flat(a), dim, False)
        expected = sorted(np.linalg.eigvals(a),
                          key=lambda z: (z.real, z.imag))
        np.testing.assert_allclose(result.values, expected, rtol=1e-9,
                                   atol=1e-10)
        self.assertEqual(result.certificate.regime,
                         cob.CertificateRegime.NonNormal)
        self.assertGreaterEqual(result.certificate.conditioning, 1.0)


class TestDenseFockOracle(unittest.TestCase):
    def test_fock_spectrum_matches_brute_force(self):
        rng = np.random.default_rng(73)
        dim = 5
        h = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
        h = h + h.conj().T
        dense = cob.DenseReference(64)
        eigenvalues = np.linalg.eigvalsh(h)
        import itertools
        for particles in range(dim + 1):
            oracle = dense.fock_spectrum(_flat(h), dim, particles, True)
            expected = sorted(
                sum(c) if c else 0.0
                for c in itertools.combinations(eigenvalues, particles))
            np.testing.assert_allclose(np.real(oracle.values), expected,
                                       rtol=0, atol=1e-12)
            self.assertTrue(oracle.certificate.holds())


if __name__ == "__main__":
    unittest.main()
