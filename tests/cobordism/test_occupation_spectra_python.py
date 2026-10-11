# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""Spectrum-level fermionic second quantization (#764).

Validates, against independent numpy/itertools brute force and the dense-Fock
oracle:
  - occupation subset sums = the exact free N-particle spectrum of d_gamma(h);
  - the direct-sum identity F_-(h_A + h_B) ~ F_-(h_A) (x) F_-(h_B) at the
    spectrum level (merged pairwise sums over particle splits);
  - one-particle direct-sum and hopping-block assembly (Hermitian default
    C' = C^dagger, explicit reverse block for the non-normal regime);
  - input-ordering invariance (spectra are multisets, sorted (Re, Im)).
"""

import itertools
import unittest

import numpy as np

import tessera

cob = tessera.cobordism


def _brute_subset_sums(spectrum, particles):
    return sorted(
        (complex(sum(c)) if c else 0j
         for c in itertools.combinations(spectrum, particles)),
        key=lambda z: (z.real, z.imag))


def _flat(matrix):
    return [complex(z) for z in np.asarray(matrix).reshape(-1)]


class TestSubsetSums(unittest.TestCase):
    def test_matches_brute_force_real(self):
        spectrum = [0.5, 1.0, 2.25, 4.0, 8.5]
        for particles in range(0, 6):
            expected = _brute_subset_sums(spectrum, particles)
            got = cob.OccupationSpectra.subset_sums(
                [complex(x) for x in spectrum], particles)
            np.testing.assert_allclose(got, expected, rtol=0, atol=1e-15)

    def test_matches_brute_force_complex(self):
        # Complex one-particle eigenvalues: the non-normal Lorentzian regime.
        rng = np.random.default_rng(7)
        spectrum = [complex(a, b) for a, b in rng.normal(size=(6, 2))]
        for particles in (1, 2, 3, 6):
            expected = _brute_subset_sums(spectrum, particles)
            got = cob.OccupationSpectra.subset_sums(spectrum, particles)
            np.testing.assert_allclose(got, expected, rtol=0, atol=1e-14)

    def test_vacuum_and_pauli(self):
        spectrum = [1 + 0j, 2 + 0j]
        self.assertEqual(cob.OccupationSpectra.subset_sums(spectrum, 0), [0j])
        # No 3-particle sector on 2 modes (Pauli exclusion).
        self.assertEqual(cob.OccupationSpectra.subset_sums(spectrum, 3), [])

    def test_input_ordering_invariance(self):
        rng = np.random.default_rng(11)
        spectrum = [complex(a, b) for a, b in rng.normal(size=(7, 2))]
        shuffled = list(spectrum)
        rng.shuffle(shuffled)
        for particles in (2, 4):
            a = cob.OccupationSpectra.subset_sums(spectrum, particles)
            b = cob.OccupationSpectra.subset_sums(shuffled, particles)
            np.testing.assert_allclose(a, b, rtol=0, atol=1e-14)

    def test_negative_particles_raises(self):
        with self.assertRaises(ValueError):
            cob.OccupationSpectra.subset_sums([1 + 0j], -1)

    def test_max_terms_refusal(self):
        spectrum = [complex(i) for i in range(30)]
        with self.assertRaises(Exception):
            cob.OccupationSpectra.subset_sums(spectrum, 15, 1000)


class TestFockSums(unittest.TestCase):
    def test_equals_union_of_sectors(self):
        spectrum = [0.5 + 0.25j, 1.5 - 1j, 3 + 0j, -2 + 0.5j]
        union = []
        for particles in range(len(spectrum) + 1):
            union.extend(cob.OccupationSpectra.subset_sums(spectrum, particles))
        union.sort(key=lambda z: (z.real, z.imag))
        got = cob.OccupationSpectra.fock_sums(spectrum)
        self.assertEqual(len(got), 2 ** len(spectrum))
        np.testing.assert_allclose(got, union, rtol=0, atol=1e-14)

    def test_max_terms_refusal(self):
        with self.assertRaises(Exception):
            cob.OccupationSpectra.fock_sums([1 + 0j] * 24, 1000)


class TestDirectSumIdentity(unittest.TestCase):
    def test_spectrum_level_direct_sum_identity(self):
        """subset sums of A + B (direct sum) == merged pairwise sums over
        particle splits — d_gamma(L_A + L_B) = d_gamma(L_A) (x) I + I (x)
        d_gamma(L_B) read at the spectrum level."""
        rng = np.random.default_rng(3)
        factor_a = [complex(a, b) for a, b in rng.normal(size=(4, 2))]
        factor_b = [complex(a, b) for a, b in rng.normal(size=(5, 2))]
        for particles in range(0, 10):
            via_factors = cob.OccupationSpectra.direct_sum_subset_sums(
                factor_a, factor_b, particles)
            direct = cob.OccupationSpectra.subset_sums(
                factor_a + factor_b, particles)
            np.testing.assert_allclose(via_factors, direct, rtol=0, atol=1e-14)

    def test_against_dense_fock_oracle(self):
        """The factor path agrees with the dense-Fock oracle applied to the
        assembled one-particle direct-sum MATRIX (independent eigensolve +
        enumeration)."""
        rng = np.random.default_rng(5)
        a = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
        a = a + a.conj().T
        b = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
        b = b + b.conj().T
        block = cob.OccupationSpectra.direct_sum(_flat(a), 3, _flat(b), 4)
        dense = cob.DenseReference(64)
        spec_a = sorted(np.linalg.eigvalsh(a))
        spec_b = sorted(np.linalg.eigvalsh(b))
        for particles in (1, 2, 3):
            oracle = dense.fock_spectrum(block, 7, particles, True)
            self.assertTrue(oracle.certificate.holds())
            via_factors = cob.OccupationSpectra.direct_sum_subset_sums(
                [complex(x) for x in spec_a], [complex(x) for x in spec_b],
                particles)
            np.testing.assert_allclose(
                sorted(np.real(oracle.values)),
                sorted(np.real(via_factors)), rtol=0, atol=1e-12)


class TestBlockAssembly(unittest.TestCase):
    def test_direct_sum_layout(self):
        a = np.array([[1, 2], [3, 4]], dtype=complex)
        b = np.array([[5]], dtype=complex)
        got = np.array(
            cob.OccupationSpectra.direct_sum(_flat(a), 2, _flat(b), 1)
        ).reshape(3, 3)
        expected = np.zeros((3, 3), dtype=complex)
        expected[:2, :2] = a
        expected[2:, 2:] = b
        np.testing.assert_array_equal(got, expected)

    def test_hopping_block_hermitian_default(self):
        rng = np.random.default_rng(13)
        a = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
        a = a + a.conj().T
        b = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
        b = b + b.conj().T
        c = rng.normal(size=(3, 2)) + 1j * rng.normal(size=(3, 2))
        got = np.array(
            cob.OccupationSpectra.hopping_block(_flat(a), 3, _flat(b), 2,
                                               _flat(c))
        ).reshape(5, 5)
        expected = np.block([[a, c], [c.conj().T, b]])
        np.testing.assert_array_equal(got, expected)
        # Hermitian in, Hermitian out — the hopping term + h.c. at the
        # one-particle level.
        np.testing.assert_allclose(got, got.conj().T, rtol=0, atol=0)

    def test_hopping_block_explicit_reverse_non_normal(self):
        # The non-normal regime: the reverse block is independent data.
        a = np.array([[0, 1], [0, 0]], dtype=complex)
        b = np.array([[2]], dtype=complex)
        c = np.array([[3], [0]], dtype=complex)
        c_rev = np.array([[0, 7]], dtype=complex)
        got = np.array(
            cob.OccupationSpectra.hopping_block(_flat(a), 2, _flat(b), 1,
                                               _flat(c), _flat(c_rev))
        ).reshape(3, 3)
        expected = np.block([[a, c], [c_rev, b]])
        np.testing.assert_array_equal(got, expected)

    def test_hopping_spectrum_against_dense(self):
        """Coupled two-component one-particle spectra: assembled hopping
        block diagonalized by the dense reference matches numpy."""
        rng = np.random.default_rng(17)
        a = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
        a = a + a.conj().T
        b = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
        b = b + b.conj().T
        c = 0.1 * (rng.normal(size=(4, 3)) + 1j * rng.normal(size=(4, 3)))
        flat = cob.OccupationSpectra.hopping_block(_flat(a), 4, _flat(b), 3,
                                                  _flat(c))
        dense = cob.DenseReference(64)
        got = dense.spectrum(flat, 7, True)
        self.assertTrue(got.certificate.holds())
        expected = np.linalg.eigvalsh(np.block([[a, c], [c.conj().T, b]]))
        np.testing.assert_allclose(np.real(got.values), expected,
                                   rtol=0, atol=1e-12)

    def test_dimension_mismatch_raises(self):
        with self.assertRaises(ValueError):
            cob.OccupationSpectra.hopping_block([1 + 0j], 1, [1 + 0j], 1,
                                               [1 + 0j, 2 + 0j])
        with self.assertRaises(ValueError):
            cob.OccupationSpectra.direct_sum([1 + 0j, 2 + 0j], 2, [1 + 0j], 1)


if __name__ == "__main__":
    unittest.main()
