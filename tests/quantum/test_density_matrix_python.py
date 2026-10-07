# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The N-qubit density-matrix utilities bound from quantum/DensityMatrix.hpp.

* ``partialTrace(rho, n, keep)`` agrees with an independent numpy partial
  trace on random states, keeps the listed order of the kept qubits, returns
  the trace for an empty ``keep``, and refuses malformed arguments;
* ``randomCorrelatedState(n, seed)`` returns a density matrix (Hermitian,
  unit trace, positive semidefinite) that a seed reproduces and in which
  every pair of qubits shares mutual information.

Skips cleanly when tessera was built without the quantum subsystem.
"""

from __future__ import annotations

import itertools
import unittest

import numpy as np

try:
    from tessera.quantum import (MutualInformation, mutualInformation, partialTrace,
                                 randomCorrelatedState)
    HAVE_QUANTUM = True
except ImportError:
    HAVE_QUANTUM = False


def numpy_partial_trace(rho, n, keep):
    """Reference partial trace by tensor contraction (qubit 0 most significant)."""
    tensor = rho.reshape((2,) * (2 * n))
    axes = list(range(2 * n))
    for q in range(n):
        if q not in keep:
            axes[n + q] = axes[q]
    out = [axes[q] for q in keep] + [axes[n + q] for q in keep]
    d = 2 ** len(keep)
    return np.einsum(tensor, axes, out).reshape(d, d)


def ginibre_state(rng, n):
    d = 2 ** n
    g = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    rho = g @ g.conj().T
    return rho / np.trace(rho).real


@unittest.skipUnless(HAVE_QUANTUM, "tessera built without the quantum subsystem")
class TestPartialTrace(unittest.TestCase):

    def test_agrees_with_tensor_contraction(self):
        rng = np.random.default_rng(1)
        for n in (2, 3, 4, 5):
            rho = ginibre_state(rng, n)
            for k in range(1, n + 1):
                for keep in itertools.permutations(range(n), k):
                    if k > 2 and keep != tuple(sorted(keep)):
                        continue                     # orders of larger sets: sorted only
                    np.testing.assert_allclose(partialTrace(rho, n, list(keep)),
                                               numpy_partial_trace(rho, n, keep), atol=1e-13)

    def test_kept_order_swaps_the_factors(self):
        a, b = ginibre_state(np.random.default_rng(2), 1), ginibre_state(np.random.default_rng(3), 1)
        rho = np.kron(np.kron(a, np.eye(2) / 2), b)
        np.testing.assert_allclose(partialTrace(rho, 3, [0, 2]), np.kron(a, b), atol=1e-14)
        np.testing.assert_allclose(partialTrace(rho, 3, [2, 0]), np.kron(b, a), atol=1e-14)

    def test_empty_keep_is_the_trace(self):
        rho = ginibre_state(np.random.default_rng(4), 3)
        out = partialTrace(rho, 3, [])
        self.assertEqual(out.shape, (1, 1))
        self.assertAlmostEqual(out[0, 0].real, 1.0, places=13)

    def test_refuses_malformed_arguments(self):
        rho = np.eye(8, dtype=complex) / 8
        for n, keep in ((3, [0, 0]), (3, [3]), (3, [-1]), (2, [0]), (0, [])):
            with self.assertRaises(ValueError, msg=(n, keep)):
                partialTrace(rho, n, keep)


@unittest.skipUnless(HAVE_QUANTUM, "tessera built without the quantum subsystem")
class TestRandomCorrelatedState(unittest.TestCase):

    def test_is_a_density_matrix(self):
        for n in (1, 2, 3, 4):
            rho = randomCorrelatedState(n, 7 + n)
            self.assertEqual(rho.shape, (2 ** n, 2 ** n))
            np.testing.assert_allclose(rho, rho.conj().T, atol=1e-14)
            self.assertAlmostEqual(np.trace(rho).real, 1.0, places=13)
            self.assertGreater(np.linalg.eigvalsh(rho).min(), -1e-14)

    def test_seed_reproduces_and_distinguishes(self):
        np.testing.assert_array_equal(randomCorrelatedState(3, 11), randomCorrelatedState(3, 11))
        self.assertGreater(np.abs(randomCorrelatedState(3, 11)
                                  - randomCorrelatedState(3, 12)).max(), 1e-3)

    def test_every_pair_shares_mutual_information(self):
        rho = randomCorrelatedState(4, 5)
        for i, j in itertools.combinations(range(4), 2):
            self.assertGreater(mutualInformation(partialTrace(rho, 4, [i, j]), 2, 2), 1e-6)

    def test_refuses_a_bad_qubit_count(self):
        for n in (0, 31):
            with self.assertRaises(ValueError):
                randomCorrelatedState(n, 1)

    def test_entropy_of_the_reduced_state_matches_numpy(self):
        rho = randomCorrelatedState(3, 9)
        reduced = partialTrace(rho, 3, [1])
        w = np.linalg.eigvalsh(reduced)
        self.assertAlmostEqual(MutualInformation.vonNeumannEntropy(reduced),
                               float(-(w * np.log(w)).sum()), places=12)


if __name__ == "__main__":
    unittest.main()
