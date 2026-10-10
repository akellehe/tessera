# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""Structure-exact Woodbury / secular update helpers (#764).

Woodbury solves of (A + U W) x = b against dense reference solves of the
updated operator; the exactness contract (factors must span the FULL affected
change) with the cold-recompute fallback when they do not; and the secular
rank-one Hermitian eigenvalue update against numpy's dense eigensolve,
including deflation, duplicate eigenvalues, negative rho, and the refusal of
a non-ascending (non-Hermitian-certified) input.
"""

import unittest

import numpy as np

import tessera

cob = tessera.cobordism


def _flat(matrix):
    return [complex(z) for z in np.asarray(matrix).reshape(-1)]


def _random_complex(rng, shape, scale=1.0):
    return scale * (rng.normal(size=shape) + 1j * rng.normal(size=shape))


class TestWoodburySolve(unittest.TestCase):
    def test_matches_dense_solve_of_updated_operator(self):
        rng = np.random.default_rng(23)
        for dim, rank in ((6, 1), (10, 2), (16, 4)):
            base = _random_complex(rng, (dim, dim)) + 3 * np.eye(dim)
            left = _random_complex(rng, (dim, rank))
            right = _random_complex(rng, (rank, dim))
            rhs = _random_complex(rng, dim)

            solver = cob.LowRankUpdate(_flat(base), dim)
            solver.set_update(_flat(left), _flat(right), rank)
            result = solver.solve([complex(z) for z in rhs])

            expected = np.linalg.solve(base + left @ right, rhs)
            np.testing.assert_allclose(result.values, expected,
                                       rtol=1e-10, atol=1e-12)
            cert = result.certificate
            self.assertEqual(cert.grade, cob.CertificateGrade.StructureExact)
            self.assertTrue(cert.holds())
            self.assertLess(cert.residual, 1e-12)
            self.assertGreaterEqual(cert.conditioning, 1.0)

    def test_no_update_is_a_plain_factor_solve(self):
        rng = np.random.default_rng(29)
        dim = 8
        base = _random_complex(rng, (dim, dim)) + 2 * np.eye(dim)
        rhs = _random_complex(rng, dim)
        solver = cob.LowRankUpdate(_flat(base), dim)
        result = solver.solve([complex(z) for z in rhs])
        np.testing.assert_allclose(result.values, np.linalg.solve(base, rhs),
                                   rtol=1e-10, atol=1e-12)

    def test_apply_is_the_updated_operator(self):
        rng = np.random.default_rng(31)
        dim, rank = 5, 2
        base = _random_complex(rng, (dim, dim))
        left = _random_complex(rng, (dim, rank))
        right = _random_complex(rng, (rank, dim))
        x = _random_complex(rng, dim)
        solver = cob.LowRankUpdate(_flat(base), dim)
        solver.set_update(_flat(left), _flat(right), rank)
        got = solver.apply([complex(z) for z in x])
        np.testing.assert_allclose(got, (base + left @ right) @ x,
                                   rtol=1e-12, atol=1e-13)


class TestExactnessContract(unittest.TestCase):
    def test_factors_from_touched_span_a_local_change(self):
        """A change confined to the declared touched star factors exactly and
        the Woodbury path equals the dense solve of the updated operator."""
        rng = np.random.default_rng(37)
        dim = 12
        touched = [2, 3, 7]
        base = _random_complex(rng, (dim, dim)) + 4 * np.eye(dim)
        updated = base.copy()
        # Arbitrary support inside touched rows/columns, including couplings
        # to untouched indices (rows AND columns of the star).
        for i in touched:
            updated[i, :] += _random_complex(rng, dim, 0.5)
            updated[:, i] += _random_complex(rng, dim, 0.5)

        factors = cob.LowRankUpdate.factors_from_touched(
            _flat(base), _flat(updated), dim, touched)
        self.assertTrue(factors.spans_change)
        self.assertLessEqual(factors.rank, 2 * len(touched))

        left = np.array(factors.left).reshape(dim, factors.rank)
        right = np.array(factors.right).reshape(factors.rank, dim)
        # The factored update IS the change, exactly (machine zero).
        np.testing.assert_allclose(left @ right, updated - base,
                                   rtol=0, atol=1e-15)

        solver = cob.LowRankUpdate(_flat(base), dim)
        solver.set_update(factors.left, factors.right, factors.rank)
        self.assertTrue(solver.spans_affected_change(_flat(updated)))
        rhs = _random_complex(rng, dim)
        result = solver.solve([complex(z) for z in rhs])
        np.testing.assert_allclose(result.values,
                                   np.linalg.solve(updated, rhs),
                                   rtol=1e-9, atol=1e-11)

    def test_leaked_change_is_refused_and_cold_fallback_recovers(self):
        """Support outside the declared star: spansChange == False (the
        low-rank path may NOT be called exact), and the documented fallback
        — refactor cold — produces the right answer."""
        rng = np.random.default_rng(41)
        dim = 9
        base = _random_complex(rng, (dim, dim)) + 4 * np.eye(dim)
        updated = base.copy()
        updated[1, 2] += 0.5       # inside the star
        updated[6, 7] += 1e-3      # leak: outside rows/cols {1, 2}

        factors = cob.LowRankUpdate.factors_from_touched(
            _flat(base), _flat(updated), dim, [1, 2])
        self.assertFalse(factors.spans_change)
        self.assertEqual(factors.rank, 0)

        solver = cob.LowRankUpdate(_flat(base), dim)
        # A partial update that misses the leak fails the exactness check.
        partial = cob.LowRankUpdate.factors_from_touched(
            _flat(base), _flat(base + (updated - base) *
                               (np.abs(updated - base) > 1e-2)), dim, [1, 2])
        solver.set_update(partial.left, partial.right, partial.rank)
        self.assertFalse(solver.spans_affected_change(_flat(updated)))

        # Cold-recompute fallback.
        solver.refactor(_flat(updated), dim)
        self.assertEqual(solver.update_rank, 0)
        rhs = _random_complex(rng, dim)
        result = solver.solve([complex(z) for z in rhs])
        np.testing.assert_allclose(result.values,
                                   np.linalg.solve(updated, rhs),
                                   rtol=1e-10, atol=1e-12)

    def test_full_star_on_wider_declared_set_still_exact(self):
        # Declaring MORE than was touched stays exact (rank just larger).
        rng = np.random.default_rng(43)
        dim = 7
        base = _random_complex(rng, (dim, dim)) + 3 * np.eye(dim)
        updated = base.copy()
        updated[4, 4] += 1.0
        factors = cob.LowRankUpdate.factors_from_touched(
            _flat(base), _flat(updated), dim, [0, 4, 5])
        self.assertTrue(factors.spans_change)
        self.assertEqual(factors.rank, 1)  # zero rows/columns are trimmed

    def test_out_of_range_touched_raises(self):
        base = np.eye(3, dtype=complex)
        with self.assertRaises(ValueError):
            cob.LowRankUpdate.factors_from_touched(_flat(base), _flat(base), 3,
                                                 [3])


class TestSecularRankOneEigenvalues(unittest.TestCase):
    def _check_against_dense(self, d, z, rho, atol=1e-10):
        result = cob.LowRankUpdate.rank_one_eigenvalues(
            list(map(float, d)), [complex(v) for v in z], rho)
        z = np.asarray(z, dtype=complex)
        dense = np.linalg.eigvalsh(np.diag(np.asarray(d, dtype=float))
                                   + rho * np.outer(z, z.conj()))
        np.testing.assert_allclose(np.real(result.values), dense,
                                   rtol=0, atol=atol)
        self.assertEqual(result.certificate.grade,
                         cob.CertificateGrade.CertifiedNumerical)
        self.assertEqual(result.certificate.regime,
                         cob.CertificateRegime.HermitianIndefinite)
        self.assertTrue(result.certificate.holds())
        return result

    def test_generic_positive_update(self):
        rng = np.random.default_rng(47)
        d = np.sort(rng.normal(size=8))
        z = rng.normal(size=8) + 1j * rng.normal(size=8)
        self._check_against_dense(d, z, 0.75)

    def test_negative_rho(self):
        rng = np.random.default_rng(53)
        d = np.sort(rng.normal(size=6))
        z = rng.normal(size=6) + 1j * rng.normal(size=6)
        self._check_against_dense(d, z, -1.25)

    def test_deflation_zero_components(self):
        d = [-1.0, 0.0, 1.0, 2.0]
        z = [0.0, 1.0, 0.0, 0.5]  # two exactly-deflated modes
        self._check_against_dense(d, z, 1.0)

    def test_duplicate_eigenvalues(self):
        d = [0.0, 1.0, 1.0, 1.0, 3.0]
        z = [0.3, 0.4, 0.5, 0.6, 0.7]
        self._check_against_dense(d, z, 0.9)

    def test_indefinite_base_spectrum(self):
        # Hermitian INDEFINITE domain (signed eigenvalues) is in scope.
        d = [-5.0, -1.0, 2.0, 7.0]
        z = [1.0, 2.0, 0.5, 1.5]
        self._check_against_dense(d, z, 0.4)

    def test_zero_rho_is_identity(self):
        d = [0.0, 2.0]
        result = cob.LowRankUpdate.rank_one_eigenvalues(d, [1 + 0j, 1 + 0j], 0.0)
        np.testing.assert_allclose(np.real(result.values), d, rtol=0, atol=0)

    def test_interlacing(self):
        rng = np.random.default_rng(59)
        d = np.sort(rng.normal(size=7))
        z = rng.normal(size=7) + 0j
        result = cob.LowRankUpdate.rank_one_eigenvalues(
            list(map(float, d)), [complex(v) for v in z], 2.0)
        lam = np.real(result.values)
        # rho > 0: d_k <= lambda_k <= d_{k+1} (last one above d_max).
        for k in range(len(d) - 1):
            self.assertGreaterEqual(lam[k], d[k] - 1e-12)
            self.assertLessEqual(lam[k], d[k + 1] + 1e-12)
        self.assertGreaterEqual(lam[-1], d[-1] - 1e-12)

    def test_dense_reference_error_reported_on_crossover_fixture(self):
        """The iterative (secular) result is cross-checked against the
        DenseReference kernel on a below-crossover fixture and the measured
        error is attached to its certificate — the reporting contract for
        every iterative result."""
        rng = np.random.default_rng(101)
        n = 24
        d = np.sort(rng.normal(size=n))
        z = rng.normal(size=n) + 1j * rng.normal(size=n)
        rho = 0.6
        result = cob.LowRankUpdate.rank_one_eigenvalues(
            list(map(float, d)), [complex(v) for v in z], rho)

        dense = cob.DenseReference(64)
        self.assertTrue(dense.below_crossover(n))
        updated = np.diag(d) + rho * np.outer(z, np.conj(z))
        reference = dense.spectrum(
            [complex(v) for v in updated.reshape(-1)], n, True)
        error = float(np.max(np.abs(np.real(result.values) -
                                    np.real(reference.values))))
        result.certificate.set_dense_reference_error(error)
        self.assertLess(result.certificate.dense_reference_error, 1e-10)
        self.assertTrue(result.certificate.holds())

    def test_non_ascending_input_refused(self):
        # The Hermitian domain is certified by the caller via ascending real
        # eigenvalues; anything else must be refused, never coerced.
        with self.assertRaises(ValueError):
            cob.LowRankUpdate.rank_one_eigenvalues([1.0, 0.0], [1 + 0j, 1 + 0j],
                                                 1.0)

    def test_size_mismatch_refused(self):
        with self.assertRaises(ValueError):
            cob.LowRankUpdate.rank_one_eigenvalues([0.0, 1.0], [1 + 0j], 1.0)


if __name__ == "__main__":
    unittest.main()
