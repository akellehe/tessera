# Copyright (c) 2026 Twin Vector Labs LLC. All rights reserved.

"""Sparse assembly of the exact analytic Regge Hessian (``action_hessian_exact_sparse``).

``ReggeSolver.action_hessian_exact_sparse`` returns the same Hessian as the dense
``action_hessian_exact``, but assembled directly as an Eigen ``SparseMatrix``
(exposed as a COO tuple ``(rows, cols, values, n)``): ∂²S/∂ℓ²_e∂ℓ²_f is nonzero
only for edge pairs e,f that share a hinge (local coupling), so the matrix is
sparse and costs O(nnz)=O(|E|·k) memory instead of O(|E|²).

These tests verify it equals the dense Hessian to machine precision (which also
proves the dense matrix is zero off the sparse pattern — the locality claim),
that the coupling is genuinely local (nnz/row bounded as |E| grows, so density
→ 0 at scale), and that the assembly is symmetric and deterministic.
"""

import os
import unittest

import numpy as np
from scipy.sparse import coo_matrix

import tessera

_HERE = os.path.dirname(os.path.abspath(__file__))
_TOL = 1e-9


def _make_cdt(n):
    sig = tessera.Signature(4, tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    st = tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0, tessera.PREFERRED,
                           tessera.Toroid())
    st.build(n)
    st.materialize_facets()
    return st


def _solver(st):
    return tessera.ReggeSolver(st, tessera.MatterConfiguration())


def _dense(rs):
    return np.array([[complex(z) for z in row] for row in rs.action_hessian_exact()])


def _sparse(rs):
    """Wrap the COO tuple into a dense ndarray plus (nnz, n)."""
    rows, cols, vals, n = rs.action_hessian_exact_sparse()
    dense = coo_matrix(
        (np.array(vals, dtype=complex), (np.array(rows, int), np.array(cols, int))),
        shape=(n, n)).toarray()
    return dense, len(vals), n


class SparseMatchesDenseTest(unittest.TestCase):
    """The issue's core: sparse == dense to machine precision on the pattern."""

    def _check(self, st):
        rs = _solver(st)
        dense = _dense(rs)
        sparse, nnz, n = _sparse(rs)
        self.assertEqual(sparse.shape, dense.shape)
        self.assertEqual(n, dense.shape[0])
        worst = float(np.max(np.abs(dense - sparse))) if n else 0.0
        # Reproduces the FULL dense matrix to machine precision. Since the sparse
        # matrix is zero off its stored pattern, this simultaneously proves the
        # dense Hessian is zero there too (the local-coupling sparsity claim).
        self.assertLess(worst, _TOL,
                        f"sparse Hessian != dense actionHessianExact: max|Δ|={worst:.3e}")
        return dense, sparse, nnz, n

    def test_complex_boost_hinges(self):
        # Lorentzian 4D CDT: triangle hinges with genuinely complex (boost) deficits.
        dense, _, _, _ = self._check(_make_cdt(200))
        self.assertGreater(float(np.max(np.abs(dense.imag))), 0.1,
                           "Hessian should be materially complex (boost hinges)")

    def test_cdt_real_triangle_hinges(self):
        # A second, larger CDT mesh — the sparse==dense identity at scale.
        self._check(_make_cdt(600))


class SparsityTest(unittest.TestCase):
    def test_fewer_nonzeros_than_dense(self):
        _, nnz, n = _sparse(_solver(_make_cdt(200)))
        self.assertLess(nnz, n * n,
                        "sparse Hessian should store fewer entries than the dense E²")

    def test_coupling_is_local_at_scale(self):
        # Local coupling ⇒ nnz ≈ |E|·k with k bounded, so density → 0 as |E|
        # grows. On a larger mesh the Hessian is mostly empty — the memory win.
        _, nnz, n = _sparse(_solver(_make_cdt(600)))
        density = nnz / (n * n)
        per_row = nnz / n
        self.assertLess(density, 0.1,
                        f"expected a sparse Hessian at scale; density={density:.3f}")
        self.assertLess(per_row, n / 4,
                        f"coupling is not local: {per_row:.0f} nnz/row vs |E|={n}")


class SymmetryTest(unittest.TestCase):
    def test_symmetric(self):
        sparse, _, _ = _sparse(_solver(_make_cdt(200)))
        worst = float(np.max(np.abs(sparse - sparse.T)))
        self.assertLess(worst, _TOL, f"sparse Hessian not symmetric: max|S-Sᵀ|={worst:.3e}")


class DeterminismTest(unittest.TestCase):
    def test_coo_identical_across_calls(self):
        rs = _solver(_make_cdt(200))
        self.assertEqual(rs.action_hessian_exact_sparse(),
                         rs.action_hessian_exact_sparse(),
                         "sparse Hessian COO is not deterministic across calls")


if __name__ == "__main__":
    unittest.main()
