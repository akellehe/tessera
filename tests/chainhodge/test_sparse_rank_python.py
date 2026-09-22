# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""SparseRank (#1204): numerical ranks of sparse matrices with the singular
values on either side of the decision, against dense SVDs of the same
matrices. `kernel` reads a sparse matrix's kernel from a thresholded sparse QR;
`congruence` reads the rank of P = C^T X^{+-1} C at the exact rank of an
integer matrix C without forming P."""
import math

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.linalg import subspace_angles

from tessera import chainhodge as ch


def _planted(m, n, values, seed):
    """A dense m x n complex matrix with the given singular values (descending,
    padded with zeros) and random singular vectors, stored sparse."""
    rng = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rng.normal(size=(m, m)) + 1j * rng.normal(size=(m, m)))
    V, _ = np.linalg.qr(rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))
    k = min(m, n)
    sv = np.zeros(k)
    sv[:len(values)] = values
    A = U[:, :k] @ np.diag(sv) @ V[:, :k].conj().T
    return sp.csc_matrix(A), sv


def _incidence_torus(N):
    """The boundary maps of an N x N triangulated torus (integer, exact ranks)."""
    from tessera import cobordism as cob
    from tests.chainhodge._fixtures import torus_cells
    cells, _ = torus_cells(N)
    K = cob.ChainComplex.fromTopCells(cells)
    hodge = ch.ChainHodge(K, [1.0 + 0j] * K.numSimplices(1))
    return K, hodge.boundary(1).toarray().real, hodge.boundary(2).toarray().real


def _random_symmetric(n, seed, density=0.2):
    """A sparse, complex symmetric, well-conditioned matrix."""
    rng = np.random.default_rng(seed)
    B = sp.random(n, n, density=density, random_state=seed, format="csc")
    B = B + 1j * sp.random(n, n, density=density, random_state=seed + 1, format="csc")
    X = (B + B.T) * 0.3 + sp.diags(2.0 + rng.uniform(0.5, 1.5, n) + 0.3j * rng.normal(size=n))
    return sp.csc_matrix(X)


class TestKernel:
    def test_planted_split_is_read_on_both_sides(self):
        """sigma_r = 1e-6 is kept, sigma_{r+1} = 5e-14 is discarded (below the
        tolerance 10 * 60 * eps ~ 1.3e-13): the sparse read reproduces both,
        the second through ||A N|| on the computed kernel."""
        values = list(np.geomspace(1.0, 1e-6, 50)) + [5e-14, 3e-15]
        A, sv = _planted(60, 55, values, seed=3)
        read = ch.SparseRank.kernel(A)
        split = read.split
        assert not split.dense
        assert split.rank == 50 == split.at
        assert split.largest == pytest.approx(1.0, rel=1e-10)
        assert split.sigmaAt == pytest.approx(1e-6, rel=1e-6)
        assert split.sigmaNext == pytest.approx(5e-14, rel=0.1)
        assert split.gap == pytest.approx(2e7, rel=0.1)
        assert read.basis.shape == (55, 5)
        np.testing.assert_allclose(read.basis.conj().T @ read.basis, np.eye(5), atol=1e-13)
        _, _, vh = np.linalg.svd(A.toarray())
        # The kernel of a matrix within ||R22|| ~ 5e-14 of A: its angle to the
        # SVD's is at most ~ 5e-14 / sigma_r = 5e-8 radians.
        assert np.max(subspace_angles(read.basis, vh[50:].conj().T)) < 1e-6

    def test_agrees_with_dense_split(self):
        A, _ = _planted(40, 48, list(np.geomspace(3.0, 0.01, 30)), seed=5)
        sparse = ch.SparseRank.kernel(A).split
        dense = ch.SparseRank.fromSingularValues(np.linalg.svd(A.toarray(), compute_uv=False), 40, 48)
        assert dense.dense and dense.rank == sparse.rank == 30
        assert sparse.tolerance == pytest.approx(dense.tolerance, rel=1e-10)
        assert sparse.sigmaAt == pytest.approx(dense.sigmaAt, rel=1e-9)
        assert sparse.sigmaNext < 1e-13 and dense.sigmaNext < 1e-13

    def test_full_column_rank_and_empty_rows(self):
        """Nothing discarded: sigma_{r+1} = 0 and the gap is infinite. An empty
        row carries no singular value and is ignored."""
        A, _ = _planted(30, 20, list(np.geomspace(2.0, 0.5, 20)), seed=7)
        A = sp.vstack([A, sp.csc_matrix((3, 20))]).tocsc()
        read = ch.SparseRank.kernel(A)
        assert read.split.rank == 20 and read.basis.shape == (20, 0)
        assert read.split.sigmaNext == 0.0 and math.isinf(read.split.gap)
        assert read.split.sigmaAt == pytest.approx(0.5, rel=1e-9)

    def test_torus_stacked_matrix(self):
        """The Whitney stacked matrix of a Euclidean torus: nullity b_1 = 2."""
        from tests.chainhodge._fixtures import flat_torus
        K, s, _ = flat_torus(8, 0.25, False, seed=2)
        hodge = ch.ChainHodge(K, s)
        S = sp.vstack([hodge.boundary(2).T, hodge.boundary(1) @ hodge.Minv(1)]).tocsc()
        read = ch.SparseRank.kernel(S)
        sv = np.linalg.svd(S.toarray(), compute_uv=False)
        assert read.basis.shape[1] == 2
        assert read.split.sigmaAt == pytest.approx(sv[-3], rel=1e-9)
        assert read.split.largest == pytest.approx(sv[0], rel=1e-10)
        assert read.split.sigmaNext < 1e-13 * sv[0]


class TestCongruence:
    @pytest.mark.parametrize("inverse", [False, True])
    @pytest.mark.parametrize("which", ["d1", "d1T", "d2", "d2T"])
    def test_rank_conditions_shape_against_dense(self, which, inverse):
        """P = C^T X^{+-1} C with C an incidence matrix of the 5 x 5 torus (or
        its transpose) and X random complex symmetric: the split at the exact
        rank of C reproduces the dense SVD of the formed P."""
        K, d1, d2 = _incidence_torus(5)
        C = {"d1": d1, "d1T": d1.T, "d2": d2, "d2T": d2.T}[which]
        rho = int(np.linalg.matrix_rank(C))
        X = _random_symmetric(C.shape[0], seed=11)
        Xd = X.toarray()
        P = C.T @ (np.linalg.solve(Xd, C) if inverse else Xd @ C)
        sv = np.linalg.svd(P, compute_uv=False)
        split = ch.SparseRank.congruence(sp.csc_matrix(C.astype(complex)), X, inverse, rho)
        assert not split.dense and split.at == rho
        assert split.rank == rho == int(np.sum(sv > split.tolerance))
        assert split.largest == pytest.approx(sv[0], rel=1e-9)
        assert split.sigmaAt == pytest.approx(sv[rho - 1], rel=1e-8)
        if rho < C.shape[1]:
            assert 0.0 <= split.sigmaNext < 1e-12 * sv[0]
            assert split.gap > 1e8
        else:
            assert split.sigmaNext == 0.0 and math.isinf(split.gap)

    def test_near_failure_is_resolved(self):
        """X tuned so that one direction of range(C) is nearly X-null: the
        smallest required singular value falls far below the others and is
        still measured to its dense value."""
        K, d1, _ = _incidence_torus(4)
        C = d1.T                                       # P = d1 X d1^T
        rho = int(np.linalg.matrix_rank(C))
        rng = np.random.default_rng(13)
        n = C.shape[0]
        base = np.diag(rng.uniform(1.0, 2.0, n)) + 0j
        # V spans range(C); the form X restricted to it is D, one entry 1e-7.
        U, s, _ = np.linalg.svd(C, full_matrices=False)
        V = U[:, :rho]
        D = np.diag(np.r_[np.ones(rho - 1), 1e-7])
        Xd = base + V @ (D - V.T @ base @ V) @ V.T
        assert np.linalg.cond(Xd) < 1e12
        split = ch.SparseRank.congruence(sp.csc_matrix(C.astype(complex)), sp.csc_matrix(Xd), False, rho)
        sv = np.linalg.svd(C.T @ Xd @ C, compute_uv=False)
        assert sv[rho - 1] < 1e-5 * sv[0]
        assert split.sigmaAt == pytest.approx(sv[rho - 1], rel=1e-6)
        assert split.rank == rho

    def test_rank_mismatch_refused(self):
        _, d1, _ = _incidence_torus(4)
        X = _random_symmetric(d1.shape[0], seed=3)
        with pytest.raises(RuntimeError):
            ch.SparseRank.congruence(sp.csc_matrix(d1.astype(complex)), X, False, d1.shape[0])


class TestFromSingularValues:
    def test_conventions(self):
        sv = np.array([4.0, 2.0, 1e-3, 1e-17])
        split = ch.SparseRank.fromSingularValues(sv, 4, 6)
        assert split.rank == 3 == split.at
        assert split.sigmaAt == 1e-3 and split.sigmaNext == 1e-17
        assert split.gap == pytest.approx(1e14)
        at0 = ch.SparseRank.fromSingularValues(sv, 4, 6, 10.0, 0)
        assert math.isinf(at0.sigmaAt) and at0.sigmaNext == 4.0
        past = ch.SparseRank.fromSingularValues(sv, 4, 6, 10.0, 4)
        assert past.sigmaNext == 0.0 and math.isinf(past.gap)
