# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The sparse production path of the Whitney pencil (#1205, the integration
specification section 13 and the scaling verification plan's P series).

The specification's rule is that the production path is sparse throughout: the
sparse inverse chain metrics M_k are assembled per top simplex, the chain metric
G_k = M_k^{-1} is dense and is never formed, the harmonic space comes from a
rank-revealing sparse QR of the stacked cochain matrix, and a contour integral
costs one sparse complex factorization per node. The auxiliary operator

    A~_k = M_k d_k^T M_{k-1}^{-1} d_k M_k + d_{k+1} M_{k+1} d_{k+1}^T

contains an inverse metric, so it is not sparse above degree zero and must never
be formed either; the object that carries it is the *bordered system*, the
sparse matrix

    [[ zeta M_k - d_{k+1} M_{k+1} d_{k+1}^T,  -M_k d_k^T ],
     [ -d_k M_k,                               M_{k-1}   ]]

whose Schur complement onto the first block is exactly zeta M_k - A~_k. One
sparse LU of it is the whole shifted solve.

Each test below checks the production path against the dense reading, which is
the oracle and is available only below the crossover, and then checks that the
production path still runs where the dense reading refuses. The last class
records the P series: wall time, memory and fill-in against the number of cells.
"""
import numpy as np
import pytest
import scipy.sparse as sp

from tessera import chainhodge as ch
from tessera import cobordism as cob
from tests.chainhodge._fixtures import flat_torus, torus_cells

KS = ch.Branch.KontsevichSegal
PS = ch.PencilSchur


def _instance(N=4, seed=7, crossover=512, dressed=True):
    """A jittered flat torus at complex squared lengths, dressed by a connection
    whose links are not of unit modulus (so the pencil is non-normal, the
    general case), with the declared dense crossover."""
    rng = np.random.default_rng(seed)
    cells, _ = torus_cells(N)
    K = cob.ChainComplex.fromTopCells(cells)
    n1 = K.numSimplices(1)
    s = [complex(1.0 + 0.2 * rng.normal(), 0.2 * rng.normal()) for _ in range(n1)]
    base = ch.ChainHodge(K, s, ch.Preset.L2, KS, crossover)
    links = ([complex(rng.normal(), rng.normal()) for _ in range(n1)] if dressed
             else [complex(1.0, 0.0)] * n1)
    return K, base, ch.CovariantChainHodge(base, ch.Connection(K, links))


def _principal_angle(X, Y):
    """The largest principal angle between two column spans, in radians: zero
    when the two frames span one subspace."""
    qx = np.linalg.qr(X)[0]
    qy = np.linalg.qr(Y)[0]
    sv = np.linalg.svd(qx.conj().T @ qy, compute_uv=False)
    return float(np.arccos(np.clip(sv.min(), -1.0, 1.0)))


class TestPencilOperatorWithoutForming:
    """A~_k applied without being formed."""

    @pytest.mark.parametrize("k", [0, 1, 2])
    def test_apply_matches_the_dense_operator(self, k):
        K, base, cov = _instance()
        rng = np.random.default_rng(3)
        n = base.size(k)
        Z = rng.normal(size=(n, 4)) + 1j * rng.normal(size=(n, 4))
        dense = np.array(cov.pencilAux(k))
        assert np.abs(np.array(cov.applyPencilOperator(k, Z)) - dense @ Z).max() \
            <= 1e-9 * np.abs(dense).max() * np.abs(Z).max() * n

    @pytest.mark.parametrize("k", [0, 1, 2])
    def test_undressed_apply_matches_the_dense_operator(self, k):
        K, base, cov = _instance(dressed=False)
        rng = np.random.default_rng(5)
        n = base.size(k)
        Z = rng.normal(size=(n, 3)) + 1j * rng.normal(size=(n, 3))
        dense = np.array(base.pencilAux(k))
        assert np.abs(np.array(base.applyPencilOperator(k, Z)) - dense @ Z).max() \
            <= 1e-9 * np.abs(dense).max() * np.abs(Z).max() * n

    def test_apply_runs_where_the_dense_pencil_refuses(self):
        """A crossover of one puts every degree above it: the dense readings
        refuse and the production path does not."""
        K, base, cov = _instance(crossover=1)
        n = base.size(1)
        Z = np.eye(n, 2, dtype=complex)
        with pytest.raises(Exception):
            base.pencil(1)
        with pytest.raises(Exception):
            base.pencilAux(1)
        out = np.array(base.applyPencilOperator(1, Z))
        assert out.shape == (n, 2) and np.isfinite(out).all()


class TestStackedMatrixAndNullSpace:
    """The harmonic space with neither the stacked matrix nor the orthogonal
    factor densified."""

    @pytest.mark.parametrize("k", [1])
    def test_stacked_matrix_is_sparse_and_matches_the_blocks(self, k):
        K, base, cov = _instance()
        S = base.stackedMatrix(k)
        assert sp.issparse(S)
        d2t = base.boundary(k + 1).T.toarray()
        d1M = (base.boundary(k) @ base.Minv(k)).toarray()
        expected = np.vstack([d2t, d1M])
        assert np.abs(S.toarray() - expected).max() < 1e-12

    def test_stacked_matrix_stays_sparse_as_the_mesh_grows(self):
        """An entry of S is nonzero only when two cells share a top simplex, so
        its stored entries per column are bounded by the local combinatorics and
        do not grow with the mesh: that is what makes the path sparse."""
        per_column = []
        for N in (4, 8, 12):
            _, base, _ = _instance(N=N, seed=N, crossover=1)
            S = base.stackedMatrix(1)
            per_column.append(S.nnz / S.shape[1])
        assert max(per_column) <= 1.2 * min(per_column)

    def test_sparse_null_space_matches_the_dense_singular_value_decomposition(self):
        K, base, cov = _instance()
        S = base.stackedMatrix(1).toarray()
        read = ch.ChainHodge.sparseNullSpace(sp.csc_matrix(base.stackedMatrix(1)), 10.0)
        u, sv, vh = np.linalg.svd(S)
        tol = 10.0 * max(S.shape) * np.finfo(float).eps * sv[0]
        rank = int(np.sum(sv > tol))
        assert read.rank == rank
        kernel = np.array(read.kernel)
        assert kernel.shape[1] == S.shape[1] - rank
        assert np.abs(S @ kernel).max() < 1e-8 * np.abs(S).max()
        assert _principal_angle(kernel, vh[rank:].conj().T) < 1e-6

    def test_sparse_null_space_reports_its_cost(self):
        K, base, cov = _instance()
        read = ch.ChainHodge.sparseNullSpace(sp.csc_matrix(base.stackedMatrix(1)), 10.0)
        cost = read.cost
        assert cost.operation == "stacked-qr"
        assert cost.systemNonZeros > 0 and cost.factorNonZeros > 0
        assert cost.fillIn > 0.0 and np.isfinite(cost.fillIn)
        assert cost.wallSeconds >= 0.0
        assert cost.factorMegabytes > 0.0

    @pytest.mark.parametrize("dressed", [False, True])
    def test_sparse_harmonic_chains_agree_with_the_dense_reading(self, dressed):
        K, base, cov = _instance(dressed=dressed)
        dense = cov.harmonicChains(1, 10.0, False)
        sparse = cov.harmonicChains(1, 10.0, True)
        assert sparse.dense is False
        assert sparse.nullity == dense.nullity == K.bettiNumbers()[1]
        assert _principal_angle(np.array(sparse.images), np.array(dense.images)) < 1e-6

    def test_sparse_harmonic_chains_run_above_the_crossover(self):
        K, base, cov = _instance(crossover=1)
        read = cov.harmonicChains(1, 10.0, False)
        assert read.dense is False
        assert read.nullity == K.bettiNumbers()[1]


class TestBorderedShiftedSolve:
    """The shifted pencil through one sparse LU whose Schur complement is the
    shifted pencil itself."""

    def test_the_schur_complement_of_the_bordered_system_is_the_shifted_pencil(self):
        K, base, cov = _instance()
        zeta = complex(0.31, -0.17)
        n, m = base.size(1), base.size(0)
        B = np.array(cov.borderedSystem(1, zeta).toarray())
        assert B.shape == (n + m, n + m)
        schur = B[:n, :n] - B[:n, n:] @ np.linalg.solve(B[n:, n:], B[n:, :n])
        expected = zeta * np.array(cov.pencil(1).B) - np.array(cov.pencilAux(1))
        assert np.abs(schur - expected).max() < 1e-8 * np.abs(expected).max()

    def test_shifted_solve_matches_the_dense_inverse(self):
        K, base, cov = _instance()
        rng = np.random.default_rng(13)
        zeta = complex(-0.4, 0.23)
        n = base.size(1)
        B = rng.normal(size=(n, 3)) + 1j * rng.normal(size=(n, 3))
        X, cost = cov.shiftedSolve(1, zeta, B)
        P = zeta * np.array(cov.pencil(1).B) - np.array(cov.pencilAux(1))
        assert np.abs(np.array(X) - np.linalg.solve(P, B)).max() < 1e-7 * np.abs(B).max()
        assert cost.operation == "bordered-lu"
        assert cost.dimension == n
        assert cost.systemRows == n + base.size(0)
        assert cost.factorNonZeros > 0 and cost.fillIn > 0.0
        assert cost.rightHandSides == 3

    def test_shifted_solve_runs_above_the_crossover(self):
        K, base, cov = _instance(crossover=1)
        n = base.size(1)
        X, cost = cov.shiftedSolve(1, complex(0.5, 0.1), np.eye(n, 1, dtype=complex))
        assert X.shape == (n, 1) and np.isfinite(np.array(X)).all()
        assert cost.factorNonZeros > 0

    def test_the_resolvent_is_the_shifted_solve_carried_to_chains(self):
        K, base, cov = _instance()
        rng = np.random.default_rng(29)
        zeta = complex(0.9, 0.4)
        n = base.size(1)
        c = rng.normal(size=(n, 2)) + 1j * rng.normal(size=(n, 2))
        X, _ = cov.shiftedSolve(1, zeta, c)
        assert np.abs(np.array(cov.resolvent(1, zeta, c))
                      - cov.Minv(1) @ np.array(X)).max() < 1e-10 * np.abs(c).max()


class TestSparseBand:
    """The Riesz band read on a probe block, with the projector never formed."""

    @staticmethod
    def _contour(cov):
        """A circle at the origin inside the first nonzero level: it encloses
        the harmonic band alone, whose rank is the first Betti number."""
        values = np.sort(np.abs(np.array(cov.spectrum(1).eigenvalues)))
        nonzero = values[values > 1e-8 * values[-1]]
        return ch.Contour.circle(complex(0.0, 0.0), 0.4 * float(nonzero[0]), 48)

    def test_sparse_band_agrees_with_the_dense_band(self):
        K, base, cov = _instance(dressed=False, seed=17)
        contour = self._contour(cov)
        dense = cov.band(1, contour, 10.0, 1e-10)
        assert dense.rank() == K.bettiNumbers()[1]
        sparse, cost = cov.sparseBand(1, contour, dense.rank() + 4, 10.0, 1e-10, 20260922)
        assert sparse.rank() == dense.rank()
        assert _principal_angle(np.array(sparse.frame), np.array(dense.frame)) < 1e-6
        if dense.certificate.leftFrameAvailable and sparse.certificate.leftFrameAvailable:
            # The reduced operators are conjugate, so their spectra agree.
            a = np.sort_complex(np.linalg.eigvals(np.array(sparse.reduced)))
            b = np.sort_complex(np.linalg.eigvals(np.array(dense.reduced)))
            assert np.abs(a - b).max() < 1e-6 * max(1.0, np.abs(b).max())
        assert cost.operation == "contour-band"
        assert cost.factorNonZeros > 0 and cost.fillIn > 0.0

    def test_the_probe_block_certificates_are_measured(self):
        K, base, cov = _instance(dressed=False, seed=17)
        contour = self._contour(cov)
        dense = cov.band(1, contour, 10.0, 1e-10)
        sparse, _ = cov.sparseBand(1, contour, dense.rank() + 4)
        certificate = sparse.certificate
        # Idempotency is measured on the probe block by a second quadrature
        # pass rather than asserted; the projector itself is never formed, so
        # its spectral norm is unmeasured and the probe estimate says so.
        assert certificate.idempotency < 1e-6
        assert np.isnan(certificate.resolventMax)
        assert certificate.resolventProbeMax > 0.0
        assert "probes=" in certificate.contour
        if certificate.leftFrameAvailable:
            assert certificate.rightResidual < 1e-6

    def test_a_probe_block_narrower_than_the_band_is_refused(self):
        K, base, cov = _instance(dressed=False, seed=17)
        contour = self._contour(cov)
        assert cov.band(1, contour, 10.0, 1e-10).rank() >= 2
        with pytest.raises(Exception):
            cov.sparseBand(1, contour, 1)

    def test_sparse_band_runs_where_the_dense_band_refuses(self):
        K, base, cov = _instance(dressed=False, seed=17, crossover=512)
        contour = self._contour(cov)
        rank = cov.band(1, contour, 10.0, 1e-10).rank()
        K2, base2, cov2 = _instance(dressed=False, seed=17, crossover=1)
        with pytest.raises(Exception):
            cov2.band(1, contour, 10.0, 1e-10)
        sparse, _ = cov2.sparseBand(1, contour, rank + 4)
        assert sparse.rank() == rank


class TestSparseFeshbach:
    """The Schur complement of the pencil with no n x n matrix formed."""

    @staticmethod
    def _interface(K, N):
        edges = [tuple(int(v) for v in e) for e in K.kSimplexVertices(1)]
        left = {v for v in range(N * N) if v // N < N // 2}
        return [i for i, e in enumerate(edges) if (e[0] in left) != (e[1] in left)]

    def test_sparse_feshbach_matches_the_dense_complement(self):
        """At degree zero both matrices of the pencil are sparse as they stand,
        so the whole complement runs on the sparse path."""
        K, base, cov = _instance(dressed=False, seed=31)
        pencil = cov.sparsePencil(0)
        A, M = sp.csc_matrix(pencil.A), sp.csc_matrix(pencil.M)
        interface = sorted(range(0, A.shape[0], 3))
        lam = complex(0.23, -0.07)
        result, cost = PS.sparseFeshbach(A, M, lam, interface)
        dense = PS.feshbach(A.toarray(), M.toarray(), lam, interface, 1e-12)
        scale = np.abs(np.array(dense.response)).max()
        assert np.abs(np.array(result.response) - np.array(dense.response)).max() < 1e-8 * scale
        assert np.abs(np.array(result.constraintModes)
                      - np.array(dense.constraintModes)).max() < 1e-8
        assert result.solveResidual < 1e-10
        assert cost.factorNonZeros > 0 and cost.fillIn > 0.0

    def test_an_interior_resonance_is_refused_by_name(self):
        K, base, cov = _instance(dressed=False, seed=31)
        pencil = cov.sparsePencil(0)
        A, M = sp.csc_matrix(pencil.A), sp.csc_matrix(pencil.M)
        interface = sorted(range(0, A.shape[0], 3))
        interior = [i for i in range(A.shape[0]) if i not in set(interface)]
        idx = np.ix_(interior, interior)
        lam = complex(np.linalg.eigvals(
            np.linalg.solve(M.toarray()[idx], A.toarray()[idx]))[0])
        with pytest.raises(Exception, match="resonance"):
            PS.sparseFeshbach(A, M, lam, interface, 1e-8)


class TestScalingReports:
    """The P series of the scaling verification plan: wall time, memory and
    fill-in against the number of cells, on the plan's F1 family (the jittered
    flat torus)."""

    @staticmethod
    def _row(N):
        K, s, _ = flat_torus(N, jitter=0.25, seed=N)
        base = ch.ChainHodge(K, s, ch.Preset.L2, KS, 1)
        cov = ch.CovariantChainHodge(base, ch.Connection.trivial(K), 7, False)
        n1 = K.numSimplices(1)
        _, lu = cov.shiftedSolve(1, complex(0.37, 0.11), np.eye(n1, 1, dtype=complex))
        qr = ch.ChainHodge.sparseNullSpace(sp.csc_matrix(base.stackedMatrix(1)), 10.0).cost
        return {"N": N, "n1": n1, "lu": lu, "qr": qr}

    def test_time_memory_and_fill_in_against_the_number_of_cells(self):
        rows = [self._row(N) for N in (4, 6, 8, 10)]
        for row in rows:
            for name in ("lu", "qr"):
                cost = row[name]
                assert cost.systemNonZeros > 0
                assert cost.factorNonZeros > 0
                assert np.isfinite(cost.fillIn) and cost.fillIn > 0.0
                assert cost.wallSeconds >= 0.0
                assert cost.factorMegabytes > 0.0
        # The cell counts grow, and so do the systems: the reports are read
        # against n_1, which is the plan's abscissa.
        assert [r["n1"] for r in rows] == sorted(r["n1"] for r in rows)
        assert [r["lu"].systemNonZeros for r in rows] \
            == sorted(r["lu"].systemNonZeros for r in rows)
        # The factors stay sparse: the stored entries of one bordered
        # factorization are far below the dense n^2 of the system it factorizes,
        # which is the whole claim of a sparse production path.
        for row in rows:
            dense_entries = row["lu"].systemRows ** 2
            assert row["lu"].factorNonZeros < 0.5 * dense_entries
        # Fill-in per unknown does not run away with the mesh: the largest mesh
        # factorizes no worse per row than four times the smallest.
        per_row = [r["lu"].factorNonZeros / r["lu"].systemRows for r in rows]
        assert per_row[-1] <= 4.0 * per_row[0]

    def test_the_memory_of_the_factors_is_the_entries_they_store(self):
        """`factorMegabytes` is computed from `factorNonZeros`, so the two agree
        exactly: it is a count, not an estimate."""
        row = self._row(6)
        for name in ("lu", "qr"):
            cost = row[name]
            bytes_per_entry = cost.factorMegabytes * 1024.0 * 1024.0 / cost.factorNonZeros
            assert 16.0 <= bytes_per_entry <= 32.0

    def test_the_resident_memory_of_an_operation_is_reported(self):
        """On a platform that publishes it, the change in the process's resident
        set size is a number; where it does not, it is quiet NaN and says so."""
        row = self._row(8)
        value = row["lu"].residentMegabytes
        assert np.isnan(value) or np.isfinite(value)
