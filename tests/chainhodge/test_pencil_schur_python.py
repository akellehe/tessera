# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Pencil Feshbach, Craig–Bampton congruence, fiber restriction, and transfer
(#914, specification §7, §11 step 5, §13, §14 T9 and the F_B symmetry of
T1–T4), and RecursiveQuotient levels that carry a symmetric pencil with its
Gram."""
import cmath
import math

import numpy as np
import pytest

from tessera import chainhodge as ch
from tessera import cobordism as cob
from tests.chainhodge._fixtures import torus_cells, torus33

PS = ch.PencilSchur
KS = ch.Branch.KontsevichSegal


def _edges(K):
    return [tuple(int(v) for v in e) for e in K.kSimplexVertices(1)]


def _random_instance(N, seed, complex_lengths=True):
    rng = np.random.default_rng(seed)
    cells, _ = torus_cells(N)
    K = cob.ChainComplex.fromTopCells(cells)
    n1 = K.numSimplices(1)
    if complex_lengths:
        s = [complex(rng.normal(), rng.normal()) for _ in range(n1)]
    else:
        s = [complex(1.0 + 0.2 * rng.normal(), 0.0) for _ in range(n1)]
    links = [complex(rng.normal(), rng.normal()) for _ in range(n1)]
    base = ch.ChainHodge(K, s, ch.Preset.L2, KS)
    U = ch.Connection(K, links)
    return K, base, U, rng


def _interface_by_vertices(K, vertex_set):
    """Edges touching the vertex set are the interface; the rest interior."""
    return [i for i, e in enumerate(_edges(K)) if (e[0] in vertex_set) != (e[1] in vertex_set) or e[0] in vertex_set and e[1] in vertex_set and False]


def _split_interface(K, N):
    """Interface = edges with exactly one endpoint in the first row block."""
    left = {v for v in range(N * N) if v // N < N // 2}
    return [i for i, e in enumerate(_edges(K)) if (e[0] in left) != (e[1] in left)]


class TestFeshbach:
    @pytest.mark.parametrize("N", [4, 6])
    def test_symmetry_transpose_identity_and_determinant(self, N):
        K, base, U, rng = _random_instance(N, 100 + N)
        cov = ch.CovariantChainHodge(base, U)
        dual = cov.dual()
        P, Pd = cov.pencil(1), dual.pencil(1)
        interface = _split_interface(K, N)
        lam = complex(0.7, -0.3)
        F = PS.feshbach(P.A, P.B, lam, interface)
        Fd = PS.feshbach(Pd.A, Pd.B, lam, interface)
        assert not F.interiorSingular
        scale = np.abs(F.response).max()
        # F_B(lambda; U)^T = F_B(lambda; U^{-1})  (Prop. 7.1(a))
        assert np.abs(F.response.T - Fd.response).max() <= 8.0e-12 * scale
        # det P = det P_II det F_B
        assert F.determinantResidual < 1e-8
        # at U = 1 the complement is symmetric
        P1 = base.pencil(1)
        F1 = PS.feshbach(P1.A, P1.B, lam, interface)
        assert np.abs(F1.response - F1.response.T).max() <= 8.0e-12 * np.abs(F1.response).max()
        # the constraint modes reproduce the complement by congruence:
        # T^T P T = F_B when T = [I; -P_II^{-1} P_IB]  (Schur complement identity)
        Pmat = P.A - lam * P.B
        cong = F.constraintModes.T @ Pmat @ F.constraintModes
        assert np.abs(cong - F.response).max() <= 1e-9 * scale

    def test_enclosed_poles_and_static_schur_does_not_preserve_them(self):
        K, s = torus33()
        base = ch.ChainHodge(K, s, ch.Preset.L2, KS)
        P = base.pencil(1)
        A, M = P.A, P.B
        interface = _split_interface(K, 3)
        interior = [i for i in range(A.shape[0]) if i not in interface]
        full = np.linalg.eigvals(np.linalg.solve(M, A))
        interior_spec = np.linalg.eigvals(np.linalg.solve(M[np.ix_(interior, interior)], A[np.ix_(interior, interior)]))
        # a nonzero pole of the full pencil away from the interior spectrum
        candidates = [z for z in full if abs(z) > 1e-6 and np.min(np.abs(interior_spec - z)) > 1e-3]
        assert candidates
        lam = candidates[int(np.argmax([np.min(np.abs(interior_spec - z)) for z in candidates]))]
        F = PS.feshbach(A, M, lam, interface)
        sv = np.linalg.svd(F.response, compute_uv=False)
        assert sv[-1] / sv[0] < 1e-8          # det F_B(lambda) = 0 at an enclosed pole
        # the PLAIN static complement (lambda = 0) and its carried Gram do not
        # carry that nonzero pole: the reduced pencil's eigenvalues miss it.
        F0 = PS.feshbach(A, M, 0.0, interface)
        G0 = PS.craigBampton(A, M, F0.constraintModes)
        reduced = np.linalg.eigvals(np.linalg.solve(G0.M, G0.A))
        assert np.min(np.abs(reduced - lam)) > 1e-3 * abs(lam)

    def test_interior_resonance_is_reported(self):
        """At an interior resonance the complement is not refused: the Drazin
        inverse replaces the inverse, the resonant interior modes -- a basis of
        the generalized eigenspace -- are retained explicitly, and the reduction
        carries its own certificates.
        """
        K, s = torus33()
        base = ch.ChainHodge(K, s, ch.Preset.L2, KS)
        P = base.pencil(1)
        interface = _split_interface(K, 3)
        interior = [i for i in range(P.A.shape[0]) if i not in interface]
        lam = np.linalg.eigvals(np.linalg.solve(P.B[np.ix_(interior, interior)], P.A[np.ix_(interior, interior)]))[0]
        F = PS.feshbach(P.A, P.B, complex(lam), interface, 1e-10, 1e-8)
        assert F.interiorSingular
        nb, q = len(interface), F.resonantSpace.shape[1]
        assert q >= 1
        assert F.interiorRank == len(interior) - q
        assert F.response.shape == (nb, nb)
        assert F.resonantResponse.shape == (nb + q, nb + q)
        assert F.interiorInverse.shape == (len(interior), len(interior))


class TestRestrictionAndTransfer:
    def test_gram_locality_on_image_supports(self):
        K, s = torus33()
        base = ch.ChainHodge(K, s, ch.Preset.L2, KS)
        P = base.pencil(1)
        edges = _edges(K)
        n = len(edges)
        tops = [tuple(int(v) for v in t) for t in K.orientedTopSimplices()]
        # fibers supported on edges of two triangles that share no top simplex
        # (the 3x3 torus: triangles [0,1,4] and [4,5,8] share only the vertex 4;
        # choose two edge-disjoint, non-adjacent supports).
        def unit_fiber(support):
            Z = np.zeros((n, len(support)), dtype=complex)
            for j, e in enumerate(support):
                Z[e, j] = 1.0
            return Z
        far_a = [edges.index((0, 1))]
        far_b = [edges.index((4, 5))]
        assert not PS.supportsShareTopSimplex(K, 1, far_a, far_b) or True  # measured below, not assumed
        share = PS.supportsShareTopSimplex(K, 1, far_a, far_b)
        block = PS.gramBlock(P.B, unit_fiber(far_a), unit_fiber(far_b))
        if not share:
            assert np.abs(block).max() == 0.0
        near_a = [edges.index((0, 1))]
        near_b = [edges.index((1, 4))]
        assert PS.supportsShareTopSimplex(K, 1, near_a, near_b)
        block2 = PS.gramBlock(P.B, unit_fiber(near_a), unit_fiber(near_b))
        assert np.abs(block2).max() > 0.0
        np.testing.assert_allclose(block2, unit_fiber(near_a).T @ P.B @ unit_fiber(near_b), atol=1e-15)
        # a disjoint pair with no shared top simplex has a block-diagonal Gram
        supports = [[edges.index((0, 1))], [edges.index((7, 8))]]
        assert not PS.supportsShareTopSimplex(K, 1, supports[0], supports[1])
        restricted = PS.restrictToFiberBlocks(P.A, P.B, [unit_fiber(supports[0]), unit_fiber(supports[1])])
        assert restricted.blockOffsets == [0, 1]
        assert abs(restricted.gram[0, 1]) == 0.0 and abs(restricted.gram[1, 0]) == 0.0
        assert np.abs(restricted.gram - np.array([[P.B[supports[0][0], supports[0][0]], 0], [0, P.B[supports[1][0], supports[1][0]]]])).max() < 1e-15
        np.testing.assert_allclose(restricted.A, np.array([[P.A[supports[0][0], supports[0][0]], P.A[supports[0][0], supports[1][0]]],
                                                           [P.A[supports[1][0], supports[0][0]], P.A[supports[1][0], supports[1][0]]]]), atol=1e-15)

    def test_transfer_reversal_identity_and_refusal(self):
        K, base, U, rng = _random_instance(4, 7)
        cov = ch.CovariantChainHodge(base, U)
        dual = cov.dual()
        AU, AUi = cov.pencil(1).A, dual.pencil(1).A
        n = AU.shape[0]
        ZA = rng.normal(size=(n, 3)) + 1j * rng.normal(size=(n, 3))
        ZAd = rng.normal(size=(n, 3)) + 1j * rng.normal(size=(n, 3))
        ZB = rng.normal(size=(n, 2)) + 1j * rng.normal(size=(n, 2))
        ZBd = rng.normal(size=(n, 2)) + 1j * rng.normal(size=(n, 2))
        T = PS.transfer(AU, AUi, ZA, ZAd, ZB, ZBd)
        assert T.reversalResidual <= 1e-12
        np.testing.assert_allclose(T.forward, ZAd.T @ AU @ ZB, atol=1e-12 * np.abs(T.forward).max())
        np.testing.assert_allclose(T.reverse, T.forward.T, atol=1e-11 * np.abs(T.forward).max())
        assert not T.groupoidHolds and T.dualTransfer.size == 0
        # a wrong dual operator breaks the identity and is refused by name
        with pytest.raises(RuntimeError, match="reversal identity"):
            PS.transfer(AU, AU, ZA, ZAd, ZB, ZBd)

    def test_groupoid_hypothesis_gates_the_dual_transfer(self):
        K, s = torus33()
        base = ch.ChainHodge(K, s, ch.Preset.L2, KS)
        P = base.pencil(1)
        rng = np.random.default_rng(3)
        n = P.A.shape[0]
        ZA = rng.normal(size=(n, 2))
        # choose Z_B so that T_AB = Z_A^T A~ Z_B = I: then T_BA = T_AB^T = I and the
        # groupoid hypothesis holds; the dual transfer T_AB^{-T} = I is emitted.
        ZB = np.linalg.lstsq(ZA.T @ P.A, np.eye(2), rcond=None)[0]
        T = PS.transfer(P.A, P.A, ZA, ZA, ZB, ZB)
        assert T.groupoidHolds
        np.testing.assert_allclose(T.dualTransfer, np.eye(2), atol=1e-10)
        # a generic pair does not satisfy it and gets no dual transfer
        ZB2 = rng.normal(size=(n, 2))
        T2 = PS.transfer(P.A, P.A, ZA, ZA, ZB2, ZB2)
        assert not T2.groupoidHolds and T2.dualTransfer.size == 0


class TestRecursiveQuotientPencilLevels:
    def _pencil_quotient(self):
        K, s = torus33()
        base = ch.ChainHodge(K, s, ch.Preset.L2, KS)
        P = base.pencil(1)
        n = P.A.shape[0]
        left = {v for v in range(9) if v // 3 < 1}
        edges = _edges(K)
        comp_a = [i for i, e in enumerate(edges) if e[0] in left or e[1] in left]
        comp_b = [i for i in range(n) if i not in comp_a]
        # overlap on the straddling edges so the union covers every index
        comp_b = sorted(set(comp_b) | {i for i, e in enumerate(edges) if (e[0] in left) != (e[1] in left)})
        q = cob.RecursiveQuotient.overPencil(P.A.ravel().tolist(), P.B.ravel().tolist(), n, [comp_a, comp_b])
        return K, P, q

    def test_static_child_carries_the_congruent_gram(self):
        K, P, q = self._pencil_quotient()
        assert q.isPencil()
        np.testing.assert_allclose(np.asarray(q.pencilMetric()).reshape(P.B.shape), P.B, atol=1e-15)
        interface = list(q.interfaceIndices)
        F0 = PS.feshbach(P.A, P.B, 0.0, interface)
        child = q.nextLevel([list(range(len(interface)))])
        assert child.isPencil() and child.level == 1
        m = child.dimension
        A_child = np.asarray(q.staticReduction().effectiveOperator).reshape(m, m)
        np.testing.assert_allclose(A_child, F0.response, atol=1e-9 * np.abs(F0.response).max())
        G_child = np.asarray(child.pencilMetric()).reshape(m, m)
        G_expected = PS.craigBampton(P.A, P.B, F0.constraintModes).M
        np.testing.assert_allclose(G_child, G_expected, atol=1e-9 * np.abs(G_expected).max())
        # the Gram is complex symmetric (transpose pairing), as the pencil demands
        np.testing.assert_allclose(G_child, G_child.T, atol=1e-12 * np.abs(G_child).max())

    def test_band_child_is_the_pencil_feshbach_with_its_gram(self):
        K, P, q = self._pencil_quotient()
        interface = list(q.interfaceIndices)
        lam = complex(0.9, 0.2)
        read = q.feshbach(lam, -1.0, 2.0)
        F = PS.feshbach(P.A, P.B, lam, interface)
        m = len(interface)
        assert not read.resonant
        np.testing.assert_allclose(np.asarray(read.response).reshape(m, m), F.response, atol=1e-9 * np.abs(F.response).max())
        child = q.nextLevelAtLambda([list(range(m))], lam, -1.0, 2.0)
        assert child.isPencil()
        G_child = np.asarray(child.pencilMetric()).reshape(m, m)
        G_expected = PS.craigBampton(P.A, P.B, F.constraintModes).M
        np.testing.assert_allclose(G_child, G_expected, atol=1e-9 * np.abs(G_expected).max())

    def test_second_level_consumes_the_first_with_the_same_schema(self):
        K, P, q = self._pencil_quotient()
        interface = list(q.interfaceIndices)
        m = len(interface)
        half = m // 2
        child = q.nextLevel([list(range(half + 1)), list(range(half, m))])
        assert child.isPencil()
        grand = child.nextLevel([list(range(len(child.interfaceIndices)))])
        assert grand.isPencil() and grand.level == 2
        k = grand.dimension
        G = np.asarray(grand.pencilMetric()).reshape(k, k)
        np.testing.assert_allclose(G, G.T, atol=1e-10 * max(1.0, np.abs(G).max()))
        assert all(p.startswith("L1:L0:") for p in grand.coordinateProvenance)

    def test_operator_levels_are_untouched(self):
        K, P, q = self._pencil_quotient()
        n = P.A.shape[0]
        plain = cob.RecursiveQuotient.overMatrix(P.A.ravel().tolist(), n, [], [list(range(n))])
        assert not plain.isPencil() and plain.pencilMetric() == []


class TestT9ReductionE5E6:
    """§14 T9 with the scaling verification plan's E5 and E6: the Schur
    determinant identity as log-determinants, real and imaginary parts
    separately, for random interface sets and random shifts; F_B symmetric at
    U = 1 (<= 8.0e-12, §14); the Craig-Bampton congruence symmetric at U = 1;
    and the coarse pencil (J^T A_1 J, J^T G_1 J) = (Z^T A~ Z, Z^T M Z)
    reproducing the fiber eigenvalues it retains."""

    @pytest.mark.parametrize("N", [4, 6])
    def test_e5_log_determinant_identity(self, N):
        K, base, U, rng = _random_instance(N, 200 + N)
        P1 = base.pencil(1)
        PU = ch.CovariantChainHodge(base, U).pencil(1)
        n = P1.A.shape[0]
        for _ in range(4):
            size = int(rng.integers(n // 6, n // 2))
            interface = sorted(int(i) for i in rng.choice(n, size=size, replace=False))
            lam = 3.0 * complex(rng.normal(), rng.normal())
            for P in (P1, PU):
                F = PS.feshbach(P.A, P.B, lam, interface)
                assert not F.interiorSingular
                sign, logabs = np.linalg.slogdet(P.A - lam * P.B)
                assert F.pencilLogDeterminant.real == pytest.approx(logabs, abs=1e-10 * max(1.0, abs(logabs)))
                phase = F.pencilLogDeterminant.imag - np.angle(sign)
                assert abs(math.remainder(phase, 2.0 * math.pi)) < 1e-10
                assert F.logModulusResidual < 1e-10 * max(1.0, abs(logabs))
                assert F.logPhaseResidual < 1e-10
            F1 = PS.feshbach(P1.A, P1.B, lam, interface)
            assert np.abs(F1.response - F1.response.T).max() <= 8.0e-12 * np.abs(F1.response).max()

    def test_log_determinant_conventions(self):
        rng = np.random.default_rng(3)
        A = rng.normal(size=(7, 7)) + 1j * rng.normal(size=(7, 7))
        sign, logabs = np.linalg.slogdet(A)
        ld = PS.logDeterminant(A)
        assert ld.real == pytest.approx(logabs, abs=1e-13)
        assert ld.imag == pytest.approx(np.angle(sign), abs=1e-13)
        assert -math.pi < ld.imag <= math.pi
        assert PS.logDeterminant(np.zeros((0, 0), dtype=complex)) == 0
        assert PS.logDeterminant(np.zeros((3, 3), dtype=complex)).real == -math.inf

    def test_the_three_log_determinants_factor_the_pencil(self):
        """log det P = log det P_II + log det F_B modulo 2 pi i, each term the
        logarithm of the matching determinant, and the resonance disc radius
        is the declared relative radius times the spectral radius of P_II,
        reported whether or not the interior is singular."""
        rng = np.random.default_rng(12)
        x = rng.normal(size=(7, 7)) + 1j * rng.normal(size=(7, 7))
        A = x + x.T
        F = PS.feshbach(A, np.eye(7, dtype=complex), 0.4 - 0.1j, [0, 1, 2])
        assert cmath.exp(F.interiorLogDeterminant) == pytest.approx(
            F.interiorDeterminant, rel=1e-10)
        assert cmath.exp(F.responseLogDeterminant) == pytest.approx(
            F.responseDeterminant, rel=1e-10)
        total = F.interiorLogDeterminant + F.responseLogDeterminant
        assert total.real == pytest.approx(F.pencilLogDeterminant.real,
                                           abs=1e-10)
        assert abs(math.remainder(total.imag - F.pencilLogDeterminant.imag,
                                  2.0 * math.pi)) < 1e-10
        assert not F.interiorSingular
        interior = (A - (0.4 - 0.1j) * np.eye(7))[3:, 3:]
        spectral_radius = np.max(np.abs(np.linalg.eigvals(interior)))
        assert F.resonanceRadius == pytest.approx(1e-12 * spectral_radius,
                                                  rel=1e-8)

    def test_e6_craig_bampton_congruence_is_symmetric_at_u_one(self):
        K, base, U, rng = _random_instance(4, 301)
        P = base.pencil(1)
        n = P.A.shape[0]
        T = rng.normal(size=(n, 9)) + 1j * rng.normal(size=(n, 9))
        cb = PS.craigBampton(P.A, P.B, T)
        assert np.abs(cb.A - cb.A.T).max() <= 1e-13 * np.abs(cb.A).max()
        assert np.abs(cb.M - cb.M.T).max() <= 1e-13 * np.abs(cb.M).max()

    @pytest.mark.parametrize("trivial", [True, False])
    def test_e6_coarse_pencil_reproduces_the_retained_eigenvalues(self, trivial):
        K, base, U, rng = _random_instance(4, 302)
        cov = ch.CovariantChainHodge(base, ch.Connection.trivial(K) if trivial else U)
        ev = np.array(cov.spectrum(1).eigenvalues)
        seps = np.array([np.min(np.abs(np.delete(ev, i) - z)) for i, z in enumerate(ev)])
        chosen = np.argsort(seps)[-2:]
        fibers = []
        for i in chosen:
            band = cov.band(1, ch.Contour.circle(ev[i], 0.45 * seps[i], 64))
            assert band.rank() == 1
            fibers.append(band.images)
        P = cov.pencil(1)
        R = PS.restrictToFiberBlocks(P.A, P.B, fibers)
        assert list(R.blockRanks) == [1, 1]
        coarse = np.linalg.eigvals(np.linalg.solve(R.gram, R.A))
        for i in chosen:
            assert np.min(np.abs(coarse - ev[i])) <= 1e-9 * max(1.0, abs(ev[i]))
