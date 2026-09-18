# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Face anchors from the Whitney triangle block (#915, specification §8, §14 T10):
rank 3 of M_1^{(t)} on every triangle of the Lorentzian torus and rank 2 under the
Grassmann preset; alpha_tau invariant under the fiber's frame gauge and under
rho_1(g); identically zero under the Grassmann preset; generically nonzero for a
rank-three fiber; Pi_tau(U) by solves agrees with the image pairing."""
import numpy as np
import pytest

from tessera import chainhodge as ch
from tessera import cobordism as cob
from tests.chainhodge._fixtures import edges, torus33

FA = ch.FaceAnchor
KS = ch.Branch.KontsevichSegal

TWO_COMPLEX = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3], [2, 3, 4]]
THREE_COMPLEX = [[0, 1, 2, 3], [1, 2, 3, 4], [0, 1, 2, 4]]


def _random_links(K, rng):
    return [complex(rng.normal(), rng.normal()) for _ in range(K.numSimplices(1))]


def _random_gauge(K, rng):
    return {int(v[0]): complex(rng.normal(), rng.normal()) + 0.3 for v in K.kSimplexVertices(0)}


def _complex_lengths(K, rng, scale=0.3):
    return [complex(1.0 + scale * rng.normal(), scale * rng.normal()) for _ in range(K.numSimplices(1))]


def _band(cov, rank, rng):
    """A rank-`rank` fiber on geometric images: eigenvectors of the dressed pencil
    for U, and the matching eigenvectors for U^{-1} (the pencils are transposes
    of each other, so their spectra coincide); returns (Z_dual, Z) with
    columns paired by eigenvalue."""
    spec = cov.spectrum(1)
    dual = cov.dual().spectrum(1)
    ev = np.asarray(spec.eigenvalues)
    evd = np.asarray(dual.eigenvalues)
    order = np.argsort(-np.abs(ev))
    chosen = order[:rank]
    Z = np.asarray(spec.vectors)[:, chosen]
    cols = []
    for i in chosen:
        j = int(np.argmin(np.abs(evd - ev[i])))
        assert abs(evd[j] - ev[i]) < 1e-8 * max(1.0, abs(ev[i]))
        cols.append(j)
    Zd = np.asarray(dual.vectors)[:, cols]
    return Zd, Z


class TestT10FaceRank:
    def test_whitney_blocks_have_rank_three_on_the_lorentzian_torus(self):
        K, s = torus33()
        blocks = FA.whitneyFaceBlocks(K, s, KS)
        assert len(blocks) == 18
        for b in blocks:
            assert b.rank == 3
            assert np.linalg.matrix_rank(np.asarray(b.block), tol=1e-12) == 3
            assert b.preset == ch.Preset.L2
            # the block is the triangle's own local block at d = 2
            top = [t for t in ch.WhitneyMass.topSimplexBlocks(K, s, 1, KS) if t.topIndex == b.faceIndex][0]
            np.testing.assert_allclose(np.asarray(b.block), np.asarray(top.block), atol=1e-15)
            assert list(b.edgeIndices) == list(top.cellIndices)

    def test_grassmann_blocks_have_rank_two(self):
        K, s = torus33()
        for t in range(18):
            g = FA.grassmannFaceBlock(K, s, t)
            assert g.rank == 2 and g.preset == ch.Preset.GRASSMANN_ALL
            assert np.linalg.matrix_rank(np.asarray(g.block), tol=1e-12) == 2
            # the blade block is the polarization pairing of the three edge vectors,
            # symmetric, and its diagonal is the squared lengths
            B = np.asarray(g.block)
            np.testing.assert_allclose(B, B.T, atol=1e-15)
            np.testing.assert_allclose(np.diag(B), [s[i] for i in g.edgeIndices], atol=1e-15)

    def test_three_complex_face_block_sums_the_containing_top_simplices(self):
        rng = np.random.default_rng(3)
        K = cob.ChainComplex.fromTopCells(THREE_COMPLEX)
        s = _complex_lengths(K, rng, 0.1)
        tops = ch.WhitneyMass.topSimplexBlocks(K, s, 1)
        for t in range(K.numSimplices(2)):
            fb = FA.whitneyFaceBlock(K, s, t)
            expected = np.zeros((3, 3), dtype=complex)
            for tb in tops:
                cells = list(tb.cellIndices)
                if all(e in cells for e in fb.edgeIndices):
                    idx = [cells.index(e) for e in fb.edgeIndices]
                    expected += np.asarray(tb.block)[np.ix_(idx, idx)]
            np.testing.assert_allclose(np.asarray(fb.block), expected, atol=1e-14)
            assert fb.rank == 3


class TestAnchorCoordinate:
    def test_dressing_and_endomorphism_agree_with_the_image_pairing(self):
        rng = np.random.default_rng(11)
        K = cob.ChainComplex.fromTopCells(TWO_COMPLEX)
        s = _complex_lengths(K, rng)
        U = ch.Connection(K, _random_links(K, rng))
        cov = ch.CovariantChainHodge(ch.ChainHodge(K, s, ch.Preset.L2, KS), U)
        es = edges(K)
        for t in range(K.numSimplices(2)):
            fb = FA.faceBlock(cov.base(), t)
            D = np.asarray(FA.dressedFaceBlock(fb, K, U))
            B = np.asarray(fb.block)
            for p, ep in enumerate(fb.edgeIndices):
                for q, eq in enumerate(fb.edgeIndices):
                    assert D[p, q] == pytest.approx(B[p, q] * U.link(es[ep][0], es[eq][0]), rel=1e-14, abs=1e-15)
        # (Phi^vee)^T Pi_tau(U) Phi == (Z^vee)^T M^{(tau)U} Z with Z = G^U Phi, Z^vee = G^{U^-1} Phi^vee
        n = K.numSimplices(1)
        Phi = rng.normal(size=(n, 3)) + 1j * rng.normal(size=(n, 3))
        PhiD = rng.normal(size=(n, 3)) + 1j * rng.normal(size=(n, 3))
        Z = np.asarray(cov.applyG(1, Phi))
        Zd = np.asarray(cov.dual().applyG(1, PhiD))
        for t in range(K.numSimplices(2)):
            left = np.linalg.det(PhiD.T @ np.asarray(FA.applyFaceEndomorphism(cov, t, Phi)))
            right = FA.anchorCoordinate(cov, t, Zd, Z)
            assert left == pytest.approx(right, rel=1e-9)
            assert FA.anchorCoordinateFromChains(cov, t, PhiD, Phi) == pytest.approx(right, rel=1e-9)

    def test_invariant_under_paired_frame_gauge_and_rank_three_generic(self):
        rng = np.random.default_rng(13)
        K = cob.ChainComplex.fromTopCells(TWO_COMPLEX)
        s = _complex_lengths(K, rng)
        U = ch.Connection(K, _random_links(K, rng))
        cov = ch.CovariantChainHodge(ch.ChainHodge(K, s, ch.Preset.L2, KS), U)
        Zd, Z = _band(cov, 3, rng)
        alpha = np.asarray(FA.anchorCoordinates(cov, Zd, Z))
        assert alpha.shape == (K.numSimplices(2),)
        assert np.all(np.abs(alpha) > 1e-12 * np.abs(alpha).max())  # generically nonzero
        g = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
        g_inv_t = np.linalg.inv(g).T
        again = np.asarray(FA.anchorCoordinates(cov, Zd @ g_inv_t, Z @ g))
        # atol floored at the vector's own scale: an entry that is small through
        # cancellation carries the absolute error of the whole computation, not
        # the error its own magnitude suggests, so a bare rtol asks it for
        # accuracy the arithmetic never had.
        np.testing.assert_allclose(again, alpha, rtol=1e-9,
                                   atol=1e-9 * np.abs(alpha).max())

    def test_invariant_under_orthogonal_frame_gauge_at_trivial_connection(self):
        rng = np.random.default_rng(17)
        K = cob.ChainComplex.fromTopCells(TWO_COMPLEX)
        s = _complex_lengths(K, rng)
        cov = ch.CovariantChainHodge(ch.ChainHodge(K, s, ch.Preset.L2, KS), ch.Connection.trivial(K))
        Zd, Z = _band(cov, 3, rng)
        # at U = 1 the dual band is the band itself: one frame, O(3, C) gauge
        alpha = np.asarray(FA.anchorCoordinates(cov, Z, Z))
        A = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
        A = A - A.T  # complex antisymmetric generator
        from scipy.linalg import expm
        O = expm(A)  # O^T O = I, det O = 1
        np.testing.assert_allclose(O.T @ O, np.eye(3), atol=1e-12)
        again = np.asarray(FA.anchorCoordinates(cov, Z @ O, Z @ O))
        np.testing.assert_allclose(again, alpha, rtol=1e-9,
                                   atol=1e-9 * np.abs(alpha).max())

    def test_invariant_under_vertex_gauge(self):
        rng = np.random.default_rng(19)
        K = cob.ChainComplex.fromTopCells(TWO_COMPLEX)
        s = _complex_lengths(K, rng)
        U = ch.Connection(K, _random_links(K, rng))
        base = ch.ChainHodge(K, s, ch.Preset.L2, KS)
        cov = ch.CovariantChainHodge(base, U)
        Zd, Z = _band(cov, 3, rng)
        alpha = np.asarray(FA.anchorCoordinates(cov, Zd, Z))
        g = _random_gauge(K, rng)
        rho = np.asarray(cov.rho(1, g)).ravel()
        gauged = cov.gauged(g)
        # under U -> U^g the images move as Z -> rho Z and Z^vee -> rho^{-1} Z^vee
        again = np.asarray(FA.anchorCoordinates(gauged, (1.0 / rho)[:, None] * Zd, rho[:, None] * Z))
        np.testing.assert_allclose(again, alpha, rtol=1e-9)

    def test_vanishes_identically_under_the_grassmann_preset(self):
        rng = np.random.default_rng(23)
        K = cob.ChainComplex.fromTopCells(TWO_COMPLEX)
        s = _complex_lengths(K, rng)
        U = ch.Connection(K, _random_links(K, rng))
        cov = ch.CovariantChainHodge(ch.ChainHodge(K, s, ch.Preset.GRASSMANN_ALL), U)
        n = K.numSimplices(1)
        for _ in range(3):
            Z = rng.normal(size=(n, 3)) + 1j * rng.normal(size=(n, 3))
            Zd = rng.normal(size=(n, 3)) + 1j * rng.normal(size=(n, 3))
            alpha = np.asarray(FA.anchorCoordinates(cov, Zd, Z))
            scale = max(np.abs(Z).max() * np.abs(Zd).max(), 1.0) ** 3
            assert np.max(np.abs(alpha)) < 1e-12 * scale
        for t in range(K.numSimplices(2)):
            assert FA.faceBlock(cov.base(), t).rank == 2

    def test_errors(self):
        K, s = torus33()
        with pytest.raises(ValueError):
            FA.whitneyFaceBlock(K, s, 99)
        with pytest.raises(ValueError):
            FA.grassmannFaceBlock(K, s[:-1], 0)
        one = cob.ChainComplex.fromTopCells([[0, 1], [1, 2], [0, 2]])
        with pytest.raises(ValueError):
            FA.whitneyFaceBlock(one, [1.0, 1.0, 1.0], 0)
