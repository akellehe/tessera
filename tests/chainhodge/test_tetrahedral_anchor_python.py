# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The tetrahedral anchor at degree 0 (#939): the face anchor of specification
§8 one degree down and one dimension up. M_0^{(T)} = vol(T)(I + 11^T)/20 on a
single tetrahedron (rank 4); the sum of the containing top simplices' blocks
at d >= 4; Pi_T(U) by solves agrees with the image pairing; alpha_T invariant
under the paired frame gauge and under rho_0(g), generically nonzero for a
rank-four fiber; the Grassmann preset is refused by name; the flat-zero-mode
certificate is 1 on the trivial-holonomy harmonic band and ~0 on a band away
from it. Transpose pairing throughout; no dagger."""
import itertools

import numpy as np
import pytest

from tessera import chainhodge as ch
from tessera import cobordism as cob

FA = ch.FaceAnchor

TETRAHEDRON = [[0, 1, 2, 3]]
THREE_COMPLEX = [[0, 1, 2, 3], [1, 2, 3, 4], [0, 1, 2, 4]]
FOUR_COMPLEX = [[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]]


def _complex(cells):
    return cob.ChainComplex.from_top_cells(cells)


def _complex_lengths(K, rng, scale=0.3):
    return [complex(1.0 + scale * rng.normal(), scale * rng.normal()) for _ in range(K.num_simplices(1))]


def _random_links(K, rng):
    return [complex(rng.normal(), rng.normal()) for _ in range(K.num_simplices(1))]


def _random_gauge(K, rng):
    return {int(v[0]): complex(rng.normal(), rng.normal()) + 0.3 for v in K.k_simplex_vertices(0)}


def _covariant(K, s, links=None, preset=ch.Preset.L2):
    hodge = ch.ChainHodge(K, s, preset)
    if links is None:
        links = [1.0 + 0j] * K.num_simplices(1)  # trivial holonomy
    U = ch.Connection(K, links)
    return ch.CovariantChainHodge(hodge, U)


def _band(cov, rank, rng, degree=0):
    """A rank-`rank` degree-0 fiber on geometric images from the dressed pencil's
    eigenvectors for U and the matching ones for U^{-1} (transposed pencils share
    a spectrum); columns paired by eigenvalue, the largest |lambda| first."""
    spec = cov.spectrum(degree)
    dual = cov.dual().spectrum(degree)
    ev = np.asarray(spec.eigenvalues)
    evd = np.asarray(dual.eigenvalues)
    chosen = np.argsort(-np.abs(ev))[:rank]
    Z = np.asarray(spec.vectors)[:, chosen]
    cols = []
    for i in chosen:
        j = int(np.argmin(np.abs(evd - ev[i])))
        assert abs(evd[j] - ev[i]) < 1e-8 * max(1.0, abs(ev[i]))
        cols.append(j)
    return np.asarray(dual.vectors)[:, cols], Z


class TestTetrahedronBlock:
    def test_single_tetrahedron_block_is_volume_times_identity_plus_ones_over_twenty(self):
        K = _complex(TETRAHEDRON)
        rng = np.random.default_rng(0)
        s = _complex_lengths(K, rng)
        tb = FA.whitney_tetrahedron_block(K, s, 0)
        assert tb.rank == 4 and list(tb.vertex_indices) == [0, 1, 2, 3]
        block = np.asarray(tb.block)
        volume = block[0, 1] * 20.0  # the Gram of the four barycentric functions: vol (I + 11^T)/20
        np.testing.assert_allclose(block, volume * (np.eye(4) + np.ones((4, 4))) / 20.0, rtol=1e-12, atol=1e-14)
        assert abs(volume) > 0.0

    def test_three_complex_blocks_have_rank_four_and_are_the_top_cells_own_blocks(self):
        K = _complex(THREE_COMPLEX)
        rng = np.random.default_rng(1)
        s = _complex_lengths(K, rng)
        blocks = FA.whitney_tetrahedron_blocks(K, s)
        assert len(blocks) == 3 and all(b.rank == 4 for b in blocks)
        # at d = 3 the block is the top simplex's own local degree-0 block
        for tb, top in zip(blocks, ch.WhitneyMass.top_simplex_blocks(K, s, 0)):
            assert list(tb.vertex_indices) == list(top.cell_indices)
            np.testing.assert_allclose(np.asarray(tb.block), np.asarray(top.block), rtol=1e-12, atol=1e-14)

    def test_four_complex_block_sums_the_containing_top_simplices(self):
        K = _complex(FOUR_COMPLEX)
        rng = np.random.default_rng(2)
        s = _complex_lengths(K, rng)
        tets = [tuple(int(v) for v in t) for t in K.k_simplex_vertices(3)]
        shared = tets.index((1, 2, 3, 4))  # the 3-face both 4-simplices contain
        tb = FA.whitney_tetrahedron_block(K, s, shared)
        assert tb.rank == 4
        expected = np.zeros((4, 4), dtype=complex)
        for top in ch.WhitneyMass.top_simplex_blocks(K, s, 0):
            cells = list(top.cell_indices)
            if all(v in cells for v in tb.vertex_indices):
                local = [cells.index(v) for v in tb.vertex_indices]
                expected += np.asarray(top.block)[np.ix_(local, local)]
        np.testing.assert_allclose(np.asarray(tb.block), expected, rtol=1e-12, atol=1e-14)

    def test_grassmann_preset_is_refused_by_name(self):
        K = _complex(THREE_COMPLEX)
        s = _complex_lengths(K, np.random.default_rng(3))
        hodge = ch.ChainHodge(K, s, ch.Preset.GRASSMANN_ALL)
        with pytest.raises(ValueError, match="Grassmann degree-0 block pairs scalars"):
            FA.tetrahedron_block(hodge, 0)
        assert FA.tetrahedron_block(ch.ChainHodge(K, s), 1).rank == 4


class TestAnchorCoordinate:
    def test_dressing_and_endomorphism_agree_with_the_image_pairing(self):
        K = _complex(THREE_COMPLEX)
        rng = np.random.default_rng(4)
        s = _complex_lengths(K, rng)
        U = ch.Connection(K, _random_links(K, rng))
        cov = ch.CovariantChainHodge(ch.ChainHodge(K, s), U)
        vertices = [int(v[0]) for v in K.k_simplex_vertices(0)]
        for t in range(K.num_simplices(3)):
            tb = FA.tetrahedron_block(cov.base(), t)
            D = np.asarray(FA.dressed_tetrahedron_block(tb, K, U))
            B = np.asarray(tb.block)
            for p, vp in enumerate(tb.vertex_indices):
                for q, vq in enumerate(tb.vertex_indices):
                    assert D[p, q] == pytest.approx(B[p, q] * U.link(vertices[vp], vertices[vq]), rel=1e-14, abs=1e-15)
        # (Phi^vee)^T Pi_T(U) Phi == (Z^vee)^T M_0^{(T)U} Z with Z = G_0^U Phi, Z^vee = G_0^{U^-1} Phi^vee
        n0 = K.num_simplices(0)
        Phi = rng.normal(size=(n0, 4)) + 1j * rng.normal(size=(n0, 4))
        PhiD = rng.normal(size=(n0, 4)) + 1j * rng.normal(size=(n0, 4))
        Z = np.asarray(cov.apply_g(0, Phi))
        Zd = np.asarray(cov.dual().apply_g(0, PhiD))
        for t in range(K.num_simplices(3)):
            left = np.linalg.det(PhiD.T @ np.asarray(FA.apply_tetrahedron_endomorphism(cov, t, Phi)))
            right = FA.tetrahedron_anchor_coordinate(cov, t, Zd, Z)
            assert left == pytest.approx(right, rel=1e-9)
            assert FA.tetrahedron_anchor_coordinate_from_chains(cov, t, PhiD, Phi) == pytest.approx(right, rel=1e-9)

    def test_invariant_under_paired_frame_gauge_and_rank_four_generic(self):
        K = _complex(THREE_COMPLEX)
        rng = np.random.default_rng(5)
        cov = _covariant(K, _complex_lengths(K, rng), _random_links(K, rng))
        Zd, Z = _band(cov, 4, rng)
        alphas = np.asarray(FA.tetrahedron_anchor_coordinates(cov, Zd, Z))
        assert alphas.shape == (3,) and np.all(np.abs(alphas) > 1e-12)
        g = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
        gauged = np.asarray(FA.tetrahedron_anchor_coordinates(cov, Zd @ np.linalg.inv(g).T, Z @ g))
        np.testing.assert_allclose(gauged, alphas, rtol=1e-9)

    def test_invariant_under_vertex_gauge(self):
        K = _complex(THREE_COMPLEX)
        rng = np.random.default_rng(6)
        cov = _covariant(K, _complex_lengths(K, rng), _random_links(K, rng))
        Zd, Z = _band(cov, 4, rng)
        alphas = np.asarray(FA.tetrahedron_anchor_coordinates(cov, Zd, Z))
        g = _random_gauge(K, rng)
        rho = np.asarray(cov.rho(0, g)).ravel()
        gauged = cov.gauged(g)
        # under U -> U^g the images move as Z -> rho Z and Z^vee -> rho^{-1} Z^vee
        again = np.asarray(FA.tetrahedron_anchor_coordinates(gauged, (1.0 / rho)[:, None] * Zd, rho[:, None] * Z))
        np.testing.assert_allclose(again, alphas, rtol=1e-9)

    def test_errors(self):
        K = _complex(THREE_COMPLEX)
        rng = np.random.default_rng(7)
        s = _complex_lengths(K, rng)
        with pytest.raises(ValueError, match="out of range"):
            FA.whitney_tetrahedron_block(K, s, 3)
        K2 = _complex([[0, 1, 2], [1, 2, 3]])
        with pytest.raises(ValueError, match="no tetrahedra"):
            FA.whitney_tetrahedron_block(K2, _complex_lengths(K2, rng), 0)
        cov = _covariant(K, s)
        with pytest.raises(ValueError, match="same shape"):
            FA.tetrahedron_anchor_coordinate(cov, 0, np.ones((K.num_simplices(0), 2)), np.ones((K.num_simplices(0), 3)))


class TestFlatZeroMode:
    def test_harmonic_band_at_trivial_holonomy_is_the_flat_zero_mode(self):
        K = _complex(THREE_COMPLEX)
        rng = np.random.default_rng(8)
        s = _complex_lengths(K, rng)
        hodge = ch.ChainHodge(K, s)
        images = np.asarray(hodge.harmonic_chains(0).images)
        assert images.shape[1] == 1  # b_0 = 1
        # the image of the vertex-volume chain is the constant vector
        np.testing.assert_allclose(images[:, 0] / images[0, 0], np.ones(K.num_simplices(0)), atol=1e-12)
        cov = _covariant(K, s)
        assert FA.flat_zero_mode_overlap(cov, images, images) == pytest.approx(1.0, abs=1e-10)

    def test_band_of_other_eigenvalues_contains_none_of_it(self):
        K = _complex(THREE_COMPLEX)
        rng = np.random.default_rng(9)
        cov = _covariant(K, _complex_lengths(K, rng))
        Zd, Z = _band(cov, 4, rng)  # the four largest |lambda| of the degree-0 pencil at U = 1
        assert FA.flat_zero_mode_overlap(cov, Zd, Z) < 1e-10
        assert FA.flat_zero_mode_overlap(cov, np.zeros((K.num_simplices(0), 0)), np.zeros((K.num_simplices(0), 0))) == 0.0
        # mixing the zero mode into the band raises the overlap toward 1
        z0 = np.asarray(ch.ChainHodge(K, cov.base().squared_lengths()).harmonic_chains(0).images)
        assert FA.flat_zero_mode_overlap(cov, np.hstack([Zd, z0]), np.hstack([Z, z0])) == pytest.approx(1.0, abs=1e-8)
