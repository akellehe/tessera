# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Whitney mass matrices (#906): the specification's §14 values (T5a, T5b, T6),
symmetry and sparsity, allowability, the scaling identity of the derivative,
branch agreement on the allowable domain, and relabeling covariance."""
import cmath
import itertools
import math

import numpy as np
import pytest

from tessera import chainhodge as ch
from tessera import cobordism as cob

WM = ch.WhitneyMass


from tests.chainhodge._fixtures import edges as _edges, lengths as _lengths, torus33 as _torus33


def _random_allowable(K, rng, scale=0.05):
    """Unit Euclidean lengths with a small complex perturbation (allowable)."""
    n = K.numSimplices(1)
    return [complex(1.0 + scale * rng.normal(), scale * rng.normal()) for _ in range(n)]


# The dense oracles moved to `_svp` when they were generalized to every
# dimension for the scaling verification plan's F7 families (#1208): the
# Whitney reference is now the specification's closed form at any d (its d >= 3
# values are tested in test_svp_three_dimensional_python.py), and both are
# measured against the library at the plan's tolerance in
# test_svp_oracle_python.py.
from tests.chainhodge._svp import (blade_dot as _blade_dot,
                                   grassmann_reference as _grassmann_reference,
                                   whitney_reference as _whitney_reference)


FIXTURES = {
    "2-complex": [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3], [2, 3, 4]],
    "3-complex": [[0, 1, 2, 3], [1, 2, 3, 4]],
    "4-complex": [[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]],
}


class TestT5aEuclideanThreeCycle:
    K = cob.ChainComplex.fromTopCells([[0, 1], [0, 2], [1, 2]])
    s = [1.0, 4.0, 9.0]  # l = (1, 2, 3) on edges (01), (02), (12)

    def test_m1_is_inverse_lengths(self):
        M1 = WM.assemble(self.K, self.s, 1).toarray()
        np.testing.assert_allclose(M1, np.diag([1.0, 0.5, 1.0 / 3.0]), atol=1e-15)

    def test_m0_exact(self):
        M0 = WM.assemble(self.K, self.s, 0).toarray()
        expected = np.array([[1.0, 1 / 6, 1 / 3], [1 / 6, 4 / 3, 1 / 2], [1 / 3, 1 / 2, 5 / 3]])
        np.testing.assert_allclose(M0, expected, atol=1e-15)

    def test_certificate_euclidean(self):
        cert = WM.certificate(self.K, self.s)
        assert cert.allowable
        assert cert.margin == pytest.approx(math.pi)
        np.testing.assert_allclose(cert.volumes, [1.0, 2.0, 3.0], atol=1e-15)
        assert not cert.continuationAmbiguous


class TestT5bMixedSignatureThreeCycle:
    K = cob.ChainComplex.fromTopCells([[0, 1], [0, 2], [1, 2]])
    s = [1.0, 1.0, -1.0]

    def test_m1_principal_branch(self):
        M1 = WM.assemble(self.K, self.s, 1, ch.Branch.KontsevichSegal).toarray()
        np.testing.assert_allclose(M1, np.diag([1.0, 1.0, -1j]), atol=1e-15)

    def test_continuation_falls_back_and_reports(self):
        M1 = WM.assemble(self.K, self.s, 1, ch.Branch.Continuation).toarray()
        np.testing.assert_allclose(M1, np.diag([1.0, 1.0, -1j]), atol=1e-15)
        cert = WM.certificate(self.K, self.s, ch.Branch.Continuation)
        assert cert.continuationAmbiguous
        assert list(cert.ambiguousTopSimplices) == [2]
        assert not cert.allowable  # the timelike edge sits on the cut (margin 0)
        assert cert.margin == pytest.approx(0.0, abs=1e-15)


class TestT6LorentzianTorus:
    def test_m2_is_minus_two_root_two_i(self):
        K, s = _torus33()
        M2 = WM.assemble(K, s, 2, ch.Branch.KontsevichSegal).toarray()
        np.testing.assert_allclose(M2, -2 * math.sqrt(2) * 1j * np.eye(18), atol=1e-14)
        cert = WM.certificate(K, s, ch.Branch.KontsevichSegal)
        np.testing.assert_allclose(cert.gramDeterminants, [-0.5] * 18, atol=1e-15)
        assert cert.margin == pytest.approx(0.0, abs=1e-14)
        assert not cert.allowable

    def test_continuation_agrees_with_kontsevich_segal_up_to_reporting(self):
        K, s = _torus33()
        a = WM.assemble(K, s, 1, ch.Branch.Continuation).toarray()
        b = WM.assemble(K, s, 1, ch.Branch.KontsevichSegal).toarray()
        np.testing.assert_allclose(a, b, atol=1e-14)
        assert WM.certificate(K, s, ch.Branch.Continuation).continuationAmbiguous

    def test_global_factor_i(self):
        """Real Lorentzian data: every M_k is i times a real matrix (§4.2)."""
        K, s = _torus33()
        for k in range(3):
            M = WM.assemble(K, s, k, ch.Branch.KontsevichSegal).toarray()
            assert np.max(np.abs(M.real)) < 1e-14
            assert np.max(np.abs(M.imag)) > 0

    def test_rotated_family_is_allowable(self):
        """Rotating the timelike part by e^{-2i eps} lands on the allowable side."""
        K, s = _torus33()
        eps = 0.1
        s_eps = [v * cmath.exp(-2j * eps) if v.real < 0 else v for v in s]
        cert = WM.certificate(K, s_eps, ch.Branch.Continuation)
        assert cert.allowable and cert.margin > 0
        assert not cert.continuationAmbiguous


class TestStructure:
    @pytest.mark.parametrize("name", sorted(FIXTURES))
    def test_symmetric_and_sparsity_rule(self, name):
        rng = np.random.default_rng(3)
        K = cob.ChainComplex.fromTopCells(FIXTURES[name])
        s = _random_allowable(K, rng)
        tops = [tuple(t) for t in K.orientedTopSimplices()]
        for k in range(K.dimension() + 1):
            M = WM.assemble(K, s, k).toarray()
            np.testing.assert_allclose(M, M.T, atol=1e-14)
            cells = [tuple(c) for c in K.kSimplexVertices(k)]
            for i, j in zip(*np.nonzero(np.abs(M) > 1e-15)):
                union = set(cells[i]) | set(cells[j])
                assert any(union <= set(t) for t in tops)

    @pytest.mark.parametrize("name", sorted(FIXTURES))
    def test_scaling_identity(self, name):
        """sum_e s_e dM_k/ds_e = (d/2 - k) M_k at every degree."""
        rng = np.random.default_rng(5)
        K = cob.ChainComplex.fromTopCells(FIXTURES[name])
        s = _random_allowable(K, rng)
        d = K.dimension()
        for k in range(d + 1):
            M = WM.assemble(K, s, k).toarray()
            total = np.zeros_like(M)
            for e, se in enumerate(s):
                total += se * WM.assembleDerivative(K, s, k, e).toarray()
            np.testing.assert_allclose(total, (d / 2 - k) * M, atol=1e-12 * max(1.0, np.abs(M).max()))

    @pytest.mark.parametrize("name", sorted(FIXTURES))
    def test_contraction_matches_full_derivative(self, name):
        rng = np.random.default_rng(7)
        K = cob.ChainComplex.fromTopCells(FIXTURES[name])
        s = _random_allowable(K, rng)
        for k in range(K.dimension() + 1):
            n = K.numSimplices(k)
            X = rng.normal(size=(n, 2)) + 1j * rng.normal(size=(n, 2))
            Y = rng.normal(size=(n, 2)) + 1j * rng.normal(size=(n, 2))
            c = WM.derivativeContraction(K, s, k, X, Y)
            for e in range(K.numSimplices(1)):
                D = WM.assembleDerivative(K, s, k, e).toarray()
                assert c[e] == pytest.approx(np.trace(X.T @ D @ Y), abs=1e-12)

    @pytest.mark.parametrize("name", sorted(FIXTURES))
    def test_branches_agree_on_allowable_domain(self, name):
        rng = np.random.default_rng(11)
        K = cob.ChainComplex.fromTopCells(FIXTURES[name])
        s = _random_allowable(K, rng, scale=0.2)
        cert = WM.certificate(K, s)
        assert cert.allowable and not cert.continuationAmbiguous
        for k in range(K.dimension() + 1):
            a = WM.assemble(K, s, k, ch.Branch.Continuation).toarray()
            b = WM.assemble(K, s, k, ch.Branch.KontsevichSegal).toarray()
            np.testing.assert_allclose(a, b, atol=1e-13)

    def test_top_degree_is_inverse_volume(self):
        rng = np.random.default_rng(13)
        for name, cells in FIXTURES.items():
            K = cob.ChainComplex.fromTopCells(cells)
            s = _random_allowable(K, rng)
            d = K.dimension()
            Md = WM.assemble(K, s, d).toarray()
            cert = WM.certificate(K, s)
            np.testing.assert_allclose(np.diag(Md), 1.0 / np.array(cert.volumes), rtol=1e-12)
            assert np.max(np.abs(Md - np.diag(np.diag(Md)))) < 1e-15

    def test_local_blocks_sum_to_assembly(self):
        rng = np.random.default_rng(17)
        K = cob.ChainComplex.fromTopCells(FIXTURES["3-complex"])
        s = _random_allowable(K, rng)
        for k in range(4):
            M = WM.assemble(K, s, k).toarray()
            total = np.zeros_like(M)
            for b in WM.topSimplexBlocks(K, s, k):
                idx = list(b.cellIndices)
                total[np.ix_(idx, idx)] += b.block
            np.testing.assert_allclose(total, M, atol=1e-14)


class TestAllowability:
    def test_non_allowable_complex_conformal_on_lorentzian_base(self):
        K, s = _torus33()
        s_bad = [v * (0.3 + 0.2j) for v in s]
        assert WM.allowabilityMargin(K, s_bad) <= 0.0
        assert not WM.certificate(K, s_bad).allowable

    def test_margin_of_unit_gram(self):
        g = 0.5 * (np.ones((3, 3)) + np.eye(3)).astype(complex)
        assert WM.marginOf(g) == pytest.approx(math.pi)
        vol, ambiguous = WM.volumeOnBranch(g)
        assert vol == pytest.approx(math.sqrt(0.5) / 6.0)  # unit regular tetrahedron: sqrt(2)/12
        assert not ambiguous

    def test_grassmann_preset_matches_dense_reference(self):
        """assembleGrassmann equals multiplicity o blade pairing (the CH §6 oracle),
        is real on real data, and each face's own blade block has rank two."""
        K, s = _torus33()
        for k in range(3):
            G = WM.assembleGrassmann(K, s, k).toarray()
            np.testing.assert_allclose(G, _grassmann_reference(K, s, k), atol=1e-14)
            assert np.max(np.abs(G.imag)) == 0.0
            np.testing.assert_allclose(G, G.T, atol=1e-15)
        G1p = WM.assemblePreset(K, s, 1, ch.Preset.GRASSMANN_ALL).toarray()
        np.testing.assert_allclose(G1p, WM.assembleGrassmann(K, s, 1).toarray(), atol=1e-15)
        # The per-face blade block (three edge vectors of one triangle) has rank two (§4.4).
        table = dict(zip(_edges(K), s))
        for t in K.orientedTopSimplices():
            es = list(itertools.combinations(sorted(int(v) for v in t), 2))
            blk = np.array([[_blade_dot(table, a, b) for b in es] for a in es])
            assert np.linalg.matrix_rank(blk, tol=1e-12) == 2

    def test_whitney_matches_dense_reference_d2(self):
        """The sparse assembly equals the dense Whitney mass matrix of the
        specification's oracle at d = 2, on the Lorentzian torus and on random
        complex allowable data."""
        rng = np.random.default_rng(23)
        K, s = _torus33()
        for k in range(3):
            M = WM.assemble(K, s, k, ch.Branch.KontsevichSegal).toarray()
            np.testing.assert_allclose(M, _whitney_reference(K, s, k), atol=1e-13)
        K2 = cob.ChainComplex.fromTopCells(FIXTURES["2-complex"])
        s2 = _random_allowable(K2, rng, scale=0.3)
        for k in range(3):
            M = WM.assemble(K2, s2, k).toarray()
            np.testing.assert_allclose(M, _whitney_reference(K2, s2, k), atol=1e-13)

    def test_errors(self):
        K = cob.ChainComplex.fromTopCells([[0, 1, 2]])
        with pytest.raises(ValueError):
            WM.assemble(K, [1.0, 1.0], 1)
        with pytest.raises(ValueError):
            WM.assemble(K, [1.0, 1.0, 1.0], 3)
        with pytest.raises(ValueError):
            cob.ChainComplex.fromTopCells([[0, 1, 2], [0, 1]])


class TestRelabeling:
    def test_signed_permutation_similarity(self):
        rng = np.random.default_rng(19)
        K, s = _torus33()
        s = [v + 0.05j for v in s]  # complex, off the cut
        perm = {i: 8 - i for i in range(9)}
        cells = [[perm[v] for v in t] for t in K.orientedTopSimplices()]
        K2 = cob.ChainComplex.fromTopCells(cells)
        edges, edges2 = _edges(K), _edges(K2)
        table = dict(zip(edges, s))
        inv = {v: k for k, v in perm.items()}
        s2 = [table[tuple(sorted((inv[a], inv[b])))] for (a, b) in edges2]
        for k in range(3):
            M = WM.assemble(K, s, k).toarray()
            M2 = WM.assemble(K2, s2, k).toarray()
            cells_k = [tuple(c) for c in K.kSimplexVertices(k)]
            cells2 = [tuple(c) for c in K2.kSimplexVertices(k)]
            P = np.zeros((len(cells_k), len(cells_k)))
            for j, c in enumerate(cells_k):
                image = [perm[v] for v in c]
                i2 = cells2.index(tuple(sorted(image)))
                # sign = parity of the permutation sorting the relabeled tuple
                order = sorted(range(len(image)), key=lambda p: image[p])
                sign = 1.0
                seen = [False] * len(order)
                for start in range(len(order)):
                    if seen[start]:
                        continue
                    length, p = 0, start
                    while not seen[p]:
                        seen[p] = True
                        p = order[p]
                        length += 1
                    if length % 2 == 0:
                        sign = -sign
                P[i2, j] = sign
            np.testing.assert_allclose(P @ M @ P.T, M2, atol=1e-13)
