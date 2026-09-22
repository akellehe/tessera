# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Chain Hodge pencil (#907): the specification's §14 values (T5a, T5b, T6, T7),
the one-complex proposition of §9, the rank conditions of Prop. 4.2, the
Grassmann preset against its dense oracle, and the sparse kernel path.

The scaling verification plan's geometric-fidelity sweeps G1-G6, which lived
here, are in `test_svp_geometric_fidelity_python.py` (#1208), at the plan's own
sizes and fixtures and with the plan's JSON records."""
import math

import numpy as np
import pytest
from scipy.linalg import subspace_angles

from tessera import chainhodge as ch
from tessera import cobordism as cob
from tests.chainhodge._fixtures import flat_torus, random_allowable, torus33

KS = ch.Branch.KontsevichSegal


def _expand(table):
    out = []
    for value, mult in table:
        out.extend([value] * mult)
    return np.array(out)


def _angles_deg(A, B):
    return np.degrees(subspace_angles(A, B))


def _ray(v):
    """Normalize a vector to its first nonzero component (a ray representative)."""
    v = np.asarray(v)
    pivot = next(x for x in v if abs(x) > 1e-12)
    return v / pivot


class TestT5aEuclideanThreeCycle:
    K = cob.ChainComplex.fromTopCells([[0, 1], [0, 2], [1, 2]])
    s = [1.0, 4.0, 9.0]

    def test_spectrum(self):
        hodge = ch.ChainHodge(self.K, self.s)
        spec = hodge.spectrum(1)
        np.testing.assert_allclose(sorted(np.real(spec.eigenvalues)), [0.0, 1.299254, 2.518928], atol=1e-6)
        assert np.max(np.abs(np.imag(spec.eigenvalues))) < 1e-14
        assert spec.residual < 1e-13

    def test_harmonic_chain_and_signed_lengths(self):
        hodge = ch.ChainHodge(self.K, self.s)
        read = hodge.harmonicChains(1)
        assert read.nullity == 1 and read.dense
        np.testing.assert_allclose(_ray(read.chains[:, 0]), [1.0, -1.0, 1.0], atol=1e-14)
        np.testing.assert_allclose(_ray(read.images[:, 0]), [1.0, -2.0, 3.0], atol=1e-14)
        assert hodge.betti() == [1, 1]
        rep = hodge.rankConditions(1)
        assert rep.kernelIsHarmonic

    def test_pencil_is_symmetric_and_equivalent_to_hodge_operator(self):
        hodge = ch.ChainHodge(self.K, self.s)
        P = hodge.pencil(1)
        assert P.variable == ch.PencilVariable.GeometricImage
        np.testing.assert_allclose(P.A, P.A.T, atol=1e-14)
        L = hodge.hodgeOperator(1)
        np.testing.assert_allclose(sorted(np.linalg.eigvals(L).real), sorted(np.real(hodge.spectrum(1).eigenvalues)), atol=1e-12)
        np.testing.assert_allclose(hodge.pencilAux(1), P.A, atol=1e-15)


class TestT5bMixedSignatureThreeCycle:
    K = cob.ChainComplex.fromTopCells([[0, 1], [0, 2], [1, 2]])
    s = [1.0, 1.0, -1.0]

    def test_spectrum_and_signed_lengths(self):
        hodge = ch.ChainHodge(self.K, self.s, ch.Preset.L2, KS)
        ev = np.array(hodge.spectrum(1).eigenvalues)
        expected = np.array([0.0, -6.0j, 4.8 - 3.6j])
        for e in expected:
            assert np.min(np.abs(ev - e)) < 1e-9
        read = hodge.harmonicChains(1)
        assert read.nullity == 1
        np.testing.assert_allclose(_ray(read.chains[:, 0]), [1.0, -1.0, 1.0], atol=1e-14)
        np.testing.assert_allclose(_ray(read.images[:, 0]), [1.0, -1.0, 1j], atol=1e-14)


class TestOneComplexProposition:
    """§9: a 1-complex with real s_e of any signs has a real spectrum and the
    harmonic image is the signed lengths on the declared branch."""

    def test_all_timelike_cycle(self):
        K = cob.ChainComplex.fromTopCells([[0, 1], [0, 2], [1, 2]])
        s = [-1.0, -4.0, -9.0]
        hodge = ch.ChainHodge(K, s, ch.Preset.L2, KS)
        ev = np.array(hodge.spectrum(1).eigenvalues)
        assert np.max(np.abs(ev.imag)) < 1e-13 * np.max(np.abs(ev))
        read = hodge.harmonicChains(1)
        np.testing.assert_allclose(_ray(read.chains[:, 0]), [1.0, -1.0, 1.0], atol=1e-14)
        np.testing.assert_allclose(read.images[:, 0] / read.images[0, 0], [1.0, -2.0, 3.0], atol=1e-14)
        # l_e = sqrt(s_e) on the +i branch: the image is i times the Euclidean signed lengths
        assert np.allclose(np.angle(read.images[0, 0] / read.chains[0, 0]), math.pi / 2, atol=1e-14)


class TestT6LorentzianTorus:
    TABLE = [(-13.9921, 2), (-13.0909, 2), (-12.0, 4), (-10.6274, 2), (-6.0, 2), (0.0, 2),
             (4.9655, 2), (6.0, 2), (48.0, 4), (92.9132, 2), (144.0, 1), (493.9921, 2)]

    def test_rank_conditions(self):
        K, s = torus33()
        hodge = ch.ChainHodge(K, s, ch.Preset.L2, KS)
        rep = hodge.rankConditions(1)
        assert list(rep.measured) == [17, 8, 8, 17]
        assert list(rep.expected) == [17, 8, 8, 17]
        assert rep.kernelIsHarmonic
        assert hodge.betti() == [1, 2, 1]

    def test_spectrum_table(self):
        K, s = torus33()
        hodge = ch.ChainHodge(K, s, ch.Preset.L2, KS)
        spec = hodge.spectrum(1)
        ev = np.array(spec.eigenvalues)
        assert np.max(np.abs(ev.imag)) < 3e-14 * np.max(np.abs(ev))
        np.testing.assert_allclose(np.sort(ev.real), np.sort(_expand(self.TABLE)), atol=1e-3)
        assert spec.residual < 1e-12

    def test_harmonic_gram_and_signature(self):
        """The harmonic Gram Z^T M_1 Z is non-isotropic, and with M_1 = i M_1^real
        the real Gram Z^T M_1^real Z of a real orthonormal kernel basis has
        signature (1, 1): one spacelike and one timelike harmonic cycle, so
        det(Z^T M_1 Z) = -det(Z^T M_1^real Z) > 0.

        The specification quotes det = 0.211555 without stating the kernel
        basis; the value depends on that normalization. With the kernel basis
        orthonormal in image space (this implementation's convention) the
        determinant is exactly 1/3, and with a chain-orthonormal basis it is
        exactly 3; neither reproduces 0.211555, which is recorded on #907."""
        K, s = torus33()
        hodge = ch.ChainHodge(K, s, ch.Preset.L2, KS)
        read = hodge.harmonicChains(1)
        assert read.nullity == 2
        assert read.gap > 1e6
        gram = hodge.harmonicGram(read)
        assert np.linalg.matrix_rank(gram, tol=1e-12) == 2
        assert np.linalg.det(gram) == pytest.approx(1.0 / 3.0, abs=1e-12)
        M1 = hodge.Minv(1).toarray()
        Mreal = (M1 / 1j).real
        S = np.vstack([hodge.boundary(2).toarray().T.real, hodge.boundary(1).toarray().real @ Mreal])
        _, sv, vh = np.linalg.svd(S)
        Z = vh[np.sum(sv > 1e-10):].T
        greal = Z.T @ Mreal @ Z
        w = np.linalg.eigvalsh(greal)
        assert (w > 0).sum() == 1 and (w < 0).sum() == 1
        assert np.linalg.det(greal) == pytest.approx(-1.0 / 3.0, abs=1e-12)

    def test_certificate(self):
        K, s = torus33()
        hodge = ch.ChainHodge(K, s, ch.Preset.L2, KS)
        cert = hodge.certificate()
        assert not cert.allowable and cert.margin == pytest.approx(0.0, abs=1e-14)


class TestT7EuclideanTorus:
    TABLE = [(0.0, 2), (5.671, 6), (8.0, 6), (16.0, 4), (24.0, 2), (31.2521, 6), (48.0, 1)]

    def test_spectrum_table(self):
        K, s = torus33(1.0, 1.0, 1.0)
        hodge = ch.ChainHodge(K, s)
        ev = np.array(hodge.spectrum(1).eigenvalues)
        assert np.max(np.abs(ev.imag)) < 1e-12
        np.testing.assert_allclose(np.sort(ev.real), np.sort(_expand(self.TABLE)), atol=1e-3)
        rep = hodge.rankConditions(1)
        assert rep.kernelIsHarmonic
        assert hodge.certificate().allowable


class TestGrassmannPreset:
    def test_pencil_matches_dense_oracle(self):
        """A_1 = d_1^T G_0 d_1 + G_1 d_2 G_2^{-1} d_2^T G_1 on chains, B = G_1."""
        K, s = torus33()
        hodge = ch.ChainHodge(K, s, ch.Preset.GRASSMANN_ALL)
        G = [ch.WhitneyMass.assembleGrassmann(K, s, k).toarray() for k in range(3)]
        B1 = hodge.boundary(1).toarray()
        B2 = hodge.boundary(2).toarray()
        A = B1.T @ G[0] @ B1 + G[1] @ B2 @ np.linalg.inv(G[2]) @ B2.T @ G[1]
        P = hodge.pencil(1)
        assert P.variable == ch.PencilVariable.Chain
        np.testing.assert_allclose(P.A, A, atol=1e-12)
        np.testing.assert_allclose(P.B, G[1], atol=1e-15)
        rep = hodge.rankConditions(1)
        assert rep.kernelIsHarmonic
        read = hodge.harmonicChains(1)
        assert read.nullity == 2
        np.testing.assert_allclose(read.images, G[1] @ read.chains, atol=1e-13)
        with pytest.raises(RuntimeError):
            hodge.Minv(1)
        with pytest.raises(RuntimeError):
            hodge.pencilAux(1)


class TestSparseKernelPath:
    def test_sparse_qr_agrees_with_dense_svd(self):
        rng = np.random.default_rng(29)
        K, s, _ = flat_torus(6, 0.25, False, seed=4)
        s = [v + 0.01j * rng.normal() for v in s]
        hodge = ch.ChainHodge(K, s)
        dense = hodge.harmonicChains(1)
        sparse = hodge.harmonicChains(1, 10.0, True)
        assert dense.dense and not sparse.dense
        assert dense.nullity == sparse.nullity == 2
        assert np.max(_angles_deg(dense.images, sparse.images)) < 1e-8
        assert math.isnan(sparse.gap)

    def test_crossover_refuses_dense(self):
        K, s = torus33()
        hodge = ch.ChainHodge(K, s, ch.Preset.L2, KS, 4)
        with pytest.raises(ValueError):
            hodge.pencil(1)
        read = hodge.harmonicChains(1)
        assert not read.dense and read.nullity == 2


class TestRandomAllowable:
    def test_kernel_dimension_is_betti_and_pencil_symmetric(self):
        rng = np.random.default_rng(31)
        for cells, betti in [([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3], [2, 3, 4]], [1, 0, 1]),
                             ([[0, 1, 2, 3], [1, 2, 3, 4]], [1, 0, 0, 0])]:
            K = cob.ChainComplex.fromTopCells(cells)
            s = random_allowable(K, rng, 0.2)
            hodge = ch.ChainHodge(K, s)
            assert hodge.betti() == betti
            for k in range(K.dimension() + 1):
                rep = hodge.rankConditions(k)
                assert rep.kernelIsHarmonic
                assert hodge.harmonicChains(k).nullity == betti[k]
                P = hodge.pencil(k)
                np.testing.assert_allclose(P.A, P.A.T, atol=1e-12 * max(1.0, np.abs(P.A).max()))
                np.testing.assert_allclose(P.B, P.B.T, atol=1e-14)
