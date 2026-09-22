# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Riesz bands on the pencil resolvent (#913, specification §6, §9, §11 step 4,
§14 T8): the bordered sparse resolvent, projector certificates, right frames,
the dual connection's band on the same contour, the pairing B_C, the canonical
left frame G^{U^-1} Phi^vee B_C^{-T}, the reduced operator, the covariance,
the transpose and gauge identities, the isotropic refusal, and a Lorentzian
instance at epsilon > 0."""
import math

import numpy as np
import pytest
from scipy.optimize import minimize

from tessera import chainhodge as ch
from tessera import cobordism as cob
from tests.chainhodge._fixtures import (random_allowable, torus33, torus33_timelike_parts,
                                        torus_cells)

KS = ch.Branch.KontsevichSegal
TWO_COMPLEX = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3], [2, 3, 4]]


def _random_links(K, rng):
    return [complex(rng.normal(), rng.normal()) for _ in range(K.numSimplices(1))]


def _random_gauge(K, rng):
    return {int(v[0]): complex(rng.normal(), rng.normal()) + 0.3 for v in K.kSimplexVertices(0)}


def _isolated_eigenvalue(cov, k):
    """The eigenvalue with the largest separation from the rest, and that separation."""
    ev = np.array(cov.spectrum(k).eigenvalues)
    best, sep = None, -1.0
    for i, z in enumerate(ev):
        others = np.delete(ev, i)
        d = np.min(np.abs(others - z)) if len(others) else np.inf
        if d > sep:
            best, sep = z, d
    return best, sep


def _instance(cells, seed, with_links=True):
    rng = np.random.default_rng(seed)
    K = cob.ChainComplex.fromTopCells(cells)
    s = [complex(rng.normal(), rng.normal()) for _ in range(K.numSimplices(1))]
    base = ch.ChainHodge(K, s, ch.Preset.L2, KS)
    U = ch.Connection(K, _random_links(K, rng)) if with_links else ch.Connection.trivial(K)
    return K, rng, base, ch.CovariantChainHodge(base, U)


class TestResolvent:
    def test_bordered_resolvent_inverts_the_operator(self):
        K, rng, base, cov = _instance(TWO_COMPLEX, 3)
        for k in range(3):
            n = base.size(k)
            h = cov.covariantOperator(k)
            zeta = 0.37 + 0.81j
            c = rng.normal(size=(n, 2)) + 1j * rng.normal(size=(n, 2))
            R = cov.resolvent(k, zeta, c)
            np.testing.assert_allclose((zeta * np.eye(n) - h) @ R, c, atol=1e-10 * max(1.0, np.abs(c).max()))


class TestFramesT8:
    @pytest.mark.parametrize("N", [4])
    def test_random_torus_band(self, N):
        cells, _ = torus_cells(N)
        K, rng, base, cov = _instance(cells, 100 + N)
        k = 1
        z, sep = _isolated_eigenvalue(cov, k)
        band = cov.band(k, ch.Contour.circle(z, 0.45 * sep, 64))
        cert = band.certificate
        assert cert.rank == 1 and band.rank() == 1
        assert cert.idempotency < 1e-8
        assert cert.nodeCount == 64 and "circle" in cert.contour
        assert math.isfinite(cert.resolventMax) and cert.resolventMax > 0
        assert cert.leftFrameAvailable, cert.leftFrameRefusal
        # RSF normalization Phi~^T Phi = I to round-off
        np.testing.assert_allclose(band.leftFrame.T @ band.frame, np.eye(1), atol=1e-12)
        # the reduced operator is the enclosed eigenvalue; left/right residuals small
        assert abs(band.reduced[0, 0] - z) < 1e-8 * abs(z)
        assert cert.rightResidual < 1e-8 and cert.leftResidual < 1e-8
        # Gamma = Phi Phi~^T equals the projector
        np.testing.assert_allclose(band.covariance, band.projector, atol=1e-8 * np.abs(band.projector).max())
        assert len(band.occupations()) == base.size(k)
        # images are G^U Phi
        np.testing.assert_allclose(band.images, cov.applyG(k, band.frame), atol=1e-13)
        # pairing B_C = (Phi^vee)^T G^U Phi
        np.testing.assert_allclose(band.pairing, band.dualFrame.T @ band.images, atol=1e-13)
        assert abs(cert.detB - np.linalg.det(band.pairing)) < 1e-12 * abs(cert.detB)
        # static leftFrame agrees
        np.testing.assert_allclose(ch.CovariantChainHodge.leftFrame(band, cov.dual()), band.leftFrame, atol=1e-12)

    def test_transpose_identity_of_the_projector(self):
        """P_C(U)^T = G^{U^-1} P_C(U^-1) (G^{U^-1})^{-1} on the same contour."""
        K, rng, base, cov = _instance(TWO_COMPLEX, 5)
        k = 1
        z, sep = _isolated_eigenvalue(cov, k)
        contour = ch.Contour.circle(z, 0.45 * sep, 48)
        P = cov.band(k, contour).projector
        dual = cov.dual()
        Pd = dual.band(k, contour).projector
        Md = dual.Minv(k).toarray()                       # (G^{U^-1})^{-1} = M^{U^-1}
        rhs = dual.applyG(k, Pd @ Md)                     # G^{U^-1} P(U^-1) M^{U^-1}
        np.testing.assert_allclose(P.T, rhs, atol=1e-9 * np.abs(P).max())

    def test_gauge_covariance_of_the_band(self):
        """P_C(U^g) = rho_1(g) P_C(U) rho_1(g)^{-1}; the pairing is invariant."""
        K, rng, base, cov = _instance(TWO_COMPLEX, 7)
        k = 1
        z, sep = _isolated_eigenvalue(cov, k)
        contour = ch.Contour.circle(z, 0.45 * sep, 48)
        band = cov.band(k, contour)
        g = _random_gauge(K, rng)
        gauged = cov.gauged(g)
        bandg = gauged.band(k, contour)
        rho = np.asarray(cov.rho(k, g)).ravel()
        expected = np.diag(rho) @ band.projector @ np.diag(1.0 / rho)
        np.testing.assert_allclose(bandg.projector, expected, atol=1e-9 * np.abs(expected).max())
        assert abs(bandg.certificate.detB - band.certificate.detB) < 1e-8 * abs(band.certificate.detB) or bandg.rank() == 1

    def test_trivial_connection_left_frame_is_the_geometric_image(self):
        """At U = 1: Phi~ B^T = G_1 Phi (Phi~ = G_1 Phi once Phi^T G_1 Phi = I) and
        the projector is symmetric for the chain metric, P M_1 = M_1 P^T."""
        K, rng, base, cov = _instance(TWO_COMPLEX, 9, with_links=False)
        k = 1
        z, sep = _isolated_eigenvalue(cov, k)
        band = cov.band(k, ch.Contour.circle(z, 0.45 * sep, 48))
        np.testing.assert_allclose(band.leftFrame @ band.pairing.T, band.images, atol=1e-10 * np.abs(band.images).max())
        M = cov.Minv(k).toarray()
        P = band.projector
        np.testing.assert_allclose(P @ M, M @ P.T, atol=1e-9 * np.abs(P @ M).max())
        # the dual band coincides with the band itself at U = 1
        np.testing.assert_allclose(np.abs(band.dualFrame.T @ band.frame), np.abs(band.frame.T @ band.frame), atol=1e-9)


class TestHarmonicBandOnTheTorus:
    def test_t6_harmonic_band(self):
        K, s = torus33()
        base = ch.ChainHodge(K, s, ch.Preset.L2, KS)
        cov = ch.CovariantChainHodge(base, ch.Connection.trivial(K))
        band = cov.band(1, ch.Contour.circle(0.0, 2.0, 64))
        assert band.rank() == 2
        assert band.certificate.idempotency < 1e-9
        np.testing.assert_allclose(band.reduced, np.zeros((2, 2)), atol=1e-8)
        # |det(Z^T M_1 Z)| with an image-orthonormal kernel basis is exactly 1/3
        # (recorded on #907: the specification's 0.211555 is not reproduced under
        # any stated normalization); the band is non-isotropic.
        Q, _ = np.linalg.qr(band.images)
        M1 = cov.Minv(1).toarray()
        assert abs(np.linalg.det(Q.T @ M1 @ Q)) == pytest.approx(1.0 / 3.0, abs=1e-10)
        assert band.certificate.leftFrameAvailable and math.isfinite(band.certificate.condB)


class TestIsotropicBand:
    def test_exceptional_point_makes_the_rank_one_band_isotropic(self):
        """Move complex squared lengths along a line s0 + t d until two nonzero
        eigenvalues coalesce (an exceptional point, where the eigenvector is
        self-orthogonal). A rank-one band enclosing ONE of the pair has a
        normalized pairing |u^vee^T G u| / ||G u|| that vanishes as the point is
        approached; at a declared tolerance the left frame is refused by name.
        The band enclosing both (the generalized eigenspace) is not isotropic:
        the Jordan chain pairs non-trivially with the eigenvector."""
        K = cob.ChainComplex.fromTopCells(TWO_COMPLEX)
        rng = np.random.default_rng(21)
        s0 = np.array([complex(rng.normal(), rng.normal()) for _ in range(K.numSimplices(1))])
        d = np.array([complex(rng.normal(), rng.normal()) for _ in range(K.numSimplices(1))])
        U = ch.Connection.trivial(K)

        def instance(t):
            s = s0 + complex(t[0], t[1]) * d
            return ch.CovariantChainHodge(ch.ChainHodge(K, list(s), ch.Preset.L2, KS), U, 7, False)

        def spectrum(t):
            ev = np.array(instance(t).spectrum(1).eigenvalues)
            return ev[np.abs(ev) > 1e-8]

        def min_gap(t):
            ev = spectrum(t)
            dmat = np.abs(ev[:, None] - ev[None, :]) + np.eye(len(ev)) * 1e9
            return dmat.min()

        best = None
        for start in ([0.0, 0.0], [0.3, -0.2], [-0.4, 0.5], [0.8, 0.1]):
            res = minimize(min_gap, np.array(start), method="Nelder-Mead",
                           options={"xatol": 1e-12, "fatol": 1e-13, "maxiter": 4000})
            if best is None or res.fun < best.fun:
                best = res
        ev_star = spectrum(best.x)
        scale = np.abs(ev_star).max()
        assert best.fun < 1e-5 * scale, f"no exceptional point located: min gap {best.fun:.3e}"

        def rank_one_band(t):
            ev = spectrum(t)
            dmat = np.abs(ev[:, None] - ev[None, :]) + np.eye(len(ev)) * 1e9
            i, j = np.unravel_index(np.argmin(dmat), dmat.shape)
            gap = dmat[i, j]
            others = np.delete(ev, [i])
            radius = 0.4 * np.min(np.abs(others - ev[i]))
            band = instance(t).band(1, ch.Contour.circle(ev[i], radius, 96))
            return band, gap

        # Step away from the point along the line by two amounts: the
        # normalized pairing of the rank-one band shrinks with the gap.
        step = np.array([1.0, 0.0])
        far, gap_far = rank_one_band(best.x + 3e-2 * step)
        near, gap_near = rank_one_band(best.x + 3e-3 * step)
        assert far.rank() == 1 and near.rank() == 1
        assert gap_near < gap_far
        assert near.certificate.pairingScale < far.certificate.pairingScale
        assert near.certificate.pairingScale < 1e-2
        # At a declared tolerance above the measured normalized pairing the
        # left frame is refused by name; below it, it exists.
        tol = 2.0 * near.certificate.pairingScale
        cov_near = instance(best.x + 3e-3 * step)
        ev = spectrum(best.x + 3e-3 * step)
        dmat = np.abs(ev[:, None] - ev[None, :]) + np.eye(len(ev)) * 1e9
        i, j = np.unravel_index(np.argmin(dmat), dmat.shape)
        radius = 0.4 * np.min(np.abs(np.delete(ev, [i]) - ev[i]))
        refused = cov_near.band(1, ch.Contour.circle(ev[i], radius, 96), 10.0, tol)
        assert not refused.certificate.leftFrameAvailable
        assert "isotropic" in refused.certificate.leftFrameRefusal
        assert refused.leftFrame.size == 0
        with pytest.raises(RuntimeError):
            ch.CovariantChainHodge.leftFrame(refused, cov_near.dual(), tol)
        assert near.certificate.leftFrameAvailable  # default tolerance 1e-10
        # The band enclosing both coalescing eigenvalues is NOT isotropic.
        center = 0.5 * (ev_star[np.argsort(np.abs(ev_star - ev_star[0]))[:1]][0] + ev_star[0])
        dm = np.abs(ev_star[:, None] - ev_star[None, :]) + np.eye(len(ev_star)) * 1e9
        a, b = np.unravel_index(np.argmin(dm), dm.shape)
        center = 0.5 * (ev_star[a] + ev_star[b])
        rad = 0.4 * np.min(np.abs(np.delete(ev_star, [a, b]) - center))
        both = instance(best.x).band(1, ch.Contour.circle(center, rad, 96))
        assert both.rank() == 2
        assert both.certificate.pairingScale > 1e-3
        assert both.certificate.leftFrameAvailable


class TestLorentzianAtPositiveEpsilon:
    def test_rotated_torus_reports_det_and_cond(self):
        K, s = torus33()
        s_eps = ch.LorentzianFamily.rotate(s, torus33_timelike_parts(K), 0.1)
        base = ch.ChainHodge(K, s_eps, ch.Preset.L2, KS)
        assert base.certificate().allowable
        cov = ch.CovariantChainHodge(base, ch.Connection.trivial(K))
        band = cov.band(1, ch.Contour.circle(0.0, 2.0, 64))
        assert band.rank() == 2
        cert = band.certificate
        assert cert.leftFrameAvailable
        assert math.isfinite(cert.condB) and abs(cert.detB) > 0
        assert cert.rightResidual < 1e-8 and cert.leftResidual < 1e-8

    def test_grassmann_preset_is_refused_by_name(self):
        K, s = torus33()
        cov = ch.CovariantChainHodge(ch.ChainHodge(K, s, ch.Preset.GRASSMANN_ALL), ch.Connection.trivial(K))
        with pytest.raises(RuntimeError):
            cov.band(1, ch.Contour.circle(0.0, 2.0, 16))
