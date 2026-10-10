# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The complex-symmetric pencil regime (#930): its verification on the
chain-level pencil, the fiber tracker reading Riesz bands on the pencil with
the bilinear pairing certificates and no inertia, the recursive quotient's
pencil levels naming the regime, and the observables' regime vocabulary."""
import cmath
import math

import numpy as np
import pytest

import tessera
from tessera import chainhodge as ch
from tessera import cobordism as cob
from tessera import observables as obs
from tests.chainhodge._fixtures import conformal_torus, edges, torus33, torus33_timelike_parts

Regime = cob.CertificateRegime
Pencil = Regime.ComplexSymmetricPencil
HL = cob.HodgeLaplacian
Whitney = cob.HodgeMetricSource.WhitneyPencil
Diagonal = cob.HodgeMetricSource.DiagonalWeights


def _spacetime_from(K, s):
    cells = [list(int(v) for v in t) for t in K.oriented_top_simplices()]
    st = tessera.Spacetime.from_vertex_tuples(2, cells, 1.0, 0.0)
    table = dict(zip(edges(K), s))
    for e in st.get_edge_list().to_vector():
        a, b = e.get_source().get_id(), e.get_target().get_id()
        e.set_length(cmath.sqrt(table[(min(a, b), max(a, b))]))
        e.set_phase(0.0)
    st.materialize_facets()
    return st


def _lorentzian_torus_spacetime(eps):
    K, s = torus33()
    s_eps = ch.LorentzianFamily.rotate(s, torus33_timelike_parts(K), eps)
    return _spacetime_from(K, s_eps), K


class TestRegimeVocabulary:
    def test_enum_value_and_name(self):
        assert Pencil != Regime.NonNormal
        assert str(Pencil).endswith("ComplexSymmetricPencil")


class TestPencilRegimeCertificate:
    def test_t6_instance_is_the_pencil_regime(self):
        K, s = torus33()
        cov = ch.CovariantChainHodge(ch.ChainHodge(K, s, ch.Preset.L2, ch.Branch.KontsevichSegal),
                                     ch.Connection.trivial(K))
        c = cov.regime_certificate(1)
        assert c.regime == Pencil
        assert c.trivial_connection
        assert c.symmetry_defect < 1e-13 and c.metric_symmetry_defect < 1e-15

    def test_dressed_instance_keeps_the_regime_by_the_transpose_identity(self):
        rng = np.random.default_rng(5)
        K, s = torus33()
        s = [v + 0.05j for v in s]
        links = [complex(rng.normal(), rng.normal()) for _ in range(K.num_simplices(1))]
        cov = ch.CovariantChainHodge(ch.ChainHodge(K, s), ch.Connection(K, links))
        c = cov.regime_certificate(1)
        assert c.regime == Pencil and not c.trivial_connection
        assert c.symmetry_defect < 1e-12


class TestTrackerOnThePencil:
    @pytest.mark.parametrize("eps", [0.1, 0.0])
    def test_lorentzian_torus_bands_report_the_bilinear_pairing(self, eps):
        st, K = _lorentzian_torus_spacetime(eps)
        cfg = obs.SpectralFiberConfig()
        cfg.degrees = [1]
        # The torus is translation-symmetric: every band is delocalized by
        # construction (measured excess ~0.6), so the localization conjunct is
        # opened to "any measured localization" and the pencil conjuncts decide.
        cfg.max_localization_excess = 1.0
        tracker = obs.SpectralFiberTracker(st, cfg, Whitney)
        assert tracker.metric_source() == Whitney
        support = [int(v[0]) for v in K.k_simplex_vertices(0)]
        read = tracker.enumerate_bands(support, 1)
        assert read.regime == Pencil
        assert read.solver_path == "pencil-riesz"
        assert read.dimension == 27
        assert len(read.fibers) >= 2
        # every band carries the bilinear certificates and no inertia
        for fiber in read.fibers:
            cert = fiber.certificate()
            assert cert.certificate.regime == Pencil
            assert cert.positive_signature == 0 and cert.negative_signature == 0
            assert math.isfinite(cert.pairing_condition)
            assert math.isfinite(cert.pairing_determinant.real)
            assert math.isfinite(cert.metric_symmetry_defect) and cert.metric_symmetry_defect < 1e-12
        # the harmonic band: rank two, certified, non-isotropic, Riesz projector idempotent
        harmonic = [f for f in read.fibers if all(abs(z) < 1e-8 for z in f.eigenvalues())]
        assert len(harmonic) == 1
        h = harmonic[0]
        cert = h.certificate()
        assert h.rank() == 2 and not cert.isotropic
        if eps > 0.0:
            assert cert.accepted, cert.describe()
        else:
            # #1193: at eps_L = 0 the instance is real Lorentzian and sits
            # exactly ON the Kontsevich-Segal boundary, margin zero. The band
            # is still read, still reported and still certified-looking in
            # every other measurement — and it is NOT accepted alone, which is
            # the whitepaper's rule for a result at eps_L = 0.
            assert not cert.allowable and cert.allowability_margin <= 0.0
            assert not cert.accepted
        P = np.asarray(h.projector())
        assert np.linalg.norm(P @ P - P) < 1e-8 * max(1.0, np.linalg.norm(P))
        assert abs(cert.pairing_determinant) > 1e-3
        # the left frame is the canonical bilinear one: Phi~^T Phi = I
        Phi, Psi = np.asarray(h.right_frame()), np.asarray(h.left_frame())
        np.testing.assert_allclose(Psi.T @ Phi, np.eye(2), atol=1e-8)
        # the record round-trips the pencil fields
        rec = h.to_record()
        back = obs.SpectralFiber.from_record(rec)
        assert back.certificate().certificate.regime == Pencil
        assert back.certificate().pairing_condition == pytest.approx(cert.pairing_condition)
        assert "complex-symmetric pencil" in cert.describe()

    @pytest.mark.parametrize("field,value", [("resolvent_bound_cap", 1e-6),
                                             ("min_allowability_margin", 10.0)])
    def test_the_contour_thresholds_decide_acceptance(self, field, value):
        """The harmonic band of the rotated torus is accepted under the default
        thresholds; a cap on the contour resolvent bound below the band's own,
        or a Kontsevich-Segal margin floor above the instance's, refuses it
        while reporting the same band."""
        st, K = _lorentzian_torus_spacetime(0.1)
        support = [int(v[0]) for v in K.k_simplex_vertices(0)]

        def harmonic(cfg):
            cfg.degrees = [1]
            cfg.max_localization_excess = 1.0
            read = obs.SpectralFiberTracker(st, cfg, Whitney).enumerate_bands(
                support, 1)
            (band,) = [f for f in read.fibers
                       if all(abs(z) < 1e-8 for z in f.eigenvalues())]
            return band.certificate()

        default = obs.SpectralFiberConfig()
        assert default.resolvent_bound_cap == pytest.approx(1e8)
        assert default.min_allowability_margin == 0.0
        accepted = harmonic(default)
        assert accepted.accepted
        assert accepted.resolvent_bound > value if field == "resolvent_bound_cap" \
            else accepted.allowability_margin < value
        strict = obs.SpectralFiberConfig()
        setattr(strict, field, value)
        refused = harmonic(strict)
        assert not refused.accepted
        assert refused.rank == accepted.rank

    @pytest.mark.parametrize("eps", [0.1, 0.0])
    def test_the_bilinear_left_frame_is_the_transpose_dual(self, eps):
        # #1186: on the pencil path the stored left frame IS Phi~, so the one
        # pairing reads it without conjugation, and the pair is the band's
        # biorthogonal Slater covariance Gamma = Phi Phi~^T.
        from tessera.quantum import CovarianceState
        st, K = _lorentzian_torus_spacetime(eps)
        cfg = obs.SpectralFiberConfig()
        cfg.degrees = [1]
        cfg.max_localization_excess = 1.0
        support = [int(v[0]) for v in K.k_simplex_vertices(0)]
        read = obs.SpectralFiberTracker(st, cfg, Whitney).enumerate_bands(support, 1)
        paired = 0
        for fiber in read.fibers:
            cert = fiber.certificate()
            assert cert.bilinear_left_frame
            dual = np.asarray(fiber.dual_frame())
            np.testing.assert_array_equal(dual, np.asarray(fiber.left_frame()))
            if cert.isotropic:
                continue
            paired += 1
            phi = np.asarray(fiber.right_frame())
            np.testing.assert_allclose(dual.T @ phi, np.eye(fiber.rank()), atol=1e-8)
            state = CovarianceState.from_biorthogonal_frames(phi, dual)
            np.testing.assert_allclose(np.asarray(state.gamma()),
                                       np.asarray(fiber.projector()), atol=1e-13)
            assert state.duality_defect() == pytest.approx(cert.gram_defect, abs=1e-12)
        assert paired >= 1
        # A record written before the flag existed falls back on the regime.
        rec = read.fibers[0].to_record()
        del rec["certificate"]["bilinear_left_frame"]
        back = obs.SpectralFiber.from_record(rec)
        assert back.certificate().bilinear_left_frame

    def test_fiber_transport_pairs_the_bilinear_frame_without_conjugation(self):
        # The self-transport of a pencil band through the identity transfer
        # is its own pairing, Phi~^T Phi = I; the conjugate Phi~^dagger Phi is
        # not (the complex frame makes the two differ).
        st, K = _lorentzian_torus_spacetime(0.1)
        cfg = obs.SpectralFiberConfig()
        cfg.degrees = [1]
        cfg.max_localization_excess = 1.0
        support = [int(v[0]) for v in K.k_simplex_vertices(0)]
        read = obs.SpectralFiberTracker(st, cfg, Whitney).enumerate_bands(support, 1)
        fiber = next(f for f in read.fibers if not f.certificate().isotropic)
        n = len(fiber.cell_vertices())
        transport = obs.FiberConnection().transport(fiber, fiber, np.eye(n, dtype=complex))
        raw = np.asarray(transport.raw_map)
        np.testing.assert_allclose(raw, np.eye(fiber.rank()), atol=1e-8)
        phi, dual = np.asarray(fiber.right_frame()), np.asarray(fiber.dual_frame())
        assert np.abs(dual.conj().T @ phi - np.eye(fiber.rank())).max() > 1e-6

    def test_diagonal_source_is_unchanged(self):
        st, K = _lorentzian_torus_spacetime(0.0)
        cfg = obs.SpectralFiberConfig()
        cfg.degrees = [1]
        support = [int(v[0]) for v in K.k_simplex_vertices(0)]
        legacy = obs.SpectralFiberTracker(st, cfg, metric_source=Diagonal).enumerate_bands(support, 1)
        assert legacy.regime != Pencil
        assert legacy.solver_path != "pencil-riesz"

    def test_degree_zero_keeps_the_connection_operator(self):
        st, K = _lorentzian_torus_spacetime(0.1)
        cfg = obs.SpectralFiberConfig()
        cfg.degrees = [0]
        support = [int(v[0]) for v in K.k_simplex_vertices(0)]
        read = obs.SpectralFiberTracker(st, cfg, Whitney).enumerate_bands(support, 0)
        assert read.regime != Pencil


class TestRecursiveQuotientPencilLevels:
    def test_pencil_level_names_the_regime_and_reduces_in_it(self):
        K, s = torus33()
        cov = ch.CovariantChainHodge(ch.ChainHodge(K, s), ch.Connection.trivial(K))
        P = cov.pencil(1)
        n = P.A.shape[0]
        A = np.asarray(P.A); M = np.asarray(P.B)
        comp_a = list(range(n // 2))
        comp_b = [i for i in range(n) if i not in comp_a]
        q = cob.RecursiveQuotient.over_pencil(A.flatten().tolist(), M.flatten().tolist(), n, [comp_a, comp_b])
        assert q.regime == Pencil
        # The complex-symmetric pencil has no Hermitian form to reduce against,
        # so the surrogate pairs with the transpose and the level's carried
        # metric instead of refusing.
        read = q.craig_bampton(0.0, 10.0, 1e6, 1e-6)
        assert read.window_eigenvalues
        assert max(read.eigen_residuals) < 1e-6
        exact = np.linalg.eigvals(np.linalg.solve(M, A)).real
        for value in read.window_eigenvalues:
            assert float(np.min(np.abs(exact - value))) < 1e-6 * max(1.0, abs(value))
        sheaf = q.sheaf_realization()
        assert not sheaf.emitted
        assert sheaf.certificate.regime == Pencil
