# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Acceptance tests for localized spectral fibers and their certificates
(:class:`tessera.observables.SpectralFiber` /
:class:`tessera.observables.SpectralFiberTracker`), ticket #769 / design
spec sections 5.4, 6.3, and 9 (Algorithm B).

Covers every ticket acceptance bullet:

* exact small fixtures recover their known projectors and multiplicities
  (hand-built projectors, an exactly degenerate band, independent numpy /
  scipy references, and the holed-surface harmonic space);
* random in-band unitary rotations change no band identity or downstream
  projector observable;
* relabeling leaves localization, gaps, and persistence invariant;
* a deliberately non-normal fixture uses the general (biorthogonal) path
  and reports its conditioning;
* cold recomputation matches incremental projector tracking (through the
  #764 AnalyticCache contract with touched-star invalidation);
* no eigenvalue threshold is used as a Betti-number oracle — band ranks
  come from the relative gap rule, and the exact integer homology appears
  below only as a test-side cross-reference.

Closed-form anchors (verified, not fitted):

* path P3 at k=0: spec {0, 1, 3} with hand-computable eigenprojectors;
* triangle C3 at k=0 and k=1: spec {0, 3, 3}, degenerate projector
  I - J/3;
* one-timelike-edge triangle (l^2 = -alpha^2 on edge (1,2)), SquaredContent:
  spec(L_1) = {0, 3, 1 - 2/alpha^2}, harmonic Krein norm (2 - alpha^2)/3
  (positive below alpha = sqrt(2), negative above, defective crossing at
  sqrt(2)); Content: spec(L_1) = {0, 3, 1 - 2i/alpha} (non-normal);
* ring C_n at k=0: spec {2 - 2 cos(2 pi j / n)} with rank-2 cosine pairs.
"""
import cmath
import math
import random
import sys
import unittest
from pathlib import Path

import numpy as np

import tessera

obs = tessera.observables
cob = tessera.cobordism

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "cobordism"))
from _holed_surface import holed_surface  # noqa: E402

MACHINE = 1e-12   # machine-precision claims on exact closed forms
SOLVER = 1e-9     # certified-numerical solver-level agreement

# The repository's NaN-aware every-channel record gate (two NaNs agree; any
# numeric drift, shape, or status change is a flagged channel).
_delta = obs.ObservableGates.report_delta


# --------------------------------------------------------------------------- #
# fixture builders
# --------------------------------------------------------------------------- #
def _from_simplices(num_vertices, simplices, ids=None):
    """Explicit-complex idiom shared with the Hodge Laplacian tests.  `ids`
    optionally relabels vertex identifiers (position i gets ids[i])."""
    sig = tessera.Signature(4, tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    st = tessera.Spacetime(metric, tessera.HERMITIAN_WEIGHTED, 1.0, 1.0,
                           tessera.PREFERRED, tessera.Toroid())
    ids = list(range(num_vertices)) if ids is None else ids
    verts = [st.createVertex(i) for i in ids]
    for simplex in simplices:
        st.createSimplex([verts[i] for i in simplex])
    for e in st.getEdgeList().toVector():
        e.setLength(1.0 + 0j)
        e.setPhase(0.0)
    return st


def _triangle(alpha=None):
    """3-cycle 0-1-2-0; when `alpha` is given, edge (1,2) is timelike with
    l^2 = -alpha^2 (the closed-form Lorentzian fixture)."""
    st = _from_simplices(3, [(0, 1), (1, 2), (2, 0)])
    if alpha is not None:
        for e in st.getEdgeList().toVector():
            pair = {e.getSource().getId(), e.getTarget().getId()}
            if pair == {1, 2}:
                e.setLength(cmath.sqrt(complex(-(alpha ** 2))))
    return st


def _ring(n):
    return _from_simplices(n, [(i, (i + 1) % n) for i in range(n)])


def _two_triangles():
    """Two disjoint triangles on non-contiguous ids (the AnalyticCache test
    fixture)."""
    st = _from_simplices(
        6, [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)],
        ids=[0, 1, 2, 10, 11, 12])
    return st


def _edge(st, a, b):
    for e in st.getEdgeList().toVector():
        if {e.getSource().getId(), e.getTarget().getId()} == {a, b}:
            return e
    raise KeyError((a, b))


def _all_vertices(st):
    return sorted(v.getId() for v in st.getVertexList().toVector())


def _np_k0(st, support):
    """Independent numpy D - A oracle on the induced subgraph, under the
    documented HodgeLaplacian k=0 conventions."""
    ids = sorted(i for i in _all_vertices(st) if i in set(support))
    idx = {v: i for i, v in enumerate(ids)}
    n = len(ids)
    A = np.zeros((n, n), complex)
    D = np.zeros(n)
    for e in st.getEdgeList().toVector():
        s, t = e.getSource().getId(), e.getTarget().getId()
        if s not in idx or t not in idx or s == t:
            continue
        w = e.getLength() * e.getLength()
        z = w * np.exp(1j * e.getPhase())
        A[idx[s], idx[t]] += z
        A[idx[t], idx[s]] += np.conj(z)
        D[idx[s]] += abs(w)
        D[idx[t]] += abs(w)
    return np.diag(D).astype(complex) - A, ids


def _reconstruct(read):
    """Spectral resolution sum_bands Phi Lambda Psi^dagger W — equals the
    restricted operator when the enumeration is complete and correct."""
    n = read.dimension
    L = np.zeros((n, n), complex)
    for f in read.fibers:
        Phi = np.asarray(f.rightFrame())
        Psi = np.asarray(f.leftFrame())
        W = np.asarray(f.weightDiagonal())
        lam = np.array(f.eigenvalues())
        L += Phi @ np.diag(lam) @ Psi.conj().T @ np.diag(W)
    return L


def _tracker(st, **cfg_kwargs):
    cfg = obs.SpectralFiberConfig()
    for k, v in cfg_kwargs.items():
        setattr(cfg, k, v)
    return obs.SpectralFiberTracker(st, cfg)


# The #808 localization acceptance conjunct: a band certifies only when its
# rank-normalized localization excess (n_eff - rank)/(n - rank) is at most
# `maxLocalizationExcess`.  Every band of a VERTEX-TRANSITIVE fixture (C_n,
# the closed icosahedron, an unjittered metric) has a uniform projector
# diagonal, hence n_eff = n and excess EXACTLY 1: it is perfectly
# delocalized and never certifies under the default.  Tests whose subject is
# a different property declare the permissive analysis cap below, which
# accepts any MEASURED localization (an unmeasured NaN still fails) and
# reproduces the pre-#808 acceptance; the conjunct itself is exercised in
# TestLocalizationConjunct.
ANY_LOCALIZATION = {"maxLocalizationExcess": 1.0}


BANNED_LABELS = ("taste", "flavor", "flavour", "color", "colour",
                 "antiparticle")


# --------------------------------------------------------------------------- #
# exact small fixtures recover known projectors and multiplicities
# --------------------------------------------------------------------------- #
class TestExactSmallFixtures(unittest.TestCase):

    def test_path_graph_known_spectrum_and_projectors(self):
        # P3 (path 0-1-2), k=0: L = [[1,-1,0],[-1,2,-1],[0,-1,1]],
        # spec {0, 1, 3} with hand-computed eigenprojectors.
        st = _from_simplices(3, [(0, 1), (1, 2)])
        read = _tracker(st, **ANY_LOCALIZATION).enumerateBands([0, 1, 2], 0)
        self.assertEqual(read.solverPath, "dense-self-adjoint")
        lam = np.array(read.coveredEigenvalues).real
        np.testing.assert_allclose(lam, [0.0, 1.0, 3.0], atol=MACHINE)
        self.assertEqual([f.rank() for f in read.fibers], [1, 1, 1])
        vecs = [np.array([1, 1, 1]) / math.sqrt(3),
                np.array([1, 0, -1]) / math.sqrt(2),
                np.array([1, -2, 1]) / math.sqrt(6)]
        for fiber, v in zip(read.fibers, vecs):
            P = np.asarray(fiber.projector())
            self.assertLessEqual(np.abs(P - np.outer(v, v)).max(), MACHINE)
            self.assertTrue(fiber.accepted())
            self.assertTrue(fiber.certificate().certificate.holds())

    def test_triangle_exactly_degenerate_band_projector(self):
        # C3, k=0: spec {0, 3, 3}; the degenerate rank-2 band's projector is
        # exactly I - J/3 (hand-built), a single band object.
        st = _triangle()
        read = _tracker(st, **ANY_LOCALIZATION).enumerateBands([0, 1, 2], 0)
        self.assertEqual([f.rank() for f in read.fibers], [1, 2])
        deg = read.fibers[1]
        self.assertEqual(deg.rank(), 2)
        P = np.asarray(deg.projector())
        self.assertLessEqual(
            np.abs(P - (np.eye(3) - np.ones((3, 3)) / 3.0)).max(), MACHINE)
        c = deg.certificate()
        self.assertEqual((c.positiveSignature, c.negativeSignature), (2, 0))
        self.assertEqual(c.lowerGap, 3.0)
        self.assertEqual(c.upperGap, math.inf)
        self.assertTrue(c.accepted)

    def test_band_center_and_window(self):
        # Band center = mean of the band eigenvalues; the frequency window
        # is [min Re, max Re].  Degenerate C3 band: center exactly 3;
        # Content-triangle complex band: center exactly 1 - 2i.
        st = _triangle()
        read = _tracker(st).enumerateBands([0, 1, 2], 0)
        deg = read.fibers[1]
        self.assertLessEqual(abs(deg.bandCenter() - 3.0), MACHINE)
        c = deg.certificate()
        self.assertLessEqual(abs(c.frequencyLower - 3.0), MACHINE)
        self.assertLessEqual(abs(c.frequencyUpper - 3.0), MACHINE)
        st2 = _triangle(alpha=1.0)
        tracker = obs.SpectralFiberTracker(
            st2, obs.SpectralFiberConfig(), cob.HodgeWeightConvention.Content)
        read2 = tracker.enumerateBands([0, 1, 2], 1)
        complex_band = min(
            read2.fibers,
            key=lambda f: abs(np.array(f.eigenvalues())[0] - (1.0 - 2.0j)))
        self.assertLessEqual(abs(complex_band.bandCenter() - (1.0 - 2.0j)),
                             1e-10)

    def test_certificate_domains_and_cache_kind(self):
        # Band certificates speak for a frequency window (BandWindow); the
        # solve certificate is the whole-operator Static claim.
        st = _triangle()
        read = _tracker(st).enumerateBands([0, 1, 2], 0)
        self.assertEqual(read.solveCertificate.domain,
                         cob.CertificateDomain.Static)
        for f in read.fibers:
            self.assertEqual(f.certificate().certificate.domain,
                             cob.CertificateDomain.BandWindow)
        self.assertEqual(obs.SpectralFiberTracker.CACHE_KIND,
                         "spectral-fiber")

    def test_triangle_k1_metric_spectrum(self):
        # All-spacelike C3 at k=1: spec(L_1) = {0, 3, 3} (no 2-cells).
        st = _triangle()
        read = _tracker(st).enumerateBands([0, 1, 2], 1)
        lam = np.array(read.coveredEigenvalues).real
        np.testing.assert_allclose(np.sort(lam), [0.0, 3.0, 3.0],
                                   atol=MACHINE)
        self.assertEqual([f.rank() for f in read.fibers], [1, 2])

    def test_projectors_match_independent_numpy_eigendecomposition(self):
        # Square 00-01-11-10 + diagonal (the testbed): band projectors must
        # equal the projectors built from an independent numpy eigh.
        st = _from_simplices(4, [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)])
        read = _tracker(st).enumerateBands([0, 1, 2, 3], 0)
        L, _ = _np_k0(st, [0, 1, 2, 3])
        lam_ref, U = np.linalg.eigh(L)
        lam_mine = np.array(read.coveredEigenvalues).real
        np.testing.assert_allclose(lam_mine, lam_ref, atol=1e-10)
        start = 0
        for fiber in read.fibers:
            r = fiber.rank()
            block = U[:, start:start + r]
            P_ref = block @ block.conj().T
            self.assertLessEqual(
                np.abs(np.asarray(fiber.projector()) - P_ref).max(), 1e-10)
            start += r

    def test_spectral_resolution_reconstructs_the_operator(self):
        # sum_bands Phi Lambda Psi^dagger W == the restricted operator; with
        # support = all vertices this pins the restricted assembly to the
        # whole-complex conventions on every regime path.
        st = _from_simplices(4, [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)])
        read = _tracker(st).enumerateBands([0, 1, 2, 3], 0)
        L_ref, _ = _np_k0(st, [0, 1, 2, 3])
        self.assertLessEqual(np.abs(_reconstruct(read) - L_ref).max(), 1e-10)

    def test_krein_reconstruction_matches_hodge_laplacian(self):
        # Signed-regime whole-support reconstruction equals the global
        # HodgeLaplacian d'Alembertian entry for entry.
        st = _triangle(alpha=2.0)
        read = _tracker(st).enumerateBands([0, 1, 2], 1)
        n = read.dimension
        L_ref = np.array(cob.HodgeLaplacian(st).laplacian(1)).reshape(n, n)
        self.assertLessEqual(np.abs(_reconstruct(read) - L_ref).max(), 1e-10)

    def test_holed_surface_recovers_harmonic_rank_by_gap_rule(self):
        # Holed icosahedron (3 disjoint windows -> b1 = 2).  The lowest band
        # is found by the RELATIVE GAP RULE; the exact integer homology is a
        # test-side cross-reference only, never a detector input.
        st, _, _, _ = holed_surface(degree=1, jitter=True)
        read = _tracker(st, **ANY_LOCALIZATION).enumerateBands(
            _all_vertices(st), 1)
        self.assertEqual(read.regime,
                         cob.CertificateRegime.PositiveSemidefinite)
        lowest = read.fibers[0]
        b1 = cob.ChainComplex.fromSpacetime(st).bettiNumbers()[1]
        self.assertEqual(b1, 2)               # the reference
        self.assertEqual(lowest.rank(), 2)    # recovered by the gap rule
        self.assertTrue(lowest.accepted())
        # ... and that band is NOT localized: its effective support covers
        # 62% of the 30 edges, so under the #808 default localization
        # conjunct this fixture's harmonic band does NOT certify as a fiber.
        # The whitepaper says as much: a fiber "need not be a harmonic space
        # and therefore need not be supported by a hole.  What it does
        # require is a spectral gap, localization, and persistence."
        self.assertGreater(lowest.certificate().localizationSupportFraction,
                           0.5)
        self.assertFalse(_tracker(st).enumerateBands(
            _all_vertices(st), 1).fibers[0].accepted())
        P = np.asarray(lowest.projector())
        self.assertLessEqual(abs(P.trace().real - 2.0), 1e-9)
        self.assertLessEqual(np.abs(P @ P - P).max(), 1e-9)
        # The full covered spectrum agrees with the independent
        # HodgeLaplacian eigensolve of the same operator.
        lam_ref = np.sort(np.array(cob.HodgeLaplacian(st).eigenvalues(1)).real)
        lam = np.sort(np.array(read.coveredEigenvalues).real)
        self.assertLessEqual(np.abs(lam - lam_ref).max(), 1e-10)

    def test_icosahedron_top_degree_band(self):
        # Closed icosahedron S^2: b2 = 1; the k=2 kernel band has rank 1.
        st, _, _, _ = holed_surface(degree=1, jitter=False)
        # holed_surface removes 3 faces; rebuild the closed surface instead.
        faces = [
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
        ]
        closed = tessera.Spacetime.fromVertexTuples(2, faces, 1.0, 0.0)
        closed.materializeFacets()
        read = _tracker(closed, **ANY_LOCALIZATION).enumerateBands(
            _all_vertices(closed), 2)
        betti = cob.ChainComplex.fromSpacetime(closed).bettiNumbers()
        self.assertEqual(betti[2], 1)
        self.assertEqual(read.fibers[0].rank(), 1)
        self.assertTrue(read.fibers[0].accepted())
        self.assertLessEqual(
            abs(np.array(read.fibers[0].eigenvalues())[0]), 1e-9)

    def test_ring_cosine_pairs(self):
        # C12 at k=0: spec {2 - 2 cos(2 pi j / 12)}: rank-1 kernel, rank-2
        # cosine pairs, rank-1 top (j = 6).
        st = _ring(12)
        read = _tracker(st).enumerateBands(list(range(12)), 0)
        lam = np.array(read.coveredEigenvalues).real
        exact = np.sort(2.0 - 2.0 * np.cos(2.0 * np.pi * np.arange(12) / 12))
        np.testing.assert_allclose(lam, exact, atol=1e-10)
        self.assertEqual([f.rank() for f in read.fibers], [1, 2, 2, 2, 2, 2, 1])


# --------------------------------------------------------------------------- #
# metric regimes: positive, Krein signed, non-normal
# --------------------------------------------------------------------------- #
class TestRegimes(unittest.TestCase):

    def test_positive_regime_is_verified_self_adjoint(self):
        read = _tracker(_triangle()).enumerateBands([0, 1, 2], 0)
        self.assertEqual(read.regime,
                         cob.CertificateRegime.PositiveSemidefinite)
        for f in read.fibers:
            c = f.certificate()
            self.assertTrue(c.selfAdjoint)
            self.assertEqual(c.leftResidual, c.eigenResidual)
            self.assertEqual((c.positiveSignature, c.negativeSignature),
                             (f.rank(), 0))
            # W-orthonormal frame: Phi^dagger W Phi = I to machine precision.
            Phi = np.asarray(f.rightFrame())
            W = np.diag(np.asarray(f.weightDiagonal()))
            G = Phi.conj().T @ W @ Phi
            self.assertLessEqual(np.abs(G - np.eye(f.rank())).max(), MACHINE)
            self.assertLessEqual(c.gramDefect, MACHINE)

    def test_krein_closed_form_spectrum_and_inertia(self):
        # spec(L_1) = {0, 3, 1 - 2/alpha^2}; harmonic Krein norm
        # (2 - alpha^2)/3: POSITIVE at alpha=1, NEGATIVE at alpha=2.
        for alpha, harmonic_sig in ((1.0, (1, 0)), (2.0, (0, 1))):
            st = _triangle(alpha=alpha)
            read = _tracker(st, **ANY_LOCALIZATION).enumerateBands(
                [0, 1, 2], 1)
            self.assertEqual(read.regime,
                             cob.CertificateRegime.HermitianIndefinite)
            self.assertEqual(read.solverPath, "dense-general")
            lam = sorted(np.array(read.coveredEigenvalues).real)
            expect = sorted([0.0, 3.0, 1.0 - 2.0 / alpha ** 2])
            np.testing.assert_allclose(lam, expect, atol=1e-10)
            harmonic = min(
                read.fibers,
                key=lambda f: abs(np.array(f.eigenvalues())[0]))
            c = harmonic.certificate()
            self.assertEqual((c.positiveSignature, c.negativeSignature),
                             harmonic_sig)
            self.assertFalse(c.selfAdjoint)
            self.assertTrue(c.accepted)

    def test_krein_gram_normalized_to_signature_matrix(self):
        # Phi^dagger W Phi = J = diag(I_p, -I_q) after normalization, and
        # Psi = Phi J gives Psi^dagger W Phi = I.
        st = _triangle(alpha=2.0)
        read = _tracker(st).enumerateBands([0, 1, 2], 1)
        for f in read.fibers:
            c = f.certificate()
            p, q = c.positiveSignature, c.negativeSignature
            self.assertEqual(p + q, f.rank())  # nonsingular W-Gram here
            Phi = np.asarray(f.rightFrame())
            Psi = np.asarray(f.leftFrame())
            W = np.diag(np.asarray(f.weightDiagonal()))
            J = np.diag([1.0] * p + [-1.0] * q)
            self.assertLessEqual(
                np.abs(Phi.conj().T @ W @ Phi - J).max(), 1e-10)
            self.assertLessEqual(
                np.abs(Psi.conj().T @ W @ Phi - np.eye(f.rank())).max(),
                1e-10)
            self.assertLessEqual(c.gramDefect, 1e-10)

    def test_negative_signature_is_a_certificate_not_a_label(self):
        st = _triangle(alpha=2.0)
        read = _tracker(st).enumerateBands([0, 1, 2], 1)
        harmonic = min(read.fibers,
                       key=lambda f: abs(np.array(f.eigenvalues())[0]))
        text = harmonic.certificate().describe().lower()
        for banned in BANNED_LABELS:
            self.assertNotIn(banned, text)

    def test_non_normal_content_convention_known_spectrum(self):
        # Content weights put the timelike weight on the imaginary axis:
        # spec(L_1) = {0, 3, 1 - 2i/alpha} — a deliberately non-normal
        # operator with a KNOWN spectrum.
        alpha = 1.0
        st = _triangle(alpha=alpha)
        cfg = obs.SpectralFiberConfig()
        cfg.maxLocalizationExcess = 1.0   # subject: the spectrum
        tracker = obs.SpectralFiberTracker(
            st, cfg, cob.HodgeWeightConvention.Content)
        read = tracker.enumerateBands([0, 1, 2], 1)
        self.assertEqual(read.regime, cob.CertificateRegime.NonNormal)
        self.assertEqual(read.solverPath, "dense-general")
        lam = sorted(np.array(read.coveredEigenvalues),
                     key=lambda z: (z.real, z.imag))
        expect = sorted([0.0, 3.0, 1.0 - 2.0j / alpha],
                        key=lambda z: (z.real, z.imag))
        np.testing.assert_allclose(lam, expect, atol=1e-10)
        for f in read.fibers:
            c = f.certificate()
            self.assertFalse(c.selfAdjoint)
            # Both residuals and the conditioning are REPORTED.
            self.assertTrue(np.isfinite(c.eigenResidual))
            self.assertTrue(np.isfinite(c.leftResidual))
            # #808: the projector norm and the FRAME condition number are
            # separate quantities, both reported under their own names.
            self.assertGreaterEqual(c.projectorNorm, 1.0 - 1e-12)
            self.assertGreaterEqual(c.frameConditionNumber, 1.0 - 1e-12)
            # Matched biorthogonal frames: Psi^dagger W Phi = I.
            Phi = np.asarray(f.rightFrame())
            Psi = np.asarray(f.leftFrame())
            W = np.diag(np.asarray(f.weightDiagonal()))
            self.assertLessEqual(
                np.abs(Psi.conj().T @ W @ Phi - np.eye(f.rank())).max(),
                1e-11)

    def test_non_normal_matches_scipy_left_right_eigenvectors(self):
        # Independent scipy left/right eigendecomposition of the SAME
        # operator (HodgeLaplacian's Content d'Alembertian): each rank-1
        # band projector must equal the scipy Riesz projector
        # v (u^H v)^{-1} u^H.
        from scipy.linalg import eig
        st = _triangle(alpha=1.0)
        tracker = obs.SpectralFiberTracker(
            st, obs.SpectralFiberConfig(), cob.HodgeWeightConvention.Content)
        read = tracker.enumerateBands([0, 1, 2], 1)
        n = read.dimension
        L = np.array(
            cob.HodgeLaplacian(st, cob.HodgeWeightConvention.Content)
            .laplacian(1)).reshape(n, n)
        w, vl, vr = eig(L, left=True, right=True)
        for f in read.fibers:
            lam = np.array(f.eigenvalues())[0]
            i = int(np.argmin(np.abs(w - lam)))
            self.assertLessEqual(abs(w[i] - lam), 1e-10)
            u, v = vl[:, i], vr[:, i]
            P_ref = np.outer(v, u.conj()) / (u.conj() @ v)
            self.assertLessEqual(
                np.abs(np.asarray(f.projector()) - P_ref).max(), 1e-9)

    def test_self_adjoint_solver_never_applied_to_non_self_adjoint(self):
        # Every non-normal read reports the general path and
        # selfAdjoint=False on every band; the solve certificate's regime is
        # NonNormal.
        st = _triangle(alpha=1.5)
        tracker = obs.SpectralFiberTracker(
            st, obs.SpectralFiberConfig(), cob.HodgeWeightConvention.Content)
        read = tracker.enumerateBands([0, 1, 2], 1)
        self.assertEqual(read.solverPath, "dense-general")
        self.assertEqual(read.solveCertificate.regime,
                         cob.CertificateRegime.NonNormal)
        for f in read.fibers:
            self.assertFalse(f.certificate().selfAdjoint)


# --------------------------------------------------------------------------- #
# one pairing across regimes: the transpose dual of the right frame (#1186)
# --------------------------------------------------------------------------- #
class TestTransposeDualFrame(unittest.TestCase):
    """dualFrame() is Phi~ with Phi~^T Phi = I in every regime, the projector
    is Phi Phi~^T, and the pair is the band's biorthogonal Slater
    covariance."""

    def _reads(self):
        positive = _tracker(_triangle()).enumerateBands([0, 1, 2], 0)
        krein = _tracker(_triangle(alpha=2.0), **ANY_LOCALIZATION) \
            .enumerateBands([0, 1, 2], 1)
        cfg = obs.SpectralFiberConfig()
        cfg.maxLocalizationExcess = 1.0
        non_normal = obs.SpectralFiberTracker(
            _triangle(alpha=1.0), cfg,
            cob.HodgeWeightConvention.Content).enumerateBands([0, 1, 2], 1)
        self.assertEqual(positive.regime,
                         cob.CertificateRegime.PositiveSemidefinite)
        self.assertEqual(krein.regime,
                         cob.CertificateRegime.HermitianIndefinite)
        self.assertEqual(non_normal.regime, cob.CertificateRegime.NonNormal)
        return positive, krein, non_normal

    def test_dual_frame_pairs_to_the_identity_by_the_transpose(self):
        for read in self._reads():
            for f in read.fibers:
                phi = np.asarray(f.rightFrame())
                dual = np.asarray(f.dualFrame())
                self.assertFalse(f.certificate().bilinearLeftFrame)
                np.testing.assert_allclose(dual.T @ phi, np.eye(f.rank()),
                                           rtol=0, atol=1e-10)
                # Off the pencil path Phi~ = W conj(Psi): the transpose of
                # Psi^dagger W, the solver's own normalization.
                psi = np.asarray(f.leftFrame())
                w = np.asarray(f.weightDiagonal())
                np.testing.assert_allclose(dual, w[:, None] * psi.conj(),
                                           rtol=0, atol=1e-15)

    def test_projector_is_the_transpose_product_and_unchanged(self):
        for read in self._reads():
            for f in read.fibers:
                phi = np.asarray(f.rightFrame())
                psi = np.asarray(f.leftFrame())
                w = np.diag(np.asarray(f.weightDiagonal()))
                p = np.asarray(f.projector())
                np.testing.assert_allclose(
                    p, phi @ np.asarray(f.dualFrame()).T, rtol=0, atol=1e-14)
                np.testing.assert_allclose(p, phi @ psi.conj().T @ w,
                                           rtol=0, atol=1e-13)

    def test_frames_give_the_bands_biorthogonal_covariance(self):
        from tessera.quantum import CovarianceDual, CovarianceState
        for read in self._reads():
            for f in read.fibers:
                state = CovarianceState.fromBiorthogonalFrames(
                    np.asarray(f.rightFrame()), np.asarray(f.dualFrame()))
                self.assertEqual(state.dual(), CovarianceDual.Transpose)
                np.testing.assert_allclose(np.asarray(state.gamma()),
                                           np.asarray(f.projector()),
                                           rtol=0, atol=1e-13)
                self.assertLess(state.dualityDefect(), 1e-9)
                self.assertLess(state.purityDefect(), 1e-9)
                self.assertLess(abs(state.particleNumber() - f.rank()), 1e-9)

    def test_record_keeps_the_frame_convention(self):
        _, _, non_normal = self._reads()
        f = non_normal.fibers[0]
        back = obs.SpectralFiber.fromRecord(f.toRecord())
        self.assertFalse(back.certificate().bilinearLeftFrame)
        np.testing.assert_array_equal(np.asarray(back.dualFrame()),
                                      np.asarray(f.dualFrame()))


# --------------------------------------------------------------------------- #
# the relative gap rule: closure, straddling, truncation — negative controls
# --------------------------------------------------------------------------- #
class TestGapRule(unittest.TestCase):

    def test_gap_closure_returns_uncertified_not_a_flip(self):
        # At alpha = sqrt(2) the third eigenvalue 1 - 2/alpha^2 collides
        # with the harmonic 0 (the defective crossing).  The near-zero bands
        # are REPORTED but UNCERTIFIED (conditioning blows up, grade never
        # holds); the isolated band at 3 stays certified.
        st = _triangle(alpha=math.sqrt(2.0))
        read = _tracker(st, **ANY_LOCALIZATION).enumerateBands([0, 1, 2], 1)
        near_zero = [f for f in read.fibers
                     if abs(np.array(f.eigenvalues())[0]) < 1.0]
        self.assertGreaterEqual(len(near_zero), 1)
        for f in near_zero:
            self.assertFalse(f.accepted())
            self.assertFalse(f.certificate().certificate.holds())
            self.assertEqual(f.certificate().certificate.grade,
                             cob.CertificateGrade.HeuristicDiscovery)
        top = max(read.fibers, key=lambda f: np.array(f.eigenvalues())[0].real)
        self.assertTrue(top.accepted())

    def test_near_degenerate_pair_straddles_the_gap_rule(self):
        # Eigenvalues {0, delta, 3} with delta = 1 - 2/alpha^2 tunable.
        # Inside the grouping width: ONE rank-2 band.  Outside grouping but
        # inside the isolation floor: TWO UNCERTIFIED rank-1 bands.  Well
        # separated: TWO CERTIFIED bands.  The treatment moves continuously
        # with the gap — never a discontinuous identity change.
        def bands_at(delta, grouping, min_gap):
            alpha = math.sqrt(2.0 / (1.0 - delta))
            st = _triangle(alpha=alpha)
            read = _tracker(st, groupingTolerance=grouping,
                            minRelativeGap=min_gap,
                            **ANY_LOCALIZATION).enumerateBands([0, 1, 2], 1)
            return [(f.rank(), f.accepted()) for f in read.fibers
                    if np.array(f.eigenvalues())[0].real < 1.0]

        # delta/scale ~ 3e-7 < grouping 1e-6: one rank-2 band.
        merged = bands_at(1e-6, 1e-6, 1e-4)
        self.assertEqual([r for r, _ in merged], [2])
        # grouping 1e-8 < delta/scale ~ 3e-7 < minRelativeGap 1e-4: split
        # into two rank-1 bands, both uncertified (the gap is closing).
        split = bands_at(1e-6, 1e-8, 1e-4)
        self.assertEqual([r for r, _ in split], [1, 1])
        self.assertEqual([a for _, a in split], [False, False])
        # A wide-open gap: two certified rank-1 bands.
        wide = bands_at(0.5, 1e-8, 1e-4)
        self.assertEqual(wide, [(1, True), (1, True)])

    def test_uncertified_bands_are_reported_not_dropped(self):
        st = _triangle(alpha=math.sqrt(2.0))
        read = _tracker(st).enumerateBands([0, 1, 2], 1)
        total_rank = sum(f.rank() for f in read.fibers)
        self.assertEqual(total_rank, read.dimension)

    def test_zero_operator_single_certified_band(self):
        # A single isolated vertex (no internal edges): L = [[0]] — one
        # rank-1 band with infinite gaps on both sides.
        st = _triangle()
        read = _tracker(st).enumerateBands([0], 0)
        self.assertEqual(read.dimension, 1)
        self.assertEqual(len(read.fibers), 1)
        c = read.fibers[0].certificate()
        self.assertEqual(c.lowerGap, math.inf)
        self.assertEqual(c.upperGap, math.inf)
        self.assertTrue(c.accepted)

    def test_no_betti_oracle_weak_bridge_dumbbell(self):
        # Two K4 cliques joined by one weak edge: lambda_2 is tiny but
        # NONZERO while b0 = 1.  With a coarse grouping width the detector
        # reports a rank-2 near-zero BAND — bands are gap-rule objects, not
        # Betti numbers; the exact homology (test-side) says b0 = 1.
        simplices = [(a, b) for a in range(4) for b in range(a + 1, 4)]
        simplices += [(a, b) for a in range(4, 8) for b in range(a + 1, 8)]
        simplices += [(0, 4)]
        st = _from_simplices(8, simplices)
        _edge(st, 0, 4).setLength(cmath.sqrt(complex(1e-6)))
        betti0 = cob.ChainComplex.fromSpacetime(st).bettiNumbers()[0]
        self.assertEqual(betti0, 1)
        read = _tracker(st, groupingTolerance=1e-4).enumerateBands(
            list(range(8)), 0)
        self.assertEqual(read.fibers[0].rank(), 2)  # NOT b0

    def test_truncated_sparse_read_bounds_the_last_gap(self):
        # 600-ring at k=0 crosses the sparse threshold; requesting 12 pairs
        # covers the lowest bands only.  Covered bands are certified, the
        # read is marked truncated, and the last covered band's upper gap is
        # finite (bounded by the first uncovered Ritz value).
        st = _ring(600)
        read = _tracker(st, requestedEigenpairs=12,
                        **ANY_LOCALIZATION).enumerateBands(
                            list(range(600)), 0)
        self.assertEqual(read.solverPath, "sparse-block-self-adjoint")
        self.assertTrue(read.truncated)
        self.assertLess(len(read.coveredEigenvalues), 600)
        self.assertGreater(len(read.fibers), 0)
        for f in read.fibers:
            self.assertTrue(f.accepted())
        last = read.fibers[-1].certificate()
        self.assertTrue(np.isfinite(last.upperGap))
        # The GATED isolation is likewise shield-bounded, never a silently
        # generous +infinity on the uncovered side (#808).
        self.assertTrue(np.isfinite(last.nearestDiscardedSeparation))
        self.assertLessEqual(last.nearestDiscardedSeparation, last.upperGap)


# --------------------------------------------------------------------------- #
# property tests: gauge (in-band rotations), relabeling, input order
# --------------------------------------------------------------------------- #
class TestPropertyInvariance(unittest.TestCase):

    def test_in_band_rotation_changes_no_projector_observable(self):
        # The projector and every certificate quantity derived from it are
        # invariant under Phi -> Phi U, Psi -> Psi U for unitary U (the
        # eigenvector gauge).  Verified directly on the exactly degenerate
        # band.
        st = _triangle()
        read = _tracker(st).enumerateBands([0, 1, 2], 0)
        f = read.fibers[1]
        self.assertEqual(f.rank(), 2)
        Phi = np.asarray(f.rightFrame())
        Psi = np.asarray(f.leftFrame())
        W = np.diag(np.asarray(f.weightDiagonal()))
        P = np.asarray(f.projector())
        rng = np.random.default_rng(7)
        for _ in range(5):
            A = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
            U, _ = np.linalg.qr(A)
            P_rot = (Phi @ U) @ (Psi @ U).conj().T @ W
            self.assertLessEqual(np.abs(P_rot - P).max(), MACHINE)
            # Localization recomputed from the rotated frame equals the
            # certificate value (it reads only the projector diagonal).
            d = np.abs(np.diag(P_rot))
            ipr = float(np.sum((d / d.sum()) ** 2))
            self.assertLessEqual(abs(ipr - f.certificate().localization),
                                 MACHINE)

    def test_repeated_enumeration_identical_certificates(self):
        st = _triangle()
        tr = _tracker(st)
        a = tr.enumerateBands([0, 1, 2], 0)
        b = tr.enumerateBands([0, 1, 2], 0)
        self.assertEqual(_delta(a.toRecord(), b.toRecord()), 0.0)

    def test_relabeling_leaves_localization_gaps_and_rank_invariant(self):
        # RELABEL gate: rebuild the holed surface with randomly permuted
        # vertex ids; every certificate quantity must agree.
        rng = random.Random(11)
        faces = [
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
        ]
        windows = {(0, 5, 11), (2, 3, 6), (1, 8, 9)}
        holed = [f for f in faces if tuple(sorted(f)) not in windows]
        perm = dict(zip(range(12), rng.sample(range(500, 620), 12)))
        relabeled = [[perm[v] for v in f] for f in holed]
        st_a = tessera.Spacetime.fromVertexTuples(2, holed, 1.0, 0.0)
        st_b = tessera.Spacetime.fromVertexTuples(2, relabeled, 1.0, 0.0)
        for st in (st_a, st_b):
            st.materializeFacets()
        read_a = _tracker(st_a).enumerateBands(_all_vertices(st_a), 1)
        read_b = _tracker(st_b).enumerateBands(_all_vertices(st_b), 1)
        self.assertEqual(len(read_a.fibers), len(read_b.fibers))
        for fa, fb in zip(read_a.fibers, read_b.fibers):
            ca, cb = fa.certificate(), fb.certificate()
            self.assertEqual(fa.rank(), fb.rank())
            self.assertEqual(ca.accepted, cb.accepted)
            self.assertLessEqual(abs(ca.localization - cb.localization), 1e-9)
            for ga, gb in ((ca.lowerGap, cb.lowerGap),
                           (ca.upperGap, cb.upperGap)):
                if math.isinf(ga):
                    self.assertTrue(math.isinf(gb))
                else:
                    self.assertLessEqual(abs(ga - gb), 1e-9)
            np.testing.assert_allclose(
                np.array(fa.eigenvalues()), np.array(fb.eigenvalues()),
                atol=1e-9)
        # Cell supports map under the permutation (as SETS — no imposed
        # vertex order anywhere).
        cells_a = {tuple(sorted(perm[v] for v in cell))
                   for cell in read_a.cellVertices}
        cells_b = {tuple(sorted(cell)) for cell in read_b.cellVertices}
        self.assertEqual(cells_a, cells_b)

    def test_simplex_input_order_invariance(self):
        # Same complex, shuffled simplex insertion order: identical reads
        # (canonical ChainComplex cell order carries no input convention).
        simplices = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]
        st_a = _from_simplices(4, simplices)
        rng = random.Random(3)
        shuffled = list(simplices)
        rng.shuffle(shuffled)
        st_b = _from_simplices(4, shuffled)
        a = _tracker(st_a).enumerateBands([0, 1, 2, 3], 0)
        b = _tracker(st_b).enumerateBands([0, 1, 2, 3], 0)
        self.assertEqual(_delta(a.toRecord(), b.toRecord()), 0.0)

    def test_edge_orientation_flip_invariance_k0(self):
        # Reversing stored edge orientations leaves the Hermitian operator
        # (zero phases) and every certificate unchanged.
        st_a = _from_simplices(3, [(0, 1), (1, 2), (2, 0)])
        st_b = _from_simplices(3, [(1, 0), (2, 1), (0, 2)])
        a = _tracker(st_a).enumerateBands([0, 1, 2], 0)
        b = _tracker(st_b).enumerateBands([0, 1, 2], 0)
        self.assertEqual(_delta(a.toRecord(), b.toRecord()), 0.0)


# --------------------------------------------------------------------------- #
# tracking across frames and resolutions
# --------------------------------------------------------------------------- #
class TestTracking(unittest.TestCase):

    def test_identity_tracking(self):
        st = _triangle()
        tr = _tracker(st, **ANY_LOCALIZATION)
        a = tr.enumerateBands([0, 1, 2], 0)
        b = tr.enumerateBands([0, 1, 2], 0)
        matches = obs.SpectralFiberTracker.matchFibers(a.fibers, b.fibers)
        self.assertEqual(len(matches), len(a.fibers))
        for m in matches:
            self.assertEqual(m.fromIndex, m.toIndex)
            self.assertLessEqual(abs(m.overlap.subspaceOverlap - 1.0),
                                 MACHINE)
            self.assertTrue(m.certifiedContinuation)
            self.assertEqual(m.overlap.supportOverlap, 1.0)
            for angle in m.overlap.principalAngles:
                self.assertLessEqual(angle, 1e-7)

    def test_tracking_across_a_metric_move(self):
        # A small accepted metric change on one edge of a SIMPLE-spectrum
        # fixture (P3: {0, 1, 3}): the same bands continue — high subspace
        # overlap, certified continuation.
        st = _from_simplices(3, [(0, 1), (1, 2)])
        tr = _tracker(st, **ANY_LOCALIZATION)
        before = tr.enumerateBands([0, 1, 2], 0)
        _edge(st, 0, 1).setLength(cmath.sqrt(complex(1.02)))
        after = tr.enumerateBands([0, 1, 2], 0)
        matches = obs.SpectralFiberTracker.matchFibers(before.fibers,
                                                       after.fibers)
        self.assertEqual(len(matches), len(before.fibers))
        for m in matches:
            self.assertGreaterEqual(m.overlap.subspaceOverlap, 0.99)
            self.assertTrue(m.ranksEqual)
            self.assertTrue(m.certifiedContinuation)

    def test_degenerate_band_split_is_reported_not_relabeled(self):
        # Perturbing the triangle splits its exactly degenerate rank-2 band
        # into two rank-1 bands.  The tracker reports the overlap honestly
        # (each half carries cos^2 = 1 against the parent, scaled by the
        # rank mismatch to 1/2) and never certifies the continuation — a
        # changed multiplicity is a changed band, not a relabeled one.
        st = _triangle()
        tr = _tracker(st)
        before = tr.enumerateBands([0, 1, 2], 0)
        self.assertEqual(before.fibers[1].rank(), 2)
        _edge(st, 0, 1).setLength(cmath.sqrt(complex(1.5)))
        after = tr.enumerateBands([0, 1, 2], 0)
        self.assertEqual([f.rank() for f in after.fibers], [1, 1, 1])
        matches = obs.SpectralFiberTracker.matchFibers([before.fibers[1]],
                                                       after.fibers, 0.5)
        self.assertEqual(len(matches), 1)
        m = matches[0]
        self.assertFalse(m.ranksEqual)
        self.assertFalse(m.certifiedContinuation)
        self.assertLessEqual(abs(m.overlap.subspaceOverlap - 0.5), 1e-9)

    def test_tracking_across_resolutions_by_support_and_angles(self):
        # Different component supports (a resolution change): the shared
        # cells carry the principal-angle comparison; the support overlap is
        # the documented Jaccard.
        st = _two_triangles()
        tr = _tracker(st)
        whole = tr.enumerateBands([0, 1, 2, 10, 11, 12], 0)
        part = tr.enumerateBands([0, 1, 2], 0)
        matches = obs.SpectralFiberTracker.matchFibers(part.fibers,
                                                       whole.fibers, 0.1)
        self.assertGreater(len(matches), 0)
        m = matches[0]
        self.assertEqual(m.overlap.sharedCells, 3)
        self.assertEqual(m.overlap.supportOverlap, 0.5)  # 3 shared / 6 union

    def test_gap_closed_endpoint_never_certified_continuation(self):
        # Match a certified band against its gap-closed (uncertified)
        # continuation: reported, never certified — identity does not flip.
        healthy = _tracker(_triangle(alpha=1.0)).enumerateBands([0, 1, 2], 1)
        closed = _tracker(
            _triangle(alpha=math.sqrt(2.0))).enumerateBands([0, 1, 2], 1)
        matches = obs.SpectralFiberTracker.matchFibers(
            healthy.fibers, closed.fibers, 0.1)
        self.assertGreater(len(matches), 0)
        for m in matches:
            to_band = closed.fibers[m.toIndex]
            if not to_band.accepted():
                self.assertFalse(m.certifiedContinuation)
        self.assertTrue(any(not closed.fibers[m.toIndex].accepted()
                            for m in matches))

    def test_persistent_modularity_projector_hook_integration(self):
        # The #765 projector-overlap hook, fed by THIS ticket's fibers: the
        # documented integration point.  Component matching stays
        # support-based; the hook's value is reported per match.
        st = _two_triangles()
        pm = obs.PersistentModularity.fromSpacetime(
            st, obs.PersistentModularity.WeightMap.Unit)
        cfg = obs.PersistentModularityConfig()
        s = pm.discover(1.0, cfg)
        tr = _tracker(st)
        reads = {tuple(c.support): tr.enumerateBands(c.support, 0)
                 for c in s.components}
        by_id = {c.id.canonicalHash(): tuple(c.support)
                 for c in s.components}

        def hook(from_id, to_id):
            fa = reads[by_id[from_id.canonicalHash()]].fibers[0]
            fb = reads[by_id[to_id.canonicalHash()]].fibers[0]
            return obs.SpectralFiber.overlap(fa, fb).subspaceOverlap

        pm.setProjectorOverlapHook(hook)
        matches = pm.matchComponents(s.components, s.components)
        self.assertEqual(len(matches), 2)
        for m in matches:
            self.assertIsNotNone(m.projectorOverlap)
        pm.setProjectorOverlapHook(None)


# --------------------------------------------------------------------------- #
# caching: cold == incremental, touched-star invalidation, replay
# --------------------------------------------------------------------------- #
class TestCaching(unittest.TestCase):

    A = [0, 1, 2]
    B = [10, 11, 12]

    def test_cold_equals_cached_and_key_is_order_free(self):
        st = _two_triangles()
        tr = _tracker(st)
        cache = cob.AnalyticCache(st)
        cold = tr.enumerateBands(self.A, 0)
        first = tr.enumerateBandsCached(cache, self.A, 0)
        self.assertEqual(_delta(cold.toRecord(), first.toRecord()),
                         0.0)
        served = tr.enumerateBandsCached(cache, [2, 0, 1], 0)  # same set
        self.assertEqual(cache.hits, 1)
        self.assertEqual(
            _delta(cold.toRecord(), served.toRecord()), 0.0)

    def test_touched_star_invalidates_component_and_spares_sibling(self):
        st = _two_triangles()
        tr = _tracker(st)
        cache = cob.AnalyticCache(st)
        a0 = tr.enumerateBandsCached(cache, self.A, 0)
        b0 = tr.enumerateBandsCached(cache, self.B, 0)
        self.assertEqual(cache.size, 2)

        _edge(st, 0, 1).setLength(cmath.sqrt(complex(2.5)))
        star = cob.TouchedStar()
        star.addChangedEdge(0, 1)
        cache.publish(star)

        a1 = tr.enumerateBandsCached(cache, self.A, 0)  # recomputed
        b1 = tr.enumerateBandsCached(cache, self.B, 0)  # served
        self.assertEqual(cache.invalidations, 1)
        self.assertGreater(
            _delta(a0.toRecord(), a1.toRecord()), 0.0)
        self.assertEqual(
            _delta(b0.toRecord(), b1.toRecord()), 0.0)
        # The served sibling equals its cold recompute exactly.
        self.assertEqual(
            _delta(b1.toRecord(),
                             tr.enumerateBands(self.B, 0).toRecord()), 0.0)

    def test_unpublished_drift_is_fail_safe(self):
        # A mutation that was never published serves NOTHING — the cached
        # read can only be recomputed, never stale.
        st = _two_triangles()
        tr = _tracker(st)
        cache = cob.AnalyticCache(st)
        tr.enumerateBandsCached(cache, self.A, 0)
        _edge(st, 0, 1).setLength(cmath.sqrt(complex(3.0)))  # no publish
        fresh = tr.enumerateBandsCached(cache, self.A, 0)
        self.assertEqual(
            _delta(fresh.toRecord(),
                             tr.enumerateBands(self.A, 0).toRecord()), 0.0)

    def test_replay_disabled_cache_matches_incremental(self):
        st = _two_triangles()
        tr = _tracker(st)
        cache = cob.AnalyticCache(st)
        served = tr.enumerateBandsCached(cache, self.A, 0)
        cache.setEnabled(False)
        replay = tr.enumerateBandsCached(cache, self.A, 0)
        self.assertEqual(
            _delta(served.toRecord(), replay.toRecord()), 0.0)

    def test_cold_recompute_matches_incremental_tracking(self):
        # The acceptance bullet end-to-end: frame 0 -> accepted move on A ->
        # frame 1.  Incremental tracking (cache serves the untouched
        # sibling) must equal fully cold tracking, match for match.
        st = _two_triangles()
        tr = _tracker(st)
        cache = cob.AnalyticCache(st)
        frame0 = [tr.enumerateBandsCached(cache, s, 0)
                  for s in (self.A, self.B)]
        _edge(st, 0, 1).setLength(cmath.sqrt(complex(1.7)))
        star = cob.TouchedStar()
        star.addChangedEdge(0, 1)
        cache.publish(star)
        frame1_inc = [tr.enumerateBandsCached(cache, s, 0)
                      for s in (self.A, self.B)]
        self.assertEqual(cache.hits, 1)  # sibling B served incrementally
        frame1_cold = [tr.enumerateBands(s, 0) for s in (self.A, self.B)]
        for inc, cold in zip(frame1_inc, frame1_cold):
            self.assertEqual(
                _delta(inc.toRecord(), cold.toRecord()), 0.0)
        fibers0 = [f for r in frame0 for f in r.fibers]
        inc_m = obs.SpectralFiberTracker.matchFibers(
            fibers0, [f for r in frame1_inc for f in r.fibers])
        cold_m = obs.SpectralFiberTracker.matchFibers(
            fibers0, [f for r in frame1_cold for f in r.fibers])
        self.assertEqual(len(inc_m), len(cold_m))
        for mi, mc in zip(inc_m, cold_m):
            self.assertEqual((mi.fromIndex, mi.toIndex, mi.ranksEqual,
                              mi.certifiedContinuation),
                             (mc.fromIndex, mc.toIndex, mc.ranksEqual,
                              mc.certifiedContinuation))
            self.assertEqual(mi.overlap.subspaceOverlap,
                             mc.overlap.subspaceOverlap)


# --------------------------------------------------------------------------- #
# band windows for the response consumer; boundary discipline
# --------------------------------------------------------------------------- #
class TestBandWindowsAndBoundaries(unittest.TestCase):

    def test_accepted_windows_are_plain_data(self):
        st = _triangle()
        tr = _tracker(st)
        reads = [tr.enumerateBands([0, 1, 2], k) for k in (0, 1)]
        windows = obs.SpectralFiberTracker.acceptedWindows(reads)
        accepted = [f for r in reads for f in r.fibers if f.accepted()]
        self.assertEqual(len(windows), len(accepted))
        for w, f in zip(windows, accepted):
            c = f.certificate()
            self.assertEqual(w.degree, c.degree)
            self.assertEqual(w.rank, c.rank)
            self.assertEqual(w.frequencyLower, c.frequencyLower)
            self.assertEqual(w.frequencyUpper, c.frequencyUpper)
            self.assertTrue(w.certificate.accepted)

    def test_uncertified_bands_never_reach_the_window_list(self):
        st = _triangle(alpha=math.sqrt(2.0))
        read = _tracker(st).enumerateBands([0, 1, 2], 1)
        windows = obs.SpectralFiberTracker.acceptedWindows([read])
        self.assertLess(len(windows), len(read.fibers))
        for w in windows:
            self.assertTrue(w.certificate.certificate.holds())

    def test_no_rank_is_ever_requested(self):
        # The config carries no target rank; a 3-fold degeneracy is
        # enumerated exactly like any other multiplicity (three disjoint
        # triangles -> rank-3 kernel band; four -> rank 4).
        cfg = obs.SpectralFiberConfig()
        self.assertFalse(any("rank" in name.lower()
                             for name in dir(cfg)
                             if not name.startswith("_")))
        for n_tri, expect in ((3, 3), (4, 4)):
            simplices = []
            for t in range(n_tri):
                base = 10 * t
                simplices += [(base, base + 1), (base + 1, base + 2),
                              (base, base + 2)]
            ids = sorted({v for s in simplices for v in s})
            remap = {v: i for i, v in enumerate(ids)}
            st = _from_simplices(len(ids),
                                 [tuple(remap[v] for v in s)
                                  for s in simplices],
                                 ids=ids)
            read = _tracker(st).enumerateBands(ids, 0)
            self.assertEqual(read.fibers[0].rank(), expect)

    def test_degeneracy_reported_without_taste_or_flavor_labels(self):
        st = _triangle()
        read = _tracker(st).enumerateBands([0, 1, 2], 0)
        degenerate = read.fibers[1]
        self.assertGreaterEqual(degenerate.rank(), 2)
        text = degenerate.certificate().describe().lower()
        for banned in BANNED_LABELS:
            self.assertNotIn(banned, text)
        # And nowhere in the serialized record either.
        def walk(node):
            if isinstance(node, dict):
                for k, v in node.items():
                    self.assertFalse(
                        any(bad in str(k).lower() for bad in BANNED_LABELS))
                    walk(v)
            elif isinstance(node, list):
                for v in node:
                    walk(v)
            elif isinstance(node, str):
                self.assertFalse(
                    any(bad in node.lower() for bad in BANNED_LABELS))
        walk(read.toRecord())

    def test_read_only_observable(self):
        # The first ChainComplex/facet materialization on a hand-built
        # complex is the complex completing its own skeleton (the
        # pre-existing lazy materializeFacets idiom, revision-stamped once).
        # After that one-time completion, the fiber reads themselves must
        # leave the geometry, the revision, and every count untouched.
        st = _triangle(alpha=2.0)
        st.materializeFacets()
        cob.ChainComplex.fromSpacetime(st)  # one-time lazy face closure
        rev = st.metricRevisionKey()
        n_v = st.getVertexCount()
        n_e = st.getEdgeList().size()
        n_s = st.getTopSimplexCount()
        lengths = [e.getLength() for e in st.getEdgeList().toVector()]
        tr = _tracker(st, crossValidateDense=True)
        tr.enumerateBands([0, 1, 2], 0)
        tr.enumerateBands([0, 1, 2], 1)
        tr.enumerateBands([0, 1], 1)
        self.assertEqual(st.metricRevisionKey(), rev)
        self.assertEqual(st.getVertexCount(), n_v)
        self.assertEqual(st.getEdgeList().size(), n_e)
        self.assertEqual(st.getTopSimplexCount(), n_s)
        self.assertEqual(
            [e.getLength() for e in st.getEdgeList().toVector()], lengths)

    def test_errors_and_empty_reads(self):
        st = _triangle()
        tr = _tracker(st)
        with self.assertRaises(ValueError):
            tr.enumerateBands([0, 1, 2], -1)
        empty = tr.enumerateBands([], 0)
        self.assertEqual(empty.dimension, 0)
        self.assertEqual(len(empty.fibers), 0)
        unknown = tr.enumerateBands([777, 888], 0)
        self.assertEqual(unknown.dimension, 0)
        above = tr.enumerateBands([0, 1, 2], 5)  # above the top dimension
        self.assertEqual(above.dimension, 0)

    def test_enumerate_on_components_covers_configured_degrees(self):
        st = _two_triangles()
        pm = obs.PersistentModularity.fromSpacetime(
            st, obs.PersistentModularity.WeightMap.Unit)
        s = pm.discover(1.0, obs.PersistentModularityConfig())
        cfg = obs.SpectralFiberConfig()
        cfg.degrees = [0, 1]
        tr = obs.SpectralFiberTracker(st, cfg)
        reads = tr.enumerateOnComponents(s.components)
        self.assertEqual(len(reads), 2 * 2)
        self.assertEqual([r.degree for r in reads], [0, 1, 0, 1])
        for r in reads:
            self.assertEqual(r.dimension, 3)
            self.assertEqual(sum(f.rank() for f in r.fibers), 3)


# --------------------------------------------------------------------------- #
# checkpoint serialization
# --------------------------------------------------------------------------- #
class TestSerialization(unittest.TestCase):

    def _reads(self):
        out = [
            _tracker(_triangle()).enumerateBands([0, 1, 2], 0),
            _tracker(_triangle(alpha=2.0)).enumerateBands([0, 1, 2], 1),
        ]
        st = _triangle(alpha=1.0)
        out.append(obs.SpectralFiberTracker(
            st, obs.SpectralFiberConfig(),
            cob.HodgeWeightConvention.Content).enumerateBands([0, 1, 2], 1))
        return out

    def test_fiber_round_trip_every_regime(self):
        for read in self._reads():
            for f in read.fibers:
                rec = f.toRecord()
                back = obs.SpectralFiber.fromRecord(rec)
                # NaN-aware every-channel gate: two NaNs agree, any numeric
                # drift is a flagged channel.
                self.assertEqual(_delta(rec, back.toRecord()), 0.0)
                np.testing.assert_array_equal(np.asarray(f.projector()),
                                              np.asarray(back.projector()))
                self.assertEqual(f.rank(), back.rank())
                self.assertEqual(f.accepted(), back.accepted())

    def test_read_round_trip(self):
        for read in self._reads():
            rec = read.toRecord()
            back = obs.ComponentBandRead.fromRecord(rec)
            self.assertEqual(_delta(rec, back.toRecord()), 0.0)

    def test_unknown_schema_version_rejected(self):
        read = self._reads()[0]
        bad = dict(read.toRecord())
        bad["schema_version"] = 99
        with self.assertRaises(ValueError):
            obs.ComponentBandRead.fromRecord(bad)
        fiber_bad = dict(read.fibers[0].toRecord())
        fiber_bad["schema_version"] = 99
        with self.assertRaises(ValueError):
            obs.SpectralFiber.fromRecord(fiber_bad)

    def test_wrong_record_type_rejected(self):
        read = self._reads()[0]
        rec = dict(read.toRecord())
        with self.assertRaises(ValueError):
            obs.SpectralFiber.fromRecord(rec)  # a read record, not a fiber

    def test_certificate_grades_survive_the_round_trip(self):
        healthy = self._reads()[0]
        closed = _tracker(
            _triangle(alpha=math.sqrt(2.0))).enumerateBands([0, 1, 2], 1)
        for read in (healthy, closed):
            for f in read.fibers:
                back = obs.SpectralFiber.fromRecord(f.toRecord())
                self.assertEqual(back.certificate().certificate.grade,
                                 f.certificate().certificate.grade)
                self.assertEqual(back.certificate().certificate.holds(),
                                 f.certificate().certificate.holds())


# --------------------------------------------------------------------------- #
# sparse block solve vs the dense reference (the crossover comparison)
# --------------------------------------------------------------------------- #
class TestSparseVsDense(unittest.TestCase):

    N = 600  # above the default crossover (512)

    def test_sparse_matches_analytic_ring_spectrum(self):
        st = _ring(self.N)
        read = _tracker(st, requestedEigenpairs=12).enumerateBands(
            list(range(self.N)), 0)
        self.assertEqual(read.solverPath, "sparse-block-self-adjoint")
        lam = np.array(read.coveredEigenvalues).real
        exact = np.sort(
            2.0 - 2.0 * np.cos(2.0 * np.pi * np.arange(self.N) / self.N))
        self.assertLessEqual(np.abs(lam - exact[:len(lam)]).max(), SOLVER)

    def test_sparse_matches_dense_path_and_projectors_align(self):
        st = _ring(self.N)
        sparse = _tracker(st, requestedEigenpairs=12).enumerateBands(
            list(range(self.N)), 0)
        dense = _tracker(st, denseCrossover=10 ** 6).enumerateBands(
            list(range(self.N)), 0)
        self.assertEqual(dense.solverPath, "dense-self-adjoint")
        n_cov = len(sparse.coveredEigenvalues)
        np.testing.assert_allclose(
            np.array(sparse.coveredEigenvalues).real,
            np.array(dense.coveredEigenvalues).real[:n_cov], atol=SOLVER)
        # Cold (dense) and sparse tracking of the same bands agree:
        # principal angles ~ 0 between corresponding projected subspaces.
        matches = obs.SpectralFiberTracker.matchFibers(
            sparse.fibers, dense.fibers)
        self.assertEqual(len(matches), len(sparse.fibers))
        for m in matches:
            self.assertTrue(m.ranksEqual)
            self.assertGreaterEqual(m.overlap.subspaceOverlap, 1.0 - 1e-8)

    def test_sparse_path_is_deterministic(self):
        st = _ring(self.N)
        a = _tracker(st, requestedEigenpairs=8).enumerateBands(
            list(range(self.N)), 0)
        b = _tracker(st, requestedEigenpairs=8).enumerateBands(
            list(range(self.N)), 0)
        self.assertEqual(_delta(a.toRecord(), b.toRecord()), 0.0)

    def test_dense_cross_validation_records_reference_error(self):
        st = _triangle()
        read = _tracker(st, crossValidateDense=True).enumerateBands(
            [0, 1, 2], 0)
        err = read.solveCertificate.denseReferenceError
        self.assertTrue(np.isfinite(err))
        self.assertLessEqual(err, 1e-12)
        for f in read.fibers:
            if f.accepted():
                self.assertTrue(
                    np.isfinite(f.certificate().certificate
                                .denseReferenceError))
        # Without the flag the field stays unmeasured (NaN), never zero.
        plain = _tracker(st).enumerateBands([0, 1, 2], 0)
        self.assertTrue(math.isnan(
            plain.solveCertificate.denseReferenceError))


# --------------------------------------------------------------------------- #
# #808 negative controls: the acceptance conjuncts the whitepaper NAMES are
# enforced on the quantity it names
# --------------------------------------------------------------------------- #
class TestLocalizationConjunct(unittest.TestCase):
    """The whitepaper lists "a localized spectral projector with stable rank"
    as a conjunct of FIBER ACCEPTANCE.  It is enforced there (#808), on the
    rank-normalized localization excess."""

    def test_perfectly_delocalized_band_is_rejected(self):
        # C12 is vertex-transitive: every band projector has a UNIFORM
        # diagonal, so n_eff = n and the excess is exactly 1 -- perfectly
        # delocalized.  Every OTHER conjunct passes with room to spare, so
        # localization is the only reason each band is uncertified.
        read = _tracker(_ring(12)).enumerateBands(list(range(12)), 0)
        self.assertEqual(len(read.fibers), 7)
        for f in read.fibers:
            c = f.certificate()
            self.assertAlmostEqual(c.localization, 1.0 / 12.0, delta=MACHINE)
            self.assertAlmostEqual(c.localizationSupportFraction, 1.0,
                                   delta=MACHINE)
            self.assertAlmostEqual(c.localizationExcess, 1.0, delta=MACHINE)
            # the other conjuncts hold:
            self.assertGreater(c.nearestDiscardedSeparation, 0.26)
            self.assertLess(c.gramDefect, 1e-12)
            self.assertLess(c.eigenResidual, 1e-12)
            self.assertLess(c.projectorResidual, 1e-12)
            self.assertLess(c.projectorNorm, 1.0 + 1e-12)
            # ... and the band is REJECTED anyway.
            self.assertFalse(c.accepted)
            self.assertFalse(c.certificate.holds())

    def test_the_permissive_cap_restores_the_pre_gate_acceptance(self):
        read = _tracker(_ring(12), **ANY_LOCALIZATION).enumerateBands(
            list(range(12)), 0)
        self.assertTrue(all(f.accepted() for f in read.fibers))

    def test_a_localized_band_certifies(self):
        # The jittered holed surface breaks the transitivity: about half its
        # degree-one bands sit below the default cap and certify, and a band
        # is accepted EXACTLY when its excess clears the cap -- the gate
        # discriminates, it does not reject everything.
        st, _, _, _ = holed_surface(degree=1, jitter=True)
        read = _tracker(st).enumerateBands(_all_vertices(st), 1)
        accepted = [f for f in read.fibers if f.accepted()]
        self.assertGreater(len(accepted), 5)
        self.assertLess(len(accepted), len(read.fibers))
        cap = obs.SpectralFiberConfig().maxLocalizationExcess
        self.assertEqual(cap, 0.5)
        for f in read.fibers:
            c = f.certificate()
            self.assertEqual(c.accepted, c.localizationExcess <= cap)

    def test_excess_is_rank_normalized_and_relabeling_invariant(self):
        # The bare IPR of a rank-r band cannot exceed 1/r, so the RAW
        # fraction is not comparable across ranks; the excess is.  Both are
        # projector reads, so both survive a vertex relabeling.
        st, _, _, _ = holed_surface(degree=1, jitter=True)
        ids = _all_vertices(st)
        base = _tracker(st).enumerateBands(ids, 1)
        for f in base.fibers:
            c = f.certificate()
            self.assertGreaterEqual(c.localization,
                                    1.0 / base.dimension - 1e-12)
            self.assertLessEqual(c.localization, 1.0 / c.rank + 1e-12)
            self.assertGreaterEqual(c.localizationExcess, 0.0)
            self.assertLessEqual(c.localizationExcess, 1.0)
        shifted = _tracker(st).enumerateBands(list(reversed(ids)), 1)
        for a, b in zip(base.fibers, shifted.fibers):
            self.assertAlmostEqual(a.certificate().localizationExcess,
                                   b.certificate().localizationExcess,
                                   delta=MACHINE)

    def test_a_full_space_band_is_not_called_delocalized(self):
        # A band spanning the WHOLE operator (n == rank) leaves no room to
        # be localized in: the excess is 0 by definition, never 1.  The
        # single isolated vertex is the smallest such case.
        read = _tracker(_triangle()).enumerateBands([0], 0)
        c = read.fibers[0].certificate()
        self.assertEqual(read.dimension, 1)
        self.assertEqual(c.rank, 1)
        self.assertEqual(c.localizationSupportFraction, 1.0)
        self.assertEqual(c.localizationExcess, 0.0)
        self.assertTrue(c.accepted)


class TestBandSeparationIsMeasuredInThePlane(unittest.TestCase):
    """The whitepaper's conjunct is "a nonzero band gap separating it from
    discarded modes".  With a genuinely complex spectrum the (Re, Im)-sorted
    neighbour need not be the nearest eigenvalue in the plane (#808)."""

    def _content_read(self, alpha, **cfg_kwargs):
        cfg = obs.SpectralFiberConfig()
        cfg.maxLocalizationExcess = 1.0     # subject: the isolation rule
        for k, v in cfg_kwargs.items():
            setattr(cfg, k, v)
        tracker = obs.SpectralFiberTracker(
            _triangle(alpha=alpha), cfg, cob.HodgeWeightConvention.Content)
        return tracker.enumerateBands([0, 1, 2], 1)

    def test_nearest_eigenvalue_is_not_the_sort_adjacent_one(self):
        # Content weights at alpha = 1/2: spec(L_1) = {0, 1 - 4i, 3}.
        # Sorted by (Re, Im) the neighbour of the band at 3 is 1 - 4i, at
        # distance |2 + 4i| = sqrt(20) -- but the NEAREST discarded
        # eigenvalue in the plane is 0, at distance 3.
        read = self._content_read(0.5)
        self.assertEqual(read.regime, cob.CertificateRegime.NonNormal)
        top = max(read.fibers, key=lambda f: f.eigenvalues()[0].real)
        c = top.certificate()
        self.assertAlmostEqual(c.lowerGap, math.sqrt(20.0), delta=1e-9)
        self.assertAlmostEqual(c.nearestDiscardedSeparation, 3.0, delta=1e-9)
        self.assertLess(c.nearestDiscardedSeparation, c.lowerGap)
        bottom = min(read.fibers, key=lambda f: f.eigenvalues()[0].real)
        cb = bottom.certificate()
        self.assertAlmostEqual(cb.upperGap, math.sqrt(17.0), delta=1e-9)
        self.assertAlmostEqual(cb.nearestDiscardedSeparation, 3.0, delta=1e-9)

    def test_the_isolation_gate_reads_the_true_separation(self):
        # scale = max|lambda| = |1 - 4i| = sqrt(17).  A floor of 0.9 * scale
        # = 3.71 sits BETWEEN the true separation (3) and the sort-order gap
        # (4.47): the outer bands are rejected on the separation the paper
        # names, where the sort-order rule would have certified them.
        loose = self._content_read(0.5, minRelativeGap=0.5)
        self.assertTrue(all(f.accepted() for f in loose.fibers))
        tight = self._content_read(0.5, minRelativeGap=0.9)
        verdicts = {round(f.eigenvalues()[0].real, 6): f.accepted()
                    for f in tight.fibers}
        self.assertFalse(verdicts[0.0])
        self.assertFalse(verdicts[3.0])
        self.assertTrue(verdicts[1.0])   # its own nearest neighbour is 4.12


class TestFrameConditionIsNotTheProjectorNorm(unittest.TestCase):
    """The whitepaper asks for the FRAME condition number in the non-normal
    regime; the code reported the projector spectral norm under that name.
    Both are now measured and reported separately (#808)."""

    def test_the_two_conditionings_are_separate_quantities(self):
        # The defective Krein crossing at alpha = sqrt(2): the near-zero
        # rank-2 band's frames are nearly parallel, so their Riesz
        # conditioning explodes (1e8) while the PROJECTOR norm stays near 1.
        st = _triangle(alpha=math.sqrt(2.0))
        read = _tracker(st, groupingTolerance=1e-6,
                        **ANY_LOCALIZATION).enumerateBands([0, 1, 2], 1)
        degenerate = max(read.fibers, key=lambda f: f.rank())
        c = degenerate.certificate()
        self.assertEqual(c.rank, 2)
        self.assertLess(c.projectorNorm, 2.0)
        self.assertGreater(c.frameConditionNumber, 1e6)
        self.assertNotAlmostEqual(c.projectorNorm, c.frameConditionNumber,
                                  delta=1.0)

    def test_a_w_orthonormal_frame_is_perfectly_conditioned(self):
        # On the self-adjoint path Phi^dagger W Phi = I exactly, so the
        # frame condition number is 1 -- the value the substitute happened
        # to coincide with, and the reason the divergence was invisible.
        read = _tracker(_ring(12), **ANY_LOCALIZATION).enumerateBands(
            list(range(12)), 0)
        for f in read.fibers:
            c = f.certificate()
            self.assertAlmostEqual(c.frameConditionNumber, 1.0, delta=1e-9)
            self.assertAlmostEqual(c.projectorNorm, 1.0, delta=1e-9)

    def test_both_travel_through_the_checkpoint(self):
        st = _triangle(alpha=math.sqrt(2.0))
        read = _tracker(st, groupingTolerance=1e-6,
                        **ANY_LOCALIZATION).enumerateBands([0, 1, 2], 1)
        for f in read.fibers:
            rec = f.toRecord()
            self.assertEqual(rec["schema_version"], 2)
            back = obs.SpectralFiber.fromRecord(rec).certificate()
            self.assertEqual(back.projectorNorm, f.certificate().projectorNorm)
            self.assertEqual(back.frameConditionNumber,
                             f.certificate().frameConditionNumber)
            self.assertEqual(back.nearestDiscardedSeparation,
                             f.certificate().nearestDiscardedSeparation)

    def test_a_schema_one_record_leaves_the_new_leaves_unknown(self):
        # Backwards compatibility: schema 1 carried `condition_number`, the
        # projector norm.  It still reads, the projector norm survives, and
        # the quantities it never measured are NaN -- never zero.
        st = _triangle()
        fiber = _tracker(st, **ANY_LOCALIZATION).enumerateBands(
            [0, 1, 2], 0).fibers[0]
        rec = dict(fiber.toRecord())
        cert = dict(rec["certificate"])
        cert["condition_number"] = cert.pop("projector_norm")
        for key in ("frame_condition_number", "nearest_discarded_separation",
                    "localization_support_fraction", "localization_excess"):
            cert.pop(key)
        rec["certificate"] = cert
        rec["schema_version"] = 1
        back = obs.SpectralFiber.fromRecord(rec).certificate()
        self.assertEqual(back.projectorNorm,
                         fiber.certificate().projectorNorm)
        for value in (back.frameConditionNumber,
                      back.nearestDiscardedSeparation,
                      back.localizationSupportFraction,
                      back.localizationExcess):
            self.assertTrue(math.isnan(value))


if __name__ == "__main__":
    unittest.main()
