# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""#1193 — fiber acceptance on a certified Riesz band.

Sections 3 and 5 of the whitepaper accept a fiber on a band selected by a
closed contour in the complex spectral plane, with nonzero separation and a
CONTROLLED RESOLVENT separating it from the discarded modes, taken at a
reported Lorentzian rotation ``eps_L > 0`` on the Kontsevich-Segal allowable
side.  The exact Riesz projector is

    P_C = (1 / 2 pi i) oint_{gamma_C} (zeta I - h_C)^-1 dzeta ,

so the contour, the spectral separation and the resolvent bound are all
certificates OF that contour and mean nothing without it.

These tests pin that a contour-selected band carries the contour it was
selected by and the measured resolvent on it; that the reported bound really
bounds the projector it certifies; that a band whose resolvent exceeds the
declared cap is REPORTED and REFUSED rather than silently accepted; that a
real Lorentzian instance, which sits exactly on the Kontsevich-Segal boundary
with a zero margin, is never accepted alone while its rotated family member at
``eps_L > 0`` is; and that the register decides the contour conjunct on those
measurements and NAMES each half of it separately.
"""

import cmath
import math
import unittest

import numpy as np
import pytest

import tessera
from tessera import chainhodge as ch
from tests.chainhodge._fixtures import edges, torus33, torus33_causal_types

obs = tessera.observables
cob = tessera.cobordism

Whitney = cob.HodgeMetricSource.WhitneyPencil
Pencil = cob.CertificateRegime.ComplexSymmetricPencil
NAN = float("nan")


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
def _spacetime_from(K, s):
    """A spacetime whose edge lengths square to the supplied squared lengths."""
    cells = [list(int(v) for v in t) for t in K.orientedTopSimplices()]
    st = tessera.Spacetime.fromVertexTuples(2, cells, 1.0, 0.0)
    table = dict(zip(edges(K), s))
    for e in st.getEdgeList().toVector():
        a, b = e.getSource().getId(), e.getTarget().getId()
        e.setLength(cmath.sqrt(table[(min(a, b), max(a, b))]))
        e.setPhase(0.0)
    st.materializeFacets()
    return st


def _lorentzian_torus(eps):
    """The 3x3 torus with its timelike squared lengths rotated by e^{-2 i eps}.

    The causal types are DECLARED by the fixture (`torus33_causal_types`) and
    the rotation is the library's own `LorentzianFamily.rotate`: nothing here
    infers a causal type from a squared length.
    """
    K, s = torus33()
    return _spacetime_from(K, ch.LorentzianFamily.rotate(
        s, torus33_causal_types(K), eps)), K


def _euclidean_torus():
    """The same torus with every squared length real and positive."""
    K, s = torus33()
    return _spacetime_from(K, [abs(v) for v in s]), K


def _config(**overrides):
    cfg = obs.SpectralFiberConfig()
    cfg.degrees = [1]
    # The torus is translation-symmetric, so every band is delocalized by
    # construction: the localization conjunct is opened to "any measured
    # localization" and the contour conjuncts decide (the pencil-regime suite
    # opens it the same way and for the same reason).
    cfg.maxLocalizationExcess = 1.0
    for name, value in overrides.items():
        setattr(cfg, name, value)
    return cfg


def _bands(st, K, cfg):
    support = [int(v[0]) for v in K.kSimplexVertices(0)]
    return obs.SpectralFiberTracker(st, cfg, Whitney).enumerateBands(support, 1)


def _harmonic(read):
    """The zero band of the torus: rank two, the one the suite tracks."""
    zero = [f for f in read.fibers
            if all(abs(z) < 1e-8 for z in f.eigenvalues())]
    assert len(zero) == 1, "fixture must carry exactly one harmonic band"
    return zero[0]


# --------------------------------------------------------------------------- #
# the contour certificate
# --------------------------------------------------------------------------- #
class TestContourCertificate:

    def test_a_contour_selected_band_carries_the_contour(self):
        """The band reports the contour it was selected by: description, node
        count, centre and radius."""
        st, K = _euclidean_torus()
        cfg = _config(contourNodes=48)
        band = _harmonic(_bands(st, K, cfg))
        cert = band.certificate()
        assert cert.certificate.regime == Pencil
        assert cert.contour.startswith("circle")
        assert cert.contourNodeCount == 48
        assert math.isfinite(cert.contourRadius) and cert.contourRadius > 0.0
        assert math.isfinite(cert.contourCenter.real)
        assert math.isfinite(cert.contourCenter.imag)
        assert cert.describe().count("contour") >= 1

    def test_the_resolvent_bound_is_the_riesz_estimate(self):
        """resolventBound is (|gamma_C| / 2 pi) max ||(zeta I - h)^-1||, i.e.
        radius * resolventMax for the circular contour, and it really does
        bound the projector it certifies."""
        st, K = _euclidean_torus()
        band = _harmonic(_bands(st, K, _config()))
        cert = band.certificate()
        assert math.isfinite(cert.resolventMax) and cert.resolventMax > 0.0
        assert cert.resolventBound == pytest.approx(
            cert.contourRadius * cert.resolventMax, rel=1e-12)
        projector_norm = np.linalg.norm(np.asarray(band.projector()), 2)
        assert cert.resolventBound >= projector_norm - 1e-9, (
            "the Riesz estimate must bound the projector norm it certifies")

    def test_a_blown_up_resolvent_is_reported_and_refused(self):
        """Tightening the cap below the measured bound refuses the band by
        acceptance; the band, its contour and its bound are still REPORTED."""
        st, K = _euclidean_torus()
        measured = _harmonic(_bands(st, K, _config())).certificate()
        assert measured.accepted, measured.describe()
        tightened = _harmonic(_bands(st, K, _config(
            resolventBoundCap=0.5 * measured.resolventBound))).certificate()
        assert not tightened.accepted
        assert tightened.resolventBound == pytest.approx(
            measured.resolventBound, rel=1e-12)
        assert tightened.contour == measured.contour
        assert not tightened.certificate.holds()

    def test_a_band_with_no_contour_claims_none(self):
        """The diagonal-weights path groups by the sort-and-gap rule and draws
        no contour: it reports no contour and no resolvent, never a zero.

        The metric source is NAMED here rather than defaulted, since the
        default is the Whitney pencil and this read is about the other path.
        """
        st, K = _euclidean_torus()
        support = [int(v[0]) for v in K.kSimplexVertices(0)]
        read = obs.SpectralFiberTracker(
            st, _config(),
            cob.HodgeMetricSource.DiagonalWeights).enumerateBands(support, 1)
        assert read.solverPath != "pencil-riesz"
        assert read.fibers
        for fiber in read.fibers:
            cert = fiber.certificate()
            assert cert.contour == ""
            assert cert.contourNodeCount == 0
            assert math.isnan(cert.resolventBound)
            assert math.isnan(cert.resolventMax)
            assert math.isnan(cert.allowabilityMargin)

    def test_the_contour_certificate_round_trips(self):
        st, K = _euclidean_torus()
        band = _harmonic(_bands(st, K, _config()))
        cert = band.certificate()
        back = obs.SpectralFiber.fromRecord(band.toRecord()).certificate()
        assert back.contour == cert.contour
        assert back.contourNodeCount == cert.contourNodeCount
        assert back.resolventMax == pytest.approx(cert.resolventMax)
        assert back.resolventBound == pytest.approx(cert.resolventBound)
        assert back.allowabilityMargin == pytest.approx(cert.allowabilityMargin)
        assert back.allowable == cert.allowable


# --------------------------------------------------------------------------- #
# the Kontsevich-Segal margin and the declared rotation
# --------------------------------------------------------------------------- #
class TestAllowabilityMargin:

    def test_a_euclidean_instance_is_allowable_with_margin_pi(self):
        st, K = _euclidean_torus()
        cert = _harmonic(_bands(st, K, _config())).certificate()
        assert cert.allowable
        assert cert.allowabilityMargin == pytest.approx(math.pi, rel=1e-9)
        assert cert.accepted, cert.describe()

    def test_a_rotated_lorentzian_instance_is_accepted(self):
        """eps_L > 0 puts the instance strictly inside the allowable side."""
        st, K = _lorentzian_torus(0.1)
        cert = _harmonic(_bands(st, K, _config())).certificate()
        assert cert.allowable
        assert cert.allowabilityMargin > 0.0
        assert cert.accepted, cert.describe()

    def test_a_real_lorentzian_instance_is_reported_but_never_accepted(self):
        """At eps_L = 0 the instance sits exactly on the Kontsevich-Segal
        boundary: the band is reported with its contour, its separation and
        its margin, and it is NOT accepted on its own."""
        st, K = _lorentzian_torus(0.0)
        band = _harmonic(_bands(st, K, _config()))
        cert = band.certificate()
        assert not cert.allowable
        assert cert.allowabilityMargin <= 0.0
        assert not cert.accepted
        # reported, not discarded: the gap certificate travels with it
        assert cert.contour.startswith("circle")
        assert math.isfinite(cert.nearestDiscardedSeparation)
        assert band.rank() == 2

    def test_the_declared_rotation_travels_on_every_band(self):
        st, K = _lorentzian_torus(0.1)
        read = _bands(st, K, _config(lorentzianEpsilon=0.1))
        assert read.fibers
        for fiber in read.fibers:
            assert fiber.certificate().lorentzianEpsilon == pytest.approx(0.1)

    def test_a_declared_zero_rotation_is_not_accepted(self):
        """A complex DECLARED Lorentzian must be read at eps_L > 0, even when
        its squared lengths happen to be allowable."""
        st, K = _euclidean_torus()
        cert = _harmonic(_bands(st, K, _config(lorentzianEpsilon=0.0))
                         ).certificate()
        assert cert.allowable, "the fixture's instance is allowable"
        assert not cert.accepted, "a declared eps_L = 0 is never accepted"

    def test_an_undeclared_complex_reports_the_rotation_unmeasured(self):
        st, K = _euclidean_torus()
        cert = _harmonic(_bands(st, K, _config())).certificate()
        assert math.isnan(cert.lorentzianEpsilon)


# --------------------------------------------------------------------------- #
# the register's contour conjunct
# --------------------------------------------------------------------------- #
def _split(name, values, record):
    arr = np.asarray(values, dtype=complex).reshape(-1)
    record[name + "_re"] = [float(v.real) for v in arr]
    record[name + "_im"] = [float(v.imag) for v in arr]


def _contour_fiber(*, contour="circle c=(1,0), r=0.5, N=32", nodes=32,
                   resolvent_bound=2.0, margin=math.pi, allowable=True,
                   epsilon=NAN, accepted=True):
    """A rank-1 degree-0 band whose contour certificate is driven directly.

    Rehydrated from its record (the replay route) so each certificate field
    moves independently: the register ASSEMBLES what the detector measured and
    re-deriving a band here would test the detector instead.
    """
    record = {
        "schema_version": 3, "record_type": "spectral_fiber",
        "cells": [[0], [1], [2]], "rows": 3, "rank": 1,
        "certificate": {
            "degree": 0, "rank": 1,
            "lower_gap": 1.0, "upper_gap": 1.0,
            "nearest_discarded_separation": 1.0,
            "localization": NAN, "localization_support_fraction": NAN,
            "localization_excess": 0.0,
            "projector_residual": 1e-16, "eigen_residual": 1e-16,
            "left_residual": 1e-16, "gram_defect": 0.0,
            "projector_norm": 1.0, "frame_condition_number": 1.0,
            "positive_signature": 1, "negative_signature": 0,
            "frequency_lower": 0.0, "frequency_upper": 2.0,
            "self_adjoint": True, "accepted": bool(accepted),
            "contour": contour, "contour_node_count": int(nodes),
            "contour_center_re": 1.0, "contour_center_im": 0.0,
            "contour_radius": 0.5,
            "resolvent_max": 4.0,
            "resolvent_bound": float(resolvent_bound),
            "allowable": bool(allowable),
            "allowability_margin": float(margin),
            "lorentzian_epsilon": float(epsilon),
            "certificate": {"grade": "certified-numerical",
                            "domain": "band-window",
                            "regime": "positive-semidefinite",
                            "residual": 1e-16, "conditioning": 1.0,
                            "tolerance": 1e-9,
                            "dense_reference_error": NAN}}}
    right = np.zeros((3, 1), dtype=complex)
    right[0, 0] = 1.0
    _split("eigenvalues", [1.0 + 0j], record)
    _split("right_frame", right, record)
    _split("left_frame", right, record)
    _split("weights", np.ones(3, dtype=complex), record)
    return obs.SpectralFiber.fromRecord(record)


def _clique_edges(vertices, src, tgt):
    for i in range(len(vertices)):
        for j in range(i + 1, len(vertices)):
            src.append(vertices[i])
            tgt.append(vertices[j])


def _track(frames=3):
    """A FrameTrack the library itself produced (never hand-built)."""
    cfg = tessera.PersistentModularityConfig()
    cfg.resolutions = [1.0]
    cfg.restarts = 4
    cfg.baseSeed = 0
    cfg.overlapThreshold = 0.0
    out = []
    for _ in range(frames):
        src, tgt = [], []
        _clique_edges(list(range(6)), src, tgt)
        _clique_edges(list(range(10, 16)), src, tgt)
        graph = tessera.PersistentModularity.fromWeightedEdges(
            src, tgt, [1.0] * len(src))
        out.append(graph.discover(1.0, cfg).components)
    src, tgt = [], []
    _clique_edges(list(range(6)), src, tgt)
    _clique_edges(list(range(10, 16)), src, tgt)
    graph = tessera.PersistentModularity.fromWeightedEdges(
        src, tgt, [1.0] * len(src))
    tracks = graph.trackAcrossFrames(out, 0.0)
    assert tracks, "fixture produced no frame track"
    return max(tracks, key=lambda t: t.frames)


def _transport(leakage=0.0):
    def one(cells):
        record = {
            "schema_version": 3, "record_type": "spectral_fiber",
            "cells": cells, "rows": 2, "rank": 2,
            "certificate": {
                "degree": 1, "rank": 2, "lower_gap": 1.0, "upper_gap": 1.0,
                "nearest_discarded_separation": 1.0, "localization": NAN,
                "localization_support_fraction": NAN,
                "localization_excess": 0.0, "projector_residual": 1e-16,
                "eigen_residual": 1e-16, "left_residual": 1e-16,
                "gram_defect": 0.0, "projector_norm": 1.0,
                "frame_condition_number": 1.0, "positive_signature": 2,
                "negative_signature": 0, "frequency_lower": 0.0,
                "frequency_upper": 2.0, "self_adjoint": True,
                "accepted": True,
                "certificate": {"grade": "certified-numerical",
                                "domain": "band-window",
                                "regime": "positive-semidefinite",
                                "residual": 1e-16, "conditioning": 1.0,
                                "tolerance": 1e-9,
                                "dense_reference_error": NAN}}}
        _split("eigenvalues", [1.0 + 0j, 1.0 + 0j], record)
        _split("right_frame", np.eye(2), record)
        _split("left_frame", np.eye(2), record)
        _split("weights", np.ones(2, dtype=complex), record)
        return obs.SpectralFiber.fromRecord(record)
    return obs.FiberConnection().transport(
        one([[0], [1]]), one([[2], [3]]), np.eye(2) * (1.0 + leakage))


def _triangle():
    sig = tessera.Signature(4, tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    st = tessera.Spacetime(metric, tessera.HERMITIAN_WEIGHTED, 1.0, 1.0,
                           tessera.PREFERRED, tessera.Toroid())
    verts = [st.createVertex(i) for i in range(3)]
    for simplex in [(0, 1), (1, 2), (2, 0)]:
        st.createSimplex([verts[i] for i in simplex])
    for e in st.getEdgeList().toVector():
        e.setLength(1.0 + 0j)
        e.setPhase(0.0)
    return st


class RegisterCase(unittest.TestCase):
    """A filled triangle, a contour-selected band on it, a track, a
    transport: every conjunct satisfiable, so one test spoils exactly one."""

    def setUp(self):
        self.st = _triangle()
        self.support = [0, 1, 2]
        self.track = _track()
        self.transports = [_transport()]

    def read(self, band=None, **cfg_overrides):
        cfg = obs.ClusterRegisterConfig()
        for name, value in cfg_overrides.items():
            setattr(cfg, name, value)
        return obs.ClusterRegister(cfg).read(
            self.st, self.support,
            _contour_fiber() if band is None else band,
            self.track, self.transports)


class TestRegisterContourConjunct(RegisterCase):

    def test_the_register_reports_the_contour_and_its_bound(self):
        read = self.read()
        self.assertTrue(read.accepted, read.describe())
        self.assertTrue(read.contour.startswith("circle"))
        self.assertEqual(read.contourNodeCount, 32)
        self.assertAlmostEqual(read.resolventBound, 2.0)
        self.assertAlmostEqual(read.allowabilityMargin, math.pi)
        self.assertIn("contour", read.describe())

    def test_a_blown_up_resolvent_fails_by_name(self):
        read = self.read(maxResolventBound=1.0)
        self.assertFalse(read.accepted)
        self.assertIn(obs.RegisterConjunct.CONTOUR_RESOLVENT,
                      list(read.failedConjuncts))

    def test_an_unmeasured_resolvent_is_unmeasured_not_failed(self):
        read = self.read(band=_contour_fiber(resolvent_bound=NAN))
        self.assertFalse(read.accepted)
        self.assertIn(obs.RegisterUnmeasured.RESOLVENT_UNMEASURED,
                      list(read.unmeasured))
        self.assertNotIn(obs.RegisterConjunct.CONTOUR_RESOLVENT,
                         list(read.failedConjuncts))

    def test_a_band_with_no_contour_passes_until_one_is_required(self):
        band = _contour_fiber(contour="", nodes=0, resolvent_bound=NAN)
        self.assertTrue(self.read(band=band).accepted,
                        "a band selected without a contour is not refused")
        required = self.read(band=band, requireContour=True)
        self.assertFalse(required.accepted)
        self.assertIn(obs.RegisterUnmeasured.NO_CONTOUR,
                      list(required.unmeasured))

    def test_a_lorentzian_declaration_needs_a_positive_rotation(self):
        declared = self.read(band=_contour_fiber(epsilon=0.1, margin=0.7),
                             lorentzian=True)
        self.assertTrue(declared.accepted, declared.describe())
        at_zero = self.read(band=_contour_fiber(epsilon=0.0, margin=0.0,
                                                allowable=False),
                            lorentzian=True)
        self.assertFalse(at_zero.accepted)
        self.assertIn(obs.RegisterConjunct.LORENTZIAN_ROTATION,
                      list(at_zero.failedConjuncts))

    def test_an_undeclared_rotation_on_a_lorentzian_complex_is_unmeasured(self):
        read = self.read(band=_contour_fiber(), lorentzian=True)
        self.assertFalse(read.accepted)
        self.assertIn(obs.RegisterUnmeasured.ROTATION_UNMEASURED,
                      list(read.unmeasured))

    def test_a_euclidean_read_never_decides_the_rotation(self):
        """Without the declaration the rotation is reported, never gated."""
        read = self.read(band=_contour_fiber(epsilon=NAN))
        self.assertTrue(read.accepted, read.describe())
        self.assertTrue(math.isnan(read.lorentzianEpsilon))

    def test_the_read_round_trips(self):
        read = self.read()
        again = obs.ClusterRegisterRead.fromRecord(read.toRecord())
        self.assertEqual(again.contour, read.contour)
        self.assertEqual(again.contourNodeCount, read.contourNodeCount)
        self.assertAlmostEqual(again.resolventBound, read.resolventBound)
        self.assertAlmostEqual(again.allowabilityMargin,
                               read.allowabilityMargin)
        self.assertEqual(again.accepted, read.accepted)
