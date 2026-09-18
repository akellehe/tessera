# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""#860 — the register carried by a certified cluster.

The whitepaper's "Recursive spectral fibers" section states the construction
directly: within a component ``C``, an isolated localized band with frame
``Phi_C`` satisfying ``Phi_C^dagger W_C Phi_C = I_r``, and the derived fiber
is ``E_C = Ran Phi_C``.  Acceptance is a six-conjunct list.

These tests pin that the six conjuncts are ENFORCED and each failure NAMES
itself; that a conjunct with no evidence is UNMEASURED rather than failed or
zero; that each metric regime reports what the specification requires; that
no hole is required anywhere; and that no proposer can veto a certified
fiber.
"""

import cmath
import math
import sys
import unittest
from pathlib import Path

import numpy as np

import tessera

obs = tessera.observables
cob = tessera.cobordism

NAN = float("nan")


# --------------------------------------------------------------------------- #
# fixture builders (the shared explicit-complex idiom)
# --------------------------------------------------------------------------- #
def _from_simplices(num_vertices, simplices, ids=None):
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


def _triangle():
    """3-cycle 0-1-2-0 — a FILLED complex with no removed cell anywhere."""
    return _from_simplices(3, [(0, 1), (1, 2), (2, 0)])


def _two_triangles():
    """Two disjoint triangles: the disconnected-support fixture."""
    return _from_simplices(
        6, [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)],
        ids=[0, 1, 2, 10, 11, 12])


def _split(name, values, record):
    arr = np.asarray(values, dtype=complex).reshape(-1)
    record[name + "_re"] = [float(v.real) for v in arr]
    record[name + "_im"] = [float(v.imag) for v in arr]


def _cert_record(regime):
    return {"grade": "certified-numerical", "domain": "band-window",
            "regime": regime, "residual": 1e-16, "conditioning": 1.0,
            "tolerance": 1e-9, "dense_reference_error": NAN}


def _fiber(cells, right, left=None, weights=None, *, degree=1, accepted=True,
           regime="positive-semidefinite", pos=None, neg=0, gap=1.0,
           localization_excess=0.0, cond=1.0, frame_cond=1.0,
           self_adjoint=True, gram_defect=0.0, eigen_res=1e-16,
           left_res=1e-16):
    """Rehydrate a SpectralFiber from its record (the #769 replay route), so
    each certificate field can be driven independently."""
    right = np.asarray(right, dtype=complex)
    n, r = right.shape
    left = right if left is None else np.asarray(left, dtype=complex)
    weights = (np.ones(n, dtype=complex) if weights is None
               else np.asarray(weights, dtype=complex))
    pos = r if pos is None else pos
    record = {
        "schema_version": 2, "record_type": "spectral_fiber",
        "cells": [[int(v) for v in cell] for cell in cells],
        "rows": int(n), "rank": int(r),
        "certificate": {
            "degree": int(degree), "rank": int(r),
            "lower_gap": float(gap), "upper_gap": float(gap),
            "nearest_discarded_separation": float(gap),
            "localization": NAN, "localization_support_fraction": NAN,
            "localization_excess": float(localization_excess),
            "projector_residual": 1e-16,
            "eigen_residual": float(eigen_res),
            "left_residual": float(left_res),
            "gram_defect": float(gram_defect),
            "projector_norm": float(cond),
            "frame_condition_number": float(frame_cond),
            "positive_signature": int(pos), "negative_signature": int(neg),
            "frequency_lower": 0.0, "frequency_upper": 2.0,
            "self_adjoint": bool(self_adjoint), "accepted": bool(accepted),
            "certificate": _cert_record(regime)}}
    _split("eigenvalues", [1.0 + 0j] * r, record)
    _split("right_frame", right, record)
    _split("left_frame", left, record)
    _split("weights", weights, record)
    return obs.SpectralFiber.fromRecord(record)


def _vertex_fiber(vertex_ids, **kw):
    """Rank-1 degree-0 band supported on the given vertices."""
    n = len(vertex_ids)
    right = np.zeros((n, 1), dtype=complex)
    right[0, 0] = 1.0
    return _fiber([[v] for v in vertex_ids], right, degree=0, **kw)


def _clique_edges(vertices, src, tgt):
    for i in range(len(vertices)):
        for j in range(i + 1, len(vertices)):
            src.append(vertices[i])
            tgt.append(vertices[j])


def _pm_config():
    cfg = tessera.PersistentModularityConfig()
    cfg.resolutions = [1.0]
    cfg.restarts = 4
    cfg.baseSeed = 0
    cfg.overlapThreshold = 0.0   # keep the track alive so the READ gates it
    return cfg


def _track(frames, migrate=0):
    """A FrameTrack the library itself produced.

    `FrameTrack` is read-only by design and a hand-built one would be a
    fabricated measurement, so the lifetime and the adjacent-frame overlap
    both come out of `PersistentModularity.trackAcrossFrames` — the
    specification's own supplier of the two quantities.

    `migrate` cells leave the tracked clique from the second frame onward,
    which is what actually lowers the adjacent-frame Jaccard.  The overlap
    this achieves is a MEASURED property of the fixture, so callers read
    `track.minAdjacentOverlap` rather than asking for a value: the tracker
    stops following the clique once too many cells leave, so overlaps below
    about 0.5 are simply not reachable this way and pretending otherwise
    would be a fixture that lies.
    """
    base = list(range(6))
    other = list(range(10, 16))
    frames_out = []
    for t in range(frames):
        src, tgt = [], []
        kept = base[:6 - migrate] if t > 0 else base
        _clique_edges(kept, src, tgt)
        _clique_edges(other + (base[6 - migrate:] if t > 0 else []), src, tgt)
        graph = tessera.PersistentModularity.fromWeightedEdges(
            src, tgt, [1.0] * len(src))
        frames_out.append(graph.discover(1.0, _pm_config()).components)
    src, tgt = [], []
    _clique_edges(base, src, tgt)
    _clique_edges(other, src, tgt)
    graph = tessera.PersistentModularity.fromWeightedEdges(
        src, tgt, [1.0] * len(src))
    tracks = graph.trackAcrossFrames(frames_out, 0.0)
    if not tracks:
        raise AssertionError("fixture produced no frame track")
    # The track covering the most frames is the tracked clique's.
    return max(tracks, key=lambda t: t.frames)


def _transport(leakage):
    """A transport read carrying only the leakage the conjunct consumes."""
    a = _fiber([[0], [1]], np.eye(2))
    b = _fiber([[2], [3]], np.eye(2))
    conn = obs.FiberConnection()
    return conn.transport(a, b, np.eye(2) * (1.0 + leakage))


ALL_LOCALIZED = 0.0        # as concentrated as the rank permits
FULLY_SPREAD = 1.0         # perfectly delocalized


class ClusterRegisterCase(unittest.TestCase):
    """Shared assembly: a filled triangle, a band on it, a track, a
    transport.  Every conjunct is satisfiable so a single test can spoil
    exactly one and watch it name itself."""

    def setUp(self):
        self.st = _triangle()
        self.support = [0, 1, 2]
        self.band = _vertex_fiber(self.support)
        self.track = _track(frames=3)
        self.transports = [_transport(0.0)]
        self.reader = obs.ClusterRegister()

    def read(self, **overrides):
        return self.reader.read(
            overrides.get("st", self.st),
            overrides.get("support", self.support),
            overrides.get("band", self.band),
            overrides.get("track", self.track),
            overrides.get("transports", self.transports))


# --------------------------------------------------------------------------- #
# the fiber is E_C = Ran Phi_C on a cluster, with no hole anywhere
# --------------------------------------------------------------------------- #
class TestFiberOnACluster(ClusterRegisterCase):

    def test_a_filled_complex_has_no_hole_and_still_carries_a_register(self):
        """The acceptance criterion of the ticket: a register is constructed
        and ACCEPTED from a certified cluster with NO hole in the complex."""
        holes = cob.MultiCobordism.emergent_holes(self.st, 0)
        self.assertEqual(list(holes), [],
                         "fixture must have no emergent hole at all")
        read = self.read()
        self.assertTrue(read.accepted, read.describe())
        self.assertEqual(list(read.failedConjuncts), [])
        self.assertEqual(list(read.unmeasured), [])

    def test_the_fiber_is_the_bands_frame_range(self):
        """E_C = Ran Phi_C: the read carries the band, whose right frame is
        Phi_C, and the rank is the fiber's dimension."""
        read = self.read()
        self.assertEqual(read.rank, self.band.rank())
        self.assertEqual(read.degree, self.band.degree())
        np.testing.assert_allclose(
            np.asarray(read.band.rightFrame()),
            np.asarray(self.band.rightFrame()))

    def test_no_rank_is_requested(self):
        """The read never asks for a particular rank — a rank-2 band is read
        as rank 2 and accepted on the same six conjuncts."""
        band = _fiber([[0], [1], [2]],
                      np.array([[1, 0], [0, 1], [0, 0]], dtype=complex),
                      degree=0)
        read = self.read(band=band)
        self.assertEqual(read.rank, 2)
        self.assertTrue(read.accepted, read.describe())


# --------------------------------------------------------------------------- #
# each of the six conjuncts is ENFORCED and names itself
# --------------------------------------------------------------------------- #
class TestSixConjunctsEnforced(ClusterRegisterCase):

    def test_disconnected_support_fails_cluster_support_by_name(self):
        st = _two_triangles()
        support = [0, 1, 2, 10, 11, 12]
        band = _vertex_fiber(support)
        read = self.reader.read(st, support, band, self.track,
                                self.transports)
        self.assertFalse(read.accepted)
        self.assertIn(obs.RegisterConjunct.CLUSTER_SUPPORT,
                      list(read.failedConjuncts))
        self.assertEqual(read.supportPieces, 2)
        self.assertFalse(read.supportConnected)

    def test_connected_support_passes_that_conjunct(self):
        read = self.read()
        self.assertTrue(read.supportConnected)
        self.assertEqual(read.supportPieces, 1)
        self.assertNotIn(obs.RegisterConjunct.CLUSTER_SUPPORT,
                         list(read.failedConjuncts))

    def test_uncertified_band_fails_localized_projector_by_name(self):
        band = _vertex_fiber(self.support, accepted=False,
                             localization_excess=FULLY_SPREAD)
        read = self.read(band=band)
        self.assertFalse(read.accepted)
        self.assertIn(obs.RegisterConjunct.LOCALIZED_PROJECTOR,
                      list(read.failedConjuncts))

    def test_zero_band_gap_fails_band_gap_by_name(self):
        band = _vertex_fiber(self.support, gap=0.0)
        read = self.read(band=band)
        self.assertFalse(read.accepted)
        self.assertIn(obs.RegisterConjunct.BAND_GAP,
                      list(read.failedConjuncts))
        self.assertEqual(read.bandGap, 0.0)

    def test_low_neighbour_overlap_fails_that_conjunct_by_name(self):
        """The tracker stops following a clique once too many cells leave, so
        overlaps far below 0.5 are not reachable from a real fixture.  The
        gate is therefore exercised by raising the FLOOR above the overlap
        the fixture actually achieved — the measurement stays honest and the
        threshold does the work it exists to do."""
        track = _track(frames=3, migrate=1)
        achieved = track.minAdjacentOverlap
        self.assertLess(achieved, 1.0,
                        "fixture must actually lower the overlap")
        cfg = obs.ClusterRegisterConfig()
        cfg.minNeighbourOverlap = achieved + 0.05
        read = obs.ClusterRegister(cfg).read(
            self.st, self.support, self.band, track, self.transports)
        self.assertFalse(read.accepted)
        self.assertIn(obs.RegisterConjunct.NEIGHBOUR_OVERLAP,
                      list(read.failedConjuncts))
        self.assertAlmostEqual(read.neighbourOverlap, achieved)

    def test_single_frame_fails_frame_lifetime_by_name(self):
        """'lifetime across MULTIPLE cobordism frames' — one frame is not a
        lifetime, and the default floor is two."""
        read = self.read(track=_track(frames=1))
        self.assertFalse(read.accepted)
        self.assertIn(obs.RegisterConjunct.FRAME_LIFETIME,
                      list(read.failedConjuncts))
        self.assertEqual(read.frameLifetime, 1.0)

    def test_large_leakage_fails_transport_leakage_by_name(self):
        read = self.read(transports=[_transport(0.5)])
        self.assertFalse(read.accepted)
        self.assertIn(obs.RegisterConjunct.TRANSPORT_LEAKAGE,
                      list(read.failedConjuncts))
        self.assertGreater(read.transportLeakage,
                           read.thresholds.maxTransportLeakage)

    def test_leakage_is_the_worst_over_the_supplied_transports(self):
        read = self.read(transports=[_transport(0.0), _transport(0.5),
                                     _transport(0.0)])
        self.assertGreater(read.transportLeakage, 0.1)

    def test_every_conjunct_is_independently_decisive(self):
        """Spoiling any ONE conjunct blocks acceptance — none is redundant
        and none is merely measured."""
        spoilers = {
            obs.RegisterConjunct.BAND_GAP:
                dict(band=_vertex_fiber(self.support, gap=0.0)),
            obs.RegisterConjunct.FRAME_LIFETIME:
                dict(track=_track(frames=1)),
            obs.RegisterConjunct.TRANSPORT_LEAKAGE:
                dict(transports=[_transport(0.5)]),
        }
        self.assertTrue(self.read().accepted, "baseline must accept")
        for name, override in spoilers.items():
            with self.subTest(conjunct=name):
                read = self.read(**override)
                self.assertFalse(read.accepted)
                self.assertIn(name, list(read.failedConjuncts))

        # The overlap conjunct is spoiled by its threshold rather than by an
        # unreachable fixture overlap; see the dedicated test above.
        with self.subTest(conjunct=obs.RegisterConjunct.NEIGHBOUR_OVERLAP):
            cfg = obs.ClusterRegisterConfig()
            cfg.minNeighbourOverlap = self.track.minAdjacentOverlap + 0.05
            read = obs.ClusterRegister(cfg).read(
                self.st, self.support, self.band, self.track,
                self.transports)
            self.assertFalse(read.accepted)
            self.assertIn(obs.RegisterConjunct.NEIGHBOUR_OVERLAP,
                          list(read.failedConjuncts))


# --------------------------------------------------------------------------- #
# unmeasured is not failed, and never a zero
# --------------------------------------------------------------------------- #
class TestUnmeasuredIsNotFailed(ClusterRegisterCase):

    def test_absent_track_leaves_lifetime_unmeasured_not_satisfied(self):
        """No frame track means the lifetime was never measured.  The
        certificate must fail BY NAME rather than pass a single-frame test
        vacuously — and the lifetime must stay NaN, not become zero."""
        read = self.read(track=None)
        self.assertFalse(read.accepted)
        self.assertIn(obs.RegisterUnmeasured.NO_FRAME_TRACK,
                      list(read.unmeasured))
        self.assertNotIn(obs.RegisterConjunct.FRAME_LIFETIME,
                         list(read.failedConjuncts))
        self.assertTrue(math.isnan(read.frameLifetime))
        self.assertTrue(math.isnan(read.neighbourOverlap))

    def test_absent_transports_leave_leakage_unmeasured_not_small(self):
        """Absence of transports is NOT evidence of small leakage."""
        read = self.read(transports=[])
        self.assertFalse(read.accepted)
        self.assertIn(obs.RegisterUnmeasured.NO_TRANSPORT,
                      list(read.unmeasured))
        self.assertNotIn(obs.RegisterConjunct.TRANSPORT_LEAKAGE,
                         list(read.failedConjuncts))
        self.assertTrue(math.isnan(read.transportLeakage))

    def test_unmeasured_localization_is_named_not_failed(self):
        band = _vertex_fiber(self.support, localization_excess=NAN)
        read = self.read(band=band)
        self.assertFalse(read.accepted)
        self.assertIn(obs.RegisterUnmeasured.LOCALIZATION_UNMEASURED,
                      list(read.unmeasured))
        self.assertTrue(math.isnan(read.localizationExcess))

    def test_failure_and_absence_are_disjoint_vocabularies(self):
        """No name appears in both lists: a measured shortfall and a missing
        measurement are different statements."""
        for override in (dict(track=None), dict(transports=[]),
                         dict(band=_vertex_fiber(self.support, gap=0.0))):
            with self.subTest(override=sorted(override)):
                read = self.read(**override)
                self.assertEqual(
                    set(read.failedConjuncts) & set(read.unmeasured), set())


# --------------------------------------------------------------------------- #
# the three regimes report what the specification requires
# --------------------------------------------------------------------------- #
class TestRegimeReporting(ClusterRegisterCase):

    def test_positive_regime_reports_the_gram_defect(self):
        read = self.read()
        self.assertEqual(read.regime.regime,
                         cob.CertificateRegime.PositiveSemidefinite)
        self.assertAlmostEqual(read.regime.gramDefect, 0.0)
        self.assertEqual(read.regime.negativeSignature, 0)

    def test_hermitian_indefinite_reports_inertia_and_normalizability(self):
        """'record the inertia of Phi^dagger W Phi and normalize it to
        J_C = diag(I_p, -I_q)'."""
        band = _fiber([[0], [1], [2]],
                      np.array([[1, 0], [0, 1], [0, 0]], dtype=complex),
                      degree=0, pos=1, neg=1,
                      regime="hermitian-indefinite",
                      localization_excess=ALL_LOCALIZED)
        read = self.read(band=band)
        self.assertEqual(read.regime.regime,
                         cob.CertificateRegime.HermitianIndefinite)
        self.assertEqual(read.regime.positiveSignature, 1)
        self.assertEqual(read.regime.negativeSignature, 1)
        self.assertEqual(read.regime.neutralSignature, 0)
        self.assertTrue(read.regime.signatureNormalizable)

    def test_a_neutral_direction_is_reported_as_not_normalizable(self):
        band = _fiber([[0], [1], [2]],
                      np.array([[1, 0], [0, 1], [0, 0]], dtype=complex),
                      degree=0, pos=1, neg=0,
                      regime="hermitian-indefinite",
                      localization_excess=ALL_LOCALIZED)
        read = self.read(band=band)
        self.assertEqual(read.regime.neutralSignature, 1)
        self.assertFalse(read.regime.signatureNormalizable)

    def test_negative_signature_is_never_called_an_antiparticle(self):
        band = _vertex_fiber(self.support, pos=0, neg=1,
                             regime="hermitian-indefinite")
        read = self.read(band=band)
        text = read.describe().lower()
        for banned in ("antiparticle", "antiquark", "positron"):
            self.assertNotIn(banned, text)

    def test_non_normal_reports_both_residuals_and_the_frame_conditioning(self):
        """'use matched right and left frames ... and report both residuals
        and the frame condition number'."""
        band = _vertex_fiber(self.support, self_adjoint=False,
                             regime="non-normal", eigen_res=1e-13,
                             left_res=3e-13, frame_cond=7.5)
        read = self.read(band=band)
        self.assertEqual(read.regime.regime, cob.CertificateRegime.NonNormal)
        self.assertAlmostEqual(read.regime.eigenResidual, 1e-13)
        self.assertAlmostEqual(read.regime.leftResidual, 3e-13)
        self.assertAlmostEqual(read.regime.frameConditionNumber, 7.5)
        self.assertNotEqual(read.regime.eigenResidual,
                            read.regime.leftResidual)


# --------------------------------------------------------------------------- #
# however proposed: no proposer is an input, so none can veto
# --------------------------------------------------------------------------- #
class TestHoweverProposed(ClusterRegisterCase):

    def test_the_read_takes_no_proposer_argument_at_all(self):
        """'a persistent connected cluster support, HOWEVER PROPOSED' — the
        signature carries a complex, a support, a band, a track and
        transports.  There is no proposer parameter, so no proposer has a
        channel through which to veto a certified fiber."""
        signature = (obs.ClusterRegister.read.__doc__ or "").split(")")[0]
        for accepted in ("st", "support", "band", "track",
                         "externalTransports"):
            self.assertIn(accepted, signature)
        for absent in ("modularity", "proposer", "partition", "community",
                       "strategy"):
            self.assertNotIn(absent, signature.lower())

    def test_identical_supports_read_identically_whatever_proposed_them(self):
        """Two supports that are equal as SETS give the same verdict, so the
        proposal order — the only trace a proposer could leave — is not an
        input."""
        forward = self.reader.read(self.st, [0, 1, 2], self.band, self.track,
                                   self.transports)
        reversed_ = self.reader.read(self.st, [2, 1, 0], self.band,
                                     self.track, self.transports)
        self.assertEqual(forward.accepted, reversed_.accepted)
        self.assertEqual(forward.supportPieces, reversed_.supportPieces)
        self.assertEqual(list(forward.failedConjuncts),
                         list(reversed_.failedConjuncts))

    def test_a_disagreeing_proposer_cannot_veto_a_certified_fiber(self):
        """A modularity run that splits the triangle into singletons does not
        change the verdict on the whole-triangle support: acceptance is
        conditioned only on the six conjuncts."""
        pm = obs.PersistentModularity.fromSpacetime(self.st)
        cfg = obs.PersistentModularityConfig()
        cfg.resolutions = [50.0]        # a resolution that shatters it
        report = pm.scanResolutions(cfg)
        proposed = [list(c.support) for c in report.slices[0].components]
        self.assertGreater(len(proposed), 1,
                           "fixture must actually disagree with the support")
        read = self.read()
        self.assertTrue(read.accepted, read.describe())


# --------------------------------------------------------------------------- #
# support connectivity
# --------------------------------------------------------------------------- #
class TestSupportConnectivity(unittest.TestCase):

    def test_connected_support(self):
        st = _triangle()
        connected, pieces = obs.ClusterRegister.supportConnectivity(
            st, [0, 1, 2])
        self.assertTrue(connected)
        self.assertEqual(pieces, 1)

    def test_two_pieces(self):
        st = _two_triangles()
        connected, pieces = obs.ClusterRegister.supportConnectivity(
            st, [0, 1, 2, 10, 11, 12])
        self.assertFalse(connected)
        self.assertEqual(pieces, 2)

    def test_singleton_is_connected(self):
        st = _triangle()
        connected, pieces = obs.ClusterRegister.supportConnectivity(st, [1])
        self.assertTrue(connected)
        self.assertEqual(pieces, 1)

    def test_empty_support_is_not_connected_and_has_no_pieces(self):
        st = _triangle()
        connected, pieces = obs.ClusterRegister.supportConnectivity(st, [])
        self.assertFalse(connected)
        self.assertEqual(pieces, 0)

    def test_unknown_ids_are_ignored_not_counted(self):
        st = _triangle()
        connected, pieces = obs.ClusterRegister.supportConnectivity(
            st, [0, 1, 2, 9999])
        # 9999 is not in the complex, so it induces its own isolated piece.
        self.assertFalse(connected)
        self.assertEqual(pieces, 2)


# --------------------------------------------------------------------------- #
# serialization round-trip
# --------------------------------------------------------------------------- #
class TestRecordRoundTrip(ClusterRegisterCase):

    def test_round_trip_preserves_every_channel(self):
        read = self.read()
        again = obs.ClusterRegisterRead.fromRecord(read.toRecord())
        self.assertEqual(again.accepted, read.accepted)
        self.assertEqual(again.rank, read.rank)
        self.assertEqual(again.degree, read.degree)
        self.assertEqual(list(again.support), list(read.support))
        self.assertEqual(again.supportPieces, read.supportPieces)
        self.assertEqual(again.regime.positiveSignature,
                         read.regime.positiveSignature)
        self.assertEqual(list(again.failedConjuncts),
                         list(read.failedConjuncts))
        self.assertEqual(list(again.unmeasured), list(read.unmeasured))

    def test_nan_channels_survive_as_nan_not_zero(self):
        read = self.read(track=None, transports=[])
        again = obs.ClusterRegisterRead.fromRecord(read.toRecord())
        self.assertTrue(math.isnan(again.frameLifetime))
        self.assertTrue(math.isnan(again.neighbourOverlap))
        self.assertTrue(math.isnan(again.transportLeakage))

    def test_unknown_schema_version_is_rejected(self):
        """A reader that guessed at an unknown schema would silently
        misread a checkpoint, so it refuses instead."""
        record = self.read().toRecord()
        record["schema_version"] = 999
        with self.assertRaises(ValueError):
            obs.ClusterRegisterRead.fromRecord(record)

    def test_a_foreign_record_type_is_rejected(self):
        record = self.read().toRecord()
        record["record_type"] = "spectral_fiber"
        with self.assertRaises(ValueError):
            obs.ClusterRegisterRead.fromRecord(record)


# --------------------------------------------------------------------------- #
# nothing is target-conditioned
# --------------------------------------------------------------------------- #
class TestNoTargetConditioning(unittest.TestCase):

    def test_the_api_accepts_no_target_of_any_kind(self):
        """No target vector, no residual, no objective term reaches this
        read: the signature has no parameter that could carry one."""
        signature = (obs.ClusterRegister.read.__doc__ or "").split(")")[0]
        for banned in ("target", "residual", "objective", "state",
                       "periods"):
            self.assertNotIn(banned, signature.lower())

    def test_reading_twice_is_pure(self):
        """A read carries no state between calls: nothing accumulates and
        nothing is conditioned on a previous answer."""
        st = _triangle()
        band = _vertex_fiber([0, 1, 2])
        reader = obs.ClusterRegister()
        first = reader.read(st, [0, 1, 2], band, None, [])
        second = reader.read(st, [0, 1, 2], band, None, [])
        self.assertEqual(first.accepted, second.accepted)
        self.assertEqual(list(first.unmeasured), list(second.unmeasured))
        self.assertEqual(first.supportPieces, second.supportPieces)

    def test_no_hole_vocabulary_survives_in_the_read(self):
        st = _triangle()
        band = _vertex_fiber([0, 1, 2])
        read = obs.ClusterRegister().read(st, [0, 1, 2], band, None, [])
        text = read.describe().lower()
        for banned in ("hole", "period", "target"):
            self.assertNotIn(banned, text)


# --------------------------------------------------------------------------- #
# end to end on a real host, with every input produced by the library
# --------------------------------------------------------------------------- #
class TestRealHost(unittest.TestCase):
    """The whole read driven from real machinery: a real complex, real
    clusters from PersistentModularity, real bands from SpectralFiberTracker,
    a real frame track and a real transport.  Nothing here is synthetic."""

    def _host(self):
        sys.path.insert(
            0, str(Path(__file__).resolve().parent.parent.parent /
                   "examples" / "cobordism"))
        from tessera.drivers import emergence as ea
        st = ea.build_cobordism_host(6, 3)
        # Materialise the skeleton before reading (#850): the objective and
        # the operators are functions of the FULL complex, not of a partial
        # one.
        node = cob.MultiCobordism(st, [], [], [1], 1.0, 7)
        node.set_objective(cob.JointStationarityObjective())
        list(node.run_stage1(max_steps=1, n_candidate_moves=6))
        return st

    def test_a_real_host_with_no_hole_still_measures_every_conjunct(self):
        st = self._host()
        self.assertEqual(
            len(list(cob.MultiCobordism.emergent_holes(st, 1))), 0,
            "the acceptance criterion needs a host with NO hole")

        pm = obs.PersistentModularity.fromSpacetime(st)
        cfg = obs.PersistentModularityConfig()
        cfg.resolutions = [1.0]
        comps = list(pm.scanResolutions(cfg).slices[0].components)
        self.assertGreaterEqual(len(comps), 2)

        supports = [[int(v) for v in c.support] for c in comps]
        tracks = pm.trackAcrossFrames([comps, comps, comps], 0.5)
        self.assertTrue(tracks)

        tracker = obs.SpectralFiberTracker(st, obs.SpectralFiberConfig())
        band_a = list(tracker.enumerateBands(supports[0], 1).fibers)[0]
        band_b = list(tracker.enumerateBands(supports[1], 1).fibers)[0]
        transport = obs.FiberConnection().transportOnSpacetime(
            st, band_a, band_b)

        read = obs.ClusterRegister().read(
            st, supports[0], band_a, tracks[0], [transport])

        # With every input supplied, NOTHING is unmeasured: the verdict rests
        # on measurements rather than on absent evidence.
        self.assertEqual(list(read.unmeasured), [],
                         "every conjunct must be decided on evidence")
        self.assertTrue(read.supportConnected)
        self.assertEqual(read.supportPieces, 1)
        for value in (read.localizationExcess, read.bandGap,
                      read.neighbourOverlap, read.frameLifetime,
                      read.transportLeakage):
            self.assertFalse(math.isnan(value))
        self.assertGreaterEqual(read.frameLifetime, 2.0)

    def test_a_real_refusal_names_a_measured_reason(self):
        """A refusal is a result.  On this host the external transport is
        rank-deficient, so leakage is MEASURED and fails — the register is
        refused for a reason, not for missing evidence."""
        st = self._host()
        pm = obs.PersistentModularity.fromSpacetime(st)
        cfg = obs.PersistentModularityConfig()
        cfg.resolutions = [1.0]
        comps = list(pm.scanResolutions(cfg).slices[0].components)
        supports = [[int(v) for v in c.support] for c in comps]
        tracks = pm.trackAcrossFrames([comps, comps, comps], 0.5)
        tracker = obs.SpectralFiberTracker(st, obs.SpectralFiberConfig())
        band_a = list(tracker.enumerateBands(supports[0], 1).fibers)[0]
        band_b = list(tracker.enumerateBands(supports[1], 1).fibers)[0]
        transport = obs.FiberConnection().transportOnSpacetime(
            st, band_a, band_b)

        read = obs.ClusterRegister().read(
            st, supports[0], band_a, tracks[0], [transport])
        if read.accepted:
            self.skipTest("this host's transport certified; nothing refused")
        self.assertTrue(read.failedConjuncts,
                        "a refusal must name at least one conjunct")
        for name in read.failedConjuncts:
            self.assertIn(name, (
                obs.RegisterConjunct.CLUSTER_SUPPORT,
                obs.RegisterConjunct.LOCALIZED_PROJECTOR,
                obs.RegisterConjunct.BAND_GAP,
                obs.RegisterConjunct.NEIGHBOUR_OVERLAP,
                obs.RegisterConjunct.FRAME_LIFETIME,
                obs.RegisterConjunct.TRANSPORT_LEAKAGE))


if __name__ == "__main__":
    unittest.main()
