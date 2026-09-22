# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Acceptance tests for quark/antiquark discovery and the emergent
flavor-charge reads (:class:`tessera.observables.ParticleClusters`),
ticket #773 / design spec sections 6.8 and 16.1 (Algorithm I, quark
classifier) and the whitepaper "Quarks as modular clusters".

Covers every ticket acceptance bullet:

* synthetic anchored quark/antiquark fixtures differ by orientation, have
  certified winding nu = +-1, and provisionally carry B = +-1/3;
* a two-quark anti-triplet is NOT mislabeled an antiquark (distinguished
  by total occupation and determinant-line data, not color alone);
* a certified gap-preserving conjugate-pair creation path has zero total
  determinant winding/baryon flux and even total parity; a singular
  (gap-closing) path returns UNKNOWN flux;
* a missing/unstable flavor doublet yields unknown flavor AND charge;
* certified u/d fixtures return +2/3 / -1/3 Gauss-consistent charge, and
  Q = I3 + B/2 is tested only when both baryon flux and the doublet are
  certified (the proposed u/d identification);
* unknown fields are None and every missing certificate is NAMED in
  failedCertificates (negative control per certificate);
* relabeling and refinement preserve accepted classifications;
* cached classification equals cold recomputation under the #764 cache;
* no quark-specific quantity enters the emergence objective.

Fixtures are built by composing the MERGED public APIs (#765 ComponentId,
#769 SpectralFiber/ComponentBandRead record synthesis, #767 ColorAnchor,
#770 FiberConnection transports/windings, #780 CovarianceState Wick
reads, and the existing EigenstateSynthesis.gaussLawCharge) — the
classifier consumes them; nothing is faked past its own public surface.
"""
import itertools
import math
import time
import unittest
import warnings
from pathlib import Path

import numpy as np

import tessera

obs = tessera.observables
cob = tessera.cobordism
qm = tessera.quantum

MACHINE = 1e-12
NAN = float("nan")
TWO_PI = 2.0 * math.pi

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


# --------------------------------------------------------------------------- #
# fiber fixtures (the sanctioned #769 record-rehydration route)
# --------------------------------------------------------------------------- #
def _cert_record(regime, grade="certified-numerical"):
    return {"grade": grade, "domain": "band-window", "regime": regime,
            "residual": 1e-15, "conditioning": 1.0,
            "dense_reference_error": NAN, "tolerance": 1e-9}


def _split(name, values, record):
    arr = np.asarray(values, dtype=complex).reshape(-1)
    record[name + "_re"] = [float(v.real) for v in arr]
    record[name + "_im"] = [float(v.imag) for v in arr]


def _fiber(cells, right, left=None, weights=None, *, degree=1, accepted=True,
           regime="positive-semidefinite", pos=None, neg=0, lower_gap=1.0,
           upper_gap=1.0, cond=1.0, self_adjoint=True, gram_defect=0.0,
           localization=0.5, eigenvalues=None):
    """Rehydrate a SpectralFiber from its record (the #769 replay route)."""
    right = np.asarray(right, dtype=complex)
    n, r = right.shape
    left = right if left is None else np.asarray(left, dtype=complex)
    weights = (np.ones(n, dtype=complex) if weights is None
               else np.asarray(weights, dtype=complex))
    pos = r if pos is None else pos
    eigenvalues = ([1.0 + 0j] * r if eigenvalues is None else eigenvalues)
    record = {
        "schema_version": 1, "record_type": "spectral_fiber",
        "cells": [[int(v) for v in cell] for cell in cells],
        "rows": int(n), "rank": int(r),
        "certificate": {
            "degree": int(degree), "rank": int(r),
            "lower_gap": float(lower_gap), "upper_gap": float(upper_gap),
            "localization": float(localization),
            "projector_residual": 1e-16,
            "eigen_residual": 1e-16, "left_residual": 1e-16,
            "gram_defect": float(gram_defect),
            "condition_number": float(cond),
            "positive_signature": int(pos), "negative_signature": int(neg),
            "frequency_lower": 0.0, "frequency_upper": 2.0,
            "self_adjoint": bool(self_adjoint), "accepted": bool(accepted),
            "certificate": _cert_record("positive-semidefinite"
                                        if regime is None else regime)}}
    _split("eigenvalues", eigenvalues, record)
    _split("right_frame", right, record)
    _split("left_frame", left, record)
    _split("weights", weights, record)
    return obs.SpectralFiber.fromRecord(record)


def _unit_fiber(base_id, r, **kw):
    """Rank-r fiber with identity frame on r synthetic cells."""
    return _fiber([[base_id + i] for i in range(r)], np.eye(r), **kw)


def _band_read(fibers, degree=1, support=None):
    """Synthesize a ComponentBandRead carrying `fibers` (the #769 replay
    route for a whole enumeration frame)."""
    cells = []
    for f in fibers:
        cells.extend(f.cellVertices())
    record = {
        "schema_version": 1, "record_type": "spectral_band_read",
        "support": [int(v) for v in (support or [])],
        "degree": int(degree),
        "dimension": len(cells),
        "cell_vertices": [[int(v) for v in cell] for cell in cells],
        "regime": "positive-semidefinite",
        "solver_path": "dense-self-adjoint",
        "truncated": False,
        "fibers": [f.toRecord() for f in fibers],
        "solve_certificate": _cert_record("positive-semidefinite"),
    }
    _split("covered_eigenvalues", [], record)
    return obs.ComponentBandRead.fromRecord(record)


def _phase_link(conn, A, B, phi):
    """Accepted rank-3 transport whose determinant phase is exactly phi."""
    v = np.diag([np.exp(1j * phi), 1.0, 1.0]).astype(complex)
    return conn.transport(A, B, v)


def _winding_family(conn, A, B, turns=1, samples=8):
    """A closed transport family with certified integer winding `turns`."""
    return [_phase_link(conn, A, B, TWO_PI * turns * k / samples)
            for k in range(samples)]


def _parity_occupation(occupations):
    """#780 Wick parity/total-number reads of a diagonal covariance."""
    state = qm.CovarianceState.fromOccupations(np.asarray(occupations,
                                                          dtype=float))
    return state.wickParity(), state.wickTotalNumber()


_ANCHOR_WEIGHTS = np.array([2.0, 0.5, 1.25, 0.8])


def _overlap_frame(terms=(0.98, 0.02), weights=_ANCHOR_WEIGHTS):
    """A |W|-orthonormal rank-three frame over four oriented edge rows whose
    two anchor terms are EXACTLY `terms`.

    Construction (Cauchy-Binet / cofactor identity): with
    Psi = |W|^(1/2) Phi a Euclidean isometry, the 3x3 minor omitting row i
    has |det|^2 = |n_i|^2 for the unit vector n spanning ker(Psi^dagger).
    Choosing n fixes every |det A_tau|^2 exactly, so the fixture's score and
    its overlap participation are declared, not fitted."""
    n = np.zeros(4, dtype=complex)
    n[3] = np.sqrt(terms[0])   # omit row 3 -> the (0, 1, 2) term
    n[2] = np.sqrt(terms[1])   # omit row 2 -> the (0, 1, 3) term
    # The four 3-row minors sum to one (Cauchy-Binet), so any weight the two
    # declared triangles do not claim goes to the two undeclared subsets.
    rest = 1.0 - terms[0] - terms[1]
    assert rest >= -1e-12, "declared anchor terms must not exceed one"
    n[0] = np.sqrt(max(0.0, rest) / 2.0)
    n[1] = n[0]
    _, _, vh = np.linalg.svd(n.reshape(1, 4).conj())
    psi = vh.conj().T[:, 1:]
    return np.diag(1.0 / np.sqrt(weights)) @ psi


def _anchor_profile(terms=(0.98, 0.02)):
    """#767 OVERLAPPING two-triangle atlas: (0,1,2) and (0,1,3) share the
    edge rows 0 and 1, so the determinant-phase coherence has genuine
    overlap content (#808).  Declared weighting (0.8, 0.2): score 0.788,
    coherence 0.99, held certificate."""
    phi = _overlap_frame(terms)
    anchor = obs.ColorAnchor([tessera.OrientedTriangle([0, 1, 2], [1, 1, 1]),
                              tessera.OrientedTriangle([0, 1, 3], [1, 1, 1])],
                             [0.8, 0.2])
    return anchor.evaluate(phi, _ANCHOR_WEIGHTS)


def _disjoint_anchor_profile():
    """The single-triangle atlas: nothing to overlap, so the coherence is
    UNKNOWN (NaN) and the anchor certificate fails by name (#808)."""
    anchor = obs.ColorAnchor([tessera.OrientedTriangle([0, 1, 2], [1, 1, 1])])
    return anchor.evaluate(_overlap_frame(), _ANCHOR_WEIGHTS)


def _doublet_frames(frames=3, base=200, ranks=(1, 2, 3), drop_rank2_at=None,
                    unaccept_rank2_at=None, extra_rank2=False):
    """Synthetic per-frame band enumerations for the flavor search: one
    band per rank in `ranks`, identical cells across frames (overlap 1)."""
    out = []
    for t in range(frames):
        fibers = []
        for r in ranks:
            if r == 2 and drop_rank2_at == t:
                # moved to disjoint cells: no positive overlap into frame t
                fibers.append(_unit_fiber(base + 50 + 10 * t, 2))
                continue
            accepted = not (r == 2 and unaccept_rank2_at == t)
            fibers.append(_unit_fiber(base + 10 * r, r, accepted=accepted))
        if extra_rank2:
            fibers.append(_unit_fiber(base + 80, 2))
        out.append(_band_read(fibers))
    return out


# --------------------------------------------------------------------------- #
# spacetime fixtures (shared explicit-complex idiom)
# --------------------------------------------------------------------------- #
def _from_simplices(num_vertices, simplices, ids=None, timelike=True):
    sig = tessera.Signature(4, tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    st = tessera.Spacetime(metric, tessera.HERMITIAN_WEIGHTED, 1.0, 1.0,
                           tessera.PREFERRED, tessera.Toroid())
    ids = list(range(num_vertices)) if ids is None else ids
    verts = [st.createVertex(i) for i in ids]
    for simplex in simplices:
        st.createSimplex([verts[i] for i in simplex])
    for e in st.getEdgeList().toVector():
        e.setLength(1j if timelike else (1.0 + 0j))
        e.setPhase(0.0)
    return st


_TETRA_CHAIN = [(0, 1, 2, 3), (1, 2, 3, 4), (2, 3, 4, 5), (3, 4, 5, 6)]


def _gauss_fixture(target, ids=None):
    """A tetrahedron-chain complex (all edges timelike -> every plaquette
    electric) plus a field-strength cochain whose electric flux equals
    `target` on BOTH nested enclosing surfaces (solved through the real
    gaussLawCharge by linearity)."""
    st = _from_simplices(7, _TETRA_CHAIN, ids=ids)
    base = 0 if ids is None else ids[0]
    es = cob.EigenstateSynthesis(st, 2)
    surfaces = obs.ParticleClusters.nestedEnclosures(st, [base], 2)
    n = es.order()

    def probe(vset):
        rows = []
        for c in range(n):
            f = [0j] * n
            f[c] = 1.0 + 0j
            rows.append(es.gaussLawCharge(f, vset, True).real)
        return np.array(rows)

    A = np.vstack([probe(surfaces[0]), probe(surfaces[1])])
    F, *_ = np.linalg.lstsq(A, np.array([target, target]), rcond=None)
    return st, [complex(v) for v in F], surfaces


# --------------------------------------------------------------------------- #
# the full certified evidence bundle
# --------------------------------------------------------------------------- #
def _certified_evidence(turns=1, *, occupations=(1.0, 0.0, 0.0),
                        band_base=1, with_flavor=False, occupancy=None,
                        orientation=1, with_charge=None, conn=None,
                        anchor=None):
    conn = conn or obs.FiberConnection()
    A = _unit_fiber(band_base, 3)
    B = _unit_fiber(band_base + 10, 3)
    family = _winding_family(conn, A, B, turns=turns)
    winding = conn.closedFamilyWinding(family)

    parity, occupation = _parity_occupation(list(occupations))

    ev = obs.QuarkCandidateEvidence()
    ev.component = obs.ComponentId("ab" * 16, 1)
    ev.colorBand = A
    ev.anchor = anchor if anchor is not None else _anchor_profile()
    ev.lifetimeTransports = family
    ev.winding = winding
    ev.parityRead = parity
    ev.occupationRead = occupation
    # The modularity RESOLUTION-slice numbers are reported, never gated; the
    # COBORDISM-FRAME lifetime is the gated persistence quantity (#808).
    ev.persistenceLifetime = 3.0
    ev.persistenceMinOverlap = 1.0
    ev.frameLifetime = 3.0
    ev.frameMinOverlap = 1.0
    ev.refinementOverlap = 1.0
    # The two STABILITY windows: the same rank-three band and the same
    # anchor profile at each of three cobordism frames (an unchanging
    # candidate is the stable case; the moving ones are their own tests).
    ev.colorBandFrames = [A, A, A]
    ev.anchorFrames = [ev.anchor, ev.anchor, ev.anchor]
    if with_flavor:
        pc = obs.ParticleClusters()
        ev.flavor = pc.flavorDoubletSearch(_doublet_frames())
        assert ev.flavor.found
        ev.doubletOccupancy = (np.array([1.0, 0.0], dtype=complex)
                               if occupancy is None
                               else np.asarray(occupancy, dtype=complex))
        ev.doubletOrientation = orientation
    if with_charge is not None:
        pc = obs.ParticleClusters()
        st, F, surfaces = _gauss_fixture(with_charge)
        ev.charge = pc.gaussFluxOnSurfaces(st, F, surfaces, True)
        assert ev.charge.consistent
    return ev


CORE = ["persistence", "localization", "parity-odd", "occupation-one",
        "color-rank-three", "color-rank-stability", "anchor",
        "anchor-stability", "transport-leakage", "winding",
        "winding-unit", "refinement-stability"]


# =========================================================================== #
# core classification
# =========================================================================== #
class TestCoreClassification(unittest.TestCase):
    def setUp(self):
        self.pc = obs.ParticleClusters()

    def test_certified_quark(self):
        read = self.pc.classifyQuark(_certified_evidence(turns=1))
        self.assertEqual(read.classification, "quark")
        self.assertEqual(read.determinantWinding, 1)
        self.assertAlmostEqual(read.baryonFlux, 1.0 / 3.0, delta=MACHINE)
        self.assertEqual(read.exteriorParity, -1)
        self.assertEqual(read.colorRank, 3)
        self.assertEqual(read.confidence, 1.0)
        self.assertEqual(read.windingClosure, "closed-family")
        for name in CORE:
            self.assertNotIn(name, read.failedCertificates)
        self.assertTrue(read.certificate.holds())

    def test_certified_antiquark_is_the_orientation_reverse(self):
        read = self.pc.classifyQuark(_certified_evidence(turns=-1))
        self.assertEqual(read.classification, "antiquark")
        self.assertEqual(read.determinantWinding, -1)
        self.assertAlmostEqual(read.baryonFlux, -1.0 / 3.0, delta=MACHINE)
        self.assertEqual(read.confidence, 1.0)

    def test_quark_and_antiquark_differ_only_by_orientation(self):
        q = self.pc.classifyQuark(_certified_evidence(turns=1))
        aq = self.pc.classifyQuark(_certified_evidence(turns=-1))
        # identical anchored/parity/persistence evidence; opposite line
        self.assertEqual(q.triangleAnchorScore, aq.triangleAnchorScore)
        self.assertEqual(q.exteriorParity, aq.exteriorParity)
        self.assertEqual(q.occupationTotal, aq.occupationTotal)
        self.assertEqual(q.determinantWinding, -aq.determinantWinding)
        self.assertEqual(q.baryonFlux, -aq.baryonFlux)

    def test_reversed_family_is_the_antiquark(self):
        # reversing the tube = traversing the same transport family in the
        # opposite parameter order (the #770 orientation convention)
        conn = obs.FiberConnection()
        A, B = _unit_fiber(1, 3), _unit_fiber(11, 3)
        family = _winding_family(conn, A, B, turns=1)
        ev = _certified_evidence(turns=1)
        ev.winding = conn.closedFamilyWinding(list(reversed(family)))
        read = self.pc.classifyQuark(ev)
        self.assertEqual(read.classification, "antiquark")

    def test_unknown_winding_leaves_baryon_flux_unknown(self):
        ev = _certified_evidence()
        # open segment with NO declared closure: raw endpoint phase is
        # never promoted to baryon-flux evidence
        conn = obs.FiberConnection()
        A, B = _unit_fiber(1, 3), _unit_fiber(11, 3)
        segment = [_phase_link(conn, A, B, p) for p in (0.0, 0.4, 0.8)]
        ev.winding = conn.openSegmentWinding(segment,
                                             obs.WindingClosureSpec())
        read = self.pc.classifyQuark(ev)
        self.assertIsNone(read.determinantWinding)
        self.assertIsNone(read.baryonFlux)
        self.assertEqual(read.windingClosure, "none")
        self.assertIn("winding", read.failedCertificates)
        self.assertEqual(read.classification, "none")
        self.assertFalse(read.certificate.holds())

    def test_open_segment_with_declared_closure_carries_its_specification(self):
        conn = obs.FiberConnection()
        A, B = _unit_fiber(1, 3), _unit_fiber(11, 3)
        segment = [_phase_link(conn, A, B, TWO_PI * k / 4) for k in range(5)]
        spec = obs.WindingClosureSpec()
        spec.mode = obs.WindingClosureSpec.Mode.MATCHED_REFERENCE
        spec.referenceId = "co-moving-reference"
        spec.referenceTransports = [np.eye(3)] * len(segment)
        ev = _certified_evidence()
        ev.winding = conn.openSegmentWinding(segment, spec)
        read = self.pc.classifyQuark(ev)
        self.assertEqual(read.classification, "quark")
        self.assertEqual(read.windingClosure, "matched-reference")
        self.assertEqual(read.windingReferenceId, "co-moving-reference")
        self.assertAlmostEqual(read.baryonFlux, 1.0 / 3.0, delta=MACHINE)

    def test_boundary_register_trivialization_closure(self):
        # the other declared closure of the ticket: an open segment closed
        # through boundary-register trivializations
        conn = obs.FiberConnection()
        A, B = _unit_fiber(1, 3), _unit_fiber(11, 3)
        segment = [_phase_link(conn, A, B, TWO_PI * k / 4) for k in range(5)]
        spec = obs.WindingClosureSpec()
        spec.mode = obs.WindingClosureSpec.Mode.ENDPOINT_TRIVIALIZATION
        spec.referenceId = "boundary-registers"
        spec.startTrivialization = np.eye(3)
        spec.endTrivialization = np.eye(3)
        ev = _certified_evidence()
        ev.winding = conn.openSegmentWinding(segment, spec)
        read = self.pc.classifyQuark(ev)
        self.assertEqual(read.classification, "quark")
        self.assertEqual(read.windingClosure, "endpoint-trivialization")
        self.assertEqual(read.windingReferenceId, "boundary-registers")
        self.assertAlmostEqual(read.baryonFlux, 1.0 / 3.0, delta=MACHINE)

    def test_dual_transport_carries_the_conjugate_determinant_line(self):
        # the DUAL color transport of a link is its W-adjoint reverse
        # (#770 transportReverse): its determinant line is the conjugate,
        # so the dual traversal winds opposite — the transport-level
        # statement behind quark vs antiquark orientation
        conn = obs.FiberConnection()
        A, B = _unit_fiber(1, 3), _unit_fiber(11, 3)
        for phi in (0.3, 1.1, -0.7):
            v = np.diag([np.exp(1j * phi), 1.0, 1.0]).astype(complex)
            fwd = conn.transport(A, B, v)
            dual = conn.transportReverse(B, A, v)
            self.assertTrue(fwd.accepted and dual.accepted)
            self.assertLess(abs(dual.determinantPhase
                                - np.conj(fwd.determinantPhase)), 1e-12)

    def test_certified_zero_winding_is_a_certified_zero_flux_not_a_quark(self):
        read = self.pc.classifyQuark(_certified_evidence(turns=0))
        self.assertEqual(read.determinantWinding, 0)
        self.assertEqual(read.baryonFlux, 0.0)  # certified zero, not unknown
        self.assertIn("winding-unit", read.failedCertificates)
        self.assertNotIn("winding", read.failedCertificates)
        self.assertEqual(read.classification, "none")

    def test_anchor_profile_travels_on_the_read(self):
        profile = _anchor_profile()
        read = self.pc.classifyQuark(_certified_evidence(anchor=profile))
        self.assertAlmostEqual(read.triangleAnchorScore, profile.score,
                               delta=MACHINE)
        self.assertAlmostEqual(read.triangleAnchorMaxTerm, profile.max_term,
                               delta=MACHINE)
        self.assertAlmostEqual(read.triangleAnchorParticipation,
                               profile.participation_ratio, delta=MACHINE)
        self.assertAlmostEqual(read.anchorPhaseDispersion,
                               profile.phase_dispersion, delta=MACHINE)
        self.assertEqual(read.anchorWeightingId, "declared")
        # The coherence is an OVERLAP datum (#808): both declared triangles
        # share edge rows 0 and 1, so both take part in the resultant.
        self.assertEqual(profile.overlapping_triangles, 2)
        self.assertEqual(profile.overlap_relation, "shared-edge")
        self.assertGreater(read.anchorPhaseCoherence, 0.98)

    def test_confidence_is_the_passed_core_fraction(self):
        ev = _certified_evidence()
        ev.refinementOverlap = NAN  # remove exactly one core certificate
        read = self.pc.classifyQuark(ev)
        self.assertAlmostEqual(read.confidence, 11.0 / 12.0, delta=MACHINE)
        self.assertEqual(read.classification, "none")

    def test_thresholds_are_recorded_on_every_read(self):
        cfg = obs.ParticleClustersConfig()
        cfg.minAnchorScore = 0.75
        pc = obs.ParticleClusters(cfg)
        read = pc.classifyQuark(_certified_evidence())
        self.assertEqual(read.thresholds.minAnchorScore, 0.75)
        rec = read.toRecord()
        self.assertEqual(rec["thresholds"]["min_anchor_score"], 0.75)

    def test_classify_quarks_stream_preserves_order(self):
        reads = self.pc.classifyQuarks(
            [_certified_evidence(turns=1), _certified_evidence(turns=-1)])
        self.assertEqual([r.classification for r in reads],
                         ["quark", "antiquark"])


# =========================================================================== #
# negative controls: a candidate missing ANY one certificate is NOT a quark
# and the failed certificate is NAMED
# =========================================================================== #
class TestNegativeControls(unittest.TestCase):
    def setUp(self):
        self.pc = obs.ParticleClusters()

    def _assert_named_failure(self, ev, name):
        read = self.pc.classifyQuark(ev)
        self.assertEqual(read.classification, "none")
        self.assertIn(name, read.failedCertificates)
        self.assertLess(read.confidence, 1.0)
        self.assertFalse(read.certificate.holds())
        return read

    def test_missing_anchor(self):
        ev = _certified_evidence()
        ev.anchor = obs.AnchorProfile()
        read = self._assert_named_failure(ev, "anchor")
        self.assertTrue(math.isnan(read.triangleAnchorScore))

    def test_low_anchor_score(self):
        cfg = obs.ParticleClustersConfig()
        cfg.minAnchorScore = 1.5  # unreachable: a^2 <= 1
        pc = obs.ParticleClusters(cfg)
        read = pc.classifyQuark(_certified_evidence())
        self.assertIn("anchor", read.failedCertificates)
        self.assertEqual(read.classification, "none")

    def test_even_parity(self):
        ev = _certified_evidence(occupations=(1.0, 1.0, 0.0))
        read = self._assert_named_failure(ev, "parity-odd")
        self.assertEqual(read.exteriorParity, +1)

    def test_uncertified_parity_never_emits_a_sign(self):
        ev = _certified_evidence()
        ev.parityRead = qm.WickCertificateRead()  # default: never holds
        read = self._assert_named_failure(ev, "parity-odd")
        self.assertEqual(read.exteriorParity, 0)

    def test_rank_two_band(self):
        ev = _certified_evidence()
        ev.colorBand = _unit_fiber(1, 2)
        read = self._assert_named_failure(ev, "color-rank-three")
        self.assertEqual(read.colorRank, 2)

    def test_unaccepted_band_gap_closed(self):
        ev = _certified_evidence()
        ev.colorBand = _unit_fiber(1, 3, accepted=False)
        self._assert_named_failure(ev, "color-rank-three")

    def test_leaking_transport(self):
        ev = _certified_evidence()
        conn = obs.FiberConnection()
        leaky = conn.transport(_unit_fiber(1, 3), _unit_fiber(11, 3),
                               np.diag([0.1, 1.0, 1.0]))
        self.assertFalse(leaky.accepted)
        ev.lifetimeTransports = list(ev.lifetimeTransports) + [leaky]
        self._assert_named_failure(ev, "transport-leakage")

    def test_missing_transports(self):
        ev = _certified_evidence()
        ev.lifetimeTransports = []
        read = self._assert_named_failure(ev, "transport-leakage")
        self.assertEqual(read.transportCount, 0)
        self.assertTrue(math.isnan(read.transportLeakageMax))

    def test_insufficient_frame_lifetime(self):
        # A candidate seen in ONE cobordism frame has no lifetime across
        # frames: the whitepaper conjunct fails, and it fails for a physical
        # reason about the candidate.
        ev = _certified_evidence()
        ev.frameLifetime = 1.0
        read = self._assert_named_failure(ev, "persistence")
        self.assertEqual(read.frameLifetime, 1.0)

    def test_missing_frame_lifetime(self):
        ev = _certified_evidence()
        ev.frameLifetime = NAN
        self._assert_named_failure(ev, "persistence")

    def test_low_frame_overlap(self):
        ev = _certified_evidence()
        ev.frameMinOverlap = 0.2
        self._assert_named_failure(ev, "persistence")

    def test_single_resolution_read_is_not_a_structural_persistence_failure(
            self):
        # #808 negative control: at ONE modularity resolution the slice
        # lifetime is identically 1.  That used to make `persistence`
        # structurally unpassable "for a reason that is not physics"; the
        # gate now reads the COBORDISM-FRAME lifetime, so the same candidate
        # certifies while its resolution-slice numbers stay reported.
        ev = _certified_evidence()
        ev.persistenceLifetime = 1.0
        ev.persistenceMinOverlap = 1.0
        read = self.pc.classifyQuark(ev)
        self.assertEqual(read.classification, "quark")
        self.assertNotIn("persistence", read.failedCertificates)
        self.assertEqual(read.persistenceLifetime, 1.0)
        self.assertEqual(read.frameLifetime, 3.0)

    def test_modularity_resolution_lifetime_never_vetoes(self):
        # The same statement in the other direction: no resolution-slice
        # value, however bad, can veto an otherwise certified fiber.
        for lifetime, overlap in ((NAN, NAN), (0.0, 0.0), (1.0, 0.1)):
            with self.subTest(lifetime=lifetime):
                ev = _certified_evidence()
                ev.persistenceLifetime = lifetime
                ev.persistenceMinOverlap = overlap
                read = self.pc.classifyQuark(ev)
                self.assertEqual(read.classification, "quark")

    def test_low_localization(self):
        cfg = obs.ParticleClustersConfig()
        cfg.minLocalization = 0.9  # fixture band carries 0.5
        pc = obs.ParticleClusters(cfg)
        read = pc.classifyQuark(_certified_evidence())
        self.assertIn("localization", read.failedCertificates)
        self.assertEqual(read.classification, "none")

    def test_missing_refinement_stability(self):
        ev = _certified_evidence()
        ev.refinementOverlap = NAN
        self._assert_named_failure(ev, "refinement-stability")

    def test_unstable_refinement(self):
        ev = _certified_evidence()
        ev.refinementOverlap = 0.3
        self._assert_named_failure(ev, "refinement-stability")

    def test_gap_closed_winding_family_invalidates(self):
        ev = _certified_evidence()
        conn = obs.FiberConnection()
        A, B = _unit_fiber(1, 3), _unit_fiber(11, 3)
        family = _winding_family(conn, A, B)
        bad = conn.transport(A, B, np.diag([0.1, 1.0, 1.0]))
        ev.winding = conn.closedFamilyWinding(family + [bad])
        read = self._assert_named_failure(ev, "winding")
        self.assertIsNone(read.determinantWinding)
        self.assertIsNone(read.baryonFlux)


# =========================================================================== #
# the two-quark anti-triplet is not an antiquark
# =========================================================================== #
class TestAntiTriplet(unittest.TestCase):
    def test_anti_triplet_not_mislabeled_antiquark(self):
        # Two quarks in the Lambda^2 C^3 anti-triplet: the COLOR
        # representation looks like an antiquark (3bar) and the tube even
        # carries nu = -1 here — but total occupation is TWO and parity is
        # EVEN, and the classifier refuses on exactly those channels.
        pc = obs.ParticleClusters()
        ev = _certified_evidence(turns=-1, occupations=(1.0, 1.0, 0.0))
        read = pc.classifyQuark(ev)
        self.assertEqual(read.classification, "none")
        self.assertEqual(read.exteriorParity, +1)
        self.assertAlmostEqual(read.occupationTotal, 2.0, delta=MACHINE)
        self.assertIn("parity-odd", read.failedCertificates)
        self.assertIn("occupation-one", read.failedCertificates)
        # the color-alone channels would NOT have refused:
        self.assertNotIn("color-rank-three", read.failedCertificates)
        self.assertNotIn("winding", read.failedCertificates)

    def test_top_wedge_triple_occupation_is_not_a_quark(self):
        # N = 3 (odd parity!) still fails: single-fermion occupation is a
        # separate certificate from parity.
        pc = obs.ParticleClusters()
        ev = _certified_evidence(occupations=(1.0, 1.0, 1.0))
        read = pc.classifyQuark(ev)
        self.assertEqual(read.exteriorParity, -1)
        self.assertIn("occupation-one", read.failedCertificates)
        self.assertNotIn("parity-odd", read.failedCertificates)
        self.assertEqual(read.classification, "none")


# =========================================================================== #
# conjugate-pair conservation
# =========================================================================== #
class TestConjugatePair(unittest.TestCase):
    def setUp(self):
        self.pc = obs.ParticleClusters()

    def test_certified_conjugate_pair_conserves(self):
        quark = self.pc.classifyQuark(_certified_evidence(turns=1))
        anti = self.pc.classifyQuark(_certified_evidence(turns=-1))
        pair = self.pc.conjugatePair(quark, anti)
        self.assertEqual(pair.totalWinding, 0)
        self.assertEqual(pair.totalBaryonFlux, 0.0)
        self.assertEqual(pair.totalParity, +1)
        self.assertTrue(pair.parityEven)
        self.assertTrue(pair.conserved)
        self.assertEqual(pair.failedCertificates, [])
        self.assertTrue(pair.certificate.holds())

    def test_singular_path_returns_unknown_flux(self):
        quark = self.pc.classifyQuark(_certified_evidence(turns=1))
        ev = _certified_evidence(turns=-1)
        conn = obs.FiberConnection()
        A, B = _unit_fiber(1, 3), _unit_fiber(11, 3)
        bad = conn.transport(A, B, np.diag([0.1, 1.0, 1.0]))
        ev.winding = conn.closedFamilyWinding(
            _winding_family(conn, A, B, turns=-1) + [bad])
        singular = self.pc.classifyQuark(ev)
        self.assertIsNone(singular.determinantWinding)
        pair = self.pc.conjugatePair(quark, singular)
        self.assertIsNone(pair.totalWinding)
        self.assertIsNone(pair.totalBaryonFlux)  # UNKNOWN, never zero
        self.assertFalse(pair.conserved)
        self.assertIn("winding-second", pair.failedCertificates)
        self.assertFalse(pair.certificate.holds())

    def test_non_conjugate_pair_fails_conservation(self):
        quark = self.pc.classifyQuark(_certified_evidence(turns=1))
        pair = self.pc.conjugatePair(quark, quark)
        self.assertEqual(pair.totalWinding, 2)
        self.assertAlmostEqual(pair.totalBaryonFlux, 2.0 / 3.0,
                               delta=MACHINE)
        self.assertFalse(pair.conserved)
        self.assertIn("winding-conservation", pair.failedCertificates)

    def test_uncertified_parity_leaves_total_parity_unknown(self):
        quark = self.pc.classifyQuark(_certified_evidence(turns=1))
        ev = _certified_evidence(turns=-1)
        ev.parityRead = qm.WickCertificateRead()
        anti = self.pc.classifyQuark(ev)
        pair = self.pc.conjugatePair(quark, anti)
        self.assertEqual(pair.totalParity, 0)
        self.assertFalse(pair.parityEven)
        self.assertIn("parity-second", pair.failedCertificates)
        self.assertFalse(pair.conserved)

    def test_pair_of_odd_clusters_is_even(self):
        # whitepaper parity table: quark + antiquark -> even composite
        quark = self.pc.classifyQuark(_certified_evidence(turns=1))
        anti = self.pc.classifyQuark(_certified_evidence(turns=-1))
        self.assertEqual(quark.exteriorParity * anti.exteriorParity, +1)
        self.assertEqual(self.pc.conjugatePair(quark, anti).totalParity, +1)


# =========================================================================== #
# the emergent flavor doublet (no requested dimension)
# =========================================================================== #
class TestFlavorDoublet(unittest.TestCase):
    def setUp(self):
        self.pc = obs.ParticleClusters()

    def test_planted_doublet_emerges(self):
        read = self.pc.flavorDoubletSearch(_doublet_frames())
        self.assertTrue(read.found)
        self.assertEqual(read.rank, 2)
        self.assertEqual(read.framesTracked, 3)
        self.assertEqual(read.twoStateCount, 1)
        self.assertAlmostEqual(read.minContinuationOverlap, 1.0,
                               delta=MACHINE)
        self.assertTrue(read.certificate.holds())
        # the search never requested a dimension: every stable rank is
        # reported, not only the two-state one
        self.assertEqual(sorted(read.stableSubclassRanks), [1, 2, 3])

    def test_no_two_state_subclass_is_unknown(self):
        read = self.pc.flavorDoubletSearch(
            _doublet_frames(ranks=(1, 3)))
        self.assertFalse(read.found)
        self.assertIn("flavor-doublet", read.failedCertificates)
        self.assertEqual(read.invalidationReason,
                         "no-stable-two-state-subclass")
        self.assertFalse(read.certificate.holds())

    def test_doublet_dropping_out_is_unstable(self):
        read = self.pc.flavorDoubletSearch(_doublet_frames(drop_rank2_at=2))
        self.assertFalse(read.found)
        self.assertEqual(read.invalidationReason,
                         "no-stable-two-state-subclass")

    def test_gap_closing_doublet_is_uncertified(self):
        # an unaccepted (gap-closed) band anywhere on the track breaks the
        # certified continuation — the #769 semantics
        read = self.pc.flavorDoubletSearch(
            _doublet_frames(unaccept_rank2_at=1))
        self.assertFalse(read.found)

    def test_single_frame_is_insufficient(self):
        read = self.pc.flavorDoubletSearch(_doublet_frames(frames=1))
        self.assertFalse(read.found)
        self.assertEqual(read.invalidationReason, "insufficient-frames")

    def test_ambiguous_two_doublets_stay_uncertified(self):
        read = self.pc.flavorDoubletSearch(
            _doublet_frames(extra_rank2=True))
        self.assertFalse(read.found)
        self.assertEqual(read.twoStateCount, 2)
        self.assertEqual(read.invalidationReason,
                         "ambiguous-two-state-subclasses")

    def test_merging_chains_invalidate_each_other(self):
        # two rank-2 bands spanning the SAME cells at frame 0 both best-
        # match the single frame-1 band: the continuation is ambiguous
        f0 = _band_read([_unit_fiber(300, 2),
                         _fiber([[300], [301]], np.eye(2)[:, ::-1])])
        f1 = _band_read([_unit_fiber(300, 2)])
        read = self.pc.flavorDoubletSearch([f0, f1])
        self.assertFalse(read.found)

    def test_doublet_carries_the_recorded_trivialization(self):
        read = self.pc.flavorDoubletSearch(_doublet_frames(base=400))
        self.assertTrue(read.found)
        self.assertEqual(read.doublet.rank(), 2)
        self.assertEqual(read.doublet.cellVertices(), [[420], [421]])

    def test_search_is_deterministic(self):
        a = self.pc.flavorDoubletSearch(_doublet_frames())
        b = self.pc.flavorDoubletSearch(_doublet_frames())
        self.assertEqual(a.found, b.found)
        self.assertEqual(a.stableSubclassRanks, b.stableSubclassRanks)
        self.assertEqual(a.minContinuationOverlap, b.minContinuationOverlap)


# =========================================================================== #
# isospin, Gauss charge, and the proposed u/d identification
# =========================================================================== #
class TestIsospinCharge(unittest.TestCase):
    def setUp(self):
        self.pc = obs.ParticleClusters()

    def test_u_fixture(self):
        ev = _certified_evidence(with_flavor=True, occupancy=[1.0, 0.0],
                                 with_charge=2.0 / 3.0)
        read = self.pc.classifyQuark(ev)
        self.assertEqual(read.classification, "quark")
        self.assertEqual(read.isospin, 0.5)
        self.assertAlmostEqual(read.electricFlux, 2.0 / 3.0, delta=1e-9)
        self.assertAlmostEqual(read.baryonFlux, 1.0 / 3.0, delta=MACHINE)
        self.assertTrue(read.udIdentificationProposed)
        self.assertNotIn("ud-identification", read.failedCertificates)
        self.assertEqual(read.failedCertificates, [])

    def test_d_fixture(self):
        ev = _certified_evidence(with_flavor=True, occupancy=[0.0, 1.0],
                                 with_charge=-1.0 / 3.0)
        read = self.pc.classifyQuark(ev)
        self.assertEqual(read.isospin, -0.5)
        self.assertAlmostEqual(read.electricFlux, -1.0 / 3.0, delta=1e-9)
        self.assertTrue(read.udIdentificationProposed)

    def test_missing_doublet_yields_unknown_flavor_and_charge(self):
        # even with a CONSISTENT Gauss read, the quark charge stays
        # unknown without the doublet (ticket acceptance)
        ev = _certified_evidence(with_charge=2.0 / 3.0)
        read = self.pc.classifyQuark(ev)
        self.assertIsNone(read.isospin)
        self.assertIsNone(read.electricFlux)
        self.assertIn("flavor-doublet", read.failedCertificates)
        self.assertEqual(read.classification, "quark")  # quark-ness intact

    def test_unstable_doublet_yields_unknown_flavor_and_charge(self):
        ev = _certified_evidence(with_charge=2.0 / 3.0)
        ev.flavor = self.pc.flavorDoubletSearch(
            _doublet_frames(drop_rank2_at=1))
        ev.doubletOccupancy = np.array([1.0, 0.0], dtype=complex)
        read = self.pc.classifyQuark(ev)
        self.assertIsNone(read.isospin)
        self.assertIsNone(read.electricFlux)
        self.assertIn("flavor-doublet", read.failedCertificates)

    def test_superposition_occupancy_yields_unknown_isospin(self):
        ev = _certified_evidence(with_flavor=True,
                                 occupancy=[1.0, 1.0])
        read = self.pc.classifyQuark(ev)
        self.assertIsNone(read.isospin)
        self.assertIn("isospin", read.failedCertificates)

    def test_inconsistent_gauss_yields_unknown_charge(self):
        ev = _certified_evidence(with_flavor=True, occupancy=[1.0, 0.0])
        ev.charge = self.pc.gaussFluxConsistency(
            [complex(2.0 / 3.0), complex(1.0)])
        read = self.pc.classifyQuark(ev)
        self.assertIsNone(read.electricFlux)
        self.assertIn("gauss-consistency", read.failedCertificates)
        # Q = I3 + B/2 is NOT tested without a certified charge
        self.assertNotIn("ud-identification", read.failedCertificates)
        self.assertFalse(read.udIdentificationProposed)

    def test_violated_ud_relation_is_named(self):
        # occupancy says d (I3 = -1/2) but the Gauss flux says +2/3: the
        # proposed identification fails BY NAME; the independently
        # certified fields stay reported
        ev = _certified_evidence(with_flavor=True, occupancy=[0.0, 1.0],
                                 with_charge=2.0 / 3.0)
        read = self.pc.classifyQuark(ev)
        self.assertEqual(read.isospin, -0.5)
        self.assertAlmostEqual(read.electricFlux, 2.0 / 3.0, delta=1e-9)
        self.assertIn("ud-identification", read.failedCertificates)
        self.assertFalse(read.udIdentificationProposed)

    def test_declared_orientation_is_recorded(self):
        ev = _certified_evidence(with_flavor=True, occupancy=[0.0, 1.0],
                                 orientation=-1, with_charge=2.0 / 3.0)
        read = self.pc.classifyQuark(ev)
        # orientation -1: member 2 carries +1/2 under the declared
        # convention -> the SAME occupancy now reads as the u member
        self.assertEqual(read.isospin, 0.5)
        self.assertEqual(read.doubletOrientation, -1)
        self.assertTrue(read.udIdentificationProposed)

    def test_ud_not_tested_without_baryon_flux(self):
        ev = _certified_evidence(with_flavor=True, occupancy=[1.0, 0.0],
                                 with_charge=2.0 / 3.0)
        conn = obs.FiberConnection()
        A, B = _unit_fiber(1, 3), _unit_fiber(11, 3)
        segment = [_phase_link(conn, A, B, p) for p in (0.0, 0.3)]
        ev.winding = conn.openSegmentWinding(segment,
                                             obs.WindingClosureSpec())
        read = self.pc.classifyQuark(ev)
        self.assertIsNone(read.baryonFlux)
        self.assertNotIn("ud-identification", read.failedCertificates)
        self.assertFalse(read.udIdentificationProposed)


# =========================================================================== #
# the reused Gauss-flux read on nested enclosing surfaces
# =========================================================================== #
class TestGaussFlux(unittest.TestCase):
    def setUp(self):
        self.pc = obs.ParticleClusters()

    def test_nested_surfaces_consistent_read(self):
        st, F, surfaces = _gauss_fixture(2.0 / 3.0)
        read = self.pc.gaussFluxOnSurfaces(st, F, surfaces, True)
        self.assertTrue(read.consistent)
        self.assertAlmostEqual(read.electricFlux, 2.0 / 3.0, delta=1e-9)
        self.assertEqual(len(read.fluxes), 2)
        self.assertLess(read.maxDeviation, 1e-12)
        self.assertTrue(read.certificate.holds())
        self.assertEqual(read.surfaceVertexCounts, [1, 4])

    def test_inconsistent_flux_is_unknown(self):
        st, F, surfaces = _gauss_fixture(2.0 / 3.0)
        # perturb one electric plaquette between the two surfaces
        es = cob.EigenstateSynthesis(st, 2)
        n = es.order()
        # find a cell seen by exactly one surface
        for c in range(n):
            probe = [0j] * n
            probe[c] = 1.0 + 0j
            s1 = es.gaussLawCharge(probe, surfaces[0], True)
            s2 = es.gaussLawCharge(probe, surfaces[1], True)
            if abs(s1 - s2) > 0.5:
                F2 = list(F)
                F2[c] += 1.0
                break
        read = self.pc.gaussFluxOnSurfaces(st, F2, surfaces, True)
        self.assertFalse(read.consistent)
        self.assertIsNone(read.electricFlux)
        self.assertIn("gauss-consistency", read.failedCertificates)
        self.assertFalse(read.certificate.holds())

    def test_all_spacelike_complex_reads_certified_zero(self):
        st = _from_simplices(7, _TETRA_CHAIN, timelike=False)
        es = cob.EigenstateSynthesis(st, 2)
        surfaces = obs.ParticleClusters.nestedEnclosures(st, [0], 2)
        F = [0.7 + 0j] * es.order()
        read = self.pc.gaussFluxOnSurfaces(st, F, surfaces, True)
        self.assertTrue(read.consistent)
        self.assertEqual(read.electricFlux, 0.0)  # a CERTIFIED zero
        self.assertEqual(read.failedCertificates, [])

    def test_single_surface_makes_no_consistency_claim(self):
        st, F, surfaces = _gauss_fixture(2.0 / 3.0)
        read = self.pc.gaussFluxOnSurfaces(st, F, [surfaces[0]], True)
        self.assertFalse(read.consistent)
        self.assertIsNone(read.electricFlux)
        self.assertIn("gauss-consistency", read.failedCertificates)

    def test_exact_field_strength_has_zero_total_flux(self):
        # topological protection: F = dA is exact, so the FULL (electric +
        # magnetic) closed-surface flux vanishes on every surface
        st = _from_simplices(7, _TETRA_CHAIN)
        es1 = cob.EigenstateSynthesis(st, 1)
        es2 = cob.EigenstateSynthesis(st, 2)
        rng = np.random.default_rng(7)
        A = [complex(v) for v in rng.normal(size=es1.order())]
        F = es2.curvatureFromConnection(A)
        surfaces = obs.ParticleClusters.nestedEnclosures(st, [0], 2)
        read = self.pc.gaussFluxOnSurfaces(st, F, surfaces,
                                           electric_only=False)
        self.assertTrue(read.consistent)
        self.assertAlmostEqual(read.electricFlux, 0.0, delta=1e-12)

    def test_pure_combination_equals_spacetime_path(self):
        st, F, surfaces = _gauss_fixture(0.25)
        es = cob.EigenstateSynthesis(st, 2)
        fluxes = [es.gaussLawCharge(F, s, True) for s in surfaces]
        via_st = self.pc.gaussFluxOnSurfaces(st, F, surfaces, True)
        pure = self.pc.gaussFluxConsistency(fluxes, [1, 4], True)
        self.assertEqual(via_st.consistent, pure.consistent)
        self.assertEqual(via_st.electricFlux, pure.electricFlux)
        self.assertEqual(via_st.fluxes, pure.fluxes)

    def test_nested_enclosures_are_strictly_growing_here(self):
        st = _from_simplices(7, _TETRA_CHAIN)
        sets_ = obs.ParticleClusters.nestedEnclosures(st, [0], 3)
        self.assertEqual(len(sets_), 3)
        self.assertEqual(sets_[0], [0])
        for a, b in zip(sets_, sets_[1:]):
            self.assertTrue(set(a) < set(b))

    def test_nested_enclosures_validates(self):
        st = _from_simplices(7, _TETRA_CHAIN)
        with self.assertRaises(ValueError):
            obs.ParticleClusters.nestedEnclosures(st, [], 2)
        with self.assertRaises(ValueError):
            obs.ParticleClusters.nestedEnclosures(st, [0], 0)
        with self.assertRaises(ValueError):
            obs.ParticleClusters.nestedEnclosures(st, [999], 2)

    def test_imaginary_leakage_is_reported_not_discarded(self):
        read = self.pc.gaussFluxConsistency([complex(0.5, 0.3),
                                             complex(0.5, 0.3)])
        self.assertFalse(read.consistent)  # |Im| above tolerance
        self.assertAlmostEqual(read.imagLeakage, 0.3, delta=MACHINE)
        self.assertIsNone(read.electricFlux)

    def test_gauss_read_is_read_only(self):
        st, F, surfaces = _gauss_fixture(2.0 / 3.0)
        before = st.metricRevisionKey()
        self.pc.gaussFluxOnSurfaces(st, F, surfaces, True)
        self.assertEqual(st.metricRevisionKey(), before)


# =========================================================================== #
# relabeling / refinement / gauge invariance
# =========================================================================== #
class TestRelabelRefinementGauge(unittest.TestCase):
    def setUp(self):
        self.pc = obs.ParticleClusters()
        self.delta = staticmethod(obs.ObservableGates.report_delta)

    def test_relabeling_preserves_accepted_classification(self):
        base = self.pc.classifyQuark(
            _certified_evidence(with_flavor=True, occupancy=[1.0, 0.0],
                                with_charge=2.0 / 3.0))
        shifted_ev = _certified_evidence(band_base=1001, with_flavor=True,
                                         occupancy=[1.0, 0.0],
                                         with_charge=2.0 / 3.0)
        shifted = self.pc.classifyQuark(shifted_ev)
        self.assertEqual(
            obs.ObservableGates.report_delta(base.toRecord(),
                                             shifted.toRecord()), 0.0)

    def test_vertex_relabeling_of_the_gauss_complex(self):
        st, F, surfaces = _gauss_fixture(2.0 / 3.0)
        ids = [i + 500 for i in range(7)]
        st2, F2, surfaces2 = _gauss_fixture(2.0 / 3.0, ids=ids)
        a = self.pc.gaussFluxOnSurfaces(st, F, surfaces, True)
        b = self.pc.gaussFluxOnSurfaces(st2, F2, surfaces2, True)
        self.assertEqual(a.consistent, b.consistent)
        self.assertAlmostEqual(a.electricFlux, b.electricFlux, delta=1e-9)

    def test_refinement_preserves_accepted_classification(self):
        # the refined band adds cells while keeping the original subspace:
        # the measured overlap feeds the refinement certificate
        band = _unit_fiber(1, 3)
        refined = _fiber([[1], [2], [3], [77]],
                         np.vstack([np.eye(3), np.zeros((1, 3))]))
        overlap = obs.SpectralFiber.overlap(band, refined).subspaceOverlap
        ev = _certified_evidence()
        ev.refinementOverlap = overlap
        read = self.pc.classifyQuark(ev)
        self.assertAlmostEqual(overlap, 1.0, delta=1e-12)
        self.assertEqual(read.classification, "quark")
        self.assertNotIn("refinement-stability", read.failedCertificates)

    def test_in_band_gauge_rotation_leaves_the_verdict(self):
        # an SU(3) in-band frame change of the anchored frame leaves the
        # anchor profile (hence the classification evidence) invariant
        rng = np.random.default_rng(11)
        w = np.array([2.0, 0.5, 1.25])
        z = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
        phi = obs.ColorAnchor.orthonormalizeFrame(z, w)
        tri = [tessera.OrientedTriangle([0, 1, 2], [1, 1, 1])]
        p1 = obs.ColorAnchor(tri).evaluate(phi, w)
        # a special-unitary in-band rotation
        g = np.linalg.qr(rng.normal(size=(3, 3))
                         + 1j * rng.normal(size=(3, 3)))[0]
        g = g / np.linalg.det(g) ** (1.0 / 3.0)
        p2 = obs.ColorAnchor(tri).evaluate(phi @ g, w)
        r1 = self.pc.classifyQuark(_certified_evidence(anchor=p1))
        r2 = self.pc.classifyQuark(_certified_evidence(anchor=p2))
        self.assertEqual(r1.classification, r2.classification)
        self.assertAlmostEqual(r1.triangleAnchorScore,
                               r2.triangleAnchorScore, delta=1e-12)

    def test_transport_order_does_not_matter_for_leakage(self):
        ev = _certified_evidence()
        base = self.pc.classifyQuark(ev)
        ev.lifetimeTransports = list(reversed(list(ev.lifetimeTransports)))
        permuted = self.pc.classifyQuark(ev)
        self.assertEqual(base.transportLeakageMax,
                         permuted.transportLeakageMax)
        self.assertEqual(base.classification, permuted.classification)


# =========================================================================== #
# tracking, checkpointing, and the #764 cache
# =========================================================================== #
class TestTrackingCheckpointCache(unittest.TestCase):
    def setUp(self):
        self.pc = obs.ParticleClusters()

    def test_track_candidates_across_frames(self):
        a1 = _certified_evidence(band_base=1)
        a2 = _certified_evidence(band_base=101)
        b1 = _certified_evidence(band_base=1)
        matches = obs.ParticleClusters.trackCandidates([a1, a2], [b1])
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].fromIndex, 0)
        self.assertEqual(matches[0].toIndex, 0)
        self.assertTrue(matches[0].certifiedContinuation)

    def test_record_roundtrip_is_exact(self):
        read = self.pc.classifyQuark(
            _certified_evidence(with_flavor=True, occupancy=[1.0, 0.0],
                                with_charge=2.0 / 3.0))
        rec = read.toRecord()
        back = obs.QuarkRead.fromRecord(rec)
        self.assertEqual(
            obs.ObservableGates.report_delta(rec, back.toRecord()), 0.0)

    def test_unknown_fields_serialize_as_null_never_zero(self):
        read = self.pc.classifyQuark(obs.QuarkCandidateEvidence())
        rec = read.toRecord()
        self.assertIsNone(rec["baryon_flux"])
        self.assertIsNone(rec["isospin"])
        self.assertIsNone(rec["electric_flux"])
        self.assertIsNone(rec["determinant_winding"])
        self.assertEqual(rec["exterior_parity"], 0)
        self.assertTrue(math.isnan(rec["triangle_anchor_score"]))

    def test_full_evidence_is_checkpointed(self):
        read = self.pc.classifyQuark(_certified_evidence())
        rec = read.toRecord()
        for key in ("component_hash", "winding_closure",
                    "failed_certificates", "thresholds", "certificate",
                    "transport_leakage_max", "persistence_lifetime",
                    "localization", "refinement_overlap",
                    "occupation_total", "confidence"):
            self.assertIn(key, rec)
        self.assertEqual(rec["classification"], "quark")
        self.assertEqual(rec["winding_closure"], "closed-family")

    def test_from_record_rejects_unknown_schema(self):
        read = self.pc.classifyQuark(_certified_evidence())
        rec = read.toRecord()
        rec["schema_version"] = 99
        with self.assertRaises(ValueError):
            obs.QuarkRead.fromRecord(rec)

    def test_describe_smoke(self):
        read = self.pc.classifyQuark(_certified_evidence())
        text = read.describe()
        self.assertIn("quark", text)
        self.assertIn("nu=1", text)

    def _spacetime_backed_evidence(self):
        # a real spacetime band so the cache has a live star to publish
        st = _from_simplices(7, _TETRA_CHAIN, timelike=False)
        tracker = obs.SpectralFiberTracker(st)
        bands = tracker.enumerateBands(list(range(7)), 0)
        fiber = bands.fibers[0]
        ev = _certified_evidence()
        ev.colorBand = fiber  # rank/acceptance of the REAL band applies
        return st, ev

    def test_cached_classification_equals_cold(self):
        st, ev = self._spacetime_backed_evidence()
        cache = cob.AnalyticCache(st)
        cold = self.pc.classifyQuark(ev)
        first = self.pc.classifyQuarkCached(cache, ev)
        served = self.pc.classifyQuarkCached(cache, ev)
        self.assertEqual(
            obs.ObservableGates.report_delta(cold.toRecord(),
                                             first.toRecord()), 0.0)
        self.assertEqual(
            obs.ObservableGates.report_delta(cold.toRecord(),
                                             served.toRecord()), 0.0)
        self.assertGreaterEqual(cache.hits, 1)

    def test_touched_star_invalidates_only_the_touching_candidate(self):
        st, ev = self._spacetime_backed_evidence()
        cache = cob.AnalyticCache(st)
        self.pc.classifyQuarkCached(cache, ev)
        self.assertEqual(cache.size, 1)
        star = cob.TouchedStar()
        star.addChangedEdge(0, 1)  # touches the band's support
        cache.publish(star)
        self.assertEqual(cache.size, 0)

    def test_disjoint_star_keeps_the_entry(self):
        st, ev = self._spacetime_backed_evidence()
        cache = cob.AnalyticCache(st)
        self.pc.classifyQuarkCached(cache, ev)
        star = cob.TouchedStar()
        star.addChangedEdge(9001, 9002)  # disjoint from the band support
        cache.publish(star)
        self.assertEqual(cache.size, 1)

    def test_changed_evidence_never_serves_a_stale_read(self):
        st, ev = self._spacetime_backed_evidence()
        cache = cob.AnalyticCache(st)
        quark = self.pc.classifyQuarkCached(cache, ev)
        conn = obs.FiberConnection()
        A, B = _unit_fiber(1, 3), _unit_fiber(11, 3)
        ev.winding = conn.closedFamilyWinding(
            _winding_family(conn, A, B, turns=-1))
        anti = self.pc.classifyQuarkCached(cache, ev)
        self.assertNotEqual(quark.determinantWinding,
                            anti.determinantWinding)

    def test_different_thresholds_have_different_fingerprints(self):
        ev = _certified_evidence()
        loose = obs.ParticleClusters()
        strict_cfg = obs.ParticleClustersConfig()
        strict_cfg.minAnchorScore = 0.99
        strict = obs.ParticleClusters(strict_cfg)
        self.assertNotEqual(loose.evidenceFingerprint(ev),
                            strict.evidenceFingerprint(ev))

    def test_the_stability_window_is_part_of_the_cache_key(self):
        # #808: the frame lifetime and the two stability windows are
        # DECISION evidence, so a change in any of them must recompute
        # rather than serve another window's verdict.
        base = _certified_evidence()
        fingerprint = self.pc.evidenceFingerprint(base)
        for mutate in (
                lambda e: setattr(e, "frameLifetime", 9.0),
                lambda e: setattr(e, "frameMinOverlap", 0.6),
                lambda e: setattr(e, "colorBandFrames",
                                  list(e.colorBandFrames)[:2]),
                lambda e: setattr(e, "anchorFrames",
                                  list(e.anchorFrames)[:2]),
        ):
            with self.subTest(mutate=mutate):
                ev = _certified_evidence()
                mutate(ev)
                self.assertNotEqual(fingerprint,
                                    self.pc.evidenceFingerprint(ev))

    def test_a_changed_stability_window_never_serves_a_stale_read(self):
        st, ev = self._spacetime_backed_evidence()
        cache = cob.AnalyticCache(st)
        stable = self.pc.classifyQuarkCached(cache, ev)
        ev.colorBandFrames = [ev.colorBandFrames[0], _unit_fiber(1, 2)]
        unstable = self.pc.classifyQuarkCached(cache, ev)
        self.assertNotIn("color-rank-stability", stable.failedCertificates)
        self.assertIn("color-rank-stability", unstable.failedCertificates)
        cold = obs.ParticleClusters().classifyQuark(ev)
        self.assertEqual(obs.ObservableGates.report_delta(
            cold.toRecord(), unstable.toRecord()), 0.0)


# =========================================================================== #
# the emergence objective stays particle-blind; performance contract
# =========================================================================== #

#: The ONE translation unit under an objective home that is allowed to name
#: the particle classifier: the #776 post-hoc analysis overlay, whose whole
#: job is to DRIVE it after a move has already been accepted. The design
#: spec's source-layout row puts the orchestration on
#: `include/cobordism/MultiCobordism.h`, so it necessarily lives here. It is
#: exempted by NAME, never by pattern, and the objective's own sources below
#: are checked separately and have no exemption at all.
_OVERLAY_TRANSLATION_UNIT = "RecursiveFiberSimulation.cpp"

#: Where the emergence objective and its gradient actually live. Nothing on
#: this list may name a derived observable under any circumstances.
_OBJECTIVE_SOURCES = ("include/cobordism/MultiCobordism.h",
                      "src/cobordism/MultiCobordism.cpp")


def _objective_source_offenders(needles):
    """Files under an objective home that name one of `needles`.

    The #776 overlay translation unit is exempt (see
    `_OVERLAY_TRANSLATION_UNIT`); every other file, including the two files
    the objective itself lives in, is checked.
    """
    objective_homes = [REPO_ROOT / "src" / "cobordism",
                       REPO_ROOT / "include" / "cobordism",
                       REPO_ROOT / "src" / "rl",
                       REPO_ROOT / "include" / "rl",
                       REPO_ROOT / "src" / "simulations",
                       REPO_ROOT / "include" / "simulations"]
    offenders = []
    for home in objective_homes:
        if not home.exists():
            continue
        for path in home.rglob("*"):
            if path.suffix not in (".h", ".cpp", ".cu", ".hpp"):
                continue
            if path.name == _OVERLAY_TRANSLATION_UNIT:
                continue
            text = path.read_text(errors="ignore")
            if any(needle in text for needle in needles):
                offenders.append(str(path))
    return offenders


class TestObjectiveGuardAndBenchmark(unittest.TestCase):
    def test_no_quark_quantity_enters_the_emergence_objective(self):
        # The emergence objective lives in the cobordism optimizer and the
        # RL harness; neither may reference the particle classifier.
        self.assertEqual(
            _objective_source_offenders(("ParticleClusters", "QuarkRead")), [])

    def test_the_objective_sources_themselves_name_nothing_derived(self):
        # The complement of the overlay exemption: the two files the
        # objective and its gradient live in are checked with NO exemption,
        # against the whole derived vocabulary of the epic.
        needles = ("ParticleClusters", "QuarkRead", "GluonRead", "MesonRead",
                   "DiquarkRead", "BaryonRead", "SpectralFiber",
                   "FiberConnection", "ColorFiber", "ColorAnchor",
                   "ExchangeHolonomy", "PersistentModularity",
                   "RecursiveQuotient", "WilsonHolonomyRead",
                   "DeterminantWindingRead", "classifyQuark", "classifyBaryon")
        offenders = []
        for relative in _OBJECTIVE_SOURCES:
            path = REPO_ROOT / relative
            if not path.exists():
                continue
            text = path.read_text(errors="ignore")
            for needle in needles:
                if needle in text:
                    offenders.append("%s names %s" % (relative, needle))
        self.assertEqual(offenders, [])

    def test_classification_is_read_only_on_the_spacetime(self):
        st = _from_simplices(7, _TETRA_CHAIN, timelike=False)
        tracker = obs.SpectralFiberTracker(st)
        bands = tracker.enumerateBands(list(range(7)), 0)
        ev = _certified_evidence()
        ev.colorBand = bands.fibers[0]
        before = st.metricRevisionKey()
        obs.ParticleClusters().classifyQuark(ev)
        self.assertEqual(st.metricRevisionKey(), before)

    def test_classification_cost_per_candidate(self):
        # merge-gate benchmark: classification cost per candidate, cold
        # versus cache-served (numbers reported in the PR body)
        pc = obs.ParticleClusters()
        ev = _certified_evidence()
        n = 200
        t0 = time.perf_counter()
        for _ in range(n):
            pc.classifyQuark(ev)
        cold = (time.perf_counter() - t0) / n

        st = _from_simplices(7, _TETRA_CHAIN, timelike=False)
        cache = cob.AnalyticCache(st)
        tracker = obs.SpectralFiberTracker(st)
        ev.colorBand = tracker.enumerateBands(list(range(7)), 0).fibers[0]
        pc.classifyQuarkCached(cache, ev)  # warm
        t0 = time.perf_counter()
        for _ in range(n):
            pc.classifyQuarkCached(cache, ev)
        cached = (time.perf_counter() - t0) / n
        print(f"\n[benchmark] classifyQuark cold: {cold * 1e6:.1f} us; "
              f"cache-served: {cached * 1e6:.1f} us per candidate")
        self.assertLess(cold, 0.05)  # generous ceiling: 50 ms/candidate


# =========================================================================== #
# #774: the even sectors -- quasi-free octet bilinear read and the gluon /
# meson / diquark candidate classifiers.  Fixtures compose the MERGED
# public APIs (#767 ColorFiber, #770 FiberConnection, #773 QuarkReads,
# #780 CovarianceState, #771 LazyFockEngine as the dense oracle).
# =========================================================================== #
def _rank2_state(hole=(0.0, 0.0, 1.0)):
    """N = 2 anti-triplet Slater covariance on three modes:
    Gamma = I - c c^dag (even parity, occupation two)."""
    c = np.asarray(hole, dtype=complex)
    c = c / np.linalg.norm(c)
    return qm.CovarianceState(np.eye(3, dtype=complex) - np.outer(c, c.conj()))


def _gluon_evidence(turns=0, state=None, modes=(0, 1, 2), lifetime=3.0,
                    band_base=1, conn=None):
    """A gluon-candidate evidence bundle: quasi-free octet read of the
    carried state, carried-state Wick parity/occupation, an accepted
    rank-three transport family with certified winding `turns`, and the
    persistence lifetime."""
    conn = conn or obs.FiberConnection()
    A, B = _unit_fiber(band_base, 3), _unit_fiber(band_base + 10, 3)
    family = _winding_family(conn, A, B, turns=turns)
    state = _rank2_state() if state is None else state
    pc = obs.ParticleClusters()
    ev = obs.GluonCandidateEvidence()
    ev.component = obs.ComponentId("1a" * 16, 1)
    ev.bindingComponent = obs.ComponentId("2b" * 16, 2)
    ev.octet = pc.octetBilinearRead(state, list(modes))
    ev.parityRead = state.wickParity()
    ev.occupationRead = state.wickTotalNumber()
    ev.lifetimeTransports = family
    ev.winding = conn.closedFamilyWinding(family)
    ev.persistenceLifetime = lifetime
    # The gated persistence quantity is the COBORDISM-FRAME lifetime (#808);
    # the resolution-slice number travels beside it as a report.
    ev.frameLifetime = lifetime
    return ev


def _meson_evidence(first_turns=1, second_turns=-1, pairing="singlet"):
    """A two-cluster composite bundle: one #773 quark + one antiquark,
    the carried composite occupation, and the pair color bilinear."""
    pc = obs.ParticleClusters()
    first = pc.classifyQuark(_certified_evidence(turns=first_turns))
    second = pc.classifyQuark(
        _certified_evidence(turns=second_turns, band_base=31))
    ev = obs.CompositeCandidateEvidence()
    ev.bindingComponent = obs.ComponentId("3c" * 16, 2)
    ev.first = first
    ev.second = second
    ev.occupationRead = qm.CovarianceState.fromOccupations(
        np.array([1.0, 1.0])).wickTotalNumber()
    if pairing == "singlet":
        ev.colorPairing = np.eye(3, dtype=complex) / math.sqrt(3.0)
    elif pairing == "octet":
        ev.colorPairing = np.asarray(obs.ColorFiber.gellMann(1))
    ev.persistenceLifetime = 3.0
    return ev


def _diquark_evidence(columns=None, second_turns=1):
    """A two-quark composite bundle with the certified anti-triplet wedge
    occupation det(C^dag Gamma C) of the pair's carried Slater state."""
    pc = obs.ParticleClusters()
    ev = obs.CompositeCandidateEvidence()
    ev.bindingComponent = obs.ComponentId("4d" * 16, 2)
    ev.first = pc.classifyQuark(_certified_evidence(turns=1))
    ev.second = pc.classifyQuark(
        _certified_evidence(turns=second_turns, band_base=51))
    C = (np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=complex)
         if columns is None else np.asarray(columns, dtype=complex))
    slater = qm.CovarianceState.fromSlaterFrame(
        np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=complex))
    ev.antiTripletRead = slater.wickGramDeterminant(C, C)
    ev.occupationRead = slater.wickTotalNumber()
    ev.persistenceLifetime = 3.0
    return ev


class TestOctetBilinearRead(unittest.TestCase):
    """Exact small-sector fixtures of the quasi-free octet read."""

    def setUp(self):
        self.pc = obs.ParticleClusters()

    def test_anti_triplet_slater_exact_values(self):
        read = self.pc.octetBilinearRead(_rank2_state(), [0, 1, 2])
        self.assertAlmostEqual(read.occupation, 2.0, delta=MACHINE)
        self.assertEqual(read.subsetParity, +1)
        self.assertAlmostEqual(read.octetWeight, 2.0 / 3.0, delta=MACHINE)
        self.assertAlmostEqual(read.singletWeight, 4.0 / 3.0, delta=MACHINE)
        self.assertAlmostEqual(read.casimir, 3.0, delta=1e-12)
        # C2(3bar) = 4/3 by quartic Wick sums -- exact algebra.
        self.assertAlmostEqual(read.casimirExpectation, 4.0 / 3.0,
                               delta=1e-12)
        self.assertLessEqual(read.octetProjectorResidual, 1e-14)
        self.assertTrue(read.certificate.holds())
        self.assertEqual(read.certificate.grade,
                         cob.CertificateGrade.AlgebraicallyExact)

    def test_fundamental_exact_values(self):
        c = np.array([1.0, 0.0, 0.0], dtype=complex)
        state = qm.CovarianceState(np.outer(c, c.conj()))
        read = self.pc.octetBilinearRead(state, [0, 1, 2])
        self.assertAlmostEqual(read.occupation, 1.0, delta=MACHINE)
        self.assertEqual(read.subsetParity, -1)
        self.assertAlmostEqual(read.octetWeight, 2.0 / 3.0, delta=MACHINE)
        self.assertAlmostEqual(read.singletWeight, 1.0 / 3.0, delta=MACHINE)
        # C2(3) = 4/3 on the fundamental.
        self.assertAlmostEqual(read.casimirExpectation, 4.0 / 3.0,
                               delta=1e-12)

    def test_vacuum_and_full_singlet_read_zero_casimir(self):
        vacuum = qm.CovarianceState(np.zeros((3, 3), dtype=complex))
        read = self.pc.octetBilinearRead(vacuum, [0, 1, 2])
        self.assertEqual(read.occupation, 0.0)
        self.assertEqual(read.subsetParity, +1)
        self.assertEqual(read.octetWeight, 0.0)
        # a vanished excitation is UNKNOWN, never zero
        self.assertTrue(math.isnan(read.casimir))
        self.assertTrue(math.isnan(read.octetProjectorResidual))
        self.assertAlmostEqual(read.casimirExpectation, 0.0, delta=1e-12)

        full = qm.CovarianceState(np.eye(3, dtype=complex))
        read = self.pc.octetBilinearRead(full, [0, 1, 2])
        self.assertAlmostEqual(read.occupation, 3.0, delta=MACHINE)
        self.assertEqual(read.subsetParity, -1)
        self.assertAlmostEqual(read.octetWeight, 0.0, delta=1e-15)
        # the fully occupied top wedge is a color SINGLET: C2 = 0 exactly.
        self.assertAlmostEqual(read.casimirExpectation, 0.0, delta=1e-12)

    def test_bilinear_is_the_transposed_submatrix(self):
        rng = np.random.default_rng(81)
        z = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
        q = np.linalg.qr(z)[0]
        gamma = q @ np.diag([0.9, 0.4, 0.1]) @ q.conj().T
        state = qm.CovarianceState(gamma)
        read = self.pc.octetBilinearRead(state, [0, 1, 2])
        self.assertTrue(np.array_equal(read.bilinear, gamma.T))

    def test_split_delegates_to_color_fiber_bitwise(self):
        read = self.pc.octetBilinearRead(_rank2_state((0.3, 0.5, 0.9)),
                                         [0, 1, 2])
        want = obs.ColorFiber.octetRead(read.bilinear)
        self.assertEqual(read.octetWeight, want.octet)
        self.assertEqual(read.singletWeight, want.singlet)
        self.assertTrue(np.array_equal(
            read.octetComponent,
            obs.ColorFiber.tracelessPart(read.bilinear)))
        self.assertAlmostEqual(
            read.casimir, obs.ColorFiber.adjointCasimir(read.octetComponent),
            delta=1e-15)

    def test_gell_mann_components_reconstruct_the_bilinear(self):
        read = self.pc.octetBilinearRead(_rank2_state((0.2, 0.7, 0.4)),
                                         [0, 1, 2])
        m = np.asarray(read.bilinear)
        recon = (np.trace(m) / 3.0) * np.eye(3, dtype=complex)
        for a in range(1, 9):
            recon = recon + read.gellMannComponents[a - 1] \
                * np.asarray(obs.ColorFiber.gellMann(a))
        self.assertLessEqual(np.max(np.abs(recon - m)), 1e-13)

    def test_dense_lazy_oracle_cross_validation(self):
        # #771 is the dense oracle: the SAME Slater state through the lazy
        # engine's exact covariance read gives the SAME octet read.
        rng = np.random.default_rng(82)
        orbitals = rng.normal(size=(3, 2)) + 1j * rng.normal(size=(3, 2))
        eng = qm.LazyFockEngine(3)
        wedge = eng.wedgeState([0, 1, 2], orbitals)
        gamma = np.asarray(eng.covarianceMatrix(wedge).matrix)
        via_oracle = self.pc.octetBilinearRead(
            qm.CovarianceState(gamma), [0, 1, 2])
        direct = self.pc.octetBilinearRead(
            qm.CovarianceState.fromSlaterFrame(orbitals), [0, 1, 2])
        for field in ("occupation", "octetWeight", "singletWeight",
                      "casimir", "casimirExpectation"):
            self.assertAlmostEqual(getattr(via_oracle, field),
                                   getattr(direct, field), delta=1e-12,
                                   msg=field)
        self.assertEqual(via_oracle.subsetParity, direct.subsetParity)

    def test_mode_validation(self):
        state = _rank2_state()
        with self.assertRaises(ValueError):
            self.pc.octetBilinearRead(state, [0, 1])
        with self.assertRaises(ValueError):
            self.pc.octetBilinearRead(state, [0, 1, 1])
        with self.assertRaises(ValueError):
            self.pc.octetBilinearRead(state, [0, 1, 7])

    def test_embedded_triad_reads_like_the_small_fixture(self):
        # the color triad on modes (2, 3, 4) of a 6-mode state reads
        # exactly like the standalone 3-mode fixture
        small = self.pc.octetBilinearRead(_rank2_state(), [0, 1, 2])
        gamma = np.zeros((6, 6), dtype=complex)
        gamma[2:5, 2:5] = np.asarray(_rank2_state().gamma())
        embedded = self.pc.octetBilinearRead(qm.CovarianceState(gamma),
                                             [2, 3, 4])
        for field in ("occupation", "subsetParity", "octetWeight",
                      "singletWeight", "casimir", "casimirExpectation"):
            self.assertEqual(getattr(small, field),
                             getattr(embedded, field), msg=field)

    def test_global_relabeling_invariance(self):
        # permute the whole mode universe and carry the declared color
        # modes through the permutation: the read is IDENTICAL
        state = _rank2_state((0.1, 0.6, 0.5))
        gamma = np.asarray(state.gamma())
        perm = [2, 0, 1]  # new index of old mode i
        p = np.zeros((3, 3))
        for old, new in enumerate(perm):
            p[new, old] = 1.0
        relabeled = qm.CovarianceState(p @ gamma @ p.T)
        base = self.pc.octetBilinearRead(state, [0, 1, 2])
        moved = self.pc.octetBilinearRead(relabeled,
                                          [perm[0], perm[1], perm[2]])
        # the echoed color-mode LABELS legitimately track the relabeling;
        # every physical channel is invariant (the permutation reorders
        # the Wick trace accumulation: identical algebra, double
        # round-off ~1e-16)
        rec_base, rec_moved = base.toRecord(), moved.toRecord()
        self.assertEqual(rec_moved["color_modes"], perm)
        del rec_base["color_modes"], rec_moved["color_modes"]
        self.assertLessEqual(
            obs.ObservableGates.report_delta(rec_base, rec_moved), 1e-14)
        self.assertEqual(base.subsetParity, moved.subsetParity)
        self.assertEqual(base.occupation, moved.occupation)

    def test_color_order_is_the_recorded_trivialization(self):
        # reordering the DECLARED color modes conjugates the bilinear by
        # the permutation: invariant weights/casimir/occupation/parity,
        # covariant components
        state = _rank2_state((0.1, 0.6, 0.5))
        base = self.pc.octetBilinearRead(state, [0, 1, 2])
        swapped = self.pc.octetBilinearRead(state, [1, 0, 2])
        for field in ("occupation", "subsetParity", "octetWeight",
                      "singletWeight", "casimir", "casimirExpectation"):
            self.assertAlmostEqual(getattr(base, field),
                                   getattr(swapped, field), delta=1e-12,
                                   msg=field)
        p = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=complex)
        self.assertLessEqual(
            np.max(np.abs(np.asarray(swapped.bilinear)
                          - p @ np.asarray(base.bilinear) @ p.T)), 1e-15)

    def test_read_is_read_only_on_the_state(self):
        state = _rank2_state()
        before = state.covarianceHash()
        self.pc.octetBilinearRead(state, [0, 1, 2])
        self.assertEqual(state.covarianceHash(), before)

    def test_record_roundtrip_is_exact(self):
        read = self.pc.octetBilinearRead(_rank2_state((0.2, 0.3, 0.9)),
                                         [0, 1, 2])
        rec = read.toRecord()
        back = obs.OctetBilinearRead.fromRecord(rec)
        self.assertEqual(
            obs.ObservableGates.report_delta(rec, back.toRecord()), 0.0)

    def test_from_record_rejects_unknown_schema(self):
        rec = self.pc.octetBilinearRead(_rank2_state(), [0, 1, 2]).toRecord()
        rec["schema_version"] = 99
        with self.assertRaises(ValueError):
            obs.OctetBilinearRead.fromRecord(rec)


class TestOctetCollectiveGrowth(unittest.TestCase):
    """Arbitrarily many collective excitations by ADDING microscopic
    modes: the vacuum embedding changes no sector read and no
    two-dimensional edge-mode factor."""

    def setUp(self):
        self.pc = obs.ParticleClusters()

    def test_zero_block_vacuum_extension_leaves_the_read_unchanged(self):
        base = self.pc.octetBilinearRead(_rank2_state(), [0, 1, 2])
        for extra in (1, 5, 13):
            gamma = np.zeros((3 + extra, 3 + extra), dtype=complex)
            gamma[:3, :3] = np.asarray(_rank2_state().gamma())
            grown = self.pc.octetBilinearRead(qm.CovarianceState(gamma),
                                              [0, 1, 2])
            self.assertEqual(obs.ObservableGates.report_delta(
                base.toRecord(), grown.toRecord()), 0.0,
                msg=f"extra={extra}")

    def test_lazy_vacuum_embedding_leaves_the_read_unchanged(self):
        # the #771 inductive-limit route: iota(psi) = psi (x) |0> adds
        # microscopic modes; every existing sector read is unchanged
        rng = np.random.default_rng(83)
        orbitals = rng.normal(size=(3, 2)) + 1j * rng.normal(size=(3, 2))
        eng = qm.LazyFockEngine(8)
        small = eng.wedgeState([0, 1, 2], orbitals)
        grown = eng.embedInVacuum(small, [3, 4, 5, 6, 7])
        g_small = np.asarray(eng.covarianceMatrix(small).matrix)
        g_grown = np.asarray(eng.covarianceMatrix(grown).matrix)
        a = self.pc.octetBilinearRead(qm.CovarianceState(g_small),
                                      [0, 1, 2])
        b = self.pc.octetBilinearRead(qm.CovarianceState(g_grown),
                                      [0, 1, 2])
        # identical algebra; the engine's closed-form Slater covariance
        # evaluates through a different GEMM shape after the embedding,
        # so the doubles agree to round-off (~1e-16), not bitwise
        self.assertLessEqual(obs.ObservableGates.report_delta(
            a.toRecord(), b.toRecord()), 1e-14)
        self.assertEqual(a.subsetParity, b.subsetParity)
        # the added modes carry NOTHING: their covariance rows are
        # EXACTLY zero -- the two-level factor of every new mode is
        # untouched vacuum
        self.assertEqual(np.max(np.abs(g_grown[3:, :])), 0.0)

    def test_two_level_edge_mode_factor_never_changes(self):
        # each finite edge-mode factor remains TWO-dimensional: the stage
        # dimension exactly doubles per added microscopic mode, before and
        # after any collective excitation
        for m in range(1, 12):
            self.assertEqual(qm.LazyFockEngine.stageDimension(m + 1),
                             2 * qm.LazyFockEngine.stageDimension(m))

    def test_multiple_collective_excitations_coexist(self):
        # two independent octet excitations on disjoint triads of a
        # 6-mode state: each triad's read equals its standalone fixture
        gamma = np.zeros((6, 6), dtype=complex)
        gamma[:3, :3] = np.asarray(_rank2_state().gamma())
        gamma[3:, 3:] = np.asarray(_rank2_state((0.5, 0.5, 0.0)).gamma())
        state = qm.CovarianceState(gamma)
        a = self.pc.octetBilinearRead(state, [0, 1, 2])
        b = self.pc.octetBilinearRead(state, [3, 4, 5])
        ref_a = self.pc.octetBilinearRead(_rank2_state(), [0, 1, 2])
        ref_b = self.pc.octetBilinearRead(_rank2_state((0.5, 0.5, 0.0)),
                                          [0, 1, 2])
        for read, ref in ((a, ref_a), (b, ref_b)):
            self.assertEqual(read.occupation, ref.occupation)
            self.assertEqual(read.octetWeight, ref.octetWeight)
            self.assertEqual(read.subsetParity, ref.subsetParity)
            self.assertAlmostEqual(read.casimirExpectation,
                                   ref.casimirExpectation, delta=1e-12)

    def test_scaling_with_mode_count(self):
        # scaling test: the SAME embedded triad read at growing mode
        # count -- values constant, cost polynomial (timed and printed)
        base = self.pc.octetBilinearRead(_rank2_state(), [0, 1, 2])
        timings = []
        for total in (3, 12, 24, 48):
            gamma = np.zeros((total, total), dtype=complex)
            gamma[:3, :3] = np.asarray(_rank2_state().gamma())
            state = qm.CovarianceState(gamma)
            t0 = time.perf_counter()
            read = self.pc.octetBilinearRead(state, [0, 1, 2])
            timings.append((total, time.perf_counter() - t0))
            self.assertEqual(read.occupation, base.occupation)
            self.assertEqual(read.octetWeight, base.octetWeight)
            self.assertAlmostEqual(read.casimirExpectation,
                                   base.casimirExpectation, delta=1e-12)
        print("\n[benchmark] octetBilinearRead scaling: "
              + "; ".join(f"M={m}: {dt * 1e3:.2f} ms" for m, dt in timings))
        self.assertLess(timings[-1][1], 1.0)


class TestOctetReadCache(unittest.TestCase):
    """The #764 AnalyticCache contract of the cached octet read."""

    def setUp(self):
        self.pc = obs.ParticleClusters()
        self.st = _from_simplices(7, _TETRA_CHAIN, timelike=False)
        self.ids = [0, 1, 2, 3]

    def test_cached_equals_cold(self):
        cache = cob.AnalyticCache(self.st)
        state = _rank2_state()
        cold = self.pc.octetBilinearRead(state, [0, 1, 2])
        first = self.pc.octetBilinearReadCached(cache, self.ids, state,
                                                [0, 1, 2])
        served = self.pc.octetBilinearReadCached(cache, self.ids, state,
                                                 [0, 1, 2])
        self.assertEqual(obs.ObservableGates.report_delta(
            cold.toRecord(), first.toRecord()), 0.0)
        self.assertEqual(obs.ObservableGates.report_delta(
            cold.toRecord(), served.toRecord()), 0.0)
        self.assertGreaterEqual(cache.hits, 1)

    def test_touched_star_invalidates(self):
        cache = cob.AnalyticCache(self.st)
        self.pc.octetBilinearReadCached(cache, self.ids, _rank2_state(),
                                        [0, 1, 2])
        self.assertEqual(cache.size, 1)
        star = cob.TouchedStar()
        star.addChangedEdge(0, 1)
        cache.publish(star)
        self.assertEqual(cache.size, 0)

    def test_gamma_change_never_serves_a_stale_read(self):
        cache = cob.AnalyticCache(self.st)
        a = self.pc.octetBilinearReadCached(cache, self.ids, _rank2_state(),
                                            [0, 1, 2])
        changed = _rank2_state((1.0, 0.0, 0.0))
        b = self.pc.octetBilinearReadCached(cache, self.ids, changed,
                                            [0, 1, 2])
        cold = self.pc.octetBilinearRead(changed, [0, 1, 2])
        self.assertEqual(obs.ObservableGates.report_delta(
            b.toRecord(), cold.toRecord()), 0.0)
        self.assertFalse(np.array_equal(np.asarray(a.bilinear),
                                        np.asarray(b.bilinear)))

    def test_fingerprint_sensitivity(self):
        state = _rank2_state()
        base = self.pc.octetFingerprint(state, [0, 1, 2])
        self.assertNotEqual(base,
                            self.pc.octetFingerprint(state, [1, 0, 2]))
        self.assertNotEqual(
            base, self.pc.octetFingerprint(_rank2_state((1.0, 0.0, 0.0)),
                                           [0, 1, 2]))
        strict_cfg = obs.ParticleClustersConfig()
        strict_cfg.parityTolerance = 1e-3
        self.assertNotEqual(
            base, obs.ParticleClusters(strict_cfg).octetFingerprint(
                state, [0, 1, 2]))


class TestGluonClassification(unittest.TestCase):
    """Design spec section 14.3: a gluon candidate is a persistent
    transported octet excitation with zero baryon flux and even parity."""

    GATES = ["parity-even", "octet-excitation", "octet-purity",
             "octet-transport", "winding-zero", "persistence"]

    def setUp(self):
        self.pc = obs.ParticleClusters()

    def test_certified_gluon_candidate(self):
        read = self.pc.classifyGluon(_gluon_evidence())
        self.assertEqual(read.classification, "gluon-candidate")
        self.assertEqual(read.confidence, 1.0)
        self.assertEqual(read.failedCertificates, [])
        self.assertEqual(read.exteriorParity, +1)
        self.assertEqual(read.determinantWinding, 0)
        # a CERTIFIED zero flux -- 0.0 as evidence, not a default
        self.assertEqual(read.baryonFlux, 0.0)
        self.assertAlmostEqual(read.occupationTotal, 2.0, delta=MACHINE)
        self.assertAlmostEqual(read.casimir, 3.0, delta=1e-12)
        # C2(3bar) = 4/3: the flat consumed-scalar summary (one source of
        # truth -- the full OctetBilinearRead travels on the evidence)
        self.assertAlmostEqual(read.casimirExpectation, 4.0 / 3.0,
                               delta=1e-12)
        self.assertLessEqual(read.octetProjectorResidual, 1e-14)
        self.assertEqual(read.windingClosure, "closed-family")
        self.assertTrue(read.certificate.holds())
        self.assertEqual(read.certificate.grade,
                         cob.CertificateGrade.StructureExact)

    def test_candidate_is_the_strongest_claim(self):
        # ticket out-of-scope: no even octet excitation is claimed to be a
        # physical gluon -- the accepted verdict string is EXACTLY
        # "gluon-candidate"
        read = self.pc.classifyGluon(_gluon_evidence())
        self.assertEqual(read.classification, "gluon-candidate")
        self.assertNotEqual(read.classification, "gluon")

    def test_odd_carried_state_is_rejected_from_the_even_read(self):
        # negative control: an ODD-sector object (N = 1 fundamental) fails
        # the even-parity gate by name
        c = np.array([1.0, 0.0, 0.0], dtype=complex)
        odd = qm.CovarianceState(np.outer(c, c.conj()))
        read = self.pc.classifyGluon(_gluon_evidence(state=odd))
        self.assertEqual(read.classification, "none")
        self.assertEqual(read.exteriorParity, -1)
        self.assertIn("parity-even", read.failedCertificates)
        self.assertNotIn("winding-zero", read.failedCertificates)

    def test_nonzero_winding_octet_excitation_is_not_a_gluon(self):
        # negative control: certified nu = 1 is honest evidence (B = 1/3
        # reported) but NOT a gluon candidate
        read = self.pc.classifyGluon(_gluon_evidence(turns=1))
        self.assertEqual(read.classification, "none")
        self.assertIn("winding-zero", read.failedCertificates)
        self.assertEqual(read.determinantWinding, 1)
        self.assertAlmostEqual(read.baryonFlux, 1.0 / 3.0, delta=MACHINE)

    def test_unknown_winding_leaves_flux_unknown(self):
        ev = _gluon_evidence()
        conn = obs.FiberConnection()
        A, B = _unit_fiber(1, 3), _unit_fiber(11, 3)
        segment = [_phase_link(conn, A, B, p) for p in (0.0, 0.3, 0.6)]
        ev.winding = conn.openSegmentWinding(segment,
                                             obs.WindingClosureSpec())
        read = self.pc.classifyGluon(ev)
        self.assertIsNone(read.determinantWinding)
        self.assertIsNone(read.baryonFlux)  # UNKNOWN, never zero
        self.assertIn("winding-zero", read.failedCertificates)

    def test_missing_octet_read_fails_by_name(self):
        ev = _gluon_evidence()
        ev.octet = obs.OctetBilinearRead()
        read = self.pc.classifyGluon(ev)
        self.assertIn("octet-excitation", read.failedCertificates)
        self.assertIn("octet-purity", read.failedCertificates)
        self.assertTrue(math.isnan(read.casimir))
        self.assertTrue(math.isnan(read.casimirExpectation))
        self.assertTrue(math.isnan(read.octetWeight))

    def test_vacuum_carries_no_excitation(self):
        vacuum = qm.CovarianceState(np.zeros((3, 3), dtype=complex))
        read = self.pc.classifyGluon(_gluon_evidence(state=vacuum))
        self.assertEqual(read.classification, "none")
        self.assertIn("octet-excitation", read.failedCertificates)
        # vacuum parity is even -- that gate PASSES; the excitation gate
        # is what refuses
        self.assertNotIn("parity-even", read.failedCertificates)

    def test_rank_two_transport_is_not_an_octet_transport(self):
        ev = _gluon_evidence()
        conn = obs.FiberConnection()
        A2, B2 = _unit_fiber(61, 2), _unit_fiber(71, 2)
        ev.lifetimeTransports = [conn.transport(A2, B2,
                                                np.eye(2, dtype=complex))]
        read = self.pc.classifyGluon(ev)
        self.assertIn("octet-transport", read.failedCertificates)

    def test_leaky_transport_fails(self):
        ev = _gluon_evidence()
        conn = obs.FiberConnection()
        A, B = _unit_fiber(1, 3), _unit_fiber(11, 3)
        leaky = conn.transport(A, B, np.diag([0.4, 1.0, 1.0]))
        ev.lifetimeTransports = list(ev.lifetimeTransports) + [leaky]
        read = self.pc.classifyGluon(ev)
        self.assertIn("octet-transport", read.failedCertificates)

    def test_missing_transports_fail(self):
        ev = _gluon_evidence()
        ev.lifetimeTransports = []
        read = self.pc.classifyGluon(ev)
        self.assertIn("octet-transport", read.failedCertificates)
        self.assertTrue(math.isnan(read.transportLeakageMax))

    def test_insufficient_persistence(self):
        read = self.pc.classifyGluon(_gluon_evidence(lifetime=1.0))
        self.assertIn("persistence", read.failedCertificates)
        self.assertEqual(read.classification, "none")

    def test_uncertified_parity_never_emits_a_sign(self):
        ev = _gluon_evidence()
        ev.parityRead = qm.WickCertificateRead()
        read = self.pc.classifyGluon(ev)
        self.assertEqual(read.exteriorParity, 0)
        self.assertIn("parity-even", read.failedCertificates)

    def test_confidence_is_the_passed_fraction(self):
        ev = _gluon_evidence(turns=1, lifetime=1.0)  # two gates fail
        read = self.pc.classifyGluon(ev)
        self.assertAlmostEqual(read.confidence, 4.0 / 6.0, delta=MACHINE)
        self.assertEqual(sorted(read.failedCertificates),
                         ["persistence", "winding-zero"])

    def test_thresholds_are_recorded(self):
        cfg = obs.ParticleClustersConfig()
        cfg.minOctetWeight = 0.123
        read = obs.ParticleClusters(cfg).classifyGluon(_gluon_evidence())
        self.assertEqual(read.thresholds.minOctetWeight, 0.123)

    def test_record_roundtrip_and_null_semantics(self):
        read = self.pc.classifyGluon(_gluon_evidence())
        rec = read.toRecord()
        back = obs.GluonRead.fromRecord(rec)
        self.assertEqual(
            obs.ObservableGates.report_delta(rec, back.toRecord()), 0.0)
        empty = self.pc.classifyGluon(obs.GluonCandidateEvidence())
        rec = empty.toRecord()
        self.assertIsNone(rec["determinant_winding"])
        self.assertIsNone(rec["baryon_flux"])
        self.assertEqual(rec["exterior_parity"], 0)
        self.assertTrue(math.isnan(rec["occupation_total"]))
        for name in self.GATES:
            self.assertIn(name, empty.failedCertificates)

    def test_relabeling_preserves_the_read(self):
        base = self.pc.classifyGluon(_gluon_evidence(band_base=1))
        shifted = self.pc.classifyGluon(_gluon_evidence(band_base=901))
        self.assertEqual(obs.ObservableGates.report_delta(
            base.toRecord(), shifted.toRecord()), 0.0)

    def test_simplex_reorientation_preserves_the_verdict(self):
        # the ORIENTATION channel: a common row sign flip (reversing a
        # cell's orientation flips its cochain component on every column
        # alike).  det C picks up det(S) = +-1 and the SINGLET certificate
        # |det C|^2 is exactly invariant.
        base = self.pc.classifyBaryon(_baryon_evidence())
        for signs in ([1, 1, -1], [-1, -1, -1], [-1, 1, -1]):
            columns = np.diag(signs).astype(complex) @ _color_triad()
            read = self.pc.classifyBaryon(_baryon_evidence(color=columns))
            self.assertEqual(read.classification, base.classification)
            self.assertAlmostEqual(read.colorGramDeterminant,
                                   base.colorGramDeterminant, delta=MACHINE)
            expected = base.colorWedge * float(np.prod(signs))
            self.assertLess(abs(read.colorWedge - expected), 1e-13)

    def test_refinement_sample_order_does_not_change_stability(self):
        samples = _scale_samples()
        forward = self.pc.scaleProfile(samples)
        backward = self.pc.scaleProfile(list(reversed(samples)))
        self.assertEqual(forward.stable, backward.stable)
        self.assertEqual(forward.radiusRatioSpread,
                         backward.radiusRatioSpread)
        self.assertEqual(forward.profileMaxDeviation,
                         backward.profileMaxDeviation)
        drifting = _scale_samples(drift=0.05)
        self.assertEqual(
            self.pc.scaleProfile(drifting).failedCertificates,
            self.pc.scaleProfile(list(reversed(drifting)))
            .failedCertificates)

    def test_cold_replay_is_deterministic(self):
        a = self.pc.classifyGluon(_gluon_evidence())
        b = obs.ParticleClusters().classifyGluon(_gluon_evidence())
        self.assertEqual(obs.ObservableGates.report_delta(
            a.toRecord(), b.toRecord()), 0.0)
        # and the checkpoint replay: fromRecord(toRecord) re-serializes
        # bit-identically (the cold-replay acceptance channel)
        rec = a.toRecord()
        self.assertEqual(obs.ObservableGates.report_delta(
            rec, obs.GluonRead.fromRecord(rec).toRecord()), 0.0)

    def test_describe_smoke(self):
        text = self.pc.classifyGluon(_gluon_evidence()).describe()
        self.assertIn("gluon-candidate", text)
        self.assertIn("B=0", text)


class TestMesonClassification(unittest.TestCase):
    """Quark-antiquark singlet composites as meson candidates."""

    def setUp(self):
        self.pc = obs.ParticleClusters()

    def test_certified_meson_candidate(self):
        read = self.pc.classifyMeson(_meson_evidence())
        self.assertEqual(read.classification, "meson-candidate")
        self.assertEqual(read.failedCertificates, [])
        # EVEN color singlet with ZERO total baryon flux (acceptance)
        self.assertEqual(read.exteriorParity, +1)
        self.assertEqual(read.totalWinding, 0)
        self.assertEqual(read.totalBaryonFlux, 0.0)
        self.assertLessEqual(read.pairingOctetFraction, 1e-15)
        self.assertAlmostEqual(read.occupationTotal, 2.0, delta=MACHINE)
        self.assertTrue(read.certificate.holds())

    def test_order_insensitive(self):
        ev = _meson_evidence()
        swapped = _meson_evidence()
        swapped.first, swapped.second = ev.second, ev.first
        a = self.pc.classifyMeson(ev)
        b = self.pc.classifyMeson(swapped)
        self.assertEqual(a.classification, b.classification)
        self.assertEqual(a.totalWinding, b.totalWinding)
        self.assertEqual(a.exteriorParity, b.exteriorParity)

    def test_octet_pairing_is_not_a_meson(self):
        # a q-qbar pair in the OCTET channel is a gluon-sector object,
        # not a color-singlet meson
        read = self.pc.classifyMeson(_meson_evidence(pairing="octet"))
        self.assertEqual(read.classification, "none")
        self.assertIn("color-singlet", read.failedCertificates)
        self.assertAlmostEqual(read.pairingOctetFraction, 1.0,
                               delta=1e-15)

    def test_missing_pairing_fails_by_name(self):
        read = self.pc.classifyMeson(_meson_evidence(pairing="none"))
        self.assertIn("color-singlet", read.failedCertificates)
        self.assertTrue(math.isnan(read.pairingOctetFraction))

    def test_two_quarks_are_not_a_meson(self):
        read = self.pc.classifyMeson(
            _meson_evidence(first_turns=1, second_turns=1))
        self.assertEqual(read.classification, "none")
        self.assertIn("constituent-antiquark", read.failedCertificates)
        self.assertNotIn("constituent-quark", read.failedCertificates)
        # and the flux channel refuses too: nu total = 2, not 0
        self.assertIn("flux-zero", read.failedCertificates)

    def test_two_antiquarks_are_not_a_meson(self):
        read = self.pc.classifyMeson(
            _meson_evidence(first_turns=-1, second_turns=-1))
        self.assertIn("constituent-quark", read.failedCertificates)
        self.assertNotIn("constituent-antiquark", read.failedCertificates)

    def test_singular_constituent_leaves_flux_unknown(self):
        ev = _meson_evidence()
        conn = obs.FiberConnection()
        A, B = _unit_fiber(1, 3), _unit_fiber(11, 3)
        bad_ev = _certified_evidence(turns=-1)
        bad = conn.transport(A, B, np.diag([0.1, 1.0, 1.0]))
        bad_ev.winding = conn.closedFamilyWinding(
            _winding_family(conn, A, B, turns=-1) + [bad])
        ev.second = self.pc.classifyQuark(bad_ev)
        read = self.pc.classifyMeson(ev)
        self.assertIsNone(read.totalWinding)
        self.assertIsNone(read.totalBaryonFlux)  # UNKNOWN, never zero
        self.assertIn("flux-zero", read.failedCertificates)

    def test_uncertified_constituent_parity_is_unknown(self):
        ev = _meson_evidence()
        blind = _certified_evidence(turns=-1, band_base=31)
        blind.parityRead = qm.WickCertificateRead()
        ev.second = self.pc.classifyQuark(blind)
        read = self.pc.classifyMeson(ev)
        self.assertEqual(read.exteriorParity, 0)
        self.assertIn("parity-even", read.failedCertificates)

    def test_composite_parity_is_the_exact_graded_product(self):
        # whitepaper parity table: two odd constituents compose EVEN
        ev = _meson_evidence()
        read = self.pc.classifyMeson(ev)
        self.assertEqual(read.exteriorParity,
                         ev.first.exteriorParity * ev.second.exteriorParity)
        self.assertEqual(read.exteriorParity, +1)

    def test_composite_transport_leakage_is_reported(self):
        # the ticket's report set: transport leakage travels on the read
        # (report-only for the two-cluster composites -- never a gate)
        ev = _meson_evidence()
        conn = obs.FiberConnection()
        A, B = _unit_fiber(1, 3), _unit_fiber(11, 3)
        ev.lifetimeTransports = [_phase_link(conn, A, B, 0.1)]
        read = self.pc.classifyMeson(ev)
        self.assertEqual(read.transportCount, 1)
        self.assertLessEqual(read.transportLeakageMax, 1e-9)
        self.assertEqual(read.classification, "meson-candidate")

    def test_record_roundtrip_and_describe(self):
        read = self.pc.classifyMeson(_meson_evidence())
        rec = read.toRecord()
        back = obs.MesonRead.fromRecord(rec)
        self.assertEqual(
            obs.ObservableGates.report_delta(rec, back.toRecord()), 0.0)
        self.assertIn("meson-candidate", read.describe())
        rec["schema_version"] = 99
        with self.assertRaises(ValueError):
            obs.MesonRead.fromRecord(rec)

    def test_relabeling_and_cold_replay(self):
        a = self.pc.classifyMeson(_meson_evidence())
        b = obs.ParticleClusters().classifyMeson(_meson_evidence())
        self.assertEqual(obs.ObservableGates.report_delta(
            a.toRecord(), b.toRecord()), 0.0)


class TestDiquarkClassification(unittest.TestCase):
    """Two-quark anti-triplet even composites with preserved B = 2/3."""

    def setUp(self):
        self.pc = obs.ParticleClusters()

    def test_certified_diquark_candidate(self):
        read = self.pc.classifyDiquark(_diquark_evidence())
        self.assertEqual(read.classification, "diquark-candidate")
        self.assertEqual(read.failedCertificates, [])
        # even 3bar state with B = 2/3 (acceptance)
        self.assertEqual(read.exteriorParity, +1)
        self.assertEqual(read.totalWinding, 2)
        self.assertEqual(read.totalBaryonFlux, 2.0 / 3.0)
        self.assertAlmostEqual(read.antiTripletWeight, 1.0, delta=MACHINE)
        self.assertAlmostEqual(read.occupationTotal, 2.0, delta=MACHINE)
        self.assertTrue(read.certificate.holds())

    def test_explicitly_not_an_antiquark(self):
        # The #773 distinction fixture, composed: the SAME two-quark
        # carried state fed to the QUARK classifier (anti-triplet color,
        # nu = -1 tube) refuses on occupation/parity; the diquark read
        # accepts with B = +2/3 -- opposite sign and triple the magnitude
        # of an antiquark's B = -1/3.
        quark_view = self.pc.classifyQuark(
            _certified_evidence(turns=-1, occupations=(1.0, 1.0, 0.0)))
        self.assertEqual(quark_view.classification, "none")
        self.assertIn("parity-odd", quark_view.failedCertificates)
        self.assertIn("occupation-one", quark_view.failedCertificates)

        read = self.pc.classifyDiquark(_diquark_evidence())
        self.assertEqual(read.classification, "diquark-candidate")
        self.assertEqual(read.totalBaryonFlux, 2.0 / 3.0)
        self.assertNotEqual(read.totalBaryonFlux, -1.0 / 3.0)
        self.assertAlmostEqual(read.occupationTotal, 2.0, delta=MACHINE)
        self.assertEqual(read.exteriorParity, +1)

    def test_duplicated_color_mode_is_pauli_zero(self):
        # det(C^dag Gamma C) with a repeated color column is EXACTLY zero
        # (the Gram/Pauli identity): the anti-triplet gate refuses
        dup = np.array([[1.0, 1.0], [0.0, 0.0], [0.0, 0.0]], dtype=complex)
        ev = _diquark_evidence(columns=dup)
        self.assertEqual(ev.antiTripletRead.value, 0.0 + 0.0j)
        read = self.pc.classifyDiquark(ev)
        self.assertEqual(read.classification, "none")
        self.assertIn("anti-triplet", read.failedCertificates)
        self.assertEqual(read.antiTripletWeight, 0.0)

    def test_quark_antiquark_pair_is_not_a_diquark(self):
        read = self.pc.classifyDiquark(_diquark_evidence(second_turns=-1))
        self.assertEqual(read.classification, "none")
        self.assertIn("constituent-quarks", read.failedCertificates)
        # nu total = 0, not 2: the flux channel refuses independently
        self.assertIn("baryon-flux-two-thirds", read.failedCertificates)
        self.assertEqual(read.totalWinding, 0)

    def test_missing_wedge_read_fails_by_name(self):
        ev = _diquark_evidence()
        ev.antiTripletRead = qm.WickCertificateRead()
        read = self.pc.classifyDiquark(ev)
        self.assertIn("anti-triplet", read.failedCertificates)
        self.assertTrue(math.isnan(read.antiTripletWeight))

    def test_constituent_flux_is_preserved(self):
        ev = _diquark_evidence()
        read = self.pc.classifyDiquark(ev)
        self.assertEqual(read.totalBaryonFlux,
                         ev.first.baryonFlux + ev.second.baryonFlux)

    def test_singular_constituent_leaves_flux_unknown(self):
        ev = _diquark_evidence()
        conn = obs.FiberConnection()
        A, B = _unit_fiber(1, 3), _unit_fiber(11, 3)
        bad_ev = _certified_evidence(turns=1, band_base=51)
        bad = conn.transport(A, B, np.diag([0.1, 1.0, 1.0]))
        bad_ev.winding = conn.closedFamilyWinding(
            _winding_family(conn, A, B, turns=1) + [bad])
        ev.second = self.pc.classifyQuark(bad_ev)
        read = self.pc.classifyDiquark(ev)
        self.assertIsNone(read.totalWinding)
        self.assertIsNone(read.totalBaryonFlux)
        self.assertIn("baryon-flux-two-thirds", read.failedCertificates)

    def test_record_roundtrip_and_describe(self):
        read = self.pc.classifyDiquark(_diquark_evidence())
        rec = read.toRecord()
        back = obs.DiquarkRead.fromRecord(rec)
        self.assertEqual(
            obs.ObservableGates.report_delta(rec, back.toRecord()), 0.0)
        self.assertIn("diquark-candidate", read.describe())

    def test_cold_replay_is_deterministic(self):
        a = self.pc.classifyDiquark(_diquark_evidence())
        b = obs.ParticleClusters().classifyDiquark(_diquark_evidence())
        self.assertEqual(obs.ObservableGates.report_delta(
            a.toRecord(), b.toRecord()), 0.0)


class TestEvenSectorGuardsAndBenchmark(unittest.TestCase):
    """Shared #763 merge gates for the even sectors."""

    def test_no_even_sector_quantity_enters_the_emergence_objective(self):
        objective_homes = [REPO_ROOT / "src" / "cobordism",
                           REPO_ROOT / "include" / "cobordism",
                           REPO_ROOT / "src" / "rl",
                           REPO_ROOT / "include" / "rl",
                           REPO_ROOT / "src" / "simulations",
                           REPO_ROOT / "include" / "simulations"]
        needles = ("GluonRead", "MesonRead", "DiquarkRead",
                   "OctetBilinearRead", "classifyGluon", "classifyMeson",
                   "classifyDiquark")
        offenders = []
        for home in objective_homes:
            if not home.exists():
                continue
            for path in home.rglob("*"):
                if path.suffix not in (".h", ".cpp", ".cu", ".hpp"):
                    continue
                text = path.read_text(errors="ignore")
                if any(needle in text for needle in needles):
                    offenders.append(str(path))
        self.assertEqual(offenders, [])

    def test_old_threshold_records_still_rehydrate(self):
        # pre-#774 checkpoints lack the new threshold keys: the reader
        # falls back to the defaults instead of rejecting
        pc = obs.ParticleClusters()
        rec = pc.classifyQuark(_certified_evidence()).toRecord()
        for key in ("min_octet_weight", "octet_purity_tolerance",
                    "composite_octet_tolerance", "min_anti_triplet_weight"):
            self.assertIn(key, rec["thresholds"])
            del rec["thresholds"][key]
        back = obs.QuarkRead.fromRecord(rec)
        defaults = obs.ParticleClustersConfig()
        self.assertEqual(back.thresholds.minOctetWeight,
                         defaults.minOctetWeight)
        self.assertEqual(back.thresholds.minAntiTripletWeight,
                         defaults.minAntiTripletWeight)

    def test_new_thresholds_enter_the_evidence_fingerprint(self):
        ev = _certified_evidence()
        base = obs.ParticleClusters()
        cfg = obs.ParticleClustersConfig()
        cfg.minOctetWeight = 0.5
        self.assertNotEqual(base.evidenceFingerprint(ev),
                            obs.ParticleClusters(cfg).evidenceFingerprint(ev))

    def test_classification_cost_per_candidate(self):
        # merge-gate benchmark: even-sector classification cost (numbers
        # reported in the PR body)
        pc = obs.ParticleClusters()
        gluon_ev = _gluon_evidence()
        meson_ev = _meson_evidence()
        diquark_ev = _diquark_evidence()
        state = _rank2_state()
        n = 200
        t0 = time.perf_counter()
        for _ in range(n):
            pc.octetBilinearRead(state, [0, 1, 2])
        octet = (time.perf_counter() - t0) / n
        t0 = time.perf_counter()
        for _ in range(n):
            pc.classifyGluon(gluon_ev)
        gluon = (time.perf_counter() - t0) / n
        t0 = time.perf_counter()
        for _ in range(n):
            pc.classifyMeson(meson_ev)
        meson = (time.perf_counter() - t0) / n
        t0 = time.perf_counter()
        for _ in range(n):
            pc.classifyDiquark(diquark_ev)
        diquark = (time.perf_counter() - t0) / n
        print(f"\n[benchmark] octetBilinearRead: {octet * 1e6:.1f} us; "
              f"classifyGluon: {gluon * 1e6:.1f} us; "
              f"classifyMeson: {meson * 1e6:.1f} us; "
              f"classifyDiquark: {diquark * 1e6:.1f} us per candidate")
        for cost in (octet, gluon, meson, diquark):
            self.assertLess(cost, 0.05)


if __name__ == "__main__":
    unittest.main()


# =========================================================================== #
# #775 — three-quark baryons and the complete proton certificate
#
# Design spec sections 16.2 (bound-supercomponent search), 16.3 (color
# singlet), 16.4 (proton classifier) and 5.12 (sharp spin); whitepaper "The
# proton as the maximally informative baryon".  Every fixture below is
# composed from MERGED public APIs — #765 PersistentModularity discovery,
# #767 ColorFiber, #770 FiberConnection, #772 ExchangeHolonomy, #780
# CovarianceState Wick reads, and the existing #575/#566/#593 mass-radius
# battery through RegisterContext.
# =========================================================================== #

BARYON_STRUCTURAL = ["constituent-quarks", "bound-supercomponent"]
BARYON_PROTON = ["color-singlet", "color-flux-zero", "baryon-flux-unit",
                 "composite-parity-odd", "flavor-uud", "electric-flux-unit",
                 "spin-expectation", "sharp-spin", "rotation-character",
                 "spin-lift", "finite-radius", "profile-stability"]
BARYON_GATES = BARYON_STRUCTURAL + BARYON_PROTON

_UD_CACHE = {}


def _ud_quark(kind):
    """A CERTIFIED #773 u or d quark: nu = +1, odd parity, I3 = +-1/2 under
    the recorded doublet orientation, and a Gauss-consistent electric flux
    (the merged #773 fixtures, memoized -- each rebuild solves a Gauss
    least-squares problem)."""
    if kind not in _UD_CACHE:
        _UD_CACHE[kind] = obs.ParticleClusters().classifyQuark(
            _certified_evidence(
                with_flavor=True,
                occupancy=[1.0, 0.0] if kind == "u" else [0.0, 1.0],
                with_charge=2.0 / 3.0 if kind == "u" else -1.0 / 3.0))
    return _UD_CACHE[kind]


def _relabel_quark(read, level=1, tag="cd"):
    """The same certified read carried by a DIFFERENT label-free component
    identity (the relabeling channel: identity is a hash, never a name)."""
    record = read.toRecord()
    record["component_hash"] = tag * 16
    record["component_level"] = int(level)
    return obs.QuarkRead.fromRecord(record)


def _pauli_over_sites(n_sites):
    """J_alpha = direct sum over sites of sigma_alpha/2 on 2*n modes (the
    #780 fixture convention: modes ordered site-major)."""
    sx = np.array([[0, 1], [1, 0]], dtype=complex) / 2
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex) / 2
    sz = np.diag([1.0, -1.0]).astype(complex) / 2

    def blocks(s):
        out = np.zeros((2 * n_sites, 2 * n_sites), dtype=complex)
        for k in range(n_sites):
            out[2 * k:2 * k + 2, 2 * k:2 * k + 2] = s
        return out
    return blocks(sx), blocks(sy), blocks(sz)


def _sharp_spin_reads():
    """The exact J^2 = 3/4 EIGENSTATE: one particle in one spin-1/2
    doublet.  <J^2> = 3/4 and Var(J^2) = 0, both exact Wick sums."""
    js = _pauli_over_sites(1)
    state = qm.CovarianceState.fromOccupations(np.array([1.0, 0.0]))
    return (state.wickSpinSquaredExpectation(*js),
            state.wickSpinSquaredVariance(*js))


def _generic_slater_spin_reads():
    """The GENERIC Slater fixture: <J^2> = 3/4 EXACTLY but Var = 15/16 > 0
    (design spec 5.12 — expectation alone is not a sharp spin).  A spin-0
    singlet mode plus a standard spin-1 triplet, one particle in
    sqrt(5/8)|singlet> + sqrt(3/8)|m=1>."""
    s = 1 / np.sqrt(2.0)
    sx1 = np.array([[0, s, 0], [s, 0, s], [0, s, 0]], dtype=complex)
    sy1 = np.array([[0, -1j * s, 0], [1j * s, 0, -1j * s],
                    [0, 1j * s, 0]], dtype=complex)
    sz1 = np.diag([1.0, 0.0, -1.0]).astype(complex)

    def pad(m3):
        out = np.zeros((4, 4), dtype=complex)
        out[1:, 1:] = m3
        return out
    js = (pad(sx1), pad(sy1), pad(sz1))
    orbital = np.zeros((4, 1), dtype=complex)
    orbital[0, 0] = np.sqrt(5.0 / 8.0)
    orbital[1, 0] = np.sqrt(3.0 / 8.0)
    state = qm.CovarianceState.fromSlaterFrame(orbital)
    return (state.wickSpinSquaredExpectation(*js),
            state.wickSpinSquaredVariance(*js))


def _delta_spin_reads():
    """The Delta oracle: three aligned spins, J^2 = 15/4 with Var = 0 (a
    SHARP spin that is simply not 3/4)."""
    js = _pauli_over_sites(3)
    orbitals = np.zeros((6, 3), dtype=complex)
    for k in range(3):
        orbitals[2 * k, k] = 1.0
    state = qm.CovarianceState.fromSlaterFrame(orbitals)
    return (state.wickSpinSquaredExpectation(*js),
            state.wickSpinSquaredVariance(*js))


_ROTATION_CACHE = {}


def _rotation_character(turns=1, steps=16, d=4):
    """The #772 executable total-space 2pi cluster-frame cycle against its
    matched co-moving non-rotating reference: chi_hat(2pi) = -1 for one
    turn, +1 for two.  ONE global rotation of the whole carried frame —
    never a product of per-hole Bloch vectors."""
    key = (turns, steps, d)
    if key not in _ROTATION_CACHE:
        EH = obs.ExchangeHolonomy
        frame0 = EH.transverseSpinorFrame(0, 1, d)
        weights = np.ones(d, dtype=complex)
        loop = EH.loopHolonomy(
            EH.rotationLoopFrames(frame0, 0, 1, d, turns, steps), weights)
        reference = EH.loopHolonomy(
            EH.referenceLoopFrames(frame0, steps), weights)
        _ROTATION_CACHE[key] = EH.rotationCharacter(loop, reference)
    return _ROTATION_CACHE[key]


def _localized_mode(x, n):
    """A localized unit mode at ring position x (the #772 fixture idiom)."""
    k = int(math.floor(x)) % n
    f = x - math.floor(x)
    v = np.zeros(n, dtype=complex)
    v[k] += math.cos(f * math.pi / 2.0)
    v[(k + 1) % n] += math.sin(f * math.pi / 2.0)
    return v


def _exchange_character(steps=8, n=8, distance=4):
    """A genuine #772 PARTICLE-EXCHANGE character: two localized modes at
    0 and 4 advancing half an n-cell ring, against the matched
    non-exchanging reference.  chi_hat = -1 for one exchange.  This is the
    WRONG channel for the proton rotation certificate — the #772 channels
    are never interchangeable."""
    EH = obs.ExchangeHolonomy
    frames = []
    for t in range(steps):
        x = distance * t / steps
        frames.append(np.stack([_localized_mode((p + x) % n, n)
                                for p in (0, 4)], axis=1))
    weights = np.ones(n, dtype=complex)
    return EH.exchangeCharacter(
        EH.loopHolonomy(frames, weights),
        EH.loopHolonomy([frames[0]] * steps, weights))


def _rotation3(axis, theta):
    """A plane rotation of SO(3) about `axis` (the #772 Cech fixture idiom)."""
    c, s = math.cos(theta), math.sin(theta)
    out = np.eye(3)
    a, b = [i for i in range(3) if i != axis]
    out[a, a] = c
    out[b, b] = c
    out[a, b] = -s
    out[b, a] = s
    return out


def _accepted_spin_lift():
    """A tetrahedron of SO(3) transition data from a global vertex frame:
    the cocycle is exact and the lift EXISTS (no w2 obstruction)."""
    EH = obs.ExchangeHolonomy
    frames = {v: _rotation3(v % 3, 0.3 + 0.17 * v) for v in range(4)}
    edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    rotations = [frames[i] @ frames[j].T for i, j in edges]
    triangles = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
    return EH.spinLift(edges, rotations, triangles, 3)


def _obstructed_spin_lift():
    """The pillowcase class: two triangles glued along three pi-rotation
    edges — w2 evaluates 1 and the lift is REJECTED (#772 fixture)."""
    EH = obs.ExchangeHolonomy
    edges = [(0, 1), (1, 2), (2, 0)]
    rotations = [_rotation3(0, math.pi), _rotation3(1, math.pi),
                 _rotation3(2, math.pi)]
    return EH.spinLift(edges, rotations, [[0, 1, 2], [0, 2, 1]], 3)


def _color_triad():
    """An ORTHONORMAL anchored color triad: the exact #767 Fourier frame F3
    (assembled from the algebraic omega table), |det F3| = 1 exactly."""
    return np.asarray(obs.ColorFiber.fourierFrame())


def _su3_element(theta=0.7):
    """A g in SU(3) (certified by ColorFiber.isSpecialUnitary): a plane
    rotation with unit determinant."""
    c, s = math.cos(theta), math.sin(theta)
    g = np.eye(3, dtype=complex)
    g[0, 0] = c
    g[1, 1] = c
    g[0, 1] = -s
    g[1, 0] = s
    return g


def _filled_triplet_flux():
    """The bound object's octet bilinear on a FULLY OCCUPIED color triplet:
    M = I, so the traceless (net-color-flux) weight is EXACTLY zero."""
    state = qm.CovarianceState(np.eye(3, dtype=complex))
    return obs.ParticleClusters().octetBilinearRead(state, [0, 1, 2])


def _polarized_flux(hole=(0.0, 0.0, 1.0)):
    """A color-POLARIZED (anti-triplet) carried state: nonzero net color
    flux -- octet weight 2/3."""
    return obs.ParticleClusters().octetBilinearRead(_rank2_state(hole),
                                                    [0, 1, 2])


def _scale_samples(count=3, radius=0.75, cross=0.5, mass=2.25,
                   localization=0.8, profile=(0.6, 0.3, 0.1), drift=0.0,
                   profile_drift=0.0, color_gram=1.0, rotation=-1.0 + 0j,
                   baryon_flux=1.0, electric_flux=1.0, parity=-1,
                   anchor_score=0.788, certificate_drift=0.0):
    """A refinement window of EVERY dimensionless certificate channel: the
    existing mass-radius battery plus the colour Gram, rotation character,
    baryon flux, electric flux, composite parity and anchor score the
    whitepaper's "stability of every dimensionless certificate under
    refinement" covers (#808).  `certificate_drift` moves the six added
    channels; `drift` / `profile_drift` move the battery ones."""
    out = []
    for k in range(count):
        sample = obs.ScaleProfileSample()
        sample.radius = radius + drift * k
        sample.radiusCrossCheck = cross
        sample.spectralMass = mass
        sample.localization = localization
        sample.radialWeightProfile = [p + profile_drift * k for p in profile]
        sample.colorGramDeterminant = color_gram + certificate_drift * k
        sample.rotationCharacter = rotation + certificate_drift * k
        sample.baryonFlux = baryon_flux + certificate_drift * k
        sample.electricFlux = electric_flux + certificate_drift * k
        sample.compositeParity = parity
        sample.anchorScore = anchor_score + certificate_drift * k
        out.append(sample)
    return out


def _clique_edges(vertices, src, tgt):
    for i in range(len(vertices)):
        for j in range(i + 1, len(vertices)):
            src.append(vertices[i])
            tgt.append(vertices[j])


_HIERARCHY_CACHE = {}


def _modular_hierarchy():
    """A REAL two-level #765 hierarchy: three planted K6 cliques bridged
    into one coarse community.  gamma = 1 resolves the three cliques
    (level 1); gamma = 0.05 merges them into ONE level-2 supercomponent —
    exactly the "next modular level" of design spec 16.2."""
    if "h" not in _HIERARCHY_CACHE:
        groups = [list(range(0, 6)), list(range(10, 16)),
                  list(range(20, 26))]
        src, tgt = [], []
        for group in groups:
            _clique_edges(group, src, tgt)
        for a, b in ((0, 10), (10, 20), (20, 0)):
            src.append(a)
            tgt.append(b)
        graph = obs.PersistentModularity.fromWeightedEdges(
            src, tgt, [1.0] * len(src))
        cfg = tessera.PersistentModularityConfig()
        cfg.restarts = 4
        cfg.baseSeed = 0
        cfg.overlapThreshold = 0.5
        cfg.resolutions = [1.0]
        fine = graph.discover(1.0, cfg)
        cfg.resolutions = [0.05]
        coarse = graph.discover(0.05, cfg)
        _HIERARCHY_CACHE["h"] = (groups, fine, coarse)
    return _HIERARCHY_CACHE["h"]


def _bound_candidates(kinds=("u", "u", "d"), lifetimes=None, supports=None,
                      leakage_ok=True, transports=True, levels=None,
                      quarks=None):
    """Three #775 bound-supercomponent candidates: certified #773 quark
    verdicts, the planted level-0 supports, overlapping #765 lifetime
    windows, and accepted #770 mutual transports.  `quarks` overrides the
    constituent reads (so the binding stays COHERENT with an evidence
    bundle that carries custom legs)."""
    groups, _fine, coarse = _modular_hierarchy()
    super_level = coarse.components[0].id.level()
    conn = obs.FiberConnection()
    A, B = _unit_fiber(1, 3), _unit_fiber(11, 3)
    good = _phase_link(conn, A, B, 0.3)
    leaky = conn.transport(A, B, np.diag([0.5, 1.0, 1.0]).astype(complex))
    kinds = kinds if quarks is None else tuple(range(len(quarks)))
    out = []
    for i, kind in enumerate(kinds):
        cand = obs.BoundCandidateEvidence()
        level = super_level - 1 if levels is None else levels[i]
        cand.quark = (quarks[i] if quarks is not None
                      else _relabel_quark(_ud_quark(kind), level=level,
                                          tag="%02x" % (0xa0 + i)))
        cand.support = (list(groups[i % len(groups)]) if supports is None
                        else list(supports[i]))
        cand.lifetime = ((2, 6) if lifetimes is None else lifetimes[i])
        if transports:
            cand.mutualTransports = [good if leakage_ok else leaky]
        out.append(cand)
    return out


def _binding(**kw):
    """The certified bound-supercomponent read of the planted hierarchy."""
    _groups, _fine, coarse = _modular_hierarchy()
    reads = obs.ParticleClusters().boundSupercomponentSearch(
        coarse.components, _bound_candidates(**kw))
    return reads[0]


def _baryon_evidence(kinds=("u", "u", "d"), spin="sharp", rotation_turns=1,
                     color=None, flux=None, samples=None, binding=None,
                     continuum=False, spin_lift=None, class_variances=None,
                     dense_j2=None, quarks=None, exchange=None,
                     crossing_mass=None, crossing_baryon=None):
    """A complete #775 three-cluster evidence bundle."""
    ev = obs.BaryonCandidateEvidence()
    _groups, _fine, coarse = _modular_hierarchy()
    ev.boundComponent = coarse.components[0].id
    if quarks is None:
        candidates = _bound_candidates(kinds=kinds)
        ev.quarks = [c.quark for c in candidates]
    else:
        ev.quarks = list(quarks)
    if binding is not None:
        ev.binding = binding
    elif quarks is None:
        ev.binding = _binding(kinds=kinds)
    else:
        # the binding stays COHERENT with the supplied legs: the read must
        # contain exactly these three label-free identities.
        ev.binding = _binding(quarks=list(quarks))
    ev.colorColumns = _color_triad() if color is None else np.asarray(color)
    ev.colorFlux = _filled_triplet_flux() if flux is None else flux
    ev.rotation = _rotation_character(turns=rotation_turns)
    ev.continuumSpinClaim = continuum
    if spin_lift is not None:
        ev.spinLift = spin_lift
    if exchange is not None:
        ev.exchange = exchange
    if spin == "sharp":
        ev.spinSquaredRead, ev.spinVarianceRead = _sharp_spin_reads()
    elif spin == "generic":
        ev.spinSquaredRead, ev.spinVarianceRead = _generic_slater_spin_reads()
    elif spin == "delta":
        ev.spinSquaredRead, ev.spinVarianceRead = _delta_spin_reads()
    elif spin == "expectation-only":
        ev.spinSquaredRead, _v = _sharp_spin_reads()
    elif spin == "none":
        pass
    if class_variances is not None:
        ev.classVarianceReads = list(class_variances)
    if dense_j2 is not None:
        ev.totalSpaceJ2 = dense_j2
    ev.scaleSamples = _scale_samples() if samples is None else samples
    ev.persistenceLifetime = 4.0
    if crossing_mass is not None:
        ev.crossingMass = crossing_mass
    if crossing_baryon is not None:
        ev.crossingBaryon = crossing_baryon
    return ev


# --------------------------------------------------------------------------- #
# world-tube crossing evidence (#807): a real causal cone, real bands
# --------------------------------------------------------------------------- #
_CONE_M0 = [0]
_CONE_RUNGS = ((0, 1), (0, 2), (0, 3))


def _crossing_cone():
    """A cone cobordism: M0 = {0}, upper spacelike triangle 1-2-3, three
    TIMELIKE rungs with l = i so each rung's proper time is exactly 1."""
    sig = tessera.Signature(4, tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    st = tessera.Spacetime(metric, tessera.HERMITIAN_WEIGHTED, 1.0, 1.0,
                           tessera.PREFERRED, tessera.Toroid())
    verts = [st.createVertex(i) for i in range(4)]
    for a, b in [(0, 1), (0, 2), (0, 3), (1, 2), (2, 3), (1, 3)]:
        st.createSimplex([verts[a], verts[b]])
    for e in st.getEdgeList().toVector():
        e.setLength(1.0 + 0j)
        e.setPhase(0.0)
    for a, b in _CONE_RUNGS:
        for e in st.getEdgeList().toVector():
            if {e.getSource().getId(), e.getTarget().getId()} == {a, b}:
                e.setLength(1j)
    return st


_CROSSING_BAND_RECORD = None


def _crossing_band(cell):
    """A rank-one degree-one band on `cell`, carrying the REAL certificate a
    tracker issued for an accepted positive band -- never a fabricated one."""
    global _CROSSING_BAND_RECORD
    if _CROSSING_BAND_RECORD is None:
        sig = tessera.Signature(4, tessera.Lorentzian)
        metric = tessera.Metric(True, sig)
        st = tessera.Spacetime(metric, tessera.HERMITIAN_WEIGHTED, 1.0, 1.0,
                               tessera.PREFERRED, tessera.Toroid())
        verts = [st.createVertex(i) for i in range(3)]
        for a, b in [(0, 1), (1, 2), (0, 2)]:
            st.createSimplex([verts[a], verts[b]])
        for e in st.getEdgeList().toVector():
            e.setLength(1.0 + 0j)
            e.setPhase(0.0)
        # Subject: the crossing conjunct, not localization.  The 3-cycle is
        # vertex-transitive, so every band carries localization excess
        # exactly 1; declare the permissive analysis cap.
        band_cfg = obs.SpectralFiberConfig()
        band_cfg.maxLocalizationExcess = 1.0
        tracker = obs.SpectralFiberTracker(st, band_cfg)
        for fiber in tracker.enumerateBands([0, 1, 2], 1).fibers:
            cert = fiber.certificate()
            if (fiber.rank() == 1 and cert.accepted
                    and cert.positiveSignature == 1
                    and cert.negativeSignature == 0):
                _CROSSING_BAND_RECORD = fiber.toRecord()
                break
        else:
            raise AssertionError("no certified rank-one positive band")
    record = {k: (list(v) if isinstance(v, list) else v)
              for k, v in _CROSSING_BAND_RECORD.items()}
    record["cells"] = [list(cell)]
    record["rows"] = 1
    record["rank"] = 1
    for name in ("right_frame", "left_frame"):
        record[f"{name}_re"] = [1.0]
        record[f"{name}_im"] = [0.0]
    record["weights_re"] = [1.0]
    record["weights_im"] = [0.0]
    record["eigenvalues_re"] = [0.0]
    record["eigenvalues_im"] = [0.0]
    return obs.SpectralFiber.fromRecord(record)


def _crossing_reads(orientations=(1, 1, 1), windings=None):
    """The crossing mass and coherent baryon sum of three quark tubes on the
    cone, read at the level 0.5 against the M0 reference level 0.0."""
    temporal = obs.CrossingReadouts.temporalFunction(
        _crossing_cone(), _CONE_M0)
    windings = windings or [None] * len(orientations)
    tubes = []
    for index, (orientation, winding) in enumerate(zip(orientations, windings)):
        tube = obs.WorldTubeInput()
        tube.tubeId = f"t{index}"
        tube.band = _crossing_band(_CONE_RUNGS[index])
        tube.orientation = orientation
        tube.certifiedQuarkTube = True
        if winding is not None:
            tube.determinantWinding = winding
        tubes.append(tube)
    mass = obs.CrossingReadouts.crossingMass(tubes, temporal, 0.5, 0.0)
    baryon = obs.CrossingReadouts.baryonNumber(tubes, temporal, 0.5, 0.0)
    return mass, baryon


class TestCrossingReadoutGate(unittest.TestCase):
    """The whitepaper's world-tube crossing conjunct on the baryon
    certificate (#807).  APPLICABLE-GATED exactly like `spin-lift`: a
    candidate assembled without crossing evidence is graded as it was before
    the readouts existed, while supplied evidence is ENFORCED."""

    def setUp(self):
        self.pc = obs.ParticleClusters()

    def test_absent_evidence_passes_vacuously(self):
        """Backwards compatibility: no crossing evidence means the gate is
        not applicable and never appears among the failures."""
        read = self.pc.classifyBaryon(_baryon_evidence())
        self.assertFalse(read.crossingMassApplicable)
        self.assertNotIn("crossing-readouts", read.failedCertificates)
        self.assertIsNone(read.crossingBaryonNumber)
        self.assertTrue(math.isnan(read.crossingMassValue))

    def test_three_forward_tubes_satisfy_the_gate(self):
        mass, baryon = _crossing_reads((1, 1, 1))
        self.assertAlmostEqual(baryon.baryonNumber, 1.0, delta=MACHINE)
        read = self.pc.classifyBaryon(
            _baryon_evidence(crossing_mass=mass, crossing_baryon=baryon))
        self.assertTrue(read.crossingMassApplicable)
        self.assertNotIn("crossing-readouts", read.failedCertificates)
        self.assertAlmostEqual(read.crossingBaryonNumber, 1.0, delta=MACHINE)
        self.assertAlmostEqual(read.crossingMassValue, 3.0, delta=MACHINE)

    def test_wrong_baryon_number_fails_the_gate_by_name(self):
        """Two forward tubes and one reversed give B = 1/3, not 1."""
        mass, baryon = _crossing_reads((1, 1, -1))
        self.assertAlmostEqual(baryon.baryonNumber, 1.0 / 3.0, delta=MACHINE)
        read = self.pc.classifyBaryon(
            _baryon_evidence(crossing_mass=mass, crossing_baryon=baryon))
        self.assertIn("crossing-readouts", read.failedCertificates)

    def test_determinant_sign_defect_fails_the_gate_by_name(self):
        """A tube whose crossing sign disagrees with its certified winding is
        a defect signal, and the certificate refuses on it."""
        mass, baryon = _crossing_reads((1, 1, 1), windings=[1, 1, -1])
        self.assertEqual(list(baryon.signDefects), ["t2"])
        read = self.pc.classifyBaryon(
            _baryon_evidence(crossing_mass=mass, crossing_baryon=baryon))
        self.assertIn("crossing-readouts", read.failedCertificates)
        self.assertEqual(list(read.crossingSignDefects), ["t2"])

    def test_half_a_bundle_fails_rather_than_grading_half(self):
        mass, _baryon = _crossing_reads((1, 1, 1))
        read = self.pc.classifyBaryon(_baryon_evidence(crossing_mass=mass))
        self.assertTrue(read.crossingMassApplicable)
        self.assertIn("crossing-readouts", read.failedCertificates)

    def test_crossing_fields_survive_the_record_round_trip(self):
        mass, baryon = _crossing_reads((1, 1, 1), windings=[1, 1, -1])
        read = self.pc.classifyBaryon(
            _baryon_evidence(crossing_mass=mass, crossing_baryon=baryon))
        rebuilt = obs.BaryonRead.fromRecord(read.toRecord())
        self.assertEqual(rebuilt.crossingMassApplicable,
                         read.crossingMassApplicable)
        self.assertAlmostEqual(rebuilt.crossingMassValue,
                               read.crossingMassValue, delta=MACHINE)
        self.assertAlmostEqual(rebuilt.crossingBaryonNumber,
                               read.crossingBaryonNumber, delta=MACHINE)
        self.assertEqual(list(rebuilt.crossingSignDefects),
                         list(read.crossingSignDefects))


class TestColorSingletCertificate(unittest.TestCase):
    """Design spec 16.3 / whitepaper: S_ABC = det[c_A c_B c_C] and the Gram
    determinant det(C^dag C), with the three-mode wedge built exactly ONCE."""

    def setUp(self):
        self.pc = obs.ParticleClusters()

    def test_orthonormal_triad_gives_unit_gram_determinant(self):
        read = self.pc.classifyBaryon(_baryon_evidence())
        self.assertAlmostEqual(read.colorGramDeterminant, 1.0, delta=MACHINE)
        self.assertAlmostEqual(abs(read.colorWedge), 1.0, delta=MACHINE)
        self.assertNotIn("color-singlet", read.failedCertificates)

    def test_gram_is_the_color_fiber_singlet_gram(self):
        # the delegation is PINNED: |det C|^2 read off the single wedge
        # equals the #767 kernel's own det(C^dag C).
        columns = _color_triad()
        read = self.pc.classifyBaryon(_baryon_evidence(color=columns))
        self.assertAlmostEqual(read.colorGramDeterminant,
                               obs.ColorFiber.singletGram(columns),
                               delta=MACHINE)
        self.assertAlmostEqual(
            abs(read.colorWedge - obs.ColorFiber.colorWedge(columns)),
            0.0, delta=MACHINE)

    def test_wedge_is_su3_invariant(self):
        g = _su3_element()
        self.assertTrue(obs.ColorFiber.isSpecialUnitary(g))
        base = self.pc.classifyBaryon(_baryon_evidence())
        rotated = self.pc.classifyBaryon(
            _baryon_evidence(color=g @ _color_triad()))
        self.assertLess(abs(base.colorWedge - rotated.colorWedge), 1e-14)
        self.assertAlmostEqual(base.colorGramDeterminant,
                               rotated.colorGramDeterminant, delta=MACHINE)
        self.assertEqual(base.classification, rotated.classification)

    def test_duplicate_color_modes_fail_the_singlet_certificate(self):
        columns = _color_triad()
        columns[:, 2] = columns[:, 0]        # duplicated color mode
        read = self.pc.classifyBaryon(_baryon_evidence(color=columns))
        self.assertLess(abs(read.colorGramDeterminant), 1e-25)
        self.assertLess(abs(read.colorWedge), 1e-13)
        self.assertIn("color-singlet", read.failedCertificates)
        self.assertEqual(read.classification, "baryon-candidate")

    def test_wedge_is_built_once_no_extra_fermion_sign(self):
        # A transposition of two color columns flips det C (the epsilon is
        # ALREADY inside the determinant) and leaves the SINGLET certificate
        # |det C|^2 exactly invariant.  No second fermion sign is applied:
        # the composite statistics come from the constituent parities.
        columns = _color_triad()
        swapped = columns[:, [1, 0, 2]]
        a = self.pc.classifyBaryon(_baryon_evidence(color=columns))
        b = self.pc.classifyBaryon(_baryon_evidence(color=swapped))
        self.assertLess(abs(a.colorWedge + b.colorWedge), 1e-14)
        self.assertAlmostEqual(a.colorGramDeterminant,
                               b.colorGramDeterminant, delta=MACHINE)
        self.assertEqual(a.exteriorParity, b.exteriorParity)
        self.assertEqual(a.classification, b.classification)

    def test_missing_color_evidence_is_unknown_never_zero(self):
        read = self.pc.classifyBaryon(
            _baryon_evidence(color=np.zeros((3, 3), dtype=complex)))
        self.assertTrue(math.isnan(read.colorGramDeterminant))
        self.assertTrue(math.isnan(read.colorWedge.real))
        self.assertIn("color-singlet", read.failedCertificates)

    def test_unnormalized_columns_are_normalized_once(self):
        # scaling a column is a frame convention, not physics: the singlet
        # certificate is unchanged because normalization happens once,
        # before the single wedge.
        columns = _color_triad()
        columns[:, 0] *= 7.5
        read = self.pc.classifyBaryon(_baryon_evidence(color=columns))
        self.assertAlmostEqual(read.colorGramDeterminant, 1.0, delta=MACHINE)
        self.assertEqual(read.classification, "certified-proton")

    def test_collinear_columns_are_degenerate(self):
        columns = _color_triad()
        columns[:, 1] = columns[:, 0] + columns[:, 2]
        read = self.pc.classifyBaryon(_baryon_evidence(color=columns))
        self.assertLess(read.colorGramDeterminant, 1e-25)
        self.assertIn("color-singlet", read.failedCertificates)


class TestNetColorFluxDiagnostic(unittest.TestCase):
    """The INDEPENDENT vanishing net-color-flux check (never on its own a
    proof of confinement on a finite complex)."""

    def setUp(self):
        self.pc = obs.ParticleClusters()

    def test_filled_triplet_carries_zero_net_color_flux(self):
        flux = _filled_triplet_flux()
        self.assertEqual(flux.octetWeight, 0.0)
        self.assertAlmostEqual(flux.singletWeight, 3.0, delta=MACHINE)
        read = self.pc.classifyBaryon(_baryon_evidence(flux=flux))
        self.assertEqual(read.colorFlux, 0.0)
        self.assertNotIn("color-flux-zero", read.failedCertificates)

    def test_polarized_color_state_fails_flux_zero(self):
        flux = _polarized_flux()
        self.assertAlmostEqual(flux.octetWeight, 2.0 / 3.0, delta=MACHINE)
        read = self.pc.classifyBaryon(_baryon_evidence(flux=flux))
        self.assertAlmostEqual(read.colorFlux, 2.0 / 3.0, delta=MACHINE)
        self.assertIn("color-flux-zero", read.failedCertificates)
        self.assertEqual(read.classification, "baryon-candidate")

    def test_missing_octet_read_fails_by_name(self):
        read = self.pc.classifyBaryon(
            _baryon_evidence(flux=obs.OctetBilinearRead()))
        self.assertTrue(math.isnan(read.colorFlux))
        self.assertIn("color-flux-zero", read.failedCertificates)

    def test_flux_is_independent_of_the_gram_certificate(self):
        # unit Gram columns with a POLARIZED carried state: the singlet
        # certificate passes and the flux diagnostic refuses on its own.
        read = self.pc.classifyBaryon(_baryon_evidence(flux=_polarized_flux()))
        self.assertAlmostEqual(read.colorGramDeterminant, 1.0, delta=MACHINE)
        self.assertNotIn("color-singlet", read.failedCertificates)
        self.assertIn("color-flux-zero", read.failedCertificates)


class TestBoundSupercomponentSearch(unittest.TestCase):
    """Design spec 16.2: the next modular level, three lifetime-overlapping
    quark candidates, containment, and bounded mutual transport."""

    GATES = ["supercomponent-level", "quark-count", "support-containment",
             "lifetime-overlap", "transport-containment"]

    def setUp(self):
        self.pc = obs.ParticleClusters()
        self.groups, self.fine, self.coarse = _modular_hierarchy()

    def test_planted_hierarchy_has_two_modular_levels(self):
        # the real #765 discovery: three level-1 cliques, ONE level-2
        # community containing all of them.
        self.assertEqual(len(self.fine.components), 3)
        self.assertEqual(len(self.coarse.components), 1)
        self.assertEqual(
            sorted(tuple(c.support) for c in self.fine.components),
            sorted(tuple(g) for g in self.groups))
        self.assertGreater(self.coarse.components[0].id.level(),
                           self.fine.components[0].id.level())

    def test_certified_bound_supercomponent(self):
        read = self.pc.boundSupercomponentSearch(self.coarse.components,
                                                 _bound_candidates())
        self.assertEqual(len(read), 1)
        self.assertTrue(read[0].found)
        self.assertEqual(read[0].failedCertificates, [])
        self.assertEqual(len(read[0].quarks), 3)
        self.assertEqual(read[0].quarkIndices, [0, 1, 2])
        self.assertEqual(read[0].lifetimeWindow, (2, 6))
        self.assertEqual(read[0].lifetimeOverlap, 5.0)
        self.assertEqual(read[0].minContainment, 1.0)
        self.assertTrue(read[0].certificate.holds())

    def test_same_level_is_not_the_next_modular_level(self):
        level = self.coarse.components[0].id.level()
        read = self.pc.boundSupercomponentSearch(
            self.coarse.components,
            _bound_candidates(levels=[level] * 3))[0]
        self.assertFalse(read.found)
        self.assertIn("supercomponent-level", read.failedCertificates)

    def test_two_candidates_fail_the_quark_count(self):
        read = self.pc.boundSupercomponentSearch(
            self.coarse.components, _bound_candidates(kinds=("u", "d")))[0]
        self.assertFalse(read.found)
        self.assertIn("quark-count", read.failedCertificates)
        self.assertEqual(len(read.quarks), 2)

    def test_four_candidates_fail_the_quark_count(self):
        read = self.pc.boundSupercomponentSearch(
            self.coarse.components,
            _bound_candidates(kinds=("u", "u", "d", "d")))[0]
        self.assertFalse(read.found)
        self.assertIn("quark-count", read.failedCertificates)
        self.assertEqual(len(read.quarks), 4)

    def test_uncertified_candidate_is_not_a_quark_candidate(self):
        candidates = _bound_candidates()
        # an antiquark leg: a certified read, but not a "quark" verdict
        candidates[2].quark = self.pc.classifyQuark(
            _certified_evidence(turns=-1))
        read = self.pc.boundSupercomponentSearch(self.coarse.components,
                                                 candidates)[0]
        self.assertEqual(len(read.quarks), 2)
        self.assertIn("quark-count", read.failedCertificates)

    def test_support_escaping_the_supercomponent_fails_containment(self):
        supports = [list(self.groups[0]), list(self.groups[1]),
                    list(self.groups[2]) + [999]]
        read = self.pc.boundSupercomponentSearch(
            self.coarse.components,
            _bound_candidates(supports=supports))[0]
        self.assertFalse(read.found)
        self.assertIn("support-containment", read.failedCertificates)
        self.assertAlmostEqual(read.minContainment, 6.0 / 7.0, delta=MACHINE)

    def test_disjoint_lifetimes_fail_the_overlap(self):
        read = self.pc.boundSupercomponentSearch(
            self.coarse.components,
            _bound_candidates(lifetimes=[(0, 1), (2, 3), (4, 5)]))[0]
        self.assertFalse(read.found)
        self.assertIn("lifetime-overlap", read.failedCertificates)
        self.assertEqual(read.lifetimeOverlap, 0.0)
        self.assertIsNone(read.lifetimeWindow)

    def test_missing_lifetime_is_unknown_never_presumed(self):
        read = self.pc.boundSupercomponentSearch(
            self.coarse.components,
            _bound_candidates(lifetimes=[(2, 6), None, (2, 6)]))[0]
        self.assertFalse(read.found)
        self.assertIn("lifetime-overlap", read.failedCertificates)
        self.assertIsNone(read.lifetimeWindow)

    def test_partial_lifetime_overlap_is_measured(self):
        read = self.pc.boundSupercomponentSearch(
            self.coarse.components,
            _bound_candidates(lifetimes=[(0, 4), (3, 9), (2, 6)]))[0]
        self.assertTrue(read.found)
        self.assertEqual(read.lifetimeWindow, (3, 4))
        self.assertEqual(read.lifetimeOverlap, 2.0)

    def test_leaky_mutual_transport_fails_containment(self):
        read = self.pc.boundSupercomponentSearch(
            self.coarse.components,
            _bound_candidates(leakage_ok=False))[0]
        self.assertFalse(read.found)
        self.assertIn("transport-containment", read.failedCertificates)

    def test_missing_transports_fail_by_name(self):
        read = self.pc.boundSupercomponentSearch(
            self.coarse.components, _bound_candidates(transports=False))[0]
        self.assertFalse(read.found)
        self.assertIn("transport-containment", read.failedCertificates)
        self.assertEqual(read.transportCount, 0)
        self.assertTrue(math.isnan(read.transportLeakageMax))

    def test_components_without_candidates_emit_no_read(self):
        candidates = _bound_candidates(
            supports=[[900], [901], [902]])
        self.assertEqual(
            self.pc.boundSupercomponentSearch(self.coarse.components,
                                              candidates), [])

    def test_fine_components_are_not_supercomponents(self):
        # each level-1 clique contains exactly ONE candidate: three reads,
        # none of them bound.
        reads = self.pc.boundSupercomponentSearch(self.fine.components,
                                                  _bound_candidates())
        self.assertEqual(len(reads), 3)
        for read in reads:
            self.assertFalse(read.found)
            self.assertIn("quark-count", read.failedCertificates)

    def test_thresholds_and_describe_travel(self):
        read = self.pc.boundSupercomponentSearch(self.coarse.components,
                                                 _bound_candidates())[0]
        self.assertEqual(read.thresholds.minSupportContainment, 1.0)
        self.assertIn("bound", read.describe())

    def test_candidate_order_does_not_change_the_verdict(self):
        candidates = _bound_candidates()
        a = self.pc.boundSupercomponentSearch(self.coarse.components,
                                              candidates)[0]
        b = self.pc.boundSupercomponentSearch(
            self.coarse.components, list(reversed(candidates)))[0]
        self.assertEqual(a.found, b.found)
        self.assertEqual(a.lifetimeOverlap, b.lifetimeOverlap)
        self.assertEqual(sorted(q.canonicalHash() for q in a.quarks),
                         sorted(q.canonicalHash() for q in b.quarks))


class TestScaleProfile(unittest.TestCase):
    """The refinement-window read over the EXISTING #575/#566/#593
    mass-radius battery: a finite radius plus refinement-stable
    DIMENSIONLESS channels.  Nothing here is a form factor (see the header
    banner) and no dimensionful mass is ever emitted."""

    GATES = ["refinement-window", "finite-radius", "radius-ratio-stability",
             "spectral-mass-stability", "localization-stability",
             "profile-stability", "color-gram-stability",
             "rotation-character-stability", "baryon-flux-stability",
             "electric-flux-stability", "composite-parity-stability",
             "anchor-score-stability"]

    def setUp(self):
        self.pc = obs.ParticleClusters()

    def test_stable_window_certifies(self):
        read = self.pc.scaleProfile(_scale_samples())
        self.assertTrue(read.stable)
        self.assertEqual(read.failedCertificates, [])
        self.assertEqual(read.sampleCount, 3)
        self.assertEqual(read.radius, 0.75)
        self.assertTrue(read.radiusFinite)
        self.assertAlmostEqual(read.radiusRatio, 1.5, delta=MACHINE)
        self.assertEqual(read.spectralMass, 2.25)
        self.assertEqual(read.radiusRatioSpread, 0.0)
        self.assertEqual(read.spectralMassSpread, 0.0)
        self.assertEqual(read.profileMaxDeviation, 0.0)
        self.assertEqual(read.profileShells, 3)
        self.assertTrue(read.certificate.holds())
        self.assertEqual(read.certificate.grade,
                         cob.CertificateGrade.CertifiedNumerical)

    def test_single_sample_cannot_measure_stability(self):
        read = self.pc.scaleProfile(_scale_samples(count=1))
        self.assertFalse(read.stable)
        self.assertIn("refinement-window", read.failedCertificates)
        # every stability channel is UNMEASURED (NaN), never zero
        self.assertTrue(math.isnan(read.radiusRatioSpread))
        self.assertTrue(math.isnan(read.profileMaxDeviation))
        self.assertTrue(read.radiusFinite)   # the radius itself is finite

    def test_empty_window_is_not_stable(self):
        read = self.pc.scaleProfile([])
        self.assertFalse(read.stable)
        self.assertFalse(read.radiusFinite)
        self.assertEqual(sorted(read.failedCertificates), sorted(self.GATES))

    def test_infinite_radius_fails(self):
        samples = _scale_samples()
        samples[1].radius = float("inf")
        read = self.pc.scaleProfile(samples)
        self.assertFalse(read.radiusFinite)
        self.assertIn("finite-radius", read.failedCertificates)

    def test_nan_radius_fails(self):
        samples = _scale_samples()
        samples[0].radius = NAN
        read = self.pc.scaleProfile(samples)
        self.assertFalse(read.radiusFinite)
        self.assertIn("finite-radius", read.failedCertificates)

    def test_nonpositive_radius_fails(self):
        samples = _scale_samples(radius=0.0)
        read = self.pc.scaleProfile(samples)
        self.assertFalse(read.radiusFinite)
        self.assertIn("finite-radius", read.failedCertificates)

    def test_drifting_radius_ratio_fails(self):
        read = self.pc.scaleProfile(_scale_samples(drift=0.05))
        self.assertFalse(read.stable)
        self.assertIn("radius-ratio-stability", read.failedCertificates)
        # ratios 1.5/1.6/1.7 (cross = 0.5): (max - min)/max(|mean|, 1)
        # = 0.2/1.6 = 0.125 exactly.
        self.assertAlmostEqual(read.radiusRatioSpread, 0.125, delta=1e-12)

    def test_drifting_spectral_mass_fails(self):
        samples = _scale_samples()
        samples[2].spectralMass = 2.30
        read = self.pc.scaleProfile(samples)
        self.assertFalse(read.stable)
        self.assertIn("spectral-mass-stability", read.failedCertificates)

    def test_drifting_localization_fails(self):
        samples = _scale_samples()
        samples[1].localization = 0.4
        read = self.pc.scaleProfile(samples)
        self.assertFalse(read.stable)
        self.assertIn("localization-stability", read.failedCertificates)

    def test_drifting_radial_profile_fails(self):
        read = self.pc.scaleProfile(_scale_samples(profile_drift=0.01))
        self.assertFalse(read.stable)
        self.assertIn("profile-stability", read.failedCertificates)
        self.assertAlmostEqual(read.profileMaxDeviation, 0.02, delta=1e-12)

    def test_missing_radial_profile_fails_by_name(self):
        # no shell seeds: the radial profile is UNKNOWN, never "stable at
        # zero" (the honest reading of an unavailable channel).
        read = self.pc.scaleProfile(_scale_samples(profile=()))
        self.assertFalse(read.stable)
        self.assertIn("profile-stability", read.failedCertificates)
        self.assertTrue(math.isnan(read.profileMaxDeviation))
        self.assertEqual(read.profileShells, 0)

    def test_shell_count_mismatch_fails(self):
        samples = _scale_samples()
        samples[1].radialWeightProfile = [0.5, 0.5]
        read = self.pc.scaleProfile(samples)
        self.assertFalse(read.stable)
        self.assertIn("profile-stability", read.failedCertificates)
        self.assertTrue(math.isnan(read.profileMaxDeviation))

    def test_physical_mass_is_always_unknown(self):
        # keeping any dimensionful mass UNKNOWN until a physical scale is
        # independently established (ticket scope, stated verbatim).
        read = self.pc.scaleProfile(_scale_samples())
        self.assertTrue(read.stable)
        self.assertIsNone(read.physicalMass)

    def test_describe_names_the_failures(self):
        read = self.pc.scaleProfile(_scale_samples(count=1))
        self.assertIn("refinement-window", read.describe())
        self.assertIn("physical mass unknown", read.describe())


class TestScaleProfileFromTheExistingBattery(unittest.TestCase):
    """The adapter over RegisterContext.interiorHinges — the same reader
    EmergentRadius / EmergentMass use, on the hand-checkable closed forms."""

    @staticmethod
    def _boundary_delta5():
        st = tessera.Spacetime.fromVertexTuples(
            4, [list(c) for c in itertools.combinations(range(6), 5)],
            1.0, 0.0)
        st.materializeFacets()
        return st

    @staticmethod
    def _star_of_apex():
        st = tessera.Spacetime.fromVertexTuples(
            4, [list(c) for c in itertools.combinations(range(6), 5)
                if 5 in c], 1.0, 0.0)
        st.materializeFacets()
        return st

    def test_closed_s4_sample_matches_the_closed_forms(self):
        # dDelta^5: every deficit is 2pi - 3*arccos(1/4) and the six unit
        # pentatopes have total 4-volume 6*sqrt(5)/96 (the #575 anchors).
        deficit = 2.0 * math.pi - 3.0 * math.acos(0.25)
        volume = 6.0 * math.sqrt(5.0) / 96.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ctx = obs.RegisterContext(self._boundary_delta5(), 0, 3,
                                      cob.ProtonSynthesis.singlet())
        sample = obs.ParticleClusters.scaleProfileSample(ctx)
        self.assertAlmostEqual(sample.radius, volume ** 0.25, places=12)
        self.assertAlmostEqual(sample.radiusCrossCheck, volume ** 0.25,
                               places=12)
        self.assertAlmostEqual(sample.spectralMass, deficit, places=9)
        self.assertAlmostEqual(sample.localization, 1.0, places=9)
        # no emergent holes seed the BFS: the radial profile is UNKNOWN
        self.assertEqual(sample.radialWeightProfile, [])

    def test_closed_s4_window_is_stable_but_has_no_radial_profile(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ctx = obs.RegisterContext(self._boundary_delta5(), 0, 3,
                                      cob.ProtonSynthesis.singlet())
        sample = obs.ParticleClusters.scaleProfileSample(ctx)
        read = obs.ParticleClusters().scaleProfile([sample, sample])
        self.assertTrue(read.radiusFinite)
        self.assertEqual(read.radiusRatioSpread, 0.0)
        self.assertEqual(read.spectralMassSpread, 0.0)
        self.assertFalse(read.stable)
        # The battery sample fills the MASS-RADIUS channels only.  Since
        # #808 the window also carries the candidate's other dimensionless
        # certificates, which this geometry read cannot know: they are
        # UNKNOWN and are NAMED, never passed vacuously.
        self.assertEqual(read.failedCertificates,
                         ["profile-stability", "color-gram-stability",
                          "rotation-character-stability",
                          "baryon-flux-stability", "electric-flux-stability",
                          "composite-parity-stability",
                          "anchor-score-stability"])

    def test_hole_seeded_star_carries_a_radial_profile(self):
        # the dropped pentatope {0..4} is the hole seeding the BFS
        # shells: one shell carrying the whole curvature weight.
        ctx = obs.RegisterContext(self._star_of_apex(), [[0, 1, 2, 3, 4]], 1,
                                  3, cob.ProtonSynthesis.singlet())
        sample = obs.ParticleClusters.scaleProfileSample(ctx)
        self.assertEqual(sample.radialWeightProfile, [1.0])
        deficit = 2.0 * math.pi - 3.0 * math.acos(0.25)
        self.assertAlmostEqual(sample.spectralMass, deficit, places=9)
        self.assertGreater(sample.radius, 0.0)
        read = obs.ParticleClusters().scaleProfile([sample, sample])
        self.assertEqual(read.profileShells, 1)
        self.assertEqual(read.profileMaxDeviation, 0.0)
        # Every BATTERY channel is stable; the candidate's remaining
        # dimensionless certificates were never supplied by this geometry
        # read, so they are named (#808) and the window is not `stable`.
        self.assertFalse(read.stable)
        for name in ("refinement-window", "finite-radius",
                     "radius-ratio-stability", "spectral-mass-stability",
                     "localization-stability", "profile-stability"):
            self.assertNotIn(name, read.failedCertificates)
        self.assertEqual(read.failedCertificates,
                         ["color-gram-stability",
                          "rotation-character-stability",
                          "baryon-flux-stability", "electric-flux-stability",
                          "composite-parity-stability",
                          "anchor-score-stability"])
        # Filling them from the candidate's own certificates completes the
        # window: the battery read is one supplier, not the whole list.
        for entry in (sample,):
            entry.colorGramDeterminant = 1.0
            entry.rotationCharacter = -1.0 + 0j
            entry.baryonFlux = 1.0
            entry.electricFlux = 1.0
            entry.compositeParity = -1
            entry.anchorScore = 0.788
        completed = obs.ParticleClusters().scaleProfile([sample, sample])
        self.assertTrue(completed.stable)
        self.assertEqual(completed.failedCertificates, [])

    def test_sample_is_read_only_on_the_context(self):
        st = self._star_of_apex()
        ctx = obs.RegisterContext(st, [[0, 1, 2, 3, 4]], 1, 3,
                                  cob.ProtonSynthesis.singlet())
        before = (len(st.getTopSimplices()), len(st.getSimplices()))
        a = obs.ParticleClusters.scaleProfileSample(ctx)
        b = obs.ParticleClusters.scaleProfileSample(ctx)
        self.assertEqual(before,
                         (len(st.getTopSimplices()), len(st.getSimplices())))
        self.assertEqual(a.radius, b.radius)
        self.assertEqual(a.radialWeightProfile, b.radialWeightProfile)


class TestBaryonClassification(unittest.TestCase):
    """Design spec 16.4: the complete proton certificate and the four-way
    verdict."""

    def setUp(self):
        self.pc = obs.ParticleClusters()

    def test_certified_proton(self):
        read = self.pc.classifyBaryon(_baryon_evidence())
        self.assertEqual(read.classification, "certified-proton")
        self.assertEqual(read.failedCertificates, [])
        self.assertEqual(read.confidence, 1.0)
        # every row of the spec 16.4 table
        self.assertAlmostEqual(read.colorGramDeterminant, 1.0, delta=MACHINE)
        self.assertEqual(read.colorFlux, 0.0)
        self.assertEqual(read.totalWinding, 3)
        self.assertAlmostEqual(read.baryonFlux, 1.0, delta=MACHINE)
        self.assertEqual(read.flavorPattern, "uud")
        self.assertAlmostEqual(read.totalIsospin, 0.5, delta=MACHINE)
        self.assertAlmostEqual(read.electricFlux, 1.0, delta=1e-9)
        self.assertAlmostEqual(read.totalJ2, 0.75, delta=1e-14)
        self.assertLess(abs(read.totalJ2Variance), 1e-13)
        self.assertTrue(read.sharpSpin)
        self.assertEqual(read.rotationCharacterSign, -1)
        self.assertLess(abs(read.rotationCharacter + 1.0), 1e-12)
        self.assertEqual(read.exteriorParity, -1)
        self.assertTrue(read.radiusFinite)
        self.assertTrue(read.profileStable)
        self.assertTrue(read.certificate.holds())
        self.assertEqual(read.certificate.grade,
                         cob.CertificateGrade.StructureExact)

    def test_delta_oracle_is_a_baryon_but_never_a_proton(self):
        # J^2 = 15/4 with Var = 0: a SHARP spin that is simply not 3/4.
        read = self.pc.classifyBaryon(_baryon_evidence(spin="delta"))
        self.assertEqual(read.classification, "baryon-candidate")
        self.assertNotEqual(read.classification, "certified-proton")
        self.assertAlmostEqual(read.totalJ2, 15.0 / 4.0, delta=1e-13)
        self.assertLess(abs(read.totalJ2Variance), 1e-13)
        self.assertTrue(read.sharpSpin)          # sharp, but at 15/4
        self.assertEqual(read.failedCertificates, ["spin-expectation"])

    def test_delta_dense_772_oracle_is_a_baryon_but_never_a_proton(self):
        # the #772 dense total-space oracle: |uuu> -> 15/4 (the exact
        # measuring stick), consulted when no quasi-free read certified.
        dense = obs.ExchangeHolonomy.totalJSquared(
            np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=complex))
        self.assertEqual(dense, 15.0 / 4.0)
        read = self.pc.classifyBaryon(
            _baryon_evidence(spin="none", dense_j2=dense))
        self.assertEqual(read.classification, "baryon-candidate")
        self.assertEqual(read.totalJ2, 15.0 / 4.0)
        self.assertIsNone(read.totalJ2Variance)
        self.assertIn("spin-expectation", read.failedCertificates)
        self.assertIn("sharp-spin", read.failedCertificates)

    def test_dense_772_proton_eigenstate_still_needs_a_variance(self):
        # 2|uud> - |udu> - |duu> -> 3/4 exactly, but a DENSE expectation
        # supplies no variance: expectation alone is never a sharp spin.
        state = np.zeros(8, dtype=complex)
        state[0b001], state[0b010], state[0b100] = 2.0, -1.0, -1.0
        self.assertAlmostEqual(obs.ExchangeHolonomy.totalJSquared(state),
                               0.75, delta=1e-14)
        read = self.pc.classifyBaryon(
            _baryon_evidence(spin="none", dense_j2=0.75))
        self.assertEqual(read.totalJ2, 0.75)
        self.assertIsNone(read.totalJ2Variance)
        self.assertFalse(read.sharpSpin)
        self.assertEqual(read.failedCertificates, ["sharp-spin"])
        self.assertEqual(read.classification, "baryon-candidate")

    def test_generic_slater_expectation_without_sharp_variance(self):
        # THE ticket's scientific point: <J^2> = 3/4 EXACTLY with
        # Var = 15/16 > 0 is NOT a certified proton.
        read = self.pc.classifyBaryon(_baryon_evidence(spin="generic"))
        self.assertAlmostEqual(read.totalJ2, 0.75, delta=1e-13)
        self.assertAlmostEqual(read.totalJ2Variance, 15.0 / 16.0,
                               delta=1e-12)
        self.assertFalse(read.sharpSpin)
        self.assertNotIn("spin-expectation", read.failedCertificates)
        self.assertIn("sharp-spin", read.failedCertificates)
        self.assertNotEqual(read.classification, "certified-proton")

    def test_exact_eigenstate_passes_the_sharp_certificate(self):
        # the other half of the pair: an exact J^2 eigenstate has Var = 0.
        read = self.pc.classifyBaryon(_baryon_evidence(spin="sharp"))
        self.assertAlmostEqual(read.totalJ2, 0.75, delta=1e-14)
        self.assertLess(abs(read.totalJ2Variance), 1e-13)
        self.assertTrue(read.sharpSpin)
        self.assertEqual(read.classification, "certified-proton")

    def test_missing_variance_read_is_unknown_never_zero(self):
        read = self.pc.classifyBaryon(
            _baryon_evidence(spin="expectation-only"))
        self.assertAlmostEqual(read.totalJ2, 0.75, delta=1e-14)
        self.assertIsNone(read.totalJ2Variance)
        self.assertFalse(read.sharpSpin)
        self.assertIn("sharp-spin", read.failedCertificates)

    def test_no_baryon_without_three_certified_quarks(self):
        quarks = list(_baryon_evidence().quarks)
        quarks[2] = self.pc.classifyQuark(_certified_evidence(turns=-1))
        read = self.pc.classifyBaryon(_baryon_evidence(quarks=quarks))
        self.assertEqual(read.classification, "no-baryon")
        self.assertIn("constituent-quarks", read.failedCertificates)

    def test_no_baryon_without_a_bound_supercomponent(self):
        read = self.pc.classifyBaryon(
            _baryon_evidence(binding=obs.BoundSupercomponentRead()))
        self.assertEqual(read.classification, "no-baryon")
        self.assertIn("bound-supercomponent", read.failedCertificates)

    def test_no_baryon_dominates_a_full_proton_certificate(self):
        # the structural gates decide "no baryon" even when every proton
        # certificate below them holds.
        read = self.pc.classifyBaryon(
            _baryon_evidence(binding=_binding(transports=False)))
        self.assertEqual(read.classification, "no-baryon")
        self.assertEqual(read.failedCertificates, ["bound-supercomponent"])

    def test_wrong_baryon_flux_is_named(self):
        quarks = list(_baryon_evidence().quarks)
        quarks[2] = self.pc.classifyQuark(_certified_evidence(turns=1))
        quarks[2] = obs.QuarkRead.fromRecord(quarks[2].toRecord())
        ev = _baryon_evidence()
        # replace one leg with an UNCERTIFIED winding: nu is unknown
        broken = _certified_evidence()
        conn = obs.FiberConnection()
        A, B = _unit_fiber(1, 3), _unit_fiber(11, 3)
        broken.winding = conn.openSegmentWinding(
            [_phase_link(conn, A, B, p) for p in (0.0, 0.4, 0.8)],
            obs.WindingClosureSpec())
        legs = list(ev.quarks)
        legs[2] = self.pc.classifyQuark(broken)
        read = self.pc.classifyBaryon(_baryon_evidence(quarks=legs))
        self.assertIsNone(read.totalWinding)
        self.assertIsNone(read.baryonFlux)   # UNKNOWN, never zero
        self.assertIn("baryon-flux-unit", read.failedCertificates)

    def test_flavor_pattern_uuu_is_not_a_proton(self):
        read = self.pc.classifyBaryon(_baryon_evidence(kinds=("u", "u", "u")))
        self.assertEqual(read.flavorPattern, "uuu")
        self.assertAlmostEqual(read.totalIsospin, 1.5, delta=MACHINE)
        self.assertIn("flavor-uud", read.failedCertificates)
        self.assertIn("electric-flux-unit", read.failedCertificates)
        self.assertEqual(read.classification, "baryon-candidate")

    def test_flavor_pattern_udd_is_not_a_proton(self):
        read = self.pc.classifyBaryon(_baryon_evidence(kinds=("u", "d", "d")))
        self.assertEqual(read.flavorPattern, "udd")
        self.assertAlmostEqual(read.electricFlux, 0.0, delta=1e-9)
        self.assertIn("flavor-uud", read.failedCertificates)
        self.assertIn("electric-flux-unit", read.failedCertificates)

    def test_color_singlet_fixture_with_unknown_flavor_is_partial(self):
        # a color-singlet three-cluster candidate whose constituents have NO
        # certified doublet: flavor AND charge stay unknown and the read is
        # a partial candidate naming exactly those gaps.
        plain = self.pc.classifyQuark(_certified_evidence())
        legs = [_relabel_quark(plain, level=1, tag="%02x" % (0xb0 + i))
                for i in range(3)]
        read = self.pc.classifyBaryon(_baryon_evidence(quarks=legs))
        self.assertEqual(read.classification, "baryon-candidate")
        self.assertAlmostEqual(read.colorGramDeterminant, 1.0, delta=MACHINE)
        self.assertEqual(read.flavorPattern, "")
        self.assertIsNone(read.totalIsospin)
        self.assertIsNone(read.electricFlux)
        self.assertEqual(sorted(read.failedCertificates),
                         ["electric-flux-unit", "flavor-uud"])
        # the certified channels are still reported
        self.assertEqual(read.totalWinding, 3)
        self.assertAlmostEqual(read.baryonFlux, 1.0, delta=MACHINE)

    def test_even_composite_parity_is_named(self):
        legs = list(_baryon_evidence().quarks)
        legs[1] = self.pc.classifyQuark(
            _certified_evidence(occupations=(1.0, 1.0, 0.0)))
        read = self.pc.classifyBaryon(_baryon_evidence(quarks=legs))
        self.assertNotEqual(read.exteriorParity, -1)
        self.assertIn("composite-parity-odd", read.failedCertificates)

    def test_uncertified_constituent_parity_is_unknown(self):
        legs = list(_baryon_evidence().quarks)
        blank = _certified_evidence()
        blank.parityRead = qm.WickCertificateRead()
        legs[0] = self.pc.classifyQuark(blank)
        read = self.pc.classifyBaryon(_baryon_evidence(quarks=legs))
        self.assertEqual(read.exteriorParity, 0)
        self.assertIn("composite-parity-odd", read.failedCertificates)

    def test_rotation_character_plus_one_fails(self):
        # the 4pi cycle: chi_hat = +1, a vector-like cycle, not spin 1/2.
        read = self.pc.classifyBaryon(_baryon_evidence(rotation_turns=2))
        self.assertEqual(read.rotationCharacterSign, +1)
        self.assertIn("rotation-character", read.failedCertificates)
        self.assertNotEqual(read.classification, "certified-proton")

    def test_exchange_channel_is_never_the_rotation_certificate(self):
        # the #772 channels are not interchangeable: an exchange-tagged
        # character leaves the rotation certificate UNKNOWN.
        ev = _baryon_evidence()
        ev.rotation = _exchange_character()
        read = self.pc.classifyBaryon(ev)
        self.assertIsNone(read.rotationCharacter)
        self.assertEqual(read.rotationCharacterSign, 0)
        self.assertIn("rotation-character", read.failedCertificates)

    def test_uncertified_rotation_read_never_emits_a_sign(self):
        # a TIMING mismatch voids the cancellation premise: the #772 read is
        # uncertified, so the certificate is unknown rather than a sign.
        EH = obs.ExchangeHolonomy
        frame0 = EH.transverseSpinorFrame(0, 1, 4)
        weights = np.ones(4, dtype=complex)
        loop = EH.loopHolonomy(
            EH.rotationLoopFrames(frame0, 0, 1, 4, 1, 16), weights)
        mistimed = EH.loopHolonomy(
            EH.referenceLoopFrames(frame0, 8), weights)
        ev = _baryon_evidence()
        ev.rotation = EH.rotationCharacter(loop, mistimed)
        self.assertFalse(ev.rotation.certificate.holds())
        read = self.pc.classifyBaryon(ev)
        self.assertIsNone(read.rotationCharacter)
        self.assertEqual(read.rotationCharacterSign, 0)
        self.assertIn("rotation-character", read.failedCertificates)

    def test_spin_lift_is_not_demanded_without_a_continuum_claim(self):
        read = self.pc.classifyBaryon(_baryon_evidence())
        self.assertFalse(read.spinLiftApplicable)
        self.assertFalse(read.spinLiftAccepted)
        self.assertNotIn("spin-lift", read.failedCertificates)
        self.assertEqual(read.classification, "certified-proton")

    def test_continuum_claim_accepts_a_certified_lift(self):
        lift = _accepted_spin_lift()
        self.assertTrue(lift.liftExists)
        read = self.pc.classifyBaryon(
            _baryon_evidence(continuum=True, spin_lift=lift))
        self.assertTrue(read.spinLiftApplicable)
        self.assertTrue(read.spinLiftAccepted)
        self.assertEqual(read.classification, "certified-proton")

    def test_continuum_claim_without_a_lift_fails_by_name(self):
        read = self.pc.classifyBaryon(_baryon_evidence(continuum=True))
        self.assertTrue(read.spinLiftApplicable)
        self.assertFalse(read.spinLiftAccepted)
        self.assertEqual(read.failedCertificates, ["spin-lift"])

    def test_obstructed_lift_fails_the_continuum_claim(self):
        lift = _obstructed_spin_lift()
        self.assertTrue(lift.obstructed)
        read = self.pc.classifyBaryon(
            _baryon_evidence(continuum=True, spin_lift=lift))
        self.assertFalse(read.spinLiftAccepted)
        self.assertIn("spin-lift", read.failedCertificates)

    def test_missing_radius_is_not_certified(self):
        samples = _scale_samples()
        samples[1].radius = float("inf")
        read = self.pc.classifyBaryon(_baryon_evidence(samples=samples))
        self.assertFalse(read.radiusFinite)
        self.assertIn("finite-radius", read.failedCertificates)
        self.assertNotEqual(read.classification, "certified-proton")

    def test_unstable_profile_is_not_certified(self):
        read = self.pc.classifyBaryon(
            _baryon_evidence(samples=_scale_samples(profile_drift=0.01)))
        self.assertFalse(read.profileStable)
        self.assertAlmostEqual(read.profileMaxDeviation, 0.02, delta=1e-12)
        self.assertIn("profile-stability", read.failedCertificates)
        self.assertNotIn("finite-radius", read.failedCertificates)

    def test_no_scale_evidence_fails_both_scale_gates(self):
        read = self.pc.classifyBaryon(_baryon_evidence(samples=[]))
        self.assertTrue(math.isnan(read.radius))
        self.assertIn("finite-radius", read.failedCertificates)
        self.assertIn("profile-stability", read.failedCertificates)

    def test_physical_mass_is_always_unknown_on_the_read(self):
        read = self.pc.classifyBaryon(_baryon_evidence())
        self.assertEqual(read.classification, "certified-proton")
        self.assertIsNone(read.physicalMass)
        self.assertAlmostEqual(read.spectralMass, 2.25, delta=MACHINE)

    def test_confidence_is_the_passed_fraction(self):
        # Fifteen gates since the world-tube crossing conjunct joined them;
        # with no crossing evidence supplied that gate passes VACUOUSLY, so
        # exactly one certificate (sharp-spin) fails here.
        read = self.pc.classifyBaryon(_baryon_evidence(spin="generic"))
        self.assertAlmostEqual(read.confidence, 14.0 / 15.0, delta=MACHINE)
        self.assertEqual(len(read.failedCertificates), 1)

    def test_thresholds_are_recorded(self):
        cfg = obs.ParticleClustersConfig()
        cfg.spinVarianceTolerance = 2.0     # a class that tolerates anything
        read = obs.ParticleClusters(cfg).classifyBaryon(
            _baryon_evidence(spin="generic"))
        self.assertEqual(read.thresholds.spinVarianceTolerance, 2.0)
        self.assertTrue(read.sharpSpin)
        self.assertEqual(read.classification, "certified-proton")

    def test_reported_identities_travel(self):
        read = self.pc.classifyBaryon(_baryon_evidence())
        self.assertEqual(read.persistence, 4.0)
        self.assertEqual(read.lifetimeOverlap, 5.0)
        self.assertEqual(len(read.quarks), 3)
        self.assertEqual(read.boundComponent.canonicalHash(),
                         _modular_hierarchy()[2].components[0]
                         .id.canonicalHash())

    def test_describe_names_the_verdict_and_gaps(self):
        read = self.pc.classifyBaryon(_baryon_evidence(spin="generic"))
        text = read.describe()
        self.assertIn("baryon-candidate", text)
        self.assertIn("sharp-spin", text)


class TestQuasiFreeSharpSpinObstruction(unittest.TestCase):
    """The fourth verdict: every other certificate passes but Var(J^2) fails
    to converge to zero ACROSS the accepted covariance-only class."""

    def setUp(self):
        self.pc = obs.ParticleClusters()
        self._generic = _generic_slater_spin_reads()

    def _class(self, n=4):
        """The swept covariance-only class: n certified Var(J^2) reads that
        all sit at 15/16, never approaching zero."""
        return [self._generic[1]] * n

    def test_obstruction_verdict(self):
        read = self.pc.classifyBaryon(
            _baryon_evidence(spin="generic", class_variances=self._class()))
        self.assertEqual(read.classification,
                         "quasi-free-sharp-spin-obstruction")
        self.assertEqual(read.failedCertificates, ["sharp-spin"])
        self.assertTrue(read.quasiFreeClassSwept)
        self.assertAlmostEqual(read.classVarianceFloor, 15.0 / 16.0,
                               delta=1e-12)
        self.assertAlmostEqual(read.totalJ2, 0.75, delta=1e-13)

    def test_obstruction_is_reported_never_held(self):
        # a branch point mandating an explicit non-Gaussian mechanism, not a
        # held claim and not a refutation of the geometry.
        read = self.pc.classifyBaryon(
            _baryon_evidence(spin="generic", class_variances=self._class()))
        self.assertFalse(read.certificate.holds())
        self.assertEqual(read.certificate.grade,
                         cob.CertificateGrade.HeuristicDiscovery)

    def test_unswept_class_is_a_plain_candidate(self):
        read = self.pc.classifyBaryon(_baryon_evidence(spin="generic"))
        self.assertEqual(read.classification, "baryon-candidate")
        self.assertFalse(read.quasiFreeClassSwept)
        self.assertTrue(math.isnan(read.classVarianceFloor))

    def test_uncertified_class_member_is_not_a_sweep(self):
        variances = self._class() + [qm.WickCertificateRead()]
        read = self.pc.classifyBaryon(
            _baryon_evidence(spin="generic", class_variances=variances))
        self.assertFalse(read.quasiFreeClassSwept)
        self.assertEqual(read.classification, "baryon-candidate")

    def test_class_reaching_zero_is_not_an_obstruction(self):
        # one accepted member of the class IS an exact eigenstate: the
        # variance converges, so nothing is obstructed.
        sharp = _sharp_spin_reads()[1]
        read = self.pc.classifyBaryon(
            _baryon_evidence(spin="generic",
                             class_variances=self._class() + [sharp]))
        self.assertTrue(read.quasiFreeClassSwept)
        self.assertLess(read.classVarianceFloor, 1e-13)
        self.assertEqual(read.classification, "baryon-candidate")

    def test_unmeasured_own_variance_is_not_an_obstruction(self):
        # the class does not converge, but THIS candidate's own Var(J^2)
        # was never measured: unknown is not an obstruction.
        read = self.pc.classifyBaryon(
            _baryon_evidence(spin="expectation-only",
                             class_variances=self._class()))
        self.assertIsNone(read.totalJ2Variance)
        self.assertTrue(read.quasiFreeClassSwept)
        self.assertEqual(read.failedCertificates, ["sharp-spin"])
        self.assertEqual(read.classification, "baryon-candidate")

    def test_obstruction_requires_every_other_certificate(self):
        # the Delta-like case: the expectation ALSO fails, so this is a
        # plain baryon candidate, never the obstruction branch.
        read = self.pc.classifyBaryon(
            _baryon_evidence(spin="delta", class_variances=self._class()))
        self.assertEqual(read.classification, "baryon-candidate")

    def test_obstruction_requires_a_bound_baryon(self):
        read = self.pc.classifyBaryon(
            _baryon_evidence(spin="generic", class_variances=self._class(),
                             binding=obs.BoundSupercomponentRead()))
        self.assertEqual(read.classification, "no-baryon")

    def test_obstruction_survives_a_second_failing_certificate(self):
        read = self.pc.classifyBaryon(
            _baryon_evidence(spin="generic", class_variances=self._class(),
                             flux=_polarized_flux()))
        self.assertEqual(read.classification, "baryon-candidate")
        self.assertEqual(sorted(read.failedCertificates),
                         ["color-flux-zero", "sharp-spin"])


class TestBaryonInvarianceAndReplay(unittest.TestCase):
    """Relabeling, in-band rotation, constituent permutation, and cold
    replay preserve the verdict (the shared #763 merge gates)."""

    def setUp(self):
        self.pc = obs.ParticleClusters()

    def test_relabeling_preserves_the_verdict(self):
        base = self.pc.classifyBaryon(_baryon_evidence())
        legs = [_relabel_quark(q, level=1, tag="%02x" % (0xe0 + i))
                for i, q in enumerate(_baryon_evidence().quarks)]
        relabelled = self.pc.classifyBaryon(_baryon_evidence(quarks=legs))
        self.assertEqual(base.classification, relabelled.classification)
        self.assertEqual(base.confidence, relabelled.confidence)
        self.assertEqual(base.totalWinding, relabelled.totalWinding)
        self.assertEqual(base.flavorPattern, relabelled.flavorPattern)
        self.assertEqual(base.colorGramDeterminant,
                         relabelled.colorGramDeterminant)
        self.assertNotEqual(base.quarks[0].canonicalHash(),
                            relabelled.quarks[0].canonicalHash())

    def test_in_band_su3_rotation_preserves_the_verdict(self):
        base = self.pc.classifyBaryon(_baryon_evidence())
        for theta in (0.3, 1.1, 2.7):
            g = _su3_element(theta)
            rotated = self.pc.classifyBaryon(
                _baryon_evidence(color=g @ _color_triad()))
            self.assertEqual(rotated.classification, base.classification)
            self.assertLess(abs(rotated.colorWedge - base.colorWedge), 1e-13)

    def test_constituent_permutation_preserves_the_verdict(self):
        ev = _baryon_evidence()
        legs = list(ev.quarks)
        permuted = _baryon_evidence(quarks=[legs[2], legs[0], legs[1]],
                                    color=_color_triad()[:, [2, 0, 1]])
        a = self.pc.classifyBaryon(ev)
        b = self.pc.classifyBaryon(permuted)
        self.assertEqual(a.classification, b.classification)
        # the flavor PATTERN is canonical: a permutation cannot change it
        self.assertEqual(a.flavorPattern, b.flavorPattern)
        self.assertEqual(a.totalWinding, b.totalWinding)
        self.assertEqual(a.exteriorParity, b.exteriorParity)
        # an EVEN color-column permutation leaves even the wedge alone
        self.assertLess(abs(a.colorWedge - b.colorWedge), 1e-13)

    def test_cold_replay_is_deterministic(self):
        first = self.pc.classifyBaryon(_baryon_evidence()).toRecord()
        for _ in range(3):
            again = obs.ParticleClusters().classifyBaryon(
                _baryon_evidence()).toRecord()
            self.assertEqual(
                obs.ObservableGates.report_delta(first, again), 0.0)
            self.assertEqual(first["classification"],
                             again["classification"])
            self.assertEqual(first["failed_certificates"],
                             again["failed_certificates"])

    def test_record_roundtrip_is_exact(self):
        read = self.pc.classifyBaryon(_baryon_evidence())
        back = obs.BaryonRead.fromRecord(read.toRecord())
        self.assertEqual(back.classification, read.classification)
        self.assertEqual(back.colorGramDeterminant, read.colorGramDeterminant)
        self.assertEqual(back.colorWedge, read.colorWedge)
        self.assertEqual(back.totalWinding, read.totalWinding)
        self.assertEqual(back.baryonFlux, read.baryonFlux)
        self.assertEqual(back.electricFlux, read.electricFlux)
        self.assertEqual(back.totalJ2, read.totalJ2)
        self.assertEqual(back.totalJ2Variance, read.totalJ2Variance)
        self.assertEqual(back.rotationCharacter, read.rotationCharacter)
        self.assertEqual(back.flavorPattern, read.flavorPattern)
        self.assertEqual(back.confidence, read.confidence)
        self.assertEqual(back.failedCertificates, read.failedCertificates)
        self.assertEqual(back.thresholds.spinVarianceTolerance,
                         read.thresholds.spinVarianceTolerance)
        self.assertEqual(back.certificate.holds(), read.certificate.holds())
        self.assertEqual(obs.ObservableGates.report_delta(
            read.toRecord(), back.toRecord()), 0.0)

    def test_record_null_semantics(self):
        # unknown values serialize as null, never as zero.
        EH = obs.ExchangeHolonomy
        frame0 = EH.transverseSpinorFrame(0, 1, 4)
        weights = np.ones(4, dtype=complex)
        ev = _baryon_evidence(spin="none", quarks=[
            self.pc.classifyQuark(_certified_evidence())] * 3)
        ev.rotation = EH.rotationCharacter(
            EH.loopHolonomy(EH.rotationLoopFrames(frame0, 0, 1, 4, 1, 16),
                            weights),
            EH.loopHolonomy(EH.referenceLoopFrames(frame0, 8), weights))
        read = self.pc.classifyBaryon(ev)
        record = read.toRecord()
        for key in ("total_j2", "total_j2_variance", "electric_flux",
                    "physical_mass", "rotation_character_re",
                    "rotation_character_im", "total_isospin"):
            self.assertIsNone(record[key], key)
        back = obs.BaryonRead.fromRecord(record)
        self.assertIsNone(back.totalJ2)
        self.assertIsNone(back.totalJ2Variance)
        self.assertIsNone(back.physicalMass)
        self.assertIsNone(back.rotationCharacter)

    def test_from_record_rejects_unknown_schema(self):
        record = self.pc.classifyBaryon(_baryon_evidence()).toRecord()
        record["schema_version"] = 99
        with self.assertRaises(ValueError):
            obs.BaryonRead.fromRecord(record)

    def test_from_record_rejects_a_foreign_record_type(self):
        record = self.pc.classifyQuark(_certified_evidence()).toRecord()
        with self.assertRaises(ValueError):
            obs.BaryonRead.fromRecord(record)

    def test_verdict_surface_is_stable_for_wave_four(self):
        # #776/#777/#778 consume the verdict unchanged, serialized through
        # the existing Record convention and stable under replay.
        record = self.pc.classifyBaryon(_baryon_evidence()).toRecord()
        self.assertEqual(record["record_type"], "baryon_read")
        self.assertEqual(record["classification"], "certified-proton")
        for key in ("quark0_hash", "quark1_hash", "quark2_hash",
                    "bound_component_hash", "color_gram_determinant",
                    "color_flux", "baryon_flux", "electric_flux", "total_j2",
                    "total_j2_variance", "failed_certificates", "confidence",
                    "thresholds", "certificate"):
            self.assertIn(key, record)


class TestBaryonGuardsAndBenchmark(unittest.TestCase):
    """Shared #763 merge gates for the three-cluster sector."""

    def test_no_baryon_quantity_enters_the_emergence_objective(self):
        needles = ("BaryonRead", "BaryonCandidateEvidence", "classifyBaryon",
                   "BoundSupercomponentRead", "boundSupercomponentSearch",
                   "ScaleProfileRead", "scaleProfile")
        self.assertEqual(_objective_source_offenders(needles), [])

    def test_no_target_proton_wavefunction_is_introduced(self):
        # ticket out-of-scope: nothing here supplies or optimizes toward a
        # target proton state -- the classifier only READS evidence.
        source = (REPO_ROOT / "src" / "observables" /
                  "ParticleClusters.cpp").read_text()
        for needle in ("targetProton", "protonTarget", "targetWavefunction"):
            self.assertNotIn(needle, source)

    def test_new_thresholds_enter_the_evidence_fingerprint(self):
        ev = _certified_evidence()
        base = obs.ParticleClusters()
        for name, value in (("colorGramTolerance", 0.5),
                            ("colorFluxTolerance", 0.5),
                            ("spinExpectationTolerance", 0.5),
                            ("spinVarianceTolerance", 0.5),
                            ("minSupportContainment", 0.5),
                            ("minLifetimeOverlap", 3.0),
                            ("minRadius", 0.5),
                            ("maxProfileDeviation", 0.5)):
            cfg = obs.ParticleClustersConfig()
            setattr(cfg, name, value)
            self.assertNotEqual(base.evidenceFingerprint(ev),
                                obs.ParticleClusters(cfg)
                                .evidenceFingerprint(ev), name)

    def test_old_threshold_records_still_rehydrate(self):
        # pre-#775 checkpoints lack the new threshold keys: the reader falls
        # back to the defaults instead of rejecting.
        pc = obs.ParticleClusters()
        record = pc.classifyQuark(_certified_evidence()).toRecord()
        keys = ("color_gram_tolerance", "color_flux_tolerance",
                "spin_expectation_tolerance", "spin_variance_tolerance",
                "min_support_containment", "min_lifetime_overlap",
                "min_radius", "max_profile_deviation")
        for key in keys:
            self.assertIn(key, record["thresholds"])
            del record["thresholds"][key]
        back = obs.QuarkRead.fromRecord(record)
        defaults = obs.ParticleClustersConfig()
        self.assertEqual(back.thresholds.colorGramTolerance,
                         defaults.colorGramTolerance)
        self.assertEqual(back.thresholds.maxProfileDeviation,
                         defaults.maxProfileDeviation)

    def test_cached_color_flux_read_gives_an_identical_verdict(self):
        # the cached-versus-cold merge gate on the one CACHED read the
        # baryon certificate consumes (#764 AnalyticCache contract).
        pc = obs.ParticleClusters()
        cache = cob.AnalyticCache(_from_simplices(7, _TETRA_CHAIN,
                                                  timelike=False))
        state = qm.CovarianceState(np.eye(3, dtype=complex))
        cold = pc.octetBilinearRead(state, [0, 1, 2])
        warm = pc.octetBilinearReadCached(cache, [1, 2, 3], state, [0, 1, 2])
        cached = pc.octetBilinearReadCached(cache, [1, 2, 3], state, [0, 1, 2])
        self.assertGreaterEqual(cache.hits, 1)
        self.assertEqual(obs.ObservableGates.report_delta(
            cold.toRecord(), warm.toRecord()), 0.0)
        self.assertEqual(obs.ObservableGates.report_delta(
            cold.toRecord(), cached.toRecord()), 0.0)
        a = pc.classifyBaryon(_baryon_evidence(flux=cold))
        b = pc.classifyBaryon(_baryon_evidence(flux=cached))
        self.assertEqual(obs.ObservableGates.report_delta(
            a.toRecord(), b.toRecord()), 0.0)
        self.assertEqual(a.classification, "certified-proton")
        self.assertEqual(b.classification, "certified-proton")

    def test_classification_cost_per_candidate(self):
        # merge-gate benchmark: three-cluster classification cost (numbers
        # reported in the PR body).
        pc = obs.ParticleClusters()
        evidence = _baryon_evidence()
        samples = _scale_samples()
        candidates = _bound_candidates()
        components = _modular_hierarchy()[2].components
        n = 200
        t0 = time.perf_counter()
        for _ in range(n):
            pc.classifyBaryon(evidence)
        baryon = (time.perf_counter() - t0) / n
        t0 = time.perf_counter()
        for _ in range(n):
            pc.scaleProfile(samples)
        scale = (time.perf_counter() - t0) / n
        t0 = time.perf_counter()
        for _ in range(n):
            pc.boundSupercomponentSearch(components, candidates)
        search = (time.perf_counter() - t0) / n
        print(f"\n[benchmark] classifyBaryon: {baryon * 1e6:.1f} us; "
              f"scaleProfile: {scale * 1e6:.1f} us; "
              f"boundSupercomponentSearch: {search * 1e6:.1f} us "
              f"per candidate")
        for cost in (baryon, scale, search):
            self.assertLess(cost, 0.05)


class TestExchangeChannelReport(unittest.TestCase):
    """REPORT-ONLY reuse of the #772 Berry-cancelled exchange channel: the
    exchange character and the doubly cancelled spin-statistics ratio
    travel on the read but gate nothing (neither the ticket's
    proton-certificate list nor design spec 16.4 has an exchange row)."""

    def setUp(self):
        self.pc = obs.ParticleClusters()

    def test_exchange_character_is_minus_one_on_the_fixture(self):
        chi = _exchange_character()
        self.assertEqual(chi.channel, obs.HolonomyChannel.ParticleExchange)
        self.assertLess(abs(chi.character + 1.0), 1e-12)
        self.assertTrue(chi.certificate.holds())

    def test_doubly_cancelled_ratio_is_plus_one(self):
        read = self.pc.classifyBaryon(
            _baryon_evidence(exchange=_exchange_character()))
        self.assertLess(abs(read.exchangeCharacter + 1.0), 1e-12)
        self.assertLess(abs(read.rotationCharacter + 1.0), 1e-12)
        self.assertLess(abs(read.spinStatisticsRatio - 1.0), 1e-12)
        self.assertEqual(read.classification, "certified-proton")

    def test_exchange_channel_never_gates(self):
        # a DOUBLE exchange (chi_hat = +1) leaves the verdict untouched:
        # the channel is reported, never a certificate.
        doubled = _exchange_character(steps=16, distance=8)
        self.assertLess(abs(doubled.character - 1.0), 1e-12)
        read = self.pc.classifyBaryon(_baryon_evidence(exchange=doubled))
        self.assertEqual(read.classification, "certified-proton")
        self.assertEqual(read.failedCertificates, [])
        self.assertLess(abs(read.spinStatisticsRatio + 1.0), 1e-12)

    def test_absent_exchange_read_is_unknown(self):
        read = self.pc.classifyBaryon(_baryon_evidence())
        self.assertIsNone(read.exchangeCharacter)
        self.assertIsNone(read.spinStatisticsRatio)
        self.assertEqual(read.classification, "certified-proton")

    def test_mislabeled_channel_is_refused_not_reinterpreted(self):
        # a ROTATION-tagged read offered as the exchange channel is
        # ignored: the ratio stays unknown and nothing throws.
        read = self.pc.classifyBaryon(
            _baryon_evidence(exchange=_rotation_character()))
        self.assertIsNone(read.exchangeCharacter)
        self.assertIsNone(read.spinStatisticsRatio)

    def test_ratio_needs_both_certified_channels(self):
        read = self.pc.classifyBaryon(
            _baryon_evidence(rotation_turns=2,
                             exchange=_exchange_character()))
        # the 4pi rotation IS certified, so the ratio is still reported
        self.assertIsNotNone(read.spinStatisticsRatio)
        self.assertLess(abs(read.spinStatisticsRatio + 1.0), 1e-12)
        self.assertIn("rotation-character", read.failedCertificates)

    def test_exchange_channels_serialize(self):
        read = self.pc.classifyBaryon(
            _baryon_evidence(exchange=_exchange_character()))
        record = read.toRecord()
        self.assertAlmostEqual(record["exchange_character_re"], -1.0,
                               delta=1e-12)
        self.assertAlmostEqual(record["spin_statistics_ratio_re"], 1.0,
                               delta=1e-12)
        back = obs.BaryonRead.fromRecord(record)
        self.assertEqual(back.exchangeCharacter, read.exchangeCharacter)
        self.assertEqual(back.spinStatisticsRatio, read.spinStatisticsRatio)
        blank = self.pc.classifyBaryon(_baryon_evidence()).toRecord()
        self.assertIsNone(blank["exchange_character_re"])
        self.assertIsNone(blank["spin_statistics_ratio_im"])


class TestBindingCoherence(unittest.TestCase):
    """The whitepaper's "one persistent bound supercluster CONTAINING THEM":
    the binding read's contained-candidate set must be exactly the three
    constituents' label-free identities."""

    def setUp(self):
        self.pc = obs.ParticleClusters()

    def test_coherent_binding_certifies(self):
        read = self.pc.classifyBaryon(_baryon_evidence())
        self.assertEqual(read.classification, "certified-proton")
        self.assertEqual(
            sorted(q.canonicalHash()
                   for q in _baryon_evidence().binding.quarks),
            sorted(q.canonicalHash() for q in read.quarks))

    def test_binding_for_other_components_is_refused(self):
        # a CERTIFIED binding read of three DIFFERENT constituents is not
        # this candidate's supercomponent: the gate refuses rather than
        # accepting an incoherent bundle.
        strangers = [_relabel_quark(_ud_quark(k), level=1,
                                    tag="%02x" % (0xf0 + i))
                     for i, k in enumerate(("u", "u", "d"))]
        foreign = _binding(quarks=strangers)
        self.assertTrue(foreign.found)
        read = self.pc.classifyBaryon(_baryon_evidence(binding=foreign))
        self.assertEqual(read.classification, "no-baryon")
        self.assertEqual(read.failedCertificates, ["bound-supercomponent"])

    def test_binding_order_does_not_matter(self):
        # the comparison is an order-insensitive SET statement.
        ev = _baryon_evidence()
        legs = list(ev.quarks)
        permuted = _baryon_evidence(
            quarks=[legs[2], legs[0], legs[1]],
            color=_color_triad()[:, [2, 0, 1]], binding=ev.binding)
        read = self.pc.classifyBaryon(permuted)
        self.assertEqual(read.classification, "certified-proton")


class TestClassifyBoundSupercomponents(unittest.TestCase):
    """#802 — the composition the #776 analysis overlay runs.

    ``classifyBoundSupercomponents`` is what turns a §16.2 search result
    into §16.4 verdicts: every binding that grouped EXACTLY three certified
    constituents is classified, nothing is padded, and every quantity the
    caller did not supply stays ABSENT so ``classifyBaryon`` names it.
    """

    def setUp(self):
        self.pc = obs.ParticleClusters()
        self.candidates = _bound_candidates()
        self.constituents = [c.quark for c in self.candidates]
        _groups, _fine, self.coarse = _modular_hierarchy()
        self.bindings = list(self.pc.boundSupercomponentSearch(
            self.coarse.components, self.candidates))

    def assertRecordsEqual(self, mine, theirs, path="record"):
        """Records compared with the UNMEASURED NaNs matched as NaNs.

        ``nan != nan``, so a raw dict comparison of a record carrying
        honest unknowns could never hold — and silently passing one that
        did would mean the unknowns had been zero-filled."""
        if isinstance(mine, dict):
            self.assertIsInstance(theirs, dict, path)
            self.assertEqual(sorted(mine), sorted(theirs), path)
            for key in mine:
                self.assertRecordsEqual(mine[key], theirs[key],
                                        f"{path}.{key}")
        elif isinstance(mine, list):
            self.assertIsInstance(theirs, list, path)
            self.assertEqual(len(mine), len(theirs), path)
            for index, item in enumerate(mine):
                self.assertRecordsEqual(item, theirs[index],
                                        f"{path}[{index}]")
        elif isinstance(mine, float) and math.isnan(mine):
            self.assertTrue(isinstance(theirs, float) and math.isnan(theirs),
                            path)
        else:
            self.assertEqual(mine, theirs, path)

    # ---- the emission rule -------------------------------------------

    def test_three_certified_constituents_emit_one_baryon_read(self):
        self.assertEqual(len(self.bindings), 1)
        self.assertTrue(self.bindings[0].found)
        reads = self.pc.classifyBoundSupercomponents(
            self.bindings, self.constituents)
        self.assertEqual(len(reads), 1)
        read = reads[0]
        self.assertEqual(read.boundComponent.canonicalHash(),
                         self.bindings[0].boundComponent.canonicalHash())
        self.assertEqual(
            sorted(q.canonicalHash() for q in read.quarks),
            sorted(q.canonicalHash() for q in self.bindings[0].quarks))

    def test_both_structural_gates_hold_so_the_verdict_is_not_no_baryon(self):
        read = self.pc.classifyBoundSupercomponents(
            self.bindings, self.constituents)[0]
        self.assertNotIn("constituent-quarks", read.failedCertificates)
        self.assertNotIn("bound-supercomponent", read.failedCertificates)
        self.assertEqual(read.classification, "baryon-candidate")

    def test_two_certified_constituents_emit_nothing(self):
        """A three-cluster verdict is NEVER assembled by padding the
        missing legs: a padded leg would report a structural gap the
        geometry did not have."""
        candidates = _bound_candidates(kinds=("u", "d"))
        bindings = list(self.pc.boundSupercomponentSearch(
            self.coarse.components, candidates))
        self.assertEqual(len(bindings), 1)
        self.assertEqual(len(bindings[0].quarkIndices), 2)
        self.assertEqual(
            self.pc.classifyBoundSupercomponents(
                bindings, [c.quark for c in candidates]), [])

    def test_an_uncertified_candidate_is_never_a_constituent(self):
        """The search counts only CERTIFIED quark candidates, so three
        candidates of which one is uncertified never reach the classifier."""
        candidates = _bound_candidates()
        candidates[2].quark = obs.QuarkRead()     # classification "none"
        bindings = list(self.pc.boundSupercomponentSearch(
            self.coarse.components, candidates))
        self.assertEqual(len(bindings[0].quarkIndices), 2)
        self.assertEqual(
            self.pc.classifyBoundSupercomponents(
                bindings, [c.quark for c in candidates]), [])

    def test_no_binding_emits_no_verdict(self):
        self.assertEqual(
            self.pc.classifyBoundSupercomponents([], self.constituents), [])

    def test_one_read_per_qualifying_binding_in_binding_order(self):
        doubled = self.bindings + self.bindings
        reads = self.pc.classifyBoundSupercomponents(
            doubled, self.constituents)
        self.assertEqual(len(reads), 2)
        self.assertEqual(reads[0].boundComponent.canonicalHash(),
                         reads[1].boundComponent.canonicalHash())
        self.assertEqual(reads[0].classification, reads[1].classification)

    # ---- the delegation is exactly classifyBaryon ---------------------

    def test_it_is_classify_baryon_on_the_same_bundle(self):
        evidence = obs.BaryonCandidateEvidence()
        evidence.boundComponent = self.bindings[0].boundComponent
        evidence.binding = self.bindings[0]
        evidence.quarks = [self.constituents[i]
                           for i in self.bindings[0].quarkIndices]
        evidence.persistenceLifetime = 4.0
        direct = self.pc.classifyBaryon(evidence)
        composed = self.pc.classifyBoundSupercomponents(
            self.bindings, self.constituents, [4.0])[0]
        self.assertEqual(composed.classification, direct.classification)
        self.assertEqual(composed.failedCertificates,
                         direct.failedCertificates)
        self.assertEqual(composed.confidence, direct.confidence)
        self.assertEqual(composed.persistence, direct.persistence)
        self.assertRecordsEqual(composed.toRecord(), direct.toRecord())

    # ---- unsupplied evidence is NAMED, never presumed ----------------

    def test_every_unsupplied_certificate_is_named(self):
        """Exactly the COMPOSITE-level evidence is missing, and each gap is
        named. The constituent-derived rows (winding, parity, flavor,
        charge) are supplied by the #773 verdicts themselves and hold."""
        read = self.pc.classifyBoundSupercomponents(
            self.bindings, self.constituents)[0]
        self.assertEqual(
            read.failedCertificates,
            ["color-singlet", "color-flux-zero", "spin-expectation",
             "sharp-spin", "rotation-character", "finite-radius",
             "profile-stability"])
        # The fifteenth gate (crossing-readouts) is not applicable here -- no
        # crossing evidence travels through classifyBoundSupercomponents --
        # so it passes vacuously and does not appear among the failures.
        self.assertEqual(read.confidence, 8.0 / 15.0)

    def test_the_constituent_derived_rows_hold_on_certified_legs(self):
        read = self.pc.classifyBoundSupercomponents(
            self.bindings, self.constituents)[0]
        for name in ("baryon-flux-unit", "composite-parity-odd",
                     "flavor-uud", "electric-flux-unit"):
            self.assertNotIn(name, read.failedCertificates)
        self.assertEqual(read.totalWinding, 3)
        self.assertEqual(read.baryonFlux, 1.0)
        self.assertEqual(read.exteriorParity, -1)
        self.assertEqual(read.flavorPattern, "uud")

    def test_unknown_values_are_null_or_nan_never_zero(self):
        read = self.pc.classifyBoundSupercomponents(
            self.bindings, self.constituents)[0]
        self.assertTrue(math.isnan(read.colorGramDeterminant))
        self.assertTrue(math.isnan(read.colorWedge.real))
        self.assertTrue(math.isnan(read.colorFlux))
        self.assertIsNone(read.totalJ2)
        self.assertIsNone(read.totalJ2Variance)
        self.assertIsNone(read.rotationCharacter)
        self.assertIsNone(read.exchangeCharacter)
        self.assertIsNone(read.physicalMass)
        self.assertTrue(math.isnan(read.classVarianceFloor))
        self.assertFalse(read.quasiFreeClassSwept)
        self.assertTrue(math.isnan(read.radius))
        self.assertTrue(math.isnan(read.transportLeakageMax))

    def test_a_continuum_spin_claim_is_never_declared_here(self):
        """The composition makes no continuum claim, so the SO(d)->Spin(d)
        lift is not demanded and `spin-lift` is not a failure."""
        read = self.pc.classifyBoundSupercomponents(
            self.bindings, self.constituents)[0]
        self.assertFalse(read.spinLiftApplicable)
        self.assertNotIn("spin-lift", read.failedCertificates)

    # ---- the bound component's lifetime -------------------------------

    def test_an_unsupplied_lifetime_is_nan(self):
        read = self.pc.classifyBoundSupercomponents(
            self.bindings, self.constituents)[0]
        self.assertTrue(math.isnan(read.persistence))

    def test_a_supplied_lifetime_travels_verbatim(self):
        read = self.pc.classifyBoundSupercomponents(
            self.bindings, self.constituents, [7.0])[0]
        self.assertEqual(read.persistence, 7.0)

    def test_a_mismatched_lifetime_list_is_refused(self):
        with self.assertRaises(ValueError):
            self.pc.classifyBoundSupercomponents(
                self.bindings, self.constituents, [1.0, 2.0])

    def test_a_constituent_index_outside_the_read_list_is_refused(self):
        with self.assertRaises(ValueError):
            self.pc.classifyBoundSupercomponents(
                self.bindings, self.constituents[:2])

    # ---- ordering / relabeling ----------------------------------------

    def test_the_verdict_is_insensitive_to_constituent_order(self):
        order = [2, 0, 1]
        candidates = [self.candidates[i] for i in order]
        bindings = list(self.pc.boundSupercomponentSearch(
            self.coarse.components, candidates))
        permuted = self.pc.classifyBoundSupercomponents(
            bindings, [c.quark for c in candidates])[0]
        base = self.pc.classifyBoundSupercomponents(
            self.bindings, self.constituents)[0]
        self.assertEqual(permuted.classification, base.classification)
        self.assertEqual(permuted.failedCertificates, base.failedCertificates)
        self.assertEqual(
            sorted(q.canonicalHash() for q in permuted.quarks),
            sorted(q.canonicalHash() for q in base.quarks))

    # ---- the record schema round-trips with its nulls ------------------

    def test_the_record_round_trips_with_nulls_for_unknowns(self):
        read = self.pc.classifyBoundSupercomponents(
            self.bindings, self.constituents, [4.0])[0]
        record = read.toRecord()
        # Every UNSUPPLIED optional serializes as null, never as zero.
        for key in ("total_j2", "total_j2_variance", "physical_mass",
                    "rotation_character_re", "rotation_character_im",
                    "exchange_character_re", "exchange_character_im",
                    "spin_statistics_ratio_re", "spin_statistics_ratio_im"):
            self.assertIsNone(record[key], f"{key} is not null")
        # Every UNMEASURED double serializes as NaN, never as zero.
        for key in ("color_gram_determinant", "color_flux", "color_wedge_re",
                    "color_wedge_im", "class_variance_floor", "radius",
                    "radius_ratio", "spectral_mass", "profile_max_deviation",
                    "transport_leakage_max"):
            self.assertTrue(math.isnan(record[key]), f"{key} is not NaN")
        rehydrated = obs.BaryonRead.fromRecord(record)
        self.assertEqual(rehydrated.classification, read.classification)
        self.assertEqual(rehydrated.failedCertificates,
                         read.failedCertificates)
        self.assertIsNone(rehydrated.totalJ2)
        self.assertIsNone(rehydrated.totalJ2Variance)
        self.assertIsNone(rehydrated.physicalMass)
        self.assertTrue(math.isnan(rehydrated.colorGramDeterminant))
        self.assertEqual(rehydrated.persistence, 4.0)
        self.assertRecordsEqual(rehydrated.toRecord(), record)

    def test_an_unknown_record_schema_version_is_rejected(self):
        record = self.pc.classifyBoundSupercomponents(
            self.bindings, self.constituents)[0].toRecord()
        record["schema_version"] = record["schema_version"] + 97
        with self.assertRaises(ValueError):
            obs.BaryonRead.fromRecord(record)

    # ---- the three-outcome vocabulary is reachable through here -------

    def test_a_fully_evidenced_bundle_certifies_a_proton(self):
        """The same three constituents, given the rest of the evidence,
        reach `certified-proton` — so the composition's `baryon-candidate`
        is a statement about the MISSING evidence, not a ceiling."""
        read = self.pc.classifyBaryon(_baryon_evidence())
        self.assertEqual(read.classification, "certified-proton")
        self.assertEqual(read.failedCertificates, [])
        self.assertEqual(read.confidence, 1.0)

    def test_the_obstruction_verdict_is_reachable(self):
        read = self.pc.classifyBaryon(_baryon_evidence(
            spin="generic",
            class_variances=[_generic_slater_spin_reads()[1]] * 3))
        self.assertEqual(read.classification,
                         "quasi-free-sharp-spin-obstruction")
        self.assertEqual(read.failedCertificates, ["sharp-spin"])
        self.assertTrue(read.quasiFreeClassSwept)
# =========================================================================== #
# #808 negative controls: the conditions the whitepaper calls STABLE are
# compared ACROSS FRAMES, and the coherence is an OVERLAP datum
# =========================================================================== #
class TestStabilityIsAcrossFrames(unittest.TestCase):
    """Whitepaper quark conditions two and three: "its selected color fiber
    has STABLE rank three" and "its calibrated triangle-anchor profile and
    determinant-line coherence are STABLE".  A single frame cannot establish
    either."""

    def setUp(self):
        self.pc = obs.ParticleClusters()

    def _read(self, ev):
        return self.pc.classifyQuark(ev)

    def test_a_stable_candidate_certifies(self):
        read = self._read(_certified_evidence())
        self.assertEqual(read.classification, "quark")
        self.assertEqual(read.stabilityFrames, 3)
        self.assertEqual(read.anchorScoreSpread, 0.0)
        self.assertEqual(read.anchorCoherenceSpread, 0.0)
        self.assertAlmostEqual(read.bandContinuationOverlap, 1.0,
                               delta=MACHINE)

    def test_a_band_that_loses_rank_three_at_the_next_frame_is_rejected(self):
        # Stable at frame 0, rank two at frame 1: the condition holds at ONE
        # frame and fails across them, which is exactly what "stable" is
        # supposed to catch.
        ev = _certified_evidence()
        first = ev.colorBandFrames[0]
        ev.colorBandFrames = [first, _unit_fiber(1, 2)]
        read = self._read(ev)
        self.assertEqual(read.classification, "none")
        self.assertIn("color-rank-stability", read.failedCertificates)
        self.assertNotIn("color-rank-three", read.failedCertificates)

    def test_a_band_uncertified_at_the_next_frame_is_rejected(self):
        ev = _certified_evidence()
        first = ev.colorBandFrames[0]
        ev.colorBandFrames = [first, _unit_fiber(1, 3, accepted=False)]
        read = self._read(ev)
        self.assertEqual(read.classification, "none")
        self.assertIn("color-rank-stability", read.failedCertificates)

    def test_frames_must_be_certified_continuations(self):
        # Two accepted rank-three bands on DISJOINT cells are two different
        # bands, not one stable band: the continuation is never certified.
        ev = _certified_evidence()
        first = ev.colorBandFrames[0]
        ev.colorBandFrames = [first, _unit_fiber(500, 3)]
        read = self._read(ev)
        self.assertEqual(read.classification, "none")
        self.assertIn("color-rank-stability", read.failedCertificates)
        self.assertEqual(read.bandContinuationOverlap, 0.0)

    def test_one_frame_never_establishes_stability(self):
        ev = _certified_evidence()
        ev.colorBandFrames = [ev.colorBandFrames[0]]
        ev.anchorFrames = [ev.anchorFrames[0]]
        read = self._read(ev)
        self.assertEqual(read.classification, "none")
        self.assertIn("color-rank-stability", read.failedCertificates)
        self.assertIn("anchor-stability", read.failedCertificates)
        # the single-frame conditions themselves still pass -- it is the
        # ACROSS-FRAME comparison that is missing, and it is named
        self.assertNotIn("color-rank-three", read.failedCertificates)
        self.assertNotIn("anchor", read.failedCertificates)
        self.assertEqual(read.stabilityFrames, 1)
        self.assertTrue(math.isnan(read.anchorScoreSpread))

    def test_missing_stability_windows_fail_by_name(self):
        ev = _certified_evidence()
        ev.colorBandFrames = []
        ev.anchorFrames = []
        read = self._read(ev)
        self.assertEqual(read.stabilityFrames, 0)
        self.assertIn("color-rank-stability", read.failedCertificates)
        self.assertIn("anchor-stability", read.failedCertificates)

    def test_an_anchor_that_decays_at_the_next_frame_is_rejected(self):
        # Same atlas, a frame where the anchor score falls below the floor:
        # the profile is not stable, and the spread is reported.
        ev = _certified_evidence()
        good = ev.anchorFrames[0]
        weak = _anchor_profile(terms=(0.4, 0.02))
        self.assertLess(weak.score, obs.ParticleClustersConfig().minAnchorScore)
        ev.anchorFrames = [good, weak]
        read = self._read(ev)
        self.assertEqual(read.classification, "none")
        self.assertIn("anchor-stability", read.failedCertificates)
        self.assertNotIn("anchor", read.failedCertificates)
        self.assertGreater(read.anchorScoreSpread, 0.3)

    def test_stability_frames_are_configurable(self):
        cfg = obs.ParticleClustersConfig()
        cfg.minStabilityFrames = 4          # more frames than supplied
        read = obs.ParticleClusters(cfg).classifyQuark(_certified_evidence())
        self.assertIn("color-rank-stability", read.failedCertificates)
        self.assertIn("anchor-stability", read.failedCertificates)


class TestOverlapRestrictedCoherenceGatesTheAnchor(unittest.TestCase):
    """The classifier gates on the determinant-phase coherence, which is an
    OVERLAP datum (#808): a disjoint atlas has none."""

    def setUp(self):
        self.pc = obs.ParticleClusters()

    def test_a_disjoint_atlas_fails_the_anchor_certificate(self):
        disjoint = _disjoint_anchor_profile()
        self.assertEqual(disjoint.overlapping_triangles, 0)
        self.assertTrue(math.isnan(disjoint.phase_coherence))
        self.assertAlmostEqual(disjoint.score, 0.98, delta=1e-9)
        ev = _certified_evidence(anchor=disjoint)
        ev.anchorFrames = [disjoint, disjoint, disjoint]
        read = self.pc.classifyQuark(ev)
        # the SCORE clears its floor and the certificate still fails: the
        # missing quantity is the overlap coherence, and it is named
        self.assertGreater(read.triangleAnchorScore,
                           read.thresholds.minAnchorScore)
        self.assertTrue(math.isnan(read.anchorPhaseCoherence))
        self.assertEqual(read.classification, "none")
        self.assertIn("anchor", read.failedCertificates)
        self.assertIn("anchor-stability", read.failedCertificates)

    def test_an_overlapping_atlas_carries_the_coherence(self):
        overlapping = _anchor_profile()
        self.assertEqual(overlapping.overlapping_triangles, 2)
        self.assertEqual(overlapping.overlap_relation, "shared-edge")
        read = self.pc.classifyQuark(_certified_evidence(anchor=overlapping))
        self.assertGreater(read.anchorPhaseCoherence,
                           read.thresholds.minPhaseCoherence)
        self.assertEqual(read.classification, "quark")


