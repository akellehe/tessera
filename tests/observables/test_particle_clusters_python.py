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
  failed_certificates (negative control per certificate);
* relabeling and refinement preserve accepted classifications;
* cached classification equals cold recomputation under the #764 cache;
* no quark-specific quantity enters the emergence objective.

Fixtures are built by composing the MERGED public APIs (#765 ComponentId,
#769 SpectralFiber/ComponentBandRead record synthesis, #767 ColorAnchor,
#770 FiberConnection transports/windings, #780 CovarianceState Wick
reads, and the existing EigenstateSynthesis.gauss_law_charge) — the
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
    return obs.SpectralFiber.from_record(record)


def _unit_fiber(base_id, r, **kw):
    """Rank-r fiber with identity frame on r synthetic cells."""
    return _fiber([[base_id + i] for i in range(r)], np.eye(r), **kw)


def _band_read(fibers, degree=1, support=None):
    """Synthesize a ComponentBandRead carrying `fibers` (the #769 replay
    route for a whole enumeration frame)."""
    cells = []
    for f in fibers:
        cells.extend(f.cell_vertices())
    record = {
        "schema_version": 1, "record_type": "spectral_band_read",
        "support": [int(v) for v in (support or [])],
        "degree": int(degree),
        "dimension": len(cells),
        "cell_vertices": [[int(v) for v in cell] for cell in cells],
        "regime": "positive-semidefinite",
        "solver_path": "dense-self-adjoint",
        "truncated": False,
        "fibers": [f.to_record() for f in fibers],
        "solve_certificate": _cert_record("positive-semidefinite"),
    }
    _split("covered_eigenvalues", [], record)
    return obs.ComponentBandRead.from_record(record)


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
    state = qm.CovarianceState.from_occupations(np.asarray(occupations,
                                                          dtype=float))
    return state.wick_parity(), state.wick_total_number()


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
    verts = [st.create_vertex(i) for i in ids]
    for simplex in simplices:
        st.create_simplex([verts[i] for i in simplex])
    for e in st.get_edge_list().to_vector():
        e.set_length(1j if timelike else (1.0 + 0j))
        e.set_phase(0.0)
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
    surfaces = obs.ParticleClusters.nested_enclosures(st, [base], 2)
    n = es.order()

    def probe(vset):
        rows = []
        for c in range(n):
            f = [0j] * n
            f[c] = 1.0 + 0j
            rows.append(es.gauss_law_charge(f, vset, True).real)
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
    winding = conn.closed_family_winding(family)

    parity, occupation = _parity_occupation(list(occupations))

    ev = obs.QuarkCandidateEvidence()
    ev.component = obs.ComponentId("ab" * 16, 1)
    ev.color_band = A
    ev.anchor = anchor if anchor is not None else _anchor_profile()
    ev.lifetime_transports = family
    ev.winding = winding
    ev.parity_read = parity
    ev.occupation_read = occupation
    # The modularity RESOLUTION-slice numbers are reported, never gated; the
    # COBORDISM-FRAME lifetime is the gated persistence quantity (#808).
    ev.persistence_lifetime = 3.0
    ev.persistence_min_overlap = 1.0
    ev.frame_lifetime = 3.0
    ev.frame_min_overlap = 1.0
    ev.refinement_overlap = 1.0
    # The two STABILITY windows: the same rank-three band and the same
    # anchor profile at each of three cobordism frames (an unchanging
    # candidate is the stable case; the moving ones are their own tests).
    ev.color_band_frames = [A, A, A]
    ev.anchor_frames = [ev.anchor, ev.anchor, ev.anchor]
    if with_flavor:
        pc = obs.ParticleClusters()
        ev.flavor = pc.flavor_doublet_search(_doublet_frames())
        assert ev.flavor.found
        ev.doublet_occupancy = (np.array([1.0, 0.0], dtype=complex)
                               if occupancy is None
                               else np.asarray(occupancy, dtype=complex))
        ev.doublet_orientation = orientation
    if with_charge is not None:
        pc = obs.ParticleClusters()
        st, F, surfaces = _gauss_fixture(with_charge)
        ev.charge = pc.gauss_flux_on_surfaces(st, F, surfaces, True)
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
        read = self.pc.classify_quark(_certified_evidence(turns=1))
        self.assertEqual(read.classification, "quark")
        self.assertEqual(read.determinant_winding, 1)
        self.assertAlmostEqual(read.baryon_flux, 1.0 / 3.0, delta=MACHINE)
        self.assertEqual(read.exterior_parity, -1)
        self.assertEqual(read.color_rank, 3)
        self.assertEqual(read.confidence, 1.0)
        self.assertEqual(read.winding_closure, "closed-family")
        for name in CORE:
            self.assertNotIn(name, read.failed_certificates)
        self.assertTrue(read.certificate.holds())

    def test_the_dressed_anchor_is_reported_and_never_gating(self):
        """A supplied dressed-anchor read that refuses names the
        'dressed-anchor' certificate without changing the classification; an
        anchored one and an absent one name nothing."""
        K = tessera.cobordism.ChainComplex.from_top_cells([[0, 1, 2, 3]])
        U = tessera.chainhodge.Connection(K, [1.0 + 0j] * 6)
        paths = tessera.chainhodge.DeclaredPaths.breadth_first(K, 0)
        boundary = np.asarray(K.boundary_matrix(2), dtype=float).reshape(6, 4)
        coexact = tessera.chainhodge.DressedAnchor.profile(
            K, U, paths, list(range(4)), boundary[:, :3].astype(complex))
        exact = tessera.chainhodge.DressedAnchor.profile(
            K, U, paths, list(range(4)),
            np.asarray(K.boundary_matrix(1), dtype=float).reshape(4, 6)
            .T[:, :3].astype(complex))
        self.assertTrue(coexact.anchored)
        self.assertFalse(exact.anchored)
        for read, named in ((None, False), (coexact, False), (exact, True)):
            evidence = _certified_evidence(turns=1)
            if read is not None:
                evidence.dressed_anchor = read
                self.assertIs(evidence.dressed_anchor.anchored, read.anchored)
            verdict = self.pc.classify_quark(evidence)
            self.assertEqual(verdict.classification, "quark")
            self.assertEqual("dressed-anchor" in verdict.failed_certificates,
                             named)

    def test_certified_antiquark_is_the_orientation_reverse(self):
        read = self.pc.classify_quark(_certified_evidence(turns=-1))
        self.assertEqual(read.classification, "antiquark")
        self.assertEqual(read.determinant_winding, -1)
        self.assertAlmostEqual(read.baryon_flux, -1.0 / 3.0, delta=MACHINE)
        self.assertEqual(read.confidence, 1.0)

    def test_quark_and_antiquark_differ_only_by_orientation(self):
        q = self.pc.classify_quark(_certified_evidence(turns=1))
        aq = self.pc.classify_quark(_certified_evidence(turns=-1))
        # identical anchored/parity/persistence evidence; opposite line
        self.assertEqual(q.triangle_anchor_score, aq.triangle_anchor_score)
        self.assertEqual(q.exterior_parity, aq.exterior_parity)
        self.assertEqual(q.occupation_total, aq.occupation_total)
        self.assertEqual(q.determinant_winding, -aq.determinant_winding)
        self.assertEqual(q.baryon_flux, -aq.baryon_flux)

    def test_reversed_family_is_the_antiquark(self):
        # reversing the tube = traversing the same transport family in the
        # opposite parameter order (the #770 orientation convention)
        conn = obs.FiberConnection()
        A, B = _unit_fiber(1, 3), _unit_fiber(11, 3)
        family = _winding_family(conn, A, B, turns=1)
        ev = _certified_evidence(turns=1)
        ev.winding = conn.closed_family_winding(list(reversed(family)))
        read = self.pc.classify_quark(ev)
        self.assertEqual(read.classification, "antiquark")

    def test_unknown_winding_leaves_baryon_flux_unknown(self):
        ev = _certified_evidence()
        # open segment with NO declared closure: raw endpoint phase is
        # never promoted to baryon-flux evidence
        conn = obs.FiberConnection()
        A, B = _unit_fiber(1, 3), _unit_fiber(11, 3)
        segment = [_phase_link(conn, A, B, p) for p in (0.0, 0.4, 0.8)]
        ev.winding = conn.open_segment_winding(segment,
                                             obs.WindingClosureSpec())
        read = self.pc.classify_quark(ev)
        self.assertIsNone(read.determinant_winding)
        self.assertIsNone(read.baryon_flux)
        self.assertEqual(read.winding_closure, "none")
        self.assertIn("winding", read.failed_certificates)
        self.assertEqual(read.classification, "none")
        self.assertFalse(read.certificate.holds())

    def test_open_segment_with_declared_closure_carries_its_specification(self):
        conn = obs.FiberConnection()
        A, B = _unit_fiber(1, 3), _unit_fiber(11, 3)
        segment = [_phase_link(conn, A, B, TWO_PI * k / 4) for k in range(5)]
        spec = obs.WindingClosureSpec()
        spec.mode = obs.WindingClosureSpec.Mode.MATCHED_REFERENCE
        spec.reference_id = "co-moving-reference"
        spec.reference_transports = [np.eye(3)] * len(segment)
        ev = _certified_evidence()
        ev.winding = conn.open_segment_winding(segment, spec)
        read = self.pc.classify_quark(ev)
        self.assertEqual(read.classification, "quark")
        self.assertEqual(read.winding_closure, "matched-reference")
        self.assertEqual(read.winding_reference_id, "co-moving-reference")
        self.assertAlmostEqual(read.baryon_flux, 1.0 / 3.0, delta=MACHINE)

    def test_boundary_register_trivialization_closure(self):
        # the other declared closure of the ticket: an open segment closed
        # through boundary-register trivializations
        conn = obs.FiberConnection()
        A, B = _unit_fiber(1, 3), _unit_fiber(11, 3)
        segment = [_phase_link(conn, A, B, TWO_PI * k / 4) for k in range(5)]
        spec = obs.WindingClosureSpec()
        spec.mode = obs.WindingClosureSpec.Mode.ENDPOINT_TRIVIALIZATION
        spec.reference_id = "boundary-registers"
        spec.start_trivialization = np.eye(3)
        spec.end_trivialization = np.eye(3)
        ev = _certified_evidence()
        ev.winding = conn.open_segment_winding(segment, spec)
        read = self.pc.classify_quark(ev)
        self.assertEqual(read.classification, "quark")
        self.assertEqual(read.winding_closure, "endpoint-trivialization")
        self.assertEqual(read.winding_reference_id, "boundary-registers")
        self.assertAlmostEqual(read.baryon_flux, 1.0 / 3.0, delta=MACHINE)

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
            dual = conn.transport_reverse(B, A, v)
            self.assertTrue(fwd.accepted and dual.accepted)
            self.assertLess(abs(dual.determinant_phase
                                - np.conj(fwd.determinant_phase)), 1e-12)

    def test_certified_zero_winding_is_a_certified_zero_flux_not_a_quark(self):
        read = self.pc.classify_quark(_certified_evidence(turns=0))
        self.assertEqual(read.determinant_winding, 0)
        self.assertEqual(read.baryon_flux, 0.0)  # certified zero, not unknown
        self.assertIn("winding-unit", read.failed_certificates)
        self.assertNotIn("winding", read.failed_certificates)
        self.assertEqual(read.classification, "none")

    def test_anchor_profile_travels_on_the_read(self):
        profile = _anchor_profile()
        read = self.pc.classify_quark(_certified_evidence(anchor=profile))
        self.assertAlmostEqual(read.triangle_anchor_score, profile.score,
                               delta=MACHINE)
        self.assertAlmostEqual(read.triangle_anchor_max_term, profile.max_term,
                               delta=MACHINE)
        self.assertAlmostEqual(read.triangle_anchor_participation,
                               profile.participation_ratio, delta=MACHINE)
        self.assertAlmostEqual(read.anchor_phase_dispersion,
                               profile.phase_dispersion, delta=MACHINE)
        self.assertEqual(read.anchor_weighting_id, "declared")
        # The coherence is an OVERLAP datum (#808): both declared triangles
        # share edge rows 0 and 1, so both take part in the resultant.
        self.assertEqual(profile.overlapping_triangles, 2)
        self.assertEqual(profile.overlap_relation, "shared-edge")
        self.assertGreater(read.anchor_phase_coherence, 0.98)

    def test_confidence_is_the_passed_core_fraction(self):
        ev = _certified_evidence()
        ev.refinement_overlap = NAN  # remove exactly one core certificate
        read = self.pc.classify_quark(ev)
        self.assertAlmostEqual(read.confidence, 11.0 / 12.0, delta=MACHINE)
        self.assertEqual(read.classification, "none")

    def test_thresholds_are_recorded_on_every_read(self):
        cfg = obs.ParticleClustersConfig()
        cfg.min_anchor_score = 0.75
        pc = obs.ParticleClusters(cfg)
        read = pc.classify_quark(_certified_evidence())
        self.assertEqual(read.thresholds.min_anchor_score, 0.75)
        rec = read.to_record()
        self.assertEqual(rec["thresholds"]["min_anchor_score"], 0.75)

    def test_classify_quarks_stream_preserves_order(self):
        reads = self.pc.classify_quarks(
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
        read = self.pc.classify_quark(ev)
        self.assertEqual(read.classification, "none")
        self.assertIn(name, read.failed_certificates)
        self.assertLess(read.confidence, 1.0)
        self.assertFalse(read.certificate.holds())
        return read

    def test_missing_anchor(self):
        ev = _certified_evidence()
        ev.anchor = obs.AnchorProfile()
        read = self._assert_named_failure(ev, "anchor")
        self.assertTrue(math.isnan(read.triangle_anchor_score))

    def test_low_anchor_score(self):
        cfg = obs.ParticleClustersConfig()
        cfg.min_anchor_score = 1.5  # unreachable: a^2 <= 1
        pc = obs.ParticleClusters(cfg)
        read = pc.classify_quark(_certified_evidence())
        self.assertIn("anchor", read.failed_certificates)
        self.assertEqual(read.classification, "none")

    def test_even_parity(self):
        ev = _certified_evidence(occupations=(1.0, 1.0, 0.0))
        read = self._assert_named_failure(ev, "parity-odd")
        self.assertEqual(read.exterior_parity, +1)

    def test_uncertified_parity_never_emits_a_sign(self):
        ev = _certified_evidence()
        ev.parity_read = qm.WickCertificateRead()  # default: never holds
        read = self._assert_named_failure(ev, "parity-odd")
        self.assertEqual(read.exterior_parity, 0)

    def test_rank_two_band(self):
        ev = _certified_evidence()
        ev.color_band = _unit_fiber(1, 2)
        read = self._assert_named_failure(ev, "color-rank-three")
        self.assertEqual(read.color_rank, 2)

    def test_unaccepted_band_gap_closed(self):
        ev = _certified_evidence()
        ev.color_band = _unit_fiber(1, 3, accepted=False)
        self._assert_named_failure(ev, "color-rank-three")

    def test_leaking_transport(self):
        ev = _certified_evidence()
        conn = obs.FiberConnection()
        leaky = conn.transport(_unit_fiber(1, 3), _unit_fiber(11, 3),
                               np.diag([0.1, 1.0, 1.0]))
        self.assertFalse(leaky.accepted)
        ev.lifetime_transports = list(ev.lifetime_transports) + [leaky]
        self._assert_named_failure(ev, "transport-leakage")

    def test_missing_transports(self):
        ev = _certified_evidence()
        ev.lifetime_transports = []
        read = self._assert_named_failure(ev, "transport-leakage")
        self.assertEqual(read.transport_count, 0)
        self.assertTrue(math.isnan(read.transport_leakage_max))

    def test_insufficient_frame_lifetime(self):
        # A candidate seen in ONE cobordism frame has no lifetime across
        # frames: the whitepaper conjunct fails, and it fails for a physical
        # reason about the candidate.
        ev = _certified_evidence()
        ev.frame_lifetime = 1.0
        read = self._assert_named_failure(ev, "persistence")
        self.assertEqual(read.frame_lifetime, 1.0)

    def test_missing_frame_lifetime(self):
        ev = _certified_evidence()
        ev.frame_lifetime = NAN
        self._assert_named_failure(ev, "persistence")

    def test_low_frame_overlap(self):
        ev = _certified_evidence()
        ev.frame_min_overlap = 0.2
        self._assert_named_failure(ev, "persistence")

    def test_single_resolution_read_is_not_a_structural_persistence_failure(
            self):
        # #808 negative control: at ONE modularity resolution the slice
        # lifetime is identically 1.  That used to make `persistence`
        # structurally unpassable "for a reason that is not physics"; the
        # gate now reads the COBORDISM-FRAME lifetime, so the same candidate
        # certifies while its resolution-slice numbers stay reported.
        ev = _certified_evidence()
        ev.persistence_lifetime = 1.0
        ev.persistence_min_overlap = 1.0
        read = self.pc.classify_quark(ev)
        self.assertEqual(read.classification, "quark")
        self.assertNotIn("persistence", read.failed_certificates)
        self.assertEqual(read.persistence_lifetime, 1.0)
        self.assertEqual(read.frame_lifetime, 3.0)

    def test_modularity_resolution_lifetime_never_vetoes(self):
        # The same statement in the other direction: no resolution-slice
        # value, however bad, can veto an otherwise certified fiber.
        for lifetime, overlap in ((NAN, NAN), (0.0, 0.0), (1.0, 0.1)):
            with self.subTest(lifetime=lifetime):
                ev = _certified_evidence()
                ev.persistence_lifetime = lifetime
                ev.persistence_min_overlap = overlap
                read = self.pc.classify_quark(ev)
                self.assertEqual(read.classification, "quark")

    def test_low_localization(self):
        cfg = obs.ParticleClustersConfig()
        cfg.min_localization = 0.9  # fixture band carries 0.5
        pc = obs.ParticleClusters(cfg)
        read = pc.classify_quark(_certified_evidence())
        self.assertIn("localization", read.failed_certificates)
        self.assertEqual(read.classification, "none")

    def test_missing_refinement_stability(self):
        ev = _certified_evidence()
        ev.refinement_overlap = NAN
        self._assert_named_failure(ev, "refinement-stability")

    def test_unstable_refinement(self):
        ev = _certified_evidence()
        ev.refinement_overlap = 0.3
        self._assert_named_failure(ev, "refinement-stability")

    def test_gap_closed_winding_family_invalidates(self):
        ev = _certified_evidence()
        conn = obs.FiberConnection()
        A, B = _unit_fiber(1, 3), _unit_fiber(11, 3)
        family = _winding_family(conn, A, B)
        bad = conn.transport(A, B, np.diag([0.1, 1.0, 1.0]))
        ev.winding = conn.closed_family_winding(family + [bad])
        read = self._assert_named_failure(ev, "winding")
        self.assertIsNone(read.determinant_winding)
        self.assertIsNone(read.baryon_flux)


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
        read = pc.classify_quark(ev)
        self.assertEqual(read.classification, "none")
        self.assertEqual(read.exterior_parity, +1)
        self.assertAlmostEqual(read.occupation_total, 2.0, delta=MACHINE)
        self.assertIn("parity-odd", read.failed_certificates)
        self.assertIn("occupation-one", read.failed_certificates)
        # the color-alone channels would NOT have refused:
        self.assertNotIn("color-rank-three", read.failed_certificates)
        self.assertNotIn("winding", read.failed_certificates)

    def test_top_wedge_triple_occupation_is_not_a_quark(self):
        # N = 3 (odd parity!) still fails: single-fermion occupation is a
        # separate certificate from parity.
        pc = obs.ParticleClusters()
        ev = _certified_evidence(occupations=(1.0, 1.0, 1.0))
        read = pc.classify_quark(ev)
        self.assertEqual(read.exterior_parity, -1)
        self.assertIn("occupation-one", read.failed_certificates)
        self.assertNotIn("parity-odd", read.failed_certificates)
        self.assertEqual(read.classification, "none")


# =========================================================================== #
# conjugate-pair conservation
# =========================================================================== #
class TestConjugatePair(unittest.TestCase):
    def setUp(self):
        self.pc = obs.ParticleClusters()

    def test_certified_conjugate_pair_conserves(self):
        quark = self.pc.classify_quark(_certified_evidence(turns=1))
        anti = self.pc.classify_quark(_certified_evidence(turns=-1))
        pair = self.pc.conjugate_pair(quark, anti)
        self.assertEqual(pair.total_winding, 0)
        self.assertEqual(pair.total_baryon_flux, 0.0)
        self.assertEqual(pair.total_parity, +1)
        self.assertTrue(pair.parity_even)
        self.assertTrue(pair.conserved)
        self.assertEqual(pair.failed_certificates, [])
        self.assertTrue(pair.certificate.holds())

    def test_singular_path_returns_unknown_flux(self):
        quark = self.pc.classify_quark(_certified_evidence(turns=1))
        ev = _certified_evidence(turns=-1)
        conn = obs.FiberConnection()
        A, B = _unit_fiber(1, 3), _unit_fiber(11, 3)
        bad = conn.transport(A, B, np.diag([0.1, 1.0, 1.0]))
        ev.winding = conn.closed_family_winding(
            _winding_family(conn, A, B, turns=-1) + [bad])
        singular = self.pc.classify_quark(ev)
        self.assertIsNone(singular.determinant_winding)
        pair = self.pc.conjugate_pair(quark, singular)
        self.assertIsNone(pair.total_winding)
        self.assertIsNone(pair.total_baryon_flux)  # UNKNOWN, never zero
        self.assertFalse(pair.conserved)
        self.assertIn("winding-second", pair.failed_certificates)
        self.assertFalse(pair.certificate.holds())

    def test_non_conjugate_pair_fails_conservation(self):
        quark = self.pc.classify_quark(_certified_evidence(turns=1))
        pair = self.pc.conjugate_pair(quark, quark)
        self.assertEqual(pair.total_winding, 2)
        self.assertAlmostEqual(pair.total_baryon_flux, 2.0 / 3.0,
                               delta=MACHINE)
        self.assertFalse(pair.conserved)
        self.assertIn("winding-conservation", pair.failed_certificates)

    def test_uncertified_parity_leaves_total_parity_unknown(self):
        quark = self.pc.classify_quark(_certified_evidence(turns=1))
        ev = _certified_evidence(turns=-1)
        ev.parity_read = qm.WickCertificateRead()
        anti = self.pc.classify_quark(ev)
        pair = self.pc.conjugate_pair(quark, anti)
        self.assertEqual(pair.total_parity, 0)
        self.assertFalse(pair.parity_even)
        self.assertIn("parity-second", pair.failed_certificates)
        self.assertFalse(pair.conserved)

    def test_pair_of_odd_clusters_is_even(self):
        # whitepaper parity table: quark + antiquark -> even composite
        quark = self.pc.classify_quark(_certified_evidence(turns=1))
        anti = self.pc.classify_quark(_certified_evidence(turns=-1))
        self.assertEqual(quark.exterior_parity * anti.exterior_parity, +1)
        self.assertEqual(self.pc.conjugate_pair(quark, anti).total_parity, +1)


# =========================================================================== #
# the emergent flavor doublet (no requested dimension)
# =========================================================================== #
class TestFlavorDoublet(unittest.TestCase):
    def setUp(self):
        self.pc = obs.ParticleClusters()

    def test_planted_doublet_emerges(self):
        read = self.pc.flavor_doublet_search(_doublet_frames())
        self.assertTrue(read.found)
        self.assertEqual(read.rank, 2)
        self.assertEqual(read.frames_tracked, 3)
        self.assertEqual(read.two_state_count, 1)
        self.assertAlmostEqual(read.min_continuation_overlap, 1.0,
                               delta=MACHINE)
        self.assertTrue(read.certificate.holds())
        # the search never requested a dimension: every stable rank is
        # reported, not only the two-state one
        self.assertEqual(sorted(read.stable_subclass_ranks), [1, 2, 3])

    def test_no_two_state_subclass_is_unknown(self):
        read = self.pc.flavor_doublet_search(
            _doublet_frames(ranks=(1, 3)))
        self.assertFalse(read.found)
        self.assertIn("flavor-doublet", read.failed_certificates)
        self.assertEqual(read.invalidation_reason,
                         "no-stable-two-state-subclass")
        self.assertFalse(read.certificate.holds())

    def test_doublet_dropping_out_is_unstable(self):
        read = self.pc.flavor_doublet_search(_doublet_frames(drop_rank2_at=2))
        self.assertFalse(read.found)
        self.assertEqual(read.invalidation_reason,
                         "no-stable-two-state-subclass")

    def test_gap_closing_doublet_is_uncertified(self):
        # an unaccepted (gap-closed) band anywhere on the track breaks the
        # certified continuation — the #769 semantics
        read = self.pc.flavor_doublet_search(
            _doublet_frames(unaccept_rank2_at=1))
        self.assertFalse(read.found)

    def test_single_frame_is_insufficient(self):
        read = self.pc.flavor_doublet_search(_doublet_frames(frames=1))
        self.assertFalse(read.found)
        self.assertEqual(read.invalidation_reason, "insufficient-frames")

    def test_ambiguous_two_doublets_stay_uncertified(self):
        read = self.pc.flavor_doublet_search(
            _doublet_frames(extra_rank2=True))
        self.assertFalse(read.found)
        self.assertEqual(read.two_state_count, 2)
        self.assertEqual(read.invalidation_reason,
                         "ambiguous-two-state-subclasses")

    def test_merging_chains_invalidate_each_other(self):
        # two rank-2 bands spanning the SAME cells at frame 0 both best-
        # match the single frame-1 band: the continuation is ambiguous
        f0 = _band_read([_unit_fiber(300, 2),
                         _fiber([[300], [301]], np.eye(2)[:, ::-1])])
        f1 = _band_read([_unit_fiber(300, 2)])
        read = self.pc.flavor_doublet_search([f0, f1])
        self.assertFalse(read.found)

    def test_doublet_carries_the_recorded_trivialization(self):
        read = self.pc.flavor_doublet_search(_doublet_frames(base=400))
        self.assertTrue(read.found)
        self.assertEqual(read.doublet.rank(), 2)
        self.assertEqual(read.doublet.cell_vertices(), [[420], [421]])

    def test_search_is_deterministic(self):
        a = self.pc.flavor_doublet_search(_doublet_frames())
        b = self.pc.flavor_doublet_search(_doublet_frames())
        self.assertEqual(a.found, b.found)
        self.assertEqual(a.stable_subclass_ranks, b.stable_subclass_ranks)
        self.assertEqual(a.min_continuation_overlap, b.min_continuation_overlap)


# =========================================================================== #
# isospin, Gauss charge, and the proposed u/d identification
# =========================================================================== #
class TestIsospinCharge(unittest.TestCase):
    def setUp(self):
        self.pc = obs.ParticleClusters()

    def test_u_fixture(self):
        ev = _certified_evidence(with_flavor=True, occupancy=[1.0, 0.0],
                                 with_charge=2.0 / 3.0)
        read = self.pc.classify_quark(ev)
        self.assertEqual(read.classification, "quark")
        self.assertEqual(read.isospin, 0.5)
        self.assertAlmostEqual(read.electric_flux, 2.0 / 3.0, delta=1e-9)
        self.assertAlmostEqual(read.baryon_flux, 1.0 / 3.0, delta=MACHINE)
        self.assertTrue(read.ud_identification_proposed)
        self.assertNotIn("ud-identification", read.failed_certificates)
        self.assertEqual(read.failed_certificates, [])

    def test_d_fixture(self):
        ev = _certified_evidence(with_flavor=True, occupancy=[0.0, 1.0],
                                 with_charge=-1.0 / 3.0)
        read = self.pc.classify_quark(ev)
        self.assertEqual(read.isospin, -0.5)
        self.assertAlmostEqual(read.electric_flux, -1.0 / 3.0, delta=1e-9)
        self.assertTrue(read.ud_identification_proposed)

    def test_missing_doublet_yields_unknown_flavor_and_charge(self):
        # even with a CONSISTENT Gauss read, the quark charge stays
        # unknown without the doublet (ticket acceptance)
        ev = _certified_evidence(with_charge=2.0 / 3.0)
        read = self.pc.classify_quark(ev)
        self.assertIsNone(read.isospin)
        self.assertIsNone(read.electric_flux)
        self.assertIn("flavor-doublet", read.failed_certificates)
        self.assertEqual(read.classification, "quark")  # quark-ness intact

    def test_unstable_doublet_yields_unknown_flavor_and_charge(self):
        ev = _certified_evidence(with_charge=2.0 / 3.0)
        ev.flavor = self.pc.flavor_doublet_search(
            _doublet_frames(drop_rank2_at=1))
        ev.doublet_occupancy = np.array([1.0, 0.0], dtype=complex)
        read = self.pc.classify_quark(ev)
        self.assertIsNone(read.isospin)
        self.assertIsNone(read.electric_flux)
        self.assertIn("flavor-doublet", read.failed_certificates)

    def test_superposition_occupancy_yields_unknown_isospin(self):
        ev = _certified_evidence(with_flavor=True,
                                 occupancy=[1.0, 1.0])
        read = self.pc.classify_quark(ev)
        self.assertIsNone(read.isospin)
        self.assertIn("isospin", read.failed_certificates)

    def test_inconsistent_gauss_yields_unknown_charge(self):
        ev = _certified_evidence(with_flavor=True, occupancy=[1.0, 0.0])
        ev.charge = self.pc.gauss_flux_consistency(
            [complex(2.0 / 3.0), complex(1.0)])
        read = self.pc.classify_quark(ev)
        self.assertIsNone(read.electric_flux)
        self.assertIn("gauss-consistency", read.failed_certificates)
        # Q = I3 + B/2 is NOT tested without a certified charge
        self.assertNotIn("ud-identification", read.failed_certificates)
        self.assertFalse(read.ud_identification_proposed)

    def test_violated_ud_relation_is_named(self):
        # occupancy says d (I3 = -1/2) but the Gauss flux says +2/3: the
        # proposed identification fails BY NAME; the independently
        # certified fields stay reported
        ev = _certified_evidence(with_flavor=True, occupancy=[0.0, 1.0],
                                 with_charge=2.0 / 3.0)
        read = self.pc.classify_quark(ev)
        self.assertEqual(read.isospin, -0.5)
        self.assertAlmostEqual(read.electric_flux, 2.0 / 3.0, delta=1e-9)
        self.assertIn("ud-identification", read.failed_certificates)
        self.assertFalse(read.ud_identification_proposed)

    def test_declared_orientation_is_recorded(self):
        ev = _certified_evidence(with_flavor=True, occupancy=[0.0, 1.0],
                                 orientation=-1, with_charge=2.0 / 3.0)
        read = self.pc.classify_quark(ev)
        # orientation -1: member 2 carries +1/2 under the declared
        # convention -> the SAME occupancy now reads as the u member
        self.assertEqual(read.isospin, 0.5)
        self.assertEqual(read.doublet_orientation, -1)
        self.assertTrue(read.ud_identification_proposed)

    def test_ud_not_tested_without_baryon_flux(self):
        ev = _certified_evidence(with_flavor=True, occupancy=[1.0, 0.0],
                                 with_charge=2.0 / 3.0)
        conn = obs.FiberConnection()
        A, B = _unit_fiber(1, 3), _unit_fiber(11, 3)
        segment = [_phase_link(conn, A, B, p) for p in (0.0, 0.3)]
        ev.winding = conn.open_segment_winding(segment,
                                             obs.WindingClosureSpec())
        read = self.pc.classify_quark(ev)
        self.assertIsNone(read.baryon_flux)
        self.assertNotIn("ud-identification", read.failed_certificates)
        self.assertFalse(read.ud_identification_proposed)


# =========================================================================== #
# the reused Gauss-flux read on nested enclosing surfaces
# =========================================================================== #
class TestGaussFlux(unittest.TestCase):
    def setUp(self):
        self.pc = obs.ParticleClusters()

    def test_nested_surfaces_consistent_read(self):
        st, F, surfaces = _gauss_fixture(2.0 / 3.0)
        read = self.pc.gauss_flux_on_surfaces(st, F, surfaces, True)
        self.assertTrue(read.consistent)
        self.assertAlmostEqual(read.electric_flux, 2.0 / 3.0, delta=1e-9)
        self.assertEqual(len(read.fluxes), 2)
        self.assertLess(read.max_deviation, 1e-12)
        self.assertTrue(read.certificate.holds())
        self.assertEqual(read.surface_vertex_counts, [1, 4])

    def test_inconsistent_flux_is_unknown(self):
        st, F, surfaces = _gauss_fixture(2.0 / 3.0)
        # perturb one electric plaquette between the two surfaces
        es = cob.EigenstateSynthesis(st, 2)
        n = es.order()
        # find a cell seen by exactly one surface
        for c in range(n):
            probe = [0j] * n
            probe[c] = 1.0 + 0j
            s1 = es.gauss_law_charge(probe, surfaces[0], True)
            s2 = es.gauss_law_charge(probe, surfaces[1], True)
            if abs(s1 - s2) > 0.5:
                F2 = list(F)
                F2[c] += 1.0
                break
        read = self.pc.gauss_flux_on_surfaces(st, F2, surfaces, True)
        self.assertFalse(read.consistent)
        self.assertIsNone(read.electric_flux)
        self.assertIn("gauss-consistency", read.failed_certificates)
        self.assertFalse(read.certificate.holds())

    def test_all_spacelike_complex_reads_certified_zero(self):
        st = _from_simplices(7, _TETRA_CHAIN, timelike=False)
        es = cob.EigenstateSynthesis(st, 2)
        surfaces = obs.ParticleClusters.nested_enclosures(st, [0], 2)
        F = [0.7 + 0j] * es.order()
        read = self.pc.gauss_flux_on_surfaces(st, F, surfaces, True)
        self.assertTrue(read.consistent)
        self.assertEqual(read.electric_flux, 0.0)  # a CERTIFIED zero
        self.assertEqual(read.failed_certificates, [])

    def test_single_surface_makes_no_consistency_claim(self):
        st, F, surfaces = _gauss_fixture(2.0 / 3.0)
        read = self.pc.gauss_flux_on_surfaces(st, F, [surfaces[0]], True)
        self.assertFalse(read.consistent)
        self.assertIsNone(read.electric_flux)
        self.assertIn("gauss-consistency", read.failed_certificates)

    def test_exact_field_strength_has_zero_total_flux(self):
        # topological protection: F = dA is exact, so the FULL (electric +
        # magnetic) closed-surface flux vanishes on every surface
        st = _from_simplices(7, _TETRA_CHAIN)
        es1 = cob.EigenstateSynthesis(st, 1)
        es2 = cob.EigenstateSynthesis(st, 2)
        rng = np.random.default_rng(7)
        A = [complex(v) for v in rng.normal(size=es1.order())]
        F = es2.curvature_from_connection(A)
        surfaces = obs.ParticleClusters.nested_enclosures(st, [0], 2)
        read = self.pc.gauss_flux_on_surfaces(st, F, surfaces,
                                           electric_only=False)
        self.assertTrue(read.consistent)
        self.assertAlmostEqual(read.electric_flux, 0.0, delta=1e-12)

    def test_pure_combination_equals_spacetime_path(self):
        st, F, surfaces = _gauss_fixture(0.25)
        es = cob.EigenstateSynthesis(st, 2)
        fluxes = [es.gauss_law_charge(F, s, True) for s in surfaces]
        via_st = self.pc.gauss_flux_on_surfaces(st, F, surfaces, True)
        pure = self.pc.gauss_flux_consistency(fluxes, [1, 4], True)
        self.assertEqual(via_st.consistent, pure.consistent)
        self.assertEqual(via_st.electric_flux, pure.electric_flux)
        self.assertEqual(via_st.fluxes, pure.fluxes)

    def test_nested_enclosures_are_strictly_growing_here(self):
        st = _from_simplices(7, _TETRA_CHAIN)
        sets_ = obs.ParticleClusters.nested_enclosures(st, [0], 3)
        self.assertEqual(len(sets_), 3)
        self.assertEqual(sets_[0], [0])
        for a, b in zip(sets_, sets_[1:]):
            self.assertTrue(set(a) < set(b))

    def test_nested_enclosures_validates(self):
        st = _from_simplices(7, _TETRA_CHAIN)
        with self.assertRaises(ValueError):
            obs.ParticleClusters.nested_enclosures(st, [], 2)
        with self.assertRaises(ValueError):
            obs.ParticleClusters.nested_enclosures(st, [0], 0)
        with self.assertRaises(ValueError):
            obs.ParticleClusters.nested_enclosures(st, [999], 2)

    def test_imaginary_leakage_is_reported_not_discarded(self):
        read = self.pc.gauss_flux_consistency([complex(0.5, 0.3),
                                             complex(0.5, 0.3)])
        self.assertFalse(read.consistent)  # |Im| above tolerance
        self.assertAlmostEqual(read.imag_leakage, 0.3, delta=MACHINE)
        self.assertIsNone(read.electric_flux)

    def test_gauss_read_is_read_only(self):
        st, F, surfaces = _gauss_fixture(2.0 / 3.0)
        before = st.metric_revision_key()
        self.pc.gauss_flux_on_surfaces(st, F, surfaces, True)
        self.assertEqual(st.metric_revision_key(), before)


# =========================================================================== #
# relabeling / refinement / gauge invariance
# =========================================================================== #
class TestRelabelRefinementGauge(unittest.TestCase):
    def setUp(self):
        self.pc = obs.ParticleClusters()
        self.delta = staticmethod(obs.ObservableGates.report_delta)

    def test_relabeling_preserves_accepted_classification(self):
        base = self.pc.classify_quark(
            _certified_evidence(with_flavor=True, occupancy=[1.0, 0.0],
                                with_charge=2.0 / 3.0))
        shifted_ev = _certified_evidence(band_base=1001, with_flavor=True,
                                         occupancy=[1.0, 0.0],
                                         with_charge=2.0 / 3.0)
        shifted = self.pc.classify_quark(shifted_ev)
        self.assertEqual(
            obs.ObservableGates.report_delta(base.to_record(),
                                             shifted.to_record()), 0.0)

    def test_vertex_relabeling_of_the_gauss_complex(self):
        st, F, surfaces = _gauss_fixture(2.0 / 3.0)
        ids = [i + 500 for i in range(7)]
        st2, F2, surfaces2 = _gauss_fixture(2.0 / 3.0, ids=ids)
        a = self.pc.gauss_flux_on_surfaces(st, F, surfaces, True)
        b = self.pc.gauss_flux_on_surfaces(st2, F2, surfaces2, True)
        self.assertEqual(a.consistent, b.consistent)
        self.assertAlmostEqual(a.electric_flux, b.electric_flux, delta=1e-9)

    def test_refinement_preserves_accepted_classification(self):
        # the refined band adds cells while keeping the original subspace:
        # the measured overlap feeds the refinement certificate
        band = _unit_fiber(1, 3)
        refined = _fiber([[1], [2], [3], [77]],
                         np.vstack([np.eye(3), np.zeros((1, 3))]))
        overlap = obs.SpectralFiber.overlap(band, refined).subspace_overlap
        ev = _certified_evidence()
        ev.refinement_overlap = overlap
        read = self.pc.classify_quark(ev)
        self.assertAlmostEqual(overlap, 1.0, delta=1e-12)
        self.assertEqual(read.classification, "quark")
        self.assertNotIn("refinement-stability", read.failed_certificates)

    def test_in_band_gauge_rotation_leaves_the_verdict(self):
        # an SU(3) in-band frame change of the anchored frame leaves the
        # anchor profile (hence the classification evidence) invariant
        rng = np.random.default_rng(11)
        w = np.array([2.0, 0.5, 1.25])
        z = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
        phi = obs.ColorAnchor.orthonormalize_frame(z, w)
        tri = [tessera.OrientedTriangle([0, 1, 2], [1, 1, 1])]
        p1 = obs.ColorAnchor(tri).evaluate(phi, w)
        # a special-unitary in-band rotation
        g = np.linalg.qr(rng.normal(size=(3, 3))
                         + 1j * rng.normal(size=(3, 3)))[0]
        g = g / np.linalg.det(g) ** (1.0 / 3.0)
        p2 = obs.ColorAnchor(tri).evaluate(phi @ g, w)
        r1 = self.pc.classify_quark(_certified_evidence(anchor=p1))
        r2 = self.pc.classify_quark(_certified_evidence(anchor=p2))
        self.assertEqual(r1.classification, r2.classification)
        self.assertAlmostEqual(r1.triangle_anchor_score,
                               r2.triangle_anchor_score, delta=1e-12)

    def test_transport_order_does_not_matter_for_leakage(self):
        ev = _certified_evidence()
        base = self.pc.classify_quark(ev)
        ev.lifetime_transports = list(reversed(list(ev.lifetime_transports)))
        permuted = self.pc.classify_quark(ev)
        self.assertEqual(base.transport_leakage_max,
                         permuted.transport_leakage_max)
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
        matches = obs.ParticleClusters.track_candidates([a1, a2], [b1])
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].from_index, 0)
        self.assertEqual(matches[0].to_index, 0)
        self.assertTrue(matches[0].certified_continuation)

    def test_record_roundtrip_is_exact(self):
        read = self.pc.classify_quark(
            _certified_evidence(with_flavor=True, occupancy=[1.0, 0.0],
                                with_charge=2.0 / 3.0))
        rec = read.to_record()
        back = obs.QuarkRead.from_record(rec)
        self.assertEqual(
            obs.ObservableGates.report_delta(rec, back.to_record()), 0.0)

    def test_unknown_fields_serialize_as_null_never_zero(self):
        read = self.pc.classify_quark(obs.QuarkCandidateEvidence())
        rec = read.to_record()
        self.assertIsNone(rec["baryon_flux"])
        self.assertIsNone(rec["isospin"])
        self.assertIsNone(rec["electric_flux"])
        self.assertIsNone(rec["determinant_winding"])
        self.assertEqual(rec["exterior_parity"], 0)
        self.assertTrue(math.isnan(rec["triangle_anchor_score"]))

    def test_full_evidence_is_checkpointed(self):
        read = self.pc.classify_quark(_certified_evidence())
        rec = read.to_record()
        for key in ("component_hash", "winding_closure",
                    "failed_certificates", "thresholds", "certificate",
                    "transport_leakage_max", "persistence_lifetime",
                    "localization", "refinement_overlap",
                    "occupation_total", "confidence"):
            self.assertIn(key, rec)
        self.assertEqual(rec["classification"], "quark")
        self.assertEqual(rec["winding_closure"], "closed-family")

    def test_from_record_rejects_unknown_schema(self):
        read = self.pc.classify_quark(_certified_evidence())
        rec = read.to_record()
        rec["schema_version"] = 99
        with self.assertRaises(ValueError):
            obs.QuarkRead.from_record(rec)

    def test_describe_smoke(self):
        read = self.pc.classify_quark(_certified_evidence())
        text = read.describe()
        self.assertIn("quark", text)
        self.assertIn("nu=1", text)

    def _spacetime_backed_evidence(self):
        # a real spacetime band so the cache has a live star to publish
        st = _from_simplices(7, _TETRA_CHAIN, timelike=False)
        tracker = obs.SpectralFiberTracker(st)
        bands = tracker.enumerate_bands(list(range(7)), 0)
        fiber = bands.fibers[0]
        ev = _certified_evidence()
        ev.color_band = fiber  # rank/acceptance of the REAL band applies
        return st, ev

    def test_cached_classification_equals_cold(self):
        st, ev = self._spacetime_backed_evidence()
        cache = cob.AnalyticCache(st)
        cold = self.pc.classify_quark(ev)
        first = self.pc.classify_quark_cached(cache, ev)
        served = self.pc.classify_quark_cached(cache, ev)
        self.assertEqual(
            obs.ObservableGates.report_delta(cold.to_record(),
                                             first.to_record()), 0.0)
        self.assertEqual(
            obs.ObservableGates.report_delta(cold.to_record(),
                                             served.to_record()), 0.0)
        self.assertGreaterEqual(cache.hits, 1)

    def test_touched_star_invalidates_only_the_touching_candidate(self):
        st, ev = self._spacetime_backed_evidence()
        cache = cob.AnalyticCache(st)
        self.pc.classify_quark_cached(cache, ev)
        self.assertEqual(cache.size, 1)
        star = cob.TouchedStar()
        star.add_changed_edge(0, 1)  # touches the band's support
        cache.publish(star)
        self.assertEqual(cache.size, 0)

    def test_disjoint_star_keeps_the_entry(self):
        st, ev = self._spacetime_backed_evidence()
        cache = cob.AnalyticCache(st)
        self.pc.classify_quark_cached(cache, ev)
        star = cob.TouchedStar()
        star.add_changed_edge(9001, 9002)  # disjoint from the band support
        cache.publish(star)
        self.assertEqual(cache.size, 1)

    def test_changed_evidence_never_serves_a_stale_read(self):
        st, ev = self._spacetime_backed_evidence()
        cache = cob.AnalyticCache(st)
        quark = self.pc.classify_quark_cached(cache, ev)
        conn = obs.FiberConnection()
        A, B = _unit_fiber(1, 3), _unit_fiber(11, 3)
        ev.winding = conn.closed_family_winding(
            _winding_family(conn, A, B, turns=-1))
        anti = self.pc.classify_quark_cached(cache, ev)
        self.assertNotEqual(quark.determinant_winding,
                            anti.determinant_winding)

    def test_different_thresholds_have_different_fingerprints(self):
        ev = _certified_evidence()
        loose = obs.ParticleClusters()
        strict_cfg = obs.ParticleClustersConfig()
        strict_cfg.min_anchor_score = 0.99
        strict = obs.ParticleClusters(strict_cfg)
        self.assertNotEqual(loose.evidence_fingerprint(ev),
                            strict.evidence_fingerprint(ev))

    def test_the_stability_window_is_part_of_the_cache_key(self):
        # #808: the frame lifetime and the two stability windows are
        # DECISION evidence, so a change in any of them must recompute
        # rather than serve another window's verdict.
        base = _certified_evidence()
        fingerprint = self.pc.evidence_fingerprint(base)
        for mutate in (
                lambda e: setattr(e, "frame_lifetime", 9.0),
                lambda e: setattr(e, "frame_min_overlap", 0.6),
                lambda e: setattr(e, "color_band_frames",
                                  list(e.color_band_frames)[:2]),
                lambda e: setattr(e, "anchor_frames",
                                  list(e.anchor_frames)[:2]),
        ):
            with self.subTest(mutate=mutate):
                ev = _certified_evidence()
                mutate(ev)
                self.assertNotEqual(fingerprint,
                                    self.pc.evidence_fingerprint(ev))

    def test_a_changed_stability_window_never_serves_a_stale_read(self):
        st, ev = self._spacetime_backed_evidence()
        cache = cob.AnalyticCache(st)
        stable = self.pc.classify_quark_cached(cache, ev)
        ev.color_band_frames = [ev.color_band_frames[0], _unit_fiber(1, 2)]
        unstable = self.pc.classify_quark_cached(cache, ev)
        self.assertNotIn("color-rank-stability", stable.failed_certificates)
        self.assertIn("color-rank-stability", unstable.failed_certificates)
        cold = obs.ParticleClusters().classify_quark(ev)
        self.assertEqual(obs.ObservableGates.report_delta(
            cold.to_record(), unstable.to_record()), 0.0)


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

#: Where the emergence objective and its gradient actually live: the class
#: header and every file of its implementation directory. Nothing on this
#: list may name a derived observable under any circumstances.
_OBJECTIVE_SOURCES = ("include/cobordism/MultiCobordism.h",) + tuple(
    sorted(str(p.relative_to(REPO_ROOT))
           for p in (REPO_ROOT / "src" / "cobordism" / "multicobordism").glob("*")))


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
        if not (REPO_ROOT / "src" / "cobordism").exists():
            self.skipTest("source tree not available")
        # The header plus at least one implementation unit and the private
        # header: an empty glob would make the check vacuous.
        self.assertGreaterEqual(len(_OBJECTIVE_SOURCES), 3)
        offenders = []
        for relative in _OBJECTIVE_SOURCES:
            path = REPO_ROOT / relative
            text = path.read_text(errors="ignore")
            for needle in needles:
                if needle in text:
                    offenders.append("%s names %s" % (relative, needle))
        self.assertEqual(offenders, [])

    def test_classification_is_read_only_on_the_spacetime(self):
        st = _from_simplices(7, _TETRA_CHAIN, timelike=False)
        tracker = obs.SpectralFiberTracker(st)
        bands = tracker.enumerate_bands(list(range(7)), 0)
        ev = _certified_evidence()
        ev.color_band = bands.fibers[0]
        before = st.metric_revision_key()
        obs.ParticleClusters().classify_quark(ev)
        self.assertEqual(st.metric_revision_key(), before)

    def test_classification_cost_per_candidate(self):
        # merge-gate benchmark: classification cost per candidate, cold
        # versus cache-served (numbers reported in the PR body)
        pc = obs.ParticleClusters()
        ev = _certified_evidence()
        n = 200
        t0 = time.perf_counter()
        for _ in range(n):
            pc.classify_quark(ev)
        cold = (time.perf_counter() - t0) / n

        st = _from_simplices(7, _TETRA_CHAIN, timelike=False)
        cache = cob.AnalyticCache(st)
        tracker = obs.SpectralFiberTracker(st)
        ev.color_band = tracker.enumerate_bands(list(range(7)), 0).fibers[0]
        pc.classify_quark_cached(cache, ev)  # warm
        t0 = time.perf_counter()
        for _ in range(n):
            pc.classify_quark_cached(cache, ev)
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
    ev.binding_component = obs.ComponentId("2b" * 16, 2)
    ev.octet = pc.octet_bilinear_read(state, list(modes))
    ev.parity_read = state.wick_parity()
    ev.occupation_read = state.wick_total_number()
    ev.lifetime_transports = family
    ev.winding = conn.closed_family_winding(family)
    ev.persistence_lifetime = lifetime
    # The gated persistence quantity is the COBORDISM-FRAME lifetime (#808);
    # the resolution-slice number travels beside it as a report.
    ev.frame_lifetime = lifetime
    return ev


def _meson_evidence(first_turns=1, second_turns=-1, pairing="singlet"):
    """A two-cluster composite bundle: one #773 quark + one antiquark,
    the carried composite occupation, and the pair color bilinear."""
    pc = obs.ParticleClusters()
    first = pc.classify_quark(_certified_evidence(turns=first_turns))
    second = pc.classify_quark(
        _certified_evidence(turns=second_turns, band_base=31))
    ev = obs.CompositeCandidateEvidence()
    ev.binding_component = obs.ComponentId("3c" * 16, 2)
    ev.first = first
    ev.second = second
    ev.occupation_read = qm.CovarianceState.from_occupations(
        np.array([1.0, 1.0])).wick_total_number()
    if pairing == "singlet":
        ev.color_pairing = np.eye(3, dtype=complex) / math.sqrt(3.0)
    elif pairing == "octet":
        ev.color_pairing = np.asarray(obs.ColorFiber.gell_mann(1))
    ev.persistence_lifetime = 3.0
    return ev


def _diquark_evidence(columns=None, second_turns=1):
    """A two-quark composite bundle with the certified anti-triplet wedge
    occupation det(C^dag Gamma C) of the pair's carried Slater state."""
    pc = obs.ParticleClusters()
    ev = obs.CompositeCandidateEvidence()
    ev.binding_component = obs.ComponentId("4d" * 16, 2)
    ev.first = pc.classify_quark(_certified_evidence(turns=1))
    ev.second = pc.classify_quark(
        _certified_evidence(turns=second_turns, band_base=51))
    C = (np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=complex)
         if columns is None else np.asarray(columns, dtype=complex))
    slater = qm.CovarianceState.from_slater_frame(
        np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=complex))
    ev.anti_triplet_read = slater.wick_gram_determinant(C, C)
    ev.occupation_read = slater.wick_total_number()
    ev.persistence_lifetime = 3.0
    return ev


class TestOctetBilinearRead(unittest.TestCase):
    """Exact small-sector fixtures of the quasi-free octet read."""

    def setUp(self):
        self.pc = obs.ParticleClusters()

    def test_anti_triplet_slater_exact_values(self):
        read = self.pc.octet_bilinear_read(_rank2_state(), [0, 1, 2])
        self.assertAlmostEqual(read.occupation, 2.0, delta=MACHINE)
        self.assertEqual(read.subset_parity, +1)
        self.assertAlmostEqual(read.octet_weight, 2.0 / 3.0, delta=MACHINE)
        self.assertAlmostEqual(read.singlet_weight, 4.0 / 3.0, delta=MACHINE)
        self.assertAlmostEqual(read.casimir, 3.0, delta=1e-12)
        # C2(3bar) = 4/3 by quartic Wick sums -- exact algebra.
        self.assertAlmostEqual(read.casimir_expectation, 4.0 / 3.0,
                               delta=1e-12)
        self.assertLessEqual(read.octet_projector_residual, 1e-14)
        self.assertTrue(read.certificate.holds())
        self.assertEqual(read.certificate.grade,
                         cob.CertificateGrade.AlgebraicallyExact)

    def test_fundamental_exact_values(self):
        c = np.array([1.0, 0.0, 0.0], dtype=complex)
        state = qm.CovarianceState(np.outer(c, c.conj()))
        read = self.pc.octet_bilinear_read(state, [0, 1, 2])
        self.assertAlmostEqual(read.occupation, 1.0, delta=MACHINE)
        self.assertEqual(read.subset_parity, -1)
        self.assertAlmostEqual(read.octet_weight, 2.0 / 3.0, delta=MACHINE)
        self.assertAlmostEqual(read.singlet_weight, 1.0 / 3.0, delta=MACHINE)
        # C2(3) = 4/3 on the fundamental.
        self.assertAlmostEqual(read.casimir_expectation, 4.0 / 3.0,
                               delta=1e-12)

    def test_vacuum_and_full_singlet_read_zero_casimir(self):
        vacuum = qm.CovarianceState(np.zeros((3, 3), dtype=complex))
        read = self.pc.octet_bilinear_read(vacuum, [0, 1, 2])
        self.assertEqual(read.occupation, 0.0)
        self.assertEqual(read.subset_parity, +1)
        self.assertEqual(read.octet_weight, 0.0)
        # a vanished excitation is UNKNOWN, never zero
        self.assertTrue(math.isnan(read.casimir))
        self.assertTrue(math.isnan(read.octet_projector_residual))
        self.assertAlmostEqual(read.casimir_expectation, 0.0, delta=1e-12)

        full = qm.CovarianceState(np.eye(3, dtype=complex))
        read = self.pc.octet_bilinear_read(full, [0, 1, 2])
        self.assertAlmostEqual(read.occupation, 3.0, delta=MACHINE)
        self.assertEqual(read.subset_parity, -1)
        self.assertAlmostEqual(read.octet_weight, 0.0, delta=1e-15)
        # the fully occupied top wedge is a color SINGLET: C2 = 0 exactly.
        self.assertAlmostEqual(read.casimir_expectation, 0.0, delta=1e-12)

    def test_bilinear_is_the_transposed_submatrix(self):
        rng = np.random.default_rng(81)
        z = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
        q = np.linalg.qr(z)[0]
        gamma = q @ np.diag([0.9, 0.4, 0.1]) @ q.conj().T
        state = qm.CovarianceState(gamma)
        read = self.pc.octet_bilinear_read(state, [0, 1, 2])
        self.assertTrue(np.array_equal(read.bilinear, gamma.T))

    def test_split_delegates_to_color_fiber_bitwise(self):
        read = self.pc.octet_bilinear_read(_rank2_state((0.3, 0.5, 0.9)),
                                         [0, 1, 2])
        want = obs.ColorFiber.octet_read(read.bilinear)
        self.assertEqual(read.octet_weight, want.octet)
        self.assertEqual(read.singlet_weight, want.singlet)
        self.assertTrue(np.array_equal(
            read.octet_component,
            obs.ColorFiber.traceless_part(read.bilinear)))
        self.assertAlmostEqual(
            read.casimir, obs.ColorFiber.adjoint_casimir(read.octet_component),
            delta=1e-15)

    def test_gell_mann_components_reconstruct_the_bilinear(self):
        read = self.pc.octet_bilinear_read(_rank2_state((0.2, 0.7, 0.4)),
                                         [0, 1, 2])
        m = np.asarray(read.bilinear)
        recon = (np.trace(m) / 3.0) * np.eye(3, dtype=complex)
        for a in range(1, 9):
            recon = recon + read.gell_mann_components[a - 1] \
                * np.asarray(obs.ColorFiber.gell_mann(a))
        self.assertLessEqual(np.max(np.abs(recon - m)), 1e-13)

    def test_dense_lazy_oracle_cross_validation(self):
        # #771 is the dense oracle: the SAME Slater state through the lazy
        # engine's exact covariance read gives the SAME octet read.
        rng = np.random.default_rng(82)
        orbitals = rng.normal(size=(3, 2)) + 1j * rng.normal(size=(3, 2))
        eng = qm.LazyFockEngine(3)
        wedge = eng.wedge_state([0, 1, 2], orbitals)
        gamma = np.asarray(eng.covariance_matrix(wedge).matrix)
        via_oracle = self.pc.octet_bilinear_read(
            qm.CovarianceState(gamma), [0, 1, 2])
        direct = self.pc.octet_bilinear_read(
            qm.CovarianceState.from_slater_frame(orbitals), [0, 1, 2])
        for field in ("occupation", "octet_weight", "singlet_weight",
                      "casimir", "casimir_expectation"):
            self.assertAlmostEqual(getattr(via_oracle, field),
                                   getattr(direct, field), delta=1e-12,
                                   msg=field)
        self.assertEqual(via_oracle.subset_parity, direct.subset_parity)

    def test_mode_validation(self):
        state = _rank2_state()
        with self.assertRaises(ValueError):
            self.pc.octet_bilinear_read(state, [0, 1])
        with self.assertRaises(ValueError):
            self.pc.octet_bilinear_read(state, [0, 1, 1])
        with self.assertRaises(ValueError):
            self.pc.octet_bilinear_read(state, [0, 1, 7])

    def test_embedded_triad_reads_like_the_small_fixture(self):
        # the color triad on modes (2, 3, 4) of a 6-mode state reads
        # exactly like the standalone 3-mode fixture
        small = self.pc.octet_bilinear_read(_rank2_state(), [0, 1, 2])
        gamma = np.zeros((6, 6), dtype=complex)
        gamma[2:5, 2:5] = np.asarray(_rank2_state().gamma())
        embedded = self.pc.octet_bilinear_read(qm.CovarianceState(gamma),
                                             [2, 3, 4])
        for field in ("occupation", "subset_parity", "octet_weight",
                      "singlet_weight", "casimir", "casimir_expectation"):
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
        base = self.pc.octet_bilinear_read(state, [0, 1, 2])
        moved = self.pc.octet_bilinear_read(relabeled,
                                          [perm[0], perm[1], perm[2]])
        # the echoed color-mode LABELS legitimately track the relabeling;
        # every physical channel is invariant (the permutation reorders
        # the Wick trace accumulation: identical algebra, double
        # round-off ~1e-16)
        rec_base, rec_moved = base.to_record(), moved.to_record()
        self.assertEqual(rec_moved["color_modes"], perm)
        del rec_base["color_modes"], rec_moved["color_modes"]
        self.assertLessEqual(
            obs.ObservableGates.report_delta(rec_base, rec_moved), 1e-14)
        self.assertEqual(base.subset_parity, moved.subset_parity)
        self.assertEqual(base.occupation, moved.occupation)

    def test_color_order_is_the_recorded_trivialization(self):
        # reordering the DECLARED color modes conjugates the bilinear by
        # the permutation: invariant weights/casimir/occupation/parity,
        # covariant components
        state = _rank2_state((0.1, 0.6, 0.5))
        base = self.pc.octet_bilinear_read(state, [0, 1, 2])
        swapped = self.pc.octet_bilinear_read(state, [1, 0, 2])
        for field in ("occupation", "subset_parity", "octet_weight",
                      "singlet_weight", "casimir", "casimir_expectation"):
            self.assertAlmostEqual(getattr(base, field),
                                   getattr(swapped, field), delta=1e-12,
                                   msg=field)
        p = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=complex)
        self.assertLessEqual(
            np.max(np.abs(np.asarray(swapped.bilinear)
                          - p @ np.asarray(base.bilinear) @ p.T)), 1e-15)

    def test_read_is_read_only_on_the_state(self):
        state = _rank2_state()
        before = state.covariance_hash()
        self.pc.octet_bilinear_read(state, [0, 1, 2])
        self.assertEqual(state.covariance_hash(), before)

    def test_record_roundtrip_is_exact(self):
        read = self.pc.octet_bilinear_read(_rank2_state((0.2, 0.3, 0.9)),
                                         [0, 1, 2])
        rec = read.to_record()
        back = obs.OctetBilinearRead.from_record(rec)
        self.assertEqual(
            obs.ObservableGates.report_delta(rec, back.to_record()), 0.0)

    def test_from_record_rejects_unknown_schema(self):
        rec = self.pc.octet_bilinear_read(_rank2_state(), [0, 1, 2]).to_record()
        rec["schema_version"] = 99
        with self.assertRaises(ValueError):
            obs.OctetBilinearRead.from_record(rec)


class TestOctetCollectiveGrowth(unittest.TestCase):
    """Arbitrarily many collective excitations by ADDING microscopic
    modes: the vacuum embedding changes no sector read and no
    two-dimensional edge-mode factor."""

    def setUp(self):
        self.pc = obs.ParticleClusters()

    def test_zero_block_vacuum_extension_leaves_the_read_unchanged(self):
        base = self.pc.octet_bilinear_read(_rank2_state(), [0, 1, 2])
        for extra in (1, 5, 13):
            gamma = np.zeros((3 + extra, 3 + extra), dtype=complex)
            gamma[:3, :3] = np.asarray(_rank2_state().gamma())
            grown = self.pc.octet_bilinear_read(qm.CovarianceState(gamma),
                                              [0, 1, 2])
            self.assertEqual(obs.ObservableGates.report_delta(
                base.to_record(), grown.to_record()), 0.0,
                msg=f"extra={extra}")

    def test_lazy_vacuum_embedding_leaves_the_read_unchanged(self):
        # the #771 inductive-limit route: iota(psi) = psi (x) |0> adds
        # microscopic modes; every existing sector read is unchanged
        rng = np.random.default_rng(83)
        orbitals = rng.normal(size=(3, 2)) + 1j * rng.normal(size=(3, 2))
        eng = qm.LazyFockEngine(8)
        small = eng.wedge_state([0, 1, 2], orbitals)
        grown = eng.embed_in_vacuum(small, [3, 4, 5, 6, 7])
        g_small = np.asarray(eng.covariance_matrix(small).matrix)
        g_grown = np.asarray(eng.covariance_matrix(grown).matrix)
        a = self.pc.octet_bilinear_read(qm.CovarianceState(g_small),
                                      [0, 1, 2])
        b = self.pc.octet_bilinear_read(qm.CovarianceState(g_grown),
                                      [0, 1, 2])
        # identical algebra; the engine's closed-form Slater covariance
        # evaluates through a different GEMM shape after the embedding,
        # so the doubles agree to round-off (~1e-16), not bitwise
        self.assertLessEqual(obs.ObservableGates.report_delta(
            a.to_record(), b.to_record()), 1e-14)
        self.assertEqual(a.subset_parity, b.subset_parity)
        # the added modes carry NOTHING: their covariance rows are
        # EXACTLY zero -- the two-level factor of every new mode is
        # untouched vacuum
        self.assertEqual(np.max(np.abs(g_grown[3:, :])), 0.0)

    def test_two_level_edge_mode_factor_never_changes(self):
        # each finite edge-mode factor remains TWO-dimensional: the stage
        # dimension exactly doubles per added microscopic mode, before and
        # after any collective excitation
        for m in range(1, 12):
            self.assertEqual(qm.LazyFockEngine.stage_dimension(m + 1),
                             2 * qm.LazyFockEngine.stage_dimension(m))

    def test_multiple_collective_excitations_coexist(self):
        # two independent octet excitations on disjoint triads of a
        # 6-mode state: each triad's read equals its standalone fixture
        gamma = np.zeros((6, 6), dtype=complex)
        gamma[:3, :3] = np.asarray(_rank2_state().gamma())
        gamma[3:, 3:] = np.asarray(_rank2_state((0.5, 0.5, 0.0)).gamma())
        state = qm.CovarianceState(gamma)
        a = self.pc.octet_bilinear_read(state, [0, 1, 2])
        b = self.pc.octet_bilinear_read(state, [3, 4, 5])
        ref_a = self.pc.octet_bilinear_read(_rank2_state(), [0, 1, 2])
        ref_b = self.pc.octet_bilinear_read(_rank2_state((0.5, 0.5, 0.0)),
                                          [0, 1, 2])
        for read, ref in ((a, ref_a), (b, ref_b)):
            self.assertEqual(read.occupation, ref.occupation)
            self.assertEqual(read.octet_weight, ref.octet_weight)
            self.assertEqual(read.subset_parity, ref.subset_parity)
            self.assertAlmostEqual(read.casimir_expectation,
                                   ref.casimir_expectation, delta=1e-12)

    def test_scaling_with_mode_count(self):
        # scaling test: the SAME embedded triad read at growing mode
        # count -- values constant, cost polynomial (timed and printed)
        base = self.pc.octet_bilinear_read(_rank2_state(), [0, 1, 2])
        timings = []
        for total in (3, 12, 24, 48):
            gamma = np.zeros((total, total), dtype=complex)
            gamma[:3, :3] = np.asarray(_rank2_state().gamma())
            state = qm.CovarianceState(gamma)
            t0 = time.perf_counter()
            read = self.pc.octet_bilinear_read(state, [0, 1, 2])
            timings.append((total, time.perf_counter() - t0))
            self.assertEqual(read.occupation, base.occupation)
            self.assertEqual(read.octet_weight, base.octet_weight)
            self.assertAlmostEqual(read.casimir_expectation,
                                   base.casimir_expectation, delta=1e-12)
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
        cold = self.pc.octet_bilinear_read(state, [0, 1, 2])
        first = self.pc.octet_bilinear_read_cached(cache, self.ids, state,
                                                [0, 1, 2])
        served = self.pc.octet_bilinear_read_cached(cache, self.ids, state,
                                                 [0, 1, 2])
        self.assertEqual(obs.ObservableGates.report_delta(
            cold.to_record(), first.to_record()), 0.0)
        self.assertEqual(obs.ObservableGates.report_delta(
            cold.to_record(), served.to_record()), 0.0)
        self.assertGreaterEqual(cache.hits, 1)

    def test_touched_star_invalidates(self):
        cache = cob.AnalyticCache(self.st)
        self.pc.octet_bilinear_read_cached(cache, self.ids, _rank2_state(),
                                        [0, 1, 2])
        self.assertEqual(cache.size, 1)
        star = cob.TouchedStar()
        star.add_changed_edge(0, 1)
        cache.publish(star)
        self.assertEqual(cache.size, 0)

    def test_gamma_change_never_serves_a_stale_read(self):
        cache = cob.AnalyticCache(self.st)
        a = self.pc.octet_bilinear_read_cached(cache, self.ids, _rank2_state(),
                                            [0, 1, 2])
        changed = _rank2_state((1.0, 0.0, 0.0))
        b = self.pc.octet_bilinear_read_cached(cache, self.ids, changed,
                                            [0, 1, 2])
        cold = self.pc.octet_bilinear_read(changed, [0, 1, 2])
        self.assertEqual(obs.ObservableGates.report_delta(
            b.to_record(), cold.to_record()), 0.0)
        self.assertFalse(np.array_equal(np.asarray(a.bilinear),
                                        np.asarray(b.bilinear)))

    def test_fingerprint_sensitivity(self):
        state = _rank2_state()
        base = self.pc.octet_fingerprint(state, [0, 1, 2])
        self.assertNotEqual(base,
                            self.pc.octet_fingerprint(state, [1, 0, 2]))
        self.assertNotEqual(
            base, self.pc.octet_fingerprint(_rank2_state((1.0, 0.0, 0.0)),
                                           [0, 1, 2]))
        strict_cfg = obs.ParticleClustersConfig()
        strict_cfg.parity_tolerance = 1e-3
        self.assertNotEqual(
            base, obs.ParticleClusters(strict_cfg).octet_fingerprint(
                state, [0, 1, 2]))


class TestGluonClassification(unittest.TestCase):
    """Design spec section 14.3: a gluon candidate is a persistent
    transported octet excitation with zero baryon flux and even parity."""

    GATES = ["parity-even", "octet-excitation", "octet-purity",
             "octet-transport", "winding-zero", "persistence"]

    def setUp(self):
        self.pc = obs.ParticleClusters()

    def test_certified_gluon_candidate(self):
        read = self.pc.classify_gluon(_gluon_evidence())
        self.assertEqual(read.classification, "gluon-candidate")
        self.assertEqual(read.confidence, 1.0)
        self.assertEqual(read.failed_certificates, [])
        self.assertEqual(read.exterior_parity, +1)
        self.assertEqual(read.determinant_winding, 0)
        # a CERTIFIED zero flux -- 0.0 as evidence, not a default
        self.assertEqual(read.baryon_flux, 0.0)
        self.assertAlmostEqual(read.occupation_total, 2.0, delta=MACHINE)
        self.assertAlmostEqual(read.casimir, 3.0, delta=1e-12)
        # C2(3bar) = 4/3: the flat consumed-scalar summary (one source of
        # truth -- the full OctetBilinearRead travels on the evidence)
        self.assertAlmostEqual(read.casimir_expectation, 4.0 / 3.0,
                               delta=1e-12)
        self.assertLessEqual(read.octet_projector_residual, 1e-14)
        self.assertEqual(read.winding_closure, "closed-family")
        self.assertTrue(read.certificate.holds())
        self.assertEqual(read.certificate.grade,
                         cob.CertificateGrade.StructureExact)

    def test_candidate_is_the_strongest_claim(self):
        # ticket out-of-scope: no even octet excitation is claimed to be a
        # physical gluon -- the accepted verdict string is EXACTLY
        # "gluon-candidate"
        read = self.pc.classify_gluon(_gluon_evidence())
        self.assertEqual(read.classification, "gluon-candidate")
        self.assertNotEqual(read.classification, "gluon")

    def test_odd_carried_state_is_rejected_from_the_even_read(self):
        # negative control: an ODD-sector object (N = 1 fundamental) fails
        # the even-parity gate by name
        c = np.array([1.0, 0.0, 0.0], dtype=complex)
        odd = qm.CovarianceState(np.outer(c, c.conj()))
        read = self.pc.classify_gluon(_gluon_evidence(state=odd))
        self.assertEqual(read.classification, "none")
        self.assertEqual(read.exterior_parity, -1)
        self.assertIn("parity-even", read.failed_certificates)
        self.assertNotIn("winding-zero", read.failed_certificates)

    def test_nonzero_winding_octet_excitation_is_not_a_gluon(self):
        # negative control: certified nu = 1 is honest evidence (B = 1/3
        # reported) but NOT a gluon candidate
        read = self.pc.classify_gluon(_gluon_evidence(turns=1))
        self.assertEqual(read.classification, "none")
        self.assertIn("winding-zero", read.failed_certificates)
        self.assertEqual(read.determinant_winding, 1)
        self.assertAlmostEqual(read.baryon_flux, 1.0 / 3.0, delta=MACHINE)

    def test_unknown_winding_leaves_flux_unknown(self):
        ev = _gluon_evidence()
        conn = obs.FiberConnection()
        A, B = _unit_fiber(1, 3), _unit_fiber(11, 3)
        segment = [_phase_link(conn, A, B, p) for p in (0.0, 0.3, 0.6)]
        ev.winding = conn.open_segment_winding(segment,
                                             obs.WindingClosureSpec())
        read = self.pc.classify_gluon(ev)
        self.assertIsNone(read.determinant_winding)
        self.assertIsNone(read.baryon_flux)  # UNKNOWN, never zero
        self.assertIn("winding-zero", read.failed_certificates)

    def test_missing_octet_read_fails_by_name(self):
        ev = _gluon_evidence()
        ev.octet = obs.OctetBilinearRead()
        read = self.pc.classify_gluon(ev)
        self.assertIn("octet-excitation", read.failed_certificates)
        self.assertIn("octet-purity", read.failed_certificates)
        self.assertTrue(math.isnan(read.casimir))
        self.assertTrue(math.isnan(read.casimir_expectation))
        self.assertTrue(math.isnan(read.octet_weight))

    def test_vacuum_carries_no_excitation(self):
        vacuum = qm.CovarianceState(np.zeros((3, 3), dtype=complex))
        read = self.pc.classify_gluon(_gluon_evidence(state=vacuum))
        self.assertEqual(read.classification, "none")
        self.assertIn("octet-excitation", read.failed_certificates)
        # vacuum parity is even -- that gate PASSES; the excitation gate
        # is what refuses
        self.assertNotIn("parity-even", read.failed_certificates)

    def test_rank_two_transport_is_not_an_octet_transport(self):
        ev = _gluon_evidence()
        conn = obs.FiberConnection()
        A2, B2 = _unit_fiber(61, 2), _unit_fiber(71, 2)
        ev.lifetime_transports = [conn.transport(A2, B2,
                                                np.eye(2, dtype=complex))]
        read = self.pc.classify_gluon(ev)
        self.assertIn("octet-transport", read.failed_certificates)

    def test_leaky_transport_fails(self):
        ev = _gluon_evidence()
        conn = obs.FiberConnection()
        A, B = _unit_fiber(1, 3), _unit_fiber(11, 3)
        leaky = conn.transport(A, B, np.diag([0.4, 1.0, 1.0]))
        ev.lifetime_transports = list(ev.lifetime_transports) + [leaky]
        read = self.pc.classify_gluon(ev)
        self.assertIn("octet-transport", read.failed_certificates)

    def test_missing_transports_fail(self):
        ev = _gluon_evidence()
        ev.lifetime_transports = []
        read = self.pc.classify_gluon(ev)
        self.assertIn("octet-transport", read.failed_certificates)
        self.assertTrue(math.isnan(read.transport_leakage_max))

    def test_insufficient_persistence(self):
        read = self.pc.classify_gluon(_gluon_evidence(lifetime=1.0))
        self.assertIn("persistence", read.failed_certificates)
        self.assertEqual(read.classification, "none")

    def test_uncertified_parity_never_emits_a_sign(self):
        ev = _gluon_evidence()
        ev.parity_read = qm.WickCertificateRead()
        read = self.pc.classify_gluon(ev)
        self.assertEqual(read.exterior_parity, 0)
        self.assertIn("parity-even", read.failed_certificates)

    def test_confidence_is_the_passed_fraction(self):
        ev = _gluon_evidence(turns=1, lifetime=1.0)  # two gates fail
        read = self.pc.classify_gluon(ev)
        self.assertAlmostEqual(read.confidence, 4.0 / 6.0, delta=MACHINE)
        self.assertEqual(sorted(read.failed_certificates),
                         ["persistence", "winding-zero"])

    def test_thresholds_are_recorded(self):
        cfg = obs.ParticleClustersConfig()
        cfg.min_octet_weight = 0.123
        read = obs.ParticleClusters(cfg).classify_gluon(_gluon_evidence())
        self.assertEqual(read.thresholds.min_octet_weight, 0.123)

    def test_record_roundtrip_and_null_semantics(self):
        read = self.pc.classify_gluon(_gluon_evidence())
        rec = read.to_record()
        back = obs.GluonRead.from_record(rec)
        self.assertEqual(
            obs.ObservableGates.report_delta(rec, back.to_record()), 0.0)
        empty = self.pc.classify_gluon(obs.GluonCandidateEvidence())
        rec = empty.to_record()
        self.assertIsNone(rec["determinant_winding"])
        self.assertIsNone(rec["baryon_flux"])
        self.assertEqual(rec["exterior_parity"], 0)
        self.assertTrue(math.isnan(rec["occupation_total"]))
        for name in self.GATES:
            self.assertIn(name, empty.failed_certificates)

    def test_relabeling_preserves_the_read(self):
        base = self.pc.classify_gluon(_gluon_evidence(band_base=1))
        shifted = self.pc.classify_gluon(_gluon_evidence(band_base=901))
        self.assertEqual(obs.ObservableGates.report_delta(
            base.to_record(), shifted.to_record()), 0.0)

    def test_simplex_reorientation_preserves_the_verdict(self):
        # the ORIENTATION channel: a common row sign flip (reversing a
        # cell's orientation flips its cochain component on every column
        # alike).  det C picks up det(S) = +-1 and the SINGLET certificate
        # |det C|^2 is exactly invariant.
        base = self.pc.classify_baryon(_baryon_evidence())
        for signs in ([1, 1, -1], [-1, -1, -1], [-1, 1, -1]):
            columns = np.diag(signs).astype(complex) @ _color_triad()
            read = self.pc.classify_baryon(_baryon_evidence(color=columns))
            self.assertEqual(read.classification, base.classification)
            self.assertAlmostEqual(read.color_gram_determinant,
                                   base.color_gram_determinant, delta=MACHINE)
            expected = base.color_wedge * float(np.prod(signs))
            self.assertLess(abs(read.color_wedge - expected), 1e-13)

    def test_refinement_sample_order_does_not_change_stability(self):
        samples = _scale_samples()
        forward = self.pc.scale_profile(samples)
        backward = self.pc.scale_profile(list(reversed(samples)))
        self.assertEqual(forward.stable, backward.stable)
        self.assertEqual(forward.radius_ratio_spread,
                         backward.radius_ratio_spread)
        self.assertEqual(forward.profile_max_deviation,
                         backward.profile_max_deviation)
        drifting = _scale_samples(drift=0.05)
        self.assertEqual(
            self.pc.scale_profile(drifting).failed_certificates,
            self.pc.scale_profile(list(reversed(drifting)))
            .failed_certificates)

    def test_cold_replay_is_deterministic(self):
        a = self.pc.classify_gluon(_gluon_evidence())
        b = obs.ParticleClusters().classify_gluon(_gluon_evidence())
        self.assertEqual(obs.ObservableGates.report_delta(
            a.to_record(), b.to_record()), 0.0)
        # and the checkpoint replay: from_record(toRecord) re-serializes
        # bit-identically (the cold-replay acceptance channel)
        rec = a.to_record()
        self.assertEqual(obs.ObservableGates.report_delta(
            rec, obs.GluonRead.from_record(rec).to_record()), 0.0)

    def test_describe_smoke(self):
        text = self.pc.classify_gluon(_gluon_evidence()).describe()
        self.assertIn("gluon-candidate", text)
        self.assertIn("B=0", text)


class TestMesonClassification(unittest.TestCase):
    """Quark-antiquark singlet composites as meson candidates."""

    def setUp(self):
        self.pc = obs.ParticleClusters()

    def test_certified_meson_candidate(self):
        read = self.pc.classify_meson(_meson_evidence())
        self.assertEqual(read.classification, "meson-candidate")
        self.assertEqual(read.failed_certificates, [])
        # EVEN color singlet with ZERO total baryon flux (acceptance)
        self.assertEqual(read.exterior_parity, +1)
        self.assertEqual(read.total_winding, 0)
        self.assertEqual(read.total_baryon_flux, 0.0)
        self.assertLessEqual(read.pairing_octet_fraction, 1e-15)
        self.assertAlmostEqual(read.occupation_total, 2.0, delta=MACHINE)
        self.assertTrue(read.certificate.holds())

    def test_order_insensitive(self):
        ev = _meson_evidence()
        swapped = _meson_evidence()
        swapped.first, swapped.second = ev.second, ev.first
        a = self.pc.classify_meson(ev)
        b = self.pc.classify_meson(swapped)
        self.assertEqual(a.classification, b.classification)
        self.assertEqual(a.total_winding, b.total_winding)
        self.assertEqual(a.exterior_parity, b.exterior_parity)

    def test_octet_pairing_is_not_a_meson(self):
        # a q-qbar pair in the OCTET channel is a gluon-sector object,
        # not a color-singlet meson
        read = self.pc.classify_meson(_meson_evidence(pairing="octet"))
        self.assertEqual(read.classification, "none")
        self.assertIn("color-singlet", read.failed_certificates)
        self.assertAlmostEqual(read.pairing_octet_fraction, 1.0,
                               delta=1e-15)

    def test_missing_pairing_fails_by_name(self):
        read = self.pc.classify_meson(_meson_evidence(pairing="none"))
        self.assertIn("color-singlet", read.failed_certificates)
        self.assertTrue(math.isnan(read.pairing_octet_fraction))

    def test_two_quarks_are_not_a_meson(self):
        read = self.pc.classify_meson(
            _meson_evidence(first_turns=1, second_turns=1))
        self.assertEqual(read.classification, "none")
        self.assertIn("constituent-antiquark", read.failed_certificates)
        self.assertNotIn("constituent-quark", read.failed_certificates)
        # and the flux channel refuses too: nu total = 2, not 0
        self.assertIn("flux-zero", read.failed_certificates)

    def test_two_antiquarks_are_not_a_meson(self):
        read = self.pc.classify_meson(
            _meson_evidence(first_turns=-1, second_turns=-1))
        self.assertIn("constituent-quark", read.failed_certificates)
        self.assertNotIn("constituent-antiquark", read.failed_certificates)

    def test_singular_constituent_leaves_flux_unknown(self):
        ev = _meson_evidence()
        conn = obs.FiberConnection()
        A, B = _unit_fiber(1, 3), _unit_fiber(11, 3)
        bad_ev = _certified_evidence(turns=-1)
        bad = conn.transport(A, B, np.diag([0.1, 1.0, 1.0]))
        bad_ev.winding = conn.closed_family_winding(
            _winding_family(conn, A, B, turns=-1) + [bad])
        ev.second = self.pc.classify_quark(bad_ev)
        read = self.pc.classify_meson(ev)
        self.assertIsNone(read.total_winding)
        self.assertIsNone(read.total_baryon_flux)  # UNKNOWN, never zero
        self.assertIn("flux-zero", read.failed_certificates)

    def test_uncertified_constituent_parity_is_unknown(self):
        ev = _meson_evidence()
        blind = _certified_evidence(turns=-1, band_base=31)
        blind.parity_read = qm.WickCertificateRead()
        ev.second = self.pc.classify_quark(blind)
        read = self.pc.classify_meson(ev)
        self.assertEqual(read.exterior_parity, 0)
        self.assertIn("parity-even", read.failed_certificates)

    def test_composite_parity_is_the_exact_graded_product(self):
        # whitepaper parity table: two odd constituents compose EVEN
        ev = _meson_evidence()
        read = self.pc.classify_meson(ev)
        self.assertEqual(read.exterior_parity,
                         ev.first.exterior_parity * ev.second.exterior_parity)
        self.assertEqual(read.exterior_parity, +1)

    def test_composite_transport_leakage_is_reported(self):
        # the ticket's report set: transport leakage travels on the read
        # (report-only for the two-cluster composites -- never a gate)
        ev = _meson_evidence()
        conn = obs.FiberConnection()
        A, B = _unit_fiber(1, 3), _unit_fiber(11, 3)
        ev.lifetime_transports = [_phase_link(conn, A, B, 0.1)]
        read = self.pc.classify_meson(ev)
        self.assertEqual(read.transport_count, 1)
        self.assertLessEqual(read.transport_leakage_max, 1e-9)
        self.assertEqual(read.classification, "meson-candidate")

    def test_record_roundtrip_and_describe(self):
        read = self.pc.classify_meson(_meson_evidence())
        rec = read.to_record()
        back = obs.MesonRead.from_record(rec)
        self.assertEqual(
            obs.ObservableGates.report_delta(rec, back.to_record()), 0.0)
        self.assertIn("meson-candidate", read.describe())
        rec["schema_version"] = 99
        with self.assertRaises(ValueError):
            obs.MesonRead.from_record(rec)

    def test_relabeling_and_cold_replay(self):
        a = self.pc.classify_meson(_meson_evidence())
        b = obs.ParticleClusters().classify_meson(_meson_evidence())
        self.assertEqual(obs.ObservableGates.report_delta(
            a.to_record(), b.to_record()), 0.0)


class TestDiquarkClassification(unittest.TestCase):
    """Two-quark anti-triplet even composites with preserved B = 2/3."""

    def setUp(self):
        self.pc = obs.ParticleClusters()

    def test_certified_diquark_candidate(self):
        read = self.pc.classify_diquark(_diquark_evidence())
        self.assertEqual(read.classification, "diquark-candidate")
        self.assertEqual(read.failed_certificates, [])
        # even 3bar state with B = 2/3 (acceptance)
        self.assertEqual(read.exterior_parity, +1)
        self.assertEqual(read.total_winding, 2)
        self.assertEqual(read.total_baryon_flux, 2.0 / 3.0)
        self.assertAlmostEqual(read.anti_triplet_weight, 1.0, delta=MACHINE)
        self.assertAlmostEqual(read.occupation_total, 2.0, delta=MACHINE)
        self.assertTrue(read.certificate.holds())

    def test_explicitly_not_an_antiquark(self):
        # The #773 distinction fixture, composed: the SAME two-quark
        # carried state fed to the QUARK classifier (anti-triplet color,
        # nu = -1 tube) refuses on occupation/parity; the diquark read
        # accepts with B = +2/3 -- opposite sign and triple the magnitude
        # of an antiquark's B = -1/3.
        quark_view = self.pc.classify_quark(
            _certified_evidence(turns=-1, occupations=(1.0, 1.0, 0.0)))
        self.assertEqual(quark_view.classification, "none")
        self.assertIn("parity-odd", quark_view.failed_certificates)
        self.assertIn("occupation-one", quark_view.failed_certificates)

        read = self.pc.classify_diquark(_diquark_evidence())
        self.assertEqual(read.classification, "diquark-candidate")
        self.assertEqual(read.total_baryon_flux, 2.0 / 3.0)
        self.assertNotEqual(read.total_baryon_flux, -1.0 / 3.0)
        self.assertAlmostEqual(read.occupation_total, 2.0, delta=MACHINE)
        self.assertEqual(read.exterior_parity, +1)

    def test_duplicated_color_mode_is_pauli_zero(self):
        # det(C^dag Gamma C) with a repeated color column is EXACTLY zero
        # (the Gram/Pauli identity): the anti-triplet gate refuses
        dup = np.array([[1.0, 1.0], [0.0, 0.0], [0.0, 0.0]], dtype=complex)
        ev = _diquark_evidence(columns=dup)
        self.assertEqual(ev.anti_triplet_read.value, 0.0 + 0.0j)
        read = self.pc.classify_diquark(ev)
        self.assertEqual(read.classification, "none")
        self.assertIn("anti-triplet", read.failed_certificates)
        self.assertEqual(read.anti_triplet_weight, 0.0)

    def test_quark_antiquark_pair_is_not_a_diquark(self):
        read = self.pc.classify_diquark(_diquark_evidence(second_turns=-1))
        self.assertEqual(read.classification, "none")
        self.assertIn("constituent-quarks", read.failed_certificates)
        # nu total = 0, not 2: the flux channel refuses independently
        self.assertIn("baryon-flux-two-thirds", read.failed_certificates)
        self.assertEqual(read.total_winding, 0)

    def test_missing_wedge_read_fails_by_name(self):
        ev = _diquark_evidence()
        ev.anti_triplet_read = qm.WickCertificateRead()
        read = self.pc.classify_diquark(ev)
        self.assertIn("anti-triplet", read.failed_certificates)
        self.assertTrue(math.isnan(read.anti_triplet_weight))

    def test_constituent_flux_is_preserved(self):
        ev = _diquark_evidence()
        read = self.pc.classify_diquark(ev)
        self.assertEqual(read.total_baryon_flux,
                         ev.first.baryon_flux + ev.second.baryon_flux)

    def test_singular_constituent_leaves_flux_unknown(self):
        ev = _diquark_evidence()
        conn = obs.FiberConnection()
        A, B = _unit_fiber(1, 3), _unit_fiber(11, 3)
        bad_ev = _certified_evidence(turns=1, band_base=51)
        bad = conn.transport(A, B, np.diag([0.1, 1.0, 1.0]))
        bad_ev.winding = conn.closed_family_winding(
            _winding_family(conn, A, B, turns=1) + [bad])
        ev.second = self.pc.classify_quark(bad_ev)
        read = self.pc.classify_diquark(ev)
        self.assertIsNone(read.total_winding)
        self.assertIsNone(read.total_baryon_flux)
        self.assertIn("baryon-flux-two-thirds", read.failed_certificates)

    def test_record_roundtrip_and_describe(self):
        read = self.pc.classify_diquark(_diquark_evidence())
        rec = read.to_record()
        back = obs.DiquarkRead.from_record(rec)
        self.assertEqual(
            obs.ObservableGates.report_delta(rec, back.to_record()), 0.0)
        self.assertIn("diquark-candidate", read.describe())

    def test_cold_replay_is_deterministic(self):
        a = self.pc.classify_diquark(_diquark_evidence())
        b = obs.ParticleClusters().classify_diquark(_diquark_evidence())
        self.assertEqual(obs.ObservableGates.report_delta(
            a.to_record(), b.to_record()), 0.0)


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
        rec = pc.classify_quark(_certified_evidence()).to_record()
        for key in ("min_octet_weight", "octet_purity_tolerance",
                    "composite_octet_tolerance", "min_anti_triplet_weight"):
            self.assertIn(key, rec["thresholds"])
            del rec["thresholds"][key]
        back = obs.QuarkRead.from_record(rec)
        defaults = obs.ParticleClustersConfig()
        self.assertEqual(back.thresholds.min_octet_weight,
                         defaults.min_octet_weight)
        self.assertEqual(back.thresholds.min_anti_triplet_weight,
                         defaults.min_anti_triplet_weight)

    def test_new_thresholds_enter_the_evidence_fingerprint(self):
        ev = _certified_evidence()
        base = obs.ParticleClusters()
        cfg = obs.ParticleClustersConfig()
        cfg.min_octet_weight = 0.5
        self.assertNotEqual(base.evidence_fingerprint(ev),
                            obs.ParticleClusters(cfg).evidence_fingerprint(ev))

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
            pc.octet_bilinear_read(state, [0, 1, 2])
        octet = (time.perf_counter() - t0) / n
        t0 = time.perf_counter()
        for _ in range(n):
            pc.classify_gluon(gluon_ev)
        gluon = (time.perf_counter() - t0) / n
        t0 = time.perf_counter()
        for _ in range(n):
            pc.classify_meson(meson_ev)
        meson = (time.perf_counter() - t0) / n
        t0 = time.perf_counter()
        for _ in range(n):
            pc.classify_diquark(diquark_ev)
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
                 "spin-expectation", "sharp-spin", "odd-monopole",
                 "projective-cocycle", "spin-lift", "finite-radius",
                 "profile-stability"]
BARYON_GATES = BARYON_STRUCTURAL + BARYON_PROTON

_UD_CACHE = {}


def _ud_quark(kind):
    """A CERTIFIED #773 u or d quark: nu = +1, odd parity, I3 = +-1/2 under
    the recorded doublet orientation, and a Gauss-consistent electric flux
    (the merged #773 fixtures, memoized -- each rebuild solves a Gauss
    least-squares problem)."""
    if kind not in _UD_CACHE:
        _UD_CACHE[kind] = obs.ParticleClusters().classify_quark(
            _certified_evidence(
                with_flavor=True,
                occupancy=[1.0, 0.0] if kind == "u" else [0.0, 1.0],
                with_charge=2.0 / 3.0 if kind == "u" else -1.0 / 3.0))
    return _UD_CACHE[kind]


def _relabel_quark(read, level=1, tag="cd"):
    """The same certified read carried by a DIFFERENT label-free component
    identity (the relabeling channel: identity is a hash, never a name)."""
    record = read.to_record()
    record["component_hash"] = tag * 16
    record["component_level"] = int(level)
    return obs.QuarkRead.from_record(record)


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
    state = qm.CovarianceState.from_occupations(np.array([1.0, 0.0]))
    return (state.wick_spin_squared_expectation(*js),
            state.wick_spin_squared_variance(*js))


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
    state = qm.CovarianceState.from_slater_frame(orbital)
    return (state.wick_spin_squared_expectation(*js),
            state.wick_spin_squared_variance(*js))


def _delta_spin_reads():
    """The Delta oracle: three aligned spins, J^2 = 15/4 with Var = 0 (a
    SHARP spin that is simply not 3/4)."""
    js = _pauli_over_sites(3)
    orbitals = np.zeros((6, 3), dtype=complex)
    for k in range(3):
        orbitals[2 * k, k] = 1.0
    state = qm.CovarianceState.from_slater_frame(orbitals)
    return (state.wick_spin_squared_expectation(*js),
            state.wick_spin_squared_variance(*js))


_MONOPOLE_CACHE = {}


def _monopole_spin_read(monopole=1):
    """The #1196 odd-monopole evidence: the tetrahedral cut carrying total
    outward flux 2 pi mu, its projective rotation action D_k(g) and the
    j = 1/2 doublet that action protects.  mu = 1 is the odd sector the
    proton certificate demands; mu = 0 and mu = 2 are the even controls."""
    if monopole not in _MONOPOLE_CACHE:
        support = obs.MonopoleSupport.tetrahedron(monopole)
        _MONOPOLE_CACHE[monopole] = support.spin_read(
            obs.MonopoleSupport.tetrahedral_rotations())
    return _MONOPOLE_CACHE[monopole]


_EIGEN_CACHE = {}


def _sharp_spin_eigen(kind):
    """The #1196 sharp-spin eigen read on a superposition of determinants.

    'sharp'     -- one occupied mode of one spin-1/2 doublet: an exact
                   J^2 = 3/4 eigenstate, so both eigen-equations hold.
    'isotropic' -- a right state contaminated by a j = 3/2 component paired
                   with a PURE left eigenvector: the biorthogonal
                   expectation is exactly 3/4 and the complex variance
                   exactly zero, so the variance criterion would accept it,
                   while the right eigen-equation refuses it.  This is the
                   case the whitepaper says a variance cannot decide.
    'delta'     -- three aligned spins: an exact eigenstate, but at 15/4.
    """
    if kind in _EIGEN_CACHE:
        return _EIGEN_CACHE[kind]
    if kind == "sharp":
        js = obs.SharpSpin.doublet_spin_matrices(1)
        state = obs.SharpSpin.determinant([0], 2)
        read = obs.SharpSpin.read(js, state, state.conj())
    elif kind == "delta":
        js = obs.SharpSpin.doublet_spin_matrices(3)
        state = obs.SharpSpin.determinant([0, 2, 4], 6)
        read = obs.SharpSpin.read(js, state, state.conj())
    elif kind == "isotropic":
        js = obs.SharpSpin.doublet_spin_matrices(3)
        basis = np.column_stack([
            obs.SharpSpin.determinant([a, 2 + b, 4 + c], 6)
            for a in (0, 1) for b in (0, 1) for c in (0, 1)])
        block = basis.conj().T @ obs.SharpSpin.total_spin_squared_matrix(js) \
            @ basis
        _values, vectors = np.linalg.eigh(block)
        half = basis @ vectors[:, 0]
        three_half = basis @ vectors[:, 7]
        read = obs.SharpSpin.read(js, half + (0.4 + 0.3j) * three_half,
                                  half.conj())
    else:
        raise ValueError("unknown sharp-spin eigen fixture %r" % (kind,))
    _EIGEN_CACHE[kind] = read
    return read


# The eigen fixture each quasi-free spin fixture is paired with.  A mode with
# no eigen fixture supplies no eigen read at all, so `sharp-spin` fails as
# missing evidence rather than as a refuted claim.
_SPIN_EIGEN_KIND = {"sharp": "sharp", "generic": "isotropic",
                    "delta": "delta", "expectation-only": None,
                    "none": None}


_ROTATION_CACHE = {}


def _rotation_character(turns=1, steps=16, d=4):
    """The #772 executable total-space 2pi cluster-frame cycle against its
    matched co-moving non-rotating reference: chi_hat(2pi) = -1 for one
    turn, +1 for two.  ONE global rotation of the whole carried frame —
    never a product of per-hole Bloch vectors."""
    key = (turns, steps, d)
    if key not in _ROTATION_CACHE:
        EH = obs.ExchangeHolonomy
        frame0 = EH.transverse_spinor_frame(0, 1, d)
        weights = np.ones(d, dtype=complex)
        loop = EH.loop_holonomy(
            EH.rotation_loop_frames(frame0, 0, 1, d, turns, steps), weights)
        reference = EH.loop_holonomy(
            EH.reference_loop_frames(frame0, steps), weights)
        _ROTATION_CACHE[key] = EH.rotation_character(loop, reference)
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
    return EH.exchange_character(
        EH.loop_holonomy(frames, weights),
        EH.loop_holonomy([frames[0]] * steps, weights))


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
    return EH.spin_lift(edges, rotations, triangles, 3)


def _obstructed_spin_lift():
    """The pillowcase class: two triangles glued along three pi-rotation
    edges — w2 evaluates 1 and the lift is REJECTED (#772 fixture)."""
    EH = obs.ExchangeHolonomy
    edges = [(0, 1), (1, 2), (2, 0)]
    rotations = [_rotation3(0, math.pi), _rotation3(1, math.pi),
                 _rotation3(2, math.pi)]
    return EH.spin_lift(edges, rotations, [[0, 1, 2], [0, 2, 1]], 3)


def _color_triad():
    """An ORTHONORMAL anchored color triad: the exact #767 Fourier frame F3
    (assembled from the algebraic omega table), |det F3| = 1 exactly."""
    return np.asarray(obs.ColorFiber.fourier_frame())


def _su3_element(theta=0.7):
    """A g in SU(3) (certified by ColorFiber.is_special_unitary): a plane
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
    return obs.ParticleClusters().octet_bilinear_read(state, [0, 1, 2])


def _polarized_flux(hole=(0.0, 0.0, 1.0)):
    """A color-POLARIZED (anti-triplet) carried state: nonzero net color
    flux -- octet weight 2/3."""
    return obs.ParticleClusters().octet_bilinear_read(_rank2_state(hole),
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
        sample.radius_cross_check = cross
        sample.spectral_mass = mass
        sample.localization = localization
        sample.radial_weight_profile = [p + profile_drift * k for p in profile]
        sample.color_gram_determinant = color_gram + certificate_drift * k
        sample.rotation_character = rotation + certificate_drift * k
        sample.baryon_flux = baryon_flux + certificate_drift * k
        sample.electric_flux = electric_flux + certificate_drift * k
        sample.composite_parity = parity
        sample.anchor_score = anchor_score + certificate_drift * k
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
        graph = obs.PersistentModularity.from_weighted_edges(
            src, tgt, [1.0] * len(src))
        cfg = tessera.PersistentModularityConfig()
        cfg.restarts = 4
        cfg.base_seed = 0
        cfg.overlap_threshold = 0.5
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
            cand.mutual_transports = [good if leakage_ok else leaky]
        out.append(cand)
    return out


def _binding(**kw):
    """The certified bound-supercomponent read of the planted hierarchy."""
    _groups, _fine, coarse = _modular_hierarchy()
    reads = obs.ParticleClusters().bound_supercomponent_search(
        coarse.components, _bound_candidates(**kw))
    return reads[0]


def _baryon_evidence(kinds=("u", "u", "d"), spin="sharp", rotation_turns=1,
                     color=None, flux=None, samples=None, binding=None,
                     continuum=False, spin_lift=None, class_variances=None,
                     dense_j2=None, quarks=None, exchange=None,
                     crossing_mass=None, crossing_baryon=None,
                     monopole=1, spin_eigen="from-spin"):
    """A complete #775 three-cluster evidence bundle."""
    ev = obs.BaryonCandidateEvidence()
    _groups, _fine, coarse = _modular_hierarchy()
    ev.bound_component = coarse.components[0].id
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
    ev.color_columns = _color_triad() if color is None else np.asarray(color)
    ev.color_flux = _filled_triplet_flux() if flux is None else flux
    ev.rotation = _rotation_character(turns=rotation_turns)
    ev.continuum_spin_claim = continuum
    if spin_lift is not None:
        ev.spin_lift = spin_lift
    if exchange is not None:
        ev.exchange = exchange
    if monopole is not None:
        ev.monopole_spin = _monopole_spin_read(monopole)
    eigen_kind = (_SPIN_EIGEN_KIND.get(spin) if spin_eigen == "from-spin"
                  else spin_eigen)
    if eigen_kind is not None:
        ev.sharp_spin_eigen = _sharp_spin_eigen(eigen_kind)
    if spin == "sharp":
        ev.spin_squared_read, ev.spin_variance_read = _sharp_spin_reads()
    elif spin == "generic":
        ev.spin_squared_read, ev.spin_variance_read = _generic_slater_spin_reads()
    elif spin == "delta":
        ev.spin_squared_read, ev.spin_variance_read = _delta_spin_reads()
    elif spin == "expectation-only":
        ev.spin_squared_read, _v = _sharp_spin_reads()
    elif spin == "none":
        pass
    if class_variances is not None:
        ev.class_variance_reads = list(class_variances)
    if dense_j2 is not None:
        ev.total_space_j2 = dense_j2
    ev.scale_samples = _scale_samples() if samples is None else samples
    ev.persistence_lifetime = 4.0
    if crossing_mass is not None:
        ev.crossing_mass = crossing_mass
    if crossing_baryon is not None:
        ev.crossing_baryon = crossing_baryon
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
    verts = [st.create_vertex(i) for i in range(4)]
    for a, b in [(0, 1), (0, 2), (0, 3), (1, 2), (2, 3), (1, 3)]:
        st.create_simplex([verts[a], verts[b]])
    for e in st.get_edge_list().to_vector():
        e.set_length(1.0 + 0j)
        e.set_phase(0.0)
    for a, b in _CONE_RUNGS:
        for e in st.get_edge_list().to_vector():
            if {e.get_source().get_id(), e.get_target().get_id()} == {a, b}:
                e.set_length(1j)
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
        verts = [st.create_vertex(i) for i in range(3)]
        for a, b in [(0, 1), (1, 2), (0, 2)]:
            st.create_simplex([verts[a], verts[b]])
        for e in st.get_edge_list().to_vector():
            e.set_length(1.0 + 0j)
            e.set_phase(0.0)
        # Subject: the crossing conjunct, not localization.  The 3-cycle is
        # vertex-transitive, so every band carries localization excess
        # exactly 1; declare the permissive analysis cap.  A positive band's
        # Krein signature is the diagonal weights' certificate (the Whitney
        # pencil's bands carry the bilinear pairing and no inertia), so the
        # diagonal source is named.
        band_cfg = obs.SpectralFiberConfig()
        band_cfg.max_localization_excess = 1.0
        tracker = obs.SpectralFiberTracker(
            st, band_cfg, metric_source=cob.HodgeMetricSource.DiagonalWeights)
        for fiber in tracker.enumerate_bands([0, 1, 2], 1).fibers:
            cert = fiber.certificate()
            if (fiber.rank() == 1 and cert.accepted
                    and cert.positive_signature == 1
                    and cert.negative_signature == 0):
                _CROSSING_BAND_RECORD = fiber.to_record()
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
    return obs.SpectralFiber.from_record(record)


def _crossing_reads(orientations=(1, 1, 1), windings=None):
    """The crossing mass and coherent baryon sum of three quark tubes on the
    cone, read at the level 0.5 against the M0 reference level 0.0."""
    temporal = obs.CrossingReadouts.temporal_function(
        _crossing_cone(), _CONE_M0)
    windings = windings or [None] * len(orientations)
    tubes = []
    for index, (orientation, winding) in enumerate(zip(orientations, windings)):
        tube = obs.WorldTubeInput()
        tube.tube_id = f"t{index}"
        tube.band = _crossing_band(_CONE_RUNGS[index])
        tube.orientation = orientation
        tube.certified_quark_tube = True
        if winding is not None:
            tube.determinant_winding = winding
        tubes.append(tube)
    mass = obs.CrossingReadouts.crossing_mass(tubes, temporal, 0.5, 0.0)
    baryon = obs.CrossingReadouts.baryon_number(tubes, temporal, 0.5, 0.0)
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
        read = self.pc.classify_baryon(_baryon_evidence())
        self.assertFalse(read.crossing_mass_applicable)
        self.assertNotIn("crossing-readouts", read.failed_certificates)
        self.assertIsNone(read.crossing_baryon_number)
        self.assertTrue(math.isnan(read.crossing_mass_value))

    def test_three_forward_tubes_satisfy_the_gate(self):
        mass, baryon = _crossing_reads((1, 1, 1))
        self.assertAlmostEqual(baryon.baryon_number, 1.0, delta=MACHINE)
        read = self.pc.classify_baryon(
            _baryon_evidence(crossing_mass=mass, crossing_baryon=baryon))
        self.assertTrue(read.crossing_mass_applicable)
        self.assertNotIn("crossing-readouts", read.failed_certificates)
        self.assertAlmostEqual(read.crossing_baryon_number, 1.0, delta=MACHINE)
        self.assertAlmostEqual(read.crossing_mass_value, 3.0, delta=MACHINE)

    def test_wrong_baryon_number_fails_the_gate_by_name(self):
        """Two forward tubes and one reversed give B = 1/3, not 1."""
        mass, baryon = _crossing_reads((1, 1, -1))
        self.assertAlmostEqual(baryon.baryon_number, 1.0 / 3.0, delta=MACHINE)
        read = self.pc.classify_baryon(
            _baryon_evidence(crossing_mass=mass, crossing_baryon=baryon))
        self.assertIn("crossing-readouts", read.failed_certificates)

    def test_determinant_sign_defect_fails_the_gate_by_name(self):
        """A tube whose crossing sign disagrees with its certified winding is
        a defect signal, and the certificate refuses on it."""
        mass, baryon = _crossing_reads((1, 1, 1), windings=[1, 1, -1])
        self.assertEqual(list(baryon.sign_defects), ["t2"])
        read = self.pc.classify_baryon(
            _baryon_evidence(crossing_mass=mass, crossing_baryon=baryon))
        self.assertIn("crossing-readouts", read.failed_certificates)
        self.assertEqual(list(read.crossing_sign_defects), ["t2"])

    def test_half_a_bundle_fails_rather_than_grading_half(self):
        mass, _baryon = _crossing_reads((1, 1, 1))
        read = self.pc.classify_baryon(_baryon_evidence(crossing_mass=mass))
        self.assertTrue(read.crossing_mass_applicable)
        self.assertIn("crossing-readouts", read.failed_certificates)

    def test_crossing_fields_survive_the_record_round_trip(self):
        mass, baryon = _crossing_reads((1, 1, 1), windings=[1, 1, -1])
        read = self.pc.classify_baryon(
            _baryon_evidence(crossing_mass=mass, crossing_baryon=baryon))
        rebuilt = obs.BaryonRead.from_record(read.to_record())
        self.assertEqual(rebuilt.crossing_mass_applicable,
                         read.crossing_mass_applicable)
        self.assertAlmostEqual(rebuilt.crossing_mass_value,
                               read.crossing_mass_value, delta=MACHINE)
        self.assertAlmostEqual(rebuilt.crossing_baryon_number,
                               read.crossing_baryon_number, delta=MACHINE)
        self.assertEqual(list(rebuilt.crossing_sign_defects),
                         list(read.crossing_sign_defects))


class TestColorSingletCertificate(unittest.TestCase):
    """Design spec 16.3 / whitepaper: S_ABC = det[c_A c_B c_C] and the Gram
    determinant det(C^dag C), with the three-mode wedge built exactly ONCE."""

    def setUp(self):
        self.pc = obs.ParticleClusters()

    def test_orthonormal_triad_gives_unit_gram_determinant(self):
        read = self.pc.classify_baryon(_baryon_evidence())
        self.assertAlmostEqual(read.color_gram_determinant, 1.0, delta=MACHINE)
        self.assertAlmostEqual(abs(read.color_wedge), 1.0, delta=MACHINE)
        self.assertNotIn("color-singlet", read.failed_certificates)

    def test_gram_is_the_color_fiber_singlet_gram(self):
        # the delegation is PINNED: |det C|^2 read off the single wedge
        # equals the #767 kernel's own det(C^dag C).
        columns = _color_triad()
        read = self.pc.classify_baryon(_baryon_evidence(color=columns))
        self.assertAlmostEqual(read.color_gram_determinant,
                               obs.ColorFiber.singlet_gram(columns),
                               delta=MACHINE)
        self.assertAlmostEqual(
            abs(read.color_wedge - obs.ColorFiber.color_wedge(columns)),
            0.0, delta=MACHINE)

    def test_wedge_is_su3_invariant(self):
        g = _su3_element()
        self.assertTrue(obs.ColorFiber.is_special_unitary(g))
        base = self.pc.classify_baryon(_baryon_evidence())
        rotated = self.pc.classify_baryon(
            _baryon_evidence(color=g @ _color_triad()))
        self.assertLess(abs(base.color_wedge - rotated.color_wedge), 1e-14)
        self.assertAlmostEqual(base.color_gram_determinant,
                               rotated.color_gram_determinant, delta=MACHINE)
        self.assertEqual(base.classification, rotated.classification)

    def test_duplicate_color_modes_fail_the_singlet_certificate(self):
        columns = _color_triad()
        columns[:, 2] = columns[:, 0]        # duplicated color mode
        read = self.pc.classify_baryon(_baryon_evidence(color=columns))
        self.assertLess(abs(read.color_gram_determinant), 1e-25)
        self.assertLess(abs(read.color_wedge), 1e-13)
        self.assertIn("color-singlet", read.failed_certificates)
        self.assertEqual(read.classification, "baryon-candidate")

    def test_wedge_is_built_once_no_extra_fermion_sign(self):
        # A transposition of two color columns flips det C (the epsilon is
        # ALREADY inside the determinant) and leaves the SINGLET certificate
        # |det C|^2 exactly invariant.  No second fermion sign is applied:
        # the composite statistics come from the constituent parities.
        columns = _color_triad()
        swapped = columns[:, [1, 0, 2]]
        a = self.pc.classify_baryon(_baryon_evidence(color=columns))
        b = self.pc.classify_baryon(_baryon_evidence(color=swapped))
        self.assertLess(abs(a.color_wedge + b.color_wedge), 1e-14)
        self.assertAlmostEqual(a.color_gram_determinant,
                               b.color_gram_determinant, delta=MACHINE)
        self.assertEqual(a.exterior_parity, b.exterior_parity)
        self.assertEqual(a.classification, b.classification)

    def test_missing_color_evidence_is_unknown_never_zero(self):
        read = self.pc.classify_baryon(
            _baryon_evidence(color=np.zeros((3, 3), dtype=complex)))
        self.assertTrue(math.isnan(read.color_gram_determinant))
        self.assertTrue(math.isnan(read.color_wedge.real))
        self.assertIn("color-singlet", read.failed_certificates)

    def test_unnormalized_columns_are_normalized_once(self):
        # scaling a column is a frame convention, not physics: the singlet
        # certificate is unchanged because normalization happens once,
        # before the single wedge.
        columns = _color_triad()
        columns[:, 0] *= 7.5
        read = self.pc.classify_baryon(_baryon_evidence(color=columns))
        self.assertAlmostEqual(read.color_gram_determinant, 1.0, delta=MACHINE)
        self.assertEqual(read.classification, "certified-proton")

    def test_collinear_columns_are_degenerate(self):
        columns = _color_triad()
        columns[:, 1] = columns[:, 0] + columns[:, 2]
        read = self.pc.classify_baryon(_baryon_evidence(color=columns))
        self.assertLess(read.color_gram_determinant, 1e-25)
        self.assertIn("color-singlet", read.failed_certificates)


class TestNetColorFluxDiagnostic(unittest.TestCase):
    """The INDEPENDENT vanishing net-color-flux check (never on its own a
    proof of confinement on a finite complex)."""

    def setUp(self):
        self.pc = obs.ParticleClusters()

    def test_filled_triplet_carries_zero_net_color_flux(self):
        flux = _filled_triplet_flux()
        self.assertEqual(flux.octet_weight, 0.0)
        self.assertAlmostEqual(flux.singlet_weight, 3.0, delta=MACHINE)
        read = self.pc.classify_baryon(_baryon_evidence(flux=flux))
        self.assertEqual(read.color_flux, 0.0)
        self.assertNotIn("color-flux-zero", read.failed_certificates)

    def test_polarized_color_state_fails_flux_zero(self):
        flux = _polarized_flux()
        self.assertAlmostEqual(flux.octet_weight, 2.0 / 3.0, delta=MACHINE)
        read = self.pc.classify_baryon(_baryon_evidence(flux=flux))
        self.assertAlmostEqual(read.color_flux, 2.0 / 3.0, delta=MACHINE)
        self.assertIn("color-flux-zero", read.failed_certificates)
        self.assertEqual(read.classification, "baryon-candidate")

    def test_missing_octet_read_fails_by_name(self):
        read = self.pc.classify_baryon(
            _baryon_evidence(flux=obs.OctetBilinearRead()))
        self.assertTrue(math.isnan(read.color_flux))
        self.assertIn("color-flux-zero", read.failed_certificates)

    def test_flux_is_independent_of_the_gram_certificate(self):
        # unit Gram columns with a POLARIZED carried state: the singlet
        # certificate passes and the flux diagnostic refuses on its own.
        read = self.pc.classify_baryon(_baryon_evidence(flux=_polarized_flux()))
        self.assertAlmostEqual(read.color_gram_determinant, 1.0, delta=MACHINE)
        self.assertNotIn("color-singlet", read.failed_certificates)
        self.assertIn("color-flux-zero", read.failed_certificates)


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
        read = self.pc.bound_supercomponent_search(self.coarse.components,
                                                 _bound_candidates())
        self.assertEqual(len(read), 1)
        self.assertTrue(read[0].found)
        self.assertEqual(read[0].failed_certificates, [])
        self.assertEqual(len(read[0].quarks), 3)
        self.assertEqual(read[0].quark_indices, [0, 1, 2])
        self.assertEqual(read[0].lifetime_window, (2, 6))
        self.assertEqual(read[0].lifetime_overlap, 5.0)
        self.assertEqual(read[0].min_containment, 1.0)
        self.assertTrue(read[0].certificate.holds())

    def test_same_level_is_not_the_next_modular_level(self):
        level = self.coarse.components[0].id.level()
        read = self.pc.bound_supercomponent_search(
            self.coarse.components,
            _bound_candidates(levels=[level] * 3))[0]
        self.assertFalse(read.found)
        self.assertIn("supercomponent-level", read.failed_certificates)

    def test_two_candidates_fail_the_quark_count(self):
        read = self.pc.bound_supercomponent_search(
            self.coarse.components, _bound_candidates(kinds=("u", "d")))[0]
        self.assertFalse(read.found)
        self.assertIn("quark-count", read.failed_certificates)
        self.assertEqual(len(read.quarks), 2)

    def test_four_candidates_fail_the_quark_count(self):
        read = self.pc.bound_supercomponent_search(
            self.coarse.components,
            _bound_candidates(kinds=("u", "u", "d", "d")))[0]
        self.assertFalse(read.found)
        self.assertIn("quark-count", read.failed_certificates)
        self.assertEqual(len(read.quarks), 4)

    def test_uncertified_candidate_is_not_a_quark_candidate(self):
        candidates = _bound_candidates()
        # an antiquark leg: a certified read, but not a "quark" verdict
        candidates[2].quark = self.pc.classify_quark(
            _certified_evidence(turns=-1))
        read = self.pc.bound_supercomponent_search(self.coarse.components,
                                                 candidates)[0]
        self.assertEqual(len(read.quarks), 2)
        self.assertIn("quark-count", read.failed_certificates)

    def test_support_escaping_the_supercomponent_fails_containment(self):
        supports = [list(self.groups[0]), list(self.groups[1]),
                    list(self.groups[2]) + [999]]
        read = self.pc.bound_supercomponent_search(
            self.coarse.components,
            _bound_candidates(supports=supports))[0]
        self.assertFalse(read.found)
        self.assertIn("support-containment", read.failed_certificates)
        self.assertAlmostEqual(read.min_containment, 6.0 / 7.0, delta=MACHINE)

    def test_disjoint_lifetimes_fail_the_overlap(self):
        read = self.pc.bound_supercomponent_search(
            self.coarse.components,
            _bound_candidates(lifetimes=[(0, 1), (2, 3), (4, 5)]))[0]
        self.assertFalse(read.found)
        self.assertIn("lifetime-overlap", read.failed_certificates)
        self.assertEqual(read.lifetime_overlap, 0.0)
        self.assertIsNone(read.lifetime_window)

    def test_missing_lifetime_is_unknown_never_presumed(self):
        read = self.pc.bound_supercomponent_search(
            self.coarse.components,
            _bound_candidates(lifetimes=[(2, 6), None, (2, 6)]))[0]
        self.assertFalse(read.found)
        self.assertIn("lifetime-overlap", read.failed_certificates)
        self.assertIsNone(read.lifetime_window)

    def test_partial_lifetime_overlap_is_measured(self):
        read = self.pc.bound_supercomponent_search(
            self.coarse.components,
            _bound_candidates(lifetimes=[(0, 4), (3, 9), (2, 6)]))[0]
        self.assertTrue(read.found)
        self.assertEqual(read.lifetime_window, (3, 4))
        self.assertEqual(read.lifetime_overlap, 2.0)

    def test_leaky_mutual_transport_fails_containment(self):
        read = self.pc.bound_supercomponent_search(
            self.coarse.components,
            _bound_candidates(leakage_ok=False))[0]
        self.assertFalse(read.found)
        self.assertIn("transport-containment", read.failed_certificates)

    def test_missing_transports_fail_by_name(self):
        read = self.pc.bound_supercomponent_search(
            self.coarse.components, _bound_candidates(transports=False))[0]
        self.assertFalse(read.found)
        self.assertIn("transport-containment", read.failed_certificates)
        self.assertEqual(read.transport_count, 0)
        self.assertTrue(math.isnan(read.transport_leakage_max))

    def test_components_without_candidates_emit_no_read(self):
        candidates = _bound_candidates(
            supports=[[900], [901], [902]])
        self.assertEqual(
            self.pc.bound_supercomponent_search(self.coarse.components,
                                              candidates), [])

    def test_fine_components_are_not_supercomponents(self):
        # each level-1 clique contains exactly ONE candidate: three reads,
        # none of them bound.
        reads = self.pc.bound_supercomponent_search(self.fine.components,
                                                  _bound_candidates())
        self.assertEqual(len(reads), 3)
        for read in reads:
            self.assertFalse(read.found)
            self.assertIn("quark-count", read.failed_certificates)

    def test_thresholds_and_describe_travel(self):
        read = self.pc.bound_supercomponent_search(self.coarse.components,
                                                 _bound_candidates())[0]
        self.assertEqual(read.thresholds.min_support_containment, 1.0)
        self.assertIn("bound", read.describe())

    def test_candidate_order_does_not_change_the_verdict(self):
        candidates = _bound_candidates()
        a = self.pc.bound_supercomponent_search(self.coarse.components,
                                              candidates)[0]
        b = self.pc.bound_supercomponent_search(
            self.coarse.components, list(reversed(candidates)))[0]
        self.assertEqual(a.found, b.found)
        self.assertEqual(a.lifetime_overlap, b.lifetime_overlap)
        self.assertEqual(sorted(q.canonical_hash() for q in a.quarks),
                         sorted(q.canonical_hash() for q in b.quarks))


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
        read = self.pc.scale_profile(_scale_samples())
        self.assertTrue(read.stable)
        self.assertEqual(read.failed_certificates, [])
        self.assertEqual(read.sample_count, 3)
        self.assertEqual(read.radius, 0.75)
        self.assertTrue(read.radius_finite)
        self.assertAlmostEqual(read.radius_ratio, 1.5, delta=MACHINE)
        self.assertEqual(read.spectral_mass, 2.25)
        self.assertEqual(read.radius_ratio_spread, 0.0)
        self.assertEqual(read.spectral_mass_spread, 0.0)
        self.assertEqual(read.profile_max_deviation, 0.0)
        self.assertEqual(read.profile_shells, 3)
        self.assertTrue(read.certificate.holds())
        self.assertEqual(read.certificate.grade,
                         cob.CertificateGrade.CertifiedNumerical)

    def test_single_sample_cannot_measure_stability(self):
        read = self.pc.scale_profile(_scale_samples(count=1))
        self.assertFalse(read.stable)
        self.assertIn("refinement-window", read.failed_certificates)
        # every stability channel is UNMEASURED (NaN), never zero
        self.assertTrue(math.isnan(read.radius_ratio_spread))
        self.assertTrue(math.isnan(read.profile_max_deviation))
        self.assertTrue(read.radius_finite)   # the radius itself is finite

    def test_empty_window_is_not_stable(self):
        read = self.pc.scale_profile([])
        self.assertFalse(read.stable)
        self.assertFalse(read.radius_finite)
        self.assertEqual(sorted(read.failed_certificates), sorted(self.GATES))

    def test_infinite_radius_fails(self):
        samples = _scale_samples()
        samples[1].radius = float("inf")
        read = self.pc.scale_profile(samples)
        self.assertFalse(read.radius_finite)
        self.assertIn("finite-radius", read.failed_certificates)

    def test_nan_radius_fails(self):
        samples = _scale_samples()
        samples[0].radius = NAN
        read = self.pc.scale_profile(samples)
        self.assertFalse(read.radius_finite)
        self.assertIn("finite-radius", read.failed_certificates)

    def test_nonpositive_radius_fails(self):
        samples = _scale_samples(radius=0.0)
        read = self.pc.scale_profile(samples)
        self.assertFalse(read.radius_finite)
        self.assertIn("finite-radius", read.failed_certificates)

    def test_drifting_radius_ratio_fails(self):
        read = self.pc.scale_profile(_scale_samples(drift=0.05))
        self.assertFalse(read.stable)
        self.assertIn("radius-ratio-stability", read.failed_certificates)
        # ratios 1.5/1.6/1.7 (cross = 0.5): (max - min)/max(|mean|, 1)
        # = 0.2/1.6 = 0.125 exactly.
        self.assertAlmostEqual(read.radius_ratio_spread, 0.125, delta=1e-12)

    def test_drifting_spectral_mass_fails(self):
        samples = _scale_samples()
        samples[2].spectral_mass = 2.30
        read = self.pc.scale_profile(samples)
        self.assertFalse(read.stable)
        self.assertIn("spectral-mass-stability", read.failed_certificates)

    def test_drifting_localization_fails(self):
        samples = _scale_samples()
        samples[1].localization = 0.4
        read = self.pc.scale_profile(samples)
        self.assertFalse(read.stable)
        self.assertIn("localization-stability", read.failed_certificates)

    def test_drifting_radial_profile_fails(self):
        read = self.pc.scale_profile(_scale_samples(profile_drift=0.01))
        self.assertFalse(read.stable)
        self.assertIn("profile-stability", read.failed_certificates)
        self.assertAlmostEqual(read.profile_max_deviation, 0.02, delta=1e-12)

    def test_missing_radial_profile_fails_by_name(self):
        # no shell seeds: the radial profile is UNKNOWN, never "stable at
        # zero" (the honest reading of an unavailable channel).
        read = self.pc.scale_profile(_scale_samples(profile=()))
        self.assertFalse(read.stable)
        self.assertIn("profile-stability", read.failed_certificates)
        self.assertTrue(math.isnan(read.profile_max_deviation))
        self.assertEqual(read.profile_shells, 0)

    def test_shell_count_mismatch_fails(self):
        samples = _scale_samples()
        samples[1].radial_weight_profile = [0.5, 0.5]
        read = self.pc.scale_profile(samples)
        self.assertFalse(read.stable)
        self.assertIn("profile-stability", read.failed_certificates)
        self.assertTrue(math.isnan(read.profile_max_deviation))

    def test_physical_mass_is_always_unknown(self):
        # keeping any dimensionful mass UNKNOWN until a physical scale is
        # independently established (ticket scope, stated verbatim).
        read = self.pc.scale_profile(_scale_samples())
        self.assertTrue(read.stable)
        self.assertIsNone(read.physical_mass)

    def test_describe_names_the_failures(self):
        read = self.pc.scale_profile(_scale_samples(count=1))
        self.assertIn("refinement-window", read.describe())
        self.assertIn("physical mass unknown", read.describe())


class TestScaleProfileFromTheExistingBattery(unittest.TestCase):
    """The adapter over RegisterContext.interiorHinges — the same reader
    EmergentRadius / EmergentMass use, on the hand-checkable closed forms."""

    @staticmethod
    def _boundary_delta5():
        st = tessera.Spacetime.from_vertex_tuples(
            4, [list(c) for c in itertools.combinations(range(6), 5)],
            1.0, 0.0)
        st.materialize_facets()
        return st

    @staticmethod
    def _star_of_apex():
        st = tessera.Spacetime.from_vertex_tuples(
            4, [list(c) for c in itertools.combinations(range(6), 5)
                if 5 in c], 1.0, 0.0)
        st.materialize_facets()
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
        sample = obs.ParticleClusters.scale_profile_sample(ctx)
        self.assertAlmostEqual(sample.radius, volume ** 0.25, places=12)
        self.assertAlmostEqual(sample.radius_cross_check, volume ** 0.25,
                               places=12)
        self.assertAlmostEqual(sample.spectral_mass, deficit, places=9)
        self.assertAlmostEqual(sample.localization, 1.0, places=9)
        # no emergent holes seed the BFS: the radial profile is UNKNOWN
        self.assertEqual(sample.radial_weight_profile, [])

    def test_closed_s4_window_is_stable_but_has_no_radial_profile(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ctx = obs.RegisterContext(self._boundary_delta5(), 0, 3,
                                      cob.ProtonSynthesis.singlet())
        sample = obs.ParticleClusters.scale_profile_sample(ctx)
        read = obs.ParticleClusters().scale_profile([sample, sample])
        self.assertTrue(read.radius_finite)
        self.assertEqual(read.radius_ratio_spread, 0.0)
        self.assertEqual(read.spectral_mass_spread, 0.0)
        self.assertFalse(read.stable)
        # The battery sample fills the MASS-RADIUS channels only.  Since
        # #808 the window also carries the candidate's other dimensionless
        # certificates, which this geometry read cannot know: they are
        # UNKNOWN and are NAMED, never passed vacuously.
        self.assertEqual(read.failed_certificates,
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
        sample = obs.ParticleClusters.scale_profile_sample(ctx)
        self.assertEqual(sample.radial_weight_profile, [1.0])
        deficit = 2.0 * math.pi - 3.0 * math.acos(0.25)
        self.assertAlmostEqual(sample.spectral_mass, deficit, places=9)
        self.assertGreater(sample.radius, 0.0)
        read = obs.ParticleClusters().scale_profile([sample, sample])
        self.assertEqual(read.profile_shells, 1)
        self.assertEqual(read.profile_max_deviation, 0.0)
        # Every BATTERY channel is stable; the candidate's remaining
        # dimensionless certificates were never supplied by this geometry
        # read, so they are named (#808) and the window is not `stable`.
        self.assertFalse(read.stable)
        for name in ("refinement-window", "finite-radius",
                     "radius-ratio-stability", "spectral-mass-stability",
                     "localization-stability", "profile-stability"):
            self.assertNotIn(name, read.failed_certificates)
        self.assertEqual(read.failed_certificates,
                         ["color-gram-stability",
                          "rotation-character-stability",
                          "baryon-flux-stability", "electric-flux-stability",
                          "composite-parity-stability",
                          "anchor-score-stability"])
        # Filling them from the candidate's own certificates completes the
        # window: the battery read is one supplier, not the whole list.
        for entry in (sample,):
            entry.color_gram_determinant = 1.0
            entry.rotation_character = -1.0 + 0j
            entry.baryon_flux = 1.0
            entry.electric_flux = 1.0
            entry.composite_parity = -1
            entry.anchor_score = 0.788
        completed = obs.ParticleClusters().scale_profile([sample, sample])
        self.assertTrue(completed.stable)
        self.assertEqual(completed.failed_certificates, [])

    def test_sample_is_read_only_on_the_context(self):
        st = self._star_of_apex()
        ctx = obs.RegisterContext(st, [[0, 1, 2, 3, 4]], 1, 3,
                                  cob.ProtonSynthesis.singlet())
        before = (len(st.get_top_simplices()), len(st.get_simplices()))
        a = obs.ParticleClusters.scale_profile_sample(ctx)
        b = obs.ParticleClusters.scale_profile_sample(ctx)
        self.assertEqual(before,
                         (len(st.get_top_simplices()), len(st.get_simplices())))
        self.assertEqual(a.radius, b.radius)
        self.assertEqual(a.radial_weight_profile, b.radial_weight_profile)


class TestBaryonClassification(unittest.TestCase):
    """Design spec 16.4: the complete proton certificate and the four-way
    verdict."""

    def setUp(self):
        self.pc = obs.ParticleClusters()

    def test_certified_proton(self):
        read = self.pc.classify_baryon(_baryon_evidence())
        self.assertEqual(read.classification, "certified-proton")
        self.assertEqual(read.failed_certificates, [])
        self.assertEqual(read.confidence, 1.0)
        # every row of the spec 16.4 table
        self.assertAlmostEqual(read.color_gram_determinant, 1.0, delta=MACHINE)
        self.assertEqual(read.color_flux, 0.0)
        self.assertEqual(read.total_winding, 3)
        self.assertAlmostEqual(read.baryon_flux, 1.0, delta=MACHINE)
        self.assertEqual(read.flavor_pattern, "uud")
        self.assertAlmostEqual(read.total_isospin, 0.5, delta=MACHINE)
        self.assertAlmostEqual(read.electric_flux, 1.0, delta=1e-9)
        self.assertAlmostEqual(read.total_j2, 0.75, delta=1e-14)
        self.assertLess(abs(read.total_j2_variance), 1e-13)
        self.assertTrue(read.sharp_spin)
        self.assertLess(read.sharp_spin_right_residual, 1e-12)
        self.assertLess(read.sharp_spin_left_residual, 1e-12)
        self.assertTrue(read.odd_monopole)
        self.assertTrue(read.projective_cocycle_nontrivial)
        self.assertEqual(read.monopole_number, 1)
        # The 2pi character is recorded and gates nothing.
        self.assertEqual(read.rotation_character_sign, -1)
        self.assertLess(abs(read.rotation_character + 1.0), 1e-12)
        self.assertEqual(read.exterior_parity, -1)
        self.assertTrue(read.radius_finite)
        self.assertTrue(read.profile_stable)
        self.assertTrue(read.certificate.holds())
        self.assertEqual(read.certificate.grade,
                         cob.CertificateGrade.StructureExact)

    def test_delta_oracle_is_a_baryon_but_never_a_proton(self):
        # J^2 = 15/4 with Var = 0: an exact eigenstate that is simply not at
        # 3/4.  The proton certificate's eigen-equations are stated AT 3/4,
        # so a Delta fails both the expectation row and the sharpness row;
        # that it is sharp at its own eigenvalue is a separate statement and
        # is checked below.
        read = self.pc.classify_baryon(_baryon_evidence(spin="delta"))
        self.assertEqual(read.classification, "baryon-candidate")
        self.assertNotEqual(read.classification, "certified-proton")
        self.assertAlmostEqual(read.total_j2, 15.0 / 4.0, delta=1e-13)
        self.assertLess(abs(read.total_j2_variance), 1e-13)
        self.assertFalse(read.sharp_spin)
        self.assertEqual(sorted(read.failed_certificates),
                         ["sharp-spin", "spin-expectation"])

    def test_the_delta_state_is_sharp_at_its_own_eigenvalue(self):
        js = obs.SharpSpin.doublet_spin_matrices(3)
        state = obs.SharpSpin.determinant([0, 2, 4], 6)
        at_own = obs.SharpSpin.read(js, state, state.conj(),
                                    target_eigenvalue=15.0 / 4.0)
        self.assertTrue(at_own.sharp)

    def test_delta_dense_772_oracle_is_a_baryon_but_never_a_proton(self):
        # the #772 dense total-space oracle: |uuu> -> 15/4 (the exact
        # measuring stick), consulted when no quasi-free read certified.
        dense = obs.ExchangeHolonomy.total_j_squared(
            np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=complex))
        self.assertEqual(dense, 15.0 / 4.0)
        read = self.pc.classify_baryon(
            _baryon_evidence(spin="none", dense_j2=dense))
        self.assertEqual(read.classification, "baryon-candidate")
        self.assertEqual(read.total_j2, 15.0 / 4.0)
        self.assertIsNone(read.total_j2_variance)
        self.assertIn("spin-expectation", read.failed_certificates)
        self.assertIn("sharp-spin", read.failed_certificates)

    def test_dense_772_proton_expectation_still_needs_the_eigen_equations(
            self):
        # 2|uud> - |udu> - |duu> -> 3/4 exactly, but a DENSE expectation
        # supplies no eigen-equation residuals: expectation alone is never a
        # sharp spin.
        state = np.zeros(8, dtype=complex)
        state[0b001], state[0b010], state[0b100] = 2.0, -1.0, -1.0
        self.assertAlmostEqual(obs.ExchangeHolonomy.total_j_squared(state),
                               0.75, delta=1e-14)
        read = self.pc.classify_baryon(
            _baryon_evidence(spin="none", dense_j2=0.75))
        self.assertEqual(read.total_j2, 0.75)
        self.assertIsNone(read.total_j2_variance)
        self.assertFalse(read.sharp_spin)
        self.assertEqual(read.failed_certificates, ["sharp-spin"])
        self.assertEqual(read.classification, "baryon-candidate")

    def test_generic_slater_expectation_without_sharp_variance(self):
        # <J^2> = 3/4 EXACTLY with Var = 15/16 > 0 is NOT a certified
        # proton.
        read = self.pc.classify_baryon(_baryon_evidence(spin="generic"))
        self.assertAlmostEqual(read.total_j2, 0.75, delta=1e-13)
        self.assertAlmostEqual(read.total_j2_variance, 15.0 / 16.0,
                               delta=1e-12)
        self.assertFalse(read.sharp_spin)
        self.assertNotIn("spin-expectation", read.failed_certificates)
        self.assertIn("sharp-spin", read.failed_certificates)
        self.assertNotEqual(read.classification, "certified-proton")

    def test_a_vanishing_variance_does_not_make_a_state_sharp(self):
        """The case a variance cannot decide: the biorthogonal expectation
        is exactly 3/4 and the complex variance exactly zero, so the
        variance criterion accepts the state, while the right
        eigen-equation refuses it outright."""
        read = self.pc.classify_baryon(_baryon_evidence(spin="generic"))
        self.assertTrue(read.variance_would_accept)
        self.assertFalse(read.sharp_spin)
        self.assertGreater(read.sharp_spin_right_residual, 0.1)
        self.assertLess(read.sharp_spin_left_residual, 1e-12)
        self.assertIn("sharp-spin", read.failed_certificates)

    def test_exact_eigenstate_passes_the_sharp_certificate(self):
        # the other half of the pair: an exact J^2 eigenstate has Var = 0.
        read = self.pc.classify_baryon(_baryon_evidence(spin="sharp"))
        self.assertAlmostEqual(read.total_j2, 0.75, delta=1e-14)
        self.assertLess(abs(read.total_j2_variance), 1e-13)
        self.assertTrue(read.sharp_spin)
        self.assertEqual(read.classification, "certified-proton")

    def test_missing_variance_read_is_unknown_never_zero(self):
        read = self.pc.classify_baryon(
            _baryon_evidence(spin="expectation-only"))
        self.assertAlmostEqual(read.total_j2, 0.75, delta=1e-14)
        self.assertIsNone(read.total_j2_variance)
        self.assertFalse(read.sharp_spin)
        self.assertIn("sharp-spin", read.failed_certificates)

    def test_no_baryon_without_three_certified_quarks(self):
        quarks = list(_baryon_evidence().quarks)
        quarks[2] = self.pc.classify_quark(_certified_evidence(turns=-1))
        read = self.pc.classify_baryon(_baryon_evidence(quarks=quarks))
        self.assertEqual(read.classification, "no-baryon")
        self.assertIn("constituent-quarks", read.failed_certificates)

    def test_no_baryon_without_a_bound_supercomponent(self):
        read = self.pc.classify_baryon(
            _baryon_evidence(binding=obs.BoundSupercomponentRead()))
        self.assertEqual(read.classification, "no-baryon")
        self.assertIn("bound-supercomponent", read.failed_certificates)

    def test_no_baryon_dominates_a_full_proton_certificate(self):
        # the structural gates decide "no baryon" even when every proton
        # certificate below them holds.
        read = self.pc.classify_baryon(
            _baryon_evidence(binding=_binding(transports=False)))
        self.assertEqual(read.classification, "no-baryon")
        self.assertEqual(read.failed_certificates, ["bound-supercomponent"])

    def test_wrong_baryon_flux_is_named(self):
        quarks = list(_baryon_evidence().quarks)
        quarks[2] = self.pc.classify_quark(_certified_evidence(turns=1))
        quarks[2] = obs.QuarkRead.from_record(quarks[2].to_record())
        ev = _baryon_evidence()
        # replace one leg with an UNCERTIFIED winding: nu is unknown
        broken = _certified_evidence()
        conn = obs.FiberConnection()
        A, B = _unit_fiber(1, 3), _unit_fiber(11, 3)
        broken.winding = conn.open_segment_winding(
            [_phase_link(conn, A, B, p) for p in (0.0, 0.4, 0.8)],
            obs.WindingClosureSpec())
        legs = list(ev.quarks)
        legs[2] = self.pc.classify_quark(broken)
        read = self.pc.classify_baryon(_baryon_evidence(quarks=legs))
        self.assertIsNone(read.total_winding)
        self.assertIsNone(read.baryon_flux)   # UNKNOWN, never zero
        self.assertIn("baryon-flux-unit", read.failed_certificates)

    def test_flavor_pattern_uuu_is_not_a_proton(self):
        read = self.pc.classify_baryon(_baryon_evidence(kinds=("u", "u", "u")))
        self.assertEqual(read.flavor_pattern, "uuu")
        self.assertAlmostEqual(read.total_isospin, 1.5, delta=MACHINE)
        self.assertIn("flavor-uud", read.failed_certificates)
        self.assertIn("electric-flux-unit", read.failed_certificates)
        self.assertEqual(read.classification, "baryon-candidate")

    def test_flavor_pattern_udd_is_not_a_proton(self):
        read = self.pc.classify_baryon(_baryon_evidence(kinds=("u", "d", "d")))
        self.assertEqual(read.flavor_pattern, "udd")
        self.assertAlmostEqual(read.electric_flux, 0.0, delta=1e-9)
        self.assertIn("flavor-uud", read.failed_certificates)
        self.assertIn("electric-flux-unit", read.failed_certificates)

    def test_color_singlet_fixture_with_unknown_flavor_is_partial(self):
        # a color-singlet three-cluster candidate whose constituents have NO
        # certified doublet: flavor AND charge stay unknown and the read is
        # a partial candidate naming exactly those gaps.
        plain = self.pc.classify_quark(_certified_evidence())
        legs = [_relabel_quark(plain, level=1, tag="%02x" % (0xb0 + i))
                for i in range(3)]
        read = self.pc.classify_baryon(_baryon_evidence(quarks=legs))
        self.assertEqual(read.classification, "baryon-candidate")
        self.assertAlmostEqual(read.color_gram_determinant, 1.0, delta=MACHINE)
        self.assertEqual(read.flavor_pattern, "")
        self.assertIsNone(read.total_isospin)
        self.assertIsNone(read.electric_flux)
        self.assertEqual(sorted(read.failed_certificates),
                         ["electric-flux-unit", "flavor-uud"])
        # the certified channels are still reported
        self.assertEqual(read.total_winding, 3)
        self.assertAlmostEqual(read.baryon_flux, 1.0, delta=MACHINE)

    def test_even_composite_parity_is_named(self):
        legs = list(_baryon_evidence().quarks)
        legs[1] = self.pc.classify_quark(
            _certified_evidence(occupations=(1.0, 1.0, 0.0)))
        read = self.pc.classify_baryon(_baryon_evidence(quarks=legs))
        self.assertNotEqual(read.exterior_parity, -1)
        self.assertIn("composite-parity-odd", read.failed_certificates)

    def test_uncertified_constituent_parity_is_unknown(self):
        legs = list(_baryon_evidence().quarks)
        blank = _certified_evidence()
        blank.parity_read = qm.WickCertificateRead()
        legs[0] = self.pc.classify_quark(blank)
        read = self.pc.classify_baryon(_baryon_evidence(quarks=legs))
        self.assertEqual(read.exterior_parity, 0)
        self.assertIn("composite-parity-odd", read.failed_certificates)

    def test_the_two_pi_character_is_recorded_and_gates_nothing(self):
        """A rigid rotation leaves every band constant, so the 2pi
        character is +1 along any rigid cycle whatever the spin.  The 4pi
        cycle gives +1 here, and the verdict does not move: the character
        travels on the read and certifies nothing."""
        read = self.pc.classify_baryon(_baryon_evidence(rotation_turns=2))
        self.assertEqual(read.rotation_character_sign, +1)
        self.assertEqual(read.classification, "certified-proton")
        self.assertEqual(read.failed_certificates, [])

    def test_the_exchange_and_rotation_channels_stay_distinct(self):
        # the #772 channels are not interchangeable: an exchange-tagged
        # character leaves the rotation report UNKNOWN rather than being
        # reinterpreted as a rotation.
        ev = _baryon_evidence()
        ev.rotation = _exchange_character()
        read = self.pc.classify_baryon(ev)
        self.assertIsNone(read.rotation_character)
        self.assertEqual(read.rotation_character_sign, 0)
        self.assertEqual(read.classification, "certified-proton")

    def test_uncertified_rotation_read_never_emits_a_sign(self):
        # a TIMING mismatch voids the cancellation premise: the #772 read is
        # uncertified, so the reported character is unknown rather than a
        # sign.  It still gates nothing.
        EH = obs.ExchangeHolonomy
        frame0 = EH.transverse_spinor_frame(0, 1, 4)
        weights = np.ones(4, dtype=complex)
        loop = EH.loop_holonomy(
            EH.rotation_loop_frames(frame0, 0, 1, 4, 1, 16), weights)
        mistimed = EH.loop_holonomy(
            EH.reference_loop_frames(frame0, 8), weights)
        ev = _baryon_evidence()
        ev.rotation = EH.rotation_character(loop, mistimed)
        self.assertFalse(ev.rotation.certificate.holds())
        read = self.pc.classify_baryon(ev)
        self.assertIsNone(read.rotation_character)
        self.assertEqual(read.rotation_character_sign, 0)
        self.assertEqual(read.classification, "certified-proton")

    def test_an_even_monopole_support_is_never_a_proton(self):
        """Half-integer spin is a topological charge of the connection: at
        even monopole number the projective class is trivial, no
        symmetry-protected band is a spinor doublet, and both spin rows
        fail."""
        for monopole in (0, 2):
            read = self.pc.classify_baryon(
                _baryon_evidence(monopole=monopole))
            self.assertFalse(read.odd_monopole, msg=f"mu={monopole}")
            self.assertFalse(read.projective_cocycle_nontrivial)
            self.assertEqual(read.monopole_number, monopole)
            self.assertEqual(sorted(read.failed_certificates),
                             ["odd-monopole", "projective-cocycle"])
            self.assertEqual(read.classification, "baryon-candidate")

    def test_missing_monopole_evidence_is_named_never_assumed(self):
        read = self.pc.classify_baryon(_baryon_evidence(monopole=None))
        self.assertIsNone(read.monopole_number)
        self.assertFalse(read.odd_monopole)
        self.assertFalse(read.projective_cocycle_nontrivial)
        self.assertEqual(sorted(read.failed_certificates),
                         ["odd-monopole", "projective-cocycle"])

    def test_an_odd_monopole_support_carries_the_spinor_doublet(self):
        read = _monopole_spin_read(1)
        self.assertTrue(read.monopole.odd)
        self.assertTrue(read.cocycle.nontrivial)
        self.assertTrue(read.half_integer_doublet)

    def test_spin_lift_is_not_demanded_without_a_continuum_claim(self):
        read = self.pc.classify_baryon(_baryon_evidence())
        self.assertFalse(read.spin_lift_applicable)
        self.assertFalse(read.spin_lift_accepted)
        self.assertNotIn("spin-lift", read.failed_certificates)
        self.assertEqual(read.classification, "certified-proton")

    def test_continuum_claim_accepts_a_certified_lift(self):
        lift = _accepted_spin_lift()
        self.assertTrue(lift.lift_exists)
        read = self.pc.classify_baryon(
            _baryon_evidence(continuum=True, spin_lift=lift))
        self.assertTrue(read.spin_lift_applicable)
        self.assertTrue(read.spin_lift_accepted)
        self.assertEqual(read.classification, "certified-proton")

    def test_continuum_claim_without_a_lift_fails_by_name(self):
        read = self.pc.classify_baryon(_baryon_evidence(continuum=True))
        self.assertTrue(read.spin_lift_applicable)
        self.assertFalse(read.spin_lift_accepted)
        self.assertEqual(read.failed_certificates, ["spin-lift"])

    def test_obstructed_lift_fails_the_continuum_claim(self):
        lift = _obstructed_spin_lift()
        self.assertTrue(lift.obstructed)
        read = self.pc.classify_baryon(
            _baryon_evidence(continuum=True, spin_lift=lift))
        self.assertFalse(read.spin_lift_accepted)
        self.assertIn("spin-lift", read.failed_certificates)

    def test_missing_radius_is_not_certified(self):
        samples = _scale_samples()
        samples[1].radius = float("inf")
        read = self.pc.classify_baryon(_baryon_evidence(samples=samples))
        self.assertFalse(read.radius_finite)
        self.assertIn("finite-radius", read.failed_certificates)
        self.assertNotEqual(read.classification, "certified-proton")

    def test_unstable_profile_is_not_certified(self):
        read = self.pc.classify_baryon(
            _baryon_evidence(samples=_scale_samples(profile_drift=0.01)))
        self.assertFalse(read.profile_stable)
        self.assertAlmostEqual(read.profile_max_deviation, 0.02, delta=1e-12)
        self.assertIn("profile-stability", read.failed_certificates)
        self.assertNotIn("finite-radius", read.failed_certificates)

    def test_no_scale_evidence_fails_both_scale_gates(self):
        read = self.pc.classify_baryon(_baryon_evidence(samples=[]))
        self.assertTrue(math.isnan(read.radius))
        self.assertIn("finite-radius", read.failed_certificates)
        self.assertIn("profile-stability", read.failed_certificates)

    def test_physical_mass_is_always_unknown_on_the_read(self):
        read = self.pc.classify_baryon(_baryon_evidence())
        self.assertEqual(read.classification, "certified-proton")
        self.assertIsNone(read.physical_mass)
        self.assertAlmostEqual(read.spectral_mass, 2.25, delta=MACHINE)

    def test_confidence_is_the_passed_fraction(self):
        # Sixteen gates: the world-tube crossing conjunct, which passes
        # VACUOUSLY with no crossing evidence supplied, and the two
        # half-integer-spin rows.  Exactly one certificate (sharp-spin)
        # fails here.
        read = self.pc.classify_baryon(_baryon_evidence(spin="generic"))
        self.assertAlmostEqual(read.confidence, 15.0 / 16.0, delta=MACHINE)
        self.assertEqual(len(read.failed_certificates), 1)

    def test_thresholds_are_recorded(self):
        cfg = obs.ParticleClustersConfig()
        cfg.spin_variance_tolerance = 2.0     # a cap that tolerates anything
        read = obs.ParticleClusters(cfg).classify_baryon(
            _baryon_evidence(spin="generic"))
        self.assertEqual(read.thresholds.spin_variance_tolerance, 2.0)
        # Widening the variance cap no longer buys a sharp spin: the
        # certificate is the pair of eigen-equations, and the right one
        # still fails on this state.
        self.assertFalse(read.sharp_spin)
        self.assertIn("sharp-spin", read.failed_certificates)

    def test_reported_identities_travel(self):
        read = self.pc.classify_baryon(_baryon_evidence())
        self.assertEqual(read.persistence, 4.0)
        self.assertEqual(read.lifetime_overlap, 5.0)
        self.assertEqual(len(read.quarks), 3)
        self.assertEqual(read.bound_component.canonical_hash(),
                         _modular_hierarchy()[2].components[0]
                         .id.canonical_hash())

    def test_describe_names_the_verdict_and_gaps(self):
        read = self.pc.classify_baryon(_baryon_evidence(spin="generic"))
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
        read = self.pc.classify_baryon(
            _baryon_evidence(spin="generic", class_variances=self._class()))
        self.assertEqual(read.classification,
                         "quasi-free-sharp-spin-obstruction")
        self.assertEqual(read.failed_certificates, ["sharp-spin"])
        self.assertTrue(read.quasi_free_class_swept)
        self.assertAlmostEqual(read.class_variance_floor, 15.0 / 16.0,
                               delta=1e-12)
        self.assertAlmostEqual(read.total_j2, 0.75, delta=1e-13)

    def test_obstruction_is_reported_never_held(self):
        # a branch point mandating an explicit non-Gaussian mechanism, not a
        # held claim and not a refutation of the geometry.
        read = self.pc.classify_baryon(
            _baryon_evidence(spin="generic", class_variances=self._class()))
        self.assertFalse(read.certificate.holds())
        self.assertEqual(read.certificate.grade,
                         cob.CertificateGrade.HeuristicDiscovery)

    def test_unswept_class_is_a_plain_candidate(self):
        read = self.pc.classify_baryon(_baryon_evidence(spin="generic"))
        self.assertEqual(read.classification, "baryon-candidate")
        self.assertFalse(read.quasi_free_class_swept)
        self.assertTrue(math.isnan(read.class_variance_floor))

    def test_uncertified_class_member_is_not_a_sweep(self):
        variances = self._class() + [qm.WickCertificateRead()]
        read = self.pc.classify_baryon(
            _baryon_evidence(spin="generic", class_variances=variances))
        self.assertFalse(read.quasi_free_class_swept)
        self.assertEqual(read.classification, "baryon-candidate")

    def test_class_reaching_zero_is_not_an_obstruction(self):
        # one accepted member of the class IS an exact eigenstate: the
        # variance converges, so nothing is obstructed.
        sharp = _sharp_spin_reads()[1]
        read = self.pc.classify_baryon(
            _baryon_evidence(spin="generic",
                             class_variances=self._class() + [sharp]))
        self.assertTrue(read.quasi_free_class_swept)
        self.assertLess(read.class_variance_floor, 1e-13)
        self.assertEqual(read.classification, "baryon-candidate")

    def test_unmeasured_own_variance_is_not_an_obstruction(self):
        # the class does not converge, but THIS candidate's own Var(J^2)
        # was never measured: unknown is not an obstruction.
        read = self.pc.classify_baryon(
            _baryon_evidence(spin="expectation-only",
                             class_variances=self._class()))
        self.assertIsNone(read.total_j2_variance)
        self.assertTrue(read.quasi_free_class_swept)
        self.assertEqual(read.failed_certificates, ["sharp-spin"])
        self.assertEqual(read.classification, "baryon-candidate")

    def test_obstruction_requires_every_other_certificate(self):
        # the Delta-like case: the expectation ALSO fails, so this is a
        # plain baryon candidate, never the obstruction branch.
        read = self.pc.classify_baryon(
            _baryon_evidence(spin="delta", class_variances=self._class()))
        self.assertEqual(read.classification, "baryon-candidate")

    def test_obstruction_requires_a_bound_baryon(self):
        read = self.pc.classify_baryon(
            _baryon_evidence(spin="generic", class_variances=self._class(),
                             binding=obs.BoundSupercomponentRead()))
        self.assertEqual(read.classification, "no-baryon")

    def test_obstruction_survives_a_second_failing_certificate(self):
        read = self.pc.classify_baryon(
            _baryon_evidence(spin="generic", class_variances=self._class(),
                             flux=_polarized_flux()))
        self.assertEqual(read.classification, "baryon-candidate")
        self.assertEqual(sorted(read.failed_certificates),
                         ["color-flux-zero", "sharp-spin"])


class TestBaryonInvarianceAndReplay(unittest.TestCase):
    """Relabeling, in-band rotation, constituent permutation, and cold
    replay preserve the verdict (the shared #763 merge gates)."""

    def setUp(self):
        self.pc = obs.ParticleClusters()

    def test_relabeling_preserves_the_verdict(self):
        base = self.pc.classify_baryon(_baryon_evidence())
        legs = [_relabel_quark(q, level=1, tag="%02x" % (0xe0 + i))
                for i, q in enumerate(_baryon_evidence().quarks)]
        relabelled = self.pc.classify_baryon(_baryon_evidence(quarks=legs))
        self.assertEqual(base.classification, relabelled.classification)
        self.assertEqual(base.confidence, relabelled.confidence)
        self.assertEqual(base.total_winding, relabelled.total_winding)
        self.assertEqual(base.flavor_pattern, relabelled.flavor_pattern)
        self.assertEqual(base.color_gram_determinant,
                         relabelled.color_gram_determinant)
        self.assertNotEqual(base.quarks[0].canonical_hash(),
                            relabelled.quarks[0].canonical_hash())

    def test_in_band_su3_rotation_preserves_the_verdict(self):
        base = self.pc.classify_baryon(_baryon_evidence())
        for theta in (0.3, 1.1, 2.7):
            g = _su3_element(theta)
            rotated = self.pc.classify_baryon(
                _baryon_evidence(color=g @ _color_triad()))
            self.assertEqual(rotated.classification, base.classification)
            self.assertLess(abs(rotated.color_wedge - base.color_wedge), 1e-13)

    def test_constituent_permutation_preserves_the_verdict(self):
        ev = _baryon_evidence()
        legs = list(ev.quarks)
        permuted = _baryon_evidence(quarks=[legs[2], legs[0], legs[1]],
                                    color=_color_triad()[:, [2, 0, 1]])
        a = self.pc.classify_baryon(ev)
        b = self.pc.classify_baryon(permuted)
        self.assertEqual(a.classification, b.classification)
        # the flavor PATTERN is canonical: a permutation cannot change it
        self.assertEqual(a.flavor_pattern, b.flavor_pattern)
        self.assertEqual(a.total_winding, b.total_winding)
        self.assertEqual(a.exterior_parity, b.exterior_parity)
        # an EVEN color-column permutation leaves even the wedge alone
        self.assertLess(abs(a.color_wedge - b.color_wedge), 1e-13)

    def test_cold_replay_is_deterministic(self):
        first = self.pc.classify_baryon(_baryon_evidence()).to_record()
        for _ in range(3):
            again = obs.ParticleClusters().classify_baryon(
                _baryon_evidence()).to_record()
            self.assertEqual(
                obs.ObservableGates.report_delta(first, again), 0.0)
            self.assertEqual(first["classification"],
                             again["classification"])
            self.assertEqual(first["failed_certificates"],
                             again["failed_certificates"])

    def test_record_roundtrip_is_exact(self):
        read = self.pc.classify_baryon(_baryon_evidence())
        back = obs.BaryonRead.from_record(read.to_record())
        self.assertEqual(back.classification, read.classification)
        self.assertEqual(back.color_gram_determinant, read.color_gram_determinant)
        self.assertEqual(back.color_wedge, read.color_wedge)
        self.assertEqual(back.total_winding, read.total_winding)
        self.assertEqual(back.baryon_flux, read.baryon_flux)
        self.assertEqual(back.electric_flux, read.electric_flux)
        self.assertEqual(back.total_j2, read.total_j2)
        self.assertEqual(back.total_j2_variance, read.total_j2_variance)
        self.assertEqual(back.rotation_character, read.rotation_character)
        self.assertEqual(back.monopole_number, read.monopole_number)
        self.assertEqual(back.odd_monopole, read.odd_monopole)
        self.assertEqual(back.projective_cocycle_nontrivial,
                         read.projective_cocycle_nontrivial)
        self.assertEqual(back.sharp_spin_right_residual,
                         read.sharp_spin_right_residual)
        self.assertEqual(back.sharp_spin_left_residual,
                         read.sharp_spin_left_residual)
        self.assertEqual(back.variance_would_accept, read.variance_would_accept)
        self.assertEqual(back.flavor_pattern, read.flavor_pattern)
        self.assertEqual(back.confidence, read.confidence)
        self.assertEqual(back.failed_certificates, read.failed_certificates)
        self.assertEqual(back.thresholds.spin_variance_tolerance,
                         read.thresholds.spin_variance_tolerance)
        self.assertEqual(back.certificate.holds(), read.certificate.holds())
        self.assertEqual(obs.ObservableGates.report_delta(
            read.to_record(), back.to_record()), 0.0)

    def test_record_null_semantics(self):
        # unknown values serialize as null, never as zero.
        EH = obs.ExchangeHolonomy
        frame0 = EH.transverse_spinor_frame(0, 1, 4)
        weights = np.ones(4, dtype=complex)
        ev = _baryon_evidence(spin="none", monopole=None, quarks=[
            self.pc.classify_quark(_certified_evidence())] * 3)
        ev.rotation = EH.rotation_character(
            EH.loop_holonomy(EH.rotation_loop_frames(frame0, 0, 1, 4, 1, 16),
                            weights),
            EH.loop_holonomy(EH.reference_loop_frames(frame0, 8), weights))
        read = self.pc.classify_baryon(ev)
        record = read.to_record()
        for key in ("total_j2", "total_j2_variance", "electric_flux",
                    "physical_mass", "rotation_character_re",
                    "rotation_character_im", "total_isospin",
                    "monopole_number"):
            self.assertIsNone(record[key], key)
        back = obs.BaryonRead.from_record(record)
        self.assertIsNone(back.total_j2)
        self.assertIsNone(back.total_j2_variance)
        self.assertIsNone(back.physical_mass)
        self.assertIsNone(back.rotation_character)
        self.assertIsNone(back.monopole_number)

    def test_from_record_rejects_unknown_schema(self):
        record = self.pc.classify_baryon(_baryon_evidence()).to_record()
        record["schema_version"] = 99
        with self.assertRaises(ValueError):
            obs.BaryonRead.from_record(record)

    def test_from_record_rejects_a_foreign_record_type(self):
        record = self.pc.classify_quark(_certified_evidence()).to_record()
        with self.assertRaises(ValueError):
            obs.BaryonRead.from_record(record)

    def test_verdict_surface_is_stable_for_wave_four(self):
        # #776/#777/#778 consume the verdict unchanged, serialized through
        # the existing Record convention and stable under replay.
        record = self.pc.classify_baryon(_baryon_evidence()).to_record()
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
        for name, value in (("color_gram_tolerance", 0.5),
                            ("color_flux_tolerance", 0.5),
                            ("spin_expectation_tolerance", 0.5),
                            ("spin_variance_tolerance", 0.5),
                            ("min_support_containment", 0.5),
                            ("min_lifetime_overlap", 3.0),
                            ("min_radius", 0.5),
                            ("max_profile_deviation", 0.5)):
            cfg = obs.ParticleClustersConfig()
            setattr(cfg, name, value)
            self.assertNotEqual(base.evidence_fingerprint(ev),
                                obs.ParticleClusters(cfg)
                                .evidence_fingerprint(ev), name)

    def test_old_threshold_records_still_rehydrate(self):
        # pre-#775 checkpoints lack the new threshold keys: the reader falls
        # back to the defaults instead of rejecting.
        pc = obs.ParticleClusters()
        record = pc.classify_quark(_certified_evidence()).to_record()
        keys = ("color_gram_tolerance", "color_flux_tolerance",
                "spin_expectation_tolerance", "spin_variance_tolerance",
                "min_support_containment", "min_lifetime_overlap",
                "min_radius", "max_profile_deviation")
        for key in keys:
            self.assertIn(key, record["thresholds"])
            del record["thresholds"][key]
        back = obs.QuarkRead.from_record(record)
        defaults = obs.ParticleClustersConfig()
        self.assertEqual(back.thresholds.color_gram_tolerance,
                         defaults.color_gram_tolerance)
        self.assertEqual(back.thresholds.max_profile_deviation,
                         defaults.max_profile_deviation)

    def test_cached_color_flux_read_gives_an_identical_verdict(self):
        # the cached-versus-cold merge gate on the one CACHED read the
        # baryon certificate consumes (#764 AnalyticCache contract).
        pc = obs.ParticleClusters()
        cache = cob.AnalyticCache(_from_simplices(7, _TETRA_CHAIN,
                                                  timelike=False))
        state = qm.CovarianceState(np.eye(3, dtype=complex))
        cold = pc.octet_bilinear_read(state, [0, 1, 2])
        warm = pc.octet_bilinear_read_cached(cache, [1, 2, 3], state, [0, 1, 2])
        cached = pc.octet_bilinear_read_cached(cache, [1, 2, 3], state, [0, 1, 2])
        self.assertGreaterEqual(cache.hits, 1)
        self.assertEqual(obs.ObservableGates.report_delta(
            cold.to_record(), warm.to_record()), 0.0)
        self.assertEqual(obs.ObservableGates.report_delta(
            cold.to_record(), cached.to_record()), 0.0)
        a = pc.classify_baryon(_baryon_evidence(flux=cold))
        b = pc.classify_baryon(_baryon_evidence(flux=cached))
        self.assertEqual(obs.ObservableGates.report_delta(
            a.to_record(), b.to_record()), 0.0)
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
            pc.classify_baryon(evidence)
        baryon = (time.perf_counter() - t0) / n
        t0 = time.perf_counter()
        for _ in range(n):
            pc.scale_profile(samples)
        scale = (time.perf_counter() - t0) / n
        t0 = time.perf_counter()
        for _ in range(n):
            pc.bound_supercomponent_search(components, candidates)
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
        read = self.pc.classify_baryon(
            _baryon_evidence(exchange=_exchange_character()))
        self.assertLess(abs(read.exchange_character + 1.0), 1e-12)
        self.assertLess(abs(read.rotation_character + 1.0), 1e-12)
        self.assertLess(abs(read.spin_statistics_ratio - 1.0), 1e-12)
        self.assertEqual(read.classification, "certified-proton")

    def test_exchange_channel_never_gates(self):
        # a DOUBLE exchange (chi_hat = +1) leaves the verdict untouched:
        # the channel is reported, never a certificate.
        doubled = _exchange_character(steps=16, distance=8)
        self.assertLess(abs(doubled.character - 1.0), 1e-12)
        read = self.pc.classify_baryon(_baryon_evidence(exchange=doubled))
        self.assertEqual(read.classification, "certified-proton")
        self.assertEqual(read.failed_certificates, [])
        self.assertLess(abs(read.spin_statistics_ratio + 1.0), 1e-12)

    def test_absent_exchange_read_is_unknown(self):
        read = self.pc.classify_baryon(_baryon_evidence())
        self.assertIsNone(read.exchange_character)
        self.assertIsNone(read.spin_statistics_ratio)
        self.assertEqual(read.classification, "certified-proton")

    def test_mislabeled_channel_is_refused_not_reinterpreted(self):
        # a ROTATION-tagged read offered as the exchange channel is
        # ignored: the ratio stays unknown and nothing throws.
        read = self.pc.classify_baryon(
            _baryon_evidence(exchange=_rotation_character()))
        self.assertIsNone(read.exchange_character)
        self.assertIsNone(read.spin_statistics_ratio)

    def test_ratio_needs_both_certified_channels(self):
        read = self.pc.classify_baryon(
            _baryon_evidence(rotation_turns=2,
                             exchange=_exchange_character()))
        # the 4pi rotation IS certified, so the ratio is still reported --
        # and, the 2pi character gating nothing, the verdict is untouched by
        # its value.
        self.assertIsNotNone(read.spin_statistics_ratio)
        self.assertLess(abs(read.spin_statistics_ratio + 1.0), 1e-12)
        self.assertEqual(read.classification, "certified-proton")

    def test_exchange_channels_serialize(self):
        read = self.pc.classify_baryon(
            _baryon_evidence(exchange=_exchange_character()))
        record = read.to_record()
        self.assertAlmostEqual(record["exchange_character_re"], -1.0,
                               delta=1e-12)
        self.assertAlmostEqual(record["spin_statistics_ratio_re"], 1.0,
                               delta=1e-12)
        back = obs.BaryonRead.from_record(record)
        self.assertEqual(back.exchange_character, read.exchange_character)
        self.assertEqual(back.spin_statistics_ratio, read.spin_statistics_ratio)
        blank = self.pc.classify_baryon(_baryon_evidence()).to_record()
        self.assertIsNone(blank["exchange_character_re"])
        self.assertIsNone(blank["spin_statistics_ratio_im"])


class TestBindingCoherence(unittest.TestCase):
    """The whitepaper's "one persistent bound supercluster CONTAINING THEM":
    the binding read's contained-candidate set must be exactly the three
    constituents' label-free identities."""

    def setUp(self):
        self.pc = obs.ParticleClusters()

    def test_coherent_binding_certifies(self):
        read = self.pc.classify_baryon(_baryon_evidence())
        self.assertEqual(read.classification, "certified-proton")
        self.assertEqual(
            sorted(q.canonical_hash()
                   for q in _baryon_evidence().binding.quarks),
            sorted(q.canonical_hash() for q in read.quarks))

    def test_binding_for_other_components_is_refused(self):
        # a CERTIFIED binding read of three DIFFERENT constituents is not
        # this candidate's supercomponent: the gate refuses rather than
        # accepting an incoherent bundle.
        strangers = [_relabel_quark(_ud_quark(k), level=1,
                                    tag="%02x" % (0xf0 + i))
                     for i, k in enumerate(("u", "u", "d"))]
        foreign = _binding(quarks=strangers)
        self.assertTrue(foreign.found)
        read = self.pc.classify_baryon(_baryon_evidence(binding=foreign))
        self.assertEqual(read.classification, "no-baryon")
        self.assertEqual(read.failed_certificates, ["bound-supercomponent"])

    def test_binding_order_does_not_matter(self):
        # the comparison is an order-insensitive SET statement.
        ev = _baryon_evidence()
        legs = list(ev.quarks)
        permuted = _baryon_evidence(
            quarks=[legs[2], legs[0], legs[1]],
            color=_color_triad()[:, [2, 0, 1]], binding=ev.binding)
        read = self.pc.classify_baryon(permuted)
        self.assertEqual(read.classification, "certified-proton")


class TestClassifyBoundSupercomponents(unittest.TestCase):
    """#802 — the composition the #776 analysis overlay runs.

    ``classify_bound_supercomponents`` is what turns a §16.2 search result
    into §16.4 verdicts: every binding that grouped EXACTLY three certified
    constituents is classified, nothing is padded, and every quantity the
    caller did not supply stays ABSENT so ``classify_baryon`` names it.
    """

    def setUp(self):
        self.pc = obs.ParticleClusters()
        self.candidates = _bound_candidates()
        self.constituents = [c.quark for c in self.candidates]
        _groups, _fine, self.coarse = _modular_hierarchy()
        self.bindings = list(self.pc.bound_supercomponent_search(
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
        reads = self.pc.classify_bound_supercomponents(
            self.bindings, self.constituents)
        self.assertEqual(len(reads), 1)
        read = reads[0]
        self.assertEqual(read.bound_component.canonical_hash(),
                         self.bindings[0].bound_component.canonical_hash())
        self.assertEqual(
            sorted(q.canonical_hash() for q in read.quarks),
            sorted(q.canonical_hash() for q in self.bindings[0].quarks))

    def test_both_structural_gates_hold_so_the_verdict_is_not_no_baryon(self):
        read = self.pc.classify_bound_supercomponents(
            self.bindings, self.constituents)[0]
        self.assertNotIn("constituent-quarks", read.failed_certificates)
        self.assertNotIn("bound-supercomponent", read.failed_certificates)
        self.assertEqual(read.classification, "baryon-candidate")

    def test_two_certified_constituents_emit_nothing(self):
        """A three-cluster verdict is NEVER assembled by padding the
        missing legs: a padded leg would report a structural gap the
        geometry did not have."""
        candidates = _bound_candidates(kinds=("u", "d"))
        bindings = list(self.pc.bound_supercomponent_search(
            self.coarse.components, candidates))
        self.assertEqual(len(bindings), 1)
        self.assertEqual(len(bindings[0].quark_indices), 2)
        self.assertEqual(
            self.pc.classify_bound_supercomponents(
                bindings, [c.quark for c in candidates]), [])

    def test_an_uncertified_candidate_is_never_a_constituent(self):
        """The search counts only CERTIFIED quark candidates, so three
        candidates of which one is uncertified never reach the classifier."""
        candidates = _bound_candidates()
        candidates[2].quark = obs.QuarkRead()     # classification "none"
        bindings = list(self.pc.bound_supercomponent_search(
            self.coarse.components, candidates))
        self.assertEqual(len(bindings[0].quark_indices), 2)
        self.assertEqual(
            self.pc.classify_bound_supercomponents(
                bindings, [c.quark for c in candidates]), [])

    def test_no_binding_emits_no_verdict(self):
        self.assertEqual(
            self.pc.classify_bound_supercomponents([], self.constituents), [])

    def test_one_read_per_qualifying_binding_in_binding_order(self):
        doubled = self.bindings + self.bindings
        reads = self.pc.classify_bound_supercomponents(
            doubled, self.constituents)
        self.assertEqual(len(reads), 2)
        self.assertEqual(reads[0].bound_component.canonical_hash(),
                         reads[1].bound_component.canonical_hash())
        self.assertEqual(reads[0].classification, reads[1].classification)

    # ---- the delegation is exactly classifyBaryon ---------------------

    def test_it_is_classify_baryon_on_the_same_bundle(self):
        evidence = obs.BaryonCandidateEvidence()
        evidence.bound_component = self.bindings[0].bound_component
        evidence.binding = self.bindings[0]
        evidence.quarks = [self.constituents[i]
                           for i in self.bindings[0].quark_indices]
        evidence.persistence_lifetime = 4.0
        direct = self.pc.classify_baryon(evidence)
        composed = self.pc.classify_bound_supercomponents(
            self.bindings, self.constituents, [4.0])[0]
        self.assertEqual(composed.classification, direct.classification)
        self.assertEqual(composed.failed_certificates,
                         direct.failed_certificates)
        self.assertEqual(composed.confidence, direct.confidence)
        self.assertEqual(composed.persistence, direct.persistence)
        self.assertRecordsEqual(composed.to_record(), direct.to_record())

    # ---- unsupplied evidence is NAMED, never presumed ----------------

    def test_every_unsupplied_certificate_is_named(self):
        """Exactly the COMPOSITE-level evidence is missing, and each gap is
        named. The constituent-derived rows (winding, parity, flavor,
        charge) are supplied by the #773 verdicts themselves and hold."""
        read = self.pc.classify_bound_supercomponents(
            self.bindings, self.constituents)[0]
        self.assertEqual(
            read.failed_certificates,
            ["color-singlet", "color-flux-zero", "spin-expectation",
             "sharp-spin", "odd-monopole", "projective-cocycle",
             "finite-radius", "profile-stability"])
        # The crossing-readouts gate is not applicable here -- no crossing
        # evidence travels through classifyBoundSupercomponents -- so it
        # passes vacuously and does not appear among the failures.
        self.assertEqual(read.confidence, 8.0 / 16.0)

    def test_the_constituent_derived_rows_hold_on_certified_legs(self):
        read = self.pc.classify_bound_supercomponents(
            self.bindings, self.constituents)[0]
        for name in ("baryon-flux-unit", "composite-parity-odd",
                     "flavor-uud", "electric-flux-unit"):
            self.assertNotIn(name, read.failed_certificates)
        self.assertEqual(read.total_winding, 3)
        self.assertEqual(read.baryon_flux, 1.0)
        self.assertEqual(read.exterior_parity, -1)
        self.assertEqual(read.flavor_pattern, "uud")

    def test_unknown_values_are_null_or_nan_never_zero(self):
        read = self.pc.classify_bound_supercomponents(
            self.bindings, self.constituents)[0]
        self.assertTrue(math.isnan(read.color_gram_determinant))
        self.assertTrue(math.isnan(read.color_wedge.real))
        self.assertTrue(math.isnan(read.color_flux))
        self.assertIsNone(read.total_j2)
        self.assertIsNone(read.total_j2_variance)
        self.assertIsNone(read.rotation_character)
        self.assertIsNone(read.monopole_number)
        self.assertIsNone(read.exchange_character)
        self.assertIsNone(read.physical_mass)
        self.assertTrue(math.isnan(read.sharp_spin_right_residual))
        self.assertTrue(math.isnan(read.sharp_spin_left_residual))
        self.assertTrue(math.isnan(read.class_variance_floor))
        self.assertFalse(read.quasi_free_class_swept)
        self.assertTrue(math.isnan(read.radius))
        self.assertTrue(math.isnan(read.transport_leakage_max))

    def test_a_continuum_spin_claim_is_never_declared_here(self):
        """The composition makes no continuum claim, so the SO(d)->Spin(d)
        lift is not demanded and `spin-lift` is not a failure."""
        read = self.pc.classify_bound_supercomponents(
            self.bindings, self.constituents)[0]
        self.assertFalse(read.spin_lift_applicable)
        self.assertNotIn("spin-lift", read.failed_certificates)

    # ---- the bound component's lifetime -------------------------------

    def test_an_unsupplied_lifetime_is_nan(self):
        read = self.pc.classify_bound_supercomponents(
            self.bindings, self.constituents)[0]
        self.assertTrue(math.isnan(read.persistence))

    def test_a_supplied_lifetime_travels_verbatim(self):
        read = self.pc.classify_bound_supercomponents(
            self.bindings, self.constituents, [7.0])[0]
        self.assertEqual(read.persistence, 7.0)

    def test_a_mismatched_lifetime_list_is_refused(self):
        with self.assertRaises(ValueError):
            self.pc.classify_bound_supercomponents(
                self.bindings, self.constituents, [1.0, 2.0])

    def test_a_constituent_index_outside_the_read_list_is_refused(self):
        with self.assertRaises(ValueError):
            self.pc.classify_bound_supercomponents(
                self.bindings, self.constituents[:2])

    # ---- ordering / relabeling ----------------------------------------

    def test_the_verdict_is_insensitive_to_constituent_order(self):
        order = [2, 0, 1]
        candidates = [self.candidates[i] for i in order]
        bindings = list(self.pc.bound_supercomponent_search(
            self.coarse.components, candidates))
        permuted = self.pc.classify_bound_supercomponents(
            bindings, [c.quark for c in candidates])[0]
        base = self.pc.classify_bound_supercomponents(
            self.bindings, self.constituents)[0]
        self.assertEqual(permuted.classification, base.classification)
        self.assertEqual(permuted.failed_certificates, base.failed_certificates)
        self.assertEqual(
            sorted(q.canonical_hash() for q in permuted.quarks),
            sorted(q.canonical_hash() for q in base.quarks))

    # ---- the record schema round-trips with its nulls ------------------

    def test_the_record_round_trips_with_nulls_for_unknowns(self):
        read = self.pc.classify_bound_supercomponents(
            self.bindings, self.constituents, [4.0])[0]
        record = read.to_record()
        # Every UNSUPPLIED optional serializes as null, never as zero.
        for key in ("total_j2", "total_j2_variance", "physical_mass",
                    "rotation_character_re", "rotation_character_im",
                    "monopole_number",
                    "exchange_character_re", "exchange_character_im",
                    "spin_statistics_ratio_re", "spin_statistics_ratio_im"):
            self.assertIsNone(record[key], f"{key} is not null")
        # Every UNMEASURED double serializes as NaN, never as zero.
        for key in ("color_gram_determinant", "color_flux", "color_wedge_re",
                    "color_wedge_im", "class_variance_floor", "radius",
                    "radius_ratio", "spectral_mass", "profile_max_deviation",
                    "transport_leakage_max", "sharp_spin_right_residual",
                    "sharp_spin_left_residual"):
            self.assertTrue(math.isnan(record[key]), f"{key} is not NaN")
        rehydrated = obs.BaryonRead.from_record(record)
        self.assertEqual(rehydrated.classification, read.classification)
        self.assertEqual(rehydrated.failed_certificates,
                         read.failed_certificates)
        self.assertIsNone(rehydrated.total_j2)
        self.assertIsNone(rehydrated.total_j2_variance)
        self.assertIsNone(rehydrated.physical_mass)
        self.assertTrue(math.isnan(rehydrated.color_gram_determinant))
        self.assertEqual(rehydrated.persistence, 4.0)
        self.assertRecordsEqual(rehydrated.to_record(), record)

    def test_an_unknown_record_schema_version_is_rejected(self):
        record = self.pc.classify_bound_supercomponents(
            self.bindings, self.constituents)[0].to_record()
        record["schema_version"] = record["schema_version"] + 97
        with self.assertRaises(ValueError):
            obs.BaryonRead.from_record(record)

    # ---- the three-outcome vocabulary is reachable through here -------

    def test_a_fully_evidenced_bundle_certifies_a_proton(self):
        """The same three constituents, given the rest of the evidence,
        reach `certified-proton` — so the composition's `baryon-candidate`
        is a statement about the MISSING evidence, not a ceiling."""
        read = self.pc.classify_baryon(_baryon_evidence())
        self.assertEqual(read.classification, "certified-proton")
        self.assertEqual(read.failed_certificates, [])
        self.assertEqual(read.confidence, 1.0)

    def test_the_obstruction_verdict_is_reachable(self):
        read = self.pc.classify_baryon(_baryon_evidence(
            spin="generic",
            class_variances=[_generic_slater_spin_reads()[1]] * 3))
        self.assertEqual(read.classification,
                         "quasi-free-sharp-spin-obstruction")
        self.assertEqual(read.failed_certificates, ["sharp-spin"])
        self.assertTrue(read.quasi_free_class_swept)
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
        return self.pc.classify_quark(ev)

    def test_a_stable_candidate_certifies(self):
        read = self._read(_certified_evidence())
        self.assertEqual(read.classification, "quark")
        self.assertEqual(read.stability_frames, 3)
        self.assertEqual(read.anchor_score_spread, 0.0)
        self.assertEqual(read.anchor_coherence_spread, 0.0)
        self.assertAlmostEqual(read.band_continuation_overlap, 1.0,
                               delta=MACHINE)

    def test_a_band_that_loses_rank_three_at_the_next_frame_is_rejected(self):
        # Stable at frame 0, rank two at frame 1: the condition holds at ONE
        # frame and fails across them, which is exactly what "stable" is
        # supposed to catch.
        ev = _certified_evidence()
        first = ev.color_band_frames[0]
        ev.color_band_frames = [first, _unit_fiber(1, 2)]
        read = self._read(ev)
        self.assertEqual(read.classification, "none")
        self.assertIn("color-rank-stability", read.failed_certificates)
        self.assertNotIn("color-rank-three", read.failed_certificates)

    def test_a_band_uncertified_at_the_next_frame_is_rejected(self):
        ev = _certified_evidence()
        first = ev.color_band_frames[0]
        ev.color_band_frames = [first, _unit_fiber(1, 3, accepted=False)]
        read = self._read(ev)
        self.assertEqual(read.classification, "none")
        self.assertIn("color-rank-stability", read.failed_certificates)

    def test_frames_must_be_certified_continuations(self):
        # Two accepted rank-three bands on DISJOINT cells are two different
        # bands, not one stable band: the continuation is never certified.
        ev = _certified_evidence()
        first = ev.color_band_frames[0]
        ev.color_band_frames = [first, _unit_fiber(500, 3)]
        read = self._read(ev)
        self.assertEqual(read.classification, "none")
        self.assertIn("color-rank-stability", read.failed_certificates)
        self.assertEqual(read.band_continuation_overlap, 0.0)

    def test_one_frame_never_establishes_stability(self):
        ev = _certified_evidence()
        ev.color_band_frames = [ev.color_band_frames[0]]
        ev.anchor_frames = [ev.anchor_frames[0]]
        read = self._read(ev)
        self.assertEqual(read.classification, "none")
        self.assertIn("color-rank-stability", read.failed_certificates)
        self.assertIn("anchor-stability", read.failed_certificates)
        # the single-frame conditions themselves still pass -- it is the
        # ACROSS-FRAME comparison that is missing, and it is named
        self.assertNotIn("color-rank-three", read.failed_certificates)
        self.assertNotIn("anchor", read.failed_certificates)
        self.assertEqual(read.stability_frames, 1)
        self.assertTrue(math.isnan(read.anchor_score_spread))

    def test_missing_stability_windows_fail_by_name(self):
        ev = _certified_evidence()
        ev.color_band_frames = []
        ev.anchor_frames = []
        read = self._read(ev)
        self.assertEqual(read.stability_frames, 0)
        self.assertIn("color-rank-stability", read.failed_certificates)
        self.assertIn("anchor-stability", read.failed_certificates)

    def test_an_anchor_that_decays_at_the_next_frame_is_rejected(self):
        # Same atlas, a frame where the anchor score falls below the floor:
        # the profile is not stable, and the spread is reported.
        ev = _certified_evidence()
        good = ev.anchor_frames[0]
        weak = _anchor_profile(terms=(0.4, 0.02))
        self.assertLess(weak.score, obs.ParticleClustersConfig().min_anchor_score)
        ev.anchor_frames = [good, weak]
        read = self._read(ev)
        self.assertEqual(read.classification, "none")
        self.assertIn("anchor-stability", read.failed_certificates)
        self.assertNotIn("anchor", read.failed_certificates)
        self.assertGreater(read.anchor_score_spread, 0.3)

    def test_stability_frames_are_configurable(self):
        cfg = obs.ParticleClustersConfig()
        cfg.min_stability_frames = 4          # more frames than supplied
        read = obs.ParticleClusters(cfg).classify_quark(_certified_evidence())
        self.assertIn("color-rank-stability", read.failed_certificates)
        self.assertIn("anchor-stability", read.failed_certificates)


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
        ev.anchor_frames = [disjoint, disjoint, disjoint]
        read = self.pc.classify_quark(ev)
        # the SCORE clears its floor and the certificate still fails: the
        # missing quantity is the overlap coherence, and it is named
        self.assertGreater(read.triangle_anchor_score,
                           read.thresholds.min_anchor_score)
        self.assertTrue(math.isnan(read.anchor_phase_coherence))
        self.assertEqual(read.classification, "none")
        self.assertIn("anchor", read.failed_certificates)
        self.assertIn("anchor-stability", read.failed_certificates)

    def test_an_overlapping_atlas_carries_the_coherence(self):
        overlapping = _anchor_profile()
        self.assertEqual(overlapping.overlapping_triangles, 2)
        self.assertEqual(overlapping.overlap_relation, "shared-edge")
        read = self.pc.classify_quark(_certified_evidence(anchor=overlapping))
        self.assertGreater(read.anchor_phase_coherence,
                           read.thresholds.min_phase_coherence)
        self.assertEqual(read.classification, "quark")


