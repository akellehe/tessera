# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Acceptance tests for derived U(r) fiber transport, Wilson observables,
and center-aware rank-three holonomies
(:class:`tessera.observables.FiberConnection`), ticket #770 / design spec
sections 5.5, 5.11, 6.6, and 12 (Algorithm E).

Covers every ticket acceptance bullet:

* independent random local U(r) frame changes act bifundamentally
  (V -> g_A^dag V g_B); only a closed holonomy is conjugated at its base
  component, so its normalized trace is invariant;
* identity transport returns the identity holonomy; reversing a link
  returns the adjoint/inverse within certificate;
* all three determinant cube-root branches agree on projective/adjoint
  reads and expose the expected distinct Z3 center lifts;
* a known center loop returns the analytic Z3 sector when an explicit
  fundamental lift is requested;
* closed determinant winding is integer and orientation-odd on a gapped
  full-rank fixture;
* an open-segment fixture yields the same integer relative winding under
  both declared closures (matched-reference and boundary-register
  trivialization) and returns unknown with no declared closure;
* a deliberately leaking/ill-conditioned map is REJECTED before polar
  reduction even though its polar factor exists (verified in numpy);
* non-normal and Krein-signature fixtures take the certified general
  path (GL(r,C) retained; pseudo-unitary reduction only on matching
  signatures, retaining inertia); cold and cached products agree, and a
  published TouchedStar invalidates only the loops touching the star.

Analytic fixtures are built through the REAL pipeline: SpectralFiber
records with identity/weighted frames realize any prescribed overlap
exactly, so every expected holonomy is closed form.
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

MACHINE = 1e-12   # closed-form fixtures (double round-off)
SOLVER = 1e-9     # certified-numerical agreement
TWO_PI = 2.0 * math.pi
NAN = float("nan")
OMEGA = complex(-0.5, 0.5 * math.sqrt(3.0))  # the algebraic cube root


def _anchored_profile():
    """An anchor profile that PASSES the acceptance predicate.

    Two distinct oriented faces sharing the boundary edge rows {0, 1}, so the
    determinant-phase coherence is measured rather than unknown: an atlas of
    one isolated face has no overlap partner and reports NaN coherence, which
    the predicate correctly refuses. The band concentrates 80/20 between the
    two faces under a declared convex weighting, giving a score comfortably
    above the floor and one common determinant-line trivialization.
    """
    weights = np.array([1.0, 1.0, 1.0, 1.0])
    band = np.zeros((4, 3), dtype=complex)
    band[0, 0] = 1.0
    band[1, 1] = 1.0
    band[2, 2] = np.sqrt(0.8)
    band[3, 2] = np.sqrt(0.2)
    anchor = obs.ColorAnchor([obs.OrientedTriangle([0, 1, 2], [1, 1, 1]),
                              obs.OrientedTriangle([0, 1, 3], [1, 1, 1])])
    anchor.declare_weights([0.75, 0.25])
    return anchor.evaluate(band, weights)


def _open_anchor_gate():
    """An ACCEPTED triangle-anchor gate.

    The exactness contract gates the colour-specific kernels on this
    certificate, so every lift/projective fixture below has to carry one.
    """
    return obs.ColorAnchor.gate_for(_anchored_profile())


GATE = _open_anchor_gate()


# --------------------------------------------------------------------------- #
# fiber fixtures (through the sanctioned record-rehydration route)
# --------------------------------------------------------------------------- #
def _cert_record(regime):
    return {"grade": "certified-numerical", "domain": "band-window",
            "regime": regime, "residual": 1e-15, "conditioning": 1.0,
            "dense_reference_error": NAN, "tolerance": 1e-9}


def _split(name, values, record):
    arr = np.asarray(values, dtype=complex).reshape(-1)
    record[name + "_re"] = [float(v.real) for v in arr]
    record[name + "_im"] = [float(v.imag) for v in arr]


def _fiber(cells, right, left=None, weights=None, *, degree=1, accepted=True,
           regime="positive-semidefinite", pos=None, neg=0, lower_gap=1.0,
           upper_gap=1.0, cond=1.0, frame_cond=1.0, self_adjoint=True,
           gram_defect=0.0):
    """Rehydrate a SpectralFiber from its record (the #769 replay route)."""
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
            "lower_gap": float(lower_gap), "upper_gap": float(upper_gap),
            "nearest_discarded_separation": min(float(lower_gap),
                                                float(upper_gap)),
            "localization": NAN, "localization_support_fraction": NAN,
            "localization_excess": NAN, "projector_residual": 1e-16,
            "eigen_residual": 1e-16, "left_residual": 1e-16,
            "gram_defect": float(gram_defect),
            # #808: the projector norm and the FRAME condition number are
            # separate quantities; this synthetic frame is orthonormal, so
            # its Riesz conditioning is exactly 1.
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
    return obs.SpectralFiber.from_record(record)


def _unit_fiber(base_id, r, **kw):
    """Rank-r fiber with identity frame on r synthetic cells: transport with
    transfer T returns rawMap == T exactly."""
    return _fiber([[base_id + i] for i in range(r)], np.eye(r), **kw)


def _random_unitary(rng, n):
    z = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    q, r = np.linalg.qr(z)
    return q * (np.diag(r) / np.abs(np.diag(r)))


def _rotation(theta):
    return np.array([[math.cos(theta), -math.sin(theta)],
                     [math.sin(theta), math.cos(theta)]], dtype=complex)


def _phase_link(conn, A, B, phi, su_part=None):
    """Accepted rank-3 transport whose determinant phase is exactly phi."""
    v = np.diag([cmath.exp(1j * phi), 1.0, 1.0]).astype(complex)
    if su_part is not None:
        v = v @ su_part
    return conn.transport(A, B, v)


# --------------------------------------------------------------------------- #
# spacetime fixtures (shared explicit-complex idiom)
# --------------------------------------------------------------------------- #
def _from_simplices(num_vertices, simplices, ids=None):
    sig = tessera.Signature(4, tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    st = tessera.Spacetime(metric, tessera.HERMITIAN_WEIGHTED, 1.0, 1.0,
                           tessera.PREFERRED, tessera.Toroid())
    ids = list(range(num_vertices)) if ids is None else ids
    verts = [st.create_vertex(i) for i in ids]
    for simplex in simplices:
        st.create_simplex([verts[i] for i in simplex])
    for e in st.get_edge_list().to_vector():
        e.set_length(1.0 + 0j)
        e.set_phase(0.0)
    return st


def _set_phase(st, a, b, phi):
    for e in st.get_edge_list().to_vector():
        if {e.get_source().get_id(), e.get_target().get_id()} == {a, b}:
            e.set_phase(phi if e.get_source().get_id() == a else -phi)
            return
    raise AssertionError(f"edge ({a},{b}) not found")


def _tracker_fiber(st, support, degree, band=0):
    # These end-to-end fixtures are symmetric complexes whose bands are
    # perfectly delocalized (#808): the subject here is the TRANSPORT, so
    # the analysis declares the permissive localization cap, which accepts
    # any MEASURED localization.
    cfg = obs.SpectralFiberConfig()
    cfg.max_localization_excess = 1.0
    read = obs.SpectralFiberTracker(st, cfg).enumerate_bands(support, degree)
    return read.fibers[band]


# =========================================================================== #
# chain-transfer sources
# =========================================================================== #
class TestChainTransferSources(unittest.TestCase):
    def test_degree0_block_matches_the_connection_laplacian(self):
        # chainTransfer reads the U(1) CONNECTION operator at degree zero
        # (#805): its identity is the oriented link entry -l^2 e^{i phi}, which
        # is what the Wilson-loop machinery it is compared against carries. The
        # derived Hodge L_0 has no link phase (off-diagonal -1/W_1) and is a
        # different block, asserted here so the distinction stays pinned.
        st = _from_simplices(3, [(0, 1), (1, 2), (2, 0)])
        _set_phase(st, 0, 1, 0.7)
        L = np.array(cob.HodgeLaplacian(st).connection_laplacian()).reshape(3, 3)
        block = np.asarray(obs.FiberConnection.chain_transfer(
            st, 0, [[0]], [[1]]))
        self.assertAlmostEqual(abs(block[0, 0] - L[0, 1]), 0.0, delta=MACHINE)
        # oriented U(1) entry: -l^2 e^{i phi} with the stored orientation
        self.assertAlmostEqual(abs(block[0, 0] + cmath.exp(0.7j)), 0.0,
                               delta=MACHINE)
        hodge = np.array(cob.HodgeLaplacian(st).laplacian(0)).reshape(3, 3)
        self.assertGreater(abs(hodge[0, 1] - L[0, 1]), 1e-3)

    def test_degree1_block_matches_whole_operator(self):
        # bowtie: two triangles sharing vertex 0 — the connecting simplices
        st = _from_simplices(5, [(0, 1, 2), (0, 3, 4)])
        cc = cob.ChainComplex.from_spacetime(st)
        cells = [tuple(c) for c in cc.k_simplex_vertices(1)]
        L = np.array(cob.HodgeLaplacian(st).laplacian(1)).reshape(
            len(cells), len(cells))
        cells_a = [c for c in cells if set(c) <= {0, 1, 2}]
        cells_b = [c for c in cells if set(c) <= {0, 3, 4}]
        block = np.asarray(obs.FiberConnection.chain_transfer(
            st, 1, [list(c) for c in cells_a], [list(c) for c in cells_b]))
        idx = {c: i for i, c in enumerate(cells)}
        expected = np.array([[L[idx[ra], idx[cb]] for cb in cells_b]
                             for ra in cells_a])
        np.testing.assert_allclose(block, expected, rtol=0, atol=MACHINE)
        # the shared vertex induces a nonzero transfer
        self.assertGreater(np.abs(block).max(), 0.1)

    def test_disconnected_components_have_zero_transfer(self):
        st = _from_simplices(6, [(0, 1, 2), (3, 4, 5)])
        block = np.asarray(obs.FiberConnection.chain_transfer(
            st, 1, [[0, 1], [1, 2], [0, 2]], [[3, 4], [4, 5], [3, 5]]))
        self.assertEqual(np.abs(block).max(), 0.0)

    def test_unknown_cell_raises(self):
        st = _from_simplices(3, [(0, 1), (1, 2), (2, 0)])
        with self.assertRaises(ValueError):
            obs.FiberConnection.chain_transfer(st, 0, [[0]], [[9]])

    def test_negative_degree_raises(self):
        st = _from_simplices(3, [(0, 1), (1, 2), (2, 0)])
        with self.assertRaises(ValueError):
            obs.FiberConnection.chain_transfer(st, -1, [[0]], [[1]])

    def test_response_transfer_returns_the_effective_block(self):
        # P4 path; components {0,1} / {2,3}: L_eff = [[1,-1],[-1,1]], one
        # kept cell per component, so the intercomponent block is [[-1]].
        L = [[1, -1, 0, 0], [-1, 2, -1, 0], [0, -1, 2, -1], [0, 0, -1, 1]]
        flat = [complex(x) for row in L for x in row]
        q = cob.RecursiveQuotient.over_matrix(flat, 4, [], [[0, 1], [2, 3]])
        network = q.response_network()
        fwd = np.asarray(obs.FiberConnection.response_transfer(network, 0, 1))
        bwd = np.asarray(obs.FiberConnection.response_transfer(network, 1, 0))
        np.testing.assert_allclose(fwd, [[-1.0]], rtol=0, atol=MACHINE)
        np.testing.assert_allclose(bwd, [[-1.0]], rtol=0, atol=MACHINE)

    def test_response_transfer_absent_edge_is_zero_block(self):
        # P4 (+) an uncoupled pair {4,5}: no network edge to component 2.
        L = np.zeros((6, 6))
        L[:4, :4] = [[1, -1, 0, 0], [-1, 2, -1, 0],
                     [0, -1, 2, -1], [0, 0, -1, 1]]
        L[4:, 4:] = [[1, -1], [-1, 1]]
        flat = [complex(x) for x in L.reshape(-1)]
        q = cob.RecursiveQuotient.over_matrix(flat, 6, [],
                                             [[0, 1], [2, 3], [4, 5]])
        network = q.response_network()
        block = np.asarray(obs.FiberConnection.response_transfer(network, 0, 2))
        self.assertEqual(block.shape[0],
                         network.stalk_dimensions[0])
        self.assertEqual(block.shape[1],
                         network.stalk_dimensions[2])
        if block.size:
            self.assertEqual(np.abs(block).max(), 0.0)

    def test_response_transfer_bad_component_raises(self):
        L = [[1, -1, 0, 0], [-1, 2, -1, 0], [0, -1, 2, -1], [0, 0, -1, 1]]
        flat = [complex(x) for row in L for x in row]
        q = cob.RecursiveQuotient.over_matrix(flat, 4, [], [[0, 1], [2, 3]])
        with self.assertRaises(IndexError):
            obs.FiberConnection.response_transfer(q.response_network(), 0, 7)

    def test_response_route_transport_is_exactly_unitary(self):
        # The coarse [[-1]] block between two rank-1 stalk fibers is a unit
        # transfer: |M| = 1, V = -1, determinant phase -1.
        L = [[1, -1, 0, 0], [-1, 2, -1, 0], [0, -1, 2, -1], [0, 0, -1, 1]]
        flat = [complex(x) for row in L for x in row]
        q = cob.RecursiveQuotient.over_matrix(flat, 4, [], [[0, 1], [2, 3]])
        transfer = np.asarray(
            obs.FiberConnection.response_transfer(q.response_network(), 0, 1))
        conn = obs.FiberConnection()
        a = _unit_fiber(100, 1)
        b = _unit_fiber(200, 1)
        read = conn.transport(a, b, transfer)
        self.assertTrue(read.accepted)
        self.assertAlmostEqual(abs(read.unitary_map[0, 0] + 1.0), 0.0,
                               delta=MACHINE)
        self.assertAlmostEqual(abs(read.determinant_phase + 1.0), 0.0,
                               delta=MACHINE)


# =========================================================================== #
# transport diagnostics and gates
# =========================================================================== #
class TestTransportGates(unittest.TestCase):
    def setUp(self):
        self.conn = obs.FiberConnection()
        self.A = _unit_fiber(10, 3)
        self.B = _unit_fiber(20, 3)

    def test_identity_transport(self):
        read = self.conn.transport(self.A, self.B, np.eye(3))
        self.assertTrue(read.accepted)
        self.assertEqual(read.numerical_rank, 3)
        self.assertAlmostEqual(read.leakage, 0.0, delta=MACHINE)
        np.testing.assert_allclose(read.unitary_map, np.eye(3), rtol=0,
                                   atol=MACHINE)
        self.assertAlmostEqual(abs(read.determinant_phase - 1.0), 0.0,
                               delta=MACHINE)
        self.assertTrue(read.certificate.holds())

    def test_known_unitary_polar_factor_is_the_unitary(self):
        u = _random_unitary(np.random.default_rng(7), 3)
        read = self.conn.transport(self.A, self.B, u)
        np.testing.assert_allclose(read.raw_map, u, rtol=0, atol=MACHINE)
        np.testing.assert_allclose(read.unitary_map, u, rtol=0, atol=MACHINE)
        self.assertAlmostEqual(
            abs(read.determinant_phase - np.linalg.det(u)), 0.0, delta=1e-10)

    def test_leaking_map_rejected_before_polar_though_polar_exists(self):
        u = _random_unitary(np.random.default_rng(3), 3)
        m = np.diag([0.5, 2.0, 1.0]) @ u
        read = self.conn.transport(self.A, self.B, m)
        self.assertFalse(read.accepted)
        self.assertIn("leaking", read.rejection_reason)
        self.assertEqual(read.unitary_map.size, 0)  # nothing was normalized
        self.assertFalse(read.certificate.holds())
        # the polar factor EXISTS (numpy) — the point is we refused it
        w, _, vh = np.linalg.svd(m)
        polar = w @ vh
        np.testing.assert_allclose(polar.conj().T @ polar, np.eye(3),
                                   rtol=0, atol=MACHINE)
        # leakage was reported before normalization, and honestly
        self.assertAlmostEqual(
            read.leakage,
            np.linalg.norm(m.conj().T @ m - np.eye(3), 2), delta=SOLVER)
        np.testing.assert_allclose(sorted(read.singular_values),
                                   [0.5, 1.0, 2.0], rtol=0, atol=MACHINE)

    def test_rank_deficient_overlap_rejected(self):
        m = np.diag([1.0, 1.0, 0.0])
        read = self.conn.transport(self.A, self.B, m)
        self.assertFalse(read.accepted)
        self.assertIn("rank-deficient", read.rejection_reason)
        self.assertEqual(read.numerical_rank, 2)

    def test_ill_conditioned_overlap_rejected(self):
        # full numerical rank at 1e-9 relative, conditioning 5e8 > 1e8 cap
        m = np.diag([1.0, 1.0, 2e-9])
        read = self.conn.transport(self.A, self.B, m)
        self.assertFalse(read.accepted)
        self.assertIn("ill-conditioned", read.rejection_reason)
        self.assertEqual(read.numerical_rank, 3)
        self.assertGreater(read.overlap_condition_number, 1e8)

    def test_uncertified_endpoint_band_rejected(self):
        stale = _fiber([[40], [41], [42]], np.eye(3), accepted=False)
        read = self.conn.transport(self.A, stale, np.eye(3))
        self.assertFalse(read.accepted)
        self.assertIn("uncertified", read.rejection_reason)

    def test_endpoint_gap_floor_rejects(self):
        cfg = obs.FiberConnectionConfig()
        cfg.min_endpoint_gap = 0.5
        conn = obs.FiberConnection(cfg)
        narrow = _fiber([[50], [51], [52]], np.eye(3), lower_gap=0.1,
                        upper_gap=2.0)
        read = conn.transport(self.A, narrow, np.eye(3))
        self.assertFalse(read.accepted)
        self.assertIn("gap", read.rejection_reason)

    def test_frame_conditioning_cap_rejects(self):
        skewed = _fiber([[60], [61], [62]], np.eye(3), cond=1e12)
        read = self.conn.transport(self.A, skewed, np.eye(3))
        self.assertFalse(read.accepted)
        self.assertIn("conditioning", read.rejection_reason)

    def test_rank_mismatch_rejected_with_rectangular_raw_map(self):
        two = _fiber([[70], [71]], np.eye(2))
        read = self.conn.transport(two, self.B, np.zeros((2, 3)) + np.eye(3)[:2])
        self.assertFalse(read.accepted)
        self.assertIn("rank mismatch", read.rejection_reason)
        self.assertEqual(read.raw_map.shape, (2, 3))

    def test_degree_mismatch_raises(self):
        other = _fiber([[80], [81], [82]], np.eye(3), degree=2)
        with self.assertRaises(ValueError):
            self.conn.transport(self.A, other, np.eye(3))

    def test_transfer_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            self.conn.transport(self.A, self.B, np.eye(4))

    def test_diagnostics_are_reported_on_rejected_reads(self):
        m = np.diag([0.5, 2.0, 1.0])
        read = self.conn.transport(self.A, self.B, m)
        self.assertEqual(len(read.singular_values), 3)
        self.assertEqual(read.to_positive_signature, 3)
        self.assertEqual(read.from_negative_signature, 0)
        self.assertEqual(read.to_gap, 1.0)
        self.assertEqual(read.frame_condition_number, 1.0)
        self.assertIn("REJECTED", read.describe())

    def test_config_thresholds_take_effect(self):
        cfg = obs.FiberConnectionConfig()
        cfg.leakage_tolerance = 10.0  # caller-declared loose gate
        conn = obs.FiberConnection(cfg)
        self.assertEqual(conn.config().leakage_tolerance, 10.0)
        m = np.diag([0.9, 1.1, 1.0])
        read = conn.transport(self.A, self.B, m)
        self.assertTrue(read.accepted)  # within the declared tolerance
        self.assertGreater(read.leakage, 0.1)


# =========================================================================== #
# gauge covariance (independent random U(r) frame changes)
# =========================================================================== #
class TestGaugeCovariance(unittest.TestCase):
    def test_transport_is_bifundamental(self):
        rng = np.random.default_rng(11)
        t = _random_unitary(rng, 3)  # unitary transfer: both reads accepted
        for seed in range(5):
            g_rng = np.random.default_rng(100 + seed)
            g_a = _random_unitary(g_rng, 3)
            g_b = _random_unitary(g_rng, 3)
            conn = obs.FiberConnection()
            a0 = _fiber([[1], [2], [3]], np.eye(3))
            b0 = _fiber([[4], [5], [6]], np.eye(3))
            a1 = _fiber([[1], [2], [3]], g_a)
            b1 = _fiber([[4], [5], [6]], g_b)
            base = conn.transport(a0, b0, t)
            changed = conn.transport(a1, b1, t)
            np.testing.assert_allclose(
                changed.raw_map, g_a.conj().T @ base.raw_map @ g_b,
                rtol=0, atol=MACHINE)
            np.testing.assert_allclose(
                changed.unitary_map, g_a.conj().T @ base.unitary_map @ g_b,
                rtol=0, atol=MACHINE)

    def test_diagnostics_are_frame_invariant(self):
        rng = np.random.default_rng(21)
        t = _random_unitary(rng, 3) @ np.diag([1.0, 1.0, 1.0 - 1e-8])
        g_a = _random_unitary(rng, 3)
        g_b = _random_unitary(rng, 3)
        conn = obs.FiberConnection()
        base = conn.transport(_fiber([[1], [2], [3]], np.eye(3)),
                              _fiber([[4], [5], [6]], np.eye(3)), t)
        changed = conn.transport(_fiber([[1], [2], [3]], g_a),
                                 _fiber([[4], [5], [6]], g_b), t)
        np.testing.assert_allclose(changed.singular_values,
                                   base.singular_values, rtol=0, atol=MACHINE)
        self.assertAlmostEqual(changed.leakage, base.leakage, delta=MACHINE)
        self.assertEqual(changed.accepted, base.accepted)

    def test_closed_holonomy_conjugates_at_the_base_point(self):
        rng = np.random.default_rng(31)
        transfers = [_random_unitary(rng, 3) for _ in range(3)]
        frames = [np.eye(3)] * 3
        changed_frames = [_random_unitary(np.random.default_rng(400 + i), 3)
                          for i in range(3)]
        conn = obs.FiberConnection()

        def loop(frame_list):
            fibers = [_fiber([[10 * i + 1], [10 * i + 2], [10 * i + 3]],
                             frame_list[i]) for i in range(3)]
            links = [conn.transport(fibers[i], fibers[(i + 1) % 3],
                                    transfers[i]) for i in range(3)]
            return conn.holonomy(links)

        base = loop(frames)
        changed = loop(changed_frames)
        g0 = changed_frames[0]
        self.assertTrue(base.closed and changed.closed)
        np.testing.assert_allclose(
            changed.holonomy, g0.conj().T @ base.holonomy @ g0,
            rtol=0, atol=MACHINE)
        # base-point conjugation observables are invariant
        self.assertAlmostEqual(abs(changed.normalized_trace
                                   - base.normalized_trace), 0.0, delta=MACHINE)
        self.assertAlmostEqual(abs(changed.determinant - base.determinant),
                               0.0, delta=MACHINE)
        self.assertAlmostEqual(abs(changed.adjoint_trace - base.adjoint_trace),
                               0.0, delta=1e-10)

    def test_krein_frame_change_is_bifundamental(self):
        # J-unitary frame changes g in U(2) x U(1) preserve J = diag(1,1,-1)
        rng = np.random.default_rng(41)
        j = np.diag([1.0, 1.0, -1.0]).astype(complex)
        a = 0.3
        boost = np.array([[1, 0, 0],
                          [0, math.cosh(a), math.sinh(a)],
                          [0, math.sinh(a), math.cosh(a)]], dtype=complex)
        w = np.array([1.0, 1.0, -1.0], dtype=complex)

        def krein_fiber(base, frame):
            return _fiber([[base], [base + 1], [base + 2]], frame,
                          left=frame @ j, weights=w,
                          regime="hermitian-indefinite", pos=2, neg=1,
                          self_adjoint=False)

        def block_unitary(seed):
            g_rng = np.random.default_rng(seed)
            g = np.zeros((3, 3), dtype=complex)
            g[:2, :2] = _random_unitary(g_rng, 2)
            g[2, 2] = cmath.exp(1j * g_rng.normal())
            return g

        conn = obs.FiberConnection()
        transfer = np.diag(1.0 / w) @ boost  # M = W T = boost (J-unitary)
        base = conn.transport(krein_fiber(1, np.eye(3)),
                              krein_fiber(11, np.eye(3)), transfer)
        self.assertTrue(base.accepted)
        g_a, g_b = block_unitary(1), block_unitary(2)
        # frames must stay J-normalized: Phi -> Phi g with g^dag J g = J
        for g in (g_a, g_b):
            np.testing.assert_allclose(g.conj().T @ j @ g, j, rtol=0,
                                       atol=MACHINE)
        changed = conn.transport(krein_fiber(1, g_a), krein_fiber(11, g_b),
                                 transfer)
        self.assertTrue(changed.accepted)
        np.testing.assert_allclose(changed.unitary_map,
                                   g_a.conj().T @ base.unitary_map @ g_b,
                                   rtol=0, atol=1e-10)


# =========================================================================== #
# identity / reverse
# =========================================================================== #
class TestIdentityAndReverse(unittest.TestCase):
    def test_identity_self_transport_returns_identity_holonomy(self):
        conn = obs.FiberConnection()
        a = _unit_fiber(1, 3)
        link = conn.transport(a, a, np.eye(3))
        read = conn.holonomy([link])
        self.assertTrue(read.closed)
        np.testing.assert_allclose(read.holonomy, np.eye(3), rtol=0,
                                   atol=MACHINE)
        self.assertAlmostEqual(abs(read.normalized_trace - 1.0), 0.0,
                               delta=MACHINE)
        self.assertAlmostEqual(abs(read.determinant - 1.0), 0.0,
                               delta=MACHINE)
        # chi_adj(1) = dim of the adjoint = r^2 - 1
        self.assertAlmostEqual(abs(read.adjoint_trace - 8.0), 0.0,
                               delta=MACHINE)

    def test_reverse_link_is_the_adjoint_with_nontrivial_weights(self):
        # W-orthonormal frames with genuinely different endpoint metrics:
        # the W-adjoint reverse block returns exactly M_BA = M_AB^dagger.
        w_a = np.array([1.0, 2.0, 3.0])
        w_b = np.array([4.0, 5.0, 6.0])
        phi_a = np.diag(1.0 / np.sqrt(w_a)).astype(complex)
        phi_b = np.diag(1.0 / np.sqrt(w_b)).astype(complex)
        a = _fiber([[1], [2], [3]], phi_a, weights=w_a)
        b = _fiber([[4], [5], [6]], phi_b, weights=w_b)
        u = _random_unitary(np.random.default_rng(5), 3)
        transfer = np.diag(1.0 / np.sqrt(w_a)) @ u @ np.diag(np.sqrt(w_b))
        conn = obs.FiberConnection()
        fwd = conn.transport(a, b, transfer)
        rev = conn.transport_reverse(a, b, transfer)
        self.assertTrue(fwd.accepted and rev.accepted)
        np.testing.assert_allclose(rev.raw_map, fwd.raw_map.conj().T,
                                   rtol=0, atol=MACHINE)
        np.testing.assert_allclose(rev.unitary_map,
                                   fwd.unitary_map.conj().T,
                                   rtol=0, atol=MACHINE)
        np.testing.assert_allclose(rev.unitary_map @ fwd.unitary_map,
                                   np.eye(3), rtol=0, atol=MACHINE)
        # direction bookkeeping swaps
        self.assertEqual(rev.to_key, fwd.from_key)
        self.assertEqual(rev.from_key, fwd.to_key)

    def test_forward_then_reverse_loop_is_identity(self):
        conn = obs.FiberConnection()
        a = _unit_fiber(1, 2)
        b = _unit_fiber(11, 2)
        u = _rotation(0.8)
        fwd = conn.transport(a, b, u)
        rev = conn.transport_reverse(a, b, u)
        read = conn.holonomy([fwd, rev])
        self.assertTrue(read.closed)
        np.testing.assert_allclose(read.holonomy, np.eye(2), rtol=0,
                                   atol=MACHINE)


# =========================================================================== #
# Wilson observables
# =========================================================================== #
class TestWilsonObservables(unittest.TestCase):
    def test_rotation_loop_has_the_known_product(self):
        thetas = [0.3, 0.5, -0.2, 0.9]
        conn = obs.FiberConnection()
        fibers = [_unit_fiber(100 * (i + 1), 2) for i in range(4)]
        links = [conn.transport(fibers[i], fibers[(i + 1) % 4],
                                _rotation(thetas[i])) for i in range(4)]
        read = conn.holonomy(links)
        total = sum(thetas)
        np.testing.assert_allclose(read.holonomy, _rotation(total),
                                   rtol=0, atol=MACHINE)
        self.assertAlmostEqual(abs(read.normalized_trace
                                   - math.cos(total)), 0.0, delta=MACHINE)
        self.assertAlmostEqual(abs(read.determinant - 1.0), 0.0,
                               delta=MACHINE)
        self.assertTrue(read.unitary)
        self.assertLess(read.unitarity_residual, MACHINE)
        self.assertTrue(read.certificate.holds())

    def test_determinant_line_is_the_product_of_link_determinants(self):
        rng = np.random.default_rng(17)
        conn = obs.FiberConnection()
        fibers = [_unit_fiber(100 * (i + 1), 3) for i in range(3)]
        links = [conn.transport(fibers[i], fibers[(i + 1) % 3],
                                _random_unitary(rng, 3)) for i in range(3)]
        read = conn.holonomy(links)
        expected = np.prod([np.linalg.det(l.unitary_map) for l in links])
        self.assertAlmostEqual(abs(read.determinant - expected), 0.0,
                               delta=1e-10)

    def test_open_chain_reports_not_closed(self):
        conn = obs.FiberConnection()
        a, b, c = (_unit_fiber(100 * (i + 1), 2) for i in range(3))
        links = [conn.transport(a, b, np.eye(2)),
                 conn.transport(b, c, np.eye(2))]
        read = conn.holonomy(links)
        self.assertFalse(read.closed)
        self.assertEqual(read.loop_length, 2)

    def test_rejected_link_refuses_to_multiply(self):
        conn = obs.FiberConnection()
        a, b = _unit_fiber(1, 3), _unit_fiber(11, 3)
        good = conn.transport(a, b, np.eye(3))
        bad = conn.transport(b, a, np.diag([0.1, 1.0, 1.0]))
        self.assertFalse(bad.accepted)
        with self.assertRaises(ValueError):
            conn.holonomy([good, bad])

    def test_empty_and_rank_mismatch_raise(self):
        conn = obs.FiberConnection()
        with self.assertRaises(ValueError):
            conn.holonomy([])
        two = conn.transport(_unit_fiber(1, 2), _unit_fiber(11, 2), np.eye(2))
        three = conn.transport(_unit_fiber(21, 3), _unit_fiber(31, 3),
                               np.eye(3))
        with self.assertRaises(ValueError):
            conn.holonomy([two, three])

    def test_adjoint_matrix_matches_numpy_and_is_center_blind(self):
        rng = np.random.default_rng(23)
        conn = obs.FiberConnection()
        a, b = _unit_fiber(1, 3), _unit_fiber(11, 3)
        u = _random_unitary(rng, 3)
        read = conn.holonomy([conn.transport(a, b, u),
                              conn.transport(b, a, u.conj().T @ u)])
        h = read.holonomy
        kron = np.kron(h.conj(), h)  # vec(H M H^dag) = (conj(H) x H) vec(M)
        p8 = np.asarray(obs.ColorFiber.adjoint_octet_projector())
        np.testing.assert_allclose(read.adjoint_matrix, p8 @ kron @ p8,
                                   rtol=0, atol=1e-10)
        # center-blind: Ad(omega H) = Ad(H)
        np.testing.assert_allclose(
            np.asarray(obs.FiberConnection.adjoint_representation(OMEGA * h)),
            read.adjoint_matrix, rtol=0, atol=1e-10)
        # trace of the octet restriction is the adjoint character
        self.assertAlmostEqual(
            abs(np.trace(read.adjoint_matrix) - read.adjoint_trace), 0.0,
            delta=1e-9)
        self.assertAlmostEqual(
            abs(read.adjoint_trace - (abs(np.trace(h)) ** 2 - 1.0)), 0.0,
            delta=1e-10)

    def test_generic_rank_has_no_adjoint_matrix_but_a_character(self):
        conn = obs.FiberConnection()
        a, b = _unit_fiber(1, 2), _unit_fiber(11, 2)
        read = conn.holonomy([conn.transport(a, b, _rotation(0.4)),
                              conn.transport(b, a, _rotation(-0.4))])
        self.assertEqual(read.adjoint_matrix.size, 0)
        self.assertAlmostEqual(
            abs(read.adjoint_trace
                - (abs(np.trace(read.holonomy)) ** 2 - 1.0)),
            0.0, delta=MACHINE)


# =========================================================================== #
# rank-three center structure
# =========================================================================== #
class TestRankThreeCenter(unittest.TestCase):
    def setUp(self):
        self.conn = obs.FiberConnection()
        self.A = _unit_fiber(1, 3)
        self.B = _unit_fiber(11, 3)
        self.C = _unit_fiber(21, 3)

    def _center_loop(self, sector_phase):
        """Three links, each with determinant phase sector_phase."""
        links = []
        chain = [(self.A, self.B), (self.B, self.C), (self.C, self.A)]
        for to_f, from_f in chain:
            links.append(_phase_link(self.conn, to_f, from_f, sector_phase))
        return links

    def test_all_three_cube_root_branches(self):
        u = _random_unitary(np.random.default_rng(29), 3)
        read = self.conn.transport(self.A, self.B, u)
        delta = read.determinant_phase
        root = cmath.exp(1j * cmath.phase(delta) / 3.0)
        adjoints, traces = [], []
        for s in range(3):
            branch = read.unitary_map / (root * OMEGA ** s)
            # every branch is a genuine SU(3) lift
            self.assertLess(abs(np.linalg.det(branch) - 1.0), 1e-10)
            adjoints.append(np.asarray(
                obs.FiberConnection.adjoint_representation(branch)))
            traces.append(np.trace(branch))
        # projective/adjoint reads agree on ALL THREE branches
        np.testing.assert_allclose(adjoints[0], adjoints[1], rtol=0,
                                   atol=1e-10)
        np.testing.assert_allclose(adjoints[0], adjoints[2], rtol=0,
                                   atol=1e-10)
        # the fundamental traces expose the three distinct center lifts
        self.assertAlmostEqual(abs(traces[1] - traces[0] / OMEGA), 0.0,
                               delta=1e-10)
        self.assertAlmostEqual(abs(traces[2] - traces[0] / OMEGA ** 2), 0.0,
                               delta=1e-10)

    def test_projective_representative_is_special_unitary(self):
        u = _random_unitary(np.random.default_rng(31), 3)
        rep = np.asarray(obs.FiberConnection.projective_representative(u, GATE))
        self.assertLess(abs(np.linalg.det(rep) - 1.0), 1e-10)
        # same projective class for every center twist of the input
        rep_twisted = np.asarray(
            obs.FiberConnection.projective_representative(OMEGA * u, GATE))
        ratios = [np.linalg.norm(rep_twisted - (OMEGA ** s) * rep)
                  for s in range(3)]
        self.assertLess(min(ratios), 1e-10)

    def test_fundamental_lift_branches_shift_by_center(self):
        links = self._center_loop(0.4)
        lifts = [self.conn.fundamental_lift(links, GATE, s) for s in range(3)]
        for lift in lifts:
            self.assertTrue(lift.valid)
            self.assertLess(lift.det_residual, 1e-12)
            self.assertLess(
                np.linalg.norm(np.asarray(lift.lift).conj().T
                               @ np.asarray(lift.lift) - np.eye(3), 2),
                1e-12)
        # the recorded center sector is branch-INDEPENDENT
        self.assertEqual({l.center_sector for l in lifts}, {0})
        # the lift itself shifts by omega^{-s}
        base = np.asarray(lifts[0].lift)
        np.testing.assert_allclose(np.asarray(lifts[1].lift),
                                   base / OMEGA, rtol=0, atol=1e-12)
        np.testing.assert_allclose(np.asarray(lifts[2].lift),
                                   base / OMEGA ** 2, rtol=0, atol=1e-12)
        # projective/adjoint reads of the lift are branch-independent
        ad = [np.asarray(obs.FiberConnection.adjoint_representation(
            np.asarray(l.lift))) for l in lifts]
        np.testing.assert_allclose(ad[0], ad[1], rtol=0, atol=1e-10)
        np.testing.assert_allclose(ad[0], ad[2], rtol=0, atol=1e-10)
        # and the fundamental traces are the three distinct lifts
        self.assertAlmostEqual(
            abs(lifts[1].lift_trace - lifts[0].lift_trace / OMEGA), 0.0,
            delta=1e-12)

    def test_known_center_loop_returns_the_analytic_sector(self):
        # three links of determinant phase 2*pi/3: Theta = 2*pi, sector 1
        lift = self.conn.fundamental_lift(self._center_loop(TWO_PI / 3.0), GATE)
        self.assertTrue(lift.valid)
        self.assertEqual(lift.center_sector, 1)
        self.assertAlmostEqual(lift.accumulated_determinant_phase, TWO_PI,
                               delta=MACHINE)
        # conjugate loop: Theta = -2*pi, sector 2 (= -1 mod 3)
        lift_bar = self.conn.fundamental_lift(self._center_loop(-TWO_PI / 3.0), GATE)
        self.assertEqual(lift_bar.center_sector, 2)
        # trivial loop: sector 0
        lift0 = self.conn.fundamental_lift(self._center_loop(0.1), GATE)
        self.assertEqual(lift0.center_sector, 0)

    def test_lift_refuses_generic_rank(self):
        two_a, two_b = _unit_fiber(31, 2), _unit_fiber(41, 2)
        link = self.conn.transport(two_a, two_b, _rotation(0.3))
        lift = self.conn.fundamental_lift([link], GATE)
        self.assertFalse(lift.valid)
        self.assertIn("rank", lift.invalid_reason)
        self.assertFalse(lift.certificate.holds())

    def test_lift_refuses_gl_transport(self):
        phi = np.array([[1.0, 0.2, 0.0], [0.0, 1.0, 0.3], [0.1, 0.0, 1.0]],
                       dtype=complex)
        psi = np.linalg.inv(phi.conj().T)
        nn = _fiber([[51], [52], [53]], phi, left=psi, regime="non-normal",
                    self_adjoint=False)
        link = self.conn.transport(nn, self.B, np.eye(3))
        self.assertTrue(link.accepted)
        self.assertEqual(link.unitary_map.size, 0)
        lift = self.conn.fundamental_lift([link], GATE)
        self.assertFalse(lift.valid)
        self.assertIn("GL", lift.invalid_reason)

    def test_lift_determinant_identity(self):
        links = self._center_loop(0.7)
        lift = self.conn.fundamental_lift(links, GATE)
        h = np.asarray(self.conn.holonomy(links).holonomy)
        theta = lift.accumulated_determinant_phase
        self.assertAlmostEqual(
            abs(np.linalg.det(h) - cmath.exp(1j * theta)), 0.0, delta=1e-12)
        np.testing.assert_allclose(np.asarray(lift.lift),
                                   h * cmath.exp(-1j * theta / 3.0),
                                   rtol=0, atol=1e-12)

    def test_bad_branch_raises(self):
        links = self._center_loop(0.2)
        with self.assertRaises(ValueError):
            self.conn.fundamental_lift(links, GATE, 3)


# =========================================================================== #
# determinant winding
# =========================================================================== #
class TestDeterminantWinding(unittest.TestCase):
    def setUp(self):
        self.conn = obs.FiberConnection()
        self.A = _unit_fiber(1, 3)
        self.B = _unit_fiber(11, 3)

    def _family(self, phases):
        return [_phase_link(self.conn, self.A, self.B, p) for p in phases]

    def test_closed_family_integer_winding(self):
        family = self._family([TWO_PI * k / 8 for k in range(8)])
        read = self.conn.closed_family_winding(family)
        self.assertEqual(read.winding, 1)
        self.assertEqual(read.winding_closure, "closed-family")
        self.assertAlmostEqual(read.accumulated_phase, TWO_PI, delta=MACHINE)
        self.assertAlmostEqual(read.max_phase_step, TWO_PI / 8, delta=MACHINE)
        self.assertTrue(read.certificate.holds())

    def test_double_winding(self):
        family = self._family([math.pi * k / 2 for k in range(8)])
        read = self.conn.closed_family_winding(family)
        self.assertEqual(read.winding, 2)
        self.assertAlmostEqual(read.accumulated_phase, 2 * TWO_PI,
                               delta=MACHINE)

    def test_orientation_reversal_flips_the_sign(self):
        # reversing the tube orientation = traversing the SAME family of
        # transports in the opposite parameter order (the t-circle
        # reversed); inverting each map is LINK reversal, a different
        # operation.
        family = self._family([TWO_PI * k / 8 for k in range(8)])
        self.assertEqual(self.conn.closed_family_winding(family).winding, 1)
        self.assertEqual(
            self.conn.closed_family_winding(list(reversed(family))).winding,
            -1)

    def test_single_sample_family_winds_zero(self):
        read = self.conn.closed_family_winding(self._family([0.3]))
        self.assertEqual(read.winding, 0)

    def test_unaccepted_sample_invalidates(self):
        family = self._family([0.0, TWO_PI / 4])
        bad = self.conn.transport(self.A, self.B, np.diag([0.1, 1.0, 1.0]))
        self.assertFalse(bad.accepted)
        read = self.conn.closed_family_winding(family + [bad])
        self.assertIsNone(read.winding)
        self.assertIn("not an accepted transport", read.invalidation_reason)
        self.assertFalse(read.certificate.holds())

    def test_aliasing_step_invalidates(self):
        family = self._family([0.0, math.pi])
        read = self.conn.closed_family_winding(family)
        self.assertIsNone(read.winding)
        self.assertIn("aliasing", read.invalidation_reason)

    def test_rank_change_invalidates(self):
        family = self._family([0.0, 0.4])
        two = self.conn.transport(_unit_fiber(31, 2), _unit_fiber(41, 2),
                                  _rotation(0.1))
        read = self.conn.closed_family_winding(family + [two])
        self.assertIsNone(read.winding)
        self.assertIn("rank", read.invalidation_reason)

    def test_open_segment_agrees_under_both_declared_closures(self):
        # segment with unit-determinant endpoints accumulating 2*pi
        phases = [TWO_PI * k / 4 for k in range(5)]  # 0 .. 2*pi inclusive
        segment = self._family(phases)
        # matched reference: the constant identity family, matched at both
        # endpoints (V(0) = V(end) = I exactly)
        matched = obs.WindingClosureSpec()
        matched.mode = obs.WindingClosureSpec.Mode.MATCHED_REFERENCE
        matched.reference_id = "constant-identity-reference"
        matched.reference_transports = [np.eye(3)] * len(segment)
        by_reference = self.conn.open_segment_winding(segment, matched)
        # boundary-register trivializations at both endpoints
        trivialized = obs.WindingClosureSpec()
        trivialized.mode = obs.WindingClosureSpec.Mode.ENDPOINT_TRIVIALIZATION
        trivialized.reference_id = "register-identity-frames"
        trivialized.start_trivialization = np.eye(3)
        trivialized.end_trivialization = np.eye(3)
        by_registers = self.conn.open_segment_winding(segment, trivialized)
        self.assertEqual(by_reference.winding, 1)
        self.assertEqual(by_registers.winding, 1)
        self.assertEqual(by_reference.winding_closure, "matched-reference")
        self.assertEqual(by_registers.winding_closure,
                         "endpoint-trivialization")
        self.assertEqual(by_reference.winding_reference_id,
                         "constant-identity-reference")
        self.assertLess(by_reference.closure_defect, MACHINE)
        self.assertTrue(by_reference.certificate.holds())
        self.assertTrue(by_registers.certificate.holds())

    def test_open_segment_without_closure_is_unknown(self):
        segment = self._family([TWO_PI * k / 4 for k in range(5)])
        read = self.conn.open_segment_winding(segment,
                                            obs.WindingClosureSpec())
        self.assertIsNone(read.winding)
        self.assertEqual(read.winding_closure, "none")
        self.assertIn("no closure declared", read.invalidation_reason)
        # the raw open-path phase is reported, but never as an integer claim
        self.assertAlmostEqual(read.accumulated_phase, TWO_PI, delta=MACHINE)
        self.assertFalse(read.certificate.holds())

    def test_open_segment_orientation_reversal(self):
        # same segment traversed backwards (trivializations swap ends)
        forward = self._family([TWO_PI * k / 4 for k in range(5)])
        spec = obs.WindingClosureSpec()
        spec.mode = obs.WindingClosureSpec.Mode.ENDPOINT_TRIVIALIZATION
        spec.start_trivialization = np.eye(3)
        spec.end_trivialization = np.eye(3)
        self.assertEqual(self.conn.open_segment_winding(forward, spec).winding,
                         1)
        self.assertEqual(
            self.conn.open_segment_winding(list(reversed(forward)),
                                         spec).winding, -1)

    def test_mismatched_reference_is_graded_honestly(self):
        segment = self._family([TWO_PI * k / 8 for k in range(5)])  # open end
        spec = obs.WindingClosureSpec()
        spec.mode = obs.WindingClosureSpec.Mode.MATCHED_REFERENCE
        spec.reference_transports = [np.eye(3)] * len(segment)
        read = self.conn.open_segment_winding(segment, spec)
        # the reference does NOT match the far endpoint: defect reported and
        # the closure certificate refuses to hold
        self.assertGreater(read.closure_defect, 0.1)
        self.assertFalse(read.certificate.holds())

    def test_reference_length_mismatch_raises(self):
        segment = self._family([0.0, 0.1])
        spec = obs.WindingClosureSpec()
        spec.mode = obs.WindingClosureSpec.Mode.MATCHED_REFERENCE
        spec.reference_transports = [np.eye(3)]
        with self.assertRaises(ValueError):
            self.conn.open_segment_winding(segment, spec)

    def test_trivialization_shape_mismatch_raises(self):
        segment = self._family([0.0, 0.1])
        spec = obs.WindingClosureSpec()
        spec.mode = obs.WindingClosureSpec.Mode.ENDPOINT_TRIVIALIZATION
        spec.start_trivialization = np.eye(2)
        spec.end_trivialization = np.eye(3)
        with self.assertRaises(ValueError):
            self.conn.open_segment_winding(segment, spec)

    def test_empty_family_raises(self):
        with self.assertRaises(ValueError):
            self.conn.closed_family_winding([])


# =========================================================================== #
# Krein and non-normal regimes
# =========================================================================== #
class TestKreinAndNonNormal(unittest.TestCase):
    J = np.diag([1.0, 1.0, -1.0]).astype(complex)
    W = np.array([1.0, 1.0, -1.0], dtype=complex)

    def _krein_fiber(self, base, frame=None):
        frame = np.eye(3) if frame is None else frame
        return _fiber([[base], [base + 1], [base + 2]], frame,
                      left=frame @ self.J, weights=self.W,
                      regime="hermitian-indefinite", pos=2, neg=1,
                      self_adjoint=False)

    def _boost(self, a):
        return np.array([[1, 0, 0],
                         [0, math.cosh(a), math.sinh(a)],
                         [0, math.sinh(a), math.cosh(a)]], dtype=complex)

    def test_matching_krein_sectors_reduce_pseudo_unitarily(self):
        conn = obs.FiberConnection()
        boost = self._boost(0.4)
        transfer = np.diag(1.0 / self.W) @ boost  # M = W T = boost
        read = conn.transport(self._krein_fiber(1), self._krein_fiber(11),
                              transfer)
        self.assertTrue(read.accepted)
        self.assertLess(read.leakage, MACHINE)  # J-isometry defect
        v = np.asarray(read.unitary_map)
        # inertia retained: V^dag J V = J exactly (pseudo-unitary)
        np.testing.assert_allclose(v.conj().T @ self.J @ v, self.J,
                                   rtol=0, atol=1e-10)
        np.testing.assert_allclose(v, boost, rtol=0, atol=1e-10)
        # a boost is NOT Euclidean-unitary — the reduction kept the metric
        self.assertGreater(np.linalg.norm(v.conj().T @ v - np.eye(3), 2),
                           0.1)
        self.assertEqual(read.regime, cob.CertificateRegime.HermitianIndefinite)

    def test_signature_mismatch_is_rejected_before_reduction(self):
        conn = obs.FiberConnection()
        flipped = _fiber([[21], [22], [23]], np.eye(3),
                         left=np.diag([1.0, -1.0, -1.0]).astype(complex),
                         weights=np.array([1.0, -1.0, -1.0], dtype=complex),
                         regime="hermitian-indefinite", pos=1, neg=2,
                         self_adjoint=False)
        read = conn.transport(self._krein_fiber(1), flipped, np.eye(3))
        self.assertFalse(read.accepted)
        self.assertIn("signature mismatch", read.rejection_reason)
        self.assertEqual(read.unitary_map.size, 0)
        # inertia was reported, not silently Euclideanized
        self.assertEqual((read.to_positive_signature, read.to_negative_signature),
                         (2, 1))
        self.assertEqual((read.from_positive_signature,
                          read.from_negative_signature), (1, 2))

    def test_krein_loop_retains_the_metric(self):
        conn = obs.FiberConnection()
        t1 = np.diag(1.0 / self.W) @ self._boost(0.3)
        t2 = np.diag(1.0 / self.W) @ self._boost(-0.3)
        a, b = self._krein_fiber(1), self._krein_fiber(11)
        read = conn.holonomy([conn.transport(a, b, t1),
                              conn.transport(b, a, t2)])
        h = np.asarray(read.holonomy)
        np.testing.assert_allclose(h.conj().T @ self.J @ h, self.J,
                                   rtol=0, atol=1e-10)
        np.testing.assert_allclose(h, np.eye(3), rtol=0, atol=1e-10)

    def test_krein_loop_is_graded_against_its_own_metric(self):
        # a NON-cancelling pseudo-unitary loop: exactly J-unitary, far from
        # Euclidean-unitary — the residual is the J-isometry defect, so the
        # certificate holds instead of failing against the wrong metric
        conn = obs.FiberConnection()
        t = np.diag(1.0 / self.W) @ self._boost(0.5)
        a = self._krein_fiber(1)
        read = conn.holonomy([conn.transport(a, a, t)])
        h = np.asarray(read.holonomy)
        self.assertLess(read.unitarity_residual, 1e-10)  # J-defect
        self.assertGreater(np.linalg.norm(h.conj().T @ h - np.eye(3), 2),
                           0.1)  # Euclidean defect is real and large
        self.assertTrue(read.certificate.holds())
        self.assertEqual(read.certificate.regime,
                         cob.CertificateRegime.HermitianIndefinite)

    def test_lift_refuses_pseudo_unitary_links(self):
        # a Krein link carries a J-unitary factor, not a U(3) one: the
        # SU(3) lift refuses outside the positive regime
        conn = obs.FiberConnection()
        t = np.diag(1.0 / self.W) @ self._boost(0.4)
        a = self._krein_fiber(1)
        link = conn.transport(a, a, t)
        self.assertTrue(link.accepted)
        lift = conn.fundamental_lift([link], GATE)
        self.assertFalse(lift.valid)
        self.assertIn("positive regime", lift.invalid_reason)

    def _nonnormal_fiber(self, base, seed=13):
        rng = np.random.default_rng(seed)
        phi = np.eye(3) + 0.3 * rng.normal(size=(3, 3))
        phi = phi.astype(complex)
        psi = np.linalg.inv(phi.conj().T)  # Psi^dag W Phi = I with W = I
        return _fiber([[base], [base + 1], [base + 2]], phi, left=psi,
                      regime="non-normal", self_adjoint=False), phi, psi

    def test_non_normal_takes_the_certified_biorthogonal_path(self):
        conn = obs.FiberConnection()
        nn, phi, psi = self._nonnormal_fiber(51)
        b = _unit_fiber(11, 3)
        t = np.diag([1.0, 2.0, 3.0]).astype(complex)
        read = conn.transport(nn, b, t)
        # the LEFT frame is used: M = Psi^dag W T Phi_B
        np.testing.assert_allclose(read.raw_map, psi.conj().T @ t,
                                   rtol=0, atol=MACHINE)
        self.assertTrue(read.accepted)
        self.assertEqual(read.unitary_map.size, 0)  # GL(r,C) retained
        self.assertEqual(read.regime, cob.CertificateRegime.NonNormal)
        # conditioning and leakage are part of the observable
        self.assertGreater(read.overlap_condition_number, 1.0)
        self.assertTrue(np.isfinite(read.leakage))
        # the determinant phase is never discarded
        det = np.linalg.det(read.raw_map)
        self.assertAlmostEqual(
            abs(read.determinant_phase - det / abs(det)), 0.0, delta=1e-10)

    def test_no_unitary_wilson_value_outside_the_positive_domain(self):
        conn = obs.FiberConnection()
        nn, _, _ = self._nonnormal_fiber(51)
        b = _unit_fiber(11, 3)
        fwd = conn.transport(nn, b, np.eye(3))
        back = conn.transport(b, nn, np.eye(3))
        read = conn.holonomy([fwd, back])
        self.assertFalse(read.unitary)  # the GL product, flagged as such
        np.testing.assert_allclose(
            read.holonomy,
            np.asarray(fwd.raw_map) @ np.asarray(back.raw_map),
            rtol=0, atol=MACHINE)
        self.assertEqual(read.certificate.regime,
                         cob.CertificateRegime.NonNormal)

    def test_biorthogonal_frame_change_is_bifundamental(self):
        conn = obs.FiberConnection()
        rng = np.random.default_rng(37)
        nn, phi, psi = self._nonnormal_fiber(51)
        b = _unit_fiber(11, 3)
        t = _random_unitary(rng, 3)
        base = conn.transport(nn, b, t)
        g_a, g_b = _random_unitary(rng, 3), _random_unitary(rng, 3)
        nn_changed = _fiber([[51], [52], [53]], phi @ g_a, left=psi @ g_a,
                            regime="non-normal", self_adjoint=False)
        b_changed = _fiber([[11], [12], [13]], np.asarray(np.eye(3)) @ g_b)
        changed = conn.transport(nn_changed, b_changed, t)
        np.testing.assert_allclose(changed.raw_map,
                                   g_a.conj().T @ base.raw_map @ g_b,
                                   rtol=0, atol=1e-10)


# =========================================================================== #
# spacetime end-to-end (real complexes, cross-machinery consistency)
# =========================================================================== #
class TestSpacetimeEndToEnd(unittest.TestCase):
    def _phased_triangle(self, phases):
        st = _from_simplices(3, [(0, 1), (1, 2), (2, 0)])
        for (a, b), phi in phases.items():
            _set_phase(st, a, b, phi)
        return st

    def test_u1_loop_matches_wilson_loop_machinery(self):
        phases = {(0, 1): 0.4, (1, 2): 1.1, (2, 0): -0.3}
        st = self._phased_triangle(phases)
        fibers = [_tracker_fiber(st, [i], 0) for i in range(3)]
        for f in fibers:
            self.assertTrue(f.accepted)
            self.assertEqual(f.rank(), 1)
        conn = obs.FiberConnection()
        read = conn.holonomy_on_spacetime(st, fibers)
        self.assertTrue(read.closed and read.unitary)
        # H = (-1)^3 e^{i Phi_cycle} for the oriented cycle 0->1->2->0
        wl = obs.WilsonLoop(st)
        verts = sorted(st.get_vertex_list().to_vector(),
                       key=lambda v: v.get_id())
        wilson = wl.evaluate_u1_connection([verts[0], verts[1], verts[2]])
        h = complex(np.asarray(read.holonomy)[0, 0])
        self.assertAlmostEqual(abs(h), 1.0, delta=SOLVER)
        diff = cmath.phase(-h) - wilson.value.real
        diff = (diff + math.pi) % TWO_PI - math.pi
        self.assertAlmostEqual(diff, 0.0, delta=SOLVER)
        # rank-1: the determinant line IS the holonomy
        self.assertAlmostEqual(abs(read.determinant - h), 0.0, delta=MACHINE)

    def test_degree1_bowtie_reversal_is_the_adjoint(self):
        # two filled triangles sharing vertex 0; tracker bands at k = 1
        st = _from_simplices(5, [(0, 1, 2), (0, 3, 4)])
        fiber_a = _tracker_fiber(st, [0, 1, 2], 1)
        fiber_b = _tracker_fiber(st, [0, 3, 4], 1)
        self.assertTrue(fiber_a.accepted() and fiber_b.accepted())
        self.assertEqual(fiber_a.rank(), fiber_b.rank())
        r = fiber_a.rank()
        conn = obs.FiberConnection()
        fwd = conn.transport_on_spacetime(st, fiber_a, fiber_b)
        rev = conn.transport_on_spacetime(st, fiber_b, fiber_a)
        # W-self-adjoint operator: reversing the link is the adjoint
        np.testing.assert_allclose(np.asarray(rev.raw_map),
                                   np.asarray(fwd.raw_map).conj().T,
                                   rtol=0, atol=SOLVER)
        # the leakage report is honest against numpy
        m = np.asarray(fwd.raw_map)
        self.assertAlmostEqual(
            fwd.leakage, np.linalg.norm(m.conj().T @ m - np.eye(r), 2),
            delta=SOLVER)

    def test_matrix_route_equals_spacetime_route(self):
        st = self._phased_triangle({(0, 1): 0.9})
        fibers = [_tracker_fiber(st, [i], 0) for i in range(3)]
        conn = obs.FiberConnection()
        direct = conn.transport_on_spacetime(st, fibers[0], fibers[1])
        transfer = obs.FiberConnection.chain_transfer(
            st, 0, fibers[0].cell_vertices(), fibers[1].cell_vertices())
        explicit = conn.transport(fibers[0], fibers[1], transfer)
        np.testing.assert_allclose(np.asarray(direct.raw_map),
                                   np.asarray(explicit.raw_map),
                                   rtol=0, atol=MACHINE)
        self.assertEqual(direct.to_key, explicit.to_key)

    def test_cell_order_permutation_leaves_the_read_invariant(self):
        # permuting the fiber's cell rows (frame + transfer alike) is a pure
        # relabeling of the chain basis: the overlap is IDENTICAL
        rng = np.random.default_rng(43)
        t = _random_unitary(rng, 3)
        perm = [2, 0, 1]
        base_cells = [[1], [2], [3]]
        a0 = _fiber(base_cells, np.eye(3))
        a1 = _fiber([base_cells[p] for p in perm], np.eye(3)[perm, :])
        b = _unit_fiber(11, 3)
        conn = obs.FiberConnection()
        r0 = conn.transport(a0, b, t)
        r1 = conn.transport(a1, b, t[perm, :])
        np.testing.assert_allclose(r0.raw_map, r1.raw_map, rtol=0, atol=MACHINE)
        np.testing.assert_allclose(r0.unitary_map, r1.unitary_map, rtol=0,
                                   atol=MACHINE)
        self.assertEqual(r0.to_key, r1.to_key)

    def test_fiber_key_is_order_and_multiplicity_invariant(self):
        a = _fiber([[1, 2], [2, 3], [1, 3]], np.eye(3))
        b = _fiber([[3, 1], [3, 2], [2, 1]], np.eye(3))
        self.assertEqual(obs.FiberConnection.fiber_key(a),
                         obs.FiberConnection.fiber_key(b))
        c = _fiber([[4, 2], [2, 3], [1, 3]], np.eye(3))
        self.assertNotEqual(obs.FiberConnection.fiber_key(a),
                            obs.FiberConnection.fiber_key(c))

    def test_vertex_relabeling_preserves_the_invariants(self):
        phases = {(0, 1): 0.4, (1, 2): 1.1, (2, 0): -0.3}
        st = self._phased_triangle(phases)
        st2 = _from_simplices(3, [(0, 1), (1, 2), (2, 0)],
                              ids=[70, 50, 90])
        _set_phase(st2, 70, 50, 0.4)
        _set_phase(st2, 50, 90, 1.1)
        _set_phase(st2, 90, 70, -0.3)
        conn = obs.FiberConnection()
        r1 = conn.holonomy_on_spacetime(
            st, [_tracker_fiber(st, [i], 0) for i in (0, 1, 2)])
        r2 = conn.holonomy_on_spacetime(
            st2, [_tracker_fiber(st2, [i], 0) for i in (70, 50, 90)])
        # the loop scalar is identical under global relabeling
        self.assertAlmostEqual(abs(r1.normalized_trace - r2.normalized_trace),
                               0.0, delta=SOLVER)
        self.assertAlmostEqual(abs(r1.determinant - r2.determinant), 0.0,
                               delta=SOLVER)


# =========================================================================== #
# caching (the #764 AnalyticCache contract)
# =========================================================================== #
class TestCaching(unittest.TestCase):
    def _square(self):
        # 4-cycle 0-1-2-3-0: four single-vertex components at k = 0
        st = _from_simplices(4, [(0, 1), (1, 2), (2, 3), (3, 0)])
        _set_phase(st, 0, 1, 0.5)
        _set_phase(st, 2, 3, -0.2)
        fibers = [_tracker_fiber(st, [i], 0) for i in range(4)]
        return st, fibers

    def _assert_reads_equal(self, a, b):
        np.testing.assert_allclose(np.asarray(a.raw_map),
                                   np.asarray(b.raw_map), rtol=0, atol=0)
        np.testing.assert_allclose(np.asarray(a.unitary_map),
                                   np.asarray(b.unitary_map), rtol=0, atol=0)
        self.assertEqual(a.accepted, b.accepted)
        self.assertEqual(a.leakage, b.leakage)
        self.assertEqual(list(a.singular_values), list(b.singular_values))

    def test_cached_transport_equals_cold(self):
        st, fibers = self._square()
        cache = cob.AnalyticCache(st)
        conn = obs.FiberConnection()
        cold = conn.transport_on_spacetime(st, fibers[0], fibers[1])
        first = conn.transport_on_spacetime_cached(cache, st, fibers[0],
                                                fibers[1])
        hits_before = cache.hits
        second = conn.transport_on_spacetime_cached(cache, st, fibers[0],
                                                 fibers[1])
        self.assertEqual(cache.hits, hits_before + 1)
        self._assert_reads_equal(cold, first)
        self._assert_reads_equal(first, second)

    def test_two_bands_of_one_component_pair_do_not_collide(self):
        """Distinct BANDS of one component pair get distinct cache entries.

        Every band of a component restricts to the same cells, so the
        component key — the fingerprint of that cell-vertex set — is shared
        across them. Keying on it alone served the first band's read for
        every later band of the pair (measured in the #776
        incremental-versus-cold comparison: 169 of 170 transports stale).
        """
        st = _from_simplices(4, [(0, 1), (1, 2), (2, 3), (3, 0)])
        _set_phase(st, 0, 1, 0.5)
        cache = cob.AnalyticCache(st)
        conn = obs.FiberConnection()
        left = obs.SpectralFiberTracker(st).enumerate_bands([0, 1], 0)
        right = obs.SpectralFiberTracker(st).enumerate_bands([2, 3], 0)
        if len(left.fibers) < 2 or len(right.fibers) < 2:
            self.skipTest("fixture did not produce two bands per component")
        # Every band combination of the SAME component pair, in one cache:
        # each must equal its own cold recomputation, none may be served the
        # first combination's read.
        for i in range(2):
            for j in range(2):
                cached = conn.transport_on_spacetime_cached(
                    cache, st, left.fibers[i], right.fibers[j])
                cold = conn.transport_on_spacetime(st, left.fibers[i],
                                                 right.fibers[j])
                self._assert_reads_equal(cached, cold)
        # Four distinct band pairs over one component pair: four entries.
        self.assertGreaterEqual(cache.size, 4)

    def test_direction_and_pair_do_not_collide(self):
        st, fibers = self._square()
        cache = cob.AnalyticCache(st)
        conn = obs.FiberConnection()
        fwd = conn.transport_on_spacetime_cached(cache, st, fibers[0], fibers[1])
        bwd = conn.transport_on_spacetime_cached(cache, st, fibers[1], fibers[0])
        # same component union, opposite direction: distinct entries
        self.assertEqual(fwd.to_key, bwd.from_key)
        np.testing.assert_allclose(np.asarray(bwd.raw_map),
                                   np.asarray(fwd.raw_map).conj().T,
                                   rtol=0, atol=SOLVER)
        self.assertEqual(cache.size, 2)

    def test_touched_star_invalidates_only_touching_transports(self):
        st, fibers = self._square()
        cache = cob.AnalyticCache(st)
        conn = obs.FiberConnection()
        ab_before = conn.transport_on_spacetime_cached(cache, st, fibers[0],
                                                    fibers[1])
        cd_before = conn.transport_on_spacetime_cached(cache, st, fibers[2],
                                                    fibers[3])
        # metric move on edge (2,3): publish its star
        _set_phase(st, 2, 3, 0.9)
        star = cob.TouchedStar()
        star.add_changed_edge(2, 3)
        cache.publish(star)
        hits_before = cache.hits
        ab_after = conn.transport_on_spacetime_cached(cache, st, fibers[0],
                                                   fibers[1])
        # the disjoint sibling was SERVED from cache
        self.assertEqual(cache.hits, hits_before + 1)
        self._assert_reads_equal(ab_before, ab_after)
        # the touched transport was recomputed and equals a cold read
        cd_after = conn.transport_on_spacetime_cached(cache, st, fibers[2],
                                                   fibers[3])
        cd_cold = conn.transport_on_spacetime(st, fibers[2], fibers[3])
        self._assert_reads_equal(cd_after, cd_cold)
        self.assertGreater(
            np.abs(np.asarray(cd_after.raw_map)
                   - np.asarray(cd_before.raw_map)).max(), 1e-3)

    def test_loop_product_cache_cold_equals_cached(self):
        st, fibers = self._square()
        cache = cob.AnalyticCache(st)
        conn = obs.FiberConnection()
        cold = conn.holonomy_on_spacetime(st, fibers)
        first = conn.holonomy_on_spacetime_cached(cache, st, fibers)
        second = conn.holonomy_on_spacetime_cached(cache, st, fibers)
        for read in (first, second):
            np.testing.assert_allclose(np.asarray(read.holonomy),
                                       np.asarray(cold.holonomy),
                                       rtol=0, atol=0)
            self.assertEqual(read.normalized_trace, cold.normalized_trace)

    def test_loop_order_does_not_collide_in_the_cache(self):
        st, fibers = self._square()
        cache = cob.AnalyticCache(st)
        conn = obs.FiberConnection()
        loop_a = conn.holonomy_on_spacetime_cached(
            cache, st, [fibers[0], fibers[1], fibers[2], fibers[3]])
        loop_b = conn.holonomy_on_spacetime_cached(
            cache, st, [fibers[0], fibers[3], fibers[2], fibers[1]])
        cold_b = conn.holonomy_on_spacetime(
            st, [fibers[0], fibers[3], fibers[2], fibers[1]])
        np.testing.assert_allclose(np.asarray(loop_b.holonomy),
                                   np.asarray(cold_b.holonomy),
                                   rtol=0, atol=0)
        # reversed traversal is the inverse loop
        np.testing.assert_allclose(
            np.asarray(loop_b.holonomy),
            np.asarray(loop_a.holonomy).conj().T, rtol=0, atol=SOLVER)

    def test_loop_cache_invalidates_only_loops_touching_the_star(self):
        st, fibers = self._square()
        cache = cob.AnalyticCache(st)
        conn = obs.FiberConnection()
        # a loop over {0,1} only (there and back) and the full square loop
        small_before = conn.holonomy_on_spacetime_cached(
            cache, st, [fibers[0], fibers[1]])
        full_before = conn.holonomy_on_spacetime_cached(cache, st, fibers)
        _set_phase(st, 2, 3, 1.3)
        star = cob.TouchedStar()
        star.add_changed_edge(2, 3)
        cache.publish(star)
        hits_before = cache.hits
        small_after = conn.holonomy_on_spacetime_cached(
            cache, st, [fibers[0], fibers[1]])
        # untouched loop: product AND its two links all served from cache
        self.assertEqual(cache.hits, hits_before + 1)
        self.assertEqual(small_after.normalized_trace,
                         small_before.normalized_trace)
        # the touched loop recomputes and equals cold
        full_after = conn.holonomy_on_spacetime_cached(cache, st, fibers)
        full_cold = conn.holonomy_on_spacetime(st, fibers)
        np.testing.assert_allclose(np.asarray(full_after.holonomy),
                                   np.asarray(full_cold.holonomy),
                                   rtol=0, atol=0)
        self.assertGreater(abs(full_after.determinant
                               - full_before.determinant), 1e-3)

    def test_disabled_cache_still_computes_correctly(self):
        st, fibers = self._square()
        cache = cob.AnalyticCache(st)
        cache.set_enabled(False)
        conn = obs.FiberConnection()
        read = conn.transport_on_spacetime_cached(cache, st, fibers[0],
                                               fibers[1])
        cold = conn.transport_on_spacetime(st, fibers[0], fibers[1])
        self._assert_reads_equal(read, cold)


# =========================================================================== #
# checkpoint serialization (design spec section 20, `transports`)
# =========================================================================== #
class TestRecordSerialization(unittest.TestCase):
    """Round-trips through the repository's NaN-aware every-channel record
    gate (two NaNs agree; any numeric drift, shape, or status change is a
    flagged channel)."""
    _delta = staticmethod(obs.ObservableGates.report_delta)

    def setUp(self):
        self.conn = obs.FiberConnection()
        self.A = _unit_fiber(1, 3)
        self.B = _unit_fiber(11, 3)

    def test_accepted_transport_round_trip(self):
        u = _random_unitary(np.random.default_rng(2), 3)
        read = self.conn.transport(self.A, self.B, u)
        rec = read.to_record()
        back = obs.FiberTransportRead.from_record(rec)
        self.assertEqual(self._delta(rec, back.to_record()), 0.0)
        np.testing.assert_allclose(np.asarray(back.unitary_map),
                                   np.asarray(read.unitary_map),
                                   rtol=0, atol=0)
        self.assertEqual(back.to_key, read.to_key)
        self.assertEqual(back.accepted, True)
        # the rank-three quartet travels: full U(3) factor + det V (and
        # thereby the PU(3) class, which V alone determines)
        np.testing.assert_allclose(
            np.asarray(obs.FiberConnection.projective_representative(
                np.asarray(back.unitary_map), GATE)),
            np.asarray(obs.FiberConnection.projective_representative(
                np.asarray(read.unitary_map), GATE)), rtol=0, atol=0)

    def test_rejected_transport_round_trip(self):
        read = self.conn.transport(self.A, self.B, np.diag([0.5, 2.0, 1.0]))
        back = obs.FiberTransportRead.from_record(read.to_record())
        self.assertEqual(self._delta(read.to_record(), back.to_record()), 0.0)
        self.assertFalse(back.accepted)
        self.assertEqual(back.rejection_reason, read.rejection_reason)
        self.assertEqual(back.unitary_map.size, 0)
        self.assertFalse(back.certificate.holds())

    def test_lift_round_trip_carries_the_center_sector(self):
        links = [_phase_link(self.conn, self.A, self.B, TWO_PI / 3.0)
                 for _ in range(3)]
        lift = self.conn.fundamental_lift(links, GATE, 2)
        back = obs.FundamentalLiftRead.from_record(lift.to_record())
        self.assertEqual(self._delta(lift.to_record(), back.to_record()), 0.0)
        self.assertEqual(back.center_sector, 1)
        self.assertEqual(back.base_branch, 2)
        np.testing.assert_allclose(np.asarray(back.lift),
                                   np.asarray(lift.lift), rtol=0, atol=0)

    def test_winding_round_trip_known_and_unknown(self):
        family = [_phase_link(self.conn, self.A, self.B, TWO_PI * k / 8)
                  for k in range(8)]
        known = self.conn.closed_family_winding(family)
        back = obs.DeterminantWindingRead.from_record(known.to_record())
        self.assertEqual(self._delta(known.to_record(), back.to_record()), 0.0)
        self.assertEqual(back.winding, 1)
        # unknown stays unknown — never rehydrated as zero
        unknown = self.conn.open_segment_winding(family,
                                               obs.WindingClosureSpec())
        back_u = obs.DeterminantWindingRead.from_record(unknown.to_record())
        self.assertIsNone(back_u.winding)
        self.assertEqual(back_u.winding_closure, "none")

    def test_unknown_schema_version_is_rejected(self):
        read = self.conn.transport(self.A, self.B, np.eye(3))
        rec = read.to_record()
        rec["schema_version"] = 99
        with self.assertRaises(ValueError):
            obs.FiberTransportRead.from_record(rec)
        with self.assertRaises(ValueError):
            obs.FundamentalLiftRead.from_record(rec)


if __name__ == "__main__":
    unittest.main()



# =========================================================================== #
# whitepaper line 891 — polar normalization must not conceal a bad fiber
# =========================================================================== #
class TestSpecHolonomyDiagnostics(unittest.TestCase):
    """The spec requires every ACCEPTED holonomy to be reported together
    with the leakage, the rank/singular-value evidence, the endpoint band
    gaps and the frame conditioning of the links it was built from.  Polar
    normalization discards each factor's defect, so the loop must carry it
    forward or a barely-accepted chain reads as clean."""

    def _links(self):
        conn = obs.FiberConnection()
        a, b, c = _unit_fiber(1, 3), _unit_fiber(2, 3), _unit_fiber(3, 3)
        return [conn.transport(a, b, _random_unitary(np.random.default_rng(4), 3)),
                conn.transport(b, c, _random_unitary(np.random.default_rng(5), 3)),
                conn.transport(c, a, _random_unitary(np.random.default_rng(6), 3))]

    def test_every_required_diagnostic_is_measured(self):
        links = self._links()
        loop = obs.FiberConnection().holonomy(links)
        for name in ("max_leakage", "min_endpoint_gap",
                     "max_frame_condition_number", "min_singular_value"):
            self.assertFalse(math.isnan(getattr(loop, name)),
                             name + " must be measured, never NaN, on an "
                                    "accepted holonomy")
        self.assertGreaterEqual(loop.min_numerical_rank, 1)

    def test_each_diagnostic_is_the_worst_case_over_the_links(self):
        links = self._links()
        loop = obs.FiberConnection().holonomy(links)
        self.assertAlmostEqual(loop.max_leakage,
                               max(l.leakage for l in links), delta=MACHINE)
        self.assertAlmostEqual(loop.max_frame_condition_number,
                               max(l.frame_condition_number for l in links),
                               delta=MACHINE)
        self.assertAlmostEqual(
            loop.min_endpoint_gap,
            min(min(l.to_gap, l.from_gap) for l in links), delta=MACHINE)
        self.assertEqual(loop.min_numerical_rank,
                         min(l.numerical_rank for l in links))


# --------------------------------------------------------------------------- #
# the exactness contract's anchor gate on the colour-specific kernels
# --------------------------------------------------------------------------- #
class TestAnchorGatedColourKernels(unittest.TestCase):
    """The contract says the exact 3x3 colour determinants and the centre lift
    are used "only after a rank-three band passes its triangle-anchor
    certificate". Rank three and an accepted transport are NOT sufficient, so
    an ungated call must refuse rather than emit a colour datum."""

    def setUp(self):
        self.conn = obs.FiberConnection()
        self.link = self.conn.transport(_unit_fiber(1, 3), _unit_fiber(11, 3),
                                        np.eye(3))
        self.assertTrue(self.link.accepted)
        self.assertEqual(self.link.rank, 3)

    def test_default_constructed_gate_is_closed(self):
        gate = obs.AnchorGate()
        self.assertFalse(gate.accepted)
        self.assertIn("absent", gate.refusal_reason)

    def test_projective_representative_refuses_without_an_anchor(self):
        with self.assertRaises(ValueError) as caught:
            obs.FiberConnection.projective_representative(np.eye(3),
                                                         obs.AnchorGate())
        self.assertIn("triangle-anchor", str(caught.exception))

    def test_projective_representative_admits_an_anchored_band(self):
        rep = np.asarray(
            obs.FiberConnection.projective_representative(np.eye(3), GATE))
        np.testing.assert_allclose(rep, np.eye(3), rtol=0, atol=MACHINE)

    def test_fundamental_lift_refuses_without_an_anchor(self):
        lift = self.conn.fundamental_lift([self.link], obs.AnchorGate())
        self.assertFalse(lift.valid)
        self.assertIn("triangle-anchor", lift.invalid_reason)

    def test_fundamental_lift_admits_an_anchored_band(self):
        lift = self.conn.fundamental_lift([self.link], GATE)
        self.assertTrue(lift.valid)
        self.assertEqual(lift.rank, 3)

    def test_the_gate_is_checked_before_rank(self):
        # An unanchored GENERIC-rank family must name the anchor, not the rank:
        # the contract gates the kernel, so that refusal comes first.
        link = self.conn.transport(_unit_fiber(31, 2), _unit_fiber(41, 2),
                                    np.eye(2))
        lift = self.conn.fundamental_lift([link], obs.AnchorGate())
        self.assertFalse(lift.valid)
        self.assertIn("triangle-anchor", lift.invalid_reason)


class TestAnchorAcceptancePredicate(unittest.TestCase):
    """One predicate, shared by the quark verdict and the colour kernels."""

    def _profile(self):
        return _anchored_profile()

    def test_an_isolated_face_reports_unknown_coherence_and_is_refused(self):
        # One face has no overlap partner, so its determinant-phase coherence
        # is UNKNOWN. Unknown is not "coherent enough": the predicate must
        # refuse it rather than let a NaN slip through a >= comparison.
        weights = np.array([2.0, 0.5, 1.25])
        rng = np.random.default_rng(61)
        frame = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
        band = obs.ColorAnchor.orthonormalize_frame(frame, weights)
        anchor = obs.ColorAnchor([obs.OrientedTriangle([0, 1, 2], [1, 1, 1])])
        profile = anchor.evaluate(band, weights)
        self.assertTrue(math.isnan(profile.phase_coherence))
        self.assertFalse(obs.ColorAnchor.accepts(profile))
        self.assertFalse(obs.ColorAnchor.gate_for(profile).accepted)

    def test_a_measured_anchor_passes(self):
        profile = self._profile()
        self.assertTrue(obs.ColorAnchor.accepts(profile))
        self.assertTrue(obs.ColorAnchor.gate_for(profile).accepted)

    def test_missing_evidence_is_refused_not_scored_zero(self):
        # A default-constructed profile declared no weighting at all: that is
        # ABSENT evidence, and it must be named as such.
        gate = obs.ColorAnchor.gate_for(obs.AnchorProfile())
        self.assertFalse(gate.accepted)
        self.assertIn("absent", gate.refusal_reason)

    def test_a_score_below_the_floor_is_refused_and_named(self):
        profile = self._profile()
        gate = obs.ColorAnchor.gate_for(profile, min_score=1.5)
        self.assertFalse(gate.accepted)
        self.assertIn("atlas score", gate.refusal_reason)

    def test_a_coherence_below_the_floor_is_refused_and_named(self):
        profile = self._profile()
        gate = obs.ColorAnchor.gate_for(profile, min_phase_coherence=1.5)
        self.assertFalse(gate.accepted)
        self.assertIn("coherence", gate.refusal_reason)

    def test_the_gate_records_what_admitted_it(self):
        profile = self._profile()
        gate = obs.ColorAnchor.gate_for(profile)
        self.assertEqual(gate.weighting_id, profile.weighting_id)
        self.assertAlmostEqual(gate.score, profile.score, delta=MACHINE)
        self.assertAlmostEqual(gate.phase_coherence, profile.phase_coherence,
                               delta=MACHINE)
        self.assertEqual(gate.refusal_reason, "")
