"""Tests for the QuantumSimplex factories, QuantumVertex, and KI bindings."""

from __future__ import annotations

import math
import unittest

import numpy as np
import pytest


try:
    from tessera import (
        Foliation,
        Edge,
        Metric,
        Signature,
        SignatureType,
        Spacetime,
        SpacetimeType,
    )
    from tessera.quantum import (
        QuantumSimplex,
        QuantumSimplexPosition as P,
        QuantumVertex,
        create_quantum_vertex,
        koashi_imoto_decompose,
        mutual_information,
        partial_trace_a,
        partial_trace_b,
    )
    _has_quantum = True
except ImportError:
    _has_quantum = False


pytestmark = pytest.mark.skipif(
    not _has_quantum,
    reason="tessera.quantum.QuantumSimplex not built (needs TESSERA_QUANTUM=1)",
)


def fresh_spacetime():
    metric = Metric(True, Signature(4, SignatureType.Euclidean))
    return Spacetime(metric, SpacetimeType.REGGE, 1.0, 1.0,
                     Foliation.NONE, None)


def bell_phi_plus():
    """ρ_AB = |Φ+⟩⟨Φ+| in (A ⊗ B) basis."""
    rho = np.zeros((4, 4), dtype=complex)
    rho[0, 0] = 0.5
    rho[0, 3] = 0.5
    rho[3, 0] = 0.5
    rho[3, 3] = 0.5
    return rho


def product_state(rhoA, rhoB):
    """ρ_A ⊗ ρ_B in (A ⊗ B) ordering."""
    dA = rhoA.shape[0]
    dB = rhoB.shape[0]
    out = np.zeros((dA * dB, dA * dB), dtype=complex)
    for i in range(dA):
        for j in range(dA):
            for a in range(dB):
                for b in range(dB):
                    out[i * dB + a, j * dB + b] = rhoA[i, j] * rhoB[a, b]
    return out


def half_I(d=2):
    return np.eye(d) / d


def vertex_at_position(s, p):
    """Return the QuantumVertex at the given QuantumSimplexPosition in
    the cell. Walks the underlying Simplex's vertex list."""
    return s.get_vertices()[int(p)]


def edge_squared_length(s, p, q):
    """Squared edge length for the (p, q) edge of the cell."""
    verts = s.get_vertices()
    u, v = verts[int(p)], verts[int(q)]
    for e in s.get_edges():
        src = e.get_source()
        tgt = e.get_target()
        if (src.get_id() == u.get_id() and tgt.get_id() == v.get_id()) \
                or (src.get_id() == v.get_id() and tgt.get_id() == u.get_id()):
            return (e.get_length() * e.get_length()).real
    raise KeyError(f"no edge between {p} and {q}")


class TestPartialTraceAndMI(unittest.TestCase):
    """Standalone KI helpers behave on hand-calculable inputs."""

    def test_partial_trace_of_bell(self):
        rho = bell_phi_plus()
        rho_A = partial_trace_b(rho, 2, 2)
        rho_B = partial_trace_a(rho, 2, 2)
        np.testing.assert_allclose(rho_A, half_I(), atol=1e-12)
        np.testing.assert_allclose(rho_B, half_I(), atol=1e-12)

    def test_mutual_information_bell(self):
        rho = bell_phi_plus()
        I = mutual_information(rho, 2, 2)
        self.assertAlmostEqual(I, 2 * math.log(2.0), places=10)

    def test_mutual_information_product_is_zero(self):
        rhoA = np.diag([0.6, 0.4]).astype(complex)
        rhoB = np.diag([0.3, 0.7]).astype(complex)
        rho = product_state(rhoA, rhoB)
        I = mutual_information(rho, 2, 2)
        self.assertAlmostEqual(I, 0.0, places=10)

    def test_mutual_information_marginal_overload(self):
        rhoA = np.diag([0.6, 0.4]).astype(complex)
        rhoB = np.diag([0.3, 0.7]).astype(complex)
        rho = product_state(rhoA, rhoB)
        I = mutual_information(rho, rhoA, rhoB)
        self.assertAlmostEqual(I, 0.0, places=10)


class TestKoashiImotoDecompose(unittest.TestCase):
    def test_bell_decomposition(self):
        rho = bell_phi_plus()
        r = koashi_imoto_decompose(rho, 2, 2)
        self.assertEqual(len(r.blocks), 1)
        blk = r.blocks[0]
        self.assertEqual(blk.dim_left_a, 2)
        self.assertEqual(blk.dim_left_b, 2)
        self.assertEqual(blk.dim_right_a, 1)
        self.assertEqual(blk.dim_right_b, 1)
        self.assertAlmostEqual(blk.weight, 1.0, places=10)

    def test_product_decomposition_is_trivial_block(self):
        rhoA = np.diag([0.6, 0.4]).astype(complex)
        rhoB = np.diag([0.3, 0.7]).astype(complex)
        rho = product_state(rhoA, rhoB)
        r = koashi_imoto_decompose(rho, 2, 2)
        self.assertEqual(len(r.blocks), 1)
        blk = r.blocks[0]
        self.assertEqual(blk.dim_left_a, 1)
        self.assertEqual(blk.dim_left_b, 1)
        self.assertEqual(blk.dim_right_a, 2)
        self.assertEqual(blk.dim_right_b, 2)

    def test_marginal_overload(self):
        rho = bell_phi_plus()
        rhoA = partial_trace_b(rho, 2, 2)
        rhoB = partial_trace_a(rho, 2, 2)
        r1 = koashi_imoto_decompose(rho, 2, 2)
        r2 = koashi_imoto_decompose(rho, rhoA, rhoB)
        self.assertEqual(len(r1.blocks), len(r2.blocks))
        np.testing.assert_allclose(r1.sigma, r2.sigma, atol=1e-10)


class TestQuantumVertex(unittest.TestCase):
    def test_create_in_spacetime(self):
        st = fresh_spacetime()
        rho = half_I(2)
        qv = create_quantum_vertex(st, rho)
        self.assertEqual(st.get_vertex_count(), 1)
        self.assertEqual(qv.state_dim(), 2)
        np.testing.assert_allclose(qv.get_state(), rho, atol=1e-12)

    def test_van_raamsdonk_distance_of_an_uncorrelated_pair_is_inf(self):
        # Two systems with no mutual information are unrelated, so the law
        # diverges. The KI factory asks Edge for this with the floor opted
        # out (epsilon = 0), which is what makes +∞ reachable.
        self.assertEqual(
            Edge.van_raamsdonk_length(0.0, 2.0 * math.log(2.0), 0.0),
            math.inf)


class TestQuantumSimplexFromExplicitJoint(unittest.TestCase):
    def setUp(self):
        self.spacetime = fresh_spacetime()
        self.rho = bell_phi_plus()
        self.i_max = 2.0 * math.log(2.0)
        self.qva = create_quantum_vertex(
            self.spacetime, half_I(2).astype(complex))
        self.qvb = create_quantum_vertex(
            self.spacetime, half_I(2).astype(complex))
        self.s = QuantumSimplex.from_explicit_joint(
            self.spacetime, self.qva, self.qvb, self.rho, self.i_max)

    def test_five_vertices_added(self):
        self.assertEqual(self.spacetime.get_vertex_count(), 5)

    def test_simplex_returned(self):
        # The factory returns a regular mesh.Simplex.
        self.assertIsNotNone(self.s)
        self.assertEqual(len(self.s.get_vertices()), 5)

    def test_marginals_are_half_I(self):
        np.testing.assert_allclose(
            vertex_at_position(self.s, P.A).get_state(),
            half_I(), atol=1e-12)
        np.testing.assert_allclose(
            vertex_at_position(self.s, P.B).get_state(),
            half_I(), atol=1e-12)

    def test_sigma_is_bell_state(self):
        sigma = vertex_at_position(self.s, P.Sigma).get_state()
        self.assertEqual(sigma.shape, (4, 4))
        self.assertAlmostEqual(sigma[0, 0].real, 0.5, places=10)
        self.assertAlmostEqual(sigma[0, 3].real, 0.5, places=10)

    def test_tails_are_trivial(self):
        self.assertEqual(
            vertex_at_position(self.s, P.APrime).get_state().shape,
            (1, 1))
        self.assertEqual(
            vertex_at_position(self.s, P.BPrime).get_state().shape,
            (1, 1))

    def test_ab_edge_length_squared_is_zero(self):
        # Bell joint at MI = iMax → d_VR = 0 → squaredLength = 0.
        self.assertAlmostEqual(
            edge_squared_length(self.s, P.A, P.B), 0.0, places=10)

    def test_product_edges_have_large_length_squared(self):
        # All non-(A, B) edges use the product joint → I = 0 →
        # d_VR = +∞ → squaredLength = +∞.
        for p, q in [(P.A, P.Sigma), (P.B, P.Sigma),
                     (P.A, P.APrime), (P.B, P.BPrime),
                     (P.APrime, P.BPrime), (P.APrime, P.Sigma),
                     (P.BPrime, P.Sigma)]:
            ls = edge_squared_length(self.s, p, q)
            self.assertTrue(math.isinf(ls) or ls > 900.0,
                            msg=f"({p}, {q}) squaredLength = {ls}")


class TestQuantumSimplexFromSchmidtPurification(unittest.TestCase):
    def test_matched_diag_marginals(self):
        st = fresh_spacetime()
        marg = np.diag([0.7, 0.3]).astype(complex)
        qva = create_quantum_vertex(st, marg)
        qvb = create_quantum_vertex(st, marg.copy())
        i_max = 2.0 * math.log(2.0)
        s = QuantumSimplex.from_schmidt_purification(st, qva, qvb, i_max)
        # MI = 2 H(p) at the AB edge.
        expected_mi = -2.0 * (0.7 * math.log(0.7) + 0.3 * math.log(0.3))
        expected_dvr = -math.log(expected_mi / i_max)
        ls = edge_squared_length(s, P.A, P.B)
        self.assertAlmostEqual(math.sqrt(ls), expected_dvr, places=6)

    def test_mismatched_spectra_throws(self):
        st = fresh_spacetime()
        qva = create_quantum_vertex(st, np.diag([0.6, 0.4]).astype(complex))
        qvb = create_quantum_vertex(st, np.diag([0.8, 0.2]).astype(complex))
        with self.assertRaises(Exception):
            QuantumSimplex.from_schmidt_purification(st, qva, qvb, 1.0)


class TestQuantumSimplexFromClassicalCorrelation(unittest.TestCase):
    def test_matched_diag_marginals(self):
        st = fresh_spacetime()
        marg = np.diag([0.7, 0.3]).astype(complex)
        qva = create_quantum_vertex(st, marg)
        qvb = create_quantum_vertex(st, marg.copy())
        i_max = 2.0 * math.log(2.0)
        s = QuantumSimplex.from_classical_correlation(st, qva, qvb, i_max)
        expected_mi = -(0.7 * math.log(0.7) + 0.3 * math.log(0.3))
        expected_dvr = -math.log(expected_mi / i_max)
        ls = edge_squared_length(s, P.A, P.B)
        self.assertAlmostEqual(math.sqrt(ls), expected_dvr, places=6)


class TestQuantumSimplexFromTargetMI(unittest.TestCase):
    def test_zero_target_gives_inf_dvr(self):
        st = fresh_spacetime()
        marg = np.diag([0.7, 0.3]).astype(complex)
        qva = create_quantum_vertex(st, marg)
        qvb = create_quantum_vertex(st, marg.copy())
        s = QuantumSimplex.from_target_mutual_information(
            st, qva, qvb, 0.0, 2.0 * math.log(2.0))
        # MI = 0 → d_VR = +∞ → squaredLength = +∞ (or extremely large).
        ls = edge_squared_length(s, P.A, P.B)
        self.assertTrue(math.isinf(ls) or ls > 900.0)

    def test_intermediate_target_hits(self):
        st = fresh_spacetime()
        marg = np.diag([0.7, 0.3]).astype(complex)
        qva = create_quantum_vertex(st, marg)
        qvb = create_quantum_vertex(st, marg.copy())
        target = 0.3
        i_max = 2.0 * math.log(2.0)
        s = QuantumSimplex.from_target_mutual_information(
            st, qva, qvb, target, i_max)
        ls = edge_squared_length(s, P.A, P.B)
        recovered_mi = i_max * math.exp(-math.sqrt(ls))
        self.assertAlmostEqual(recovered_mi, target, places=3)

    def test_target_above_max_throws(self):
        st = fresh_spacetime()
        marg = np.diag([0.7, 0.3]).astype(complex)
        qva = create_quantum_vertex(st, marg)
        qvb = create_quantum_vertex(st, marg.copy())
        max_mi = -2.0 * (0.7 * math.log(0.7) + 0.3 * math.log(0.3))
        with self.assertRaises(Exception):
            QuantumSimplex.from_target_mutual_information(
                st, qva, qvb, max_mi + 1.0, 2.0 * math.log(2.0))


class TestQuantumSimplexInvalidInputs(unittest.TestCase):
    def test_rejects_zero_imax(self):
        st = fresh_spacetime()
        qva = create_quantum_vertex(st, half_I(2).astype(complex))
        qvb = create_quantum_vertex(st, half_I(2).astype(complex))
        with self.assertRaises(Exception):
            QuantumSimplex.from_explicit_joint(
                st, qva, qvb, bell_phi_plus(), 0.0)

    def test_rejects_dim_mismatch(self):
        st = fresh_spacetime()
        qva = create_quantum_vertex(st, half_I(2).astype(complex))
        qvb = create_quantum_vertex(st, half_I(3).astype(complex))
        with self.assertRaises(Exception):
            QuantumSimplex.from_explicit_joint(
                st, qva, qvb, bell_phi_plus(), 1.0)


if __name__ == "__main__":
    unittest.main()
