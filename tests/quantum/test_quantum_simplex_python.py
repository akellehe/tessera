"""Tests for the QuantumSimplex factories, QuantumVertex, and KI bindings."""

from __future__ import annotations

import math
import unittest

import numpy as np
import pytest


try:
    from tessera import (
        Foliation,
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
        createQuantumVertex,
        koashiImotoDecompose,
        mutualInformation,
        partialTraceA,
        partialTraceB,
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
    return s.getVertices()[int(p)]


def edge_squared_length(s, p, q):
    """Edge squaredLength_ for the (p, q) edge of the cell."""
    verts = s.getVertices()
    u, v = verts[int(p)], verts[int(q)]
    for e in s.getEdges():
        src = e.getSource()
        tgt = e.getTarget()
        if (src.getId() == u.getId() and tgt.getId() == v.getId()) \
                or (src.getId() == v.getId() and tgt.getId() == u.getId()):
            return (e.getLength() * e.getLength()).real
    raise KeyError(f"no edge between {p} and {q}")


class TestPartialTraceAndMI(unittest.TestCase):
    """Standalone KI helpers behave on hand-calculable inputs."""

    def test_partial_trace_of_bell(self):
        rho = bell_phi_plus()
        rho_A = partialTraceB(rho, 2, 2)
        rho_B = partialTraceA(rho, 2, 2)
        np.testing.assert_allclose(rho_A, half_I(), atol=1e-12)
        np.testing.assert_allclose(rho_B, half_I(), atol=1e-12)

    def test_mutual_information_bell(self):
        rho = bell_phi_plus()
        I = mutualInformation(rho, 2, 2)
        self.assertAlmostEqual(I, 2 * math.log(2.0), places=10)

    def test_mutual_information_product_is_zero(self):
        rhoA = np.diag([0.6, 0.4]).astype(complex)
        rhoB = np.diag([0.3, 0.7]).astype(complex)
        rho = product_state(rhoA, rhoB)
        I = mutualInformation(rho, 2, 2)
        self.assertAlmostEqual(I, 0.0, places=10)

    def test_mutual_information_marginal_overload(self):
        rhoA = np.diag([0.6, 0.4]).astype(complex)
        rhoB = np.diag([0.3, 0.7]).astype(complex)
        rho = product_state(rhoA, rhoB)
        I = mutualInformation(rho, rhoA, rhoB)
        self.assertAlmostEqual(I, 0.0, places=10)


class TestKoashiImotoDecompose(unittest.TestCase):
    def test_bell_decomposition(self):
        rho = bell_phi_plus()
        r = koashiImotoDecompose(rho, 2, 2)
        self.assertEqual(len(r.blocks), 1)
        blk = r.blocks[0]
        self.assertEqual(blk.dimLeftA, 2)
        self.assertEqual(blk.dimLeftB, 2)
        self.assertEqual(blk.dimRightA, 1)
        self.assertEqual(blk.dimRightB, 1)
        self.assertAlmostEqual(blk.weight, 1.0, places=10)

    def test_product_decomposition_is_trivial_block(self):
        rhoA = np.diag([0.6, 0.4]).astype(complex)
        rhoB = np.diag([0.3, 0.7]).astype(complex)
        rho = product_state(rhoA, rhoB)
        r = koashiImotoDecompose(rho, 2, 2)
        self.assertEqual(len(r.blocks), 1)
        blk = r.blocks[0]
        self.assertEqual(blk.dimLeftA, 1)
        self.assertEqual(blk.dimLeftB, 1)
        self.assertEqual(blk.dimRightA, 2)
        self.assertEqual(blk.dimRightB, 2)

    def test_marginal_overload(self):
        rho = bell_phi_plus()
        rhoA = partialTraceB(rho, 2, 2)
        rhoB = partialTraceA(rho, 2, 2)
        r1 = koashiImotoDecompose(rho, 2, 2)
        r2 = koashiImotoDecompose(rho, rhoA, rhoB)
        self.assertEqual(len(r1.blocks), len(r2.blocks))
        np.testing.assert_allclose(r1.sigma, r2.sigma, atol=1e-10)


class TestQuantumVertex(unittest.TestCase):
    def test_create_in_spacetime(self):
        st = fresh_spacetime()
        rho = half_I(2)
        qv = createQuantumVertex(st, rho)
        self.assertEqual(st.getVertexCount(), 1)
        self.assertEqual(qv.stateDim(), 2)
        np.testing.assert_allclose(qv.getState(), rho, atol=1e-12)

    def test_van_raamsdonk_distance_of_an_uncorrelated_pair_is_inf(self):
        # Two systems with no mutual information are unrelated, so the law
        # diverges. The KI factory asks Edge for this with the floor opted
        # out (epsilon = 0), which is what makes +∞ reachable.
        self.assertEqual(
            tessera.Edge.vanRaamsdonkLength(0.0, 2.0 * math.log(2.0), 0.0),
            math.inf)


class TestQuantumSimplexFromExplicitJoint(unittest.TestCase):
    def setUp(self):
        self.spacetime = fresh_spacetime()
        self.rho = bell_phi_plus()
        self.i_max = 2.0 * math.log(2.0)
        self.qva = createQuantumVertex(
            self.spacetime, half_I(2).astype(complex))
        self.qvb = createQuantumVertex(
            self.spacetime, half_I(2).astype(complex))
        self.s = QuantumSimplex.fromExplicitJoint(
            self.spacetime, self.qva, self.qvb, self.rho, self.i_max)

    def test_five_vertices_added(self):
        self.assertEqual(self.spacetime.getVertexCount(), 5)

    def test_simplex_returned(self):
        # The factory returns a regular mesh.Simplex.
        self.assertIsNotNone(self.s)
        self.assertEqual(len(self.s.getVertices()), 5)

    def test_marginals_are_half_I(self):
        np.testing.assert_allclose(
            vertex_at_position(self.s, P.A).getState(),
            half_I(), atol=1e-12)
        np.testing.assert_allclose(
            vertex_at_position(self.s, P.B).getState(),
            half_I(), atol=1e-12)

    def test_sigma_is_bell_state(self):
        sigma = vertex_at_position(self.s, P.Sigma).getState()
        self.assertEqual(sigma.shape, (4, 4))
        self.assertAlmostEqual(sigma[0, 0].real, 0.5, places=10)
        self.assertAlmostEqual(sigma[0, 3].real, 0.5, places=10)

    def test_tails_are_trivial(self):
        self.assertEqual(
            vertex_at_position(self.s, P.APrime).getState().shape,
            (1, 1))
        self.assertEqual(
            vertex_at_position(self.s, P.BPrime).getState().shape,
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
        qva = createQuantumVertex(st, marg)
        qvb = createQuantumVertex(st, marg.copy())
        i_max = 2.0 * math.log(2.0)
        s = QuantumSimplex.fromSchmidtPurification(st, qva, qvb, i_max)
        # MI = 2 H(p) at the AB edge.
        expected_mi = -2.0 * (0.7 * math.log(0.7) + 0.3 * math.log(0.3))
        expected_dvr = -math.log(expected_mi / i_max)
        ls = edge_squared_length(s, P.A, P.B)
        self.assertAlmostEqual(math.sqrt(ls), expected_dvr, places=6)

    def test_mismatched_spectra_throws(self):
        st = fresh_spacetime()
        qva = createQuantumVertex(st, np.diag([0.6, 0.4]).astype(complex))
        qvb = createQuantumVertex(st, np.diag([0.8, 0.2]).astype(complex))
        with self.assertRaises(Exception):
            QuantumSimplex.fromSchmidtPurification(st, qva, qvb, 1.0)


class TestQuantumSimplexFromClassicalCorrelation(unittest.TestCase):
    def test_matched_diag_marginals(self):
        st = fresh_spacetime()
        marg = np.diag([0.7, 0.3]).astype(complex)
        qva = createQuantumVertex(st, marg)
        qvb = createQuantumVertex(st, marg.copy())
        i_max = 2.0 * math.log(2.0)
        s = QuantumSimplex.fromClassicalCorrelation(st, qva, qvb, i_max)
        expected_mi = -(0.7 * math.log(0.7) + 0.3 * math.log(0.3))
        expected_dvr = -math.log(expected_mi / i_max)
        ls = edge_squared_length(s, P.A, P.B)
        self.assertAlmostEqual(math.sqrt(ls), expected_dvr, places=6)


class TestQuantumSimplexFromTargetMI(unittest.TestCase):
    def test_zero_target_gives_inf_dvr(self):
        st = fresh_spacetime()
        marg = np.diag([0.7, 0.3]).astype(complex)
        qva = createQuantumVertex(st, marg)
        qvb = createQuantumVertex(st, marg.copy())
        s = QuantumSimplex.fromTargetMutualInformation(
            st, qva, qvb, 0.0, 2.0 * math.log(2.0))
        # MI = 0 → d_VR = +∞ → squaredLength = +∞ (or extremely large).
        ls = edge_squared_length(s, P.A, P.B)
        self.assertTrue(math.isinf(ls) or ls > 900.0)

    def test_intermediate_target_hits(self):
        st = fresh_spacetime()
        marg = np.diag([0.7, 0.3]).astype(complex)
        qva = createQuantumVertex(st, marg)
        qvb = createQuantumVertex(st, marg.copy())
        target = 0.3
        i_max = 2.0 * math.log(2.0)
        s = QuantumSimplex.fromTargetMutualInformation(
            st, qva, qvb, target, i_max)
        ls = edge_squared_length(s, P.A, P.B)
        recovered_mi = i_max * math.exp(-math.sqrt(ls))
        self.assertAlmostEqual(recovered_mi, target, places=3)

    def test_target_above_max_throws(self):
        st = fresh_spacetime()
        marg = np.diag([0.7, 0.3]).astype(complex)
        qva = createQuantumVertex(st, marg)
        qvb = createQuantumVertex(st, marg.copy())
        max_mi = -2.0 * (0.7 * math.log(0.7) + 0.3 * math.log(0.3))
        with self.assertRaises(Exception):
            QuantumSimplex.fromTargetMutualInformation(
                st, qva, qvb, max_mi + 1.0, 2.0 * math.log(2.0))


class TestQuantumSimplexInvalidInputs(unittest.TestCase):
    def test_rejects_zero_imax(self):
        st = fresh_spacetime()
        qva = createQuantumVertex(st, half_I(2).astype(complex))
        qvb = createQuantumVertex(st, half_I(2).astype(complex))
        with self.assertRaises(Exception):
            QuantumSimplex.fromExplicitJoint(
                st, qva, qvb, bell_phi_plus(), 0.0)

    def test_rejects_dim_mismatch(self):
        st = fresh_spacetime()
        qva = createQuantumVertex(st, half_I(2).astype(complex))
        qvb = createQuantumVertex(st, half_I(3).astype(complex))
        with self.assertRaises(Exception):
            QuantumSimplex.fromExplicitJoint(
                st, qva, qvb, bell_phi_plus(), 1.0)


if __name__ == "__main__":
    unittest.main()
