# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""The Kronecker-sum/Kuenneth rule for actual product complexes (#764).

L_{AxB} = L_A (x) I + I (x) L_B is algebraically exact as a matrix identity
(and its spectrum is exactly the pairwise sums), but as a statement about a
COMPLEX it holds only for an actual product cell structure: at degree zero, a
weighted 1-skeleton that IS the Cartesian product of the factors'.
productCertificate grants the rule on a hand-built Cartesian product (with
complex weights and phases) and REFUSES the staircase SimplicialProduct,
whose diagonal edges break the identity. Matching is by vertex identifier,
never an imposed order: relabeled products and shuffled pairings certify
identically.
"""

import cmath
import unittest

import numpy as np

import tessera

cob = tessera.cobordism


def _graph(num_vertices, edges, vertex_ids=None):
    """Spacetime holding an explicit weighted graph. `edges` is a list of
    (src, tgt, squared_length, phase) with src/tgt indexing into
    `vertex_ids` (default identity ids 0..n-1)."""
    sig = tessera.Signature(4, tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    st = tessera.Spacetime(metric, tessera.HERMITIAN_WEIGHTED, 1.0, 1.0,
                           tessera.PREFERRED, tessera.Toroid())
    ids = vertex_ids if vertex_ids is not None else list(range(num_vertices))
    verts = {i: st.createVertex(ids[i]) for i in range(num_vertices)}
    for src, tgt, _, _ in edges:
        st.createSimplex([verts[src], verts[tgt]])
    by_pair = {}
    for e in st.getEdgeList().toVector():
        key = (e.getSource().getId(), e.getTarget().getId())
        by_pair[key] = e
    for src, tgt, squared_length, phase in edges:
        key = (ids[src], ids[tgt])
        reverse = (ids[tgt], ids[src])
        if key in by_pair:
            edge = by_pair[key]
            edge.setLength(cmath.sqrt(complex(squared_length)))
            edge.setPhase(phase)
        else:
            edge = by_pair[reverse]
            edge.setLength(cmath.sqrt(complex(squared_length)))
            edge.setPhase(-phase)
    return st


# Factor A: path 0-1-2 with distinct weights and one nonzero phase.
_A_EDGES = [(0, 1, 1.0, 0.0), (1, 2, 2.25, 0.4)]
# Factor B: edge pair 0-1, 1-2? Keep B a single edge plus a second vertex
# component-free: B = path 0-1 with weight 0.5 and phase -0.2.
_B_EDGES = [(0, 1, 0.5, -0.2)]


def _cartesian_product_edges(a_vertices, a_edges, b_vertices, b_edges,
                             pair_to_index):
    """Cartesian-product graph edges over pair_to_index[(u, v)] -> product
    vertex index: A-edges replicated at fixed v, B-edges at fixed u, with
    the factor edge's weight and phase."""
    edges = []
    for (u, up, w, phi) in a_edges:
        for v in range(b_vertices):
            edges.append((pair_to_index[(u, v)], pair_to_index[(up, v)], w,
                          phi))
    for (v, vp, w, phi) in b_edges:
        for u in range(a_vertices):
            edges.append((pair_to_index[(u, v)], pair_to_index[(u, vp)], w,
                          phi))
    return edges


class TestKroneckerSumMatrix(unittest.TestCase):
    def test_matches_numpy_kron(self):
        rng = np.random.default_rng(79)
        a = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
        b = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
        got = np.array(cob.SpacetimeComposition.kroneckerSum(
            [complex(z) for z in a.reshape(-1)], 3,
            [complex(z) for z in b.reshape(-1)], 4)).reshape(12, 12)
        expected = np.kron(a, np.eye(4)) + np.kron(np.eye(3), b)
        np.testing.assert_array_equal(got, expected)

    def test_pairwise_spectrum_is_the_kronecker_sum_spectrum(self):
        rng = np.random.default_rng(83)
        a = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
        b = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
        spec_a = np.linalg.eigvals(a)
        spec_b = np.linalg.eigvals(b)
        pairwise = cob.KuennethProduct.pairwiseSpectrum(
            [complex(z) for z in spec_a], [complex(z) for z in spec_b])
        dense = sorted(np.linalg.eigvals(np.kron(a, np.eye(4)) +
                                         np.kron(np.eye(3), b)),
                       key=lambda z: (z.real, z.imag))
        np.testing.assert_allclose(pairwise, dense, rtol=1e-9, atol=1e-9)


class TestProductCertificate(unittest.TestCase):
    def _pairing(self, pair_to_id):
        return [(pid, u, v) for (u, v), pid in pair_to_id.items()]

    def test_actual_cartesian_product_certifies(self):
        a = _graph(3, _A_EDGES)
        b = _graph(2, _B_EDGES)
        pair_to_index = {(u, v): 2 * u + v for u in range(3) for v in
                         range(2)}
        product = _graph(6, _cartesian_product_edges(3, _A_EDGES, 2, _B_EDGES,
                                                     pair_to_index))
        pairing = self._pairing(pair_to_index)
        cert = cob.KuennethProduct.productCertificate(product, a, b, pairing)
        self.assertEqual(cert.grade, cob.CertificateGrade.AlgebraicallyExact)
        self.assertTrue(cert.holds(), cert.describe())
        self.assertLess(cert.residual, 1e-14)

        # Acceptance: the product-complex spectrum matches the pairwise
        # one-particle sums — no product eigensolve.
        spec_a = cob.HodgeLaplacian(a).eigenvalues(0)
        spec_b = cob.HodgeLaplacian(b).eigenvalues(0)
        spec_product = cob.HodgeLaplacian(product).eigenvalues(0)
        pairwise = cob.KuennethProduct.pairwiseSpectrum(spec_a, spec_b)
        np.testing.assert_allclose(np.sort(np.real(spec_product)),
                                   np.sort(np.real(pairwise)),
                                   rtol=0, atol=1e-10)

    def test_relabeled_product_certifies_identically(self):
        """Vertex identifiers are matched as a set through the explicit
        pairing: an arbitrary relabeling of the product's vertex ids and a
        shuffled pairing list certify with the same verdict."""
        a = _graph(3, _A_EDGES)
        b = _graph(2, _B_EDGES)
        pair_to_index = {(u, v): 2 * u + v for u in range(3) for v in
                         range(2)}
        edges = _cartesian_product_edges(3, _A_EDGES, 2, _B_EDGES,
                                         pair_to_index)
        scrambled_ids = [905, 17, 3, 411, 62, 700]
        product = _graph(6, edges, vertex_ids=scrambled_ids)
        pairing = [(scrambled_ids[2 * u + v], u, v)
                   for u in range(3) for v in range(2)]
        rng = np.random.default_rng(89)
        rng.shuffle(pairing)
        cert = cob.KuennethProduct.productCertificate(product, a, b, pairing)
        self.assertTrue(cert.holds(), cert.describe())
        self.assertLess(cert.residual, 1e-14)

    def test_reversed_edge_orientations_certify_identically(self):
        """Storing an edge in the opposite direction with the negated phase
        is the SAME Hermitian operator (the stored source->target
        orientation carries +phase, the reverse -phase): the certificate is
        orientation-independent."""
        a = _graph(3, _A_EDGES)
        b = _graph(2, _B_EDGES)
        pair_to_index = {(u, v): 2 * u + v for u in range(3) for v in
                         range(2)}
        edges = _cartesian_product_edges(3, _A_EDGES, 2, _B_EDGES,
                                         pair_to_index)
        # Reverse every other product edge: swap endpoints, negate phase.
        reversed_edges = [
            (tgt, src, w, -phi) if k % 2 else (src, tgt, w, phi)
            for k, (src, tgt, w, phi) in enumerate(edges)]
        product = _graph(6, reversed_edges)
        cert = cob.KuennethProduct.productCertificate(
            product, a, b, self._pairing(pair_to_index))
        self.assertTrue(cert.holds(), cert.describe())
        self.assertLess(cert.residual, 1e-14)

    def test_wrong_pairing_fails_to_certify(self):
        # A transposed pairing misassigns the weights: the identity must NOT
        # certify (holds() False), it must not throw — the pairing is
        # well-formed, just wrong.
        a = _graph(3, _A_EDGES)
        b = _graph(2, _B_EDGES)
        pair_to_index = {(u, v): 2 * u + v for u in range(3) for v in
                         range(2)}
        product = _graph(6, _cartesian_product_edges(3, _A_EDGES, 2, _B_EDGES,
                                                     pair_to_index))
        swapped = {(u, v): 2 * ((u + 1) % 3) + v
                   for u in range(3) for v in range(2)}
        cert = cob.KuennethProduct.productCertificate(
            product, a, b, self._pairing(swapped))
        self.assertFalse(cert.holds())

    def test_staircase_simplicial_product_is_refused(self):
        """The staircase (Eilenberg-Zilber) SimplicialProduct subdivides the
        product cells with diagonal edges: NOT an actual product 1-skeleton,
        so the Kuenneth rule must be refused for it."""
        def build(topology):
            sig = tessera.Signature(topology.dimension(), tessera.Lorentzian)
            metric = tessera.Metric(True, sig)
            st = tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0,
                                   tessera.PREFERRED, topology)
            st.build()
            for e in st.getEdgeList().toVector():
                e.setLength(1.0 + 0j)
                e.setPhase(0.0)
            return st

        circle_a = build(tessera.SimplexBoundarySphere(1))
        circle_b = build(tessera.SimplexBoundarySphere(1))
        torus = build(tessera.SimplicialProduct(
            tessera.SimplexBoundarySphere(1),
            tessera.SimplexBoundarySphere(1)))
        # SimplicialProduct assigns product vertex (u, v) the id u*|V(B)|+v.
        pairing = [(u * 3 + v, u, v) for u in range(3) for v in range(3)]
        cert = cob.KuennethProduct.productCertificate(torus, circle_a,
                                                      circle_b, pairing)
        self.assertFalse(cert.holds())
        self.assertGreater(cert.residual, 1e-3)

    def test_malformed_pairing_raises(self):
        a = _graph(3, _A_EDGES)
        b = _graph(2, _B_EDGES)
        pair_to_index = {(u, v): 2 * u + v for u in range(3) for v in
                         range(2)}
        product = _graph(6, _cartesian_product_edges(3, _A_EDGES, 2, _B_EDGES,
                                                     pair_to_index))
        good = self._pairing(pair_to_index)
        with self.assertRaises(ValueError):  # wrong size
            cob.KuennethProduct.productCertificate(product, a, b, good[:-1])
        with self.assertRaises(ValueError):  # duplicate product vertex
            cob.KuennethProduct.productCertificate(
                product, a, b, good[:-1] + [good[0]])
        with self.assertRaises(ValueError):  # unknown identifier
            cob.KuennethProduct.productCertificate(
                product, a, b, good[:-1] + [(999, 2, 1)])


if __name__ == "__main__":
    unittest.main()
