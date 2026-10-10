# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

import unittest

import tessera
from tessera import Edge, Vertex, Spacetime


class TestEdge(unittest.TestCase):

    def test_edge_instantiates(self):
        v1 = Vertex(1, [0, 0, 0, 0])
        v2 = Vertex(2, [1, 1, 1, 1])
        edge = Edge(v1, v2)

        self.assertIsInstance(edge, Edge)
        src = edge.get_source().get_id()
        tgt = edge.get_target().get_id()
        self.assertIs(src, v1.get_id())
        self.assertIs(tgt, v2.get_id())

    def test_sets_of_edges(self):
        vertices = [Vertex(i, []) for i in range(1, 52)]
        edges = [Edge(vertices[i], vertices[i+1]) for i in range(50)]
        self.assertEqual(len(set(edges)), len(edges))

        v1 = vertices[0]
        v2 = vertices[1]

        e1 = Edge(v1, v2)
        e2 = Edge(v1, v2)
        e3 = Edge(v2, v1)
        edges = set()
        edges.add(e1)
        edges.add(e2)
        edges.add(e3)
        self.assertEqual(len(edges), 1)

    def test_maps_of_edges(self):
        vertices = [Vertex(i+1, []) for i in range(51)]
        edges = [Edge(vertices[i], vertices[i+1]) for i in range(50)]
        edge_dict = {Edge(vertices[i], vertices[i+1]): i for i in range(50)}
        for i, e in enumerate(edges):
            self.assertEqual(e, edges[i])
            self.assertEqual(edge_dict.get(e), i)

        self.assertEqual(len(set(edges)), len(edges))

    def test_equality(self):
        v1 = Vertex(1, [])
        v2 = Vertex(2, [])
        v3 = Vertex(3, [])
        e1 = Edge(v1, v2)
        e2 = Edge(v2, v1)
        self.assertEqual(e1, e2)
        e3 = Edge(v1, v2)
        self.assertEqual(e1, e3)
        e4 = Edge(v2, v3)
        self.assertNotEqual(e1, e4)


class TestEdgePhase(unittest.TestCase):

    def test_default_phase_is_zero(self):
        v1 = Vertex(1, [0, 0, 0, 0])
        v2 = Vertex(2, [1, 1, 1, 1])
        edge = Edge(v1, v2)
        self.assertEqual(edge.get_phase(), 0.0)

    def test_default_phase_is_zero_with_explicit_squared_length(self):
        v1 = Vertex(1, [0, 0, 0, 0])
        v2 = Vertex(2, [1, 1, 1, 1])
        edge = Edge(v1, v2, 1.0)
        self.assertEqual(edge.get_phase(), 0.0)

    def test_set_phase_round_trip(self):
        v1 = Vertex(1, [0, 0, 0, 0])
        v2 = Vertex(2, [1, 1, 1, 1])
        edge = Edge(v1, v2)
        for value in (0.5, -1.25, 3.14159, 0.0):
            edge.set_phase(value)
            self.assertEqual(edge.get_phase(), value)

    def test_the_phase_is_complex(self):
        # The structure group is C* = U(1) x R+, so the phase carries a
        # compact angle in Re and a non-compact log-scale in Im (#804).
        v1 = Vertex(1, [0, 0, 0, 0])
        v2 = Vertex(2, [1, 1, 1, 1])
        edge = Edge(v1, v2)
        self.assertIsInstance(edge.get_phase(), complex)
        for value in (complex(0.5, 1.5), complex(-1.25, -0.75),
                      complex(0.0, 2.0)):
            edge.set_phase(value)
            self.assertEqual(edge.get_phase(), value)
            self.assertEqual(edge.get_phase().imag, value.imag)

    def test_the_phase_is_independent_of_the_length(self):
        # Two distinct fields: writing one must not disturb the other.
        v1 = Vertex(1, [0, 0, 0, 0])
        v2 = Vertex(2, [1, 1, 1, 1])
        edge = Edge(v1, v2)
        edge.set_length(complex(2.0, -3.0))
        edge.set_phase(complex(0.25, 0.75))
        self.assertEqual(edge.get_length(), complex(2.0, -3.0))
        self.assertEqual(edge.get_phase(), complex(0.25, 0.75))
        edge.set_length(complex(-1.0, 0.5))
        self.assertEqual(edge.get_phase(), complex(0.25, 0.75))
        edge.set_phase(complex(1.0, 1.0))
        self.assertEqual(edge.get_length(), complex(-1.0, 0.5))


class TestHermitianWeightedSpacetimeType(unittest.TestCase):

    def test_hermitian_weighted_value_exists(self):
        self.assertTrue(hasattr(tessera, "HERMITIAN_WEIGHTED"))

    def test_spacetime_constructs_with_hermitian_weighted(self):
        signature = tessera.Signature(4, tessera.Lorentzian)
        metric = tessera.Metric(True, signature)
        spacetime = tessera.Spacetime(metric, tessera.HERMITIAN_WEIGHTED, 1.0, 1.0,
                                      tessera.PREFERRED, tessera.Toroid())
        self.assertIsInstance(spacetime, tessera.Spacetime)


if __name__ == '__main__':
    unittest.main()
