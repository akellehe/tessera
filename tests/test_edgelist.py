# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

import unittest
from tessera import Edge, EdgeList, Vertex
import cmath

class TestEdgeList(unittest.TestCase):

    def test_adding_and_removing_unique_edges(self):
        el = EdgeList()
        v1 = Vertex(1, [])
        v2 = Vertex(2, [])
        self.assertEqual(el.size(), 0)
        self.assertEqual(len(el.to_vector()), 0)
        el.add(v1, v2)
        self.assertEqual(el.size(), 1)
        self.assertEqual(len(el.to_vector()), 1)
        el.add(v1, v2)
        self.assertEqual(el.size(), 1)
        self.assertEqual(len(el.to_vector()), 1)
        self.assertEqual(el.size(), 1)
        self.assertEqual(len(el.to_vector()), 1)
        el.add(v1, v2, cmath.sqrt(complex(3.)))
        self.assertEqual(el.size(), 1)
        self.assertEqual(len(el.to_vector()), 1)
