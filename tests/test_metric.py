# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

import unittest

from tessera import Edge, Vertex, Metric, Signature, SignatureType


class TestMetric(unittest.TestCase):

    def test_metric_instantiates(self):
        v1 = Vertex(1, [0, 0, 0, 0])
        v2 = Vertex(2, [0, 0, 0, 1])
        edge = Edge(v1, v2, complex(5.0, 0.0))  # spacelike length 5 (l^2 = 25)

        self.assertIsInstance(edge, Edge)
        src = edge.get_source().get_id()
        tgt = edge.get_target().get_id()
        self.assertIs(src, v1.get_id())
        self.assertIs(tgt, v2.get_id())

        signature = Signature(4, SignatureType.Lorentzian)
        self.assertEqual(signature.get_diagonal(), [-1, 1, 1, 1])
        metric = Metric(True, signature)
        with self.assertRaisesRegex(RuntimeError, "You asked a coordinate free metric to compute the squared length of an edge"):
            metric.get_squared_length(v1.get_coordinates(), v2.get_coordinates())

        signature = Signature(4, SignatureType.Lorentzian)
        self.assertEqual(signature.get_diagonal(), [-1, 1, 1, 1])
        metric = Metric(False, signature)
        self.assertEqual(metric.get_squared_length(v1.get_coordinates(), v2.get_coordinates()), 1)


if __name__ == '__main__':
    unittest.main()
