# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

import unittest

from tessera import Vertex, Spacetime


class TestVertex(unittest.TestCase):

    def test_vertex_creation(self):
        v1 = Vertex(1, [1, 2, 3, 4])
        coords = v1.get_coordinates()
        self.assertEqual(coords, [1, 2, 3, 4])

    def test_remove_outedge(self):
        st = Spacetime()
        v1 = st.create_vertex(1, [1, 2, 3, 4])
        v2 = st.create_vertex(2, [5, 6, 7, 8])
        edge = st.create_edge(v1, v2)

        self.assertEqual(len(v1.get_out_edges()), 1)
        self.assertEqual([e for e in v1.get_out_edges()][0].get_target().get_id(), v2.get_id())

        v1.remove_out_edge(edge)
        self.assertEqual(len(v1.get_out_edges()), 0)

