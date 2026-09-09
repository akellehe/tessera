# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""The edge's simplex index is what the shift and inverse-flip moves propose off.

`Spacetime::registerSimplex` mirrors every simplex into each of its edges, so the
simplices registered on an edge are exactly the simplices carrying both of its
endpoints.  The moves rely on that: walking the edge's index has to find the same
cells as walking a vertex's incidence list and filtering by the other endpoint,
which is the O(N4) scan it replaces (#970).
"""

import unittest

import tessera


class TestEdgeSimplexIndex(unittest.TestCase):
    TOP_SIZE = 5  # a 4-simplex carries five vertices

    @classmethod
    def setUpClass(cls):
        """One thermalized CDT complex, shared: these tests only read it."""
        sig = tessera.Signature(4, tessera.Lorentzian)
        st = tessera.Spacetime(tessera.Metric(True, sig), tessera.CDT,
                               1.0, 1.0, tessera.PREFERRED, tessera.Toroid())
        st.setSeed(20260906)
        st.build(1600)
        cdt = tessera.CDTSimulation(st, 2.2, 0.5, 0.6, 1.0 / 2000, 2000)
        cdt.setSeed(20260906)
        cdt.tune()
        cdt.sweep(50)
        cls.spacetime = st

    def _edges(self, limit):
        return self.spacetime.getEdgeList().toVector()[:limit]

    @staticmethod
    def _ids(simplices):
        return {tuple(sorted(v.getId() for v in s.getVertices()))
                for s in simplices}

    def test_edge_index_finds_what_a_vertex_scan_finds(self):
        """The cells on an edge are the cells carrying both its endpoints."""
        checked = 0
        for edge in self._edges(400):
            source, target = edge.getSource(), edge.getTarget()
            by_edge = self._ids(s for s in edge.simplices()
                                if len(s.getVertices()) == self.TOP_SIZE)
            by_scan = self._ids(s for s in source.getSimplices()
                                if len(s.getVertices()) == self.TOP_SIZE
                                and s.hasVertex(target))
            self.assertEqual(by_edge, by_scan)
            checked += 1
        self.assertGreater(checked, 0, "no edges to check")

    def test_the_index_is_symmetric_in_the_endpoints(self):
        """Reading from either endpoint gives the same cells, so a move may
        start from whichever endpoint it happens to hold."""
        for edge in self._edges(200):
            source, target = edge.getSource(), edge.getTarget()
            from_source = self._ids(s for s in source.getSimplices()
                                    if len(s.getVertices()) == self.TOP_SIZE
                                    and s.hasVertex(target))
            from_target = self._ids(s for s in target.getSimplices()
                                    if len(s.getVertices()) == self.TOP_SIZE
                                    and s.hasVertex(source))
            self.assertEqual(from_source, from_target)

    def test_an_edges_incidence_is_far_smaller_than_its_endpoints(self):
        """The reason the moves were rewritten: a vertex's incidence list grows
        with the four-volume and an edge's does not."""
        edges = self._edges(400)
        edge_degrees = [len(e.simplices()) for e in edges]
        vertex_degrees = [len(e.getSource().getSimplices()) for e in edges]
        mean_edge = sum(edge_degrees) / len(edge_degrees)
        mean_vertex = sum(vertex_degrees) / len(vertex_degrees)
        self.assertLess(mean_edge, mean_vertex)


if __name__ == "__main__":
    unittest.main()
