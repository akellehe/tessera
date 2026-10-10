# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

import gc
import unittest

import tessera
from tessera import Spacetime, Edge, Vertex
import cmath

class TestSpacetime(unittest.TestCase):

    def test_create_vertex(self):
        st = Spacetime()
        v1 = st.create_vertex(1)
        v2 = st.create_vertex(2)

        self.assertEqual(v1.get_id(), 1)
        self.assertEqual(v2.get_id(), 2)

        self.assertNotEqual(v1, v2)

    def test_create_edge(self):
        st = Spacetime()
        v1 = st.create_vertex(1)
        v2 = st.create_vertex(2)
        v3 = st.create_vertex(3)

        self.assertEqual(v1.get_id(), 1)
        self.assertEqual(v2.get_id(), 2)
        self.assertEqual(v3.get_id(), 3)

        e1 = st.create_edge(v1, v2)
        e2 = st.create_edge(v2, v3)

        self.assertNotEqual(v1, v2)
        self.assertNotEqual(v2, v3)
        self.assertNotEqual(e1, e2)

        self.assertEqual(e1.get_source().get_id(), v1.get_id())
        self.assertEqual(e1.get_target().get_id(), v2.get_id())
        self.assertEqual(e2.get_source().get_id(), v2.get_id())
        self.assertEqual(e2.get_target().get_id(), v3.get_id())

    def test_create_simplex(self):
        st = Spacetime()
        simplex, _ = st.create_simplex((2, 3))
        self.assertEqual(len(simplex.get_vertices()), 5)
        edges = simplex.get_edges()
        self.assertEqual(len(edges), 10)

        v1, v2, v3, v4, v5 = simplex.get_vertices()
        a, b, c, d = v1.get_out_edges()
        self.assertEqual(a.get_source().get_id(), v1.get_id())
        self.assertEqual(b.get_source().get_id(), v1.get_id())
        self.assertEqual(c.get_source().get_id(), v1.get_id())
        self.assertEqual(d.get_source().get_id(), v1.get_id())
        self.assertEqual(len(v1.get_edges()), 4)

        a, b, c = v2.get_out_edges()
        self.assertEqual(a.get_source().get_id(), v2.get_id())
        self.assertEqual(b.get_source().get_id(), v2.get_id())
        self.assertEqual(c.get_source().get_id(), v2.get_id())
        self.assertEqual(len(v2.get_edges()), 4)
        a, = v2.get_in_edges()
        self.assertEqual(a.get_target().get_id(), v2.get_id())

        a, b = v3.get_out_edges()
        c, d = v3.get_in_edges()
        self.assertEqual(a.get_source().get_id(), v3.get_id())
        self.assertEqual(b.get_source().get_id(), v3.get_id())
        self.assertEqual(c.get_target().get_id(), v3.get_id())
        self.assertEqual(d.get_target().get_id(), v3.get_id())
        self.assertEqual(len(v3.get_edges()), 4)

        a, = v4.get_out_edges()
        b, c, d = v4.get_in_edges()
        self.assertEqual(a.get_source().get_id(), v4.get_id())
        self.assertEqual(b.get_target().get_id(), v4.get_id())
        self.assertEqual(c.get_target().get_id(), v4.get_id())
        self.assertEqual(d.get_target().get_id(), v4.get_id())
        self.assertEqual(len(v4.get_edges()), 4)

        a, b, c, d = v5.get_in_edges()
        self.assertEqual(a.get_target().get_id(), v5.get_id())
        self.assertEqual(b.get_target().get_id(), v5.get_id())
        self.assertEqual(c.get_target().get_id(), v5.get_id())
        self.assertEqual(d.get_target().get_id(), v5.get_id())
        self.assertEqual(len(v5.get_edges()), 4)

        self.assertEqual(len(st.get_simplices_with_orientation((2, 3))), 1)
        self.assertEqual(len(st.get_simplices_with_orientation((1, 1))), 0)

        simplex2, _ = st.create_simplex((2, 3))
        self.assertEqual(len(st.get_simplices_with_orientation((2, 3))), 2)
        self.assertEqual(len(st.get_simplices_with_orientation((1, 1))), 0)

    @unittest.skip
    def test_euclidean_embedding(self):
        st = Spacetime()
        simplex14, _ = st.create_simplex((1, 4))
        simplex23, _ = st.create_simplex((2, 3))
        st.embedEuclidean()
        vertices = st.get_vertex_list().to_vector()

    def test_attaching_faces4D(self):
        st = Spacetime()

        firstVertexList = st.get_vertex_list()
        firstEdgeList = st.get_edge_list()

        simplex14, _ = st.create_simplex((1, 4))
        simplex23, _ = st.create_simplex((2, 3))

        self.assertEqual(len(simplex14.get_vertices()), 5)
        self.assertEqual(len(simplex14.get_edges()), 10)

        self.assertEqual(len(simplex23.get_vertices()), 5)
        self.assertEqual(len(simplex23.get_edges()), 10)

        allVertices = [v.get_id() for v in simplex14.get_vertices() + simplex23.get_vertices()]
        self.assertEqual(len(allVertices), len(set(allVertices)))

        allEdges = [(e.get_source().get_id(), e.get_target().get_id()) for e in [_ for _ in simplex14.get_edges()] + [_ for _ in simplex23.get_edges()]]
        self.assertEqual(len(simplex14.get_edges()), len(set(simplex14.get_edges())))
        self.assertEqual(len(simplex23.get_edges()), len(set(simplex23.get_edges())))

        edges23 = {(e.get_source().get_id(), e.get_target().get_id()) for e in simplex23.get_edges()}
        edges14 = {(e.get_source().get_id(), e.get_target().get_id()) for e in simplex14.get_edges()}

        self.assertTrue(edges23.isdisjoint(edges14))
        self.assertEqual(len(allEdges), len(set(allEdges)))

        totalVerticesBefore = len(st.get_vertex_list().to_vector())
        self.assertEqual(totalVerticesBefore, 10)

        totalEdgesBefore = len(st.get_edge_list().to_vector())
        self.assertEqual(totalEdgesBefore, 20)

        for edge in firstEdgeList.to_vector():
            source = firstVertexList.get(edge.get_source().get_id())
            target = firstVertexList.get(edge.get_target().get_id())
            self.assertIsNotNone(source)
            self.assertIsNotNone(target)

        left, right = None, None
        facets14 = simplex14.get_facets()
        ntime, nspace = 0, 0
        for facet in facets14:
            self.assertTrue(facet.is_initialized())
            if facet.is_spatial():
                self.assertTrue(facet.get_orientation().numeric()[0] == 0 or facet.get_orientation().numeric()[1] == 0)
                ntime += 1
                ti, tf = float('inf'), float('-inf')
                for v in facet.get_vertices():
                    ti = min(ti, v.get_time())
                    tf = max(tf, v.get_time())
                self.assertEqual(ti, tf)
            else:
                self.assertNotEqual(facet.get_orientation().numeric()[0], 0)
                self.assertNotEqual(facet.get_orientation().numeric()[1], 0)
                nspace += 1
                ti, tf = float('inf'), float('-inf')
                for v in facet.get_vertices():
                    ti = min(ti, v.get_time())
                    tf = max(tf, v.get_time())
                self.assertNotEqual(ti, tf)

            if facet.get_orientation().numeric() == (1, 3):
                left = facet

        self.assertEqual(ntime, 1)
        self.assertEqual(nspace, 4)
        self.assertIsNotNone(left)

        facets23 = simplex23.get_facets()
        nspace = 0
        ntime = 0
        for face in facets23:
            if face.is_spatial():
                ntime += 1
            else:
                nspace += 1
            if face.get_orientation().numeric() == (1, 3):
                right = face

        self.assertEqual(nspace, 5)
        self.assertEqual(ntime, 0)

        self.assertIsNotNone(left)
        self.assertIsNotNone(right)

        totalVerticesBefore = st.get_vertex_list().to_vector()
        totalEdgesBefore = st.get_edge_list().to_vector()

        self.assertEqual(len(totalVerticesBefore), 10)
        self.assertEqual(len(totalEdgesBefore), 20)

        leftVerticesBefore = [v.get_id() for v in left.get_vertices()]
        self.assertEqual(len(leftVerticesBefore), 4)
        leftEdgesBefore = [(e.get_source().get_id(), e.get_target().get_id()) for e in left.get_edges()]
        self.assertEqual(len(leftEdgesBefore), 6)

        rightVerticesBefore = [v.get_id() for v in right.get_vertices()]
        self.assertEqual(len(rightVerticesBefore), 4)
        rightEdgesBefore = [(e.get_source().get_id(), e.get_target().get_id()) for e in right.get_edges()]
        self.assertEqual(len(rightEdgesBefore), 6)

    def test_we_get_connected_components_when_constructing_from_primitives(self):
        st = Spacetime()

        vertices = []
        for i in range(10):
            vertices.append(st.create_vertex(i))

        edges = []
        for i in range(0, 9, 2):
            edges.append(st.create_edge(vertices[i], vertices[i+1]))

        components = st.get_connected_components()
        self.assertEqual(len(components), 5)

    def test_we_get_connected_components_when_constructing_from_simplexes(self):
        st = Spacetime()

        st.create_simplex((1, 4))
        st.create_simplex((2, 3))

        components = st.get_connected_components()
        self.assertEqual(len(components), 2)


    def test_vertex_stores_all_simplices_in_which_it_resides(self):
        st = Spacetime()
        s14, _ = st.create_simplex((1, 4))
        k3facets = s14.get_facets()
        for k3facet in k3facets:
            for k2facet in k3facet.get_facets():
                for k1facet in k2facet.get_facets():
                    for vertex in k1facet.get_vertices():
                        self.assertIn(vertex, k1facet.get_vertices())
                        self.assertIn(vertex, k2facet.get_vertices())
                        self.assertIn(vertex, k3facet.get_vertices())
                        self.assertIn(k1facet, vertex.get_simplices())
                        self.assertIn(k2facet, vertex.get_simplices())
                        self.assertIn(k3facet, vertex.get_simplices())
                        for coface in k1facet.get_cofaces():
                            self.assertIn(coface, vertex.get_simplices())
                        for coface in k2facet.get_cofaces():
                            self.assertIn(coface, vertex.get_simplices())
                        for coface in k3facet.get_cofaces():
                            self.assertIn(coface, vertex.get_simplices())


class TestExternalSimplicesNonCDT(unittest.TestCase):
    """get_external_simplices() on hand-built (non-CDT) complexes.

    Regression: facets are materialized lazily by Simplex.get_facets(), which
    registers them back into the spacetime's simplex vector. A from-scratch
    complex has no facets until something asks for them, so getExternalSimplices
    used to (a) grow the vector it was iterating — a segfault — and (b) read
    incomplete coface counts mid-pass. CDT-built complexes never hit this
    because gluing materializes facets up front.
    """

    def _tetra_boundary(self):
        """S^2 = boundary of a tetrahedron: 4 vertices, 4 triangles, closed."""
        import itertools
        st = Spacetime()
        V = [st.create_vertex(i, [0.0]) for i in range(4)]
        for a, b in itertools.combinations(range(4), 2):
            st.create_edge(V[a], V[b], cmath.sqrt(complex(1.0)))
        tris = [st.create_simplex([V[i] for i in c])[0]
                for c in itertools.combinations(range(4), 3)]
        return st, V, tris

    def test_closed_surface_has_no_external_top_simplices(self):
        # Every edge of S^2 is shared by exactly two triangles, so no triangle
        # has a boundary facet. (Must not segfault.)
        st, _V, tris = self._tetra_boundary()
        ext = st.get_external_simplices()
        triangles = [s for s in ext if len(s.get_vertices()) == 3]
        self.assertEqual(triangles, [],
                         "closed S^2 should have no boundary triangles")

    def test_open_complex_reports_boundary(self):
        # Drop one triangle: the three edges it covered now have a single
        # coface, so the three remaining triangles each gain a boundary edge.
        st, _V, tris = self._tetra_boundary()
        st.remove_simplex(tris[0])
        ext = st.get_external_simplices()
        triangles = [s for s in ext if len(s.get_vertices()) == 3]
        self.assertEqual(len(triangles), 3,
                         "each remaining triangle should touch the new boundary")


class TestCreateSimplexVertexCap(unittest.TestCase):
    """createSimplex must reject simplices beyond the Fingerprint capacity.

    The Fingerprint stores at most kMax = 8 vertex IDs; past that it truncates,
    so a >8-vertex simplex would collide with another and be silently dropped
    (returned created=true but never registered). createSimplex now raises
    instead of corrupting the complex (issue #77).
    """

    KMAX = 8  # mesh/Fingerprint.h

    def _verts(self, st, n):
        return [st.create_vertex(i) for i in range(n)]

    def test_max_capacity_simplex_registers(self):
        # A kMax-vertex simplex (dimension 7) is the largest that round-trips.
        st = Spacetime()
        s, created = st.create_simplex(self._verts(st, self.KMAX))
        self.assertTrue(created)
        self.assertEqual(len(s.get_vertices()), self.KMAX)
        self.assertIn(s, st.get_simplices())

    def test_over_capacity_simplex_raises(self):
        st = Spacetime()
        verts = self._verts(st, self.KMAX + 1)  # 9 vertices
        with self.assertRaises(Exception):
            st.create_simplex(verts)

    def test_no_silent_drop_for_S8(self):
        # S^8 = ∂Δ^9 has ten 8-simplices (9 vertices each). Previously only 9 of
        # the 10 registered silently; now each over-capacity create raises, so
        # the complex is never partially built behind the caller's back.
        st = Spacetime()
        V = self._verts(st, 10)
        raised = 0
        for omit in range(10):
            verts = [V[i] for i in range(10) if i != omit]  # 9 vertices
            try:
                st.create_simplex(verts)
            except Exception:
                raised += 1
        self.assertEqual(raised, 10)
        self.assertEqual(len([s for s in st.get_simplices()
                              if len(s.get_vertices()) == 9]), 0)


class TestHandleLifetime(unittest.TestCase):
    """Simplex/Vertex handles returned by the query methods point into the
    Spacetime's storage, so the Spacetime must stay alive while they are used.
    The bindings enforce this (keep_alive), so the handles remain valid even
    after the Spacetime variable goes out of scope. Each helper builds a
    Spacetime in a local, returns handles, and lets the local drop — without
    the keep_alive these accesses are use-after-free (segfault).
    """

    def _sphere_spacetime(self, n=2):
        sig = tessera.Signature(4, tessera.Lorentzian)
        metric = tessera.Metric(True, sig)
        st = tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0,
                               tessera.PREFERRED, tessera.SimplexBoundarySphere(n))
        st.build()
        return st

    def test_getSimplices_outlives_spacetime_local(self):
        def handles():
            return self._sphere_spacetime(2).get_simplices()  # temporary Spacetime
        simplices = handles()
        gc.collect()
        # S^2 = boundary of a tetrahedron has 4 triangles; accessing them must
        # not touch freed storage.
        self.assertEqual(len(simplices), 4)
        self.assertTrue(all(len(s.get_vertices()) == 3 for s in simplices))

    def test_getExternalSimplices_outlives_spacetime_local(self):
        def handles():
            sig = tessera.Signature(4, tessera.Lorentzian)
            metric = tessera.Metric(True, sig)
            st = tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0,
                                   tessera.PREFERRED, tessera.SolidSimplex(4))
            st.build()
            return st.get_external_simplices()
        external = handles()
        gc.collect()
        self.assertGreater(len(external), 0)
        for s in external:
            _ = s.get_vertices()  # no crash

    def test_getRandomVertex_outlives_spacetime_local(self):
        v = self._sphere_spacetime(2).get_random_vertex()
        gc.collect()
        self.assertIsNotNone(v)
        _ = v.get_id()  # no crash

    def test_getRandomTopSimplex_outlives_spacetime_local(self):
        def handle():
            sig = tessera.Signature(4, tessera.Lorentzian)
            metric = tessera.Metric(True, sig)
            st = tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0,
                                   tessera.PREFERRED, tessera.Toroid())
            st.build(50)
            return st.get_random_top_simplex()
        s = handle()
        gc.collect()
        self.assertEqual(len(s.get_vertices()), 5)  # 4D top simplex


if __name__ == '__main__':
    unittest.main()