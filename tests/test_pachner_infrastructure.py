# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""
Tests for the Phase 1 infrastructure underpinning the transactional
Pachner-move system (see ``docs/source/modularity-plan.md``):

* :class:`TestEdgeListTryAdd` — ``EdgeList::tryAdd`` returns
  ``(EdgePtr, inserted)`` where the bool reports fresh-insert vs
  dedupe-hit.

* :class:`TestCreateSimplexTracked` — ``Spacetime::createSimplexTracked``
  returns ``(simplex, created, newEdges)`` and accurately reports the
  edges this call freshly inserted into the EdgeList.

These primitives are what the Pachner move ``apply()`` / ``rollback()``
implementations build on, so they need exhaustive coverage.
"""
import unittest
import tessera


def _make_st(d=4, n_simplices=200):
    sig = tessera.Signature(d, tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    st = tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0,
                           tessera.PREFERRED, tessera.Toroid())
    st.build(n_simplices)
    return st


# ---------------------------------------------------------------------------
# EdgeList::tryAdd
# ---------------------------------------------------------------------------


class TestEdgeListTryAdd(unittest.TestCase):
    """``EdgeList::tryAdd(source, target, squaredLength)`` returns
    ``(edge, inserted)`` where ``inserted`` is true on a fresh insert
    and false on a dedupe-hit.
    """

    def _two_fresh_vertices(self, st):
        v1 = st.create_vertex(99991, [0.0])
        v2 = st.create_vertex(99992, [0.0])
        return v1, v2

    def test_fresh_insert_returns_true(self):
        st = _make_st()
        v1, v2 = self._two_fresh_vertices(st)
        edge_list = st.get_edge_list()
        edge, inserted = edge_list.try_add(v1, v2, 1.0)
        self.assertIsNotNone(edge)
        self.assertTrue(inserted, "tryAdd on a never-seen pair should "
                                  "return inserted=True")

    def test_second_call_returns_false(self):
        st = _make_st()
        v1, v2 = self._two_fresh_vertices(st)
        edge_list = st.get_edge_list()
        edge1, inserted1 = edge_list.try_add(v1, v2, 1.0)
        edge2, inserted2 = edge_list.try_add(v1, v2, 1.0)
        self.assertTrue(inserted1)
        self.assertFalse(inserted2,
                         "Second tryAdd on the same edge should return "
                         "inserted=False (dedupe-hit)")
        self.assertEqual(hash(edge1), hash(edge2),
                         "Both calls should return the same edge")

    def test_returns_existing_edge_object_on_dedupe(self):
        """``edge`` returned from a dedupe-hit must be the SAME edge
        object (by Python identity) as the one already in the list."""
        st = _make_st()
        v1, v2 = self._two_fresh_vertices(st)
        edge_list = st.get_edge_list()
        edge1, _ = edge_list.try_add(v1, v2, 1.0)
        edge2, inserted = edge_list.try_add(v1, v2, 1.0)
        self.assertFalse(inserted)
        # Edges in the bindings are returned by reference; they are the
        # same C++ object (and therefore equal by Edge::operator==).
        self.assertEqual(edge1, edge2)

    def test_size_increments_only_on_fresh_insert(self):
        st = _make_st()
        v1, v2 = self._two_fresh_vertices(st)
        edge_list = st.get_edge_list()
        size_before = edge_list.size()
        _, inserted1 = edge_list.try_add(v1, v2, 1.0)
        size_after_first = edge_list.size()
        _, inserted2 = edge_list.try_add(v1, v2, 1.0)
        size_after_second = edge_list.size()
        self.assertTrue(inserted1)
        self.assertEqual(size_after_first, size_before + 1)
        self.assertFalse(inserted2)
        self.assertEqual(size_after_second, size_after_first)

    def test_existing_built_lattice_edge_returns_false(self):
        """An edge that's already present from ``build()`` must be
        reported as a dedupe-hit."""
        st = _make_st()
        # Pick any existing edge from the built lattice
        edges = st.get_edge_list().to_vector()
        self.assertGreater(len(edges), 0)
        e = edges[0]
        src, tgt = e.get_source(), e.get_target()
        edge, inserted = st.get_edge_list().try_add(src, tgt, 1.0)
        self.assertFalse(inserted, "Pre-existing edge must be a dedupe-hit")
        self.assertEqual(edge, e)


# ---------------------------------------------------------------------------
# Spacetime::createSimplexTracked
# ---------------------------------------------------------------------------


class TestCreateSimplexTracked(unittest.TestCase):
    """``Spacetime::createSimplexTracked(vertices)`` returns
    ``(simplex, created, newEdges)``.

    Invariants:
    * ``created`` mirrors ``create_simplex(vertices)[1]``.
    * ``newEdges`` lists exactly the edges this call freshly inserted
      into the EdgeList.
    * When ``created`` is False (simplex already existed), ``newEdges``
      is empty.
    * Calling on a fresh d-simplex made of brand-new vertices yields
      ``newEdges`` of length C(d+1, 2) (every pair edge is new).
    """

    def _fresh_vertices(self, st, d):
        """d+1 fresh vertices not connected to anything in the lattice.
        Place them at a time-slice that doesn't already exist so they
        can't accidentally share edges with built-lattice vertices."""
        verts = []
        for i in range(d + 1):
            # Spread across time slices so edges inside this
            # simplex include both spacelike and timelike kinds.
            t = float(i % 2)
            verts.append(st.create_vertex(900000 + i, [t]))
        return verts

    def test_brand_new_simplex_marks_all_edges_new(self):
        d = 4
        st = _make_st(d=d)
        verts = self._fresh_vertices(st, d)
        edges_before = st.get_edge_list().size()
        simplex, created, new_edges = st.create_simplex_tracked(verts)
        edges_after = st.get_edge_list().size()

        self.assertTrue(created)
        # C(d+1, 2) = (d+1)*d/2 edges in a d-simplex.
        expected_edges = (d + 1) * d // 2
        self.assertEqual(len(new_edges), expected_edges,
                         f"All {expected_edges} edges should be new")
        self.assertEqual(edges_after - edges_before, expected_edges)

    def test_existing_simplex_returns_empty_new_edges(self):
        """Calling on an already-existing simplex yields ``created=False``
        and an empty newEdges list."""
        d = 4
        st = _make_st(d=d)
        # Pick an existing top simplex and call createSimplexTracked
        # with its vertices.
        existing = None
        for s in st.get_simplices():
            if len(s.get_vertices()) == d + 1:
                existing = s
                break
        self.assertIsNotNone(existing)
        edges_before = st.get_edge_list().size()
        verts = list(existing.get_vertices())
        simplex, created, new_edges = st.create_simplex_tracked(verts)
        edges_after = st.get_edge_list().size()

        self.assertFalse(created)
        self.assertEqual(len(new_edges), 0)
        self.assertEqual(edges_after, edges_before,
                         "EdgeList size unchanged when simplex already exists")
        self.assertEqual(simplex, existing)

    def test_simplex_sharing_some_edges_reports_only_new_ones(self):
        """Build a top simplex sigma1; then build sigma2 sharing one
        face with sigma1 (so the d edges in that face are pre-existing
        and only the d+1 new edges are reported)."""
        d = 4
        st = _make_st(d=d)
        # First, create d+1 fresh vertices and the simplex through them.
        v_shared = self._fresh_vertices(st, d)
        s1, c1, _ = st.create_simplex_tracked(v_shared)
        self.assertTrue(c1)

        # Now create d+1 vertices that share d of them with the prior
        # simplex (i.e., d shared, 1 new). Edges among the d shared
        # already exist, edges from the new vertex to each shared are
        # new.
        v_new = st.create_vertex(910000, [2.0])  # different time slice
        v_partial = list(v_shared[:d]) + [v_new]
        edges_before = st.get_edge_list().size()
        s2, c2, new_edges = st.create_simplex_tracked(v_partial)
        edges_after = st.get_edge_list().size()

        self.assertTrue(c2)
        # New edges: from v_new to each of the d shared vertices = d.
        self.assertEqual(len(new_edges), d,
                         f"Should be {d} new edges (v_new connecting "
                         f"to each of {d} shared)")
        self.assertEqual(edges_after - edges_before, d)

    def test_existing_createSimplex_still_returns_correct_pair(self):
        """The non-tracked overload must still behave identically."""
        d = 4
        st = _make_st(d=d)
        verts = self._fresh_vertices(st, d)
        simplex, created = st.create_simplex(verts)
        self.assertTrue(created)
        # Calling again on the same vertices: dedupe-hit.
        simplex2, created2 = st.create_simplex(verts)
        self.assertFalse(created2)
        self.assertEqual(simplex, simplex2)


if __name__ == "__main__":
    unittest.main()
