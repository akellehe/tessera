# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""
Tests for vertex relabeling infrastructure.

[BGL] Sec. 2.2.1: The acceptance formula includes a 1/(N0+1) factor
from the vertex labeling. Correct vertex relabeling requires updating
all dependent data structures: VertexList keys, Edge fingerprints,
Simplex vertex-ID maps, and Simplex fingerprints.

References:
  [RU]  Ambjorn, Jurkiewicz, Loll, "Reconstructing the Universe",
        Phys. Rev. D 72 (2005), arXiv:hep-th/0505154v2
  [BGL] Brunekreef, Gorlich, Loll, "Simulating CDT quantum gravity",
        arXiv:2310.16744v1 (2023)
"""

import unittest
import tessera


def _make_spacetime(d=4):
    """Create a d-dimensional Lorentzian CDT spacetime."""
    sig = tessera.Signature(d, tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    return tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0, tessera.PREFERRED,
                           tessera.Toroid())


def _top_simplices(st, d=4):
    """All top-dimensional simplices (d+1 vertices)."""
    return [s for s in st.getSimplices() if len(s.getVertices()) == d + 1]


def _vids(simplex):
    """Vertex IDs of a simplex as a frozenset."""
    return frozenset(v.getId() for v in simplex.getVertices())


def _all_vertex_ids(st):
    """Set of all vertex IDs in the spacetime."""
    return {v.getId() for v in st.getVertexList().toVector()}


def _all_edge_pairs(st):
    """Set of (min_id, max_id) for all edges in the edge list."""
    pairs = set()
    for e in st.getEdgeList().toVector():
        a, b = e.getSource().getId(), e.getTarget().getId()
        pairs.add((min(a, b), max(a, b)))
    return pairs


def _edge_pairs_of(vertex):
    """Set of (min_id, max_id) for all edges of a vertex."""
    pairs = set()
    for e in vertex.getEdges():
        a, b = e.getSource().getId(), e.getTarget().getId()
        pairs.add((min(a, b), max(a, b)))
    return pairs


def _count_orientations(st, d=4):
    """Orientation -> count for top simplices."""
    counts = {}
    for s in _top_simplices(st, d):
        o = s.getOrientation().numeric()
        counts[o] = counts.get(o, 0) + 1
    return counts


# =====================================================================
# Basic swap correctness
# =====================================================================

class TestSwapBasic(unittest.TestCase):
    """[BGL] Sec. 2.2.1: Basic vertex label swap on a single simplex."""

    def test_swap_changes_vertex_ids(self):
        """After swap, vertex IDs are exchanged."""
        st = _make_spacetime()
        st.createSimplex((1, 4))
        verts = st.getVertexList().toVector()
        v0 = [v for v in verts if v.getId() == 0][0]
        v1 = [v for v in verts if v.getId() == 1][0]

        st.swapVertexLabels(v0, v1)

        self.assertEqual(v0.getId(), 1, "v0 should now have id 1")
        self.assertEqual(v1.getId(), 0, "v1 should now have id 0")

    def test_swap_self_is_noop(self):
        """Swapping a vertex with itself does nothing."""
        st = _make_spacetime()
        st.createSimplex((1, 4))
        v0 = st.getVertexList().get(0)
        n0 = st.getVertexCount()
        n4 = st.getTopSimplexCount()

        st.swapVertexLabels(v0, v0)

        self.assertEqual(v0.getId(), 0)
        self.assertEqual(st.getVertexCount(), n0)
        self.assertEqual(st.getTopSimplexCount(), n4)

    def test_swap_preserves_vertex_count(self):
        """Swap does not change the total number of vertices."""
        st = _make_spacetime()
        st.build(20)
        n0 = st.getVertexCount()
        v0 = st.getVertexList().get(0)
        v1 = st.getVertexList().get(1)

        st.swapVertexLabels(v0, v1)

        self.assertEqual(st.getVertexCount(), n0)

    def test_swap_preserves_simplex_count(self):
        """Swap does not change the total number of simplices."""
        st = _make_spacetime()
        st.build(20)
        n4 = st.getTopSimplexCount()
        v0 = st.getVertexList().get(0)
        v1 = st.getVertexList().get(1)

        st.swapVertexLabels(v0, v1)

        self.assertEqual(st.getTopSimplexCount(), n4)


# =====================================================================
# VertexList consistency
# =====================================================================

class TestSwapVertexList(unittest.TestCase):
    """Vertex list keying consistency after swap."""

    def test_vertex_list_lookup_by_new_id(self):
        """After swap, VertexList.get(newId) returns the correct vertex."""
        st = _make_spacetime()
        st.createSimplex((1, 4))
        v0 = st.getVertexList().get(0)
        v1 = st.getVertexList().get(1)

        st.swapVertexLabels(v0, v1)

        # v0 now has id=1, v1 now has id=0
        self.assertIs(st.getVertexList().get(1), v0)
        self.assertIs(st.getVertexList().get(0), v1)

    def test_vertex_set_preserved(self):
        """The set of all vertex IDs is unchanged after swap."""
        st = _make_spacetime()
        st.build(20)
        ids_before = _all_vertex_ids(st)
        v0 = st.getVertexList().get(0)
        v1 = st.getVertexList().get(1)

        st.swapVertexLabels(v0, v1)

        self.assertEqual(_all_vertex_ids(st), ids_before)


# =====================================================================
# Edge consistency
# =====================================================================

class TestSwapEdges(unittest.TestCase):
    """[BGL] Sec. 2.2.1: Edge fingerprints and EdgeList re-keying after swap."""

    def test_edge_set_preserved(self):
        """The set of edge pairs {min,max} is unchanged after swap."""
        st = _make_spacetime()
        st.build(20)
        # Map old IDs to new IDs for the pair we'll swap
        v0 = st.getVertexList().get(0)
        v1 = st.getVertexList().get(1)
        id0, id1 = 0, 1

        edges_before = _all_edge_pairs(st)

        st.swapVertexLabels(v0, v1)

        # After swap, edges that had id0 now have id1 and vice versa
        # So the edge set should reflect the swapped IDs
        expected = set()
        for (a, b) in edges_before:
            a2 = id1 if a == id0 else (id0 if a == id1 else a)
            b2 = id1 if b == id0 else (id0 if b == id1 else b)
            expected.add((min(a2, b2), max(a2, b2)))

        self.assertEqual(_all_edge_pairs(st), expected)

    def test_edge_count_preserved(self):
        """Swap does not change the total number of edges."""
        st = _make_spacetime()
        st.build(20)
        n_edges = st.getEdgeList().size()
        v0 = st.getVertexList().get(0)
        v1 = st.getVertexList().get(1)

        st.swapVertexLabels(v0, v1)

        self.assertEqual(st.getEdgeList().size(), n_edges)

    def test_vertex_edge_degree_swapped(self):
        """After swap, each vertex's edge degree matches the other's original."""
        st = _make_spacetime()
        st.build(20)
        v0 = st.getVertexList().get(0)
        v1 = st.getVertexList().get(1)
        deg0 = v0.degree()
        deg1 = v1.degree()

        st.swapVertexLabels(v0, v1)

        # Degrees should be unchanged (same vertex objects, same edges)
        self.assertEqual(v0.degree(), deg0)
        self.assertEqual(v1.degree(), deg1)


# =====================================================================
# Simplex consistency
# =====================================================================

class TestSwapSimplices(unittest.TestCase):
    """Simplex fingerprints, orientations, and hash tables after swap."""

    def test_n41_n32_preserved(self):
        """N41 and N32 counts are unchanged by swap."""
        st = _make_spacetime()
        st.build(20)
        n41 = st.getN41()
        n32 = st.getN32()
        v0 = st.getVertexList().get(0)
        v1 = st.getVertexList().get(1)

        st.swapVertexLabels(v0, v1)

        self.assertEqual(st.getN41(), n41)
        self.assertEqual(st.getN32(), n32)
        self.assertEqual(st.getTopSimplexCount(), n41 + n32)

    def test_orientation_counts_preserved(self):
        """Orientation distribution is unchanged by swap."""
        st = _make_spacetime()
        st.build(20)
        counts_before = _count_orientations(st)
        v0 = st.getVertexList().get(0)
        v1 = st.getVertexList().get(1)

        st.swapVertexLabels(v0, v1)

        self.assertEqual(_count_orientations(st), counts_before)

    def test_simplex_vertex_sets_updated(self):
        """Simplex vertex IDs reflect the swapped labels."""
        st = _make_spacetime()
        st.createSimplex((1, 4))
        # Seed simplex has vertices {0, 1, 2, 3, 4}
        v0 = st.getVertexList().get(0)
        v4 = st.getVertexList().get(4)

        st.swapVertexLabels(v0, v4)

        # v0 now has id 4, v4 now has id 0
        top = _top_simplices(st)[0]
        ids = _vids(top)
        # The set of IDs should still be {0, 1, 2, 3, 4}
        self.assertEqual(ids, frozenset({0, 1, 2, 3, 4}))

    def test_hasVertex_works_after_swap(self):
        """Simplex.hasVertex correctly uses updated IDs."""
        st = _make_spacetime()
        st.createSimplex((1, 4))
        v0 = st.getVertexList().get(0)
        v4 = st.getVertexList().get(4)

        top = _top_simplices(st)[0]
        self.assertTrue(top.hasVertex(v0))
        self.assertTrue(top.hasVertex(v4))

        st.swapVertexLabels(v0, v4)

        # v0 now has id 4, v4 now has id 0
        # The simplex still contains both vertex objects
        self.assertTrue(top.hasVertex(v0))
        self.assertTrue(top.hasVertex(v4))


# =====================================================================
# Causality and invariants after swap
# =====================================================================

class TestSwapInvariants(unittest.TestCase):
    """All CDT invariants hold after vertex relabeling."""

    def test_causality_preserved(self):
        """Every top simplex still spans exactly 2 time slices."""
        st = _make_spacetime()
        st.build(50)
        v0 = st.getVertexList().get(0)
        v1 = st.getVertexList().get(1)

        st.swapVertexLabels(v0, v1)

        for s in _top_simplices(st):
            times = {v.getTime() for v in s.getVertices()}
            self.assertEqual(len(times), 2,
                             f"Non-causal simplex after swap: {times}")

    def test_vertex_times_swapped(self):
        """After swap, vertex times follow the vertex objects, not IDs."""
        st = _make_spacetime()
        st.createSimplex((1, 4))
        v0 = st.getVertexList().get(0)
        v1 = st.getVertexList().get(1)
        t0 = v0.getTime()
        t1 = v1.getTime()

        st.swapVertexLabels(v0, v1)

        # Vertex objects keep their times; only IDs change
        self.assertEqual(v0.getTime(), t0)
        self.assertEqual(v1.getTime(), t1)

    def test_simulation_works_after_swap(self):
        """CDT simulation continues to work after vertex relabeling."""
        st = _make_spacetime()
        st.build(50)
        target = st.getN41()
        cdt = tessera.CDTSimulation(st, 2.2, 0.5, 0.6, 1.0 / max(target, 1), target)

        v0 = st.getVertexList().get(0)
        v1 = st.getVertexList().get(1)
        st.swapVertexLabels(v0, v1)

        # Should not crash
        cdt.sweep(10)

        # Invariants hold
        self.assertEqual(st.getTopSimplexCount(), st.getN41() + st.getN32())
        for s in _top_simplices(st):
            times = {v.getTime() for v in s.getVertices()}
            self.assertEqual(len(times), 2)


# =====================================================================
# Multiple swaps and stress tests
# =====================================================================

class TestSwapStress(unittest.TestCase):
    """Repeated swaps and swap-during-simulation correctness."""

    def test_double_swap_restores_original(self):
        """Swapping the same pair twice restores original state."""
        st = _make_spacetime()
        st.build(20)
        v0 = st.getVertexList().get(0)
        v1 = st.getVertexList().get(1)

        edges_before = _all_edge_pairs(st)

        st.swapVertexLabels(v0, v1)
        st.swapVertexLabels(v0, v1)

        self.assertEqual(v0.getId(), 0)
        self.assertEqual(v1.getId(), 1)
        self.assertEqual(_all_edge_pairs(st), edges_before)

    def test_many_random_swaps_preserve_invariants(self):
        """100 random swaps preserve all counting invariants."""
        import random
        random.seed(42)

        st = _make_spacetime()
        st.build(50)
        n41_plus_n32 = st.getN41() + st.getN32()

        for _ in range(100):
            verts = st.getVertexList().toVector()
            if len(verts) < 2:
                break
            v1, v2 = random.sample(verts, 2)
            st.swapVertexLabels(v1, v2)

        # Invariants
        self.assertEqual(st.getTopSimplexCount(), st.getN41() + st.getN32())
        self.assertEqual(st.getVertexCount(), len(_all_vertex_ids(st)))

        # Causality
        for s in _top_simplices(st):
            times = {v.getTime() for v in s.getVertices()}
            self.assertEqual(len(times), 2)

    def test_swap_interleaved_with_moves(self):
        """Interleave swaps with CDT moves, verify invariants throughout."""
        st = _make_spacetime()
        st.build(100)
        target = st.getN41()
        cdt = tessera.CDTSimulation(st, 2.2, 0.5, 0.6, 1.0 / max(target, 1), target)
        cdt.tune()

        import random
        random.seed(123)

        for step in range(10):
            cdt.sweep(5)

            # Do a random swap
            verts = st.getVertexList().toVector()
            if len(verts) >= 2:
                v1, v2 = random.sample(verts, 2)
                st.swapVertexLabels(v1, v2)

            # Verify invariants
            self.assertEqual(st.getTopSimplexCount(),
                             st.getN41() + st.getN32(),
                             f"Step {step}: N4 != N41 + N32")

            for s in _top_simplices(st):
                times = {v.getTime() for v in s.getVertices()}
                self.assertEqual(len(times), 2,
                                 f"Step {step}: non-causal simplex")


# =====================================================================
# Swap on lattice with shared edges (v1 and v2 are neighbors)
# =====================================================================

class TestSwapNeighbors(unittest.TestCase):
    """Swap two vertices that share an edge."""

    def test_swap_connected_vertices(self):
        """Swapping two vertices connected by an edge preserves edge count."""
        st = _make_spacetime()
        st.createSimplex((1, 4))
        # All 5 vertices are pairwise connected (10 edges)
        n_edges = st.getEdgeList().size()
        v0 = st.getVertexList().get(0)
        v1 = st.getVertexList().get(1)

        # v0 and v1 share an edge
        st.swapVertexLabels(v0, v1)

        self.assertEqual(st.getEdgeList().size(), n_edges)
        self.assertEqual(v0.getId(), 1)
        self.assertEqual(v1.getId(), 0)

    def test_swap_connected_on_built_lattice(self):
        """Swap neighbors on a real lattice, check edge set size."""
        st = _make_spacetime()
        st.build(20)
        n_edges = st.getEdgeList().size()

        # Find two connected vertices
        v0 = st.getVertexList().get(0)
        neighbor = None
        for e in v0.getEdges():
            other = e.getTarget() if e.getSource().getId() == v0.getId() else e.getSource()
            if other.getId() != v0.getId():
                neighbor = other
                break
        self.assertIsNotNone(neighbor)

        st.swapVertexLabels(v0, neighbor)

        self.assertEqual(st.getEdgeList().size(), n_edges)
        self.assertEqual(st.getTopSimplexCount(), st.getN41() + st.getN32())


if __name__ == "__main__":
    unittest.main()
