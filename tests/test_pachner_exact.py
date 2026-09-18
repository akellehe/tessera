# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""
Exact structural tests for Pachner moves on minimal known lattices.

Each test builds the smallest lattice needed, executes the topology change
directly (no random selection, no Metropolis acceptance), and asserts the
exact vertex sets, edge connections, and orientations of every simplex.

Dimensions tested:
  - d=2,3,4 for cone (add) and its reverse (remove)
  - d=2,3,4 for the (2,d) flip and its reverse
  - d=4 for the (3,3) shift and its reverse

References:
  [RU]  Ambjorn, Jurkiewicz, Loll, "Reconstructing the Universe",
        Phys. Rev. D 72 (2005), arXiv:hep-th/0505154v2
  [BGL] Brunekreef, Gorlich, Loll, "Simulating CDT quantum gravity",
        arXiv:2310.16744v1 (2023)
"""

import unittest
import tessera


# =====================================================================
# Helpers
# =====================================================================

def _make_spacetime(d):
    """Create a d-dimensional Lorentzian CDT spacetime (no initial complex)."""
    sig = tessera.Signature(d, tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    return tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0, tessera.PREFERRED,
                           tessera.Toroid())


def _vids(simplex):
    """Vertex IDs of a simplex as a frozenset."""
    return frozenset(v.getId() for v in simplex.getVertices())


def _top_simplices(st, d):
    """All top-dimensional simplices (d+1 vertices)."""
    return [s for s in st.getSimplices() if len(s.getVertices()) == d + 1]


def _top_vids(st, d):
    """Set of frozensets of vertex IDs for all top simplices."""
    return {_vids(s) for s in _top_simplices(st, d)}


def _edge_pairs(vertex):
    """Set of (min_id, max_id) tuples for all edges incident to vertex."""
    pairs = set()
    for e in vertex.getEdges():
        a, b = e.getSource().getId(), e.getTarget().getId()
        pairs.add((min(a, b), max(a, b)))
    return pairs


def _assert_counts(test, st, expected_n41, expected_n32):
    """Assert exact N41, N32, and total counts."""
    test.assertEqual(st.getN41(), expected_n41,
                     f"N41: expected {expected_n41}, got {st.getN41()}")
    test.assertEqual(st.getN32(), expected_n32,
                     f"N32: expected {expected_n32}, got {st.getN32()}")
    test.assertEqual(st.getTopSimplexCount(), expected_n41 + expected_n32)


def _get_vertex(st, vid):
    """Get a vertex by ID from the vertex list."""
    return st.getVertexList().get(vid)


# =====================================================================
# Cone (Add) — exact structure, dimensions 2-4
# =====================================================================

class TestConeExact(unittest.TestCase):
    """[BGL] Sec. 2.3.1: Build a single seed simplex, cone a known facet, assert exact structure.

    Seed: (1, d) simplex → v0@t=0, v1..vd@t=1.
    Chosen facet: first non-timelike = {v0, v2, ..., vd} (skips v1).
    New vertex: v_{d+1}@t=0.
    New simplex: {v0, v2, ..., vd, v_{d+1}} with orientation (2, d-1).
    """

    def _run_cone_forward(self, d):
        st = _make_spacetime(d)
        seed, _ = st.createSimplex((1, d))

        # Verify seed structure
        seed_ids = set(range(d + 1))
        self.assertEqual(_vids(seed), frozenset(seed_ids))
        self.assertEqual(seed.getOrientation().numeric(), (1, d))

        # Pick first non-spatial facet (spans multiple time slices)
        facet = None
        for f in seed.getFacets():
            if not f.isSpatial():
                facet = f
                break
        self.assertIsNotNone(facet)

        facet_ids = _vids(facet)
        expected_facet = frozenset({0} | set(range(2, d + 1)))
        self.assertEqual(facet_ids, expected_facet,
                         f"d={d}: expected facet {expected_facet}, got {facet_ids}")

        # Create new vertex and new simplex
        new_vid = d + 1
        new_v = st.createVertex(new_vid, [0.0])
        self.assertEqual(new_v.getTime(), 0.0)

        new_verts = list(facet.getVertices()) + [new_v]
        new_simplex, created = st.createSimplex(new_verts)
        self.assertTrue(created)

        return st, d, seed, new_simplex, new_v, new_vid, facet_ids

    def test_cone_forward_d2(self):
        self._assert_cone_forward(2)

    def test_cone_forward_d3(self):
        self._assert_cone_forward(3)

    def test_cone_forward_d4(self):
        self._assert_cone_forward(4)

    def _assert_cone_forward(self, d):
        st, d, seed, new_simplex, new_v, new_vid, facet_ids = \
            self._run_cone_forward(d)

        # Exactly 2 top simplices with known vertex sets
        top_sets = _top_vids(st, d)
        expected_seed_vids = frozenset(range(d + 1))
        expected_new_vids = facet_ids | {new_vid}
        self.assertEqual(top_sets, {expected_seed_vids, expected_new_vids})

        # New simplex orientation: (2, d-1)
        self.assertEqual(new_simplex.getOrientation().numeric(), (2, d - 1))

        # Seed orientation unchanged
        self.assertEqual(seed.getOrientation().numeric(), (1, d))

        # New vertex has exactly d edges (one to each facet vertex)
        new_v_edges = _edge_pairs(new_v)
        expected_edges = {(min(new_vid, fv), max(new_vid, fv))
                          for fv in facet_ids}
        self.assertEqual(new_v_edges, expected_edges,
                         f"d={d}: new vertex edges {new_v_edges} != {expected_edges}")

        # Edge types: v0→new_v is spacelike (same time=0, sqlen>0),
        # all others are timelike (different times, sqlen<0)
        for e in new_v.getEdges():
            src_id, tgt_id = e.getSource().getId(), e.getTarget().getId()
            other_id = tgt_id if src_id == new_vid else src_id
            if other_id == 0:
                # Both at t=0 → spacelike
                self.assertGreater((e.getLength()**2).real, 0,
                                   "v0-new_v edge should be spacelike (sqlen>0)")
            else:
                # new_v@t=0 to other@t=1 → timelike
                self.assertLess((e.getLength()**2).real, 0,
                                f"v{other_id}-new_v edge should be timelike (sqlen<0)")

        # Orientation counts
        if d == 2:
            # (1,2) is N41, (2,1) is N41
            _assert_counts(self, st, 2, 0)
        elif d == 3:
            # (1,3) is N41, (2,2) is N32
            _assert_counts(self, st, 1, 1)
        elif d == 4:
            # (1,4) is N41, (2,3) is N32
            _assert_counts(self, st, 1, 1)

    def test_cone_reverse_d2(self):
        self._assert_cone_reverse(2)

    def test_cone_reverse_d3(self):
        self._assert_cone_reverse(3)

    def test_cone_reverse_d4(self):
        self._assert_cone_reverse(4)

    def _assert_cone_reverse(self, d):
        """Cone forward then removeSimplex: back to 1 top simplex."""
        st, d, seed, new_simplex, new_v, new_vid, facet_ids = \
            self._run_cone_forward(d)

        # Remove the new simplex
        st.removeSimplex(new_simplex)

        # Back to 1 top simplex with original vertex set
        tops = _top_simplices(st, d)
        self.assertEqual(len(tops), 1)
        self.assertEqual(_vids(tops[0]), frozenset(range(d + 1)))

        # Counts: only the seed remains
        _assert_counts(self, st, 1, 0)


# =====================================================================
# Flip (2 → d) — exact structure, dimensions 2-4
# =====================================================================

class TestFlipExact(unittest.TestCase):
    """[BGL] Sec. 2.3.2: Build 2 top simplices sharing a (d-1)-face via cone, then flip.

    Starting lattice (after cone):
      Seed: {v0, v1, ..., vd} orient (1, d)
      Coned: {v0, v2, ..., vd, v_{d+1}} orient (2, d-1)
      Shared face: {v0, v2, ..., vd}
      Unique: {v1, v_{d+1}}

    After (2→d) flip: d new simplices, each with both unique vertices
    and (d-1) of d shared vertices.
    """

    def _build_flip_lattice(self, d):
        """Build 2 simplices sharing a (d-1)-face, return all refs."""
        st = _make_spacetime(d)
        seed, _ = st.createSimplex((1, d))

        facet = [f for f in seed.getFacets() if not f.isSpatial()][0]
        facet_ids = _vids(facet)

        new_vid = d + 1
        new_v = st.createVertex(new_vid, [0.0])
        coned, _ = st.createSimplex(list(facet.getVertices()) + [new_v])

        # Shared: facet vertex objects; Unique: v1 and v_{d+1}
        shared = list(facet.getVertices())  # [v0, v2, ..., vd]
        shared_ids = sorted([v.getId() for v in shared])
        unique = [_get_vertex(st, 1), new_v]  # [v1, v_{d+1}]

        return st, seed, coned, shared, unique, shared_ids

    def _expected_new_simplices(self, d, shared_ids, unique_ids):
        """Compute the expected vertex sets for new simplices after flip."""
        expected = set()
        for skip_idx in range(d):
            verts = frozenset(
                [shared_ids[i] for i in range(d) if i != skip_idx] +
                list(unique_ids)
            )
            expected.add(verts)
        return expected

    def test_flip_forward_d2(self):
        self._assert_flip_forward(2)

    def test_flip_forward_d3(self):
        self._assert_flip_forward(3)

    def test_flip_forward_d4(self):
        self._assert_flip_forward(4)

    def _assert_flip_forward(self, d):
        st, seed, coned, shared, unique, shared_ids = \
            self._build_flip_lattice(d)

        unique_ids = [v.getId() for v in unique]
        before_n0 = st.getVertexCount()

        # Record vertices of old simplices for reverse
        seed_verts_list = list(seed.getVertices())
        coned_verts_list = list(coned.getVertices())

        # Forward flip: remove 2 old simplices
        st.removeSimplex(seed)
        st.removeSimplex(coned)
        self.assertEqual(st.getTopSimplexCount(), 0)

        # Create d new simplices: each skips one shared vertex
        new_simplices = []
        for skip_idx in range(d):
            verts = [shared[i] for i in range(d) if i != skip_idx] + unique
            ns, created = st.createSimplex(verts)
            self.assertTrue(created, f"d={d}, skip={skip_idx}: simplex already existed")
            new_simplices.append(ns)

        # Exactly d top simplices
        self.assertEqual(st.getTopSimplexCount(), d)

        # Vertex sets match expected
        actual_sets = {_vids(s) for s in new_simplices}
        expected_sets = self._expected_new_simplices(d, shared_ids, unique_ids)
        self.assertEqual(actual_sets, expected_sets)

        # Vertex count unchanged (no vertices added or removed)
        self.assertEqual(st.getVertexCount(), before_n0)

        # New edge: unique[0]-unique[1] (v1-v_{d+1}) now exists
        v1_edges = _edge_pairs(_get_vertex(st, unique_ids[0]))
        v_new_edges = _edge_pairs(_get_vertex(st, unique_ids[1]))
        pair = (min(unique_ids), max(unique_ids))
        self.assertIn(pair, v1_edges,
                      f"Edge {pair} should exist on v{unique_ids[0]}")
        self.assertIn(pair, v_new_edges,
                      f"Edge {pair} should exist on v{unique_ids[1]}")

        # Verify orientations of all new simplices
        for ns in new_simplices:
            o = ns.getOrientation().numeric()
            self.assertIn(o, self._valid_orientations(d),
                          f"Invalid orientation {o}")

        # Verify counts: compute expected N41/N32 from orientations
        n41, n32 = 0, 0
        for ns in new_simplices:
            o = ns.getOrientation().numeric()
            if o in ((d, 1), (1, d)):
                n41 += 1
            elif o in ((d - 1, 2), (2, d - 1)):
                n32 += 1
        _assert_counts(self, st, n41, n32)

    def _valid_orientations(self, d):
        return ((d, 1), (1, d), (d - 1, 2), (2, d - 1))

    def test_flip_reverse_d2(self):
        self._assert_flip_reverse(2)

    def test_flip_reverse_d3(self):
        self._assert_flip_reverse(3)

    def test_flip_reverse_d4(self):
        self._assert_flip_reverse(4)

    def _assert_flip_reverse(self, d):
        """Flip forward then reverse: restore original 2 simplices."""
        st, seed, coned, shared, unique, shared_ids = \
            self._build_flip_lattice(d)

        seed_verts = list(seed.getVertices())
        coned_verts = list(coned.getVertices())
        original_vids = _top_vids(st, d)

        # Forward flip
        st.removeSimplex(seed)
        st.removeSimplex(coned)
        new_simplices = []
        for skip_idx in range(d):
            verts = [shared[i] for i in range(d) if i != skip_idx] + unique
            ns, _ = st.createSimplex(verts)
            new_simplices.append(ns)
        self.assertEqual(st.getTopSimplexCount(), d)

        # Reverse flip: remove d new, recreate original 2
        for ns in new_simplices:
            st.removeSimplex(ns)
        self.assertEqual(st.getTopSimplexCount(), 0)

        st.createSimplex(seed_verts)
        st.createSimplex(coned_verts)

        # Back to original 2 simplices
        self.assertEqual(st.getTopSimplexCount(), 2)
        restored_vids = _top_vids(st, d)
        self.assertEqual(restored_vids, original_vids)


# =====================================================================
# Shift (3 → 3) — exact structure, d=4 only
# =====================================================================

class TestShiftExact(unittest.TestCase):
    """[BGL] Sec. 2.3.3: Build 3 top simplices sharing a triangle (3 vertices), shift.

    Lattice: 6 vertices total.
      Shared: a(0)@t=0, b(1)@t=0, c(2)@t=0
      Unique: x(3)@t=1, y(4)@t=1, z(5)@t=1

    Old simplices (each has all 3 shared + 2 of 3 unique):
      S1: {a, b, c, x, y}  orient (3, 2)
      S2: {a, b, c, x, z}  orient (3, 2)
      S3: {a, b, c, y, z}  orient (3, 2)

    New simplices (each has all 3 unique + 2 of 3 shared):
      N1: {b, c, x, y, z}  orient (2, 3)  (skip a)
      N2: {a, c, x, y, z}  orient (2, 3)  (skip b)
      N3: {a, b, x, y, z}  orient (2, 3)  (skip c)
    """

    def _build_shift_lattice(self):
        d = 4
        st = _make_spacetime(d)

        a = st.createVertex(0, [0.0])
        b = st.createVertex(1, [0.0])
        c = st.createVertex(2, [0.0])
        x = st.createVertex(3, [1.0])
        y = st.createVertex(4, [1.0])
        z = st.createVertex(5, [1.0])

        s1, _ = st.createSimplex([a, b, c, x, y])
        s2, _ = st.createSimplex([a, b, c, x, z])
        s3, _ = st.createSimplex([a, b, c, y, z])

        shared = [a, b, c]
        unique = [x, y, z]
        return st, [s1, s2, s3], shared, unique

    def test_shift_lattice_structure(self):
        """Verify the manually built shift lattice."""
        st, old, shared, unique = self._build_shift_lattice()

        self.assertEqual(st.getTopSimplexCount(), 3)
        self.assertEqual(st.getVertexCount(), 6)
        _assert_counts(self, st, 0, 3)

        expected = {
            frozenset({0, 1, 2, 3, 4}),
            frozenset({0, 1, 2, 3, 5}),
            frozenset({0, 1, 2, 4, 5}),
        }
        self.assertEqual(_top_vids(st, 4), expected)

        for s in old:
            self.assertEqual(s.getOrientation().numeric(), (3, 2))

    def test_shift_forward(self):
        """Remove 3 old, create 3 new: exact vertex sets and orientations."""
        st, old, shared, unique = self._build_shift_lattice()

        old_vert_sets = [list(s.getVertices()) for s in old]

        # Forward shift: remove old
        for s in old:
            st.removeSimplex(s)
        self.assertEqual(st.getTopSimplexCount(), 0)

        # Create new: each skips one shared vertex, includes all unique
        new_simplices = []
        for skip_idx in range(3):
            verts = [shared[i] for i in range(3) if i != skip_idx] + unique
            ns, created = st.createSimplex(verts)
            self.assertTrue(created)
            new_simplices.append(ns)

        # 3 top simplices
        self.assertEqual(st.getTopSimplexCount(), 3)

        # Exact vertex sets
        expected = {
            frozenset({1, 2, 3, 4, 5}),  # skip a(0)
            frozenset({0, 2, 3, 4, 5}),  # skip b(1)
            frozenset({0, 1, 3, 4, 5}),  # skip c(2)
        }
        self.assertEqual(_top_vids(st, 4), expected)

        # All new simplices have orientation (2, 3)
        for ns in new_simplices:
            self.assertEqual(ns.getOrientation().numeric(), (2, 3))

        # Still 6 vertices, no change
        self.assertEqual(st.getVertexCount(), 6)

        # Counts: all N32 type
        _assert_counts(self, st, 0, 3)

        # Edge check: all 15 edges (C(6,2)) still exist since no edges removed.
        # Every unique vertex should connect to every shared vertex and
        # to the other unique vertices.
        for u in unique:
            u_edges = _edge_pairs(u)
            for s in shared:
                pair = (min(u.getId(), s.getId()), max(u.getId(), s.getId()))
                self.assertIn(pair, u_edges,
                              f"Edge {pair} missing on unique vertex {u.getId()}")
            for u2 in unique:
                if u2.getId() != u.getId():
                    pair = (min(u.getId(), u2.getId()),
                            max(u.getId(), u2.getId()))
                    self.assertIn(pair, u_edges,
                                  f"Edge {pair} missing between unique vertices")

    def test_shift_reverse(self):
        """Shift forward then reverse: restore original 3 simplices."""
        st, old, shared, unique = self._build_shift_lattice()

        original_vids = _top_vids(st, 4)
        old_vert_sets = [list(s.getVertices()) for s in old]

        # Forward shift
        for s in old:
            st.removeSimplex(s)
        new_simplices = []
        for skip_idx in range(3):
            verts = [shared[i] for i in range(3) if i != skip_idx] + unique
            ns, _ = st.createSimplex(verts)
            new_simplices.append(ns)
        self.assertEqual(st.getTopSimplexCount(), 3)
        self.assertNotEqual(_top_vids(st, 4), original_vids)

        # Reverse shift: remove new, recreate old
        for ns in new_simplices:
            st.removeSimplex(ns)
        for ov in old_vert_sets:
            st.createSimplex(ov)

        self.assertEqual(st.getTopSimplexCount(), 3)
        self.assertEqual(_top_vids(st, 4), original_vids)
        _assert_counts(self, st, 0, 3)

    def test_shift_double_round_trip(self):
        """Shift forward, reverse, forward again, reverse again."""
        st, old, shared, unique = self._build_shift_lattice()
        original_vids = _top_vids(st, 4)
        old_vert_sets = [list(s.getVertices()) for s in old]

        for cycle in range(2):
            # Forward: remove old, create new
            tops = _top_simplices(st, 4)
            for s in tops:
                st.removeSimplex(s)

            new_simplices = []
            for skip_idx in range(3):
                verts = [shared[i] for i in range(3) if i != skip_idx] + unique
                ns, created = st.createSimplex(verts)
                new_simplices.append(ns)
            self.assertEqual(st.getTopSimplexCount(), 3)

            # Reverse: remove new, recreate old
            for ns in new_simplices:
                st.removeSimplex(ns)
            for ov in old_vert_sets:
                st.createSimplex(ov)
            self.assertEqual(_top_vids(st, 4), original_vids)
            _assert_counts(self, st, 0, 3)


# =====================================================================
# Cross-dimension: verify dimension-specific properties
# =====================================================================

class TestDimensionProperties(unittest.TestCase):
    """[BGL] Sec. 2.3: Verify that the topology changes produce correct results
    specific to each dimension's combinatorics."""

    def test_flip_simplex_count_change(self):
        """Flip (2→d): net change in top simplex count is d - 2."""
        for d in [2, 3, 4]:
            with self.subTest(d=d):
                st = _make_spacetime(d)
                seed, _ = st.createSimplex((1, d))
                facet = [f for f in seed.getFacets()
                         if not f.isSpatial()][0]
                new_v = st.createVertex(d + 1, [0.0])
                coned, _ = st.createSimplex(
                    list(facet.getVertices()) + [new_v])

                self.assertEqual(st.getTopSimplexCount(), 2)

                shared = list(facet.getVertices())
                unique = [_get_vertex(st, 1), new_v]

                st.removeSimplex(seed)
                st.removeSimplex(coned)
                for skip in range(d):
                    verts = [shared[i] for i in range(d)
                             if i != skip] + unique
                    st.createSimplex(verts)

                # d new simplices created, 2 removed → net d - 2
                self.assertEqual(st.getTopSimplexCount(), d,
                                 f"d={d}: flip should produce {d} simplices")

    def test_cone_new_vertex_degree(self):
        """After cone, new vertex has exactly d edges (one per facet vertex)."""
        for d in [2, 3, 4]:
            with self.subTest(d=d):
                st = _make_spacetime(d)
                st.createSimplex((1, d))
                facet = None
                for s in st.getSimplices():
                    if len(s.getVertices()) == d + 1:
                        for f in s.getFacets():
                            if not f.isSpatial():
                                facet = f
                                break
                        break

                new_v = st.createVertex(d + 1, [0.0])
                st.createSimplex(list(facet.getVertices()) + [new_v])

                self.assertEqual(len(_edge_pairs(new_v)), d,
                                 f"d={d}: new vertex should have {d} edges")

    def test_shift_is_involution(self):
        """Applying shift then its inverse returns to original structure.

        The (3,3) shift swaps the roles of shared and unique vertices.
        Old simplices share {a, b, c} and vary over {x, y, z}.
        New simplices share {x, y, z} and vary over {a, b, c}.
        Shifting again with swapped roles should restore the original.
        """
        st, old, shared, unique = TestShiftExact()._build_shift_lattice()
        original_vids = _top_vids(st, 4)

        # First shift: skip from shared, include all unique
        tops = _top_simplices(st, 4)
        for s in tops:
            st.removeSimplex(s)
        for skip_idx in range(3):
            verts = [shared[i] for i in range(3)
                     if i != skip_idx] + unique
            st.createSimplex(verts)

        mid_vids = _top_vids(st, 4)
        self.assertNotEqual(mid_vids, original_vids)

        # Second shift: roles swap — skip from unique, include all shared
        tops = _top_simplices(st, 4)
        for s in tops:
            st.removeSimplex(s)
        for skip_idx in range(3):
            verts = [unique[i] for i in range(3)
                     if i != skip_idx] + shared
            st.createSimplex(verts)

        self.assertEqual(_top_vids(st, 4), original_vids)


if __name__ == "__main__":
    unittest.main()
