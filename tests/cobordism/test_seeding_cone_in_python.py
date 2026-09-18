# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""SurgicalCone.coneIn contract + connectivity (#503).

coneIn requires exactly d targets (a facet of a top cell), NOT the full (d+1)-vertex
cell — passing the whole cell fails the arg-count check, so a cone-in could only ever
shrink, never grow. These tests pin coneIn's contract and the connectivity invariant
it must preserve (no vertices left unconnected by edges), and show that repeated
cone-in genuinely grows a connected complex from a single simplex. coneIn is one of
runStage1's gated surgical candidate moves; the input/output blocks are now seeded
without it (growBlockRegions grows them emergently).
"""
import collections
import unittest

import tessera

cob = tessera.cobordism


def _pentatope():
    """A single solid 4-simplex (one top cell, 5 vertices, a 4-ball)."""
    return tessera.Spacetime.fromVertexTuples(4, [[0, 1, 2, 3, 4]], 1.0, 0.0)


def _counts(st):
    verts = [v.getId() for v in st.getVertexList().toVector()]
    edges = st.getEdgeList().toVector()
    cells = st.getTopSimplices()
    return len(verts), len(edges), len(cells)


def _loose_vertices(st):
    """Vertex ids not incident to any edge — what 'loose vertices floating' means."""
    edged = set()
    for e in st.getEdgeList().toVector():
        edged.add(e.getSource().getId())
        edged.add(e.getTarget().getId())
    return {v.getId() for v in st.getVertexList().toVector()} - edged


def _components(st):
    """Number of connected components of the 1-skeleton (edge graph)."""
    adj = collections.defaultdict(set)
    verts = {v.getId() for v in st.getVertexList().toVector()}
    for e in st.getEdgeList().toVector():
        a, b = e.getSource().getId(), e.getTarget().getId()
        adj[a].add(b)
        adj[b].add(a)
    seen, comps = set(), 0
    for v in verts:
        if v in seen:
            continue
        comps += 1
        stack = [v]
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            stack.extend(adj[x] - seen)
    return comps


class ConeInContractTest(unittest.TestCase):
    """coneIn requires exactly d (= top-cell-vertices - 1) target vertices."""

    def test_full_cell_is_rejected(self):
        # The exact bug: passing the whole (d+1)-vertex top cell.
        sc = cob.SurgicalCone(_pentatope())
        ok, reason = sc.coneIn([0, 1, 2, 3, 4])
        self.assertFalse(ok)
        self.assertIn("4", reason)   # "cone-in needs 4 target vertices (got 5)"
        self.assertIn("5", reason)

    def test_too_few_targets_is_rejected(self):
        sc = cob.SurgicalCone(_pentatope())
        self.assertFalse(sc.coneIn([0, 1, 2])[0])

    def test_facet_is_accepted(self):
        # A d-vertex facet of the cell is the correct payload.
        sc = cob.SurgicalCone(_pentatope())
        ok, reason = sc.coneIn([0, 1, 2, 3])
        self.assertTrue(ok, reason)


class ConeInConnectivityTest(unittest.TestCase):
    """coneIn wires the fresh apex with edges — no loose vertices."""

    def test_apex_is_edge_connected(self):
        st = _pentatope()
        v0, e0, c0 = _counts(st)
        self.assertEqual(_loose_vertices(st), set())
        sc = cob.SurgicalCone(st)
        self.assertTrue(sc.coneIn([0, 1, 2, 3])[0])
        v1, e1, c1 = _counts(st)
        self.assertEqual(v1, v0 + 1)          # one fresh apex
        self.assertEqual(e1, e0 + 4)          # apex joined to the 4 targets by edges
        self.assertEqual(c1, c0 + 1)          # one new top cell
        self.assertEqual(_loose_vertices(st), set(), "cone-in left a loose vertex")

    def test_rollback_restores_exactly(self):
        st = _pentatope()
        before = _counts(st)
        sc = cob.SurgicalCone(st)
        self.assertTrue(sc.coneIn([0, 1, 2, 3])[0])
        self.assertTrue(sc.rollback())
        self.assertEqual(_counts(st), before)
        self.assertEqual(_loose_vertices(st), set())


class ConeInGrowthTest(unittest.TestCase):
    """Repeated cone-in 'increases the size' of a single simplex, staying a single
    connected complex with no loose vertices (the intended seeding mechanism)."""

    def test_repeated_cone_in_grows_connected(self):
        st = _pentatope()
        sc = cob.SurgicalCone(st)
        grown = 0
        for _ in range(8):
            boundary = sorted(tuple(sorted(f)) for f in st.getBoundary())
            self.assertTrue(boundary, "ball lost its boundary")
            ok, _reason = sc.coneIn(list(boundary[0]))   # cone onto a boundary facet
            if ok:
                grown += 1
                self.assertEqual(_loose_vertices(st), set(),
                                 "growth left a loose vertex")
                self.assertEqual(_components(st), 1, "growth disconnected the complex")
        self.assertGreaterEqual(grown, 4, "cone-in could not grow the seed")
        _, _, cells = _counts(st)
        self.assertEqual(cells, 1 + grown)   # each accepted cone-in adds one cell


if __name__ == "__main__":
    unittest.main()
