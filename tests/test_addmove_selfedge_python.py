"""growInterior / AddMove must never propose a self-edge (#267).

`Spacetime::createVertex()` (no-arg) drew its id from a counter that ignored
explicitly-assigned ids. On a complex built with explicit ids — `build()` plus
`create_vertex(id)` — the counter was stale, so the no-arg create returned an id
already in use; `VertexList::add` on a duplicate id returns the **existing**
vertex, and coning that aliased vertex in the 1→(d+1) stellar subdivision made an
edge from a vertex to itself (`grow_interior` threw for the seeds that picked a
cell containing the aliased id). The fix skips used ids.
"""

from __future__ import annotations

import unittest

import pytest
import cmath

try:
    import tessera
    cob = tessera.cobordism
    _IMPORT_OK = True
except Exception:  # pragma: no cover
    _IMPORT_OK = False

pytestmark = pytest.mark.skipif(not _IMPORT_OK, reason="tessera not built")


def _bipyramid():
    """Two triangles 012, 013 sharing edge 01 — built with EXPLICIT ids
    (build() creates 0,1,2; create_vertex(3) adds 3), the configuration that
    desynced the no-arg id counter."""
    sig = tessera.Signature(2, tessera.Lorentzian)
    st = tessera.Spacetime(tessera.Metric(True, sig), tessera.CDT, 1.0, 1.0,
                           tessera.PREFERRED, tessera.SolidSimplex(2))
    st.build()
    v = {x.get_id(): x for x in st.get_vertex_list().to_vector()}
    v3 = st.create_vertex(3)
    st.create_simplex([v[0], v[1], v3])
    for e in st.get_edge_list().to_vector():
        e.set_length(cmath.sqrt(complex(1.0)))
        e.set_phase(0.0)
    return st


class GrowInteriorSelfEdge(unittest.TestCase):
    def test_grow_never_self_edges_across_seeds(self):
        # Every seed must grow the interior by exactly one fresh vertex — no
        # "cannot create an edge from a vertex to itself", for any seed.
        for seed in range(32):
            st = _bipyramid()
            n0 = st.get_vertex_list().size()
            es = cob.EigenstateSynthesis(st)
            grew = es.grow_interior(seed)            # must not raise
            self.assertTrue(grew, f"seed {seed} failed to grow")
            self.assertEqual(st.get_vertex_list().size(), n0 + 1,
                             f"seed {seed} did not add exactly one vertex")

    def test_new_vertex_id_is_fresh(self):
        # The coned-in vertex must carry a brand-new id, never an alias of an
        # existing one.
        st = _bipyramid()
        before = {v.get_id() for v in st.get_vertex_list().to_vector()}
        cob.EigenstateSynthesis(st).grow_interior(0)
        after = {v.get_id() for v in st.get_vertex_list().to_vector()}
        new_ids = after - before
        self.assertEqual(len(new_ids), 1)            # exactly one new vertex
        self.assertTrue(new_ids.isdisjoint(before))  # and it didn't alias

    def test_repeated_growth_keeps_adding_fresh_vertices(self):
        # Coning repeatedly (each step calls the no-arg createVertex) keeps
        # producing fresh ids — never aliasing back onto the growing complex.
        st = _bipyramid()
        es = cob.EigenstateSynthesis(st)
        n = st.get_vertex_list().size()
        for step in range(3):
            self.assertTrue(es.grow_interior(step + 1))
            n += 1
            self.assertEqual(st.get_vertex_list().size(), n)


if __name__ == "__main__":
    unittest.main()
