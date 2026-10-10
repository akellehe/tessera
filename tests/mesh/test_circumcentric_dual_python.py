"""Circumcentric dual cell volumes on Simplex (#246).

Intrinsic, signature-aware circumcenters / dual volumes from the edge lengths
(Cayley-Menger / Gram), verified on the equilateral triangle where everything
is known in closed form:
  * circumradius²  R² = a²/3   (here a² = 1 → 1/3);
  * circumcenter = centroid (barycentric 1/3, 1/3, 1/3);
  * edge circumradius² = a²/4;
  * each vertex's circumcentric dual area = area/3, and the three sum to the
    triangle area (the circumcentric subdivision partitions the cell);
  * top cell's dual is a point (content 1); Hodge star ⋆ = |★σ|/|σ|.

Continuous/spectral only; Lorentzian (signed) content, not Euclidean.
"""

from __future__ import annotations

import math
import unittest

import pytest
import cmath

try:
    import tessera
    _IMPORT_OK = True
except Exception:  # pragma: no cover
    _IMPORT_OK = False

pytestmark = pytest.mark.skipif(not _IMPORT_OK, reason="tessera not built")


def _spacetime(dim, topology):
    sig = tessera.Signature(dim, tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    return tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0,
                             tessera.PREFERRED, topology)


def _solid_triangle():
    st = _spacetime(2, tessera.SolidSimplex(2))
    st.build()
    return st


def _edge_map(st):
    out = {}
    for e in st.get_edge_list().to_vector():
        a, b = e.get_source().get_id(), e.get_target().get_id()
        out[(min(a, b), max(a, b))] = e
    return out


def _triangle(st):
    for s in st.get_simplices():
        if len(s.get_edges()) == 3:
            return s
    raise AssertionError("no triangle simplex in spacetime")


def _set_equilateral(st, a2=1.0):
    for e in _edge_map(st).values():
        e.set_length(cmath.sqrt(complex(a2)))
        e.set_phase(0.0)


class TestCircumcenter(unittest.TestCase):
    def test_equilateral_circumradius_squared(self):
        st = _solid_triangle(); _set_equilateral(st, 1.0)
        self.assertAlmostEqual(_triangle(st).circumradius_squared(),
                               1.0 / 3.0, places=10)

    def test_equilateral_circumcenter_is_centroid(self):
        st = _solid_triangle(); _set_equilateral(st, 1.0)
        for b in _triangle(st).circumcenter_barycentric():
            self.assertAlmostEqual(b, 1.0 / 3.0, places=10)

    def test_edge_circumradius_squared(self):
        st = _solid_triangle(); _set_equilateral(st, 1.0)
        edge_simplex = _triangle(st).get_facets()[0]  # a 1-simplex
        self.assertAlmostEqual(edge_simplex.circumradius_squared(), 0.25, places=10)

    def test_timelike_edge_carries_signed_content(self):
        # A timelike edge (squaredLength < 0) yields a signed circumradius² —
        # no crash, no Euclidean |.|.
        st = _solid_triangle(); _set_equilateral(st, 1.0)
        em = _edge_map(st)
        em[(1, 2)].set_length(cmath.sqrt(complex(-1.0)))
        edge_simplex = None
        for e in _triangle(st).get_facets():
            ids = sorted(v.get_id() for v in e.get_vertices())
            if ids == [1, 2]:
                edge_simplex = e
        self.assertIsNotNone(edge_simplex)
        self.assertAlmostEqual(edge_simplex.circumradius_squared(), -0.25, places=10)


class TestDualVolume(unittest.TestCase):
    def test_top_cell_dual_is_a_point(self):
        st = _solid_triangle(); _set_equilateral(st, 1.0)
        self.assertAlmostEqual(_triangle(st).dual_volume(), 1.0, places=10)

    def test_hodge_star_top_cell(self):
        st = _solid_triangle(); _set_equilateral(st, 1.0)
        tri = _triangle(st)
        self.assertAlmostEqual(tri.hodge_star(), 1.0 / tri.volume(), places=10)

    def test_vertex_dual_partitions_the_area(self):
        st = _solid_triangle(); _set_equilateral(st, 1.0)
        tri = _triangle(st)
        area = tri.volume()  # sqrt(3)/4 for a = 1
        # Materialize the facet/coface skeleton: triangle -> edges -> vertices.
        verts = {}
        for e in tri.get_facets():
            for v in e.get_facets():
                key = tuple(x.get_id() for x in v.get_vertices())
                verts[key] = v
        self.assertEqual(len(verts), 3)
        duals = [v.dual_volume() for v in verts.values()]
        for d in duals:
            self.assertAlmostEqual(d, area / 3.0, places=8)
        self.assertAlmostEqual(sum(duals), area, places=8)


# --------------------------------------------------------------------------- #
# Hand calculations on explicit geometries.
# --------------------------------------------------------------------------- #
def _all_vertex_simplices(st):
    """0-simplices of every top triangle, keyed by vertex id (materializes the
    facet/coface skeleton via getFacets). Snapshot the top simplices first:
    get_facets() registers new simplices, which would invalidate a live
    iteration over get_simplices()."""
    tops = [s for s in st.get_simplices() if len(s.get_edges()) == 3]
    verts = {}
    for s in tops:
        for e in s.get_facets():
            for v in e.get_facets():
                verts[v.get_vertices()[0].get_id()] = v
    return verts


class TestRightTriangleHandCalc(unittest.TestCase):
    """Legs 1 and 1, right angle at vertex 0. Circumcenter sits at the
    hypotenuse midpoint (R² = 1/2); the per-vertex circumcentric dual areas are
    1/4 (right-angle vertex) and 1/8, 1/8 — hand-computed."""

    def _right_triangle(self):
        st = _solid_triangle()
        em = _edge_map(st)
        em[(0, 1)].set_length(cmath.sqrt(complex(1.0)))   # leg
        em[(0, 2)].set_length(cmath.sqrt(complex(1.0)))   # leg
        em[(1, 2)].set_length(cmath.sqrt(complex(2.0)))   # hypotenuse (√2)
        for e in em.values():
            e.set_phase(0.0)
        return st

    def test_circumcenter_is_hypotenuse_midpoint(self):
        st = self._right_triangle()  # keep the spacetime alive (owns the simplices)
        tri = _triangle(st)
        self.assertAlmostEqual(tri.circumradius_squared(), 0.5, places=10)
        order = [v.get_id() for v in tri.get_vertices()]
        bmap = dict(zip(order, tri.circumcenter_barycentric()))
        self.assertAlmostEqual(bmap[0], 0.0, places=10)   # on the hypotenuse
        self.assertAlmostEqual(bmap[1], 0.5, places=10)
        self.assertAlmostEqual(bmap[2], 0.5, places=10)

    def test_per_vertex_dual_areas(self):
        st = self._right_triangle()  # keep the spacetime alive (owns the simplices)
        verts = _all_vertex_simplices(st)
        dv = {vid: v.dual_volume() for vid, v in verts.items()}
        self.assertAlmostEqual(dv[0], 0.25, places=8)    # right-angle vertex
        self.assertAlmostEqual(dv[1], 0.125, places=8)
        self.assertAlmostEqual(dv[2], 0.125, places=8)
        self.assertAlmostEqual(sum(dv.values()), 0.5, places=8)  # = area


# --------------------------------------------------------------------------- #
# Full triangulations: one unit square, split two ways. Every vertex's dual
# area is 1/4 and the total equals the area, independent of the triangulation
# (re-triangulation invariance).
# --------------------------------------------------------------------------- #
def _unit_square_diag(diag):
    st = _solid_triangle()  # triangle 0-1-2
    vmap = {x.get_id(): x for x in st.get_vertex_list().to_vector()}
    v3 = st.create_vertex(3)
    if diag == (0, 2):
        st.create_simplex([vmap[0], vmap[2], v3])     # triangle 0-2-3
        sides = [(0, 1), (1, 2), (2, 3), (0, 3)]
    elif diag == (0, 1):
        st.create_simplex([vmap[0], vmap[1], v3])     # triangle 0-1-3
        sides = [(0, 2), (1, 2), (1, 3), (0, 3)]
    else:
        raise ValueError(diag)
    em = _edge_map(st)
    for key in sides:
        em[key].set_length(cmath.sqrt(complex(1.0)))
        em[key].set_phase(0.0)
    em[diag].set_length(cmath.sqrt(complex(2.0)))   # diagonal = √2
    em[diag].set_phase(0.0)
    return st


class TestUnitSquareTriangulations(unittest.TestCase):
    def _check_square(self, st):
        tops = [s for s in st.get_simplices() if len(s.get_edges()) == 3]
        self.assertEqual(len(tops), 2)
        self.assertAlmostEqual(sum(t.volume() for t in tops), 1.0, places=8)
        verts = _all_vertex_simplices(st)
        self.assertEqual(len(verts), 4)
        duals = [v.dual_volume() for v in verts.values()]
        for d in duals:
            self.assertAlmostEqual(d, 0.25, places=8)        # hand calc
        self.assertAlmostEqual(sum(duals), 1.0, places=8)    # = total area

    def test_diagonal_02(self):
        self._check_square(_unit_square_diag((0, 2)))

    def test_diagonal_01(self):
        self._check_square(_unit_square_diag((0, 1)))


if __name__ == "__main__":
    unittest.main()
