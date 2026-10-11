"""Lorentzian (Sorkin) Regge angles + dual Regge action (#247).

All targets are hand-computed.

Euclidean / all-spacelike (real):
  - equilateral-triangle dihedral angle at a vertex = π/3;
  - flat flower (4 right triangles tiling a square around a centre) → centre
    deficit 0;
  - regular tetrahedron surface (closed S²): each vertex's complex deficit = π,
    Σ deficits = 4π (Gauss–Bonnet 2πχ, χ=2); each vertex's circumcentric dual
    cell = √3/4, Σ = √3 = total surface area; and
    dualReggeAction = Σ |*h|·ε_h = √3·π (real).

Lorentzian (complex / boost):
  - a 1+1 triangle with two timelike edges meeting at vertex 0: the dihedral
    there is a boost of rapidity acosh(2/√3) ≈ 0.5493 — captured as a non-zero
    imaginary part, where the existing clamped/Wick `dihedral_angle` loses it;
  - a tetra surface with one timelike edge → the dual action goes complex.

Materialization note: the facet/coface skeleton is built in C++ by constructing
a ReggeSolver (its constructor walks get_facets() down to the hinges).  We do NOT
materialize from Python: get_facets()/get_cofaces() are bound with
return_value_policy::copy, so calling them from Python yields *copies* of the
sub-simplices; registering those copies — each with a partial coface list — onto
the shared vertices corrupts dual_volume().  After constructing the solver we read
the canonical sub-simplices back from get_simplices().

Refs: Regge (1961); Sorkin (Lorentzian angles, arXiv:1908.10022);
Asante–Dittrich–Padua-Argüelles (arXiv:2104.00485, Eq. 10).
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


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
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


def _set_edges(st, mapping, default=None):
    for k, e in _edge_map(st).items():
        if k in mapping:
            e.set_length(cmath.sqrt(complex(mapping[k])))
        elif default is not None:
            e.set_length(cmath.sqrt(complex(default)))
        e.set_phase(0.0)


def _solver(st):
    """Construct a ReggeSolver, which materializes the facet/coface skeleton in
    C++ (canonical sub-simplices + cofaces).  Return the solver."""
    return tessera.ReggeSolver(st, tessera.MatterConfiguration())


def _materialize(st):
    """Build the skeleton in C++ (see module docstring) and discard the solver;
    the sub-simplices now live in st.get_simplices()."""
    _solver(st)


def _by_size(st, n):
    return [s for s in st.get_simplices() if len(s.get_vertices()) == n]


def _vertices_by_id(st):
    """Canonical vertex 0-simplices keyed by vertex id (from getSimplices, not
    the copying get_facets() binding).  Requires a prior _materialize/_solver."""
    return {s.get_vertices()[0].get_id(): s for s in _by_size(st, 1)}


def _triangle(st):
    tris = _by_size(st, 3)
    if not tris:
        raise AssertionError("no triangle simplex (did you materialize?)")
    return tris[0]


def _flat_flower():
    """Unit square tiled by 4 right triangles around centre 0: triangles 012,
    023, 034, 041; centre-corner edges = √(1/2) (s=0.5), sides = 1. The four
    right angles at the centre sum to 2π → zero deficit."""
    st = _solid_triangle()  # triangle 0-1-2
    v = {x.get_id(): x for x in st.get_vertex_list().to_vector()}
    v3 = st.create_vertex(3)
    v4 = st.create_vertex(4)
    st.create_simplex([v[0], v[2], v3])   # 0-2-3
    st.create_simplex([v[0], v3, v4])     # 0-3-4
    st.create_simplex([v[0], v4, v[1]])   # 0-4-1
    _set_edges(st, {(0, 1): 0.5, (0, 2): 0.5, (0, 3): 0.5, (0, 4): 0.5,
                    (1, 2): 1.0, (2, 3): 1.0, (3, 4): 1.0, (1, 4): 1.0})
    return st


def _tetra_surface(timelike_edge=None):
    """Closed S² = the 4 triangles of a regular tetrahedron's surface
    (012, 013, 023, 123), all edges equilateral (s=1) unless one is overridden
    to a timelike value."""
    st = _solid_triangle()  # 0-1-2
    v = {x.get_id(): x for x in st.get_vertex_list().to_vector()}
    v3 = st.create_vertex(3)
    st.create_simplex([v[0], v[1], v3])   # 0-1-3
    st.create_simplex([v[0], v[2], v3])   # 0-2-3
    st.create_simplex([v[1], v[2], v3])   # 1-2-3
    overrides = {} if timelike_edge is None else {timelike_edge: -1.0}
    _set_edges(st, overrides, default=1.0)
    return st


# --------------------------------------------------------------------------- #
# Euclidean / all-spacelike (real)
# --------------------------------------------------------------------------- #
class TestEuclideanAngles(unittest.TestCase):
    def test_equilateral_dihedral_is_pi_over_3(self):
        st = _solid_triangle()
        _set_edges(st, {}, default=1.0)             # equilateral
        _materialize(st)
        tri = _triangle(st)
        v0 = _vertices_by_id(st)[0]
        theta = tri.dihedral_angle(v0)     # interior angle at vertex 0
        self.assertAlmostEqual(theta.real, math.pi / 3.0, places=10)
        self.assertAlmostEqual(theta.imag, 0.0, places=12)

    def test_flat_flower_centre_deficit_zero(self):
        st = _flat_flower()
        _materialize(st)
        centre = _vertices_by_id(st)[0]
        eps = centre.deficit_angle()       # 2π − 4·(π/2) = 0
        self.assertAlmostEqual(eps.real, 0.0, places=8)
        self.assertAlmostEqual(eps.imag, 0.0, places=10)


class TestTetrahedronSurface(unittest.TestCase):
    def test_vertex_deficit_and_gauss_bonnet(self):
        st = _tetra_surface()
        _materialize(st)
        verts = _vertices_by_id(st)
        self.assertEqual(len(verts), 4)
        defs = [v.deficit_angle() for v in verts.values()]
        for e in defs:
            self.assertAlmostEqual(e.real, math.pi, places=8)   # 2π − 3·(π/3)
            self.assertAlmostEqual(e.imag, 0.0, places=10)
        self.assertAlmostEqual(sum(e.real for e in defs),
                               4.0 * math.pi, places=8)          # Gauss–Bonnet

    def test_vertex_dual_cells_partition_the_surface(self):
        # Circumcentric dual: each equilateral face (R²=1/3) contributes a
        # quadrilateral to each of its 3 vertices; per vertex the dual cell is
        # √3/4, and Σ over the 4 vertices = √3 = total surface area (4·√3/4).
        st = _tetra_surface()
        _materialize(st)
        duals = [v.dual_volume() for v in _vertices_by_id(st).values()]
        for dv in duals:
            self.assertAlmostEqual(dv, math.sqrt(3.0) / 4.0, places=8)
        self.assertAlmostEqual(sum(duals), math.sqrt(3.0), places=8)

    def test_dual_regge_action_is_pi_root3(self):
        st = _tetra_surface()
        S = _solver(st).dual_regge_action()
        # Σ_h |*h|·ε_h = (√3/4)·π summed over 4 vertices = √3·π.
        self.assertAlmostEqual(S.real, math.pi * math.sqrt(3.0), places=7)
        self.assertAlmostEqual(S.imag, 0.0, places=9)


# --------------------------------------------------------------------------- #
# Lorentzian (complex / boost)
# --------------------------------------------------------------------------- #
class TestLorentzianBoost(unittest.TestCase):
    def test_timelike_dihedral_is_a_boost(self):
        # Two timelike edges at vertex 0 (s = −4, −3), spacelike opposite (s = 1).
        # ⟨v01,v02⟩ = (s01+s02−s12)/2 = −4; cosh β = 4/√(4·3) = 2/√3.
        st = _solid_triangle()
        _set_edges(st, {(0, 1): -4.0, (0, 2): -3.0, (1, 2): 1.0})
        _materialize(st)
        tri = _triangle(st)
        v0 = _vertices_by_id(st)[0]
        theta = tri.dihedral_angle(v0)
        beta = math.acosh(2.0 / math.sqrt(3.0))     # ≈ 0.54931
        self.assertAlmostEqual(abs(theta.imag), beta, places=8)   # boost survives
        # Real part sits at a light-cone branch point (0 or π).
        self.assertTrue(abs(theta.real) < 1e-8 or
                        abs(theta.real - math.pi) < 1e-8)
        # The boost is carried, not thrown away: the real part sits at a branch
        # point (0 or pi) and the rapidity is the imaginary part. There is no
        # longer a clamped real-typed overload to contrast against (#638).
        self.assertGreater(abs(theta.imag), 1e-2)

    def test_dual_action_goes_complex_with_a_timelike_edge(self):
        st = _tetra_surface(timelike_edge=(0, 1))   # one edge timelike
        S = _solver(st).dual_regge_action()
        self.assertTrue(math.isfinite(S.real) and math.isfinite(S.imag))
        self.assertGreater(abs(S.imag), 1e-6)        # boost ⇒ imaginary part


if __name__ == "__main__":
    unittest.main()
