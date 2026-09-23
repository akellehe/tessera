"""The primal Regge action on meshes of three, four and five dimensions.

The primal Regge action is S = Σ_h |h| ε_h: the sum over the hinges h of the
mesh (its codimension-two faces, the (d-2)-simplices of a d-dimensional mesh)
of the (d-2)-content |h| of the hinge times its deficit angle
ε_h = 2π − Σ_σ θ_h^(σ), where σ runs over the top cells containing h and
θ_h^(σ) is the dihedral angle of σ at h. ``ReggeSolver.hingeContent`` is |h|,
evaluated as ``Simplex.volume``: sqrt(det G)/(d-2)! under the principal complex
root of the signed Gram determinant. An edge hinge (d = 3) therefore
contributes its length, a triangular hinge (d = 4) its Heron area, a
tetrahedral hinge (d = 5) its volume, and a timelike hinge an imaginary content.

All targets are hand-computed.

Three dimensions (Euclidean): three regular unit tetrahedra 0123, 0134 and
0142 fanned around the shared edge 01. With α = arccos(1/3) the dihedral angle
of a regular tetrahedron, the interior edge 01 (three cells) has deficit
2π − 3α, the six spoke edges 02, 03, 04, 12, 13, 14 (two cells each) have
2π − 2α, and the three rim edges 23, 34, 42 (one cell each) have 2π − α. Every
edge has unit length, so S = 10·2π − 18α = 20π − 18α ≈ 40.67.

Four dimensions (Lorentzian): a single 4-simplex. Every hinge is a triangle,
so ``hingeContent`` equals Heron's formula: √3/4 on a unit equilateral triangle
and i√5/4 on the triangle with squared sides (−1, 1, 1) that contains one
timelike edge. ``reggeAction`` equals the Heron-weighted deficit sum, which is
what the area-weighted sum gave. On the all-spacelike cell each of the ten
triangles sees the regular 4-simplex dihedral arccos(1/4) once, so
S = 10 · (√3/4) · (2π − arccos(1/4)).

Five dimensions (Euclidean): a single regular unit 5-simplex. Every one of the
fifteen hinges is a regular unit tetrahedron of content 1/(6√2), each in one
cell with the regular 5-simplex dihedral angle arccos(1/5), so
S = 15 · (2π − arccos(1/5)) / (6√2).

Materialization: the facet/coface skeleton is built in C++ by constructing a
ReggeSolver (its constructor walks getFacets() down to the hinges); the
canonical sub-simplices are then read back from getSimplices(). See
tests/mesh/test_lorentzian_regge_python.py for why this is not done from Python.

Refs: Regge (1961); Sorkin (Lorentzian angles, arXiv:1908.10022).
"""

from __future__ import annotations

import cmath
import math
import unittest

import pytest

try:
    import tessera
    _IMPORT_OK = True
except Exception:  # pragma: no cover
    _IMPORT_OK = False

pytestmark = pytest.mark.skipif(not _IMPORT_OK, reason="tessera not built")


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def _spacetime(dim, signature, topology):
    sig = tessera.Signature(dim, signature)
    metric = tessera.Metric(True, sig)
    return tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0,
                             tessera.PREFERRED, topology)


def _edge_map(st):
    out = {}
    for e in st.getEdgeList().toVector():
        a, b = e.getSource().getId(), e.getTarget().getId()
        out[(min(a, b), max(a, b))] = e
    return out


def _set_edges(st, squared, overrides=None):
    """Give every edge the signed squared length ``squared`` (its length is the
    principal root: real for spacelike, imaginary for timelike), except the
    (min id, max id) pairs in ``overrides``."""
    overrides = overrides or {}
    for k, e in _edge_map(st).items():
        e.setLength(cmath.sqrt(complex(overrides.get(k, squared))))
        e.setPhase(0.0)


def _solver(st):
    """Construct a ReggeSolver, which materializes the facet/coface skeleton in
    C++ (canonical sub-simplices + cofaces). Return the solver."""
    return tessera.ReggeSolver(st, tessera.MatterConfiguration())


def _ids(s):
    return tuple(sorted(v.getId() for v in s.getVertices()))


def _hinges(st, dim):
    """The genuine hinges of a dim-dimensional mesh: the registered simplices
    on dim-1 vertices that are faces of a current top cell. Requires a prior
    _solver(st)."""
    return [s for s in st.getSimplices()
            if len(s.getVertices()) == dim - 1 and s.hasTopCoface()]


def _heron(tri):
    """Heron's formula on the three signed squared edge lengths of a triangle,
    under the principal complex root: the area, imaginary when the radicand
    is negative."""
    a2, b2, c2 = [e.getLength() ** 2 for e in tri.getEdges()]
    radicand = 2.0 * (a2 * b2 + b2 * c2 + c2 * a2) - (a2 * a2 + b2 * b2 + c2 * c2)
    return cmath.sqrt(radicand) / 4.0


def _fan_of_three_tetrahedra():
    """Three regular unit tetrahedra 0123, 0134, 0142 sharing the edge 01, on
    a three-dimensional Euclidean mesh. The edge 01 is interior (three cells
    around it); every other edge lies in one or two cells."""
    st = _spacetime(3, tessera.Euclidean, tessera.SolidSimplex(3))
    st.build()                                    # tetrahedron 0-1-2-3
    v = {x.getId(): x for x in st.getVertexList().toVector()}
    v4 = st.createVertex(4)
    st.createSimplex([v[0], v[1], v[3], v4])      # 0-1-3-4
    st.createSimplex([v[0], v[1], v4, v[2]])      # 0-1-4-2
    _set_edges(st, 1.0)
    return st


def _solid_four_simplex(timelike_edge=None):
    """A single 4-simplex on a four-dimensional Lorentzian mesh, every squared
    edge length 1 unless one edge is overridden to the timelike value -1."""
    st = _spacetime(4, tessera.Lorentzian, tessera.SolidSimplex(4))
    st.build()
    overrides = {} if timelike_edge is None else {timelike_edge: -1.0}
    _set_edges(st, 1.0, overrides)
    return st


def _solid_five_simplex():
    """A single regular unit 5-simplex on a five-dimensional Euclidean mesh."""
    st = _spacetime(5, tessera.Euclidean, tessera.SolidSimplex(5))
    st.build()
    _set_edges(st, 1.0)
    return st


# --------------------------------------------------------------------------- #
# Three dimensions: the hinges are edges
# --------------------------------------------------------------------------- #
class TestThreeDimensionalMesh(unittest.TestCase):
    ALPHA = math.acos(1.0 / 3.0)   # regular-tetrahedron dihedral angle

    def test_interior_edge_deficit_is_two_pi_minus_three_dihedrals(self):
        st = _fan_of_three_tetrahedra()
        _solver(st)
        hinges = {_ids(h): h for h in _hinges(st, 3)}
        self.assertEqual(len(hinges), 10)          # the ten edges of the fan
        eps = hinges[(0, 1)].deficitAngle()
        self.assertAlmostEqual(eps.real, 2.0 * math.pi - 3.0 * self.ALPHA,
                               places=8)
        self.assertAlmostEqual(eps.imag, 0.0, places=10)

    def test_hinge_content_is_the_edge_length(self):
        st = _fan_of_three_tetrahedra()
        _solver(st)
        edges = _edge_map(st)
        for h in _hinges(st, 3):
            content = tessera.ReggeSolver.hingeContent(h)
            length = edges[_ids(h)].getLength()
            self.assertAlmostEqual(content.real, length.real, places=12)
            self.assertAlmostEqual(content.imag, length.imag, places=12)
            self.assertAlmostEqual(content.real, 1.0, places=12)

    def test_action_is_the_length_weighted_deficit_sum(self):
        st = _fan_of_three_tetrahedra()
        solver = _solver(st)
        S = solver.reggeAction()
        self.assertTrue(cmath.isfinite(S))
        self.assertGreater(abs(S), 1.0)            # not the zero area() gave
        # Hand sum over the hinges: edge length times deficit angle.
        edges = _edge_map(st)
        by_hand = sum(edges[_ids(h)].getLength() * h.deficitAngle()
                      for h in _hinges(st, 3))
        self.assertAlmostEqual(S.real, by_hand.real, places=9)
        self.assertAlmostEqual(S.imag, by_hand.imag, places=9)
        # Closed form: one edge in three cells, six in two, three in one.
        self.assertAlmostEqual(S.real, 20.0 * math.pi - 18.0 * self.ALPHA,
                               places=7)
        self.assertAlmostEqual(S.imag, 0.0, places=9)


# --------------------------------------------------------------------------- #
# Four dimensions: the hinges are triangles, so the content is Heron's area
# --------------------------------------------------------------------------- #
class TestFourDimensionalMesh(unittest.TestCase):
    def test_spacelike_triangle_content_is_heron(self):
        st = _solid_four_simplex()
        _solver(st)
        tris = _hinges(st, 4)
        self.assertEqual(len(tris), 10)            # C(5, 3) faces
        for t in tris:
            content = tessera.ReggeSolver.hingeContent(t)
            heron = _heron(t)
            self.assertAlmostEqual(content.real, heron.real, places=12)
            self.assertAlmostEqual(content.imag, heron.imag, places=12)
            self.assertAlmostEqual(content.real, math.sqrt(3.0) / 4.0,
                                   places=12)
            self.assertAlmostEqual(content.imag, 0.0, places=12)

    def test_timelike_triangle_content_is_imaginary_heron(self):
        st = _solid_four_simplex(timelike_edge=(0, 1))
        _solver(st)
        tris = {_ids(t): t for t in _hinges(st, 4)}
        for k in (2, 3, 4):                        # the triangles on edge 01
            t = tris[(0, 1, k)]
            content = tessera.ReggeSolver.hingeContent(t)
            heron = _heron(t)
            # Squared sides (-1, 1, 1): radicand 2(-1+1-1) - 3 = -5.
            self.assertAlmostEqual(content.real, 0.0, places=12)
            self.assertAlmostEqual(content.imag, math.sqrt(5.0) / 4.0,
                                   places=12)
            self.assertAlmostEqual(content.real, heron.real, places=12)
            self.assertAlmostEqual(content.imag, heron.imag, places=12)

    def test_action_is_the_heron_weighted_deficit_sum(self):
        for st in (_solid_four_simplex(),
                   _solid_four_simplex(timelike_edge=(0, 1))):
            solver = _solver(st)
            S = solver.reggeAction()
            self.assertTrue(cmath.isfinite(S))
            by_hand = sum(_heron(t) * t.deficitAngle() for t in _hinges(st, 4))
            self.assertAlmostEqual(S.real, by_hand.real, places=9)
            self.assertAlmostEqual(S.imag, by_hand.imag, places=9)

    def test_all_spacelike_action_closed_form(self):
        st = _solid_four_simplex()
        S = _solver(st).reggeAction()
        expected = 10.0 * (math.sqrt(3.0) / 4.0) * (2.0 * math.pi - math.acos(0.25))
        self.assertAlmostEqual(S.real, expected, places=7)
        self.assertAlmostEqual(S.imag, 0.0, places=9)


# --------------------------------------------------------------------------- #
# Five dimensions: the hinges are tetrahedra
# --------------------------------------------------------------------------- #
class TestFiveDimensionalMesh(unittest.TestCase):
    TET_VOLUME = 1.0 / (6.0 * math.sqrt(2.0))     # regular unit tetrahedron

    def test_tetrahedral_hinge_content_is_the_volume(self):
        st = _solid_five_simplex()
        _solver(st)
        tets = _hinges(st, 5)
        self.assertEqual(len(tets), 15)            # C(6, 4) faces
        for t in tets:
            content = tessera.ReggeSolver.hingeContent(t)
            self.assertAlmostEqual(content.real, self.TET_VOLUME, places=12)
            self.assertAlmostEqual(content.imag, 0.0, places=12)

    def test_action_closed_form(self):
        st = _solid_five_simplex()
        S = _solver(st).reggeAction()
        expected = 15.0 * self.TET_VOLUME * (2.0 * math.pi - math.acos(0.2))
        self.assertAlmostEqual(S.real, expected, places=7)
        self.assertAlmostEqual(S.imag, 0.0, places=9)


if __name__ == "__main__":
    unittest.main()
