"""Mixed-hinge (light-cone-crossing) branch of the Lorentzian dihedral angle (#581).

A hinge wedge whose two adjacent facet directions have different causal
character (one spacelike, one timelike normal direction — the m=1 case of the
Sorkin/Asante–Dittrich m∈{0,1,2} classification) has opposite-sign Cayley–
Menger diagonal cofactors, ``C_ii·C_jj < 0``.  The true complex denominator
``sqrt(C_ii)·sqrt(C_jj)`` (principal branches) is then purely imaginary, the
cosine ratio is purely imaginary, and the angle is

    theta = pi/2 - i*asinh(C_ij / sqrt(|C_ii*C_jj|)).

The pre-#581 code forced the ratio real (``acos(complex(r, 0))`` over a
one-sidedly sign-fixed real denominator), which corrupted BOTH parts of the
angle on every mixed hinge — generic in CDT (every base-tet triangle of a
(4,1) cell; the seeded toroid has 240/1200 such wedges).

Coverage (the decisive tests of #581 scope item 1):

* the 2D Minkowski analytic repro — triangle ℓ² = (+1, −3, −4) → the mixed
  vertex angle is exactly π/2 + i·asinh(1/√3);
* flat-Minkowski closure, the sign-convention arbiter — a flat vertex star
  covering all four light-cone quadrants sums to 2π + 0i (symmetric star:
  exactly; generic asymmetric star: the four crossing boosts telescope to 0);
* a 4D (4,1)-type cell — base-triangle hinge angle π/2 + i·asinh(1/(4√2)),
  independently derived from the coordinate embedding in R^{3,1};
* relabeling invariance on mixed cells (the #465 canonical-frame property);
* gradient/Hessian on mixed hinges: Euler identities (the sanctioned
  validation — θ and ε are degree-0 homogeneous in ℓ², so Σℓ²·∂ = 0 and
  Σℓ²·∂² = −∂) plus internal consistency against central differences of the
  implemented functions;
* the same-sign regimes are regression-pinned (all-spacelike real angle and
  the |r|>1 boost branch are unchanged).

Known, documented limitation (NOT under test): same-sign (m=0/boost) wedges'
imaginary parts keep the principal-branch sign, so a flat star containing
several rays in one light-cone sector does not close — the wedge boost
orientation is not determined by edge lengths alone (a PT reflection flips it
while preserving every ℓ²).  Stars with one ray per quadrant close exactly.
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
# Fixtures (the test_lorentzian_regge_python.py materialization pattern)
# --------------------------------------------------------------------------- #
def _spacetime(dim, topology):
    sig = tessera.Signature(dim, tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    return tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0,
                             tessera.PREFERRED, topology)


def _solid_simplex(dim):
    st = _spacetime(dim, tessera.SolidSimplex(dim))
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


def _materialize(st):
    """Build the facet/coface skeleton in C++ (ReggeSolver ctor); the canonical
    sub-simplices then live in st.get_simplices()."""
    tessera.ReggeSolver(st, tessera.MatterConfiguration())


def _simplex_by_verts(st, ids):
    want = tuple(sorted(ids))
    for s in st.get_simplices():
        if tuple(sorted(v.get_id() for v in s.get_vertices())) == want:
            return s
    raise AssertionError(f"no simplex {want} (did you materialize?)")


def _mixed_triangle():
    """The audit's minimal repro: 2D Minkowski triangle, ℓ² = (+1, −3, −4).
    At vertex 0 the wedge crosses the light cone: edge (0,1) spacelike,
    edge (0,2) timelike."""
    st = _solid_simplex(2)
    _set_edges(st, {(0, 1): 1.0, (0, 2): -3.0, (1, 2): -4.0})
    return st


def _mink_sq(p, q):
    """Exact Minkowski ℓ² of the segment p→q, (t, x) coordinates, (−,+)."""
    dt, dx = q[0] - p[0], q[1] - p[1]
    return -dt * dt + dx * dx


def _flat_star(points):
    """A flat 2D vertex star: boundary vertex k+1 at ``points[k]`` (a (t, x)
    coordinate), triangles (0, k, k+1) fanned around centre vertex 0, every
    edge carrying its exact Minkowski ℓ².  Returns (st, centre_hinge)."""
    st = _spacetime(2, tessera.SolidSimplex(2))
    st.build()  # triangle 0-1-2 (vertices 0, 1, 2 exist)
    v = {x.get_id(): x for x in st.get_vertex_list().to_vector()}
    n = len(points)
    for k in range(3, n + 1):
        v[k] = st.create_vertex(k)
    for k in range(1, n + 1):
        b = 1 if k == n else k + 1
        if {k, b} != {1, 2}:  # triangle {0,1,2} already built
            st.create_simplex([v[0], v[k], v[b]])
    O = (0.0, 0.0)
    mapping = {}
    for k in range(1, n + 1):
        mapping[(0, k)] = _mink_sq(O, points[k - 1])
        b = 1 if k == n else k + 1
        mapping[(min(k, b), max(k, b))] = _mink_sq(points[k - 1],
                                                   points[b - 1])
    _set_edges(st, mapping)
    _materialize(st)
    return st, _simplex_by_verts(st, [0])


def _cell_41():
    """A (4,1)-type Lorentzian 4-simplex: regular unit base tet {0,1,2,3}
    (ℓ² = +1) + apex 4 with all apex edges timelike (ℓ² = −1, α = 1)."""
    st = _solid_simplex(4)
    mapping = {}
    for i in range(4):
        for j in range(i + 1, 4):
            mapping[(i, j)] = 1.0
        mapping[(i, 4)] = -1.0
    _set_edges(st, mapping)
    _materialize(st)
    return st


def _grad_by_edge(hinge):
    return {tuple(k): v
            for k, v in hinge.deficit_angle_gradient().items()}


def _hess_by_pair(hinge):
    return {(tuple(ke), tuple(kf)): v
            for (ke, kf), v in hinge.deficit_angle_hessian().items()}


def _fd_deficit_grad(st, hinge, key, h=1e-6):
    """Central difference of the implemented deficitAngle in ℓ² of
    the given edge (internal consistency, not a correctness oracle)."""
    e = _edge_map(st)[key]
    orig = (e.get_length() * e.get_length())
    e.set_length(cmath.sqrt(complex(orig + h)))
    fp = complex(hinge.deficit_angle())
    e.set_length(cmath.sqrt(complex(orig - h)))
    fm = complex(hinge.deficit_angle())
    e.set_length(cmath.sqrt(complex(orig)))
    return (fp - fm) / (2.0 * h)


# --------------------------------------------------------------------------- #
# The analytic 2D repro
# --------------------------------------------------------------------------- #
class TestMixedVertexAnalytic(unittest.TestCase):
    def test_mixed_vertex_angle_is_pi_2_plus_i_asinh(self):
        st = _mixed_triangle()
        _materialize(st)
        tri = _simplex_by_verts(st, [0, 1, 2])
        v0 = _simplex_by_verts(st, [0])
        theta = complex(tri.dihedral_angle(v0))
        expect = complex(math.pi / 2.0, math.asinh(1.0 / math.sqrt(3.0)))
        self.assertAlmostEqual(theta.real, expect.real, delta=1e-12)
        self.assertAlmostEqual(theta.imag, expect.imag, delta=1e-12)
        # the crossing is a genuine light-cone crossing, not a boost artifact
        self.assertGreater(theta.imag, 0.0)

    def test_other_vertices_of_the_repro_triangle(self):
        # v1 is also mixed (s10 = +1, s12 = −4) but its Gram inner product
        # vanishes (1 + (−4) − (−3) = 0): exactly π/2, zero boost.  v2 is a
        # same-sign (two-timelike-edge) wedge: the untouched boost branch.
        st = _mixed_triangle()
        _materialize(st)
        tri = _simplex_by_verts(st, [0, 1, 2])
        th1 = complex(tri.dihedral_angle(_simplex_by_verts(st, [1])))
        self.assertAlmostEqual(th1.real, math.pi / 2.0, delta=1e-12)
        self.assertAlmostEqual(th1.imag, 0.0, delta=1e-12)
        th2 = complex(tri.dihedral_angle(_simplex_by_verts(st, [2])))
        # cos θ = ⟨e20,e21⟩/(√(−3)√(−4)) = (−3−4−1)/2 / (−√12) = 2/√3 > 1
        boost = cmath.acos(complex(2.0 / math.sqrt(3.0), 0.0))
        self.assertAlmostEqual(th2.real, boost.real, delta=1e-12)
        self.assertAlmostEqual(th2.imag, boost.imag, delta=1e-12)


# --------------------------------------------------------------------------- #
# Flat-Minkowski closure — the sign-convention arbiter
# --------------------------------------------------------------------------- #
class TestFlatMinkowskiClosure(unittest.TestCase):
    def test_symmetric_four_quadrant_star_closes_exactly(self):
        # (±1,0),(0,±1) in (t,x): every wedge crosses one light-cone quadrant
        # boundary; the null hypotenuses zero the Gram inner products, so each
        # interior angle is exactly π/2 + 0i and the deficit is exactly 0.
        pts = [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)]
        st, centre = _flat_star(pts)
        angles = []
        for tri_ids in [(0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 1, 4)]:
            tri = _simplex_by_verts(st, tri_ids)
            angles.append(complex(tri.dihedral_angle(centre)))
        for th in angles:
            self.assertEqual(th.real, math.pi / 2.0)
            self.assertEqual(th.imag, 0.0)
        total = sum(angles)
        self.assertEqual(total.real, 2.0 * math.pi)
        self.assertEqual(total.imag, 0.0)
        eps = complex(centre.deficit_angle())
        self.assertEqual(eps.real, 0.0)
        self.assertEqual(eps.imag, 0.0)

    def test_asymmetric_flat_star_deficit_vanishes(self):
        # Generic flat star, one ray per light-cone quadrant (in (t,x)):
        # R-sector (x>|t|), F (t>|x|), L (x<−|t|), P (t<−|x|), counterclockwise.
        # All four wedges are m=1 crossings; the asinh boosts are nonzero and
        # must telescope to zero — this pins the crossing branch's sign.
        pts = [(0.25, 1.3), (1.6, 0.4), (0.3, -1.5), (-1.4, 0.5)]
        st, centre = _flat_star(pts)
        eps = complex(centre.deficit_angle())
        self.assertAlmostEqual(eps.real, 0.0, delta=1e-12)
        self.assertAlmostEqual(eps.imag, 0.0, delta=1e-12)
        # every wedge is a genuine crossing (Re = π/2) with a nonzero boost
        boosts = []
        for tri_ids in [(0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 1, 4)]:
            tri = _simplex_by_verts(st, tri_ids)
            th = complex(tri.dihedral_angle(centre))
            self.assertAlmostEqual(th.real, math.pi / 2.0, delta=1e-12)
            boosts.append(th.imag)
        self.assertGreater(max(abs(b) for b in boosts), 0.05)
        self.assertAlmostEqual(sum(boosts), 0.0, delta=1e-12)

    def test_more_asymmetric_flat_stars(self):
        # A few more generic one-per-quadrant stars (different radii and
        # rapidities), all must close to 2π + 0i.
        cases = [
            [(0.9, 2.1), (2.3, -0.7), (0.05, -0.4), (-1.1, 0.15)],
            [(-0.6, 0.8), (1.05, 1.0), (0.55, -0.85), (-2.5, -2.0)],
        ]
        for pts in cases:
            st, centre = _flat_star(pts)
            eps = complex(centre.deficit_angle())
            self.assertAlmostEqual(eps.real, 0.0, delta=1e-11)
            self.assertAlmostEqual(eps.imag, 0.0, delta=1e-11)


# --------------------------------------------------------------------------- #
# 4D mixed hinge on a (4,1)-type cell
# --------------------------------------------------------------------------- #
class TestMixedHinge4D(unittest.TestCase):
    # Independent derivation (R^{3,1} embedding, signature (−,+,+,+)): unit
    # regular base tet at t=0, apex over the centroid at t = √(11/8) so every
    # apex edge has ℓ² = −1.  At the base-triangle hinge {0,1,2} the orthogonal
    # (z,t) plane carries metric (+,−); the away-from-hinge facet directions
    # are u = (√(2/3), 0) (spacelike, base tet) and w = (√(2/3)/4, √(11/8))
    # (timelike, side tet): ⟨u,u⟩ = 2/3, ⟨w,w⟩ = −4/3, ⟨u,w⟩ = 1/6, so
    # cos θ = −i/(4√2) and θ = π/2 + i·asinh(1/(4√2)).
    EXPECT = complex(math.pi / 2.0, math.asinh(1.0 / (4.0 * math.sqrt(2.0))))

    def test_base_triangle_hinge_angle(self):
        st = _cell_41()
        cell = _simplex_by_verts(st, [0, 1, 2, 3, 4])
        hinge = _simplex_by_verts(st, [0, 1, 2])
        theta = complex(cell.dihedral_angle(hinge))
        self.assertAlmostEqual(theta.real, self.EXPECT.real, delta=1e-12)
        self.assertAlmostEqual(theta.imag, self.EXPECT.imag, delta=1e-12)

    def test_every_base_triangle_is_mixed_and_equal(self):
        # All four base triangles are equivalent under the tet symmetry.
        st = _cell_41()
        cell = _simplex_by_verts(st, [0, 1, 2, 3, 4])
        for ids in [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]:
            theta = complex(cell.dihedral_angle(
                _simplex_by_verts(st, ids)))
            self.assertAlmostEqual(theta.real, self.EXPECT.real, delta=1e-12)
            self.assertAlmostEqual(theta.imag, self.EXPECT.imag, delta=1e-12)


# --------------------------------------------------------------------------- #
# Relabeling invariance on mixed cells (the #465 canonical-frame property)
# --------------------------------------------------------------------------- #
class TestRelabelingInvariance(unittest.TestCase):
    def _cell_41_with_ids(self, ids, order):
        """The _cell_41 geometry with vertex ids ``ids[abstract]`` and the
        cell's stored vertex order permuted by ``order``."""
        st = _spacetime(4, tessera.SolidSimplex(4))
        v = {}
        for a in range(5):
            v[a] = st.create_vertex(ids[a])
        st.create_simplex([v[a] for a in order])
        mapping = {}
        for i in range(4):
            for j in range(i + 1, 4):
                mapping[(min(ids[i], ids[j]), max(ids[i], ids[j]))] = 1.0
            mapping[(min(ids[i], ids[4]), max(ids[i], ids[4]))] = -1.0
        _set_edges(st, mapping)
        _materialize(st)
        return st, v

    def test_permuted_ids_and_stored_order_give_identical_angles(self):
        base_st = _cell_41()
        base_cell = _simplex_by_verts(base_st, [0, 1, 2, 3, 4])
        base_theta = complex(base_cell.dihedral_angle(
            _simplex_by_verts(base_st, [0, 1, 2])))

        perms = [
            ([10, 3, 7, 42, 0], (4, 2, 0, 3, 1)),
            ([5, 90, 11, 2, 33], (1, 3, 4, 0, 2)),
        ]
        for ids, order in perms:
            st, v = self._cell_41_with_ids(ids, order)
            cell = _simplex_by_verts(st, ids)
            hinge = _simplex_by_verts(st, [ids[0], ids[1], ids[2]])
            theta = complex(cell.dihedral_angle(hinge))
            self.assertAlmostEqual(theta.real, base_theta.real, delta=1e-13)
            self.assertAlmostEqual(theta.imag, base_theta.imag, delta=1e-13)

    def test_mixed_2d_triangle_relabeled(self):
        st = _spacetime(2, tessera.SolidSimplex(2))
        v = {a: st.create_vertex(i) for a, i in enumerate([17, 4, 99])}
        st.create_simplex([v[2], v[0], v[1]])
        _set_edges(st, {(4, 17): 1.0, (17, 99): -3.0, (4, 99): -4.0})
        _materialize(st)
        tri = _simplex_by_verts(st, [4, 17, 99])
        theta = complex(tri.dihedral_angle(
            _simplex_by_verts(st, [17])))
        expect = complex(math.pi / 2.0, math.asinh(1.0 / math.sqrt(3.0)))
        self.assertAlmostEqual(theta.real, expect.real, delta=1e-12)
        self.assertAlmostEqual(theta.imag, expect.imag, delta=1e-12)


# --------------------------------------------------------------------------- #
# Gradient and Hessian on mixed hinges
# --------------------------------------------------------------------------- #
class TestMixedHingeGradient(unittest.TestCase):
    def _euler_gradient(self, st, hinge, tol=1e-10):
        """θ (hence ε) is degree-0 homogeneous in ℓ²: Σ_e ℓ²_e ∂ε/∂ℓ²_e = 0,
        real AND imaginary parts (the repo's sanctioned validation)."""
        em = _edge_map(st)
        total = complex(0.0, 0.0)
        seen = 0
        for key, g in _grad_by_edge(hinge).items():
            total += (em[key].get_length() * em[key].get_length()) * complex(g)
            seen += 1
        self.assertGreater(seen, 0)
        self.assertAlmostEqual(total.real, 0.0, delta=tol)
        self.assertAlmostEqual(total.imag, 0.0, delta=tol)

    def test_euler_identity_2d_mixed(self):
        st = _mixed_triangle()
        _materialize(st)
        self._euler_gradient(st, _simplex_by_verts(st, [0]))

    def test_euler_identity_4d_mixed(self):
        st = _cell_41()
        self._euler_gradient(st, _simplex_by_verts(st, [0, 1, 2]))

    def test_euler_identity_flat_star_centre(self):
        st, centre = _flat_star([(0.25, 1.3), (1.6, 0.4),
                                 (0.3, -1.5), (-1.4, 0.5)])
        self._euler_gradient(st, centre)

    def test_gradient_matches_fd_2d_mixed(self):
        st = _mixed_triangle()
        _materialize(st)
        hinge = _simplex_by_verts(st, [0])
        grad = _grad_by_edge(hinge)
        for key in [(0, 1), (0, 2), (1, 2)]:
            fd = _fd_deficit_grad(st, hinge, key)
            g = complex(grad[key])
            self.assertAlmostEqual(g.real, fd.real, delta=5e-6)
            self.assertAlmostEqual(g.imag, fd.imag, delta=5e-6)

    def test_gradient_matches_fd_4d_mixed(self):
        st = _cell_41()
        hinge = _simplex_by_verts(st, [0, 1, 2])
        grad = _grad_by_edge(hinge)
        for key in [(0, 1), (0, 3), (0, 4), (2, 4)]:
            fd = _fd_deficit_grad(st, hinge, key)
            g = complex(grad[key])
            self.assertAlmostEqual(g.real, fd.real, delta=5e-6)
            self.assertAlmostEqual(g.imag, fd.imag, delta=5e-6)

    def test_gradient_is_complex_on_the_crossing(self):
        # the crossing branch must move the boost: some ∂ε/∂ℓ² has Im ≠ 0
        st = _cell_41()
        grad = _grad_by_edge(_simplex_by_verts(st, [0, 1, 2]))
        self.assertGreater(max(abs(complex(g).imag) for g in grad.values()),
                           1e-3)


class TestMixedHingeHessian(unittest.TestCase):
    def _euler_hessian(self, st, hinge, tol=1e-8):
        """∂ε/∂ℓ²_e is degree-(−1) homogeneous: Σ_f ℓ²_f ∂²ε/∂ℓ²_e∂ℓ²_f =
        −∂ε/∂ℓ²_e for every e."""
        em = _edge_map(st)
        grad = _grad_by_edge(hinge)
        hess = _hess_by_pair(hinge)
        rows = {}
        for (ke, kf), v in hess.items():
            rows.setdefault(ke, complex(0.0, 0.0))
            rows[ke] += (em[kf].get_length() * em[kf].get_length()) * complex(v)
        self.assertGreater(len(rows), 0)
        for ke, total in rows.items():
            want = -complex(grad[ke])
            self.assertAlmostEqual(total.real, want.real, delta=tol)
            self.assertAlmostEqual(total.imag, want.imag, delta=tol)

    def test_euler_identity_2d_mixed(self):
        st = _mixed_triangle()
        _materialize(st)
        self._euler_hessian(st, _simplex_by_verts(st, [0]))

    def test_euler_identity_4d_mixed(self):
        st = _cell_41()
        self._euler_hessian(st, _simplex_by_verts(st, [0, 1, 2]))

    def test_hessian_symmetric_and_matches_fd_of_gradient(self):
        st = _mixed_triangle()
        _materialize(st)
        hinge = _simplex_by_verts(st, [0])
        hess = _hess_by_pair(hinge)
        em = _edge_map(st)
        keys = [(0, 1), (0, 2), (1, 2)]
        h = 1e-6
        for ke in keys:
            for kf in keys:
                self.assertAlmostEqual(
                    abs(complex(hess[(ke, kf)]) - complex(hess[(kf, ke)])),
                    0.0, delta=1e-12)
                e = em[kf]
                orig = (e.get_length() * e.get_length())
                e.set_length(cmath.sqrt(complex(orig + h)))
                gp = complex(_grad_by_edge(hinge)[ke])
                e.set_length(cmath.sqrt(complex(orig - h)))
                gm = complex(_grad_by_edge(hinge)[ke])
                e.set_length(cmath.sqrt(complex(orig)))
                fd = (gp - gm) / (2.0 * h)
                got = complex(hess[(ke, kf)])
                self.assertAlmostEqual(got.real, fd.real, delta=5e-5)
                self.assertAlmostEqual(got.imag, fd.imag, delta=5e-5)


# --------------------------------------------------------------------------- #
# The action pipeline on a mixed-hinge complex
# --------------------------------------------------------------------------- #
class TestActionOnMixedComplex(unittest.TestCase):
    def test_action_gradient_exact_matches_fd_with_mixed_hinges(self):
        # tetra surface with one timelike edge: its vertex hinges include
        # genuine crossings; actionGradientExact must match a central
        # difference of dual_regge_action (Re and Im) through them.
        st = _spacetime(2, tessera.SolidSimplex(2))
        st.build()
        v = {x.get_id(): x for x in st.get_vertex_list().to_vector()}
        v3 = st.create_vertex(3)
        st.create_simplex([v[0], v[1], v3])
        st.create_simplex([v[0], v[2], v3])
        st.create_simplex([v[1], v[2], v3])
        _set_edges(st, {(0, 1): -1.0}, default=1.0)
        solver = tessera.ReggeSolver(st, tessera.MatterConfiguration())
        edges = st.get_edge_list().to_vector()
        grad = solver.action_gradient_exact()
        h = 1e-6
        for i, e in enumerate(edges):
            orig = (e.get_length() * e.get_length())
            e.set_length(cmath.sqrt(complex(orig + h)))
            sp = complex(solver.dual_regge_action())
            e.set_length(cmath.sqrt(complex(orig - h)))
            sm = complex(solver.dual_regge_action())
            e.set_length(cmath.sqrt(complex(orig)))
            fd = (sp - sm) / (2.0 * h)
            g = complex(grad[i])
            self.assertAlmostEqual(g.real, fd.real, delta=2e-5)
            self.assertAlmostEqual(g.imag, fd.imag, delta=2e-5)


if __name__ == "__main__":
    unittest.main()
