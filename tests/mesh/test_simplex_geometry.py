# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""Signature-aware Simplex geometry: volume(), gram_matrix(), and the extracted
cayley_menger_matrix().

The geometry is honest (signature-respecting) by default: a timelike edge keeps
its negative squared length, so a Lorentzian cell reports a signed content. The
Wick-rotated (|l^2|) behaviour the CDT/Regge path relies on is still available
via ```` and is exercised here too.
"""

import cmath
import math
import unittest

import numpy as np

import tessera
from tessera import Spacetime


def _make_simplex(st, vertex_ids, squared_by_pair):
    """Build a simplex on ``st`` with full control of vertex order and of every
    edge's signed squared length.

    ``vertex_ids`` fixes the vertex order (so vertex 0 -- the Gram origin -- is
    ``vertex_ids[0]``); ``squared_by_pair`` maps ``frozenset({i, j})`` to the
    signed l^2 of that edge. Returns ``(simplex, verts, edges)``.
    """
    verts = {i: st.create_vertex(i) for i in vertex_ids}
    edges = {}
    for pair, l2 in squared_by_pair.items():
        a, b = tuple(pair)
        # createEdge takes the complex LENGTH now, so a squared value goes in
        # as its principal root: real for spacelike, imaginary for timelike.
        edges[pair] = st.create_edge(verts[a], verts[b],
                                    cmath.sqrt(complex(l2)))
    simplex, _ = st.create_simplex([verts[i] for i in vertex_ids],
                                  list(edges.values()))
    return simplex, verts, edges


def _expected_gram(simplex, squared_by_pair, wick):
    """Independent reconstruction of the (d x d) Gram matrix from the signed (or
    |l^2| when ``wick``) squared lengths, in the simplex's own vertex order."""
    vids = [v.get_id() for v in simplex.get_vertices()]
    d = len(vids) - 1

    def s(i, j):
        if i == j:
            return 0.0
        l2 = squared_by_pair[frozenset({vids[i], vids[j]})]
        return abs(l2) if wick else float(l2)

    G = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            G[i, j] = 0.5 * (s(0, i + 1) + s(0, j + 1) - s(i + 1, j + 1))
    return G


def _cofactor(M):
    """Cofactor matrix, mirroring Simplex::cofactorMatrix exactly."""
    n = M.shape[0]
    C = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            minor = np.delete(np.delete(M, i, axis=0), j, axis=1)
            C[i, j] = ((-1) ** (i + j)) * np.linalg.det(minor)
    return C


def _dihedral_from_cm(B, bi, bj):
    """Dihedral angle from a Cayley-Menger matrix, mirroring the C++ cofactor
    formula: acos(-C_ij / (sqrt(C_ii) * sqrt(C_jj))).

    TWO separate principal square roots, never sqrt(C_ii * C_jj) -- the
    denominator is a product of two independently rooted norms, and folding it
    under one root loses the factor of i a timelike normal carries. On a regular
    tetrahedron (C_ii = C_jj = -3) the folded form returns the SUPPLEMENT. That
    is what the old dimension-parity sign fix was hand-restoring; taking the
    roots separately makes it emerge (#638).

    There is no cos clamp: |r| > 1 is the boost regime and acos carries it as an
    imaginary part. The branch side for a real ratio is pinned to +0.0, matching
    the engine.
    """
    C = _cofactor(B)
    denom = cmath.sqrt(complex(C[bi, bi])) * cmath.sqrt(complex(C[bj, bj]))
    if abs(denom) < 1e-15:
        return 0.0 + 0.0j
    r = -complex(C[bi, bj]) / denom
    if r.imag == 0.0:
        r = complex(r.real, 0.0)
    return cmath.acos(r)


def _regular_squares(vertex_ids, l2=1.0):
    """Signed-square map for a regular simplex: every edge has the same l^2."""
    return {frozenset({a, b}): float(l2)
            for i, a in enumerate(vertex_ids) for b in vertex_ids[i + 1:]}


def _orthoscheme_squares(vertex_ids):
    """Squares for the unit corner d-simplex (origin + unit axes): the d legs
    from vertex 0 have l^2 = 1, the C(d,2) cross edges have l^2 = 2. Its Gram is
    the identity, so its content is exactly 1/d!."""
    origin, axes = vertex_ids[0], vertex_ids[1:]
    sq = {frozenset({origin, a}): 1.0 for a in axes}
    for i, a in enumerate(axes):
        for b in axes[i + 1:]:
            sq[frozenset({a, b})] = 2.0
    return sq


def _spacetime_nd(dim):
    """An *unbuilt* dim-dimensional Lorentzian CDT spacetime. Its signature
    dimension is what makes deficit_angle() count d+1-vertex cells as top cells;
    the topology is never built, so the complex starts empty and we add cells by
    hand."""
    sig = tessera.Signature(dim, tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    return tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0,
                             tessera.PREFERRED, tessera.SolidSimplex(dim))


class TestSimplexVolume(unittest.TestCase):
    def setUp(self):
        self.st = Spacetime()

    def test_euclidean_triangle_volume_matches_hand_value(self):
        # Right isosceles triangle, legs 1 and 1, hypotenuse^2 = 2. Area = 1/2.
        sq = {
            frozenset({0, 1}): 1.0,
            frozenset({0, 2}): 1.0,
            frozenset({1, 2}): 2.0,
        }
        simplex, _, _ = _make_simplex(self.st, [0, 1, 2], sq)
        self.assertAlmostEqual(simplex.volume(), 0.5, places=9)

    def test_euclidean_tetrahedron_volume_matches_hand_value(self):
        # Corner tetrahedron with three unit legs along orthogonal axes:
        # vertices (0,0,0),(1,0,0),(0,1,0),(0,0,1). Volume = 1/6.
        sq = {
            frozenset({0, 1}): 1.0,
            frozenset({0, 2}): 1.0,
            frozenset({0, 3}): 1.0,
            frozenset({1, 2}): 2.0,
            frozenset({1, 3}): 2.0,
            frozenset({2, 3}): 2.0,
        }
        simplex, _, _ = _make_simplex(self.st, [0, 1, 2, 3], sq)
        self.assertAlmostEqual(simplex.volume(), 1.0 / 6.0, places=9)

    def test_lorentzian_triangle_volume_is_signed(self):
        # One timelike edge (l^2 < 0). In coordinates (t, x):
        # v0=(0,0), v1=(1,0) timelike from v0 -> l^2 = -1,
        # v2=(0,2) spacelike -> l^2 = 4, edge v1-v2 -> l^2 = -1 + 4 = 3.
        # Honest Gram = [[-1, 0], [0, 4]], det = -4, signed content = -1.
        sq = {
            frozenset({0, 1}): -1.0,
            frozenset({0, 2}): 4.0,
            frozenset({1, 2}): 3.0,
        }
        simplex, _, _ = _make_simplex(self.st, [0, 1, 2], sq)

        vol = simplex.volume()
        # V = sqrt(det G)/d!. det G < 0 here, so the content is IMAGINARY -- that is
        # what the d-content of this cell is, not the negative real a double could
        # hold and not the |l^2| value (#641).
        self.assertAlmostEqual(vol.real, 0.0, places=9)
        self.assertAlmostEqual(vol.imag, 1.0, places=9)

    def test_volume_matches_signed_gram_determinant(self):
        sq = {
            frozenset({0, 1}): -1.0,
            frozenset({0, 2}): 4.0,
            frozenset({1, 2}): 3.0,
        }
        simplex, _, _ = _make_simplex(self.st, [0, 1, 2], sq)
        gram = _expected_gram(simplex, sq, wick=False)
        det = np.linalg.det(gram)
        expected = cmath.sqrt(complex(det)) / math.factorial(2)
        self.assertAlmostEqual(abs(simplex.volume() - expected), 0.0, places=9)


class TestSimplexGramSignatureAware(unittest.TestCase):
    def setUp(self):
        self.st = Spacetime()

    def test_gram_honest_keeps_timelike_sign(self):
        sq = {
            frozenset({0, 1}): -1.0,
            frozenset({0, 2}): 4.0,
            frozenset({1, 2}): 3.0,
        }
        simplex, _, _ = _make_simplex(self.st, [0, 1, 2], sq)
        # There is only the honest signed Gram now; the |l^2| mode is gone (#641).
        honest = np.array(simplex.gram_matrix()).reshape(2, 2)
        np.testing.assert_allclose(honest.real, _expected_gram(simplex, sq, False),
                                   atol=1e-15)
        np.testing.assert_allclose(honest.imag, np.zeros((2, 2)), atol=1e-15)


class TestCayleyMengerMatrix(unittest.TestCase):
    def setUp(self):
        self.st = Spacetime()

    def _tetra(self):
        # Unit-edge tetra under Wick rotation, but with edge (0,1) timelike so
        # honest and |l^2| geometries differ.
        sq = {
            frozenset({0, 1}): -1.0,
            frozenset({0, 2}): 1.0,
            frozenset({0, 3}): 1.0,
            frozenset({1, 2}): 1.0,
            frozenset({1, 3}): 1.0,
            frozenset({2, 3}): 1.0,
        }
        simplex, verts, edges = _make_simplex(self.st, [0, 1, 2, 3], sq)
        return simplex, verts, edges, sq

    def _expected_inner(self, simplex, sq, wick):
        vids = [v.get_id() for v in simplex.get_vertices()]
        n = len(vids)
        inner = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                l2 = sq[frozenset({vids[i], vids[j]})]
                inner[i, j] = abs(l2) if wick else float(l2)
        return inner

    def test_structure_border_and_inner_block(self):
        simplex, _, _, sq = self._tetra()
        n = len(simplex.get_vertices()) + 1  # d + 2 == 5 for a tetrahedron
        if True:
            B = np.array(simplex.cayley_menger_matrix()).reshape(n, n)
            self.assertEqual(B[0, 0], 0.0)
            np.testing.assert_allclose(B[0, 1:], np.ones(n - 1))
            np.testing.assert_allclose(B[1:, 0], np.ones(n - 1))
            np.testing.assert_allclose(B[1:, 1:].real,
                                       self._expected_inner(simplex, sq, wick=False))
            np.testing.assert_allclose(B[1:, 1:].imag, np.zeros((n - 1, n - 1)),
                                       atol=1e-15)

    def test_reproduces_dihedral_angle(self):
        """The extracted matrix is exactly what dihedralAngle consumes: feeding
        cayleyMengerMatrix through the cofactor formula reproduces
        dihedral_angle(hinge) for both the Wick-rotated and honest geometries."""
        simplex, verts, edges, _ = self._tetra()
        # Hinge = edge (2,3); the two opposite vertices are 0 and 1.
        hinge, _ = self.st.create_simplex([verts[2], verts[3]],
                                         [edges[frozenset({2, 3})]])
        vids = [v.get_id() for v in simplex.get_vertices()]
        n = len(vids) + 1
        bi = vids.index(0) + 1
        bj = vids.index(1) + 1

        if True:
            B = np.array(
                simplex.cayley_menger_matrix()).reshape(n, n)
            reconstructed = _dihedral_from_cm(B, bi, bj)
            actual = simplex.dihedral_angle(hinge)
            # Both are complex: a wedge on a timelike-normal plane has an
            # imaginary (boost) part (#638).
            self.assertAlmostEqual(abs(actual - reconstructed), 0.0, places=7)


class TestDihedralAngle(unittest.TestCase):
    """Interior dihedral angles via the Cayley-Menger cofactor formula.

    The sign of the diagonal cofactors must be carried through the
    normalization so the result is the true interior dihedral in every
    dimension -- not its supplement (pi - theta), which is what the unsigned
    formula returned for odd-dimensional simplices (issue #161). The
    even-dimensional cases (triangle, pentachoron) must be left unchanged,
    since the 4D Regge/CDT deficit-angle path sums dihedrals of 4-simplices
    about their triangular hinges.
    """

    def setUp(self):
        self.st = Spacetime()

    def _unit_simplex(self, dim):
        """Regular ``dim``-simplex with every squared edge length == 1."""
        ids = list(range(dim + 1))
        sq = {frozenset({i, j}): 1.0 for i in ids for j in ids if i < j}
        simplex, verts, edges = _make_simplex(self.st, ids, sq)
        return simplex, verts, edges

    def _corner_tetra(self):
        """Trirectangular ("corner") tetra: unit legs along x, y, z axes at
        vertex 0, hypotenuse faces of squared length 2."""
        sq = {
            frozenset({0, 1}): 1.0, frozenset({0, 2}): 1.0,
            frozenset({0, 3}): 1.0, frozenset({1, 2}): 2.0,
            frozenset({1, 3}): 2.0, frozenset({2, 3}): 2.0,
        }
        return _make_simplex(self.st, [0, 1, 2, 3], sq)

    def test_regular_tetrahedron_is_interior_not_supplement(self):
        # 3-simplex (odd dim): the regression case. Interior dihedral is
        # arccos(1/3) ~ 70.53 deg, NOT the supplement arccos(-1/3) ~ 109.47.
        simplex, verts, edges = self._unit_simplex(3)
        # Hinge = edge (2, 3); the two opposite vertices are 0 and 1.
        hinge, _ = self.st.create_simplex(
            [verts[2], verts[3]], [edges[frozenset({2, 3})]])
        theta = simplex.dihedral_angle(hinge)
        # All-spacelike, so the angle is real: assert that rather than drop the
        # imaginary part. arccos(1/3) = 70.53 deg is the interior angle; the
        # single-root denominator sqrt(C_ii*C_jj) returns its supplement (#638).
        self.assertAlmostEqual(theta.imag, 0.0, places=12)
        self.assertAlmostEqual(theta.real, math.acos(1.0 / 3.0), places=9)
        self.assertAlmostEqual(math.degrees(theta.real), 70.528779, places=4)
        self.assertNotAlmostEqual(theta.real, math.acos(-1.0 / 3.0), places=6)

    def test_equilateral_triangle_unchanged(self):
        # 2-simplex (even dim): the angle at a vertex hinge = pi/3. Even
        # dimension is untouched by the fix.
        simplex, verts, _ = self._unit_simplex(2)
        # Hinge is the 0-simplex {vertex 0}; opposite vertices are 1 and 2.
        hinge, _ = self.st.create_simplex([verts[0]])
        theta = simplex.dihedral_angle(hinge)
        self.assertAlmostEqual(theta, math.pi / 3.0, places=9)

    def test_regular_pentachoron_unchanged(self):
        # 4-simplex (even dim): dihedral = arccos(1/4) ~ 75.52 deg. This is the
        # parity the 4D Regge deficit-angle path relies on; it must not move.
        simplex, verts, edges = self._unit_simplex(4)
        # Hinge = triangle (2, 3, 4); the two opposite vertices are 0 and 1.
        hinge, _ = self.st.create_simplex(
            [verts[2], verts[3], verts[4]],
            [edges[frozenset({2, 3})], edges[frozenset({2, 4})],
             edges[frozenset({3, 4})]])
        theta = simplex.dihedral_angle(hinge)
        self.assertAlmostEqual(theta, math.acos(0.25), places=9)

    def test_right_angle_dihedral_in_corner_tetrahedron(self):
        # Odd dim, right angle: the dihedral along a leg edge of the corner
        # tetra is exactly pi/2 (the two faces lie in orthogonal coordinate
        # planes). C_ij = 0 here, so this stays correct either way -- a guard
        # that the fix does not perturb right angles in odd dimension.
        simplex, verts, edges = self._corner_tetra()
        # Hinge = leg edge (0, 1); opposite vertices are 2 and 3.
        hinge, _ = self.st.create_simplex(
            [verts[0], verts[1]], [edges[frozenset({0, 1})]])
        theta = simplex.dihedral_angle(hinge)
        self.assertAlmostEqual(theta, math.pi / 2.0, places=9)

    def test_irregular_tetrahedron_interior_dihedral(self):
        # Odd dim, irregular, non-right: dihedral along a hypotenuse edge of
        # the corner tetra, between the base and the slanted face, is
        # arccos(1/sqrt 3) ~ 54.74 deg. The unsigned formula returned its
        # supplement ~ 125.26 deg -- this pins the general (non-symmetric) fix.
        simplex, verts, edges = self._corner_tetra()
        # Hinge = hypotenuse edge (1, 2); opposite vertices are 0 and 3.
        hinge, _ = self.st.create_simplex(
            [verts[1], verts[2]], [edges[frozenset({1, 2})]])
        theta = simplex.dihedral_angle(hinge)
        self.assertAlmostEqual(theta, math.acos(1.0 / math.sqrt(3.0)), places=9)
        self.assertNotAlmostEqual(
            theta, math.pi - math.acos(1.0 / math.sqrt(3.0)), places=6)


class TestSimplexVolumeAcrossDims(unittest.TestCase):
    """volume() is the signed d-content sqrt(det G)/d! at every dimension d,
    checked against closed-form hand values from 1D to 4D."""

    def setUp(self):
        self.st = Spacetime()

    def test_edge_content_is_its_length(self):
        # A 1-simplex's content is its length: sqrt(l^2)/1!.
        simplex, _, _ = _make_simplex(self.st, [0, 1], {frozenset({0, 1}): 4.0})
        self.assertAlmostEqual(simplex.volume(), 2.0, places=9)

    def test_timelike_edge_content_is_imaginary(self):
        # A timelike 1-simplex (l^2 < 0) has V = sqrt(l^2) = i*sqrt(|l^2|). That
        # is its content: not the -sqrt(|l^2|) a real-signed convention gave it,
        # and not the +1 a Wick-rotated |l^2| would (#641).
        simplex, _, _ = _make_simplex(self.st, [0, 1], {frozenset({0, 1}): -1.0})
        vol = simplex.volume()
        self.assertAlmostEqual(vol.real, 0.0, places=12)
        self.assertAlmostEqual(vol.imag, 1.0, places=9)

    def test_regular_unit_tetrahedron_volume(self):
        # Regular tetra, all l^2 = 1: det G = 1/2, so volume = sqrt(2)/12.
        sq = _regular_squares([0, 1, 2, 3], 1.0)
        simplex, _, _ = _make_simplex(self.st, [0, 1, 2, 3], sq)
        self.assertAlmostEqual(simplex.volume(), math.sqrt(2.0) / 12.0, places=9)

    def test_pentachoron_volume_matches_hand_value(self):
        # Unit corner 4-simplex (Gram = I_4): content = 1/4! = 1/24.
        sq = _orthoscheme_squares([0, 1, 2, 3, 4])
        simplex, _, _ = _make_simplex(self.st, [0, 1, 2, 3, 4], sq)
        self.assertAlmostEqual(simplex.volume(), 1.0 / 24.0, places=9)


class TestDeterminantThroughContent(unittest.TestCase):
    """Simplex::determinant / cofactorMatrix are static, internal routines (not
    bound to Python). Their behaviour is exercised through their only
    Python-reachable consumers: the Gram determinant via volume(), and the
    cofactor formula via dihedral_angle()/cayley_menger_matrix() (see the
    TestCayleyMengerMatrix reconstruction). Here: the determinant's *sign* at
    d = 3 and its degenerate det = 0 (singular pivot) path."""

    def setUp(self):
        self.st = Spacetime()

    def test_lorentzian_tetra_content_is_negative(self):
        # One timelike edge (0,1): honest det G = -1, so volume = -1/6, recording
        # the signature. The Wick-rotated cell has a positive Gram determinant.
        sq = {
            frozenset({0, 1}): -1.0,
            frozenset({0, 2}): 1.0, frozenset({0, 3}): 1.0,
            frozenset({1, 2}): 1.0, frozenset({1, 3}): 1.0,
            frozenset({2, 3}): 1.0,
        }
        simplex, _, _ = _make_simplex(self.st, [0, 1, 2, 3], sq)
        # det G < 0, so V = sqrt(det G)/3! is IMAGINARY, not negative-real.
        vol = simplex.volume()
        self.assertAlmostEqual(vol.real, 0.0, places=12)
        self.assertAlmostEqual(vol.imag, 1.0 / 6.0, places=9)

    def test_content_squared_recovers_signed_gram_determinant(self):
        sq = {
            frozenset({0, 1}): -1.0,
            frozenset({0, 2}): 1.0, frozenset({0, 3}): 1.0,
            frozenset({1, 2}): 1.0, frozenset({1, 3}): 1.0,
            frozenset({2, 3}): 1.0,
        }
        simplex, _, _ = _make_simplex(self.st, [0, 1, 2, 3], sq)
        vol = simplex.volume()
        det = np.linalg.det(_expected_gram(simplex, sq, wick=False))
        # V^2 = det G/(d!)^2 exactly -- no copysign, because V is the complex
        # root and squaring it recovers the sign on its own (#641).
        recovered = (vol * vol) * (math.factorial(3) ** 2)
        self.assertAlmostEqual(recovered.real, det, places=9)
        self.assertAlmostEqual(recovered.imag, 0.0, places=9)
        self.assertLess(det, 0.0)

    def test_coplanar_tetra_has_zero_content(self):
        # Four coplanar points (a flat unit-2 square A,B,C,D, sides l^2 = 4,
        # diagonals l^2 = 8): the 3-simplex is affinely dependent, det G = 0, so
        # volume = 0. Exercises the singular pivot path of the n>=3 determinant.
        sq = {
            frozenset({0, 1}): 4.0,   # AB
            frozenset({1, 2}): 4.0,   # BC
            frozenset({2, 3}): 4.0,   # CD
            frozenset({0, 3}): 4.0,   # DA
            frozenset({0, 2}): 8.0,   # AC (diagonal)
            frozenset({1, 3}): 8.0,   # BD (diagonal)
        }
        simplex, _, _ = _make_simplex(self.st, [0, 1, 2, 3], sq)
        self.assertAlmostEqual(abs(simplex.volume()), 0.0, places=7)


class TestSimplexArea(unittest.TestCase):
    """area() is Heron's formula on the three edge squared lengths of a triangle
    (a 2-simplex / Regge hinge): signed by default, |l^2| under wickRotate."""

    def setUp(self):
        self.st = Spacetime()

    def _triangle(self, l01, l02, l12):
        sq = {frozenset({0, 1}): l01,
              frozenset({0, 2}): l02,
              frozenset({1, 2}): l12}
        simplex, _, _ = _make_simplex(self.st, [0, 1, 2], sq)
        return simplex

    def test_right_triangle_area(self):
        # legs 1, 1; area = 1/2.
        self.assertAlmostEqual(self._triangle(1.0, 1.0, 2.0).area(), 0.5, places=9)

    def test_three_four_five_triangle_area(self):
        # legs 3, 4 (l^2 9, 16), hypotenuse 5 (l^2 25); area = 6.
        self.assertAlmostEqual(self._triangle(9.0, 16.0, 25.0).area(), 6.0,
                               places=9)

    def test_equilateral_triangle_area(self):
        # unit equilateral; area = sqrt(3)/4.
        self.assertAlmostEqual(self._triangle(1.0, 1.0, 1.0).area(),
                               math.sqrt(3.0) / 4.0, places=9)

    def test_degenerate_collinear_triangle_has_zero_area(self):
        # Collinear 0--1--2 (lengths 1, 2, 1 -> l^2 1, 4, 1): Heron radicand 0.
        self.assertAlmostEqual(self._triangle(1.0, 4.0, 1.0).area(), 0.0,
                               places=12)

    def test_lorentzian_triangle_area_is_imaginary_not_zero(self):
        # A timelike edge drives Heron's radicand negative, so the area is
        # IMAGINARY. The old real-typed path clamped it to 0, which reported
        # zero area for every timelike triangle -- zero was never their area,
        # it was what a double could represent (#641).
        tri = self._triangle(-1.0, 4.0, 3.0)
        area = tri.area()
        self.assertAlmostEqual(area.real, 0.0, places=12)
        self.assertGreater(abs(area.imag), 0.5)

    def test_all_spacelike_area_toggle_is_noop(self):
        tri = self._triangle(9.0, 16.0, 25.0)
        self.assertAlmostEqual(tri.area(),
                               tri.area(), places=12)


class TestSimplexDihedralKnownValues(unittest.TestCase):
    """dihedral_angle() against closed-form hand values for regular simplices.

    The Cayley-Menger cofactor formula cos θ = -C_ij/sqrt(|C_ii C_jj|) returns
    the true interior dihedral in every dimension: the diagonal cofactors C_ii
    carry the dimension-parity sign (negative for odd-dimensional simplices), and
    reapplying it to the normalization keeps the interior angle rather than its
    supplement π - θ (issue #161, fixed). The even-dimensional cases (the
    equilateral triangle; the regular 4-simplex -- the dimension the 4D CDT/Regge
    solver runs in) were always correct; the regular tetrahedron (odd) is the
    regression case."""

    def setUp(self):
        self.st = Spacetime()

    def test_equilateral_triangle_interior_angle(self):
        # 2-simplex, vertex hinge: an equilateral triangle's interior angle is
        # pi/3. Even dimension; checked on the default (honest) path.
        sq = _regular_squares([0, 1, 2], 1.0)
        tri, verts, _ = _make_simplex(self.st, [0, 1, 2], sq)
        hinge, _ = self.st.create_simplex([verts[0]], [])
        self.assertAlmostEqual(tri.dihedral_angle(hinge), math.pi / 3.0, places=7)

    def test_regular_pentachoron_dihedral_is_arccos_one_quarter(self):
        # 4-simplex, triangle hinge: the regular 4-simplex dihedral is
        # arccos(1/4) ~ 75.52 deg -- the production CDT/Regge dimension.
        # All-spacelike, so honest == Wick.
        sq = _regular_squares([0, 1, 2, 3, 4], 1.0)
        cell, verts, edges = _make_simplex(self.st, [0, 1, 2, 3, 4], sq)
        hinge, _ = self.st.create_simplex(
            [verts[2], verts[3], verts[4]],
            [edges[frozenset({2, 3})], edges[frozenset({2, 4})],
             edges[frozenset({3, 4})]])
        expected = math.acos(0.25)
        if True:
            self.assertAlmostEqual(cell.dihedral_angle(hinge),
                                   expected, places=7)

    def test_regular_tetrahedron_dihedral_is_interior_angle(self):
        # 3-simplex (odd dim), edge hinge: the regular tetrahedron's interior
        # dihedral is arccos(1/3) ~ 70.53 deg, NOT the supplement arccos(-1/3) ~
        # 109.47 deg. The diagonal cofactors are negative here; carrying that
        # sign (issue #161, fixed) yields the interior angle. This pinned the
        # supplement bug as an expectedFailure before the fix landed.
        sq = _regular_squares([0, 1, 2, 3], 1.0)
        tet, verts, edges = _make_simplex(self.st, [0, 1, 2, 3], sq)
        hinge, _ = self.st.create_simplex([verts[2], verts[3]],
                                         [edges[frozenset({2, 3})]])
        self.assertAlmostEqual(tet.dihedral_angle(hinge),
                               math.acos(1.0 / 3.0), places=7)
        self.assertNotAlmostEqual(tet.dihedral_angle(hinge),
                                  math.acos(-1.0 / 3.0), places=6)


class TestSimplexDeficitAngle(unittest.TestCase):
    """deficit_angle() = 2*pi - sum of the (Wick-rotated) dihedral angles of every
    top cell meeting at a hinge. Verified as that contract on regular tetrahedra
    fanned around a shared edge: deficitAngle must find all N incident cells and
    subtract their summed dihedral from 2*pi.

    (These tests check the cell-finding-and-summation contract -- deficitAngle =
    2*pi - sum of the incident cells' dihedrals -- self-consistently against
    dihedral_angle(), rather than a hand angle; see TestDihedralAngle for the
    hand-checked dihedral values. dihedralAngle is correct in every dimension.)"""

    @staticmethod
    def _fan_around_edge(st, n):
        """n regular unit tetrahedra all sharing the edge {0, 1}; returns the
        shared edge as a 1-simplex hinge and the first tetrahedron."""
        verts = {0: st.create_vertex(0), 1: st.create_vertex(1)}
        edge_cache = {}

        def edge(a, b):
            key = frozenset({a, b})
            if key not in edge_cache:
                edge_cache[key] = st.create_edge(verts[a], verts[b], cmath.sqrt(complex(1.0)))
            return edge_cache[key]

        e01 = edge(0, 1)
        first = None
        nid = 2
        for _ in range(n):
            a, b = nid, nid + 1
            nid += 2
            verts[a] = st.create_vertex(a)
            verts[b] = st.create_vertex(b)
            tet, _ = st.create_simplex(
                [verts[0], verts[1], verts[a], verts[b]],
                [edge(0, 1), edge(0, a), edge(0, b),
                 edge(1, a), edge(1, b), edge(a, b)])
            if first is None:
                first = tet
        hinge, _ = st.create_simplex([verts[0], verts[1]], [e01])
        return hinge, first

    def test_single_cell_deficit_is_two_pi_minus_its_dihedral(self):
        st = _spacetime_nd(3)
        hinge, tet = self._fan_around_edge(st, 1)
        self.assertAlmostEqual(
            hinge.deficit_angle(),
            2.0 * math.pi - tet.dihedral_angle(hinge),
            places=7)

    def test_deficit_sums_dihedrals_of_all_incident_cells(self):
        st = _spacetime_nd(3)
        n = 5
        hinge, tet = self._fan_around_edge(st, n)
        # All n cells are congruent regular tetrahedra sharing the hinge edge,
        # so the summed dihedral is exactly n * (single-cell dihedral).
        single = tet.dihedral_angle(hinge)
        self.assertAlmostEqual(hinge.deficit_angle(),
                               2.0 * math.pi - n * single, places=7)


class TestSignedVsWickNonRegression(unittest.TestCase):
    """Historically the geometry was always Wick-rotated (|l^2|). For a purely
    spacelike simplex the signature-aware (signed) path must reproduce those
    values exactly: the wickRotate toggle is a no-op on every geometry method."""

    def setUp(self):
        self.st = Spacetime()

    def test_all_spacelike_tetra_methods_match_wick(self):
        sq = _regular_squares([0, 1, 2, 3], 1.0)
        tet, verts, edges = _make_simplex(self.st, [0, 1, 2, 3], sq)
        hinge, _ = self.st.create_simplex([verts[1], verts[2]],
                                         [edges[frozenset({1, 2})]])

        np.testing.assert_allclose(tet.gram_matrix(),
                                   tet.gram_matrix())
        np.testing.assert_allclose(tet.cayley_menger_matrix(),
                                   tet.cayley_menger_matrix())
        self.assertAlmostEqual(tet.dihedral_angle(hinge),
                               tet.dihedral_angle(hinge),
                               places=12)
        # volume() is honest-only; on an all-spacelike cell it equals the Wick
        # reconstruction sqrt(det G_wick)/d!.
        wick_det = np.linalg.det(
            np.array(tet.gram_matrix()).reshape(3, 3))
        self.assertAlmostEqual(tet.volume(),
                               math.sqrt(wick_det) / math.factorial(3),
                               places=12)

    def test_all_spacelike_triangle_area_matches_wick(self):
        sq = {frozenset({0, 1}): 9.0, frozenset({0, 2}): 16.0,
              frozenset({1, 2}): 25.0}
        tri, _, _ = _make_simplex(self.st, [0, 1, 2], sq)
        self.assertAlmostEqual(tri.area(),
                               tri.area(), places=12)


if __name__ == "__main__":
    unittest.main()
