# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""Cobordism fixture triangulations (#63), built via Topology subclasses.

These topologies build exact, minimal, pre-geometric (coordinate-free)
triangulations. We verify each built complex against its known invariants
across a wide range of dimensions:

  * f-vector matches the closed form (C(n+2,k+1) for S^n, C(n+1,k+1) for Δ^n),
    checked both combinatorially from the top simplices and against tessera's
    own materialized face set;
  * Euler characteristic and CombinatorialDimension;
  * structural manifold properties — closed pseudomanifold (every codim-1 face
    in exactly two top simplices) for spheres, and ∂Δ^n ≅ S^{n-1} for balls.
"""

import itertools
import math
import unittest

import tessera

cobordism = tessera.cobordism

# How far to push the dimension sweeps. Well beyond the {1,2,3,4} the cobordism
# spec needs (4-manifolds use 5-vertex simplices; 5-dim cobordisms use 6-vertex),
# the construction being otherwise dimension-agnostic.
#
# Upper bound: tessera's Fingerprint stores at most kMax = 8 vertex IDs
# (mesh/Fingerprint.h), so a simplex may have at most 8 vertices — dimension 7.
# Beyond that the fingerprint truncates and createSimplex silently fails to
# register the simplex (see GH issue on the fingerprint limit). S^7 (8-vertex
# top simplices) is the largest sphere that round-trips; SWEEP_MAX reflects that.
SWEEP_MAX = 7
STRUCT_MAX = 7


def _build(topology):
    sig = tessera.Signature(topology.dimension(), tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    st = tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0,
                           tessera.PREFERRED, topology)
    st.build()  # delegates to topology.build(); numSimplices ignored
    return st


def _top_tuples(st):
    """Vertex-id tuples of the registered top simplices (facets are lazy, so
    immediately after build the registered simplices are exactly the tops)."""
    by_size = {}
    for s in st.get_simplices():
        t = tuple(sorted(v.get_id() for v in s.get_vertices()))
        by_size.setdefault(len(t), []).append(t)
    top_card = max(by_size)
    return by_size[top_card]


def _f_vector_from_tops(tops):
    maxsz = max(len(t) for t in tops)
    f = []
    for card in range(1, maxsz + 1):
        faces = set()
        for t in tops:
            faces.update(itertools.combinations(t, card))
        f.append(len(faces))
    return f


def _euler(fvec):
    return sum((-1) ** k * n for k, n in enumerate(fvec))


def _materialized_face_counts(st):
    """Count tessera's own k-simplices once all facets are materialized.
    materialize_facets() forces lazy facet materialization to a fixpoint."""
    st.materialize_facets()
    counts = {}
    for s in st.get_simplices():
        k = len(s.get_vertices()) - 1
        counts[k] = counts.get(k, 0) + 1
    return [counts.get(k, 0) for k in range(max(counts) + 1)] if counts else []


class TestSimplexBoundarySphere(unittest.TestCase):
    """S^n = ∂Δ^{n+1} across a wide range of n."""

    def test_f_vector_and_euler_sweep(self):
        for n in range(1, SWEEP_MAX + 1):
            with self.subTest(n=n):
                st = _build(tessera.SimplexBoundarySphere(n))
                tops = _top_tuples(st)
                expected = [math.comb(n + 2, k + 1) for k in range(n + 1)]
                # exactly n+2 top n-simplices, each on n+1 vertices
                self.assertEqual(len(tops), n + 2)
                self.assertTrue(all(len(t) == n + 1 for t in tops))
                self.assertEqual(_f_vector_from_tops(tops), expected)
                # χ(S^n) = 1 + (-1)^n
                self.assertEqual(_euler(expected), 1 + (-1) ** n)
                self.assertEqual(
                    cobordism.CombinatorialDimension().compute(st), float(n))

    def test_materialized_faces_match_closed_form(self):
        # Cross-check tessera's own face generation against C(n+2,k+1).
        for n in range(1, STRUCT_MAX + 1):
            with self.subTest(n=n):
                st = _build(tessera.SimplexBoundarySphere(n))
                expected = [math.comb(n + 2, k + 1) for k in range(n + 1)]
                self.assertEqual(_materialized_face_counts(st), expected)

    def test_closed_pseudomanifold_sweep(self):
        # Every codim-1 (i.e. (n-1)-) face lies in exactly two top n-simplices.
        for n in range(1, SWEEP_MAX + 1):
            with self.subTest(n=n):
                tops = _top_tuples(_build(tessera.SimplexBoundarySphere(n)))
                facet_count = {}
                for t in tops:
                    for f in itertools.combinations(t, n):  # drop one vertex
                        facet_count[f] = facet_count.get(f, 0) + 1
                self.assertTrue(all(c == 2 for c in facet_count.values()),
                                f"S^{n} not a closed pseudomanifold")

    def test_cofaces_via_tessera_small_n(self):
        # Exercise tessera's coface bookkeeping on these hand-built complexes.
        for n in range(1, STRUCT_MAX + 1):
            with self.subTest(n=n):
                st = _build(tessera.SimplexBoundarySphere(n))
                st.materialize_facets()  # materialize facets/cofaces
                checked = 0
                for s in st.get_simplices():
                    if len(s.get_vertices()) == n + 1:  # a top n-simplex
                        for facet in s.get_facets():
                            self.assertEqual(len(facet.get_cofaces()), 2)
                            checked += 1
                self.assertGreater(checked, 0)


class TestSolidSimplex(unittest.TestCase):
    """Δ^n (closed n-ball) across a wide range of n."""

    def test_f_vector_and_euler_sweep(self):
        for n in range(1, SWEEP_MAX + 1):
            with self.subTest(n=n):
                st = _build(tessera.SolidSimplex(n))
                tops = _top_tuples(st)
                expected = [math.comb(n + 1, k + 1) for k in range(n + 1)]
                self.assertEqual(len(tops), 1)
                self.assertEqual(_f_vector_from_tops(tops), expected)
                self.assertEqual(_euler(expected), 1)  # contractible
                self.assertEqual(
                    cobordism.CombinatorialDimension().compute(st), float(n))

    def test_materialized_faces_match_closed_form(self):
        for n in range(1, STRUCT_MAX + 1):
            with self.subTest(n=n):
                st = _build(tessera.SolidSimplex(n))
                expected = [math.comb(n + 1, k + 1) for k in range(n + 1)]
                self.assertEqual(_materialized_face_counts(st), expected)

    def test_boundary_is_sphere_sweep(self):
        # ∂Δ^n = its n+1 facets (codim-1 faces in exactly one top simplex),
        # which form S^{n-1} = ∂Δ^n: the boundary facets equal the top simplices
        # of SimplexBoundarySphere(n-1).
        for n in range(2, SWEEP_MAX + 1):
            with self.subTest(n=n):
                top = _top_tuples(_build(tessera.SolidSimplex(n)))[0]
                boundary = {f for f in itertools.combinations(top, n)}
                self.assertEqual(len(boundary), n + 1)
                sphere_tops = {tuple(sorted(t)) for t in
                               _top_tuples(_build(tessera.SimplexBoundarySphere(n - 1)))}
                # relabel boundary facets to 0..n-1 vertex set of the sphere
                # (both are "all n-subsets of n+1 vertices"): compare as the
                # same combinatorial sphere.
                self.assertEqual(len(boundary), len(sphere_tops))


class TestRealProjectivePlane(unittest.TestCase):

    def test_invariants(self):
        st = _build(tessera.RealProjectivePlane())
        tops = _top_tuples(st)
        self.assertEqual(_f_vector_from_tops(tops), [6, 15, 10])
        self.assertEqual(_materialized_face_counts(st), [6, 15, 10])
        self.assertEqual(_euler([6, 15, 10]), 1)
        self.assertEqual(cobordism.CombinatorialDimension().compute(st), 2.0)

    def test_closed_pseudomanifold(self):
        tops = _top_tuples(_build(tessera.RealProjectivePlane()))
        edge_count = {}
        for t in tops:
            for e in itertools.combinations(t, 2):
                edge_count[e] = edge_count.get(e, 0) + 1
        self.assertEqual(len(edge_count), 15)
        self.assertTrue(all(c == 2 for c in edge_count.values()))


class TestComplexProjectivePlane(unittest.TestCase):
    """Kühnel's minimal 9-vertex CP^2: f=(9,36,84,90,36), χ=3,
    Betti (1,0,1,0,1), orientable with a rank-1 definite intersection form
    (|signature| = 1)."""

    F_VECTOR = [9, 36, 84, 90, 36]

    def test_f_vector_and_euler(self):
        st = _build(tessera.ComplexProjectivePlane())
        tops = _top_tuples(st)
        self.assertEqual(len(tops), 36)
        self.assertEqual(_f_vector_from_tops(tops), self.F_VECTOR)
        self.assertEqual(_materialized_face_counts(st), self.F_VECTOR)
        self.assertEqual(_euler(self.F_VECTOR), 3)
        self.assertEqual(cobordism.CombinatorialDimension().compute(st), 4.0)

    def test_closed_pseudomanifold(self):
        # Every codimension-one face (a tetrahedron, 4 vertices) lies in exactly
        # two of the 36 four-simplices — the hallmark of a closed manifold.
        tops = _top_tuples(_build(tessera.ComplexProjectivePlane()))
        tet_count = {}
        for t in tops:
            for tet in itertools.combinations(t, 4):
                tet_count[tet] = tet_count.get(tet, 0) + 1
        self.assertEqual(len(tet_count), 90)
        self.assertTrue(all(c == 2 for c in tet_count.values()))

    def test_homology_and_signature(self):
        st = _build(tessera.ComplexProjectivePlane())
        cc = cobordism.ChainComplex.from_spacetime(st)
        self.assertEqual(cc.f_vector(), self.F_VECTOR)
        self.assertTrue(cc.boundary_composes_to_zero())
        # Betti numbers of CP^2 agree over Q and GF(2) (the homology is
        # torsion-free), so there is no 2-torsion to split them apart.
        self.assertEqual(cc.betti_numbers(), [1, 0, 1, 0, 1])
        self.assertEqual(cc.betti_numbers_gf2(), [1, 0, 1, 0, 1])
        self.assertEqual(cc.euler_characteristic(), 3)
        self.assertEqual(cc.torsion(2), [])
        self.assertEqual(cc.torsion(3), [])
        # H^2 is rank one with a definite, unimodular intersection form: the
        # 1x1 matrix [±1]. |signature| = 1 is the orientation-independent fact
        # (the sign is a convention fixed by the choice of fundamental class).
        form = cc.intersection_form()
        self.assertEqual(len(form), 1)
        self.assertEqual(abs(form[0]), 1.0)
        self.assertEqual(abs(cc.signature()), 1)


class TestPreGeometric(unittest.TestCase):

    def test_vertices_are_coordinate_free(self):
        for topology in (tessera.SimplexBoundarySphere(3),
                         tessera.SolidSimplex(4),
                         tessera.RealProjectivePlane(),
                         tessera.ComplexProjectivePlane()):
            with self.subTest(topology=type(topology).__name__):
                st = _build(topology)
                for v in st.get_vertex_list().to_vector():
                    with self.assertRaises(Exception):
                        v.get_coordinates()


if __name__ == "__main__":
    unittest.main()
