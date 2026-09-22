# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The lineage number on the mapping cylinder with a cooriented cut
(:class:`tessera.observables.ClusterLineage`), ticket #1197 and whitepaper
sections 3 and 13.1-13.2.

The whitepaper is the specification these tests hold the code to.

* One tick is the interaction cobordism ``W_l``: the mapping cylinder of the
  reduction map from the level's complex ``K_l`` onto the response vertices of
  the next level, with the cells of ``K_{l+1}`` attached on its outgoing end.
  It contains ``K_l``, the fiber edges joining each vertex to its response
  vertex, and ``K_{l+1}``, with ``dW_l = K_l disjoint-union K_{l+1}``. The
  geometric realization is required: without interior cells a separating cut
  has nothing to separate and a lineage nothing to cross.
* A separating slice ``Sigma`` is a cooriented closed codimension-one
  simplicial cut separating the incoming boundary from the outgoing boundary.
  It is carried here by the 0-cochain that is 0 on the incoming side and 1 on
  the outgoing side, whose coboundary is a cocycle for free, so ``Sigma`` is
  closed without any closedness being imposed.
* A quark lineage is an integral one-chain ``c_Q`` of ``W`` relative to ``dW``,
  and ``N_Q = c_Q . Sigma`` is the simplicial intersection pairing. Reversing
  the lineage sends ``c_Q -> -c_Q`` and ``N_Q -> -N_Q``; no sign is extracted
  from a spectral coordinate, from the connection, from an eigenvalue or from a
  density, and no level set of a real part is used.
* Homologous cuts give the same integer when the lineage has no source in the
  slab between them; a source there is exactly what makes two cuts disagree.
* Pair creation is the boundary of an oriented pair surface and therefore
  creates ``+1`` and ``-1`` together.
* ``N_q(Sigma) = sum_Q n_Q c_Q . Sigma`` and ``B(Sigma) = N_q(Sigma) / 3``,
  where ``n_Q`` is the fermion number carried on the lineage. The factor
  ``1/3`` is an explicit physical calibration, not a topological theorem.

Every quantity below is an exact integer computed by integer arithmetic, so
every assertion is an equality with no tolerance.
"""
import unittest

import numpy as np

import tessera

obs = tessera.observables
cob = tessera.cobordism

TETRAHEDRON = [[0, 1, 2, 3]]


def _level(cells, vertices):
    return obs.LevelComplex(cells, vertices)


def _product_history(levels=3):
    """``K x I x I ...``: `levels` copies of the tetrahedron joined by identity
    reductions. Level ``l``'s own vertex ``v`` is the cobordism's vertex
    ``4 * l + v``."""
    identity = [0, 1, 2, 3]
    return obs.ClusterLineage.history(
        [_level(TETRAHEDRON, 4) for _ in range(levels)],
        [identity for _ in range(levels - 1)],
    )


def _collapsing_history():
    """A reduction that is not injective: the tetrahedron's four vertices
    reduce onto two response vertices, which then reduce onto one. Vertices
    0..3 are level 0, 4 and 5 are level 1, and 6 is level 2."""
    return obs.ClusterLineage.history(
        [_level(TETRAHEDRON, 4), _level([[0, 1]], 2), _level([[0]], 1)],
        [[0, 0, 1, 1], [0, 0]],
    )


def _triangle_index(W, vertices):
    """The canonical ``C_2(W)`` index of the triangle on the given vertices."""
    triangles = W.complex.kSimplexVertices(2)
    key = sorted(vertices)
    for index, triangle in enumerate(triangles):
        if list(triangle) == key:
            return index
    raise AssertionError(f"{key} is not a triangle of the cobordism")


def _coefficients(lineage):
    return np.asarray(lineage.coefficients, dtype=np.int64)


# --------------------------------------------------------------------------- #
# The impure builder the cobordism needs
# --------------------------------------------------------------------------- #
class TestImpureChainComplex(unittest.TestCase):
    """``ChainComplex.fromCells`` accepts cells of mixed dimensions, which is
    what a cobordism with cells attached on one end requires: the prisms over
    the incoming level's cells sit beside the outgoing level's own cells, and
    the latter are one dimension lower."""

    def test_mixed_dimensions_are_accepted(self):
        K = cob.ChainComplex.fromCells([[0, 1, 2], [2, 3]])
        self.assertEqual(K.dimension(), 2)
        self.assertEqual(K.numSimplices(0), 4)
        self.assertEqual(K.numSimplices(1), 4)  # 01, 02, 12, 23
        self.assertEqual(K.numSimplices(2), 1)
        self.assertTrue(K.boundaryComposesToZero())

    def test_a_declared_face_of_a_declared_cell_adds_nothing(self):
        both = cob.ChainComplex.fromCells([[0, 1, 2], [0, 1]])
        alone = cob.ChainComplex.fromCells([[0, 1, 2]])
        self.assertEqual(list(both.fVector()), list(alone.fVector()))

    def test_the_pure_builder_still_refuses_an_impure_list(self):
        with self.assertRaises(ValueError):
            cob.ChainComplex.fromTopCells([[0, 1, 2], [2, 3]])

    def test_a_repeated_vertex_is_refused(self):
        with self.assertRaises(ValueError):
            cob.ChainComplex.fromCells([[0, 1, 1]])


# --------------------------------------------------------------------------- #
# The mapping cylinder
# --------------------------------------------------------------------------- #
class TestMappingCylinder(unittest.TestCase):
    """``W`` contains each level, the fiber edges, and nothing that is not a
    prism over a level's cells."""

    def test_the_product_cobordism_has_the_expected_shape(self):
        W = _product_history(3)
        self.assertEqual(W.levels, 3)
        self.assertEqual(W.complex.numSimplices(0), 12)
        # The prism over a 3-simplex is a 4-simplex: time is the fourth
        # simplex dimension of W and of no level.
        self.assertEqual(W.complex.dimension(), 4)
        self.assertTrue(W.complex.boundaryComposesToZero())

    def test_the_levels_are_laid_out_in_order(self):
        W = _product_history(3)
        self.assertEqual(list(W.vertexOffsets), [0, 4, 8, 12])
        self.assertEqual(list(W.levelOf), [0] * 4 + [1] * 4 + [2] * 4)
        self.assertEqual(list(W.incomingVertices()), [0, 1, 2, 3])
        self.assertEqual(list(W.outgoingVertices()), [8, 9, 10, 11])

    def test_every_vertex_carries_a_fiber_edge_to_its_response_vertex(self):
        W = _product_history(3)
        self.assertEqual(list(W.responseOf), [4, 5, 6, 7, 8, 9, 10, 11, 8, 9, 10, 11])
        fiber = W.fiberEdges()
        self.assertEqual(len(fiber), 8)  # four per interaction step
        for w in range(8):
            self.assertIn(W.edgeIndex(w, W.responseOf[w]), fiber)

    def test_each_level_is_a_subcomplex_of_the_cobordism(self):
        W = _product_history(3)
        for level in range(3):
            base = 4 * level
            self.assertIn(
                sorted([base, base + 1, base + 2, base + 3]),
                [list(c) for c in W.complex.kSimplexVertices(3)],
            )

    def test_the_image_of_a_cell_is_a_cell_of_the_outgoing_end(self):
        W = _collapsing_history()
        # The tetrahedron's image under the reduction [0,0,1,1] is the edge
        # (4, 5), which level 1 declares in its own right.
        self.assertGreaterEqual(W.edgeIndex(4, 5), 0)

    def test_a_reduction_that_identifies_vertices_collapses_the_prism(self):
        W = _collapsing_history()
        self.assertEqual(W.complex.numSimplices(0), 7)
        self.assertTrue(W.complex.boundaryComposesToZero())
        # Both vertices that reduce onto response vertex 4 keep their own fiber
        # edge; the collapse is in the prism's dimension, never in a lost edge.
        for w, response in ((0, 4), (1, 4), (2, 5), (3, 5), (4, 6), (5, 6)):
            self.assertGreaterEqual(W.edgeIndex(w, response), 0)

    def test_one_step_is_the_two_level_history(self):
        step = obs.ClusterLineage.mappingCylinder(
            _level(TETRAHEDRON, 4), [0, 1, 2, 3], _level(TETRAHEDRON, 4)
        )
        pair = _product_history(2)
        self.assertEqual(list(step.complex.fVector()), list(pair.complex.fVector()))

    def test_a_malformed_history_is_refused_by_name(self):
        with self.assertRaises(ValueError):
            obs.ClusterLineage.history([_level(TETRAHEDRON, 4)], [])
        with self.assertRaises(ValueError):
            obs.ClusterLineage.history(
                [_level(TETRAHEDRON, 4), _level(TETRAHEDRON, 4)], [[0, 1, 2]]
            )
        with self.assertRaises(ValueError):
            obs.ClusterLineage.history(
                [_level(TETRAHEDRON, 4), _level(TETRAHEDRON, 4)], [[0, 1, 2, 9]]
            )
        with self.assertRaises(ValueError):
            obs.ClusterLineage.history(
                [_level([[0, 1, 2, 9]], 4), _level(TETRAHEDRON, 4)], [[0, 1, 2, 3]]
            )


# --------------------------------------------------------------------------- #
# The cut
# --------------------------------------------------------------------------- #
class TestCoorientedCut(unittest.TestCase):
    """A separating slice, its coorientation, and the refusals."""

    def test_a_level_cut_separates_and_is_cooriented_outward(self):
        W = _product_history(3)
        cut = obs.ClusterLineage.levelCut(W, 0)
        self.assertTrue(cut.separates)
        self.assertEqual(list(cut.side), [0] * 4 + [1] * 8)
        # Every incoming vertex id is below every outgoing one, so the
        # canonical orientation of a crossing edge already runs from the
        # incoming side to the outgoing side.
        self.assertTrue(all(sign == +1 for sign in cut.crossingSigns))
        self.assertEqual(len(cut.crossingEdges), len(set(cut.crossingEdges)))

    def test_every_level_boundary_supplies_a_cut(self):
        W = _product_history(3)
        for after in (0, 1):
            self.assertTrue(obs.ClusterLineage.levelCut(W, after).separates)
        with self.assertRaises(ValueError):
            obs.ClusterLineage.levelCut(W, 2)

    def test_a_cut_may_be_moved_past_an_interior_vertex(self):
        W = _product_history(3)
        side = list(obs.ClusterLineage.levelCut(W, 0).side)
        side[4] = 0  # vertex 4 is interior: it lies on level 1
        moved = obs.ClusterLineage.cutFromSides(W, side)
        self.assertTrue(moved.separates)

    def test_a_cut_that_leaves_a_boundary_on_the_wrong_side_refuses_by_name(self):
        W = _product_history(3)
        allIncoming = obs.ClusterLineage.cutFromSides(W, [0] * 12)
        self.assertFalse(allIncoming.separates)
        self.assertIn(
            "outgoing-boundary-not-on-the-outgoing-side", list(allIncoming.failedCertificates)
        )
        allOutgoing = obs.ClusterLineage.cutFromSides(W, [1] * 12)
        self.assertFalse(allOutgoing.separates)
        self.assertIn(
            "incoming-boundary-not-on-the-incoming-side", list(allOutgoing.failedCertificates)
        )

    def test_a_side_that_is_not_binary_refuses_by_name(self):
        W = _product_history(3)
        side = [0] * 4 + [2] * 4 + [1] * 4
        cut = obs.ClusterLineage.cutFromSides(W, side)
        self.assertFalse(cut.separates)
        self.assertIn("cut-side-not-binary", list(cut.failedCertificates))

    def test_a_side_of_the_wrong_length_is_refused(self):
        W = _product_history(3)
        with self.assertRaises(ValueError):
            obs.ClusterLineage.cutFromSides(W, [0] * 11)


# --------------------------------------------------------------------------- #
# The lineage number
# --------------------------------------------------------------------------- #
class TestLineageNumber(unittest.TestCase):
    """``N_Q = +1`` on a cluster, ``-1`` on its oppositely oriented lineage, and
    the same integer for every cut."""

    def test_a_cluster_crossing_the_cobordism_has_lineage_number_one(self):
        W = _product_history(3)
        cut = obs.ClusterLineage.levelCut(W, 0)
        lineage = obs.ClusterLineage.fromFiberPath(W, 0, 1, "Q")
        read = obs.ClusterLineage.read(W, cut, lineage)
        self.assertEqual(read.number, +1)
        self.assertEqual(read.clusterId, "Q")
        self.assertTrue(read.relativeCycle)
        self.assertTrue(read.cutSeparates)
        self.assertEqual(list(read.interiorSources), [])
        self.assertEqual(list(read.failedCertificates), [])

    def test_the_oppositely_oriented_lineage_has_lineage_number_minus_one(self):
        W = _product_history(3)
        cut = obs.ClusterLineage.levelCut(W, 0)
        lineage = obs.ClusterLineage.fromFiberPath(W, 0)
        antiLineage = obs.ClusterLineage.reversed(lineage)
        self.assertEqual(obs.ClusterLineage.read(W, cut, antiLineage).number, -1)
        np.testing.assert_array_equal(
            _coefficients(antiLineage), -_coefficients(lineage)
        )

    def test_the_lineage_number_does_not_depend_on_the_cut(self):
        W = _product_history(3)
        lineage = obs.ClusterLineage.fromFiberPath(W, 1)
        numbers = []
        for after in (0, 1):
            numbers.append(
                obs.ClusterLineage.intersectionNumber(
                    W, obs.ClusterLineage.levelCut(W, after), lineage
                )
            )
        # and a cut that is not a level cut: one interior vertex moved across
        side = list(obs.ClusterLineage.levelCut(W, 0).side)
        side[5] = 0
        numbers.append(
            obs.ClusterLineage.intersectionNumber(
                W, obs.ClusterLineage.cutFromSides(W, side), lineage
            )
        )
        self.assertEqual(numbers, [+1, +1, +1])

    def test_the_pairing_is_the_sum_over_the_crossing_edges(self):
        W = _product_history(3)
        cut = obs.ClusterLineage.levelCut(W, 1)
        lineage = obs.ClusterLineage.fromFiberPath(W, 2)
        coefficients = _coefficients(lineage)
        byCrossing = sum(
            int(coefficients[edge]) * sign
            for edge, sign in zip(cut.crossingEdges, cut.crossingSigns)
        )
        self.assertEqual(byCrossing, obs.ClusterLineage.intersectionNumber(W, cut, lineage))

    def test_a_lineage_with_a_source_in_the_slab_makes_two_cuts_disagree(self):
        W = _product_history(3)
        stopped = obs.ClusterLineage.fromVertexPath(W, [0, 4], 1, "stopped")
        below = obs.ClusterLineage.read(W, obs.ClusterLineage.levelCut(W, 0), stopped)
        above = obs.ClusterLineage.read(W, obs.ClusterLineage.levelCut(W, 1), stopped)
        self.assertEqual(below.number, +1)
        self.assertEqual(above.number, 0)
        for read in (below, above):
            self.assertFalse(read.relativeCycle)
            self.assertEqual(list(read.interiorSources), [4])
            self.assertIn("lineage-not-a-relative-cycle", list(read.failedCertificates))

    def test_the_boundary_of_a_lineage_is_its_two_endpoints(self):
        W = _product_history(3)
        lineage = obs.ClusterLineage.fromFiberPath(W, 3)
        boundary = np.asarray(obs.ClusterLineage.relativeBoundary(W, lineage), dtype=np.int64)
        expected = np.zeros(12, dtype=np.int64)
        expected[3] = -1
        expected[11] = +1
        np.testing.assert_array_equal(boundary, expected)

    def test_a_reading_against_a_cut_that_does_not_separate_refuses_by_name(self):
        W = _product_history(3)
        cut = obs.ClusterLineage.cutFromSides(W, [0] * 12)
        read = obs.ClusterLineage.read(W, cut, obs.ClusterLineage.fromFiberPath(W, 0))
        self.assertFalse(read.cutSeparates)
        self.assertIn("cut-does-not-separate", list(read.failedCertificates))

    def test_the_lineage_number_is_one_through_a_collapsing_reduction(self):
        W = _collapsing_history()
        for start in (0, 1, 2, 3):
            lineage = obs.ClusterLineage.fromFiberPath(W, start)
            for after in (0, 1):
                cut = obs.ClusterLineage.levelCut(W, after)
                self.assertEqual(obs.ClusterLineage.intersectionNumber(W, cut, lineage), +1)


# --------------------------------------------------------------------------- #
# The lineage of a tracked cluster
# --------------------------------------------------------------------------- #
class TestTrackedLineage(unittest.TestCase):
    """A lineage built from the cluster supports a tracking graph supplies."""

    def test_a_tracked_support_gives_the_fiber_path_of_its_representative(self):
        W = _collapsing_history()
        tracked = obs.ClusterLineage.fromTrackedSupports(W, 0, [[0, 1], [0], [0]], 1, "Q")
        np.testing.assert_array_equal(
            _coefficients(tracked), _coefficients(obs.ClusterLineage.fromFiberPath(W, 0))
        )

    def test_which_representative_is_chosen_does_not_change_the_number(self):
        W = _collapsing_history()
        cut = obs.ClusterLineage.levelCut(W, 0)
        first = obs.ClusterLineage.fromTrackedSupports(W, 0, [[0, 1], [0], [0]])
        second = obs.ClusterLineage.fromTrackedSupports(W, 0, [[2, 3], [1], [0]])
        self.assertNotEqual(list(first.coefficients), list(second.coefficients))
        self.assertEqual(
            obs.ClusterLineage.intersectionNumber(W, cut, first),
            obs.ClusterLineage.intersectionNumber(W, cut, second),
        )

    def test_a_support_the_representative_does_not_reduce_into_is_refused(self):
        W = _collapsing_history()
        with self.assertRaises(ValueError):
            obs.ClusterLineage.fromTrackedSupports(W, 0, [[0, 1], [1], [0]])

    def test_supports_running_past_the_last_level_are_refused(self):
        W = _collapsing_history()
        with self.assertRaises(ValueError):
            obs.ClusterLineage.fromTrackedSupports(W, 1, [[0], [0], [0]])


# --------------------------------------------------------------------------- #
# Pair creation
# --------------------------------------------------------------------------- #
class TestPairCreation(unittest.TestCase):
    """Pair creation is the boundary of an oriented pair surface, so it creates
    ``+1`` and ``-1`` together."""

    def test_the_boundary_of_any_pair_surface_pairs_to_zero(self):
        W = _product_history(3)
        rng = np.random.default_rng(17)
        surface = [int(value) for value in rng.integers(-3, 4, W.complex.numSimplices(2))]
        boundary = obs.ClusterLineage.pairSurfaceBoundary(W, surface)
        for after in (0, 1):
            cut = obs.ClusterLineage.levelCut(W, after)
            self.assertEqual(obs.ClusterLineage.intersectionNumber(W, cut, boundary), 0)

    def test_an_explicit_pair_surface_creates_plus_one_and_minus_one(self):
        W = _product_history(3)
        cut = obs.ClusterLineage.levelCut(W, 0)
        # The prism over level 0's edge (0, 1) is the square with corners 0, 1
        # below and 4, 5 above, triangulated as (0,1,5) and (0,4,5). Its
        # boundary is the loop 0 -> 1 -> 5 -> 4 -> 0.
        surface = [0] * W.complex.numSimplices(2)
        surface[_triangle_index(W, [0, 1, 5])] = +1
        surface[_triangle_index(W, [0, 4, 5])] = -1
        boundary = obs.ClusterLineage.pairSurfaceBoundary(W, surface, 1, "pair")
        self.assertEqual(obs.ClusterLineage.intersectionNumber(W, cut, boundary), 0)

        # The loop is the sum of the four declared edges, and its two crossing
        # pieces are a cluster and an anti-cluster.
        pieces = [
            obs.ClusterLineage.fromVertexPath(W, [1, 5]),
            obs.ClusterLineage.fromVertexPath(W, [0, 1]),
            obs.ClusterLineage.reversed(obs.ClusterLineage.fromVertexPath(W, [4, 5])),
            obs.ClusterLineage.reversed(obs.ClusterLineage.fromVertexPath(W, [0, 4])),
        ]
        total = sum(_coefficients(piece) for piece in pieces)
        np.testing.assert_array_equal(total, _coefficients(boundary))
        numbers = [obs.ClusterLineage.intersectionNumber(W, cut, piece) for piece in pieces]
        self.assertEqual(numbers, [+1, 0, 0, -1])


# --------------------------------------------------------------------------- #
# The totals
# --------------------------------------------------------------------------- #
class TestTotals(unittest.TestCase):
    """``N_q = sum_Q n_Q c_Q . Sigma`` and ``B = N_q / 3``."""

    def test_three_one_sheeted_clusters_give_baryon_number_one(self):
        W = _product_history(3)
        cut = obs.ClusterLineage.levelCut(W, 0)
        lineages = [obs.ClusterLineage.fromFiberPath(W, v, 1, f"q{v}") for v in (0, 1, 2)]
        total = obs.ClusterLineage.totals(W, cut, lineages)
        self.assertEqual(total.fermionNumber, 3)
        self.assertEqual(total.baryonNumber, 1.0)
        self.assertEqual([read.number for read in total.perLineage], [1, 1, 1])
        self.assertEqual(list(total.failedCertificates), [])

    def test_one_three_sheeted_cluster_gives_the_same_baryon_number(self):
        W = _product_history(3)
        cut = obs.ClusterLineage.levelCut(W, 0)
        sheeted = obs.ClusterLineage.fromFiberPath(W, 0, 3, "Q")
        total = obs.ClusterLineage.totals(W, cut, [sheeted])
        self.assertEqual(total.fermionNumber, 3)
        self.assertEqual(total.baryonNumber, 1.0)

    def test_a_cluster_and_an_anti_cluster_cancel(self):
        W = _product_history(3)
        cut = obs.ClusterLineage.levelCut(W, 0)
        lineage = obs.ClusterLineage.fromFiberPath(W, 0, 1, "Q")
        total = obs.ClusterLineage.totals(
            W, cut, [lineage, obs.ClusterLineage.reversed(lineage)]
        )
        self.assertEqual(total.fermionNumber, 0)
        self.assertEqual(total.baryonNumber, 0.0)

    def test_a_failed_per_lineage_certificate_reaches_the_total(self):
        W = _product_history(3)
        cut = obs.ClusterLineage.levelCut(W, 0)
        stopped = obs.ClusterLineage.fromVertexPath(W, [0, 4])
        total = obs.ClusterLineage.totals(W, cut, [stopped])
        self.assertIn("lineage-not-a-relative-cycle", list(total.failedCertificates))


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #
class TestRecords(unittest.TestCase):
    """Every read reports itself as a JSON-able record."""

    def test_the_cobordism_and_the_reads_produce_records(self):
        W = _product_history(3)
        cut = obs.ClusterLineage.levelCut(W, 0)
        lineage = obs.ClusterLineage.fromFiberPath(W, 0, 2, "Q")
        record = W.toRecord()
        self.assertEqual(record["levels"], 3)
        self.assertEqual(record["dimension"], 4)
        self.assertEqual(record["fiber_edges"], 8)
        self.assertTrue(cut.toRecord()["separates"])
        self.assertEqual(lineage.toRecord()["fermion_number"], 2)
        read = obs.ClusterLineage.read(W, cut, lineage).toRecord()
        self.assertEqual(read["number"], 1)
        self.assertEqual(read["cluster_id"], "Q")
        total = obs.ClusterLineage.totals(W, cut, [lineage]).toRecord()
        self.assertEqual(total["fermion_number"], 2)
        self.assertEqual(total["schema_version"], obs.ClusterLineage.kSchemaVersion)


if __name__ == "__main__":
    unittest.main()
