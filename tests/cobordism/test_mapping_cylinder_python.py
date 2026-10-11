# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""#1191 — the mapping-cylinder tick W^l that carries one level into the next.

Section 3 of the whitepaper fixes the dimension and time conventions once. A
level is a spatial complex; the fibering direction is the passage of a certified
cluster at level l to a response vertex at level l + 1; that direction is time;
and one tick is realized as the interaction cobordism W^l, the mapping cylinder
of the reduction map from K^l onto the vertex set of R^{l+1}, with the cells of
K^{l+1} attached on its outgoing end. Four claims carry that here.

THE EXTRA DIMENSION IS THE FIBERING DIRECTION AND NOTHING ELSE. The cylinder over
a three-dimensional level is four-dimensional, and it is four-dimensional only
because the prism over each of its top cells is. Time is not a simplex dimension
of any level.

THE CYLINDER IS THE STAIRCASE TRIANGULATION OF THE PRISM. Over an incoming top
cell with images r_i, the top cells of W^l are the simplices [v_0..v_i,
r_i..r_d] whose vertices are distinct. A reduction map that is injective on a
cell gives d + 1 of them, and one that collapses the cell to a single response
vertex gives exactly one, the cone, which is the mapping cylinder of a constant
map.

A DEGENERATE STAIRCASE SIMPLEX KEEPS ITS CELL. One whose response vertices repeat
is read with the repeats removed, and where no other cell of the cylinder
contains that form it is a maximal cell below the top dimension, a collapsed
cell. A triangle carried to (a, b, b) has the top cell [v_0, v_1, v_2, b] and the
collapsed cell [v_0, a, b], which alone carries the fiber edge (v_0, a) and the
image edge [a, b]. A collapsed cell lies below the top dimension, so it carries
no facet of the cylinder and leaves the boundary checks as they were.

ITS BOUNDARY IS THE DISJOINT UNION OF THE TWO ENDS. On a closed level every
facet of W^l that is not at an end is shared by two of its top cells, so the free
facets are the incoming complex and the image of the reduction, and the second is
checked to lie inside the declared K^{l+1}. On a level with boundary the cylinder
over that boundary is a side wall, and the class reports it as one rather than
calling the boundary a disjoint union anyway.

THE FIBER EDGES ARE THE ONES THE PROTOCOL NAMES. They are (v, r(v)), one per
vertex of K^l, and they are the only timelike edges of the history. The prism
diagonals that join a vertex to a response vertex other than its own are reported
separately, under their own name.
"""

import itertools
import unittest

import tessera as T

cob = T.cobordism

#: The boundary of the 4-simplex read as a closed 3-complex: five vertices, five
#: tetrahedra, and every triangle carried by exactly two of them.
SPHERE3 = [list(cell) for cell in itertools.combinations(range(5), 4)]

#: A single tetrahedron, which is a ball and therefore has a boundary for the
#: cylinder to carry a side wall over.
TETRAHEDRON = [[0, 1, 2, 3]]


def _declaration(incoming, reduction, outgoing=()):
    declaration = cob.MappingCylinderDeclaration()
    declaration.incoming_top_cells = [list(cell) for cell in incoming]
    declaration.reduction_map = dict(reduction)
    declaration.outgoing_top_cells = [list(cell) for cell in outgoing]
    return declaration


def _shift(vertices, offset=100):
    """The reduction map that renames every vertex, which is injective."""
    return {vertex: vertex + offset for vertex in vertices}


def _cells(cells):
    return sorted(tuple(cell) for cell in cells)


def _a(vertex):
    """Response vertex a, b, c, ... is 100, 101, 102, ..."""
    return 100 + "abcdefgh".index(vertex)


def _r(images):
    """The reduction map carrying vertex i to the i-th named response vertex."""
    return {vertex: _a(image) for vertex, image in enumerate(images)}


def _cluster_lineage_maximal_cells(incoming, reduction):
    """The maximal cells of ClusterLineage's cylinder over the same input,
    with no outgoing cell declared and its outgoing vertices renamed to the
    response identifiers. Every response vertex must be an image."""
    obs = T.observables
    vertices = 1 + max(vertex for cell in incoming for vertex in cell)
    responses = sorted(set(reduction.values()))
    W = obs.ClusterLineage.mapping_cylinder(
        obs.LevelComplex([list(cell) for cell in incoming], vertices),
        [responses.index(reduction[vertex]) for vertex in range(vertices)],
        obs.LevelComplex([], len(responses)))
    cells = {frozenset(cell) for cell in W.cells}
    maximal = [cell for cell in cells if not any(cell < other for other in cells)]
    return sorted(
        tuple(sorted(vertex if vertex < vertices
                     else responses[vertex - vertices] for vertex in cell))
        for cell in maximal)


class TheCylinderIsTheStaircaseTriangulationTest(unittest.TestCase):
    """The prism over each incoming top cell, degenerate simplices aside."""

    def test_an_injective_map_gives_one_simplex_per_staircase_step(self):
        """A tetrahedron carried to four distinct response vertices.

        The prism over a 3-simplex has four 4-simplices, one for each place the
        staircase can step from the bottom to the top.
        """
        cylinder = cob.MappingCylinder(
            _declaration(TETRAHEDRON, _shift(range(4))))
        read = cylinder.read()
        self.assertEqual(read.incoming_dimension, 3)
        self.assertEqual(read.cylinder_dimension, 4)
        self.assertEqual(len(read.cylinder_top_cells), 4)
        for cell in read.cylinder_top_cells:
            self.assertEqual(len(cell), 5)
            self.assertEqual(len(set(cell)), 5)
        self.assertEqual(
            sorted(tuple(cell) for cell in read.cylinder_top_cells),
            [(0, 1, 2, 3, 103), (0, 1, 2, 102, 103),
             (0, 1, 101, 102, 103), (0, 100, 101, 102, 103)])

    def test_a_collapsing_map_gives_the_cone(self):
        """Every vertex to one response vertex: the mapping cylinder of a
        constant map is the cone, and the staircase produces exactly it."""
        cylinder = cob.MappingCylinder(
            _declaration(TETRAHEDRON, {vertex: 100 for vertex in range(4)},
                         [[100]]))
        read = cylinder.read()
        self.assertEqual(len(read.cylinder_top_cells), 1)
        self.assertEqual(list(read.cylinder_top_cells[0]), [0, 1, 2, 3, 100])
        self.assertEqual([list(cell) for cell in read.image_top_cells], [[100]])
        self.assertEqual(read.outgoing_dimension, 0)

    def test_the_cylinder_over_a_closed_level_is_a_valid_chain_complex(self):
        """It is the complex of the cylinder, and it carries the level's
        homotopy type.

        The mapping cylinder of an isomorphism deformation-retracts onto the
        incoming complex, so the cylinder over the 3-sphere has the Betti
        numbers of the 3-sphere and not those of a 4-sphere or a point.
        """
        cylinder = cob.MappingCylinder(
            _declaration(SPHERE3, _shift(range(5)),
                         [[vertex + 100 for vertex in cell] for cell in SPHERE3]))
        read = cylinder.read()
        complex_ = cob.ChainComplex.from_top_cells(
            [list(cell) for cell in read.cylinder_top_cells])
        self.assertTrue(complex_.boundary_composes_to_zero())
        self.assertEqual(complex_.dimension(), 4)
        self.assertEqual(list(complex_.betti_numbers()), [1, 0, 0, 1, 0])


class ADegenerateStaircaseSimplexKeepsItsCellTest(unittest.TestCase):
    """A reduction map neither injective nor constant on a cell leaves cells
    below the top dimension, and the cylinder keeps them."""

    def _read(self, incoming, images, outgoing=()):
        return cob.MappingCylinder(
            _declaration(incoming, _r(images), outgoing)).read()

    def _assert_every_fiber_edge_lies_on_a_cell(self, read):
        cells = [set(cell) for cell in read.cylinder_top_cells] + \
            [set(cell) for cell in read.cylinder_collapsed_cells]
        for vertex, image in read.fiber_edges:
            self.assertTrue(any({vertex, image} <= cell for cell in cells),
                            (vertex, image))

    def test_a_triangle_carried_to_a_b_b_keeps_the_cell_through_v0_a(self):
        """[v0, a, b, b] is degenerate and its form [v0, a, b] lies in no
        other cell; without it the fiber edge (v0, a), the response vertex a
        and the image edge [a, b] are not in the cylinder at all."""
        read = self._read([[0, 1, 2]], "abb")
        a, b = _a("a"), _a("b")
        self.assertEqual(_cells(read.cylinder_top_cells), [(0, 1, 2, b)])
        self.assertEqual(_cells(read.cylinder_collapsed_cells), [(0, a, b)])
        self._assert_every_fiber_edge_lies_on_a_cell(read)
        self.assertIn((0, a), {tuple(edge) for edge in read.cross_edges})
        complex_ = cob.ChainComplex.from_cells(
            [list(cell) for cell in read.cylinder_top_cells] +
            [list(cell) for cell in read.cylinder_collapsed_cells])
        self.assertEqual(complex_.num_simplices(0),
                         len(read.incoming_vertices) +
                         len(read.response_vertices))
        self.assertTrue(complex_.boundary_composes_to_zero())
        # The mapping cylinder retracts onto the image edge, a point up to
        # homotopy.
        self.assertEqual(list(complex_.betti_numbers()), [1, 0, 0, 0])

    def test_a_tetrahedron_with_a_two_to_one_collapse(self):
        """(a, b, c, c): the forms of steps 0 and 1 each keep a response
        vertex no later step reaches, so both are collapsed cells; the form
        of step 2 lies in the top cell of step 3."""
        read = self._read(TETRAHEDRON, "abcc")
        a, b, c = _a("a"), _a("b"), _a("c")
        self.assertEqual(_cells(read.cylinder_top_cells), [(0, 1, 2, 3, c)])
        self.assertEqual(_cells(read.cylinder_collapsed_cells),
                         [(0, 1, b, c), (0, a, b, c)])
        for cell in read.cylinder_collapsed_cells:
            self.assertEqual(len(set(cell)), len(cell))
            self.assertLess(len(cell), read.cylinder_dimension + 1)
        self._assert_every_fiber_edge_lies_on_a_cell(read)

    def test_a_tetrahedron_with_a_three_to_one_collapse(self):
        """(a, b, b, b): one collapsed cell, two dimensions below the top."""
        read = self._read(TETRAHEDRON, "abbb")
        a, b = _a("a"), _a("b")
        self.assertEqual(_cells(read.cylinder_top_cells), [(0, 1, 2, 3, b)])
        self.assertEqual(_cells(read.cylinder_collapsed_cells), [(0, a, b)])
        self._assert_every_fiber_edge_lies_on_a_cell(read)

    def test_a_collapse_onto_the_lowest_vertices_leaves_no_collapsed_cell(self):
        """(a, a, b, c) and (a, a, a, b): every degenerate form lies in the
        next step's top cell."""
        for images in ("aabc", "aaab"):
            read = self._read(TETRAHEDRON, images)
            self.assertEqual(list(read.cylinder_collapsed_cells), [], images)
            self._assert_every_fiber_edge_lies_on_a_cell(read)

    def test_the_constant_map_and_an_injective_map_are_unchanged(self):
        """The cone is the one top cell of the constant map and carries every
        degenerate form; an injective map has no degenerate form."""
        cone = self._read(TETRAHEDRON, "aaaa", [[_a("a")]])
        self.assertEqual(_cells(cone.cylinder_top_cells),
                         [(0, 1, 2, 3, _a("a"))])
        self.assertEqual(list(cone.cylinder_collapsed_cells), [])
        prism = self._read(TETRAHEDRON, "abcd")
        self.assertEqual(
            _cells(prism.cylinder_top_cells),
            [(0, 1, 2, 3, 103), (0, 1, 2, 102, 103),
             (0, 1, 101, 102, 103), (0, 100, 101, 102, 103)])
        self.assertEqual(list(prism.cylinder_collapsed_cells), [])

    def test_a_form_a_neighbouring_cell_carries_is_not_reported(self):
        """Two tetrahedra share the triangle (v0, v1, v2), whose edge
        (v1, v2) collapses onto b. The first, carried to (a, b, b, b), leaves
        the form [v0, a, b]; the second, carried to (a, b, b, d), leaves the
        collapsed cell [v0, a, b, d], which contains it. Two triangles where
        (v0, v1, v2) goes to (a, b, b) and (v0, v2, v3) to (a, b, c) leave
        none: the second prism's top cell [v0, a, b, c] contains [v0, a, b]."""
        a, b, c, d = _a("a"), _a("b"), _a("c"), _a("d")
        pair = self._read([[0, 1, 2, 3], [0, 1, 2, 4]], "abbbd")
        self.assertEqual(_cells(pair.cylinder_top_cells),
                         [(0, 1, 2, 3, b), (0, 1, 2, 4, d), (0, 1, 2, b, d)])
        self.assertEqual(_cells(pair.cylinder_collapsed_cells),
                         [(0, a, b, d)])
        self._assert_every_fiber_edge_lies_on_a_cell(pair)
        triangles = self._read([[0, 1, 2], [0, 2, 3]], "abbc")
        self.assertIn((0, a, b, c), _cells(triangles.cylinder_top_cells))
        self.assertEqual(list(triangles.cylinder_collapsed_cells), [])
        self._assert_every_fiber_edge_lies_on_a_cell(triangles)

    def test_the_maximal_cells_agree_with_cluster_lineage(self):
        """ClusterLineage builds the same staircase and keeps every form; the
        maximal cells of its cylinder are the top and collapsed cells here."""
        cases = [
            ([[0, 1, 2]], "abb"),
            (TETRAHEDRON, "abcc"),
            (TETRAHEDRON, "abbc"),
            (TETRAHEDRON, "abbb"),
            (TETRAHEDRON, "aaaa"),
            (TETRAHEDRON, "abcd"),
            ([[0, 1, 2, 3], [0, 1, 2, 4]], "abbbd"),
            ([[0, 1, 2], [1, 2, 3]], "abbc"),
            ([[0, 1, 2], [0, 2, 3]], "abbc"),
            ([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], "abbc"),
            ([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], "abbb"),
        ]
        for incoming, images in cases:
            read = self._read(incoming, images)
            self.assertEqual(
                sorted(_cells(read.cylinder_top_cells) +
                       _cells(read.cylinder_collapsed_cells)),
                _cluster_lineage_maximal_cells(incoming, _r(images)),
                (incoming, images))

    def test_the_boundary_checks_read_the_top_cells_alone(self):
        """The closed sphere carried to (a, b, b, b): the top cells are the
        cone over the sphere with apex b, and the collapsed cell [v0, a, b]
        alone carries (v0, a). The sphere has no boundary, so there is no side
        wall, and the free facets are the sphere's triangles. Over the
        triangle carried to (a, b, b) the free facets are the top cell's."""
        a, b = _a("a"), _a("b")
        sphere = [list(cell) for cell in itertools.combinations(range(4), 3)]
        read = self._read(sphere, "abbb", [[a, b]])
        self.assertEqual(
            _cells(read.cylinder_top_cells),
            [(0, 1, 2, b), (0, 1, 3, b), (0, 2, 3, b), (1, 2, 3, b)])
        self.assertEqual(_cells(read.cylinder_collapsed_cells), [(0, a, b)])
        self._assert_every_fiber_edge_lies_on_a_cell(read)
        self.assertEqual(_cells(read.incoming_free_facets), _cells(sphere))
        self.assertEqual(list(read.outgoing_free_facets), [])
        self.assertEqual(list(read.side_free_facets), [])
        self.assertTrue(read.incoming_boundary_is_the_incoming_complex)
        self.assertTrue(read.outgoing_complex_contains_the_image)
        self.assertTrue(read.has_no_side_wall)
        self.assertTrue(read.boundary_is_the_disjoint_union)
        self.assertEqual(read.boundary_residual, 0.0)
        triangle = self._read([[0, 1, 2]], "abb", [[a, b]])
        self.assertEqual(_cells(triangle.incoming_free_facets), [(0, 1, 2)])
        self.assertEqual(list(triangle.outgoing_free_facets), [])
        self.assertEqual(_cells(triangle.side_free_facets),
                         [(0, 1, b), (0, 2, b), (1, 2, b)])


class TheBoundaryIsTheDisjointUnionOfTheTwoEndsTest(unittest.TestCase):
    """dW^l = K^l disjoint union K^{l+1}, and what happens when it is not."""

    def test_a_closed_level_has_no_side_wall(self):
        """Every interior facet of the cylinder is shared by two top cells."""
        outgoing = [[vertex + 100 for vertex in cell] for cell in SPHERE3]
        cylinder = cob.MappingCylinder(
            _declaration(SPHERE3, _shift(range(5)), outgoing))
        read = cylinder.read()
        self.assertTrue(read.incoming_boundary_is_the_incoming_complex)
        self.assertTrue(read.outgoing_complex_contains_the_image)
        self.assertTrue(read.has_no_side_wall)
        self.assertTrue(read.boundary_is_the_disjoint_union)
        self.assertEqual(read.boundary_residual, 0.0)
        self.assertTrue(read.certificate.holds())
        self.assertEqual(
            sorted(tuple(sorted(cell)) for cell in read.incoming_free_facets),
            sorted(tuple(sorted(cell)) for cell in SPHERE3))
        self.assertEqual(
            sorted(tuple(sorted(cell)) for cell in read.outgoing_free_facets),
            sorted(tuple(sorted(cell)) for cell in outgoing))

    def test_a_level_with_boundary_carries_a_side_wall(self):
        """A tetrahedron is a ball, so the cylinder over its boundary is a wall
        and the report says so rather than calling the boundary a disjoint
        union anyway."""
        cylinder = cob.MappingCylinder(
            _declaration(TETRAHEDRON, _shift(range(4)),
                         [[100, 101, 102, 103]]))
        read = cylinder.read()
        self.assertTrue(read.incoming_boundary_is_the_incoming_complex)
        self.assertTrue(read.outgoing_complex_contains_the_image)
        self.assertFalse(read.has_no_side_wall)
        self.assertFalse(read.boundary_is_the_disjoint_union)
        self.assertGreater(read.boundary_residual, 0.0)

    def test_the_side_wall_is_the_cylinder_over_the_boundary(self):
        """A closed level has no side facets; over a single tetrahedron, whose
        boundary is four triangles, every free facet of the cylinder that lies
        on neither end is a side facet, and each side facet meets both ends."""
        outgoing = [[vertex + 100 for vertex in cell] for cell in SPHERE3]
        closed = cob.MappingCylinder(
            _declaration(SPHERE3, _shift(range(5)), outgoing)).read()
        self.assertEqual(list(closed.side_free_facets), [])
        ball = cob.MappingCylinder(
            _declaration(TETRAHEDRON, _shift(range(4)),
                         [[100, 101, 102, 103]])).read()
        self.assertGreater(len(ball.side_free_facets), 0)
        for facet in ball.side_free_facets:
            self.assertTrue(any(v < 100 for v in facet))
            self.assertTrue(any(v >= 100 for v in facet))

    def test_an_image_the_outgoing_complex_does_not_carry_is_reported(self):
        """The interactions attach the cells of K^{l+1}; when they have not
        attached the image, the cylinder's outgoing end is not inside the next
        level and the report says so."""
        cylinder = cob.MappingCylinder(
            _declaration(SPHERE3, _shift(range(5))))
        read = cylinder.read()
        self.assertFalse(read.outgoing_complex_contains_the_image)
        self.assertFalse(read.boundary_is_the_disjoint_union)
        self.assertFalse(read.certificate.holds())


class TheFiberEdgesAreTheOnlyTimelikeEdgesTest(unittest.TestCase):
    """(v, r(v)) per vertex, with the prism diagonals named separately."""

    def test_there_is_one_fiber_edge_per_incoming_vertex(self):
        cylinder = cob.MappingCylinder(
            _declaration(SPHERE3, _shift(range(5)),
                         [[vertex + 100 for vertex in cell] for cell in SPHERE3]))
        read = cylinder.read()
        self.assertEqual(list(read.incoming_vertices), [0, 1, 2, 3, 4])
        self.assertEqual(list(read.response_vertices),
                         [100, 101, 102, 103, 104])
        self.assertEqual(sorted(tuple(edge) for edge in read.fiber_edges),
                         [(vertex, vertex + 100) for vertex in range(5)])

    def test_the_fiber_edges_are_among_the_cross_edges(self):
        """The cross edges are every edge with one endpoint at each end, so
        they contain the fiber edges and the prism diagonals besides."""
        cylinder = cob.MappingCylinder(
            _declaration(SPHERE3, _shift(range(5))))
        read = cylinder.read()
        cross = {tuple(edge) for edge in read.cross_edges}
        for edge in read.fiber_edges:
            self.assertIn(tuple(edge), cross)
        self.assertGreater(len(cross), len(read.fiber_edges))

    def test_a_collapsing_map_has_no_prism_diagonal(self):
        """When the whole cell goes to one response vertex there is only one
        response vertex for a diagonal to reach."""
        cylinder = cob.MappingCylinder(
            _declaration(TETRAHEDRON, {vertex: 100 for vertex in range(4)},
                         [[100]]))
        read = cylinder.read()
        self.assertEqual(sorted(tuple(edge) for edge in read.cross_edges),
                         sorted(tuple(edge) for edge in read.fiber_edges))


class TheDeclarationIsCheckedTest(unittest.TestCase):
    """Every malformed tick is refused by name."""

    def test_an_empty_incoming_complex_is_refused(self):
        with self.assertRaises(ValueError):
            cob.MappingCylinder(_declaration([], {}))

    def test_an_impure_incoming_complex_is_refused(self):
        with self.assertRaises(ValueError):
            cob.MappingCylinder(
                _declaration([[0, 1, 2, 3], [0, 1, 2]], _shift(range(4))))

    def test_a_vertex_with_no_image_is_refused(self):
        with self.assertRaises(ValueError):
            cob.MappingCylinder(
                _declaration(TETRAHEDRON, {0: 100, 1: 101, 2: 102}))

    def test_a_response_vertex_that_is_also_an_incoming_vertex_is_refused(self):
        with self.assertRaises(ValueError):
            cob.MappingCylinder(
                _declaration(TETRAHEDRON, {vertex: 0 for vertex in range(4)}))

    def test_an_outgoing_cell_on_a_vertex_that_is_no_response_vertex(self):
        with self.assertRaises(ValueError):
            cob.MappingCylinder(
                _declaration(TETRAHEDRON, _shift(range(4)), [[100, 101, 999]]))


if __name__ == "__main__":
    unittest.main()
