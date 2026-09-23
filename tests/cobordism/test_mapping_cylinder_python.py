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
r_i..r_d], with the degenerate ones dropped. A reduction map that is injective on
a cell gives d + 1 of them, and one that collapses the cell to a single response
vertex gives exactly one, the cone, which is the mapping cylinder of a constant
map.

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


class TheCylinderIsTheStaircaseTriangulationTest(unittest.TestCase):
    """The prism over each incoming top cell, degenerate simplices dropped."""

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
        complex_ = cob.ChainComplex.fromTopCells(
            [list(cell) for cell in read.cylinder_top_cells])
        self.assertTrue(complex_.boundaryComposesToZero())
        self.assertEqual(complex_.dimension(), 4)
        self.assertEqual(list(complex_.bettiNumbers()), [1, 0, 0, 1, 0])


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
