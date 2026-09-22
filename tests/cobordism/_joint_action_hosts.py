# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""Shared complexes for the holomorphic joint action and its two solves.

Three hosts, each the smallest complex that carries the structure the suite
using it needs.

``sphere3`` is the boundary of the 4-simplex read as a closed 3-complex: five
vertices, ten edges, ten triangles and five tetrahedra. It is closed, so the
Regge action has every hinge it needs, and it is small enough that a Newton
solve over both edge fields is a twenty-variable system.

``kuhn_ball`` is the Kuhn (staircase) triangulation of a cube, which is a
triangulated ball. Its interior vertices carry the lowest modes a self-trapping
computation fills, and the number of divisions per axis fixes how many there
are: ``divisions = 2`` has one interior vertex, ``3`` has eight, and so on.

``tetrahedron`` is a single tetrahedron, the complex the whitepaper quotes its
connection-stiffness numbers on.

Every builder sets both edge fields explicitly. The squared length is written
through ``setLength(sqrt(z))``, because the mesh stores the length and not its
square; the connection is written through ``setPhase``, the stored coordinate of
the link ``U = exp(i phi)``.
"""

import cmath
import itertools

import tessera as T


def _apply_geometry(spacetime, squared, phase):
    """Write a squared length and a phase onto every edge by index.

    ``squared`` and ``phase`` are callables of the edge index, so a caller can
    give a host a deterministic non-uniform metric and a deterministic
    connection without touching the mesh itself.
    """
    for index, edge in enumerate(spacetime.getEdgeList().toVector()):
        edge.setLength(cmath.sqrt(complex(squared(index))))
        edge.setPhase(complex(phase(index)))
    return spacetime


def sphere3(squared=lambda index: 1.0, phase=lambda index: 0.0):
    """The boundary of the 4-simplex as a closed 3-complex."""
    cells = [list(cell) for cell in itertools.combinations(range(5), 4)]
    spacetime = T.Spacetime.fromVertexTuples(3, cells, 1.0, 0.0)
    return _apply_geometry(spacetime, squared, phase)


def tetrahedron(squared=lambda index: 1.0, phase=lambda index: 0.0):
    """A single tetrahedron: four vertices, six edges, four triangles."""
    spacetime = T.Spacetime.fromVertexTuples(3, [[0, 1, 2, 3]], 1.0, 0.0)
    return _apply_geometry(spacetime, squared, phase)


def kuhn_cells(divisions):
    """The Kuhn triangulation of a cube of ``divisions`` cells per axis.

    Each grid cube is cut into six tetrahedra along its main diagonal, one per
    permutation of the three axes: the tetrahedron on ``p0 = (i, j, l)``,
    ``p1 = p0 + e_pi(1)``, ``p2 = p1 + e_pi(2)``, ``p3 = p2 + e_pi(3)``.
    Adjacent cubes cut their shared face along the same diagonal, so the result
    is a consistent complex and a triangulated ball.

    Reference: Kuhn, "Some combinatorial lemmas in topology", IBM Journal of
    Research and Development 4, 518 (1960).
    """
    side = divisions + 1

    def identifier(i, j, l):
        return (i * side + j) * side + l

    cells = []
    for i in range(divisions):
        for j in range(divisions):
            for l in range(divisions):
                for permutation in itertools.permutations(range(3)):
                    corner = [i, j, l]
                    cell = [identifier(*corner)]
                    for axis in permutation:
                        corner[axis] += 1
                        cell.append(identifier(*corner))
                    cells.append(sorted(cell))
    return cells


def kuhn_interior_vertices(divisions):
    """The vertex ids of ``kuhn_cells`` that touch no face of the cube."""
    side = divisions + 1
    interior = []
    for i in range(1, divisions):
        for j in range(1, divisions):
            for l in range(1, divisions):
                interior.append((i * side + j) * side + l)
    return interior


def kuhn_ball(divisions=2, squared=lambda index: 1.0,
              phase=lambda index: 0.0):
    """A Kuhn-triangulated cube, which is a triangulated ball."""
    spacetime = T.Spacetime.fromVertexTuples(3, kuhn_cells(divisions), 1.0, 0.0)
    return _apply_geometry(spacetime, squared, phase)
