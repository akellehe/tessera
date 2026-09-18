# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""A 3-complex carrying a genuine degree-2 register.

Higher-degree gradient work needs holes whose periods are NOT trivially zero,
and that is harder to arrange than it looks. Two constructions that seem
plausible both fail:

* a complex with `b_k = 0` — every target is then the constant full leak, so
  the functional is constant and its gradient is correctly but uselessly zero;
* holes taken as cells of a CLOSED manifold — a cell's boundary is exact, so
  its period vanishes (measured: a period matrix of `~1e-16`), the
  least-squares fit is degenerate, and the value jumps discontinuously under
  perturbation.

What works is removing cells so the surviving cycle does not bound. `S³` minus
two VERTEX-DISJOINT balls is `S² × I`, whose middle `S²` is a non-bounding
2-cycle: `b₂ = 1`, with boundary, and two emergent holes. Disjointness is what
needs the subdivision — the five-tetrahedron `S³` has no two tetrahedra that
avoid each other, and removing two that meet leaves a ball with `b₂ = 0`.
"""
import itertools

import tessera

cobordism = tessera.cobordism


class B2Register:
    """The fixture and the pieces it is built from."""

    @staticmethod
    def stellarSubdivision(cells, cell, apex):
        """Replace `cell` with the cone from `apex` over each of its facets."""
        subdivided = [c for c in cells if c != cell]
        for dropped in range(len(cell)):
            subdivided.append(
                sorted(cell[:dropped] + cell[dropped + 1:] + [apex]))
        return subdivided

    @staticmethod
    def subdividedThreeSphere(rounds=2):
        """`S³` as the boundary of a 4-simplex, stellar-subdivided `rounds`
        times. Two rounds give 80 tetrahedra, which is the smallest that
        reliably contains a vertex-disjoint pair while staying quick to
        eigendecompose."""
        cells = [sorted(c) for c in itertools.combinations(range(5), 4)]
        apex = 5
        for _round in range(rounds):
            for cell in list(cells):
                if cell in cells:
                    cells = B2Register.stellarSubdivision(cells, cell, apex)
                    apex += 1
        return cells

    @staticmethod
    def disjointPair(cells):
        """The first pair of cells sharing no vertex, or None."""
        for left, right in itertools.combinations(range(len(cells)), 2):
            if not set(cells[left]) & set(cells[right]):
                return left, right
        return None

    @staticmethod
    def build(rounds=2):
        """`(spacetime, holes)` — a 3-complex with `b₂ = 1` and its emergent
        degree-2 holes."""
        cells = B2Register.subdividedThreeSphere(rounds)
        pair = B2Register.disjointPair(cells)
        if pair is None:
            raise RuntimeError(
                "no vertex-disjoint pair of tetrahedra after "
                f"{rounds} subdivision rounds; removing two that meet leaves a "
                "ball, whose b_2 is 0")
        kept = [cell for index, cell in enumerate(cells) if index not in pair]
        spacetime = tessera.Spacetime.fromVertexTuples(3, kept, 1.0, 0.0)
        holes = cobordism.MultiCobordism.emergent_holes(spacetime, 2)
        return spacetime, holes
