# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Shared construction for the periodic-grid tests: the covariant degree-zero
pencil of a Kuhn-triangulated three-torus at a crystal momentum."""
import numpy as np

import tessera
from tessera import chainhodge as ch
from tessera import cobordism as cob


def cubic_grid(n, a=1.0):
    return tessera.PeriodicKuhnGrid.cubic(n, a)


def covariant_torus(grid, kappa=(0.0, 0.0, 0.0), crossover=512, measure_certificate=False):
    """(K, base ChainHodge, CovariantChainHodge) of the grid with the flat
    connection of the crystal momentum `kappa` (reciprocal coordinates)."""
    K = cob.ChainComplex.fromTopCells(grid.cells())
    edges = K.kSimplexVertices(1)
    base = ch.ChainHodge(K, grid.squaredLengths(edges), ch.Preset.L2, ch.Branch.Continuation,
                         crossover)
    U = ch.Connection(K, grid.blochLinks(edges, list(kappa)))
    return K, base, ch.CovariantChainHodge(base, U, 7, measure_certificate)


def free_levels(grid, kappa, count):
    """The lowest `count` values of |k + G|^2 on the cubic cell of the grid,
    with k = 2 pi kappa / a and G = 2 pi n / a, sorted ascending."""
    a = np.sqrt(grid.latticeGram()[0][0])
    span = range(-4, 5)
    levels = sorted(
        (2 * np.pi / a) ** 2 * ((kappa[0] + i) ** 2 + (kappa[1] + j) ** 2 + (kappa[2] + l) ** 2)
        for i in span for j in span for l in span)
    return np.array(levels[:count])
