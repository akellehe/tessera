# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The periodic Kuhn grid (#1157): the staircase triangulation of the
three-torus, its flat metric from the lattice Gram matrix, and the flat U(1)
connection of a crystal momentum with its holonomy certificate."""
import cmath
import collections

import numpy as np
import pytest

import tessera
from tessera import chainhodge as ch
from tessera import cobordism as cob

FCC = [[0.5, 0.25, 0.25], [0.25, 0.5, 0.25], [0.25, 0.25, 0.5]]  # primitive cell, a = 1


class TestCombinatorics:
    def test_counts_degree_and_homology(self):
        grid = tessera.PeriodicKuhnGrid.cubic(3)
        cells = grid.cells()
        K = cob.ChainComplex.fromTopCells(cells)
        assert grid.vertexCount() == 27
        assert list(K.fVector()) == [27, 7 * 27, 12 * 27, 6 * 27]
        degree = collections.Counter(v for e in K.kSimplexVertices(1) for v in e)
        assert set(degree.values()) == {14}
        assert K.boundaryComposesToZero()
        assert list(K.bettiNumbers()) == [1, 3, 3, 1]
        ok, reason = cob.ChainComplex.dualComplexIsValid(cells, 3)
        assert ok, reason

    def test_anisotropic_divisions(self):
        grid = tessera.PeriodicKuhnGrid(3, 4, 5, np.eye(3))
        K = cob.ChainComplex.fromTopCells(grid.cells())
        assert list(K.fVector()) == [60, 420, 720, 360]
        assert K.boundaryComposesToZero()
        ok, reason = cob.ChainComplex.dualComplexIsValid(grid.cells(), 3)
        assert ok, reason

    def test_vertex_indexing_round_trips_and_wraps(self):
        grid = tessera.PeriodicKuhnGrid(3, 4, 5, np.eye(3))
        for vid in range(grid.vertexCount()):
            i, j, l = grid.gridIndex(vid)
            assert grid.vertexId(i, j, l) == vid
            assert grid.fractionalCoordinates(vid) == pytest.approx([i / 3, j / 4, l / 5])
        assert grid.vertexId(-1, 4, 5) == grid.vertexId(2, 0, 0)
        with pytest.raises(IndexError):
            grid.gridIndex(grid.vertexCount())

    def test_refusals(self):
        with pytest.raises(ValueError, match="at least 3"):
            tessera.PeriodicKuhnGrid(2, 3, 3, np.eye(3))
        with pytest.raises(ValueError, match="symmetric"):
            tessera.PeriodicKuhnGrid(3, 3, 3, [[1, 0.5, 0], [0, 1, 0], [0, 0, 1]])
        with pytest.raises(ValueError, match="positive definite"):
            tessera.PeriodicKuhnGrid(3, 3, 3, [[1, 1, 0], [1, 1, 0], [0, 0, 1]])


class TestGeometry:
    def test_squared_lengths_are_the_step_metric_on_the_displacement(self):
        n = (3, 4, 5)
        grid = tessera.PeriodicKuhnGrid(*n, FCC)
        g = np.array(grid.stepMetric())
        assert g == pytest.approx(np.array(FCC) / np.outer(n, n))
        K = cob.ChainComplex.fromTopCells(grid.cells())
        edges = K.kSimplexVertices(1)
        s = grid.squaredLengths(edges)
        seen = set()
        for (x, y), se in zip(edges, s):
            d = np.array(grid.displacement(x, y))
            assert np.array(grid.displacement(y, x)) == pytest.approx(-d)
            assert set(np.abs(d)) <= {0, 1} and (np.all(d >= 0) or np.all(d <= 0))
            seen.add(tuple(np.abs(d)))
            assert se == pytest.approx(d @ g @ d, rel=1e-14)
            assert se.imag == 0.0
        assert len(seen) == 7  # three axis steps, three face diagonals, the body diagonal
        cert = ch.WhitneyMass.certificate(K, s)
        assert cert.allowable and cert.margin == pytest.approx(np.pi)
        # The tetrahedra tile the cell: their volumes sum to sqrt(det A).
        assert sum(cert.volumes) == pytest.approx(np.sqrt(np.linalg.det(FCC)), rel=1e-12)

    def test_a_pair_that_is_not_an_edge_is_refused(self):
        grid = tessera.PeriodicKuhnGrid.cubic(4)
        with pytest.raises(ValueError, match="not joined"):
            grid.displacement(grid.vertexId(0, 0, 0), grid.vertexId(1, -1, 0))  # the other face diagonal
        with pytest.raises(ValueError, match="not joined"):
            grid.displacement(grid.vertexId(0, 0, 0), grid.vertexId(2, 0, 0))
        with pytest.raises(ValueError, match="not joined"):
            grid.displacement(0, 0)

    def test_build_carries_the_lengths_into_a_spacetime(self):
        grid = tessera.PeriodicKuhnGrid(3, 3, 4, FCC)
        sig = tessera.Signature(grid.dimension(), tessera.Lorentzian)
        st = tessera.Spacetime(tessera.Metric(True, sig), tessera.CDT, 1.0, 1.0,
                               tessera.PREFERRED, grid)
        st.build()
        K = ch.WhitneyMass.complexOf(st)
        assert list(K.fVector()) == [36, 252, 432, 216]
        direct = grid.squaredLengths(K.kSimplexVertices(1))
        assert ch.WhitneyMass.squaredLengthsOf(st, K) == pytest.approx(direct, rel=1e-13)


class TestBlochConnection:
    KAPPA = [0.3, -0.45, 0.125]

    def test_flat_with_the_momentum_as_holonomy(self):
        grid = tessera.PeriodicKuhnGrid(3, 4, 5, FCC)
        K = cob.ChainComplex.fromTopCells(grid.cells())
        U = ch.Connection(K, grid.blochLinks(K.kSimplexVertices(1), self.KAPPA))
        assert U.isUnitary()
        for p, q, r in K.kSimplexVertices(2):
            assert abs(U.curvature(p, q, r) - 1.0) < 1e-12
        for axis in range(3):
            for base in (0, 17):
                walk = grid.fundamentalCycle(axis, base)
                assert len(walk) == grid.divisions()[axis]
                assert walk[0][0] == base and walk[-1][1] == base
                expected = cmath.exp(2j * cmath.pi * self.KAPPA[axis])
                assert abs(U.holonomy(walk) - expected) < 1e-12

    def test_phase_is_antisymmetric_and_a_reciprocal_vector_is_a_gauge(self):
        grid = tessera.PeriodicKuhnGrid.cubic(4)
        K = cob.ChainComplex.fromTopCells(grid.cells())
        edges = K.kSimplexVertices(1)
        for x, y in edges[:20]:
            assert grid.blochPhase(x, y, self.KAPPA) == pytest.approx(-grid.blochPhase(y, x, self.KAPPA))
        # kappa in Z^3 is a reciprocal lattice vector: trivial holonomy on every cycle.
        U = ch.Connection(K, grid.blochLinks(edges, [1.0, -2.0, 3.0]))
        for axis in range(3):
            assert abs(U.holonomy(grid.fundamentalCycle(axis)) - 1.0) < 1e-12
        with pytest.raises(ValueError):
            grid.fundamentalCycle(3)
