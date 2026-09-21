# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Integrals of products of piecewise-linear basis functions (#1157): the
closed forms on one top simplex, the potential-weighted mass matrix M_0[V], the
vertex density contraction, and the connection-dressed M_0[V]."""
import itertools
import math

import numpy as np
import pytest

import tessera
from tessera import chainhodge as ch
from tessera import cobordism as cob
from tests.chainhodge._fixtures import random_allowable, torus33
from tests.chainhodge._periodic import covariant_torus

FCC = [[0.5, 0.25, 0.25], [0.25, 0.5, 0.25], [0.25, 0.25, 0.5]]


def _grid_instance():
    grid = tessera.PeriodicKuhnGrid(3, 3, 4, FCC)
    K = cob.ChainComplex.fromTopCells(grid.cells())
    return grid, K, grid.squaredLengths(K.kSimplexVertices(1))


class TestOneSimplex:
    def test_closed_forms_on_the_regular_tetrahedron(self):
        K = cob.ChainComplex.fromTopCells([[0, 1, 2, 3]])
        s = [1.0] * 6
        volume = math.sqrt(2) / 12
        integral = lambda vs: ch.WhitneyMass.vertexProductIntegral(K, s, 0, vs)
        assert integral([]) == pytest.approx(volume)
        assert integral([2]) == pytest.approx(volume / 4)
        assert integral([0, 1]) == pytest.approx(volume / 20)
        assert integral([1, 1]) == pytest.approx(volume / 10)
        for vs, mu in (([0, 1, 2], 1), ([0, 0, 3], 2), ([2, 2, 2], 6)):
            assert integral(vs) == pytest.approx(volume * mu / 120)
        for vs, mu in (([0, 1, 2, 3], 1), ([0, 0, 1, 2], 2), ([0, 0, 1, 1], 4),
                       ([3, 3, 3, 0], 6), ([1, 1, 1, 1], 24)):
            assert integral(vs) == pytest.approx(volume * mu / 840)
        # The barycentric coordinates sum to one, so summing a factor out lowers the order.
        assert sum(integral([0, 1, c]) for c in range(4)) == pytest.approx(integral([0, 1]))
        assert sum(integral(list(t)) for t in itertools.product(range(4), repeat=4)) == pytest.approx(volume)

    def test_a_triangle_with_complex_lengths_follows_the_branch_volume(self):
        K, s = torus33(h=1.0, v=-0.5, dgl=0.5)
        cert = ch.WhitneyMass.certificate(K, s)
        top = K.orientedTopSimplices()[4]
        value = ch.WhitneyMass.vertexProductIntegral(K, s, 4, [top[0], top[0], top[2]])
        assert value == pytest.approx(cert.volumes[4] * 2 * 2 / math.factorial(5))

    def test_refusals(self):
        K = cob.ChainComplex.fromTopCells([[0, 1, 2, 3], [1, 2, 3, 4]])
        s = [1.0] * K.numSimplices(1)
        with pytest.raises(ValueError, match="not a vertex"):
            ch.WhitneyMass.vertexProductIntegral(K, s, 0, [0, 4])
        with pytest.raises(ValueError, match="out of range"):
            ch.WhitneyMass.vertexProductIntegral(K, s, 2, [0])


class TestWeightedMass:
    def test_unit_potential_is_the_mass_matrix(self):
        rng = np.random.default_rng(2)
        K2, _ = torus33()
        _, K3, s3 = _grid_instance()
        for K, s in ((K2, random_allowable(K2, rng)), (K3, s3)):
            M0 = ch.WhitneyMass.assemble(K, s, 0)
            MV = ch.WhitneyMass.assembleVertexPotential(K, s, [1.0] * K.numSimplices(0))
            assert abs(MV - M0).max() < 1e-14 * abs(M0).max()

    def test_entries_are_the_three_factor_integrals(self):
        rng = np.random.default_rng(4)
        K = cob.ChainComplex.fromTopCells([[0, 1, 2, 3], [1, 2, 3, 4]])
        s = random_allowable(K, rng)
        V = [complex(rng.normal(), rng.normal()) for _ in range(5)]
        MV = ch.WhitneyMass.assembleVertexPotential(K, s, V).toarray()
        assert np.abs(MV - MV.T).max() < 1e-15
        expected = np.zeros((5, 5), dtype=complex)
        for t, cell in enumerate(K.orientedTopSimplices()):
            for a, b, c in itertools.product(cell, repeat=3):
                expected[a, b] += V[c] * ch.WhitneyMass.vertexProductIntegral(K, s, t, [a, b, c])
        assert np.abs(MV - expected).max() < 1e-14
        with pytest.raises(ValueError, match="one value per vertex"):
            ch.WhitneyMass.assembleVertexPotential(K, s, V[:-1])

    def test_a_linear_potential_is_integrated_exactly(self):
        """With V linear in the fractional coordinates inside one cell the
        interpolant is V itself, so sum_ab M_0[V]_ab = int V over the mesh. The
        periodic wrap makes V discontinuous, so compare on a tetrahedron."""
        K = cob.ChainComplex.fromTopCells([[0, 1, 2, 3]])
        s = [1.0] * 6
        V = [0.3, -1.0, 2.0, 0.7]
        MV = ch.WhitneyMass.assembleVertexPotential(K, s, V)
        assert MV.sum() == pytest.approx(math.sqrt(2) / 12 * sum(V) / 4)


class TestDensity:
    def test_density_is_the_derivative_of_the_weighted_trace(self):
        rng = np.random.default_rng(6)
        _, K, s = _grid_instance()
        n0 = K.numSimplices(0)
        X = rng.normal(size=(n0, 3)) + 1j * rng.normal(size=(n0, 3))
        Y = rng.normal(size=(n0, 3)) + 1j * rng.normal(size=(n0, 3))
        rho = np.array(ch.WhitneyMass.vertexDensityContraction(K, s, X, Y))
        M0 = ch.WhitneyMass.assemble(K, s, 0)
        assert rho.sum() == pytest.approx(np.trace(X.T @ (M0 @ Y)), rel=1e-12)
        for c in (0, 7, n0 - 1):
            unit = [0.0] * n0
            unit[c] = 1.0
            Mc = ch.WhitneyMass.assembleVertexPotential(K, s, unit)
            assert rho[c] == pytest.approx(np.trace(X.T @ (Mc @ Y)), rel=1e-12)
        with pytest.raises(ValueError, match="same m"):
            ch.WhitneyMass.vertexDensityContraction(K, s, X, Y[:, :2])

    def test_a_normalized_plane_wave_has_uniform_density(self):
        grid, K, s = _grid_instance()
        n0 = K.numSimplices(0)
        frac = np.array([grid.fractionalCoordinates(v) for v in range(n0)])
        z = np.exp(2j * np.pi * frac[:, 0])[:, None]
        rho = np.array(ch.WhitneyMass.vertexDensityContraction(K, s, z.conj(), z))
        assert np.abs(rho.imag).max() < 1e-14
        assert np.ptp(rho.real) < 1e-14


class TestDressedPotential:
    def test_dressing_follows_the_mass_matrix(self):
        grid = tessera.PeriodicKuhnGrid.cubic(3)
        K, base, cov = covariant_torus(grid, (0.3, -0.2, 0.1))
        n0 = K.numSimplices(0)
        ones = cov.dressedVertexPotential([1.0] * n0)
        assert abs(ones - cov.Minv(0)).max() < 1e-14
        rng = np.random.default_rng(8)
        V = list(rng.normal(size=n0))
        dressed = cov.dressedVertexPotential(V).toarray()
        plain = ch.WhitneyMass.assembleVertexPotential(K, base.squaredLengths(), V).toarray()
        U = cov.connection()
        for a, b in zip(*np.nonzero(plain)):
            assert dressed[a, b] == pytest.approx(plain[a, b] * U.link(int(a), int(b)), rel=1e-13)
        # A real potential keeps the dressed pencil Hermitian.
        assert np.abs(dressed - dressed.conj().T).max() < 1e-15

    def test_a_constant_potential_shifts_every_level(self):
        grid = tessera.PeriodicKuhnGrid.cubic(3)
        K, _, cov = covariant_torus(grid, (0.3, -0.2, 0.1))
        pencil = cov.sparsePencil()
        shift = 2.5
        shifted = pencil.A + cov.dressedVertexPotential([shift] * K.numSimplices(0))
        free = ch.SparsePencilSolver.lowest(pencil.A, pencil.M, 6, -1.0)
        moved = ch.SparsePencilSolver.lowest(shifted, pencil.M, 6, -1.0)
        assert moved.eigenvalues.certificate.holds()
        assert np.array(moved.eigenvalues.values).real == pytest.approx(
            np.array(free.eigenvalues.values).real + shift, rel=1e-10)
