# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""`WhitneyMass.pairLoads` (the load of a product of sections of two
connections) and `CovariantChainHodge.sparsePencilPhaseDerivativeAlong` (the
derivative of the sparse pencil along a uniform change of the link phases), on
the periodic grid with the flat connections of crystal momenta (#1163)."""
import numpy as np
import pytest
import scipy.sparse as sp

from tessera import chainhodge as ch
from tessera.drivers.bands.crystal import CrystalCell


@pytest.fixture(scope="module")
def cell():
    return CrystalCell(np.array([[5.0, 0.3, 0.0], [0.0, 6.0, 0.2], [0.1, 0.0, 7.0]]), (4, 5, 3), kinetic_scale=1.0)


def _links(cell, kappa):
    return cell.grid.blochLinks(cell.edges, [float(v) for v in kappa])


def test_pair_loads_with_a_trivial_first_connection_are_the_dressed_weighted_mass_matrix(cell):
    rng = np.random.default_rng(3)
    x = rng.standard_normal(cell.size) + 1j * rng.standard_normal(cell.size)
    Y = rng.standard_normal((cell.size, 3)) + 1j * rng.standard_normal((cell.size, 3))
    kappa = (0.2, -0.1, 0.35)
    loads = np.asarray(ch.WhitneyMass.pairLoads(cell.complex, cell.squared_lengths, _links(cell, (0, 0, 0)),
                                                _links(cell, kappa), x, Y))
    reference = sp.csc_matrix(cell.covariant(kappa).dressedVertexPotential(list(x))) @ Y
    assert np.abs(loads - reference).max() < 1e-14


def test_pair_loads_between_two_momenta_carry_their_difference(cell):
    """conj(psi') psi between sections of the momenta k' and k: the driver's
    per-simplex twists, and the charge of the load, which at equal momenta is
    the inner product in the dressed mass matrix."""
    rng = np.random.default_rng(4)
    z1 = rng.standard_normal(cell.size) + 1j * rng.standard_normal(cell.size)
    Z2 = rng.standard_normal((cell.size, 2)) + 1j * rng.standard_normal((cell.size, 2))
    k1, k2 = (0.5, 0.0, 0.0), (0.2, -0.3, 0.1)
    minus = tuple(-v for v in k1)
    loads = np.asarray(ch.WhitneyMass.pairLoads(cell.complex, cell.squared_lengths, _links(cell, minus),
                                                _links(cell, k2), z1.conj(), Z2))
    # Moving a reciprocal vector from the link phases into the vertex values leaves the sections, and so the
    # load, unchanged: exp(i G x) rides on the first factor, exp(-i G x) on the result.
    shift = np.array([1.0, 0.0, -1.0])
    wave = np.exp(2j * np.pi * (cell.index @ (shift / np.array(cell.divisions))))
    moved = np.asarray(ch.WhitneyMass.pairLoads(cell.complex, cell.squared_lengths,
                                                _links(cell, tuple(np.array(minus) + shift)), _links(cell, k2),
                                                z1.conj() * wave.conj(), Z2))
    assert np.abs(moved * wave[:, None] - loads).max() < 1e-13
    same = np.asarray(ch.WhitneyMass.pairLoads(cell.complex, cell.squared_lengths, _links(cell, tuple(-v for v in k2)),
                                               _links(cell, k2), z1.conj(), Z2))
    assert np.abs(same.sum(axis=0) - z1.conj() @ (cell.pencil(k2)[1] @ Z2)).max() < 1e-13
    with pytest.raises(ValueError):
        ch.WhitneyMass.pairLoads(cell.complex, cell.squared_lengths, _links(cell, k1)[:-1], _links(cell, k2), z1, Z2)


@pytest.mark.parametrize("kappa", [(0.0, 0.0, 0.0), (0.2, -0.1, 0.35)])
def test_the_phase_derivative_along_the_edge_displacements_is_the_momentum_derivative(cell, kappa):
    """With the weights w_e = dx_e (one Cartesian component of the displacement
    of every stored link) the derivative is that of the pencil with respect to
    the crystal momentum: a central difference of the assembled pencil, at the
    zone centre and at a generic momentum, and the weighted mass matrix with it."""
    axis, step = 1, 1e-5
    steps = np.array([cell.grid.displacement(int(x), int(y)) for x, y in cell.edges], dtype=float)
    weights = list(((steps / np.array(cell.divisions)) @ cell.lattice)[:, axis])          # Cartesian, per stored link
    V = np.cos(2.0 * np.pi * cell.fractional[:, 0]) + 0.3 * np.sin(2.0 * np.pi * cell.fractional[:, 2])
    cov = cell.covariant(kappa)
    derivative = cov.sparsePencilPhaseDerivativeAlong(weights)
    weighted = sp.csc_matrix(cov.dressedVertexPotentialPhaseDerivativeAlong(list(V.astype(complex)), weights))
    shift = np.linalg.solve(cell.reciprocal.T, np.eye(3)[axis] * step)            # reciprocal coordinates of dk
    plus, minus = tuple(np.array(kappa) + shift), tuple(np.array(kappa) - shift)
    A_plus, M_plus = cell.pencil(plus, V)
    A_minus, M_minus = cell.pencil(minus, V)
    dA = (A_plus - A_minus) / (2.0 * step)
    dM = (M_plus - M_minus) / (2.0 * step)
    assert abs(sp.csc_matrix(derivative.M) - dM).max() < 1e-8 * abs(M_plus).max() / step * step + 1e-9
    assert abs(sp.csc_matrix(derivative.A) + weighted - dA).max() < 1e-7 * abs(A_plus).max()
    with pytest.raises(ValueError):
        cov.sparsePencilPhaseDerivativeAlong(weights[:-1])
