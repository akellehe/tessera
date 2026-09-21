# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Stage 7 of the band-structure drivers (#1159): the derivative of a band
energy with respect to the squared edge lengths, held to Euler's identity and
to finite differences."""
import numpy as np
import pytest
import scipy.sparse as sp

from tessera import chainhodge as ch
from tessera.drivers.bands import potentials as pot
from tessera.drivers.bands import response
from tessera.drivers.bands.crystal import CrystalCell, solve_pencil


def _energy(cell, squared, potential, sigma):
    base = ch.ChainHodge(cell.complex, squared, ch.Preset.L2, ch.Branch.Continuation, 8)
    pencil = ch.CovariantChainHodge(base, ch.Connection.trivial(cell.complex), 7, False).sparsePencil()
    weighted = ch.WhitneyMass.assembleVertexPotential(cell.complex, squared, list(potential.astype(complex)))
    return solve_pencil(sp.csc_matrix(pencil.A) + weighted, pencil.M, 1, sigma).energies[0]


def test_free_levels_obey_eulers_identity():
    cell = CrystalCell.cubic(1.0, 4, kinetic_scale=1.0)
    read = cell.solve((0, 0, 0), 2)
    derivative = response.length_derivative(cell, read, 1)
    assert abs(response.euler_defect(cell, read, 1, derivative)) < 1e-10 * read.energies[1]
    # The constant is a level at zero for every geometry.
    assert np.abs(response.length_derivative(cell, read, 0)).max() < 1e-10


def test_a_potential_drops_out_of_the_euler_sum_and_the_force_matches_finite_differences():
    cell = CrystalCell.cubic(1.0, 4, kinetic_scale=1.0)
    V = pot.cosine_potential(cell, 10.0)
    read = cell.solve((0, 0, 0), 1, V)
    derivative = response.length_derivative(cell, read, 0, V)
    z = read.vectors[:, 0]
    potential_energy = (z.conj() @ (cell.weighted_mass(V).dressed() @ z)).real
    assert abs(response.euler_defect(cell, read, 0, derivative, potential_energy)) < 1e-10
    # An occupied mode pushes its support to expand: the length-weighted force is dilating.
    assert np.dot(np.array(cell.squared_lengths).real, derivative) < 0.0

    step = 1e-6
    for edge in (0, 17, 101):
        up, down = list(cell.squared_lengths), list(cell.squared_lengths)
        up[edge] += step
        down[edge] -= step
        finite = (_energy(cell, up, V, -40.0) - _energy(cell, down, V, -40.0)) / (2 * step)
        assert derivative[edge] == pytest.approx(finite, rel=1e-6, abs=1e-9)


def test_away_from_the_zone_centre_is_refused():
    cell = CrystalCell.cubic(1.0, 4, kinetic_scale=1.0)
    with pytest.raises(NotImplementedError):
        response.length_derivative(cell, cell.solve((0.3, 0.0, 0.0), 1), 0)
