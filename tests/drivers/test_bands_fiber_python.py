# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Stage 2b of the band-structure drivers (#1159): a static potential as the
phase of the timelike edges of the history complex. A constant potential is a
pure gauge and shifts every decay rate of the tick map exactly; the free tick
map converges at second order in the tick to a Klein-Gordon pencil; and its
non-relativistic reduction approaches the static route under refinement."""
import numpy as np
import pytest

from tessera.drivers.bands import potentials as pot
from tessera.drivers.bands.crystal import CrystalCell
from tessera.drivers.bands.fiber import HistorySlab, klein_gordon_levels, static_levels


@pytest.fixture(scope="module")
def cell():
    return CrystalCell.cubic(1.0, 3, kinetic_scale=1.0)


def test_a_constant_potential_is_a_pure_gauge_and_shifts_every_level(cell):
    free = HistorySlab(cell, 0.002, None, 40.0)
    constant = HistorySlab(cell, 0.002, np.full(cell.size, 0.7), 40.0)
    assert constant.max_curvature() < 1e-14 and free.max_curvature() == 0.0
    shift = np.sort(constant.tick_levels(cell.size).real) - np.sort(free.tick_levels(cell.size).real)
    assert np.abs(shift - 0.7).max() < 1e-8


def test_a_varying_potential_has_the_electric_field_as_curvature(cell):
    V = pot.cosine_potential(cell, 0.4)
    tau = 0.01
    slab = HistorySlab(cell, tau, V, 40.0)
    # A vertical triangle (a, b, b') is closed by the fiber edges a -> b' and
    # b -> b', whose mean potentials differ by half the step of V along a -> b.
    steepest = max(abs(V[a] - V[b]) for a, b in cell.edges)
    assert slab.max_curvature() == pytest.approx(np.exp(0.5 * tau * steepest) - 1.0, rel=1e-9)


@pytest.mark.parametrize("mass_term", ["lumped", "consistent"])
def test_the_free_tick_map_converges_to_its_klein_gordon_limit_at_second_order(cell, mass_term):
    mass = 40.0
    limit = klein_gordon_levels(cell, mass, cell.size, mass_term)
    errors = [np.abs(np.sort(HistorySlab(cell, tau, None, mass, mass_term).tick_levels(cell.size).real) - limit).max()
              for tau in (0.004, 0.002, 0.001)]
    assert 3.7 < errors[0] / errors[1] < 4.3 and 3.7 < errors[1] / errors[2] < 4.3
    if mass_term == "lumped":
        # m + L / 2m to first order, at any mesh spacing.
        assert limit[0] == pytest.approx(mass, rel=1e-12)
        kinetic = static_levels(cell, None, mass, 2)[1]                  # L / 2m
        assert limit[1] == pytest.approx(np.sqrt(mass ** 2 + 2 * mass * kinetic), rel=1e-12)
        assert limit[1] - mass == pytest.approx(kinetic - kinetic ** 2 / (2 * mass), rel=1e-4)
    else:
        # The consistent mass term mixes two mass matrices: with m h = 13 the
        # levels fall far below the rest mass and the reduction fails.
        assert limit[0] < 0.6 * mass


def _reduction_defect(divisions, scale, mass=100.0):
    cell = CrystalCell.cubic(1.0, divisions, kinetic_scale=1.0)
    V = scale * pot.cosine_potential(cell, 0.4)
    tau = 0.2 / mass ** 1.5
    coarse = np.sort(HistorySlab(cell, tau, V, mass).tick_levels(1).real)
    fine = np.sort(HistorySlab(cell, tau / 2, V, mass).tick_levels(1).real)
    tick = (4.0 * fine - coarse) / 3.0 - mass              # the tick error removed
    return (tick - static_levels(cell, V, mass, 1))[0]


def test_the_reduction_to_the_static_route_closes_under_refinement():
    """What is left between the two routes is first order in the potential and
    independent of the mass: on a connection with curvature the covariant
    operator transports through the base vertex of each cell, which samples the
    potential one mesh step away. It closes with the mesh."""
    coarse, fine = _reduction_defect(3, 1.0), _reduction_defect(6, 1.0)
    assert 0.0 < fine < 0.45 * coarse
    assert _reduction_defect(3, 0.5) == pytest.approx(0.5 * coarse, rel=0.2)
    assert _reduction_defect(3, 1.0, mass=200.0) == pytest.approx(coarse, rel=0.1)
