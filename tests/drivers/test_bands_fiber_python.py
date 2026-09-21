# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Stage 2b of the band-structure drivers (#1159): a static potential as the
phase of the timelike edges of the history complex. A constant potential is a
pure gauge and shifts every decay rate of the tick map exactly; the free tick
map converges at second order in the tick to a Klein-Gordon pencil; and the
level shift a potential induces approaches the static route's under refinement."""
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


def test_the_free_tick_map_converges_to_its_exact_limit_at_second_order(cell):
    """As the tick tends to zero the decay rates solve (A + m^2 M) u = E^2 D u, with
    M the consistent and D the lumped mass matrix: the time stiffness of a
    staircase slab is lumped, the mass term is the Whitney mass matrix."""
    mass = 40.0
    limit = klein_gordon_levels(cell, mass, cell.size)
    errors = [np.abs(np.sort(HistorySlab(cell, tau, None, mass).tick_levels(cell.size).real) - limit).max()
              for tau in (0.004, 0.002, 0.001)]
    assert 3.7 < errors[0] / errors[1] < 4.3 and 3.7 < errors[1] / errors[2] < 4.3
    # Here m h = 13, so the mismatch of the two mass matrices is multiplied by m^2
    # and the levels spread far below the rest mass.
    assert limit[0] < 0.6 * mass


def test_the_reduction_to_the_static_route_approaches_it_with_the_mesh():
    """The level shift a potential induces, in the tick map and in the static
    route with the kinetic scale 1 / 2m. On cells of three to six divisions the
    tick map over-responds by a factor that falls toward one as the mesh is
    refined."""
    mass, tau = 6.0, 0.002
    ratios = []
    for divisions in (3, 4, 6):
        cell = CrystalCell.cubic(1.0, divisions, kinetic_scale=1.0)
        V = pot.cosine_potential(cell, 0.4)
        lowest = lambda potential: np.sort(HistorySlab(cell, tau, potential, mass).tick_levels(1).real)[0]
        tick = lowest(V) - lowest(None)
        static = static_levels(cell, V, mass, 1)[0] - static_levels(cell, None, mass, 1)[0]
        ratios.append(tick / static)
    assert ratios[0] > ratios[1] > ratios[2] > 1.0 and ratios[2] < 2.0


def test_the_declared_fields_and_the_layer_api_give_the_same_slab(cell):
    """The slab built from a `Spacetime` whose edges carry the squared lengths
    and the phases, assembled and reduced by `PencilLayer`, is the slab of
    `HistorySlab`; over two ticks its boundary response is the Schur complement
    of two stacked slabs onto the outer levels."""
    from tessera.drivers.bands.fiber import layered_response
    V = pot.cosine_potential(cell, 0.4)
    mass, tau = 40.0, 0.004
    F00, F01, F10, F11 = HistorySlab(cell, tau, V, mass).blocks
    one, assembly_residual, _ = layered_response(cell, tau, V, mass, layers=1)
    direct = np.block([[F00, F01], [F10, F11]])
    assert assembly_residual < 1e-12
    assert np.abs(one - direct).max() < 1e-10 * np.abs(direct).max()
    two, _, solve_residual = layered_response(cell, tau, V, mass, layers=2)
    inner = np.linalg.inv(F11 + F00)
    schur = np.block([[F00 - F01 @ inner @ F10, -F01 @ inner @ F01],
                      [-F10 @ inner @ F10, F11 - F10 @ inner @ F01]])
    assert solve_residual < 1e-12
    assert np.abs(two - schur).max() < 1e-10 * np.abs(schur).max()


def test_the_stiffness_of_the_timelike_connection_is_the_spatial_stiffness(cell):
    """Eliminating the fiber-edge phase at tree level leaves the Coulomb kernel
    because its stiffness on the history is the spatial stiffness matrix, up to
    the fiber measure 1 / tau, exactly."""
    from tessera.drivers.bands.fiber import fiber_edge_stiffness
    rng = np.random.default_rng(4)
    stiffness = cell.stiffness.dressed().real
    for tau in (0.05, 0.2):
        phase = rng.normal(size=cell.size)
        expected = phase @ (stiffness @ phase) / tau
        assert fiber_edge_stiffness(cell, tau, phase) == pytest.approx(expected, rel=1e-10)
    assert fiber_edge_stiffness(cell, 0.1, np.ones(cell.size)) == pytest.approx(0.0, abs=1e-10)   # pure gauge
