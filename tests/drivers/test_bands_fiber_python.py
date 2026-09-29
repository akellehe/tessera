# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Stage 2b of the band-structure drivers (#1159): a static potential as the
phase of the timelike edges of the history complex. A constant potential is a
pure gauge and shifts every decay rate of the tick map exactly; the free tick
map converges at second order in the tick to a Klein-Gordon pencil; and with a
potential it closes at second order in the mesh on the static relativistic
problem, in which nothing is expanded. The limit of the tick map with a
potential is a closed form (#1171), held to the expansion of the dressed
Whitney mass matrix in exact arithmetic and to the blocks of the slab."""
from fractions import Fraction

import numpy as np
import pytest

from tessera.drivers.bands import potentials as pot
from tessera.drivers.bands.crystal import CrystalCell
from tessera.drivers.bands.fiber import (HistorySlab, klein_gordon_levels, staircase_blocks, static_levels, tick_curvature,
                                         tick_limit, tick_limit_from_blocks, tick_limit_levels)


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


def test_the_tick_map_closes_on_the_static_relativistic_problem_at_second_order():
    """The baseline keeps everything: (A + m^2 M) u = D (E - V)^2 u, quadratic in
    E, with no expansion in the potential or in 1 / m. The lowest decay rate of
    the tick map approaches its lowest level like the square of the mesh
    spacing, and the response to the potential approaches the static one from
    below; neither depends on the tick."""
    mass, tau = 6.0, 0.002
    differences, responses = [], []
    for divisions in (3, 4, 6):
        cell = CrystalCell.cubic(1.0, divisions, kinetic_scale=1.0)
        V = pot.cosine_potential(cell, 0.4)
        lowest = lambda potential: np.sort(HistorySlab(cell, tau, potential, mass).tick_levels(1).real)[0]
        static, free = static_levels(cell, V, mass, 1)[0], static_levels(cell, None, mass, 1)[0]
        differences.append(lowest(V) - static)
        responses.append((lowest(V) - lowest(None)) / (static - free))
    assert responses[0] < responses[1] < responses[2] < 1.0 and responses[2] > 0.9
    assert differences[0] > differences[1] > differences[2] > 0.0
    # Second order: the difference times the square of the divisions stays bounded and falls.
    scaled = [d * n ** 2 for d, n in zip(differences, (3, 4, 6))]
    assert scaled[2] < scaled[1] < scaled[0] < 0.6


def staircase_expansion(V):
    """The time part of the dressed stiffness of one staircase prism over a
    spatial simplex with the potentials `V` on its vertices (ascending id),
    expanded in the tick in exact arithmetic, per unit volume of the spatial
    simplex: the coefficients of 1 / tau, 1 and tau, each as the blocks between
    the two levels. Nothing here is taken from the closed form. The Whitney
    mass matrix of degree one is |s| / ((D + 1)(D + 2)) [(1 + d_ik) G_jl - ...]
    on a simplex s of dimension D = d + 1 and volume tau |T| / (d + 1), of
    which the time part of G = <d lambda, d lambda> is t t^T / tau^2 with
    t = -1, +1 on the two ends of the vertical edge; the dressed stiffness
    transports from a vertex to the base vertex (the smallest id) of each edge
    at it, between the two base vertices, and on to the other vertex."""
    d = len(V) - 1
    D = d + 1
    rank = lambda vertex: (vertex[1], vertex[0])                 # the lower level has the smaller ids

    def exponent(x, y):                                          # log of the link U_xy, per unit tick
        if x[1] == y[1]:
            return Fraction(0)
        mean = (V[x[0]] + V[y[0]]) / 2
        return mean if x[1] < y[1] else -mean

    index = lambda vertex: vertex[0] + (d + 1) * vertex[1]
    orders = [[[Fraction(0)] * (2 * d + 2) for _ in range(2 * d + 2)] for _ in range(3)]
    for k in range(d + 1):
        vertices = [(i, 0) for i in range(k + 1)] + [(j, 1) for j in range(k, d + 1)]
        t = {vertex: 0 for vertex in vertices}
        t[(k, 0)], t[(k, 1)] = -1, 1
        moment = lambda a, b: Fraction(2 if a == b else 1, (D + 1) * (D + 2) * (d + 1))
        for v in vertices:
            for w in vertices:
                for q in vertices:
                    for r in vertices:
                        if q == v or r == w:
                            continue
                        value = (t[v] * t[w] * moment(q, r) - t[v] * t[r] * moment(q, w)
                                 - t[q] * t[w] * moment(v, r) + t[q] * t[r] * moment(v, w))
                        base, other = min(v, q, key=rank), min(w, r, key=rank)
                        s = exponent(v, base) + exponent(base, other) + exponent(other, w)
                        for order, factor in enumerate((1, s, s * s / 2)):
                            orders[order][index(v)][index(w)] += value * factor
    n = d + 1
    split = lambda F: [[[F[a + p * n][b + q * n] for b in range(n)] for a in range(n)] for p in (0, 1) for q in (0, 1)]
    return [split(F) for F in orders]                            # [order][F00, F01, F10, F11]


@pytest.mark.parametrize("dimension", [1, 2, 3, 4])
def test_the_curvature_blocks_are_the_expansion_of_the_dressed_whitney_mass_matrix(dimension):
    """In exact rational arithmetic: the sum of the four blocks has no term in
    1 / tau or of order one, F10 - F01 none in 1 / tau, tau (F10 + F01) / 2
    tends to minus the lumped mass, and what is left of S1 and S0 beyond 2 D V
    and -D V^2 is `staircase_blocks`, entry by entry."""
    rng = np.random.default_rng(dimension)
    n = dimension + 1
    for _ in range(4):
        V = [Fraction(int(a), int(b)) for a, b in zip(rng.integers(-9, 10, n), rng.integers(1, 7, n))]
        orders = staircase_expansion(V)
        total = lambda blocks: [[sum(block[a][b] for block in blocks) for b in range(n)] for a in range(n)]
        zero = [[0] * n for _ in range(n)]
        assert total(orders[0]) == zero and total(orders[1]) == zero
        F00, F01, F10, F11 = orders[0]
        assert [[F10[a][b] - F01[a][b] for b in range(n)] for a in range(n)] == zero
        assert [[(F10[a][b] + F01[a][b]) / 2 for b in range(n)] for a in range(n)] == \
            [[-Fraction(int(a == b), n) for b in range(n)] for a in range(n)]
        F00, F01, F10, F11 = orders[1]
        S1 = [[F10[a][b] - F01[a][b] for b in range(n)] for a in range(n)]
        S0 = total(orders[2])
        c0, c1 = staircase_blocks(np.array(V, dtype=object))
        for a in range(n):
            for b in range(n):
                assert c1[a, b] == S1[a][b] - (2 * V[a] / n if a == b else 0)
                assert c0[a, b] == S0[a][b] + (V[a] ** 2 / n if a == b else 0)
        assert any(c1[a, b] != 0 for a in range(n) for b in range(n))


def test_the_curvature_vanishes_for_a_constant_potential_and_is_symmetric(cell):
    C0, C1 = tick_curvature(cell, np.full(cell.size, 0.3))
    assert np.abs(C0).max() < 1e-16 and np.abs(C1).max() < 1e-16
    C0, C1 = tick_curvature(cell, pot.cosine_potential(cell, 0.4))
    assert np.abs(C0 - C0.T).max() == 0.0 and np.abs(C1 - C1.T).max() == 0.0
    assert 1e-3 < np.abs(C0).max() < 0.05 and 1e-3 < np.abs(C1).max() < 0.05
    # A shift of the potential by c is the shift of every level by c: S(E; V + c) = S(E - c; V).
    shift = 0.7
    V = pot.cosine_potential(cell, 0.4)
    S0, S1, S2 = tick_limit(cell, V, 6.0)
    T0, T1, T2 = tick_limit(cell, V + shift, 6.0)
    assert np.abs(T2 - S2).max() == 0.0
    assert np.abs(T1 - (S1 - 2.0 * shift * S2)).max() < 1e-14
    assert np.abs(T0 - (S0 - shift * S1 + shift ** 2 * S2)).max() < 1e-13


def rough_potential(cell, seed=1):
    """A potential that is not a pure gauge and not smooth on the mesh."""
    return pot.cosine_potential(cell, 0.4) + 0.3 * np.random.default_rng(seed).normal(size=cell.size)


@pytest.mark.parametrize("divisions", [3, 4, 5, 6])
def test_the_closed_form_is_the_limit_of_the_blocks_of_the_slab(divisions):
    """`tick_limit` against the symmetric parts of the slab's blocks at a small
    tick, extrapolated once: the difference falls like the fourth power of the
    tick, to below 1e-8 of matrices whose entries are of order one."""
    cell = CrystalCell.cubic(1.0, divisions, kinetic_scale=1.0)
    V, mass = rough_potential(cell), 6.0
    closed = tick_limit(cell, V, mass)
    errors = [max(np.abs(read - exact).max() for read, exact in zip(tick_limit_from_blocks(cell, V, mass, tau), closed))
              for tau in (4e-2, 2e-2)]
    assert errors[1] < 1e-8 and 12.0 < errors[0] / errors[1] < 20.0
    # Without the curvature terms the same comparison fails by their size.
    A, M = cell.stiffness.dressed().toarray().real, cell.mass.dressed().toarray().real
    D = np.diag(M.sum(axis=1))
    read = tick_limit_from_blocks(cell, V, mass, 2e-2)
    assert np.abs(read[0] - (A + mass ** 2 * M - D @ np.diag(V ** 2))).max() > 1e-3
    assert np.abs(read[1] - 2.0 * D @ np.diag(V)).max() > 1e-3
    assert np.abs(read[2] + D).max() < 1e-8


def test_the_closed_form_holds_on_a_skew_cell_with_unequal_divisions():
    lattice = np.array([[1.0, 0.1, 0.0], [0.2, 1.3, 0.1], [0.0, 0.3, 0.9]])
    cell = CrystalCell(lattice, (3, 4, 5), kinetic_scale=1.0)
    V, mass = rough_potential(cell, seed=2), 3.0
    closed = tick_limit(cell, V, mass)
    read = tick_limit_from_blocks(cell, V, mass, 1e-2)
    assert max(np.abs(a - b).max() for a, b in zip(read, closed)) < 1e-8


def test_the_tick_map_with_a_potential_converges_to_its_exact_limit_at_second_order(cell):
    """The decay rates of the tick map approach the levels of the closed-form
    limit like the square of the tick."""
    mass = 6.0
    V = pot.cosine_potential(cell, 0.4)
    limit = tick_limit_levels(cell, V, mass, 3)
    errors = [np.abs(np.sort(HistorySlab(cell, tau, V, mass).tick_levels(3).real) - limit).max() for tau in (4e-3, 2e-3)]
    assert errors[1] < 1e-4 and 3.8 < errors[0] / errors[1] < 4.2


def test_the_limit_closes_on_the_static_relativistic_problem_at_second_order_in_the_mesh():
    """With the tick gone what separates the framework from
    (A + m^2 M) u = D (E - V)^2 u is the curvature alone: the lowest level of
    the limit approaches the static one like the square of the mesh spacing,
    and the response to the potential approaches the static response."""
    mass = 6.0
    differences, responses = [], []
    meshes = (3, 4, 5, 6)
    for divisions in meshes:
        cell = CrystalCell.cubic(1.0, divisions, kinetic_scale=1.0)
        V = pot.cosine_potential(cell, 0.4)
        static, free = static_levels(cell, V, mass, 1)[0], static_levels(cell, None, mass, 1)[0]
        limit = tick_limit_levels(cell, V, mass, 1)[0]
        assert tick_limit_levels(cell, None, mass, 1)[0] == pytest.approx(free, abs=1e-10)
        differences.append(limit - static)
        responses.append((limit - free) / (static - free))
    assert differences == pytest.approx([0.06186, 0.02895, 0.01576, 0.01030], abs=2e-5)
    # Second order: the difference times the square of the divisions is bounded and settles (0.353 at 8, 0.346 at 10).
    scaled = [difference * n ** 2 for difference, n in zip(differences, meshes)]
    assert 0.34 < scaled[3] < scaled[2] < scaled[1] < scaled[0] < 0.56
    assert responses == pytest.approx([0.7366, 0.8203, 0.8766, 0.9102], abs=2e-4)


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
