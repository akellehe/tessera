# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Effective components (#1194): an effective Betti number is certified by the
gap between its band and the rest of the spectrum and by persistence across a
stated range of scales; the near-kernel of L_1 splits into the exact bridge
flows and the coexact near-cycles (the effective holes); the effective voids
are the coexact near-cycles of L_2 in three dimensions; and the supports of the
effective components come from the degree-zero band itself."""
import itertools

import numpy as np
import pytest

from tessera import chainhodge as ch
from tessera import cobordism as cob
from tessera import observables as obs
from tests.chainhodge._periodic import covariant_torus, cubic_grid

KAPPA = (0.3, -0.2, 0.1)
BRIDGE = 1.0e6  # the squared length of a bridge: a bottleneck of low conductance


def _operator(K, s, crossover=512):
    base = ch.ChainHodge(K, s, ch.Preset.L2, ch.Branch.Continuation, crossover)
    return ch.CovariantChainHodge(base, ch.Connection.trivial(K), 7, False)


def bridged_rings(count, bridge=BRIDGE, crossover=512):
    """`count` hexagonal rings, ring j on vertices 6j..6j+5, with ring j joined
    to ring j+1 by one edge (6j, 6j+6) of squared length `bridge`: one
    component and `count` holes by incidence, `count` effective components
    above the bridge level."""
    cells = [[6 * j + i, 6 * j + (i + 1) % 6] for j in range(count) for i in range(6)]
    bridges = [(6 * j, 6 * j + 6) for j in range(count - 1)]
    K = cob.ChainComplex.fromTopCells(cells + [list(b) for b in bridges])
    edges = [tuple(e) for e in K.kSimplexVertices(1)]
    s = [bridge if e in bridges else 1.0 for e in edges]
    return K, edges, _operator(K, s, crossover)


def kuhn_block(n, hollow=False):
    """The Kuhn triangulation of an n x n x n block of unit cubes at its
    Euclidean squared lengths; `hollow` removes the centre cube, leaving a
    cavity (Betti numbers (1, 0, 1, 0) instead of (1, 0, 0, 0))."""
    side = n + 1

    def vid(p):
        return p[0] + side * (p[1] + side * p[2])

    cells = []
    for origin in itertools.product(range(n), repeat=3):
        if hollow and origin == (n // 2,) * 3:
            continue
        for order in itertools.permutations(range(3)):
            p = list(origin)
            cell = [vid(p)]
            for axis in order:
                p[axis] += 1
                cell.append(vid(p))
            cells.append(cell)
    K = cob.ChainComplex.fromTopCells(cells)
    point = {vid(p): np.array(p, dtype=float) for p in itertools.product(range(side), repeat=3)}
    s = [float(np.sum((point[a] - point[b]) ** 2)) for a, b in K.kSimplexVertices(1)]
    return K, _operator(K, s)


def tetrahedron_boundary():
    K = cob.ChainComplex.fromTopCells([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
    return K, _operator(K, [1.0] * 6)


def boundary_matrix(K, k):
    """The integer boundary map d_k as a dense array (n_{k-1} x n_k)."""
    rows = {tuple(c): i for i, c in enumerate(K.kSimplexVertices(k - 1))}
    cols = K.kSimplexVertices(k)
    D = np.zeros((len(rows), len(cols)))
    for j, cell in enumerate(cols):
        for i in range(len(cell)):
            D[rows[tuple(cell[:i] + cell[i + 1:])], j] = (-1) ** i
    return D


class TestTheGapIsRequired:
    """A count is certified only when the enclosed band is separated from the
    rest of the spectrum by the declared minimum gap."""

    def test_a_window_that_cuts_through_the_spectrum_is_not_certified(self):
        _, _, cov = covariant_torus(cubic_grid(3))
        first = np.sort(np.abs(np.array(cov.spectrum(0).eigenvalues)))[1]
        zero = obs.EffectiveTopology.read(cov, 1.01 * first).degrees()[0]
        assert zero.rank == 7 and zero.converged
        assert zero.gap < obs.EffectiveTopology.DEFAULT_MINIMUM_GAP
        assert not zero.separated and not zero.certified
        verdict = obs.EffectiveComponents(7, 3).certify(cov, 1.01 * first)
        assert verdict.matches and not verdict.certified and not verdict.holds()

    def test_the_declared_gap_decides(self):
        """The first shell of the four-torus stands at half the second (gap 2):
        separated at a minimum gap of 1.5 and not at the default of 10."""
        _, _, cov = covariant_torus(cubic_grid(4), crossover=8)
        default = obs.EffectiveTopology.read(cov, 60.0).degrees()[0]
        assert default.rank == 7 and default.gap == pytest.approx(2.0, rel=1e-9)
        assert default.minimumGap == 10.0 and not default.certified
        relaxed = obs.EffectiveTopology.read(cov, 60.0, minimum_gap=1.5).degrees()[0]
        assert relaxed.minimumGap == 1.5 and relaxed.separated and relaxed.certified

    @pytest.mark.parametrize("fixture, betti", [
        (lambda: covariant_torus(cubic_grid(3))[::2], [1, 3, 3, 1]),
        (tetrahedron_boundary, [1, 0, 1]),
        (lambda: kuhn_block(3, hollow=True), [1, 0, 1, 0]),
        (lambda: kuhn_block(3), [1, 0, 0, 0]),
    ], ids=["torus", "sphere", "cavity", "ball"])
    def test_counts_equal_the_betti_numbers_only_where_the_gap_holds(self, fixture, betti):
        """Across scales from far below the first nonzero level to beyond the
        top of every spectrum, a certified count equals the Betti number of
        its degree, and every count that differs from it is refused by its gap."""
        K, cov = fixture()
        assert list(K.bettiNumbers()) == betti
        sweep = obs.EffectivePersistence.sweep(cov, list(np.geomspace(1e-8, 1e4, 49)))
        certified_somewhere = [False] * len(betti)
        for read in sweep.reads():
            for degree in read.degrees():
                if degree.certified:
                    assert degree.rank == betti[degree.degree], (read.epsilon(), degree.degree)
                    certified_somewhere[degree.degree] = True
                if degree.rank != betti[degree.degree]:
                    assert not degree.separated, (read.epsilon(), degree.degree, degree.gap)
        assert all(certified_somewhere)
        smallest = sweep.reads()[0]
        assert smallest.certified() and smallest.betti() == betti

    def test_a_minimum_gap_below_one_is_refused(self):
        _, _, cov = covariant_torus(cubic_grid(3))
        with pytest.raises(ValueError, match="at least 1"):
            obs.EffectiveTopology.read(cov, 1.0, minimum_gap=0.5)


class TestPersistence:
    """An effective count must persist, certified, across a stated range of
    scales."""

    def test_two_components_persist_between_the_bridge_and_the_ring_levels(self):
        _, _, cov = bridged_rings(2)
        scales = list(np.geomspace(1e-3, 2e-2, 6))
        persistence = obs.EffectivePersistence.sweep(cov, scales)
        assert persistence.persistentRank(0) == 2 and persistence.persistentRank(1) == 3
        assert persistence.betti() == [2, 3] and persistence.certified()
        [plateau] = persistence.plateaus(0)
        assert (plateau.rank, plateau.first, plateau.last, plateau.length()) == (2, 0, 5, 6)
        assert plateau.firstEpsilon == scales[0] and plateau.lastEpsilon == scales[-1]
        assert plateau.gap > 100.0
        verdict = obs.EffectiveComponents(2, 1).certify(persistence)
        assert verdict.holds() and verdict.scales == scales and verdict.epsilon == scales[0]
        assert not obs.EffectiveComponents(1, 1).certify(persistence).matches

    def test_a_range_that_crosses_the_bridge_level_does_not_persist(self):
        _, _, cov = bridged_rings(2)
        bridge = np.sort(np.abs(np.array(cov.spectrum(0).eigenvalues)))[1]
        persistence = obs.EffectivePersistence.sweep(cov, [bridge / 100, bridge / 10, 10 * bridge, 100 * bridge])
        runs = persistence.plateaus(0)
        assert [(p.rank, p.first, p.last) for p in runs] == [(1, 0, 1), (2, 2, 3)]
        assert persistence.persistentRank(0) == -1 and not persistence.certified()
        verdict = obs.EffectiveComponents(2, 1).certify(persistence)
        assert verdict.measured[0] == -1 and not verdict.certified and not verdict.holds()

    def test_an_uncertified_read_breaks_the_run(self):
        _, _, cov = covariant_torus(cubic_grid(3))
        first = np.sort(np.abs(np.array(cov.spectrum(0).eigenvalues)))[1]
        persistence = obs.EffectivePersistence.sweep(cov, [0.1 * first, 1.01 * first, 0.2 * first])
        assert [(p.rank, p.first, p.last) for p in persistence.plateaus(0)] == [(1, 0, 0), (1, 2, 2)]
        assert persistence.persistentRank(0) == -1

    def test_persistence_across_a_family_of_operators(self):
        """A sequence of operators read at one scale, here bridges of growing
        length standing for a relaxation: two components throughout."""
        reads = [obs.EffectiveTopology.read(bridged_rings(2, bridge)[2], 0.01) for bridge in (1e4, 1e5, 1e6)]
        persistence = obs.EffectivePersistence(reads)
        assert persistence.persistentRank(0) == 2 and len(persistence.reads()) == 3
        assert obs.EffectiveComponents(2, 1).certify(persistence).holds()

    def test_refusals(self):
        _, _, cov = bridged_rings(2)
        with pytest.raises(ValueError, match="no scales"):
            obs.EffectivePersistence.sweep(cov, [])
        with pytest.raises(ValueError, match="no reads"):
            obs.EffectivePersistence([])
        _, sphere = tetrahedron_boundary()
        with pytest.raises(ValueError, match="different dimensions"):
            obs.EffectivePersistence([obs.EffectiveTopology.read(cov, 0.01),
                                      obs.EffectiveTopology.read(sphere, 0.01)])
        with pytest.raises(ValueError, match="outside"):
            obs.EffectivePersistence.sweep(cov, [0.01]).plateaus(2)


class TestExactCoexactSplit:
    """ker_eps L_1 = d_1^# ker_eps L_0 (+) {coexact near-cycles}: the bridge
    flows of the effective components and the effective holes."""

    def test_two_bridged_rings_have_one_bridge_flow_and_two_holes(self):
        K, edges, cov = bridged_rings(2)
        assert list(K.bettiNumbers()) == [1, 2]
        split = obs.EffectiveTopology.split(cov, 1, 0.01)
        assert split.band.rank == 3 and split.band.certified
        assert (split.exact, split.coexact) == (1, 2) and split.certified
        # The exact part is d_1^# of the degree-zero band outside its kernel:
        # beta_0^eff(0.01) - beta_0^eff(0) bridge flows.
        zero = obs.EffectiveTopology.read(cov, 0.01).degrees()[0].rank
        kernel = obs.EffectiveTopology.read(cov, 1e-10).degrees()[0].rank
        assert split.exact == zero - kernel == 1
        # The effective holes are the coexact part, and here the incidence holes.
        assert split.coexact == K.bettiNumbers()[1]
        # The bridge flow is largest on the bridge and carries one ring's
        # divergence into the other; the holes are cycles.
        flow = split.exactFrame[:, 0]
        assert edges[int(np.argmax(np.abs(flow)))] == (0, 6)
        D = boundary_matrix(K, 1)
        divergence = (D @ flow) / flow[edges.index((0, 6))]
        assert np.all(divergence[:6].real < 0) and np.all(divergence[6:].real > 0)
        assert np.abs(D @ split.coexactFrame).max() < 1e-12
        assert split.closure < 1e-10 and split.frameResidual < 1e-10
        assert split.splitGap > 1e6 and split.rankTolerance < 1e-10
        assert len(split.boundarySingularValues) == 3

    def test_below_the_bridge_level_only_the_holes_remain(self):
        _, _, cov = bridged_rings(2)
        split = obs.EffectiveTopology.split(cov, 1, 1e-10)
        assert (split.band.rank, split.exact, split.coexact) == (2, 0, 2) and split.certified

    def test_the_torus_band_is_all_holes(self):
        K, _, cov = covariant_torus(cubic_grid(3))
        for k in (1, 2):
            split = obs.EffectiveTopology.split(cov, k, 1.0)
            assert (split.exact, split.coexact) == (0, 3) and split.certified
            assert np.abs(boundary_matrix(K, k) @ split.coexactFrame).max() < 1e-10
        # A coarser window takes in the exact first shell of degree one:
        # d_1^# of the six axis waves of degree zero.
        first = np.sort(np.abs(np.array(cov.spectrum(0).eigenvalues)))[1]
        coarse = obs.EffectiveTopology.split(cov, 1, 1.01 * first, minimum_gap=1.01)
        assert coarse.exact == 6 and coarse.closure < 1e-10

    def test_degree_zero_is_all_coexact(self):
        _, _, cov = bridged_rings(2)
        split = obs.EffectiveTopology.split(cov, 0, 0.01)
        assert (split.band.rank, split.exact, split.coexact) == (2, 0, 2) and split.certified
        sparse = obs.EffectiveTopology.split(bridged_rings(2, crossover=8)[2], 0, 0.01)
        assert sparse.band.method == obs.EffectiveBettiNumber.Method.SparsePencil
        assert (sparse.exact, sparse.coexact) == (0, 2) and sparse.certified

    def test_higher_degrees_above_the_crossover_are_unmeasured(self):
        _, _, cov = covariant_torus(cubic_grid(4), crossover=8)
        split = obs.EffectiveTopology.split(cov, 1, 1.0)
        assert split.exact == -1 and split.coexact == -1 and not split.certified
        assert "inverse metric" in split.reason
        with pytest.raises(ValueError, match="outside"):
            obs.EffectiveTopology.split(cov, 4, 1.0)


class TestVoids:
    """In three dimensions the effective voids are the coexact near-cycles of
    L_2, the proposed reading of an anti-cluster."""

    def test_a_cavity_is_one_effective_void(self):
        K, cov = kuhn_block(3, hollow=True)
        assert list(K.bettiNumbers()) == [1, 0, 1, 0]
        voids = obs.EffectiveTopology.voids(cov, 1e-6)
        assert voids.band.degree == 2 and voids.band.rank == 1 and voids.band.gap > 1e8
        assert (voids.exact, voids.coexact) == (0, 1) and voids.certified
        # The enclosing surface is a 2-cycle that bounds nothing in the complex.
        surface = voids.coexactFrame[:, 0]
        assert np.abs(boundary_matrix(K, 2) @ surface).max() < 1e-10
        D3 = boundary_matrix(K, 3)
        residual = surface - D3 @ np.linalg.lstsq(D3, surface, rcond=None)[0]
        assert np.linalg.norm(residual) > 0.1

    def test_the_filled_block_has_no_void(self):
        K, cov = kuhn_block(3)
        assert list(K.bettiNumbers()) == [1, 0, 0, 0]
        voids = obs.EffectiveTopology.voids(cov, 1e-6)
        assert (voids.band.rank, voids.exact, voids.coexact) == (0, 0, 0) and voids.certified

    def test_voids_are_read_in_three_dimensions(self):
        _, sphere = tetrahedron_boundary()
        with pytest.raises(ValueError, match="dimension three"):
            obs.EffectiveTopology.voids(sphere, 1e-6)


class TestComponentSupports:
    """The supports of the effective components are read from the degree-zero
    band: its committors, one per component."""

    def _assert_rings(self, partition, count):
        assert partition.band.rank == count and partition.certified
        supports = sorted(sorted(c.support) for c in partition.components)
        assert supports == [list(range(6 * j, 6 * j + 6)) for j in range(count)]
        pivots = [c.pivot for c in partition.components]
        for c in partition.components:
            assert c.pivot in c.support
            at_pivots = np.array([1.0 if p == c.pivot else 0.0 for p in pivots])
            assert np.abs(c.committor[pivots] - at_pivots).max() < 1e-10
            inside = np.array([v in c.support for v in range(6 * count)])
            assert np.abs(c.committor[inside] - 1.0).max() < 0.05
            assert np.abs(c.committor[~inside]).max() < 0.05
        assert partition.partitionDefect < 1e-10 and partition.pivotConditioning < 10.0

    def test_two_bridged_rings_are_two_supports(self):
        self._assert_rings(obs.EffectiveTopology.components(bridged_rings(2)[2], 0.01), 2)

    def test_three_bridged_rings_are_three_supports(self):
        self._assert_rings(obs.EffectiveTopology.components(bridged_rings(3)[2], 0.01), 3)

    def test_the_sparse_band_gives_the_same_supports_above_the_crossover(self):
        partition = obs.EffectiveTopology.components(bridged_rings(3, crossover=8)[2], 0.01)
        assert partition.band.method == obs.EffectiveBettiNumber.Method.SparsePencil
        self._assert_rings(partition, 3)

    def test_below_the_bridge_level_there_is_one_component(self):
        partition = obs.EffectiveTopology.components(bridged_rings(2)[2], 1e-10)
        [component] = partition.components
        assert component.support == list(range(12)) and partition.certified
        assert np.abs(component.committor - 1.0).max() < 1e-10

    def test_a_connection_with_holonomy_has_no_component(self):
        _, _, cov = covariant_torus(cubic_grid(3), KAPPA)
        partition = obs.EffectiveTopology.components(cov, 1.0)
        assert partition.band.rank == 0 and partition.components == [] and partition.certified
