# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Newman's leading-eigenvector community discovery (#848).

The strategy bisects on the sign pattern of the leading eigenvector of the
modularity matrix ``B_gamma = A - gamma k k^T / 2m``, recurses, and stops
where no group has a positive leading eigenvalue.  Two properties separate it
from the incumbent multilevel search and both are asserted here rather than
assumed:

* the community COUNT is a reading of the spectrum, not a parameter — no
  resolution ladder is supplied to any recovery test below; and
* the search carries no seed, so the same graph gives the same partition.

The leading-to-second eigenvalue gap is the strategy's certificate.  Where the
pair is degenerate the bisection is genuinely undetermined and the split is
REFUSED with a named reason, in the same discipline the rest of this layer
applies to spectral isolation.  Both directions of that rule are exercised.

Nothing here upgrades modularity's epistemic status: it remains a proposal
generator whose reads may never veto a certified fiber.
"""

import math
import unittest

import tessera

obs = tessera.observables
PM = obs.PersistentModularity
STRATEGY = obs.DiscoveryStrategy
REASON = obs.SplitReason


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------

def clique_chain(sizes, bridge=0.05):
    """Cliques joined in a PATH by one weak edge each.

    A path rather than a ring on purpose: a ring of identical cliques is
    cyclically symmetric, which makes the leading eigenpair exactly
    degenerate.  That case is a fixture of its own below.
    """
    src, tgt, weight, base, firsts = [], [], [], 0, []
    for size in sizes:
        firsts.append(base)
        for i in range(size):
            for j in range(i + 1, size):
                src.append(base + i)
                tgt.append(base + j)
                weight.append(1.0)
        base += size
    for a, b in zip(firsts, firsts[1:]):
        src.append(a)
        tgt.append(b)
        weight.append(bridge)
    return PM.fromWeightedEdges(src, tgt, weight)


def clique_ring(k, size, bridge=0.05):
    """Identical cliques joined in a RING: cyclically symmetric by design."""
    src, tgt, weight = [], [], []
    for c in range(k):
        base = c * size
        for i in range(size):
            for j in range(i + 1, size):
                src.append(base + i)
                tgt.append(base + j)
                weight.append(1.0)
    for c in range(k):
        src.append(c * size)
        tgt.append(((c + 1) % k) * size)
        weight.append(bridge)
    return PM.fromWeightedEdges(src, tgt, weight)


def square_grid(n, jitter=0.0):
    """An n x n lattice.  With jitter = 0 the lattice symmetry makes the
    leading pair degenerate; a tiny jitter breaks it."""
    src, tgt, weight = [], [], []

    def idx(r, c):
        return r * n + c

    for r in range(n):
        for c in range(n):
            if c + 1 < n:
                src.append(idx(r, c))
                tgt.append(idx(r, c + 1))
                weight.append(1.0 + jitter * (r * n + c))
            if r + 1 < n:
                src.append(idx(r, c))
                tgt.append(idx(r + 1, c))
                weight.append(1.0 + jitter * (r * n + c) * 0.5)
    return PM.fromWeightedEdges(src, tgt, weight)


def planted(groups, p_in, p_out, seed=12345):
    """Deterministic noisy planted partition — no numpy, no RNG state."""
    state = seed

    def nxt():
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        return ((state >> 11) & ((1 << 53) - 1)) / float(1 << 53)

    blocks, base = [], 0
    for size in groups:
        blocks.append(set(range(base, base + size)))
        base += size
    src, tgt, weight = [], [], []
    for i in range(base):
        for j in range(i + 1, base):
            same = any(i in b and j in b for b in blocks)
            if nxt() < (p_in if same else p_out):
                src.append(i)
                tgt.append(j)
                weight.append(1.0)
    return PM.fromWeightedEdges(src, tgt, weight)


def spectral_config(**overrides):
    cfg = obs.PersistentModularityConfig()
    cfg.strategy = STRATEGY.LeadingEigenvector
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def partition_of(slice_):
    """The partition as a hashable, order-free signature."""
    return tuple(sorted(tuple(sorted(c.support)) for c in slice_.components))


def recompute_q(graph, slice_):
    """Independent recompute of Q_gamma from the reported supports through
    the class's own exact closed form."""
    ids = list(graph.cellIds())
    position = {cell: i for i, cell in enumerate(ids)}
    labels = [0] * len(ids)
    for index, comp in enumerate(slice_.components):
        for cell in comp.support:
            labels[position[cell]] = index
    return graph.modularityGamma(labels, slice_.gamma)


def reasons_of(slice_):
    return [s.reason for s in slice_.splits]


# --------------------------------------------------------------------------


class StrategySelectionTest(unittest.TestCase):
    """The strategy is chosen by an enum, and the default is the incumbent."""

    def test_the_default_strategy_is_multilevel_aggregation(self):
        cfg = obs.PersistentModularityConfig()
        self.assertEqual(cfg.strategy, STRATEGY.MultilevelAggregation)

    def test_the_default_slice_reports_the_incumbent_strategy(self):
        graph = clique_chain([4, 5, 6])
        slice_ = graph.discover(1.0, obs.PersistentModularityConfig())
        self.assertEqual(slice_.strategy, STRATEGY.MultilevelAggregation)
        # The incumbent performs no bisections, so it has no split record.
        self.assertEqual(slice_.splits, [])

    def test_a_spectral_slice_reports_the_spectral_strategy(self):
        graph = clique_chain([4, 5, 6])
        slice_ = graph.discover(1.0, spectral_config())
        self.assertEqual(slice_.strategy, STRATEGY.LeadingEigenvector)
        self.assertGreater(len(slice_.splits), 0)

    def test_a_misspelled_strategy_is_an_error_not_a_silent_default(self):
        cfg = obs.PersistentModularityConfig()
        with self.assertRaises((AttributeError, TypeError)):
            cfg.strategy = "LeadingEigenvektor"
        self.assertEqual(cfg.strategy, STRATEGY.MultilevelAggregation)


class CountComesFromTheSpectrumTest(unittest.TestCase):
    """The community count is read off the spectrum: no resolution ladder is
    supplied to any test in this class."""

    def test_it_recovers_three_planted_cliques_of_equal_size(self):
        graph = clique_chain([5, 5, 5])
        slice_ = graph.discover(1.0, spectral_config())
        self.assertEqual(len(slice_.components), 3)
        self.assertEqual(sorted(len(c.support) for c in slice_.components),
                         [5, 5, 5])

    def test_it_recovers_three_planted_cliques_of_unequal_size(self):
        graph = clique_chain([4, 5, 6])
        slice_ = graph.discover(1.0, spectral_config())
        self.assertEqual(len(slice_.components), 3)
        self.assertEqual(sorted(len(c.support) for c in slice_.components),
                         [4, 5, 6])

    def test_it_recovers_six_planted_cliques(self):
        graph = clique_chain([4] * 6)
        slice_ = graph.discover(1.0, spectral_config())
        self.assertEqual(len(slice_.components), 6)

    def test_the_recovered_supports_are_exactly_the_planted_blocks(self):
        graph = clique_chain([4, 5, 6])
        slice_ = graph.discover(1.0, spectral_config())
        self.assertEqual(
            partition_of(slice_),
            (tuple(range(0, 4)), tuple(range(4, 9)), tuple(range(9, 15))))

    def test_the_count_is_not_a_parameter_the_caller_supplied(self):
        """Only ONE gamma is ever passed, and different planted counts come
        back different: the number is a reading, not an input."""
        counts = []
        for blocks in ([5, 5], [5, 5, 5], [5, 5, 5, 5]):
            slice_ = clique_chain(blocks).discover(1.0, spectral_config())
            counts.append(len(slice_.components))
        self.assertEqual(counts, [2, 3, 4])


class StoppingRuleTest(unittest.TestCase):
    """Newman's rule in BOTH directions: a positive leading eigenvalue
    splits, a non-positive one refuses and says so."""

    def test_a_positive_leading_eigenvalue_is_split(self):
        graph = clique_chain([4, 5, 6])
        slice_ = graph.discover(1.0, spectral_config())
        accepted = [s for s in slice_.splits
                    if s.reason == REASON.SPLIT_ACCEPTED]
        self.assertGreater(len(accepted), 0)
        for split in accepted:
            self.assertTrue(split.accepted)
            self.assertTrue(split.resolved)
            self.assertGreater(split.leadingEigenvalue, 0.0)
            self.assertGreater(split.deltaQ, 0.0)
            self.assertGreater(split.sizeA, 0)
            self.assertGreater(split.sizeB, 0)

    def test_a_non_positive_leading_eigenvalue_refuses_and_is_named(self):
        graph = clique_chain([4, 5, 6])
        slice_ = graph.discover(1.0, spectral_config())
        stopped = [s for s in slice_.splits
                   if s.reason == REASON.NO_POSITIVE_EIGENVALUE]
        self.assertGreater(len(stopped), 0)
        for split in stopped:
            self.assertFalse(split.accepted)
            # A determined "do not split" is RESOLVED: the spectrum answered.
            self.assertTrue(split.resolved)
            self.assertLessEqual(split.leadingEigenvalue, 1e-9)

    def test_a_single_clique_is_indivisible(self):
        graph = clique_chain([6])
        slice_ = graph.discover(1.0, spectral_config())
        self.assertEqual(len(slice_.components), 1)
        self.assertEqual(reasons_of(slice_), [REASON.NO_POSITIVE_EIGENVALUE])

    def test_every_leaf_group_terminates_with_a_named_reason(self):
        graph = clique_chain([4, 5, 6])
        slice_ = graph.discover(1.0, spectral_config())
        self.assertGreater(len(slice_.splits), 0)
        for split in slice_.splits:
            self.assertTrue(split.reason)


class DegenerateLeadingPairTest(unittest.TestCase):
    """The gap is the certificate.  A degenerate pair means the bisection is
    not determined, and the split is refused rather than taken."""

    def test_a_symmetric_clique_ring_has_an_exactly_degenerate_pair(self):
        graph = clique_ring(4, 5)
        slice_ = graph.discover(1.0, spectral_config())
        self.assertEqual(len(slice_.splits), 1)
        split = slice_.splits[0]
        self.assertEqual(split.reason, REASON.DEGENERATE_LEADING_PAIR)
        self.assertFalse(split.accepted)
        # Refused because UNDETERMINED, which is distinct from a determined
        # "do not split".
        self.assertFalse(split.resolved)
        self.assertLess(abs(split.eigenvalueGap), 1e-9)
        self.assertLess(
            abs(split.leadingEigenvalue - split.secondEigenvalue), 1e-9)

    def test_the_refused_ring_is_left_as_one_community(self):
        slice_ = clique_ring(3, 5).discover(1.0, spectral_config())
        self.assertEqual(len(slice_.components), 1)

    def test_an_exact_lattice_is_degenerate_and_a_jittered_one_is_not(self):
        exact = square_grid(6).discover(1.0, spectral_config())
        self.assertEqual(exact.splits[0].reason,
                         REASON.DEGENERATE_LEADING_PAIR)
        self.assertEqual(len(exact.components), 1)

        jittered = square_grid(6, jitter=1e-3).discover(
            1.0, spectral_config())
        self.assertEqual(jittered.splits[0].reason, REASON.SPLIT_ACCEPTED)
        self.assertGreater(len(jittered.components), 1)
        self.assertGreater(jittered.splits[0].eigenvalueGap, 0.0)

    def test_relaxing_the_gap_threshold_admits_the_degenerate_split(self):
        """The refusal is the declared threshold speaking, not an inability
        to compute: drop the threshold below the measured gap and the same
        bisection is taken."""
        graph = clique_ring(4, 5)
        refused = graph.discover(1.0, spectral_config())
        self.assertEqual(refused.splits[0].reason,
                         REASON.DEGENERATE_LEADING_PAIR)
        admitted = graph.discover(
            1.0, spectral_config(minEigenvalueGap=-1.0))
        self.assertGreater(len(admitted.components), 1)

    def test_the_gap_is_reported_whenever_both_eigenvalues_are(self):
        slice_ = clique_chain([4, 5, 6]).discover(1.0, spectral_config())
        for split in slice_.splits:
            if not math.isnan(split.secondEigenvalue):
                self.assertAlmostEqual(
                    split.eigenvalueGap,
                    split.leadingEigenvalue - split.secondEigenvalue,
                    places=12)


class UnmeasuredIsNaNTest(unittest.TestCase):
    """Unmeasured quantities are NaN, never zero."""

    def test_a_group_that_stops_has_no_second_eigenvalue(self):
        slice_ = clique_chain([6]).discover(1.0, spectral_config())
        split = slice_.splits[0]
        self.assertTrue(math.isnan(split.secondEigenvalue))
        self.assertTrue(math.isnan(split.eigenvalueGap))
        self.assertTrue(math.isnan(split.deltaQ))

    def test_the_spectral_strategy_reports_no_restart_spread(self):
        slice_ = clique_chain([4, 5, 6]).discover(1.0, spectral_config())
        self.assertTrue(math.isnan(slice_.restartSpread))
        self.assertEqual(slice_.restarts, [])

    def test_the_incumbent_still_reports_a_real_restart_spread(self):
        slice_ = clique_chain([4, 5, 6]).discover(
            1.0, obs.PersistentModularityConfig())
        self.assertFalse(math.isnan(slice_.restartSpread))
        self.assertGreater(len(slice_.restarts), 0)


class DeterminismTest(unittest.TestCase):
    """No seed: the search is a pure function of the graph and gamma."""

    def test_repeated_runs_are_identical(self):
        graph = clique_chain([4, 5, 6])
        first = graph.discover(1.0, spectral_config())
        for _ in range(4):
            again = graph.discover(1.0, spectral_config())
            self.assertEqual(partition_of(again), partition_of(first))
            self.assertEqual(again.q, first.q)

    def test_the_seed_and_restart_count_are_ignored(self):
        graph = clique_chain([4, 5, 6])
        base = graph.discover(1.0, spectral_config())
        for seed, restarts in ((1, 1), (7, 3), (999999, 8), (2 ** 40, 2)):
            other = graph.discover(
                1.0, spectral_config(baseSeed=seed, restarts=restarts))
            self.assertEqual(partition_of(other), partition_of(base))
            self.assertEqual(other.q, base.q)

    def test_the_split_sequence_is_identical_across_runs(self):
        graph = clique_chain([4, 5, 6])
        first = graph.discover(1.0, spectral_config())
        again = graph.discover(1.0, spectral_config())
        self.assertEqual(reasons_of(first), reasons_of(again))
        self.assertEqual([s.groupSize for s in first.splits],
                         [s.groupSize for s in again.splits])


class ComparableScoreTest(unittest.TestCase):
    """Both strategies are scored by the SAME exact closed form, so their
    slices land on one scale."""

    FIXTURES = (
        ("chain 5,5,5", lambda: clique_chain([5, 5, 5])),
        ("chain 4,5,6", lambda: clique_chain([4, 5, 6])),
        ("chain 4x6", lambda: clique_chain([4] * 6)),
        ("grid 6 jitter", lambda: square_grid(6, jitter=1e-3)),
        ("planted 4x8", lambda: planted([8] * 4, 0.7, 0.08)),
    )

    def test_the_reported_q_is_the_exact_closed_form(self):
        for name, build in self.FIXTURES:
            with self.subTest(fixture=name):
                graph = build()
                for label, cfg in (("spectral", spectral_config()),
                                   ("incumbent",
                                    obs.PersistentModularityConfig())):
                    slice_ = graph.discover(1.0, cfg)
                    self.assertAlmostEqual(
                        recompute_q(graph, slice_), slice_.q, delta=5e-15,
                        msg="%s / %s" % (name, label))

    def test_the_spectral_ledger_agrees_with_the_cold_recompute(self):
        for name, build in self.FIXTURES:
            with self.subTest(fixture=name):
                slice_ = build().discover(1.0, spectral_config())
                self.assertAlmostEqual(slice_.qIncremental, slice_.q,
                                       delta=1e-14)

    def test_every_accepted_split_raises_the_exact_modularity(self):
        for name, build in self.FIXTURES:
            with self.subTest(fixture=name):
                slice_ = build().discover(1.0, spectral_config())
                for split in slice_.splits:
                    if split.reason == REASON.SPLIT_ACCEPTED:
                        self.assertGreater(split.deltaQ, 0.0)

    def test_the_accepted_deltas_sum_to_the_final_score(self):
        """Q starts at zero for the trivial one-community partition, so the
        accepted deltas of the exact closed form must telescope to q."""
        for name, build in self.FIXTURES:
            with self.subTest(fixture=name):
                slice_ = build().discover(1.0, spectral_config())
                total = sum(s.deltaQ for s in slice_.splits
                            if s.reason == REASON.SPLIT_ACCEPTED)
                self.assertAlmostEqual(total, slice_.q, delta=1e-13)


class KernighanLinRefinementTest(unittest.TestCase):
    """The refinement is on by default and its contribution is measured, not
    asserted."""

    def test_it_is_enabled_by_default(self):
        self.assertTrue(obs.PersistentModularityConfig().kernighanLinRefinement)

    def test_it_never_lowers_the_score(self):
        fixtures = (clique_chain([4, 5, 6]), square_grid(6, jitter=1e-3),
                    planted([8] * 4, 0.7, 0.08), planted([6] * 5, 0.8, 0.05),
                    planted([10] * 3, 0.6, 0.10))
        for index, graph in enumerate(fixtures):
            with self.subTest(fixture=index):
                withkl = graph.discover(1.0, spectral_config())
                without = graph.discover(
                    1.0, spectral_config(kernighanLinRefinement=False))
                self.assertGreaterEqual(withkl.objectiveValue,
                                        without.objectiveValue - 1e-12)

    def test_it_measurably_raises_the_score_on_a_noisy_partition(self):
        """On planted partitions with inter-block noise the sign bisection is
        not optimal and the refinement recovers a measurable amount."""
        graph = planted([8] * 4, 0.7, 0.08)
        withkl = graph.discover(1.0, spectral_config())
        without = graph.discover(
            1.0, spectral_config(kernighanLinRefinement=False))
        self.assertGreater(withkl.objectiveValue - without.objectiveValue, 0.01)


class DenseAndIterativeAgreeTest(unittest.TestCase):
    """The exact dense path and the power-iteration fallback are the same
    computation by two routes."""

    def test_the_iterative_route_reproduces_the_dense_partition(self):
        graph = clique_chain([4, 5, 6])
        dense = graph.discover(1.0, spectral_config())
        iterative = graph.discover(
            1.0, spectral_config(denseEigenSolveMaxGroup=0,
                                 maxPowerIterations=500000))
        self.assertEqual(partition_of(iterative), partition_of(dense))
        self.assertAlmostEqual(iterative.q, dense.q, places=9)

    def test_a_starved_iteration_refuses_rather_than_answering_wrongly(self):
        graph = clique_chain([4, 5, 6])
        starved = graph.discover(
            1.0, spectral_config(denseEigenSolveMaxGroup=0,
                                 maxPowerIterations=50))
        self.assertIn(REASON.POWER_ITERATION_NOT_CONVERGED,
                      reasons_of(starved))
        for split in starved.splits:
            if split.reason == REASON.POWER_ITERATION_NOT_CONVERGED:
                self.assertFalse(split.resolved)
                self.assertFalse(split.accepted)


class DegenerateInputTest(unittest.TestCase):
    """Edge cases are handled explicitly, not by luck."""

    def test_a_disconnected_graph_splits_into_its_components(self):
        graph = PM.fromWeightedEdges([0, 1, 3, 4], [1, 2, 4, 5], [1.0] * 4)
        slice_ = graph.discover(1.0, spectral_config())
        self.assertEqual(partition_of(slice_), ((0, 1, 2), (3, 4, 5)))

    def test_a_single_edge_is_indivisible(self):
        slice_ = PM.fromWeightedEdges([0], [1], [1.0]).discover(
            1.0, spectral_config())
        self.assertEqual(len(slice_.components), 1)

    def test_isolated_cells_do_not_break_the_search(self):
        graph = PM.fromWeightedEdges([0], [1], [1.0], [50, 51, 52])
        slice_ = graph.discover(1.0, spectral_config())
        self.assertGreaterEqual(len(slice_.components), 1)
        self.assertFalse(math.isnan(slice_.objectiveValue))

    def test_a_singleton_side_terminates_with_group_too_small(self):
        """A high gamma drives the recursion down to singletons, which is the
        only way a group of fewer than two cells is ever examined."""
        slice_ = clique_chain([4, 5, 6]).discover(8.0, spectral_config())
        self.assertIn(REASON.GROUP_TOO_SMALL, reasons_of(slice_))
        for split in slice_.splits:
            if split.reason == REASON.GROUP_TOO_SMALL:
                self.assertLess(split.groupSize, 2)
                self.assertTrue(math.isnan(split.leadingEigenvalue))


class HeuristicStatusUnchangedTest(unittest.TestCase):
    """Better proposals do not upgrade the proposal's status."""

    def test_the_spectral_search_is_not_claimed_to_be_optimal(self):
        """It is a heuristic for an NP-hard problem exactly as the incumbent
        is, and on some graphs it scores lower.  This test exists so that
        fact stays visible rather than being quietly assumed away."""
        graph = planted([10] * 3, 0.6, 0.10)
        spectral = graph.discover(1.0, spectral_config())
        incumbent = graph.discover(1.0, obs.PersistentModularityConfig())
        self.assertLess(spectral.objectiveValue, incumbent.objectiveValue)

    def test_both_strategies_produce_the_same_shape_of_component(self):
        graph = clique_chain([4, 5, 6])
        spectral = graph.discover(1.0, spectral_config())
        incumbent = graph.discover(1.0, obs.PersistentModularityConfig())
        for slice_ in (spectral, incumbent):
            for comp in slice_.components:
                self.assertTrue(comp.id.canonicalHash())
                self.assertGreater(len(comp.support), 0)
                self.assertFalse(math.isnan(abs(comp.modularityContribution)))

    def test_components_from_either_strategy_are_matchable(self):
        """Identity is hashed by the same rule, so the cross-strategy match
        is meaningful rather than an accident of representation."""
        graph = clique_chain([4, 5, 6])
        spectral = graph.discover(1.0, spectral_config())
        incumbent = graph.discover(1.0, obs.PersistentModularityConfig())
        matches = graph.matchComponents(spectral.components,
                                        incumbent.components)
        self.assertEqual(len(matches), len(spectral.components))
        for match in matches:
            self.assertAlmostEqual(match.supportOverlap, 1.0, places=12)


class ScanUsesTheSelectedStrategyTest(unittest.TestCase):
    """The resolution scan honours the strategy on every slice."""

    def test_every_slice_of_a_scan_is_spectral(self):
        graph = clique_chain([4, 5, 6])
        cfg = spectral_config()
        cfg.resolutions = [0.5, 1.0, 2.0]
        report = graph.scanResolutions(cfg)
        self.assertEqual(len(report.slices), 3)
        for slice_ in report.slices:
            self.assertEqual(slice_.strategy, STRATEGY.LeadingEigenvector)
            self.assertTrue(math.isnan(slice_.restartSpread))

    def test_a_scan_still_chains_persistence_tracks(self):
        graph = clique_chain([4, 5, 6])
        cfg = spectral_config()
        cfg.resolutions = [0.8, 1.0, 1.2]
        report = graph.scanResolutions(cfg)
        self.assertGreater(len(report.tracks), 0)


if __name__ == "__main__":
    unittest.main()
