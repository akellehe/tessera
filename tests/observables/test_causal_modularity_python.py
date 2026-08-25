# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Causally-aware community discovery (#849).

``PersistentModularity`` built its similarity graph from ``exp(-|l|)``, which
is CAUSALLY BLIND: a timelike edge and a spacelike edge of equal magnitude
produce the identical weight, so no partition read off that operator can
depend on causal structure at all.  ``CausalExpNegAbsLength`` carries the
causal character as the weight's SIGN -- read from ``Edge.disposition()``,
the single ``arg(l^2)`` classifier -- and the resulting signed graph is
scored with the Gomez-Jensen-Arenas null model.

Three properties are asserted here rather than assumed:

* **exact reduction** -- a wholly nonnegative graph scores BIT-IDENTICALLY
  to the incumbent, tested with ``==`` and never with a tolerance.  The
  signed branch is a branch, not a generalization the unsigned case falls
  out of: ``frac*frac`` with ``frac = s/2m`` is a different floating-point
  computation from ``s*s/(2m*2m)`` even where the two agree exactly in the
  reals, so the unsigned path runs the incumbent's expressions verbatim;
* **causal discrimination** -- the acceptance criterion of the ticket.  A
  timelike and a spacelike edge of equal magnitude give different operators,
  and on a complex where causal character is the ONLY distinguishing feature
  the two maps discover different communities;
* **refusal over approximation** -- an edge whose ``arg(l^2)`` is generic has
  no causal sign, and the map raises naming its reason rather than falling
  back on the magnitude.

Closed-form anchors (derived, not fitted).  On the causal ``K6`` fixture
below -- six cells, every edge of magnitude one, the two triples joined by
timelike edges and each triple internally spacelike -- the signed score of
the two-triple partition is exactly ``1/2`` and of the one-community
partition exactly ``0``, both INDEPENDENT of the common magnitude ``w``:

    2m+ = 12w,  2m- = 18w,  T = 30w,  S_c+ = 6w,  S_c- = 9w,
    Q = 2 (6w - (36w^2/12w - 81w^2/18w)) / 30w = 2 (7.5w) / 30w = 1/2.

The SAME complex under the blind map is the homogeneous complete graph, on
which that partition scores ``2 (6w/30w - (15w/30w)^2) = -1/10`` -- worse
than the null model, so the blind operator refuses the split it cannot see a
reason for.  That pair of numbers, +1/2 against -1/10 on one complex, is the
causal discrimination.

Nothing here upgrades modularity's epistemic status: it remains a proposal
generator whose reads may never veto a certified fiber.
"""

import math
import unittest

import tessera

obs = tessera.observables
PM = obs.PersistentModularity
STRATEGY = obs.DiscoveryStrategy
WEIGHT = PM.WeightMap

MACHINE = 1e-12  # machine-precision claims on exact closed forms

SPACELIKE_UNIT = 1.0 + 0j   # l^2 = +1: arg(l^2) = 0
TIMELIKE_UNIT = 1j          # l^2 = -1: arg(l^2) = pi
LIGHTLIKE_UNIT = complex(math.cos(math.pi / 4), math.sin(math.pi / 4))
MIXED_LENGTH = complex(1.0, 0.5)  # a generic argument: no causal character

TRIPLE_A = (0, 1, 2)
TRIPLE_B = (3, 4, 5)


# --------------------------------------------------------------------------
# fixture builders
# --------------------------------------------------------------------------
def _from_simplices(num_vertices, simplices):
    """The explicit-complex idiom shared with the crossing-readout and
    spectral-fiber suites."""
    sig = tessera.Signature(4, tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    st = tessera.Spacetime(metric, tessera.HERMITIAN_WEIGHTED, 1.0, 1.0,
                           tessera.PREFERRED, tessera.Toroid())
    verts = [st.createVertex(i) for i in range(num_vertices)]
    for simplex in simplices:
        st.createSimplex([verts[i] for i in simplex])
    for e in st.getEdgeList().toVector():
        e.setLength(SPACELIKE_UNIT)
        e.setPhase(0.0)
    return st


def _edge(st, a, b):
    for e in st.getEdgeList().toVector():
        if {e.getSource().getId(), e.getTarget().getId()} == {a, b}:
            return e
    raise KeyError((a, b))


def _complete_pairs(vertices):
    return [(a, b) for i, a in enumerate(vertices)
            for b in vertices[i + 1:]]


def causal_k6(inter_length=TIMELIKE_UNIT):
    """K6 with EVERY edge of magnitude one, so the blind map sees one
    homogeneous complete graph, and with causal character the only thing
    telling the two triples apart: spacelike inside each triple, and
    ``inter_length`` on the nine edges between them."""
    st = _from_simplices(6, _complete_pairs(list(TRIPLE_A + TRIPLE_B)))
    for a in TRIPLE_A:
        for b in TRIPLE_B:
            _edge(st, a, b).setLength(inter_length)
    return st


def two_vertices(length):
    st = _from_simplices(2, [(0, 1)])
    _edge(st, 0, 1).setLength(length)
    return st


def clique_chain(sizes, bridge=0.05):
    """Nonnegative cliques joined in a path -- the incumbent's own fixture,
    used here for the exact-reduction claims."""
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


def spectral_config(**overrides):
    cfg = obs.PersistentModularityConfig()
    cfg.strategy = STRATEGY.LeadingEigenvector
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def partition_of(slice_):
    """The partition as a hashable, order-free signature."""
    return tuple(sorted(tuple(sorted(c.support)) for c in slice_.components))


def labels_for(graph, groups):
    """Labels in ``cellIds()`` order for an explicit grouping of cell ids."""
    position = {cell: i for i, cell in enumerate(graph.cellIds())}
    labels = [0] * graph.nCells()
    for index, group in enumerate(groups):
        for cell in group:
            labels[position[cell]] = index
    return labels


class KahanSum:
    """The accumulator the C++ closed form uses, replicated so a reference
    score can be compared with ``==`` rather than a tolerance."""

    def __init__(self):
        self.total = 0.0
        self.compensation = 0.0

    def add(self, x):
        y = x - self.compensation
        t = self.total + y
        self.compensation = (t - self.total) - y
        self.total = t
        return self


def reference_unsigned_q(edges, labels_by_cell, gamma):
    """Q_gamma from the definition, in the incumbent's exact operation order:
    per-community ``lin/2m - gamma (S_c/2m)^2``, communities combined in
    ascending label order through a Kahan accumulator."""
    strength, internal = {}, {}
    two_m = 0.0
    for a, b, w in edges:
        strength[a] = strength.get(a, 0.0) + w
        strength[b] = strength.get(b, 0.0) + w
        if labels_by_cell[a] == labels_by_cell[b]:
            key = labels_by_cell[a]
            internal[key] = internal.get(key, 0.0) + 2.0 * w
    for cell in sorted(strength):
        two_m += strength[cell]
    totals = {}
    for cell in sorted(strength):
        key = labels_by_cell[cell]
        totals[key] = totals.get(key, 0.0) + strength[cell]
    accumulator = KahanSum()
    for label in sorted(totals):
        frac = totals[label] / two_m
        accumulator.add(internal.get(label, 0.0) / two_m - gamma * frac * frac)
    return accumulator.total


# --------------------------------------------------------------------------


class WeightMapDomainTest(unittest.TestCase):
    """The causal map is a third map, and the incumbent two are unchanged."""

    def test_the_blind_maps_are_still_offered(self):
        self.assertIsNotNone(WEIGHT.Unit)
        self.assertIsNotNone(WEIGHT.ExpNegAbsLength)

    def test_the_default_map_is_the_incumbent(self):
        st = causal_k6()
        default = PM.fromSpacetime(st)
        blind = PM.fromSpacetime(st, WEIGHT.ExpNegAbsLength)
        self.assertEqual(default.totalWeight2(), blind.totalWeight2())
        self.assertFalse(default.isSigned())

    def test_the_signed_branch_is_decided_by_the_graph_not_a_flag(self):
        """There is no setter: a graph is signed exactly when some weight
        is negative."""
        self.assertFalse(hasattr(PM, "setSigned"))
        self.assertFalse(clique_chain([4, 4]).isSigned())
        self.assertTrue(
            PM.fromWeightedEdges([0], [1], [-1.0]).isSigned())


class CausalDiscriminationTest(unittest.TestCase):
    """The ticket's acceptance criterion, demonstrated rather than assumed."""

    def test_equal_magnitude_timelike_and_spacelike_are_blind_map_identical(
            self):
        """The defect itself: the incumbent map cannot tell them apart."""
        spacelike = PM.fromSpacetime(two_vertices(SPACELIKE_UNIT),
                                     WEIGHT.ExpNegAbsLength)
        timelike = PM.fromSpacetime(two_vertices(TIMELIKE_UNIT),
                                    WEIGHT.ExpNegAbsLength)
        self.assertEqual(abs(SPACELIKE_UNIT), abs(TIMELIKE_UNIT))
        self.assertEqual(spacelike.totalWeight2(), timelike.totalWeight2())
        self.assertFalse(spacelike.isSigned())
        self.assertFalse(timelike.isSigned())

    def test_equal_magnitude_timelike_and_spacelike_give_different_operators(
            self):
        """The fix: same magnitude, different causal character, different
        operator."""
        spacelike = PM.fromSpacetime(two_vertices(SPACELIKE_UNIT),
                                     WEIGHT.CausalExpNegAbsLength)
        timelike = PM.fromSpacetime(two_vertices(TIMELIKE_UNIT),
                                    WEIGHT.CausalExpNegAbsLength)
        magnitude = 2.0 * math.exp(-1.0)

        # The spacelike edge leaves the graph unsigned, so it is scored by
        # the incumbent rank-one null model and carries no per-sign channel.
        self.assertFalse(spacelike.isSigned())
        self.assertAlmostEqual(spacelike.totalWeight2(), magnitude,
                               delta=MACHINE)
        self.assertEqual(spacelike.totalWeight2Positive(), 0.0)
        self.assertEqual(spacelike.totalWeight2Negative(), 0.0)
        # The timelike one switches the branch, and its weight lands wholly
        # in the negative channel.
        self.assertTrue(timelike.isSigned())
        self.assertEqual(timelike.totalWeight2Positive(), 0.0)
        self.assertAlmostEqual(timelike.totalWeight2Negative(), magnitude,
                               delta=MACHINE)
        # T is the ABSOLUTE total, so it agrees with 2m here even though the
        # signed sum of the timelike graph is -magnitude.
        self.assertAlmostEqual(timelike.totalWeight2(), magnitude,
                               delta=MACHINE)

    def test_causal_structure_alone_changes_the_communities(self):
        """A complex where causal character is the ONLY thing distinguishing
        the two blocks: every edge has magnitude one, so the blind map sees a
        homogeneous complete graph and finds nothing to split."""
        st = causal_k6()
        blind = PM.fromSpacetime(st, WEIGHT.ExpNegAbsLength)
        causal = PM.fromSpacetime(st, WEIGHT.CausalExpNegAbsLength)

        blind_slice = blind.discover(1.0, obs.PersistentModularityConfig())
        causal_slice = causal.discover(1.0, obs.PersistentModularityConfig())

        self.assertNotEqual(partition_of(blind_slice),
                            partition_of(causal_slice))
        self.assertEqual(len(blind_slice.components), 1)
        self.assertEqual(partition_of(causal_slice),
                         (tuple(TRIPLE_A), tuple(TRIPLE_B)))

    def test_the_two_maps_score_the_same_split_oppositely(self):
        """The closed-form anchor: +1/2 under the causal map against -1/10
        under the blind one, on ONE complex and ONE partition."""
        st = causal_k6()
        blind = PM.fromSpacetime(st, WEIGHT.ExpNegAbsLength)
        causal = PM.fromSpacetime(st, WEIGHT.CausalExpNegAbsLength)
        groups = (TRIPLE_A, TRIPLE_B)

        self.assertAlmostEqual(
            blind.modularityGamma(labels_for(blind, groups), 1.0), -0.1,
            delta=MACHINE)
        self.assertAlmostEqual(
            causal.modularityGamma(labels_for(causal, groups), 1.0), 0.5,
            delta=MACHINE)

    def test_the_one_community_partition_scores_zero_in_both(self):
        """The null partition is the null model, signed or not."""
        st = causal_k6()
        for weight_map in (WEIGHT.ExpNegAbsLength,
                           WEIGHT.CausalExpNegAbsLength):
            graph = PM.fromSpacetime(st, weight_map)
            whole = labels_for(graph, [TRIPLE_A + TRIPLE_B])
            self.assertAlmostEqual(graph.modularityGamma(whole, 1.0), 0.0,
                                   delta=MACHINE)

    def test_the_closed_form_is_independent_of_the_common_magnitude(self):
        """1/2 and 0 are properties of the causal pattern, not of ``w``."""
        st = causal_k6()
        for e in st.getEdgeList().toVector():
            e.setLength(e.getLength() * 3.0)
        causal = PM.fromSpacetime(st, WEIGHT.CausalExpNegAbsLength)
        self.assertAlmostEqual(
            causal.modularityGamma(labels_for(causal, (TRIPLE_A, TRIPLE_B)),
                                   1.0),
            0.5, delta=MACHINE)


class ExactReductionTest(unittest.TestCase):
    """A nonnegative graph reproduces the incumbent BIT-IDENTICALLY.  Every
    assertion in this class uses ``==``."""

    def test_an_all_spacelike_complex_scores_identically_under_both_maps(self):
        """The reduction through the real path: the causal map differs from
        the blind one only by a sign that is everywhere +1, so not a bit of
        the score may move."""
        st = causal_k6(inter_length=SPACELIKE_UNIT)
        blind = PM.fromSpacetime(st, WEIGHT.ExpNegAbsLength)
        causal = PM.fromSpacetime(st, WEIGHT.CausalExpNegAbsLength)
        self.assertFalse(causal.isSigned())
        self.assertEqual(blind.totalWeight2(), causal.totalWeight2())
        for groups in ((TRIPLE_A, TRIPLE_B),
                       (TRIPLE_A + TRIPLE_B,),
                       ((0, 3), (1, 4), (2, 5))):
            for gamma in (0.5, 1.0, 2.0):
                self.assertEqual(
                    blind.modularityGamma(labels_for(blind, groups), gamma),
                    causal.modularityGamma(labels_for(causal, groups), gamma))

    def test_discovery_is_identical_under_both_maps_when_all_spacelike(self):
        """Bit-identity through the full search, on both strategies."""
        st = causal_k6(inter_length=SPACELIKE_UNIT)
        blind = PM.fromSpacetime(st, WEIGHT.ExpNegAbsLength)
        causal = PM.fromSpacetime(st, WEIGHT.CausalExpNegAbsLength)
        for cfg in (obs.PersistentModularityConfig(), spectral_config()):
            blind_slice = blind.discover(1.0, cfg)
            causal_slice = causal.discover(1.0, cfg)
            self.assertEqual(partition_of(blind_slice),
                             partition_of(causal_slice))
            self.assertEqual(blind_slice.q, causal_slice.q)

    def test_the_closed_form_matches_an_independent_reference_bitwise(self):
        """The score against Q_gamma computed from the definition outside the
        library, in the same operation order."""
        gamma = 1.0
        edges = [(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0),
                 (3, 4, 1.0), (4, 5, 1.0), (3, 5, 1.0),
                 (0, 3, 0.25)]
        graph = PM.fromWeightedEdges([a for a, _, _ in edges],
                                     [b for _, b, _ in edges],
                                     [w for _, _, w in edges])
        self.assertFalse(graph.isSigned())
        for groups in ((TRIPLE_A, TRIPLE_B), (TRIPLE_A + TRIPLE_B,)):
            by_cell = {}
            for index, group in enumerate(groups):
                for cell in group:
                    by_cell[cell] = index
            self.assertEqual(
                graph.modularityGamma(labels_for(graph, groups), gamma),
                reference_unsigned_q(edges, by_cell, gamma))

    def test_the_incumbent_fixture_is_untouched_by_the_signed_branch(self):
        """The nonnegative graph the incumbent suite scores must not have
        acquired a signed channel."""
        graph = clique_chain([4, 5, 6])
        self.assertFalse(graph.isSigned())
        self.assertEqual(graph.totalWeight2Positive(), 0.0)
        self.assertEqual(graph.totalWeight2Negative(), 0.0)
        slice_ = graph.discover(1.0, obs.PersistentModularityConfig())
        self.assertEqual(len(slice_.components), 3)


class RefusalTest(unittest.TestCase):
    """An unreadable causal character is refused BY NAME, never replaced by
    the magnitude it was supposed to qualify."""

    def test_a_mixed_edge_makes_the_map_unavailable(self):
        st = causal_k6()
        _edge(st, 0, 1).setLength(MIXED_LENGTH)
        read = PM.causalWeightAvailability(st)
        self.assertFalse(read.available)
        self.assertEqual(read.reason, "mixed-causal-character")
        self.assertEqual(read.mixed, 1)

    def test_a_mixed_edge_raises_rather_than_falling_back(self):
        st = causal_k6()
        _edge(st, 0, 1).setLength(MIXED_LENGTH)
        with self.assertRaises(ValueError) as caught:
            PM.fromSpacetime(st, WEIGHT.CausalExpNegAbsLength)
        self.assertIn("mixed-causal-character", str(caught.exception))
        # The BLIND map still reads it: refusal is the causal map's, not a
        # property of the complex.
        self.assertGreater(
            PM.fromSpacetime(st, WEIGHT.ExpNegAbsLength).nEdges(), 0)

    def test_a_wholly_lightlike_complex_has_nothing_to_score(self):
        st = causal_k6()
        for e in st.getEdgeList().toVector():
            e.setLength(LIGHTLIKE_UNIT)
        read = PM.causalWeightAvailability(st)
        self.assertFalse(read.available)
        self.assertEqual(read.reason, "no-scorable-edges")
        self.assertEqual(read.lightlike, len(_complete_pairs(list(range(6)))))
        with self.assertRaises(ValueError) as caught:
            PM.fromSpacetime(st, WEIGHT.CausalExpNegAbsLength)
        self.assertIn("no-scorable-edges", str(caught.exception))

    def test_an_available_census_names_no_reason(self):
        read = PM.causalWeightAvailability(causal_k6())
        self.assertTrue(read.available)
        self.assertEqual(read.reason, "")
        self.assertEqual(read.spacelike, 6)   # three inside each triple
        self.assertEqual(read.timelike, 9)    # the three-by-three between
        self.assertEqual(read.mixed, 0)

    def test_the_census_totals_the_edges(self):
        """Every edge lands in exactly one bucket: the census cannot lose
        one and report availability it did not verify."""
        st = causal_k6()
        _edge(st, 0, 1).setLength(MIXED_LENGTH)
        _edge(st, 0, 2).setLength(LIGHTLIKE_UNIT)
        read = PM.causalWeightAvailability(st)
        total = (read.spacelike + read.timelike + read.lightlike +
                 read.mixed + read.degenerate)
        self.assertEqual(total, len(st.getEdgeList().toVector()))

    def test_non_finite_weights_are_refused(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(ValueError):
                PM.fromWeightedEdges([0], [1], [bad])


class DegenerateCaseTest(unittest.TestCase):
    """The cases the ticket names explicitly."""

    def test_a_wholly_timelike_complex_has_a_vanishing_positive_channel(self):
        """2m+ = 0 makes the positive channel's null term zero, not 0/0."""
        st = causal_k6()
        for e in st.getEdgeList().toVector():
            e.setLength(TIMELIKE_UNIT)
        graph = PM.fromSpacetime(st, WEIGHT.CausalExpNegAbsLength)
        self.assertTrue(graph.isSigned())
        self.assertEqual(graph.totalWeight2Positive(), 0.0)
        self.assertGreater(graph.totalWeight2Negative(), 0.0)
        score = graph.modularityGamma(
            labels_for(graph, (TRIPLE_A, TRIPLE_B)), 1.0)
        self.assertTrue(math.isfinite(score))

    def test_the_normalizer_is_the_absolute_total_not_the_signed_sum(self):
        """A graph whose SIGNED total vanishes still has a positive
        normalizer, which is why T rather than 2m is divided by."""
        graph = PM.fromWeightedEdges([0, 2], [1, 3], [1.0, -1.0])
        self.assertTrue(graph.isSigned())
        self.assertEqual(graph.totalWeight2Positive(), 2.0)
        self.assertEqual(graph.totalWeight2Negative(), 2.0)
        self.assertEqual(graph.totalWeight2(), 4.0)
        score = graph.modularityGamma(labels_for(graph, ((0, 1), (2, 3))), 1.0)
        self.assertTrue(math.isfinite(score))

    def test_cancelling_parallel_edges_drop_out(self):
        """A pair that consolidates to zero is a measured absence of net
        similarity, handled exactly as a zero weight always was."""
        graph = PM.fromWeightedEdges([0, 0, 1], [1, 1, 2], [1.0, -1.0, 1.0])
        self.assertEqual(graph.nEdges(), 1)
        self.assertFalse(graph.isSigned())

    def test_a_lightlike_edge_carries_no_similarity(self):
        """Zero Lorentzian interval, so zero signed weight, so no edge --
        distinct from an edge that was never there, which is why the census
        counts it."""
        st = causal_k6()
        _edge(st, 0, 1).setLength(LIGHTLIKE_UNIT)
        graph = PM.fromSpacetime(st, WEIGHT.CausalExpNegAbsLength)
        blind = PM.fromSpacetime(st, WEIGHT.ExpNegAbsLength)
        self.assertEqual(graph.nEdges(), blind.nEdges() - 1)
        self.assertEqual(PM.causalWeightAvailability(st).lightlike, 1)

    def test_a_disconnected_signed_graph_scores_and_splits(self):
        graph = PM.fromWeightedEdges(
            [0, 1, 0, 3, 4, 3], [1, 2, 2, 4, 5, 5],
            [1.0, 1.0, -1.0, 1.0, 1.0, -1.0])
        self.assertTrue(graph.isSigned())
        slice_ = graph.discover(1.0, obs.PersistentModularityConfig())
        self.assertTrue(math.isfinite(slice_.q))
        self.assertGreaterEqual(len(slice_.components), 2)

    def test_an_empty_edge_list_scores_zero(self):
        graph = PM.fromWeightedEdges([], [], [], [7, 8, 9])
        self.assertEqual(graph.nEdges(), 0)
        self.assertFalse(graph.isSigned())
        self.assertEqual(graph.modularityGamma([0, 0, 0], 1.0), 0.0)


class BothStrategiesTest(unittest.TestCase):
    """Both discovery strategies keep working on the signed operator, and
    they interact with it differently -- so both are exercised on it."""

    def _causal_graph(self):
        return PM.fromSpacetime(causal_k6(), WEIGHT.CausalExpNegAbsLength)

    def test_multilevel_aggregation_recovers_the_causal_blocks(self):
        graph = self._causal_graph()
        slice_ = graph.discover(1.0, obs.PersistentModularityConfig())
        self.assertEqual(slice_.strategy, STRATEGY.MultilevelAggregation)
        self.assertEqual(partition_of(slice_),
                         (tuple(TRIPLE_A), tuple(TRIPLE_B)))

    def test_leading_eigenvector_recovers_the_causal_blocks(self):
        graph = self._causal_graph()
        slice_ = graph.discover(1.0, spectral_config())
        self.assertEqual(slice_.strategy, STRATEGY.LeadingEigenvector)
        self.assertEqual(partition_of(slice_),
                         (tuple(TRIPLE_A), tuple(TRIPLE_B)))

    def test_both_strategies_agree_on_the_signed_score(self):
        graph = self._causal_graph()
        incumbent = graph.discover(1.0, obs.PersistentModularityConfig())
        spectral = graph.discover(1.0, spectral_config())
        self.assertAlmostEqual(incumbent.q, spectral.q, delta=MACHINE)
        self.assertAlmostEqual(incumbent.q, 0.5, delta=MACHINE)

    def test_the_signed_local_move_ledger_matches_the_cold_recompute(self):
        """The multilevel search accumulates the signed delta-Q closed form
        move by move; ``q`` recomputes the same partition from scratch.  A
        wrong delta-Q would agree with neither, so this is the direct check
        on the derived form

            dQ(v: a->b) = 2 (w_vb - w_va) / T
                        - (2 gamma / T) [ k_v+ (k_v+ + S_b+ - S_a+) / 2m+
                                        - k_v- (k_v- + S_b- - S_a-) / 2m- ]

        and on the two channels being inherited by summation across
        aggregation levels rather than recomputed from the coarse adjacency.
        """
        graph = self._causal_graph()
        self.assertTrue(graph.isSigned())
        for gamma in (0.5, 1.0, 2.0):
            slice_ = graph.discover(gamma, obs.PersistentModularityConfig())
            self.assertAlmostEqual(slice_.qIncremental, slice_.q,
                                   delta=MACHINE)

    def test_the_unsigned_ledger_is_unaffected(self):
        graph = clique_chain([4, 5, 6])
        self.assertFalse(graph.isSigned())
        slice_ = graph.discover(1.0, obs.PersistentModularityConfig())
        self.assertAlmostEqual(slice_.qIncremental, slice_.q, delta=MACHINE)

    def test_the_power_iteration_path_agrees_with_the_exact_one(self):
        """The shifted power iteration finds the most POSITIVE eigenvalue
        only while its Gershgorin shift really bounds the spectral radius.
        Once weights carry sign the adjacency part of that bound has to be
        the absolute row sum ``sum_j |A_ij|``: the group degree is a
        difference and can sit anywhere below it, so it bounds nothing.  (On
        a nonnegative graph the two coincide, which is why the incumbent's
        bound was correct for the graphs it had.)  This forces the iterative
        path onto a signed graph and holds it to the exact answer."""
        graph = self._causal_graph()
        exact = graph.discover(1.0, spectral_config(denseEigenSolveMaxGroup=1024))
        iterative = graph.discover(
            1.0, spectral_config(denseEigenSolveMaxGroup=0))
        self.assertEqual(partition_of(exact), partition_of(iterative))
        self.assertAlmostEqual(exact.q, iterative.q, delta=MACHINE)

    def test_a_degenerate_signed_leading_pair_is_refused_by_name(self):
        """The refusal discipline carries onto the signed operator.  This
        graph -- a uniformly negative K6 with three positive pairs laid over
        it -- has the exact spectrum {-3, -3, -3, 0, 9, 9}, so the leading
        pair is degenerate and the bisection is genuinely undetermined.  It
        is reported unresolved rather than taken."""
        graph = PM.fromWeightedEdges(
            [a for a, b in _complete_pairs(list(range(6)))] + [0, 2, 4],
            [b for a, b in _complete_pairs(list(range(6)))] + [1, 3, 5],
            [-3.0] * 15 + [6.0] * 3)
        self.assertTrue(graph.isSigned())
        slice_ = graph.discover(1.0, spectral_config())
        self.assertEqual([s.reason for s in slice_.splits],
                         ["degenerate-leading-pair"])
        self.assertFalse(slice_.splits[0].resolved)
        self.assertAlmostEqual(slice_.splits[0].leadingEigenvalue, 9.0,
                               delta=1e-9)
        self.assertEqual(len(slice_.components), 1)

    def test_the_spectral_search_still_carries_no_seed(self):
        graph = self._causal_graph()
        first = graph.discover(1.0, spectral_config(baseSeed=1))
        second = graph.discover(1.0, spectral_config(baseSeed=99999))
        self.assertEqual(partition_of(first), partition_of(second))


if __name__ == "__main__":
    unittest.main()
