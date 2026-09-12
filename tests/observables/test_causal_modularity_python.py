# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Causally-aware community discovery (#849).

``PersistentModularity`` built its similarity graph from ``exp(-|l|)``, which
is CAUSALLY BLIND: a timelike edge and a spacelike edge of equal magnitude
produce the identical weight, so no partition read off that operator can
depend on causal structure at all.  ``CausalPhaseExpNegAbsLength`` keeps that
magnitude and carries the causal character as the weight's ARGUMENT,
``arg(l^2)`` -- the same measured quantity #870 classifies dispositions by.

The adjacency is therefore COMPLEX and SYMMETRIC.  Symmetric because a weight
is a property of the edge: its magnitude and argument do not depend on which
end you read it from.  Complex because both parts are physics -- and because
nothing then has to be bucketed, so a generic ``arg(l^2)`` is an ordinary edge
rather than a case to refuse.

What is asserted here rather than assumed:

* **exact reduction** -- a nonnegative real graph scores BIT-IDENTICALLY to
  the incumbent, tested with ``==`` and never a tolerance;
* **causal discrimination** -- a timelike and a spacelike edge of equal
  magnitude give different operators, and on a complex where causal character
  is the ONLY distinguishing feature the two maps discover different
  communities;
* **anti-communities are found and are distinguishable** -- a community bound
  by dissimilarity is a target, not a failure mode.  ``arg(Q) = pi`` says so,
  against ``arg(Q) = 0`` for an ordinary community, and the ``Magnitude``
  objective is what pursues it;
* **generic arguments are ordinary** -- the fixture with no definite edge at
  all scores without a single refusal.

Closed-form anchors, derived and independent of the common magnitude ``w``.
On the causal ``K6`` fixture -- six cells, every edge of magnitude one, each
triple internally spacelike and the nine edges between them timelike --

    k_i = 2w - 3w = -w,  SA = -6w,  T = 30w,  S_c = -3w,
    Q(two triples) = 2 (6w - (-3w)^2/(-6w)) / 30w = 2 (7.5w) / 30w = +1/2,

while the SAME complex under the blind map is the homogeneous complete graph,
where that partition scores ``2 (6w/30w - (15w/30w)^2) = -1/10``.  Both maps
put the one-community partition at exactly ``0``: that anchor is what makes
``|Q|`` mean "how much structure", and it holds for any complex ``A`` because
the null model carries the same total weight the graph does.

Nothing here upgrades modularity's epistemic status: it remains a proposal
generator whose reads may never veto a certified fiber.
"""

import cmath
import math
import unittest

import tessera

obs = tessera.observables
PM = obs.PersistentModularity
STRATEGY = obs.DiscoveryStrategy
OBJECTIVE = obs.ModularityObjective
WEIGHT = PM.WeightMap

MACHINE = 1e-12  # machine-precision claims on exact closed forms

SPACELIKE_UNIT = 1.0 + 0j   # l^2 = +1: arg(l^2) = 0
TIMELIKE_UNIT = 1j          # l^2 = -1: arg(l^2) = pi
LIGHTLIKE_UNIT = complex(math.cos(math.pi / 4), math.sin(math.pi / 4))
MIXED_LENGTH = complex(1.0, 0.5)  # a generic argument

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
    return [(a, b) for i, a in enumerate(vertices) for b in vertices[i + 1:]]


def causal_k6(intra=SPACELIKE_UNIT, inter=TIMELIKE_UNIT):
    """K6 with EVERY edge of magnitude one, so the blind map sees a single
    homogeneous complete graph and causal character is the only thing telling
    the two triples apart."""
    st = _from_simplices(6, _complete_pairs(list(TRIPLE_A + TRIPLE_B)))
    for a, b in (_complete_pairs(list(TRIPLE_A)) +
                 _complete_pairs(list(TRIPLE_B))):
        _edge(st, a, b).setLength(intra)
    for a in TRIPLE_A:
        for b in TRIPLE_B:
            _edge(st, a, b).setLength(inter)
    return st


def anti_community_k6():
    """The MIRROR of the matter fixture: each triple is bound internally by
    TIMELIKE edges and the cross edges are spacelike.  A community held
    together by dissimilarity -- the thing that must be found, not avoided."""
    return causal_k6(intra=TIMELIKE_UNIT, inter=SPACELIKE_UNIT)


def lightlike_cohesion_k6():
    """Each triple bound internally by LIGHTLIKE edges: zero Lorentzian
    interval, so the cohesion is neither similarity nor dissimilarity and the
    argument should say so."""
    return causal_k6(intra=LIGHTLIKE_UNIT, inter=TIMELIKE_UNIT)


def generic_argument_k6(seed=20260825):
    """Every edge argument generic -- no edge within the causal tolerance of
    0, +-pi/2 or pi.  This is what a random initialization actually produces,
    and it must simply score."""
    state = seed

    def nxt():
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        return ((state >> 11) & ((1 << 53) - 1)) / float(1 << 53)

    st = _from_simplices(6, _complete_pairs(list(TRIPLE_A + TRIPLE_B)))
    for a, b in _complete_pairs(list(range(6))):
        # arg(l) in (0.17, 1.37) rad, so arg(l^2) = 2 arg(l) is generic too:
        # it clears 0, pi/2 and pi by more than the causal tolerance.
        _edge(st, a, b).setLength(cmath.rect(1.0, 0.17 + 1.2 * nxt()))
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


def config(**overrides):
    cfg = obs.PersistentModularityConfig()
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def spectral_config(**overrides):
    return config(strategy=STRATEGY.LeadingEigenvector, **overrides)


def partition_of(slice_):
    return tuple(sorted(tuple(sorted(c.support)) for c in slice_.components))


def labels_for(graph, groups):
    position = {cell: i for i, cell in enumerate(graph.cellIds())}
    labels = [0] * graph.nCells()
    for index, group in enumerate(groups):
        for cell in group:
            labels[position[cell]] = index
    return labels


def arg_over_pi(z):
    return cmath.phase(z) / math.pi


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


def reference_real_q(edges, labels_by_cell, gamma):
    """Q_gamma from the definition, in the incumbent's exact operation order:
    per-community ``lin/2m - gamma (S_c/2m)^2``, communities combined in
    ascending label order through a Kahan accumulator."""
    strength, internal = {}, {}
    for a, b, w in edges:
        strength[a] = strength.get(a, 0.0) + w
        strength[b] = strength.get(b, 0.0) + w
        if labels_by_cell[a] == labels_by_cell[b]:
            key = labels_by_cell[a]
            internal[key] = internal.get(key, 0.0) + 2.0 * w
    two_m = 0.0
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
    """The causal map is a third map; the incumbent two are unchanged."""

    def test_the_blind_maps_are_still_offered(self):
        self.assertIsNotNone(WEIGHT.Unit)
        self.assertIsNotNone(WEIGHT.ExpNegAbsLength)

    def test_the_default_map_is_the_incumbent(self):
        st = causal_k6()
        default = PM.fromSpacetime(st)
        blind = PM.fromSpacetime(st, WEIGHT.ExpNegAbsLength)
        self.assertEqual(default.totalWeight2(), blind.totalWeight2())
        self.assertFalse(default.isComplex())

    def test_the_branch_is_decided_by_the_graph_not_a_flag(self):
        self.assertFalse(hasattr(PM, "setComplex"))
        self.assertFalse(clique_chain([4, 4]).isComplex())
        self.assertTrue(
            PM.fromComplexWeightedEdges([0], [1], [1 + 1j]).isComplex())

    def test_a_complex_edge_list_is_accepted_directly(self):
        graph = PM.fromComplexWeightedEdges([0, 1], [1, 2], [1 + 0j, 0 + 1j])
        self.assertEqual(graph.nEdges(), 2)
        self.assertTrue(graph.isComplex())


class CausalDiscriminationTest(unittest.TestCase):
    """The ticket's acceptance criterion, demonstrated rather than assumed."""

    def _two_vertices(self, length):
        st = _from_simplices(2, [(0, 1)])
        _edge(st, 0, 1).setLength(length)
        return st

    def test_equal_magnitude_edges_are_blind_map_identical(self):
        """The defect itself: the incumbent map cannot tell them apart."""
        self.assertEqual(abs(SPACELIKE_UNIT), abs(TIMELIKE_UNIT))
        spacelike = PM.fromSpacetime(self._two_vertices(SPACELIKE_UNIT),
                                     WEIGHT.ExpNegAbsLength)
        timelike = PM.fromSpacetime(self._two_vertices(TIMELIKE_UNIT),
                                    WEIGHT.ExpNegAbsLength)
        self.assertEqual(spacelike.totalWeight2(), timelike.totalWeight2())
        self.assertEqual(spacelike.totalWeightSum(), timelike.totalWeightSum())

    def test_equal_magnitude_edges_give_different_operators(self):
        """The fix: same magnitude, different causal character, different
        operator -- and the three definite characters land on three different
        points of the same circle, so nothing is conflated."""
        magnitude = 2.0 * math.exp(-1.0)
        expected = {
            SPACELIKE_UNIT: complex(magnitude, 0.0),
            TIMELIKE_UNIT: complex(-magnitude, 0.0),
            LIGHTLIKE_UNIT: complex(0.0, magnitude),
        }
        seen = []
        for length, want in expected.items():
            graph = PM.fromSpacetime(self._two_vertices(length),
                                     WEIGHT.CausalPhaseExpNegAbsLength)
            got = graph.totalWeightSum()
            self.assertAlmostEqual(got.real, want.real, delta=MACHINE)
            self.assertAlmostEqual(got.imag, want.imag, delta=MACHINE)
            # the MAGNITUDE is the same in all three: only the argument moved
            self.assertAlmostEqual(graph.totalWeight2(), magnitude,
                                   delta=MACHINE)
            seen.append(complex(round(got.real, 9), round(got.imag, 9)))
        self.assertEqual(len(set(seen)), 3)

    def test_causal_structure_alone_changes_the_communities(self):
        """A complex where causal character is the ONLY thing distinguishing
        the blocks: every edge has magnitude one, so the blind map sees a
        homogeneous complete graph and finds nothing to split."""
        st = causal_k6()
        blind = PM.fromSpacetime(st, WEIGHT.ExpNegAbsLength)
        causal = PM.fromSpacetime(st, WEIGHT.CausalPhaseExpNegAbsLength)

        blind_slice = blind.discover(1.0, config())
        causal_slice = causal.discover(1.0, config())

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
        causal = PM.fromSpacetime(st, WEIGHT.CausalPhaseExpNegAbsLength)
        groups = (TRIPLE_A, TRIPLE_B)

        self.assertAlmostEqual(
            blind.modularityGamma(labels_for(blind, groups), 1.0).real, -0.1,
            delta=MACHINE)
        self.assertAlmostEqual(
            causal.modularityGamma(labels_for(causal, groups), 1.0).real, 0.5,
            delta=MACHINE)

    def test_the_closed_form_is_independent_of_the_common_magnitude(self):
        st = causal_k6()
        for e in st.getEdgeList().toVector():
            e.setLength(e.getLength() * 3.0)
        causal = PM.fromSpacetime(st, WEIGHT.CausalPhaseExpNegAbsLength)
        self.assertAlmostEqual(
            causal.modularityGamma(labels_for(causal, (TRIPLE_A, TRIPLE_B)),
                                   1.0).real,
            0.5, delta=MACHINE)


class AntiCommunityTest(unittest.TestCase):
    """A community bound by DISSIMILARITY is a target, not a failure mode.
    It must be findable, and distinguishable from an ordinary community rather
    than merely scoring differently."""

    def _q(self, build, groups=(TRIPLE_A, TRIPLE_B)):
        graph = PM.fromSpacetime(build(), WEIGHT.CausalPhaseExpNegAbsLength)
        return graph.modularityGamma(labels_for(graph, groups), 1.0)

    def test_the_anti_community_scores_the_negative_of_the_community(self):
        """Mirror fixtures: swap spacelike for timelike everywhere and the
        score reflects through zero, exactly."""
        self.assertAlmostEqual(self._q(causal_k6).real, 0.5, delta=MACHINE)
        self.assertAlmostEqual(self._q(anti_community_k6).real, -0.5,
                               delta=MACHINE)

    def test_the_argument_distinguishes_community_from_anti_community(self):
        """|Q| alone CONFLATES them -- both are 1/2.  The argument is what
        separates them, and it is carried rather than derived away."""
        q_matter = self._q(causal_k6)
        q_anti = self._q(anti_community_k6)
        self.assertAlmostEqual(abs(q_matter), abs(q_anti), delta=MACHINE)
        self.assertAlmostEqual(abs(arg_over_pi(q_matter)), 0.0, delta=1e-9)
        self.assertAlmostEqual(abs(arg_over_pi(q_anti)), 1.0, delta=1e-9)

    def test_lightlike_cohesion_has_its_own_argument(self):
        """Neither similarity nor dissimilarity: a zero Lorentzian interval
        puts the cohesion off the real axis entirely, so it is conflated with
        neither of the other two."""
        q = self._q(lightlike_cohesion_k6)
        self.assertGreater(abs(q.imag), 0.1)
        self.assertNotAlmostEqual(abs(arg_over_pi(q)), 0.0, delta=0.05)
        self.assertNotAlmostEqual(abs(arg_over_pi(q)), 1.0, delta=0.05)

    def test_the_magnitude_objective_finds_the_anti_community(self):
        """Maximizing Q passes an anti-community over -- the one-community
        partition scores 0, which beats -1/2.  Maximizing |Q| finds it."""
        graph = PM.fromSpacetime(anti_community_k6(),
                                 WEIGHT.CausalPhaseExpNegAbsLength)
        found = graph.discover(1.0, config(objective=OBJECTIVE.Magnitude))
        self.assertEqual(partition_of(found),
                         (tuple(TRIPLE_A), tuple(TRIPLE_B)))
        self.assertAlmostEqual(found.q.real, -0.5, delta=MACHINE)
        self.assertAlmostEqual(abs(arg_over_pi(found.q)), 1.0, delta=1e-9)

    def test_the_score_objective_passes_the_anti_community_over(self):
        """Both readings exist and they differ -- which is why the objective
        is a choice that gets reported, not an assumption."""
        graph = PM.fromSpacetime(anti_community_k6(),
                                 WEIGHT.CausalPhaseExpNegAbsLength)
        by_score = graph.discover(1.0, config(objective=OBJECTIVE.Score))
        by_magnitude = graph.discover(
            1.0, config(objective=OBJECTIVE.Magnitude))
        self.assertNotEqual(partition_of(by_score),
                            partition_of(by_magnitude))
        self.assertGreaterEqual(by_score.q.real, by_magnitude.q.real)

    def test_the_default_objective_is_the_incumbent(self):
        graph = clique_chain([4, 5, 6])
        self.assertEqual(obs.PersistentModularityConfig().objective,
                         OBJECTIVE.Score)
        slice_ = graph.discover(1.0, config())
        self.assertEqual(slice_.objective, OBJECTIVE.Score)
        self.assertEqual(len(slice_.components), 3)

    def test_a_complex_graph_reports_the_objective_it_actually_used(self):
        """Score is not an ordering on a complex Q, so it is not silently
        honoured -- the slice says which functional ran."""
        graph = PM.fromSpacetime(lightlike_cohesion_k6(),
                                 WEIGHT.CausalPhaseExpNegAbsLength)
        slice_ = graph.discover(1.0, config(objective=OBJECTIVE.Score))
        self.assertTrue(graph.isComplex())
        self.assertEqual(slice_.objective, OBJECTIVE.Magnitude)
        self.assertAlmostEqual(slice_.objectiveValue, abs(slice_.q),
                               delta=MACHINE)


class GenericArgumentTest(unittest.TestCase):
    """The case a random initialization actually produces.  Under a uniformly
    drawn argument almost every edge is MIXED, and a complex weight carries it
    without classifying it -- so there is nothing to refuse."""

    def test_the_fixture_really_has_no_definite_edge(self):
        read = PM.causalWeightAvailability(generic_argument_k6())
        self.assertEqual(read.spacelike, 0)
        self.assertEqual(read.timelike, 0)
        self.assertEqual(read.lightlike, 0)
        self.assertEqual(read.mixed, 15)

    def test_a_wholly_generic_complex_is_available(self):
        read = PM.causalWeightAvailability(generic_argument_k6())
        self.assertTrue(read.available)
        self.assertEqual(read.reason, "")

    def test_a_wholly_generic_complex_scores_and_discovers(self):
        graph = PM.fromSpacetime(generic_argument_k6(),
                                 WEIGHT.CausalPhaseExpNegAbsLength)
        self.assertEqual(graph.nEdges(), 15)
        self.assertTrue(graph.isComplex())
        for cfg in (config(), spectral_config()):
            slice_ = graph.discover(1.0, cfg)
            self.assertTrue(math.isfinite(slice_.objectiveValue))
            self.assertTrue(math.isfinite(abs(slice_.q)))
            self.assertGreaterEqual(len(slice_.components), 1)

    def test_the_one_community_anchor_holds_whatever_the_arguments(self):
        """Q_1(one community) = 0 for ANY complex A.  That anchor is what
        makes |Q| mean 'how much structure' rather than an offset."""
        for build in (causal_k6, anti_community_k6, lightlike_cohesion_k6,
                      generic_argument_k6):
            graph = PM.fromSpacetime(build(),
                                     WEIGHT.CausalPhaseExpNegAbsLength)
            whole = labels_for(graph, [TRIPLE_A + TRIPLE_B])
            self.assertAlmostEqual(abs(graph.modularityGamma(whole, 1.0)), 0.0,
                                   delta=MACHINE)

    def test_a_single_mixed_edge_no_longer_refuses(self):
        st = causal_k6()
        _edge(st, 0, 1).setLength(MIXED_LENGTH)
        read = PM.causalWeightAvailability(st)
        self.assertEqual(read.mixed, 1)
        self.assertTrue(read.available)
        graph = PM.fromSpacetime(st, WEIGHT.CausalPhaseExpNegAbsLength)
        self.assertEqual(graph.nEdges(), 15)


class ExactReductionTest(unittest.TestCase):
    """A nonnegative real graph reproduces the incumbent BIT-IDENTICALLY.
    Every assertion in this class uses ``==``."""

    def test_an_all_spacelike_complex_scores_identically_under_both_maps(self):
        """The causal map differs from the blind one only by an argument that
        is everywhere zero, so not a bit of the score may move."""
        st = causal_k6(intra=SPACELIKE_UNIT, inter=SPACELIKE_UNIT)
        blind = PM.fromSpacetime(st, WEIGHT.ExpNegAbsLength)
        causal = PM.fromSpacetime(st, WEIGHT.CausalPhaseExpNegAbsLength)
        self.assertFalse(causal.isComplex())
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
        st = causal_k6(intra=SPACELIKE_UNIT, inter=SPACELIKE_UNIT)
        blind = PM.fromSpacetime(st, WEIGHT.ExpNegAbsLength)
        causal = PM.fromSpacetime(st, WEIGHT.CausalPhaseExpNegAbsLength)
        for cfg in (config(), spectral_config()):
            blind_slice = blind.discover(1.0, cfg)
            causal_slice = causal.discover(1.0, cfg)
            self.assertEqual(partition_of(blind_slice),
                             partition_of(causal_slice))
            self.assertEqual(blind_slice.q, causal_slice.q)
            self.assertEqual(blind_slice.objectiveValue,
                             causal_slice.objectiveValue)

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
        self.assertFalse(graph.isComplex())
        for groups in ((TRIPLE_A, TRIPLE_B), (TRIPLE_A + TRIPLE_B,)):
            by_cell = {}
            for index, group in enumerate(groups):
                for cell in group:
                    by_cell[cell] = index
            got = graph.modularityGamma(labels_for(graph, groups), gamma)
            self.assertEqual(got.imag, 0.0)
            self.assertEqual(got.real,
                             reference_real_q(edges, by_cell, gamma))

    def test_a_real_graph_has_an_exactly_zero_imaginary_part(self):
        """Zero, not small: a real graph never takes the complex path."""
        graph = clique_chain([4, 5, 6])
        slice_ = graph.discover(1.0, config())
        self.assertEqual(slice_.q.imag, 0.0)
        self.assertEqual(slice_.qIncremental.imag, 0.0)
        for comp in slice_.components:
            self.assertEqual(comp.modularityContribution.imag, 0.0)

    def test_the_incumbent_fixture_is_untouched(self):
        graph = clique_chain([4, 5, 6])
        self.assertFalse(graph.isComplex())
        self.assertFalse(graph.isSigned())
        self.assertEqual(graph.totalWeightSum(), graph.totalWeight2())
        slice_ = graph.discover(1.0, config())
        self.assertEqual(len(slice_.components), 3)


class RefusalTest(unittest.TestCase):
    """Refusal is kept for genuine ABSENCES only.  An indefinite argument is
    not an absence: the complex weight carries it as it stands."""

    def test_a_degenerate_edge_has_no_argument_to_carry(self):
        st = causal_k6()
        _edge(st, 0, 1).setLength(0.0 + 0j)
        read = PM.causalWeightAvailability(st)
        self.assertEqual(read.degenerate, 1)
        self.assertFalse(read.available)
        self.assertEqual(read.reason, "degenerate-edge-length")

    def test_a_degenerate_edge_raises_by_name(self):
        st = causal_k6()
        _edge(st, 0, 1).setLength(0.0 + 0j)
        with self.assertRaises(ValueError) as caught:
            PM.fromSpacetime(st, WEIGHT.CausalPhaseExpNegAbsLength)
        self.assertIn("degenerate-edge-length", str(caught.exception))
        # The BLIND map still reads it: the refusal belongs to the causal
        # map, not to the complex.
        self.assertGreater(
            PM.fromSpacetime(st, WEIGHT.ExpNegAbsLength).nEdges(), 0)

    def test_an_available_census_names_no_reason(self):
        read = PM.causalWeightAvailability(causal_k6())
        self.assertTrue(read.available)
        self.assertEqual(read.reason, "")
        self.assertEqual(read.spacelike, 6)   # three inside each triple
        self.assertEqual(read.timelike, 9)    # the three-by-three between
        self.assertEqual(read.mixed, 0)

    def test_the_census_totals_the_edges(self):
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
            with self.assertRaises(ValueError):
                PM.fromComplexWeightedEdges([0], [1], [complex(1.0, bad)])

    def test_a_vanishing_total_weight_is_refused_by_name(self):
        """SA = 0 leaves the configuration null model with no weight to
        redistribute, so Q is undefined -- a property of the GRAPH, not of the
        partition, and named rather than silently treated as a zero null."""
        graph = PM.fromWeightedEdges([0, 2], [1, 3], [1.0, -1.0])
        self.assertEqual(graph.totalWeightSum(), 0.0)
        self.assertEqual(graph.totalWeight2(), 4.0)
        with self.assertRaises(ValueError) as caught:
            graph.modularityGamma([0, 0, 1, 1], 1.0)
        self.assertIn("vanishes", str(caught.exception))


class DegenerateCaseTest(unittest.TestCase):
    """The cases the ticket names explicitly."""

    def test_the_scale_is_the_absolute_total_not_the_signed_sum(self):
        """T cannot vanish while any edge exists, which is what the signed
        sum could do -- that is why it, and not 2m, is what Q divides by."""
        graph = PM.fromWeightedEdges([0, 2], [1, 3], [1.0, -1.0])
        self.assertEqual(graph.totalWeight2(), 4.0)
        self.assertEqual(graph.totalWeightSum(), 0.0)

    def test_a_wholly_timelike_complex_scores(self):
        st = causal_k6(intra=TIMELIKE_UNIT, inter=TIMELIKE_UNIT)
        graph = PM.fromSpacetime(st, WEIGHT.CausalPhaseExpNegAbsLength)
        self.assertTrue(graph.isSigned())
        self.assertLess(graph.totalWeightSum().real, 0.0)
        score = graph.modularityGamma(
            labels_for(graph, (TRIPLE_A, TRIPLE_B)), 1.0)
        self.assertTrue(math.isfinite(abs(score)))

    def test_cancelling_parallel_edges_drop_out(self):
        graph = PM.fromWeightedEdges([0, 0, 1], [1, 1, 2], [1.0, -1.0, 1.0])
        self.assertEqual(graph.nEdges(), 1)
        self.assertFalse(graph.isSigned())

    def test_cancelling_complex_parallel_edges_drop_out(self):
        graph = PM.fromComplexWeightedEdges(
            [0, 0, 1], [1, 1, 2], [1 + 1j, -1 - 1j, 1 + 0j])
        self.assertEqual(graph.nEdges(), 1)

    def test_a_disconnected_complex_graph_scores_and_splits(self):
        graph = PM.fromComplexWeightedEdges(
            [0, 1, 0, 3, 4, 3], [1, 2, 2, 4, 5, 5],
            [1 + 0j, 1 + 0j, 0.3j, 1 + 0j, 1 + 0j, 0.3j])
        self.assertTrue(graph.isComplex())
        slice_ = graph.discover(1.0, config())
        self.assertTrue(math.isfinite(abs(slice_.q)))
        self.assertGreaterEqual(len(slice_.components), 2)

    def test_an_empty_edge_list_scores_zero(self):
        graph = PM.fromWeightedEdges([], [], [], [7, 8, 9])
        self.assertEqual(graph.nEdges(), 0)
        self.assertEqual(graph.modularityGamma([0, 0, 0], 1.0), 0.0)

    def test_a_complex_with_no_edges_is_refused_by_name(self):
        st = _from_simplices(2, [])
        read = PM.causalWeightAvailability(st)
        self.assertFalse(read.available)
        self.assertEqual(read.reason, "no-scorable-edges")


class BothStrategiesTest(unittest.TestCase):
    """Both discovery strategies keep working on the complex operator, and
    they interact with it differently -- so both are exercised on it."""

    def _causal_graph(self):
        return PM.fromSpacetime(causal_k6(),
                                WEIGHT.CausalPhaseExpNegAbsLength)

    def test_multilevel_aggregation_recovers_the_causal_blocks(self):
        slice_ = self._causal_graph().discover(1.0, config())
        self.assertEqual(slice_.strategy, STRATEGY.MultilevelAggregation)
        self.assertEqual(partition_of(slice_),
                         (tuple(TRIPLE_A), tuple(TRIPLE_B)))

    def test_leading_eigenvector_recovers_the_causal_blocks(self):
        slice_ = self._causal_graph().discover(1.0, spectral_config())
        self.assertEqual(slice_.strategy, STRATEGY.LeadingEigenvector)
        self.assertEqual(partition_of(slice_),
                         (tuple(TRIPLE_A), tuple(TRIPLE_B)))

    def test_both_strategies_agree_on_the_score(self):
        graph = self._causal_graph()
        incumbent = graph.discover(1.0, config())
        spectral = graph.discover(1.0, spectral_config())
        self.assertAlmostEqual(incumbent.q.real, spectral.q.real,
                               delta=MACHINE)
        self.assertAlmostEqual(incumbent.q.real, 0.5, delta=MACHINE)

    def test_both_strategies_find_the_anti_community(self):
        """The spectral search needs a candidate the most POSITIVE eigenvector
        cannot supply, so the most negative one is proposed too -- and, like
        every candidate, accepted only on an exact improvement."""
        graph = PM.fromSpacetime(anti_community_k6(),
                                 WEIGHT.CausalPhaseExpNegAbsLength)
        for cfg in (config(objective=OBJECTIVE.Magnitude),
                    spectral_config(objective=OBJECTIVE.Magnitude)):
            slice_ = graph.discover(1.0, cfg)
            self.assertEqual(partition_of(slice_),
                             (tuple(TRIPLE_A), tuple(TRIPLE_B)))
            self.assertAlmostEqual(slice_.q.real, -0.5, delta=MACHINE)

    def test_the_ledger_matches_the_cold_recompute_on_a_complex_graph(self):
        """The multilevel search accumulates the complex delta-Q closed form
        move by move; ``q`` recomputes the same partition from scratch.  A
        wrong delta-Q would agree with neither, so this is the direct check on

            dQ(v: a->b) = [2 (w_vb - w_va)
                           - 2 gamma k_v (k_v + S_b - S_a)/SA] / T

        and on the degrees being inherited by summation across aggregation
        levels rather than recomputed from the coarse adjacency.
        """
        for build in (causal_k6, lightlike_cohesion_k6, generic_argument_k6):
            graph = PM.fromSpacetime(build(),
                                     WEIGHT.CausalPhaseExpNegAbsLength)
            for gamma in (0.5, 1.0, 2.0):
                slice_ = graph.discover(gamma, config())
                self.assertAlmostEqual(slice_.qIncremental.real,
                                       slice_.q.real, delta=MACHINE)
                self.assertAlmostEqual(slice_.qIncremental.imag,
                                       slice_.q.imag, delta=MACHINE)

    def test_the_unsigned_ledger_is_unaffected(self):
        graph = clique_chain([4, 5, 6])
        slice_ = graph.discover(1.0, config())
        self.assertAlmostEqual(slice_.qIncremental.real, slice_.q.real,
                               delta=MACHINE)

    def test_the_power_iteration_path_agrees_with_the_exact_one(self):
        """The shifted power iteration finds the most POSITIVE eigenvalue
        only while its Gershgorin shift really bounds the spectral radius.
        Once weights carry phase the adjacency part of that bound has to be
        the absolute row sum ``sum_j |A_ij|``: the group degree is a complex
        sum and bounds nothing.  (On a nonnegative graph the two coincide,
        which is why the incumbent's bound was correct for its graphs.)"""
        graph = self._causal_graph()
        exact = graph.discover(
            1.0, spectral_config(denseEigenSolveMaxGroup=1024))
        iterative = graph.discover(
            1.0, spectral_config(denseEigenSolveMaxGroup=0))
        self.assertEqual(partition_of(exact), partition_of(iterative))

    def test_the_spectral_search_still_carries_no_seed(self):
        graph = self._causal_graph()
        first = graph.discover(1.0, spectral_config(baseSeed=1))
        second = graph.discover(1.0, spectral_config(baseSeed=99999))
        self.assertEqual(partition_of(first), partition_of(second))


class PerComponentReadTest(unittest.TestCase):
    """What a component REPORTS has to be the score it actually contributes,
    on the complex operator as much as on the real one."""

    def test_the_contributions_sum_to_the_score(self):
        for graph in (PM.fromSpacetime(causal_k6(),
                                       WEIGHT.CausalPhaseExpNegAbsLength),
                      PM.fromSpacetime(generic_argument_k6(),
                                       WEIGHT.CausalPhaseExpNegAbsLength),
                      clique_chain([4, 5, 6])):
            for cfg in (config(), spectral_config()):
                slice_ = graph.discover(1.0, cfg)
                total = sum((c.modularityContribution
                             for c in slice_.components), 0j)
                self.assertAlmostEqual(total.real, slice_.q.real,
                                       delta=MACHINE)
                self.assertAlmostEqual(total.imag, slice_.q.imag,
                                       delta=MACHINE)

    def test_conductance_is_unmeasured_off_the_nonnegative_regime(self):
        """A community's strength is then a complex sum, so there is no volume
        for the cut to be a fraction of.  Reported NaN, never zero -- zero
        would read as a perfectly isolated community."""
        graph = PM.fromSpacetime(causal_k6(),
                                 WEIGHT.CausalPhaseExpNegAbsLength)
        slice_ = graph.discover(1.0, config())
        for comp in slice_.components:
            self.assertTrue(math.isnan(comp.conductance))

    def test_conductance_is_still_measured_on_a_nonnegative_graph(self):
        slice_ = clique_chain([4, 5, 6]).discover(1.0, config())
        for comp in slice_.components:
            self.assertFalse(math.isnan(comp.conductance))
            self.assertGreaterEqual(comp.conductance, 0.0)


if __name__ == "__main__":
    unittest.main()
