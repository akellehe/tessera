# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Acceptance tests for label-free persistent modular component discovery
(:class:`tessera.PersistentModularity` + the :class:`tessera.ModularityOptimizer`
extension), ticket #765 / design spec section 8.

Covers every ticket acceptance bullet:

* planted disconnected/modular fixtures are recovered;
* homogeneous rings and a Fortunato-Barthelemy resolution-limit fixture do
  not manufacture a persistent particle scale merely from modularity;
* incremental delta-Q sums equal a full exact modularity recomputation;
* random vertex relabeling returns an isomorphic hierarchy and identical
  scores;
* fixed-partition behavior (and the legacy Newman-Girvan reads) remain
  available and unchanged;
* local changes invalidate only affected component ancestry.

Exactness bars: incremental-vs-cold and cross-implementation identities are
held to double round-off (~1e-15; 1e-14 where an extra summation-order
degree of freedom exists and is documented).  Unit-weight fixtures assert
bitwise score equality under relabeling (all sums are exact in double).
"""
import math
import random
import unittest

import tessera

PM = tessera.PersistentModularity


# ---------------------------------------------------------------------------
# fixture builders
# ---------------------------------------------------------------------------


def _clique_edges(vertices, src, tgt):
    for i in range(len(vertices)):
        for j in range(i + 1, len(vertices)):
            src.append(vertices[i])
            tgt.append(vertices[j])


def _two_disconnected_k6():
    src, tgt = [], []
    _clique_edges(list(range(6)), src, tgt)
    _clique_edges(list(range(10, 16)), src, tgt)
    return src, tgt


def _planted_modular_two_k8():
    """Two K8 cliques joined by a single edge."""
    src, tgt = [], []
    _clique_edges(list(range(8)), src, tgt)
    _clique_edges(list(range(20, 28)), src, tgt)
    src.append(0)
    tgt.append(20)
    return src, tgt


def _fb_ring(n_cliques=40, k=5):
    """Fortunato-Barthelemy resolution-limit fixture: a ring of K_k cliques
    joined by single edges.  With n_cliques > sqrt(2m) the gamma = 1 optimum
    merges adjacent cliques (the resolution limit)."""
    src, tgt = [], []
    for c in range(n_cliques):
        base = c * k
        _clique_edges(list(range(base, base + k)), src, tgt)
    for c in range(n_cliques):
        src.append(c * k)
        tgt.append(((c + 1) % n_cliques) * k + 1)
    return src, tgt


def _ring(n=60):
    return list(range(n)), [(i + 1) % n for i in range(n)]


def _fb_partition_labels(g, n_cliques=40, k=5, merge_pairs=False):
    """Fixed-partition labels (indexed like ``g.cellIds()``) for the FB ring:
    one community per clique, or adjacent cliques merged in pairs."""
    pos = {cid: i for i, cid in enumerate(g.cellIds())}
    labels = [0] * g.nCells()
    for c in range(n_cliques):
        community = c // 2 if merge_pairs else c
        for v in range(c * k, c * k + k):
            labels[pos[v]] = community
    return labels


def _relabeled(src, tgt, weights, seed, id_pool_base=10_000):
    """Random vertex relabeling + random edge input reorder (labels and
    ordering are both conventions the discovery must not depend on)."""
    rng = random.Random(seed)
    ids = sorted(set(src) | set(tgt))
    perm = dict(zip(ids, rng.sample(range(id_pool_base, id_pool_base + 10 * len(ids)),
                                    len(ids))))
    order = list(range(len(src)))
    rng.shuffle(order)
    return ([perm[src[i]] for i in order],
            [perm[tgt[i]] for i in order],
            [weights[i] for i in order],
            perm)


def _cfg(resolutions=(1.0,), restarts=4, base_seed=0, overlap=0.5):
    cfg = tessera.PersistentModularityConfig()
    cfg.resolutions = list(resolutions)
    cfg.restarts = restarts
    cfg.baseSeed = base_seed
    cfg.overlapThreshold = overlap
    return cfg


def _make_spacetime(n_simplices=120):
    sig = tessera.Signature(4, tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    st = tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0,
                           tessera.PREFERRED, tessera.Toroid())
    st.build(n_simplices)
    return st


# ---------------------------------------------------------------------------
# planted fixtures are recovered
# ---------------------------------------------------------------------------


class TestPlantedRecovery(unittest.TestCase):

    def test_disconnected_cliques_recovered_exactly(self):
        src, tgt = _two_disconnected_k6()
        g = PM.fromWeightedEdges(src, tgt, [1.0] * len(src))
        s = g.discover(1.0, _cfg())
        supports = sorted(tuple(c.support) for c in s.components)
        self.assertEqual(supports,
                         [tuple(range(6)), tuple(range(10, 16))])
        # Analytic exact Q: two communities, each Sigma_in = 30, S = 30,
        # 2m = 60  ->  Q = 2 * (30/60 - (30/60)^2) = 1/2 exactly.
        self.assertEqual(s.q, 0.5)
        # Deterministic restarts all land on the planted optimum.
        self.assertEqual(s.restartSpread, 0.0)
        self.assertEqual(len(s.restarts), 4)

    def test_disconnected_unequal_cliques_recovered(self):
        src, tgt = [], []
        _clique_edges(list(range(5)), src, tgt)
        _clique_edges(list(range(100, 106)), src, tgt)
        _clique_edges(list(range(200, 207)), src, tgt)
        g = PM.fromWeightedEdges(src, tgt, [1.0] * len(src))
        s = g.discover(1.0, _cfg())
        supports = sorted(tuple(c.support) for c in s.components)
        self.assertEqual(supports, [tuple(range(5)),
                                    tuple(range(100, 106)),
                                    tuple(range(200, 207))])

    def test_planted_modular_recovered_with_analytic_q(self):
        src, tgt = _planted_modular_two_k8()
        g = PM.fromWeightedEdges(src, tgt, [1.0] * len(src))
        s = g.discover(1.0, _cfg())
        supports = sorted(tuple(c.support) for c in s.components)
        self.assertEqual(supports, [tuple(range(8)), tuple(range(20, 28))])
        # Analytic: m = 57, communities symmetric, Sigma_in = 56 each,
        # S = 57 each: Q = 2 * (56/114 - (57/114)^2).
        q_expected = 2.0 * (56.0 / 114.0 - (57.0 / 114.0) ** 2)
        self.assertLessEqual(abs(s.q - q_expected), 1e-15)

    def test_weighted_planted_blocks_recovered(self):
        rng = random.Random(3)
        src, tgt, w = [], [], []
        blocks = [list(range(b * 12, b * 12 + 12)) for b in range(6)]
        for block in blocks:
            for i in range(len(block)):
                for j in range(i + 1, len(block)):
                    if rng.random() < 0.7:
                        src.append(block[i])
                        tgt.append(block[j])
                        w.append(0.5 + rng.random())
        for _ in range(40):
            a, b = rng.sample(range(6), 2)
            src.append(rng.choice(blocks[a]))
            tgt.append(rng.choice(blocks[b]))
            w.append(0.1 * rng.random())
        g = PM.fromWeightedEdges(src, tgt, w)
        s = g.discover(1.0, _cfg())
        self.assertEqual(len(s.components), 6)
        recovered = sorted(tuple(c.support) for c in s.components)
        self.assertEqual(recovered, sorted(tuple(b) for b in blocks))


# ---------------------------------------------------------------------------
# no manufactured persistent scale: homogeneous ring + Fortunato-Barthelemy
# ---------------------------------------------------------------------------


class TestNoManufacturedScale(unittest.TestCase):

    SCAN = (0.5, 1.0, 2.0, 4.0)

    def test_ring_partition_is_resolution_dependent(self):
        src, tgt = _ring(60)
        g = PM.fromWeightedEdges(src, tgt, [1.0] * len(src))
        report = g.scanResolutions(_cfg(self.SCAN))
        counts = [len(s.components) for s in report.slices]
        # The arc scale tracks gamma: strictly more communities at the top
        # of the scan than at the bottom -> no intrinsic scale.
        self.assertLess(counts[0], counts[-1])
        # No track survives the whole scan with near-perfect support
        # overlap: a homogeneous ring has no persistent component.
        full_range_stable = [
            t for t in report.tracks
            if t.firstSlice == 0 and t.lastSlice == len(report.slices) - 1
            and t.minAdjacentOverlap >= 0.9
        ]
        self.assertEqual(full_range_stable, [])

    def test_ring_restart_spread_reported_honestly(self):
        src, tgt = _ring(60)
        g = PM.fromWeightedEdges(src, tgt, [1.0] * len(src))
        s = g.discover(1.0, _cfg())
        self.assertEqual(len(s.restarts), 4)
        qs = [r.objectiveValue for r in s.restarts]
        self.assertEqual(s.restartSpread, max(qs) - min(qs))
        # The ring's degenerate arc placements genuinely disagree across
        # restarts; the spread must be surfaced, not hidden.
        self.assertGreater(s.restartSpread, 0.0)
        # The winner is the best exact restart score.
        self.assertEqual(s.q, max(qs))

    def test_fb_resolution_limit_exact_scores(self):
        """The analytic Fortunato-Barthelemy statement, checked with the
        exact fixed-partition evaluator: at gamma = 1 merged clique pairs
        beat single cliques (the resolution limit); at gamma = 4 singles
        win."""
        src, tgt = _fb_ring()
        g = PM.fromWeightedEdges(src, tgt, [1.0] * len(src))
        singles = _fb_partition_labels(g, merge_pairs=False)
        pairs = _fb_partition_labels(g, merge_pairs=True)
        self.assertGreater(g.modularityGamma(pairs, 1.0).real,
                           g.modularityGamma(singles, 1.0).real)
        self.assertGreater(g.modularityGamma(singles, 4.0).real,
                           g.modularityGamma(pairs, 4.0).real)

    def test_fb_discovery_exhibits_not_hides_the_limit(self):
        src, tgt = _fb_ring()
        g = PM.fromWeightedEdges(src, tgt, [1.0] * len(src))
        singles = _fb_partition_labels(g, merge_pairs=False)
        s1 = g.discover(1.0, _cfg())
        s4 = g.discover(4.0, _cfg())
        # gamma = 1: the resolution limit is real - cliques merge, and the
        # discovered exact score is at least the single-clique score.
        self.assertLess(len(s1.components), 40)
        self.assertGreaterEqual(s1.objectiveValue,
                                g.modularityGamma(singles, 1.0).real)
        # gamma = 4: all 40 planted cliques recovered exactly.
        self.assertEqual(len(s4.components), 40)
        self.assertEqual(sorted(tuple(c.support) for c in s4.components),
                         [tuple(range(c * 5, c * 5 + 5)) for c in range(40)])
        self.assertLessEqual(abs(s4.q - g.modularityGamma(singles, 4.0)),
                             1e-15)

    def test_fb_scan_reports_scale_dependence(self):
        src, tgt = _fb_ring()
        g = PM.fromWeightedEdges(src, tgt, [1.0] * len(src))
        report = g.scanResolutions(_cfg(self.SCAN))
        counts = [len(s.components) for s in report.slices]
        # The partition changes across the scan; the report exposes it.
        self.assertNotEqual(counts[0], counts[-1])
        # No support-stable track spans the entire scan: modularity alone
        # does not hand the recursion one persistent particle scale here.
        full_range_stable = [
            t for t in report.tracks
            if t.firstSlice == 0 and t.lastSlice == len(report.slices) - 1
            and t.minAdjacentOverlap >= 0.9
        ]
        self.assertEqual(full_range_stable, [])
        # Every track carries the unknown downstream status as None (never
        # zero) - the weight-aware certificates belong to later tickets.
        for t in report.tracks:
            self.assertIsNone(t.weightAwareStatus)


# ---------------------------------------------------------------------------
# incremental delta-Q ledger == cold exact recomputation
# ---------------------------------------------------------------------------


class TestIncrementalEqualsCold(unittest.TestCase):

    TOL = 1e-14  # documented double round-off standard (measured ~1e-16)

    def _check(self, g, gammas):
        for gamma in gammas:
            s = g.discover(gamma, _cfg())
            self.assertLessEqual(
                abs(s.q - s.qIncremental), self.TOL,
                f"gamma={gamma}: ledger {s.qIncremental!r} vs cold {s.q!r}")

    def test_unit_weight_fixtures(self):
        for build in (_two_disconnected_k6, _planted_modular_two_k8,
                      _fb_ring, _ring):
            src, tgt = build()
            g = PM.fromWeightedEdges(src, tgt, [1.0] * len(src))
            self._check(g, (0.5, 1.0, 2.0, 4.0))

    def test_weighted_nondyadic_fixture(self):
        rng = random.Random(11)
        src, tgt, w = [], [], []
        for i in range(80):
            for _ in range(4):
                j = rng.randrange(80)
                if i != j:
                    src.append(i)
                    tgt.append(j)
                    w.append(0.05 + rng.random())
        g = PM.fromWeightedEdges(src, tgt, w)
        self._check(g, (0.5, 1.0, 2.0))

    def test_gamma_affinity_exact_identity(self):
        """Q_gamma is affine in gamma: Q_2 = 2 Q_1 - Q_0 exactly (the
        closed-form check that the evaluator implements the stated
        identity)."""
        src, tgt = _fb_ring()
        g = PM.fromWeightedEdges(src, tgt, [1.0] * len(src))
        labels = _fb_partition_labels(g)
        q0 = g.modularityGamma(labels, 0.0)
        q1 = g.modularityGamma(labels, 1.0)
        q2 = g.modularityGamma(labels, 2.0)
        self.assertLessEqual(abs(q2 - (2.0 * q1 - q0)), 1e-15)


# ---------------------------------------------------------------------------
# relabeling / ordering / orientation
# ---------------------------------------------------------------------------


class TestRelabelingInvariance(unittest.TestCase):

    SCAN = (0.5, 1.0, 2.0, 4.0)

    def _scan_pair(self, src, tgt, w, seed):
        g = PM.fromWeightedEdges(src, tgt, w)
        src2, tgt2, w2, perm = _relabeled(src, tgt, w, seed)
        g2 = PM.fromWeightedEdges(src2, tgt2, w2)
        cfg = _cfg(self.SCAN)
        return g.scanResolutions(cfg), g2.scanResolutions(cfg), perm

    def _assert_isomorphic(self, a, b, perm, forced_supports=False):
        """Ticket acceptance: relabeling returns an isomorphic hierarchy and
        identical scores.  On a symmetric fixture the relabeled result is
        the automorphic image (supports rotate by a graph automorphism); on
        a planted fixture with a forced partition the supports must map
        pointwise under the permutation."""
        self.assertEqual(len(a.slices), len(b.slices))
        for sa, sb in zip(a.slices, b.slices):
            # Unit / dyadic weights: every sum is exact in double, so
            # relabeling gives bitwise-identical scores.
            self.assertEqual(sa.q, sb.q)
            self.assertEqual(sa.qIncremental, sb.qIncremental)
            # Isomorphic hierarchy: identical per-level canonical hash
            # multisets (hashes derive from oriented incidence + lineage,
            # never raw vertex numbers) and identical size multisets.
            self.assertEqual(sa.levels, sb.levels)
            for la, lb in zip(sa.hierarchy, sb.hierarchy):
                self.assertEqual(sorted(c.id.canonicalHash() for c in la),
                                 sorted(c.id.canonicalHash() for c in lb))
                self.assertEqual(sorted(len(c.support) for c in la),
                                 sorted(len(c.support) for c in lb))
            if forced_supports:
                mapped = sorted(tuple(sorted(perm[v] for v in c.support))
                                for c in sa.components)
                plain = sorted(tuple(c.support) for c in sb.components)
                self.assertEqual(mapped, plain)
        # Track structure is isomorphic (same lifetimes).
        self.assertEqual(
            sorted((t.firstSlice, t.lastSlice, len(t.members))
                   for t in a.tracks),
            sorted((t.firstSlice, t.lastSlice, len(t.members))
                   for t in b.tracks))

    def test_fb_ring_relabeling(self):
        src, tgt = _fb_ring()
        a, b, perm = self._scan_pair(src, tgt, [1.0] * len(src), seed=7)
        self._assert_isomorphic(a, b, perm)

    def test_homogeneous_ring_relabeling(self):
        src, tgt = _ring(60)
        a, b, perm = self._scan_pair(src, tgt, [1.0] * len(src), seed=13)
        self._assert_isomorphic(a, b, perm)

    def test_planted_dyadic_weighted_relabeling(self):
        # Dyadic weights: relabeling-exact arithmetic without relying on
        # unit weights.  The planted two-community partition is forced, so
        # supports must map pointwise under the permutation.
        rng = random.Random(5)
        src, tgt = _planted_modular_two_k8()
        w = [rng.choice((0.5, 1.0, 1.5, 2.0)) for _ in src]
        a, b, perm = self._scan_pair(src, tgt, w, seed=21)
        self._assert_isomorphic(a, b, perm, forced_supports=True)

    def test_fb_relabeling_forced_at_high_gamma(self):
        # Above the resolution limit the FB partition is forced (single
        # cliques): supports must map pointwise even on this symmetric
        # fixture.
        src, tgt = _fb_ring()
        w = [1.0] * len(src)
        g = PM.fromWeightedEdges(src, tgt, w)
        src2, tgt2, w2, perm = _relabeled(src, tgt, w, seed=7)
        g2 = PM.fromWeightedEdges(src2, tgt2, w2)
        sa = g.discover(4.0, _cfg())
        sb = g2.discover(4.0, _cfg())
        self.assertEqual(sa.q, sb.q)
        mapped = sorted(tuple(sorted(perm[v] for v in c.support))
                        for c in sa.components)
        plain = sorted(tuple(c.support) for c in sb.components)
        self.assertEqual(mapped, plain)

    def test_edge_input_order_only(self):
        # Pure ordering test: same labels, shuffled edge input order.
        src, tgt = _fb_ring()
        w = [1.0] * len(src)
        rng = random.Random(2)
        order = list(range(len(src)))
        rng.shuffle(order)
        g = PM.fromWeightedEdges(src, tgt, w)
        g2 = PM.fromWeightedEdges([src[i] for i in order],
                                  [tgt[i] for i in order],
                                  [w[i] for i in order])
        s, s2 = g.discover(1.0, _cfg()), g2.discover(1.0, _cfg())
        self.assertEqual(s.q, s2.q)
        self.assertEqual([c.id.canonicalHash() for c in s.components],
                         [c.id.canonicalHash() for c in s2.components])
        self.assertEqual([c.support for c in s.components],
                         [c.support for c in s2.components])

    def test_orientation_flip_leaves_scores_and_supports(self):
        # Modularity is undirected: flipping stored edge orientations must
        # not change scores or supports.  (Component identity hashes read
        # the oriented incidence and may legitimately differ.)
        src, tgt = _planted_modular_two_k8()
        rng = random.Random(9)
        src2, tgt2 = list(src), list(tgt)
        for i in range(len(src2)):
            if rng.random() < 0.5:
                src2[i], tgt2[i] = tgt2[i], src2[i]
        g = PM.fromWeightedEdges(src, tgt, [1.0] * len(src))
        g2 = PM.fromWeightedEdges(src2, tgt2, [1.0] * len(src2))
        s, s2 = g.discover(1.0, _cfg()), g2.discover(1.0, _cfg())
        self.assertEqual(s.q, s2.q)
        self.assertEqual(sorted(tuple(c.support) for c in s.components),
                         sorted(tuple(c.support) for c in s2.components))

    def test_determinism_same_config_twice(self):
        src, tgt = _fb_ring()
        g = PM.fromWeightedEdges(src, tgt, [1.0] * len(src))
        cfg = _cfg(self.SCAN)
        a, b = g.scanResolutions(cfg), g.scanResolutions(cfg)
        for sa, sb in zip(a.slices, b.slices):
            self.assertEqual(sa.q, sb.q)
            self.assertEqual(sa.qIncremental, sb.qIncremental)
            self.assertEqual([c.id.canonicalHash() for c in sa.components],
                             [c.id.canonicalHash() for c in sb.components])
            self.assertEqual([c.support for c in sa.components],
                             [c.support for c in sb.components])
            self.assertEqual([r.seed for r in sa.restarts],
                             [r.seed for r in sb.restarts])


# ---------------------------------------------------------------------------
# fixed-partition behavior remains available and consistent
# ---------------------------------------------------------------------------


class TestFixedPartitionContinuity(unittest.TestCase):

    def test_gamma_one_unit_weights_equals_sparsegraph_newman_girvan(self):
        rng = random.Random(5)
        n = 30
        edges = set()
        while len(edges) < 60:
            a, b = rng.sample(range(n), 2)
            edges.add((min(a, b), max(a, b)))
        rows = [e[0] for e in edges]
        cols = [e[1] for e in edges]
        sg = tessera.SparseGraph.fromCOO(rows, cols, n)
        g = PM.fromWeightedEdges(rows, cols, [1.0] * len(rows),
                                 list(range(n)))
        ids = g.cellIds()
        labels_pm = [ids[i] % 4 for i in range(g.nCells())]
        labels_sg = [v % 4 for v in range(n)]
        self.assertLessEqual(
            abs(sg.modularity(labels_sg) - g.modularityGamma(labels_pm, 1.0)),
            1e-15)

    def test_gamma_one_matches_spacetime_modularity_on_skeleton(self):
        st = _make_spacetime()
        g = PM.fromSpacetime(st, PM.WeightMap.Unit)
        M = 4
        labels = [int(v % M) for v in g.cellIds()]
        self.assertLessEqual(
            abs(g.modularityGamma(labels, 1.0) - st.modularityOnSkeleton(M)),
            1e-15)

    def test_label_length_mismatch_raises(self):
        src, tgt = _ring(10)
        g = PM.fromWeightedEdges(src, tgt, [1.0] * len(src))
        with self.assertRaises(ValueError):
            g.modularityGamma([0] * 3, 1.0)

    def test_negative_weight_accepted_and_switches_the_null_model(self):
        # The domain is the REAL weighted graph (#849): a negative weight is
        # a measured dissimilarity, and it selects the signed null model.
        # Nonnegative graphs are untouched -- the reduction is held to
        # bit-identity in test_causal_modularity_python.py.
        g = PM.fromWeightedEdges([0], [1], [-1.0])
        self.assertTrue(g.isSigned())
        self.assertFalse(g.isComplex())
        # T is the ABSOLUTE total; the signed sum SA is what the null model
        # redistributes, and here the two differ in sign.
        self.assertEqual(g.totalWeight2(), 2.0)
        self.assertEqual(g.totalWeightSum(), -2.0)

    def test_non_finite_weight_rejected(self):
        for bad in (float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                PM.fromWeightedEdges([0], [1], [bad])

    def test_edge_list_normalization(self):
        # Parallel edges (either direction) consolidate by weight sum;
        # self-loops and zero-weight edges are ignored at level 0.
        g = PM.fromWeightedEdges([0, 1, 0, 2, 3],
                                 [1, 0, 0, 2, 4],
                                 [1.0, 0.5, 0.5, 3.0, 0.0])
        self.assertEqual(g.nEdges(), 1)
        # (0,1,1.0) + (1,0,0.5) + (0,0,self) + (2,2,self) + (3,4,zero)
        # -> one edge of weight 1.5, 2m = 3.0 exactly.
        self.assertEqual(g.totalWeight2(), 3.0)


# ---------------------------------------------------------------------------
# matching across resolution / cobordism time + the projector hook
# ---------------------------------------------------------------------------


class TestComponentMatching(unittest.TestCase):

    def _two_frames(self):
        # "Time" t0: cliques {0..5}, {10..15}; t1: cell 5 migrated to the
        # second community (a local change on a common cell-id universe).
        src0, tgt0 = _two_disconnected_k6()
        src1, tgt1 = [], []
        _clique_edges(list(range(5)), src1, tgt1)
        _clique_edges([5] + list(range(10, 16)), src1, tgt1)
        g0 = PM.fromWeightedEdges(src0, tgt0, [1.0] * len(src0))
        g1 = PM.fromWeightedEdges(src1, tgt1, [1.0] * len(src1))
        s0 = g0.discover(1.0, _cfg())
        s1 = g1.discover(1.0, _cfg())
        return g0, s0, s1

    def test_support_overlap_matching_is_exact(self):
        g0, s0, s1 = self._two_frames()
        matches = g0.matchComponents(s0.components, s1.components)
        self.assertEqual(len(matches), 2)
        by_from = {tuple(s0.components[m.fromIndex].support): m
                   for m in matches}
        m_a = by_from[tuple(range(6))]
        m_b = by_from[tuple(range(10, 16))]
        # {0..5} vs {0..4}: |I| = 5, |U| = 6 -> 5/6 exactly.
        self.assertEqual(m_a.supportOverlap, 5.0 / 6.0)
        self.assertEqual(tuple(s1.components[m_a.toIndex].support),
                         tuple(range(5)))
        # {10..15} vs {5, 10..15}: |I| = 6, |U| = 7 -> 6/7 exactly.
        self.assertEqual(m_b.supportOverlap, 6.0 / 7.0)

    def test_projector_overlap_hook_unknown_until_installed(self):
        g0, s0, s1 = self._two_frames()
        matches = g0.matchComponents(s0.components, s1.components)
        for m in matches:
            # Unknown means absent, never zero.
            self.assertIsNone(m.projectorOverlap)
        seen = []

        def hook(from_id, to_id):
            seen.append((from_id.canonicalHash(), to_id.canonicalHash()))
            return 0.25

        g0.setProjectorOverlapHook(hook)
        matches = g0.matchComponents(s0.components, s1.components)
        for m in matches:
            self.assertEqual(m.projectorOverlap, 0.25)
        self.assertEqual(len(seen), len(matches))
        # Matching decisions stayed support-based (documented: the hook is
        # an interface for a later ticket, not a decision input here).
        self.assertEqual(sorted(m.supportOverlap for m in matches),
                         sorted((5.0 / 6.0, 6.0 / 7.0)))
        g0.setProjectorOverlapHook(None)
        matches = g0.matchComponents(s0.components, s1.components)
        for m in matches:
            self.assertIsNone(m.projectorOverlap)

    def test_component_id_semantics(self):
        src, tgt = _two_disconnected_k6()
        g = PM.fromWeightedEdges(src, tgt, [1.0] * len(src))
        s = g.discover(1.0, _cfg())
        ids = [c.id for c in s.components]
        for cid in ids:
            self.assertEqual(len(cid.canonicalHash()), 32)
            self.assertTrue(all(ch in "0123456789abcdef"
                                for ch in cid.canonicalHash()))
            self.assertEqual(cid.level(), 1)
        # Automorphic twins (two identical K6) share the structural hash by
        # construction; disambiguation is positional.
        self.assertEqual(ids[0], ids[1])
        self.assertEqual(hash(ids[0]), hash(ids[1]))


# ---------------------------------------------------------------------------
# local changes invalidate only affected component ancestry
# ---------------------------------------------------------------------------


class TestInvalidation(unittest.TestCase):

    def _report(self):
        src, tgt = [], []
        _clique_edges(list(range(5)), src, tgt)
        _clique_edges(list(range(100, 106)), src, tgt)
        _clique_edges(list(range(200, 207)), src, tgt)
        g = PM.fromWeightedEdges(src, tgt, [1.0] * len(src))
        report = g.scanResolutions(_cfg((0.5, 1.0, 2.0)))
        return g, report

    def test_only_touched_ancestry_invalidated(self):
        g, report = self._report()
        touched = [101]  # a cell of the middle clique only
        inv = PM.invalidatedAncestry(report, touched)
        touched_set = set(touched)
        # Every invalidated position intersects the touched cells...
        listed = set()
        for s, k, j in inv.positions:
            support = set(report.slices[s].hierarchy[k][j].support)
            self.assertTrue(support & touched_set)
            listed.add((s, k, j))
        # ...and every component NOT listed is disjoint from them: siblings
        # remain valid at every hierarchy level of every slice.
        for s, sl in enumerate(report.slices):
            for k, level in enumerate(sl.hierarchy):
                for j, comp in enumerate(level):
                    if (s, k, j) not in listed:
                        self.assertFalse(set(comp.support) & touched_set)
        # Exactly one track (the middle clique's) is affected.
        self.assertEqual(len(inv.tracks), 1)
        t = report.tracks[inv.tracks[0]]
        member0 = report.slices[t.firstSlice].components[t.memberIndices[0]]
        self.assertIn(101, member0.support)
        # The untouched cliques' tracks are valid.
        for ti, t in enumerate(report.tracks):
            if ti in inv.tracks:
                continue
            for i, idx in enumerate(t.memberIndices):
                comp = report.slices[t.firstSlice + i].components[idx]
                self.assertFalse(set(comp.support) & touched_set)

    def test_untouched_report_invalidates_nothing(self):
        g, report = self._report()
        inv = PM.invalidatedAncestry(report, [9999])
        self.assertEqual(inv.components, [])
        self.assertEqual(inv.positions, [])
        self.assertEqual(inv.tracks, [])

    def test_hierarchy_is_nested_partition(self):
        # The ancestry that invalidation walks is real: every hierarchy
        # level partitions the cells and refines upward into the next.
        src, tgt = _ring(60)
        g = PM.fromWeightedEdges(src, tgt, [1.0] * len(src))
        s = g.discover(0.5, _cfg())
        self.assertGreaterEqual(s.levels, 2)  # aggregation actually ran
        all_cells = set(g.cellIds())
        prev = None
        for level in s.hierarchy:
            cells = [v for c in level for v in c.support]
            self.assertEqual(sorted(cells), sorted(all_cells))
            self.assertEqual(len(cells), len(set(cells)))  # a partition
            if prev is not None:
                supports = [set(c.support) for c in level]
                for child in prev:
                    child_set = set(child.support)
                    holders = [sup for sup in supports if child_set & sup]
                    self.assertEqual(len(holders), 1)
                    self.assertTrue(child_set <= holders[0])  # nested
            prev = level
        self.assertEqual([c.id.canonicalHash() for c in s.hierarchy[-1]],
                         [c.id.canonicalHash() for c in s.components])


# ---------------------------------------------------------------------------
# spacetime entry points (read-only) and the ModularityOptimizer extension
# ---------------------------------------------------------------------------


class TestSpacetimeDiscovery(unittest.TestCase):

    def test_discover_components_is_read_only(self):
        st = _make_spacetime()
        n_v = st.getVertexCount()
        n_e = st.getEdgeList().size()
        n_s = st.getSimplexCount()
        opt = tessera.ModularityOptimizer(tessera.ModularityOptimizerConfig(),
                                          seed=0)
        report = opt.discoverComponents(st, _cfg((0.5, 1.0, 2.0), restarts=2))
        self.assertEqual(len(report.slices), 3)
        for s in report.slices:
            self.assertTrue(math.isfinite(s.objectiveValue))
            self.assertLessEqual(abs(s.q - s.qIncremental), 1e-13)
            self.assertGreater(len(s.components), 0)
        # Observables are read-only: the spacetime is untouched.
        self.assertEqual(st.getVertexCount(), n_v)
        self.assertEqual(st.getEdgeList().size(), n_e)
        self.assertEqual(st.getSimplexCount(), n_s)

    def test_weight_maps_documented_monotone_similarity(self):
        st = _make_spacetime()
        unit = PM.fromSpacetime(st, PM.WeightMap.Unit)
        expneg = PM.fromSpacetime(st, PM.WeightMap.ExpNegAbsLength)
        self.assertEqual(unit.nCells(), expneg.nCells())
        self.assertEqual(unit.nEdges(), expneg.nEdges())
        # Unit: 2m = 2 |E| exactly (the combinatorial one-skeleton).
        self.assertEqual(unit.totalWeight2(), 2.0 * unit.nEdges())
        # exp(-|l|) weights lie in (0, 1].
        self.assertGreater(expneg.totalWeight2(), 0.0)
        self.assertLessEqual(expneg.totalWeight2(), unit.totalWeight2())

    def test_spacetime_relabeling_scores(self):
        # RELABEL gate over the real complex loader: a relabeled rebuild
        # must give identical discovery scores and an isomorphic partition.
        # The toroidal CDT skeleton has nontrivial automorphisms
        # (translations), so — as on the ring fixtures — supports are only
        # guaranteed up to automorphism; pointwise support mapping is
        # covered by the planted (forced-partition) fixtures.  Hash
        # equality is not asserted either: canonical hashes read the
        # ORIENTED incidence and the fromCells rebuild inside
        # LiveComplex.relabel does not preserve stored source/target roles
        # (the pure-graph relabeling tests cover hashes with orientation
        # preserved).
        st = _make_spacetime()
        relabeled = tessera.LiveComplex.relabel(st, seed=4)
        g = PM.fromSpacetime(st, PM.WeightMap.Unit)
        g2 = PM.fromSpacetime(relabeled.spacetime, PM.WeightMap.Unit)
        cfg = _cfg((1.0,), restarts=2)
        a = g.discover(1.0, cfg)
        b = g2.discover(1.0, cfg)
        self.assertEqual(a.q, b.q)
        self.assertEqual(a.qIncremental, b.qIncremental)
        self.assertEqual(sorted(len(c.support) for c in a.components),
                         sorted(len(c.support) for c in b.components))


# ---------------------------------------------------------------------------
# #808: lifetime across COBORDISM FRAMES, distinct from resolution slices
# ---------------------------------------------------------------------------


class TestCobordismFrameTracks(unittest.TestCase):
    """The whitepaper's fiber-acceptance conjunct is "lifetime across
    multiple cobordism frames".  `matchComponents` always supported a time
    track; `trackAcrossFrames` is the supplier that calls it with more than
    one frame."""

    def _frames(self, count=3, migrate_at=None):
        """`count` cobordism frames over a common cell-id universe: two K6
        cliques, with cell 5 migrating to the second clique from frame
        `migrate_at` onward."""
        out = []
        for t in range(count):
            if migrate_at is not None and t >= migrate_at:
                src, tgt = [], []
                _clique_edges(list(range(5)), src, tgt)
                _clique_edges([5] + list(range(10, 16)), src, tgt)
            else:
                src, tgt = _two_disconnected_k6()
            g = PM.fromWeightedEdges(src, tgt, [1.0] * len(src))
            out.append(g.discover(1.0, _cfg()).components)
        return out

    def _graph(self):
        src, tgt = _two_disconnected_k6()
        return PM.fromWeightedEdges(src, tgt, [1.0] * len(src))

    def test_an_unchanged_component_lives_for_every_frame(self):
        g = self._graph()
        tracks = g.trackAcrossFrames(self._frames(4))
        self.assertEqual(len(tracks), 2)
        for track in tracks:
            self.assertEqual(track.frames, 4)
            self.assertEqual(track.firstFrame, 0)
            self.assertEqual(track.lastFrame, 3)
            self.assertEqual(track.minAdjacentOverlap, 1.0)
            self.assertEqual(len(track.members), 4)
            self.assertEqual(len(track.memberIndices), 4)

    def test_a_single_frame_gives_a_one_frame_lifetime(self):
        # Not a structural artifact: one frame is one frame, and the read
        # says so instead of borrowing a resolution-slice count.
        g = self._graph()
        tracks = g.trackAcrossFrames(self._frames(1))
        self.assertEqual(len(tracks), 2)
        for track in tracks:
            self.assertEqual(track.frames, 1)
            self.assertEqual(track.minAdjacentOverlap, 1.0)

    def test_no_frames_yields_no_tracks(self):
        self.assertEqual(self._graph().trackAcrossFrames([]), [])

    def test_a_migrating_cell_lowers_the_adjacent_frame_overlap(self):
        # Frames 0, 1 are identical; at frame 2 cell 5 migrates.  The exact
        # Jaccard overlaps are 5/6 and 6/7 -- the same numbers
        # matchComponents reports, since the chaining IS matchComponents.
        g = self._graph()
        tracks = g.trackAcrossFrames(self._frames(3, migrate_at=2))
        worst = sorted(t.minAdjacentOverlap for t in tracks)
        self.assertEqual(len(tracks), 2)
        self.assertEqual(worst, [5.0 / 6.0, 6.0 / 7.0])
        for track in tracks:
            self.assertEqual(track.frames, 3)

    def test_a_break_in_the_track_starts_a_new_lifetime(self):
        # Frame 1 relabels the whole universe: nothing overlaps, so no
        # component continues and every frame starts its own track.
        src, tgt = _two_disconnected_k6()
        far_src = [v + 10_000 for v in src]
        far_tgt = [v + 10_000 for v in tgt]
        g = self._graph()
        g_far = PM.fromWeightedEdges(far_src, far_tgt, [1.0] * len(far_src))
        frames = [g.discover(1.0, _cfg()).components,
                  g_far.discover(1.0, _cfg()).components]
        tracks = g.trackAcrossFrames(frames)
        self.assertEqual(len(tracks), 4)
        for track in tracks:
            self.assertEqual(track.frames, 1)

    def test_the_frame_axis_is_not_the_resolution_axis(self):
        # The same chaining rule on the two axes gives two DIFFERENT
        # numbers, which is why they are separate fields: a scan over three
        # resolutions of ONE frame reports lifetime 3 on the resolution
        # axis, while that single frame is one frame.
        g = self._graph()
        report = g.scanResolutions(_cfg(resolutions=(0.5, 1.0, 2.0)))
        self.assertEqual(len(report.slices), 3)
        resolution_lifetimes = sorted(t.lastSlice - t.firstSlice + 1
                                      for t in report.tracks)
        self.assertEqual(resolution_lifetimes, [3, 3])
        frame_tracks = g.trackAcrossFrames([report.slices[1].components])
        self.assertEqual(sorted(t.frames for t in frame_tracks), [1, 1])

    def test_the_threshold_is_honoured(self):
        # Below the overlap threshold the chain breaks, exactly as the
        # resolution scan breaks: same rule, different axis.
        g = self._graph()
        frames = self._frames(2, migrate_at=1)
        kept = g.trackAcrossFrames(frames, 0.8)
        self.assertEqual(sorted(t.frames for t in kept), [2, 2])
        broken = g.trackAcrossFrames(frames, 0.9)
        self.assertEqual(sorted(t.frames for t in broken), [1, 1, 1, 1])


if __name__ == "__main__":
    unittest.main()
