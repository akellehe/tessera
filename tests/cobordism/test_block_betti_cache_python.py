# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""The parent-side sub-complex Betti cache (#705) is invisible in values.

Input-block residuals materialize each block's region as a FRESH spacetime,
so the per-instance Betti slot (#681) never carried across evaluations and
the Smith normal form recomputed per block, per line-search trial, per
candidate (measured: 47.5% of live-run cycles). The parent spacetime now
caches each block's Betti numbers per (structural revision, vertex-set key).
The cache changes WHEN the computation runs, never its input or output:
warm and cold evaluations must agree bitwise, and evaluations must stay
exact across metric changes and structural moves.
"""
import cmath
import math
import unittest

import tessera

cob = tessera.cobordism


def _seed():
    return tessera.Spacetime.fromVertexTuples(4, [[0, 1, 2, 3, 4]], 1.0, 0.0)


def _node(seed=5):
    return cob.ProtonSynthesis(seed=seed).direct_node(seed)


class RegionFingerprintTest(unittest.TestCase):
    """`Fingerprint.fingerprintOf` names a set of identifiers at any size."""

    def test_holds_sets_larger_than_an_instance_can(self):
        # A Fingerprint INSTANCE stores at most tessera::mesh::kMax = 8
        # identifiers and drops the rest silently, so sets agreeing on their
        # first eight members would share a name. Block regions routinely
        # exceed that (a live-run complex carried 21 vertices), which is why
        # the region key calls the static instead of holding an instance.
        name = tessera.mesh.Fingerprint.fingerprintOf
        base = set(range(1, 9))
        names = {name(base | {tail}) for tail in range(9, 20)}
        self.assertEqual(len(names), 11)
        self.assertNotIn(name(base), names)

    def test_instance_and_static_are_one_implementation(self):
        # `fingerprint()` delegates to `fingerprintOf`, so an instance and a
        # caller hashing the same identifiers cannot drift apart.
        fingerprint = tessera.mesh.Fingerprint
        for ids in ({1, 2, 3}, {7}, {11, 4, 9, 2}, set(range(1, 9))):
            self.assertEqual(fingerprint(sorted(ids)).fingerprint(),
                             fingerprint.fingerprintOf(ids))

    def test_instance_truncates_where_the_static_does_not(self):
        # Nine identifiers: the instance keeps kMax = 8 of them and its hash
        # no longer names the set it was given, which is exactly why the
        # region key calls the static.
        fingerprint = tessera.mesh.Fingerprint
        ids = set(range(1, 10))
        self.assertNotEqual(fingerprint(sorted(ids)).fingerprint(),
                            fingerprint.fingerprintOf(ids))
        self.assertEqual(fingerprint(sorted(ids)).fingerprint(),
                         fingerprint.fingerprintOf(set(range(1, 9))))

    def test_names_the_set_not_the_order(self):
        name = tessera.mesh.Fingerprint.fingerprintOf
        self.assertEqual(name({1, 2, 3}), name({3, 2, 1}))
        self.assertNotEqual(name({1, 2, 3}), name({1, 2, 4}))
        self.assertNotEqual(name({1, 2, 3}), name({1, 2}))


class BlockBettiCacheTest(unittest.TestCase):

    def test_warm_repeat_is_bitwise_identical(self):
        node = _node()
        st = _seed()
        cold = node.r_u(st)      # fills the parent-side slots
        warm = node.r_u(st)      # every block served from the cache
        self.assertEqual(cold, warm)

    def test_warm_equals_cold_on_a_fresh_identical_parent(self):
        # A separately constructed identical parent starts with empty slots:
        # its first (cold) evaluation must equal the warmed one's bitwise.
        node_a = _node()
        node_b = _node()
        st_warmed = _seed()
        node_a.r_u(st_warmed)
        warm = node_a.r_u(st_warmed)
        cold = node_b.r_u(_seed())
        self.assertEqual(warm, cold)

    def test_metric_change_reuses_structure_exactly(self):
        # Scaling every edge changes the metric but not the combinatorics:
        # the cached Betti numbers stay valid, and the residual still matches
        # a cold evaluation of the same scaled geometry bitwise.
        st_a, st_b = _seed(), _seed()
        node_a, node_b = _node(), _node()
        node_a.r_u(st_a)                       # warm the slots pre-scale
        for st in (st_a, st_b):
            for e in st.getEdgeList().toVector():
                e.setLength(e.getLength() * cmath.sqrt(2.0))
            st.materializeFacets()
        self.assertEqual(node_a.r_u(st_a), node_b.r_u(st_b))

    def test_structural_move_invalidates_and_stays_exact(self):
        # One combined-drive iteration applies gated structural moves to the
        # node's own host; the revision bump must invalidate the slots and the
        # objective must keep evaluating (finite) on the changed complex.
        node = _node()
        node.r_u(node.st)
        node.run(max_iters=1, n_candidate_moves=4, grow_boundaries=True,
                 beta=1.0, alpha0=0.05, tolerance=1e-6, max_lookahead=1,
                 relax_budget_per_move=2)
        after = node.r_u(node.st)
        self.assertTrue(math.isfinite(after))
        self.assertGreaterEqual(after, 0.0)
        self.assertEqual(after, node.r_u(node.st))


if __name__ == "__main__":
    unittest.main()
