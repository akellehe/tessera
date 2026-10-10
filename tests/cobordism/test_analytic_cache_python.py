# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""Geometry-revision + touched-star cache keys (#764).

AnalyticCache entries are keyed by the order-independent component
vertex-set fingerprint and stamped with Spacetime.metric_revision_key(). Every
accepted move publishes its TouchedStar; entries meeting the star are
invalidated while disjoint SIBLINGS SURVIVE, and an unpublished revision
drift serves nothing (fail-safe). Cached and cold results are compared to
machine precision across randomized accepted metric moves and a structural
(edge-creating) move.
"""

import cmath
import unittest

import numpy as np

import tessera

cob = tessera.cobordism

TOL = 1e-15

# Two disjoint triangle components in one complex: A on vertices {0,1,2},
# B on {10,11,12} (deliberately non-contiguous identifiers).
_A_VERTS = [0, 1, 2]
_B_VERTS = [10, 11, 12]


def _two_triangles():
    sig = tessera.Signature(4, tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    st = tessera.Spacetime(metric, tessera.HERMITIAN_WEIGHTED, 1.0, 1.0,
                           tessera.PREFERRED, tessera.Toroid())
    verts = {i: st.create_vertex(i) for i in _A_VERTS + _B_VERTS}
    for a, b in [(0, 1), (1, 2), (0, 2), (10, 11), (11, 12), (10, 12)]:
        st.create_simplex([verts[a], verts[b]])
    for e in st.get_edge_list().to_vector():
        e.set_length(1.0 + 0j)
        e.set_phase(0.0)
    return st


def _component_block(st, component_vertices):
    """Cold recompute: the k=0 Laplacian block of one component, extracted
    from the whole-complex operator in the sorted-vertex-id basis."""
    ids = sorted(v.get_id() for v in st.get_vertex_list().to_vector())
    index = {vid: i for i, vid in enumerate(ids)}
    n = len(ids)
    lap = np.array(cob.HodgeLaplacian(st).laplacian(0)).reshape(n, n)
    rows = [index[v] for v in sorted(component_vertices)]
    return lap[np.ix_(rows, rows)]


def _edges_of(st, component_vertices):
    members = set(component_vertices)
    return [e for e in st.get_edge_list().to_vector()
            if e.get_source().get_id() in members
            and e.get_target().get_id() in members]


def _exact_cert():
    return cob.Certificate.algebraically_exact(
        cob.CertificateDomain.Static,
        cob.CertificateRegime.PositiveSemidefinite, 0.0, TOL)


class TestComponentKey(unittest.TestCase):
    def test_order_independent(self):
        # The key is a set fingerprint: any permutation, same key.
        key = cob.AnalyticCache.component_key([3, 1, 2])
        self.assertEqual(key, cob.AnalyticCache.component_key([2, 3, 1]))
        self.assertEqual(key, cob.AnalyticCache.component_key([1, 2, 3]))
        self.assertNotEqual(key, cob.AnalyticCache.component_key([1, 2, 4]))


class TestTouchedStar(unittest.TestCase):
    def test_records_support_and_structural_flag(self):
        star = cob.TouchedStar()
        self.assertTrue(star.empty)
        star.add_changed_edge(4, 7)
        star.add_touched_simplex([7, 8, 9])
        self.assertFalse(star.structural_change)
        star.add_created_cell([9, 10])
        self.assertTrue(star.structural_change)
        self.assertEqual(sorted(star.vertices), [4, 7, 8, 9, 10])


class TestSiblingSurvival(unittest.TestCase):
    def test_local_metric_move_keeps_disjoint_sibling(self):
        st = _two_triangles()
        cache = cob.AnalyticCache(st)
        cache.store(_A_VERTS, "hodge-block", 0, _component_block(st, _A_VERTS),
                    _exact_cert())
        cache.store(_B_VERTS, "hodge-block", 0, _component_block(st, _B_VERTS),
                    _exact_cert())
        self.assertEqual(cache.size, 2)

        # Accepted move: one edge of component A changes length.
        edge = _edges_of(st, _A_VERTS)[0]
        edge.set_length(cmath.sqrt(2.5 + 0j))
        star = cob.TouchedStar()
        star.add_changed_edge(edge.get_source().get_id(),
                            edge.get_target().get_id())
        cache.publish(star)

        # Touched component: invalidated. Sibling: served, and equal to the
        # cold recompute bit-for-bit.
        self.assertIsNone(cache.fetch(_A_VERTS, "hodge-block", 0))
        cached_b = cache.fetch(_B_VERTS, "hodge-block", 0)
        self.assertIsNotNone(cached_b)
        np.testing.assert_array_equal(cached_b,
                                      _component_block(st, _B_VERTS))
        self.assertEqual(cache.invalidations, 1)

    def test_structural_move_keeps_disjoint_sibling(self):
        st = _two_triangles()
        cache = cob.AnalyticCache(st)
        cache.store(_A_VERTS, "hodge-block", 0, _component_block(st, _A_VERTS),
                    _exact_cert())
        cache.store(_B_VERTS, "hodge-block", 0, _component_block(st, _B_VERTS),
                    _exact_cert())
        before = cache.structural_revision()

        # Accepted structural move: a new vertex + cell coned onto B.
        new_vertex = st.create_vertex(99)
        anchor = next(v for v in st.get_vertex_list().to_vector()
                      if v.get_id() == 10)
        st.create_simplex([anchor, new_vertex])
        for e in st.get_edge_list().to_vector():
            if 99 in (e.get_source().get_id(), e.get_target().get_id()):
                e.set_length(1.0 + 0j)
                e.set_phase(0.0)
        self.assertGreater(cache.structural_revision(), before)

        star = cob.TouchedStar()
        star.add_created_cell([10, 99])
        cache.publish(star)

        # B was touched (10 is in the star): gone. A is a disjoint sibling:
        # served and equal to cold.
        self.assertIsNone(cache.fetch(_B_VERTS, "hodge-block", 0))
        cached_a = cache.fetch(_A_VERTS, "hodge-block", 0)
        self.assertIsNotNone(cached_a)
        np.testing.assert_array_equal(cached_a,
                                      _component_block(st, _A_VERTS))

    def test_whole_complex_entry_dies_on_any_touch(self):
        st = _two_triangles()
        cache = cob.AnalyticCache(st)
        everything = _A_VERTS + _B_VERTS
        cache.store(everything, "hodge-block", 0,
                    _component_block(st, everything), _exact_cert())
        edge = _edges_of(st, _B_VERTS)[0]
        edge.set_length(cmath.sqrt(3.0 + 0j))
        star = cob.TouchedStar()
        star.add_changed_edge(edge.get_source().get_id(),
                            edge.get_target().get_id())
        cache.publish(star)
        self.assertIsNone(cache.fetch(everything, "hodge-block", 0))


class TestFreshnessContract(unittest.TestCase):
    def test_unpublished_drift_serves_nothing(self):
        st = _two_triangles()
        cache = cob.AnalyticCache(st)
        cache.store(_B_VERTS, "hodge-block", 0, _component_block(st, _B_VERTS),
                    _exact_cert())
        # Mutate WITHOUT publishing: even the untouched sibling is refused —
        # a stale hit is impossible, only recomputation.
        _edges_of(st, _A_VERTS)[0].set_length(cmath.sqrt(1.7 + 0j))
        self.assertIsNone(cache.fetch(_B_VERTS, "hodge-block", 0))
        # Publishing the drift restores service to survivors.
        star = cob.TouchedStar()
        star.add_changed_edge(0, 1)
        star.add_changed_edge(1, 2)
        star.add_changed_edge(0, 2)
        cache.publish(star)
        self.assertIsNotNone(cache.fetch(_B_VERTS, "hodge-block", 0))

    def test_store_after_drift_serves_the_fresh_entry_only(self):
        st = _two_triangles()
        cache = cob.AnalyticCache(st)
        cache.store(_A_VERTS, "hodge-block", 0, _component_block(st, _A_VERTS),
                    _exact_cert())
        _edges_of(st, _A_VERTS)[0].set_length(cmath.sqrt(0.9 + 0j))
        # Re-store A at the current revision: served. (Nothing else exists.)
        cache.store(_A_VERTS, "hodge-block", 0, _component_block(st, _A_VERTS),
                    _exact_cert())
        self.assertIsNotNone(cache.fetch(_A_VERTS, "hodge-block", 0))

    def test_disabled_cache_serves_nothing(self):
        st = _two_triangles()
        cache = cob.AnalyticCache(st)
        cache.store(_A_VERTS, "hodge-block", 0, _component_block(st, _A_VERTS),
                    _exact_cert())
        cache.set_enabled(False)
        self.assertIsNone(cache.fetch(_A_VERTS, "hodge-block", 0))
        self.assertFalse(cache.enabled)
        cache.set_enabled(True)
        self.assertIsNotNone(cache.fetch(_A_VERTS, "hodge-block", 0))

    def test_certificate_travels_with_the_entry(self):
        st = _two_triangles()
        cache = cob.AnalyticCache(st)
        cert = cob.Certificate.certified_numerical(
            cob.CertificateDomain.BandWindow,
            cob.CertificateRegime.HermitianIndefinite, 1e-11, 5.0, 1e-9)
        cache.store(_A_VERTS, "spectral-projector", 2, object(), cert)
        fetched = cache.fetch_certificate(_A_VERTS, "spectral-projector", 2)
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.grade, cob.CertificateGrade.CertifiedNumerical)
        self.assertEqual(fetched.domain, cob.CertificateDomain.BandWindow)
        self.assertEqual(fetched.residual, 1e-11)
        self.assertIsNone(
            cache.fetch_certificate(_A_VERTS, "spectral-projector", 3))

    def test_kind_and_parameter_are_part_of_the_key(self):
        st = _two_triangles()
        cache = cob.AnalyticCache(st)
        cache.store(_A_VERTS, "hodge-block", 1, "k1", _exact_cert())
        cache.store(_A_VERTS, "hodge-block", 2, "k2", _exact_cert())
        cache.store(_A_VERTS, "lu-factorization", 1, "lu", _exact_cert())
        self.assertEqual(cache.fetch(_A_VERTS, "hodge-block", 1), "k1")
        self.assertEqual(cache.fetch(_A_VERTS, "hodge-block", 2), "k2")
        self.assertEqual(cache.fetch(_A_VERTS, "lu-factorization", 1), "lu")


class TestRandomAcceptedMoves(unittest.TestCase):
    def test_cached_equals_cold_across_random_accepted_moves(self):
        """Acceptance: cached and cold results agree to the declared
        tolerance across random accepted moves, with the correct-component
        miss/hit pattern and sibling survival throughout."""
        st = _two_triangles()
        cache = cob.AnalyticCache(st)
        rng = np.random.default_rng(97)
        components = {"A": _A_VERTS, "B": _B_VERTS}
        for name, verts in components.items():
            cache.store(verts, "hodge-block", 0, _component_block(st, verts),
                        _exact_cert())

        for _ in range(40):
            touched_name = "A" if rng.random() < 0.5 else "B"
            sibling_name = "B" if touched_name == "A" else "A"
            touched = components[touched_name]
            sibling = components[sibling_name]

            edges = _edges_of(st, touched)
            edge = edges[int(rng.integers(len(edges)))]
            new_sq = complex(0.25 + 2.0 * rng.random(),
                             0.5 * (rng.random() - 0.5))
            edge.set_length(cmath.sqrt(new_sq))
            edge.set_phase(float(rng.random() - 0.5))
            star = cob.TouchedStar()
            star.add_changed_edge(edge.get_source().get_id(),
                                edge.get_target().get_id())
            cache.publish(star)

            # Touched entry: refused; sibling: served, equal to cold.
            self.assertIsNone(cache.fetch(touched, "hodge-block", 0))
            served = cache.fetch(sibling, "hodge-block", 0)
            self.assertIsNotNone(served)
            cold = _component_block(st, sibling)
            np.testing.assert_allclose(served, cold, rtol=0, atol=TOL)

            # Recompute + re-store the touched entry; it must now serve and
            # equal a second cold recompute exactly.
            cache.store(touched, "hodge-block", 0,
                        _component_block(st, touched), _exact_cert())
            np.testing.assert_allclose(
                cache.fetch(touched, "hodge-block", 0),
                _component_block(st, touched), rtol=0, atol=TOL)

        self.assertGreater(cache.hits, 0)
        self.assertEqual(cache.invalidations, 40)


if __name__ == "__main__":
    unittest.main()
