# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The edge list holds exactly one edge per vertex pair.

``Spacetime.swap_vertex_labels`` rewrites the ids two vertices hold, which changes
the vertex pair of every edge incident to exactly one of them, and so changes
the key those edges are stored under. The relabeling that ``AddMove`` performs
after a successful add drives that path on every accepted move.

If an edge is not found under its old key it is never rewritten, and the next
lookup for its pair creates a second edge for the same pair. Two live edges on
one pair make the edge list unusable as an index: the action's derivatives are
keyed by vertex pair, so one handle receives the pair's whole derivative and the
others report exactly zero.
"""

import collections
import unittest

import tessera


def _pair(edge):
    a, b = edge.get_source().get_id(), edge.get_target().get_id()
    return (min(a, b), max(a, b))


def _pair_counts(spacetime):
    return collections.Counter(
        _pair(edge) for edge in spacetime.get_edge_list().to_vector())


def _build(seed, n_simplices=400):
    metric = tessera.Metric(
        coordinate_free=True,
        signature=tessera.Signature(dimensions=4,
                                    signature_type=tessera.Lorentzian),
    )
    spacetime = tessera.Spacetime(
        metric=metric, spacetime_type=tessera.CDT, alpha=1.0, a=1.0,
        foliation=tessera.PREFERRED, topology=tessera.Toroid(),
    )
    spacetime.set_seed(seed)
    spacetime.build(n_simplices)
    return spacetime


class TestEdgePairUniqueness(unittest.TestCase):

    def test_a_freshly_built_spacetime_has_one_edge_per_pair(self):
        spacetime = _build(7)
        counts = _pair_counts(spacetime)
        self.assertEqual(sorted(set(counts.values())), [1])
        self.assertEqual(spacetime.get_edge_list().size(), sum(counts.values()))

    def test_label_swaps_alone_keep_one_edge_per_pair(self):
        import random

        spacetime = _build(7)
        vertices = spacetime.get_vertex_list().to_vector()
        rng = random.Random(7)
        for _ in range(200):
            first, second = rng.sample(range(len(vertices)), 2)
            spacetime.swap_vertex_labels(vertices[first], vertices[second])
        counts = _pair_counts(spacetime)
        self.assertEqual(sorted(set(counts.values())), [1])

    def test_relabeling_sweeps_keep_one_edge_per_pair(self):
        """The regression: sweeps that relabel used to duplicate pairs.

        Before ``swap_vertex_labels`` detached edges by a key derived from their
        endpoints, 300 relabeling sweeps left 123 vertex pairs carrying more
        than one edge.
        """
        spacetime = _build(7)
        target = spacetime.get_n41()
        cdt = tessera.CDTSimulation(
            spacetime=spacetime, k0=2.2, k4=0.5, delta=0.6,
            epsilon=1.0 / target, target_n41=target,
        )
        cdt.set_seed(7)
        cdt.set_relabel_vertices(True)
        cdt.tune()
        cdt.sweep(300)

        counts = _pair_counts(spacetime)
        duplicated = {pair: n for pair, n in counts.items() if n > 1}
        self.assertEqual(duplicated, {})

    def test_size_agrees_with_the_vector_it_hands_out(self):
        spacetime = _build(7)
        target = spacetime.get_n41()
        cdt = tessera.CDTSimulation(
            spacetime=spacetime, k0=2.2, k4=0.5, delta=0.6,
            epsilon=1.0 / target, target_n41=target,
        )
        cdt.set_seed(7)
        cdt.set_relabel_vertices(True)
        cdt.tune()
        cdt.sweep(300)

        edge_list = spacetime.get_edge_list()
        self.assertEqual(edge_list.size(), len(edge_list.to_vector()))

    def test_every_edge_is_findable_under_its_derived_key(self):
        """An edge that cannot be found again is what creates a duplicate."""
        spacetime = _build(7)
        target = spacetime.get_n41()
        cdt = tessera.CDTSimulation(
            spacetime=spacetime, k0=2.2, k4=0.5, delta=0.6,
            epsilon=1.0 / target, target_n41=target,
        )
        cdt.set_seed(7)
        cdt.set_relabel_vertices(True)
        cdt.tune()
        cdt.sweep(200)

        edge_list = spacetime.get_edge_list()
        before = len(edge_list.to_vector())
        for edge in list(edge_list.to_vector()):
            spacetime.create_edge(edge.get_source(), edge.get_target())
        self.assertEqual(len(edge_list.to_vector()), before)


if __name__ == "__main__":
    unittest.main()
