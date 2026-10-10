# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Pinning constrains the geometry; `dual_complex_valid` gates the topology (#835).

The engine used to conflate the two: `strandsPinned` rejected any surgery that
removed a pinned vertex, layered on top of the manifold check that already
decides whether a move leaves a valid complex. That gate was stronger than the
invariant it protected, and it forecloses legitimate topology changes for a
bookkeeping reason rather than a geometric one — which matters because surgery
is the only topology-changing mechanism the engine has. Pachner moves are
bistellar and preserve Betti numbers; geometric relaxation changes no topology
at all.

What these tests pin, along the axis that actually separates the two concerns:

  * a pinned edge does NOT move under stage-2 relaxation, and an unpinned one
    does — pinning constrains the geometry, and that is its whole effect;
  * a surgery that removes a pinned vertex is ACCEPTED when the result is a
    valid manifold, and REJECTED (naming the manifold verdict) when it is not —
    `dual_complex_valid` is the whole gate;
  * a region declared against a node with targets behaves identically to one
    declared against a node without them — pinning is target-free;
  * a region is a declared thing with an identity, so re-declaring a name
    replaces it rather than accumulating.
"""
import cmath
import unittest

import tessera

cob = tessera.cobordism

_DIM = 4

# Minimal well-formed targets. The point of several tests below is that pinning
# behaves the same whether or not these are present, so both shapes are used.
_IN = [[complex(1.0, 0.0)]]
_OUT = [[complex(1.0, 0.0)]]
_NO_TARGETS = []


def _single_delta4():
    """A single Δ⁴ pentatope with a uniform ℓ²=1 metric — the minimal emergent seed."""
    sig = tessera.Signature(_DIM, tessera.Lorentzian)
    st = tessera.Spacetime(tessera.Metric(True, sig), tessera.CDT, 1.0, 1.0,
                           tessera.PREFERRED, tessera.SolidSimplex(_DIM))
    st.build()
    for edge in st.get_edge_list().to_vector():
        edge.set_length(cmath.sqrt(complex(1.0)))
    return st


def _perturbed_delta4():
    """The same seed with a NON-uniform metric.

    The uniform ℓ²=1 seed is already stationary: stage 2 moves nothing on it, so a
    "the pinned edge did not move" assertion would pass there for the wrong reason.
    Perturbing the lengths gives the relaxation real work, which is what makes the
    pinned/unpinned contrast meaningful. Measured: 10 of 10 edges move.
    """
    st = _single_delta4()
    for index, edge in enumerate(st.get_edge_list().to_vector()):
        edge.set_length(cmath.sqrt(complex(0.8 + 0.13 * (index % 5))))
    return st


def _node(inputs=_IN, outputs=_OUT, seed=1, precone=0, host=None):
    return cob.MultiCobordism(host if host is not None else _single_delta4(),
                              inputs, outputs, degrees=[3],
                              gamma=1.0, seed=seed, precone=precone)


def _relaxing_node(inputs=_IN, outputs=_OUT):
    """A node whose geometry actually relaxes — see `_perturbed_delta4`."""
    return _node(inputs=inputs, outputs=outputs, host=_perturbed_delta4())


def _cells(st):
    """Top cells as sorted vertex-id tuples."""
    return [tuple(sorted(v.get_id() for v in s.get_vertices()))
            for s in st.get_top_simplices()]


def _vertex_ids(st):
    return sorted({v for cell in _cells(st) for v in cell})


def _edge_lengths(st):
    """Map an edge's canonical endpoint pair to its complex length."""
    lengths = {}
    for edge in st.get_edge_list().to_vector():
        a, b = edge.get_source().get_id(), edge.get_target().get_id()
        lengths[(min(a, b), max(a, b))] = edge.get_length()
    return lengths


class PinnedRegionDeclarationTest(unittest.TestCase):
    """A pinned region is a declared thing with an identity, not a derived set."""

    def test_a_fresh_node_pins_nothing(self):
        node = _node()
        self.assertEqual(node.pinned_regions(), [])
        self.assertEqual(node.pinned_vertices(), set())

    def test_a_declared_region_is_reported_back(self):
        node = _node()
        ids = _vertex_ids(node.st)
        node.declare_pinned_region("boundary", {ids[0], ids[1]})
        regions = node.pinned_regions()
        self.assertEqual(len(regions), 1)
        self.assertEqual(regions[0][0], "boundary")
        self.assertEqual(regions[0][1], {ids[0], ids[1]})

    def test_redeclaring_a_name_replaces_rather_than_accumulates(self):
        # Identity is what makes a region referrable later (a per-region objective
        # is #837); a name that accumulated duplicates could not be referred to.
        node = _node()
        ids = _vertex_ids(node.st)
        node.declare_pinned_region("m0", {ids[0], ids[1]})
        node.declare_pinned_region("m0", {ids[2], ids[3]})
        regions = node.pinned_regions()
        self.assertEqual(len(regions), 1)
        self.assertEqual(regions[0][1], {ids[2], ids[3]})

    def test_distinct_names_coexist_in_declaration_order(self):
        node = _node()
        ids = _vertex_ids(node.st)
        node.declare_pinned_region("m0", {ids[0], ids[1]})
        node.declare_pinned_region("m1", {ids[2], ids[3]})
        self.assertEqual([r[0] for r in node.pinned_regions()], ["m0", "m1"])
        self.assertEqual(node.pinned_vertices(), {ids[0], ids[1], ids[2], ids[3]})

    def test_clearing_drops_every_region(self):
        node = _node()
        ids = _vertex_ids(node.st)
        node.declare_pinned_region("m0", {ids[0], ids[1]})
        node.clear_pinned_regions()
        self.assertEqual(node.pinned_regions(), [])
        self.assertEqual(node.pinned_vertices(), set())


class PinnedEdgeMembershipTest(unittest.TestCase):
    """An edge is pinned iff ONE region holds both endpoints."""

    def test_both_endpoints_in_one_region_pins_the_edge(self):
        node = _node()
        ids = _vertex_ids(node.st)
        node.declare_pinned_region("m0", {ids[0], ids[1]})
        self.assertTrue(node.edge_is_pinned(ids[0], ids[1]))

    def test_one_free_endpoint_leaves_the_edge_free(self):
        node = _node()
        ids = _vertex_ids(node.st)
        node.declare_pinned_region("m0", {ids[0], ids[1]})
        self.assertFalse(node.edge_is_pinned(ids[0], ids[2]))

    def test_an_edge_spanning_two_regions_is_bulk(self):
        # Two independently declared regions do not implicitly weld the gap between
        # them: the edge between them belongs to the bulk and must stay free.
        node = _node()
        ids = _vertex_ids(node.st)
        node.declare_pinned_region("m0", {ids[0]})
        node.declare_pinned_region("m1", {ids[1]})
        self.assertFalse(node.edge_is_pinned(ids[0], ids[1]))

    def test_membership_is_symmetric(self):
        node = _node()
        ids = _vertex_ids(node.st)
        node.declare_pinned_region("m0", {ids[0], ids[1]})
        self.assertEqual(node.edge_is_pinned(ids[0], ids[1]),
                         node.edge_is_pinned(ids[1], ids[0]))


class PinningConstrainsTheGeometryTest(unittest.TestCase):
    """Pinning's whole effect: a pinned edge does not move under relaxation."""

    def test_a_pinned_edge_holds_its_length_while_the_bulk_relaxes(self):
        node = _relaxing_node()
        ids = _vertex_ids(node.st)
        pinned_pair = (min(ids[0], ids[1]), max(ids[0], ids[1]))
        node.declare_pinned_region("held", {ids[0], ids[1]})

        before = _edge_lengths(node.st)
        node.run_stage2(beta=1.0, max_iters=12, alpha0=0.05)
        after = _edge_lengths(node.st)

        self.assertEqual(before[pinned_pair], after[pinned_pair],
                         "a pinned edge must keep its resident length exactly")

    def test_an_unpinned_edge_is_free_to_move(self):
        # The companion to the test above: without it, "pinned edge did not move"
        # would also pass on a complex where nothing moves at all.
        node = _relaxing_node()
        before = _edge_lengths(node.st)
        node.run_stage2(beta=1.0, max_iters=12, alpha0=0.05)
        after = _edge_lengths(node.st)
        moved = [k for k in before if before[k] != after[k]]
        self.assertTrue(moved,
                        "with nothing pinned the relaxation must move some edge, "
                        "otherwise the pinned-edge assertion is vacuous")

    def test_pinning_every_vertex_leaves_the_geometry_untouched(self):
        node = _relaxing_node()
        node.declare_pinned_region("all", set(_vertex_ids(node.st)))
        before = _edge_lengths(node.st)
        node.run_stage2(beta=1.0, max_iters=12, alpha0=0.05)
        self.assertEqual(before, _edge_lengths(node.st))


class PinningIsTargetFreeTest(unittest.TestCase):
    """A region means the same thing with or without targets present."""

    def test_membership_is_identical_with_and_without_targets(self):
        with_targets = _node(inputs=_IN, outputs=_OUT)
        without_targets = _node(inputs=_IN, outputs=_NO_TARGETS)
        for node in (with_targets, without_targets):
            ids = _vertex_ids(node.st)
            node.declare_pinned_region("m0", {ids[0], ids[1]})
        self.assertEqual(with_targets.pinned_vertices(),
                         without_targets.pinned_vertices())
        self.assertEqual(
            with_targets.edge_is_pinned(*sorted(with_targets.pinned_vertices())[:2]),
            without_targets.edge_is_pinned(
                *sorted(without_targets.pinned_vertices())[:2]))

    def test_the_geometric_hold_is_identical_with_and_without_targets(self):
        held = {}
        for label, outputs in (("with", _OUT), ("without", _NO_TARGETS)):
            node = _relaxing_node(inputs=_IN, outputs=outputs)
            ids = _vertex_ids(node.st)
            pair = (min(ids[0], ids[1]), max(ids[0], ids[1]))
            node.declare_pinned_region("m0", {ids[0], ids[1]})
            before = _edge_lengths(node.st)
            node.run_stage2(beta=1.0, max_iters=12, alpha0=0.05)
            after = _edge_lengths(node.st)
            held[label] = (before[pair] == after[pair])
        self.assertEqual(held["with"], held["without"])
        self.assertTrue(held["with"], "the pinned edge must be held in both shapes")


class ManifoldValidityIsTheOnlyGateTest(unittest.TestCase):
    """Surgery is decided by `dual_complex_valid`, not by whether a pin was removed."""

    def _cell_whose_removal_is_valid(self, node):
        """A top cell whose cone-out the manifold gate accepts, and the vertex that
        removing it strands. Rolled back, so the node is left untouched.

        Not every cell qualifies — most removals pinch the complex — so the cell is
        FOUND rather than assumed. Without this the acceptance test below would sit
        on the rejection branch and never assert what the ticket is about.
        """
        for cell in _cells(node.st):
            before = set(_vertex_ids(node.st))
            cone = cob.SurgicalCone(node.st)
            accepted, _ = cone.cone_out(list(cell))
            if accepted:
                removed = before - set(_vertex_ids(node.st))
                cone.rollback()
                if removed:
                    return list(cell), removed
        return None, set()

    def test_a_coneout_removing_a_pinned_vertex_is_accepted_when_valid(self):
        # THE headline claim: pinning does not veto a topology change that leaves a
        # valid manifold. Before this ticket `strandsPinned` rejected exactly this.
        node = _node(precone=5, seed=2)
        cell, removed = self._cell_whose_removal_is_valid(node)
        self.assertIsNotNone(
            cell, "fixture must offer a cone-out the manifold gate accepts")
        self.assertTrue(removed, "the cone-out must actually strand a vertex, "
                                 "otherwise pinning is not under test")

        # Pin precisely the vertex the surgery is about to strand.
        node.declare_pinned_region("doomed", set(removed))
        self.assertTrue(node.pinned_vertices() & removed)

        accepted, why = cob.SurgicalCone(node.st).cone_out(cell)
        self.assertTrue(accepted,
                        f"a valid cone-out must be accepted even though it removes a "
                        f"pinned vertex; got: {why}")
        self.assertFalse(set(_vertex_ids(node.st)) & removed,
                         "the pinned vertex really was removed")
        ok, _ = cob.EigenstateSynthesis(node.st, 3).dual_complex_valid()
        self.assertTrue(ok, "and what it left is a valid manifold")

    def test_stage_one_commits_vertex_removing_moves_with_everything_pinned(self):
        """The move gate no longer consults pinning — this is the path that had it.

        `applyMoveSpecification` used to reject any move that left a pinned vertex
        no longer live, BEFORE consulting `dual_complex_valid`. Two identical drives,
        one with every vertex pinned, must now reach the same topology: pinning is
        not an input to the gate.
        """
        results = {}
        for label, pin_everything in (("free", False), ("pinned", True)):
            node = _node(precone=4, seed=3)
            if pin_everything:
                node.declare_pinned_region("all", set(_vertex_ids(node.st)))
            before = set(_vertex_ids(node.st))
            node.run_stage1(max_steps=30, n_candidate_moves=8)
            results[label] = (sorted(_cells(node.st)),
                              sorted(before - set(_vertex_ids(node.st))))

        # Non-vacuity: the drive must actually remove vertices, or a gate on removal
        # would have had nothing to block and the comparison would prove nothing.
        self.assertTrue(results["free"][1],
                        "the drive must commit vertex-removing moves for this to test "
                        "anything")
        self.assertEqual(results["free"], results["pinned"],
                         "pinning every vertex must not change which moves commit")

    def test_pinning_does_not_change_the_verdict_either_way(self):
        # The same cone-out, with and without the pin declared, must agree — pinning
        # is not an input to the gate at all.
        unpinned = _node(precone=5, seed=2)
        cell, removed = self._cell_whose_removal_is_valid(unpinned)
        self.assertIsNotNone(cell)
        verdict_unpinned, _ = cob.SurgicalCone(unpinned.st).cone_out(cell)

        pinned = _node(precone=5, seed=2)
        pinned.declare_pinned_region("doomed", set(removed))
        verdict_pinned, _ = cob.SurgicalCone(pinned.st).cone_out(cell)

        self.assertEqual(verdict_unpinned, verdict_pinned)

    def test_an_invalid_result_is_rejected_with_a_manifold_reason(self):
        # Removing the only top cell would drop the complex dimension: the manifold
        # gate refuses, and its reason is about the complex, not about pinning.
        node = _node()
        cells = _cells(node.st)
        self.assertEqual(len(cells), 1, "the bare seed has exactly one top cell")
        node.declare_pinned_region("all", set(_vertex_ids(node.st)))

        cone = cob.SurgicalCone(node.st)
        accepted, why = cone.cone_out(list(cells[0]))
        self.assertFalse(accepted)
        self.assertTrue(why and why != "ok", "a rejection must name its reason")
        self.assertNotIn("pinned", why.lower(),
                         f"rejection must cite the complex, not pinning, got: {why}")

    def test_the_complex_is_still_a_valid_manifold_after_a_rejected_move(self):
        node = _node()
        node.declare_pinned_region("all", set(_vertex_ids(node.st)))
        cells = _cells(node.st)
        cob.SurgicalCone(node.st).cone_out(list(cells[0]))
        ok, _ = cob.EigenstateSynthesis(node.st, 3).dual_complex_valid()
        self.assertTrue(ok, "a rejected move must roll back to a valid complex")


if __name__ == "__main__":
    unittest.main()
