# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""A pinned region carries its own injected objective (#837).

Pinning says WHICH cells are held; a pinned-region objective says WHAT they are
held to. The two roles are independent — an objective scoped to a region never
learns whether that region's coordinates are frozen — and keeping them apart is
what lets a boundary be declared without target-conditioning the bulk geometry.

The scoring rules these tests pin, all of them decided rather than derived:

  * the objective is computed over the ENTIRE cobordism; there is no partition
    into separately-scored halves;
  * the bulk objective scores everything INCLUDING the pinned interior, and a
    pinned objective ADDS its term on top, so a boundary-interior edge
    contributes to both — additive by design, not double-counting;
  * whether the straddling edges (exactly one endpoint in the region) are
    scored is a DECLARED property of the objective, which the engine reads
    rather than deciding by role;
  * with no pinned objective supplied, the run is BIT-IDENTICAL to the
    single-objective run that existed before this.
"""
import cmath
import unittest

import tessera

cob = tessera.cobordism

_DIM = 4

_IN = [[complex(1.0, 0.0)]]
_OUT = [[complex(1.0, 0.0)]]

_REGION = "boundary"


def _perturbed_delta4():
    """A single Δ⁴ with a NON-uniform metric.

    The uniform ℓ²=1 seed is already stationary, so every stationarity term
    would be zero on it and a comparison between scopes would pass by reading
    0.0 twice. Perturbing gives the objective something to measure.
    """
    sig = tessera.Signature(_DIM, tessera.Lorentzian)
    st = tessera.Spacetime(tessera.Metric(True, sig), tessera.CDT, 1.0, 1.0,
                           tessera.PREFERRED, tessera.SolidSimplex(_DIM))
    st.build()
    for index, edge in enumerate(st.get_edge_list().to_vector()):
        edge.set_length(cmath.sqrt(complex(0.8 + 0.13 * (index % 5))))
    return st


def _node(host=None):
    return cob.MultiCobordism(host if host is not None else _perturbed_delta4(),
                              _IN, _OUT, degrees=[3], gamma=1.0, seed=1)


def _vertex_ids(st):
    return sorted({v.get_id() for s in st.get_top_simplices()
                   for v in s.get_vertices()})


def _edge_lengths(st):
    lengths = {}
    for edge in st.get_edge_list().to_vector():
        a, b = edge.get_source().get_id(), edge.get_target().get_id()
        lengths[(min(a, b), max(a, b))] = edge.get_length()
    return lengths


def _cells(st):
    return sorted(tuple(sorted(v.get_id() for v in s.get_vertices()))
                  for s in st.get_top_simplices())


def _scope_for(node, region_name, straddling):
    """A scope naming a DECLARED region, with a declared straddling rule.

    The handle can only come from `region_handle`, so a mis-spelling raises here
    rather than producing a scope that silently matches nothing.
    """
    scope = cob.ObjectiveScope()
    scope.region = node.region_handle(region_name)
    scope.includes_straddling_edges = straddling
    return scope


def _scoped(objective, node, region_name, straddling):
    """Point `objective` at a declared region with a declared straddling rule."""
    objective.set_scope(_scope_for(node, region_name, straddling))
    return objective


def _half_region(node):
    """Three of the five vertices, so the region has both interior and
    straddling edges — the configuration that makes the straddling declaration
    observable at all."""
    ids = _vertex_ids(node.st)
    return {ids[0], ids[1], ids[2]}


class RegionEdgePartitionTest(unittest.TestCase):
    """The fixture itself must exercise the distinction, or the straddling
    tests below would compare two identical sums and pass for the wrong
    reason."""

    def test_the_region_has_both_interior_and_straddling_edges(self):
        node = _node()
        region = _half_region(node)
        interior, straddling = 0, 0
        for edge in node.st.get_edge_list().to_vector():
            inside = ((edge.get_source().get_id() in region) +
                      (edge.get_target().get_id() in region))
            if inside == 2:
                interior += 1
            elif inside == 1:
                straddling += 1
        self.assertGreater(interior, 0, "no interior edge to score")
        self.assertGreater(straddling, 0, "no straddling edge to declare on")


class SingleObjectiveIsUnchangedTest(unittest.TestCase):
    """With no pinned objective supplied, nothing about the run changes."""

    def test_a_fresh_node_has_no_pinned_objective(self):
        self.assertIsNone(_node().pinned_objective)

    def test_the_objective_is_bit_identical_without_a_pinned_objective(self):
        plain, declared = _node(), _node()
        declared.declare_pinned_region(_REGION, _half_region(declared))
        for node in (plain, declared):
            node.set_objective(cob.JointStationarityObjective())
        # Declaring a region but no pinned objective must not move the number:
        # pinning names a region, and naming one scores nothing by itself.
        self.assertEqual(plain.objective(), declared.objective())

    def test_the_contribution_list_holds_the_bulk_alone(self):
        node = _node()
        node.set_objective(cob.JointStationarityObjective())
        contributions = node.objective_contributions
        self.assertEqual(len(contributions), 1)
        self.assertEqual(contributions[0].region_name, "")

    def test_relaxation_is_bit_identical_without_a_pinned_objective(self):
        """Setting a pinned objective and clearing it leaves no trace.

        Note what this deliberately does NOT compare: a node with a region
        declared against one without. Those relax differently and should —
        declaring a region freezes its edges, which is pinning's OTHER role and
        has nothing to do with scoring. Holding the region fixed in both arms
        isolates the only thing this ticket changes, which is whether a second
        objective is in force.
        """
        never, cleared = _node(), _node()
        for node in (never, cleared):
            node.set_objective(cob.JointStationarityObjective())
            node.declare_pinned_region(_REGION, _half_region(node))
        cleared.set_pinned_objective(
            _scoped(cob.JointStationarityObjective(), cleared, _REGION, False))
        cleared.clear_pinned_objective()
        for node in (never, cleared):
            node.run_stage2(beta=1.0, max_iters=6, alpha0=0.05)
        self.assertEqual(_edge_lengths(never.st), _edge_lengths(cleared.st))

    def test_clearing_a_pinned_objective_restores_the_single_objective(self):
        node = _node()
        node.set_objective(cob.JointStationarityObjective())
        node.declare_pinned_region(_REGION, _half_region(node))
        before = node.objective()
        node.set_pinned_objective(
            _scoped(cob.JointStationarityObjective(), node, _REGION, False))
        self.assertNotEqual(node.objective(), before)
        node.clear_pinned_objective()
        self.assertIsNone(node.pinned_objective)
        self.assertEqual(node.objective(), before)


class AdditiveScoringTest(unittest.TestCase):
    """The bulk scores everything including the pinned interior; the pinned
    objective adds on top."""

    def _prepared(self, straddling=False):
        node = _node()
        node.set_objective(cob.JointStationarityObjective())
        node.declare_pinned_region(_REGION, _half_region(node))
        bulk_only = node.objective()
        node.set_pinned_objective(
            _scoped(cob.JointStationarityObjective(), node, _REGION,
                    straddling))
        return node, bulk_only

    def test_both_contributions_are_separately_visible(self):
        node, _ = self._prepared()
        contributions = node.objective_contributions
        self.assertEqual(len(contributions), 2)
        self.assertEqual(contributions[0].region_name, "")
        self.assertEqual(contributions[1].region_name, _REGION)
        # A reader must be able to tell where descent came from, so neither
        # contribution may be silently zero on a fixture that has work to do.
        self.assertGreater(contributions[0].terms.regge_stationarity, 0.0)
        self.assertGreater(contributions[1].terms.regge_stationarity, 0.0)

    def test_the_contributions_sum_to_the_objective_exactly(self):
        node, _ = self._prepared()
        contributions = node.objective_contributions
        terms = node.objective_terms()
        self.assertEqual(
            sum(c.terms.regge_stationarity for c in contributions),
            terms.regge_stationarity)
        self.assertEqual(
            sum(c.terms.hodge_stationarity for c in contributions),
            terms.hodge_stationarity)

    def test_the_pinned_objective_raises_the_total(self):
        node, bulk_only = self._prepared()
        # Additive by design: the pinned region is scored twice, once by the
        # bulk seeing one coherent cobordism and once as an additional hold.
        self.assertGreater(node.objective(), bulk_only)

    def test_the_bulk_contribution_is_unchanged_by_the_pinned_one(self):
        node = _node()
        node.set_objective(cob.JointStationarityObjective())
        node.declare_pinned_region(_REGION, _half_region(node))
        bulk_before = node.objective_contributions[0].terms.regge_stationarity
        node.set_pinned_objective(
            _scoped(cob.JointStationarityObjective(), node, _REGION, False))
        self.assertEqual(
            node.objective_contributions[0].terms.regge_stationarity,
            bulk_before)


class StraddlingDeclarationTest(unittest.TestCase):
    """Whether the straddling edges count is the objective's declaration, and
    the engine honours it. Both declarations are tested, not only the
    default."""

    def _pinned_term(self, straddling):
        node = _node()
        node.set_objective(cob.JointStationarityObjective())
        node.declare_pinned_region(_REGION, _half_region(node))
        node.set_pinned_objective(
            _scoped(cob.JointStationarityObjective(), node, _REGION,
                    straddling))
        return node.objective_contributions[1].terms.regge_stationarity

    def test_excluding_straddling_edges_scores_strictly_less(self):
        excluded = self._pinned_term(False)
        included = self._pinned_term(True)
        self.assertGreater(excluded, 0.0)
        self.assertGreater(included, excluded,
                           "the straddling edges contributed nothing, so the "
                           "declaration was not honoured")

    def test_the_whole_cobordism_scope_is_not_reachable_by_declaring_all(self):
        """Including the straddling edges widens the region's border; it does
        not silently promote the objective to the whole complex."""
        node = _node()
        node.set_objective(cob.JointStationarityObjective())
        node.declare_pinned_region(_REGION, _half_region(node))
        node.set_pinned_objective(
            _scoped(cob.JointStationarityObjective(), node, _REGION, True))
        contributions = node.objective_contributions
        self.assertLess(contributions[1].terms.regge_stationarity,
                        contributions[0].terms.regge_stationarity)

    def test_a_region_with_no_scored_edge_scores_nothing(self):
        """A present-but-empty scope is not the whole cobordism.

        A single-vertex region has no interior edge, so with the straddling
        edges declared out it scores no coordinate at all. Were absent and
        empty conflated, this would silently promote the objective to scoring
        the entire complex — the largest possible value instead of zero.
        """
        node = _node()
        node.set_objective(cob.JointStationarityObjective())
        node.declare_pinned_region(_REGION, {_vertex_ids(node.st)[0]})
        node.set_pinned_objective(
            _scoped(cob.JointStationarityObjective(), node, _REGION, False))
        contributions = node.objective_contributions
        self.assertEqual(contributions[1].terms.regge_stationarity, 0.0)
        self.assertEqual(contributions[1].terms.hodge_stationarity, 0.0)


class HoldToUnitObjective(cob.CobordismObjective):
    """A Python-defined objective: `sum |z_e - 1|^2` over the edges in scope.

    Deliberately not one of the built-ins. A pinned region is most useful when
    a caller can hold it to something of their own devising, so the feature has
    to be reachable from Python rather than only from C++.
    """

    def name(self):
        return "hold_to_unit"

    def term_names(self):
        return cob.CobordismObjective.declared_term_names()

    def terms(self, context):
        terms = cob.ObjectiveTerms()
        edges = context.spacetime.get_edge_list().to_vector()
        # None means the whole cobordism; a list — even an empty one — means
        # exactly those coordinates.
        indices = (range(len(edges)) if context.scored_edges is None
                   else context.scored_edges)
        terms.regge_stationarity = sum(
            abs(edges[index].get_length() ** 2 - 1.0) ** 2 for index in indices)
        return terms

    def direction(self, context):
        return cob.ObjectiveDirection()

    def is_target_conditioned(self):
        return False


class PythonObjectiveHoldsARegionTest(unittest.TestCase):
    """A caller's own objective can hold a declared region."""

    def _held(self, straddling=False):
        node = _node()
        node.set_objective(cob.JointStationarityObjective())
        node.declare_pinned_region(_REGION, _half_region(node))
        held = HoldToUnitObjective()
        held.set_scope(_scope_for(node, _REGION, straddling))
        node.set_pinned_objective(held)
        return node

    def test_a_python_objective_contributes_under_its_own_name(self):
        contributions = self._held().objective_contributions
        self.assertEqual(contributions[1].objective_name, "hold_to_unit")
        self.assertEqual(contributions[1].region_name, _REGION)
        self.assertGreater(contributions[1].terms.regge_stationarity, 0.0)

    def test_it_scores_its_region_and_not_the_whole_complex(self):
        interior = self._held(False).objective_contributions[1]
        wider = self._held(True).objective_contributions[1]
        # Honouring the declaration is observable from Python exactly as it is
        # from C++: widening the border raises the sum.
        self.assertGreater(wider.terms.regge_stationarity,
                           interior.terms.regge_stationarity)


class CompositeBehaviourTest(unittest.TestCase):
    """Two of the branches that moved onto the objective (#836) have to be
    answered for the COMPOSITE rather than inherited from the bulk objective.
    Both answers are choices, so both are pinned here."""

    def test_target_conditioning_is_the_disjunction(self):
        """A run whose region is held to a declared state is
        target-conditioned however geometric the bulk objective is.

        A search policy asks this to find out whether the run it drives is
        unforced. Reporting the bulk alone would let it believe it was unforced
        while a target steered part of the complex.
        """
        node = _node()
        node.set_objective(cob.JointStationarityObjective())
        self.assertFalse(node.objective_is_target_conditioned)
        node.declare_pinned_region(_REGION, _half_region(node))
        node.set_pinned_objective(
            _scoped(cob.LegacyObjective(), node, _REGION, False))
        self.assertTrue(node.objective_is_target_conditioned)
        node.clear_pinned_objective()
        self.assertFalse(node.objective_is_target_conditioned)

    def test_the_reported_scalar_stays_the_composite_through_stage_one(self):
        """Stage 1 must score the scalar it reports.

        With two objectives over two scopes a localized delta differenced from
        the bulk alone would optimize a surrogate that is not the objective —
        exactly what the localized path exists to avoid — so a pinned objective
        drops the node back to global re-evaluation, which is always correct and
        merely more expensive.
        """
        node = _node()
        node.set_objective(cob.JointStationarityObjective())
        node.declare_pinned_region(_REGION, _half_region(node))
        node.set_pinned_objective(
            _scoped(cob.JointStationarityObjective(), node, _REGION, False))
        node.run_stage1(max_steps=3, n_candidate_moves=4)
        contributions = node.objective_contributions
        self.assertEqual(len(contributions), 2)
        self.assertEqual(
            sum(c.terms.regge_stationarity for c in contributions),
            node.objective_terms().regge_stationarity)


class ScopeIsIndependentOfFreezingTest(unittest.TestCase):
    """Pinning's two roles do not depend on each other."""

    def test_a_scoped_objective_reads_the_same_whether_or_not_edges_move(self):
        # The region is the same either way; what differs is that one node has
        # relaxed and the other has not. The scope declaration itself is
        # unaffected by whether the coordinates it names are frozen.
        node = _node()
        node.set_objective(cob.JointStationarityObjective())
        node.declare_pinned_region(_REGION, _half_region(node))
        pinned = _scoped(cob.JointStationarityObjective(), node, _REGION, False)
        node.set_pinned_objective(pinned)
        self.assertEqual(pinned.scope().region.name(), _REGION)
        self.assertFalse(pinned.scope().includes_straddling_edges)
        node.run_stage2(beta=1.0, max_iters=4, alpha0=0.05)
        self.assertEqual(pinned.scope().region.name(), _REGION)
        self.assertFalse(pinned.scope().includes_straddling_edges)


class PinnedObjectiveRejectionTest(unittest.TestCase):
    """A pinned objective that cannot be honoured fails loudly."""

    def test_a_null_pinned_objective_is_rejected(self):
        with self.assertRaises((ValueError, TypeError)):
            _node().set_pinned_objective(None)

    def test_an_undeclared_region_cannot_be_named(self):
        node = _node()
        node.declare_pinned_region(_REGION, _half_region(node))
        with self.assertRaises(ValueError) as caught:
            node.region_handle("boundry")
        self.assertIn("boundry", str(caught.exception))

    def test_a_cleared_region_makes_its_objective_unacceptable(self):
        node = _node()
        node.declare_pinned_region(_REGION, _half_region(node))
        pinned = _scoped(cob.JointStationarityObjective(), node, _REGION, False)
        node.clear_pinned_regions()
        with self.assertRaises(ValueError) as caught:
            node.set_pinned_objective(pinned)
        self.assertIn(_REGION, str(caught.exception))


class RelaxationUnderTwoObjectivesTest(unittest.TestCase):
    """The line search descends the composite scalar, not the bulk alone."""

    def test_the_fixture_actually_relaxes(self):
        """Guard against every relaxation assertion below passing vacuously.

        If stage 2 moved nothing, "the geometry differs" and "the objective did
        not rise" would both hold on a complex that never budged. Measured: 7 of
        10 edges move, and the 3 that do not are exactly the region's interior
        edges, which pinning freezes.
        """
        node = _node()
        node.set_objective(cob.JointStationarityObjective())
        node.declare_pinned_region(_REGION, _half_region(node))
        before = dict(_edge_lengths(node.st))
        node.run_stage2(beta=1.0, max_iters=8, alpha0=0.05)
        after = _edge_lengths(node.st)
        moved = [key for key in before if before[key] != after[key]]
        self.assertEqual(len(moved), 7)
        region = _half_region(node)
        for key in before:
            if key[0] in region and key[1] in region:
                self.assertNotIn(key, moved, "a pinned edge moved")

    def test_relaxation_lowers_the_composite_objective(self):
        node = _node()
        node.set_objective(cob.JointStationarityObjective())
        node.declare_pinned_region(_REGION, _half_region(node))
        node.set_pinned_objective(
            _scoped(cob.JointStationarityObjective(), node, _REGION, False))
        before = node.objective()
        node.run_stage2(beta=1.0, max_iters=8, alpha0=0.05)
        self.assertLessEqual(node.objective(), before)

    def test_a_pinned_objective_changes_where_the_geometry_goes(self):
        """The pinned objective is not decorative: it moves the relaxation."""
        plain, held = _node(), _node()
        for node in (plain, held):
            node.set_objective(cob.JointStationarityObjective())
            node.declare_pinned_region(_REGION, _half_region(node))
        held.set_pinned_objective(
            _scoped(cob.JointStationarityObjective(), held, _REGION, False))
        for node in (plain, held):
            node.run_stage2(beta=1.0, max_iters=8, alpha0=0.05)
        self.assertNotEqual(_edge_lengths(plain.st), _edge_lengths(held.st))


if __name__ == "__main__":
    unittest.main()
