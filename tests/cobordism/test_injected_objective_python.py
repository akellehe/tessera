# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""#834 -- the objective is an injected specification object, not an enum.

The engine holds a `CobordismObjective` and calls through it. These tests pin
the two properties that make that safe: the injected objective reproduces the
functional it replaced EXACTLY, and the no-feedback firewall survives the move
from a static function to an object.
"""

import cmath
import math
import unittest

import tessera as T


cob = T.cobordism


def _complex_sphere4():
    """The same fixture the objective-selection suite scores, so the values
    here are comparable with the ones that suite already pins."""
    st = T.Spacetime(T.Metric(True, T.Signature(4, T.Lorentzian)),
                     T.CDT, 1.0, 1.0, T.PREFERRED,
                     T.SimplexBoundarySphere(4))
    st.build()
    for index, edge in enumerate(st.getEdgeList().toVector()):
        z = complex(1.0 + 0.019 * (index % 5),
                    0.011 * (1 + index % 4))
        edge.setLength(cmath.sqrt(z))
    return st


def _first_vertices(st, count=3):
    """A small vertex set to declare a region over."""
    return {vertex.getId()
            for vertex in st.getVertexList().toVector()[:count]}


def _node(st, degree=3, gamma=0.0):
    return cob.MultiCobordism(st, [], [], degrees=[degree], gamma=gamma,
                              seed=7)


class InjectedObjectiveIsTheOneScoredTest(unittest.TestCase):
    """The engine scores whatever it is handed, and says which."""

    def test_each_builtin_reports_its_own_name(self):
        node = _node(_complex_sphere4())
        for objective, expected in (
                (cob.JointStationarityObjective(), "joint_stationarity"),
                (cob.LegacyObjective(), "legacy"),
                (cob.MediatedCorrespondenceObjective(),
                 "mediated_correspondence")):
            with self.subTest(objective=expected):
                node.set_objective(objective)
                self.assertEqual(node.objective_name, expected)

    def test_an_injected_objective_replaces_the_previous_one(self):
        node = _node(_complex_sphere4())
        node.set_objective(cob.LegacyObjective())
        self.assertEqual(node.objective_name, "legacy")
        node.set_objective(cob.JointStationarityObjective())
        self.assertEqual(node.objective_name, "joint_stationarity")

    def test_there_is_no_second_answer_to_what_is_optimized(self):
        """The enum is retired, so nothing can report a functional that
        disagrees with the injected one. `objective_name` is the only answer."""
        node = _node(_complex_sphere4())
        self.assertFalse(hasattr(node, "objective_mode"))
        self.assertFalse(hasattr(node, "set_objective_mode"))
        self.assertFalse(hasattr(cob, "CobordismObjectiveMode"))

    def test_a_null_objective_is_refused(self):
        node = _node(_complex_sphere4())
        with self.assertRaises((ValueError, TypeError)):
            node.set_objective(None)

    def test_target_conditioning_is_declared_not_inferred(self):
        self.assertFalse(
            cob.JointStationarityObjective().is_target_conditioned())
        self.assertTrue(cob.LegacyObjective().is_target_conditioned())
        self.assertTrue(
            cob.MediatedCorrespondenceObjective().is_target_conditioned())

    def test_only_target_conditioned_objectives_ask_for_the_residual(self):
        # A purely geometric objective never pays for r_U.
        self.assertFalse(
            cob.JointStationarityObjective().needs_register_residual())
        self.assertTrue(cob.LegacyObjective().needs_register_residual())
        self.assertTrue(
            cob.MediatedCorrespondenceObjective().needs_register_residual())


class ExactnessTest(unittest.TestCase):
    """The injected objective is the SAME functional, to the bit."""

    def test_joint_hodge_term_equals_the_primitive_it_is_built_from(self):
        st = _complex_sphere4()
        node = _node(st)
        node.set_objective(cob.JointStationarityObjective())
        # The Hodge degrees are DECLARED (#859) and no longer follow the
        # register degrees, so the degree this reference is built at is stated
        # here rather than inherited from the node's construction. Declaring
        # the degree the term was scored at before reproduces it to the bit,
        # which is the guard that decoupling moved the CONFIGURATION and not
        # the term.
        node.set_hodge_degrees([3])
        terms = node.objective_terms()
        reference = (node.hodge_entropy_weight *
                     cob.HodgeLaplacian(st).spectralEntropyGradientNorm(
                         3, node.hodge_entropy_phase_mode))
        # Bit-identical, not merely close: the objective computes this from the
        # same primitive the engine used before it became injectable.
        self.assertEqual(terms.hodge_stationarity, reference)

    def test_joint_regge_term_equals_the_engine_gradient_norm(self):
        st = _complex_sphere4()
        node = _node(st)
        node.set_objective(cob.JointStationarityObjective())
        terms = node.objective_terms()
        reference = (node.regge_weight *
                     cob.MultiCobordism.regge_action_gradient(st))
        self.assertEqual(terms.regge_stationarity, reference)

    def test_the_scalar_is_the_sum_of_the_declared_terms(self):
        st = _complex_sphere4()
        node = _node(st)
        for mode in (cob.JointStationarityObjective(),
                     cob.LegacyObjective(),
                     cob.MediatedCorrespondenceObjective()):
            with self.subTest(mode=str(mode)):
                node.set_objective(mode)
                terms = node.objective_terms()
                self.assertEqual(node.objective(),
                                 cob.CobordismObjective.total(terms))
                # And the static collapse agrees with the engine's own.
                self.assertEqual(cob.MultiCobordism.objective_of(terms),
                                 cob.CobordismObjective.total(terms))

    def test_joint_stationarity_carries_no_register_residual(self):
        st = _complex_sphere4()
        node = _node(st)
        node.set_objective(cob.JointStationarityObjective())
        # Not merely zero-valued: the term is structurally absent because the
        # objective never asks the engine to compute it.
        self.assertEqual(node.objective_terms().register_residual, 0.0)

    def test_stage_two_still_descends_under_the_injected_objective(self):
        st = _complex_sphere4()
        node = _node(st)
        node.set_objective(cob.JointStationarityObjective())
        before = node.objective()
        node.run_stage2(1.0, 12, 0.05)
        after = node.objective()
        self.assertLessEqual(after, before)
        self.assertTrue(math.isfinite(after))


class FirewallTest(unittest.TestCase):
    """The no-feedback guarantee survives the move to an injected object.

    It used to be mechanical: `objective_of` was static, so it had no `this`
    and therefore no pointer through which to reach a member holding an
    analysis read. An injected objective HAS a `this`, so the guarantee moves
    to the input side -- `ObjectiveContext` is plain data that leads nowhere.
    """

    #: Every analysis product `runRecursiveAnalysis` writes. None of it may be
    #: nameable from an objective's declared inputs or outputs.
    ANALYSIS_WORDS = ("component", "cluster", "fiber", "fibre", "transport",
                      "amplitude", "color", "colour", "charge", "flavor",
                      "flavour", "exchange", "spin", "certificate", "verdict",
                      "holonomy", "anchor", "betti", "crossing", "baryon",
                      "quark", "proton", "wilson", "winding")

    def test_no_analysis_quantity_is_reachable_from_an_objective_input(self):
        for name in cob.ObjectiveContext.input_names():
            for word in self.ANALYSIS_WORDS:
                with self.subTest(field=name, word=word):
                    self.assertNotIn(word, name.lower())

    def test_no_analysis_quantity_appears_among_the_declared_terms(self):
        for name in cob.CobordismObjective.declared_term_names():
            for word in self.ANALYSIS_WORDS:
                with self.subTest(term=name, word=word):
                    self.assertNotIn(word, name.lower())

    def test_the_context_is_the_complete_input_list(self):
        # The firewall is only checkable if the enumeration is the whole story,
        # so pin the list itself. A field added without updating this test is
        # a field nobody audited.
        # `scored_edges` (#837) is the objective's declared scope resolved to
        # edge INDICES. Audited on the same terms as every other field: it is
        # geometry, it names coordinates rather than carrying values, and
        # nothing reachable from an edge index is a component, fiber,
        # transport, amplitude, colour, charge, flavour, exchange, spin
        # certificate or verdict.
        # `hodge_degrees` and `hodge_degree_weights` (#859) declare which
        # Laplacian degrees the entropy term is summed over and how each is
        # weighted. Audited on the same terms: both are declared configuration
        # -- a list of integers and a list of reals chosen by the caller -- and
        # neither is, or leads to, an analysis product.
        # `connection_entropy_weight` (#853) declares the weight on the
        # connection-entropy stationarity term, the one term with a gradient in
        # the connection phase. Audited on the same terms: it is a configured
        # real chosen by the caller, exactly like the Regge and Hodge weights
        # beside it, and nothing reachable from it is an analysis product.
        self.assertEqual(
            cob.ObjectiveContext.input_names(),
            ["spacetime", "region", "scored_edges", "region_targets",
             "register_degrees", "hodge_degrees", "hodge_degree_weights",
             "regge_weight", "hodge_entropy_weight",
             "connection_entropy_weight", "gamma",
             "carried_state_energy_weight", "einstein_hilbert",
             "hodge_entropy_phase_mode", "register_residual",
             "carried_state_energy", "moment_stiffness_weight",
             "moment_stiffness_degrees", "moment_stiffness_coefficients",
             "moment_stiffness_reference"])
        # The spectral-moment stiffness (#1183) adds a configured weight, the
        # configured degrees and order weights, and the local spectral moments
        # of the carrier -- geometry, computed from the edge lengths when the
        # stiffness is declared. None of them is, or leads to, an analysis
        # product.

    def test_the_declared_term_list_is_unchanged(self):
        self.assertEqual(
            cob.CobordismObjective.declared_term_names(),
            ["regge_stationarity", "hodge_stationarity",
             "connection_stationarity", "register_residual",
             "action_magnitude", "carried_state_energy", "moment_stiffness"])
        # The engine's own list is the same list, so a record stays comparable.
        self.assertEqual(cob.MultiCobordism.objective_term_names(),
                         cob.CobordismObjective.declared_term_names())

    def test_an_objective_holds_no_route_back_to_the_engine(self):
        # An objective is constructible and usable with NO node in existence.
        # If the interface required a MultiCobordism -- or anything reachable
        # from one -- this could not be written at all, which is the property
        # under test: analysis access is impossible, not merely unused.
        objective = cob.JointStationarityObjective()
        self.assertEqual(objective.name(), "joint_stationarity")
        self.assertEqual(objective.term_names(),
                         cob.CobordismObjective.declared_term_names())

    def test_the_static_collapse_takes_no_instance(self):
        # `total` remains static: the step from a decomposition to the number
        # the optimizer compares still has no `this`.
        terms = cob.MultiCobordism.ObjectiveTerms()
        terms.regge_stationarity = 2.0
        terms.hodge_stationarity = 0.5
        self.assertEqual(cob.CobordismObjective.total(terms), 2.5)


class ScopeTest(unittest.TestCase):
    """An objective declares what it references; declaring nothing is
    declaring everything, and the engine honours the declaration."""

    def test_declaring_nothing_means_the_whole_cobordism(self):
        for objective in (cob.JointStationarityObjective(),
                          cob.LegacyObjective(),
                          cob.MediatedCorrespondenceObjective()):
            with self.subTest(objective=objective.name()):
                scope = objective.scope()
                self.assertTrue(scope.is_whole_cobordism())
                self.assertTrue(scope.region.is_whole_cobordism())
                self.assertEqual(scope.region.name(), "")

    def test_a_whole_cobordism_scope_has_no_straddling_question(self):
        # Nothing straddles when the scope is everything, so the flag is moot
        # and the single-objective path is the one that ran before.
        self.assertTrue(
            cob.JointStationarityObjective().scope().includes_straddling_edges)

    def test_the_scope_defaults_to_the_whole_cobordism(self):
        scope = cob.ObjectiveScope()
        self.assertTrue(scope.is_whole_cobordism())
        self.assertTrue(scope.region.is_whole_cobordism())
        self.assertEqual(scope.region.name(), "")

    def test_a_handle_can_only_be_minted_from_a_declared_region(self):
        st = _complex_sphere4()
        node = _node(st)
        # Nothing declared yet: the lookup fails BY NAME rather than returning
        # a handle that would silently match nothing.
        with self.assertRaises(ValueError) as caught:
            node.region_handle("shell")
        self.assertIn("shell", str(caught.exception))

        node.declare_pinned_region("shell", _first_vertices(st))
        handle = node.region_handle("shell")
        self.assertFalse(handle.is_whole_cobordism())
        self.assertEqual(handle.name(), "shell")

        # A near miss is still refused, which is the whole point of the type.
        with self.assertRaises(ValueError):
            node.region_handle("shel")

    def test_scope_does_not_depend_on_whether_the_region_is_frozen(self):
        # Pinning names a region AND constrains relaxation; neither justifies
        # the other. A scope is a reference, not a statement about motion.
        st = _complex_sphere4()
        node = _node(st)
        node.declare_pinned_region("shell", _first_vertices(st))
        scope = cob.ObjectiveScope()
        scope.region = node.region_handle("shell")
        self.assertFalse(scope.is_whole_cobordism())
        # The handle reads the same regardless of whether those coordinates
        # move; nothing about freezing is encoded in the reference.
        self.assertEqual(scope.region.name(), "shell")


class NamedConstantsTest(unittest.TestCase):
    """Identifiers are named constants, not literals repeated at each site."""

    def test_objective_names_come_from_the_constants(self):
        self.assertEqual(cob.JointStationarityObjective().name(),
                         cob.ObjectiveName.JOINT_STATIONARITY)
        self.assertEqual(cob.LegacyObjective().name(), cob.ObjectiveName.LEGACY)
        self.assertEqual(cob.MediatedCorrespondenceObjective().name(),
                         cob.ObjectiveName.MEDIATED_CORRESPONDENCE)

    def test_the_term_list_is_assembled_from_the_constants(self):
        self.assertEqual(
            cob.CobordismObjective.declared_term_names(),
            [cob.ObjectiveTermName.REGGE_STATIONARITY,
             cob.ObjectiveTermName.HODGE_STATIONARITY,
             cob.ObjectiveTermName.CONNECTION_STATIONARITY,
             cob.ObjectiveTermName.REGISTER_RESIDUAL,
             cob.ObjectiveTermName.ACTION_MAGNITUDE,
             cob.ObjectiveTermName.CARRIED_STATE_ENERGY,
             cob.ObjectiveTermName.MOMENT_STIFFNESS])


if __name__ == "__main__":
    unittest.main()
