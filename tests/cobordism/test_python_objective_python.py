# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""#841 -- a Python subclass can supply the functional the engine descends.

The trampoline widens WHO may write an objective. These tests pin that it does
not widen WHAT one can read: a Python objective sits behind the same
no-feedback firewall as a C++ one, and still cannot fabricate a region handle.
"""

import cmath
import gc
import unittest

import numpy as np

import tessera as T


cob = T.cobordism


def _complex_sphere4():
    """The fixture the injected-objective suite scores, so values here are
    comparable with the ones that suite already pins."""
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
    return {vertex.getId()
            for vertex in st.getVertexList().toVector()[:count]}


def _node(st, degree=3, gamma=0.0):
    return cob.MultiCobordism(st, [], [], degrees=[degree], gamma=gamma,
                              seed=7)


class ConstantObjective(cob.CobordismObjective):
    """The smallest complete objective: a declared constant on one term.

    Deliberately trivial. What is under test is the crossing, not the physics,
    and a constant makes the engine's reading of it unambiguous.
    """

    NAME = "python_constant"

    def __init__(self, value=1.5):
        super().__init__()
        self.value = value
        self.contexts_seen = 0

    def name(self):
        return self.NAME

    def term_names(self):
        return cob.CobordismObjective.declared_term_names()

    def terms(self, context):
        self.contexts_seen += 1
        out = cob.ObjectiveTerms()
        out.regge_stationarity = self.value
        return out

    def direction(self, context):
        out = cob.ObjectiveDirection()
        # A constant has no descent direction; a zero ascent is the honest
        # answer rather than a fabricated one.
        out.ascent = np.zeros(context.edge_count, dtype=complex)
        out.baseline = self.value
        out.baseline_computed = True
        return out

    def is_target_conditioned(self):
        return False


class PythonObjectiveIsCalledTest(unittest.TestCase):
    """The engine calls through a Python objective exactly as through a C++
    one, and reports it as the functional in force."""

    def test_a_python_objective_can_be_injected_and_is_named(self):
        node = _node(_complex_sphere4())
        node.set_objective(ConstantObjective())
        self.assertEqual(node.objective_name, ConstantObjective.NAME)

    def test_the_engine_reads_the_python_decomposition(self):
        node = _node(_complex_sphere4())
        objective = ConstantObjective(2.25)
        node.set_objective(objective)
        terms = node.objective_terms()
        self.assertEqual(terms.regge_stationarity, 2.25)
        # Every other declared slot stays at its zero; a Python objective
        # records into the same enumerable slots as any other.
        self.assertEqual(terms.hodge_stationarity, 0.0)
        self.assertEqual(terms.register_residual, 0.0)
        self.assertGreater(objective.contexts_seen, 0)

    def test_the_scalar_is_the_sum_of_the_python_terms(self):
        node = _node(_complex_sphere4())
        node.set_objective(ConstantObjective(3.5))
        self.assertEqual(
            cob.CobordismObjective.total(node.objective_terms()), 3.5)

    def test_declared_term_names_reach_the_engine(self):
        node = _node(_complex_sphere4())
        node.set_objective(ConstantObjective())
        self.assertEqual(node.objective_spec.term_names(),
                         cob.CobordismObjective.declared_term_names())

    def test_a_python_objective_declares_whether_it_is_target_conditioned(self):
        node = _node(_complex_sphere4())
        node.set_objective(ConstantObjective())
        self.assertFalse(node.objective_is_target_conditioned)


class PythonObjectiveDefaultsTest(unittest.TestCase):
    """The three virtuals with C++ defaults fall back when a subclass is
    silent, so a minimal objective stays minimal."""

    def test_scope_defaults_to_the_whole_cobordism(self):
        scope = ConstantObjective().scope()
        self.assertTrue(scope.is_whole_cobordism())
        self.assertEqual(scope.region.name(), "")

    def test_register_residual_is_not_requested_by_default(self):
        self.assertFalse(ConstantObjective().needs_register_residual())

    def test_the_numerical_residual_weight_defaults_to_zero(self):
        context = cob.ObjectiveContext()
        self.assertEqual(
            ConstantObjective().numerical_register_residual_weight(context),
            0.0)

    def test_an_overridden_default_is_honoured(self):
        class AsksForTheResidual(ConstantObjective):
            def needs_register_residual(self):
                return True

        node = _node(_complex_sphere4())
        node.set_objective(AsksForTheResidual())
        self.assertTrue(node.objective_spec.needs_register_residual())


class FirewallTest(unittest.TestCase):
    """A Python objective reads what it is handed and nothing else. The
    property is impossibility, not restraint."""

    def test_a_python_objective_holds_no_route_back_to_the_engine(self):
        # Constructible and usable with NO node in existence -- the same
        # property #834 pins for a C++ objective. If the interface required a
        # MultiCobordism, or anything reachable from one, this could not be
        # written at all.
        objective = ConstantObjective(4.0)
        self.assertEqual(objective.name(), "python_constant")
        self.assertEqual(objective.term_names(),
                         cob.CobordismObjective.declared_term_names())
        terms = objective.terms(cob.ObjectiveContext())
        self.assertEqual(terms.regge_stationarity, 4.0)

    def test_the_context_a_python_objective_receives_is_the_declared_list(self):
        seen = {}

        class Recording(ConstantObjective):
            def terms(self, context):
                seen["fields"] = [field for field in cob.ObjectiveContext
                                  .input_names()
                                  if hasattr(context, field)]
                seen["has_node"] = any(
                    "cobordism" in type(getattr(context, field)).__name__
                    .lower()
                    for field in cob.ObjectiveContext.input_names()
                    if hasattr(context, field))
                return super().terms(context)

        node = _node(_complex_sphere4())
        node.set_objective(Recording())
        node.objective_terms()
        # Every declared input is readable from Python...
        self.assertEqual(seen["fields"], cob.ObjectiveContext.input_names())
        # ...and none of them is a node.
        self.assertFalse(seen["has_node"])

    def test_the_context_carries_geometry_but_no_analysis(self):
        # Geometry access is intended: an objective must read what it scores.
        # What must be absent is any analysis product.
        seen = {}

        class Recording(ConstantObjective):
            def terms(self, context):
                seen["spacetime"] = context.spacetime is not None
                seen["attrs"] = set(dir(context))
                return super().terms(context)

        node = _node(_complex_sphere4())
        node.set_objective(Recording())
        node.objective_terms()
        self.assertTrue(seen["spacetime"])
        for forbidden in ("clusters", "fibers", "transports", "verdict",
                          "certificates", "colour", "color", "charge",
                          "flavour", "flavor", "spin", "exchange"):
            self.assertNotIn(forbidden, seen["attrs"])

    def test_the_static_collapse_still_takes_no_instance(self):
        terms = cob.ObjectiveTerms()
        terms.regge_stationarity = 2.0
        terms.hodge_stationarity = 0.5
        self.assertEqual(cob.CobordismObjective.total(terms), 2.5)


class RegionHandleTest(unittest.TestCase):
    """A Python subclass cannot spell a region into existence."""

    def test_a_handle_cannot_be_constructed_with_a_name(self):
        # The only public construction is the whole cobordism. A name is not
        # an accepted argument, so `RegionHandle("boundry")` cannot be written.
        with self.assertRaises(TypeError):
            cob.RegionHandle("shell")
        self.assertTrue(cob.RegionHandle().is_whole_cobordism())

    def test_an_undeclared_region_raises_by_name(self):
        st = _complex_sphere4()
        node = _node(st)
        node.declare_pinned_region("shell", _first_vertices(st))
        # The near miss is the whole point: it must fail loudly rather than
        # silently scoping to everything.
        with self.assertRaises(Exception) as caught:
            node.region_handle("shel")
        self.assertIn("shel", str(caught.exception))

    def test_a_declared_region_mints_a_handle_a_python_objective_can_use(self):
        st = _complex_sphere4()
        node = _node(st)
        node.declare_pinned_region("shell", _first_vertices(st))
        handle = node.region_handle("shell")

        class Scoped(ConstantObjective):
            def scope(self):
                out = cob.ObjectiveScope()
                out.region = handle
                out.includes_straddling_edges = False
                return out

        scope = Scoped().scope()
        self.assertFalse(scope.is_whole_cobordism())
        self.assertEqual(scope.region.name(), "shell")
        self.assertFalse(scope.includes_straddling_edges)


class LifetimeTest(unittest.TestCase):
    """The node keeps its objective alive; dropping the caller's reference
    must not leave the engine descending through a collected object."""

    def test_the_objective_survives_its_last_python_reference(self):
        node = _node(_complex_sphere4())
        node.set_objective(ConstantObjective(5.5))
        gc.collect()
        gc.collect()
        # The only remaining reference is the node's.
        self.assertEqual(node.objective_terms().regge_stationarity, 5.5)
        self.assertEqual(node.objective_name, "python_constant")

    def test_the_objective_survives_repeated_collection_under_use(self):
        node = _node(_complex_sphere4())
        node.set_objective(ConstantObjective(0.75))
        for _ in range(3):
            gc.collect()
            self.assertEqual(node.objective_terms().regge_stationarity, 0.75)


class QuadraticObjective(cob.CobordismObjective):
    """Sum |z_e - 1|^2 over the complex's edges.

    A real functional with an exact analytic direction, so a run against it has
    a known answer: every squared length goes to one and the objective goes to
    zero. That makes an end-to-end descent checkable rather than merely
    observable.
    """

    def __init__(self):
        super().__init__()
        self.direction_calls = 0

    @staticmethod
    def _squared_lengths(spacetime):
        return [complex(edge.getLength()) ** 2
                for edge in spacetime.getEdgeList().toVector()]

    def name(self):
        return "python_quadratic"

    def term_names(self):
        return cob.CobordismObjective.declared_term_names()

    def terms(self, context):
        out = cob.ObjectiveTerms()
        out.regge_stationarity = sum(
            abs(z - 1.0) ** 2 for z in self._squared_lengths(context.spacetime))
        return out

    def direction(self, context):
        self.direction_calls += 1
        squared = self._squared_lengths(context.scalar.spacetime)
        out = cob.ObjectiveDirection()
        out.ascent = np.array([2.0 * (z - 1.0) for z in squared],
                              dtype=complex)
        out.baseline = sum(abs(z - 1.0) ** 2 for z in squared)
        out.baseline_computed = True
        return out

    def is_target_conditioned(self):
        return False


class EndToEndRunTest(unittest.TestCase):
    """A Python objective drives a real relaxation, not just a scalar read."""

    def test_stage_two_descends_a_python_objective_to_its_minimum(self):
        node = _node(_complex_sphere4())
        objective = QuadraticObjective()
        node.set_objective(objective)

        before = cob.CobordismObjective.total(node.objective_terms())
        trace = node.run_stage2(beta=1.0, max_iters=25)
        after = cob.CobordismObjective.total(node.objective_terms())

        # The engine really called through Python for its search direction --
        # through an entry point that releases the GIL, so this also pins that
        # the trampoline re-enters Python safely rather than deadlocking.
        self.assertGreater(objective.direction_calls, 0)
        self.assertEqual(len(trace), objective.direction_calls)
        # This functional's minimum is exactly zero, at z_e = 1 for every edge.
        self.assertLess(after, before)
        self.assertLess(after, 1e-20)

    def test_the_run_stamps_the_python_objectives_name(self):
        node = _node(_complex_sphere4())
        node.set_objective(QuadraticObjective())
        node.run_stage2(beta=1.0, max_iters=5)
        self.assertEqual(node.objective_name, "python_quadratic")

    def test_the_engine_reads_geometry_the_python_objective_moved(self):
        # The descent is real: the coordinates themselves end at the minimum,
        # not merely the reported scalar.
        st = _complex_sphere4()
        node = _node(st)
        node.set_objective(QuadraticObjective())
        node.run_stage2(beta=1.0, max_iters=25)
        for edge in st.getEdgeList().toVector():
            self.assertAlmostEqual(complex(edge.getLength()) ** 2, 1.0,
                                   places=9)


class MissingOverrideTest(unittest.TestCase):
    """A pure virtual left unimplemented fails loudly."""

    def test_an_unimplemented_pure_virtual_raises(self):
        class Incomplete(cob.CobordismObjective):
            pass

        node = _node(_complex_sphere4())
        node.set_objective(Incomplete())
        with self.assertRaises(RuntimeError):
            node.objective_terms()


if __name__ == "__main__":
    unittest.main()
