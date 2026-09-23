# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""#836 -- the directed cone probes score the INJECTED objective.

Surgery is the only topology-changing mechanism the engine has. Pachner moves
are bistellar, so they preserve the PL homeomorphism type and therefore the
Betti numbers exactly, and geometric relaxation changes no topology at all.
The directed cone probes are where a higher `b_k` becomes reachable -- so what
steers them decides whether a topology change is EMERGENT (the objective in
force wanted it) or PRESCRIBED (a target-conditioned residual wanted it).

These tests pin the acceptance rule in both directions: a candidate that lowers
the injected objective is kept, one that does not is rolled back, and an
objective indifferent to topology commits nothing at all.

On non-vacuity: the interesting branch is the one where surgery COMMITS, and
the bare boundary-of-a-5-simplex host is too small to reach it -- no cone-out
there ever opens a hole, so a test written on it would pass while exercising
nothing. Every test that needs a commit therefore runs on a refined host and
ASSERTS that the commit happened rather than tolerating either outcome.
"""

import cmath
import math
import unittest

import tessera as T


cob = T.cobordism
MC = cob.MultiCobordism

#: The register degree these fixtures declare. `JointStationarityObjective`
#: declares itself over degrees >= 1, so degree 3 is inside every built-in
#: objective's declared domain and one fixture serves all of them.
DEGREE = 3

#: Refinement steps for the host that is large enough for surgery to commit.
#: Measured: 21 cells, where a cone-out opens a hole and drops the objective by
#: about 25. Four is the smallest value that reaches a commit here.
COMMITTING_REFINEMENTS = 4

#: A trial cone-out that is rolled back restores the geometry exactly but not
#: the internal ordering of the cell container, so a functional summed over
#: cells can come back differing in the last bit. Measured at 2.2e-16 -- one
#: ULP at this scale -- while `r_U`, which is not a sum over cells, is
#: bit-stable across the same rollback. The reordering is a property of
#: `SurgicalCone::rollback` and predates objective-scored probing: the
#: trial-and-rollback sequence is untouched by this change, only the scalar
#: READ between the two calls is. It cannot manufacture a commit that matters,
#: because the probe commits on a STRICT decrease -- a candidate sitting within
#: an ULP of the base is one whose commit or refusal is immaterial either way.
ROLLBACK_ULP_TOLERANCE = 1e-14


def _rollback_tolerance(value):
    """The absolute floor above, plus four ULPs of the value: the Whitney
    pencil reads the objective through sparse factorizations whose rollback
    reorders the assembly, and a joint scalar of 35.6 came back 2 ULPs
    (1.4e-14) above itself on an uncommitted sweep, against the 1e-14 the
    diagonal path needed. A candidate within a few ULPs of the base is one
    whose commit or refusal is immaterial either way."""
    return ROLLBACK_ULP_TOLERANCE + 4.0 * math.ulp(abs(value))


def _small_sphere4():
    """The bare boundary of a 5-simplex with a mild non-uniform complex metric.

    Uniform `l^2 = 1` is a degenerate starting point -- several objective terms
    sit at an exact stationary value there -- so the metric is perturbed. Too
    small for surgery to commit, which is what makes it the right host for the
    tests that require nothing to be committed.
    """
    st = T.Spacetime(T.Metric(True, T.Signature(4, T.Lorentzian)),
                     T.CDT, 1.0, 1.0, T.PREFERRED,
                     T.SimplexBoundarySphere(4))
    st.build()
    for index, edge in enumerate(st.getEdgeList().toVector()):
        edge.setLength(cmath.sqrt(complex(1.0 + 0.019 * (index % 5),
                                          0.011 * (1 + index % 4))))
    return st


def _refined_host(n_refine=COMMITTING_REFINEMENTS, seed=3):
    """A refined 4-sphere large enough that a cone-out can open a hole.

    The refinement is by Pachner add-moves, which cannot change the topology --
    that is the point. Whatever `b_k` this host starts with, only surgery can
    move it, so a change observed after `directed_cone_out` came from surgery.
    """
    st = T.Spacetime(T.Metric(True, T.Signature(4, T.Lorentzian)),
                     T.CDT, 1.0, 1.0, T.PREFERRED,
                     T.SimplexBoundarySphere(4))
    st.build()
    for edge in st.getEdgeList().toVector():
        edge.setLength(cmath.sqrt(complex(1.0)))
    applied = 0
    for step in range(seed, seed + n_refine * 4):
        move = T.AddMove(st, step, False, T.PachnerMode.PreGeometric, False)
        if move.propose() and move.apply():
            applied += 1
        if applied >= n_refine:
            break
    for index, edge in enumerate(st.getEdgeList().toVector()):
        edge.setLength(cmath.sqrt(complex(1.0 + 0.01 * (index % 6),
                                          0.007 * (1 + index % 4))))
    return st


def _node(st, gamma=0.0):
    """The node mutates `st` in place, so the caller keeps the reference."""
    return MC(st, [], [], degrees=[DEGREE], gamma=gamma, seed=7)


def _topology(st):
    """The topological facts a Pachner move cannot change but surgery can."""
    return (len(st.getTopSimplices()),
            tuple(MC.betti(st)),
            len(MC.emergent_holes(st, DEGREE)))


BUILTINS = (
    ("joint_stationarity", cob.JointStationarityObjective),
    ("legacy", cob.LegacyObjective),
    ("mediated_correspondence", cob.MediatedCorrespondenceObjective),
)


class ConstantObjective(cob.CobordismObjective):
    """Identically zero on every complex -- genuinely indifferent to topology.

    Defined in Python, which #841 made possible. That matters here: an
    objective assembled from zeroed weights would only be *numerically* flat
    for the terms it happens to carry, whereas this one cannot depend on the
    geometry at all because it never looks at it. No candidate can strictly
    lower a constant, so a probe that honours its own acceptance rule must
    refuse every surgery.
    """

    def name(self):
        return "constant_zero"

    def term_names(self):
        return [cob.ObjectiveTermName.REGGE_STATIONARITY]

    def terms(self, context):
        return MC.ObjectiveTerms()

    def direction(self, context):
        return cob.ObjectiveDirection()

    def is_target_conditioned(self):
        return False


class ACommittedProbeLoweredTheInjectedObjectiveTest(unittest.TestCase):
    """The discriminating direction: a commit means the objective went DOWN.

    The probe used to keep whichever candidate most lowered `r_U`. That is a
    different functional, so a commit could raise the objective actually in
    force. Scoring the injected objective makes that impossible.
    """

    def test_a_cone_out_commits_and_strictly_lowers_the_objective(self):
        # Joint stationarity and the legacy objective both find an opener on
        # this host; mediated correspondence does not, which is the subject of
        # `TheDecisionFollowsTheInjectedObjectiveTest` below.
        for name, factory in BUILTINS[:2]:
            with self.subTest(objective=name):
                st = _refined_host()
                node = _node(st, gamma=1.0)
                node.set_objective(factory())
                before = node.objective()
                opened = node.directed_cone_out()
                after = node.objective()
                # Non-vacuity: the branch under test is the committing one.
                self.assertGreater(opened, 0)
                self.assertLess(after, before)

    def test_an_objective_that_finds_no_opener_leaves_everything_alone(self):
        st = _refined_host()
        node = _node(st, gamma=1.0)
        node.set_objective(cob.MediatedCorrespondenceObjective())
        before_topology = _topology(st)
        before_objective = node.objective()
        self.assertEqual(node.directed_cone_out(), 0)
        self.assertEqual(_topology(st), before_topology)
        self.assertAlmostEqual(node.objective(), before_objective,
                               delta=ROLLBACK_ULP_TOLERANCE)

    def test_a_commit_actually_changed_the_topology(self):
        """A commit is a real surgery, not a bookkeeping increment."""
        st = _refined_host()
        node = _node(st, gamma=1.0)
        node.set_objective(cob.JointStationarityObjective())
        before = _topology(st)
        self.assertGreater(node.directed_cone_out(), 0)
        self.assertNotEqual(_topology(st), before)

    def test_a_cone_in_that_commits_strictly_lowers_the_objective(self):
        for name, factory in BUILTINS[:2]:
            with self.subTest(objective=name):
                st = _refined_host()
                node = _node(st, gamma=1.0)
                node.set_objective(factory())
                # Open a hole first, so cone-in has something to cap.
                self.assertGreater(node.directed_cone_out(), 0)
                before = node.objective()
                closed = node.directed_cone_in()
                after = node.objective()
                if closed > 0:
                    self.assertLess(after, before)
                else:
                    self.assertAlmostEqual(after, before,
                                           delta=ROLLBACK_ULP_TOLERANCE)


class ProbeThatCommitsNothingChangesNothingTest(unittest.TestCase):
    """The other direction: refused candidates are rolled back."""

    def test_the_small_host_commits_nothing_and_is_left_untouched(self):
        for name, factory in BUILTINS[:2]:
            with self.subTest(objective=name):
                st = _small_sphere4()
                node = _node(st, gamma=1.0)
                node.set_objective(factory())
                before_topology = _topology(st)
                before_objective = node.objective()
                # Non-vacuity in the opposite sense: this host is chosen
                # precisely because nothing can be committed on it.
                self.assertEqual(node.directed_cone_out(), 0)
                self.assertEqual(_topology(st), before_topology)
                self.assertAlmostEqual(node.objective(), before_objective,
                                       delta=ROLLBACK_ULP_TOLERANCE)


class TopologyIndifferentObjectiveCommitsNothingTest(unittest.TestCase):
    """The probes do not open holes for their own sake."""

    @staticmethod
    def _indifferent_node():
        """A node scored by a constant, on the host where surgery DOES commit.

        Running on the refined host is what makes the refusal meaningful: a
        discriminating objective opens a hole there, so nothing being committed
        is a property of the objective rather than of a host too small to offer
        a candidate.
        """
        st = _refined_host()
        node = _node(st, gamma=0.0)
        node.set_objective(ConstantObjective())
        return node, st

    def test_the_objective_really_is_flat(self):
        """Guard against vacuity: the scalar must be identically zero, not
        merely small, or 'nothing was committed' would prove nothing."""
        node, st = self._indifferent_node()
        self.assertEqual(node.objective_name, "constant_zero")
        self.assertEqual(node.objective(), 0.0)

    def test_no_cone_out_is_committed_on_a_host_where_one_otherwise_would_be(self):
        node, st = self._indifferent_node()
        before = _topology(st)
        self.assertEqual(node.directed_cone_out(), 0)
        self.assertEqual(_topology(st), before)

    def test_no_cone_in_is_committed(self):
        node, st = self._indifferent_node()
        before = _topology(st)
        self.assertEqual(node.directed_cone_in(), 0)
        self.assertEqual(_topology(st), before)

    def test_betti_is_frozen_under_repeated_probing(self):
        """Pachner moves and relaxation cannot move `b_k`, and this objective
        refuses every surgery, so the topology cannot move at all."""
        node, st = self._indifferent_node()
        before = tuple(MC.betti(st))
        for _ in range(3):
            node.directed_cone_out()
            node.directed_cone_in()
        self.assertEqual(tuple(MC.betti(st)), before)


class BettiIsPermittedToEmergeTest(unittest.TestCase):
    """Higher Betti numbers are reachable, never required."""

    def test_a_discriminating_objective_reaches_a_hole_the_host_did_not_have(self):
        st = _refined_host()
        node = _node(st, gamma=1.0)
        node.set_objective(cob.JointStationarityObjective())
        holes_before = len(MC.emergent_holes(st, DEGREE))
        self.assertEqual(holes_before, 0)
        self.assertGreater(node.directed_cone_out(), 0)
        self.assertGreater(len(MC.emergent_holes(st, DEGREE)), holes_before)

    def test_the_same_host_stays_flat_under_an_indifferent_objective(self):
        """The controlled comparison: identical host, identical probe, and the
        only difference is the functional in force."""
        st = _refined_host()
        node = _node(st, gamma=0.0)
        node.set_objective(ConstantObjective())
        self.assertEqual(node.directed_cone_out(), 0)
        self.assertEqual(len(MC.emergent_holes(st, DEGREE)), 0)


class TheDecisionFollowsTheInjectedObjectiveTest(unittest.TestCase):
    """The property that could not hold under a fixed residual.

    Three objectives, one identical host, three decisions taken independently.
    Under the old scoring every probe consulted `r_U` no matter what objective
    was in force, so the decision could not depend on the objective. Here it
    demonstrably does: two objectives find an opener and one does not, on a
    host that is byte-identical in all three runs.
    """

    def test_objectives_disagree_about_whether_to_operate(self):
        decisions = {}
        for name, factory in BUILTINS:
            st = _refined_host()
            node = _node(st, gamma=1.0)
            node.set_objective(factory())
            decisions[name] = node.directed_cone_out()
        self.assertGreater(decisions["joint_stationarity"], 0)
        self.assertGreater(decisions["legacy"], 0)
        self.assertEqual(decisions["mediated_correspondence"], 0)
        # The disagreement is the point: a single fixed residual could not
        # produce a split verdict on one host.
        self.assertNotEqual(len(set(decisions.values())), 1)


class SurgeryNeverRaisesTheInjectedObjectiveTest(unittest.TestCase):
    """The sharpest form of the regression test.

    `r_U` and joint stationarity are different functionals. If the probe still
    scored `r_U` it could commit a cone-out that lowered `r_U` while RAISING
    joint stationarity. Asserting that the joint scalar never rises under
    repeated surgery therefore discriminates the two scorings.
    """

    def test_repeated_surgery_never_raises_the_joint_scalar(self):
        st = _refined_host()
        node = _node(st, gamma=1.0)
        node.set_objective(cob.JointStationarityObjective())
        previous = node.objective()
        committed = 0
        for _ in range(4):
            committed += node.directed_cone_out()
            current = node.objective()
            self.assertLessEqual(current, previous + _rollback_tolerance(previous))
            previous = current
            committed += node.directed_cone_in()
            current = node.objective()
            self.assertLessEqual(current, previous + _rollback_tolerance(previous))
            previous = current
        # Non-vacuity: at least one surgery was actually committed over the
        # loop, so the monotonicity above was tested against real commits.
        self.assertGreater(committed, 0)


if __name__ == "__main__":
    unittest.main()
