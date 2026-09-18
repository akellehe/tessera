# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""Causal disposition proposed as a gated move (#613).

The disposition belongs in stage 1's move draw beside add / remove / flip /
iflip / cone_out / cone_in: proposed at random, scored by `deltaF`, and committed
only when it lowers `F`. These moves propose a direct discrete jump between
causal sectors; the complex-z Stage 2 can also rotate continuously around z=0.
Nothing prescribes causal structure—the objective decides whether it wants any.

These tests pin two things: that the causal moves are ON by default (#632 -- they
are the seed's ONLY descent directions, so a draw without them hides the physics
and leaves stage 1 reporting a stall it does not have), and that the mechanism does
what it claims in both settings. They deliberately do NOT assert that timelike edges
emerge from any particular seed -- that is a measurement whose outcome is a result
either way, not an invariant.
"""

import collections
import unittest

import tessera

cob = tessera.cobordism
MC = cob.MultiCobordism


def _pentatope_host():
    return tessera.Spacetime.fromVertexTuples(4, [[0, 1, 2, 3, 4]], 1.0, 0.0)


def _dispositions(st):
    hist = collections.Counter()
    for e in st.getEdgeList().toVector():
        if e.isTimelike():
            hist["timelike"] += 1
        elif e.isNull():
            hist["null"] += 1
        else:
            hist["spacelike"] += 1
    return hist


class TestConeInTimelikeFlag(unittest.TestCase):
    """SurgicalCone.coneIn(timelike=...) writes only the apex edges."""

    def _cone(self, timelike):
        st = _pentatope_host()
        cone = cob.SurgicalCone(st)
        facet = [0, 1, 2, 3]
        ok, reason = cone.coneIn(facet, timelike=timelike)
        self.assertTrue(ok, f"cone-in rejected: {reason}")
        return st

    def test_default_is_spacelike_and_unchanged(self):
        """The default must be byte-identical to before the flag existed."""
        st = self._cone(timelike=False)
        self.assertEqual(_dispositions(st).get("timelike", 0), 0,
                         "coneIn defaults to spacelike; a timelike edge here "
                         "would mean the default changed behaviour")

    def test_timelike_writes_apex_edges_only(self):
        st = self._cone(timelike=True)
        # The apex is the vertex that did not exist in the seed pentatope.
        apex = max(v.getId() for v in st.getVertexList().toVector())
        self.assertGreater(apex, 4, "the apex must be a fresh vertex")
        for e in st.getEdgeList().toVector():
            a, b = e.getSource().getId(), e.getTarget().getId()
            if apex in (a, b):
                self.assertTrue(e.isTimelike(),
                                f"apex edge ({a},{b}) must be timelike")
            else:
                self.assertFalse(e.isTimelike(),
                                 f"pre-existing edge ({a},{b}) must be untouched")


class TestDispositionMovesAreOnByDefault(unittest.TestCase):
    """The causal moves are in the draw by default; opting out still works."""

    def _node(self, propose):
        node = MC(_pentatope_host(), [[1 + 0j]], [], [3], 50.0, 1, 0, propose)
        node.seed_inputs([0])
        return node

    def test_default_is_on(self):
        """Measured on the single-Delta^4 seed over EVERY move that adds a vertex:
        each of the five spacelike cone-ins raises F by +0.777 and the Pachner add
        by +2.58, while each of the five timelike cone-ins LOWERS F, ||grad S||^2
        and Re S (dF = -0.208, all five equal by the seed's S5 symmetry), and they
        are the only moves giving Im S != 0. Without them in the draw the seed's
        only descent directions are never proposed at all."""
        node = MC(_pentatope_host(), [[1 + 0j]], [], [3], 50.0, 1, 0)
        self.assertTrue(node.should_propose_dispositions,
                        "the causal moves are the seed's only descent directions; "
                        "they must be drawn by default")

    def test_flag_is_reported(self):
        self.assertTrue(self._node(True).should_propose_dispositions)
        self.assertFalse(self._node(False).should_propose_dispositions)

    def test_off_removes_the_disposition_moves_from_stage_one(self):
        """Opting out removes the disposition MOVES from the stage-1 draw.

        That is all the flag governs now. It no longer pins the complex
        spacelike: the complex-step stage 2 explores the full complex length
        plane by design (#644), so causal character can rotate continuously
        there regardless of the stage-1 draw. The stage-1-only guarantee is
        pinned by running ONLY stage 1: with the flag off, no combinatorial
        move may change a disposition.
        """
        node = self._node(False)
        node.run_stage1(60, 8, True)
        node.run_stage1(30, 8, False)
        self.assertEqual(_dispositions(node.st).get("timelike", 0), 0,
                         "with dispositions off, no move in the stage-1 draw "
                         "may change a causal disposition")


class TestDispositionMovesAreGated(unittest.TestCase):
    """When on, dispositions are accepted only through the ordinary gate."""

    def test_objective_stays_finite_and_register_still_carries(self):
        """The feature must not break the invariants the drive relies on."""
        node = MC(_pentatope_host(), [[1 + 0j]], [], [3], 50.0, 2, 0, True)
        node.seed_inputs([0])
        node.run_stage1(60, 8, True)
        node.run_stage1(30, 8, False)
        st = node.st
        objective = node.objective()
        self.assertGreaterEqual(objective, 0.0,
                                "F is a sum of non-negative terms")
        self.assertTrue(objective == objective, "F must not be NaN")
        self.assertGreaterEqual(node.r_u(st), 0.0)

    def test_complex_stays_a_valid_manifold(self):
        """Every disposition move goes through the same gate as every other."""
        node = MC(_pentatope_host(), [[1 + 0j]], [], [3], 50.0, 3, 0, True)
        node.seed_inputs([0])
        node.run_stage1(60, 8, True)
        ok, reason = cob.SurgicalCone(node.st).validate()
        self.assertTrue(ok, f"drive left an invalid complex: {reason}")


if __name__ == "__main__":
    unittest.main()
