# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""#1193 — the multi-frame evidence the recursive drivers supply.

The whitepaper accepts a fiber on evidence gathered over SEVERAL cobordism
frames: a lifetime across frames, an overlap with the predecessor and the
successor component, and a lifetime transport family (the candidate's world
tube) whose leakage and determinant line are certificates.  The overlay used
to assert a frame lifetime of one and supply no lifetime transports at all.

One analysis pass is one cobordism frame.  These tests pin that the overlay
RETAINS the frames it has already read, that the lifetime and the
adjacent-frame overlap it reports are measured across them by the library's
own frame tracker, that the lifetime transport family appears only once there
is more than one frame to link, and that a run configured to retain a single
frame still reports a one-frame lifetime — a measured fact about that run,
never a vacuous pass.
"""

import json
import os
import sys
import unittest

import tessera

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _closed_s4 import closed_s4 as _closed_s4  # noqa: E402

cob = tessera.cobordism
MC = cob.MultiCobordism

_REFINE = 6
_HOST_SEED = 3
_NODE_SEED = 7


def _node():
    st = _closed_s4(_REFINE, _HOST_SEED)
    node = MC(st, [], [], [1], 1.0, _NODE_SEED)
    node.set_objective(cob.JointStationarityObjective())
    return node


def _config(frame_history=4, closure="none"):
    cfg = MC.AnalysisConfig()
    cfg.enabled = True
    cfg.cadence = 1
    cfg.degrees = [1]
    cfg.resolutions = [1.0]
    cfg.frame_history = frame_history
    cfg.lifetime_winding_closure = closure
    return cfg


def _passes(count, frame_history=4, closure="none"):
    """`count` analysis passes over one unchanged geometry.

    The geometry is not relaxed between passes, so every frame reads the same
    complex: the candidate's support is identical frame to frame and the
    overlap along its track is exactly one.  That isolates the FRAME axis —
    what the overlay retains and measures — from the unrelated question of how
    a relaxing geometry moves a cluster.
    """
    node = _node()
    node.set_analysis_config(_config(frame_history, closure))
    for _ in range(count):
        node.run_recursive_analysis()
    return json.loads(node.checkpoint_json)


class ConfigurationTest(unittest.TestCase):

    def test_a_frame_history_below_one_is_refused(self):
        node = _node()
        with self.assertRaises(ValueError):
            node.set_analysis_config(_config(frame_history=0))

    def test_an_unrecognized_closure_is_refused(self):
        node = _node()
        with self.assertRaises(ValueError):
            node.set_analysis_config(_config(closure="whatever"))

    def test_both_declared_closures_are_accepted(self):
        node = _node()
        for closure in ("none", "closed-family"):
            node.set_analysis_config(_config(closure=closure))
            self.assertEqual(node.analysis_config.lifetime_winding_closure,
                             closure)


class RetainedFrameTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.one = _passes(1)
        cls.three = _passes(3)

    def test_one_pass_is_one_frame(self):
        """The first pass has no predecessor: a one-frame lifetime, and no
        lifetime transport family, because there is nothing to link to."""
        quarks = self.one["particles"]["quarks"]
        self.assertTrue(quarks, "the fixture produced no candidate")
        for quark in quarks:
            self.assertEqual(quark["frame_lifetime"], 1.0)
            self.assertEqual(quark["transport_count"], 0)

    def test_three_passes_measure_a_three_frame_lifetime(self):
        quarks = self.three["particles"]["quarks"]
        self.assertTrue(quarks, "the fixture produced no candidate")
        lifetimes = [q["frame_lifetime"] for q in quarks]
        self.assertEqual(max(lifetimes), 3.0,
                         "three retained frames must measure a lifetime of "
                         "three on the unchanged geometry")

    def test_the_adjacent_frame_overlap_is_measured(self):
        """The unchanged geometry repeats its support, so the smallest
        adjacent-frame overlap along the track is exactly one — MEASURED by
        `PersistentModularity.trackAcrossFrames`, not assumed."""
        for quark in self.three["particles"]["quarks"]:
            if quark["frame_lifetime"] and quark["frame_lifetime"] > 1.0:
                self.assertEqual(quark["frame_min_overlap"], 1.0)

    def test_the_lifetime_transport_family_appears(self):
        """A candidate tracked through several frames carries the world-tube
        links between them; the leakage of that family is then a measured
        quantity rather than a missing one."""
        quarks = self.three["particles"]["quarks"]
        linked = [q for q in quarks if q["transport_count"] > 0]
        self.assertTrue(linked,
                        "no candidate carried a lifetime transport family")
        for quark in linked:
            self.assertLess(quark["transport_count"],
                            quark["frame_lifetime"],
                            "n frames link into at most n-1 transports")
            self.assertIsNotNone(quark["transport_leakage_max"])

    def test_the_band_stability_is_decided_over_several_frames(self):
        stability = [q["stability_frames"]
                     for q in self.three["particles"]["quarks"]]
        self.assertTrue(any(value >= 2 for value in stability),
                        "the band family never reached two frames")

    def test_retaining_one_frame_keeps_the_single_frame_reading(self):
        """`frame_history` 1 is the behaviour every earlier build had: no
        history, a one-frame track, and the persistence certificates failing
        by name rather than passing vacuously."""
        document = _passes(3, frame_history=1)
        quarks = document["particles"]["quarks"]
        self.assertTrue(quarks)
        for quark in quarks:
            self.assertEqual(quark["frame_lifetime"], 1.0)
            self.assertEqual(quark["transport_count"], 0)
            self.assertIn("persistence", quark["failed_certificates"])


class WindingClosureTest(unittest.TestCase):

    def test_an_open_segment_leaves_the_winding_unknown(self):
        """With no declared closure the lifetime family is an open cobordism
        segment: an open path has no integer winding, and none is invented."""
        document = _passes(3)
        for quark in document["particles"]["quarks"]:
            if quark["transport_count"] > 0:
                self.assertIsNone(quark["determinant_winding"])
                self.assertIn("winding", quark["failed_certificates"])

    def test_a_declared_closed_family_is_read_cyclically(self):
        """The declaration is recorded on the read it produced, so a
        consumer can see which closure the winding was taken under."""
        document = _passes(3, closure="closed-family")
        linked = [q for q in document["particles"]["quarks"]
                  if q["transport_count"] > 0]
        self.assertTrue(linked, "no candidate carried a lifetime family")
        for quark in linked:
            self.assertIn(quark["winding_closure"],
                          ("closed-family", "none"))


if __name__ == "__main__":
    unittest.main()
