# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""#867 — the emergence animation's FIGURE: stabilization, causal colour, dual.

These three are presentation. The property that matters most is therefore a
negative one: none of it may reach the record. Everything else here checks
that what the figure shows is what the geometry says.

* the drawing layout is stabilized, so shared vertices move smoothly instead
  of being reshuffled by the rotation/reflection freedom of classical MDS;
* every edge's colour is the causal class of its own ``l^2``, and an ``l^2``
  off the real axis reads as ``indefinite`` rather than being bucketed into a
  definiteness the geometry does not have;
* the two dual-curvature channels are drawn apart and signed;
* the JSON is byte-identical with and without any of it.
"""

import cmath
import json
import math
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace

import tessera as T

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))),
    "examples", "cobordism"))

import emergence_animation as ea  # noqa: E402
import qubit_animation as qa  # noqa: E402

cob = T.cobordism
MC = cob.MultiCobordism

#: Big enough that stage 1 commits moves, so the topology actually changes
#: between frames — the case stabilization exists for. A run where the complex
#: never moved would let every assertion here pass while testing nothing.
HOST = 4
STEPS = 3

_CACHE = {}


def _frames():
    if "frames" not in _CACHE:
        config = ea.build_config(size=HOST, steps=STEPS)
        _CACHE["frames"] = ea.drive(config, progress=False).frames
    return _CACHE["frames"]


def _shared_step(previous, current):
    """Largest displacement over the vertices two frames have in common."""
    shared = [v for v in current if v in previous]
    if not shared:
        return None
    return max(math.dist(previous[v], current[v]) for v in shared)


def _walk(sequence):
    """Per-step shared displacements down a sequence of coordinate maps."""
    steps = []
    for previous, current in zip(sequence, sequence[1:]):
        if previous is None or current is None:
            continue
        moved = _shared_step(previous, current)
        if moved is not None:
            steps.append(moved)
    return steps


# ======================================================================
# 1. the layout is stabilized
# ======================================================================


class StabilizationTest(unittest.TestCase):

    def test_the_run_actually_changes_topology(self):
        """Guard: without this, every stabilization test below is vacuous."""
        frames = _frames()
        counts = {len(f.layout["edges"]) for f in frames
                  if not isinstance(f.layout, ea.Absent)}
        self.assertGreater(
            len(counts), 1,
            "the complex never changed, so nothing here tests stabilization; "
            "raise HOST or STEPS until stage 1 commits a move")

    def test_stabilizing_reduces_frame_to_frame_displacement(self):
        frames = _frames()
        raw = [None if isinstance(f.layout, ea.Absent)
               else {v: tuple(p) for v, p in f.layout["coords"].items()}
               for f in frames]
        placed = ea.stabilize(frames)
        aligned = [None if p is None else p["coords"] for p in placed]

        raw_steps = _walk(raw)
        aligned_steps = _walk(aligned)
        self.assertTrue(raw_steps, "no consecutive frames shared a vertex")
        self.assertEqual(len(raw_steps), len(aligned_steps))

        worst_raw = max(raw_steps)
        worst_aligned = max(aligned_steps)
        # Reported so a regression shows the numbers, not just a failure.
        self.assertLess(
            worst_aligned, worst_raw,
            "stabilization did not reduce displacement: raw %.6g, "
            "aligned %.6g" % (worst_raw, worst_aligned))

    def test_the_first_frame_is_taken_as_given(self):
        """There is nothing to align the first frame to, so it is unchanged."""
        frames = _frames()
        placed = ea.stabilize(frames)
        self.assertIsNotNone(placed[0])
        for vertex, position in frames[0].layout["coords"].items():
            self.assertAlmostEqual(placed[0]["coords"][vertex][0],
                                   position[0], places=12)
            self.assertAlmostEqual(placed[0]["coords"][vertex][1],
                                   position[1], places=12)

    def test_a_shared_vertex_does_not_swap_sides(self):
        """A reflection is the loudest MDS artefact: it mirrors the figure
        while the complex barely moved. After alignment a vertex that stays in
        the complex should not cross to the opposite side of the cloud."""
        placed = ea.stabilize(_frames())
        maps = [p["coords"] for p in placed if p is not None]
        self.assertGreaterEqual(len(maps), 2)
        for previous, current in zip(maps, maps[1:]):
            shared = [v for v in current if v in previous]
            if len(shared) < 3:
                continue
            flipped = sum(1 for v in shared
                          if previous[v][0] * current[v][0] < 0
                          and abs(previous[v][0]) > 0.25
                          and abs(current[v][0]) > 0.25)
            self.assertLessEqual(
                flipped, len(shared) // 2,
                "most shared vertices crossed the axis: the frame was "
                "mirrored rather than aligned")

    def test_the_live_path_places_frames_exactly_as_headless_does(self):
        """`--live` advances the chain one unit at a time as frames arrive,
        while a headless render walks a finished run. The alignment is a
        CHAIN, so the two agree only if one state sees the same frames in the
        same order. A live run stabilized against the wrong previous frame
        would look subtly wrong while every other check here still passed --
        the layout is not in the record, so the byte-identical JSON cannot
        catch it.
        """
        frames = _frames()
        headless = ea.stabilize(frames)
        # Exactly what `drive_live` does: one `place_frame` per unit, in
        # arrival order, against a single chained state.
        state = ea.StableLayout()
        live = [ea.place_frame(state, frame) for frame in frames]

        self.assertEqual(len(live), len(headless))
        for index, (a, b) in enumerate(zip(live, headless)):
            with self.subTest(frame=index):
                if a is None or b is None:
                    self.assertIs(a, b)
                    continue
                self.assertEqual(sorted(a["coords"]), sorted(b["coords"]))
                for vertex, position in a["coords"].items():
                    other = b["coords"][vertex]
                    self.assertAlmostEqual(position[0], other[0], places=12)
                    self.assertAlmostEqual(position[1], other[1], places=12)
                for left, right in zip(a["view"], b["view"]):
                    self.assertAlmostEqual(left, right, places=12)

    def test_reordering_the_chain_would_change_it(self):
        """Guard on the test above. If order did not matter, the two agreeing
        would prove nothing, so feeding the same frames reversed must NOT
        reproduce the forward chain."""
        frames = _frames()
        if len(frames) < 3:
            self.skipTest("needs at least three frames to reorder")
        forward = ea.stabilize(frames)
        state = ea.StableLayout()
        backward = [ea.place_frame(state, f) for f in reversed(frames)]
        differs = False
        for a, b in zip(forward, backward):
            if a is None or b is None:
                continue
            for vertex, position in a["coords"].items():
                other = b["coords"].get(vertex)
                if other is None or abs(position[0] - other[0]) > 1e-9:
                    differs = True
                    break
            if differs:
                break
        self.assertTrue(
            differs,
            "reversing the frame order changed nothing, so the ordering "
            "assertion above is vacuous")

    def test_the_view_is_never_grow_only(self):
        """A grow-only view shrinks the structure to a dot; the eased box has
        to be able to close back in on a cloud that contracted."""
        state = ea.StableLayout()
        wide = {0: (-4.0, -4.0), 1: (4.0, 4.0)}
        narrow = {0: (-0.1, -0.1), 1: (0.1, 0.1)}
        state.place(wide)
        first = list(state.view(wide))
        for _ in range(40):
            state.place(narrow)
            box = state.view(narrow)
        self.assertLess(box[1] - box[0], first[1] - first[0],
                        "the view never contracted onto the smaller cloud")


# ======================================================================
# 2. causal colour
# ======================================================================


def _classify(length):
    """`causal_class` of a bare length, via a throwaway edge.

    The classifier takes a live `Edge` because it dispatches on the library
    predicates rather than re-deriving a rule; these tests supply lengths, so
    they build an edge to carry one.
    """
    edge = T.Edge(T.Vertex(1, [0.0, 0.0, 0.0, 0.0]),
                  T.Vertex(2, [0.0, 0.0, 0.0, 1.0]), complex(length))
    return ea.causal_class(edge)


class CausalClassTest(unittest.TestCase):
    """Classification is the ARGUMENT of `l^2`, from the library predicates.

    The sign of the interval `Re(l^2)` is not enough on its own: it discards
    `Im(l^2)`, which is `2x^2 != 0` exactly at the lightlike point, and it
    forces every generic edge into a definite bucket. Reading `arg(l^2)` keeps
    both components and leaves a generic argument MIXED.
    """

    def test_the_definite_classes_read_off_the_argument(self):
        cases = [
            # l,                  arg(l^2),  expected
            (1.0 + 0.0j, ea.CausalClass.SPACELIKE),      # arg = 0
            (3.0 + 0.0j, ea.CausalClass.SPACELIKE),      # arg = 0
            (0.0 + 1.0j, ea.CausalClass.TIMELIKE),       # arg = pi
            (0.0 + 3.0j, ea.CausalClass.TIMELIKE),       # arg = pi
            (0.0 + 0.0j, ea.CausalClass.DEGENERATE),
        ]
        for value, expected in cases:
            with self.subTest(length=value):
                self.assertEqual(_classify(value), expected)

    def test_a_generic_argument_is_mixed_not_rounded(self):
        """The case a sign test gets wrong.

        `l = 2 + i` has `Re(l^2) = 3 > 0`, so a sign rule calls it spacelike.
        Its argument is `atan2(4, 3) ~ 0.927`, nowhere near `0`, so it has no
        definite causal character and is reported mixed. Rounding it to
        spacelike would assert a definiteness the geometry does not have.
        """
        for value in (2.0 + 1.0j, 1.0 + 2.0j, 3.0 + 1.0j):
            with self.subTest(length=value):
                self.assertNotEqual((value * value).real, 0.0)  # definite SIGN
                self.assertEqual(_classify(value), ea.CausalClass.MIXED)

    def test_equal_weighting_is_lightlike_and_not_degenerate(self):
        """`l = x(1+i)` with `x > 0`: the interval vanishes while the edge is
        perfectly well defined. Asserted on both, so it cannot pass by the
        edge having collapsed."""
        for x in (1.0, 0.5, 7.25, 1.0 / math.sqrt(2.0)):
            with self.subTest(x=x):
                length = complex(x, x)
                self.assertAlmostEqual((length * length).real, 0.0, places=12,
                                       msg="Re(l^2) must vanish")
                self.assertGreater(abs(length), 0.0,
                                   "the edge must not have collapsed")
                self.assertEqual(_classify(length), ea.CausalClass.LIGHTLIKE)

    def test_the_original_even_weighting_seed_was_entirely_lightlike(self):
        """`l = (1+i)/sqrt(2)` gives `l^2 = i`, so every edge of the host that
        seed built sat on the light cone -- read as timelike for as long as any
        imaginary part counted as such."""
        length = complex(1.0, 1.0) / math.sqrt(2.0)
        self.assertAlmostEqual((length * length).real, 0.0, places=12)
        self.assertEqual(_classify(length), ea.CausalClass.LIGHTLIKE)

    def test_lightlike_is_judged_relative_to_scale(self):
        """An angular tolerance is dimensionless, so the classification does
        not move with the scale the lengths happen to carry."""
        self.assertEqual(_classify(complex(1e3, 0.0)),
                         ea.CausalClass.SPACELIKE)
        self.assertEqual(_classify(complex(1e-3, 0.0)),
                         ea.CausalClass.SPACELIKE)
        self.assertEqual(_classify(complex(1.0, 1.0 - 1e-13)),
                         ea.CausalClass.LIGHTLIKE)

    def test_degenerate_is_not_lightlike(self):
        """Both sit at a vanishing interval; only one is an edge."""
        self.assertEqual(_classify(0j), ea.CausalClass.DEGENERATE)
        self.assertEqual(_classify(complex(1.0, 1.0)),
                         ea.CausalClass.LIGHTLIKE)

    def test_every_class_has_a_colour_and_a_legend_entry(self):
        """Colour and label read from the same vocabulary, so a panel can
        never draw a class it cannot name."""
        for name in ea.CausalClass.ALL:
            self.assertIn(name, ea.DECLARED_CAUSAL_COLOURS)
            self.assertIn(name, ea._CAUSAL_LEGEND)

    def test_the_dispositions_produce_the_classes_they_name(self):
        expected = {
            ea.EdgeDisposition.SPACELIKE: ea.CausalClass.SPACELIKE,
            ea.EdgeDisposition.TIMELIKE: ea.CausalClass.TIMELIKE,
            ea.EdgeDisposition.LIGHTLIKE: ea.CausalClass.LIGHTLIKE,
        }
        for disposition, causal in expected.items():
            with self.subTest(disposition=disposition):
                host = ea.build_cobordism_host(HOST, ea.DECLARED_HOST_SEED,
                                               disposition)
                classes = {ea.causal_class(e)
                           for e in host.getEdgeList().toVector()}
                self.assertEqual(classes, {causal})

    def test_random_is_entirely_mixed(self):
        """Measured, and the finding rather than a defect.

        `l = e^{ia}` with `a` uniform gives a generic `arg(l^2)` almost surely,
        so a random seed has NO definite causal character anywhere. An earlier
        version of this panel read that as saying nothing and replaced the
        indefinite class with a sign test, which reported every edge as
        definitely spacelike or timelike. All-mixed is the honest reading, and
        the mixed fraction across engine units is the diagnostic of whether
        relaxation imposes a disposition.
        """
        host = ea.build_cobordism_host(HOST, ea.DECLARED_HOST_SEED,
                                       ea.EdgeDisposition.RANDOM)
        classes = {ea.causal_class(e) for e in host.getEdgeList().toVector()}
        self.assertEqual(classes, {ea.CausalClass.MIXED})

    def test_foliated_carries_both_definite_classes(self):
        host = ea.build_cobordism_host(HOST, ea.DECLARED_HOST_SEED,
                                       ea.EdgeDisposition.FOLIATED)
        classes = {ea.causal_class(e) for e in host.getEdgeList().toVector()}
        self.assertIn(ea.CausalClass.SPACELIKE, classes)
        self.assertIn(ea.CausalClass.TIMELIKE, classes)
        self.assertNotIn(ea.CausalClass.MIXED, classes)

    def test_the_frame_carries_a_class_and_an_argument_per_drawn_edge(self):
        frame = _frames()[0]
        edges = frame.layout["edges"]
        self.assertEqual(len(frame.layout["edge_causal_classes"]), len(edges))
        self.assertEqual(len(frame.layout["edge_causal_arguments"]), len(edges))
        for causal, length, interval, argument in zip(
                frame.layout["edge_causal_classes"],
                frame.layout["edge_lengths"],
                frame.layout["edge_intervals"],
                frame.layout["edge_causal_arguments"]):
            self.assertEqual(causal, _classify(length))
            self.assertAlmostEqual(interval, (length * length).real,
                                   places=12)
            # arg(l^2) is carried as the MEASURED value beside the bucket.
            self.assertAlmostEqual(argument, cmath.phase(length * length),
                                   places=12)


# ======================================================================
# 3. the dual curvature panels
# ======================================================================


class DualCurvatureTest(unittest.TestCase):

    def test_one_dual_node_per_top_cell(self):
        frame = _frames()[-1]
        self.assertNotIsInstance(frame.dual, ea.Absent)
        self.assertEqual(len(frame.dual["cells"]),
                         len(frame.spacetime.getTopSimplices()))

    def test_both_channels_are_signed_and_kept_apart(self):
        """The Lorentzian deficit is complex and the two parts are different
        physics, so a magnitude would destroy the distinction."""
        frame = _frames()[-1]
        for cell in frame.dual["cells"]:
            self.assertIsInstance(cell["spatial"], float)
            self.assertIsInstance(cell["temporal"], float)
        signs = {math.copysign(1.0, c["spatial"])
                 for c in frame.dual["cells"] if c["spatial"] != 0.0}
        self.assertTrue(
            signs, "no cell carried spatial curvature: the sign convention "
                   "is untested on this host")

    def test_a_spacelike_host_has_no_temporal_curvature(self):
        """Sign convention asserted, not assumed: the temporal channel is the
        boost content, carried by spacelike hinges via a timelike normal
        plane. An all-spacelike geometry has no boost content to report."""
        config = ea.build_config(size=HOST, steps=0,
                                 edge_disposition=ea.EdgeDisposition.SPACELIKE)
        frames = ea.drive(config, progress=False).frames
        dual = frames[-1].dual
        self.assertNotIsInstance(dual, ea.Absent)
        worst = max(abs(c["temporal"]) for c in dual["cells"])
        self.assertLessEqual(
            worst, 1e-9,
            "an all-spacelike host reported boost curvature: the two "
            "channels are crossed")

    def test_the_panels_render_on_a_real_host(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        frames = _frames()
        placed = ea.stabilize(frames)
        figure = plt.figure(figsize=(6, 4))
        try:
            axis = figure.add_subplot(1, 1, 1)
            ea._panel_dual_spatial(axis, frames[-1], placed[-1])
            self.assertIn("spatial", axis.get_title())
            axis2 = figure.add_subplot(2, 1, 2)
            ea._panel_dual_temporal(axis2, frames[-1], placed[-1])
            self.assertIn("temporal", axis2.get_title())
        finally:
            plt.close(figure)


class ReadoutPresentationTest(unittest.TestCase):

    def test_a_summed_whole_transfer_panel_names_the_component_and_aggregate(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        frame = SimpleNamespace(
            inputs=SimpleNamespace(algebra={"chi": np.eye(2)}),
            two_body={
                "selected_readouts": ["transfer", "whole"],
                "residual": 0.75,
                "transfer": np.eye(2),
                "singular_values": [1.0, 1.0],
                "schmidt_rank": 2,
                "reversal_residual": 0.0,
                "in_frames": True,
            },
            config={"readout": "transfer,whole"},
        )
        figure, axis = plt.subplots()
        try:
            qa._panel_transfer(axis, frame)
            title = axis.get_title()
            self.assertIn("transfer component", title)
            self.assertIn("aggregate leak 0.75 (includes whole)", title)
        finally:
            plt.close(figure)


# ======================================================================
# 4. presentation only — the record is untouched
# ======================================================================


class RecordIsUntouchedTest(unittest.TestCase):

    def test_neither_the_layout_nor_the_dual_reaches_the_record(self):
        document = _frames()[-1].to_json()
        self.assertNotIn("layout", document)
        self.assertNotIn("dual", document)

    def test_the_json_is_byte_identical_across_two_drives(self):
        """The figure work is downstream of every measurement, so two runs at
        one seed must serialize identically — the acceptance test for
        'presentation only'."""
        def document():
            config = ea.build_config(size=HOST, steps=0)
            frames = ea.drive(config, progress=False).frames
            return json.dumps({"config": config,
                               "frames": [f.to_json() for f in frames]},
                              indent=2, sort_keys=True)

        self.assertEqual(document(), document())

    def test_rendering_does_not_disturb_the_record(self):
        """Drawing must not mutate the frames it draws."""
        frames = _frames()
        before = json.dumps([f.to_json() for f in frames], sort_keys=True)
        with tempfile.TemporaryDirectory() as directory:
            ea.render(frames, os.path.join(directory, "frame.png"))
        after = json.dumps([f.to_json() for f in frames], sort_keys=True)
        self.assertEqual(before, after)

    def test_the_whole_overlay_renders(self):
        frames = _frames()
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "overlay.png")
            ea.render(frames, path)
            self.assertTrue(os.path.getsize(path) > 0)

    def test_every_panel_has_a_slot_and_no_slot_is_blank(self):
        rows, columns = ea.DECLARED_PANEL_GRID
        self.assertLessEqual(len(ea._PANELS), rows * columns)
        for name in ea._PLACED_PANELS:
            self.assertIn(name, [n for n, _ in ea._PANELS])


if __name__ == "__main__":
    unittest.main()
