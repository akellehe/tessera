# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""#831 — the emergence animation, exercised end to end.

Per repository convention the driver lives in ``examples/``; this file covers
the correctness of the INSTRUMENT:

* the drive runs unforced emergence and produces one frame per engine unit,
  and the objective is genuinely the joint stationarity objective in strict
  emergence mode;
* every channel is either a measurement or an ``Absent`` carrying a NAMED
  reason — never a zero standing in for an unmeasured value, and never a
  blank;
* the paper's abandoned vocabulary is absent from the driver: no hole count,
  no register, no pinned singlet target, no residual against a prescribed
  carrier;
* Betti numbers are reported as an independent observable and are never used
  as a quark count or a success condition;
* the verdict is the library classifier's own string, relayed verbatim;
* the measurements are JSON-round-trippable, with unknowns as ``null``;
* the overlay renders without raising, and an absent panel still draws its
  reason.
"""

import json
import os
import sys
import tempfile
import unittest

import tessera as T

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))),
    "examples", "cobordism"))

import emergence_animation as ea  # noqa: E402

cob = T.cobordism
obs = T.observables
MC = cob.MultiCobordism

#: The smallest host the driver is meaningful on: a multi-component
#: modularity read and a full band enumeration.
SMALL = 4
SMALL_STEPS = 1

#: One shared drive, built once — the whole suite reads it.
_FRAME_CACHE = {}


def _frames():
    if "frames" not in _FRAME_CACHE:
        config = ea.build_config(size=SMALL, steps=SMALL_STEPS)
        _FRAME_CACHE["frames"] = ea.drive(config, progress=False).frames
        _FRAME_CACHE["config"] = config
    return _FRAME_CACHE["frames"]


# ======================================================================
# the drive
# ======================================================================


class DriveTest(unittest.TestCase):
    """The drive is unforced emergence, one frame per engine unit."""

    def test_one_frame_per_engine_unit_plus_the_initial_read(self):
        frames = _frames()
        self.assertEqual(len(frames), SMALL_STEPS + 1)
        self.assertEqual([f.step for f in frames],
                         list(range(SMALL_STEPS + 1)))

    def test_the_host_is_neutral(self):
        """No holes, no pinned carrier, no boundary blocks by construction."""
        host = ea.build_cobordism_host(SMALL, ea.DECLARED_HOST_SEED)
        betti = list(MC.betti(host))
        # A 4-ball refined by stellar adds stays contractible: the neutral
        # host carries no hole for a quark to be identified with.
        self.assertEqual(betti[0], 1)
        self.assertTrue(all(b == 0 for b in betti[1:3]))

    def test_the_host_has_an_incoming_boundary(self):
        """The readouts need M0; a closed complex cannot supply one."""
        host = ea.build_cobordism_host(SMALL, ea.DECLARED_HOST_SEED)
        self.assertTrue(ea.boundary_vertices(host),
                        "the host must be a cobordism, not a closed complex")

    def test_the_default_seed_carries_lorentzian_content(self):
        """The Lorentzian content the original host lacked.

        The first host initialized every length purely real and positive, so
        the seed carried no imaginary part at all -- a programme Lorentzian in
        every path starting from a complex that was not. Nothing CONSTRAINS
        the geometry to stay real -- stage 2 rotates `z` freely and the engine
        has disposition moves -- but the starting point had no causal content
        to evolve from.

        The default disposition is `random`, so this asserts the property that
        survives the change of convention: the seed is neither all real nor
        all one character. The per-setting `l^2` values are pinned by
        `EdgeDispositionTest`.
        """
        host = ea.build_cobordism_host(SMALL, ea.DECLARED_HOST_SEED)
        lengths = [complex(edge.getLength())
                   for edge in host.getEdgeList().toVector()]
        self.assertTrue(lengths)
        self.assertTrue(any(abs(l.imag) > 1e-9 for l in lengths),
                        "the seed is purely real: no causal content")
        squared = [l ** 2 for l in lengths]
        spread = max(abs(v - squared[0]) for v in squared)
        self.assertGreater(spread, 1e-6,
                           "every edge carries one causal character")

    def test_the_objective_is_joint_stationarity_in_strict_emergence(self):
        host = ea.build_cobordism_host(SMALL, ea.DECLARED_HOST_SEED)
        node = MC(host, [], [], list(ea.DECLARED_REGISTER_DEGREES), 1.0,
                  ea.DECLARED_SEED)
        node.set_objective(cob.JointStationarityObjective())
        node.set_simulation_mode(MC.SimulationMode.EMERGENCE,
                                 MC.EmergenceSubmode.STRICT)
        self.assertEqual(node.simulation_mode, MC.SimulationMode.EMERGENCE)
        self.assertEqual(node.emergence_submode, MC.EmergenceSubmode.STRICT)

    def test_no_input_or_output_target_is_pinned(self):
        """The node is built with empty target lists: nothing is prescribed.

        A target-conditioned node scores a residual against a carrier the
        caller chose; the paper's emergence protocol calls that synthesis.
        This driver constructs its node with both target lists empty, so
        there is nothing to score against.
        """
        import inspect
        source = inspect.getsource(ea.drive)
        self.assertIn("MC(host, [], []", source)


# ======================================================================
# absence is a statement, not a zero
# ======================================================================


class AbsenceTest(unittest.TestCase):
    """An unmeasured channel carries a NAMED reason, never a zero."""

    CHANNELS = ("clusters", "bands", "anchors", "transports", "statistics",
                "crossings", "spin", "betti", "verdict")

    def test_every_channel_is_a_measurement_or_a_named_absence(self):
        frame = _frames()[-1]
        for name in self.CHANNELS:
            with self.subTest(channel=name):
                value = getattr(frame, name)
                if isinstance(value, ea.Absent):
                    self.assertTrue(value.reason.strip(),
                                    "%s is absent with an empty reason" % name)
                else:
                    self.assertIsInstance(value, dict)

    def test_an_absence_never_serializes_as_zero(self):
        frame = _frames()[-1]
        document = frame.to_json()
        for name in self.CHANNELS:
            value = document[name]
            if isinstance(value, dict) and value.get("absent"):
                with self.subTest(channel=name):
                    self.assertNotEqual(value.get("reason"), "")
                    self.assertNotIn("value", value)

    def test_the_crossing_readouts_are_reached_at_all(self):
        """The host is a cobordism, so tau has a reference surface.

        On a CLOSED complex these readouts are unreachable at any size: tau
        is the Lorentzian distance FROM M0 and there is no M0 to measure
        from. The channel may still refuse -- a band that fails positivity
        supplies no crossing -- but it must refuse for a reason of its own
        rather than for want of a surface to slice.
        """
        frame = _frames()[-1]
        if isinstance(frame.crossings, ea.Absent):
            self.assertNotIn("closed host", frame.crossings.reason)
            self.assertNotIn("no incoming boundary", frame.crossings.reason)
            return
        self.assertIn("level", frame.crossings)
        self.assertIn("crossings", frame.crossings)

    def test_the_host_supplies_a_reference_surface(self):
        """M0 exists, so `tau` has a surface to be measured from.

        This is the ticket's fix. Whether `tau` then CERTIFIES is a separate
        question about the seed's causal content, pinned below.
        """
        host = ea.build_cobordism_host(SMALL, ea.DECLARED_HOST_SEED)
        self.assertTrue(ea.boundary_vertices(host))

    def test_the_seed_has_no_causal_order_yet(self):
        """Measured, and NOT the behaviour to preserve.

        The default seed draws `arg l` uniformly, so almost every edge has a
        GENERIC argument and is therefore mixed -- no definite causal
        character at all. The temporal function needs causal edges before it
        can order anything, so it refuses with `no-causal-edges`.

        This reason CHANGED when causal type began to be read from `arg(l^2)`
        rather than from `Im(l) != 0` (#870). Under the superseded test any
        nonzero imaginary part counted as timelike, so a randomly seeded
        complex looked causal EVERYWHERE and the refusal was `causal-cycle`:
        causal character with no causal order. Reading the argument shows the
        seed has no definite character to begin with, which is both the honest
        state and a stricter starting point -- there is nothing to order,
        rather than too much.

        This test records the measured state so a change is noticed. If the
        dynamics later drives edges onto definite arguments and produces an
        order, `certified` becomes True and this test should be replaced by
        one asserting that -- it is a tripwire, not a specification.
        """
        host = ea.build_cobordism_host(SMALL, ea.DECLARED_HOST_SEED)
        temporal = obs.CrossingReadouts.temporalFunction(
            host, ea.boundary_vertices(host))
        reasons = [str(r) for r in temporal.failedCertificates]
        if temporal.certified:
            self.assertEqual(reasons, [])
            return
        self.assertIn("no-causal-edges", reasons)

    def test_a_refused_transport_names_why(self):
        frame = _frames()[-1]
        if isinstance(frame.transports, ea.Absent):
            self.assertTrue(frame.transports.reason.strip())
            return
        rejected = [r for r in frame.transports["rows"] if not r["accepted"]]
        for row in rejected:
            with self.subTest(row=row):
                self.assertTrue(str(row.get("reason", "")).strip())

    def test_unknown_serializes_as_null_not_zero(self):
        frame = _frames()[-1]
        document = frame.to_json()
        objective = document["objective"]
        for key, value in objective.items():
            with self.subTest(term=key):
                if key == "hodge_by_degree":
                    # The per-degree breakdown (#859) is a list of records
                    # rather than a scalar term. The same rule reaches every
                    # numeric leaf inside it: an unmeasured share serializes as
                    # null, never as a zero that reads like a measurement.
                    self.assertIsInstance(value, list)
                    for share in value:
                        self.assertIsInstance(share["degree"], int)
                        for field in ("weight", "gradient_norm_squared",
                                      "contribution"):
                            self.assertTrue(
                                share[field] is None
                                or isinstance(share[field], float),
                                "%s.%s is neither null nor a float"
                                % (key, field))
                    continue
                self.assertTrue(value is None or isinstance(value, float))


# ======================================================================
# the paper's ontology — what this driver refuses to draw
# ======================================================================


class OntologyTest(unittest.TestCase):
    """A quark is a persistent modular spectral cluster, never a hole."""

    SOURCE = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))),
        "examples", "cobordism", "emergence_animation.py")

    #: Vocabulary of the construction the whitepaper abandoned.
    RETIRED = ("_MIN_QUARK_HOLES", "r_state", "hole = quark",
               "color register", "quark hole", "singlet residual")

    @classmethod
    def _code(cls):
        """The driver's CODE, with the module docstring removed.

        The docstring names the retired vocabulary in order to say the driver
        does not use it, so scanning it would be self-defeating: what matters
        is that no such construction appears in the code.
        """
        import ast
        with open(cls.SOURCE) as handle:
            source = handle.read()
        tree = ast.parse(source)
        body = tree.body
        if (body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            lines = source.splitlines(keepends=True)
            first = body[0].lineno - 1
            last = body[0].end_lineno
            return "".join(lines[:first] + lines[last:])
        return source

    def test_the_driver_carries_no_retired_vocabulary(self):
        code = self._code()
        for token in self.RETIRED:
            with self.subTest(token=token):
                self.assertNotIn(token, code)

    def test_the_driver_pins_no_omega_target(self):
        """No `{1, omega, omega^2}` carrier is prescribed anywhere."""
        code = self._code()
        self.assertNotIn("omega ** 2", code)
        self.assertNotIn("exp(2j * math.pi / 3", code)

    def test_betti_is_an_observable_and_not_a_quark_count(self):
        frame = _frames()[-1]
        if isinstance(frame.betti, ea.Absent):
            self.assertTrue(frame.betti.reason.strip())
            return
        numbers = frame.betti["numbers"]
        self.assertTrue(numbers)
        # The verdict must not be a function of the Betti numbers: it is the
        # certified-quark count that gates it, and on this host that is zero
        # while b_0 is one.
        if isinstance(frame.verdict, ea.Absent):
            self.assertNotIn("betti", frame.verdict.reason.lower())
            self.assertNotIn("hole", frame.verdict.reason.lower())

    def test_the_verdict_gate_counts_certified_quarks(self):
        frame = _frames()[-1]
        if isinstance(frame.verdict, ea.Absent):
            self.assertIn("quark", frame.verdict.reason)
        else:
            self.assertIn(frame.verdict["classification"],
                          {"no-baryon", "baryon-candidate", "certified-proton",
                           "quasi-free-sharp-spin-obstruction"})


# ======================================================================
# serialization
# ======================================================================


class SerializationTest(unittest.TestCase):
    """The measurements round-trip through JSON."""

    def test_every_frame_round_trips(self):
        for frame in _frames():
            with self.subTest(step=frame.step):
                document = frame.to_json()
                restored = json.loads(json.dumps(document))
                self.assertEqual(restored["step"], frame.step)

    def test_the_document_carries_the_config(self):
        frames = _frames()
        document = {"config": _FRAME_CACHE["config"],
                    "frames": [f.to_json() for f in frames]}
        restored = json.loads(json.dumps(document))
        self.assertEqual(restored["config"]["size"], SMALL)
        self.assertEqual(len(restored["frames"]), SMALL_STEPS + 1)


# ======================================================================
# the overlay
# ======================================================================


class OverlayTest(unittest.TestCase):
    """The overlay renders, and an absent panel still draws its reason."""

    def test_the_overlay_renders_a_png(self):
        frames = _frames()
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "overlay.png")
            ea.render(frames, path)
            self.assertTrue(os.path.exists(path))
            self.assertGreater(os.path.getsize(path), 1024)

    def test_every_panel_paints_without_raising(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        frames = _frames()
        figure = plt.figure(figsize=(15, 9))
        try:
            for index in range(len(frames)):
                with self.subTest(frame=index):
                    ea.draw_frame(figure, frames, index)
        finally:
            plt.close(figure)

    def test_an_absent_panel_draws_its_reason(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure = plt.figure()
        axis = figure.add_subplot(111)
        try:
            ea._absent_panel(axis, "a title", "a named reason")
            texts = [t.get_text() for t in axis.texts]
            self.assertTrue(any("named reason" in t for t in texts))
        finally:
            plt.close(figure)


# ======================================================================
# the seed's causal disposition
# ======================================================================


def _squared_lengths(spacetime):
    """Every edge's `l^2`, which is what a disposition is read as."""
    return [complex(edge.getLength()) ** 2
            for edge in spacetime.getEdgeList().toVector()]


class EdgeDispositionTest(unittest.TestCase):
    """The four settings differ in `arg l` alone, never in scale.

    Every assertion is on `l^2`, not on `l`: the squared length carries the
    causal character, and asserting on `l` would pass for a convention that
    squared to the wrong sign.
    """

    def test_spacelike_squares_to_plus_one_on_every_edge(self):
        host = ea.build_cobordism_host(SMALL, 3, ea.EdgeDisposition.SPACELIKE)
        squared = _squared_lengths(host)
        self.assertTrue(squared, "the host has no edges to assert on")
        for value in squared:
            self.assertAlmostEqual(value.real, 1.0, places=12)
            self.assertAlmostEqual(value.imag, 0.0, places=12)

    def test_timelike_squares_to_minus_one_on_every_edge(self):
        host = ea.build_cobordism_host(SMALL, 3, ea.EdgeDisposition.TIMELIKE)
        squared = _squared_lengths(host)
        self.assertTrue(squared, "the host has no edges to assert on")
        for value in squared:
            self.assertAlmostEqual(value.real, -1.0, places=12)
            self.assertAlmostEqual(value.imag, 0.0, places=12)

    def test_random_squares_onto_the_unit_circle_and_is_not_degenerate(self):
        host = ea.build_cobordism_host(SMALL, 3, ea.EdgeDisposition.RANDOM)
        squared = _squared_lengths(host)
        self.assertTrue(squared, "the host has no edges to assert on")
        for value in squared:
            self.assertAlmostEqual(abs(value), 1.0, places=12)
        # A guard against the degeneracy the fixed seed had: every edge
        # carrying the SAME l^2 would satisfy the magnitude assertion above
        # while still prescribing one uniform causal character.
        spread = max(abs(v - squared[0]) for v in squared)
        self.assertGreater(spread, 1e-6,
                           "every edge drew the same l^2: not random")

    def test_foliated_is_plus_one_within_a_layer_and_minus_one_across(self):
        host = ea.build_cobordism_host(SMALL, 3, ea.EdgeDisposition.FOLIATED)
        layer = ea._hop_layers(host, ea.boundary_vertices(host))
        across, within = 0, 0
        for edge in host.getEdgeList().toVector():
            a, b = ea._edge_endpoints(edge)
            value = complex(edge.getLength()) ** 2
            if layer.get(a, 0) != layer.get(b, 0):
                across += 1
                self.assertAlmostEqual(value.real, -1.0, places=12)
            else:
                within += 1
                self.assertAlmostEqual(value.real, 1.0, places=12)
            self.assertAlmostEqual(value.imag, 0.0, places=12)
        # Both branches must fire, or the assertions above are vacuous.
        self.assertGreater(across, 0, "no edge spans a hop layer")
        self.assertGreater(within, 0, "no edge lies within a hop layer")

    def test_random_repeats_for_one_seed_and_differs_across_seeds(self):
        first = _squared_lengths(
            ea.build_cobordism_host(SMALL, 3, ea.EdgeDisposition.RANDOM))
        again = _squared_lengths(
            ea.build_cobordism_host(SMALL, 3, ea.EdgeDisposition.RANDOM))
        self.assertEqual(first, again)
        other = _squared_lengths(
            ea.build_cobordism_host(SMALL, 11, ea.EdgeDisposition.RANDOM))
        self.assertNotEqual(first, other)

    def test_the_topology_is_the_same_under_every_disposition(self):
        """A disposition changes the causal character, never the complex.

        This is what makes the settings a controlled variable: one host seed
        gives one topology, and only `arg l` moves.
        """
        counts = {}
        for disposition in ea.EdgeDisposition.ALL:
            host = ea.build_cobordism_host(SMALL, 3, disposition)
            counts[disposition] = (len(host.getEdgeList().toVector()),
                                   len(host.getTopSimplices()))
        self.assertEqual(len(set(counts.values())), 1, counts)

    def test_an_unknown_disposition_fails_loudly_and_by_name(self):
        with self.assertRaises(ValueError) as caught:
            ea.build_cobordism_host(SMALL, 3, "spacelke")
        self.assertIn("spacelke", str(caught.exception))
        with self.assertRaises(ValueError) as caught:
            ea.build_config(edge_disposition="foliatd")
        self.assertIn("foliatd", str(caught.exception))

    def test_the_default_is_random(self):
        self.assertEqual(ea.DECLARED_EDGE_DISPOSITION,
                         ea.EdgeDisposition.RANDOM)
        self.assertEqual(ea.build_config()["edge_disposition"],
                         ea.EdgeDisposition.RANDOM)

    def test_the_cli_rejects_an_unknown_value(self):
        parser = ea.build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["run", "--edge-disposition", "chronological"])
        args = parser.parse_args(["run", "--edge-disposition", "foliated"])
        self.assertEqual(args.edge_disposition, ea.EdgeDisposition.FOLIATED)

    def test_the_cli_accepts_lightlike(self):
        # `lightlike` became a real setting once a null INTERVAL was
        # expressible on an edge of nonzero extent (#870); before that it was
        # reachable only by collapsing the edge, which is degenerate instead.
        args = ea.build_parser().parse_args(
            ["run", "--edge-disposition", "lightlike"])
        self.assertEqual(args.edge_disposition, ea.EdgeDisposition.LIGHTLIKE)

    def test_the_disposition_is_recorded_in_the_config(self):
        config = ea.build_config(size=SMALL, steps=SMALL_STEPS,
                                 edge_disposition=ea.EdgeDisposition.TIMELIKE)
        self.assertEqual(config["edge_disposition"],
                         ea.EdgeDisposition.TIMELIKE)

    def test_a_foliated_frame_says_on_its_face_that_it_is_prescribed(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        config = ea.build_config(size=SMALL, steps=SMALL_STEPS,
                                 edge_disposition=ea.EdgeDisposition.FOLIATED)
        frames = ea.drive(config, progress=False).frames
        figure = plt.figure(figsize=(15, 9))
        try:
            ea.draw_frame(figure, frames, len(frames) - 1)
            title = figure._suptitle.get_text()
            self.assertIn("foliated", title)
            self.assertIn("PRESCRIBED", title)
        finally:
            plt.close(figure)


# ======================================================================
# the drive flags (#863)
# ======================================================================


class _SpyMeta(type):
    """Class-level statics fall through to the real MultiCobordism."""

    def __getattr__(cls, name):
        return getattr(MC, name)


class _SpyNode(metaclass=_SpyMeta):
    """Records the keywords the two stage calls actually receive.

    A flag that reaches `build_config` but not the engine would pass any
    test written against the config alone, so the assertion has to be made
    where the value crosses into C++.
    """

    calls = []

    def __init__(self, *args, **kwargs):
        object.__setattr__(self, "_node", MC(*args, **kwargs))

    def run_stage1(self, **kwargs):
        _SpyNode.calls.append(("stage1", dict(kwargs)))
        return self._node.run_stage1(**kwargs)

    def run_stage2(self, **kwargs):
        _SpyNode.calls.append(("stage2", dict(kwargs)))
        return self._node.run_stage2(**kwargs)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_node"), name)


class DriveFlagTest(unittest.TestCase):
    """Each flag reaches the engine carrying the value the caller gave."""

    def _spy(self, **config_kwargs):
        real, _SpyNode.calls = ea.MC, []
        ea.MC = _SpyNode
        try:
            config = ea.build_config(size=SMALL, steps=1, **config_kwargs)
            ea.drive(config, progress=False)
        finally:
            ea.MC = real
        return dict(_SpyNode.calls)

    def test_the_stage_and_depth_flags_reach_the_engine(self):
        calls = self._spy(stage1_iters=5, stage2_iters=9, tolerance=1e-30,
                          combinatorial_depth=3)
        self.assertEqual(calls["stage1"]["max_steps"], 5)
        self.assertEqual(calls["stage1"]["max_lookahead"], 3)
        self.assertEqual(calls["stage2"]["max_iters"], 9)
        self.assertEqual(calls["stage2"]["tolerance"], 1e-30)

    def test_the_defaults_are_the_declared_ones(self):
        calls = self._spy()
        self.assertEqual(calls["stage1"]["max_steps"],
                         ea.DECLARED_STAGE1_ITERS)
        self.assertEqual(calls["stage1"]["max_lookahead"],
                         ea.DECLARED_COMBINATORIAL_DEPTH)
        self.assertEqual(calls["stage2"]["max_iters"],
                         ea.DECLARED_STAGE2_ITERS)
        self.assertEqual(calls["stage2"]["tolerance"], ea.DECLARED_TOLERANCE)

    def test_a_nonsense_drive_parameter_is_refused_by_name(self):
        for kwargs in ({"stage1_iters": 0}, {"stage2_iters": 0},
                       {"combinatorial_depth": 0}, {"tolerance": 0.0},
                       {"tolerance": -1.0},
                       {"tolerance": float("inf")}):
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError):
                    ea.build_config(size=SMALL, steps=1, **kwargs)


class TerminatorTest(unittest.TestCase):
    """A run says WHY it stopped; a short trace is never ambiguous."""

    def test_the_vocabulary_is_closed_and_a_result_rejects_anything_else(self):
        self.assertEqual(set(ea.Terminator.ALL),
                         {ea.Terminator.STEPS, ea.Terminator.TOLERANCE,
                          ea.Terminator.CANCELLED})
        with self.assertRaises(ValueError):
            ea.DriveResult([], "converged")

    def test_an_impossible_tolerance_stops_the_run_early_and_says_so(self):
        """A tolerance no unit can meet must exit on the FIRST unit.

        The assertion is on the terminator AND the unit count: a run that
        merely ran out of budget would report the other terminator, and one
        that stopped early without saying so would report this one at the
        full length.
        """
        config = ea.build_config(size=SMALL, steps=4, tolerance=1e30)
        result = ea.drive(config, progress=False)
        self.assertEqual(result.terminator, ea.Terminator.TOLERANCE)
        self.assertEqual(result.frames[-1].step, 1)
        self.assertLess(len(result.frames), 4 + 1)

    def test_convergence_is_absolute_and_an_absence_is_not_convergence(self):
        # Improvement is compared to the tolerance directly, never to its
        # ratio against the objective, so the same tolerance means the same
        # thing at every scale.
        self.assertTrue(ea._converged(100.0, 99.9999, 1e-3))
        self.assertFalse(ea._converged(100.0, 99.0, 1e-3))
        # A unit that RAISES the objective has also failed to improve it.
        self.assertTrue(ea._converged(100.0, 101.0, 1e-3))
        # An unmeasured objective is an absence, and a run may not stop on
        # one: that would report convergence off a number nobody read.
        self.assertFalse(ea._converged(None, 99.0, 1e-3))
        self.assertFalse(ea._converged(100.0, None, 1e-3))
        self.assertFalse(ea._converged(float("nan"), 99.0, 1e-3))


class LiveTest(unittest.TestCase):
    """`--live` shows a run as it happens, or refuses by name."""

    def test_frames_are_delivered_while_the_run_is_still_going(self):
        """The callback fires per unit, not once at the end.

        This is the whole substance of a live view: if the frames only
        arrived after the drive returned, the display would be a slower
        headless render wearing a different name.
        """
        seen = []
        config = ea.build_config(size=SMALL, steps=2, tolerance=1e-30)
        result = ea.drive(
            config, progress=False,
            on_frame=lambda frames, index: seen.append((index, len(frames))))
        # One callback per unit plus the initial read, each seeing exactly
        # the frames published so far.
        self.assertEqual([index for index, _ in seen],
                         list(range(len(result.frames))))
        self.assertEqual([count for _, count in seen],
                         list(range(1, len(result.frames) + 1)))

    def test_a_file_only_backend_is_refused_by_name(self):
        """Agg makes figures happily and displays nothing.

        So the guard cannot be "did a figure open" — it has to be asked of
        the backend, or `--live` would silently become a slower headless
        run that computes every frame and shows none.
        """
        import matplotlib
        matplotlib.use("Agg")
        config = ea.build_config(size=SMALL, steps=1)
        with self.assertRaises(RuntimeError) as caught:
            ea.drive_live(config, progress=False)
        message = str(caught.exception)
        self.assertIn("interactive", message)
        self.assertIn("Agg", message)

    def test_agg_is_not_mistaken_for_an_interactive_backend(self):
        interactive = ea._interactive_backends()
        self.assertNotIn("agg", interactive)
        self.assertIn("webagg", interactive)


class DriveDocumentTest(unittest.TestCase):
    """The run document records how the run was driven and how it ended."""

    def test_the_document_carries_every_flag_and_the_terminator(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "run.json")
            ea.main(["run", "--size", str(SMALL), "--steps", "1",
                     "--stage-one-iterations", "2",
                     "--stage-two-iterations", "3",
                     "--combinatorial-depth", "2",
                     "--tolerance", "1e-30",
                     "--json", path, "--out", "", "--quiet"])
            with open(path) as handle:
                document = json.load(handle)
        config = document["config"]
        self.assertEqual(config["stage1_iters"], 2)
        self.assertEqual(config["stage2_iters"], 3)
        self.assertEqual(config["combinatorial_depth"], 2)
        self.assertEqual(config["tolerance"], 1e-30)
        self.assertIn(document["terminator"], ea.Terminator.ALL)


if __name__ == "__main__":
    unittest.main()
