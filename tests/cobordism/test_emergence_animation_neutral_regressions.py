# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Focused regressions for the neutral emergence-animation readout."""

import json
import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))),
    "examples", "cobordism"))

import emergence_animation as ea  # noqa: E402


class _Fiber:

    def __init__(self, name="fiber"):
        self.name = name

    @staticmethod
    def accepted():
        return True

    @staticmethod
    def degree():
        return 1

    @staticmethod
    def rank():
        return 3

    @staticmethod
    def certificate():
        return SimpleNamespace(
            lowerGap=0.5, upperGap=1.5, localization=0.25,
            localizationExcess=0.0, gramDefect=0.0)


def _quark(winding):
    return SimpleNamespace(
        classification="quark", determinantWinding=winding, colorRank=3,
        triangleAnchorScore=1.0, triangleAnchorMaxTerm=1.0,
        triangleAnchorParticipation=3.0, anchorPhaseDispersion=0.0,
        anchorPhaseCoherence=1.0, failedCertificates=[])


class NeutralReadoutRegressionTest(unittest.TestCase):

    def test_complex_nonfinite_parts_are_strict_json_safe(self):
        safe = ea._json_safe({
            "outer": [complex(float("nan"), float("inf")),
                      {"finite": complex(2.0, -3.0)}],
        })
        self.assertEqual(safe["outer"][0], [None, None])
        self.assertEqual(safe["outer"][1]["finite"], [2.0, -3.0])
        json.dumps(safe, allow_nan=False)

    def test_band_failure_keeps_candidate_and_component_slots_aligned(self):
        fiber = _Fiber()
        components = [SimpleNamespace(id="first", support=[0]),
                      SimpleNamespace(id="second", support=[1])]

        class Tracker:

            def __init__(self, _spacetime, _settings):
                self.calls = 0

            def enumerateBands(self, _support, _degree):
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("first component failed")
                return SimpleNamespace(fibers=[fiber])

        captured = []

        class Classifier:

            @staticmethod
            def classifyQuark(evidence):
                captured.append(evidence.component)
                return _quark(1)

        frame = object.__new__(ea.EmergenceFrame)
        frame.components = components
        with mock.patch.object(ea.obs, "SpectralFiberTracker", Tracker):
            bands = frame._read_bands(object(), {"degrees": [1]})
        self.assertEqual(bands["accepted"], 1)
        self.assertEqual(frame.candidates, [None, fiber])
        self.assertEqual(frame.candidate_components, components)

        frame.states = [None, None]
        evidence_type = lambda: SimpleNamespace()  # noqa: E731
        with mock.patch.object(ea.obs, "QuarkCandidateEvidence",
                               evidence_type), \
                mock.patch.object(ea.obs, "ParticleClusters", Classifier):
            frame._read_anchors()
        self.assertEqual(captured, ["second"])
        self.assertIsNone(frame.candidate_quarks[0])
        self.assertIs(frame.candidate_quarks[1], frame.quarks[0])

    def test_transport_exception_is_a_named_rejected_read(self):
        class Connection:

            @staticmethod
            def transportOnSpacetime(_spacetime, _to_fiber, _from_fiber):
                raise RuntimeError("singular overlap")

        frame = object.__new__(ea.EmergenceFrame)
        frame.candidates = [_Fiber("a"), _Fiber("b")]
        with mock.patch.object(ea.obs, "FiberConnection", Connection):
            result = frame._read_transports(object())
        self.assertNotIsInstance(result, ea.Absent)
        self.assertEqual(result["total"], 2)
        self.assertEqual(result["accepted"], 0)
        self.assertTrue(all("singular overlap" in row["reason"]
                            for row in result["rows"]))

    def test_crossings_use_regular_re_tau_level_and_actual_quark_reads(self):
        temporal = SimpleNamespace(
            certified=True, tau=[0j, 10 + 4j, 20 - 2j],
            failedCertificates=[])
        calls = {"mass": [], "baryon": [], "levels": []}
        mass_read = SimpleNamespace(
            crossingMass=3.0, admissibleCrossings=3, refusedCrossings=0,
            calibrated=True, units="proper-time")
        baryon_read = SimpleNamespace(
            baryonNumber=1.0, quarkTubes=3, signDefects=[])

        class Tube:

            def __init__(self):
                self.determinantWinding = None

        class CrossingReadouts:

            @staticmethod
            def temporalFunction(_spacetime, _boundary):
                return temporal

            @staticmethod
            def crossingMass(tubes, _temporal, level, _reference):
                calls["mass"].append(list(tubes))
                calls["levels"].append(level)
                return mass_read

            @staticmethod
            def baryonNumber(tubes, _temporal, level, _reference):
                calls["baryon"].append(list(tubes))
                calls["levels"].append(level)
                return baryon_read

            @staticmethod
            def chargePowerProfile(_tubes, _temporal, _level):
                return SimpleNamespace(
                    eigenvalues=[], power=[], normalized=False, monopole=0.0,
                    failedCertificates=[])

            @staticmethod
            def crossing(tube, _temporal, level):
                calls["levels"].append(level)
                return SimpleNamespace(
                    tubeId=tube.tubeId, sign=1, admissible=True,
                    perpendicular=1 + 0j, failedCertificates=[])

        fibers = [_Fiber("zero"), _Fiber("one"), _Fiber("two"),
                  _Fiber("three")]
        quarks = [_quark(1), None, _quark(None), _quark(-1)]
        frame = object.__new__(ea.EmergenceFrame)
        frame.candidates = fibers
        frame.candidate_quarks = quarks
        with mock.patch.object(ea.EmergenceFrame, "_m0_vertices",
                               return_value=[0]), \
                mock.patch.object(ea.obs, "WorldTubeInput", Tube), \
                mock.patch.object(ea.obs, "CrossingReadouts",
                                  CrossingReadouts):
            result = frame._read_crossings(object())

        # The range midpoint 10 is a vertex value, so the nearest open-gap
        # midpoint is used. The integer layer labels are never consulted.
        self.assertEqual(result["level"], 5.0)
        self.assertTrue(all(level == 5.0 for level in calls["levels"]))
        tubes = calls["mass"][0]
        self.assertEqual([tube.tubeId for tube in tubes],
                         ["band-0", "band-1", "band-2", "band-3"])
        self.assertEqual([tube.certifiedQuarkTube for tube in tubes],
                         [True, False, True, True])
        self.assertEqual([tube.determinantWinding for tube in tubes],
                         [1, None, None, -1])
        self.assertEqual(frame.crossing_candidate_quarks,
                         [quarks[0], quarks[2], quarks[3]])
        self.assertIs(frame.crossing_mass_read, mass_read)
        self.assertIs(frame.crossing_baryon_read, baryon_read)

        captured = {}

        class Evidence:
            pass

        class Classifier:

            @staticmethod
            def classifyBaryon(evidence):
                captured["evidence"] = evidence
                return SimpleNamespace(
                    classification="candidate", confidence=0.5,
                    failedCertificates=["binding"])

        frame.quarks = [quarks[0], quarks[2], quarks[3]]
        with mock.patch.object(ea.obs, "BaryonCandidateEvidence", Evidence), \
                mock.patch.object(ea.obs, "ParticleClusters", Classifier):
            frame._read_verdict()
        evidence = captured["evidence"]
        self.assertEqual(evidence.quarks, frame.quarks)
        self.assertIs(evidence.crossingMass, mass_read)
        self.assertIs(evidence.crossingBaryon, baryon_read)

    def test_candidate_crossing_failures_survive_into_verdict_and_json(self):
        temporal = SimpleNamespace(
            certified=True, tau=[0j, 1 + 0j], failedCertificates=[])
        mass_read = SimpleNamespace(
            crossingMass=4.0, admissibleCrossings=4, refusedCrossings=0,
            calibrated=True, units="proper-time")
        baryon_read = SimpleNamespace(
            baryonNumber=1.0, quarkTubes=3, signDefects=[])

        class Tube:

            def __init__(self):
                self.determinantWinding = None

        class CrossingReadouts:

            @staticmethod
            def temporalFunction(_spacetime, _boundary):
                return temporal

            @staticmethod
            def crossingMass(tubes, _temporal, _level, _reference):
                if len(tubes) == 3:
                    raise RuntimeError("candidate mass exploded")
                return mass_read

            @staticmethod
            def baryonNumber(tubes, _temporal, _level, _reference):
                if len(tubes) == 3:
                    raise RuntimeError("candidate baryon exploded")
                return baryon_read

            @staticmethod
            def chargePowerProfile(_tubes, _temporal, _level):
                return SimpleNamespace(
                    eigenvalues=[], power=[], normalized=False, monopole=0.0,
                    failedCertificates=[])

            @staticmethod
            def crossing(tube, _temporal, _level):
                return SimpleNamespace(
                    tubeId=tube.tubeId, sign=1, admissible=True,
                    perpendicular=1 + 0j, failedCertificates=[])

        failures = [
            "candidate crossing mass failed: candidate mass exploded",
            "candidate baryon sum failed: candidate baryon exploded",
        ]
        for candidate_count in (4, 3):
            with self.subTest(candidate_count=candidate_count):
                quarks = [_quark(1), _quark(1), _quark(1)]
                quarks.extend([None] * (candidate_count - 3))
                frame = object.__new__(ea.EmergenceFrame)
                frame.candidates = [
                    _Fiber(str(index)) for index in range(candidate_count)]
                frame.candidate_quarks = quarks
                with mock.patch.object(ea.EmergenceFrame, "_m0_vertices",
                                       return_value=[0]), \
                        mock.patch.object(ea.obs, "WorldTubeInput", Tube), \
                        mock.patch.object(ea.obs, "CrossingReadouts",
                                          CrossingReadouts):
                    crossings = frame._read_crossings(object())

                self.assertEqual(
                    crossings["candidateReadFailures"], failures)
                self.assertIsNone(frame.crossing_mass_read)
                self.assertIsNone(frame.crossing_baryon_read)

                captured = {}

                class Evidence:
                    pass

                class Classifier:

                    @staticmethod
                    def classifyBaryon(evidence):
                        captured["evidence"] = evidence
                        return SimpleNamespace(
                            classification="candidate", confidence=0.5,
                            failedCertificates=["binding"])

                frame.quarks = quarks[:3]
                with mock.patch.object(
                        ea.obs, "BaryonCandidateEvidence", Evidence), \
                        mock.patch.object(
                            ea.obs, "ParticleClusters", Classifier):
                    verdict = frame._read_verdict()
                self.assertFalse(
                    hasattr(captured["evidence"], "crossingMass"))
                self.assertFalse(
                    hasattr(captured["evidence"], "crossingBaryon"))
                self.assertEqual(verdict, {
                    "classification": "candidate", "confidence": 0.5,
                    "reasons": ["binding"], "readFailures": failures,
                })
                document = json.dumps(
                    ea._json_safe(
                        {"crossings": crossings, "verdict": verdict}),
                    allow_nan=False)
                self.assertIn("candidate mass exploded", document)
                self.assertIn("candidate baryon exploded", document)

    def test_mass_panel_handles_each_absent_measurement_independently(self):
        cases = [
            ({"crossingMass": 2.5,
              "baryonNumber": ea.Absent("baryon unavailable"),
              "units": "u"}, "crossing mass: 2.5", "B = n/a"),
            ({"crossingMass": ea.Absent("mass unavailable"),
              "baryonNumber": 1.0}, "crossing mass: n/a", "B = 1"),
        ]
        for crossings, mass_text, baryon_text in cases:
            with self.subTest(crossings=crossings):
                figure, axis = plt.subplots()
                try:
                    ea._panel_mass(axis, SimpleNamespace(crossings=crossings))
                    text = "\n".join(item.get_text() for item in axis.texts)
                    self.assertIn(mass_text, text)
                    self.assertIn(baryon_text, text)
                    self.assertIn("unavailable", text)
                finally:
                    plt.close(figure)

    def test_panels_do_not_draw_unmeasured_values_as_zero(self):
        figure, axes = plt.subplots(1, 3)
        try:
            ea._panel_bands(axes[0], SimpleNamespace(bands={
                "rows": [{"accepted": True, "rank": 3,
                          "lowerGap": None}],
            }))
            self.assertEqual(len(figure.axes), 3,
                             "an unmeasured gap should not create a trace")
            self.assertIn("unmeasured",
                          " ".join(text.get_text()
                                   for text in axes[0].texts))

            ea._panel_anchors(axes[1], SimpleNamespace(anchors={
                "rows": [{"score": 0.75, "maxTerm": None,
                          "participationRatio": None,
                          "phaseDispersion": None}],
                "certified": 0,
            }))
            self.assertEqual(len(axes[1].patches), 1)
            self.assertEqual([tick.get_text()
                              for tick in axes[1].get_yticklabels()],
                             ["score"])

            ea._panel_betti(axes[2], SimpleNamespace(betti={
                "numbers": {0: 2, 1: None, 2: 3},
            }))
            self.assertEqual([bar.get_height() for bar in axes[2].patches],
                             [2, 3])
            self.assertEqual([tick.get_text()
                              for tick in axes[2].get_xticklabels()],
                             ["0", "2"])
        finally:
            plt.close(figure)

    def test_layout_label_names_the_classifying_quantity(self):
        figure, axis = plt.subplots()
        try:
            ea._panel_layout(
                axis, SimpleNamespace(layout=ea.Absent("no layout")))
            self.assertIn("arg(l^2)", axis.get_title())
            self.assertNotIn("Re(l^2)", axis.get_title())
        finally:
            plt.close(figure)


if __name__ == "__main__":
    unittest.main()
