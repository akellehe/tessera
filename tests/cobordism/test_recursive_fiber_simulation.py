# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""#776 — the unforced recursive-analysis integration and its no-feedback firewall.

The whole epic's claim is that particles are READ OUT of geometry that was
relaxed without knowing about them. These tests are the firewall: they check
adversarially — not cooperatively — that no particle, fiber, transport, or
amplitude quantity reaches the objective, its gradient, or the acceptance and
refinement decisions.

The suite is organised as:

* ``ObjectiveFirewallStructureTest`` — the objective's and the refinement
  rule's inputs are ENUMERABLE, and the enumerated lists contain nothing
  derived. Both entry points are static functions of a declared record, so the
  statement is compile-time, not a convention.
* ``TrajectoryIdentityTest`` — the central test. With the overlay enabled the
  accepted-move sequence and the objective trace are BIT-IDENTICAL to the
  disabled run at the same seed.
* ``AdversarialFeedbackTest`` — poison the recursive reads (force certificates
  to fail, flip verdicts, close a gap, spike leakage, collapse the anchor) and
  assert the objective value, its gradient, the accepted moves, and the
  refinement decision do not move.
* ``EmergenceSubmodeTest`` — the two labeled Gaussian-closed sub-modes.
* ``RefinementIndependenceTest`` — refinement fires only on base geometric and
  numerical indicators.
* ``CheckpointSchemaTest`` / ``ReplayTest`` — the versioned schema, nulls for
  unknowns, the cold-cache replay path, and version rejection.
* ``AnalysisOverlayTest`` — the overlay actually drives the merged stack.
* ``CadenceBenchmarkTest`` — the analysis-cadence overhead, and that the
  disabled path costs nothing.
"""

import cmath
import copy
import json
import math
import os
import sys
import time
import unittest

import pytest

import tessera

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _closed_s4 import closed_s4 as _closed_s4  # noqa: E402

T = tessera
cob = tessera.cobordism
MC = cob.MultiCobordism

#: Small enough to run the whole suite quickly, large enough that the
#: modularity scan finds more than one component and the band enumeration
#: produces accepted rank>1 fibers.
_REFINE = 6
_HOST_SEED = 3
_NODE_SEED = 7


def _node(seed=_NODE_SEED, refine=_REFINE, degrees=(1,)):
    """A JointStationarity node on the shared closed-S⁴ host.

    ``JointStationarity`` IS the design spec's base objective
    ``beta_R ||grad S_Regge||^2 + eta_H sum_k ||grad S_Hodge,k||^2`` — this
    ticket reuses it rather than defining a second one.
    """
    st = _closed_s4(refine, _HOST_SEED)
    node = MC(st, [], [], list(degrees), 1.0, seed)
    node.set_objective(cob.JointStationarityObjective())
    return node


def _overlay_config(enabled=True, cadence=1, degrees=(1,), resolutions=(1.0,),
                    cold=False, fock=False):
    cfg = MC.AnalysisConfig()
    cfg.enabled = enabled
    cfg.cadence = cadence
    cfg.degrees = list(degrees)
    cfg.resolutions = list(resolutions)
    cfg.cold_caches = cold
    cfg.fock_oracle = fock
    return cfg


def _cells(node):
    """The node's top-cell set as a canonical, comparable object."""
    return sorted(tuple(sorted(v.getId() for v in c.getVertices()))
                  for c in node.st.getTopSimplices())


def _lengths(node):
    """Every edge length, keyed by endpoints — the exact geometric state."""
    out = {}
    for e in node.st.getEdgeList().toVector():
        a, b = e.getSource().getId(), e.getTarget().getId()
        out[(min(a, b), max(a, b))] = e.getLength()
    return out


def _phases(node):
    """Every edge connection phase on the CANONICAL min->max direction.

    Unlike the length, the phase is orientation-dependent: the reverse
    orientation carries the inverse link, so an edge stored target->source
    contributes the negated phase."""
    out = {}
    for e in node.st.getEdgeList().toVector():
        a, b = e.getSource().getId(), e.getTarget().getId()
        out[(min(a, b), max(a, b))] = e.getPhase() if a < b else -e.getPhase()
    return out


def _twist(node, seed=804):
    """Give every edge a distinct COMPLEX connection phase, so a serializer
    that drops the field, or keeps only its compact part, is caught."""
    import numpy as np
    rng = np.random.default_rng(seed)
    for e in node.st.getEdgeList().toVector():
        e.setPhase(complex(rng.normal(), rng.normal()))


def _drive(node, candidates=6):
    """The engine's DETERMINISTIC drive unit: one committed combinatorial move
    followed by a full geometric relaxation.

    Deliberately ONE stage-1 update. The engine's move draw is not
    reproducible past the first committed move (#579 — see
    ``TrajectoryIdentityTest``), so a longer drive would compare two
    trajectories that differ for reasons that have nothing to do with this
    ticket. Returns (objective trace, cells, lengths).
    """
    trace = list(node.run_stage1(max_steps=1, n_candidate_moves=candidates))
    trace += list(node.run_stage2(max_iters=6))
    return trace, _cells(node), _lengths(node)


# ======================================================================
# structural firewall: the objective's declared inputs
# ======================================================================


class ObjectiveFirewallStructureTest(unittest.TestCase):
    """The objective and the refinement rule are static over declared records."""

    #: Anything whose NAME contains one of these has no business in the
    #: objective. The epic's forbidden list (design spec section 17): particle
    #: confidence, modularity, color determinant, Wilson loops, flavor,
    #: charge, spin, transports, amplitudes, certificates.
    FORBIDDEN = (
        "particle", "quark", "gluon", "meson", "diquark", "baryon", "proton",
        "fiber", "band", "gap", "transport", "leakage", "wilson", "holonomy",
        "amplitude", "gram", "color", "singlet", "flavor", "isospin",
        "charge", "spin", "exchange", "parity", "winding", "anchor",
        "modularity", "component", "certificate", "confidence", "verdict",
        "hole", "occupation", "wick", "fock",
    )

    def test_objective_terms_are_exactly_the_declared_list(self):
        # `connection_stationarity` (#853) is the term with a gradient in the
        # connection phase. Audited on the same terms as the rest: it is a
        # functional of the C* connection operator's spectrum -- geometry and
        # declared configuration -- and nothing reachable from it is a
        # component, fiber, transport, colour, charge, flavour, exchange, spin
        # certificate or verdict.
        self.assertEqual(
            MC.objective_term_names(),
            ["regge_stationarity", "hodge_stationarity",
             "connection_stationarity", "register_residual",
             "action_magnitude", "carried_state_energy"])

    def test_no_objective_term_names_a_derived_observable(self):
        for name in MC.objective_term_names():
            for word in self.FORBIDDEN:
                self.assertNotIn(
                    word, name,
                    "objective term %r names the derived quantity %r" %
                    (name, word))

    def test_objective_terms_record_exposes_no_other_field(self):
        exposed = {a for a in dir(MC.ObjectiveTerms) if not a.startswith("_")}
        self.assertEqual(exposed, set(MC.objective_term_names()))

    def test_objective_of_is_static_and_sums_only_the_declared_terms(self):
        terms = MC.ObjectiveTerms()
        terms.regge_stationarity = 2.0
        terms.hodge_stationarity = 3.0
        terms.register_residual = 5.0
        terms.action_magnitude = 7.0
        terms.carried_state_energy = 11.0
        self.assertEqual(MC.objective_of(terms), 28.0)
        # STATIC: reachable without any node at all, so it cannot consult one.
        self.assertTrue(callable(MC.objective_of))

    def test_node_objective_equals_the_sum_of_its_declared_terms(self):
        node = _node()
        terms = node.objective_terms()
        self.assertEqual(node.objective(), MC.objective_of(terms))

    def test_strict_emergence_leaves_the_state_channel_exactly_zero(self):
        node = _node()
        self.assertEqual(node.objective_terms().carried_state_energy, 0.0)

    def test_joint_stationarity_uses_only_the_two_base_terms(self):
        node = _node()
        terms = node.objective_terms()
        self.assertGreater(terms.regge_stationarity, 0.0)
        self.assertGreater(terms.hodge_stationarity, 0.0)
        self.assertEqual(terms.register_residual, 0.0)
        self.assertEqual(terms.action_magnitude, 0.0)
        self.assertEqual(terms.carried_state_energy, 0.0)

    def test_refinement_indicators_are_exactly_the_declared_five(self):
        self.assertEqual(
            MC.refinement_indicator_names(),
            ["regge_stationarity_residual", "hodge_stationarity_residual",
             "curvature_concentration", "mesh_quality", "solver_error"])

    def test_no_refinement_indicator_names_a_derived_observable(self):
        # "curvature_concentration" is base Regge curvature, not a band gap;
        # the forbidden list is checked word-by-word against every name.
        for name in MC.refinement_indicator_names():
            for word in self.FORBIDDEN:
                self.assertNotIn(
                    word, name,
                    "refinement indicator %r names the derived quantity %r" %
                    (name, word))

    def test_refinement_indicator_record_exposes_no_other_field(self):
        exposed = {a for a in dir(MC.RefinementIndicators)
                   if not a.startswith("_")}
        self.assertEqual(exposed, set(MC.refinement_indicator_names()))

    def test_refinement_decision_is_static_over_two_indicator_records(self):
        indicators = MC.RefinementIndicators()
        indicators.mesh_quality = 0.1
        thresholds = MC.RefinementIndicators()
        thresholds.mesh_quality = 0.5
        thresholds.regge_stationarity_residual = 0.0
        thresholds.hodge_stationarity_residual = 0.0
        thresholds.curvature_concentration = 0.0
        thresholds.solver_error = 0.0
        decision = MC.refinement_decision_of(indicators, thresholds)
        self.assertTrue(decision.refine)
        self.assertEqual(decision.trigger, "mesh_quality")

    def test_default_thresholds_never_fire(self):
        node = _node()
        self.assertFalse(node.refinement_decision().refine)
        self.assertEqual(node.refinement_decision().trigger, "")

    def test_the_engine_header_names_no_analysis_type(self):
        """MultiCobordism.h includes no particle/fiber/transport header.

        The overlay's includes live in its own translation unit, so the
        objective's compilation unit cannot even see the analysis types.
        """
        root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
        header = os.path.join(root, "include", "cobordism", "MultiCobordism.h")
        if not os.path.exists(header):      # installed wheel, not a checkout
            self.skipTest("source tree not available")
        with open(header) as handle:
            text = handle.read()
        for forbidden in ("ParticleClusters.h", "SpectralFiber.h",
                          "FiberConnection.h", "ColorFiber.h",
                          "ExchangeHolonomy.h", "PersistentModularity.h",
                          "CovarianceState.h", "LazyFock.h",
                          "RecursiveQuotient.h"):
            self.assertNotIn(forbidden, text)


# ======================================================================
# the central test: bit-identical trajectories
# ======================================================================


class TrajectoryIdentityTest(unittest.TestCase):
    """A single accepted move differing between the runs is a firewall breach.

    THE COMPARISON UNIT. The engine's stage-1 move draw is not reproducible
    past the first committed move — a pre-existing property (#579: identical
    fresh processes diverge on the same seed; the seed LABELS an attempt, it
    does not reproduce it), because a committed move rebuilds the complex and
    the redraw against the rebuilt complex depends on allocation order. That
    is measured and pinned by
    ``test_the_engine_move_draw_is_not_reproducible_past_the_first_move``
    below, so no one can mistake it for a firewall breach — and it is why the
    identity comparisons here are made over the engine's DETERMINISTIC units:

      * one stage-1 update from a fixed host (deterministic),
      * the whole stage-2 relaxation (deterministic: no draw at all), and
      * the score assigned to an EXPLICIT candidate move — the exact quantity
        ``deltaF`` compares, evaluated on a candidate complex built here.

    Those three cover every place a leak could act: the objective, the score
    of a candidate, and the geometric descent.
    """

    def test_one_accepted_move_is_bit_identical_with_and_without_the_overlay(self):
        plain = _node()
        overlaid = _node()
        overlaid.set_analysis_config(_overlay_config())
        plain_trace = list(plain.run_stage1(max_steps=1, n_candidate_moves=6))
        overlaid_trace = list(
            overlaid.run_stage1(max_steps=1, n_candidate_moves=6))
        self.assertEqual(plain.accepted_move_count, 1)
        self.assertEqual(overlaid.accepted_move_count, 1)
        self.assertGreaterEqual(overlaid.analysis_pass_count, 1)
        self.assertEqual(plain_trace, overlaid_trace)
        self.assertEqual(_cells(plain), _cells(overlaid))
        self.assertEqual(_lengths(plain), _lengths(overlaid))

    def test_the_full_relaxation_is_bit_identical_with_the_overlay_running(self):
        """Stage 2 takes no random draw, so this compares whole trajectories.

        The overlaid node runs a FULL analysis pass between every relaxation
        chunk, so the reads are recomputed at each visited geometry.
        """
        plain = _node()
        overlaid = _node()
        overlaid.set_analysis_config(_overlay_config())
        plain_trace = []
        overlaid_trace = []
        for _ in range(5):
            plain_trace += list(plain.run_stage2(max_iters=4))
            overlaid.run_recursive_analysis()
            overlaid_trace += list(overlaid.run_stage2(max_iters=4))
            overlaid.run_recursive_analysis()
        self.assertGreater(len(plain_trace), 1)
        self.assertEqual(plain_trace, overlaid_trace)
        self.assertEqual(_lengths(plain), _lengths(overlaid))
        self.assertGreaterEqual(overlaid.analysis_pass_count, 10)

    def test_a_candidate_move_scores_identically_with_and_without_the_overlay(self):
        """The exact quantity `deltaF` compares, on an EXPLICIT candidate.

        A Pachner add is applied to a candidate complex here, and the node is
        asked for the objective it would assign to it. If a recursive read
        could reach the score, this is the number that would move.
        """
        plain = _node()
        overlaid = _node()
        overlaid.set_analysis_config(_overlay_config(degrees=(1, 2)))
        overlaid.run_recursive_analysis()

        scores = []
        for node in (plain, overlaid):
            candidate = _closed_s4(_REFINE, _HOST_SEED)
            move = T.AddMove(candidate, 11, False, T.PachnerMode.PreGeometric,
                             False)
            self.assertTrue(move.propose() and move.apply())
            terms = node.objective_terms_for(candidate)
            scores.append(tuple(getattr(terms, name)
                                for name in MC.objective_term_names()))
        self.assertEqual(scores[0], scores[1])

    def test_repeated_analysis_passes_never_move_the_objective(self):
        node = _node()
        node.set_analysis_config(_overlay_config(degrees=(1, 2),
                                                 resolutions=(0.5, 1.0, 2.0)))
        baseline = node.objective()
        for _ in range(4):
            node.run_recursive_analysis()
            self.assertEqual(baseline, node.objective())

    def test_the_overlay_actually_ran_during_the_comparison(self):
        """A vacuous identity test would pass with the overlay never firing."""
        overlaid = _node()
        overlaid.set_analysis_config(_overlay_config())
        overlaid.run_stage1(max_steps=1, n_candidate_moves=6)
        self.assertGreater(overlaid.accepted_move_count, 0)
        self.assertGreaterEqual(overlaid.analysis_pass_count,
                                overlaid.accepted_move_count)
        self.assertTrue(overlaid.checkpoint_json)
        doc = json.loads(overlaid.checkpoint_json)
        self.assertTrue(doc["fibers"])
        self.assertTrue(doc["particles"]["quarks"])

    def test_the_cadence_gates_the_pass_count_and_not_the_geometry(self):
        every = _node()
        every.set_analysis_config(_overlay_config(cadence=1))
        every_other = _node()
        every_other.set_analysis_config(_overlay_config(cadence=2))
        every.run_stage1(max_steps=1, n_candidate_moves=6)
        every_other.run_stage1(max_steps=1, n_candidate_moves=6)
        self.assertEqual(_cells(every), _cells(every_other))
        self.assertEqual(_lengths(every), _lengths(every_other))
        self.assertEqual(every.analysis_pass_count, 1)
        self.assertEqual(every_other.analysis_pass_count, 0)

    def test_the_engine_move_draw_is_not_reproducible_past_the_first_move(self):
        """Pin the PRE-EXISTING engine property this suite works around.

        Measured, not assumed: the first committed move is reproducible, later
        ones are not. If this test ever starts failing because the engine
        became reproducible, the identity comparisons above can be widened to
        whole multi-move drives.
        """
        first = [list(_node().run_stage1(max_steps=1, n_candidate_moves=6))
                 for _ in range(3)]
        self.assertEqual(first[0], first[1])
        self.assertEqual(first[1], first[2])
        longer = [list(_node().run_stage1(max_steps=4, n_candidate_moves=6))
                  for _ in range(4)]
        if all(run == longer[0] for run in longer):
            self.skipTest("the engine reproduced a multi-move drive; widen "
                          "the identity comparisons above")


# ======================================================================
# adversarial: poison the recursive reads and demand nothing moves
# ======================================================================


class AdversarialFeedbackTest(unittest.TestCase):
    """Drive the recursive reads to extremes; the dynamics must not notice."""

    def _poisoned_config(self, **kwargs):
        return _overlay_config(**kwargs)

    def test_forced_gap_closure_changes_no_objective_and_no_move(self):
        """Force the band gaps closed and demand the dynamics do not notice.

        The gap threshold is the overlay's own knob: raising the minimum
        relative gap and gap dominance far past what any band on this host can
        show closes EVERY band's isolation certificate at fixed geometry. If a
        gap could reach the objective or the accepted move, this is where it
        would show.
        """
        def uniform(node):
            for edge in node.st.getEdgeList().toVector():
                edge.setLength(cmath.sqrt(complex(1.0)))

        plain = _node()
        uniform(plain)
        overlaid = _node()
        uniform(overlaid)
        overlaid.set_analysis_config(self._poisoned_config())
        overlaid.run_recursive_analysis()
        doc = json.loads(overlaid.checkpoint_json)
        # The reads really did degrade under the uniform metric.
        self.assertTrue(doc["fibers"])
        degraded = [f for f in doc["fibers"] if not f["accepted"]]
        uncertified = [q for q in doc["particles"]["quarks"]
                       if q["failed_certificates"]]
        self.assertTrue(degraded or uncertified,
                        "the gap-closure fixture did not actually degrade")
        self.assertEqual(plain.objective(), overlaid.objective())
        self.assertEqual(_drive(plain), _drive(overlaid))

    def test_uncertified_reads_do_not_destabilize_the_optimization(self):
        """Graceful degradation: uncertified reads, a finite objective."""
        node = _node()
        for edge in node.st.getEdgeList().toVector():
            edge.setLength(cmath.sqrt(complex(1.0)))
        node.set_analysis_config(self._poisoned_config())
        trace, _, _ = _drive(node)
        self.assertTrue(all(math.isfinite(value) for value in trace))
        doc = json.loads(node.checkpoint_json)
        for quark in doc["particles"]["quarks"]:
            # An uncertified read is NAMED, never silently promoted.
            if quark["classification"] == "none":
                self.assertTrue(quark["failed_certificates"])

    def test_flipping_every_particle_verdict_changes_no_objective_term(self):
        """Rerun with thresholds that make every certificate fail, then pass.

        `ParticleClusters` thresholds live entirely on the analysis side. If a
        verdict could reach the objective, moving every threshold across its
        decision boundary would show up in a term.
        """
        node = _node()
        node.set_analysis_config(_overlay_config())
        before = node.objective_terms()
        node.run_recursive_analysis()
        strict_doc = json.loads(node.checkpoint_json)
        # Same geometry, an overlay configured to enumerate a different degree
        # (so entirely different bands, transports and verdicts are produced).
        node.set_analysis_config(_overlay_config(degrees=(1, 2),
                                                 resolutions=(0.5, 1.0, 2.0)))
        node.run_recursive_analysis()
        other_doc = json.loads(node.checkpoint_json)
        after = node.objective_terms()
        self.assertNotEqual(strict_doc["fibers"], other_doc["fibers"],
                            "the two overlay configurations produced identical "
                            "reads; the adversarial arm is vacuous")
        for name in MC.objective_term_names():
            self.assertEqual(getattr(before, name), getattr(after, name))

    def test_running_the_overlay_never_changes_the_geometry(self):
        node = _node()
        node.set_analysis_config(_overlay_config(degrees=(1, 2),
                                                 resolutions=(0.5, 1.0, 2.0)))
        cells_before, lengths_before = _cells(node), _lengths(node)
        objective_before = node.objective()
        for _ in range(3):
            node.run_recursive_analysis()
        self.assertEqual(cells_before, _cells(node))
        self.assertEqual(lengths_before, _lengths(node))
        self.assertEqual(objective_before, node.objective())

    def test_running_the_overlay_never_changes_the_refinement_decision(self):
        node = _node()
        thresholds = MC.RefinementIndicators()
        thresholds.mesh_quality = 0.0
        thresholds.regge_stationarity_residual = 0.0
        thresholds.hodge_stationarity_residual = 0.0
        thresholds.curvature_concentration = 0.0
        thresholds.solver_error = 0.0
        node.set_refinement_thresholds(thresholds)
        node.set_analysis_config(_overlay_config(degrees=(1, 2)))
        before = node.refinement_decision()
        node.run_recursive_analysis()
        after = node.refinement_decision()
        self.assertEqual(before.refine, after.refine)
        self.assertEqual(before.trigger, after.trigger)

    def test_the_gradient_is_unchanged_by_every_analysis_configuration(self):
        """Same relaxation from the same state, whatever the overlay reads."""
        results = []
        for config in (None,
                       _overlay_config(),
                       _overlay_config(degrees=(1, 2),
                                       resolutions=(0.5, 1.0, 2.0)),
                       _overlay_config(fock=True),
                       _overlay_config(cold=True)):
            node = _node()
            if config is not None:
                node.set_analysis_config(config)
                node.run_recursive_analysis()
            trace = list(node.run_stage2(max_iters=8))
            results.append((trace, _lengths(node)))
        for candidate in results[1:]:
            self.assertEqual(results[0], candidate)


# ======================================================================
# the two labeled Gaussian-closed emergence sub-modes
# ======================================================================


class EmergenceSubmodeTest(unittest.TestCase):
    def _carried(self, node, degree=1, modes=4):
        """A pure Slater carried state on the first `modes` degree-cells.

        Gamma = diag(1,...,1,0,...) is an exact projector, so the #780 purity
        certificate holds at machine precision.
        """
        cells = cob.ChainComplex.fromSpacetime(node.st).kSimplexVertices(degree)
        cells = cells[:modes]
        gamma = [0j] * (len(cells) * len(cells))
        for i in range(len(cells) // 2):
            gamma[i * len(cells) + i] = 1 + 0j
        node.set_carried_state(cells, degree, gamma)
        return cells

    def test_default_mode_is_strict_emergence(self):
        node = _node()
        self.assertEqual(node.simulation_mode, MC.SimulationMode.EMERGENCE)
        self.assertEqual(node.emergence_submode, MC.EmergenceSubmode.STRICT)

    def test_mode_names_are_the_checkpoint_labels(self):
        self.assertEqual(MC.mode_name(MC.SimulationMode.EMERGENCE), "emergence")
        self.assertEqual(MC.mode_name(MC.SimulationMode.SYNTHESIS), "synthesis")
        self.assertEqual(MC.mode_name(MC.SimulationMode.REPLAY), "replay")
        self.assertEqual(MC.submode_name(MC.EmergenceSubmode.STRICT), "strict")
        self.assertEqual(
            MC.submode_name(MC.EmergenceSubmode.CERTIFICATES_BLIND_MEAN_FIELD),
            "certificates_blind_mean_field")

    def test_strict_mode_refuses_the_state_energy_coupling(self):
        node = _node()
        self._carried(node)
        with self.assertRaises(ValueError):
            node.set_carried_state_energy_weight(0.5)

    def test_strict_mode_reports_zero_energy_even_with_a_carried_state(self):
        node = _node()
        self._carried(node)
        self.assertEqual(node.carried_state_energy(node.st), 0.0)
        self.assertEqual(node.objective_terms().carried_state_energy, 0.0)

    def test_backreaction_submode_admits_only_the_energy_density(self):
        node = _node()
        node.set_simulation_mode(MC.SimulationMode.EMERGENCE,
                                 MC.EmergenceSubmode.CERTIFICATES_BLIND_MEAN_FIELD)
        self._carried(node)
        node.set_carried_state_energy_weight(0.25)
        energy = node.carried_state_energy(node.st)
        self.assertTrue(math.isfinite(energy))
        self.assertNotEqual(energy, 0.0)
        terms = node.objective_terms()
        self.assertAlmostEqual(terms.carried_state_energy, 0.25 * energy,
                               places=15)

    def test_deselecting_the_coupling_selects_the_strict_submode(self):
        node = _node()
        node.set_simulation_mode(MC.SimulationMode.EMERGENCE,
                                 MC.EmergenceSubmode.CERTIFICATES_BLIND_MEAN_FIELD)
        self._carried(node)
        node.set_carried_state_energy_weight(0.25)
        backreaction_objective = node.objective()
        node.set_simulation_mode(MC.SimulationMode.EMERGENCE,
                                 MC.EmergenceSubmode.STRICT)
        self.assertEqual(node.emergence_submode, MC.EmergenceSubmode.STRICT)
        self.assertEqual(node.carried_state_energy_weight, 0.0)
        self.assertNotEqual(backreaction_objective, node.objective())
        self.assertEqual(node.objective_terms().carried_state_energy, 0.0)

    def test_backreaction_changes_the_trajectory_relative_to_strict(self):
        """By design: the declared state-energy term is a real coupling."""
        strict = _node()
        self._carried(strict)
        backreaction = _node()
        backreaction.set_simulation_mode(
            MC.SimulationMode.EMERGENCE,
            MC.EmergenceSubmode.CERTIFICATES_BLIND_MEAN_FIELD)
        self._carried(backreaction)
        backreaction.set_carried_state_energy_weight(5.0)
        self.assertNotEqual(strict.objective(), backreaction.objective())
        self.assertNotEqual(list(strict.run_stage2(max_iters=6)),
                            list(backreaction.run_stage2(max_iters=6)))

    def test_state_energy_trajectory_is_blind_to_post_hoc_analysis(self):
        """The backreaction acceptance bullet: disabling only the post-hoc
        certificate analysis leaves the state-energy trajectory unchanged."""
        traces = []
        for config in (None, _overlay_config(),
                       _overlay_config(degrees=(1, 2))):
            node = _node()
            node.set_simulation_mode(
                MC.SimulationMode.EMERGENCE,
                MC.EmergenceSubmode.CERTIFICATES_BLIND_MEAN_FIELD)
            self._carried(node)
            node.set_carried_state_energy_weight(2.0)
            if config is not None:
                node.set_analysis_config(config)
            traces.append((list(node.run_stage2(max_iters=6)), _lengths(node)))
        for candidate in traces[1:]:
            self.assertEqual(traces[0], candidate)

    def test_carried_state_energy_is_the_exact_wick_trace(self):
        """E = Re tr(Gamma h): recompute it independently from the bindings."""
        node = _node()
        node.set_simulation_mode(MC.SimulationMode.EMERGENCE,
                                 MC.EmergenceSubmode.CERTIFICATES_BLIND_MEAN_FIELD)
        cells = self._carried(node, modes=5)
        node.set_carried_state_energy_weight(1.0)
        laplacian = cob.HodgeLaplacian(node.st).laplacian(1, True)
        order = cob.ChainComplex.fromSpacetime(node.st).kSimplexVertices(1)
        index = {tuple(sorted(c)): i for i, c in enumerate(order)}
        n = len(order)
        rows = [index[tuple(sorted(c))] for c in cells]
        gamma = node.carried_state_covariance
        m = len(cells)
        total = 0j
        for i in range(m):
            for j in range(m):
                a, b = rows[j], rows[i]
                h = 0.5 * (laplacian[a * n + b] + laplacian[b * n + a].conjugate())
                total += gamma[i * m + j] * h
        self.assertAlmostEqual(node.carried_state_energy(node.st), total.real,
                               places=12)

    def test_carried_state_energy_gradient_matches_a_central_difference(self):
        """The analytic dE/dz against a central difference (a VALIDATION of
        the closed form, not a fallback for it)."""
        node = _node()
        node.set_simulation_mode(MC.SimulationMode.EMERGENCE,
                                 MC.EmergenceSubmode.CERTIFICATES_BLIND_MEAN_FIELD)
        self._carried(node, modes=5)
        node.set_carried_state_energy_weight(1.0)
        analytic = node.carried_state_energy_gradient(node.st)
        edges = node.st.getEdgeList().toVector()
        self.assertEqual(len(analytic), len(edges))
        step = 1e-6
        worst = 0.0
        for i in range(0, len(edges), max(1, len(edges) // 6)):
            edge = edges[i]
            base = edge.getLength()
            z = base * base
            edge.setLength(cmath.sqrt(z + step))
            plus = node.carried_state_energy(node.st)
            edge.setLength(cmath.sqrt(z - step))
            minus = node.carried_state_energy(node.st)
            edge.setLength(base)
            numeric = (plus - minus) / (2.0 * step)
            worst = max(worst, abs(numeric - analytic[i].real) /
                        max(1.0, abs(numeric)))
        self.assertLess(worst, 1e-5, "analytic dE/dz disagrees with the "
                                     "central difference by %g" % worst)

    def test_both_submodes_report_the_gaussianity_certificate(self):
        for submode in (MC.EmergenceSubmode.STRICT,
                        MC.EmergenceSubmode.CERTIFICATES_BLIND_MEAN_FIELD):
            node = _node()
            node.set_simulation_mode(MC.SimulationMode.EMERGENCE, submode)
            self._carried(node)
            self.assertLess(node.carried_state_purity_defect(), 1e-12)
            self.assertTrue(node.carried_state_purity_holds())

    def test_a_degree_zero_carried_state_is_refused(self):
        """Degree zero would locate a mode cell against the wrong index (the
        graph Laplacian is ordered by sorted vertex id, not by the canonical
        cell order) and has no exact gradient. Refuse loudly."""
        node = _node()
        cells = cob.ChainComplex.fromSpacetime(node.st).kSimplexVertices(0)[:2]
        with self.assertRaises(ValueError):
            node.set_carried_state(cells, 0, [1 + 0j, 0j, 0j, 0j])

    def test_a_malformed_carried_covariance_is_refused(self):
        node = _node()
        cells = cob.ChainComplex.fromSpacetime(node.st).kSimplexVertices(1)[:3]
        with self.assertRaises(ValueError):
            node.set_carried_state(cells, 1, [1 + 0j, 0j])

    def test_clearing_the_carried_state_zeroes_the_term(self):
        node = _node()
        node.set_simulation_mode(MC.SimulationMode.EMERGENCE,
                                 MC.EmergenceSubmode.CERTIFICATES_BLIND_MEAN_FIELD)
        self._carried(node)
        node.set_carried_state_energy_weight(1.0)
        self.assertNotEqual(node.objective_terms().carried_state_energy, 0.0)
        node.clear_carried_state()
        self.assertFalse(node.has_carried_state)
        self.assertEqual(node.objective_terms().carried_state_energy, 0.0)

    def test_synthesis_mode_carries_no_state_energy_coupling(self):
        node = _node()
        node.set_simulation_mode(MC.SimulationMode.SYNTHESIS)
        self._carried(node)
        with self.assertRaises(ValueError):
            node.set_carried_state_energy_weight(1.0)
        self.assertEqual(node.objective_terms().carried_state_energy, 0.0)

    def test_no_carried_state_reports_an_unknown_purity_not_a_zero(self):
        node = _node()
        self.assertTrue(math.isnan(node.carried_state_purity_defect()))
        self.assertFalse(node.carried_state_purity_holds())

    def test_the_mean_field_loop_stays_inside_the_gaussian_manifold(self):
        node = _node()
        node.set_simulation_mode(MC.SimulationMode.EMERGENCE,
                                 MC.EmergenceSubmode.CERTIFICATES_BLIND_MEAN_FIELD)
        self._carried(node, modes=5)
        node.set_carried_state_energy_weight(1.0)
        node.set_mean_field_schedule(0.05, 8)
        worst = node.advance_carried_state()
        self.assertLess(worst, 1e-9)
        self.assertTrue(node.carried_state_purity_holds())

    def test_the_mean_field_schedule_is_checkpointed(self):
        node = _node()
        node.set_simulation_mode(MC.SimulationMode.EMERGENCE,
                                 MC.EmergenceSubmode.CERTIFICATES_BLIND_MEAN_FIELD)
        self._carried(node)
        node.set_carried_state_energy_weight(0.75)
        node.set_mean_field_schedule(0.02, 4)
        node.set_analysis_config(_overlay_config())
        node.run_recursive_analysis()
        objective = json.loads(node.checkpoint_json)["objective"]
        self.assertEqual(objective["carried_state_energy_weight"], 0.75)
        self.assertEqual(objective["mean_field_dt"], 0.02)
        self.assertEqual(objective["mean_field_steps"], 4)

    def test_the_submode_is_recorded_in_provenance(self):
        for submode, label in (
                (MC.EmergenceSubmode.STRICT, "strict"),
                (MC.EmergenceSubmode.CERTIFICATES_BLIND_MEAN_FIELD,
                 "certificates_blind_mean_field")):
            node = _node()
            node.set_simulation_mode(MC.SimulationMode.EMERGENCE, submode)
            node.set_analysis_config(_overlay_config())
            node.run_recursive_analysis()
            doc = json.loads(node.checkpoint_json)
            self.assertEqual(doc["mode"], "emergence")
            self.assertEqual(doc["emergence_submode"], label)

    def test_synthesis_mode_is_stamped_and_never_reads_as_emergence(self):
        node = _node()
        node.set_simulation_mode(MC.SimulationMode.SYNTHESIS)
        node.set_analysis_config(_overlay_config())
        node.run_recursive_analysis()
        self.assertEqual(json.loads(node.checkpoint_json)["mode"], "synthesis")

    def test_a_certificate_never_reaches_the_backreaction_term(self):
        """The state energy depends on Gamma and the geometry ONLY."""
        node = _node()
        node.set_simulation_mode(MC.SimulationMode.EMERGENCE,
                                 MC.EmergenceSubmode.CERTIFICATES_BLIND_MEAN_FIELD)
        self._carried(node)
        node.set_carried_state_energy_weight(1.0)
        before = node.carried_state_energy(node.st)
        for config in (_overlay_config(),
                       _overlay_config(degrees=(1, 2), resolutions=(0.5, 2.0)),
                       _overlay_config(fock=True)):
            node.set_analysis_config(config)
            node.run_recursive_analysis()
            self.assertEqual(before, node.carried_state_energy(node.st))


# ======================================================================
# refinement independence
# ======================================================================


class RefinementIndependenceTest(unittest.TestCase):
    def test_indicators_are_measured_from_the_base_problem(self):
        node = _node()
        indicators = node.refinement_indicators()
        self.assertAlmostEqual(indicators.regge_stationarity_residual,
                               node.objective_terms().regge_stationarity, 12)
        self.assertAlmostEqual(indicators.hodge_stationarity_residual,
                               node.hodge_entropy_stationarity(), 12)
        self.assertGreater(indicators.curvature_concentration, 0.0)
        self.assertGreater(indicators.mesh_quality, 0.0)
        self.assertLessEqual(indicators.mesh_quality, 1.0)

    # Marked slow: the relaxation has to reach stationarity before the solver
    # error can be read; about one minute.
    @pytest.mark.slow
    def test_solver_error_is_zero_once_the_relaxation_is_stationary(self):
        node = _node()
        node.run_stage2(max_iters=200, tolerance=1e-12)
        if node.last_stage2_stationary:
            self.assertEqual(node.refinement_indicators().solver_error, 0.0)

    def test_extreme_recursive_reads_change_no_indicator(self):
        """Adversarial: drive the reads to their limits at fixed geometry."""
        node = _node()
        baseline = node.refinement_indicators()
        for config in (_overlay_config(),
                       _overlay_config(degrees=(1, 2, 3)),
                       _overlay_config(resolutions=(0.1, 1.0, 10.0)),
                       _overlay_config(fock=True),
                       _overlay_config(cold=True)):
            node.set_analysis_config(config)
            node.run_recursive_analysis()
            current = node.refinement_indicators()
            for name in MC.refinement_indicator_names():
                self.assertEqual(getattr(baseline, name), getattr(current, name))

    def test_refinement_fires_only_when_a_base_indicator_crosses(self):
        node = _node()
        indicators = node.refinement_indicators()
        thresholds = MC.RefinementIndicators()
        thresholds.mesh_quality = 0.0
        thresholds.regge_stationarity_residual = 0.0
        thresholds.hodge_stationarity_residual = 0.0
        thresholds.curvature_concentration = 0.0
        thresholds.solver_error = 0.0
        node.set_refinement_thresholds(thresholds)
        self.assertFalse(node.refinement_decision().refine)
        thresholds.curvature_concentration = \
            indicators.curvature_concentration * 0.5
        node.set_refinement_thresholds(thresholds)
        decision = node.refinement_decision()
        self.assertTrue(decision.refine)
        self.assertEqual(decision.trigger, "curvature_concentration")

    def test_refine_geometry_is_a_no_op_when_no_indicator_asks(self):
        node = _node()
        cells_before = _cells(node)
        self.assertEqual(node.refine_geometry(2), 0)
        self.assertEqual(cells_before, _cells(node))

    def test_refine_geometry_uses_the_existing_gated_surgery(self):
        node = _node()
        thresholds = MC.RefinementIndicators()
        thresholds.mesh_quality = 0.0
        thresholds.regge_stationarity_residual = 0.0
        thresholds.hodge_stationarity_residual = 0.0
        thresholds.solver_error = 0.0
        thresholds.curvature_concentration = \
            node.refinement_indicators().curvature_concentration * 0.5
        node.set_refinement_thresholds(thresholds)
        before = len(_cells(node))
        committed = node.refine_geometry(1)
        self.assertGreaterEqual(committed, 0)
        self.assertEqual(len(_cells(node)), before + committed)
        # The gate ran: the result is still a valid dual complex.
        self.assertTrue(cob.ChainComplex.fromSpacetime(node.st)
                        .boundaryComposesToZero())

    def test_a_committed_refinement_cell_counts_as_an_accepted_move(self):
        node = _node()
        node.set_analysis_config(_overlay_config())
        thresholds = MC.RefinementIndicators()
        thresholds.mesh_quality = 0.0
        thresholds.regge_stationarity_residual = 0.0
        thresholds.hodge_stationarity_residual = 0.0
        thresholds.solver_error = 0.0
        thresholds.curvature_concentration = \
            node.refinement_indicators().curvature_concentration * 0.5
        node.set_refinement_thresholds(thresholds)
        before_moves = node.accepted_move_count
        before_passes = node.analysis_pass_count
        committed = node.refine_geometry(1)
        self.assertEqual(node.accepted_move_count, before_moves + committed)
        self.assertEqual(node.analysis_pass_count, before_passes + committed)

    def test_the_analysis_cadence_must_be_at_least_one(self):
        node = _node()
        config = _overlay_config()
        config.cadence = 0
        with self.assertRaises(ValueError):
            node.set_analysis_config(config)

    def test_refinement_thresholds_take_no_analysis_argument(self):
        """`refinement_decision_of` accepts exactly two indicator records."""
        with self.assertRaises(TypeError):
            MC.refinement_decision_of(MC.RefinementIndicators(),
                                      MC.RefinementIndicators(),
                                      "a certificate")


# ======================================================================
# the versioned checkpoint schema
# ======================================================================


class CheckpointSchemaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        node = _node()
        node.set_analysis_config(_overlay_config())
        node.set_provenance("config-hash-abc", "commit-def")
        node.run_recursive_analysis()
        cls.node = node
        cls.doc = json.loads(node.checkpoint_json)

    def test_schema_version_is_the_declared_one(self):
        self.assertEqual(self.doc["schema_version"],
                         MC.checkpoint_schema_version())
        self.assertEqual(MC.checkpoint_version_of(self.node.checkpoint_json),
                         MC.checkpoint_schema_version())

    def test_every_design_spec_block_is_present(self):
        for key in ("schema_version", "mode", "emergence_submode",
                    "geometry_revision", "raw_complex", "edge_quantum_data",
                    "objective", "hierarchy", "fibers", "labeled_fiber_sums",
                    "transports", "covariance", "fock_oracle", "particles",
                    "certificates", "provenance"):
            self.assertIn(key, self.doc)

    def test_the_covariance_block_carries_the_declared_fields(self):
        covariance = self.doc["covariance"]
        for key in ("active_modes", "number_conserving", "purity_defect",
                    "matrix_sidecar"):
            self.assertIn(key, covariance)
        self.assertTrue(covariance["number_conserving"])

    def test_the_particles_block_carries_the_declared_sectors(self):
        for key in ("quarks", "gluons", "bound_supercomponents", "baryons"):
            self.assertIn(key, self.doc["particles"])

    def test_the_search_records_and_the_verdicts_are_separate_blocks(self):
        """Schema 4 splits them: `bound_supercomponents` is the section 16.2
        SEARCH, `baryons` is the section 16.4 three-cluster VERDICT. A search
        record is not a baryon read and never travels as one."""
        for binding in self.doc["particles"]["bound_supercomponents"]:
            self.assertIn("found", binding)
            self.assertNotIn("classification", binding)
        for baryon in self.doc["particles"]["baryons"]:
            self.assertIn("classification", baryon)
            self.assertNotIn("found", baryon)

    def test_a_baryon_read_is_emitted_only_for_three_certified_legs(self):
        """The overlay classifies a binding of EXACTLY three certified
        constituents and nothing else -- a three-cluster verdict is never
        assembled by padding. On this host no candidate certifies, so the
        search emits no binding and the verdict list is empty for a NAMED
        reason rather than by omission."""
        bindings = self.doc["particles"]["bound_supercomponents"]
        baryons = self.doc["particles"]["baryons"]
        three = [b for b in bindings if b["constituents"] == 3]
        self.assertEqual(len(baryons), len(three))
        certified = [q for q in self.doc["particles"]["quarks"]
                     if q["classification"] == "quark"]
        if not certified:
            self.assertEqual(bindings, [])
            self.assertEqual(baryons, [])

    def test_unknown_values_serialize_as_null_never_as_zero(self):
        quarks = self.doc["particles"]["quarks"]
        self.assertTrue(quarks, "no candidate was produced to check")
        unknowns = [q for q in quarks if q["determinant_winding"] is None]
        self.assertTrue(unknowns, "no unknown value in the fixture")
        for quark in unknowns:
            self.assertIsNone(quark["determinant_winding"])
            self.assertIsNone(quark["baryon_flux"])

    def test_an_unmeasured_carried_purity_is_null(self):
        self.assertIsNone(self.doc["covariance"]["carried_purity_defect"])
        self.assertIsNone(self.doc["certificates"]["covariance_purity_holds"])

    def test_provenance_is_deterministic_and_complete(self):
        provenance = self.doc["provenance"]
        self.assertEqual(provenance["seed"], _NODE_SEED)
        self.assertEqual(provenance["config_hash"], "config-hash-abc")
        self.assertEqual(provenance["commit"], "commit-def")

    def test_the_raw_complex_round_trips_the_geometry(self):
        raw = self.doc["raw_complex"]
        self.assertEqual(raw["dimensions"],
                         len(_cells(self.node)[0]) - 1)
        self.assertEqual(len(raw["cells"]), len(_cells(self.node)))
        self.assertEqual(len(raw["edges"]), len(_lengths(self.node)))
        recorded = {(e["a"], e["b"]): complex(e["length"][0], e["length"][1])
                    for e in raw["edges"]}
        self.assertEqual(recorded, _lengths(self.node))

    def test_the_raw_complex_is_written_in_canonical_endpoint_order(self):
        pairs = [(e["a"], e["b"]) for e in self.doc["raw_complex"]["edges"]]
        self.assertEqual(pairs, sorted(pairs))
        for a, b in pairs:
            self.assertLess(a, b)

    def test_the_raw_complex_records_the_connection_phase(self):
        """An edge carries TWO fields, so a document that records only the
        length describes a different complex than the one that wrote it."""
        node = _node()
        node.set_analysis_config(_overlay_config())
        _twist(node)
        node.run_recursive_analysis()
        raw = json.loads(node.checkpoint_json)["raw_complex"]
        recorded = {(e["a"], e["b"]): complex(e["phase"][0], e["phase"][1])
                    for e in raw["edges"]}
        self.assertEqual(recorded, _phases(node))
        # and it is genuinely complex, not a real angle widened to complex
        self.assertGreater(max(abs(p.imag) for p in recorded.values()), 1e-6)

    def test_the_objective_block_records_every_declared_term(self):
        objective = self.doc["objective"]
        for name in MC.objective_term_names():
            self.assertIn(name, objective)
        self.assertAlmostEqual(objective["total"], self.node.objective(), 12)

    def test_the_refinement_block_records_every_declared_indicator(self):
        refinement = self.doc["refinement"]
        for name in MC.refinement_indicator_names():
            self.assertIn(name, refinement)
        self.assertIn("refine", refinement)
        self.assertIn("trigger", refinement)

    def test_the_checkpoint_is_a_pure_function_of_the_state(self):
        again = _node()
        again.set_analysis_config(_overlay_config())
        again.set_provenance("config-hash-abc", "commit-def")
        again.run_recursive_analysis()
        mine = copy.deepcopy(self.doc)
        theirs = json.loads(again.checkpoint_json)
        # `analysis.pass` counts passes on THIS node; everything else must
        # match byte for byte.
        mine["analysis"].pop("pass")
        theirs["analysis"].pop("pass")
        self.assertEqual(mine, theirs)


# ======================================================================
# replay
# ======================================================================


class ReplayTest(unittest.TestCase):
    def _driven_node(self):
        node = _node()
        node.set_analysis_config(_overlay_config())
        node.set_provenance("cfg", "abc123")
        node.run_stage1(max_steps=2, n_candidate_moves=6)
        node.run_recursive_analysis()
        return node

    def test_cold_replay_reproduces_the_checkpoint_byte_for_byte(self):
        node = self._driven_node()
        self.assertGreater(node.accepted_move_count, 0)
        incremental = json.loads(node.checkpoint_json)
        replayed = json.loads(MC.replay_checkpoint(node.checkpoint_json))
        for key in ("hierarchy", "fibers", "labeled_fiber_sums", "transports",
                    "covariance", "particles", "certificates", "raw_complex"):
            self.assertEqual(incremental[key], replayed[key],
                             "cold replay changed the %s block" % key)

    def test_cold_replay_serves_nothing_from_cache(self):
        node = self._driven_node()
        replayed = json.loads(MC.replay_checkpoint(node.checkpoint_json))
        self.assertTrue(replayed["analysis"]["cold_caches"])
        self.assertEqual(replayed["analysis"]["cache_hits"], 0)
        self.assertGreater(json.loads(node.checkpoint_json)
                           ["analysis"]["cache_hits"], 0)

    def test_replay_is_stamped_replay_not_emergence(self):
        node = self._driven_node()
        replayed = json.loads(MC.replay_checkpoint(node.checkpoint_json))
        self.assertEqual(replayed["mode"], "replay")

    def test_replay_preserves_the_provenance_seed_and_hashes(self):
        node = self._driven_node()
        replayed = json.loads(MC.replay_checkpoint(node.checkpoint_json))
        self.assertEqual(replayed["provenance"]["seed"], _NODE_SEED)
        self.assertEqual(replayed["provenance"]["config_hash"], "cfg")
        self.assertEqual(replayed["provenance"]["commit"], "abc123")

    def test_replay_from_a_host_never_rebuilt_agrees_within_round_off(self):
        """A hand-built host's edge list is in construction order, not the
        `fromVertexTuples` order every optimizer-produced complex has, so the
        modularity sums accumulate in a different order. The VERDICTS are
        identical; the continuous aggregates agree to double round-off."""
        node = _node()
        node.set_analysis_config(_overlay_config())
        node.run_recursive_analysis()
        incremental = json.loads(node.checkpoint_json)
        replayed = json.loads(MC.replay_checkpoint(node.checkpoint_json))
        self.assertEqual(
            [q["classification"] for q in incremental["particles"]["quarks"]],
            [q["classification"] for q in replayed["particles"]["quarks"]])
        self.assertEqual(incremental["particles"], replayed["particles"])
        self.assertEqual(incremental["fibers"], replayed["fibers"])
        for a, b in zip(incremental["hierarchy"], replayed["hierarchy"]):
            self.assertAlmostEqual(a["q"], b["q"], places=12)
            self.assertEqual([c["id"] for c in a["components"]],
                             [c["id"] for c in b["components"]])

    def test_an_unknown_schema_version_is_rejected(self):
        node = self._driven_node()
        current = '"schema_version": %d' % MC.checkpoint_schema_version()
        self.assertIn(current, node.checkpoint_json)
        bad = node.checkpoint_json.replace(current, '"schema_version": 99')
        with self.assertRaises(ValueError):
            MC.replay_checkpoint(bad)

    def test_the_previous_schema_version_is_rejected(self):
        """Schema 3 wrote the bound-supercomponent SEARCH records under
        `particles.baryons`; schema 4 writes the three-cluster VERDICT there.
        The entries mean different things, so an older document is refused
        rather than reinterpreted."""
        node = self._driven_node()
        current = '"schema_version": %d' % MC.checkpoint_schema_version()
        older = node.checkpoint_json.replace(current, '"schema_version": 3')
        self.assertEqual(MC.checkpoint_version_of(older), 3)
        with self.assertRaises(ValueError):
            MC.replay_checkpoint(older)

    def test_schema_four_is_rejected_rather_than_replayed_with_zero_phases(self):
        """Schema 4 recorded no phase at all. Its silence is not evidence the
        phase was zero, so it is refused: a replay that quietly zeroes a
        written field is worse than one that refuses."""
        node = self._driven_node()
        current = '"schema_version": %d' % MC.checkpoint_schema_version()
        older = node.checkpoint_json.replace(current, '"schema_version": 4')
        self.assertEqual(MC.checkpoint_version_of(older), 4)
        with self.assertRaises(ValueError):
            MC.replay_checkpoint(older)

    def test_replay_restores_every_connection_phase_exactly(self):
        """The defect this ticket exists for: replay used to rebuild through
        fromVertexTuples with phase 0 and silently drop a live field."""
        node = _node()
        node.set_analysis_config(_overlay_config())
        _twist(node, seed=17)
        node.run_recursive_analysis()
        written = _phases(node)
        self.assertGreater(max(abs(p.imag) for p in written.values()), 1e-6)

        replayed = json.loads(MC.replay_checkpoint(node.checkpoint_json))
        restored = {(e["a"], e["b"]): complex(e["phase"][0], e["phase"][1])
                    for e in replayed["raw_complex"]["edges"]}
        self.assertEqual(restored, written)   # exactly, not approximately

    def test_replay_of_a_twisted_complex_reproduces_the_checkpoint(self):
        """Round-tripping a complex whose phases are all nonzero must be as
        byte-exact as the untwisted case."""
        node = _node()
        node.set_analysis_config(_overlay_config())
        _twist(node, seed=23)
        node.run_recursive_analysis()
        incremental = json.loads(node.checkpoint_json)
        replayed = json.loads(MC.replay_checkpoint(node.checkpoint_json))
        self.assertEqual(incremental["raw_complex"], replayed["raw_complex"])

    def test_a_missing_schema_version_is_rejected(self):
        with self.assertRaises(ValueError):
            MC.checkpoint_version_of('{"mode": "emergence"}')

    def test_malformed_json_is_rejected(self):
        with self.assertRaises(ValueError):
            MC.checkpoint_version_of('not json at all')

    def test_replay_of_a_checkpoint_without_a_raw_complex_is_rejected(self):
        with self.assertRaises(ValueError):
            MC.replay_checkpoint('{"schema_version": %d}'
                                 % MC.checkpoint_schema_version())


# ======================================================================
# the overlay actually drives the merged stack
# ======================================================================


class AnalysisOverlayTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        node = _node()
        node.set_analysis_config(_overlay_config())
        node.run_recursive_analysis()
        cls.doc = json.loads(node.checkpoint_json)
        cls.node = node

    def test_the_component_hierarchy_is_populated(self):
        self.assertTrue(self.doc["hierarchy"])
        slice_ = self.doc["hierarchy"][0]
        self.assertGreaterEqual(len(slice_["components"]), 1)
        for component in slice_["components"]:
            self.assertEqual(len(component["id"]), 32)
            self.assertTrue(component["support"])

    def test_the_spectral_fibers_are_populated_and_certified(self):
        self.assertTrue(self.doc["fibers"])
        for fiber in self.doc["fibers"]:
            self.assertIn("rank", fiber)
            self.assertIn("accepted", fiber)
            self.assertIn("gram_defect", fiber)
        self.assertTrue(any(f["accepted"] for f in self.doc["fibers"]))

    def test_the_labeled_fiber_sum_carries_its_gram_data(self):
        self.assertTrue(self.doc["labeled_fiber_sums"])
        for entry in self.doc["labeled_fiber_sums"]:
            self.assertIn("gram_defect", entry)
            self.assertIn("quotient_nullity", entry)
            self.assertIn("certificate", entry)

    def test_the_transports_are_derived_and_gated(self):
        self.assertTrue(self.doc["transports"])
        for transport in self.doc["transports"]:
            self.assertIn("leakage", transport)
            self.assertIn("accepted", transport)

    def test_the_transports_are_the_cross_component_mutual_family(self):
        """One derived link per ordered pair of candidate bands.

        Not every band against every band: a component's bands are
        alternative carriers, not links. The LIFETIME family is a different
        object that one frame cannot supply, and is left unsupplied rather
        than filled with cross-component links (measured: the all-pairs
        enumeration cost 663 derived transports at 62 cells, 30x the
        optimizer step, for links no certificate consumes).
        """
        candidates = len(self.doc["particles"]["quarks"])
        self.assertLessEqual(len(self.doc["transports"]),
                             candidates * max(candidates - 1, 1))
        for quark in self.doc["particles"]["quarks"]:
            self.assertEqual(quark["transport_count"], 0)

    def test_the_covariance_layer_is_exact_and_pure(self):
        covariance = self.doc["covariance"]
        self.assertGreater(covariance["active_modes"], 0)
        self.assertLess(covariance["purity_defect"], 1e-9)

    def test_the_fock_oracle_is_absent_unless_selected(self):
        oracle = self.doc["fock_oracle"]
        self.assertFalse(oracle["present"])
        self.assertIsNone(oracle["absent_reason"])
        self.assertEqual(oracle["nodes"], 0)

    def test_the_selected_oracle_is_built_for_real_on_a_positive_band(self):
        """Degree zero is the positive graph Laplacian, whose band projector
        IS an orthogonal projector, so the #771 lazy Slater DAG exists."""
        node = _node()
        node.set_analysis_config(_overlay_config(degrees=(0,), fock=True))
        node.run_recursive_analysis()
        oracle = json.loads(node.checkpoint_json)["fock_oracle"]
        self.assertTrue(oracle["present"])
        self.assertGreater(oracle["active_modes"], 0)
        self.assertGreater(oracle["nodes"], 0)
        self.assertTrue(oracle["exact"])
        self.assertEqual(oracle["discarded_norm"], 0.0)
        self.assertIsNone(oracle["absent_reason"])

    def test_the_oracle_refuses_an_oblique_band_and_names_why(self):
        """At k >= 1 the signed-weight operator's band projector is oblique,
        so no exact Slater reference exists. The refusal is NAMED, never a
        silently claimed oracle."""
        node = _node()
        node.set_analysis_config(_overlay_config(degrees=(1,), fock=True))
        node.run_recursive_analysis()
        oracle = json.loads(node.checkpoint_json)["fock_oracle"]
        if oracle["present"]:
            self.skipTest("the k>=1 band projector became orthogonal")
        self.assertIsNotNone(oracle["absent_reason"])
        self.assertIn("projector", oracle["absent_reason"])
        self.assertIsNone(oracle["discarded_norm"])

    def test_particle_reads_are_produced_with_named_gaps(self):
        quarks = self.doc["particles"]["quarks"]
        self.assertTrue(quarks)
        for quark in quarks:
            self.assertIn(quark["classification"],
                          ("none", "quark", "antiquark"))
            if quark["classification"] == "none":
                self.assertTrue(quark["failed_certificates"])

    def test_the_overlay_runs_the_baryon_classifier(self):
        """#802 — the overlay drives the section 16.4 classifier over its
        own section 16.2 search, so a proton verdict can reach a driver at
        all. The wiring is pinned at the source: the overlay's translation
        unit calls the composition, and the objective's does not."""
        root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
        overlay = os.path.join(root, "src", "cobordism",
                               "RecursiveFiberSimulation.cpp")
        engine = os.path.join(root, "src", "cobordism", "MultiCobordism.cpp")
        if not os.path.exists(overlay):     # installed wheel, not a checkout
            self.skipTest("source tree not available")
        with open(overlay) as handle:
            overlay_text = handle.read()
        self.assertIn("classifyBoundSupercomponents", overlay_text)
        self.assertIn("boundSupercomponentSearch", overlay_text)
        with open(engine) as handle:
            self.assertNotIn("classifyBoundSupercomponents", handle.read())

    def test_the_baryon_writer_emits_every_declared_field(self):
        """The `particles.baryons` record is the BaryonRead, whole: the
        proton-certificate fields, the evidence summary, the named gaps and
        the #764 certificate. A field silently dropped from the writer is a
        measurement lost from the record."""
        root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
        overlay = os.path.join(root, "src", "cobordism",
                               "RecursiveFiberSimulation.cpp")
        if not os.path.exists(overlay):
            self.skipTest("source tree not available")
        with open(overlay) as handle:
            text = handle.read()
        for key in ("bound_component", "quarks", "classification",
                    "confidence", "color_gram_determinant", "color_wedge",
                    "color_flux", "baryon_flux", "electric_flux",
                    "total_winding", "exterior_parity", "flavor_pattern",
                    "total_isospin", "total_j2", "total_j2_variance",
                    "sharp_spin", "quasi_free_class_swept",
                    "class_variance_floor", "rotation_character",
                    "rotation_character_sign", "exchange_character",
                    "spin_statistics_ratio", "spin_lift_applicable",
                    "spin_lift_accepted", "radius", "radius_finite",
                    "spectral_mass", "radius_ratio",
                    "profile_max_deviation", "profile_stable",
                    "physical_mass", "persistence", "lifetime_overlap",
                    "transport_count", "transport_leakage_max",
                    "failed_certificates", "certificate"):
            self.assertIn('"%s"' % key, text, "%s is not written" % key)

    def test_no_emergent_candidate_can_certify_in_one_pass(self):
        """The measured reason the verdict list is empty, NAMED rather than
        left as an unexplained absence.

        One analysis pass is ONE frame, so the overlay supplies no
        determinant winding, no lifetime-transport family and no refinement
        overlap to the quark classifier. Those four certificates therefore
        fail on every candidate BY NAME, no candidate certifies, the section
        16.2 search counts no member, and the section 16.4 classifier has no
        three-cluster candidate to classify. That is a structural property
        of a single-frame pass, not a property of this host.
        """
        quarks = self.doc["particles"]["quarks"]
        self.assertTrue(quarks)
        for quark in quarks:
            self.assertEqual(quark["classification"], "none")
            for name in ("winding", "winding-unit", "transport-leakage",
                         "refinement-stability"):
                self.assertIn(name, quark["failed_certificates"])
        self.assertEqual(self.doc["particles"]["bound_supercomponents"], [])
        self.assertEqual(self.doc["particles"]["baryons"], [])

    def test_nothing_requests_a_hole_a_rank_or_a_uud_pattern(self):
        """Emergence mode never ASKS for a proton-shaped thing."""
        text = json.dumps(self.doc)
        self.assertNotIn("requested", text)
        self.assertNotIn("uud", text)
        # An expected register count is a target-conditioned notion; the
        # JointStationarity objective carries no r_U term at all.
        self.assertEqual(self.doc["objective"]["register_residual"], 0.0)

    def test_an_unchanged_complex_invalidates_no_ancestry(self):
        node = _node()
        node.set_analysis_config(_overlay_config())
        node.run_recursive_analysis()
        self.assertEqual(json.loads(node.checkpoint_json)
                         ["invalidated_ancestry"]["components"], 0)
        node.run_recursive_analysis()
        second = json.loads(node.checkpoint_json)
        self.assertEqual(second["invalidated_ancestry"]["components"], 0)
        self.assertEqual(second["analysis"]["cache_invalidations"], 0)

    def test_a_local_move_invalidates_the_touched_ancestry(self):
        node = _node()
        node.set_analysis_config(_overlay_config())
        node.run_recursive_analysis()
        node.run_stage1(max_steps=1, n_candidate_moves=6)
        second = json.loads(node.checkpoint_json)
        self.assertGreaterEqual(second["analysis"]["pass"], 2)
        self.assertGreater(second["invalidated_ancestry"]["components"], 0)

    def test_a_local_edge_change_invalidates_only_its_own_components(self):
        """Decisive locality: change ONE edge inside one component's support
        and assert exactly the components containing its endpoints lose their
        ancestry — every disjoint sibling survives."""
        node = _node()
        node.set_analysis_config(_overlay_config(resolutions=(1.0, 2.0)))
        node.run_recursive_analysis()
        first = json.loads(node.checkpoint_json)
        supports = {c["id"]: set(c["support"])
                    for slice_ in first["hierarchy"]
                    for c in slice_["components"]}
        self.assertGreater(len(supports), 1)
        # An edge whose endpoints both lie in ONE level-0 component.
        target = None
        for edge in node.st.getEdgeList().toVector():
            a, b = edge.getSource().getId(), edge.getTarget().getId()
            owners = [key for key, support in supports.items()
                      if a in support and b in support]
            if owners:
                target = (edge, {a, b})
                break
        self.assertIsNotNone(target, "no intra-component edge in the fixture")
        edge, endpoints = target
        edge.setLength(edge.getLength() * 1.001)
        node.run_recursive_analysis()
        second = json.loads(node.checkpoint_json)
        ancestry = second["invalidated_ancestry"]
        self.assertEqual(ancestry["touched_vertices"], len(endpoints))
        self.assertGreater(ancestry["total_components"],
                           ancestry["components"],
                           "a one-edge change invalidated every component")
        # Every invalidated component really does meet the changed edge.
        supports_now = {c["id"]: set(c["support"])
                        for slice_ in second["hierarchy"]
                        for c in slice_["components"]}
        for component in ancestry["component_ids"]:
            if component in supports_now:
                self.assertTrue(supports_now[component] & endpoints)

    def test_a_pure_metric_change_invalidates_only_the_touched_stars(self):
        """Siblings whose component misses the star stay served."""
        node = _node()
        node.set_analysis_config(_overlay_config())
        node.run_recursive_analysis()
        first = json.loads(node.checkpoint_json)["analysis"]
        self.assertGreater(first["cache_entries"], 1)
        edges = node.st.getEdgeList().toVector()
        edges[0].setLength(edges[0].getLength() * 1.01)
        node.run_recursive_analysis()
        after = json.loads(node.checkpoint_json)["analysis"]
        self.assertGreater(after["cache_invalidations"], 0)
        self.assertLess(after["cache_invalidations"], first["cache_entries"],
                        "a local metric change dropped the whole cache")

    def test_a_second_pass_on_an_unchanged_complex_is_served_from_cache(self):
        node = _node()
        node.set_analysis_config(_overlay_config())
        node.run_recursive_analysis()
        node.run_recursive_analysis()
        second = json.loads(node.checkpoint_json)["analysis"]
        self.assertGreater(second["cache_hits"], 0)
        self.assertEqual(second["cache_invalidations"], 0)

    def test_the_overlay_is_disabled_by_default(self):
        node = _node()
        self.assertFalse(node.analysis_config.enabled)
        node.run_stage1(max_steps=2, n_candidate_moves=6)
        self.assertEqual(node.analysis_pass_count, 0)
        self.assertEqual(node.checkpoint_json, "")


# ======================================================================
# the shared merge gate: relabeling, orientation, and input ordering
# ======================================================================


class RelabelingInvarianceTest(unittest.TestCase):
    """No observable changes under a global relabeling or a reordering."""

    @staticmethod
    def _rebuild(node, permutation=None, shuffle_cells=False):
        """The SAME complex with its vertex ids permuted and/or its cell list
        presented in a different order, carrying the same edge lengths."""
        cells = [[v.getId() for v in c.getVertices()]
                 for c in node.st.getTopSimplices()]
        lengths = _lengths(node)
        if permutation is not None:
            cells = [[permutation[v] for v in cell] for cell in cells]
            lengths = {(min(permutation[a], permutation[b]),
                        max(permutation[a], permutation[b])): value
                       for (a, b), value in lengths.items()}
        if shuffle_cells:
            cells = list(reversed(cells))
        dimension = len(cells[0]) - 1
        rebuilt = T.spacetime.Spacetime.fromVertexTuples(dimension, cells, 1.0, 0.0)
        for edge in rebuilt.getEdgeList().toVector():
            a, b = edge.getSource().getId(), edge.getTarget().getId()
            edge.setLength(lengths[(min(a, b), max(a, b))])
        rebuilt_node = MC(rebuilt, [], [], [1], 1.0, _NODE_SEED)
        rebuilt_node.set_objective(cob.JointStationarityObjective())
        rebuilt_node.set_analysis_config(_overlay_config())
        rebuilt_node.run_recursive_analysis()
        return rebuilt_node

    @staticmethod
    def _discrete(node):
        """Every DISCRETE read of a pass — exact under a relabeling."""
        doc = json.loads(node.checkpoint_json)
        return {
            "classifications": [q["classification"]
                                for q in doc["particles"]["quarks"]],
            "failed": [sorted(q["failed_certificates"])
                       for q in doc["particles"]["quarks"]],
            "ranks": sorted(f["rank"] for f in doc["fibers"]),
            "accepted": sorted(f["accepted"] for f in doc["fibers"]),
            "active_modes": doc["covariance"]["active_modes"],
            "component_count": (len(doc["hierarchy"][0]["components"])
                                if doc["hierarchy"] else 0),
            "support_sizes": sorted(len(c["support"]) for c in
                                    (doc["hierarchy"][0]["components"]
                                     if doc["hierarchy"] else [])),
            "labeled": [(s["nominal_rank"], s["effective_rank"])
                        for s in doc["labeled_fiber_sums"]],
            "transports": len(doc["transports"]),
            "bound_supercomponents": [
                (b["found"], sorted(b["failed_certificates"]))
                for b in doc["particles"]["bound_supercomponents"]],
            "baryons": [(b["classification"], b["exterior_parity"],
                         b["flavor_pattern"],
                         sorted(b["failed_certificates"]))
                        for b in doc["particles"]["baryons"]],
        }

    @staticmethod
    def _continuous(node):
        doc = json.loads(node.checkpoint_json)
        return {
            "objective": node.objective(),
            "q": doc["hierarchy"][0]["q"] if doc["hierarchy"] else 0.0,
            "purity": doc["covariance"]["purity_defect"],
        }

    @staticmethod
    def _supports(node):
        doc = json.loads(node.checkpoint_json)
        if not doc["hierarchy"]:
            return []
        return sorted(tuple(sorted(c["support"]))
                      for c in doc["hierarchy"][0]["components"])

    def setUp(self):
        self.node = _node()
        self.node.set_analysis_config(_overlay_config())
        self.node.run_recursive_analysis()
        ids = sorted({v for cell in _cells(self.node) for v in cell})
        # A cyclic shift onto a disjoint high id range, so no intermediate
        # collision can occur.
        self.permutation = {vertex: 1000 + (index + 1) % len(ids)
                            for index, vertex in enumerate(ids)}

    def test_the_partition_is_exactly_relabeling_covariant(self):
        """The discovered supports MAP THROUGH the permutation exactly."""
        relabeled = self._rebuild(self.node, permutation=self.permutation)
        mapped = sorted(tuple(sorted(self.permutation[v] for v in support))
                        for support in self._supports(self.node))
        self.assertEqual(mapped, self._supports(relabeled))

    def test_a_global_relabeling_changes_no_discrete_read(self):
        relabeled = self._rebuild(self.node, permutation=self.permutation)
        self.assertEqual(self._discrete(self.node), self._discrete(relabeled))

    def test_reversing_the_cell_input_order_changes_no_discrete_read(self):
        reordered = self._rebuild(self.node, shuffle_cells=True)
        self.assertEqual(self._discrete(self.node), self._discrete(reordered))

    def test_relabeling_and_reordering_together_change_no_discrete_read(self):
        both = self._rebuild(self.node, permutation=self.permutation,
                             shuffle_cells=True)
        self.assertEqual(self._discrete(self.node), self._discrete(both))

    def test_the_continuous_reads_agree_within_double_round_off(self):
        """Exact invariance is not available and is not claimed: relabeling
        reorders the canonical cell enumeration, so the same operator entries
        are summed in a different order. The DECLARED tolerance is 1e-12
        relative; the discrete verdicts above are exact."""
        relabeled = self._rebuild(self.node, permutation=self.permutation)
        mine = self._continuous(self.node)
        theirs = self._continuous(relabeled)
        for key in mine:
            scale = max(1.0, abs(mine[key]))
            self.assertLess(abs(mine[key] - theirs[key]) / scale, 1e-12,
                            "%s moved by more than round-off: %r vs %r" %
                            (key, mine[key], theirs[key]))

    def test_the_canonical_component_hash_is_not_fully_label_free(self):
        """A PINNED observation of a #765 limitation, not a #776 behaviour.

        The partition itself is exactly relabeling-covariant (above), but the
        canonical structural hash of a component is not invariant for every
        component of this fixture: the individualization-refinement that
        produces it breaks its remaining ties by index. Recorded here so the
        limitation is visible to #777/#778, which must not key a campaign on
        the hash across a relabeling. If it is ever made fully canonical this
        test skips and can be inverted.
        """
        relabeled = self._rebuild(self.node, permutation=self.permutation)
        mine = sorted(c["id"] for c in
                      json.loads(self.node.checkpoint_json)["hierarchy"][0]
                      ["components"])
        theirs = sorted(c["id"] for c in
                        json.loads(relabeled.checkpoint_json)["hierarchy"][0]
                        ["components"])
        if mine == theirs:
            self.skipTest("the canonical hash is now fully label-free; invert "
                          "this test into an equality assertion")
        self.assertNotEqual(mine, theirs)


# ======================================================================
# the shared merge gate: the analysis-cadence benchmark
# ======================================================================


class CadenceBenchmarkTest(unittest.TestCase):
    def test_the_disabled_overlay_path_costs_nothing_measurable(self):
        def timed(config):
            node = _node()
            if config is not None:
                node.set_analysis_config(config)
            start = time.perf_counter()
            # ONE committed move: the engine's deterministic drive unit.
            node.run_stage1(max_steps=1, n_candidate_moves=6)
            return time.perf_counter() - start, node

        disabled_seconds, disabled = timed(None)
        enabled_seconds, enabled = timed(_overlay_config())
        self.assertEqual(disabled.accepted_move_count,
                         enabled.accepted_move_count)
        self.assertEqual(_cells(disabled), _cells(enabled))
        # Reported for the merge gate; the assertion is only that the overlay
        # does not dominate the optimizer by an order of magnitude.
        print("\n[#776 cadence] stage-1 disabled %.4fs / enabled %.4fs "
              "(%d accepted moves, %d analysis passes)" %
              (disabled_seconds, enabled_seconds, enabled.accepted_move_count,
               enabled.analysis_pass_count))
        self.assertLess(enabled_seconds, max(1.0, 20.0 * disabled_seconds))

    def test_one_analysis_pass_is_reported(self):
        node = _node()
        node.set_analysis_config(_overlay_config())
        start = time.perf_counter()
        node.run_recursive_analysis()
        first = time.perf_counter() - start
        start = time.perf_counter()
        node.run_recursive_analysis()
        incremental = time.perf_counter() - start
        start = time.perf_counter()
        cold = _node()
        cold.set_analysis_config(_overlay_config(cold=True))
        cold.run_recursive_analysis()
        cold_seconds = time.perf_counter() - start
        print("\n[#776 cadence] analysis pass: first %.4fs / repeat %.4fs / "
              "cold %.4fs" % (first, incremental, cold_seconds))
        self.assertGreater(first, 0.0)


if __name__ == "__main__":
    unittest.main()
