# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Fitting one bulk to several input pairs at once (#1017).

A bulk fitted to a SINGLE pair reproduces that pair and nothing else. Measured
on three converged geometries, each against 324 swap readings covering all 108
attachment automorphisms:

    gate   own pair      best other pair
    XX     1.4381e-11    2.4649e-01
    CZ     2.4848e-07    2.0299e-01
    CNOT   3.0719e-07    3.7368e-01

That is what the counting predicts rather than a defect. One 2x2 transfer
against one pair is 8 real numbers less 2 for the complex scale -- six real
constraints -- against some eighty free real bulk coordinates. Nothing ever
asked the geometry to be a MAP.

Cases are how it is asked: each pair adds six constraints on ONE shared bulk.
The sum lives in the objective, which is what makes stage 1 price every
candidate move and stage 2 accept every step against all of them, neither stage
needing to know there is more than one state.

A case carries only a boundary metric and a target, because the transfer depends
on the geometry and the marking's CYCLES alone -- `deriveFrame` normalizes by
transported periods and `readTwoBody` never reads the marking's coefficients.
The input state enters through chi.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))),
    "examples", "cobordism"))

import emergence_animation as ea  # noqa: E402

#: Extra pairs beyond the seed pair. Chosen apart in the upper half plane so
#: no two cases are nearly the same constraint.
EXTRA = ["0.5+0.9j,0.1+1.3j", "1j,1j", "-0.4+1.4j,0.6+0.7j"]


def built(states=()):
    config = ea.build_config(
        inputs="qubit", operator="cnot", score_leak=False, pin_boundary=True,
        input_weight=100.0, grid=3, steps=0, tau_a=0.3 + 1.1j,
        tau_b=-0.2 + 0.8j, regge=False, seed=5, states=list(states))
    node, _inputs = ea.build_qubit_node(config)
    return node


def lengths(node):
    return [complex(e.getLength())
            for e in node.spacetime().getEdgeList().toVector()]


def test_no_cases_is_the_single_target_unchanged():
    """Additive: a run that set one target must score exactly as before."""
    node = built()
    assert node.two_body_case_count() == 0
    assert node.two_body_residual_over_cases() == node.two_body_residual()
    assert node.r_u(node.spacetime()) == pytest.approx(
        node.two_body_residual(), rel=1e-12)


def test_the_objective_is_the_sum_over_cases():
    """Arithmetic, so a reader can see exactly what is being minimised."""
    singles = []
    for pair in EXTRA:
        one = built([pair])
        # Two cases: the seed pair is always case zero, the extra is case one.
        assert one.two_body_case_count() == 2
        singles.append(one.two_body_residual_over_cases())

    seed_only = built().two_body_residual()
    # Each two-case total is the seed case plus that pair's own case, so the
    # per-pair contributions can be recovered and summed independently.
    contributions = [total - seed_only for total in singles]
    together = built(EXTRA)
    assert together.two_body_case_count() == len(EXTRA) + 1
    assert together.two_body_residual_over_cases() == pytest.approx(
        seed_only + sum(contributions), rel=1e-9)


def test_the_case_count_grows_the_objective():
    """More states cannot be easier: each adds a non-negative term."""
    totals = [built(EXTRA[:k]).two_body_residual_over_cases()
              for k in range(len(EXTRA) + 1)]
    assert all(b >= a for a, b in zip(totals, totals[1:])), totals
    assert totals[-1] > totals[0]


def test_evaluating_restores_the_geometry():
    """The cases READ the complex under several boundary conditions.

    Each case writes its own boundary metric to be scored. If a write outlived
    the read, every later case and everything the caller did next would be
    scored on a boundary nobody asked for -- and the drive would be optimising
    a geometry it never saw.
    """
    node = built(EXTRA)
    before = lengths(node)
    for _ in range(3):
        node.two_body_residual_over_cases()
    assert lengths(node) == before

    # The same through the objective the drive actually calls.
    node.r_u(node.spacetime())
    assert lengths(node) == before


def test_the_descent_direction_accounts_for_every_case():
    """The gradient is the sum, so a step trades the cases against each other.

    Asserted as a difference: adding cases must CHANGE the direction stage 2
    steps along. A gradient that ignored the extra cases would leave it
    identical and the run would silently optimise one state.
    """
    one = np.asarray(built().fiber_mode_ascent()[0]).ravel()
    many = np.asarray(built(EXTRA).fiber_mode_ascent()[0]).ravel()
    assert one.shape == many.shape
    assert np.abs(many - one).max() > 1e-9, "the extra cases moved nothing"


def test_relaxation_lowers_the_summed_objective():
    """The point of the feature: one bulk improving for several states at once.

    Not that any single case improves -- a step may trade one against another,
    which is exactly what fitting a map means -- but that the TOTAL falls.
    """
    node = built(EXTRA)
    before = node.two_body_residual_over_cases()
    list(node.run_stage2(max_iters=6, tolerance=1e-15))
    assert node.two_body_residual_over_cases() < before


def test_a_case_without_a_target_is_refused_by_name():
    node = built()
    with pytest.raises(ValueError, match="empty target"):
        node.set_two_body_cases([([], np.zeros((0, 0), dtype=complex))])


def test_the_run_document_records_the_states():
    """A record must say which states a geometry was fitted to."""
    config = ea.build_config(inputs="qubit", states=EXTRA)
    assert config["states"] == EXTRA
    assert ea.build_config(inputs="qubit")["states"] == []


def test_a_candidate_complex_is_scored_on_every_case():
    """Stage 1 must rank moves by the SAME objective stage 2 minimises.

    Every candidate move is priced on a complex REBUILT from a snapshot, never
    on the live one. An implementation that scored the cases only when handed
    the live complex falls back to the single target -- ranking combinatorial
    moves by the FIRST input pair while stage 2 optimises the sum over all of
    them. Two different objectives, with the move search on the wrong one.

    Asserted through what stage 1 reports: its trace opens with the objective it
    is descending, which must be the sum.
    """
    node = built(EXTRA)
    summed = node.two_body_residual_over_cases()
    case_zero = node.two_body_residual()
    # The two must be genuinely different numbers or this proves nothing.
    assert summed > case_zero * 1.5, (summed, case_zero)

    trace = list(node.run_stage1(max_steps=1, n_candidate_moves=0,
                                 max_lookahead=1))
    assert trace, "stage 1 reported no objective"
    assert trace[0] == pytest.approx(summed, rel=1e-6), (trace[0], summed, case_zero)
