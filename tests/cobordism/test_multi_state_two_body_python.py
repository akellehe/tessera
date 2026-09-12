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

Cases are how it is asked. Under the transfer reading each pair adds six
constraints on ONE shared bulk; another selected reading adds its own residual
for that case. The sum lives in the objective, which is what makes stage 1
price every candidate move and stage 2 accept every step against all of them,
neither stage needing to know there is more than one state.

A case carries a boundary metric, a target, and its marking coefficients. The
transfer depends on the geometry and marking CYCLES alone, but the whole-harmonic
read also needs that case's coefficients to select the input-determined form.
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
TAU_OUT = 0.1 + 1.3j


def built(states=()):
    config = ea.build_config(
        inputs="qubit", operator="cnot", score_leak=False, pin_boundary=True,
        input_weight=100.0, grid=3, steps=0, tau_a=0.3 + 1.1j,
        tau_b=-0.2 + 0.8j, regge=False, seed=5, states=list(states))
    node, _inputs = ea.build_qubit_node(config)
    return node


def whole_built():
    config = ea.build_config(
        inputs="qubit", readout="whole", output_state=str(TAU_OUT),
        pin_boundary=True, input_weight=100.0, grid=3, steps=0,
        tau_a=0.3 + 1.1j, tau_b=-0.2 + 0.8j, regge=False)
    node, _inputs = ea.build_qubit_node(config)
    return node


def lengths(node):
    return [complex(e.getLength())
            for e in node.spacetime().getEdgeList().toVector()]


def first_input_edge(node):
    vertices = {int(v) for v in node.inputs[0].vertices}
    return next(
        edge for edge in node.spacetime().getEdgeList().toVector()
        if int(edge.getSource().getId()) in vertices
        and int(edge.getTarget().getId()) in vertices)


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


def test_case_reads_restore_the_exact_square_root_branch():
    """Restoring l^2 through sqrt must not replace a live length by -l."""
    node = built(EXTRA[:1])
    edge = first_input_edge(node)
    edge.setLength(-complex(edge.getLength()))
    before = lengths(node)
    node.two_body_residual_over_cases()
    assert lengths(node) == before


def test_duplicate_case_edges_are_refused_before_they_can_restore_wrong():
    node = built()
    edge = first_input_edge(node)
    source = int(edge.getSource().getId())
    target = int(edge.getTarget().getId())
    squared = complex(edge.getLength()) ** 2
    chi = np.asarray(node.two_body_target().chi)
    with pytest.raises(ValueError, match="repeats boundary edge"):
        node.set_two_body_cases([(
            [(source, target, squared), (target, source, squared)], chi)])


def test_case_shape_validation_does_not_depend_on_readout_setter_order():
    node = built()
    node.set_readout_modes([ea.MC.ReadoutMode.WHOLE])
    with pytest.raises(ValueError, match="single target"):
        node.set_two_body_cases([([], np.ones((3, 3), dtype=complex))])


def test_cases_set_before_targets_or_attachments_share_target_shapes():
    node = ea.MC(ea.MC.seed_simplex(3), [], [], degrees=[1],
                 einstein_hilbert=False)
    with pytest.raises(ValueError, match=r"case 1 target.*case 0"):
        node.set_two_body_cases([
            ([], np.eye(2, dtype=complex)),
            ([], np.eye(3, dtype=complex)),
        ])
    with pytest.raises(ValueError,
                       match=r"case 1 paired direct-sum.*case 0"):
        node.set_two_body_cases([
            ([], np.eye(2, dtype=complex), True, np.eye(2, dtype=complex)),
            ([], np.eye(2, dtype=complex), True, np.eye(3, dtype=complex)),
        ])


def test_whole_cases_read_their_own_input_coefficients():
    node = whole_built()
    node.set_input_residual_weight(0.0)
    chi = np.asarray(node.two_body_target().chi)
    hit = np.array([1.0, TAU_OUT, 1.0, TAU_OUT], dtype=complex)
    miss = np.array([1.0, 0.0, 1.0, 0.0], dtype=complex)
    node.set_two_body_cases([([], chi, True, None, hit)])
    assert node.two_body_residuals_per_case() == pytest.approx(
        [0.0], abs=1e-12)
    assert np.max(np.abs(np.asarray(node.fiber_mode_ascent()[0]))) < 1e-10

    node.set_two_body_cases([
        ([], chi, True, None, hit),
        ([], chi, True, None, miss),
    ])
    residuals = node.two_body_residuals_per_case()
    assert residuals[0] < 1e-12
    assert residuals[1] == pytest.approx(
        abs(TAU_OUT) ** 2 / (1.0 + abs(TAU_OUT) ** 2), abs=1e-10)


def test_one_refused_selected_gradient_does_not_erase_another():
    node = whole_built()
    node.set_input_residual_weight(0.0)
    markings = [node.input_marking(index) for index in range(2)]
    # One independent cycle from each torus frames the whole rank-2 space, but
    # neither rank-1 marking can frame its own rank-2 torus. Whole can read;
    # the transfer therefore contributes the constant full leak and refuses
    # only its own derivative.
    node.set_input_marking(0, [markings[0].cycles[0]], [1.0 + 0j])
    node.set_input_marking(1, [markings[1].cycles[1]], [TAU_OUT])
    node.set_two_body_target(np.eye(1, dtype=complex))

    node.set_readout_modes([ea.MC.ReadoutMode.TRANSFER])
    assert node.two_body_residual() == 1.0
    with pytest.raises(RuntimeError, match=r"own kernel has rank 2"):
        node.two_body_residual_gradient()

    node.set_readout_modes([ea.MC.ReadoutMode.WHOLE])
    assert node.whole_harmonic_residual(np.eye(1, dtype=complex)) < 1.0
    assert node.whole_harmonic_obstruction == ""
    whole_only = np.asarray(node.fiber_mode_ascent()[0])

    node.set_readout_modes(
        [ea.MC.ReadoutMode.TRANSFER, ea.MC.ReadoutMode.WHOLE])
    np.testing.assert_array_equal(
        np.asarray(node.fiber_mode_ascent()[0]), whole_only)


@pytest.mark.parametrize("case,match", [
    (([], np.zeros((2, 2), dtype=complex)), "finite, nonzero target"),
    (([], np.full((2, 2), np.inf, dtype=complex)), "finite, nonzero target"),
    (([], np.eye(2), True, np.zeros((2, 2), dtype=complex)),
     "finite, nonzero paired direct-sum"),
    (([], np.eye(2), True, None, np.ones(3, dtype=complex)),
     "marked input blocks"),
    (([], np.eye(2), True, None, np.zeros(4, dtype=complex)),
     "all-zero input coefficients"),
])
def test_invalid_case_payloads_are_refused_at_the_setter(case, match):
    node = built()
    with pytest.raises(ValueError, match=match):
        node.set_two_body_cases([case])


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
