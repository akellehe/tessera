# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The paired-frame transfer reads the OPERATOR, not its image (#1048).

With conjugate pairs the boundary is four tori, two per state, and the transfer
between the two PAIRS is read between rank-4 frames. That makes it 4x4 --
sixteen numbers, the dimension of an operator on C^2 (x) C^2 -- so `vec(T)` is
that operator's Choi state and the target is the GATE rather than the gate's
image of one chosen input pair.

That is a different claim from every earlier reading. A geometry representing G
acts correctly on every input by construction, where one fitted to chi has been
fitted to one of them: the swap sweep measured a fitted bulk holding its own
states at 1e-8 and every other at 1e-2 to 9e-1, and the state ladder measured
five states competing for one bulk with the outlier paying.

A reading that STRUCTURALLY cannot produce a number is refused when the config
is built, not scored as a constant 1.0. A constant term has no gradient, so the
flag would silently be a no-op while the run looked like it was optimizing.
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

TAU_A, TAU_B, GRID = 0.3 + 1.1j, -0.2 + 0.8j, 3


def seeded(**overrides):
    held = {}
    config = ea.build_config(steps=0, inputs=ea.InputMode.QUBIT, grid=GRID,
                             operator="xx", tau_a=TAU_A, tau_b=TAU_B,
                             input_weight=100.0, regge=False,
                             pin_boundary=True, **overrides)
    ea.drive(config, progress=False, on_node=lambda node: held.update(node=node))
    return held["node"]


# ---- the reading ----

def test_the_target_is_the_gate_itself():
    """4x4, not the 2x2 image of one input pair."""
    node = seeded(readout="operator", conjugate_inputs=True)
    gate = np.asarray(node.gate_target)
    assert gate.shape == (4, 4)
    expected = np.asarray(ea.DECLARED_GATES["xx"], dtype=complex)
    assert np.allclose(gate, expected)


def test_it_scores_a_number_and_that_number_is_the_objective_term():
    node = seeded(readout="operator", conjugate_inputs=True)
    residual = node.operator_residual()
    assert 0.0 <= residual <= 1.0
    assert node.two_body_residual() == pytest.approx(residual, rel=1e-12)


def test_a_summed_set_adds_the_readings():
    """Every reading is the same projective leak on one scale, so they sum."""
    whole = seeded(readout="whole", conjugate_inputs=True).two_body_residual()
    operator = seeded(readout="operator", conjugate_inputs=True).two_body_residual()
    both = seeded(readout="whole,operator", conjugate_inputs=True).two_body_residual()
    assert both == pytest.approx(whole + operator, rel=1e-9)


# ---- incompatible inputs are refused by name ----

@pytest.mark.parametrize("overrides,expected", [
    (dict(readout="whole"), "conjugate-inputs"),
    (dict(readout="operator"), "conjugate-inputs"),
    (dict(readout="transfer", conjugate_inputs=True), "cannot be used"),
    (dict(readout="transfer,operator", conjugate_inputs=True), "cannot be used"),
    (dict(readout="bulk", layers=1), "two collar layers"),
    (dict(readout=""), "no reading"),
    (dict(readout="nope"), "unknown readout"),
])
def test_refused(overrides, expected):
    with pytest.raises(ValueError) as caught:
        ea.build_config(**overrides)
    assert expected in str(caught.value), str(caught.value)


@pytest.mark.parametrize("overrides", [
    dict(readout="transfer"),
    dict(readout="whole,operator", conjugate_inputs=True),
    dict(readout="bulk", layers=3),
])
def test_accepted(overrides):
    assert ea.build_config(**overrides)["readout"] == overrides["readout"]
