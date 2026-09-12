# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Stage 2 descends the gradient of the objective it is minimising (#1055).

`fiberModeAscent` dispatches each two-body term to the selected reading's own
gradient. Transfer and whole-harmonic reads have analytic derivatives; bulk
and paired-operator reads take the numerical ascent of `rU` instead of
borrowing another objective's direction.
"""
import os
import sys

import numpy as np
import pytest

from tessera import cobordism as cob

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))),
    "examples", "cobordism"))

import emergence_animation as ea  # noqa: E402

MC = cob.MultiCobordism
R = MC.ReadoutMode


def node():
    """A host with no boundary: the flag is about the READING, not the host."""
    return MC(MC.seed_simplex(3), [], [], degrees=[1], seed=0, precone=2,
              einstein_hilbert=False)


@pytest.mark.parametrize("modes,analytic", [
    ([R.TRANSFER], True),
    ([R.WHOLE], True),
    ([R.BULK], False),
    ([R.OPERATOR], False),
    ([R.TRANSFER, R.WHOLE], True),
    ([R.WHOLE, R.OPERATOR], False),
])
def test_only_implemented_readings_have_one(modes, analytic):
    """A SET is analytic only if every reading in it is."""
    live = node()
    live.set_readout_modes(modes)
    assert live.readouts_have_analytic_gradient is analytic


def test_the_default_is_the_transfer_and_is_analytic():
    """The path every recorded run took stays the analytic one."""
    live = node()
    assert [mode for mode in live.readout_modes] == [R.TRANSFER]
    assert live.readouts_have_analytic_gradient is True


def test_readout_modes_are_a_set_preserving_first_occurrence():
    live = node()
    live.set_readout_modes(
        [R.WHOLE, R.TRANSFER, R.WHOLE, R.BULK, R.TRANSFER])
    assert list(live.readout_modes) == [R.WHOLE, R.TRANSFER, R.BULK]

    live.set_readout_modes([R.TRANSFER, R.TRANSFER])
    assert list(live.readout_modes) == [R.TRANSFER]
    assert live.readouts_have_analytic_gradient is True


@pytest.mark.parametrize("mode", [R.BULK, R.OPERATOR])
def test_public_fiber_ascent_refuses_an_unimplemented_selected_gradient(mode):
    live = node()
    live.set_two_body_target(np.eye(1, dtype=complex))
    live.set_readout_modes([mode])
    with pytest.raises(RuntimeError, match="no analytic gradient"):
        live.fiber_mode_ascent()
