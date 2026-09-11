# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Stage 2 descends the gradient of the objective it is minimising (#1055).

`fiberModeAscent`'s two-body term is the gradient through the frame TRANSFER,
and it was taken whatever `--readout` selected. Under `whole`, `bulk` or
`operator` the objective is one function and that ascent belongs to another.

The line search evaluates the true objective and accepts only genuine
decreases, so the consequence was wasted trials rather than a wrong answer --
the objective never rose. It is still the wrong direction to be proposing, and
the numerical ascent of `rU` is correct for any of them.

So the analytic path is taken only while every selected reading is the
transfer. The others fall back until they have gradients of their own.
"""
import os
import sys

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
    ([R.WHOLE], False),
    ([R.BULK], False),
    ([R.OPERATOR], False),
    ([R.TRANSFER, R.WHOLE], False),
    ([R.WHOLE, R.OPERATOR], False),
])
def test_only_the_transfer_has_one(modes, analytic):
    """A SET is analytic only if every reading in it is."""
    live = node()
    live.set_readout_modes(modes)
    assert live.readouts_have_analytic_gradient is analytic


def test_the_default_is_the_transfer_and_is_analytic():
    """The path every recorded run took stays the analytic one."""
    live = node()
    assert [mode for mode in live.readout_modes] == [R.TRANSFER]
    assert live.readouts_have_analytic_gradient is True
