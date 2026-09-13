# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""A run that never moved says why (#1080).

A drive that does not move exits as `tolerance-reached`, which reads as
convergence. Two configurations reach it while doing nothing:

  * `--readout whole` under the periods pairing cannot respond to the
    geometry at all. The reading pairs cohomology with homology, so it sees
    only a class; a change of metric moves the harmonic representative by
    exactly a coboundary, whose period around a closed cycle telescopes to
    zero. Every unit is a stall by construction.
  * `--edge-disposition random` seeds the collar's interior at uniformly
    random phase and leaves no improving move on the first unit. That flag
    was inert in qubit mode until it was made real, so a command that carried
    it harmlessly now stalls.

Reported from a real run as "exits after one try, ignoring --patience".
Patience was working and counting genuine stalls; nothing said what was
stalling. These tests hold the naming, and that it stays quiet for a run that
actually moved.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))),
    "examples", "cobordism"))

import emergence_animation as ea  # noqa: E402
import qubit_animation as qa  # noqa: E402


class FakeFrame:
    """Only what `_objective_total` reads."""

    def __init__(self, total):
        self.objective = {"total": total}


def config_for(**overrides):
    base = dict(steps=4, grid=3, tau_a=0.3 + 1.1j, tau_b=-0.2 + 0.8j, tori=2,
                regge=False, pin_boundary=True, readout="transfer")
    base.update(overrides)
    return qa.build_config(**base)


def test_a_frozen_whole_reading_is_named():
    config = config_for(readout="whole", output_state="0.1+1.3j")
    frames = [FakeFrame(0.0243)] * 5
    lines = qa.stall_report(config, frames)
    assert lines, "a run that never moved said nothing"
    body = "\n".join(lines)
    assert "did not move over 4 engine units" in body
    assert "--readout whole" in body
    assert "coboundary" in body
    assert "--whole-pairing gram" in body


def test_a_non_default_disposition_is_named():
    config = config_for(interior_disposition="random")
    frames = [FakeFrame(3.93735)] * 3
    body = "\n".join(qa.stall_report(config, frames))
    assert "--edge-disposition random" in body


def test_both_causes_are_named_together():
    config = config_for(readout="whole", output_state="0.1+1.3j",
                        interior_disposition="random")
    body = "\n".join(qa.stall_report(config, [FakeFrame(0.0243)] * 3))
    assert "--readout whole" in body and "--edge-disposition random" in body


def test_a_run_that_moved_says_nothing():
    config = config_for(readout="whole", output_state="0.1+1.3j")
    frames = [FakeFrame(0.5262), FakeFrame(0.0049), FakeFrame(0.0)]
    assert qa.stall_report(config, frames) == []


def test_a_frozen_run_with_no_known_cause_says_nothing():
    """Silence rather than a guess: a transfer run on a default collar that
    stalls has stalled on its geometry, which is a result."""
    config = config_for()
    assert qa.stall_report(config, [FakeFrame(1.0)] * 4) == []


def test_one_frame_is_not_a_stall():
    assert qa.stall_report(config_for(readout="whole",
                                      output_state="0.1+1.3j"),
                           [FakeFrame(0.0243)]) == []


def test_the_gram_pairing_is_not_accused():
    """It responds to the metric, so a stall under it is not this fault."""
    config = config_for(readout="whole", output_state="0.1+1.3j",
                        whole_pairing="gram")
    body = "\n".join(qa.stall_report(config, [FakeFrame(0.1)] * 3))
    assert "--readout whole" not in body
