# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The whole reading can contract through the Gram instead of periods (#1075).

`--readout whole` under the PERIODS pairing is metric-independent, so a run
scored on it cannot improve. The mechanism is exact rather than approximate:
a change of metric moves the harmonic representative of a fixed cohomology
class by a COBOUNDARY, and a coboundary's period around a closed cycle
telescopes to zero.

A boundary torus escapes this because in two dimensions the Hodge star is an
endomorphism of `H^1` with `star^2 = -1`, a complex structure whose
holomorphic line the metric selects; `tau` is a period ratio taken in that
line, which is why it moves. The bulk is three-dimensional, where the star
sends 1-forms to 2-forms, so there is no such line and nothing for the metric
to choose.

The metric content is still in the harmonic space. The period pairing is
simply the one contraction that annihilates it: `Z^T M_1 Z` moves under the
same jitter that leaves the periods at 1e-15. GRAM contracts the block's live
frame against the harmonic columns through `M_1` instead, which is the same
transpose pairing the harmonic Gram uses.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))),
    "examples", "cobordism"))

import qubit_animation as qa  # noqa: E402
from tessera._tessera.cobordism import MultiCobordism as MC  # noqa: E402

GRID, TAU_A, TAU_B, TAU_OUT = 3, 0.3 + 1.1j, -0.2 + 0.8j, "0.1+1.3j"


def seeded(pairing):
    held = {}
    config = qa.build_config(steps=0, grid=GRID, tau_a=TAU_A, tau_b=TAU_B,
                             input_weight=100.0, regge=False, pin_boundary=True,
                             readout="whole", tori=2, output_state=TAU_OUT,
                             whole_pairing=pairing)
    qa.drive(config, progress=False, on_node=lambda node: held.update(node=node))
    return held["node"]


def swept(node):
    """The reading at four jitter amplitudes, lengths restored afterwards."""
    spacetime = node.spacetime()
    edges = list(spacetime.getEdgeList().toVector())
    original = [edge.getLength() for edge in edges]
    target = np.eye(2, dtype=complex)
    values = []
    try:
        for amplitude in (0.0, 0.05, 0.25, 0.75):
            generator = np.random.default_rng(7)
            for edge, length in zip(edges, original):
                step = generator.standard_normal() + 1j * generator.standard_normal()
                edge.setLength(length * (1.0 + amplitude * step))
            values.append(node.whole_harmonic_residual(target))
    finally:
        for edge, length in zip(edges, original):
            edge.setLength(length)
    return values


def test_periods_is_the_default_and_is_unchanged():
    """The reading every recorded run used, to its last digit."""
    node = seeded("periods")
    assert node.whole_pairing == MC.WholePairing.PERIODS
    assert qa.build_config()["whole_pairing"] == "periods"
    assert node.whole_harmonic_obstruction == ""
    assert abs(node.whole_harmonic_residual(np.eye(2, dtype=complex))
               - 0.02430251774083792) < 1e-14


def test_the_periods_reading_does_not_respond_to_the_metric():
    """Not a weak response -- none. This is what makes it undrivable."""
    values = swept(seeded("periods"))
    assert max(abs(v - values[0]) for v in values) < 1e-12


def test_the_gram_reading_does_respond():
    """The point of the flag."""
    node = seeded("gram")
    assert node.whole_pairing == MC.WholePairing.GRAM
    assert node.whole_harmonic_obstruction == "", node.whole_harmonic_obstruction
    values = swept(node)
    assert max(abs(v - values[0]) for v in values) > 1e-3


def test_both_pairings_produce_a_number_on_the_seed():
    for pairing in ("periods", "gram"):
        node = seeded(pairing)
        value = node.whole_harmonic_residual(np.eye(2, dtype=complex))
        assert node.whole_harmonic_obstruction == ""
        assert 0.0 <= value <= 1.0
        assert np.isfinite(value)


def test_an_unknown_pairing_is_refused_by_name():
    with pytest.raises(ValueError, match="unknown whole pairing"):
        qa.build_config(whole_pairing="cycles")


def test_the_flag_parses():
    args = qa.build_parser().parse_args(["run", "--whole-pairing", "gram"])
    assert args.whole_pairing == "gram"
    assert qa.build_parser().parse_args(["run"]).whole_pairing == "periods"
