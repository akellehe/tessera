# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The whole-complex reading is taken in the period frame (#1058).

`wholeHarmonicResidualOn` fits the transported periods of the band's columns
and compares the resulting coefficient vector to the wanted state. Those
coefficients live in whatever basis the band returned, and a band's basis is
not a quantity the geometry carries: a Riesz band hands back the singular
vectors of its projector, a null-space band those of `S^U`. Read there, the
two disagreed -- 0.9412 against 0.9272 on one geometry -- while spanning the
same subspace to 5e-15.

The frame the spec names is the period frame (S6, D3, and the glossary's
`SimplicialQubit::periodFrame`): the band normalised so a marking group's
cycles read periods (1,0) and (0,1). The coefficients there are the periods
of the FORM, which is basis-free, and `wanted`'s own (1, tau) shape is the
same kind of object.

These tests hold the two properties that makes true: the reading does not
depend on which band produced it, and the marking groups agree about the
frame -- the monodromy the spec already names.
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
from tessera._tessera.cobordism import PencilLayer  # noqa: E402

TAU_A, TAU_B, GRID = 0.3 + 1.1j, -0.2 + 0.8j, 3
TAU_OUT = 0.1 + 1.3j


@pytest.fixture(scope="module")
def seeded():
    held = {}
    config = ea.build_config(steps=0, inputs=ea.InputMode.QUBIT, grid=GRID,
                             operator="xx", tau_a=TAU_A, tau_b=TAU_B,
                             input_weight=100.0, regge=False, pin_boundary=True,
                             readout="whole", tori=2, output_state=str(TAU_OUT))
    ea.drive(config, progress=False, on_node=lambda node: held.update(node=node))
    node = held["node"]
    assembled = PencilLayer.assemble([node.spacetime()])
    cycles, inputs, spans = [], [], []
    index = 0
    while True:
        try:
            marking = node.input_marking(index)
        except Exception:  # noqa: BLE001 -- the engine says "no such block" by throwing
            break
        if marking is None:
            break
        start = len(cycles)
        cycles.extend(marking.cycles)
        inputs.extend(complex(v) for v in marking.coefficients)
        spans.append((start, len(cycles)))
        index += 1
    wanted = np.array([1.0 + 0j, TAU_OUT], dtype=complex)
    return dict(node=node, assembled=assembled, op=assembled.op,
                connection=assembled.op.connection(), cycles=cycles, spans=spans,
                inputs=np.array(inputs, dtype=complex),
                wanted=wanted / np.linalg.norm(wanted))


def periods_of(connection, images, cycles):
    images = np.asarray(images)
    return np.array([[connection.transportedPeriod(images[:, a], cycle)
                      for a in range(images.shape[1])] for cycle in cycles])


def reading(seeded, images):
    """`wholeHarmonicResidualOn`'s algebra, in the period frame, for any band."""
    periods = periods_of(seeded["connection"], images, seeded["cycles"])
    state, *_ = np.linalg.lstsq(periods, seeded["inputs"], rcond=None)
    low, high = seeded["spans"][0]
    state = periods[low:high, :] @ state           # the period-frame coordinates
    wanted = seeded["wanted"]
    squared = float(np.vdot(state, state).real)
    norm = float(np.vdot(wanted, wanted).real)
    return max(0.0, (norm - abs(np.vdot(state, wanted)) ** 2 / squared) / norm)


def test_the_engine_reads_what_the_period_frame_says(seeded):
    """The replication is faithful, so the tests below are about the engine."""
    band = seeded["op"].harmonicBand(1)
    engine = seeded["node"].whole_harmonic_residual(np.eye(2, dtype=complex))
    assert seeded["node"].whole_harmonic_obstruction == ""
    assert abs(engine - reading(seeded, band.images)) < 1e-12


def test_the_reading_does_not_depend_on_which_band_produced_it(seeded):
    """The property the whole change is for.

    Read in the band's own basis these two differ in the second decimal. Read
    in the period frame they are one number, because the frame is fixed by the
    markings and not by whichever matrix the band came out of.
    """
    contour = seeded["op"].band(
        1, PencilLayer.harmonic_contour(seeded["assembled"], 1))
    harmonic = seeded["op"].harmonicBand(1)
    assert abs(reading(seeded, contour.images) - reading(seeded, harmonic.images)) < 1e-12


def test_the_bands_really_do_return_different_bases(seeded):
    """Otherwise the test above would hold for an uninteresting reason."""
    contour = np.asarray(seeded["op"].band(
        1, PencilLayer.harmonic_contour(seeded["assembled"], 1)).images)
    harmonic = np.asarray(seeded["op"].harmonicBand(1).images)
    transform, *_ = np.linalg.lstsq(contour, harmonic, rcond=None)
    rank = transform.shape[0]
    scalar = np.trace(transform) / rank * np.eye(rank)
    assert np.linalg.norm(transform - scalar) / np.linalg.norm(transform) > 1e-3


def test_every_marking_group_gives_the_same_frame(seeded):
    """The monodromy of spec S6, as the condition that makes the frame the
    whole's rather than a chosen block's."""
    periods = periods_of(seeded["connection"], seeded["op"].harmonicBand(1).images,
                         seeded["cycles"])
    rank = periods.shape[1]
    blocks = [periods[low:high, :] for low, high in seeded["spans"]
              if high - low == rank]
    assert len(blocks) >= 2, "this geometry has only one marking group to compare"
    first = np.linalg.inv(blocks[0])
    for other in blocks[1:]:
        assert np.linalg.norm(other @ first - np.eye(rank)) < 1e-8


def test_the_reading_is_the_same_through_either_marking_group(seeded):
    """What the monodromy buys: the residual itself does not move."""
    band = seeded["op"].harmonicBand(1)
    periods = periods_of(seeded["connection"], band.images, seeded["cycles"])
    rank = periods.shape[1]
    state, *_ = np.linalg.lstsq(periods, seeded["inputs"], rcond=None)
    wanted = seeded["wanted"]
    norm = float(np.vdot(wanted, wanted).real)
    values = []
    for low, high in seeded["spans"]:
        if high - low != rank:
            continue
        coordinates = periods[low:high, :] @ state
        squared = float(np.vdot(coordinates, coordinates).real)
        values.append(max(0.0, (norm - abs(np.vdot(coordinates, wanted)) ** 2 / squared) / norm))
    assert len(values) >= 2
    assert max(values) - min(values) < 1e-12
