# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""`--edge-disposition` on the qubit collar's interior (#1074).

`seed_collar` writes each torus's own lengths onto its edges and wires every
other edge to 1.0, so a qubit host was all-spacelike in its bulk with no way
to say otherwise. The flag existed in the combined driver but was INERT in
qubit mode -- its only consumers were the neutral host builder and a caption
in the neutral title branch -- and #1073 dropped it correctly. This makes it
real rather than restoring a no-op.

The tori's OWN edges are never written. They carry the declared input moduli,
so a disposition over them would change the input states rather than the bulk
they sit in.
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
import qubit_animation as qa  # noqa: E402

GRID, TAU_A, TAU_B = 3, 0.3 + 1.1j, -0.2 + 0.8j


def seeded(disposition):
    held = {}
    config = qa.build_config(steps=0, grid=GRID,
                             tau_a=TAU_A, tau_b=TAU_B, input_weight=100.0,
                             regge=False, pin_boundary=True, readout="whole",
                             tori=2, output_state="0.1+1.3j",
                             interior_disposition=disposition)
    qa.drive(config, progress=False, on_node=lambda node: held.update(node=node))
    node = held["node"]
    boundary, interior = [], []
    ids = held.get("ids")
    spacetime = node.spacetime()
    # A torus edge is one whose endpoints are BOTH in an input block's vertex
    # set and which the torus itself carries; the engine's seeded inputs name
    # those vertex sets.
    blocks = [set(int(v) for v in node.inputs[i].vertices)
              for i in range(len(node.inputs))]
    for edge in spacetime.getEdgeList().toVector():
        if edge is None or edge.getSource() is None or edge.getTarget() is None:
            continue
        u, v = int(edge.getSource().getId()), int(edge.getTarget().getId())
        if any(u in block and v in block for block in blocks):
            boundary.append(complex(edge.getLength()))
        else:
            interior.append(complex(edge.getLength()))
    return boundary, interior


@pytest.fixture(scope="module")
def spacelike():
    return seeded("spacelike")


def test_timelike_squares_the_interior_negative(spacelike):
    """l = i, so l^2 = -1 on every interior edge."""
    _, interior = seeded("timelike")
    assert interior
    assert all(abs(length ** 2 + 1.0) < 1e-12 for length in interior)


def test_lightlike_puts_the_interior_on_the_cone(spacelike):
    """Re(l) == Im(l) > 0 with |l| = 1, so l^2 = i exactly."""
    _, interior = seeded("lightlike")
    assert interior
    assert all(abs(length ** 2 - 1j) < 1e-12 for length in interior)


def test_random_keeps_unit_modulus_and_is_not_all_spacelike(spacelike):
    _, interior = seeded("random")
    assert interior
    assert all(abs(abs(length) - 1.0) < 1e-12 for length in interior)
    assert any(abs(length - 1.0) > 1e-6 for length in interior)


def test_the_tori_are_left_alone(spacelike):
    """The declared input states are not rewritten by a bulk disposition."""
    reference, _ = spacelike
    for disposition in ("timelike", "lightlike", "random"):
        boundary, _ = seeded(disposition)
        assert len(boundary) == len(reference)
        assert max(abs(a - b) for a, b in zip(sorted(boundary, key=lambda z: (z.real, z.imag)),
                                              sorted(reference, key=lambda z: (z.real, z.imag)))) < 1e-12


def test_spacelike_is_what_the_collar_already_wires(spacelike):
    _, interior = spacelike
    assert interior
    assert all(abs(length - 1.0) < 1e-12 for length in interior)


def test_the_flag_reaches_the_config_and_the_record():
    config = qa.build_config(interior_disposition="timelike")
    assert config["interior_disposition"] == "timelike"
    args = qa.build_parser().parse_args(["run", "--edge-disposition", "foliated"])
    assert args.interior_disposition == "foliated"


def test_an_unknown_disposition_is_refused_by_name():
    with pytest.raises(ValueError, match="unknown edge disposition"):
        qa.build_config(interior_disposition="sideways")
