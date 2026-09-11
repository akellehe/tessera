# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""A conjugate torus pair carries one state, so it is imposed once (#1062).

A torus and its orientation reversal have the SAME period block on the whole's
zero mode -- measured at 1.4e-15 with the relating map the identity -- so they
are one constraint on it, not two. Imposing `(1, tau)` on one and
`(1, -conj tau)` on the other asks that constraint to take two values, which
is the equation `psi = Z conj(psi)`, whose solutions are exactly `Re tau = 0`.
The least-squares fit returned the nearest such state, and the real part of
every input was annihilated.

No gluing repairs that: `psi -> Z conj(psi)` is ANTILINEAR while a cobordism's
period map is complex-linear, so no linear map relates the two blocks. The
transpose adjoint on a twisted I-bundle is the structural answer and is
tracked separately; imposing the pair once is what makes the reading usable
in the meantime.

The pairing is DECLARED. At two tori the blocks are equally identical in their
periods but carry two DIFFERENT states, so no measurement of the periods can
distinguish a conjugate pair from a pair of qubits.
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


def seed(tori, **overrides):
    held = {}
    config = ea.build_config(steps=0, inputs=ea.InputMode.QUBIT, grid=GRID,
                             operator="xx", tau_a=TAU_A, tau_b=TAU_B,
                             input_weight=100.0, regge=False, pin_boundary=True,
                             readout="whole", tori=tori, **overrides)
    ea.drive(config, progress=False, on_node=lambda node: held.update(node=node))
    return held["node"]


def read(node):
    """The periods of the whole's zero mode over every marked cycle."""
    assembled = PencilLayer.assemble([node.spacetime()])
    connection = assembled.op.connection()
    band = assembled.op.harmonicBand(1)
    images = np.asarray(band.images)
    cycles, inputs, spans = [], [], []
    index = 0
    while True:
        try:
            marking = node.input_marking(index)
        except Exception:  # noqa: BLE001 -- absence is signalled by throwing
            break
        if marking is None:
            break
        start = len(cycles)
        cycles.extend(marking.cycles)
        inputs.extend(complex(v) for v in marking.coefficients)
        spans.append((start, len(cycles)))
        index += 1
    periods = np.array([[connection.transportedPeriod(images[:, a], cycle)
                         for a in range(band.rank())] for cycle in cycles])
    return periods, np.array(inputs, dtype=complex), spans


@pytest.fixture(scope="module")
def four():
    node = seed(4)
    periods, inputs, spans = read(node)
    return dict(node=node, periods=periods, inputs=inputs, spans=spans)


def test_four_tori_declares_conjugate_pairs(four):
    assert four["node"].conjugate_input_pairs is True


def test_two_tori_does_not(four):
    assert seed(2, output_state="0.1+1.3j").conjugate_input_pairs is False


def test_a_conjugate_pair_is_one_constraint_not_two(four):
    """The measurement the whole change rests on: the partners' period blocks
    are the same block, so the pair cannot hold two different values."""
    periods, spans = four["periods"], four["spans"]
    rank = periods.shape[1]
    for x, y in ((0, 1), (2, 3)):
        first = periods[spans[x][0]:spans[x][1], :]
        second = periods[spans[y][0]:spans[y][1], :]
        transform, *_ = np.linalg.lstsq(first.T, second.T, rcond=None)
        assert np.linalg.norm(second - transform.T @ first) / np.linalg.norm(second) < 1e-10
        assert np.linalg.norm(transform.T - np.eye(2)) < 1e-8
        # and together they span half of what the space has room for
        singular = np.linalg.svd(np.vstack([first, second]), compute_uv=False)
        assert int((singular > singular[0] * 1e-10).sum()) == rank // 2


def test_the_rank_was_never_the_problem(four):
    """Two pairs, one state each, four in total: the space IS fully spanned."""
    singular = np.linalg.svd(four["periods"], compute_uv=False)
    assert int((singular > singular[0] * 1e-10).sum()) == four["periods"].shape[1] == 4


def test_the_imposed_rows_are_carried_exactly(four):
    """Square and consistent once each pair is imposed once."""
    periods, inputs, spans = four["periods"], four["inputs"], four["spans"]
    kept = [row for block in range(0, len(spans), 2)
            for row in range(spans[block][0], spans[block][1])]
    state, *_ = np.linalg.lstsq(periods[kept, :], inputs[kept], rcond=None)
    relative = (np.linalg.norm(periods[kept, :] @ state - inputs[kept])
                / np.linalg.norm(inputs[kept]))
    assert relative < 1e-12


def test_the_real_part_of_every_modulus_survives(four):
    """What the bug destroyed. Feeding both values annihilated Re tau."""
    periods, inputs, spans = four["periods"], four["inputs"], four["spans"]
    kept = [row for block in range(0, len(spans), 2)
            for row in range(spans[block][0], spans[block][1])]
    state, *_ = np.linalg.lstsq(periods[kept, :], inputs[kept], rcond=None)
    carried = periods @ state
    for block in range(0, len(spans), 2):
        low, _ = spans[block]
        asked = inputs[low + 1] / inputs[low]
        got = carried[low + 1] / carried[low]
        assert abs(got - asked) < 1e-10, f"block {block}: asked {asked}, got {got}"
        assert abs(asked.real) > 1e-3, "this test is vacuous on a purely imaginary modulus"


def test_the_partner_reads_back_its_partners_state(four):
    """The conjugate is the same state seen the other way, so it reads the
    same value rather than its own -conj(tau)."""
    periods, inputs, spans = four["periods"], four["inputs"], four["spans"]
    kept = [row for block in range(0, len(spans), 2)
            for row in range(spans[block][0], spans[block][1])]
    state, *_ = np.linalg.lstsq(periods[kept, :], inputs[kept], rcond=None)
    carried = periods @ state
    for block in range(0, len(spans), 2):
        low, _ = spans[block]
        partner, _ = spans[block + 1]
        assert abs(carried[partner + 1] / carried[partner]
                   - carried[low + 1] / carried[low]) < 1e-10


def test_two_tori_is_unchanged(four):
    """The declaration is off there, so the reading is what it was."""
    node = seed(2, output_state="0.1+1.3j")
    assert node.whole_harmonic_obstruction == ""
    assert abs(node.whole_harmonic_residual(np.eye(2, dtype=complex))
               - 0.02430251774083792) < 1e-14
