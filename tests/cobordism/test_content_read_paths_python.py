# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The operators a content's read forms, against their definitions.

* The quartic of the eliminated fluctuations is
  -1/2 sum_ab (A^-1)_ab O_a O_b on the many-body space, with O_a the second
  quantization of the a-th coupling. `DressedFluctuation.effective_action`
  forms it as -1/2 sum_a O_a (sum_b (A^-1)_ab O_b).
* The second derivative of the covariant operator in two edges of different
  connected components of the complex is zero, and the Hessian of the action
  does not form those pairs.
"""
import itertools

import numpy as np
import pytest

import tessera as T
from tessera import cobordism as cob
from tessera.drivers import baryon_poles as bp
from tessera.drivers import cell_solve as cs

MODES, PARTICLES = 6, 3


def _fluctuation(carrier, couplings, stiffness):
    declaration = cob.DressedFluctuationDeclaration()
    declaration.carrier_dimension = MODES
    declaration.carrier = list(np.asarray(carrier, dtype=complex).reshape(-1))
    declaration.couplings = [list(np.asarray(c, dtype=complex).reshape(-1))
                             for c in couplings]
    declaration.bare_stiffness = list(
        np.asarray(stiffness, dtype=complex).reshape(-1))
    declaration.occupied_modes = PARTICLES
    return cob.DressedFluctuation(declaration)


def _read(carrier, couplings, stiffness):
    frame = list(np.eye(MODES, dtype=complex).reshape(-1))
    read = _fluctuation(carrier, couplings, stiffness).effective_action(
        frame, frame, PARTICLES)
    dimension = int(read.dimension)
    return read, dimension


def _matrix(flat, dimension):
    return np.asarray(flat, dtype=complex).reshape(dimension, dimension)


@pytest.mark.parametrize("count", [1, 2, 5])
def test_the_quartic_is_the_sum_over_the_pairs_of_couplings(count):
    """With a non-normal carrier, ``count`` complex couplings and a complex
    symmetric stiffness: the quartic the library forms equals
    -1/2 sum_ab (A^-1)_ab O_a O_b with every O_a taken from the library's
    own second quantization (the one-body operator of a read whose carrier
    is the coupling), to the rounding of the sums."""
    rng = np.random.default_rng(11 + count)
    carrier = rng.normal(size=(MODES, MODES)) \
        + 1j * rng.normal(size=(MODES, MODES))
    couplings = [rng.normal(size=(MODES, MODES))
                 + 1j * rng.normal(size=(MODES, MODES))
                 for _ in range(count)]
    half = rng.normal(size=(count, count)) + 1j * rng.normal(size=(count,
                                                                    count))
    stiffness = half + half.T + 4.0 * np.eye(count)
    read, dimension = _read(carrier, couplings, stiffness)
    assert dimension == 20
    lifted = []
    for coupling in couplings:
        alone, _ = _read(coupling, [], np.zeros((0, 0)))
        lifted.append(_matrix(alone.one_body, dimension))
    response = np.linalg.inv(stiffness)
    expected = sum(-0.5 * response[a, b] * (lifted[a] @ lifted[b])
                   for a, b in itertools.product(range(count), repeat=2))
    quartic = _matrix(read.quartic, dimension)
    scale = np.linalg.norm(expected)
    assert np.linalg.norm(quartic - expected) <= 1e-13 * scale
    one_body = _matrix(read.one_body, dimension)
    np.testing.assert_allclose(_matrix(read.effective_action, dimension),
                               one_body + quartic, rtol=0, atol=1e-13 * scale)


def test_the_one_body_operator_alone_is_the_full_read_s():
    """`many_body_operators(..., one_body_only=True)` returns the one-body
    operator of the full read, bit for bit, and no read of the quartic."""
    rng = np.random.default_rng(3)
    carrier = rng.normal(size=(MODES, MODES)) \
        + 1j * rng.normal(size=(MODES, MODES))
    couplings = [rng.normal(size=(MODES, MODES)) + 0j for _ in range(3)]
    stiffness = 3.0 * np.eye(3) + 0j
    frame = np.eye(MODES, dtype=complex)
    full, read = bp.many_body_operators(carrier, couplings, stiffness, frame,
                                        frame)
    alone, none = bp.many_body_operators(carrier, couplings, stiffness, frame,
                                         frame, one_body_only=True)
    assert none is None and read is not None
    np.testing.assert_array_equal(alone, full)


def test_the_supremum_over_a_span_takes_a_given_triangular_factor():
    """`_worst_relative` with the triangular factor of the states given is
    the same number as without it."""
    rng = np.random.default_rng(5)
    states = rng.normal(size=(40, 3)) + 1j * rng.normal(size=(40, 3))
    images = rng.normal(size=(40, 3)) + 1j * rng.normal(size=(40, 3))
    assert bp._worst_relative(images, states, bp._triangular(states)) == \
        bp._worst_relative(images, states)
    # the supremum itself: the largest singular value of the map on the span
    orthonormal, _ = np.linalg.qr(states)
    coefficients = np.linalg.lstsq(states, orthonormal, rcond=None)[0]
    expected = np.linalg.norm(images @ coefficients, 2)
    assert bp._worst_relative(images, states) == pytest.approx(expected,
                                                               rel=1e-12)


def _sheeted_action(sheets):
    """The joint action on ``sheets`` copies of a tetrahedron with unequal
    complex squared lengths and links off the unit circle, with a carried
    covariance that is not block diagonal in any special way."""
    rng = np.random.default_rng(17)
    base = T.Spacetime.fromVertexTuples(3, [[0, 1, 2, 3]], 1.0, 0.0)
    for edge in base.getEdgeList().toVector():
        edge.setLength(np.sqrt(complex(8.0 + rng.uniform(-1, 1),
                                       0.1 * rng.normal())))
        edge.setPhase(complex(rng.uniform(0, 2 * np.pi), 0.1 * rng.normal()))
    support = cs.sheeted_support(base, sheets)
    declaration = bp.action_declaration(support.spacetime, 1.0, 1.0)
    order = 6 * sheets
    covariance = rng.normal(size=(order, order)) \
        + 1j * rng.normal(size=(order, order))
    declaration.covariance = list(covariance.reshape(-1))
    return support, cob.JointAction(support.spacetime, declaration)


def test_the_hessian_has_no_entry_between_two_sheets():
    """The sheets of a sheeted support are components of its complex. The
    matter term's Hessian couples no coordinate of one sheet to a coordinate
    of another, exactly, whatever the carried covariance is; and its block
    on the edges of one sheet is not zero."""
    support, action = _sheeted_action(3)
    edges = cs.edge_fields(support.spacetime)
    count = len(edges)
    hessian = np.asarray(action.action_hessian()).reshape(2 * count,
                                                          2 * count)
    sheet = [a // support.count for a, _, _, _ in edges]
    for row, column in itertools.product(range(2 * count), repeat=2):
        if sheet[row % count] != sheet[column % count]:
            assert hessian[row, column] == 0.0
    same = [(r, c) for r, c in itertools.product(range(2 * count), repeat=2)
            if sheet[r % count] == sheet[c % count]]
    assert max(abs(hessian[r, c]) for r, c in same) > 1e-3


def test_the_hessian_is_the_derivative_of_the_stationarity():
    """The Hessian of the action on a sheeted support against a central
    difference of its stationarity in one squared length and in one link, as
    an oracle."""
    support, action = _sheeted_action(2)
    edges = support.spacetime.getEdgeList().toVector()
    count = len(edges)
    hessian = np.asarray(action.action_hessian()).reshape(2 * count,
                                                          2 * count)

    def stationarity():
        return np.concatenate([np.asarray(action.length_stationarity()),
                               np.asarray(action.link_stationarity())])

    for index in (0, count - 1):
        edge = edges[index]
        length, phase = edge.getLength(), edge.getPhase()
        step = 1e-5
        squared = length * length
        edge.setLength(np.sqrt(squared + step))
        up = stationarity()
        edge.setLength(np.sqrt(squared - step))
        down = stationarity()
        edge.setLength(length)
        column = (up - down) / (2 * step)
        assert np.linalg.norm(column - hessian[:, index]) <= \
            1e-7 * np.linalg.norm(hessian[:, index])
        # the link's Maurer-Cartan increment delta: phi -> phi - i delta
        edge.setPhase(phase - 1j * step)
        up = stationarity()
        edge.setPhase(phase + 1j * step)
        down = stationarity()
        edge.setPhase(phase)
        column = (up - down) / (2 * step)
        assert np.linalg.norm(column - hessian[:, count + index]) <= \
            1e-7 * np.linalg.norm(hessian[:, count + index])
