# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The length derivatives of the Whitney mass matrices and of the covariant
operator, formed for every edge from one pass over the top simplices.

`WhitneyMass.assembleDerivative` forms the local block of every top simplex,
with its derivative in every edge, and keeps the entries of one edge;
`WhitneyMass.assembleDerivatives` scatters the same blocks, in the same
order, to one matrix per edge. `CovariantChainHodge` keeps the dressed
derivatives of a degree once formed, so a sweep over the edges of
`covariantOperatorDerivative` and `dressedDerivative` makes one pass per
degree. Every matrix is the same, bit for bit, as the one formed for its
edge alone.
"""
import itertools

import numpy as np
import pytest

from tessera import chainhodge as ch
from tessera import cobordism as cob


def _instance(cells, seed):
    """A complex with complex squared lengths near a Euclidean geometry and
    links off the unit circle, and its covariant chain Hodge instance."""
    rng = np.random.default_rng(seed)
    complex_ = cob.ChainComplex.fromTopCells([list(c) for c in cells])
    count = complex_.numSimplices(1)
    squared = [complex(8.0 + rng.uniform(-0.5, 0.5), 0.05 * rng.normal())
               for _ in range(count)]
    links = [np.exp(0.2 * rng.normal() + 1j * rng.uniform(0.0, 2.0 * np.pi))
             for _ in range(count)]
    base = ch.ChainHodge(complex_, squared)
    covariant = ch.CovariantChainHodge(base, ch.Connection(complex_, links))
    return complex_, squared, covariant


FAN = [[0, 1, 2, 3], [0, 1, 3, 4]]
SUBDIVIDED = [[1, 2, 3, 4], [0, 2, 3, 4], [0, 1, 3, 4], [0, 1, 2, 4]]
BOUNDARY = [list(c) for c in itertools.combinations(range(5), 4)]


@pytest.mark.parametrize("cells", [FAN, SUBDIVIDED, BOUNDARY],
                         ids=["fan", "subdivided", "boundary"])
def test_one_pass_gives_every_edge_s_matrix_bit_for_bit(cells):
    """For every degree and every edge, the matrix of the one pass equals
    the matrix assembled for that edge alone, entry for entry."""
    complex_, squared, _ = _instance(cells, 3)
    for degree in range(complex_.dimension() + 1):
        every = ch.WhitneyMass.assembleDerivatives(complex_, squared, degree)
        assert len(every) == complex_.numSimplices(1)
        for edge, matrix in enumerate(every):
            alone = ch.WhitneyMass.assembleDerivative(complex_, squared,
                                                      degree, edge)
            assert matrix.shape == alone.shape
            np.testing.assert_array_equal(np.asarray(matrix.todense()),
                                          np.asarray(alone.todense()))


@pytest.mark.parametrize("cells", [FAN, SUBDIVIDED, BOUNDARY],
                         ids=["fan", "subdivided", "boundary"])
def test_the_operator_s_derivative_does_not_depend_on_what_was_asked_first(
        cells):
    """The derivative of h_1 and of the dressed metric in an edge is the
    same matrix, bit for bit, on an instance asked for that edge first and
    on an instance that was asked for every other edge before it: the kept
    derivatives are those of the edge alone."""
    complex_, _, swept = _instance(cells, 5)
    count = complex_.numSimplices(1)
    operator = [np.asarray(swept.covariantOperatorDerivative(1, edge))
                for edge in range(count)]
    metric = [np.asarray(swept.dressedDerivative(1, edge))
              for edge in range(count)]
    for edge in (0, count // 2, count - 1):
        _, _, fresh = _instance(cells, 5)
        np.testing.assert_array_equal(
            np.asarray(fresh.covariantOperatorDerivative(1, edge)),
            operator[edge])
        np.testing.assert_array_equal(
            np.asarray(fresh.dressedDerivative(1, edge)), metric[edge])
    # asked again, the same
    np.testing.assert_array_equal(
        np.asarray(swept.covariantOperatorDerivative(1, 0)), operator[0])


def test_the_operator_s_derivative_agrees_with_a_difference_of_the_operator():
    """dh_1/ds_e against a central difference of h_1 in that squared length,
    as an oracle: the disagreement falls with the square of the step."""
    cells = SUBDIVIDED
    complex_, squared, covariant = _instance(cells, 7)
    rng = np.random.default_rng(7)
    links = [np.exp(0.2 * rng.normal() + 1j * rng.uniform(0.0, 2.0 * np.pi))
             for _ in range(complex_.numSimplices(1))]

    def operator(lengths):
        base = ch.ChainHodge(complex_, lengths)
        return np.asarray(ch.CovariantChainHodge(
            base, ch.Connection(complex_, links)).covariantOperator(1))

    covariant = ch.CovariantChainHodge(ch.ChainHodge(complex_, squared),
                                       ch.Connection(complex_, links))
    for edge in (0, 4, complex_.numSimplices(1) - 1):
        exact = np.asarray(covariant.covariantOperatorDerivative(1, edge))
        errors = []
        for step in (1e-3, 1e-4):
            up, down = list(squared), list(squared)
            up[edge] += step
            down[edge] -= step
            difference = (operator(up) - operator(down)) / (2.0 * step)
            errors.append(np.linalg.norm(difference - exact)
                          / np.linalg.norm(exact))
        assert errors[0] < 1e-5 and errors[1] < 2e-2 * errors[0] + 1e-9
