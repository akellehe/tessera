# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The phase Hessian of the covariant operator (#1157): central differences of
the first phase derivative, symmetry, and the first- and second-order gauge
identities along a pure-gauge direction."""
import cmath
import itertools

import numpy as np
import pytest

from tessera import chainhodge as ch
from tessera import cobordism as cob

TETRAHEDRON = [[0, 1, 2, 3]]
TWO_TETRAHEDRA = [[0, 1, 2, 3], [1, 2, 3, 4]]


def _instance(cells, seed, complex_data):
    rng = np.random.default_rng(seed)
    K = cob.ChainComplex.fromTopCells(cells)
    n1 = K.numSimplices(1)
    if complex_data:
        s = [complex(1.0 + 0.1 * rng.normal(), 0.1 * rng.normal()) for _ in range(n1)]
        links = [complex(rng.normal(), rng.normal()) + 0.5 for _ in range(n1)]
    else:
        s = [complex(1.0 + 0.1 * rng.uniform(-1, 1)) for _ in range(n1)]
        links = [cmath.exp(1j * rng.uniform(-np.pi, np.pi)) for _ in range(n1)]
    return K, ch.ChainHodge(K, s), links


def _shifted(K, base, links, edge, delta):
    moved = list(links)
    moved[edge] *= cmath.exp(1j * delta)
    return ch.CovariantChainHodge(base, ch.Connection(K, moved), 7, False)


@pytest.mark.parametrize("cells", [TETRAHEDRON, TWO_TETRAHEDRA])
@pytest.mark.parametrize("complex_data", [False, True])
@pytest.mark.parametrize("k", [0, 1, 2])
def test_hessian_is_the_central_difference_of_the_first_derivative(cells, complex_data, k):
    K, base, links = _instance(cells, 3, complex_data)
    cov = ch.CovariantChainHodge(base, ch.Connection(K, links), 7, False)
    delta = 1e-5
    n1 = K.numSimplices(1)
    for a, b in itertools.product(range(n1), repeat=2):
        hessian = cov.covariantOperatorPhaseHessian(k, a, b)
        plus = _shifted(K, base, links, b, +delta).covariantOperatorPhaseDerivative(k, a)
        minus = _shifted(K, base, links, b, -delta).covariantOperatorPhaseDerivative(k, a)
        scale = max(1.0, np.abs(cov.covariantOperator(k)).max())
        assert np.abs(hessian - (plus - minus) / (2 * delta)).max() < 1e-6 * scale
        assert np.abs(hessian - cov.covariantOperatorPhaseHessian(k, b, a)).max() < 1e-10 * scale


@pytest.mark.parametrize("complex_data", [False, True])
@pytest.mark.parametrize("k", [0, 1, 2, 3])
def test_pure_gauge_directions_are_commutators(complex_data, k):
    """Under g_x = exp(i t chi_x) the operator moves by the similarity
    h(t) = exp(-i t X) h exp(i t X), X = diag(chi at each cell's base vertex),
    and every link phase by t (chi_y - chi_x). The first and second t-derivatives
    are i[h, X] and -[X, [X, h]]: the Ward identity and its second-order form."""
    K, base, links = _instance(TWO_TETRAHEDRA, 11, complex_data)
    cov = ch.CovariantChainHodge(base, ch.Connection(K, links), 7, False)
    rng = np.random.default_rng(5)
    chi = {int(v[0]): rng.normal() for v in K.kSimplexVertices(0)}
    direction = [chi[int(y)] - chi[int(x)] for x, y in K.kSimplexVertices(1)]
    X = np.diag([chi[int(cell[0])] for cell in K.kSimplexVertices(k)])
    h = cov.covariantOperator(k)
    n1 = K.numSimplices(1)

    first = sum(direction[a] * cov.covariantOperatorPhaseDerivative(k, a) for a in range(n1))
    assert np.abs(first - 1j * (h @ X - X @ h)).max() < 1e-10 * max(1.0, np.abs(h).max())

    second = sum(direction[a] * direction[b] * cov.covariantOperatorPhaseHessian(k, a, b)
                 for a in range(n1) for b in range(n1))
    inner = X @ h - h @ X
    assert np.abs(second + (X @ inner - inner @ X)).max() < 1e-9 * max(1.0, np.abs(h).max())


def test_refusals():
    K, base, links = _instance(TETRAHEDRON, 1, False)
    cov = ch.CovariantChainHodge(base, ch.Connection(K, links), 7, False)
    with pytest.raises(ValueError):
        cov.covariantOperatorPhaseHessian(4, 0, 0)
    with pytest.raises(ValueError):
        cov.covariantOperatorPhaseHessian(1, 0, 6)
