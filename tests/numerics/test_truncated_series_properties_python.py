# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Identities of the truncated power-series arithmetic on random complex
series (#1372).

`tests/numerics/test_truncated_series_python.py` compares the arithmetic of
`tessera.numerics` with closed forms in exact rational arithmetic. The tests
here hold it to identities that need no closed form, on series whose
coefficients are random complex numbers of modulus about one, to the largest
declared order, ten:

* the product against the truncated product of polynomials
  (`numpy.polynomial`), and the composition with a polynomial against
  Horner's rule on those truncated products;
* the quotient, the square root, the logarithm and the inverse cosine as the
  inverses of the product, the square, the exponential and the cosine;
* the reversion of a random quadratic map of three variables against its
  defining identity F(x_0 + s(t)) = (1 - t) F(x_0) + O(t^11).

Tolerances. A coefficient of order at most ten of one recurrence is the
result of at most 110 rounded products and sums of terms whose size the
random coefficients keep below about 1e3 (ten-fold products of numbers of
modulus about one, with binomial counts), so a chain of three recurrences
is held to 1e-9 of the largest coefficient; the measured departures are
below 1e-12.
"""
import numpy as np
import pytest
from numpy.polynomial import polynomial as npp

from tessera import numerics as nm

TS = nm.TruncatedSeries

#: The largest order a series declares.
P = 10

#: The bound of a chain of recurrences, relative to the largest coefficient.
BOUND = 1e-9


def _series(rng, leading=None):
    """A series of order ten with random complex coefficients of modulus
    about one, its order-0 coefficient ``leading`` when given."""
    values = rng.normal(size=P + 1) + 1j * rng.normal(size=P + 1)
    if leading is not None:
        values[0] = leading
    return TS([complex(v) for v in values])


def _c(series):
    return np.array(series.coefficients, dtype=complex)


def _departure(actual, expected):
    """The largest difference of two coefficient arrays over the largest
    expected coefficient."""
    expected = np.asarray(expected, dtype=complex)
    return float(np.max(np.abs(np.asarray(actual) - expected))
                 / max(np.max(np.abs(expected)), 1.0))


@pytest.mark.parametrize("seed", range(5))
def test_the_product_is_the_truncated_product_of_polynomials(seed):
    rng = np.random.default_rng(seed)
    a, b = _series(rng), _series(rng)
    expected = npp.polymul(_c(a), _c(b))[:P + 1]
    assert _departure(_c(a * b), expected) < BOUND


@pytest.mark.parametrize("seed", range(5))
def test_the_composition_with_a_polynomial_is_horners_rule(seed):
    rng = np.random.default_rng(100 + seed)
    f = _series(rng)
    polynomial = [complex(v) for v in rng.normal(size=6)
                  + 1j * rng.normal(size=6)]
    expected = np.zeros(P + 1, dtype=complex)
    for coefficient in reversed(polynomial):
        expected = npp.polymul(expected, _c(f))[:P + 1]
        expected[0] += coefficient
    assert _departure(_c(nm.composePolynomial(polynomial, f)),
                      expected) < BOUND


@pytest.mark.parametrize("seed", range(5))
def test_the_quotient_and_the_roots_invert_the_product_and_the_powers(seed):
    """(a b) / b = a; sqrt(a)^2 = a on both roots of the order-0
    coefficient; exp(log(a)) = a on the principal logarithm and on the one
    continued once around the origin; cos(acos(a)) = a; sin^2 + cos^2 = 1."""
    rng = np.random.default_rng(200 + seed)
    a = _series(rng, leading=complex(1.5 + 0.2 * seed, 0.7))
    b = _series(rng, leading=complex(-0.8, 1.1 + 0.1 * seed))
    assert _departure(_c((a * b) / b), _c(a)) < BOUND
    root = nm.sqrt(a)
    assert _departure(_c(root * root), _c(a)) < BOUND
    other = nm.sqrt(a, -root.coefficients[0])
    assert _departure(_c(other), -_c(root)) < BOUND
    logarithm = nm.log(a)
    assert _departure(_c(nm.exp(logarithm)), _c(a)) < BOUND
    continued = nm.log(a, logarithm.coefficients[0] + 2j * np.pi)
    assert _departure(_c(nm.exp(continued)), _c(a)) < BOUND
    small = _series(rng, leading=complex(0.3, 0.1 * seed))
    assert _departure(_c(nm.cos(nm.acos(small))), _c(small)) < BOUND
    sine, cosine = nm.sinCos(a)
    one = np.zeros(P + 1, dtype=complex)
    one[0] = 1.0
    assert _departure(_c(sine * sine + cosine * cosine), one) < BOUND


@pytest.mark.parametrize("seed", range(5))
def test_the_reversion_of_a_quadratic_map_solves_its_defining_identity(seed):
    """F(x) = c + A x + (x^T B_i x)_i on three complex variables, with
    random c, A and B_i, about a random point x_0. The reversion to order
    ten with the solve s -> J^-1 s gives a path x_0 + s(t), and F on that
    path, evaluated in series arithmetic, has the coefficients F(x_0),
    -F(x_0) and zero to order ten: F(x_0 + s(t)) = (1 - t) F(x_0)."""
    rng = np.random.default_rng(300 + seed)

    def draw(*shape):
        return rng.normal(size=shape) + 1j * rng.normal(size=shape)

    constant, linear = draw(3), draw(3, 3) + 3.0 * np.eye(3)
    quadratic = 0.1 * draw(3, 3, 3)
    point = 0.3 * draw(3)

    def evaluate(x):
        out = []
        for i in range(3):
            value = complex(constant[i]) + 0.0 * x[0]
            for j in range(3):
                value = value + complex(linear[i, j]) * x[j]
                for k in range(3):
                    value = value + complex(quadratic[i, j, k]) * (x[j] * x[k])
            out.append(value)
        return out

    jacobian = linear + np.einsum("ijk,k->ij", quadratic, point) \
        + np.einsum("ikj,k->ij", quadratic, point)
    reversion = nm.revert(evaluate, point, P,
                          lambda right: np.linalg.solve(jacobian, right))
    coefficients = np.array(reversion.coefficients, dtype=complex)
    start = constant + linear @ point \
        + np.einsum("ijk,j,k->i", quadratic, point, point)
    assert _departure(reversion.residual, start) < BOUND
    path = [TS([complex(point[i])]
               + [complex(coefficients[k][i]) for k in range(1, P + 1)])
            for i in range(3)]
    on_path = np.array([_c(component) for component in evaluate(path)])
    expected = np.zeros((3, P + 1), dtype=complex)
    expected[:, 0] = start
    expected[:, 1] = -start
    assert _departure(on_path, expected) < BOUND
