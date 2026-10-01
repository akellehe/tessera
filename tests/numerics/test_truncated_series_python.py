# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Truncated power-series arithmetic against closed forms.

`tessera.numerics` carries the Taylor coefficients c_0, ..., c_p of a function
of one parameter t to a declared order p of at most ten, and propagates them
through arithmetic, elementary functions, dense matrices and the reversion of
a system of equations by coefficient recurrences. Every test here compares
those coefficients with numbers that are derived independently, in exact
integer or rational arithmetic (`fractions.Fraction`) wherever the closed form
is rational.

Tolerances. `EPS` is the rounding unit 2^-52 of a double. Where every
intermediate value of a computation is an integer or a dyadic rational that a
double holds exactly, the comparison is exact equality. Elsewhere a
coefficient of order at most ten of one recurrence is the result of at most
2 (1 + 2 + ... + 10) = 110 rounded products and sums, each with a relative
rounding error of at most EPS, so its error is at most 110 EPS times the size
of the largest term; `_rounding(recurrences, scale)` is that bound, with 110
rounded up to 128, for a chain of recurrences. Each test states the scale it
uses.
"""
import cmath
import itertools
import math
from fractions import Fraction

import numpy as np
import pytest

import tessera
from tessera import numerics as nm

TS = nm.TruncatedSeries
TSM = nm.TruncatedSeriesMatrix

#: The rounding unit of a double, 2^-52.
EPS = float(np.finfo(float).eps)

#: The largest order a series declares.
P = 10


def _rounding(recurrences=1, scale=1.0):
    """The rounding bound of a chain of `recurrences` coefficient recurrences
    whose terms are no larger than `scale`: 128 EPS per recurrence (at most
    110 rounded operations feed one coefficient of order ten)."""
    return 128.0 * recurrences * EPS * scale


def _binomial(alpha, k):
    """The generalized binomial coefficient C(alpha, k) =
    alpha (alpha - 1) ... (alpha - k + 1) / k!, exactly, for a rational
    alpha: the coefficient of t^k of (1 + t)^alpha."""
    out = Fraction(1)
    for j in range(k):
        out = out * (alpha - j) / (j + 1)
    return out


def _t(order=P):
    """The series of the parameter itself, f(t) = t."""
    return TS.variable(order, 0.0)


def _c(series):
    return np.array(series.coefficients)


def _floats(fractions):
    return np.array([float(x) for x in fractions], dtype=complex)


def _max_error(actual, expected):
    return float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))


def _stack(matrix):
    return np.array(matrix.coefficients)


# ---------------------------------------------------------------------------
# Construction and the declared order
# ---------------------------------------------------------------------------


class TestDeclaredOrder:
    def test_the_namespace_is_a_module_of_the_package(self):
        """`tessera.numerics` is importable as a module and is the same object
        as the attribute of the package, like every namespace that mirrors a
        C++ namespace."""
        import importlib
        import sys

        module = importlib.import_module("tessera.numerics")
        assert module is nm
        assert sys.modules["tessera.numerics"] is tessera.numerics

    @pytest.mark.parametrize("order", range(P + 1))
    def test_every_order_from_zero_to_ten_is_declared_at_construction(self, order):
        """A series of order p carries p + 1 coefficients; the zero series,
        the constant series and the variable x_0 + d t are read back
        exactly."""
        zero = TS(order)
        assert zero.order == order
        assert zero.coefficients == [0.0] * (order + 1)
        constant = TS.constant(order, 2.5 - 1j)
        assert constant.coefficients == [2.5 - 1j] + [0.0] * order
        variable = TS.variable(order, 3.0, 0.5j)
        expected = [3.0] + ([0.5j] + [0.0] * (order - 1) if order else [])
        assert variable.coefficients == expected
        listed = TS([complex(k, -k) for k in range(order + 1)])
        assert listed.order == order
        assert listed.coefficient(order) == complex(order, -order)

    def test_an_order_above_ten_is_rejected(self):
        """Ten is the declared maximum: order eleven, a negative order, a list
        of twelve coefficients and an empty list have no series."""
        assert nm.MAXIMUM_ORDER == P
        with pytest.raises(ValueError, match="order 11"):
            TS(11)
        with pytest.raises(ValueError, match="order -1"):
            TS(-1)
        with pytest.raises(ValueError, match="order 11"):
            TS([1.0] * 12)
        with pytest.raises(ValueError, match="empty"):
            TS([])
        with pytest.raises(ValueError, match="order 11"):
            TS.constant(11, 1.0)
        with pytest.raises(ValueError, match="order 11"):
            TSM(11, 2, 2)
        with pytest.raises(ValueError, match="order 11"):
            TSM([np.eye(2)] * 12)

    def test_a_coefficient_outside_the_order_is_an_index_error(self):
        f = TS([1.0, 2.0, 3.0])
        assert f.coefficient(2) == 3.0
        with pytest.raises(IndexError, match="index 3"):
            f.coefficient(3)
        with pytest.raises(IndexError, match="index -1"):
            f.coefficient(-1)

    def test_two_series_enter_a_binary_operation_only_at_one_order(self):
        """A series of order 3 and one of order 4 are not combined: the
        result would be known to order 3 only, and the mismatch is reported
        instead of being resolved silently."""
        f = TS([1.0, 2.0, 3.0, 4.0])
        g = TS([1.0, 2.0, 3.0, 4.0, 5.0])
        for operation in (
            lambda: f + g,
            lambda: f - g,
            lambda: f * g,
            lambda: f / g,
            lambda: nm.atan2(f, g),
        ):
            with pytest.raises(ValueError, match="orders 3 and 4"):
                operation()
        # the explicit cut is the way to combine them
        assert (f + g.truncated(3)).coefficients == [2.0, 4.0, 6.0, 8.0]


# ---------------------------------------------------------------------------
# Arithmetic
# ---------------------------------------------------------------------------


class TestArithmetic:
    def test_sum_difference_and_cauchy_product_of_two_polynomials(self):
        """f = 1 + 2t + 3t^2 and g = 2 - t + t^3, as series of order ten.

        Multiplying out, f g = 2 + 3t + 4t^2 - 2t^3 + 2t^4 + 3t^5: the
        coefficient of t^k is sum_j f_j g_(k-j), e.g. t^3: 1*1 + 2*0 + 3*(-1)
        = -2. The inputs and every partial sum are small integers, which a
        double holds exactly, so the comparison is exact equality."""
        f = TS([1, 2, 3] + [0] * 8)
        g = TS([2, -1, 0, 1] + [0] * 7)
        assert (f + g).coefficients == [3, 1, 3, 1] + [0] * 7
        assert (f - g).coefficients == [-1, 3, 3, -1] + [0] * 7
        assert (f * g).coefficients == [2, 3, 4, -2, 2, 3] + [0] * 5
        assert (-f).coefficients == [-1, -2, -3] + [0] * 8

    def test_the_product_leaves_out_every_power_above_the_order(self):
        """At order 3 the product of f = 1 + 2t + 3t^2 and g = 2 - t + t^3 is
        2 + 3t + 4t^2 - 2t^3: the terms 2t^4 + 3t^5 of the full product lie
        above the declared order and are not carried."""
        f = TS([1, 2, 3, 0])
        g = TS([2, -1, 0, 1])
        assert (f * g).coefficients == [2, 3, 4, -2]

    def test_a_number_enters_as_the_constant_series(self):
        """With f = 1 + 2t + 3t^2: 5 + f, f + 5, f - 5, 5 - f, 3 f, f 3 and
        f / 2 act on the coefficients as the constant series would; the
        results are small dyadic rationals, exact in a double."""
        f = TS([1, 2, 3])
        assert (5 + f).coefficients == [6, 2, 3]
        assert (f + 5).coefficients == [6, 2, 3]
        assert (f - 5).coefficients == [-4, 2, 3]
        assert (5 - f).coefficients == [4, -2, -3]
        assert (3 * f).coefficients == [3, 6, 9]
        assert (f * 3).coefficients == [3, 6, 9]
        assert (f / 2).coefficients == [0.5, 1, 1.5]
        assert (f * 1j).coefficients == [1j, 2j, 3j]

    def test_the_quotient_recurrence_gives_the_geometric_series(self):
        """1 / (1 - 2t) = sum_k 2^k t^k. The recurrence
        g_0 h_k = f_k - sum_{j>=1} g_j h_(k-j) with g = 1 - 2t and f = 1
        reads h_k = 2 h_(k-1), so h_k = 2^k: powers of two, exact."""
        g = TS([1, -2] + [0] * 9)
        expected = [2.0**k for k in range(P + 1)]
        assert (1 / g).coefficients == expected
        assert (TS.constant(P, 1.0) / g).coefficients == expected

    def test_the_quotient_undoes_the_product(self):
        """(f g) / g = f for f = 1 + 2t + 3t^2 and g = 2 - t + t^3. With
        g_0 = 2 each step of the recurrence divides an integer multiple of a
        power of two by two, so every intermediate value is a dyadic rational
        with a short mantissa and the result is exact."""
        f = TS([1, 2, 3] + [0] * 8)
        g = TS([2, -1, 0, 1] + [0] * 7)
        assert ((f * g) / g).coefficients == f.coefficients

    def test_integer_powers_are_binomial_coefficients(self):
        """(1 + t)^7 has the coefficients C(7, k) = 1, 7, 21, 35, 35, 21, 7, 1
        and (1 + t)^-3 the coefficients C(-3, k) = (-1)^k (k+1)(k+2)/2 =
        1, -3, 6, -10, ...; both are integers, formed exactly. The power zero
        is the constant series 1, also of a series without constant term."""
        f = TS([1, 1] + [0] * 9)
        assert (f**7).coefficients == [math.comb(7, k) for k in range(P + 1)]
        assert nm.pow(f, 7).coefficients == (f**7).coefficients
        expected = [float(_binomial(Fraction(-3), k)) for k in range(P + 1)]
        assert expected[:4] == [1, -3, 6, -10]
        assert (f**-3).coefficients == expected
        assert (f**0).coefficients == [1] + [0] * P
        t = _t()
        assert (t**0).coefficients == [1] + [0] * P
        assert (t**3).coefficients == [0, 0, 0, 1] + [0] * 7

    def test_derivative_integral_truncation_and_evaluation(self):
        """f = sum_k (k + 1) t^k at order ten.

        The derivative has the coefficients (k + 1)(k + 2) and order nine,
        the order to which a series of order ten determines it. The integral
        with constant 7 has the coefficients 7, then c_(k-1)/k = 1 for every
        k >= 1; an integral raises the order by one, up to ten. Integrating
        the derivative from f_0 returns f. At t = 1/2 the value is
        sum_{k=0..10} (k + 1) / 2^k; from sum_{k=0..n} (k + 1) x^k =
        (1 - (n + 2) x^(n+1) + (n + 1) x^(n+2)) / (1 - x)^2 at x = 1/2 and
        n = 10 it is 4 (1 - 12/2048 + 11/4096) = 4 - 13/1024 = 4083/1024.
        All of these are dyadic rationals with short mantissas, so the
        comparisons are exact."""
        f = TS([k + 1 for k in range(P + 1)])
        derivative = f.derivative()
        assert derivative.order == P - 1
        assert derivative.coefficients == [(k + 1) * (k + 2) for k in range(P)]
        integral = f.integral(7.0)
        assert integral.order == P
        assert integral.coefficients == [7.0] + [1.0] * P
        assert derivative.integral(f.coefficient(0)).coefficients == f.coefficients
        short = TS([1.0, 2.0, 3.0])
        assert short.integral().order == 3
        assert short.integral().coefficients == [0.0, 1.0, 1.0, 1.0]
        assert f.truncated(2).coefficients == [1, 2, 3]
        value = sum(Fraction(k + 1, 2**k) for k in range(P + 1))
        assert value == Fraction(4083, 1024)
        assert f.evaluate(0.5) == float(value)
        assert f.evaluate(0.0) == 1.0
        with pytest.raises(ValueError, match="derivative"):
            TS([3.0]).derivative()
        with pytest.raises(ValueError, match="truncated"):
            f.truncated(11)

    def test_composition_of_a_polynomial_with_a_series(self):
        """P(x) = 1 - 3x + 2x^3 composed with f.

        With f = 1 + t: P(1 + t) = 1 - 3(1 + t) + 2(1 + 3t + 3t^2 + t^3) =
        3t + 6t^2 + 2t^3, in integers, exactly.

        With f = sin t: sin^3 t = (3 sin t - sin 3t) / 4, so
        P(sin t) = 1 - (3/2) sin t - (1/2) sin 3t, whose coefficient of
        t^k is -(-1)^((k-1)/2) (3 + 3^k) / (2 k!) for odd k and 0 for even
        k >= 2. Three chained recurrences at most (sine, two products in
        Horner's rule each summing terms of size at most 1), terms no larger
        than the largest coefficient, 3^9 (3 + 1) / (2 * 9!) < 1."""
        polynomial = [1, -3, 0, 2]
        f = TS([1, 1] + [0] * 9)
        assert nm.composePolynomial(polynomial, f).coefficients == [0, 3, 6, 2] + [0] * 7
        assert nm.composePolynomial([], f).coefficients == [0] * (P + 1)
        expected = [Fraction(1)]
        for k in range(1, P + 1):
            if k % 2 == 0:
                expected.append(Fraction(0))
            else:
                sign = -1 if ((k - 1) // 2) % 2 == 0 else 1
                expected.append(Fraction(sign * (3 + 3**k), 2 * math.factorial(k)))
        actual = nm.composePolynomial(polynomial, nm.sin(_t()))
        assert _max_error(_c(actual), _floats(expected)) <= _rounding(3)


# ---------------------------------------------------------------------------
# Elementary functions against known Taylor coefficients
# ---------------------------------------------------------------------------


class TestElementaryFunctions:
    def test_exp_of_sin_to_order_ten(self):
        """The coefficients of g(t) = exp(sin t) are a_k / k! with
        a = 1, 1, 1, 0, -3, -8, -3, 56, 217, 64, -2951.

        Derivation: g' = cos(t) g, and cos has the derivatives
        cos^(j)(0) = (-1)^(j/2) for even j and 0 for odd j, so by Leibniz's
        rule a_(k+1) = sum over even j <= k of C(k, j) (-1)^(j/2) a_(k-j),
        a_0 = 1. The integers are recomputed here from that rule and compared
        with the list. Two chained recurrences (sine, exponential) with terms
        no larger than 1."""
        a = [1]
        for k in range(P):
            a.append(sum(math.comb(k, j) * (-1) ** (j // 2) * a[k - j] for j in range(0, k + 1, 2)))
        assert a == [1, 1, 1, 0, -3, -8, -3, 56, 217, 64, -2951]
        expected = [Fraction(a[k], math.factorial(k)) for k in range(P + 1)]
        actual = nm.exp(nm.sin(_t()))
        assert _max_error(_c(actual), _floats(expected)) <= _rounding(2)

    def test_exp_at_a_complex_point(self):
        """exp(a + b t) = e^a sum_k (b t)^k / k!, at a = 0.3 - 0.4i and
        b = 1.5 + 0.5i. One recurrence; the largest coefficient is
        |e^a| |b| = 2.13 (at k = 1; |b|^k / k! falls from k = 2 on), so the
        scale is taken as 2.4."""
        a, b = 0.3 - 0.4j, 1.5 + 0.5j
        actual = nm.exp(TS.variable(P, a, b))
        expected = [cmath.exp(a) * b**k / math.factorial(k) for k in range(P + 1)]
        assert _max_error(_c(actual), expected) <= _rounding(1, 2.4)

    def test_sqrt_of_one_plus_t_on_both_branches(self):
        """sqrt(1 + t) = sum_k C(1/2, k) t^k = 1 + t/2 - t^2/8 + t^3/16 - ...,
        the binomial series. The coefficients are dyadic rationals
        (C(1/2, k) has a power of two as its denominator) and the recurrence
        2 g_0 g_k = f_k - sum g_j g_(k-j) with g_0 = 1 forms them from
        products and halvings of dyadic rationals, exactly. The branch
        g_0 = -1 gives the negated series: the other root of 1 + t."""
        expected = [float(_binomial(Fraction(1, 2), k)) for k in range(P + 1)]
        assert expected[:4] == [1.0, 0.5, -0.125, 0.0625]
        f = TS([1, 1] + [0] * 9)
        assert nm.sqrt(f).coefficients == expected
        assert nm.sqrt(f, 1.0).coefficients == expected
        assert nm.sqrt(f, -1.0).coefficients == [-x for x in expected]

    def test_sqrt_continues_the_root_the_caller_declares(self):
        """f = -4 + t: the principal root of f_0 is 2i, and
        sqrt(-4 + t) = 2i sqrt(1 - t/4) = 2i sum_k C(1/2, k) (-1/4)^k t^k.
        Declaring the branch -2i (the root a caller reaches by continuing
        around the branch point) gives the negated series. The coefficients
        are dyadic rationals times i, formed exactly."""
        expected = [2j * float(_binomial(Fraction(1, 2), k) * Fraction(-1, 4) ** k) for k in range(P + 1)]
        f = TS([-4, 1] + [0] * 9)
        assert nm.sqrt(f).coefficients == expected
        assert nm.sqrt(f, -2j).coefficients == [-x for x in expected]
        product = nm.sqrt(f, -2j) * nm.sqrt(f, -2j)
        assert product.coefficients == f.coefficients

    def test_log_of_one_plus_t_and_a_continued_branch(self):
        """log(1 + t) = t - t^2/2 + t^3/3 - ...: the coefficient of t^k is
        (-1)^(k+1) / k. One recurrence with terms no larger than 1. A branch
        value 2 pi i for the order-0 coefficient (the logarithm after one turn
        about the origin) moves the constant term only: the derivative
        f'/f does not depend on the branch."""
        expected = [Fraction(0)] + [Fraction((-1) ** (k + 1), k) for k in range(1, P + 1)]
        f = TS([1, 1] + [0] * 9)
        principal = nm.log(f)
        assert _max_error(_c(principal), _floats(expected)) <= _rounding(1)
        continued = nm.log(f, 2j * math.pi)
        assert continued.coefficient(0) == 2j * math.pi
        assert continued.coefficients[1:] == principal.coefficients[1:]

    def test_log_inverts_exp_for_a_complex_series(self):
        """log(exp(f)) = f for f with f_0 = 0.2 + 0.3i inside the principal
        strip |Im| < pi, and coefficients of size at most 0.8. Two chained
        recurrences; the scale is the largest coefficient of exp(f), which is
        its order-0 coefficient e^0.2 = 1.22."""
        f = TS([0.2 + 0.3j, 0.8, -0.5j, 0.3, 0.1 + 0.2j, -0.4, 0.2, 0.1j, -0.3, 0.25, 0.1])
        g = nm.exp(f)
        scale = float(np.max(np.abs(_c(g))))
        assert 1.2 < scale < 1.3
        assert _max_error(_c(nm.log(g)), _c(f)) <= _rounding(2, scale)

    def test_sin_and_cos_at_zero_and_at_a_point(self):
        """sin t and cos t have the coefficients (-1)^m / (2m+1)! at odd
        orders and (-1)^m / (2m)! at even orders. At a point a the k-th
        derivative of the sine is sin(a + k pi / 2), so sin(a + t) has the
        coefficients sin(a + k pi/2) / k! and cos(a + t) the coefficients
        cos(a + k pi/2) / k!; a = 0.7. One recurrence, terms no larger than
        1; the reference values carry one more rounding each (the library
        sine of a + k pi / 2), which the bound of 128 EPS covers."""
        sine, cosine = nm.sinCos(_t())
        expected_sine = [Fraction(0) if k % 2 == 0 else Fraction((-1) ** (k // 2), math.factorial(k)) for k in range(P + 1)]
        expected_cosine = [Fraction((-1) ** (k // 2), math.factorial(k)) if k % 2 == 0 else Fraction(0) for k in range(P + 1)]
        assert _max_error(_c(sine), _floats(expected_sine)) <= _rounding(1)
        assert _max_error(_c(cosine), _floats(expected_cosine)) <= _rounding(1)
        assert nm.sin(_t()).coefficients == sine.coefficients
        assert nm.cos(_t()).coefficients == cosine.coefficients
        a = 0.7
        shifted_sine, shifted_cosine = nm.sinCos(TS.variable(P, a))
        expected_sine = [math.sin(a + k * math.pi / 2) / math.factorial(k) for k in range(P + 1)]
        expected_cosine = [math.cos(a + k * math.pi / 2) / math.factorial(k) for k in range(P + 1)]
        assert _max_error(_c(shifted_sine), expected_sine) <= _rounding(1)
        assert _max_error(_c(shifted_cosine), expected_cosine) <= _rounding(1)

    def test_sin_squared_plus_cos_squared_is_one_for_a_complex_series(self):
        """sin^2 f + cos^2 f = 1 as an identity between series, for a complex
        f: the constant term is 1 and every higher coefficient vanishes. Two
        chained recurrences (the pair, then the products); the scale is the
        square of the largest coefficient of the pair (1.12 for this f),
        since each product multiplies two of them."""
        f = TS([0.4 - 0.6j, 0.9, 0.3j, -0.5, 0.2, 0.1 - 0.1j, 0.3, -0.2j, 0.15, 0.1, -0.05])
        sine, cosine = nm.sinCos(f)
        scale = float(max(np.max(np.abs(_c(sine))), np.max(np.abs(_c(cosine))))) ** 2
        identity = sine * sine + cosine * cosine
        assert _max_error(_c(identity), [1.0] + [0.0] * P) <= _rounding(2, scale)

    @pytest.mark.parametrize(
        "x_sign, y_sign, constant, slope",
        [
            (1, 1, math.pi / 4, 1),
            (-1, 1, 3 * math.pi / 4, -1),
            (-1, -1, -3 * math.pi / 4, 1),
            (1, -1, -math.pi / 4, -1),
        ],
    )
    def test_atan2_along_a_line_in_each_quadrant(self, x_sign, y_sign, constant, slope):
        """The curve (x, y) = (1 - t, 1 + t) and its reflections.

        tan(pi/4 + a) = (1 + tan a) / (1 - tan a), so with a = arctan t the
        angle of (1 - t, 1 + t) is pi/4 + arctan t = pi/4 + t - t^3/3 +
        t^5/5 - t^7/7 + t^9/9. Reflecting x gives pi minus that angle,
        3 pi/4 - arctan t; reflecting both gives the angle less pi,
        -3 pi/4 + arctan t; reflecting y gives -pi/4 - arctan t. The order-0
        value is the angle of the quadrant in (-pi, pi]. Two logarithm
        recurrences with terms no larger than 1 (the scale 1/|x_0 + i y_0|^k
        is at most 1)."""
        t = _t()
        x = x_sign * (1 - t)
        y = y_sign * (1 + t)
        arctangent = [Fraction(0)] + [
            Fraction((-1) ** ((k - 1) // 2), k) if k % 2 else Fraction(0) for k in range(1, P + 1)
        ]
        expected = slope * _floats(arctangent)
        expected[0] = constant
        actual = nm.atan2(y, x)
        assert _max_error(_c(actual), expected) <= _rounding(2)
        assert np.all(np.imag(_c(actual)) == 0.0)

    def test_atan2_reads_back_the_angle_of_a_parametrized_curve(self):
        """The curve (x, y) = r(t) (cos phi(t), sin phi(t)) with
        r(t) = exp(t) and phi(t) = 2.5 + 0.3 t - 0.2 t^2 has the angle
        phi(t) exactly, since 2.5 lies inside (-pi, pi): the coefficients are
        2.5, 0.3, -0.2 and zeros. Four chained recurrences (sine and cosine,
        exponential, product, logarithms); the scale is the largest
        coefficient of x and y, below 3. A declared branch 2.5 + 2 pi (the
        same point reached after one more turn) moves the constant only."""
        t = _t()
        phi = 2.5 + 0.3 * t - 0.2 * t * t
        radius = nm.exp(t)
        sine, cosine = nm.sinCos(phi)
        x, y = radius * cosine, radius * sine
        scale = float(max(np.max(np.abs(_c(x))), np.max(np.abs(_c(y)))))
        assert scale < 3.0
        expected = [2.5, 0.3, -0.2] + [0.0] * 8
        actual = nm.atan2(y, x)
        assert _max_error(_c(actual), expected) <= _rounding(4, 3.0)
        continued = nm.atan2(y, x, 2.5 + 2 * math.pi)
        assert continued.coefficient(0) == 2.5 + 2 * math.pi
        assert continued.coefficients[1:] == actual.coefficients[1:]

    def test_atan2_of_complex_series_satisfies_its_defining_relations(self):
        """For complex x and y the angle theta = atan2(y, x) is defined by
        cos(theta) = x / rho and sin(theta) = y / rho with rho a square root
        of x^2 + y^2; hence x sin(theta) - y cos(theta) = 0 and
        (x cos(theta) + y sin(theta))^2 = x^2 + y^2 as identities between
        series. Four chained recurrences; the scale is the largest
        coefficient entering the two identities."""
        x = TS([0.9 + 0.2j, 0.3, -0.4j, 0.2, 0.1, -0.3, 0.2j, 0.1, 0.05, -0.1, 0.2])
        y = TS([0.4 - 0.5j, -0.6, 0.2, 0.3j, -0.1, 0.2, 0.1, -0.2j, 0.1, 0.05, 0.1])
        theta = nm.atan2(y, x)
        sine, cosine = nm.sinCos(theta)
        first = x * sine - y * cosine
        along = x * cosine + y * sine
        second = along * along - (x * x + y * y)
        scale = float(
            max(
                np.max(np.abs(_c(x * sine))),
                np.max(np.abs(_c(y * cosine))),
                np.max(np.abs(_c(along * along))),
            )
        )
        assert _max_error(_c(first), 0.0) <= _rounding(4, scale)
        assert _max_error(_c(second), 0.0) <= _rounding(4, scale)
        # the order-0 value is the angle whose cosine and sine are x_0 / rho_0
        # and y_0 / rho_0 with the principal root rho_0
        rho = cmath.sqrt(x.coefficient(0) ** 2 + y.coefficient(0) ** 2)
        assert abs(cmath.cos(theta.coefficient(0)) - x.coefficient(0) / rho) <= 8 * EPS
        assert abs(cmath.sin(theta.coefficient(0)) - y.coefficient(0) / rho) <= 8 * EPS

    def test_acos_of_half_t_on_three_sheets(self):
        """arccos(u) = pi/2 - sum_n C(2n, n) u^(2n+1) / (4^n (2n+1)), the
        integral of -1/sqrt(1 - u^2) = -sum_n C(2n, n) u^(2n) / 4^n. With
        u = t/2 the coefficient of t^(2n+1) is
        -C(2n, n) / (4^n (2n+1) 2^(2n+1)): -1/2, -1/48, -3/1280, ...

        Three recurrences (product, root, quotient) with terms no larger than
        1. On the sheet theta = 2 pi - Arccos (declared by the order-0 value
        3 pi / 2, whose sine is -1) every higher coefficient changes sign; on
        the sheet theta = 2 pi + Arccos (order-0 value 5 pi / 2, sine +1) the
        higher coefficients are those of the principal sheet."""
        expected = [Fraction(0)] * (P + 1)
        for n in range(5):
            expected[2 * n + 1] = -Fraction(math.comb(2 * n, n), 4**n * (2 * n + 1) * 2 ** (2 * n + 1))
        assert expected[1:6:2] == [Fraction(-1, 2), Fraction(-1, 48), Fraction(-3, 1280)]
        expected = _floats(expected)
        f = TS([0, 0.5] + [0] * 9)
        principal = _c(nm.acos(f))
        assert principal[0] == math.pi / 2
        assert _max_error(principal[1:], expected[1:]) <= _rounding(3)
        reflected = _c(nm.acos(f, 3 * math.pi / 2))
        assert reflected[0] == 3 * math.pi / 2
        assert _max_error(reflected[1:], -expected[1:]) <= _rounding(3)
        shifted = _c(nm.acos(f, 5 * math.pi / 2))
        assert shifted[0] == 5 * math.pi / 2
        assert _max_error(shifted[1:], expected[1:]) <= _rounding(3)

    def test_cos_inverts_acos_for_a_complex_series(self):
        """cos(arccos f) = f for a complex f with f_0 = 0.3 + 0.2i. Four
        chained recurrences (product, root, quotient, cosine); the scale is
        the square of the largest coefficient of the angle (1.29 for this
        f), since the cosine recurrence multiplies two such coefficients."""
        f = TS([0.3 + 0.2j, 0.5, -0.2j, 0.3, 0.1, -0.2, 0.15j, 0.1, -0.1, 0.05, 0.2])
        theta = nm.acos(f)
        scale = float(np.max(np.abs(_c(theta)))) ** 2
        assert _max_error(_c(nm.cos(theta)), _c(f)) <= _rounding(4, scale)


# ---------------------------------------------------------------------------
# Arguments at which an operation has no value
# ---------------------------------------------------------------------------


class TestNoValue:
    def test_an_exactly_zero_order_zero_coefficient_has_no_quotient_root_or_logarithm(self):
        """Division by a series, the square root and the logarithm have no
        power series when the order-0 coefficient is exactly zero; each
        refusal names its operation. The same holds for the angle of the
        origin, the inverse cosine at +1 and -1, a negative power, and the
        division by the number zero."""
        t = _t()
        one = TS.constant(P, 1.0)
        with pytest.raises(ValueError, match="division"):
            one / t
        with pytest.raises(ValueError, match="division"):
            1 / t
        with pytest.raises(ValueError, match="division"):
            one / 0.0
        with pytest.raises(ValueError, match="sqrt"):
            nm.sqrt(t)
        with pytest.raises(ValueError, match="sqrt"):
            nm.sqrt(one, 0.0)
        with pytest.raises(ValueError, match="log"):
            nm.log(t)
        with pytest.raises(ValueError, match="atan2"):
            nm.atan2(t, t)
        with pytest.raises(ValueError, match="acos"):
            nm.acos(1 + t)
        with pytest.raises(ValueError, match="acos"):
            nm.acos(-1 + t)
        with pytest.raises(ValueError, match="pow"):
            t**-1

    def test_a_small_nonzero_order_zero_coefficient_is_divided_by_as_it_stands(self):
        """With e = 2^-100 no operation refuses.

        1 / (e + t) = sum_k (-1)^k t^k / e^(k+1): the coefficients
        2^100, -2^200, 2^300, -2^400 are powers of two, exact. The logarithm
        of e + t has the coefficients log e, 1/e, -1/(2 e^2), 1/(3 e^3), of
        which the first three higher ones are exact powers of two or their
        thirds (one rounding). The square root of e + t starts
        2^-50, 2^49, -2^147: sqrt(e) (1 + t/(2e) - t^2/(8 e^2))."""
        e = 2.0**-100
        f = TS([e, 1, 0, 0])
        assert (1 / f).coefficients == [2.0**100, -(2.0**200), 2.0**300, -(2.0**400)]
        logarithm = nm.log(f).coefficients
        assert logarithm[0] == math.log(e)
        assert logarithm[1] == 2.0**100
        assert logarithm[2] == -(2.0**199)
        assert abs(logarithm[3] - 2.0**300 / 3) <= 2 * EPS * 2.0**300 / 3
        assert nm.sqrt(f).coefficients[:3] == [2.0**-50, 2.0**49, -(2.0**147)]


# ---------------------------------------------------------------------------
# Dense matrices of series
# ---------------------------------------------------------------------------


def _integer_series_matrix(rng, n, degree, order=P, bound=3):
    """Coefficient matrices with integer entries in [-bound, bound] up to
    `degree`, zero above, as an integer array of shape (order + 1, n, n)."""
    out = np.zeros((order + 1, n, n), dtype=np.int64)
    out[: degree + 1] = rng.integers(-bound, bound + 1, size=(degree + 1, n, n))
    return out


def _exact_determinant(coefficients, order):
    """The determinant of a polynomial matrix with integer coefficients, by
    the Leibniz expansion over permutations in exact integer arithmetic, cut
    after t^order."""
    n = coefficients.shape[1]
    total = [0] * (order + 1)
    for permutation in itertools.permutations(range(n)):
        inversions = sum(1 for a in range(n) for b in range(a + 1, n) if permutation[a] > permutation[b])
        term = [1] + [0] * order
        for row in range(n):
            entry = [int(coefficients[k][row][permutation[row]]) for k in range(coefficients.shape[0])]
            term = [sum(term[j] * entry[k - j] for j in range(k + 1)) for k in range(order + 1)]
        sign = -1 if inversions % 2 else 1
        total = [a + sign * b for a, b in zip(total, term)]
    return total


class TestMatrixSeries:
    def test_the_product_is_the_cauchy_product_with_matrix_coefficients(self):
        """A = A_0 + A_1 t and B = B_0 + B_1 t + B_2 t^2 with integer
        entries: A B = A_0 B_0 + (A_0 B_1 + A_1 B_0) t + (A_0 B_2 + A_1 B_1)
        t^2 + A_1 B_2 t^3, and nothing above. Integer arithmetic, exact. The
        sum and the difference act coefficient by coefficient."""
        rng = np.random.default_rng(1334)
        a = _integer_series_matrix(rng, 3, 1)
        b = _integer_series_matrix(rng, 3, 2)
        A, B = TSM(list(a.astype(complex))), TSM(list(b.astype(complex)))
        assert (A.order, A.rows, A.cols) == (P, 3, 3)
        expected = np.zeros((P + 1, 3, 3), dtype=np.int64)
        expected[0] = a[0] @ b[0]
        expected[1] = a[0] @ b[1] + a[1] @ b[0]
        expected[2] = a[0] @ b[2] + a[1] @ b[1]
        expected[3] = a[1] @ b[2]
        assert np.array_equal(_stack(A @ B), expected)
        assert np.array_equal(_stack(A + B), a + b)
        assert np.array_equal(_stack(A - B), a - b)

    def test_rectangular_products_entries_and_evaluation(self):
        """A 2 by 3 series times a 3 by 1 series is 2 by 1; entries are read
        and written as series; the value at t = 1/2 of an integer matrix
        series of degree 2 is A_0 + A_1 / 2 + A_2 / 4, exactly."""
        rng = np.random.default_rng(7)
        a = rng.integers(-3, 4, size=(3, 2, 3))
        b = rng.integers(-3, 4, size=(3, 3, 1))
        A, B = TSM(list(a.astype(complex))), TSM(list(b.astype(complex)))
        C = A @ B
        assert (C.rows, C.cols, C.order) == (2, 1, 2)
        assert np.array_equal(C.coefficients[1], a[0] @ b[1] + a[1] @ b[0])
        assert A.entry(1, 2).coefficients == [a[k][1][2] for k in range(3)]
        A.setEntry(1, 2, TS([7.0, 8.0, 9.0]))
        assert A.entry(1, 2).coefficients == [7.0, 8.0, 9.0]
        assert np.array_equal(B.evaluate(0.5), b[0] + b[1] / 2 + b[2] / 4)
        with pytest.raises(ValueError, match="product"):
            B @ A
        with pytest.raises(ValueError, match="one shape"):
            TSM([np.eye(2), np.eye(3)])
        with pytest.raises(IndexError, match="entry"):
            A.entry(2, 0)
        with pytest.raises(ValueError, match="setEntry"):
            A.setEntry(0, 0, TS([1.0]))

    def test_the_inverse_of_identity_minus_t_n_is_the_neumann_series(self):
        """(I - t N)^(-1) = sum_k N^k t^k for any square N: comparing
        coefficients, B_0 = I and B_k = N B_(k-1). N has small integer
        entries, the order-0 factorization is that of the identity, and the
        powers N^k stay below 2^53, so the comparison is exact."""
        N = np.array([[1, 2, 0], [0, -1, 1], [2, 0, 1]])
        coefficients = [np.eye(3), -N.astype(float)] + [np.zeros((3, 3))] * 9
        inverse = TSM(coefficients).inverse()
        power = np.eye(3, dtype=np.int64)
        for k in range(P + 1):
            assert np.array_equal(inverse.coefficients[k], power), k
            power = power @ N
        assert np.max(np.abs(power)) < 2**53

    def test_the_inverse_uses_the_order_zero_factorization_at_every_order(self):
        """A(t) = A_0 (I - t N) has the inverse sum_k N^k A_0^(-1) t^k.

        A_0 = [[2, 1, 0], [1, 3, 1], [0, 1, 4]] has determinant
        2 (12 - 1) - 1 (4 - 0) = 18 and the inverse adj(A_0) / 18 with
        adj(A_0) = [[11, -4, 1], [-4, 8, -2], [1, -2, 5]], in rationals. The
        expected coefficients are formed in exact rational arithmetic. The
        scale of order k is the largest entry of N^k A_0^(-1); the solve with
        the factors of A_0 loses at most the condition number of A_0, below
        5, on top of the 128 EPS of the recurrence."""
        A0 = np.array([[2, 1, 0], [1, 3, 1], [0, 1, 4]])
        N = np.array([[1, 2, 0], [0, -1, 1], [2, 0, 1]])
        adjugate = [[11, -4, 1], [-4, 8, -2], [1, -2, 5]]
        inverse0 = [[Fraction(x, 18) for x in row] for row in adjugate]
        check = [[sum(int(A0[i][k]) * inverse0[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
        assert check == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        assert np.linalg.cond(A0) < 5.0
        coefficients = [A0.astype(float), -(A0 @ N).astype(float)] + [np.zeros((3, 3))] * 9
        inverse = TSM(coefficients).inverse()
        current = inverse0
        for k in range(P + 1):
            expected = np.array([[float(x) for x in row] for row in current])
            scale = float(np.max(np.abs(expected)))
            assert _max_error(inverse.coefficients[k], expected) <= 5.0 * _rounding(1, scale), k
            current = [[sum(int(N[i][m]) * current[m][j] for m in range(3)) for j in range(3)] for i in range(3)]

    def test_a_singular_order_zero_coefficient_has_no_inverse(self):
        """A_0 = [[1, 2], [2, 4]] is singular: elimination with partial
        pivoting takes the pivot 2 and leaves 2 - 4/2 = 0 exactly. The
        refusal names the inverse. A pivot of 2^-100 is not refused: the
        inverse of the 1 by 1 series 2^-100 + t starts 2^100, -2^200."""
        singular = TSM([np.array([[1.0, 2.0], [2.0, 4.0]]), np.eye(2)])
        with pytest.raises(ValueError, match="inverse"):
            singular.inverse()
        with pytest.raises(ValueError, match="square"):
            TSM(1, 2, 3).inverse()
        tiny = TSM([np.array([[2.0**-100]]), np.array([[1.0]])]).inverse()
        assert tiny.coefficients[0][0, 0] == 2.0**100
        assert tiny.coefficients[1][0, 0] == -(2.0**200)

    def test_the_determinant_of_identity_minus_t_n_is_the_reversed_characteristic_polynomial(self):
        """det(I - t N) = 1 - tr(N) t + e_2(N) t^2 - det(N) t^3 for a 3 by 3
        matrix N, with e_2 the sum of the principal 2 by 2 minors.

        N = [[1, 2, 0], [0, -1, 1], [2, 0, 1]]: tr = 1; e_2 = (1*-1 - 2*0) +
        (1*1 - 0*2) + (-1*1 - 1*0) = -1 + 1 - 1 = -1; det = 1 (-1 - 0) -
        2 (0 - 2) + 0 = 3. So the series is 1 - t - t^2 - 3 t^3. Integer
        arithmetic throughout, exact."""
        N = np.array([[1, 2, 0], [0, -1, 1], [2, 0, 1]])
        coefficients = [np.eye(3), -N.astype(float)] + [np.zeros((3, 3))] * 9
        determinant = TSM(coefficients).determinant()
        assert determinant.coefficients == [1, -1, -1, -3] + [0] * 7

    @pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
    def test_the_determinant_agrees_with_the_exact_expansion(self, n):
        """An n by n matrix whose entries are polynomials of degree ten with
        integer coefficients in [-3, 3]. The reference is the Leibniz
        expansion over the n! permutations in exact integer arithmetic, cut
        after t^10.

        The division-free algorithm forms sums and products of integers
        only. After m of its n - 1 passes an entry is bounded by
        3 (3 * 11 * n)^m, at most 3 * 165^4 < 2.3e9 for n = 5, far below
        2^53, so every operation is exact and the comparison is exact
        equality."""
        rng = np.random.default_rng(100 + n)
        a = _integer_series_matrix(rng, n, P)
        determinant = TSM(list(a.astype(complex))).determinant()
        assert determinant.coefficients == _exact_determinant(a, P)

    def test_the_determinant_has_a_value_when_the_order_zero_coefficient_is_singular(self):
        """A(t) = t M + t^2 K has A_0 = 0, and
        det A = t^2 det(M + t K) = t^2 det M + t^3 (m11 k22 + k11 m22 -
        m12 k21 - k12 m21) + t^4 det K.

        M = [[1, 2], [3, 5]], K = [[2, -1], [1, 4]]: det M = -1,
        det K = 9, and the middle term 1*4 + 2*5 - 2*1 - (-1)*3 = 15. No
        inverse of A_0 exists, and no division is made. The determinant of
        the 0 by 0 matrix is 1 and of a 1 by 1 matrix its entry."""
        M = np.array([[1.0, 2.0], [3.0, 5.0]])
        K = np.array([[2.0, -1.0], [1.0, 4.0]])
        coefficients = [np.zeros((2, 2)), M, K] + [np.zeros((2, 2))] * 8
        determinant = TSM(coefficients).determinant()
        assert determinant.coefficients == [0, 0, -1, 15, 9] + [0] * 6
        assert TSM(P, 0, 0).determinant().coefficients == [1] + [0] * P
        single = TSM(2, 1, 1)
        single.setEntry(0, 0, TS([3.0, -2.0, 5.0]))
        assert single.determinant().coefficients == [3.0, -2.0, 5.0]
        with pytest.raises(ValueError, match="square"):
            TSM(1, 2, 3).determinant()

    def test_determinant_and_inverse_satisfy_jacobis_formula(self):
        """(det A)' = det A * tr(A^(-1) A') as an identity between series,
        for a complex 4 by 4 matrix series: the determinant comes from the
        division-free algorithm and the right-hand side from the inverse
        recurrence, two independent routes.

        A_0 is the identity plus a perturbation of norm below 1/2, so its
        condition number is below 3. The two sides are series of order nine;
        the scale is the largest coefficient of the right-hand side, and the
        bound is that of four chained recurrences (inverse, product, trace
        sum, product) times the condition number."""
        rng = np.random.default_rng(4)
        n = 4
        a = 0.1 * (rng.normal(size=(P + 1, n, n)) + 1j * rng.normal(size=(P + 1, n, n)))
        a[0] += np.eye(n)
        assert np.linalg.cond(a[0]) < 3.0
        A = TSM(list(a))
        left = A.determinant().derivative()
        inverse = TSM(A.inverse().coefficients[:P])
        derivative = TSM([(k + 1) * a[k + 1] for k in range(P)])
        product = inverse @ derivative
        trace = product.entry(0, 0)
        for i in range(1, n):
            trace = trace + product.entry(i, i)
        right = A.determinant().truncated(P - 1) * trace
        scale = float(np.max(np.abs(_c(right))))
        assert _max_error(_c(left), _c(right)) <= 3.0 * _rounding(4, scale)


# ---------------------------------------------------------------------------
# The Riesz projector of a group of eigenvalues, as a series
# ---------------------------------------------------------------------------


def _inverse_square_root_series(order):
    """The coefficients of (1 + t^2)^(-1/2) = sum_m C(-1/2, m) t^(2m)."""
    out = [Fraction(0)] * (order + 1)
    for m in range(order // 2 + 1):
        out[2 * m] = _binomial(Fraction(-1, 2), m)
    return out


def _cosine_and_sine_of_two_t(order):
    """The coefficients of cos 2t and sin 2t: (-1)^m 2^(2m) / (2m)! and
    (-1)^m 2^(2m+1) / (2m+1)!."""
    cosine = [Fraction(0)] * (order + 1)
    sine = [Fraction(0)] * (order + 1)
    for k in range(order + 1):
        value = Fraction((-1) ** (k // 2) * 2**k, math.factorial(k))
        if k % 2 == 0:
            cosine[k] = value
        else:
            sine[k] = value
    return cosine, sine


class TestRieszProjectorSeries:
    def test_a_symmetric_two_by_two_family(self):
        """A(t) = [[1, t], [t, -1]] has the eigenvalues +-sqrt(1 + t^2), and
        the projector onto the positive one is (A - lambda_- I) /
        (lambda_+ - lambda_-) = (I + A / sqrt(1 + t^2)) / 2:

            P(t) = 1/2 [[1 + w, t w], [t w, 1 - w]],  w = (1 + t^2)^(-1/2)
                 = sum_m C(-1/2, m) t^(2m) = 1 - t^2/2 + 3 t^4/8 - ...

        The eigenvalue gap at order 0 is 2 and the adapted basis is
        orthonormal, so the quotients do not amplify: ten orders of the
        recurrence, each a few products of 2 by 2 matrices with entries no
        larger than 1, within the bound of two chained recurrences. The
        complementary group has the projector I - P."""
        w = _inverse_square_root_series(P)
        expected = np.zeros((P + 1, 2, 2))
        for k in range(P + 1):
            shifted = float(w[k - 1]) if k >= 1 else 0.0
            expected[k] = 0.5 * np.array([[float(w[k]), shifted], [shifted, -float(w[k])]])
        expected[0] += 0.5 * np.eye(2)
        coefficients = [np.diag([1.0, -1.0]), np.array([[0.0, 1.0], [1.0, 0.0]])] + [np.zeros((2, 2))] * 9
        A = TSM(coefficients)
        projector = A.rieszProjector(lambda value: value.real > 0)
        assert _max_error(_stack(projector), expected) <= _rounding(2)
        complement = A.rieszProjector(lambda value: value.real < 0)
        expected_complement = -expected
        expected_complement[0] += np.eye(2)
        assert _max_error(_stack(complement), expected_complement) <= _rounding(2)

    def test_a_complex_triangular_family_and_its_similarity_transform(self):
        """A(t) = [[2 + i + t, 1 - i], [0, -1 + i t]] is upper triangular
        with the eigenvalues a(t) = 2 + i + t and d(t) = -1 + i t. The
        projector of the first is P = [[1, y], [0, 0]], and A P = P A gives
        a y = (1 - i) + y d, so

            y(t) = (1 - i) / (a - d) = (1 - i) / (3 + i + (1 - i) t)
                 = (1 - i)/(3 + i) sum_k (-(1 - i)/(3 + i))^k t^k,

        a geometric series with ratio of modulus sqrt(2/10) < 1/2. The
        family is not normal and its coefficients are complex. The transform
        A' = S A S^(-1), P' = S P S^(-1) with S = [[1, 0], [i, 1]] (inverse
        [[1, 0], [-i, 1]], Gaussian integers) fills the lower triangle, so
        the Schur form of A'_0 is not read off the input.

        The entries are at most 1 in modulus, the gap at order 0 is
        |3 + i| > 3, and the condition number of S is below 3; the bound is
        that of two chained recurrences times the square of that condition
        number (the transform enters on both sides)."""
        ratio = -(1 - 1j) / (3 + 1j)
        y = [(1 - 1j) / (3 + 1j) * ratio**k for k in range(P + 1)]
        expected = np.zeros((P + 1, 2, 2), dtype=complex)
        expected[0, 0, 0] = 1.0
        expected[:, 0, 1] = y
        a = np.zeros((P + 1, 2, 2), dtype=complex)
        a[0] = [[2 + 1j, 1 - 1j], [0, -1]]
        a[1] = [[1, 0], [0, 1j]]
        select = lambda value: value.real > 0
        projector = TSM(list(a)).rieszProjector(select)
        assert _max_error(_stack(projector), expected) <= _rounding(2)
        S = np.array([[1, 0], [1j, 1]])
        S_inverse = np.array([[1, 0], [-1j, 1]])
        assert np.array_equal(S @ S_inverse, np.eye(2))
        condition = float(np.linalg.cond(S))
        assert condition < 3.0
        transformed = TSM([S @ x @ S_inverse for x in a]).rieszProjector(select)
        expected_transformed = np.array([S @ x @ S_inverse for x in expected])
        assert _max_error(_stack(transformed), expected_transformed) <= condition**2 * _rounding(2)

    def test_a_group_whose_eigenvalues_split_at_a_branch_point(self):
        """A(t) = [[1, 1, c1], [t, 1, c2], [0, 0, 3]] with c1 = 3/4 and
        c2 = -5/4. Its leading block J(t) = [[1, 1], [t, 1]] has the
        eigenvalues 1 +- sqrt(t): at t = 0 it is a Jordan block, and the two
        eigenvalues leave it with a square-root branch point, so neither
        eigenvector is a power series in t. The projector of the pair is one
        all the same: P = [[I, Y], [0, 0]] with (J - 3 I) Y = C from
        A P = P A, and (J - 3I)^(-1) = [[-2, -1], [-t, -2]] / (4 - t), so

            Y_1 = -(2 c1 + c2) / (4 - t),  Y_2 = -(c1 t + 2 c2) / (4 - t),
            1 / (4 - t) = sum_k t^k / 4^(k+1).

        Every coefficient is a dyadic rational. The gap between the group
        and the eigenvalue 3 is 2. The bound is that of two chained
        recurrences with terms no larger than 1."""
        c1, c2 = Fraction(3, 4), Fraction(-5, 4)
        geometric = [Fraction(1, 4 ** (k + 1)) for k in range(P + 1)]
        expected = np.zeros((P + 1, 3, 3))
        expected[0, 0, 0] = expected[0, 1, 1] = 1.0
        for k in range(P + 1):
            expected[k, 0, 2] = float(-(2 * c1 + c2) * geometric[k])
            below = geometric[k - 1] if k >= 1 else Fraction(0)
            expected[k, 1, 2] = float(-c1 * below - 2 * c2 * geometric[k])
        a = np.zeros((P + 1, 3, 3))
        a[0] = [[1, 1, float(c1)], [0, 1, float(c2)], [0, 0, 3]]
        a[1, 1, 0] = 1.0
        select = lambda value: abs(value - 1.0) < 1.0
        projector = TSM(list(a)).rieszProjector(select)
        assert _max_error(_stack(projector), expected) <= _rounding(2)

    def test_the_branch_point_family_in_a_basis_that_hides_its_structure(self):
        """The family of the previous test conjugated by
        S = [[1, 0, 0], [1, 1, 0], [2, 1, 1]], whose inverse
        [[1, 0, 0], [-1, 1, 0], [-1, -1, 1]] is integral: A' = S A S^(-1)
        is a full matrix whose order-0 coefficient is defective (one Jordan
        block of size two at the eigenvalue 1), and P' = S P S^(-1).

        The QR iteration returns the double eigenvalue of a Jordan block
        only to about sqrt(EPS), but the projector of the group does not
        depend on how the eigenvalue 1 is resolved inside the group, only on
        the separation from the eigenvalue 3. The bound is that of two
        chained recurrences times the square of the condition number of S
        (below 12) and the scale of the expected coefficients."""
        c1, c2 = Fraction(3, 4), Fraction(-5, 4)
        geometric = [Fraction(1, 4 ** (k + 1)) for k in range(P + 1)]
        projector0 = np.zeros((P + 1, 3, 3))
        projector0[0, 0, 0] = projector0[0, 1, 1] = 1.0
        for k in range(P + 1):
            projector0[k, 0, 2] = float(-(2 * c1 + c2) * geometric[k])
            below = geometric[k - 1] if k >= 1 else Fraction(0)
            projector0[k, 1, 2] = float(-c1 * below - 2 * c2 * geometric[k])
        a = np.zeros((P + 1, 3, 3))
        a[0] = [[1, 1, float(c1)], [0, 1, float(c2)], [0, 0, 3]]
        a[1, 1, 0] = 1.0
        S = np.array([[1.0, 0, 0], [1, 1, 0], [2, 1, 1]])
        S_inverse = np.array([[1.0, 0, 0], [-1, 1, 0], [-1, -1, 1]])
        assert np.array_equal(S @ S_inverse, np.eye(3))
        condition = float(np.linalg.cond(S))
        assert condition < 12.0
        expected = np.array([S @ x @ S_inverse for x in projector0])
        scale = float(np.max(np.abs(expected)))
        projector = TSM([S @ x @ S_inverse for x in a]).rieszProjector(lambda value: abs(value - 1.0) < 1.0)
        assert _max_error(_stack(projector), expected) <= condition**2 * _rounding(2, scale)

    def test_a_group_of_two_eigenvalues_in_a_rotating_frame(self):
        """A(t) = R(t) diag(1, 2, 5) R(t)^T with R(t) the rotation by the
        angle t in the plane of the second and third axes. The group
        {1, 2} has the projector R diag(1, 1, 0) R^T, whose eigenvectors are
        known in closed form (the columns of R):

            A(t) = [[1, 0, 0], [0, 7/2 - 3/2 cos 2t, -3/2 sin 2t],
                    [0, -3/2 sin 2t, 7/2 + 3/2 cos 2t]],
            P(t) = [[1, 0, 0], [0, (1 + cos 2t)/2, (sin 2t)/2],
                    [0, (sin 2t)/2, (1 - cos 2t)/2]],

        from cos^2 = (1 + cos 2t)/2, sin^2 = (1 - cos 2t)/2 and
        sin cos = (sin 2t)/2. The coefficients of cos 2t and sin 2t are
        (-1)^m 2^k / k!. A(t) is an infinite series, entered to order ten;
        the projector to order ten depends on A to order ten only.

        The gap between the group and the eigenvalue 5 is 3, the frame is
        orthonormal, and the coefficients of P are at most 1 (2^k / (2 k!)
        is 1 at k = 1 and k = 2 and falls from there), so the bound is that
        of two chained recurrences with terms no larger than the largest
        coefficient of A, 7/2 + 3/2 = 5."""
        cosine, sine = _cosine_and_sine_of_two_t(P)
        a = np.zeros((P + 1, 3, 3))
        expected = np.zeros((P + 1, 3, 3))
        for k in range(P + 1):
            c, s = float(cosine[k]), float(sine[k])
            a[k, 1, 1], a[k, 2, 2] = -1.5 * c, 1.5 * c
            a[k, 1, 2] = a[k, 2, 1] = -1.5 * s
            expected[k, 1, 1], expected[k, 2, 2] = 0.5 * c, -0.5 * c
            expected[k, 1, 2] = expected[k, 2, 1] = 0.5 * s
        a[0] += np.diag([1.0, 3.5, 3.5])
        expected[0] += np.diag([1.0, 0.5, 0.5])
        assert np.array_equal(a[0], np.diag([1.0, 2.0, 5.0]))
        projector = TSM(list(a)).rieszProjector(lambda value: abs(value) < 3.0)
        assert _max_error(_stack(projector), expected) <= _rounding(2, 5.0)

    def test_the_series_satisfies_the_defining_identities_for_a_complex_family(self):
        """P(t)^2 = P(t) and A(t) P(t) = P(t) A(t) order by order, for a
        complex, non-normal 5 by 5 family, with trace(P_0) = 2 (the number of
        eigenvalues in the group) and trace(P_k) = 0 above (the rank of a
        projector is constant).

        A_0 = S D S^(-1) with D = diag(1, 1.5 + 0.5i, 4, 5 + i, 6 - i) and S
        the identity plus a complex perturbation of size 0.2; the group is
        {1, 1.5 + 0.5i}, at distance at least 2.5 from the rest. The higher
        coefficients have entries of size 0.2.

        The residual of each identity at order k is a sum of at most
        2 (k + 1) products of 5 by 5 matrices; its rounding scale is the sum
        of the products of the norms of the factors, which is computed here,
        times EPS and the number of terms of an inner product (5). The
        recurrence itself solves each order with quotients by differences of
        at least 2.5 in a basis of condition number kappa (the condition
        number of the eigenvector matrix S), which enters squared; the bound
        is 128 * kappa^2 * EPS times that scale."""
        rng = np.random.default_rng(5)
        n = 5
        S = np.eye(n) + 0.2 * (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))
        D = np.diag([1.0, 1.5 + 0.5j, 4.0, 5.0 + 1j, 6.0 - 1j])
        a = 0.2 * (rng.normal(size=(P + 1, n, n)) + 1j * rng.normal(size=(P + 1, n, n)))
        a[0] = S @ D @ np.linalg.inv(S)
        kappa = float(np.linalg.cond(S))
        assert kappa < 10.0
        A = TSM(list(a))
        projector = A.rieszProjector(lambda value: abs(value) < 3.0)
        p = _stack(projector)
        norm = lambda matrix: float(np.linalg.norm(matrix, 2))
        assert abs(np.trace(p[0]) - 2.0) <= kappa**2 * _rounding(1)
        squared = _stack(projector @ projector)
        left, right = _stack(A @ projector), _stack(projector @ A)
        for k in range(P + 1):
            idempotency_scale = sum(norm(p[j]) * norm(p[k - j]) for j in range(k + 1))
            commutation_scale = 2.0 * sum(norm(a[j]) * norm(p[k - j]) for j in range(k + 1))
            bound = 128.0 * kappa**2 * EPS * n
            assert _max_error(squared[k], p[k]) <= bound * idempotency_scale, k
            assert _max_error(left[k], right[k]) <= bound * commutation_scale, k
            if k >= 1:
                assert abs(np.trace(p[k])) <= bound * norm(p[k]), k

    def test_the_first_order_is_the_perturbation_formula(self):
        """For a diagonal A_0 = diag(lambda_1, ..., lambda_n) the first-order
        coefficient of the projector of a group G is
        (P_1)_ab = (A_1)_ab / (lambda_a - lambda_b) when exactly one of a, b
        lies in G (with a in G for that sign, and the opposite sign with b in
        G), and 0 otherwise.

        lambda = (1, 2, 5, 7), G = {1, 2}, and A_1 with the integer entries
        (A_1)_ab = 3a - 2b + 1 (indices from zero): for example
        (P_1)_02 = (0 - 4 + 1) / (1 - 5) = 3/4. The quotients are single
        rounded divisions of small integers and the frame is the identity,
        so the bound is a few EPS; 8 EPS is used."""
        eigenvalues = [1.0, 2.0, 5.0, 7.0]
        group = [True, True, False, False]
        A1 = np.array([[3.0 * a - 2.0 * b + 1.0 for b in range(4)] for a in range(4)])
        projector = TSM([np.diag(eigenvalues), A1]).rieszProjector(lambda value: value.real < 3.0)
        expected = np.zeros((4, 4))
        for a in range(4):
            for b in range(4):
                if group[a] and not group[b]:
                    expected[a, b] = A1[a, b] / (eigenvalues[a] - eigenvalues[b])
                elif group[b] and not group[a]:
                    expected[a, b] = A1[a, b] / (eigenvalues[b] - eigenvalues[a])
        assert expected[0, 2] == 0.75
        assert np.array_equal(projector.coefficients[0], np.diag([1.0, 1.0, 0.0, 0.0]))
        assert _max_error(projector.coefficients[1], expected) <= 8 * EPS

    def test_the_empty_group_and_the_whole_spectrum(self):
        """A group without an eigenvalue has the zero projector and a group
        with every eigenvalue the identity, at every order, exactly."""
        rng = np.random.default_rng(9)
        a = rng.normal(size=(4, 3, 3))
        A = TSM(list(a))
        nothing = A.rieszProjector(lambda value: False)
        assert np.array_equal(_stack(nothing), np.zeros((4, 3, 3)))
        everything = A.rieszProjector(lambda value: True)
        expected = np.zeros((4, 3, 3))
        expected[0] = np.eye(3)
        assert np.array_equal(_stack(everything), expected)
        with pytest.raises(ValueError, match="square"):
            TSM(1, 2, 3).rieszProjector(lambda value: True)

    def test_an_eigenvalue_inside_and_outside_the_group_has_no_projector(self):
        """A_0 = diag(1, 1, 3) with a selection that takes one of the two
        eigenvalues 1 and leaves the other: their difference is exactly zero
        and has no quotient. A difference of 2^-40 is divided by as it
        stands: for A(t) = [[1 + 2^-40, t], [0, 1]] the projector of the
        first eigenvalue is [[1, t / 2^-40], [0, 0]], the first-order entry
        2^40 exactly."""
        calls = []

        def first_only(value):
            calls.append(value)
            return len(calls) == 1

        A = TSM([np.diag([1.0, 1.0, 3.0]), np.ones((3, 3))])
        with pytest.raises(ValueError, match="inside the group and outside it"):
            A.rieszProjector(first_only)
        delta = 2.0**-40
        close = TSM([np.array([[1.0 + delta, 0.0], [0.0, 1.0]]), np.array([[0.0, 1.0], [0.0, 0.0]])])
        projector = close.rieszProjector(lambda value: value.real > 1.0)
        assert np.array_equal(projector.coefficients[0], np.array([[1.0, 0.0], [0.0, 0.0]]))
        assert np.array_equal(projector.coefficients[1], np.array([[0.0, 2.0**40], [0.0, 0.0]]))


# ---------------------------------------------------------------------------
# Reversion: the order-p step
# ---------------------------------------------------------------------------


def _square_minus_two(x):
    """F(x) = x^2 - 2 on series."""
    return [x[0] * x[0] - 2.0]


def _sum_and_product(x):
    """F(x, y) = (x + y - 3, x y - 2) on series; its roots are (1, 2) and
    (2, 1)."""
    return [x[0] + x[1] - 3.0, x[0] * x[1] - 2.0]


def _exact_square_root_series(radicand, root, order):
    """The coefficients of the square root of a polynomial with rational
    coefficients `radicand` whose order-0 coefficient has the rational root
    `root`, in exact rational arithmetic: g_k = (q_k - sum_{j=1..k-1} g_j
    g_(k-j)) / (2 g_0)."""
    q = list(radicand) + [Fraction(0)] * (order + 1)
    g = [Fraction(root)]
    for k in range(1, order + 1):
        g.append((q[k] - sum(g[j] * g[k - j] for j in range(1, k))) / (2 * g[0]))
    return g


class TestReversion:
    def test_the_path_of_x_squared_minus_two_from_one_is_the_binomial_series(self):
        """F(x) = x^2 - 2 at x_0 = 1, where F = -1. The path solves
        (1 + s)^2 - 2 = -(1 - t), that is (1 + s)^2 = 1 + t, so
        s(t) = sqrt(1 + t) - 1 = sum_{k>=1} C(1/2, k) t^k =
        t/2 - t^2/8 + t^3/16 - ...

        J = 2, and the caller's solve divides by it. Every s_k is a dyadic
        rational with a short mantissa and the series evaluation of F forms
        sums and products of such numbers only, so the comparison is exact
        equality."""
        calls = []

        def solve(rhs):
            calls.append(np.array(rhs))
            return rhs / 2.0

        reversion = nm.revert(_square_minus_two, np.array([1.0]), P, solve)
        assert reversion.residual[0] == -1.0
        expected = [0.0] + [float(_binomial(Fraction(1, 2), k)) for k in range(1, P + 1)]
        assert [c[0] for c in reversion.coefficients] == expected
        # one linear solve per order, the first with the right-hand side -F
        assert len(calls) == P
        assert calls[0][0] == 1.0

    def test_the_order_p_step_of_x_squared_minus_two_errs_as_the_series_predicts(self):
        """From x_0 = 1 the order-p step lands on the p-th partial sum of the
        binomial series of sqrt(2) = sum_k C(1/2, k): 3/2, 11/8, 23/16, ...

        The terms alternate in sign from k = 1 and decrease in magnitude
        (|C(1/2, k+1)| / |C(1/2, k)| = (k - 1/2) / (k + 1) < 1), so by the
        alternating-series estimate the error of the p-th partial sum lies
        between |a_(p+1)| - |a_(p+2)| and |a_(p+1)|, a_k = C(1/2, k), and
        falls with p. The value t = 1 is on the circle of convergence (the
        branch point of the path is t = -1), so the error falls only
        algebraically: 8.6e-2 at order 1, 4.3e-3 at order 10."""
        reversion = nm.revert(_square_minus_two, np.array([1.0]), P, lambda rhs: rhs / 2.0)
        a = [float(_binomial(Fraction(1, 2), k)) for k in range(P + 3)]
        errors = []
        for p in range(1, P + 1):
            point = 1.0 + sum(c[0].real for c in reversion.coefficients[1 : p + 1])
            assert point == float(sum(_binomial(Fraction(1, 2), k) for k in range(p + 1)))
            errors.append(abs(point - math.sqrt(2.0)))
            assert abs(a[p + 1]) - abs(a[p + 2]) <= errors[-1] <= abs(a[p + 1])
        assert all(later < earlier for earlier, later in zip(errors, errors[1:]))
        assert 8.5e-2 < errors[0] < 8.6e-2 and 4.2e-3 < errors[-1] < 4.3e-3
        # the step is the sum of the ten coefficients: dyadic rationals,
        # summed exactly in either order
        assert abs(1.0 + reversion.step()[0].real - math.sqrt(2.0)) == errors[-1]

    def test_from_a_closer_point_the_error_falls_geometrically(self):
        """F(x) = x^2 - 2 at x_0 = 3/2, where F = 1/4. The path solves
        (x_0 + s)^2 = x_0^2 - t (x_0^2 - 2), so x_0 + s = x_0 sqrt(1 - e t)
        with e = (x_0^2 - 2) / x_0^2 = 1/9, and s_k = x_0 C(1/2, k) (-e)^k.

        For k >= 1 the terms x_0 C(1/2, k) (-1/9)^k are all negative, so the
        order-p step overshoots sqrt(2) by the tail sum_{k>p} |s_k|, which
        lies between its first term |s_(p+1)| and the geometric bound
        |s_(p+1)| / (1 - e): the error falls by about a factor of nine per
        order, from 2.4e-3 at order 1 to below 1e-12 at order 10. The
        coefficients carry the rounding of one recurrence with terms no
        larger than x_0."""
        x0, e = Fraction(3, 2), Fraction(1, 9)
        reversion = nm.revert(_square_minus_two, np.array([1.5]), P, lambda rhs: rhs / 3.0)
        assert reversion.residual[0] == 0.25
        exact = [x0 * _binomial(Fraction(1, 2), k) * (-e) ** k for k in range(P + 2)]
        actual = np.array([c[0] for c in reversion.coefficients])
        assert _max_error(actual[1:], _floats(exact[1 : P + 1])) <= _rounding(1, 1.5)
        errors = []
        for p in range(1, P + 1):
            point = 1.5 + sum(c[0].real for c in reversion.coefficients[1 : p + 1])
            error = point - math.sqrt(2.0)
            first_omitted = abs(float(exact[p + 1]))
            slack = _rounding(1, 1.5)
            assert first_omitted - slack <= error <= first_omitted / (1 - float(e)) + slack
            errors.append(error)
        assert 2.3e-3 < errors[0] < 2.5e-3
        assert errors[-1] < 1e-12
        assert all(later < earlier / 8.0 for earlier, later in zip(errors, errors[1:9]))

    def test_a_complex_scalar_equation(self):
        """F(z) = z^2 + 1 at z_0 = 1 + i, where F = 1 + 2i. The path solves
        z^2 = -1 + (1 - t)(1 + 2i) = 2i - (1 + 2i) t, so
        z = (1 + i) sqrt(1 - t (1 + 2i) / (2i)) = (1 + i) sqrt(1 - (1 - i/2) t)
        and s_k = (1 + i) C(1/2, k) (-(1 - i/2))^k. The modulus of the ratio
        is sqrt(5)/2, so the scale of the coefficients is at most
        sqrt(2) (sqrt(5)/2)^10 / 2 < 2.2."""
        def equation(z):
            return [z[0] * z[0] + 1.0]

        z0 = 1.0 + 1.0j
        reversion = nm.revert(equation, np.array([z0]), P, lambda rhs: rhs / (2.0 * z0))
        assert reversion.residual[0] == 1.0 + 2.0j
        ratio = -(1.0 - 0.5j)
        expected = [z0 * float(_binomial(Fraction(1, 2), k)) * ratio**k for k in range(P + 1)]
        actual = np.array([c[0] for c in reversion.coefficients])
        assert _max_error(actual[1:], expected[1:]) <= _rounding(1, 2.2)

    def test_the_jacobian_is_read_from_first_order_series(self):
        """F(x, y) = (x + y - 3, x y - 2) has the Jacobian
        [[1, 1], [y, x]]; at (3/4, 5/2) that is [[1, 1], [5/2, 3/4]]. Column
        i is the coefficient of t of F(x_0 + t e_i): sums and products of
        dyadic rationals, exact."""
        jacobian = nm.jacobian(_sum_and_product, np.array([0.75, 2.5]))
        assert np.array_equal(jacobian, np.array([[1.0, 1.0], [2.5, 0.75]]))

    def test_order_one_is_newtons_step(self):
        """At order 1 the step is s_1 = -J^(-1) F(x_0), Newton's step, and
        s_1 is the same at every higher order.

        For F(x, y) = (x + y - 3, x y - 2) at (3/4, 5/2): F = (1/4, -1/8),
        J = [[1, 1], [5/2, 3/4]] with determinant -7/4, and
        -J^(-1) F = (4/7) [[3/4, -1], [-5/2, 1]] (1/4, -1/8) =
        (4/7) (5/16, -3/4) = (5/28, -3/7). The comparison with the same
        linear solve applied to -F is exact (one and the same computation);
        the comparison with the rational value carries the rounding of one
        2 by 2 solve, a few EPS times the condition number of J (below 6)."""
        point = np.array([0.75, 2.5])
        J = nm.jacobian(_sum_and_product, point)
        assert np.linalg.cond(J) < 6.0
        solve = lambda rhs: np.linalg.solve(J, rhs)
        first = nm.revert(_sum_and_product, point, 1, solve)
        assert np.array_equal(first.residual, np.array([0.25, -0.125]))
        newton = np.linalg.solve(J, -first.residual)
        assert np.array_equal(first.step(), newton)
        assert np.array_equal(first.coefficients[1], newton)
        assert _max_error(first.step(), [5 / 28, -3 / 7]) <= 6.0 * 8 * EPS
        tenth = nm.revert(_sum_and_product, point, P, solve)
        assert np.array_equal(tenth.coefficients[1], newton)
        scalar = nm.revert(_square_minus_two, np.array([1.0]), 1, lambda rhs: rhs / 2.0)
        assert scalar.step()[0] == 0.5  # -F / F' = 1 / 2

    def test_a_two_variable_system_follows_its_closed_form_path(self):
        """F(x, y) = (x + y - 3, x y - 2) from (3/4, 5/2).

        Along the path the sum and the product move linearly to their
        targets: sigma(t) = 3 + (1 - t)(13/4 - 3) and
        pi(t) = 2 + (1 - t)(15/8 - 2). x and y are the two roots of
        z^2 - sigma z + pi, x = (sigma - sqrt(q)) / 2 and
        y = (sigma + sqrt(q)) / 2 with q = sigma^2 - 4 pi a quadratic in t
        and sqrt(q(0)) = y_0 - x_0 = 7/4, so the Taylor coefficients of the
        path are rational; they are formed here in exact rational arithmetic.

        Here sigma = 13/4 - t/4 and pi = 15/8 + t/8, so
        q = 49/16 - 17 t/8 + t^2/16, with the zeros t = 17 -+ sqrt(240):
        the nearer one, 1.508, is the branch point of the path, and t = 1
        lies inside the disc of convergence. The coefficients fall off like
        1.508^-k, their scale is below 1, and the condition number of J is
        below 6: the bound is that of one recurrence times the condition
        number of the linear solve, accumulated over ten orders,
        10 * 6 * 128 EPS."""
        x0, y0 = Fraction(3, 4), Fraction(5, 2)
        sigma = [x0 + y0, 3 - (x0 + y0)]  # sigma(t) = sigma_0 + t (3 - sigma_0)
        pi = [x0 * y0, 2 - x0 * y0]
        q = [
            sigma[0] ** 2 - 4 * pi[0],
            2 * sigma[0] * sigma[1] - 4 * pi[1],
            sigma[1] ** 2,
        ]
        assert q == [Fraction(49, 16), Fraction(-17, 8), Fraction(1, 16)]
        assert q[0] == (y0 - x0) ** 2
        terms = 200
        root = _exact_square_root_series(q, y0 - x0, terms)
        sigma_series = sigma + [Fraction(0)] * (terms - 1)
        x_exact = [(sigma_series[k] - root[k]) / 2 for k in range(terms + 1)]
        y_exact = [(sigma_series[k] + root[k]) / 2 for k in range(terms + 1)]
        assert (x_exact[0], y_exact[0]) == (x0, y0)

        point = np.array([0.75, 2.5])
        J = np.array([[1.0, 1.0], [2.5, 0.75]])
        reversion = nm.revert(_sum_and_product, point, P, lambda rhs: np.linalg.solve(J, rhs))
        actual = np.array(reversion.coefficients)
        bound = 10 * 6.0 * _rounding(1)
        assert _max_error(actual[1:, 0], _floats(x_exact[1 : P + 1])) <= bound
        assert _max_error(actual[1:, 1], _floats(y_exact[1 : P + 1])) <= bound

        # The order-p point approaches the root (1, 2), and its error is the
        # tail of the series: the exact coefficients above order p, summed
        # here to order 200, where they are below 1.508^-200 < 1e-35.
        assert abs(x_exact[terms]) < Fraction(1, 10**35)
        errors = []
        for p in range(1, P + 1):
            landed = point + np.sum(actual[1 : p + 1], axis=0).real
            tail_x = float(sum(x_exact[p + 1 :]))
            tail_y = float(sum(y_exact[p + 1 :]))
            assert abs((1.0 - landed[0]) - tail_x) <= bound
            assert abs((2.0 - landed[1]) - tail_y) <= bound
            errors.append(max(abs(1.0 - landed[0]), abs(2.0 - landed[1])))
        assert all(later < earlier for earlier, later in zip(errors, errors[1:]))
        assert errors[0] > 5e-2 and errors[-1] < 2e-4
        # the step is the path at t = 1; the path at t = 1/2 is the sum of
        # s_k / 2^k (Horner's rule and the plain sum differ by rounding only)
        assert _max_error(reversion.step(), np.sum(actual[1:], axis=0)) <= bound
        assert _max_error(reversion.evaluate(1.0), reversion.step()) <= bound
        half = sum(actual[k] * 0.5**k for k in range(P + 1))
        assert _max_error(reversion.evaluate(0.5), half) <= bound

    def test_the_caller_decides_how_a_rank_deficient_jacobian_is_solved(self):
        """One equation in two unknowns: F(x, y) = x^2 + y^2 - 2 at
        (1/2, 1/2), where F = -3/2 and J = [1, 1] has rank one. The caller
        solves each order by the minimum-norm solution J^T (J J^T)^(-1) r =
        (r/2, r/2), which keeps the path on the diagonal x = y; there
        2 x^2 - 2 = -(3/2)(1 - t) gives x = (1/2) sqrt(1 + 3 t) and
        s_k = (1/2) C(1/2, k) 3^k for both unknowns: 3/4, -9/16, 27/32, ...
        Dyadic rationals with short mantissas, exact."""
        def circle(x):
            return [x[0] * x[0] + x[1] * x[1] - 2.0]

        point = np.array([0.5, 0.5])
        assert np.array_equal(nm.jacobian(circle, point), np.array([[1.0, 1.0]]))
        minimum_norm = lambda rhs: np.array([rhs[0] / 2.0, rhs[0] / 2.0])
        reversion = nm.revert(circle, point, P, minimum_norm)
        assert reversion.residual.shape == (1,)
        assert reversion.residual[0] == -1.5
        for k in range(1, P + 1):
            expected = float(Fraction(1, 2) * _binomial(Fraction(1, 2), k) * 3**k)
            assert reversion.coefficients[k][0] == expected, k
            assert reversion.coefficients[k][1] == expected, k

    def test_order_zero_and_the_shapes_the_callables_must_return(self):
        """At order 0 the path has no coefficient beyond s_0 = 0 and only
        the residual is evaluated. A linear solve that returns a vector of
        the wrong length, a map that returns a series of another order, and
        an order above ten are reported."""
        point = np.array([0.75, 2.5])
        solve = lambda rhs: np.linalg.solve(np.array([[1.0, 1.0], [2.5, 0.75]]), rhs)
        zeroth = nm.revert(_sum_and_product, point, 0, solve)
        assert np.array_equal(zeroth.residual, np.array([0.25, -0.125]))
        assert len(zeroth.coefficients) == 1
        assert np.array_equal(zeroth.step(), np.zeros(2))
        with pytest.raises(ValueError, match="length 3"):
            nm.revert(_sum_and_product, point, 2, lambda rhs: np.zeros(3))
        with pytest.raises(ValueError, match="order 0"):
            nm.revert(lambda x: [TS([1.0]), TS([1.0])], point, 2, solve)
        with pytest.raises(ValueError, match="order 11"):
            nm.revert(_sum_and_product, point, 11, solve)
