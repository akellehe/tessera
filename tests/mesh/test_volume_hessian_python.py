# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Exact directional second derivative of a simplex's signed `volume()`.

`Simplex.volumeGradientDirectionalDerivative(v)` returns
`sum_f v_f d^2V/dl^2_e dl^2_f`, keyed by `e`. It is closed form because the Gram
matrix is **linear** in the squared lengths, so `d^2G/dl^2 dl^2` vanishes
identically and Jacobi's formula differentiated twice leaves only

    d^2V/dz_e dz_f = (V/4) t_e t_f - (V/2) tr(G^-1 dG_f G^-1 dG_e),
    t_e = tr(G^-1 dG_e).

The rigorous checks here are the exact **Euler homogeneity identity** and
**symmetry of the second derivative**, both at machine precision. A finite
difference is only ever a roundoff-limited cross-check: the descent direction
this feeds uses the analytic value, never FD.

`det G` is homogeneous of degree `d` in `l^2`, so `V` is homogeneous of degree
`d/2` and, differentiating Euler's relation once more,

    sum_f z_f d^2V/dz_e dz_f = (d/2 - 1) dV/dz_e.
"""
import cmath
import math
import unittest

import tessera as T

# Machine-precision bar: these are closed-form identities, not approximations.
EXACT = 1e-13


def _top_of_dim(st, nverts):
    return next(s for s in st.getSimplices()
                if len([v for v in s.getVertices()]) == nverts)


def _key(a, b):
    return (min(a, b), max(a, b))


def _squared_lengths(st):
    return {_key(e.getSource().getId(), e.getTarget().getId()):
            e.getLength() * e.getLength()
            for e in st.getEdgeList().toVector()}


def _jittered_simplex(nverts, imaginary=True):
    """One `nverts`-vertex cell with deliberately asymmetric complex l^2, so no
    identity below can pass by symmetry alone."""
    st = T.Spacetime.fromVertexTuples(nverts - 1, [list(range(nverts))], 1.0, 0.0)
    for i, e in enumerate(st.getEdgeList().toVector()):
        value = complex(1.0 + 0.031 * (i % 7), 0.011 * (i % 3) if imaginary else 0.0)
        e.setLength(cmath.sqrt(value))
    return st, _top_of_dim(st, nverts)


class VolumeHessianEulerIdentityTest(unittest.TestCase):
    """The exact homogeneity identity, at every degree the mesh builds."""

    def test_euler_identity_every_degree(self):
        for nverts in (3, 4, 5):
            with self.subTest(nverts=nverts):
                st, simplex = _jittered_simplex(nverts)
                degree = nverts - 1
                gradient = simplex.volumeGradient()
                contracted = simplex.volumeGradientDirectionalDerivative(
                    _squared_lengths(st))
                self.assertEqual(set(contracted), set(gradient))
                expected_factor = degree / 2.0 - 1.0
                scale = max(abs(v) for v in gradient.values())
                for edge, value in gradient.items():
                    self.assertLessEqual(
                        abs(contracted[edge] - expected_factor * value) / scale,
                        EXACT,
                        f"Euler identity failed at degree {degree}, edge {edge}")

    def test_triangle_second_derivative_vanishes_on_the_scaling_ray(self):
        # d = 2 makes the Euler factor (d/2 - 1) exactly zero: area is
        # homogeneous of degree one, so its gradient is degree zero and does
        # not move along the scaling ray at all.
        st, triangle = _jittered_simplex(3)
        contracted = triangle.volumeGradientDirectionalDerivative(
            _squared_lengths(st))
        for edge, value in contracted.items():
            self.assertLessEqual(abs(value), EXACT, f"edge {edge}")

    def test_real_lengths_give_the_same_identity(self):
        # The identity is algebraic, so it must not depend on the l^2 being
        # complex; a purely spacelike cell exercises the real path.
        st, simplex = _jittered_simplex(5, imaginary=False)
        gradient = simplex.volumeGradient()
        contracted = simplex.volumeGradientDirectionalDerivative(
            _squared_lengths(st))
        scale = max(abs(v) for v in gradient.values())
        for edge, value in gradient.items():
            self.assertLessEqual(
                abs(contracted[edge] - 1.0 * value) / scale, EXACT, f"edge {edge}")


class VolumeHessianSymmetryTest(unittest.TestCase):
    """d^2V/dz_e dz_f is symmetric in (e, f) — the mixed partials commute."""

    def test_mixed_partials_commute(self):
        st, simplex = _jittered_simplex(5)
        edges = sorted(simplex.volumeGradient())
        scale = max(abs(v) for v in simplex.volumeGradient().values())
        for i, first in enumerate(edges):
            for second in edges[i + 1:]:
                one = simplex.volumeGradientDirectionalDerivative(
                    {second: complex(1.0, 0.0)})[first]
                other = simplex.volumeGradientDirectionalDerivative(
                    {first: complex(1.0, 0.0)})[second]
                self.assertLessEqual(abs(one - other) / scale, EXACT,
                                     f"{first} vs {second}")

    def test_linear_in_the_direction(self):
        # Contracting a tensor against v is linear in v: the identity that lets
        # one contraction stand in for the whole Hessian.
        st, simplex = _jittered_simplex(4)
        first_edge, second_edge = sorted(simplex.volumeGradient())[:2]
        alpha, beta = complex(0.37, -0.11), complex(-0.52, 0.23)
        combined = simplex.volumeGradientDirectionalDerivative(
            {first_edge: alpha, second_edge: beta})
        separate_a = simplex.volumeGradientDirectionalDerivative(
            {first_edge: alpha})
        separate_b = simplex.volumeGradientDirectionalDerivative(
            {second_edge: beta})
        scale = max(abs(v) for v in simplex.volumeGradient().values())
        for edge in combined:
            self.assertLessEqual(
                abs(combined[edge] - separate_a[edge] - separate_b[edge]) / scale,
                EXACT, f"edge {edge}")


class VolumeHessianDegenerateInputTest(unittest.TestCase):
    """Shape contracts: an absent edge is a zero direction, never an error."""

    def test_empty_direction_gives_zero(self):
        st, simplex = _jittered_simplex(4)
        contracted = simplex.volumeGradientDirectionalDerivative({})
        for edge, value in contracted.items():
            self.assertLessEqual(abs(value), EXACT, f"edge {edge}")

    def test_unknown_edges_in_the_direction_are_ignored(self):
        st, simplex = _jittered_simplex(4)
        direction = _squared_lengths(st)
        direction[(9998, 9999)] = complex(3.0, 1.0)  # not an edge of this cell
        contracted = simplex.volumeGradientDirectionalDerivative(direction)
        reference = simplex.volumeGradientDirectionalDerivative(
            _squared_lengths(st))
        for edge in reference:
            self.assertLessEqual(abs(contracted[edge] - reference[edge]), EXACT)


class VolumeHessianFiniteDifferenceCrossCheckTest(unittest.TestCase):
    """A roundoff-limited cross-check ONLY.

    The finite difference is the approximate side of this comparison; the
    analytic value is exact. The tolerance is therefore loose by construction
    and must never be read as the accuracy of the analytic result, which the
    Euler identity above pins at machine precision.
    """

    def test_matches_a_central_difference_to_finite_difference_accuracy(self):
        st, simplex = _jittered_simplex(4)
        edges = st.getEdgeList().toVector()
        base = _squared_lengths(st)
        direction = {k: complex(0.41, -0.17) * (1 + (i % 3))
                     for i, k in enumerate(sorted(base))}
        exact = simplex.volumeGradientDirectionalDerivative(direction)

        def set_squared(values):
            for edge in edges:
                key = _key(edge.getSource().getId(), edge.getTarget().getId())
                edge.setLength(cmath.sqrt(values[key]))

        step = math.pow(2.220446049250313e-16, 1.0 / 3.0)
        plus = {k: base[k] + step * direction[k] for k in base}
        minus = {k: base[k] - step * direction[k] for k in base}
        set_squared(plus)
        gradient_plus = simplex.volumeGradient()
        set_squared(minus)
        gradient_minus = simplex.volumeGradient()
        set_squared(base)

        scale = max(abs(v) for v in exact.values())
        for edge in exact:
            approximated = (gradient_plus[edge] - gradient_minus[edge]) / (2 * step)
            self.assertLessEqual(abs(exact[edge] - approximated) / scale, 1e-8,
                                 f"edge {edge}")


if __name__ == "__main__":
    unittest.main()
