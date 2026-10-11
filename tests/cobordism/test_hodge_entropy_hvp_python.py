# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Exact analytic Hessian-vector product of the spectral entropy.

`HodgeLaplacian.spectral_entropy_gradient_directional_derivative(k, v)` is the
directional derivative of `spectral_entropy_gradient` along `v` for a REAL
parameter — the object the descent direction of `||grad_z S||^2` needs. The
joint Regge-Hodge objective used to take it as a central finite difference; the
exactness contract admits no finite-difference direction, so it is closed form:
the simplex volume Hessian, the second derivative of `L_k` contracted against
the direction, and the Daleckii-Krein derivative of `dS/d_a` on the fixed-rank
stratum the value already selects.

The rigorous check is an exact **homogeneity identity**. Every weight `W_j` is
homogeneous of degree `j` in `z = l^2`, so both terms of
`L_k = W_k^-1 d^T W_{k-1} d + d W_{k+1}^-1 d^T W_k` scale as `1/lambda` and
`L_k(lambda z) = L_k(z)/lambda` for any complex `lambda`. Then
`A = L^dagger L` scales by `|lambda|^-2`, the normalized `rho = A/tr A` is
INVARIANT, and so is the entropy. Hence `S(lambda z) = S(z)`, its gradient is
homogeneous of degree -1, and

    D_z h = -h        (direction = the squared lengths themselves).

That is exact to the conditioning of the eigendecomposition — no step size
enters — and it is what these tests assert. Any comparison against the retired
finite difference is reported as the roundoff-limited cross-check it is.
"""
import cmath
import math
import unittest

import tessera as T
from tessera import cobordism as cob


# Every closed form in this module is the DIAGONAL-weight operator's:
# L_k = W_k^-1 d_k^T W_{k-1} d_k + d_{k+1} W_{k+1}^-1 d_{k+1}^T W_k of
# HodgeWeightConvention. The process default metric source is the chain-level
# Whitney pencil (#1185), whose operator is the covariant h_k(s, U) on
# geometric images, so this module names the diagonal source at every operator
# it builds. The default's own properties are pinned in
# tests/cobordism/test_whitney_default_metric_python.py.
DIAGONAL = cob.HodgeMetricSource.DiagonalWeights


def _hodge(spacetime, weights=None, source=DIAGONAL):
    """The diagonal-weight Hodge operator this module's anchors are taken of."""
    if weights is None:
        weights = cob.HodgeLaplacian.default_weight_convention()
    return cob.HodgeLaplacian(spacetime, weights, source)

# The identity is closed form; the residual floor is the eigensolve's
# conditioning, not a step size.
EXACT = 1e-11
MODES = (
    ("include_phase", cob.HodgeEntropyPhaseMode.IncludeComplexPhase),
    ("ignore_phase", cob.HodgeEntropyPhaseMode.IgnoreComplexPhase),
)


def _jittered_pentatope_sphere(scale=1.0):
    signature = T.Signature(4, T.Lorentzian)
    spacetime = T.Spacetime(T.Metric(True, signature), T.CDT, 1.0, 1.0,
                            T.PREFERRED, T.SimplexBoundarySphere(4))
    spacetime.build()
    for index, edge in enumerate(spacetime.get_edge_list().to_vector()):
        edge.set_length(cmath.sqrt(scale * complex(1.0 + 0.023 * (index % 5),
                                                  0.007 * (index % 3))))
    return spacetime


def _jittered_cells(dimension, cells):
    spacetime = T.Spacetime.from_vertex_tuples(dimension, cells, 1.0, 0.0)
    for index, edge in enumerate(spacetime.get_edge_list().to_vector()):
        edge.set_length(cmath.sqrt(complex(1.0 + 0.037 * (index % 6),
                                          0.013 * (index % 4))))
    return spacetime


def _squared_lengths(spacetime):
    return [edge.get_length() * edge.get_length()
            for edge in spacetime.get_edge_list().to_vector()]


def _relative_sup(left, right, scale):
    if not left:
        return 0.0
    return max(abs(a - b) for a, b in zip(left, right)) / scale


def _scale_of(values):
    return max((abs(v) for v in values), default=0.0) or 1.0


class EntropyHessianEulerIdentityTest(unittest.TestCase):
    """D_z h = -h, the exact consequence of scale-invariant entropy."""

    def test_identity_on_the_lorentzian_sphere(self):
        spacetime = _jittered_pentatope_sphere()
        squared = _squared_lengths(spacetime)
        for name, mode in MODES:
            for degree in (0, 1, 2):
                with self.subTest(mode=name, degree=degree):
                    hodge = _hodge(spacetime)
                    gradient = hodge.spectral_entropy_gradient(degree, mode)
                    contracted = (
                        hodge.spectral_entropy_gradient_directional_derivative(
                            degree, squared, mode))
                    negated = [-g for g in gradient]
                    self.assertLessEqual(
                        _relative_sup(contracted, negated, _scale_of(gradient)),
                        EXACT)

    def test_identity_on_a_glued_two_complex(self):
        spacetime = _jittered_cells(2, [[0, 1, 2], [1, 2, 3], [2, 3, 4]])
        squared = _squared_lengths(spacetime)
        for name, mode in MODES:
            for degree in (0, 1):
                with self.subTest(mode=name, degree=degree):
                    hodge = _hodge(spacetime)
                    gradient = hodge.spectral_entropy_gradient(degree, mode)
                    contracted = (
                        hodge.spectral_entropy_gradient_directional_derivative(
                            degree, squared, mode))
                    negated = [-g for g in gradient]
                    self.assertLessEqual(
                        _relative_sup(contracted, negated, _scale_of(gradient)),
                        EXACT)

    def test_identity_survives_a_global_rescaling(self):
        # S is invariant under z -> lambda z, so the identity is not an
        # artifact of the particular scale the fixture was built at.
        for scale in (0.25, 4.0):
            spacetime = _jittered_pentatope_sphere(scale)
            squared = _squared_lengths(spacetime)
            hodge = _hodge(spacetime)
            gradient = hodge.spectral_entropy_gradient(1)
            contracted = hodge.spectral_entropy_gradient_directional_derivative(
                1, squared)
            negated = [-g for g in gradient]
            with self.subTest(scale=scale):
                self.assertLessEqual(
                    _relative_sup(contracted, negated, _scale_of(gradient)),
                    EXACT)


class EntropyHessianLinearityTest(unittest.TestCase):
    """Contraction against a direction is linear in that direction."""

    def test_additive_and_homogeneous_in_the_direction(self):
        spacetime = _jittered_pentatope_sphere()
        hodge = _hodge(spacetime)
        count = len(spacetime.get_edge_list().to_vector())
        first = [complex(0.3 + 0.1 * (i % 3), -0.2 * (i % 2)) for i in range(count)]
        second = [complex(-0.17 * (i % 4), 0.29 + 0.04 * (i % 5))
                  for i in range(count)]
        combined = [a + b for a, b in zip(first, second)]
        for degree in (1, 2):
            with self.subTest(degree=degree):
                left = hodge.spectral_entropy_gradient_directional_derivative(
                    degree, first)
                right = hodge.spectral_entropy_gradient_directional_derivative(
                    degree, second)
                total = hodge.spectral_entropy_gradient_directional_derivative(
                    degree, combined)
                summed = [a + b for a, b in zip(left, right)]
                self.assertLessEqual(
                    _relative_sup(total, summed, _scale_of(total)), EXACT)

    def test_zero_direction_gives_zero(self):
        spacetime = _jittered_pentatope_sphere()
        hodge = _hodge(spacetime)
        count = len(spacetime.get_edge_list().to_vector())
        contracted = hodge.spectral_entropy_gradient_directional_derivative(
            1, [complex(0.0, 0.0)] * count)
        for value in contracted:
            self.assertLessEqual(abs(value), EXACT)


class EntropyHessianShapeContractTest(unittest.TestCase):
    """A mis-sized direction is refused loudly, never silently padded."""

    def test_wrong_direction_length_raises(self):
        spacetime = _jittered_pentatope_sphere()
        hodge = _hodge(spacetime)
        count = len(spacetime.get_edge_list().to_vector())
        with self.assertRaises(RuntimeError):
            hodge.spectral_entropy_gradient_directional_derivative(
                1, [complex(1.0, 0.0)] * (count - 1))
        with self.assertRaises(RuntimeError):
            hodge.spectral_entropy_gradient_directional_derivative(
                1, [complex(1.0, 0.0)] * (count + 1))

    def test_negative_degree_raises(self):
        spacetime = _jittered_pentatope_sphere()
        hodge = _hodge(spacetime)
        squared = _squared_lengths(spacetime)
        with self.assertRaises(RuntimeError):
            hodge.spectral_entropy_gradient_directional_derivative(-1, squared)

    def test_result_length_matches_the_gradient(self):
        spacetime = _jittered_pentatope_sphere()
        hodge = _hodge(spacetime)
        squared = _squared_lengths(spacetime)
        for degree in (0, 1, 2, 3):
            with self.subTest(degree=degree):
                self.assertEqual(
                    len(hodge.spectral_entropy_gradient_directional_derivative(
                        degree, squared)),
                    len(hodge.spectral_entropy_gradient(degree)))


class EntropyHessianAscentDirectionTest(unittest.TestCase):
    """The contraction stage 2 actually performs: the direction is conj(h)."""

    def test_ascent_contraction_is_finite_and_nonzero(self):
        spacetime = _jittered_pentatope_sphere()
        for name, mode in MODES:
            for degree in (1, 2):
                with self.subTest(mode=name, degree=degree):
                    hodge = _hodge(spacetime)
                    gradient = hodge.spectral_entropy_gradient(degree, mode)
                    ascent = [g.conjugate() for g in gradient]
                    contracted = (
                        hodge.spectral_entropy_gradient_directional_derivative(
                            degree, ascent, mode))
                    self.assertTrue(all(math.isfinite(v.real) and
                                        math.isfinite(v.imag)
                                        for v in contracted))
                    self.assertGreater(_scale_of(contracted), 0.0)


class EntropyHessianConnectionBlindnessTest(unittest.TestCase):
    """The diagonal-weight HVP is exactly blind to the C* connection phase; the
    default operator's is not.

    The diagonal-weight `laplacian(k)` is built from the complex squared
    lengths alone, so its entropy, gradient and Hessian-vector product are
    BITWISE unchanged by an arbitrary complex phase on every edge. The default
    operator is the covariant `h_k(z, U)` of the Whitney pencil, which sees the
    connection at every degree: the same phase moves its entropy gradient and
    Hessian-vector product, and the size of that move is recorded below.
    """

    def test_the_default_operator_sees_the_phase(self):
        spacetime = _jittered_pentatope_sphere()
        edges = spacetime.get_edge_list().to_vector()
        squared = _squared_lengths(spacetime)
        for edge in edges:
            edge.set_phase(complex(0.0, 0.0))
        baseline = {}
        for name, mode in MODES:
            for degree in (0, 1, 2):
                hodge = cob.HodgeLaplacian(spacetime)
                baseline[(name, degree)] = (
                    hodge.spectral_entropy(degree, mode),
                    hodge.spectral_entropy_gradient(degree, mode),
                    hodge.spectral_entropy_gradient_directional_derivative(
                        degree, squared, mode))
        for index, edge in enumerate(edges):
            edge.set_phase(complex(0.31 * (index % 5) - 0.6,
                                  0.17 * (index % 3) - 0.2))
        for name, mode in MODES:
            for degree in (0, 1, 2):
                with self.subTest(mode=name, degree=degree):
                    hodge = cob.HodgeLaplacian(spacetime)
                    entropy = hodge.spectral_entropy(degree, mode)
                    gradient = hodge.spectral_entropy_gradient(degree, mode)
                    contracted = (
                        hodge.spectral_entropy_gradient_directional_derivative(
                            degree, squared, mode))
                    before_entropy, before_gradient, before_contracted = (
                        baseline[(name, degree)])
                    # The moves relative to the unphased values are asserted
                    # above a floor well below the measured ones (recorded in
                    # the pull request that inverted this assertion).
                    self.assertGreater(
                        abs(entropy - before_entropy) / max(abs(before_entropy), 1e-300),
                        1e-4)
                    self.assertGreater(
                        _relative_sup(gradient, before_gradient, _scale_of(before_gradient)),
                        1e-3)
                    self.assertGreater(
                        _relative_sup(contracted, before_contracted, _scale_of(before_contracted)),
                        1e-3)

    def test_an_arbitrary_complex_phase_changes_nothing_bitwise(self):
        spacetime = _jittered_pentatope_sphere()
        edges = spacetime.get_edge_list().to_vector()
        squared = _squared_lengths(spacetime)
        for edge in edges:
            edge.set_phase(complex(0.0, 0.0))

        baseline = {}
        for name, mode in MODES:
            for degree in (0, 1, 2):
                hodge = _hodge(spacetime)
                baseline[(name, degree)] = (
                    hodge.spectral_entropy_gradient(degree, mode),
                    hodge.spectral_entropy_gradient_directional_derivative(
                        degree, squared, mode))

        # A genuinely C* phase: a compact U(1) part AND a non-compact R+ part.
        for index, edge in enumerate(edges):
            edge.set_phase(complex(0.31 * (index % 5) - 0.6,
                                  0.17 * (index % 3) - 0.2))

        for name, mode in MODES:
            for degree in (0, 1, 2):
                with self.subTest(mode=name, degree=degree):
                    hodge = _hodge(spacetime)
                    gradient = hodge.spectral_entropy_gradient(degree, mode)
                    contracted = (
                        hodge.spectral_entropy_gradient_directional_derivative(
                            degree, squared, mode))
                    expected_gradient, expected_contracted = baseline[
                        (name, degree)]
                    self.assertEqual(list(gradient), list(expected_gradient))
                    self.assertEqual(list(contracted),
                                     list(expected_contracted))

    def test_the_euler_identity_still_holds_under_a_phase(self):
        spacetime = _jittered_pentatope_sphere()
        for index, edge in enumerate(spacetime.get_edge_list().to_vector()):
            edge.set_phase(complex(0.4 * (index % 4), -0.25 * (index % 3)))
        squared = _squared_lengths(spacetime)
        for name, mode in MODES:
            for degree in (0, 1, 2):
                with self.subTest(mode=name, degree=degree):
                    hodge = _hodge(spacetime)
                    gradient = hodge.spectral_entropy_gradient(degree, mode)
                    contracted = (
                        hodge.spectral_entropy_gradient_directional_derivative(
                            degree, squared, mode))
                    negated = [-g for g in gradient]
                    self.assertLessEqual(
                        _relative_sup(contracted, negated, _scale_of(gradient)),
                        EXACT)


class EntropyHessianFiniteDifferenceCrossCheckTest(unittest.TestCase):
    """A roundoff-limited cross-check ONLY.

    The central difference is the APPROXIMATE side; the analytic value is
    exact. Its agreement here to ~1e-8 is the accuracy of the retired
    finite-difference direction, not the accuracy of the analytic one — which
    the Euler identity above pins some four decades tighter.
    """

    def test_agrees_with_the_retired_central_difference(self):
        spacetime = _jittered_pentatope_sphere()
        edges = spacetime.get_edge_list().to_vector()
        base = _squared_lengths(spacetime)

        def set_squared(values):
            for edge, value in zip(edges, values):
                edge.set_length(cmath.sqrt(value))

        for name, mode in MODES:
            for degree in (1, 2):
                with self.subTest(mode=name, degree=degree):
                    set_squared(base)
                    gradient = _hodge(
                        spacetime).spectral_entropy_gradient(degree, mode)
                    direction = [g.conjugate() for g in gradient]
                    set_squared(base)
                    exact = _hodge(
                        spacetime).spectral_entropy_gradient_directional_derivative(
                            degree, direction, mode)

                    norm = math.sqrt(sum(abs(v) ** 2 for v in direction))
                    base_norm = math.sqrt(sum(abs(v) ** 2 for v in base))
                    step = (math.pow(2.220446049250313e-16, 1.0 / 3.0) *
                            max(base_norm, 1.0) / norm)
                    set_squared([b + step * d for b, d in zip(base, direction)])
                    plus = _hodge(
                        spacetime).spectral_entropy_gradient(degree, mode)
                    set_squared([b - step * d for b, d in zip(base, direction)])
                    minus = _hodge(
                        spacetime).spectral_entropy_gradient(degree, mode)
                    set_squared(base)

                    approximated = [(p - m) / (2.0 * step)
                                    for p, m in zip(plus, minus)]
                    self.assertLessEqual(
                        _relative_sup(exact, approximated, _scale_of(exact)),
                        1e-6)


if __name__ == "__main__":
    unittest.main()
