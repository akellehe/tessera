# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""The degree-generic period-gap gradient (#630).

`periodGapForPeriodsGradient` used to route unconditionally through the
edge-loop core, which is degree-1 machinery and throws for `k >= 2` — so at the
proton's register degree the gradient of `r_U`'s whole-complex term did not
exist. It now routes by degree, exactly as `residualForPeriodsGradient` does.

The gap `r_psi = ||A c - t||^2` (with `A = Q U_n` and `c` the least-squares fit)
is homogeneous of degree ZERO: `L_k` is degree -1 in `l^2`, so a uniform rescale
sends `L -> L/s` and leaves the kernel — hence the normalized harmonic basis,
`A`, `c` and the gap — unchanged. That gives the Euler identity

    sum_e l^2_e d r_psi / d l^2_e = 0,

in contrast to `r_U`'s `-2 r_U`, and it certifies the gradient without a
finite-difference step size. Finite differences are checked too, as an
independent witness against a value computed outside the gradient code.
"""
import cmath
import importlib.util
import os
import unittest

import numpy as np

import tessera

cobordism = tessera.cobordism
# r_psi's contract is the diagonal-weight path's, named by its source: on the
# default Whitney pencil the harmonic representatives are closed cochains whose
# periods span the cohomology image, so the period gap is blind to the geometry
# (tests/cobordism/test_period_gap_python.py, TestWhitneyPeriodGapIsTopological).
DIAGONAL = cobordism.HodgeMetricSource.DiagonalWeights

_FIXTURE = os.path.join(os.path.dirname(__file__), "_b2_register.py")
_spec = importlib.util.spec_from_file_location("_b2_register", _FIXTURE)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
B2Register = _module.B2Register

_HOLED = os.path.join(os.path.dirname(__file__), "_holed_surface.py")
_hspec = importlib.util.spec_from_file_location("_holed_surface", _HOLED)
_hs = importlib.util.module_from_spec(_hspec)
_hspec.loader.exec_module(_hs)


def _edgeSquaredLengths(spacetime):
    """Edge `l^2` in ChainComplex 1-cell order — the gradient's own layout."""
    chain = cobordism.ChainComplex.fromSpacetime(spacetime)
    byPair = {}
    for edge in spacetime.getEdgeList().toVector():
        a, b = edge.getSource().getId(), edge.getTarget().getId()
        byPair[(min(a, b), max(a, b))] = edge
    pairs = [tuple(sorted(v)) for v in chain.kSimplexVertices(1)]
    return pairs, byPair, np.array(
        [(byPair[p].getLength() ** 2).real for p in pairs])


def _gapValue(spacetime, degree, holes, target):
    """The gap, computed OUTSIDE the gradient code: periods, least-squares fit,
    squared residual."""
    synthesis = cobordism.EigenstateSynthesis(spacetime, degree, DIAGONAL)
    flat = np.array(synthesis.cyclePeriods(holes), complex)
    periodMatrix = flat.reshape(-1, len(holes)).T
    targetVector = np.array(target, complex)
    coefficients, *_ = np.linalg.lstsq(periodMatrix, targetVector, rcond=None)
    return float(np.linalg.norm(periodMatrix @ coefficients - targetVector) ** 2)


class PeriodGapGradientDegreeTwoTest(unittest.TestCase):
    """The degree-generic core, on a complex that actually carries a b_2."""

    @classmethod
    def setUpClass(cls):
        cobordism.HodgeLaplacian.setDefaultWeightConvention(
            cobordism.HodgeWeightConvention.SquaredContent)
        cls.spacetime, cls.holes = B2Register.build()
        cls.degree = 2
        cls.target = [complex(1.0, 0.0), complex(0.3, 0.6)][:len(cls.holes)]
        cls.synthesis = cobordism.EigenstateSynthesis(cls.spacetime, cls.degree, DIAGONAL)
        # COMPLEX now (#746): Re is the real-locus derivative these tests
        # certified before, so they keep asserting exactly what they did.
        cls.gradientComplex = np.array(
            cls.synthesis.periodGapForPeriodsGradient(cls.holes, cls.target),
            complex)
        cls.gradient = cls.gradientComplex.real

    def test_the_fixture_is_not_degenerate(self):
        # The trap this fixture exists to avoid: holes that BOUND have periods
        # of ~1e-16, and a gradient checked against them proves nothing.
        self.assertEqual(list(cobordism.MultiCobordism.betti(self.spacetime))[2], 1)
        self.assertEqual(len(self.holes), 2)
        flat = np.array(self.synthesis.cyclePeriods(self.holes), complex)
        self.assertGreater(np.abs(flat).max(), 1e-3)
        self.assertGreater(
            _gapValue(self.spacetime, self.degree, self.holes, self.target), 1e-3)

    def test_gradient_is_not_trivially_zero(self):
        self.assertGreater(int((np.abs(self.gradient) > 1e-14).sum()), 0)

    def test_euler_identity_for_a_degree_zero_functional(self):
        _pairs, _byPair, squared = _edgeSquaredLengths(self.spacetime)
        self.assertEqual(len(squared), len(self.gradient))
        self.assertLess(abs(float(np.dot(squared, self.gradient))), 1e-9)

    def test_agrees_with_finite_differences(self):
        pairs, byPair, _squared = _edgeSquaredLengths(self.spacetime)
        moving = [i for i in range(len(pairs)) if abs(self.gradient[i]) > 1e-9][:4]
        self.assertTrue(moving, "no edge moves the gap; fixture is degenerate")
        for index in moving:
            edge = byPair[pairs[index]]
            length = edge.getLength()
            squaredLength = length * length
            step = 1e-6
            edge.setLength(cmath.sqrt(squaredLength + step))
            self.spacetime.materializeFacets()
            up = _gapValue(self.spacetime, self.degree, self.holes, self.target)
            edge.setLength(cmath.sqrt(squaredLength - step))
            self.spacetime.materializeFacets()
            down = _gapValue(self.spacetime, self.degree, self.holes, self.target)
            edge.setLength(length)
            self.spacetime.materializeFacets()
            finiteDifference = (up - down) / (2 * step)
            self.assertAlmostEqual(
                finiteDifference, self.gradient[index],
                delta=1e-5 * max(abs(finiteDifference), 1.0),
                msg=f"edge {pairs[index]}")


class ComplexGradientOffTheRealLocusTest(unittest.TestCase):
    """The reason the family is complex (#746).

    A balanced-edge run starts every interval at `l^2 = +-i*m`, so the states
    these gradients are read on are NOT on the real-`l^2` locus. A real-valued
    derivative describes one direction of a plane the objective moves in; the
    complex one describes both, and the degree-0 Euler identity has to hold in
    BOTH parts off the locus, which is a check a real gradient cannot even
    state.
    """

    @classmethod
    def setUpClass(cls):
        cobordism.HodgeLaplacian.setDefaultWeightConvention(
            cobordism.HodgeWeightConvention.SquaredContent)
        cls.spacetime, cls.holes = B2Register.build()
        cls.degree = 2
        cls.target = [complex(1.0, 0.0), complex(0.3, 0.6)][:len(cls.holes)]
        pairs, byPair, _squared = _edgeSquaredLengths(cls.spacetime)
        for index, pair in enumerate(pairs):          # push l^2 off the locus
            edge = byPair[pair]
            squared = edge.getLength() ** 2
            edge.setLength(cmath.sqrt(
                squared * complex(1.0, 0.35 * (1 + index % 3) / 3)))
        cls.spacetime.materializeFacets()
        cls.pairs, cls.byPair = pairs, byPair
        cls.gradient = np.array(
            cobordism.EigenstateSynthesis(cls.spacetime, cls.degree, DIAGONAL)
            .periodGapForPeriodsGradient(cls.holes, cls.target), complex)

    def test_the_state_is_actually_off_the_locus(self):
        imaginary = max(abs((self.byPair[p].getLength() ** 2).imag)
                        for p in self.pairs)
        self.assertGreater(imaginary, 1e-2)

    def test_the_gradient_has_a_nonzero_imaginary_part(self):
        self.assertGreater(np.abs(self.gradient.imag).max(), 1e-6)

    def test_euler_identity_holds_in_both_parts(self):
        squared = np.array([self.byPair[p].getLength() ** 2
                            for p in self.pairs], complex)
        euler = complex(np.dot(squared, self.gradient))
        self.assertLess(abs(euler.real), 1e-9)
        self.assertLess(abs(euler.imag), 1e-9)

    def test_both_directional_derivatives_match_finite_differences(self):
        """Re(g) is d/d(Re l^2) and -Im(g) is d/d(Im l^2)."""
        moving = [i for i in range(len(self.pairs))
                  if abs(self.gradient[i]) > 1e-9][:3]
        self.assertTrue(moving)
        for index in moving:
            edge = self.byPair[self.pairs[index]]
            base = edge.getLength() ** 2
            step = 1e-6
            for shift, expected in ((step, self.gradient[index].real),
                                    (1j * step, -self.gradient[index].imag)):
                edge.setLength(cmath.sqrt(base + shift))
                self.spacetime.materializeFacets()
                up = _gapValue(self.spacetime, self.degree, self.holes, self.target)
                edge.setLength(cmath.sqrt(base - shift))
                self.spacetime.materializeFacets()
                down = _gapValue(self.spacetime, self.degree, self.holes, self.target)
                edge.setLength(cmath.sqrt(base))
                self.spacetime.materializeFacets()
                finiteDifference = (up - down) / (2 * step)
                self.assertAlmostEqual(
                    finiteDifference, expected,
                    delta=1e-4 * max(abs(finiteDifference), 1.0),
                    msg=f"edge {self.pairs[index]}, shift {shift}")


class PeriodGapGradientRoutingTest(unittest.TestCase):
    """Degree routing, including the contract at k = 0."""

    def test_degree_one_still_uses_the_loop_core_and_satisfies_euler(self):
        cobordism.HodgeLaplacian.setDefaultWeightConvention(
            cobordism.HodgeWeightConvention.SquaredContent)
        spacetime, synthesis, holes, _periods = _hs.holed_surface(degree=1)
        target = [complex(1.0, 0.0)] * len(holes)
        gradient = np.array(
            synthesis.periodGapForPeriodsGradient(holes, target), complex).real
        _pairs, _byPair, squared = _edgeSquaredLengths(spacetime)
        self.assertLess(abs(float(np.dot(squared, gradient))), 1e-9)

    def test_degree_zero_has_no_period_gap_core(self):
        spacetime, _synthesis, holes, periods = _hs.holed_surface(degree=1)
        atDegreeZero = cobordism.EigenstateSynthesis(spacetime, 0, DIAGONAL)
        target = [complex(z) for z in periods[0]]
        with self.assertRaises(RuntimeError) as raised:
            atDegreeZero.periodGapForPeriodsGradient(holes, target)
        self.assertIn("degree", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
