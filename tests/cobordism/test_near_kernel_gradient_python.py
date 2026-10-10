# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""The register residual's gradient (#748).

`near_kernel_residual` is `n · Σ(m smallest σ²) / Σ(all σ²)` of the metric `L_k`
— the register residual, and the only part of `r_U` that is smooth and
non-zero BEFORE a register exists. The other parts are period gaps, which sit
at their constant full leak until holes open, and a constant cannot move the
geometry. Until this it had no derivative at all.

Two things this file pins, both of which were got wrong first:

* **Holomorphy.** `L` is holomorphic in `ℓ²`, so `∂(L†L) = 2 L†∂L`. Building
  the Hermitian combination `∂L†L + L†∂L` instead gives a real `wᴴ∂Hw` by
  construction, so the gradient's imaginary part comes out identically zero —
  which defeats the entire purpose. The real parts agree either way, so only
  an off-locus imaginary-direction check catches it.
* **Degeneracy.** The subdivided-simplex fixture is highly symmetric and its
  spectrum has coincident `σ`; where the `m`-smallest cut splits a degenerate
  block the SELECTION is discontinuous and no derivative exists. The fixture is
  jittered to lift that, and the tests assert the gap is real before relying on
  it.
"""
import cmath
import importlib.util
import os
import unittest

import numpy as np

import tessera

cobordism = tessera.cobordism

_FIXTURE = os.path.join(os.path.dirname(__file__), "_b2_register.py")
_spec = importlib.util.spec_from_file_location("_b2_register", _FIXTURE)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
B2Register = _module.B2Register

_DEGREE = 2
_REGISTERS = 3


def _edges(spacetime):
    chain = cobordism.ChainComplex.from_spacetime(spacetime)
    byPair = {}
    for edge in spacetime.get_edge_list().to_vector():
        a, b = edge.get_source().get_id(), edge.get_target().get_id()
        byPair[(min(a, b), max(a, b))] = edge
    return [tuple(sorted(v)) for v in chain.k_simplex_vertices(1)], byPair


def _spectrumGapAtTheCut(spacetime):
    """`σ_{m+1} − σ_m` ascending: the derivative only exists when this is > 0."""
    flat = np.array(cobordism.HodgeLaplacian(spacetime).laplacian(_DEGREE, True))
    size = int(round(np.sqrt(flat.size)))
    ascending = np.linalg.svd(flat.reshape(size, size), compute_uv=False)[::-1]
    return ascending[_REGISTERS] - ascending[_REGISTERS - 1]


def _fixture(twist=0.0):
    spacetime, _holes = B2Register.build()
    pairs, byPair = _edges(spacetime)
    for index, pair in enumerate(pairs):          # lift spectral degeneracy
        edge = byPair[pair]
        edge.set_length(edge.get_length() * (1.0 + 0.11 * ((index * 37) % 13) / 13))
    if twist:
        for index, pair in enumerate(pairs):      # push l^2 off the real locus
            edge = byPair[pair]
            squared = edge.get_length() ** 2
            edge.set_length(cmath.sqrt(
                squared * complex(1.0, twist * (1 + index % 3) / 3)))
    spacetime.materialize_facets()
    return spacetime, pairs, byPair


class NearKernelGradientTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cobordism.HodgeLaplacian.set_default_weight_convention(
            cobordism.HodgeWeightConvention.SquaredContent)

    def _gradient(self, spacetime):
        return np.array(cobordism.MultiCobordism.near_kernel_residual_gradient(
            spacetime, _DEGREE, _REGISTERS), complex)

    def _value(self, spacetime):
        return cobordism.MultiCobordism.near_kernel_residual(
            spacetime, _DEGREE, _REGISTERS)

    def test_the_cut_is_not_degenerate(self):
        # Without this the "m smallest" selection is ambiguous and there is no
        # derivative to test against.
        spacetime, _pairs, _byPair = _fixture()
        self.assertGreater(_spectrumGapAtTheCut(spacetime), 1e-6)

    def test_scale_invariance_euler_identity_on_the_locus(self):
        spacetime, pairs, byPair = _fixture()
        gradient = self._gradient(spacetime)
        squared = np.array([byPair[p].get_length() ** 2 for p in pairs], complex)
        self.assertLess(abs(complex(np.dot(squared, gradient))), 1e-9)

    def test_scale_invariance_euler_identity_off_the_locus(self):
        # Both parts must vanish: the term is degree 0 in l^2, which is why it
        # is a ratio rather than a raw spectral sum.
        spacetime, pairs, byPair = _fixture(twist=0.35)
        gradient = self._gradient(spacetime)
        squared = np.array([byPair[p].get_length() ** 2 for p in pairs], complex)
        euler = complex(np.dot(squared, gradient))
        self.assertLess(abs(euler.real), 1e-9)
        self.assertLess(abs(euler.imag), 1e-9)

    def test_imaginary_part_is_not_identically_zero_off_the_locus(self):
        # The regression the holomorphic form fixes: the Hermitian combination
        # returns a purely real gradient, which cannot move the geometry in the
        # plane stage 2 steps in.
        spacetime, _pairs, _byPair = _fixture(twist=0.35)
        self.assertGreater(np.abs(self._gradient(spacetime).imag).max(), 1e-8)

    def test_both_directional_derivatives_match_finite_differences(self):
        spacetime, pairs, byPair = _fixture(twist=0.35)
        gradient = self._gradient(spacetime)
        moving = [i for i in range(len(pairs)) if abs(gradient[i]) > 1e-8][:3]
        self.assertTrue(moving, "gradient is everywhere zero; fixture is flat")
        for index in moving:
            edge = byPair[pairs[index]]
            base = edge.get_length() ** 2
            step = 1e-7
            for shift, expected in ((step, gradient[index].real),
                                    (1j * step, -gradient[index].imag)):
                edge.set_length(cmath.sqrt(base + shift))
                spacetime.materialize_facets()
                up = self._value(spacetime)
                edge.set_length(cmath.sqrt(base - shift))
                spacetime.materialize_facets()
                down = self._value(spacetime)
                edge.set_length(cmath.sqrt(base))
                spacetime.materialize_facets()
                finiteDifference = (up - down) / (2 * step)
                self.assertAlmostEqual(
                    finiteDifference, expected,
                    delta=1e-3 * max(abs(finiteDifference), 1e-6),
                    msg=f"edge {pairs[index]}, shift {shift}")

    def test_constant_cases_have_a_zero_gradient(self):
        spacetime, _pairs, _byPair = _fixture()
        # No registers expected: the value is the constant 0.
        zero = np.array(cobordism.MultiCobordism.near_kernel_residual_gradient(
            spacetime, _DEGREE, 0), complex)
        self.assertTrue(np.all(np.abs(zero) == 0.0))
        # Above the top dimension there are no k-cells: the value is constant.
        above = np.array(cobordism.MultiCobordism.near_kernel_residual_gradient(
            spacetime, 9, _REGISTERS), complex)
        self.assertTrue(np.all(np.abs(above) == 0.0))


if __name__ == "__main__":
    unittest.main()
