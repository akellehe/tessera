# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Positive-operator Hodge entropy and its complex-z gradient."""

import cmath
import math
import unittest

import numpy as np

import tessera as T


cob = T.cobordism


def _complex_sphere4():
    metric = T.Metric(True, T.Signature(4, T.Lorentzian))
    st = T.Spacetime(metric, T.CDT, 1.0, 1.0, T.PREFERRED,
                     T.SimplexBoundarySphere(4))
    st.build()
    for index, edge in enumerate(st.get_edge_list().to_vector()):
        z = complex(1.0 + 0.021 * (index % 5),
                    0.017 * (1 + (index % 3)))
        edge.set_length(cmath.sqrt(z))
    return st


def _entropy_oracle(st, degree, ignore_phase):
    flat = np.asarray(cob.HodgeLaplacian(st).laplacian(degree), complex)
    n = int(round(math.sqrt(flat.size)))
    L = flat.reshape(n, n)
    M = np.abs(L) if ignore_phase else L
    eigenvalues = np.linalg.eigvalsh(M.conj().T @ M)
    trace = float(eigenvalues.sum())
    if trace <= 0.0:
        return 0.0
    cutoff = np.finfo(float).eps * max(n, 1) * max(trace, 1.0) * 64.0
    p = eigenvalues[eigenvalues > cutoff] / trace
    return float(-(p * np.log(p)).sum())


class HodgeEntropyTest(unittest.TestCase):
    def test_positive_operator_entropy_matches_numpy(self):
        st = _complex_sphere4()
        hl = cob.HodgeLaplacian(st)
        modes = (
            (cob.HodgeEntropyPhaseMode.IncludeComplexPhase, False),
            (cob.HodgeEntropyPhaseMode.IgnoreComplexPhase, True),
        )
        for mode, ignore in modes:
            with self.subTest(mode=mode):
                measured = float(hl.spectral_entropy(3, mode))
                expected = _entropy_oracle(st, 3, ignore)
                self.assertAlmostEqual(measured, expected, places=11)
                self.assertGreaterEqual(measured, 0.0)
                self.assertLessEqual(measured, math.log(hl.laplacian(3).__len__() ** 0.5)
                                     + 1e-12)

    def test_complex_z_gradient_matches_two_axis_finite_difference(self):
        st = _complex_sphere4()
        modes = (
            cob.HodgeEntropyPhaseMode.IncludeComplexPhase,
            cob.HodgeEntropyPhaseMode.IgnoreComplexPhase,
        )
        edges = st.get_edge_list().to_vector()
        step = 2e-6
        for mode in modes:
            gradient = np.asarray(
                cob.HodgeLaplacian(st).spectral_entropy_gradient(3, mode),
                dtype=complex)
            self.assertEqual(gradient.shape, (len(edges),))
            for edge_index in (0, 2):
                edge = edges[edge_index]
                original_length = complex(edge.get_length())
                z0 = original_length * original_length

                def value(z):
                    edge.set_length(cmath.sqrt(z))
                    return float(cob.HodgeLaplacian(st).spectral_entropy(3, mode))

                f_re_plus = value(z0 + step)
                f_re_minus = value(z0 - step)
                f_im_plus = value(z0 + 1j * step)
                f_im_minus = value(z0 - 1j * step)
                edge.set_length(original_length)

                fd_re = (f_re_plus - f_re_minus) / (2.0 * step)
                fd_im = (f_im_plus - f_im_minus) / (2.0 * step)
                scale = max(abs(fd_re), abs(fd_im), 1.0)
                self.assertLess(abs(gradient[edge_index].real - fd_re) / scale,
                                2e-5)
                self.assertLess(abs(-gradient[edge_index].imag - fd_im) / scale,
                                2e-5)

            expected_norm = float(np.vdot(gradient, gradient).real)
            measured_norm = float(
                cob.HodgeLaplacian(st).spectral_entropy_gradient_norm(3, mode))
            self.assertAlmostEqual(measured_norm, expected_norm, places=10)

    def test_degree_zero_entropy_and_gradient_are_both_available(self):
        # Degree zero used to throw here because the old magnitude-weighted
        # diagonal of D - A was non-holomorphic in z. L_0 = d_1 W_1^-1 d_1^dagger
        # is holomorphic like every other degree (#805), so the same complex-z
        # gradient applies -- FD-checked on both axes below.
        st = _complex_sphere4()
        hl = cob.HodgeLaplacian(st)
        self.assertTrue(math.isfinite(hl.spectral_entropy(0)))
        gradient = np.asarray(hl.spectral_entropy_gradient(0), dtype=complex)
        self.assertEqual(gradient.shape, (len(st.get_edge_list().to_vector()),))
        self.assertTrue(np.all(np.isfinite(gradient)))
        self.assertGreater(np.max(np.abs(gradient)), 0.0)

    def test_degree_zero_gradient_matches_two_axis_finite_difference(self):
        st = _complex_sphere4()
        edges = st.get_edge_list().to_vector()
        step = 2e-6
        for mode in (cob.HodgeEntropyPhaseMode.IncludeComplexPhase,
                     cob.HodgeEntropyPhaseMode.IgnoreComplexPhase):
            gradient = np.asarray(
                cob.HodgeLaplacian(st).spectral_entropy_gradient(0, mode),
                dtype=complex)
            for edge_index in (0, 3):
                edge = edges[edge_index]
                original_length = complex(edge.get_length())
                z0 = original_length * original_length

                def value(z):
                    edge.set_length(cmath.sqrt(z))
                    return float(cob.HodgeLaplacian(st).spectral_entropy(0, mode))

                f_re_plus = value(z0 + step)
                f_re_minus = value(z0 - step)
                f_im_plus = value(z0 + 1j * step)
                f_im_minus = value(z0 - 1j * step)
                edge.set_length(original_length)

                fd_re = (f_re_plus - f_re_minus) / (2.0 * step)
                fd_im = (f_im_plus - f_im_minus) / (2.0 * step)
                scale = max(abs(fd_re), abs(fd_im), 1.0)
                with self.subTest(mode=mode, edge=edge_index):
                    self.assertLess(
                        abs(gradient[edge_index].real - fd_re) / scale, 2e-5)
                    self.assertLess(
                        abs(-gradient[edge_index].imag - fd_im) / scale, 2e-5)

    def test_degree_zero_laplacian_gradient_matches_finite_difference(self):
        # The exact dL_0/dz underneath it: with W_0 = I the only surviving term
        # is -d_1 W_1^-1 (dW_1) W_1^-1 d_1^dagger.
        st = _complex_sphere4()
        cc = cob.ChainComplex.from_spacetime(st)
        n0 = cc.num_simplices(0)
        one_cells = cc.k_simplex_vertices(1)
        edges = {tuple(sorted((e.get_source().get_id(), e.get_target().get_id()))): e
                 for e in st.get_edge_list().to_vector()}
        step = 1e-6
        for cell in (one_cells[0], one_cells[4]):
            edge = edges[tuple(sorted(cell))]
            analytic = np.asarray(
                cob.HodgeLaplacian(st).laplacian_gradient(0, cell[0], cell[1]),
                dtype=complex).reshape(n0, n0)
            original_length = complex(edge.get_length())
            z0 = original_length * original_length

            def operator(z):
                edge.set_length(cmath.sqrt(z))
                return np.asarray(cob.HodgeLaplacian(st).laplacian(0),
                                  dtype=complex).reshape(n0, n0)

            fd = (operator(z0 + step) - operator(z0 - step)) / (2.0 * step)
            edge.set_length(original_length)
            with self.subTest(cell=tuple(cell)):
                self.assertLess(np.max(np.abs(analytic - fd)), 1e-6)


if __name__ == "__main__":
    unittest.main()
