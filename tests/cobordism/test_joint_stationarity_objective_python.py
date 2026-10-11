# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Objective selection and Stage-2 optimization in z=l^2 coordinates."""

import cmath
import math
import unittest

import numpy as np

import tessera as T


cob = T.cobordism


def _complex_sphere4():
    st = T.Spacetime(T.Metric(True, T.Signature(4, T.Lorentzian)),
                     T.CDT, 1.0, 1.0, T.PREFERRED,
                     T.SimplexBoundarySphere(4))
    st.build()
    for index, edge in enumerate(st.get_edge_list().to_vector()):
        z = complex(1.0 + 0.019 * (index % 5),
                    0.011 * (1 + index % 4))
        edge.set_length(cmath.sqrt(z))
    return st


def _node(st, degree=3, gamma=0.0):
    return cob.MultiCobordism(st, [], [], degrees=[degree], gamma=gamma,
                              seed=7)


class ObjectiveSelectionTest(unittest.TestCase):
    def test_joint_objective_is_two_stationarity_residuals(self):
        st = _complex_sphere4()
        node = _node(st)
        node.set_objective(cob.JointStationarityObjective())
        node.set_regge_weight(0.7)
        node.set_hodge_entropy_weight(1.9)
        expected = (0.7 * cob.MultiCobordism.regge_action_gradient(st)
                    + 1.9 * node.hodge_entropy_stationarity())
        self.assertAlmostEqual(node.objective(), expected, places=10)
        self.assertTrue(math.isfinite(node.hodge_entropy()))

    def test_joint_objective_is_independent_of_target_and_gamma(self):
        first = cob.MultiCobordism(
            _complex_sphere4(), [], [[1.0]], degrees=[3], gamma=2.0, seed=7)
        second = cob.MultiCobordism(
            _complex_sphere4(), [], [[1.0, -1.0, 1j]], degrees=[3],
            gamma=9000.0, seed=7)
        for node in (first, second):
            node.set_objective(cob.JointStationarityObjective())
            node.set_regge_weight(0.7)
            node.set_hodge_entropy_weight(1.9)
        self.assertNotAlmostEqual(first.r_u(first.st), second.r_u(second.st))
        self.assertAlmostEqual(first.objective(), second.objective(), places=10)

    def test_mediated_objective_matches_documented_experiment(self):
        st = _complex_sphere4()
        node = _node(st)
        node.set_objective(cob.MediatedCorrespondenceObjective())
        node.set_regge_weight(0.23)
        expected = (node.r_u(st)
                    + 0.23 * abs(T.ReggeSolver(
                        st, T.MatterConfiguration()).dual_regge_action()))
        self.assertAlmostEqual(node.objective(), expected, places=10)

    def test_joint_mode_rejects_degree_zero_gradient(self):
        node = _node(_complex_sphere4(), degree=0)
        with self.assertRaises(ValueError):
            node.set_objective(cob.JointStationarityObjective())

    def test_zero_weights_disable_both_joint_terms(self):
        node = _node(_complex_sphere4())
        node.set_objective(cob.JointStationarityObjective())
        node.set_hodge_entropy_weight(0.0)
        node.set_regge_weight(0.0)
        original_lengths = [complex(edge.get_length())
                            for edge in node.st.get_edge_list().to_vector()]

        self.assertEqual(node.objective(), 0.0)
        trace = node.run_stage2(beta=0.0, max_iters=1, alpha0=0.05,
                                tolerance=1e-14)

        self.assertEqual(trace, [0.0])
        self.assertTrue(node.last_stage2_stationary)
        self.assertEqual(
            [complex(edge.get_length())
             for edge in node.st.get_edge_list().to_vector()],
            original_lengths)

    def test_entropy_only_stage_two_is_variational(self):
        node = _node(_complex_sphere4())
        node.set_objective(cob.JointStationarityObjective())
        node.set_hodge_entropy_weight(1.0)
        trace = node.run_stage2(beta=0.0, max_iters=1, alpha0=1e-3,
                                tolerance=1e-14)
        self.assertTrue(all(math.isfinite(value) for value in trace))
        if len(trace) == 2:
            self.assertLess(trace[-1], trace[0])
        else:
            self.assertTrue(node.last_stage2_stationary)


class ComplexSquaredCoordinateStepTest(unittest.TestCase):
    def test_regge_step_subtracts_from_z_with_a_real_step_scale(self):
        st = _complex_sphere4()
        node = _node(st, gamma=0.0)
        node.set_objective(cob.LegacyObjective())
        edges = st.get_edge_list().to_vector()
        original_lengths = np.asarray([complex(e.get_length()) for e in edges])
        z0 = original_lengths ** 2

        solver = T.ReggeSolver(st, T.MatterConfiguration())
        gradient = np.asarray(solver.action_gradient_exact(), complex)
        hessian = np.asarray(solver.action_hessian_exact(), complex)
        ascent = 2.0 * (np.conj(hessian) @ gradient)

        alpha0 = 1e-5
        trace = node.run_stage2(beta=1.0, max_iters=1, alpha0=alpha0,
                                tolerance=1e-14)
        self.assertEqual(len(trace), 2, "fixture should accept one descent step")
        self.assertLess(trace[-1], trace[0])
        actual_lengths = np.asarray(
            [complex(e.get_length()) for e in node.st.get_edge_list().to_vector()])
        actual_z = actual_lengths ** 2

        errors = []
        for backoff in range(24):
            expected_z = z0 - (alpha0 * 0.5 ** backoff) * ascent
            errors.append(np.max(np.abs(actual_z - expected_z)))
        scale = max(float(np.max(np.abs(actual_z))), 1.0)
        self.assertLess(min(errors) / scale, 2e-11)

        # The square-root choice is branch-continuous relative to the resident l.
        self.assertTrue(np.all(np.abs(actual_lengths - original_lengths)
                               <= np.abs(-actual_lengths - original_lengths)
                               + 1e-14))
        self.assertGreater(float(np.max(np.abs(actual_z.imag))), 0.0)

    def test_entropy_stationarity_step_uses_its_complex_hessian_vector_product(self):
        st = _complex_sphere4()
        node = _node(st)
        node.set_objective(cob.JointStationarityObjective())
        node.set_hodge_entropy_weight(1.0)
        # The Hodge degrees are DECLARED (#859) rather than inherited from the
        # register degrees, so the degree this test builds its finite-difference
        # reference at is stated explicitly.
        node.set_hodge_degrees([3])
        mode = cob.HodgeEntropyPhaseMode.IncludeComplexPhase
        node.set_hodge_entropy_phase_mode(mode)
        edges = st.get_edge_list().to_vector()
        original_lengths = np.asarray([complex(e.get_length()) for e in edges])
        z0 = original_lengths ** 2

        h0 = np.asarray(
            cob.HodgeLaplacian(st).spectral_entropy_gradient(3, mode), complex)
        entropy_ascent = np.conj(h0)
        hvp_step = (np.cbrt(np.finfo(float).eps) * max(np.linalg.norm(z0), 1.0)
                    / np.linalg.norm(entropy_ascent))

        def set_z(values):
            for index, (edge, value) in enumerate(zip(edges, values)):
                root = cmath.sqrt(value)
                if abs(-root - original_lengths[index]) < abs(
                        root - original_lengths[index]):
                    root = -root
                edge.set_length(root)

        set_z(z0 + hvp_step * entropy_ascent)
        h_plus = np.asarray(
            cob.HodgeLaplacian(st).spectral_entropy_gradient(3, mode), complex)
        set_z(z0 - hvp_step * entropy_ascent)
        h_minus = np.asarray(
            cob.HodgeLaplacian(st).spectral_entropy_gradient(3, mode), complex)
        for edge, length in zip(edges, original_lengths):
            edge.set_length(length)
        ascent = 2.0 * np.conj((h_plus - h_minus) / (2.0 * hvp_step))

        alpha0 = 1e-5
        trace = node.run_stage2(beta=0.0, max_iters=1, alpha0=alpha0,
                                tolerance=1e-14)
        self.assertEqual(len(trace), 2, "fixture should accept one entropy step")
        actual_z = np.asarray(
            [complex(e.get_length()) ** 2
             for e in node.st.get_edge_list().to_vector()])
        errors = [
            np.max(np.abs(actual_z - (z0 - alpha0 * 0.5 ** backoff * ascent)))
            for backoff in range(24)
        ]
        self.assertLess(min(errors) / max(np.max(np.abs(actual_z)), 1.0),
                        2e-10)


if __name__ == "__main__":
    unittest.main()
