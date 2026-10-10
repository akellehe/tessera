# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""
Tests for :class:`tessera.ModularityOptimizer`.
"""
import math
import unittest
import tessera


def _make_cdt(d=4, n_simplices=200):
    sig = tessera.Signature(d, tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    st = tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0,
                           tessera.PREFERRED, tessera.Toroid())
    st.build(n_simplices)
    cdt = tessera.CDTSimulation(st, 2.2, 0.5, 0.6, 0.02, st.get_n41())
    return cdt, st


def _default_config(target_dq=0.05, max_iter=80, M=4):
    cfg = tessera.ModularityOptimizerConfig()
    cfg.target_dq = target_dq
    cfg.max_iterations = max_iter
    cfg.n_diffusion_walks = 20
    cfg.max_sigma = 50.0
    cfg.target_n_modules = M
    return cfg


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------


class TestOptimizerConfig(unittest.TestCase):

    def test_defaults_sane(self):
        cfg = tessera.ModularityOptimizerConfig()
        self.assertGreater(cfg.target_dq, 0)
        self.assertGreater(cfg.max_iterations, 0)
        self.assertGreater(cfg.target_n_modules, 0)

    def test_writable(self):
        cfg = tessera.ModularityOptimizerConfig()
        cfg.target_dq = 0.1
        cfg.max_iterations = 50
        cfg.target_n_modules = 3
        self.assertEqual(cfg.target_dq, 0.1)
        self.assertEqual(cfg.max_iterations, 50)
        self.assertEqual(cfg.target_n_modules, 3)


# ---------------------------------------------------------------------------
# sweep() basic invariants
# ---------------------------------------------------------------------------


class TestSweepInvariants(unittest.TestCase):

    def test_sweep_up_returns_measurements(self):
        cdt, st = _make_cdt()
        opt = tessera.ModularityOptimizer(_default_config(), seed=0)
        meas = opt.sweep(cdt, "up")
        self.assertGreater(len(meas), 0)
        self.assertEqual(meas[0].direction, "up")
        self.assertEqual(meas[0].iter, 0)

    def test_sweep_down_returns_measurements(self):
        cdt, st = _make_cdt()
        opt = tessera.ModularityOptimizer(_default_config(), seed=0)
        meas = opt.sweep(cdt, "down")
        self.assertGreater(len(meas), 0)
        self.assertEqual(meas[0].direction, "down")

    def test_sweep_invalid_direction_returns_empty(self):
        cdt, st = _make_cdt()
        opt = tessera.ModularityOptimizer(_default_config(), seed=0)
        meas = opt.sweep(cdt, "sideways")
        self.assertEqual(len(meas), 0)

    def test_measurement_fields_finite(self):
        cdt, st = _make_cdt()
        opt = tessera.ModularityOptimizer(_default_config(), seed=0)
        for m in opt.sweep(cdt, "up"):
            self.assertTrue(math.isfinite(m.Q),
                            f"Q={m.Q} not finite")
            self.assertGreater(m.n_vertices, 0)
            self.assertGreater(m.n_edges, 0)
            self.assertGreater(m.n_simplices, 0)
            self.assertGreaterEqual(m.iter, 0)


class TestSweepUpDirection(unittest.TestCase):
    """Up-sweep: Q is non-decreasing across measurements."""

    def test_up_sweep_Q_non_decreasing(self):
        cdt, st = _make_cdt()
        opt = tessera.ModularityOptimizer(_default_config(), seed=0)
        meas = opt.sweep(cdt, "up")
        if len(meas) < 2:
            self.skipTest("Not enough measurements to test direction")
        for i in range(1, len(meas)):
            self.assertGreaterEqual(
                meas[i].Q, meas[i-1].Q - 1e-9,
                f"up sweep Q decreased: {meas[i-1].Q} → {meas[i].Q}"
            )

    def test_down_sweep_Q_non_increasing(self):
        cdt, st = _make_cdt()
        opt = tessera.ModularityOptimizer(_default_config(), seed=0)
        meas = opt.sweep(cdt, "down")
        if len(meas) < 2:
            self.skipTest("Not enough measurements to test direction")
        for i in range(1, len(meas)):
            self.assertLessEqual(
                meas[i].Q, meas[i-1].Q + 1e-9,
                f"down sweep Q increased: {meas[i-1].Q} → {meas[i].Q}"
            )


class TestProgressCallback(unittest.TestCase):
    """Optional progress callback fires at least once."""

    def test_progress_callback_fires(self):
        cdt, st = _make_cdt()
        opt = tessera.ModularityOptimizer(_default_config(), seed=0)
        calls = []
        opt.sweep(cdt, "up",
                  progress=lambda it, mi, q, n: calls.append(
                      (it, mi, q, n)))
        self.assertGreater(len(calls), 0,
                           "Progress callback never fired")


class TestEarlyExit(unittest.TestCase):
    """Up-sweep early-exit: stop when Q is within epsilon of 1 - 1/M."""

    def test_early_exit_bounds_iterations(self):
        cdt, st = _make_cdt()
        cfg = _default_config(target_dq=0.5, max_iter=2000, M=4)
        # Aggressive epsilon so early exit fires even on short
        # sweeps.
        cfg.epsilon_q_max = 0.5
        opt = tessera.ModularityOptimizer(cfg, seed=0)
        meas = opt.sweep(cdt, "up")
        # Should not have used the full 2000 iterations.
        self.assertLess(meas[-1].iter, 2000)


if __name__ == "__main__":
    unittest.main()
