# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The C++ MultiCobordism engine — deterministic objective core + emergent run (#491, #524).

Originally a parity oracle against the Python `emergent_optimizer` prototype; that prototype
has been retired (the C++ engine is the source of truth), so the deterministic objective core
(betti, emergent_holes, regge_action_gradient, r_state, objective) is now pinned to **golden
constants** captured from the engine on a fixed host — a regression guard against C++ drift.
The slow tests exercise the emergent two-stage run (the b₃ register grows), the canonical
two-step Proton, and a CobordismDAG chaining merges output->input.
"""
import cmath
import math
import os
import sys
import unittest

import pytest

import tessera
from tessera import cobordism as cob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _closed_s4 import closed_s4 as _closed_s4  # noqa: E402  (the shared host fixture)

T = tessera
cob = tessera.cobordism


class MultiCobordismCxxTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.CXX = cob.MultiCobordism
        cls.w = cmath.exp(2j * math.pi / 3)
        cls.host = _closed_s4(n_refine=20, seed=3)   # 86 cells, 26 verts, closed S⁴

    def test_deterministic_objective_core_is_stable(self):
        # Golden constants captured from the trusted C++ engine on _closed_s4(20, 3); a
        # regression guard that the deterministic objective core does not drift. The host is
        # a closed S⁴ with no removed cells, so there are no emergent holes at k=3 and the
        # singlet r_state is the full zero-filled leak (|1|² + |ω|² + |ω²|² = 3). The values
        # are exact run-to-run; the FP tolerance only covers cross-machine round-off.
        CXX, w = self.CXX, self.w
        tgt = [1, w, w * w]
        self.assertEqual(list(CXX.betti(self.host)), [1, 0, 0, 0, 1])       # closed S⁴
        self.assertEqual(list(CXX.emergent_holes(self.host, 3)), [])        # no holes (closed)
        self.assertAlmostEqual(CXX.regge_action_gradient(self.host),
                               499.9710921237928, places=6)
        self.assertAlmostEqual(CXX.r_state(self.host, 3, tgt), 3.0, places=10)  # the leak
        obj = CXX(self.host, [[1, w, w * w], [1, w * w, w]], [tgt], degrees=[3],
                  gamma=1.0, seed=0,
                  metric_source=cob.HodgeMetricSource.DiagonalWeights).objective()
        # 502.9710921237928 before #644; the near-kernel residual — the
        # pre-topological register signal, +0.0093451402 on this closed S⁴ at
        # k=3 for a 3-component target, computed on the METRIC L_3 (geometric
        # by design: the causal-tuning channel is intended) — enters r_u. The
        # decomposition: regge_action_gradient 499.9710921 + r_u (3.0 leak
        # + 0.0093451 near-kernel).
        self.assertAlmostEqual(obj, 502.9804372639758, places=6)
        # Under the chain-level Whitney pencil the
        # near-kernel term of the same closed S⁴ reads +0.0044286: the pencil's
        # L_3 couples cells sharing a top simplex, so its lowest singular values
        # differ from the diagonal-weight operator's (#910 findings note).
        obj_whitney = CXX(self.host, [[1, w, w * w], [1, w * w, w]], [tgt], degrees=[3],
                          gamma=1.0, seed=0,
                          metric_source=cob.HodgeMetricSource.WhitneyPencil).objective()
        self.assertAlmostEqual(obj_whitney, 502.9755207221824, places=6)

    # Marked slow: the emergent b3 register has to grow through both stages;
    # about three and a quarter minutes.
    @pytest.mark.slow
    def test_two_stage_grows_emergent_register(self):
        # The two-stage emergent run grows a b₃ color register out of the closed-S⁴ host.
        # run_stage1's greedy ΔF tie-breaks read FP values from the OpenMP-reduced Regge
        # gradient (summation order varies run-to-run), so a single (seed, budget) can —
        # even at a fixed RNG seed — occasionally net no register within the budget. The
        # PROPERTY under test is that the run grows a register, not that one fixed seed does
        # so every run; so try a few surgery seeds with a healthy budget and require one to
        # emerge (short-circuits on the first seed that grows one).
        CXX, w = self.CXX, self.w
        self.assertEqual(list(CXX.betti(_closed_s4(n_refine=20, seed=3)))[4], 1)  # bare S⁴
        grew = False
        for seed in range(3, 11):
            host = _closed_s4(n_refine=20, seed=3)
            opt = CXX(host, [[1, w, w * w], [1, w * w, w]], [[1, w, w * w]], degrees=[3],
                      gamma=1.0, seed=seed)
            sv = [v.getId() for v in host.getVertexList().toVector()][:2]
            opt.seed_inputs(sv)
            opt.run_stage1(max_steps=25, n_candidate_moves=8)
            if list(CXX.betti(opt.st))[3] >= 1:
                grew = True
                break
        self.assertTrue(grew, "no b₃ register emerged across surgery seeds 3..10")

    def test_run_stage2_stops_on_absolute_stationarity(self):
        # run_stage2 stops when no line-search step lowers F by the absolute
        # tolerance, and last_stage2_stationary distinguishes that result (True)
        # from the max_iters budget cap (False).
        CXX, w = self.CXX, self.w
        host = _closed_s4(n_refine=12, seed=3)
        opt = CXX(host, [[1, w, w * w], [1, w * w, w]], [[1, w, w * w]],
                  degrees=[3], gamma=1.0, seed=3)
        opt.seed_inputs([v.getId() for v in host.getVertexList().toVector()][:2])

        # Budget cap: one iteration under a tight tol on the fresh, jittered (non-
        # stationary) geometry takes a single improving step and stops on the iteration
        # budget — NOT the stationarity test. last_stage2_stationary is False.
        t_budget = opt.run_stage2(beta=1.0, max_iters=1, alpha0=0.05, tolerance=1e-13)
        self.assertFalse(opt.last_stage2_stationary)         # stopped: budget
        self.assertLess(t_budget[-1], t_budget[0])           # the step strictly lowered F

        # Stationary stop: with an absolute threshold wider than any achievable
        # decrease, the first line search accepts nothing and reports stationarity
        # before exhausting max_iters.
        t_stat = opt.run_stage2(beta=1.0, max_iters=50, alpha0=0.05, tolerance=10.0)
        self.assertTrue(opt.last_stage2_stationary)          # stopped: stationary
        self.assertLess(len(t_stat), 51)                     # broke before the budget cap
        self.assertTrue(all(t_stat[i + 1] <= t_stat[i]       # never increases
                            for i in range(len(t_stat) - 1)))

    def test_two_step_proton_via_canonical_class(self):
        # The canonical two-step proton build goes through tessera.cobordism.Proton (Step A
        # recombination -> a *colored* diquark, Step B formation -> the color singlet). A fast
        # smoke that both steps run end-to-end and expose the 3-vector proton singlet; the
        # thorough convergence test lives in tests/cobordism/test_proton_cpp_python.py.
        Proton = cob.Proton
        self.assertEqual(len(Proton.singlet()), 3)            # the proton is a 3-vector
        p = Proton(seed=3)
        p.build(max_restarts=1, init_steps=8, evolve_steps=4,
                stage1_candidate_moves=4, stage2_max_iters=6,
                min_emergent_holes=1)
        # both steps ran: Step A (diquark recombination) and Step B (proton formation)
        self.assertTrue(math.isfinite(p.diquark_residual()))
        self.assertTrue(math.isfinite(p.color_residual()))
        # block() is the whole relaxed cobordism (None only if nothing emerged)
        block = p.block()
        if block is not None:
            self.assertGreater(len(block.getEdgeList().toVector()), 0)

    def test_recombination_two_in_two_out(self):
        # 2->2 recombination in ONE co-optimized node: 2 input pairs, 2 outputs.
        CXX, w = self.CXX, self.w
        host = _closed_s4(n_refine=14, seed=3)
        opt = CXX(host, [[1, -1, 0], [1, 0, -1]], [[1, w, w * w], [1, w * w, w]],
                  degrees=[3], gamma=1.0, seed=3)
        sv = [v.getId() for v in host.getVertexList().toVector()]
        opt.seed_inputs(sv[:2])
        opt.seed_outputs(sv[2:4])   # two output blocks, co-optimized
        opt.run_stage1(max_steps=6, n_candidate_moves=4)
        self.assertTrue(math.isfinite(opt.r_u(opt.st)))   # all 4 blocks scored, no crash

    # Marked slow: the DAG drives two output branches to completion; about one
    # minute.
    @pytest.mark.slow
    def test_dag_recombination_routes_two_outputs(self):
        # recombination node (2 outputs) -> two independent legs via output index.
        w = self.w
        dag = cob.CobordismDAG()
        hr = _closed_s4(n_refine=12, seed=3)
        hp = _closed_s4(n_refine=12, seed=5)
        ha = _closed_s4(n_refine=12, seed=6)
        rec = dag.add_node(hr, [[1, -1, 0], [1, 0, -1]],
                           [], [[1, w, w * w], [1, w * w, w]], degrees=[3], seed=3)
        pro = dag.add_node(hp, [[0, 1, -1]], [(rec, 0)], [[1, w, w * w]],
                           degrees=[3], seed=5)
        apr = dag.add_node(ha, [[0, -1, 1]], [(rec, 1)], [[1, w, w * w]],
                           degrees=[3], seed=6)
        dag.run(stage1_max_steps=6, stage1_candidate_moves=3,
                stage2_max_iters=6)
        self.assertEqual(dag.num_outputs(rec), 2)
        self.assertEqual(len(dag.output(rec, 0)), 3)   # CobordismDAG.output() threading
        for nd in (rec, pro, apr):
            self.assertTrue(math.isfinite(dag.residual(nd)))


if __name__ == "__main__":
    unittest.main()
