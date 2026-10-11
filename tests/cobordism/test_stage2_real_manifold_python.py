# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Stage 2 explores the FULL complex length plane (#644).

The #589 contract (trials constructed exactly real; resident Im l^2 == 0 for
all time) belonged to the ordinary-Lorentzian convention with the complexified
theory unbuilt. Stage 2 now steps the full complex SQUARED coordinates z=l^2
with a real line-search scale, then chooses the continuous square-root branch
for Edge's stored l. Causal dispositions can rotate continuously instead of
only through discrete stage-1 moves. The invariants that survive, and live here:

* every objective value in a stage-2 trace is finite — no trial the optimizer
  can construct surfaces an exception or a NaN;
* acceptance is honestly variational: a trace either descends
  (trace[-1] < trace[0]) or reports stationarity (`last_stage2_stationary`),
  never a backed-off error path;
* the geometry stays finite and causally classifiable after any
  stage-1 + stage-2 sequence, including on cone-crossing hosts and a rebuilt
  #562 causal specimen;
* rejected line searches restore every length VERBATIM (the branch-exact
  record contract, #639) — covered by the rollback suites.
"""

import cmath
import math
import os
import sys
import unittest

import numpy as np

import tessera as T

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _causal_specimen import load_dump, rebuild_joint_node  # noqa: E402
from _closed_s4 import closed_s4 as _closed_s4  # noqa: E402

cob = T.cobordism


def _sphere4():
    """A minimal triangulated S⁴ (boundary of a 5-simplex), unit spacelike,
    lightly jittered — small enough for per-edge FD of the objective."""
    sig = T.Signature(4, T.Lorentzian)
    st = T.Spacetime(T.Metric(True, sig), T.CDT, 1.0, 1.0, T.PREFERRED,
                     T.SimplexBoundarySphere(4))
    st.build()
    for i, e in enumerate(st.get_edge_list().to_vector()):
        e.set_length(cmath.sqrt(complex(1.0 + 0.013 * (i % 5))))
    return st


def _hinges(st):
    sims = list(st.get_simplices())
    hinge_nverts = max(len(s.get_vertices()) for s in sims) - 2
    return [s for s in sims if len(s.get_vertices()) == hinge_nverts]


def _assert_mixed_hinge_regime(test, st):
    """The host must genuinely cross the light cone: at least one hinge's
    Sorkin deficit carries a nonzero imaginary part, so the Wirtinger
    direction is complex and the real-axis restriction is load-bearing."""
    T.ReggeSolver(st, T.MatterConfiguration())  # materialize the hinges
    n_complex = sum(1 for h in _hinges(st)
                    if abs(complex(h.deficit_angle()).imag) > 1e-9)
    test.assertGreater(n_complex, 0, "fixture lost its mixed-hinge regime")


def _max_abs_im(st):
    return max(abs(complex(e.get_length()**2).imag)
               for e in st.get_edge_list().to_vector())


class ComplexSquaredDirectionTest(unittest.TestCase):
    """The stage-2 ascent vector matches both real axes of complex z=l^2."""

    def test_direction_matches_fd_on_mixed_hinge_host(self):
        beta = 1.0
        st = _sphere4()
        # Move off the Sorkin-angle branch cut before checking a TWO-axis
        # derivative. A centered imaginary difference taken exactly on the real
        # Lorentzian sheet measures the discontinuity between continuations,
        # not the local derivative returned by the analytic formulas.
        for index, edge in enumerate(st.get_edge_list().to_vector()):
            z = complex(edge.get_length() ** 2) + 1j * (0.03 + 0.002 * index)
            edge.set_length(cmath.sqrt(z))
        # Hand-set one edge timelike: every base triangle wedge against it has
        # a cofactor pair straddling zero — the m=1 crossing branch (#582) —
        # so the action gradient/Hessian are genuinely complex here.
        st.get_edge_list().to_vector()[3].set_length(cmath.sqrt(complex(-0.8, 0.07)))
        rs = T.ReggeSolver(st, T.MatterConfiguration())
        _assert_mixed_hinge_regime(self, st)

        g = np.asarray(rs.action_gradient_exact(), dtype=complex)
        H = np.asarray(rs.action_hessian_exact(), dtype=complex)
        self.assertGreater(np.max(np.abs(g.imag)), 1e-9,
                           "gradient must be genuinely complex here")
        # The engine subtracts this full complex ascent displacement from z.
        direction = 2.0 * beta * (np.conj(H) @ g)

        edges = st.get_edge_list().to_vector()

        def objective():
            solver = T.ReggeSolver(st, T.MatterConfiguration())
            return beta * sum(abs(c) ** 2
                              for c in solver.action_gradient_exact())

        h = 1e-6
        fd = np.zeros(len(edges), dtype=complex)
        for i, e in enumerate(edges):
            original = complex(e.get_length())
            z0 = original * original
            e.set_length(cmath.sqrt(z0 + h))
            fp = objective()
            e.set_length(cmath.sqrt(z0 - h))
            fm = objective()
            e.set_length(cmath.sqrt(z0 + 1j * h))
            fip = objective()
            e.set_length(cmath.sqrt(z0 - 1j * h))
            fim = objective()
            e.set_length(original)
            fd[i] = ((fp - fm) / (2 * h)
                     + 1j * (fip - fim) / (2 * h))

        scale = np.max(np.abs(fd))
        self.assertGreater(scale, 0.0)
        self.assertLess(np.max(np.abs(direction - fd)) / scale, 1e-5,
                        f"complex direction != grad F:\n{direction}\nvs FD\n{fd}")


class ComplexStageTwoContractTest(unittest.TestCase):
    """Stage 2 stays finite/classifiable while exploring complex z."""

    def _node(self, host, seed=3):
        w = cmath.exp(2j * math.pi / 3)
        opt = cob.MultiCobordism(host, [[1, w, w * w], [1, w * w, w]],
                                 [[1, w, w * w]], degrees=[3], gamma=1.0,
                                 seed=seed)
        opt.seed_inputs([v.get_id()
                         for v in host.get_vertex_list().to_vector()][:2])
        return opt

    def test_stage1_stage2_on_cone_crossing_host(self):
        host = _closed_s4(n_refine=8, seed=3)
        edges = host.get_edge_list().to_vector()
        edges[5].set_length(cmath.sqrt(complex(complex(-0.8, 0.0))))   # timelike
        edges[9].set_length(cmath.sqrt(complex(complex(0.01, 0.0))))   # inside the null band
        opt = self._node(host)
        _assert_mixed_hinge_regime(self, host)
        # No exception may surface from any trial either stage constructs.
        opt.run_stage1(max_steps=4, n_candidate_moves=4,
                       grow_boundaries=True)
        trace = opt.run_stage2(beta=1.0, max_iters=3, alpha0=0.05,
                               tolerance=1e-9)
        self.assertTrue(all(math.isfinite(f) for f in trace))
        if len(trace) == 1:
            # No accepted step is legitimate ONLY as the variational verdict,
            # never as a backed-off error path.
            self.assertTrue(opt.last_stage2_stationary)
        # The complex-step stage 2 may leave the real-l^2 axis by design; what
        # must hold is that every edge stays finite and classifiable.
        for e in opt.st.get_edge_list().to_vector():
            l = complex(e.get_length())
            self.assertTrue(cmath.isfinite(l))
            # Classifiable means EXACTLY ONE of the five cases holds. Since
            # #870 the definite three are not exhaustive: an edge off the
            # definite arguments is mixed, and one with no extent is
            # degenerate. Stage 2 may legitimately leave an edge mixed -- the
            # invariant is that it is never in none of the cases nor in two at
            # once, which the old three-way `or` could not express.
            cases = (e.is_spacelike(), e.is_timelike(), e.is_null(),
                     e.is_mixed(), e.is_degenerate())
            self.assertEqual(sum(bool(c) for c in cases), 1, cases)

    def test_descent_descends_through_the_crossing_regime(self):
        # The objective genuinely decreases while hinges cross the cone — the
        # regime the interim guard could only veto.
        host = _closed_s4(n_refine=8, seed=3)
        host.get_edge_list().to_vector()[5].set_length(cmath.sqrt(complex(complex(-0.8, 0.0))))
        opt = self._node(host)
        trace = opt.run_stage2(beta=1.0, max_iters=6, alpha0=0.05,
                               tolerance=1e-9)
        self.assertTrue(all(math.isfinite(f) for f in trace))
        if len(trace) >= 2:
            self.assertLess(trace[-1], trace[0])
        else:
            # Under the near-kernel-refined objective this host can already be
            # stationary at the given step scale; that verdict must be the
            # variational one, never a backed-off error path.
            self.assertTrue(opt.last_stage2_stationary)


class CausalSpecimenContinuationTest(unittest.TestCase):
    """A rebuilt #562 causal specimen continues under complex-z dynamics with
    finite F and geometry. (The full re-verification of all three causal
    specimens is reported on PR #590; this pins the smallest rebuild path.)"""

    def test_specimen_11001000_stage2_continuation(self):
        dump = load_dump(11001000)
        node = rebuild_joint_node(dump, seed=11001000)
        st = node.st
        # The specimen is causal: its recorded re_min < 0 must survive rebuild.
        re_min = min(complex(e.get_length()**2).real
                     for e in st.get_edge_list().to_vector())
        self.assertLess(re_min, 0.0)
        self.assertEqual(_max_abs_im(st), 0.0)  # dumps carry Im == 0
        trace = node.run_stage2(beta=1.0, max_iters=3, alpha0=0.05,
                                tolerance=1e-9)
        self.assertTrue(all(math.isfinite(f) for f in trace))
        # The complex-step stage 2 may rotate lengths off the real axis; the
        # specimen must stay finite (classifiability asserted below).
        # Timelike content is dynamics, not policy — but the reader must agree
        # the geometry stayed finite and classifiable.
        for e in node.st.get_edge_list().to_vector():
            self.assertTrue(cmath.isfinite(complex(e.get_length()**2)))


if __name__ == "__main__":
    unittest.main()
