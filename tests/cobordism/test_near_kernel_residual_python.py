# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""The near-kernel residual — the pre-topological register signal in r_U (#644).

The period residual is a STEP function in the topology: before the first
register opens it sits exactly at its zero-filled-leak floor, so F carries no
register-seeking gradient at a seed (measured: gamma * r_U = 50.000 for the
single-pentatope seed and for every candidate cone-in). The near-kernel
residual is the same functional continued below the topological threshold —
on the near-kernel the period residual is a target-weighted sum of the
smallest |lambda|^2 — evaluated as the normalized sum of the m smallest
squared SINGULAR values of the METRIC L_k. Metric by design: the term feels
the continuously-valued edge lengths, so it descends both by stage-1 surgery
(a genuine hole zeroes the sigma exactly) and by stage-2 tuning the causal
structure toward null directions — near-kernels with NO holes are the intended
exploration (whether they can carry a register is the next level):

    n * (sum of the m smallest sigma^2) / (sum of all sigma^2)  in [0, m],

with m the EXPECTED register count read off the targets (one register per
target component), never a constant.
"""
import cmath
import importlib.util
import os
import unittest

import tessera

cob = tessera.cobordism

_HS = os.path.join(os.path.dirname(__file__), "_holed_surface.py")
_spec = importlib.util.spec_from_file_location("_holed_surface", _HS)
_hs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_hs)

_OMEGA = complex(cmath.exp(2j * cmath.pi / 3))
_SINGLET = [1 + 0j, _OMEGA, _OMEGA * _OMEGA]


def _seed():
    return tessera.Spacetime.fromVertexTuples(4, [[0, 1, 2, 3, 4]], 1.0, 0.0)


class NearKernelResidualTest(unittest.TestCase):

    def test_saturates_at_zero_once_registers_exist(self):
        # The holed surface has b_1 = 2: asking for one or two registers finds
        # exact kernel modes, and the residual is exactly zero — the term stops
        # interfering the moment the topology delivers.
        st, _es, _holes, _periods = _hs.holed_surface(degree=1)
        self.assertEqual(cob.MultiCobordism.nearKernelResidual(st, 1, 1), 0.0)
        self.assertEqual(cob.MultiCobordism.nearKernelResidual(st, 1, 2), 0.0)

    def test_asking_for_one_more_register_is_small_but_nonzero(self):
        # A third register does not exist, so the third-smallest mode carries a
        # small positive weight: the "almost-register" signal.
        st, _es, _holes, _periods = _hs.holed_surface(degree=1)
        r = cob.MultiCobordism.nearKernelResidual(st, 1, 3)
        self.assertGreater(r, 0.0)
        self.assertLess(r, 1.0)

    def test_scale_invariant(self):
        # L_k is homogeneous of degree -1 in l^2, so a RAW spectral sum would
        # hand stage 2 a pure conformal-inflation descent channel. The
        # normalized ratio is degree 0: a UNIFORM rescale changes nothing,
        # while genuine shape/causal changes still move the term (by design).
        st, _es, _holes, _periods = _hs.holed_surface(degree=1)
        r0 = cob.MultiCobordism.nearKernelResidual(st, 1, 3)
        edges = st.getEdgeList().toVector()
        base = [e.getLength() for e in edges]
        for e, l in zip(edges, base):
            e.setLength(l * cmath.sqrt(2.0))
        st.materializeFacets()
        r1 = cob.MultiCobordism.nearKernelResidual(st, 1, 3)
        self.assertAlmostEqual(r0, r1, places=9)

    def test_zero_expected_registers_is_zero(self):
        st = _seed()
        self.assertEqual(cob.MultiCobordism.nearKernelResidual(st, 3, 0), 0.0)

    def test_expected_count_comes_from_the_targets(self):
        node = cob.MultiCobordism(_seed(), [_SINGLET], [], [3], 50.0, 1, 0, True)
        self.assertEqual(node.expectedRegisterCount(), 3)
        pair = cob.MultiCobordism(_seed(), [[1 + 0j, _OMEGA]], [], [3],
                                  50.0, 1, 0, True)
        self.assertEqual(pair.expectedRegisterCount(), 2)

    def test_seed_has_register_seeking_descent_directions(self):
        # THE motivating fact: without this term the objective was exactly flat
        # across every candidate cone-in at the seed (gamma * r_U = 50.000 for
        # all of them) and strict-descent stage 1 could accept nothing. With it,
        # both cone-in dispositions descend, the timelike one hardest.
        deltas = {}
        for timelike in (False, True):
            node = cob.MultiCobordism(_seed(), [_SINGLET], [], [3],
                                      50.0, 2, 0, True)
            node.seed_inputs([0])
            f0 = node.objective()
            cone = cob.SurgicalCone(node.st)
            ok, reason = cone.coneIn([0, 1, 2, 3], timelike=timelike)
            self.assertTrue(ok, reason)
            deltas[timelike] = node.objective() - f0
        self.assertLess(deltas[False], 0.0)
        self.assertLess(deltas[True], deltas[False])

    def test_stage1_accepts_moves_at_the_seed_again(self):
        node = cob.MultiCobordism(_seed(), [_SINGLET], [], [3], 50.0, 2, 0, True)
        node.seed_inputs([0])
        before = len(node.st.getEdgeList().toVector())
        trace = node.run_stage1(40, 8, True)
        self.assertGreater(len(trace), 1, "no move was ever accepted")
        self.assertGreater(len(node.st.getEdgeList().toVector()), before)
        self.assertLess(trace[-1], trace[0])


if __name__ == "__main__":
    unittest.main()
