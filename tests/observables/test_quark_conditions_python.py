# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""The v16 quark verdict: seven Section 10 conditions, each passed, failed or
not evaluable from the certificates a caller measured; and the 18-mode states
of `SharpSpin` the three-sheeted tetrahedron needs."""

import unittest

import numpy as np

from tessera import observables as obs

E = obs.QuarkConditionEvidence
Status = obs.QuarkConditionStatus


def _all_holding():
    return [[E(name, True, "measured")
             for name in obs.QuarkConditions.required_evidence(n)]
            for n in range(1, 8)]


class TestQuarkConditions(unittest.TestCase):
    def test_seven_named_conditions(self):
        self.assertEqual(obs.QuarkConditions.kConditionCount, 7)
        self.assertEqual(obs.QuarkConditions.condition_names(),
                         ["persistent-cluster", "color-spin-fiber",
                          "anchor-atlas", "odd-occupation", "color-transport",
                          "lineage", "fingerprint"])
        self.assertIn("j = 1/2 doublet", obs.QuarkConditions.statement(2))
        with self.assertRaises(ValueError):
            obs.QuarkConditions.statement(8)

    def test_every_certificate_held_certifies(self):
        verdict = obs.QuarkConditions.evaluate(_all_holding())
        self.assertTrue(verdict.certified)
        self.assertEqual(list(verdict.failed), [])
        self.assertEqual(list(verdict.not_evaluable), [])
        self.assertTrue(all(c.status == Status.Passed
                            for c in verdict.conditions))

    def test_a_failing_certificate_fails_its_condition(self):
        evidence = _all_holding()
        evidence[3] = [E("odd-occupation-parity", False, "parity +1")]
        verdict = obs.QuarkConditions.evaluate(evidence)
        self.assertFalse(verdict.certified)
        self.assertEqual(list(verdict.failed), ["odd-occupation"])
        self.assertEqual(list(verdict.conditions[3].failing),
                         ["odd-occupation-parity"])

    def test_an_unmeasured_certificate_is_not_evaluable(self):
        evidence = _all_holding()
        evidence[5] = [E("lineage-intersection", None, "no history")]
        evidence[0] = evidence[0][:3]          # lifetime etc. not supplied
        verdict = obs.QuarkConditions.evaluate(evidence)
        self.assertFalse(verdict.certified)
        self.assertEqual(list(verdict.not_evaluable),
                         ["persistent-cluster", "lineage"])
        missing = list(verdict.conditions[0].missing)
        self.assertEqual(missing, ["successor-overlap", "multi-frame-lifetime",
                                   "external-leakage"])
        # the unsupplied required certificates are listed as unmeasured
        names = [e.name for e in verdict.conditions[0].evidence]
        self.assertIn("multi-frame-lifetime", names)

    def test_failure_outranks_absence(self):
        read = obs.QuarkConditions.evaluate_condition(
            5, [E("color-transport-full-rank", False, "det S = 0")])
        self.assertEqual(read.status, Status.Failed)

    def test_an_optional_certificate_can_fail_but_never_blocks(self):
        read = obs.QuarkConditions.evaluate_condition(
            6, [E("lineage-intersection", True, "N_Q = 1")])
        self.assertEqual(read.status, Status.Passed)
        read = obs.QuarkConditions.evaluate_condition(
            6, [E("lineage-intersection", True, "N_Q = 1"),
                E("determinant-winding", False, "nu = 0")])
        self.assertEqual(read.status, Status.Failed)

    def test_an_unnamed_certificate_is_refused(self):
        with self.assertRaises(ValueError):
            obs.QuarkConditions.evaluate_condition(4, [E("", True, "")])


class TestEighteenModeStates(unittest.TestCase):
    def test_state_vectors_reach_the_exterior_algebra_limit(self):
        self.assertEqual(obs.SharpSpin.kMaxStateModes, 24)
        self.assertEqual(obs.SharpSpin.kMaxDenseModes, 16)
        state = np.asarray(obs.SharpSpin.determinant([0, 7, 17], 18))
        self.assertEqual(state.size, 1 << 18)
        self.assertEqual(int(np.count_nonzero(state)), 1)
        with self.assertRaises(ValueError):
            obs.SharpSpin.determinant([0, 1], 25)

    def test_nine_carriers_give_eighteen_mode_spin_matrices(self):
        matrices = obs.SharpSpin.doubletSpinMatrices(9)
        self.assertEqual(np.asarray(matrices[0]).shape, (18, 18))
        with self.assertRaises(ValueError):
            obs.SharpSpin.doubletSpinMatrices(13)

    def test_an_eighteen_mode_spin_read(self):
        """Three up spins on carriers 0, 1 and 2 of nine: J = 3/2, sharp at
        its own eigenvalue and not at 3/4."""
        matrices = obs.SharpSpin.doubletSpinMatrices(9)
        state = np.asarray(obs.SharpSpin.determinant([0, 2, 4], 18))
        self.assertTrue(obs.SharpSpin.read(matrices, state, state,
                                           15.0 / 4.0).sharp)
        self.assertFalse(obs.SharpSpin.read(matrices, state, state,
                                            0.75).sharp)


if __name__ == "__main__":
    unittest.main()
