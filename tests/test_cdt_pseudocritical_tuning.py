# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""Tests for the pseudo-critical tuning of the cosmological coupling k4.

The pseudo-critical coupling is the value of k4 at which the four-volume neither
grows nor shrinks under the bare action.  It is fixed by the entropy of the
triangulations available at a given volume and so cannot be written down from the
action alone; tune() locates it by measuring the sign of the four-volume drift.
Below it the volume grows, and since the volume-fixing term constrains N41 alone,
the (4,1) sector then settles at a multiple of its target rather than at the
target (#965).

References:
  [RU]  Ambjorn, Jurkiewicz, Loll, "Reconstructing the Universe",
        Phys. Rev. D 72 (2005), arXiv:hep-th/0505154v2
"""

import unittest

import tessera


class TestPseudoCriticalTuning(unittest.TestCase):
    K0 = 2.2
    DELTA = 0.6
    SEED = 20260905
    BUILD = 1600
    TARGET = 2000

    #: Sweeps discarded before a drift measurement.  A freshly built staircase
    #: is not a typical configuration at any coupling and relaxes for a few tens
    #: of sweeps whatever k4 is; that relaxation is not the drift being measured.
    SETTLE = 50
    #: Sweeps the drift is measured over, once settled.
    MEASURE = 250

    #: The closed form tune() starts its search from: the coupling that zeroes
    #: the Regge action change of one (2,2d) add move, entropy not accounted for.
    CLOSED_FORM_K4 = (K0 + 6.0 * DELTA) / 6.0 - 2.0 * DELTA

    @classmethod
    def setUpClass(cls):
        """Tune once; the coupling is deterministic under a fixed seed and the
        search is the expensive part of these tests."""
        _, cdt = cls._simulation(k4=0.5)
        cdt.tune()
        cls.TUNED_K4 = cdt.getK4()

    @classmethod
    def _simulation(cls, k4, epsilon=None, target=None, build=None, seed=SEED):
        target = cls.TARGET if target is None else target
        epsilon = (1.0 / target) if epsilon is None else epsilon
        sig = tessera.Signature(4, tessera.Lorentzian)
        st = tessera.Spacetime(tessera.Metric(True, sig), tessera.CDT,
                               1.0, 1.0, tessera.PREFERRED, tessera.Toroid())
        st.setSeed(seed)
        st.build(cls.BUILD if build is None else build)
        cdt = tessera.CDTSimulation(st, cls.K0, k4, cls.DELTA, epsilon, target)
        cdt.setSeed(seed)
        return st, cdt

    @staticmethod
    def _volume(st):
        return st.getN41() + st.getN32()

    def _settled_drift(self, k4):
        """Relative change in the four-volume per sweep at k4, with no
        volume-fixing term, after the build transient has passed."""
        st, cdt = self._simulation(k4, epsilon=0.0, target=1)
        cdt.sweep(self.SETTLE)
        before = self._volume(st)
        cdt.sweep(self.MEASURE)
        return (self._volume(st) - before) / (before * self.MEASURE)

    def test_tuned_coupling_is_above_the_closed_form_estimate(self):
        """The closed form ignores the entropy of the available triangulations,
        so it lies below the coupling at which the volume stops growing: it
        returns -0.233 here, against a measured 0.76."""
        self.assertGreater(self.TUNED_K4, self.CLOSED_FORM_K4 + 0.5)

    def test_four_volume_is_stationary_at_the_tuned_coupling(self):
        """With no volume-fixing term k4 alone sets the volume, so at the tuned
        coupling the four-volume neither grows nor collapses."""
        self.assertLess(abs(self._settled_drift(self.TUNED_K4)), 1.0e-3)

    def test_a_sub_critical_coupling_grows_the_volume(self):
        """The tuned value means something only if a coupling below it behaves
        differently.  0.6 lower, the same complex inflates by more than half
        again over the same 250 sweeps that leave it flat at the tuned value."""
        self.assertGreater(self._settled_drift(self.TUNED_K4 - 0.6), 2.0e-3)

    def test_volume_fixing_holds_the_four_one_sector_near_its_target(self):
        """Below the pseudo-critical coupling the volume-fixing term cannot hold
        N41 at the target: it settles where the restoring force balances the
        supercritical drive, measured at 1.55x the target under the closed form.
        At the tuned coupling N41 stays near the target itself."""
        st, cdt = self._simulation(self.TUNED_K4)
        cdt.sweep(800)
        self.assertLess(abs(st.getN41() - self.TARGET), 0.25 * self.TARGET)

    def test_tuning_keeps_the_configuration_it_was_given(self):
        """Measurements above critical shrink the complex and measurements below
        it inflate one.  Tuning bounds both, so the caller gets back a
        configuration it can still run: the band is a factor of two either way,
        crossed at the end of whichever sweep crosses it."""
        st, cdt = self._simulation(k4=0.5, build=800)
        before = self._volume(st)
        cdt.tune()
        self.assertGreaterEqual(self._volume(st), before // 2)
        self.assertLessEqual(self._volume(st), 2.1 * before)

    def test_tuning_is_reproducible_under_a_fixed_seed(self):
        """Both the spacetime and the Markov chain are seeded, so two runs of
        the search return the same coupling."""
        _, first = self._simulation(k4=0.5, build=400)
        first.tune()
        _, second = self._simulation(k4=0.5, build=400)
        second.tune()
        self.assertEqual(first.getK4(), second.getK4())


if __name__ == "__main__":
    unittest.main()
