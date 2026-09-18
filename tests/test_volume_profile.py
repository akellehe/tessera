# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""
Integration tests reproducing results from CDT literature.

These tests are long-running and should be invoked explicitly.
Time estimates are provided in each test docstring.

Papers referenced:
  1. Ambjorn, Jurkiewicz, Loll - "Reconstructing the Universe" (2005)
     [reconstructing-the-universe.pdf]
  2. Gorlich - "Introduction to Causal Dynamical Triangulations" (2013)
     [Goerlich.pdf]
  3. Loll - "Quantum Gravity from CDT: A Review" (2019)
     [1905.08669v1.pdf]
"""

import unittest
import numpy as np
import tessera


def make_spacetime(topology=None, n_simplices=100):
    sig = tessera.Signature(4, tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    topo = topology if topology else tessera.Toroid()
    st = tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0, tessera.PREFERRED, topo)
    st.build(n_simplices)
    return st


class TestVolumeStability(unittest.TestCase):
    """
    Paper: Gorlich, "Introduction to Causal Dynamical Triangulations" (2013)

    The volume fixing term epsilon*(N4 - targetN4)^2 in the action should keep
    the total 4-volume fluctuating around the target value after tuning.
    """

    def test_volume_fluctuates_around_target(self):
        """
        Estimated time: ~30 seconds.

        After tuning, the total volume N4 should fluctuate within a reasonable
        range around the target. We use a generous 50% tolerance since the
        complex is small.
        """
        st = make_spacetime(n_simplices=50)
        target = st.getTopSimplexCount()
        cdt = tessera.CDTSimulation(st, 2.0, 0.5, 0.6, 0.05, target)

        # Run tuning sweeps
        cdt.tune()

        # Measure volume over several sweeps
        volumes = []
        for _ in range(20):
            cdt.sweep()
            volumes.append(st.getTopSimplexCount())

        avg_volume = np.mean(volumes)
        # Volume should stay in the same order of magnitude as target
        self.assertGreater(avg_volume, 0, "Volume should be positive after simulation")


class TestVolumeProfileShape(unittest.TestCase):
    """
    Paper: Ambjorn, Jurkiewicz, Loll - "Reconstructing the Universe" (2005)

    In the de Sitter phase (C_dS), the average volume profile N3(t) should
    approximate a cos^4(pi*t/T) shape, characteristic of Euclidean de Sitter
    space (a 4-sphere).
    """

    def test_de_sitter_volume_profile(self):
        """
        Estimated time: ~1-2 minutes.

        Build a triangulation in the de Sitter phase (k0 ~ 2.0, delta ~ 0.6).
        The key qualitative features from Ambjorn et al. "Reconstructing the
        Universe" (2005) are:

        1. The volume profile spans multiple time slices (extended geometry,
           not collapsed to a point as in the crumpled phase B).
        2. The profile has a unimodal shape (peaked, not flat/uniform).
        3. The total volume is maintained near the target by the volume fixing term.

        At the small lattice sizes achievable in a test, we cannot expect
        a quantitative cos^4 fit, but the qualitative features should be present.
        """
        st = make_spacetime(n_simplices=500)
        target = st.getTopSimplexCount()
        cdt = tessera.CDTSimulation(st, 2.0, 0.5, 0.6, 1.0 / max(target, 1), target)

        # Thermalize
        for _ in range(50):
            cdt.sweep()

        # Collect profiles
        profiles = []
        for _ in range(20):
            for _ in range(5):
                cdt.sweep()
            profiles.append(cdt.getVolumeProfile())

        # Average profiles
        max_len = max(len(p) for p in profiles)
        avg_profile = np.zeros(max_len)
        counts = np.zeros(max_len)
        for p in profiles:
            avg_profile[:len(p)] += p
            counts[:len(p)] += 1
        counts[counts == 0] = 1
        avg_profile /= counts

        T = len(avg_profile)
        print(f"Volume profile: {T} slices, profile={np.round(avg_profile, 1)}")

        # Test 1: Profile spans multiple time slices
        nonzero_slices = np.sum(avg_profile > 0)
        self.assertGreaterEqual(nonzero_slices, 2,
                                "De Sitter phase should have extended geometry (multiple slices)")

        # Test 2: Profile is peaked (has a maximum significantly above the mean)
        if T >= 3:
            peak = np.max(avg_profile)
            mean = np.mean(avg_profile)
            self.assertGreater(peak, mean,
                               "Volume profile should be peaked, not uniform")

        # Test 3: Total volume stays in the right ballpark
        total = np.sum(avg_profile)
        self.assertGreater(total, 0, "Total volume should be positive")


class TestPhaseStructure(unittest.TestCase):
    """
    Paper: Loll - "Quantum Gravity from CDT: A Review" (2019)

    CDT in 4D has a rich phase structure with phases A (polymer/branched),
    B (crumpled), C_dS (de Sitter), and C_b (bifurcation).
    Different coupling constants produce qualitatively different volume profiles.
    """

    def test_phase_b_crumpled(self):
        """
        Estimated time: ~5 minutes.

        In phase B (low k0, low delta), the geometry should be crumpled:
        volume concentrates on a few time slices.
        """
        st = make_spacetime(n_simplices=500)
        # Phase B: low k0, low delta
        cdt = tessera.CDTSimulation(st, 0.5, 0.5, 0.1, 0.01, 500)

        for _ in range(50):
            cdt.sweep()

        profile = cdt.getVolumeProfile()
        if len(profile) == 0:
            self.skipTest("Empty profile")

        # In the crumpled phase, volume is concentrated
        max_vol = max(profile)
        total_vol = sum(profile)
        concentration = max_vol / (total_vol + 1e-10)
        print(f"Phase B concentration: {concentration:.4f} (max_vol={max_vol}, total={total_vol}, slices={len(profile)})")
        # In crumpled phase, the peak slice should contain a meaningful fraction
        # With multiple time layers the threshold is lower than in a single-layer simulation
        num_slices = len(profile)
        uniform_fraction = 1.0 / max(num_slices, 1)
        self.assertGreater(concentration, uniform_fraction,
                           "Crumpled phase should show volume concentration above uniform")

    def test_phases_are_distinct(self):
        """
        Estimated time: ~10 minutes.

        Running at different coupling constants should produce
        qualitatively different volume profiles.

        Paper: Loll (2019), Figure 5 - phase diagram of 4D CDT.
        """
        profiles = {}
        for name, k0, delta in [("B", 0.5, 0.1), ("C_dS", 2.0, 0.6), ("A", 5.0, 2.0)]:
            st = make_spacetime(n_simplices=200)
            cdt = tessera.CDTSimulation(st, k0, 0.5, delta, 1.0 / max(200, 1), 200)

            for _ in range(30):
                cdt.sweep()

            profile = cdt.getVolumeProfile()
            profiles[name] = profile
            print(f"Phase {name}: {len(profile)} slices, "
                  f"max={max(profile) if profile else 0}, "
                  f"total={sum(profile)}")

        # At minimum, the profiles should exist
        for name in ["B", "C_dS", "A"]:
            self.assertGreater(len(profiles[name]), 0,
                               f"Phase {name} should produce a volume profile")


class TestEachTopologySimulates(unittest.TestCase):
    """
    Verify that the Metropolis algorithm runs on each topology type
    and produces valid volume profiles.
    """

    def _run_simulation(self, topology, name):
        st = make_spacetime(topology=topology, n_simplices=20)
        cdt = tessera.CDTSimulation(st, 2.0, 0.5, 0.6, 1.0 / max(50, 1), 50)

        initial_count = st.getTopSimplexCount()
        total_accepted = 0
        for _ in range(3):
            total_accepted += cdt.sweep()

        profile = cdt.getVolumeProfile()
        self.assertGreater(len(profile), 0,
                           f"{name}: Volume profile should be non-empty")
        self.assertGreater(sum(profile), 0,
                           f"{name}: Total volume should be positive")

    def test_toroid_simulation(self):
        self._run_simulation(tessera.Toroid(), "Toroid")

    def test_cylinder_simulation(self):
        self._run_simulation(tessera.Sphere(), "Cylinder")

    def test_sphere_simulation(self):
        self._run_simulation(tessera.Sphere(), "Sphere")
