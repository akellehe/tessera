# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""
Tests for CDT simulation infrastructure.

References:
  [RU]  Ambjorn, Jurkiewicz, Loll, "Reconstructing the Universe",
        Phys. Rev. D 72 (2005), arXiv:hep-th/0505154v2
  [BGL] Brunekreef, Gorlich, Loll, "Simulating CDT quantum gravity",
        arXiv:2310.16744v1 (2023)
"""

import unittest
import tessera


class TestCDTAction(unittest.TestCase):
    """[RU] eq. 2: Test the Regge action computation for 4D CDT."""

    def _make_spacetime(self, topology=None, n_simplices=10):
        sig = tessera.Signature(4, tessera.Lorentzian)
        metric = tessera.Metric(True, sig)
        topo = topology if topology else tessera.Toroid()
        st = tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0, tessera.PREFERRED, topo)
        st.build(n_simplices)
        return st

    def test_action_computation(self):
        """Verify action is computed from the correct formula (Eq. 2, hep-th/0505154):
        S = -(k0 + 6*delta)*N0 + (k4 + 2*delta)*N41 + (k4 + delta)*N32 + epsilon*(N41 - target)^2
        Volume-fix targets N41 per Reconstructing the Universe eq. 6.
        """
        st = self._make_spacetime(n_simplices=5)
        k0, k4, delta, epsilon = 2.0, 0.5, 0.6, 0.02
        target = st.getN41()

        cdt = tessera.CDTSimulation(st, k0, k4, delta, epsilon, target)
        action = cdt.computeAction()

        # Compute expected action from counts
        n0 = st.getVertexCount()
        n41 = st.getN41()
        n32 = st.getN32()

        expected = -(k0 + 6 * delta) * n0 + (k4 + 2 * delta) * n41 + (k4 + delta) * n32
        expected += epsilon * (n41 - target) ** 2

        self.assertAlmostEqual(action, expected, places=6,
                               msg=f"Action mismatch: got {action}, expected {expected}")

    def test_action_changes_with_couplings(self):
        """Action should change when coupling constants change."""
        st = self._make_spacetime(n_simplices=5)
        cdt1 = tessera.CDTSimulation(st, 1.0, 0.5, 0.3, 0.01, 50)
        cdt2 = tessera.CDTSimulation(st, 5.0, 0.5, 0.3, 0.01, 50)
        self.assertNotAlmostEqual(cdt1.computeAction(), cdt2.computeAction())


class TestCDTMoves(unittest.TestCase):
    """[BGL] Sec. 2.3: Test individual Pachner moves."""

    def _make_cdt(self, n_simplices=20, topology=None):
        sig = tessera.Signature(4, tessera.Lorentzian)
        metric = tessera.Metric(True, sig)
        topo = topology if topology else tessera.Toroid()
        st = tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0, tessera.PREFERRED, topo)
        st.build(n_simplices)
        return tessera.CDTSimulation(st, 2.0, 0.5, 0.6, 0.0, 10000), st

    def test_add_move_changes_counts(self):
        """The add move should increase simplex and vertex counts."""
        cdt, st = self._make_cdt(n_simplices=30)
        initial_n4 = st.getTopSimplexCount()
        initial_n0 = st.getVertexCount()

        # Try add moves until one succeeds (may take several attempts)
        accepted = False
        for _ in range(200):
            if cdt.add():
                accepted = True
                break

        if accepted:
            self.assertGreater(st.getTopSimplexCount(), initial_n4,
                               "Add move should increase simplex count")
            self.assertGreater(st.getVertexCount(), initial_n0,
                               "Add move should increase vertex count")

    def test_flip_move_preserves_vertex_count(self):
        """The flip move should not change the vertex count."""
        cdt, st = self._make_cdt(n_simplices=30)
        initial_n0 = st.getVertexCount()

        for _ in range(200):
            if cdt.flip():
                break

        self.assertEqual(st.getVertexCount(), initial_n0,
                         "Flip move should preserve vertex count")

    def test_sweep_runs_without_error(self):
        """A full sweep should complete without crashing."""
        cdt, st = self._make_cdt(n_simplices=20)
        accepted = cdt.sweep()
        self.assertIsInstance(accepted, int)
        self.assertGreaterEqual(accepted, 0)

    def test_multiple_sweeps(self):
        """Multiple sweeps should run successfully."""
        cdt, st = self._make_cdt(n_simplices=20)
        total_accepted = 0
        for _ in range(5):
            total_accepted += cdt.sweep()
        self.assertGreaterEqual(total_accepted, 0)


class TestCDTTopologies(unittest.TestCase):
    """[RU] Sec. 3: Test that each topology builds and supports CDT simulation."""

    def _test_topology_builds_and_simulates(self, topology):
        sig = tessera.Signature(4, tessera.Lorentzian)
        metric = tessera.Metric(True, sig)
        st = tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0, tessera.PREFERRED, topology)
        st.build(10)
        self.assertGreater(st.getTopSimplexCount(), 0, "Topology should produce simplices")
        self.assertGreater(st.getVertexCount(), 0, "Topology should produce vertices")

        # Run a sweep
        cdt = tessera.CDTSimulation(st, 2.0, 0.5, 0.6, 1.0 / max(100, 1), 100)
        accepted = cdt.sweep()
        self.assertIsInstance(accepted, int)

    def test_toroid_topology(self):
        self._test_topology_builds_and_simulates(tessera.Toroid())

    def test_cylinder_topology(self):
        self._test_topology_builds_and_simulates(tessera.Cylinder())

    def test_sphere_topology(self):
        self._test_topology_builds_and_simulates(tessera.Sphere())


class TestCDTVolumeProfile(unittest.TestCase):
    """[RU] eq. 6: Test volume profile computation."""

    def test_volume_profile_nonempty(self):
        """Volume profile should be non-empty after building a spacetime."""
        sig = tessera.Signature(4, tessera.Lorentzian)
        metric = tessera.Metric(True, sig)
        st = tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0, tessera.PREFERRED, tessera.Toroid())
        st.build(15)

        cdt = tessera.CDTSimulation(st, 2.0, 0.5, 0.6, 1.0 / max(100, 1), 100)
        profile = cdt.getVolumeProfile()
        self.assertGreater(len(profile), 0, "Volume profile should not be empty")
        self.assertGreater(sum(profile), 0, "Total volume should be positive")

    def test_volume_profile_sums_to_total(self):
        """Volume profile entries should sum to the total simplex count."""
        sig = tessera.Signature(4, tessera.Lorentzian)
        metric = tessera.Metric(True, sig)
        st = tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0, tessera.PREFERRED, tessera.Toroid())
        st.build(15)

        cdt = tessera.CDTSimulation(st, 2.0, 0.5, 0.6, 1.0 / max(100, 1), 100)
        profile = cdt.getVolumeProfile()
        self.assertEqual(sum(profile), st.getTopSimplexCount(),
                         "Volume profile should sum to total simplex count")


class TestCDTAcceptanceRates(unittest.TestCase):
    """[BGL] eq. 11: Test acceptance rate tracking."""

    def test_acceptance_rates_reported(self):
        """Acceptance rates should be a dict with all move types."""
        sig = tessera.Signature(4, tessera.Lorentzian)
        metric = tessera.Metric(True, sig)
        st = tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0, tessera.PREFERRED, tessera.Toroid())
        st.build(20)

        cdt = tessera.CDTSimulation(st, 2.0, 0.5, 0.6, 1.0 / max(100, 1), 100)
        cdt.sweep()
        rates = cdt.getAcceptanceRates()

        for move_type in [tessera.AddMove.MOVE_TYPE, tessera.RemoveMove.MOVE_TYPE,
                          tessera.FlipMove.MOVE_TYPE, tessera.ShiftMove.MOVE_TYPE,
                          "ishift"]:
            self.assertIn(move_type, rates, f"Missing acceptance rate for {move_type}")
            self.assertGreaterEqual(rates[move_type], 0.0)
            self.assertLessEqual(rates[move_type], 1.0)


class TestSpacetimeCounting(unittest.TestCase):
    """[RU] eq. 2: Test Spacetime counting methods."""

    def test_counts_after_build(self):
        sig = tessera.Signature(4, tessera.Lorentzian)
        metric = tessera.Metric(True, sig)
        st = tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0, tessera.PREFERRED, tessera.Toroid())
        st.build(10)

        self.assertGreater(st.getTopSimplexCount(), 0)
        self.assertGreater(st.getVertexCount(), 0)
        # N41 and N32 are subsets of the total simplex count
        self.assertEqual(st.getTopSimplexCount(), st.getN41() + st.getN32(),
                         "N4 should equal N41 + N32")

    def test_random_simplex(self):
        sig = tessera.Signature(4, tessera.Lorentzian)
        metric = tessera.Metric(True, sig)
        st = tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0, tessera.PREFERRED, tessera.Toroid())
        st.build(10)

        simplex = st.getRandomSimplex()
        self.assertIsNotNone(simplex, "getRandomSimplex should return a simplex")
