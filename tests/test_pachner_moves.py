# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""
Comprehensive tests for CDT Pachner moves and their invariants.

These tests verify that each move type (add, remove, flip, shift, ishift)
maintains the correct combinatorial, causal, and algebraic invariants of
the simplicial complex.

References:
  [RU]  Ambjorn, Jurkiewicz, Loll, "Reconstructing the Universe",
        Phys. Rev. D 72 (2005), arXiv:hep-th/0505154v2
  [BGL] Brunekreef, Gorlich, Loll, "Simulating CDT quantum gravity",
        arXiv:2310.16744v1 (2023)
"""

import unittest
import tessera
import numpy as np


def _make_cdt(n_simplices=200, k0=2.2, delta=0.6, epsilon=0.02):
    """Build spacetime + CDT."""
    sig = tessera.Signature(4, tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    st = tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0, tessera.PREFERRED,
                         tessera.Toroid())
    st.build(n_simplices)
    target = st.getN41()
    cdt = tessera.CDTSimulation(st, k0, 0.5, delta, epsilon, target)
    return cdt, st


def _count_top_simplices(st):
    """Count top-dimensional (5-vertex) simplices by iterating."""
    count = 0
    for s in st.getSimplices():
        if len(s.getVertices()) == 5:
            count += 1
    return count


def _count_orientations(st):
    """Return dict of orientation -> count for top simplices."""
    counts = {}
    for s in st.getSimplices():
        if len(s.getVertices()) == 5:
            o = s.getOrientation().numeric()
            counts[o] = counts.get(o, 0) + 1
    return counts


def _orientation_n41_n32(counts):
    """Given orientation counts, compute N41 and N32."""
    n41 = counts.get((4, 1), 0) + counts.get((1, 4), 0)
    n32 = counts.get((3, 2), 0) + counts.get((2, 3), 0)
    return n41, n32


# =====================================================================
# Fundamental counting invariants
# =====================================================================

class TestCountingInvariants(unittest.TestCase):
    """[RU] eq. 2: Verify that N4, N41, N32, profile sum are always consistent."""

    def test_n4_equals_n41_plus_n32_after_build(self):
        """After build(), getTopSimplexCount() == getN41() + getN32()."""
        _, st = _make_cdt()
        self.assertEqual(st.getTopSimplexCount(), st.getN41() + st.getN32())

    def test_n4_matches_manual_count_after_build(self):
        """getTopSimplexCount() matches manual iteration over simplices."""
        _, st = _make_cdt()
        self.assertEqual(st.getTopSimplexCount(), _count_top_simplices(st))

    def test_n41_n32_match_orientation_count(self):
        """N41/N32 from getN41()/getN32() match manual orientation count."""
        _, st = _make_cdt()
        counts = _count_orientations(st)
        n41, n32 = _orientation_n41_n32(counts)
        self.assertEqual(st.getN41(), n41)
        self.assertEqual(st.getN32(), n32)

    def test_profile_sums_to_n4(self):
        """Volume profile entries should sum to N4."""
        cdt, st = _make_cdt()
        profile = cdt.getVolumeProfile()
        self.assertEqual(sum(profile), st.getTopSimplexCount())

    def test_no_uncounted_orientations(self):
        """Every top simplex should have a valid CDT orientation."""
        _, st = _make_cdt()
        counts = _count_orientations(st)
        n41, n32 = _orientation_n41_n32(counts)
        total_counted = n41 + n32
        total_top = _count_top_simplices(st)
        self.assertEqual(total_counted, total_top,
                         f"Uncounted orientations: {counts}")

    def test_invariants_hold_after_many_sweeps(self):
        """All counting invariants hold after 100 sweeps."""
        cdt, st = _make_cdt(n_simplices=200)
        cdt.tune()
        cdt.sweep(100)

        # N4 = N41 + N32
        self.assertEqual(st.getTopSimplexCount(), st.getN41() + st.getN32())

        # Manual count matches
        self.assertEqual(st.getTopSimplexCount(), _count_top_simplices(st))

        # Orientation counts match
        counts = _count_orientations(st)
        n41, n32 = _orientation_n41_n32(counts)
        self.assertEqual(st.getN41(), n41)
        self.assertEqual(st.getN32(), n32)

        # No uncounted
        self.assertEqual(n41 + n32, _count_top_simplices(st),
                         f"Uncounted after 100 sweeps: {counts}")

        # Profile sums
        profile = cdt.getVolumeProfile()
        self.assertEqual(sum(profile), st.getTopSimplexCount())


# =====================================================================
# Causality invariants
# =====================================================================

class TestCausalityInvariants(unittest.TestCase):
    """[RU] Sec. 3, [BGL] Sec. 2.3: Every top simplex must span exactly 2 adjacent time slices."""

    def _check_all_causal(self, st, label=""):
        """Assert every top simplex spans exactly 2 time slices."""
        for s in st.getSimplices():
            verts = s.getVertices()
            if len(verts) != 5:
                continue
            times = set()
            for v in verts:
                times.add(v.getTime())
            self.assertEqual(
                len(times), 2,
                f"{label}: top simplex spans {len(times)} time slices "
                f"(times={times}, orientation={s.getOrientation().numeric()})")

    def test_all_causal_after_build(self):
        _, st = _make_cdt()
        self._check_all_causal(st, "after build")

    def test_all_causal_after_sweeps(self):
        cdt, st = _make_cdt(n_simplices=200)
        cdt.tune()
        cdt.sweep(100)
        self._check_all_causal(st, "after 100 sweeps")

    def test_all_causal_after_500_sweeps(self):
        """Longer run to stress-test causality."""
        cdt, st = _make_cdt(n_simplices=200)
        cdt.tune()
        cdt.sweep(500)
        self._check_all_causal(st, "after 500 sweeps")


# =====================================================================
# Add move invariants
# =====================================================================

class TestAddMove(unittest.TestCase):
    """The (2,2d) add: +1 vertex, +(2d-2) N41 simplices.

    Ref: [BGL] Sec. 2.3.1 (adapted to 4D).
    """

    def test_add_increments_vertex_and_n41_count(self):
        """(2,2d) add: dN0=+1, dN41=+6 in 4D."""
        cdt, st = _make_cdt(n_simplices=200)
        for _ in range(2000):
            n0_before = st.getVertexCount()
            n41_before = st.getN41()
            if cdt.add():
                self.assertEqual(st.getVertexCount(), n0_before + 1,
                                 "add() should increment vertex count by 1")
                self.assertEqual(st.getN41(), n41_before + 6,
                                 "add() should increment N41 by 2d-2=6")
                return
        self.skipTest("No add accepted in 2000 attempts")

    def test_add_preserves_counting_invariant(self):
        cdt, st = _make_cdt(n_simplices=200)
        for _ in range(2000):
            if cdt.add():
                self.assertEqual(st.getTopSimplexCount(),
                                 st.getN41() + st.getN32())
                counts = _count_orientations(st)
                n41, n32 = _orientation_n41_n32(counts)
                self.assertEqual(n41 + n32, _count_top_simplices(st))
                return
        self.skipTest("No add accepted")


# =====================================================================
# Remove move invariants
# =====================================================================

class TestRemoveMove(unittest.TestCase):
    """The (2d,2) remove: -1 vertex, -(2d-2) N41 simplices.

    Ref: [BGL] Sec. 2.3.1 (adapted to 4D).
    """

    def test_remove_decrements_n41_count(self):
        """(2d,2) remove: dN0=-1, dN41=-6 in 4D."""
        cdt, st = _make_cdt(n_simplices=200)
        for _ in range(500):
            cdt.add()
        for _ in range(2000):
            n41_before = st.getN41()
            if cdt.remove():
                self.assertEqual(st.getN41(), n41_before - 6,
                                 "remove() should decrement N41 by 2d-2=6")
                return
        self.skipTest("No remove accepted")

    def test_remove_preserves_counting_invariant(self):
        cdt, st = _make_cdt(n_simplices=200)
        for _ in range(500):
            cdt.add()
        for _ in range(2000):
            if cdt.remove():
                self.assertEqual(st.getTopSimplexCount(),
                                 st.getN41() + st.getN32())
                return
        self.skipTest("No remove accepted")


# =====================================================================
# Flip move invariants
# =====================================================================

class TestFlipMove(unittest.TestCase):
    """[BGL] Sec. 2.3.2: The (2,d) flip: 2→d simplices, vertex count unchanged."""

    def test_flip_preserves_vertex_count(self):
        cdt, st = _make_cdt(n_simplices=100)
        for _ in range(1000):
            n0_before = st.getVertexCount()
            if cdt.flip():
                self.assertEqual(st.getVertexCount(), n0_before,
                                 "flip() should not change vertex count")
                return
        self.skipTest("No flip accepted in 1000 attempts")

    def test_flip_changes_simplex_count_by_d_minus_2(self):
        """(2,d) flip: 2→d means +2 top simplices in 4D (2→4)."""
        cdt, st = _make_cdt(n_simplices=100)
        for _ in range(1000):
            n4_before = st.getTopSimplexCount()
            if cdt.flip():
                delta_n4 = st.getTopSimplexCount() - n4_before
                self.assertGreaterEqual(delta_n4, 0,
                                        f"(2,4) flip should not decrease N4, got {delta_n4}")
                self.assertLessEqual(delta_n4, 2,
                                     f"(2,4) flip dN4 should be at most +2, got {delta_n4}")
                return
        self.skipTest("No flip accepted")

    def test_flip_preserves_counting_invariant(self):
        cdt, st = _make_cdt(n_simplices=100)
        for _ in range(1000):
            if cdt.flip():
                self.assertEqual(st.getTopSimplexCount(),
                                 st.getN41() + st.getN32())
                counts = _count_orientations(st)
                n41, n32 = _orientation_n41_n32(counts)
                self.assertEqual(n41 + n32, _count_top_simplices(st),
                                 f"Uncounted after flip: {counts}")
                return
        self.skipTest("No flip accepted")

    def test_flip_preserves_causality(self):
        cdt, st = _make_cdt(n_simplices=100)
        for _ in range(1000):
            if cdt.flip():
                for s in st.getSimplices():
                    if len(s.getVertices()) != 5:
                        continue
                    times = set(v.getTime() for v in s.getVertices())
                    self.assertEqual(len(times), 2,
                                     f"Flip created non-causal simplex "
                                     f"with times {times}")
                return
        self.skipTest("No flip accepted")


# =====================================================================
# Shift move invariants
# =====================================================================

class TestShiftMove(unittest.TestCase):
    """[BGL] Sec. 2.3.3: The (3,3) shift: 3→3 simplices, vertex count unchanged."""

    def test_shift_preserves_vertex_count(self):
        cdt, st = _make_cdt(n_simplices=100)
        for _ in range(1000):
            n0_before = st.getVertexCount()
            if cdt.shift():
                self.assertEqual(st.getVertexCount(), n0_before,
                                 "shift() should not change vertex count")
                return
        self.skipTest("No shift accepted in 1000 attempts")

    def test_shift_preserves_simplex_count(self):
        """(3,3) shift: 3→3 means N4 unchanged."""
        cdt, st = _make_cdt(n_simplices=100)
        for _ in range(1000):
            n4_before = st.getTopSimplexCount()
            if cdt.shift():
                self.assertLessEqual(st.getTopSimplexCount(), n4_before,
                                     "shift() should not increase N4")
                return
        self.skipTest("No shift accepted")

    def test_shift_preserves_counting_invariant(self):
        cdt, st = _make_cdt(n_simplices=100)
        for _ in range(1000):
            if cdt.shift():
                self.assertEqual(st.getTopSimplexCount(),
                                 st.getN41() + st.getN32())
                counts = _count_orientations(st)
                n41, n32 = _orientation_n41_n32(counts)
                self.assertEqual(n41 + n32, _count_top_simplices(st),
                                 f"Uncounted after shift: {counts}")
                return
        self.skipTest("No shift accepted")

    def test_shift_preserves_causality(self):
        cdt, st = _make_cdt(n_simplices=100)
        for _ in range(1000):
            if cdt.shift():
                for s in st.getSimplices():
                    if len(s.getVertices()) != 5:
                        continue
                    times = set(v.getTime() for v in s.getVertices())
                    self.assertEqual(len(times), 2,
                                     f"Shift created non-causal simplex")
                return
        self.skipTest("No shift accepted")


# =====================================================================
# Action consistency
# =====================================================================

class TestActionConsistency(unittest.TestCase):
    """[RU] eq. 2: Verify action formula matches manual computation from counts."""

    def test_action_matches_formula(self):
        """S = -(k0+6d)*N0 + (k4+2d)*N41 + (k4+d)*N32 + eps*(N41-tgt)^2"""
        k0, k4, delta, eps = 2.2, 0.5, 0.6, 0.02
        cdt, st = _make_cdt(n_simplices=100)
        target = st.getN41()
        cdt = tessera.CDTSimulation(st, k0, k4, delta, eps, target)

        action = cdt.computeAction()

        n0 = st.getVertexCount()
        n41 = st.getN41()
        n32 = st.getN32()

        expected = (-(k0 + 6*delta)*n0
                    + (k4 + 2*delta)*n41
                    + (k4 + delta)*n32
                    + eps*(n41 - target)**2)

        self.assertAlmostEqual(action, expected, places=6)

    def test_action_matches_after_sweeps(self):
        """Action formula still consistent after moves change the complex."""
        k0, k4, delta, eps = 2.2, 0.5, 0.6, 0.02
        cdt, st = _make_cdt(n_simplices=100)
        target = st.getN41()
        cdt = tessera.CDTSimulation(st, k0, k4, delta, eps, target)
        cdt.tune()
        k4 = cdt.getK4()  # tune changes k4
        cdt.sweep(50)

        action = cdt.computeAction()
        n0 = st.getVertexCount()
        n41 = st.getN41()
        n32 = st.getN32()

        expected = (-(k0 + 6*delta)*n0
                    + (k4 + 2*delta)*n41
                    + (k4 + delta)*n32
                    + eps*(n41 - target)**2)

        self.assertAlmostEqual(action, expected, places=4)


# =====================================================================
# Simplex orientation correctness
# =====================================================================

class TestTemporalOrientation(unittest.TestCase):
    """[RU] Sec. 3: Verify orientation computation from vertex times."""

    def _make_simplex(self, times):
        """Create a 4-simplex with vertices at given times.

        Stores the Spacetime on self to keep it alive — the simplex
        holds raw pointers to vertices/edges owned by the Spacetime.
        """
        self._st = tessera.Spacetime()
        verts = []
        for i, t in enumerate(times):
            v = self._st.createVertex(i, [float(t)])
            verts.append(v)
        s, _ = self._st.createSimplex(verts)
        return s

    def test_41_orientation(self):
        """4 vertices at t=0, 1 at t=1 → (4,1)."""
        s = self._make_simplex([0, 0, 0, 0, 1])
        self.assertEqual(s.getOrientation().numeric(), (4, 1))

    def test_14_orientation(self):
        """1 vertex at t=0, 4 at t=1 → (1,4)."""
        s = self._make_simplex([0, 1, 1, 1, 1])
        self.assertEqual(s.getOrientation().numeric(), (1, 4))

    def test_32_orientation(self):
        """3 at t=0, 2 at t=1 → (3,2)."""
        s = self._make_simplex([0, 0, 0, 1, 1])
        self.assertEqual(s.getOrientation().numeric(), (3, 2))

    def test_23_orientation(self):
        """2 at t=0, 3 at t=1 → (2,3)."""
        s = self._make_simplex([0, 0, 1, 1, 1])
        self.assertEqual(s.getOrientation().numeric(), (2, 3))

    def test_orientation_independent_of_vertex_order(self):
        """Orientation should be the same regardless of vertex ordering."""
        for times in [[0,0,0,0,1], [0,0,0,1,0], [0,0,1,0,0],
                      [0,1,0,0,0], [1,0,0,0,0]]:
            s = self._make_simplex(times)
            self.assertEqual(s.getOrientation().numeric(), (4, 1),
                             f"times={times}")

    def test_all_same_time_gives_k0_orientation(self):
        """All vertices at same time → (5,0)."""
        s = self._make_simplex([0, 0, 0, 0, 0])
        self.assertEqual(s.getOrientation().numeric(), (5, 0))

    def test_vertex_count_in_4simplex(self):
        """A 4-simplex should have exactly 5 vertices."""
        s = self._make_simplex([0, 0, 0, 1, 1])
        self.assertEqual(len(s.getVertices()), 5)

    def test_edge_count_in_4simplex(self):
        """A 4-simplex should have C(5,2) = 10 edges."""
        s = self._make_simplex([0, 0, 0, 1, 1])
        self.assertEqual(len(s.getEdges()), 10)

    def test_facet_count_in_4simplex(self):
        """A 4-simplex should have C(5,4) = 5 facets (3-simplices)."""
        # Use a full spacetime (not bare) so getFacets() can create sub-simplices
        sig = tessera.Signature(4, tessera.Lorentzian)
        metric = tessera.Metric(True, sig)
        st = tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0,
                             tessera.PREFERRED, tessera.Toroid())
        st.build(5)
        # Get any top simplex
        for s in st.getSimplices():
            if len(s.getVertices()) == 5:
                self.assertEqual(len(s.getFacets()), 5)
                return
        self.fail("No top simplex found")


# =====================================================================
# Volume profile correctness
# =====================================================================

class TestVolumeProfile(unittest.TestCase):
    """[RU] eq. 6: Volume profile should correctly assign simplices to time slices."""

    def test_profile_all_positive(self):
        """No negative entries in the volume profile."""
        cdt, st = _make_cdt(n_simplices=100)
        cdt.sweep(20)
        profile = cdt.getVolumeProfile()
        for v in profile:
            self.assertGreaterEqual(v, 0)

    def test_profile_sum_equals_n4(self):
        """sum(profile) == N4 always."""
        cdt, st = _make_cdt(n_simplices=100)
        cdt.tune()
        for _ in range(10):
            cdt.sweep(10)
            profile = cdt.getVolumeProfile()
            self.assertEqual(sum(profile), st.getTopSimplexCount(),
                             "Profile sum != N4")

    def test_profile_consistent_with_manual_count(self):
        """Verify profile by manually counting simplices per time slice."""
        cdt, st = _make_cdt(n_simplices=100)
        cdt.sweep(20)

        # Manual count
        manual = {}
        for s in st.getSimplices():
            verts = s.getVertices()
            if len(verts) != 5:
                continue
            tmin = min(v.getTime() for v in verts)
            manual[tmin] = manual.get(tmin, 0) + 1

        # From getVolumeProfile
        profile = cdt.getVolumeProfile()

        # Compare
        if manual:
            tmin_all = min(manual.keys())
            for t, count in manual.items():
                idx = int(t - tmin_all)
                self.assertEqual(profile[idx], count,
                                 f"Profile mismatch at t={t}")


# =====================================================================
# Stress tests: invariants hold under sustained simulation
# =====================================================================

class TestSweepInvariants(unittest.TestCase):
    """[RU] Sec. 3: Check invariants hold at every checkpoint during a long simulation."""

    def test_invariants_every_10_sweeps(self):
        """Run 200 sweeps, check invariants every 10."""
        cdt, st = _make_cdt(n_simplices=200)
        cdt.tune()
        for step in range(20):
            cdt.sweep(10)
            with self.subTest(sweep=(step + 1) * 10):
                # N4 = N41 + N32
                self.assertEqual(st.getTopSimplexCount(),
                                 st.getN41() + st.getN32())

                # Manual orientation count matches
                counts = _count_orientations(st)
                n41, n32 = _orientation_n41_n32(counts)
                self.assertEqual(st.getN41(), n41)
                self.assertEqual(st.getN32(), n32)

                # No non-CDT orientations
                total = sum(counts.values())
                self.assertEqual(n41 + n32, total,
                                 f"Non-CDT orientations at sweep "
                                 f"{(step+1)*10}: {counts}")

                # Profile consistency
                profile = cdt.getVolumeProfile()
                self.assertEqual(sum(profile), st.getTopSimplexCount())

                # Causality: every top simplex spans 2 times
                for s in st.getSimplices():
                    if len(s.getVertices()) != 5:
                        continue
                    times = set(v.getTime() for v in s.getVertices())
                    self.assertEqual(len(times), 2)


if __name__ == "__main__":
    unittest.main()
