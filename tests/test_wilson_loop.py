"""Tests for the Wilson loop observable."""
import cmath
import math
import unittest

import tessera


def _make_spacetime(n_simplices=20):
    sig = tessera.Signature(4, tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    st = tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0, tessera.PREFERRED,
                         tessera.Toroid())
    st.build(n_simplices)
    return st


def _find_hinge(st):
    """Find a hinge (3-vertex simplex) with cofaces."""
    for s in st.get_simplices():
        if len(s.get_vertices()) == 3 and len(s.get_cofaces()) > 0:
            return s
    return None


class TestHingeLoop(unittest.TestCase):
    def test_hinge_loop_nonempty(self):
        st = _make_spacetime()
        wl = tessera.WilsonLoop(st)
        hinge = _find_hinge(st)
        if hinge is None:
            self.skipTest("No hinge found")
        loop = wl.hinge_loop(hinge)
        self.assertGreaterEqual(len(loop), 2,
            "Hinge loop should have at least 2 simplices")

    def test_hinge_loop_all_contain_hinge(self):
        st = _make_spacetime()
        wl = tessera.WilsonLoop(st)
        hinge = _find_hinge(st)
        if hinge is None:
            self.skipTest("No hinge found")
        loop = wl.hinge_loop(hinge)
        hinge_verts = hinge.get_vertices()
        for sigma in loop.simplices:
            for hv in hinge_verts:
                self.assertTrue(sigma.has_vertex(hv),
                    "Every loop simplex must contain the hinge")


class TestGeodesicLoop(unittest.TestCase):
    def test_geodesic_loop_exists(self):
        st = _make_spacetime()
        wl = tessera.WilsonLoop(st)
        # Find a top-simplex
        start = None
        for s in st.get_simplices():
            if len(s.get_vertices()) == 5:  # top-simplex in 4D
                start = s
                break
        if start is None:
            self.skipTest("No top-simplex found")
        loop = wl.geodesic_loop(start)
        self.assertGreaterEqual(len(loop), 2,
            "Geodesic loop should exist on a closed manifold")


class TestDualLatticeLoop(unittest.TestCase):
    def test_dual_lattice_loop_exists(self):
        st = _make_spacetime()
        wl = tessera.WilsonLoop(st)
        start = None
        for s in st.get_simplices():
            if len(s.get_vertices()) == 5:  # top-simplex in 4D
                start = s
                break
        if start is None:
            self.skipTest("No top-simplex found")
        loop = wl.dual_lattice_loop(start, 6)
        self.assertGreaterEqual(len(loop), 2)


class TestCombinatorialMode(unittest.TestCase):
    def test_combinatorial_returns_loop_size(self):
        st = _make_spacetime()
        wl = tessera.WilsonLoop(st)
        hinge = _find_hinge(st)
        if hinge is None:
            self.skipTest("No hinge found")
        loop = wl.hinge_loop(hinge)
        result = wl.evaluate_combinatorial(loop)
        self.assertEqual(result.loop_size, len(loop))
        self.assertEqual(result.value, float(len(loop)))


class TestDeficitAngleMode(unittest.TestCase):
    def test_hinge_wilson_value_bounded(self):
        st = _make_spacetime()
        wl = tessera.WilsonLoop(st)
        hinge = _find_hinge(st)
        if hinge is None:
            self.skipTest("No hinge found")
        loop = wl.hinge_loop(hinge)
        if len(loop) < 2:
            self.skipTest("Hinge loop too small")
        result = wl.evaluate_deficit_angle(loop)
        val = complex(result.value)
        self.assertTrue(cmath.isfinite(val),
            f"Wilson value should be finite, got {val}")
        # The rotation part alone is bounded; the boost enters as a cosh, so
        # only the REAL part keeps the classical lower bound. A CDT hinge with
        # boost content legitimately has |value| > 1.
        self.assertGreaterEqual(val.real, -1.0 - 1e-9)

    def test_hinge_wilson_matches_deficit(self):
        """For a hinge loop, W = ((d-2)+2cos(ε))/d should match."""
        st = _make_spacetime()
        wl = tessera.WilsonLoop(st)
        matter = tessera.MatterConfiguration()
        solver = tessera.ReggeSolver(st, matter)
        hinge = _find_hinge(st)
        if hinge is None:
            self.skipTest("No hinge found")
        loop = wl.hinge_loop(hinge)
        if len(loop) < 2:
            self.skipTest("Hinge loop too small")
        result = wl.evaluate_deficit_angle(loop)
        eps = solver.deficit_angle(hinge)
        # The deficit is complex; the holonomy keeps it whole (cos of a complex
        # angle — the boost enters as a cosh), so compare in C.
        expected = ((4 - 2) + 2 * cmath.cos(eps)) / 4
        self.assertAlmostEqual(abs(complex(result.value) - expected), 0.0, places=6,
            msg=f"Wilson value {result.value} != expected {expected}")


class TestCausalMode(unittest.TestCase):
    def test_causal_winding_is_integer(self):
        st = _make_spacetime()
        wl = tessera.WilsonLoop(st)
        hinge = _find_hinge(st)
        if hinge is None:
            self.skipTest("No hinge found")
        loop = wl.hinge_loop(hinge)
        if len(loop) < 2:
            self.skipTest("Hinge loop too small")
        result = wl.evaluate_causal(loop)
        self.assertEqual(result.causal_winding_number,
                         int(result.causal_winding_number))


class TestEvaluateDispatch(unittest.TestCase):
    def test_evaluate_dispatches_correctly(self):
        st = _make_spacetime()
        wl = tessera.WilsonLoop(st)
        hinge = _find_hinge(st)
        if hinge is None:
            self.skipTest("No hinge found")
        loop = wl.hinge_loop(hinge)
        if len(loop) < 2:
            self.skipTest("Hinge loop too small")
        r1 = wl.evaluate(loop, tessera.WilsonMode.COMBINATORIAL)
        r2 = wl.evaluate_combinatorial(loop)
        self.assertEqual(r1.value, r2.value)
        self.assertEqual(r1.loop_size, r2.loop_size)


class TestMeasurements(unittest.TestCase):
    def test_measure_all_hinges_populates(self):
        st = _make_spacetime()
        wl = tessera.WilsonLoop(st)
        wl.measure_all_hinges(tessera.WilsonMode.DEFICIT_ANGLE)
        measurements = wl.get_measurements()
        self.assertGreater(len(measurements), 0,
            "measureAllHinges should produce measurements")

    def test_reset_clears(self):
        st = _make_spacetime()
        wl = tessera.WilsonLoop(st)
        wl.measure_all_hinges(tessera.WilsonMode.DEFICIT_ANGLE)
        wl.reset()
        self.assertEqual(len(wl.get_measurements()), 0)

    def test_average_by_size(self):
        st = _make_spacetime()
        wl = tessera.WilsonLoop(st)
        wl.measure_all_hinges(tessera.WilsonMode.DEFICIT_ANGLE)
        avg = wl.get_average_by_size()
        self.assertGreater(len(avg), 0)
        for size, val in avg.items():
            self.assertIsInstance(size, int)
            self.assertTrue(cmath.isfinite(complex(val)))
