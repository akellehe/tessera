"""Tests for the Regge equation solver."""
import cmath
import math
import unittest

import tessera


def _make_spacetime(n_simplices=200):
    sig = tessera.Signature(4, tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    st = tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0, tessera.PREFERRED,
                         tessera.Toroid())
    st.build(n_simplices)
    return st


class TestDihedralAngles(unittest.TestCase):
    """Basic sanity checks for dihedral angle computation."""

    def test_dihedral_angles_are_positive(self):
        st = _make_spacetime(20)
        matter = tessera.MatterConfiguration()
        solver = tessera.ReggeSolver(st, matter)

        # Find a top-simplex and one of its hinges (triangles, 3 verts)
        for s in st.getSimplices():
            if len(s.getVertices()) == 5:  # top-simplex in 4D
                for facet in s.getFacets():
                    for hinge in facet.getFacets():
                        if len(hinge.getVertices()) == 3:
                            angle = solver.dihedralAngle(s, hinge)
                            # The Lorentzian dihedral angle is complex: the real
                            # part is the rotation content (continuous in (0, π)
                            # for a spacelike-normal wedge, quantized to
                            # {0, π/2, π} in the boost/crossing regimes) and the
                            # imaginary part is the boost. Both bounds hold on
                            # the real part in every regime.
                            self.assertTrue(cmath.isfinite(angle))
                            self.assertGreaterEqual(angle.real, 0.0)
                            self.assertLessEqual(angle.real, math.pi)
                            return
        self.skipTest("No suitable hinge found")

    def test_deficit_angles_exist(self):
        st = _make_spacetime(20)
        matter = tessera.MatterConfiguration()
        solver = tessera.ReggeSolver(st, matter)

        # Find a hinge and compute its deficit angle
        for s in st.getSimplices():
            if len(s.getVertices()) == 3 and len(s.getCofaces()) > 0:
                eps = solver.deficitAngle(s)
                # The Lorentzian deficit is complex: Re is the angle defect,
                # Im the boost (rapidity) content around the hinge.
                self.assertIsInstance(eps, complex)
                self.assertTrue(cmath.isfinite(eps))
                return
        self.skipTest("No hinge with cofaces found")


class TestReggeAction(unittest.TestCase):
    """Test that the Regge action is computable."""

    def test_regge_action_is_finite(self):
        st = _make_spacetime(20)
        matter = tessera.MatterConfiguration()
        solver = tessera.ReggeSolver(st, matter)
        S = solver.reggeAction()
        # Complex Lorentzian action: Re from the angle defects, Im from the
        # boost content of spacelike hinges.
        self.assertTrue(cmath.isfinite(S), f"Regge action should be finite, got {S}")


class TestActionGradientNorm(unittest.TestCase):
    """In vacuum (no matter), ||∇S||² measures how far from the Regge equations."""

    def test_vacuum_gradient_norm_is_nonnegative(self):
        st = _make_spacetime(20)
        matter = tessera.MatterConfiguration()  # no matter = vacuum
        solver = tessera.ReggeSolver(st, matter)
        F = solver.actionGradientNorm()
        self.assertGreaterEqual(F, 0.0)


class TestMatterConfiguration(unittest.TestCase):
    """Test matter configuration specification."""

    def test_worldline_mass_creates_nonzero_gradient(self):
        st = _make_spacetime(20)
        matter = tessera.MatterConfiguration()
        v = st.getVertexList().toVector()[0]
        matter.setWorldlineMass(v, 1.0, st)
        solver = tessera.ReggeSolver(st, matter)
        # With matter, gradient norm should be nonzero (not at solution yet)
        F = solver.actionGradientNorm()
        self.assertGreater(F, 0.0)

    def test_matter_action_is_negative(self):
        """S_matter = -M Σ √(-ℓ²) should be negative for positive mass."""
        st = _make_spacetime(20)
        matter = tessera.MatterConfiguration()
        v = st.getVertexList().toVector()[0]
        matter.setWorldlineMass(v, 1.0, st)
        solver = tessera.ReggeSolver(st, matter)
        S_matter = solver.matterAction()
        self.assertLess(S_matter, 0.0,
            "Proper-time matter action should be negative for positive mass")

    def test_radial_profile(self):
        st = _make_spacetime(20)
        matter = tessera.MatterConfiguration()
        v = st.getVertexList().toVector()[0]
        # Exponential profile: ρ(r) = exp(-r)
        matter.setRadialProfile(v, lambda r: math.exp(-r))
        # Radial profiles don't contribute to proper-time action,
        # so just check that it doesn't crash
        solver = tessera.ReggeSolver(st, matter)
        S = solver.totalAction()
        self.assertTrue(cmath.isfinite(S))


class TestHingeContent(unittest.TestCase):
    def test_hinge_content_is_real_or_imaginary(self):
        st = _make_spacetime(20)
        # Creating a ReggeSolver registers hinges (triangles) via getFacets()
        matter = tessera.MatterConfiguration()
        tessera.ReggeSolver(st, matter)
        for s in st.getSimplices():
            if len(s.getVertices()) == 3 and len(s.getEdges()) >= 3:
                content = tessera.ReggeSolver.hingeContent(s)
                # The content of a triangular hinge is its complex area. On
                # real signed l^2 the Gram determinant is real, so the content
                # is either real (spacelike triangle) or purely imaginary
                # (timelike) — never generic complex. Assert that invariant
                # instead of positivity.
                self.assertTrue(cmath.isfinite(content))
                self.assertAlmostEqual(min(abs(content.real), abs(content.imag)),
                                       0.0, places=9)
                return
        self.skipTest("No triangle found")
