# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""The exact analytic gradient of the complex dual (Sorkin) Regge action matches
finite differences to machine precision -- the regression guard for the
hand-derived gradient (Simplex::deficitAngleGradient,
Simplex::dualVolumeGradient, ReggeSolver::actionGradientExact). Run on a
Lorentzian 4D CDT mesh so both real (spacelike) and complex (boost) triangle
hinges are exercised."""

import unittest

import tessera
import cmath

_FD = 1e-6           # central-difference step in l^2
_TOL = 1e-5          # analytic vs FD agreement


def _make_cdt(n):
    """A Lorentzian 4D CDT mesh: triangle hinges with both real (spacelike) and
    genuinely complex (boost) deficit angles."""
    sig = tessera.Signature(4, tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    st = tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0, tessera.PREFERRED,
                           tessera.Toroid())
    st.build(n)
    st.materialize_facets()
    return st


def _central(setter, get, h=_FD):
    o = setter(None)
    setter(o + h); sp = get()
    setter(o - h); sm = get()
    setter(o)
    return (sp - sm) / (2.0 * h)


class ExactActionGradientTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.st = _make_cdt(80)
        cls.edges = cls.st.get_edge_list().to_vector()
        cls.emap, cls.eidx = {}, {}
        for i, e in enumerate(cls.edges):
            a, b = e.get_source().get_id(), e.get_target().get_id()
            key = (min(a, b), max(a, b))
            cls.emap[key] = e
            cls.eidx[key] = i
        # In 4D the hinges (codim-2) are triangles.
        cls.hinges = {tuple(sorted(int(v.get_id()) for v in s.get_vertices())): s
                      for s in cls.st.get_simplices()
                      if len(s.get_vertices()) == 3}

    def _action(self):
        return complex(tessera.ReggeSolver(
            self.st, tessera.MatterConfiguration()).dual_regge_action())

    def _set(self, edge):
        def setter(val):
            if val is None:
                return (edge.get_length() * edge.get_length()).real
            edge.set_length(cmath.sqrt(complex(float(val)))); self.st.materialize_facets()
            return None
        return setter

    def test_action_gradient_exact_matches_fd(self):
        rs = tessera.ReggeSolver(self.st, tessera.MatterConfiguration())
        g = [complex(z) for z in rs.action_gradient_exact()]
        self.assertEqual(len(g), len(self.edges))
        worst = 0.0
        saw_complex = False
        for key, edge in list(self.emap.items())[:24]:
            fd = _central(self._set(edge), self._action)
            ana = g[self.eidx[key]]
            worst = max(worst, abs(ana - fd))
            saw_complex = saw_complex or abs(fd.imag) > 1e-3
        self.assertLess(worst, _TOL, f"worst |analytic-FD| = {worst:.2e}")
        self.assertTrue(saw_complex, "expected some boost (complex) gradients")

    def test_deficit_gradient_matches_fd_complex_hinge(self):
        # the most-complex-deficit hinge: exercises the boost branch of d(eps)/dl^2
        hk = max(self.hinges, key=lambda k: abs(
            complex(self.hinges[k].deficit_angle()).imag))
        hs = self.hinges[hk]
        grad = hs.deficit_angle_gradient()
        worst = 0.0
        for e in list(grad)[:8]:
            fd = _central(self._set(self.emap[e]),
                          lambda: complex(hs.deficit_angle()))
            worst = max(worst, abs(complex(grad[e]) - fd))
        self.assertLess(worst, _TOL)

    def test_dualvolume_gradient_matches_fd(self):
        hk = next(iter(self.hinges))
        hs = self.hinges[hk]
        grad = hs.dual_volume_gradient()
        self.assertGreater(len(grad), 0)
        worst = 0.0
        for e in list(grad)[:8]:
            fd = _central(self._set(self.emap[e]), hs.dual_volume)
            worst = max(worst, abs(complex(grad[e]) - complex(fd)))
        self.assertLess(worst, _TOL)

    # --- the Lorentzian action is genuinely complex; never reduce it to Re ---
    # These guard a recurring mistake: building a variational principle / EOM / diagnostic
    # on Re S alone. The imaginary part (boost & spacelike-hinge contributions) is real
    # physics; a stationary-action objective is delta S = 0 for the FULL complex S, i.e.
    # the gradient norm Sum|dS/dl^2|^2 must include |d Im S|^2. Dropping it lets the
    # geometry drift along directions the imaginary part actually constrains (observed:
    # the Re-only objective drifted up in overall scale because d Im S -- which curbs that
    # drift -- was missing). The margins below are comfortably met on the Lorentzian
    # CDT mesh; the threshold catches a regression that silently zeroes the imaginary part.

    def test_action_is_materially_complex(self):
        S = self._action()
        self.assertGreater(
            abs(S.imag), 0.1 * abs(S.real),
            f"Im S={S.imag:.3f} is not material vs Re S={S.real:.3f}; the Lorentzian "
            "action is complex and its imaginary part must not be dropped")

    def test_stationarity_residual_requires_imaginary_part(self):
        rs = tessera.ReggeSolver(self.st, tessera.MatterConfiguration())
        g = [complex(z) for z in rs.action_gradient_exact()]
        reN = sum(z.real * z.real for z in g)   # ||d Re S||^2
        imN = sum(z.imag * z.imag for z in g)   # ||d Im S||^2
        # A Re-only stationary-action objective would use reN and miss imN entirely.
        self.assertGreater(
            imN, 0.1 * reN,
            f"||d Im S||^2={imN:.2f} is not material vs ||d Re S||^2={reN:.2f}; the "
            "stationary-action objective must use the full complex gradient Sum|dS|^2")


if __name__ == "__main__":
    unittest.main()
