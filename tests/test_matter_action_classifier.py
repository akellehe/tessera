"""matterAction uses the canonical Edge causal classifier (#581 scope item 4).

``ReggeSolver::matterAction`` hand-rolled its causal test as
``Re(l^2) < 0``, divergent from the canonical ``Edge::isTimelike()`` (which
reads the imaginary part of the complex length).  It now classifies via the
canonical helper; the proper time of a timelike step is ``sqrt(-Re l^2)``,
exact under the ordinary-Lorentzian convention (resident ``l^2`` real and
signed, #581 item 6), and null or spacelike worldline steps contribute
nothing.
"""

import math
import unittest

import pytest
import cmath

try:
    import tessera
    _IMPORT_OK = True
except Exception:  # pragma: no cover
    _IMPORT_OK = False

pytestmark = pytest.mark.skipif(not _IMPORT_OK, reason="tessera not built")


def _chain_spacetime(step_sq):
    """A minimal worldline host: one vertex per time slice t = 0..n,
    consecutive slices joined by an edge with squared length step_sq[t].
    With a single vertex per slice the traced worldline is exactly the
    chain."""
    sig = tessera.Signature(2, tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    st = tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0,
                           tessera.PREFERRED, tessera.SolidSimplex(2))
    verts = []
    for t in range(len(step_sq) + 1):
        v = st.create_vertex(t)
        v.set_time(float(t))
        verts.append(v)
    for t, sq in enumerate(step_sq):
        st.create_edge(verts[t], verts[t + 1], cmath.sqrt(complex(sq)))
    return st, verts


class TestMatterActionClassifier(unittest.TestCase):
    def test_timelike_worldline_proper_time(self):
        # steps l^2 = -4 and -9: S = -M*(2 + 3)
        st, verts = _chain_spacetime([-4.0, -9.0])
        for e in st.get_edge_list().to_vector():
            self.assertTrue(e.is_timelike())   # the canonical classifier
        matter = tessera.MatterConfiguration()
        matter.set_worldline_mass(verts[1], 1.5, st)
        solver = tessera.ReggeSolver(st, matter)
        self.assertAlmostEqual(solver.matter_action(), -1.5 * 5.0, delta=1e-12)

    def test_degenerate_worldline_contributes_zero(self):
        # l = 0 is a DEGENERATE (absent) edge, not a lightlike ray. The two were
        # conflated while causal type came from the Euclidean modulus, which can
        # only vanish when the edge itself does (#870). Either way it is not
        # timelike, so it carries no proper time.
        st, verts = _chain_spacetime([0.0, 0.0])
        for e in st.get_edge_list().to_vector():
            self.assertTrue(e.is_degenerate())
            self.assertFalse(e.is_null())
            self.assertFalse(e.is_timelike())
        matter = tessera.MatterConfiguration()
        matter.set_worldline_mass(verts[1], 2.0, st)
        solver = tessera.ReggeSolver(st, matter)
        self.assertEqual(solver.matter_action(), 0.0)

    def test_a_genuinely_lightlike_worldline_contributes_zero(self):
        # The case the old classifier could not express: Re(l) == Im(l) > 0, so
        # the interval vanishes on an edge of nonzero extent. Not timelike, so
        # it carries no proper time either -- but for a physical reason rather
        # than because the edge is absent.
        st, verts = _chain_spacetime([1.0, 1.0])
        component = math.sqrt(0.5)
        for e in st.get_edge_list().to_vector():
            e.set_length(complex(component, component))
            self.assertTrue(e.is_null())
            self.assertFalse(e.is_degenerate())
            self.assertFalse(e.is_timelike())
        matter = tessera.MatterConfiguration()
        matter.set_worldline_mass(verts[1], 2.0, st)
        solver = tessera.ReggeSolver(st, matter)
        self.assertEqual(solver.matter_action(), 0.0)

    def test_spacelike_steps_do_not_contribute(self):
        # one timelike step (l^2 = -4) and one spacelike (l^2 = +1):
        # only the timelike step carries proper time.
        st, verts = _chain_spacetime([-4.0, 1.0])
        matter = tessera.MatterConfiguration()
        matter.set_worldline_mass(verts[1], 1.0, st)
        solver = tessera.ReggeSolver(st, matter)
        self.assertAlmostEqual(solver.matter_action(), -2.0, delta=1e-12)

    def test_total_action_includes_matter_term(self):
        st, verts = _chain_spacetime([-4.0, -9.0])
        matter = tessera.MatterConfiguration()
        matter.set_worldline_mass(verts[1], 1.0, st)
        solver = tessera.ReggeSolver(st, matter)
        self.assertAlmostEqual(solver.total_action() - solver.regge_action(),
                               solver.matter_action(), delta=1e-12)


if __name__ == "__main__":
    unittest.main()
