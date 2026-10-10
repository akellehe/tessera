"""One-forward-step Van Raamsdonk metric on Edge / Simplex (#245).

Covers the three acceptance criteria:

  1. the spacelike VR metric law d_VR = −log(I/iMax) with the I < ε·iMax
     floor — ``Edge.van_raamsdonk_length`` (static law). It returns the LENGTH,
     not its square (#639): the whole Edge API speaks lengths now;
  2. the time-aware signed length — a forward-time *worldline* edge
     (endpoints on different time slices) is null (0), a same-slice edge is
     spacelike — ``Edge.van_raamsdonk_length_for``;
  3. fail-loudly admissibility — an inadmissible spacelike simplex raises, a
     valid one does not, and a simplex carrying a null/timelike (worldline)
     edge is skipped — ``Simplex.assert_spacelike_admissible``.

Continuous/spectral only; Lorentzian (signed squared lengths), not Euclidean.
"""

from __future__ import annotations

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

IMAX = 2.0 * math.log(2.0)  # kIMax: maximally-entangled qubit pair


# --------------------------------------------------------------------------- #
# Fixtures (mirror the cobordism bulk-synthesis idiom).
# --------------------------------------------------------------------------- #
def _spacetime(dim, topology):
    sig = tessera.Signature(dim, tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    return tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0,
                             tessera.PREFERRED, topology)


def _solid_triangle():
    st = _spacetime(2, tessera.SolidSimplex(2))
    st.build()
    return st


def _edge_map(st):
    out = {}
    for e in st.get_edge_list().to_vector():
        a, b = e.get_source().get_id(), e.get_target().get_id()
        out[(min(a, b), max(a, b))] = e
    return out


def _vertex_map(st):
    return {v.get_id(): v for v in st.get_vertex_list().to_vector()}


def _triangle(st):
    for s in st.get_simplices():
        if len(s.get_edges()) == 3:
            return s
    raise AssertionError("no triangle simplex in spacetime")


# --------------------------------------------------------------------------- #
# (1) The spacelike VR metric law (Edge static).
# --------------------------------------------------------------------------- #
class TestMetricLaw(unittest.TestCase):
    def test_spacelike_matches_minus_log(self):
        for frac in (0.9, 0.5, 0.1):
            I = frac * IMAX
            expected = -math.log(I / IMAX)
            got = tessera.Edge.van_raamsdonk_length(I, IMAX, 1e-10)
            self.assertAlmostEqual(got, expected, places=12)

    def test_maximal_correlation_is_zero_length(self):
        self.assertAlmostEqual(
            tessera.Edge.van_raamsdonk_length(IMAX, IMAX), 0.0, places=12)

    def test_floor_below_epsilon(self):
        for eps in (1e-10, 1e-3, 0.1):
            cap = -math.log(eps)
            self.assertAlmostEqual(
                tessera.Edge.van_raamsdonk_length(0.0, IMAX, eps),
                cap, places=9)
            self.assertAlmostEqual(
                tessera.Edge.van_raamsdonk_length(0.5 * eps * IMAX, IMAX, eps),
                cap, places=9)
        eps = 0.1
        self.assertLess(
            tessera.Edge.van_raamsdonk_length(2.0 * eps * IMAX, IMAX, eps),
            -math.log(eps))


# --------------------------------------------------------------------------- #
# (2) Time-aware signed squared length: worldline edges are null (Edge).
# --------------------------------------------------------------------------- #
class TestForwardTimeNullEdges(unittest.TestCase):
    def test_worldline_edges_null_same_slice_spacelike(self):
        st = _solid_triangle()  # triangle 0-1-2
        v = _vertex_map(st)
        v[0].set_time(0.0)
        v[1].set_time(0.0)  # 0,1 on the t=0 slice
        v[2].set_time(1.0)  # 2 on the t=1 slice (one forward step)
        em = _edge_map(st)

        I = 0.5 * IMAX  # un-floored, so we test the real metric value
        expected_spacelike = -math.log(0.5)

        # (0, 1): same slice -> spacelike, the metric law value.
        self.assertAlmostEqual(
            em[(0, 1)].van_raamsdonk_length_for(I, IMAX, 1e-10),
            expected_spacelike, places=12)

        # (0, 2) and (1, 2): cross-slice worldline edges -> null.
        self.assertEqual(
            em[(0, 2)].van_raamsdonk_length_for(I, IMAX, 1e-10), 0.0)
        self.assertEqual(
            em[(1, 2)].van_raamsdonk_length_for(I, IMAX, 1e-10), 0.0)


# --------------------------------------------------------------------------- #
# (3) Fail-loudly admissibility of spacelike simplices (Simplex).
# --------------------------------------------------------------------------- #
class TestSpacelikeAdmissibility(unittest.TestCase):
    def _triangle_with(self, s01, s02, s12):
        st = _solid_triangle()
        em = _edge_map(st)
        em[(0, 1)].set_length(cmath.sqrt(complex(s01)))
        em[(0, 2)].set_length(cmath.sqrt(complex(s02)))
        em[(1, 2)].set_length(cmath.sqrt(complex(s12)))
        return _triangle(st)

    def test_valid_spacelike_triangle_ok(self):
        tri = self._triangle_with(1.0, 1.0, 1.0)  # equilateral, lengths 1
        tri.assert_spacelike_admissible()  # must not raise

    def test_inadmissible_spacelike_triangle_raises(self):
        # lengths 1, 1, 3 -> 1 + 1 < 3: triangle inequality violated.
        tri = self._triangle_with(1.0, 1.0, 9.0)
        with self.assertRaises(RuntimeError):
            tri.assert_spacelike_admissible()

    def test_worldline_edge_is_skipped(self):
        # One timelike (squaredLength < 0) edge: not a spacelike cell, so the
        # spacelike check is skipped (no raise).
        tri = self._triangle_with(1.0, 1.0, -1.0)
        tri.assert_spacelike_admissible()  # must not raise


if __name__ == "__main__":
    unittest.main()
