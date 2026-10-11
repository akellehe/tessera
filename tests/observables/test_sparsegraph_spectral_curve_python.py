# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Per-sigma spectral-dimension bindings on ``tessera.SparseGraph``.

Covers the ``return_probability`` / ``spectral_dimension_curve`` /
``spectral_dimension_smoothed`` bindings: the same Krylov-Lanczos
heat-kernel + finite-difference machinery ``EmergentGraph`` exposes,
now reachable straight off the dual graph so the CDT examples
(``spectral_dimension.py``, ``probe_*.py``) can drop their hand-rolled
NumPy/SciPy dual-graph diffusion and read the full D_S(sigma) curve from
C++ instead.

Textbook acceptance targets mirror
``tests/quantum/test_spectral_dimension_known_graphs_python.py``:

* 1D chain on N vertices: D_S -> 1 in the diffusion regime.
* 2D square lattice: D_S peaks at 2.
"""
import math
import unittest

import tessera

try:
    from tessera.quantum.holography import EmergentGraph
    HAVE_EMERGENT = True
except ImportError:
    HAVE_EMERGENT = False


def _chain_coo(n):
    """COO (rows, cols) for a 1D open chain on ``n`` vertices."""
    return list(range(n - 1)), [i + 1 for i in range(n - 1)]


def _square_lattice_coo(w, h):
    """COO for an open w x h square lattice (nearest-neighbour)."""
    rows, cols = [], []
    for r in range(h):
        for c in range(w):
            v = r * w + c
            if c + 1 < w:
                rows.append(v)
                cols.append(v + 1)
            if r + 1 < h:
                rows.append(v)
                cols.append(v + w)
    return rows, cols


def _log_sigmas(lo, hi, n):
    return [lo * (hi / lo) ** (k / (n - 1)) for k in range(n)]


class TestReturnProbability(unittest.TestCase):

    def test_in_unit_interval_and_monotone(self):
        n = 40
        rows, cols = _chain_coo(n)
        g = tessera.SparseGraph.from_coo(rows, cols, n)
        sigmas = _log_sigmas(0.5, 200.0, 48)
        P = g.return_probability(sigmas, m=n)  # m=n -> exact trace
        self.assertEqual(len(P), len(sigmas))
        for p in P:
            self.assertGreater(p, 0.0)
            self.assertLessEqual(p, 1.0 + 1e-9)
        for j in range(1, len(P)):
            self.assertLessEqual(P[j], P[j - 1] + 1e-9,
                                 f"P not non-increasing at index {j}")

    def test_matches_diagonal_mean(self):
        """``return_probability(m=N)`` equals the mean over all start
        vertices of the ``diagonal_heat_kernel`` diagonal — ties the new
        binding to the existing exact heat-kernel path."""
        g = tessera.SparseGraph.from_coo([0, 1, 2, 3], [1, 2, 3, 0], 4)
        sigmas = [0.5, 1.0, 2.0, 5.0]
        P = g.return_probability(sigmas, m=4)
        K = g.diagonal_heat_kernel([0, 1, 2, 3], sigmas)
        for j, _ in enumerate(sigmas):
            mean = sum(K[s][j] for s in range(4)) / 4.0
            self.assertAlmostEqual(P[j], mean, places=6)


class TestSpectralDimensionCurve(unittest.TestCase):

    def test_power_law_recovers_exponent(self):
        """For P(sigma) = sigma^{-alpha} the centered finite difference
        gives D_S = -2 d log P / d log sigma = 2 alpha exactly on the
        interior."""
        sigmas = _log_sigmas(0.5, 100.0, 40)
        alpha = 1.5
        P = [s ** (-alpha) for s in sigmas]
        dS = tessera.SparseGraph.spectral_dimension_curve(sigmas, P)
        self.assertEqual(len(dS), len(sigmas))
        for d in dS[1:-1]:
            self.assertAlmostEqual(d, 2.0 * alpha, places=6)

    @unittest.skipUnless(HAVE_EMERGENT, "tessera built without quantum")
    def test_curve_matches_emergent_static(self):
        """``SparseGraph.spectral_dimension_curve`` and
        ``EmergentGraph.spectral_dimension`` are the same inherited
        finite-difference static — confirm the binding points at it."""
        sigmas = _log_sigmas(0.5, 100.0, 40)
        P = [s ** (-1.2) for s in sigmas]
        a = tessera.SparseGraph.spectral_dimension_curve(sigmas, P)
        b = EmergentGraph.spectral_dimension(sigmas, P)
        self.assertEqual(len(a), len(b))
        for x, y in zip(a, b):
            self.assertAlmostEqual(x, y, places=10)

    def test_chain_plateau_near_one(self):
        """1D chain: D_S(sigma) plateau ~ 1 in the diffusion regime."""
        n = 80
        rows, cols = _chain_coo(n)
        g = tessera.SparseGraph.from_coo(rows, cols, n)
        sigmas = _log_sigmas(0.5, 200.0, 64)
        P = g.return_probability(sigmas, m=n)
        dS = tessera.SparseGraph.spectral_dimension_curve(sigmas, P)
        window = sorted(d for s, d in zip(sigmas, dS)
                        if 3.0 <= s <= 30.0 and d == d)
        plateau = window[len(window) // 2] if window else float("nan")
        self.assertAlmostEqual(plateau, 1.0, delta=0.15,
                               msg=f"chain plateau D_S = {plateau:.3f}")


class TestSpectralDimensionSmoothed(unittest.TestCase):

    def test_square_lattice_peak_near_two(self):
        """2D square lattice: smoothed D_S(sigma) peaks at ~ 2."""
        w = h = 16
        n = w * h
        rows, cols = _square_lattice_coo(w, h)
        g = tessera.SparseGraph.from_coo(rows, cols, n)
        sigmas = _log_sigmas(0.1, 1000.0, 64)
        P = g.return_probability(sigmas, m=n)
        dS = tessera.SparseGraph.spectral_dimension_smoothed(sigmas, P, 7, 2)
        finite = [d for d in dS if d == d]
        self.assertTrue(finite, "all-NaN smoothed curve")
        peak = max(finite)
        self.assertAlmostEqual(peak, 2.0, delta=0.25,
                               msg=f"lattice peak D_S = {peak:.3f}")

    def test_bad_window_raises(self):
        sigmas = _log_sigmas(0.5, 50.0, 20)
        P = [s ** (-1.0) for s in sigmas]
        with self.assertRaises(Exception):
            # even window is invalid (must be odd, >= polyOrder + 1)
            tessera.SparseGraph.spectral_dimension_smoothed(sigmas, P, 4, 2)


if __name__ == "__main__":
    unittest.main()
