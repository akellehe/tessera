# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The spectral-dimension driver (`tessera.drivers.entanglement_spectral`).

Covers the instrument:
* the return probability of the mutual-information graph is the exact
  heat-kernel trace: a complete graph with equal weights and a path graph
  match their closed forms;
* the readings: the peak overshoots the lattice dimension (1.21 for a long
  chain), while D_half at the half-decay time reads it (1.0 for the chain,
  2.0 for a square grid), on exact spectra and on small graphs;
* the edge lists carry only the pairs above the floor, or only the pairs
  within the scale of the skeleton walk;
* the analysis of a simulated network gives one curve per slice, with
  finite readings and the fit fields;
* the command line runs both schedules end to end and writes the figure.
The tests that simulate skip cleanly when tessera was built without the
quantum subsystem; the graph tests need it too.
"""

import math
import os
import tempfile
import unittest

import numpy as np

from tessera.drivers import entanglement_spectral as es


def _quantum_available():
    try:
        import tessera.quantum as quantum_package
    except ImportError:
        return False
    return bool(quantum_package.is_available())


def _path_graph(n):
    return [(i, i + 1, 1.0) for i in range(n - 1)]


def _grid_graph(side):
    edges = []
    for r in range(side):
        for c in range(side):
            v = r * side + c
            if c + 1 < side:
                edges.append((v, v + 1, 1.0))
            if r + 1 < side:
                edges.append((v, v + side, 1.0))
    return edges


@unittest.skipUnless(_quantum_available(), "tessera was built without the quantum subsystem")
class TestReturnProbability(unittest.TestCase):

    def test_the_complete_graph_matches_its_closed_form(self):
        n, w = 6, 0.3
        edges = [(i, j, w) for i in range(n) for j in range(i + 1, n)]
        sigmas = es.sigma_grid(0.01, 100.0, 25)
        P = es.return_probability(n, edges, sigmas)
        expected = (1.0 + (n - 1) * np.exp(-sigmas * n * w)) / n
        np.testing.assert_allclose(P, expected, rtol=1e-8, atol=1e-10)

    def test_the_path_graph_matches_its_spectrum(self):
        n = 7
        sigmas = es.sigma_grid(0.05, 50.0, 20)
        P = es.return_probability(n, _path_graph(n), sigmas)
        lam = 2.0 - 2.0 * np.cos(math.pi * np.arange(n) / n)
        expected = np.exp(-np.outer(sigmas, lam)).sum(axis=1) / n
        np.testing.assert_allclose(P, expected, rtol=1e-8, atol=1e-10)

    def test_no_edges_means_no_return_decay(self):
        np.testing.assert_allclose(es.return_probability(5, [], [0.1, 1.0, 10.0]), 1.0)

    def test_the_readings_of_a_chain_and_a_grid(self):
        # exact spectra, so the readings are tested apart from the trace
        sigmas = es.sigma_grid(0.01, 1e4, 80)
        n = 4000
        lam = 2.0 - 2.0 * np.cos(math.pi * np.arange(n) / n)
        chain = es.readings(n, sigmas, np.exp(-np.outer(sigmas, lam)).sum(axis=1) / n)
        self.assertAlmostEqual(chain["peak"], 1.21, delta=0.03)      # the lattice overshoot
        self.assertAlmostEqual(chain["sigma_peak"], 0.8, delta=0.3)
        self.assertAlmostEqual(chain["D_half"], 1.0, delta=0.03)     # the dimension
        side = 60
        l1 = 2.0 - 2.0 * np.cos(math.pi * np.arange(side) / side)
        lam = (l1[:, None] + l1[None, :]).ravel()
        grid = es.readings(side * side, sigmas,
                           np.exp(-np.outer(sigmas, lam)).sum(axis=1) / (side * side))
        self.assertGreater(grid["peak"], 2.2)
        self.assertAlmostEqual(grid["D_half"], 2.0, delta=0.12)
        self.assertGreater(grid["sigma_half"], 0)
        for key in ("D_inf", "C", "B", "chi2", "D_short"):
            self.assertIn(key, chain["fit"])
        self.assertTrue(np.all(sigmas[sigmas <= chain["sigma_peak"]].size >= 4))

    def test_small_graphs_read_their_dimension_at_the_half_decay_time(self):
        sigmas = es.sigma_grid(0.01, 1e4, 80)
        chain = es.spectral_curve(120, _path_graph(120), sigmas)
        self.assertAlmostEqual(chain["D_half"], 0.97, delta=0.05)
        grid = es.spectral_curve(121, _grid_graph(11), sigmas)
        self.assertAlmostEqual(grid["D_half"], 1.96, delta=0.08)
        self.assertEqual(chain["edges"], 119)
        self.assertEqual(len(chain["dS"]), 80)
        self.assertEqual(len(chain["dS_smooth"]), 80)


class TestEdges(unittest.TestCase):
    MI = np.array([[0.0, 0.5, 1e-14], [0.5, 0.0, 0.2], [1e-14, 0.2, 0.0]])

    def test_mutual_information_edges_are_the_pairs_above_the_floor(self):
        self.assertEqual(es.mi_edges(self.MI, 1e-12), [(0, 1, 0.5), (1, 2, 0.2)])
        self.assertEqual(es.mi_edges(self.MI, 0.3), [(0, 1, 0.5)])

    def test_skeleton_edges_are_the_pairs_within_the_scale(self):
        S = np.array([0.6, 0.6, 0.6])
        edges = es.skeleton_edges(self.MI, S, 1e-12, "inverse", 1.0)
        # lengths 1/I are 2 and 5, scaled to mean 1: 4/7 and 10/7; only (0, 1) is within 1
        self.assertEqual(edges, [(0, 1, 1.0)])
        self.assertEqual(es.skeleton_edges(self.MI, S, 1e-12, "inverse", 2.0),
                         [(0, 1, 1.0), (1, 2, 1.0)])
        self.assertEqual(es.skeleton_edges(self.MI, S, 1.0, "inverse", 2.0), [])

    def test_the_sigma_grid_is_log_spaced_and_checked(self):
        g = es.sigma_grid(0.01, 100.0, 5)
        np.testing.assert_allclose(g, [0.01, 0.1, 1.0, 10.0, 100.0])
        with self.assertRaises(ValueError):
            es.sigma_grid(1.0, 0.1, 10)
        with self.assertRaises(ValueError):
            es.sigma_grid(0.1, 1.0, 3)

    def test_the_fitted_curve_has_the_ambjorn_loll_form(self):
        fit = {"D_inf": 4.0, "C": 119.0, "B": 54.0}
        np.testing.assert_allclose(es.fitted_curve(fit, [0.0, 54.0]), [4.0 - 119.0 / 54.0,
                                                                        4.0 - 119.0 / 108.0])


@unittest.skipUnless(_quantum_available(), "tessera was built without the quantum subsystem")
class TestNetwork(unittest.TestCase):

    def test_one_curve_per_slice_with_finite_readings(self):
        from tessera.drivers import entanglement_complex as ec
        result = ec.simulate(4, None, "1/2", 0, "global", 5, "all", False)
        sigmas = es.sigma_grid(0.01, 100.0, 30)
        rep = es.analyse(result, sigmas, "mi", 1e-12)
        self.assertEqual(len(rep["curves"]), 6)
        self.assertEqual(rep["T"], 5)
        for c in rep["curves"]:
            self.assertEqual(len(c["P"]), 30)
            self.assertTrue(np.all(np.isfinite(c["P"])))
            self.assertTrue(np.all((c["P"] > 0) & (c["P"] <= 1 + 1e-12)))
            self.assertGreaterEqual(c["edges"], 2)
            self.assertTrue(math.isfinite(c["peak"]))
        # the input pairs are disconnected, so P never falls to 1/sqrt(n) at t = 0
        self.assertTrue(math.isnan(rep["curves"][0]["D_half"]))
        self.assertTrue(math.isfinite(rep["final"]["D_half"]))
        skeleton = es.analyse(result, sigmas, "skeleton", 1e-12, "log1p", 1.0)
        self.assertEqual(len(skeleton["curves"]), 6)
        with self.assertRaises(ValueError):
            es.analyse(result, sigmas, "teleport")

    def test_the_command_line_runs_both_schedules(self):
        import io
        from contextlib import redirect_stdout
        out = io.StringIO()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "spectral.png")
            with redirect_stdout(out):
                rc = es.main(["--qubits", "4", "--timesteps", "3", "--sigma-count", "12",
                              "--save", path, "--no-show"])
            self.assertEqual(rc, 0)
            self.assertTrue(os.path.exists(path) and os.path.getsize(path) > 0)
        text = out.getvalue()
        self.assertIn("SPECTRAL DIMENSION (4 qubits, all schedule", text)
        self.assertIn("SPECTRAL DIMENSION (4 qubits, chain schedule", text)
        self.assertIn("peak D_S", text)
        self.assertIn("D_half", text)


if __name__ == "__main__":
    unittest.main()
