# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The root-lattice-spacetime driver (`tessera.drivers.root_lattice_spacetime`).

Covers the instrument:
* the Fourier coordinates are orthonormal in the zero-sum hyperplane, the
  first two are the regular n-gon of the Coxeter plane, and they are exact
  (an isometry) for three qubits in two coordinates and four in three, where
  the twelve roots are the vertices of a cuboctahedron;
* the lattice points within two root steps are the known shells of the
  hexagonal lattice for three qubits, and every point has zero sum;
* the causal order is the transitive closure of "later and sharing a qubit",
  its covers are the Hasse diagram, the depths are the longest chains, and
  the tessera Poset holds the covers;
* the walk rides exactly the events of its current qubit and its steps add
  up to e_end - e_start; the geodesic is the shortest path;
* the lengths are a ln(1 + I_0 / I), unnormalised;
* the command line runs end to end, prints what it promises, writes the
  figure in two and three coordinates and the animation.
The end-to-end tests skip cleanly when tessera was built without the quantum
subsystem.
"""

import math
import os
import tempfile
import unittest

import numpy as np

from tessera.drivers import root_lattice_spacetime as rls


class TestWeightsAndRoots(unittest.TestCase):

    def test_the_fourier_coordinates_are_orthonormal_and_sum_to_zero(self):
        for n in (3, 4, 5, 6, 7):
            for dims in (2, 3):
                W = rls.weights(n, dims)
                self.assertEqual(W.shape, (n, dims))
                np.testing.assert_allclose(W.sum(axis=0), 0.0, atol=1e-12)
                used = min(dims, n - 1)
                np.testing.assert_allclose(W[:, :used].T @ W[:, :used], np.eye(used),
                                           atol=1e-12)
                np.testing.assert_allclose(W[:, used:], 0.0)

    def test_the_coxeter_plane_is_the_regular_polygon(self):
        for n in (3, 4, 6, 7):
            w = rls.coxeter_weights(n)
            np.testing.assert_allclose(np.linalg.norm(w, axis=1), math.sqrt(2.0 / n))
            angles = np.arctan2(w[:, 1], w[:, 0]) % (2 * math.pi)
            np.testing.assert_allclose(angles, 2 * math.pi * np.arange(n) / n, atol=1e-12)

    def test_three_and_four_qubits_are_drawn_exactly(self):
        for n, dims in ((3, 2), (4, 3)):
            W = rls.weights(n, dims)
            for X, Y in ((0, 1), (0, 2), (1, 2), (0, n - 1)):
                self.assertAlmostEqual(np.linalg.norm(W[X] - W[Y]), math.sqrt(2.0), places=12)

    def test_the_roots_are_the_differences_of_weights(self):
        n = 5
        rs = rls.roots(n)
        self.assertEqual(len(rs), n * (n - 1))
        W = rls.weights(n, 3)
        for (X, Y), m in rs:
            self.assertEqual(m.sum(), 0)
            self.assertEqual(m[Y], 1)
            self.assertEqual(m[X], -1)
            np.testing.assert_allclose(rls.project(m, W), W[Y] - W[X], atol=1e-12)

    def test_four_qubits_give_the_cuboctahedron(self):
        W = rls.weights(4, 3)
        R = rls.project([m for _, m in rls.roots(4)], W)
        self.assertEqual(len(R), 12)
        np.testing.assert_allclose(np.linalg.norm(R, axis=1), math.sqrt(2.0))
        for v in R:                                   # every vertex has four neighbours
            d = np.linalg.norm(R - v, axis=1)
            self.assertEqual(int(np.sum(np.abs(d - math.sqrt(2.0)) < 1e-9)), 4)
        self.assertEqual(len({tuple(np.round(v, 9)) for v in R}), 12)

    def test_four_qubits_project_onto_the_square_lattice_in_the_plane(self):
        w = rls.coxeter_weights(4)
        projected = {tuple(np.round(rls.project(m, w), 9)) for _, m in rls.roots(4)}
        a, b = round(1 / math.sqrt(2.0), 9), round(math.sqrt(2.0), 9)
        expected = {(a, a), (a, -a), (-a, a), (-a, -a), (b, 0.0), (-b, 0.0), (0.0, b), (0.0, -b)}
        self.assertEqual({(float(x), float(y)) for x, y in projected}, expected)

    def test_the_lattice_within_two_steps_is_the_hexagonal_shells(self):
        pts = rls.lattice_points(3, steps=2)
        self.assertEqual(len(pts), 19)                 # origin, 6 roots, 12 second shell
        self.assertTrue((pts.sum(axis=1) == 0).all())
        self.assertEqual(tuple(pts[0]), (0, 0, 0))
        radii = np.round(np.linalg.norm(rls.project(pts, rls.coxeter_weights(3)), axis=1), 6)
        self.assertEqual(sorted(set(radii.tolist())),
                         [0.0, round(math.sqrt(2), 6), round(math.sqrt(6), 6),
                          round(2 * math.sqrt(2), 6)])
        self.assertEqual(len(rls.lattice_points(4, steps=1)), 13)

    def test_qubit_names(self):
        self.assertEqual(rls.qubit_index("A", 4), 0)
        self.assertEqual(rls.qubit_index("d", 4), 3)
        self.assertEqual(rls.qubit_index("2", 4), 2)
        with self.assertRaises(ValueError):
            rls.qubit_index("E", 4)
        with self.assertRaises(ValueError):
            rls.qubit_index("AB", 4)


EVENTS = [(1, (0, 1)), (2, (1, 2)), (3, (3, 4)), (4, (0, 1)), (5, (2, 3))]


class TestCausalOrder(unittest.TestCase):

    def test_the_order_is_the_closure_of_sharing_a_qubit(self):
        P, covers = rls.causal_order(EVENTS)
        self.assertTrue(P[0, 1])                 # (0,1) then (1,2): share 1
        self.assertTrue(P[0, 3])                 # (0,1) twice
        self.assertTrue(P[1, 3])                 # (1,2) then (0,1): share 1
        self.assertTrue(P[1, 4])                 # (1,2) then (2,3): share 2
        self.assertTrue(P[2, 4])                 # (3,4) then (2,3): share 3
        self.assertTrue(P[0, 4])                 # through (1,2)
        self.assertFalse(P[2, 3])                # (3,4) and (0,1) are disjoint
        self.assertFalse(P[0, 2])
        self.assertFalse(P[3, 4])                # (0,1) at t=4 and (2,3) at t=5 are disjoint
        self.assertFalse(np.diag(P).any())
        self.assertFalse((P & P.T).any())
        closure = P.copy()
        for k in range(len(EVENTS)):
            closure |= np.outer(closure[:, k], closure[k, :])
        self.assertTrue((closure == P).all())

    def test_the_covers_are_the_hasse_diagram(self):
        P, covers = rls.causal_order(EVENTS)
        self.assertEqual(sorted(covers), [(0, 1), (1, 3), (1, 4), (2, 4)])
        for i, j in covers:
            self.assertTrue(P[i, j])
            self.assertFalse((P[i, :] & P[:, j]).any())

    def test_the_depths_are_the_longest_chains(self):
        P, _ = rls.causal_order(EVENTS)
        self.assertEqual(rls.depths(P).tolist(), [0, 1, 0, 2, 2])
        self.assertEqual(rls.depths(np.zeros((0, 0), dtype=bool)).tolist(), [])

    def test_the_poset_holds_the_covers(self):
        from tessera import Poset
        ps, P, covers = rls.poset(EVENTS)
        self.assertIsInstance(ps, Poset)
        self.assertEqual(ps.getCoverCount(), len(covers))
        self.assertEqual(sorted(tuple(c) for c in ps.covers), sorted(covers))

    def test_events_are_read_from_the_slices(self):
        slices = [{"t": 0, "pair": None}, {"t": 1, "pair": (2, 0)}, {"t": 2, "pair": (1, 3)}]
        self.assertEqual(rls.events(slices), [(1, (2, 0)), (2, (1, 3))])


class TestPaths(unittest.TestCase):

    def test_the_walk_rides_the_events_of_its_qubit(self):
        steps = rls.walk(EVENTS, 0)
        self.assertEqual(steps, [(1, 0, 1), (2, 1, 2), (5, 2, 3)])
        m = rls.displacement(5, steps)
        self.assertEqual(m.tolist(), [-1, 0, 0, 1, 0])      # e_3 - e_0
        self.assertEqual(rls.walk(EVENTS, 4), [(3, 4, 3), (5, 3, 2)])
        self.assertEqual(rls.walk([], 2), [])

    def test_the_walk_is_a_chain_of_the_causal_set(self):
        P, _ = rls.causal_order(EVENTS)
        times = [e[0] for e in EVENTS]
        steps = rls.walk(EVENTS, 0)
        for (t1, _, _), (t2, _, _) in zip(steps, steps[1:]):
            self.assertTrue(P[times.index(t1), times.index(t2)])

    def test_the_walk_returns_on_a_repeated_pair(self):
        steps = rls.walk([(1, (0, 1)), (2, (0, 1))], 0)
        self.assertEqual(steps, [(1, 0, 1), (2, 1, 0)])
        self.assertEqual(rls.displacement(2, steps).tolist(), [0, 0])

    def test_the_geodesic_is_the_shortest_path(self):
        D = np.array([[0.0, 1.0, 5.0], [1.0, 0.0, 1.0], [5.0, 1.0, 0.0]])
        W = (D > 0).astype(float)
        self.assertEqual(rls.geodesic(D, W, 0, 2), [0, 1, 2])
        self.assertEqual(rls.geodesic(D, W, 0, 0), [0])
        W[0, 1] = W[1, 0] = 0.0
        self.assertEqual(rls.geodesic(D, W, 0, 2), [0, 2])
        W[0, 2] = W[2, 0] = 0.0
        self.assertEqual(rls.geodesic(D, W, 0, 2), [])

    def test_the_lengths_are_the_regularised_log(self):
        I = 0.3
        MI = np.array([[0.0, I, 0.0], [I, 0.0, 2e-13], [0.0, 2e-13, 0.0]])
        D, W = rls.lengths(MI, 1e-12, scale=2.0, reference=0.7)
        self.assertEqual(W.tolist(), [[0, 1, 0], [1, 0, 0], [0, 0, 0]])
        self.assertAlmostEqual(D[0, 1], 2.0 * math.log(1.0 + 0.7 / I), places=12)
        self.assertEqual(D[1, 2], 0.0)
        D0, W0 = rls.lengths(np.zeros((2, 2)), 1e-12, 1.0, 1.0)
        self.assertEqual(W0.sum(), 0)
        self.assertEqual(D0.shape, (2, 2))


def _quantum_available():
    # ``import tessera.quantum`` is the Python package with ``is_available``;
    # ``from tessera import quantum`` can return the C submodule instead.
    try:
        import tessera.quantum as quantum_package
    except ImportError:
        return False
    return bool(quantum_package.is_available())


@unittest.skipUnless(_quantum_available(), "tessera was built without the quantum subsystem")
class TestCommandLine(unittest.TestCase):

    def _run(self, extra):
        import io
        from contextlib import redirect_stdout
        out = io.StringIO()
        with redirect_stdout(out):
            rc = rls.main(extra)
        return rc, out.getvalue()

    def test_the_walk_runs_and_writes_the_figure_in_three_coordinates(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "walk.png")
            rc, out = self._run(["--qubits", "4", "--timesteps", "5", "--save", path,
                                 "--no-show"])
            self.assertEqual(rc, 0)
            self.assertTrue(os.path.exists(path) and os.path.getsize(path) > 0)
        self.assertIn("INFORMATION NETWORK", out)
        self.assertIn("ROOT LATTICE A_3: 12 roots", out)
        self.assertIn("3 Fourier coordinates (exact for 4 qubits)", out)
        self.assertIn("CAUSAL ORDER: 5 events", out)
        self.assertIn("WORLDLINE of a unit of charge from A", out)
        self.assertIn("step  1", out)
        self.assertIn("displacement:", out)

    def test_the_plane_and_the_geodesic(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "geodesic.png")
            rc, out = self._run(["--qubits", "5", "--timesteps", "6", "--path", "geodesic",
                                 "--start", "B", "--end", "D", "--lattice-dims", "2",
                                 "--save", path, "--no-show"])
            self.assertEqual(rc, 0)
            self.assertTrue(os.path.exists(path) and os.path.getsize(path) > 0)
        self.assertIn("GEODESIC from B to D", out)
        self.assertIn("2 Fourier coordinates (a projection for 5 qubits)", out)

    def test_the_animation_has_one_frame_per_slice(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "walk.gif")
            rc, out = self._run(["--qubits", "4", "--timesteps", "3", "--animate", path,
                                 "--no-show"])
            self.assertEqual(rc, 0)
            self.assertIn("ANIMATION: 4 frames", out)
            with Image.open(path) as gif:
                self.assertEqual(gif.n_frames, 4)

    def test_twenty_qubits_run_on_the_pure_engine(self):
        rc, out = self._run(["--qubits", "20", "--timesteps", "4", "--state", "pure",
                             "--no-show"])
        self.assertEqual(rc, 0)
        self.assertIn("ROOT LATTICE A_19: 380 roots", out)
        self.assertIn("381 lattice points within 1 root step(s)", out)
        self.assertIn("a projection for 20 qubits", out)

    def test_the_analysis_is_consistent_with_the_slices(self):
        from tessera.drivers import entanglement_complex as ec
        result = ec.simulate(4, None, "1/2", 0, "global", 6, "all", False)
        rep = rls.analyse(result, 1e-12, 1.0, rls.I_MAX, "walk", 0)
        self.assertEqual(len(rep["events"]), 6)
        self.assertEqual(rep["T"], 6)
        self.assertEqual(rep["weights"].shape, (4, 3))
        visited = rep["visited"]
        self.assertEqual(visited[0], 0)
        self.assertEqual(len(visited), len(rep["steps"]) + 1)
        m = rep["displacement"]
        expected = np.zeros(4, dtype=int)
        expected[visited[-1]] += 1
        expected[0] -= 1
        self.assertEqual(m.tolist(), expected.tolist())
        for (t, X, Y), l in zip(rep["steps"], rep["step_lengths"]):
            I = result["slices"][t]["MI"][X, Y]
            self.assertAlmostEqual(l, math.log1p(rls.I_MAX / I), places=10)
        self.assertEqual(rep["poset"].getCoverCount(), len(rep["covers"]))
        self.assertEqual(len(rep["depth"]), 6)
        with self.assertRaises(ValueError):
            rls.analyse(result, 1e-12, 1.0, rls.I_MAX, "walk", 0, dims=4)


if __name__ == "__main__":
    unittest.main()
