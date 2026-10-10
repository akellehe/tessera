# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The root-lattice-spacetime driver (`tessera.drivers.root_lattice_spacetime`).

Covers the instrument:
* the Coxeter-plane weights are the vertices of the regular n-gon, sum to
  zero, and the roots are their differences; for four qubits the twelve
  roots project onto the eight vectors (+-1, +-1), (+-2, 0), (0, +-2) of the
  square lattice, opposite chords coinciding;
* the lattice points within two root steps are the known shells of the
  hexagonal lattice for three qubits, and every point has zero sum;
* the causal order is the transitive closure of "later and sharing a qubit",
  its covers are the Hasse diagram, and the tessera Poset holds them;
* the walk rides exactly the events of its current qubit and its steps add
  up to e_end - e_start; the geodesic is the shortest path;
* the lengths are a ln(1 + I_0 / I), unnormalised;
* the command line runs end to end, prints what it promises and writes the
  figure.
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

    def test_the_weights_are_the_regular_polygon_and_sum_to_zero(self):
        for n in (3, 4, 6, 7):
            w = rls.coxeter_weights(n)
            self.assertEqual(w.shape, (n, 2))
            np.testing.assert_allclose(np.linalg.norm(w, axis=1), 1.0)
            np.testing.assert_allclose(w.sum(axis=0), 0.0, atol=1e-12)
            angles = np.arctan2(w[:, 1], w[:, 0]) % (2 * math.pi)
            np.testing.assert_allclose(angles, 2 * math.pi * np.arange(n) / n, atol=1e-12)

    def test_the_roots_are_the_differences_of_weights(self):
        n = 5
        rs = rls.roots(n)
        self.assertEqual(len(rs), n * (n - 1))
        w = rls.coxeter_weights(n)
        for (X, Y), m in rs:
            self.assertEqual(m.sum(), 0)
            self.assertEqual(m[Y], 1)
            self.assertEqual(m[X], -1)
            np.testing.assert_allclose(rls.project(m, w), w[Y] - w[X], atol=1e-12)

    def test_four_qubits_project_onto_the_square_lattice(self):
        w = rls.coxeter_weights(4)
        projected = {tuple(np.round(rls.project(m, w), 9)) for _, m in rls.roots(4)}
        expected = {(1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0),
                    (2.0, 0.0), (-2.0, 0.0), (0.0, 2.0), (0.0, -2.0)}
        self.assertEqual({(float(a), float(b)) for a, b in projected}, expected)
        self.assertEqual(len(rls.roots(4)), 12)

    def test_the_lattice_within_two_steps_is_the_hexagonal_shells(self):
        pts = rls.lattice_points(3, steps=2)
        self.assertEqual(len(pts), 19)                 # origin, 6 roots, 12 second shell
        self.assertTrue((pts.sum(axis=1) == 0).all())
        self.assertEqual(tuple(pts[0]), (0, 0, 0))
        radii = np.round(np.linalg.norm(rls.project(pts, rls.coxeter_weights(3)), axis=1), 6)
        self.assertEqual(sorted(set(radii.tolist())), [0.0, round(math.sqrt(3), 6), 3.0,
                                                       round(2 * math.sqrt(3), 6)])
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

    def test_the_poset_holds_the_covers(self):
        from tessera import quantum
        ps, P, covers = rls.poset(EVENTS)
        self.assertIsInstance(ps, quantum.Poset)
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

    def test_the_walk_runs_and_writes_the_figure(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "walk.png")
            rc, out = self._run(["--qubits", "4", "--timesteps", "5", "--save", path,
                                 "--no-show"])
            self.assertEqual(rc, 0)
            self.assertTrue(os.path.exists(path) and os.path.getsize(path) > 0)
        self.assertIn("INFORMATION NETWORK", out)
        self.assertIn("ROOT LATTICE A_3: 12 roots", out)
        self.assertIn("CAUSAL ORDER: 5 events", out)
        self.assertIn("WORLDLINE of a unit of charge from A", out)
        self.assertIn("displacement:", out)

    def test_the_geodesic_runs(self):
        rc, out = self._run(["--qubits", "4", "--timesteps", "6", "--path", "geodesic",
                             "--start", "B", "--end", "D", "--no-show"])
        self.assertEqual(rc, 0)
        self.assertIn("GEODESIC from B to D", out)

    def test_the_analysis_is_consistent_with_the_slices(self):
        from tessera.drivers import entanglement_complex as ec
        result = ec.simulate(4, None, "1/2", 0, "global", 6, "all", False)
        rep = rls.analyse(result, 1e-12, 1.0, rls.I_MAX, "walk", 0)
        self.assertEqual(len(rep["events"]), 6)
        self.assertEqual(rep["T"], 6)
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


if __name__ == "__main__":
    unittest.main()
