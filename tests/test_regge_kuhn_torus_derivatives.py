# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The exact Regge gradient and Hessian on the flat periodic Kuhn torus (#1209).

The Kuhn triangulation of a cubic lattice cuts every unit cube into six
tetrahedra along a monotone path of lattice steps. Their faces are right
triangles, and the circumcentre of a right triangle is the midpoint of its
hypotenuse; so on a hypotenuse hinge the circumcentres of the hinge and of the
facet coincide, and the circumcentre of a tetrahedron (the centre of its cube)
coincides with that of its faces through the body diagonal.

The circumcentric dual writes the height between two successive circumcentres as
``+-sqrt(R^2_coface - R^2_face)``. Differentiated through that root, the chain
rule divides zero by zero wherever the circumcentres coincide, so
``actionGradientExact`` and ``actionHessianExact`` were NaN or inf on this
torus, while ``dualReggeAction`` is smooth there. The heights are now
differentiated in their equal product form (``Simplex.dualVolumeGradient``), and
these tests hold the exact derivatives to finite differences of the action on
the torus of 4^3 vertices, the one the gravity study used
(``tessera-notes/experiments/gravity/spectral_stiffness.py``).

The second difference of the action along the transverse-traceless wave of
wavevector 2 pi / 4, normalised to a unit vector, converges to 0.64645; that is
the number the issue records and the exact Hessian must reproduce.
"""

import itertools
import math
import unittest

import numpy as np

import tessera

SIZE = 4


def kuhn_torus(n):
    """The flat periodic Kuhn triangulation of the unit cubic lattice of n^3 vertices.

    Returns the spacetime with every edge at its Euclidean length, the lattice
    step of each edge (in ``getEdgeList`` order), the edge midpoints, and the
    coordinates of every vertex.
    """
    index = lambda c: ((c[0] % n) * n + (c[1] % n)) * n + (c[2] % n)
    cells = []
    for corner in itertools.product(range(n), repeat=3):
        for order in itertools.permutations(range(3)):
            c = list(corner)
            path = [index(c)]
            for axis in order:
                c[axis] += 1
                path.append(index(c))
            cells.append(path)
    spacetime = tessera.Spacetime.fromVertexTuples(3, cells, 1.0, 0.0)
    coordinates = {index(c): np.array(c, dtype=float)
                   for c in itertools.product(range(n), repeat=3)}
    steps, midpoints = [], []
    for edge in spacetime.getEdgeList().toVector():
        a = coordinates[edge.getSource().getId()]
        b = coordinates[edge.getTarget().getId()]
        d = b - a
        d -= n * np.rint(d / n)
        edge.setLength(math.sqrt(float(d @ d)))
        steps.append(d)
        midpoints.append(a + 0.5 * d)
    spacetime.materializeFacets()
    return spacetime, np.array(steps), np.array(midpoints), coordinates


def unit_directions(n, spacetime, steps, midpoints, coordinates):
    """Squared-length fluctuations of unit norm: two metric waves and a gauge mode.

    A metric perturbation h(x) moves the squared length of the edge with step d
    at midpoint x by d^T h(x) d; a vertex displacement u moves it by
    2 d . (u(target) - u(source)) to first order.
    """
    q = 2.0 * np.pi / n
    wave = np.cos(q * midpoints[:, 0])
    out = {
        "transverse traceless": wave * (steps[:, 1] ** 2 - steps[:, 2] ** 2),
        "conformal": wave * np.sum(steps ** 2, axis=1),
    }
    displacement = {vid: np.array([0.0, np.sin(q * c[0]), 0.0])
                    for vid, c in coordinates.items()}
    gauge = []
    for edge, d in zip(spacetime.getEdgeList().toVector(), steps):
        gauge.append(2.0 * d @ (displacement[edge.getTarget().getId()]
                                - displacement[edge.getSource().getId()]))
    out["vertex displacement"] = np.array(gauge)
    return {name: v / np.linalg.norm(v) for name, v in out.items()}


class TestExactReggeDerivativesOnTheKuhnTorus(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        (cls.spacetime, cls.steps, cls.midpoints,
         cls.coordinates) = kuhn_torus(SIZE)
        cls.edges = cls.spacetime.getEdgeList().toVector()
        cls.squared = np.sum(cls.steps ** 2, axis=1)
        cls.solver = tessera.ReggeSolver(cls.spacetime, tessera.MatterConfiguration())
        cls.gradient = np.asarray(cls.solver.actionGradientExact(), dtype=complex)
        cls.hessian = np.array([[complex(z) for z in row]
                                for row in cls.solver.actionHessianExact()])
        cls.directions = unit_directions(SIZE, cls.spacetime, cls.steps,
                                         cls.midpoints, cls.coordinates)

    def _set(self, squared):
        for edge, value in zip(self.edges, squared):
            edge.setLength(math.sqrt(value))

    def _action(self, t, v):
        """The dual Regge action at squared lengths s + t v (then back to s)."""
        self._set(self.squared + t * v)
        value = complex(self.solver.dualReggeAction())
        self._set(self.squared)
        return value

    def _gradient_at(self, t, v):
        self._set(self.squared + t * v)
        value = np.asarray(self.solver.actionGradientExact(), dtype=complex)
        self._set(self.squared)
        return value

    def test_the_torus_has_coincident_circumcentres(self):
        """Guards the rest of the file from passing on a torus without the defect."""
        hinges = [s for s in self.spacetime.getSimplices()
                  if len(s.getVertices()) == 2 and s.hasTopCoface()]
        self.assertEqual(len(hinges), len(self.edges))
        coincident = [h for h in hinges if h.dualGeometryIsDegenerate()]
        # Every face diagonal and every body diagonal is a hypotenuse.
        self.assertGreater(len(coincident), len(hinges) // 2)

    def test_the_exact_gradient_and_hessian_are_finite(self):
        self.assertTrue(np.isfinite(self.gradient).all())
        self.assertTrue(np.isfinite(self.hessian).all())

    def test_the_hessian_is_symmetric(self):
        scale = float(np.max(np.abs(self.hessian)))
        self.assertGreater(scale, 0.0)
        self.assertLess(float(np.max(np.abs(self.hessian - self.hessian.T))),
                        1e-12 * scale)

    def test_the_sparse_hessian_equals_the_dense_one(self):
        rows, cols, values, n = self.solver.actionHessianExactSparse()
        sparse = np.zeros((n, n), dtype=complex)
        np.add.at(sparse, (np.array(rows), np.array(cols)),
                  np.array(values, dtype=complex))
        self.assertLess(float(np.max(np.abs(sparse - self.hessian))), 1e-12)

    def test_the_gradient_matches_first_differences_of_the_action(self):
        """dS/dl^2_e against a Richardson-combined central difference.

        Three edges of each of the seven lattice-step classes: the three axes,
        the three face diagonals (hypotenuses) and the body diagonal.
        """
        classes = {}
        for i, d in enumerate(np.abs(self.steps)):
            classes.setdefault(tuple(d), []).append(i)
        self.assertEqual(len(classes), 7)
        t = 1e-3
        worst = 0.0
        for members in classes.values():
            for i in members[:3]:
                unit = np.zeros(len(self.edges))
                unit[i] = 1.0
                central = lambda h: (self._action(h, unit) - self._action(-h, unit)) / (2 * h)
                difference = (4.0 * central(t / 2) - central(t)) / 3.0
                worst = max(worst, abs(difference - self.gradient[i]))
        self.assertLess(worst, 1e-9)

    def test_the_hessian_matches_second_differences_of_the_action(self):
        """v . H v against the second difference of the action along v.

        The issue's reference is the transverse-traceless wave, whose second
        difference converges to 0.64645.
        """
        t = 2e-3
        for name, v in self.directions.items():
            with self.subTest(direction=name):
                second = lambda h: (self._action(h, v) - 2.0 * self._action(0.0, v)
                                    + self._action(-h, v)) / h ** 2
                difference = (4.0 * second(t / 2) - second(t)) / 3.0
                exact = v @ self.hessian @ v
                self.assertLess(abs(exact - difference), 1e-6)
        tt = self.directions["transverse traceless"]
        self.assertAlmostEqual((tt @ self.hessian @ tt).real, 0.64645, delta=1e-5)

    def test_the_hessian_matches_first_differences_of_the_gradient(self):
        """H v against the central difference of the exact gradient along v,
        every component, so the whole Hessian column space is checked and not
        only its quadratic form."""
        t = 1e-3
        for name, v in self.directions.items():
            with self.subTest(direction=name):
                central = lambda h: (self._gradient_at(h, v) - self._gradient_at(-h, v)) / (2 * h)
                difference = (4.0 * central(t / 2) - central(t)) / 3.0
                self.assertTrue(np.isfinite(difference).all())
                self.assertLess(float(np.max(np.abs(self.hessian @ v - difference))), 1e-8)


if __name__ == "__main__":
    unittest.main()
