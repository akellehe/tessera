# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The dual-volume derivatives where two circumcentres coincide (#1209).

The circumcentric dual of a hinge is built from the heights between successive
circumcentres (hinge to facet, facet to top cell), each written as
``+-sqrt(R^2_coface - R^2_face)``. The radicand factors exactly as
``lambda_v^2 det G_coface / det G_face``, with ``lambda_v`` the barycentric
coordinate of the coface's circumcentre at its vertex outside the face, so the
height is ``lambda_v sqrt(det G_coface / det G_face)``: smooth where the two
circumcentres coincide (``lambda_v = 0``), although the root of the difference
has a zero-over-zero chain rule there.

The Kuhn triangulation of a cubic lattice puts coincident circumcentres on
every face diagonal and body diagonal, in Euclidean signature and equally with
one lattice axis timelike (the metric stays diagonal, so the right angles
survive). On both, ``Simplex.dualVolumeGradient`` and
``Simplex.dualVolumeHessian`` must be finite and must match central differences
of ``Simplex.dualVolume`` and of the gradient itself.
"""

import cmath
import itertools
import unittest

import numpy as np

import tessera

SIZE = 4
STEP = 1e-3


def kuhn_torus(n, timelike_weight=None):
    """The periodic Kuhn torus of n^3 vertices, lengths set by the metric
    diag(1, 1, 1), or diag(1, 1, -timelike_weight) when that is given."""
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
    edges = {}
    for edge in spacetime.getEdgeList().toVector():
        a = edge.getSource().getId()
        b = edge.getTarget().getId()
        d = coordinates[b] - coordinates[a]
        d -= n * np.rint(d / n)
        squared = d[0] ** 2 + d[1] ** 2
        squared += d[2] ** 2 if timelike_weight is None else -timelike_weight * d[2] ** 2
        edges[(min(a, b), max(a, b))] = [edge, complex(squared)]
        edge.setLength(cmath.sqrt(squared))
    spacetime.materializeFacets()
    return spacetime, edges


class _CoincidentCircumcentres:
    """Shared checks; a subclass supplies the torus."""

    timelike_weight = None

    @classmethod
    def setUpClass(cls):
        cls.spacetime, cls.edges = kuhn_torus(SIZE, cls.timelike_weight)
        hinges = [s for s in cls.spacetime.getSimplices()
                  if len(s.getVertices()) == 2 and s.hasTopCoface()]
        cls.coincident = [h for h in hinges if h.dualGeometryIsDegenerate()]
        # One hinge of each kind of coincidence: the lattice step of the hinge.
        chosen = {}
        for hinge in cls.coincident:
            a, b = (v.getId() for v in hinge.getVertices())
            chosen.setdefault(cls._kind(a, b), hinge)
        cls.chosen = list(chosen.values())

    @classmethod
    def _kind(cls, a, b):
        edge, squared = cls.edges[(min(a, b), max(a, b))]
        return round(squared.real, 6)

    def _shift(self, key, t):
        edge, squared = self.edges[key]
        edge.setLength(cmath.sqrt(squared + t))

    def _central(self, read, key, t=STEP):
        """Richardson-combined central difference of read() in l^2 of edge key."""
        def at(h):
            self._shift(key, h)
            plus = read()
            self._shift(key, -h)
            minus = read()
            self._shift(key, 0.0)
            return (plus - minus) / (2.0 * h)
        return (4.0 * at(t / 2) - at(t)) / 3.0

    def test_the_torus_has_coincident_circumcentres(self):
        self.assertGreater(len(self.coincident), 0)
        self.assertGreaterEqual(len(self.chosen), 2)

    def test_the_dual_volume_gradient_is_finite_and_exact(self):
        for hinge in self.chosen:
            gradient = hinge.dualVolumeGradient()
            self.assertGreater(len(gradient), 0)
            values = np.array(list(gradient.values()), dtype=complex)
            self.assertTrue(np.isfinite(values).all())
            for key, value in gradient.items():
                difference = self._central(lambda: complex(hinge.dualVolume()), key)
                self.assertLess(abs(difference - value), 1e-8, msg=f"edge {key}")

    def test_the_dual_volume_hessian_is_finite_symmetric_and_exact(self):
        for hinge in self.chosen:
            hessian = hinge.dualVolumeHessian()
            values = np.array(list(hessian.values()), dtype=complex)
            self.assertTrue(np.isfinite(values).all())
            keys = sorted(hinge.dualVolumeGradient())
            for e in keys:
                for f in keys:
                    self.assertAlmostEqual(hessian[(e, f)], hessian[(f, e)], delta=1e-12)
            for f in keys:
                column = self._central(
                    lambda: np.array([hinge.dualVolumeGradient()[e] for e in keys],
                                     dtype=complex), f)
                exact = np.array([hessian[(e, f)] for e in keys], dtype=complex)
                self.assertLess(float(np.max(np.abs(column - exact))), 1e-8,
                                msg=f"column {f}")


class TestEuclideanKuhnTorus(_CoincidentCircumcentres, unittest.TestCase):
    timelike_weight = None


class TestLorentzianKuhnTorus(_CoincidentCircumcentres, unittest.TestCase):
    """One axis timelike, squared lengths dx^2 + dy^2 - dz^2 / 2: no null edge,
    and the dual heights through a timelike normal are imaginary."""
    timelike_weight = 0.5

    def test_some_heights_are_imaginary(self):
        dual = np.array([complex(h.dualVolume()) for h in self.chosen])
        self.assertGreater(float(np.max(np.abs(dual.imag))), 1e-3)


if __name__ == "__main__":
    unittest.main()
