# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""S^3 dimensional-spike smoke test (#418).

A SPIKE smoke check that the existing, dimension-generic machinery stands up a
genuine triangulated ``S^3`` spatial slice and its ``S^3 x I`` extrusion -- the
home for the full ``E``/``B`` 3-vectors and the 4x4 Dirac-Kahler structure that
the 2+1 D ``S^2 x I`` proton sector cannot carry (see
``docs/design/s3_dimensional_spike.md``).

It is deliberately **isolated**: it builds the ``S^3`` complex purely from the
already-bound, dimension-generic ``Spacetime.fromVertexTuples`` / ``Spacetime.prismCells``
and reads the spectrum off ``HodgeLaplacian`` -- it adds **no** production C++ and
shares **no** state with the 2+1 D proton pipeline, so it cannot perturb the
golden proton results (``tests/cobordism/test_epic410_invariants.py`` is the
guard for that).

The minimal closed ``S^3`` is the boundary of the 4-simplex, ``dDelta^4`` (the
5-cell): 5 vertices, 10 edges, 10 triangles, 5 tetrahedra, Betti ``[1, 0, 0, 1]``.
``prismCells`` Freudenthal-extrudes its 5 tetrahedra one layer into a 4D
``S^3 x I`` (20 four-simplices) -- the first time the ``n = 4`` top-cell path is
exercised here.

Why ``prismCells`` and not ``Spacetime::symmetricStackCells`` for the stack:
``symmetricStackCells`` is 2D-only -- it cones *triangles* and skips any base
cell with ``t.size() != 3`` (``src/spacetime/Spacetime.cpp:437``), so on a
tetrahedral ``S^3`` base it returns an empty interior. ``prismCells`` is
dimension-generic (it Freudenthal-splits any m-vertex base cell), so it is the
existing tool that reaches ``n = 4`` without new production code. The symmetric
apex generalization to a tetrahedral base is the recommended *future* build,
scoped by the design note -- it is not needed for this smoke check.
"""

import itertools
import unittest

import numpy as np

import tessera

cob = tessera.cobordism

ZERO_TOL = 1e-9  # the |lambda| < tol cut for a near-zero (harmonic) Hodge mode


def _facet_connected_components(simplices):
    """Split same-dimensional simplices into facet-connected pieces (two simplices
    adjacent iff they share a codimension-one facet). Replaces the retired
    cob.Cobordism.connectedComponents with the identical union-find."""
    simplices = [tuple(sorted(s)) for s in simplices]
    n = len(simplices)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    owner = {}  # facet -> first simplex index owning it
    for i, s in enumerate(simplices):
        for drop in range(len(s)):
            facet = s[:drop] + s[drop + 1:]
            if facet in owner:
                parent[find(i)] = find(owner[facet])
            else:
                owner[facet] = i
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(simplices[i])
    return list(groups.values())

# S^3 Betti / harmonic vector (b_0 = b_3 = 1, b_1 = b_2 = 0). S^3 x I deformation
# retracts to S^3, so it carries the same homology (its b_4 = 0).
S3_BETTI = [1, 0, 0, 1]


def _boundary_4simplex_cells():
    """The 5 tetrahedra of ``dDelta^4`` (= ``S^3``): every 4-subset of {0..4}."""
    return [list(c) for c in itertools.combinations(range(5), 4)]


def _closed_s3():
    """The closed ``S^3`` = ``dDelta^4`` on the uniform ``l^2 = 1`` metric."""
    return tessera.Spacetime.fromVertexTuples(3, _boundary_4simplex_cells(), 1.0, 0.0)


def _s3_cross_interval():
    """``S^3 x I`` = one Freudenthal-extruded layer of ``dDelta^4`` (4D top cells)."""
    stacked = tessera.Spacetime.prismCells(_boundary_4simplex_cells(), 1)
    return tessera.Spacetime.fromVertexTuples(4, stacked, 1.0, 0.0)


def _kernel_dims(hl, top_k, metric=True, tol=ZERO_TOL):
    """Harmonic dims ``b_k`` read OFF the Hodge spectrum: ``#{ |lambda| < tol }``.

    Emergent (G8): the Betti vector is measured from the spectrum, never the
    topological closed form hard-coded into the operator.
    """
    return [
        int(np.sum(np.abs(np.array(hl.eigenvalues(k, metric))) < tol))
        for k in range(top_k + 1)
    ]


class TestS3Smoke(unittest.TestCase):
    """F1-F5 of the #418 spike faithfulness contract."""

    def test_f1_closed_s3_dimension(self):
        """F1: ``dDelta^4`` is a genuine 3-complex (top cells are tetrahedra)."""
        st = _closed_s3()
        self.assertEqual(cob.CombinatorialDimension().compute(st), 3.0)
        cc = cob.ChainComplex.fromSpacetime(st)
        # f-vector pins the triangulation: 5 verts, 10 edges, 10 tris, 5 tets.
        self.assertEqual(list(cc.fVector()), [5, 10, 10, 5])
        self.assertEqual(cc.eulerCharacteristic(), 0)  # chi(S^3) = 0

    def test_f2_s3_betti_from_hodge_spectrum(self):
        """F2: ``spectrum(k)`` runs k=0..3 and ``#{|lambda|<1e-9} == [1,0,0,1]``."""
        st = _closed_s3()
        hl = cob.HodgeLaplacian(st)
        # spectrum(k) must not raise for any k up to the top dimension.
        for k in range(4):
            self.assertEqual(len(hl.spectrum(k)), cob.ChainComplex
                             .fromSpacetime(st).numSimplices(k))
        # The harmonic dims equal the S^3 Betti vector exactly (volume + unit
        # weights both, mirroring the Hodge hardening suite's rigor).
        for metric in (True, False):
            self.assertEqual(_kernel_dims(hl, 3, metric), S3_BETTI,
                             msg=f"metric={metric}")
        # Independent combinatorial cross-check (rational Betti).
        self.assertEqual(list(cob.ChainComplex.fromSpacetime(st)
                              .bettiNumbers()), S3_BETTI)

    def test_f3_stacked_s3xi_dimension(self):
        """F3: ``S^3 x I`` is a genuine 4-complex -- the n=4 top path, no raise."""
        st = _s3_cross_interval()
        self.assertEqual(cob.CombinatorialDimension().compute(st), 4.0)
        # 5 tets x 4 Freudenthal four-simplices per layer = 20 top cells.
        self.assertEqual(cob.ChainComplex.fromSpacetime(st).numSimplices(4), 20)
        # The n=4 Hodge path runs for every degree without raising.
        hl = cob.HodgeLaplacian(st)
        for k in range(5):
            hl.spectrum(k)  # must not raise
        # S^3 x I retracts to S^3: same homology (b_4 = 0).
        self.assertEqual(_kernel_dims(hl, 4), S3_BETTI + [0])

    def test_f4_boundary_detection(self):
        """F4: closed S^3 has empty boundary; S^3 x I has dW = S^3 ⊔ S^3."""
        self.assertEqual(_closed_s3().getBoundary(), [])
        boundary = _s3_cross_interval().getBoundary()
        # dW = two S^3 copies, 5 tetrahedra each = 10 boundary tetrahedra.
        self.assertEqual(len(boundary), 10)
        self.assertEqual(len(_facet_connected_components(boundary)), 2)

    def test_f5_determinism(self):
        """F5 (G7): two in-process builds give bit-for-bit identical spectra."""
        h1 = cob.HodgeLaplacian(_closed_s3())
        h2 = cob.HodgeLaplacian(_closed_s3())
        for k in range(4):
            a = np.sort(np.abs(np.array(h1.eigenvalues(k))))
            b = np.sort(np.abs(np.array(h2.eigenvalues(k))))
            self.assertTrue(np.array_equal(a, b), msg=f"k={k} spectrum drifted")
        self.assertEqual(_kernel_dims(h1, 3), _kernel_dims(h2, 3))


if __name__ == "__main__":
    unittest.main()
