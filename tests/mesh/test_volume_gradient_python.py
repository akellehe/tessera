# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Exact analytic gradient of a simplex's signed `volume()` w.r.t. edge ℓ² (#461).

`Simplex.volumeGradient` is the per-degree **Hodge inner-product weight** gradient
(the weights `W_k` are signed simplex volumes) — the keystone for an arbitrary-degree
analytic `∂L_k/∂ℓ²`, hence the general-k `r_U` gradient. By Jacobi's formula on the
Gram determinant, `∂V/∂ℓ²_e = (V/2)·tr(G⁻¹ ∂_e G)`, reusing the same
`gramMatrix`/`cofactorMatrix` machinery as the circumcentric `dualVolumeGradient`
(#354). Here it must match a central finite difference of `volume()` to ~machine
precision across every degree (triangle, tetrahedron, pentatope).
"""
import math
import unittest

import tessera as T
import cmath


def _top_of_dim(st, nverts):
    return next(s for s in st.getSimplices()
               if len([v for v in s.getVertices()]) == nverts)


def _l2(st):
    return {(min(e.getSource().getId(), e.getTarget().getId()),
             max(e.getSource().getId(), e.getTarget().getId())):
            (e.getLength() * e.getLength()).real for e in st.getEdgeList().toVector()}


def _jittered_s4():
    sig = T.Signature(4, T.Lorentzian)
    st = T.Spacetime(T.Metric(True, sig), T.CDT, 1.0, 1.0, T.PREFERRED,
                     T.SimplexBoundarySphere(4))
    st.build()
    for i, e in enumerate(st.getEdgeList().toVector()):
        e.setLength(cmath.sqrt(complex(1.0 + 0.017 * (i % 5))))
    # materialize sub-simplices down to triangles
    for s in list(st.getTopSimplices()):
        for f in s.getFacets():
            for g in f.getFacets():
                g.getFacets()
    return st


def _key(a, b):
    return (min(a, b), max(a, b))


class VolumeGradientHandCalcTest(unittest.TestCase):
    """Closed-form values and the exact Euler homogeneity identity — the rigorous
    checks (finite difference is only a roundoff-limited cross-check; the optimizer
    uses the analytic gradient, never FD)."""

    def test_equilateral_triangle_closed_form(self):
        # All ℓ²=1: A = √3/4, and ∂A/∂ℓ²_e = 1/(4√3) for every edge (by symmetry).
        st = T.Spacetime.fromVertexTuples(2, [[0, 1, 2]], 1.0, 0.0)
        tri = _top_of_dim(st, 3)
        self.assertAlmostEqual(tri.volume(), math.sqrt(3) / 4, places=12)
        for _e, dA in tri.volumeGradient().items():
            self.assertAlmostEqual(dA, 1.0 / (4 * math.sqrt(3)), places=12)

    def test_regular_tetrahedron_closed_form(self):
        # All ℓ²=1: V = 1/(6√2), and ∂V/∂ℓ²_e = 1/(24√2) for every edge.
        st = T.Spacetime.fromVertexTuples(3, [[0, 1, 2, 3]], 1.0, 0.0)
        tet = _top_of_dim(st, 4)
        self.assertAlmostEqual(tet.volume(), 1.0 / (6 * math.sqrt(2)), places=12)
        for _e, dV in tet.volumeGradient().items():
            self.assertAlmostEqual(dV, 1.0 / (24 * math.sqrt(2)), places=12)

    def test_euler_homogeneity_identity(self):
        # A j-simplex volume is homogeneous of degree j/2 in ℓ² ⇒
        # Σ_e ℓ²_e ∂V/∂ℓ²_e = (j/2)·V exactly (independent of finite difference).
        for nverts, cell in [(3, [0, 1, 2]), (4, [0, 1, 2, 3])]:
            st = T.Spacetime.fromVertexTuples(nverts - 1, [cell], 1.0, 0.0)
            # jitter so the identity is non-trivial (not just the symmetric point)
            for i, e in enumerate(st.getEdgeList().toVector()):
                e.setLength(cmath.sqrt(complex(1.0 + 0.07 * (i % 4))))
            s = _top_of_dim(st, nverts)
            l2 = _l2(st)
            j = nverts - 1
            euler = sum(l2[e] * dV for e, dV in s.volumeGradient().items())
            self.assertAlmostEqual(euler, (j / 2.0) * s.volume(), places=12)


class VolumeGradientTest(unittest.TestCase):
    def test_matches_finite_difference_every_degree(self):
        st = _jittered_s4()
        em = {_key(e.getSource().getId(), e.getTarget().getId()): e
              for e in st.getEdgeList().toVector()}
        by_size = {}
        for s in st.getSimplices():
            n = len([v for v in s.getVertices()])
            by_size.setdefault(n, []).append(s)
        # 3 = triangle (W_2), 4 = tetrahedron (W_3), 5 = pentatope (top)
        self.assertTrue({3, 4, 5}.issubset(by_size.keys()))

        h = 1e-6
        for size in (3, 4, 5):
            worst, tested = 0.0, 0
            for s in by_size[size]:
                for (a, b), ga in s.volumeGradient().items():
                    e = em.get((a, b))
                    if e is None:
                        continue
                    o = (e.getLength() * e.getLength())
                    e.setLength(cmath.sqrt(complex(o + h)))
                    vp = s.volume()
                    e.setLength(cmath.sqrt(complex(o - h)))
                    vm = s.volume()
                    e.setLength(cmath.sqrt(complex(o)))
                    worst = max(worst, abs(ga - (vp - vm) / (2 * h)))
                    tested += 1
            self.assertGreater(tested, 0, f"no edge-derivatives tested at size {size}")
            self.assertLess(worst, 1e-8, f"volumeGradient inexact at size {size}")


if __name__ == "__main__":
    unittest.main()
