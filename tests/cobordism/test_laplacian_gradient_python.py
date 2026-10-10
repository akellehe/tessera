# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Exact analytic gradient of the metric Hodge Laplacian (#461, degree zero #805).

`HodgeLaplacian.laplacian_gradient(k, a, b)` is `∂L_k/∂ℓ²_e` at arbitrary degree
— the keystone (with `Simplex.volume_gradient`) for the arbitrary-k `r_U` analytic
gradient. The rigorous check is the **exact Euler homogeneity identity**: under
the `V²` weights `L_k` is homogeneous of degree `−1` in `ℓ²` at every degree, so

    Σ_e ℓ²_e · ∂L_k/∂ℓ²_e  =  −L_k     (exactly).

Degree zero is included: `L_0 = d_1 W_1^-1 d_1^†` is the same construction with
the lower term absent, so the identity is the same one — checked here on a
complex carrying a spacelike, a timelike (`ℓ² < 0`), and a complex `ℓ²` edge at
once, where the old magnitude-convention degree-zero operator had no gradient at
all.

This is independent of finite difference (which is roundoff-limited and does not
converge — the optimizer uses the analytic gradient, never FD).
"""
import math
import unittest

import numpy as np

import tessera as T
import cmath

cob = T.cobordism


def _holed_s3(n_refine=12):
    # A refined S^3 (Betti [1,0,0,1]) opened by a surgical cone-out so it carries a
    # nonempty b_2 register; the Euler-homogeneity identity below is exact on any
    # such metric 3-complex (it does not depend on the register's topology).
    sig = T.Signature(3, T.Lorentzian)
    st = T.Spacetime(T.Metric(True, sig), T.CDT, 1.0, 1.0, T.PREFERRED,
                     T.SimplexBoundarySphere(3))
    st.build()
    for e in st.get_edge_list().to_vector():
        e.set_length(cmath.sqrt(complex(1.0)))
    for seed in range(n_refine):
        mv = T.AddMove(st, seed, False, T.PachnerMode.PreGeometric, False)
        if mv.propose():
            mv.apply()
    tops = [tuple(sorted(v.get_id() for v in c.get_vertices()))
            for c in st.get_top_simplices()]
    sc = cob.SurgicalCone(st)
    for t in tops:                      # open one hole (b_3 -> 0, exposes a 2-cycle)
        if sc.cone_out(list(t))[0]:
            break
    for i, e in enumerate(st.get_edge_list().to_vector()):
        e.set_length(cmath.sqrt(complex(1.0 + 0.013 * (i % 6))))
    return st


def _mixed_signature_graph():
    """A 1-complex carrying all three edge characters at once: spacelike
    (l^2 > 0), timelike (l^2 < 0), and a genuinely complex l^2. No degree-zero
    fixture in the tree carried a negative squared length before #805."""
    sig = T.Signature(4, T.Lorentzian)
    st = T.Spacetime(T.Metric(True, sig), T.HERMITIAN_WEIGHTED, 1.0, 1.0,
                     T.PREFERRED, T.Toroid())
    squared = {(0, 1): complex(1.3), (1, 2): complex(-0.7),
               (0, 2): complex(0.9, 0.4), (2, 3): complex(2.1)}
    verts = {i: st.create_vertex(i) for i in range(4)}
    for (a, b) in squared:
        st.create_simplex([verts[a], verts[b]])
    for e in st.get_edge_list().to_vector():
        key = tuple(sorted((e.get_source().get_id(), e.get_target().get_id())))
        e.set_length(cmath.sqrt(squared[key]))
    return st


def _Lmat(hl, k):
    # The signed operator is real on real signed l^2; assert Im = 0 rather than
    # project it away, then hand back the real part.
    L = np.asarray(hl.laplacian(k, True), complex)
    n = int(round(math.sqrt(L.size)))
    L = L.reshape(n, n)
    np.testing.assert_allclose(L.imag, 0.0, atol=1e-12)
    return L.real


class LaplacianGradientHandCalcTest(unittest.TestCase):
    def test_euler_homogeneity_identity(self):
        # Σ_e ℓ²_e ∂L_k/∂ℓ²_e = −L_k exactly: under the V^2 weights every W_k
        # is homogeneous of degree +1 in l^2 (edge W_1 = l^2; triangle
        # W_2 = detG/4, quadratic overall but degree +1 per the Gram scaling —
        # detG(s l^2) = s^2 detG makes W_2 degree 2... the OPERATOR
        # L = W_k^-1 d^T W_{k-1} d + d W_{k+1}^-1 d^T W_k is degree -1 at every
        # k, measured exactly on this fixture). The old -1/2 belonged to the
        # removed sqrt(W)-conjugated form.
        st = _holed_s3()
        hl = cob.HodgeLaplacian(st)
        for k in (0, 1, 2):
            L = _Lmat(hl, k)
            n = L.shape[0]
            self.assertGreater(n, 0)
            acc = np.zeros((n, n))
            for e in st.get_edge_list().to_vector():
                a, b = e.get_source().get_id(), e.get_target().get_id()
                g = np.asarray(hl.laplacian_gradient(k, a, b), complex).reshape(n, n)
                np.testing.assert_allclose(g.imag, 0.0, atol=1e-12)
                acc += (e.get_length() * e.get_length()).real * g.real
            self.assertLess(np.max(np.abs(acc + L)), 1e-10,
                            f"Euler identity Σℓ²∂L = −L failed at k={k}")

    def test_euler_identity_holds_at_degree_zero_on_mixed_signature(self):
        # The same exact identity for the derived L_0 = d_1 W_1^-1 d_1^dagger
        # (#805), on a genuinely MIXED complex: one spacelike edge, one timelike
        # (l^2 < 0), one with a complex l^2. L_0 is degree -1 in l^2 like every
        # other degree, so Σ_e z_e ∂L_0/∂z_e = −L_0, in complex arithmetic.
        st = _mixed_signature_graph()
        hl = cob.HodgeLaplacian(st)
        cc = cob.ChainComplex.from_spacetime(st)
        n = cc.num_simplices(0)
        L = np.asarray(hl.laplacian(0, True), complex).reshape(n, n)
        by_pair = {tuple(sorted((e.get_source().get_id(), e.get_target().get_id()))): e
                   for e in st.get_edge_list().to_vector()}
        acc = np.zeros((n, n), dtype=complex)
        for cell in cc.k_simplex_vertices(1):
            g = np.asarray(hl.laplacian_gradient(0, cell[0], cell[1]),
                           complex).reshape(n, n)
            edge = by_pair[tuple(sorted(cell))]
            z = complex(edge.get_length()) ** 2
            acc += z * g
        self.assertLess(np.max(np.abs(acc + L)), 1e-13,
                        "Euler identity Σz ∂L_0 = −L_0 failed")
        # ...and the row sums vanish identically at this mixed signature.
        self.assertLess(np.max(np.abs(L @ np.ones(n, dtype=complex))), 1e-14)

    def test_matches_finite_difference_and_empty_below_degree_zero(self):
        # The signed operator is non-symmetric, so its gradient is too; the
        # honest check is a central finite difference of L itself.
        st = _holed_s3()
        hl = cob.HodgeLaplacian(st)
        n = _Lmat(hl, 2).shape[0]
        e = st.get_edge_list().to_vector()[0]
        g = np.asarray(hl.laplacian_gradient(2, e.get_source().get_id(),
                                            e.get_target().get_id()), complex).reshape(n, n)
        h = 1e-6
        l = e.get_length()
        e.set_length(cmath.sqrt(l * l + h)); st.materialize_facets()
        Lp = np.asarray(cob.HodgeLaplacian(st).laplacian(2, True), complex).reshape(n, n)
        e.set_length(cmath.sqrt(l * l - h)); st.materialize_facets()
        Lm = np.asarray(cob.HodgeLaplacian(st).laplacian(2, True), complex).reshape(n, n)
        e.set_length(l); st.materialize_facets()
        fd = (Lp - Lm) / (2.0 * h)
        self.assertLess(np.max(np.abs(g - fd)), 1e-5)
        # Degree zero is now the derived L_0 = d_1 W_1^-1 d_1^dagger, which has
        # an exact gradient too (#805): the same central-difference check.
        n0 = _Lmat(hl, 0).shape[0]
        g0 = np.asarray(hl.laplacian_gradient(0, e.get_source().get_id(),
                                             e.get_target().get_id()),
                        complex).reshape(n0, n0)
        e.set_length(cmath.sqrt(l * l + h)); st.materialize_facets()
        L0p = np.asarray(cob.HodgeLaplacian(st).laplacian(0, True),
                         complex).reshape(n0, n0)
        e.set_length(cmath.sqrt(l * l - h)); st.materialize_facets()
        L0m = np.asarray(cob.HodgeLaplacian(st).laplacian(0, True),
                         complex).reshape(n0, n0)
        e.set_length(l); st.materialize_facets()
        self.assertLess(np.max(np.abs(g0 - (L0p - L0m) / (2.0 * h))), 1e-5)
        self.assertGreater(np.max(np.abs(g0)), 0.0)
        # k < 0 still has no chain to differentiate.
        self.assertEqual(hl.laplacian_gradient(-1, e.get_source().get_id(),
                                              e.get_target().get_id()), [])


if __name__ == "__main__":
    unittest.main()
