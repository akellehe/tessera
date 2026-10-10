# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""Chain complex + exact integer/GF(2) linear algebra (#64).

Validates simplicial homology of the available fixtures (Betti numbers over ℚ
and GF(2), torsion, ∂²=0, χ) and the linear-algebra primitives (Smith Normal
Form, GF(2) rank, symmetric inertia) directly.
"""

import unittest

import tessera

cob = tessera.cobordism


def _build(topology):
    sig = tessera.Signature(topology.dimension(), tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    st = tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0,
                           tessera.PREFERRED, topology)
    st.build()
    return st


def _cc(topology):
    return cob.ChainComplex.from_spacetime(_build(topology))


class TestHomology(unittest.TestCase):

    def test_sphere_homology_sweep(self):
        # H_*(S^n) = Z in degrees 0 and n, else 0; torsion-free.
        for n in range(1, 6):
            with self.subTest(n=n):
                cc = _cc(tessera.SimplexBoundarySphere(n))
                expected = [1] + [0] * (n - 1) + [1]
                self.assertEqual(cc.betti_numbers(), expected)
                self.assertEqual(cc.betti_numbers_gf2(), expected)
                self.assertEqual(cc.euler_characteristic(), 1 + (-1) ** n)
                for k in range(n + 1):
                    self.assertEqual(list(cc.torsion(k)), [])

    def test_ball_homology_sweep(self):
        # Δ^n is contractible: H_0 = Z, else 0.
        for n in range(1, 6):
            with self.subTest(n=n):
                cc = _cc(tessera.SolidSimplex(n))
                self.assertEqual(cc.betti_numbers(), [1] + [0] * n)
                self.assertEqual(cc.euler_characteristic(), 1)

    def test_rp2_homology(self):
        # ℝP²: H_0=Z, H_1=Z/2 (torsion!), H_2=0 over Z; (1,1,1) over GF(2).
        cc = _cc(tessera.RealProjectivePlane())
        self.assertEqual(cc.betti_numbers(), [1, 0, 0])
        self.assertEqual(cc.betti_numbers_gf2(), [1, 1, 1])
        self.assertEqual(list(cc.torsion(1)), [2])
        self.assertEqual(cc.euler_characteristic(), 1)

    def test_boundary_composes_to_zero_all_fixtures(self):
        for topology in (tessera.SimplexBoundarySphere(1),
                         tessera.SimplexBoundarySphere(4),
                         tessera.SolidSimplex(3),
                         tessera.RealProjectivePlane()):
            with self.subTest(topology=type(topology).__name__):
                self.assertTrue(_cc(topology).boundary_composes_to_zero())

    def test_euler_matches_betti_alternating_sum(self):
        # χ = Σ (−1)^k b_k  (free ranks; torsion doesn't affect χ).
        for topology in (tessera.SimplexBoundarySphere(2),
                         tessera.SimplexBoundarySphere(4),
                         tessera.SolidSimplex(4),
                         tessera.RealProjectivePlane()):
            with self.subTest(topology=type(topology).__name__):
                cc = _cc(topology)
                betti = cc.betti_numbers()
                chi = sum((-1) ** k * b for k, b in enumerate(betti))
                self.assertEqual(chi, cc.euler_characteristic())

    def test_boundary_matrix_shape_and_entries(self):
        cc = _cc(tessera.SimplexBoundarySphere(2))  # S^2: f=(4,6,4)
        d1 = cc.boundary_matrix(1)  # 4x6
        self.assertEqual(len(d1), 4 * 6)
        self.assertTrue(all(v in (-1, 0, 1) for v in d1))
        # each edge column has exactly one +1 and one -1
        for j in range(6):
            col = [d1[i * 6 + j] for i in range(4)]
            self.assertEqual(sorted(col), [-1, 0, 0, 1])


class TestIntegerLinalg(unittest.TestCase):

    def test_snf_diagonal(self):
        # SNF(diag(2,3)) = diag(1,6): gcd then lcm.
        snf = cob.smith_normal_form([2, 0, 0, 0, 3, 0, 0, 0, 0], 3, 3)
        self.assertEqual(snf.rank, 2)
        self.assertEqual(list(snf.invariant_factors), [1, 6])

    def test_snf_rank_and_factors(self):
        snf = cob.smith_normal_form([1, 0, 0, 0], 2, 2)  # diag(1,0)
        self.assertEqual(snf.rank, 1)
        self.assertEqual(list(snf.invariant_factors), [1])
        self.assertEqual(cob.integer_rank([2, 4, 1, 2], 2, 2), 1)  # rank-1

    def test_gf2_rank(self):
        self.assertEqual(cob.gf2_rank([1, 1, 1, 1], 2, 2), 1)         # rows equal mod 2
        self.assertEqual(cob.gf2_rank([1, 0, 0, 1], 2, 2), 2)         # identity
        self.assertEqual(cob.gf2_rank([1, 1, 0, 1, 1, 0], 2, 3), 1)   # equal rows

    def test_inertia(self):
        self.assertEqual(cob.symmetric_inertia([1], 1).signature(), 1)
        self.assertEqual(cob.symmetric_inertia([-1], 1).signature(), -1)
        # CP²-like form [+1] -> signature +1; CP²-bar [-1] -> -1.
        hyper = cob.symmetric_inertia([0, 1, 1, 0], 2)   # S²×S² form
        self.assertEqual((hyper.n_pos, hyper.n_neg, hyper.signature()), (1, 1, 0))
        diag = cob.symmetric_inertia([1, 0, 0, 0, -1, 0, 0, 0, 1], 3)
        self.assertEqual((diag.n_pos, diag.n_neg, diag.n_zero), (2, 1, 0))
        # degenerate: a zero eigenvalue is counted as such
        self.assertEqual(cob.symmetric_inertia([1, 0, 0, 0], 2).n_zero, 1)


if __name__ == "__main__":
    unittest.main()
