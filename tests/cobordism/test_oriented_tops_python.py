# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""Oriented top-simplices and the fundamental class [W] = ε_t (#107).

The Stage-2 Dijkgraaf–Witten weight ω(·)^{ε_t} needs each top simplex's
orientation sign relative to a coherent orientation of the manifold W — the
fundamental class [W] ∈ H_d. ``ChainComplex.fundamental_class()`` returns those
signs ε_t = ±1 (the generator of ker ∂_d), and
``ChainComplex.oriented_top_simplices()`` returns the top simplices in the column
order the signs refer to.

We check on closed oriented manifolds (T³, T², S²) that the signed top chain is
a cycle — applying ∂_d gives 0, i.e. every codimension-one face's two incident
top simplices cancel — that the signs are ±1 and deterministically normalized,
and that oriented_top_simplices() lines up with the boundary-matrix columns and
the manifold's actual top simplices.
"""

import unittest

import tessera

cobordism = tessera.cobordism


def _circle():
    return tessera.SimplexBoundarySphere(1)  # S^1


def _two_sphere():
    return tessera.SimplexBoundarySphere(2)  # S^2


def _torus():
    return tessera.SimplicialProduct(_circle(), _circle())  # T^2 = S^1 x S^1


def _three_torus():
    return tessera.SimplicialProduct(_torus(), _circle())  # T^3 = T^2 x S^1


def _build(topology):
    signature = tessera.Signature(topology.dimension(), tessera.Lorentzian)
    metric = tessera.Metric(True, signature)
    spacetime = tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0,
                                  tessera.PREFERRED, topology)
    spacetime.build()
    return spacetime


def _top_tuples(spacetime):
    """The top simplices of a Spacetime as sorted vertex-id tuples."""
    tuples = [tuple(sorted(v.get_id() for v in s.get_vertices()))
              for s in spacetime.get_simplices()]
    top_size = max(len(t) for t in tuples)
    return [t for t in tuples if len(t) == top_size]


def _boundary_of_signed_top_chain(chain, d, epsilon):
    """∂_d (Σ_t ε_t·t) as a list of coefficients over the (d-1)-simplices.

    ∂_d is flat row-major with cols = |C_d| (= len(epsilon)); the chain
    coefficient on row r is Σ_c ∂_d[r, c]·ε_c.
    """
    flat = chain.boundary_matrix(d)
    cols = len(epsilon)
    rows = len(flat) // cols if cols else 0
    return [sum(flat[r * cols + c] * epsilon[c] for c in range(cols))
            for r in range(rows)]


class TestFundamentalClass(unittest.TestCase):

    def _check_closed_oriented(self, topology, expected_dimension):
        spacetime = _build(topology)
        chain = cobordism.ChainComplex.from_spacetime(spacetime)
        d = chain.dimension()
        self.assertEqual(d, expected_dimension)

        epsilon = list(chain.fundamental_class())
        tops = [tuple(t) for t in chain.oriented_top_simplices()]

        # orientedTopSimplices length == number of top simplices == |ε|.
        num_top = chain.num_simplices(d)
        self.assertEqual(len(tops), num_top)
        self.assertEqual(len(epsilon), num_top)

        # The returned top simplices are exactly the manifold's top simplices,
        # as sorted vertex-id tuples (the canonical ∂_d column order).
        self.assertEqual(set(tops), set(_top_tuples(spacetime)))
        self.assertEqual(len(set(tops)), len(tops))  # no duplicates
        for t in tops:
            self.assertEqual(list(t), sorted(t))

        # ε_t ∈ {±1}, deterministically sign-normalized (first nonzero is +1).
        self.assertTrue(all(e in (-1, 1) for e in epsilon))
        self.assertEqual(next(e for e in epsilon if e != 0), 1)

        # The signed top chain is a fundamental cycle: ∂_d (Σ ε_t·t) = 0, so
        # every codimension-one face's two incident top simplices cancel.
        self.assertTrue(
            all(coefficient == 0
                for coefficient in _boundary_of_signed_top_chain(chain, d, epsilon)))

    def test_three_torus_is_a_fundamental_cycle(self):
        # T^3 = S^1 x S^1 x S^1: a closed oriented 3-manifold (b_3 = 1).
        self._check_closed_oriented(_three_torus(), 3)

    def test_two_sphere_sanity_check(self):
        # S^2: closed oriented surface (b_2 = 1).
        self._check_closed_oriented(_two_sphere(), 2)

    def test_two_torus_is_a_fundamental_cycle(self):
        # T^2 = S^1 x S^1: another closed oriented surface (b_2 = 1).
        self._check_closed_oriented(_torus(), 2)

    def test_consistent_with_homology_top_betti(self):
        # The fundamental class exists exactly when b_d = 1; cross-check that the
        # closed oriented fixtures have a one-dimensional top homology.
        for topology, dimension in ((_three_torus(), 3),
                                    (_two_sphere(), 2),
                                    (_torus(), 2)):
            with self.subTest(dimension=dimension):
                chain = cobordism.ChainComplex.from_spacetime(_build(topology))
                self.assertEqual(chain.betti_numbers()[dimension], 1)


class TestFundamentalClassRequiresClosedOriented(unittest.TestCase):
    """fundamental_class() must raise when dim ker ∂_d ≠ 1 (#160).

    A ball SolidSimplex(n) is contractible (b_n = 0) and ℝP² is closed but
    non-orientable (b_2(ℝP²; ℚ) = 0); in both, ker ∂_d is 0-dimensional, so no
    fundamental class exists and the documented contract is to raise.

    Regression for the bug: Eigen's FullPivLU::kernel() returns a single
    all-zero column for a 0-dimensional kernel (never a zero-column matrix), so
    the old kernel.cols() != 1 guard never fired — fundamental_class() instead
    returned an all-zero ε vector and read one past the end of the generator
    during sign normalization (undefined behavior).
    """

    def _assert_no_fundamental_class(self, topology, dimension):
        chain = cobordism.ChainComplex.from_spacetime(_build(topology))
        self.assertEqual(chain.dimension(), dimension)
        # The reason it must raise: the top Betti number is 0 (ker ∂_d = 0).
        self.assertEqual(chain.betti_numbers()[dimension], 0)
        with self.assertRaises(RuntimeError):
            chain.fundamental_class()

    def test_balls_have_no_fundamental_class(self):
        # SolidSimplex(n) is the n-ball: contractible, so b_n = 0.
        for n in (2, 3, 4):
            with self.subTest(n=n):
                self._assert_no_fundamental_class(tessera.SolidSimplex(n), n)

    def test_real_projective_plane_has_no_fundamental_class(self):
        # ℝP² is closed but non-orientable: b_2(ℝP²; ℚ) = 0, no [W] over ℤ.
        self._assert_no_fundamental_class(tessera.RealProjectivePlane(), 2)

    def test_closed_oriented_fixtures_still_return_pm1_class(self):
        # The fix must leave b_d = 1 untouched: a valid ±1 fundamental class is
        # still returned (the oriented state-sum weight depends on this).
        for topology, dimension in ((_two_sphere(), 2),
                                    (_torus(), 2),
                                    (_three_torus(), 3)):
            with self.subTest(dimension=dimension):
                chain = cobordism.ChainComplex.from_spacetime(_build(topology))
                self.assertEqual(chain.betti_numbers()[dimension], 1)
                epsilon = list(chain.fundamental_class())
                self.assertEqual(len(epsilon), chain.num_simplices(dimension))
                self.assertTrue(all(e in (-1, 1) for e in epsilon))
                self.assertEqual(next(e for e in epsilon if e != 0), 1)


if __name__ == "__main__":
    unittest.main()
