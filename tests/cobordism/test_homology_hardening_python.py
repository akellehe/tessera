# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""ChainComplex + IntegerLinalg homology hardening (#153).

A single comprehensive sweep of the simplicial homology (#64) over **every**
cobordism topology fixture, plus property tests for the exact linear-algebra
primitives (#106) against independent numpy / number-theoretic oracles.

* **Homology sweep.** For each fixture the chain complex must reproduce its
  known Betti numbers over ℚ and GF(2), its torsion coefficients at every
  degree, its Euler characteristic (matching both the face-count χ and the
  alternating Betti sum), the chain-complex axiom ∂²=0, and — for the closed
  oriented manifolds — a fundamental class of ±1 signs whose signed top chain is
  a cycle. The non-closed / non-orientable fixtures (balls, RP²) must instead
  *raise* when a fundamental class is requested.

* **orientedTopSimplices consistency.** The top-simplex list lines up with the
  d-th column basis k_simplex_vertices(d), the boundary-matrix column count, the
  fundamental-class length, and the manifold's actual top simplices.

* **GF(2) primitives.** gf2Rank / gf2Nullspace match a numpy GF(2) oracle on
  random binary matrices, and — applied to the fixtures' boundary matrices —
  reconstruct bettiNumbersGF2 from the GF(2) rank–nullity identity.

* **Smith normal form.** smith_normal_form matches an independent determinantal-
  divisor oracle on random small integer matrices (rank, invariant factors,
  transpose invariance, det = ∏ factors) and recovers the fixtures' torsion.
"""

import collections
import itertools
import unittest
from math import gcd

import numpy as np

import tessera

cob = tessera.cobordism


# --------------------------------------------------------------------------- #
# Fixture builders.
# --------------------------------------------------------------------------- #
def _build(topology):
    sig = tessera.Signature(topology.dimension(), tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    st = tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0,
                           tessera.PREFERRED, topology)
    st.build()
    return st


def _chain(topology):
    return cob.ChainComplex.from_spacetime(_build(topology))


def _torus2():
    return tessera.SimplicialProduct(tessera.SimplexBoundarySphere(1),
                                     tessera.SimplexBoundarySphere(1))


def _torus3():
    return tessera.SimplicialProduct(_torus2(), tessera.SimplexBoundarySphere(1))


def _sphere_circle():
    return tessera.SphereCircleProduct()                    # S²×S¹


def _sphere_sphere():
    return tessera.SimplicialProduct(tessera.SimplexBoundarySphere(2),
                                     tessera.SimplexBoundarySphere(2))  # S²×S²


def _stellar_t3():
    return tessera.StellarSubdivision(_torus3())


def _stellar_sphere_circle():
    return tessera.StellarSubdivision(tessera.SphereCircleProduct())


# (name, topology factory, Betti over ℚ, Betti over GF(2), {degree: torsion},
#  Euler characteristic, is-closed-oriented). Torsion defaults to [] per degree.
Fixture = collections.namedtuple(
    "Fixture", "name make betti_q betti_gf2 torsion euler closed_oriented")

FIXTURES = []
for _n in range(1, 6):  # S^1 .. S^5
    _b = [1] + [0] * (_n - 1) + [1]
    FIXTURES.append(Fixture(f"S^{_n}", (lambda n=_n: tessera.SimplexBoundarySphere(n)),
                            _b, _b, {}, 1 + (-1) ** _n, True))
for _n in range(1, 5):  # D^1 .. D^4 (contractible balls, with boundary)
    _b = [1] + [0] * _n
    FIXTURES.append(Fixture(f"D^{_n}", (lambda n=_n: tessera.SolidSimplex(n)),
                            _b, _b, {}, 1, False))
FIXTURES += [
    Fixture("RP^2", tessera.RealProjectivePlane, [1, 0, 0], [1, 1, 1], {1: [2]}, 1, False),
    Fixture("CP^2", tessera.ComplexProjectivePlane,
            [1, 0, 1, 0, 1], [1, 0, 1, 0, 1], {}, 3, True),
    Fixture("RP^3", tessera.RealProjectiveSpace, [1, 0, 0, 1], [1, 1, 1, 1], {1: [2]}, 0, True),
    Fixture("S^2xS^1", _sphere_circle, [1, 1, 1, 1], [1, 1, 1, 1], {}, 0, True),
    Fixture("T^2", _torus2, [1, 2, 1], [1, 2, 1], {}, 0, True),
    Fixture("T^3", _torus3, [1, 3, 3, 1], [1, 3, 3, 1], {}, 0, True),
    Fixture("S^2xS^2", _sphere_sphere, [1, 0, 2, 0, 1], [1, 0, 2, 0, 1], {}, 4, True),
    Fixture("stellar(T^3)", _stellar_t3, [1, 3, 3, 1], [1, 3, 3, 1], {}, 0, True),
    Fixture("stellar(S^2xS^1)", _stellar_sphere_circle,
            [1, 1, 1, 1], [1, 1, 1, 1], {}, 0, True),
]


# --------------------------------------------------------------------------- #
# Independent oracles.
# --------------------------------------------------------------------------- #
def _gf2_rank_np(matrix):
    """GF(2) rank of a 0/1 matrix via Gaussian elimination (pure numpy)."""
    a = (np.asarray(matrix, dtype=np.int64) & 1).copy()
    if a.ndim == 1:
        a = a.reshape(1, -1) if a.size else a.reshape(0, 0)
    rows, cols = a.shape
    rank = 0
    for col in range(cols):
        if rank >= rows:
            break
        piv = next((i for i in range(rank, rows) if a[i, col] & 1), None)
        if piv is None:
            continue
        a[[rank, piv]] = a[[piv, rank]]
        for i in range(rows):
            if i != rank and (a[i, col] & 1):
                a[i] ^= a[rank]
        rank += 1
    return rank


def _determinantal_invariant_factors(matrix):
    """Invariant factors of an integer matrix via determinantal divisors.

    D_k = gcd of all k×k minors; the k-th invariant factor is D_k / D_{k-1}
    (D_0 = 1), and the rank is the number of nonzero D_k. An exact, independent
    construction of the Smith normal form for small matrices (the minors are
    evaluated through numpy and rounded — kept exact by small entries/sizes).
    """
    m = np.asarray(matrix, dtype=np.int64)
    rows, cols = m.shape
    prev, factors = 1, []
    for k in range(1, min(rows, cols) + 1):
        divisor = 0
        for ri in itertools.combinations(range(rows), k):
            for ci in itertools.combinations(range(cols), k):
                minor = int(round(np.linalg.det(m[np.ix_(ri, ci)].astype(float))))
                divisor = gcd(divisor, abs(minor))
        if divisor == 0:
            break
        factors.append(divisor // prev)
        prev = divisor
    return factors


# --------------------------------------------------------------------------- #
# 1. Homology across every topology fixture.
# --------------------------------------------------------------------------- #
class TestHomologySweep(unittest.TestCase):

    def _signed_top_chain_boundary(self, chain, d, epsilon):
        """Coefficients of ∂_d(Σ_t ε_t·t) over the (d-1)-simplices."""
        flat = chain.boundary_matrix(d)
        cols = len(epsilon)
        rows = len(flat) // cols if cols else 0
        return [sum(flat[r * cols + c] * epsilon[c] for c in range(cols))
                for r in range(rows)]

    def test_all_fixtures(self):
        for fx in FIXTURES:
            with self.subTest(fixture=fx.name):
                chain = _chain(fx.make())
                dimension = chain.dimension()

                # Betti over ℚ and GF(2).
                self.assertEqual(chain.betti_numbers(), fx.betti_q)
                self.assertEqual(chain.betti_numbers_gf2(), fx.betti_gf2)

                # Torsion at every degree.
                for k in range(dimension + 1):
                    self.assertEqual(list(chain.torsion(k)), fx.torsion.get(k, []),
                                     f"{fx.name}: torsion({k})")

                # Euler characteristic: face-count and homological agree.
                self.assertEqual(chain.euler_characteristic(), fx.euler)
                self.assertEqual(
                    sum((-1) ** k * b for k, b in enumerate(chain.betti_numbers())),
                    fx.euler)

                # Chain-complex axiom.
                self.assertTrue(chain.boundary_composes_to_zero())

                # Fundamental class: a ±1 cycle exactly for the closed oriented
                # fixtures (b_d = 1). The non-closed / non-orientable ones have
                # trivial top homology (b_d ≠ 1); the contract for *requesting* a
                # fundamental class there is exercised in TestFundamentalClassContract.
                if fx.closed_oriented:
                    self.assertEqual(chain.betti_numbers()[dimension], 1)
                    epsilon = list(chain.fundamental_class())
                    self.assertEqual(len(epsilon), chain.num_simplices(dimension))
                    self.assertTrue(all(e in (-1, 1) for e in epsilon))
                    self.assertEqual(next(e for e in epsilon if e != 0), 1)  # normalized
                    self.assertTrue(
                        all(c == 0 for c in
                            self._signed_top_chain_boundary(chain, dimension, epsilon)))
                else:
                    self.assertNotEqual(chain.betti_numbers()[dimension], 1)


# --------------------------------------------------------------------------- #
# 2. orientedTopSimplices consistency.
# --------------------------------------------------------------------------- #
class TestOrientedTopSimplices(unittest.TestCase):

    @staticmethod
    def _actual_top_tuples(spacetime):
        tuples = [tuple(sorted(v.get_id() for v in s.get_vertices()))
                  for s in spacetime.get_simplices()]
        top_size = max(len(t) for t in tuples)
        return {t for t in tuples if len(t) == top_size}

    def test_consistency_on_closed_fixtures(self):
        for fx in FIXTURES:
            if not fx.closed_oriented:
                continue
            with self.subTest(fixture=fx.name):
                spacetime = _build(fx.make())               # held alive locally
                chain = cob.ChainComplex.from_spacetime(spacetime)
                dimension = chain.dimension()
                tops = [tuple(t) for t in chain.oriented_top_simplices()]
                column_basis = [tuple(t) for t in chain.k_simplex_vertices(dimension)]

                # Same length as the d-th cell count, the column basis, and ε.
                self.assertEqual(len(tops), chain.num_simplices(dimension))
                self.assertEqual(tops, column_basis)
                self.assertEqual(len(tops), len(chain.fundamental_class()))

                # Each is a sorted, unique vertex-id tuple.
                for t in tops:
                    self.assertEqual(list(t), sorted(t))
                self.assertEqual(len(set(tops)), len(tops))

                # The boundary matrix ∂_d has exactly |tops| columns.
                flat = chain.boundary_matrix(dimension)
                rows = chain.num_simplices(dimension - 1)
                self.assertEqual(len(flat), rows * len(tops))

                # They are the manifold's actual top simplices.
                self.assertEqual(set(tops), self._actual_top_tuples(spacetime))


# --------------------------------------------------------------------------- #
# 2b. fundamental_class() contract — documents a real production bug (#153).
# --------------------------------------------------------------------------- #
class TestFundamentalClassContract(unittest.TestCase):
    """``ChainComplex.fundamental_class()`` raises when no fundamental class
    exists — when ``dim ker ∂_d ≠ 1`` (the header's @throws, and
    ChainComplex.cpp's own comment: "anything else has no fundamental class …
    and the call throws"). This holds for every complex whose top boundary
    ``∂_d`` has full column rank (trivial kernel, ``b_d = 0``): every ball
    ``Δⁿ`` and the non-orientable ``RP²``.

    The guard was fixed in #163: ``FullPivLU::dimensionOfKernel()`` (plus a
    past-the-end ``firstNonzero`` check) replaced the ``kernel.cols() != 1``
    test, which never fired for a 0-dimensional kernel — Eigen's
    ``FullPivLU::kernel()`` returns a single all-zero column there rather than a
    zero-column matrix, so the old guard returned an all-zero ε vector and read
    one past the end of the generator. The closed oriented case (``b_d = 1``,
    kernel of one column) is covered by the homology sweep above.
    """

    def test_ball_should_raise_without_fundamental_class(self):
        # Δ³ is contractible (b_3 = 0): no fundamental class ⇒ must raise.
        with self.assertRaises(RuntimeError):
            _chain(tessera.SolidSimplex(3)).fundamental_class()

    def test_non_orientable_should_raise_without_fundamental_class(self):
        # RP² is closed but non-orientable (b_2 = 0): must raise.
        with self.assertRaises(RuntimeError):
            _chain(tessera.RealProjectivePlane()).fundamental_class()


# --------------------------------------------------------------------------- #
# 3. GF(2) primitives vs a numpy oracle + the homology rank–nullity identity.
# --------------------------------------------------------------------------- #
class TestGf2Primitives(unittest.TestCase):

    def test_random_binary_matrices(self):
        rng = np.random.default_rng(153)
        shapes = [(1, 1), (2, 3), (3, 2), (4, 4), (5, 8), (8, 5), (6, 6), (7, 9)]
        for rows, cols in shapes:
            for trial in range(6):
                with self.subTest(rows=rows, cols=cols, trial=trial):
                    a = rng.integers(0, 2, size=(rows, cols))
                    flat = [int(v) for v in a.reshape(-1)]
                    rank = cob.gf2_rank(flat, rows, cols)
                    basis = [list(v) for v in cob.gf2_nullspace(flat, rows, cols)]

                    self.assertEqual(rank, _gf2_rank_np(a))
                    self.assertEqual(rank + len(basis), cols)        # rank–nullity
                    for x in basis:
                        self.assertEqual(len(x), cols)
                        self.assertTrue(set(x) <= {0, 1})
                        self.assertTrue(np.all((a @ np.asarray(x)) % 2 == 0))
                    if basis:                                        # independent
                        self.assertEqual(_gf2_rank_np(basis), len(basis))

    def test_betti_gf2_from_boundary_ranks(self):
        # b_k(GF2) = |C_k| − rank₂ ∂_k − rank₂ ∂_{k+1}, with the boundary ranks
        # taken from the C++ gf2_rank and cross-checked against the numpy oracle;
        # the assembled vector must equal betti_numbers_gf2().
        for fx in (Fixture("S^2", lambda: tessera.SimplexBoundarySphere(2),
                           [1, 0, 1], [1, 0, 1], {}, 2, True),
                   FIXTURES[next(i for i, f in enumerate(FIXTURES) if f.name == "RP^2")],
                   FIXTURES[next(i for i, f in enumerate(FIXTURES) if f.name == "RP^3")],
                   FIXTURES[next(i for i, f in enumerate(FIXTURES) if f.name == "T^2")]):
            with self.subTest(fixture=fx.name):
                chain = _chain(fx.make())
                dimension = chain.dimension()

                def gf2_boundary_rank(k):
                    if k <= 0 or k > dimension:
                        return 0
                    rows = chain.num_simplices(k - 1)
                    cols = chain.num_simplices(k)
                    if rows == 0 or cols == 0:
                        return 0
                    flat = [int(v) & 1 for v in chain.boundary_matrix(k)]
                    cpp = cob.gf2_rank(flat, rows, cols)
                    self.assertEqual(cpp, _gf2_rank_np(np.asarray(flat).reshape(rows, cols)))
                    return cpp

                betti = []
                for k in range(dimension + 1):
                    ck = chain.num_simplices(k)
                    betti.append(ck - gf2_boundary_rank(k) - gf2_boundary_rank(k + 1))
                self.assertEqual(betti, chain.betti_numbers_gf2())
                self.assertEqual(betti, fx.betti_gf2)


# --------------------------------------------------------------------------- #
# 4. Smith normal form vs a determinantal-divisor oracle.
# --------------------------------------------------------------------------- #
class TestSmithNormalForm(unittest.TestCase):

    def test_diagonal_anchors(self):
        # SNF(diag(2,3)) = diag(1,6) (gcd then lcm); SNF(diag(2,4,8)) = diag(2,4,8).
        snf = cob.smith_normal_form([2, 0, 0, 0, 3, 0, 0, 0, 0], 3, 3)
        self.assertEqual(snf.rank, 2)
        self.assertEqual(list(snf.invariant_factors), [1, 6])
        snf = cob.smith_normal_form([2, 0, 0, 0, 4, 0, 0, 0, 8], 3, 3)
        self.assertEqual(snf.rank, 3)
        self.assertEqual(list(snf.invariant_factors), [2, 4, 8])

    def test_random_integer_matrices_against_oracle(self):
        rng = np.random.default_rng(64064)
        shapes = [(2, 2), (3, 3), (4, 4), (2, 4), (4, 2), (3, 5), (4, 3)]
        for rows, cols in shapes:
            for trial in range(8):
                with self.subTest(rows=rows, cols=cols, trial=trial):
                    m = rng.integers(-2, 3, size=(rows, cols))
                    flat = [int(v) for v in m.reshape(-1)]
                    snf = cob.smith_normal_form(flat, rows, cols)
                    factors = list(snf.invariant_factors)
                    oracle = _determinantal_invariant_factors(m)

                    self.assertEqual(snf.rank, len(oracle))
                    self.assertEqual(factors, oracle)
                    self.assertEqual(snf.rank, cob.integer_rank(flat, rows, cols))
                    self.assertEqual(snf.rank, int(np.linalg.matrix_rank(m)))
                    # Each invariant factor divides the next.
                    for a, b in zip(factors, factors[1:]):
                        self.assertEqual(b % a, 0)
                    # Transpose has the same invariant factors.
                    snf_t = cob.smith_normal_form(
                        [int(v) for v in m.T.reshape(-1)], cols, rows)
                    self.assertEqual(list(snf_t.invariant_factors), factors)
                    # Square & full rank: ∏ factors = |det|.
                    if rows == cols and snf.rank == rows:
                        det = int(round(np.linalg.det(m.astype(float))))
                        product = 1
                        for f in factors:
                            product *= f
                        self.assertEqual(product, abs(det))

    def test_torsion_recovered_from_boundary_snf(self):
        # torsion(k) = the invariant factors > 1 of ∂_{k+1}: RP² and RP³ each
        # have a single 2 (from ∂₂ and ∂₂ respectively).
        for name, make in (("RP^2", tessera.RealProjectivePlane),
                           ("RP^3", tessera.RealProjectiveSpace)):
            with self.subTest(fixture=name):
                chain = _chain(make())
                rows = chain.num_simplices(1)        # |C_1|
                cols = chain.num_simplices(2)        # |C_2|
                snf = cob.smith_normal_form(list(chain.boundary_matrix(2)), rows, cols)
                nontrivial = [int(f) for f in snf.invariant_factors if f > 1]
                self.assertEqual(nontrivial, [2])
                self.assertEqual(list(chain.torsion(1)), [2])


if __name__ == "__main__":
    unittest.main()
