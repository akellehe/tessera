# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""Capability A: characteristic numbers (#65).

Euler characteristic and signature (as Observables) plus the aggregate
CharacteristicNumbers. The signature is validated on the intersection form:
the 4-sphere gives the empty form, S^2 x S^2 gives the hyperbolic form
(signature 0 but rank 2 -- which only happens if the cup product is computed
correctly), and CP^2 gives a rank-1 definite form (|signature| = 1).

Stiefel-Whitney numbers are validated against their textbook values:
w1^2[RP^2] = 1, w2[RP^2] = 1, w2^2[CP^2] = w4[CP^2] = 1, every number of S^4
and S^2 x S^2 vanishes, and RP^2 ⊔ RP^2 has all numbers ≡ 0 (mod 2).

Orientation note: CP^2 and its reversal are the same simplicial complex, so the
signature comes out with a fixed but convention-dependent sign; the tests assert
the orientation-independent facts (|signature| = 1, p1 = 3*signature).
"""

import unittest

import tessera

cobordism = tessera.cobordism

# The 10 triangles of the minimal 6-vertex RP^2, as a vertex-tuple template we
# can stamp out at a vertex offset to build disjoint unions by hand.
_RP2_TRIANGLES = [(0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 4, 5), (0, 1, 5),
                  (1, 2, 4), (2, 3, 5), (1, 3, 4), (1, 3, 5), (2, 4, 5)]


def _build(topology):
    signature = tessera.Signature(topology.dimension(), tessera.Lorentzian)
    metric = tessera.Metric(True, signature)
    spacetime = tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0,
                                  tessera.PREFERRED, topology)
    spacetime.build()
    return spacetime


def _empty_spacetime():
    signature = tessera.Signature(4, tessera.Lorentzian)
    metric = tessera.Metric(True, signature)
    return tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0,
                             tessera.PREFERRED, tessera.Toroid())


def _two_disjoint_rp2():
    """RP^2 ⊔ RP^2, built by hand on two disjoint 6-vertex blocks."""
    spacetime = _empty_spacetime()
    vertices = [spacetime.create_vertex(i) for i in range(12)]
    for triangle in _RP2_TRIANGLES:
        spacetime.create_simplex([vertices[i] for i in triangle])
        spacetime.create_simplex([vertices[i + 6] for i in triangle])
    return spacetime


def _stiefel_whitney(spacetime):
    return dict(cobordism.ChainComplex.from_spacetime(
        spacetime).stiefel_whitney_numbers())


def _sphere(n):
    return tessera.SimplexBoundarySphere(n)


def _ball(n):
    return tessera.SolidSimplex(n)


def _s2_cross_s2():
    return tessera.SimplicialProduct(_sphere(2), _sphere(2))


def _torus():
    return tessera.SimplicialProduct(_sphere(1), _sphere(1))


def _s1_cross_s2():
    return tessera.SimplicialProduct(_sphere(1), _sphere(2))


class TestEulerCharacteristic(unittest.TestCase):

    def test_spheres(self):
        # chi(S^n) = 1 + (-1)^n.
        euler = cobordism.EulerCharacteristic()
        for n in range(1, 7):
            with self.subTest(sphere=n):
                self.assertEqual(euler.compute(_build(_sphere(n))),
                                 float(1 + (-1) ** n))

    def test_balls_are_contractible(self):
        # A solid simplex is contractible, so chi = 1.
        euler = cobordism.EulerCharacteristic()
        for n in range(1, 7):
            with self.subTest(ball=n):
                self.assertEqual(euler.compute(_build(_ball(n))), 1.0)

    def test_named_manifolds(self):
        euler = cobordism.EulerCharacteristic()
        cases = [
            (tessera.RealProjectivePlane(), 1),
            (_torus(), 0),
            (_s1_cross_s2(), 0),
            (_s2_cross_s2(), 4),
        ]
        for topology, expected in cases:
            with self.subTest(manifold=type(topology).__name__):
                self.assertEqual(euler.compute(_build(topology)), float(expected))

    def test_matches_alternating_sum_of_betti_numbers(self):
        # chi should equal sum_k (-1)^k b_k for every complex (a consistency
        # check between the f-vector count and homology).
        euler = cobordism.EulerCharacteristic()
        for topology in (_sphere(2), _sphere(4), _ball(3),
                         tessera.RealProjectivePlane(), _torus(), _s2_cross_s2()):
            with self.subTest(manifold=type(topology).__name__):
                spacetime = _build(topology)
                chain = cobordism.ChainComplex.from_spacetime(spacetime)
                betti = chain.betti_numbers()
                from_betti = sum((-1) ** k * b for k, b in enumerate(betti))
                self.assertEqual(euler.compute(spacetime), float(from_betti))

    def test_empty_complex(self):
        empty = tessera.Spacetime()
        self.assertEqual(cobordism.EulerCharacteristic().compute(empty), 0.0)


class TestSignature(unittest.TestCase):

    def test_four_sphere_has_empty_form(self):
        spacetime = _build(_sphere(4))
        chain = cobordism.ChainComplex.from_spacetime(spacetime)
        self.assertEqual(chain.betti_numbers()[2], 0)
        self.assertEqual(list(chain.intersection_form()), [])
        self.assertEqual(cobordism.Signature().compute(spacetime), 0.0)

    def test_s2_cross_s2_is_hyperbolic(self):
        # The intersection form of S^2 x S^2 is the hyperbolic form [[0,1],[1,0]]:
        # signature 0 but rank 2 (nondegenerate, indefinite).
        spacetime = _build(_s2_cross_s2())
        chain = cobordism.ChainComplex.from_spacetime(spacetime)
        self.assertEqual(chain.betti_numbers()[2], 2)
        self.assertEqual(cobordism.Signature().compute(spacetime), 0.0)
        form = list(chain.intersection_form())
        self.assertEqual(len(form), 4)
        diag0, off01, off10, diag1 = form
        self.assertAlmostEqual(diag0, 0.0, places=6)
        self.assertAlmostEqual(diag1, 0.0, places=6)
        self.assertAlmostEqual(off01, off10, places=6)        # symmetric
        self.assertGreater(abs(off01), 1e-6)                  # nonzero crossing
        self.assertLess(diag0 * diag1 - off01 * off10, 0.0)   # det < 0 (indefinite)

    def test_complex_projective_plane_is_definite_rank_one(self):
        # CP^2 has b2 = 1 and a unimodular definite intersection form [±1], so
        # |signature| = 1. (The sign is a convention; see the module docstring.)
        spacetime = _build(tessera.ComplexProjectivePlane())
        chain = cobordism.ChainComplex.from_spacetime(spacetime)
        self.assertEqual(chain.betti_numbers()[2], 1)
        form = list(chain.intersection_form())
        self.assertEqual(len(form), 1)
        self.assertAlmostEqual(abs(form[0]), 1.0, places=6)
        self.assertEqual(abs(chain.signature()), 1)
        self.assertEqual(abs(cobordism.Signature().compute(spacetime)), 1.0)

    def test_intersection_form_is_symmetric_and_sized_by_b2(self):
        spacetime = _build(_s2_cross_s2())
        chain = cobordism.ChainComplex.from_spacetime(spacetime)
        b2 = chain.betti_numbers()[2]
        form = list(chain.intersection_form())
        self.assertEqual(len(form), b2 * b2)
        for i in range(b2):
            for j in range(b2):
                self.assertAlmostEqual(form[i * b2 + j], form[j * b2 + i], places=6)

    def test_zero_below_dimension_four(self):
        # The signature here is defined for 4-manifolds; lower dimensions give 0.
        for topology in (_sphere(1), _sphere(2), _sphere(3),
                         tessera.RealProjectivePlane(), _torus(), _s1_cross_s2()):
            with self.subTest(manifold=type(topology).__name__):
                self.assertEqual(cobordism.Signature().compute(_build(topology)), 0.0)


class TestCharacteristicNumbers(unittest.TestCase):

    def test_four_manifolds_have_signature_and_pontryagin(self):
        for topology in (_sphere(4), _s2_cross_s2(),
                         tessera.ComplexProjectivePlane()):
            with self.subTest(manifold=type(topology).__name__):
                numbers = cobordism.CharacteristicNumbers.of(_build(topology))
                self.assertIsNotNone(numbers.signature)
                # Hirzebruch signature theorem (cross-validation V1): the only
                # Pontryagin number in dimension four is p1 = 3 * signature.
                self.assertEqual(numbers.pontryagin_numbers["p1"], 3 * numbers.signature)

    def test_complex_projective_plane_pontryagin(self):
        # p1(CP^2) = 3*signature with |signature| = 1, so |p1| = 3.
        numbers = cobordism.CharacteristicNumbers.of(
            _build(tessera.ComplexProjectivePlane()))
        self.assertEqual(abs(numbers.signature), 1)
        self.assertEqual(numbers.pontryagin_numbers["p1"], 3 * numbers.signature)
        self.assertEqual(abs(numbers.pontryagin_numbers["p1"]), 3)

    def test_four_sphere_pontryagin_vanishes(self):
        numbers = cobordism.CharacteristicNumbers.of(_build(_sphere(4)))
        self.assertEqual(numbers.signature, 0)
        self.assertEqual(numbers.pontryagin_numbers["p1"], 0)

    def test_euler_agrees_with_observable(self):
        euler = cobordism.EulerCharacteristic()
        for topology in (_sphere(2), _sphere(4), _ball(4),
                         tessera.RealProjectivePlane(), _s2_cross_s2()):
            with self.subTest(manifold=type(topology).__name__):
                spacetime = _build(topology)
                numbers = cobordism.CharacteristicNumbers.of(spacetime)
                self.assertEqual(float(numbers.euler), euler.compute(spacetime))

    def test_below_dimension_four_has_no_signature_or_pontryagin(self):
        for topology in (tessera.RealProjectivePlane(), _torus(), _sphere(3)):
            with self.subTest(manifold=type(topology).__name__):
                numbers = cobordism.CharacteristicNumbers.of(_build(topology))
                self.assertIsNone(numbers.signature)
                self.assertEqual(dict(numbers.pontryagin_numbers), {})

    def test_stiefel_whitney_numbers_exposed(self):
        # CharacteristicNumbers.of surfaces the same numbers as the ChainComplex.
        spacetime = _build(tessera.ComplexProjectivePlane())
        numbers = cobordism.CharacteristicNumbers.of(spacetime)
        self.assertEqual(dict(numbers.stiefel_whitney_numbers),
                         _stiefel_whitney(spacetime))
        self.assertEqual(numbers.stiefel_whitney_numbers["w2^2"], 1)


class TestStiefelWhitneyNumbers(unittest.TestCase):
    """Mod-2 Stiefel-Whitney numbers against their textbook values."""

    def test_real_projective_plane(self):
        # w(RP^2) = (1 + a)^3 = 1 + a + a^2, so w1 = a, w2 = a^2. The two SW
        # numbers w1^2[RP^2] and w2[RP^2] both evaluate to 1.
        numbers = _stiefel_whitney(_build(tessera.RealProjectivePlane()))
        self.assertEqual(numbers, {"w1^2": 1, "w2": 1})

    def test_complex_projective_plane(self):
        # w(CP^2) = (1 + x)^3 = 1 + x + x^2 with |x| = 2, so w2 = x, w4 = x^2.
        # The dimension-4 numbers w2^2 and w4 are 1; everything containing w1
        # vanishes because CP^2 is orientable (w1 = 0).
        numbers = _stiefel_whitney(_build(tessera.ComplexProjectivePlane()))
        self.assertEqual(numbers["w2^2"], 1)
        self.assertEqual(numbers["w4"], 1)
        self.assertEqual(numbers["w1w3"], 0)
        self.assertEqual(numbers["w1^2w2"], 0)
        self.assertEqual(numbers["w1^4"], 0)

    def test_spheres_have_vanishing_numbers(self):
        # Every Stiefel-Whitney number of a sphere is zero (spheres bound disks).
        for n in (2, 4):
            with self.subTest(sphere=n):
                numbers = _stiefel_whitney(_build(tessera.SimplexBoundarySphere(n)))
                self.assertTrue(all(value == 0 for value in numbers.values()))

    def test_product_of_spheres_vanishes(self):
        # S^2 x S^2 bounds, so all its Stiefel-Whitney numbers vanish.
        numbers = _stiefel_whitney(_build(_s2_cross_s2()))
        self.assertTrue(all(value == 0 for value in numbers.values()))

    def test_top_number_is_euler_characteristic_mod_two(self):
        # The top Stiefel-Whitney number w_n[M] equals chi(M) mod 2.
        euler = cobordism.EulerCharacteristic()
        cases = [(tessera.RealProjectivePlane(), "w2"),
                 (tessera.SimplexBoundarySphere(2), "w2"),
                 (tessera.ComplexProjectivePlane(), "w4"),
                 (tessera.SimplexBoundarySphere(4), "w4")]
        for topology, top_key in cases:
            with self.subTest(manifold=type(topology).__name__):
                spacetime = _build(topology)
                chi = int(euler.compute(spacetime))
                numbers = _stiefel_whitney(spacetime)
                self.assertEqual(numbers[top_key], chi % 2)

    def test_disjoint_union_is_additive_and_vanishes(self):
        # SW numbers are additive over disjoint union, so RP^2 ⊔ RP^2 has
        # w1^2 = 1 + 1 = 0 and w2 = 1 + 1 = 0 (mod 2).
        numbers = _stiefel_whitney(_two_disjoint_rp2())
        self.assertEqual(numbers, {"w1^2": 0, "w2": 0})

    def test_empty_complex_has_no_numbers(self):
        numbers = _stiefel_whitney(_empty_spacetime())
        self.assertEqual(numbers, {})

    def test_torus_numbers_vanish(self):
        # The 2-torus is parallelizable (a Lie group), hence bounds, hence all
        # of its Stiefel-Whitney numbers vanish.
        self.assertEqual(_stiefel_whitney(_build(_torus())), {"w1^2": 0, "w2": 0})

    def test_higher_steenrod_squares_are_deferred(self):
        # A manifold with b1 > 0 needs a higher cup-i Steenrod square to find
        # the Wu class v_1; the general Sq^i is deferred (#65). The low-level
        # call raises, and the aggregate CharacteristicNumbers.of degrades
        # gracefully by leaving the Stiefel-Whitney family empty.
        deferred = [
            ("S^1 x S^3", tessera.SimplicialProduct(_sphere(1), _sphere(3))),
            ("RP^2 x S^1",
             tessera.SimplicialProduct(tessera.RealProjectivePlane(), _sphere(1))),
        ]
        for label, topology in deferred:
            with self.subTest(manifold=label):
                spacetime = _build(topology)
                with self.assertRaises(Exception):
                    cobordism.ChainComplex.from_spacetime(
                        spacetime).stiefel_whitney_numbers()
                numbers = cobordism.CharacteristicNumbers.of(spacetime)
                self.assertEqual(dict(numbers.stiefel_whitney_numbers), {})


# (label, topology factory, expected Betti numbers) for a spread of manifolds
# whose homology is textbook. Products are assembled from the sphere and
# real/complex projective-plane fixtures via SimplicialProduct.
_KNOWN_MANIFOLDS = [
    ("S^1", lambda: _sphere(1), [1, 1]),
    ("S^2", lambda: _sphere(2), [1, 0, 1]),
    ("S^3", lambda: _sphere(3), [1, 0, 0, 1]),
    ("S^4", lambda: _sphere(4), [1, 0, 0, 0, 1]),
    ("RP^2", tessera.RealProjectivePlane, [1, 0, 0]),
    ("T^2", _torus, [1, 2, 1]),
    ("S^1 x S^2", _s1_cross_s2, [1, 1, 1, 1]),
    ("S^1 x S^3", lambda: tessera.SimplicialProduct(_sphere(1), _sphere(3)),
     [1, 1, 0, 1, 1]),
    ("T^3", lambda: tessera.SimplicialProduct(_torus(), _sphere(1)), [1, 3, 3, 1]),
    ("S^2 x S^2", _s2_cross_s2, [1, 0, 2, 0, 1]),
    ("CP^2", tessera.ComplexProjectivePlane, [1, 0, 1, 0, 1]),
]


class TestKnownManifoldHomology(unittest.TestCase):
    """Betti numbers (over Q) of a spread of known manifolds, cross-checked
    against chi = sum (-1)^k b_k and the boundary^2 = 0 chain-complex axiom."""

    def test_betti_numbers_and_euler(self):
        for label, make, betti in _KNOWN_MANIFOLDS:
            with self.subTest(manifold=label):
                chain = cobordism.ChainComplex.from_spacetime(_build(make()))
                self.assertTrue(chain.boundary_composes_to_zero())
                self.assertEqual(chain.betti_numbers(), betti)
                expected_euler = sum((-1) ** k * b for k, b in enumerate(betti))
                self.assertEqual(chain.euler_characteristic(), expected_euler)


class TestKnownFourManifoldSignatures(unittest.TestCase):
    """Signature and middle Betti number of the closed oriented 4-manifold
    fixtures. (CP^2's signature sign is a convention; |signature| = 1.)"""

    def test_signatures(self):
        # (label, factory, |signature|, b2)
        cases = [
            ("S^4", lambda: _sphere(4), 0, 0),
            ("S^1 x S^3", lambda: tessera.SimplicialProduct(_sphere(1), _sphere(3)),
             0, 0),
            ("S^2 x S^2", _s2_cross_s2, 0, 2),
            ("CP^2", tessera.ComplexProjectivePlane, 1, 1),
        ]
        for label, make, abs_signature, b2 in cases:
            with self.subTest(manifold=label):
                chain = cobordism.ChainComplex.from_spacetime(_build(make()))
                self.assertEqual(chain.betti_numbers()[2], b2)
                self.assertEqual(abs(chain.signature()), abs_signature)


if __name__ == "__main__":
    unittest.main()
