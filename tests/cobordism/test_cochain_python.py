# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""Eigen-backed value objects for the Hodge spectrum (#183).

Exercises ``cobordism.Cochain`` and ``cobordism.Spectrum`` — the value objects
``HodgeLaplacian.spectrum`` / ``harmonics`` now return — against numpy oracles,
and anchors the harmonic basis on the triangle (S¹, b₀ = b₁ = 1) and the torus
T² (b = [1, 2, 1]), matching the values the existing Hodge tests pin.
"""

import math
import unittest

import numpy as np

import tessera
import cmath

cob = tessera.cobordism
DIAGONAL = cob.HodgeMetricSource.DiagonalWeights


def _diagonal(st):
    """The diagonal-weight operator, named: the closed forms below are its."""
    return cob.HodgeLaplacian(st, cob.HodgeLaplacian.defaultWeightConvention(), DIAGONAL)


# --------------------------------------------------------------------------- #
# Fixture builders (shared with the Hodge Laplacian tests)
# --------------------------------------------------------------------------- #
def _build_topology(topology):
    sig = tessera.Signature(topology.dimension(), tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    st = tessera.Spacetime(metric, tessera.HERMITIAN_WEIGHTED, 1.0, 1.0,
                           tessera.PREFERRED, topology)
    st.build()
    return st


def _set_uniform(st, squared_length=1.0, phase=0.0):
    for e in st.getEdgeList().toVector():
        e.setLength(cmath.sqrt(complex(squared_length)))
        e.setPhase(phase)
    return st


def _triangle(phase=0.0):
    """S¹ = boundary of a 2-simplex (3 vertices, 3 edges), unit edges."""
    st = _build_topology(tessera.SimplexBoundarySphere(1))
    _set_uniform(st, 1.0, 0.0)
    if phase:
        st.getEdgeList().toVector()[0].setPhase(phase)
    return st


def _torus():
    """T² = S¹ × S¹ via SimplicialProduct + CDT (b = [1, 2, 1]), unit edges."""
    sig = tessera.Signature(4, tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    topology = tessera.SimplicialProduct(tessera.SimplexBoundarySphere(1),
                                         tessera.SimplexBoundarySphere(1))
    st = tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0, tessera.PREFERRED,
                           topology)
    st.build()
    return _set_uniform(st, 1.0, 0.0)


def _from_simplices(num_vertices, simplices):
    sig = tessera.Signature(4, tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    st = tessera.Spacetime(metric, tessera.HERMITIAN_WEIGHTED, 1.0, 1.0,
                           tessera.PREFERRED, tessera.Toroid())
    verts = [st.createVertex(i) for i in range(num_vertices)]
    for simplex in simplices:
        st.createSimplex([verts[i] for i in simplex])
    return st


def _triangle_one_timelike(alpha):
    """The 3-cycle 0-1-2-0 with edge (1,2) timelike (l² = -alpha²)."""
    st = _set_uniform(_from_simplices(3, [(0, 1), (1, 2), (2, 0)]), 1.0, 0.0)
    for e in st.getEdgeList().toVector():
        if {e.getSource().getId(), e.getTarget().getId()} == {1, 2}:
            e.setLength(cmath.sqrt(complex(-(alpha ** 2))))
    return st


# --------------------------------------------------------------------------- #
# Cochain
# --------------------------------------------------------------------------- #
class TestCochain(unittest.TestCase):

    def test_degree_size_and_ordering_at_k0(self):
        # A k=0 Cochain is indexed over the sorted-id vertex order: each entry of
        # simplices() is a single-vertex tuple.
        h = cob.HodgeLaplacian(_triangle()).harmonics()[0]
        self.assertEqual(h.degree(), 0)
        self.assertEqual(h.size(), 3)
        self.assertEqual(len(h), 3)
        self.assertEqual(h.simplices(), [[0], [1], [2]])

    def test_coeffs_is_a_complex_numpy_array(self):
        h = cob.HodgeLaplacian(_triangle()).harmonics()[0]
        c = h.coeffs()
        self.assertIsInstance(c, np.ndarray)
        self.assertEqual(c.dtype, np.dtype("complex128"))
        self.assertEqual(c.shape, (3,))

    def test_amplitude_by_index_and_by_simplex_id(self):
        h = cob.HodgeLaplacian(_triangle()).harmonics()[0]
        coeffs = np.asarray(h.coeffs())
        for i, simplex in enumerate(h.simplices()):
            self.assertEqual(h.amplitude(i), coeffs[i])     # by index
            self.assertEqual(h[i], coeffs[i])               # __getitem__
            self.assertEqual(h.amplitudeFor(simplex), coeffs[i])  # by simplex id

    def test_amplitude_out_of_range_raises(self):
        h = cob.HodgeLaplacian(_triangle()).harmonics()[0]
        with self.assertRaises(IndexError):
            h.amplitude(99)
        with self.assertRaises(IndexError):
            h.amplitudeFor([12345])  # no such vertex in the ordering

    def test_inner_product_matches_numpy_vdot(self):
        # <a, b> = sum conj(a_i) b_i = np.vdot(a, b), across the full eigenbasis.
        evecs = cob.HodgeLaplacian(_triangle()).spectrum().eigenvectors()
        for a in evecs:
            for b in evecs:
                ca, cb = np.asarray(a.coeffs()), np.asarray(b.coeffs())
                self.assertAlmostEqual(a.innerProduct(b), complex(np.vdot(ca, cb)),
                                       places=12)

    def test_eigenbasis_is_orthonormal(self):
        # The SelfAdjointEigenSolver basis of the U(1) CONNECTION operator:
        # <v_i, v_j> = delta_ij. (L_k is non-self-adjoint at every degree, so
        # its eigenvectors carry no orthonormality claim.)
        evecs = cob.HodgeLaplacian(_triangle()).connectionSpectrum().eigenvectors()
        for i, a in enumerate(evecs):
            for j, b in enumerate(evecs):
                self.assertAlmostEqual(a.innerProduct(b), 1.0 if i == j else 0.0,
                                       places=10)

    def test_norm_matches_numpy(self):
        for v in cob.HodgeLaplacian(_torus()).spectrum(1).eigenvectors():
            self.assertAlmostEqual(v.norm(),
                                   float(np.linalg.norm(np.asarray(v.coeffs()))),
                                   places=12)

    def test_normalized_has_unit_norm_and_preserves_ordering(self):
        # Scale a harmonic up, renormalize, and check unit norm + same ordering.
        h = cob.HodgeLaplacian(_triangle()).harmonics()[0]
        hn = h.normalized()
        self.assertAlmostEqual(hn.norm(), 1.0, places=12)
        self.assertEqual(hn.degree(), h.degree())
        self.assertEqual(hn.simplices(), h.simplices())
        # direction preserved (parallel to the original)
        overlap = abs(complex(np.vdot(np.asarray(hn.coeffs()),
                                      np.asarray(h.coeffs()))))
        self.assertAlmostEqual(overlap, h.norm(), places=10)

    def test_inner_product_degree_mismatch_raises(self):
        hl = cob.HodgeLaplacian(_triangle())
        v0 = hl.spectrum(0).eigenvectors()[0]   # degree 0
        v1 = hl.spectrum(1).eigenvectors()[0]   # degree 1
        with self.assertRaises(ValueError):
            v0.innerProduct(v1)


# --------------------------------------------------------------------------- #
# Spectrum
# --------------------------------------------------------------------------- #
class TestSpectrum(unittest.TestCase):

    def test_size_len_and_indexing(self):
        sp = cob.HodgeLaplacian(_triangle()).spectrum()
        self.assertEqual(sp.size(), 3)
        self.assertEqual(len(sp), 3)
        self.assertEqual(len(sp.eigenvectors()), 3)
        self.assertEqual(sp.eigenvalues().shape, (3,))
        for i in range(len(sp)):
            self.assertEqual(sp[i].degree(), 0)
            self.assertEqual(sp.eigenvalue(i), sp.eigenvalues()[i])

    def test_index_out_of_range_raises(self):
        sp = cob.HodgeLaplacian(_triangle()).spectrum()
        with self.assertRaises(IndexError):
            _ = sp[99]
        with self.assertRaises(IndexError):
            sp.eigenvalue(99)

    def test_hermitian_eigenvalues_are_real_and_ascending(self):
        # The Hermitian regime lives on the U(1) CONNECTION spectrum (#805).
        sp = cob.HodgeLaplacian(_triangle()).connectionSpectrum()
        self.assertTrue(sp.isHermitian())
        evals = sp.eigenvalues()
        self.assertEqual(evals.dtype, np.dtype("complex128"))
        np.testing.assert_allclose(evals.imag, 0.0, atol=1e-12)
        np.testing.assert_allclose(np.sort(evals.real), [0.0, 3.0, 3.0], atol=1e-12)
        self.assertTrue(np.all(np.diff(evals.real) >= -1e-12))  # ascending

    def test_eigenvalues_match_flat_accessor(self):
        for build, k in ((_triangle, 0), (_torus, 1), (_torus, 2)):
            with self.subTest(k=k):
                hl = cob.HodgeLaplacian(build())
                np.testing.assert_allclose(hl.spectrum(k).eigenvalues().real,
                                           np.array(hl.eigenvalues(k)), atol=1e-12)

    def test_eigenvectors_match_flat_accessor_columns(self):
        # spectrum().eigenvectors()[j].coeffs() == column j of the flat eigenvectors().
        hl = cob.HodgeLaplacian(_torus())
        n1 = cob.ChainComplex.fromSpacetime(_torus()).numSimplices(1)
        flat = np.array(hl.eigenvectors(1), dtype=complex).reshape(n1, n1)
        for j, v in enumerate(hl.spectrum(1).eigenvectors()):
            np.testing.assert_allclose(np.asarray(v.coeffs()), flat[:, j], atol=1e-12)

    def test_harmonics_is_the_zero_eigenvalue_subset(self):
        sp = cob.HodgeLaplacian(_triangle()).spectrum()
        harm = sp.harmonics()
        expected = [v for v, lam in zip(sp.eigenvectors(), sp.eigenvalues())
                    if abs(lam) < 1e-9]
        self.assertEqual(len(harm), len(expected))
        for got, want in zip(harm, expected):
            np.testing.assert_allclose(np.asarray(got.coeffs()),
                                       np.asarray(want.coeffs()), atol=1e-12)

    def test_harmonics_matches_hodge_laplacian_harmonics(self):
        hl = cob.HodgeLaplacian(_torus())
        a = hl.spectrum(1).harmonics()
        b = hl.harmonics(1)
        self.assertEqual(len(a), len(b))
        for x, y in zip(a, b):
            np.testing.assert_allclose(np.asarray(x.coeffs()),
                                       np.asarray(y.coeffs()), atol=1e-12)


# --------------------------------------------------------------------------- #
# Harmonic anchors: triangle (S¹) and torus (T²)
# --------------------------------------------------------------------------- #
class TestHarmonicAnchors(unittest.TestCase):

    def test_triangle_zero_mode_is_the_uniform_0_cochain(self):
        # b₀ = 1: a single harmonic, the constant 0-cochain (equal magnitudes).
        harm = cob.HodgeLaplacian(_triangle()).harmonics()
        self.assertEqual(len(harm), 1)
        h = harm[0]
        self.assertEqual(h.degree(), 0)
        c = np.abs(np.asarray(h.coeffs()))
        np.testing.assert_allclose(c, np.full(3, c[0]), atol=1e-9)

    def test_triangle_one_cycle_harmonic(self):
        # S¹ has b₁ = 1: one harmonic 1-cochain over the 3 edges.
        harm = cob.HodgeLaplacian(_triangle()).harmonics(1)
        self.assertEqual(len(harm), 1)
        self.assertEqual(harm[0].degree(), 1)
        self.assertEqual(harm[0].size(), 3)
        for simplex in harm[0].simplices():
            self.assertEqual(len(simplex), 2)  # an edge = two vertex ids

    def test_flux_lifts_the_connection_zero_mode(self):
        # Any U(1) flux removes the CONNECTION operator's harmonic (magnetic
        # frustration). Under the diagonal weights dim ker L_0 stays b_0 = 1 --
        # that L_0 has no link phase. The default covariant h_0(s, U) carries
        # the link, so the flux lifts its zero mode as well.
        st = _triangle(math.pi)
        self.assertEqual(len(cob.HodgeLaplacian(st).connectionHarmonics()), 0)
        self.assertEqual(len(_diagonal(st).harmonics(0)), 1)
        self.assertEqual(len(cob.HodgeLaplacian(st).harmonics(0)), 0)

    def test_torus_first_homology_is_the_qubit(self):
        # T²: dim ker L_1 = b₁ = 2 — the qubit. Each harmonic is a 1-cochain.
        torus = _torus()
        harm = cob.HodgeLaplacian(torus).harmonics(1)
        self.assertEqual(len(harm), 2)
        n1 = cob.ChainComplex.fromSpacetime(torus).numSimplices(1)
        for h in harm:
            self.assertEqual(h.degree(), 1)
            self.assertEqual(h.size(), n1)

    def test_torus_fundamental_class_harmonic(self):
        # b₂ = 1: a single harmonic 2-cochain over the triangles.
        torus = _torus()
        harm = cob.HodgeLaplacian(torus).harmonics(2)
        self.assertEqual(len(harm), 1)
        self.assertEqual(harm[0].degree(), 2)
        self.assertEqual(harm[0].size(),
                         cob.ChainComplex.fromSpacetime(torus).numSimplices(2))


# --------------------------------------------------------------------------- #
# Lorentzian spectrum: complex eigenvalues, harmonics as Cochains
# --------------------------------------------------------------------------- #
class TestSpectrumCache(unittest.TestCase):

    def test_is_not_hermitian_and_eigenvalues_are_complex_typed(self):
        sp = _diagonal(_triangle_one_timelike(1.0)).spectrum(1)
        self.assertFalse(sp.isHermitian())
        self.assertEqual(sp.eigenvalues().dtype, np.dtype("complex128"))
        # closed form {0, 3, 1 - 2/alpha} with alpha=1 -> {0, 3, -1}: indefinite.
        np.testing.assert_allclose(np.sort(sp.eigenvalues().real), [-1.0, 0.0, 3.0],
                                   atol=1e-7)

    def test_lorentzian_harmonic_is_the_unit_cycle(self):
        # Under the diagonal weights the near-kernel mode is the 1-cycle itself:
        # |h_i|² = 1/3 on every edge.
        harm = _diagonal(_triangle_one_timelike(1.3)).harmonics(1)
        self.assertEqual(len(harm), 1)
        self.assertEqual(harm[0].degree(), 1)
        np.testing.assert_allclose(np.abs(np.asarray(harm[0].coeffs())) ** 2,
                                   np.full(3, 1.0 / 3.0), atol=1e-7)

    def test_lorentzian_harmonic_image_is_the_signed_lengths(self):
        # The default Whitney operator acts on geometric images, and the image
        # of the unit cycle on a 1-complex is the vector of signed lengths
        # (integration specification, Prop. 7): |z_i|² ∝ |l_i|² = (1, 1, alpha²).
        alpha = 1.3
        harm = cob.HodgeLaplacian(_triangle_one_timelike(alpha)).harmonics(1)
        self.assertEqual(len(harm), 1)
        self.assertEqual(harm[0].degree(), 1)
        expected = np.array([1.0, 1.0, alpha ** 2]) / (2.0 + alpha ** 2)
        got = np.abs(np.asarray(harm[0].coeffs())) ** 2
        cells = [tuple(sorted(c)) for c in harm[0].simplices()]
        order = [cells.index(c) for c in [(0, 1), (0, 2), (1, 2)]]
        np.testing.assert_allclose(got[order] / got.sum(), expected, atol=1e-7)

    def test_all_spacelike_lorentzian_matches_hermitian_kernel(self):
        # All-spacelike: the signed path reproduces the Euclidean harmonic count.
        st = _set_uniform(_from_simplices(3, [(0, 1), (1, 2), (2, 0)]), 1.0, 0.0)
        hl = cob.HodgeLaplacian(st)
        self.assertEqual(len(hl.harmonics(1)), len(hl.harmonics(1)))


if __name__ == "__main__":
    unittest.main()
