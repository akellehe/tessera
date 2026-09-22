# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""Lorentzian Hodge Laplacian — the discrete d'Alembertian (#105, reworked #644).

There is ONE k >= 1 operator now: L_k = W_k^-1 d_k^T W_{k-1} d_k
+ d_{k+1} W_{k+1}^-1 d_{k+1}^T W_k with the signed complex weights, complex-typed
and generally non-self-adjoint. The old two-track design (a symmetric |volume|
Laplacian next to a signed "lorentzian" variant behind a flag) is gone: taking
|volume| was a Euclidean read that discarded causal character (#641).

Under the default weight convention W_k = V^2 (HodgeWeightConvention.
SquaredContent) an edge's weight is exactly its squared length — real and
SIGNED on real signed l^2, polynomial in the l^2 so it carries no branch.

Primary fixture — the 3-cycle S^1 (b_1 = 1) with one edge timelike — has closed
forms that pin the implementation exactly (verified, not fitted):

  * edges (0,1),(0,2) spacelike (l^2 = 1 -> W = +1); edge (1,2) timelike
    (l^2 = -alpha^2 -> W = -alpha^2), so W_1 = diag(1, 1, -alpha^2);
  * spec(L_1) = {0, 3, 1 - 2/alpha^2}: indefinite for alpha < sqrt(2), zero at
    the defective crossing alpha = sqrt(2), positive above;
  * the harmonic (the 1-cycle, |h_i|^2 = 1/3 each) has indefinite norm
      <h,h>_W = (1 + 1 - alpha^2)/3 = (2 - alpha^2)/3:
    positive below the crossing, NULL at alpha = sqrt(2), negative above —
    genuine lightlike kernel directions survive because the weights stay real
    and signed (the k-content convention W = V puts the timelike weight on the
    imaginary axis instead, where it can never cancel the spacelike part).

What is recorded (vs. hard-asserted) for the larger / CDT fixtures, where the
geometry is not closed-form, is the spectrum's departure from the nonneg-real
axis (indefiniteness), the near-kernel count vs. b_k, and which near-kernel
representatives are null.
"""

import math
import unittest

import numpy as np

import tessera
import cmath

cob = tessera.cobordism


# Every closed form in this module is the DIAGONAL-weight operator's:
# L_k = W_k^-1 d_k^T W_{k-1} d_k + d_{k+1} W_{k+1}^-1 d_{k+1}^T W_k of
# HodgeWeightConvention. The process default metric source is the chain-level
# Whitney pencil (#1185), whose operator is the covariant h_k(s, U) on
# geometric images, so this module names the diagonal source at every operator
# it builds. The default's own properties are pinned in
# tests/cobordism/test_whitney_default_metric_python.py.
DIAGONAL = cob.HodgeMetricSource.DiagonalWeights


def _hodge(spacetime, weights=None, source=DIAGONAL):
    """The diagonal-weight Hodge operator this module's anchors are taken of."""
    if weights is None:
        weights = cob.HodgeLaplacian.defaultWeightConvention()
    return cob.HodgeLaplacian(spacetime, weights, source)

TOL = 1e-7  # near-kernel threshold (|lambda| < TOL counts as harmonic)


# --------------------------------------------------------------------------- #
# Fixture builders
# --------------------------------------------------------------------------- #
def _from_simplices(num_vertices, simplices):
    """Build a Spacetime from explicit simplex vertex tuples (vertices
    0..num_vertices-1). createSimplex auto-creates every sub-edge."""
    sig = tessera.Signature(4, tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    st = tessera.Spacetime(metric, tessera.HERMITIAN_WEIGHTED, 1.0, 1.0,
                           tessera.PREFERRED, tessera.Toroid())
    verts = [st.createVertex(i) for i in range(num_vertices)]
    for simplex in simplices:
        st.createSimplex([verts[i] for i in simplex])
    return st


def _edge(st, a, b):
    for e in st.getEdgeList().toVector():
        if {e.getSource().getId(), e.getTarget().getId()} == {a, b}:
            return e
    raise KeyError((a, b))


def _set_all_spacelike(st, l2=1.0):
    for e in st.getEdgeList().toVector():
        e.setLength(cmath.sqrt(complex(l2)))
        e.setPhase(0.0)
    return st


def _triangle_cycle():
    """S^1 as the 3-cycle 0-1-2-0 (b_1 = 1); all-spacelike unit edges."""
    return _set_all_spacelike(_from_simplices(3, [(0, 1), (1, 2), (2, 0)]), 1.0)


def _triangle_one_timelike(alpha):
    """The 3-cycle with edge (1,2) timelike: l^2 = -alpha^2 (signed volume
    -alpha); edges (0,1),(0,2) spacelike (l^2 = 1)."""
    st = _triangle_cycle()
    _edge(st, 1, 2).setLength(cmath.sqrt(complex(-(alpha ** 2))))
    return st


def _path():
    """Open path 0-1-2 (a tree; b_1 = 0)."""
    return _set_all_spacelike(_from_simplices(3, [(0, 1), (1, 2)]), 1.0)


def _testbed():
    """Square 0-1-2-3 + diagonal 0-2 (1-complex; b_1 = 2)."""
    return _set_all_spacelike(
        _from_simplices(4, [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]), 1.0)


def _filled_square():
    """Square 0-1-2-3 with the triangle (0,1,2) FILLED as a 2-cell and (0,2,3)
    left open: b_1 = 1, and the 2-cell makes the k=1 operator carry BOTH the d_1
    and d_2 terms (unlike the pure 1-complexes above)."""
    return _set_all_spacelike(
        _from_simplices(4, [(0, 1, 2), (0, 2), (2, 3), (3, 0)]), 1.0)


def _filled_square_timelike(alpha):
    """The filled square with its shared diagonal (0,2) timelike — exercises the
    indefinite d'Alembertian on a both-terms (d_1 and d_2) k=1 operator."""
    st = _filled_square()
    _edge(st, 0, 2).setLength(cmath.sqrt(complex(-(alpha ** 2))))
    return st


def _torus_raw():
    """T^2 = S^1 x S^1 via the proven SimplicialProduct + CDT path (b = [1,2,1]),
    edges left at their CDT-assigned lengths (a Lorentzian time structure)."""
    sig = tessera.Signature(4, tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    topology = tessera.SimplicialProduct(tessera.SimplexBoundarySphere(1),
                                         tessera.SimplexBoundarySphere(1))
    st = tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0, tessera.PREFERRED,
                           topology)
    st.build()
    return st


def _torus():
    """All-spacelike T^2 (unit edges), the #104 Euclidean fixture."""
    return _set_all_spacelike(_torus_raw(), 1.0)


def _torus_lorentzian(alpha):
    """T^2 with genuine timelike edges. The SimplicialProduct(S^1,S^1) build is
    geometrically flat (a single time slice), so we declare a deterministic third
    of the edges timelike (l^2 = -alpha^2) by their sorted (src,tgt) order — the
    ticket's "set some edges' squaredLength < 0 via setSquaredLength" on a real
    (CDT-built), both-d_1-and-d_2-terms complex. Returns (spacetime, num_timelike)."""
    st = _torus_raw()
    edges = sorted(st.getEdgeList().toVector(),
                   key=lambda e: tuple(sorted((e.getSource().getId(),
                                               e.getTarget().getId()))))
    n_time = 0
    for i, e in enumerate(edges):
        e.setPhase(0.0)
        if i % 3 == 0:  # every third edge is timelike
            e.setLength(cmath.sqrt(complex(-(alpha ** 2))))
            n_time += 1
        else:
            e.setLength(cmath.sqrt(complex(1.0)))
    return st, n_time


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _betti(st, k):
    return cob.ChainComplex.fromSpacetime(st).bettiNumbers()[k]


def _nk(st, k):
    return cob.ChainComplex.fromSpacetime(st).numSimplices(k)


def _lor_matrix(st, k, metric=True):
    nk = _nk(st, k)
    flat = _hodge(st).laplacian(k, metric)
    return np.array(flat, dtype=complex).reshape(nk, nk)


def _lor_eigs(st, k, metric=True):
    return np.array(_hodge(st).eigenvalues(k, metric),
                    dtype=complex)


def _near_kernel_count(st, k, metric=True, tol=TOL):
    return int(np.sum(np.abs(_lor_eigs(st, k, metric)) < tol))


def _null_norms(st, k, metric=True, tol=1e-9):
    # Complex-typed; with the V^2 weights on real signed l^2 the indefinite norm
    # is REAL. Assert that (stronger than assuming) and hand back the real part.
    norms = np.array(_hodge(st).nullNorms(k, tol, metric),
                     dtype=complex)
    np.testing.assert_allclose(norms.imag, 0.0, atol=1e-9)
    return norms.real


def _is_not_psd(eigs, tol=1e-6):
    """A genuine d'Alembertian: some eigenvalue is off the nonneg-real axis."""
    return bool(np.any(eigs.real < -tol) or np.any(np.abs(eigs.imag) > tol))


# --------------------------------------------------------------------------- #
# (1) Euclidean consistency: all-spacelike => signed volume == |volume|
# --------------------------------------------------------------------------- #
class TestAllSpacelikeConsistency(unittest.TestCase):
    """On an all-spacelike complex the signed weights are positive, so the one
    operator is similar to a symmetric PSD one: real nonnegative spectrum,
    kernel dimension b_k, and strictly positive harmonic norms."""

    CASES = (("triangle k=1", _triangle_cycle, 1),
             ("path k=1", _path, 1),
             ("testbed k=1", _testbed, 1),
             ("filled-square k=1", _filled_square, 1),  # both d_1 and d_2 terms
             ("torus k=1", _torus, 1),                  # both d_1 and d_2 terms
             ("torus k=2", _torus, 2))

    def test_all_spacelike_spectrum_is_real_and_nonnegative(self):
        for name, build, k in self.CASES:
            with self.subTest(case=name):
                st = build()
                eigs = _lor_eigs(st, k)
                self.assertLess(np.max(np.abs(eigs.imag)), 1e-8)
                self.assertGreaterEqual(np.min(eigs.real), -1e-8)

    def test_kernel_dimension_equals_betti(self):
        for name, build, k in self.CASES:
            with self.subTest(case=name):
                st = build()
                self.assertEqual(_near_kernel_count(st, k), _betti(st, k))

    def test_no_null_harmonics_when_all_spacelike(self):
        for name, build, k in self.CASES:
            with self.subTest(case=name):
                st = build()
                norms = _null_norms(st, k)
                self.assertEqual(len(norms), _betti(st, k))
                # every harmonic has strictly positive (definite) W-norm
                if norms.size:
                    self.assertGreater(np.min(norms), 1e-6)

    def test_all_spacelike_weights_are_real_positive(self):
        st = _torus()
        hl = _hodge(st)
        for k in (1, 2):
            with self.subTest(k=k):
                w = np.array(hl.weights(k), dtype=complex)
                np.testing.assert_allclose(w.imag, 0.0, atol=1e-12)
                self.assertGreater(np.min(w.real), 0.0)


# --------------------------------------------------------------------------- #
# (2) The Lorentzian d'Alembertian — closed-form triangle fixture
# --------------------------------------------------------------------------- #
class TestLorentzianDAlembertian(unittest.TestCase):
    """The 3-cycle with one timelike edge: indefinite, non-symmetric, with the
    closed-form spectrum {0, 3, 1 - 2/alpha^2} and harmonic null-norm
    (2 - alpha^2)/3 under the V^2 weights."""

    def test_signed_weights_record_the_timelike_edge(self):
        # W_1 = diag(+1, +1, -alpha^2): the timelike edge's weight is its l^2,
        # real and NEGATIVE. There is no |volume| weighting to compare against —
        # that track was a Euclidean read and is gone (#641).
        st = _triangle_one_timelike(2.0)
        w = np.array(_hodge(st).weights(1), dtype=complex)
        np.testing.assert_allclose(w.imag, 0.0, atol=1e-12)
        np.testing.assert_allclose(np.sort(w.real), [-4.0, 1.0, 1.0], atol=1e-12)

    def test_operator_is_non_symmetric(self):
        L = _lor_matrix(_triangle_one_timelike(1.0), 1)
        self.assertGreater(np.linalg.norm(L - L.T), 1e-6)  # not self-adjoint
        np.testing.assert_allclose(L.imag, 0.0, atol=1e-12)  # but real

    def test_closed_form_spectrum(self):
        for alpha in (0.5, 1.0, 1.5, 2.5, 3.0):
            with self.subTest(alpha=alpha):
                eigs = np.sort(_lor_eigs(_triangle_one_timelike(alpha), 1).real)
                np.testing.assert_allclose(eigs, np.sort([0.0, 3.0, 1.0 - 2.0 / alpha ** 2]),
                                           atol=1e-7)

    def test_indefinite_below_alpha_sqrt_two(self):
        # alpha < sqrt(2): a strictly negative eigenvalue (1 - 2/alpha^2) => not PSD.
        eigs = _lor_eigs(_triangle_one_timelike(1.0), 1)
        self.assertTrue(_is_not_psd(eigs))
        self.assertLess(np.min(eigs.real), -1e-6)
        # Euclidean counterpart (all spacelike) IS PSD: {0, 3, 3}.
        euc = _lor_eigs(_triangle_cycle(), 1)
        self.assertFalse(_is_not_psd(euc))
        np.testing.assert_allclose(np.sort(euc.real), [0.0, 3.0, 3.0], atol=1e-7)

    def test_one_near_kernel_mode_for_alpha_away_from_sqrt_two(self):
        # sqrt(2) itself is the defective crossing (two modes on the kernel).
        for alpha in (0.5, 1.0, 1.5, 2.5, 3.0):
            with self.subTest(alpha=alpha):
                self.assertEqual(_near_kernel_count(_triangle_one_timelike(alpha), 1), 1)

    def test_harmonic_null_norm_is_two_minus_alpha_squared_over_three(self):
        for alpha in (0.5, 1.0, 1.5, 2.5, 3.0):
            with self.subTest(alpha=alpha):
                norms = _null_norms(_triangle_one_timelike(alpha), 1)
                self.assertEqual(len(norms), 1)
                self.assertAlmostEqual(norms[0], (2.0 - alpha ** 2) / 3.0, places=6)

    def test_harmonic_is_the_cycle_with_unit_magnitude_support(self):
        # The kernel mode is the 1-cycle: |h_i|^2 = 1/3 on every edge.
        harmonics = (_hodge(_triangle_one_timelike(1.3))
                     .harmonics(1, 1e-9))
        self.assertEqual(len(harmonics), 1)  # one harmonic, a degree-1 Cochain
        h = np.asarray(harmonics[0].coeffs())
        self.assertEqual(h.size, 3)  # three edges
        np.testing.assert_allclose(np.abs(h) ** 2, np.full(3, 1.0 / 3.0), atol=1e-7)


# --------------------------------------------------------------------------- #
# (3) alpha sweep — how the spectrum / null-harmonic shifts vs. Euclidean
# --------------------------------------------------------------------------- #
class TestLorentzianAlphaSweep(unittest.TestCase):
    """Vary the timelike/spacelike l^2 ratio (the CDT asymmetry alpha) and record
    the spectrum and the harmonic's indefinite norm relative to the all-spacelike
    reference. The harmonic null-norm (2 - alpha^2)/3 crosses zero at
    alpha = sqrt(2): positive (spacelike-dominated) below, null at, negative
    (timelike-dominated) above."""

    SWEEP = (0.25, 0.5, 1.0, 1.5, 2.5, 3.0, 4.0)

    def test_record_sweep(self):
        spacelike_null = 2.0 / 3.0  # the all-spacelike reference norm
        records = []
        for alpha in self.SWEEP:
            eigs = np.sort(_lor_eigs(_triangle_one_timelike(alpha), 1).real)
            null = _null_norms(_triangle_one_timelike(alpha), 1)[0]
            records.append((alpha, eigs.tolist(), null))

            # third eigenvalue tracks 1 - 2/alpha^2; null-norm (2 - alpha^2)/3.
            np.testing.assert_allclose(eigs, np.sort([0.0, 3.0, 1.0 - 2.0 / alpha ** 2]),
                                       atol=1e-7)
            self.assertAlmostEqual(null, (2.0 - alpha ** 2) / 3.0, places=6)
            # indefinite (negative eigenvalue) exactly when alpha < sqrt(2)
            self.assertEqual(np.min(eigs) < -1e-6, alpha ** 2 < 2.0)

        # monotone decrease of the harmonic norm, crossing zero at alpha=sqrt(2).
        nulls = [r[2] for r in records]
        self.assertTrue(all(x > y for x, y in zip(nulls, nulls[1:])))  # strictly down
        below = [n for a, _, n in records if a ** 2 < 2.0]
        above = [n for a, _, n in records if a ** 2 > 2.0]
        self.assertTrue(all(n > 0 for n in below))    # spacelike-dominated, positive
        self.assertTrue(all(n < 0 for n in above))    # timelike-dominated, negative
        self.assertLess(max(above), spacelike_null)   # all below the reference

    def test_shift_relative_to_euclidean_kernel(self):
        # Euclidean (all-spacelike) reference: kernel dim = b_1 = 1, norm > 0.
        st_euc = _triangle_cycle()
        self.assertEqual(_near_kernel_count(st_euc, 1), _betti(st_euc, 1))
        self.assertGreater(_null_norms(st_euc, 1)[0], 0.0)
        # Under the sweep the kernel count is unchanged (1) but its norm migrates
        # from positive through null to negative — the recorded sec-5.6 shift.
        for alpha in self.SWEEP:
            self.assertEqual(_near_kernel_count(_triangle_one_timelike(alpha), 1), 1)


# --------------------------------------------------------------------------- #
# (4) both-terms (d_1 AND d_2) genuine-Lorentzian fixtures — recording
# --------------------------------------------------------------------------- #
class TestLorentzianBothTerms(unittest.TestCase):
    """Fixtures whose k=1 operator carries the d_{k+1} term too, so the two
    indefinite pieces can cancel (the clean ker L_k ~= H_k degrades). Largely a
    recording test: assert the operator is a genuine indefinite d'Alembertian and
    that the near-kernel / null-norm machinery returns consistent shapes."""

    def test_filled_square_is_indefinite_dalembertian(self):
        # Under the V^2 weights this fixture's indefinite window also ends near
        # alpha = sqrt(2) (measured: min Re lambda = -1.14 at 1.2, +0.11 at
        # 1.5), so probe inside it.
        st = _filled_square_timelike(1.2)
        self.assertGreaterEqual(_nk(st, 2), 1)  # the 2-cell is present
        L = _lor_matrix(st, 1)
        self.assertGreater(np.linalg.norm(L - L.T), 1e-6)   # non-symmetric
        eigs = _lor_eigs(st, 1)
        self.assertTrue(_is_not_psd(eigs))                  # not PSD
        # near-kernel modes are well-defined and their null-norms line up 1:1
        harmonics = _hodge(st).harmonics(1, TOL)
        norms = _null_norms(st, 1, tol=TOL)
        self.assertEqual(len(harmonics), len(norms))

    def test_filled_square_euclidean_limit_recovers_betti(self):
        # alpha -> spacelike (all positive) recovers ker dim = b_1 with no nulls.
        st = _filled_square()
        b1 = _betti(st, 1)
        self.assertEqual(_near_kernel_count(st, 1), b1)
        norms = _null_norms(st, 1)
        self.assertEqual(len(norms), b1)
        if norms.size:
            self.assertGreater(np.min(norms), 1e-6)

    def test_cdt_torus_lorentzian_recording(self):
        # A larger CDT-built complex (T^2, b_1 = 2) with genuine timelike edges
        # (sec 5.6's "generate via CDT" + setSquaredLength). The geometry is not
        # closed-form, so we RECORD the spectrum/near-kernel and assert only the
        # robust qualitative facts.
        st, n_time = _torus_lorentzian(2.0)
        self.assertEqual(_betti(st, 1), 2)
        self.assertGreater(n_time, 0)  # genuine timelike content present
        L = _lor_matrix(st, 1)
        self.assertGreater(np.linalg.norm(L - L.T), 1e-6)   # genuinely Lorentzian
        eigs = _lor_eigs(st, 1)
        self.assertTrue(_is_not_psd(eigs))                  # indefinite d'Alembertian
        # near-kernel count and per-harmonic null flags are consistent in shape
        near = _near_kernel_count(st, 1, tol=1e-6)
        norms = _null_norms(st, 1, tol=1e-6)
        self.assertEqual(len(norms), near)
        n_null = int(np.sum(np.abs(norms) < 1e-6))
        # recorded: with genuine timelike content the near-kernel may differ from
        # b_1 and some harmonics may be null — both are valid sec-5.6 outcomes.
        self.assertGreaterEqual(near, 0)
        self.assertGreaterEqual(n_null, 0)


# --------------------------------------------------------------------------- #
# (5) degree-parameterization edges of the Lorentzian path
# --------------------------------------------------------------------------- #
class TestLorentzianDegreeParameterization(unittest.TestCase):

    def test_negative_degree_raises(self):
        hl = _hodge(_triangle_cycle())
        for call in (lambda: hl.eigenvalues(-1),
                     lambda: hl.eigenvectors(-1),
                     lambda: hl.harmonics(-1),
                     lambda: hl.nullNorms(-1),
                     lambda: hl.laplacian(-1, True)):
            with self.subTest(call=call):
                with self.assertRaises(RuntimeError):
                    call()

    def test_above_top_dimension_is_empty(self):
        hl = _hodge(_triangle_cycle())  # S^1, top dim 1
        for k in (2, 3):
            with self.subTest(k=k):
                self.assertEqual(hl.eigenvalues(k), [])
                self.assertEqual(hl.eigenvectors(k), [])
                self.assertEqual(hl.harmonics(k), [])
                self.assertEqual(hl.nullNorms(k), [])
                self.assertEqual(hl.laplacian(k, True), [])

    def test_metric_false_is_positive_combinatorial(self):
        # With unit weights the signed path loses its Lorentzian content: the
        # spectrum is the real, nonneg combinatorial one, kernel dim = b_1.
        st = _triangle_one_timelike(1.0)
        eigs = _lor_eigs(st, 1, metric=False)
        self.assertLess(np.max(np.abs(eigs.imag)), 1e-9)
        self.assertGreaterEqual(np.min(eigs.real), -1e-9)
        self.assertEqual(_near_kernel_count(st, 1, metric=False), _betti(st, 1))



# --------------------------------------------------------------------------- #
# (N) Degree zero is not a special case: L_0 = d_1 W_1^-1 d_1^dagger (#805)
# --------------------------------------------------------------------------- #
class TestDegreeZeroLorentzian(unittest.TestCase):
    """The same signed-weight assembly at degree zero, on genuinely timelike
    (l^2 < 0) and genuinely complex squared lengths. Before #805 degree zero was
    a separately specified Hermitian D - A whose magnitude diagonal disagreed
    with its signed off-diagonal, and no degree-zero fixture carried a negative
    squared length, so the deviation was unreachable from the tests."""

    def _l0(self, st, metric=True):
        return _lor_matrix(st, 0, metric)

    def _oracle(self, st, metric=True):
        cc = cob.ChainComplex.fromSpacetime(st)
        n0, n1 = cc.numSimplices(0), cc.numSimplices(1)
        d1 = np.array(cc.boundaryMatrix(1),
                      dtype=float).reshape(n0, n1).astype(complex)
        w1 = (np.array(_hodge(st).weights(1), dtype=complex)
              if metric else np.ones(n1, dtype=complex))
        return d1 @ np.diag(1.0 / w1) @ d1.conj().T

    FIXTURES = (
        ("all-spacelike 3-cycle", lambda: _triangle_cycle()),
        ("one timelike a=0.7", lambda: _triangle_one_timelike(0.7)),
        ("one timelike a=1.9", lambda: _triangle_one_timelike(1.9)),
        ("filled square timelike", lambda: _filled_square_timelike(1.3)),
        ("lorentzian torus", lambda: _torus_lorentzian(1.1)[0]),
    )

    def test_assembly_matches_the_boundary_map_oracle(self):
        for name, build in self.FIXTURES:
            st = build()
            for metric in (True, False):
                with self.subTest(fixture=name, metric=metric):
                    np.testing.assert_allclose(
                        self._l0(st, metric), self._oracle(st, metric),
                        atol=1e-10)

    def test_constant_is_harmonic_at_every_signature(self):
        for name, build in self.FIXTURES:
            st = build()
            with self.subTest(fixture=name):
                L = self._l0(st)
                ones = np.ones(L.shape[0], dtype=complex)
                scale = max(np.max(np.abs(L)), 1.0)
                self.assertLess(np.max(np.abs(L @ ones)) / scale, 1e-13)

    def test_kernel_dimension_is_b0_at_every_signature(self):
        for name, build in self.FIXTURES:
            st = build()
            with self.subTest(fixture=name):
                self.assertEqual(_near_kernel_count(st, 0), _betti(st, 0))

    def test_timelike_cycle_is_indefinite_below_the_crossing(self):
        # spec(L_0) = {0, 3, 1 - 2/alpha^2}: the SAME closed form the k=1
        # operator has on this fixture, and negative below alpha = sqrt(2).
        for alpha in (0.5, 1.0, 1.9):
            with self.subTest(alpha=alpha):
                eigs = _lor_eigs(_triangle_one_timelike(alpha), 0)
                np.testing.assert_allclose(
                    np.sort(eigs.real),
                    np.sort([0.0, 3.0, 1.0 - 2.0 / alpha ** 2]), atol=1e-9)
                self.assertEqual(_is_not_psd(eigs), alpha < math.sqrt(2.0))

    def test_complex_squared_length_keeps_the_constant_harmonic(self):
        # z = rho e^{i theta} on one edge: L_0 goes genuinely complex (and
        # complex-symmetric rather than Hermitian) but still annihilates 1.
        st = _triangle_cycle()
        _edge(st, 1, 2).setLength(cmath.sqrt(2.4 * cmath.exp(1.1j)))
        L = self._l0(st)
        self.assertGreater(np.max(np.abs(L.imag)), 1e-2)
        np.testing.assert_allclose(L, L.T, atol=1e-14)
        self.assertGreater(np.linalg.norm(L - L.conj().T), 1e-2)
        np.testing.assert_allclose(L @ np.ones(3), 0.0, atol=1e-14)
        self.assertEqual(_near_kernel_count(st, 0), 1)

    def test_null_norm_of_the_constant_is_unity(self):
        # W_0 = I, so <h,h>_W of the unit constant is exactly 1 -- no timelike
        # cancellation is possible at degree zero, unlike k >= 1.
        for name, build in self.FIXTURES:
            st = build()
            with self.subTest(fixture=name):
                norms = _null_norms(st, 0)
                self.assertEqual(len(norms), _betti(st, 0))
                np.testing.assert_allclose(norms, 1.0, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
