# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""Degree-zero operators (#90, reworked #805).

TWO distinct operators live at degree zero and this module pins both:

  * the DERIVED Hodge Laplacian L_0 = d_1 W_1^-1 d_1^dagger W_0 (W_0 = I),
    HodgeLaplacian.laplacian(0) -- assembled from the boundary map and the
    weight exactly as every other degree is. Its row sums vanish identically,
    so the constant 0-cochain is harmonic at ANY weights (positive, signed,
    or complex) and dim ker L_0 = b_0. TestDerivedDegreeZero.

  * the C* CONNECTION graph Laplacian L = D - A,
    HodgeLaplacian.connectionLaplacian() -- off-diagonal
    squaredLength * exp(i*phase) on the stored orientation and its INVERSE on
    the reverse, diagonal sum |squaredLength| (the MAGNITUDE convention). Its
    zero mode IS lifted by a U(1) flux. It is not L_0 and is not of the derived
    form for any weight. Hermitian, PSD by Gershgorin and unitary under
    exp(-iLt) only in the compact case -- real phases on real weights -- since
    a complex phase twists by a similarity rather than a unitary.
    TestAssemblyAndSpectrum / TestHermiticityUnitarity / TestFluxSpectrum /
    TestGaugeInvariance / TestComplexConnectionPhase.

The two coincide exactly at unit real weights and zero phase, which is why the
deviation went unpinned for so long; every deviation test here therefore uses a
genuinely timelike edge (l^2 < 0) or a nonzero phase.

Validated against independent numpy oracles, plus the spec checks that live at
degree 0: Hermiticity / unitary evolution, the Aharonov-Bohm flux spectrum on
the triangle (C4/C5 anchors), gauge invariance (C3, computed test-side), and a
b1 cross-check against ChainComplex.

Fixtures (per the cobordism plan):
  triangle = SimplexBoundarySphere(1)  (S^1 = boundary of a 2-simplex; b1=1)
  path 0-1-2                            (a tree; b1=0)
  testbed  square 00-01-11-10 + diag 00-11  (b1=2; the representative cyclic
                                             fixture, ids 0=00,1=01,2=11,3=10)
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


# --------------------------------------------------------------------------- #
# Fixture builders
# --------------------------------------------------------------------------- #
def _build_topology(topology):
    sig = tessera.Signature(topology.dimension(), tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    st = tessera.Spacetime(metric, tessera.HERMITIAN_WEIGHTED, 1.0, 1.0,
                           tessera.PREFERRED, topology)
    st.build()
    return st


def _from_simplices(num_vertices, simplices):
    """Build a Spacetime directly from explicit simplex vertex tuples (vertices
    0..num_vertices-1), the hand-crafted-complex idiom shared with the cobordism
    verification tests."""
    sig = tessera.Signature(4, tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    st = tessera.Spacetime(metric, tessera.HERMITIAN_WEIGHTED, 1.0, 1.0,
                           tessera.PREFERRED, tessera.Toroid())
    verts = [st.createVertex(i) for i in range(num_vertices)]
    for simplex in simplices:
        st.createSimplex([verts[i] for i in simplex])
    return st


def _set_uniform(st, squared_length=1.0, phase=0.0):
    for e in st.getEdgeList().toVector():
        e.setLength(cmath.sqrt(complex(squared_length)))
        e.setPhase(phase)


def _triangle():
    """S^1 = boundary of a 2-simplex, with unit real edge weights (Φ = 0)."""
    st = _build_topology(tessera.SimplexBoundarySphere(1))
    _set_uniform(st, 1.0, 0.0)
    return st


def _path():
    st = _from_simplices(3, [(0, 1), (1, 2)])
    _set_uniform(st, 1.0, 0.0)
    return st


def _testbed():
    # square 00-01-11-10-00 plus the entangling diagonal 00-11.
    st = _from_simplices(4, [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)])
    _set_uniform(st, 1.0, 0.0)
    return st


def _torus():
    """T^2 = S^1 x S^1 (the qubit, spec sec 5.2): b = [1, 2, 1]. Built via the
    proven SimplicialProduct + CDT path the homology tests use, then given unit
    spacelike edge weights so every cell has a well-defined positive volume."""
    sig = tessera.Signature(4, tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    topology = tessera.SimplicialProduct(tessera.SimplexBoundarySphere(1),
                                         tessera.SimplexBoundarySphere(1))
    st = tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0, tessera.PREFERRED,
                           topology)
    st.build()
    _set_uniform(st, 1.0, 0.0)
    return st


# --------------------------------------------------------------------------- #
# numpy oracle and small helpers
# --------------------------------------------------------------------------- #
def _ordering(st):
    ids = sorted(v.getId() for v in st.getVertexList().toVector())
    return ids, {vid: i for i, vid in enumerate(ids)}


def _np_connection_laplacian(st):
    """Independent D - A reference for the C* CONNECTION operator, built from
    the same edges with the same stable (sorted-id) vertex order it uses.

    The stored orientation carries the link U = exp(i*phase) and the reverse
    carries its INVERSE U**-1, never its conjugate; the geometry keeps the
    conjugate it always had. For a real phase the two conventions agree entry
    for entry, which is why every pre-existing real-phase expectation in this
    module is unchanged."""
    ids, idx = _ordering(st)
    n = len(ids)
    A = np.zeros((n, n), dtype=complex)
    D = np.zeros(n)
    for e in st.getEdgeList().toVector():
        s = e.getSource().getId()
        t = e.getTarget().getId()
        if s == t:
            continue
        i, j = idx[s], idx[t]
        w = (e.getLength() * e.getLength()).real
        phase = e.getPhase()
        A[i, j] += w * np.exp(1j * phase)
        A[j, i] += np.conj(w) * np.exp(-1j * phase)
        D[i] += abs(w)
        D[j] += abs(w)
    L = np.diag(D).astype(complex) - A
    return n, ids, idx, A, D, L


def _edge(st, a, b):
    for e in st.getEdgeList().toVector():
        if {e.getSource().getId(), e.getTarget().getId()} == {a, b}:
            return e
    raise KeyError((a, b))


def _cycle_flux(st, cycle):
    """Directed U(1) holonomy Σ phase around a closed vertex cycle, honoring
    each edge's stored source->target orientation. The COMPACT part only:
    winding is a U(1) quantity, and Im(phase) is the non-compact scale."""
    total = 0.0
    n = len(cycle)
    for k in range(n):
        a, b = cycle[k], cycle[(k + 1) % n]
        e = _edge(st, a, b)
        if e.getSource().getId() == a and e.getTarget().getId() == b:
            total += e.getPhase().real
        else:
            total -= e.getPhase().real
    return total


def _matrix(flat, n):
    return np.array(flat, dtype=complex).reshape(n, n)


def _cluster_ranges(evals, tol=1e-6):
    """Contiguous index ranges of (near-)equal ascending eigenvalues."""
    ranges = []
    start = 0
    for i in range(1, len(evals) + 1):
        if i == len(evals) or abs(evals[i] - evals[start]) > tol:
            ranges.append((start, i))
            start = i
    return ranges


# --------------------------------------------------------------------------- #
# Metric Hodge Laplacian (k >= 1): independent numpy oracle and kernel helpers
# --------------------------------------------------------------------------- #
def _boundary(cc, k):
    """Boundary matrix d_k as a dense (|C_{k-1}| x |C_k|) numpy array."""
    rows, cols = cc.numSimplices(k - 1), cc.numSimplices(k)
    if rows == 0 or cols == 0:
        return np.zeros((rows, cols))
    return np.array(cc.boundaryMatrix(k), dtype=float).reshape(rows, cols)


def _np_metric_laplacian(st, k, metric=True):
    """Independent reconstruction of the signed-weight Hodge Laplacian

        L_k = W_k^-1 d_k^T W_{k-1} d_k + d_{k+1} W_{k+1}^-1 d_{k+1}^T W_k

    from the boundary maps (ChainComplex) and the COMPLEX signed weights W_k
    (HodgeLaplacian.weights, or unit weights when metric is False).

    There is no sqrt(W) anywhere. The symmetric W^{-1/2} form the old oracle
    built needs positive weights, and a Lorentzian cell's d-content is
    imaginary, so that form no longer exists (#641)."""
    cc = cob.ChainComplex.fromSpacetime(st)
    n = cc.dimension()
    hl = _hodge(st)

    def weight(kk):
        if not metric or kk == 0:
            return np.ones(cc.numSimplices(kk), dtype=complex)
        return np.array(hl.weights(kk), dtype=complex)

    nk = cc.numSimplices(k)
    L = np.zeros((nk, nk), dtype=complex)
    if nk == 0:
        return L
    inv_wk = 1.0 / weight(k)
    if k >= 1 and cc.numSimplices(k - 1) > 0:   # W_k^-1 d_k^T W_{k-1} d_k
        dk = _boundary(cc, k).astype(complex)
        L += np.diag(inv_wk) @ dk.T @ np.diag(weight(k - 1)) @ dk
    if k + 1 <= n and cc.numSimplices(k + 1) > 0:  # d_{k+1} W_{k+1}^-1 d_{k+1}^T W_k
        dkp1 = _boundary(cc, k + 1).astype(complex)
        L += dkp1 @ np.diag(1.0 / weight(k + 1)) @ dkp1.T @ np.diag(weight(k))
    return L


def _betti(st, k):
    return cob.ChainComplex.fromSpacetime(st).bettiNumbers()[k]


def _real_spectrum(evals):
    """The U(1) connection operator D - A is Hermitian, so its spectrum is REAL.

    `connectionEigenvalues()` is complex-typed for parity with the L_k family.
    This asserts the imaginary part vanishes rather than discarding it, so a
    connection spectrum that stopped being real fails here instead of being
    silently projected."""
    a = np.asarray(evals)
    np.testing.assert_allclose(a.imag, 0.0, atol=1e-12,
                               err_msg="degree-0 spectrum must be real")
    return a.real


def _kernel_dim_from_eigenvalues(st, k, metric=True, tol=1e-7):
    evals = np.array(_hodge(st).eigenvalues(k, metric))
    return int(np.sum(np.abs(evals) < tol))


def _harmonic_dim(st, k, metric=True, tol=1e-9):
    # harmonics() is one Cochain per basis vector of ker L_k, so its length is
    # the harmonic dimension (= b_k) directly.
    return len(_hodge(st).harmonics(k, tol, metric))


# --------------------------------------------------------------------------- #
# Assembly + spectrum vs numpy, and known-answer anchors
# --------------------------------------------------------------------------- #
class TestAssemblyAndSpectrum(unittest.TestCase):

    def _check_against_numpy(self, st):
        n, _ids, _idx, A, D, L = _np_connection_laplacian(st)
        hl = _hodge(st)
        np.testing.assert_allclose(_matrix(hl.adjacency(), n), A, atol=1e-12)
        np.testing.assert_allclose(np.array(hl.degree()), D, atol=1e-12)
        np.testing.assert_allclose(_matrix(hl.connectionLaplacian(), n), L,
                                   atol=1e-12)
        np.testing.assert_allclose(
            _real_spectrum(hl.connectionEigenvalues()),
            np.linalg.eigvalsh(L), atol=1e-12)
        return n

    def test_triangle_is_three_vertices_three_edges(self):
        st = _triangle()
        self.assertEqual(st.getVertexCount(), 3)
        self.assertEqual(len(st.getEdgeList().toVector()), 3)

    def test_triangle_spectrum_matches_numpy(self):
        self.assertEqual(self._check_against_numpy(_triangle()), 3)

    def test_path_spectrum_matches_numpy(self):
        self._check_against_numpy(_path())

    def test_testbed_spectrum_matches_numpy(self):
        self.assertEqual(self._check_against_numpy(_testbed()), 4)

    def test_random_weighted_spectra_match_numpy(self):
        # Generic complex Hermitian weights on each fixture: the operator must
        # still equal the independent numpy D - A bit for bit (to tolerance).
        rng = np.random.default_rng(20240601)
        for name, st in (("triangle", _triangle()),
                         ("path", _path()),
                         ("testbed", _testbed())):
            with self.subTest(fixture=name):
                for e in st.getEdgeList().toVector():
                    e.setLength(cmath.sqrt(complex(float(rng.uniform(0.5, 2.0)))))
                    e.setPhase(float(rng.uniform(-math.pi, math.pi)))
                self._check_against_numpy(st)

    def test_triangle_zero_phase_known_eigenvalues(self):
        # Equal-weight S^1 with no flux: L = 2I - A(K3) -> {0, 3, 3}.
        hl = _hodge(_triangle())
        np.testing.assert_allclose(sorted(_real_spectrum(hl.eigenvalues())), [0.0, 3.0, 3.0],
                                   atol=1e-12)

    def test_path_known_eigenvalues(self):
        # Open path 0-1-2 Laplacian -> {0, 1, 3}.
        hl = _hodge(_path())
        np.testing.assert_allclose(sorted(_real_spectrum(hl.eigenvalues())), [0.0, 1.0, 3.0],
                                   atol=1e-12)

    def test_complex_weight_round_trips_through_pybind(self):
        # phase = pi/2 on a unit edge -> that adjacency entry is purely +i (and
        # its mirror -i), confirming complex values cross the binding intact.
        st = _path()
        _ids, idx = _ordering(st)
        e = _edge(st, 0, 1)
        e.setLength(cmath.sqrt(complex(1.0)))
        e.setPhase(math.pi / 2.0)
        s, t = e.getSource().getId(), e.getTarget().getId()
        A = _matrix(_hodge(st).adjacency(), st.getVertexCount())
        self.assertAlmostEqual(A[idx[s], idx[t]], 1j, places=12)
        self.assertAlmostEqual(A[idx[t], idx[s]], -1j, places=12)


# --------------------------------------------------------------------------- #
# Hermiticity and unitary time evolution
# --------------------------------------------------------------------------- #
class TestHermiticityUnitarity(unittest.TestCase):

    def _fixtures_with_random_weights(self):
        rng = np.random.default_rng(7)
        out = []
        for name, st in (("triangle", _triangle()), ("testbed", _testbed())):
            for e in st.getEdgeList().toVector():
                e.setLength(cmath.sqrt(complex(float(rng.uniform(0.5, 2.0)))))
                e.setPhase(float(rng.uniform(-math.pi, math.pi)))
            out.append((name, st))
        return out

    def test_laplacian_is_hermitian(self):
        for name, st in self._fixtures_with_random_weights():
            with self.subTest(fixture=name):
                n = st.getVertexCount()
                L = _matrix(_hodge(st).connectionLaplacian(), n)
                self.assertLess(np.linalg.norm(L - L.conj().T), 1e-12)
                self.assertTrue(_hodge(st).isHermitian(1e-12))

    def test_time_evolution_is_unitary(self):
        for name, st in self._fixtures_with_random_weights():
            with self.subTest(fixture=name):
                hl = _hodge(st)
                self.assertLess(hl.unitarityResidual(), 1e-12)
                self.assertLess(hl.unitarityResidual(2.5), 1e-12)


# --------------------------------------------------------------------------- #
# Flux in the spectrum (Aharonov-Bohm ring) — C4/C5 anchors
# --------------------------------------------------------------------------- #
class TestFluxSpectrum(unittest.TestCase):

    @staticmethod
    def _ring_eigs(phi):
        return sorted(2.0 - 2.0 * math.cos((phi + 2.0 * math.pi * k) / 3.0)
                      for k in range(3))

    def _triangle_with_flux(self, phi):
        # The spectrum depends only on the gauge-invariant total flux, so place
        # all of Φ on a single edge; degree is phase-independent (= 2 each).
        st = _triangle()
        st.getEdgeList().toVector()[0].setPhase(phi)
        return st

    def test_flux_spectrum_matches_ring_formula(self):
        for phi in (0.0, math.pi / 3, math.pi / 2, 2 * math.pi / 3, math.pi, 1.234):
            with self.subTest(phi=phi):
                hl = _hodge(self._triangle_with_flux(phi))
                np.testing.assert_allclose(
                    sorted(_real_spectrum(hl.connectionEigenvalues())),
                    self._ring_eigs(phi), atol=1e-12)

    def test_half_flux_quantum_gives_degenerate_pair(self):
        # Φ = π -> {1, 1, 4}; the spectral gap λ1 - λ0 collapses to 0.
        hl = _hodge(self._triangle_with_flux(math.pi))
        eigs = sorted(_real_spectrum(hl.connectionEigenvalues()))
        np.testing.assert_allclose(eigs, [1.0, 1.0, 4.0], atol=1e-12)
        self.assertAlmostEqual(eigs[1] - eigs[0], 0.0, places=12)

    def test_flux_lifts_the_connection_zero_mode(self):
        # No flux: one harmonic (the constant 0-cochain, b0 = 1). Any flux lifts
        # it, so the CONNECTION operator's harmonic dimension drops to 0.
        hl0 = _hodge(_triangle())
        self.assertEqual(len(hl0.connectionHarmonics()), 1)
        hlpi = _hodge(self._triangle_with_flux(math.pi))
        self.assertEqual(len(hlpi.connectionHarmonics()), 0)

    def test_flux_never_lifts_the_derived_zero_mode(self):
        # The contrast: L_0 = d_1 W_1^-1 d_1^dagger has no link phase at all,
        # so its kernel is b_0 = 1 at every flux -- the constant is annihilated
        # to machine precision.
        for phi in (0.0, math.pi / 3, math.pi, 1.234):
            with self.subTest(phi=phi):
                st = self._triangle_with_flux(phi)
                hl = _hodge(st)
                self.assertEqual(len(hl.harmonics(0)), 1)
                L = np.array(hl.laplacian(0)).reshape(3, 3)
                np.testing.assert_allclose(L @ np.ones(3), 0.0, atol=1e-15)

    def test_zero_mode_is_uniform(self):
        # The Φ=0 harmonic is the uniform vector (equal magnitudes on every vertex).
        n = 3
        harmonics = _hodge(_triangle()).connectionHarmonics()
        self.assertEqual(len(harmonics), 1)  # one harmonic, a degree-0 Cochain
        h = harmonics[0]
        self.assertEqual(h.degree(), 0)
        vec = np.asarray(h.coeffs())
        self.assertEqual(vec.size, n)  # length N over the vertex ordering
        np.testing.assert_allclose(np.abs(vec), np.full(n, abs(vec[0])), atol=1e-9)


# --------------------------------------------------------------------------- #
# C3 — gauge invariance (computed test-side; the operator has no gauge() method)
# --------------------------------------------------------------------------- #
class TestGaugeInvariance(unittest.TestCase):

    @staticmethod
    def _apply_gauge(st, alpha):
        # Rephase every edge: θ -> θ + α_src - α_tgt  (A -> G A G^†, L -> G L G^†).
        for e in st.getEdgeList().toVector():
            s, t = e.getSource().getId(), e.getTarget().getId()
            e.setPhase(e.getPhase() + alpha[s] - alpha[t])

    def test_testbed_spectrum_eigenvectors_and_flux_are_gauge_invariant(self):
        rng = np.random.default_rng(2718)
        st = _testbed()
        # Random Hermitian weights + base phases (a generic point in the b1=2
        # connection space).
        for e in st.getEdgeList().toVector():
            e.setLength(cmath.sqrt(complex(float(rng.uniform(0.5, 2.0)))))
            e.setPhase(float(rng.uniform(-math.pi, math.pi)))

        ids, idx = _ordering(st)
        n = len(ids)
        hl_old = _hodge(st)
        evals_old = _real_spectrum(hl_old.connectionEigenvalues())
        V_old = _matrix(hl_old.connectionEigenvectors(), n)

        # Two independent cycles of the testbed (b1 = 2).
        cycles = ([0, 1, 2], [0, 2, 3])
        flux_old = [_cycle_flux(st, c) for c in cycles]

        # Random vertex-phase gauge.
        alpha = {vid: float(rng.uniform(-math.pi, math.pi)) for vid in ids}
        self._apply_gauge(st, alpha)

        hl_new = _hodge(st)
        evals_new = _real_spectrum(hl_new.connectionEigenvalues())
        V_new = _matrix(hl_new.connectionEigenvectors(), n)

        # (i) spectrum unchanged
        np.testing.assert_allclose(evals_new, evals_old, atol=1e-12)

        # (ii) eigenvectors rephased v -> G v: the spectral projector of every
        # eigenvalue cluster transforms as P -> G P G^† (robust to the
        # eigenvector phase/degeneracy ambiguity).
        g = np.array([np.exp(1j * alpha[vid]) for vid in ids])
        G = np.diag(g)
        for (a, b) in _cluster_ranges(evals_old):
            p_old = V_old[:, a:b] @ V_old[:, a:b].conj().T
            p_new = V_new[:, a:b] @ V_new[:, a:b].conj().T
            np.testing.assert_allclose(p_new, G @ p_old @ G.conj().T, atol=1e-9)

        # (iii) every cycle flux unchanged
        flux_new = [_cycle_flux(st, c) for c in cycles]
        np.testing.assert_allclose(flux_new, flux_old, atol=1e-10)

    def test_tree_eigenvectors_rephase_vectorwise(self):
        # On the tree (non-degenerate {0,1,3}) the per-eigenvector statement
        # v_k -> G v_k holds directly (up to a global phase).
        rng = np.random.default_rng(99)
        st = _path()
        ids, idx = _ordering(st)
        n = len(ids)
        hl_old = _hodge(st)
        V_old = _matrix(hl_old.connectionEigenvectors(), n)

        alpha = {vid: float(rng.uniform(-math.pi, math.pi)) for vid in ids}
        TestGaugeInvariance._apply_gauge(st, alpha)
        hl_new = _hodge(st)
        np.testing.assert_allclose(
            _real_spectrum(hl_new.connectionEigenvalues()),
            _real_spectrum(hl_old.connectionEigenvalues()), atol=1e-12)
        V_new = _matrix(hl_new.connectionEigenvectors(), n)

        g = np.array([np.exp(1j * alpha[vid]) for vid in ids])
        for k in range(n):
            gv = g * V_old[:, k]
            overlap = abs(np.vdot(gv, V_new[:, k]))  # parallel <=> |overlap| = 1
            self.assertAlmostEqual(overlap, 1.0, places=8)


# --------------------------------------------------------------------------- #
# b1 cross-check against the topological oracle, and degree parameterization
# --------------------------------------------------------------------------- #
class TestBettiCrossCheck(unittest.TestCase):

    def test_first_betti_numbers(self):
        for name, st, expected in (("triangle", _triangle(), 1),
                                   ("path", _path(), 0),
                                   ("testbed", _testbed(), 2)):
            with self.subTest(fixture=name):
                cc = cob.ChainComplex.fromSpacetime(st)
                self.assertEqual(cc.bettiNumbers()[1], expected)


class TestDegreeParameterization(unittest.TestCase):

    def test_negative_degree_raises(self):
        hl = _hodge(_triangle())
        for call in (lambda: hl.laplacian(-1),
                     lambda: hl.eigenvalues(-1),
                     lambda: hl.eigenvectors(-2),
                     lambda: hl.harmonics(-1)):
            with self.subTest(call=call):
                with self.assertRaises(RuntimeError):
                    call()

    def test_degree_above_top_dimension_is_empty(self):
        # The triangle is S^1 (top dimension 1): there are no 2- or 3-cells, so
        # L_k is the empty operator (no raise) and ker L_k is trivial.
        hl = _hodge(_triangle())
        for k in (2, 3):
            with self.subTest(k=k):
                self.assertEqual(hl.laplacian(k), [])
                self.assertEqual(hl.eigenvalues(k), [])
                self.assertEqual(hl.eigenvectors(k), [])
                self.assertEqual(hl.harmonics(k), [])

    def test_k_zero_is_the_default(self):
        hl = _hodge(_triangle())
        np.testing.assert_allclose(np.array(hl.eigenvalues()),
                                   np.array(hl.eigenvalues(0)), atol=1e-12)

    def test_k_zero_honours_the_metric_flag_like_every_other_degree(self):
        # metric=False selects unit weights at EVERY degree now, degree zero
        # included: with l^2 = 4 on every edge the signed-content L_0 is the
        # combinatorial one scaled by 1/4.
        st = _testbed()
        for e in st.getEdgeList().toVector():
            e.setLength(cmath.sqrt(complex(4.0)))
        hl = _hodge(st)
        n = cob.ChainComplex.fromSpacetime(st).numSimplices(0)
        metric = np.array(hl.laplacian(0, True)).reshape(n, n)
        combinatorial = np.array(hl.laplacian(0, False)).reshape(n, n)
        np.testing.assert_allclose(metric, combinatorial / 4.0, atol=1e-12)
        # Same kernel either way: b_0 = 1.
        self.assertEqual(len(hl.harmonics(0, 1e-9, True)), 1)
        self.assertEqual(len(hl.harmonics(0, 1e-9, False)), 1)


# --------------------------------------------------------------------------- #
# Metric Hodge Laplacian at k >= 1 (#104)
# --------------------------------------------------------------------------- #
class TestMetricHodgeAssembly(unittest.TestCase):
    """The C++ assembly equals the independent numpy oracle, for both the metric
    (volume) and combinatorial (unit) weightings, and its eigenvalues match."""

    CASES = (("triangle", _triangle, 1), ("path", _path, 1),
             ("testbed", _testbed, 1), ("torus k=1", _torus, 1),
             ("torus k=2", _torus, 2))

    def test_assembly_matches_numpy_oracle(self):
        for name, build, k in self.CASES:
            st = build()
            for metric in (True, False):
                with self.subTest(case=name, metric=metric):
                    nk = cob.ChainComplex.fromSpacetime(st).numSimplices(k)
                    L_cpp = np.array(_hodge(st).laplacian(k, metric),
                                     dtype=complex).reshape(nk, nk)
                    L_ref = _np_metric_laplacian(st, k, metric)
                    np.testing.assert_allclose(L_cpp, L_ref, atol=1e-10)

    def test_eigenvalues_match_numpy(self):
        for name, build, k in self.CASES:
            st = build()
            for metric in (True, False):
                with self.subTest(case=name, metric=metric):
                    L_ref = _np_metric_laplacian(st, k, metric)
                    evals = np.array(_hodge(st).eigenvalues(k, metric))
                    # The signed-weight operator is generally NOT self-adjoint, so
                    # eigvals (not eigvalsh) and no PSD claim. Both sides are
                    # sorted by (Re, Im) to compare set-wise.
                    def _key(z):
                        return (round(z.real, 9), round(z.imag, 9))
                    np.testing.assert_allclose(
                        sorted(evals, key=_key),
                        sorted(np.linalg.eigvals(L_ref), key=_key), atol=1e-9)

    def test_assembly_matches_oracle_with_random_weights(self):
        # Non-uniform positive edge weights make every W_k a genuine non-scalar
        # diagonal, exercising the full W^{1/2} formula and the column ordering.
        rng = np.random.default_rng(104)
        for name, build, k in (("testbed", _testbed, 1), ("torus k=1", _torus, 1),
                               ("torus k=2", _torus, 2)):
            st = build()
            for e in st.getEdgeList().toVector():
                e.setLength(cmath.sqrt(complex(float(rng.uniform(0.3, 3.0)))))
                e.setPhase(0.0)
            with self.subTest(case=name):
                nk = cob.ChainComplex.fromSpacetime(st).numSimplices(k)
                L_cpp = np.array(_hodge(st).laplacian(k, True),
                                 dtype=complex).reshape(nk, nk)
                np.testing.assert_allclose(L_cpp, _np_metric_laplacian(st, k, True),
                                           atol=1e-10)


class TestMetricHodgeKernel(unittest.TestCase):
    """The discrete Hodge theorem: dim ker L_k = b_k, for metric and
    combinatorial weights alike, cross-checked against ChainComplex."""

    def test_first_betti_is_first_harmonic_dimension(self):
        for name, build in (("triangle", _triangle), ("path", _path),
                            ("testbed", _testbed), ("torus", _torus)):
            st = build()
            b1 = _betti(st, 1)
            with self.subTest(fixture=name):
                for metric in (True, False):
                    self.assertEqual(_kernel_dim_from_eigenvalues(st, 1, metric), b1)
                    self.assertEqual(_harmonic_dim(st, 1, metric), b1)

    def test_torus_first_homology_is_the_qubit(self):
        # T^2: dim ker L_1 == 2 (spec sec 5.2), == b_1 from ChainComplex.
        st = _torus()
        self.assertEqual(_betti(st, 1), 2)
        for metric in (True, False):
            with self.subTest(metric=metric):
                self.assertEqual(_kernel_dim_from_eigenvalues(st, 1, metric), 2)
                self.assertEqual(_harmonic_dim(st, 1, metric), 2)

    def test_higher_degree_betti_on_the_torus(self):
        # b_2 = 1 (the fundamental class): dim ker L_2 == 1.
        st = _torus()
        self.assertEqual(_betti(st, 2), 1)
        for metric in (True, False):
            with self.subTest(metric=metric):
                self.assertEqual(_kernel_dim_from_eigenvalues(st, 2, metric), 1)
                self.assertEqual(_harmonic_dim(st, 2, metric), 1)

    def test_random_metric_weights_preserve_the_kernel_dimension(self):
        # ker L_k = b_k for ANY positive weights (the metric only moves the
        # representatives and eigenvalues, not the kernel dimension).
        rng = np.random.default_rng(2)
        st = _torus()
        for e in st.getEdgeList().toVector():
            e.setLength(cmath.sqrt(complex(float(rng.uniform(0.3, 3.0)))))
            e.setPhase(0.0)
        self.assertEqual(_kernel_dim_from_eigenvalues(st, 1, True), 2)
        self.assertEqual(_harmonic_dim(st, 1, True), 2)


class TestMetricWeights(unittest.TestCase):
    """weights(k) is the per-k-simplex volume diagonal, in ChainComplex order."""

    def test_vertex_weights_are_unit(self):
        st = _testbed()
        np.testing.assert_allclose(np.array(_hodge(st).weights(0)),
                                   np.ones(4), atol=1e-12)

    def test_edge_weights_are_squared_length_in_column_order(self):
        # Distinct edge lengths pin down both the values (the V^2 weight of an
        # edge is exactly l^2) and the canonical sorted-vertex-id column order.
        st = _from_simplices(4, [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)])
        lengths = {(0, 1): 1.0, (0, 2): 4.0, (0, 3): 9.0, (1, 2): 16.0, (2, 3): 25.0}
        for e in st.getEdgeList().toVector():
            key = tuple(sorted((e.getSource().getId(), e.getTarget().getId())))
            e.setLength(cmath.sqrt(complex(lengths[key])))
            e.setPhase(0.0)
        order = sorted(lengths)  # (0,1),(0,2),(0,3),(1,2),(2,3)
        expected = [lengths[t] for t in order]
        np.testing.assert_allclose(np.array(_hodge(st).weights(1)),
                                   expected, atol=1e-12)

    def test_weights_out_of_range_are_empty(self):
        st = _triangle()  # S^1, top dimension 1
        self.assertEqual(_hodge(st).weights(-1), [])
        self.assertEqual(_hodge(st).weights(5), [])



# --------------------------------------------------------------------------- #
# The DERIVED degree-zero Hodge Laplacian L_0 = d_1 W_1^-1 d_1^dagger (#805)
# --------------------------------------------------------------------------- #
def _np_derived_zero_laplacian(st, metric=True):
    """Independent reconstruction of L_0 = d_1 W_1^-1 d_1^dagger W_0 (W_0 = I)
    from ChainComplex's boundary map and HodgeLaplacian.weights(1), with no
    reference to the C++ degree-zero assembly."""
    cc = cob.ChainComplex.fromSpacetime(st)
    n0, n1 = cc.numSimplices(0), cc.numSimplices(1)
    if n0 == 0:
        return np.zeros((0, 0), dtype=complex)
    L = np.zeros((n0, n0), dtype=complex)
    if n1 == 0:
        return L
    d1 = np.array(cc.boundaryMatrix(1), dtype=float).reshape(n0, n1).astype(complex)
    w1 = (np.array(_hodge(st).weights(1), dtype=complex)
          if metric else np.ones(n1, dtype=complex))
    w0 = np.ones(n0, dtype=complex)                     # W_0 = I
    return d1 @ np.diag(1.0 / w1) @ d1.conj().T @ np.diag(w0)


def _timelike_cycle(alpha):
    """The 3-cycle 0-1-2-0 with edge (1,2) genuinely TIMELIKE (l^2 = -alpha^2)
    and the other two spacelike (l^2 = 1). No current degree-zero fixture had a
    negative squared length; the derived and magnitude conventions coincide
    without one."""
    st = _from_simplices(3, [(0, 1), (1, 2), (2, 0)])
    _set_uniform(st, 1.0, 0.0)
    _edge(st, 1, 2).setLength(cmath.sqrt(complex(-(alpha ** 2))))
    return st


def _complex_l2_cycle(rho, theta):
    """The 3-cycle with a genuinely COMPLEX squared length z = rho e^{i theta}
    on edge (1,2) -- the whitepaper's z_e = rho_e e^{i theta_e}, whose phase is
    part of the edge geometry rather than a second link field."""
    st = _from_simplices(3, [(0, 1), (1, 2), (2, 0)])
    _set_uniform(st, 1.0, 0.0)
    _edge(st, 1, 2).setLength(cmath.sqrt(rho * cmath.exp(1j * theta)))
    return st


class TestDerivedDegreeZero(unittest.TestCase):
    """L_0 is assembled from d_1 and the weight, uniformly with every other
    degree. Every check here is run at a genuinely timelike squared length or a
    genuinely complex one, where the derived and magnitude conventions differ."""

    FIXTURES = (
        ("spacelike triangle", lambda: _triangle()),
        ("path", lambda: _path()),
        ("testbed", lambda: _testbed()),
        ("timelike cycle a=1", lambda: _timelike_cycle(1.0)),
        ("timelike cycle a=0.5", lambda: _timelike_cycle(0.5)),
        ("timelike cycle a=2", lambda: _timelike_cycle(2.0)),
        ("complex l2", lambda: _complex_l2_cycle(1.7, 0.9)),
        ("torus", lambda: _torus()),
    )

    def test_equals_the_boundary_map_identity_entrywise(self):
        for name, build in self.FIXTURES:
            st = build()
            n = cob.ChainComplex.fromSpacetime(st).numSimplices(0)
            for metric in (True, False):
                with self.subTest(fixture=name, metric=metric):
                    got = np.array(_hodge(st).laplacian(0, metric),
                                   dtype=complex).reshape(n, n)
                    np.testing.assert_allclose(
                        got, _np_derived_zero_laplacian(st, metric), atol=1e-12)

    def test_constant_is_annihilated_to_machine_precision(self):
        for name, build in self.FIXTURES:
            st = build()
            n = cob.ChainComplex.fromSpacetime(st).numSimplices(0)
            with self.subTest(fixture=name):
                L = np.array(_hodge(st).laplacian(0),
                             dtype=complex).reshape(n, n)
                ones = np.ones(n, dtype=complex)
                # Row sums vanish identically: |L @ 1| is at rounding level
                # relative to the operator scale, not merely "small".
                scale = max(np.max(np.abs(L)), 1.0)
                self.assertLess(np.max(np.abs(L @ ones)) / scale, 1e-14)

    def test_harmonic_dimension_is_the_component_count(self):
        # dim ker L_0 = b_0, at signed and complex weights alike.
        cases = (
            ("one component", _timelike_cycle(0.5), 1),
            ("complex l2", _complex_l2_cycle(2.0, -1.1), 1),
            ("two components", _two_timelike_components(), 2),
        )
        for name, st, expected in cases:
            with self.subTest(fixture=name):
                b0 = cob.ChainComplex.fromSpacetime(st).bettiNumbers()[0]
                self.assertEqual(b0, expected)
                self.assertEqual(len(_hodge(st).harmonics(0)), b0)

    def test_timelike_cycle_closed_form_spectrum(self):
        # W_1 = diag(1, 1, -alpha^2) on 1-cells (0,1),(0,2),(1,2), so
        # L_0 = d_1 W_1^-1 d_1^T has spec {0, 3, 1 - 2/alpha^2} -- computed by
        # hand, not read off the implementation. NEGATIVE below alpha = sqrt(2):
        # the derived degree-zero operator is genuinely indefinite on a
        # Lorentzian complex.
        for alpha in (0.5, 1.0, math.sqrt(2.0), 2.0, 3.0):
            with self.subTest(alpha=alpha):
                st = _timelike_cycle(alpha)
                evals = np.array(_hodge(st).eigenvalues(0),
                                 dtype=complex)
                np.testing.assert_allclose(np.sort(evals.imag), 0.0, atol=1e-12)
                np.testing.assert_allclose(
                    np.sort(evals.real),
                    np.sort([0.0, 3.0, 1.0 - 2.0 / alpha ** 2]), atol=1e-12)

    def test_lorentzian_degree_zero_is_not_positive_semidefinite(self):
        # The honest regime statement: below the crossing the smallest
        # eigenvalue is strictly negative, so no PSD claim survives here.
        st = _timelike_cycle(1.0)
        evals = np.array(_hodge(st).eigenvalues(0), dtype=complex)
        self.assertLess(np.min(evals.real), -0.5)

    def test_complex_weight_makes_it_complex_symmetric_not_hermitian(self):
        # z = rho e^{i theta} on one edge: L_0 stays SYMMETRIC (the conductance
        # -1/z sits on both off-diagonals) but is no longer Hermitian.
        st = _complex_l2_cycle(1.7, 0.9)
        L = np.array(_hodge(st).laplacian(0),
                     dtype=complex).reshape(3, 3)
        np.testing.assert_allclose(L, L.T, atol=1e-14)
        self.assertGreater(np.linalg.norm(L - L.conj().T), 1e-2)
        # ...and the constant is still exactly harmonic.
        np.testing.assert_allclose(L @ np.ones(3), 0.0, atol=1e-14)

    def test_conductance_is_the_reciprocal_weight(self):
        # The hand identity, entry by entry: off-diagonal -1/W_1(e), diagonal
        # the incident sum. Distinct edge weights pin the column order too.
        st = _from_simplices(3, [(0, 1), (1, 2), (2, 0)])
        squared = {(0, 1): 2.0, (0, 2): -4.0, (1, 2): 5.0}
        for e in st.getEdgeList().toVector():
            key = tuple(sorted((e.getSource().getId(), e.getTarget().getId())))
            e.setLength(cmath.sqrt(complex(squared[key])))
        c = {k: 1.0 / v for k, v in squared.items()}
        expected = np.array([
            [c[(0, 1)] + c[(0, 2)], -c[(0, 1)], -c[(0, 2)]],
            [-c[(0, 1)], c[(0, 1)] + c[(1, 2)], -c[(1, 2)]],
            [-c[(0, 2)], -c[(1, 2)], c[(0, 2)] + c[(1, 2)]]], dtype=complex)
        got = np.array(_hodge(st).laplacian(0),
                       dtype=complex).reshape(3, 3)
        np.testing.assert_allclose(got, expected, atol=1e-13)

    def test_setPhase_does_not_enter_the_derived_operator(self):
        # An edge carries TWO fields: the complex squared length, which alone
        # determines the metric weights, and Edge.setPhase, the independent C*
        # link field the CONNECTION operator reads. L_0 is built from d_1 and
        # W_1 alone, so rephasing the edges cannot move it -- recorded here so
        # a change is noticed.
        st = _timelike_cycle(0.7)
        before = np.array(_hodge(st).laplacian(0), dtype=complex)
        for e in st.getEdgeList().toVector():
            e.setPhase(0.83)
        after = np.array(_hodge(st).laplacian(0), dtype=complex)
        np.testing.assert_allclose(after, before, atol=0.0, rtol=0.0)
        # The connection operator, by contrast, moves.
        self.assertGreater(
            np.linalg.norm(
                np.array(_hodge(st).connectionLaplacian()) -
                np.array(_hodge(_timelike_cycle(0.7))
                         .connectionLaplacian())), 1e-2)

    def test_relabeling_the_vertices_permutes_the_operator(self):
        # Vertex ids are matched by SET, never sorted into a convention: a
        # relabeled complex gives the permuted operator and the same spectrum.
        squared = {(0, 1): 2.0, (0, 2): -4.0, (1, 2): 5.0}
        relabel = {0: 70, 1: 5, 2: 31}

        def graph(idmap):
            sig = tessera.Signature(4, tessera.Lorentzian)
            metric = tessera.Metric(True, sig)
            st = tessera.Spacetime(metric, tessera.HERMITIAN_WEIGHTED, 1.0, 1.0,
                                   tessera.PREFERRED, tessera.Toroid())
            verts = {i: st.createVertex(idmap[i]) for i in range(3)}
            for (a, b) in squared:
                st.createSimplex([verts[a], verts[b]])
            for e in st.getEdgeList().toVector():
                pair = (e.getSource().getId(), e.getTarget().getId())
                inv = {v: k for k, v in idmap.items()}
                key = tuple(sorted((inv[pair[0]], inv[pair[1]])))
                e.setLength(cmath.sqrt(complex(squared[key])))
            return st

        plain = graph({0: 0, 1: 1, 2: 2})
        moved = graph(relabel)
        cc_plain = cob.ChainComplex.fromSpacetime(plain).kSimplexVertices(0)
        cc_moved = cob.ChainComplex.fromSpacetime(moved).kSimplexVertices(0)
        # position in the moved operator of each original vertex
        order = [cc_moved.index([relabel[c[0]]]) for c in cc_plain]
        A = np.array(_hodge(plain).laplacian(0),
                     dtype=complex).reshape(3, 3)
        B = np.array(_hodge(moved).laplacian(0),
                     dtype=complex).reshape(3, 3)
        np.testing.assert_allclose(A, B[np.ix_(order, order)], atol=1e-13)

    def test_null_norms_align_with_the_harmonics(self):
        # nullNorms(0) is one entry per harmonic of the SAME operator; with
        # W_0 = I the constant's norm is exactly its squared length, 1.
        for name, build in self.FIXTURES:
            st = build()
            with self.subTest(fixture=name):
                hl = _hodge(st)
                harmonics = hl.harmonics(0, 1e-9)
                norms = hl.nullNorms(0, 1e-9)
                self.assertEqual(len(norms), len(harmonics))
                for value in norms:
                    self.assertAlmostEqual(complex(value).real, 1.0, places=9)
                    self.assertAlmostEqual(complex(value).imag, 0.0, places=9)

    def test_isolated_vertex_is_not_a_zero_cell(self):
        # ChainComplex is built from simplices, so a bare vertex is not a
        # 0-cell and does not appear in L_0. The CONNECTION operator reads the
        # vertex set directly and does carry it.
        st = _from_simplices(3, [(0, 1)])          # vertex 2 is bare
        hl = _hodge(st)
        self.assertEqual(cob.ChainComplex.fromSpacetime(st).numSimplices(0), 2)
        self.assertEqual(len(hl.laplacian(0)), 4)          # 2x2
        self.assertEqual(len(hl.connectionLaplacian()), 9)  # 3x3


def _two_timelike_components(alpha=0.6):
    """Two disjoint 3-cycles, each carrying one timelike edge: b_0 = 2."""
    st = _from_simplices(6, [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)])
    _set_uniform(st, 1.0, 0.0)
    _edge(st, 1, 2).setLength(cmath.sqrt(complex(-(alpha ** 2))))
    _edge(st, 4, 5).setLength(cmath.sqrt(complex(-(alpha ** 2))))
    return st


def _gauge(st, chi):
    """Apply the gauge transformation g = exp(i*chi) in place.

    U_xy -> g_x**-1 U_xy g_y, i.e. phase -> phase + chi[target] - chi[source]
    on each edge's own stored orientation. `chi` may be complex: the structure
    group is C*, not U(1)."""
    for e in st.getEdgeList().toVector():
        s, t = e.getSource().getId(), e.getTarget().getId()
        e.setPhase(e.getPhase() + chi[t] - chi[s])


class TestComplexConnectionPhase(unittest.TestCase):
    """The connection phase is COMPLEX and its reverse orientation carries the
    INVERSE link, not the conjugate (#804).

    The structure group is C* = U(1) x R+: exp(i*phase) factors into a compact
    winding exp(i*Re) and a non-compact scale exp(-Im). The whole point of the
    inverse convention is that gauge covariance survives the non-compact part,
    where conj(g) != g**-1."""

    @staticmethod
    def _connection(st):
        ids, _ = _ordering(st)
        return _matrix(_hodge(st).connectionLaplacian(), len(ids))

    def test_phase_round_trips_as_a_complex_number(self):
        st = _triangle()
        e = st.getEdgeList().toVector()[0]
        self.assertEqual(e.getPhase(), 0j)          # default, not 0.0
        e.setPhase(0.75)                            # a real value converts
        self.assertEqual(e.getPhase(), complex(0.75, 0.0))
        e.setPhase(complex(0.25, -1.5))
        self.assertEqual(e.getPhase(), complex(0.25, -1.5))

    def test_real_phase_reproduces_the_hermitian_magnetic_operator(self):
        # The compact case on real weights: the previous convention and this
        # one agree entry for entry, and the operator stays Hermitian.
        st = _triangle()
        _set_uniform(st, 1.0, 0.0)
        for k, e in enumerate(st.getEdgeList().toVector()):
            e.setPhase(0.31 * (k + 1))
        L = self._connection(st)
        np.testing.assert_allclose(L, L.conj().T, atol=0.0, rtol=0.0)
        _, _, _, _, _, oracle = _np_connection_laplacian(st)
        np.testing.assert_allclose(L, oracle, atol=1e-14)

    def test_complex_phase_is_not_hermitian(self):
        # A complex phase twists by a similarity rather than a unitary, so the
        # operator leaves the Hermitian class by design.
        st = _triangle()
        _set_uniform(st, 1.0, 0.0)
        for e in st.getEdgeList().toVector():
            e.setPhase(complex(0.3, 0.9))
        L = self._connection(st)
        self.assertGreater(np.abs(L - L.conj().T).max(), 1e-3)

    def test_gauge_transformation_acts_by_an_exact_similarity(self):
        # THE claim the inverse convention exists for. A complex gauge function
        # must act by diag(g)**-1 L diag(g) entrywise, not merely up to a phase.
        st = _testbed()
        rng = np.random.default_rng(804)
        for e in st.getEdgeList().toVector():
            e.setLength(complex(rng.normal(), rng.normal()))   # complex weight
            e.setPhase(complex(rng.normal(), rng.normal()))
        ids, _ = _ordering(st)
        before = self._connection(st)
        chi = {v: complex(rng.normal(), rng.normal()) for v in ids}
        _gauge(st, chi)
        after = self._connection(st)
        g = np.array([np.exp(1j * chi[v]) for v in ids])
        predicted = np.diag(1.0 / g) @ before @ np.diag(g)
        self.assertLess(
            np.abs(predicted - after).max() / np.abs(after).max(), 1e-14)

    def test_complex_gauge_leaves_the_spectrum_invariant(self):
        # The observable consequence of the similarity above.
        st = _testbed()
        rng = np.random.default_rng(29)
        for e in st.getEdgeList().toVector():
            e.setLength(complex(rng.normal(), rng.normal()))
            e.setPhase(complex(rng.normal(), rng.normal()))
        ids, _ = _ordering(st)
        before = np.sort_complex(np.linalg.eigvals(self._connection(st)))
        _gauge(st, {v: complex(rng.normal(), rng.normal()) for v in ids})
        after = np.sort_complex(np.linalg.eigvals(self._connection(st)))
        np.testing.assert_allclose(after, before, atol=1e-12)

    def test_the_conjugate_convention_would_break_complex_gauge_covariance(self):
        # Why the inverse is not a matter of taste. Assemble both conventions
        # over the SAME complex geometry and gauge-transform: the conjugate one
        # moves the spectrum by an O(1) amount.
        st = _testbed()
        rng = np.random.default_rng(77)
        for e in st.getEdgeList().toVector():
            e.setLength(complex(rng.normal(), rng.normal()))
            e.setPhase(complex(rng.normal(), rng.normal()))
        ids, idx = _ordering(st)
        chi = {v: complex(rng.normal(), rng.normal()) for v in ids}
        stored = [(e.getSource().getId(), e.getTarget().getId(),
                   e.getLength() * e.getLength(), e.getPhase())
                  for e in st.getEdgeList().toVector()]

        def assemble(gauged, inverse_link):
            n = len(ids)
            A = np.zeros((n, n), dtype=complex)
            D = np.zeros(n)
            for s, t, w, phase in stored:
                if gauged:
                    phase = phase + chi[t] - chi[s]
                i, j = idx[s], idx[t]
                A[i, j] += w * np.exp(1j * phase)
                A[j, i] += (np.conj(w) * np.exp(-1j * phase) if inverse_link
                            else np.conj(w * np.exp(1j * phase)))
                D[i] += abs(w)
                D[j] += abs(w)
            return np.diag(D).astype(complex) - A

        def drift(inverse_link):
            a = np.sort_complex(np.linalg.eigvals(assemble(False, inverse_link)))
            b = np.sort_complex(np.linalg.eigvals(assemble(True, inverse_link)))
            return np.abs(a - b).max()

        # The shipped convention matches the C++ operator ...
        np.testing.assert_allclose(
            assemble(False, True), self._connection(st), atol=1e-12)
        self.assertLess(drift(inverse_link=True), 1e-12)
        self.assertGreater(drift(inverse_link=False), 1e-2)

    def test_a_pure_gauge_connection_is_isospectral_with_the_untwisted_one(self):
        # A flat (pure-gauge) connection carries no flux, so it is a similarity
        # of the zero-phase operator even when the gauge function is complex.
        st = _testbed()
        _set_uniform(st, 1.0, 0.0)
        flat = np.sort_complex(np.linalg.eigvals(self._connection(st)))
        rng = np.random.default_rng(3)
        ids, _ = _ordering(st)
        _gauge(st, {v: complex(rng.normal(), rng.normal()) for v in ids})
        pure = np.sort_complex(np.linalg.eigvals(self._connection(st)))
        np.testing.assert_allclose(pure, flat, atol=1e-12)

    def test_a_real_flux_still_lifts_the_zero_mode(self):
        # The documented behaviour of the connection operator, preserved: a
        # genuine (non-pure-gauge) U(1) flux moves the zero mode off zero,
        # which ker L_0 = b_0 can never see.
        st = _triangle()
        _set_uniform(st, 1.0, 0.0)
        flat = np.abs(np.linalg.eigvals(self._connection(st))).min()
        self.assertLess(flat, 1e-12)
        _edge(st, 0, 1).setPhase(1.1)
        self.assertAlmostEqual(_cycle_flux(st, [0, 1, 2]) % (2 * math.pi),
                               1.1, places=12)
        self.assertGreater(
            np.abs(np.linalg.eigvals(self._connection(st))).min(), 1e-3)

    def test_the_geometric_laplacian_is_blind_to_the_phase_at_every_degree(self):
        # laplacian(k) is built from the lengths alone. Writing the phase into
        # a metric weight would make the metric gauge-variant and destroy the
        # derived form, so the operator must not move at ANY degree.
        st = _torus()
        rng = np.random.default_rng(13)
        before = {k: np.array(_hodge(st).laplacian(k), dtype=complex)
                  for k in (0, 1, 2)}
        for e in st.getEdgeList().toVector():
            e.setPhase(complex(rng.normal(), rng.normal()))
        for k in (0, 1, 2):
            after = np.array(_hodge(st).laplacian(k), dtype=complex)
            np.testing.assert_allclose(after, before[k], atol=0.0, rtol=0.0)

    def test_the_degree_matrix_is_phase_independent(self):
        # The AB operator's diagonal carries the MAGNITUDE, so no part of the
        # phase -- compact or not -- may reach it.
        st = _testbed()
        _set_uniform(st, 1.0, 0.0)
        before = np.array(_hodge(st).degree())
        for e in st.getEdgeList().toVector():
            e.setPhase(complex(0.9, -0.4))
        np.testing.assert_allclose(
            np.array(_hodge(st).degree()), before,
            atol=0.0, rtol=0.0)


if __name__ == "__main__":
    unittest.main()
