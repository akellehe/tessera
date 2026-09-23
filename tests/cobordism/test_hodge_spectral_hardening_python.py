# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""HodgeLaplacian + spectral-observable hardening (#152).

A cross-fixture hardening suite that strengthens the operator
(`cobordism::HodgeLaplacian`) and its scalar wrappers
(`observables::SpectralGap` / `observables::HarmonicDimension`) against
independent numpy oracles and closed-form / topological invariants. It
complements the per-feature suites (``test_hodge_laplacian*`` and
``test_spectral_observables_python``) by checking the spec's invariants on the
*full* fixture zoo at once:

  * **Gauge invariance** (C3) — a random vertex rephasing ``A ↦ G A G†`` leaves
    the U(1) CONNECTION spectrum and every cycle flux ``Φ_γ`` unchanged, and rephases the
    eigenvectors ``v ↦ G v`` (checked via the spectral projector ``P ↦ G P G†``,
    which is robust to the per-eigenvector phase/degeneracy ambiguity). Verified
    on hand-built graphs with explicit cycle bases and on the CDT-built T²/S²
    where every 2-simplex boundary supplies a cycle.

  * **Discrete Hodge theorem** — ``dim ker L_k == bettiNumbers()[k]`` for every k
    across the whole fixture zoo: a cycle (S¹), a path/tree, a disconnected
    complex (b₀ > 1), T², S², S²×S¹, T³, RP³, and Kühnel's ℂP². Cross-checked
    two ways (eigenvalue near-kernel count and ``harmonics()`` column count) and
    for both the metric (|volume|) and combinatorial (unit) weightings, whose
    operators genuinely differ yet share the kernel dimension.

  * **Hermiticity / unitarity** — ``L = L†`` and ``e^{-iLt}`` unitary under
    generic complex Hermitian weights, for the U(1) CONNECTION operator
    ``connectionLaplacian`` (#805). The derived ``L_0 = d_1 W_1^-1 d_1^†`` is
    complex symmetric, not Hermitian, once a weight is complex; what IS exact
    there is ``L_0 @ 1 = 0``, checked in ``test_hodge_laplacian_python``.

  * **Flux in the spectrum** (U(1) connection operator) — the Aharonov–Bohm ring
    ``λ = 2 − 2cos((Φ + 2πk)/3)``, dependence on the gauge-invariant *total* flux
    only, the half-quantum gap collapse, 2π-periodic restoration of the zero
    mode, and the flux lifting the harmonic at Φ ≠ 0. ``dim ker L_0`` is ``b_0``
    at every flux and is checked separately.

  * **Lorentzian d'Alembertian (§5.6)** — the harmonic null-norm ``(2 − α)/3``
    crossing zero at α = 2 (sign flip + bracketed root), tracked alongside the
    eigenvalue ``1 − 2/α`` whose sign is ``sign(α − 2)``.

  * **Edge cases** — the single vertex, the empty complex, k above the top
    dimension, and negative k.
"""

import math
import unittest

import numpy as np

import tessera
import cmath


def _real_spectrum(evals):
    """The U(1) connection operator D - A is Hermitian, so its spectrum is REAL.

    `connectionEigenvalues()` is complex-typed for parity with the L_k family.
    This ASSERTS the imaginary part vanishes rather than discarding it (#644)."""
    a = np.asarray(evals)
    np.testing.assert_allclose(a.imag, 0.0, atol=1e-12,
                               err_msg="degree-0 spectrum must be real")
    return a.real



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
obs = tessera.observables

PI = math.pi
TOL_KER = 1e-7   # |λ| < TOL_KER counts as a (near-)kernel mode


# --------------------------------------------------------------------------- #
# Fixture builders
# --------------------------------------------------------------------------- #
def _spacetime(topology, st_type=tessera.CDT):
    """Build + materialize a Spacetime over a Topology (the canonical fixture
    path shared with test_fixtures_python / test_three_manifold_fixtures)."""
    sig = tessera.Signature(4, tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    st = tessera.Spacetime(metric, st_type, 1.0, 1.0, tessera.PREFERRED, topology)
    st.build()
    return st


def _from_simplices(num_vertices, simplices):
    """Hand-built complex on vertices 0..num_vertices-1 (createSimplex auto-creates
    every sub-edge); the idiom shared with the cobordism verification tests."""
    sig = tessera.Signature(4, tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    st = tessera.Spacetime(metric, tessera.HERMITIAN_WEIGHTED, 1.0, 1.0,
                           tessera.PREFERRED, tessera.Toroid())
    verts = [st.createVertex(i) for i in range(num_vertices)]
    for s in simplices:
        st.createSimplex([verts[i] for i in s])
    return st


def _set_uniform(st, sq=1.0, phase=0.0):
    for e in st.getEdgeList().toVector():
        e.setLength(cmath.sqrt(complex(sq)))
        e.setPhase(phase)
    return st


# -- hand-built 1-complexes (explicit cycle structure) -- #
def _cycle():
    """S¹ as the 3-cycle 0-1-2-0 (b = [1, 1])."""
    return _set_uniform(_from_simplices(3, [(0, 1), (1, 2), (2, 0)]))


def _path():
    """Open path 0-1-2 (a tree; b = [1, 0])."""
    return _set_uniform(_from_simplices(3, [(0, 1), (1, 2)]))


def _testbed():
    """Square 0-1-2-3 + diagonal 0-2 (b = [1, 2])."""
    return _set_uniform(_from_simplices(4, [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]))


def _complete_k4():
    """K₄, the complete graph on four vertices (b = [1, 3])."""
    return _set_uniform(_from_simplices(
        4, [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]))


def _disconnected():
    """A triangle 0-1-2-0 plus a disjoint edge 3-4: two components with one cycle
    (b = [2, 1]) — the b₀ > 1 control."""
    return _set_uniform(_from_simplices(5, [(0, 1), (1, 2), (2, 0), (3, 4)]))


def _single_vertex():
    """One isolated vertex, no edges (b = [1]); L₀ is the 1×1 zero operator."""
    return _from_simplices(1, [])


def _two_vertex_edge():
    """One edge 0-1 (b = [1, 0]); L₀ = [[1,-1],[-1,1]] with spectrum {0, 2}."""
    return _set_uniform(_from_simplices(2, [(0, 1)]))


# -- closed topological fixtures (built via Topology, then unit-spacelike) -- #
def _circle():
    return tessera.SimplexBoundarySphere(1)


def _t2():
    """T² = S¹ × S¹ (b = [1, 2, 1])."""
    return _set_uniform(_spacetime(
        tessera.SimplicialProduct(_circle(), _circle())))


def _t3():
    """T³ = S¹ × S¹ × S¹ via the staircase product (b = [1, 3, 3, 1])."""
    return _set_uniform(_spacetime(tessera.SimplicialProduct(
        tessera.SimplicialProduct(_circle(), _circle()), _circle())))


def _s2():
    """S² = ∂Δ³ (b = [1, 0, 1]); its 1-skeleton is K₄."""
    return _set_uniform(_spacetime(tessera.SimplexBoundarySphere(2)))


def _s2_cross_s1():
    """S² × S¹ (b = [1, 1, 1, 1])."""
    return _set_uniform(_spacetime(tessera.SphereCircleProduct()))


def _rp3():
    """RP³ = Walkup's 11-vertex triangulation (rational b = [1, 0, 0, 1]; the
    ℤ₂ in H₁ is torsion, invisible to the real Hodge Laplacian)."""
    return _set_uniform(_spacetime(tessera.RealProjectiveSpace()))


def _cp2():
    """ℂP² = Kühnel's minimal 9-vertex triangulation (b = [1, 0, 1, 0, 1])."""
    return _set_uniform(_spacetime(tessera.ComplexProjectivePlane()))


# (name, builder, expected rational Betti vector) — the Hodge-theorem zoo.
FIXTURES = (
    ("cycle S^1", _cycle, [1, 1]),
    ("path/tree", _path, [1, 0]),
    ("disconnected", _disconnected, [2, 1]),
    ("T^2", _t2, [1, 2, 1]),
    ("S^2", _s2, [1, 0, 1]),
    ("S^2xS^1", _s2_cross_s1, [1, 1, 1, 1]),
    ("T^3", _t3, [1, 3, 3, 1]),
    ("RP^3", _rp3, [1, 0, 0, 1]),
    ("CP^2", _cp2, [1, 0, 1, 0, 1]),
)


# --------------------------------------------------------------------------- #
# numpy / spectral helpers
# --------------------------------------------------------------------------- #
def _ordering(st):
    ids = sorted(v.getId() for v in st.getVertexList().toVector())
    return ids, {vid: i for i, vid in enumerate(ids)}


def _matrix(flat, n):
    return np.array(flat, dtype=complex).reshape(n, n)


def _cluster_ranges(evals, tol=1e-6):
    """Contiguous index ranges of (near-)equal ascending eigenvalues."""
    ranges, start = [], 0
    for i in range(1, len(evals) + 1):
        if i == len(evals) or abs(evals[i] - evals[start]) > tol:
            ranges.append((start, i))
            start = i
    return ranges


def _edge(st, a, b):
    for e in st.getEdgeList().toVector():
        if {e.getSource().getId(), e.getTarget().getId()} == {a, b}:
            return e
    raise KeyError((a, b))


def _cycle_flux(st, cycle):
    """Directed holonomy Σ phase around a closed vertex cycle, honoring each
    edge's stored source→target orientation."""
    total, n = 0.0, len(cycle)
    for k in range(n):
        a, b = cycle[k], cycle[(k + 1) % n]
        e = _edge(st, a, b)
        if e.getSource().getId() == a and e.getTarget().getId() == b:
            total += e.getPhase()
        else:
            total -= e.getPhase()
    return total


def _apply_gauge(st, alpha):
    """Rephase every edge θ → θ + α_src − α_tgt, i.e. A → G A G† with
    G = diag(e^{iα})."""
    for e in st.getEdgeList().toVector():
        s, t = e.getSource().getId(), e.getTarget().getId()
        e.setPhase(e.getPhase() + alpha[s] - alpha[t])


def _kernel_dim(hl, k, metric=True, tol=TOL_KER):
    evals = np.array(hl.eigenvalues(k, metric))
    return int(np.sum(np.abs(evals) < tol))


def _harmonic_dim(hl, nk, k, metric=True, tol=TOL_KER):
    if nk == 0:
        return 0
    # harmonics() is one Cochain per ker L_k basis vector => its length is b_k.
    return len(hl.harmonics(k, tol, metric))


# --------------------------------------------------------------------------- #
# (1) Discrete Hodge theorem across the whole fixture zoo
# --------------------------------------------------------------------------- #
class TestHodgeTheoremAllFixtures(unittest.TestCase):
    """dim ker L_k == b_k for every k, on every fixture, for metric and
    combinatorial weights, cross-checked against ChainComplex.bettiNumbers()."""

    def test_kernel_dimension_equals_betti(self):
        for name, build, expected in FIXTURES:
            st = build()
            hl = _hodge(st)
            cc = cob.ChainComplex.fromSpacetime(st)
            betti = cc.bettiNumbers()
            with self.subTest(fixture=name):
                # the fixture is what we think it is (guards against regressions)
                self.assertEqual(betti, expected)
                for k in range(cc.dimension() + 1):
                    nk = cc.numSimplices(k)
                    for metric in (True, False):
                        # eigenvalue near-kernel count == b_k ...
                        self.assertEqual(
                            _kernel_dim(hl, k, metric), betti[k],
                            msg=f"{name} ker L_{k} (metric={metric})")
                        # ... and the harmonics() column count agrees.
                        self.assertEqual(
                            _harmonic_dim(hl, nk, k, metric), betti[k],
                            msg=f"{name} harmonics L_{k} (metric={metric})")

    def test_alternating_kernel_sum_is_euler_characteristic(self):
        # Σ (−1)^k dim ker L_k = Σ (−1)^k b_k = χ (Euler–Poincaré), an independent
        # cross-check tying the analytic kernel back to the combinatorial χ.
        for name, build, _ in FIXTURES:
            st = build()
            hl = _hodge(st)
            cc = cob.ChainComplex.fromSpacetime(st)
            with self.subTest(fixture=name):
                kernel_chi = sum((-1) ** k * _kernel_dim(hl, k, True)
                                 for k in range(cc.dimension() + 1))
                self.assertEqual(kernel_chi, cc.eulerCharacteristic())

    def test_top_dimension_kernel_is_fundamental_class(self):
        # Each closed oriented n-manifold here has b_n = 1: a single harmonic
        # n-cochain (the Poincaré dual of the fundamental class).
        for name, build in (("T^2", _t2), ("S^2", _s2), ("S^2xS^1", _s2_cross_s1),
                            ("T^3", _t3), ("RP^3", _rp3), ("CP^2", _cp2)):
            st = build()
            hl = _hodge(st)
            cc = cob.ChainComplex.fromSpacetime(st)
            with self.subTest(fixture=name):
                self.assertEqual(_kernel_dim(hl, cc.dimension(), True), 1)


# --------------------------------------------------------------------------- #
# (2) Gauge invariance (C3): spec, cycle flux, and eigenvector rephasing
# --------------------------------------------------------------------------- #
class TestGaugeInvariance(unittest.TestCase):

    def _check(self, st, cycles, seed, proj_atol=1e-8):
        """Random Hermitian weights + base phases, then a random vertex-phase
        gauge: assert spec(L₀) and all cycle fluxes are unchanged and the
        eigenvectors rephase (projector P → G P G†)."""
        rng = np.random.default_rng(seed)
        for e in st.getEdgeList().toVector():
            e.setLength(cmath.sqrt(complex(float(rng.uniform(0.5, 2.0)))))
            e.setPhase(float(rng.uniform(-PI, PI)))

        ids, _ = _ordering(st)
        n = len(ids)
        hl_old = _hodge(st)
        evals_old = _real_spectrum(hl_old.connectionEigenvalues())
        V_old = _matrix(hl_old.connectionEigenvectors(), n)
        flux_old = [_cycle_flux(st, c) for c in cycles]

        alpha = {vid: float(rng.uniform(-PI, PI)) for vid in ids}
        _apply_gauge(st, alpha)

        hl_new = _hodge(st)
        evals_new = _real_spectrum(hl_new.connectionEigenvalues())
        V_new = _matrix(hl_new.connectionEigenvectors(), n)

        # (i) spectrum unchanged
        np.testing.assert_allclose(evals_new, evals_old, atol=1e-9)

        # (ii) eigenvectors rephase v → G v, via the cluster spectral projector
        g = np.array([np.exp(1j * alpha[vid]) for vid in ids])
        G = np.diag(g)
        for (a, b) in _cluster_ranges(evals_old):
            p_old = V_old[:, a:b] @ V_old[:, a:b].conj().T
            p_new = V_new[:, a:b] @ V_new[:, a:b].conj().T
            np.testing.assert_allclose(p_new, G @ p_old @ G.conj().T, atol=proj_atol)

        # (iii) every cycle flux unchanged
        if cycles:
            np.testing.assert_allclose([_cycle_flux(st, c) for c in cycles],
                                       flux_old, atol=1e-9)

    def test_graphs_with_explicit_cycles(self):
        cases = (
            ("cycle", _cycle(), [[0, 1, 2]]),
            ("testbed", _testbed(), [[0, 1, 2], [0, 2, 3]]),
            ("K4", _complete_k4(), [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]),
            ("disconnected", _disconnected(), [[0, 1, 2]]),
        )
        for i, (name, st, cycles) in enumerate(cases):
            with self.subTest(fixture=name):
                self._check(st, cycles, seed=1000 + i)

    def test_topological_fixtures_triangle_fluxes(self):
        # Every 2-simplex boundary is a closed 1-cycle; its flux (the discrete
        # field strength through that face) must be gauge invariant, alongside the
        # full spectrum/eigenvector statement on a larger operator.
        for i, (name, build) in enumerate((("T^2", _t2), ("S^2", _s2))):
            st = build()
            st.getExternalSimplices()  # materialize faces so every edge exists
            cc = cob.ChainComplex.fromSpacetime(st)
            cycles = [list(t) for t in cc.kSimplexVertices(2)]
            with self.subTest(fixture=name):
                self.assertGreater(len(cycles), 0)
                self._check(st, cycles, seed=2000 + i, proj_atol=1e-7)

    def test_degenerate_cluster_rephases_as_a_block(self):
        # Φ = π on the triangle gives the degenerate pair {1, 1}; gauge leaves the
        # flux (hence the degeneracy) intact, and the 2-dim eigenspace projector
        # transforms as G P G† — the genuinely degenerate (non-vectorwise) case.
        st = _cycle()
        _edge(st, 0, 1).setPhase(PI)
        ids, _ = _ordering(st)
        n = len(ids)
        hl_old = _hodge(st)
        evals_old = _real_spectrum(hl_old.connectionEigenvalues())
        V_old = _matrix(hl_old.connectionEigenvectors(), n)
        # the degenerate low pair really is present
        self.assertAlmostEqual(evals_old[1] - evals_old[0], 0.0, places=9)

        rng = np.random.default_rng(7)
        alpha = {vid: float(rng.uniform(-PI, PI)) for vid in ids}
        _apply_gauge(st, alpha)
        hl_new = _hodge(st)
        np.testing.assert_allclose(
            _real_spectrum(hl_new.connectionEigenvalues()), evals_old,
            atol=1e-12)
        V_new = _matrix(hl_new.connectionEigenvectors(), n)
        G = np.diag([np.exp(1j * alpha[vid]) for vid in ids])
        p_old = V_old[:, 0:2] @ V_old[:, 0:2].conj().T
        p_new = V_new[:, 0:2] @ V_new[:, 0:2].conj().T
        np.testing.assert_allclose(p_new, G @ p_old @ G.conj().T, atol=1e-9)


# --------------------------------------------------------------------------- #
# (3) Hermiticity / unitarity (k=0); metric vs combinatorial agree on the kernel
# --------------------------------------------------------------------------- #
class TestHermiticityUnitarity(unittest.TestCase):

    def test_hermitian_and_unitary_under_random_weights(self):
        rng = np.random.default_rng(31)
        for name, build in (("triangle", _cycle), ("testbed", _testbed),
                            ("K4", _complete_k4), ("S^2", _s2)):
            st = build()
            for e in st.getEdgeList().toVector():
                e.setLength(cmath.sqrt(complex(float(rng.uniform(0.5, 2.0)))))
                e.setPhase(float(rng.uniform(-PI, PI)))
            with self.subTest(fixture=name):
                hl = _hodge(st)
                n = st.getVertexCount()
                L = _matrix(hl.connectionLaplacian(), n)
                self.assertLess(np.linalg.norm(L - L.conj().T), 1e-12)
                self.assertTrue(hl.isHermitian(1e-12))
                self.assertLess(hl.unitarityResidual(1.0), 1e-12)
                self.assertLess(hl.unitarityResidual(2.5), 1e-12)


class TestMetricVsCombinatorial(unittest.TestCase):
    """The metric (|volume|) and combinatorial (unit) L_k are genuinely different
    operators that nonetheless share the kernel dimension b_k (the Hodge theorem
    holds for any positive weights)."""

    def test_operators_differ_but_kernels_agree(self):
        for name, build, betti in FIXTURES:
            st = build()
            hl = _hodge(st)
            cc = cob.ChainComplex.fromSpacetime(st)
            with self.subTest(fixture=name):
                for k in range(1, cc.dimension() + 1):
                    nk = cc.numSimplices(k)
                    self.assertEqual(_kernel_dim(hl, k, True),
                                     _kernel_dim(hl, k, False))
                    self.assertEqual(_kernel_dim(hl, k, True), betti[k])
                    # With unit edges the (k+1)-cells (k ≥ 1) have non-unit
                    # regular-simplex volume, so the metric reweights the
                    # d_{k+1} term: the two operators must actually differ.
                    if cc.numSimplices(k + 1) > 0:
                        Lm = _matrix(hl.laplacian(k, True), nk)
                        Lc = _matrix(hl.laplacian(k, False), nk)
                        self.assertGreater(np.linalg.norm(Lm - Lc), 1e-6)


# --------------------------------------------------------------------------- #
# (4) Flux in the spectrum (Aharonov–Bohm ring)
# --------------------------------------------------------------------------- #
class TestFluxSpectrum(unittest.TestCase):

    @staticmethod
    def _ring(phi):
        return sorted(2.0 - 2.0 * math.cos((phi + 2.0 * PI * k) / 3.0)
                      for k in range(3))

    @staticmethod
    def _triangle_total_flux(phi, spread=False):
        st = _cycle()
        edges = st.getEdgeList().toVector()
        if spread:  # same total flux, distributed across all three edges
            for e in edges:
                e.setPhase(phi / 3.0)
        else:       # all of Φ on one edge
            edges[0].setPhase(phi)
        return st

    def test_ring_formula_dense_sweep(self):
        for phi in np.linspace(-2.0 * PI, 2.0 * PI, 25):
            with self.subTest(phi=float(phi)):
                hl = _hodge(self._triangle_total_flux(phi))
                np.testing.assert_allclose(
                    sorted(_real_spectrum(hl.connectionEigenvalues())),
                    self._ring(phi), atol=1e-12)

    def test_spectrum_depends_only_on_total_flux(self):
        # Concentrated vs. spread flux of the same total are gauge-equivalent.
        for phi in (PI / 3, PI / 2, 2 * PI / 3, 1.234):
            with self.subTest(phi=phi):
                concentrated = sorted(_real_spectrum(_hodge(
                    self._triangle_total_flux(phi)).connectionEigenvalues()))
                spread = sorted(_real_spectrum(_hodge(
                    self._triangle_total_flux(
                        phi, spread=True)).connectionEigenvalues()))
                np.testing.assert_allclose(concentrated, spread, atol=1e-12)

    def test_half_quantum_collapses_gap_and_lifts_zero_mode(self):
        hl = _hodge(self._triangle_total_flux(PI))
        np.testing.assert_allclose(
            sorted(_real_spectrum(hl.connectionEigenvalues())),
            [1.0, 1.0, 4.0], atol=1e-12)
        self.assertEqual(len(hl.connectionHarmonics()), 0)  # zero mode gone

    def test_zero_mode_restored_at_full_flux_quantum(self):
        # Φ = 2π is gauge-equivalent to Φ = 0: spectrum {0, 3, 3}, zero mode back.
        hl = _hodge(self._triangle_total_flux(2.0 * PI))
        np.testing.assert_allclose(
            sorted(_real_spectrum(hl.connectionEigenvalues())),
            [0.0, 3.0, 3.0], atol=1e-10)
        self.assertEqual(len(hl.connectionHarmonics()), 1)

    def test_harmonic_dimension_tracks_flux(self):
        # Zero (or full-quantum) flux: one harmonic; any intermediate flux: none.
        self.assertEqual(len(_hodge(
            self._triangle_total_flux(0.0)).connectionHarmonics()), 1)
        for phi in (0.3, 1.0, PI / 2, 2.0):
            with self.subTest(phi=phi):
                self.assertEqual(len(_hodge(
                    self._triangle_total_flux(phi)).connectionHarmonics()), 0)

    def test_derived_degree_zero_kernel_is_flux_blind(self):
        # The contrast the connection tests above rest on (#805): L_0 is built
        # from d_1 and W_1, so no flux can lift its zero mode -- dim ker L_0 is
        # b_0 = 1 at every flux, and the constant is annihilated exactly.
        for phi in (0.0, 0.3, 1.0, PI / 2, PI, 2.0, 2.0 * PI):
            with self.subTest(phi=phi):
                st = self._triangle_total_flux(phi)
                hl = _hodge(st)
                self.assertEqual(len(hl.harmonics(0)), 1)
                n = cob.ChainComplex.fromSpacetime(st).numSimplices(0)
                L = np.array(hl.laplacian(0), dtype=complex).reshape(n, n)
                np.testing.assert_allclose(L @ np.ones(n), 0.0, atol=1e-15)


# --------------------------------------------------------------------------- #
# (5) Lorentzian d'Alembertian (§5.6): the null-norm (2 − α)/3 crossing α = 2
# --------------------------------------------------------------------------- #
def _triangle_one_timelike(alpha):
    """The 3-cycle 0-1-2-0 with edge (1,2) timelike (l² = −α²), edges (0,1),(0,2)
    spacelike (l² = 1).

    The timelike edge's 1-volume is sqrt(l²) = iα — IMAGINARY, not the −α a
    real-signed convention gave it (#641). The harmonic indefinite norm is
    therefore ⟨h,h⟩_W = 2/3 + iα/3: the two spacelike edges contribute the real
    2/3 and the timelike edge the imaginary α/3, exactly 90° apart."""
    st = _cycle()  # all-spacelike unit triangle
    _edge(st, 1, 2).setLength(cmath.sqrt(complex(-(alpha ** 2))))
    return st


class TestLorentzianNullNormCrossing(unittest.TestCase):

    def _null_norm(self, alpha):
        # Explicitly the k-content convention: the default is SquaredContent.
        norms = np.array(_hodge(_triangle_one_timelike(alpha),
                                            cob.HodgeWeightConvention.Content)
                         .nullNorms(1, 1e-9), dtype=complex)
        self.assertEqual(len(norms), 1)  # the single 1-cycle harmonic
        return norms[0]

    def _sorted_spectrum(self, alpha):
        # spec(L₁) = {0, 3, 1 − 2i/α} under the k-content convention, complex,
        # ordered by (Re, Im). The default is SquaredContent.
        ev = np.array(_hodge(_triangle_one_timelike(alpha),
                                         cob.HodgeWeightConvention.Content)
                      .eigenvalues(1), dtype=complex)
        return sorted(ev, key=lambda z: (round(z.real, 9), round(z.imag, 9)))

    def test_null_norm_closed_form_is_orthogonal_in_the_complex_plane(self):
        # ⟨h,h⟩_W = 2/3 + iα/3. The two spacelike edges give the real part and
        # the timelike edge the imaginary part, so the two contributions are 90°
        # apart and CANNOT cancel. The real part is α-independent; only the
        # imaginary part grows, linearly.
        sweep = (0.5, 1.0, 1.5, 1.9, 1.99, 2.01, 2.1, 2.5, 3.0)
        imags = []
        for alpha in sweep:
            with self.subTest(alpha=alpha):
                norm = self._null_norm(alpha)
                self.assertAlmostEqual(norm.real, 2.0 / 3.0, places=6)
                self.assertAlmostEqual(norm.imag, alpha / 3.0, places=6)
                imags.append(norm.imag)
        self.assertTrue(all(x < y for x, y in zip(imags, imags[1:])))

    def test_there_is_no_null_crossing(self):
        # PHYSICS CHANGE (#641), not a re-baselining. Under the old real-signed
        # convention the timelike edge's 1-volume was −α, so it could cancel the
        # spacelike +2/3 and the norm passed through zero at α = 2 — a genuine
        # null (lightlike) harmonic direction. With the honest imaginary volume
        # the two contributions are orthogonal, |⟨h,h⟩_W|² = (2/3)² + (α/3)² > 0
        # for every α, and no null direction exists at any α.
        for alpha in (0.5, 1.0, 1.9, 2.0, 2.1, 3.0, 10.0):
            with self.subTest(alpha=alpha):
                norm = self._null_norm(alpha)
                self.assertAlmostEqual(abs(norm) ** 2,
                                       (2.0 / 3.0) ** 2 + (alpha / 3.0) ** 2,
                                       places=6)
                self.assertGreater(abs(norm), 0.0)

    def test_third_eigenvalue_is_one_minus_two_i_over_alpha(self):
        # PHYSICS CHANGE (#641). spec(L₁) = {0, 3, 1 − 2i/α}, where the
        # real-signed convention gave {0, 3, 1 − 2/α}. The timelike edge's
        # 1-volume rotated from −α to +iα, so the third eigenvalue's α-dependence
        # rotated with it: its REAL part is pinned at 1 and only the imaginary
        # part moves. It therefore never reaches 0, and the α = 2 crossing that
        # the old form had is gone.
        for alpha in (0.5, 1.0, 1.5, 1.9, 2.1, 2.5, 3.0):
            with self.subTest(alpha=alpha):
                eigs = self._sorted_spectrum(alpha)
                expected = sorted([complex(0.0, 0.0), complex(3.0, 0.0),
                                   complex(1.0, -2.0 / alpha)],
                                  key=lambda z: (round(z.real, 9), round(z.imag, 9)))
                np.testing.assert_allclose(eigs, expected, atol=1e-7)
                self.assertGreater(abs(eigs[1] - 0.0), 0.5)  # never joins the kernel

    def test_no_second_mode_joins_the_kernel_at_alpha_two(self):
        # PHYSICS CHANGE (#641). The third eigenvalue is 1 − 2i/α, whose real
        # part is pinned at 1, so it never hits 0 and α = 2 is no longer a
        # defective coincidence. The kernel stays one-dimensional (the 1-cycle)
        # for every α, where the real-signed convention gave 2 at α = 2.
        for alpha in (1.5, 2.0, 2.5):
            with self.subTest(alpha=alpha):
                hl = _hodge(_triangle_one_timelike(alpha),
                                         cob.HodgeWeightConvention.Content)
                n = int(np.sum(np.abs(np.array(
                    hl.eigenvalues(1), dtype=complex)) < 1e-6))
                self.assertEqual(n, 1)

    def test_squared_content_convention_restores_the_null_crossing(self):
        # The two weight conventions are both fully Lorentzian and complex-typed;
        # they differ in WHAT the diagonal W_k is built from (#644).
        #
        #   W = V     (Content)         edge weight sqrt(l^2) = i*alpha
        #   W = V^2   (SquaredContent)  edge weight l^2       = -alpha^2
        #
        # Closed forms, verified exactly rather than fitted:
        #
        #                    Content              SquaredContent
        #   <h,h>_W          2/3 + i*alpha/3      (2 - alpha^2)/3
        #   spec(L_1) third  1 - 2i/alpha         1 - 2/alpha^2
        #
        # Content puts the timelike contribution on the imaginary axis, so it can
        # never cancel the spacelike part and no null direction exists at any
        # alpha. SquaredContent keeps the weights real and signed, so the norm
        # passes through zero at alpha = sqrt(2) and a second mode joins the
        # kernel there.
        sq = cob.HodgeWeightConvention.SquaredContent  # now the default
        for alpha in (0.5, 1.0, 1.5, 2.0, 3.0):
            with self.subTest(alpha=alpha):
                hl = _hodge(_triangle_one_timelike(alpha), sq)
                norm = np.array(hl.nullNorms(1, 1e-9), dtype=complex)[0]
                self.assertAlmostEqual(norm.imag, 0.0, places=9)
                self.assertAlmostEqual(norm.real, (2.0 - alpha ** 2) / 3.0, places=6)
                ev = np.array(hl.eigenvalues(1), dtype=complex)
                third = [z for z in ev if abs(z - 3.0) > 1e-6 and abs(z) > 1e-9]
                if third:
                    self.assertAlmostEqual(third[0].real,
                                           1.0 - 2.0 / alpha ** 2, places=6)
                    self.assertAlmostEqual(third[0].imag, 0.0, places=9)

    def test_squared_content_crossing_is_at_sqrt_two(self):
        # The kernel is one-dimensional (the 1-cycle) away from the crossing and
        # two-dimensional exactly at it, where 1 - 2/alpha^2 hits zero.
        sq = cob.HodgeWeightConvention.SquaredContent
        for alpha, expected in ((1.30, 1), (math.sqrt(2.0), 2), (1.55, 1)):
            with self.subTest(alpha=alpha):
                ev = np.array(_hodge(_triangle_one_timelike(alpha), sq)
                              .eigenvalues(1), dtype=complex)
                self.assertEqual(int(np.sum(np.abs(ev) < 1e-6)), expected)

    def test_harmonic_is_the_unit_cycle(self):
        # The kernel mode is the 1-cycle: |h_i|² = 1/3 on every edge, for any α.
        for alpha in (0.7, 1.3, 2.4):
            with self.subTest(alpha=alpha):
                harmonics = (_hodge(_triangle_one_timelike(alpha))
                             .harmonics(1, 1e-9))
                self.assertEqual(len(harmonics), 1)
                h = np.asarray(harmonics[0].coeffs())
                self.assertEqual(h.size, 3)
                np.testing.assert_allclose(np.abs(h) ** 2, np.full(3, 1.0 / 3.0),
                                           atol=1e-7)

    def test_all_spacelike_limit_is_definite(self):
        # The plain (all-spacelike, W = diag(1,1,1)) triangle: the cycle harmonic
        # has |h_i|² = 1/3 each, so its norm is Σ W_i|h_i|² = 1 > 0 (definite),
        # the spectrum is the Euclidean {0, 3, 3}, and the kernel dim = b₁ = 1.
        st = _cycle()
        norms = np.array(_hodge(st).nullNorms(1, 1e-9))
        self.assertEqual(len(norms), 1)
        self.assertAlmostEqual(norms[0], 1.0, places=9)
        eigs = np.sort(np.array(_hodge(st)
                                .eigenvalues(1), dtype=complex).real)
        np.testing.assert_allclose(eigs, [0.0, 3.0, 3.0], atol=1e-7)


# --------------------------------------------------------------------------- #
# (6) Edge cases
# --------------------------------------------------------------------------- #
class TestEdgeCases(unittest.TestCase):

    def test_single_vertex(self):
        # The 1×1 CONNECTION operator L = [0]: a single harmonic zero-mode, no
        # gap. It reads the vertex set directly, so it handles the lone vertex;
        # ChainComplex (built from *simplices*) registers a bare vertex as
        # nothing, so the DERIVED L_0 — indexed over 0-cells — is the EMPTY
        # operator here. Both are recorded so a change in either is noticed.
        st = _single_vertex()
        hl = _hodge(st)
        np.testing.assert_allclose(
            _real_spectrum(hl.connectionEigenvalues()), [0.0], atol=1e-12)
        self.assertEqual(len(hl.connectionHarmonics()), 1)
        self.assertEqual(obs.HarmonicDimension().compute(st), 1.0)
        self.assertEqual(obs.SpectralGap().compute(st), 0.0)      # no gap
        self.assertEqual(cob.ChainComplex.fromSpacetime(st).bettiNumbers(), [])
        # The derived L_0 has no 0-cells to act on.
        self.assertEqual(cob.ChainComplex.fromSpacetime(st).numSimplices(0), 0)
        self.assertEqual(hl.laplacian(0), [])
        self.assertEqual(hl.eigenvalues(0), [])
        self.assertEqual(hl.harmonics(0), [])

    def test_two_vertex_edge(self):
        st = _two_vertex_edge()
        hl = _hodge(st)
        np.testing.assert_allclose(sorted(_real_spectrum(hl.eigenvalues())), [0.0, 2.0], atol=1e-12)
        self.assertAlmostEqual(obs.SpectralGap().compute(st), 2.0, places=12)

    def test_disconnected_kernel_is_component_count(self):
        st = _disconnected()
        hl = _hodge(st)
        cc = cob.ChainComplex.fromSpacetime(st)
        self.assertEqual(cc.bettiNumbers()[0], 2)
        self.assertEqual(_kernel_dim(hl, 0, True), 2)             # b₀ = 2 components
        self.assertEqual(obs.HarmonicDimension().compute(st), 2.0)
        self.assertAlmostEqual(obs.SpectralGap().compute(st), 0.0, places=12)

    def test_empty_complex(self):
        empty = tessera.Spacetime()
        self.assertEqual(obs.SpectralGap().compute(empty), 0.0)
        self.assertEqual(obs.HarmonicDimension().compute(empty), 0.0)

    def test_degree_above_top_dimension_is_empty(self):
        for name, build in (("cycle", _cycle), ("S^2", _s2), ("T^2", _t2)):
            st = build()
            hl = _hodge(st)
            top = cob.ChainComplex.fromSpacetime(st).dimension()
            with self.subTest(fixture=name):
                for k in (top + 1, top + 2):
                    self.assertEqual(hl.laplacian(k), [])
                    self.assertEqual(hl.eigenvalues(k), [])
                    self.assertEqual(hl.eigenvectors(k), [])
                    self.assertEqual(hl.harmonics(k), [])
                    self.assertEqual(hl.weights(k), [])
                    self.assertEqual(hl.eigenvalues(k), [])
                    self.assertEqual(hl.nullNorms(k), [])

    def test_negative_degree_raises(self):
        hl = _hodge(_cycle())
        for call in (lambda: hl.laplacian(-1),
                     lambda: hl.eigenvalues(-1),
                     lambda: hl.eigenvectors(-1),
                     lambda: hl.harmonics(-1),
                     lambda: hl.eigenvalues(-1),
                     lambda: hl.eigenvectors(-1),
                     lambda: hl.harmonics(-1),
                     lambda: hl.nullNorms(-1)):
            with self.subTest(call=call):
                with self.assertRaises(RuntimeError):
                    call()
        # weights(k) is the exception: it returns empty for out-of-range k
        # (including k < 0) rather than raising.
        self.assertEqual(hl.weights(-1), [])


# --------------------------------------------------------------------------- #
# (7) Spectral observables (SpectralGap / HarmonicDimension) hardening
# --------------------------------------------------------------------------- #
class TestSpectralObservables(unittest.TestCase):

    def test_spectral_gap_matches_operator_across_fixtures(self):
        for name, build, _ in FIXTURES:
            st = build()
            evals = sorted(_real_spectrum(_hodge(st).eigenvalues()))
            with self.subTest(fixture=name):
                self.assertAlmostEqual(obs.SpectralGap().compute(st),
                                       evals[1] - evals[0], places=10)

    def test_harmonic_dimension_equals_b0_across_fixtures(self):
        # dim ker L₀ at zero flux == number of connected components == b₀.
        for name, build, expected in FIXTURES:
            st = build()
            b0 = cob.ChainComplex.fromSpacetime(st).bettiNumbers()[0]
            with self.subTest(fixture=name):
                self.assertEqual(b0, expected[0])
                self.assertEqual(obs.HarmonicDimension().compute(st), float(b0))

    def test_flux_collapses_gap_and_lifts_harmonic(self):
        st0 = _cycle()
        st_pi = _cycle()
        _edge(st_pi, 0, 1).setPhase(PI)
        self.assertAlmostEqual(obs.SpectralGap().compute(st0), 3.0, places=12)
        self.assertAlmostEqual(obs.SpectralGap().compute(st_pi), 0.0, places=12)
        self.assertEqual(obs.HarmonicDimension().compute(st0), 1.0)
        self.assertEqual(obs.HarmonicDimension().compute(st_pi), 0.0)
        # the topological b₀ is flux-independent
        self.assertEqual(
            cob.ChainComplex.fromSpacetime(st_pi).bettiNumbers()[0], 1)

    def test_observables_are_gauge_invariant(self):
        # A random vertex rephasing leaves both scalar observables unchanged.
        rng = np.random.default_rng(515)
        st = _testbed()
        for e in st.getEdgeList().toVector():
            e.setLength(cmath.sqrt(complex(float(rng.uniform(0.5, 2.0)))))
            e.setPhase(float(rng.uniform(-PI, PI)))
        gap0 = obs.SpectralGap().compute(st)
        dim0 = obs.HarmonicDimension().compute(st)
        ids, _ = _ordering(st)
        _apply_gauge(st, {vid: float(rng.uniform(-PI, PI)) for vid in ids})
        self.assertAlmostEqual(obs.SpectralGap().compute(st), gap0, places=10)
        self.assertEqual(obs.HarmonicDimension().compute(st), dim0)


if __name__ == "__main__":
    unittest.main()
