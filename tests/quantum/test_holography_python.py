"""Python acceptance tests for the holography submodule.

Mirrors the test plan in
``docs/source/quantum-experiments/earlier-work/emergent-spectral-dimension-schwinger-tdvp.md`` §9:

* MutualInformationProfile and EmergentGraph construction on a TDVP run.
* Spectral-dimension self-consistency (P monotone decreasing in σ;
  D_S finite on the diffusion-regime grid points).
* m/g sensitivity: heavy-quark vs light-quark runs give distinguishable
  D_S(σ) profiles (rejects the trivial-confirmation falsification).
* HolographyConfig validation: invalid configs raise before any TDVP work.

Skips cleanly when tessera was built without TESSERA_QUANTUM=1.
"""

from __future__ import annotations

import math
import unittest

try:
    from tessera.quantum import TDVPConfig, SchwingerQuench
    from tessera.quantum.holography import (
        HolographyConfig,
        MutualInformationProfile,
        EmergentGraph,
        AmbjornLollFit,
        EmergentSpectralDimension,
    )
    HAVE_QUANTUM = True
except ImportError:
    HAVE_QUANTUM = False


def _small_holography_config(N: int = 6, m: float = 0.5,
                              T: float = 0.4) -> "HolographyConfig":
    cfg = HolographyConfig()
    cfg.tdvp = TDVPConfig()
    cfg.tdvp.N = N
    cfg.tdvp.a = 1.0; cfg.tdvp.g = 1.0; cfg.tdvp.m = m; cfg.tdvp.L0 = 0.0
    cfg.tdvp.dmrg_max_bond_dim = 32; cfg.tdvp.dmrg_n_sweeps = 8
    cfg.tdvp.dmrg_krylov_dim = 4; cfg.tdvp.dmrg_cutoff = 1e-12
    cfg.tdvp.i0 = 1; cfg.tdvp.d = 3
    cfg.tdvp.dt = 0.2; cfg.tdvp.T = T; cfg.tdvp.snapshot_every = 1
    cfg.tdvp.max_bond_dim = 40; cfg.tdvp.cutoff = 1e-10; cfg.tdvp.krylov_dim = 10
    cfg.tdvp.quiet = True; cfg.tdvp.conserve_qns = True
    cfg.sigma_min = 0.1; cfg.sigma_max = 100.0; cfg.sigma_count = 24
    cfg.epsilon_i = 1e-8
    cfg.krylov_dim = 30
    return cfg


@unittest.skipUnless(HAVE_QUANTUM, "tessera built without TESSERA_QUANTUM=1")
class TestHolographyConfig(unittest.TestCase):
    """Construction-time validation matches the spec §4.2 contract."""

    def test_default_construction(self) -> None:
        cfg = HolographyConfig()
        cfg.tdvp = TDVPConfig()
        cfg.validate()  # default σ-grid is valid

    def test_sigma_min_must_be_positive(self) -> None:
        cfg = HolographyConfig()
        cfg.sigma_min = -0.1
        with self.assertRaises(Exception):
            cfg.validate()

    def test_sigma_max_must_exceed_min(self) -> None:
        cfg = HolographyConfig()
        cfg.sigma_min = 1.0; cfg.sigma_max = 0.5
        with self.assertRaises(Exception):
            cfg.validate()

    def test_sigma_count_minimum(self) -> None:
        cfg = HolographyConfig()
        cfg.sigma_count = 4
        with self.assertRaises(Exception):
            cfg.validate()

    def test_negative_epsilon_rejected(self) -> None:
        cfg = HolographyConfig()
        cfg.epsilon_i = -1e-10
        with self.assertRaises(Exception):
            cfg.validate()


@unittest.skipUnless(HAVE_QUANTUM, "tessera built without TESSERA_QUANTUM=1")
class TestMutualInformationProfile(unittest.TestCase):
    """Profile construction from real TDVP snapshots."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = _small_holography_config(N=6, m=0.5, T=0.4)
        cls.cfg.tdvp.record_mutual_information = True
        cls.quench = SchwingerQuench(cls.cfg.tdvp).evolve()

    def test_profile_dimensions(self) -> None:
        profile = MutualInformationProfile(self.quench.snapshots, self.cfg)
        self.assertEqual(profile.n_sites, self.cfg.tdvp.N)
        self.assertEqual(profile.n_snapshots, len(self.quench.snapshots))
        self.assertEqual(
            profile.n_labels,
            self.cfg.tdvp.N * len(self.quench.snapshots))

    def test_diagonal_is_zero(self) -> None:
        """I(v, v) = 0 by construction (S(ρ) - S(ρ) = 0)."""
        profile = MutualInformationProfile(self.quench.snapshots, self.cfg)
        for v in range(profile.n_labels):
            self.assertAlmostEqual(profile.at_flat(v, v), 0.0, places=12)

    def test_symmetry(self) -> None:
        profile = MutualInformationProfile(self.quench.snapshots, self.cfg)
        n = profile.n_labels
        for v in range(n):
            for w in range(v + 1, n):
                self.assertAlmostEqual(
                    profile.at_flat(v, w), profile.at_flat(w, v), places=12)

    def test_non_negativity(self) -> None:
        """Mutual information is non-negative on a pure state."""
        profile = MutualInformationProfile(self.quench.snapshots, self.cfg)
        n = profile.n_labels
        for v in range(n):
            for w in range(n):
                I = profile.at_flat(v, w)
                # Allow tiny negative numerical noise from finite-precision SVD
                self.assertGreaterEqual(I, -1e-10,
                    msg=f"I({v},{w}) = {I} is too negative")

    def test_within_snapshot_block_is_dense_cross_block_zero(self) -> None:
        """v1 has spatial MI only — cross-snapshot blocks are exact zero."""
        profile = MutualInformationProfile(self.quench.snapshots, self.cfg)
        nSites = profile.n_sites
        for v in range(profile.n_labels):
            for w in range(profile.n_labels):
                sv = profile.snapshot_of(v)
                sw = profile.snapshot_of(w)
                if sv != sw:
                    self.assertEqual(profile.at_flat(v, w), 0.0,
                        msg="v1: cross-snapshot MI must be exact 0")

    def test_weighted_adjacency_coo_format(self) -> None:
        profile = MutualInformationProfile(self.quench.snapshots, self.cfg)
        rows, cols, weights, n = profile.weighted_adjacency()
        self.assertEqual(n, profile.n_labels)
        self.assertEqual(len(rows), len(cols))
        self.assertEqual(len(rows), len(weights))
        # Symmetric — each edge listed twice.
        edge_set = set(zip(rows, cols))
        for r, c in edge_set:
            self.assertIn((c, r), edge_set, "asymmetric COO output")
        # Weights all > epsilonI
        for w in weights:
            self.assertGreater(w, self.cfg.epsilon_i)


@unittest.skipUnless(HAVE_QUANTUM, "tessera built without TESSERA_QUANTUM=1")
class TestEmergentGraph(unittest.TestCase):
    """Graph and heat-kernel properties on a real TDVP run."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = _small_holography_config(N=6, m=0.5, T=0.4)
        cls.cfg.tdvp.record_mutual_information = True
        cls.quench = SchwingerQuench(cls.cfg.tdvp).evolve()

    def test_graph_has_vertices(self) -> None:
        profile = MutualInformationProfile(self.quench.snapshots, self.cfg)
        graph = EmergentGraph(profile)
        self.assertEqual(graph.n_vertices, profile.n_labels)
        self.assertGreater(graph.n_edges, 0)

    @staticmethod
    def _dense_laplacian(graph):
        import scipy.sparse as sp
        rows, cols, vals, n = graph.laplacian_coo()
        return sp.csr_matrix((vals, (rows, cols)), shape=(n, n)).toarray()

    def test_laplacian_is_symmetric(self) -> None:
        """L = D - W where W is symmetric → L is symmetric."""
        profile = MutualInformationProfile(self.quench.snapshots, self.cfg)
        graph = EmergentGraph(profile)
        L = self._dense_laplacian(graph)
        for i in range(L.shape[0]):
            for j in range(L.shape[1]):
                self.assertAlmostEqual(
                    L[i, j], L[j, i], places=12,
                    msg=f"L[{i},{j}] != L[{j},{i}]")

    def test_laplacian_row_sums_are_zero(self) -> None:
        """L · 1 = (D - W) · 1 = D - W·1 = D - D = 0."""
        profile = MutualInformationProfile(self.quench.snapshots, self.cfg)
        graph = EmergentGraph(profile)
        L = self._dense_laplacian(graph)
        row_sums = L.sum(axis=1)
        for r in row_sums:
            self.assertAlmostEqual(r, 0.0, places=10)

    def test_return_probability_is_one_at_zero_diffusion(self) -> None:
        """P(σ → 0) → 1 because exp(0) = I and Tr(I)/|V| = 1. At small
        but nonzero σ the diagonal heat-kernel entry is 1 - σ·degree +
        O(σ²), so the test takes the σ → 0 limit numerically by going
        to very small σ where the linear correction is also negligible."""
        profile = MutualInformationProfile(self.quench.snapshots, self.cfg)
        graph = EmergentGraph(profile)
        sigmas = [1e-14, 1e-12, 1e-10]
        P = graph.return_probability(sigmas, 30)
        for p, s in zip(P, sigmas):
            # σ · max_degree is the leading deviation; bounded above by
            # σ · (N · max_MI) and we know N is small.
            self.assertAlmostEqual(p, 1.0, places=8,
                msg=f"P({s}) = {p}; expected ≈ 1")

    def test_return_probability_is_monotone_decreasing(self) -> None:
        """For a graph with at least one edge, the diagonal heat kernel
        is non-increasing in σ — diffusion spreads probability away
        from the start, never back."""
        profile = MutualInformationProfile(self.quench.snapshots, self.cfg)
        graph = EmergentGraph(profile)
        sigmas = [0.01 * (2 ** k) for k in range(12)]
        P = graph.return_probability(sigmas, 30)
        for k in range(len(P) - 1):
            self.assertGreaterEqual(P[k], P[k + 1] - 1e-10,
                msg=f"P({sigmas[k]}) = {P[k]} < P({sigmas[k+1]}) = {P[k+1]}")

    def test_to_dot_is_valid_dot(self) -> None:
        profile = MutualInformationProfile(self.quench.snapshots, self.cfg)
        graph = EmergentGraph(profile)
        dot = graph.to_dot()
        self.assertIn("graph emergent {", dot)
        self.assertIn("--", dot)  # at least one undirected edge


@unittest.skipUnless(HAVE_QUANTUM, "tessera built without TESSERA_QUANTUM=1")
class TestSpectralDimensionPureMath(unittest.TestCase):
    """The D_S(σ) = -2 d log P / d log σ formula on synthetic P arrays."""

    def test_dS_zero_on_constant_P(self) -> None:
        """If P is constant, log P has slope 0, so D_S = 0."""
        sigmas = [0.1 * (2 ** k) for k in range(10)]
        P = [0.5] * 10
        dS = EmergentGraph.spectral_dimension(sigmas, P)
        for d in dS:
            self.assertAlmostEqual(d, 0.0, places=10)

    def test_dS_two_on_power_law_minus_one(self) -> None:
        """If P(σ) = σ^{-1}, then d log P / d log σ = -1, D_S = 2."""
        sigmas = [0.1 * (10 ** (0.1 * k)) for k in range(40)]
        P = [1.0 / s for s in sigmas]
        dS = EmergentGraph.spectral_dimension(sigmas, P)
        # Interior points: D_S = 2 exactly (the finite-difference formula
        # on a perfect power law is exact). Endpoints use one-sided
        # differences and pick up the same slope.
        for d in dS:
            self.assertAlmostEqual(d, 2.0, places=10)


@unittest.skipUnless(HAVE_QUANTUM, "tessera built without TESSERA_QUANTUM=1")
class TestEmergentSpectralDimensionPipeline(unittest.TestCase):
    """End-to-end pipeline via EmergentSpectralDimension.compute()."""

    def test_pipeline_runs_and_returns_valid_result(self) -> None:
        cfg = _small_holography_config(N=6, m=0.5, T=0.4)
        result = EmergentSpectralDimension(cfg).compute()

        self.assertEqual(len(result.sigmas), cfg.sigma_count)
        self.assertEqual(len(result.P),      cfg.sigma_count)
        self.assertEqual(len(result.d_s),     cfg.sigma_count)
        self.assertEqual(result.graph_n_vertices,
                         cfg.tdvp.N * len(result.snapshot_times))
        self.assertGreater(result.graph_n_edges, 0)

    def test_pipeline_records_tdvp_summary(self) -> None:
        cfg = _small_holography_config(N=6, m=0.5, T=0.4)
        result = EmergentSpectralDimension(cfg).compute()
        # Snapshot times, bondDims, energies all populated.
        self.assertGreater(len(result.snapshot_times), 0)
        self.assertEqual(len(result.snapshot_times),
                         len(result.snapshot_bond_dims))
        self.assertEqual(len(result.snapshot_times),
                         len(result.snapshot_energies))

    def test_pipeline_idempotent(self) -> None:
        """Two compute() calls on the same instance give the same numbers."""
        cfg = _small_holography_config(N=6, m=0.5, T=0.4)
        runner = EmergentSpectralDimension(cfg)
        a = runner.compute()
        b = runner.compute()
        for i, (xa, xb) in enumerate(zip(a.sigmas, b.sigmas)):
            self.assertAlmostEqual(xa, xb, places=12,
                msg=f"sigma[{i}] differs")
        for i, (pa, pb) in enumerate(zip(a.P, b.P)):
            self.assertAlmostEqual(pa, pb, places=8,
                msg=f"P[{i}] differs")

    def test_compute_from_snapshots_matches_compute(self) -> None:
        """The two entry points must agree when fed the same TDVP run."""
        cfg = _small_holography_config(N=6, m=0.5, T=0.4)
        # Force MI recording so we can re-use the quench result.
        cfg.tdvp.record_mutual_information = True
        quench = SchwingerQuench(cfg.tdvp).evolve()
        runner = EmergentSpectralDimension(cfg)
        # compute() will re-run TDVP from scratch — deterministic, so
        # should give the same result as computeFromSnapshots.
        a = runner.compute_from_snapshots(quench)
        b = runner.compute()
        for i, (pa, pb) in enumerate(zip(a.P, b.P)):
            self.assertAlmostEqual(pa, pb, places=8)
        self.assertAlmostEqual(a.d_infinity, b.d_infinity, places=6)


@unittest.skipUnless(HAVE_QUANTUM, "tessera built without TESSERA_QUANTUM=1")
class TestAmbjornLollFit(unittest.TestCase):
    """The three-parameter D_S(σ) = D_∞ - C / (B + σ) curve fit."""

    def test_exact_recovery_on_noiseless_data(self) -> None:
        # Generate D_S from a known (D_∞, C, B), fit, check we recover.
        D_inf_true, C_true, B_true = 2.5, 1.0, 0.5
        sigmas = [0.01 * (10 ** (0.05 * k)) for k in range(60)]
        dS = [D_inf_true - C_true / (B_true + s) for s in sigmas]
        result = AmbjornLollFit.fit(sigmas, dS)
        self.assertAlmostEqual(result.d_infinity, D_inf_true, places=6)
        self.assertAlmostEqual(result.C,         C_true,     places=6)
        self.assertAlmostEqual(result.B,         B_true,     places=6)
        self.assertLess(result.chi_squared, 1e-12)


@unittest.skipUnless(HAVE_QUANTUM, "tessera built without TESSERA_QUANTUM=1")
class TestMassSensitivity(unittest.TestCase):
    """The H_SD hypothesis says D_S(σ) depends on m/g (not trivially
    constant in the underlying physics). Test the falsification criterion
    'trivial confirmation' by checking that heavy-quark and light-quark
    runs produce distinguishable D_S(σ) profiles."""

    def test_heavy_vs_light_mass_sensitivity(self) -> None:
        heavy = _small_holography_config(N=6, m=20.0, T=0.4)
        light = _small_holography_config(N=6, m=0.5,  T=0.4)
        r_heavy = EmergentSpectralDimension(heavy).compute()
        r_light = EmergentSpectralDimension(light).compute()

        # At least one D_S(σ) value must differ noticeably between the
        # two — a numeric stand-in for "the profile responds to physics".
        max_abs_diff = max(
            abs(a - b) for a, b in zip(r_heavy.d_s, r_light.d_s)
            if not (math.isnan(a) or math.isnan(b))
        )
        self.assertGreater(max_abs_diff, 1e-3,
            msg=f"Heavy and light D_S profiles are indistinguishable "
                f"(max |Δ| = {max_abs_diff:.2e})")


if __name__ == "__main__":
    unittest.main()
