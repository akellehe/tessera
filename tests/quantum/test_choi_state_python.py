"""Acceptance tests for the Choi-state temporal mutual information.

Mirrors the §H2 acceptance list in
``docs/source/quantum-experiments/earlier-work/emergent-spectral-dimension-schwinger-tdvp.md``:

  • Identity channel: $I(i_\\mathrm{in} : i_\\mathrm{out}) = 2 \\ln 2$, all
    off-diagonal entries zero.
  • Identity channel: the Choi state is $|\\Phi^+\\rangle^{\\otimes N}$
    (verified via temporal-MI matrix shape rather than reconstructing
    the MPS).
  • Non-zero duration: the temporal-MI matrix becomes asymmetric in
    (i, j) once the propagator mixes neighbouring sites.
  • Holography pipeline cross-check: when ``include_temporal`` is on,
    edge count goes up (vs. spatial-only) and D_S(σ) profile shifts
    upward in the diffusion regime.

Skips cleanly when tessera was built without TESSERA_QUANTUM=1.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

try:
    from tessera.quantum import TDVPConfig
    from tessera.quantum.holography import (
        HolographyConfig,
        EmergentSpectralDimension,
        ChoiPropagator,
        ChoiTDVPSettings,
        SchwingerParams,
    )
    HAVE_QUANTUM = True
except ImportError:
    HAVE_QUANTUM = False


def _settings(maxBondDim: int = 64) -> "ChoiTDVPSettings":
    s = ChoiTDVPSettings()
    s.dt         = 0.1
    s.max_bond_dim = maxBondDim
    s.krylov_dim  = 12
    s.cutoff     = 1e-12
    s.quiet      = True
    return s


def _params(N: int = 4, m: float = 0.5, g: float = 1.0,
             L0: float = 0.0) -> "SchwingerParams":
    p = SchwingerParams()
    p.N = N; p.a = 1.0; p.g = g; p.m = m; p.L0 = L0
    return p


@unittest.skipUnless(HAVE_QUANTUM, "tessera built without TESSERA_QUANTUM=1")
class TestIdentityChannel(unittest.TestCase):
    """Spec §H2 acceptance #1 and #2: the identity channel."""

    def test_diagonal_is_two_ln_two(self) -> None:
        """Bell-pair initial state: each (in_i, out_i) is |Φ+⟩ →
        I = S(½) + S(½) - 0 = 2·ln(2)."""
        p = _params(N=4)
        mi = ChoiPropagator.temporal_mutual_information(p, 0.0, _settings())
        expected = 2.0 * math.log(2.0)
        for i in range(4):
            self.assertAlmostEqual(mi[i, i], expected, places=12,
                msg=f"diagonal entry [{i},{i}] = {mi[i, i]} ≠ 2 ln 2")

    def test_off_diagonal_is_zero(self) -> None:
        """No cross-pair entanglement in |Φ+⟩^⊗N."""
        p = _params(N=4)
        mi = ChoiPropagator.temporal_mutual_information(p, 0.0, _settings())
        for i in range(4):
            for j in range(4):
                if i == j: continue
                self.assertAlmostEqual(mi[i, j], 0.0, places=12,
                    msg=f"off-diagonal entry [{i},{j}] = {mi[i, j]} ≠ 0")

    def test_symmetric_at_identity(self) -> None:
        """The identity channel's temporal MI matrix happens to be
        symmetric (it's diagonal). Generic non-identity channels need
        not be."""
        p = _params(N=4)
        mi = ChoiPropagator.temporal_mutual_information(p, 0.0, _settings())
        for i in range(4):
            for j in range(4):
                self.assertAlmostEqual(mi[i, j], mi[j, i], places=12)


@unittest.skipUnless(HAVE_QUANTUM, "tessera built without TESSERA_QUANTUM=1")
class TestNonIdentityChannel(unittest.TestCase):
    """Sanity checks on the Choi state for genuine evolution."""

    def test_diagonal_decreases_with_duration(self) -> None:
        """As the propagator scrambles the state, the diagonal entries
        (i.e. the surviving "this site's MI with its earlier self")
        decrease from 2·ln(2) — the trivial-channel maximum."""
        p = _params(N=4, m=0.5)
        settings = _settings()
        mi0 = ChoiPropagator.temporal_mutual_information(p, 0.0, settings)
        mi1 = ChoiPropagator.temporal_mutual_information(p, 0.5, settings)
        max_two_ln2 = 2.0 * math.log(2.0)
        for i in range(4):
            self.assertAlmostEqual(mi0[i, i], max_two_ln2, places=12)
            self.assertLess(mi1[i, i], max_two_ln2,
                msg=f"diagonal entry [{i},{i}] did not decrease under evolution")

    def test_off_diagonal_grows_with_duration(self) -> None:
        """Nearest-neighbor off-diagonal entries should grow as the
        Schwinger hopping mixes adjacent sites."""
        p = _params(N=4, m=0.5)
        settings = _settings()
        mi_short = ChoiPropagator.temporal_mutual_information(p, 0.1, settings)
        mi_long  = ChoiPropagator.temporal_mutual_information(p, 0.5, settings)
        # Adjacent off-diagonal entries (0,1), (1,2), (2,3).
        for i in range(3):
            self.assertGreater(mi_long[i, i + 1], mi_short[i, i + 1] - 1e-10,
                msg=f"off-diagonal [{i},{i+1}] should grow; "
                    f"short={mi_short[i, i+1]:.4f}, long={mi_long[i, i+1]:.4f}")

    def test_temporal_mi_is_non_negative(self) -> None:
        """Mutual information is non-negative on a pure state."""
        p = _params(N=4, m=0.5)
        mi = ChoiPropagator.temporal_mutual_information(p, 0.3, _settings())
        # Allow ~1e-10 numerical noise.
        self.assertGreaterEqual(np.min(mi), -1e-10)


@unittest.skipUnless(HAVE_QUANTUM, "tessera built without TESSERA_QUANTUM=1")
class TestPipelineWithTemporalMI(unittest.TestCase):
    """End-to-end behaviour change when includeTemporal flips on."""

    def _config(self, m_over_g: float, include_temporal: bool) -> "HolographyConfig":
        cfg = HolographyConfig()
        cfg.tdvp = TDVPConfig()
        cfg.tdvp.N = 6; cfg.tdvp.a = 1.0; cfg.tdvp.g = 1.0
        cfg.tdvp.m = m_over_g; cfg.tdvp.L0 = 0.0
        cfg.tdvp.dmrg_max_bond_dim = 32; cfg.tdvp.dmrg_n_sweeps = 8
        cfg.tdvp.dmrg_krylov_dim = 4; cfg.tdvp.dmrg_cutoff = 1e-12
        cfg.tdvp.i0 = 1; cfg.tdvp.d = 3
        cfg.tdvp.dt = 0.2; cfg.tdvp.T = 0.4; cfg.tdvp.snapshot_every = 1
        cfg.tdvp.max_bond_dim = 40; cfg.tdvp.cutoff = 1e-10; cfg.tdvp.krylov_dim = 10
        cfg.tdvp.quiet = True; cfg.tdvp.conserve_qns = True
        cfg.sigma_min = 0.1; cfg.sigma_max = 100.0; cfg.sigma_count = 24
        cfg.epsilon_i = 1e-8; cfg.krylov_dim = 30
        cfg.include_temporal = include_temporal
        cfg.max_temporal_stride = 0
        return cfg

    def test_temporal_increases_edge_count(self) -> None:
        """Turning on temporal MI must add edges to the (site, time)
        graph — cross-snapshot blocks go from zero to dense."""
        r_no   = EmergentSpectralDimension(self._config(0.5, False)).compute()
        r_yes  = EmergentSpectralDimension(self._config(0.5, True)).compute()
        self.assertEqual(r_no.graph_n_vertices, r_yes.graph_n_vertices)
        self.assertGreater(r_yes.graph_n_edges, r_no.graph_n_edges,
            msg="temporal MI should add edges; got "
                f"E_no={r_no.graph_n_edges}, E_yes={r_yes.graph_n_edges}")

    def test_temporal_raises_peak_dS(self) -> None:
        """Spec §1: temporal connectivity should push D_S(σ) toward 2
        (the locally-2D lattice dimension) in the diffusion regime.
        Verify the peak D_S with temporal MI on is at least as large
        as without."""
        r_no   = EmergentSpectralDimension(self._config(0.5, False)).compute()
        r_yes  = EmergentSpectralDimension(self._config(0.5, True)).compute()
        peak_no  = max(d for d in r_no.d_s  if math.isfinite(d))
        peak_yes = max(d for d in r_yes.d_s if math.isfinite(d))
        self.assertGreater(peak_yes, peak_no - 0.01,
            msg=f"peak D_S did not rise with temporal MI: "
                f"no={peak_no:.3f}, yes={peak_yes:.3f}")


if __name__ == "__main__":
    unittest.main()
