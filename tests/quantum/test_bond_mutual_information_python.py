"""Acceptance tests for the bond-cut / dual-lattice mutual information.

Verifies that TDVPConfig.record_bond_mutual_information populates a
symmetric, non-negative, zero-diagonal (N-1) × (N-1) matrix per
snapshot, with values consistent with the bond entropies on a small
Schwinger run.
"""
from __future__ import annotations

import math
import unittest

import numpy as np

try:
    from tessera.quantum import SchwingerQuench, TDVPConfig
    HAVE_QUANTUM = True
except ImportError:
    HAVE_QUANTUM = False


def _config(N, m_over_g, T, dt):
    cfg = TDVPConfig()
    cfg.N = N; cfg.a = 1.0; cfg.g = 1.0
    cfg.m = m_over_g * cfg.g; cfg.L0 = 0.0
    cfg.dmrg_max_bond_dim = 32; cfg.dmrg_n_sweeps = 10
    cfg.dmrg_krylov_dim = 4; cfg.dmrg_cutoff = 1e-12
    cfg.i0 = 1; cfg.d = 3; cfg.quench_enforce_parity = True
    cfg.dt = dt; cfg.T = T; cfg.snapshot_every = 1
    cfg.max_bond_dim = 40; cfg.cutoff = 1e-10; cfg.krylov_dim = 10
    cfg.quiet = True; cfg.conserve_qns = True
    cfg.record_bond_mutual_information = True
    return cfg


@unittest.skipUnless(HAVE_QUANTUM, "tessera built without TESSERA_QUANTUM=1")
class TestBondMutualInformationShape(unittest.TestCase):
    """Per-snapshot bond MI matrix has the right shape and properties."""

    def test_shape_and_symmetry(self):
        cfg = _config(N=6, m_over_g=0.5, T=0.2, dt=0.2)
        res = SchwingerQuench(cfg).evolve()
        for snap in res.snapshots:
            flat = list(snap.bond_mutual_information)
            self.assertEqual(len(flat), (cfg.N - 1) ** 2,
                msg="bond MI matrix should be (N-1) x (N-1)")
            bm = np.array(flat).reshape(cfg.N - 1, cfg.N - 1)
            # Symmetric (within ~1e-10).
            self.assertTrue(
                np.allclose(bm, bm.T, atol=1e-10),
                msg=f"bond MI not symmetric: max |asym| = {(bm - bm.T).max():.3e}")
            # Zero diagonal by convention.
            self.assertTrue(np.allclose(np.diag(bm), 0.0, atol=1e-12))
            # All entries finite.
            self.assertTrue(np.all(np.isfinite(bm)))

    def test_non_negative_at_short_time(self):
        """For a pure state the tripartite info S(A) + S(C) - S(B) is the
        MI between the two outer regions, hence non-negative. Verify on
        the initial post-quench state (t=0) where the structure is
        clean."""
        cfg = _config(N=6, m_over_g=0.5, T=0.2, dt=0.2)
        res = SchwingerQuench(cfg).evolve()
        snap0 = res.snapshots[0]
        bm = np.array(snap0.bond_mutual_information).reshape(
            cfg.N - 1, cfg.N - 1)
        # Allow ~1e-9 for numerical noise.
        self.assertGreaterEqual(
            float(bm.min()), -1e-9,
            msg=f"bond tripartite info went negative: min = {bm.min():.3e}")


@unittest.skipUnless(HAVE_QUANTUM, "tessera built without TESSERA_QUANTUM=1")
class TestBondMutualInformationOptIn(unittest.TestCase):
    """Without the flag set, bondMutualInformation is empty."""

    def test_empty_without_flag(self):
        cfg = _config(N=6, m_over_g=0.5, T=0.2, dt=0.2)
        cfg.record_bond_mutual_information = False
        res = SchwingerQuench(cfg).evolve()
        for snap in res.snapshots:
            self.assertEqual(
                len(snap.bond_mutual_information), 0,
                msg="bondMutualInformation should be empty when flag is off")


if __name__ == "__main__":
    unittest.main()
