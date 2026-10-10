"""Python tests — end-to-end causal-comparison pipeline through
:meth:`SchwingerQuench.compareCausalOrders`, plus low-level
:meth:`Majorization.agreement` unit tests on hand-built posets.

Skips cleanly when tessera was built without TESSERA_QUANTUM=1.
"""

from __future__ import annotations

import unittest

try:
    from tessera import OrderAgreement, Poset
    from tessera.quantum import (
        TDVPConfig,
        CausalComparisonReport,
        Majorization,
        SchwingerQuench,
    )
    HAVE_QUANTUM = True
except ImportError:
    HAVE_QUANTUM = False


def _light_quark_config(N: int = 10, T: float = 0.4) -> "TDVPConfig":
    cfg = TDVPConfig()
    cfg.N = N; cfg.a = 1.0; cfg.g = 1.0; cfg.m = 0.5; cfg.L0 = 0.0
    cfg.dmrgMaxBondDim = 32; cfg.dmrgNSweeps = 10
    cfg.i0 = 1; cfg.d = 3
    cfg.dt = 0.2; cfg.T = T; cfg.snapshotEvery = 1
    cfg.maxBondDim = 60
    cfg.cutoff = 1e-10; cfg.krylovDim = 12
    cfg.quiet = True; cfg.conserveQns = True
    return cfg


@unittest.skipUnless(HAVE_QUANTUM, "tessera built without TESSERA_QUANTUM=1")
class TestAgreementPure(unittest.TestCase):
    """Unit tests on Majorization.agreement() without running TDVP."""

    def _poset(self, getNodeCount: int, covers: list[tuple[int, int]]) -> "Poset":
        p = Poset()
        p.getNodeCount = getNodeCount
        p.covers = covers
        return p

    def test_identical_posets(self) -> None:
        p = self._poset(4, [(0, 1), (1, 2), (2, 3)])
        agr = Majorization.agreement(p, p, 4)
        self.assertAlmostEqual(agr.kendallTau, 1.0)
        self.assertAlmostEqual(agr.discordantFraction, 0.0)
        self.assertAlmostEqual(agr.hasseEditDistance, 0.0)
        self.assertEqual(agr.nDiscordant, 0)

    def test_reversed_posets(self) -> None:
        p = self._poset(3, [(0, 1), (1, 2)])
        q = self._poset(3, [(2, 1), (1, 0)])
        agr = Majorization.agreement(p, q, 3)
        self.assertAlmostEqual(agr.kendallTau, -1.0)
        self.assertEqual(agr.nConcordant, 0)
        self.assertGreater(agr.nDiscordant, 0)

    def test_disjoint_posets(self) -> None:
        p = self._poset(4, [(0, 1)])
        q = self._poset(4, [(2, 3)])
        agr = Majorization.agreement(p, q, 4)
        self.assertAlmostEqual(agr.hasseEditDistance, 1.0)

    def test_no_pairs_in_common(self) -> None:
        p = self._poset(5, [])
        q = self._poset(5, [])
        agr = Majorization.agreement(p, q, 5)
        self.assertEqual(agr.nComparableBoth, 0)
        self.assertAlmostEqual(agr.kendallTau, 0.0)
        self.assertAlmostEqual(agr.discordantFraction, 0.0)
        self.assertAlmostEqual(agr.hasseEditDistance, 0.0)

    def test_kendall_tau_in_range(self) -> None:
        p = self._poset(5, [(0, 1), (0, 2), (1, 3), (2, 4)])
        q = self._poset(5, [(0, 4), (4, 3), (3, 2), (2, 1)])
        agr = Majorization.agreement(p, q, 5)
        self.assertGreaterEqual(agr.kendallTau, -1.0)
        self.assertLessEqual(agr.kendallTau, 1.0)
        self.assertGreaterEqual(agr.discordantFraction, 0.0)
        self.assertLessEqual(agr.discordantFraction, 1.0)


@unittest.skipUnless(HAVE_QUANTUM, "tessera built without TESSERA_QUANTUM=1")
class TestCompareCausalOrders(unittest.TestCase):
    """End-to-end pipeline tests through SchwingerQuench.compareCausalOrders."""

    def test_pipeline_runs(self) -> None:
        cfg = _light_quark_config(N=8, T=0.4)
        r = SchwingerQuench(cfg).compareCausalOrders(vLr=1.0)

        # 8(8+1)/2 - 1 = 35 cuts × 3 snapshots (t=0, t=0.2, t=0.4) = 105 labels.
        self.assertEqual(r.nSnapshots, 3)
        self.assertEqual(r.nLabels, 35 * 3)
        self.assertAlmostEqual(r.vLr, 1.0)

    def test_kendall_tau_in_range(self) -> None:
        cfg = _light_quark_config(N=8, T=0.4)
        r = SchwingerQuench(cfg).compareCausalOrders(vLr=1.0)
        for agr in (r.majVsLr, r.majVsCs, r.lrVsCs):
            self.assertGreaterEqual(agr.kendallTau, -1.0)
            self.assertLessEqual(agr.kendallTau, 1.0)
            self.assertGreaterEqual(agr.discordantFraction, 0.0)
            self.assertLessEqual(agr.discordantFraction, 1.0)
            self.assertGreaterEqual(agr.hasseEditDistance, 0.0)
            self.assertLessEqual(agr.hasseEditDistance, 1.0)

    def test_lr_subset_of_cs(self) -> None:
        """≼_LR ⊂ ≼_cs by construction — τ(LR, cs) must be exactly 1.0."""
        cfg = _light_quark_config(N=8, T=0.4)
        r = SchwingerQuench(cfg).compareCausalOrders(vLr=1.0)
        self.assertAlmostEqual(r.lrVsCs.kendallTau, 1.0, places=12)
        self.assertEqual(r.lrVsCs.nDiscordant, 0)

    def test_v_LR_monotonicity(self) -> None:
        """Larger vLr ⇒ ≼_LR has at least as many transitive-closure
        relations ⇒ nComparableBoth with the fixed ≼_cs is non-decreasing."""
        cfg = _light_quark_config(N=8, T=0.4)
        quench = SchwingerQuench(cfg)
        prev = -1
        for v in (0.5, 1.0, 2.0, 8.0):
            r = quench.compareCausalOrders(vLr=v)
            self.assertGreaterEqual(
                r.lrVsCs.nComparableBoth, prev,
                msg=f"comparability decreased going from vLr=prev to vLr={v}")
            prev = r.lrVsCs.nComparableBoth

    def test_record_spectra_forced(self) -> None:
        """compareCausalOrders must force recordSpectra=True regardless of
        the input config."""
        cfg = _light_quark_config(N=6, T=0.2)
        cfg.recordSpectra = False
        r = SchwingerQuench(cfg).compareCausalOrders(vLr=1.0)
        self.assertGreater(r.nLabels, 0)


@unittest.skipUnless(HAVE_QUANTUM, "tessera built without TESSERA_QUANTUM=1")
class TestPipelineWithRecordedSpectra(unittest.TestCase):
    """Confirm that compareCausalOrders stitches the per-snapshot spectra
    into one global majorization poset with cross-time edges."""

    def test_maj_has_cross_time_edges(self) -> None:
        cfg = _light_quark_config(N=8, T=0.4)
        r = SchwingerQuench(cfg).compareCausalOrders(vLr=1.0)
        # majVsCs nonzero comparability proves maj has cross-time relations
        # (only cross-time pairs are comparable in ≼_cs).
        self.assertGreater(r.majVsCs.nConcordant, 0)
