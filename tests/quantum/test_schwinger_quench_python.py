"""Python tests: TDVPConfig validation, end-to-end
:meth:`SchwingerQuench.evolve` through the Python API, energy
conservation, flux-tube formation, snapshot schedule. Mirrors the
C++ test_tdvp_string.cpp at the binding layer.

Skips cleanly when tessera was built without TESSERA_QUANTUM=1.
"""

from __future__ import annotations

import unittest

try:
    from tessera.quantum import (
        TDVPConfig,
        TDVPSnapshot,
        QuenchResult,
        SchwingerQuench,
    )
    HAVE_QUANTUM = True
except ImportError:
    HAVE_QUANTUM = False


def _heavy_quark_config(
    N: int = 14, m: float = 20.0, i0: int = 5, d: int = 5,
    T: float | None = None, dt: float = 0.05,
    snapshotEvery: int = 5, maxBondDim: int = 80,
    recordSpectra: bool = False, recordPoset: bool = False,
) -> "TDVPConfig":
    cfg = TDVPConfig()
    cfg.N = N
    cfg.a = 1.0; cfg.g = 1.0; cfg.m = m; cfg.L0 = 0.0
    cfg.dmrg_max_bond_dim = 32; cfg.dmrg_n_sweeps = 10
    cfg.i0 = i0; cfg.d = d
    cfg.dt = dt
    cfg.T = T if T is not None else d * cfg.a
    cfg.max_bond_dim = maxBondDim
    cfg.cutoff = 1e-10
    cfg.krylov_dim = 12
    cfg.snapshot_every = snapshotEvery
    cfg.record_spectra = recordSpectra
    cfg.record_poset = recordPoset
    return cfg


def _heavy_quark_vacuum_L(N: int) -> list[float]:
    """L_n profile of the heavy-quark Néel |↑↓↑↓ … ⟩ at L0=0."""
    return [-1.0 if (n % 2 == 1) else 0.0 for n in range(1, N)]


def _expected_flux_tube_L(N: int, i0: int, d: int) -> list[float]:
    v = _heavy_quark_vacuum_L(N)
    for n in range(i0, i0 + d):
        v[n - 1] += 1.0
    return v


@unittest.skipUnless(HAVE_QUANTUM, "tessera built without TESSERA_QUANTUM=1")
class TestTDVPConfigDefaults(unittest.TestCase):
    def test_defaults_have_invalid_N(self) -> None:
        cfg = TDVPConfig()
        self.assertEqual(cfg.N, 0)
        with self.assertRaises(Exception):
            SchwingerQuench(cfg).evolve()

    def test_field_round_trip(self) -> None:
        cfg = _heavy_quark_config(N=12, m=10.0, i0=3, d=5, T=2.0, dt=0.1)
        self.assertEqual(cfg.N, 12)
        self.assertEqual(cfg.i0, 3)
        self.assertEqual(cfg.d, 5)
        self.assertAlmostEqual(cfg.T, 2.0)
        self.assertAlmostEqual(cfg.dt, 0.1)
        self.assertEqual(cfg.dmrg_n_sweeps, 10)
        self.assertEqual(cfg.snapshot_every, 5)


@unittest.skipUnless(HAVE_QUANTUM, "tessera built without TESSERA_QUANTUM=1")
class TestParityValidation(unittest.TestCase):
    def test_even_i0_rejected(self) -> None:
        cfg = _heavy_quark_config(i0=4, d=5)
        cfg.quench_enforce_parity = True
        with self.assertRaises(Exception):
            SchwingerQuench(cfg).evolve()

    def test_even_d_rejected(self) -> None:
        cfg = _heavy_quark_config(i0=3, d=4)
        cfg.quench_enforce_parity = True
        with self.assertRaises(Exception):
            SchwingerQuench(cfg).evolve()

    def test_parity_bypass_runs(self) -> None:
        cfg = _heavy_quark_config(N=12, m=20.0, i0=4, d=4, T=0.4, dt=0.1,
                                  snapshotEvery=2)
        cfg.quench_enforce_parity = False
        r = SchwingerQuench(cfg).evolve()
        self.assertGreater(len(r.snapshots), 0)


@unittest.skipUnless(HAVE_QUANTUM, "tessera built without TESSERA_QUANTUM=1")
class TestFluxTube(unittest.TestCase):
    def test_initial_post_quench_profile(self) -> None:
        cfg = _heavy_quark_config(N=14, m=20.0, i0=5, d=5,
                                  T=5.0, dt=0.05,
                                  snapshotEvery=5)
        r = SchwingerQuench(cfg).evolve()
        s0 = r.snapshots[0]
        ref = _expected_flux_tube_L(cfg.N, cfg.i0, cfg.d)
        for n, (v, vref) in enumerate(zip(s0.l_profile, ref), start=1):
            self.assertLess(
                abs(v - vref), 0.05,
                msg=f"link {n}: got {v}, expected {vref}",
            )

    def test_mid_run_flux_tube_preserved(self) -> None:
        cfg = _heavy_quark_config(N=14, m=20.0, i0=5, d=5,
                                  T=5.0, dt=0.05, snapshotEvery=5)
        r = SchwingerQuench(cfg).evolve()
        mid = r.snapshots[len(r.snapshots) // 2]
        ref = _expected_flux_tube_L(cfg.N, cfg.i0, cfg.d)
        for n, (v, vref) in enumerate(zip(mid.l_profile, ref), start=1):
            self.assertLess(
                abs(v - vref), 0.05,
                msg=f"t={mid.time} link {n}: got {v}, expected {vref}",
            )

    def test_energy_conservation(self) -> None:
        cfg = _heavy_quark_config(N=14, m=20.0, i0=5, d=5,
                                  T=5.0, dt=0.05, snapshotEvery=5)
        r = SchwingerQuench(cfg).evolve()
        E0 = r.snapshots[0].energy
        Eend = r.snapshots[-1].energy
        rel = abs((Eend - E0) / E0)
        self.assertLess(rel, 1e-3, msg=f"|ΔE|/|E0| = {rel}")

    def test_total_charge_conserved(self) -> None:
        cfg = _heavy_quark_config(N=14, m=20.0, i0=5, d=5,
                                  T=2.0, dt=0.1, snapshotEvery=5)
        r = SchwingerQuench(cfg).evolve()
        for snap in r.snapshots:
            total_sz = 0.5 * sum(snap.z_profile)
            self.assertLess(abs(total_sz), 1e-8,
                            msg=f"t={snap.time}: total Sz = {total_sz}")


@unittest.skipUnless(HAVE_QUANTUM, "tessera built without TESSERA_QUANTUM=1")
class TestSnapshotSchedule(unittest.TestCase):
    def test_n_snapshots_matches_schedule(self) -> None:
        cfg = _heavy_quark_config(N=10, m=20.0, i0=3, d=3,
                                  T=1.0, dt=0.1, snapshotEvery=2)
        r = SchwingerQuench(cfg).evolve()
        self.assertEqual(len(r.snapshots), 6)
        times = [s.time for s in r.snapshots]
        self.assertAlmostEqual(times[0], 0.0)
        self.assertAlmostEqual(times[-1], 1.0)
        for i in range(1, len(times) - 1):
            self.assertGreater(times[i], times[i - 1])

    def test_initial_snapshot_at_zero(self) -> None:
        cfg = _heavy_quark_config(N=10, T=0.5, dt=0.1, snapshotEvery=10)
        r = SchwingerQuench(cfg).evolve()
        self.assertAlmostEqual(r.snapshots[0].time, 0.0)


@unittest.skipUnless(HAVE_QUANTUM, "tessera built without TESSERA_QUANTUM=1")
class TestObservableRecording(unittest.TestCase):
    def test_default_no_spectra(self) -> None:
        cfg = _heavy_quark_config(N=8, i0=1, d=3, T=0.2, dt=0.1,
                                  snapshotEvery=2,
                                  recordSpectra=False, recordPoset=False)
        r = SchwingerQuench(cfg).evolve()
        self.assertEqual(len(r.snapshots[0].spectra.intervals), 0)
        self.assertEqual(r.snapshots[0].poset.get_node_count, 0)

    def test_record_spectra_populates(self) -> None:
        cfg = _heavy_quark_config(N=8, i0=1, d=3, T=0.2, dt=0.1,
                                  snapshotEvery=2,
                                  recordSpectra=True, recordPoset=False)
        r = SchwingerQuench(cfg).evolve()
        for snap in r.snapshots:
            self.assertEqual(len(snap.spectra.intervals), 35)
            self.assertEqual(snap.poset.get_node_count, 0)

    def test_record_poset_populates(self) -> None:
        cfg = _heavy_quark_config(N=8, i0=1, d=3, T=0.2, dt=0.1,
                                  snapshotEvery=2,
                                  recordSpectra=True, recordPoset=True)
        r = SchwingerQuench(cfg).evolve()
        for snap in r.snapshots:
            self.assertEqual(snap.poset.get_node_count, 35)

    def test_schmidt_spectra_normalization_invariant(self) -> None:
        """PLAN.md §9: every Schmidt spectrum sums to 1 even after TDVP
        truncation."""
        cfg = _heavy_quark_config(N=8, i0=1, d=3, T=1.0, dt=0.1,
                                  snapshotEvery=2,
                                  recordSpectra=True, recordPoset=False)
        cfg.m = 0.5
        cfg.cutoff = 1e-10
        r = SchwingerQuench(cfg).evolve()
        for snap in r.snapshots:
            for spec, iv in zip(snap.spectra.spectra,
                                snap.spectra.intervals):
                total = sum(spec)
                self.assertAlmostEqual(
                    total, 1.0, places=8,
                    msg=f"t={snap.time} cut [{iv.i},{iv.j}] sums to {total}")


@unittest.skipUnless(HAVE_QUANTUM, "tessera built without TESSERA_QUANTUM=1")
class TestRepr(unittest.TestCase):
    def test_snapshot_repr(self) -> None:
        cfg = _heavy_quark_config(N=8, i0=1, d=3, T=0.1, dt=0.1,
                                  snapshotEvery=1)
        r = SchwingerQuench(cfg).evolve()
        text = repr(r.snapshots[0])
        self.assertIn("TDVPSnapshot", text)
        self.assertIn("time=", text)
        self.assertIn("energy=", text)
        self.assertIn("bondDim=", text)
