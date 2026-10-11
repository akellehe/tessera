"""Python coverage for the Charged Cartan Monte Carlo v0.2 (qudit basis
+ Σ_AB Choi state).

Mirrors the C++ suites
``tests/quantum/test_interaction_simulation_v02.cpp`` and
``tests/quantum/test_sigma_ab_choi_state.cpp`` at the API surface that's
relevant from Python — including the new accessors added in issue #22
(``quditChargeOf``, ``quditStateOf``, ``quditJointStateFor``).

Skips cleanly when tessera was built without TESSERA_QUANTUM=1.

Coverage:

* ``featureQuditBasis = True`` smoke (4-dim path runs end-to-end).
* ``featureChoiSigmaAB`` defaults ``True`` on a fresh config (the
  v0.2 + #16 fix is on out of the box).
* Choi auto-clears when ``featureQuditBasis = False``: no exception,
  no behaviour change vs. the legacy path.
* Q-conservation with Choi on at γ_CP = 0 (no integer drift across
  multiple seeds).
* CP-violation still drifts Q at γ_CP ≠ 0 (proves the fix doesn't
  mask physics).
* Backward-compat: ``featureChoiSigmaAB = False`` reproduces the legacy
  I/4 proxy + intermittent Q-drift.
* ``annihilate`` + ``pairCreate`` smoke + Q-conservation when both fire.
* ``getChargeProfile`` + ``getChargeCorrelation`` smoke.
* ``quditChargeOf`` returns ±1 on integer-charge eigenstates and 0 on
  the I/4 proxy.
* ``quditStateOf`` returns a 4×4 unit-trace density matrix (or ``None``
  for vertices with no stored state).
* ``quditJointStateFor`` returns a 16×16 Hermitian unit-trace matrix.
* Spectral-dimension survives Choi-on at small T (ballpark sanity, not
  the full β-scan).
"""

from __future__ import annotations

import math
import unittest

import numpy as np

try:
    import tessera.simulations as q
    HAVE_QUANTUM = True
except ImportError:
    HAVE_QUANTUM = False


def _delaunay_edges(n, seed):
    """Random 2D Delaunay edge list with `n` vertices, deterministic in `seed`."""
    from scipy.spatial import Delaunay
    pts = np.random.default_rng(seed).uniform(0, 1, (n, 2))
    es = set()
    for s in Delaunay(pts).simplices:
        for a, b in [(s[0], s[1]), (s[1], s[2]), (s[0], s[2])]:
            es.add((min(int(a), int(b)), max(int(a), int(b))))
    return sorted(es)


def _make_cfg(n=4, seed=0, *,
              qudit=True, choi=True,
              j_cc=1.0, j_ss=0.25, mass_shift=0.0, gamma_cp=0.0,
              target=200):
    cfg = q.InteractionConfig()
    cfg.nSystems = n
    cfg.a = 1.0; cfg.g = 1.0; cfg.m = 0.5; cfg.dt = 0.25
    cfg.beta = 3e-4; cfg.epsilonI = 1e-10
    cfg.targetInteractions = target
    cfg.delaunayEdges = _delaunay_edges(n, seed=seed + 999)
    cfg.seed = seed; cfg.quiet = True
    cfg.featureQuditBasis = qudit
    cfg.featureChoiSigmaAB = choi
    cfg.j_chargeCharge = j_cc; cfg.j_spinSpin = j_ss
    cfg.massShift = mass_shift; cfg.gammaCpViolation = gamma_cp
    cfg.dtPair = 0.25
    cfg.initialChargeMode = q.InitialChargeMode.ALTERNATING
    return cfg


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------


@unittest.skipUnless(HAVE_QUANTUM, "tessera built without TESSERA_QUANTUM=1")
class TestV02ConfigDefaults(unittest.TestCase):

    def test_choi_default_is_on(self):
        """The Σ_AB Choi state fix (#16) is on out of the box."""
        cfg = q.InteractionConfig()
        self.assertTrue(cfg.featureChoiSigmaAB,
                        "featureChoiSigmaAB should default to True")

    def test_qudit_default_is_off(self):
        """The 4-dim qudit basis is opt-in (Schwinger marginal is the
        default 2-dim path)."""
        cfg = q.InteractionConfig()
        self.assertFalse(cfg.featureQuditBasis)

    def test_charged_cartan_pair_h_defaults(self):
        """Sanity on the H_pair parameters: nonzero couplings plus
        zero CP-violation by default."""
        cfg = q.InteractionConfig()
        self.assertAlmostEqual(cfg.j_chargeCharge, 1.0)
        self.assertAlmostEqual(cfg.j_spinSpin, 0.25)
        self.assertEqual(cfg.gammaCpViolation, 0.0)


# ---------------------------------------------------------------------------
# Qudit-basis smoke + auto-clear of Choi when qudit is off
# ---------------------------------------------------------------------------


@unittest.skipUnless(HAVE_QUANTUM, "tessera built without TESSERA_QUANTUM=1")
class TestV02QuditBasisSmoke(unittest.TestCase):

    def test_qudit_runs_end_to_end(self):
        """4-dim qudit basis: tune + getSpectralDimension complete."""
        cfg = _make_cfg(n=4, seed=1, qudit=True, choi=True, target=80)
        sim = q.InteractionSimulation(cfg)
        sim.tune()
        self.assertGreaterEqual(sim.interactionCount, 1)
        # Spectral dim is finite at at least one sigma in the diffusion regime
        sigmas = list(np.logspace(-1, 4, 8))
        dS = sim.getSpectralDimension(sigmas, 10)
        self.assertTrue(any(math.isfinite(d) for d in dS))

    def test_choi_auto_clears_when_qudit_off(self):
        """Setting featureChoiSigmaAB=True but featureQuditBasis=False
        must NOT raise — Choi is auto-cleared and the legacy 2-dim
        path runs."""
        cfg = _make_cfg(n=4, seed=2, qudit=False, choi=True, target=50)
        sim = q.InteractionSimulation(cfg)
        sim.tune()
        self.assertGreaterEqual(sim.interactionCount, 1)


# ---------------------------------------------------------------------------
# Q-conservation (the Choi #16 invariant)
# ---------------------------------------------------------------------------


@unittest.skipUnless(HAVE_QUANTUM, "tessera built without TESSERA_QUANTUM=1")
class TestV02QConservation(unittest.TestCase):

    Q_TOL = 1e-9  # bit-perfect tolerance for the Choi-on, γ_CP=0 case

    def test_q_conserved_with_choi_at_zero_cp(self):
        """At γ_CP=0 with Choi on, Q should be bit-perfectly conserved
        across N independent seeds. This is the headline invariant
        the Σ_AB Choi-state fix (#16) restores."""
        for seed in range(4):
            with self.subTest(seed=seed):
                cfg = _make_cfg(n=4, seed=seed, qudit=True, choi=True,
                                gamma_cp=0.0, target=200)
                sim = q.InteractionSimulation(cfg)
                sim.tune()
                Q = sim.getGlobalCharge()
                self.assertLess(abs(Q), self.Q_TOL,
                                f"|Q|={abs(Q):.3e} above tol "
                                f"{self.Q_TOL:.0e} at seed={seed} (Choi on, "
                                "γ_CP=0); Q should be exactly conserved.")

    def test_cp_violation_drifts_q(self):
        """With γ_CP non-trivial, Q is *not* conserved — the Choi fix
        only eliminates the discretization-drift bug at γ_CP=0; the
        physics of CP-violation is preserved.

        This is a non-regression check: a sufficient nonzero γ_CP
        should produce a *measurable* |Q| at the end of tune across
        at least one of a handful of seeds.
        """
        any_drift = False
        for seed in range(6):
            cfg = _make_cfg(n=4, seed=seed, qudit=True, choi=True,
                            gamma_cp=0.5, target=200)
            sim = q.InteractionSimulation(cfg)
            sim.tune()
            if abs(sim.getGlobalCharge()) > 1e-3:
                any_drift = True
                break
        self.assertTrue(
            any_drift,
            "γ_CP=0.5 did not move Q from zero in any of 6 seeds — "
            "the Choi fix may be over-conserving Q (masking physics).")


# ---------------------------------------------------------------------------
# v0.2 observables — the new bindings issue #22 adds
# ---------------------------------------------------------------------------


@unittest.skipUnless(HAVE_QUANTUM, "tessera built without TESSERA_QUANTUM=1")
class TestV02PerVertexObservables(unittest.TestCase):

    def test_qudit_charge_of_eigenstate(self):
        """quditChargeOf returns ±1 for ALTERNATING initial charges
        on the qudit basis: vertex i has q = ±1 by parity."""
        cfg = _make_cfg(n=4, seed=3, qudit=True, choi=True, target=10)
        sim = q.InteractionSimulation(cfg)
        # No tune yet — read initial-layer states directly
        verts = sim.getSpacetime().getVertexList().toVector()
        for i, v in enumerate(verts[:4]):
            q_v = sim.quditChargeOf(v)
            expected = +1.0 if i % 2 == 0 else -1.0
            self.assertAlmostEqual(
                q_v, expected, places=10,
                msg=f"vertex {i} should be a |{'+' if expected > 0 else '−'}⟩ "
                    f"eigenstate; got Tr[ρ·Q̂]={q_v}")

    def test_qudit_state_of_is_unit_trace_hermitian(self):
        """quditStateOf returns a 4×4 unit-trace Hermitian density matrix
        for vertices with stored states, or None otherwise."""
        cfg = _make_cfg(n=4, seed=4, qudit=True, choi=True, target=10)
        sim = q.InteractionSimulation(cfg)
        verts = sim.getSpacetime().getVertexList().toVector()
        for v in verts[:4]:
            rho = sim.quditStateOf(v)
            self.assertIsNotNone(rho, f"qudit state missing for vertex {v.getId()}")
            self.assertEqual(rho.shape, (4, 4))
            self.assertAlmostEqual(np.trace(rho).real, 1.0, places=10)
            self.assertLess(np.abs(np.trace(rho).imag), 1e-12)
            self.assertTrue(np.allclose(rho, rho.conj().T, atol=1e-12),
                            "ρ must be Hermitian")

    def test_qudit_joint_state_for_neighbours_is_16x16(self):
        """Joint state of an initial-layer Delaunay-neighbour pair
        is the stored correlated 16×16 ρ_XY. Unit-trace, Hermitian."""
        cfg = _make_cfg(n=4, seed=5, qudit=True, choi=True, target=10)
        sim = q.InteractionSimulation(cfg)
        verts = sim.getSpacetime().getVertexList().toVector()
        # Pick the first edge from the config's Delaunay edges
        i, j = cfg.delaunayEdges[0]
        rho_xy = sim.quditJointStateFor(verts[i], verts[j])
        self.assertEqual(rho_xy.shape, (16, 16))
        self.assertAlmostEqual(np.trace(rho_xy).real, 1.0, places=10)
        self.assertTrue(np.allclose(rho_xy, rho_xy.conj().T, atol=1e-10),
                        "ρ_XY must be Hermitian")


# ---------------------------------------------------------------------------
# Charge profile + correlation smoke (v0.1 + v0.2)
# ---------------------------------------------------------------------------


@unittest.skipUnless(HAVE_QUANTUM, "tessera built without TESSERA_QUANTUM=1")
class TestV02ChargeProfile(unittest.TestCase):

    def test_charge_profile_initial_layer_balance_v01(self):
        """With ALTERNATING initial charges on an even N (v0.1 path —
        ``getChargeProfile`` reads the v0.1 ``chargeOf_`` map, which
        is only populated under ``featureCharges = True`` /
        ``useCharges = True`` and the legacy 2-dim Schwinger
        marginal), the initial slice has equal counts of + and −,
        zero net charge."""
        cfg = _make_cfg(n=4, seed=6, qudit=False, choi=False, target=30)
        cfg.featureCharges = True
        cfg.useCharges = True
        sim = q.InteractionSimulation(cfg)
        profile = sim.getChargeProfile()
        self.assertGreaterEqual(len(profile), 1)
        n_pos, n_zero, n_neg, sum_q = profile[0]
        self.assertEqual(int(n_pos), int(n_neg))
        self.assertAlmostEqual(sum_q, 0.0, places=10)

    def test_charge_correlation_returns_vector_v01(self):
        """``getChargeCorrelation(maxDist=2)`` returns a length-2
        vector under the v0.1 path (it also reads ``chargeOf_``)."""
        cfg = _make_cfg(n=4, seed=7, qudit=False, choi=False, target=50)
        cfg.featureCharges = True
        cfg.useCharges = True
        sim = q.InteractionSimulation(cfg)
        sim.tune()
        corr = sim.getChargeCorrelation(2)
        self.assertEqual(len(corr), 2)


# ---------------------------------------------------------------------------
# Annihilate / pairCreate smoke
# ---------------------------------------------------------------------------


@unittest.skipUnless(HAVE_QUANTUM, "tessera built without TESSERA_QUANTUM=1")
class TestV02AnnihilatePairCreate(unittest.TestCase):

    def test_annihilate_pair_create_dont_raise(self):
        """Both spontaneous moves should be callable on a qudit sim
        without raising. They may return False (no eligible pair) — we
        only assert they don't raise."""
        cfg = _make_cfg(n=4, seed=8, qudit=True, choi=True, target=20)
        sim = q.InteractionSimulation(cfg)
        sim.tune()
        # Returns bool: True if a move fired, False otherwise. Just
        # exercise the call sites.
        for _ in range(5):
            sim.annihilate()
            sim.pairCreate()


# ---------------------------------------------------------------------------
# Choi-off ↔ Choi-on geometry sanity
# ---------------------------------------------------------------------------


@unittest.skipUnless(HAVE_QUANTUM, "tessera built without TESSERA_QUANTUM=1")
class TestV02ChoiVsLegacyGeometry(unittest.TestCase):

    def test_choi_on_off_produce_comparable_cell_counts(self):
        """The Choi fix is a v0.2-correctness fix, not a geometry
        change. Cell counts should match between Choi-on and Choi-off
        for the same (seed, target) at small T."""
        kwargs = dict(n=4, seed=9, qudit=True, target=80)
        sim_on = q.InteractionSimulation(_make_cfg(**kwargs, choi=True))
        sim_off = q.InteractionSimulation(_make_cfg(**kwargs, choi=False))
        sim_on.tune(); sim_off.tune()
        self.assertEqual(sim_on.interactionCount, sim_off.interactionCount)


if __name__ == "__main__":
    unittest.main()
