"""Re-run the small-N cases through the
:class:`tessera.quantum.SchwingerModel` Python API and confirm the
energies match the C++-side numbers (PLAN.md §5: the wrapper re-runs
the MPO acceptance with numerics unchanged).

Reference values are the (operator-only) ground-state energies produced
by tests/quantum/test_schwinger_spectrum.cpp at the same parameters,
which were themselves cross-checked against dense Eigen ED to 1e-8.

Skips cleanly if tessera.quantum isn't available (TESSERA_QUANTUM=0 build).
"""

from __future__ import annotations

import unittest

try:
    from tessera.quantum import QuantumConfig, SchwingerModel
    HAVE_QUANTUM = True
except ImportError:
    HAVE_QUANTUM = False


# (N, m/g, L0) -> operator-only ground-state energy from
# tests/quantum/test_schwinger_spectrum.cpp at 12-digit precision (the
# C++ test cross-checks each value against dense Eigen ED to <1e-12).
PHASE1_REFERENCE = {
    (4, 0.0,   0.0): -1.738676174000,
    (4, 0.125, 0.0): -1.639323676070,
    (4, 0.25,  0.0): -1.592621519330,
    (6, 0.0,   0.0): -3.432512144510,
    (6, 0.125, 0.0): -3.279435472200,
    (6, 0.25,  0.0): -3.197318242230,
    (8, 0.0,   0.0): -5.629231623340,
    (8, 0.125, 0.0): -5.424944357400,
    (8, 0.25,  0.0): -5.308213565820,
    # L0 = 0.5 corner: exercises the c_n / A_k tail-sum machinery.
    (4, 0.0,   0.5): -1.592621519330,
    (4, 0.25,  0.5): -1.738676174000,
    (6, 0.125, 0.5): -3.279435472200,
}


@unittest.skipUnless(HAVE_QUANTUM, "tessera built without TESSERA_QUANTUM=1")
class TestSchwingerModelSolve(unittest.TestCase):
    def test_phase1_reference_match(self) -> None:
        """Each (N, m/g, L0) in the small-N sweep matches its reference
        operator-only energy to within 1e-6."""
        for (N, m_over_g, L0), e_ref in PHASE1_REFERENCE.items():
            cfg = QuantumConfig()
            cfg.N = N
            cfg.a = 1.0
            cfg.g = 1.0
            cfg.m = m_over_g          # since g=1, m_over_g IS m
            cfg.L0 = L0
            cfg.max_bond_dim = 64
            cfg.n_sweeps = 8

            result = SchwingerModel(cfg).solve()
            self.assertAlmostEqual(
                result.operator_energy,
                e_ref,
                places=8,
                msg=f"N={N} m/g={m_over_g} L0={L0}: got {result.operator_energy} expected {e_ref}",
            )
            # Sanity: energy = operatorEnergy + constant by construction.
            self.assertAlmostEqual(
                result.energy,
                result.operator_energy + result.constant,
                places=12,
            )
            # bondDim should never exceed what we asked for.
            self.assertLessEqual(result.bond_dim, cfg.max_bond_dim)

    def test_n20_runs_and_returns_diagnostics(self) -> None:
        """N=20 from the PLAN.md §5 spec — verify the run completes,
        returns a sane bondDim, and the operator energy + constant
        identity holds."""
        cfg = QuantumConfig()
        cfg.N = 20
        cfg.a = 1.0
        cfg.g = 1.0
        cfg.m = 0.0
        cfg.L0 = 0.0
        cfg.max_bond_dim = 100
        cfg.n_sweeps = 12

        r = SchwingerModel(cfg).solve()
        # Sign and order of magnitude: at this scale the GS is well below 0.
        self.assertLess(r.operator_energy, -10.0)
        self.assertGreater(r.bond_dim, 0)
        self.assertLessEqual(r.bond_dim, cfg.max_bond_dim)
        self.assertAlmostEqual(r.energy, r.operator_energy + r.constant, places=12)

    def test_default_config_rejected(self) -> None:
        """SchwingerModel.solve must reject a default-constructed config
        (N=0 is invalid)."""
        cfg = QuantumConfig()
        with self.assertRaises(Exception):
            SchwingerModel(cfg).solve()

    def test_config_is_readable(self) -> None:
        cfg = QuantumConfig()
        cfg.N = 8
        m = SchwingerModel(cfg)
        self.assertEqual(m.config.N, 8)
