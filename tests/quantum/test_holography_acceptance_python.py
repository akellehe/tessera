"""Remaining §H1 and §H2 acceptance from the holography spec.

* §H1: spatial MI on simple two-qubit states + Schwinger ground-state
  nearest-neighbour MI decay with distance.
* §H2: temporal MI in the heavy-quark limit (the spec's
  "single-qubit unitary" idealisation; here we test the matching
  empirical claim that off-diagonal entries are suppressed when the
  Hamiltonian is dominated by local terms).

Skips cleanly when tessera was built without TESSERA_QUANTUM=1.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

try:
    from tessera.quantum import (
        QuantumConfig, SchwingerModel, MutualInformation,
        TDVPConfig, SchwingerQuench,
    )
    from tessera.quantum.holography import (
        SchwingerParams, ChoiPropagator, ChoiTDVPSettings,
    )
    HAVE_QUANTUM = True
except ImportError:
    HAVE_QUANTUM = False


@unittest.skipUnless(HAVE_QUANTUM, "tessera built without TESSERA_QUANTUM=1")
class TestSpatialMIOnHandBuiltDensityMatrices(unittest.TestCase):
    """Spec §H1 acceptance #1 + #2: Bell-pair I = 2 ln 2; product I = 0."""

    @staticmethod
    def _partial_trace_B(rho_AB: np.ndarray) -> np.ndarray:
        return np.trace(rho_AB.reshape(2, 2, 2, 2), axis1=1, axis2=3)

    @staticmethod
    def _partial_trace_A(rho_AB: np.ndarray) -> np.ndarray:
        return np.trace(rho_AB.reshape(2, 2, 2, 2), axis1=0, axis2=2)

    def _mi(self, rho_AB: np.ndarray) -> float:
        rho_A = self._partial_trace_B(rho_AB)
        rho_B = self._partial_trace_A(rho_AB)
        sA = MutualInformation.von_neumann_entropy(rho_A)
        sB = MutualInformation.von_neumann_entropy(rho_B)
        sAB = MutualInformation.von_neumann_entropy(rho_AB)
        return sA + sB - sAB

    def test_bell_pair_mi_is_two_ln_two(self) -> None:
        bell = (np.array([1, 0, 0, 1], dtype=complex) / math.sqrt(2)).reshape(4, 1)
        rho = bell @ bell.conj().T
        self.assertAlmostEqual(self._mi(rho), 2.0 * math.log(2.0), places=12)

    def test_product_state_mi_is_zero(self) -> None:
        psi = np.array([1, 0, 0, 0], dtype=complex).reshape(4, 1)
        rho = psi @ psi.conj().T
        self.assertAlmostEqual(self._mi(rho), 0.0, places=12)

    def test_singlet_mi_is_two_ln_two(self) -> None:
        singlet = (np.array([0, 1, -1, 0], dtype=complex) / math.sqrt(2)).reshape(4, 1)
        rho = singlet @ singlet.conj().T
        self.assertAlmostEqual(self._mi(rho), 2.0 * math.log(2.0), places=12)


@unittest.skipUnless(HAVE_QUANTUM, "tessera built without TESSERA_QUANTUM=1")
class TestSchwingerGroundStateMIDecay(unittest.TestCase):
    """Spec §H1 acceptance #3: I(site_i : site_{i+k}) decreases
    monotonically in k on a gapped Schwinger ground state."""

    def test_mi_decays_with_distance(self) -> None:
        tdvp = TDVPConfig()
        tdvp.N = 8; tdvp.a = 1.0; tdvp.g = 1.0; tdvp.m = 5.0; tdvp.L0 = 0.0
        tdvp.dmrg_max_bond_dim = 64; tdvp.dmrg_n_sweeps = 12
        tdvp.dmrg_krylov_dim = 4;   tdvp.dmrg_cutoff = 1e-12
        tdvp.i0 = 1; tdvp.d = 1; tdvp.quench_enforce_parity = False
        tdvp.dt = 0.1; tdvp.T = 0.0
        tdvp.snapshot_every = 1
        tdvp.max_bond_dim = 64
        tdvp.quiet = True; tdvp.conserve_qns = True
        tdvp.record_mutual_information = True

        quench = SchwingerQuench(tdvp).evolve()
        snap = quench.snapshots[0]
        N = tdvp.N
        mi = np.array(snap.mutual_information).reshape(N, N)

        def mi_at_distance(k: int) -> float:
            vals = [mi[i, i + k] for i in range(N - k)]
            return sum(vals) / len(vals)

        mi_1 = mi_at_distance(1)
        mi_2 = mi_at_distance(2)
        mi_3 = mi_at_distance(3)
        self.assertGreater(mi_1, mi_2,
            msg=f"NN MI ({mi_1:.4e}) should exceed next-NN MI ({mi_2:.4e})")
        self.assertGreater(mi_2, mi_3,
            msg=f"next-NN MI ({mi_2:.4e}) should exceed range-3 MI ({mi_3:.4e})")


@unittest.skipUnless(HAVE_QUANTUM, "tessera built without TESSERA_QUANTUM=1")
class TestTemporalMIHeavyQuarkLimit(unittest.TestCase):
    """Spec §H2 acceptance #3 (adapted): a single-qubit unitary on
    site i leaves the temporal MI matrix factored, with I(i_in : i_out)
    = 2 ln 2 and all other off-diagonal entries zero. The Schwinger
    Hamiltonian is multi-site, so we test the matching empirical claim
    via the heavy-quark limit: at m ≫ 1/a the diagonal mass term
    dominates and the temporal-MI matrix is nearly diagonal."""

    def test_heavy_quark_choi_is_nearly_diagonal(self) -> None:
        p = SchwingerParams()
        p.N = 4; p.a = 1.0; p.g = 1.0; p.m = 200.0; p.L0 = 0.0
        s = ChoiTDVPSettings()
        s.dt = 0.001; s.max_bond_dim = 64; s.cutoff = 1e-12
        s.krylov_dim = 12; s.quiet = True

        mi = ChoiPropagator.temporal_mutual_information(p, 0.005, s)

        diag = np.diag(mi)
        off  = mi - np.diag(diag)
        for i in range(p.N):
            self.assertGreater(diag[i], 1.3,
                msg=f"diag[{i}] = {diag[i]:.4f}; expected near 2 ln 2 = "
                    f"{2 * math.log(2):.4f}")
        self.assertLess(np.max(np.abs(off)) / np.min(diag), 1e-3,
            msg=f"heavy-quark off-diagonal max = {np.max(np.abs(off)):.4e} "
                f"vs diag min = {np.min(diag):.4f}")


if __name__ == "__main__":
    unittest.main()
