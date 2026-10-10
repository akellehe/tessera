"""Pytest wrappers for the C++ quantum acceptance executables.

Each test invokes a standalone C++ binary built when ``TESSERA_QUANTUM=ON``
is passed to CMake. The binaries are the source of truth for what passes
— these wrappers only gate the executables behind pytest discovery and
check the return code.

Skips cleanly when the quantum subsystem is not built.
"""

from __future__ import annotations

import subprocess
import unittest
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _find_executable(name: str) -> Path | None:
    # Search the build dirs that scikit-build-core (cmake-build-debug/<wheel_tag>/),
    # the conftest.py rebuild path, and a manual cmake invocation (build-quantum/)
    # might write to.
    patterns = [
        f"build-quantum/{name}",
        f"cmake-build-debug/{name}",
        f"cmake-build-debug/*/{name}",
        f"cmake-build-*/*/{name}",
        f"cmake-build-*/{name}",
        f"build/*/{name}",
    ]
    for p in patterns:
        for hit in REPO_ROOT.glob(p):
            if hit.is_file():
                return hit
    return None


class TestQuantumExecutables(unittest.TestCase):
    def _run(self, exe_name: str, must_contain: str, timeout: int) -> None:
        exe = _find_executable(exe_name)
        if exe is None:
            self.skipTest(
                f"{exe_name} not found — build with `cmake -DTESSERA_QUANTUM=ON` "
                "or set the env var TESSERA_QUANTUM=1 before running pytest"
            )
        result = subprocess.run(
            [str(exe)],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=REPO_ROOT,
        )
        self.assertEqual(
            result.returncode,
            0,
            msg=f"{exe_name} exited {result.returncode}\n"
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )
        self.assertIn(must_contain, result.stdout)

    def test_itensor_heisenberg_smoke(self) -> None:
        """Heisenberg N=8 DMRG vs dense ED, agree to <1e-6."""
        self._run("test_itensor_hello", "PASS", timeout=60)

    def test_schwinger_mpo_vs_dense_ed(self) -> None:
        """SchwingerMPO vs dense ED on the Sz=0 sector for N=4,6,8 ×
        m/g ∈ {0, 0.125, 0.25}, plus N=20 smoke test."""
        self._run("test_schwinger_spectrum", "ALL PASS", timeout=180)

    def test_schwinger_analytic_limits(self) -> None:
        """Analytic-limit checks on SchwingerMPO — free-fermion
        half-filling at g=m=0, and strong-coupling vacuum at m≫g."""
        self._run("test_schwinger_limits", "ALL PASS", timeout=120)

    @pytest.mark.slow
    def test_schwinger_continuum_trends(self) -> None:
        """Continuum trend ω_0 → -1/π (Bañuls fig. 6), vector mass gap
        (Bañuls table), chiral condensate benchmark observable. Slow —
        geometric scan over x with finite-N DMRG runs scaling as N ~ 20·√x."""
        self._run("test_schwinger_paper", "ALL PASS", timeout=600)

    def test_majorization_predicate(self) -> None:
        """Unit tests on the majorization predicate / poset: reflexivity,
        transitivity, antisymmetry, the canonical (1,0)≻(½,½) strict
        relation, and Hasse-diagram transitive reduction on a small
        synthetic chain."""
        self._run("test_majorization", "ALL PASS", timeout=30)

    def test_schmidt_spectra(self) -> None:
        """Schmidt-spectrum extraction on hand-checkable MPSes: product
        |↑↑↑↑⟩ gives spectrum (1), N-qubit GHZ gives (½,½) at every
        contiguous cut, Bell |Φ⁺⟩ and singlet give (½,½) at the center
        cut, all spectra sum to 1."""
        self._run("test_schmidt_spectra", "ALL PASS", timeout=60)

    def test_majorization_poset(self) -> None:
        """Full pipeline Schmidt::allOf → Majorization::posetOf on
        product, GHZ, and Bell-vs-product inputs, matching the three
        acceptance criteria from PLAN.md §5."""
        self._run("test_majorization_poset", "ALL PASS", timeout=60)

    def test_schwinger_schmidt_cross_check(self) -> None:
        """Cross-check: MPS-side Schmidt spectra against dense ED of the
        Schwinger Hamiltonian for small N — verifies the spectra
        extraction works on a non-trivial physical state, not just
        product/GHZ/Bell."""
        self._run("test_schwinger_schmidt_cross_check", "ALL PASS", timeout=120)

    @pytest.mark.slow
    def test_tdvp_flux_tube(self) -> None:
        """Heavy-quark q-qbar flux-tube preservation under 2-site TDVP.
        d=5, T=5.0 (matches the plan's "T = d·a" prescription with d
        odd for parity). Marked slow because it runs ~100 TDVP sweeps
        at N=14."""
        self._run("test_tdvp_string", "ALL PASS", timeout=300)

    def test_schwinger_n4_hamiltonian(self) -> None:
        """PLAN.md §7 trap: independent symbolic evaluation of the
        H_m + H_E formula on every N=4 basis state, compared to
        SchwingerHamiltonian::denseMatrix to machine precision. Catches
        L_n² expansion errors that MPO-vs-dense would miss."""
        self._run("test_schwinger_n4_hamiltonian", "ALL PASS", timeout=30)

    def test_tdvp_vs_dense_unitary(self) -> None:
        """Cross-check TDVP integrator against full e^{-iHt} dense
        unitary evolution on the 2^N Hilbert space, for N ≤ 8 across
        heavy / light / massless regimes. Guarantees the TDVP runner
        reproduces exact dynamics to better than 1e-4 over genuinely
        time-dependent profiles (not just preserved plateaus)."""
        self._run("test_tdvp_vs_dense", "ALL PASS", timeout=120)

    def test_causal_order_comparison(self) -> None:
        """Build the three partial orders (maj/LR/cs) on a TDVP snapshot
        history and verify the comparison statistics are sensible
        (Kendall-τ in [-1, 1], no NaNs, vLr-monotonicity holds)."""
        self._run("test_causal_compare", "ALL PASS", timeout=180)

    def test_causet_chain_extraction(self) -> None:
        """Causet::chainFrom on hand-built spacetimes: a trivial chain of
        single-vertex slices collapses to the 1D hopping list (0,1), (1,2),
        …; a branching antichain and the other small cases of
        docs/source/quantum-plan.md §6 give the documented site counts and
        hopping pairs."""
        self._run("test_causet_chain", "ALL PASS", timeout=120)

    def test_cdt_causet_invariants(self) -> None:
        """Structural invariants of Poset::fromSpacetime on a foliated CDT:
        every timelike edge joins adjacent slices, so every Hasse cover spans
        exactly one slice and the covers coincide with the chain-of-antichains
        hopping pairs."""
        self._run("test_cdt_causet_invariants", "ALL PASS", timeout=120)

    def test_chain_causet_mpo(self) -> None:
        """SchwingerHamiltonian::mpoChain reproduces SchwingerHamiltonian::mpo
        on a chain causet: the ground-state energies of the two MPOs agree to
        DMRG noise for a synthetic hopping list and for a chain extracted from
        a toy spacetime (N ≤ 8), and malformed hopping lists are rejected."""
        self._run("test_chain_causet_mpo", "ALL PASS", timeout=300)

    def test_interaction_simulation_smoke(self) -> None:
        """End-to-end smoke test of InteractionSimulation: the randomised
        mixed-state initial layer, interaction and un-interaction moves, the
        spectral dimension, the acceptance rate and the action after tune()
        are all finite and in range."""
        self._run("test_interaction_simulation", "PASS", timeout=300)

    def test_interaction_simulation_charged_cartan_v01(self) -> None:
        """The charged Cartan Monte Carlo (v0.1) additions to
        InteractionSimulation: charge bookkeeping, the sign-bucketed
        frontier, the annihilate and pairCreate moves, the observables, and
        their interplay with interact and unInteract."""
        self._run("test_interaction_simulation_v01", "PASS", timeout=300)

    def test_interaction_simulation_qudit_basis_v02(self) -> None:
        """The v0.2 qudit-basis additions to InteractionSimulation
        (docs/source/quantum-experiments/charged_cartan_monte_carlo_v0.2.md)."""
        self._run("test_interaction_simulation_v02", "PASS", timeout=300)

    def test_mps_site_insertion_and_three_site_gate(self) -> None:
        """The MPS mechanics an interaction event needs: insertSite splices a
        product-state site between two sites and preserves the norm, the new
        site's purity and the correlations across the cut; apply3SiteGate
        with the identity is a no-op and with an entangler preserves the norm
        and creates mutual information."""
        self._run("test_mps_site_insertion", "PASS", timeout=120)

    def test_poset_from_spacetime(self) -> None:
        """Poset::fromSpacetime on hand-crafted spacetimes: a two-slice
        ladder where every cross-slice timelike edge is a cover, a
        three-slice chain whose slice-skipping edge the transitive reduction
        removes, and the empty spacetime giving the empty poset."""
        self._run("test_poset_from_spacetime", "ALL PASS", timeout=120)

    def test_schmidt_invariants_dmrg(self) -> None:
        """Universal invariants of the Schmidt spectra of Schwinger-model DMRG
        ground states across a parameter sweep at N=6: every spectrum is
        non-negative, sums to one, and has the rank the cut allows."""
        self._run("test_schmidt_invariants_dmrg", "ALL PASS", timeout=300)

    def test_sigma_ab_choi_state(self) -> None:
        """Σ_AB as the 256-dimensional Choi state of the interaction unitary
        (tessera issue 16): the pinned v0.2 baseline and the behaviour after
        the fix, with the single-vertex marginal and the joint content
        resolving to the same quantum object."""
        self._run("test_sigma_ab_choi_state", "PASS", timeout=120)
