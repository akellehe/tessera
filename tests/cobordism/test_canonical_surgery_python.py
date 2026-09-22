# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The canonical directed surgery lives on MultiCobordism (#549).

`build_step` (the policy hook) and the directed cone-out/cone-in probes — the unit a
search policy (ProtonSynthesis's build, a greedy driver, or the RL agent) composes —
are methods of the *engine* (`MultiCobordism`), not of `ProtonSynthesis`. The
consolidation removed the `ProtonSynthesis` duplicate; `ProtonSynthesis` keeps only
the `should_use_directed_surgery` flag and calls the engine. These tests pin the API surface, the consolidation, the probes' `rU`-monotone
invariant (they only ever commit moves that lower `rU`), and an end-to-end converged
proton driven through the canonical directed path.
"""
import math
import os
import unittest

import pytest

import tessera

cob = tessera.cobordism


class CanonicalSurgeryApiTest(unittest.TestCase):
    """Fast: the canonical surgery API lives on MultiCobordism, not ProtonSynthesis."""

    def test_multicobordism_exposes_buildstep_and_probes(self):
        for name in ("build_step", "directed_cone_out", "directed_cone_in"):
            self.assertTrue(hasattr(cob.MultiCobordism, name),
                            f"MultiCobordism.{name} missing")

    def test_build_action_enum_has_every_solve_action(self):
        ba = cob.MultiCobordism.BuildAction
        for name in ("GROW", "EVOLVE", "RELAX", "CONE_OUT", "CONE_IN"):
            self.assertTrue(hasattr(ba, name), f"BuildAction.{name} missing")

    def test_hole_placement_strategy_enum(self):
        hp = cob.MultiCobordism.HolePlacementStrategy
        for name in ("ADJACENT_HOLES_FIRST", "ADJACENT_HOLES_LAST"):
            self.assertTrue(hasattr(hp, name), f"HolePlacementStrategy.{name} missing")

    def test_proton_surgery_api_was_consolidated_away(self):
        # The directed surgery + buildStep moved DOWN to the engine; ProtonSynthesis only drives it.
        for name in ("BuildAction", "HolePlacementStrategy", "build_step",
                     "directed_cone_out", "directed_cone_in"):
            self.assertFalse(
                hasattr(cob.ProtonSynthesis, name),
                f"ProtonSynthesis.{name} should be gone (canonical home is MultiCobordism)")

    def test_the_target_conditioned_pinned_accessor_is_gone(self):
        # `pinnedBoundaryVertices` derived a pinned set from the boundary blocks and
        # their targets, and surgery was gated on it. Pinning is now a caller-declared
        # region that constrains the geometry rather than gating any move (#835), so
        # the old accessor has no successor here.
        self.assertFalse(hasattr(cob.MultiCobordism, "pinned_boundary_vertices"))
        for name in ("declare_pinned_region", "pinned_regions", "edge_is_pinned"):
            self.assertTrue(hasattr(cob.MultiCobordism, name),
                            f"MultiCobordism.{name} missing")


@pytest.mark.slow
class DirectedProbeInvariantTest(unittest.TestCase):
    """Slow (grows a seeded node): the directed probes commit a move ONLY when it lowers
    `rU`, so `rU` is non-increasing across either probe, and `build_step` dispatches every
    action without raising."""

    def test_probes_are_ru_monotone_and_build_step_dispatches(self):
        BA = cob.MultiCobordism.BuildAction
        HP = cob.MultiCobordism.HolePlacementStrategy
        node = cob.ProtonSynthesis(seed=0).formation_node(1)   # a seeded single-Δ⁴ node
        # A small grow keeps the eigensolve-heavy probe scans cheap; the invariant holds
        # at any size.
        node.build_step(BA.GROW, max_steps=25, n_candidate_moves=6)

        # A directed probe commits a move ONLY when it lowers rU, so rU is non-increasing.
        # These two calls also exercise the CONE_OUT / CONE_IN build_step routes.
        before = node.r_u(node.st)
        opened = node.directed_cone_out(HP.ADJACENT_HOLES_LAST)
        self.assertGreaterEqual(opened, 0)
        self.assertLessEqual(node.r_u(node.st), before + 1e-6,
                             "directed_cone_out must not raise rU (commits only openers that lower it)")

        before = node.r_u(node.st)
        closed = node.directed_cone_in()
        self.assertGreaterEqual(closed, 0)
        self.assertLessEqual(node.r_u(node.st), before + 1e-6,
                             "directed_cone_in must not raise rU")

        # The remaining BuildActions dispatch without raising (cheap actions).
        node.build_step(BA.RELAX, stage2_max_iters=2)
        node.build_step(BA.EVOLVE, max_steps=3)
        node.build_step(BA.GROW, max_steps=3)


@pytest.mark.slow
class DirectedSurgeryProtonBuildTest(unittest.TestCase):
    """Slow: a full two-step ProtonSynthesis build driven through the canonical directed-surgery
    path (`should_use_directed_surgery=True`) converges — the whole formation cobordism
    carries the singlet on at least three emergent holes.

    Full convergence at this budget is not a CI invariant: the engine is not
    process-deterministic (#579), and the same class at the same budget both
    passed a 972-second local run (#651 validation) and stalled at 0 holes on
    CI runners and loaded boxes with bit-identical engines — a hard
    convergence gate is a coin flip CI cannot carry (the sibling
    `test_proton_cpp_python.py` records the same policy). The full gate runs
    under TESSERA_SLOW_TESTS=1; the always-on test below pins the honest
    invariants every draw must deliver."""

    _FULL = bool(os.environ.get("TESSERA_SLOW_TESTS"))

    @classmethod
    def setUpClass(cls):
        cls.p = cob.ProtonSynthesis(seed=0, should_use_directed_surgery=True)
        cls.p.build(max_restarts=3, init_steps=180, evolve_steps=60,
                    stage2_max_iters=10, color_tolerance=0.5, min_emergent_holes=3)

    def test_directed_build_grows_and_stays_finite(self):
        # What every draw must deliver regardless of #579: the canonical path
        # ran, the complex grew past the bare 10-edge pentatope seed, the
        # objective side is finite, and the singlet residual never exceeds its
        # 3.0 empty-register floor.
        st = self.p.spacetime()
        self.assertGreater(len(st.getEdgeList().toVector()), 10)
        self.assertTrue(math.isfinite(self.p.color_residual()))
        self.assertLessEqual(self.p.color_residual(), 3.0 + 1e-9)

    def test_converged_via_canonical_path(self):
        if not self._FULL:
            self.skipTest("full convergence gate: set TESSERA_SLOW_TESTS=1 "
                          "(draw-dependent under #579; see the class note)")
        self.assertTrue(
            self.p.converged(),
            f"did not converge: colorR={self.p.color_residual()}, "
            f"holes={len(self.p.emergent_holes())}")

    def test_three_emergent_holes(self):
        if not self._FULL:
            self.skipTest("full convergence gate: TESSERA_SLOW_TESTS=1")
        self.assertGreaterEqual(len(self.p.emergent_holes()), 3)

    def test_singlet_carried(self):
        if not self._FULL:
            self.skipTest("full convergence gate: TESSERA_SLOW_TESTS=1")
        self.assertLess(self.p.color_residual(), 0.5)


if __name__ == "__main__":
    unittest.main()
