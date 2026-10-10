# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""Stage-1 incremental delta must agree with the recomputed objective (#607).

``MultiCobordism.run_stage1`` decides whether to commit each surgical move by
comparing an *incremental* objective delta against a tolerance, and accumulates
those same deltas into the trace it returns. Two invariants follow, and both are
asserted here because the acceptance rule is only meaningful if they hold:

  * **Non-negativity.** ``objective()`` is
    ``reggeActionGradient + gamma * rU``. The first term is
    ``sum over edges of |actionGradientExact_e|^2`` (via ``std::norm``) and the
    second is a sum of least-squares residuals, so the objective cannot be
    negative. A negative value means the reported quantity is not the objective.
  * **Agreement.** The accumulated trace must equal ``objective()`` recomputed
    from scratch, to machine precision. If it does not, moves are being accepted
    or rejected on a number that does not reflect the real change in the
    objective.

The host must have MORE THAN ONE top cell. The defect does not appear on the
bare single-simplex seed (``precone = 0``): with one top cell the drift is at
machine precision, and only multi-cell hosts expose it. ``precone`` pre-grows the
seed by that many gated cone-in moves, so ``precone = 2`` gives three top cells.
"""

import unittest

import tessera

cob = tessera.cobordism

# The drift under test is quantized in the thousands, so any honest tolerance
# separates it from floating-point noise by many orders of magnitude. Scaled by
# the objective's own magnitude, since it is not a small number.
RELATIVE_TOLERANCE = 1e-9


def _joint_node(seed, precone):
    """A joint MultiCobordism node on a host pre-grown to `precone + 1` cells."""
    return cob.ProtonIngredients(seed=seed, precone=precone).joint_node(seed)


def _top_cell_count(st):
    return sum(1 for s in st.get_simplices() if len(s.get_vertices()) == 5)


class TestStage1DeltaObjectiveAgreement(unittest.TestCase):
    """The accumulated stage-1 trace tracks the recomputed objective."""

    def _drive_and_check(self, seed, precone, calls=4, steps=15):
        node = _joint_node(seed, precone)
        st = node.st
        self.assertGreater(
            _top_cell_count(st), 1,
            "this regression needs a multi-top-cell host; precone = 0 does not "
            "exercise the defect",
        )

        for call_index in range(calls):
            trace = node.run_stage1(max_steps=steps, n_candidate_moves=8,
                                    grow_boundaries=True)
            objective = node.objective()

            self.assertGreaterEqual(
                objective, 0.0,
                f"seed {seed} precone {precone} call {call_index}: objective() is "
                f"||grad S||^2 + gamma * r_U, a sum of non-negative terms, so it "
                f"cannot be negative; got {objective}",
            )
            self.assertGreaterEqual(
                trace[-1], 0.0,
                f"seed {seed} precone {precone} call {call_index}: the accumulated "
                f"stage-1 trace reports the objective and so cannot be negative; "
                f"got {trace[-1]}",
            )
            self.assertAlmostEqual(
                trace[-1], objective,
                delta=RELATIVE_TOLERANCE * max(abs(objective), 1.0),
                msg=(f"seed {seed} precone {precone} call {call_index}: the "
                     f"accumulated incremental delta disagrees with the "
                     f"recomputed objective by {trace[-1] - objective}. The same "
                     f"delta is the move-acceptance test, so moves are being "
                     f"committed against a quantity that is not the objective."),
            )

    def test_agreement_on_a_three_cell_host(self):
        self._drive_and_check(seed=1, precone=2)

    def test_agreement_on_a_five_cell_host(self):
        self._drive_and_check(seed=1, precone=4)

    def test_agreement_across_seeds(self):
        for seed in (2, 3):
            with self.subTest(seed=seed):
                self._drive_and_check(seed=seed, precone=2, calls=3)

    def test_single_simplex_seed_is_already_clean(self):
        """Guards the claim that precone = 0 is not affected.

        Documents why the tests above must precone: if this ever starts failing,
        the defect has spread to the bare seed and the diagnosis in #607 needs
        revisiting.
        """
        node = _joint_node(seed=1, precone=0)
        self.assertEqual(_top_cell_count(node.st), 1,
                         "precone = 0 must leave exactly one 4-simplex")
        for _ in range(3):
            trace = node.run_stage1(max_steps=15, n_candidate_moves=8,
                                    grow_boundaries=True)
            objective = node.objective()
            self.assertAlmostEqual(
                trace[-1], objective,
                delta=RELATIVE_TOLERANCE * max(abs(objective), 1.0))


if __name__ == "__main__":
    unittest.main()
