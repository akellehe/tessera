# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""How many move specifications stage 1 draws per unit (#1010).

Stage 1 commits essentially no move on the 3x3 collar (#987), and the cause is
sampling. ``drawRandomMoveSpecification`` picks a KIND uniformly from six --
add, remove, flip, inverse-flip, cone-out, cone-in -- and only then a site
within that kind, so ``DECLARED_CANDIDATE_MOVES = 6`` gives ONE sample per kind
per unit against site sets of order the cell count (57 cells, 36 boundary
facets on the collar). That is roughly 2% coverage, after which the run reports
itself combinatorially stationary.

Measured on the converged ``pinned-cnot`` geometry with the whole-complex leak
scored: 6 draws commit nothing, 50 commit nothing, 200 commit a move that adds
an edge and lowers the objective from 6.773284 to 6.359595; six successive
passes at 200 draws took it to 2.809, with the first move REMOVING edges before
later ones added them back. A candidate costs 73-106ms against 16-25 minutes
for a relaxation unit, so looking harder is nearly free.

``--candidate-moves`` exposes the number. The default is unchanged, so no run
already recorded changes meaning.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))),
    "examples", "cobordism"))

import emergence_animation as ea  # noqa: E402

SMALL = 4


def test_the_default_is_unchanged():
    """Additive: an existing run must draw exactly what it drew before."""
    assert ea.build_config()["candidate_moves"] == ea.DECLARED_CANDIDATE_MOVES
    assert ea.DECLARED_CANDIDATE_MOVES == 6


def test_the_value_is_carried_into_the_config():
    assert ea.build_config(candidate_moves=200)["candidate_moves"] == 200
    assert ea.build_config(candidate_moves=1)["candidate_moves"] == 1


def test_zero_draws_means_every_candidate():
    """Zero is the exhaustive sentinel, not a refusal (#1019).

    It used to be refused, on the reading that "sample nothing" is a run that
    cannot move. Walking the move space gave the number a second, better
    meaning -- "do not sample, take every candidate there is" -- and that is
    what the drive now does with it. A negative count still means nothing.
    """
    assert ea.build_config(candidate_moves=0)["candidate_moves"] == 0
    with pytest.raises(ValueError) as caught:
        ea.build_config(candidate_moves=-5)
    assert "negative" in str(caught.value)


def test_the_drive_asks_stage_one_for_that_many(monkeypatch):
    """The number must reach `run_stage1`, not merely sit in the document.

    Recorded through the engine call itself: a config key that no caller reads
    would pass every other test here and change nothing about a run.
    """
    seen = []
    original = ea.MC.run_stage1

    def spy(self, *args, **kwargs):
        seen.append(kwargs.get("n_candidate_moves"))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(ea.MC, "run_stage1", spy)
    ea.drive(ea.build_config(size=SMALL, steps=1, candidate_moves=17,
                             stage2_iters=1),
             progress=False)
    assert seen and all(value == 17 for value in seen), seen


def test_the_run_document_records_it():
    """A record must say how hard the run looked, or its silence is ambiguous."""
    config = ea.build_config(size=SMALL, steps=1, candidate_moves=200)
    assert config["candidate_moves"] == 200
