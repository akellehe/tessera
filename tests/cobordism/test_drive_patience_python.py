# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""A stalled unit is not a finished run (#1003).

``emergence_animation.py run`` stopped the moment ONE engine unit failed to
improve the objective by the tolerance, and reported it as
``tolerance-reached``. That name reads like an achievement and the exit is a
stall: it says the drive stopped moving, never that the objective reached any
particular value. The pinned gate solves all stopped that way at unit 3 with
1997 units of budget left, on an improvement of exactly 0.0, at residuals from
4.4e-16 to 1.9e-1 -- the terminator does not distinguish them.

``--patience N`` requires N CONSECUTIVE stalled units before the drive stops,
and any unit that improves resets the count. Stage 1 draws its candidate moves
at random, so a unit that commits nothing is one unlucky draw and the next unit
draws again; the knob buys those extra draws. The default is 1, which is the
long-standing behaviour exactly, so no run already recorded changes meaning.
``DriveResult.stalls`` records how many consecutive stalls ended the run, which
the terminator alone cannot say.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))),
    "examples", "cobordism"))

import emergence_animation as ea  # noqa: E402

#: A host small enough that a unit is cheap; the assertions here are about the
#: loop's counting, not about any geometry it drives.
SMALL = 4

#: No unit can improve the objective by this much, so EVERY unit stalls.
IMPOSSIBLE = 1e30

#: Any change at all beats this, so a unit that moves the geometry improves.
TRIVIAL = 1e-30


def test_the_default_is_one_which_is_the_long_standing_behaviour():
    """Backwards compatibility, asserted rather than assumed.

    The flag is only safe to add because its default reproduces the old exit
    exactly. If the default ever moved, every existing run document's
    terminator would quietly start meaning something else.
    """
    assert ea.DECLARED_PATIENCE == 1
    assert ea.build_config()["patience"] == 1
    assert ea.build_config(patience=5)["patience"] == 5


def test_a_patience_below_one_is_refused_by_name():
    """Zero means "never stop on a stall", which the drive cannot do.

    Reading it as 1 would run the opposite of what the caller asked for, so
    it is refused where it is written rather than clamped where it is used.
    """
    with pytest.raises(ValueError) as caught:
        ea.build_config(patience=0)
    assert "at least 1" in str(caught.value)
    with pytest.raises(ValueError):
        ea.build_config(patience=-3)


def test_the_default_still_stops_on_the_first_stalled_unit():
    """The pre-existing exit, unchanged, and now with its count recorded."""
    config = ea.build_config(size=SMALL, steps=4, tolerance=IMPOSSIBLE)
    result = ea.drive(config, progress=False)
    assert result.terminator == ea.Terminator.TOLERANCE
    assert result.frames[-1].step == 1
    assert result.stalls == 1


def test_patience_runs_past_a_stalled_unit_and_stops_at_the_declared_count():
    """The substance of the flag.

    Every unit stalls here, so the run stops at exactly the unit where the
    consecutive count reaches the patience -- not before, which would ignore
    the flag, and not later, which would ignore the tolerance.
    """
    config = ea.build_config(size=SMALL, steps=8, tolerance=IMPOSSIBLE,
                             patience=3)
    result = ea.drive(config, progress=False)
    assert result.terminator == ea.Terminator.TOLERANCE
    assert result.frames[-1].step == 3
    assert result.stalls == 3


def test_a_patience_the_budget_cannot_reach_ends_on_the_budget():
    """Both limits are real, and the budget is the one that binds here.

    A patience above the unit count must not stop the run early, and must not
    relabel a budget exhaustion as a stall.
    """
    config = ea.build_config(size=SMALL, steps=2, tolerance=IMPOSSIBLE,
                             patience=50)
    result = ea.drive(config, progress=False)
    assert result.terminator == ea.Terminator.STEPS
    assert result.frames[-1].step == 2
    # Every unit stalled; none of them ended the run.
    assert result.stalls == 2


def test_an_improving_unit_leaves_no_stall_behind():
    config = ea.build_config(size=SMALL, steps=2, tolerance=TRIVIAL,
                             patience=2)
    result = ea.drive(config, progress=False)
    assert result.terminator == ea.Terminator.STEPS
    assert result.stalls == 0


def test_the_count_is_consecutive_and_an_improving_unit_resets_it(monkeypatch):
    """A run that stalls, recovers and stalls again is still making progress.

    Scripted through ``_converged`` -- the predicate the loop asks -- because
    no engine configuration produces a chosen stall pattern on demand, and the
    claim under test is about the loop's arithmetic, not about the numbers fed
    to it. A cumulative count would end this run at unit 3, on the strength of
    its history rather than of anything true of its geometry.
    """
    script = iter([True, False, True, True, True])
    monkeypatch.setattr(ea, "_converged",
                        lambda before, after, tolerance: next(script))
    config = ea.build_config(size=SMALL, steps=5, tolerance=TRIVIAL,
                             patience=2)
    result = ea.drive(config, progress=False)
    assert result.terminator == ea.Terminator.TOLERANCE
    # Units 3 and 4 are the first CONSECUTIVE pair; unit 1's stall was reset
    # by unit 2 and does not count toward them.
    assert result.frames[-1].step == 4
    assert result.stalls == 2


def test_the_run_document_records_the_patience_and_the_stall_count():
    """A reader of a record must be able to tell the two stops apart.

    A run that stopped on its first stalled unit and one that stopped after
    twenty carry the same terminator, so the count has to be written down.
    """
    config = ea.build_config(size=SMALL, steps=4, tolerance=IMPOSSIBLE,
                             patience=2)
    result = ea.drive(config, progress=False)
    document = {"config": config, "terminator": result.terminator,
                "stalls": result.stalls}
    assert document["config"]["patience"] == 2
    assert document["stalls"] == 2
    assert document["terminator"] == ea.Terminator.TOLERANCE
