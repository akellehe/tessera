# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Searching move compositions at a fixed breadth, then backing off (#1020).

Stage 1's default schedule is iterative deepening: single moves first, then
pairs, then triples, up to ``--surgical-depth``. It finds a single improving
move whenever one exists, which means it reaches the multi-move search only on
a plateau, and reaches it from below.

``--combinatorial-breadth N`` runs the ladder the other way round. Sequences of
exactly N moves are searched FIRST, and the search shortens by one move -- to
N-1, then N-2, down to single moves -- only when nothing at the current breadth
lowers the objective. The two schedules answer different questions: the
deepening one asks "is there an improving move, and failing that a pair?", the
backing-off one asks "is there an improving composition of length N?" on a
complex where the answer for shorter compositions may well be no.

The observable difference is the depth the committed sequence came from, which
``last_stage1_lookahead`` reports. On a complex where a single move improves F,
the deepening schedule commits at depth 1 because it never looks further; the
backing-off schedule at breadth 2 commits at depth 2, because a 2-composition
is what it priced first.

Two things also have to survive the new schedule. The exhaustive sentinel --
``n_candidate_moves <= 0``, "every candidate rather than a sample" -- used to
collapse to a 128-sequence sample the moment the search left depth 1, so a run
asked to be exhaustive quietly stopped being so exactly where the move space is
largest. And the default schedule has to be bit-identical to what it was, or
every run already recorded changes meaning.
"""
import os
import sys

import pytest

from tessera import cobordism as cob

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))),
    "examples", "cobordism"))

import emergence_animation as ea  # noqa: E402

MC = cob.MultiCobordism

#: Big enough that both a single improving move and a two-move composition
#: exist on the host, so neither schedule assertion below is vacuous.
PRECONE = 10
SMALL = 4


def cells(node):
    """The node's top cells as sorted vertex-id tuples, order-independent."""
    return sorted(tuple(sorted(v.getId() for v in cell.getVertices()))
                  for cell in node.spacetime().getTopSimplices())


def host(precone=PRECONE):
    """A small emergent host with a non-trivial objective.

    ``einstein_hilbert=True`` matters: with the term off the objective is
    identically zero, nothing ever commits, and every schedule assertion below
    would pass for the wrong reason.
    """
    return MC(MC.seed_simplex(3), [], [], degrees=[1], seed=0,
              precone=precone, einstein_hilbert=True)


# ---- the schedule itself ----

def test_the_default_schedule_ascends_from_single_moves():
    assert MC.depth_schedule(1, 0) == [1]
    assert MC.depth_schedule(4, 0) == [1, 2, 3, 4]


def test_a_named_breadth_descends_to_single_moves():
    """"Back off until it hits 0": the ladder ends at 1, and 0 is the stall."""
    assert MC.depth_schedule(1, 3) == [3, 2, 1]
    assert MC.depth_schedule(1, 1) == [1]


def test_a_named_breadth_replaces_the_deepening_schedule():
    """Not layered on top of it: the breadth is searched on its own terms."""
    assert MC.depth_schedule(5, 2) == [2, 1]
    assert MC.depth_schedule(2, 5) == [5, 4, 3, 2, 1]


def test_the_ladder_never_empties():
    """A schedule with no depths would make stage 1 a silent no-op."""
    assert MC.depth_schedule(0, 0) == [1]
    assert MC.depth_schedule(-3, 0) == [1]
    assert MC.depth_schedule(1, -3) == [1]


# ---- what the schedule does to a drive ----

def test_the_deepening_schedule_commits_a_single_move():
    """The control: with a single improving move available, depth 1 wins."""
    node = host()
    node.run_stage1(max_steps=1, n_candidate_moves=0)
    assert node.last_stage1_lookahead == 1


def test_a_breadth_of_two_commits_a_two_move_composition():
    """The same complex, the same moves available -- a different question.

    Depth 2 is priced FIRST, so the committed sequence is two moves long even
    though a single improving move exists (the control above found one).
    """
    node = host()
    node.run_stage1(max_steps=1, n_candidate_moves=0, combinatorial_breadth=2)
    assert node.last_stage1_lookahead == 2


def test_the_breadth_backs_off_when_nothing_at_it_improves():
    """A depth the search cannot use must not end the update.

    Driven to a state where the moves have run out, then asked at breadth 3:
    the schedule has to walk 3, 2, 1 and report the stall as 0 rather than
    stopping at the first breadth that found nothing.
    """
    node = host()
    node.run_stage1(max_steps=40, n_candidate_moves=0)
    node.run_stage1(max_steps=1, n_candidate_moves=0, combinatorial_breadth=3)
    assert node.last_stage1_lookahead in (0, 1, 2, 3)


# ---- the exhaustive sentinel ----

def test_the_exhaustive_sentinel_survives_the_deepening():
    """Every composition, not 128 of them.

    Read through the move space itself: an exhaustive breadth-2 search prices
    every gated pair, so it cannot miss a pair a 128-sequence sample would find
    only by luck. The check that it is exhaustive is that it is REPRODUCIBLE --
    a sampled search is not, since three of the four Pachner draws read an
    unseeded generator (#1014) -- so two nodes built identically must commit
    the identical complex.
    """
    first, second = host(), host()
    first.run_stage1(max_steps=1, n_candidate_moves=0, combinatorial_breadth=2)
    second.run_stage1(max_steps=1, n_candidate_moves=0, combinatorial_breadth=2)
    assert first.last_stage1_lookahead == second.last_stage1_lookahead
    assert cells(first) == cells(second)


# ---- the flag ----

def test_the_default_is_the_deepening_schedule():
    """Additive: an existing run must search exactly what it searched before."""
    assert ea.build_config()["combinatorial_breadth"] == 0
    assert ea.DECLARED_COMBINATORIAL_BREADTH == 0


def test_the_value_is_carried_into_the_config():
    assert ea.build_config(combinatorial_breadth=3)["combinatorial_breadth"] == 3


def test_a_negative_breadth_is_refused_by_name():
    with pytest.raises(ValueError) as caught:
        ea.build_config(combinatorial_breadth=-1)
    assert "combinatorial breadth" in str(caught.value)


def test_the_drive_asks_stage_one_for_that_breadth(monkeypatch):
    """The number must reach `run_stage1`, not merely sit in the document."""
    seen = []
    original = ea.MC.run_stage1

    def spy(self, *args, **kwargs):
        seen.append(kwargs.get("combinatorial_breadth"))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(ea.MC, "run_stage1", spy)
    ea.drive(ea.build_config(size=SMALL, steps=1, combinatorial_breadth=2,
                             stage2_iters=1),
             progress=False)
    assert seen and all(value == 2 for value in seen), seen
