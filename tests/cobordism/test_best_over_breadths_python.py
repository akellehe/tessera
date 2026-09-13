# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Best-improver ACROSS the depth ladder, not just within a depth (#1037).

Stage 1 scores every candidate at one depth and commits the lowest delta, so
within a breadth the rule has always been best-improver. Across breadths it was
first-improver: the first depth that lowered the objective at all was taken and
the remaining depths were never priced.

That is right when the shallow depths come first, and wrong under a named
breadth. With a few hundred sampled five-move compositions against a move space
of 828, some composition nearly always improves a little, so the search commits
a mediocre five-move sequence in preference to an excellent single move it never
looked at. Measured on the breadth-3 run: it held an objective an order of
magnitude above the depth-1 search at the same unit count while never once
paying for the exhaustive depth-1 fallback.

`best_over_breadths` prices every depth against the SAME base complex and
commits the lowest delta found at any of them. Pricing has to be separated from
committing for that to be possible at all: a ladder that commits as it goes
cannot compare two depths, because the second would be scored against a complex
the first had already changed.
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

#: Big enough that improving moves exist at more than one depth, so the two
#: policies have something to disagree about.
PRECONE = 10
SMALL = 4


def host(precone=PRECONE, seed=0):
    """A small emergent host with a non-trivial objective.

    `einstein_hilbert=True` matters: with the term off the objective is
    identically zero, nothing commits, and every assertion below would pass
    for the wrong reason.
    """
    return MC(MC.seed_simplex(3), [], [], degrees=[1], seed=seed,
              precone=precone, einstein_hilbert=True)


def objective_after_one_unit(best_over_breadths, breadth=3, draws=40):
    node = host()
    before = node.objective()
    node.run_stage1(max_steps=1, n_candidate_moves=draws,
                    combinatorial_breadth=breadth,
                    best_over_breadths=best_over_breadths)
    return before, node.objective(), node.last_stage1_lookahead


# ---- the policy ----

def test_pricing_every_breadth_never_commits_a_worse_move():
    """The whole claim: the best over the ladder is at least the first on it.

    Both policies search the same schedule from the same complex with the same
    seed, so the best-over-breadths delta cannot be the larger of the two.
    """
    first_before, first_after, _ = objective_after_one_unit(False)
    best_before, best_after, _ = objective_after_one_unit(True)
    assert first_before == best_before, "same starting complex"
    assert best_after <= first_after + 1e-12, (first_after, best_after)


def test_it_still_commits_something():
    """A policy that priced everything and took nothing would pass the test
    above trivially."""
    before, after, depth = objective_after_one_unit(True)
    assert after < before, (before, after)
    assert depth >= 1


def test_the_default_is_unchanged():
    """Additive: an existing run must search exactly what it searched before."""
    assert ea.build_config()["best_over_breadths"] is False
    assert ea.DECLARED_BEST_OVER_BREADTHS is False


# ---- pricing is separate from committing ----

def test_pricing_a_depth_does_not_change_the_complex():
    """The property that makes the comparison possible at all.

    Two depths can only be compared if the first leaves the complex alone, so
    the second is scored against the same base.
    """
    node = host()
    before = sorted(tuple(sorted(v.getId() for v in cell.getVertices()))
                    for cell in node.spacetime().getTopSimplices())
    objective_before = node.objective()
    node.run_stage1(max_steps=0, n_candidate_moves=40)
    after = sorted(tuple(sorted(v.getId() for v in cell.getVertices()))
                   for cell in node.spacetime().getTopSimplices())
    assert after == before and node.objective() == objective_before


# ---- the flag ----

def test_the_value_is_carried_into_the_config():
    assert ea.build_config(best_over_breadths=True)["best_over_breadths"] is True


def test_the_drive_asks_stage_one_for_it(monkeypatch):
    """The flag must reach `run_stage1`, not merely sit in the document."""
    seen = []
    original = ea.MC.run_stage1

    def spy(self, *args, **kwargs):
        seen.append(kwargs.get("best_over_breadths"))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(ea.MC, "run_stage1", spy)
    ea.drive(ea.build_config(size=SMALL, steps=1, best_over_breadths=True,
                             stage2_iters=1),
             progress=False)
    assert seen and all(value is True for value in seen), seen
