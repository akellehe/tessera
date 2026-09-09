# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""A Pachner move must draw its target from the generator it was handed (#1013).

Runs were not reproducible from their seed. Identical fresh processes given the
same seed diverge -- measured during #579 and since treated as inherent to the
engine. It was not inherent. `MultiCobordism` hands each Pachner move a payload
seed and builds a local ``std::mt19937`` from it, but only ``AddMove`` used that
engine to choose its target. ``RemoveMove`` called ``getRandomVertex()`` and
``FlipMove``/``IFlipMove``/``ShiftMove`` called ``getRandomTopSimplex()``, and
those no-argument overloads read ``Spacetime::rng``:

    std::mt19937 rng{std::random_device{}()};

Every candidate is scored on a complex rebuilt from a snapshot, so each got a
fresh ``random_device`` seed and the target came from entropy.

Two things followed beyond irreproducibility. ``step()`` justifies evaluating
candidates in parallel with "applyMoveSpecification is deterministic given its
spec (it seeds a local engine from the payload)", which held only for
``AddMove``; and candidates are deduplicated by ``(kind, payload)``, so for
those kinds two specs sharing a payload were not the same move and a genuine
candidate could be dropped as a duplicate.

These assert the OUTCOME rather than the call: two nodes given the same seed
must take the same stage-1 step. That is the property the engine promises and
the one a caller relies on.
"""
import pytest

from tessera import cobordism as cob

MC = cob.MultiCobordism

#: Enough draws that the step reaches the kinds whose target was unseeded --
#: the draw picks a kind uniformly from six, so a handful might see only Add.
DRAWS = 30

#: Cone-ins applied to the seed simplex before the fixture is handed over.
#:
#: Sized so BOTH properties below are non-vacuous. Measured across six seeds:
#: at precone 4 with 60 draws the sample covers the whole move space and all
#: six seeds reach the SAME outcome, so "different seeds explore differently"
#: cannot be tested there; at precone 10 all six differ, and every run still
#: reproduces itself.
PRECONE = 10


def node(seed, precone=PRECONE):
    """A host with something to minimise.

    The Regge term is ON deliberately. Without an objective F is identically
    zero, no move can lower it by the acceptance tolerance, and NOTHING is ever
    committed -- which would make every reproducibility assertion below pass
    vacuously, agreeing because nothing happened. Measured: with the term off,
    0 moves commit across four seeds; with it on, 2-3 commit and F falls
    (4.88 -> 1.67 at precone 4).
    """
    return MC(MC.seed_simplex(3), [], [], degrees=[1], seed=seed,
              precone=precone, einstein_hilbert=True)


def shape(spacetime):
    """What the complex IS, as counts a move would change."""
    return (len(spacetime.getTopSimplices()),
            len(spacetime.getEdgeList().toVector()),
            len(spacetime.getVertexList().toVector()))


def stepped(seed, draws=DRAWS, steps=3):
    n = node(seed)
    list(n.run_stage1(max_steps=steps, n_candidate_moves=draws, max_lookahead=1))
    # The objective is part of the outcome: two runs that reach the same shape
    # by different moves have not agreed in the way that matters.
    return shape(n.spacetime()), n.accepted_move_count, round(n.objective(), 12)


def test_the_same_seed_takes_the_same_step():
    """The substance of the fix, and what fails without it.

    Two nodes, same seed, same complex, same draw count. Their stage-1 outcome
    must agree. Before the change the three unseeded kinds drew their target
    from `std::random_device` on every freshly built candidate complex, so this
    disagrees at random.
    """
    first = stepped(7)
    second = stepped(7)
    assert first == second, (first, second)


def test_it_holds_across_several_seeds():
    """One seed agreeing could be a complex with only one reachable move."""
    for seed in (0, 1, 13, 99):
        assert stepped(seed) == stepped(seed), seed


def test_different_seeds_still_explore_differently():
    """Reproducible must not mean degenerate.

    A move that ignored its generator entirely would pass the tests above and
    have made the engine deterministic by making it blind, so at least one pair
    of seeds must reach different outcomes.
    """
    outcomes = {stepped(seed) for seed in range(6)}
    assert len(outcomes) > 1, outcomes


def test_a_longer_run_stays_reproducible():
    """Divergence compounds: a single step may agree by luck where ten do not."""
    assert stepped(3, steps=10) == stepped(3, steps=10)


def test_the_complex_still_grows():
    """The fix must not stop moves being found.

    Reproducibility bought by proposing nothing would satisfy every assertion
    above; assert instead that a seeded run still commits moves.
    """
    committed = 0
    for seed in range(4):
        n = node(seed)
        list(n.run_stage1(max_steps=3, n_candidate_moves=DRAWS, max_lookahead=1))
        committed += n.accepted_move_count
    assert committed > 0
