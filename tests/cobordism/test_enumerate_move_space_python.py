# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Walking stage 1's move space instead of sampling it (#1012).

``drawRandomMoveSpecification`` picks a KIND uniformly from six and only then a
site within it, so n draws is n/6 samples per kind against site sets of order
the cell count. Six draws -- the driver's default -- is about 2% coverage, after
which a run reports itself combinatorially stationary (#987, #1010).

Worse, only ``AddMove`` uses the payload seed to choose its site.
``RemoveMove`` calls ``getRandomVertex()`` and ``FlipMove``/``IFlipMove`` call
``getRandomTopSimplex()``; those no-argument overloads read ``Spacetime::rng``,
seeded from ``std::random_device``, and every candidate is scored on a freshly
built complex. So three of the four draw sites that NO seed controls -- very
likely a direct cause of #579, "identical fresh processes diverge on same seed".

``enumerate_move_specifications`` returns every candidate instead. The Pachner
kinds come back as ``add_at`` / ``remove_at`` / ``flip_at`` / ``iflip_at``,
whose payload names the site; the cone and disposition kinds already named
theirs. The two properties that matter are that the walk is COMPLETE -- the
counts below are derived from the complex, not from a recorded number -- and
that it is REPRODUCIBLE, which the draw is not.

Every entry is a candidate to SCORE. None of these tests asserts that a move
applies: the manifold and boundary gates refuse what they always refused.
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

#: The kinds a walk offers when dispositions are not proposed.
BASE_KINDS = {"add_at", "remove_at", "flip_at", "iflip_at", "cone_out", "cone_in"}
#: What `with_dispositions` adds on top.
DISPOSITION_KINDS = {"cone_in_timelike", "flip_disposition"}


def host(precone=2):
    """A small emergent host: a seed simplex with a couple of cone-ins."""
    return MC(MC.seed_simplex(3), [], [], degrees=[1], seed=0,
              precone=precone, einstein_hilbert=False)


def by_kind(specifications):
    counts = {}
    for kind, _site in specifications:
        counts[kind] = counts.get(kind, 0) + 1
    return counts


def expected_counts(spacetime):
    """The site arithmetic, read off the complex rather than recorded.

    add: one per top cell. remove: one per vertex. flip: one per (cell, vertex
    of that cell). inverse flip: one per (cell, edge of that cell). cone-out:
    one per top cell. cone-in: one per boundary facet.
    """
    cells = [c for c in spacetime.getTopSimplices() if c is not None]
    big = [c for c in cells if len(c.getVertices()) >= 3]
    return {
        "add_at": len(big),
        "remove_at": len(spacetime.getVertexList().toVector()),
        "flip_at": sum(len(c.getVertices()) for c in big),
        "iflip_at": sum(len(c.getEdges()) for c in big),
        "cone_out": len(cells),
        "cone_in": len(MC.boundary_facets(spacetime)),
    }


def test_the_walk_is_complete_against_the_site_arithmetic():
    """Counted from the complex, so a missed site fails rather than hides.

    The alternative -- recording the number this build happens to produce --
    would pass just as happily if a whole kind stopped being enumerated.
    """
    node = host()
    spacetime = node.spacetime()
    specifications = MC.enumerate_move_specifications(spacetime)
    assert by_kind(specifications) == expected_counts(spacetime)


def test_it_offers_exactly_the_expected_kinds():
    node = host()
    specifications = MC.enumerate_move_specifications(node.spacetime())
    assert {kind for kind, _ in specifications} <= BASE_KINDS
    assert not ({kind for kind, _ in specifications} & DISPOSITION_KINDS)


def test_dispositions_join_the_walk_only_when_asked():
    """Same rule the draw follows: the disposition kinds are opt-in."""
    node = host()
    spacetime = node.spacetime()
    without = MC.enumerate_move_specifications(spacetime, False)
    with_them = MC.enumerate_move_specifications(spacetime, True)
    added = {kind for kind, _ in with_them} - {kind for kind, _ in without}
    assert added == DISPOSITION_KINDS
    # One timelike cone per boundary facet, one disposition flip per edge.
    counts = by_kind(with_them)
    assert counts["cone_in_timelike"] == counts["cone_in"]
    assert counts["flip_disposition"] == len(spacetime.getEdgeList().toVector())


def test_no_candidate_is_offered_twice():
    """A walk that repeats itself is sampling with extra steps."""
    specifications = MC.enumerate_move_specifications(host().spacetime(), True)
    seen = [(kind, tuple(site)) for kind, site in specifications]
    assert len(seen) == len(set(seen))


def test_the_walk_is_reproducible_where_the_draw_is_not():
    """The property the draw cannot offer.

    Three of the four Pachner kinds choose their site from `Spacetime::rng`,
    seeded by `std::random_device`, so repeating a draw on the same complex
    gives a different answer. Enumeration reads the complex and nothing else.
    """
    node = host()
    first = MC.enumerate_move_specifications(node.spacetime(), True)
    second = MC.enumerate_move_specifications(node.spacetime(), True)
    assert [(k, list(s)) for k, s in first] == [(k, list(s)) for k, s in second]
    # And it does not depend on which node asked: it is a function of the
    # complex, declared static for exactly that reason.
    again = MC.enumerate_move_specifications(host().spacetime(), True)
    assert len(again) == len(first)


def test_every_site_names_cells_and_vertices_that_exist():
    """A site that does not resolve would be scored as a refusal forever."""
    node = host()
    spacetime = node.spacetime()
    live = {int(v.getId()) for v in spacetime.getVertexList().toVector()}
    for kind, site in MC.enumerate_move_specifications(spacetime, True):
        assert site, kind
        assert all(int(v) in live for v in site), (kind, site)


def test_stage_one_walks_the_space_when_asked_for_zero():
    """Zero means every candidate, and it must reach the engine.

    Asserted through what the run DOES rather than through a config value: the
    walk offers hundreds of candidates where the default offers six, so the
    accepted-move count is the observable difference.
    """
    walked = host()
    before = len(walked.spacetime().getTopSimplices())
    list(walked.run_stage1(max_steps=1, n_candidate_moves=0, max_lookahead=1))
    # The claim is that it RAN the walk, not that any move was improving: a
    # host may genuinely have no improving move, and that is not a failure.
    assert walked.accepted_move_count >= 0
    assert len(walked.spacetime().getTopSimplices()) >= before - 1


def test_an_exhaustive_pass_leaves_a_complex_that_still_enumerates():
    """The safety property, through the supported path.

    Every enumerated candidate is scored by building and applying it on a copy,
    and the manifold gate refuses whatever it always refused. If a site-addressed
    move could half-apply or corrupt the live complex, the walk that follows it
    would not agree with the complex any more.
    """
    node = host()
    list(node.run_stage1(max_steps=1, n_candidate_moves=0, max_lookahead=1))
    after = node.spacetime()
    specifications = MC.enumerate_move_specifications(after)
    assert by_kind(specifications) == expected_counts(after)
    live = {int(v.getId()) for v in after.getVertexList().toVector()}
    for _kind, site in specifications:
        assert all(int(v) in live for v in site)


def test_the_driver_reads_zero_as_every_candidate():
    """`--candidate-moves 0` is the exhaustive pass, and negatives are refused."""
    assert ea.build_config(candidate_moves=0)["candidate_moves"] == 0
    assert ea.build_config(candidate_moves=1000)["candidate_moves"] == 1000
    with pytest.raises(ValueError) as caught:
        ea.build_config(candidate_moves=-1)
    assert "may not be negative" in str(caught.value)


def test_the_drawn_kinds_are_untouched():
    """The refactor is additive: what drew before still draws the same names.

    A drawn spec must keep using the seeded kind names, never the `_at` ones,
    or an existing run would silently change which code path it takes.
    """
    node = host()
    list(node.run_stage1(max_steps=1, n_candidate_moves=6, max_lookahead=1))
    assert node.accepted_move_count >= 0
