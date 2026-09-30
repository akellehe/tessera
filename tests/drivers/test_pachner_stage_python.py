# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The combinatorial moves of the recursion's growth step (`pachner_stage`):
a stage-1 search of `MultiCobordism` on each grown level's base under the
joint stationarity objective, the four Pachner kinds alone.

Terms used below:

* a *Pachner move* is one of the four bistellar moves of a 3-complex (the
  1-4 and 4-1 moves on a tetrahedron and a vertex, the 2-3 and 3-2 flips);
  each keeps the complex the manifold it is, with the boundary it has;
* *surgery* is a cone-out or a cone-in, which adds or removes boundary;
* the *joint stationarity objective* is the squared norm of the gradients of
  the Regge action and of the Hodge spectral entropies, whose minimum records
  how nearly both are stationary at one metric.
"""
import itertools

import numpy as np
import pytest

from tessera import cobordism as cob
from tessera.drivers import emergence as em
from tessera.drivers import recursion as R

from tests.drivers import _recursion_run_2026_09_23 as RUN

MC = cob.MultiCobordism
PACHNER_KINDS = {"add_at", "remove_at", "flip_at", "iflip_at"}


def _base():
    config = R.default_config(tolerances=RUN.TOLERANCES)
    cells, z, links, _ = R.level_zero(config)
    return config, cells, z, links


def test_the_stage_runs_as_the_emergence_driver_runs_its_stage_1():
    """The declared unit is the emergence driver's: as many updates, as
    many candidates, as deep, over the same degrees; and the run records
    it."""
    assert R.DECLARED_PACHNER_UPDATES == em.DECLARED_STAGE1_ITERS == 1
    assert R.DECLARED_PACHNER_CANDIDATES == em.DECLARED_CANDIDATE_MOVES
    assert R.DECLARED_PACHNER_DEPTH == em.DECLARED_COMBINATORIAL_DEPTH
    assert R.PACHNER_REGISTER_DEGREES == tuple(em.DECLARED_REGISTER_DEGREES)
    assert R.PACHNER_HODGE_DEGREES == tuple(em.DECLARED_HODGE_DEGREES)
    config = R.default_config()
    assert config["pachner_updates"] == 1
    assert config["pachner_candidates"] == R.DECLARED_PACHNER_CANDIDATES
    assert config["pachner_depth"] == 1 and config["pachner_seed"] == 0
    assert "no cone-out, cone-in or disposition move" in config["pachner_moves"]
    args = R.build_parser().parse_args(
        ["run", "--pachner-updates", "0", "--pachner-candidates", "3",
         "--pachner-depth", "2", "--pachner-seed", "7"])
    assert (args.pachner_updates, args.pachner_candidates, args.pachner_depth,
            args.pachner_seed) == (0, 3, 2, 7)


def test_zero_updates_leave_the_base_as_it_is():
    config, cells, z, links = _base()
    config["pachner_updates"] = 0
    out_cells, out_z, out_links, record = R.pachner_stage(cells, z, links,
                                                          config)
    assert out_cells is cells and out_z is z and out_links is links
    assert record["updates"] == 0 and not record["changed"]
    assert record["before"] == record["after"] == {
        "vertices": 5, "edges": 9, "cells": 2}
    assert "objective_before" not in record


def test_the_stage_returns_a_base_the_next_level_is_built_from():
    """One declared update on the level-0 base: the base that comes back has
    its vertices 0..n-1, one squared length and one link per edge of its
    cells, an objective that did not rise (a move is committed only when it
    lowers the objective), and the next level builds from it. When no move
    was committed the fields come back as they went in."""
    config, cells, z, links = _base()
    out_cells, out_z, out_links, record = R.pachner_stage(cells, z, links,
                                                          config)
    vertices = sorted({v for c in out_cells for v in c})
    assert vertices == list(range(len(vertices)))
    assert all(len(c) == 4 and c == sorted(c) for c in out_cells)
    edges = {e for c in out_cells for e in itertools.combinations(c, 2)}
    assert edges == set(out_z) == set(out_links)
    assert record["updates"] == 1
    assert record["after"] == {"vertices": len(vertices), "edges": len(edges),
                               "cells": len(out_cells)}
    assert np.isfinite(record["objective_before"])
    assert np.isfinite(record["objective_after"])
    assert record["objective_after"] <= record["objective_before"] * (1 + 1e-12)
    assert record["vertex_relabeling"] == {str(v): v for v in range(5)} or \
        record["changed"]
    spacetime, count = R.build_level(out_cells, out_z, out_links)
    assert count == len(vertices)
    if not record["changed"]:
        assert out_cells == [sorted(c) for c in cells]
        for e in z:
            assert out_z[e] == pytest.approx(z[e], rel=1e-12)
            assert out_links[e] == pytest.approx(links[e], rel=1e-12)


def test_the_search_offers_no_surgical_move():
    """With surgery off the walk offers the four Pachner kinds alone, and a
    stage-1 search over every one of them leaves the boundary as it is:
    only a cone changes the boundary."""
    _, cells, z, links = _base()
    spacetime, _ = R.build_level(cells, z, links, sheets=1)
    every = {k for k, _ in MC.enumerate_move_specifications(spacetime)}
    assert "cone_out" in every and "cone_in" in every
    kinds = {k for k, _ in MC.enumerate_move_specifications(spacetime, False,
                                                            False)}
    assert kinds <= PACHNER_KINDS and "add_at" in kinds
    node = MC(spacetime, [], [], [1], 1.0, 0, 0, False)
    assert node.should_propose_surgery
    node.should_propose_surgery = False
    assert not node.should_propose_surgery
    node.set_objective(cob.JointStationarityObjective())
    node.set_hodge_degrees(list(R.PACHNER_HODGE_DEGREES))
    node.set_simulation_mode(MC.SimulationMode.EMERGENCE,
                             MC.EmergenceSubmode.STRICT)
    before = {frozenset(int(v) for v in f)
              for f in MC.boundary_facets(node.spacetime())}
    list(node.run_stage1(max_steps=3, n_candidate_moves=0))
    after = {frozenset(int(v) for v in f)
             for f in MC.boundary_facets(node.spacetime())}
    assert after == before
