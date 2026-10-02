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

MC = cob.MultiCobordism
PACHNER_KINDS = {"add_at", "remove_at", "flip_at", "iflip_at"}


def _base():
    config = R.default_config()
    cells, z, links, _ = R.level_zero(config)
    return config, cells, z, links


def test_the_stage_runs_as_the_emergence_driver_runs_its_stage_1():
    """The declared unit is the emergence driver's: as many updates, as deep,
    over the same degrees, but over every candidate move of the base rather
    than a drawn sample, so no candidate count and no seed is declared; and
    the run records it."""
    assert R.DECLARED_PACHNER_UPDATES == em.DECLARED_STAGE1_ITERS == 1
    assert R.DECLARED_PACHNER_DEPTH == em.DECLARED_COMBINATORIAL_DEPTH
    assert R.PACHNER_REGISTER_DEGREES == tuple(em.DECLARED_REGISTER_DEGREES)
    assert R.PACHNER_HODGE_DEGREES == tuple(em.DECLARED_HODGE_DEGREES)
    assert not hasattr(R, "DECLARED_PACHNER_CANDIDATES")
    assert not hasattr(R, "DECLARED_PACHNER_SEED")
    config = R.default_config()
    assert config["pachner_updates"] == 1 and config["pachner_depth"] == 1
    assert "pachner_candidates" not in config and "pachner_seed" not in config
    assert "no cone-out, cone-in or disposition move" in config["pachner_stage"]
    # the drive of every relaxation scores Pachner moves as well
    assert config["pachner_moves"] is True
    assert "no sample and no seed" in config["pachner_stage"]
    args = R.build_parser().parse_args(
        ["run", "--pachner-updates", "0", "--pachner-depth", "2"])
    assert (args.pachner_updates, args.pachner_depth) == (0, 2)
    for option in ("--pachner-candidates", "--pachner-seed"):
        with pytest.raises(SystemExit):
            R.build_parser().parse_args(["run", option, "3"])


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


# ------------------------------------------- the growth step's objective


def _grown_level():
    from tests.drivers import _recursion_levels_2026_10_01 as LEVELS
    return (LEVELS.TICK1_CELLS, dict(LEVELS.GROWN_SQUARED_LENGTHS),
            dict(LEVELS.GROWN_LINKS))


def _objective_before(name, squared, links):
    cells, _, _ = _grown_level()
    config = R.default_config(pachner_objective=name)
    record = R.pachner_stage(cells, squared, links, config)[3]
    assert record["objective_name"] == name
    return record["objective_before"]


def test_the_growth_objective_is_declared_and_is_the_joint_action_s():
    """The objective of the growth step is a declared option: the joint
    action's stationarity, which the level's relaxation descends, unless the
    engine's joint stationarity objective is named; another name has no
    value."""
    assert R.PACHNER_OBJECTIVES == ("joint-action", "engine")
    assert R.default_config()["pachner_objective"] == "joint-action"
    args = R.build_parser().parse_args(["run"])
    assert args.pachner_objective == "joint-action"
    args = R.build_parser().parse_args(
        ["run", "--pachner-objective", "engine"])
    assert args.pachner_objective == "engine"
    with pytest.raises(ValueError, match="the growth step's objective is "
                                         "one of joint-action, engine"):
        R.default_config(pachner_objective="regge")


def test_the_growth_step_draws_its_candidates_when_a_count_is_declared():
    """With no count the growth step scores every candidate move of the
    base; with ``--pachner-candidate-moves`` N it draws N at every update,
    each a sequence of the search's depth, and says so in its record. A
    negative count has no meaning."""
    cells, squared, links = _grown_level()
    assert R.default_config()["pachner_candidate_moves"] == 0
    every = R.pachner_stage(cells, squared, links, R.default_config())[3]
    assert every["candidate_moves"] == 0
    assert every["candidates"].startswith("every candidate move")
    args = R.build_parser().parse_args(
        ["run", "--pachner-candidate-moves", "6", "--pachner-depth", "3"])
    config = R.default_config(
        pachner_candidate_moves=args.pachner_candidate_moves,
        pachner_depth=args.pachner_depth)
    drawn = R.pachner_stage(cells, squared, links, config)[3]
    assert (drawn["candidate_moves"], drawn["depth"]) == (6, 3)
    assert drawn["candidates"].startswith("6 candidates drawn at random")
    assert drawn["objective_before"] == pytest.approx(
        every["objective_before"], rel=1e-13)
    with pytest.raises(ValueError, match="candidates the growth step draws"):
        R.default_config(pachner_candidate_moves=-1)


def test_the_joint_action_objective_is_the_level_s_residual_norm():
    """Under ``joint-action`` the growth step scores a base with the norm of
    the joint action's stationarity residual over the level's sheets, nothing
    held: on the level grown at tick 0 of the run of 2026-10-01 it is the
    residual norm the level's relaxation starts from, 33.4178200400092."""
    cells, squared, links = _grown_level()
    value = _objective_before("joint-action", squared, links)
    assert value == pytest.approx(33.4178200400092, rel=1e-12)
    level, count = R.build_level(cells, squared, links)
    base = R.sheet_base(level, count)
    system, _ = R.level_system(base, R.default_config(), [], R.SHEETS)
    assert value == pytest.approx(
        np.linalg.norm(system.point(base).relaxation.residual()), rel=1e-13)


def test_the_joint_action_objective_does_not_turn_on_a_rounding_or_a_gauge():
    """The level grown at tick 0 has squared lengths real to rounding on
    eight of its ten edges. The joint action's Regge term is read on the
    continued sheet, so the objective is the same number whatever the signs
    of those imaginary parts are, and it is the same under a gauge
    transformation of the links whose moduli are not one. The engine's
    objective on the same level takes values from 2.91 to 8.97 over the
    same signs."""
    cells, squared, links = _grown_level()
    real = {edge: complex(value.real,
                          value.imag if abs(value.imag) > 1e-10 else 0.0)
            for edge, value in squared.items()}
    reference = _objective_before("joint-action", real, links)
    engine = _objective_before("engine", real, links)
    spread = []
    for edge, value in real.items():
        if value.imag != 0.0:
            continue
        for sign in (1e-16, -1e-16):
            moved = dict(real)
            moved[edge] = complex(value.real, sign)
            assert _objective_before("joint-action", moved, links) == \
                pytest.approx(reference, rel=1e-12)
            spread.append(_objective_before("engine", moved, links))
    assert max(spread) - min(spread) > 1.0 and min(spread) < engine

    rng = np.random.default_rng(5)
    gauge = {vertex: np.exp(0.7 * rng.normal()
                            + 1j * rng.uniform(0.0, 2.0 * np.pi))
             for vertex in range(5)}
    gauged = {(a, b): gauge[a] * link / gauge[b]
              for (a, b), link in links.items()}
    assert _objective_before("joint-action", real, gauged) == \
        pytest.approx(reference, rel=1e-11)
