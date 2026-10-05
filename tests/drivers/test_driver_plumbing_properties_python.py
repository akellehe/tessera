# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Properties of the drivers as programs: `tessera.drivers.recursion`,
`tessera.drivers.baryon_poles`, `tessera.drivers.isospin_doublet` and
`tessera.drivers.cell_solve`.

Each test states a property and checks it at the declared tolerances. The
properties are of three kinds.

* Options: a value given on the command line reaches the declaration or the
  call that consumes it, on every path of a run (the level's relaxation, the
  cells' solves, the Section 15 box, the growth step, the reads), and a
  value the run cannot use is refused before anything is computed.
* Records and files: what a run writes is the record it holds, the files are
  JSON, and a tick that stops early is written and reported like any other.
* State: a run leaves the configuration it is given as it is, and a second
  read is the first.

The cells' reads are replaced by the stand-in of
`tests.drivers.test_recursion_driver_cli_python` wherever a tick is run,
because a real read solves every declared content of every cell; the level's
relaxation, the box, the interaction stage, the grown-cell rule and the growth
step are the real ones. A solve is stopped where `cell_solve.solve` is called,
with what it was handed kept, wherever a test reads the options a solve
receives.
"""
import argparse
import copy
import json
import math
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

from tessera import cobordism as cob
from tessera import observables as obs
from tessera.drivers import baryon_poles as bp
from tessera.drivers import cell_solve as cs
from tessera.drivers import isospin_doublet as ISO
from tessera.drivers import recursion as R
from tests.drivers import _recursion_run_2026_09_23 as RUN
from tests.drivers.test_baryon_poles_cli_python import _doublet_read
from tests.drivers.test_recursion_driver_cli_python import READ

HALF = str(bp.SPIN_HALF)
THREE = str(bp.SPIN_THREE_HALVES)

#: A value for every tolerance of the registry, each different from every
#: other and from the declared 1e-15, so that a tolerance that lands in
#: another's place is seen.
SENTINEL_TOLERANCES = {key: 10.0 ** -(3.0 + 0.1 * k)
                       for k, (key, _) in enumerate(bp.TOLERANCES)}
SENTINEL_LIMITS = {"iteration_limit": 7, "update_limit": 5,
                   "time_limit_seconds": 2.5}
SENTINEL_SOLVE = {"direction_order": 3, "band_reference": "host",
                  "pachner_moves": False, "combinatorial_depth": 1,
                  "combinatorial_length": 2, "candidate_moves": 4,
                  "admissibility_gate": True}
#: A cosmological constant different from the declared zero.
SENTINEL_COSMOLOGICAL_CONSTANT = -0.45


class _Reached(Exception):
    """Raised by a stand-in where a solve or a library call would start."""


def _sentinel_arguments():
    """The command line that sets every option of the registries to its
    sentinel, and the action and the mean field to non-default values."""
    argv = ["run", "--quiet", "--kappa", "0.7", "--beta", "1.3",
            "--villain-order", "4",
            "--cosmological-constant", repr(SENTINEL_COSMOLOGICAL_CONSTANT),
            "--band-selection", "sort-every-iterate",
            "--fiber-pinning", "power-sums", "--fiber-moments", "bands",
            "--trace-terms", "--direction-order", "3", "--band-reference",
            "host", "--no-pachner-moves", "--combinatorial-length", "2",
            "--candidate-moves", "4", "--admissibility-gate",
            "--iteration-limit", "7",
            "--update-limit", "5", "--time-limit-seconds", "2.5"]
    for key, value in SENTINEL_TOLERANCES.items():
        argv += ["--" + key.replace("_", "-"), repr(value)]
    return argv


def _config_of(driver, argv, monkeypatch):
    """The config `driver.main` builds from a command line, the drive itself
    replaced by a stand-in that keeps the config and computes nothing."""
    seen = {}

    def drive(config, **options):
        seen["config"] = copy.deepcopy(config)
        return {"config": config, "host": {"monopole_numbers": []},
                "points": [], "ticks": [], "stopped": False}

    monkeypatch.setattr(driver, "drive", drive)
    driver.main(argv)
    return seen["config"]


def _first_cell_config(config, monkeypatch):
    """The config `recursion.cell_reads` hands the read of the first cell of
    the declared level, with the level's cells and fields."""
    seen = []

    def scan(kappa, beta, cell, on_content=None):
        seen.append((kappa, beta, dict(cell)))
        return {"failed_contents": [], "flagged_contents": [],
                "contents": [], "ratios": {}, "pole_table": {}}

    cells, z, links, _ = R.level_zero(config)
    with monkeypatch.context() as patch:
        patch.setattr(bp, "scan_point", scan)
        R.cell_reads(cells, z, links, config)
    return seen[0], (cells, z, links)


# ---------------------------------------------------------------- options


@pytest.mark.parametrize("driver", [bp, R], ids=["baryon_poles", "recursion"])
def test_the_command_line_without_options_is_the_declared_config(
        driver, monkeypatch):
    """The parser's defaults and `default_config`'s are one set of values:
    the config a bare ``run`` builds is `default_config()` key for key,
    with the environment of the run beside it."""
    config = _config_of(driver, ["run", "--quiet"], monkeypatch)
    config.pop("environment", None)
    assert config == driver.default_config()


def test_the_two_drivers_declare_the_same_host():
    assert R.DECLARED_EDGE_SQUARED == bp.DECLARED_EDGE_SQUARED == 8.0
    assert R.DECLARED_MONOPOLE == bp.DECLARED_MONOPOLE == 1
    assert R.SHEETS == bp.SHEETS == 3
    assert R.DECLARED_KAPPA in bp.DECLARED_KAPPAS
    assert R.DECLARED_BETA in bp.DECLARED_BETAS


@pytest.mark.parametrize("driver", [bp, R], ids=["baryon_poles", "recursion"])
def test_every_registry_key_on_the_command_line_is_recorded_at_its_value(
        driver, monkeypatch):
    """Every tolerance, limit and solve option given on the command line is
    in the config at the value given, under its own key."""
    config = _config_of(driver, _sentinel_arguments(), monkeypatch)
    assert {key: config[key] for key, _ in bp.TOLERANCES} == \
        SENTINEL_TOLERANCES
    assert {key: config[key] for key, _, _ in bp.LIMITS} == SENTINEL_LIMITS
    for key, value in SENTINEL_SOLVE.items():
        assert config[key] == value, key
    assert config["villain_order"] == 4
    assert config["cosmological_constant"] == SENTINEL_COSMOLOGICAL_CONSTANT
    assert config["band_selection"] == "sort-every-iterate"
    assert config["fiber_pinning"] == "power-sums"
    assert config["fiber_moments"] == "bands"
    assert config["trace_terms"] is True


def _solve_options_are_the_sentinels(options):
    assert options["tolerance"] == SENTINEL_TOLERANCES["step_tolerance"]
    assert options["move_tolerance"] == SENTINEL_TOLERANCES["move_tolerance"]
    assert options["direction_order"] == 3
    assert options["moves"] is False
    assert options["combinatorial_depth"] == 1
    assert options["combinatorial_length"] == 2
    assert options["candidate_moves"] == 4
    assert options["iteration_limit"] == 7
    assert options["update_limit"] == 5
    assert options["time_limit_seconds"] == 2.5

    class Node:
        """What `node_configuration` sets on the engine's node."""

    node = Node()
    options["configure"](node)
    assert node.admissibility_tolerance == \
        SENTINEL_TOLERANCES["admissibility_tolerance"]
    assert node.admissibility_gate is True
    # no stiffness and no pinned region is declared on the node by default
    assert set(vars(node)) == {"admissibility_gate",
                               "admissibility_tolerance"}


def test_the_command_line_reaches_the_level_relaxation(monkeypatch):
    """The options of a run reach the drive of the level's relaxation: the
    arguments `cell_solve.solve` is called with, the geometry declaration of
    the system and the action it is posed on."""
    config = _config_of(R, _sentinel_arguments(), monkeypatch)
    cells, z, links, _ = R.level_zero(config)
    spacetime, count = R.build_level(cells, z, links)
    seen = {}

    def solve(base, system, **options):
        support = cs.sheeted_support(base, system.sheets)
        seen["options"] = options
        seen["geometry"] = system._geometry_of(support)
        seen["action"] = system._declare(support.spacetime)
        raise _Reached()

    monkeypatch.setattr(cs, "solve", solve)
    with pytest.raises(_Reached):
        R.relax_level(spacetime, config, [], count=count)
    _solve_options_are_the_sentinels(seen["options"])
    assert seen["geometry"].rank_tolerance == \
        SENTINEL_TOLERANCES["rank_tolerance"]
    assert seen["geometry"].record_terms is True
    action = seen["action"]
    assert action.gravitational_weight == 1.0 / 0.7
    assert action.holonomy_weight == 1.3
    assert action.villain_order == 4
    assert action.regge_hinges == cob.ReggeHinges.Interior
    assert action.cosmological_constant == SENTINEL_COSMOLOGICAL_CONSTANT


def test_the_command_line_reaches_every_cell_solve(monkeypatch):
    """The options of a run reach the solve of a cell's content: the cell's
    config carries every registry key at the run's value, and the arguments
    of `cell_solve.solve`, the mean-field declaration, its geometry
    declaration, the action and the band reference are the run's."""
    config = _config_of(R, _sentinel_arguments() + [
        "--contents", "1", "1", "1", "--eliminate", "lengths"], monkeypatch)
    (kappa, beta, cell), _ = _first_cell_config(config, monkeypatch)
    assert (kappa, beta) == (0.7, 1.3)
    carried = ([key for key, _ in bp.TOLERANCES]
               + [key for key, _, _ in bp.LIMITS]
               + [key for key, _, _ in bp.SOLVE_OPTIONS]
               + ["villain_order", "cosmological_constant",
                  "band_selection", "fiber_moments",
                  "fiber_pinning", "trace_terms", "elimination",
                  "regge_hinges", "contents"])
    assert {key: cell[key] for key in carried} == \
        {key: config[key] for key in carried}
    seen = {}

    def solve(base, system, **options):
        support = cs.sheeted_support(base, system.sheets)
        seen["options"] = options
        seen["mean_field"] = system._declaration(support)
        seen["action"] = system._declare(support.spacetime)
        seen["band_reference"] = system.band_reference
        raise _Reached()

    monkeypatch.setattr(cs, "solve", solve)
    with pytest.raises(_Reached):
        bp.relax_content((1, 1, 1), kappa, beta, cell)
    _solve_options_are_the_sentinels(seen["options"])
    mean_field = seen["mean_field"]
    assert mean_field.band_tolerance == SENTINEL_TOLERANCES["band_tolerance"]
    assert mean_field.tolerance == SENTINEL_TOLERANCES["mean_field_tolerance"]
    assert mean_field.band_selection == cob.BandSelection.SortEveryIterate
    assert mean_field.fiber_constraint_form == \
        cob.FiberConstraintForm.PowerSums
    # "bands": one power sum per occupied band of the content (1, 1, 1)
    assert mean_field.fiber_moments == 3
    assert mean_field.geometry.rank_tolerance == \
        SENTINEL_TOLERANCES["rank_tolerance"]
    assert mean_field.geometry.record_terms is True
    assert seen["band_reference"] == "host"
    action = seen["action"]
    assert action.gravitational_weight == 1.0 / 0.7
    assert action.holonomy_weight == 1.3
    assert action.villain_order == 4
    assert action.cosmological_constant == SENTINEL_COSMOLOGICAL_CONSTANT


def test_the_cosmological_constant_reaches_the_growth_step(monkeypatch):
    """``--cosmological-constant`` reaches the joint action the growth
    step's search scores a base with (the declared ``joint-action``
    objective, the stationarity of the level's action), and the stage's
    record names it; the engine's objective has no such term, and its
    record says so."""
    config = _config_of(R, _sentinel_arguments() + [
        "--pachner-updates", "1"], monkeypatch)
    seen = {}

    def cell_node(spacetime, objective, register_degrees=(1,)):
        system = objective._system
        support = cs.sheeted_support(spacetime, system.sheets)
        seen["action"] = system._declare(support.spacetime)
        raise _Reached()

    monkeypatch.setattr(cs, "cell_node", cell_node)
    cells, z, links, _ = R.level_zero(config)
    with pytest.raises(_Reached):
        R.pachner_stage(cells, z, links, config)
    assert seen["action"].cosmological_constant == \
        SENTINEL_COSMOLOGICAL_CONSTANT
    assert seen["action"].gravitational_weight == 1.0 / 0.7

    def run_stage1(node, **options):
        return []

    monkeypatch.setattr(cob.MultiCobordism, "run_stage1", run_stage1)
    _, _, _, record = R.pachner_stage(
        cells, z, links, dict(config, pachner_objective="engine"))
    assert record["cosmological_constant"] == SENTINEL_COSMOLOGICAL_CONSTANT
    assert record["cosmological_term_in_objective"] is False


@pytest.mark.parametrize("driver", [bp, R], ids=["baryon_poles", "recursion"])
def test_without_the_cosmological_constant_nothing_carries_it(
        driver, monkeypatch):
    """Without ``--cosmological-constant`` the config has no key for it and
    every action the drivers declare carries Lambda = 0, the term absent;
    a value that is not a finite number is refused by name."""
    config = _config_of(driver, ["run", "--quiet"], monkeypatch)
    assert "cosmological_constant" not in config
    assert bp.declared_cosmological_constant(config) == 0.0
    cells, z, links, _ = R.level_zero(R.default_config())
    spacetime, count = R.build_level(cells, z, links)
    system, _ = R.level_system(R.sheet_base(spacetime, count),
                               R.default_config(), [], R.SHEETS)
    assert system._declare(spacetime).cosmological_constant == 0.0
    for text in ("nan", "inf", "-inf", "one"):
        with pytest.raises(SystemExit):
            driver.build_parser().parse_args(
                ["run", "--cosmological-constant", text])
    with pytest.raises(ValueError, match="cosmological constant"):
        bp.default_config(cosmological_constant=math.inf)


def test_the_command_line_reaches_the_box_and_the_growth_step(monkeypatch):
    """The resolutions, the fiber rank and the box's two tolerances reach
    the level recursion's declaration, and the growth step's updates, depth
    and length and the engine's two tolerances reach its stage-1 search."""
    config = _config_of(R, _sentinel_arguments() + [
        "--resolutions", "1", "2", "--band-rank", "2", "--pachner-updates",
        "3", "--pachner-length", "2"], monkeypatch)
    seen = {}

    def over_pencil(operator, metric, dimension, declaration):
        seen["box"] = (list(declaration.resolutions),
                       declaration.bands.band_rank, declaration.tolerance,
                       declaration.rank_tolerance)
        raise _Reached()

    monkeypatch.setattr(cob.LevelRecursion, "overPencil",
                        staticmethod(over_pencil))
    with pytest.raises(_Reached):
        R.recursion_turn(np.diag([1.0, 2.0, 3.0, 4.0]).astype(complex),
                         config)
    assert seen["box"] == ([1.0, 2.0], 2,
                           SENTINEL_TOLERANCES["recursion_tolerance"],
                           SENTINEL_TOLERANCES["quotient_rank_tolerance"])

    def run_stage1(node, **options):
        seen["growth"] = (options, node.move_tolerance,
                          node.admissibility_tolerance)
        return []

    monkeypatch.setattr(cob.MultiCobordism, "run_stage1", run_stage1)
    cells, z, links, _ = R.level_zero(config)
    R.pachner_stage(cells, z, links, config)
    options, move, admissibility = seen["growth"]
    assert options == {"max_steps": 3, "n_candidate_moves": 0,
                       "grow_boundaries": False, "max_lookahead": 1,
                       "combinatorial_breadth": 2}
    assert move == SENTINEL_TOLERANCES["move_tolerance"]
    assert admissibility == SENTINEL_TOLERANCES["admissibility_tolerance"]


def _tied_records(shift):
    """One content with two doublet contents whose lowest poles differ by
    ``shift`` in both spins. Both have triality zero, so the reading by 2T
    and the reading by the spin of the lift compare the same sectors."""
    return [{"content": [1, 1, 1], "doublet_reads": [
        _doublet_read([1, 1, 1], 0, {HALF: [1.0 + 0j], THREE: [2.0 + 0j]}),
        _doublet_read([3, 0, 0], 0, {HALF: [1.0 + shift + 0j],
                                     THREE: [2.0 + shift + 0j]})]}]


def test_the_reading_by_2T_names_ties_at_the_tie_tolerance_it_is_given():
    records = _tied_records(1e-6)
    declared = bp.ratios(records, 1e-15)["quasi_free"]["by_2T_reading"]
    assert declared["nucleon_tied_pairs"] == []
    assert declared["delta_tied_pairs"] == []
    loose = bp.ratios(records, 1e-3)["quasi_free"]["by_2T_reading"]
    assert [pair["doublet_content"] for pair in loose["nucleon_tied_pairs"]] \
        == [[3, 0, 0]]
    assert [pair["doublet_content"] for pair in loose["delta_tied_pairs"]] \
        == [[3, 0, 0]]


def test_both_pairings_name_ties_at_the_tie_tolerance_they_are_given():
    """The two pairings of a ratio compare the same two sectors here, so
    they name the same tied pairs at the same tolerance."""
    column = bp.ratios(_tied_records(1e-6), 1e-3)["quasi_free"]
    for role in ("nucleon_tied_pairs", "delta_tied_pairs"):
        assert column["by_spin_lift"][role] == column["by_2T_reading"][role]


def test_the_printed_minima_name_the_ties_of_the_record(monkeypatch):
    """A scan point read at a tie tolerance prints, in its 'lowest over'
    lines, the ties its record names at that tolerance."""
    (record,) = _tied_records(1e-6)
    monkeypatch.setattr(bp, "evaluate_content",
                        lambda content, kappa, beta, config: record)
    config = bp.default_config([1.0], [1.0], selected_contents=[(1, 1, 1)],
                               tolerances={"tie_tolerance": 1e-3})
    point = bp.scan_point(1.0, 1.0, config)
    recorded = point["ratios"]["quasi_free"]["by_2T_reading"]
    assert recorded["nucleon_tied_pairs"] and recorded["delta_tied_pairs"]
    printed = [line for line in bp.point_lines(point)
               if "lowest over" in line]
    assert len(printed) == 2
    assert all("tied" in line for line in printed)


@pytest.mark.parametrize("content", [(1, 1, 0), (0, 0, 4), (2, 2, 2),
                                     (-1, 2, 2)])
def test_a_content_that_is_not_three_quarks_is_refused_by_name(
        content, monkeypatch):
    argv = ["run", "--quiet", "--contents"] + [str(n) for n in content]
    with pytest.raises((SystemExit, ValueError)):
        _config_of(R, argv, monkeypatch)


@pytest.mark.parametrize("argv", [
    ["--combinatorial-depth", "0"],
    ["--combinatorial-depth", "2", "--combinatorial-length", "2"],
    ["--candidate-moves", "-1"]])
def test_an_invalid_solve_option_is_refused_before_anything_is_computed(
        argv, monkeypatch):
    """An option of the solves' drive that has no meaning is refused when
    the config is built, before the drive starts."""
    with pytest.raises((SystemExit, ValueError)):
        _config_of(R, ["run", "--quiet"] + argv, monkeypatch)


@pytest.fixture
def stub_reads(monkeypatch):
    """Replace the per-cell reads by the stand-in read and count the calls."""
    calls = []

    def reads(cells, z, links, config):
        calls.append(len(cells))
        return READ

    monkeypatch.setattr(R, "cell_reads", reads)
    return calls


@pytest.mark.parametrize("argv", [
    ["--pachner-depth", "0"],
    ["--pachner-depth", "2", "--pachner-length", "2"],
    ["--pachner-updates", "-1"],
    ["--ticks", "-1"],
    ["--kappa", "0"],
    ["--kappa", "nan"],
    ["--band-rank", "0"],
    ["--resolutions", "1.0", "-2.0"],
    ["--persistence-required", "-1"]])
def test_an_invalid_growth_schedule_is_refused_before_the_first_tick(
        argv, stub_reads, tmp_path):
    """A schedule of the growth step that has no meaning is refused before
    a tick is computed: no cell is read first."""
    with pytest.raises((SystemExit, ValueError)):
        R.main(["run", "--ticks", "1", "--persistence-required", "1",
                "--no-pachner-moves", "--quiet", "--json",
                str(tmp_path / "run.json")] + argv)
    assert stub_reads == []
    assert not (tmp_path / "run.points.jsonl").exists()


@pytest.mark.parametrize("argv", [["--rank-tolerance", "inf"],
                                  ["--tie-tolerance", "nan"],
                                  ["--band-tolerance", "-1e-15"]])
@pytest.mark.parametrize("driver", [bp, R, ISO],
                         ids=["baryon_poles", "recursion", "isospin_doublet"])
def test_a_tolerance_that_is_not_a_positive_finite_number_is_refused(
        driver, argv):
    """A tolerance on the command line is a positive finite number: an
    infinite tolerance, one that is not a number and a negative one are
    refused by the parser."""
    with pytest.raises(SystemExit):
        driver.build_parser().parse_args(["run", "--quiet"] + argv)


@pytest.mark.parametrize("value", [math.inf, math.nan, -1e-15])
def test_a_tolerance_with_no_meaning_is_refused_when_the_config_is_built(
        value):
    for build in (bp.default_config, R.default_config):
        with pytest.raises(ValueError, match="a tolerance is a finite"):
            build(tolerances={"rank_tolerance": value})


@pytest.mark.parametrize("argv", [["--kappa", "0"], ["--kappa", "1", "0"],
                                  ["--kappa", "nan"], ["--beta", "inf"]])
def test_a_scan_coupling_with_no_value_is_refused_before_anything_is_computed(
        argv, monkeypatch):
    """A kappa of zero (the action carries 1/kappa) and a coupling that is
    not a finite number are refused when the scan's config is built."""
    with pytest.raises(ValueError):
        _config_of(bp, ["run", "--quiet"] + argv, monkeypatch)


@pytest.mark.parametrize("argv", [
    ["--kappa", "1", "--content", "2", "2", "2"],
    ["--kappa", "1", "--content", "1", "1", "0"],
    ["--kappa", "1", "--content", "1", "1", "1", "--content", "0", "0", "4"],
    ["--kappa", "0"],
    ["--rank-tolerance", "inf"]])
def test_the_isospin_driver_refuses_a_declaration_with_no_value_first(
        argv, monkeypatch):
    """A content that is not three quarks in three bands, a kappa of zero
    and a tolerance that is not finite are refused before a host is
    read."""
    def refuse(*arguments, **options):
        raise _Reached()

    monkeypatch.setattr(ISO, "declared_carrier", refuse)
    monkeypatch.setattr(ISO, "relaxed_carrier", refuse)
    with pytest.raises((SystemExit, ValueError)):
        ISO.main(["run", "--quiet"] + argv)


def _base_and_pins(options, base):
    """The vertices of the base complex a solve is handed and the vertex
    sets its node is told to hold."""
    class Node:
        """What `node_configuration` declares on the engine's node."""

        def __init__(self):
            self.regions = []

        def declare_pinned_region(self, name, vertices):
            self.regions.append(sorted(vertices))

    node = Node()
    options["configure"](node)
    vertices = sorted({v for a, b, _, _ in cs.edge_fields(base)
                       for v in (a, b)})
    return vertices, node.regions


def test_the_pinned_vertices_are_ids_of_every_complex_a_solve_starts_on(
        monkeypatch):
    """``--pinned-vertices`` holds the same ids on the base complex of every
    solve it reaches: the level's base complex, with its own vertex ids,
    and every cell's base cell, one tetrahedron on its local vertices 0 to
    3. Its help names both, and so does the moment stiffness's, whose
    reference is the complex each solve starts on."""
    config = _config_of(R, ["run", "--quiet", "--pinned-vertices", "0", "4"],
                        monkeypatch)
    seen = []

    def solve(base, system, **options):
        seen.append(_base_and_pins(options, base))
        raise _Reached()

    cells, z, links, _ = R.level_zero(config)
    spacetime, count = R.build_level(cells, z, links)
    (kappa, beta, cell), _ = _first_cell_config(config, monkeypatch)
    monkeypatch.setattr(cs, "solve", solve)
    with pytest.raises(_Reached):
        R.relax_level(spacetime, config, [], count=count)
    with pytest.raises(_Reached):
        bp.relax_content((1, 1, 1), kappa, beta, cell)
    (level, level_pins), (local, cell_pins) = seen
    assert len(level) > 4 and {0, 4} <= set(level)
    assert local == [0, 1, 2, 3]
    assert level_pins == cell_pins == [[0, 4]]
    helps = _helps(R)
    for flag in ("--pinned-vertices", "--moment-stiffness-weight"):
        assert "base cell" in helps[flag]
        assert "level's base complex" in helps[flag]


def test_the_hinges_of_the_regge_sum_are_an_option_of_the_recursion():
    """``--regge-hinges`` reaches the run's config, and from it every
    level's and every cell's action; the default is the interior hinges."""
    assert R.default_config()["regge_hinges"] == "interior"
    args = R.build_parser().parse_args(["run", "--regge-hinges", "all"])
    assert R.default_config(regge_hinges=args.regge_hinges)[
        "regge_hinges"] == "all"
    with pytest.raises(SystemExit):
        R.build_parser().parse_args(["run", "--regge-hinges", "some"])


def _helps(driver):
    """The help string of every option of a driver's ``run``, by flag."""
    run = next(action for action in driver.build_parser()._actions
               if isinstance(action, argparse._SubParsersAction)).choices[
                   "run"]
    return {action.option_strings[0]: action.help for action in run._actions
            if action.option_strings}


def _options_without_help(driver):
    return [flag for flag, text in _helps(driver).items() if not text]


@pytest.mark.parametrize("driver", [bp, R, ISO],
                         ids=["baryon_poles", "recursion", "isospin_doublet"])
def test_every_option_but_quiet_has_a_help_string(driver):
    assert _options_without_help(driver) == ["--quiet"]


@pytest.mark.parametrize("driver", [bp, R, ISO],
                         ids=["baryon_poles", "recursion", "isospin_doublet"])
def test_the_help_of_a_tolerance_that_decides_nothing_says_so(driver):
    """The tolerances of the isospin-doublet detector that need a second
    frame, a further resolution or an observed doublet decide nothing on
    the one frame every driver gives the detector; the help of each says
    so and names the record's key, and no other tolerance's help does."""
    helps = _helps(driver)
    for key, _ in bp.TOLERANCES:
        assert ("unread_tolerances" in helps["--" + key.replace("_", "-")]) \
            == (key in bp.ISOSPIN_NEEDS), key
    assert set(bp.ISOSPIN_NEEDS) <= {key for key, _ in bp.ISOSPIN_TOLERANCES}


@pytest.fixture(scope="module")
def declared_isospin():
    """The detector's inputs on the declared host, one frame."""
    actions, spinorial = ISO.symmetry()
    return ISO.declared_carrier(), actions, spinorial


def test_a_tolerance_an_isospin_read_names_unread_changes_nothing(
        declared_isospin):
    """On the one frame the drivers give it, the detector's read names the
    seven tolerances that need more (`ISOSPIN_NEEDS`), and the read is the
    same, record for record, with those seven at 1e-300 and at 0.9; a
    tolerance it does read (the grouping) moves it at 0.9."""
    carrier, actions, spinorial = declared_isospin

    def read(tolerances):
        return json.dumps(bp._jsonable(ISO.observe_host(
            carrier, actions, spinorial, tolerances)), sort_keys=True,
            allow_nan=False)

    declared = read({})
    for key in ("covariant", "t_averaged"):
        record = json.loads(declared)[key]
        assert len(record["frames"]) == 1
        assert set(record["unread_tolerances"]) == set(bp.ISOSPIN_NEEDS)
    for value in (1e-300, 0.9):
        assert read({key: value for key in bp.ISOSPIN_NEEDS}) == declared
    assert read({"isospin_grouping_tolerance": 0.9}) != declared


def test_a_cells_recursion_read_names_the_resolutions_it_is_taken_at(
        monkeypatch):
    """The run's window of resolutions reaches the level's box
    (`test_the_command_line_reaches_the_box_and_the_growth_step`); a cell's
    recursion read is taken at the library's own resolutions, and its
    record names the ones its turn was taken at. The help of
    ``--resolutions`` says so."""
    config = _config_of(R, ["run", "--quiet", "--resolutions", "0.5", "2"],
                        monkeypatch)
    (_, _, cell), _ = _first_cell_config(config, monkeypatch)
    taken = []
    original = cob.LevelRecursion.overSpacetime

    def over_spacetime(spacetime, degree, source, declaration):
        taken.append([float(r) for r in declaration.resolutions])
        return original(spacetime, degree, source, declaration)

    monkeypatch.setattr(cob.LevelRecursion, "overSpacetime",
                        staticmethod(over_spacetime))
    record = bp.recursion_read(bp.build_host(), cell)
    assert config["resolutions"] == [0.5, 2.0]
    assert record["resolutions"] == taken[0] == [
        float(r) for r in cob.LevelRecursionDeclaration().resolutions]
    assert "the library's own resolutions" in _helps(R)["--resolutions"]


def test_the_options_one_driver_offers_and_the_other_does_not():
    """The two drivers share the action's, the mean field's, the
    tolerances', the limits' and the solves' options; the options of one
    alone are its scan (baryon_poles) and its recursion (recursion)."""
    def flags(driver):
        run = next(action for action in driver.build_parser()._actions
                   if isinstance(action, argparse._SubParsersAction)).choices[
                       "run"]
        return {action.option_strings[0] for action in run._actions
                if action.option_strings}
    assert flags(bp) - flags(R) == {"--isospin-doublet"}
    assert flags(R) - flags(bp) == {
        "--band-rank", "--contents", "--cosmological-constant-from-tick",
        "--max-cells", "--pachner-depth",
        "--pachner-candidate-moves", "--pachner-length",
        "--pachner-objective", "--pachner-updates",
        "--persistence-required",
        "--resolutions", "--tetrahedra", "--ticks"}
    registry = {"--" + key.replace("_", "-") for key, *_ in
                bp.TOLERANCES + bp.LIMITS + bp.SOLVE_OPTIONS
                if key != "pachner_moves"} | {"--no-pachner-moves"}
    assert registry <= flags(bp) & flags(R)


def test_the_registry_reaches_every_tolerance_of_the_isospin_detector():
    """Every real-valued field of the detector's configuration is set from
    a key of the registry: with every key at one value, every field is at
    that value. The one exception is the overlap a continuation must
    exceed, a declared threshold of one half and not a tolerance of a
    rounding; the detector's caps are not declared."""
    config = {key: 0.125 for key, _ in bp.TOLERANCES}
    detector = bp.isospin_doublet_config(config)
    fields = {name: getattr(detector, name) for name in dir(detector)
              if not name.startswith("_")
              and isinstance(getattr(detector, name), float)}
    assert len(fields) == 14
    assert {name: value for name, value in fields.items()
            if value != 0.125} == {"track_overlap_threshold": 0.5}
    assert (detector.condition_number_cap, detector.decomposed_rank_limit,
            detector.decomposed_commutant_limit) == (None, None, None)


def test_the_isospin_detector_takes_its_tolerances_from_the_registry():
    config = {key: 0.125 for key, _ in bp.TOLERANCES}
    detector = bp.isospin_doublet_config(config)
    assert [getattr(detector, field) for _, field in bp.ISOSPIN_TOLERANCES] \
        == [0.125] * 13
    assert {key for key, _ in bp.ISOSPIN_TOLERANCES} <= \
        {key for key, _ in bp.TOLERANCES}
    declared = obs.IsospinDoubletConfig()
    assert (declared.min_relative_gap, declared.condition_number_cap,
            declared.track_overlap_threshold) == (1e-15, None, 0.5)


# ------------------------------------------------------ records and files


def _one_tick_arguments(directory):
    return ["run", "--ticks", "1", "--persistence-required", "1",
            "--no-pachner-moves", "--trace-terms", "--quiet", "--json",
            str(directory / "run.json"), "--out", str(directory / "run.png")]


@pytest.fixture(scope="module")
def one_tick(tmp_path_factory):
    """`recursion.main` for one tick with --json and --out, the cells' reads
    the stand-in, every component accepted so that tick 0 grows cells and
    the growth step runs, and the geometry relaxed without Pachner moves."""
    directory = tmp_path_factory.mktemp("plumbing")
    original = R.cell_reads
    R.cell_reads = lambda cells, z, links, config: READ
    try:
        result = R.main(_one_tick_arguments(directory))
    finally:
        R.cell_reads = original
    return directory, result


def _without_seconds(value):
    if isinstance(value, dict):
        return {key: _without_seconds(item) for key, item in value.items()
                if key != "seconds"}
    if isinstance(value, list):
        return [_without_seconds(item) for item in value]
    return value


def _same(left, right):
    """Equality of two parsed JSON values, a NaN equal to a NaN."""
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _same(left[key], right[key]) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _same(a, b) for a, b in zip(left, right))
    if isinstance(left, float) and isinstance(right, float) \
            and math.isnan(left) and math.isnan(right):
        return True
    return left == right


def test_the_points_file_holds_the_config_the_host_and_every_tick(one_tick):
    """The file written at the end and the file appended tick by tick hold
    the same run: the first line is the config and the host, and line i + 1
    is tick i, each equal to the final file's after the round trip."""
    directory, result = one_tick
    document = json.loads((directory / "run.json").read_text())
    lines = [json.loads(line) for line in
             (directory / "run.points.jsonl").read_text().splitlines()]
    assert len(lines) == 1 + len(document["ticks"]) == 2
    assert set(lines[0]) == {"config", "host"}
    assert _same(lines[0]["config"], document["config"])
    assert _same(lines[0]["host"], document["host"])
    for line, tick in zip(lines[1:], document["ticks"]):
        assert _same(line, tick)
    # and the file is the record the run returned
    assert _same(_without_seconds(document),
                 _without_seconds(json.loads(json.dumps(R._jsonable(result)))))


def test_the_record_of_a_tick_names_what_the_command_line_declared(one_tick):
    directory, result = one_tick
    document = json.loads((directory / "run.json").read_text())
    config = document["config"]
    assert config["ticks"] == 1 and config["persistence_required"] == 1
    assert config["pachner_moves"] is False and config["trace_terms"] is True
    relaxation = document["ticks"][0]["relaxation"]
    assert relaxation["pachner_moves"] is False
    assert relaxation["moves_committed"] == 0 and not relaxation["changed"]
    # --trace-terms: every point a step was proposed from carries the terms,
    # one line per step proposal in the step file beside the JSON file,
    # which the config names and the level's record refers to
    assert config["steps_file"] == str(directory / "run.steps.jsonl")
    steps = relaxation["steps"]
    assert "term_trace" not in relaxation
    assert "jacobian_ranks" not in relaxation
    assert steps["file"] == config["steps_file"] and steps["solve"] == 0
    lines = [json.loads(line) for line in
             (directory / "run.steps.jsonl").read_text().splitlines()]
    assert steps["count"] == steps["term_lines"] == len(lines) >= 1
    assert [(line["tick"], line["level"], line["step"]) for line in lines] \
        == [(0, 0, step) for step in range(len(lines))]
    names = [term["name"] for term in lines[0]["terms"]]
    assert names[-1] == "action" and "regge" in names and "holonomy" in names
    assert len(bp.term_trace(relaxation)) == len(lines)
    growth = document["ticks"][0]["pachner"]
    assert growth["updates"] == config["pachner_updates"] == 1
    assert growth["depth"] == config["pachner_depth"] == 1
    assert growth["length"] == config["pachner_length"] == 0


def test_without_trace_terms_no_term_is_recorded(stub_reads):
    result = R.main(["run", "--ticks", "1", "--no-pachner-moves", "--quiet"])
    assert result["ticks"][0]["relaxation"]["term_trace"] == []


def _strict(text):
    def refuse(constant):
        raise ValueError("%s is not a JSON value" % constant)
    return json.loads(text, parse_constant=refuse)


def test_the_files_of_a_tick_are_json(one_tick):
    """Every line of the points file and the file written at the end are
    JSON. A tick of the stand-in run holds six infinities: the isolation gap
    of each of the three components whose band is its whole block, in the
    level's partition and in its image-supported fibers
    (partition.isolation_gaps, fibers.isolation_gap)."""
    directory, result = one_tick
    tick = result["ticks"][0]
    for gaps in (tick["partition"]["isolation_gaps"],
                 tick["fibers"]["isolation_gap"]):
        assert sum(1 for gap in gaps if math.isinf(gap)) == 3
    for line in (directory / "run.points.jsonl").read_text().splitlines():
        _strict(line)
    _strict((directory / "run.json").read_text())


def test_a_record_with_an_unmeasured_number_is_written_as_json(tmp_path):
    """A record holds a NaN where a number is unmeasured (the rank gap of a
    constrained step that was not taken); the line written for it is
    JSON."""
    path = tmp_path / "run.points.jsonl"
    R._append_line(path, {"relaxation": {"trace": [
        {"constrained_rank_gap": math.nan}]}})
    _strict(path.read_text())


def test_the_header_of_the_points_file_is_json(one_tick):
    """The first line, the config and the host, holds no NaN and no
    infinity and is JSON as it stands."""
    directory, _ = one_tick
    header = (directory / "run.points.jsonl").read_text().splitlines()[0]
    assert set(_strict(header)) == {"config", "host"}


@pytest.mark.parametrize("driver", [bp, R], ids=["baryon_poles", "recursion"])
def test_a_record_survives_the_round_trip_through_json(driver):
    """Complex numbers, numpy scalars, tuples, tuple keys and None come back
    from the written line as the values `_jsonable` gave them."""
    record = {"pole": 1.5 - 2j, "poles": [np.complex128(1 + 2j), 3j],
              "rank": np.int64(3), "norm": np.float64(0.25),
              "single": np.float32(1.5), "flag": True, "none": None,
              "cells": (0, 1, 2, 3), "edges": {(0, 1): 8.0, (0, 2): 8.0},
              "nested": [{"content": [1, 1, 1], "failed": "no value"}]}
    converted = driver._jsonable(record)
    assert json.loads(json.dumps(converted)) == converted
    assert converted["pole"] == {"re": 1.5, "im": -2.0}
    assert converted["poles"] == [{"re": 1.0, "im": 2.0},
                                  {"re": 0.0, "im": 3.0}]
    assert converted["rank"] == 3 and type(converted["rank"]) is int
    assert converted["cells"] == [0, 1, 2, 3]
    assert set(converted["edges"]) == {"(0, 1)", "(0, 2)"}


@pytest.mark.parametrize("driver,value", [
    (bp, np.bool_(True)), (R, np.bool_(True)), (bp, [np.bool_(False)]),
    (bp, np.array([1.0, 2.0])), (bp, np.array([1 + 1j]))],
    ids=["baryon_poles-bool", "recursion-bool", "baryon_poles-bool-in-list",
         "baryon_poles-array", "baryon_poles-complex-array"])
def test_every_numpy_value_of_a_record_is_written(driver, value, tmp_path):
    """A comparison of numpy numbers is a numpy boolean and a read is a numpy
    array; a record that holds one is written like any other."""
    driver._append_line(tmp_path / "run.points.jsonl", {"value": value})


def _stop_at_the_relaxation(monkeypatch):
    def refuse(spacetime, config, sectors=None, count=None):
        raise ValueError("VillainCharacter: W is not certified nonzero")
    monkeypatch.setattr(R, "relax_level", refuse)


def _stop_at_the_box(monkeypatch):
    def refuse(operator, config):
        raise ValueError("the declared selection separates two eigenvalues "
                         "that are equal exactly")
    monkeypatch.setattr(R, "recursion_turn", refuse)


@pytest.mark.parametrize("stop,reason,relaxed", [
    (_stop_at_the_relaxation, "the level's relaxation was refused", False),
    (_stop_at_the_box, "the recursion's turn has no value", True)],
    ids=["relaxation", "box"])
def test_a_tick_that_stops_early_is_written_and_reported(
        stop, reason, relaxed, stub_reads, monkeypatch, tmp_path, capsys):
    """A tick whose relaxation the library refuses, and a tick whose turn
    of the Section 15 box has no value, is a record like any other: it is
    appended to the points file, it is the run's last tick, the progress
    line, the summary and the drawn frame read it, and it says why it
    stopped. A tick that relaxed makes its cell reads, which need only the
    level, whatever the box gives; a tick without a relaxed level makes
    none."""
    stop(monkeypatch)
    config = R.default_config(ticks=3, solve={"pachner_moves": False})
    points = tmp_path / "run.points.jsonl"
    result = R.drive(config, progress=True, points_file=str(points))
    out = capsys.readouterr().out
    assert len(result["ticks"]) == 1 and result["stopped"] is False
    (record,) = result["ticks"]
    assert record["stopped"].startswith(reason)
    assert (len(stub_reads) == 1) is relaxed
    assert (record["reads"] != []) is relaxed
    assert ("failed" in record["relaxation"]) is not relaxed
    lines = points.read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[1])["stopped"] == record["stopped"]
    assert out.startswith("tick 0: 0 response vertices, 0 interactions, 0 "
                          "grown cells, row-sum defects []")
    text = R.summary(result)
    assert ("tick 0: the relaxation was refused: VillainCharacter" in text) \
        is not relaxed
    assert ("  stopped: the recursion's turn has no value" in text) is relaxed
    data = R.frame_data(result["ticks"], 0)
    assert data["counts"] == [{"tick": 0, "response_vertices": 0,
                               "grown_cells": 0, "row_sum_defects": []}]
    assert (data["marks"] != []) is relaxed
    R.render(result, str(tmp_path / "run.png"))
    assert (tmp_path / "run.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_a_run_of_no_tick_writes_its_header_and_an_empty_frame(tmp_path,
                                                                capsys):
    result = R.main(["run", "--ticks", "0", "--json",
                     str(tmp_path / "run.json"), "--out",
                     str(tmp_path / "run.png")])
    assert result["ticks"] == [] and result["stopped"] is False
    lines = (tmp_path / "run.points.jsonl").read_text().splitlines()
    assert len(lines) == 1 and set(json.loads(lines[0])) == {"config", "host"}
    assert json.loads((tmp_path / "run.json").read_text())["ticks"] == []
    assert capsys.readouterr().out == (
        "mode: controlled synthesis; host monopole numbers [1, 1]\n")


def test_a_stop_request_is_read_before_every_tick(stub_reads, tmp_path):
    """`stop_requested` is asked once before each tick; a run stopped after
    its first tick holds that tick, says it was stopped, and its points file
    holds the header and the tick."""
    asked = []

    def stop():
        asked.append(len(asked))
        return len(asked) > 1

    config = R.default_config(ticks=3, persistence_required=1,
                              solve={"pachner_moves": False})
    points = tmp_path / "run.points.jsonl"
    result = R.drive(config, stop_requested=stop, points_file=str(points))
    assert asked == [0, 1]
    assert result["stopped"] is True
    assert [tick["tick"] for tick in result["ticks"]] == [0]
    assert len(points.read_text().splitlines()) == 2


def _git(directory, *arguments):
    """git's own answer, the oracle of the commit the record reads from the
    checkout's files."""
    return subprocess.run(["git", "-C", str(directory)] + list(arguments),
                          check=True, capture_output=True,
                          text=True).stdout.strip()


def _linked_blas(module):
    """The BLAS the dynamic linker resolves for a compiled module (``ldd``),
    as the real path of its file."""
    lines = subprocess.run(["ldd", module], check=True, capture_output=True,
                           text=True).stdout.splitlines()
    return {os.path.realpath(line.split("=>")[1].split()[0])
            for line in lines if "=>" in line and "blas" in line
            and line.split("=>")[1].split()}


@pytest.mark.skipif(not (os.path.exists("/proc/self/maps")
                         and shutil.which("ldd") and shutil.which("git")),
                    reason="the oracles of this test are Linux's: the "
                           "process's map, ldd and git")
def test_the_record_names_the_command_line_the_commit_the_threads_and_the_blas(
        one_tick):
    """The header of the points file names what the run's numbers depend
    on beside its declarations: the arguments the driver parsed, the commit
    of the checkout the package is imported from (git's own answer), the
    thread variables, and every BLAS loaded in the process with the threads
    it runs: among them the build numpy is linked to (its configuration as
    numpy reports it) and the one the package's compiled module is linked
    to (as the dynamic linker resolves it)."""
    directory, _ = one_tick
    header = _strict((directory / "run.points.jsonl").read_text()
                     .splitlines()[0])
    environment = header["config"]["environment"]
    assert environment["arguments"] == _one_tick_arguments(directory)
    package = os.path.dirname(os.path.abspath(R.__file__))
    assert environment["checkout"]["commit"] == _git(package, "rev-parse",
                                                     "HEAD")
    assert os.path.realpath(environment["checkout"]["root"]) == \
        os.path.realpath(_git(package, "rev-parse", "--show-toplevel"))
    assert environment["threads"]["OMP_NUM_THREADS"] == \
        os.environ.get("OMP_NUM_THREADS")
    runtimes = environment["runtimes"]
    blas = [r for r in runtimes if r.get("kind") == "openblas"]
    assert blas and all(isinstance(r["threads"], int) and r["threads"] >= 1
                        for r in blas)
    numpy_build = np.show_config(mode="dicts")["Build Dependencies"]["blas"]
    assert numpy_build["openblas configuration"].split() in \
        [r["configuration"].split() for r in blas]
    module = environment["module"]
    assert module == os.path.realpath(sys.modules["tessera._tessera"].__file__)
    linked = _linked_blas(module)
    assert linked and linked <= {os.path.realpath(r["file"])
                                 for r in runtimes}


#: A process that prints the runtimes its environment record names.
_RUNTIMES = """
import json
from tessera.drivers import baryon_poles as bp
print(json.dumps(bp.environment_record([])["runtimes"]))
"""


@pytest.mark.skipif(not os.path.exists("/proc/self/maps"),
                    reason="the process's libraries are listed on Linux")
@pytest.mark.parametrize("threads", [1, 2])
def test_the_record_names_the_threads_each_runtime_runs(threads):
    """With OMP_NUM_THREADS set and no library variable, every threaded
    BLAS and OpenMP runtime of the process reports that many threads, and
    the record names it for each."""
    environment = {key: value for key, value in os.environ.items()
                   if key not in ("OPENBLAS_NUM_THREADS", "GOTO_NUM_THREADS",
                                  "MKL_NUM_THREADS")}
    environment["OMP_NUM_THREADS"] = str(threads)
    environment["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    done = subprocess.run([sys.executable, "-c", _RUNTIMES], env=environment,
                          check=True, capture_output=True, text=True)
    runtimes = json.loads(done.stdout)
    counted = [r for r in runtimes if r.get("kind") in ("openblas", "openmp")]
    assert {r["kind"] for r in counted} == {"openblas", "openmp"}
    for runtime in counted:
        expected = 1 if runtime.get("threading") == "sequential" else threads
        assert runtime["threads"] == expected, runtime["file"]


def test_the_isospin_driver_writes_json_with_its_environment(tmp_path):
    """The isospin driver's file is JSON and names the arguments it parsed
    and the runtimes of its process."""
    arguments = ["run", "--quiet", "--json", str(tmp_path / "doublet.json")]
    ISO.main(arguments)
    document = _strict((tmp_path / "doublet.json").read_text())
    assert document["environment"]["arguments"] == arguments
    assert "runtimes" in document["environment"]
    assert set(document["declared_host"]["covariant"]["unread_tolerances"]) \
        == set(bp.ISOSPIN_NEEDS)


def test_the_points_path_is_beside_the_json(tmp_path):
    assert R.points_path("run.json") == "run.points.jsonl"
    assert R.points_path("run.v2.json") == "run.v2.points.jsonl"
    assert R.points_path("run") == "run.points.jsonl"
    assert R.points_path(tmp_path / "out.d" / "run") == \
        str(tmp_path / "out.d" / "run.points.jsonl")
    assert bp.points_path("a/b.json") == "a/b.points.jsonl"


# ------------------------------------------------------------------ state


def test_a_run_leaves_its_config_as_it_was_given(stub_reads):
    """`drive` reads its config: after a tick that relaxes the level, turns
    the box, grows cells and takes the growth step, the config is the one
    it was given, and the config the result records is equal to it."""
    config = R.default_config(ticks=1, persistence_required=1,
                              solve={"pachner_moves": False})
    before = copy.deepcopy(config)
    result = R.drive(config)
    assert config == before
    assert result["config"] == json.loads(json.dumps(R._jsonable(before)))
    assert "pachner" in result["ticks"][0]


def test_the_cells_reads_leave_the_runs_config_and_fields_as_they_were(
        monkeypatch):
    config = R.default_config(tetrahedra=2)
    before = copy.deepcopy(config)
    (_, _, cell), (cells, z, links) = _first_cell_config(config, monkeypatch)
    assert config == before
    # the cell's own config is a new one: writing to it leaves the run's
    cell["band_tolerance"] = 0.5
    cell["contents"].append([9, 9, 9])
    assert config == before
    assert z == {edge: complex(8.0) for edge in z}


def test_the_doublet_sectors_are_formed_once_per_content_and_tolerance():
    """The cached sectors of a doublet content are the ones formed afresh,
    at each tolerance on its own."""
    trialities = [0, 1, 2]
    for tolerance in (1e-15, 1e-9):
        content, triality, cached = bp.doublet_sectors((1, 1, 1), trialities,
                                                       tolerance)
        again = bp.doublet_sectors((1, 1, 1), trialities, tolerance)[2]
        assert again is cached
        states, _ = bp.singlet_states(content)
        fresh = bp.spin_sectors(states, tolerance)[0]
        assert fresh.keys() == cached.keys()
        for j2 in fresh:
            np.testing.assert_array_equal(fresh[j2], cached[j2])
    assert (tuple(content), 1e-15) in bp._DOUBLET_SECTORS
    assert (tuple(content), 1e-9) in bp._DOUBLET_SECTORS


def test_the_level_built_twice_is_the_same_level():
    """The declared level, its connection and its fields are a function of
    the config: two builds agree bit for bit, and so do two levels built
    from them."""
    config = R.default_config(tetrahedra=2)
    first, second = R.level_zero(config), R.level_zero(config)
    assert first[0] == second[0] and first[1] == second[1]
    assert first[2] == second[2]
    one, _ = R.build_level(*first[:3])
    two, _ = R.build_level(*second[:3])
    assert cs.edge_fields(one) == cs.edge_fields(two)


#: A process that forms the T-average of one fixed three-quark operator in
#: the aligned frame of the declared host, as a read forms the operator of
#: its with-quartic column (`rotation_averaged_many_body`), and prints the
#: hash of the result's bytes.
_AVERAGE = """
import hashlib
import numpy as np
from tessera.drivers import baryon_poles as bp
host = bp.monopole_support()
actions = bp.rotation_action([host] * bp.SHEETS)
alignment = bp.aligned_doublet_frame(host, bp.rotation_group())
frame = bp._micro_frame([alignment] * bp.SHEETS)
dual = np.linalg.inv(frame)
size = len(bp.occupation_basis())
rng = np.random.default_rng(0)
operator = rng.standard_normal((size, size)) \
    + 1j * rng.standard_normal((size, size))
averaged = bp.rotation_averaged_many_body(operator, actions, frame, dual)
print(hashlib.sha1(np.ascontiguousarray(averaged).tobytes()).hexdigest())
"""


def _average_at(threads):
    """The hash the process above prints at a thread count, set as the test
    runner sets it (OMP_NUM_THREADS)."""
    environment = {key: value for key, value in os.environ.items()
                   if key != "OPENBLAS_NUM_THREADS"}
    environment["OMP_NUM_THREADS"] = str(threads)
    environment["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    done = subprocess.run([sys.executable, "-c", _AVERAGE], env=environment,
                          check=True, capture_output=True, text=True)
    return done.stdout.strip()


def test_the_many_body_average_is_the_same_twice_at_one_thread_count():
    assert _average_at(1) == _average_at(1)
    assert len(_average_at(1)) == 40


@pytest.mark.xfail(strict=True, reason=(
    "`rotation_averaged_many_body` multiplies and solves 816 x 816 matrices "
    "with numpy, whose rounding depends on the thread count of its linear "
    "algebra (OMP_NUM_THREADS); every stage of a read before it is the same "
    "bit for bit at 1, 2 and 4 threads, the with-quartic operator and its "
    "poles differ in their last bits (4e-16 of their size), and the "
    "certificates decided on them at 1e-15 differ: sharp_spinor False at "
    "one thread and True at two, determinant_count 8 and 1, 18 and 9. The "
    "record names the threads and the BLAS; whether decisions at 1e-15 on "
    "such values stand is a question for the user (fix(drivers): defects "
    "of the drivers' options, records and files found by the audit, "
    "https://github.com/akellehe/tessera/issues/1393)"))
def test_the_many_body_average_does_not_depend_on_the_thread_count():
    """The operator a read's with-quartic poles are taken from is the same
    array whatever the number of threads the run is given."""
    assert _average_at(1) == _average_at(2)


# ----------------------------------------------------------- test helpers


def test_the_recorded_runs_tolerances_cover_the_registry():
    """A test made "at the run's tolerances" declares `RUN.TOLERANCES`; a
    key of the registry the helper leaves out would be read at 1e-15 there
    without a word."""
    assert set(RUN.TOLERANCES) == {key for key, _ in bp.TOLERANCES}
    assert all(value > 0.0 for value in RUN.TOLERANCES.values())
    assert set(RUN.HOST_CELLS) == {tuple(cell)
                                   for cell in RUN.LEVEL_ZERO_CELLS}
