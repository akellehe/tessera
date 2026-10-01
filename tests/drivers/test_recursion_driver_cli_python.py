# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The command line, the outputs and the band helpers of the level-recursion
driver (`tessera.drivers.recursion`).

`main` is run end to end on the declared host with the per-cell baryon reads
(`cell_reads`) replaced by a fixed stand-in, because a real read relaxes every
declared content of every cell and takes minutes; every other step of a tick
(the relaxation, the Section 15 box, the interaction stage and the grown-cell
rule) is the real one. The claims under test are the parsing and its refusals
by name, which outputs each flag writes, what the text summary and the drawn
frame contain, and the closed-form behaviour of the band helpers. The stand-in
read carries two doublet contents with different poles, so that the text, the
tick's summary and the frame can be checked to report both.
"""
import json
import math

import numpy as np
import pytest

from tessera import cobordism as cob
from tessera.drivers import baryon_poles as bp
from tessera.drivers import recursion as R
from tests.drivers import _recursion_run_2026_09_23 as RUN

HALF = str(bp.SPIN_HALF)
THREE = str(bp.SPIN_THREE_HALVES)


def _column(poles, failed=()):
    """One column of a sector as `baryon_poles.sector_entry` writes it."""
    poles = [complex(p) for p in poles]
    lowest = min(poles, key=lambda p: (p.real, p.imag)) if poles else None
    return {"poles": poles, "multiplicity": [4] * len(poles),
            "lowest_pole": lowest, "failed_certificates": list(failed),
            "compression_leakage": 2e-16,
            "pole_certificates": [{"sharp_spinor": True,
                                   "spin_lift_sharp": True,
                                   "colour_casimir_residual": 0.0}
                                  for _ in poles]}


def _sector(j2, triality, quasi_free, with_quartic, failed=()):
    irreps = bp.restriction(float(j2), triality)
    return {"restriction_to_2T": irreps, "nucleon_reading": "2" in irreps,
            "delta_reading": sorted(irreps) == ["2'", "2''"],
            "quasi_free": _column(quasi_free),
            "with_quartic": _column(with_quartic, failed)}


#: The read content of the stand-in: two doublet contents with different
#: poles. (0, 2, 1) has only a spin-3/2 sector, whose quasi-free pole 4+0.5i
#: is the lowest spin-3/2 pole and whose quartic read found no pole; (1, 1, 1)
#: has a spin-1/2 pole 6 and a spin-3/2 pole 5+0.25i, and the quartic poles
#: -2 and -1.
CONTENTS = [
    {"content": [0, 3, 0], "failed": "band 1 has rank 2",
     "doublet_reads": []},
    {"content": [3, 0, 0],
     "doublet_reads": [
         {"doublet_content": [0, 2, 1], "total_triality": 1, "sectors": {
             THREE: _sector(THREE, 1, [4.0 + 0.5j], [],
                            failed=["no-zero-enclosed"])}},
         {"doublet_content": [1, 1, 1], "total_triality": 0, "sectors": {
             HALF: _sector(HALF, 0, [6.0], [-2.0]),
             THREE: _sector(THREE, 0, [5.0 + 0.25j], [-1.0])}}],
     "relaxation": {"converged": False, "force_norm": 0.25,
                    "iterations": 40},
     "quark_conditions": {"certified": False, "conditions": [
         {"name": "persistent-cluster", "status": "Failed"}]},
     "isospin_doublet": {"covariant": {"status": "x", "found": False,
                                       "other": 1}},
     "quartic": {"truncation": {"induced_displacement_norm": 0.1,
                                "unrelated": 2.0}}}]

#: A read of one cell, as `cell_reads` returns it: one content at which the
#: library names no value and the read content above, with the cell's ratios
#: and pole table.
READ = [{
    "cell": [0, 1, 2, 3],
    "host_cell": {},
    "failed_contents": [[0, 3, 0]],
    "flagged_contents": [],
    "contents": CONTENTS,
    "ratios": bp.ratios(CONTENTS), "pole_table": bp.pole_table(CONTENTS)}]


@pytest.fixture
def stub_reads(monkeypatch):
    """Replace the per-cell reads and record the configuration each tick
    passes to them."""
    seen = []

    def reads(cells, z, links, config):
        seen.append(dict(config))
        return READ

    monkeypatch.setattr(R, "cell_reads", reads)
    return seen


# ------------------------------------------------------------ the parser


def test_the_declared_defaults():
    args = R.build_parser().parse_args(["run"])
    assert args.ticks == R.DECLARED_TICKS == 3
    assert args.tetrahedra == R.DECLARED_TETRAHEDRA == 2
    assert args.kappa == 1.0 and args.beta == 1.0
    assert args.resolutions == list(R.DECLARED_RESOLUTIONS)
    assert args.band_rank == 1 and args.edge_squared == 8.0
    assert args.eliminate == "lengths-and-phases"
    assert args.contents is None and args.max_cells is None
    assert args.json is None and args.out is None
    assert not args.live and not args.quiet


def test_contents_are_repeatable_triples():
    args = R.build_parser().parse_args(
        ["run", "--contents", "1", "1", "1", "--contents", "3", "0", "0",
         "--max-cells", "1", "--resolutions", "1", "2"])
    assert args.contents == [[1, 1, 1], [3, 0, 0]]
    assert args.max_cells == 1 and args.resolutions == [1.0, 2.0]


@pytest.mark.parametrize("argv,name", [
    (["run", "--contents", "1", "1"], "--contents"),
    (["run", "--band-selection", "by-hand"], "--band-selection"),
    (["run", "--eliminate", "phases"], "--eliminate"),
    (["run", "--ticks", "two"], "--ticks"),
    ([], "command"),
])
def test_invalid_arguments_are_refused_by_name(argv, name, capsys):
    with pytest.raises(SystemExit) as stop:
        R.build_parser().parse_args(argv)
    assert stop.value.code == 2
    assert name in capsys.readouterr().err


def test_the_declared_config():
    config = R.default_config(ticks=2, selected_contents=[(1, 1, 1)],
                              max_cells=1)
    assert config["emergence"] == "strict"
    assert config["contents"] == [[1, 1, 1]]
    assert config["kappas"] == [1.0] and config["betas"] == [1.0]
    assert config["sheets"] == R.SHEETS == 3
    assert "contour_nodes" not in config
    assert config["max_cells"] == 1
    assert R.points_path("a/run.json") == "a/run.points.jsonl"
    assert all(config[key] == bp.DECLARED_TOLERANCE == 1e-15
               for key, _ in bp.TOLERANCES)
    tight = R.default_config(ticks=2, tolerances={"rank_tolerance": 1e-10})
    assert tight["rank_tolerance"] == 1e-10
    assert tight["newton_tolerance"] == bp.DECLARED_TOLERANCE


def test_every_tolerance_is_an_option_of_the_run():
    args = R.build_parser().parse_args(
        ["run", "--rank-tolerance", "1e-12", "--recursion-tolerance", "1e-9"])
    tolerances = bp.tolerances_from(args)
    assert tolerances["rank_tolerance"] == 1e-12
    assert tolerances["recursion_tolerance"] == 1e-9
    assert tolerances["certificate_tolerance"] == bp.DECLARED_TOLERANCE
    # every key of the registry, each its own option, each 1e-15 by default
    declared = bp.tolerances_from(R.build_parser().parse_args(["run"]))
    assert declared == {key: 1e-15 for key, _ in bp.TOLERANCES}
    for key, _ in bp.TOLERANCES:
        option = "--" + key.replace("_", "-")
        set_ = bp.tolerances_from(
            R.build_parser().parse_args(["run", option, "1e-7"]))
        assert set_[key] == 1e-7
        assert all(value == 1e-15 for other, value in set_.items()
                   if other != key)


def test_every_tolerance_and_limit_is_carried_into_every_cell(monkeypatch):
    """`cell_reads` hands every cell's read a config that carries every
    tolerance of the registry and every declared limit at the run's values,
    whether the run left them at the declared 1e-15 or set them."""
    seen = []

    def scan(kappa, beta, config, on_content=None):
        seen.append(dict(config))
        return {"failed_contents": [], "contents": [], "ratios": {},
                "pole_table": {}}

    monkeypatch.setattr(bp, "scan_point", scan)
    declared = R.default_config(tetrahedra=2)
    cells, z, links, _ = R.level_zero(declared)
    R.cell_reads(cells, z, links, declared)
    chosen = {key: 10.0 ** -(3 + k) for k, (key, _) in
              enumerate(bp.TOLERANCES)}
    limits = {"iteration_limit": 7, "halving_limit": 5,
              "time_limit_seconds": 2.5}
    run = R.default_config(tetrahedra=2, tolerances=chosen, limits=limits)
    R.cell_reads(cells, z, links, run)
    assert len(seen) == 2 * len(cells)
    for config in seen[:len(cells)]:
        assert {key: config[key] for key, _ in bp.TOLERANCES} == {
            key: 1e-15 for key, _ in bp.TOLERANCES}
        assert all(config[key] is None for key, _, _ in bp.LIMITS)
    for config in seen[len(cells):]:
        assert {key: config[key] for key, _ in bp.TOLERANCES} == chosen
        assert {key: config[key] for key, _, _ in bp.LIMITS} == limits


def test_the_recursion_turn_and_the_pachner_stage_read_the_registry(
        monkeypatch):
    """The level recursion's declaration carries the config's recursion
    tolerance and quotient rank tolerance, and the Pachner stage's node the
    config's move tolerance and admissibility tolerance; each is 1e-15 when
    the run declares nothing. The turn and the stage's search are replaced
    by stand-ins that record what they are handed."""
    seen = {}

    class Turn:
        def advance(self):
            pass

        def level(self, index):
            return index

    def over_pencil(operator, metric, n, declaration):
        seen["recursion"] = (declaration.tolerance, declaration.rank_tolerance)
        return Turn()

    monkeypatch.setattr(cob.LevelRecursion, "overPencil",
                        staticmethod(over_pencil))
    operator = np.diag([1.0, 2.0, 3.0, 4.0]).astype(complex)
    R.recursion_turn(operator, R.default_config())
    assert seen["recursion"] == (1e-15, 1e-15)
    R.recursion_turn(operator, R.default_config(
        tolerances={"recursion_tolerance": 1e-9,
                    "quotient_rank_tolerance": 1e-7}))
    assert seen["recursion"] == (1e-9, 1e-7)
    library = cob.LevelRecursionDeclaration()
    assert library.tolerance == library.rank_tolerance == 1e-15

    def run_stage1(node, *args, **kwargs):
        seen["stage"] = (node.move_tolerance, node.admissibility_tolerance)
        return []

    monkeypatch.setattr(cob.MultiCobordism, "run_stage1", run_stage1)
    for tolerances, expected in (
            (None, (1e-15, 1e-15)),
            ({"move_tolerance": 1e-9, "admissibility_tolerance": 1e-12},
             (1e-9, 1e-12))):
        config = R.default_config(tolerances=tolerances)
        cells, z, links, _ = R.level_zero(config)
        R.pachner_stage(cells, z, links, config)
        assert seen["stage"] == expected


def test_the_villain_order_is_an_option_carried_into_every_cell(monkeypatch):
    """M, the order the Villain weight of the holonomy term is summed to, is
    an option of the run (``--villain-order``, an integer from 1 to 10, 10 by
    default), recorded in the config under ``villain_order``, declared on
    every level's action, and carried into the config of every cell's read.
    The cell read is taken on the recorded tick-0 level
    (`_recursion_run_2026_09_23`) with the scan point replaced by a recorder
    of the config it is given. The level relaxation is run at the recorded
    run's tolerances with one accepted step and one halving declared as its
    limits, so that the read of the declaration does not wait on a solve."""
    assert R.build_parser().parse_args(["run"]).villain_order == 10
    args = R.build_parser().parse_args(["run", "--villain-order", "6"])
    assert args.villain_order == 6
    assert R.default_config()["villain_order"] == bp.DECLARED_VILLAIN_ORDER
    config = R.default_config(
        villain_order=6, selected_contents=[(1, 1, 1)],
        tolerances=RUN.TOLERANCES,
        limits={"iteration_limit": 1, "halving_limit": 1})
    assert config["villain_order"] == 6
    with pytest.raises(ValueError, match="integer from 1 to 10"):
        R.default_config(villain_order=11)

    seen = []

    def scan(kappa, beta, cell_config, on_content=None):
        seen.append(dict(cell_config))
        return {"failed_contents": [], "contents": [], "ratios": {},
                "pole_table": []}

    monkeypatch.setattr(bp, "scan_point", scan)
    reads = R.cell_reads(RUN.LEVEL_ZERO_CELLS, RUN.LEVEL_ZERO_SQUARED_LENGTHS,
                         RUN.LEVEL_ZERO_LINKS, config)
    assert len(reads) == len(seen) == 2
    assert [cell["villain_order"] for cell in seen] == [6, 6]

    declared = []
    original = bp.action_declaration

    def declaration(*arguments, **keywords):
        declared.append(keywords.get("villain_order"))
        return original(*arguments, **keywords)

    monkeypatch.setattr(bp, "action_declaration", declaration)
    spacetime, count = R.build_level(RUN.LEVEL_ZERO_CELLS,
                                     RUN.LEVEL_ZERO_SQUARED_LENGTHS,
                                     RUN.LEVEL_ZERO_LINKS)
    cut = R.bounding_cut(RUN.LEVEL_ZERO_CELLS)
    R.relax_level(spacetime, config, R.cut_sectors(
        cut, R.cut_monopole_number(cut, RUN.LEVEL_ZERO_LINKS), count),
        count=count)
    assert declared == [6]


@pytest.mark.parametrize("text", ["0", "11", "1.5", "many"])
def test_a_villain_order_outside_one_to_ten_is_refused_by_name(text, capsys):
    with pytest.raises(SystemExit) as stop:
        R.build_parser().parse_args(["run", "--villain-order", text])
    assert stop.value.code == 2
    assert "--villain-order is an integer from 1 to 10" in \
        capsys.readouterr().err


def test_the_recorded_run_is_read_at_order_ten():
    """The run of 2026-09-23 (`_recursion_run_2026_09_23`) summed the Villain
    weight at beta = 1 until its coefficient exp(-m^2 / 2) fell below 1e-18.
    exp(-81 / 2) = 2.6e-18 is not below it and exp(-100 / 2) = 1.9e-22 is, so
    the run kept |m| <= 10, which is the order ten the drivers declare by
    default. On the recorded tick-0 level, whose links have unit modulus, the
    order-ten sums are the run's sums: what the order leaves out of W, DW and
    D^2W, relative to the sums of the moduli of their terms, is reported as
    4.2e-27, 6.4e-26 and 5.1e-25 (the first pair beyond the order,
    2 exp(-121 / 2) = 1.1e-26, times 1, 11 and 121, over sums of 2.507,
    1.824 and 2.507), each below the run's floor. Order nine drops the pair
    2 exp(-50) = 3.9e-22 times the same weights, which is below the rounding
    2^-53 = 1.1e-16 of sums of order one, so the holonomy term of the level
    at order nine equals the one at order ten to 1e-15."""
    least = next(m for m in range(1, 100)
                 if math.exp(-m * m / (2.0 * RUN.BETA))
                 < RUN.VILLAIN_COEFFICIENT_FLOOR)
    assert least == RUN.VILLAIN_ORDER == bp.DECLARED_VILLAIN_ORDER == 10
    assert math.exp(-81 / 2.0) > RUN.VILLAIN_COEFFICIENT_FLOOR \
        > math.exp(-100 / 2.0)
    assert R.default_config(tolerances=RUN.TOLERANCES)["villain_order"] == \
        RUN.VILLAIN_ORDER

    spacetime, _ = R.build_level(RUN.LEVEL_ZERO_CELLS,
                                 RUN.LEVEL_ZERO_SQUARED_LENGTHS,
                                 RUN.LEVEL_ZERO_LINKS)

    def action(order):
        return cob.JointAction(spacetime, bp.action_declaration(
            spacetime, 1.0, RUN.BETA, villain_order=order))

    ten = action(RUN.VILLAIN_ORDER)
    faces = np.asarray(ten.face_holonomies())
    assert np.max(np.abs(np.abs(faces) - 1.0)) < 1e-14
    read = ten.holonomy_truncation()
    assert read.order == 10
    assert read.relative_value_tail == pytest.approx(4.24e-27, rel=1e-2)
    assert read.relative_first_tail == pytest.approx(6.41e-26, rel=1e-2)
    assert read.relative_second_tail == pytest.approx(5.13e-25, rel=1e-2)
    assert read.relative_second_tail < RUN.VILLAIN_COEFFICIENT_FLOOR
    nine = action(9)
    assert abs(complex(nine.holonomy_term()) - complex(ten.holonomy_term())) \
        < 1e-15 * abs(complex(ten.holonomy_term()))
    assert np.max(np.abs(np.asarray(nine.holonomy_hessian())
                         - np.asarray(ten.holonomy_hessian()))) < 1e-15


# ------------------------------------------------------------ main


@pytest.fixture(scope="module")
def two_ticks(tmp_path_factory):
    """`main` for two ticks with --json and --out, the reads stubbed. Every
    component is accepted (--persistence-required 1), so that tick 0 grows
    cells and the grown-cell path is written; with the declared default the
    recursion stops at tick 0."""
    directory = tmp_path_factory.mktemp("recursion")
    original = R.cell_reads
    R.cell_reads = lambda cells, z, links, config: READ
    try:
        result = R.main(["run", "--ticks", "2", "--persistence-required", "1",
                         "--json",
                         str(directory / "run.json"), "--out",
                         str(directory / "run.png"), "--quiet"])
    finally:
        R.cell_reads = original
    return directory, result


def test_main_writes_every_tick_to_the_json(two_ticks):
    directory, result = two_ticks
    document = json.loads((directory / "run.json").read_text())
    assert [t["tick"] for t in document["ticks"]] == [0, 1]
    assert document["stopped"] is False
    assert document["host"]["monopole_numbers"] == [1, 1]
    assert document["host"]["cells"] == [[0, 1, 2, 3], [0, 1, 3, 4]]
    assert document["host"]["connection_residual"] < 1e-12
    # the recursion's arrays are nested lists and complex numbers {"re", "im"}
    first = document["ticks"][0]
    assert isinstance(first["grown_cells"][0]["frame_invariant_ratios"], list)
    assert set(first["grown_cells"][0]["scale"]) == {"re", "im"}
    assert first["summary"]["response_vertices"] == len(
        first["lineage"])
    assert first["summary"]["quark_verdicts"] == [
        [None, {"certified": False,
                "status": {"persistent-cluster": "Failed"}}]]
    assert first["summary"]["isospin_doublet"][0][1] == {
        "covariant": {"status": "x", "found": False}}
    poles = first["summary"]["poles"][0][1]
    truncation = poles["quartic_truncation"]
    assert truncation["induced_displacement_norm"] == 0.1
    assert "unrelated" not in truncation
    # every doublet content keeps its poles in the tick's summary
    assert [p["doublet_content"] for p in poles["per_doublet_content"]] == \
        [[0, 2, 1], [1, 1, 1]]
    assert poles["per_doublet_content"][0]["sectors"][THREE]["quasi_free"][
        "poles"] == [{"re": 4.0, "im": 0.5}]
    assert poles["per_doublet_content"][1]["sectors"][HALF]["with_quartic"][
        "poles"] == [{"re": -2.0, "im": 0.0}]
    # and the minima over the doublet contents name the one they came from
    lowest = poles["lowest_over_doublet_contents"]
    assert lowest["quasi_free"][THREE]["doublet_content"] == [0, 2, 1]
    assert lowest["with_quartic"][THREE]["doublet_content"] == [1, 1, 1]
    assert lowest["quasi_free"][HALF]["doublet_content"] == [1, 1, 1]
    assert len(result["ticks"]) == 2


def test_main_appends_every_tick_to_the_points_file(two_ticks):
    directory, _ = two_ticks
    lines = (directory / "run.points.jsonl").read_text().splitlines()
    assert len(lines) == 3
    assert set(json.loads(lines[0])) == {"config", "host"}
    assert [json.loads(line)["tick"] for line in lines[1:]] == [0, 1]


def test_main_renders_the_final_frame(two_ticks):
    directory, _ = two_ticks
    assert (directory / "run.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_main_passes_the_declared_options_to_every_tick(stub_reads):
    R.main(["run", "--ticks", "1", "--contents", "1", "1", "1",
            "--max-cells", "1", "--kappa", "0.5", "--beta", "2",
            "--band-selection", "sort-every-iterate", "--band-rank", "1",
            "--quiet"])
    (config,) = stub_reads
    assert config["contents"] == [[1, 1, 1]]
    assert config["max_cells"] == 1
    assert config["kappa"] == 0.5 and config["beta"] == 2.0
    assert config["band_selection"] == "sort-every-iterate"
    assert config["villain_order"] == bp.DECLARED_VILLAIN_ORDER == 10


def test_progress_and_summary_are_printed_unless_quiet(stub_reads, capsys):
    R.main(["run", "--ticks", "1"])
    out = capsys.readouterr().out
    assert out.startswith("tick 0: 2 response vertices, 1 interactions, 0 "
                          "grown cells")
    assert "  the Regge term is structurally zero on this level: it has 0 " \
        "hinges under the interior hinge rule" in out
    assert "  component 2 (edges 0-4) rejected: it persists over 1 of the 5 " \
        "required resolutions" in out
    assert "held bounding cut" in out
    assert "mode: controlled synthesis; host monopole numbers [1, 1]" in out
    assert "tick 0: level with 5 vertices, 9 edges, 2 tetrahedra per sheet" \
        in out
    assert "host cell [0, 1, 2, 3] content [0, 3, 0] (quarks per band of " \
        "h_1): no value: band 1 has rank 2" in out
    assert "content [3, 0, 0] (quarks per band of h_1): read; quark " \
        "certified False; mean field converged False (force norm 0.25 after " \
        "40 iterations)" in out
    # every quark condition's status closes the content's line
    assert "; quark conditions persistent-cluster Failed\n" in out
    # the tick's progress and the final summary both carry one line per
    # (content, doublet content) pair, then the labelled minima and ratios
    first = ("      host cell [0, 1, 2, 3] content [3, 0, 0], doublet "
             "content [0, 2, 1] (triality 1) | spin 1/2: no sector | spin 3/2 "
             "(restricts to 2''+2): quasi-free 4+0.5i x4 [spinor sharp, lift "
             "sharp, colour 0] {read certified, leakage 2e-16}; with quartic "
             "no pole {read failed no-zero-enclosed, leakage 2e-16}")
    second = ("      host cell [0, 1, 2, 3] content [3, 0, 0], doublet "
              "content [1, 1, 1] (triality 0) | spin 1/2 (restricts to 2): "
              "quasi-free 6+0i x4")
    assert out.count(first + "\n") == 2
    assert out.count(second) == 2
    lowest = ("    host cell [0, 1, 2, 3] lowest over the doublet contents of "
              "content [3, 0, 0]: spin 1/2 quasi-free 6+0i from doublet "
              "content [1, 1, 1]; spin 1/2 with quartic -2+0i from doublet "
              "content [1, 1, 1]; spin 3/2 quasi-free 4+0.5i from doublet "
              "content [0, 2, 1]; spin 3/2 with quartic -1+0i from doublet "
              "content [1, 1, 1]")
    assert out.count(lowest + "\n") == 2
    assert out.count("host cell [0, 1, 2, 3] lowest over every (content, "
                     "doublet content) pair") == 2
    assert out.count("s_N=6+0i (content [3, 0, 0], doublet content "
                     "[1, 1, 1]) s_D=4+0.5i (content [3, 0, 0], doublet "
                     "content [0, 2, 1])") == 2
    R.main(["run", "--ticks", "1", "--quiet"])
    assert capsys.readouterr().out == ""


def _tick_with_reads():
    return {"tick": 0, "reads": READ,
            "summary": {"response_vertices": 2, "grown_cells": 0,
                        "row_sum_defects": []}}


def test_the_frame_data_carries_every_pole_of_every_doublet_content():
    data = R.frame_data([_tick_with_reads()], 0)
    marks = {(m["group"], tuple(m["doublet_content"]), m["spin"],
              m["column"], m["pole"]) for m in data["marks"]}
    group = "0123\n300"
    assert marks == {
        (group, (0, 2, 1), THREE, "quasi_free", 4.0 + 0.5j),
        (group, (1, 1, 1), HALF, "quasi_free", 6.0 + 0j),
        (group, (1, 1, 1), THREE, "quasi_free", 5.0 + 0.25j),
        (group, (1, 1, 1), HALF, "with_quartic", -2.0 + 0j),
        (group, (1, 1, 1), THREE, "with_quartic", -1.0 + 0j)}
    # the content with no value keeps a slot of its own, labelled as such
    assert data["slots"] == [(0.0, "no value"), (2.0, "021"), (3.0, "111")]
    assert [g["label"] for g in data["groups"]] == ["0123\n030", group]
    # each group carries the solve behind it: the library names no value at
    # 030, and the solve of 300 did not converge in its 40 iterations
    assert [g["solve"] for g in data["groups"]] == [
        {"state": "no value", "reason": None, "iterations": None},
        {"state": "not converged", "reason": None, "iterations": 40}]
    quasi_free = [r for r in data["ratios"] if r["column"] == "quasi_free"]
    # by 2T reading: the lowest pole of a sector restricting to a 2 is the
    # spin-3/2 pole of doublet content (0, 2, 1), whose sector restricts to
    # 2'' + 2 (the tetrahedral ambiguity), and the Delta reading is the
    # 2' + 2'' sector of (1, 1, 1)
    assert [bp.ratio_pair_text(r) for r in quasi_free] == \
        ["N 300|021 / D 300|111"]
    assert [bp.unconverged_poles(r) for r in quasi_free] == [["N", "D"]]
    assert data["counts"] == [{"tick": 0, "response_vertices": 2,
                               "grown_cells": 0, "row_sum_defects": []}]


def test_the_drawn_frame_has_one_mark_per_pole():
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    figure = plt.figure(figsize=bp.FIGURE_SIZE)
    try:
        R.draw_frame(figure, [_tick_with_reads()], 0)
        quasi_free, quartic = figure.axes[:2]
        for axis, expected in ((quasi_free, [4.0, 5.0, 6.0]),
                               (quartic, [-2.0, -1.0])):
            drawn = sorted(float(y) for line in axis.get_lines()
                           if line.get_label() in ("spin 1/2", "spin 3/2")
                           for y in line.get_ydata())
            assert drawn == expected
        labels = [t.get_text() for t in quasi_free.get_xticklabels(minor=True)]
        assert labels == ["no value", "021", "111"]
        # a callout over each group names its solve
        assert [t.get_text() for t in quasi_free.texts] == [
            "\u2717 no value", "\u2717 not converged\n40 iterations"]
    finally:
        plt.close(figure)


def test_main_live_refuses_a_file_backend_by_name(stub_reads, monkeypatch):
    import matplotlib
    monkeypatch.setattr(matplotlib, "get_backend", lambda: "agg")
    with pytest.raises(RuntimeError, match="--live needs an interactive"):
        R.main(["run", "--ticks", "1", "--live", "--quiet"])
    assert stub_reads == []


def test_main_live_runs_the_live_drive(stub_reads, monkeypatch, tmp_path):
    calls = []

    def live(config, progress=False, points_file=None, keep_open=False):
        calls.append((config["ticks"], progress, points_file, keep_open))
        return R.drive(config, points_file=points_file)

    monkeypatch.setattr(R, "drive_live", live)
    path = tmp_path / "live.json"
    held = []
    monkeypatch.setattr(R.bp, "hold_live_window",
                        lambda message: held.append(path.exists()))
    R.main(["run", "--ticks", "1", "--live", "--quiet", "--json", str(path)])
    assert calls == [(1, False, str(tmp_path / "live.points.jsonl"), True)]
    assert held == [True]


def test_a_stopped_drive_says_so():
    result = R.drive(R.default_config(ticks=2), stop_requested=lambda: True)
    assert result["stopped"] is True and result["ticks"] == []


def test_the_summary_of_a_failed_grown_cell_and_a_stop():
    record = {
        "tick": 3, "relaxation": {"converged": True, "residual": 0.0},
        "level": {"vertices": 4, "edges": 6, "tetrahedra": 1,
                  "declared_monopole_numbers": [1],
                  "monopole_numbers": [1]},
        "partition": {"partition": [[0], [1]], "selected_resolution": 1.0,
                      "bands_accepted": [True, True],
                      "isolation_gaps": [1.0, 1.0],
                      "determinant_residual": 0.0},
        "summary": {"response_vertices": 2, "interactions": 1,
                    "grown_cells": 0},
        "grown_cells": [{"vertices": [0, 1, 2, 3], "failed": "singular"}],
        "reads": [], "stopped": "no grown 3-simplex"}
    text = R.summary({"host": {"monopole_numbers": [1]}, "ticks": [record]})
    assert "cell [0, 1, 2, 3] failed: singular" in text
    assert text.endswith("  stopped: no grown 3-simplex")


def test_jsonable_turns_arrays_into_nested_lists():
    out = R._jsonable({(0, 1): np.array([[1 + 1j, 2]]), "x": (np.int64(2),)})
    assert out == {"(0, 1)": [[{"re": 1.0, "im": 1.0}, {"re": 2.0, "im": 0.0}]],
                   "x": [2]}
    json.dumps(out)


# ------------------------------------------------------------ the host


def test_the_oriented_face_sign():
    index = {(0, 1, 2): 7}
    assert R._oriented_face((0, 1, 2), index) == (7, 1)
    assert R._oriented_face((1, 0, 2), index) == (7, -1)
    assert R._oriented_face((1, 2, 0), index) == (7, 1)


@pytest.mark.parametrize("count", [1, 2, 3])
def test_the_dirac_string_is_integer_and_the_flux_is_two_pi(count):
    cells = R.fan(count)
    connection = R.monopole_connection(cells)
    assert connection["residual"] < 1e-12
    assert np.all(connection["dirac_string"] == np.rint(
        connection["dirac_string"]))
    assert len(connection["edges"]) == 3 + 3 * count
    assert len(connection["faces"]) == 1 + 3 * count


# ------------------------------------------------------------ the band helpers


def test_the_riesz_band_selects_the_lowest_modes():
    """The two lowest eigenvalues, 1 and 2, are the band; the selection is
    recorded as the circle about their mean 1.5 whose radius 1 is halfway
    between the farthest selected (distance 0.5) and the nearest excluded
    eigenvalue 3 (distance 1.5); the isolation gap is the distance 1 between
    2 and 3. The projector is exact: the block is diagonal, so its Schur form
    is the block itself, the reordering is a permutation, and the projector
    is diag(0, 1, 1, 0) to the declared tolerance 1e-15."""
    block = np.diag([3.0, 1.0, 2.0, 9.0]).astype(complex)
    band = R.riesz_band(block, 2, 1e-15)
    assert [v.real for v in band["eigenvalues"]] == [1.0, 2.0]
    assert np.abs(band["projector"]
                  - np.diag([0, 1, 1, 0]).astype(complex)).max() < 1e-15
    assert np.abs(band["left"] @ band["frame"] - np.eye(2)).max() < 1e-15
    assert band["radius"] == 1.0
    assert band["centre"] == 1.5
    assert band["isolation_gap"] == 1.0
    assert not band["encloses_everything"]
    assert band["projector_idempotency"] < 1e-15
    assert band["invariant_subspace_residual"] < 1e-15
    assert band["accepted"]


def test_a_band_of_the_whole_block_has_nothing_excluded():
    """A band rank above the block's order selects every eigenvalue: the
    projector is the identity, and with no excluded eigenvalue to measure to
    the recorded radius and the isolation gap are infinite."""
    band = R.riesz_band(np.diag([1.0, 3.0]).astype(complex), 5, 1e-15)
    assert band["frame"].shape == (2, 2)
    assert np.abs(band["projector"] - np.eye(2)).max() < 1e-15
    assert band["encloses_everything"]
    assert band["radius"] == math.inf and band["isolation_gap"] == math.inf


def test_a_band_that_splits_an_exactly_multiple_eigenvalue_has_no_value():
    """diag(1, 1, 3) with band rank one: the two eigenvalues 1 are equal
    exactly, so the band would take one and leave the other out. No rank-one
    part of their eigenspace is an invariant subspace of its own, the
    Sylvester equation of the projector is singular, and the library says by
    name that the projector has no value."""
    with pytest.raises(ValueError, match="leaves an eigenvalue exactly "
                                         "equal to it out"):
        R.riesz_band(np.diag([1.0, 1.0, 3.0]).astype(complex), 1, 1e-15)


def test_a_band_through_a_near_degenerate_pair_is_read_and_not_accepted():
    """diag(1, 1 + 2^-50, 3) with band rank one at the tolerance 1e-15: the
    block's Frobenius norm is about 3.317, so eigenvalues at most 3.317e-15
    apart are equal at the tolerance, and 1 and 1 + 2^-50 are 8.9e-16 apart.
    The band is read all the same: it is the eigenvalue 1, the first in the
    order of the exact keys, with the projector diag(1, 0, 0) of the diagonal
    block and zero residuals. The read reports the isolation gap 2^-50 and
    ``accepted`` false."""
    gap = 2.0 ** -50
    band = R.riesz_band(np.diag([1.0, 1.0 + gap, 3.0]).astype(complex), 1,
                        1e-15)
    assert band["eigenvalues"] == [1.0]
    assert np.array_equal(band["projector"], np.diag([1.0, 0.0, 0.0]))
    assert band["isolation_gap"] == gap
    assert band["projector_idempotency"] == 0.0
    assert band["invariant_subspace_residual"] == 0.0
    assert not band["accepted"]


def test_the_fibers_of_a_partition_are_supported_on_their_images():
    """Each fiber's geometric image is zero off its component, its two images
    pair to det((Z^vee)^T Z) = 3, and its left frame is the dual of its chain
    frame."""
    config = R.default_config(tetrahedra=2)
    cells, z, links, _ = R.level_zero(config)
    base = R.base_operator(cells, z, links)
    partition = [list(p) for p in R.recursion_turn(base["operator"],
                                                   config).partition]
    fibers = R.fibers_for_partition(
        base, partition, 1,
        bp.declared_tolerance(config, "recursion_tolerance"))
    assert len(fibers["frames"]) == len(partition) == len(fibers["reads"])
    for image, dual_image, frame, left, part in zip(
            fibers["images"], fibers["dual_images"], fibers["frames"],
            fibers["lefts"], partition):
        assert np.linalg.det(dual_image.T @ image) == pytest.approx(
            R.IMAGE_PAIRING_UNIT, abs=1e-10)
        np.testing.assert_allclose(left @ frame, np.eye(frame.shape[1]),
                                   atol=1e-10)
        outside = [i for i in range(image.shape[0]) if i not in part]
        assert np.all(image[outside] == 0) and np.all(left[:, outside] == 0)
    reads = fibers["reads"]
    assert max(r["dual_eigenvalue_mismatch"] for r in reads) < 1e-8
    transports = R.transport_matrix(base["pencil"], fibers["images"],
                                    fibers["lefts"])
    assert len(transports) == len(partition) ** 2
    # the pencil form of the transfer is the chain form Y~^T h_1 Y
    for (v, w), block in transports.items():
        np.testing.assert_allclose(
            fibers["lefts"][v] @ base["operator"] @ fibers["frames"][w],
            block, rtol=1e-12, atol=1e-12 * np.abs(block).max())


def test_the_level_record_is_consistent():
    config = R.default_config(tetrahedra=2)
    cells, z, links, _ = R.level_zero(config)
    level = R.recursion_turn(R.base_operator(cells, z, links)["operator"],
                             config)
    record = R.level_record(level)
    assert len(record["partition"]) == len(record["band_ranks"]) \
        == len(record["bands_accepted"]) == len(record["isolation_gaps"])
    assert sorted(i for p in record["partition"] for i in p) == \
        list(range(len(z)))
    assert record["resolutions"] == list(R.DECLARED_RESOLUTIONS)
    assert record["selected_resolution"] in record["resolutions"]
    json.dumps(R._jsonable(record))


def test_a_failed_grown_cell_is_recorded_and_skipped():
    """A pairing block the length inversion cannot read is recorded with its
    reason, contributes no edge, and does not stop the other cells."""
    bad = np.full((4, 4), np.nan, dtype=complex)
    reads, z, links, spread, groupoid = R.grow([(0, 1, 2, 3)], bad, {})
    assert "failed" in reads[0]
    assert z == {} and links == {} and spread == {} and groupoid == {}
