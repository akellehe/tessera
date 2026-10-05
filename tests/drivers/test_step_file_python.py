# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The per-step records of a run's solves, streamed to a step file (#1421).

Terms used below:

* a *step proposal* is one stage-2 update of a drive
  (`cell_solve.StationarityObjective.direction`): a point at which a step is
  formed, with its record (the residual norm, the step's rank decisions,
  the base complex, the point's measurements);
* a *solve record* is what a solve leaves in a run's records: a content's
  ``relaxation`` (`baryon_poles.relaxation_record`, whose per-step entries
  are ``trace``) or a level's (`recursion.relax_level`, whose per-step
  entries are ``jacobian_ranks``, ``rank_gaps`` and ``term_trace``);
* the *step file* (`baryon_poles.StepSink`) holds one JSON line per step
  proposal, written when the proposal is made; under one, a solve record
  holds a reference to its lines (``steps``) in place of its per-step
  entries;
* *bit for bit after JSON*: the JSON text of a line's step fields is the
  JSON text the drivers write for the in-memory entry (`_jsonable`, no NaN
  or infinity as a number).

The solves are those of `tests.cobordism.test_action_term_trace_python`:
the content (0, 3, 0) on the host cell (0134) of the 2026-09-23 run, ten
step proposals, and the level relaxation of the same host, every hinge in
the Regge sum, each with the action's terms traced.
"""
import gc
import json

import pytest

from tessera import cobordism as cob
from tessera.drivers import baryon_poles as bp
from tessera.drivers import cell_solve as cs
from tessera.drivers import recursion as R

from tests.drivers import _recursion_run_2026_09_23 as RUN

#: The keys of a line that say where it is: the place in the run, the
#: solve and the step's index in it.
PLACE = {"kappa", "beta", "tick", "level", "cell", "content", "solve",
         "step"}


def _host(content=(0, 3, 0)):
    config = bp.default_config([1.0], [1.0], selected_contents=[content],
                               fiber_moments="r", fiber_pinning="power-sums",
                               tolerances=RUN.TOLERANCES, trace_terms=True)
    config["host_cell"] = RUN.HOST_CELLS[(0, 1, 3, 4)]
    config["held_sectors"] = R.held_sectors([[0, 1, 2, 3]], [1], 4)
    return config


def _level_config(config):
    return dict(config, kappa=1.0, beta=1.0, regge_hinges="all")


def _content(config):
    _, _, report, drive = bp.relax_content((0, 3, 0), 1.0, 1.0, config)
    return bp.relaxation_record(report, drive)


def _level(config):
    spacetime = bp.build_host(config["edge_squared"], config["host_cell"])
    return R.relax_level(spacetime, _level_config(config), [], count=4)


@pytest.fixture(scope="module")
def solves(tmp_path_factory):
    """The level relaxation and the content's solve without a step file,
    and the same two in one step file, the level first, named as the
    recursion names them."""
    config = _host()
    plain = {"content": _content(config), "level": _level(config)}
    path = tmp_path_factory.mktemp("steps") / "run.steps.jsonl"
    with bp.declared_steps(path):
        with bp.steps_at(tick=0):
            with bp.steps_at(level=0):
                level = _level(config)
            with bp.steps_at(cell=[0, 1, 3, 4], content=[0, 3, 0]):
                content = _content(config)
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    return plain, {"content": content, "level": level}, path, lines


def _text(value):
    return json.dumps(bp._jsonable(value), allow_nan=False)


def _fields(line):
    return {key: value for key, value in line.items() if key not in PLACE}


def test_a_content_solve_writes_the_entries_its_record_holds(solves):
    """Every line of a content's solve is, field for field and bit for bit
    after JSON, the entry its record holds without a step file, in order;
    the line names the tick, the cell, the content, the solve and the
    step."""
    plain, streamed, path, lines = solves
    trace = plain["content"]["trace"]
    reference = streamed["content"]["steps"]
    assert len(trace) == reference["count"] == 10
    ours = lines[reference["first_line"]:][:reference["count"]]
    for step, (entry, line) in enumerate(zip(trace, ours)):
        assert {key: line[key] for key in line if key in PLACE} == {
            "tick": 0, "cell": [0, 1, 3, 4], "content": [0, 3, 0],
            "solve": 1, "step": step}
        assert list(line)[:5] == ["tick", "cell", "content", "solve", "step"]
        assert json.dumps(_fields(line), allow_nan=False) == _text(entry)
    assert all(entry["terms"] for entry in trace)


def test_a_level_relaxation_writes_the_entries_its_record_holds(solves):
    """Every line of a level's relaxation holds the Jacobian's rank, its
    rank gap and the action's terms of one step proposal, bit for bit after
    JSON the entries of the level's record without a step file."""
    plain, streamed, path, lines = solves
    level = plain["level"]
    reference = streamed["level"]["steps"]
    count = len(level["jacobian_ranks"])
    assert count == len(level["rank_gaps"]) == len(level["term_trace"])
    assert reference["count"] == count > 10
    for step, line in enumerate(lines[:count]):
        assert {key: line[key] for key in line if key in PLACE} == {
            "tick": 0, "level": 0, "solve": 0, "step": step}
        assert json.dumps(_fields(line), allow_nan=False) == _text({
            "jacobian_rank": level["jacobian_ranks"][step],
            "rank_gap": level["rank_gaps"][step],
            "terms": level["term_trace"][step]})


def test_the_records_refer_to_their_lines_and_hold_the_rest(solves):
    """Under a step file a solve record holds, in place of its per-step
    entries, the file, the solve's number, the index and byte offset of its
    first line and the number of its lines; every other key is the record
    without a step file, the residual-norm trace and the aggregates of the
    steps included (all but the seconds the solve took)."""
    plain, streamed, path, lines = solves
    level, content = streamed["level"]["steps"], streamed["content"]["steps"]
    count = len(plain["level"]["jacobian_ranks"])
    assert level == {"file": str(path), "solve": 0, "first_line": 0,
                     "offset": 0, "count": count, "term_lines": count}
    first = path.read_bytes().split(b"\n")[:count]
    assert content == {"file": str(path), "solve": 1, "first_line": count,
                       "offset": sum(len(line) + 1 for line in first),
                       "count": 10, "term_lines": 10}
    assert len(lines) == count + 10
    assert [_fields(line) for line in bp.step_lines(content)] == \
        [_fields(line) for line in lines[count:]]
    per_step = {"content": {"trace", "steps", "seconds"},
                "level": {"jacobian_ranks", "rank_gaps", "term_trace",
                          "steps", "seconds"}}
    for name, keys in per_step.items():
        assert _text({k: v for k, v in plain[name].items() if k not in keys}) \
            == _text({k: v for k, v in streamed[name].items()
                      if k not in keys})
    assert set(plain["content"]) - set(streamed["content"]) == {"trace"}
    assert set(plain["level"]) - set(streamed["level"]) == {
        "jacobian_ranks", "rank_gaps", "term_trace"}


def test_the_log_lines_are_the_same_read_from_the_file(solves):
    """The --trace-terms lines of a solve and its summary line, read from
    the step file, are the lines the record without a step file prints."""
    plain, streamed, _, _ = solves
    for name in ("content", "level"):
        lines = bp.term_trace_lines(plain[name], "    ")
        assert lines and bp.term_trace_lines(streamed[name], "    ") == lines
    assert bp.relaxation_text(plain["content"]) == \
        bp.relaxation_text(streamed["content"])
    assert "of 10 iterates" in bp.relaxation_text(streamed["content"])


def _retained(objective):
    """The number of objects the objective holds, its system and its step
    writer apart (`gc.get_referents`)."""
    skip = {id(objective._system), id(objective.steps)}
    seen = set()
    stack = [value for key, value in vars(objective).items()
             if key not in ("_system", "steps")]
    while stack:
        item = stack.pop()
        if id(item) in seen or id(item) in skip or isinstance(item, type):
            continue
        seen.add(id(item))
        stack.extend(gc.get_referents(item))
    return len(seen)


def _proposals(sink, path, count=300, marks=(30, 300)):
    """``count`` step proposals of the level system of the host (0134),
    each with the action's terms, made at the host as the engine makes them
    (`StationarityObjective.direction`), with a step file at ``path`` when
    ``sink``: the objective, its step writer, and the number of objects it
    holds after each proposal of ``marks``."""
    config = _level_config(_host())
    spacetime = bp.build_host(config["edge_squared"], config["host_cell"])
    base = R.sheet_base(spacetime, 4)
    system, _ = R.level_system(base, config, [], bp.SHEETS)
    system.begin(base)
    writer = bp.StepSink(path).begin(R.level_step_record) if sink else None
    objective = cs.StationarityObjective(system, steps=writer)
    scalar = cob.ObjectiveContext()
    scalar.spacetime = base
    context = cob.ObjectiveDirectionContext()
    context.scalar = scalar
    context.edge_count = len(cs.edge_fields(base))
    held = []
    for proposal in range(1, count + 1):
        objective.direction(context)
        if proposal in marks:
            gc.collect()
            held.append(_retained(objective))
    return objective, writer, held


def test_the_objective_holds_no_step_proposal_under_a_step_file(tmp_path):
    """Three hundred step proposals at one point: without a step file the
    objective holds every record, the objects it holds growing by 17 per
    proposal (4591 from the thirtieth to the three hundredth); with one it
    holds no record and the objects it holds grow by fewer than one
    proposal's (by one: Python shares the integers up to 256 between the
    places that hold them, and a count past 256 is an object of its own).
    The proposals are made at one point, so the residual-norm trace, one
    entry per distinct point, holds one entry either way."""
    held, _, objects = _proposals(False, None)
    assert len(held.updates) == held.proposals == 300
    grown = objects[1] - objects[0]
    assert grown >= 17 * 270
    written, writer, objects = _proposals(True, tmp_path / "run.steps.jsonl")
    assert written.updates == [] and written.proposals == 300
    assert writer.count == writer.term_lines == 300
    assert 0 <= objects[1] - objects[0] < grown / 270
    for objective in (held, written):
        assert objective.proposed_trace() == [
            objective.last_update["residual_norm"]]
    assert len((tmp_path / "run.steps.jsonl").read_text().splitlines()) \
        == 300


def _cheap_reads(content, kappa, beta, config, started, spacetime, action,
                 report, solve, flags, stage):
    """The reads of a solved cell replaced by its solve record alone."""
    return {"content": list(content), "relaxation": solve, "flags": flags,
            "doublet_reads": []}


def test_a_scan_with_a_step_file_records_its_path_and_refers_to_it(
        monkeypatch, tmp_path):
    """`baryon_poles.drive` with a step file: the configuration (and so
    the points file's first line) records its path, every content's solve
    record refers to its lines, each named by the scan point and the
    content, and the scan's lines, the --trace-terms lines among them, are
    the lines the same scan without a step file prints."""
    monkeypatch.setattr(bp, "_content_reads", _cheap_reads)
    config = bp.default_config([1.0], [1.0], selected_contents=[(0, 3, 0)],
                               tolerances=RUN.TOLERANCES, trace_terms=True)
    steps = tmp_path / "run.steps.jsonl"
    points = tmp_path / "run.points.jsonl"
    result = bp.drive(config, points_file=str(points),
                      steps_file=str(steps))
    plain = bp.drive(config)
    assert "steps_file" not in config and "steps_file" not in plain["config"]
    assert result["config"]["steps_file"] == str(steps)
    header = json.loads(points.read_text().splitlines()[0])
    assert header["config"]["steps_file"] == str(steps)
    (record,) = result["points"][0]["contents"]
    reference = record["relaxation"]["steps"]
    lines = [json.loads(line) for line in steps.read_text().splitlines()]
    assert reference["file"] == str(steps) and reference["solve"] == 0
    assert reference["count"] == len(lines) >= 1
    assert all(line["kappa"] == 1.0 and line["beta"] == 1.0
               and line["content"] == [0, 3, 0] for line in lines)
    assert "trace" not in record["relaxation"]
    (held,) = plain["points"][0]["contents"]
    assert [_text(entry) for entry in held["relaxation"]["trace"]] == \
        [json.dumps(_fields(line), allow_nan=False) for line in lines]
    assert any("iterate 0: S = " in line
               for line in bp.point_lines(result["points"][0]))
    assert bp.point_lines(result["points"][0]) == \
        bp.point_lines(plain["points"][0])


@pytest.mark.parametrize("driver", [bp, R], ids=["baryon_poles", "recursion"])
def test_a_run_with_json_declares_the_step_file_beside_it(
        driver, monkeypatch, tmp_path):
    """`main` with --json declares the step file beside the JSON file
    (``<json stem>.steps.jsonl``), and without it none."""
    seen = []

    def drive(config, **options):
        seen.append(options.get("steps_file"))
        return {"config": config, "host": {"monopole_numbers": []},
                "points": [], "ticks": [], "stopped": False}

    monkeypatch.setattr(driver, "drive", drive)
    driver.main(["run", "--quiet", "--json", str(tmp_path / "run.json")])
    driver.main(["run", "--quiet"])
    assert seen == [str(tmp_path / "run.steps.jsonl"), None]
    assert bp.steps_path("a/run.json") == "a/run.steps.jsonl"


def test_without_a_step_file_the_records_hold_the_entries():
    """No step sink is in force outside a declared one, and a place named
    without one names nothing."""
    assert bp.step_sink() is None
    with bp.steps_at(tick=3) as sink:
        assert sink is None and bp.step_sink() is None
