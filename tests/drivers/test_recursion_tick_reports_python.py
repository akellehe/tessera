# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""What a tick of the level recursion (`tessera.drivers.recursion`) records
when one of its steps has no value, and what its records name.

A step that has no value is one at which the library or the driver raises
one of `baryon_poles.NO_VALUE_ERRORS`. The tick records it by name with the
reason, makes every read that does not depend on it, and writes its record;
any other error is a defect of the code and is raised with what the tick
reached. The steps are replaced here by stand-ins that raise, on the declared
fan of two as built, so that every test is the tick's own bookkeeping."""
import itertools
import json
import types

import numpy as np
import pytest

import tessera as T
from tessera.drivers import baryon_poles as bp
from tessera.drivers import cell_solve as cs
from tessera.drivers import recursion as R


def _still(monkeypatch, **extra):
    """Replace the level's relaxation by a record that leaves the level as
    built (the fan of two is stationary as built), with ``extra`` entries."""
    def relax(spacetime, config, sectors=None, count=None):
        record = {
            "converged": True, "residual": 0.0, "stop_reason": "converged",
            "stop_detail": "the residual norm is 0", "accepted_updates": 0,
            "moves_committed": 0, "moved_base": None,
            "sector_monopole_numbers": [int(s.monopole_number)
                                        for s in sectors or []],
            "regge_structurally_zero": True, "regge_hinge_count": 0,
            "regge_hinges": "interior"}
        record.update(extra)
        return record
    monkeypatch.setattr(R, "relax_level", relax)


def _read(cell):
    """A cell's read with no content, as `cell_reads` returns one."""
    return {"cell": sorted(cell), "host_cell": {}, "failed_contents": [],
            "flagged_contents": [], "contents": [], "ratios": {},
            "pole_table": {}}


def _reads(monkeypatch):
    """Replace the per-cell reads by one empty read per cell, and return the
    list of the calls made."""
    calls = []

    def reads(cells, z, links, config):
        calls.append((cells, z, links))
        return [_read(cell) for cell in cells]
    monkeypatch.setattr(R, "cell_reads", reads)
    return calls


def _host(**options):
    options.setdefault("pachner_updates", 0)
    config = R.default_config(tetrahedra=2, **options)
    cells, z, links, _ = R.level_zero(config)
    return config, cells, z, links


def _text(record):
    return R.summary({"host": {"monopole_numbers": [1, 1]},
                      "ticks": [record]})


# ------------------------------------------------- a step without a value


def test_a_level_with_no_value_as_built_is_recorded():
    """A level one of whose links is zero has no phase to carry: the tick
    stops before its relaxation and says so, with the level as given."""
    config, cells, z, links = _host()
    links = dict(links)
    links[(0, 1)] = 0.0
    record, following = R.tick(0, cells, z, links, config)
    assert following is None
    assert record["stopped"].startswith("the level has no value as built: ")
    assert "relaxation" not in record and record["reads"] == []
    assert record["level"]["cells"] == [[0, 1, 2, 3], [0, 1, 3, 4]]
    assert record["summary"] == {
        "response_vertices": 0, "interactions": 0, "grown_cells": 0,
        "failed_cells": 0, "rejected_cells": 0, "row_sum_defects": []}
    json.dumps(R._jsonable(record))
    assert _text(record).endswith("tick 0: stopped: " + record["stopped"])
    assert R.frame_data([record], 0)["counts"] == [
        {"tick": 0, "response_vertices": 0, "grown_cells": 0,
         "row_sum_defects": []}]


@pytest.mark.parametrize("error", [ValueError, RuntimeError,
                                   ZeroDivisionError])
def test_a_relaxation_without_a_value_stops_the_tick_by_name(monkeypatch,
                                                              error):
    """Every way the library says a relaxation has no value is recorded as
    the tick's stop: `std::invalid_argument` (ValueError),
    `std::runtime_error` (RuntimeError) and a zero divisor."""
    def refuse(spacetime, config, sectors=None, count=None):
        raise error("no stationary point")
    monkeypatch.setattr(R, "relax_level", refuse)
    config, cells, z, links = _host()
    record, following = R.tick(0, cells, z, links, config)
    assert following is None
    assert record["relaxation"] == {"failed": "no stationary point"}
    assert record["stopped"] == \
        "the level's relaxation was refused: no stationary point"
    assert record["level"]["bulk_monopole_numbers_before"] == [1, 1]
    assert len(record["level"]["face_holonomies"]) == 7


@pytest.mark.parametrize("step, message, stopped", [
    ("base_operator", "CovariantChainHodge: Proposition 3 fails",
     "the level's operator has no value: "),
    ("recursion_turn", "LevelRecursion::readBand: no projector",
     "the recursion's turn has no value: "),
    ("level_record", "the certificate has no description",
     "the recursion's turn has no value: "),
])
def test_a_box_without_a_value_keeps_the_level_and_reads_its_cells(
        monkeypatch, step, message, stopped):
    """The operator of the relaxed level, and the turn of the Section 15 box
    on it, are steps the cells' reads do not depend on: when one has no
    value the tick records why, keeps the level and its relaxation, reads
    every cell of the level on its relaxed fields, and returns no next
    level."""
    _still(monkeypatch)
    calls = _reads(monkeypatch)

    def no_value(*args, **kwargs):
        raise RuntimeError(message)
    monkeypatch.setattr(R, step, no_value)
    config, cells, z, links = _host()
    record, following = R.tick(0, cells, z, links, config)
    assert following is None
    assert record["partition"] == {"failed": message}
    assert record["stopped"] == stopped + message
    assert record["relaxation"]["converged"]
    assert [read["cell"] for read in record["reads"]] == \
        [[0, 1, 2, 3], [0, 1, 3, 4]]
    (call,) = calls
    assert call[0] == cells and set(call[1]) == set(z)
    level = record["level"]
    assert (level["vertices"], level["edges"], level["tetrahedra"]) == \
        (5, 9, 2)
    assert level["bulk_monopole_numbers_after"] == [1, 1]
    assert "grown_cells" not in record and "interaction" not in record
    summary = record["summary"]
    assert summary["response_vertices"] == summary["grown_cells"] == 0
    assert summary["held_cut_monopole_numbers"] == [2, 2]
    assert summary["quark_verdicts"] == [[], []]
    json.dumps(R._jsonable(record))
    text = _text(record)
    assert "tick 0: level with 5 vertices, 9 edges, 2 tetrahedra" in text
    assert text.endswith("  stopped: " + stopped + message)


def test_an_interaction_stage_without_a_value_keeps_the_partition_and_reads(
        monkeypatch):
    """The transports, the pairing and the grown-cell rule are one stage:
    when it has no value the partition of the turn and the reads of the
    cells are kept, and the tick stops there by name."""
    _still(monkeypatch)
    _reads(monkeypatch)

    def no_transport(pencil, images, lefts):
        raise ValueError("the transports have no value")
    monkeypatch.setattr(R, "transport_matrix", no_transport)
    config, cells, z, links = _host(persistence_required=1)
    record, following = R.tick(0, cells, z, links, config)
    assert following is None
    assert record["interaction"] == {"failed": "the transports have no value"}
    assert record["stopped"] == \
        "the interaction stage has no value: the transports have no value"
    covered = sorted(i for part in record["partition"]["partition"]
                     for i in part)
    assert covered == list(range(9))
    assert sorted(i for part in record["response_components"]
                  for i in part) == covered
    assert len(record["reads"]) == 2
    assert "grown_cells" not in record
    assert record["summary"]["response_vertices"] == 0
    json.dumps(R._jsonable(record))
    assert "partition " in _text(record)


def test_a_component_whose_fiber_has_no_value_is_not_a_response_vertex(
        monkeypatch):
    """A component whose fiber has no value is named with its edges and the
    reason, and the stage is made on the other components: their fibers,
    transports and pairing are those the stage gives without it."""
    config, cells, z, links = _host(persistence_required=1)
    base = R.base_operator(cells, z, links)
    turn = R.recursion_turn(base["operator"], config)
    partition, _ = R.persistent_components(turn, base["edges"], 1)
    assert len(partition) >= 2
    lost = partition[0]
    whole = R.fibers_for_partition

    def fibers(base_, parts, rank, tolerance):
        if list(parts[0]) == list(lost):
            raise ValueError("the band separates two equal eigenvalues")
        return whole(base_, parts, rank, tolerance)
    monkeypatch.setattr(R, "fibers_for_partition", fibers)
    stage = R.interaction_stage(base, partition, config)
    assert stage["partition"] == [list(p) for p in partition[1:]]
    assert stage["failed_components"] == [{
        "component": 0,
        "edges": ["%d-%d" % base["edges"][c] for c in lost],
        "failed": "the band separates two equal eigenvalues"}]
    assert len(stage["frames"]) == len(partition) - 1
    monkeypatch.setattr(R, "fibers_for_partition", whole)
    without = R.interaction_stage(base, partition[1:], config)
    assert stage["pairs"] == without["pairs"]
    np.testing.assert_array_equal(stage["pairing"], without["pairing"])

    # the tick names the component and counts the response vertices left
    monkeypatch.setattr(R, "fibers_for_partition", fibers)
    _still(monkeypatch)
    _reads(monkeypatch)
    record, _ = R.tick(0, cells, z, links, config)
    assert record["failed_components"] == stage["failed_components"]
    assert record["response_components"] == stage["partition"]
    assert record["summary"]["failed_components"] == 1
    assert record["summary"]["response_vertices"] == len(partition) - 1
    assert any("is not a response vertex: its fiber has no value: the band "
               "separates two equal eigenvalues" in line
               for line in R._notices(record))


def test_a_read_of_the_relaxed_level_without_a_value_is_named(monkeypatch):
    """A read of the relaxed level that has no value is None, the level names
    it with the reason, and the tick goes on."""
    _still(monkeypatch)
    _reads(monkeypatch)

    def no_shift(cells, z, links, rank_tolerance):
        raise ValueError("the rule has no value on this cell")
    monkeypatch.setattr(R, "level_rule_shift", no_shift)
    config, cells, z, links = _host()
    record, _ = R.tick(0, cells, z, links, config)
    level = record["level"]
    assert level["rule_shift"] is None
    assert level["without_value"] == {
        "rule_shift": "the rule has no value on this cell"}
    assert record["summary"]["rule_shift_on_level"] is None
    assert "partition" in record["partition"]
    assert "the level's rule_shift has no value: the rule has no value on " \
        "this cell" in R._notices(record)
    json.dumps(R._jsonable(record))


def test_a_cell_whose_read_has_no_value_is_recorded_and_the_others_are_read(
        monkeypatch):
    """A cell whose read as a host has no value before any content is
    reached is recorded with the reason, and the other cells are read."""
    def scan(kappa, beta, config):
        if config["host_cell"]["links"][0] == links[(0, 1)] and \
                not seen:
            seen.append(True)
            raise RuntimeError("the monopole read did not converge")
        return {"failed_contents": [], "flagged_contents": [],
                "contents": [], "ratios": {}, "pole_table": {}}
    seen = []
    monkeypatch.setattr(bp, "scan_point", scan)
    config, cells, z, links = _host()
    reads = R.cell_reads(cells, z, links, config)
    assert [read["cell"] for read in reads] == [[0, 1, 2, 3], [0, 1, 3, 4]]
    assert reads[0]["failed"] == "the monopole read did not converge"
    assert reads[0]["contents"] == [] and "failed" not in reads[1]
    record = {"tick": 0, "reads": reads,
              "summary": {"response_vertices": 0, "grown_cells": 0,
                          "row_sum_defects": []}}
    lines = R.read_lines(record)
    assert lines[0] == ("    host cell [0, 1, 2, 3] no value: the monopole "
                        "read did not converge")
    assert "host cell [0, 1, 2, 3] has no read: the monopole read did not " \
        "converge" in R._notices(record)
    assert R.frame_data([record], 0)["marks"] == []


def test_a_growth_step_without_a_value_leaves_the_grown_base(monkeypatch):
    """When the search over the Pachner moves of the growth step has no
    value, the grown base is the next level as it is, and the tick's record
    says why."""
    _still(monkeypatch)
    _reads(monkeypatch)

    def no_search(cells, z, links, config):
        raise RuntimeError("the objective is not finite")
    monkeypatch.setattr(R, "pachner_stage", no_search)
    config, cells, z, links = _host(persistence_required=1,
                                    pachner_updates=1)
    record, following = R.tick(0, cells, z, links, config)
    grown = [r for r in record["grown_cells"]
             if "failed" not in r and "rejected" not in r]
    assert len(grown) >= 1 and "stopped" not in record
    next_cells, next_z, next_links = following
    assert len(next_cells) == len(grown)
    assert set(next_z) == set(next_links) == {
        tuple(sorted(e)) for c in next_cells
        for e in itertools.combinations(c, 2)}
    pachner = record["pachner"]
    assert pachner["failed"] == "the objective is not finite"
    assert pachner["changed"] is False
    assert pachner["before"] == pachner["after"] == {
        "vertices": 1 + max(max(c) for c in next_cells),
        "edges": len(next_z), "cells": len(next_cells)}
    assert record["summary"]["pachner"]["failed"] == pachner["failed"]
    assert any(line.startswith("the growth step's search over the Pachner "
                               "moves has no value")
               for line in R._notices(record))
    json.dumps(R._jsonable(record))


def test_the_lineage_names_the_vertices_the_growth_step_leaves(monkeypatch):
    """A response vertex's lineage names a vertex of the next level. When the
    growth step's moves relabel the grown base, the lineage takes the labels
    they leave, and a vertex a move removed has none."""
    _still(monkeypatch)
    _reads(monkeypatch)
    seen = {}

    def moved(cells, z, links, config):
        count = 1 + max(max(c) for c in cells)
        seen["count"] = count
        # vertex 0 removed, the others moved down by one
        relabeling = {str(v): v - 1 for v in range(1, count)}
        return cells, z, links, {
            "updates": 1, "changed": True,
            "before": {"vertices": count, "edges": len(z),
                       "cells": len(cells)},
            "after": {"vertices": count - 1, "edges": len(z),
                      "cells": len(cells)},
            "vertex_relabeling": relabeling}
    monkeypatch.setattr(R, "pachner_stage", moved)
    config, cells, z, links = _host(persistence_required=1,
                                    pachner_updates=1)
    record, _ = R.tick(0, cells, z, links, config)
    lineage = record["lineage"]
    grown = sorted({v for r in record["grown_cells"]
                    if "failed" not in r and "rejected" not in r
                    for v in r["vertices"]})
    assert len(lineage) == record["summary"]["response_vertices"]
    for place, vertex in enumerate(grown):
        assert lineage[str(vertex)] == (None if place == 0 else place - 1)
    for vertex in set(range(len(lineage))) - set(grown):
        assert lineage[str(vertex)] is None


# ------------------------------------------------- an error that is a defect


def test_a_defect_is_raised_with_what_the_tick_reached(monkeypatch, tmp_path):
    """An error that is not one of `NO_VALUE_ERRORS` is a defect of the code:
    the tick raises it, carrying its record so far, and the drive appends
    that record to the points file before the error leaves it."""
    _still(monkeypatch)
    _reads(monkeypatch)

    def defect(cells, z, links):
        raise KeyError("a missing edge")
    monkeypatch.setattr(R, "base_operator", defect)
    config, cells, z, links = _host()
    with pytest.raises(KeyError) as caught:
        R.tick(0, cells, z, links, config)
    reached = caught.value.tick_record
    assert reached["tick"] == 0 and reached["relaxation"]["converged"]
    assert reached["level"]["bulk_monopole_numbers_after"] == [1, 1]
    assert reached["stopped"] == \
        "the tick ended on an error: KeyError: 'a missing edge'"
    assert "seconds" in reached

    points = tmp_path / "run.points.jsonl"
    with pytest.raises(KeyError):
        R.drive(config, points_file=str(points))
    lines = [json.loads(line) for line in points.read_text().splitlines()]
    assert len(lines) == 2 and "config" in lines[0]
    assert lines[1]["stopped"] == reached["stopped"]
    assert lines[1]["level"]["cells"] == [[0, 1, 2, 3], [0, 1, 3, 4]]


def test_what_a_tick_reached_is_written_whatever_it_holds():
    """The record of a tick that ended on a defect may hold an entry with no
    JSON form; it is written by its `repr`, and every other entry as the
    points file writes it."""
    marker = types.SimpleNamespace(a=1)
    reached = R._reached({"tick": 2, (0, 1): np.array([1.0 + 2.0j]),
                          "flag": np.bool_(True), "object": marker,
                          "nested": [{"object": marker}]})
    assert reached == {"tick": 2, "(0, 1)": [{"re": 1.0, "im": 2.0}],
                       "flag": True, "object": repr(marker),
                       "nested": [{"object": repr(marker)}]}
    json.dumps(reached)


def test_a_record_with_numpy_truth_values_and_arrays_has_a_json_form():
    record = {"accepted": np.bool_(False), "z": np.array([[1.0, 2.0]]),
              "n": np.int64(3), "w": np.complex128(1 - 1j)}
    assert bp._jsonable(record) == {
        "accepted": False, "z": [[1.0, 2.0]], "n": 3,
        "w": {"re": 1.0, "im": -1.0}}
    json.dumps(bp._jsonable(record))


# ------------------------------------------------------------ the grown cells


def _flat_pairing(cells, vertices):
    """The inherited pairing and the transports of flat tetrahedra on
    ``vertices`` points, as the manifold-gate test builds them."""
    from tessera import chainhodge as ch
    from tessera import cobordism as cob
    points = np.random.default_rng(8).normal(size=(vertices, 3))
    pairing = np.zeros((vertices, vertices), dtype=complex)
    single = cob.ChainComplex.fromTopCells([[0, 1, 2, 3]])
    for cell in cells:
        local_z = [complex(np.sum((points[a] - points[b]) ** 2))
                   for a, b in itertools.combinations(cell, 2)]
        block = np.asarray(ch.WhitneyMass.topSimplexBlocks(
            single, local_z, 1)[0].block)
        coboundary = np.zeros((6, 4))
        for a, (x, y) in enumerate(itertools.combinations(range(4), 2)):
            coboundary[a, x], coboundary[a, y] = -1.0, 1.0
        frames = block @ coboundary
        local = np.asarray(ch.GrownCellRule.determinantPairing(
            [frames[:, [v]] for v in range(4)],
            [np.linalg.solve(block, frames[:, [v]]) for v in range(4)]))
        for i, v in enumerate(cell):
            for j, w in enumerate(cell):
                pairing[v, w] = local[i, j]
    transports = {(v, w): np.array([[1.0 + 0j]])
                  for v, w in itertools.permutations(range(vertices), 2)}
    return pairing, transports


def test_a_cell_the_manifold_gate_leaves_unattached_keeps_its_read():
    """Three tetrahedra on one triangle: the third would make the triangle a
    face of three cells, so it is not attached and gives the next level no
    edge (WP v18 §15). Its read is made like the others' and recorded beside
    the violation."""
    cells = [(0, 1, 2, 3), (0, 1, 2, 4), (0, 1, 2, 5)]
    pairing, transports = _flat_pairing(cells, 6)
    reads, z, links, spread, groupoid = R.grow(cells, pairing, transports)
    assert ["rejected" in r for r in reads] == [False, False, True]
    assert reads[2]["rejected"] == R.manifold_violation(cells)
    for read in reads:
        assert len(read["squared_lengths"]) == 6
        assert np.isfinite(read["row_sum_defect"])
    # the pairing's entries on the shared triangle are those of the last
    # cell written, so the unattached cell's block is the flat cell's own
    assert reads[2]["row_sum_defect"] < 1e-12
    assert reads[2]["asymmetry"] < 1e-12
    assert (0, 5) not in z and (0, 3) in z and (0, 4) in z
    assert set(z) == set(links) == set(spread) == set(groupoid)


def test_a_cell_with_a_transport_that_carries_no_connection_has_no_value():
    """A transport that is not square carries no connection: the cells on
    that edge are recorded with the library's reason, and the other cells
    are read and attached."""
    cells = [(0, 1, 2, 3), (1, 2, 3, 4)]
    pairing, transports = _flat_pairing(cells, 5)
    transports[(0, 1)] = np.ones((1, 2), dtype=complex)
    reads, z, links, spread, groupoid = R.grow(cells, pairing, transports)
    assert "transport must be square" in reads[0]["failed"]
    assert "failed" not in reads[1] and "rejected" not in reads[1]
    assert set(z) == set(links) == {
        tuple(e) for e in itertools.combinations((1, 2, 3, 4), 2)}


# ------------------------------------------------------------------- the cut


def test_the_cut_of_a_subdivided_tetrahedron_is_the_tetrahedrons_own():
    """A tetrahedron and its 1-4 subdivision enclose the same region, so
    they have the same bounding cut with the same outward orientation: the
    four tetrahedra of the subdivision are oriented coherently, two of them
    against their ascending vertex order, and the flux through the cut is
    the one unit the tetrahedron encloses."""
    whole = [[0, 1, 2, 3]]
    parts = [[0, 1, 2, 4], [0, 1, 3, 4], [0, 2, 3, 4], [1, 2, 3, 4]]
    assert R.cell_orientations(whole) == [1]
    assert R.cell_orientations(parts) == [1, -1, 1, -1]

    def cyclic(face):
        low = face.index(min(face))
        return tuple(face[low:] + face[:low])
    assert sorted(cyclic(tuple(f)) for f in R.bounding_cut(parts)) == \
        sorted(cyclic(tuple(f)) for f in R.bounding_cut(whole))
    # a closed surface: every edge of the cut is run once each way
    runs = [(f[i], f[(i + 1) % 3]) for f in R.bounding_cut(parts)
            for i in range(3)]
    assert sorted(runs) == sorted((b, a) for a, b in runs)
    connection = R.monopole_connection(whole)
    links = {e: np.exp(1j * p) for e, p in zip(connection["edges"],
                                               connection["phases"])}
    assert R.cut_monopole_number(R.bounding_cut(whole), links) == 1
    for v in range(4):
        links[(v, 4)] = 1.0 + 0j
    assert R.cut_monopole_number(R.bounding_cut(parts), links) == 1


def test_the_fan_is_oriented_by_its_ascending_orders():
    for count in (2, 3, 4):
        assert R.cell_orientations(R.fan(count)) == [1] * count


def test_a_cluster_with_no_coherent_orientation_has_no_cut():
    with pytest.raises(ValueError, match=r"the face \[0, 1, 2\] is shared "
                                         "by 3 tetrahedra"):
        R.bounding_cut([[0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 2, 5]])


# ----------------------------------------------------- a base a move changed


def test_the_records_of_a_level_whose_base_a_move_changed(monkeypatch):
    """A relaxation whose committed moves leave other cells (here the 1-4
    subdivision of the first tetrahedron, as `relax_level` returns it): the
    level is the moved base; the monopole numbers before are named by the
    cells before, those after by the cells after; the held cut is carried by
    its vertices, every one of its faces a boundary face of the moved base,
    and its monopole number is the one held."""
    config, cells, z, links = _host()
    moved_cells = [[0, 1, 2, 5], [0, 1, 3, 4], [0, 1, 3, 5], [0, 2, 3, 5],
                   [1, 2, 3, 5]]
    moved_z, moved_links = dict(z), dict(links)
    for v in range(4):
        moved_z[(v, 5)] = 3.0 + 0j
        moved_links[(v, 5)] = 1.0 + 0j
    _still(monkeypatch, moves_committed=1, changed=True,
           moved_base=(moved_cells, moved_z, moved_links,
                       {v: v for v in range(6)}))
    _reads(monkeypatch)

    def no_turn(operator, config):
        raise ValueError("stop here")
    monkeypatch.setattr(R, "recursion_turn", no_turn)
    record, _ = R.tick(0, cells, z, links, config)
    level = record["level"]
    assert level["cells_before"] == [[0, 1, 2, 3], [0, 1, 3, 4]]
    assert level["cells"] == moved_cells
    assert (level["vertices"], level["edges"], level["tetrahedra"]) == \
        (6, 13, 5)
    assert level["bulk_monopole_numbers_before"] == [1, 1]
    assert len(level["bulk_monopole_numbers_after"]) == 5
    held = level["held_cut"]
    assert held["faces_after"] == held["faces"]
    assert held["faces_after_on_boundary"] is True
    assert held["monopole_number_before"] == held["monopole_number_after"] \
        == 2
    assert record["relaxation"]["vertex_relabeling"] == {
        str(v): v for v in range(6)}
    assert [read["cell"] for read in record["reads"]] == moved_cells
    assert ("committed moves changed the level's cells: the numbers before "
            "are of the cells [[0, 1, 2, 3], [0, 1, 3, 4]]") in _text(record)
    json.dumps(R._jsonable(record))


# -------------------------------------------------------- the solve's record


def _report(converged):
    """An end-point report with the fields `relaxation_record` reads."""
    nan = float("nan")
    return types.SimpleNamespace(
        converged=converged, stop_detail="the force norm 3e-15 is not at or "
        "below the tolerance", band_selection=bp.BAND_SELECTIONS[
            "continuation"], force_norm=3e-15, purity_defect=0.0,
        spectral_gap=1.0, band_isolation=1.0, band_ranks=[3], bands=[],
        jacobian_size=13, jacobian_rank=10, largest_singular_value=1.0,
        smallest_retained_singular_value=1e-3,
        largest_discarded_singular_value=0.0, rank_gap=nan,
        kontsevich_segal_margin=3.0, largest_length_ratio=1.0, fiber_rank=3,
        moment_targets=[], moment_scale=1.0, multipliers=[],
        moment_residuals=[], fiber_constraint_form=bp.FIBER_PINNINGS[
            "eigenvalues"], hellmann_feynman_force_norm=0.0,
        force_hessian=nan, force_hessian_scale=1.0, action=0j,
        action_available=True, action_unavailable="", occupied_energy=0j)


def _drive(trace):
    counts = {"vertices": 4, "edges": 6, "cells": 1}
    return {"objective": types.SimpleNamespace(updates=[], undefined=[],
                                               direction_order=1),
            "stop_reason": cs.STOP_STATIONARY,
            "stop_detail": "no scaled step", "accepted_updates": 3,
            "moves_committed": 0, "moves": True, "combinatorial_depth": 1,
            "combinatorial_length": 0, "candidate_moves": 0,
            "complex_before": counts, "complex_after": counts,
            "changed": False, "trace": list(trace), "seconds": 0.5}


def test_converged_has_one_definition():
    """`converged` is the residual norm at the point the drive ended on, at
    or below the step tolerance, for a content as for a level; the end-point
    report's statement about the force and the pinned moments is recorded
    under its own name."""
    assert bp.solve_converged(1e-15) and bp.solve_converged(0.0)
    assert not bp.solve_converged(1.0000001e-15)
    assert not bp.solve_converged(float("nan"))
    assert bp.solve_converged(1e-10, 1e-9)

    record = bp.relaxation_record(_report(False), _drive([2.0, 4e-16]))
    assert record["converged"] is True and record["residual"] == 4e-16
    assert record["self_consistent"] is False
    assert record["self_consistent_detail"].startswith("the force norm 3e-15")
    assert record["stop_reason"] == "converged"
    assert record["stop_detail"] == (
        "the residual norm 4e-16 is at or below the step tolerance 1e-15; "
        "the drive ended: no scaled step")
    assert record["accepted_updates"] == 3 and "iterations" not in record

    stalled = bp.relaxation_record(_report(True), _drive([2.0, 1e-9]))
    assert stalled["converged"] is False and stalled["self_consistent"]
    assert stalled["stop_reason"] == cs.STOP_STATIONARY
    assert bp.relaxation_record(_report(True), _drive([2.0, 1e-9]),
                                step_tolerance=1e-8)["converged"] is True
    # a drive that ended on an error has no trace and no residual
    refused = bp.relaxation_record(_report(True), _drive([]))
    assert refused["converged"] is False and np.isnan(refused["residual"])

    text = bp.relaxation_text(record)
    assert text.startswith(
        "solve converged True (residual norm 4e-16, force norm 3e-15, "
        "self-consistent False) after 3 accepted updates; stopped: "
        "converged (")
    assert bp.solve_state({"content": [0, 0, 3], "relaxation": record}) == {
        "state": "converged", "reason": None, "accepted_updates": 3}


def test_the_level_and_the_content_records_name_the_same_counts():
    """The level's record and the content's record carry the same keys for
    what a drive did: whether it converged, the residual it ended on, the
    relaxation updates it accepted and the moves it committed."""
    config, cells, z, links = _host()
    spacetime, count = R.build_level(cells, z, links)
    level = R.relax_level(spacetime, config, count=count)
    content = bp.relaxation_record(_report(False), _drive([2.0, 4e-16]))
    for key in ("converged", "residual", "stop_reason", "stop_detail",
                "accepted_updates", "moves_committed", "residual_trace"):
        assert key in level and key in content
    assert "iterations" not in level
    assert level["converged"] == bp.solve_converged(
        level["residual"], bp.declared_tolerance(config, "step_tolerance"))


# --------------------------------------------- a content without a value


def _solved(monkeypatch):
    """Replace a content's solve by a base tetrahedron and a record."""
    solve = {"converged": False, "force_norm": 1.0, "accepted_updates": 7}
    report = types.SimpleNamespace(kontsevich_segal_margin=3.0, bands=[])
    drive = {"spacetime": bp.build_base(8.0), "moves_committed": 0}
    monkeypatch.setattr(bp, "relax_content",
                        lambda content, kappa, beta, config:
                        (bp.build_host(), None, report, drive))
    monkeypatch.setattr(bp, "relaxation_record",
                        lambda report, drive, *tolerances: dict(solve))
    return solve


@pytest.mark.parametrize("error", [
    bp.NoSpinorDoublet("the support carries no j = 1/2 doublet"),
    RuntimeError("the aligned frame's trialities are not the three Z_3 "
                 "characters"),
    ValueError("PencilSchur: the block is singular"),
    ZeroDivisionError("division by zero")])
def test_a_read_the_poles_are_built_on_without_a_value_keeps_the_solve(
        monkeypatch, error):
    """A read of the solved cell that the poles are built on and that has no
    value leaves the content without a pole, by name, with the solve's
    record kept; the scan goes on to the next content."""
    solve = _solved(monkeypatch)

    def reads(content, kappa, beta, config, started, spacetime, action,
              report, record, flags, stage):
        if tuple(content) == (0, 3, 0):
            stage[0] = "the spin frame"
            raise error
        return {"content": list(content), "doublet_reads": [], "flags": []}
    monkeypatch.setattr(bp, "_content_reads", reads)
    config = bp.default_config([1.0], [1.0],
                               selected_contents=[(0, 3, 0), (1, 1, 1)])
    with pytest.raises(bp.ReadWithoutValue) as caught:
        bp.evaluate_content((0, 3, 0), 1.0, 1.0, config)
    assert caught.value.name == bp.NO_VALUE_READ
    assert caught.value.relaxation == solve
    message = "%s: the spin frame (%s)" % (bp.NO_VALUE_READ, error)
    assert str(caught.value) == message
    point = bp.scan_point(1.0, 1.0, config)
    assert point["failed_contents"] == [[0, 3, 0]]
    failed, read = point["contents"]
    assert failed["failed"] == message
    assert failed["reason"] == bp.NO_VALUE_READ
    assert failed["relaxation"] == solve and failed["doublet_reads"] == []
    assert read["content"] == [1, 1, 1] and "failed" not in read
    assert bp.solve_state(failed) == {
        "state": "no value", "reason": "read without a value",
        "accepted_updates": 7}
    assert bp.point_lines(point)[0].startswith(
        "kappa=1 beta=1: 2 contents, 1 without a value, 0 flagged; ")


@pytest.mark.parametrize("error", [ValueError, RuntimeError,
                                   ZeroDivisionError, OverflowError])
def test_a_solve_without_a_value_is_recorded_and_the_scan_continues(
        monkeypatch, error):
    """Every way the library says a content's solve has no value is recorded
    with its message, and the other contents of the point are read."""
    def evaluate(content, kappa, beta, config):
        if tuple(content) == (0, 3, 0):
            raise error("the decomposition did not converge")
        return {"content": list(content), "doublet_reads": [], "flags": []}
    monkeypatch.setattr(bp, "evaluate_content", evaluate)
    config = bp.default_config([1.0], [1.0],
                               selected_contents=[(0, 3, 0), (1, 1, 1)])
    point = bp.scan_point(1.0, 1.0, config)
    assert point["failed_contents"] == [[0, 3, 0]]
    assert point["contents"][0]["failed"] == \
        "the decomposition did not converge"
    assert point["contents"][1]["content"] == [1, 1, 1]


def test_a_defect_in_a_content_is_raised(monkeypatch):
    """An error that is not one of `NO_VALUE_ERRORS` is a defect, and the
    scan raises it."""
    def evaluate(content, kappa, beta, config):
        raise KeyError("a missing key")
    monkeypatch.setattr(bp, "evaluate_content", evaluate)
    config = bp.default_config([1.0], [1.0], selected_contents=[(1, 1, 1)])
    with pytest.raises(KeyError):
        bp.scan_point(1.0, 1.0, config)


def test_the_base_a_drive_ended_on_decides_whether_the_poles_are_read():
    """`geometry_without_value` reads the base the drive ended on: a
    tetrahedron is read whatever its vertices are called and however many
    moves were committed, and any other complex has no pole read."""
    host = bp.build_host()
    tetrahedron = T.Spacetime.fromVertexTuples(3, [[2, 5, 7, 9]], 1.0, 0.0)
    assert bp.geometry_without_value(
        host, {"spacetime": tetrahedron, "moves_committed": 2}) is None
    two = T.Spacetime.fromVertexTuples(3, [[0, 1, 2, 3], [0, 1, 2, 4]], 1.0,
                                       0.0)
    name, message = bp.geometry_without_value(
        host, {"spacetime": two, "moves_committed": 0})
    assert name == bp.NO_VALUE_MOVED
    assert "a base complex of 5 vertices, 9 edges and 2 cells" in message


# ------------------------------------------- reads made and marked as flags


def test_a_band_that_holds_more_than_its_rank_is_flagged():
    """A band filled above its rank is read as declared and flagged with its
    rank and its occupation; a library whose bands carry no such mark, and a
    band that is not overfilled, give no flag."""
    def band(**marks):
        return types.SimpleNamespace(
            declared_index=2, occupation=3.0, rank=1, eigenvalues=[1 + 0j],
            positions=[6], declared_positions=[6], overlap=1 + 0j,
            crossed=False, ambiguous=False, **marks)
    over = types.SimpleNamespace(bands=[band(overfilled=True),
                                        band(overfilled=False)])
    (flag,) = bp.band_flags(over)
    assert flag["name"] == "a band holds more particles than its rank"
    assert flag["detail"] == ("the declared band 2 has rank 1 and holds 3 "
                              "particles, so its filling 3 per mode is above "
                              "one")
    assert (flag["declared_index"], flag["rank"], flag["occupation"]) == \
        (2, 1, 3.0)
    assert bp.band_flags(types.SimpleNamespace(bands=[band()])) == []
    assert bp._band_record(band(overfilled=True))["overfilled"] is True
    assert bp._band_record(band())["overfilled"] is False


def test_the_eigenbasis_marks_are_those_the_library_reports(monkeypatch):
    """The marks of the carrier operator's eigenbasis are the library's: a
    read that reports them gives both, and a read that reports none gives
    None. On a normal operator the read is not defective."""
    operator = list(np.diag([1.0, 2.0, 3.0]).astype(complex).reshape(-1))
    marks = bp.eigenbasis_marks(operator)
    assert marks is None or (marks["defective"] is False
                             and 0.0 < marks["reciprocal_condition"] <= 1.0)

    def follower(read):
        return lambda declaration: types.SimpleNamespace(
            read=lambda flat: read)
    monkeypatch.setattr(bp.cob, "BandFollower", follower(
        types.SimpleNamespace(defective=True,
                              eigenbasis_reciprocal_condition=3e-17)))
    assert bp.eigenbasis_marks(operator) == {
        "defective": True, "reciprocal_condition": 3e-17}
    monkeypatch.setattr(bp.cob, "BandFollower",
                        follower(types.SimpleNamespace()))
    assert bp.eigenbasis_marks(operator) is None


def test_the_covariance_certificates_of_the_levels_operator_are_recorded(
        monkeypatch):
    """The level's operator is read whether or not its covariance
    certificate holds. On the declared host both certificates hold and name
    nothing; a certificate that does not hold is recorded with the
    properties the library names, counted in the summary and said in the
    tick's notices."""
    def instance(holds, **fields):
        return types.SimpleNamespace(certificate=lambda: types.SimpleNamespace(
            holds=holds, tolerance=2e-13, **fields))
    assert R.proposition_three(instance(True)) == {
        "holds": True, "tolerance": 2e-13, "failed": []}
    failed = ["Proposition 3 (ii) covariance of the pencil: residual 4e-9 "
              "exceeds the tolerance 2e-13"]
    assert R.proposition_three(instance(False, failed=failed)) == {
        "holds": False, "tolerance": 2e-13, "failed": failed}

    config, cells, z, links = _host()
    base = R.base_operator(cells, z, links)
    assert set(base["certificates"]) == {"connection", "inverse_connection"}
    for certificate in base["certificates"].values():
        assert certificate["holds"] is True and certificate["failed"] == []

    _still(monkeypatch)
    _reads(monkeypatch)
    whole = R.base_operator

    def marked(cells, z, links):
        out = whole(cells, z, links)
        out["certificates"]["connection"] = {
            "holds": False, "tolerance": 2e-13, "failed": failed}
        return out
    monkeypatch.setattr(R, "base_operator", marked)
    record, _ = R.tick(0, cells, z, links, config)
    certificates = record["level"]["operator_certificates"]
    assert certificates["connection"]["failed"] == failed
    assert certificates["inverse_connection"]["holds"] is True
    assert record["summary"]["operator_certificates_hold"] is False
    assert "partition" in record["partition"]
    assert ("the covariance certificate of the level's operator "
            "(connection) does not hold at the tolerance 2e-13, and the "
            "level is read on it as built: " + failed[0]) in \
        R._notices(record)
    json.dumps(R._jsonable(record))
