# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The level-recursion driver (`tessera.drivers.recursion`): the declared host
carries a unit monopole on every tetrahedron, the interaction stage grows the
flag complex, the grown-cell rule returns level zero, one real tick on the host
produces a next level, and the --live path runs the same drive, survives the
window being closed and writes every tick as it completes.

The live path is exercised with matplotlib stubbed and the tick replaced by a
deterministic stand-in, as the baryon driver's live tests do."""
import itertools
import json
import math
import time

import numpy as np
import pytest

from tessera import chainhodge as ch
from tessera.drivers import recursion as R


@pytest.mark.parametrize("count", [2, 3, 4])
def test_every_tetrahedron_of_the_fan_carries_a_unit_monopole(count):
    config = R.default_config(tetrahedra=count)
    cells, z, links, connection = R.level_zero(config)
    assert connection["residual"] < 1e-12
    assert np.max(np.abs(connection["face_angles"])) < math.pi
    assert R.monopole_numbers(cells, links) == [1] * count


def test_the_level_carries_its_fields_on_every_sheet():
    config = R.default_config(tetrahedra=2)
    cells, z, links, _ = R.level_zero(config)
    spacetime, count = R.build_level(cells, z, links)
    fields = R.sheet_fields(spacetime, count)
    assert len(fields) == R.SHEETS
    for squared, stored in fields:
        for e in z:
            assert abs(squared[e] - z[e]) < 1e-12
            assert abs(stored[e] - links[e]) < 1e-12


def test_the_interaction_graph_and_its_flag_complex():
    # one tetrahedron whose six edges are six response vertices: every pair
    # shares the top simplex, so the flag complex has all 15 quadruples
    edges = list(itertools.combinations(range(4), 2))
    tops = [(0, 1, 2, 3)]
    pairs = R.interaction_graph([[i] for i in range(6)], edges, tops)
    assert len(pairs) == 15
    assert len(R.grown_cells(6, pairs)) == 15
    # a path of four vertices has no 3-simplex
    assert R.grown_cells(4, {(0, 1), (1, 2), (2, 3)}) == []
    # a complete graph on four vertices has exactly one
    complete = set(itertools.combinations(range(4), 2))
    assert R.grown_cells(4, complete) == [(0, 1, 2, 3)]


def _relaxed_base(count=2):
    """A level-0 base relaxed with its monopole sectors held, and its
    operators."""
    config = R.default_config(tetrahedra=count)
    cells, z, links, _ = R.level_zero(config)
    spacetime, vertices = R.build_level(cells, z, links)
    R.relax_level(spacetime, config, R.held_sectors(
        cells, R.monopole_numbers(cells, links), vertices))
    base_z, base_links = R.sheet_fields(spacetime, vertices)[0]
    return config, cells, base_z, base_links


def _grown_lengths(stage):
    return {tuple(c["vertices"]): np.array(c["squared_lengths"])
            for c in stage["reads"] if "failed" not in c}


@pytest.mark.parametrize("unitary", [True, False])
def test_grown_lengths_are_invariant_under_pure_gauge(unitary):
    config, cells, z, links = _relaxed_base()
    base = R.base_operator(cells, z, links)
    partition = [list(p) for p in R.recursion_turn(base["operator"],
                                                   config).partition]
    reference = R.interaction_stage(base, partition, config)
    assert _grown_lengths(reference)
    rng = np.random.default_rng(3 if unitary else 4)
    vertices = 1 + max(max(c) for c in cells)
    theta = rng.normal(size=vertices) + (
        0.0 if unitary else 0.3j * rng.normal(size=vertices))
    g = np.exp(1j * theta)
    gauged = {e: u * g[e[1]] / g[e[0]] for e, u in links.items()}
    moved = R.interaction_stage(R.base_operator(cells, z, gauged), partition,
                                config)
    for key, value in _grown_lengths(reference).items():
        np.testing.assert_allclose(_grown_lengths(moved)[key], value,
                                   rtol=1e-8)


def test_grown_lengths_are_invariant_under_frame_changes():
    config, cells, z, links = _relaxed_base()
    base = R.base_operator(cells, z, links)
    partition = [list(p) for p in R.recursion_turn(base["operator"],
                                                   config).partition]
    stage = R.interaction_stage(base, partition, config)
    rng = np.random.default_rng(5)
    frames, duals = stage["frames"], stage["duals"]
    changes = [rng.normal(size=(f.shape[1],) * 2)
               + 1j * rng.normal(size=(f.shape[1],) * 2) for f in frames]
    # det-1 changes of the frame and arbitrary changes of the dual, which the
    # normalization undoes
    changes = [x / np.linalg.det(x) ** (1.0 / x.shape[0]) for x in changes]
    moved_frames = [f @ x for f, x in zip(frames, changes)]
    moved_duals = [np.asarray(ch.GrownCellRule.normalizeDualFrame(
        f, d @ (2.0 * np.eye(d.shape[1])))) for f, d in zip(moved_frames, duals)]
    moved_transports = {
        (v, w): np.linalg.solve(changes[v], block) @ changes[w]
        for (v, w), block in stage["transports"].items()}
    moved = R.inherited_pairing(moved_frames, moved_duals, moved_transports,
                                base["covariant"])
    np.testing.assert_allclose(moved, stage["pairing"], rtol=1e-8,
                               atol=1e-12 * np.abs(stage["pairing"]).max())


def test_the_rule_shift_on_the_monopole_host_and_at_pure_gauge():
    config = R.default_config(tetrahedra=3)
    cells, z, links, _ = R.level_zero(config)
    curved = R.level_rule_shift(cells, z, links)
    assert all(r["relative_shift"] > 1e-3 for r in curved)
    flat = {e: 1.0 + 0.0j for e in links}
    rng = np.random.default_rng(6)
    g = np.exp(1j * (rng.normal(size=7) + 0.3j * rng.normal(size=7)))
    pure = {e: u * g[e[1]] / g[e[0]] for e, u in flat.items()}
    assert all(r["relative_shift"] < 1e-10 for r in R.level_rule_shift(
        cells, z, pure))


def test_held_sectors_keep_the_monopole_numbers_through_relaxation():
    """On the fan of three the held relaxation stops at the sector boundary
    unconverged, but it never leaves the sector; without the hold the same
    relaxation converges in another sector."""
    config = R.default_config(tetrahedra=3)
    cells, z, links, _ = R.level_zero(config)
    spacetime, vertices = R.build_level(cells, z, links)
    declared = R.monopole_numbers(cells, links)
    report = R.relax_level(spacetime, config, R.held_sectors(
        cells, declared, vertices))
    after = R.sheet_fields(spacetime, vertices)
    for t in range(R.SHEETS):
        assert R.monopole_numbers(cells, after[t][1]) == declared
    assert report["sector_monopole_numbers"] == declared * R.SHEETS
    assert report["held_modulus_drift"] < 1e-12
    # without the hold the same relaxation leaves the sector
    free, _ = R.build_level(cells, z, links)
    R.relax_level(free, config)
    assert R.monopole_numbers(cells, R.sheet_fields(free, vertices)[0][1]) \
        != declared


def test_the_grown_cell_rule_returns_level_zero():
    """Level zero as a level: the response vertices are the vertices of one
    tetrahedron, each fiber the exact chain of its vertex, the transports the
    edge connection. The grown cell returns the squared lengths, and the grown
    edges carry the connection."""
    rng = np.random.default_rng(4)
    points = rng.normal(size=(4, 3))
    pairs = list(itertools.combinations(range(4), 2))
    s = [complex(np.sum((points[j] - points[i]) ** 2)) for i, j in pairs]
    from tessera import cobordism as cob
    K = cob.ChainComplex.fromTopCells([[0, 1, 2, 3]])
    block = np.asarray(ch.WhitneyMass.topSimplexBlocks(K, s, 1)[0].block)
    Z = np.zeros((6, 4))
    for a, (x, y) in enumerate(pairs):
        Z[a, x], Z[a, y] = -1.0, 1.0
    Y = block @ Z
    frames = [Y[:, [v]] for v in range(4)]
    images = [np.linalg.solve(block, f) for f in frames]
    links = np.ones(6, dtype=complex)
    pairing = np.asarray(ch.GrownCellRule.determinantPairing(frames, images))
    links = np.exp(1j * rng.normal(size=6))
    transports = {}
    for m, (v, w) in enumerate(pairs):
        transports[(v, w)] = np.array([[links[m]]])
        transports[(w, v)] = np.array([[1.0 / links[m]]])
    reads, z, grown_links, spread, groupoid = R.grow([(0, 1, 2, 3)], pairing,
                                                     transports)
    assert "failed" not in reads[0]
    assert reads[0]["row_sum_defect"] < 1e-13
    for m, e in enumerate(pairs):
        assert abs(z[e] - s[m]) < 1e-11 * abs(s[m])
        assert abs(grown_links[e] - links[m]) < 1e-14
        assert spread[e] == 0.0
        assert groupoid[e] < 1e-14


def test_one_tick_on_the_host_grows_the_next_level(monkeypatch):
    """A real tick (relaxation, box, interaction stage, grown-cell rule) with
    the per-cell baryon reads left out for time: every certificate the tick
    reports is present, and the next level is a three-dimensional complex on
    the relabelled response vertices."""
    monkeypatch.setattr(R, "cell_reads", lambda cells, z, links, config: [])
    config = R.default_config(tetrahedra=2)
    cells, z, links, _ = R.level_zero(config)
    record, following = R.tick(0, cells, z, links, config)
    assert record["tick"] == 0
    assert record["relaxation"]["converged"]
    assert record["level"]["declared_monopole_numbers"] == [1, 1]
    assert record["level"]["monopole_numbers"] == [1, 1]
    assert record["relaxation"]["held_modulus_drift"] < 1e-12
    assert max(record["fibers"]["projector_agreement_with_library"]) < 1e-8
    assert record["level"]["sheet_isomorphism_residual"] < 1e-10
    partition = record["partition"]
    covered = sorted(i for part in partition["partition"] for i in part)
    assert covered == list(range(record["level"]["edges"]))
    assert all(partition["bands_accepted"])
    assert partition["determinant_residual"] < 1e-8 or \
        math.isnan(partition["determinant_residual"])
    kept = [c for c in record["grown_cells"] if "failed" not in c]
    assert record["summary"]["grown_cells"] == len(kept)
    json.dumps(R._jsonable(record))
    if following is None:
        assert "stopped" in record
        return
    next_cells, next_z, next_links = following
    assert len(next_cells) == len(kept)
    for cell in kept:
        assert len(cell["squared_lengths"]) == 6
        assert cell["row_sum_defect"] >= 0.0
    for cell in next_cells:
        for a, b in itertools.combinations(sorted(cell), 2):
            assert (a, b) in next_z and (a, b) in next_links
    spacetime, count = R.build_level(next_cells, next_z, next_links)
    assert count == 1 + max(max(c) for c in next_cells)


# -------------------------------------------------------------------- live


class _Canvas:
    def __init__(self):
        self.draws = 0
        self.callbacks = {}

    def draw_idle(self):
        self.draws += 1

    def start_event_loop(self, interval):
        time.sleep(0.001)

    def mpl_connect(self, name, callback):
        self.callbacks[name] = callback
        return len(self.callbacks)

    def close(self):
        self.callbacks["close_event"](object())


class _Figure:
    def __init__(self):
        self.canvas = _Canvas()


def _stub_matplotlib(monkeypatch, backend="qtagg", figures=None):
    import matplotlib
    import matplotlib.pyplot as plt

    def figure(**kwargs):
        made = _Figure()
        if figures is not None:
            figures.append(made)
        return made

    monkeypatch.setattr(matplotlib, "get_backend", lambda: backend)
    monkeypatch.setattr(plt, "isinteractive", lambda: True)
    monkeypatch.setattr(plt, "figure", figure)
    monkeypatch.setattr(plt, "show", lambda **kwargs: None)
    monkeypatch.setattr(plt, "close", lambda figure: None)


def _cheap_tick(index, cells, z, links, config):
    """A deterministic stand-in for one tick; the claim under test is that the
    live worker runs the same `drive`."""
    record = {"tick": index, "seconds": 0.0,
              "summary": {"response_vertices": 4 + index, "interactions": 6,
                          "grown_cells": 1, "row_sum_defects": [0.1 * index]}}
    return record, (cells, z, links)


@pytest.mark.parametrize("backend", ["agg", "webagg"])
def test_live_refuses_a_non_interactive_or_webagg_backend(monkeypatch,
                                                          backend):
    _stub_matplotlib(monkeypatch, backend)
    with pytest.raises(RuntimeError, match="--live needs an interactive"):
        R.drive_live(R.default_config(ticks=1))


def test_live_sets_the_window_not_to_raise(monkeypatch):
    import matplotlib
    monkeypatch.setattr(R, "tick", _cheap_tick)
    monkeypatch.setattr(R, "draw_frame", lambda figure, frames, index: None)
    _stub_matplotlib(monkeypatch)
    monkeypatch.setitem(matplotlib.rcParams, "figure.raise_window", True)
    R.drive_live(R.default_config(ticks=1))
    assert matplotlib.rcParams["figure.raise_window"] is False


def test_live_and_headless_outputs_are_identical(monkeypatch):
    monkeypatch.setattr(R, "tick", _cheap_tick)
    config = R.default_config(ticks=3)
    headless = R.drive(dict(config))
    _stub_matplotlib(monkeypatch)
    drawn = []
    monkeypatch.setattr(R, "draw_frame",
                        lambda figure, frames, index: drawn.append(index))
    live = R.drive_live(dict(config))
    assert drawn == [0, 1, 2]
    assert json.dumps(R._jsonable(live), sort_keys=True) == \
        json.dumps(R._jsonable(headless), sort_keys=True)


def test_closing_the_window_switches_the_run_to_headless(monkeypatch,
                                                         tmp_path, capsys):
    monkeypatch.setattr(R, "tick", _cheap_tick)
    config = R.default_config(ticks=3)
    headless = R.drive(dict(config))
    _stub_matplotlib(monkeypatch)
    drawn = []

    def draw_then_close(figure, frames, index):
        drawn.append(index)
        figure.canvas.close()

    monkeypatch.setattr(R, "draw_frame", draw_then_close)
    points = tmp_path / "run.points.jsonl"
    live = R.drive_live(dict(config), points_file=str(points))
    assert drawn == [0]
    assert len(live["ticks"]) == 3 and not live["stopped"]
    assert json.dumps(R._jsonable(live), sort_keys=True) == \
        json.dumps(R._jsonable(headless), sort_keys=True)
    assert "continues headless" in capsys.readouterr().out
    lines = points.read_text().splitlines()
    assert len(lines) == 4
    assert "config" in json.loads(lines[0])
    assert [json.loads(line)["tick"] for line in lines[1:]] == [0, 1, 2]


def test_a_worker_error_reaches_the_main_thread(monkeypatch):
    def exploding(*args, **kwargs):
        raise ValueError("the tick failed")
    monkeypatch.setattr(R, "tick", exploding)
    _stub_matplotlib(monkeypatch)
    with pytest.raises(ValueError, match="the tick failed"):
        R.drive_live(R.default_config(ticks=1))


def test_the_recursion_stops_at_a_level_with_no_grown_cell(monkeypatch):
    def last_tick(index, cells, z, links, config):
        record, _ = _cheap_tick(index, cells, z, links, config)
        record["stopped"] = "no grown 3-simplex"
        return record, None
    monkeypatch.setattr(R, "tick", last_tick)
    result = R.drive(R.default_config(ticks=5))
    assert len(result["ticks"]) == 1


def test_a_refused_relaxation_stops_the_recursion_with_its_reason(monkeypatch):
    def refuse(spacetime, config, sectors=None):
        raise ValueError("VillainCharacter::logarithm: refused")
    monkeypatch.setattr(R, "relax_level", refuse)
    config = R.default_config(tetrahedra=2)
    cells, z, links, _ = R.level_zero(config)
    record, following = R.tick(0, cells, z, links, config)
    assert following is None
    assert "refused" in record["stopped"]
    assert record["relaxation"]["failed"].startswith("VillainCharacter")
    json.dumps(R._jsonable(record))
    assert "refused" in R.summary({"host": {"monopole_numbers": []},
                                   "ticks": [record]})
