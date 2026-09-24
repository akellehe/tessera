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
    config = R.default_config(tetrahedra=3)
    cells, z, links, _ = R.level_zero(config)
    record, following = R.tick(0, cells, z, links, config)
    assert record["relaxation"]["converged"]
    assert record["level"]["declared_monopole_numbers"] == [1, 1, 1]
    assert len(record["level"]["monopole_numbers"]) == 3
    assert record["level"]["sheet_isomorphism_residual"] < 1e-10
    partition = record["partition"]
    covered = sorted(i for part in partition["partition"] for i in part)
    assert covered == list(range(record["level"]["edges"]))
    assert all(partition["bands_accepted"])
    assert partition["determinant_residual"] < 1e-8 or \
        math.isnan(partition["determinant_residual"])
    kept = [c for c in record["grown_cells"] if "failed" not in c]
    assert record["summary"]["grown_cells"] == len(kept)
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
