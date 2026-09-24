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
    """A level-0 base relaxed with its bounding cut held, and its
    operators."""
    config = R.default_config(tetrahedra=count)
    cells, z, links, _ = R.level_zero(config)
    spacetime, vertices = R.build_level(cells, z, links)
    cut = R.bounding_cut(cells)
    R.relax_level(spacetime, config, R.cut_sectors(
        cut, R.cut_monopole_number(cut, links), vertices))
    base_z, base_links = R.sheet_fields(spacetime, vertices)[0]
    return config, cells, base_z, base_links


def _grown_lengths(stage):
    return {tuple(c["vertices"]): np.array(c["squared_lengths"])
            for c in stage["reads"]
            if "failed" not in c and "rejected" not in c}


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
    images, dual_images = stage["images"], stage["dual_images"]
    changes = [rng.normal(size=(f.shape[1],) * 2)
               + 1j * rng.normal(size=(f.shape[1],) * 2) for f in images]
    # arbitrary changes of the frame and of the dual, which the image-pairing
    # normalization undoes on the determinant line
    moved_images = [f @ x for f, x in zip(images, changes)]
    moved_dual_images = [R.normalize_image_pairing(
        f, d @ (2.0 * np.eye(d.shape[1])))[0]
        for f, d in zip(moved_images, dual_images)]
    moved_frames = [base["metric"] @ f for f in moved_images]
    moved_duals = [base["dual_metric"] @ d for d in moved_dual_images]
    moved_transports = {
        (v, w): np.linalg.solve(changes[v], block) @ changes[w]
        for (v, w), block in stage["transports"].items()}
    moved = R.inherited_pairing(moved_frames, moved_duals, moved_transports,
                                base["covariant"], images=moved_images)
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


def test_the_bounding_cut_leaves_out_the_shared_face():
    """The host's bounding cut is every face no two host tetrahedra share,
    oriented outward; the shared face (0, 1, 3) is bulk."""
    cells = R.fan(2)
    cut = R.bounding_cut(cells)
    assert len(cut) == 6
    assert (0, 1, 3) not in {tuple(sorted(f)) for f in cut}
    assert all(f in R.outward_faces(c) for c in cells for f in cut
               if set(f) <= set(c))
    # the flux through the cut is the sum of the tetrahedra's flux
    _, _, links, _ = R.level_zero(R.default_config(tetrahedra=2))
    assert R.cut_monopole_number(cut, links) == 2


def test_the_held_cut_keeps_its_monopole_number_through_relaxation():
    """On the fan of three the relaxation with the bounding cut held stops at
    the sector boundary unconverged, but the flux through the cut and the
    moduli on it do not move; without the hold the same relaxation converges
    in another sector."""
    config = R.default_config(tetrahedra=3)
    cells, z, links, _ = R.level_zero(config)
    spacetime, vertices = R.build_level(cells, z, links)
    cut = R.bounding_cut(cells)
    declared = R.cut_monopole_number(cut, links)
    assert declared == 3
    report = R.relax_level(spacetime, config, R.cut_sectors(
        cut, declared, vertices))
    after = R.sheet_fields(spacetime, vertices)
    for t in range(R.SHEETS):
        assert R.cut_monopole_number(cut, after[t][1]) == declared
    assert report["sector_monopole_numbers"] == [declared] * R.SHEETS
    assert report["held_modulus_drift"] < 1e-12
    for face in cut:
        assert abs(abs(R.face_holonomy(after[0][1], face)) - 1.0) < 1e-12
    # without the hold the same relaxation leaves the sector
    free, _ = R.build_level(cells, z, links)
    R.relax_level(free, config)
    assert R.cut_monopole_number(cut, R.sheet_fields(free, vertices)[0][1]) \
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


def test_one_tick_on_the_host_holds_its_cut_and_rejects_transient_components(
        monkeypatch, capsys):
    """A real tick (relaxation, box, interaction stage, grown-cell rule) with
    the per-cell baryon reads left out for time. Every certificate the tick
    reports is present. Only the host's bounding cut is held, and the Regge
    term is reported as zero by structure: the fan of two has no interior
    hinge. Three of the five components persist over one of the five declared
    resolutions and are rejected by name. That leaves two response vertices,
    no grown 3-simplex, and a stop."""
    monkeypatch.setattr(R, "cell_reads", lambda cells, z, links, config: [])
    config = R.default_config(tetrahedra=2)
    cells, z, links, _ = R.level_zero(config)
    record, following = R.tick(0, cells, z, links, config)
    assert record["tick"] == 0
    assert record["relaxation"]["converged"]
    held = record["level"]["held_cut"]
    assert sorted(tuple(sorted(f)) for f in held["faces"]) == sorted(
        tuple(sorted(f)) for f in R.bounding_cut(cells))
    assert held["monopole_number_before"] == held["monopole_number_after"] \
        == 2
    assert held["sector_monopole_numbers_after"] == [2] * R.SHEETS
    assert record["level"]["bulk_monopole_numbers_before"] == [1, 1]
    assert record["level"]["bulk_monopole_numbers_after"] == [1, 1]
    assert record["relaxation"]["held_modulus_drift"] < 1e-12
    assert record["relaxation"]["regge_hinge_count"] == 0
    assert record["relaxation"]["regge_structurally_zero"]
    assert record["level"]["sheet_isomorphism_residual"] < 1e-10
    partition = record["partition"]
    covered = sorted(i for part in partition["partition"] for i in part)
    assert covered == list(range(record["level"]["edges"]))
    assert partition["determinant_residual"] < 1e-8 or \
        math.isnan(partition["determinant_residual"])
    rejected = record["rejected_components"]
    assert [r["persistence"] for r in rejected] == [1.0, 1.0, 1.0]
    assert all(r["required"] == len(R.DECLARED_RESOLUTIONS)
               for r in rejected)
    assert sorted(e for r in rejected for e in r["edges"]) == \
        ["0-4", "1-4", "3-4"]
    assert record["summary"]["response_vertices"] == 2
    assert max(record["fibers"]["pairing_defect"]) < 1e-12
    assert record["summary"]["grown_cells"] == 0
    assert following is None and "stopped" in record
    json.dumps(R._jsonable(record))
    notices = R._notices(record)
    assert notices[0].startswith("the Regge term is structurally zero")
    assert "component 2 (edges 0-4) rejected" in notices[1]


def test_a_grown_level_holds_nothing(monkeypatch):
    """Monopole numbers of a grown level are bulk data: the relaxation of any
    tick after the first is given no held sector, and its cells are read
    without one."""
    seen = []

    def refuse(spacetime, config, sectors=None, count=None):
        seen.append(list(sectors or []))
        raise ValueError("stop here")
    monkeypatch.setattr(R, "relax_level", refuse)
    config = R.default_config(tetrahedra=2)
    cells, z, links, _ = R.level_zero(config)
    R.tick(0, cells, z, links, config)
    record, _ = R.tick(1, cells, z, links, config)
    assert len(seen[0]) == R.SHEETS and seen[1] == []
    assert record["level"]["held_cut"]["faces"] == []
    assert record["level"]["bulk_monopole_numbers_before"] == [1, 1]


def test_the_manifold_gate_rejects_a_cell_that_makes_a_non_manifold():
    """Three tetrahedra on one triangle: the third makes the triangle a face
    of three top cells, which no manifold with boundary has. It is recorded
    with the violation and contributes no edge."""
    cells = [(0, 1, 2, 3), (0, 1, 2, 4), (0, 1, 2, 5)]
    rng = np.random.default_rng(8)
    points = rng.normal(size=(6, 3))
    pairing = np.zeros((6, 6), dtype=complex)
    transports = {}
    for cell in cells:
        s_local = [complex(np.sum((points[a] - points[b]) ** 2))
                   for a, b in itertools.combinations(cell, 2)]
        from tessera import cobordism as cob
        K = cob.ChainComplex.fromTopCells([[0, 1, 2, 3]])
        block = np.asarray(ch.WhitneyMass.topSimplexBlocks(K, s_local, 1)[0]
                           .block)
        Z = np.zeros((6, 4))
        for a, (x, y) in enumerate(itertools.combinations(range(4), 2)):
            Z[a, x], Z[a, y] = -1.0, 1.0
        Y = block @ Z
        local = np.asarray(ch.GrownCellRule.determinantPairing(
            [Y[:, [v]] for v in range(4)],
            [np.linalg.solve(block, Y[:, [v]]) for v in range(4)]))
        for i, v in enumerate(cell):
            for j, w in enumerate(cell):
                pairing[v, w] = local[i, j]
    for v, w in itertools.permutations(range(6), 2):
        transports[(v, w)] = np.array([[1.0 + 0j]])
    assert R.manifold_violation(cells[:2]) is None
    assert R.manifold_violation(cells) is not None
    reads, z, links, spread, groupoid = R.grow(cells, pairing, transports)
    assert ["rejected" in r for r in reads] == [False, False, True]
    assert reads[2]["rejected"] == R.manifold_violation(cells)
    assert (0, 5) not in z and (0, 3) in z and (0, 4) in z


def test_image_supported_fibers_pair_exactly_to_zero_off_shared_simplices():
    """Spec Prop. 5.3: with every fiber supported on its geometric image, the
    pairing between two fibers whose image supports share no top simplex is
    exactly zero. The fan of four at level zero, one component per edge, has
    54 such pairs. The transports between them are not zero: the transfer
    from the pencil block contains (M_0^U)^{-1}, the elimination of the
    vertex degree, which couples every pair."""
    config = R.default_config(tetrahedra=4)
    cells, z, links, _ = R.level_zero(config)
    base = R.base_operator(cells, z, links)
    partition = [[i] for i in range(len(base["edges"]))]
    stage = R.interaction_stage(base, partition, config)
    tops = [set(t) for t in base["tops"]]
    apart = [(v, w) for v, w in itertools.permutations(range(len(partition)), 2)
             if not any(set(base["edges"][v]) <= t
                        and set(base["edges"][w]) <= t for t in tops)]
    assert len(apart) == 2 * 54
    assert stage["locality"]["pairs_without_a_shared_top_simplex"] == 54
    for v, w in apart:
        assert stage["pairing"][v, w] == 0.0
    assert stage["locality"]["largest_pairing_between_them"] == 0.0
    for v, (image, frame, part) in enumerate(zip(
            stage["images"], stage["frames"], partition)):
        outside = [i for i in range(image.shape[0]) if i not in part]
        assert np.all(image[outside] == 0)
        ring = {i for i, e in enumerate(base["edges"])
                if any(set(e) <= t and set(base["edges"][part[0]]) <= t
                       for t in tops)}
        assert np.all(frame[[i for i in range(frame.shape[0])
                             if i not in ring]] == 0)
    for v, w in itertools.combinations(range(len(partition)), 2):
        if (v, w) not in stage["pairs"]:
            assert stage["pairing"][v, w] == 0.0
    assert stage["locality"]["largest_transport_between_them"] > 1.0


def test_the_image_pairing_normalization_returns_level_zero():
    """The level-0 vertex fibers: Z_v = delta^U e_v on one tetrahedron, whose
    two images pair to 3, the number of edges at a vertex. Rescaled frames,
    normalized to det((Z^vee)^T Z) = 3, give back the tetrahedron's squared
    lengths exactly (1e-12) under the trivial connection and a complex pure
    gauge, and the same grown lengths under a complex gauge transformation of
    a curved connection."""
    from tessera import cobordism as cob
    rng = np.random.default_rng(12)
    pairs = list(itertools.combinations(range(4), 2))
    s = [8.0, 6.5, 9.0, 7.2, 8.8, 5.9]
    K = cob.ChainComplex.fromTopCells([[0, 1, 2, 3]])
    block = np.asarray(ch.WhitneyMass.topSimplexBlocks(K, s, 1)[0].block)

    def grown(U, scales):
        full = dict(U)
        full.update({(j, i): 1.0 / u for (i, j), u in U.items()})
        full.update({(v, v): 1.0 for v in range(4)})

        def dressed(link):
            return np.array([[block[a, b] * link[(pairs[a][0], pairs[b][0])]
                              for b in range(6)] for a in range(6)])

        def coboundary(link):
            out = np.zeros((6, 4), dtype=complex)
            for a, (x, y) in enumerate(pairs):
                out[a, x], out[a, y] = -1.0, link[(x, y)]
            return out
        inverse = {k: 1.0 / u for k, u in full.items()}
        images = coboundary(full) * scales[0]
        dual_images = coboundary(inverse) * scales[1]
        for v in range(4):
            assert (coboundary(inverse)[:, v] @ coboundary(full)[:, v]) == \
                pytest.approx(R.IMAGE_PAIRING_UNIT, abs=1e-14)
        dual_images = np.hstack([R.normalize_image_pairing(
            images[:, [v]], dual_images[:, [v]])[0] for v in range(4)])
        duals = dressed(inverse) @ dual_images
        # the level-0 transport along an edge is its link, read in the
        # rescaled frames
        connection = np.array([[full[(v, w)] * scales[0][w] / scales[0][v]
                                for w in range(4)] for v in range(4)])
        pairing = np.asarray(ch.GrownCellRule.gaugeInvariantPairing(
            [duals[:, [v]] for v in range(4)],
            [images[:, [v]] for v in range(4)], connection))
        return np.array(ch.GrownCellRule.invertVertexPairing(pairing)
                        .squaredLengths)

    scales = (np.exp(rng.normal(size=4) + 1j * rng.normal(size=4)),
              np.exp(rng.normal(size=4) + 1j * rng.normal(size=4)))
    flat = {e: 1.0 + 0j for e in pairs}
    g = np.exp(rng.normal(size=4) + 1j * rng.normal(size=4))
    pure = {(i, j): g[i] / g[j] for i, j in pairs}
    for U in (flat, pure):
        np.testing.assert_allclose(grown(U, scales), s, rtol=1e-12)
    curved = {e: np.exp(1j * rng.normal()) for e in pairs}
    gauged = {(i, j): g[i] * curved[(i, j)] / g[j] for i, j in pairs}
    np.testing.assert_allclose(grown(gauged, scales), grown(curved, scales),
                               rtol=1e-12)


def test_components_that_do_not_persist_across_the_range_are_rejected():
    class Level:
        partition = [[0, 1], [2], [3, 4]]
        component_persistence = [5.0, 1.0, 4.0]
    edges = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)]
    accepted, rejected = R.persistent_components(Level, edges, 5)
    assert accepted == [[0, 1]]
    assert rejected == [
        {"component": 1, "edges": ["1-2"], "persistence": 1.0,
         "required": 5},
        {"component": 2, "edges": ["1-3", "2-3"], "persistence": 4.0,
         "required": 5}]
    accepted, _ = R.persistent_components(Level, edges, 2)
    assert accepted == [[0, 1], [3, 4]]


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
    def refuse(spacetime, config, sectors=None, count=None):
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
