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
frame contain, and the closed-form behaviour of the band helpers.
"""
import json

import numpy as np
import pytest

from tessera.drivers import baryon_poles as bp
from tessera.drivers import recursion as R

THREE = str(bp.SPIN_THREE_HALVES)

#: A read of one cell, as `cell_reads` returns it: one content the library
#: refused and one read content with a spin-3/2 pole, a quark verdict and the
#: quartic's truncation certificates.
READ = [{
    "cell": [0, 1, 2, 3],
    "host_cell": {},
    "failed_contents": [[0, 3, 0]],
    "contents": [
        {"content": [0, 3, 0], "failed": "band 1 has rank 2", "sectors": {}},
        {"content": [3, 0, 0],
         "sectors": {THREE: {"quasi_free": {"lowest_pole": 4.0 + 0.5j},
                             "with_quartic": {"lowest_pole": None}}},
         "quark_conditions": {"certified": False, "conditions": [
             {"name": "persistent-cluster", "status": "Failed"}]},
         "isospin_doublet": {"covariant": {"status": "x", "found": False,
                                           "other": 1}},
         "quartic": {"truncation": {"induced_displacement_norm": 0.1,
                                    "unrelated": 2.0}}}],
    "ratios": {}, "pole_table": {}}]


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
    assert args.holonomy == "villain"
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
    (["run", "--holonomy", "plaquette"], "--holonomy"),
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
    assert config["contour_nodes"] == R.DECLARED_CONTOUR_NODES
    assert config["max_cells"] == 1
    assert R.points_path("a/run.json") == "a/run.points.jsonl"


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
    truncation = first["summary"]["lowest_poles"][0][1]["quartic_truncation"]
    assert truncation["induced_displacement_norm"] == 0.1
    assert "unrelated" not in truncation
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
            "--holonomy", "wilson", "--band-rank", "1", "--quiet"])
    (config,) = stub_reads
    assert config["contents"] == [[1, 1, 1]]
    assert config["max_cells"] == 1
    assert config["kappa"] == 0.5 and config["beta"] == 2.0
    assert config["holonomy"] == "wilson"


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
    assert "host cell [0, 1, 2, 3] content [0, 3, 0]: failed: band 1 has " \
        "rank 2" in out
    assert "content [3, 0, 0]: read; quark certified False" in out
    R.main(["run", "--ticks", "1", "--quiet"])
    assert capsys.readouterr().out == ""


def test_main_live_refuses_a_file_backend_by_name(stub_reads, monkeypatch):
    import matplotlib
    monkeypatch.setattr(matplotlib, "get_backend", lambda: "agg")
    with pytest.raises(RuntimeError, match="--live needs an interactive"):
        R.main(["run", "--ticks", "1", "--live", "--quiet"])
    assert stub_reads == []


def test_main_live_runs_the_live_drive(stub_reads, monkeypatch, tmp_path):
    calls = []

    def live(config, progress=False, points_file=None):
        calls.append((config["ticks"], progress, points_file))
        return R.drive(config, points_file=points_file)

    monkeypatch.setattr(R, "drive_live", live)
    R.main(["run", "--ticks", "1", "--live", "--quiet", "--json",
            str(tmp_path / "live.json")])
    assert calls == [(1, False, str(tmp_path / "live.points.jsonl"))]


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
    """The two lowest eigenvalues are enclosed by the circle about their mean
    whose radius is halfway to the nearest excluded one; the trapezoidal
    projector converges geometrically, as (radius / distance)^nodes, which is
    about 5e-12 for the excluded eigenvalue 3 at 64 nodes."""
    block = np.diag([3.0, 1.0, 2.0, 9.0]).astype(complex)
    band = R.riesz_band(block, 2, 64)
    np.testing.assert_allclose(sorted(v.real for v in band["eigenvalues"]),
                               [1.0, 2.0])
    np.testing.assert_allclose(band["projector"],
                               np.diag([0, 1, 1, 0]).astype(complex),
                               atol=1e-10)
    np.testing.assert_allclose(band["left"] @ band["frame"], np.eye(2),
                               atol=1e-12)
    assert band["radius"] == pytest.approx(1.0)
    assert band["centre"] == pytest.approx(1.5)


def test_a_band_of_the_whole_block_has_nothing_excluded():
    band = R.riesz_band(np.diag([1.0, 3.0]).astype(complex), 5, 64)
    assert band["frame"].shape == (2, 2)
    np.testing.assert_allclose(band["projector"], np.eye(2), atol=1e-12)


def test_the_fibers_of_a_partition_are_supported_on_their_images():
    """Each fiber's geometric image is zero off its component, its two images
    pair to det((Z^vee)^T Z) = 3, and its left frame is the dual of its chain
    frame."""
    config = R.default_config(tetrahedra=2)
    cells, z, links, _ = R.level_zero(config)
    base = R.base_operator(cells, z, links)
    partition = [list(p) for p in R.recursion_turn(base["operator"],
                                                   config).partition]
    fibers = R.fibers_for_partition(base, partition, 1, 64)
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
