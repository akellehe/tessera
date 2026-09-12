# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Command-line, configuration, cancellation, and output regressions."""

import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))),
    "examples", "cobordism"))

import emergence_animation as ea  # noqa: E402


def test_combinatorial_names_are_canonical_with_legacy_replay_aliases():
    depth = ea.build_config(combinatorial_depth=3)
    assert depth["combinatorial_depth"] == 3
    assert "surgical_depth" not in depth
    length = ea.build_config(combinatorial_length=4)
    assert length["combinatorial_length"] == 4
    assert "combinatorial_breadth" not in length

    assert ea.build_config(surgical_depth=3)["combinatorial_depth"] == 3
    assert ea.build_config(combinatorial_breadth=4)["combinatorial_length"] == 4
    with pytest.raises(ValueError, match="legacy alias"):
        ea.build_config(combinatorial_depth=2, surgical_depth=3)
    with pytest.raises(ValueError, match="alternative search schedules"):
        ea.build_config(combinatorial_depth=2, combinatorial_length=3)


def test_canonical_and_hidden_legacy_cli_flags_reach_the_same_config(monkeypatch):
    seen = []

    def fake_drive(config, progress=False, on_setup=None, **_kwargs):
        seen.append(config)
        return ea.DriveResult([SimpleNamespace(step=0)], ea.Terminator.STEPS)

    monkeypatch.setattr(ea, "drive", fake_drive)
    for flag, value in (("--combinatorial-depth", "3"),
                        ("--surgical-depth", "3"),
                        ("--combinatorial-length", "4"),
                        ("--combinatorial-breadth", "4")):
        assert ea.main(["run", flag, value, "--out", "", "--quiet"]) == 0
    assert [config["combinatorial_depth"] for config in seen[:2]] == [3, 3]
    assert [config["combinatorial_length"] for config in seen[2:]] == [4, 4]

    help_text = ea.build_parser()._subparsers._group_actions[0].choices[
        "run"].format_help()
    assert "--combinatorial-depth" in help_text
    assert "--combinatorial-length" in help_text
    assert "--surgical-depth" not in help_text
    assert "--combinatorial-breadth" not in help_text


@pytest.mark.parametrize("overrides,match", [
    ({"host_seed": (1 << 64) - 1, "size": 2}, "refinement attempts"),
    ({"seed": 1 << 64}, "unsigned 64-bit"),
    ({"coupling": 1e308, "time": 1e308,
      "inputs": ea.InputMode.QUBIT}, "J times time"),
    ({"resolution": 0.0}, "positive finite"),
    ({"output_state": "inf",
      "inputs": ea.InputMode.QUBIT,
      "readout": "whole"}, "must be finite"),
])
def test_invalid_values_fail_before_the_engine(overrides, match):
    with pytest.raises(ValueError, match=match):
        ea.build_config(**overrides)


def test_extreme_finite_output_state_is_normalized_without_overflow():
    config = ea.build_config(
        inputs=ea.InputMode.QUBIT, readout="whole",
        output_state="1.7e308+1.7e308j", steps=0, regge=False)
    node, _inputs = ea.build_qubit_node(config)
    target = np.asarray(node.output_state_target)
    assert np.all(np.isfinite(target))
    assert np.linalg.norm(target) == pytest.approx(1.0)
    assert target[1] == pytest.approx((1.0 + 1.0j) / np.sqrt(2.0))


@pytest.mark.parametrize("name", [
    "stage1_iters", "stage2_iters", "candidate_moves",
    "combinatorial_depth", "combinatorial_length", "grid", "layers",
])
def test_engine_integer_arguments_are_bounded_before_pybind(name):
    with pytest.raises(ValueError, match="engine's integer API"):
        ea.build_config(**{name: 1 << 31})


def test_cooperative_stop_has_an_unambiguous_terminator():
    result = ea.drive(ea.build_config(size=0, steps=3), progress=False,
                      stop_requested=lambda: True)
    assert result.terminator == ea.Terminator.CANCELLED
    assert [frame.step for frame in result.frames] == [0]


def test_output_preflight_rejects_suffix_aliases_and_hard_links(tmp_path):
    with pytest.raises(ValueError, match="must end"):
        ea._validate_output_paths(tmp_path / "frame.jpg", None, None)
    path = tmp_path / "one.json"
    alias = tmp_path / "two.json"
    path.write_text("one", encoding="utf-8")
    os.link(path, alias)
    with pytest.raises(ValueError, match="same output file"):
        ea._validate_output_paths(None, path, alias)


def test_bad_output_path_is_rejected_before_drive(monkeypatch, tmp_path):
    monkeypatch.setattr(
        ea, "drive",
        lambda *_args, **_kwargs: pytest.fail("drive ran before output validation"))
    with pytest.raises(SystemExit) as caught:
        ea.main(["run", "--out", str(tmp_path / "frame.jpg"), "--quiet"])
    assert caught.value.code == 2


def test_output_directory_is_rejected_before_drive(monkeypatch, tmp_path):
    directory = tmp_path / "frame.gif"
    directory.mkdir()
    monkeypatch.setattr(
        ea, "drive",
        lambda *_args, **_kwargs: pytest.fail("drive ran before output validation"))
    with pytest.raises(SystemExit) as caught:
        ea.main(["run", "--out", str(directory), "--quiet"])
    assert caught.value.code == 2


def test_json_serialization_failure_does_not_truncate_destination(tmp_path):
    path = tmp_path / "record.json"
    path.write_text("keep me", encoding="utf-8")
    with pytest.raises(TypeError):
        ea._write_json_document(path, {"unsupported": object()})
    assert path.read_text(encoding="utf-8") == "keep me"


def test_run_outputs_share_one_source_snapshot(monkeypatch, tmp_path):
    snapshots = []
    documents = []
    geometries = []

    def fake_source():
        snapshot = {"head": str(len(snapshots)), "branch": "test",
                    "dirty": False}
        snapshots.append(snapshot)
        return snapshot

    def fake_drive(config, progress=False, on_setup=None, **_kwargs):
        if on_setup is not None:
            on_setup("node", None)
        frame = SimpleNamespace(step=0, to_json=lambda: {"step": 0})
        return ea.DriveResult([frame], ea.Terminator.STEPS)

    monkeypatch.setattr(ea, "source_commit", fake_source)
    monkeypatch.setattr(ea, "drive", fake_drive)
    monkeypatch.setattr(
        ea, "_write_json_document",
        lambda path, document: documents.append((path, document)))
    monkeypatch.setattr(
        ea, "_write_geometry",
        lambda path, node, inputs, quiet, source=None:
        geometries.append((path, node, inputs, quiet, source)))

    json_path = tmp_path / "run.json"
    geometry_path = tmp_path / "geometry.json"
    assert ea.main(["run", "--steps", "0", "--out", "",
                    "--json", str(json_path), "--geometry",
                    str(geometry_path), "--quiet"]) == 0
    assert len(snapshots) == 1
    assert documents[0][1]["source"] is snapshots[0]
    assert geometries[0][-1] is snapshots[0]


def test_source_is_captured_before_drive_and_reused_on_failure(monkeypatch,
                                                               tmp_path):
    events = []
    snapshot = {"head": "start", "branch": "test", "dirty": False}

    def fake_source():
        events.append("source")
        return snapshot

    def failed_drive(config, progress=False, on_setup=None, **_kwargs):
        events.append("drive")
        on_setup("node", "inputs")
        raise RuntimeError("engine failed")

    geometries = []
    monkeypatch.setattr(ea, "source_commit", fake_source)
    monkeypatch.setattr(ea, "drive", failed_drive)
    monkeypatch.setattr(
        ea, "_write_geometry",
        lambda path, node, inputs, quiet, source=None:
        geometries.append((path, node, inputs, quiet, source)))

    path = tmp_path / "recovery.json"
    with pytest.raises(RuntimeError, match="engine failed"):
        ea.main(["run", "--out", "", "--geometry", str(path), "--quiet"])
    assert events == ["source", "drive"]
    assert geometries == [(str(path), "node", "inputs", True, snapshot)]


def test_recovery_write_failure_does_not_mask_drive_error(monkeypatch,
                                                          tmp_path):
    def failed_drive(config, progress=False, on_setup=None, **_kwargs):
        on_setup("node", None)
        raise RuntimeError("engine failed")

    monkeypatch.setattr(ea, "drive", failed_drive)
    monkeypatch.setattr(
        ea, "_write_geometry",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            KeyboardInterrupt("geometry failed")))

    with pytest.raises(RuntimeError, match="engine failed") as caught:
        ea.main(["run", "--out", "", "--geometry",
                 str(tmp_path / "recovery.json"), "--quiet"])
    assert caught.value.__notes__ == [
        "could not write recovery geometry %r: geometry failed"
        % str(tmp_path / "recovery.json")]


def test_json_failure_cannot_suppress_successful_geometry(monkeypatch,
                                                          tmp_path):
    writes = []

    def fake_drive(config, progress=False, on_setup=None, **_kwargs):
        on_setup("node", None)
        frame = SimpleNamespace(step=0, to_json=lambda: {"step": 0})
        return ea.DriveResult([frame], ea.Terminator.STEPS)

    monkeypatch.setattr(ea, "drive", fake_drive)
    monkeypatch.setattr(
        ea, "_write_geometry",
        lambda *_args, **_kwargs: writes.append("geometry"))
    monkeypatch.setattr(
        ea, "_write_json_document",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            TypeError("json failed")))

    with pytest.raises(TypeError, match="json failed"):
        ea.main(["run", "--out", "", "--json", str(tmp_path / "run.json"),
                 "--geometry", str(tmp_path / "geometry.json"), "--quiet"])
    assert writes == ["geometry"]
