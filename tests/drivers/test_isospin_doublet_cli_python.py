# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The command line and the record of the isospin-doublet driver
(`tessera.drivers.isospin_doublet`).

Without ``--kappa`` the driver reads only the declared host, which takes well
under a second, so `main` is run for real. With ``--kappa`` every content is
first relaxed to self-consistency (about ten seconds each); that path is run
once, for one content, and marked slow.
"""
import json

import numpy as np
import pytest

from tessera.drivers import baryon_poles as bp
from tessera.drivers import isospin_doublet as iso


def test_the_declared_defaults():
    args = iso.build_parser().parse_args(["run"])
    assert args.kappa is None and args.beta is None and args.content is None
    assert args.edge_squared == bp.DECLARED_EDGE_SQUARED
    assert args.json is None and not args.quiet


def test_contents_are_repeatable_triples():
    args = iso.build_parser().parse_args(
        ["run", "--kappa", "1", "2", "--beta", "0.5", "--content", "1", "1",
         "1", "--content", "3", "0", "0"])
    assert args.kappa == [1.0, 2.0] and args.beta == [0.5]
    assert args.content == [[1, 1, 1], [3, 0, 0]]


@pytest.mark.parametrize("argv,name", [
    (["run", "--content", "1", "1"], "--content"),
    (["run", "--kappa", "x"], "--kappa"),
    ([], "command"),
])
def test_invalid_arguments_are_refused_by_name(argv, name, capsys):
    with pytest.raises(SystemExit) as stop:
        iso.build_parser().parse_args(argv)
    assert stop.value.code == 2
    assert name in capsys.readouterr().err


def test_main_reads_the_declared_host_and_writes_the_json(tmp_path):
    path = tmp_path / "doublet.json"
    result = iso.main(["run", "--json", str(path), "--quiet"])
    assert result["relaxed"] == [] and result["spinorial"] is True
    document = json.loads(path.read_text())
    assert set(document) == {"declared_host", "spinorial", "relaxed"}
    for key in ("covariant", "t_averaged"):
        read = document["declared_host"][key]
        assert read["candidates"] == []
        assert read["falsifier_8_no_isospin_doublet"] is True
        assert read["falsifier_10_refinement_measured"] is False
        assert read["symmetry"].startswith("T = A_4")
        bands = read["frames"][0]["bands"]
        assert sum(b["rank"] for b in bands) == 18
        # complex numbers are written as {"re", "im"}
        assert set(bands[0]["center"]) == {"re", "im"}
    assert document["declared_host"]["covariant"]["operator"] == \
        "covariant h_1(z, U)"


def test_main_prints_every_band_unless_quiet(capsys):
    iso.main(["run"])
    out = capsys.readouterr().out
    assert out.count("declared host covariant:") == 1
    assert out.count("declared host t_averaged:") == 1
    assert out.count("    band ") == 6 + 3
    assert out.count("emergence Failed") == 2
    iso.main(["run", "--quiet"])
    assert capsys.readouterr().out == ""


def test_a_scaled_host_has_the_same_bands():
    """The carrier operator is homogeneous in the squared lengths, so a host
    of another edge length has the same band structure and reads the same."""
    scaled = iso.drive(edge_squared=2.0)["declared_host"]
    declared = iso.drive()["declared_host"]
    for key in ("covariant", "t_averaged"):
        assert [b["rank"] for b in scaled[key]["frames"][0]["bands"]] == \
            [b["rank"] for b in declared[key]["frames"][0]["bands"]]
        assert scaled[key]["candidates"] == []


def test_the_declaration_carries_the_layout_and_the_symmetry():
    actions, spinorial = iso.symmetry()
    assert len(actions) == 12 and spinorial
    declaration = iso.declaration(np.eye(18), "identity", actions, spinorial)
    assert declaration.operator_name == "identity"
    assert declaration.degree == 1
    assert len(declaration.cells) == 18
    assert declaration.sheet_of_cell == [t for t in range(3) for _ in range(6)]
    assert len(declaration.symmetry) == 12
    assert len(declaration.frames) == 1
    assert declaration.frames[0].label == "single-level synthesis"


def test_the_record_of_an_operator_with_one_degenerate_band():
    """The identity on the 18 cells is one band of rank 18: the sheets act on
    it and so do the rotations, and the record says so."""
    actions, spinorial = iso.symmetry()
    read = iso.observe_host(np.eye(18, dtype=complex), actions, spinorial)
    for key in ("covariant", "t_averaged"):
        bands = read[key]["frames"][0]["bands"]
        assert [b["rank"] for b in bands] == [18]
        assert bands[0]["colour_acts"] and bands[0]["symmetry_acts"]


@pytest.mark.slow
def test_main_relaxes_each_declared_content(tmp_path, capsys):
    """With --kappa the host is relaxed to self-consistency for every declared
    (kappa, beta, content) and each relaxed host is read. Whether the solve
    converged is recorded, not required: under ruling (a) the density of
    content (1, 1, 1) is (1/3)(P_0 + P_1 + P_2) over bands of h_1, not I/6,
    and the self-consistency at kappa = beta = 1 need not settle within the
    declared iterations."""
    path = tmp_path / "relaxed.json"
    result = iso.main(["run", "--kappa", "1", "--beta", "1", "--content", "1",
                       "1", "1", "--json", str(path)])
    (entry,) = result["relaxed"]
    assert entry["kappa"] == 1.0 and entry["beta"] == 1.0
    assert entry["content"] == [1, 1, 1]
    assert isinstance(entry["relaxation_converged"], bool)
    assert set(entry["reads"]) == {"covariant", "t_averaged"}
    assert "kappa=1 beta=1 content=[1, 1, 1] covariant:" in \
        capsys.readouterr().out
    assert json.loads(path.read_text())["relaxed"][0]["content"] == [1, 1, 1]
