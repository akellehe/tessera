# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""The isospin-doublet observation on the three-sheeted unit-monopole host
(`tessera.drivers.isospin_doublet`), and the baryon driver's
``--isospin-doublet`` option.

The assertions record what the detector observes on this host; nothing is
tuned to make a doublet appear. On the covariant h_1(z, U) the rotations do not
act on any band (h_1 is not rotation covariant at the monopole connection), and
every band is one simple base eigenvalue times the three sheets. On the
T-averaged operator the bands are the three spinor doublets 2, 2', 2'' times the
three sheets: spin content and colour, with no multiplicity left over. So no
unlabeled two-dimensional band emerges, falsifier 8 holds on this host, and no
band carries a falsifier-10 multiplicity.
"""

import numpy as np
import pytest

from tessera.drivers import baryon_poles as bp
from tessera.drivers import isospin_doublet as iso


@pytest.fixture(scope="module")
def declared():
    return iso.observe_host(iso.declared_carrier())


def test_the_declared_layout_is_three_sheets_of_six_edges():
    cells, sheet_of, base_of = iso.sheet_layout()
    assert len(cells) == 18
    assert sheet_of == [t for t in range(3) for _ in range(6)]
    assert base_of == list(range(6)) * 3
    _, spinorial = iso.symmetry()
    assert spinorial


def test_the_covariant_operator_is_six_colour_triplets(declared):
    read = declared["covariant"]
    bands = read["frames"][0]["bands"]
    assert [b["rank"] for b in bands] == [3] * 6
    for band in bands:
        assert band["colour_acts"]
        assert not band["symmetry_acts"]
        assert band["content"] == "1 x 3 sheets x 1"
        assert band["isolated"]
        assert not band["doublet_candidate"]


def test_the_averaged_operator_is_three_spin_doublets_times_colour(declared):
    read = declared["t_averaged"]
    bands = read["frames"][0]["bands"]
    assert [b["rank"] for b in bands] == [6, 6, 6]
    for band in bands:
        assert band["colour_acts"] and band["symmetry_acts"]
        assert band["content"] == "2 x 3 sheets x 1"
        assert band["spin_doublet"]
        assert not band["doublet_candidate"]


@pytest.mark.parametrize("operator", ["covariant", "t_averaged"])
def test_no_isospin_doublet_on_the_host(declared, operator):
    read = declared[operator]
    assert read["candidates"] == []
    assert [c["status"] for c in read["conditions"]] == [
        "Failed", "NotEvaluable", "NotEvaluable"]
    assert read["conditions"][0]["failing"] == ["two-dimensional-flavour-band"]
    assert "deferred" in read["conditions"][2]["evidence"][0]["detail"]
    assert not read["doublet_observed"]
    assert read["falsifier_8_no_isospin_doublet"]
    assert read["falsifier_10_unexplained_multiplicities"] == []


def test_the_baryon_driver_adds_the_read_only_when_asked():
    alignment = bp.aligned_doublet_frame(bp.monopole_support(),
                                         bp.rotation_group())
    config = bp.default_config([1.0], [1.0], selected_contents=[(1, 1, 1)])
    assert "isospin_doublet" not in config
    plain = bp.evaluate_content((1, 1, 1), 1.0, 1.0, config, alignment)
    assert "isospin_doublet" not in plain
    config["isospin_doublet"] = True
    extended = bp.evaluate_content((1, 1, 1), 1.0, 1.0, config, alignment)
    read = extended["isospin_doublet"]
    assert set(read) == {"covariant", "t_averaged"}
    assert read["t_averaged"]["candidates"] == []
    # every other output of the record is the same
    for key in ("covariant_spectrum", "averaged_spectrum", "band_energies"):
        assert np.allclose(np.asarray(plain[key]), np.asarray(extended[key]))
    assert set(extended) - set(plain) == {"isospin_doublet"}


def test_the_command_line_option_is_declared():
    args = bp.build_parser().parse_args(["run", "--isospin-doublet"])
    assert args.isospin_doublet
    assert not bp.build_parser().parse_args(["run"]).isospin_doublet
