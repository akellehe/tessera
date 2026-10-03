# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""The isospin-doublet observation on the three-sheeted unit-monopole host
(`tessera.drivers.isospin_doublet`), and the baryon driver's
``--isospin-doublet`` option.

The assertions record what the detector observes on this host at the declared
tolerances (1e-15 each); nothing is tuned to make a doublet appear. On the
covariant h_1(z, U) the rotations do not act on any band (h_1 is not rotation
covariant at the monopole connection), and each of its six eigenvalues is one
simple base eigenvalue times the three sheets. On the T-averaged operator the
eigenvalues are the three spinor doublets 2, 2', 2'' times the three sheets.
No unlabeled two-dimensional band emerges, falsifier 8 holds on this host,
and no band carries a multiplicity the declared symmetry does not explain.
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


def test_the_covariant_operator_is_colour_triplets(declared):
    """The six eigenvalues of the covariant operator are read as isolated
    colour triplets on which the rotations do not act."""
    read = declared["covariant"]
    bands = read["frames"][0]["bands"]
    assert [b["rank"] for b in bands] == [3] * 6
    np.testing.assert_allclose(
        [b["center"].real for b in bands],
        [0.4891, 4.1503, 9.8921, 18.1801, 20.6346, 21.9576], atol=1e-4)
    assert not any(b["symmetry_acts"] or b["doublet_candidate"]
                   for b in bands)
    for band in bands:
        assert band["colour_acts"]
        assert band["content"] == "1 x 3 sheets x 1"
        assert band["commutant_dimension"] == 1
        assert band["isolated"]


def test_the_averaged_operator_is_three_bands_of_rank_six(declared):
    """The T-averaged operator's three bands are isolated, of rank six each,
    with the sheets and the rotations acting, each a spinor doublet times
    the three sheets."""
    read = declared["t_averaged"]
    bands = read["frames"][0]["bands"]
    assert [b["rank"] for b in bands] == [6, 6, 6]
    for band in bands:
        assert band["colour_acts"] and band["symmetry_acts"]
        assert band["isolated"]
        assert not band["doublet_candidate"]
    assert [b["content"] for b in bands] == ["2 x 3 sheets x 1"] * 3
    assert [b["spin_doublet"] for b in bands] == [True, True, True]


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


@pytest.mark.slow
def test_the_baryon_driver_adds_the_read_only_when_asked():
    """At the tolerances of the 2026-09-23 run the declared host with the
    content (1, 1, 1) is stationary as built and keeps its tetrahedral group,
    so the three preconditions of the cell's own spin read hold: the spin is
    read in the relaxed cell's own frame and the record carries no flag."""
    from tests.drivers import _recursion_run_2026_09_23 as RUN

    config = bp.default_config([1.0], [1.0], selected_contents=[(1, 1, 1)],
                               tolerances=RUN.TOLERANCES)
    assert "isospin_doublet" not in config
    plain = bp.evaluate_content((1, 1, 1), 1.0, 1.0, config)
    assert "isospin_doublet" not in plain
    assert plain["flags"] == []
    assert plain["relaxation"]["symmetry"]["tetrahedral"]
    assert plain["relaxation"]["spin_frame"] == bp.SPIN_FRAME_OF_THE_CELL == \
        "the relaxed cell's own rotation group"
    # the record carries the frame's certificate and the measurements of
    # every doublet content's spin split
    for sheet in plain["relaxation"]["frames"]:
        assert sheet["reference_certified"] is True
        assert sheet["half_turn_trace_residual"] is not None
    for read in plain["doublet_reads"]:
        split = read["spin_split"]
        assert set(split["dimensions"]) == set(read["sectors"])
        assert sum(split["dimensions"].values()) > 0
        assert split["polynomial_residual"] < 1e-13
    # the with-quartic read's flags are flags of the content
    names = [flag["name"] for flag in plain["flags"]]
    for flag in plain["quartic"]["truncation"]["flags"]:
        assert flag["name"] in names
    config["isospin_doublet"] = True
    extended = bp.evaluate_content((1, 1, 1), 1.0, 1.0, config)
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
