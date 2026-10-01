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
No unlabeled two-dimensional band emerges and falsifier 8 holds on this host.

Two reads depend on the rounding of the eigenvalues at these tolerances. The
three copies of the covariant eigenvalue 20.6346 are computed 2.5e-14 apart,
1.1e-15 of the largest eigenvalue modulus, so the grouping tolerance reads
them as two groups, neither isolated, and the first is reported as a
falsifier-10 multiplicity. The isotypic read of the highest T-averaged band
splits its one isotype in two at the isotypic tolerance, so that band is read
as "1 x 3 sheets x 1 + 0 x 3 sheets x 1" and not as a spin doublet.
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
    """Five of the six eigenvalues of the covariant operator are read as
    isolated colour triplets on which the rotations do not act. The three
    copies of the eigenvalue 20.6346 are read as two groups 2.5e-14 apart: a
    band of rank three and an empty band, neither isolated, the first
    reducible with a commutant of dimension nine."""
    read = declared["covariant"]
    bands = read["frames"][0]["bands"]
    assert [b["rank"] for b in bands] == [3, 3, 3, 3, 3, 0, 3]
    np.testing.assert_allclose(
        [b["center"].real for b in bands],
        [0.4891, 4.1503, 9.8921, 18.1801, 20.6346, 20.6346, 21.9576],
        atol=1e-4)
    assert not any(b["symmetry_acts"] or b["doublet_candidate"]
                   for b in bands)
    triplets = bands[:4] + bands[6:]
    for band in triplets:
        assert band["colour_acts"]
        assert band["content"] == "1 x 3 sheets x 1"
        assert band["isolated"]
    split, empty = bands[4], bands[5]
    assert not split["isolated"] and not empty["isolated"]
    assert split["gap"] == pytest.approx(2.5e-14, rel=0.1)
    assert split["content"] == "1 x 1 sheet x 3"
    assert split["commutant_dimension"] == 9
    assert not split["colour_acts"]
    assert empty["classification"] == "an empty band"


def test_the_averaged_operator_is_three_bands_of_rank_six(declared):
    """The T-averaged operator's three bands are isolated, of rank six each,
    with the sheets and the rotations acting. The two lower bands are read
    as a spinor doublet times the three sheets; the highest is read as
    "1 x 3 sheets x 1 + 0 x 3 sheets x 1" and not as a spin doublet."""
    read = declared["t_averaged"]
    bands = read["frames"][0]["bands"]
    assert [b["rank"] for b in bands] == [6, 6, 6]
    for band in bands:
        assert band["colour_acts"] and band["symmetry_acts"]
        assert band["isolated"]
        assert not band["doublet_candidate"]
    assert [b["content"] for b in bands] == [
        "2 x 3 sheets x 1", "2 x 3 sheets x 1",
        "1 x 3 sheets x 1 + 0 x 3 sheets x 1"]
    assert [b["spin_doublet"] for b in bands] == [True, True, False]


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
    # the split covariant eigenvalue is reported under falsifier 10
    unexplained = read["falsifier_10_unexplained_multiplicities"]
    if operator == "covariant":
        (item,) = unexplained
        assert item.startswith("single-level synthesis band 4 at (20.6346")
        assert item.endswith(": 1 x 1 sheet x 3")
    else:
        assert unexplained == []


@pytest.mark.slow
def test_the_baryon_driver_adds_the_read_only_when_asked():
    """At the tolerances of the 2026-09-23 run the declared host with the
    content (1, 1, 1) is stationary as built and keeps its tetrahedral group,
    so the read proceeds; at the declared 1e-15 the joint Newton's one
    accepted step moves the six squared lengths by 1e-8 in an asymmetric
    pattern before it finds no descent, the cell keeps only the identity,
    and the read is refused by name (#1298)."""
    from tests.drivers import _recursion_run_2026_09_23 as RUN

    config = bp.default_config([1.0], [1.0], selected_contents=[(1, 1, 1)],
                               tolerances=RUN.TOLERANCES)
    assert "isospin_doublet" not in config
    plain = bp.evaluate_content((1, 1, 1), 1.0, 1.0, config)
    assert "isospin_doublet" not in plain
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
