# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The command line and the small helpers of the nucleon-to-Delta synthesis
driver (`tessera.drivers.baryon_poles`).

The command line is exercised end to end through `main` with the scan point
replaced by a deterministic stand-in, because one real scan point relaxes ten
contents and takes minutes; the claims under test are the parsing, the
refusals by name, which outputs each flag writes, and what those outputs
contain. The helpers are exercised on inputs whose answer is known in closed
form. The quark verdict is read on the declared (unrelaxed) host, where the
three sheets are disjoint and so no component couples to another.

The stand-in content carries two doublet contents with different poles, so
that every output can be checked to report both: the summary text, the pole
table, the frame data and the labelled minima.
"""
import argparse
import itertools
import json

import numpy as np
import pytest

from tessera import cobordism as cob
from tessera import observables as obs
from tessera.drivers import baryon_poles as bp

HALF = str(bp.SPIN_HALF)
THREE = str(bp.SPIN_THREE_HALVES)


def _column(poles):
    """One column of a sector as `sector_entry` writes it: every pole with
    multiplicity two and its own (passing) certificates, and the lowest pole
    with its certificates repeated beside it."""
    poles = [complex(p) for p in poles]
    certificates = [{"sharp_spinor": True, "spinor_right_residual": 0.0,
                     "spinor_left_residual": 0.0, "spin_lift_sharp": True,
                     "spin_lift_right_residual": 0.0,
                     "spin_lift_left_residual": 0.0,
                     "colour_casimir_residual": 0.0} for _ in poles]
    lowest = min(poles, key=lambda p: (p.real, p.imag))
    column = {"poles": poles, "multiplicity": [2] * len(poles),
              "lowest_pole": lowest, "failed_certificates": [],
              "subspace_residual": [0.0] * len(poles),
              "residue_rank": [2] * len(poles),
              "separation": [1.0] * len(poles), "scale": 1.0,
              "compression_leakage": 1e-16,
              "pole_certificates": certificates}
    column.update(certificates[poles.index(lowest)])
    return column


def _doublet_read(doublet_content, triality, poles):
    """One doublet read: per spin, the quasi-free poles, and the same poles
    less 100 with the quartic."""
    sectors = {}
    for key, values in poles.items():
        irreps = bp.restriction(float(key), triality)
        sectors[key] = {"restriction_to_2T": irreps,
                        "nucleon_reading": "2" in irreps,
                        "delta_reading": sorted(irreps) == ["2'", "2''"],
                        "quasi_free": _column(values),
                        "with_quartic": _column([v - 100 for v in values])}
    return {"doublet_content": list(doublet_content),
            "total_triality": triality, "sectors": sectors}


def _record(content, kappa=1.0, beta=2.0):
    """A content with two doublet contents of different poles: in (0, 2, 1)
    the spin-1/2 sector has two poles, kappa + 0.1i and kappa + 1, and the
    spin-3/2 pole is kappa + beta + 1 + 0.2i; in (1, 1, 1) they are
    kappa + 2 + 0.1i and kappa + beta + 0.2i. The lowest spin-1/2 pole
    therefore comes from (0, 2, 1) and the lowest spin-3/2 pole from
    (1, 1, 1)."""
    return {"content": list(content), "doublet_reads": [
        _doublet_read([0, 2, 1], 1, {
            HALF: [complex(kappa, 0.1), complex(kappa + 1, 0)],
            THREE: [complex(kappa + beta + 1, 0.2)]}),
        _doublet_read([1, 1, 1], 0, {
            HALF: [complex(kappa + 2, 0.1)],
            THREE: [complex(kappa + beta, 0.2)]})]}


def _cheap_scan_point(kappa, beta, config, on_content=None):
    """A deterministic stand-in for one scan point with both spins present,
    so every pairing of `ratios` has a pole pair."""
    records = [_record([1, 1, 1], kappa, beta)]
    return {"kappa": kappa, "beta": beta,
            "elimination": config["elimination"], "failed_contents": [],
            "flagged_contents": [],
            "contents": records, "ratios": bp.ratios(records),
            "pole_table": bp.pole_table(records)}


@pytest.fixture
def cheap(monkeypatch):
    """Replace the scan point and record the configuration it is given."""
    seen = []

    def scan(kappa, beta, config, on_content=None):
        seen.append(dict(config))
        return _cheap_scan_point(kappa, beta, config)

    monkeypatch.setattr(bp, "scan_point", scan)
    return seen


# ------------------------------------------------------------ the parser


def test_the_declared_defaults():
    args = bp.build_parser().parse_args(["run"])
    assert args.command == "run"
    assert args.kappa == list(bp.DECLARED_KAPPAS)
    assert args.beta == list(bp.DECLARED_BETAS)
    assert args.edge_squared == bp.DECLARED_EDGE_SQUARED == 8.0
    assert args.regge_hinges == "interior"
    assert args.json is None and args.out is None
    assert not args.live and not args.quiet and not args.isospin_doublet
    assert args.band_selection == "continuation"
    assert args.fiber_moments == "r"
    assert args.fiber_pinning == "eigenvalues"
    assert bp.build_parser().parse_args(
        ["run", "--fiber-moments", "1"]).fiber_moments == "1"
    assert bp.build_parser().parse_args(
        ["run", "--fiber-moments", "bands"]).fiber_moments == "bands"
    assert bp.build_parser().parse_args(
        ["run", "--fiber-pinning", "power-sums"]).fiber_pinning == "power-sums"
    # every tolerance of the stack is an option, defaulting to 1e-15
    assert bp.DECLARED_TOLERANCE == 1e-15
    assert bp.tolerances_from(args) == {
        key: 1e-15 for key, _ in bp.TOLERANCES}
    assert "rank_tolerance" in dict(bp.TOLERANCES)
    tight = bp.build_parser().parse_args(
        ["run", "--rank-tolerance", "1e-10", "--tie-tolerance", "1e-8"])
    assert bp.tolerances_from(tight)["rank_tolerance"] == 1e-10
    assert bp.tolerances_from(tight)["tie_tolerance"] == 1e-8
    assert bp.tolerances_from(tight)["step_tolerance"] == 1e-15
    # the Villain weight is governed by an order, not by a tolerance
    assert "villain_tolerance" not in dict(bp.TOLERANCES)
    assert args.villain_order == bp.DECLARED_VILLAIN_ORDER == 10


#: Every tolerance of the stack, in the registry's order.
TOLERANCE_KEYS = [
    "rank_tolerance", "step_tolerance", "mean_field_tolerance",
    "band_tolerance", "certificate_tolerance", "allowability_tolerance",
    "tie_tolerance", "degeneracy_tolerance", "pole_rank_tolerance",
    "fluctuation_tolerance", "recursion_tolerance",
    "spin_sector_tolerance", "character_tolerance", "elimination_tolerance",
    "pure_gauge_tolerance", "gauge_resonance_radius", "truncation_tolerance",
    "hessian_reality_tolerance", "fibre_lift_tolerance", "isotypic_tolerance",
    "attachment_rank_tolerance", "quotient_rank_tolerance",
    "grown_cell_rank_tolerance", "move_tolerance",
    "admissibility_tolerance", "isospin_grouping_tolerance",
    "isospin_projector_tolerance", "isospin_invariance_tolerance",
    "isospin_commutant_tolerance",
    "isospin_hermiticity_tolerance", "isospin_transport_leakage_tolerance",
    "isospin_intertwining_tolerance", "isospin_min_relative_gap",
    "isospin_span_tolerance", "isospin_transport_rank_tolerance",
    "isospin_singular_value_grouping_tolerance",
    "isospin_member_splitting_tolerance", "isospin_occupation_tolerance",
]


def test_the_registry_lists_every_tolerance():
    """The registry is the complete list of the stack's tolerances: each key
    once, each with a one-phrase meaning, and each the detector's tolerance
    it names where it is one of `ISOSPIN_TOLERANCES`."""
    assert [key for key, _ in bp.TOLERANCES] == TOLERANCE_KEYS
    assert len(set(TOLERANCE_KEYS)) == len(TOLERANCE_KEYS) == 38
    assert all(isinstance(meaning, str) and meaning
               for _, meaning in bp.TOLERANCES)
    assert [key for key, _ in bp.ISOSPIN_TOLERANCES] == [
        key for key in TOLERANCE_KEYS if key.startswith("isospin_")]
    detector = bp.isospin_doublet_config()
    assert all(getattr(detector, field) == 1e-15
               for _, field in bp.ISOSPIN_TOLERANCES)
    detector = bp.isospin_doublet_config(
        {"isospin_grouping_tolerance": 1e-8})
    assert detector.grouping_tolerance == 1e-8
    assert detector.projector_tolerance == 1e-15
    # the detector's thresholds that are not tolerances stay the library's
    library = obs.IsospinDoubletConfig()
    for field in ("min_relative_gap", "contour_nodes",
                  "track_overlap_threshold", "min_frames",
                  "condition_number_cap"):
        assert getattr(detector, field) == getattr(library, field)


@pytest.mark.parametrize("key", TOLERANCE_KEYS)
def test_every_tolerance_is_an_option_defaulting_to_1e_15(key):
    """``--<key, with dashes>`` sets the tolerance ``key`` alone; without it
    the tolerance is 1e-15; and a value that is not positive is refused by
    the option's name."""
    option = "--" + key.replace("_", "-")
    declared = bp.tolerances_from(bp.build_parser().parse_args(["run"]))
    assert declared[key] == 1e-15
    set_ = bp.tolerances_from(
        bp.build_parser().parse_args(["run", option, "1e-7"]))
    assert set_[key] == 1e-7
    assert all(value == 1e-15 for other, value in set_.items()
               if other != key)
    with pytest.raises(SystemExit):
        bp.build_parser().parse_args(["run", option, "0"])


def test_the_config_records_every_tolerance():
    config = bp.default_config([1.0], [1.0])
    assert all(config[key] == bp.DECLARED_TOLERANCE
               for key, _ in bp.TOLERANCES)
    assert {key: config[key] for key in TOLERANCE_KEYS} == {
        key: 1e-15 for key in TOLERANCE_KEYS}
    config = bp.default_config([1.0], [1.0],
                               tolerances={"tie_tolerance": 1e-8})
    assert config["tie_tolerance"] == 1e-8
    assert config["rank_tolerance"] == bp.DECLARED_TOLERANCE
    assert bp.declared_tolerance(config, "tie_tolerance") == 1e-8
    assert bp.declared_tolerance({}, "tie_tolerance") == 1e-15
    with pytest.raises(ValueError, match="unknown tolerances"):
        bp.default_config([1.0], [1.0], tolerances={"tolerance": 1e-8})


def test_lists_of_couplings_are_parsed():
    args = bp.build_parser().parse_args(
        ["run", "--kappa", "0.25", "4", "--beta", "5", "--edge-squared", "2",
         "--regge-hinges", "all"])
    assert args.kappa == [0.25, 4.0] and args.beta == [5.0]
    assert args.edge_squared == 2.0 and args.regge_hinges == "all"


@pytest.mark.parametrize("argv,name", [
    (["run", "--eliminate", "phases"], "--eliminate"),
    (["run", "--band-selection", "by-hand"], "--band-selection"),
    (["run", "--edge-squared", "eight"], "--edge-squared"),
    (["run", "--fiber-moments", "-1"], "--fiber-moments"),
    (["run", "--fiber-moments", "all"], "--fiber-moments"),
    (["run", "--fiber-pinning", "trace"], "--fiber-pinning"),
    (["run", "--rank-tolerance", "0"], "--rank-tolerance"),
    (["run", "--tie-tolerance", "tight"], "--tie-tolerance"),
    (["run", "--regge-hinges", "boundary"], "--regge-hinges"),
    (["run", "--kappa", "one"], "--kappa"),
    ([], "command"),
])
def test_invalid_arguments_are_refused_by_name(argv, name, capsys):
    with pytest.raises(SystemExit) as stop:
        bp.build_parser().parse_args(argv)
    assert stop.value.code == 2
    assert name in capsys.readouterr().err


# ------------------------------------------------------------ main


def test_main_writes_the_json_and_the_points_file(cheap, tmp_path):
    path = tmp_path / "poles.json"
    result = bp.main(["run", "--kappa", "0.5", "1", "--beta", "2",
                      "--json", str(path), "--quiet"])
    assert [(p["kappa"], p["beta"]) for p in result["points"]] == \
        [(0.5, 2.0), (1.0, 2.0)]
    document = json.loads(path.read_text())
    assert document["stopped"] is False
    assert document["config"]["kappas"] == [0.5, 1.0]
    assert document["config"]["target_mass_ratio"] == pytest.approx(
        938.272 / 1232.0)
    assert "target_mass_squared_ratio" not in document["config"]
    # complex numbers are written as {"re", "im"}
    # by 2T reading the nucleon pole is the spin-1/2 pole of (1, 1, 1),
    # kappa + 2 + 0.1i, whose sector is the 2; the spin-1/2 sector of
    # (0, 2, 1), kappa + 0.1i, restricts to 2' and is the lift's nucleon
    ratios = document["points"][0]["ratios"]["quasi_free"]
    assert ratios["by_2T_reading"]["nucleon_pole"] == {"re": 2.5, "im": 0.1}
    assert ratios["by_spin_lift"]["nucleon_pole"] == {"re": 0.5, "im": 0.1}
    assert set(document["host"]) == {"monopole", "averaged_eigenvalues",
                                      "reference_carrier",
                                      "reference_certified",
                                      "reference_doublet", "trialities",
                                      "intertwining_residual"}
    lines = (tmp_path / "poles.points.jsonl").read_text().splitlines()
    assert len(lines) == 3
    first = json.loads(lines[0])
    assert set(first) == {"config", "host"}
    assert first["host"]["monopole"]["monopole_number"] == 1
    assert [json.loads(line)["kappa"] for line in lines[1:]] == [0.5, 1.0]


def test_main_passes_the_declared_options_to_every_point(cheap):
    bp.main(["run", "--kappa", "1", "--beta", "1",
             "--band-selection", "sort-every-iterate",
             "--eliminate", "lengths", "--edge-squared", "3",
             "--isospin-doublet", "--quiet"])
    (config,) = cheap
    assert config["band_selection"] == "sort-every-iterate"
    assert config["elimination"] == "lengths"
    assert config["edge_squared"] == 3.0
    assert config["isospin_doublet"] is True
    assert len(config["contents"]) == 10


def test_without_the_isospin_flag_the_config_does_not_carry_it(cheap):
    bp.main(["run", "--kappa", "1", "--beta", "1", "--quiet"])
    assert "isospin_doublet" not in cheap[0]


def test_main_without_json_writes_no_points_file(cheap, tmp_path,
                                                 monkeypatch):
    monkeypatch.chdir(tmp_path)
    bp.main(["run", "--kappa", "1", "--beta", "1", "--quiet"])
    assert list(tmp_path.iterdir()) == []


def test_main_renders_the_final_frame(cheap, tmp_path):
    path = tmp_path / "poles.png"
    bp.main(["run", "--kappa", "1", "2", "--beta", "1", "--out", str(path),
             "--quiet"])
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_progress_and_summary_are_printed_unless_quiet(cheap, capsys):
    """The progress of each point and the final summary both carry every
    (content, doublet content) pair, the labelled minima and the ratios."""
    bp.main(["run", "--kappa", "1", "--beta", "2"])
    out = capsys.readouterr().out
    assert out.count("kappa=1 beta=2: 1 contents, 0 without a value, "
                     "0 flagged") == 2
    for pair in ("[0, 2, 1] (triality 1)", "[1, 1, 1] (triality 0)"):
        assert out.count("\n  content [1, 1, 1], doublet content %s | spin"
                         % pair) == 2
    assert out.count("lowest over the doublet contents of content") == 2
    assert out.count("lowest over every (content, doublet content) pair") \
        == 2
    assert out.count("s_N/s_D=") == 8
    assert "mode: controlled synthesis" in out
    assert "target m_N/m_Delta = 0.7616" in out
    bp.main(["run", "--kappa", "1", "--beta", "2", "--quiet"])
    assert capsys.readouterr().out == ""


def test_main_live_refuses_a_file_backend_by_name(cheap, monkeypatch):
    import matplotlib
    monkeypatch.setattr(matplotlib, "get_backend", lambda: "agg")
    with pytest.raises(RuntimeError, match="'agg'"):
        bp.main(["run", "--kappa", "1", "--beta", "1", "--live", "--quiet"])
    assert cheap == []


def test_main_live_runs_the_live_drive(cheap, monkeypatch, tmp_path):
    calls = []

    def live(config, progress=False, points_file=None, keep_open=False):
        calls.append((progress, points_file, keep_open))
        return bp.drive(config, progress=progress, points_file=points_file)

    monkeypatch.setattr(bp, "drive_live", live)
    path = tmp_path / "live.json"
    held = []
    monkeypatch.setattr(bp, "hold_live_window",
                        lambda message: held.append(path.exists()))
    bp.main(["run", "--kappa", "1", "--beta", "1", "--live", "--quiet",
             "--json", str(path)])
    assert calls == [(False, str(tmp_path / "live.points.jsonl"), True)]
    assert path.exists()
    assert held == [True]


def test_a_stopped_drive_says_so():
    result = bp.drive(bp.default_config([1.0], [1.0]),
                      stop_requested=lambda: True)
    assert result["stopped"] is True and result["points"] == []


# ------------------------------------------------------------ the outputs


@pytest.fixture
def point():
    return _cheap_scan_point(1.0, 2.0, bp.default_config([1.0], [2.0]))


def test_the_summary_names_every_pairing(point):
    lines = bp.summary({"points": [point]}).splitlines()
    # the mode, the point, the content's mean-field line, two pairs, two
    # minima lines and four ratios
    assert len(lines) == 1 + 1 + 1 + 2 + 2 + 4
    # the stand-in record carries no mean-field solve, and says so
    assert lines[2] == "  content [1, 1, 1] mean field unrecorded"
    ratio_lines = lines[7:]
    # the 2T reading, the reading of WP v18, comes first; the spin of the
    # lift beside it
    assert "by_2T_reading" in ratio_lines[0]
    assert "by_spin_lift" in ratio_lines[1]
    assert "Delta restriction 2'+2''" in ratio_lines[1]
    empty = bp.summary({"points": [dict(point, ratios={
        name: {"by_2T_reading": None, "by_spin_lift": None}
        for name in ("quasi_free", "with_quartic")})]})
    assert empty.count("no pole pair") == 4


def test_the_summary_reports_every_doublet_content_on_its_own_line(point):
    """One line per (content, doublet content) pair, both spins and both
    columns on it, every pole with its multiplicity and certificates: the
    two spin-1/2 poles of (0, 2, 1) are both there, and so are the poles of
    (1, 1, 1), which no minimum picks for spin 1/2."""
    lines = bp.summary({"points": [point]}).splitlines()
    first, second = lines[3], lines[4]
    assert first.startswith("  content [1, 1, 1], doublet content [0, 2, 1] "
                            "(triality 1) | spin 1/2 (restricts to 2'): ")
    assert ("quasi-free 1+0.1i x2 [spinor sharp, lift sharp, colour 0], "
            "2+0i x2 [spinor sharp, lift sharp, colour 0] {read certified, "
            "leakage 1e-16}") in first
    assert ("with quartic -99+0.1i x2 [spinor sharp, lift sharp, colour 0], "
            "-98+0i x2 [spinor sharp, lift sharp, colour 0]") in first
    assert "spin 3/2 (restricts to 2''+2): quasi-free 4+0.2i x2" in first
    assert second.startswith("  content [1, 1, 1], doublet content "
                             "[1, 1, 1] (triality 0) | spin 1/2 (restricts "
                             "to 2): quasi-free 3+0.1i x2")
    assert "spin 3/2 (restricts to 2'+2''): quasi-free 3+0.2i x2" in second
    assert "with quartic -97+0.2i x2" in second


def test_the_minima_are_labelled_with_their_doublet_content(point):
    lines = bp.summary({"points": [point]}).splitlines()
    assert lines[5] == (
        "  lowest over the doublet contents of content [1, 1, 1]: "
        "spin 1/2 quasi-free 1+0.1i from doublet content [0, 2, 1]; "
        "spin 1/2 with quartic -99+0.1i from doublet content [0, 2, 1]; "
        "spin 3/2 quasi-free 3+0.2i from doublet content [1, 1, 1]; "
        "spin 3/2 with quartic -97+0.2i from doublet content [1, 1, 1]")
    assert lines[6].startswith(
        "  lowest over every (content, doublet content) pair: spin 1/2 "
        "quasi-free 1+0.1i from content [1, 1, 1], doublet content "
        "[0, 2, 1];")
    # by 2T reading: the lowest pole of a sector restricting to a 2 is the
    # spin-1/2 pole of (1, 1, 1), whose sector is the 2 itself, since the
    # spin-1/2 sector of (0, 2, 1) restricts to 2'; the Delta reading is the
    # 2' + 2'' sector of (1, 1, 1)
    by_reading = lines[7]
    assert "by_2T_reading" in by_reading
    assert ("s_N=3+0.1i (content [1, 1, 1], doublet content [1, 1, 1]) "
            "s_D=3+0.2i (content [1, 1, 1], doublet content [1, 1, 1])") \
        in by_reading
    # by the spin of the lift: the lowest spin-1/2 pole over the lowest
    # spin-3/2 pole
    by_spin_lift = lines[8]
    assert "by_spin_lift" in by_spin_lift
    assert ("s_N=1+0.1i (content [1, 1, 1], doublet content [0, 2, 1]) "
            "s_D=3+0.2i (content [1, 1, 1], doublet content [1, 1, 1])") \
        in by_spin_lift


def test_a_tie_for_a_minimum_names_every_tied_doublet_content():
    """Two doublet contents whose lowest spin-1/2 poles share their real part
    are both named: the first in the record's order as the minimum, the
    other as tied with it."""
    record = {"content": [2, 1, 0], "doublet_reads": [
        _doublet_read([0, 2, 1], 1, {HALF: [5.0 + 0.5j]}),
        _doublet_read([1, 2, 0], 2, {HALF: [5.0 - 0.5j]})]}
    best = bp.lowest_poles(record, "quasi_free")[HALF]
    assert best["doublet_content"] == [0, 2, 1]
    assert [t["doublet_content"] for t in best["tied"]] == [[1, 2, 0]]
    assert bp.lowest_poles(record, "quasi_free")[THREE] is None
    text = bp.lowest_lines([record])[0]
    assert ("spin 1/2 quasi-free 5+0.5i from doublet content [0, 2, 1] "
            "(tied to 1e-15 with doublet content [1, 2, 0])") in text
    assert best["tie_tolerance"] == bp.DECLARED_TOLERANCE
    assert "spin 3/2 quasi-free none" in text
    out = bp.ratios([record])["quasi_free"]
    assert out["by_spin_lift"] is None and out["by_2T_reading"] is None


def test_the_pole_table_keeps_every_pole_of_every_doublet_content(point):
    """Every pole, not only each sector's lowest, in ascending order of real
    part, with the doublet content on every row."""
    table = point["pole_table"]
    rows = table["quasi_free"][HALF]
    assert [(row["pole"], row["doublet_content"], row["lowest_in_sector"])
            for row in rows] == [(1 + 0.1j, [0, 2, 1], True),
                                 (2 + 0j, [0, 2, 1], False),
                                 (3 + 0.1j, [1, 1, 1], True)]
    assert all(row["content"] == [1, 1, 1] and row["multiplicity"] == 2
               and row["sharp_spinor"] and row["spin_lift_sharp"]
               and row["failed_certificates"] == []
               for row in rows)
    assert [row["doublet_content"] for row in table["with_quartic"][THREE]] \
        == [[1, 1, 1], [0, 2, 1]]
    assert [row["pole"] for row in table["with_quartic"][THREE]] == \
        [-97 + 0.2j, -96 + 0.2j]


def test_the_frame_data_carries_every_pole_of_every_doublet_content(point):
    data = bp.frame_data([point], 0)
    marks = {(m["column"], m["spin"], tuple(m["doublet_content"]), m["pole"])
             for m in data["marks"]}
    assert marks == {
        ("quasi_free", HALF, (0, 2, 1), 1 + 0.1j),
        ("quasi_free", HALF, (0, 2, 1), 2 + 0j),
        ("quasi_free", THREE, (0, 2, 1), 4 + 0.2j),
        ("quasi_free", HALF, (1, 1, 1), 3 + 0.1j),
        ("quasi_free", THREE, (1, 1, 1), 3 + 0.2j),
        ("with_quartic", HALF, (0, 2, 1), -99 + 0.1j),
        ("with_quartic", HALF, (0, 2, 1), -98 + 0j),
        ("with_quartic", THREE, (0, 2, 1), -96 + 0.2j),
        ("with_quartic", HALF, (1, 1, 1), -97 + 0.1j),
        ("with_quartic", THREE, (1, 1, 1), -97 + 0.2j)}
    assert len(data["marks"]) == 10
    # each doublet content has its own slot, labelled by its digits, and the
    # two spins of one pair sit on either side of it
    assert data["slots"] == [(0.0, "021"), (1.0, "111")]
    # the stand-in record carries no mean-field solve: no callout
    assert data["groups"] == [{"label": "111", "first": 0.0, "last": 1.0,
                               "solve": None}]
    assert sorted({round(m["x"] - m["slot"], 12) for m in data["marks"]}) \
        == [-0.2, 0.2]
    # the ratio rows name the pairs they compare
    quasi_free = [r for r in data["ratios"] if r["column"] == "quasi_free"]
    assert len(quasi_free) == 1
    # the frame draws the ratio by 2T reading: the type-2 sector of
    # (1, 1, 1) over its 2' + 2'' sector
    assert bp.ratio_pair_text(quasi_free[0]) == "N 111|111 / D 111|111"
    assert quasi_free[0]["ratio"] == pytest.approx((3 + 0.1j) / (3 + 0.2j))


def test_the_drawn_frame_has_one_mark_per_pole(point):
    """`draw_frame` draws what `frame_data` lists: every pole is one plotted
    point of its column's panel."""
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    figure = plt.figure(figsize=bp.FIGURE_SIZE)
    try:
        bp.draw_frame(figure, [point], 0)
        quasi_free, quartic, ratio, pairs = figure.axes
        drawn = sorted(float(y) for line in quasi_free.get_lines()
                       if line.get_label() in ("spin 1/2", "spin 3/2")
                       for y in line.get_ydata())
        assert drawn == [1.0, 2.0, 3.0, 3.0, 4.0]
        assert [t.get_text() for t in quasi_free.get_legend().get_texts()] \
            == ["spin 1/2", "spin 3/2"]
        assert [t.get_text() for t in quasi_free.get_xticklabels(minor=True)] \
            == ["021", "111"]
        listing = "\n".join(t.get_text() for t in pairs.texts)
        assert "N 111|111 / D 111|111" in listing
    finally:
        plt.close(figure)


def _solved(record, **relaxation):
    """The record with a mean-field solve record of the given fields."""
    solved = dict(record)
    solved["relaxation"] = dict(relaxation)
    return solved


def test_solve_state_reads_the_solve_or_the_missing_value_of_each_record():
    record = {"content": [1, 1, 1], "doublet_reads": []}
    assert bp.solve_state(record) is None
    assert bp.solve_state(_solved(record, converged=True, accepted_updates=6,
                                  stop_reason="converged")) == {
        "state": "converged", "reason": None, "accepted_updates": 6}
    assert bp.solve_state(_solved(
        record, converged=False, accepted_updates=12,
        stop_reason="no move and no scaled step lowers the residual "
                    "norm")) == {
        "state": "not converged", "reason": "no descent",
        "accepted_updates": 12}
    # every stop a drive names, and every reason a read has no value, has a
    # short name
    assert bp.STOP_SHORT == {
        "no move and no scaled step lowers the residual norm": "no descent",
        "the step has no value at the point reached": "no step",
        "a declared limit was reached": "declared limit",
        "the squared lengths overflowed the double": "lengths overflowed",
        "the cell is not a tetrahedron after its Pachner moves": "cell moved",
        "a read the poles are built on has no value": "read without a value",
    }
    # a stop reason without a short name is shown whole
    assert bp.solve_state(_solved(record, converged=False, accepted_updates=2,
                                  stop_reason="new reason"))["reason"] == \
        "new reason"
    # a read with no value is marked so whatever its solve reached, with the
    # short name of the reason its record names
    missing = _solved(record, converged=False, accepted_updates=5)
    missing.update(failed="the squared lengths overflowed the double (...), "
                          "so there is no finite geometry to read a pole on",
                   reason="the squared lengths overflowed the double")
    assert bp.solve_state(missing) == {
        "state": "no value", "reason": "lengths overflowed",
        "accepted_updates": 5}
    assert bp.solve_state({"content": [0, 3, 0], "failed": "band 1 has "
                           "rank 2", "doublet_reads": []}) == {
        "state": "no value", "reason": None, "accepted_updates": None}
    # a flagged read is a read: it is marked by its solve, as any other
    flagged = _solved(record, converged=True, accepted_updates=5)
    flagged["flags"] = [{"name": "not Kontsevich-Segal allowable",
                         "detail": "margin -0.613"}]
    assert bp.solve_state(flagged) == {
        "state": "converged", "reason": None, "accepted_updates": 5}


def test_callout_lines():
    assert bp.callout_lines({"state": "converged", "reason": None,
                             "accepted_updates": 1}) == [
        "\u2713 converged", "1 update"]
    assert bp.callout_lines({"state": "not converged", "reason": "no descent",
                             "accepted_updates": 12}) == [
        "\u2717 not converged", "no descent", "12 updates"]
    assert bp.callout_lines({"state": "no value", "reason": None,
                             "accepted_updates": None}) == ["\u2717 no value"]


def test_a_ratio_row_carries_the_solve_behind_each_pole():
    ratio = {"pole_ratio": 0.5 + 0j, "nucleon_pole": 1.0, "delta_pole": 2.0,
             "nucleon_content": [2, 1, 0],
             "nucleon_doublet_content": [1, 1, 1],
             "delta_content": [3, 0, 0], "delta_doublet_content": [0, 3, 0]}
    converged = {"state": "converged", "reason": None, "accepted_updates": 3}
    stalled = {"state": "not converged", "reason": "no descent",
               "accepted_updates": 9}
    row = bp.ratio_row("k=1 b=2", "quasi_free", ratio,
                       {(2, 1, 0): converged, (3, 0, 0): stalled})
    assert row["nucleon"]["solve"] == converged
    assert row["delta"]["solve"] == stalled
    assert bp.unconverged_poles(row) == ["D"]
    assert bp.solve_tag(row) == " [unconverged: D]"
    both = bp.ratio_row("k=1 b=2", "quasi_free", ratio,
                        {(2, 1, 0): converged, (3, 0, 0): converged})
    assert bp.unconverged_poles(both) == []
    assert bp.solve_tag(both) == " [both converged]"
    # without solve records the row is neither flagged nor tagged
    bare = bp.ratio_row("k=1 b=2", "quasi_free", ratio)
    assert bp.unconverged_poles(bare) == [] and bp.solve_tag(bare) == ""
    assert bp.solve_tag(bp.ratio_row("k=1 b=2", "quasi_free", None)) == ""


def test_the_drawn_frame_marks_each_content_by_its_solve(point):
    """A content whose solve did not converge has a red band and a callout
    naming why it stopped, its group label in the same ink, its ratios drawn
    hollow and tagged in the listing."""
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    point = dict(point)
    point["contents"] = [_solved(
        point["contents"][0], converged=False, accepted_updates=12,
        stop_reason="no move and no scaled step lowers the residual norm")]
    data = bp.frame_data([point], 0)
    assert data["groups"][0]["solve"] == {
        "state": "not converged", "reason": "no descent",
        "accepted_updates": 12}
    figure = plt.figure(figsize=bp.FIGURE_SIZE)
    try:
        bp.draw_frame(figure, [point], 0)
        quasi_free, quartic, ratio, pairs = figure.axes
        style = bp.SOLVE_STYLE["not converged"]
        for axis in (quasi_free, quartic):
            callouts = [t for t in axis.texts
                        if t.get_text().startswith("\u2717")]
            assert [t.get_text() for t in callouts] == [
                "\u2717 not converged\nno descent\n12 updates"]
            assert matplotlib.colors.same_color(callouts[0].get_color(),
                                                style["ink"])
            bands = [p for p in axis.patches
                     if matplotlib.colors.same_color(p.get_facecolor(),
                                                     style["band"])]
            assert len(bands) == 1
            assert "mean-field solve not converged" in [
                t.get_text() for t in axis.get_legend().get_texts()]
            assert matplotlib.colors.same_color(
                axis.get_xticklabels()[0].get_color(), style["ink"])
        hollow = [line for line in ratio.get_lines()
                  if line.get_markerfacecolor() == "none"
                  and len(line.get_xdata())]
        assert len(hollow) == 2
        assert "hollow: a pole's solve did not converge" in [
            t.get_text() for t in ratio.get_legend().get_texts()]
        listing = "\n".join(t.get_text() for t in pairs.texts)
        assert listing.count("[unconverged: N, D]") == 2
    finally:
        plt.close(figure)


def test_render_writes_an_empty_frame_for_an_empty_scan(tmp_path):
    path = tmp_path / "empty.png"
    bp.render({"points": []}, str(path))
    assert path.stat().st_size > 0


def test_jsonable_converts_complex_and_numpy_scalars():
    value = {1: [1 + 2j, np.float64(0.5), np.int64(3), np.complex128(2 - 1j),
                 (True, None, "x")]}
    out = bp._jsonable(value)
    assert out == {"1": [{"re": 1.0, "im": 2.0}, 0.5, 3,
                         {"re": 2.0, "im": -1.0}, [True, None, "x"]]}
    json.dumps(out)


def test_the_ratio_pair_text():
    assert bp.ratio_pair_text({"ratio": None}) == "no pole pair"
    row = bp.ratio_row("k=1 b=2", "quasi_free", {
        "pole_ratio": 0.5 + 0.25j, "nucleon_pole": 1.0, "delta_pole": 2.0,
        "nucleon_content": [2, 1, 0], "nucleon_doublet_content": [1, 1, 1],
        "delta_content": [3, 0, 0], "delta_doublet_content": [0, 3, 0]})
    assert bp.ratio_pair_text(row) == "N 210|111 / D 300|030"


# ------------------------------------------------------------ helpers


def test_sector_poles_of_an_invariant_sector():
    """On an invariant subspace the compression has no leakage and its poles
    are the operator's eigenvalues on that subspace, read exactly: the
    compressed block is diagonal, so its Schur diagonal is exact and the
    poles 1 and 2 are reported to rounding with multiplicity one, residue
    rank one, a zero subspace residual and the separation 1 between them."""
    operator = np.diag([1.0, 2.0, 5.0, 7.0]).astype(complex)
    sector = np.eye(4, dtype=complex)[:, :2]
    block, leakage, read = bp.sector_poles(operator, sector)
    np.testing.assert_allclose(block, np.diag([1.0, 2.0]), atol=1e-14)
    assert leakage < 1e-14
    np.testing.assert_allclose([complex(p) for p in read.poles], [1.0, 2.0],
                               atol=1e-14)
    assert list(read.multiplicity) == [1, 1]
    assert list(read.residue_rank) == [1, 1]
    assert all(residual < 1e-14 for residual in read.subspace_residual)
    assert list(read.separation) == pytest.approx([1.0, 1.0], abs=1e-14)
    assert list(read.failed_certificates) == []
    mixed = np.array([[1, 0], [1, 0], [0, 1], [0, 0]], dtype=complex)
    _, leakage, _ = bp.sector_poles(operator, mixed)
    assert leakage > 0.1


def test_second_quantization_on_three_particles():
    """dGamma(X) of a diagonal one-particle X on three particles is diagonal
    with the sums of the occupied entries; Lambda^3 of a map is its matrix of
    3 x 3 minors."""
    modes = 5
    basis = list(itertools.combinations(range(modes), 3))
    diagonal = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    lifted = bp.second_quantized(np.diag(diagonal).astype(complex), basis)
    np.testing.assert_allclose(
        lifted, np.diag([diagonal[list(b)].sum() for b in basis]), atol=1e-12)
    rng = np.random.default_rng(1)
    x = rng.normal(size=(modes, modes))
    power = bp.third_exterior_power(x, basis)
    assert power[3, 7] == pytest.approx(np.linalg.det(
        x[np.ix_(basis[3], basis[7])]))
    # functoriality: Lambda^3(XY) = Lambda^3(X) Lambda^3(Y)
    y = rng.normal(size=(modes, modes))
    np.testing.assert_allclose(bp.third_exterior_power(x @ y, basis),
                               power @ bp.third_exterior_power(y, basis),
                               atol=1e-10)


def test_the_permutation_sign_and_the_left_inverse():
    assert bp._permutation_sign([0, 1, 2]) == 1
    assert bp._permutation_sign([1, 0, 2]) == -1
    assert bp._permutation_sign([1, 2, 0]) == 1
    columns = np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 2.0]])
    np.testing.assert_allclose(bp.left_inverse(columns) @ columns, np.eye(2),
                               atol=1e-14)


def test_the_occupation_basis_and_the_fock_embedding():
    basis = bp.occupation_basis()
    assert len(basis) == 816 and basis[0] == (0, 1, 2)
    vector = np.zeros(len(basis), dtype=complex)
    vector[0] = 1.0
    fock = bp.to_fock(vector, basis)
    np.testing.assert_allclose(
        fock, np.asarray(obs.SharpSpin.determinant([0, 1, 2], 18)))


def test_the_pure_gauge_directions_leave_every_face_holonomy_unchanged():
    """One direction per vertex but the last of each sheet (nine), with no
    length component; a finite step along each leaves every face holonomy of
    the host where it was."""
    spacetime = bp.build_host()
    assert bp.gauge_directions(spacetime, False) is None
    directions = bp.gauge_directions(spacetime, True)
    assert directions.shape == (36, 9)
    assert np.linalg.matrix_rank(directions) == 9
    assert np.all(directions[:18] == 0)
    action = cob.JointAction(spacetime, bp.action_declaration(spacetime, 1, 1))
    before = np.asarray(action.face_holonomies())
    edges = spacetime.getEdgeList().toVector()
    for edge, step in zip(edges, 0.37 * directions[18:, 4]):
        edge.setPhase(complex(edge.getPhase()) + step)
    after = np.asarray(cob.JointAction(spacetime, bp.action_declaration(
        spacetime, 1, 1)).face_holonomies())
    np.testing.assert_allclose(after, before, atol=1e-12)


def test_a_limit_is_carried_only_when_the_user_declares_it():
    """The limits a user may declare on a solve (`LIMITS`: a number of
    iterations of the drive, a number of relaxation updates per iteration, a
    wall-clock time) are options of the command line that default to None,
    are recorded in the config, and reach the solve's drive only when
    given."""
    assert [key for key, _, _ in bp.LIMITS] == [
        "iteration_limit", "update_limit", "time_limit_seconds"]
    args = bp.build_parser().parse_args(["run"])
    assert bp.limits_from(args) == {
        "iteration_limit": None, "update_limit": None,
        "time_limit_seconds": None}
    args = bp.build_parser().parse_args(
        ["run", "--iteration-limit", "40", "--update-limit", "16",
         "--time-limit-seconds", "2.5"])
    limits = bp.limits_from(args)
    assert limits == {"iteration_limit": 40, "update_limit": 16,
                      "time_limit_seconds": 2.5}
    config = bp.default_config([1.0], [1.0], limits=limits)
    arguments = bp.solve_arguments(config)
    assert arguments["iteration_limit"] == 40
    assert arguments["update_limit"] == 16
    assert arguments["time_limit_seconds"] == 2.5
    with pytest.raises(ValueError, match="unknown limits"):
        bp.default_config([1.0], [1.0], limits={"newton_iterations": 3})


def test_the_villain_order_is_an_option_recorded_in_the_config():
    """M, the order the Villain weight of the holonomy term is summed to
    (`DECLARED_VILLAIN_ORDER`): an option of the command line, an integer
    from 1 to `cob.VillainCharacter.maximum_order` = 10 that defaults to 10,
    recorded in the config under ``villain_order`` and carried into the
    action's declaration. The config and the declaration refuse an order
    outside the range by name."""
    assert cob.VillainCharacter.maximum_order == 10
    assert bp.DECLARED_VILLAIN_ORDER == 10
    assert bp.build_parser().parse_args(["run"]).villain_order == 10
    args = bp.build_parser().parse_args(["run", "--villain-order", "4"])
    assert args.villain_order == 4 and isinstance(args.villain_order, int)
    assert bp.default_config([1.0], [1.0])["villain_order"] == 10
    config = bp.default_config([1.0], [1.0], villain_order=args.villain_order)
    assert config["villain_order"] == 4
    assert bp.declared_villain_order(config) == 4
    assert bp.declared_villain_order({}) == 10
    assert bp.declared_villain_order(None) == 10
    for order in (0, 11, -1, 2.5, True):
        with pytest.raises(ValueError, match="the order of the Villain "
                                             "weight is an integer from 1 "
                                             "to 10"):
            bp.default_config([1.0], [1.0], villain_order=order)
    spacetime = bp.build_host()
    assert bp.action_declaration(spacetime, 1.0, 1.0).villain_order == 10
    declaration = bp.action_declaration(spacetime, 1.0, 1.0, villain_order=4)
    assert declaration.villain_order == 4
    assert cob.JointAction(spacetime, declaration).holonomy_truncation() \
        .order == 4
    declaration.villain_order = 11
    with pytest.raises(ValueError, match="the order of the Villain weight "
                                         "is an integer from 1 to 10; got 11"):
        cob.JointAction(spacetime, declaration)
    # the geometric action of the fluctuation elimination and the carrier of
    # the fingerprint read are built at the config's order
    assert bp._geometric_action(spacetime, 1.0, 1.0, config) \
        .holonomy_truncation().order == 4


@pytest.mark.parametrize("text", ["0", "11", "-3", "2.5", "ten"])
def test_a_villain_order_outside_one_to_ten_is_refused_by_name(text, capsys):
    with pytest.raises(SystemExit) as stop:
        bp.build_parser().parse_args(["run", "--villain-order", text])
    assert stop.value.code == 2
    error = capsys.readouterr().err
    assert "--villain-order is an integer from 1 to 10" in error


def test_main_passes_the_villain_order_to_every_point(cheap):
    bp.main(["run", "--kappa", "1", "--beta", "1", "--villain-order", "7",
             "--quiet"])
    (config,) = cheap
    assert config["villain_order"] == 7


def test_the_isospin_doublet_driver_takes_the_villain_order():
    """`tessera.drivers.isospin_doublet` offers the same option, an integer
    from 1 to 10 that defaults to 10, and declares it on the action it reads
    the host's carrier from. The carrier operator h_1(z, U) does not depend
    on the holonomy term, so the declared host's carrier at order three is
    the one at order ten entry for entry."""
    from tessera.drivers import isospin_doublet

    parser = isospin_doublet.build_parser()
    assert parser.parse_args(["run"]).villain_order == 10
    assert parser.parse_args(["run", "--villain-order", "3"]) \
        .villain_order == 3
    with pytest.raises(SystemExit):
        parser.parse_args(["run", "--villain-order", "11"])
    ten = isospin_doublet.declared_carrier()
    three = isospin_doublet.declared_carrier(villain_order=3)
    assert ten.shape == (18, 18)
    assert np.array_equal(three, ten)
    with pytest.raises(ValueError, match="integer from 1 to 10"):
        isospin_doublet.drive(villain_order=0)


def test_the_fluctuation_couplings_count_and_shape():
    spacetime = bp.build_host()
    lengths = bp.fluctuation_couplings(spacetime, False)
    both = bp.fluctuation_couplings(spacetime, True)
    assert len(lengths) == 18 and len(both) == 36
    assert all(c.shape == (18, 18) for c in both)


def test_the_declarations_carry_the_config():
    config = bp.default_config([1.0], [1.0])
    config["held_sectors"] = []
    geometry = bp.relaxation_declaration(config)
    assert geometry.relax_lengths and geometry.relax_links
    assert not geometry.relax_multipliers
    # a solve ends when its drive ends; no limit is declared
    assert "newton_iterations" not in config
    assert "mean_field_iterations" not in config
    for key, _, _ in bp.LIMITS:
        assert config[key] is None
    assert bp.solve_arguments(config)["tolerance"] == \
        bp.DECLARED_TOLERANCE == 1e-15
    assert geometry.rank_tolerance == bp.DECLARED_TOLERANCE
    assert "jacobian_radius" not in config
    mean_field = bp.mean_field_declaration((2, 1, 0), config)
    assert mean_field.covariance_rule == cob.CovarianceRule.BandFilling
    assert list(mean_field.band_occupations) == [2.0, 1.0, 0.0]
    assert mean_field.occupation_order == \
        cob.OccupationOrder.AscendingRealPart
    assert mean_field.band_tolerance == bp.DECLARED_BAND_TOLERANCE
    spacetime = bp.build_host()
    declaration = bp.action_declaration(spacetime, 2.0, 3.0,
                                        regge_hinges="all")
    assert declaration.regge_hinges == cob.ReggeHinges.All
    assert declaration.gravitational_weight == 0.5
    assert declaration.holonomy_weight == 3.0
    assert declaration.regge_form == cob.ReggeForm.Primal
    assert declaration.matter_weight == 1.0
    assert declaration.villain_order == bp.DECLARED_VILLAIN_ORDER == 10
    # kappa = 8 pi G enters through the Regge weight alone, and the record
    # says so
    assert config["fiber_moments"] == "r"
    assert config["kappa_role"].startswith(
        "kappa = 8 pi G enters only through the Regge weight 1/kappa")


def test_the_host_built_from_a_cell_carries_its_fields():
    links = list(np.exp(1j * np.array([0.1, -0.2, 0.3, 0.4, -0.5, 0.6])))
    lengths = [8.0, 7.0, 6.0, 5.0, 4.0, 3.0]
    spacetime = bp.build_host(cell={"squared_lengths": lengths,
                                    "links": links})
    for sheet in range(bp.SHEETS):
        np.testing.assert_allclose(bp.sheet_links(spacetime, sheet), links,
                                   atol=1e-14)
        np.testing.assert_allclose(bp.sheet_squared_lengths(spacetime, sheet),
                                   lengths, atol=1e-12)
    assert bp.canonical_edges()[:3] == [(0, 1), (0, 2), (0, 3)]
    assert len(bp.edge_records(spacetime)) == 18


def test_the_recursion_read_on_the_declared_host():
    """One turn of the level recursion on the declared host: one component per
    sheet, each fiber of rank six, every band accepted."""
    recursion = bp.recursion_read(bp.build_host(),
                                  bp.default_config([1.0], [1.0]))
    assert sorted(sorted(p) for p in recursion["partition"]) == [
        list(range(6)), list(range(6, 12)), list(range(12, 18))]
    assert recursion["band_ranks"] == [6, 6, 6]
    assert all(recursion["bands_accepted"])
    assert recursion["levels"] == 1 and "refusal" not in recursion


def test_a_recursion_that_takes_no_turn_leaves_condition_1_not_evaluable(
        monkeypatch):
    """The level recursion either takes its turn or refuses by name; with
    the dense crossover declared below the host's eighteen edge modes it
    refuses as it is built. The recursion read records the refusal, with no
    completed turn and no band, instead of failing the content, and the
    quark verdict reads every piece of evidence taken from the completed
    turn (condition 1's four measured pieces and condition 5's
    base-transport leakage) as not evaluable, with the refusal as the
    reason; the other conditions are read as before."""
    declared = cob.LevelRecursionDeclaration

    def crowded():
        declaration = declared()
        declaration.dense_crossover = 1
        return declaration

    monkeypatch.setattr(cob, "LevelRecursionDeclaration", crowded)
    spacetime = bp.build_host()
    recursion = bp.recursion_read(spacetime, bp.default_config([1.0], [1.0]))
    assert recursion["levels"] == 0
    assert "dense crossover" in recursion["refusal"]
    assert "bands_accepted" not in recursion
    alignment = bp.aligned_doublet_frame(bp.monopole_support(),
                                         bp.rotation_group())
    verdict = bp.quark_conditions(spacetime, [alignment] * bp.SHEETS,
                                  recursion, 0.0, None)
    assert not verdict["certified"]
    one, two, _, _, five = verdict["conditions"][:5]
    assert one["status"] == "NotEvaluable"
    reason = "the recursion took no turn: " + recursion["refusal"]
    for evidence in one["evidence"]:
        assert evidence["held"] is None
    assert [e["detail"] for e in one["evidence"]
            if e["name"] in ("persistent-support", "localized-projector-rank",
                             "contour-separation", "external-leakage")] == \
        [reason] * 4
    leakage = next(e for e in five["evidence"]
                   if e["name"] == "base-transport-leakage")
    assert leakage["held"] is None and leakage["detail"] == reason
    assert two["status"] != "NotEvaluable"


@pytest.fixture(scope="module")
def declared_verdict():
    spacetime = bp.build_host()
    alignment = bp.aligned_doublet_frame(bp.monopole_support(),
                                         bp.rotation_group())
    recursion = bp.recursion_read(spacetime, bp.default_config([1.0], [1.0]))
    return bp.quark_conditions(spacetime, [alignment] * bp.SHEETS,
                               recursion, 0.0, None)


def test_the_quark_verdict_names_the_seven_conditions(declared_verdict):
    assert [c["number"] for c in declared_verdict["conditions"]] == \
        list(range(1, 8))
    assert [c["name"] for c in declared_verdict["conditions"]] == \
        obs.QuarkConditions.condition_names()
    status = {c["name"]: c["status"] for c in declared_verdict["conditions"]}
    assert status["color-spin-fiber"] == "Passed"
    # the verdict is read on the declared host with no solved state, so the
    # occupation parity has no covariance to be measured on
    for name in ("odd-occupation", "anchor-atlas", "lineage", "fingerprint"):
        assert status[name] == "NotEvaluable"
    assert declared_verdict["certified"] is False


def test_disjoint_sheets_have_no_inter_component_leakage(declared_verdict):
    evidence = {e["name"]: e["held"]
                for c in declared_verdict["conditions"]
                for e in c["evidence"]}
    assert evidence["external-leakage"] is True
    assert evidence["base-transport-leakage"] is True


def test_every_argument_has_a_help_string_that_renders():
    parser = bp.build_parser()
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for sub in action.choices.values():
                assert sub.format_help()
