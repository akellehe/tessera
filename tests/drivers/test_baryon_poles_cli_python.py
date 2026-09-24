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


def _record(content, poles):
    """A content record with the given lowest pole for each spin, carrying
    the restriction to 2T of a triality-zero sector."""
    sectors = {}
    for key, value in poles.items():
        irreps = bp.restriction(float(key), 0)
        entry = {"lowest_pole": value}
        sectors[key] = {"quasi_free": entry, "with_quartic": entry,
                        "restriction_to_2T": irreps,
                        "nucleon_reading": "2" in irreps,
                        "delta_reading": sorted(irreps) == ["2'", "2''"]}
    return {"content": list(content), "sectors": sectors}


def _cheap_scan_point(kappa, beta, config, alignment, on_content=None):
    """A deterministic stand-in for one scan point with both spins present,
    so every pairing of `ratios` has a pole pair."""
    records = [_record([1, 1, 1], {HALF: complex(kappa, 0.1),
                                   THREE: complex(kappa + beta, 0.2)})]
    return {"kappa": kappa, "beta": beta, "holonomy": config["holonomy"],
            "elimination": config["elimination"], "failed_contents": [],
            "contents": records, "ratios": bp.ratios(records),
            "pole_table": bp.pole_table(records)}


@pytest.fixture
def cheap(monkeypatch):
    """Replace the scan point and record the configuration it is given."""
    seen = []

    def scan(kappa, beta, config, alignment, on_content=None):
        seen.append(dict(config))
        return _cheap_scan_point(kappa, beta, config, alignment)

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


def test_lists_of_couplings_are_parsed():
    args = bp.build_parser().parse_args(
        ["run", "--kappa", "0.25", "4", "--beta", "5", "--edge-squared", "2",
         "--regge-hinges", "all"])
    assert args.kappa == [0.25, 4.0] and args.beta == [5.0]
    assert args.edge_squared == 2.0 and args.regge_hinges == "all"


@pytest.mark.parametrize("argv,name", [
    (["run", "--holonomy", "plaquette"], "--holonomy"),
    (["run", "--eliminate", "phases"], "--eliminate"),
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
    assert document["config"]["holonomy"] == "villain"
    assert document["config"]["target_mass_squared_ratio"] == pytest.approx(
        (938.272 / 1232.0) ** 2)
    # complex numbers are written as {"re", "im"}
    pole = document["points"][0]["ratios"]["quasi_free"]["by_spin"][
        "nucleon_pole"]
    assert pole == {"re": 0.5, "im": 0.1}
    assert set(document["host"]) == {"monopole", "averaged_eigenvalues",
                                      "reference_carrier",
                                      "intertwining_residual"}
    lines = (tmp_path / "poles.points.jsonl").read_text().splitlines()
    assert len(lines) == 3
    first = json.loads(lines[0])
    assert set(first) == {"config", "host"}
    assert first["host"]["monopole"]["monopole_number"] == 1
    assert [json.loads(line)["kappa"] for line in lines[1:]] == [0.5, 1.0]


def test_main_passes_the_declared_options_to_every_point(cheap):
    bp.main(["run", "--kappa", "1", "--beta", "1", "--holonomy", "wilson",
             "--eliminate", "lengths", "--edge-squared", "3",
             "--isospin-doublet", "--quiet"])
    (config,) = cheap
    assert config["holonomy"] == "wilson"
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
    bp.main(["run", "--kappa", "1", "--beta", "2"])
    out = capsys.readouterr().out
    assert "kappa=1 beta=2  quasi-free s_N/s_D" in out
    assert out.count("s_N/s_D") >= 4
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

    def live(config, progress=False, points_file=None):
        calls.append((progress, points_file))
        return bp.drive(config, progress=progress, points_file=points_file)

    monkeypatch.setattr(bp, "drive_live", live)
    path = tmp_path / "live.json"
    bp.main(["run", "--kappa", "1", "--beta", "1", "--live", "--quiet",
             "--json", str(path)])
    assert calls == [(False, str(tmp_path / "live.points.jsonl"))]
    assert path.exists()


def test_a_stopped_drive_says_so():
    result = bp.drive(bp.default_config([1.0], [1.0]),
                      stop_requested=lambda: True)
    assert result["stopped"] is True and result["points"] == []


# ------------------------------------------------------------ the outputs


def test_the_summary_names_every_pairing():
    point = _cheap_scan_point(1.0, 2.0, bp.default_config([1.0], [2.0]),
                              None)
    text = bp.summary({"points": [point]})
    lines = text.splitlines()
    assert len(lines) == 1 + 4
    assert "by_spin" in lines[1] and "by_2T_reading" in lines[2]
    assert "Delta restriction 2'+2''" in lines[1]
    empty = bp.summary({"points": [dict(point, ratios={
        name: {"by_spin": None, "by_2T_reading": None}
        for name in ("quasi_free", "with_quartic")})]})
    assert empty.count("no pole pair") == 4


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


def test_the_ratio_text():
    assert bp._ratio_text(None) == "no pole pair"
    ratio = {"pole_ratio": 0.5 + 0.25j}
    assert bp._ratio_text(ratio) == "s_N/s_D = 0.5+0.25i (target 0.5800)"


# ------------------------------------------------------------ helpers


def test_the_contour_encloses_every_eigenvalue():
    rng = np.random.default_rng(3)
    for _ in range(5):
        block = rng.normal(size=(5, 5)) + 1j * rng.normal(size=(5, 5))
        centre, radius = bp.contour_of(block)
        assert centre == pytest.approx(np.trace(block) / 5)
        assert np.all(np.abs(np.linalg.eigvals(block) - centre) < radius)
    centre, radius = bp.contour_of(np.eye(3) * 2.0)
    assert centre == 2.0 and radius == pytest.approx(2e-3)


def test_sector_poles_of_an_invariant_sector():
    """On an invariant subspace the compression has no leakage and its poles
    are the operator's eigenvalues on that subspace."""
    operator = np.diag([1.0, 2.0, 5.0, 7.0]).astype(complex)
    sector = np.eye(4, dtype=complex)[:, :2]
    block, leakage, read = bp.sector_poles(operator, sector)
    np.testing.assert_allclose(block, np.diag([1.0, 2.0]), atol=1e-14)
    assert leakage < 1e-14
    np.testing.assert_allclose(sorted(complex(p).real for p in read.poles),
                               [1.0, 2.0], atol=1e-8)
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
    assert geometry.maximum_iterations == 40
    assert geometry.tolerance == 1e-11
    assert geometry.holonomy_zero_margin == bp.DECLARED_HOLONOMY_ZERO_MARGIN
    assert geometry.jacobian_mode == \
        cob.HolomorphicJacobianMode.RealAxisDifference
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
    assert declaration.stiffness_weight == 0.5
    assert declaration.holonomy_weight == 3.0
    assert declaration.regge_form == cob.ReggeForm.Primal


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


@pytest.fixture(scope="module")
def declared_verdict():
    spacetime = bp.build_host()
    alignment = bp.aligned_doublet_frame(bp.monopole_support(),
                                         bp.rotation_group())
    recursion = bp.recursion_read(spacetime, bp.default_config([1.0], [1.0]))
    return bp.quark_conditions(spacetime, alignment, recursion, 0.0, None)


def test_the_quark_verdict_names_the_seven_conditions(declared_verdict):
    assert [c["number"] for c in declared_verdict["conditions"]] == \
        list(range(1, 8))
    assert [c["name"] for c in declared_verdict["conditions"]] == \
        obs.QuarkConditions.condition_names()
    status = {c["name"]: c["status"] for c in declared_verdict["conditions"]}
    assert status["color-spin-fiber"] == "Passed"
    assert status["odd-occupation"] == "Passed"
    for name in ("anchor-atlas", "lineage", "fingerprint"):
        assert status[name] == "NotEvaluable"
    assert declared_verdict["certified"] is False


@pytest.mark.xfail(strict=True, reason=(
    "baryon_poles.recursion_read reports the norms of every LevelTransport "
    "of the level, including the diagonal blocks M_vv of each component with "
    "itself (norm 37.05 on the declared host), and quark_conditions grades "
    "'external-leakage' and 'base-transport-leakage' on those norms, so on "
    "three disjoint sheets, where no component couples to another, both "
    "certificates fail"))
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
