# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""The nucleon-to-Delta synthesis driver: every certificate it reports, on
the three-sheeted unit-monopole tetrahedron, and its --live path.

The live path is exercised with matplotlib stubbed, as the emergence driver's
live tests do: a GUI cannot be opened in a test process, and the claims under
test are the refusal of a file-only or WebAgg backend and the identity of the
outputs with and without the flag.
"""

import json
import time

import numpy as np
import pytest

from tessera import cobordism as cob
from tessera import observables as obs
from tessera.drivers import baryon_poles as bp


@pytest.fixture(scope="module")
def alignment():
    return bp.aligned_doublet_frame(bp.monopole_support(),
                                    bp.rotation_group())


# ------------------------------------------------------------------ host


def test_the_host_carries_the_unit_monopole_on_every_sheet():
    spacetime = bp.build_host()
    for sheet in range(bp.SHEETS):
        support, departure = bp.sheet_support(spacetime, sheet)
        read = support.monopoleNumber()
        assert read.monopole_number == 1 and read.odd and read.bundle
        assert departure < 1e-14
    faces = cob.JointAction(
        spacetime, bp.action_declaration(spacetime, 1.0, 1.0)).face_holonomies()
    # every face holonomy is a primitive fourth root of unity, +-i
    assert np.max(np.abs(np.abs(np.asarray(faces)) - 1.0)) < 1e-14
    assert np.max(np.abs(np.asarray(faces) ** 2 + 1.0)) < 1e-12


def test_the_sheets_are_isomorphic():
    spacetime = bp.build_host()
    read = obs.SheetedSupport(3, 6).certifyIsomorphism(
        [np.array(bp.sheet_squared_lengths(spacetime, t)) for t in range(3)],
        [np.array(bp.sheet_links(spacetime, t)) for t in range(3)], 1e-12)
    assert read.isomorphic


def test_the_primal_regge_term_is_empty_on_the_host():
    spacetime = bp.build_host()
    action = cob.JointAction(spacetime,
                             bp.action_declaration(spacetime, 1.0, 1.0))
    assert action.regge_hinge_count() == 0
    assert action.regge_term() == 0


def test_the_monopole_spin_read(alignment):
    read = alignment["spin_read"]
    assert read.monopole.monopole_number == 1
    assert read.cocycle.nontrivial
    assert abs(read.cocycle.commutator_phase + 1.0) < 1e-12
    assert read.half_integer_doublet
    values = alignment["averaged_eigenvalues"]
    expected = [4 - 2 / np.sqrt(3)] * 2 + [4.0] * 2 + [4 + 2 / np.sqrt(3)] * 2
    assert np.allclose(values, expected, atol=1e-12)
    # the genuine j = 1/2 doublet is the one at 4
    assert alignment["reference_carrier"] == 1


def test_the_doublets_are_aligned_to_one_su2_action(alignment):
    assert alignment["intertwining_residual"] < 1e-10
    frame = alignment["frame"]
    assert np.linalg.cond(frame) < 1e6


def test_the_one_per_doublet_sector_is_two_halves_and_a_three_halves():
    states, _ = bp.singlet_states((1, 1, 1))
    sectors, values = bp.spin_sectors(states)
    assert sectors[bp.SPIN_HALF].shape[1] == 4
    assert sectors[bp.SPIN_THREE_HALVES].shape[1] == 4


def test_contents_carry_the_spins_they_can():
    for content, half, three in (((3, 0, 0), 0, 4), ((2, 1, 0), 2, 4),
                                 ((1, 1, 1), 4, 4)):
        states, _ = bp.singlet_states(content)
        sectors, _ = bp.spin_sectors(states)
        assert sectors.get(bp.SPIN_HALF, np.zeros((0, 0))).shape[-1] == half
        assert sectors[bp.SPIN_THREE_HALVES].shape[1] == three


def test_every_singlet_state_is_a_colour_singlet():
    states, _ = bp.singlet_states((2, 1, 0))
    basis = bp.occupation_basis()
    for k in range(states.shape[1]):
        fock = bp.to_fock(states[:, k], basis)
        residual = np.linalg.norm(bp.colour_casimir(fock)) / \
            np.linalg.norm(fock)
        assert residual < 1e-12


def test_a_colour_nonsinglet_is_caught():
    """Three quarks on one sheet are not a colour singlet."""
    fock = np.asarray(obs.SharpSpin.determinant([0, 3, 6], 18))
    assert np.linalg.norm(bp.colour_casimir(fock)) > 0.5


def test_the_covariant_operator_at_the_monopole_is_not_rotation_symmetric():
    """A measured property of the host, recorded because the calculation
    rests on it: the Whitney covariant operator h_1(z, U) of the specification
    (Definition 2, base-vertex twist b = min) at the unit-monopole connection
    has six simple eigenvalues per sheet and does not commute with the
    projective rotation action D_1(g); the three spinor doublets appear only in
    the rotation-averaged operator (WP line 497, "the T-averaged twisted edge
    Laplacian")."""
    spacetime = bp.build_host()
    action = cob.JointAction(spacetime,
                             bp.action_declaration(spacetime, 1.0, 1.0))
    h = bp.matrix(action.carrier_operator())[:6, :6]
    values = np.sort(np.linalg.eigvals(h).real)
    assert np.min(np.diff(values)) > 0.1
    support = bp.monopole_support()
    worst = max(np.linalg.norm(np.asarray(support.edgeRepresentation(g)) @ h
                               - h @ np.asarray(support.edgeRepresentation(g)))
                for g in bp.rotation_group())
    assert worst / np.linalg.norm(h) > 0.1


# ------------------------------------------------------------------ ratios


def test_ratios_are_taken_on_the_lowest_poles():
    def record(content, sectors):
        out = {}
        for key, (value, irreps) in sectors.items():
            entry = {"lowest_pole": value}
            out[key] = {"quasi_free": entry, "with_quartic": entry,
                        "restriction_to_2T": irreps,
                        "nucleon_reading": "2" in irreps,
                        "delta_reading": sorted(irreps) == ["2'", "2''"]}
        return {"content": content, "sectors": out}
    half, three = str(bp.SPIN_HALF), str(bp.SPIN_THREE_HALVES)
    out = bp.ratios([
        record([3, 0, 0], {three: (9.0 + 0j, ["2'", "2''"])}),
        record([2, 1, 0], {half: (10.0 + 0j, ["2'"]),
                           three: (10.0 + 0j, ["2''", "2"])}),
        record([1, 1, 1], {half: (12.0 + 0j, ["2"]),
                           three: (12.0 + 0j, ["2'", "2''"])})])
    by_spin = out["quasi_free"]["by_spin"]
    assert by_spin["nucleon_content"] == [2, 1, 0]
    assert by_spin["delta_content"] == [3, 0, 0]
    assert by_spin["pole_ratio"] == pytest.approx(10.0 / 9.0)
    assert by_spin["delta_is_a_delta_reading"]
    assert not by_spin["delta_ambiguous_with_spin_half"]
    assert by_spin["target_mass_squared_ratio"] == pytest.approx(
        (938.272 / 1232.0) ** 2)
    by_reading = out["quasi_free"]["by_2T_reading"]
    # the lowest pole with a 2 in its restriction is the spin-3/2 sector of
    # (2, 1, 0): the tetrahedral ambiguity in action
    assert by_reading["nucleon_content"] == [2, 1, 0]
    assert by_reading["nucleon_spin_j_j_plus_1"] == pytest.approx(3.75)
    assert by_reading["delta_content"] == [3, 0, 0]


def test_the_restriction_to_the_binary_tetrahedral_group():
    assert bp.restriction(bp.SPIN_HALF, 0) == ["2"]
    assert bp.restriction(bp.SPIN_HALF, 1) == ["2'"]
    assert bp.restriction(bp.SPIN_THREE_HALVES, 0) == ["2'", "2''"]
    assert bp.restriction(bp.SPIN_THREE_HALVES, 1) == ["2''", "2"]


def test_the_T_averaged_operator_carries_three_doublets(alignment):
    """h-bar_1 = (1/|T|) sum_g D_1(g)^-1 h_1 D_1(g) (WP line 497) is block
    scalar in the aligned doublet frame; applied to the library's
    combinatorial twisted edge Laplacian it gives the paper's six
    eigenvalues."""
    support = bp.monopole_support()
    actions = bp.rotation_action([support] * bp.SHEETS)
    spacetime = bp.build_host()
    h = bp.matrix(cob.JointAction(
        spacetime, bp.action_declaration(spacetime, 1.0, 1.0)
    ).carrier_operator())
    averaged = bp.rotation_averaged(h, actions)
    frame = bp._micro_frame([alignment] * bp.SHEETS)
    energies, residual = bp._block_scalar(np.linalg.solve(frame,
                                                          averaged @ frame))
    assert residual < 1e-12
    assert len({round(e.real, 9) for e in energies}) == 3
    combinatorial = np.kron(np.eye(bp.SHEETS),
                            np.asarray(support.edgeLaplacian()))
    values = np.sort(np.linalg.eigvals(
        bp.rotation_averaged(combinatorial, actions)).real)[::3]
    expected = [4 - 2 / np.sqrt(3)] * 2 + [4.0] * 2 + [4 + 2 / np.sqrt(3)] * 2
    assert np.allclose(values, expected, atol=1e-12)


def test_the_trialities_are_the_three_z3_characters(alignment):
    assert sorted(alignment["trialities"]) == [0, 1, 2]
    assert alignment["trialities"][alignment["reference_carrier"]] == 0


# -------------------------------------------------------------------- live


class _Canvas:
    """Enough of a figure canvas for the live loop: repaints are counted, the
    event loop sleeps briefly, and the close callback can be fired."""

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
    monkeypatch.setattr(plt, "pause", lambda interval: time.sleep(0.001))


@pytest.mark.parametrize("backend", ["agg", "webagg"])
def test_live_refuses_a_non_interactive_or_webagg_backend(monkeypatch,
                                                          backend):
    _stub_matplotlib(monkeypatch, backend)
    with pytest.raises(RuntimeError, match="--live needs an interactive"):
        bp.drive_live(bp.default_config([1.0], [1.0],
                                        selected_contents=[(3, 0, 0)]))


def _cheap_scan_point(kappa, beta, config, alignment, on_content=None):
    """A deterministic stand-in for one scan point, so the live path is tested
    without the relaxation's cost; the claim under test is that the live
    worker runs the same `drive` and returns the same result."""
    record = {"content": [3, 0, 0], "seconds": 0.0,
              "sectors": {str(bp.SPIN_THREE_HALVES): {
                  "quasi_free": {"lowest_pole": complex(kappa, beta)},
                  "with_quartic": {"lowest_pole": complex(beta, kappa)}}}}
    return {"kappa": kappa, "beta": beta, "contents": [record],
            "ratios": bp.ratios([record])}


def test_live_and_headless_outputs_are_identical(monkeypatch):
    monkeypatch.setattr(bp, "scan_point", _cheap_scan_point)
    config = bp.default_config([1.0, 2.0], [0.5], selected_contents=[(3, 0, 0)])
    headless = bp.drive(dict(config))
    _stub_matplotlib(monkeypatch)
    drawn = []
    monkeypatch.setattr(bp, "draw_frame",
                        lambda figure, frames, index: drawn.append(index))
    live = bp.drive_live(dict(config))
    assert drawn == [0, 1]
    assert json.dumps(bp._jsonable(live), sort_keys=True) == \
        json.dumps(bp._jsonable(headless), sort_keys=True)


def test_a_worker_error_reaches_the_main_thread(monkeypatch):
    def exploding(*args, **kwargs):
        raise ValueError("the scan point failed")
    monkeypatch.setattr(bp, "scan_point", exploding)
    _stub_matplotlib(monkeypatch)
    with pytest.raises(ValueError, match="the scan point failed"):
        bp.drive_live(bp.default_config([1.0], [1.0]))


def test_closing_the_window_switches_the_run_to_headless(monkeypatch,
                                                         tmp_path, capsys):
    """The window is closed after the first frame: the scan still runs to the
    end, the result equals the headless one, nothing more is drawn, every
    point reaches the JSON-lines file, and stdout says so."""
    monkeypatch.setattr(bp, "scan_point", _cheap_scan_point)
    config = bp.default_config([1.0, 2.0, 3.0], [0.5],
                               selected_contents=[(3, 0, 0)])
    headless = bp.drive(dict(config))
    figures = []
    _stub_matplotlib(monkeypatch, figures=figures)
    drawn = []

    def draw_then_close(figure, frames, index):
        drawn.append(index)
        figure.canvas.close()

    monkeypatch.setattr(bp, "draw_frame", draw_then_close)
    points = tmp_path / "run.points.jsonl"
    live = bp.drive_live(dict(config), points_file=str(points))
    assert drawn == [0]
    assert len(live["points"]) == 3 and not live["stopped"]
    assert json.dumps(bp._jsonable(live), sort_keys=True) == \
        json.dumps(bp._jsonable(headless), sort_keys=True)
    assert "continues headless" in capsys.readouterr().out
    lines = points.read_text().splitlines()
    assert len(lines) == 4
    assert [json.loads(line)["kappa"] for line in lines[1:]] == [1.0, 2.0, 3.0]


def test_each_point_is_written_as_it_completes(monkeypatch, tmp_path):
    """A scan stopped after two points leaves both on disk."""
    monkeypatch.setattr(bp, "scan_point", _cheap_scan_point)
    config = bp.default_config([1.0, 2.0, 3.0], [0.5],
                               selected_contents=[(3, 0, 0)])
    points = tmp_path / "run.points.jsonl"
    seen = []

    def stop_after_two():
        return len(seen) >= 2

    bp.drive(dict(config), on_frame=lambda frames, index: seen.append(index),
             stop_requested=stop_after_two, points_file=str(points))
    lines = points.read_text().splitlines()
    assert "config" in json.loads(lines[0])
    assert [json.loads(line)["kappa"] for line in lines[1:]] == [1.0, 2.0]
    assert bp.points_path("out/run.json") == "out/run.points.jsonl"


def test_the_pole_table_names_the_content_of_each_lowest_pole():
    half, three = str(bp.SPIN_HALF), str(bp.SPIN_THREE_HALVES)

    def record(content, poles):
        return {"content": content, "sectors": {
            key: {"quasi_free": {"lowest_pole": value},
                  "with_quartic": {"lowest_pole": value},
                  "restriction_to_2T": ["2"]}
            for key, value in poles.items()}}

    table = bp.pole_table([record([2, 1, 0], {half: 5.0 + 0j, three: 5.0 + 0j}),
                           record([3, 0, 0], {three: 4.0 + 0j})])
    assert [row["content"] for row in table["quasi_free"][three]] == \
        [[3, 0, 0], [2, 1, 0]]
    assert table["with_quartic"][half][0]["content"] == [2, 1, 0]
