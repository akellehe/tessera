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

from tests.drivers import _recursion_run_2026_09_23 as RUN


@pytest.fixture(scope="module")
def alignment():
    # at the run's tolerances: the declared 1e-15 degeneracy tolerance splits
    # the degenerate pairs of the averaged operator, whose eigenvalues agree
    # to rounding (a few 1e-15)
    return bp.aligned_doublet_frame(
        bp.monopole_support(), bp.rotation_group(),
        RUN.TOLERANCES["degeneracy_tolerance"],
        RUN.TOLERANCES["certificate_tolerance"])


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


def test_the_declared_host_has_the_tetrahedral_rotation_group():
    """`cell_symmetry` on the declared host: every one of the twelve rotations
    leaves the six equal squared lengths invariant and carries the symmetric
    monopole connection to a gauge-equivalent one on every sheet, with a
    compensation residual at rounding, so the cell's group is the whole
    tetrahedral group at the run's certificate tolerance and at the declared
    1e-15 alike."""
    spacetime = bp.build_host()
    supports = [bp.sheet_support(spacetime, t) for t in range(bp.SHEETS)]
    for tolerance in (RUN.TOLERANCES["certificate_tolerance"],
                      bp.DECLARED_TOLERANCE):
        symmetry = bp.cell_symmetry(spacetime, supports, tolerance)
        assert symmetry["tetrahedral"] and symmetry["order"] == 12
        assert symmetry["tolerance"] == tolerance
        assert symmetry["length_departure"] == 0.0
        assert symmetry["compensation_residual"] < 1e-15
        assert [r["rotation"] for r in symmetry["rotations"]] == \
            [list(g) for g in bp.rotation_group()]
        assert symmetry["group"][0] == [0, 1, 2, 3]


def test_a_cell_with_one_length_changed_keeps_the_rotations_fixing_that_edge():
    """With the squared length of the edge (0, 2) alone moved from 8 to 8.5
    on every sheet, the rotations that are still symmetries are the identity
    and the half turn about the axis through the midpoints of (0, 2) and
    (1, 3), which exchanges 0 with 2 and 1 with 3; the other ten carry that
    edge onto an edge of squared length 8, a departure of 0.5 / 8.5 of the
    largest squared length, and the cell's group is not tetrahedral."""
    host = bp.build_host()
    cell = {"squared_lengths": [8.0, 8.5, 8.0, 8.0, 8.0, 8.0],
            "links": [complex(u) for u in bp.sheet_links(host, 0)]}
    spacetime = bp.build_host(cell=cell)
    supports = [bp.sheet_support(spacetime, t) for t in range(bp.SHEETS)]
    symmetry = bp.cell_symmetry(spacetime, supports,
                                RUN.TOLERANCES["certificate_tolerance"])
    assert not symmetry["tetrahedral"] and symmetry["order"] == 2
    assert symmetry["group"] == [[0, 1, 2, 3], [2, 3, 0, 1]]
    assert symmetry["length_departure"] == pytest.approx(0.5 / 8.5)
    moved = [r for r in symmetry["rotations"] if not r["symmetric"]]
    assert len(moved) == 10
    assert all(r["length_departure"] == pytest.approx(0.5 / 8.5)
               for r in moved)
    assert all(r["compensation_residual"] < 1e-15
               for r in symmetry["rotations"])


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


def test_the_host_connection_is_stiff_under_the_villain_term():
    """Every face of the host carries F = +-i. The Wilson form
    beta (1 - cos Theta), the reference computed here from the host's own
    face holonomies, has curvature beta cos Theta = 0 there and would leave
    all 18 phase directions flat; the Villain curvature there is
    beta_V (<m^2>_i - <m>_i^2) = kappa(i) per face, so the nine coexact
    directions have stiffness 4 kappa(i) and the nine pure-gauge ones
    zero."""
    beta = 1.0
    spacetime = bp.build_host()
    action = cob.JointAction(spacetime,
                             bp.action_declaration(spacetime, 1.0, beta))
    faces = np.asarray(action.face_holonomies())
    wilson = beta * np.cos(np.angle(faces))
    assert np.max(np.abs(wilson)) < 1e-12
    villain = np.array(action.holonomy_hessian()).reshape(18, 18)
    kappa = -cob.VillainCharacter(beta).second_derivative(1j).real
    assert kappa == pytest.approx(0.9979584724666767, abs=1e-12)
    values = np.sort(np.linalg.eigvalsh(-villain.real))
    assert np.max(np.abs(values[:9])) < 1e-11
    assert np.allclose(values[9:], 4.0 * kappa, atol=1e-11)


# --------------------------------------------- the Section 7 elimination


def _host_problem(beta, elimination="lengths-and-phases",
                  regge_hinges="interior"):
    """The unrelaxed host carrying the three lowest modes of h_1, and the
    elimination of its fluctuations at kappa = 0.5 under the declared
    action."""
    spacetime = bp.build_host()
    config = bp.default_config([0.5], [beta], regge_hinges=regge_hinges,
                               elimination=elimination)
    bare = cob.JointAction(spacetime, bp.action_declaration(
        spacetime, 0.5, beta, regge_hinges))
    declaration = bp.action_declaration(spacetime, 0.5, beta, regge_hinges)
    declaration.covariance = bare.occupation_projector(3)
    action = cob.JointAction(spacetime, declaration)
    carrier = bp.matrix(action.carrier_operator())
    return spacetime, action, carrier, bp.eliminate_fluctuations(
        spacetime, action, carrier, 0.5, beta, config)


@pytest.fixture(scope="module")
def host_problem():
    return _host_problem(1.0)


def test_the_drazin_inverse_integrates_out_exactly_the_coexact_phases(
        host_problem):
    """The declared action has no length stiffness on the lone tetrahedron:
    its Regge term is zero by structure (no interior hinge) and the holonomy
    term does not see the lengths. With the Villain term the coexact phase
    block is nonsingular at the host's F = +-i, so the null space of A is the
    eighteen lengths and the nine pure-gauge phases (twelve vertices less one
    per sheet), the record says the null space is not pure gauge, A^D is the
    Drazin inverse on the nine coexact phases, and the reduced coordinates
    rebuild it exactly."""
    _, _, _, problem = host_problem
    drazin = problem["record"]["drazin"]
    assert drazin["coordinates"] == 36
    assert drazin["null_dimension"] == 27
    assert drazin["eliminated_dimension"] == 9
    assert drazin["gauge_dimension"] == 9
    assert not drazin["null_space_is_pure_gauge"]
    assert drazin["gauge_projector_residual"] < 1e-12
    assert drazin["projector_idempotency"] < 1e-12
    assert drazin["drazin_identity_residual"] < 1e-12
    assert drazin["reduction_residual"] < 1e-12
    # the geometric action has no length-phase coupling with the matter term
    # off: the cross block is measured, and it is zero
    checks = problem["record"]["stiffness_checks"]
    assert checks["cross_block_norm"] == 0.0
    assert checks["phase_block_jacobian_departure"] < 1e-6
    assert problem["record"]["expectation_force_check"] < 1e-12


def test_the_ward_identity_holds_on_every_pure_gauge_direction(
        host_problem):
    """(D - Pi(0)) g = 0 to rounding on each pure-gauge direction, while
    Pi(0) g alone is of order one: the identity is a genuine cancellation
    between the diamagnetic and the paramagnetic terms."""
    _, _, _, problem = host_problem
    ward = problem["record"]["ward_identity"]
    assert ward["directions"] == 9
    assert ward["residual"] < 1e-12
    assert ward["paramagnetic_alone"] > 1e-2


def test_the_elimination_matches_a_dense_reference(host_problem,
                                                   alignment):
    """-1/2 J^T A^D J on the three-particle space: the library's elimination in
    the reduced coordinates against a dense assembly from A^D itself."""
    _, _, carrier, problem = host_problem
    frame = bp._micro_frame([alignment] * bp.SHEETS)
    dual = np.linalg.inv(frame)
    declaration = cob.DressedFluctuationDeclaration()
    declaration.carrier_dimension = 18
    declaration.carrier = list(carrier.reshape(-1))
    declaration.couplings = [list(o.reshape(-1))
                             for o in problem["reduced_couplings"]]
    declaration.bare_stiffness = list(
        problem["reduced_stiffness"].reshape(-1))
    declaration.occupied_modes = 3
    read = cob.DressedFluctuation(declaration).effective_action(
        list(frame.reshape(-1)), list(dual.reshape(-1)), 3)
    dimension = int(read.dimension)
    # the dense lift is in the library's three-particle basis
    one_body = np.asarray(read.one_body).reshape(dimension, dimension)
    assert np.abs(one_body - bp.second_quantized(dual @ carrier @ frame)
                  ).max() < 1e-10 * np.abs(one_body).max()
    quartic = np.asarray(read.quartic).reshape(dimension, dimension)
    reference = bp.dense_quartic_reference(problem["couplings"],
                                           problem["drazin"], frame, dual)
    assert np.abs(quartic - reference).max() < 1e-9 * np.abs(reference).max()


def test_the_lengths_only_elimination_is_the_plain_inverse():
    """With every hinge of the lone tetrahedron in the Regge sum the length
    block is nonsingular, so the lengths-only elimination inverts it whole:
    no null space, no gauge directions, nothing for the Ward identity to
    check."""
    _, _, _, problem = _host_problem(1.0, elimination="lengths",
                                     regge_hinges="all")
    drazin = problem["record"]["drazin"]
    assert drazin["coordinates"] == 18
    assert drazin["null_dimension"] == 0
    assert drazin["eliminated_dimension"] == 18
    assert problem["record"]["ward_identity"] == {"directions": 0}


def test_a_lengths_only_elimination_without_an_interior_hinge_is_refused():
    """The length stiffness is the Regge term's alone. With only the
    interior hinges in the Regge sum the lone tetrahedron has none, the
    length block is zero by structure, every coordinate lies in the null
    space, and the lengths-only elimination is refused by name instead of
    inverting an empty block."""
    with pytest.raises(ValueError, match="zero by structure"):
        _host_problem(1.0, elimination="lengths", regge_hinges="interior")


def test_a_refused_content_is_recorded_and_the_scan_continues(monkeypatch):
    def evaluate(content, kappa, beta, config):
        if tuple(content) == (0, 3, 0):
            raise ValueError("band 1 has rank 2")
        return {"content": list(content), "doublet_reads": []}
    monkeypatch.setattr(bp, "evaluate_content", evaluate)
    config = bp.default_config([0.5], [0.5],
                               selected_contents=[(0, 3, 0), (1, 1, 1)])
    point = bp.scan_point(0.5, 0.5, config)
    assert point["failed_contents"] == [[0, 3, 0]]
    assert point["contents"][0]["failed"] == "band 1 has rank 2"
    assert point["contents"][1]["content"] == [1, 1, 1]


def test_the_elimination_rule_is_declared_and_recorded():
    assert bp.build_parser().parse_args(["run"]).eliminate == \
        "lengths-and-phases"
    assert bp.default_config()["elimination"] == "lengths-and-phases"
    assert bp.build_parser().parse_args(
        ["run", "--eliminate", "lengths"]).eliminate == "lengths"


# ------------------------------------------------------------------ ratios


def test_ratios_are_taken_on_the_lowest_poles():
    def record(content, sectors, doublet_content=(1, 1, 1)):
        """A content record with one doublet read: a content names bands of
        h_1, the sectors are read per doublet content of h-bar_1."""
        out = {}
        for key, (value, irreps) in sectors.items():
            entry = {"lowest_pole": value}
            out[key] = {"quasi_free": entry, "with_quartic": entry,
                        "restriction_to_2T": irreps,
                        "nucleon_reading": "2" in irreps,
                        "delta_reading": sorted(irreps) == ["2'", "2''"]}
        return {"content": content, "doublet_reads": [
            {"doublet_content": list(doublet_content), "sectors": out}]}
    half, three = str(bp.SPIN_HALF), str(bp.SPIN_THREE_HALVES)
    out = bp.ratios([
        record([3, 0, 0], {three: (9.0 + 0j, ["2'", "2''"])}, (0, 2, 1)),
        record([2, 1, 0], {half: (10.0 + 0j, ["2'"]),
                           three: (10.0 + 0j, ["2''", "2"])}),
        record([1, 1, 1], {half: (12.0 + 0j, ["2"]),
                           three: (12.0 + 0j, ["2'", "2''"])})])
    by_spin = out["quasi_free"]["by_spin_lift"]
    assert by_spin["nucleon_content"] == [2, 1, 0]
    assert by_spin["delta_content"] == [3, 0, 0]
    assert by_spin["delta_doublet_content"] == [0, 2, 1]
    assert by_spin["nucleon_doublet_content"] == [1, 1, 1]
    assert by_spin["pole_ratio"] == pytest.approx(10.0 / 9.0)
    assert by_spin["delta_is_a_delta_reading"]
    assert not by_spin["delta_ambiguous_with_spin_half"]
    # the pole is the rest energy (WP v18 §13.3): the target is the mass
    # ratio itself
    assert by_spin["target_mass_ratio"] == pytest.approx(938.272 / 1232.0)
    assert "target_mass_squared_ratio" not in by_spin
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


class _Text:
    def __init__(self, message):
        self.messages = [message]

    def set_text(self, message):
        self.messages.append(message)


class _Figure:
    """Enough of a figure for the live loop and its status line."""

    def __init__(self):
        self.canvas = _Canvas()
        self.texts = []
        self.axes = []

    def text(self, x, y, message, **kwargs):
        made = _Text(message)
        self.texts.append(made)
        return made

    def messages(self):
        return [m for text in self.texts for m in text.messages]


def _stub_matplotlib(monkeypatch, backend="qtagg", figures=None,
                     closes=None):
    """Stub pyplot for the live loop. `plt.close` fires the figure's close
    event, as the Qt backend does, and is recorded in `closes`."""
    import matplotlib
    import matplotlib.pyplot as plt

    def close(figure):
        if closes is not None:
            closes.append(figure)
        figure.canvas.close()

    def figure(**kwargs):
        made = _Figure()
        if figures is not None:
            figures.append(made)
        return made

    monkeypatch.setattr(matplotlib, "get_backend", lambda: backend)
    monkeypatch.setattr(plt, "isinteractive", lambda: True)
    monkeypatch.setattr(plt, "figure", figure)
    monkeypatch.setattr(plt, "show", lambda **kwargs: None)
    monkeypatch.setattr(plt, "close", close)
    monkeypatch.setattr(plt, "pause", lambda interval: time.sleep(0.001))


@pytest.mark.parametrize("backend", ["agg", "webagg"])
def test_live_refuses_a_non_interactive_or_webagg_backend(monkeypatch,
                                                          backend):
    _stub_matplotlib(monkeypatch, backend)
    with pytest.raises(RuntimeError, match="--live needs an interactive"):
        bp.drive_live(bp.default_config([1.0], [1.0],
                                        selected_contents=[(3, 0, 0)]))


def _cheap_scan_point(kappa, beta, config, on_content=None):
    """A deterministic stand-in for one scan point, so the live path is tested
    without the relaxation's cost; the claim under test is that the live
    worker runs the same `drive` and returns the same result."""
    record = {"content": [3, 0, 0], "seconds": 0.0, "doublet_reads": [{
        "doublet_content": [0, 2, 1],
        "sectors": {str(bp.SPIN_THREE_HALVES): {
            "quasi_free": {"lowest_pole": complex(kappa, beta)},
            "with_quartic": {"lowest_pole": complex(beta, kappa)}}}}]}
    return {"kappa": kappa, "beta": beta, "contents": [record],
            "ratios": bp.ratios([record])}


def test_live_and_headless_outputs_are_identical(monkeypatch):
    monkeypatch.setattr(bp, "scan_point", _cheap_scan_point)
    config = bp.default_config([1.0, 2.0], [0.5], selected_contents=[(3, 0, 0)],
                               tolerances=RUN.TOLERANCES)
    headless = bp.drive(dict(config))
    _stub_matplotlib(monkeypatch)
    drawn = []
    monkeypatch.setattr(bp, "draw_frame",
                        lambda figure, frames, index: drawn.append(index))
    live = bp.drive_live(dict(config))
    assert drawn == [0, 1]
    assert json.dumps(bp._jsonable(live), sort_keys=True) == \
        json.dumps(bp._jsonable(headless), sort_keys=True)


def test_the_window_says_which_scan_point_is_running(monkeypatch):
    def slow_point(kappa, beta, config, on_content=None):
        time.sleep(0.05)
        return _cheap_scan_point(kappa, beta, config)

    monkeypatch.setattr(bp, "scan_point", slow_point)
    monkeypatch.setattr(bp, "draw_frame", lambda figure, frames, index: None)
    figures = []
    _stub_matplotlib(monkeypatch, figures=figures)
    bp.drive_live(bp.default_config([1.0, 2.0], [0.5],
                                    selected_contents=[(3, 0, 0)],
                                    tolerances=RUN.TOLERANCES))
    messages = figures[0].messages()
    assert any(m.startswith("scan point 1 of 2 (kappa 1, beta 0.5) is "
                            "running: 0:00 elapsed") for m in messages)
    assert any(m.startswith("scan point 2 of 2 (kappa 2, beta 0.5)")
               for m in messages)


def test_the_drivers_own_close_at_the_end_is_not_reported(monkeypatch,
                                                          capsys):
    monkeypatch.setattr(bp, "scan_point", _cheap_scan_point)
    monkeypatch.setattr(bp, "draw_frame", lambda figure, frames, index: None)
    closes = []
    _stub_matplotlib(monkeypatch, closes=closes)
    bp.drive_live(bp.default_config([1.0], [0.5],
                                    selected_contents=[(3, 0, 0)],
                                    tolerances=RUN.TOLERANCES))
    assert len(closes) == 1
    assert "was closed" not in capsys.readouterr().out


def test_keep_open_leaves_the_final_frame_on_screen(monkeypatch):
    monkeypatch.setattr(bp, "scan_point", _cheap_scan_point)
    monkeypatch.setattr(bp, "draw_frame", lambda figure, frames, index: None)
    figures, closes = [], []
    _stub_matplotlib(monkeypatch, figures=figures, closes=closes)
    bp.drive_live(bp.default_config([1.0], [0.5],
                                    selected_contents=[(3, 0, 0)],
                                    tolerances=RUN.TOLERANCES),
                  keep_open=True)
    assert closes == []
    assert figures[0].messages()[-1] == \
        "the scan is complete; writing the outputs"


def test_the_status_line_is_centred_until_a_frame_is_drawn():
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    figure = Figure()
    FigureCanvasAgg(figure)
    bp.draw_status(figure, "tick 0 of 1 is running: 0:00 elapsed")
    (centred,) = figure.texts
    assert centred.get_position() == (0.5, 0.5)
    bp.draw_status(figure, "tick 0 of 1 is running: 0:02 elapsed")
    assert figure.texts == [centred]
    assert centred.get_text().endswith("0:02 elapsed")
    figure.add_subplot()
    figure.clear()
    figure.add_subplot()
    bp.draw_status(figure, "tick 1 of 1 is running: 0:00 elapsed")
    (bottom,) = figure.texts
    assert bottom.get_position() == (0.5, 0.004)


def test_elapsed_text():
    assert bp.elapsed_text(0) == "0:00"
    assert bp.elapsed_text(75.9) == "1:15"
    assert bp.elapsed_text(3725) == "1:02:05"


def test_holding_without_an_open_window_returns_at_once():
    import matplotlib.pyplot as plt
    plt.close("all")
    assert bp.hold_live_window("close this window to exit") is False


def test_a_worker_error_reaches_the_main_thread(monkeypatch):
    def exploding(*args, **kwargs):
        raise ValueError("the scan point failed")
    monkeypatch.setattr(bp, "scan_point", exploding)
    _stub_matplotlib(monkeypatch)
    with pytest.raises(ValueError, match="the scan point failed"):
        bp.drive_live(bp.default_config([1.0], [1.0],
                                        tolerances=RUN.TOLERANCES))


def test_closing_the_window_switches_the_run_to_headless(monkeypatch,
                                                         tmp_path, capsys):
    """The window is closed after the first frame: the scan still runs to the
    end, the result equals the headless one, nothing more is drawn, every
    point reaches the JSON-lines file, and stdout says so."""
    monkeypatch.setattr(bp, "scan_point", _cheap_scan_point)
    config = bp.default_config([1.0, 2.0, 3.0], [0.5],
                               selected_contents=[(3, 0, 0)],
                               tolerances=RUN.TOLERANCES)
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
                               selected_contents=[(3, 0, 0)],
                               tolerances=RUN.TOLERANCES)
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


def test_the_pole_table_keeps_every_pole_with_its_pair():
    """One row per pole of every (content, doublet content) sector, not only
    each sector's lowest, in ascending order of real part, each row naming its
    content and doublet content."""
    half, three = str(bp.SPIN_HALF), str(bp.SPIN_THREE_HALVES)

    def column(values):
        return {"poles": values, "multiplicity": [2] * len(values),
                "lowest_pole": min(values, key=lambda p: p.real)}

    def record(content, doublet_content, poles):
        return {"content": content, "doublet_reads": [{
            "doublet_content": doublet_content, "sectors": {
                key: {"quasi_free": column(values),
                      "with_quartic": column(values),
                      "restriction_to_2T": ["2"]}
                for key, values in poles.items()}}]}

    table = bp.pole_table([
        record([2, 1, 0], [1, 1, 1], {half: [7.0 + 0j, 5.0 + 0j],
                                      three: [5.0 + 0j]}),
        record([3, 0, 0], [0, 3, 0], {three: [6.0 + 0j, 4.0 + 0j]})])
    rows = table["quasi_free"][three]
    assert [(row["content"], row["doublet_content"], row["pole"],
             row["lowest_in_sector"]) for row in rows] == [
        ([3, 0, 0], [0, 3, 0], 4.0, True),
        ([2, 1, 0], [1, 1, 1], 5.0, True),
        ([3, 0, 0], [0, 3, 0], 6.0, False)]
    assert [row["pole"] for row in table["with_quartic"][half]] == [5.0, 7.0]
    assert all(row["spin_j_j_plus_1"] == bp.SPIN_HALF
               for row in table["with_quartic"][half])


@pytest.fixture(scope="module")
def projectors(alignment):
    """The isotypic projectors of 2, 2', 2'' on the three-particle sector of
    the declared host, as `evaluate_content` forms them."""
    frame = bp._micro_frame([alignment] * bp.SHEETS)
    return bp.isotypic_projectors(
        alignment, bp.rotation_action([bp.monopole_support()] * bp.SHEETS),
        frame, np.linalg.inv(frame))


def test_the_isotypic_projectors_decompose_the_three_particle_sector(
        projectors):
    """The three projectors of the double cover's spinor types are
    idempotent, orthogonal, and exhaust the 816-dimensional three-particle
    sector of the 18 modes: three particles of the odd-monopole tetrahedron
    are a spinor of one of the three types (WP v18 Section 11.1)."""
    matrices, record = projectors
    assert sorted(matrices) == sorted(bp.IRREP_NAMES)
    total = sum(record[name]["rank"] for name in bp.IRREP_NAMES)
    assert total == len(bp.occupation_basis()) == 816
    for name in bp.IRREP_NAMES:
        assert record[name]["idempotency_residual"] < 1e-9
        assert record[name]["rank"] > 0
    for a, b in (("2", "2'"), ("2'", "2''"), ("2", "2''")):
        assert np.max(np.abs(matrices[a] @ matrices[b])) < 1e-9


def test_the_lift_agrees_with_the_types(alignment, projectors):
    """The spin sectors of the constructed lift are the isotypic components
    the finite group names: for the doublet content (1, 1, 1) of total
    triality zero, the spin-1/2 sector lies in the type 2 and the spin-3/2
    sector in 2' + 2'', half in each."""
    matrices, _ = projectors
    _, triality, sectors = bp.doublet_sectors([1, 1, 1],
                                              alignment["trialities"])
    assert triality == 0
    entry = bp.sector_entry(bp.SPIN_HALF, triality,
                            sectors[bp.SPIN_HALF], (), matrices)
    assert entry["spinor_type"] == "2"
    weights = entry["isotypic_weights"]
    assert abs(weights["2"] - 1.0) < 1e-9
    assert abs(weights["2'"]) < 1e-9 and abs(weights["2''"]) < 1e-9
    entry = bp.sector_entry(bp.SPIN_THREE_HALVES, triality,
                            sectors[bp.SPIN_THREE_HALVES], (), matrices)
    assert entry["spinor_type"] == "2' + 2''"
    weights = entry["isotypic_weights"]
    assert abs(weights["2"]) < 1e-9
    assert abs(weights["2'"] - 0.5) < 1e-9 and abs(weights["2''"] - 0.5) < 1e-9


def test_every_pole_of_a_sector_carries_its_own_certificates(alignment,
                                                              projectors):
    """An operator diagonal in the occupation basis with distinct entries
    splits the four-dimensional spin-1/2 sector of the doublet content
    (1, 1, 1) into distinct poles. Each pole is read with its own spinor,
    spin-lift and colour certificates, and the lowest pole's are repeated
    beside it. Every vector of the sector is a colour singlet of type 2 and
    of sharp spin 1/2 under the lift, so every certificate holds."""
    matrices, _ = projectors
    _, triality, sectors = bp.doublet_sectors([1, 1, 1],
                                              alignment["trialities"])
    sector = sectors[bp.SPIN_HALF]
    assert sector.shape[1] == 4
    rng = np.random.default_rng(7)
    operator = np.diag(rng.uniform(1.0, 2.0, size=len(
        bp.occupation_basis()))).astype(complex)
    entry = bp.sector_entry(bp.SPIN_HALF, triality, sector,
                            (("quasi_free", operator),), matrices,
                            RUN.TOLERANCES)
    read = entry["quasi_free"]
    poles = read["poles"]
    assert len(poles) >= 2
    assert len(read["pole_certificates"]) == len(poles)
    for certificate in read["pole_certificates"]:
        assert certificate["sharp_spinor"]
        assert certificate["spinor_type"] == "2"
        assert certificate["spinor_right_residual"] < 1e-9
        assert certificate["spinor_left_residual"] < 1e-9
        assert abs(certificate["spinor_weight"] - 1.0) < 1e-9
        assert certificate["spin_lift_sharp"]
        assert certificate["colour_casimir_residual"] < 1e-10
    # without projectors the spinor certificate is unmeasured, not false
    bare = bp.sector_entry(bp.SPIN_HALF, triality, sector,
                           (("quasi_free", operator),), config=RUN.TOLERANCES)
    assert bare["quasi_free"]["pole_certificates"][0]["sharp_spinor"] is None
    assert bare["quasi_free"]["pole_certificates"][0]["spin_lift_sharp"]
    lowest = poles.index(read["lowest_pole"])
    for key, value in read["pole_certificates"][lowest].items():
        assert read[key] == value


# -------------------------------------------------------- the anchor atlas


def test_the_anchor_atlas_of_the_declared_host(alignment):
    """Quark condition 3 on the declared host (WP v18 Section 10): the
    reference doublet, the coexact j = 1/2 doublet, anchors on every face of
    every sheet, with the connection-dressed covariance and the transition
    cocycle at machine precision and the invariant coordinates attached; on
    the symmetric host its profile has one modulus on every face."""
    anchor = bp.anchor_atlas_read(bp.build_host(), [alignment] * bp.SHEETS,
                                  RUN.TOLERANCES["certificate_tolerance"])
    assert anchor["anchored"] and anchor["anchoring_faces"] == 4
    assert anchor["covariance_residual"] < 1e-12
    assert anchor["transition_cocycle_residual"] < 1e-12
    assert anchor["invariant_coordinates_attached"]
    assert anchor["failed_certificates"] == []
    assert anchor["faces"] == [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
    assert len(anchor["sheets"]) == bp.SHEETS
    for sheet in anchor["sheets"]:
        assert len(sheet["coordinates"]) == 4
        assert all(len(row) == 3 for row in sheet["coordinates"])
        moduli = [abs(c) for row in sheet["coordinates"] for c in row]
        assert max(moduli) - min(moduli) < 1e-9 and min(moduli) > 0.1
        assert len(sheet["invariant_coordinates"]) == 4
        assert all(abs(a) > 1.0 for a in sheet["invariant_coordinates"])
    evidence = bp.anchor_evidence(anchor)
    assert [e.name for e in evidence] == [
        "anchor-profile-nonzero", "anchor-covariance", "anchor-transitions",
        "anchor-stable-across-frames"]
    assert [e.held for e in evidence] == [True, True, True, None]
    assert bp.anchor_text(anchor).startswith(
        "anchor atlas anchored (4 of 4 faces anchor on every sheet")


def test_the_anchor_evidence_reads_a_refusal_and_an_absent_read():
    refused = {"band": "the band", "faces": [[0, 1, 2]] * 4,
               "tolerance": 1e-8, "anchored": False, "anchoring_faces": 0,
               "covariance_residual": 0.5,
               "transition_cocycle_residual": 0.0,
               "invariant_coordinates_attached": False,
               "failed_certificates": ["identically-zero-profile"],
               "sheets": [{"anchoring_faces": 0}]}
    evidence = bp.anchor_evidence(refused)
    assert [e.held for e in evidence] == [False, False, False, None]
    assert "identically-zero-profile" in evidence[0].detail
    assert "not attached" in evidence[2].detail
    assert bp.anchor_text(refused).startswith("anchor atlas refused")
    assert [e.held for e in bp.anchor_evidence(None)] == [None] * 4
    assert bp.anchor_text(None) == "anchor atlas unread"


# ------------------------------------------------- the spectral fingerprint


def test_the_centroid_lengths_of_the_regular_tetrahedron():
    """On the regular tetrahedron of squared edge length 8 every vertex is at
    squared distance 3 from the centroid, the squared circumradius."""
    radii = bp.centroid_squared_lengths([8.0] * 6)
    assert np.allclose(radii, [3.0] * 4)
    # and on a stretched one the identity for four points holds
    squared = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    total = sum(bp.centroid_squared_lengths(squared))
    assert abs(total - sum(squared) / 4.0) < 1e-12


def test_the_refined_host_keeps_the_boundary_and_its_monopole():
    """The stellar subdivision of the declared host: fifteen vertices, thirty
    edges and twelve tetrahedra; each sheet's boundary faces, their
    holonomies and the unit monopole through them are unchanged, and the new
    edges carry the centroid's lengths and the trivial link."""
    refined, data = bp.refined_host(bp.build_host())
    assert len(refined.getVertexList().toVector()) == 15
    assert len(refined.getEdgeList().toVector()) == 30
    assert len(data) == bp.SHEETS
    for sheet in data:
        assert len(sheet["squared_lengths"]) == 10
        new = [k for k, e in enumerate(bp.REFINED_EDGES)
               if bp.REFINED_CENTRE in e]
        assert all(abs(sheet["squared_lengths"][k] - 3.0) < 1e-12
                   for k in new)
        assert all(abs(sheet["links"][k] - 1.0) < 1e-12 for k in new)
        support = bp.refined_support(sheet)
        read = support.monopoleNumber()
        assert read.monopole_number == 1 and read.odd
        spin = support.spinRead(bp.refined_rotation_group())
        assert spin.cocycle.nontrivial and spin.half_integer_doublet


def test_the_spectral_fingerprint_of_the_declared_host():
    """Quark condition 7 on the declared host: under the declared odd
    relabeling every spectrum, the doublet's energy and its edge weights are
    unchanged to rounding, and under the declared refinement the doublet's
    type is carried by a single rank-two band of the refined action whose
    restriction to the shared edges is the original doublet; the energy
    shift under refinement is reported, not gated."""
    config = bp.default_config([1.0], [1.0], tolerances=RUN.TOLERANCES)
    read = bp.spectral_fingerprint_read(bp.build_host(), 1.0, 1.0, config)
    assert read["doublet_found"] and read["sheet"] == 0
    assert np.allclose(read["doublet_weights"], [1.0 / 3.0] * 6)
    relabeling = read["relabeling"]
    assert relabeling["permutation"] == [1, 0, 2, 3]
    assert relabeling["held"]
    assert relabeling["monopole_number"] == -1  # the orientation flips
    for key in ("spectrum_shift", "averaged_spectrum_shift",
                "doublet_weight_shift", "doublet_energy_shift"):
        assert relabeling[key] < 1e-12
    refinement = read["refinement"]
    assert refinement["doublet_found"] and refinement["held"]
    assert refinement["monopole_number"] == 1
    assert refinement["isotypic_dimension"] == 2
    assert refinement["refined_rank"] == 2
    assert refinement["overlap"] > 1.0 - 1e-9
    assert refinement["energy_shift"] > 0.0
    evidence = bp.fingerprint_evidence(read)
    assert [e.name for e in evidence] == ["refinement-stability",
                                          "relabeling-stability"]
    assert [e.held for e in evidence] == [True, True]
    assert bp.fingerprint_text(read).startswith(
        "spectral fingerprint: relabeling stable")
    assert [e.held for e in bp.fingerprint_evidence(None)] == [None, None]
    assert bp.fingerprint_text(None) == "spectral fingerprint unread"


# ------------------------------------------- refusals after the mean field


def test_a_relaxed_cell_without_the_tetrahedral_group_is_refused_by_name():
    """(0123, 201) of the recursion's tick-0 run with the occupied fiber
    pinned: the joint Newton converges, and the relaxed cell has lost every
    rotation but the identity (squared lengths that differ by two thirds of
    the largest, a gauge-compensation residual of order one), so the spin
    read, which needs the whole tetrahedral group, is refused by name.
    `scan_point` records the refusal with the solve's record and the
    symmetry read, and the content supplies no pole."""
    from tessera.drivers import recursion as R

    config = bp.default_config([1.0], [1.0], selected_contents=[(2, 0, 1)],
                               tolerances=RUN.TOLERANCES)
    config["host_cell"] = RUN.HOST_CELLS[(0, 1, 2, 3)]
    config["held_sectors"] = R.held_sectors([[0, 1, 2, 3]], [1], 4)
    point = bp.scan_point(1.0, 1.0, config)
    assert point["failed_contents"] == [[2, 0, 1]]
    (record,) = point["contents"]
    assert record["refusal"] == "not tetrahedrally symmetric"
    assert record["failed"].startswith(
        "the spin read is refused: the relaxed cell has 1 of the 12 rotations "
        "of the tetrahedron as symmetries at the certificate tolerance")
    assert record["relaxation"]["band_selection"] == "continuation"
    symmetry = record["symmetry"]
    assert symmetry["order"] == 1 and symmetry["group"] == [[0, 1, 2, 3]]
    assert symmetry["length_departure"] > 0.5
    assert symmetry["compensation_residual"] > 1.0
    assert record["doublet_reads"] == []


def test_a_read_on_lengths_that_grew_without_bound_is_refused_by_name():
    """(0123, 201) of the recursion's tick-0 run: the content occupies the
    host's negative band, whose force contracts the cell, and the joint
    Newton follows it until the residual is at its floor on the held set,
    three decades of length beyond the host. `scan_point` records the
    content as refused on that geometry, which is not Kontsevich-Segal
    allowable, with the solve's record (the band selection, the iterations,
    the force, why it stopped, every iterate), and it supplies no pole; the
    report prints the refusal with the solve on the content's line."""
    from tessera.drivers import recursion as R
    from tests.drivers import _recursion_run_2026_09_23 as RUN

    # no constraint of the occupied fiber pinned, so nothing holds the
    # lengths against the band's force
    config = bp.default_config([1.0], [1.0], selected_contents=[(2, 0, 1)],
                               fiber_moments=0, tolerances=RUN.TOLERANCES)
    config["host_cell"] = RUN.HOST_CELLS[(0, 1, 2, 3)]
    config["held_sectors"] = R.held_sectors([[0, 1, 2, 3]], [1], 4)
    point = bp.scan_point(1.0, 1.0, config)
    assert point["failed_contents"] == [[2, 0, 1]]
    (record,) = point["contents"]
    assert record["refusal"] == "not Kontsevich-Segal allowable"
    assert record["failed"].startswith(
        "the pole read is refused: the geometry the mean-field solve reached "
        "is not Kontsevich-Segal allowable")
    assert record["doublet_reads"] == []
    solve = record["relaxation"]
    assert solve["band_selection"] == "continuation"
    assert solve["converged"] is False
    floor = "the residual is at its floor on the held set"
    assert solve["stop_reason"] == floor
    assert solve["iterations"] == len(solve["trace"]) - 1
    assert solve["largest_length_ratio"] > 1e3
    assert all("newton" in entry for entry in solve["trace"][:-1])
    (line,) = bp.content_pair_lines(record, "  ")
    assert line.startswith("  content [2, 0, 1]: refused: the pole read is "
                           "refused: the geometry the mean-field solve "
                           "reached is not Kontsevich-Segal allowable")
    assert "mean field converged False" in line
    assert "; stopped: " + floor + " (" in line
    text = R._content_line({"cell": [0, 1, 2, 3]}, record)
    assert "failed: the pole read is refused: the geometry the mean-field " \
        "solve reached is not Kontsevich-Segal allowable" in text
    assert "; stopped: " + floor + " (" in text
    assert bp.point_lines(point)[0] == (
        "kappa=1 beta=1: 1 contents, 1 refused; one line per (content, "
        "doublet content) pair, poles s with multiplicity x")


def test_the_declared_read_pins_every_moment_of_the_occupied_fiber():
    """The mean field pins every power sum of the occupied fiber at the host
    (m_c = r). On (0123, 030) of the recursion's tick-0 run the joint Newton
    converges with the three pinned moments held, and the record carries the
    fiber's rank, the multipliers, the residuals, the Hessian along the
    Hellmann-Feynman force with its sign, and the role of kappa; the
    content's line prints them."""
    from tessera.drivers import recursion as R
    from tests.drivers import _recursion_run_2026_09_23 as RUN

    config = bp.default_config([1.0], [1.0], selected_contents=[(0, 3, 0)],
                               fiber_pinning="power-sums",
                               tolerances=RUN.TOLERANCES)
    assert config["fiber_moments"] == "r"
    config["host_cell"] = RUN.HOST_CELLS[(0, 1, 2, 3)]
    config["held_sectors"] = R.held_sectors([[0, 1, 2, 3]], [1], 4)
    _, _, report = bp.relax_content((0, 3, 0), 1.0, 1.0, config)
    solve = bp.relaxation_record(report)
    assert solve["converged"] and solve["fiber_rank"] == 3
    assert solve["fiber_pinning"] == "power-sums"
    assert solve["fiber_moments"] == 3 and len(solve["multipliers"]) == 3
    assert max(abs(r) / abs(t) for r, t in zip(
        solve["moment_residuals"], solve["moment_targets"])) < 1e-9
    assert solve["force_hessian_sign"] in ("positive", "negative")
    text = bp.relaxation_text(solve)
    assert "3 of the occupied fiber's 3 power sums pinned at the host" in text
    assert "Hessian on the range of the Hellmann-Feynman force" in text
    assert "(%s)" % solve["force_hessian_sign"] in text


def test_the_declared_read_pins_the_eigenvalue_of_every_occupied_band():
    """The declared pinning: the eigenvalue of each occupied band, one
    constraint per band. (0123, 030) occupies one band of rank three, so one
    eigenvalue is pinned with one multiplier; the joint Newton converges with
    it held at the host's value, and the record and the content's line say
    so."""
    from tessera.drivers import recursion as R
    from tests.drivers import _recursion_run_2026_09_23 as RUN

    config = bp.default_config([1.0], [1.0], selected_contents=[(0, 3, 0)],
                               tolerances=RUN.TOLERANCES)
    assert config["fiber_pinning"] == bp.DECLARED_FIBER_PINNING == "eigenvalues"
    config["host_cell"] = RUN.HOST_CELLS[(0, 1, 2, 3)]
    config["held_sectors"] = R.held_sectors([[0, 1, 2, 3]], [1], 4)
    _, _, report = bp.relax_content((0, 3, 0), 1.0, 1.0, config)
    assert report.fiber_constraint_form == \
        cob.FiberConstraintForm.BandEigenvalues
    solve = bp.relaxation_record(report)
    assert solve["converged"] and solve["fiber_rank"] == 3
    assert solve["fiber_pinning"] == "eigenvalues"
    assert solve["fiber_moments"] == 1 and len(solve["multipliers"]) == 1
    (band,) = report.bands
    assert solve["moment_targets"][0] == pytest.approx(band.eigenvalues[0],
                                                        rel=1e-9)
    assert abs(solve["moment_residuals"][0]) < 1e-9 * abs(
        solve["moment_targets"][0])
    text = bp.relaxation_text(solve)
    assert ("the eigenvalues of 1 occupied bands (fiber rank 3) pinned at "
            "the host") in text
