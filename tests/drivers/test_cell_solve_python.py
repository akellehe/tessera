# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The cell solve as a drive of `MultiCobordism` (`tessera.drivers.cell_solve`).

Terms used below:

* the *base* is one tetrahedron on vertices 0..3 carrying the unit-monopole
  connection at squared edge length 8 (`baryon_poles.build_base`), six edges;
* the *host* is its three-sheeted support (`cell_solve.sheeted_support`):
  three disjoint copies of the base, 18 edges, which is the complex of
  `baryon_poles.build_host`;
* the *system* of a content is the self-consistent stationarity system on
  the host with one squared length and one link per base edge
  (`cell_solve.ContentSystem`): six length equations, six link equations and
  one equation per pinned band of the content;
* the *residual* R is that system's residual, and the *objective* is its
  Euclidean norm ||R||, the scalar `MultiCobordism` descends;
* the *step* of order one is the minimum-norm solution d of J d = -R, with
  J the closed-form Jacobian of R.

What is asserted: the sheeted support is the host; the objective a node
reports is the norm of the residual; the direction handed to stage 2 is the
order-one step in the engine's sign and phase conventions; a complex without
a residual scores as infinite and the reason is kept; held sectors and band
references are carried by vertex and by cell to a complex a move has
changed; a declared time ends a drive by name; and the options of a solve
reach the configuration and the command line.
"""
import math

import numpy as np
import pytest

from tessera import cobordism as cob
from tessera.drivers import baryon_poles as bp
from tessera.drivers import cell_solve as cs
from tessera.drivers import recursion as R

CONTENT = (0, 0, 3)


def _unit(angle):
    return complex(math.cos(angle), math.sin(angle))


def _content_system(content=CONTENT, sectors=()):
    """The base, the declared configuration and the content's system as
    `baryon_poles.relax_content` builds it: the fiber's constraints pinned
    at the host's values."""
    config = bp.default_config(kappas=[1.0], betas=[1.0],
                               selected_contents=[tuple(content)])
    config["held_sectors"] = list(sectors)
    base = bp.build_base(config["edge_squared"])
    host = cs.sheeted_support(base, bp.SHEETS)

    def declare(spacetime):
        return bp.action_declaration(spacetime, 1.0, 1.0,
                                     config["regge_hinges"])

    probe = bp.mean_field_declaration(content, config)
    probe.geometry = bp.support_geometry(config, host, host)
    action = cob.JointAction(host.spacetime, declare(host.spacetime))
    count = bp.fiber_moment_count(probe, action, config["fiber_moments"],
                                  config["fiber_pinning"])
    probe.fiber_moments = count
    start = cob.SelfConsistentMeanField(action, probe).read()

    def mean_field_of(support):
        declaration = bp.mean_field_declaration(content, config)
        declaration.geometry = bp.support_geometry(config, support, host)
        declaration.fiber_moments = count
        if count:
            declaration.fiber_moment_targets = list(start.moment_targets)
            declaration.fiber_moment_scale = float(start.moment_scale)
        return declaration

    system = cs.ContentSystem(declare, mean_field_of, base, bp.SHEETS)
    return base, config, system


# ------------------------------------------------------- the sheeted support


def test_the_sheeted_support_of_the_base_cell_is_the_declared_host():
    """Three copies of the base tetrahedron, vertex k of the base at
    4 t + k on sheet t, every edge carrying the base edge's length and its
    phase on the corresponding orientation: edge for edge the complex of
    `build_host`."""
    base = bp.build_base()
    support = cs.sheeted_support(base, bp.SHEETS)
    assert support.count == 4 and support.sheets == 3
    assert support.base_vertices == [0, 1, 2, 3]
    assert cs.edge_fields(support.spacetime) == cs.edge_fields(bp.build_host())
    assert cs.complex_counts(support.spacetime) == {
        "vertices": 12, "edges": 18, "cells": 3}
    assert cs.complex_counts(base) == {"vertices": 4, "edges": 6, "cells": 1}


def test_every_edge_of_the_support_names_its_base_edge():
    """`classes` is the base edge of each of the 18 edges, each base edge
    three times, and `orientations` relates the stored links: the phase of a
    support edge is the base edge's phase times the orientation."""
    base = bp.build_base()
    support = cs.sheeted_support(base, bp.SHEETS)
    base_fields = cs.edge_fields(base)
    assert sorted(support.classes) == sorted(list(range(6)) * 3)
    for (a, b, length, phase), index, sign in zip(
            cs.edge_fields(support.spacetime), support.classes,
            support.orientations):
        x, y, base_length, base_phase = base_fields[index]
        assert {a % 4, b % 4} == {x, y}
        assert sign == (1 if (a % 4, b % 4) == (x, y) else -1)
        assert length == base_length and phase == sign * base_phase


def test_a_host_cell_is_carried_by_every_sheet():
    """With a declared cell (six squared lengths and six links, complex), the
    base carries them and the support is `build_host` of the same cell."""
    cell = {"squared_lengths": [8.0 + 0.5j, 7.5, 8.25, 9.0 - 0.25j, 8.0, 7.0],
            "links": [_unit(0.3), _unit(-1.1), 1.2 * _unit(0.2), _unit(2.0),
                      0.9 * _unit(-0.4), _unit(1.3)]}
    base = bp.build_base(cell=cell)
    support = cs.sheeted_support(base, bp.SHEETS)
    assert cs.edge_fields(support.spacetime) == cs.edge_fields(
        bp.build_host(cell=cell))


def test_the_carrier_cells_are_named_by_sheet_and_base_vertices():
    """The 18 edge cells of the host, each named (sheet, base edge): six per
    sheet, in the operator's mode order."""
    support = cs.sheeted_support(bp.build_base(), bp.SHEETS)
    cells = cs.support_cells(support)
    assert len(cells) == 18 and len(set(cells)) == 18
    assert sorted({sheet for sheet, _ in cells}) == [0, 1, 2]
    assert sorted({edge for _, edge in cells}) == [
        (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]


def test_held_sectors_are_carried_to_another_support_by_base_vertex():
    """A sector declared on the host's vertices (sheet t at 4 t + k) is the
    same faces on a support whose base has a fifth vertex (sheet t at
    5 t + k); on the host itself it is returned as it is; and a face one of
    whose base vertices is gone has no value."""
    host = cs.sheeted_support(bp.build_base(), bp.SHEETS)
    sectors = R.held_sectors([[0, 1, 2, 3]], [1], 4)
    assert cs.sectors_on(sectors, host, host) == sectors
    wider = cs.Sheeted(None, 5, [0, 1, 2, 3, 12], 3, [], [])
    moved = cs.sectors_on(sectors, host, wider)
    assert [s.monopole_number for s in moved] == [1, 1, 1]
    for t, (before, after) in enumerate(zip(sectors, moved)):
        assert [[v - 4 * t for v in face] for face in before.faces] == [
            [v - 5 * t for v in face] for face in after.faces]
    without = cs.Sheeted(None, 4, [0, 1, 2, 12], 3, [], [])
    with pytest.raises(ValueError, match="lost its base vertex 3"):
        cs.sectors_on(sectors, host, without)


def test_a_reference_band_is_read_on_the_cells_two_complexes_share():
    """A projector over three cells read on a complex that keeps the first
    and the third, in the other order, and has a cell of its own: the shared
    entries are carried and the new cell carries no weight."""
    band = cob.BandReference()
    band.occupation, band.rank, band.declared_index = 1.0, 1, 0
    band.declared_positions = [0]
    band.projector = list(np.arange(1.0, 10.0).astype(complex))
    cells = ["a", "b", "c"]
    assert cs.reference_on(cells, cells, [band])[0].projector == band.projector
    (moved,) = cs.reference_on(["c", "new", "a"], cells, [band])
    assert np.asarray(moved.projector).reshape(3, 3).tolist() == [
        [9.0, 0.0, 7.0], [0.0, 0.0, 0.0], [3.0, 0.0, 1.0]]
    assert (moved.occupation, moved.rank, moved.declared_index) == (1.0, 1, 0)


# ------------------------------------------------------------- the objective


def test_the_objective_is_the_norm_of_the_residual():
    """The node on the base reports ||R|| of the three-sheeted system, its
    one nonzero term: for the content (0, 0, 3), six length equations, six
    link equations and one pinned band."""
    base, _, system = _content_system()
    point = system.point(base)
    residual = np.asarray(point.relaxation.residual())
    assert len(residual) == 13 and point.count == 6
    assert point.lengths and point.links
    objective = cs.StationarityObjective(system)
    node = cs.cell_node(base, objective)
    assert node.objective() == pytest.approx(np.linalg.norm(residual),
                                             rel=1e-13)
    terms = node.objective_terms()
    assert terms.joint_action_stationarity == node.objective()
    assert terms.regge_stationarity == 0.0
    assert terms.hodge_stationarity == 0.0
    assert objective.term_names() == ["joint_action_stationarity"]
    assert objective.name() == "joint_action_stationarity"
    assert not objective.is_target_conditioned()
    assert objective.scored >= 1 and objective.undefined == []


def test_the_direction_is_the_order_one_step_on_the_base_edges():
    """The direction handed to stage 2 is the step d of the linearization,
    J d = -R on the range of J, in the engine's conventions: stage 2
    subtracts ``ascent`` from z and ``phase_ascent`` from phi, and the step
    multiplies a link by exp(delta), which is phi - i delta."""
    base, _, system = _content_system()
    objective = cs.StationarityObjective(system)
    node = cs.cell_node(base, objective)
    context = cob.ObjectiveDirectionContext()
    context.scalar = node.objective_context()
    context.edge_count = 6
    direction = objective.direction(context)
    point = system.point(base)
    linearization = point.relaxation.linearization()
    step = np.asarray(linearization.newton_step.step)
    assert len(step) == 13
    assert np.array_equal(np.asarray(direction.ascent), -step[:6])
    assert np.array_equal(np.asarray(direction.phase_ascent), 1j * step[6:12])
    assert direction.baseline_computed
    assert direction.baseline == pytest.approx(
        np.linalg.norm(point.relaxation.residual()), rel=1e-13)
    # the step solves the linearized equations on the range of the Jacobian
    jacobian = np.asarray(point.relaxation.jacobian()).reshape(13, 13)
    residual = np.asarray(linearization.residual)
    left = np.linalg.svd(jacobian)[0][:, :linearization.newton_step
                                      .jacobian_rank]
    assert np.abs(left.conj().T @ (jacobian @ step + residual)).max() < 1e-9
    (update,) = objective.updates
    assert update["jacobian_rank"] == linearization.newton_step.jacobian_rank
    assert update["complex"] == {"vertices": 4, "edges": 6, "cells": 1}
    assert update["measured"].force_norm > 0.0


def test_a_linearization_solves_every_right_hand_side_the_same_way():
    """`HolomorphicLinearization.solve` of -R is the Newton step, and its
    solution of another right-hand side b satisfies J d = b on the range of
    J: the higher orders of a step solve against the one decomposition."""
    base, _, system = _content_system()
    point = system.point(base)
    linearization = point.relaxation.linearization()
    newton = linearization.newton_step
    minus = [-x for x in linearization.residual]
    assert np.allclose(linearization.solve(minus), newton.step, rtol=0,
                       atol=1e-12 * max(1.0, np.abs(newton.step).max()))
    assert np.array_equal(np.asarray(point.relaxation.newton_step().step),
                          np.asarray(newton.step))
    jacobian = np.asarray(point.relaxation.jacobian()).reshape(13, 13)
    rng = np.random.default_rng(0)
    b = rng.normal(size=13) + 1j * rng.normal(size=13)
    d = np.asarray(linearization.solve(list(b)))
    left = np.linalg.svd(jacobian)[0][:, :newton.jacobian_rank]
    assert np.abs(left.conj().T @ (jacobian @ d - b)).max() < 1e-9
    with pytest.raises(ValueError, match="right-hand side"):
        linearization.solve([0j] * 5)


def test_an_added_term_enters_the_residual_and_the_jacobian():
    """`linearization(added_residual, added_jacobian)` solves
    (J + H) d = -(R + g): with H the identity and g zero the step is the
    solution of the shifted system."""
    base, _, system = _content_system()
    relaxation = system.point(base).relaxation
    size = relaxation.variable_count()
    residual = np.asarray(relaxation.residual())
    jacobian = np.asarray(relaxation.jacobian()).reshape(size, size)
    shifted = relaxation.linearization(
        [0j] * size, list(np.eye(size, dtype=complex).reshape(-1)))
    step = np.asarray(shifted.newton_step.step)
    assert np.allclose((jacobian + np.eye(size)) @ step, -residual,
                       atol=1e-9 * np.linalg.norm(residual))
    with pytest.raises(ValueError, match="added residual"):
        relaxation.linearization([0j] * (size + 1))


def test_a_complex_without_a_residual_scores_as_infinite():
    """A system that has no value on a complex (here one that raises by
    name) gives an infinite objective, and the reason is kept."""
    base = bp.build_base()

    class Nowhere:
        def point(self, complex_):
            raise ValueError("the action has no value here")

        def accept(self, complex_):
            return None

    objective = cs.StationarityObjective(Nowhere())
    node = cs.cell_node(base, objective)
    assert node.objective() == math.inf
    assert objective.undefined == ["the action has no value here"]


def test_the_order_of_the_step_is_declared_between_one_and_ten():
    _, _, system = _content_system()
    for order in (0, 11):
        with pytest.raises(ValueError, match="integer from 1 to 10"):
            cs.StationarityObjective(system, order)
    with pytest.raises(ValueError, match="reversion"):
        cs.StationarityObjective(system, 3)
    assert cs.StationarityObjective(
        system, 3, series=lambda point, linearization, order: []
    ).direction_order == 3


# ------------------------------------------------------------------ the drive


def test_a_declared_time_ends_a_drive_by_name():
    """With a time limit of zero seconds the first scoring made on the
    drive's thread raises, the engine hands the error back, and the record
    names the declared limit; the base is as it was."""
    base, _, system = _content_system()
    before = cs.edge_fields(base)
    drive = cs.solve(base, system, time_limit_seconds=0.0)
    assert drive["stop_reason"] == cs.STOP_DECLARED_LIMIT
    assert drive["stop_detail"].startswith(
        "the declared time limit of 0 seconds was reached after 0 step "
        "proposals")
    assert drive["accepted_updates"] == 0 and drive["moves_committed"] == 0
    assert not drive["changed"]
    assert cs.edge_fields(drive["spacetime"]) == before


def test_a_declared_update_count_ends_a_relaxation():
    """Relaxation alone with one update declared: the engine returns after
    one stage-2 update. When that update accepted a trial the record names a
    declared limit; when its line search accepted none, the point is
    stationary for the engine and the record says so."""
    base, _, system = _content_system()
    drive = cs.solve(base, system, moves=False, iteration_limit=1)
    assert len(drive["objective"].updates) == 1
    assert drive["moves_committed"] == 0
    if drive["accepted_updates"] == 1:
        assert drive["stop_reason"] == cs.STOP_DECLARED_LIMIT
        assert drive["trace"][1] < drive["trace"][0]
    else:
        assert drive["stop_reason"] == cs.STOP_STATIONARY


# ------------------------------------------------- configuration and options


def test_the_solve_options_are_recorded_at_their_declared_values():
    config = bp.default_config()
    assert {key: config[key] for key, _, _ in bp.SOLVE_OPTIONS} == {
        "direction_order": 1, "band_reference": "previous",
        "pachner_moves": True, "move_lookahead": 1, "move_candidates": 0,
        "moment_stiffness_weight": 0.0, "moment_stiffness_coefficients": [],
        "pinned_vertices": []}
    assert {key: config[key] for key, _, _ in bp.LIMITS} == {
        "iteration_limit": None, "update_limit": None,
        "time_limit_seconds": None}
    arguments = bp.solve_arguments(config)
    assert arguments["tolerance"] == 1e-15
    assert arguments["move_tolerance"] == 1e-15
    assert arguments["direction_order"] == 1 and arguments["moves"] is True
    assert arguments["iteration_limit"] is None
    assert arguments["update_limit"] is None
    assert arguments["time_limit_seconds"] is None


def test_the_solve_options_are_options_of_the_command_line():
    args = bp.build_parser().parse_args(
        ["run", "--direction-order", "4", "--band-reference", "host",
         "--no-pachner-moves", "--move-lookahead", "2", "--move-candidates",
         "5", "--moment-stiffness-weight", "0.5",
         "--moment-stiffness-coefficients", "1", "2", "--pinned-vertices",
         "0", "1", "--update-limit", "7", "--step-tolerance", "1e-12"])
    assert bp.solve_options_from(args) == {
        "direction_order": 4, "band_reference": "host",
        "pachner_moves": False, "move_lookahead": 2, "move_candidates": 5,
        "moment_stiffness_weight": 0.5,
        "moment_stiffness_coefficients": [1.0, 2.0], "pinned_vertices": [0, 1]}
    assert bp.limits_from(args)["update_limit"] == 7
    assert bp.tolerances_from(args)["step_tolerance"] == 1e-12
    config = bp.default_config(solve=bp.solve_options_from(args))
    assert config["direction_order"] == 4 and config["pachner_moves"] is False
    with pytest.raises(SystemExit):
        bp.build_parser().parse_args(["run", "--direction-order", "11"])
    with pytest.raises(ValueError, match="unknown solve options"):
        bp.default_config(solve={"direction": 2})


def test_the_recursion_driver_carries_the_solve_options():
    args = R.build_parser().parse_args(
        ["run", "--direction-order", "2", "--no-pachner-moves"])
    assert bp.solve_options_from(args)["direction_order"] == 2
    config = R.default_config(solve=bp.solve_options_from(args))
    assert config["direction_order"] == 2
    assert config["pachner_moves"] is False


def test_a_level_as_a_base_complex_and_back():
    """`sheet_base` is sheet 0 of a level with its stored fields, and
    `base_fields` reads a base complex back as cells, squared lengths and
    links with the vertices in ascending order."""
    config = R.default_config(tetrahedra=2)
    cells, z, links, _ = R.level_zero(config)
    spacetime, count = R.build_level(cells, z, links)
    base = R.sheet_base(spacetime, count)
    assert cs.complex_counts(base) == {"vertices": 5, "edges": 9, "cells": 2}
    stored = {(a, b): (length, phase)
              for a, b, length, phase in cs.edge_fields(spacetime)}
    for a, b, length, phase in cs.edge_fields(base):
        assert stored[(a, b)] == (length, phase)
    out_cells, out_z, out_links, relabel = R.base_fields(base)
    assert out_cells == sorted(sorted(c) for c in cells)
    assert relabel == {v: v for v in range(5)}
    assert set(out_z) == set(z)
    for edge in z:
        assert out_z[edge] == pytest.approx(z[edge], rel=1e-15)
        assert out_links[edge] == pytest.approx(links[edge], rel=1e-14)
