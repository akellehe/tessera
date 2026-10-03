# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The declared options of the cell solve and its records
(`tessera.drivers.cell_solve`): a pinned region, a spectral-moment
stiffness, the places of a band on a complex a Pachner move changed, and the
record of a drive an error or a declared time ended.

Terms used below:

* the *base* is one tetrahedron on vertices 0..3 at squared edge length 8
  with the unit-monopole connection (`baryon_poles.build_base`), its link
  phases displaced by a seeded amount where a test needs a point away from
  stationarity; the *system* is the stationarity system of the joint action
  on its three-sheeted support (`cell_solve.GeometricSystem`), one squared
  length and one link per base edge: twelve variables;
* a *coordinate* is one base edge's squared length and link; a pinned region
  (`MultiCobordism.declare_pinned_region`) *holds* the coordinates of the
  edges with both ends in it (`cell_solve.held_coordinates`);
* the *step* is the minimum-norm least-squares solution d of J d = -R, with
  J the closed-form Jacobian of the residual R; with held coordinates it is
  taken over the steps that vanish on them (`cell_solve.HeldLinearization`);
* the *stiffness* is the spectral-moment stiffness of the base complex about
  the complex it was declared on (`HodgeLaplacian.spectralMomentStiffness`),
  a term of the action whose derivatives in the squared lengths and the link
  increments are `cell_solve.moment_stiffness_derivatives`;
* the *places* of a band are the positions of its modes in the ordered
  spectrum of the complex it is read on.

Differences of the stiffness are used here as an oracle for its closed-form
derivatives; nothing in the library or the drivers uses one.
"""
import cmath
import math

import numpy as np
import pytest

import tessera as T
from tessera import cobordism as cob
from tessera.drivers import baryon_poles as bp
from tessera.drivers import cell_solve as cs
from tests.drivers import test_cell_solve_python as tc


def _config(**declared):
    config = bp.default_config(kappas=[1.0], betas=[1.0])
    config["held_sectors"] = []
    config.update(declared)
    return config


def _displaced_base(seed=3, size=0.2, lengths=0.0, moduli=0.0):
    """The base with every link phase displaced by a seeded normal amount
    of scale ``size``; with ``lengths``, every squared length scaled by one
    plus a seeded normal amount of that scale; with ``moduli``, the
    logarithm of every link's modulus displaced by a seeded normal amount
    of that scale, so that the links leave the unit circle."""
    base = bp.build_base(8.0)
    rng = np.random.default_rng(seed)
    for edge in base.getEdgeList().toVector():
        edge.setPhase(edge.getPhase() + size * rng.normal())
        if lengths:
            edge.setLength(cmath.sqrt(
                edge.getLength() ** 2 * (1.0 + lengths * rng.normal())))
        if moduli:
            edge.setPhase(edge.getPhase() - 1j * moduli * rng.normal())
    return base


def _system(base, config, sheets=bp.SHEETS):
    """The stationarity system of the joint action on the sheeted support
    of ``base``, with the config's held sectors."""
    host = cs.sheeted_support(base, sheets)

    def declare(spacetime):
        return bp.action_declaration(spacetime, 1.0, 1.0,
                                     config["regge_hinges"])

    def geometry_of(support):
        return bp.support_geometry(config, support, host)

    return cs.GeometricSystem(declare, geometry_of, sheets)


def _held_face_sector(base, face=(0, 1, 2)):
    """One held sector on the three-sheeted host of ``base``: the given
    base face on every sheet, with the monopole number it reads there."""
    host = cs.sheeted_support(base, bp.SHEETS)
    links = {}
    for a, b, _, phase in cs.edge_fields(base):
        links[(a, b)] = cmath.exp(1j * phase)
        links[(b, a)] = cmath.exp(-1j * phase)
    a, b, c = face
    holonomy = links[(a, b)] * links[(b, c)] * links[(c, a)]
    sector = cob.HeldMonopoleSector()
    sector.faces = [[t * host.count + v for v in face]
                    for t in range(bp.SHEETS)]
    sector.monopole_number = int(round(
        bp.SHEETS * cmath.phase(holonomy) / (2.0 * math.pi)))
    return sector


def _scales(point):
    """The scales of the system's variables in its linear solves
    (`HolomorphicRelaxation.variable_scales`), or ones where the library
    has no such method."""
    scales = getattr(point.relaxation, "variable_scales", None)
    size = point.relaxation.variable_count()
    return (np.asarray(scales(), dtype=float) if scales is not None
            else np.ones(size))


def _add_a_vertex(base):
    """The 1-4 move on the first cell of ``base``, in place."""
    move = T.AddMove(base, 0, False, T.PachnerMode.PreGeometric, False)
    assert move.propose() and move.apply()


# ------------------------------------------------------------ a pinned region


def test_the_pinned_vertices_reach_the_node_as_a_region():
    """`node_configuration` declares the config's pinned vertices on the
    node as one region, and the engine then holds the edges inside it."""
    base = bp.build_base()
    node = cs.cell_node(base, cs.StationarityObjective(
        _system(base, _config())))
    bp.node_configuration(_config(pinned_vertices=[0, 1, 2]))(node)
    assert set(node.pinned_vertices()) == {0, 1, 2}
    assert [(name, set(vertices))
            for name, vertices in node.pinned_regions()] == \
        [("pinned", {0, 1, 2})]
    assert node.edge_is_pinned(0, 1) and node.edge_is_pinned(1, 2)
    assert not node.edge_is_pinned(0, 3)

    unpinned = cs.cell_node(base, cs.StationarityObjective(
        _system(base, _config())))
    bp.node_configuration(_config())(unpinned)
    assert unpinned.pinned_regions() == []


def test_a_region_holds_the_coordinates_of_the_edges_inside_it():
    """An edge is held when both its ends lie in one region, which is the
    engine's rule (`MultiCobordism.edge_is_pinned`); one pinned end, or two
    ends in two regions, leave it free."""
    base = bp.build_base()
    fields = cs.edge_fields(base)
    classes = list(range(len(fields)))
    index = {tuple(sorted((a, b))): k for k, (a, b, _, _) in
             enumerate(fields)}
    assert cs.held_coordinates(base, [], classes) == []
    assert cs.held_coordinates(base, [{0}], classes) == []
    assert cs.held_coordinates(base, [{0, 1}], classes) == [index[(0, 1)]]
    assert cs.held_coordinates(base, [{0, 1, 2}], classes) == sorted(
        index[e] for e in ((0, 1), (0, 2), (1, 2)))
    assert cs.held_coordinates(base, [{0, 1}, {2, 3}], classes) == sorted(
        (index[(0, 1)], index[(2, 3)]))
    assert cs.held_coordinates(base, [{0}, {1}], classes) == []
    node = cs.cell_node(base, cs.StationarityObjective(
        _system(base, _config())))
    node.declare_pinned_region("a", {0, 1, 2})
    for (a, b), k in index.items():
        assert node.edge_is_pinned(a, b) == (
            k in cs.held_coordinates(base, [{0, 1, 2}], classes))
    # edges that share a coordinate are held together
    shared = [0, 0, 1, 2, 3, 4]
    assert cs.held_coordinates(base, [{fields[1][0], fields[1][1]}],
                               shared) == [0]


def test_without_a_held_coordinate_the_restricted_step_is_the_step():
    """Over no held coordinate `HeldLinearization` poses the library's own
    least-squares problem, for the residual and for any other right-hand
    side: its solution has the image the library's has, J d, which is the
    part of the right-hand side the system can reach, and its step is not
    longer than a rounding above the library's minimum-norm one."""
    base = _displaced_base(lengths=0.1)
    point = _system(base, _config()).point(base)
    library = point.relaxation.linearization()
    restricted = cs.HeldLinearization(point, [])
    size = point.relaxation.variable_count()
    jacobian = np.asarray(point.relaxation.jacobian()).reshape(size, size)
    scale = _scales(point)
    assert restricted.newton_step.residual_norm == pytest.approx(
        library.newton_step.residual_norm, rel=1e-14)
    assert not restricted.newton_step.constrained
    assert restricted.held_variables == []
    rng = np.random.default_rng(1)
    target = rng.normal(size=size) + 1j * rng.normal(size=size)
    for mine, theirs, reached in (
            (np.asarray(restricted.newton_step.step),
             np.asarray(library.newton_step.step),
             np.asarray(point.relaxation.residual())),
            (restricted.solve(target), np.asarray(library.solve(list(target))),
             target)):
        np.testing.assert_allclose(
            scale * (jacobian @ mine), scale * (jacobian @ theirs),
            atol=1e-9 * np.linalg.norm(scale * reached))


def test_a_restricted_step_vanishes_on_held_coordinates_and_solves_the_rest():
    """With coordinates held the step has no component on their squared
    lengths and links, exactly, and on the free variables it is the
    least-squares solution: the residual of the linearized system left
    over is orthogonal to every free column, in the scaled variables the
    solve is taken in."""
    base = _displaced_base(lengths=0.1)
    point = _system(base, _config()).point(base)
    held = cs.held_coordinates(base, [{0, 1, 2}], point.classes)
    assert len(held) == 3
    linearization = cs.HeldLinearization(point, held)
    step = np.asarray(linearization.newton_step.step)
    held_variables = held + [point.count + c for c in held]
    assert list(linearization.held_variables) == held_variables
    assert np.all(step[held_variables] == 0.0)
    free = [k for k in range(len(step)) if k not in held_variables]
    assert np.linalg.norm(step[free]) > 0.0

    size = point.relaxation.variable_count()
    jacobian = np.asarray(point.relaxation.jacobian()).reshape(size, size)
    residual = np.asarray(point.relaxation.residual())
    scale = _scales(point)
    scaled = scale[:, None] * jacobian * scale[None, :]
    left_over = scale * (jacobian @ step + residual)
    normal = scaled[:, free].conj().T @ left_over
    assert np.linalg.norm(normal) <= 1e-9 * (
        np.linalg.norm(scaled) * np.linalg.norm(scale * residual))
    # holding coordinates cannot leave less of the linearized system than
    # the unrestricted step does
    unrestricted = np.asarray(point.relaxation.linearization()
                              .newton_step.step)
    assert np.linalg.norm(left_over) >= np.linalg.norm(
        scale * (jacobian @ unrestricted + residual)) - 1e-12
    assert linearization.newton_step.linear_residual == pytest.approx(
        np.linalg.norm(jacobian @ step + residual)
        / np.linalg.norm(residual), rel=1e-12)


def test_a_restricted_step_keeps_the_held_moduli():
    """With a monopole sector held and a region pinned, the real part of
    the step's link block changes no held modulus: the held faces'
    coboundary annihilates it, and it vanishes on the pinned coordinates.
    Zeroing the pinned components of the unrestricted held step, which is
    what the engine does to the direction it is handed, does not keep them.
    The links here are off the unit circle, so that the step has a real
    link part at all."""
    base = _displaced_base(lengths=0.1, moduli=0.2)
    sector = _held_face_sector(base, (0, 1, 2))
    point = _system(base, _config(held_sectors=[sector])).point(base)
    held = cs.held_coordinates(base, [{0, 1}], point.classes)
    assert len(held) == 1
    coboundary = cs.HeldLinearization._coboundary(point, [sector])
    assert coboundary.shape == (bp.SHEETS, point.count)
    assert np.all(coboundary[0] == coboundary[1])
    assert sorted(np.abs(coboundary[0])) == [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]

    linearization = cs.HeldLinearization(point, held)
    assert linearization.newton_step.constrained
    step = np.asarray(linearization.newton_step.step)
    links = step[point.count:2 * point.count]
    assert step[held[0]] == 0.0 and links[held[0]] == 0.0
    assert np.linalg.norm(links.real) > 1e-3 * np.linalg.norm(links)
    assert np.max(np.abs(coboundary @ links.real)) <= 1e-12 * np.linalg.norm(
        links)

    library = np.asarray(point.relaxation.linearization().newton_step.step)
    cut = library[point.count:2 * point.count].copy()
    assert np.max(np.abs(coboundary @ cut.real)) <= 1e-12 * np.linalg.norm(
        cut)
    cut[held[0]] = 0.0
    assert np.max(np.abs(coboundary @ cut.real)) > 1e-6 * np.linalg.norm(cut)


def test_a_pinned_drive_leaves_the_pinned_edge_and_converges_as_a_step_does():
    """The relaxation of the displaced base with vertices 0 and 1 pinned
    through the driver's configuration: the edge between them keeps its
    squared length and its link bit for bit, every proposal names the held
    coordinate, the residual norm falls at every accepted update, and at
    some update it falls by more than a factor of a thousand, as the
    updates of a Newton step do (a step that ignores the pin is cut by the
    engine and halves the residual at every update)."""
    config = _config(pinned_vertices=[0, 1])
    base = _displaced_base()
    before = {(a, b): (length, phase)
              for a, b, length, phase in cs.edge_fields(base)}
    drive = cs.solve(base, _system(base, config), moves=False,
                     configure=bp.node_configuration(config))
    after = {(a, b): (length, phase)
             for a, b, length, phase in cs.edge_fields(drive["spacetime"])}
    (pinned,) = [edge for edge in before if set(edge) == {0, 1}]
    assert after[pinned] == before[pinned]
    assert any(after[edge] != before[edge] for edge in before)
    updates = drive["objective"].updates
    held = cs.held_coordinates(base, [{0, 1}], list(range(6)))
    assert all(update["held_coordinates"] == held for update in updates)
    trace = drive["trace"]
    assert drive["accepted_updates"] >= 2
    assert all(b < a for a, b in zip(trace, trace[1:]))
    assert min(b / a for a, b in zip(trace, trace[1:])) < 1e-3
    assert trace[-1] < 1e-9 * trace[0]
    assert drive["stop_reason"] == cs.STOP_STATIONARY


# ------------------------------------------------- a spectral-moment stiffness


def _stiffness_case():
    """The base displaced in both fields, with the moments of the undisplaced
    base as the stiffness reference at two orders."""
    coefficients = [1.0, 0.5]
    reference = list(cob.HodgeLaplacian(bp.build_base())
                     .localSpectralMoments(1, len(coefficients)))
    base = _displaced_base(lengths=0.1)
    rng = np.random.default_rng(8)
    for edge in base.getEdgeList().toVector():   # complex in both fields
        edge.setLength(cmath.sqrt(edge.getLength() ** 2
                                  * (1.0 + 0.05j * rng.normal())))
        edge.setPhase(edge.getPhase() + 0.05j * rng.normal())
    return base, reference, coefficients


def test_the_stiffness_derivatives_are_those_of_the_stiffness():
    """The closed-form gradient of the stiffness in the squared lengths and
    the link increments is the derivative of its value, and the closed-form
    Hessian the derivative of the gradient: each agrees with a central
    difference to the difference's own truncation, which falls with the
    square of the displacement. The Hessian is symmetric."""
    base, reference, coefficients = _stiffness_case()
    edges = base.getEdgeList().toVector()
    count = len(edges)
    start = [(edge.getLength(), edge.getPhase()) for edge in edges]

    def displace(coordinate, amount):
        for edge, (length, phase) in zip(edges, start):
            edge.setLength(length)
            edge.setPhase(phase)
        edge = edges[coordinate % count]
        length, phase = start[coordinate % count]
        if coordinate < count:
            edge.setLength(cmath.sqrt(length * length + amount))
        else:   # U -> U exp(amount) on the stored orientation
            edge.setPhase(phase - 1j * amount)

    def value():
        return complex(cob.HodgeLaplacian(base).spectralMomentStiffness(
            1, reference, coefficients))

    def gradient():
        return cs.moment_stiffness_derivatives(base, 1, reference,
                                               coefficients, False)[0]

    closed_gradient, closed_hessian = cs.moment_stiffness_derivatives(
        base, 1, reference, coefficients)
    assert np.linalg.norm(closed_gradient[count:]) > 0.0
    errors = []
    for amount in (1e-4, 1e-5):
        first = np.zeros(2 * count, dtype=complex)
        second = np.zeros((2 * count, 2 * count), dtype=complex)
        for coordinate in range(2 * count):
            displace(coordinate, amount)
            ahead, ahead_gradient = value(), gradient()
            displace(coordinate, -amount)
            behind, behind_gradient = value(), gradient()
            first[coordinate] = (ahead - behind) / (2.0 * amount)
            second[:, coordinate] = (ahead_gradient - behind_gradient) / (
                2.0 * amount)
        displace(0, 0.0)
        errors.append((
            np.linalg.norm(first - closed_gradient)
            / np.linalg.norm(closed_gradient),
            np.linalg.norm(second - closed_hessian)
            / np.linalg.norm(closed_hessian)))
    assert errors[0][0] < 1e-4 and errors[0][1] < 1e-4
    assert errors[1][0] < errors[0][0] / 10.0
    assert errors[1][1] < errors[0][1] / 10.0
    assert np.linalg.norm(closed_hessian - closed_hessian.T) <= \
        1e-12 * np.linalg.norm(closed_hessian)


def test_the_stiffness_length_derivatives_are_the_librarys():
    """On the squared lengths the closed forms are the library's own
    gradient and Hessian-vector product; at the complex the reference was
    read on the stiffness and its gradient vanish and the Hessian is the
    sum of squares it is there, symmetric."""
    base, reference, coefficients = _stiffness_case()
    count = len(cs.edge_fields(base))
    gradient, hessian = cs.moment_stiffness_derivatives(
        base, 1, reference, coefficients)
    hodge = cob.HodgeLaplacian(base)
    library = np.asarray(hodge.spectralMomentStiffnessGradient(
        1, reference, coefficients))
    np.testing.assert_allclose(gradient[:count], library,
                               atol=1e-12 * np.linalg.norm(library))
    product = np.stack([np.asarray(hodge.spectralMomentStiffnessHessianProduct(
        1, reference, coefficients,
        [1.0 + 0j if k == f else 0j for k in range(count)]))
        for f in range(count)], axis=1)
    np.testing.assert_allclose(hessian[:count, :count], product,
                               atol=1e-12 * np.linalg.norm(product))

    carrier = bp.build_base()
    at_carrier, hessian_there = cs.moment_stiffness_derivatives(
        carrier, 1, reference, coefficients)
    assert np.linalg.norm(at_carrier) <= 1e-12 * np.linalg.norm(hessian_there)
    assert np.linalg.norm(hessian_there[count:, count:]) > 0.0
    assert np.linalg.norm(hessian_there - hessian_there.T) <= \
        1e-9 * np.linalg.norm(hessian_there)


def _stiffness_context(base, reference, coefficients, weight):
    context = cob.ObjectiveContext()
    context.spacetime = base
    context.moment_stiffness_weight = weight
    context.moment_stiffness_degrees = [1]
    context.moment_stiffness_coefficients = list(coefficients)
    context.moment_stiffness_reference = [list(reference)]
    return context


def test_a_declared_stiffness_enters_both_blocks_of_the_system():
    """The term the objective adds for a declared stiffness: its gradient,
    once per sheet, on the length and the link equations of the base edges'
    coordinates, and its Hessian on all four blocks of the Jacobian. The
    score of the complex is the norm of the residual with that term."""
    base, reference, coefficients = _stiffness_case()
    weight = 1e-6
    system = _system(base, _config())
    objective = cs.StationarityObjective(system)
    point = system.point(base)
    context = _stiffness_context(base, reference, coefficients, weight)
    residual, jacobian = objective._stiffness(point, context, True)
    gradient, hessian = cs.moment_stiffness_derivatives(
        base, 1, reference, coefficients)
    size = point.relaxation.variable_count()
    assert size == 2 * point.count == len(gradient)
    np.testing.assert_allclose(residual, weight * bp.SHEETS * gradient,
                               rtol=1e-13, atol=0.0)
    np.testing.assert_allclose(
        np.asarray(jacobian).reshape(size, size),
        weight * bp.SHEETS * hessian, rtol=1e-13, atol=0.0)
    assert np.linalg.norm(np.asarray(residual)[point.count:]) > 0.0
    scored = objective.terms(context).joint_action_stationarity
    assert scored == pytest.approx(np.linalg.norm(
        np.asarray(point.relaxation.residual()) + np.asarray(residual)),
        rel=1e-13)
    only, none = objective._stiffness(point, context, False)
    assert none == []
    np.testing.assert_allclose(only, residual, rtol=1e-13, atol=0.0)
    assert objective._stiffness(point, _stiffness_context(
        base, reference, coefficients, 0.0), True) == ([], [])


def test_a_drive_descends_the_residual_of_the_action_with_its_stiffness():
    """A stiffness declared through the driver's configuration is declared
    on the node about the starting complex, and the drive descends the
    norm of the residual of the action with it: the trace falls at every
    accepted update, and its last entry is the norm, at the end point, of
    the system's residual plus the stiffness's gradient there."""
    coefficients = [1.0, 0.5]
    config = _config(moment_stiffness_weight=1e-6,
                     moment_stiffness_coefficients=coefficients)
    base = _displaced_base()
    reference = list(cob.HodgeLaplacian(base).localSpectralMoments(1, 2))
    system = _system(base, config)
    drive = cs.solve(base, system, moves=False,
                     configure=bp.node_configuration(config))
    node = drive["node"]
    assert node.moment_stiffness_weight == 1e-6
    assert list(node.moment_stiffness_coefficients) == coefficients
    trace = drive["trace"]
    assert drive["accepted_updates"] >= 2
    assert all(b < a for a, b in zip(trace, trace[1:]))
    assert trace[-1] < 1e-3 * trace[0]
    end = drive["spacetime"]
    point = system.point(end)
    added, _ = drive["objective"]._stiffness(point, _stiffness_context(
        end, reference, coefficients, 1e-6), False)
    assert np.linalg.norm(added) > 0.0
    assert trace[-1] == pytest.approx(np.linalg.norm(
        np.asarray(point.relaxation.residual()) + np.asarray(added)),
        rel=1e-9)


def test_a_stiffness_has_no_value_on_a_complex_with_other_cells():
    """The stiffness reference is the moments of the complex it was
    declared on. On a complex with another number of edges the stiffness has
    no value: the complex scores infinite and the reason says so."""
    base = bp.build_base()
    coefficients = [1.0, 0.5]
    reference = list(cob.HodgeLaplacian(base).localSpectralMoments(1, 2))
    system = _system(base, _config())
    objective = cs.StationarityObjective(system)
    _add_a_vertex(base)
    with pytest.raises(ValueError, match="the spectral-moment stiffness has "
                                         "no value on this complex"):
        cs.moment_stiffness_derivatives(base, 1, reference, coefficients)
    context = _stiffness_context(base, reference, coefficients, 1e-6)
    assert objective.terms(context).joint_action_stationarity == math.inf
    (reason,) = objective.undefined
    assert reason.startswith(
        "the spectral-moment stiffness has no value on this complex: its "
        "reference, the moments of the complex it was declared on, has 12 "
        "entries, and the degree-1 operator here has 10 cells at 2 orders")
    assert cs.undefined_reasons(objective.undefined) == {reason: 1}


# the reason a complex one Pachner move made from a tetrahedron has no
# stiffness, the reference being the tetrahedron's six edges at two orders
NO_STIFFNESS_AFTER_A_MOVE = (
    "the spectral-moment stiffness has no value on this complex: its "
    "reference, the moments of the complex it was declared on, has 12 "
    "entries, and the degree-1 operator here has 10 cells at 2 orders")

STIFFNESS_EXCLUDES_MOVES = (
    "the stiffness reference is the moments of the complex the stiffness "
    "was declared on, and every Pachner move of a tetrahedral complex "
    "changes its number of edges, so with a stiffness declared every "
    "candidate move scores infinite; fix(drivers): a declared moment "
    "stiffness enters the cell solve without its link derivatives and "
    "excludes every Pachner move, "
    "https://github.com/akellehe/tessera/issues/1370")


def _drive_with_moves(monkeypatch, weight):
    """The drive of the base as built, which is stationary (residual norm
    zero), with its Pachner moves and, with a nonzero ``weight``, a
    stiffness at two orders declared through the driver's configuration.
    Returns the drive's record and, by the number of base edges of the
    complex scored, every score the engine asked for."""
    scores = {}
    terms = cs.StationarityObjective.terms

    def scored(self, context):
        out = terms(self, context)
        scores.setdefault(len(cs.edge_fields(context.spacetime)), []).append(
            out.joint_action_stationarity)
        return out

    monkeypatch.setattr(cs.StationarityObjective, "terms", scored)
    config = _config(moment_stiffness_weight=weight,
                     moment_stiffness_coefficients=[1.0, 0.5] if weight
                     else [])
    base = bp.build_base(8.0)
    drive = cs.solve(base, _system(base, config), moves=True,
                     configure=bp.node_configuration(config))
    monkeypatch.setattr(cs.StationarityObjective, "terms", terms)
    return drive, scores


def test_with_a_stiffness_the_move_of_a_tetrahedron_has_no_value(
        monkeypatch):
    """Ticket #1370 as it stands. The one candidate move of the base, the
    1-4 move, takes its six edges to ten. Without a stiffness the engine
    scores it 32.2 and the drive commits nothing, the base being stationary;
    with a stiffness declared the candidate has no stiffness, scores
    infinite, and the drive's record keeps the reason. The drive is the
    same otherwise: it commits no move and ends where it began."""
    plain, plain_scores = _drive_with_moves(monkeypatch, 0.0)
    assert plain["undefined_reasons"] == {}
    assert plain_scores[10] == [pytest.approx(32.21, rel=1e-3)]
    assert plain["objective"].stiffness_cells == {}
    drive, scores = _drive_with_moves(monkeypatch, 1e-6)
    assert drive["node"].moment_stiffness_weight == 1e-6
    assert drive["objective"].stiffness_cells == {
        1: cs.stiffness_cells(bp.build_base(8.0), 1)}
    assert scores[10] == [math.inf]
    assert drive["undefined_reasons"] == {NO_STIFFNESS_AFTER_A_MOVE: 1}
    assert all(score == 0.0 for score in scores[6])
    for record in (plain, drive):
        assert record["moves_committed"] == 0
        assert record["complex_after"] == record["complex_before"]
        assert record["trace"] == [0.0]


@pytest.mark.xfail(strict=True, reason=STIFFNESS_EXCLUDES_MOVES)
def test_with_a_stiffness_a_candidate_move_is_scored(monkeypatch):
    """A declared stiffness excludes no Pachner move: the base's candidate
    move has a residual with the stiffness as it has without one."""
    drive, scores = _drive_with_moves(monkeypatch, 1e-6)
    assert drive["undefined_reasons"] == {}
    assert all(math.isfinite(score) for score in scores[10])


def _four_cells(cells, phases):
    """A complex of the given cells on vertices 0..5 with squared length 8
    on every edge and the given phase on each ascending edge."""
    complex_ = T.Spacetime.fromVertexTuples(3, cells, 1.0, 0.0)
    for edge in complex_.getEdgeList().toVector():
        a = int(edge.getSource().getId())
        b = int(edge.getTarget().getId())
        edge.setLength(cmath.sqrt(8.0))
        phase = phases[(min(a, b), max(a, b))]
        edge.setPhase(phase if a < b else -phase)
    return complex_


def test_a_stiffness_has_no_value_on_other_edges_of_its_number():
    """The stiffness reference is of the cells of the complex it was
    declared on. Three cells around the edge (0, 1) and a fourth on the face
    (0, 2, 3): the 3-2 move on (0, 1) and then the 2-3 move on (0, 2, 3)
    trade the edge (0, 1) for (4, 5), thirteen edges before and after. The
    cells of the reference are the operator's, in its order, which is not
    the order of the edge list: the declared complex built from its cells in
    another order has the declared cells and the stiffness zero. On the
    complex the two moves make, which has the declared number of edges and
    other edges, the stiffness has no value when the declared cells are
    given, and the objective told them scores it infinite with the reason.
    Without the cells the reference is read place by place."""
    rng = np.random.default_rng(5)
    phases = {(a, b): 0.1 * rng.normal()
              for a in range(6) for b in range(a + 1, 6)}
    cells = [[0, 1, 2, 3], [0, 1, 3, 4], [0, 1, 2, 4], [0, 2, 3, 5]]
    coefficients = [1.0, 0.5]
    declared = _four_cells(cells, phases)
    reference = list(cob.HodgeLaplacian(declared).localSpectralMoments(1, 2))
    declared_cells = cs.stiffness_cells(declared, 1)
    assert len(declared_cells) == 13 and declared_cells == sorted(
        declared_cells)
    reordered = _four_cells(cells[::-1], phases)
    assert [e[:2] for e in cs.edge_fields(reordered)] != \
        [e[:2] for e in cs.edge_fields(declared)]
    assert cs.stiffness_cells(reordered, 1) == declared_cells
    gradient, _ = cs.moment_stiffness_derivatives(
        reordered, 1, reference, coefficients, False, declared_cells)
    assert np.linalg.norm(gradient) == 0.0
    assert cob.HodgeLaplacian(reordered).spectralMomentStiffness(
        1, reference, coefficients) == 0.0

    moved = _four_cells([[1, 2, 3, 4], [0, 2, 4, 5], [0, 3, 4, 5],
                         [2, 3, 4, 5]], phases)
    edges = cs.stiffness_cells(moved, 1)
    assert len(edges) == 13 and (0, 1) not in edges and (4, 5) in edges
    reason = ("the spectral-moment stiffness has no value on this complex: "
              "its reference is of the 13 degree-1 cells of the complex it "
              "was declared on, and 1 of those are not cells here")
    with pytest.raises(ValueError) as raised:
        cs.moment_stiffness_derivatives(moved, 1, reference, coefficients,
                                        False, declared_cells)
    assert str(raised.value) == reason
    objective = cs.StationarityObjective(_system(declared, _config()))
    objective.declare_stiffness(declared, [1])
    assert objective.stiffness_cells == {1: declared_cells}
    scores = [objective.terms(_stiffness_context(
        complex_, reference, coefficients, 1e-6)).joint_action_stationarity
        for complex_ in (declared, reordered, moved)]
    assert math.isfinite(scores[0])
    assert scores[1] == pytest.approx(scores[0], rel=1e-13)
    assert scores[2] == math.inf
    assert objective.undefined == [reason]

    # without the cells, the reference is read place by place
    gradient, _ = cs.moment_stiffness_derivatives(moved, 1, reference,
                                                  coefficients, False)
    assert np.linalg.norm(gradient) > 0.0
    moments = np.asarray(cob.HodgeLaplacian(moved).localSpectralMoments(1, 2))
    deviation = (moments - np.asarray(reference)).reshape(13, 2)
    by_place = 0.5 * np.sum(np.asarray(coefficients) * deviation ** 2)
    assert cob.HodgeLaplacian(moved).spectralMomentStiffness(
        1, reference, coefficients) == pytest.approx(by_place, rel=1e-13)


def test_the_level_records_the_residual_its_drive_minimised():
    """The level's residual, and with it ``converged``, is the norm its
    drive minimised. On the level of the displaced base with a stiffness of
    weight 1e-6 the drive ends at 1.3e-15, the norm of the residual of the
    action with the stiffness, where the action's residual alone is 2.1e-7;
    the record says the stiffness is in it. Without a stiffness the record's
    residual is the action's, and says so."""
    from tessera.drivers import recursion as R

    coefficients = [1.0, 0.5]
    for weight in (1e-6, 0.0):
        cells, z, links, _ = R.base_fields(_displaced_base())
        spacetime, count = R.build_level(cells, z, links)
        reference = list(cob.HodgeLaplacian(R.sheet_base(
            spacetime, count)).localSpectralMoments(1, 2))
        config = R.default_config()
        config.update({"moment_stiffness_weight": weight,
                       "moment_stiffness_coefficients":
                       coefficients if weight else []})
        record = R.relax_level(spacetime, config, [], count=count)
        assert record["moved_base"] is None
        end = R.sheet_base(spacetime, count)
        system, _ = R.level_system(end, config, [], R.SHEETS)
        point = system.point(end)
        action = np.asarray(point.relaxation.residual())
        added, _ = cs.StationarityObjective(system)._stiffness(
            point, _stiffness_context(end, reference, coefficients, weight),
            False)
        total = np.linalg.norm(action + np.asarray(added)) if weight \
            else np.linalg.norm(action)
        assert record["residual_includes_stiffness"] is bool(weight)
        assert record["residual"] == pytest.approx(total, rel=1e-9)
        assert record["residual"] < 1e-14
        assert record["converged"] == bp.solve_converged(
            record["residual"],
            bp.declared_tolerance(config, "step_tolerance"))
        if weight:
            assert np.linalg.norm(action) > 1e-8
        else:
            assert record["residual"] == np.linalg.norm(action)


def test_the_level_and_the_content_record_the_reasons():
    """The records the drivers write keep, beside the number of scored
    complexes without a residual, the reasons (`cell_solve.undefined_reasons`
    of the drive's objective). With a stiffness declared, the level of the
    base as built records its one candidate move as without a stiffness;
    so does the content (0, 3, 0) of the recorded run's first cell, whose
    record counts the reasons of its line search's trials with it."""
    from tessera.drivers import recursion as R
    from tests.drivers import _recursion_run_2026_09_23 as RUN

    stiffness = {"moment_stiffness_weight": 1e-6,
                 "moment_stiffness_coefficients": [1.0, 0.5]}
    cells, z, links, _ = R.base_fields(bp.build_base(8.0))
    spacetime, count = R.build_level(cells, z, links)
    config = R.default_config()
    config.update(stiffness)
    level = R.relax_level(spacetime, config, [], count=count)
    assert level["undefined_reasons"] == {NO_STIFFNESS_AFTER_A_MOVE: 1}
    assert level["undefined_points"] == 1 and level["moves_committed"] == 0

    config = bp.default_config([1.0], [1.0], selected_contents=[(0, 3, 0)])
    config["host_cell"] = RUN.HOST_CELLS[(0, 1, 2, 3)]
    config["held_sectors"] = R.held_sectors([[0, 1, 2, 3]], [1], 4)
    config.update(stiffness)
    _, _, report, drive = bp.relax_content((0, 3, 0), 1.0, 1.0, config)
    content = bp.relaxation_record(report, drive)
    assert content["undefined_reasons"] == drive["undefined_reasons"]
    assert content["undefined_reasons"][NO_STIFFNESS_AFTER_A_MOVE] == 1
    assert sum(content["undefined_reasons"].values()) == \
        content["undefined_points"]
    assert content["moves_committed"] == 0


# ---------------------------------------------- band places across a move


@pytest.mark.parametrize("band_reference", cs.BAND_REFERENCES)
def test_a_band_is_at_its_places_on_the_complex_a_move_made(band_reference):
    """After the 1-4 move the content's band is followed from the same
    reference projectors as with the places of the host carried over, so it
    takes the same modes; its declared places are the places it holds on the
    moved complex, so the read does not report a crossing there. With the
    host's places carried over the same read reports one, comparing places
    among 18 modes with places among 30."""
    base, _, system = tc._content_system()
    system.band_reference = band_reference
    host_places = [list(band.declared_positions)
                   for band in system.iterate(base)[0].bands]
    _add_a_vertex(base)
    field, action, support = system._field(base)
    carried, cells = system._reference(support, action)
    assert len(cells) == 30 and len(system.reference_cells) == 18
    with_host_places = field.iterate(carried, [])
    assert [list(band.declared_positions)
            for band in with_host_places.bands] == host_places
    assert with_host_places.band_crossing

    step = system.accept(base)
    assert not step.band_crossing
    for band, other in zip(step.bands, with_host_places.bands):
        assert list(band.positions) == list(band.declared_positions)
        assert list(band.modes) == list(other.modes)
        assert list(band.positions) == list(other.positions)
        assert band.rank == other.rank
        np.testing.assert_array_equal(np.asarray(band.projector),
                                      np.asarray(other.projector))
    # a second read of the same complex, and its end-point read, keep them
    assert not system.accept(base).band_crossing
    assert system.read(base)[2].band_crossing_iterates == 0


def test_the_host_keeps_the_places_it_declares():
    """On the host itself the places are the declared ones, as they are
    without any move."""
    base, _, system = tc._content_system()
    step = system.accept(base)
    assert not step.band_crossing
    key = tuple(system.reference_cells)
    assert system._places[key] == [list(band.declared_positions)
                                   for band in step.bands]


# ------------------------------------------------- the record of a drive


class _Clock:
    """A clock that advances one second at every reading."""

    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        self.now += 1.0
        return self.now


def test_a_drive_a_declared_time_ended_keeps_its_trace(monkeypatch):
    """A drive ended by a declared time has accepted updates behind it: the
    record's trace is the residual norm at every point a step was proposed
    from and at the last accepted point, where the base is left; it is the
    trace of the unlimited drive up to there, and the accepted updates are
    counted from it. The clock here advances one second per reading, so the
    declared time is a number of scorings and proposals."""
    base = _displaced_base()
    system = _system(base, _config())
    whole = cs.solve(_displaced_base(), system, moves=False)
    assert whole["accepted_updates"] >= 3

    clock = _Clock()
    monkeypatch.setattr(cs.time, "monotonic", clock.monotonic)
    drive = cs.solve(base, system, moves=False, time_limit_seconds=5.5)
    assert drive["stop_reason"] == cs.STOP_DECLARED_LIMIT
    assert "the declared time limit of 5.5 seconds" in drive["stop_detail"]
    updates = drive["objective"].updates
    assert 1 <= len(updates) < len(whole["objective"].updates)
    assert 2 <= len(drive["trace"]) < len(whole["trace"])
    assert drive["trace"] == pytest.approx(
        whole["trace"][:len(drive["trace"])], rel=1e-12)
    assert drive["accepted_updates"] == len(drive["trace"]) - 1
    end = _system(base, _config()).point(drive["spacetime"])
    assert np.linalg.norm(end.relaxation.residual()) == pytest.approx(
        drive["trace"][-1], rel=1e-12)


def test_the_proposed_trace_has_one_entry_per_distinct_point():
    """Proposals repeated at one point, which the engine makes when a line
    search accepts no trial, are one entry of the trace; the engine's own
    trace of a finished drive is the same list."""
    base = _displaced_base()
    drive = cs.solve(base, _system(base, _config()), moves=False)
    objective = drive["objective"]
    assert objective.proposed_trace() == pytest.approx(drive["trace"],
                                                       rel=1e-12)
    assert objective.proposed_trace(drive["spacetime"]) == \
        objective.proposed_trace()
    assert len(objective.updates) >= len(drive["trace"])
    assert drive["undefined_reasons"] == cs.undefined_reasons(
        objective.undefined)


def test_an_error_of_a_scoring_is_a_complex_without_a_residual():
    """A complex whose scoring raises an error of any kind scores infinite,
    with the reason kept and, for an error outside the kinds the library
    raises by design, the name of its type; a declared limit is not such an
    error and leaves the scoring."""
    base = bp.build_base()

    class Raises:
        def __init__(self, error):
            self.error = error

        def point(self, complex_):
            raise self.error

        def accept(self, complex_):
            return None

    for error, reason in (
            (KeyError("a cell"), "KeyError: 'a cell'"),
            (IndexError("no cell"), "IndexError: no cell"),
            (TypeError("a set"), "TypeError: a set"),
            (ValueError("no value here"), "no value here"),
            (RuntimeError("no band"), "no band"),
            (ZeroDivisionError("zero"), "zero")):
        objective = cs.StationarityObjective(Raises(error))
        node = cs.cell_node(base, objective)
        assert node.objective() == math.inf
        assert objective.undefined == [reason]
    objective = cs.StationarityObjective(
        Raises(cs.DeclaredLimitReached("the declared time")))
    with pytest.raises(cs.DeclaredLimitReached):
        objective.terms(cob.ObjectiveContext())


def test_the_reasons_of_scored_complexes_are_counted_in_the_record():
    """`undefined_reasons` counts each reason in the order first met."""
    assert cs.undefined_reasons([]) == {}
    assert cs.undefined_reasons(["a", "b", "a", "a"]) == {"a": 3, "b": 1}
    assert list(cs.undefined_reasons(["b", "a", "b"])) == ["b", "a"]


# ----------------------------------------------- the options as declared


def test_the_stiffness_and_the_pins_are_checked_where_they_are_declared():
    """What the engine requires of a stiffness when a node is configured is
    required of the options when they are declared: a weight and
    coefficients that are finite and not negative, and at least one
    coefficient with a weight that is not zero. Pinned vertices are vertex
    ids."""
    options = bp.declared_solve_options({
        "moment_stiffness_weight": 1e-6,
        "moment_stiffness_coefficients": (1.0, 0.5),
        "pinned_vertices": (0, 1)})
    assert options["moment_stiffness_coefficients"] == [1.0, 0.5]
    assert options["pinned_vertices"] == [0, 1]
    declared = bp.declared_solve_options()
    assert declared["moment_stiffness_weight"] == 0.0
    assert declared["moment_stiffness_coefficients"] == []
    assert declared["pinned_vertices"] == []
    with pytest.raises(ValueError, match="needs the coefficient of at least "
                                         "one moment"):
        bp.declared_solve_options({"moment_stiffness_weight": 1e-6})
    with pytest.raises(ValueError, match="finite and not negative"):
        bp.declared_solve_options({"moment_stiffness_weight": -1.0,
                                   "moment_stiffness_coefficients": [1.0]})
    with pytest.raises(ValueError, match="finite and not negative"):
        bp.declared_solve_options({"moment_stiffness_weight": math.inf,
                                   "moment_stiffness_coefficients": [1.0]})
    with pytest.raises(ValueError, match="coefficients of the moment "
                                         "stiffness are finite"):
        bp.declared_solve_options({"moment_stiffness_weight": 1.0,
                                   "moment_stiffness_coefficients": [-1.0]})
    for vertices in ([-1], [0.5], [True]):
        with pytest.raises(ValueError, match="pinned vertices are vertex "
                                             "ids"):
            bp.declared_solve_options({"pinned_vertices": vertices})
    # coefficients without a weight declare no stiffness
    assert bp.declared_solve_options(
        {"moment_stiffness_coefficients": [1.0]}
    )["moment_stiffness_weight"] == 0.0


def test_the_stiffness_and_the_pins_reach_the_node_from_the_command_line():
    """`--moment-stiffness-weight`, `--moment-stiffness-coefficients` and
    `--pinned-vertices` are parsed into the solve options, and the
    configuration built from them declares the stiffness and the region on
    a node."""
    import argparse
    parser = argparse.ArgumentParser()
    bp.add_solve_arguments(parser)
    args = parser.parse_args([
        "--moment-stiffness-weight", "1e-6",
        "--moment-stiffness-coefficients", "1", "0.5",
        "--pinned-vertices", "0", "1"])
    options = bp.declared_solve_options(bp.solve_options_from(args))
    assert options["moment_stiffness_weight"] == 1e-6
    assert options["moment_stiffness_coefficients"] == [1.0, 0.5]
    assert options["pinned_vertices"] == [0, 1]
    config = _config(**options)
    base = bp.build_base()
    node = cs.cell_node(base, cs.StationarityObjective(
        _system(base, config)))
    bp.node_configuration(config)(node)
    assert node.moment_stiffness_weight == 1e-6
    assert list(node.moment_stiffness_coefficients) == [1.0, 0.5]
    assert list(node.moment_stiffness_degrees) == [1]
    assert set(node.pinned_vertices()) == {0, 1}
