# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The cell solve as a drive of `MultiCobordism`.

The stationary points of the joint action (`cobordism.JointAction`: the Regge
term, the face-holonomy term and the matter term tr(Gamma h_1), with the
covariance Gamma rebuilt at every point by filling the declared bands of
h_1, and the pinned fiber's constraints with their multipliers) are the zeros
of its stationarity residual R. `StationarityObjective` hands that system to
`MultiCobordism` as an injected objective.

The complex `MultiCobordism` drives is the base complex, whose squared
lengths and links are the shared base field of the sheeted support (WP v18
Section 8, "Sheet convention (adopted)": equal squared lengths and equal
connection values on corresponding edges). The objective builds the sheeted
support of whatever base complex it is handed (`sheeted_support`) and scores
the system on all of its sheets:

* its scalar is ||R||, the Euclidean norm of the residual of the sheeted
  system, a real number that vanishes exactly at the stationary points of
  the complex action, so the drive's tolerance is a tolerance on the residual
  norm itself;
* the system is the one `cobordism.HolomorphicRelaxation` poses on the
  sheeted support: one squared length and one link per base edge (the edge
  classes), the equation of each the sum of its sheets' equations, the held
  monopole sectors as boundary data, and the multipliers of the pinned fiber;
* the multipliers are not coordinates of the complex. At every point they are
  the least-squares solution of the equations they enter linearly, which is
  how `SelfConsistentMeanField.joint_system` installs them, so ||R|| is a
  function of the complex alone: the residual norm minimized over the
  multipliers, with the same zeros as the full system;
* its stage-2 direction is the step that solves R = 0 to the declared order
  about the point, in the base edges' coordinates: order one is Newton's
  step, the minimum-norm solution d of J d = -R with J the closed-form
  Jacobian (`HolomorphicRelaxation.linearization`), taken over the steps that
  keep the held moduli when sectors are held;
* a spectral-moment stiffness declared on the node
  (`MultiCobordism.set_moment_stiffness`) is a term of the action on every
  sheet: its gradient in the squared lengths and in the links is added to R
  and its Hessian to J (`moment_stiffness_derivatives`), so that the scalar
  stays the residual norm of one action. Its reference is the one the node
  holds, taken on the complex the stiffness was declared on; a complex with
  another number of cells has no stiffness and no residual. None is
  declared by default;
* a pinned region declared on the node (`MultiCobordism.declare_pinned_region`)
  holds the squared length and the link of every base edge with both ends in
  it: the engine leaves those edges where they are, and the step is the one
  over the coordinates left free, the minimum-norm d with no component on a
  held coordinate that leaves the least of J d = -R (`held_coordinates`,
  `HeldLinearization`), taken over the steps that keep the held moduli when
  sectors are held. The scalar stays the norm of the whole residual, as the
  engine scores a complex with a pinned region. None is declared by default.

`MultiCobordism` does the rest as it stands (`solve`), and its mechanics are
the solve's: stage 1 scores every Pachner move of the base complex, which is
that move on every sheet, by ||R|| on the sheeted support and commits one
only when ||R|| falls by the move tolerance; stage 2 searches along the
direction from the declared first scale, which here is one, the step itself,
halving the scale as its line search does until a trial lowers ||R|| by the
tolerance. Every trial is judged on ||R|| itself, evaluated from the full
equations; the order of the step is the order of the proposal alone.

The engine's Kontsevich-Segal admissibility gate
(`MultiCobordism.admissibility_gate`) is off unless a caller's ``configure``
sets it: a drive scores every candidate move that leaves a manifold and
every trial of its line search, whatever the Kontsevich-Segal margin of the
geometry is, and a caller reads the margin of the geometry the drive ends on
(`HodgeLaplacian.kontsevichSegalMargin`). With the gate set, a geometry
whose margin is below minus the node's admissibility tolerance is outside
the configuration space: a candidate move that leads to one is not scored,
and a trial of the line search that lands on one is not scored and the scale
is halved.

A complex on which the declared system is not posed has no residual and
scores infinite, with the reason kept (`StationarityObjective.undefined`): a
point at which the action or its bands have no value, and a point at which a
held sector reads another monopole number or a held face has lost an edge.

The bands a content fills are followed from their projectors at the last
accepted point (or at the host, when so declared), which are carried by
cell: on a complex a move has changed, a reference projector is read on the
cells that complex shares with the reference's, and a cell the reference
does not have carries no weight in it.
"""
import math
import threading
import time
from collections import namedtuple

import numpy as np

import tessera as T
from tessera import cobordism as cob

#: The largest order of the step, the declared maximum of every truncated
#: series of the stack.
MAXIMUM_DIRECTION_ORDER = 10

#: Where a content's bands are followed from: the last accepted point of the
#: drive, or the host throughout.
BAND_REFERENCES = ("previous", "host")
DECLARED_BAND_REFERENCE = "previous"

#: The schedule of the engine's combinatorial search, as the emergence driver
#: declares it: how many moves deep the search goes when no shorter sequence
#: lowers the objective (one is single moves only), and a fixed composition
#: length to try first, backing off one move at a time (zero keeps the
#: deepening schedule). The two are alternatives.
DECLARED_COMBINATORIAL_DEPTH = 1
DECLARED_COMBINATORIAL_LENGTH = 0


def checked_schedule(combinatorial_depth, combinatorial_length):
    """The depth and the length of the combinatorial search as integers:
    the depth at least one, the length at least zero, and a nonzero length
    only with the declared depth, since the two select alternative
    schedules."""
    depth, length = int(combinatorial_depth), int(combinatorial_length)
    if depth < 1:
        raise ValueError("the combinatorial depth is at least 1; got %d"
                         % depth)
    if length < 0:
        raise ValueError("the combinatorial length is zero or positive; got "
                         "%d" % length)
    if length > 0 and depth != DECLARED_COMBINATORIAL_DEPTH:
        raise ValueError(
            "the combinatorial depth and the combinatorial length select "
            "alternative search schedules; a depth of %d would be ignored "
            "with the length %d" % (depth, length))
    return depth, length


#: The stops of a drive, by name.
STOP_STATIONARY = "no move and no scaled step lowers the residual norm"
STOP_NO_STEP = "the step has no value at the point reached"
STOP_DECLARED_LIMIT = "a declared limit was reached"


class DeclaredLimitReached(Exception):
    """A limit the user declared on a drive was reached."""

#: One point of a system: the `HolomorphicRelaxation` posed on the sheeted
#: support of a base complex; the number of its squared-length coordinates,
#: which is also the number of its link coordinates; which fields are
#: relaxed; the sheeted support itself; the geometry declaration the system
#: was posed with; for a self-consistent system, the mean-field declaration
#: and the reference bands the point's bands are followed from (None
#: otherwise); and, for every edge of the base complex in `getEdgeList()`
#: order, the coordinate it carries and the orientation of its stored link
#: relative to the coordinate's. On a sheeted support every base edge is its
#: own coordinate; on one sheet the coordinates are the declared edge
#: classes, when the geometry declares any.
Point = namedtuple("Point", "relaxation count lengths links support "
                            "geometry mean_field reference classes "
                            "orientations")

#: The sheeted support of a base complex (`sheeted_support`): the complex,
#: the number of base vertices, the base vertices in ascending order (base
#: vertex ``base_vertices[k]`` is vertex ``t * count + k`` of sheet t), the
#: number of sheets, and for every edge of the support in `getEdgeList()`
#: order the index of its base edge in the base complex's `getEdgeList()`
#: order and the orientation of its stored link relative to the base edge's
#: stored one (+1 or -1).
Sheeted = namedtuple("Sheeted", "spacetime count base_vertices sheets "
                                "classes orientations")


# ------------------------------------------------------------ the complex


def top_cells(spacetime):
    """The top cells of a complex as sorted vertex tuples."""
    return [tuple(sorted(int(v.getId()) for v in cell.getVertices()))
            for cell in spacetime.getTopSimplices() if cell is not None]


def edge_fields(spacetime):
    """(source, target, length, phase) of every edge in `getEdgeList()`
    order, the phase on the stored orientation."""
    return [(int(edge.getSource().getId()), int(edge.getTarget().getId()),
             complex(edge.getLength()), complex(edge.getPhase()))
            for edge in spacetime.getEdgeList().toVector()]


def complex_counts(spacetime):
    """The numbers of vertices, edges and top cells of a complex."""
    fields = edge_fields(spacetime)
    vertices = {v for a, b, _, _ in fields for v in (a, b)}
    return {"vertices": len(vertices), "edges": len(fields),
            "cells": len(top_cells(spacetime))}


def carrier_cells(spacetime, degree=1):
    """The cells the carrier operator h_degree is over, as sorted vertex
    tuples in the chain complex's order, which is the operator's mode
    order."""
    complex_ = cob.ChainComplex.fromSpacetime(spacetime)
    return [tuple(sorted(int(v) for v in cell))
            for cell in complex_.kSimplexVertices(degree)]


def sheeted_support(base, sheets):
    """The sheeted support of a base complex: ``sheets`` disjoint copies of
    it, every copy carrying the base complex's squared lengths and links on
    corresponding edges (`Sheeted`). With n base vertices, the k-th in
    ascending order is vertex t * n + k of sheet t. The support of one sheet
    is the base complex itself, with its own vertices."""
    fields = edge_fields(base)
    cells = top_cells(base)
    vertices = sorted({v for a, b, _, _ in fields for v in (a, b)}
                      | {v for cell in cells for v in cell})
    if int(sheets) == 1:
        return Sheeted(base, len(vertices), vertices, 1,
                       list(range(len(fields))), [1] * len(fields))
    rank = {v: k for k, v in enumerate(vertices)}
    count = len(vertices)
    spacetime = T.Spacetime.fromVertexTuples(
        len(cells[0]) - 1,
        [[t * count + rank[v] for v in cell]
         for t in range(sheets) for cell in cells], 1.0, 0.0)
    stored = {}
    for index, (a, b, length, phase) in enumerate(fields):
        stored[(rank[a], rank[b])] = (index, 1, length, phase)
        stored[(rank[b], rank[a])] = (index, -1, length, -phase)
    classes, orientations = [], []
    for edge in spacetime.getEdgeList().toVector():
        source = int(edge.getSource().getId())
        target = int(edge.getTarget().getId())
        sheet = source // count
        index, orientation, length, phase = stored[
            (source - sheet * count, target - sheet * count)]
        edge.setLength(length)
        edge.setPhase(phase)
        classes.append(index)
        orientations.append(orientation)
    return Sheeted(spacetime, count, vertices, int(sheets), classes,
                   orientations)


def sectors_on(sectors, host, support):
    """Held monopole sectors declared on the vertices of the sheeted
    ``host`` carried to the vertices of another sheeted support: vertex
    t * n + k of the host is base vertex ``host.base_vertices[k]`` on sheet
    t, and it is that base vertex on that sheet in ``support``. A held face
    one of whose base vertices ``support`` does not have is not a face of
    it, and the sectors then have no value there."""
    if (list(host.base_vertices) == list(support.base_vertices)
            and host.count == support.count):
        return list(sectors)
    rank = {v: k for k, v in enumerate(support.base_vertices)}
    out = []
    for sector in sectors:
        faces = []
        for face in sector.faces:
            vertices = []
            for vertex in face:
                # one sheet is the base complex with its own vertices
                sheet, base_vertex = (
                    (0, int(vertex)) if host.sheets == 1 else
                    (int(vertex) // host.count,
                     host.base_vertices[int(vertex) % host.count]))
                if base_vertex not in rank:
                    raise ValueError(
                        "a held face has lost its base vertex %d"
                        % base_vertex)
                vertices.append(base_vertex if support.sheets == 1
                                else sheet * support.count
                                + rank[base_vertex])
            faces.append(vertices)
        moved = cob.HeldMonopoleSector()
        moved.faces = faces
        moved.monopole_number = int(sector.monopole_number)
        out.append(moved)
    return out


def support_cells(support, degree=1):
    """The carrier cells of a sheeted support in the operator's mode order,
    each named by its sheet and its base vertices, a name that a Pachner
    move elsewhere on the base complex leaves as it is."""
    if support.sheets == 1:
        return [(0, cell) for cell in carrier_cells(support.spacetime,
                                                    degree)]
    return [(cell[0] // support.count,
             tuple(support.base_vertices[v % support.count] for v in cell))
            for cell in carrier_cells(support.spacetime, degree)]


# ------------------------------------------------------- the band reference


def references_of(bands):
    """The bands of a read as the reference a later read follows: one
    `BandReference` per occupied band."""
    out = []
    for band in bands:
        entry = cob.BandReference()
        entry.occupation = band.occupation
        entry.rank = band.rank
        entry.declared_index = band.declared_index
        entry.declared_positions = list(band.declared_positions)
        entry.projector = list(band.projector)
        out.append(entry)
    return out


def reference_on(cells, reference_cells, reference):
    """Reference bands read on the complex whose carrier cells are ``cells``:
    each projector's entry between two cells is its entry on the reference's
    complex when that complex has both cells and zero otherwise."""
    cells = list(cells)
    reference_cells = list(reference_cells)
    if cells == reference_cells:
        return list(reference)
    position = {cell: index for index, cell in enumerate(reference_cells)}
    shared = [(index, position[cell]) for index, cell in enumerate(cells)
              if cell in position]
    here = np.array([index for index, _ in shared], dtype=int)
    there = np.array([index for _, index in shared], dtype=int)
    count, host_count = len(cells), len(reference_cells)
    out = []
    for band in reference:
        projector = np.asarray(band.projector, dtype=complex).reshape(
            host_count, host_count)
        moved = np.zeros((count, count), dtype=complex)
        if len(shared):
            moved[np.ix_(here, here)] = projector[np.ix_(there, there)]
        entry = cob.BandReference()
        entry.occupation = band.occupation
        entry.rank = band.rank
        entry.declared_index = band.declared_index
        entry.declared_positions = list(band.declared_positions)
        entry.projector = list(moved.reshape(-1))
        out.append(entry)
    return out


def covariance_of(bands):
    """Gamma = sum_b (n_b / r_b) P_b of a read's bands, flat row-major."""
    total = None
    for band in bands:
        if band.rank == 0:
            continue
        part = (band.occupation / band.rank) * np.asarray(band.projector,
                                                          dtype=complex)
        total = part if total is None else total + part
    return [] if total is None else list(total)


# ------------------------------------------------------------- the systems


def _point(relaxation, geometry, base, support, mean_field=None,
           reference=None):
    edges = len(edge_fields(base))
    classes = list(range(edges))
    orientations = [1] * edges
    if support.sheets == 1 and len(geometry.edge_classes):
        # one sheet with declared classes: the base edges share coordinates
        classes = [int(index) for index in geometry.edge_classes]
        declared = [int(sign) for sign in geometry.edge_class_orientations]
        orientations = declared if declared else orientations
    return Point(relaxation, 1 + max(classes) if classes else 0,
                 bool(geometry.relax_lengths), bool(geometry.relax_links),
                 support, geometry, mean_field, reference, classes,
                 orientations)


def series_step(point, linearization, order):
    """The step of order ``order`` at a point: the reversion, to that order,
    of the power series of the stationarity residual along the step
    (`series_stationarity.SeriesStationarity`), every order solved against
    the point's one linearization, so that the rank decisions and the held
    moduli are those of the Newton step. Order one is the Newton step."""
    from tessera.drivers.series_stationarity import SeriesStationarity
    follower = None
    if point.mean_field is not None:
        follower = cob.BandFollower(point.mean_field)
        follower.set_reference(point.reference)
    series = SeriesStationarity(point.relaxation.action, point.geometry,
                                mean_field=point.mean_field,
                                follower=follower)
    return series.step(order, lambda right_hand_side: np.asarray(
        linearization.solve(list(np.asarray(right_hand_side,
                                            dtype=complex)))))


def moment_stiffness_derivatives(spacetime, degree, reference, coefficients,
                                 hessian=True):
    """The gradient and, with ``hessian``, the Hessian of the spectral-moment
    stiffness of a complex,

        S_M = 1/2 sum_j beta_j sum_x (mu_j(x) - mu_j^0(x))^2,
        mu_j(x) = (L^j)_xx,

    (`HodgeLaplacian.spectralMomentStiffness`: L the degree-``degree``
    operator, ``coefficients`` the beta_j, ``reference`` the flat row-major
    mu_j^0(x) of the carrier) in the coordinates of the stationarity system:
    the squared length z_e of every edge and the increment delta_e of its
    link, U_e -> U_e exp(delta_e) on the stored orientation, in
    `getEdgeList()` order, lengths then links.

    Closed forms, with D_p the derivative of L in coordinate p
    (`JointAction.carrier_derivatives`), W_j the diagonal matrix of
    mu_j - mu_j^0 and G = sum_j beta_j sum_{a+b=j-1} L^b W_j L^a:

        d_p S_M = tr(G D_p),
        d_p d_q S_M = tr(G d_p d_q L)
                      + sum_j beta_j sum_x (d_p L^j)_xx (d_q L^j)_xx
                      + tr((d_q G at fixed W) D_p),

    the first term of the Hessian being the second derivatives of the
    operator contracted with G (`JointAction.action_hessian` of the action
    whose only term is tr(G L)).

    Returns the gradient (2 |E| entries) and the Hessian (2 |E| by 2 |E|, or
    None). Raises ValueError when the reference is not of this complex's
    number of cells, in which case the stiffness has no value here. The
    reference carries no names of cells: on a complex with as many cells as
    the reference's, the moments of each cell are read against the
    reference's entries at its place in the operator's order of the cells,
    whichever cells the reference was read on."""
    hodge = cob.HodgeLaplacian(spacetime)
    orders = len(coefficients)
    edges = len(edge_fields(spacetime))
    gradient = np.zeros(2 * edges, dtype=complex)
    if orders == 0:
        return gradient, (np.zeros((2 * edges, 2 * edges), dtype=complex)
                          if hessian else None)
    moments = np.asarray(hodge.localSpectralMoments(int(degree), orders),
                         dtype=complex)
    size = len(moments) // orders
    if len(reference) != size * orders:
        raise ValueError(
            "the spectral-moment stiffness has no value on this complex: its "
            "reference, the moments of the complex it was declared on, has "
            "%d entries, and the degree-%d operator here has %d cells at %d "
            "orders" % (len(reference), int(degree), size, orders))
    deviation = moments.reshape(size, orders) - np.asarray(
        reference, dtype=complex).reshape(size, orders)
    declaration = cob.JointActionDeclaration()
    declaration.carrier_degree = int(degree)
    declaration.metric_source = hodge.metricSource()
    declaration.gravitational_weight = 0.0
    declaration.holonomy_weight = 0.0
    declaration.matter_weight = 1.0
    declaration.covariance = [0j] * (size * size)
    action = cob.JointAction(spacetime, declaration)
    operator = np.asarray(action.carrier_operator(),
                          dtype=complex).reshape(size, size)
    powers = [np.eye(size, dtype=complex)]
    for _ in range(orders):
        powers.append(powers[-1] @ operator)
    beta = [float(c) for c in coefficients]
    contraction = np.zeros((size, size), dtype=complex)
    for j in range(1, orders + 1):
        weights = np.diag(deviation[:, j - 1])
        for a in range(j):
            contraction += beta[j - 1] * (powers[j - 1 - a] @ weights
                                          @ powers[a])
    derivatives = action.carrier_derivatives()
    first = [np.asarray(matrix, dtype=complex).reshape(size, size)
             for matrix in list(derivatives.lengths) + list(derivatives.links)]
    for p, matrix in enumerate(first):
        gradient[p] = np.sum(contraction * matrix.T)
    if not hessian:
        return gradient, None
    action.set_covariance(list(contraction.reshape(-1)))
    second = np.asarray(action.action_hessian(True, True),
                        dtype=complex).reshape(2 * edges, 2 * edges).copy()
    # the derivatives of the powers, d_p L^j = d_p L^(j-1) L + L^(j-1) D_p
    power_derivatives = []
    for matrix in first:
        of_powers = [np.zeros((size, size), dtype=complex)]
        for j in range(1, orders + 1):
            of_powers.append(of_powers[-1] @ operator + powers[j - 1] @ matrix)
        power_derivatives.append(of_powers)
    for q in range(2 * edges):
        moved = np.zeros((size, size), dtype=complex)
        for j in range(1, orders + 1):
            weights = np.diag(deviation[:, j - 1])
            for a in range(j):
                b = j - 1 - a
                moved += beta[j - 1] * (
                    powers[b] @ weights @ power_derivatives[q][a]
                    + power_derivatives[q][b] @ weights @ powers[a])
        for p in range(2 * edges):
            value = np.sum(moved * first[p].T)
            for j in range(1, orders + 1):
                value += beta[j - 1] * np.sum(
                    np.diag(power_derivatives[p][j])
                    * np.diag(power_derivatives[q][j]))
            second[p, q] += value
    return gradient, second


def held_coordinates(spacetime, regions, classes):
    """The coordinates the pinned regions of a node hold on a base complex:
    `MultiCobordism` holds an edge, its squared length and its link, when
    both its ends lie in one region, and a coordinate is held when an edge
    that carries it is. ``regions`` are vertex sets, ``classes`` the
    coordinate of every edge in `getEdgeList()` order (`Point.classes`).
    Returns the held coordinates in ascending order."""
    regions = [set(int(v) for v in region) for region in regions]
    held = set()
    for (a, b, _, _), index in zip(edge_fields(spacetime), classes):
        if any(a in region and b in region for region in regions):
            held.add(int(index))
    return sorted(held)


class HeldStep:
    """The step of a `HeldLinearization` with the measurements of its linear
    solve, under the names of `cobordism.HolomorphicNewtonStep`."""

    def __init__(self, **fields):
        self.__dict__.update(fields)


class HeldLinearization:
    """The linearization of a system at a point over the steps that have no
    component on held coordinates: `HolomorphicRelaxation.linearization`
    restricted to the coordinates a pinned region leaves free.

    The system's variables are the squared lengths of the coordinates, the
    increments of their links (U -> U exp(delta)) and the multipliers, in
    that order (`Point`). A step of this linearization is zero on the squared
    length and the link of every coordinate in ``held``, and, when the
    geometry holds monopole sectors, the real part of its link block changes
    no held modulus: it lies in the kernel of the held faces' coboundary on
    the free link coordinates. Such steps are parametrized by real unknowns
    (the real and imaginary parts of every free length and of every
    multiplier, the imaginary part of every free link, and the coefficients
    of the link block's real part on an orthonormal basis of that kernel),
    which is the parametrization the library uses for held sectors, with the
    held coordinates left out.

    `solve(rhs)` is the minimum-norm solution, over those steps, of the
    least-squares problem J d = rhs: with D the scales of the variables
    (`HolomorphicRelaxation.variable_scales`, the modulus of a coordinate's
    squared length for a length and one otherwise; all one where the library
    has no such method), it solves (D J D) y = D rhs with d = D y, the
    singular values at or below the declared rank tolerance times the
    largest counted as zero, as the library's own linearization does.
    ``added_residual`` and ``added_jacobian`` are terms added to the system's
    residual and Jacobian (flat, the Jacobian row-major), or empty."""

    def __init__(self, point, held, added_residual=(), added_jacobian=()):
        relaxation = point.relaxation
        size = int(relaxation.variable_count())
        residual = np.asarray(relaxation.residual(), dtype=complex)
        jacobian = np.asarray(relaxation.jacobian(),
                              dtype=complex).reshape(size, size)
        if len(added_residual):
            residual = residual + np.asarray(added_residual, dtype=complex)
        if len(added_jacobian):
            jacobian = jacobian + np.asarray(
                added_jacobian, dtype=complex).reshape(size, size)
        scales = getattr(relaxation, "variable_scales", None)
        scale = (np.asarray(scales(), dtype=float) if scales is not None
                 else np.ones(size))
        tolerance = float(point.geometry.rank_tolerance)
        count = point.count
        held = sorted(int(c) for c in held)
        free = [c for c in range(count) if c not in set(held)]
        length_offset = 0
        link_offset = count if point.lengths else 0
        multiplier_offset = link_offset + (count if point.links else 0)
        self.held_variables = (
            ([length_offset + c for c in held] if point.lengths else [])
            + ([link_offset + c for c in held] if point.links else []))

        def unit(index, value):
            column = np.zeros(size, dtype=complex)
            column[index] = value
            return column

        basis = []
        if point.lengths:
            for c in free:
                basis.append(unit(length_offset + c, 1.0))
                basis.append(unit(length_offset + c, 1j))
        sectors = list(point.geometry.held_sectors)
        self.constrained = bool(point.links and sectors)
        if point.links:
            for c in free:
                basis.append(unit(link_offset + c, 1j))
            moduli = np.eye(len(free))
            if self.constrained:
                coboundary = self._coboundary(point, sectors)[:, free]
                _, singular, right = np.linalg.svd(coboundary)
                rank = (int(np.sum(singular > tolerance * singular[0]))
                        if len(singular) and singular[0] > 0.0 else 0)
                moduli = right[rank:].T
            for k in range(moduli.shape[1]):
                column = np.zeros(size, dtype=complex)
                column[[link_offset + c for c in free]] = moduli[:, k]
                basis.append(column)
        for index in range(multiplier_offset, size):
            basis.append(unit(index, 1.0))
            basis.append(unit(index, 1j))
        self._parametrization = (np.stack(basis, axis=1) if basis
                                 else np.zeros((size, 0), dtype=complex))
        self._scale = scale
        self._jacobian = jacobian
        self.residual = residual
        scaled = (scale[:, None] * jacobian * scale[None, :])
        image = scaled @ self._parametrization
        system = np.concatenate([image.real, image.imag], axis=0)
        self._left, self._singular, self._right = np.linalg.svd(
            system, full_matrices=False)
        singular = self._singular
        self._rank = (int(np.sum(singular > tolerance * singular[0]))
                      if len(singular) and singular[0] > 0.0 else 0)
        # the measurements of the solve: the complex system over the free
        # variables, as the library reports its own
        kept = [index for index in range(size)
                if index not in set(self.held_variables)]
        values = (np.linalg.svd(scaled[:, kept], compute_uv=False)
                  if kept else np.zeros(0))
        rank = (int(np.sum(values > tolerance * values[0]))
                if len(values) and values[0] > 0.0 else 0)
        step = self.solve(-residual)
        norm = float(np.linalg.norm(residual))
        gap = lambda v, r: (float("nan") if r == 0 else  # noqa: E731
                            float(v[r - 1] / v[r]) if r < len(v)
                            else float("inf"))
        self.newton_step = HeldStep(
            step=[complex(x) for x in step],
            residual_norm=norm,
            jacobian_rank=rank,
            rank_tolerance=tolerance,
            largest_singular_value=float(values[0]) if len(values) else 0.0,
            smallest_retained_singular_value=(
                float(values[rank - 1]) if rank else float("nan")),
            largest_discarded_singular_value=(
                float(values[rank]) if rank < len(values) else 0.0),
            rank_gap=gap(values, rank),
            constrained=self.constrained,
            constrained_rank=self._rank if self.constrained else 0,
            constrained_rank_gap=(gap(singular, self._rank)
                                  if self.constrained else float("nan")),
            linear_residual=(float(np.linalg.norm(jacobian @ step + residual)
                                   / norm) if norm > 0.0 else 0.0))

    @staticmethod
    def _coboundary(point, sectors):
        """The real coboundary of the held faces on the link coordinates:
        one row per held face, the sum over its three sides of the sign of
        the side on its stored edge, pulled back through the edge classes
        (a link coordinate moves every edge that carries it, each on its own
        orientation). A real link increment in its kernel changes no held
        modulus."""
        fields = edge_fields(point.support.spacetime)
        lookup = {}
        for index, (a, b, _, _) in enumerate(fields):
            lookup[(a, b)] = (index, 1.0)
            lookup[(b, a)] = (index, -1.0)
        classes = [int(c) for c in point.geometry.edge_classes]
        signs = [int(s) for s in point.geometry.edge_class_orientations]
        if not classes:
            classes = list(range(len(fields)))
        if not signs:
            signs = [1] * len(fields)
        rows = []
        for sector in sectors:
            for face in sector.faces:
                row = np.zeros(point.count)
                for k in range(3):
                    index, sign = lookup[(int(face[k]),
                                          int(face[(k + 1) % 3]))]
                    row[classes[index]] += sign * signs[index]
                rows.append(row)
        return (np.stack(rows) if rows
                else np.zeros((0, point.count)))

    def solve(self, right_hand_side):
        """The minimum-norm least-squares step d with J d = rhs over the
        steps this linearization allows."""
        target = self._scale * np.asarray(right_hand_side, dtype=complex)
        stacked = np.concatenate([target.real, target.imag])
        rank = self._rank
        unknown = self._right[:rank].T @ (
            (self._left[:, :rank].T @ stacked) / self._singular[:rank])
        step = self._scale * (self._parametrization @ unknown)
        step[self.held_variables] = 0.0
        return step


def least_squares_multipliers(action, geometry):
    """The multipliers of the constraints an action declares that leave the
    least of the stationarity equations they enter: those equations are
    linear in the multipliers, R = R_0 + G xi, so xi is the minimum-norm
    least-squares solution of G xi = -R_0, the singular values of G at or
    below the declared rank tolerance times the largest counted as zero.
    The fit is of the equations in one unit: a length equation dS/dz times
    the scale of its coordinate (`HolomorphicRelaxation.variable_scales`),
    as the step's linear solve scales them, so that it is the same fit in
    every unit of length. It is the rule
    `SelfConsistentMeanField.joint_system` installs the pinned fiber's
    multipliers by."""
    count = int(action.constraint_count())

    def geometric(multipliers):
        action.set_multipliers(list(multipliers))
        residual = np.asarray(
            cob.HolomorphicRelaxation(action, geometry).residual(),
            dtype=complex)
        return residual[:len(residual) - count]

    zero = geometric([0j] * count)
    gradients = np.stack(
        [geometric([1.0 + 0j if k == j else 0j for k in range(count)]) - zero
         for j in range(count)], axis=1)
    scales = np.asarray(
        cob.HolomorphicRelaxation(action, geometry).variable_scales(),
        dtype=float)[:len(zero)]
    zero = scales * zero
    gradients = scales[:, None] * gradients
    left, singular, right = np.linalg.svd(gradients, full_matrices=False)
    rank = (int(np.sum(singular > geometry.rank_tolerance * singular[0]))
            if len(singular) and singular[0] > 0.0 else 0)
    estimate = -(right[:rank].conj().T
                 @ ((left[:, :rank].conj().T @ zero) / singular[:rank]))
    return [complex(x) for x in estimate]


class ReggeStart:
    """The geometry the continued Regge sheets of a drive start from
    (`JointActionDeclaration.regge_start_squared_lengths`): the squared
    length of every edge of the base complex where the drive began, named by
    the edge's two base vertices. A system builds a new `JointAction` at
    every point it scores; each one is declared this start, so the Regge
    term is read on one continued sheet from the drive's start to its end.
    An edge a Pachner move creates has no start: at a point that is scored
    it starts at the squared length it has there, and from the first
    accepted point that has it, at the squared length it was accepted
    with."""

    def __init__(self):
        self._squared = None

    @staticmethod
    def _of(base):
        return {(min(a, b), max(a, b)): complex(length * length)
                for a, b, length, _ in edge_fields(base)}

    def begin(self, base):
        """The start is the geometry ``base`` holds."""
        self._squared = self._of(base)

    @property
    def begun(self):
        return self._squared is not None

    def accept(self, base):
        """An accepted point: an edge without a start takes its squared
        length there."""
        if self._squared is None:
            return
        for edge, squared in self._of(base).items():
            self._squared.setdefault(edge, squared)

    def declared(self, declaration, support):
        """A copy of ``declaration`` with the start of every edge of a
        sheeted support's complex, when a drive has begun and the
        declaration names no start of its own; ``declaration`` itself
        otherwise. The declaration given is never written to: one object can
        be the declaration of every point of a drive, scored from several
        threads."""
        if (self._squared is None
                or len(declaration.regge_start_squared_lengths)):
            return declaration
        declaration = cob.JointActionDeclaration(declaration)
        start = []
        for a, b, length, _ in edge_fields(support.spacetime):
            if support.sheets != 1:
                a = support.base_vertices[a % support.count]
                b = support.base_vertices[b % support.count]
            start.append(self._squared.get((min(a, b), max(a, b)),
                                           complex(length * length)))
        declaration.regge_start_squared_lengths = start
        return declaration


class GeometricSystem:
    """The stationarity system of the joint action at the carried state the
    action declares, on the sheeted support of any base complex.

    ``declare(spacetime)`` returns the `JointActionDeclaration` of a sheeted
    support's complex; ``geometry_of(support)`` returns a new
    `HolomorphicRelaxationDeclaration` for a `Sheeted` support (the relaxed
    fields, the support's edge classes and orientations, the held sectors on
    its sheets). When the declaration relaxes the multipliers of constraints
    the action declares, they are the least-squares ones at every point
    (`least_squares_multipliers`), as a content's are. ``begin(base)``
    declares the geometry the Regge sheets of a drive start from
    (`ReggeStart`); `solve` calls it with the complex it is given."""

    def __init__(self, declare, geometry_of, sheets):
        self._declare = declare
        self._geometry_of = geometry_of
        self.sheets = int(sheets)
        self.regge_start = ReggeStart()

    def begin(self, base):
        """A drive begins at ``base``: its Regge sheets start there."""
        self.regge_start.begin(base)

    def _action(self, support):
        return cob.JointAction(support.spacetime, self.regge_start.declared(
            self._declare(support.spacetime), support))

    def point(self, base):
        support = sheeted_support(base, self.sheets)
        geometry = self._geometry_of(support)
        action = self._action(support)
        if geometry.relax_multipliers and action.constraint_count() > 0:
            action.set_multipliers(least_squares_multipliers(action,
                                                             geometry))
        return _point(cob.HolomorphicRelaxation(action, geometry), geometry,
                      base, support)

    def accept(self, base):
        """An accepted point. Nothing is carried from one to the next; when
        the geometry declaration records terms, every term of the action
        there is returned (`cobordism.action_term_records`)."""
        self.regge_start.accept(base)
        support = sheeted_support(base, self.sheets)
        geometry = self._geometry_of(support)
        if not geometry.record_terms:
            return None
        return cob.action_term_records(self._action(support), geometry)


class ContentSystem:
    """The self-consistent stationarity system of a content on the sheeted
    support of any base complex: the covariance rebuilt from the bands the
    content fills, the pinned fiber's constraints with their targets, and
    their least-squares multipliers
    (`SelfConsistentMeanField.joint_system`).

    ``declare(spacetime)`` returns the `JointActionDeclaration` of a sheeted
    support's complex; ``mean_field_of(support)`` returns a new
    `SelfConsistentMeanFieldDeclaration` for a `Sheeted` support (the
    occupations, the number of pinned constraints of the fiber, and the
    geometry declaration with the support's edge classes, orientations and
    held sectors). ``host`` is the base complex the bands are chosen on by
    the declared order, and where the pinned fiber's targets and unit are
    read when the declaration gives none: they are numbers of the host, the
    same on every complex afterwards. ``band_reference`` says where the
    bands are followed from afterwards (`BAND_REFERENCES`). The Regge sheets
    of a drive start at the host (`ReggeStart`), or at the complex
    ``begin(base)`` is given."""

    def __init__(self, declare, mean_field_of, host, sheets,
                 band_reference=DECLARED_BAND_REFERENCE):
        if band_reference not in BAND_REFERENCES:
            raise ValueError("the band reference is one of %s; got %r"
                             % (", ".join(BAND_REFERENCES), band_reference))
        self._declare = declare
        self._mean_field_of = mean_field_of
        self.sheets = int(sheets)
        self.band_reference = band_reference
        self._targets = None
        self._scale = 0.0
        self.regge_start = ReggeStart()
        self.regge_start.begin(host)
        field, action, support = self._field(host)
        start = field.iterate()
        declared = self._mean_field_of(support)
        if (declared.fiber_moments
                and not len(declared.fiber_moment_targets)
                and not len(declared.fiber_moment_unit_targets)):
            pinned = field.joint_system().action.declaration
            self._scale = float(pinned.moment_scale)
            # the targets as the constraints carry them, in the unit they
            # are solved in
            self._targets = [complex(constraint.target)
                             for constraint in pinned.moment_constraints]
        self.reference_cells = support_cells(
            support, action.declaration.carrier_degree)
        self.reference = references_of(start.bands)
        self._covariance = covariance_of(start.bands)
        # the places each band is declared at, per complex (named by its
        # carrier cells): on the host, where the bands are chosen; on any
        # other complex, where the first measurement there finds them
        self._places = {tuple(self.reference_cells): [
            list(band.declared_positions) for band in start.bands]}

    def _declaration(self, support):
        """The mean-field declaration of a support, the fiber pinned at the
        host's targets in the host's unit."""
        declaration = self._mean_field_of(support)
        if self._targets is not None:
            declaration.fiber_moment_unit_targets = list(self._targets)
            declaration.fiber_moment_scale = self._scale
        return declaration

    def begin(self, base):
        """A drive begins at ``base``: its Regge sheets start there."""
        self.regge_start.begin(base)

    def _action(self, support):
        return cob.JointAction(support.spacetime, self.regge_start.declared(
            self._declare(support.spacetime), support))

    def _field(self, base):
        support = sheeted_support(base, self.sheets)
        action = self._action(support)
        return (cob.SelfConsistentMeanField(
            action, self._declaration(support)), action, support)

    def _reference(self, support, action):
        """The reference bands read on a support's carrier cells, with the
        cells."""
        cells = support_cells(support, action.declaration.carrier_degree)
        return (reference_on(cells, self.reference_cells, self.reference),
                cells)

    def _measured_reference(self, field, support, action):
        """`_reference` for a measurement. A band's declared places are
        places in the spectrum of one complex, so a reference read on a
        complex with other cells declares the places its bands hold at the
        first measurement on that complex; whether a band has left its
        places (`OccupiedBand.crossed`) is then said of the complex the band
        is read on. Which modes a band takes is decided by the reference's
        projectors alone and is the same either way."""
        reference, cells = self._reference(support, action)
        if cells != self.reference_cells:
            key = tuple(cells)
            if key not in self._places:
                first = field.iterate(reference, [])
                self._places[key] = [list(band.positions)
                                     for band in first.bands]
            for entry, places in zip(reference, self._places[key]):
                entry.declared_positions = list(places)
        return reference, cells

    def point(self, base):
        support = sheeted_support(base, self.sheets)
        declaration = self._declaration(support)
        action = self._action(support)
        field = cob.SelfConsistentMeanField(action, declaration)
        reference, _ = self._reference(support, action)
        return _point(field.joint_system(reference), declaration.geometry,
                      base, support, declaration, reference)

    def iterate(self, base):
        """The measurements of a point (`SelfConsistentMeanField.iterate`),
        its bands followed from the reference and its covariance change taken
        from the last accepted point's, with the point's carrier cells."""
        field, action, support = self._field(base)
        reference, cells = self._measured_reference(field, support, action)
        previous = (self._covariance
                    if len(self._covariance) == len(cells) ** 2 else [])
        return field.iterate(reference, previous), cells

    def accept(self, base):
        """An accepted point: its measurements, and the reference moved to
        its bands when they are followed from the last accepted point."""
        self.regge_start.accept(base)
        step, cells = self.iterate(base)
        self._covariance = covariance_of(step.bands)
        if self.band_reference == "previous":
            self.reference_cells = cells
            self.reference = references_of(step.bands)
            self._places.setdefault(tuple(cells), [
                list(band.declared_positions) for band in step.bands])
        return step

    def read(self, base, start_scale=0.0):
        """The end-point read of a base complex: its sheeted support, the
        action there carrying the covariance, the constraints and their
        multipliers, and the report (`SelfConsistentMeanField.read`), the
        bands followed from the reference."""
        field, action, support = self._field(base)
        reference, _ = self._measured_reference(field, support, action)
        return (support, field.joint_system(reference).action,
                field.read(reference, start_scale))


# ------------------------------------------------------------ the objective


class StationarityObjective(cob.CobordismObjective):
    """The stationarity of a system as a `MultiCobordism` objective.

    ``system`` poses the equations on the sheeted support of a base complex
    (`GeometricSystem`, `ContentSystem`): ``system.point(base)`` is the
    `Point` there and ``system.accept(base)`` is told every accepted point.
    The scalar is the residual norm there, and the stage-2 direction is the
    step of order ``direction_order`` (one to `MAXIMUM_DIRECTION_ORDER`) in
    the base edges' coordinates. ``series(point, linearization, order)``
    returns the step of an order above one in the system's variables
    (`series_step`)."""

    def __init__(self, system, direction_order=1, series=series_step):
        super().__init__()
        direction_order = int(direction_order)
        if not 1 <= direction_order <= MAXIMUM_DIRECTION_ORDER:
            raise ValueError(
                "the order of the step is an integer from 1 to %d; got %d"
                % (MAXIMUM_DIRECTION_ORDER, direction_order))
        self._system = system
        self._series = series
        self.direction_order = direction_order
        #: Why each scored complex without a residual had none, in the order
        #: met.
        self.undefined = []
        #: The number of complexes scored.
        self.scored = 0
        # stage 1 scores its candidates from several threads
        self._lock = threading.Lock()
        #: One record per stage-2 update, taken where its step was formed.
        self.updates = []
        #: The vertex sets of the pinned regions of the node the objective
        #: is injected in (`hold`); none unless the node declares one.
        self.pinned_regions = []
        # the cells and the fields of the complex at every step proposal,
        # and of the last complex scored with its score
        self._proposed = []
        self._last_scored = None
        # the declared time of the drive in progress, read on the thread
        # that runs it (`begin`)
        self._deadline = None
        self._seconds = None
        self._thread = None

    def begin(self, time_limit_seconds=None):
        """Start the clock of a drive run from the calling thread. A
        declared time that has passed raises `DeclaredLimitReached` from the
        next scoring or step proposal made on that thread, which the engine
        answers as it answers any error of an objective: the complex is
        restored to the last accepted state and the error reaches the
        caller."""
        self._thread = threading.get_ident()
        self._seconds = time_limit_seconds
        self._deadline = (None if time_limit_seconds is None
                          else time.monotonic() + float(time_limit_seconds))

    def hold(self, regions):
        """Tell the objective the pinned regions of its node, as vertex
        sets: the edges the engine holds, on whose coordinates the step has
        no component (`held_coordinates`)."""
        self.pinned_regions = [set(int(v) for v in region)
                               for region in regions]

    @staticmethod
    def _fingerprint(spacetime):
        """The cells and the fields of a complex, to tell one point of a
        drive from another."""
        return (tuple(sorted(top_cells(spacetime))),
                tuple(edge_fields(spacetime)))

    def proposed_points(self, final):
        """The distinct points of a drive in order, each by its cells and
        its fields: the points its steps were proposed from, and ``final``,
        the complex it was left on, when no step was proposed from it."""
        points = []
        for point in self._proposed + [self._fingerprint(final)]:
            if not points or point != points[-1]:
                points.append(point)
        return points

    def proposed_trace(self, final=None):
        """The residual norm at every distinct point a step was proposed
        from, in order: the first point, then one entry per committed move
        update and per accepted relaxation update that a later proposal
        followed. With ``final``, the complex a drive was left on, and when
        no step was proposed from it, its score closes the trace when it is
        the last complex scored: an accepted trial is the last one its line
        search scores. It is the engine's trace of a drive, kept here
        because the engine returns its own only when the drive returns."""
        trace = []
        previous = None
        for update, point in zip(self.updates, self._proposed):
            if point != previous:
                trace.append(float(update["residual_norm"]))
            previous = point
        if final is not None and self._last_scored is not None:
            ended = self._fingerprint(final)
            if ended != previous and self._last_scored[0] == ended:
                trace.append(float(self._last_scored[1]))
        return trace

    def _check_time(self):
        if (self._deadline is not None
                and threading.get_ident() == self._thread
                and time.monotonic() >= self._deadline):
            raise DeclaredLimitReached(
                "the declared time limit of %g seconds" % self._seconds)

    # -- the declared interface -------------------------------------------

    def name(self):
        return "joint_action_stationarity"

    def term_names(self):
        return [cob.ObjectiveTermName.JOINT_ACTION_STATIONARITY]

    def is_target_conditioned(self):
        return False

    def terms(self, context):
        self._check_time()
        terms = cob.MultiCobordism.ObjectiveTerms()
        with self._lock:
            self.scored += 1
        try:
            point = self._system.point(context.spacetime)
            residual = np.asarray(point.relaxation.residual(), dtype=complex)
            added, _ = self._stiffness(point, context, False)
            if len(added):
                residual = residual + np.asarray(added)
            value = float(np.linalg.norm(residual))
        except DeclaredLimitReached:
            raise
        except Exception as error:  # noqa: BLE001
            # the declared system is not posed on this complex: it has no
            # residual, the objective is infinite, a step that lands there is
            # shortened and a move that leads there is not committed; the
            # reason is kept for the report. An error of any kind is a
            # reason, so that the scoring of one complex does not end the
            # drive of another; one the library and the systems do not
            # raise by design carries its type's name.
            reason = str(error)
            if not isinstance(error, (ValueError, ArithmeticError,
                                      RuntimeError)):
                reason = "%s: %s" % (type(error).__name__, reason)
            self.undefined.append(reason)
            terms.joint_action_stationarity = float("inf")
            return terms
        if not math.isfinite(value):
            self.undefined.append("the residual norm is not finite")
            value = float("inf")
        with self._lock:
            self._last_scored = (self._fingerprint(context.spacetime), value)
        terms.joint_action_stationarity = value
        return terms

    def direction(self, context):
        self._check_time()
        spacetime = context.scalar.spacetime
        measured = self._system.accept(spacetime)
        point = self._system.point(spacetime)
        added_residual, added_jacobian = self._stiffness(
            point, context.scalar, True)
        held = held_coordinates(spacetime, self.pinned_regions,
                                point.classes)
        if held:
            linearization = HeldLinearization(point, held, added_residual,
                                              added_jacobian)
        else:
            linearization = point.relaxation.linearization(added_residual,
                                                           added_jacobian)
        newton = linearization.newton_step
        if self.direction_order == 1:
            step = np.asarray(newton.step, dtype=complex)
        elif len(added_residual):
            raise ValueError(
                "the step of order %d has no value with a declared moment "
                "stiffness: the stiffness is not a term of the residual's "
                "power series" % self.direction_order)
        else:
            step = np.asarray(self._series(point, linearization,
                                           self.direction_order),
                              dtype=complex)
        classes = np.asarray(point.classes, dtype=int)
        signs = np.asarray(point.orientations, dtype=float)
        out = cob.ObjectiveDirection()
        offset = 0
        # stage 2 subtracts the direction: z - a, phi - a_phi. The step moves
        # z by its length block and multiplies each link by exp(delta) on the
        # coordinate's orientation, that is phi by -i delta there, and every
        # edge of the base complex takes the step of the coordinate it
        # carries.
        if point.lengths:
            out.ascent = -step[:point.count][classes]
            offset = point.count
        else:
            out.ascent = np.zeros(len(classes), dtype=complex)
        if point.links:
            out.phase_ascent = 1j * signs * step[offset:offset
                                                 + point.count][classes]
        out.baseline = float(newton.residual_norm)
        out.baseline_computed = True
        self._proposed.append(self._fingerprint(spacetime))
        self.updates.append({
            "residual_norm": float(newton.residual_norm),
            "step_norm": float(np.linalg.norm(step)),
            "jacobian_rank": int(newton.jacobian_rank),
            "largest_singular_value": float(newton.largest_singular_value),
            "smallest_retained_singular_value": float(
                newton.smallest_retained_singular_value),
            "largest_discarded_singular_value": float(
                newton.largest_discarded_singular_value),
            "rank_gap": float(newton.rank_gap),
            "constrained_step": bool(newton.constrained),
            "constrained_rank": int(newton.constrained_rank),
            "constrained_rank_gap": float(newton.constrained_rank_gap),
            "linear_residual": float(newton.linear_residual),
            "held_coordinates": list(held),
            "complex": complex_counts(spacetime),
            "cells": sorted(top_cells(spacetime)),
            "scored_before": self.scored,
            "undefined_before": len(self.undefined),
            "measured": measured,
        })
        return out

    # -- a declared stiffness ---------------------------------------------

    def _stiffness(self, point, context, jacobian):
        """The gradient and, with ``jacobian``, the Hessian of the node's
        spectral-moment stiffness on the system's variables (flat, the
        Hessian row-major): the stiffness of the base complex, declared on
        the node, once per sheet, in the squared lengths and the links that
        the system relaxes (`moment_stiffness_derivatives`). Empty when no
        stiffness is declared or neither field is relaxed."""
        weight = (float(context.moment_stiffness_weight)
                  * point.support.sheets)
        if weight == 0.0 or not (point.lengths or point.links):
            return [], []
        spacetime = context.spacetime
        edges = len(point.classes)
        coefficients = list(context.moment_stiffness_coefficients)
        gradient = np.zeros(2 * edges, dtype=complex)
        hessian = np.zeros((2 * edges, 2 * edges), dtype=complex)
        for degree, reference in zip(context.moment_stiffness_degrees,
                                     context.moment_stiffness_reference):
            part, second = moment_stiffness_derivatives(
                spacetime, degree, list(reference), coefficients, jacobian)
            gradient += weight * part
            if jacobian:
                hessian += weight * second
        # the equation of a coordinate is the sum of its edges' equations,
        # and the coordinate moves every edge that carries it: its squared
        # length by the coordinate's step, its link by the step on the
        # coordinate's orientation
        classes = np.asarray(point.classes, dtype=int)
        carried = np.zeros((edges, point.count))
        carried[np.arange(edges), classes] = 1.0
        oriented = np.zeros((edges, point.count))
        oriented[np.arange(edges), classes] = np.asarray(point.orientations,
                                                         dtype=float)
        blocks = []
        if point.lengths:
            blocks.append((carried, slice(0, edges)))
        if point.links:
            blocks.append((oriented, slice(edges, 2 * edges)))
        size = point.relaxation.variable_count()
        residual = np.zeros(size, dtype=complex)
        for index, (expansion, rows) in enumerate(blocks):
            at = slice(index * point.count, (index + 1) * point.count)
            residual[at] = expansion.T @ gradient[rows]
        if not jacobian:
            return list(residual), []
        matrix = np.zeros((size, size), dtype=complex)
        for i, (left, rows) in enumerate(blocks):
            for j, (right, columns) in enumerate(blocks):
                matrix[i * point.count:(i + 1) * point.count,
                       j * point.count:(j + 1) * point.count] = \
                    left.T @ hessian[rows, columns] @ right
        return list(residual), list(matrix.reshape(-1))


# ----------------------------------------------------------------- the drive


def undefined_reasons(reasons):
    """How many scored complexes had no residual for each reason, in the
    order the reasons were first met."""
    counts = {}
    for reason in reasons:
        counts[reason] = counts.get(reason, 0) + 1
    return counts


def cell_node(spacetime, objective, register_degrees=(1,)):
    """A `MultiCobordism` node on ``spacetime`` that descends ``objective``
    and nothing else: no target, no boundary block, no surgery, the strict
    emergence mode."""
    node = cob.MultiCobordism(spacetime, [], [], list(register_degrees), 1.0,
                              0, 0, False)
    node.set_objective(objective)
    node.set_simulation_mode(cob.MultiCobordism.SimulationMode.EMERGENCE,
                             cob.MultiCobordism.EmergenceSubmode.STRICT)
    node.should_propose_surgery = False
    return node


def solve(spacetime, system, tolerance=1e-15, move_tolerance=1e-15,
          direction_order=1, series=series_step, moves=True,
          combinatorial_depth=DECLARED_COMBINATORIAL_DEPTH,
          combinatorial_length=DECLARED_COMBINATORIAL_LENGTH,
          candidate_moves=0, iteration_limit=None, update_limit=None,
          time_limit_seconds=None, configure=None):
    """Drive the base complex ``spacetime`` to a stationary point of
    ``system`` with `MultiCobordism`, whose mechanics are used as they are.

    With ``moves`` the combined drive runs (`MultiCobordism.run`): every
    iteration scores every Pachner move of the base complex
    (``candidate_moves`` zero; a positive number draws that many), commits
    the best that lowers the residual norm by more than ``move_tolerance``,
    and relaxes the geometry until no trial of its line search lowers it by
    ``tolerance``. The search over compositions of moves is the engine's:
    with ``combinatorial_depth`` d it deepens, when no single move lowers
    the residual norm, to sequences of two moves, then three, up to d, each
    scored and committed as a whole (its ``max_lookahead``); with
    ``combinatorial_length`` n above zero it searches sequences of exactly n
    moves first and backs off one move at a time (its
    ``combinatorial_breadth``). The two are alternatives
    (`checked_schedule`). Without ``moves`` only the geometry relaxes
    (`MultiCobordism.run_stage2`). The first scale of the line search is
    one, the step itself.

    No count and no time is declared by default. ``iteration_limit`` is the
    engine's number of iterations (each one move update and a relaxation;
    without ``moves``, relaxation updates), ``update_limit`` its number of
    relaxation updates after each move update, and ``time_limit_seconds`` a
    wall-clock time read by the objective (`StationarityObjective.begin`).
    ``configure(node)`` may declare a pinned region or a spectral-moment
    stiffness on the node before the drive; none is declared otherwise.

    The Regge term is read on the sheets continued from the geometry
    ``spacetime`` holds when the drive begins (``system.begin``,
    `ReggeStart`).

    Returns the drive's record: the base complex it ended on (``spacetime``;
    a committed move replaces the object), the node and the objective, the
    trace of the residual norm (the engine's; for a drive an error or a
    declared time ended, the residual norm at every point a step was
    proposed from and at the last accepted point,
    `StationarityObjective.proposed_trace`), the stop by name
    with its detail, the number of committed move updates and of accepted
    relaxation updates, how many scored complexes had no residual for each
    reason (``undefined_reasons``), the counts of the base complex before
    and after, and the seconds it took."""
    depth, length = checked_schedule(combinatorial_depth,
                                     combinatorial_length)
    # the drive begins here: the system's Regge sheets start at this geometry
    begin = getattr(system, "begin", None)
    if begin is not None:
        begin(spacetime)
    objective = StationarityObjective(system, direction_order, series)
    node = cell_node(spacetime, objective)
    node.move_tolerance = move_tolerance
    if configure is not None:
        configure(node)
    objective.hold([vertices for _, vertices in node.pinned_regions()])
    before = complex_counts(spacetime)
    cells_before = sorted(top_cells(spacetime))
    started = time.time()
    refusal = None
    limit = None
    trace = []
    objective.begin(time_limit_seconds)
    try:
        if moves:
            trace = node.run(
                max_iters=iteration_limit,
                n_candidate_moves=int(candidate_moves),
                grow_boundaries=False, beta=1.0, alpha0=1.0,
                tolerance=tolerance, max_lookahead=depth,
                relax_budget_per_move=update_limit,
                combinatorial_breadth=length)
        else:
            trace = node.run_stage2(beta=1.0, max_iters=iteration_limit,
                                    alpha0=1.0, tolerance=tolerance)
    except DeclaredLimitReached as error:
        limit = str(error)
    except (ValueError, ArithmeticError, RuntimeError) as error:
        refusal = str(error)
    seconds = time.time() - started
    final = node.spacetime()
    updates = objective.updates
    last = updates[-1] if updates else None
    if limit is not None or refusal is not None:
        # the engine returns its trace when the drive returns; a drive an
        # error ended keeps the residual norms of the points its steps were
        # proposed from and of the last accepted point, where the base is
        # left
        trace = objective.proposed_trace(final)
        points = objective.proposed_points(final)
    # a committed move changes the cells of the complex a step is proposed
    # on; every other entry of the engine's trace after the first is an
    # accepted relaxation update
    seen = [cells_before] + [update["cells"] for update in updates]
    seen.append(sorted(top_cells(final)))
    committed = sum(1 for first, second in zip(seen, seen[1:])
                    if first != second)
    accepted = max(len(trace) - 1 - committed, 0)
    if limit is not None or refusal is not None:
        # between two points of one complex lies an accepted relaxation
        # update
        accepted = sum(1 for first, second in zip(points, points[1:])
                       if first[0] == second[0])
    for update in updates:
        del update["cells"]
    norm = (trace[-1] if len(trace)
            else last["residual_norm"] if last else float("nan"))
    stationary = bool(node.last_stage2_stationary)
    moved = moves and int(node.last_stage1_lookahead) > 0
    if refusal is not None:
        stop = STOP_NO_STEP
        detail = "%s, after %d step proposals: %s" % (stop, len(updates),
                                                      refusal)
    elif limit is not None:
        stop = STOP_DECLARED_LIMIT
        detail = ("%s was reached after %d step proposals; the residual norm "
                  "at the last of them is %.3g" % (limit, len(updates), norm))
    elif stationary and not moved:
        stop = STOP_STATIONARY
        scored = objective.scored - (last["scored_before"] if last else 0)
        without = objective.undefined[
            (last["undefined_before"] if last else 0):]
        detail = ("%s after %d step proposals; the residual norm is %.3g"
                  % (stop, len(updates), norm))
        if scored:
            detail += ("; since the last proposal %d complexes were scored, "
                       "%d of them without a residual" % (scored,
                                                          len(without)))
            if without:
                detail += " (the last: %s)" % without[-1]
        if iteration_limit is not None:
            detail += ("; an iteration limit of %d was declared, and the "
                       "engine does not say whether it ended the drive"
                       % iteration_limit)
    else:
        # the engine returned with its last line search accepting a trial or
        # its last move update committing a move: a declared count ended it
        stop = STOP_DECLARED_LIMIT
        detail = ("a declared count ended the drive (iteration limit %s, "
                  "update limit %s) after %d step proposals; the residual "
                  "norm is %.3g" % (iteration_limit, update_limit,
                                    len(updates), norm))
    return {
        "spacetime": final,
        "node": node,
        "objective": objective,
        "trace": [float(value) for value in trace],
        "stop_reason": stop,
        "stop_detail": detail,
        "refusal": refusal,
        "moves_committed": committed,
        "accepted_updates": accepted,
        "undefined_reasons": undefined_reasons(objective.undefined),
        "moves": bool(moves),
        "combinatorial_depth": depth,
        "combinatorial_length": length,
        "candidate_moves": int(candidate_moves),
        "complex_before": before,
        "complex_after": complex_counts(final),
        "changed": sorted(top_cells(final)) != cells_before,
        "seconds": seconds,
    }


def relax(spacetime, declaration, geometry=None, mean_field=None,
          **options):
    """The drive of one complex as it stands, for one declared action: one
    sheet, the coordinates the geometry declaration names, and no Pachner
    move unless ``moves`` is given.

    ``declaration`` is the `JointActionDeclaration` of the complex. With
    ``geometry`` (a `HolomorphicRelaxationDeclaration`) the system is the
    stationarity at the carried state the action declares; with
    ``mean_field`` (a `SelfConsistentMeanFieldDeclaration`) it is the
    self-consistent system of the declared content, its fiber pinned at the
    starting point. The other arguments are those of `solve`.

    Returns the record of `solve` with the system (``system``) and the
    `Point` at the complex the drive ended on (``point``), and, for a mean
    field, the action and the report there (``action``, ``report``)."""
    if (geometry is None) == (mean_field is None):
        raise ValueError("declare the geometry of the system or the mean "
                         "field of its content, one of the two")
    options.setdefault("moves", False)
    if mean_field is None:
        system = GeometricSystem(lambda complex_: declaration,
                                 lambda support: geometry, 1)
    else:
        system = ContentSystem(lambda complex_: declaration,
                               lambda support: mean_field, spacetime, 1)
    start_scale = max((abs(length * length)
                       for _, _, length, _ in edge_fields(spacetime)),
                      default=0.0)
    record = solve(spacetime, system, **options)
    record["system"] = system
    record["point"] = system.point(record["spacetime"])
    if mean_field is not None:
        _, record["action"], record["report"] = system.read(
            record["spacetime"], start_scale)
    return record
