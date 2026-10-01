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
  sheet: its gradient is added to R and its Hessian to J, so that the scalar
  stays the residual norm of one action. None is declared by default.

`MultiCobordism` does the rest as it stands (`solve`), and its mechanics are
the solve's: stage 1 scores every Pachner move of the base complex, which is
that move on every sheet, by ||R|| on the sheeted support and commits one
only when ||R|| falls by the move tolerance; stage 2 searches along the
direction from the declared first scale, which here is one, the step itself,
halving the scale as its line search does until a trial lowers ||R|| by the
tolerance. Every trial is judged on ||R|| itself, evaluated from the full
equations; the order of the step is the order of the proposal alone.

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

#: The stops of a drive, by name.
STOP_STATIONARY = "no move and no scaled step lowers the residual norm"
STOP_NO_STEP = "the step has no value at the point reached"
STOP_DECLARED_LIMIT = "a declared limit was reached"


class DeclaredLimitReached(Exception):
    """A limit the user declared on a drive was reached."""

#: One point of a system: the `HolomorphicRelaxation` posed on the sheeted
#: support of a base complex, the number of base edges (its squared-length
#: coordinates and its link coordinates, in the base complex's
#: `getEdgeList()` order), which fields are relaxed, the sheeted support
#: itself, the geometry declaration the system was posed with, and, for a
#: self-consistent system, the mean-field declaration and the reference
#: bands the point's bands are followed from (None otherwise).
Point = namedtuple("Point", "relaxation count lengths links support "
                            "geometry mean_field reference")

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
    return Point(relaxation, len(edge_fields(base)),
                 bool(geometry.relax_lengths), bool(geometry.relax_links),
                 support, geometry, mean_field, reference)


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


def least_squares_multipliers(action, geometry):
    """The multipliers of the constraints an action declares that leave the
    least of the stationarity equations they enter: those equations are
    linear in the multipliers, R = R_0 + G xi, so xi is the minimum-norm
    least-squares solution of G xi = -R_0, the singular values of G at or
    below the declared rank tolerance times the largest counted as zero. It
    is the rule `SelfConsistentMeanField.joint_system` installs the pinned
    fiber's multipliers by."""
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
    left, singular, right = np.linalg.svd(gradients, full_matrices=False)
    rank = (int(np.sum(singular > geometry.rank_tolerance * singular[0]))
            if len(singular) and singular[0] > 0.0 else 0)
    estimate = -(right[:rank].conj().T
                 @ ((left[:, :rank].conj().T @ zero) / singular[:rank]))
    return [complex(x) for x in estimate]


class GeometricSystem:
    """The stationarity system of the joint action at the carried state the
    action declares, on the sheeted support of any base complex.

    ``declare(spacetime)`` returns the `JointActionDeclaration` of a sheeted
    support's complex; ``geometry_of(support)`` returns a new
    `HolomorphicRelaxationDeclaration` for a `Sheeted` support (the relaxed
    fields, the support's edge classes and orientations, the held sectors on
    its sheets). When the declaration relaxes the multipliers of constraints
    the action declares, they are the least-squares ones at every point
    (`least_squares_multipliers`), as a content's are."""

    def __init__(self, declare, geometry_of, sheets):
        self._declare = declare
        self._geometry_of = geometry_of
        self.sheets = int(sheets)

    def point(self, base):
        support = sheeted_support(base, self.sheets)
        geometry = self._geometry_of(support)
        action = cob.JointAction(support.spacetime,
                                 self._declare(support.spacetime))
        if geometry.relax_multipliers and action.constraint_count() > 0:
            action.set_multipliers(least_squares_multipliers(action,
                                                             geometry))
        return _point(cob.HolomorphicRelaxation(action, geometry), geometry,
                      base, support)

    def accept(self, base):
        """An accepted point. Nothing is carried from one to the next; when
        the geometry declaration records terms, every term of the action
        there is returned (`cobordism.action_term_records`)."""
        support = sheeted_support(base, self.sheets)
        geometry = self._geometry_of(support)
        if not geometry.record_terms:
            return None
        action = cob.JointAction(support.spacetime,
                                 self._declare(support.spacetime))
        return cob.action_term_records(action, geometry)


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
    bands are followed from afterwards (`BAND_REFERENCES`)."""

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

    def _declaration(self, support):
        """The mean-field declaration of a support, the fiber pinned at the
        host's targets in the host's unit."""
        declaration = self._mean_field_of(support)
        if self._targets is not None:
            declaration.fiber_moment_unit_targets = list(self._targets)
            declaration.fiber_moment_scale = self._scale
        return declaration

    def _field(self, base):
        support = sheeted_support(base, self.sheets)
        action = cob.JointAction(support.spacetime,
                                 self._declare(support.spacetime))
        return (cob.SelfConsistentMeanField(
            action, self._declaration(support)), action, support)

    def _reference(self, support, action):
        """The reference bands read on a support's carrier cells, with the
        cells."""
        cells = support_cells(support, action.declaration.carrier_degree)
        return (reference_on(cells, self.reference_cells, self.reference),
                cells)

    def point(self, base):
        support = sheeted_support(base, self.sheets)
        declaration = self._declaration(support)
        action = cob.JointAction(support.spacetime,
                                 self._declare(support.spacetime))
        field = cob.SelfConsistentMeanField(action, declaration)
        reference, _ = self._reference(support, action)
        return _point(field.joint_system(reference), declaration.geometry,
                      base, support, declaration, reference)

    def iterate(self, base):
        """The measurements of a point (`SelfConsistentMeanField.iterate`),
        its bands followed from the reference and its covariance change taken
        from the last accepted point's, with the point's carrier cells."""
        field, action, support = self._field(base)
        reference, cells = self._reference(support, action)
        previous = (self._covariance
                    if len(self._covariance) == len(cells) ** 2 else [])
        return field.iterate(reference, previous), cells

    def accept(self, base):
        """An accepted point: its measurements, and the reference moved to
        its bands when they are followed from the last accepted point."""
        step, cells = self.iterate(base)
        self._covariance = covariance_of(step.bands)
        if self.band_reference == "previous":
            self.reference_cells = cells
            self.reference = references_of(step.bands)
        return step

    def read(self, base, start_scale=0.0):
        """The end-point read of a base complex: its sheeted support, the
        action there carrying the covariance, the constraints and their
        multipliers, and the report (`SelfConsistentMeanField.read`), the
        bands followed from the reference."""
        field, action, support = self._field(base)
        reference, _ = self._reference(support, action)
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
        except (ValueError, ArithmeticError, RuntimeError) as error:
            # the declared system is not posed on this complex: it has no
            # residual, the objective is infinite, a step that lands there is
            # shortened and a move that leads there is not committed; the
            # reason is kept for the report
            self.undefined.append(str(error))
            terms.joint_action_stationarity = float("inf")
            return terms
        if not math.isfinite(value):
            self.undefined.append("the residual norm is not finite")
            value = float("inf")
        terms.joint_action_stationarity = value
        return terms

    def direction(self, context):
        self._check_time()
        spacetime = context.scalar.spacetime
        measured = self._system.accept(spacetime)
        point = self._system.point(spacetime)
        added_residual, added_jacobian = self._stiffness(
            point, context.scalar, True)
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
        out = cob.ObjectiveDirection()
        offset = 0
        # stage 2 subtracts the direction: z - a, phi - a_phi. The step moves
        # z by its length block and multiplies each link by exp(delta) on the
        # base edge's stored orientation, that is phi by -i delta there.
        if point.lengths:
            out.ascent = -step[:point.count]
            offset = point.count
        else:
            out.ascent = np.zeros(point.count, dtype=complex)
        if point.links:
            out.phase_ascent = 1j * step[offset:offset + point.count]
        out.baseline = float(newton.residual_norm)
        out.baseline_computed = True
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
        the node, once per sheet. Empty when no stiffness is declared or the
        squared lengths are not relaxed."""
        weight = (float(context.moment_stiffness_weight)
                  * point.support.sheets)
        if weight == 0.0 or not point.lengths:
            return [], []
        spacetime = context.spacetime
        edges = point.count
        hodge = cob.HodgeLaplacian(spacetime)
        coefficients = list(context.moment_stiffness_coefficients)
        gradient = np.zeros(edges, dtype=complex)
        hessian = np.zeros((edges, edges), dtype=complex)
        for degree, reference in zip(context.moment_stiffness_degrees,
                                     context.moment_stiffness_reference):
            gradient += weight * np.asarray(
                hodge.spectralMomentStiffnessGradient(degree, reference,
                                                      coefficients))
            if not jacobian:
                continue
            for column in range(edges):
                unit = [0j] * edges
                unit[column] = 1.0 + 0j
                hessian[:, column] += weight * np.asarray(
                    hodge.spectralMomentStiffnessHessianProduct(
                        degree, reference, coefficients, unit))
        size = point.relaxation.variable_count()
        residual = np.zeros(size, dtype=complex)
        residual[:edges] = gradient
        if not jacobian:
            return list(residual), []
        matrix = np.zeros((size, size), dtype=complex)
        matrix[:edges, :edges] = hessian
        return list(residual), list(matrix.reshape(-1))


# ----------------------------------------------------------------- the drive


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
          direction_order=1, series=series_step, moves=True, move_lookahead=1,
          move_candidates=0, iteration_limit=None, update_limit=None,
          time_limit_seconds=None, configure=None):
    """Drive the base complex ``spacetime`` to a stationary point of
    ``system`` with `MultiCobordism`, whose mechanics are used as they are.

    With ``moves`` the combined drive runs (`MultiCobordism.run`): every
    iteration scores every Pachner move of the base complex
    (``move_candidates`` zero; a positive number draws that many) and, when
    no single move lowers the residual norm, every composition of up to
    ``move_lookahead`` of them, commits the best that lowers it by more than
    ``move_tolerance``, and relaxes the geometry until no trial of its line
    search lowers it by ``tolerance``. Without ``moves`` only the geometry
    relaxes (`MultiCobordism.run_stage2`). The first scale of the line
    search is one, the step itself.

    No count and no time is declared by default. ``iteration_limit`` is the
    engine's number of iterations (each one move update and a relaxation;
    without ``moves``, relaxation updates), ``update_limit`` its number of
    relaxation updates after each move update, and ``time_limit_seconds`` a
    wall-clock time read by the objective (`StationarityObjective.begin`).
    ``configure(node)`` may declare a pinned region or a spectral-moment
    stiffness on the node before the drive; none is declared otherwise.

    Returns the drive's record: the base complex it ended on (``spacetime``;
    a committed move replaces the object), the node and the objective, the
    trace of the residual norm, the stop by name with its detail, the number
    of committed move updates and of accepted relaxation updates, the counts
    of the base complex before and after, and the seconds it took."""
    objective = StationarityObjective(system, direction_order, series)
    node = cell_node(spacetime, objective)
    node.move_tolerance = move_tolerance
    if configure is not None:
        configure(node)
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
                max_iters=iteration_limit, n_candidate_moves=move_candidates,
                grow_boundaries=False, beta=1.0, alpha0=1.0,
                tolerance=tolerance, max_lookahead=int(move_lookahead),
                relax_budget_per_move=update_limit, combinatorial_breadth=0)
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
    # a committed move changes the cells of the complex a step is proposed
    # on; every other entry of the engine's trace after the first is an
    # accepted relaxation update
    seen = [cells_before] + [update["cells"] for update in updates]
    seen.append(sorted(top_cells(final)))
    committed = sum(1 for first, second in zip(seen, seen[1:])
                    if first != second)
    accepted = max(len(trace) - 1 - committed, 0)
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
