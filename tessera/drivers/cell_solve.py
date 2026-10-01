# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The cell solve as a drive of `MultiCobordism`.

The stationary points of the joint action (`cobordism.JointAction`: the Regge
term, the face-holonomy term and the matter term tr(Gamma h_1), with the
covariance Gamma rebuilt at every point by filling the declared bands of
h_1) are the zeros of its stationarity residual R over the squared lengths
z_e and the links U_e of every edge. `JointActionObjective` hands that system
to `MultiCobordism` as an injected objective:

* its scalar is F = ||R||^2, the squared Euclidean norm of the residual, a
  real number that vanishes exactly at the stationary points of the complex
  action, so descending F is finding them;
* a spectral-moment stiffness declared on the node
  (`MultiCobordism.set_moment_stiffness`) is a term of the action,
  beta_M S_M(z) with S_M the stiffness of `HodgeLaplacian` about the
  geometry it was declared at: its gradient is added to R and its Hessian
  to the Jacobian, so that F stays the squared residual of one action;
* its stage-2 direction is the step that solves R = 0 to the declared order
  about the point: order one is Newton's step, the solution d of J d = -R
  with J the closed-form Jacobian of R (`HolomorphicRelaxation.jacobian`).

`MultiCobordism` does the rest as it stands: stage 1 scores every Pachner
move on the whole complex by F and commits one only when F falls; stage 2
tries the step at scale one, halves it until F falls by the tolerance or the
scaled step moves no coordinate at the datatype's resolution, and holds
every pinned edge. Every trial is judged on F itself, which is evaluated
from the full equations; the order of the step is the order of the proposal
alone.

The bands a content fills are followed from their projectors at the host
(`BandReference`), which are carried by cell: on a complex a move has
changed, a reference projector is read on the cells that complex shares with
the host, and a cell the host does not have carries no weight in it.
"""
import numpy as np

from tessera import cobordism as cob

#: The largest order of the step, the declared maximum of every truncated
#: series of the stack.
MAXIMUM_DIRECTION_ORDER = 10


def carrier_cells(spacetime, degree=1):
    """The cells the carrier operator h_degree is over, as sorted vertex
    tuples in the chain complex's order, which is the operator's mode
    order."""
    complex_ = cob.ChainComplex.fromSpacetime(spacetime)
    return [tuple(sorted(int(v) for v in cell))
            for cell in complex_.kSimplexVertices(degree)]


def host_reference(action, mean_field):
    """The bands the declared content fills at the host, as the reference
    every later point is followed from: the cells of the host's carrier
    operator and one `BandReference` per occupied band, chosen at the host
    by the declared order."""
    follower = cob.BandFollower(mean_field)
    follower.follow(follower.read(action.carrier_operator()))
    return (carrier_cells(action.spacetime,
                          action.declaration.carrier_degree),
            list(follower.reference))


def reference_on(cells, reference_cells, reference):
    """The host's reference bands read on the complex whose carrier cells are
    ``cells``: each projector's entry between two cells is its entry at the
    host when the host has both cells and zero otherwise."""
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


class JointActionObjective(cob.CobordismObjective):
    """The joint action's stationarity as a `MultiCobordism` objective.

    ``declare(spacetime)`` returns the `JointActionDeclaration` of a
    complex (the weights and the forms of the terms); ``mean_field`` is the
    `SelfConsistentMeanFieldDeclaration` of the content (the covariance
    rule, the occupations and the order they are declared in), whose
    geometry declaration relaxes the squared lengths and the links of every
    edge as its own coordinate, with no multiplier; ``reference_cells`` and
    ``reference`` are the host's bands (`host_reference`).
    ``direction_order`` is the order of the step, one to
    `MAXIMUM_DIRECTION_ORDER`; ``rank_tolerance`` is the relative
    singular-value threshold of the step's linear solve, which is the
    minimum-norm one because the Jacobian is singular along every gauge
    direction."""

    def __init__(self, declare, mean_field, reference_cells, reference,
                 direction_order=1, rank_tolerance=1e-15):
        super().__init__()
        direction_order = int(direction_order)
        if not 1 <= direction_order <= MAXIMUM_DIRECTION_ORDER:
            raise ValueError(
                "the order of the step is an integer from 1 to %d; got %d"
                % (MAXIMUM_DIRECTION_ORDER, direction_order))
        if direction_order != 1:
            raise NotImplementedError(
                "the step of order %d needs the residual's derivative "
                "coefficients to that order along the step, which the "
                "truncated power-series arithmetic supplies; order 1 is "
                "available" % direction_order)
        self._declare = declare
        self._mean_field = mean_field
        self._reference_cells = [tuple(cell) for cell in reference_cells]
        self._reference = list(reference)
        self.direction_order = direction_order
        self.rank_tolerance = float(rank_tolerance)
        #: Why each scored point without a value had none, in the order met.
        self.undefined = []

    # -- the declared interface -------------------------------------------

    def name(self):
        return "joint_action"

    def term_names(self):
        return [cob.ObjectiveTermName.JOINT_ACTION_STATIONARITY]

    def is_target_conditioned(self):
        return False

    def terms(self, context):
        terms = cob.MultiCobordism.ObjectiveTerms()
        try:
            residual = self.residual(context)
        except (ValueError, RuntimeError) as error:
            # a point at which the action or its bands have no value (a
            # defective operator, a holonomy outside the domain of the
            # holonomy term) has no residual: its objective is infinite, so
            # stage 2 shortens a step that lands there, and the reason is
            # kept for the report
            self.undefined.append(str(error))
            terms.joint_action_stationarity = float("inf")
            return terms
        terms.joint_action_stationarity = float(
            np.vdot(residual, residual).real)
        return terms

    def direction(self, context):
        residual, jacobian = self.residual(context.scalar, jacobian=True)
        step = self.step(jacobian, residual)
        edges = len(residual) // 2
        out = cob.ObjectiveDirection()
        # stage 2 subtracts the direction: z - a, phi - a_phi. The step moves
        # z by its length block and multiplies each link by exp(delta), that
        # is phi by -i delta, on the stored orientation.
        out.ascent = -step[:edges]
        out.phase_ascent = 1j * step[edges:]
        out.baseline = float(np.vdot(residual, residual).real)
        out.baseline_computed = True
        out.is_step = True
        return out

    # -- the system --------------------------------------------------------

    def residual(self, context, jacobian=False):
        """R on the context's complex, the declared stiffness included; with
        ``jacobian``, the pair (R, J)."""
        spacetime = context.spacetime
        system = self.system(spacetime)
        residual = np.asarray(system.residual(), dtype=complex).copy()
        count = len(residual)
        matrix = (np.asarray(system.jacobian(), dtype=complex)
                  .reshape(count, count).copy() if jacobian else None)
        weight = float(context.moment_stiffness_weight)
        if weight != 0.0:
            edges = count // 2
            hodge = cob.HodgeLaplacian(spacetime)
            coefficients = list(context.moment_stiffness_coefficients)
            for degree, reference in zip(context.moment_stiffness_degrees,
                                         context.moment_stiffness_reference):
                residual[:edges] += weight * np.asarray(
                    hodge.spectralMomentStiffnessGradient(
                        degree, reference, coefficients))
                if matrix is None:
                    continue
                for column in range(edges):
                    unit = [0j] * edges
                    unit[column] = 1.0 + 0j
                    matrix[:edges, column] += weight * np.asarray(
                        hodge.spectralMomentStiffnessHessianProduct(
                            degree, reference, coefficients, unit))
        return (residual, matrix) if jacobian else residual

    def system(self, spacetime):
        """The stationarity system of the joint action on ``spacetime``, the
        covariance rebuilt from the bands that continue the host's: its
        ``residual()`` is R over the squared lengths and then the links of
        the edges in `getEdgeList()` order, and its ``jacobian()`` is the
        closed-form Jacobian of R."""
        declaration = self._declare(spacetime)
        action = cob.JointAction(spacetime, declaration)
        cells = carrier_cells(spacetime, declaration.carrier_degree)
        reference = reference_on(cells, self._reference_cells,
                                 self._reference)
        return cob.SelfConsistentMeanField(
            action, self._mean_field).joint_system(reference)

    def step(self, jacobian, residual):
        """The order-one step: the minimum-norm solution d of J d = -R, the
        singular values at or below ``rank_tolerance`` times the largest
        counted as zero."""
        u, singular, vh = np.linalg.svd(jacobian)
        if not len(singular) or singular[0] == 0.0:
            return np.zeros(len(residual), dtype=complex)
        rank = int(np.sum(singular > self.rank_tolerance * singular[0]))
        coefficients = (u[:, :rank].conj().T @ (-residual)) / singular[:rank]
        return vh[:rank].conj().T @ coefficients


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
