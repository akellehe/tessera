# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The cell solve as a drive of `MultiCobordism` (`tessera.drivers.cell_solve`).

Terms used below:

* the *host* is the declared three-sheeted host of `baryon_poles.build_host`:
  three disjoint regular tetrahedra of squared edge length 8, each carrying
  the monopole connection, 18 edges in all;
* the *residual* R is the stationarity residual of the joint action over the
  squared lengths and then the links of the 18 edges in `getEdgeList()`
  order, 36 complex numbers, with the covariance rebuilt at the point by
  filling the declared bands of h_1;
* the *objective* is F = ||R||^2, the scalar `MultiCobordism` descends;
* the *step* of order one is the minimum-norm solution d of J d = -R, with
  J the closed-form Jacobian of R.

What is asserted: the objective a node reports is the squared norm of the
residual; the direction handed to stage 2 is the order-one step in the
engine's sign and phase conventions; stage 2 tries that step at scale one;
a point without a value scores as infinite; the bands are followed from the
host's reference, carried by cell; the drives carry no count unless one is
declared.
"""
import numpy as np
import pytest

from tessera import cobordism as cob
from tessera.drivers import baryon_poles as bp
from tessera.drivers import cell_solve as cs

CONTENT = (1, 1, 1)


def _setup(content=CONTENT):
    """The host, the action's declaration rule, the content's mean-field
    declaration with every edge its own coordinate and nothing pinned, the
    host's reference bands, and the objective."""
    config = bp.default_config(kappas=[1.0], betas=[1.0],
                               selected_contents=[tuple(content)])
    config["held_sectors"] = []
    spacetime = bp.build_host(config["edge_squared"])

    def declare(complex_):
        return bp.action_declaration(complex_, 1.0, 1.0,
                                     config["regge_hinges"])

    mean_field = bp.mean_field_declaration(content, config)
    mean_field.fiber_moments = 0
    action = cob.JointAction(spacetime, declare(spacetime))
    cells, reference = cs.host_reference(action, mean_field)
    objective = cs.JointActionObjective(declare, mean_field, cells, reference)
    return spacetime, declare, mean_field, cells, reference, objective


def _squared_norm(vector):
    return float(np.vdot(vector, vector).real)


def test_the_objective_is_the_squared_norm_of_the_residual():
    """The node's objective, its one nonzero term and the objective's own
    system agree: F = ||R||^2 with R the 36 residuals of the host."""
    spacetime, _, _, _, _, objective = _setup()
    residual = np.asarray(objective.system(spacetime).residual())
    assert len(residual) == 36
    node = cs.cell_node(spacetime, objective)
    assert node.objective() == pytest.approx(_squared_norm(residual),
                                             rel=1e-13)
    terms = node.objective_terms()
    assert terms.joint_action_stationarity == node.objective()
    assert terms.regge_stationarity == 0.0
    assert terms.hodge_stationarity == 0.0
    assert objective.term_names() == ["joint_action_stationarity"]
    assert objective.name() == "joint_action"
    assert not objective.is_target_conditioned()


def test_the_direction_is_the_order_one_step():
    """The direction handed to stage 2 is the step d with J d = -R on the
    range of J, in the engine's conventions: stage 2 subtracts ``ascent``
    from z and ``phase_ascent`` from the stored phase, and the step moves z
    by its length block and the phase by -i times its link block. Its
    baseline is F and it is declared a step."""
    spacetime, _, _, _, _, objective = _setup()
    system = objective.system(spacetime)
    residual = np.asarray(system.residual())
    jacobian = np.asarray(system.jacobian()).reshape(36, 36)
    context = cob.ObjectiveDirectionContext()
    scalar = cob.ObjectiveContext()
    scalar.spacetime = spacetime
    context.scalar = scalar
    context.edge_count = 18
    direction = objective.direction(context)
    step = np.concatenate([-np.asarray(direction.ascent),
                           np.asarray(direction.phase_ascent) / 1j])
    # J d + R is orthogonal to the range of J: the least-squares solution
    left = jacobian.conj().T @ (jacobian @ step + residual)
    assert np.linalg.norm(left) < 1e-9 * np.linalg.norm(
        jacobian.conj().T @ residual)
    # and d is the minimum-norm one: orthogonal to the null space of J
    _, singular, vh = np.linalg.svd(jacobian)
    null = vh[singular <= 1e-12 * singular[0]]
    assert null.shape[0] >= 3
    assert np.linalg.norm(null @ step) < 1e-9 * np.linalg.norm(step)
    assert direction.is_step and direction.baseline_computed
    assert direction.baseline == pytest.approx(_squared_norm(residual),
                                               rel=1e-13)


def test_stage_two_tries_the_step_at_scale_one():
    """One stage-2 update from the host with a first scale of 0.05 declared
    for gradients: the objective's direction is a step, so the first trial is
    the step itself, and the accepted point is z - ascent, phi - phase_ascent
    exactly when that trial lowers F by the tolerance."""
    spacetime, _, _, _, _, objective = _setup()
    context = cob.ObjectiveDirectionContext()
    scalar = cob.ObjectiveContext()
    scalar.spacetime = spacetime
    context.scalar = scalar
    context.edge_count = 18
    direction = objective.direction(context)
    edges = spacetime.getEdgeList().toVector()
    before_z = np.array([complex(e.getLength()) ** 2 for e in edges])
    before_phase = np.array([complex(e.getPhase()) for e in edges])
    node = cs.cell_node(spacetime, objective)
    trace = list(node.run_stage2(beta=1.0, max_iters=1, alpha0=0.05,
                                 tolerance=1e-15))
    assert len(trace) == 2 and trace[1] < trace[0]
    after = node.spacetime().getEdgeList().toVector()
    after_z = np.array([complex(e.getLength()) ** 2 for e in after])
    after_phase = np.array([complex(e.getPhase()) for e in after])
    np.testing.assert_allclose(after_z,
                               before_z - np.asarray(direction.ascent),
                               rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        after_phase, before_phase - np.asarray(direction.phase_ascent),
        rtol=1e-12, atol=1e-12)


def test_a_point_without_a_value_scores_as_infinite():
    """A complex on which the action or its bands have no value has no
    residual: the objective's scalar there is infinite, which stage 2 reads
    as a trial that does not lower F, and the reason is kept."""
    spacetime, _, _, _, _, objective = _setup()

    def no_value(context, jacobian=False):
        raise ValueError("the operator is defective")

    objective.residual = no_value
    context = cob.ObjectiveContext()
    context.spacetime = spacetime
    terms = objective.terms(context)
    assert terms.joint_action_stationarity == float("inf")
    assert objective.undefined == ["the operator is defective"]


def test_the_bands_are_followed_from_the_reference():
    """`BandFollower.set_reference` makes a read follow stored bands: at the
    host, the reference read back from a follower that chose the bands there
    reproduces its covariance, and the joint system built from that
    reference has the residual of the system that chooses the bands at the
    point."""
    spacetime, declare, mean_field, cells, reference, _ = _setup()
    action = cob.JointAction(spacetime, declare(spacetime))
    operator = action.carrier_operator()
    chosen = cob.BandFollower(mean_field)
    read = chosen.read(operator)
    assert not chosen.following
    seeded = cob.BandFollower(mean_field)
    seeded.set_reference(reference)
    assert seeded.following and len(seeded.reference) == len(reference)
    np.testing.assert_allclose(np.asarray(seeded.read(operator).covariance),
                               np.asarray(read.covariance), atol=1e-12)
    field = cob.SelfConsistentMeanField(action, mean_field)
    np.testing.assert_allclose(
        np.asarray(field.joint_system(reference).residual()),
        np.asarray(field.joint_system().residual()), atol=1e-12)
    wrong = cob.BandReference()
    wrong.rank = 3
    wrong.occupation = 1.0
    wrong.projector = [0j] * 4
    seeded.set_reference([wrong])
    with pytest.raises(ValueError, match="reference projector"):
        seeded.read(operator)


def test_a_reference_is_carried_by_cell():
    """`reference_on` reads a host projector on another list of cells: an
    entry between two cells the host has is the host's entry, wherever the
    cells sit in the list, and a cell the host lacks carries nothing."""
    host_cells = [(0, 1), (0, 2), (1, 2)]
    band = cob.BandReference()
    band.occupation = 2.0
    band.rank = 1
    band.declared_index = 1
    band.declared_positions = [2]
    projector = np.arange(9, dtype=complex).reshape(3, 3)
    band.projector = list(projector.reshape(-1))
    cells = [(1, 2), (5, 6), (0, 1)]
    (moved,) = cs.reference_on(cells, host_cells, [band])
    read = np.asarray(moved.projector).reshape(3, 3)
    assert read[0, 0] == projector[2, 2] and read[0, 2] == projector[2, 0]
    assert read[2, 0] == projector[0, 2] and read[2, 2] == projector[0, 0]
    assert np.all(read[1] == 0) and np.all(read[:, 1] == 0)
    assert (moved.occupation, moved.rank, moved.declared_index,
            list(moved.declared_positions)) == (2.0, 1, 1, [2])


def test_the_order_of_the_step_is_declared_between_one_and_ten():
    spacetime, declare, mean_field, cells, reference, _ = _setup()
    for order in (0, 11):
        with pytest.raises(ValueError, match="from 1 to 10"):
            cs.JointActionObjective(declare, mean_field, cells, reference,
                                    direction_order=order)
    assert cs.JointActionObjective(
        declare, mean_field, cells, reference,
        direction_order=1).direction_order == 1


def test_the_drives_carry_no_count_unless_one_is_declared():
    """The stage drives of `MultiCobordism` take their counts as options
    that are None by default; the candidate walk is exhaustive by default
    and the tolerance is 1e-15."""
    for drive, counts in ((cob.MultiCobordism.run_stage1, ("max_steps",)),
                          (cob.MultiCobordism.run_stage2, ("max_iters",)),
                          (cob.MultiCobordism.run,
                           ("max_iters", "relax_budget_per_move"))):
        signature = drive.__doc__.splitlines()[0]
        for name in counts:
            assert name in signature
            assert "%s: int | None = None" % name in signature \
                or "%s: Optional[int] = None" % name in signature
    assert "n_candidate_moves: int = 0" in \
        cob.MultiCobordism.run.__doc__.splitlines()[0]
    assert "tolerance: float = 1e-15" in \
        cob.MultiCobordism.run.__doc__.splitlines()[0]
    assert "tolerance: float = 1e-15" in \
        cob.MultiCobordism.run_stage2.__doc__.splitlines()[0]
