# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The spectral-moment part of S_0 as the holomorphic spectral constraints of
WP v17 §3.4, pinned on the occupied fiber of the recursion driver's per-cell
reads.

Terms used below:

* the *occupied fiber* of a content is the bands it occupies, chosen at the
  host and followed as the covariance's bands are; P_C is its Riesz projector
  and r its rank (three per band on the three-sheeted host);
* h_C = P_C h_1 P_C on the range of P_C is the compressed operator, and
  p_j(h_C) = tr(h_C^j) its power sums, the j-th power sum of the fiber's
  eigenvalues;
* *pinning* m_c power sums adds sum_{j <= m_c} xi_j (p_j(h_C) - p_j*) to the
  joint action with the targets p_j* the host's values and the complex
  multipliers xi_j, which at every point are the least-squares solution of
  the equations they enter linearly; the power sums are solved in a unit s,
  p_j(h_C / s), which describes the same constraints;
* the *Hessian along the Hellmann-Feynman force* is the Rayleigh quotient of
  the moment-constrained action's Hessian along the carried state's own force
  on the tangent space of the pinned constraints: the positivity condition of
  WP v17 line 265.

The host cells are those of the recursion's tick-0 run
(`_recursion_run_2026_09_23`), read with the declared action and at that
run's tolerances: no linear stiffness stand-in, kappa = 8 pi G entering only
through the Regge weight. A content's solve is the drive of
`baryon_poles.relax_content` (`tessera.drivers.cell_solve`), its Pachner
moves included; no solve below commits one.
"""
import cmath

import numpy as np
import pytest

from tessera import cobordism as cob
from tessera.drivers import baryon_poles as bp
from tessera.drivers import cell_solve as cs
from tessera.drivers import recursion as R

from tests.drivers import _recursion_run_2026_09_23 as RUN

FIRST_CELL = (0, 1, 2, 3)
SECOND_CELL = (0, 1, 3, 4)


def _config(cell, content, fiber_moments, fiber_pinning="power-sums"):
    config = bp.default_config(kappas=[1.0], betas=[1.0],
                               selected_contents=[tuple(content)],
                               fiber_moments=fiber_moments,
                               fiber_pinning=fiber_pinning,
                               tolerances=RUN.TOLERANCES)
    config["host_cell"] = RUN.HOST_CELLS[cell]
    config["held_sectors"] = R.held_sectors([[0, 1, 2, 3]], [1], 4)
    return config


def _host(cell, content):
    """The host of one read, its action under the declared action, the mean
    field's declaration and the band read at the host."""
    config = _config(cell, content, "0")
    spacetime = bp.build_host(config["edge_squared"], config["host_cell"])
    action = cob.JointAction(spacetime, bp.action_declaration(spacetime, 1.0,
                                                              1.0))
    mean_field = bp.mean_field_declaration(content, config, spacetime)
    read = cob.BandFollower(mean_field).read(action.carrier_operator())
    return spacetime, action, mean_field, read


def _relax(config, content, moments, scale):
    """The solve of `baryon_poles.relax_content` for one content with the
    number of pinned constraints and the unit they are solved in declared
    (``scale`` zero is the fiber's spectral radius at the host): the
    end-point report and the drive's record."""
    base = bp.build_base(config["edge_squared"], config["host_cell"])
    host = cs.sheeted_support(base, bp.SHEETS)

    def declare(spacetime):
        return bp.action_declaration(spacetime, 1.0, 1.0)

    def mean_field_of(support):
        declaration = bp.mean_field_declaration(content, config)
        declaration.geometry = bp.support_geometry(config, support, host)
        declaration.fiber_moments = moments
        declaration.fiber_moment_scale = scale
        return declaration

    system = cs.ContentSystem(declare, mean_field_of, base, bp.SHEETS)
    drive = cs.solve(base, system, **bp.solve_arguments(config))
    _, _, report = system.read(drive["spacetime"])
    return report, drive


def _pinned(spacetime, projector, orders, scale=1.0):
    """The declared action with the power sums of the given orders pinned on
    the fiber ``projector`` in the unit ``scale`` (targets zero, multipliers
    zero)."""
    declaration = bp.action_declaration(spacetime, 1.0, 1.0)
    declaration.moment_projector = list(np.asarray(projector).reshape(-1))
    declaration.moment_scale = scale
    constraints = []
    for order in orders:
        constraint = cob.SpectralMomentConstraint()
        constraint.order = order
        constraints.append(constraint)
    declaration.moment_constraints = constraints
    return cob.JointAction(spacetime, declaration)


def _fiber(read):
    return sum(np.asarray(band.projector) for band in read.bands)


def _band_pinned(spacetime, projectors, scale=1.0):
    """The declared action with the mean eigenvalue of each band pinned
    (`SpectralConstraintForm.BandMean`) in the unit ``scale`` (targets zero,
    multipliers zero)."""
    declaration = bp.action_declaration(spacetime, 1.0, 1.0)
    declaration.moment_band_projectors = [
        list(np.asarray(p).reshape(-1)) for p in projectors]
    declaration.moment_scale = scale
    constraints = []
    for band in range(len(projectors)):
        constraint = cob.SpectralMomentConstraint()
        constraint.form = cob.SpectralConstraintForm.BandMean
        constraint.band = band
        constraints.append(constraint)
    declaration.moment_constraints = constraints
    return cob.JointAction(spacetime, declaration)


# ------------------------------------------------ the fiber's power sums


def test_the_fiber_power_sums_are_those_of_its_eigenvalues():
    """On the host of (0134, 111) the fiber is the three lowest bands, rank
    nine, with eigenvalues -19.37, 0.313 and 5.0 once per sheet: p_j(h_C / s)
    is sum_a (lambda_a / s)^j over those nine, for every j and unit s."""
    spacetime, action, _, read = _host(SECOND_CELL, (1, 1, 1))
    assert sum(band.rank for band in read.bands) == 9
    fiber = np.concatenate([np.asarray(band.eigenvalues)
                            for band in read.bands])
    np.testing.assert_allclose(np.sort(fiber.real)[::3],
                               [-19.371, 0.313, 5.000], atol=1e-3)
    for scale in (1.0, 19.371):
        pinned = _pinned(spacetime, _fiber(read), range(1, 10), scale)
        expected = [np.sum((fiber / scale) ** j) for j in range(1, 10)]
        np.testing.assert_allclose(pinned.power_sums(), expected,
                                   rtol=1e-10)


def test_the_fiber_gradient_is_the_whole_derivative():
    """Away from the host, the analytic gradient of p_j(h_C), taken at fixed
    P_C, equals central differences of p_j(h_C(x)) with P_C rebuilt as the
    fiber's Riesz projector at every node: the projector's own variation
    contributes nothing, as it moves the range against the kernel. Checked for
    j = 1 to 6 of the fiber of (0123, 021), in the twelve shared
    coordinates."""
    content = (0, 2, 1)
    spacetime, _, mean_field, _ = _host(FIRST_CELL, content)
    classes, orientations = bp.sheet_edge_classes(spacetime)
    edges = spacetime.getEdgeList().toVector()
    for edge, c, o in zip(edges, classes, orientations):
        z = complex(edge.getLength()) ** 2 * (1.0 + 0.07 * (c - 2.5))
        edge.setLength(cmath.sqrt(z))
        edge.setPhase(complex(edge.getPhase()) + o * 0.05 * (c - 2))
    follower = cob.BandFollower(mean_field)

    def fiber_at():
        plain = cob.JointAction(spacetime,
                                bp.action_declaration(spacetime, 1.0, 1.0))
        return _fiber(follower.read(plain.carrier_operator()))

    follower.follow(follower.read(cob.JointAction(
        spacetime, bp.action_declaration(spacetime, 1.0, 1.0))
        .carrier_operator()))
    pinned = _pinned(spacetime, fiber_at(), range(1, 7), 3.0)
    gradients = [np.asarray(pinned.moment_gradient(k)) for k in range(6)]
    n = len(edges)
    radius = 1e-5
    worst = 0.0
    for c in range(6):
        members = [(i, o) for i, (cc, o) in enumerate(zip(classes,
                                                          orientations))
                   if cc == c]
        saved = [(complex(edges[i].getLength()), complex(edges[i].getPhase()))
                 for i, _ in members]
        for kind in ("length", "link"):
            sums = []
            for sign in (1.0, -1.0):
                for (i, o), (length, phase) in zip(members, saved):
                    if kind == "length":
                        edges[i].setLength(
                            cmath.sqrt(length * length + sign * radius))
                    else:
                        edges[i].setPhase(phase - 1j * o * sign * radius)
                pinned.set_moment_projector(list(fiber_at().reshape(-1)))
                sums.append(np.asarray(pinned.power_sums()))
                for (i, _), (length, phase) in zip(members, saved):
                    edges[i].setLength(length)
                    edges[i].setPhase(phase)
            difference = (sums[0] - sums[1]) / (2.0 * radius)
            if kind == "length":
                exact = np.array([sum(g[i] for i, _ in members)
                                  for g in gradients])
            else:
                exact = np.array([sum(o * g[n + i] for i, o in members)
                                  for g in gradients])
            worst = max(worst, float(np.max(np.abs(difference - exact)
                                            / np.abs(exact))))
    assert worst < 1e-7


def test_the_fiber_trace_obeys_the_euler_identity():
    """h_1 is homogeneous of degree -1 in the squared lengths, so at fixed
    P_C the trace of the compressed operator obeys
    sum_e z_e dp_1/dz_e = -p_1, exactly."""
    spacetime, _, _, read = _host(FIRST_CELL, (0, 2, 1))
    pinned = _pinned(spacetime, _fiber(read), [1])
    gradient = np.asarray(pinned.moment_gradient(0))
    z = np.array([complex(e.getLength()) ** 2
                  for e in spacetime.getEdgeList().toVector()])
    p1 = complex(pinned.power_sums()[0])
    assert abs(np.sum(z * gradient[:len(z)]) + p1) < 1e-10 * abs(p1)


def test_the_constraint_rows_of_the_jacobian_are_its_multiplier_columns():
    """In the Jacobian the pinned moments' rows are their analytic gradients,
    the transposes of the analytic multiplier columns, so that power sums
    that depend on others stay exactly dependent: on the sheeted host the
    nine power sums of a rank-nine fiber of three distinct eigenvalues span
    three directions, and the other six rows are dependent to rounding."""
    spacetime, _, _, read = _host(SECOND_CELL, (1, 1, 1))
    pinned = _pinned(spacetime, _fiber(read), range(1, 10), 19.371)
    config = _config(SECOND_CELL, (1, 1, 1), "0")
    geometry = bp.share_sheet_geometry(bp.relaxation_declaration(config),
                                       spacetime)
    geometry.relax_multipliers = True
    geometry.held_sectors = []
    relaxation = cob.HolomorphicRelaxation(pinned, geometry)
    size = relaxation.variable_count()
    assert size == 12 + 9
    jacobian = np.asarray(relaxation.jacobian()).reshape(size, size)
    rows = jacobian[12:, :12]
    columns = jacobian[:12, 12:]
    assert np.max(np.abs(rows - columns.T)) == 0.0
    singular = np.linalg.svd(rows, compute_uv=False)
    assert int(np.sum(singular > 1e-10 * singular[0])) == 3
    assert singular[3] < 1e-13 * singular[0]


# ------------------------------------------------ the pinned mean field


def test_a_full_band_pins_its_trace_with_a_multiplier_of_minus_one():
    """(0123, 030) with the trace of its fiber pinned (m_c = 1). The content
    fills its band, so Gamma = P_C and the fiber terms of the action combine
    to (1 + xi_1) tr(P_C h_1): the length equations then need xi_1 = -1,
    since the Euler identity makes the trace's length gradient nonzero
    wherever the trace is. The drive converges there in five accepted
    updates (residual norm 9.0, 1.6, 0.032, 2.6e-4, 1.4e-8, 7.3e-13), the
    pinned trace holds, and the geometry is Euclidean."""
    config = _config(FIRST_CELL, (0, 3, 0), "1")
    spacetime, action, report, drive = bp.relax_content((0, 3, 0), 1.0, 1.0,
                                                        config)
    assert report.converged
    assert drive["accepted_updates"] == 5 and drive["moves_committed"] == 0
    assert drive["stop_reason"] == cs.STOP_STATIONARY
    assert bp.relaxation_record(report, drive)["stop_reason"] == "converged"
    assert report.fiber_rank == 3 and len(report.multipliers) == 1
    assert abs(report.multipliers[0] + 1.0) < 1e-13
    assert abs(report.moment_residuals[0]) < 1e-12 * abs(
        report.moment_targets[0])
    # the target is the host's trace of the fiber, three times its eigenvalue
    _, _, _, read = _host(FIRST_CELL, (0, 3, 0))
    (band,) = read.bands
    assert report.moment_targets[0] == pytest.approx(
        3 * band.eigenvalues[0], rel=1e-10)
    # Euclidean to the solve's tolerance
    assert report.kontsevich_segal_margin == pytest.approx(np.pi, abs=1e-6)
    assert bp.hessian_sign(
        report.force_hessian, report.force_hessian_scale,
        RUN.TOLERANCES["hessian_reality_tolerance"]) == "positive"


def test_every_pinned_moment_holds_the_host_of_0134_111():
    """(0134, 111) with every power sum of its rank-nine fiber pinned
    (m_c = r). The least-squares multipliers at the host balance its whole
    stationarity force, the Villain force included, so the host is a
    stationary point of the constrained action and the drive stops there
    without an accepted update or a committed move (residual norm 7.9e-14).
    The targets are declared in the operator's own unit and solved in the
    unit s, so a pinned power sum holds to the rounding of that division.
    The joint Jacobian's rank deficiency is the three gauge directions and
    the six pinned power sums that depend on the other three, and on the
    real slice the Hessian along the Hellmann-Feynman force is real."""
    config = _config(SECOND_CELL, (1, 1, 1), "r")
    spacetime, action, report, drive = bp.relax_content((1, 1, 1), 1.0, 1.0,
                                                        config)
    assert report.converged
    assert drive["accepted_updates"] == 0 and drive["moves_committed"] == 0
    assert drive["stop_reason"] == cs.STOP_STATIONARY
    assert report.force_norm < 1e-12
    assert report.fiber_rank == 9 and len(report.multipliers) == 9
    assert max(abs(r) / abs(t) for r, t in zip(
        report.moment_residuals, report.moment_targets)) < 1e-15
    assert (report.jacobian_size, report.jacobian_rank) == (21, 12)
    assert report.moment_scale == pytest.approx(19.371, abs=1e-3)
    assert bp.hessian_sign(
        report.force_hessian, report.force_hessian_scale,
        RUN.TOLERANCES["hessian_reality_tolerance"]) in ("positive",
                                                         "negative")


def test_the_unit_of_the_power_sums_changes_no_solution():
    """(0123, 030) with every moment of its rank-three fiber pinned, solved
    in the fiber's spectral radius (the default unit) and in the operator's
    own unit. The fiber is one band, eigenvalue lambda once per sheet, so
    p_j(h_C) = 3 lambda^j and the three constraints are the one constraint
    lambda = lambda*: the fiber terms of the action are
    3 lambda + sum_j xi_j 3 lambda^j, whose length equations need
    sum_j j xi_j lambda^(j - 1) = -1. The three constraints are dependent,
    so the multipliers are not unique: in each unit they are that unit's
    least-squares ones, and they differ between the two. Each drive
    converges in six accepted updates to a solution of the same equations:
    the pin holds and the multipliers, in the operator's own unit, satisfy
    the same condition."""
    _, _, _, read = _host(FIRST_CELL, (0, 3, 0))
    (band,) = read.bands
    pinned = band.eigenvalues[0]
    multipliers = []
    for scale in (0.0, 1.0):
        config = _config(FIRST_CELL, (0, 3, 0), "r")
        report, drive = _relax(config, (0, 3, 0), 3, scale)
        assert report.converged
        assert drive["accepted_updates"] == 6
        assert drive["moves_committed"] == 0
        assert report.moment_scale == pytest.approx(
            abs(pinned) if scale == 0.0 else 1.0, rel=1e-12)
        (followed,) = report.bands
        value = followed.eigenvalues[0]
        assert abs(value - pinned) < 1e-9 * abs(pinned)
        xi = np.asarray(report.multipliers)
        combination = sum((j + 1) * xi[j] * value ** j for j in range(3))
        assert abs(combination + 1.0) < 1e-12
        multipliers.append(xi)
    assert np.max(np.abs(multipliers[0] - multipliers[1])) > 0.05


def test_more_moments_than_the_fiber_holds_are_refused():
    spacetime, action, mean_field, _ = _host(FIRST_CELL, (0, 3, 0))
    mean_field.fiber_moments = 4
    with pytest.raises(ValueError, match="the fiber has rank 3"):
        cob.SelfConsistentMeanField(action, mean_field).read()


# ------------------------------------------------ the bands' eigenvalues


def test_the_band_mean_is_the_band_eigenvalue():
    """A band-mean constraint's value is tr(P_b h_1) / (r_b s): on the host
    of (0134, 111) the mean of each occupied band's three eigenvalues, in
    the unit s. `power_sums` lists the same values under the power-sum
    form's name."""
    spacetime, _, _, read = _host(SECOND_CELL, (1, 1, 1))
    for scale in (1.0, 19.371):
        pinned = _band_pinned(spacetime, [b.projector for b in read.bands],
                              scale)
        expected = [np.mean(np.asarray(b.eigenvalues)) / scale
                    for b in read.bands]
        np.testing.assert_allclose(pinned.constraint_values(), expected,
                                   rtol=1e-12, atol=1e-13)
        assert list(pinned.power_sums()) == list(pinned.constraint_values())
    # the first power sum of the fiber is the rank-weighted sum of the means
    fiber = _pinned(spacetime, _fiber(read), [1])
    pinned = _band_pinned(spacetime, [b.projector for b in read.bands])
    assert complex(fiber.power_sums()[0]) == pytest.approx(
        sum(b.rank * v for b, v in zip(read.bands,
                                       pinned.constraint_values())),
        rel=1e-12)


def test_the_band_mean_gradient_is_the_whole_derivative():
    """Away from the host, the analytic gradient of lambda_b, the
    Hellmann-Feynman tr(P_b dh) / r_b at fixed P_b, equals central
    differences of tr(P_b(x) h(x)) / r_b with every band's Riesz projector
    rebuilt at every node, for the three occupied bands of (0134, 111) in
    the twelve shared coordinates; and the rank-weighted sum of the three
    gradients is the gradient of the fiber's first power sum."""
    content = (1, 1, 1)
    spacetime, _, mean_field, _ = _host(SECOND_CELL, content)
    classes, orientations = bp.sheet_edge_classes(spacetime)
    edges = spacetime.getEdgeList().toVector()
    for edge, c, o in zip(edges, classes, orientations):
        z = complex(edge.getLength()) ** 2 * (1.0 + 0.07 * (c - 2.5))
        edge.setLength(cmath.sqrt(z))
        edge.setPhase(complex(edge.getPhase()) + o * 0.05 * (c - 2))
    follower = cob.BandFollower(mean_field)

    def plain():
        return cob.JointAction(spacetime,
                               bp.action_declaration(spacetime, 1.0, 1.0))

    def bands_at():
        return follower.read(plain().carrier_operator()).bands

    follower.follow(follower.read(plain().carrier_operator()))
    bands = bands_at()
    assert len(bands) == 3
    pinned = _band_pinned(spacetime, [b.projector for b in bands], 3.0)
    gradients = [np.asarray(pinned.moment_gradient(k)) for k in range(3)]
    first = _pinned(spacetime, sum(np.asarray(b.projector) for b in bands),
                    [1], 3.0)
    np.testing.assert_allclose(
        sum(b.rank * g for b, g in zip(bands, gradients)),
        np.asarray(first.moment_gradient(0)), rtol=1e-10, atol=1e-13)
    n = len(edges)
    radius = 1e-5
    worst = 0.0
    for c in range(6):
        members = [(i, o) for i, (cc, o) in enumerate(zip(classes,
                                                          orientations))
                   if cc == c]
        saved = [(complex(edges[i].getLength()), complex(edges[i].getPhase()))
                 for i, _ in members]
        for kind in ("length", "link"):
            values = []
            for sign in (1.0, -1.0):
                for (i, o), (length, phase) in zip(members, saved):
                    if kind == "length":
                        edges[i].setLength(
                            cmath.sqrt(length * length + sign * radius))
                    else:
                        edges[i].setPhase(phase - 1j * o * sign * radius)
                pinned.set_moment_band_projectors(
                    [list(np.asarray(b.projector).reshape(-1))
                     for b in bands_at()])
                values.append(np.asarray(pinned.constraint_values()))
                for (i, _), (length, phase) in zip(members, saved):
                    edges[i].setLength(length)
                    edges[i].setPhase(phase)
            difference = (values[0] - values[1]) / (2.0 * radius)
            if kind == "length":
                exact = np.array([sum(g[i] for i, _ in members)
                                  for g in gradients])
            else:
                exact = np.array([sum(o * g[n + i] for i, o in members)
                                  for g in gradients])
            worst = max(worst, float(np.max(np.abs(difference - exact)
                                            / np.abs(exact))))
    assert worst < 1e-7


def test_pinning_the_band_eigenvalues_holds_the_host_of_0134_111():
    """(0134, 111) with the eigenvalue of each of its three occupied bands
    pinned (the drivers' declared pinning): three constraints in place of
    the nine power sums, whose targets are the bands' eigenvalues. The
    least-squares multipliers balance the host's whole stationarity force,
    the drive stops there without an accepted update or a committed move
    (residual norm 3.7e-14), every pinned eigenvalue holds, and the joint
    Jacobian's rank deficiency is the three gauge directions alone."""
    config = _config(SECOND_CELL, (1, 1, 1), "r", "eigenvalues")
    spacetime, action, report, drive = bp.relax_content((1, 1, 1), 1.0, 1.0,
                                                        config)
    assert report.fiber_constraint_form == \
        cob.FiberConstraintForm.BandEigenvalues
    assert report.converged
    assert drive["accepted_updates"] == 0 and drive["moves_committed"] == 0
    assert report.force_norm < 1e-12
    assert report.fiber_rank == 9 and len(report.multipliers) == 3
    np.testing.assert_allclose(np.asarray(report.moment_targets).real,
                               [-19.371, 0.313, 5.000], atol=1e-3)
    assert max(abs(r) for r in report.moment_residuals) == 0.0
    assert (report.jacobian_size, report.jacobian_rank) == (15, 12)
    assert report.moment_scale == pytest.approx(19.371, abs=1e-3)
    assert len(action.declaration.moment_band_projectors) == 3
    record = bp.relaxation_record(report, drive)
    assert record["fiber_pinning"] == "eigenvalues"
    assert record["fiber_moments"] == 3
    assert record["converged"] and record["stop_reason"] == "converged"
    assert record["iterations"] == 0 and record["moves_committed"] == 0


def test_a_full_band_pins_its_eigenvalue_with_a_multiplier_of_minus_three():
    """(0123, 030) with its one band's eigenvalue pinned. The content fills
    the band, so Gamma = P_b and the fiber terms of the action combine to
    (3 + xi) lambda: the length equations then need xi = -3 in the
    operator's own unit, in every unit the constraint is solved in. The
    pinned eigenvalue holds and the geometry is Euclidean."""
    _, _, _, read = _host(FIRST_CELL, (0, 3, 0))
    (band,) = read.bands
    pinned = band.eigenvalues[0]
    for scale in (0.0, 1.0):
        config = _config(FIRST_CELL, (0, 3, 0), "r", "eigenvalues")
        assert bp.mean_field_declaration(
            (0, 3, 0), config).fiber_constraint_form == \
            cob.FiberConstraintForm.BandEigenvalues
        report, drive = _relax(config, (0, 3, 0), 1, scale)
        assert report.converged
        assert drive["accepted_updates"] == 5
        assert drive["moves_committed"] == 0
        assert drive["stop_reason"] == cs.STOP_STATIONARY
        assert report.fiber_rank == 3 and len(report.multipliers) == 1
        assert report.moment_targets[0] == pytest.approx(pinned, rel=1e-10)
        assert abs(report.moment_residuals[0]) < 1e-12 * abs(pinned)
        assert abs(report.multipliers[0] + 3.0) < 1e-12
        assert report.kontsevich_segal_margin == pytest.approx(np.pi,
                                                               abs=1e-6)


def test_more_bands_than_the_content_occupies_are_refused():
    spacetime, action, mean_field, _ = _host(FIRST_CELL, (0, 3, 0))
    mean_field.fiber_constraint_form = cob.FiberConstraintForm.BandEigenvalues
    mean_field.fiber_moments = 2
    with pytest.raises(ValueError, match="occupies 1 bands"):
        cob.SelfConsistentMeanField(action, mean_field).read()


def test_one_pinned_constraint_per_occupied_band():
    """"r" is every occupied band's eigenvalue when the eigenvalues are
    pinned and the fiber's rank of power sums when they are; "bands" is the
    number of occupied bands under either; a count is itself, zero pinning
    nothing."""
    config = bp.default_config([1.0], [1.0])
    spacetime = bp.build_host()
    action = cob.JointAction(spacetime, bp.action_declaration(spacetime,
                                                              1.0, 1.0))
    for content, bands in (((0, 0, 3), 1), ((1, 0, 2), 2), ((1, 1, 1), 3)):
        declaration = bp.mean_field_declaration(content, config, spacetime)
        for pinning in ("eigenvalues", "power-sums"):
            assert bp.fiber_moment_count(declaration, action, "bands",
                                         pinning) == bands
            assert bp.fiber_moment_count(declaration, action, "1",
                                         pinning) == 1
            assert bp.fiber_moment_count(declaration, action, "0",
                                         pinning) == 0
        assert bp.fiber_moment_count(declaration, action, "r",
                                     "eigenvalues") == bands
        read = cob.BandFollower(declaration).read(action.carrier_operator())
        assert len(read.bands) == bands
        assert bp.fiber_moment_count(declaration, action, "r",
                                     "power-sums") == sum(
                                         b.rank for b in read.bands)
    assert bp._fiber_moments("bands") == "bands"
