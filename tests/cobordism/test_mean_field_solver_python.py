# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The self-consistent mean-field solve of the recursion driver's per-cell
reads, held to reads on the host cells of the mean-field convergence
investigation (#1257; `reports/design/mean_field_convergence_investigation.md`
in the notes repository) under the equations as they stand.

Terms used below:

* a *read* is one (host cell, content) pair of the tick-0 run of 2026-09-25,
  whose host cells are those of `_recursion_run_2026_09_23` (the level-0 host
  is stationary as built, so both runs read the same cells);
* a *content* (n_0, n_1, n_2) places n_b quarks in band b of the covariant
  operator h_1(z, U), the bands counted in ascending order of real part at
  the host;
* the *pinned fiber* is the spectral-moment part of S_0 as the driver
  declares it (WP v17 §3.4): the eigenvalue of every occupied band pinned at
  its value at the host, one multiplier per band among the solve's unknowns.
  A cell has no interior hinge, so with no fiber moment pinned the length
  equations are the matter force alone, homogeneous of degree -2 in z, which
  vanishes only at infinite length;
* the *self-consistent force* F_sc(z, U) is the stationarity force of the
  joint action on the six shared squared lengths z and six links U with the
  covariance Gamma rebuilt by the band rule at (z, U); its zeros, at which
  the pinned fiber's residuals vanish too, are the stationary pairs of WP
  v17 lines 259 and 263;
* the *held set* is the declared monopole number and the unit moduli of the
  face holonomies on the cell's four faces (WP v17 line 508).

What is asserted:

* Newton's method on the joint system converges (0123, 201) to 1e-9 in a
  few dozen steps, on the real slice, and the point it reaches satisfies the
  equations of the joint action evaluated independently;
* the occupied bands are chosen at the host and followed by continuation,
  every iterate's overlaps and crossings are reported, and re-sorting stays
  available by name; on (0123, 003) continuation follows the occupied band
  through a crossing to the fixed point, which is no fixed point of the
  re-sorted equations;
* a read without a stationary point ends by itself and by name (no damped
  step reduced the residual, its detail giving how far along the Newton
  direction the sector guard refused; the residual at its floor on the held
  set) instead of freezing or repeating, and a read is refused when the
  geometry it ended at is not Kontsevich-Segal allowable, whether or not it
  converged there; a squared length beyond the largest finite double is the
  only bound on the lengths;
* the held-modulus step is the constrained Newton step, which leaves far less
  of the linearized residual than the projection of the unconstrained step
  and keeps every held modulus exactly.
"""
import numpy as np
import pytest

from tessera import cobordism as cob
from tessera.drivers import baryon_poles as bp
from tessera.drivers import recursion as R

from tests.drivers import _recursion_run_2026_09_23 as RUN

FIRST_CELL = (0, 1, 2, 3)


def _config(cell, content, selection="continuation",
            fiber_moments=bp.DECLARED_FIBER_MOMENTS, limits=None):
    """The configuration `recursion.cell_reads` hands `baryon_poles` for one
    tick-0 host cell (kappa = beta = 1, the Villain term, the cell's four
    faces held, the eigenvalue of every occupied band pinned at the host),
    at the run's declared tolerances, with the declared band selection;
    ``fiber_moments=0`` pins nothing, and ``limits`` declares any of
    `baryon_poles.LIMITS` (none by default)."""
    config = bp.default_config(kappas=[1.0], betas=[1.0],
                               selected_contents=[tuple(content)],
                               fiber_moments=fiber_moments,
                               tolerances=RUN.TOLERANCES, limits=limits)
    config["host_cell"] = RUN.HOST_CELLS[cell]
    config["held_sectors"] = R.held_sectors([[0, 1, 2, 3]], [1], 4)
    config["band_selection"] = selection
    return config


def _relax(cell, content, **options):
    config = _config(cell, content, **options)
    spacetime, action, report = bp.relax_content(content, 1.0, 1.0, config)
    return spacetime, action, report, config


def _refusal(report, config):
    """The read's refusal at the run's declared allowability tolerance, as
    the driver decides it."""
    return bp.read_refusal(report, bp.declared_tolerance(
        config, "allowability_tolerance"))


def _class_force(action, spacetime):
    """The force on the twelve shared coordinates, from the action's own
    per-edge stationarity: per class, the sum of dS/dz_e, then the sum of
    U_e dS/dU_e on the class's orientation."""
    classes, orientations = bp.sheet_edge_classes(spacetime)
    lengths = np.asarray(action.length_stationarity())
    links = np.asarray(action.link_stationarity())
    force = np.zeros(12, dtype=complex)
    for edge, (c, o) in enumerate(zip(classes, orientations)):
        force[c] += lengths[edge]
        force[6 + c] += o * links[edge]
    return force


# ------------------------------------------------------------ joint Newton


def test_the_joint_newton_converges_a_read_in_a_few_dozen_steps():
    """(0123, 201): Newton's method on the joint system reaches the fixed
    point in 22 steps, every one accepted, on the real slice, with the joint
    Jacobian of rank 11 of 14 (twelve shared coordinates and the two
    multipliers of the pinned fiber; the three gauge directions are its null
    space) and a separated rank decision; the pinned eigenvalues hold at the
    fixed point."""
    spacetime, action, report, _ = _relax(FIRST_CELL, (2, 0, 1))
    assert report.band_selection == cob.BandSelection.Continuation
    assert report.converged
    assert report.stop_reason == cob.RelaxationStop.Converged
    assert cob.relaxation_stop_name(report.stop_reason) == "converged"
    assert report.force_norm <= 1e-9
    assert 15 <= report.iterations <= 30
    assert len(report.steps) == report.iterations + 1
    # one Newton iteration from every iterate but the last, each accepted
    for step in report.steps[:-1]:
        assert step.newton_iterated and step.newton.accepted
        assert step.newton.step_norm > 0.0
        assert step.geometry_stop_reason == cob.RelaxationStop.Continued
    assert not report.steps[-1].newton_iterated
    assert report.steps[-1].geometry_stop_reason == \
        cob.RelaxationStop.Converged
    assert (report.jacobian_size, report.jacobian_rank) == (14, 11)
    assert report.rank_gap > 1e6
    assert report.fiber_rank == 6 and len(report.multipliers) == 2
    assert np.max(np.abs(report.moment_residuals)) <= 1e-9
    # a Euclidean cell: every eigenvalue of the metric positive
    assert report.kontsevich_segal_margin == pytest.approx(np.pi, abs=1e-9)
    z = np.asarray(bp.sheet_squared_lengths(spacetime, 0))
    assert np.max(np.abs(z.imag)) < 1e-9 and np.min(z.real) > 0
    faces = np.asarray(action.face_holonomies())
    assert np.max(np.abs(np.abs(faces) - 1.0)) < 1e-9
    assert report.action_available and report.action_unavailable == ""


def test_the_fixed_point_satisfies_the_equations():
    """At the point the joint Newton reached, the covariance it reports is the
    band filling of h_1 there (it commutes with h_1 and carries the three
    quarks), and a joint action built afresh with that covariance, the pinned
    fiber's constraints and the multipliers the solve reports has a force
    below the declared tolerance on the twelve shared coordinates and
    constraint residuals below it."""
    spacetime, action, report, _ = _relax(FIRST_CELL, (2, 0, 1))
    gamma = np.asarray(report.covariance).reshape(18, 18)
    solved = action.declaration
    declaration = bp.action_declaration(spacetime, 1.0, 1.0)
    declaration.covariance = list(report.covariance)
    declaration.moment_constraints = solved.moment_constraints
    declaration.moment_projector = solved.moment_projector
    declaration.moment_band_projectors = solved.moment_band_projectors
    declaration.moment_scale = solved.moment_scale
    # the declaration pins the scaled operator h_C / s, the report speaks in
    # the eigenvalue's own unit: its multipliers are the declaration's over
    # s and its targets the declaration's times s
    s = declaration.moment_scale
    np.testing.assert_allclose(
        [c.multiplier for c in declaration.moment_constraints],
        [m * s for m in report.multipliers], rtol=1e-12)
    np.testing.assert_allclose(
        [c.target * s for c in declaration.moment_constraints],
        list(report.moment_targets), rtol=1e-12)
    fresh = cob.JointAction(spacetime, declaration)
    h = bp.matrix(fresh.carrier_operator())
    assert np.linalg.norm(gamma @ h - h @ gamma) < 1e-9 * np.linalg.norm(h)
    assert np.trace(gamma).real == pytest.approx(3.0, abs=1e-10)
    assert np.linalg.norm(_class_force(fresh, spacetime)) <= 1e-9
    assert np.max(np.abs(fresh.moment_residuals())) <= 1e-9


# ----------------------------------------------------- band continuation


def _family(eigenvalues_of_t, t, seed=7):
    """A non-normal operator with the given eigenvalues and a fixed,
    non-unitary eigenvector matrix, flat row-major."""
    rng = np.random.default_rng(seed)
    n = len(eigenvalues_of_t(0.0))
    vectors = np.eye(n) + 0.3 * rng.standard_normal((n, n)) \
        + 0.2j * rng.standard_normal((n, n))
    operator = vectors @ np.diag(eigenvalues_of_t(t)) @ np.linalg.inv(vectors)
    return list(operator.reshape(-1)), vectors


def _follower(occupations, selection):
    declaration = cob.SelfConsistentMeanFieldDeclaration()
    declaration.covariance_rule = cob.CovarianceRule.BandFilling
    declaration.band_occupations = list(occupations)
    declaration.band_selection = selection
    return cob.BandFollower(declaration)


def _projector(vectors, columns):
    inverse = np.linalg.inv(vectors)
    return vectors[:, columns] @ inverse[columns, :]


def test_a_band_is_followed_across_a_constructed_crossing():
    """A simple band at 1 + t crosses a simple band at 2 - t at t = 1/2. The
    band occupied at t = 0 (the lowest, declared band 0) is followed through
    the crossing: after it, the covariance is still the projector of the
    eigenvalue 1 + t, its overlap with the previous point's projector stays
    one, and the report says the band crossed (it now holds place 1 in the
    ascending real-part order, having been chosen at place 0)."""
    eigenvalues = lambda t: np.array([1.0 + t, 2.0 - t, 3.0, 4.5])  # noqa
    follower = _follower([1.0], cob.BandSelection.Continuation)
    for t in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        operator, vectors = _family(eigenvalues, t)
        read = follower.read(operator)
        (band,) = read.bands
        assert band.declared_index == 0 and band.rank == 1
        assert band.eigenvalues[0] == pytest.approx(1.0 + t, abs=1e-10)
        assert abs(band.overlap - 1.0) < 1e-10
        assert band.declared_positions == [0]
        assert band.positions == ([0] if t < 0.5 else [1])
        assert band.crossed == (t > 0.5) == read.crossing
        np.testing.assert_allclose(
            np.asarray(read.covariance).reshape(4, 4),
            _projector(vectors, [0]), atol=1e-10)
        follower.follow(read)


def test_a_band_of_three_sheets_is_followed_across_another():
    """Three identical sheets, as on the recursion's host: every band has one
    mode per sheet, rank three. The band at 1 + t, holding two particles,
    crosses the band at 2 - t; it is followed as one band of rank three,
    after the crossing it holds places 3 to 5, and the covariance is two
    thirds of its projector, the direct sum of one sheet's."""
    eigenvalues = lambda t: np.array([1.0 + t, 2.0 - t, 4.0])  # noqa
    follower = _follower([2.0], cob.BandSelection.Continuation)
    for t in (0.0, 0.3, 0.7, 1.0):
        sheet, vectors = _family(eigenvalues, t, seed=11)
        operator = np.kron(np.eye(3), np.asarray(sheet).reshape(3, 3))
        read = follower.read(list(operator.reshape(-1)))
        (band,) = read.bands
        assert band.rank == 3 and band.occupation == 2.0
        assert list(read.ranks) == [3, 3, 3]
        assert abs(band.overlap - 1.0) < 1e-9
        assert band.positions == ([0, 1, 2] if t < 0.5 else [3, 4, 5])
        assert not band.ambiguous
        np.testing.assert_allclose(
            np.asarray(read.covariance).reshape(9, 9),
            (2.0 / 3.0) * np.kron(np.eye(3), _projector(vectors, [0])),
            atol=1e-9)
        follower.follow(read)


def test_re_sorting_exchanges_the_occupied_band_and_says_so():
    """The explicitly named alternative, `SortEveryIterate`, re-selects the
    lowest band at every point: past the crossing it fills the eigenvalue
    2 - t instead, no band is reported crossed, and the exchange shows as an
    overlap of the new projector with the previous one near zero."""
    eigenvalues = lambda t: np.array([1.0 + t, 2.0 - t, 3.0, 4.5])  # noqa
    follower = _follower([1.0], cob.BandSelection.SortEveryIterate)
    overlaps = []
    for t in (0.0, 0.4, 0.6, 1.0):
        operator, vectors = _family(eigenvalues, t)
        read = follower.read(operator)
        (band,) = read.bands
        assert not band.crossed and band.positions == [0]
        expected = 0 if t < 0.5 else 1
        np.testing.assert_allclose(
            np.asarray(read.covariance).reshape(4, 4),
            _projector(vectors, [expected]), atol=1e-10)
        overlaps.append(abs(band.overlap))
        follower.follow(read)
    assert overlaps[1] == pytest.approx(1.0, abs=1e-10)
    assert overlaps[2] < 0.5  # the exchange between t = 0.4 and t = 0.6
    assert overlaps[3] == pytest.approx(1.0, abs=1e-10)


def test_continuation_and_re_sorting_part_at_a_crossing():
    """(0123, 003) under the two named rules. Followed from the host, the
    declared band 2 crosses below the host's middle band; the solve reports
    the crossing and converges to the fixed point with the occupied band at
    2.898, its pinned value, z = (32.144, 14.209, 17.130, 15.609, 15.144,
    17.419). At that point the followed band holds places 3 to 5 of the
    spectrum in ascending order of real part, and the band the declared
    order selects there, places 6 to 8, sits at 3.925, off the pinned value:
    the point is no fixed point of the re-sorted equations. The re-sorted
    solve itself is not run: past the crossing every Newton step exchanges
    the occupation, the residual is discontinuous across the exchange, and
    the solve follows the surface of the exchange in steps that each lower
    the residual by parts in ten million (measured: residual norm 0.3601
    after 312 iterates, #1319), so it has no end to assert on."""
    spacetime, action, followed, _ = _relax(FIRST_CELL, (0, 0, 3))
    assert followed.converged and followed.band_crossing_iterates >= 1
    (band,) = followed.bands
    assert band.declared_index == 2
    assert list(band.declared_positions) == [6, 7, 8]
    assert list(band.positions) == [3, 4, 5] and band.crossed
    assert band.eigenvalues[0].real == pytest.approx(2.8977, abs=5e-4)
    np.testing.assert_allclose(
        np.asarray(bp.sheet_squared_lengths(spacetime, 0)).real,
        [32.144, 14.209, 17.130, 15.609, 15.144, 17.419], atol=5e-3)

    spectrum = np.sort(np.linalg.eigvals(
        bp.matrix(action.carrier_operator())).real)
    np.testing.assert_allclose(spectrum[3:6], 2.8977, atol=5e-4)
    np.testing.assert_allclose(spectrum[6:9], 3.9248, atol=5e-4)


# ------------------------------------------------------ refusals by name


def test_a_held_holonomy_driven_across_minus_one_is_named():
    """(0123, 300) with no fiber moment pinned: the Newton direction drives a
    held face holonomy across -1, where the monopole number read from the
    principal arguments jumps. The sector guard cuts most of the Newton
    steps, the accepted steps end ever nearer the jump (the accepted
    fraction of the step falls from 1/2 to parts in 1e13 while the residual
    norm settles at 1.772), and the solve ends where
    the trial steps that stay in the sector no longer reduce the residual:
    "no damped step reduced the residual", its detail saying how far along
    the Newton direction the sector guard refused. It ends by itself,
    without freezing or repeating; the geometry it stopped at is
    Kontsevich-Segal allowable, so the read proceeds and its record says why
    the solve stopped."""
    _, _, report, config = _relax(FIRST_CELL, (3, 0, 0), fiber_moments=0)
    assert not report.converged
    assert report.stop_reason == cob.RelaxationStop.NoDescent
    assert report.stop_detail.startswith(
        "no damped step reduced the residual norm: the smallest trial step, "
        "2^-")
    assert "; the trial steps down to 2^-" in report.stop_detail
    assert "changed a held monopole number" in report.stop_detail
    last = report.steps[-1]
    assert last.newton_iterated and not last.newton.accepted
    assert last.newton.sector_guard_dampings >= 40
    assert last.geometry_stop_reason == cob.RelaxationStop.NoDescent
    assert report.force_norm == pytest.approx(1.7719, abs=5e-4)
    # the sector guard cut most of the accepted steps, down to parts in 1e10
    accepted = [step.newton for step in report.steps[:-1]]
    assert sum(step.sector_guard_dampings > 0 for step in accepted) >= 20
    assert min(step.damping for step in accepted) < 1e-10
    # no iterate repeats another: every accepted step moved the geometry
    assert all(step.step_norm > 0.0 for step in accepted)
    assert report.kontsevich_segal_margin == pytest.approx(np.pi, abs=1e-5)
    assert _refusal(report, config) is None
    record = bp.relaxation_record(report)
    assert record["stop_reason"] == "no damped step reduced the residual"
    assert "method" not in record


def test_lengths_that_grow_without_bound_are_left_to_the_equations():
    """(0123, 201) with no fiber moment pinned: the length equations are the
    matter force alone, which is 1/z^2 at large z and vanishes only at
    infinite length, so the lengths grow without bound and the occupied
    eigenvalues go to zero with h_1 ~ 1/z. Nothing but the datatype's bound
    stops such a solve: it runs until the residual is at its floor on the
    held set, with the largest |z| about a thousand times the host's and
    only the link block of the joint Jacobian left, and the read is refused
    on the geometry itself, which is not Kontsevich-Segal allowable. The
    overflow stop is reserved for a squared length beyond the largest finite
    double, and the declaration carries no ratio to tune."""
    _, _, report, config = _relax(FIRST_CELL, (2, 0, 1), fiber_moments=0)
    assert not report.converged
    assert report.stop_reason == cob.RelaxationStop.HeldFloor
    assert report.largest_length_ratio > 5e2
    assert report.jacobian_rank == 3
    assert max(abs(v) for v in report.occupied_eigenvalues) < 0.1
    assert report.kontsevich_segal_margin < 0
    name, message = _refusal(report, config)
    assert name == "not Kontsevich-Segal allowable"
    assert "margin %.3g" % report.kontsevich_segal_margin in message
    assert cob.relaxation_stop_name(cob.RelaxationStop.LengthRunaway) == \
        "the squared lengths overflowed the double"
    assert not hasattr(cob.HolomorphicRelaxationDeclaration(),
                       "length_runaway_ratio")


def test_a_read_that_leaves_the_allowable_domain_is_refused():
    """(0123, 021): the joint Newton drives one shared squared length
    negative and, through a stretch of steps cut to as little as parts in a
    million of their length, goes on to a fixed point off the real slice,
    z = (53.216 + 11.000i, 15.030 + 7.148i, 19.331 + 3.846i,
    14.530 + 7.683i, 18.996 + 3.932i, -5.301 - 0.277i), with both occupied
    bands at their pinned eigenvalues 0.708 and 2.898. The solve converges
    there, at a geometry that is not Kontsevich-Segal allowable (margin
    -0.535), and the read is refused with the margin, so no pole is read on
    it."""
    spacetime, _, report, config = _relax(FIRST_CELL, (0, 2, 1))
    assert report.converged
    assert report.stop_reason == cob.RelaxationStop.Converged
    z = np.asarray(bp.sheet_squared_lengths(spacetime, 0))
    np.testing.assert_allclose(
        z, [53.2161 + 10.9998j, 15.0299 + 7.1477j, 19.3312 + 3.8456j,
            14.5295 + 7.6834j, 18.9964 + 3.9316j, -5.3005 - 0.2774j],
        atol=5e-3)
    assert sorted(band.eigenvalues[0].real for band in report.bands) == \
        pytest.approx([0.7080, 2.8977], abs=5e-4)
    assert report.kontsevich_segal_margin == pytest.approx(-0.5352, abs=5e-4)
    name, message = _refusal(report, config)
    assert name == "not Kontsevich-Segal allowable"
    assert "margin %.3g" % report.kontsevich_segal_margin in message


@pytest.mark.slow
def test_a_read_that_stops_short_moves_at_every_iterate_and_says_why():
    """(0123, 300) with the pinned fiber: the joint Newton moves at every
    iteration it takes (the covariance changes at every one) and stops with
    a named reason, no damped step reducing the residual, instead of
    freezing and repeating; the geometry it reaches is Kontsevich-Segal
    allowable, so the read proceeds, and the record and its text say why the
    solve stopped."""
    _, _, report, config = _relax(FIRST_CELL, (3, 0, 0))
    assert not report.converged
    assert report.stop_reason == cob.RelaxationStop.NoDescent
    assert report.stop_detail
    assert report.iterations >= 10
    changes = [step.covariance_change for step in report.steps[1:]]
    assert min(changes) > 0.0
    assert report.kontsevich_segal_margin == pytest.approx(np.pi, abs=1e-6)
    assert _refusal(report, config) is None
    record = bp.relaxation_record(report)
    assert record["stop_reason"] == "no damped step reduced the residual"
    text = bp.relaxation_text(record)
    assert "; stopped: no damped step reduced the residual (" in text
    assert "method" not in text


# ------------------------------------------- limits the user declares


def test_no_limit_is_declared_unless_the_user_declares_one():
    """A solve carries no number of steps, no number of halvings and no
    time after which it stops: the three limits of the declaration are None,
    and the name of the stop a declared one gives exists for that case
    alone."""
    declaration = cob.HolomorphicRelaxationDeclaration()
    assert declaration.iteration_limit is None
    assert declaration.halving_limit is None
    assert declaration.time_limit_seconds is None
    assert cob.relaxation_stop_name(cob.RelaxationStop.DeclaredLimit) == \
        "a declared limit was reached"


def test_a_declared_number_of_steps_ends_the_solve_by_name():
    """(0123, 003), which converges in six steps, with two steps declared:
    the solve takes the two and stops, and says that the limit was the
    user's."""
    _, _, report, _ = _relax(FIRST_CELL, (0, 0, 3),
                             limits={"iteration_limit": 2})
    assert not report.converged
    assert report.stop_reason == cob.RelaxationStop.DeclaredLimit
    assert report.iterations == 2
    assert report.stop_detail.startswith(
        "the declared limit of 2 accepted steps was reached; the residual "
        "norm is ")
    assert bp.relaxation_record(report)["stop_reason"] == \
        "a declared limit was reached"


def test_a_declared_number_of_halvings_ends_the_solve_by_name():
    """(0123, 300) with no fiber moment pinned and no halving allowed: the
    first Newton step that is not accepted at full length ends the solve,
    and the detail says what refused that step."""
    _, _, report, _ = _relax(FIRST_CELL, (3, 0, 0), fiber_moments=0,
                             limits={"halving_limit": 0})
    assert not report.converged
    assert report.stop_reason == cob.RelaxationStop.DeclaredLimit
    assert report.stop_detail.startswith(
        "the declared limit of 0 halvings of a Newton step was reached: the "
        "smallest trial step, 2^-0 of the Newton step")
    last = report.steps[-1].newton
    assert not last.accepted
    assert (last.residual_test_dampings + last.sector_guard_dampings
            + last.domain_guard_dampings) == 1


def test_a_declared_time_ends_the_solve_by_name():
    """A time limit of zero seconds is reached before the first step."""
    _, _, report, _ = _relax(FIRST_CELL, (0, 0, 3),
                             limits={"time_limit_seconds": 0.0})
    assert not report.converged
    assert report.stop_reason == cob.RelaxationStop.DeclaredLimit
    assert report.iterations == 0
    assert report.stop_detail.startswith(
        "the declared time limit of 0 seconds was reached after 0 accepted "
        "steps")


# ------------------------------------------------ the constrained step


def _held_host_action(content):
    """The seeded action of (0123, content) at the host, the covariance fixed
    at the band filling there, and the relaxation declaration of a geometry
    relaxation at that covariance (the cell's faces held)."""
    config = _config(FIRST_CELL, content)
    spacetime = bp.build_host(config["edge_squared"], config["host_cell"])
    action = cob.JointAction(spacetime, bp.action_declaration(spacetime, 1.0,
                                                              1.0))
    mean = bp.mean_field_declaration(content, config, spacetime)
    read = cob.BandFollower(mean).read(action.carrier_operator())
    action.set_covariance(read.covariance)
    return spacetime, action, mean.geometry, config


def _held_modulus_projector(geometry, spacetime):
    """The projector onto the link-modulus directions the held faces fix, in
    the twelve shared coordinates' link block: the row space of the held
    faces' coboundary pulled back through the classes."""
    edges = bp.edge_records(spacetime)
    lookup = {}
    for index, (a, b) in enumerate(edges):
        lookup[(a, b)] = (index, 1)
        lookup[(b, a)] = (index, -1)
    rows = []
    for sector in geometry.held_sectors:
        for face in sector.faces:
            row = np.zeros(len(edges))
            for k in range(3):
                index, sign = lookup[(int(face[k]), int(face[(k + 1) % 3]))]
                row[index] += sign
            rows.append(row)
    classes, orientations = bp.sheet_edge_classes(spacetime)
    expansion = np.zeros((len(edges), 6))
    for edge, (c, o) in enumerate(zip(classes, orientations)):
        expansion[edge, c] = o
    _, s, vt = np.linalg.svd(np.array(rows) @ expansion)
    basis = vt[:int(np.sum(s > 1e-10 * s[0]))].T
    return basis @ basis.T


def _constrained_newton_step(jacobian, residual, held, rank):
    """The least-squares step over the tangent space of the held set, formed
    as the solve forms it: the twelve complex linearized equations as a real
    system over the real and imaginary length directions, the imaginary
    link directions and an orthonormal basis of the link moduli the held
    faces leave free, cut at the rank the solve recorded."""
    n = len(residual)
    columns = []
    for index in range(6):
        for value in (1.0, 1.0j):
            column = np.zeros(n, dtype=complex)
            column[index] = value
            columns.append(column)
    for index in range(6):
        column = np.zeros(n, dtype=complex)
        column[6 + index] = 1.0j
        columns.append(column)
    free, moduli, _ = np.linalg.svd(np.eye(6) - held)
    for direction in free[:, moduli > 0.5].T:
        column = np.zeros(n, dtype=complex)
        column[6:] = direction
        columns.append(column)
    parametrization = np.array(columns).T
    image = jacobian @ parametrization
    system = np.vstack([image.real, image.imag])
    right = -np.concatenate([residual.real, residual.imag])
    u, s, vh = np.linalg.svd(system, full_matrices=False)
    unknown = vh[:rank].T @ ((u[:, :rank].T @ right) / s[:rank])
    return parametrization @ unknown


def test_the_constrained_step_leaves_far_less_of_the_linear_residual():
    """(0123, 021) at the host, the covariance fixed. The projection of the
    unconstrained minimum-norm Newton step onto the held directions leaves
    28 % of the linearized residual; the constrained Newton step, the
    least-squares step over the tangent space of the held set, leaves
    2.4 %. It is a descent direction for the residual norm, the damped step
    it takes reduces the residual, and the solve keeps every held modulus
    exactly."""
    spacetime, action, geometry, _ = _held_host_action((0, 2, 1))
    relaxation = cob.HolomorphicRelaxation(action, geometry)
    jacobian = np.asarray(relaxation.jacobian()).reshape(12, 12)
    residual = np.asarray(relaxation.residual())
    held = _held_modulus_projector(geometry, spacetime)
    # the projected step, the unconstrained step made to respect the held
    # moduli after the fact
    u, s, vh = np.linalg.svd(jacobian)
    rank = int(np.sum(s > geometry.rank_tolerance * s[0]))
    unconstrained = vh[:rank].conj().T @ ((u[:, :rank].conj().T
                                           @ -residual) / s[:rank])
    projected = unconstrained.copy()
    projected[6:] -= held @ projected[6:].real
    projected_left = np.linalg.norm(residual + jacobian @ projected) / \
        np.linalg.norm(residual)
    assert projected_left == pytest.approx(0.285, abs=0.01)

    report = relaxation.solve()
    step = report.steps[0]
    assert step.constrained_step and step.accepted
    assert step.linear_residual == pytest.approx(0.0235, abs=5e-4)
    assert step.linear_residual < projected_left / 5
    assert step.constrained_rank_gap > 1e5
    # the residual at the point the first damped step reached: the next
    # iterate's, or the end point's when the solve stopped there
    after = report.steps[1].residual_norm if len(report.steps) > 1 \
        else report.residual_norm
    assert after < step.residual_norm
    assert report.held_modulus_drift < 1e-14
    # the step the solve took, formed from the same Jacobian and residual
    # over the tangent space of the held set at the rank the solve decided
    taken = _constrained_newton_step(jacobian, residual, held,
                                     step.constrained_rank)
    assert np.max(np.abs(held @ taken[6:].real)) < 1e-12
    left = np.linalg.norm(residual + jacobian @ taken) / \
        np.linalg.norm(residual)
    assert left == pytest.approx(step.linear_residual, rel=1e-6)
    descent = np.vdot(residual, jacobian @ taken).real
    assert descent < 0.0


def test_holding_the_moduli_costs_the_joint_newton_nothing_on_the_real_slice():
    """On the real slice the joint system is real-structured, so its Newton
    step changes no modulus and the constrained step solves the linearized
    equations to rounding, as the unconstrained step does."""
    _, _, report, _ = _relax(FIRST_CELL, (2, 0, 1))
    first = report.steps[0].newton
    assert first.constrained_step
    assert first.linear_residual < 1e-10
