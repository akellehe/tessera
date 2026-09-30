# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The self-consistent mean-field solve of the recursion driver's per-cell
reads, held to the reproducers of the mean-field convergence investigation
(#1257; `reports/design/mean_field_convergence_investigation.md` in the notes
repository).

Terms used below:

* a *read* is one (host cell, content) pair of the tick-0 run of 2026-09-25,
  whose host cells are those of `_recursion_run_2026_09_23` (the level-0 host
  is stationary as built, so both runs read the same cells);
* a *content* (n_0, n_1, n_2) places n_b quarks in band b of the covariant
  operator h_1(z, U), the bands counted in ascending order of real part at
  the host;
* the *self-consistent force* F_sc(z, U) is the stationarity force of the
  joint action on the six shared squared lengths z and six links U with the
  covariance Gamma rebuilt by the band rule at (z, U); its zeros are the
  stationary pairs of WP v17 lines 259 and 263;
* the *held set* is the declared monopole number and the unit moduli of the
  face holonomies on the cell's four faces (WP v17 line 508).

The investigation's reproducers are (0123, 021), the best-converging read;
(0123, 300), which froze with a held face holonomy at argument pi; and
(0134, 111), whose lengths grew without bound. What is asserted:

* Newton's method on the joint system converges (0123, 021) to 1e-9 in a
  handful of steps, and the point it reaches satisfies the equations of the
  joint action evaluated independently;
* the occupied bands are chosen at the host and followed by continuation,
  every iterate's overlaps and crossings are reported, and re-sorting stays
  available by name; the two rules reach the two fixed points the
  investigation found;
* a read without a stationary point is refused by name (no stationary point
  in the declared monopole sector, not Kontsevich-Segal allowable; a squared
  length beyond the largest finite double is the only bound on the lengths)
  instead of freezing or reading poles on a collapsed geometry;
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
SECOND_CELL = (0, 1, 3, 4)


def _config(cell, content, method="joint-newton",
            selection="continuation"):
    """The configuration `recursion.cell_reads` handed `baryon_poles` for one
    tick-0 host cell in the run the investigation studied (kappa = beta = 1,
    the Villain term, the cell's four faces held, the linear stiffness
    stand-in, no fiber moment pinned), with the declared solve method and
    band selection."""
    config = bp.default_config(kappas=[1.0], betas=[1.0],
                               selected_contents=[tuple(content)],
                               stiffness="linear-stand-in", fiber_moments=0,
                               tolerances=RUN.TOLERANCES)
    config["host_cell"] = RUN.HOST_CELLS[cell]
    config["held_sectors"] = R.held_sectors([[0, 1, 2, 3]], [1], 4)
    config["mean_field_method"] = method
    config["band_selection"] = selection
    return config


def _relax(cell, content, **options):
    config = _config(cell, content, **options)
    spacetime, action, report = bp.relax_content(content, 1.0, 1.0, config)
    return spacetime, action, report, config


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


def test_the_joint_newton_converges_the_best_read_in_a_handful_of_steps():
    """(0123, 021): the alternation contracted by 0.853 per outer iteration
    and reached a force of 8e-4 in 40; Newton's method on the joint system
    reaches the fixed point in a handful of steps, on the real slice, with
    the joint Jacobian of rank 9 of 12 (the three gauge directions are its
    null space) and a separated rank decision."""
    spacetime, action, report, _ = _relax(FIRST_CELL, (0, 2, 1))
    assert report.method == cob.SelfConsistentMethod.JointNewton
    assert report.band_selection == cob.BandSelection.Continuation
    assert report.converged
    assert report.stop_reason == cob.RelaxationStop.Converged
    assert cob.relaxation_stop_name(report.stop_reason) == "converged"
    assert report.force_norm <= 1e-9
    assert 3 <= report.iterations <= 8
    assert len(report.steps) == report.iterations + 1
    # one Newton iteration from every iterate but the last, each accepted
    for step in report.steps[:-1]:
        assert step.newton_iterated and step.newton.accepted
        assert step.newton.step_norm > 0.0
        assert step.geometry_stop_reason == cob.RelaxationStop.Continued
    assert not report.steps[-1].newton_iterated
    assert report.steps[-1].geometry_stop_reason == \
        cob.RelaxationStop.Converged
    assert (report.jacobian_size, report.jacobian_rank) == (12, 9)
    assert report.rank_gap > 1e6
    # a Euclidean cell: every eigenvalue of the metric positive
    assert report.kontsevich_segal_margin == pytest.approx(np.pi, abs=1e-9)
    z = np.asarray(bp.sheet_squared_lengths(spacetime, 0))
    assert np.max(np.abs(z.imag)) < 1e-9 and np.min(z.real) > 0
    faces = np.asarray(action.face_holonomies())
    assert np.max(np.abs(np.abs(faces) - 1.0)) < 1e-9
    assert report.action_available and report.action_unavailable == ""


def test_the_fixed_point_satisfies_the_unchanged_equations():
    """The joint Newton changes no equation: at the point it reached, the
    covariance it reports is the band filling of h_1 there (it commutes with
    h_1 and carries the three quarks), and a joint action built afresh with
    that covariance has a force below the declared tolerance on the twelve
    shared coordinates."""
    spacetime, _, report, config = _relax(FIRST_CELL, (0, 2, 1))
    gamma = np.asarray(report.covariance).reshape(18, 18)
    declaration = bp.action_declaration(spacetime, 1.0, 1.0,
                                        reference_lengths=config[
                                            "reference_lengths"],
                                        stiffness=config["stiffness"])
    declaration.covariance = list(report.covariance)
    fresh = cob.JointAction(spacetime, declaration)
    h = bp.matrix(fresh.carrier_operator())
    assert np.linalg.norm(gamma @ h - h @ gamma) < 1e-9 * np.linalg.norm(h)
    assert np.trace(gamma).real == pytest.approx(3.0, abs=1e-10)
    assert np.linalg.norm(_class_force(fresh, spacetime)) <= 1e-9


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


def test_continuation_and_re_sorting_reach_different_fixed_points():
    """(0123, 021) under the two named rules, both solved by the joint
    Newton. Followed from the host, the declared bands 1 and 2 cross below
    the host's negative band, which leaves the bottom of the spectrum after
    the first step; the solve reports the crossing and converges to the
    fixed point with the occupied bands at 0.466 and 2.669,
    z = (7.258, 9.271, 9.624, 9.524, 9.377, 7.479). Re-sorted at every
    iterate, the occupations exchange onto other modes at the first step
    (overlap below 0.1) and the solve converges to the investigation's other
    fixed point, z = (8.800, 11.394, 11.337, 11.503, 11.234, 6.355), with
    the occupied bands at 1.894 and 6.649."""
    spacetime, _, followed, _ = _relax(FIRST_CELL, (0, 2, 1))
    assert followed.converged and followed.band_crossing_iterates >= 1
    bands = followed.bands
    assert [b.declared_index for b in bands] == [1, 2]
    assert [list(b.declared_positions) for b in bands] == [[3, 4, 5],
                                                            [6, 7, 8]]
    assert [list(b.positions) for b in bands] == [[0, 1, 2], [3, 4, 5]]
    assert all(b.crossed for b in bands)
    np.testing.assert_allclose([b.eigenvalues[0].real for b in bands],
                               [0.4662, 2.6691], atol=5e-4)
    np.testing.assert_allclose(
        np.asarray(bp.sheet_squared_lengths(spacetime, 0)).real,
        [7.258, 9.271, 9.624, 9.524, 9.377, 7.479], atol=5e-3)

    spacetime, _, sorted_, _ = _relax(FIRST_CELL, (0, 2, 1),
                                      selection="sort-every-iterate")
    assert sorted_.band_selection == cob.BandSelection.SortEveryIterate
    assert sorted_.converged and sorted_.band_crossing_iterates == 0
    assert sorted_.lowest_band_overlap < 0.1
    np.testing.assert_allclose([b.eigenvalues[0].real
                                for b in sorted_.bands],
                               [1.8941, 6.6494], atol=5e-4)
    np.testing.assert_allclose(
        np.asarray(bp.sheet_squared_lengths(spacetime, 0)).real,
        [8.800, 11.394, 11.337, 11.503, 11.234, 6.355], atol=5e-3)


# ------------------------------------------------------ refusals by name


def test_a_held_holonomy_driven_across_minus_one_is_named():
    """(0123, 120): the Newton direction drives a held face holonomy across
    -1, where the monopole number read from the principal arguments jumps,
    so even the smallest damped step changes a held monopole number. The
    solve stops at once and reports "no stationary point in the declared
    monopole sector", instead of freezing and repeating; the held sectors
    keep their declared numbers; the geometry it stopped at is not
    Kontsevich-Segal allowable, so its poles are not read."""
    _, _, report, _ = _relax(FIRST_CELL, (1, 2, 0))
    assert not report.converged
    assert report.stop_reason == cob.RelaxationStop.SectorBoundary
    assert cob.relaxation_stop_name(report.stop_reason) == \
        "no stationary point in the declared monopole sector"
    assert report.stop_detail.startswith(
        "no stationary point in the declared monopole sector: the smallest "
        "trial step, 2^-16 of the Newton step changed a held monopole number")
    last = report.steps[-1]
    assert last.newton_iterated and not last.newton.accepted
    assert last.newton.sector_guard_dampings >= 1
    assert last.geometry_stop_reason == cob.RelaxationStop.SectorBoundary
    # no iterate repeats another: every accepted step moved the geometry
    assert all(step.newton.step_norm > 0.0 for step in report.steps[:-1])
    name, message = bp.read_refusal(report)
    assert name == "not Kontsevich-Segal allowable"
    assert "margin %.3g" % report.kontsevich_segal_margin in message
    assert report.kontsevich_segal_margin < 0.0


def test_lengths_that_grow_without_bound_are_left_to_the_equations():
    """(0123, 201): the joint Newton follows the contracting force of the
    occupied negative band and the lengths grow without bound. The stand-in
    stiffness' force saturates at 1/(2 kappa) per edge in z, so the residual
    has a plateau at infinite length. Nothing but the datatype's bound stops
    such a solve: it runs until no damped step lowers the residual, with the
    largest |z| four decades beyond the host's, and the read is refused on
    the geometry itself, which is not Kontsevich-Segal allowable. The
    overflow stop is reserved for a squared length beyond the largest finite
    double, and the declaration carries no ratio to tune."""
    _, _, report, _ = _relax(FIRST_CELL, (2, 0, 1))
    assert report.stop_reason == cob.RelaxationStop.NoDescent
    assert report.largest_length_ratio > 1e3
    assert report.kontsevich_segal_margin < 0
    name, message = bp.read_refusal(report)
    assert name == "not Kontsevich-Segal allowable"
    assert cob.relaxation_stop_name(cob.RelaxationStop.LengthRunaway) == \
        "the squared lengths overflowed the double"
    assert not hasattr(cob.HolomorphicRelaxationDeclaration(),
                       "length_runaway_ratio")


def test_the_run_off_read_of_0134_111_is_refused_rather_than_collapsed():
    """(0134, 111), whose lengths ran to |z| of 1e6 under the alternation and
    whose low eigenvalues collapsed to 1e-5 there: the joint Newton stops by
    name at a geometry that is not Kontsevich-Segal allowable, and the read is
    refused with the margin, so no pole is read on it."""
    _, _, report, _ = _relax(SECOND_CELL, (1, 1, 1))
    assert not report.converged
    assert report.stop_reason != cob.RelaxationStop.Converged
    assert report.stop_detail
    name, message = bp.read_refusal(report)
    assert name == "not Kontsevich-Segal allowable"
    assert "margin %.3g" % report.kontsevich_segal_margin in message


@pytest.mark.slow
def test_the_frozen_read_of_0123_300_is_refused_by_name():
    """(0123, 300), which froze under the alternation with a held face
    holonomy at argument pi and a covariance change of exactly 0 for 39
    iterations: the joint Newton moves at every iteration it takes and stops
    with a named reason, and the geometry it reaches is not Kontsevich-Segal
    allowable, so the read is refused with the margin."""
    _, _, report, _ = _relax(FIRST_CELL, (3, 0, 0))
    assert not report.converged and report.stop_detail
    changes = [step.covariance_change for step in report.steps[1:]]
    assert min(changes) > 0.0
    name, message = bp.read_refusal(report)
    assert name == "not Kontsevich-Segal allowable"
    assert "margin %.3g" % report.kontsevich_segal_margin in message


@pytest.mark.slow
def test_the_fallback_alternation_refuses_by_name_too():
    """The alternation, kept as the named fallback, no longer freezes on
    (0123, 300): its lengths grow without bound until its third outer
    iteration leaves the declared monopole sector, the solve says so, and
    the read is refused on the geometry, which is not Kontsevich-Segal
    allowable."""
    _, _, report, _ = _relax(FIRST_CELL, (3, 0, 0), method="alternation")
    assert report.method == cob.SelfConsistentMethod.Alternation
    assert report.stop_reason == cob.RelaxationStop.SectorBoundary
    assert report.iterations == 3
    assert report.largest_length_ratio > 100
    assert bp.read_refusal(report)[0] == "not Kontsevich-Segal allowable"


def test_an_outer_iteration_that_cannot_move_is_named_and_not_repeated():
    """Under the alternation an outer iteration whose inner solve accepts no
    step and whose re-occupation leaves the covariance unchanged would only
    repeat itself. The solve stops there and names both reasons, the inner
    one included, instead of running out its outer budget; the inner solve
    here is given no iterations, the plainest such case. The host is three
    sheets of the regular tetrahedron of squared length 8 with a flat
    connection, whose bands are 5 and 10, rank nine each."""
    edges = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    spacetime, _ = R.build_level([[0, 1, 2, 3]], {e: 8.0 for e in edges},
                                 {e: 1.0 + 0j for e in edges})
    action = cob.JointAction(spacetime, bp.action_declaration(spacetime, 1.0,
                                                              1.0))
    config = bp.default_config([1.0], [1.0], tolerances=RUN.TOLERANCES)
    config["newton_iterations"] = 0
    config["mean_field_method"] = "alternation"
    declaration = bp.mean_field_declaration((2, 1), config, spacetime)
    declaration.maximum_iterations = 40
    report = cob.SelfConsistentMeanField(action, declaration).solve()
    assert report.stop_reason == cob.RelaxationStop.NoProgress
    assert report.iterations == 1
    assert report.stop_detail.startswith(
        "outer iteration 1 made no progress: its inner solve accepted no step "
        "(the declared iterations ran out: the declared 0 iterations ran out")
    assert report.steps[-1].covariance_change == 0.0


# ------------------------------------------------ the constrained step


def _held_host_action(content, geometry_iterations=1):
    """The seeded action of (0123, content) at the host, the covariance fixed
    at the band filling there, and the relaxation declaration of the
    alternation's inner solve (the cell's faces held)."""
    config = _config(FIRST_CELL, content)
    spacetime = bp.build_host(config["edge_squared"], config["host_cell"])
    declaration = bp.action_declaration(spacetime, 1.0, 1.0,
                                        stiffness=config["stiffness"])
    action = cob.JointAction(spacetime, declaration)
    mean = bp.mean_field_declaration(content, config, spacetime)
    read = cob.BandFollower(mean).read(action.carrier_operator())
    action.set_covariance(read.covariance)
    geometry = mean.geometry
    geometry.maximum_iterations = geometry_iterations
    return spacetime, action, geometry, config


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


def test_the_constrained_step_leaves_far_less_of_the_linear_residual():
    """(0123, 021) at the host, the covariance fixed (the alternation's first
    inner step). The projection of the unconstrained minimum-norm Newton step
    onto the held directions leaves 39 % of the linearized residual; the
    constrained Newton step, the least-squares step over the tangent space of
    the held set, leaves 5.1 %. It is a descent direction for the residual
    norm, the damped step it takes reduces the residual, and it keeps every
    held modulus exactly."""
    spacetime, action, geometry, _ = _held_host_action((0, 2, 1))
    relaxation = cob.HolomorphicRelaxation(action, geometry)
    jacobian = np.asarray(relaxation.jacobian()).reshape(12, 12)
    residual = np.asarray(relaxation.residual())
    # the projected step, as the solve took it before
    u, s, vh = np.linalg.svd(jacobian)
    rank = int(np.sum(s > geometry.rank_tolerance * s[0]))
    unconstrained = vh[:rank].conj().T @ ((u[:, :rank].conj().T
                                           @ -residual) / s[:rank])
    projected = unconstrained.copy()
    projected[6:] -= _held_modulus_projector(geometry, spacetime) @ \
        projected[6:].real
    projected_left = np.linalg.norm(residual + jacobian @ projected) / \
        np.linalg.norm(residual)
    assert projected_left == pytest.approx(0.39, abs=0.01)

    z_before = np.asarray(bp.sheet_squared_lengths(spacetime, 0))
    links_before = np.asarray(bp.sheet_links(spacetime, 0))
    report = relaxation.solve()
    (step,) = report.steps
    assert step.constrained_step and step.accepted
    assert step.linear_residual == pytest.approx(0.0513, abs=5e-4)
    assert step.linear_residual < projected_left / 5
    assert step.constrained_rank_gap > 1e5
    assert report.residual_norm < step.residual_norm
    assert report.held_modulus_drift < 1e-14
    # the step the solve took, recovered from the geometry it wrote
    z_after = np.asarray(bp.sheet_squared_lengths(spacetime, 0))
    links_after = np.asarray(bp.sheet_links(spacetime, 0))
    taken = np.concatenate([z_after - z_before,
                            np.log(links_after / links_before)]) / step.damping
    assert np.max(np.abs(taken[6:].real
                         - (np.eye(6) - _held_modulus_projector(
                             geometry, spacetime)) @ taken[6:].real)) < 1e-12
    left = np.linalg.norm(residual + jacobian @ taken) / \
        np.linalg.norm(residual)
    assert left == pytest.approx(step.linear_residual, rel=1e-6)
    descent = np.vdot(residual, jacobian @ taken).real
    assert descent < 0.0


def test_holding_the_moduli_costs_the_joint_newton_nothing_on_the_real_slice():
    """On the real slice the joint system is real-structured, so its Newton
    step changes no modulus and the constrained step solves the linearized
    equations to rounding, as the unconstrained step does."""
    _, _, report, _ = _relax(FIRST_CELL, (0, 2, 1))
    first = report.steps[0].newton
    assert first.constrained_step
    assert first.linear_residual < 1e-10
