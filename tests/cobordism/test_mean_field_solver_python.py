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
  its value at the host, one multiplier per band among the system's
  unknowns. A cell has no interior hinge, so with no fiber moment pinned the
  length equations are the matter force alone, homogeneous of degree -2 in
  z, which vanishes only at infinite length;
* the *self-consistent force* F_sc(z, U) is the stationarity force of the
  joint action on the six shared squared lengths z and six links U with the
  covariance Gamma rebuilt by the band rule at (z, U); its zeros, at which
  the pinned fiber's residuals vanish too, are the stationary pairs of WP
  v17 lines 259 and 263;
* the *held set* is the declared monopole number and the unit moduli of the
  face holonomies on the cell's four faces (WP v17 line 508);
* the *drive* is the solve of a read: `MultiCobordism` on the base
  tetrahedron with the residual norm of the three-sheeted joint system as
  its objective (`baryon_poles.relax_content`, `cell_solve.solve`), its
  Pachner moves included, at the declared tolerances (1e-15). An *accepted
  update* is a trial of its line search that lowered the residual norm by
  the tolerance; a *step proposal* is one point at which a step was formed.

What is asserted:

* the drive takes (0123, 201) from a residual norm of 0.92 to 2.0e-14 in 21
  accepted updates, on the real slice, where no trial lowers the norm by the
  tolerance, and the point it reaches satisfies the equations of the joint
  action evaluated independently; at the declared tolerance that point is
  reported not converged;
* the occupied bands are chosen at the host and followed by continuation,
  every step proposal's overlaps and crossings are reported, and re-sorting
  stays available by name; on (0123, 003) continuation follows the occupied
  band through a crossing to a point that is no fixed point of the re-sorted
  equations;
* a read that reaches no fixed point ends by itself and by name, and its
  record says what the drive scored last; the engine scores every trial
  whatever its Kontsevich-Segal margin (its admissibility gate is off unless
  declared), so a read whose step leaves the allowable domain is made
  outside it and flagged with its margin; a squared length that is not
  finite, and a cell a committed move changed, are the only geometries with
  no value to read;
* no limit ends a drive unless the user declares one, and a declared one
  ends it by name;
* the step over the held set is the constrained Newton step, which leaves
  less of the linearized residual, in the scaled equations it solves, than
  the projection of the unconstrained step (half as much on the read
  below), and keeps every held modulus.
"""
import numpy as np
import pytest

from tessera import cobordism as cob
from tessera.drivers import baryon_poles as bp
from tessera.drivers import cell_solve as cs
from tessera.drivers import recursion as R

from tests.drivers import _recursion_run_2026_09_23 as RUN

FIRST_CELL = (0, 1, 2, 3)


def _config(cell, content, selection="continuation",
            fiber_moments=bp.DECLARED_FIBER_MOMENTS, limits=None):
    """The configuration `recursion.cell_reads` hands `baryon_poles` for one
    tick-0 host cell (kappa = beta = 1, the Villain term, the cell's four
    faces held, the eigenvalue of every occupied band pinned at the host),
    at the declared tolerances, with the declared band selection;
    ``fiber_moments=0`` pins nothing, and ``limits`` declares any of
    `baryon_poles.LIMITS` (none by default)."""
    config = bp.default_config(kappas=[1.0], betas=[1.0],
                               selected_contents=[tuple(content)],
                               fiber_moments=fiber_moments, limits=limits)
    config["host_cell"] = RUN.HOST_CELLS[cell]
    config["held_sectors"] = R.held_sectors([[0, 1, 2, 3]], [1], 4)
    config["band_selection"] = selection
    return config


def _relax(cell, content, **options):
    """The drive of one read: the three-sheeted complex it ended on, the
    action and the report there, the drive's record and the configuration."""
    config = _config(cell, content, **options)
    spacetime, action, report, drive = bp.relax_content(content, 1.0, 1.0,
                                                        config)
    return spacetime, action, report, drive, config


def _flags(report, config):
    """The flags of the geometry the solve reached, at the declared
    allowability tolerance, as the driver reads them."""
    return bp.geometry_flags(report, bp.declared_tolerance(
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


# ------------------------------------------------------------- the drive


def test_the_drive_takes_a_read_to_the_rounding_of_its_equations():
    """(0123, 201): the drive lowers the residual norm of the joint system
    from 0.916 at the host to 2.0e-14 in 21 accepted updates, every one a
    decrease, committing no Pachner move, on the real slice. It ends where
    no move and no trial of the line search lowers the norm by the tolerance
    1e-15, and the end point, whose force 2.0e-14 is above that tolerance,
    is reported not converged. The joint Jacobian there has rank 11 of 14
    (twelve shared coordinates and the two multipliers of the pinned fiber;
    the three gauge directions are its null space) with a separated rank
    decision, and the pinned eigenvalues hold to rounding.

    The norm at the host is that of the multipliers' least-squares fit
    taken on the length equations scaled by their coordinates' scales
    (`cell_solve.least_squares_multipliers`); the fit of the unscaled
    equations leaves 0.391 there."""
    spacetime, action, report, drive, _ = _relax(FIRST_CELL, (2, 0, 1))
    assert report.band_selection == cob.BandSelection.Continuation
    assert drive["stop_reason"] == cs.STOP_STATIONARY == \
        "no move and no scaled step lowers the residual norm"
    assert drive["moves_committed"] == 0 and not drive["changed"]
    trace = drive["trace"]
    assert 15 <= drive["accepted_updates"] <= 30
    assert len(trace) == drive["accepted_updates"] + 1
    assert trace[0] == pytest.approx(0.91632, rel=1e-4)
    assert all(after < before for before, after in zip(trace, trace[1:]))
    assert trace[-1] < 1e-12
    # the fixed-point conditions at the declared tolerance 1e-15
    assert 1e-15 < report.force_norm < 1e-12
    assert not report.converged
    record = bp.relaxation_record(report, drive)
    assert record["stop_reason"] == cs.STOP_STATIONARY
    assert record["accepted_updates"] == drive["accepted_updates"]
    assert record["method"] == \
        "MultiCobordism drive of the joint action's stationarity"
    # one step proposal per accepted update, and those that ended the drive
    proposals = drive["objective"].updates
    assert len(proposals) > drive["accepted_updates"]
    assert all(update["step_norm"] > 0.0 for update in proposals)
    assert all(update["jacobian_rank"] == 11 for update in proposals)
    assert (report.jacobian_size, report.jacobian_rank) == (14, 11)
    assert report.rank_gap > 1e6
    assert report.fiber_rank == 6 and len(report.multipliers) == 2
    assert np.max(np.abs(report.moment_residuals)) <= 1e-12
    # a Euclidean cell: every eigenvalue of the metric positive
    assert report.kontsevich_segal_margin == pytest.approx(np.pi, abs=1e-9)
    z = np.asarray(bp.sheet_squared_lengths(spacetime, 0))
    assert np.max(np.abs(z.imag)) < 1e-9 and np.min(z.real) > 0
    np.testing.assert_allclose(
        z.real, [25.5067, 21.7760, 8.5264, 20.7612, 10.1035, 14.5733],
        atol=5e-3)
    faces = np.asarray(action.face_holonomies())
    assert np.max(np.abs(np.abs(faces) - 1.0)) < 1e-9
    assert report.action_available and report.action_unavailable == ""


def test_the_point_the_drive_reaches_satisfies_the_equations():
    """At the point the drive reached, the covariance the report carries is
    the band filling of h_1 there (it commutes with h_1 and carries the three
    quarks), and a joint action built afresh with that covariance, the pinned
    fiber's constraints and the multipliers the report carries has a force
    of 1.7e-14 on the twelve shared coordinates, the report's force, and
    constraint residuals below 1e-12."""
    spacetime, action, report, _, _ = _relax(FIRST_CELL, (2, 0, 1))
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
    assert np.linalg.norm(gamma @ h - h @ gamma) < 1e-12 * np.linalg.norm(h)
    assert np.trace(gamma).real == pytest.approx(3.0, abs=1e-10)
    force = np.linalg.norm(_class_force(fresh, spacetime))
    assert force == pytest.approx(report.force_norm, rel=1e-6)
    assert force < 1e-12
    assert np.max(np.abs(fresh.moment_residuals())) <= 1e-12


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


def _follower(occupations, selection, band_tolerance=None):
    """A band follower of the declared occupations and selection, at the
    declared band tolerance (1e-15) unless one is given."""
    declaration = cob.SelfConsistentMeanFieldDeclaration()
    declaration.covariance_rule = cob.CovarianceRule.BandFilling
    declaration.band_occupations = list(occupations)
    declaration.band_selection = selection
    if band_tolerance is not None:
        declaration.band_tolerance = band_tolerance
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


def test_equal_decoupled_blocks_keep_their_degeneracy_exactly():
    """An operator of three equal blocks that no entry couples, its indices
    interleaved: the read decomposes each block on its own, so the copies of
    an eigenvalue are equal bit for bit, the spectrum groups into bands of
    rank three at the declared band tolerance 1e-15, every eigenvector is
    supported on one block, and the left eigenvectors invert the right
    ones."""
    eigenvalues = lambda t: np.array([1.0, 2.5, 4.0, 7.0])  # noqa
    sheet, sheet_vectors = _family(eigenvalues, 0.0, seed=5)
    sheet = np.asarray(sheet).reshape(4, 4)
    # index 3 * k + t is coordinate k of block t
    operator = np.kron(sheet, np.eye(3))

    read = _follower([2.0], cob.BandSelection.Continuation).read(
        list(operator.reshape(-1)))
    assert list(read.ranks) == [3, 3, 3, 3]
    ordered = np.asarray(read.ordered)
    for k in range(4):
        assert ordered[3 * k] == ordered[3 * k + 1] == ordered[3 * k + 2]
    np.testing.assert_allclose(ordered[::3], [1.0, 2.5, 4.0, 7.0],
                               atol=1e-12)
    vectors = np.asarray(read.eigenvectors).reshape(12, 12)
    for mode in range(12):
        support = {index % 3 for index in np.flatnonzero(vectors[:, mode])}
        assert len(support) == 1
    left = np.asarray(read.left_eigenvectors).reshape(12, 12)
    np.testing.assert_allclose(left @ vectors, np.eye(12), atol=1e-12)
    (band,) = read.bands
    assert band.rank == 3 and not band.ambiguous
    np.testing.assert_allclose(
        np.asarray(read.covariance).reshape(12, 12),
        (2.0 / 3.0) * np.kron(_projector(sheet_vectors, [0]), np.eye(3)),
        atol=1e-12)


def test_the_bands_of_the_sheeted_host_have_rank_three():
    """On the three-sheeted support of the run's first host cell the carrier
    operator is three equal blocks, and its spectrum is six bands of rank
    three at the declared band tolerance 1e-15."""
    content = (0, 0, 3)
    config = _config(FIRST_CELL, content)
    base = bp.build_base(config["edge_squared"], config["host_cell"])
    host = cs.sheeted_support(base, bp.SHEETS)
    action = cob.JointAction(host.spacetime,
                             bp.action_declaration(host.spacetime, 1.0, 1.0))
    read = cob.BandFollower(bp.mean_field_declaration(content, config)).read(
        action.carrier_operator())
    assert list(read.ranks) == [3] * 6
    ordered = np.asarray(read.ordered)
    for k in range(6):
        assert ordered[3 * k] == ordered[3 * k + 1] == ordered[3 * k + 2]


def test_a_defective_eigenbasis_is_inverted_as_computed_and_marked():
    """A Jordan block beside a simple eigenvalue. The eigensolver returns two
    eigenvectors of the block that are parallel to rounding, so the block's
    eigenvector matrix is singular at the threshold of its decomposition:
    the read says so (`defective`, the reciprocal condition at rounding) and
    is made with the inverse as computed. The two modes are one band at the
    declared band tolerance, and the band's projector, the sum over both, is
    the identity on the block; the covariance of one particle in it is half
    of that. A diagonalizable operator is not marked."""
    jordan = np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 3.0]],
                      dtype=complex)
    read = _follower([1.0], cob.BandSelection.Continuation).read(
        list(jordan.reshape(-1)))
    assert read.defective
    assert read.eigenbasis_reciprocal_condition < 1e-14
    assert list(read.ranks) == [2, 1]
    (band,) = read.bands
    assert band.rank == 2 and not band.overfilled
    expected = np.diag([0.5, 0.5, 0.0])
    np.testing.assert_allclose(np.asarray(read.covariance).reshape(3, 3),
                               expected, atol=1e-12)

    simple = np.diag([1.0, 2.0, 3.0]).astype(complex)
    simple[0, 1] = 0.5
    read = _follower([1.0], cob.BandSelection.Continuation).read(
        list(simple.reshape(-1)))
    assert not read.defective
    assert read.eigenbasis_reciprocal_condition > 0.1


def test_an_eigenbasis_near_a_defect_reports_its_condition():
    """Two eigenvalues eps apart on a Jordan-like block: the eigenbasis is
    not singular at the decomposition's threshold, each mode is its own band,
    and the reciprocal condition reported is of order eps, which is the size
    the band's projector is the reciprocal of."""
    for eps in (1e-4, 1e-8):
        block = np.array([[1.0, 1.0, 0.0], [0.0, 1.0 + eps, 0.0],
                          [0.0, 0.0, 3.0]], dtype=complex)
        read = _follower([1.0], cob.BandSelection.Continuation).read(
            list(block.reshape(-1)))
        assert not read.defective
        assert list(read.ranks) == [1, 1, 1]
        assert 0.1 * eps < read.eigenbasis_reciprocal_condition < 10.0 * eps
        largest = np.max(np.abs(np.asarray(read.covariance)))
        assert 0.1 / eps < largest < 10.0 / eps


@pytest.mark.parametrize("unit", [1e-9, 1.0, 1e9])
def test_the_bands_are_grouped_the_same_in_every_unit(unit):
    """The band tolerance is relative to the operator's largest eigenvalue
    modulus, so an operator and its multiple group alike. Two eigenvalues a
    part in 1e12 of the largest apart are two bands at the declared
    tolerance 1e-15 and one band at 1e-10, in the unit 1e-9, 1 and 1e9; a
    floor in the operator's own unit would join them at 1e-9."""
    values = np.array([1.0, 1.0 + 4e-12, 2.0, 4.0])
    operator = unit * np.diag(values).astype(complex)
    read = _follower([1.0], cob.BandSelection.Continuation).read(
        list(operator.reshape(-1)))
    assert list(read.ranks) == [1, 1, 1, 1]
    read = _follower([1.0], cob.BandSelection.Continuation, 1e-10).read(
        list(operator.reshape(-1)))
    assert list(read.ranks) == [2, 1, 1]
    # eigenvalues near zero are grouped at the same absolute distance as
    # the others: the unit is the operator's, not the eigenvalue's
    operator = unit * np.diag([0.0, 4e-12, 2.0, 4.0]).astype(complex)
    read = _follower([1.0], cob.BandSelection.Continuation, 1e-10).read(
        list(operator.reshape(-1)))
    assert list(read.ranks) == [2, 1, 1]


def test_a_band_of_three_sheets_is_followed_across_another():
    """Three identical sheets, as on the recursion's host: every band has one
    mode per sheet, rank three. The operator is built here as a Kronecker
    product, three equal blocks that no entry couples; each block is
    decomposed on its own, so the three copies of an eigenvalue are equal
    exactly and are one band at the declared band tolerance 1e-15. The band
    at 1 + t, holding two particles, crosses the band at 2 - t; it is
    followed as one band of rank three, after the crossing it holds places 3
    to 5, and the covariance is two thirds of its projector, the direct sum
    of one sheet's."""
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
    """(0123, 003) under the two named rules, each driven from the host.

    Followed by continuation, the declared band 2 (places 6 to 8 at the host)
    crosses below the host's middle band at the first accepted update, and
    every later step proposal reports the crossing. The drive ends after 38
    accepted updates at a residual norm of 7.1e-13 with the occupied band at
    2.898, its pinned value, at z = (33.214, 13.996, 19.556, 16.021, 13.651,
    21.301), off the real slice by up to 0.027. There the spectrum in
    ascending order of real part is 0.353, 2.898, 3.575, ... (three modes
    each, every triple one band of rank three at the declared band tolerance
    1e-15): the followed band holds places 3 to 5.

    Re-sorted at every point, the drive ends after 8 accepted updates at
    another point, z = (34.307, 12.155, 24.478, 33.301, 28.132, 37.576),
    with its occupied band at the pinned value too, and there it is the
    third triple, at places 6 to 8."""
    spacetime, action, followed, drive, _ = _relax(FIRST_CELL, (0, 0, 3))
    record = bp.relaxation_record(followed, drive)
    assert drive["stop_reason"] == cs.STOP_STATIONARY
    assert drive["trace"][-1] < 1e-12 and followed.force_norm < 1e-12
    crossings = [entry["band_crossing"] for entry in record["trace"]]
    assert crossings[0] is False and all(crossings[1:])
    assert record["band_crossing_iterates"] == len(crossings) - 1
    assert [entry["band_positions"] for entry in record["trace"]] == \
        [[[6, 7, 8]]] + [[[3, 4, 5]]] * (len(crossings) - 1)
    (band,) = followed.bands
    assert band.declared_index == 2
    assert list(band.declared_positions) == [6, 7, 8]
    assert list(band.positions) == [3, 4, 5] and band.crossed
    assert band.eigenvalues[0].real == pytest.approx(2.8977, abs=5e-4)
    assert drive["accepted_updates"] == 38
    z_followed = np.asarray(bp.sheet_squared_lengths(spacetime, 0))
    np.testing.assert_allclose(
        z_followed.real, [33.214, 13.996, 19.556, 16.021, 13.651, 21.301],
        atol=5e-3)
    assert np.max(np.abs(z_followed.imag)) == pytest.approx(0.0271, abs=5e-4)

    spectrum = np.sort(np.linalg.eigvals(
        bp.matrix(action.carrier_operator())).real)
    np.testing.assert_allclose(spectrum[0:3], 0.3525, atol=5e-4)
    np.testing.assert_allclose(spectrum[3:6], 2.8977, atol=5e-4)
    np.testing.assert_allclose(spectrum[6:9], 3.5751, atol=5e-4)
    assert list(followed.band_ranks) == [3] * 6

    resorted_spacetime, _, resorted, resorted_drive, _ = _relax(
        FIRST_CELL, (0, 0, 3), selection="sort-every-iterate")
    assert resorted.band_selection == cob.BandSelection.SortEveryIterate
    assert resorted_drive["stop_reason"] == cs.STOP_STATIONARY
    assert resorted_drive["trace"][-1] < 1e-12
    assert resorted_drive["accepted_updates"] == 8
    (band,) = resorted.bands
    assert band.eigenvalues[0].real == pytest.approx(2.8977, abs=5e-4)
    assert list(band.positions) == [6, 7, 8]
    z_resorted = np.asarray(bp.sheet_squared_lengths(resorted_spacetime, 0))
    np.testing.assert_allclose(
        z_resorted.real, [34.307, 12.155, 24.478, 33.301, 28.132, 37.576],
        atol=5e-3)
    assert np.max(np.abs(z_resorted - z_followed)) > 1.0


# ------------------------------------------------ stops and flags by name


def test_a_held_holonomy_driven_across_minus_one_is_named():
    """(0123, 300) with no fiber moment pinned: the step drives a held face
    holonomy across -1, where the monopole number read from the principal
    arguments jumps, and the system is not posed at a point that carries
    another number than the declared one. The drive lowers the residual
    norm from 133 to 0.333 in 22 accepted updates and ends where every one
    of the 24 trials of its line search lands in another sector and so has
    no residual: its detail says so, with the library's words for the last
    of them. It ends by itself; the geometry it stopped at is
    Kontsevich-Segal allowable (margin pi, a Euclidean cell) with the
    largest |z| 68 times the host's, so the read carries no flag of the
    geometry, and its record says why the drive stopped."""
    _, _, report, drive, config = _relax(FIRST_CELL, (3, 0, 0),
                                         fiber_moments=0)
    assert not report.converged
    assert drive["stop_reason"] == cs.STOP_STATIONARY
    assert drive["moves_committed"] == 0
    assert drive["trace"][0] == pytest.approx(133.26, rel=1e-3)
    assert report.force_norm == pytest.approx(0.33345, abs=5e-4)
    assert 15 <= drive["accepted_updates"] <= 40
    sector = ("HolomorphicRelaxation: held sector 0 is declared with "
              "monopole number 1 but the starting configuration carries 0")
    assert drive["stop_detail"].endswith(
        "since the last proposal 24 complexes were scored, 24 of them "
        "without a residual (the last: %s)" % sector)
    undefined = drive["objective"].undefined
    assert len(undefined) >= 24 and set(undefined[-24:]) == {sector}
    # every accepted update moved the geometry: the residual norm fell
    trace = drive["trace"]
    assert all(after < before for before, after in zip(trace, trace[1:]))
    assert report.kontsevich_segal_margin == pytest.approx(np.pi, abs=1e-5)
    assert report.largest_length_ratio == pytest.approx(67.94, rel=0.05)
    assert _flags(report, config) == []
    record = bp.relaxation_record(report, drive)
    assert record["stop_reason"] == \
        "no move and no scaled step lowers the residual norm"
    assert record["undefined_points"] == len(undefined)
    assert record["fiber_moments"] == 0 and record["multipliers"] == []


def test_lengths_that_grow_leave_the_allowable_domain_and_are_flagged():
    """(0123, 201) with no fiber moment pinned: the length equations are the
    matter force alone, which is 1/z^2 at large z and vanishes only at
    infinite length, so the step lengthens the cell. The engine scores
    every trial whatever its Kontsevich-Segal margin (its admissibility gate
    is off unless declared), so the drive follows the step out of the
    allowable domain: after 4 accepted updates every squared length is
    complex and five of the six have a negative real part (the largest |z|
    29 times the host's), the margin is -2.97, the residual norm is 2.14
    (from 90.9 at the host), and every one of the 24 trials of the last line
    search is scored and has no residual (a link of the trial is zero). A
    margin at or below the declared tolerance 1e-15 is flagged, with the
    margin and the tolerance it was compared with. The geometry is finite,
    so it has a value to read."""
    spacetime, _, report, drive, config = _relax(FIRST_CELL, (2, 0, 1),
                                                 fiber_moments=0)
    assert not report.converged
    assert drive["stop_reason"] == cs.STOP_STATIONARY
    assert drive["trace"][0] == pytest.approx(90.946, rel=1e-3)
    assert drive["trace"][-1] == pytest.approx(2.1402, rel=1e-2)
    assert drive["accepted_updates"] == 4
    assert drive["stop_detail"].endswith(
        "since the last proposal 24 complexes were scored, 24 of them without "
        "a residual (the last: Connection: a link must be nonzero (U_xy in "
        "C*))")
    last = drive["objective"].updates[-1]
    assert drive["objective"].scored - last["scored_before"] == 24
    assert report.largest_length_ratio == pytest.approx(28.79, rel=1e-2)
    z = np.asarray(bp.sheet_squared_lengths(spacetime, 0))
    np.testing.assert_allclose(
        sorted(z.real), [-230.30, -127.99, -53.44, -52.94, -18.23, 16.20],
        rtol=1e-2)
    assert np.min(np.abs(z.imag)) > 1.0
    margin = report.kontsevich_segal_margin
    assert margin == pytest.approx(-2.9709, abs=1e-3)
    tolerance = bp.declared_tolerance(config, "allowability_tolerance")
    assert tolerance == 1e-15
    flags = _flags(report, config)
    assert len(flags) == 1
    for flag in flags:
        assert flag["name"] == "not Kontsevich-Segal allowable"
        assert flag["kontsevich_segal_margin"] == margin
        assert flag["tolerance"] == tolerance
        assert "margin %.3g" % margin in flag["detail"]
    assert bp.geometry_without_value(spacetime, drive) is None
    assert bp.NO_VALUE_OVERFLOW == \
        "the squared lengths overflowed the double"
    assert not hasattr(cob.HolomorphicRelaxationDeclaration(),
                       "length_runaway_ratio")


def test_the_read_of_0123_021_reaches_a_fixed_point_of_its_equations():
    """(0123, 021): the drive lowers the residual norm from 8.84 at the host
    to 1.1e-13 in 13 accepted updates, every squared length staying real
    and positive (a Euclidean cell, margin pi), and ends where no trial of
    its line search lowers the norm by the tolerance; the pinned
    eigenvalues hold to rounding (residuals below 2e-14) and the read
    carries no flag of the geometry."""
    spacetime, _, report, drive, config = _relax(FIRST_CELL, (0, 2, 1))
    assert not report.converged
    assert drive["stop_reason"] == cs.STOP_STATIONARY
    assert drive["accepted_updates"] == 13
    assert drive["trace"][0] == pytest.approx(8.8403, rel=1e-3)
    assert drive["trace"][-1] < 1e-12
    assert drive["stop_detail"].endswith(
        "since the last proposal 24 complexes were scored, 0 of them without "
        "a residual")
    z = np.asarray(bp.sheet_squared_lengths(spacetime, 0))
    np.testing.assert_allclose(
        z.real, [7.040, 5.911, 9.746, 5.810, 9.623, 3.595], atol=5e-3)
    assert np.max(np.abs(z.imag)) < 1e-9
    assert max(abs(r) for r in report.moment_residuals) < 1e-13
    assert report.kontsevich_segal_margin == pytest.approx(np.pi, abs=1e-9)
    assert _flags(report, config) == []
    assert bp.geometry_without_value(spacetime, drive) is None


@pytest.mark.slow
def test_a_read_that_stops_short_moves_at_every_update_and_says_why():
    """(0123, 300) with the pinned fiber: the drive moves at every update it
    accepts (the covariance changes at every one), lengthening the cell to
    7.3e6 times the host's largest |z|, and ends by name after 462 accepted
    updates at a residual norm of 5.8e-8 (0.907 at the host, the norm of
    the multipliers' fit in the scaled equations), where no trial of the
    line search lowers the norm by the tolerance. The pinned eigenvalue
    holds there to 5.7e-8 of its target. The geometry is Kontsevich-Segal
    allowable (margin pi to 1e-5), so the read carries no flag of the
    geometry, and the record and its text say why the drive stopped."""
    _, _, report, drive, config = _relax(FIRST_CELL, (3, 0, 0))
    assert not report.converged
    assert drive["stop_reason"] == cs.STOP_STATIONARY
    assert drive["moves_committed"] == 0
    assert drive["stop_detail"].endswith(
        "since the last proposal 24 complexes were scored, 0 of them "
        "without a residual")
    accepted = drive["accepted_updates"]
    assert 400 <= accepted <= 550
    assert drive["trace"][0] == pytest.approx(0.90713, rel=1e-4)
    assert report.force_norm == pytest.approx(1.08e-8, rel=0.1)
    measured = [update["measured"] for update in drive["objective"].updates]
    changes = [step.covariance_change for step in measured[1:accepted + 1]]
    assert len(changes) == accepted and min(changes) > 0.0
    assert report.largest_length_ratio == pytest.approx(7.34e6, rel=0.05)
    assert abs(report.moment_residuals[0]) < 1e-7 * abs(
        report.moment_targets[0])
    assert report.kontsevich_segal_margin == pytest.approx(np.pi, abs=1e-5)
    assert _flags(report, config) == []
    record = bp.relaxation_record(report, drive)
    assert record["stop_reason"] == \
        "no move and no scaled step lowers the residual norm"
    assert record["accepted_updates"] == accepted
    text = bp.relaxation_text(record)
    assert text.startswith("solve converged False (residual norm ")
    assert ("; stopped: no move and no scaled step lowers the residual norm "
            "(") in text


# ------------------------------------------- limits the user declares


def test_no_limit_is_declared_unless_the_user_declares_one():
    """A drive carries no number of iterations, no number of relaxation
    updates and no time after which it stops: the three limits of the
    configuration are None, they reach `cell_solve.solve` as None, and the
    name of the stop a declared one gives exists for that case alone."""
    config = _config(FIRST_CELL, (0, 0, 3))
    assert [key for key, _, _ in bp.LIMITS] == [
        "iteration_limit", "update_limit", "time_limit_seconds"]
    arguments = bp.solve_arguments(config)
    for key, _, _ in bp.LIMITS:
        assert config[key] is None and arguments[key] is None
    assert cs.STOP_DECLARED_LIMIT == "a declared limit was reached"


def test_a_declared_number_of_updates_ends_the_solve_by_name():
    """(0123, 003), which takes 38 accepted updates, with one iteration of
    the drive and two relaxation updates declared: the drive takes the two
    and stops at a residual norm of 0.0238 (9.22 at the host), and says that
    the limit was the user's."""
    _, _, report, drive, _ = _relax(
        FIRST_CELL, (0, 0, 3),
        limits={"iteration_limit": 1, "update_limit": 2})
    assert not report.converged
    assert drive["stop_reason"] == cs.STOP_DECLARED_LIMIT
    assert drive["accepted_updates"] == 2
    assert len(drive["objective"].updates) == 2
    assert drive["trace"] == pytest.approx([9.2198, 1.4020, 0.023845],
                                           rel=1e-4)
    assert drive["stop_detail"].startswith(
        "a declared count ended the drive (iteration limit 1, update limit "
        "2) after 2 step proposals; the residual norm is ")
    record = bp.relaxation_record(report, drive)
    assert record["stop_reason"] == "a declared limit was reached"
    assert record["accepted_updates"] == 2


def test_a_declared_time_ends_the_solve_by_name():
    """A time limit of zero seconds is reached before the first step: the
    base is the host as built, and the report is the host's."""
    spacetime, _, report, drive, _ = _relax(
        FIRST_CELL, (0, 0, 3), limits={"time_limit_seconds": 0.0})
    assert not report.converged
    assert drive["stop_reason"] == cs.STOP_DECLARED_LIMIT
    assert drive["accepted_updates"] == 0 and drive["trace"] == []
    assert drive["stop_detail"].startswith(
        "the declared time limit of 0 seconds was reached after 0 step "
        "proposals")
    assert report.force_norm == pytest.approx(9.2198, rel=1e-4)
    assert report.largest_length_ratio == 1.0
    np.testing.assert_array_equal(
        np.asarray(bp.sheet_squared_lengths(spacetime, 0)),
        np.asarray(bp.sheet_squared_lengths(bp.build_host(
            8.0, RUN.HOST_CELLS[FIRST_CELL]), 0)))


# ------------------------------------------------ the constrained step


def _held_host_system(content):
    """The stationarity system of (0123, content) at the host with the
    covariance fixed at the band filling there and the cell's faces held:
    the base cell, the system over its three-sheeted support
    (`cell_solve.GeometricSystem`), the support and its geometry
    declaration."""
    config = _config(FIRST_CELL, content)
    base = bp.build_base(config["edge_squared"], config["host_cell"])
    host = cs.sheeted_support(base, bp.SHEETS)
    action = cob.JointAction(host.spacetime,
                             bp.action_declaration(host.spacetime, 1.0, 1.0))
    read = cob.BandFollower(bp.mean_field_declaration(content, config)).read(
        action.carrier_operator())
    covariance = list(read.covariance)

    def declare(spacetime):
        declaration = bp.action_declaration(spacetime, 1.0, 1.0)
        declaration.covariance = covariance
        return declaration

    def geometry_of(support):
        return bp.support_geometry(config, support, host)

    system = cs.GeometricSystem(declare, geometry_of, bp.SHEETS)
    return base, system, host, geometry_of(host)


def _held_modulus_projector(geometry, support):
    """The projector onto the link-modulus directions the held faces fix, in
    the twelve shared coordinates' link block: the row space of the held
    faces' coboundary pulled back through the support's edge classes."""
    edges = bp.edge_records(support.spacetime)
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
    expansion = np.zeros((len(edges), 6))
    for edge, (c, o) in enumerate(zip(support.classes,
                                      support.orientations)):
        expansion[edge, c] = o
    _, s, vt = np.linalg.svd(np.array(rows) @ expansion)
    basis = vt[:int(np.sum(s > 1e-10 * s[0]))].T
    return basis @ basis.T


def test_the_constrained_step_leaves_less_of_the_linear_residual():
    """(0123, 021) at the host, the covariance fixed. The constrained Newton
    step (`HolomorphicRelaxation.newton_step`) is the least-squares step,
    over the steps that keep the held moduli, of the equations with every
    row and column multiplied by its variable's scale (`variable_scales`:
    the squared length 8 of the host on the lengths, one on the links). In
    that norm it leaves 14.7 % of the linearized residual, and the
    projection of the unconstrained minimum-norm Newton step onto the held
    directions, a step that keeps them too, leaves 29.5 %; in the unscaled
    norm, which the step reports (`linear_residual`), they leave 9.0 % and
    28.5 %. The step keeps every held modulus to rounding, it is a descent
    direction for the residual norm, and the first update of a drive along
    it lowers the residual norm from 9.91 to 8.07 with the held moduli
    drifting by one unit of rounding."""
    base, system, host, geometry = _held_host_system((0, 2, 1))
    relaxation = system.point(base).relaxation
    assert relaxation.variable_count() == 12
    jacobian = np.asarray(relaxation.jacobian()).reshape(12, 12)
    residual = np.asarray(relaxation.residual())
    held = _held_modulus_projector(geometry, host)
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
    scales = np.asarray(relaxation.variable_scales())

    def scaled_left(step):
        return np.linalg.norm(scales * (residual + jacobian @ step)) / \
            np.linalg.norm(scales * residual)

    assert scaled_left(projected) == pytest.approx(0.2946, abs=5e-4)

    newton = relaxation.newton_step()
    assert newton.constrained
    assert newton.rank_tolerance == geometry.rank_tolerance == 1e-15
    assert newton.jacobian_rank == rank == 11
    assert newton.constrained_rank == 19
    assert newton.constrained_rank_gap > 1e5
    assert newton.residual_norm == pytest.approx(np.linalg.norm(residual),
                                                 rel=1e-12)
    assert newton.linear_residual == pytest.approx(0.0897, abs=5e-4)
    assert newton.linear_residual < projected_left / 3
    # the step keeps the held moduli, leaves what it reports of the
    # linearized residual, and is a descent direction
    taken = np.asarray(newton.step)
    assert np.max(np.abs(held @ taken[6:].real)) < 1e-12
    left = np.linalg.norm(residual + jacobian @ taken) / \
        np.linalg.norm(residual)
    assert left == pytest.approx(newton.linear_residual, rel=1e-6)
    assert scaled_left(taken) == pytest.approx(0.1472, abs=5e-4)
    assert scaled_left(taken) < scaled_left(projected)
    assert np.vdot(residual, jacobian @ taken).real < 0.0
    # one relaxation update of a drive along it
    before = relaxation.held_log_moduli()
    drive = cs.solve(base, system, moves=False, iteration_limit=1)
    assert drive["accepted_updates"] == 1
    assert drive["trace"] == pytest.approx([9.9098, 8.0686], rel=1e-4)
    after = system.point(drive["spacetime"]).relaxation.held_log_moduli()
    assert len(before) == len(after) == 12
    assert max(abs(a - b) for a, b in zip(after, before)) < 1e-14


def test_holding_the_moduli_costs_the_step_nothing_on_the_real_slice():
    """On the real slice the joint system is real-structured, so its Newton
    step changes no modulus and the constrained step solves the linearized
    equations to rounding, as the unconstrained step does: the first step
    proposal of (0123, 201) is constrained and leaves less than 1e-10 of the
    linearized residual."""
    _, _, _, drive, _ = _relax(
        FIRST_CELL, (2, 0, 1),
        limits={"iteration_limit": 1, "update_limit": 1})
    first = drive["objective"].updates[0]
    assert first["constrained_step"]
    assert first["linear_residual"] < 1e-10
