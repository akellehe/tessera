# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Properties of the self-consistent stationarity system of a content
(`cell_solve.ContentSystem`) and of the systems a cell solve scores, each
checked at the declared tolerances (#1372).

Terms used below:

* the *content system* of a cell and a content is the stationarity system of
  the driver's joint action (`baryon_poles.action_declaration`: the primal
  Regge term on the interior hinges, the Villain term at beta = 1 and the
  mean-field term) with the covariance rebuilt at every point from the bands
  the content fills and the occupied fiber pinned at the host
  (`SelfConsistentMeanField.joint_system`), over the squared lengths z_e,
  the links U_e and the multipliers xi_j of the pinned constraints;
* its *residual* R has the length rows dS/dz_e, the link rows U_e dS/dU_e
  and the constraint rows c_j - c_j*; its *Jacobian* J is the closed form of
  `HolomorphicRelaxation.jacobian`;
* a *gauge transformation* is U_xy -> g_x^-1 U_xy g_y with g_x = exp(i
  theta_x) and theta_x complex (WP v18 section 3): *unit-modulus* when every
  theta_x is real, *complex* otherwise;
* the *oracle* of a Jacobian block is the central difference of the residual
  at a step that is 1e-5 of the coordinate's size. It is a test device, not
  a library mode.

The cells are the two host cells of the recursion run
(`tests.drivers._recursion_run_2026_09_23.HOST_CELLS`), one tetrahedron
each, as recorded (squared lengths 8) and dilated by 1e9, the scale of a
grown level's cells. Every system is posed on one sheet unless a test says
otherwise, and is read at a point displaced from the host by 3 to 6 per cent
in the squared lengths (with imaginary parts) and by up to 0.075 in the
links (off the unit circle), where no equation is solved.
"""
import cmath
import itertools

import numpy as np
import pytest

import tessera as T
from tessera import cobordism as cob
from tessera import numerics as nm
from tessera.drivers import baryon_poles as bp
from tessera.drivers import cell_solve as cs
from tessera.drivers.series_stationarity import SeriesStationarity

from tests.drivers import _recursion_run_2026_09_23 as RUN

FIRST_CELL = (0, 1, 2, 3)
SECOND_CELL = (0, 1, 3, 4)

#: The host cell (0, 1, 2, 3) of tick 0 of the recursion run of 2026-10-01
#: (`v18-multicobordism-5t`): the cell of `HOST_CELLS` with links that differ
#: from it in their last digits, as that run's level relaxation left them.
CELL_OF_2026_10_01 = {
    "squared_lengths": [complex(8.000000000000002, 0.0)] * 6,
    "links": [complex(1.0, 2.7509237127789626e-16),
              complex(-0.4999999999999998, -0.8660254037844387),
              complex(1.0, 1.8073963429006066e-16),
              complex(-0.5000000000000002, 0.8660254037844385),
              complex(1.0, -1.337633413904918e-16),
              complex(1.0, -1.5071560638611574e-16)],
}

#: The pinned and unpinned declarations a content is posed with: the content,
#: the form of the fiber's constraints and how many are pinned.
POSED = {
    "three bands, eigenvalues pinned": ((1, 1, 1), "eigenvalues", "r"),
    "three bands, power sums pinned": ((1, 1, 1), "power-sums", "bands"),
    "one band, eigenvalue pinned": ((0, 0, 1), "eigenvalues", "r"),
    "two bands, nothing pinned": ((1, 0, 1), "eigenvalues", "0"),
}


# -------------------------------------------------------------------- fixtures


def _cell(key, scale=1.0):
    """A recorded host cell (a key of `HOST_CELLS`, or the cell itself)
    with its squared lengths multiplied by ``scale``."""
    cell = RUN.HOST_CELLS[key] if isinstance(key, tuple) else key
    return {"squared_lengths": [z * scale for z in cell["squared_lengths"]],
            "links": list(cell["links"])}


def _system(base, content, sheets=1, pinning="eigenvalues", moments="r"):
    """The content system of ``content`` with ``base`` as its host, on
    ``sheets`` sheets, as `baryon_poles.relax_content` builds it."""
    config = bp.default_config([1.0], [1.0], selected_contents=[(1, 1, 1)],
                               fiber_moments=moments, fiber_pinning=pinning)
    villain_order = bp.declared_villain_order(config)

    def declare(spacetime):
        return bp.action_declaration(spacetime, 1.0, 1.0,
                                     config["regge_hinges"],
                                     villain_order=villain_order)

    host = cs.sheeted_support(base, sheets)
    count = bp.fiber_moment_count(
        bp.mean_field_declaration(content, config),
        cob.JointAction(host.spacetime, declare(host.spacetime)), moments,
        pinning)

    def mean_field_of(support):
        declaration = bp.mean_field_declaration(content, config)
        declaration.geometry = bp.support_geometry(config, support, host)
        declaration.fiber_moments = count
        return declaration

    return cs.ContentSystem(declare, mean_field_of, base, sheets)


def _move_length(edge, step):
    """z -> z + step, written through the square root on the edge's side."""
    length = complex(edge.getLength())
    root = cmath.sqrt(length * length + step)
    if (root.conjugate() * length).real < 0.0:
        root = -root
    edge.setLength(root)


def _move_link(edge, step):
    """U -> U exp(step): on the stored phase, phi -> phi - i step."""
    edge.setPhase(complex(edge.getPhase()) - 1j * step)


def _displace(base):
    """Move the base off its host: every squared length by 3 or 6 per cent
    of itself times 1 + 0.1 i (or not at all), every link by a
    Maurer-Cartan increment of up to 0.075 with an imaginary part."""
    for index, edge in enumerate(base.getEdgeList().toVector()):
        _move_length(edge, 0.03 * ((index * 7) % 5 - 2)
                     * complex(edge.getLength()) ** 2 * (1.0 + 0.1j))
        _move_link(edge, 0.05 * ((index * 3) % 4 - 1.5)
                   + 0.02j * (index % 3 - 1))


def _gauge(base, theta):
    """U_xy -> g_x^-1 U_xy g_y with g = exp(i theta): on the stored phase,
    phi_xy -> phi_xy + theta_y - theta_x."""
    for edge in base.getEdgeList().toVector():
        source = int(edge.getSource().getId())
        target = int(edge.getTarget().getId())
        edge.setPhase(complex(edge.getPhase()) + theta[target]
                      - theta[source])


def _snapshot(base):
    return [(complex(e.getLength()), complex(e.getPhase()))
            for e in base.getEdgeList().toVector()]


def _restore(base, snapshot):
    for edge, (length, phase) in zip(base.getEdgeList().toVector(),
                                     snapshot):
        edge.setLength(length)
        edge.setPhase(phase)


#: A unit-modulus and a complex gauge transformation of the four vertices.
GAUGES = {
    "unit-modulus": {0: 0.3, 1: 2.1, 2: 4.4, 3: 1.2},
    "complex": {0: 0.3 + 0.4j, 1: 2.1 - 0.7j, 2: 4.4 + 0.2j, 3: 1.2 - 0.3j},
}


def _displaced_point(key, scale, posed, theta=None, sheets=1, content=None):
    """The base of a cell, gauge-transformed by ``theta`` when given, the
    content system built with it as the host, and the system's point after
    the displacement."""
    declared, pinning, moments = POSED[posed]
    base = bp.build_base(8.0, _cell(key, scale))
    if theta is not None:
        _gauge(base, theta)
    system = _system(base, content or declared, sheets, pinning, moments)
    _displace(base)
    return base, system, system.point(base)


def _reads(point):
    """The residual, its square Jacobian and the number of edge
    coordinates."""
    residual = np.asarray(point.relaxation.residual())
    size = len(residual)
    jacobian = np.asarray(point.relaxation.jacobian()).reshape(size, size)
    return residual, jacobian, point.count


def _relative(measured, expected):
    measured = np.asarray(measured, dtype=complex)
    expected = np.asarray(expected, dtype=complex)
    return float(np.linalg.norm(np.ravel(measured - expected))
                 / max(np.linalg.norm(np.ravel(expected)), 1e-300))


# ----------------------------------------------------------- gauge invariance


@pytest.mark.parametrize("kind", ["unit-modulus", "complex"])
@pytest.mark.parametrize("posed", list(POSED))
@pytest.mark.parametrize("key, scale", [(FIRST_CELL, 1.0), (SECOND_CELL, 1.0),
                                        (FIRST_CELL, 1e9)])
def test_the_content_system_is_constant_on_gauge_orbits(key, scale, posed,
                                                        kind):
    """With the host and the point transformed by one gauge transformation,
    unit-modulus or complex, the residual, the action, the constraints'
    values and the least-squares multipliers of the content system are the
    same (WP v18 section 3: the spectrum of h_k and every spectral gate are
    gauge-invariant). Measured over the 24 cases: the largest relative
    change of the residual is 2e-12, at the dilated cell."""
    _, _, point = _displaced_point(key, scale, posed)
    _, _, moved = _displaced_point(key, scale, posed, GAUGES[kind])
    action, other = point.relaxation.action, moved.relaxation.action
    assert _relative(moved.relaxation.residual(),
                     point.relaxation.residual()) < 1e-10
    assert _relative(other.value(), action.value()) < 1e-11
    if action.constraint_count():
        assert _relative(other.constraint_values(),
                         action.constraint_values()) < 1e-12
        assert _relative(other.multipliers(), action.multipliers()) < 1e-10


@pytest.mark.parametrize("posed", list(POSED))
@pytest.mark.parametrize("key, scale", [(FIRST_CELL, 1.0), (SECOND_CELL, 1.0),
                                        (FIRST_CELL, 1e9)])
def test_gauge_directions_are_null_directions_of_the_jacobian(key, scale,
                                                              posed):
    """An infinitesimal gauge transformation at a vertex x, real or
    imaginary, moves the link of every edge at x by the Maurer-Cartan
    increment +-i theta and nothing else. The residual does not change
    along it, so the Jacobian annihilates it: |J d| is below 1e-12 of |J|
    for each of the eight directions. Measured: at most 4e-15."""
    base, _, point = _displaced_point(key, scale, posed)
    _, jacobian, count = _reads(point)
    fields = cs.edge_fields(base)
    for vertex in range(4):
        for unit in (1.0, 1j):
            direction = np.zeros(jacobian.shape[0], dtype=complex)
            for index, (source, target, _, _) in enumerate(fields):
                direction[count + index] = 1j * unit * (
                    (target == vertex) - (source == vertex))
            assert np.linalg.norm(jacobian @ direction) \
                < 1e-12 * np.linalg.norm(jacobian)


@pytest.mark.xfail(strict=True, reason=(
    "a band is followed by the largest weight of the present eigenvectors "
    "in the reference projector, which is taken in the gauge of the host; a "
    "gauge transformation of the point alone changes the weights, and the "
    "system then fills other modes: theory(cobordism): following a band by "
    "the largest weight in its previous projector, and what happens at a "
    "crossing, https://github.com/akellehe/tessera/issues/1319"))
def test_a_gauge_transformation_of_the_point_alone_keeps_the_bands():
    """The content system of (0123, 111) built at the host and read at the
    displaced point, and read again at the same point in another gauge
    (theta = 0, 2, 4, 1 on the four vertices, unit-modulus): the bands the
    covariance fills hold the same places of the spectrum and the residual
    norm is the same. Measured: the places move from [0], [1], [2] to
    [2], [3], [5] and the residual norm from 0.4197 to 1.614."""
    base, system, point = _displaced_point(FIRST_CELL, 1.0,
                                           "three bands, eigenvalues pinned")
    places = [list(band.positions) for band in system.iterate(base)[0].bands]
    norm = float(np.linalg.norm(point.relaxation.residual()))
    _gauge(base, {0: 0.0, 1: 2.0, 2: 4.0, 3: 1.0})
    assert [list(band.positions)
            for band in system.iterate(base)[0].bands] == places
    assert float(np.linalg.norm(system.point(base).relaxation.residual())) \
        == pytest.approx(norm, rel=1e-10)


# ------------------------------------------------- the multipliers of a point


@pytest.mark.parametrize("posed", ["three bands, eigenvalues pinned",
                                   "three bands, power sums pinned",
                                   "one band, eigenvalue pinned"])
@pytest.mark.parametrize("key, scale", [(FIRST_CELL, 1.0), (SECOND_CELL, 1.0),
                                        (FIRST_CELL, 1e9)])
def test_the_multipliers_are_the_least_squares_ones(key, scale, posed):
    """The geometric rows of the residual are R = F + G xi with G the
    constraints' gradients, the multiplier columns of the Jacobian. The
    multipliers a point carries are the least-squares solution of
    G xi = -F: they agree with `numpy.linalg.lstsq` to 1e-10 of their size
    (measured: 2e-13 at most)."""
    _, _, point = _displaced_point(key, scale, posed)
    residual, jacobian, count = _reads(point)
    gradients = jacobian[:2 * count, 2 * count:]
    multipliers = np.asarray(point.relaxation.action.multipliers())
    force = residual[:2 * count] - gradients @ multipliers
    expected = np.linalg.lstsq(gradients, -force, rcond=None)[0]
    assert _relative(multipliers, expected) < 1e-10


# ------------------------------------------------------ three sheets and one


@pytest.mark.parametrize("key", [
    FIRST_CELL, SECOND_CELL,
    pytest.param(CELL_OF_2026_10_01, marks=pytest.mark.xfail(
        strict=True, reason=(
            "on the three-sheeted support of this cell the bands at the "
            "host read ranks 3, 2 and 1: fix(cobordism): the band follower "
            "splits the exact degeneracy of the sheets by the rounding of "
            "one dense eigendecomposition, https://github.com/akellehe/"
            "tessera/issues/1356")),
        id="the cell of 2026-10-01")])
def test_three_sheets_pose_three_times_the_equations_of_one(key):
    """The content (1, 1, 1) on three sheets fills every band with one
    particle over three modes, one per sheet, which is the content
    (1/3, 1/3, 1/3) on one sheet, three times over. The squared lengths and
    the links are one shared field, so the geometric rows of the
    three-sheeted residual are three times those of the one-sheet system,
    and the constraint rows (a band's eigenvalue in the fiber's unit) are
    equal. Measured on the two cells of `HOST_CELLS`: 4e-14 and 5e-16 at
    most. On the cell of 2026-10-01, whose links differ from the first
    cell's in their last digits, the geometric rows differ by 0.67 of their
    size and the constraint rows by 0.021."""
    posed = "three bands, eigenvalues pinned"
    _, _, three = _displaced_point(key, 1.0, posed, sheets=3)
    _, _, one = _displaced_point(key, 1.0, posed,
                                 content=(1 / 3, 1 / 3, 1 / 3))
    count = one.count
    assert three.count == count
    sheeted = np.asarray(three.relaxation.residual())
    single = np.asarray(one.relaxation.residual())
    assert _relative(sheeted[:2 * count], 3.0 * single[:2 * count]) < 1e-12
    assert np.linalg.norm(sheeted[2 * count:] - single[2 * count:]) < 1e-13


# ----------------------------------------------------- a dilation of the cell


def _logarithmic_residuals(scale):
    """The norms of z_e dS/dz_e and of U_e dS/dU_e at the displaced point of
    (0123, 111) with its three bands' eigenvalues pinned, the cell dilated
    by ``scale``."""
    base, _, point = _displaced_point(FIRST_CELL, scale,
                                      "three bands, eigenvalues pinned")
    residual, _, count = _reads(point)
    squared = np.array([length * length
                        for _, _, length, _ in cs.edge_fields(base)])
    return (float(np.linalg.norm(squared * residual[:count])),
            float(np.linalg.norm(residual[count:2 * count])))


@pytest.mark.xfail(strict=True, reason=(
    "the multipliers are the least squares of the residual in dS/dz and "
    "U dS/dU together, whose relative weight is the unit of the squared "
    "lengths, so a dilation changes them: fix(cobordism): the step's rank "
    "decision is taken in units that depend on the squared lengths, and on "
    "a dilated cell it discards the length equations, "
    "https://github.com/akellehe/tessera/issues/1358"))
def test_a_dilation_leaves_the_pinned_system_as_it_is():
    """With the eigenvalue of every occupied band pinned, the mean-field
    force is a combination of the constraints' gradients and the
    multipliers absorb it, so what is left of the system is the Villain
    term and the constraints, which do not change under a dilation of the
    cell (the constraints are band eigenvalues in the fiber's own unit).
    The equations in the logarithmic coordinates, z_e dS/dz_e and
    U_e dS/dU_e, are then the same at every scale. Measured at the
    displaced point of (0123, 111): 1.948 and 0.2915 at squared lengths of
    order 8, and 5.296 and 2.8e-14 at order 8e9, where the least squares
    weighs the length rows by 1e-10; with the least squares taken in
    z dS/dz they are 0.918 and 0.559 at both."""
    lengths, links = _logarithmic_residuals(1.0)
    dilated_lengths, dilated_links = _logarithmic_residuals(1e9)
    assert dilated_lengths == pytest.approx(lengths, rel=1e-8)
    assert dilated_links == pytest.approx(links, rel=1e-8)


# ------------------------------------------- the Jacobian against the oracle


def _oracle(base, point, relative_step=1e-5):
    """The geometric columns of the Jacobian by central differences of the
    point's residual: a squared length moved by ``relative_step`` of its
    size, a link by the Maurer-Cartan increment ``relative_step``, the base
    restored exactly after every evaluation."""
    relaxation = point.relaxation
    snapshot = _snapshot(base)
    edges = base.getEdgeList().toVector()
    count = point.count
    columns = np.zeros((relaxation.variable_count(), 2 * count),
                       dtype=complex)

    def displaced(move, edge, step):
        move(edge, step)
        value = np.asarray(relaxation.residual())
        _restore(base, snapshot)
        return value

    for index, edge in enumerate(edges):
        step = relative_step * abs(complex(edge.getLength()) ** 2)
        columns[:, index] = (displaced(_move_length, edge, step)
                             - displaced(_move_length, edge, -step)) \
            / (2.0 * step)
        columns[:, count + index] = (
            displaced(_move_link, edge, relative_step)
            - displaced(_move_link, edge, -relative_step)) \
            / (2.0 * relative_step)
    return columns


@pytest.mark.parametrize("posed", ["three bands, eigenvalues pinned",
                                   "three bands, power sums pinned",
                                   "one band, eigenvalue pinned"])
@pytest.mark.parametrize("key, scale", [(FIRST_CELL, 1.0), (SECOND_CELL, 1.0),
                                        (FIRST_CELL, 1e9)])
def test_every_block_of_the_jacobian_is_the_derivative_of_its_rows(
        key, scale, posed):
    """Block by block, each relative to the block's own size: the
    length-length, length-link, link-length and link-link blocks and the
    constraint rows of the content system's Jacobian agree with the oracle
    of the residual, at squared lengths of order 8 and of order 8e9, where
    the length rows are 1e-10 of the link rows and the length-length block
    1e-19 of the link-link block. The Jacobian is symmetric. Measured: the
    largest relative distance of a block is 2e-9 at both scales, and the
    departure from symmetry 2e-15."""
    base, _, point = _displaced_point(key, scale, posed)
    _, jacobian, count = _reads(point)
    oracle = _oracle(base, point)
    size = jacobian.shape[0]
    blocks = {"length-length": (slice(0, count), slice(0, count)),
              "length-link": (slice(0, count), slice(count, 2 * count)),
              "link-length": (slice(count, 2 * count), slice(0, count)),
              "link-link": (slice(count, 2 * count),
                            slice(count, 2 * count)),
              "constraint-length": (slice(2 * count, size), slice(0, count)),
              "constraint-link": (slice(2 * count, size),
                                  slice(count, 2 * count))}
    for name, (rows, columns) in blocks.items():
        assert _relative(oracle[rows, columns],
                         jacobian[rows, columns]) < 1e-6, name
    assert _relative(jacobian.T, jacobian) < 1e-10


# ------------------------------------------------- the series residual's order


@pytest.mark.parametrize("posed", ["three bands, eigenvalues pinned",
                                   "two bands, nothing pinned"])
@pytest.mark.parametrize("scale", [1.0, 1e9])
def test_the_series_residual_to_order_one_at_both_scales(scale, posed):
    """`SeriesStationarity.residual` on the path x_0 + d t: its order-0
    coefficient is the residual and its order-1 coefficient is J d, block
    by block, at squared lengths of order 8 and of order 8e9, for a
    direction d whose length entries are of the size of the squared
    lengths. Measured: 2e-13 of the block at most for the order-0
    coefficient where the block is not cancelled to rounding, and 2e-14
    for the order-1 coefficient."""
    base, _, point = _displaced_point(SECOND_CELL, scale, posed)
    residual, jacobian, count = _reads(point)
    size = len(residual)
    follower = cob.BandFollower(point.mean_field)
    follower.set_reference(point.reference)
    series = SeriesStationarity(point.relaxation.action, point.geometry,
                                mean_field=point.mean_field,
                                follower=follower)
    rng = np.random.default_rng(3)
    direction = rng.normal(size=size) + 1j * rng.normal(size=size)
    direction[:count] *= np.array([abs(length * length) for _, _, length, _
                                   in cs.edge_fields(base)])
    out = series.residual([nm.TruncatedSeries([0.0, complex(value)])
                           for value in direction])
    zeroth = np.array([entry.coefficients[0] for entry in out])
    first = np.array([entry.coefficients[1] for entry in out])
    expected = jacobian @ direction
    multipliers = np.asarray(point.relaxation.action.multipliers())
    absorbed = np.zeros(size, dtype=complex)
    if len(multipliers):
        absorbed[:2 * count] = jacobian[:2 * count, 2 * count:] @ multipliers
    for rows in (slice(0, count), slice(count, 2 * count),
                 slice(2 * count, size)):
        if rows.start == rows.stop:
            continue
        # the residual of a pinned system is what is left of the force once
        # the multipliers' term has cancelled most of it, so its rounding is
        # that of the cancelled term
        scale = max(np.linalg.norm(residual[rows]),
                    np.linalg.norm(absorbed[rows]))
        assert np.linalg.norm(zeroth[rows] - residual[rows]) < 1e-9 * scale
        assert _relative(first[rows], expected[rows]) < 1e-9


# ---------------------------------------------- the Regge sheet a solve reads


def _regge_sphere(squared_of):
    """The boundary of the 4-simplex with the squared length
    ``squared_of(a, b)`` on the edge of vertices a < b and trivial links."""
    cells = [list(c) for c in itertools.combinations(range(5), 4)]
    spacetime = T.Spacetime.fromVertexTuples(3, cells, 1.0, 0.0)
    for edge in spacetime.getEdgeList().toVector():
        a, b = sorted((int(edge.getSource().getId()),
                       int(edge.getTarget().getId())))
        edge.setLength(cmath.sqrt(complex(squared_of(a, b))))
    return spacetime


def _regge_declaration(start=None):
    declaration = cob.JointActionDeclaration()
    declaration.carrier_degree = 1
    declaration.gravitational_weight = 1.0
    declaration.regge_form = cob.ReggeForm.Primal
    declaration.regge_hinges = cob.ReggeHinges.Interior
    declaration.holonomy_weight = 0.0
    declaration.matter_weight = 0.0
    if start is not None:
        declaration.regge_start_squared_lengths = list(start)
    return declaration


@pytest.mark.xfail(strict=True, reason=(
    "`cell_solve` builds the joint action anew at every point it scores and "
    "declares no `regge_start_squared_lengths`, so the continued Regge "
    "sheets start at the real projection of the point scored and not of "
    "the geometry the solve starts from; where that projection leaves the "
    "Euclidean domain the residual is read on another sheet"))
def test_a_solve_reads_the_regge_term_on_the_sheet_of_its_start():
    """A solve that starts at the regular boundary of the 4-simplex
    (squared lengths 8) and scores the point where the squared length of
    one edge is 33 + 0.5 i (its real part violates the triangle inequality
    with two edges of 8): the residual `cell_solve.GeometricSystem.point`
    gives there is the residual on the sheets continued from the start,
    which `JointActionDeclaration.regge_start_squared_lengths` declares
    (WP specification section 4.2: for complex data the branch is fixed by
    continuation from a Euclidean reference). Measured: the residual norm
    is 54.66 at the point the solve scores and 2.184 on the continued
    sheets."""
    geometry = cob.HolomorphicRelaxationDeclaration()
    geometry.relax_lengths = True
    geometry.relax_links = False
    geometry.relax_multipliers = False
    system = cs.GeometricSystem(lambda spacetime: _regge_declaration(),
                                lambda support: geometry, 1)
    start = _regge_sphere(lambda a, b: 8.0)
    started = [complex(length * length)
               for _, _, length, _ in cs.edge_fields(start)]
    scored = _regge_sphere(
        lambda a, b: 33.0 + 0.5j if (a, b) == (0, 1) else 8.0)
    continued = cob.HolomorphicRelaxation(
        cob.JointAction(scored, _regge_declaration(started)), geometry)
    assert _relative(system.point(scored).relaxation.residual(),
                     continued.residual()) < 1e-12


# ------------------------------------------------- the count of the fiber's
# constraints


@pytest.mark.parametrize("content, occupied", [((1, 1, 1), 3), ((0, 0, 1), 1),
                                               ((1, 0, 1), 2), ((2, 1, 0), 2)])
def test_the_number_of_pinned_constraints_of_a_content(content, occupied):
    """`fiber_moment_count` on the three-sheeted host of the declared cell:
    an integer setting is that integer; "bands" is the number of bands the
    content occupies, and so is "r" when the bands' eigenvalues are
    pinned."""
    config = bp.default_config([1.0], [1.0], selected_contents=[content])
    host = cs.sheeted_support(bp.build_base(8.0), bp.SHEETS)
    action = cob.JointAction(host.spacetime,
                             bp.action_declaration(host.spacetime, 1.0, 1.0))
    declaration = bp.mean_field_declaration(content, config)
    assert bp.fiber_moment_count(declaration, action, "0") == 0
    assert bp.fiber_moment_count(declaration, action, "2",
                                 "power-sums") == 2
    assert bp.fiber_moment_count(declaration, action, "bands",
                                 "power-sums") == occupied
    assert bp.fiber_moment_count(declaration, action, "r",
                                 "eigenvalues") == occupied
