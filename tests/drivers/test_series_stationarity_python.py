# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""`tessera.drivers.series_stationarity.SeriesStationarity`: the stationarity
residual of the joint action on a truncated-series path, against the
library's own numbers (#1345).

Terms used below:

* the *system* is a `cobordism.HolomorphicRelaxation`: its ``residual()`` is
  the vector R of the stationarity equations in scope at the current point
  x_0 and its ``jacobian()`` the closed-form Jacobian J of R in the relaxed
  coordinates (one squared length per edge class, one Maurer-Cartan link
  increment per edge class, one multiplier per constraint); the
  *self-consistent* system is that of
  `SelfConsistentMeanField.joint_system`, which rebuilds the covariance and
  the pinned fiber's projectors at every point from the bands followed from
  x_0;
* the *series residual* is `SeriesStationarity.residual` of a displacement
  s(t), one truncated series per variable with s(0) = 0: the Taylor
  coefficients of t -> R(x_0 + s(t)) to the order of the series;
* the *Cauchy oracle* reads the Taylor coefficients of t -> R(x_0 + t d)
  from the library's residual on a circle |t| = 1 at 64 equally spaced
  points, c_k = (1/64) sum_n R(x_0 + t_n d) t_n^-k: the fields are written at
  every point, the residual evaluated and the geometry restored. It is a
  test device, not a library mode. Its error is the aliasing c_(k+64) and
  the rounding of the residual, about 1e-15 times the residual's size on
  the circle, at every order;
* the *reversion* is the path s(t) = s_1 t + ... + s_p t^p with
  R(x_0 + s(t)) = (1 - t) R(x_0) + O(t^(p+1)), and the *step of order p* is
  s(1).

The fixtures are those of `test_analytic_jacobian_python` (the quarter-turn
tetrahedron with the Villain term; the boundary of the 4-simplex with the
Regge term on its continued sheets, fourteen dihedral angles off their
principal sheet; the sphere with a complex metric, a complex flux and the
matter term at a fixed covariance; the recorded host of the recursion run
with the driver's mean-field declaration), with three more built here: the
sphere carrying the three terms at once, the declared three-sheeted host of
`baryon_poles.build_host()` with and without its edge classes, and a host
with complex squared lengths and links off the unit circle.
"""
import cmath

import numpy as np
import pytest

from tessera import cobordism as cob
from tessera import numerics as nm
from tessera.drivers import baryon_poles as bp
from tessera.drivers.series_stationarity import SeriesStationarity

from tests.cobordism import test_analytic_jacobian_python as J
from tests.drivers import _recursion_run_2026_09_23 as RUN

#: The points of the Cauchy oracle's circle.
NODES = 64


# ------------------------------------------------------------------- fixtures


class Case:
    """One system with what the tests need beside it: the complex, the
    declaration of its coordinates, the mean-field declaration when it is
    self-consistent, and the size of a displacement of its squared lengths
    and of its links that keeps the Cauchy oracle's circle well inside the
    domain of the residual."""

    def __init__(self, spacetime, system, declaration, mean_field=None,
                 length_scale=0.0, link_scale=0.0, moved=None):
        self.spacetime = spacetime
        self.system = system
        self.declaration = declaration
        self.mean_field = mean_field
        self.length_scale = length_scale
        self.link_scale = link_scale
        #: for a fixed-state system with multipliers as variables, a
        #: callable from the multipliers to the same system carrying them
        self.moved = moved
        self.series = SeriesStationarity(system.action, declaration,
                                         mean_field)
        self.size = system.variable_count()
        self.classes = len(J._classes(
            spacetime, list(declaration.edge_classes),
            list(declaration.edge_class_orientations)))
        self.geometric = self.classes * (int(declaration.relax_lengths)
                                         + int(declaration.relax_links))

    def direction(self, seed, multipliers=True):
        """A random complex direction: the squared lengths at the length
        scale, the links at the link scale, and the multipliers at 0.1, or
        left where they are without ``multipliers``."""
        rng = np.random.default_rng(seed)

        def block(count, scale):
            return scale * (rng.normal(size=count)
                            + 1j * rng.normal(size=count))
        parts = []
        if self.declaration.relax_lengths:
            parts.append(block(self.classes, self.length_scale))
        if self.declaration.relax_links:
            parts.append(block(self.classes, self.link_scale))
        parts.append(block(self.size - self.geometric,
                           0.1 if multipliers else 0.0))
        return np.concatenate(parts)


def _villain():
    spacetime, action = J._quarter_turn_tetrahedron(2.0)
    declaration = J._relaxation(relax_links=True)
    return Case(spacetime, cob.HolomorphicRelaxation(action, declaration),
                declaration, link_scale=0.15)


def _regge():
    spacetime, action = J._continued_sheet_fixture()
    declaration = J._relaxation(relax_lengths=True)
    return Case(spacetime, cob.HolomorphicRelaxation(action, declaration),
                declaration, length_scale=0.3)


def _matter():
    spacetime, action = J._sphere_with_matter(3)
    declaration = J._relaxation(relax_lengths=True, relax_links=True)
    return Case(spacetime, cob.HolomorphicRelaxation(action, declaration),
                declaration, length_scale=0.03, link_scale=0.05)


def _three_terms():
    """The sphere with a complex metric and a complex flux carrying the
    primal Regge term on its continued sheets (every hinge of the closed
    complex is interior), the Villain term and the matter term at a fixed
    covariance, both fields relaxed."""
    spacetime = J.sphere3(squared=J._metric, phase=J._flux)
    seed = cob.JointActionDeclaration()
    seed.carrier_degree = 1
    seed.gravitational_weight = 0.0
    seed.holonomy_weight = 0.0
    seed.matter_weight = 0.0
    seed.metric_source = cob.HodgeMetricSource.WhitneyPencil
    projector = cob.JointAction(spacetime, seed).occupation_projector(3, True)
    declared = cob.JointActionDeclaration()
    declared.carrier_degree = 1
    declared.metric_source = cob.HodgeMetricSource.WhitneyPencil
    declared.gravitational_weight = 0.7
    declared.regge_form = cob.ReggeForm.Primal
    declared.regge_hinges = cob.ReggeHinges.Interior
    declared.regge_branch = cob.ReggeBranch.Continued
    declared.holonomy_weight = 1.3
    declared.matter_weight = 1.0
    declared.covariance = list(projector)
    action = cob.JointAction(spacetime, declared)
    declaration = J._relaxation(relax_lengths=True, relax_links=True)
    return Case(spacetime, cob.HolomorphicRelaxation(action, declaration),
                declaration, length_scale=0.03, link_scale=0.05)


def _pinned(form):
    """The recorded host of (0134, 111) with three constraints pinned on the
    fiber at fixed projectors (its power sums of orders one to three, or its
    bands' mean eigenvalues), the multipliers variables."""
    spacetime, _, mean_field, read = J._host((1, 1, 1))

    def moved(multipliers):
        action = (J._pinned_power_sums(spacetime, read, [1, 2, 3],
                                       multipliers, 19.371)
                  if form == "power-sums" else
                  J._pinned_band_means(spacetime, read, multipliers, 19.371))
        return cob.HolomorphicRelaxation(action, declaration)
    declaration = mean_field.geometry
    declaration.relax_multipliers = True
    return Case(spacetime, moved(J.MULTIPLIERS), declaration,
                length_scale=0.3, link_scale=0.05, moved=moved)


#: The squared lengths and the links of a tetrahedron with no symmetry: the
#: lengths off the real axis, the links the monopole fixture's times moduli
#: from 0.9 to 1.15 and small extra phases.
COMPLEX_SQUARED_LENGTHS = [8.0 + 0.3j, 7.6 - 0.2j, 8.4 + 0.1j, 8.1 - 0.4j,
                           7.9 + 0.25j, 8.2 - 0.15j]


def _complex_cell():
    support = bp.monopole_support()
    links = []
    for index, (a, b) in enumerate(tuple(e) for e in support.edges):
        links.append(complex(support.transport(a, b)) * (0.9 + 0.05 * index)
                     * cmath.exp(0.03j * (index - 2)))
    return {"squared_lengths": COMPLEX_SQUARED_LENGTHS, "links": links}


def _host(kind, content, fiber_moments, fiber_pinning, classes):
    """A three-sheeted host with the driver's action and the self-consistent
    system of its mean-field declaration for the content. ``kind`` is
    "declared" (`build_host()` as it stands), "recorded" (the cell (0134) of
    the recursion run of 2026-09-23) or "complex" (`_complex_cell`); with
    ``classes`` the sheets carry one shared base field, and without them
    every edge is its own coordinate. The tolerances are the recorded
    run's, at which each band is one eigenvalue repeated once per sheet."""
    config = bp.default_config(kappas=[1.0], betas=[1.0],
                               selected_contents=[tuple(content)],
                               fiber_moments=fiber_moments,
                               fiber_pinning=fiber_pinning,
                               tolerances=RUN.TOLERANCES)
    if kind == "recorded":
        config["host_cell"] = RUN.HOST_CELLS[J.SECOND_CELL]
    elif kind == "complex":
        config["host_cell"] = _complex_cell()
    spacetime = bp.build_host(config["edge_squared"], config.get("host_cell"))
    action = cob.JointAction(spacetime,
                             bp.action_declaration(spacetime, 1.0, 1.0))
    mean_field = bp.mean_field_declaration(content, config,
                                           spacetime if classes else None)
    mean_field.geometry.held_sectors = []
    mean_field.fiber_moments = bp.fiber_moment_count(
        mean_field, action, fiber_moments, fiber_pinning)
    system = cob.SelfConsistentMeanField(action, mean_field).joint_system()
    return Case(spacetime, system, mean_field.geometry, mean_field,
                length_scale=0.3, link_scale=0.05)


CASES = {
    "villain": _villain,
    "regge": _regge,
    "matter": _matter,
    "three-terms": _three_terms,
    "pinned-power-sums": lambda: _pinned("power-sums"),
    "pinned-band-means": lambda: _pinned("eigenvalues"),
    "declared-classes-003-eigenvalues": lambda: _host(
        "declared", (0, 0, 3), "r", "eigenvalues", True),
    "declared-edges-111-power-sums": lambda: _host(
        "declared", (1, 1, 1), "bands", "power-sums", False),
    "declared-edges-210-unpinned": lambda: _host(
        "declared", (2, 1, 0), "0", "eigenvalues", False),
    "recorded-classes-201-eigenvalues": lambda: _host(
        "recorded", (2, 0, 1), "r", "eigenvalues", True),
    "recorded-classes-111-power-sums": lambda: _host(
        "recorded", (1, 1, 1), "bands", "power-sums", True),
    "complex-classes-003-eigenvalues": lambda: _host(
        "complex", (0, 0, 3), "r", "eigenvalues", True),
    "complex-classes-111-power-sums": lambda: _host(
        "complex", (1, 1, 1), "bands", "power-sums", True),
    "complex-edges-210-eigenvalues": lambda: _host(
        "complex", (2, 1, 0), "r", "eigenvalues", False),
    "complex-classes-300-unpinned": lambda: _host(
        "complex", (3, 0, 0), "0", "eigenvalues", True),
}


@pytest.fixture(scope="module", params=sorted(CASES))
def case(request):
    return CASES[request.param]()


# ------------------------------------------------------------------ helpers


def _line(direction, order):
    """The displacement t d as series of the given order."""
    return [nm.TruncatedSeries([0.0, complex(x)] + [0.0] * (order - 1))
            for x in direction]


def _coefficients(series):
    return np.array([s.coefficients for s in series], dtype=complex)


def _displace(case, snapshot, step):
    """Write x_0 + step on the mesh, in the system's geometric coordinates:
    a squared length moved through the square root on the edge's side, a
    link multiplied by exp(step) on each member's orientation."""
    spacetime, declaration = case.spacetime, case.declaration
    edges = spacetime.getEdgeList().toVector()
    members = J._classes(spacetime, list(declaration.edge_classes),
                         list(declaration.edge_class_orientations))
    J._restore(spacetime, snapshot)
    offset = 0
    if declaration.relax_lengths:
        for index, cls in enumerate(members):
            for edge, _ in cls:
                J._move_length(edges[edge], step[offset + index])
        offset += len(members)
    if declaration.relax_links:
        for index, cls in enumerate(members):
            for edge, orientation in cls:
                J._move_link(edges[edge], orientation * step[offset + index])


def _residual_at(case, snapshot, step):
    """The library's residual at x_0 + step, the multipliers included: the
    geometry is written and restored; a fixed-state system is rebuilt with
    the moved multipliers, and a self-consistent one, whose multipliers
    cannot be written from here, is corrected by its own closed-form
    multiplier columns at the displaced point, the residual being affine in
    the multipliers."""
    step = np.asarray(step, dtype=complex)
    geometric, extra = step[:case.geometric], step[case.geometric:]
    _displace(case, snapshot, geometric)
    try:
        if case.moved is not None:
            moved = np.array(J.MULTIPLIERS, dtype=complex) + extra
            return np.asarray(case.moved(list(moved)).residual(),
                              dtype=complex)
        value = np.asarray(case.system.residual(), dtype=complex)
        if len(extra) and np.any(extra != 0.0):
            jacobian = np.asarray(case.system.jacobian(),
                                  dtype=complex).reshape(case.size, case.size)
            value = value + jacobian[:, case.geometric:] @ extra
        return value
    finally:
        J._restore(case.spacetime, snapshot)


def _cauchy(case, direction, order):
    """The Taylor coefficients c_0 .. c_order of t -> R(x_0 + t d), one row
    per equation, by the Cauchy oracle."""
    snapshot = J._snapshot(case.spacetime)
    nodes = np.exp(2j * np.pi * np.arange(NODES) / NODES)
    values = np.array([_residual_at(case, snapshot, t * direction)
                       for t in nodes])
    return np.array([(values * nodes[:, None] ** (-k)).sum(axis=0) / NODES
                     for k in range(order + 1)]).T


def _minimum_norm(jacobian, rcond=1e-12):
    """The minimum-norm solve of J s = r, singular values at or below
    ``rcond`` times the largest counted as zero."""
    inverse = np.linalg.pinv(jacobian, rcond=rcond)
    return lambda right: inverse @ right


# ------------------------------------------------- the orders zero and one


def test_the_variables_are_the_systems(case):
    assert case.series.variable_count() == case.size


def test_the_order_zero_coefficient_is_the_residual(case):
    """Every coefficient of order zero of the series residual at the zero
    displacement equals the system's residual to rounding. Measured: at most
    1.6e-13 of the residual's scale over the fifteen cases."""
    residual = np.asarray(case.system.residual(), dtype=complex)
    series = _coefficients(case.series.residual(
        [nm.TruncatedSeries([0.0, 0.0])] * case.size))
    scale = max(1.0, np.max(np.abs(residual)))
    assert np.max(np.abs(series[:, 0] - residual)) < 1e-11 * scale
    assert np.max(np.abs(series[:, 1])) == 0.0


def test_the_order_one_coefficient_is_the_jacobian(case):
    """The coefficient of t of R(x_0 + t d) is J d: for a random complex d,
    and for every unit direction of every block (`numerics.jacobian` of the
    series residual is the closed-form Jacobian entry for entry, to
    rounding). Measured: at most 1.2e-14 of the Jacobian's scale."""
    jacobian = np.asarray(case.system.jacobian(),
                          dtype=complex).reshape(case.size, case.size)
    scale = max(1.0, np.max(np.abs(jacobian)))
    direction = case.direction(7)
    series = _coefficients(case.series.residual(_line(direction, 1)))
    assert np.max(np.abs(series[:, 1] - jacobian @ direction)) < 1e-11 * max(
        1.0, np.max(np.abs(jacobian @ direction)))
    columns = np.asarray(nm.jacobian(case.series.residual,
                                     np.zeros(case.size, dtype=complex)))
    assert np.max(np.abs(columns - jacobian)) < 1e-11 * scale


# ----------------------------------------------------- the higher orders


def test_the_coefficients_to_order_ten_match_the_cauchy_oracle(case):
    """Every Taylor coefficient of t -> R(x_0 + t d) to order ten, along a
    random complex direction d (squared lengths moved by about 0.3 on
    lengths of about 8, or 0.03 on lengths of about 1; links by about 0.05
    to 0.15; the multipliers of a fixed-state system by about 0.1, those of
    a self-consistent system left where they are, since the library's
    system does not take them from here), agrees with the Cauchy oracle to
    the oracle's accuracy. Measured: a disagreement of at most 3.5e-14 of
    the coefficients' scale at every order, on coefficients that fall from
    order one to between 1e-5 and 2e-11 of their size at order ten."""
    direction = case.direction(11, multipliers=case.mean_field is None)
    series = _coefficients(case.series.residual(_line(direction, 10)))
    oracle = _cauchy(case, direction, 10)
    scale = max(1.0, np.max(np.abs(oracle)))
    assert np.max(np.abs(series - oracle)) < 1e-11 * scale
    # the comparison is not vacuous: the direction bends the residual
    assert np.max(np.abs(oracle[:, 2])) > 1e-9 * scale


def test_a_displacement_of_the_multipliers_alone_is_linear():
    """The residual is affine in the multipliers: along a displacement that
    moves only them, the coefficients of order two and above vanish
    exactly, and the coefficient of order one is the Jacobian's multiplier
    columns applied to it."""
    case = CASES["complex-classes-111-power-sums"]()
    assert case.size > case.geometric
    direction = np.zeros(case.size, dtype=complex)
    direction[case.geometric:] = [0.4 - 0.3j, -0.2 + 0.1j, 0.05 + 0.6j][
        :case.size - case.geometric]
    series = _coefficients(case.series.residual(_line(direction, 6)))
    assert np.max(np.abs(series[:, 2:])) == 0.0
    jacobian = np.asarray(case.system.jacobian(),
                          dtype=complex).reshape(case.size, case.size)
    assert np.max(np.abs(series[:, 1] - jacobian @ direction)) < 1e-11


def test_a_curved_displacement_is_composed_with_the_residual():
    """A displacement with coefficients beyond the first, s(t) = d t + e t^2:
    the coefficient of t^2 of R(x_0 + s(t)) is J e plus the coefficient of
    t^2 along the line t d (the second-order term of R sees only d)."""
    case = CASES["three-terms"]()
    d, e = case.direction(3), case.direction(4)
    jacobian = np.asarray(case.system.jacobian(),
                          dtype=complex).reshape(case.size, case.size)
    line = _coefficients(case.series.residual(_line(d, 2)))
    curved = _coefficients(case.series.residual(
        [nm.TruncatedSeries([0.0, complex(a), complex(b)])
         for a, b in zip(d, e)]))
    assert np.max(np.abs(curved[:, 1] - line[:, 1])) < 1e-12
    assert np.max(np.abs(curved[:, 2] - line[:, 2] - jacobian @ e)) < 1e-11


def test_the_bands_are_the_ones_the_given_follower_reads():
    """The bands of the self-consistent rebuild are named by the follower's
    read of the band operator at the point. A follower that has followed
    the bands of the point (the reference a system built there holds) gives
    the series of the default follower, coefficient for coefficient."""
    case = CASES["complex-classes-003-eigenvalues"]()
    follower = cob.BandFollower(case.mean_field)
    follower.follow(follower.read(case.system.action.carrier_operator()))
    given = SeriesStationarity(case.system.action, case.declaration,
                               case.mean_field, follower=follower)
    direction = case.direction(2)
    np.testing.assert_array_equal(
        _coefficients(given.residual(_line(direction, 4))),
        _coefficients(case.series.residual(_line(direction, 4))))


# ------------------------------------------------------- the Regge sheets


def test_the_sheets_of_an_action_moved_from_its_start():
    """An action constructed at one geometry and carried to another keeps
    the Regge sheets continued from where it was constructed. With that
    start declared to `SeriesStationarity`, its order-zero coefficient is
    the moved system's residual: on the boundary of the 4-simplex moved by
    about 0.4 in every squared length, ten dihedral angles are off their
    principal sheet and the two agree to rounding."""
    spacetime, action = J._continued_sheet_fixture()
    declaration = J._relaxation(relax_lengths=True)
    edges = spacetime.getEdgeList().toVector()
    start = [complex(edge.getLength()) ** 2 for edge in edges]
    rng = np.random.default_rng(5)
    for edge in edges:
        J._move_length(edge, 0.4 * complex(rng.normal(), rng.normal()))
    system = cob.HolomorphicRelaxation(action, declaration)
    assert system.action.regge_off_principal_angles() == 10
    series = SeriesStationarity(system.action, declaration,
                                regge_start=start)
    value = _coefficients(series.residual(
        [nm.TruncatedSeries([0.0, 0.0])] * 10))[:, 0]
    assert np.max(np.abs(value - np.asarray(system.residual()))) < 1e-13


# ----------------------------------------------------------------- the step


def test_the_step_of_order_one_is_newtons_step(case):
    """`step(1, solve)` is the solution `solve` gives of J s = -R(x_0): with
    the minimum-norm solve it is the minimum-norm Newton step, to
    rounding. Measured: at most 9.5e-13 of the step's size."""
    jacobian = np.asarray(case.system.jacobian(),
                          dtype=complex).reshape(case.size, case.size)
    residual = np.asarray(case.system.residual(), dtype=complex)
    solve = _minimum_norm(jacobian)
    newton = solve(-residual)
    step = case.series.step(1, solve)
    assert np.max(np.abs(step - newton)) < 1e-10 * max(
        1.0, np.max(np.abs(newton)))


def _perturbed_villain(size):
    """The quarter-turn tetrahedron, stationary for the Villain term, with
    every phase moved by a complex number of the given size."""
    spacetime, action = J._quarter_turn_tetrahedron(2.0)
    rng = np.random.default_rng(3)
    for edge in spacetime.getEdgeList().toVector():
        edge.setPhase(complex(edge.getPhase())
                      + size * complex(rng.normal(), rng.normal()))
    declaration = J._relaxation(relax_links=True)
    system = cob.HolomorphicRelaxation(
        cob.JointAction(spacetime, action.declaration), declaration)
    return Case(spacetime, system, declaration, link_scale=0.1)


def test_the_step_approaches_a_root_as_the_order_grows():
    """Where t = 1 lies inside the disc of convergence of the reverted path
    the step of order p lands closer to a root as p grows. On the
    quarter-turn tetrahedron with every phase moved by about 0.01 off its
    stationary value (residual norm 0.206; the Jacobian is singular along
    the gauge directions, so the solve is the minimum-norm one), the
    residual norm at x_0 + s(1) is, measured: 7.0e-9 at order 1, 1.2e-9 at
    order 2, 1.4e-10 at 3, 1.4e-11 at 4, 8.7e-14 at 6 and 7.0e-15, the
    rounding of the residual, from order 8."""
    case = _perturbed_villain(0.01)
    jacobian = np.asarray(case.system.jacobian(),
                          dtype=complex).reshape(case.size, case.size)
    solve = _minimum_norm(jacobian)
    snapshot = J._snapshot(case.spacetime)
    start = np.linalg.norm(np.asarray(case.system.residual()))
    assert start == pytest.approx(0.206, abs=1e-3)
    norms = {}
    for order in (1, 2, 3, 4, 6, 8, 10):
        step = case.series.step(order, solve)
        norms[order] = np.linalg.norm(_residual_at(case, snapshot, step))
    assert norms[1] < 1e-7 * start
    assert norms[2] < norms[1] / 3.0
    assert norms[3] < norms[2] / 3.0
    assert norms[4] < norms[3] / 3.0
    assert norms[6] < norms[4] / 30.0
    assert norms[8] < 1e-13 and norms[10] < 1e-13


def test_the_step_does_not_approach_a_root_outside_the_disc():
    """The same system with the phases moved by about 0.1 (residual norm
    2.06): the coefficients of the path fall slowly, and the residual norm
    at x_0 + s(1) falls from 8.7e-7 at order 1 to 2.5e-8 at order 10, a
    factor of about 0.7 per order where the nearer start gave a tenth. The
    order of the step buys what the path's convergence at t = 1 allows."""
    case = _perturbed_villain(0.1)
    jacobian = np.asarray(case.system.jacobian(),
                          dtype=complex).reshape(case.size, case.size)
    solve = _minimum_norm(jacobian)
    snapshot = J._snapshot(case.spacetime)
    first = np.linalg.norm(_residual_at(case, snapshot,
                                        case.series.step(1, solve)))
    tenth = np.linalg.norm(_residual_at(case, snapshot,
                                        case.series.step(10, solve)))
    assert first == pytest.approx(8.7e-7, rel=0.05)
    assert tenth == pytest.approx(2.5e-8, rel=0.05)


@pytest.mark.parametrize("name, larger, smaller", [
    ("recorded-classes-201-eigenvalues", 0.3, 0.1),
    ("complex-classes-003-eigenvalues", 0.05, 0.02),
    ("three-terms", 0.1, 0.02),
])
def test_the_reverted_path_solves_the_equations_to_its_order(name, larger,
                                                             smaller):
    """The reversion's defining property, measured with the library's
    residual: the defect R(x_0 + s(tau)) - (1 - tau) R(x_0) of the path of
    order p is of the neglected order, tau^(p+1). At two parameters inside
    the path's disc of convergence the two defects are therefore in the
    ratio (larger / smaller)^(p+1), for p = 1, 2, 3, 4 and 6: measured
    between 0.90 and 1.02 of that ratio in the three cases (the
    self-consistent recorded host of (0134, 201) with its bands' eigenvalues
    pinned, the self-consistent host with complex lengths and links off the
    unit circle for (0, 0, 3), and the sphere with the three terms), the
    multipliers moved with the geometry. s(tau) is the step of order p of
    the same equations with the residual shrunk to tau R(x_0). Measured
    defects at the smaller parameter, orders 1, 2, 3, 4, 6: 1.9e-2, 1.7e-3,
    1.5e-4, 1.4e-5, 1.1e-7 on the first case; 1.7e-6, 6.4e-8, 2.7e-9,
    1.3e-10, 5.3e-13 on the second; 8.1e-3, 1.6e-4, 3.4e-6, 1.1e-7, 2.5e-10
    on the third."""
    case = CASES[name]()
    jacobian = np.asarray(case.system.jacobian(),
                          dtype=complex).reshape(case.size, case.size)
    solve = _minimum_norm(jacobian, 1e-9)
    snapshot = J._snapshot(case.spacetime)
    residual = np.asarray(case.system.residual(), dtype=complex)

    def defect(path, parameter):
        moved = _residual_at(case, snapshot,
                             np.asarray(path.evaluate(parameter)))
        return np.linalg.norm(moved - (1.0 - parameter) * residual)
    previous = None
    for order in (1, 2, 3, 4, 6):
        path = case.series.reversion(order, solve)
        assert len(path.coefficients) == order + 1
        at_larger, at_smaller = defect(path, larger), defect(path, smaller)
        expected = (larger / smaller) ** (order + 1)
        assert 0.8 * expected < at_larger / at_smaller < 1.2 * expected
        if previous is not None:
            assert at_smaller < previous
        previous = at_smaller


# ------------------------------------------------------------ what is refused


def test_a_displacement_must_fit_the_system():
    case = CASES["villain"]()
    with pytest.raises(ValueError, match="5 series for 6 variables"):
        case.series.residual([nm.TruncatedSeries([0.0, 1.0])] * 5)
    with pytest.raises(ValueError, match="differ in order"):
        case.series.residual([nm.TruncatedSeries([0.0, 1.0])] * 5
                             + [nm.TruncatedSeries([0.0, 1.0, 0.0])])
    with pytest.raises(ValueError, match="nonzero order-0 coefficient"):
        case.series.residual([nm.TruncatedSeries([0.1, 1.0])] * 6)


def test_the_forms_that_are_not_propagated_say_so():
    """The dual Regge form and a metric source other than the Whitney pencil
    have no series evaluation here; each is named."""
    spacetime, action = J._continued_sheet_fixture()
    declared = action.declaration
    declared.regge_form = cob.ReggeForm.Dual
    with pytest.raises(NotImplementedError, match="primal form"):
        SeriesStationarity(cob.JointAction(spacetime, declared),
                           J._relaxation(relax_lengths=True))
    spacetime, action = J._sphere_with_matter(3)
    declared = action.declaration
    declared.metric_source = cob.HodgeMetricSource.DiagonalWeights
    with pytest.raises(NotImplementedError, match="Whitney pencil"):
        SeriesStationarity(cob.JointAction(spacetime, declared),
                           J._relaxation(relax_lengths=True))
