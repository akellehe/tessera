# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The analytic Jacobian of the holomorphic stationarity system, block by
block, against a central-difference oracle written here (#1324, item B1 of
epic #1293).

Terms used below:

* the *residual* is the vector `HolomorphicRelaxation.residual` of the
  equations in scope, the length stationarity dS/dz_e, the link stationarity
  U_e dS/dU_e in the Maurer-Cartan coordinate on the stored orientation, and
  the constraint residuals c_j - c_j*, each summed over a declared edge class
  where classes are declared; the *Jacobian* is `HolomorphicRelaxation.jacobian`
  of that residual in the relaxed coordinates (squared lengths per class,
  Maurer-Cartan increments per class, multipliers);
* the *oracle* is the central difference (F(x + h) - F(x - h)) / (2 h) of the
  residual at a real step h along one coordinate: a squared length moved to
  z + h through the square root on the edge's side, a link moved to U e^{h}
  through the stored phase phi -> phi - i h, every edge of a class moved
  together (each link on the class's orientation), a multiplier moved to
  xi + h on a copy of the action. It is a test device, not a library mode.
  The residual is holomorphic, so the oracle's truncation is h^2 F'''/6 and
  its rounding eps |F| / h; each test states its step and the agreement it
  asserts, which the truncation at that step allows;
* the *self-consistent* Jacobian is that of the joint system of
  `SelfConsistentMeanField.joint_system`, in which the covariance Gamma and
  the pinned fiber's projectors are rebuilt at every point from the bands
  followed from the point the system is built at.

The blocks and the fixtures they are checked on:

* the Regge block on the boundary of the 4-simplex on the continued sheet
  (fourteen dihedral angles off their principal sheet);
* the Villain block on the quarter-turn tetrahedron, where every face
  holonomy is a quarter turn;
* the matter block on the sphere fixture with a complex metric and a complex
  flux, and on the declared three-sheeted host with its shared base field;
* the constraint rows and multiplier columns on the declared host with the
  fiber's power sums pinned and with its bands' eigenvalues pinned;
* the self-consistent columns on the declared host with the pinned fiber of
  the driver's own mean-field declaration, under both pinning forms and with
  nothing pinned;
* the first-order perturbation of a band's Riesz projector against a
  difference of the projector itself, and the refusal by name of a band that
  is not isolated.
"""
import cmath
import itertools
import math
import os
import sys

import numpy as np
import pytest

import tessera as T
from tessera import cobordism as cob
from tessera.drivers import baryon_poles as bp
from tessera.drivers import recursion as R

from tests.drivers import _recursion_run_2026_09_23 as RUN

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _joint_action_hosts import sphere3, tetrahedron  # noqa: E402

SECOND_CELL = (0, 1, 3, 4)

#: The boundary of the 4-simplex as a closed 3-complex.
BOUNDARY_OF_FOUR_SIMPLEX = [list(c) for c in itertools.combinations(range(5), 4)]


# ------------------------------------------------------------------ the oracle


def _snapshot(spacetime):
    return [(complex(e.getLength()), complex(e.getPhase()))
            for e in spacetime.getEdgeList().toVector()]


def _restore(spacetime, snapshot):
    for edge, (length, phase) in zip(spacetime.getEdgeList().toVector(),
                                     snapshot):
        edge.setLength(length)
        edge.setPhase(phase)


def _move_length(edge, step):
    """z -> z + step, written through the square root on the edge's side."""
    length = complex(edge.getLength())
    root = cmath.sqrt(length * length + step)
    if (root.conjugate() * length).real < 0.0:
        root = -root
    edge.setLength(root)


def _move_link(edge, step):
    """U -> U e^{step}: on the stored phase, phi -> phi - i step."""
    edge.setPhase(complex(edge.getPhase()) - 1j * step)


def _classes(spacetime, edge_classes, orientations):
    """The members (edge index, orientation) of every class; every edge its
    own class when none is declared."""
    edges = len(spacetime.getEdgeList().toVector())
    if not edge_classes:
        return [[(e, 1)] for e in range(edges)]
    members = [[] for _ in range(max(edge_classes) + 1)]
    for edge, (c, o) in enumerate(zip(edge_classes, orientations)):
        members[c].append((edge, int(o) if orientations else 1))
    return members


def _oracle(spacetime, evaluate, declaration, length_step, link_step):
    """The geometric columns of the Jacobian of ``evaluate()`` by central
    differences: one column per relaxed class coordinate in the relaxation's
    order (lengths, then links), every member edge of the class moved
    together, each link member on its orientation, the geometry restored
    exactly after every evaluation."""
    edges = spacetime.getEdgeList().toVector()
    snapshot = _snapshot(spacetime)
    members = _classes(spacetime, list(declaration.edge_classes),
                       list(declaration.edge_class_orientations))

    def displaced(index, link, sign):
        _restore(spacetime, snapshot)
        for edge, orientation in members[index]:
            if link:
                _move_link(edges[edge], sign * orientation * link_step)
            else:
                _move_length(edges[edge], sign * length_step)
        value = np.asarray(evaluate(), dtype=complex)
        _restore(spacetime, snapshot)
        return value

    columns = []
    for link in ((False,) if declaration.relax_lengths else ()) + \
            ((True,) if declaration.relax_links else ()):
        step = link_step if link else length_step
        for index in range(len(members)):
            columns.append((displaced(index, link, 1.0)
                            - displaced(index, link, -1.0)) / (2.0 * step))
    return np.array(columns).T


def _square(flat):
    flat = np.asarray(flat, dtype=complex)
    order = int(round(math.sqrt(flat.size)))
    return flat.reshape(order, order)


def _agreement(analytic, oracle):
    """max |analytic - oracle| over the scale of the analytic block."""
    scale = max(1.0, np.max(np.abs(analytic)))
    return np.max(np.abs(analytic - oracle)) / scale


# ------------------------------------------------------------------- fixtures


def _relaxation(**overrides):
    declaration = cob.HolomorphicRelaxationDeclaration()
    declaration.relax_lengths = False
    declaration.relax_links = False
    declaration.relax_multipliers = False
    for name, value in overrides.items():
        setattr(declaration, name, value)
    return declaration


def _metric(index):
    return complex(1.0 + 0.037 * (index % 5), 0.021 * (1 + index % 4))


def _flux(index):
    return complex(0.23 * ((index % 5) - 2), 0.09 * ((index % 3) - 1))


def _continued_sheet_fixture():
    """The boundary of the 4-simplex at the start of the continued-sheet
    Newton solve of `test_regge_continued_sheet_python`: squared lengths off
    the regular ones by five per cent with imaginary parts of 1e-9, the
    primal Regge term on the continued sheets, fourteen dihedral angles off
    their principal sheet."""
    edges = list(itertools.combinations(range(5), 2))
    z_star = 8.0
    start = {e: z_star * (1.0 + 0.05 * ((3 * i) % 4 - 1.5))
             + 1e-9j * ((7 * i) % 5 - 1.5) for i, e in enumerate(edges)}
    spacetime = T.Spacetime.fromVertexTuples(3, BOUNDARY_OF_FOUR_SIMPLEX,
                                             1.0, 0.0)
    for edge in spacetime.getEdgeList().toVector():
        a, b = sorted((int(edge.getSource().getId()),
                       int(edge.getTarget().getId())))
        edge.setLength(cmath.sqrt(complex(start[(a, b)])))
    declaration = cob.JointActionDeclaration()
    declaration.carrier_degree = 1
    declaration.gravitational_weight = 1.0
    declaration.regge_form = cob.ReggeForm.Primal
    declaration.regge_hinges = cob.ReggeHinges.Interior
    declaration.regge_branch = cob.ReggeBranch.Continued
    declaration.holonomy_weight = 0.0
    declaration.matter_weight = 0.0
    return spacetime, cob.JointAction(spacetime, declaration)


def _quarter_turn_tetrahedron(beta):
    """One tetrahedron carrying the unit-monopole connection of the library's
    symmetric fixture, so that every face holonomy is a quarter turn, with
    the Villain term at coupling ``beta`` and nothing else."""
    spacetime = tetrahedron()
    support = bp.monopole_support()
    for edge in spacetime.getEdgeList().toVector():
        source = int(edge.getSource().getId())
        target = int(edge.getTarget().getId())
        edge.setPhase(complex(cmath.phase(support.transport(source, target))))
    declaration = cob.JointActionDeclaration()
    declaration.carrier_degree = 1
    declaration.gravitational_weight = 0.0
    declaration.holonomy_weight = beta
    declaration.matter_weight = 0.0
    return spacetime, cob.JointAction(spacetime, declaration)


def _sphere_with_matter(occupied):
    """The sphere with a complex metric and a complex flux, the matter term
    at weight one on the projector onto the ``occupied`` lowest modes of the
    covariant operator, held fixed."""
    spacetime = sphere3(squared=_metric, phase=_flux)
    seed = cob.JointActionDeclaration()
    seed.carrier_degree = 1
    seed.gravitational_weight = 0.0
    seed.holonomy_weight = 0.0
    seed.matter_weight = 0.0
    seed.metric_source = cob.HodgeMetricSource.WhitneyPencil
    projector = cob.JointAction(spacetime, seed).occupation_projector(
        occupied, True)
    declaration = cob.JointActionDeclaration()
    declaration.carrier_degree = 1
    declaration.gravitational_weight = 0.0
    declaration.holonomy_weight = 0.0
    declaration.matter_weight = 1.0
    declaration.metric_source = cob.HodgeMetricSource.WhitneyPencil
    declaration.covariance = list(projector)
    return spacetime, cob.JointAction(spacetime, declaration)


def _config(content, fiber_moments, fiber_pinning="eigenvalues"):
    config = bp.default_config(kappas=[1.0], betas=[1.0],
                               selected_contents=[tuple(content)],
                               fiber_moments=fiber_moments,
                               fiber_pinning=fiber_pinning,
                               tolerances=RUN.TOLERANCES)
    config["host_cell"] = RUN.HOST_CELLS[SECOND_CELL]
    config["held_sectors"] = R.held_sectors([[0, 1, 2, 3]], [1], 4)
    return config


def _host(content, fiber_moments="0", fiber_pinning="eigenvalues"):
    """The declared host of one read of the recursion run, the driver's
    action on it (Regge on the interior hinges, of which the host has none;
    the Villain term at beta = 1; the matter term), the driver's mean-field
    declaration for the content with the sheets' fields shared, and the band
    read at the host."""
    config = _config(content, fiber_moments, fiber_pinning)
    spacetime = bp.build_host(config["edge_squared"], config["host_cell"])
    action = cob.JointAction(spacetime, bp.action_declaration(spacetime, 1.0,
                                                              1.0))
    mean_field = bp.mean_field_declaration(content, config, spacetime)
    mean_field.geometry.held_sectors = []
    read = cob.BandFollower(mean_field).read(action.carrier_operator())
    return spacetime, action, mean_field, read


#: The step in a squared length of the host (z about 8) and in a link.
HOST_LENGTH_STEP = 1e-4
HOST_LINK_STEP = 1e-4


# ------------------------------------------------------------ the Regge block


def test_the_regge_block_on_the_continued_sheet():
    """On the boundary of the 4-simplex with fourteen dihedral angles off
    their principal sheet, the length block of the Jacobian is the Regge
    Hessian on the continued sheets: it equals ``regge_hessian`` entry for
    entry, it is symmetric to rounding, and it agrees with the oracle at the
    step 1e-4 (the squared lengths are about 8; truncation ``h^2 F'''/6``,
    about 1e-9 of the block's scale) to 1e-7 of the block's scale."""
    spacetime, action = _continued_sheet_fixture()
    assert action.regge_off_principal_angles() == 14
    declaration = _relaxation(relax_lengths=True)
    relaxation = cob.HolomorphicRelaxation(action, declaration)
    size = relaxation.variable_count()
    assert size == 10
    jacobian = _square(relaxation.jacobian())
    np.testing.assert_array_equal(jacobian, _square(action.regge_hessian()))
    assert np.max(np.abs(jacobian - jacobian.T)) < 1e-12 * np.max(
        np.abs(jacobian))
    oracle = _oracle(spacetime, relaxation.residual, declaration, 1e-4, 1e-4)
    assert _agreement(jacobian, oracle) < 1e-7


def test_the_dual_regge_block_matches_the_oracle():
    """The dual (Sorkin) form on the sphere with a real, non-uniform metric:
    the length block is the exact Hessian of the dual Regge action, and it
    agrees with the oracle at the step 1e-5 (truncation about 1e-11 of the
    block's scale) to 1e-8 of that scale."""
    spacetime = sphere3(squared=lambda index: 1.0 + 0.05 * (index % 4))
    declaration = cob.JointActionDeclaration()
    declaration.carrier_degree = 1
    declaration.gravitational_weight = 0.7
    declaration.regge_form = cob.ReggeForm.Dual
    declaration.holonomy_weight = 0.0
    declaration.matter_weight = 0.0
    action = cob.JointAction(spacetime, declaration)
    solve = _relaxation(relax_lengths=True)
    relaxation = cob.HolomorphicRelaxation(action, solve)
    jacobian = _square(relaxation.jacobian())
    oracle = _oracle(spacetime, relaxation.residual, solve, 1e-5, 1e-5)
    assert _agreement(jacobian, oracle) < 1e-8


# ---------------------------------------------------------- the Villain block


@pytest.mark.parametrize("beta", [0.5, 2.0])
def test_the_villain_block_on_the_quarter_turn_tetrahedron(beta):
    """On the tetrahedron whose four face holonomies are quarter turns the
    link block of the Jacobian is the Villain Hessian in the Maurer-Cartan
    increments: it equals ``holonomy_hessian`` entry for entry, and it agrees
    with the oracle at the step 1e-4 in the increment (truncation
    ``h^2 F'''/6``, about 1e-9 of the block's scale) to 1e-7 of the block's
    scale. The curvature at a quarter turn does not vanish, so the block is
    not the zero the Wilson form would give there."""
    spacetime, action = _quarter_turn_tetrahedron(beta)
    holonomies = np.asarray(action.face_holonomies())
    assert np.max(np.abs(holonomies.real)) < 1e-12
    assert np.max(np.abs(np.abs(holonomies.imag) - 1.0)) < 1e-12
    declaration = _relaxation(relax_links=True)
    relaxation = cob.HolomorphicRelaxation(action, declaration)
    assert relaxation.variable_count() == 6
    jacobian = _square(relaxation.jacobian())
    np.testing.assert_array_equal(jacobian,
                                  _square(action.holonomy_hessian()))
    assert np.max(np.abs(jacobian)) > 0.1 * beta
    oracle = _oracle(spacetime, relaxation.residual, declaration, 1e-4, 1e-4)
    assert _agreement(jacobian, oracle) < 1e-7


# ----------------------------------------------------------- the matter block


def test_the_matter_block_on_the_sphere():
    """On the sphere with a complex metric and a complex flux, the matter
    term at a fixed covariance (the projector onto the three lowest modes):
    the whole geometric Jacobian, lengths and links, is the contraction of
    the operator's second derivatives against Gamma. It is symmetric to
    rounding (the Hessian of a scalar), and it agrees with the oracle at the
    step 1e-4 in both coordinates (truncation ``h^2 F'''/6``, about 1e-9 of
    the block's scale) to 1e-7 of that scale."""
    spacetime, action = _sphere_with_matter(3)
    declaration = _relaxation(relax_lengths=True, relax_links=True)
    relaxation = cob.HolomorphicRelaxation(action, declaration)
    assert relaxation.variable_count() == 20
    jacobian = _square(relaxation.jacobian())
    assert np.max(np.abs(jacobian - jacobian.T)) < 1e-11 * np.max(
        np.abs(jacobian))
    oracle = _oracle(spacetime, relaxation.residual, declaration, 1e-4, 1e-4)
    assert _agreement(jacobian[:10, :10], oracle[:10, :10]) < 1e-7
    assert _agreement(jacobian[10:, 10:], oracle[10:, 10:]) < 1e-7
    assert _agreement(jacobian[:10, 10:], oracle[:10, 10:]) < 1e-7
    assert _agreement(jacobian[10:, :10], oracle[10:, :10]) < 1e-7


def test_the_action_hessian_is_the_jacobian_without_classes():
    """Without declared classes the Jacobian's geometric blocks are
    ``action_hessian`` entry for entry: the assembly reduces nothing."""
    _, action = _sphere_with_matter(2)
    relaxation = cob.HolomorphicRelaxation(
        action, _relaxation(relax_lengths=True, relax_links=True))
    np.testing.assert_array_equal(_square(relaxation.jacobian()),
                                  _square(action.action_hessian()))


def test_the_matter_block_on_the_declared_host():
    """On the declared host of (0134, 111) with its covariance fixed at the
    band filling read there, under the driver's action (the Villain term at
    beta = 1 and the matter term; the host has no interior hinge) and with
    the sheets' fields shared: the twelve-by-twelve geometric Jacobian is the
    Hessian of the action reduced onto the six shared squared lengths and six
    shared links, symmetric to rounding, and it agrees with the oracle at the
    steps 1e-4 (truncation about 1e-9 of the block's scale) to 1e-7 of that
    scale."""
    spacetime, action, mean_field, read = _host((1, 1, 1))
    action.set_covariance(read.covariance)
    declaration = mean_field.geometry
    relaxation = cob.HolomorphicRelaxation(action, declaration)
    assert relaxation.variable_count() == 12
    jacobian = _square(relaxation.jacobian())
    assert np.max(np.abs(jacobian - jacobian.T)) < 1e-11 * np.max(
        np.abs(jacobian))
    oracle = _oracle(spacetime, relaxation.residual, declaration,
                     HOST_LENGTH_STEP, HOST_LINK_STEP)
    assert _agreement(jacobian, oracle) < 1e-7


# ------------------------------------------- the constraint rows and columns


def _pinned_power_sums(spacetime, read, orders, multipliers, scale):
    """The driver's action with the power sums of the given orders pinned on
    the fiber of ``read`` in the unit ``scale`` (targets zero), the
    multipliers as given."""
    declaration = bp.action_declaration(spacetime, 1.0, 1.0)
    declaration.covariance = list(read.covariance)
    fiber = sum(np.asarray(band.projector) for band in read.bands)
    declaration.moment_projector = list(fiber)
    declaration.moment_scale = scale
    constraints = []
    for order, multiplier in zip(orders, multipliers):
        constraint = cob.SpectralMomentConstraint()
        constraint.order = order
        constraint.multiplier = multiplier
        constraints.append(constraint)
    declaration.moment_constraints = constraints
    return cob.JointAction(spacetime, declaration)


def _pinned_band_means(spacetime, read, multipliers, scale):
    """The driver's action with the mean eigenvalue of every occupied band
    pinned (`SpectralConstraintForm.BandMean`) in the unit ``scale``."""
    declaration = bp.action_declaration(spacetime, 1.0, 1.0)
    declaration.covariance = list(read.covariance)
    declaration.moment_band_projectors = [list(band.projector)
                                         for band in read.bands]
    declaration.moment_scale = scale
    constraints = []
    for band, multiplier in enumerate(multipliers):
        constraint = cob.SpectralMomentConstraint()
        constraint.form = cob.SpectralConstraintForm.BandMean
        constraint.band = band
        constraint.multiplier = multiplier
        constraints.append(constraint)
    declaration.moment_constraints = constraints
    return cob.JointAction(spacetime, declaration)


MULTIPLIERS = [0.3 - 0.2j, -0.1 + 0.05j, 0.07 + 0.11j]


def _class_reduced(gradient, declaration, edges):
    """A per-edge gradient (lengths then links) summed over the declared
    classes, each link entry on its class orientation."""
    classes = list(declaration.edge_classes)
    orientations = list(declaration.edge_class_orientations)
    count = max(classes) + 1
    reduced = np.zeros(2 * count, dtype=complex)
    for edge in range(edges):
        c, o = classes[edge], orientations[edge]
        reduced[c] += gradient[edge]
        reduced[count + c] += o * gradient[edges + edge]
    return reduced


@pytest.mark.parametrize("form", ["power-sums", "eigenvalues"])
def test_the_constraint_rows_and_columns_on_the_declared_host(form):
    """On the declared host of (0134, 111) with three constraints pinned on
    the fiber at fixed projectors, the fiber's power sums of orders one to
    three in the unit 19.371 or its three bands' mean eigenvalues, with
    nonzero multipliers: the constraint rows and the multiplier columns are
    the analytic ``moment_gradient`` reduced onto the shared coordinates,
    entry for entry; the rows agree with the oracle of the constraint
    residuals at the steps 1e-4 (truncation about 1e-9 of the rows' scale)
    to 1e-7 of that scale; the columns agree with a difference of the
    residual in the multipliers at the step 1e-3, exactly to rounding since
    the residual is linear in them; and the geometric block, which carries
    the constraints' second derivatives through the operator, agrees with
    the oracle to 1e-7 of its scale."""
    spacetime, action, mean_field, read = _host((1, 1, 1))
    if form == "power-sums":
        pinned = _pinned_power_sums(spacetime, read, [1, 2, 3], MULTIPLIERS,
                                    19.371)
    else:
        pinned = _pinned_band_means(spacetime, read, MULTIPLIERS, 19.371)
    declaration = mean_field.geometry
    declaration.relax_multipliers = True
    relaxation = cob.HolomorphicRelaxation(pinned, declaration)
    size = relaxation.variable_count()
    assert size == 12 + 3
    jacobian = _square(relaxation.jacobian())
    edges = pinned.edge_count()
    for j in range(3):
        reduced = _class_reduced(np.asarray(pinned.moment_gradient(j)),
                                 declaration, edges)
        np.testing.assert_array_equal(jacobian[12 + j, :12], reduced)
        np.testing.assert_array_equal(jacobian[:12, 12 + j], reduced)
    assert np.max(np.abs(jacobian[12:, 12:])) == 0.0
    oracle = _oracle(spacetime, relaxation.residual, declaration,
                     HOST_LENGTH_STEP, HOST_LINK_STEP)
    assert _agreement(jacobian[12:, :12], oracle[12:, :12]) < 1e-7
    assert _agreement(jacobian[:12, :12], oracle[:12, :12]) < 1e-7
    # the multiplier columns, by a difference in the multipliers
    step = 1e-3
    for j in range(3):
        columns = []
        for sign in (1.0, -1.0):
            moved = np.array(MULTIPLIERS, dtype=complex)
            moved[j] += sign * step
            copy = _pinned_power_sums(spacetime, read, [1, 2, 3], moved,
                                      19.371) if form == "power-sums" else \
                _pinned_band_means(spacetime, read, moved, 19.371)
            columns.append(np.asarray(
                cob.HolomorphicRelaxation(copy, declaration).residual()))
        difference = (columns[0] - columns[1]) / (2.0 * step)
        assert _agreement(jacobian[:, 12 + j], difference) < 1e-9


# ------------------------------------------------ the self-consistent columns


@pytest.mark.parametrize("content, fiber_moments, fiber_pinning", [
    ((1, 1, 1), "0", "eigenvalues"),
    ((1, 1, 1), "r", "eigenvalues"),
    ((2, 1, 0), "r", "eigenvalues"),
    ((1, 1, 1), "bands", "power-sums"),
])
def test_the_self_consistent_columns_on_the_declared_host(
        content, fiber_moments, fiber_pinning):
    """The joint system of the driver's mean-field declaration on the
    declared host of (0134, content): the covariance is the band filling of
    the content, rebuilt at every point from the bands followed from the
    host, and the fiber is pinned as declared (nothing; the eigenvalue of
    every occupied band; the power sums of as many orders as there are
    occupied bands). Every geometric column of the analytic Jacobian, which
    carries the derivative of the rebuilt state through the perturbation of
    each band's Riesz projector, agrees with the oracle of the
    self-consistent residual, the constraint rows included, as a central
    difference agrees with a derivative: the disagreement is the oracle's
    truncation, which falls with the square of the step. Measured, relative
    to the block's scale: 3.5e-8, 3.2e-8, 2.1e-8 and 1.4e-7 at the step 1e-4
    in the four cases, and a hundredth of each at the step 1e-5, where the
    rounding of the difference (about 1e-10) begins to show. The
    multiplier columns are the analytic gradients reduced onto the shared
    coordinates."""
    spacetime, action, mean_field, _ = _host(content, fiber_moments,
                                             fiber_pinning)
    mean_field.fiber_moments = bp.fiber_moment_count(
        mean_field, action, fiber_moments, fiber_pinning)
    system = cob.SelfConsistentMeanField(action, mean_field).joint_system()
    declaration = mean_field.geometry
    pinned = mean_field.fiber_moments
    size = system.variable_count()
    assert size == 12 + pinned
    jacobian = _square(system.jacobian())
    coarse = _agreement(jacobian[:, :12], _oracle(
        spacetime, system.residual, declaration, 1e-4, 1e-4))
    fine = _agreement(jacobian[:, :12], _oracle(
        spacetime, system.residual, declaration, 1e-5, 1e-5))
    assert coarse < 2e-7
    assert fine < 2e-9
    assert fine < coarse / 50.0
    edges = action.edge_count()
    for j in range(pinned):
        reduced = _class_reduced(
            np.asarray(system.action.moment_gradient(j)), declaration, edges)
        np.testing.assert_array_equal(jacobian[:12, 12 + j], reduced)
        np.testing.assert_array_equal(jacobian[12 + j, :12], reduced)


def test_the_self_consistent_part_is_not_the_fixed_state_hessian():
    """On (0134, 111) with nothing pinned, the self-consistent Jacobian
    differs from the Hessian at the covariance held fixed: the derivative of
    the rebuilt covariance along the coordinates is not zero."""
    spacetime, action, mean_field, read = _host((1, 1, 1))
    system = cob.SelfConsistentMeanField(action, mean_field).joint_system()
    action.set_covariance(read.covariance)
    fixed = cob.HolomorphicRelaxation(action, mean_field.geometry)
    self_consistent = _square(system.jacobian())
    held = _square(fixed.jacobian())
    assert np.max(np.abs(self_consistent - held)) > 1e-3 * np.max(
        np.abs(held))


# ------------------------------------------ the Riesz projector perturbation


def test_the_riesz_projector_derivative_matches_a_difference_of_the_projector(
):
    """On the covariant operator of the sphere with a complex metric and a
    complex flux (non-normal), the first-order perturbation of the Riesz
    projector of the band of its three lowest eigenvalues under a
    deterministic complex variation agrees with a central difference of the
    projector itself, each side's projector formed from its own
    eigendecomposition with the band's modes matched by eigenvalue
    continuity, at the step 1e-5 (truncation ``h^2 P'''/6``, about 1e-11 of
    the projector's scale) to 1e-8 of that scale."""
    _, action = _sphere_with_matter(1)
    h = _square(action.carrier_operator())
    n = h.shape[0]
    rng = np.random.default_rng(7)
    dh = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    values, vectors = np.linalg.eig(h)
    order = np.argsort(values.real)
    modes = [int(m) for m in order[:3]]
    left = np.linalg.inv(vectors)
    analytic = _square(cob.riesz_projector_derivative(
        list(values), list(vectors.reshape(-1)), list(left.reshape(-1)),
        modes, list(dh.reshape(-1))))

    def projector(matrix):
        w, v = np.linalg.eig(matrix)
        chosen = []
        for m in modes:
            distances = np.abs(w - values[m])
            distances[chosen] = np.inf
            chosen.append(int(np.argmin(distances)))
        inverse = np.linalg.inv(v)
        return sum(np.outer(v[:, c], inverse[c, :]) for c in chosen)

    step = 1e-5
    oracle = (projector(h + step * dh) - projector(h - step * dh)) / (
        2.0 * step)
    assert _agreement(analytic, oracle) < 1e-8
    # the perturbation is off-diagonal between the band and its complement
    base = sum(np.outer(vectors[:, m], left[m, :]) for m in modes)
    assert np.max(np.abs(base @ analytic @ base)) < 1e-9 * np.max(
        np.abs(analytic))


def test_a_band_close_to_another_eigenvalue_is_perturbed_as_it_stands():
    """Two eigenvalues 1e-9 apart, one in the band: the perturbation of the
    band's projector divides by their difference and is returned as it is,
    the off-diagonal entries G_kj / (lambda_k - lambda_j) = 1 / (-1e-9) in
    both orders. Nothing is refused for being close."""
    variation = np.asarray(cob.riesz_projector_derivative(
        [1.0 + 0j, 1.0 + 1e-9 + 0j], [1, 0, 0, 1], [1, 0, 0, 1], [0],
        [0, 1, 1, 0])).reshape(2, 2)
    np.testing.assert_allclose(variation, [[0, -1e9], [-1e9, 0]], rtol=1e-6)


def test_an_exactly_equal_eigenvalue_across_the_band_has_no_quotient():
    """The one case with no value: an eigenvalue inside the band equal,
    exactly, to one outside it, where the quotient divides by zero."""
    with pytest.raises(ValueError, match="has no value"):
        cob.riesz_projector_derivative(
            [1.0 + 0j, 1.0 + 0j], [1, 0, 0, 1], [1, 0, 0, 1], [0],
            [0, 1, 1, 0])
