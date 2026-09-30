# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The geometric terms of the recursion driver's joint action held to closed
forms: the primal Regge term on known meshes and the Villain holonomy term.

`recursion.relax_level` and `baryon_poles.relax_content` both relax the joint
action S = w_R S_Regge(primal) + w_S (1/2)||l - l0||^2 + S_hol + w_M tr(Gamma
h) (whitepaper v17 §7, `JointAction`). This file holds the first and the
third term to values fixed independently of the library:

* the primal Regge sum sum_h |h| (2 pi - sum theta_h) over the hinges the
  declared rule selects (`ReggeHinges.All`: every hinge, with the library's
  2 pi convention; `ReggeHinges.Interior`: hinges whose link closes), on the
  closed fan of three unit regular tetrahedra around an edge, on a single
  regular 4-simplex and on a single regular 5-simplex, whose dihedral angles
  are arccos(1/d);
* its exact gradient in the squared lengths against a central difference;
* the Villain weight W(F) = sum_m exp(-m^2 / (2 beta)) F^m of WP v17 lines
  171-173 (`VillainCharacter`): the matched weight beta / <m^2>_beta, the
  curvature at a quarter turn F = i, beta_V (<m^2>_i - <m>_i^2), equal to
  0.43091 at beta = 1/2, 0.99796 at 1, 1.2998 at 1.3, 1.9999996 at 2 and
  5.0000 at 5, the symmetry W(1/F) = W(F), the zeros
  F = -exp(+-(2n - 1) / (2 beta)) of Jacobi's triple product, and the
  truncation's tail bounds;
* the Wilson form's zero curvature at a quarter turn, and the gauge
  invariance of the holonomy term.
"""
import cmath
import math

import numpy as np
import pytest

import tessera as T
from tessera import cobordism as cob
from tessera.drivers import baryon_poles as bp

THETA3 = math.acos(1.0 / 3.0)
#: Three regular tetrahedra around the edge (0, 1), closing up: ten edges.
CLOSED_FAN = [[0, 1, 2, 3], [0, 1, 3, 4], [0, 1, 2, 4]]


def _mesh(dimension, cells, squared=1.0):
    spacetime = T.Spacetime.fromVertexTuples(dimension, cells, 1.0, 0.0)
    for edge in spacetime.getEdgeList().toVector():
        edge.setLength(cmath.sqrt(complex(squared)))
    return spacetime


def _geometric(spacetime, hinges=cob.ReggeHinges.All, regge=1.0,
               stiffness=0.0, beta=0.0, form=cob.HolonomyForm.Villain,
               reference=None):
    declaration = cob.JointActionDeclaration()
    declaration.carrier_degree = 1
    declaration.metric_source = cob.HodgeMetricSource.WhitneyPencil
    declaration.gravitational_weight = regge
    declaration.regge_form = cob.ReggeForm.Primal
    declaration.regge_hinges = hinges
    declaration.stiffness_weight = stiffness
    declaration.reference_lengths = (
        list(reference) if reference is not None else
        [complex(e.getLength()) for e in spacetime.getEdgeList().toVector()])
    declaration.holonomy_weight = beta
    declaration.holonomy_form = form
    declaration.matter_weight = 0.0
    return cob.JointAction(spacetime, declaration)


# ------------------------------------------------------------ primal Regge


def test_the_closed_fan_over_all_ten_hinges():
    """Three unit regular tetrahedra around the edge (0, 1): the central edge
    carries three dihedral angles, the six edges from 0 and 1 to the rim two
    each, the three rim edges one each, 18 angles arccos(1/3) in all. Over
    all ten edge hinges with the 2 pi convention and unit lengths the primal
    sum is 20 pi - 18 arccos(1/3)."""
    action = _geometric(_mesh(3, CLOSED_FAN))
    assert action.regge_hinge_count() == 10
    assert not action.regge_structurally_zero()
    assert complex(action.regge_term()) == pytest.approx(
        20 * math.pi - 18 * THETA3, abs=1e-12)


def test_the_closed_fan_has_one_interior_hinge():
    """Under the interior rule only the central edge's link closes, so the
    sum is its own deficit 2 pi - 3 arccos(1/3)."""
    action = _geometric(_mesh(3, CLOSED_FAN), cob.ReggeHinges.Interior)
    assert action.regge_hinge_count() == 1
    assert complex(action.regge_term()) == pytest.approx(
        2 * math.pi - 3 * THETA3, abs=1e-12)


def test_the_gravitational_weight_multiplies_the_term():
    action = _geometric(_mesh(3, CLOSED_FAN), cob.ReggeHinges.Interior,
                        regge=0.25)
    assert complex(action.regge_term()) == pytest.approx(
        0.25 * (2 * math.pi - 3 * THETA3), abs=1e-12)


@pytest.mark.parametrize("dimension, hinges, content, dihedral", [
    (4, 10, math.sqrt(3.0) / 4.0, math.acos(1.0 / 4.0)),
    (5, 15, 1.0 / (6.0 * math.sqrt(2.0)), math.acos(1.0 / 5.0)),
])
def test_a_single_regular_simplex_in_four_and_five_dimensions(
        dimension, hinges, content, dihedral):
    """A unit regular d-simplex: its (d - 2)-faces are the hinges (10
    triangles of area sqrt 3 / 4 in 4D, 15 tetrahedra of volume
    1 / (6 sqrt 2) in 5D), each in one top cell with dihedral angle
    arccos(1/d). Over all hinges the sum is (hinges) (content)
    (2 pi - arccos(1/d)); under the interior rule nothing closes and the term
    is structurally zero."""
    cells = [list(range(dimension + 1))]
    action = _geometric(_mesh(dimension, cells))
    assert action.regge_hinge_count() == hinges
    assert complex(action.regge_term()) == pytest.approx(
        hinges * content * (2 * math.pi - dihedral), rel=1e-13)
    interior = _geometric(_mesh(dimension, cells), cob.ReggeHinges.Interior)
    assert interior.regge_hinge_count() == 0
    assert interior.regge_structurally_zero()
    assert complex(interior.regge_term()) == 0.0


def test_a_lone_tetrahedron_and_the_run_level_are_structurally_zero():
    """A single tetrahedron has no interior hinge; nor has the run's
    two-tetrahedron fan (its shared edges each lie in two cells only, whose
    link does not close), which is why the run reported 'the Regge term is
    structurally zero on this level: it has 0 hinges'. Under the all-hinge
    rule the lone unit tetrahedron sums 6 (2 pi - arccos(1/3))."""
    lone = _mesh(3, [[0, 1, 2, 3]])
    assert _geometric(lone, cob.ReggeHinges.Interior).regge_structurally_zero()
    assert complex(_geometric(lone).regge_term()) == pytest.approx(
        6 * (2 * math.pi - THETA3), abs=1e-12)
    fan = _geometric(_mesh(3, [[0, 1, 2, 3], [0, 1, 3, 4]], 8.0),
                     cob.ReggeHinges.Interior)
    assert fan.regge_hinge_count() == 0 and fan.regge_structurally_zero()


def test_the_primal_gradient_matches_a_central_difference():
    """On the closed fan with generic complex squared lengths near 1, every
    component of `length_stationarity` (the exact dS/dz_e of the Regge term
    alone, on the continued sheet) matches a central difference of
    `regge_term` in that squared length, step 1e-6, to 1e-8 relative."""
    rng = np.random.default_rng(7)
    spacetime = _mesh(3, CLOSED_FAN)
    for edge in spacetime.getEdgeList().toVector():
        edge.setLength(cmath.sqrt(1.0 + 0.05 * rng.normal()
                                  + 0.01j * rng.normal()))
    action = _geometric(spacetime, cob.ReggeHinges.All)
    exact = np.asarray(action.length_stationarity())
    edges = spacetime.getEdgeList().toVector()
    h = 1e-6
    for index, edge in enumerate(edges):
        z = complex(edge.getLength()) ** 2
        values = []
        for step in (h, -h):
            edge.setLength(cmath.sqrt(z + step))
            values.append(complex(_geometric(spacetime).regge_term()))
        edge.setLength(cmath.sqrt(z))
        difference = (values[0] - values[1]) / (2 * h)
        assert abs(exact[index] - difference) <= 1e-8 * max(1.0,
                                                            abs(exact[index]))


# ---------------------------------------------------------------- Villain


def _direct_moments(beta, holonomy, terms=400):
    m = np.arange(-terms, terms + 1, dtype=float)
    base = np.exp(-m ** 2 / (2.0 * beta))
    weights = base * holonomy ** m
    w = weights.sum()
    return (w, (m * weights).sum() / w, (m * m * weights).sum() / w,
            beta / ((m * m * base).sum() / base.sum()))


@pytest.mark.parametrize("beta, stiffness, digits", [
    (0.5, 0.43091, 5), (1.0, 0.99796, 5), (1.3, 1.2998, 4),
    (2.0, 1.9999996, 7), (5.0, 5.0000, 4)])
def test_the_villain_stiffness_at_a_quarter_turn(beta, stiffness, digits):
    """WP v17 line 173 and `VillainCharacter`'s documentation: at F = i the
    per-face curvature in the real link angle is
    kappa = beta_V (<m^2>_i - <m>_i^2), real and positive, equal to the
    stated value to its stated digits. The curvature in the angle is minus
    the Maurer-Cartan second derivative D^2 phi, D = F d/dF, and it agrees
    with the direct 801-term moment sum to 1e-11."""
    character = cob.VillainCharacter(beta, 1e-18)
    kappa = -complex(character.second_derivative(1j))
    assert abs(kappa.imag) < 1e-12
    assert kappa.real == pytest.approx(stiffness, abs=0.6 * 10 ** -digits)
    w, mean, second, matched = _direct_moments(beta, 1j)
    assert kappa == pytest.approx(matched * (second - mean ** 2), abs=1e-11)


@pytest.mark.parametrize("beta", [0.5, 1.0, 2.0])
def test_the_matched_weight_is_beta_over_the_second_moment(beta):
    """WP v17 line 171: beta_V = beta / <m^2>_beta, so that the trivial
    holonomy curvature is beta for the Villain form as for Wilson's; checked
    against the direct sum and through D^2 phi(1) = -beta."""
    character = cob.VillainCharacter(beta, 1e-18)
    _, _, _, matched = _direct_moments(beta, 1.0)
    assert character.matched_weight == pytest.approx(matched, rel=1e-13)
    assert -complex(character.second_derivative(1.0)) == pytest.approx(
        beta, rel=1e-12)
    assert complex(character.first_derivative(1.0)) == pytest.approx(
        0.0, abs=1e-15)


@pytest.mark.parametrize("holonomy", [0.3 + 0.8j, 1.7 - 0.4j, -0.2 + 0.1j])
def test_the_weight_is_even_under_inversion(holonomy):
    """The coefficients are even in m, so W(1/F) = W(F): the potential is
    unchanged and the first Maurer-Cartan derivative changes sign, the
    second does not."""
    character = cob.VillainCharacter(1.0, 1e-16)
    inverse = 1.0 / holonomy
    assert complex(character.series(inverse).value) == pytest.approx(
        complex(character.series(holonomy).value), rel=1e-13)
    assert complex(character.potential(inverse)) == pytest.approx(
        complex(character.potential(holonomy)), rel=1e-12)
    assert complex(character.first_derivative(inverse)) == pytest.approx(
        -complex(character.first_derivative(holonomy)), rel=1e-12)
    assert complex(character.second_derivative(inverse)) == pytest.approx(
        complex(character.second_derivative(holonomy)), rel=1e-12)


@pytest.mark.parametrize("beta, n, sign", [(1.0, 1, 1), (1.0, 1, -1),
                                           (2.0, 2, 1), (0.5, 1, -1)])
def test_the_zeros_of_the_weight(beta, n, sign):
    """By Jacobi's triple product W vanishes exactly at
    F = -exp(+-(2n - 1) / (2 beta)): the direct sum there is zero to
    rounding against its terms, `zero_distance` is zero, and the logarithm
    (hence the potential) is refused as not certified nonzero."""
    zero = -math.exp(sign * (2 * n - 1) / (2 * beta))
    m = np.arange(-60, 61, dtype=float)
    terms = np.exp(-m ** 2 / (2 * beta)) * zero ** m
    assert abs(terms.sum()) < 1e-14 * np.sum(np.abs(terms))
    character = cob.VillainCharacter(beta, 1e-16)
    assert character.zero_distance(zero) == pytest.approx(0.0, abs=1e-15)
    with pytest.raises(ValueError):
        character.logarithm(zero)


def test_the_zero_distance_is_relative_to_the_nearest_zero():
    """At beta = 1 the nearest zeros of F = -1 are -e^{+-1/2}: the relative
    distance min |F - F_0| / min(|F|, |F_0|) is (e^{1/2} - 1) / 1 on the
    outer side and (1 - e^{-1/2}) / e^{-1/2} = e^{1/2} - 1 on the inner, the
    same number."""
    character = cob.VillainCharacter(1.0, 1e-16)
    assert character.zero_distance(-1.0) == pytest.approx(
        math.exp(0.5) - 1.0, rel=1e-13)


@pytest.mark.parametrize("holonomy", [1.0, 1j, 3.0 + 0.5j])
def test_the_tail_bounds_bound_the_omitted_series(holonomy):
    """The series keeps |m| <= M and reports bounds on the omitted parts of
    W, DW and D^2W. Against the direct 801-term sums the omitted parts are
    within their bounds, and the declared term count at tolerance 1e-12 is
    the least M with exp(-M^2 / (2 beta)) below it (beta = 1: M = 8)."""
    beta = 1.0
    character = cob.VillainCharacter(beta, 1e-12)
    assert character.declared_term_count == 8
    series = character.series(holonomy)
    m = np.arange(-400, 401, dtype=float)
    weights = np.exp(-m ** 2 / (2 * beta)) * holonomy ** m
    for exact, kept, bound in ((weights.sum(), series.value,
                                series.value_tail),
                               ((m * weights).sum(), series.first,
                                series.first_tail),
                               ((m * m * weights).sum(), series.second,
                                series.second_tail)):
        assert abs(exact - complex(kept)) <= bound + 1e-15 * abs(exact)


def test_the_villain_declaration_is_validated():
    with pytest.raises(ValueError):
        cob.VillainCharacter(0.0, 1e-12)
    with pytest.raises(ValueError):
        cob.VillainCharacter(1.0, 1.0)


# ---------------------------------------------- the holonomy term assembled


def _phase_hessian(spacetime, beta, holonomy):
    hessian = np.asarray(cob.JointAction(spacetime, bp.action_declaration(
        spacetime, 1.0, beta, holonomy=holonomy)).holonomy_hessian())
    return -hessian.reshape(18, 18)


def test_the_wilson_curvature_vanishes_at_a_quarter_turn():
    """The Wilson potential beta (1 - cos theta) has curvature beta cos theta,
    zero at F = +-i: on the unit-monopole host every face carries +-i
    (WP v17 line 504), so the Wilson phase Hessian is exactly zero there."""
    assert np.max(np.abs(_phase_hessian(bp.build_host(), 1.0, "wilson"))) \
        < 1e-14


@pytest.mark.parametrize("form", ["villain", "wilson"])
def test_the_holonomy_term_is_gauge_invariant(form):
    """A complex vertex gauge transformation g_v = exp(chi_v) multiplies
    U_xy by g_x g_y^-1 and leaves every face holonomy, hence the holonomy
    term, its value and its Hessian, unchanged (to 1e-12)."""
    spacetime = bp.build_host()
    before = cob.JointAction(spacetime, bp.action_declaration(
        spacetime, 1.0, 1.3, holonomy=form))
    value = complex(before.holonomy_term())
    faces = np.asarray(before.face_holonomies())
    chi = {v: 0.2 * v - 0.05j * v * v for v in range(12)}
    for edge in spacetime.getEdgeList().toVector():
        a, b = int(edge.getSource().getId()), int(edge.getTarget().getId())
        # U -> U e^{chi_a - chi_b}: phase phi -> phi - i (chi_a - chi_b)
        edge.setPhase(edge.getPhase() - 1j * (chi[a] - chi[b]))
    after = cob.JointAction(spacetime, bp.action_declaration(
        spacetime, 1.0, 1.3, holonomy=form))
    assert np.max(np.abs(np.asarray(after.face_holonomies()) - faces)) < 1e-12
    assert complex(after.holonomy_term()) == pytest.approx(value, abs=1e-12)
