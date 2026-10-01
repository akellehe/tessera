# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The geometric terms of the recursion driver's joint action held to closed
forms: the primal Regge term on known meshes and the Villain holonomy term.

`recursion.relax_level` and `baryon_poles.relax_content` both relax the joint
action S = w_R S_Regge(primal) + S_hol + w_M tr(Gamma h) (whitepaper v17 §7,
`JointAction`). This file holds the first and the second term to values fixed
independently of the library:

* the primal Regge sum sum_h |h| (2 pi - sum theta_h) over the hinges the
  declared rule selects (`ReggeHinges.All`: every hinge, with the library's
  2 pi convention; `ReggeHinges.Interior`: hinges whose link closes), on the
  closed fan of three unit regular tetrahedra around an edge, on a single
  regular 4-simplex and on a single regular 5-simplex, whose dihedral angles
  are arccos(1/d);
* its exact gradient in the squared lengths against a central difference;
* the Villain weight W(F) = sum_m exp(-m^2 / (2 beta)) F^m of WP v17 lines
  171-173 summed to the declared order M, W_M = sum over |m| <= M
  (`VillainCharacter`, order ten by default): the matched weight
  beta / <m^2>_beta, the curvature at a quarter turn F = i,
  beta_V (<m^2>_i - <m>_i^2), equal at order ten to 0.43091 at beta = 1/2,
  0.99796 at 1, 1.2998 at 1.3, 1.9999996 at 2 and 5.1552 at 5 (the infinite
  series gives 5.0000 there), the symmetry W_M(1/F) = W_M(F), the zeros
  F = -exp(+-(2n - 1) / (2 beta)) of Jacobi's triple product for the
  infinite series, on whose rays log W_M has no value, and the reported
  tail bounds;
* the gauge invariance of the holonomy term.
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


def _geometric(spacetime, hinges=cob.ReggeHinges.All, regge=1.0, beta=0.0):
    declaration = cob.JointActionDeclaration()
    declaration.carrier_degree = 1
    declaration.metric_source = cob.HodgeMetricSource.WhitneyPencil
    declaration.gravitational_weight = regge
    declaration.regge_form = cob.ReggeForm.Primal
    declaration.regge_hinges = hinges
    declaration.holonomy_weight = beta
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


#: The order the Villain weight is summed to in these reads: the largest
#: that can be declared, and the default.
ORDER = 10


def _direct_moments(beta, holonomy, terms=ORDER):
    """W, <m>_F, <m^2>_F and beta_V of the sums over |m| <= terms."""
    m = np.arange(-terms, terms + 1, dtype=float)
    base = np.exp(-m ** 2 / (2.0 * beta))
    weights = base * holonomy ** m
    w = weights.sum()
    return (w, (m * weights).sum() / w, (m * m * weights).sum() / w,
            beta / ((m * m * base).sum() / base.sum()))


@pytest.mark.parametrize("beta, stiffness, digits", [
    (0.5, 0.43091, 5), (1.0, 0.99796, 5), (1.3, 1.2998, 4),
    (2.0, 1.9999996, 7), (5.0, 5.1552, 4)])
def test_the_villain_stiffness_at_a_quarter_turn(beta, stiffness, digits):
    """WP v17 line 173 and `VillainCharacter`'s documentation: at F = i the
    per-face curvature in the real link angle is
    kappa = beta_V (<m^2>_i - <m>_i^2), real and positive, equal at order
    ten to the stated value to its stated digits. The curvature in the
    angle is minus the Maurer-Cartan second derivative D^2 phi, D = F d/dF,
    and it agrees with the direct moment sums over |m| <= 10 to 1e-11. Up
    to beta = 2 the 801-term sums give the same value to 1e-10 (the first
    pair beyond the order carries exp(-121 / (2 beta)), 7e-14 at beta = 2).
    At beta = 5 they give 5.0000: there the pair m = +-11 of DW,
    22 exp(-121 / 10) = 1.2e-4, is one part in a hundred of
    W(i) = 0.011739, and the order-ten curvature is 5.1552."""
    character = cob.VillainCharacter(beta, ORDER)
    kappa = -complex(character.second_derivative(1j))
    assert abs(kappa.imag) < 1e-12
    assert kappa.real == pytest.approx(stiffness, abs=0.6 * 10 ** -digits)
    w, mean, second, matched = _direct_moments(beta, 1j)
    assert kappa == pytest.approx(matched * (second - mean ** 2), abs=1e-11)
    w, mean, second, matched = _direct_moments(beta, 1j, terms=400)
    infinite = (matched * (second - mean ** 2)).real
    if beta <= 2.0:
        assert kappa.real == pytest.approx(infinite, abs=1e-10)
    else:
        assert infinite == pytest.approx(5.0000, abs=1e-10)
        assert kappa.real - infinite == pytest.approx(0.1552, abs=1e-4)


@pytest.mark.parametrize("beta", [0.5, 1.0, 2.0])
def test_the_matched_weight_is_beta_over_the_second_moment(beta):
    """WP v17 line 171: beta_V = beta / <m^2>_beta, with the second moment of
    the sums over |m| <= 10, so that the trivial-holonomy curvature is beta;
    checked against the direct sum over the same terms and through
    D^2 phi(1) = -beta. The 801-term sum gives the same weight to 3e-12 (at
    beta = 2 the pair m = +-11 carries 242 exp(-121 / 4) = 1.8e-11 against
    a sum of 3.5)."""
    character = cob.VillainCharacter(beta, ORDER)
    _, _, _, matched = _direct_moments(beta, 1.0)
    assert character.matched_weight == pytest.approx(matched, rel=1e-13)
    _, _, _, infinite = _direct_moments(beta, 1.0, terms=400)
    assert character.matched_weight == pytest.approx(infinite, rel=3e-12)
    assert -complex(character.second_derivative(1.0)) == pytest.approx(
        beta, rel=1e-12)
    assert complex(character.first_derivative(1.0)) == pytest.approx(
        0.0, abs=1e-15)


@pytest.mark.parametrize("holonomy", [0.3 + 0.8j, 1.7 - 0.4j, -0.2 + 0.1j])
def test_the_weight_is_even_under_inversion(holonomy):
    """The coefficients are even in m, so W(1/F) = W(F): the potential is
    unchanged and the first Maurer-Cartan derivative changes sign, the
    second does not."""
    character = cob.VillainCharacter(1.0, ORDER)
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
    """By Jacobi's triple product the infinite series vanishes exactly at
    F = -exp(+-(2n - 1) / (2 beta)): the direct 121-term sum there is zero
    to rounding against its terms. The order-ten sum differs from it by its
    tail, so it has a zero on the negative real axis within the tail of
    each of these points, and the ray from -1 to the point runs along that
    axis: the logarithm (hence the potential) has no value there, because
    the path meets a zero of W_M before the point or ends on one."""
    zero = -math.exp(sign * (2 * n - 1) / (2 * beta))
    m = np.arange(-60, 61, dtype=float)
    terms = np.exp(-m ** 2 / (2 * beta)) * zero ** m
    assert abs(terms.sum()) < 1e-14 * np.sum(np.abs(terms))
    character = cob.VillainCharacter(beta, ORDER)
    series = character.series(zero)
    assert abs(complex(series.value)) <= series.value_tail \
        + series.rounding_bound
    with pytest.raises(ValueError, match="meets a zero of W_M"):
        character.logarithm(zero)


@pytest.mark.parametrize("holonomy", [1.0, 1j, 3.0 + 0.5j])
def test_the_tail_bounds_bound_the_omitted_series(holonomy):
    """The sums keep |m| <= M at the declared order M and report bounds on
    the parts of the infinite series of W, DW and D^2W beyond it. At order
    eight and beta = 1 the first pair beyond the order carries
    exp(-81 / 2) = 2.6e-18, and against the direct 801-term sums the parts
    beyond the order are within their bounds."""
    beta = 1.0
    character = cob.VillainCharacter(beta, 8)
    assert character.order == 8
    series = character.series(holonomy)
    assert series.order == 8
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
    """The coupling is positive and the order is an integer from one to
    ten, the largest order that can be declared."""
    with pytest.raises(ValueError):
        cob.VillainCharacter(0.0, ORDER)
    for order in (0, 11):
        with pytest.raises(ValueError, match="integer from 1 to 10"):
            cob.VillainCharacter(1.0, order)


# ---------------------------------------------- the holonomy term assembled


def test_the_holonomy_term_is_gauge_invariant():
    """A complex vertex gauge transformation g_v = exp(chi_v) multiplies
    U_xy by g_x g_y^-1 and leaves every face holonomy, hence the holonomy
    term, its value and its Hessian, unchanged (to 1e-12)."""
    spacetime = bp.build_host()
    before = cob.JointAction(spacetime, bp.action_declaration(
        spacetime, 1.0, 1.3))
    value = complex(before.holonomy_term())
    faces = np.asarray(before.face_holonomies())
    chi = {v: 0.2 * v - 0.05j * v * v for v in range(12)}
    for edge in spacetime.getEdgeList().toVector():
        a, b = int(edge.getSource().getId()), int(edge.getTarget().getId())
        # U -> U e^{chi_a - chi_b}: phase phi -> phi - i (chi_a - chi_b)
        edge.setPhase(edge.getPhase() - 1j * (chi[a] - chi[b]))
    after = cob.JointAction(spacetime, bp.action_declaration(
        spacetime, 1.0, 1.3))
    assert np.max(np.abs(np.asarray(after.face_holonomies()) - faces)) < 1e-12
    assert complex(after.holonomy_term()) == pytest.approx(value, abs=1e-12)
