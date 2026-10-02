# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Properties of the joint action's terms and of the stationarity system at a
fixed carried state, each checked at the declared tolerances (#1372).

Terms used below:

* the *joint action* is `cobordism.JointAction`, S = w_R S_Regge(z)
  + S_hol(U) + w_M tr(Gamma h_1(z, U)), over the complex squared lengths z_e
  and the links U_e = exp(i phi_e) of a complex; its *length stationarity* is
  dS/dz_e and its *link stationarity* is U_e dS/dU_e, the derivative along
  the Maurer-Cartan coordinate of the stored orientation;
* a *gauge transformation* is U_xy -> g_x^-1 U_xy g_y with g_x = exp(i
  theta_x) and theta_x complex (WP v18 section 3): *unit-modulus* when every
  theta_x is real, *complex* otherwise. It leaves the squared lengths as they
  are;
* a *dilation* by lambda > 0 multiplies every squared length by lambda and
  leaves the links as they are;
* the *oracle* of a derivative is the central difference of the function it
  differentiates, (f(x + h) - f(x - h)) / (2 h), at a step h that is 1e-5 of
  the coordinate's size. It is a test device, not a library mode. Its
  truncation is h^2 f''' / 6 and its rounding eps |f| / h, about 1e-10 of the
  derivative for the functions here; the tests assert 1e-8.

The fixture is the boundary of the 4-simplex (five tetrahedra, ten edges, a
closed 3-complex with ten interior hinges) with a deterministic metric that
is not regular and a deterministic connection whose links are off the unit
circle. Its squared lengths are 8 (1 + 0.05 a_e) + i y b_e times a scale,
with a_e in {-1.5, -0.5, 0.5, 1.5} and b_e in {-1.5, ..., 2.5}; y = 0.4 puts
the squared lengths on both sides of the real axis and y = -0.4 mirrors
them.
"""
import cmath
import hashlib
import itertools
import os
import subprocess
import sys

import numpy as np
import pytest

import tessera as T
from tessera import cobordism as cob

BOUNDARY_OF_FOUR_SIMPLEX = [list(c)
                            for c in itertools.combinations(range(5), 4)]

#: The scales of the squared lengths the properties are checked at.
SCALES = [1e-6, 1.0, 1e3, 1e9]

#: The relative step of the oracle and the agreement asserted with it.
ORACLE_STEP = 1e-5
ORACLE_AGREEMENT = 1e-8


# -------------------------------------------------------------------- fixtures


def _sphere(scale=1.0, imaginary=0.4, off_circle=1.0):
    """The fixture at the scale ``scale``, the imaginary parts of its squared
    lengths ``imaginary`` times b_e, and the imaginary parts of its phases
    ``off_circle`` times 0.09 c_e (zero puts every link on the unit
    circle)."""
    spacetime = T.Spacetime.fromVertexTuples(3, BOUNDARY_OF_FOUR_SIMPLEX,
                                             1.0, 0.0)
    for index, edge in enumerate(spacetime.getEdgeList().toVector()):
        squared = scale * (8.0 * (1.0 + 0.05 * ((3 * index) % 4 - 1.5))
                           + 1j * imaginary * ((7 * index) % 5 - 1.5))
        edge.setLength(cmath.sqrt(complex(squared)))
        edge.setPhase(complex(0.23 * ((index % 5) - 2),
                              off_circle * 0.09 * ((index % 3) - 1)))
    return spacetime


def _squared(spacetime):
    return np.array([complex(edge.getLength()) ** 2
                     for edge in spacetime.getEdgeList().toVector()])


def _declaration(regge=0.0, beta=0.0, matter=0.0, covariance=None,
                 form=cob.ReggeForm.Primal, branch=cob.ReggeBranch.Continued,
                 start=None):
    """The joint action with the Regge weight ``regge`` (in the declared form
    and on the declared branch, the continued sheets started at ``start``),
    the Villain coupling ``beta`` and the matter weight ``matter`` on the
    covariance ``covariance``."""
    declaration = cob.JointActionDeclaration()
    declaration.carrier_degree = 1
    declaration.metric_source = cob.HodgeMetricSource.WhitneyPencil
    declaration.gravitational_weight = regge
    declaration.regge_form = form
    declaration.regge_hinges = cob.ReggeHinges.Interior
    declaration.regge_branch = branch
    if start is not None:
        declaration.regge_start_squared_lengths = list(start)
    declaration.holonomy_weight = beta
    declaration.matter_weight = matter
    if covariance is not None:
        declaration.covariance = list(covariance)
    return declaration


def _projector(spacetime, occupied=3):
    """The spectral projector onto the ``occupied`` lowest modes of the
    covariant operator h_1 of ``spacetime``."""
    return cob.JointAction(spacetime, _declaration()).occupation_projector(
        occupied, True)


def _term(name, spacetime):
    """The declaration of one term alone, the Regge sheets started at the
    fixture's own squared lengths and the matter term on the projector of
    the fixture."""
    if name == "regge":
        return _declaration(regge=1.0, start=_squared(spacetime))
    if name == "villain":
        return _declaration(beta=1.3)
    return _declaration(matter=1.0, covariance=_projector(spacetime))


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
    """U -> U exp(step): on the stored phase, phi -> phi - i step."""
    edge.setPhase(complex(edge.getPhase()) - 1j * step)


def _gauge(spacetime, theta):
    """U_xy -> g_x^-1 U_xy g_y with g = exp(i theta): on the stored phase,
    phi_xy -> phi_xy + theta_y - theta_x."""
    for edge in spacetime.getEdgeList().toVector():
        source = int(edge.getSource().getId())
        target = int(edge.getTarget().getId())
        edge.setPhase(complex(edge.getPhase()) + theta[target]
                      - theta[source])


#: A unit-modulus and a complex gauge transformation of the five vertices.
GAUGES = {
    "unit-modulus": {0: 0.3, 1: 2.1, 2: 4.4, 3: 1.2, 4: 5.9},
    "complex": {0: 0.3 + 0.4j, 1: 2.1 - 0.7j, 2: 4.4 + 0.2j, 3: 1.2 - 0.3j,
                4: 5.9 + 0.6j},
}


def _relative(measured, expected):
    """The distance of two arrays or numbers over the size of the second."""
    measured = np.asarray(measured, dtype=complex)
    expected = np.asarray(expected, dtype=complex)
    return float(np.linalg.norm(np.ravel(measured - expected))
                 / max(np.linalg.norm(np.ravel(expected)), 1e-300))


# ------------------------------------- each term's gradient against its value


def _gradient_departures(name, scale, imaginary):
    """The relative distances of a term's length and link stationarity from
    the oracle of its value, with the norms of the two gradients."""
    spacetime = _sphere(scale, imaginary)
    declaration = _term(name, spacetime)

    def value():
        return cob.JointAction(spacetime, declaration).value()

    action = cob.JointAction(spacetime, declaration)
    lengths = np.asarray(action.length_stationarity())
    links = np.asarray(action.link_stationarity())
    snapshot = _snapshot(spacetime)
    edges = spacetime.getEdgeList().toVector()
    oracle_lengths = np.zeros(len(edges), dtype=complex)
    oracle_links = np.zeros(len(edges), dtype=complex)
    for index, edge in enumerate(edges):
        step = ORACLE_STEP * abs(complex(edge.getLength()) ** 2)
        _move_length(edge, step)
        up = value()
        _restore(spacetime, snapshot)
        _move_length(edge, -step)
        down = value()
        _restore(spacetime, snapshot)
        oracle_lengths[index] = (up - down) / (2.0 * step)
        _move_link(edge, ORACLE_STEP)
        up = value()
        _restore(spacetime, snapshot)
        _move_link(edge, -ORACLE_STEP)
        down = value()
        _restore(spacetime, snapshot)
        oracle_links[index] = (up - down) / (2.0 * ORACLE_STEP)
    return lengths, links, oracle_lengths, oracle_links


@pytest.mark.parametrize("imaginary", [0.4, -0.4])
@pytest.mark.parametrize("scale", SCALES)
@pytest.mark.parametrize("name", ["regge", "villain", "matter"])
def test_each_terms_stationarity_is_the_derivative_of_its_value(name, scale,
                                                                imaginary):
    """The length and link stationarity of each term alone agree with the
    oracle of the term's value at squared lengths of order 1e-5, 8, 8e3 and
    8e9, on both sides of the real axis, with links off the unit circle. The
    Regge term does not depend on the links and the Villain term does not
    depend on the squared lengths, exactly. Measured over the 24 cases: the
    largest relative distance is 4e-10."""
    lengths, links, oracle_lengths, oracle_links = _gradient_departures(
        name, scale, imaginary)
    if name == "villain":
        assert not np.any(lengths)
        assert not np.any(oracle_lengths)
    else:
        assert _relative(lengths, oracle_lengths) < ORACLE_AGREEMENT
    if name == "regge":
        assert not np.any(links)
        assert not np.any(oracle_links)
    else:
        assert _relative(links, oracle_links) < ORACLE_AGREEMENT


@pytest.mark.parametrize("imaginary", [0.0, 0.3, -0.3])
@pytest.mark.parametrize("count", [3, 4])
def test_the_regge_term_over_all_hinges_of_a_complex_with_a_boundary(
        count, imaginary):
    """A chain of three or four tetrahedra about the edge (0, 1),
    consecutive ones sharing a face (the cells of the recursion's declared
    host): it has no interior hinge, and the primal Regge term over all of
    its hinges (`ReggeHinges.All`, twelve and fifteen of them) has a length
    stationarity that agrees with the oracle of its value and obeys Euler's
    relation for the degree one half."""
    cells = [[0, 1, 2 + k, 3 + k] for k in range(count)]
    spacetime = T.Spacetime.fromVertexTuples(3, cells, 1.0, 0.0)
    for index, edge in enumerate(spacetime.getEdgeList().toVector()):
        edge.setLength(cmath.sqrt(complex(
            8.0 * (1.0 + 0.04 * ((3 * index) % 4 - 1.5)),
            imaginary * ((7 * index) % 5 - 1.5))))
    declaration = _declaration(regge=1.0, start=_squared(spacetime))
    declaration.regge_hinges = cob.ReggeHinges.All

    def value():
        return cob.JointAction(spacetime, declaration).value()

    action = cob.JointAction(spacetime, declaration)
    assert action.regge_hinge_count() == {3: 12, 4: 15}[count]
    gradient = np.asarray(action.length_stationarity())
    snapshot = _snapshot(spacetime)
    oracle = np.zeros(len(gradient), dtype=complex)
    for index, edge in enumerate(spacetime.getEdgeList().toVector()):
        step = ORACLE_STEP * abs(complex(edge.getLength()) ** 2)
        _move_length(edge, step)
        up = value()
        _restore(spacetime, snapshot)
        _move_length(edge, -step)
        down = value()
        _restore(spacetime, snapshot)
        oracle[index] = (up - down) / (2.0 * step)
    assert _relative(gradient, oracle) < ORACLE_AGREEMENT
    euler = np.sum(_squared(spacetime) * gradient)
    assert abs(euler - 0.5 * action.value()) < 1e-12 * abs(action.value())
    interior = _declaration(regge=1.0, start=_squared(spacetime))
    assert cob.JointAction(spacetime, interior).regge_structurally_zero()


# ------------------------------------------------- the degrees of homogeneity


#: The degree of homogeneity of each term in the squared lengths: the Regge
#: term is a sum of hinge contents (square roots of squared lengths in three
#: dimensions) times angles of degree zero, the Villain term does not depend
#: on them, and h_1 has degree -1 (WP v17 line 263).
DEGREES = {"regge": 0.5, "villain": 0.0, "matter": -1.0}


@pytest.mark.parametrize("imaginary", [0.0, 0.4, -0.4])
@pytest.mark.parametrize("name", ["regge", "villain", "matter"])
def test_eulers_relation_for_each_term(name, imaginary):
    """A term of degree k in the squared lengths obeys
    sum_e z_e dS/dz_e = k S, on real and on complex squared lengths: half the
    Regge term, zero for the Villain term, minus the matter term."""
    spacetime = _sphere(1.0, imaginary)
    action = cob.JointAction(spacetime, _term(name, spacetime))
    euler = np.sum(_squared(spacetime)
                   * np.asarray(action.length_stationarity()))
    expected = DEGREES[name] * action.value()
    if name == "villain":
        assert euler == 0.0
    else:
        assert abs(euler - expected) < 1e-12 * abs(expected)


@pytest.mark.parametrize("scale", [1e-6, 1e3, 1e9])
@pytest.mark.parametrize("name", ["regge", "villain", "matter"])
def test_a_dilation_scales_each_term_by_its_degree(name, scale):
    """Under a dilation by lambda a term of degree k is multiplied by
    lambda^k, its length stationarity by lambda^(k - 1) and its link
    stationarity by lambda^k, with the matter term's covariance the
    projector of the dilated operator (the projector does not change, h_1
    being multiplied by a number)."""
    reference = _sphere(1.0)
    dilated = _sphere(scale)
    one = cob.JointAction(reference, _term(name, reference))
    other = cob.JointAction(dilated, _term(name, dilated))
    degree = DEGREES[name]
    assert _relative(other.value(), scale ** degree * one.value()) < 1e-11
    if name != "villain":
        assert _relative(
            other.length_stationarity(),
            scale ** (degree - 1.0)
            * np.asarray(one.length_stationarity())) < 1e-10
    if name != "regge":
        assert _relative(
            other.link_stationarity(),
            scale ** degree * np.asarray(one.link_stationarity())) < 1e-10


# ----------------------------------------------------------- gauge invariance


@pytest.mark.parametrize("kind", ["unit-modulus", "complex"])
def test_the_terms_are_constant_on_gauge_orbits(kind):
    """Under a gauge transformation, unit-modulus or complex, the Regge term
    and the Villain term keep their values, the spectrum of h_1 is the same,
    and the matter term on the projector of the transformed operator keeps
    its value; the length stationarity and the link stationarity of the
    three terms together are the same (WP v18 section 3: the similarity the
    gauge group acts by)."""
    def reads(spacetime):
        action = cob.JointAction(spacetime, _declaration(
            regge=0.7, beta=1.3, matter=1.0,
            covariance=_projector(spacetime), start=_squared(spacetime)))
        return (action.regge_term(), action.holonomy_term(),
                action.matter_term(),
                np.asarray(action.length_stationarity()),
                np.asarray(action.link_stationarity()),
                np.sort_complex(np.asarray(action.carrier_eigenvalues())))

    spacetime = _sphere()
    before = reads(spacetime)
    _gauge(spacetime, GAUGES[kind])
    after = reads(spacetime)
    assert after[0] == before[0]
    for index in range(1, 6):
        assert _relative(after[index], before[index]) < 1e-13


# ------------------------------------------------------------ the term record


def test_the_term_records_add_up_to_the_action():
    """`action_term_records` lists the Regge term, the holonomy term, the
    matter term, the constraints and the action: every term's value is its
    weight times its bare value where it factors, the terms' values add up
    to the action's, the action's gradient norm is the residual norm of the
    system over the same coordinates, and the per-term gradients of
    `term_gradients` add up to the length and the link stationarity."""
    spacetime = _sphere()
    action = cob.JointAction(spacetime, _declaration(
        regge=0.7, beta=1.3, matter=1.0, covariance=_projector(spacetime)))
    geometry = cob.HolomorphicRelaxationDeclaration()
    geometry.relax_lengths = True
    geometry.relax_links = True
    records = cob.action_term_records(action, geometry)
    assert [record.name for record in records] == [
        "regge", "holonomy", "matter", "constraints", "action"]
    terms, total = records[:-1], records[-1]
    for record in terms:
        if record.factored:
            assert abs(record.value - record.weight * record.bare) \
                <= 1e-15 * abs(record.value)
    assert abs(sum(record.value for record in terms) - action.value()) \
        < 1e-13 * abs(action.value())
    assert abs(total.value - action.value()) <= 1e-15 * abs(total.value)
    residual = cob.HolomorphicRelaxation(action, geometry).residual()
    assert total.gradient_norm == pytest.approx(
        float(np.linalg.norm(residual)), rel=1e-14)
    gradients = action.term_gradients()
    assert _relative(sum(np.asarray(g.length_stationarity)
                         for g in gradients),
                     action.length_stationarity()) < 1e-14
    assert _relative(sum(np.asarray(g.link_stationarity) for g in gradients),
                     action.link_stationarity()) < 1e-14


# ------------------------------------- rounding-size imaginary parts of real z


def _regge_under_signs(form, branch):
    """The Regge term and its length stationarity on the fixture with real
    squared lengths, and with imaginary parts of 1e-16 of each squared
    length that are all positive, all negative, and of alternating sign."""
    out = {}
    for pattern in ("zero", "plus", "minus", "mixed"):
        spacetime = _sphere(1.0, 0.0, 0.0)
        for index, edge in enumerate(spacetime.getEdgeList().toVector()):
            squared = (complex(edge.getLength()) ** 2).real
            sign = {"zero": 0.0, "plus": 1.0, "minus": -1.0,
                    "mixed": 1.0 if index % 2 else -1.0}[pattern]
            edge.setLength(cmath.sqrt(complex(squared,
                                              sign * 1e-16 * squared)))
        action = cob.JointAction(spacetime, _declaration(
            regge=1.0, form=form, branch=branch))
        out[pattern] = (action.regge_term(),
                        np.asarray(action.length_stationarity()))
    return out


def test_the_continued_regge_term_does_not_read_the_sign_of_rounding():
    """On a real Euclidean metric the primal Regge term on its continued
    sheets, and its length stationarity, are the same to rounding whatever
    the signs of imaginary parts of 1e-16 of the squared lengths."""
    reads = _regge_under_signs(cob.ReggeForm.Primal,
                               cob.ReggeBranch.Continued)
    value, gradient = reads["zero"]
    for pattern in ("plus", "minus", "mixed"):
        assert _relative(reads[pattern][0], value) < 1e-14
        assert _relative(reads[pattern][1], gradient) < 1e-13


@pytest.mark.xfail(strict=True, reason=(
    "the dual Regge form keeps the principal branch, so on a real Euclidean "
    "metric the signs of imaginary parts of 1e-16 select the sheet of every "
    "dihedral angle: fix(drivers): the growth step scores the Regge term on "
    "a branch cut, where the sign of a rounding-size imaginary part decides "
    "it, https://github.com/akellehe/tessera/issues/1361"))
def test_the_dual_regge_term_does_not_read_the_sign_of_rounding():
    """The same property for the dual form (`ReggeForm.Dual`), which the
    growth step's objective reads through `ReggeSolver`. Measured: the value
    is 36.566 with the imaginary parts zero or of one sign and 22.792 with
    alternating signs."""
    reads = _regge_under_signs(cob.ReggeForm.Dual, cob.ReggeBranch.Continued)
    value, gradient = reads["zero"]
    for pattern in ("plus", "minus", "mixed"):
        assert _relative(reads[pattern][0], value) < 1e-14
        assert _relative(reads[pattern][1], gradient) < 1e-13


# ---------------------------------------- the system at a fixed carried state


def _fixed_state_system():
    """The stationarity system of the three terms on the fixture at the
    projector of its own operator, over the squared lengths and the links."""
    spacetime = _sphere()
    action = cob.JointAction(spacetime, _declaration(
        regge=0.7, beta=1.3, matter=1.0, covariance=_projector(spacetime)))
    geometry = cob.HolomorphicRelaxationDeclaration()
    geometry.relax_lengths = True
    geometry.relax_links = True
    return spacetime, cob.HolomorphicRelaxation(action, geometry)


@pytest.mark.parametrize("imaginary", [0.4, -0.4])
@pytest.mark.parametrize("scale", SCALES)
def test_the_fixed_state_jacobian_at_every_scale(scale, imaginary):
    """The Jacobian of the three terms at a fixed covariance is a Hessian:
    symmetric at every scale, and its length-length and link-link blocks
    agree with the oracle of the residual, each relative to the block's own
    size, at squared lengths of order 1e-5 to 8e9. The two mixed blocks are
    held to the oracle at the scales 1e-6 and 1, where the oracle resolves
    them: at 8e9 the length-link block is 5e-19, below the rounding of a
    difference of length rows of size 3e-5. Measured: 2e-10 at most."""
    spacetime = _sphere(scale, imaginary)
    action = cob.JointAction(spacetime, _declaration(
        regge=0.7, beta=1.3, matter=1.0, covariance=_projector(spacetime),
        start=_squared(spacetime)))
    geometry = cob.HolomorphicRelaxationDeclaration()
    geometry.relax_lengths = True
    geometry.relax_links = True
    system = cob.HolomorphicRelaxation(action, geometry)
    size = system.variable_count()
    count = size // 2
    jacobian = np.asarray(system.jacobian()).reshape(size, size)
    assert _relative(jacobian.T, jacobian) < 1e-14
    snapshot = _snapshot(spacetime)
    oracle = np.zeros((size, size), dtype=complex)

    def displaced(move, edge, step):
        move(edge, step)
        residual = np.asarray(system.residual())
        _restore(spacetime, snapshot)
        return residual

    for index, edge in enumerate(spacetime.getEdgeList().toVector()):
        step = ORACLE_STEP * abs(complex(edge.getLength()) ** 2)
        oracle[:, index] = (displaced(_move_length, edge, step)
                            - displaced(_move_length, edge, -step)) \
            / (2.0 * step)
        oracle[:, count + index] = (
            displaced(_move_link, edge, ORACLE_STEP)
            - displaced(_move_link, edge, -ORACLE_STEP)) / (2.0 * ORACLE_STEP)
    lengths, links = slice(0, count), slice(count, size)
    assert _relative(oracle[lengths, lengths],
                     jacobian[lengths, lengths]) < ORACLE_AGREEMENT
    assert _relative(oracle[links, links],
                     jacobian[links, links]) < ORACLE_AGREEMENT
    if scale <= 1.0:
        assert _relative(oracle[lengths, links],
                         jacobian[lengths, links]) < ORACLE_AGREEMENT
        assert _relative(oracle[links, lengths],
                         jacobian[links, lengths]) < ORACLE_AGREEMENT


def test_the_linearization_solves_the_newton_equation_it_reports():
    """At a fixed carried state the Jacobian is a Hessian, so it is
    symmetric. The Newton step d of `linearization` solves J d = -R on the
    range of J: the linear residual it reports is |J d + R| / |R|, the
    step is the one `solve` returns for -R, it has no component on the
    right singular vectors the rank decision discards, and the rank is the
    number of singular values above the rank tolerance times the largest.
    With an added residual and an added Jacobian the same holds of
    R + dR and J + dJ."""
    _, system = _fixed_state_system()
    residual = np.asarray(system.residual())
    size = len(residual)
    jacobian = np.asarray(system.jacobian()).reshape(size, size)
    assert _relative(jacobian.T, jacobian) < 1e-14
    linearization = system.linearization()
    newton = linearization.newton_step
    step = np.asarray(newton.step)
    _, singular, right = np.linalg.svd(jacobian)
    rank = int(np.sum(singular > newton.rank_tolerance * singular[0]))
    assert newton.jacobian_rank == rank == size - 1
    assert newton.residual_norm == pytest.approx(
        float(np.linalg.norm(residual)), rel=1e-14)
    measured = float(np.linalg.norm(jacobian @ step + residual)
                     / np.linalg.norm(residual))
    assert measured < 1e-12
    assert newton.linear_residual == pytest.approx(measured, abs=1e-13)
    np.testing.assert_array_equal(
        np.asarray(linearization.solve(list(-residual))), step)
    discarded = right[rank:]
    assert np.linalg.norm(discarded @ step) < 1e-12 * np.linalg.norm(step)

    rng = np.random.default_rng(5)
    added_residual = 0.01 * (rng.normal(size=size)
                             + 1j * rng.normal(size=size))
    added_jacobian = 0.01 * (rng.normal(size=(size, size))
                             + 1j * rng.normal(size=(size, size)))
    other = system.linearization(list(added_residual),
                                 list(added_jacobian.reshape(-1)))
    moved = np.asarray(other.newton_step.step)
    total = residual + added_residual
    assert other.newton_step.residual_norm == pytest.approx(
        float(np.linalg.norm(total)), rel=1e-14)
    assert np.linalg.norm((jacobian + added_jacobian) @ moved + total) \
        < 1e-11 * np.linalg.norm(total)


def test_the_residual_and_the_jacobian_are_the_same_at_every_call():
    """Two reads of the residual and of the Jacobian of one system, and the
    reads of a second system built from the same data, are equal bit for
    bit."""
    _, system = _fixed_state_system()
    _, again = _fixed_state_system()
    residual = np.asarray(system.residual())
    jacobian = np.asarray(system.jacobian())
    np.testing.assert_array_equal(np.asarray(system.residual()), residual)
    np.testing.assert_array_equal(np.asarray(system.jacobian()), jacobian)
    np.testing.assert_array_equal(np.asarray(again.residual()), residual)
    np.testing.assert_array_equal(np.asarray(again.jacobian()), jacobian)


_DIGEST = """
import hashlib, sys
import numpy as np
sys.path.insert(0, {directory!r})
import test_joint_action_properties_python as fixture
_, system = fixture._fixed_state_system()
data = np.asarray(system.residual()).tobytes() \\
    + np.asarray(system.jacobian()).tobytes()
print(hashlib.sha256(data).hexdigest())
"""


def test_the_residual_and_the_jacobian_do_not_depend_on_the_thread_count():
    """The residual and the Jacobian of the system are equal bit for bit
    with one OpenMP thread and with two: each is read in a fresh interpreter
    with OMP_NUM_THREADS set, and the digests of their bytes are compared
    with each other and with this process's."""
    directory = os.path.dirname(os.path.abspath(__file__))
    code = _DIGEST.format(directory=directory)
    digests = []
    for threads in ("1", "2"):
        environment = dict(os.environ, OMP_NUM_THREADS=threads)
        result = subprocess.run([sys.executable, "-c", code],
                                env=environment, capture_output=True,
                                text=True, check=True)
        digests.append(result.stdout.strip().splitlines()[-1])
    _, system = _fixed_state_system()
    here = hashlib.sha256(
        np.asarray(system.residual()).tobytes()
        + np.asarray(system.jacobian()).tobytes()).hexdigest()
    assert digests[0] == digests[1] == here
