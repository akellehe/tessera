# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The cosmological term of the joint action (#1417).

Terms used below:

* the *cosmological term* is `cobordism.JointAction.cosmological_term`,
  -w_R Lambda sum_T V_T, with w_R the declared Regge weight (1/kappa),
  Lambda the declared ``cosmological_constant`` and V_T the volume of each
  top simplex: the root of Q_T = (-1)^(d+1) det B_T / (2^d (d!)^2), B_T its
  Cayley-Menger matrix, on the sheet the Regge term is read on
  (`JointAction.top_cell_volumes`);
* a *dilation* by s > 0 multiplies every squared length by s. The primal
  Regge sum of a three-dimensional complex is homogeneous of degree one half
  in the squared lengths (one in the lengths) and the volume of degree three
  halves (three in the lengths), so along the dilation of a geometry z the
  action w_R (s^(1/2) S_Regge(z) - Lambda s^(3/2) sum_T V_T(z)) is stationary
  at s* = S_Regge / (3 Lambda sum_T V_T), where S_Regge = 3 Lambda sum_T V_T,
  when the two sides have the same sign;
* the *oracle* of a derivative is the central difference
  (f(x + h) - f(x - h)) / (2 h), a test device and not a library mode. Its
  truncation is h^2 f''' / 6, so its disagreement with an exact derivative
  falls with the square of the step until the rounding eps |f| / h takes
  over.

The fixtures are the boundary of the 4-simplex (five tetrahedra, ten
edges, every hinge interior): with the complex metric of
`test_joint_action_properties_python` for the derivatives and the degrees,
and with real squared lengths for the relaxation.
"""
import cmath
import hashlib
import itertools
import math

import numpy as np
import pytest

import tessera as T
from tessera import cobordism as cob

from tests.cobordism import test_joint_action_properties_python as P

#: The digest of the residual and the Jacobian of
#: `test_joint_action_properties_python._fixed_state_system` (the Regge,
#: Villain and matter terms on the complex metric, lengths and links
#: relaxed), recorded with the module of ac499572, the commit the term was
#: added to, on the development machine.
BASE_DIGEST = ("ee2530fcf42470a2b95a139c2be8588fad731ae58944cc026a2eeeea995324d6")

#: The deficit angle of every hinge of the regular boundary of the
#: 4-simplex: three regular tetrahedra meet at each edge.
REGULAR_DEFICIT = 2.0 * math.pi - 3.0 * math.acos(1.0 / 3.0)


def _digest(system):
    return hashlib.sha256(np.asarray(system.residual()).tobytes()
                          + np.asarray(system.jacobian()).tobytes()
                          ).hexdigest()


def _term(action, name):
    for term in action.term_gradients():
        if term.name == name:
            return term
    raise KeyError(name)


def _cosmological(spacetime, regge=0.7, constant=0.37):
    """The Regge term with the weight ``regge`` on the continued sheets
    started at the fixture, and the cosmological term at ``constant``."""
    declaration = P._declaration(regge=regge, start=P._squared(spacetime))
    declaration.cosmological_constant = constant
    return cob.JointAction(spacetime, declaration)


# ------------------------------------------------------------ Lambda = 0


def test_without_lambda_the_residual_and_jacobian_are_those_of_the_base():
    """With Lambda = 0, left at its default or declared, the residual and
    the Jacobian of the stationarity system are bit for bit those of the
    module of ac499572, and the action lists the terms it listed there."""
    _, system = P._fixed_state_system()
    assert system.action.declaration.cosmological_constant == 0.0
    assert _digest(system) == BASE_DIGEST
    spacetime = P._sphere()
    declaration = P._declaration(regge=0.7, beta=1.3, matter=1.0,
                                 covariance=P._projector(spacetime))
    declaration.cosmological_constant = 0.0
    action = cob.JointAction(spacetime, declaration)
    geometry = cob.HolomorphicRelaxationDeclaration()
    geometry.relax_lengths = True
    geometry.relax_links = True
    assert _digest(cob.HolomorphicRelaxation(action, geometry)) == BASE_DIGEST
    assert [t.name for t in action.term_gradients()] == [
        "regge", "holonomy", "matter"]
    assert action.cosmological_term() == 0.0
    assert not np.any(np.asarray(action.cosmological_hessian()))


def test_a_cosmological_constant_that_is_not_finite_is_refused():
    spacetime = P._sphere()
    for value in (math.inf, -math.inf, math.nan):
        declaration = P._declaration(regge=1.0)
        declaration.cosmological_constant = value
        with pytest.raises(ValueError, match="cosmological constant"):
            cob.JointAction(spacetime, declaration)


# ------------------------------------------------------- the derivatives


@pytest.mark.parametrize("imaginary, constant", [(0.4, 0.37), (-0.4, -1.1)])
def test_the_gradient_and_the_hessian_against_the_oracle(imaginary, constant):
    """The term's length stationarity and `cosmological_hessian` agree with
    the oracle of the term and of its stationarity on the complex metric,
    and the disagreement falls with the square of the step: from the
    relative step 1e-3 to 1e-4 it falls by a factor of about a hundred.
    The Hessian is symmetric exactly; the action's length stationarity is
    the sum of its terms', and the length block of `action_hessian` is the
    Regge Hessian plus this one exactly. Measured, relative to the largest
    entry: the gradient 2.49e-7, 2.5e-9 and 1e-10 at the relative steps
    1e-3, 1e-4 and 1e-5, the Hessian 4.09e-7, 4.09e-9 and 6e-11; the sum of
    the terms 5.6e-17 of the stationarity."""
    spacetime = P._sphere(1.0, imaginary)
    action = _cosmological(spacetime, constant=constant)
    edges = spacetime.getEdgeList().toVector()
    count = len(edges)
    z = P._squared(spacetime)
    snapshot = P._snapshot(spacetime)
    gradient = np.asarray(_term(action, "cosmological").length_stationarity)
    hessian = np.asarray(action.cosmological_hessian()).reshape(count, count)
    assert np.array_equal(hessian, hessian.T)

    def disagreement(relative):
        oracle_gradient = np.zeros(count, dtype=complex)
        oracle_hessian = np.zeros((count, count), dtype=complex)
        for e in range(count):
            h = relative * abs(z[e])
            values, rows = [], []
            for sign in (1.0, -1.0):
                P._restore(spacetime, snapshot)
                P._move_length(edges[e], sign * h)
                values.append(complex(action.cosmological_term()))
                rows.append(np.asarray(
                    _term(action, "cosmological").length_stationarity))
            P._restore(spacetime, snapshot)
            oracle_gradient[e] = (values[0] - values[1]) / (2.0 * h)
            oracle_hessian[:, e] = (rows[0] - rows[1]) / (2.0 * h)
        return (np.max(np.abs(oracle_gradient - gradient))
                / np.max(np.abs(gradient)),
                np.max(np.abs(oracle_hessian - hessian))
                / np.max(np.abs(hessian)))

    coarse = disagreement(1e-3)
    fine = disagreement(1e-4)
    for before, after in zip(coarse, fine):
        assert after < 1e-8
        assert 50.0 < before / after < 200.0
    total = np.asarray(action.length_stationarity())
    parts = sum(np.asarray(t.length_stationarity)
                for t in action.term_gradients())
    assert np.max(np.abs(total - parts)) <= 1e-15 * np.max(np.abs(total))
    block = np.asarray(action.action_hessian(True, False)).reshape(
        2 * count, 2 * count)[:count, :count]
    assert np.array_equal(
        block, np.asarray(action.regge_hessian()).reshape(count, count)
        + hessian)


def test_the_term_record_carries_its_weight_and_value():
    """The term's line: its weight is -w_R Lambda, what the weight
    multiplies is the sum of the volumes, and its value is their product,
    the term itself."""
    spacetime = P._sphere()
    action = _cosmological(spacetime, regge=0.7, constant=0.37)
    names = [t.name for t in action.term_gradients()]
    assert names == ["regge", "cosmological", "holonomy", "matter"]
    term = _term(action, "cosmological")
    assert term.label == "-(1/kappa) Lambda sum_T V_T"
    assert term.factored
    assert term.weight == -0.7 * 0.37
    assert term.value == action.cosmological_term()
    assert abs(term.bare - action.volume_sum()) <= 1e-15 * abs(
        action.volume_sum())
    assert action.value() == (action.regge_term() + action.cosmological_term()
                              + action.holonomy_term() + action.matter_term()
                              + action.spectral_term())


# ------------------------------------------------------------ the degrees


@pytest.mark.parametrize("scale", [0.5, 3.0])
def test_the_degrees_of_the_two_terms_under_a_dilation(scale):
    """Under the dilation z -> s z of the complex metric the volume sum
    scales as s^(3/2) and the Regge term as s^(1/2), and the Euler sums
    sum_e z_e dT/dz_e are 3/2 and 1/2 of the terms: so along the dilation
    the action is stationary where S_Regge = 3 Lambda sum_T V_T. Measured:
    every ratio within 3.3e-16 of its degree."""
    spacetime = P._sphere(1.0, 0.4)
    z = P._squared(spacetime)

    def read(s):
        for e, edge in enumerate(spacetime.getEdgeList().toVector()):
            edge.setLength(cmath.sqrt(complex(s * z[e])))
        action = _cosmological(spacetime, regge=1.0, constant=0.37)
        here = P._squared(spacetime)
        return (complex(action.regge_term()), complex(action.volume_sum()),
                complex(action.cosmological_term()),
                np.dot(here, _term(action, "regge").length_stationarity),
                np.dot(here, _term(action, "cosmological").length_stationarity))

    regge, volume, _, _, _ = read(1.0)
    scaled_regge, scaled_volume, cosmological, euler_regge, \
        euler_cosmological = read(scale)
    assert abs(scaled_volume / volume / scale ** 1.5 - 1.0) < 1e-15
    assert abs(scaled_regge / regge / scale ** 0.5 - 1.0) < 1e-15
    assert abs(euler_cosmological / (1.5 * cosmological) - 1.0) < 1e-15
    assert abs(euler_regge / (0.5 * scaled_regge) - 1.0) < 1e-15


def test_the_volume_is_read_on_the_continued_sheet():
    """The volume of a dilation by a complex factor s, z -> s z, continued
    along the straight segment from the starting geometry, is
    V(z) s^(3/2) with the principal power of s; past arg s = pi / 3 the
    square Q = Q(z) s^3 has crossed the negative real axis, and the
    principal root of Q is its negative. Under `ReggeBranch.Principal` the
    volume is that principal root."""
    edges = list(itertools.combinations(range(5), 2))
    spacetime = T.Spacetime.fromVertexTuples(3, P.BOUNDARY_OF_FOUR_SIMPLEX,
                                             1.0, 0.0)
    for edge in spacetime.getEdgeList().toVector():
        edge.setLength(cmath.sqrt(8.0))
    start = P._squared(spacetime)
    regular = 8.0 ** 1.5 / (6.0 * math.sqrt(2.0))
    s = 1.2 * cmath.exp(0.36j * math.pi)
    for edge in spacetime.getEdgeList().toVector():
        edge.setLength(cmath.sqrt(8.0 * s))
    for branch, sign in ((cob.ReggeBranch.Continued, 1.0),
                         (cob.ReggeBranch.Principal, -1.0)):
        declaration = P._declaration(regge=1.0, branch=branch, start=start)
        declaration.cosmological_constant = 1.0
        volumes = cob.JointAction(spacetime, declaration).top_cell_volumes()
        expected = sign * regular * s ** 1.5
        assert len(volumes) == 5 and len(edges) == 10
        for volume in volumes:
            assert abs(volume - expected) < 1e-13 * abs(expected)


# ---------------------------------------------------------- the relaxation


def _level(squared):
    spacetime = T.Spacetime.fromVertexTuples(3, P.BOUNDARY_OF_FOUR_SIMPLEX,
                                             1.0, 0.0)
    for index, edge in enumerate(spacetime.getEdgeList().toVector()):
        edge.setLength(cmath.sqrt(complex(squared(index))))
    return spacetime


def _system(spacetime, constant, start):
    """The stationarity system of the level at kappa = 1, the holonomy term
    off, over the squared lengths, the Regge sheets started at ``start``."""
    declaration = P._declaration(regge=1.0, start=start)
    declaration.cosmological_constant = constant
    geometry = cob.HolomorphicRelaxationDeclaration()
    geometry.relax_lengths = True
    geometry.relax_links = False
    geometry.relax_multipliers = False
    return cob.HolomorphicRelaxation(cob.JointAction(spacetime, declaration),
                                     geometry)


def _newton(spacetime, constant):
    """Full Newton steps of the system from the level as it stands, each
    written to the mesh, until a step no longer lowers the residual norm.
    Returns the system at the last point and the residual norms."""
    start = P._squared(spacetime)
    system = _system(spacetime, constant, start)
    norms = [float(np.linalg.norm(system.residual()))]
    while True:
        step = np.asarray(system.newton_step().step)
        for edge, dz in zip(spacetime.getEdgeList().toVector(), step):
            P._move_length(edge, dz)
        system = _system(spacetime, constant, start)
        norms.append(float(np.linalg.norm(system.residual())))
        if not norms[-1] < norms[-2]:
            return system, norms


@pytest.mark.parametrize("perturbed", [False, True])
def test_a_level_that_dilates_ends_at_a_finite_scale_with_lambda(perturbed):
    """The boundary of the 4-simplex at kappa = 1 with the holonomy term
    off, from squared lengths 8 (regular, or off it by five per cent):
    without the term a Newton step is a pure dilation that triples every
    squared length (the stationary point is at infinite scale, #1288). With
    Lambda = 1, the sign of the Regge sum and of the volumes, the Newton
    steps end at a finite scale, the regular geometry with squared length
    z* = 4 sqrt(2) eps / Lambda (eps the regular deficit angle; z* =
    14.65299), where S_Regge = 3 Lambda sum_T V_T. With Lambda = -1 the two
    sides have opposite signs and there is no stationary scale: the Newton
    step from the regular start is again a pure dilation, by more than
    three. Measured: the dilation factor 3 to 2.0e-14 (regular) and 5.5e-14
    (perturbed) of it; with Lambda = 1 five and six steps that lower the
    residual norm, quadratically, to 5.6e-16 and 6.4e-16, the squared
    lengths within 6.7e-16 of z*, S_Regge / (3 Lambda sum_T V_T) - 1 at
    2.2e-16; with Lambda = -1 the factor 7.80987."""
    def squared(index):
        return 8.0 * (1.0 + (0.05 * ((3 * index) % 4 - 1.5)
                             if perturbed else 0.0))
    spacetime = _level(squared)
    z = P._squared(spacetime)
    step = np.asarray(_system(spacetime, 0.0, z).newton_step().step)
    assert np.max(np.abs((z + step) / z - 3.0)) < 1e-13
    if not perturbed:
        step = np.asarray(_system(spacetime, -1.0, z).newton_step().step)
        factors = (z + step) / z
        assert np.max(np.abs(factors - factors[0])) < 1e-13 * abs(factors[0])
        assert factors[0].real > 3.0
    system, norms = _newton(spacetime, 1.0)
    target = 4.0 * math.sqrt(2.0) * REGULAR_DEFICIT
    assert np.max(np.abs(P._squared(spacetime) / target - 1.0)) < 1e-15
    assert min(norms) <= 1e-15
    action = system.action
    assert abs(action.regge_term() / (3.0 * action.volume_sum()) - 1.0) < 1e-15
    assert abs(action.regge_term() + 3.0 * action.cosmological_term()) < \
        1e-15 * abs(action.regge_term())
