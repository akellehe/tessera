# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The stiffness and matter terms of the recursion driver's joint action, their
forces, the Ward current, and the dressed fluctuation of whitepaper v17 §7.

The per-cell relaxation of `baryon_poles.relax_content` makes the geometry
stationary under the linear stiffness stand-in (1/kappa)(1/2)||l - l0||^2 and
the matter term tr(Gamma h_1(z, U)); the Section 7 quartic
(`evaluate_content` -> `DressedFluctuation.effective_action`) eliminates the
fluctuations with the dressed stiffness A + D - Pi(omega). Expected values:

* the stiffness term and its closed-form gradient w (l - l0) / (2 l) in z;
* the Hellmann-Feynman forces tr(Gamma dh/dz_e) and tr(Gamma U_e dh/dU_e)
  against central differences of tr(Gamma h) with Gamma held fixed, and the
  Euler identity sum_e z_e tr(Gamma dh/dz_e) = -tr(Gamma h) of the degree -1
  homogeneity (WP v17 line 263);
* the Ward identity: the divergence of the current j = U dS/dU vanishes for
  every gauge-invariant term, and for the matter term exactly when Gamma is a
  spectral projector of h (`JointAction.ward_current_divergence`);
* WP v17 line 289: D - Pi(0) is the Hessian of the occupied energy and
  vanishes on a pure-gauge direction; on a two-level carrier with one
  particle-hole energy Delta and one fluctuation, the collective mode sits at
  omega^2 = Delta^2 (1 - Pi(0) / (A + D)).
"""
import cmath
import math

import numpy as np
import pytest

import tessera as T
from tessera import cobordism as cob


def _tetrahedron(seed=3):
    """A single tetrahedron with generic complex squared lengths near 8 and a
    generic complex connection (phases with small imaginary parts)."""
    rng = np.random.default_rng(seed)
    spacetime = T.Spacetime.fromVertexTuples(3, [[0, 1, 2, 3]], 1.0, 0.0)
    for edge in spacetime.getEdgeList().toVector():
        edge.setLength(cmath.sqrt(8.0 + rng.normal() + 0.2j * rng.normal()))
        edge.setPhase(complex(rng.normal(), 0.05 * rng.normal()))
    return spacetime


def _declaration(spacetime, stiffness=0.0, beta=0.0, matter=0.0,
                 covariance=None, reference=None):
    declaration = cob.JointActionDeclaration()
    declaration.carrier_degree = 1
    declaration.metric_source = cob.HodgeMetricSource.WhitneyPencil
    declaration.gravitational_weight = 0.0
    declaration.regge_form = cob.ReggeForm.Primal
    declaration.regge_hinges = cob.ReggeHinges.Interior
    declaration.stiffness_weight = stiffness
    declaration.reference_lengths = (
        list(reference) if reference is not None else
        [complex(e.getLength()) for e in spacetime.getEdgeList().toVector()])
    declaration.holonomy_weight = beta
    declaration.holonomy_form = cob.HolonomyForm.Villain
    declaration.matter_weight = matter
    if covariance is not None:
        declaration.covariance = list(np.asarray(covariance).reshape(-1))
    return declaration


def _matrix(flat):
    flat = np.asarray(flat, dtype=complex)
    n = int(round(math.sqrt(flat.size)))
    return flat.reshape(n, n)


# ------------------------------------------------------------- stiffness


def test_the_linear_stiffness_term_and_its_gradient():
    """WP v17 §7's stand-in (1/2) w ||l - l0||^2 with w = 1/kappa = 2: the
    value is half the weighted squared stretch, and dS/dz_e is
    w (l_e - l0_e) / (2 l_e) per edge, exactly."""
    spacetime = _tetrahedron()
    edges = spacetime.getEdgeList().toVector()
    reference = [cmath.sqrt(8.0)] * 6
    action = cob.JointAction(spacetime, _declaration(
        spacetime, stiffness=2.0, reference=reference))
    lengths = np.array([complex(e.getLength()) for e in edges])
    assert complex(action.stiffness_term()) == pytest.approx(
        0.5 * 2.0 * np.sum((lengths - reference[0]) ** 2), rel=1e-14)
    np.testing.assert_allclose(
        np.asarray(action.length_stationarity()),
        2.0 * (lengths - reference[0]) / (2.0 * lengths), rtol=1e-13)


# -------------------------------------------------------------- matter


def _fixed_gamma(spacetime):
    return cob.JointAction(spacetime, _declaration(spacetime)) \
        .occupation_projector(2)


def _matter(spacetime, gamma):
    return cob.JointAction(spacetime, _declaration(
        spacetime, matter=1.0, covariance=gamma))


def test_the_matter_term_is_the_trace_against_the_carrier():
    spacetime = _tetrahedron()
    gamma = _fixed_gamma(spacetime)
    action = _matter(spacetime, gamma)
    h = _matrix(action.carrier_operator())
    assert complex(action.matter_term()) == pytest.approx(
        np.trace(_matrix(gamma) @ h), rel=1e-13)


def test_the_length_force_is_the_derivative_at_fixed_gamma():
    """tr(Gamma dh/dz_e) against a central difference of tr(Gamma h) in z_e,
    step 1e-5, Gamma held fixed: agreement to 1e-8 relative on every edge;
    `length_stationarity` of the matter-only action is the same vector."""
    spacetime = _tetrahedron()
    gamma = _fixed_gamma(spacetime)
    action = _matter(spacetime, gamma)
    force = np.asarray(action.hellmann_feynman_length_force())
    np.testing.assert_allclose(np.asarray(action.length_stationarity()),
                               force, rtol=1e-13, atol=1e-15)
    h = 1e-5
    for index, edge in enumerate(spacetime.getEdgeList().toVector()):
        z = complex(edge.getLength()) ** 2
        values = []
        for step in (h, -h):
            edge.setLength(cmath.sqrt(z + step))
            values.append(complex(_matter(spacetime, gamma).matter_term()))
        edge.setLength(cmath.sqrt(z))
        difference = (values[0] - values[1]) / (2 * h)
        assert abs(force[index] - difference) <= 1e-8 * abs(force[index])


def test_the_link_force_is_the_maurer_cartan_derivative_at_fixed_gamma():
    """tr(Gamma U_e dh/dU_e) is d/d delta tr(Gamma h(U_e e^delta)) at
    delta = 0; U_e e^delta is the phase phi_e - i delta. Central difference
    with step 1e-5 agrees to 1e-8 relative on every edge."""
    spacetime = _tetrahedron()
    gamma = _fixed_gamma(spacetime)
    force = np.asarray(_matter(spacetime, gamma).hellmann_feynman_link_force())
    h = 1e-5
    for index, edge in enumerate(spacetime.getEdgeList().toVector()):
        phase = edge.getPhase()
        values = []
        for step in (h, -h):
            edge.setPhase(phase - 1j * step)
            values.append(complex(_matter(spacetime, gamma).matter_term()))
        edge.setPhase(phase)
        difference = (values[0] - values[1]) / (2 * h)
        assert abs(force[index] - difference) <= 1e-8 * max(abs(force[index]),
                                                            1e-3)


def test_the_length_force_obeys_the_euler_identity():
    """WP v17 line 263: the Whitney operator is homogeneous of degree -1 in
    the squared lengths, so sum_e z_e tr(Gamma dh/dz_e) = -tr(Gamma h)
    exactly, for any fixed Gamma."""
    spacetime = _tetrahedron()
    gamma = _fixed_gamma(spacetime)
    action = _matter(spacetime, gamma)
    z = np.array([complex(e.getLength()) ** 2
                  for e in spacetime.getEdgeList().toVector()])
    euler = np.sum(z * np.asarray(action.hellmann_feynman_length_force()))
    assert euler == pytest.approx(-complex(action.matter_term()), rel=1e-12)


# ------------------------------------------------------------------ Ward


def test_the_ward_current_is_divergence_free_for_invariant_terms():
    """The Villain holonomy term is gauge invariant outright, and the matter
    term is when Gamma is a spectral projector of h itself
    (`occupation_projector`): the divergence of j = U dS/dU vanishes at every
    vertex to 1e-12, at a generic complex connection."""
    spacetime = _tetrahedron()
    holonomy = cob.JointAction(spacetime, _declaration(spacetime, beta=1.3))
    assert np.max(np.abs(np.asarray(holonomy.ward_current_divergence()))) \
        < 1e-12
    matter = _matter(spacetime, _fixed_gamma(spacetime))
    assert np.max(np.abs(np.asarray(matter.ward_current_divergence()))) \
        < 1e-12
    assert np.max(np.abs(np.asarray(matter.ward_current())
                         - np.asarray(matter.link_stationarity()))) == 0.0


def test_a_gamma_that_is_not_a_spectral_projector_breaks_the_ward_identity():
    """A vertex gauge transformation acts on the edge cochains by a diagonal
    similarity G, so a diagonal Gamma leaves tr(Gamma h) invariant, but a
    generic dense fixed Gamma that does not commute with h makes the matter
    term gauge dependent: the divergence tr(Gamma [X_v, h]) is nonzero, while
    the diagonal projector onto the first two cells keeps it zero."""
    spacetime = _tetrahedron()
    diagonal = np.diag([1.0, 1.0, 0, 0, 0, 0]).astype(complex)
    assert np.max(np.abs(np.asarray(
        _matter(spacetime, diagonal).ward_current_divergence()))) < 1e-12
    rng = np.random.default_rng(5)
    gamma = rng.normal(size=(6, 6)) + 1j * rng.normal(size=(6, 6))
    action = _matter(spacetime, gamma)
    assert np.max(np.abs(np.asarray(action.ward_current_divergence()))) > 1e-3


# ---------------------------------------------------- dressed fluctuation


def _dressed(h0, couplings, second, stiffness, occupied):
    declaration = cob.DressedFluctuationDeclaration()
    n = h0.shape[0]
    declaration.carrier_dimension = n
    declaration.carrier = list(np.asarray(h0, dtype=complex).reshape(-1))
    declaration.couplings = [list(np.asarray(o, dtype=complex).reshape(-1))
                             for o in couplings]
    declaration.second_derivatives = [
        list(np.asarray(s, dtype=complex).reshape(-1)) for s in second]
    declaration.bare_stiffness = list(np.asarray(stiffness,
                                                 dtype=complex).reshape(-1))
    declaration.occupied_modes = occupied
    return cob.DressedFluctuation(declaration)


def test_the_single_particle_hole_mode():
    """Two levels 0 and Delta = 2, the lower occupied, one fluctuation with
    O = [[0, g], [g, 0]], g = 1/2, d^2h = diag(1/2, 0) and bare stiffness
    A = 1: Pi(0) = 2 g^2 / Delta = 1/4, D = 1/2, and the collective mode is at
    omega^2 = Delta^2 (1 - Pi(0) / (A + D)) = 4 (1 - 1/6) = 10/3."""
    g = 0.5
    h0 = np.diag([0.0, 2.0])
    o = np.array([[0.0, g], [g, 0.0]])
    fluctuation = _dressed(h0, [o], [np.diag([0.5, 0.0])], [[1.0]], 1)
    assert complex(fluctuation.paramagnetic()[0]) == pytest.approx(0.25,
                                                                    abs=1e-15)
    assert complex(fluctuation.diamagnetic()[0]) == pytest.approx(0.5,
                                                                  abs=1e-15)
    assert complex(fluctuation.induced_stiffness()[0]) == pytest.approx(
        0.25, abs=1e-15)
    frequencies = sorted(complex(m.frequency).real
                         for m in fluctuation.collective_modes())
    np.testing.assert_allclose(frequencies,
                               [-math.sqrt(10 / 3), math.sqrt(10 / 3)],
                               rtol=1e-12)
    for mode in fluctuation.collective_modes():
        assert mode.radiation_rate == pytest.approx(0.0, abs=1e-12)
    # A_eff(omega) = A + D - Pi(omega) vanishes at the mode
    at_mode = complex(fluctuation.dressed_stiffness(math.sqrt(10 / 3))[0])
    assert abs(at_mode) < 1e-12


def _occupied_energy(h0, couplings, second, f, occupied):
    h = h0 + sum(x * o for x, o in zip(f, couplings))
    k = 0
    for a in range(len(f)):
        for b in range(a, len(f)):
            factor = 0.5 if a == b else 1.0
            h = h + factor * f[a] * f[b] * second[k]
            k += 1
    values = np.sort_complex(np.linalg.eigvals(h))
    return np.sum(values[np.argsort(values.real)][:occupied])


def test_the_induced_stiffness_is_the_hessian_of_the_occupied_energy():
    """WP v17 line 289 and `DressedFluctuation.induced_stiffness`: D - Pi(0)
    is the Hessian of sum_m lambda_m(f) in the fluctuations, for a
    non-Hermitian 4 x 4 carrier with two occupied modes and two fluctuations
    with declared second derivatives; checked against a central second
    difference with step 1e-4 to 1e-6 relative."""
    rng = np.random.default_rng(11)
    h0 = np.diag([0.0, 0.7, 2.0, 3.1]) + 0.05 * (
        rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))
    couplings = [0.3 * (rng.normal(size=(4, 4)) + 0.1j * rng.normal(
        size=(4, 4))) for _ in range(2)]
    second = [0.2 * rng.normal(size=(4, 4)) for _ in range(3)]
    fluctuation = _dressed(h0, couplings, second, np.eye(2), 2)
    induced = _matrix(fluctuation.induced_stiffness())
    h = 1e-4
    for a in range(2):
        for b in range(2):
            total = 0
            for sa, sb, sign in ((h, h, 1), (h, -h, -1), (-h, h, -1),
                                 (-h, -h, 1)):
                f = np.zeros(2)
                f[a] += sa
                f[b] += sb
                total += sign * _occupied_energy(h0, couplings, second, f, 2)
            hessian = total / (4 * h * h)
            assert abs(induced[a, b] - hessian) <= 1e-6 * max(1.0,
                                                               abs(hessian))


def test_the_induced_stiffness_vanishes_on_a_pure_gauge_direction():
    """WP v17 line 289: along h(f1, f2) = e^{-f1 X}(h0 + f2 O2) e^{f1 X},
    X diagonal, the occupied energy does not move with f1 at any f2, so with
    O1 = [h0, X], d1d1 h = [[h0, X], X], d1d2 h = [O2, X] and d2d2 h = 0 the
    first row and column of the induced stiffness D - Pi(0) vanish, while
    the paramagnetic term alone does not; the Ward residual on the gauge
    direction (1, 0) is zero to rounding relative to ||D - Pi(0)||."""
    h0 = np.array([[0.0, 0.3, 0.1j], [0.3, 1.0, 0.2], [0.1j, 0.2, 2.5]])
    x = np.diag([0.0, 1.0, -0.5])
    o1 = h0 @ x - x @ h0
    o2 = np.array([[0.4, 0.1, 0.0], [0.1, -0.2, 0.3], [0.0, 0.3, 0.1]])
    second = [o1 @ x - x @ o1, o2 @ x - x @ o2, np.zeros((3, 3))]
    fluctuation = _dressed(h0, [o1, o2], second, np.eye(2), 1)
    induced = _matrix(fluctuation.induced_stiffness())
    assert np.max(np.abs(induced[0, :])) < 1e-13
    assert np.max(np.abs(induced[:, 0])) < 1e-13
    assert abs(induced[1, 1]) > 1e-2
    assert abs(_matrix(fluctuation.paramagnetic())[0, 0]) > 1e-2
    assert fluctuation.ward_residual([1.0, 0.0]) < 1e-12
    assert fluctuation.ward_residual([0.0, 1.0]) > 0.5
