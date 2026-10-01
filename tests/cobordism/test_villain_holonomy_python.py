# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""The Villain holonomy term of the joint action.

The holonomy term of S(z, U, Gamma) is the Villain (heat-kernel) action in its
character form, with the Villain weight summed to a declared order M (an
integer from 1 to 10, 10 by default),

    S_hol = -beta_V sum_tau log W_M(F_tau),
    W_M(F) = sum_{|m| <= M} c_m F^m,   c_m = exp(-m^2/(2 beta)),

with beta_V = beta / <m^2>_beta, where <m^2>_beta is the second moment of the
same order-M sums. W_M is a Laurent polynomial: the term is the function the
order defines, and the infinite series enters only as a reported tail. The
suite asserts, each as a measured number:

* the matching: beta_V = beta / <m^2>_beta makes the second variation at
  trivial holonomy beta L_1^up entry by entry at every order, so the
  tetrahedron's coexact block is 4 beta;
* the quarter-turn stiffness: at F = +-i the Villain curvature is
  beta_V (<m^2>_i - <m>_i^2), evaluated here from its closed form at the
  order, and the tetrahedron's coexact block is four times it;
* gauge invariance under the complex gauge group, the symmetry F <-> 1/F, and
  stationarity at trivial holonomy;
* the link stationarity (the Ward current) and the Hessian against finite
  differences at complex face holonomies, in two complex directions each, which
  is also the holomorphy of the term;
* the sums at every order against the direct sums, and an order outside 1..10
  refused;
* the reported tail against the difference between an order and order ten,
  against the heat kernel (the Poisson-resummed infinite series) on the unit
  circle, and against its own formula where the geometric bound needs listed
  terms;
* the rounding bound gamma_(6M+1) sum_m c_m |F|^m of the computed W_M against
  exact rational arithmetic, and the certificate "W_M is nonzero" following
  that bound alone at a zero of W_M;
* the logarithm: real on the unit circle where W_M is positive, exp of it
  equal to W_M, the certified radial continuation against the principal
  logarithm and against the factorization of W_M over its zeros, and no value
  on a ray through a zero of W_M or from a point of the circle where W_M is
  negative.
"""

import cmath
import math
import os
import sys
import unittest
from fractions import Fraction

import numpy as np

import tessera as T

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _joint_action_hosts import sphere3, tetrahedron  # noqa: E402

cob = T.cobordism


def _declaration(beta=1.3, **overrides):
    """A joint action of the holonomy term alone."""
    declaration = cob.JointActionDeclaration()
    declaration.carrier_degree = 1
    declaration.gravitational_weight = 0.0
    declaration.regge_form = cob.ReggeForm.Dual
    declaration.holonomy_weight = beta
    declaration.matter_weight = 0.0
    declaration.metric_source = cob.HodgeMetricSource.WhitneyPencil
    for name, value in overrides.items():
        setattr(declaration, name, value)
    return declaration


def _flux(index):
    """Complex link phases that are no pure gauge, so every F is complex."""
    return complex(0.23 * ((index % 5) - 2), 0.09 * ((index % 3) - 1))


def _chi(spacetime, seed=0):
    """A complex vertex function: the parameter of a C* gauge transformation."""
    values = {}
    for index, vertex in enumerate(spacetime.getVertexList().toVector()):
        step = index + seed
        values[int(vertex.getId())] = complex(0.31 * ((step % 7) - 3),
                                              0.17 * ((step % 4) - 1.5))
    return values


def _gauge(spacetime, chi):
    """U_xy -> g_x^-1 U_xy g_y with g = exp(i chi), on the stored phases."""
    for edge in spacetime.getEdgeList().toVector():
        source = int(edge.getSource().getId())
        target = int(edge.getTarget().getId())
        edge.setPhase(edge.getPhase() + chi[target] - chi[source])


def _multiply_link(spacetime, index, delta):
    """U_e -> U_e exp(delta) on the stored orientation: phi -> phi - i delta."""
    edge = spacetime.getEdgeList().toVector()[index]
    edge.setPhase(edge.getPhase() - 1j * delta)


def _signed_up_laplacian(spacetime):
    """d_2 d_2^T in getEdgeList() order on the stored orientations."""
    complex_ = cob.ChainComplex.fromSpacetime(spacetime)
    boundary = np.array(complex_.boundaryMatrix(2), dtype=float).reshape(
        complex_.numSimplices(1), complex_.numSimplices(2))
    index_of = {tuple(cell): position for position, cell
                in enumerate(complex_.kSimplexVertices(1))}
    rows = []
    for edge in spacetime.getEdgeList().toVector():
        source = int(edge.getSource().getId())
        target = int(edge.getTarget().getId())
        sign = 1.0 if source < target else -1.0
        rows.append(sign * boundary[index_of[tuple(sorted((source,
                                                           target)))]])
    signed = np.array(rows)
    return signed @ signed.T


#: The largest order that can be declared, and the default.
ORDER = 10
#: The unit roundoff of double precision.
UNIT_ROUNDOFF = 2.0 ** -53


def _second_moment(beta, order=ORDER):
    """<m^2>_beta of the sums over |m| <= order."""
    m = np.arange(-order, order + 1)
    weights = np.exp(-m * m / (2.0 * beta))
    return float(np.sum(m * m * weights) / np.sum(weights))


def _quarter_turn_stiffness(beta, order=ORDER):
    """kappa(i) = beta_V (<m^2>_i - <m>_i^2) of the sums over |m| <= order,
    from its closed form: only the even m = 2k enter W_M(i) and <m^2>_i, with
    the sign (-1)^k, and only the odd m = 2j + 1 enter <m>_i."""
    k = np.arange(-(order // 2), order // 2 + 1)
    alternating = (-1.0) ** np.abs(k)
    even = np.exp(-2.0 * k * k / beta)
    w_i = np.sum(alternating * even)
    second = np.sum(alternating * 4.0 * k * k * even) / w_i
    j = np.arange(0, (order + 1) // 2)
    odd = (2 * j + 1).astype(float)
    mean = 2j * np.sum((-1.0) ** j * odd * np.exp(-odd * odd / (2.0 * beta))) / w_i
    beta_v = beta / _second_moment(beta, order)
    return float(np.real(beta_v * (second - mean * mean)))


def _direct_sums(coefficients, holonomy):
    """W_M, DW_M and D^2W_M summed directly over |m| <= M from the
    coefficients c_0 .. c_M."""
    order = len(coefficients) - 1
    m = np.arange(-order, order + 1)
    weights = np.array([coefficients[abs(int(k))] for k in m]) * np.array(
        [complex(holonomy) ** int(k) for k in m])
    return (np.sum(weights), np.sum(m * weights), np.sum(m * m * weights),
            float(np.sum(np.abs(weights))))


def _heat_kernel(beta, theta):
    """The infinite series on the unit circle from its Poisson form,
    sqrt(2 pi beta) sum_n exp(-beta (theta - 2 pi n)^2 / 2), a sum of
    positive terms with no cancellation."""
    return math.sqrt(2.0 * math.pi * beta) * sum(
        math.exp(-beta * (theta - 2.0 * math.pi * n) ** 2 / 2.0)
        for n in range(-5, 6))


def _zeros(character):
    """The 2M zeros of W_M: the roots of the polynomial F^M W_M(F), whose
    coefficient of F^j is c_|j - M|."""
    order = character.order
    coefficients = character.coefficients
    return np.roots([coefficients[abs(j - order)]
                     for j in range(2 * order + 1)])


def _exact_weight(coefficients, holonomy):
    """W_M at a double-precision holonomy in exact rational arithmetic: the
    real and imaginary parts as fractions."""
    x, y = Fraction(holonomy.real), Fraction(holonomy.imag)
    norm = x * x + y * y
    forward, backward = (x, y), (x / norm, -y / norm)
    up, down = (Fraction(1), Fraction(0)), (Fraction(1), Fraction(0))
    real, imaginary = Fraction(1), Fraction(0)
    for m in range(1, len(coefficients)):
        up = (up[0] * forward[0] - up[1] * forward[1],
              up[0] * forward[1] + up[1] * forward[0])
        down = (down[0] * backward[0] - down[1] * backward[1],
                down[0] * backward[1] + down[1] * backward[0])
        coefficient = Fraction(coefficients[m])
        real += coefficient * (up[0] + down[0])
        imaginary += coefficient * (up[1] + down[1])
    return real, imaginary


def _gamma(count):
    """gamma_k = k u / (1 - k u), rounded upward as the library rounds it."""
    scaled = count * UNIT_ROUNDOFF
    return math.nextafter(scaled / (1.0 - scaled), math.inf)


def _quarter_turn_tetrahedron():
    """A tetrahedron whose four face holonomies are all +-i: the unit monopole.

    The outward fluxes are pi/2, pi/2, pi/2 and -3 pi/2, which sum to zero as
    every coboundary must and give F = i on every outward face.
    """
    spacetime = tetrahedron(squared=lambda index: 8.0)
    complex_ = cob.ChainComplex.fromSpacetime(spacetime)
    edges, faces = complex_.numSimplices(1), complex_.numSimplices(2)
    d2 = np.array(complex_.boundaryMatrix(2), dtype=float).reshape(edges, faces)
    d3 = np.array(complex_.boundaryMatrix(3), dtype=float).reshape(faces, 1)
    outward = d3[:, 0]
    target = outward * np.array([0.5, 0.5, 0.5, -1.5]) * math.pi
    phases, *_ = np.linalg.lstsq(d2.T, target, rcond=None)
    index_of = {tuple(cell): position for position, cell
                in enumerate(complex_.kSimplexVertices(1))}
    for edge in spacetime.getEdgeList().toVector():
        source = int(edge.getSource().getId())
        target_id = int(edge.getTarget().getId())
        phase = phases[index_of[tuple(sorted((source, target_id)))]]
        edge.setPhase(complex(phase if source < target_id else -phase))
    return spacetime


class TheMatchingAtTrivialHolonomyTest(unittest.TestCase):
    """beta_V = beta / <m^2>_beta, with the second moment of the order-M
    sums, makes the trivial-holonomy expansion beta L_1^up exactly at every
    order."""

    def test_the_matched_weight_is_beta_over_the_second_moment(self):
        """The second moment and the matched weight are those of the sums
        over |m| <= M, at every order and coupling, and the curvature in the
        real angle at F = 1 is then beta whatever the order."""
        for beta in (0.5, 1.0, 1.3, 2.0, 5.0):
            for order in (1, 2, 5, ORDER):
                character = cob.VillainCharacter(beta, order)
                self.assertAlmostEqual(character.second_moment,
                                       _second_moment(beta, order), places=13)
                self.assertAlmostEqual(character.matched_weight,
                                       beta / _second_moment(beta, order),
                                       places=13)
                self.assertAlmostEqual(
                    -character.second_derivative(1.0).real, beta, places=12)

    def test_the_second_moment_at_order_ten_beside_the_infinite_series(self):
        """The second moment of the infinite series is beta (1 - 4 pi^2 beta
        <n^2>) with n weighted by exp(-2 pi^2 beta n^2): 1 - 2.1e-7 at
        beta = 1, and beta to 1e-15 from beta = 2 on. The order-ten sums
        leave out the pairs beyond m = 10, the first of which carries
        2 * 121 exp(-121 / (2 beta)): 1e-24 at beta = 1, 1.8e-11 at beta = 2
        and 1.3e-3 at beta = 5, over sums of 2.5, 3.5 and 5.6. So the
        order-ten moment is the infinite one to rounding at beta = 1, is
        5e-12 below 2 at beta = 2, and is 4.999739335 at beta = 5, where the
        matched weight is 1.0000521357 and not one."""
        m = np.arange(-400, 401)

        def infinite(beta):
            weights = np.exp(-m * m / (2.0 * beta))
            return float(np.sum(m * m * weights) / np.sum(weights))

        self.assertAlmostEqual(cob.VillainCharacter(1.0).second_moment,
                               infinite(1.0), places=14)
        self.assertAlmostEqual(infinite(1.0), 0.9999997887677, places=12)
        two = cob.VillainCharacter(2.0).second_moment
        self.assertAlmostEqual(infinite(2.0) - two, 4.9e-12, delta=2e-13)
        five = cob.VillainCharacter(5.0)
        self.assertAlmostEqual(infinite(5.0), 5.0, places=12)
        self.assertAlmostEqual(five.second_moment, 4.999739335013564,
                               places=12)
        self.assertAlmostEqual(five.matched_weight, 1.0000521357152783,
                               places=12)

    def test_the_trivial_holonomy_hessian_is_beta_times_the_up_laplacian(self):
        """The Hessian at trivial holonomy is -beta d_2 d_2^T entry by entry:
        the per-face curvature beta, assembled over the faces by the signed
        incidences, with the Maurer-Cartan sign."""
        beta = 1.3
        spacetime = tetrahedron()
        villain = np.array(cob.JointAction(
            spacetime, _declaration(beta=beta)).holonomy_hessian()).reshape(6, 6)
        expected = -beta * _signed_up_laplacian(spacetime)
        self.assertLess(np.max(np.abs(villain - expected)), 1e-12)
        eigenvalues = np.sort(np.linalg.eigvalsh(-villain.real))
        self.assertLess(np.max(np.abs(eigenvalues[:3])), 1e-12)
        for value in eigenvalues[3:]:
            self.assertAlmostEqual(value, 4.0 * beta, places=11)

    def test_the_relaxation_jacobian_reads_the_same_block(self):
        """HolomorphicRelaxation's Cauchy-contour Jacobian of the link
        equations is the analytic Hessian, at a complex connection."""
        spacetime = sphere3(phase=_flux)
        action = cob.JointAction(spacetime, _declaration())
        relaxation_declaration = cob.HolomorphicRelaxationDeclaration()
        relaxation_declaration.relax_lengths = False
        relaxation_declaration.relax_links = True
        relaxation_declaration.relax_multipliers = False
        relaxation = cob.HolomorphicRelaxation(action, relaxation_declaration)
        order = relaxation.variable_count()
        jacobian = np.array(relaxation.jacobian()).reshape(order, order)
        hessian = np.array(action.holonomy_hessian()).reshape(order, order)
        self.assertLess(np.max(np.abs(jacobian - hessian)),
                        1e-9 * (1.0 + np.max(np.abs(hessian))))


class TheQuarterTurnStiffnessTest(unittest.TestCase):
    """At F = +-i the Villain stiffness is beta_V (<m^2>_i - <m>_i^2)."""

    def test_the_per_face_curvature_is_the_closed_form(self):
        """The curvature at a quarter turn is the closed form over the sums
        of the order, at order ten and at order four. At order ten it is the
        quoted 0.43091 at beta = 1/2 and 0.99796 at beta = 1, and 1.9999996
        at beta = 2. At beta = 5 the weight at F = i, 0.011739, is small
        beside the sums it is formed from, and the first pair beyond order
        ten of DW, 22 exp(-121 / 10) = 1.2e-4 at F = i, is one part in a
        hundred of it: <m>_i moves from the heat kernel's
        i beta pi / 2 = 7.854 i to 7.865 i, which raises -<m>_i^2 by 0.165,
        and the pair m = +-12 of D^2W, 288 exp(-144 / 10) = 1.6e-4, lowers
        <m^2>_i by 0.014. The curvature is 5.1552 where the infinite series
        gives 5.0000."""
        for beta in (0.5, 1.0, 1.3, 2.0, 5.0):
            for order in (4, ORDER):
                character = cob.VillainCharacter(beta, order)
                expected = _quarter_turn_stiffness(beta, order)
                for holonomy in (1j, -1j):
                    got = -character.second_derivative(holonomy)
                    self.assertAlmostEqual(got.real, expected, places=10)
                    self.assertAlmostEqual(got.imag, 0.0, places=10)
            self.assertGreater(_quarter_turn_stiffness(beta), 0.4 * beta)
        # the quoted values, at order ten
        self.assertAlmostEqual(_quarter_turn_stiffness(0.5), 0.43091, places=5)
        self.assertAlmostEqual(_quarter_turn_stiffness(1.0), 0.99796, places=5)
        self.assertAlmostEqual(_quarter_turn_stiffness(2.0), 1.9999996,
                               places=7)
        self.assertAlmostEqual(_quarter_turn_stiffness(5.0), 5.15520924875,
                               places=9)

    def test_the_monopole_tetrahedron_is_stiff(self):
        """With every face at +-i the Hessian is -kappa(i) d_2 d_2^T: the
        coexact block is 4 kappa(i) and the pure-gauge block zero."""
        beta = 1.0
        spacetime = _quarter_turn_tetrahedron()
        faces = np.array(cob.JointAction(
            spacetime, _declaration(beta=beta)).face_holonomies())
        self.assertLess(np.max(np.abs(faces ** 2 + 1.0)), 1e-12)

        villain = np.array(cob.JointAction(
            spacetime, _declaration(beta=beta)).holonomy_hessian()).reshape(6, 6)
        kappa = _quarter_turn_stiffness(beta)
        expected = -kappa * _signed_up_laplacian(spacetime)
        self.assertLess(np.max(np.abs(villain - expected)), 1e-11)
        eigenvalues = np.sort(np.linalg.eigvalsh(-villain.real))
        self.assertLess(np.max(np.abs(eigenvalues[:3])), 1e-11)
        for value in eigenvalues[3:]:
            self.assertAlmostEqual(value, 4.0 * kappa, places=10)


class TheTermIsGaugeInvariantAndEvenTest(unittest.TestCase):

    def test_every_complex_gauge_transformation_leaves_it_unchanged(self):
        spacetime = sphere3(phase=_flux)
        before = cob.JointAction(spacetime, _declaration())
        value = before.holonomy_term()
        current = np.array(before.link_stationarity())
        hessian = np.array(before.holonomy_hessian())
        _gauge(spacetime, _chi(spacetime, seed=3))
        after = cob.JointAction(spacetime, _declaration())
        self.assertLess(abs(after.holonomy_term() - value), 1e-12)
        self.assertLess(np.max(np.abs(np.array(after.link_stationarity())
                                      - current)), 1e-12)
        self.assertLess(np.max(np.abs(np.array(after.holonomy_hessian())
                                      - hessian)), 1e-12)

    def test_the_ward_current_is_conserved(self):
        spacetime = sphere3(phase=_flux)
        action = cob.JointAction(spacetime, _declaration())
        scale = np.max(np.abs(action.canonical_ward_current()))
        self.assertGreater(scale, 1e-3)
        self.assertLess(np.max(np.abs(action.ward_current_divergence())),
                        1e-12 * (1.0 + scale))

    def test_the_potential_is_even_under_inversion(self):
        character = cob.VillainCharacter(1.3)
        for holonomy in (0.8 + 0.3j, 1.4 - 0.9j, -0.7 + 0.2j, 1j):
            inverse = 1.0 / holonomy
            self.assertLess(abs(character.potential(holonomy)
                                - character.potential(inverse)), 1e-12)
            self.assertLess(abs(character.first_derivative(holonomy)
                                + character.first_derivative(inverse)), 1e-12)
            self.assertLess(abs(character.second_derivative(holonomy)
                                - character.second_derivative(inverse)), 1e-12)

    def test_reversing_the_connection_leaves_the_term_unchanged(self):
        spacetime = sphere3(phase=_flux)
        value = cob.JointAction(spacetime, _declaration()).holonomy_term()
        for edge in spacetime.getEdgeList().toVector():
            edge.setPhase(-edge.getPhase())
        reversed_value = cob.JointAction(spacetime,
                                         _declaration()).holonomy_term()
        self.assertLess(abs(reversed_value - value), 1e-12)

    def test_trivial_holonomy_is_stationary(self):
        spacetime = sphere3()
        action = cob.JointAction(spacetime, _declaration())
        self.assertLess(np.max(np.abs(action.link_stationarity())), 1e-15)
        # the value is -beta_V times the face count times log W(1)
        character = cob.VillainCharacter(1.3)
        expected = -character.matched_weight * 10 * math.log(
            character.series(1.0).value.real)
        self.assertAlmostEqual(abs(action.holonomy_term() - expected), 0.0,
                               places=12)


class TheDerivativesAgainstFiniteDifferencesTest(unittest.TestCase):
    """At complex face holonomies, in two complex directions each."""

    STEP = 1e-5

    def _faces_are_complex(self, action):
        faces = np.array(action.face_holonomies())
        self.assertGreater(np.max(np.abs(np.abs(faces) - 1.0)), 0.05)

    def test_the_link_stationarity_is_the_derivative_of_the_value(self):
        spacetime = sphere3(phase=_flux)
        action = cob.JointAction(spacetime, _declaration())
        self._faces_are_complex(action)
        analytic = np.array(action.link_stationarity())
        for index in range(len(analytic)):
            for direction in (1.0, 1j):
                h = self.STEP * direction
                _multiply_link(spacetime, index, h)
                plus = cob.JointAction(spacetime, _declaration()).holonomy_term()
                _multiply_link(spacetime, index, -2.0 * h)
                minus = cob.JointAction(spacetime,
                                        _declaration()).holonomy_term()
                _multiply_link(spacetime, index, h)
                difference = (plus - minus) / (2.0 * h)
                self.assertLess(abs(difference - analytic[index]),
                                1e-8 * (1.0 + abs(analytic[index])))

    def test_the_hessian_is_the_derivative_of_the_link_stationarity(self):
        spacetime = sphere3(phase=_flux)
        action = cob.JointAction(spacetime, _declaration())
        order = action.edge_count()
        hessian = np.array(action.holonomy_hessian()).reshape(order, order)
        for index in range(order):
            for direction in (1.0, 1j):
                h = self.STEP * direction
                _multiply_link(spacetime, index, h)
                plus = np.array(cob.JointAction(
                    spacetime, _declaration()).link_stationarity())
                _multiply_link(spacetime, index, -2.0 * h)
                minus = np.array(cob.JointAction(
                    spacetime, _declaration()).link_stationarity())
                _multiply_link(spacetime, index, h)
                column = (plus - minus) / (2.0 * h)
                self.assertLess(np.max(np.abs(column - hessian[:, index])),
                                1e-8 * (1.0 + np.max(np.abs(hessian))))


class TheLogarithmBranchTest(unittest.TestCase):

    def test_the_logarithm_is_real_on_the_unit_circle_and_inverts_w(self):
        character = cob.VillainCharacter(0.8)
        for angle in np.linspace(-math.pi, math.pi, 13):
            # exp(i angle) has modulus one to rounding, so the radial leg of
            # the continuation is of rounding length
            value = character.logarithm(cmath.exp(1j * angle))
            self.assertLess(abs(value.imag), 1e-14)
        for holonomy in (0.5 + 0.1j, 2.5 - 1.0j, -1.1 + 0.05j, 0.3j, -0.2 - 1e-3j):
            logarithm = character.logarithm(holonomy)
            series = character.series(holonomy).value
            self.assertLess(abs(cmath.exp(logarithm) - series),
                            1e-12 * abs(series))

    def test_the_logarithm_is_continuous_across_the_negative_axis_inside(self):
        """Between the two zeros of W_M nearest the unit circle on the
        negative real axis, at -q and -1/q with q = exp(-1 / 1.6) = 0.535 to
        within the tail, the branch has no cut."""
        character = cob.VillainCharacter(0.8)
        above = character.logarithm(-1.2 + 1e-9j)
        below = character.logarithm(-1.2 - 1e-9j)
        self.assertLess(abs(above - below), 1e-7)

    def test_a_zero_of_w_is_refused(self):
        """The infinite series vanishes at -q and -1/q, q = exp(-1 / 1.6).
        At beta = 0.8 the order-ten sum differs from it there by its tail,
        2 exp(-121 / 1.6) (1/q)^11 = 3e-30, far below the rounding bound
        6.8e-15 times the sum of the moduli of the terms, so W_M is not
        certified nonzero at either point: the ratio DW_M / W_M has no value
        at -q, and the radial path from -1 to -1/q ends on the zero, where
        no step is certified."""
        beta = 0.8
        q = math.exp(-1.0 / (2.0 * beta))
        character = cob.VillainCharacter(beta)
        for point in (complex(-q, 0.0), complex(-1.0 / q, 0.0)):
            series = character.series(point)
            self.assertLess(abs(series.value), series.rounding_bound)
            self.assertFalse(cob.VillainCharacter.certified_nonzero(series))
        with self.assertRaisesRegex(ValueError, "meets a zero of W_M"):
            character.logarithm(complex(-1.0 / q, 0.0))
        with self.assertRaisesRegex(ValueError, "not certified nonzero"):
            character.first_derivative(complex(-q, 0.0))


class TheRoundingBoundCertifiesTheWeightTest(unittest.TestCase):
    """The computed W_M differs from the exact one by at most
    gamma_(6M+1) A(F), with A(F) = sum_{|m| <= M} c_m |F|^m and
    gamma_k = k u / (1 - k u), u = 2^-53 (`VillainCharacter`: 6M roundings on
    the monomial c_M F^-M, the longest chain of the sum of n = 2M + 1
    monomials, and one unit for gradual underflow). W_M is certified nonzero
    when the modulus of the computed sum exceeds that bound, and by nothing
    else."""

    POINTS = (1.0 + 0j, 1j, -1.0 + 0j, 0.8 + 0.3j, 1.4 - 0.9j, -0.7 + 0.2j,
              2.0 - 1.5j, 0.2 + 0.1j, -3.1 + 0.4j, 1e-3 + 2e-3j,
              40.0 - 9.0j, cmath.exp(2.4j), 0.5 * cmath.exp(-2.4621j))

    def test_the_bound_is_gamma_times_the_sum_of_the_moduli(self):
        """`rounding_bound` is gamma_(6M+1) times `magnitude`, each rounded
        upward, to the last bit; `magnitude` is not below the exact sum of
        the moduli of the terms and is within 1e-13 of it."""
        for beta in (0.5, 1.0, 5.0):
            for order in range(1, ORDER + 1):
                character = cob.VillainCharacter(beta, order)
                for holonomy in self.POINTS:
                    series = character.series(holonomy)
                    self.assertEqual(
                        series.rounding_bound,
                        math.nextafter(_gamma(6 * order + 1)
                                       * series.magnitude, math.inf))
                    modulus = abs(holonomy)
                    exact = sum(c * (modulus ** m + modulus ** -m)
                                for m, c in enumerate(character.coefficients)
                                if m) + 1.0
                    self.assertGreaterEqual(series.magnitude,
                                            exact * (1.0 - 1e-13))
                    self.assertLessEqual(series.magnitude,
                                         exact * (1.0 + 1e-13))

    def test_the_bound_holds_against_exact_arithmetic(self):
        """W_M at a double-precision holonomy, with the double-precision
        coefficients the class reports, is a rational number. Evaluated in
        exact rational arithmetic, it differs from the computed sum by no
        more than the rounding bound at every order, coupling and point, and
        the largest ratio of the error to the bound read here is above 0.01:
        the bound is the size of the rounding it bounds, not a margin over
        it."""
        largest = 0.0
        for beta in (0.5, 1.0, 5.0):
            for order in range(1, ORDER + 1):
                character = cob.VillainCharacter(beta, order)
                coefficients = list(character.coefficients)
                self.assertEqual(coefficients, [
                    math.exp(-m * m / (2.0 * beta))
                    for m in range(order + 1)])
                for holonomy in self.POINTS:
                    series = character.series(holonomy)
                    real, imaginary = _exact_weight(coefficients, holonomy)
                    error = (Fraction(series.value.real) - real) ** 2 + (
                        Fraction(series.value.imag) - imaginary) ** 2
                    bound = Fraction(series.rounding_bound) ** 2
                    self.assertLessEqual(error, bound)
                    largest = max(largest, float(error / bound) ** 0.5)
        self.assertGreater(largest, 0.01)

    def test_the_certificate_follows_the_bound_at_a_zero_of_the_weight(self):
        """At beta = 1 the order-ten sum has a zero on the negative real axis
        beside -exp(1/2) = -1.6487, the first zero of the infinite series
        (the tail there, 2 exp(-121 / 2) 1.6487^11 = 2.6e-24, displaces it
        by less than a rounding). The zero is read from the roots of the
        polynomial F^10 W_10(F). There the two largest terms, the constant
        1 and c_1 / q = 1 with the opposite sign, cancel, and the moduli of
        the terms add up to A = 2.840. At the double nearest the zero the
        computed sum is rounding, about 3e-16, below the bound
        gamma_61 A = 6.77e-15 * 2.840 = 1.92e-14, and W_M is not certified
        nonzero. Displaced along the axis by a relative delta, W_M is
        |DW_M| delta = 0.128 delta, so the certificate is off at
        delta = 1e-13 (1.3e-14, below the bound) and on at delta = 1e-12
        (1.3e-13, above it). At every displacement the certificate equals
        the comparison of the computed modulus with the bound, and the
        derivatives are refused exactly where it fails."""
        character = cob.VillainCharacter(1.0)
        zeros = _zeros(character)
        zero = min(zeros, key=lambda z: abs(z + math.exp(0.5)))
        self.assertLess(abs(zero.imag), 1e-12)
        self.assertAlmostEqual(zero.real, -math.exp(0.5), places=12)
        at_zero = character.series(complex(zero.real, 0.0))
        self.assertAlmostEqual(at_zero.magnitude, 2.840, delta=1e-3)
        self.assertAlmostEqual(at_zero.rounding_bound, 1.92e-14, delta=1e-16)
        self.assertLess(abs(at_zero.value), 1e-15)
        self.assertAlmostEqual(abs(at_zero.first), 0.128, delta=1e-3)
        self.assertFalse(cob.VillainCharacter.certified_nonzero(at_zero))
        with self.assertRaisesRegex(ValueError, "not certified nonzero"):
            character.second_derivative(complex(zero.real, 0.0))
        verdicts = {}
        for exponent in range(4, 17):
            delta = 10.0 ** -exponent
            point = complex(zero.real * (1.0 + delta), 0.0)
            series = character.series(point)
            certified = cob.VillainCharacter.certified_nonzero(series)
            self.assertEqual(certified,
                             abs(series.value) > series.rounding_bound)
            verdicts[exponent] = certified
            if certified:
                self.assertAlmostEqual(
                    abs(series.value) / delta, 0.128, delta=1e-3)
                character.first_derivative(point)
            else:
                with self.assertRaisesRegex(ValueError,
                                            "not certified nonzero"):
                    character.first_derivative(point)
        self.assertEqual([e for e in range(4, 17) if verdicts[e]],
                         list(range(4, 13)))

    def test_a_sum_that_overflows_is_not_certified(self):
        """At |F| = 1e40 the term F^10 exceeds the largest double, the
        computed sum is not finite, its rounding bound is reported as
        infinite, and the weight is not certified nonzero there."""
        series = cob.VillainCharacter(1.0).series(1e40 + 0j)
        value = complex(series.value)
        self.assertFalse(math.isfinite(value.real)
                         and math.isfinite(value.imag))
        self.assertEqual(series.rounding_bound, math.inf)
        self.assertFalse(cob.VillainCharacter.certified_nonzero(series))


class TheCertifiedContinuationTest(unittest.TestCase):
    """`logarithm` continues log W_M from the unit circle along the ray of
    the holonomy by steps each certified by the bound
    |DW_M| L + B L^2 / 2 + u_a + u_b < |W_M(a)| (`VillainCharacter`), with no
    step constant and no count of steps."""

    def test_the_continuation_is_the_principal_logarithm_where_w_stays_right(
            self):
        """At beta = 1 and holonomies of argument at most 0.6 and modulus
        between 0.7 and 1.5, W_M on the ray stays in the half-plane
        Re W > 0 (its real part is at least 1.6 there against an imaginary
        part of at most 1.1), so the logarithm continued from the real value
        on the unit circle is the principal logarithm of W_M at the
        holonomy, to 1e-14."""
        character = cob.VillainCharacter(1.0)
        for modulus in (0.7, 0.9, 1.0, 1.2, 1.5):
            for angle in (-0.6, -0.2, 0.0, 0.3, 0.6):
                holonomy = cmath.rect(modulus, angle)
                for t in np.linspace(0.0, 1.0, 21):
                    on_ray = character.series(
                        cmath.rect(modulus ** float(t), angle)).value
                    self.assertGreater(on_ray.real, 1.6)
                    self.assertLess(abs(on_ray.imag), 1.1)
                value = character.series(holonomy).value
                self.assertLess(
                    abs(character.logarithm(holonomy) - cmath.log(value)),
                    1e-14)

    def test_the_continuation_is_the_sum_over_the_zeros(self):
        """W_M(F) = c_M F^-M prod_k (F - zeta_k) over its 2M zeros, and the
        ray from F/|F| to F is a straight segment, which subtends an angle
        below pi at every zero off it. So the logarithm continued along the
        ray is log W_M(F/|F|) - M log|F| + sum_k Log((F - zeta_k) /
        (F/|F| - zeta_k)) with principal logarithms. The continuation agrees
        with it to 1e-9 (the zeros are numpy's, good to about 1e-12 relative
        to their moduli) at holonomies where it differs from the principal
        logarithm of W_M by a multiple of 2 pi i: at beta = 5,
        F = exp(i pi / 2) / 2 it is -3.5106 - 5.6358 i, the principal value
        less 2 pi i."""
        cases = ((5.0, ORDER, 0.5 * cmath.exp(0.5j * math.pi), -1),
                 (5.0, ORDER, 2.0 * cmath.exp(0.5j * math.pi), 1),
                 (1.0, ORDER, cmath.rect(0.2, 3.0), -1),
                 (1.0, ORDER, cmath.rect(3.0, 2.5), 0),
                 (10.0, ORDER, cmath.rect(1.5, 1.2), 1),
                 (0.5, 3, cmath.rect(8.0, 1.0), 0),
                 (1.3, 6, cmath.rect(0.3, -2.0), 0))
        for beta, order, holonomy, winding in cases:
            with self.subTest(beta=beta, order=order, holonomy=holonomy):
                character = cob.VillainCharacter(beta, order)
                start = holonomy / abs(holonomy)
                reference = cmath.log(character.series(start).value.real) \
                    - order * math.log(abs(holonomy))
                for zero in _zeros(character):
                    reference += cmath.log((holonomy - zero) / (start - zero))
                logarithm = character.logarithm(holonomy)
                self.assertLess(abs(logarithm - reference),
                                1e-9 * max(1.0, abs(reference)))
                principal = cmath.log(character.series(holonomy).value)
                self.assertAlmostEqual(
                    (logarithm - principal).imag / (2.0 * math.pi), winding,
                    places=9)
        self.assertLess(abs(cob.VillainCharacter(5.0).logarithm(
            0.5 * cmath.exp(0.5j * math.pi)) - (-3.5106482278 - 5.6358450958j)),
            1e-9)

    def test_a_ray_through_a_zero_of_the_weight_has_no_logarithm(self):
        """The zeros of W_M are read from the roots of F^M W_M(F). For a
        zero zeta off the unit circle, at whose point of the circle W_M is
        positive, the holonomy 1.3 zeta lies beyond it on its ray: every
        candidate step from before the first zero on the ray fails the
        certificate, the halving ends where the geometric mean of the two
        scales is one of them in double precision, and `logarithm` raises
        naming the zero. Rotated off the ray by +-1e-3 radians the path
        passes the zeros of the ray at a distance of 1e-3 of their moduli
        and the logarithm has a value on both sides; the two values differ
        by 2 pi i for every zero of W_M between the two rays and the
        holonomy (one at beta = 1, 2 and 5, whose zero is off the real
        axis; four at beta = 0.8 and order four, whose zero -44.5 is the
        outermost of the four on the negative real axis), to within the
        2e-3 change of the endpoint."""
        for beta, order, enclosed in ((1.0, ORDER, 1), (0.8, 4, 4),
                                      (2.0, ORDER, 1), (5.0, ORDER, 1)):
            character = cob.VillainCharacter(beta, order)
            zeros = [complex(z) for z in _zeros(character)]
            zero = next(
                z for z in zeros if abs(z) > 1.05
                and character.series(z / abs(z)).value.real > 1e-3)
            with self.subTest(beta=beta, order=order, zero=zero):
                holonomy = 1.3 * zero
                self.assertEqual(sum(
                    1 for z in zeros
                    if 1.0 < abs(z) < abs(holonomy)
                    and abs(cmath.phase(z / zero)) < 1e-3), enclosed)
                with self.assertRaisesRegex(ValueError,
                                            "meets a zero of W_M"):
                    character.logarithm(holonomy)
                above = character.logarithm(holonomy * cmath.exp(1e-3j))
                below = character.logarithm(holonomy * cmath.exp(-1e-3j))
                self.assertAlmostEqual(
                    abs((above - below).imag) / (2.0 * math.pi), enclosed,
                    delta=5e-3)
                self.assertLess(abs((above - below).real), 5e-2)

    def test_where_the_weight_is_negative_on_the_circle_there_is_no_branch(
            self):
        """W_M is real on the unit circle and differs from the positive heat
        kernel by at most its tail. At beta = 10 and order ten the tail
        bound is 6.9e-3 while the heat kernel at theta = 0.5361 pi is
        5.5e-6, and W_M there is -4.377e-3: no logarithm real on the unit
        circle exists on that ray, at the circle or off it. At
        theta = pi the same sum is +3.561e-3 (the heat kernel is 5.9e-21),
        and its logarithm is log 3.561e-3 = -5.6378."""
        character = cob.VillainCharacter(10.0)
        theta = 0.5361 * math.pi
        series = character.series(cmath.exp(1j * theta))
        self.assertAlmostEqual(series.value.real, -4.377e-3, delta=1e-6)
        self.assertAlmostEqual(_heat_kernel(10.0, theta), 5.5e-6, delta=1e-7)
        self.assertAlmostEqual(series.value_tail, 6.9e-3, delta=1e-4)
        self.assertLessEqual(abs(series.value.imag), series.rounding_bound)
        for modulus in (1.0, 0.8, 1.25):
            with self.assertRaisesRegex(ValueError,
                                        "does not exceed its bound"):
                character.logarithm(modulus * cmath.exp(1j * theta))
        # the derivatives use W_M itself and are evaluated there
        self.assertTrue(cob.VillainCharacter.certified_nonzero(series))
        character.first_derivative(cmath.exp(1j * theta))
        minus_one = character.series(-1.0 + 0j)
        self.assertAlmostEqual(minus_one.value.real, 3.561e-3, delta=1e-6)
        self.assertLess(_heat_kernel(10.0, math.pi), 1e-20)
        self.assertAlmostEqual(character.logarithm(-1.0 + 0j).real,
                               math.log(minus_one.value.real), places=13)
        self.assertAlmostEqual(character.logarithm(-1.0 + 0j).real, -5.6378,
                               places=4)

    def test_the_imaginary_part_on_the_circle_is_within_the_rounding_bound(
            self):
        """On the unit circle W_M = 1 + 2 sum_m c_m cos(m theta) is real, so
        the imaginary part of the computed sum is rounding: at every order,
        coupling and angle read here it is within the rounding bound of the
        sum, which is the test `logarithm` applies at the start of its
        path (with the bound on the change of W_M between the circle and
        the computed start added)."""
        for beta in (0.5, 1.0, 5.0, 10.0):
            for order in (1, 4, ORDER):
                character = cob.VillainCharacter(beta, order)
                for angle in np.linspace(-math.pi, math.pi, 41):
                    series = character.series(cmath.exp(1j * float(angle)))
                    self.assertLessEqual(abs(series.value.imag),
                                         series.rounding_bound)


class TheLogarithmWhereTheWeightIsSmallTest(unittest.TestCase):
    """At beta = 5 the held face holonomies of the kappa, beta scan passed
    arguments of about 0.76 pi and 0.78 pi, where the heat kernel is 1.5e-6
    to 3.9e-6 against sums of moduli of 5.6. The order-ten sum differs from
    the heat kernel by up to its tail there, 1.24e-5 = 2 exp(-121 / 10) /
    (1 - exp(-23 / 10)), which is larger than the heat kernel: W_M is
    5.83e-6 at -0.7837 pi and negative, -2.57e-7 and -5.01e-7, at
    -0.7595 pi and 0.7581 pi (it has zeros on the unit circle at 0.7322 pi
    and 0.7608 pi and is negative between them). The logarithm is read
    where W_M is positive, against the rounding bound of the sum, 3.8e-14,
    and has no value where W_M is negative. The action's value, the one
    quantity that needs log W_M, is reported as unavailable by name where
    the logarithm has no value, and a solve goes on."""

    BETA = 5.0
    #: The arguments over pi of the held face holonomies at which the kappa,
    #: beta scan's solves at beta = 5 read W of 1.5e-6 to 3.9e-6 (probes 14
    #: and 15 of the investigation), with W_M at order ten at each.
    ARGUMENTS = ((-0.783656641, 5.8334763e-06), (-0.759478396, -2.5708e-07),
                 (0.758148418, -5.0148e-07))

    def test_the_small_weight_on_the_circle_is_within_its_tail_of_the_kernel(
            self):
        character = cob.VillainCharacter(self.BETA)
        for over_pi, expected in self.ARGUMENTS:
            with self.subTest(argument_over_pi=over_pi):
                theta = math.pi * over_pi
                series = character.series(cmath.exp(1j * theta))
                kernel = _heat_kernel(self.BETA, theta)
                self.assertLess(kernel, 5e-6)
                self.assertAlmostEqual(series.value.real, expected,
                                       delta=1e-10)
                self.assertAlmostEqual(series.value_tail, 1.2358e-05,
                                       delta=1e-9)
                self.assertLess(abs(series.value.real - kernel),
                                series.value_tail)
                self.assertAlmostEqual(series.rounding_bound, 3.8e-14,
                                       delta=1e-15)
                self.assertLessEqual(abs(series.value.imag),
                                     series.rounding_bound)

    def test_the_logarithm_is_read_where_the_weight_is_positive(self):
        """At -0.7837 pi the logarithm on the circle is log 5.83e-6 =
        -12.0519, to the 1e-8 the sum is resolved to (its rounding bound
        3.8e-14 over its value), and off the circle along the same ray,
        inside and outside it, the continuation inverts W_M. The argument
        is given as a double, so exp(i theta) has modulus one only to
        rounding and the logarithm carries an imaginary part of that
        size."""
        character = cob.VillainCharacter(self.BETA)
        over_pi, expected = self.ARGUMENTS[0]
        holonomy = cmath.exp(1j * math.pi * over_pi)
        logarithm = character.logarithm(holonomy)
        self.assertAlmostEqual(logarithm.real, math.log(expected), delta=1e-7)
        self.assertAlmostEqual(logarithm.real, -12.0519, places=4)
        self.assertLess(abs(logarithm.imag), 1e-8)
        for modulus in (0.5, 2.0):
            with self.subTest(modulus=modulus):
                off = modulus * holonomy
                series = character.series(off).value
                self.assertLess(
                    abs(cmath.exp(character.logarithm(off)) - series),
                    1e-10 * abs(series))

    def test_the_logarithm_has_no_value_where_the_weight_is_negative(self):
        character = cob.VillainCharacter(self.BETA)
        for over_pi, _ in self.ARGUMENTS[1:]:
            for modulus in (1.0, 0.5, 2.0):
                with self.subTest(argument_over_pi=over_pi, modulus=modulus):
                    with self.assertRaisesRegex(ValueError,
                                                "does not exceed its bound"):
                        character.logarithm(
                            modulus * cmath.exp(1j * math.pi * over_pi))

    def test_minus_one_has_its_logarithm_at_order_ten(self):
        """At F = -1 the sum alternates, 1 + 2 sum_m (-1)^m c_m, and the
        first pair beyond order ten enters the infinite series with the
        sign -1, so W_M(-1) is the heat kernel plus about 2 c_11: at
        beta = 5 the heat kernel is 2.2e-10 and W_M(-1) = 1.009e-5 (2 c_11 =
        1.115e-5, less 2 c_12 = 1.1e-6), with logarithm -11.504."""
        character = cob.VillainCharacter(5.0)
        self.assertAlmostEqual(_heat_kernel(5.0, math.pi), 2.157e-10,
                               delta=1e-13)
        value = character.series(-1.0 + 0j).value.real
        self.assertAlmostEqual(value, 1.009014e-05, delta=1e-11)
        self.assertAlmostEqual(
            value, 2.0 * math.exp(-12.1) - 2.0 * math.exp(-14.4), delta=3e-7)
        logarithm = character.logarithm(-1.0 + 0j)
        self.assertEqual(logarithm.imag, 0.0)
        self.assertAlmostEqual(logarithm.real, math.log(value), places=12)
        self.assertAlmostEqual(logarithm.real, -11.504, places=3)

    def test_a_logarithm_without_a_value_leaves_the_value_unavailable_not_the_solve(
            self):
        """A tetrahedron one of whose links has the phase 0.5361 pi has two
        face holonomies of argument +-0.5361 pi, where W_M at beta = 10 and
        order ten is -4.377e-3. The action's value has no logarithm there
        and is reported as unavailable, with the reason; a solve of the
        squared lengths and the multiplier (whose equations do not need
        log W_M), the matter term tr(h) under the constraint p_1(h) = p_1*
        read at squared length 8.5, runs to its stationary point (the
        constraint met, xi = -w_M) and reports its action as unavailable by
        name at every step."""
        phase = 0.5361 * math.pi

        def host(squared):
            return tetrahedron(squared=lambda index: squared,
                               phase=lambda index: phase if index == 0
                               else 0.0)

        def constrained(spacetime, target):
            declaration = _declaration(beta=10.0, matter_weight=1.0)
            declaration.covariance = list(np.eye(6, dtype=complex).reshape(-1))
            declaration.moment_constraints = [
                cob.SpectralMomentConstraint(1, target, 0j)]
            return cob.JointAction(spacetime, declaration)

        target = constrained(host(8.5), 0j).power_sums()[0]
        spacetime = host(8.0)
        action = constrained(spacetime, target)
        faces = np.asarray(action.face_holonomies())
        self.assertEqual(int(np.sum(
            np.abs(np.abs(np.angle(faces)) - phase) < 1e-12)), 2)
        with self.assertRaisesRegex(ValueError, "does not exceed its bound"):
            action.value()
        reported = action.reported_value()
        self.assertFalse(reported.available)
        self.assertTrue(math.isnan(reported.value.real))
        self.assertIn("does not exceed its bound", reported.unavailable)

        declaration = cob.HolomorphicRelaxationDeclaration()
        declaration.relax_lengths = True
        declaration.relax_links = False
        declaration.relax_multipliers = True
        declaration.tolerance = 1e-12
        report = cob.HolomorphicRelaxation(action, declaration).solve()
        self.assertTrue(report.converged)
        self.assertEqual(report.stop_reason, cob.RelaxationStop.Converged)
        self.assertFalse(report.action_available)
        self.assertIn("does not exceed its bound", report.action_unavailable)
        self.assertGreater(len(report.steps), 0)
        for step in report.steps:
            self.assertFalse(step.action_available)
        self.assertLess(abs(report.moment_residuals[0]), 1e-10)
        self.assertAlmostEqual(abs(report.multipliers[0] + 1.0), 0.0,
                               places=8)


class TheSolveAwayFromTheZerosTest(unittest.TestCase):
    """The equations are posed where W is certified nonzero; a solve that
    stays there is not damped by anything but the residual test."""

    def test_a_clear_solve_converges_with_no_damping_but_the_residual_test(
            self):
        spacetime = sphere3(phase=_flux)
        action = cob.JointAction(spacetime, _declaration())
        relaxation_declaration = cob.HolomorphicRelaxationDeclaration()
        relaxation_declaration.relax_lengths = False
        relaxation_declaration.relax_links = True
        relaxation_declaration.relax_multipliers = False
        relaxation_declaration.tolerance = 1e-11
        report = cob.HolomorphicRelaxation(
            action, relaxation_declaration).solve()
        self.assertTrue(report.converged)
        self.assertEqual(report.sector_guard_damped_steps, 0)
        self.assertEqual(sum(step.domain_guard_dampings
                             for step in report.steps), 0)


class TheOrderIsDeclaredAndTheTailReportedTest(unittest.TestCase):
    """The Villain weight is summed to a declared order M from one to ten:
    the sums keep exactly |m| <= M at every holonomy. What the order leaves
    out of the infinite series is reported as a bound with every evaluation
    and enters nothing."""

    POINTS = (1.0 + 0j, 1j, 1.1 + 0.05j, 0.7 + 0.4j, 2.0 - 1.5j, 0.2 + 0.1j,
              -2.2e-3 - 4.3e-4j, 29.8 - 12.8j)

    def test_an_order_outside_one_to_ten_is_refused(self):
        self.assertEqual(cob.VillainCharacter.maximum_order, 10)
        self.assertEqual(cob.VillainCharacter(1.3).order, 10)
        for order in range(1, 11):
            self.assertEqual(cob.VillainCharacter(1.3, order).order, order)
        for order in (0, 11, -1, 100000):
            with self.assertRaisesRegex(
                    ValueError, "the order of the Villain weight is an "
                                "integer from 1 to 10; got %d" % order):
                cob.VillainCharacter(1.3, order)
        with self.assertRaises(TypeError):
            cob.VillainCharacter(1.3, 2.5)
        with self.assertRaises(ValueError):
            cob.VillainCharacter(0.0)

    def test_the_sums_at_every_order_are_the_direct_sums(self):
        """At every order from one to ten the three sums are the direct sums
        over |m| <= M of c_m F^m, m c_m F^m and m^2 c_m F^m with the
        coefficients c_m = exp(-m^2 / (2 beta)) the class reports: W_M to its
        rounding bound, the other two to 1e-13 of the sums of the moduli of
        their terms, which are the reported magnitudes to 1e-13. The series
        reports its order at every holonomy, near the unit circle and far
        from it."""
        for beta in (0.5, 1.3, 5.0):
            for order in range(1, ORDER + 1):
                character = cob.VillainCharacter(beta, order)
                coefficients = list(character.coefficients)
                self.assertEqual(len(coefficients), order + 1)
                for m, coefficient in enumerate(coefficients):
                    self.assertEqual(coefficient,
                                     math.exp(-m * m / (2.0 * beta)))
                for holonomy in self.POINTS:
                    series = character.series(holonomy)
                    self.assertEqual(series.order, order)
                    value, first, second, magnitude = _direct_sums(
                        coefficients, holonomy)
                    self.assertLessEqual(
                        abs(series.value - value),
                        series.rounding_bound + 1e-14 * magnitude)
                    self.assertAlmostEqual(series.magnitude / magnitude, 1.0,
                                           places=13)
                    self.assertLessEqual(abs(series.first - first),
                                         1e-13 * series.first_magnitude)
                    self.assertLessEqual(abs(series.second - second),
                                         1e-13 * series.second_magnitude)
                    self.assertLessEqual(abs(series.first),
                                         series.first_magnitude)
                    self.assertLessEqual(abs(series.second),
                                         series.second_magnitude)

    def test_the_tail_bounds_the_difference_from_order_ten(self):
        """For M below ten, the order-ten sum less the order-M sum is the
        part of the infinite series between the two orders, so its modulus
        is within the order-M tail bound (and so is the same difference of
        DW and of D^2W within theirs), to the rounding of the two sums. At
        F = 1 every term is positive: the difference is at least the first
        pair beyond the order, 2 c_(M+1), and the bound is that pair over
        1 - rho_0 with rho_0 = q^(2M+3), q = exp(-1 / (2 beta)), so the
        bound is loose there by at most 1 / (1 - rho_0)."""
        for beta in (0.5, 1.3, 5.0):
            ten = cob.VillainCharacter(beta, ORDER)
            q = math.exp(-1.0 / (2.0 * beta))
            for order in range(1, ORDER):
                character = cob.VillainCharacter(beta, order)
                for holonomy in self.POINTS:
                    low = character.series(holonomy)
                    high = ten.series(holonomy)
                    rounding = low.rounding_bound + high.rounding_bound
                    self.assertLessEqual(abs(high.value - low.value),
                                         low.value_tail + rounding)
                    self.assertLessEqual(
                        abs(high.first - low.first),
                        low.first_tail + 1e-13 * high.first_magnitude)
                    self.assertLessEqual(
                        abs(high.second - low.second),
                        low.second_tail + 1e-13 * high.second_magnitude)
                at_one = character.series(1.0 + 0j)
                pair = 2.0 * math.exp(-(order + 1) ** 2 / (2.0 * beta))
                ratio = q ** (2 * order + 3)
                self.assertAlmostEqual(
                    at_one.value_tail / (pair / (1.0 - ratio)), 1.0,
                    places=12)
                difference = (ten.series(1.0 + 0j).value
                              - at_one.value).real
                self.assertGreaterEqual(difference,
                                        pair * (1.0 - 1e-12) - 1e-15)

    def test_the_tail_bounds_the_distance_from_the_heat_kernel(self):
        """On the unit circle the infinite series is the heat kernel
        sqrt(2 pi beta) sum_n exp(-beta (theta - 2 pi n)^2 / 2). At every
        order and angle the order-M sum is within its tail bound of it (to
        1e-14 of the sum of the moduli, the rounding of the two
        evaluations). The bound does not depend on the angle. At order ten
        it is 2 exp(-121 / (2 beta)) / (1 - exp(-23 / (2 beta))): 1.06e-26
        at beta = 1, 1.236e-5 at beta = 5 and 6.90e-3 at beta = 10, which
        over the sums of the moduli of the terms, 2.507, 5.605 and 7.920, is
        4.24e-27, 2.20e-6 and 8.71e-4."""
        for beta, tail, relative in ((1.0, 1.062e-26, 4.24e-27),
                                     (5.0, 1.236e-05, 2.20e-06),
                                     (10.0, 6.90e-03, 8.71e-04)):
            for order in range(1, ORDER + 1):
                character = cob.VillainCharacter(beta, order)
                for theta in np.linspace(0.0, math.pi, 25):
                    series = character.series(cmath.exp(1j * float(theta)))
                    self.assertLessEqual(
                        abs(_heat_kernel(beta, float(theta))
                            - series.value.real),
                        series.value_tail + 1e-14 * series.magnitude)
            ten = cob.VillainCharacter(beta).series(1.0 + 0j)
            self.assertAlmostEqual(
                ten.value_tail, 2.0 * math.exp(-121.0 / (2.0 * beta))
                / (1.0 - math.exp(-23.0 / (2.0 * beta))),
                delta=1e-12 * ten.value_tail)
            self.assertAlmostEqual(ten.value_tail / tail, 1.0, places=3)
            self.assertAlmostEqual(ten.value_tail / ten.magnitude / relative,
                                   1.0, places=2)
            self.assertAlmostEqual(
                cob.VillainCharacter(beta).series(
                    cmath.exp(2.0j)).value_tail / ten.value_tail, 1.0,
                places=12)

    def test_the_tail_lists_terms_where_the_geometric_ratio_is_not_below_one(
            self):
        """With r = max(|F|, 1/|F|), the terms 2 q^(m^2) r^m beyond the order
        have ratios bounded by rho_0(N) = q^(2N+3) r past the index N. At
        beta = 5 and |F| = 20, rho_0(10) = 20 exp(-23 / 10) = 2.005 is not
        below one; rho_0(N) < 1 needs 2N + 3 > 10 log 20 = 29.96, so the
        least such index is N = 14, and the bound is the four terms
        m = 11 .. 14 listed plus the geometric bound from m = 15,
        2 t_15 / (1 - rho_0(14)), rho_0(14) = 20 exp(-31 / 10) = 0.901. It
        bounds the distance of the order-ten sum from the series summed to
        |m| <= 80 at holonomies of that modulus and its inverse. At F = 20
        every term is positive and the part beyond the order is
        sum_{m > 10} t_m (1 + 20^(-2m)), half the sum of the bounds 2 t_m,
        whose terms peak at m = beta log 20 = 15: the bound, 1.36e11, is
        within a factor of five of it."""
        beta, order, modulus = 5.0, ORDER, 20.0
        character = cob.VillainCharacter(beta, order)
        self.assertAlmostEqual(modulus * math.exp(-23.0 / 10.0), 2.005,
                               places=3)
        self.assertGreater(modulus * math.exp(-29.0 / 10.0), 1.0)
        self.assertAlmostEqual(modulus * math.exp(-31.0 / 10.0), 0.901,
                               places=3)

        def term(m):
            return math.exp(-m * m / (2.0 * beta) + m * math.log(modulus))

        expected = sum(2.0 * term(m) for m in range(11, 15)) \
            + 2.0 * term(15) / (1.0 - modulus * math.exp(-31.0 / 10.0))
        self.assertAlmostEqual(expected, 1.36e11, delta=1e9)
        for holonomy in (complex(modulus), modulus * cmath.exp(0.7j),
                         cmath.exp(-2.1j) / modulus):
            series = character.series(holonomy)
            self.assertAlmostEqual(series.value_tail / expected, 1.0,
                                   places=12)
            m = np.arange(-80, 81)
            long_sum = np.sum(np.exp(-m * m / (2.0 * beta)
                                     + m * cmath.log(holonomy)))
            omitted = abs(long_sum - series.value)
            self.assertLessEqual(omitted, series.value_tail)
            if holonomy.imag == 0.0:
                self.assertGreater(omitted, series.value_tail / 5.0)

    def test_the_action_reports_its_order_and_its_tails(self):
        """`holonomy_truncation` reports the declared order and, over the
        faces, the largest tail bound of each sum relative to the sum of the
        moduli of that sum's terms: the largest over the faces of the ratios
        the per-face series report. On the 3-sphere boundary at beta = 1.3
        the links' phases have imaginary parts of at most 0.09 in modulus, so
        r = max(|F|, 1/|F|) is at most exp(0.27) = 1.31 on every face. At
        order ten the first pair beyond the order is at most
        2 exp(-121 / 2.6) 1.31^11 = 2.4e-19, so the reported tail of W is
        below 1e-18, and those of DW and D^2W, which carry 11 and 121 times
        that against sums of at least 1.3, are below 1e-17 and 1e-16. At
        order three the first pair is 2 exp(-16 / 2.6) r^4, between 4.3e-3
        at r = 1 and 1.25e-2 at r = 1.31, against sums of the moduli between
        2.85 and 3.0: the reported tail of W lies between 1.4e-3 and
        6e-3."""
        spacetime = sphere3(phase=_flux)
        faces = np.asarray(cob.JointAction(
            spacetime, _declaration()).face_holonomies())
        self.assertEqual(len(faces), 10)
        moduli = np.abs(faces)
        self.assertGreater(float(np.max(np.abs(moduli - 1.0))), 0.05)
        self.assertLessEqual(float(np.max(np.maximum(moduli, 1.0 / moduli))),
                             math.exp(0.27) * (1.0 + 1e-12))
        for order in (ORDER, 3):
            read = cob.JointAction(spacetime, _declaration(
                villain_order=order)).holonomy_truncation()
            self.assertEqual(read.order, order)
            character = cob.VillainCharacter(1.3, order)
            series = [character.series(complex(face)) for face in faces]
            self.assertEqual(read.relative_value_tail, max(
                s.value_tail / s.magnitude for s in series))
            self.assertEqual(read.relative_first_tail, max(
                s.first_tail / s.first_magnitude for s in series))
            self.assertEqual(read.relative_second_tail, max(
                s.second_tail / s.second_magnitude for s in series))
            if order == ORDER:
                self.assertLess(read.relative_value_tail, 1e-18)
                self.assertLess(read.relative_first_tail, 1e-17)
                self.assertLess(read.relative_second_tail, 1e-16)
            else:
                self.assertGreater(read.relative_value_tail, 1.4e-3)
                self.assertLess(read.relative_value_tail, 6e-3)
        # with the holonomy term off the report carries the order alone
        off = cob.JointAction(
            spacetime, _declaration(beta=0.0)).holonomy_truncation()
        self.assertEqual(off.order, ORDER)
        self.assertEqual(off.relative_value_tail, 0.0)

    def test_the_holonomy_term_is_the_function_the_order_defines(self):
        """The action declared at order M sums -beta_V log W_M over the
        faces with the order-M weight and matched weight: at trivial
        holonomy the value is -beta_V times the face count times log W_M(1)
        at each order, and the values at orders one, three and ten differ
        (W_M(1) = 1 + 2 sum_{m <= M} c_m: 2.361, 2.858 less 4.4e-3, and
        2.858 at beta = 1.3)."""
        spacetime = sphere3()
        values = {}
        for order in (1, 3, ORDER):
            character = cob.VillainCharacter(1.3, order)
            weight = 1.0 + 2.0 * sum(character.coefficients[1:])
            self.assertAlmostEqual(character.series(1.0 + 0j).value.real,
                                   weight, places=14)
            action = cob.JointAction(spacetime,
                                     _declaration(villain_order=order))
            values[order] = action.holonomy_term()
            self.assertAlmostEqual(
                abs(values[order]
                    + character.matched_weight * 10 * math.log(weight)), 0.0,
                places=12)
        self.assertAlmostEqual(
            cob.VillainCharacter(1.3, 1).series(1.0 + 0j).value.real, 2.361,
            places=3)
        self.assertAlmostEqual(
            cob.VillainCharacter(1.3).series(1.0 + 0j).value.real, 2.858,
            places=3)
        self.assertGreater(abs(values[1] - values[ORDER]), 1e-2)
        self.assertGreater(abs(values[3] - values[ORDER]), 1e-4)

    def test_the_declaration_is_validated(self):
        spacetime = sphere3()
        self.assertEqual(cob.JointActionDeclaration().villain_order, 10)
        with self.assertRaises(ValueError):
            cob.JointAction(spacetime, _declaration(beta=-1.0))
        for order in (0, 11, -3):
            with self.assertRaisesRegex(
                    ValueError, "the order of the Villain weight is an "
                                "integer from 1 to 10; got %d" % order):
                cob.JointAction(spacetime, _declaration(villain_order=order))
        with self.assertRaises(ValueError):
            cob.VillainCharacter(0.0)


if __name__ == "__main__":
    unittest.main()
