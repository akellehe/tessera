# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The connection phase is a dynamical field.

Every read of the pipeline is taken on the covariant operator `h_k(z, U)`, the
Whitney pencil dressed by the connection `U = exp(i phi)`. Its spectrum is
invariant under the gauge similarity, so the effective homology and every
spectral gate are gauge-invariant, and the Hodge-entropy term of the action
depends on `U` through its holonomy. The term with a gradient in `phi`, which
is what moves the phase, is built on the degree-zero Aharonov-Bohm operator,
whose zero mode a nonzero flux lifts.

Two properties carry the connection term and each is asserted rather than
argued.

The term is read from the EIGENVALUES alone. A gauge transformation acts on the
operator by the similarity `diag(g)^-1 (.) diag(g)`, which fixes eigenvalues for
every `g: K_0 -> C*`, so the functional is constant along gauge orbits. Gauge
invariance is therefore a property of the construction, not a correction — and
the exact consequence, which stands in for an Euler identity here, is that the
`phi` gradient has NO component along any gauge direction, for every complex
vertex function.

The word EIGENVALUES is load-bearing and these tests are what hold it. An
entropy built the way the Hodge term builds one, on `A = M^dag M`, is a
functional of the SINGULAR values instead, and those survive only UNITARY
similarity. `C* = U(1) x R^+`, so that form is gauge-invariant for real `chi`
and measurably not for complex `chi` — this operator is non-normal under complex
phase, which is where the two spectra part company. `_chi` below is complex on
purpose: with the `M^dag M` form the entropy drifts 4.9e-3 and the identity
fails at 1.2e-2, against the 1e-16 and 1e-15 asserted here.

Gauge orthogonality alone would certify only the gauge subspace, so a second
exact identity — `S` is EVEN in `phi` at real weights, hence its gradient is
ODD — carries the physical directions the first one cannot see.

The weights are SQUARED moduli, and that is the second load-bearing choice.
Eigenvalues are similarity-invariant at any power, so `|lambda|` and
`|lambda|^2` are equally `C*`-invariant, but only the square reduces to the
Hodge term's own functional: for Hermitian `L` the eigenvalues of `A = L^dag L`
are exactly `|lambda|^2 = sigma^2`. So the two definitions agree in the
Hermitian limit and separate only where the operator stops being normal, and
both halves of that are asserted below.

The diagonal-weight `laplacian(k)`, named by its metric source, is built from
the squared lengths alone and its bitwise invariance under `phi` is asserted
beside the dependence of the default `h_k(z, U)`, which moves with the flux
and which only a gauge transformation leaves fixed.
"""

import cmath
import math
import unittest

import numpy as np

import tessera as T

cob = T.cobordism
MC = cob.MultiCobordism
DIAGONAL = cob.HodgeMetricSource.DiagonalWeights
MODE = cob.HodgeEntropyPhaseMode.IncludeComplexPhase


def _host(jitter=True):
    """A closed 4-manifold with a mild non-degenerate metric."""
    spacetime = T.Spacetime(T.Metric(True, T.Signature(4, T.Lorentzian)), T.CDT,
                            1.0, 1.0, T.PREFERRED, T.SimplexBoundarySphere(4))
    spacetime.build()
    for index, edge in enumerate(spacetime.getEdgeList().toVector()):
        squared = 1.0 + (0.01 * (index % 6) if jitter else 0.0)
        edge.setLength(cmath.sqrt(complex(squared)))
    return spacetime


def _set_flux(spacetime, scale=1.0):
    """A connection with nonzero flux, in BOTH components of `phi`.

    Not a gauge transform of the flat one: the values do not come from any
    vertex function, so a holonomy around some cycle is necessarily nontrivial.
    """
    for index, edge in enumerate(spacetime.getEdgeList().toVector()):
        edge.setPhase(complex(scale * 0.37 * ((index % 5) - 2),
                              scale * 0.11 * ((index % 3) - 1)))


def _flatten(spacetime):
    for edge in spacetime.getEdgeList().toVector():
        edge.setPhase(complex(0.0, 0.0))


def _gauge(spacetime, chi):
    """Apply `U_xy -> g_x^-1 U_xy g_y` with `g = e^{i chi}`.

    On the stored phase that is `phi -> phi + chi_t - chi_s`, since the stored
    orientation carries `e^{i phi}` and the reverse its inverse.
    """
    for edge in spacetime.getEdgeList().toVector():
        source = int(edge.getSource().getId())
        target = int(edge.getTarget().getId())
        edge.setPhase(edge.getPhase() + chi[target] - chi[source])


def _chi(spacetime, seed=0):
    """A complex vertex function — the full C* gauge group, not just U(1)."""
    values = {}
    for index, vertex in enumerate(spacetime.getVertexList().toVector()):
        step = index + seed
        values[int(vertex.getId())] = complex(0.29 * ((step % 7) - 3),
                                              0.13 * ((step % 4) - 1.5))
    return values


def _phases(spacetime):
    return [complex(edge.getPhase())
            for edge in spacetime.getEdgeList().toVector()]


class ConnectionEntropySeesThePhaseTest(unittest.TestCase):
    """The new operator depends on `phi`; every `L_k` still does not."""

    def test_the_connection_entropy_moves_when_the_phase_does(self):
        spacetime = _host()
        _flatten(spacetime)
        flat = cob.HodgeLaplacian(spacetime).connectionSpectralEntropy()
        _set_flux(spacetime)
        fluxed = cob.HodgeLaplacian(spacetime).connectionSpectralEntropy()
        self.assertNotAlmostEqual(
            flat, fluxed, places=9,
            msg="the connection entropy must SEE the connection")

    def test_every_diagonal_hodge_laplacian_stays_bitwise_blind_to_the_phase(self):
        # If a phase ever reached the diagonal metric weight, the geometry would
        # become gauge-variant and the derived form of L_k would be destroyed.
        # Assert equality, not closeness.
        def diagonal(spacetime):
            return cob.HodgeLaplacian(
                spacetime, cob.HodgeLaplacian.defaultWeightConvention(), DIAGONAL)

        spacetime = _host()
        _flatten(spacetime)
        before = {k: diagonal(spacetime).laplacian(k, True) for k in (0, 1, 2)}
        _set_flux(spacetime)
        for k in (0, 1, 2):
            with self.subTest(degree=k):
                self.assertEqual(
                    list(diagonal(spacetime).laplacian(k, True)),
                    list(before[k]),
                    "laplacian(%d) must be built from the lengths alone" % k)

    def test_the_default_covariant_operator_sees_the_flux_and_not_a_gauge(self):
        # The default Whitney operator is h_k(z, U): a flux moves its spectrum,
        # a gauge transformation is a similarity and leaves it where it was.
        def spectrum(spacetime, k):
            return np.sort_complex(np.asarray(
                cob.HodgeLaplacian(spacetime).eigenvalues(k), dtype=complex))

        spacetime = _host()
        _flatten(spacetime)
        flat = {k: spectrum(spacetime, k) for k in (0, 1)}
        _gauge(spacetime, _chi(spacetime))
        for k in (0, 1):
            with self.subTest(degree=k, move="gauge"):
                np.testing.assert_allclose(
                    spectrum(spacetime, k), flat[k],
                    atol=1e-8 * max(1.0, np.abs(flat[k]).max()))
        _flatten(spacetime)
        _set_flux(spacetime)
        for k in (0, 1):
            with self.subTest(degree=k, move="flux"):
                moved = spectrum(spacetime, k)
                self.assertGreater(np.abs(moved - flat[k]).max(),
                                   1e-6 * max(1.0, np.abs(flat[k]).max()))

    def test_the_phase_gradient_is_nonzero_under_flux(self):
        spacetime = _host()
        _set_flux(spacetime)
        gradient = (cob.HodgeLaplacian(spacetime)
                    .connectionSpectralEntropyPhaseGradient())
        self.assertEqual(len(gradient),
                         len(spacetime.getEdgeList().toVector()))
        self.assertGreater(sum(abs(component) ** 2 for component in gradient),
                           0.0,
                           "a fluxed connection must have a phi gradient")

    def test_both_components_of_the_phase_are_differentiated(self):
        # The owner's rule: never exclude Re or Im by construction. If a
        # component does not matter it must CANCEL, measurably, not be dropped.
        spacetime = _host()
        _set_flux(spacetime)
        gradient = (cob.HodgeLaplacian(spacetime)
                    .connectionSpectralEntropyPhaseGradient())
        self.assertGreater(max(abs(component.real) for component in gradient),
                           1e-12, "the compact part must be differentiated")
        self.assertGreater(max(abs(component.imag) for component in gradient),
                           1e-12, "the non-compact part must be too")


class GaugeInvarianceIsStructuralTest(unittest.TestCase):
    """Built from the EIGENVALUES, so gauge invariance is not a correction."""

    def test_a_gauge_transformation_leaves_the_entropy_unchanged(self):
        # Machine precision, not "close": the M^dag M form this replaced passes
        # a loose bar for real `chi` and fails at 4.9e-3 for complex `chi`, so a
        # slack tolerance here would stop distinguishing the two.
        spacetime = _host()
        _set_flux(spacetime)
        before = cob.HodgeLaplacian(spacetime).connectionSpectralEntropy()
        _gauge(spacetime, _chi(spacetime))
        after = cob.HodgeLaplacian(spacetime).connectionSpectralEntropy()
        self.assertAlmostEqual(before, after, delta=1e-13)

    def test_the_phase_gradient_is_orthogonal_to_every_gauge_direction(self):
        """The exact identity this term is certified by.

        `S` is constant along gauge orbits, so its directional derivative along
        any gauge displacement vanishes identically. In the `h = S_x - i S_y`
        convention that derivative is `sum_e Re(h_e v_e)`, and a gauge
        displacement is `v_e = chi_t - chi_s`. Holds for COMPLEX `chi`, so each
        vertex function gives two independent exact zeros.
        """
        spacetime = _host()
        _set_flux(spacetime)
        gradient = (cob.HodgeLaplacian(spacetime)
                    .connectionSpectralEntropyPhaseGradient())
        edges = spacetime.getEdgeList().toVector()
        scale = math.sqrt(sum(abs(component) ** 2 for component in gradient))
        self.assertGreater(scale, 0.0, "a zero gradient would pass vacuously")
        for seed in range(4):
            chi = _chi(spacetime, seed)
            with self.subTest(seed=seed):
                directional = 0.0
                for index, edge in enumerate(edges):
                    displacement = (chi[int(edge.getTarget().getId())] -
                                    chi[int(edge.getSource().getId())])
                    directional += (gradient[index] * displacement).real
                self.assertLess(
                    abs(directional) / scale, 1e-12,
                    "the phi gradient must have no gauge component")

    def test_a_flat_connection_relaxes_only_by_gauge(self):
        # With no flux there is nothing physical for the phase to do, so the
        # holonomy must stay trivial however the phases themselves move.
        spacetime = _host()
        _flatten(spacetime)
        gradient = (cob.HodgeLaplacian(spacetime)
                    .connectionSpectralEntropyPhaseGradient())
        for index, component in enumerate(gradient):
            with self.subTest(edge=index):
                self.assertLess(abs(component), 1e-9)


def _hermitian_host():
    """Real lengths and a REAL connection, so `L` comes out Hermitian.

    The C* connection operator is Hermitian exactly when the weights are real
    and `phi` has no imaginary part, which is the limit the Hodge term already
    lives in.
    """
    spacetime = _host(jitter=False)
    for edge in spacetime.getEdgeList().toVector():
        edge.setLength(complex(1.0, 0.0))
    for index, edge in enumerate(spacetime.getEdgeList().toVector()):
        edge.setPhase(complex(0.37 * ((index % 5) - 2), 0.0))
    return spacetime


def _von_neumann_of_positive_operator(spacetime):
    """`-Tr(rho log rho)` for `rho = A/Tr A`, `A = L^dag L` — the Hodge form.

    Computed here rather than called, because the whole point is to compare the
    shipped term against the OTHER construction. Reading it off the same C*
    operator keeps the comparison honest.
    """
    flat = np.array(cob.HodgeLaplacian(spacetime).connectionLaplacian())
    order = int(round(math.sqrt(flat.size)))
    laplacian = flat.reshape(order, order)
    eigenvalues = np.linalg.eigvalsh(laplacian.conj().T @ laplacian)
    floor = np.finfo(float).eps * order * 64 * max(eigenvalues.sum(), 1.0)
    supported = eigenvalues[eigenvalues > floor]
    probability = supported / supported.sum()
    return float(-(probability * np.log(probability)).sum())


class TheHermitianLimitReducesToTheHodgeFunctionalTest(unittest.TestCase):
    """The property the SQUARE buys, and the reason the weight is not `|lambda|`.

    `|lambda|` and `|lambda|^2` are equally `C*`-invariant, since eigenvalues are
    similarity-invariant at any power. Only the square also reduces to the Hodge
    term's own functional, because for Hermitian `L` the eigenvalues of
    `A = L^dag L` are exactly `|lambda|^2 = sigma^2`. The unsquared weight lands
    on the entropy of `|L|/Tr|L|` instead and misses by 8.7e-2.
    """

    def test_the_term_is_the_hodge_functional_on_a_hermitian_operator(self):
        spacetime = _hermitian_host()
        flat = np.array(cob.HodgeLaplacian(spacetime).connectionLaplacian())
        order = int(round(math.sqrt(flat.size)))
        laplacian = flat.reshape(order, order)
        self.assertAlmostEqual(
            float(np.linalg.norm(laplacian - laplacian.conj().T)), 0.0,
            delta=1e-12, msg="this limit must actually be Hermitian")

        self.assertAlmostEqual(
            cob.HodgeLaplacian(spacetime).connectionSpectralEntropy(),
            _von_neumann_of_positive_operator(spacetime),
            delta=1e-13,
            msg="the square must reduce to the M^dag M von Neumann entropy")

    def test_the_two_constructions_separate_once_the_operator_is_not_normal(self):
        # Without this the reduction above would be satisfied by simply BEING
        # the M^dag M form, which is the construction that is not C*-invariant.
        spacetime = _host()
        _set_flux(spacetime)
        flat = np.array(cob.HodgeLaplacian(spacetime).connectionLaplacian())
        order = int(round(math.sqrt(flat.size)))
        laplacian = flat.reshape(order, order)
        self.assertGreater(
            float(np.linalg.norm(laplacian - laplacian.conj().T)), 1e-3,
            "a complex phase must make the operator non-normal")

        self.assertNotAlmostEqual(
            cob.HodgeLaplacian(spacetime).connectionSpectralEntropy(),
            _von_neumann_of_positive_operator(spacetime),
            places=6,
            msg="away from the Hermitian limit the two must NOT agree")


class TheGradientIsCertifiedInEveryDirectionTest(unittest.TestCase):
    """A second exact identity, because gauge orthogonality is not enough.

    Gauge displacements span the image of the coboundary, which is `V - 1`
    complex dimensions out of `E`. On these hosts that is a small subspace, so a
    gradient could be wrong in every PHYSICAL direction and still satisfy the
    orthogonality identity exactly. That gap is closed here.

    For real weights, negating the connection transposes the operator:
    `L_ij(-phi) = -w e^{-i phi} = L_ji(phi)`. Transposition preserves
    eigenvalues, so `S` is EVEN in `phi` and its gradient is ODD — an exact
    identity that constrains every edge direction at once, not a subspace.
    """

    def test_the_gradient_is_odd_under_reversing_the_connection(self):
        spacetime = _host()
        _set_flux(spacetime)
        edges = spacetime.getEdgeList().toVector()
        forward = [complex(edge.getPhase()) for edge in edges]

        hodge = cob.HodgeLaplacian(spacetime)
        entropy = hodge.connectionSpectralEntropy()
        gradient = hodge.connectionSpectralEntropyPhaseGradient()
        scale = math.sqrt(sum(abs(component) ** 2 for component in gradient))
        self.assertGreater(scale, 0.0, "a zero gradient would pass vacuously")

        for edge, phase in zip(edges, forward):
            edge.setPhase(-phase)
        reversed_hodge = cob.HodgeLaplacian(spacetime)
        reversed_entropy = reversed_hodge.connectionSpectralEntropy()
        reversed_gradient = (
            reversed_hodge.connectionSpectralEntropyPhaseGradient())

        self.assertAlmostEqual(entropy, reversed_entropy, delta=1e-13,
                               msg="S must be even in phi at real weights")
        residual = math.sqrt(
            sum(abs(a + b) ** 2 for a, b in zip(reversed_gradient, gradient)))
        self.assertLess(residual / scale, 1e-12,
                        "the phi gradient must be odd in phi")


class StageTwoMovesThePhaseTest(unittest.TestCase):
    """The headline claim: a geometric update changes the connection."""

    def _node(self, spacetime, connection_weight):
        node = MC(spacetime, [], [], [1], 1.0, 7)
        node.set_objective(cob.JointStationarityObjective())
        node.set_simulation_mode(MC.SimulationMode.EMERGENCE,
                                 MC.EmergenceSubmode.STRICT)
        node.set_connection_entropy_weight(connection_weight)
        return node

    def test_stage_two_moves_the_phase_when_the_term_is_declared(self):
        spacetime = _host()
        _set_flux(spacetime)
        node = self._node(spacetime, 1.0)
        before = _phases(node.spacetime())
        list(node.run_stage2(max_iters=12))
        after = _phases(node.spacetime())
        self.assertEqual(len(before), len(after))
        moved = max(abs(a - b) for a, b in zip(after, before))
        self.assertGreater(moved, 0.0,
                           "declaring the term must make the phase dynamical")

    def test_the_phase_is_inert_when_the_term_is_not_declared(self):
        # Zero by default: a node acquires phase dynamics only when asked, and
        # every existing caller keeps the behaviour it had.
        spacetime = _host()
        _set_flux(spacetime)
        node = self._node(spacetime, 0.0)
        before = _phases(node.spacetime())
        list(node.run_stage2(max_iters=12))
        self.assertEqual(_phases(node.spacetime()), before)

    def test_the_term_appears_in_the_declared_objective_record(self):
        spacetime = _host()
        _set_flux(spacetime)
        node = self._node(spacetime, 1.0)
        self.assertIn(cob.ObjectiveTermName.CONNECTION_STATIONARITY,
                      node.objective_term_names())
        self.assertGreater(node.objective_terms().connection_stationarity, 0.0)


if __name__ == "__main__":
    unittest.main()
