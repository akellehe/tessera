# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Acceptance tests for the world-tube crossing readouts
(:class:`tessera.observables.CrossingReadouts`), ticket #807 and the
whitepaper section "Mass, charge, and form factor from world-tube
crossings".

The whitepaper is the specification these tests hold the code to:

* ``tau`` is the COMPLEX Lorentzian distance from the incoming boundary M0,
  built from the 1-skeleton and the stored complex edge lengths and never
  from a vertex coordinate; ``Re tau`` carries a temporal-function
  certificate and every failure is NAMED;
* ``pi_perp(c) = sum_e mu_c(e) dtau(e)`` stays COMPLEX, with ``mu_c`` the
  band density from the matched left/right frames (gauge-invariant by
  left/right cancellation) normalized to sum one over the crossing set;
* admissibility = the band's positivity certificate AND ``Re pi_perp``
  nonvanishing with a single sign; then ``sgn pi_perp := sgn Re pi_perp``;
* ``m_x = kappa_m sum |pi_perp|`` (INCOHERENT) and
  ``B = (1/3) sum sgn pi_perp`` over certified quark tubes (COHERENT), both
  as DIFFERENCES against the same sum at M0;
* the crossing sign and the determinant-line winding must agree on every
  certified tube -- disagreement is a DEFECT SIGNAL, never resolved;
* ``S(lambda)`` uses EIGENSPACE PROJECTORS, so it is basis- and
  phase-invariant and handles degeneracies; it is an incoherent structure
  factor and is NEVER the electromagnetic form factor.

Closed-form anchors (verified, not fitted):

* the causal ladder below has ``tau = 0`` on M0 and ``tau = 1`` on the
  upper layer exactly, because a timelike edge of length ``l = i`` has
  ``l^2 = -1`` and proper time ``sqrt(-l^2) = 1``;
* a tube localized on one timelike rung therefore has ``pi_perp = 1 + 0j``
  exactly, and a reversed tube ``-1 + 0j``;
* the degenerate slice-Laplacian eigenspace of a 3-cycle (spec {0, 3, 3})
  gives a closed-form eigenprojector ``I - J/3`` whose power is reproduced
  here from an INDEPENDENTLY ROTATED numpy eigenbasis of the same
  eigenspace -- the basis-invariance the projector formulation buys.
"""
import cmath
import math
import unittest

import numpy as np

import tessera

obs = tessera.observables

MACHINE = 1e-12  # machine-precision claims on exact closed forms


# --------------------------------------------------------------------------- #
# fixture builders
# --------------------------------------------------------------------------- #
def _from_simplices(num_vertices, simplices, ids=None):
    """Explicit-complex idiom shared with the spectral-fiber and Hodge
    Laplacian suites."""
    sig = tessera.Signature(4, tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    st = tessera.Spacetime(metric, tessera.HERMITIAN_WEIGHTED, 1.0, 1.0,
                           tessera.PREFERRED, tessera.Toroid())
    ids = list(range(num_vertices)) if ids is None else ids
    verts = [st.createVertex(i) for i in ids]
    for simplex in simplices:
        st.createSimplex([verts[i] for i in simplex])
    for e in st.getEdgeList().toVector():
        e.setLength(1.0 + 0j)
        e.setPhase(0.0)
    return st


def _edge(st, a, b):
    for e in st.getEdgeList().toVector():
        if {e.getSource().getId(), e.getTarget().getId()} == {a, b}:
            return e
    raise KeyError((a, b))


def _ladder():
    """The causal ladder: a spacelike triangle 0-1-2 (M0), a spacelike
    triangle 3-4-5 above it, and three TIMELIKE rungs 0-3, 1-4, 2-5 with
    l = i, so l^2 = -1 and the proper time of each rung is exactly 1."""
    st = _from_simplices(
        6,
        [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5),
         (0, 3), (1, 4), (2, 5)],
    )
    for a, b in ((0, 3), (1, 4), (2, 5)):
        _edge(st, a, b).setLength(1j)
    return st


def _cone():
    """A cone cobordism: M0 is the single vertex 0, the upper layer is the
    spacelike triangle 1-2-3, and the three TIMELIKE rungs 0-1, 0-2, 0-3 all
    meet at 0.

    Every crossing edge therefore shares the vertex 0, so the slice graph is
    the complete graph K3 whose Laplacian is 3I - J: spectrum {0, 3, 3} with
    a genuinely DEGENERATE lambda = 3 eigenspace and the closed-form
    eigenprojector I - J/3.  The ladder's rungs, by contrast, pairwise share
    no vertex and give a disconnected slice.
    """
    st = _from_simplices(
        4, [(0, 1), (0, 2), (0, 3), (1, 2), (2, 3), (1, 3)])
    for a, b in CONE_RUNGS:
        _edge(st, a, b).setLength(1j)
    return st


M0 = [0, 1, 2]
RUNGS = ((0, 3), (1, 4), (2, 5))
CONE_M0 = [0]
CONE_RUNGS = ((0, 1), (0, 2), (0, 3))


# --------------------------------------------------------------------------- #
# a real, certified, positive band -- reused as the certificate of every tube
# --------------------------------------------------------------------------- #
def _certified_rank_one_record(degree):
    """The record of a REAL rank-one accepted band with positive signature at
    `degree`, produced by the real tracker on the Euclidean 3-cycle (spec
    {0, 3, 3} at both k = 0 and k = 1, so the lambda = 0 band is rank one and
    isolated).

    The tests below re-point this record's CELLS at the ladder's timelike
    rungs through the sanctioned `fromRecord` replay path.  The CERTIFICATE
    is never fabricated: it stays exactly the one the tracker issued, which
    is what makes `accepted` and the positive signature real rather than
    asserted.
    """
    st = _from_simplices(3, [(0, 1), (1, 2), (0, 2)])
    # The subject here is the CROSSING readouts, not localization.  Every
    # band of a vertex-transitive fixture like the 3-cycle has a uniform
    # projector diagonal and therefore localization excess EXACTLY 1, which
    # never certifies under the default acceptance conjunct.  Declare the
    # permissive analysis cap the spectral-fiber suite defines for exactly
    # this situation; the conjunct itself is exercised there.
    cfg = obs.SpectralFiberConfig()
    cfg.maxLocalizationExcess = 1.0
    # A positive band with Krein signature (1, 0) is the diagonal weights'
    # certificate: the default Whitney pencil reads degree >= 1 in the
    # complex-symmetric pencil regime, whose bands carry the bilinear pairing
    # and no inertia, so the diagonal source is named.
    tracker = obs.SpectralFiberTracker(
        st, cfg, metric_source=tessera.cobordism.HodgeMetricSource.DiagonalWeights)
    read = tracker.enumerateBands([0, 1, 2], degree)
    for fiber in read.fibers:
        cert = fiber.certificate()
        if (fiber.rank() == 1 and cert.accepted
                and cert.positiveSignature == 1 and cert.negativeSignature == 0):
            return fiber.toRecord()
    raise AssertionError(
        f"no certified rank-one positive band at degree {degree}")


_BASE_RECORDS = {}


def _band_on(cells, amplitudes, degree=1):
    """A rank-one band supported on `cells` with the given complex frame
    amplitudes, carrying the real certificate of `_certified_rank_one_record`.

    With unit weights the projector is `P = Phi Psi^dagger`, so the density
    on row i is |a_i|^2; `Psi^dagger W Phi = I` requires sum |a_i|^2 = 1.
    """
    if degree not in _BASE_RECORDS:
        _BASE_RECORDS[degree] = _certified_rank_one_record(degree)
    record = {k: (list(v) if isinstance(v, list) else v)
              for k, v in _BASE_RECORDS[degree].items()}
    rows = len(cells)
    record["cells"] = [list(cell) for cell in cells]
    record["rows"] = rows
    record["rank"] = 1
    record["right_frame_re"] = [complex(a).real for a in amplitudes]
    record["right_frame_im"] = [complex(a).imag for a in amplitudes]
    record["left_frame_re"] = [complex(a).real for a in amplitudes]
    record["left_frame_im"] = [complex(a).imag for a in amplitudes]
    record["weights_re"] = [1.0] * rows
    record["weights_im"] = [0.0] * rows
    record["eigenvalues_re"] = [0.0]
    record["eigenvalues_im"] = [0.0]
    return obs.SpectralFiber.fromRecord(record)


def _tube(tube_id, cells, amplitudes, orientation=1, quark=True,
          winding=None, degree=1):
    tube = obs.WorldTubeInput()
    tube.tubeId = tube_id
    tube.band = _band_on(cells, amplitudes, degree)
    tube.orientation = orientation
    tube.certifiedQuarkTube = quark
    if winding is not None:
        tube.determinantWinding = winding
    return tube


def _rung_tube(tube_id, rung, **kwargs):
    return _tube(tube_id, [list(rung)], [1.0 + 0j], **kwargs)


# --------------------------------------------------------------------------- #
# the temporal function and its certificate
# --------------------------------------------------------------------------- #
class TestTemporalFunction(unittest.TestCase):
    def test_ladder_tau_is_exact_and_certified(self):
        """tau = 0 on M0 and exactly 1 on the upper layer: the proper time
        sqrt(-l^2) of a rung with l = i.  The certificate holds."""
        read = obs.CrossingReadouts.temporalFunction(_ladder(), M0)
        self.assertTrue(read.certified, read.failedCertificates)
        self.assertEqual(list(read.failedCertificates), [])
        for vertex in M0:
            self.assertAlmostEqual(abs(read.at(vertex)), 0.0, delta=MACHINE)
        for vertex in (3, 4, 5):
            self.assertAlmostEqual(read.at(vertex).real, 1.0, delta=MACHINE)
            self.assertAlmostEqual(read.at(vertex).imag, 0.0, delta=MACHINE)
        self.assertAlmostEqual(read.minCausalIncrement, 1.0, delta=MACHINE)
        self.assertEqual(read.causalEdgeCount, 3)
        self.assertEqual(read.unreachableCount, 0)

    def test_a_generically_complex_rung_is_mixed_and_carries_no_time(self):
        """Measured after #870, and NOT obviously the behaviour to want.

        This rung has `l^2 = -(1 + i)`, so `arg(l^2) = -3pi/4` -- a generic
        argument, hence MIXED: no definite causal character. A mixed edge
        cannot order anything, so tau does not propagate along it and vertex 3
        is unreachable.

        The consequence is worth stating plainly, because it narrows the
        readout. Causal edges are now timelike (`l^2` real negative) or null,
        and null ones are refused by name; a timelike edge has proper time
        `sqrt(-l^2)`, which is REAL. So on a CERTIFIED configuration tau is
        always real, and the complex-tau path this test used to exercise is
        unreachable -- exactly as the lightlike case was unreachable before.

        That is a direct consequence of classifying on the argument and
        refusing to snap a generic one to the nearest definite type. It may be
        right, but it is a narrowing rather than a fix, and it is recorded here
        as a tripwire so the decision is visible rather than assumed.
        """
        st = _ladder()
        edge = _edge(st, 0, 3)
        edge.setLength(cmath.sqrt(complex(-1.0, -1.0)))
        self.assertTrue(edge.isMixed())
        self.assertAlmostEqual(edge.squaredArgument(), -3.0 * math.pi / 4.0,
                               delta=1e-12)
        read = obs.CrossingReadouts.temporalFunction(st, M0)
        self.assertTrue(math.isnan(read.at(3).real))   # unreachable, not zero

    def test_layers_are_intrinsic_not_coordinates(self):
        """The time orientation is the one M0 induces combinatorially; no
        vertex coordinate is read.  Moving a vertex's stored time cannot
        change the layering."""
        st = _ladder()
        before = obs.CrossingReadouts.temporalFunction(st, M0)
        for vertex in st.getVertexList().toVector():
            if vertex.getId() in (3, 4, 5):
                vertex.setTime(-99.0)
        after = obs.CrossingReadouts.temporalFunction(st, M0)
        self.assertEqual(list(before.layer), list(after.layer))
        self.assertEqual(before.certified, after.certified)

    def test_empty_boundary_refuses_by_name(self):
        read = obs.CrossingReadouts.temporalFunction(_ladder(), [])
        self.assertFalse(read.certified)
        self.assertIn("empty-boundary", list(read.failedCertificates))

    def test_null_causal_edge_refuses_by_name(self):
        """A null edge is refused, never counted at zero.

        The edge here is genuinely LIGHTLIKE -- `Re(l) == Im(l) > 0`, so the
        interval vanishes while the edge keeps nonzero extent. Before #870 that
        case was unreachable and this test stood in a zero-length edge, which is
        DEGENERATE (absent) rather than null. Reading `arg(l^2)` separates the
        two, so the reason is now exercised by the thing it is named for.
        """
        st = _ladder()
        component = math.sqrt(0.5)
        edge = _edge(st, 0, 3)
        edge.setLength(complex(component, component))
        self.assertTrue(edge.isNull())          # a lightlike ray ...
        self.assertFalse(edge.isDegenerate())   # ... not an absent edge
        read = obs.CrossingReadouts.temporalFunction(st, M0)
        self.assertFalse(read.certified)
        self.assertIn("null-causal-edge", list(read.failedCertificates))

    def test_causal_edge_inside_one_layer_refuses_by_name(self):
        """A causal edge joining two vertices of the same layer cannot be
        ordered by the induced time orientation."""
        st = _ladder()
        _edge(st, 0, 1).setLength(1j)   # a timelike edge inside M0
        read = obs.CrossingReadouts.temporalFunction(st, M0)
        self.assertFalse(read.certified)
        self.assertIn("causal-cycle", list(read.failedCertificates))

    def test_uncertified_temporal_function_blocks_every_crossing(self):
        read = obs.CrossingReadouts.temporalFunction(_ladder(), [])
        crossing = obs.CrossingReadouts.crossing(
            _rung_tube("q", RUNGS[0]), read, 0.5)
        self.assertFalse(crossing.admissible)
        self.assertIn("uncertified-temporal-function",
                      list(crossing.failedCertificates))


# --------------------------------------------------------------------------- #
# the band density and its gauge invariance
# --------------------------------------------------------------------------- #
class TestConnectionIndependence(unittest.TestCase):
    """The whitepaper is explicit that `dtau` is built from the squared
    lengths z ALONE and never contains the connection.  Now that the edge
    phase is a live complex C* link field (Re = compact U(1) angle, Im =
    non-compact R+ scale), that claim is testable rather than aspirational:
    an arbitrary complex phase on EVERY edge must leave every crossing
    readout BITWISE unchanged.

    Any movement here would mean a connection-carrying path had been picked
    up, which is a real bug rather than a tolerance question -- so these
    assertions are exact equality, not almost-equal.
    """

    @staticmethod
    def _phased(spacetime):
        """An arbitrary, edge-dependent COMPLEX phase on every edge: a
        nontrivial compact angle and a nontrivial non-compact scale."""
        for index, edge in enumerate(spacetime.getEdgeList().toVector()):
            edge.setPhase(complex(0.37 * (index + 1), -0.21 * (index + 2)))
        return spacetime

    def _tubes(self, orientations=(1, 1, 1), windings=None):
        windings = windings or [None] * len(orientations)
        return [
            _rung_tube(f"t{i}", RUNGS[i], orientation=o, winding=w)
            for i, (o, w) in enumerate(zip(orientations, windings))
        ]

    def test_temporal_function_is_bitwise_phase_independent(self):
        plain = obs.CrossingReadouts.temporalFunction(_ladder(), M0)
        phased = obs.CrossingReadouts.temporalFunction(
            self._phased(_ladder()), M0)
        self.assertEqual(plain.certified, phased.certified)
        self.assertEqual(list(plain.vertices), list(phased.vertices))
        for a, b in zip(plain.tau, phased.tau):
            self.assertEqual(a.real, b.real)
            self.assertEqual(a.imag, b.imag)
        self.assertEqual(plain.minCausalIncrement, phased.minCausalIncrement)

    def test_pi_perp_is_bitwise_phase_independent(self):
        plain_t = obs.CrossingReadouts.temporalFunction(_ladder(), M0)
        phased_t = obs.CrossingReadouts.temporalFunction(
            self._phased(_ladder()), M0)
        tube = _rung_tube("q", RUNGS[0])
        a = obs.CrossingReadouts.crossing(tube, plain_t, 0.5)
        b = obs.CrossingReadouts.crossing(tube, phased_t, 0.5)
        self.assertTrue(a.admissible and b.admissible)
        self.assertEqual(a.perpendicular.real, b.perpendicular.real)
        self.assertEqual(a.perpendicular.imag, b.perpendicular.imag)
        self.assertEqual(a.sign, b.sign)
        self.assertEqual(list(a.density), list(b.density))

    def test_mass_baryon_and_profile_are_bitwise_phase_independent(self):
        plain_t = obs.CrossingReadouts.temporalFunction(_ladder(), M0)
        phased_t = obs.CrossingReadouts.temporalFunction(
            self._phased(_ladder()), M0)
        tubes = self._tubes()
        for temporal_a, temporal_b in ((plain_t, phased_t),):
            mass_a = obs.CrossingReadouts.crossingMass(
                tubes, temporal_a, 0.5, 0.0)
            mass_b = obs.CrossingReadouts.crossingMass(
                tubes, temporal_b, 0.5, 0.0)
            self.assertEqual(mass_a.crossingMass, mass_b.crossingMass)
            self.assertEqual(mass_a.levelSum, mass_b.levelSum)

            baryon_a = obs.CrossingReadouts.baryonNumber(
                tubes, temporal_a, 0.5, 0.0)
            baryon_b = obs.CrossingReadouts.baryonNumber(
                tubes, temporal_b, 0.5, 0.0)
            self.assertEqual(baryon_a.baryonNumber, baryon_b.baryonNumber)

            profile_a = obs.CrossingReadouts.chargePowerProfile(
                tubes, temporal_a, 0.5)
            profile_b = obs.CrossingReadouts.chargePowerProfile(
                tubes, temporal_b, 0.5)
            self.assertEqual(list(profile_a.eigenvalues),
                             list(profile_b.eigenvalues))
            self.assertEqual(list(profile_a.power), list(profile_b.power))
            self.assertEqual(profile_a.monopole, profile_b.monopole)

    def test_the_phase_really_is_live_on_the_fixture(self):
        """Guard against a vacuous invariance test: the phases must actually
        be set to nonzero complex values on the fixture being compared."""
        spacetime = self._phased(_ladder())
        phases = [e.getPhase() for e in spacetime.getEdgeList().toVector()]
        self.assertTrue(all(abs(p) > 0.0 for p in phases))
        self.assertTrue(any(abs(p.imag) > 0.0 for p in phases))
        self.assertTrue(any(abs(p.real) > 0.0 for p in phases))


class TestBandDensity(unittest.TestCase):
    def test_density_matches_the_projector_diagonal(self):
        band = _band_on([[0, 3], [1, 4]],
                        [math.sqrt(0.25), math.sqrt(0.75)])
        density = obs.CrossingReadouts.bandEdgeDensity(band)
        self.assertAlmostEqual(density[(0, 3)], 0.25, delta=MACHINE)
        self.assertAlmostEqual(density[(1, 4)], 0.75, delta=MACHINE)

    def test_density_is_gauge_invariant(self):
        """A local C* gauge factor acts on right frames by g^-1 and on left
        frames by g and CANCELS in the bilinear product, so the density is
        unchanged.  Verified through the whole crossing: pi_perp is
        identical."""
        temporal = obs.CrossingReadouts.temporalFunction(_ladder(), M0)
        plain = _tube("q", [list(RUNGS[0]), list(RUNGS[1])],
                      [math.sqrt(0.25), math.sqrt(0.75)])
        phase = cmath.exp(0.7j)
        gauged = _tube("q", [list(RUNGS[0]), list(RUNGS[1])],
                       [phase * math.sqrt(0.25), phase * math.sqrt(0.75)])
        a = obs.CrossingReadouts.crossing(plain, temporal, 0.5)
        b = obs.CrossingReadouts.crossing(gauged, temporal, 0.5)
        self.assertTrue(a.admissible and b.admissible)
        self.assertAlmostEqual(abs(a.perpendicular - b.perpendicular), 0.0,
                               delta=MACHINE)
        for x, y in zip(a.density, b.density):
            self.assertAlmostEqual(x, y, delta=MACHINE)

    def test_degree_zero_band_has_no_edge_density(self):
        """A degree-zero band lives on vertices and carries no edge density,
        so it has no crossing set at all."""
        band = _band_on([[0]], [1.0 + 0j], degree=0)
        self.assertEqual(band.degree(), 0)
        self.assertEqual(dict(obs.CrossingReadouts.bandEdgeDensity(band)), {})

    def test_degree_zero_band_refuses_by_name(self):
        temporal = obs.CrossingReadouts.temporalFunction(_ladder(), M0)
        tube = _tube("v", [[0]], [1.0 + 0j], degree=0)
        read = obs.CrossingReadouts.crossing(tube, temporal, 0.5)
        self.assertFalse(read.admissible)
        self.assertIn("degree-zero-band", list(read.failedCertificates))


# --------------------------------------------------------------------------- #
# the crossing decomposition
# --------------------------------------------------------------------------- #
class TestCrossing(unittest.TestCase):
    def setUp(self):
        self.temporal = obs.CrossingReadouts.temporalFunction(_ladder(), M0)

    def test_pi_perp_is_exact_on_one_rung(self):
        """A tube localized on one rung crosses with pi_perp = dtau = 1."""
        read = obs.CrossingReadouts.crossing(
            _rung_tube("q", RUNGS[0]), self.temporal, 0.5)
        self.assertTrue(read.admissible, read.failedCertificates)
        self.assertAlmostEqual(read.perpendicular.real, 1.0, delta=MACHINE)
        self.assertAlmostEqual(read.perpendicular.imag, 0.0, delta=MACHINE)
        self.assertEqual(read.sign, 1)
        self.assertEqual([tuple(e) for e in read.crossingEdges], [(0, 3)])
        self.assertAlmostEqual(read.density[0], 1.0, delta=MACHINE)

    def test_density_normalizes_over_the_crossing_set(self):
        """mu is normalized to sum one over the CROSSING SET, so a band with
        support off the surface still yields a normalized crossing."""
        band_cells = [list(RUNGS[0]), [3, 4]]     # one rung + one edge above
        tube = _tube("q", band_cells, [math.sqrt(0.5), math.sqrt(0.5)])
        read = obs.CrossingReadouts.crossing(tube, self.temporal, 0.5)
        self.assertTrue(read.admissible, read.failedCertificates)
        self.assertEqual([tuple(e) for e in read.crossingEdges], [(0, 3)])
        self.assertAlmostEqual(sum(read.density), 1.0, delta=MACHINE)
        self.assertAlmostEqual(read.perpendicular.real, 1.0, delta=MACHINE)

    def test_orientation_reversal_flips_the_sign(self):
        """Reversing the tube sends the crossing past-directed: the sign
        flips while the modulus is untouched."""
        forward = obs.CrossingReadouts.crossing(
            _rung_tube("q", RUNGS[0]), self.temporal, 0.5)
        reversed_ = obs.CrossingReadouts.crossing(
            _rung_tube("qbar", RUNGS[0], orientation=-1), self.temporal, 0.5)
        self.assertEqual(forward.sign, 1)
        self.assertEqual(reversed_.sign, -1)
        self.assertAlmostEqual(abs(forward.perpendicular),
                               abs(reversed_.perpendicular), delta=MACHINE)
        self.assertAlmostEqual(
            forward.perpendicular.real + reversed_.perpendicular.real, 0.0,
            delta=MACHINE)

    def test_level_outside_the_tube_refuses_by_name(self):
        read = obs.CrossingReadouts.crossing(
            _rung_tube("q", RUNGS[0]), self.temporal, 7.5)
        self.assertFalse(read.admissible)
        self.assertIn("empty-crossing", list(read.failedCertificates))
        self.assertEqual(read.sign, 0)

    def test_nonregular_level_refuses_by_name(self):
        """A level passing exactly through a vertex is not transversal."""
        read = obs.CrossingReadouts.crossing(
            _rung_tube("q", RUNGS[0]), self.temporal, 1.0)
        self.assertFalse(read.admissible)
        self.assertIn("nonregular-level", list(read.failedCertificates))

    def test_inadmissible_crossing_has_no_sign(self):
        """An inadmissible crossing reports sign 0 = UNKNOWN, never a silent
        zero that could be summed."""
        read = obs.CrossingReadouts.crossing(
            _rung_tube("q", RUNGS[0]), self.temporal, 7.5)
        self.assertEqual(read.sign, 0)
        self.assertTrue(math.isnan(read.perpendicular.real))


# --------------------------------------------------------------------------- #
# the two sums
# --------------------------------------------------------------------------- #
class TestMassAndBaryon(unittest.TestCase):
    def setUp(self):
        self.temporal = obs.CrossingReadouts.temporalFunction(_ladder(), M0)

    def _tubes(self, orientations, windings=None):
        windings = windings or [None] * len(orientations)
        return [
            _rung_tube(f"t{i}", RUNGS[i], orientation=o, winding=w)
            for i, (o, w) in enumerate(zip(orientations, windings))
        ]

    def test_three_forward_quark_tubes_give_baryon_one(self):
        read = obs.CrossingReadouts.baryonNumber(
            self._tubes([1, 1, 1]), self.temporal, 0.5, 0.0)
        self.assertIsNotNone(read.baryonNumber)
        self.assertAlmostEqual(read.baryonNumber, 1.0, delta=MACHINE)
        self.assertEqual(read.quarkTubes, 3)

    def test_conjugate_pair_cancels_baryon_and_doubles_mass(self):
        """Moduli add while signs cancel: the pair carries twice the crossing
        mass of one constituent and zero net baryon number."""
        single = obs.CrossingReadouts.crossingMass(
            self._tubes([1]), self.temporal, 0.5, 0.0)
        pair_tubes = self._tubes([1, -1])
        pair_mass = obs.CrossingReadouts.crossingMass(
            pair_tubes, self.temporal, 0.5, 0.0)
        pair_baryon = obs.CrossingReadouts.baryonNumber(
            pair_tubes, self.temporal, 0.5, 0.0)
        self.assertAlmostEqual(pair_mass.crossingMass,
                               2.0 * single.crossingMass, delta=MACHINE)
        self.assertIsNotNone(pair_baryon.baryonNumber)
        self.assertAlmostEqual(pair_baryon.baryonNumber, 0.0, delta=MACHINE)

    def test_baryon_thirds_are_the_normalization(self):
        """One certified quark tube carries exactly B = 1/3, consistent with
        the independent determinant-line proposal B = nu/3."""
        read = obs.CrossingReadouts.baryonNumber(
            self._tubes([1]), self.temporal, 0.5, 0.0)
        self.assertAlmostEqual(read.baryonNumber, 1.0 / 3.0, delta=MACHINE)
        reversed_read = obs.CrossingReadouts.baryonNumber(
            self._tubes([-1]), self.temporal, 0.5, 0.0)
        self.assertAlmostEqual(reversed_read.baryonNumber, -1.0 / 3.0,
                               delta=MACHINE)

    def test_non_quark_tubes_carry_mass_but_no_baryon_number(self):
        tubes = [_rung_tube("g", RUNGS[0], quark=False)]
        mass = obs.CrossingReadouts.crossingMass(
            tubes, self.temporal, 0.5, 0.0)
        baryon = obs.CrossingReadouts.baryonNumber(
            tubes, self.temporal, 0.5, 0.0)
        self.assertAlmostEqual(mass.crossingMass, 1.0, delta=MACHINE)
        self.assertIsNone(baryon.baryonNumber)   # unknown, never zero

    def test_mass_ships_uncalibrated(self):
        read = obs.CrossingReadouts.crossingMass(
            self._tubes([1]), self.temporal, 0.5, 0.0)
        self.assertFalse(read.calibrated)
        self.assertEqual(read.units, "uncalibrated")
        self.assertAlmostEqual(read.kappaMass, 1.0, delta=MACHINE)

    def test_kappa_scales_the_mass_only(self):
        cfg = obs.CrossingReadoutsConfig()
        cfg.kappaMass = 3.5
        scaled = obs.CrossingReadouts.crossingMass(
            self._tubes([1]), self.temporal, 0.5, 0.0, cfg)
        plain = obs.CrossingReadouts.crossingMass(
            self._tubes([1]), self.temporal, 0.5, 0.0)
        self.assertAlmostEqual(scaled.crossingMass,
                               3.5 * plain.crossingMass, delta=MACHINE)
        baryon = obs.CrossingReadouts.baryonNumber(
            self._tubes([1]), self.temporal, 0.5, 0.0, cfg)
        self.assertAlmostEqual(baryon.baryonNumber, 1.0 / 3.0, delta=MACHINE)

    def test_readout_at_m0_itself_is_zero(self):
        """Every readout is the difference against the same sum at M0, so
        reading M0 against itself is zero by construction."""
        tubes = self._tubes([1, 1, 1])
        mass = obs.CrossingReadouts.crossingMass(
            tubes, self.temporal, 0.5, 0.5)
        baryon = obs.CrossingReadouts.baryonNumber(
            tubes, self.temporal, 0.5, 0.5)
        self.assertAlmostEqual(mass.crossingMass, 0.0, delta=MACHINE)
        self.assertAlmostEqual(baryon.baryonNumber, 0.0, delta=MACHINE)

    def test_winding_disagreement_is_a_defect_signal(self):
        """A tube whose crossing sign disagrees with its certified winding is
        NAMED, and it is not dropped from the sum."""
        tubes = self._tubes([-1], windings=[+1])
        read = obs.CrossingReadouts.baryonNumber(
            tubes, self.temporal, 0.5, 0.0)
        self.assertEqual(list(read.signDefects), ["t0"])
        self.assertEqual(read.windingAgreements, 0)
        self.assertAlmostEqual(read.baryonNumber, -1.0 / 3.0, delta=MACHINE)

    def test_winding_agreement_is_counted(self):
        read = obs.CrossingReadouts.baryonNumber(
            self._tubes([1, -1], windings=[+1, -1]), self.temporal, 0.5, 0.0)
        self.assertEqual(list(read.signDefects), [])
        self.assertEqual(read.windingAgreements, 2)


# --------------------------------------------------------------------------- #
# the spectral charge-power profile
# --------------------------------------------------------------------------- #
class TestChargePowerProfile(unittest.TestCase):
    def setUp(self):
        self.temporal = obs.CrossingReadouts.temporalFunction(
            _cone(), CONE_M0)

    def _cone_tube(self, tube_id, rung, **kwargs):
        return _tube(tube_id, [list(rung)], [1.0 + 0j], **kwargs)

    def test_profile_matches_an_independent_rotated_eigenbasis(self):
        """S(lambda) is built from EIGENSPACE PROJECTORS, so it must agree
        with <rho, P_lambda rho> computed from a DIFFERENT (randomly rotated)
        orthonormal basis of the same eigenspace -- the basis-invariance the
        projector formulation buys, which a single-eigenvector coefficient
        would not have.

        The cone's three rungs all meet at vertex 0, so the slice graph is
        K3 with Laplacian 3I - J: spectrum {0, 3, 3}, and the lambda = 3
        eigenspace is genuinely two-dimensional.
        """
        tubes = [self._cone_tube(f"t{i}", CONE_RUNGS[i]) for i in range(3)]
        read = obs.CrossingReadouts.chargePowerProfile(
            tubes, self.temporal, 0.5)
        self.assertEqual(read.sliceNodes, 3)
        self.assertEqual(len(read.eigenvalues), 2)
        self.assertAlmostEqual(read.eigenvalues[0], 0.0, delta=1e-9)
        self.assertAlmostEqual(read.eigenvalues[1], 3.0, delta=1e-9)

        rho = np.ones(3)                        # three forward unit crossings
        laplacian = 3.0 * np.eye(3) - np.ones((3, 3))
        values, vectors = np.linalg.eigh(laplacian)

        # An INDEPENDENT orthonormal basis of the degenerate lambda = 3
        # eigenspace: rotate the two eigenvectors by a random 2x2 rotation.
        degenerate = vectors[:, np.abs(values - 3.0) < 1e-9]
        self.assertEqual(degenerate.shape[1], 2)
        theta = 0.837
        rotation = np.array([[math.cos(theta), -math.sin(theta)],
                             [math.sin(theta), math.cos(theta)]])
        rotated = degenerate @ rotation
        projector = rotated @ rotated.T
        # The closed form: P_3 = I - J/3.
        closed_form = np.eye(3) - np.ones((3, 3)) / 3.0
        self.assertLess(np.max(np.abs(projector - closed_form)), 1e-9)

        self.assertAlmostEqual(read.power[1], float(rho @ projector @ rho),
                               delta=1e-9)
        monopole = vectors[:, np.abs(values) < 1e-9]
        self.assertAlmostEqual(
            read.power[0],
            float(rho @ (monopole @ monopole.T) @ rho), delta=1e-9)

    def test_neutral_system_refuses_normalization_and_reports_power(self):
        """The normalizing monopole vanishes for a conjugate pair on one
        CONNECTED slice: the NORMALIZED profile refuses by name while the
        unnormalized power stays reported."""
        tubes = [
            self._cone_tube("q", CONE_RUNGS[0]),
            self._cone_tube("qbar", CONE_RUNGS[1], orientation=-1),
        ]
        read = obs.CrossingReadouts.chargePowerProfile(
            tubes, self.temporal, 0.5)
        self.assertEqual(read.sliceNodes, 2)
        self.assertFalse(read.normalized)
        self.assertIn("neutral-system", list(read.failedCertificates))
        self.assertEqual(list(read.normalizedPower), [])
        self.assertGreater(sum(read.power), 0.0)
        self.assertAlmostEqual(read.monopole, 0.0, delta=1e-9)

    def test_charged_system_normalizes_to_one_at_zero(self):
        tubes = [self._cone_tube(f"t{i}", CONE_RUNGS[i]) for i in range(3)]
        read = obs.CrossingReadouts.chargePowerProfile(
            tubes, self.temporal, 0.5)
        self.assertTrue(read.normalized, list(read.failedCertificates))
        zero_index = min(range(len(read.eigenvalues)),
                         key=lambda i: abs(read.eigenvalues[i]))
        self.assertAlmostEqual(read.normalizedPower[zero_index], 1.0,
                               delta=1e-9)
        # Total charge 3 spread over K3: the monopole is (sum rho)^2 / 3 = 3
        # and the degenerate lambda = 3 power vanishes on the constant.
        self.assertAlmostEqual(read.monopole, 3.0, delta=1e-9)
        self.assertAlmostEqual(read.power[1], 0.0, delta=1e-9)

    def test_empty_slice_refuses_by_name(self):
        read = obs.CrossingReadouts.chargePowerProfile(
            [self._cone_tube("q", CONE_RUNGS[0])], self.temporal, 7.5)
        self.assertFalse(read.normalized)
        self.assertIn("empty-slice", list(read.failedCertificates))
        self.assertEqual(read.sliceNodes, 0)


# --------------------------------------------------------------------------- #
# the conditional form factor
# --------------------------------------------------------------------------- #
class TestFormFactor(unittest.TestCase):
    def test_form_factor_is_a_refusal_scaffold(self):
        """G_E needs a certified conserved current, certified momentum-
        transfer states, and a refinement extrapolation.  None exist here, so
        the radius is UNAVAILABLE with every missing certificate named."""
        temporal = obs.CrossingReadouts.temporalFunction(_cone(), CONE_M0)
        tubes = [_tube(f"t{i}", [list(CONE_RUNGS[i])], [1.0 + 0j])
                 for i in range(3)]
        profile = obs.CrossingReadouts.chargePowerProfile(
            tubes, temporal, 0.5)
        read = obs.CrossingReadouts.formFactor(profile)
        self.assertFalse(read.available)
        self.assertIsNone(read.chargeRadiusSquared)
        self.assertEqual(
            set(read.failedCertificates),
            {"no-certified-conserved-current", "no-certified-momentum-states",
             "no-refinement-extrapolation"})

    def test_spectral_power_is_never_substituted_for_g_e(self):
        """Even on a fully normalized profile the form factor refuses: the
        incoherent structure factor is not the coherent matrix element."""
        temporal = obs.CrossingReadouts.temporalFunction(_cone(), CONE_M0)
        tubes = [_tube(f"t{i}", [list(CONE_RUNGS[i])], [1.0 + 0j])
                 for i in range(3)]
        profile = obs.CrossingReadouts.chargePowerProfile(
            tubes, temporal, 0.5)
        self.assertTrue(profile.normalized)
        read = obs.CrossingReadouts.formFactor(profile)
        self.assertFalse(read.available)
        self.assertIn("never substituted", read.note)


# --------------------------------------------------------------------------- #
# the overlay block
# --------------------------------------------------------------------------- #
class TestOverlayRecord(unittest.TestCase):
    def test_overlay_is_versioned_and_complete(self):
        temporal = obs.CrossingReadouts.temporalFunction(_ladder(), M0)
        tubes = [_rung_tube(f"t{i}", RUNGS[i]) for i in range(3)]
        record = obs.CrossingReadouts.overlayRecord(
            tubes, temporal, 0.5, 0.0)
        self.assertEqual(record["schema_version"],
                         obs.CrossingReadouts.kSchemaVersion)
        for key in ("level", "reference_level", "thresholds",
                    "temporal_function", "crossings", "crossing_mass",
                    "baryon_number", "charge_power_profile", "form_factor"):
            self.assertIn(key, record)
        self.assertEqual(len(record["crossings"]), 3)
        self.assertAlmostEqual(record["baryon_number"]["baryon_number"], 1.0,
                               delta=MACHINE)
        self.assertFalse(record["form_factor"]["available"])

    def test_complex_channels_carry_both_parts(self):
        """pi_perp is complex and both parts always travel: nothing is
        silently .real-ed."""
        temporal = obs.CrossingReadouts.temporalFunction(_ladder(), M0)
        record = obs.CrossingReadouts.overlayRecord(
            [_rung_tube("q", RUNGS[0])], temporal, 0.5, 0.0)
        crossing = record["crossings"][0]
        self.assertIn("perpendicular_re", crossing)
        self.assertIn("perpendicular_im", crossing)
        self.assertIn("tau_re", record["temporal_function"])
        self.assertIn("tau_im", record["temporal_function"])

    def test_thresholds_are_echoed(self):
        cfg = obs.CrossingReadoutsConfig()
        cfg.kappaMass = 2.25
        temporal = obs.CrossingReadouts.temporalFunction(_ladder(), M0)
        record = obs.CrossingReadouts.overlayRecord(
            [_rung_tube("q", RUNGS[0])], temporal, 0.5, 0.0, cfg)
        self.assertAlmostEqual(record["thresholds"]["kappa_mass"], 2.25,
                               delta=MACHINE)
        self.assertFalse(record["thresholds"]["mass_calibrated"])


if __name__ == "__main__":
    unittest.main()
