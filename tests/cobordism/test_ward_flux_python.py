# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""#1200 — the Ward current's flux through a cooriented cut, and the intrinsic
spectral response that cut carries.

Section 13.4 of the whitepaper defines the complex Ward current of the
gauge-invariant joint action, ``j_xy = U_xy dS/dU_xy``, and its flux
``phi_j(Sigma) = <j, Sigma>`` through a cooriented separating cut. Section 13.5
defines the intrinsic spectral response ``Upsilon_Q(lambda)``, which is what a
cut carries when stable translations have not emerged and there is therefore no
momentum transfer and no form factor to read. Four things are asserted here
rather than argued.

THE WARD IDENTITY IS MEASURED, NOT ASSERTED. ``d j = 0`` is a consequence of the
gauge invariance of the action, so every term that is gauge invariant
contributes nothing to the divergence. The Regge term does not contain the
connection at all; the face-holonomy term's current is a boundary map applied to
the face holonomies, and ``d_1 d_2 = 0`` annihilates it; the matter term
``tr(Gamma h)`` is invariant exactly when the declared covariance commutes with
the carrier operator, which is what a spectral projector of that operator does.
Each of those statements is checked as a number, on a complex whose connection
is not a pure gauge.

THE FLUX COUNTS WHAT THE CUT ENCLOSES. With the boundary convention
``d[x<y] = [y] - [x]``, the cooriented sum of the current over a cut is minus
the divergence summed over the cut's incoming side. That is the discrete
divergence theorem and it is exact. A covariance carrying three unit sources of
the current therefore gives a flux of exactly three through the cut that
separates those three from their common sink — the integer quark number
``N_q``, and ``B = N_q / 3`` — and exactly zero through a cut that encloses
none of them.

HOMOLOGOUS CUTS AGREE WHEN NO SOURCE LIES BETWEEN THEM. Two cuts whose incoming
sides differ only by source-free vertices read the same complex flux to
rounding. A cut moved across a source changes the flux by exactly that source,
and the read names the divergence the slab between the cuts carries, so a
difference is attributed rather than absorbed.

THE INTRINSIC RESPONSE IS A RESOLVENT FORM AND NOTHING ELSE. ``Upsilon_Q`` is
the bilinear resolvent of the slice operator between the two restrictions of the
current, its poles are that operator's eigenvalues, its residues come from Riesz
contours that enclose a whole degenerate band, and its slope in ``lambda`` is
the square of the resolvent taken exactly. No eigenvalue is relabelled as a
momentum transfer anywhere in the read.

A NOTE ON THE COVARIANCE THE COUNTING FIXTURE USES. A spectral projector of the
carrier operator commutes with it and therefore supplies no source at all: its
flux through every cut of a closed complex is exactly zero, which the first
suite below measures. The counting fixture is therefore built from a covariance
that does not commute with the carrier, so that the current has sources to
count, and each source is normalized to one unit. Both numbers — the flux, and
the fermion number the covariance's diagonal places inside the cut — are read
and reported, and they are not assumed equal.
"""

import os
import sys
import unittest

import numpy as np

import tessera as T

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _joint_action_hosts import sphere3  # noqa: E402

cob = T.cobordism

SUPPORT_FLOOR = 1e-9


def _metric(index):
    """A mild deterministic non-uniform complex metric."""
    return complex(1.0 + 0.037 * (index % 5), 0.021 * (1 + index % 4))


def _flux_phase(index):
    """A connection carrying flux in both components of the phase.

    The values come from no vertex function, so some closed walk carries a
    nontrivial holonomy and the connection is not a pure gauge. A pure gauge
    would make several of the statements below hold for the wrong reason.
    """
    return complex(0.23 * ((index % 5) - 2), 0.09 * ((index % 3) - 1))


def _declaration(**overrides):
    """A joint-action declaration with the named fields overridden."""
    declaration = cob.JointActionDeclaration()
    declaration.carrier_degree = 1
    declaration.gravitational_weight = 0.0
    declaration.holonomy_weight = 0.0
    declaration.matter_weight = 0.0
    declaration.metric_source = cob.HodgeMetricSource.WhitneyPencil
    for name, value in overrides.items():
        setattr(declaration, name, value)
    return declaration


def _canonical_edges(spacetime):
    """The canonical degree-one cells of the complex, as vertex pairs."""
    complex_ = cob.ChainComplex.fromSpacetime(spacetime)
    return [tuple(int(vertex) for vertex in cell)
            for cell in complex_.kSimplexVertices(1)]


def _vertices(spacetime):
    """The canonical degree-zero cell order, as vertex identifiers."""
    complex_ = cob.ChainComplex.fromSpacetime(spacetime)
    return [int(cell[0]) for cell in complex_.kSimplexVertices(0)]


def _dyad(order, row, column, weight=1.0):
    """The flat row-major matrix ``weight * e_row e_column^T``.

    This is the smallest covariance that fails to commute with the carrier
    operator. Its commutator with an operator ``h`` has the single diagonal
    entry ``h[column, row]`` at ``column`` and its negative at ``row`` and is
    zero on every other cell, so it places one source and one sink of the Ward
    current and places them nowhere else.
    """
    matrix = np.zeros((order, order), dtype=complex)
    matrix[row, column] = weight
    return [complex(value) for value in matrix.reshape(-1)]


def _sum_covariances(covariances):
    """The entrywise sum of several flat covariances.

    The commutator is linear in the covariance, so the sources of the sum are
    the sources of the terms taken together.
    """
    total = np.zeros(len(covariances[0]), dtype=complex)
    for covariance in covariances:
        total = total + np.array(covariance, dtype=complex)
    return [complex(value) for value in total]


def _support(divergence, vertices):
    """The vertices at which a divergence is measurably nonzero."""
    return {vertices[index] for index in range(len(vertices))
            if abs(divergence[index]) > SUPPORT_FLOOR}


class UnitLineages:
    """A covariance whose Ward current carries one unit on each of several
    lineages, all draining into one shared vertex.

    Each lineage is a dyad of two canonical degree-one cells, which is the
    smallest covariance that does not commute with the carrier operator. Where
    its divergence lands is not predicted here but measured: the unscaled dyad
    is built, its divergence is read from the action, and the lineages are
    chosen so that every one of them touches one shared vertex and no two of
    them touch any other vertex in common. Each dyad is then scaled so that its
    divergence at the shared vertex is exactly minus one. Because a divergence
    sums to zero over the whole complex, every lineage then carries exactly one
    unit away from the shared vertex and nothing anywhere else, which is the
    discrete picture of several quark lineages crossing one separating cut in
    the same direction.
    """

    HOLONOMY_WEIGHT = 0.8
    MATTER_WEIGHT = 1.0

    def __init__(self, wanted):
        self.spacetime = sphere3(squared=_metric, phase=_flux_phase)
        self.vertices = _vertices(self.spacetime)
        self.edges = _canonical_edges(self.spacetime)
        order = len(self.edges)

        candidates = []
        for row in range(order):
            for column in range(order):
                if row == column:
                    continue
                probe = cob.JointAction(
                    self.spacetime,
                    self._declaration(_dyad(order, row, column)))
                divergence = np.array(probe.ward_current_divergence())
                support = _support(divergence, self.vertices)
                if len(support) < 2:
                    continue
                candidates.append((row, column, divergence, support))
        # The most localized lineages first, so the fixture leaves as much of
        # the complex source-free as the complex allows. The sort is stable, so
        # the family a given complex produces is fixed.
        candidates.sort(key=lambda entry: len(entry[3]))

        chosen = None
        for shared in self.vertices:
            picked, taken = [], set()
            for row, column, divergence, support in candidates:
                if shared not in support:
                    continue
                rest = support - {shared}
                if rest & taken:
                    continue
                taken |= rest
                picked.append((row, column, divergence, rest))
                if len(picked) == wanted:
                    break
            if len(picked) == wanted:
                chosen = (shared, picked)
                break
        assert chosen is not None, (
            "no family of %d lineages shares one vertex and no other" % wanted)

        self.shared, picked = chosen
        index = self.vertices.index(self.shared)
        scaled = []
        self.divergences = []
        self.rests = []
        for row, column, divergence, rest in picked:
            weight = -1.0 / divergence[index]
            scaled.append(_dyad(order, row, column, weight))
            self.divergences.append(divergence * weight)
            self.rests.append(sorted(rest))
        self.covariance = _sum_covariances(scaled)
        self.action = cob.JointAction(self.spacetime,
                                      self._declaration(self.covariance))
        self.total = sum(self.divergences)

    def _declaration(self, covariance):
        return _declaration(holonomy_weight=self.HOLONOMY_WEIGHT,
                            matter_weight=self.MATTER_WEIGHT,
                            covariance=covariance)

    def quiet_vertices(self):
        """The vertices no lineage touches."""
        touched = {self.shared}
        for rest in self.rests:
            touched |= set(rest)
        return [vertex for vertex in self.vertices if vertex not in touched]


class TheWardIdentityHoldsForEveryGaugeInvariantTermTest(unittest.TestCase):
    """``d j = 0`` term by term, as a measured number."""

    def test_the_holonomy_current_is_conserved_exactly(self):
        """The face-holonomy current is a boundary map applied to the faces.

        Its divergence is ``d_1 d_2`` applied to the holonomies, which vanishes
        identically, so the conservation is exact rather than approximate.
        """
        spacetime = sphere3(squared=_metric, phase=_flux_phase)
        action = cob.JointAction(spacetime,
                                 _declaration(holonomy_weight=1.3,
                                              gravitational_weight=0.7))
        current = np.array(action.canonical_ward_current())
        self.assertGreater(np.max(np.abs(current)), 1e-3)
        divergence = np.array(action.ward_current_divergence())
        self.assertLess(np.max(np.abs(divergence)),
                        1e-11 * (1.0 + np.max(np.abs(current))))

    def test_a_spectral_projector_covariance_is_gauge_invariant(self):
        """``tr(Gamma h)`` is invariant when ``Gamma`` commutes with ``h``.

        A spectral projector of the carrier operator is exactly such a
        covariance, so a matter term built on one contributes a large current
        and no divergence at all.
        """
        spacetime = sphere3(squared=_metric, phase=_flux_phase)
        bare = cob.JointAction(spacetime, _declaration(holonomy_weight=0.9))
        projector = bare.occupation_projector(3)
        matter_only = cob.JointAction(
            spacetime, _declaration(matter_weight=1.7, covariance=projector))
        matter_current = np.array(matter_only.canonical_ward_current())
        self.assertGreater(np.max(np.abs(matter_current)), 1e-6)
        action = cob.JointAction(
            spacetime,
            _declaration(holonomy_weight=0.9, matter_weight=1.7,
                         covariance=projector))
        current = np.array(action.canonical_ward_current())
        divergence = np.array(action.ward_current_divergence())
        self.assertLess(np.max(np.abs(divergence)),
                        1e-8 * (1.0 + np.max(np.abs(current))))

    def test_a_spectral_projector_covariance_carries_no_flux(self):
        """A conserved current has no flux through any cut of a closed complex.

        This is the whole content of the measurement: the observable exists and
        reads zero, rather than the observable being absent.
        """
        spacetime = sphere3(squared=_metric, phase=_flux_phase)
        bare = cob.JointAction(spacetime, _declaration(holonomy_weight=0.9))
        action = cob.JointAction(
            spacetime,
            _declaration(holonomy_weight=0.9, matter_weight=1.7,
                         covariance=bare.occupation_projector(3)))
        scale = max(abs(value) for value in action.canonical_ward_current())
        for inside in ([0], [0, 1], [2, 3], [1, 3, 4]):
            read = cob.WardFlux.flux(action, cob.CooorientedCut(inside))
            self.assertLess(abs(read.flux), 1e-7 * (1.0 + scale))

    def test_the_canonical_current_is_the_stored_current_reordered(self):
        """The two indexings of one current agree edge by edge.

        The stored order is the mesh's edge list on each edge's stored
        orientation; the canonical order is the chain complex's degree-one cells
        on their ascending-vertex orientation. The current is odd under
        reversing an edge, so the two differ by a permutation and a sign and by
        nothing else.
        """
        spacetime = sphere3(squared=_metric, phase=_flux_phase)
        action = cob.JointAction(spacetime, _declaration(holonomy_weight=1.1))
        canonical = dict(zip(_canonical_edges(spacetime),
                             action.canonical_ward_current()))
        stored = action.ward_current()
        for index, edge in enumerate(spacetime.getEdgeList().toVector()):
            source = int(edge.getSource().getId())
            target = int(edge.getTarget().getId())
            sign = 1.0 if source < target else -1.0
            key = (min(source, target), max(source, target))
            self.assertAlmostEqual(
                abs(canonical[key] - sign * stored[index]), 0.0, places=12)


class TheDivergenceTheoremIsExactTest(unittest.TestCase):
    """The cooriented sum over a cut is minus the divergence it encloses."""

    @classmethod
    def setUpClass(cls):
        cls.lineages = UnitLineages(1)

    def test_the_flux_is_minus_the_enclosed_divergence(self):
        divergence = np.array(
            self.lineages.action.ward_current_divergence())
        vertices = self.lineages.vertices
        for inside in ([vertices[0]], vertices[:2], vertices[2:],
                       [vertices[1]]):
            read = cob.WardFlux.flux(self.lineages.action,
                                     cob.CooorientedCut(inside))
            expected = -sum(divergence[vertices.index(vertex)]
                            for vertex in inside)
            self.assertAlmostEqual(abs(read.flux - expected), 0.0, places=11)
            self.assertLess(read.divergence_theorem_residual, 1e-11)

    def test_reversing_the_cut_reverses_the_flux(self):
        """Declaring the complementary side reverses the coorientation.

        The whitepaper's rule is that reversing the global cobordism
        orientation reverses every oriented charge together, and on a closed
        complex the two complementary cuts carry opposite fluxes exactly.
        """
        inside = [self.lineages.shared]
        outside = [vertex for vertex in self.lineages.vertices
                   if vertex != self.lineages.shared]
        forward = cob.WardFlux.flux(self.lineages.action,
                                    cob.CooorientedCut(inside))
        reverse = cob.WardFlux.flux(self.lineages.action,
                                    cob.CooorientedCut(outside))
        self.assertGreater(abs(forward.flux), 0.5)
        self.assertAlmostEqual(abs(forward.flux + reverse.flux), 0.0,
                               places=10)


class TheFluxCountsTheQuarksOfALineageTest(unittest.TestCase):
    """#1200's acceptance: the Ward flux counts the quarks a cut encloses."""

    @classmethod
    def setUpClass(cls):
        cls.lineages = UnitLineages(3)

    def test_each_lineage_carries_exactly_one_unit(self):
        """The fixture's own certificate: one unit per lineage and no more."""
        index = self.lineages.vertices.index(self.lineages.shared)
        for divergence in self.lineages.divergences:
            self.assertAlmostEqual(abs(divergence[index] + 1.0), 0.0, places=9)
            self.assertAlmostEqual(abs(np.sum(divergence)), 0.0, places=9)
        seen = set()
        for rest in self.lineages.rests:
            self.assertTrue(rest)
            self.assertFalse(seen & set(rest))
            seen |= set(rest)
        self.assertEqual(len(self.lineages.rests), 3)

    def test_a_cut_across_the_three_lineages_reads_three(self):
        read = cob.WardFlux.flux(
            self.lineages.action,
            cob.CooorientedCut([self.lineages.shared], "sigma"))
        self.assertTrue(read.separating)
        self.assertEqual(read.failed_certificates, [])
        self.assertAlmostEqual(read.flux.real, 3.0, places=8)
        self.assertAlmostEqual(read.flux.imag, 0.0, places=8)
        self.assertEqual(read.quark_number, 3)
        self.assertAlmostEqual(read.baryon_number, 1.0, places=12)
        self.assertEqual(read.label, "sigma")
        self.assertEqual(len(read.cut_cells), len(read.coorientation))
        self.assertEqual(len(read.cut_cells), len(read.cut_current))

    def test_a_cut_enclosing_one_lineage_reads_one(self):
        """The far end of one lineage carries that lineage's unit and no other.

        The lineages share only the sink, so a cut around one lineage's far end
        encloses exactly its own unit, with the opposite coorientation to the
        cut around the shared sink.
        """
        read = cob.WardFlux.flux(
            self.lineages.action,
            cob.CooorientedCut(self.lineages.rests[0]))
        self.assertEqual(read.quark_number, -1)
        self.assertAlmostEqual(read.baryon_number, -1.0 / 3.0, places=12)

    def test_a_cut_enclosing_no_lineage_reads_nothing(self):
        quiet = self.lineages.quiet_vertices()
        self.assertTrue(quiet)
        read = cob.WardFlux.flux(self.lineages.action,
                                 cob.CooorientedCut([quiet[0]]))
        self.assertAlmostEqual(abs(read.flux), 0.0, places=8)
        self.assertEqual(read.quark_number, 0)
        self.assertAlmostEqual(read.baryon_number, 0.0, places=12)

    def test_the_whole_complex_carries_no_net_flux(self):
        """Every source has its sink, so a cut with nothing outside is empty."""
        read = cob.WardFlux.flux(
            self.lineages.action,
            cob.CooorientedCut(self.lineages.vertices))
        self.assertAlmostEqual(abs(read.flux), 0.0, places=8)
        self.assertIn("empty-cut", read.failed_certificates)
        self.assertIn("empty-outgoing-side", read.failed_certificates)
        self.assertFalse(read.separating)

    def test_the_enclosed_fermion_number_is_reported_beside_the_flux(self):
        """Both numbers are read, and their difference is named.

        Section 13.4 states that the flux equals the fermion number the cut
        encloses. The read computes that number independently, from the
        diagonal of the declared covariance, so the statement is a measurement
        of a given covariance rather than a definition inside the read. This
        fixture's dyad covariance has a vanishing diagonal, so the two numbers
        differ here and the read says by how much instead of reconciling them.
        """
        read = cob.WardFlux.flux(
            self.lineages.action,
            cob.CooorientedCut([self.lineages.shared]))
        self.assertIsNotNone(read.enclosed_fermion_number)
        self.assertAlmostEqual(abs(read.enclosed_fermion_number), 0.0,
                               places=12)
        self.assertAlmostEqual(read.fermion_number_residual, 3.0, places=8)


class HomologousCutsAgreeWithoutASourceTest(unittest.TestCase):
    """The flux is a homology invariant of the cut away from sources."""

    @classmethod
    def setUpClass(cls):
        cls.lineages = UnitLineages(1)

    def test_two_cuts_differing_by_a_source_free_vertex_agree(self):
        quiet = self.lineages.quiet_vertices()
        self.assertTrue(quiet)
        cuts = [cob.CooorientedCut([self.lineages.shared], "near"),
                cob.CooorientedCut([self.lineages.shared, quiet[0]], "far")]
        read = cob.WardFlux.homologous_fluxes(self.lineages.action, cuts)
        self.assertGreater(abs(read.cuts[0].flux), 0.5)
        self.assertLess(read.max_flux_deviation, 1e-9)
        self.assertLess(read.max_slab_divergence, 1e-9)
        self.assertTrue(read.invariant)
        self.assertEqual([cut.label for cut in read.cuts], ["near", "far"])

    def test_a_cut_moved_across_a_source_changes_by_that_source(self):
        """A difference is attributed to the slab, never absorbed."""
        moved_vertices = self.lineages.rests[0]
        cuts = [cob.CooorientedCut([self.lineages.shared], "before"),
                cob.CooorientedCut([self.lineages.shared] + moved_vertices,
                                   "after")]
        read = cob.WardFlux.homologous_fluxes(self.lineages.action, cuts)
        moved = sum(self.lineages.total[self.lineages.vertices.index(vertex)]
                    for vertex in moved_vertices)
        self.assertAlmostEqual(
            abs((read.cuts[1].flux - read.cuts[0].flux) + moved), 0.0,
            places=9)
        self.assertGreater(read.max_slab_divergence, 0.5)
        self.assertFalse(read.invariant)


class TheCutCertificatesAreNamedTest(unittest.TestCase):
    """A cut that separates nothing refuses and says why."""

    def _closed_action(self):
        spacetime = sphere3(squared=_metric, phase=_flux_phase)
        return cob.JointAction(spacetime, _declaration(holonomy_weight=0.8))

    def _pinched_action(self):
        """Two tetrahedra joined at one vertex.

        Its 1-skeleton is not complete, so a two-vertex side can be
        disconnected, which the boundary of the 4-simplex never allows.
        """
        spacetime = T.Spacetime.fromVertexTuples(
            3, [[0, 1, 2, 3], [3, 4, 5, 6]], 1.0, 0.0)
        for edge in spacetime.getEdgeList().toVector():
            edge.setLength(1.0)
            edge.setPhase(0.0)
        return cob.JointAction(spacetime, _declaration(holonomy_weight=0.8))

    def test_an_empty_incoming_side_is_named(self):
        read = cob.WardFlux.flux(self._closed_action(), cob.CooorientedCut([]))
        self.assertIn("empty-incoming-side", read.failed_certificates)
        self.assertIn("empty-cut", read.failed_certificates)
        self.assertFalse(read.separating)

    def test_a_disconnected_incoming_side_is_named(self):
        read = cob.WardFlux.flux(self._pinched_action(),
                                 cob.CooorientedCut([0, 5]))
        self.assertIn("disconnected-incoming-side", read.failed_certificates)
        self.assertFalse(read.separating)

    def test_a_connected_side_of_the_same_complex_is_accepted(self):
        read = cob.WardFlux.flux(self._pinched_action(),
                                 cob.CooorientedCut([0, 1, 2]))
        self.assertNotIn("disconnected-incoming-side", read.failed_certificates)
        self.assertNotIn("disconnected-outgoing-side", read.failed_certificates)
        self.assertTrue(read.separating)


class TheBackgroundRemovalIsCoherentTest(unittest.TestCase):
    """``Delta O = O_state - O_matched`` is formed before anything else."""

    def test_the_difference_subtracts_the_complex_numbers(self):
        lineages = UnitLineages(1)
        cut = cob.CooorientedCut([lineages.shared], "sigma")
        state = cob.WardFlux.flux(lineages.action, cut)
        matched = cob.WardFlux.flux(
            cob.JointAction(lineages.spacetime,
                            _declaration(holonomy_weight=0.8)), cut)
        removed = cob.WardFlux.difference(state, matched)
        self.assertAlmostEqual(
            abs(removed.flux - (state.flux - matched.flux)), 0.0, places=12)
        self.assertLess(removed.divergence_theorem_residual, 1e-10)
        self.assertEqual(removed.label, "sigma")
        self.assertEqual(removed.quark_number, 3)

    def test_a_difference_between_different_cuts_is_refused(self):
        spacetime = sphere3(squared=_metric, phase=_flux_phase)
        action = cob.JointAction(spacetime, _declaration(holonomy_weight=0.8))
        state = cob.WardFlux.flux(action, cob.CooorientedCut([0]))
        matched = cob.WardFlux.flux(action, cob.CooorientedCut([1]))
        with self.assertRaises(ValueError):
            cob.WardFlux.difference(state, matched)


class TheIntrinsicSpectralResponseTest(unittest.TestCase):
    """``Upsilon_Q`` is the bilinear resolvent of the slice operator."""

    @classmethod
    def setUpClass(cls):
        cls.spacetime = sphere3(squared=_metric, phase=_flux_phase)
        order = len(_canonical_edges(cls.spacetime))
        cls.action = cob.JointAction(
            cls.spacetime,
            _declaration(holonomy_weight=0.8, matter_weight=1.4,
                         covariance=_dyad(order, 0, 7)))
        cls.cut = cob.CooorientedCut([0, 1], "sigma")

    def _read(self, samples):
        return cob.WardFlux.intrinsic_response(self.action, self.cut, samples)

    def _slice(self, read):
        size = len(read.slice_cells)
        return np.array(read.slice_operator, dtype=complex).reshape(size, size)

    def test_the_response_is_the_resolvent_form(self):
        samples = [complex(0.3, 0.2), complex(-1.1, 0.0)]
        read = self._read(samples)
        self.assertEqual(read.failed_certificates, [])
        operator = self._slice(read)
        left = np.array(read.rho_left, dtype=complex)
        right = np.array(read.rho_right, dtype=complex)
        identity = np.eye(operator.shape[0], dtype=complex)
        for index, point in enumerate(samples):
            expected = left @ np.linalg.solve(operator - point * identity,
                                              right)
            self.assertAlmostEqual(abs(read.response[index] - expected), 0.0,
                                   places=9)

    def test_the_slope_is_the_square_of_the_resolvent(self):
        samples = [complex(0.3, 0.2)]
        read = self._read(samples)
        operator = self._slice(read)
        left = np.array(read.rho_left, dtype=complex)
        right = np.array(read.rho_right, dtype=complex)
        shifted = operator - samples[0] * np.eye(operator.shape[0],
                                                 dtype=complex)
        expected = left @ np.linalg.solve(shifted,
                                          np.linalg.solve(shifted, right))
        self.assertAlmostEqual(abs(read.slope[0] - expected), 0.0, places=9)

    def test_the_poles_are_the_eigenvalues_of_the_slice_operator(self):
        read = self._read([])
        operator = self._slice(read)
        expected = sorted(np.linalg.eigvals(operator),
                          key=lambda value: (value.real, value.imag))
        found = sorted(read.poles, key=lambda value: (value.real, value.imag))
        self.assertEqual(len(found), len(expected))
        for left, right in zip(found, expected):
            self.assertAlmostEqual(abs(left - right), 0.0, places=8)
        self.assertEqual(sum(read.pole_multiplicity), operator.shape[0])

    def test_the_residues_are_the_spectral_contractions(self):
        """Each residue is the projector contraction its Riesz contour returns.

        For a slice operator with a simple spectrum the Riesz projector of a
        band is the outer product of that eigenvalue's right and left
        eigenvectors under the transpose pairing, so the residue can be checked
        against an eigendecomposition the read itself never took.
        """
        read = self._read([])
        operator = self._slice(read)
        left = np.array(read.rho_left, dtype=complex)
        right = np.array(read.rho_right, dtype=complex)
        values, vectors = np.linalg.eig(operator)
        inverse = np.linalg.inv(vectors)
        for index, pole in enumerate(read.poles):
            matches = [position for position in range(len(values))
                       if abs(values[position] - pole) < 1e-7]
            self.assertEqual(len(matches), read.pole_multiplicity[index])
            projector = sum(np.outer(vectors[:, position], inverse[position, :])
                            for position in matches)
            expected = -(left @ projector @ right)
            self.assertAlmostEqual(abs(read.residues[index] - expected), 0.0,
                                   places=7)

    def test_the_residues_reconstruct_the_response(self):
        """The partial fractions of the response are its poles and residues."""
        samples = [complex(2.5, 1.5)]
        read = self._read(samples)
        reconstructed = sum(
            residue / (samples[0] - pole)
            for pole, residue in zip(read.poles, read.residues))
        self.assertAlmostEqual(abs(read.response[0] - reconstructed), 0.0,
                               places=7)

    def test_a_sample_on_a_pole_is_refused_by_name(self):
        poles = self._read([]).poles
        read = self._read([poles[0]])
        self.assertIn("sample-on-a-pole", read.failed_certificates)
        self.assertTrue(np.isnan(read.response[0].real))
        self.assertTrue(np.isnan(read.slope[0].real))

    def test_the_restrictions_sum_to_the_flux(self):
        """The right restriction carries the cut's coorientation, so its sum is
        the flux the same cut reports."""
        response = cob.WardFlux.intrinsic_response(self.action, self.cut, [])
        flux = cob.WardFlux.flux(self.action, self.cut)
        self.assertAlmostEqual(abs(sum(response.rho_right) - flux.flux), 0.0,
                               places=12)

    def test_a_declared_left_current_is_used_as_given(self):
        """The left restriction is supplied rather than assumed.

        The whitepaper names a left and a right restriction of the current
        without saying how they differ, so the left one is an input. Scaling it
        scales the whole response, which is the statement that the read uses
        the current it was handed.
        """
        default = cob.WardFlux.intrinsic_response(self.action, self.cut,
                                                  [complex(0.3, 0.2)])
        config = cob.IntrinsicResponseConfig()
        config.left_current = [2.0 * value
                               for value in
                               self.action.canonical_ward_current()]
        doubled = cob.WardFlux.intrinsic_response(self.action, self.cut,
                                                  [complex(0.3, 0.2)], config)
        self.assertAlmostEqual(
            abs(doubled.response[0] - 2.0 * default.response[0]), 0.0,
            places=9)


if __name__ == "__main__":
    unittest.main()
