# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""#1200 — the Ward current's flux through a cooriented cut of the interaction
cobordism, and the intrinsic spectral response that cut carries.

Section 13.4 of the whitepaper defines the complex Ward current of the
gauge-invariant joint action, ``j_xy = U_xy dS/dU_xy``, and its flux
``phi_j(Sigma) = <j, Sigma>`` through a cooriented cut ``Sigma`` of the
interaction cobordism ``W``, whose oriented boundary is the incoming level
``d_in W`` and the outgoing level ``d_out W``. The cuts are the ones a quark
lineage is intersected with: the 0-cochain ``u`` of
``observables.CoorientedCut``, 0 on the incoming side and 1 on the outgoing
side, with ``Sigma`` dual to ``delta u``. Section 13.5 defines the intrinsic
spectral response ``Upsilon_Q(lambda)`` a cut carries when stable translations
have not emerged. What is asserted here, each as a measured number:

THE WARD IDENTITY. ``d j = 0`` is a consequence of gauge invariance: the
face-holonomy current is a boundary map applied to the faces and ``d_1 d_2 = 0``
annihilates it, and the matter term ``tr(Gamma h)`` is invariant when the
covariance commutes with the carrier operator, which a spectral projector of
that operator does. On ``W`` the divergence at every interior vertex is read
and must vanish.

THE DIVERGENCE THEOREM AND THE SLAB IDENTITY. With the boundary convention
``d[x<y] = [y] - [x]`` the flux is minus the divergence summed over the cut's
incoming side, and two cuts differ in flux by exactly the signed divergence the
slab between them carries. Both identities are exact.

THE RELATIVE READING ON THE WHITEPAPER'S OWN STATE. The acceptance case is the
quasi-free covariance ``Gamma = Phi PhiTilde^T`` with three occupied modes,
carried across the two-step cobordism ``W = W_1 u W_2`` (three levels, so that
``W`` has interior vertices and more than one separating cut). The read reports
the flux on two homologous cuts, the bulk divergence, the charge the current
brings in through ``d_in W``, and — independently of the current — the fermion
number the incoming state places on ``d_in W``. The whitepaper states that the
flux is that fermion number, ``N_q = 3`` and ``B = 1``. What is observed is
recorded as observed: the flux is zero to machine precision on every cut, the
bulk divergence vanishes, and the flux is not the incoming fermion number.

THE INTRINSIC RESPONSE IS A RESOLVENT FORM AND NOTHING ELSE. ``Upsilon_Q`` is
the bilinear resolvent of the slice operator between the two restrictions of
the current to the crossing edges; its poles are that operator's eigenvalues,
its residues come from Riesz contours that enclose a whole degenerate band, and
its slope in ``lambda`` is the square of the resolvent taken exactly.
"""

import cmath
import itertools
import math
import os
import sys
import unittest

import numpy as np

import tessera as T

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _joint_action_hosts import sphere3  # noqa: E402

cob = T.cobordism
obs = T.observables

# The levels of the fixture's history: the boundary of a tetrahedron, a closed
# triangulated 2-sphere with four vertices and six edges.
SPHERE2 = [list(cell) for cell in itertools.combinations(range(4), 3)]
# The occupied modes of the whitepaper's quasi-free state.
OCCUPIED = 3
# The declared action weights of the fixture.
HOLONOMY_WEIGHT = 0.8
MATTER_WEIGHT = 1.0


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
    # These suites exercise the dual (Sorkin) Regge form, declared explicitly
    # now that the primal form is the default.
    declaration.regge_form = cob.ReggeForm.Dual
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


def _history(levels):
    """``S^2 x I``, ``levels`` copies of the 2-sphere joined by identity
    reductions. Level ``l``'s own vertex ``v`` is ``W``'s vertex ``4 l + v``."""
    return obs.ClusterLineage.history(
        [obs.LevelComplex(SPHERE2, 4) for _ in range(levels)],
        [[0, 1, 2, 3] for _ in range(levels - 1)])


def _triangulate(W):
    """A spacetime triangulating ``W``, with the fixture's metric and
    connection written onto every edge by index.

    ``W`` is the product of a closed 2-sphere with an interval, so its top cells
    are the tetrahedra of the staircase prisms and it is pure.
    """
    top = [list(cell) for cell in W.cells if len(cell) == 4]
    spacetime = T.Spacetime.fromVertexTuples(3, top, 1.0, 0.0)
    for index, edge in enumerate(spacetime.getEdgeList().toVector()):
        edge.setLength(cmath.sqrt(_metric(index)))
        edge.setPhase(_flux_phase(index))
    return spacetime


def _incoming_state(spacetime, W):
    """The incoming level's own quasi-free state, embedded on ``d_in W``.

    The incoming level ``K_0`` is triangulated on its own, carrying the same
    squared length and the same connection on each of its edges as ``W`` does,
    and its covariance is the spectral projector of its own carrier operator
    onto three modes. That covariance is placed on ``W``'s level-0 edges and is
    zero on every other cell. Its trace is three by construction, so the
    incoming fermion number is three.
    """
    stored = {}
    for edge in spacetime.getEdgeList().toVector():
        source = int(edge.getSource().getId())
        target = int(edge.getTarget().getId())
        stored[(min(source, target), max(source, target))] = (
            edge.getLength(), edge.getPhase(), source < target)
    level = T.Spacetime.fromVertexTuples(2, SPHERE2, 1.0, 0.0)
    for edge in level.getEdgeList().toVector():
        source = int(edge.getSource().getId())
        target = int(edge.getTarget().getId())
        length, phase, ascending = stored[(min(source, target),
                                           max(source, target))]
        edge.setLength(length)
        edge.setPhase(phase if ascending == (source < target) else -phase)
    bare = cob.JointAction(level, _declaration(holonomy_weight=HOLONOMY_WEIGHT))
    local = np.array(bare.occupation_projector(OCCUPIED),
                     dtype=complex).reshape(6, 6)
    edges = [tuple(edge) for edge in W.edges]
    positions = [edges.index(edge) for edge in _canonical_edges(level)]
    full = np.zeros((len(edges), len(edges)), dtype=complex)
    full[np.ix_(positions, positions)] = local
    return [complex(value) for value in full.reshape(-1)]


class QuasiFreeHistory:
    """The whitepaper's quasi-free state carried across a two-step history.

    ``W`` is ``S^2 x I`` built from three levels, so it has an incoming level,
    an interior level and an outgoing level. The state is
    ``Gamma = Phi PhiTilde^T``, the spectral projector of ``W``'s own carrier
    operator onto its three lowest modes: a solution of the matter equations of
    motion, since it commutes with the carrier operator.
    """

    def __init__(self):
        self.W = _history(3)
        self.spacetime = _triangulate(self.W)
        bare = cob.JointAction(self.spacetime,
                               _declaration(holonomy_weight=HOLONOMY_WEIGHT))
        self.carrier = np.array(bare.carrier_operator(), dtype=complex)
        size = len(self.W.edges)
        self.carrier = self.carrier.reshape(size, size)
        self.covariance = bare.occupation_projector(OCCUPIED)
        self.action = self.action_for(self.covariance)
        self.cuts = [
            obs.ClusterLineage.levelCut(self.W, 0),
            obs.ClusterLineage.levelCut(self.W, 1),
            self.moved_cut(),
        ]

    def action_for(self, covariance):
        return cob.JointAction(
            self.spacetime,
            _declaration(holonomy_weight=HOLONOMY_WEIGHT,
                         matter_weight=MATTER_WEIGHT, covariance=covariance))

    def moved_cut(self):
        """The first level cut moved past two interior vertices."""
        side = list(obs.ClusterLineage.levelCut(self.W, 0).side)
        side[4] = 0
        side[6] = 0
        return obs.ClusterLineage.cutFromSides(self.W, side)

    def scale(self, action=None):
        current = (action or self.action).canonical_ward_current()
        return 1.0 + max(abs(value) for value in current)


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


class TheFixtureIsTheWhitepapersStateTest(unittest.TestCase):
    """The acceptance fixture is what it says it is."""

    @classmethod
    def setUpClass(cls):
        cls.history = QuasiFreeHistory()

    def test_the_action_is_declared_over_the_cobordism(self):
        self.assertEqual(_canonical_edges(self.history.spacetime),
                         [tuple(edge) for edge in self.history.W.edges])

    def test_the_covariance_is_a_rank_three_projector_commuting_with_h(self):
        size = len(self.history.W.edges)
        gamma = np.array(self.history.covariance,
                         dtype=complex).reshape(size, size)
        h = self.history.carrier
        self.assertLess(np.max(np.abs(gamma @ gamma - gamma)), 1e-10)
        self.assertLess(np.max(np.abs(h @ gamma - gamma @ h)),
                        1e-10 * np.max(np.abs(h)))
        self.assertAlmostEqual(abs(np.trace(gamma) - OCCUPIED), 0.0, places=10)

    def test_the_cuts_separate_and_are_homologous(self):
        for cut in self.history.cuts:
            self.assertTrue(cut.separates)
            self.assertEqual(cut.side[:4], [0, 0, 0, 0])
            self.assertEqual(cut.side[8:], [1, 1, 1, 1])
        self.assertNotEqual(list(self.history.cuts[0].crossingEdges),
                            list(self.history.cuts[1].crossingEdges))

    def test_a_three_sheeted_lineage_reads_baryon_number_one_on_these_cuts(self):
        """The cuts the flux is read on are the cuts the lineage number is
        read on: a cluster carrying three occupied sheets across ``W`` has
        ``N_q = 3`` and ``B = 1`` on every one of them."""
        lineage = obs.ClusterLineage.fromFiberPath(self.history.W, 0, 3, "q")
        for cut in self.history.cuts:
            totals = obs.ClusterLineage.totals(self.history.W, cut, [lineage])
            self.assertEqual(totals.fermionNumber, 3)
            self.assertAlmostEqual(totals.baryonNumber, 1.0, places=12)


class TheDivergenceTheoremIsExactTest(unittest.TestCase):
    """The flux is minus the divergence summed over the cut's incoming side,
    and two cuts differ by exactly the signed divergence of their slab."""

    @classmethod
    def setUpClass(cls):
        cls.history = QuasiFreeHistory()
        # A covariance that does not commute with the carrier operator, so
        # that the identities are exercised on a current with sources.
        size = len(cls.history.W.edges)
        rng = np.random.default_rng(1200)
        noise = rng.normal(size=(size, size)) + 1j * rng.normal(size=(size, size))
        cls.sourced = cls.history.action_for(
            [complex(value) for value in (0.1 * noise).reshape(-1)])

    def test_the_flux_is_minus_the_incoming_side_divergence(self):
        divergence = np.array(self.sourced.ward_current_divergence())
        for cut in self.history.cuts:
            read = cob.WardFlux.flux(self.sourced, self.history.W, cut)
            expected = -sum(divergence[vertex]
                            for vertex, side in enumerate(cut.side)
                            if side == 0)
            self.assertGreater(abs(read.flux), 1e-3)
            self.assertAlmostEqual(abs(read.flux - expected), 0.0, places=11)
            self.assertLess(read.divergence_theorem_residual, 1e-11)

    def test_the_flux_is_the_pairing_with_the_coboundary_of_the_cut(self):
        current = np.array(self.sourced.canonical_ward_current())
        for cut in self.history.cuts:
            read = cob.WardFlux.flux(self.sourced, self.history.W, cut)
            expected = sum(
                current[index] * (cut.side[b] - cut.side[a])
                for index, (a, b) in enumerate(self.history.W.edges))
            self.assertAlmostEqual(abs(read.flux - expected), 0.0, places=12)
            self.assertEqual(list(read.crossing_edges), list(cut.crossingEdges))
            self.assertEqual(list(read.crossing_signs), list(cut.crossingSigns))

    def test_two_cuts_differ_by_the_divergence_of_their_slab(self):
        read = cob.WardFlux.homologous_fluxes(self.sourced, self.history.W,
                                              self.history.cuts)
        self.assertEqual(len(read.slabs), 3)
        for slab in read.slabs:
            self.assertLess(slab.slab_identity_residual, 1e-11)
            self.assertTrue(set(slab.slab_vertices) <= {4, 5, 6, 7})
        self.assertGreater(read.max_slab_divergence, 1e-3)
        self.assertFalse(read.invariant)

    def test_a_sourced_current_is_named_a_bulk_source(self):
        read = cob.WardFlux.flux(self.sourced, self.history.W,
                                 self.history.cuts[0])
        self.assertIn("bulk-source", read.failed_certificates)
        self.assertIn(read.bulk_divergence_vertex, [4, 5, 6, 7])
        self.assertEqual(read.bulk_vertices, 4)


class TheQuasiFreeStateOnTheCobordismTest(unittest.TestCase):
    """#1200's acceptance case, read as it comes out.

    The whitepaper's statement is ``phi_j(Sigma) = N_q = 3`` and ``B = 1`` on
    every homologous cut, with the bulk divergence at machine precision, the
    fermion number being the charge the incoming state carries in through
    ``d_in W``. The bulk divergence and the homology invariance hold. The flux
    does not equal the incoming fermion number: it is zero on every cut, to
    machine precision.

    The reason is visible in the numbers. The divergence of ``j`` vanishes at
    every vertex of ``W``, boundary vertices included, because the matter term's
    divergence is the diagonal of ``[h, Gamma]`` gathered by vertex and
    ``Gamma`` commutes with ``h`` on all of ``W``. The divergence theorem then
    makes the flux through every cut exactly zero, and the charge the current
    brings in through ``d_in W`` is zero as well.
    """

    @classmethod
    def setUpClass(cls):
        cls.history = QuasiFreeHistory()
        cls.reads = [cob.WardFlux.flux(cls.history.action, cls.history.W, cut)
                     for cut in cls.history.cuts]
        cls.scale = cls.history.scale()

    def test_the_bulk_divergence_is_at_machine_precision(self):
        for read in self.reads:
            self.assertEqual(read.bulk_vertices, 4)
            self.assertLess(read.bulk_divergence_max, 1e-12 * self.scale)
            self.assertNotIn("bulk-source", read.failed_certificates)
            self.assertTrue(read.cut_separates)

    def test_homologous_cuts_read_the_same_flux(self):
        read = cob.WardFlux.homologous_fluxes(
            self.history.action, self.history.W, self.history.cuts)
        self.assertLess(read.max_flux_deviation, 1e-12 * self.scale)
        self.assertLess(read.max_slab_divergence, 1e-12 * self.scale)
        self.assertTrue(read.invariant)

    def test_the_current_crossing_each_cut_is_not_small(self):
        """The flux is a cancellation across the crossing edges, not the
        absence of a current: each crossing edge carries a current of order
        one."""
        for read in self.reads:
            self.assertGreater(max(abs(value)
                                   for value in read.crossing_current), 0.1)

    def test_the_flux_is_zero_on_every_cut(self):
        """Observed: ``phi_j(Sigma) = 0`` to machine precision, so
        ``N_q = 0`` and ``B = 0`` rather than ``3`` and ``1``."""
        for read in self.reads:
            self.assertLess(abs(read.flux), 1e-12 * self.scale)
            self.assertLess(abs(read.incoming_boundary_divergence),
                            1e-12 * self.scale)
            self.assertEqual(read.quark_number, 0)
            self.assertAlmostEqual(read.baryon_number, 0.0, places=12)

    def test_the_divergence_vanishes_on_the_boundary_as_well(self):
        divergence = np.array(self.history.action.ward_current_divergence())
        self.assertLess(np.max(np.abs(divergence)), 1e-12 * self.scale)

    def test_the_incoming_charge_of_the_cobordism_state_is_not_an_integer(self):
        """Observed: the projector of ``W``'s carrier operator places
        ``0.22098 - 0.01896 i`` of its three units on the six edges of
        ``d_in W``, ``0.16448 - 0.00872 i`` on those of ``d_out W``, and the
        rest on the interior level and the edges between levels."""
        read = self.reads[0]
        self.assertEqual(read.incoming_boundary_cells, 6)
        self.assertEqual(read.outgoing_boundary_cells, 6)
        self.assertAlmostEqual(read.incoming_boundary_charge.real, 0.22098,
                               places=4)
        self.assertAlmostEqual(read.incoming_boundary_charge.imag, -0.01896,
                               places=4)
        self.assertAlmostEqual(read.outgoing_boundary_charge.real, 0.16448,
                               places=4)
        self.assertAlmostEqual(read.outgoing_boundary_charge.imag, -0.00872,
                               places=4)
        occupations = self.history.action.occupation_numbers()
        self.assertAlmostEqual(abs(sum(occupations) - OCCUPIED), 0.0,
                               places=10)
        self.assertAlmostEqual(
            read.boundary_charge_residual,
            abs(read.flux - read.incoming_boundary_charge), places=12)
        self.assertIn("flux-is-not-the-incoming-charge",
                      read.failed_certificates)

    def test_the_incoming_levels_own_state_carries_three_and_the_flux_none(self):
        """Observed: the incoming level's own quasi-free state, three
        occupied modes of ``K_0``'s carrier operator placed on ``d_in W``,
        carries ``Q_in = 3`` exactly. The current it drives has divergence of
        order one at the vertices of ``d_in W`` and none in the bulk, and that
        boundary divergence sums to zero, so the flux through every cut is again
        zero."""
        action = self.history.action_for(
            _incoming_state(self.history.spacetime, self.history.W))
        scale = self.history.scale(action)
        divergence = np.array(action.ward_current_divergence())
        self.assertGreater(np.max(np.abs(divergence[:4])), 0.1)
        self.assertLess(np.max(np.abs(divergence[4:])), 1e-12 * scale)
        read = cob.WardFlux.homologous_fluxes(action, self.history.W,
                                              self.history.cuts)
        self.assertTrue(read.invariant)
        for cut in read.cuts:
            self.assertAlmostEqual(abs(cut.incoming_boundary_charge - OCCUPIED),
                                   0.0, places=10)
            self.assertAlmostEqual(abs(cut.outgoing_boundary_charge), 0.0,
                                   places=12)
            self.assertLess(abs(cut.flux), 1e-12 * scale)
            self.assertLess(cut.bulk_divergence_max, 1e-12 * scale)
            self.assertAlmostEqual(cut.boundary_charge_residual, OCCUPIED,
                                   places=10)
            self.assertIn("flux-is-not-the-incoming-charge",
                          cut.failed_certificates)
            self.assertEqual(cut.quark_number, 0)


class TheReadRefusesByNameTest(unittest.TestCase):
    """A cut or a complex the read cannot use is named, or refused."""

    @classmethod
    def setUpClass(cls):
        cls.history = QuasiFreeHistory()

    def test_a_cut_that_does_not_separate_is_named(self):
        side = [0] * 12
        cut = obs.ClusterLineage.cutFromSides(self.history.W, side)
        read = cob.WardFlux.flux(self.history.action, self.history.W, cut)
        self.assertFalse(read.cut_separates)
        self.assertIn("cut-does-not-separate", read.failed_certificates)
        self.assertIn("empty-cut", read.failed_certificates)

    def test_a_history_of_two_levels_has_no_bulk_to_certify(self):
        W = _history(2)
        spacetime = _triangulate(W)
        bare = cob.JointAction(spacetime,
                               _declaration(holonomy_weight=HOLONOMY_WEIGHT))
        action = cob.JointAction(
            spacetime,
            _declaration(holonomy_weight=HOLONOMY_WEIGHT,
                         matter_weight=MATTER_WEIGHT,
                         covariance=bare.occupation_projector(OCCUPIED)))
        read = cob.WardFlux.flux(action, W, obs.ClusterLineage.levelCut(W, 0))
        self.assertEqual(read.bulk_vertices, 0)
        self.assertTrue(math.isnan(read.bulk_divergence_max))
        self.assertIsNone(read.bulk_divergence_vertex)
        self.assertIn("no-interior-vertex", read.failed_certificates)

    def test_an_action_over_another_complex_is_refused(self):
        action = cob.JointAction(sphere3(squared=_metric, phase=_flux_phase),
                                 _declaration(holonomy_weight=0.8))
        with self.assertRaises(ValueError):
            cob.WardFlux.flux(action, self.history.W, self.history.cuts[0])
        with self.assertRaises(ValueError):
            cob.WardFlux.intrinsic_response(action, self.history.W,
                                            self.history.cuts[0], [])

    def test_an_action_without_a_covariance_reports_no_charge(self):
        action = cob.JointAction(self.history.spacetime,
                                 _declaration(holonomy_weight=0.8))
        read = cob.WardFlux.flux(action, self.history.W, self.history.cuts[0])
        self.assertIsNone(read.incoming_boundary_charge)
        self.assertIsNone(read.outgoing_boundary_charge)
        self.assertTrue(math.isnan(read.boundary_charge_residual))
        self.assertNotIn("flux-is-not-the-incoming-charge",
                         read.failed_certificates)


class TheBackgroundRemovalIsCoherentTest(unittest.TestCase):
    """``Delta O = O_state - O_matched`` is formed before anything else."""

    @classmethod
    def setUpClass(cls):
        cls.history = QuasiFreeHistory()

    def test_the_difference_subtracts_the_complex_numbers(self):
        size = len(self.history.W.edges)
        cut = self.history.cuts[0]
        state = cob.WardFlux.flux(self.history.action, self.history.W, cut)
        matched = cob.WardFlux.flux(
            self.history.action_for([0j] * (size * size)), self.history.W, cut)
        removed = cob.WardFlux.difference(state, matched)
        self.assertAlmostEqual(
            abs(removed.flux - (state.flux - matched.flux)), 0.0, places=12)
        self.assertAlmostEqual(
            abs(removed.incoming_boundary_charge
                - (state.incoming_boundary_charge
                   - matched.incoming_boundary_charge)), 0.0, places=12)
        for index, value in enumerate(removed.crossing_current):
            self.assertAlmostEqual(
                abs(value - (state.crossing_current[index]
                             - matched.crossing_current[index])), 0.0,
                places=12)
        self.assertLess(removed.divergence_theorem_residual, 1e-10)

    def test_a_difference_between_different_cuts_is_refused(self):
        state = cob.WardFlux.flux(self.history.action, self.history.W,
                                  self.history.cuts[0])
        matched = cob.WardFlux.flux(self.history.action, self.history.W,
                                    self.history.cuts[1])
        with self.assertRaises(ValueError):
            cob.WardFlux.difference(state, matched)


class TheIntrinsicSpectralResponseTest(unittest.TestCase):
    """``Upsilon_Q`` is the bilinear resolvent of the slice operator."""

    @classmethod
    def setUpClass(cls):
        cls.history = QuasiFreeHistory()
        cls.action = cls.history.action
        cls.W = cls.history.W
        cls.cut = cls.history.cuts[0]

    def _read(self, samples, config=None):
        if config is None:
            return cob.WardFlux.intrinsic_response(self.action, self.W,
                                                   self.cut, samples)
        return cob.WardFlux.intrinsic_response(self.action, self.W, self.cut,
                                               samples, config)

    def _slice(self, read):
        size = len(read.slice_cells)
        return np.array(read.slice_operator, dtype=complex).reshape(size, size)

    def test_the_slice_is_the_cut_crossing_edges(self):
        read = self._read([])
        self.assertEqual(list(read.slice_cells), list(self.cut.crossingEdges))

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

        The Riesz projector of a band is the sum of the outer products of its
        right and left eigenvectors under the transpose pairing, so the residue
        can be checked against an eigendecomposition the read itself never
        took.
        """
        read = self._read([])
        self.assertNotIn("bands-not-separated", read.failed_certificates)
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
        response = self._read([])
        flux = cob.WardFlux.flux(self.action, self.W, self.cut)
        self.assertAlmostEqual(abs(sum(response.rho_right) - flux.flux), 0.0,
                               places=12)
        self.assertGreater(max(abs(value) for value in response.rho_right),
                           0.1)

    def test_a_declared_left_current_is_used_as_given(self):
        """The left restriction is supplied rather than assumed.

        The whitepaper names a left and a right restriction of the current
        without saying how they differ, so the left one is an input. Scaling it
        scales the whole response, which is the statement that the read uses
        the current it was handed.
        """
        default = self._read([complex(0.3, 0.2)])
        config = cob.IntrinsicResponseConfig()
        config.left_current = [2.0 * value
                               for value in self.action.canonical_ward_current()]
        doubled = self._read([complex(0.3, 0.2)], config)
        self.assertAlmostEqual(
            abs(doubled.response[0] - 2.0 * default.response[0]), 0.0,
            places=9)


if __name__ == "__main__":
    unittest.main()
