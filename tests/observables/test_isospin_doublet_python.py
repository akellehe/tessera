# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The observation of the isospin doublet (:class:`tessera.observables.IsospinDoublet`),
held to whitepaper v16.

* Section 10: "two stable subclasses of the same cluster fiber provide an
  isospin doublet ... it succeeds only if an unlabeled two-dimensional spectral
  band emerges, is transported coherently, and its flavor-derived current agrees
  with the microscopic Ward-current flux".
* Section 8: colour is sheet multiplicity, so a band E-bar (x) C^3 is not a
  flavour band. Section 11.1: the doublets 2, 2', 2'' of the unit-monopole
  tetrahedron are spin content.
* Falsifier 8 ("No isospin doublet") and falsifier 10 ("Unexpected
  multiplicity").

The fixtures are built from the library's unit-monopole tetrahedron
(`MonopoleSupport.tetrahedron(1)`): its rotation-averaged edge Laplacian
carries the three spinor doublets. A constructed flavour doubling
h-bar (x) I_2 (x) I_3 has a genuine two-dimensional multiplicity space that
neither the rotations nor the sheets act on; the detector must find it there,
and must find none on the sheets alone.
"""
import unittest

import numpy as np

import tessera

obs = tessera.observables

Passed = obs.QuarkConditionStatus.Passed
Failed = obs.QuarkConditionStatus.Failed
NotEvaluable = obs.QuarkConditionStatus.NotEvaluable


def monopole_base():
    """The averaged edge operator of the unit-monopole tetrahedron (three
    spinor doublets), the projective rotation action on its six edges, and
    whether the action's class is nontrivial."""
    support = obs.MonopoleSupport.tetrahedron(1)
    group = obs.MonopoleSupport.tetrahedralRotations()
    averaged = np.asarray(support.rotationAveragedEdgeOperator(
        support.edgeLaplacian(), group))
    actions = [np.asarray(support.edgeRepresentation(g)) for g in group]
    spinorial = bool(support.cocycle(group).nontrivial)
    return averaged, actions, spinorial


def symmetric_deformation(actions, seed=7):
    """A rotation-invariant Hermitian 6 x 6 operator: the rotation average of
    a fixed random Hermitian matrix."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(6, 6)) + 1j * rng.normal(size=(6, 6))
    x = x + x.conj().T
    return sum(d @ x @ np.linalg.inv(d) for d in actions) / len(actions)


def flavoured(base, flavours=2, sheets=3):
    """base (x) I_flavours (x) I_sheets, cells (b, a, t) at (b * F + a) * S + t;
    the base cell of a sheet copy is (b, a)."""
    op = np.kron(np.kron(base, np.eye(flavours)), np.eye(sheets))
    n = base.shape[0] * flavours * sheets
    sheet_of = [i % sheets for i in range(n)]
    base_of = [i // sheets for i in range(n)]
    return op, sheet_of, base_of


def lifted(actions, flavours=2, sheets=3):
    return [np.kron(np.kron(d, np.eye(flavours)), np.eye(sheets))
            for d in actions]


def lineage_read(reverse=False):
    """A certified lineage reading N_Q = +1 (or -1) through a level cut of a
    two-level product history."""
    identity = [0, 1, 2, 3]
    W = obs.ClusterLineage.history(
        [obs.LevelComplex([[0, 1, 2, 3]], 4) for _ in range(2)], [identity])
    cut = obs.ClusterLineage.levelCut(W, 0)
    lineage = obs.ClusterLineage.fromFiberPath(W, 0, 1, "Q")
    if reverse:
        lineage = obs.ClusterLineage.reversed(lineage)
    return obs.ClusterLineage.read(W, cut, lineage)


def declaration(frames, sheet_of, base_of, symmetry, spinorial,
                resolutions=(), name="fixture"):
    d = obs.IsospinDoubletDeclaration()
    d.operator_name = name
    d.sheet_of_cell = list(sheet_of)
    d.base_cell_of_cell = list(base_of)
    d.symmetry = list(symmetry)
    d.symmetry_name = "T (projective)"
    d.spinorial = spinorial
    d.frames = [obs.IsospinFrame("frame %d" % i, f) if not isinstance(f, tuple)
                else obs.IsospinFrame("frame %d" % i, f[0], f[1])
                for i, f in enumerate(frames)]
    d.resolutions = list(resolutions)
    return d


def padded_resolution(op, sheet_of, base_of, symmetry, extra=3, shift=50.0):
    """A second resolution of the same cluster: the coarse cells plus `extra`
    further cells per sheet whose spectrum sits far away, with the coarse
    cochains prolonged by inclusion. The sheet and symmetry declarations are
    extended by the identity on the added cells."""
    n = op.shape[0]
    sheets = max(sheet_of) + 1
    m = n + extra * sheets
    fine = np.zeros((m, m), dtype=complex)
    fine[:n, :n] = op
    fine[n:, n:] = shift * np.eye(extra * sheets)
    res = obs.IsospinResolution()
    res.label = "padded"
    res.operator = fine
    res.prolongation = np.vstack([np.eye(n), np.zeros((extra * sheets, n))])
    res.sheet_of_cell = list(sheet_of) + [i % sheets for i in range(extra * sheets)]
    base_count = max(base_of) + 1
    res.base_cell_of_cell = list(base_of) + [base_count + i // sheets
                                             for i in range(extra * sheets)]
    res.symmetry = [np.block([[d, np.zeros((n, extra * sheets))],
                              [np.zeros((extra * sheets, n)),
                               np.eye(extra * sheets)]]) for d in symmetry]
    return res


def status(read, number):
    return read.conditions[number - 1].status


class TestBandContent(unittest.TestCase):
    """What each band is, before any condition is read."""

    def setUp(self):
        self.base, self.actions, self.spinorial = monopole_base()

    def test_the_monopole_base_is_three_spinor_doublets(self):
        self.assertTrue(self.spinorial)
        frame = obs.IsospinDoublet.bands(self.base, symmetry=self.actions,
                                         spinorial=True)
        self.assertEqual([b.rank for b in frame.bands], [2, 2, 2])
        for band in frame.bands:
            self.assertTrue(band.symmetry_acts)
            self.assertEqual(band.commutant_dimension, 1)
            self.assertTrue(band.spin_doublet)
            self.assertFalse(band.doublet_candidate)
            self.assertFalse(band.unexplained_multiplicity)

    def test_sheets_are_colour_not_flavour(self):
        op, sheet_of, base_of = flavoured(self.base, flavours=1)
        frame = obs.IsospinDoublet.bands(
            op, sheet_of_cell=sheet_of, base_cell_of_cell=base_of,
            symmetry=lifted(self.actions, flavours=1), spinorial=True)
        self.assertEqual([b.rank for b in frame.bands], [6, 6, 6])
        for band in frame.bands:
            self.assertTrue(band.colour_acts)
            self.assertEqual(band.content, "2 x 3 sheets x 1")
            self.assertTrue(band.spin_doublet)
            self.assertFalse(band.doublet_candidate)

    def test_undeclared_sheets_read_as_an_unexplained_multiplicity(self):
        op = np.kron(self.base, np.eye(3))
        frame = obs.IsospinDoublet.bands(
            op, symmetry=[np.kron(d, np.eye(3)) for d in self.actions],
            spinorial=True)
        for band in frame.bands:
            self.assertEqual(band.content, "2 x 1 sheet x 3")
            self.assertTrue(band.unexplained_multiplicity)
            self.assertFalse(band.doublet_candidate)

    def test_a_flavour_doubling_is_a_candidate(self):
        op, sheet_of, base_of = flavoured(self.base)
        frame = obs.IsospinDoublet.bands(
            op, sheet_of_cell=sheet_of, base_cell_of_cell=base_of,
            symmetry=lifted(self.actions), spinorial=True)
        self.assertEqual([b.rank for b in frame.bands], [12, 12, 12])
        for band in frame.bands:
            self.assertEqual(band.content, "2 x 3 sheets x 2")
            self.assertEqual(band.commutant_dimension, 4)
            self.assertEqual(band.isotype_count, 1)
            self.assertTrue(band.doublet_candidate)
            self.assertFalse(band.spin_doublet)
            # falsifier 10: a flavour multiplicity is not an irreducible
            # dimension of the declared group, and it is reported as such.
            self.assertTrue(band.unexplained_multiplicity)
            self.assertTrue(band.isolated)
            self.assertLess(band.projector_residual, 1e-9)

    def test_a_split_doublet_is_two_bands_not_one(self):
        op, sheet_of, base_of = flavoured(self.base)
        split = np.kron(np.kron(self.base, np.diag([1.0, 1.05])), np.eye(3))
        frame = obs.IsospinDoublet.bands(
            split, sheet_of_cell=sheet_of, base_cell_of_cell=base_of,
            symmetry=lifted(self.actions), spinorial=True)
        self.assertEqual(sorted(b.rank for b in frame.bands), [6] * 6)
        self.assertFalse(any(b.doublet_candidate for b in frame.bands))

    def test_two_inequivalent_irreducibles_degenerate_by_accident(self):
        # Two different doublets (2 and 2') at one eigenvalue: reducible with
        # two isotypes, reported under falsifier 10 and not a candidate.
        values, vectors = np.linalg.eigh(self.base)
        accidental = vectors @ np.diag([1.0, 1.0, 1.0, 1.0, 5.0, 5.0]) \
            @ vectors.conj().T
        frame = obs.IsospinDoublet.bands(accidental, symmetry=self.actions,
                                         spinorial=True)
        four = [b for b in frame.bands if b.rank == 4][0]
        self.assertEqual(four.isotype_count, 2)
        self.assertEqual(four.multiplicities, [1, 1])
        self.assertTrue(four.unexplained_multiplicity)
        self.assertFalse(four.doublet_candidate)

    def test_no_symmetry_declared_leaves_spin_unread(self):
        rng = np.random.default_rng(3)
        a = rng.normal(size=(4, 4))
        a = a + a.T
        frame = obs.IsospinDoublet.bands(np.kron(a, np.eye(2)))
        self.assertTrue(all(b.doublet_candidate for b in frame.bands))
        self.assertTrue(all("no symmetry was declared" in b.classification
                            for b in frame.bands))


class TestConditions(unittest.TestCase):
    """Each of the three conditions of Section 10, separately."""

    def setUp(self):
        self.base, self.actions, self.spinorial = monopole_base()
        self.deformation = symmetric_deformation(self.actions)
        self.op, self.sheet_of, self.base_of = flavoured(self.base)
        self.op1, _, _ = flavoured(self.base + 0.05 * self.deformation)
        self.symmetry = lifted(self.actions)

    def observed(self, frames, resolutions=None, **extra):
        if resolutions is None:
            resolutions = [padded_resolution(self.op, self.sheet_of,
                                             self.base_of, self.symmetry)]
        d = declaration(frames, self.sheet_of, self.base_of, self.symmetry,
                        True, resolutions)
        for key, value in extra.items():
            setattr(d, key, value)
        return obs.IsospinDoublet.observe(d)

    def test_the_genuine_doublet_is_observed(self):
        read = self.observed([self.op, self.op1], lineage=lineage_read())
        self.assertEqual(len(read.candidates), 3)
        self.assertEqual(status(read, 1), Passed)
        self.assertEqual(status(read, 2), Passed)
        self.assertEqual(status(read, 3), NotEvaluable)
        self.assertTrue(read.doublet_observed)
        self.assertFalse(read.no_isospin_doublet)
        candidate = read.candidates[0]
        self.assertTrue(candidate.observed)
        for step in candidate.transports:
            self.assertTrue(step.transport.invertible)
            self.assertLess(step.intertwining_residual, 1e-8)
            self.assertEqual(len(step.flavour_singular_values), 2)

    def test_condition_three_is_never_passed(self):
        read = self.observed([self.op, self.op1])
        for candidate in read.candidates:
            ward = candidate.conditions[2]
            self.assertEqual(ward.status, NotEvaluable)
            self.assertEqual(ward.missing, ["ward-flux-agreement"])
            self.assertIn("deferred", ward.evidence[0].detail)

    def test_one_frame_leaves_persistence_and_transport_unmeasured(self):
        read = self.observed([self.op])
        self.assertEqual(status(read, 1), NotEvaluable)
        self.assertIn("frame-persistence", read.conditions[0].missing)
        self.assertEqual(status(read, 2), NotEvaluable)
        self.assertFalse(read.doublet_observed)
        self.assertTrue(read.no_isospin_doublet)

    def test_no_resolution_leaves_refinement_unmeasured(self):
        read = self.observed([self.op, self.op1], resolutions=[])
        self.assertEqual(status(read, 1), NotEvaluable)
        self.assertEqual(read.conditions[0].missing, ["refinement-persistence"])

    def test_a_doublet_that_splits_in_the_next_frame_fails_persistence(self):
        split, _, _ = flavoured(self.base)
        split = np.kron(np.kron(self.base, np.diag([1.0, 1.05])), np.eye(3))
        read = self.observed([self.op, split])
        self.assertEqual(status(read, 1), Failed)
        self.assertIn("frame-persistence", read.conditions[0].failing)
        self.assertFalse(read.doublet_observed)

    def test_a_refinement_that_splits_the_doublet_fails_it(self):
        split = np.kron(np.kron(self.base, np.diag([1.0, 1.05])), np.eye(3))
        res = padded_resolution(split, self.sheet_of, self.base_of,
                                self.symmetry)
        read = self.observed([self.op, self.op1], resolutions=[res])
        self.assertEqual(status(read, 1), Failed)
        self.assertIn("refinement-persistence", read.conditions[0].failing)

    def test_a_leaking_transfer_fails_coherent_transport(self):
        rng = np.random.default_rng(11)
        n = self.op.shape[0]
        transfer = np.eye(n) + 0.15 * rng.normal(size=(n, n)) / np.sqrt(n)
        read = self.observed([self.op, (self.op1, transfer)])
        candidates = [c for c in read.candidates
                      if len(c.tracked_bands) == 2]
        self.assertTrue(candidates)
        for candidate in candidates:
            self.assertEqual(candidate.conditions[1].status, Failed)
            self.assertIn("transport-leakage", candidate.conditions[1].failing)

    def test_undeclared_symmetry_cannot_pass_emergence(self):
        rng = np.random.default_rng(3)
        a = rng.normal(size=(4, 4))
        a = a + a.T
        op = np.kron(a, np.eye(2))
        d = declaration([op, op], [], [], [], False)
        read = obs.IsospinDoublet.observe(d)
        self.assertEqual(status(read, 1), NotEvaluable)
        self.assertIn("not-spin-irreducible", read.conditions[0].missing)

    def test_the_colour_triplet_alone_fails_emergence_by_name(self):
        op, sheet_of, base_of = flavoured(self.base, flavours=1)
        d = declaration([op, op], sheet_of, base_of,
                        lifted(self.actions, flavours=1), True)
        read = obs.IsospinDoublet.observe(d)
        self.assertEqual(read.candidates, [])
        self.assertEqual(status(read, 1), Failed)
        self.assertEqual(read.conditions[0].failing,
                         ["two-dimensional-flavour-band"])
        self.assertEqual(status(read, 2), NotEvaluable)
        self.assertEqual(status(read, 3), NotEvaluable)
        self.assertTrue(read.no_isospin_doublet)
        self.assertEqual(read.unexplained_multiplicities, [])


class TestIsospinAndCharge(unittest.TestCase):
    """I_3, Q = I_3 + B/2 with B from the lineage, and uud / udd."""

    def setUp(self):
        self.base, self.actions, _ = monopole_base()
        self.op, self.sheet_of, self.base_of = flavoured(self.base)
        self.op1, _, _ = flavoured(
            self.base + 0.05 * symmetric_deformation(self.actions))
        self.symmetry = lifted(self.actions)

    def read(self, **extra):
        d = declaration([self.op, self.op1], self.sheet_of, self.base_of,
                        self.symmetry, True,
                        [padded_resolution(self.op, self.sheet_of,
                                           self.base_of, self.symmetry)])
        for key, value in extra.items():
            setattr(d, key, value)
        return obs.IsospinDoublet.observe(d)

    def test_charges_follow_from_the_lineage(self):
        charges = self.read(lineage=lineage_read()).candidates[0].charges
        self.assertEqual(list(charges.isospin), [0.5, -0.5])
        self.assertAlmostEqual(charges.baryon_number, 1.0 / 3.0)
        self.assertAlmostEqual(charges.charges[0], 2.0 / 3.0)
        self.assertAlmostEqual(charges.charges[1], -1.0 / 3.0)
        self.assertEqual(charges.member_source,
                         "declared trivialization (cell-order seed)")

    def test_a_reversed_lineage_gives_the_antiquark_charges(self):
        charges = self.read(
            lineage=lineage_read(reverse=True)).candidates[0].charges
        self.assertAlmostEqual(charges.baryon_number, -1.0 / 3.0)
        self.assertAlmostEqual(charges.charges[0], 1.0 / 3.0)
        self.assertAlmostEqual(charges.charges[1], -2.0 / 3.0)

    def test_no_lineage_leaves_the_charges_unmeasured(self):
        charges = self.read().candidates[0].charges
        self.assertIsNone(charges.baryon_number)
        self.assertEqual(list(charges.charges), [])

    def test_the_members_split_the_band_in_half(self):
        charges = self.read().candidates[0].charges
        plus, minus = [np.asarray(p) for p in charges.member_projectors]
        self.assertAlmostEqual(np.trace(plus).real, 6.0)
        self.assertAlmostEqual(np.trace(minus).real, 6.0)
        self.assertLess(np.linalg.norm(plus @ minus), 1e-8)
        # each member is itself a spin doublet times the sheets: it commutes
        # with the rotations and the sheets.
        for d in self.symmetry:
            self.assertLess(np.linalg.norm(d @ plus - plus @ d), 1e-8)

    def test_the_occupation_pattern_is_read_from_the_density(self):
        charges = self.read().candidates[0].charges
        plus, minus = [np.asarray(p) for p in charges.member_projectors]

        def modes(projector, count):
            values, vectors = np.linalg.eigh((projector + projector.conj().T) / 2)
            return vectors[:, np.argsort(values)[::-1][:count]]

        up, down = modes(plus, 2), modes(minus, 1)
        gamma = up @ up.conj().T + down @ down.conj().T
        self.assertEqual(
            self.read(three_quark_density=gamma).candidates[0].charges
            .occupation_pattern, "uud")
        up, down = modes(plus, 1), modes(minus, 2)
        gamma = up @ up.conj().T + down @ down.conj().T
        self.assertEqual(
            self.read(three_quark_density=gamma).candidates[0].charges
            .occupation_pattern, "udd")

    def test_a_splitting_operator_orders_the_members(self):
        splitting = np.kron(np.kron(np.eye(6), np.diag([1.0, -1.0])),
                            np.eye(3))
        charges = self.read(member_splitting=splitting).candidates[0].charges
        self.assertEqual(charges.member_source, "declared splitting operator")
        plus = np.asarray(charges.member_projectors[0])
        # the +1/2 member is the flavour-0 copy
        flavour0 = np.kron(np.kron(np.eye(6), np.diag([1.0, 0.0])), np.eye(3))
        self.assertLess(np.linalg.norm(flavour0 @ plus - plus), 1e-8)


class TestTheReportedFields(unittest.TestCase):
    """The fields of the band, candidate, transport and whole reads, and the
    thresholds of `IsospinDoubletConfig`."""

    def setUp(self):
        self.base, self.actions, _ = monopole_base()
        self.op, self.sheet_of, self.base_of = flavoured(self.base)
        self.op1, _, _ = flavoured(
            self.base + 0.05 * symmetric_deformation(self.actions))
        self.symmetry = lifted(self.actions)

    def declared(self, frames=None, resolutions=True):
        return declaration(
            frames or [self.op, self.op1], self.sheet_of, self.base_of,
            self.symmetry, True,
            [padded_resolution(self.op, self.sheet_of, self.base_of,
                               self.symmetry)] if resolutions else [])

    def test_the_config_defaults(self):
        config = obs.IsospinDoubletConfig()
        self.assertEqual(config.min_frames, 2)
        self.assertEqual(config.contour_nodes, 64)
        self.assertGreater(config.track_overlap_threshold, 0.0)
        self.assertLess(config.track_overlap_threshold, 1.0)
        self.assertGreater(config.condition_number_cap, 1.0)
        for name in ("grouping_tolerance", "min_relative_gap",
                     "projector_tolerance", "invariance_tolerance",
                     "commutant_tolerance", "transport_leakage_tolerance",
                     "intertwining_tolerance"):
            self.assertGreater(getattr(config, name), 0.0, msg=name)

    def test_more_required_frames_than_supplied_leave_the_conditions_open(
            self):
        config = obs.IsospinDoubletConfig()
        config.min_frames = 3
        read = obs.IsospinDoublet.observe(self.declared(), config)
        self.assertEqual(status(read, 1), NotEvaluable)
        self.assertEqual(status(read, 2), NotEvaluable)

    def test_a_band_read_reports_its_content(self):
        read = obs.IsospinDoublet.observe(self.declared())
        band = read.frames[0].bands[0]
        self.assertAlmostEqual(band.center, np.mean(band.eigenvalues),
                               places=12)
        self.assertAlmostEqual(band.contour_center, band.center, places=12)
        self.assertEqual(list(band.irreducible_dimensions), [2])
        self.assertTrue(band.symmetry_declared)
        self.assertLess(band.sheet_invariance_residual, 1e-12)
        self.assertLess(band.symmetry_invariance_residual, 1e-12)
        self.assertTrue(read.multiplicity_refinement_measured)

    def test_a_candidate_reports_its_track_and_its_transports(self):
        read = obs.IsospinDoublet.observe(self.declared())
        candidate = read.candidates[0]
        self.assertEqual(candidate.band_index, candidate.tracked_bands[0])
        self.assertAlmostEqual(candidate.min_track_overlap, 1.0, places=10)
        self.assertEqual(list(candidate.resolution_found), [True])
        np.testing.assert_allclose(candidate.resolution_overlap, [1.0],
                                   atol=1e-10)
        np.testing.assert_allclose(candidate.lifetime_singular_values, 1.0,
                                   atol=1e-10)
        (step,) = candidate.transports
        self.assertEqual((step.from_frame, step.to_frame), (0, 1))

    def test_without_a_resolution_refinement_is_unmeasured(self):
        read = obs.IsospinDoublet.observe(self.declared(resolutions=False))
        self.assertFalse(read.multiplicity_refinement_measured)
        for candidate in read.candidates:
            self.assertEqual(list(candidate.resolution_found), [])

    def test_a_declared_transfer_leaves_the_flavour_structure_unmeasured(
            self):
        """A nonempty transfer declares a map between two cell sets, which
        share no symmetry action, so the intertwining of the flavour factor is
        not measured and coherent transport is not evaluable; the empty
        default is the identity between frames of one cell set."""
        size = self.op.shape[0]
        frames = [self.op, (self.op1, np.eye(size, dtype=complex))]
        read = obs.IsospinDoublet.observe(self.declared(frames))
        self.assertEqual(status(read, 1), Passed)
        self.assertEqual(status(read, 2), NotEvaluable)
        self.assertEqual(read.candidates[0].conditions[1].missing,
                         ["flavour-structure-preserved"])
        self.assertEqual(obs.IsospinFrame("x", self.op).transfer_from_previous
                         .shape, (0, 0))

    def test_a_zero_transfer_breaks_the_track(self):
        size = self.op.shape[0]
        frames = [self.op, (self.op1, np.zeros((size, size), dtype=complex))]
        read = obs.IsospinDoublet.observe(self.declared(frames))
        self.assertEqual(status(read, 2), Failed)

    def test_member_occupations_are_read_from_the_density(self):
        charges = obs.IsospinDoublet.observe(self.declared()).candidates[0] \
            .charges
        plus, minus = [np.asarray(p) for p in charges.member_projectors]
        self.assertEqual(list(charges.member_occupations), [])
        values, vectors = np.linalg.eigh((plus + plus.conj().T) / 2)
        up = vectors[:, np.argsort(values)[::-1][:2]]
        values, vectors = np.linalg.eigh((minus + minus.conj().T) / 2)
        down = vectors[:, np.argsort(values)[::-1][:1]]
        d = self.declared()
        d.three_quark_density = up @ up.conj().T + down @ down.conj().T
        charges = obs.IsospinDoublet.observe(d).candidates[0].charges
        np.testing.assert_allclose(charges.member_occupations, [2.0, 1.0],
                                   atol=1e-8)


if __name__ == "__main__":
    unittest.main()
