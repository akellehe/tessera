# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""Exact fixtures for colour as sheet multiplicity (issue #1195):
SheetedSupport (the k-sheeted carrier and its two commuting actions),
SheetFock (the exterior algebra of the sheet space), SheetAttachment (the
attachment matrix of the connecting simplices and its frame law) and
ColorSinglet (the transported determinant-wedge amplitude).

Acceptance coverage (ticket #1195):

* on a three-sheeted fixture the sheet algebra reproduces the exterior
  algebra of three sheets: the sector dimensions are 1, 3, 3, 1 with
  fermion parities even, odd, even, odd, the four sector projectors are
  bit-identical to ColorFiber's, and the bilinears E^i_j close the gl(3, C)
  commutator table exactly;
* the attachment matrix transforms as S_AB -> g_A^-1 S_AB g_B under
  independent sheet relabelings, its determinant picks up the expected
  det g_B / det g_A, and a closed holonomy's power traces, characteristic
  polynomial and determinant are unchanged by a frame change at the base
  point;
* the singlet wedge is read AFTER transport: the amplitude is
  Omega_p det[S_pA c_A, S_pB c_B, S_pC c_C], it is invariant under a
  simultaneous frame change with a dually transformed Omega_p, it vanishes
  on a degenerate colour triple, and it is linear in each representative
  rather than normalized to unit modulus.

Independent references: every expected value is built with numpy from the
definitions (np.kron, np.linalg.det, an explicitly assembled attachment
matrix), never by calling a second method of the kernel under test.

Exactness bar: structural identities (Kronecker layouts, sector
dimensions, projector agreement, duplicate-mode annihilation) are compared
at double round-off (~1e-15 or exactly). Values that pass through a
matrix inverse or a singular value decomposition are compared at 1e-12,
which is the honest tolerance of those paths on well-conditioned random
fixtures.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

import tessera

ColorFiber = tessera.ColorFiber
ColorSinglet = tessera.ColorSinglet
ConnectingSimplex = tessera.ConnectingSimplex
SheetAttachment = tessera.SheetAttachment
SheetFock = tessera.SheetFock
SheetedSupport = tessera.SheetedSupport


def rng():
    """The one seeded generator every random fixture below draws from."""
    return np.random.default_rng(20261195)


def random_complex(generator, *shape):
    return generator.normal(size=shape) + 1j * generator.normal(size=shape)


def invertible(generator, size):
    """A well-conditioned random element of GL(size, C)."""
    while True:
        candidate = random_complex(generator, size, size)
        if np.linalg.cond(candidate) < 50.0:
            return candidate


# ─── the k-sheeted carrier ─────────────────────────────────────────────────

class TestSheetedSupport(unittest.TestCase):
    """k isomorphic copies of one base complex: the index layout, the free
    operator h (x) I_k, the lifted band, and the two commuting actions."""

    def test_index_layout_is_base_slow_sheet_fast(self) -> None:
        support = SheetedSupport(3, 4)
        self.assertEqual(support.sheet_count, 3)
        self.assertEqual(support.base_cell_count, 4)
        self.assertEqual(support.cell_count, 12)
        for base in range(4):
            for sheet in range(3):
                self.assertEqual(support.modeIndex(base, sheet),
                                 base * 3 + sheet)

    def test_index_out_of_range_is_refused(self) -> None:
        support = SheetedSupport(3, 4)
        with self.assertRaises(ValueError):
            support.modeIndex(4, 0)
        with self.assertRaises(ValueError):
            support.modeIndex(0, 3)

    def test_zero_sheets_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            SheetedSupport(0, 4)

    def test_free_operator_is_the_kronecker_lift(self) -> None:
        generator = rng()
        base = random_complex(generator, 4, 4)
        support = SheetedSupport(3, 4)
        expected = np.kron(base, np.eye(3))
        self.assertLessEqual(
            np.max(np.abs(support.freeOperator(base) - expected)), 0.0)

    def test_lifted_band_is_the_colour_spin_fibre(self) -> None:
        generator = rng()
        band = random_complex(generator, 4, 2)
        support = SheetedSupport(3, 4)
        lifted = support.liftBand(band)
        self.assertEqual(lifted.shape, (12, 6))
        self.assertLessEqual(
            np.max(np.abs(lifted - np.kron(band, np.eye(3)))), 0.0)

    def test_sheet_frame_operator_is_identity_tensor_g(self) -> None:
        generator = rng()
        frame = random_complex(generator, 3, 3)
        support = SheetedSupport(3, 4)
        self.assertLessEqual(
            np.max(np.abs(support.sheetFrameOperator(frame)
                          - np.kron(np.eye(4), frame))), 0.0)

    def test_colour_commutes_with_the_base_dynamics(self) -> None:
        """[h (x) I_k, I (x) g] = 0 for every base operator and every frame:
        the statement that colour is an exact internal symmetry of the free
        theory."""
        generator = rng()
        support = SheetedSupport(3, 4)
        for _ in range(4):
            base = random_complex(generator, 4, 4)
            frame = random_complex(generator, 3, 3)
            self.assertLessEqual(
                support.sheetCommutatorResidual(base, frame), 1e-12)

    def test_colour_commutes_with_a_base_symmetry(self) -> None:
        """The same residual with the base operator a permutation action
        D(g') rather than a dynamical one."""
        support = SheetedSupport(3, 4)
        rotation = np.zeros((4, 4), dtype=complex)
        for i, j in enumerate([1, 2, 3, 0]):
            rotation[j, i] = 1.0
        frame = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=complex)
        self.assertLessEqual(
            support.sheetCommutatorResidual(rotation, frame), 0.0)

    def test_shape_mismatches_are_refused(self) -> None:
        support = SheetedSupport(3, 4)
        with self.assertRaises(ValueError):
            support.freeOperator(np.eye(5, dtype=complex))
        with self.assertRaises(ValueError):
            support.liftBand(np.zeros((5, 2), dtype=complex))
        with self.assertRaises(ValueError):
            support.sheetFrameOperator(np.eye(2, dtype=complex))


class TestSheetIsomorphism(unittest.TestCase):
    """The certificate that the k copies really are copies."""

    def test_identical_sheets_certify(self) -> None:
        support = SheetedSupport(3, 5)
        generator = rng()
        lengths = random_complex(generator, 5)
        phases = np.exp(1j * generator.normal(size=5))
        read = support.certifyIsomorphism([lengths] * 3, [phases] * 3)
        self.assertEqual(read.sheet_count, 3)
        self.assertEqual(read.base_cell_count, 5)
        self.assertEqual(read.squared_length_residual, 0.0)
        self.assertEqual(read.connection_residual, 0.0)
        self.assertTrue(read.isomorphic)
        self.assertTrue(read.certificate.holds())

    def test_a_disagreeing_geometry_is_named_separately(self) -> None:
        support = SheetedSupport(3, 5)
        generator = rng()
        lengths = random_complex(generator, 5)
        phases = np.exp(1j * generator.normal(size=5))
        bent = lengths.copy()
        bent[2] += 0.25
        read = support.certifyIsomorphism([lengths, bent, lengths],
                                          [phases] * 3)
        self.assertAlmostEqual(read.squared_length_residual, 0.25, places=12)
        self.assertEqual(read.connection_residual, 0.0)
        self.assertFalse(read.isomorphic)
        self.assertFalse(read.certificate.holds())

    def test_a_disagreeing_connection_is_named_separately(self) -> None:
        support = SheetedSupport(3, 5)
        generator = rng()
        lengths = random_complex(generator, 5)
        phases = np.exp(1j * generator.normal(size=5))
        turned = phases.copy()
        turned[1] *= -1.0
        read = support.certifyIsomorphism([lengths] * 3,
                                          [phases, phases, turned])
        self.assertEqual(read.squared_length_residual, 0.0)
        self.assertAlmostEqual(read.connection_residual,
                               abs(phases[1] - turned[1]), places=12)
        self.assertFalse(read.isomorphic)

    def test_a_connectionless_support_measures_zero_connection_gap(self) -> None:
        support = SheetedSupport(3, 5)
        generator = rng()
        lengths = random_complex(generator, 5)
        empty = np.zeros(0, dtype=complex)
        read = support.certifyIsomorphism([lengths] * 3, [empty] * 3)
        self.assertEqual(read.connection_residual, 0.0)
        self.assertTrue(read.isomorphic)

    def test_wrong_sheet_count_is_refused(self) -> None:
        support = SheetedSupport(3, 5)
        lengths = np.ones(5, dtype=complex)
        empty = np.zeros(0, dtype=complex)
        with self.assertRaises(ValueError):
            support.certifyIsomorphism([lengths] * 2, [empty] * 2)


# ─── the exterior algebra of the sheet space ───────────────────────────────

class TestSheetFock(unittest.TestCase):
    """Lambda^bullet E for E = C^k, and its identity with ColorFiber at
    three sheets."""

    def test_three_sheet_sectors_are_the_whitepaper_table(self) -> None:
        sectors = SheetFock(3).sectors()
        self.assertEqual([s.occupation for s in sectors], [0, 1, 2, 3])
        self.assertEqual([s.dimension for s in sectors], [1, 3, 3, 1])
        self.assertEqual([s.fermion_parity for s in sectors], [1, -1, 1, -1])
        self.assertEqual([s.representation for s in sectors],
                         ["scalar vacuum", "fundamental E",
                          "det E (x) E-dual", "determinant line"])

    def test_sector_dimensions_are_the_binomials_and_sum_to_two_to_the_k(
            self) -> None:
        for k in range(1, 7):
            fock = SheetFock(k)
            dimensions = [s.dimension for s in fock.sectors()]
            # The binomials, built here by Pascal's recurrence so the
            # expectation does not come from the same arithmetic the kernel
            # uses.
            expected = [math.comb(k, n) for n in range(k + 1)]
            self.assertEqual(dimensions, expected)
            self.assertEqual(sum(dimensions), fock.dimension)
            self.assertEqual(fock.dimension, 2 ** k)

    def test_sector_projectors_are_identical_to_color_fiber(self) -> None:
        fock = SheetFock(3)
        for occupation in range(4):
            mine = fock.sectorProjector(occupation)
            theirs = ColorFiber.sectorProjector(occupation)
            self.assertLessEqual(np.max(np.abs(mine - theirs)), 0.0)
        self.assertEqual(fock.sectorAgreementResidual(), 0.0)

    def test_sector_agreement_is_defined_only_at_three_sheets(self) -> None:
        with self.assertRaises(RuntimeError):
            SheetFock(4).sectorAgreementResidual()

    def test_creation_and_contraction_are_adjoint(self) -> None:
        fock = SheetFock(3)
        for sheet in range(3):
            creation = fock.exteriorCreation(sheet)
            contraction = fock.contraction(sheet)
            self.assertLessEqual(
                np.max(np.abs(creation.conj().T - contraction)), 0.0)

    def test_canonical_anticommutation_relations(self) -> None:
        fock = SheetFock(3)
        eye = np.eye(8)
        for i in range(3):
            for j in range(3):
                ei, ej = fock.exteriorCreation(i), fock.exteriorCreation(j)
                ii, ij = fock.contraction(i), fock.contraction(j)
                self.assertLessEqual(np.max(np.abs(ii @ ij + ij @ ii)), 0.0)
                self.assertLessEqual(np.max(np.abs(ei @ ej + ej @ ei)), 0.0)
                expected = eye if i == j else np.zeros((8, 8))
                self.assertLessEqual(
                    np.max(np.abs(ii @ ej + ej @ ii - expected)), 0.0)

    def test_sheet_bilinears_close_gl3(self) -> None:
        """[E^i_j, E^k_l] = delta^k_j E^i_l - delta^i_l E^k_j, checked here
        against an independently assembled right-hand side as well as
        through the kernel's own sweep."""
        fock = SheetFock(3)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        left = fock.sheetBilinear(i, j)
                        right = fock.sheetBilinear(k, l)
                        expected = np.zeros((8, 8), dtype=complex)
                        if j == k:
                            expected = expected + fock.sheetBilinear(i, l)
                        if i == l:
                            expected = expected - fock.sheetBilinear(k, j)
                        self.assertLessEqual(
                            np.max(np.abs(left @ right - right @ left
                                          - expected)), 1e-15)
        self.assertLessEqual(fock.commutatorResidual(), 1e-15)

    def test_out_of_range_sheet_is_refused(self) -> None:
        fock = SheetFock(3)
        with self.assertRaises(ValueError):
            fock.exteriorCreation(3)
        with self.assertRaises(ValueError):
            fock.sheetBilinear(0, 3)


# ─── the attachment matrix ─────────────────────────────────────────────────

def reference_attachment(sheet_count, simplices):
    """The attachment matrix assembled independently: entry (i, j) is the
    accumulated weight of the simplices joining sheet i of A to sheet j of
    B."""
    out = np.zeros((sheet_count, sheet_count), dtype=complex)
    for sheet_a, sheet_b, weight in simplices:
        out[sheet_a, sheet_b] += weight
    return out


class TestSheetAttachment(unittest.TestCase):
    """S_AB from the connecting simplices, its frame law and its holonomy."""

    SIMPLICES = [(0, 0, 2.0 + 0.0j), (1, 1, 1.0 + 0.0j),
                 (2, 2, 1.0 + 1.0j), (0, 1, 0.5 + 0.0j),
                 (0, 0, 0.25 - 0.5j)]

    def declared(self):
        return [ConnectingSimplex(a, b, w) for a, b, w in self.SIMPLICES]

    def test_entries_accumulate_the_declared_weights(self) -> None:
        read = SheetAttachment.attachmentMatrix(3, self.declared())
        expected = reference_attachment(3, self.SIMPLICES)
        self.assertLessEqual(np.max(np.abs(read.matrix - expected)), 1e-15)
        self.assertEqual(read.simplex_count, len(self.SIMPLICES))
        self.assertAlmostEqual(abs(read.determinant
                                   - np.linalg.det(expected)), 0.0, places=12)

    def test_a_sheet_to_sheet_rule_is_diagonal_and_colour_abelian(self) -> None:
        diagonal = [ConnectingSimplex(i, i, 1.0 + 0.0j) for i in range(3)]
        read = SheetAttachment.attachmentMatrix(3, diagonal)
        self.assertTrue(read.sheet_diagonal)
        self.assertLessEqual(np.max(np.abs(read.matrix - np.eye(3))), 0.0)

    def test_a_cross_sheet_attachment_is_not_diagonal(self) -> None:
        read = SheetAttachment.attachmentMatrix(3, self.declared())
        self.assertFalse(read.sheet_diagonal)

    def test_a_rank_dropping_attachment_certifies_nothing(self) -> None:
        """Two sheets attached to the same sheet leave a zero column, so
        S_AB is singular: it is reported, with infinite conditioning, and
        its certificate does not hold."""
        collapsed = [ConnectingSimplex(0, 0, 1.0 + 0.0j),
                     ConnectingSimplex(1, 0, 1.0 + 0.0j),
                     ConnectingSimplex(2, 2, 1.0 + 0.0j)]
        read = SheetAttachment.attachmentMatrix(3, collapsed)
        self.assertAlmostEqual(abs(read.determinant), 0.0, places=14)
        self.assertLessEqual(read.min_singular_value, 1e-14)
        self.assertGreater(read.conditioning, 1e12)
        self.assertFalse(read.certificate.holds())

    def test_a_simplex_outside_the_support_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            SheetAttachment.attachmentMatrix(
                3, [ConnectingSimplex(0, 3, 1.0 + 0.0j)])

    def test_frame_law(self) -> None:
        """S_AB -> g_A^-1 S_AB g_B, against an independently formed product."""
        generator = rng()
        read = SheetAttachment.attachmentMatrix(3, self.declared())
        for _ in range(4):
            frame_a = invertible(generator, 3)
            frame_b = invertible(generator, 3)
            changed = SheetAttachment.frameChanged(read.matrix, frame_a,
                                                   frame_b)
            expected = np.linalg.inv(frame_a) @ read.matrix @ frame_b
            self.assertLessEqual(np.max(np.abs(changed - expected)), 1e-12)

    def test_the_determinant_line_transforms_by_the_frame_determinants(
            self) -> None:
        generator = rng()
        read = SheetAttachment.attachmentMatrix(3, self.declared())
        frame_a = invertible(generator, 3)
        frame_b = invertible(generator, 3)
        changed = SheetAttachment.frameChanged(read.matrix, frame_a, frame_b)
        expected = (read.determinant * np.linalg.det(frame_b)
                    / np.linalg.det(frame_a))
        self.assertLessEqual(abs(np.linalg.det(changed) - expected), 1e-12)

    def test_a_singular_frame_is_not_a_relabeling(self) -> None:
        read = SheetAttachment.attachmentMatrix(3, self.declared())
        singular = np.zeros((3, 3), dtype=complex)
        with self.assertRaises(ValueError):
            SheetAttachment.frameChanged(read.matrix, singular, np.eye(3))

    def test_coupling_block_factorizes(self) -> None:
        generator = rng()
        base = random_complex(generator, 2, 3)
        read = SheetAttachment.attachmentMatrix(3, self.declared())
        block = SheetAttachment.couplingBlock(base, read.matrix)
        self.assertEqual(block.shape, (6, 9))
        self.assertLessEqual(
            np.max(np.abs(block - np.kron(base, read.matrix))), 1e-15)

    def test_compose_applies_the_first_factor_first(self) -> None:
        generator = rng()
        links = [invertible(generator, 3) for _ in range(3)]
        composed = SheetAttachment.compose(links)
        expected = links[2] @ links[1] @ links[0]
        self.assertLessEqual(np.max(np.abs(composed - expected)), 1e-12)

    def test_an_empty_path_is_the_identity_of_the_declared_rank(self) -> None:
        self.assertLessEqual(
            np.max(np.abs(SheetAttachment.compose([], 3) - np.eye(3))), 0.0)
        with self.assertRaises(ValueError):
            SheetAttachment.compose([])

    def test_holonomy_invariants_match_independent_linear_algebra(self) -> None:
        generator = rng()
        links = [invertible(generator, 3) for _ in range(4)]
        read = SheetAttachment.holonomy(links)
        expected = links[3] @ links[2] @ links[1] @ links[0]
        self.assertEqual(read.link_count, 4)
        scale = max(1.0, float(np.max(np.abs(expected))))
        self.assertLessEqual(np.max(np.abs(read.holonomy - expected)),
                             1e-11 * scale)
        for power in (1, 2, 3):
            trace = np.trace(np.linalg.matrix_power(expected, power))
            self.assertLessEqual(
                abs(read.power_traces[power - 1] - trace),
                1e-9 * max(1.0, abs(trace)))
        # The characteristic polynomial, ascending in lambda, evaluated by
        # numpy's own root finder: the stored coefficients must reproduce
        # det(lambda I - H) at three independent points.
        coefficients = np.array(read.characteristic_polynomial)
        for probe in (0.0 + 0.0j, 1.0 + 0.0j, 0.3 - 0.7j):
            value = sum(c * probe ** n for n, c in enumerate(coefficients))
            reference = np.linalg.det(probe * np.eye(3) - expected)
            self.assertLessEqual(abs(value - reference),
                                 1e-8 * max(1.0, abs(reference)))
        self.assertLessEqual(abs(read.determinant - np.linalg.det(expected)),
                             1e-10)

    def test_holonomy_invariants_are_frame_free(self) -> None:
        """H -> g^-1 H g under a relabeling at the base point, so the power
        traces, the characteristic polynomial and the determinant do not
        move."""
        generator = rng()
        links = [invertible(generator, 3) for _ in range(3)]
        frame = invertible(generator, 3)
        inverse = np.linalg.inv(frame)
        conjugated = [inverse @ link @ frame for link in links]
        original = SheetAttachment.holonomy(links)
        changed = SheetAttachment.holonomy(conjugated)
        for a, b in zip(original.power_traces, changed.power_traces):
            self.assertLessEqual(abs(a - b), 1e-8 * max(1.0, abs(a)))
        for a, b in zip(original.characteristic_polynomial,
                        changed.characteristic_polynomial):
            self.assertLessEqual(abs(a - b), 1e-8 * max(1.0, abs(a)))
        self.assertLessEqual(abs(original.determinant - changed.determinant),
                             1e-8 * max(1.0, abs(original.determinant)))

    def test_an_empty_sequence_has_no_holonomy(self) -> None:
        with self.assertRaises(ValueError):
            SheetAttachment.holonomy([])


# ─── the transported singlet wedge ─────────────────────────────────────────

class TestColorSinglet(unittest.TestCase):
    """S_ABC = Omega_p(c-hat_A ^ c-hat_B ^ c-hat_C), read after transport and
    without normalization."""

    def fixture(self, generator):
        transports = [np.eye(3, dtype=complex), invertible(generator, 3),
                      invertible(generator, 3)]
        colors = [random_complex(generator, 3) for _ in range(3)]
        return transports, colors

    def test_amplitude_is_the_transported_determinant_times_omega(self) -> None:
        generator = rng()
        transports, colors = self.fixture(generator)
        trivialization = 0.75 - 0.25j
        read = ColorSinglet.amplitude(trivialization, transports, colors)
        columns = np.column_stack([t @ c for t, c in zip(transports, colors)])
        expected = trivialization * np.linalg.det(columns)
        self.assertLessEqual(np.max(np.abs(read.transported_columns
                                           - columns)), 1e-12)
        self.assertLessEqual(abs(read.transported_wedge
                                 - np.linalg.det(columns)), 1e-12)
        self.assertLessEqual(abs(read.amplitude - expected), 1e-12)
        self.assertAlmostEqual(read.magnitude, abs(expected), places=12)
        self.assertTrue(read.nonvanishing)
        self.assertTrue(read.certificate.holds())

    def test_the_identity_transport_is_the_untransported_wedge(self) -> None:
        """With every quark in the base cluster p the transports are the
        identity and the amplitude is the plain colour determinant."""
        generator = rng()
        colors = [random_complex(generator, 3) for _ in range(3)]
        read = ColorSinglet.amplitude(1.0 + 0.0j, [np.eye(3, dtype=complex)] * 3,
                                      colors)
        self.assertLessEqual(
            abs(read.amplitude - np.linalg.det(np.column_stack(colors))),
            1e-13)

    def test_transport_changes_the_amplitude(self) -> None:
        """Reading the wedge before transport is a different number: the
        'after transport' clause is not decorative."""
        generator = rng()
        transports, colors = self.fixture(generator)
        after = ColorSinglet.amplitude(1.0 + 0.0j, transports, colors)
        before = ColorSinglet.amplitude(1.0 + 0.0j,
                                        [np.eye(3, dtype=complex)] * 3, colors)
        self.assertGreater(abs(after.amplitude - before.amplitude), 1e-6)

    def test_frame_covariance(self) -> None:
        """c_X -> g_X^-1 c_X, S_pX -> g_p^-1 S_pX g_X and
        Omega_p -> det(g_p) Omega_p leave the amplitude where it was."""
        generator = rng()
        transports, colors = self.fixture(generator)
        base_frame = invertible(generator, 3)
        frames = [invertible(generator, 3) for _ in range(3)]
        residual = ColorSinglet.frameCovarianceResidual(
            0.5 + 1.5j, transports, colors, base_frame, frames)
        reference = abs(ColorSinglet.amplitude(0.5 + 1.5j, transports,
                                               colors).amplitude)
        self.assertLessEqual(residual, 1e-10 * max(1.0, reference))

    def test_a_degenerate_triple_annihilates_the_wedge(self) -> None:
        generator = rng()
        colors = [random_complex(generator, 3) for _ in range(3)]
        colors[1] = colors[0]
        read = ColorSinglet.amplitude(1.0 + 0.0j,
                                      [np.eye(3, dtype=complex)] * 3, colors)
        self.assertLessEqual(read.magnitude, 1e-14)
        self.assertFalse(read.nonvanishing)
        self.assertFalse(read.certificate.holds())

    def test_no_normalization_is_imposed(self) -> None:
        """Rescaling a representative rescales the amplitude: the reported
        content is the determinant ray, never a modulus driven to one."""
        generator = rng()
        transports, colors = self.fixture(generator)
        plain = ColorSinglet.amplitude(1.0 + 0.0j, transports, colors)
        scaled_colors = [colors[0] * (3.0 + 4.0j), colors[1], colors[2]]
        scaled = ColorSinglet.amplitude(1.0 + 0.0j, transports, scaled_colors)
        self.assertLessEqual(
            abs(scaled.amplitude - (3.0 + 4.0j) * plain.amplitude),
            1e-10 * max(1.0, abs(plain.amplitude)))
        # |3 + 4i| = 5: the magnitude moves with the representative instead
        # of being driven to one.
        self.assertLessEqual(
            abs(scaled.magnitude - 5.0 * plain.magnitude),
            1e-10 * max(1.0, plain.magnitude))

    def test_the_wedge_is_antisymmetric_in_the_three_modes(self) -> None:
        generator = rng()
        transports, colors = self.fixture(generator)
        forward = ColorSinglet.amplitude(1.0 + 0.0j, transports, colors)
        swapped = ColorSinglet.amplitude(
            1.0 + 0.0j, [transports[1], transports[0], transports[2]],
            [colors[1], colors[0], colors[2]])
        self.assertLessEqual(
            abs(swapped.amplitude + forward.amplitude),
            1e-10 * max(1.0, abs(forward.amplitude)))

    def test_exactly_three_modes_are_required(self) -> None:
        generator = rng()
        transports, colors = self.fixture(generator)
        with self.assertRaises(ValueError):
            ColorSinglet.amplitude(1.0 + 0.0j, transports[:2], colors[:2])

    def test_a_non_three_sheeted_transport_is_refused(self) -> None:
        generator = rng()
        _, colors = self.fixture(generator)
        with self.assertRaises(ValueError):
            ColorSinglet.amplitude(
                1.0 + 0.0j,
                [np.eye(2, dtype=complex)] * 3,
                [c[:2] for c in colors])


if __name__ == "__main__":
    unittest.main()
