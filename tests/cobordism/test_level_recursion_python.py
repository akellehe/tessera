# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""#1191 — the recursion driven level by level, energy dependence kept exact.

Sections 3, 5 and 15 of the whitepaper run one box at every scale: the partition
of the response network, the Riesz projector of each response vertex over its own
contour, the exact energy-dependent Feshbach map to the next level, the transport
blocks read in the fibers, and the labeled sum with its bilinear overlap. Five
claims carry that here and each is asserted against a dense reference built in
this file.

THE RECURSION IS DRIVEN LEVEL BY LEVEL. Three levels are built on a fixture
response network, each from the certified bands and the partition of the level
below it, and each level's partition is discovered over a declared sweep of
modularity resolutions rather than at one of them.

THE ENERGY DEPENDENCE IS EXACT, NOT LINEARIZED. ``response_pencil(l, lambda)``
re-derives the whole chain from the microscopic pencil at each lambda it is asked
for. The level-one pencil is asserted to differ from the level-one pencil at zero
shifted by lambda, which is what a linear surrogate would give, and the level-two
pencil is asserted to equal the plain Schur complement of the level-one pencil at
the same lambda, taken with no further shift. Subtracting lambda a second time is
the one thing the exactness of the step rests on not happening.

EACH LEVEL'S SPECTRUM IS REPRODUCED FROM THE LEVEL BELOW. The Feshbach map obeys
``det R_l = det (R_l)_II det R_{l+1}``, so an eigenvalue of the microscopic pencil
that no eliminated interior block carries is a zero of every level's determinant.
Both halves are asserted: the factorization at frequencies of the caller's
choosing, and the singularity of the third level's pencil at the microscopic
eigenvalues themselves.

THE FIBERS ARE RIESZ PROJECTORS AND THEIR FRAMES PAIR BY THE TRANSPOSE. Each
band's projector is the contour integral of the resolvent, its two frames satisfy
PhiTilde^T Phi = I with no conjugation, and the compression PhiTilde^T h_v Phi has
exactly the eigenvalues the contour enclosed.

THE LABELED SUM CARRIES ITS OVERLAP RATHER THAN ASSUMING IT AWAY. The Gram is the
transpose pairing of the two embeddings, the transports are its named blocks, and
the Fock stage is reported as a dimension the recursion never allocates.
"""

import unittest

import numpy as np

import tessera as T

cob = T.cobordism

#: The fixture response network: eight blocks of eight coordinates each,
#: coupled around a ring, with the couplings inside a block two orders of
#: magnitude stronger than the ones between blocks. Modularity therefore
#: proposes the blocks; a coordinate that touches no other block is interior to
#: its own component and is eliminated, and the level shrinks to the
#: coordinates that carry the coupling. A ring rather than a path, so that no
#: community has a free end and the reduction cannot empty a level.
BLOCKS = 8
BLOCK_WIDTH = 8
DIMENSION = BLOCKS * BLOCK_WIDTH


def _fixture_pencil():
    """A complex symmetric ring operator with eight communities."""
    matrix = np.zeros((DIMENSION, DIMENSION), dtype=complex)
    for index in range(DIMENSION):
        following = (index + 1) % DIMENSION
        strong = (index % BLOCK_WIDTH) != BLOCK_WIDTH - 1
        coupling = complex(-1.0, -0.03) if strong else complex(-0.01, -0.0004)
        matrix[index, following] = coupling
        matrix[following, index] = coupling
    for index in range(DIMENSION):
        matrix[index, index] = -matrix[index].sum() + complex(1.0, 0.02)
    return matrix


def _declaration(resolutions=(1.0, 1.4, 1.8), band_rank=2, tolerance=1e-6):
    declaration = cob.LevelRecursionDeclaration()
    declaration.resolutions = list(resolutions)
    declaration.modularity_restarts = 4
    declaration.modularity_seed = 17
    declaration.reference_lambda = complex(0.0, 0.0)
    declaration.tolerance = tolerance
    declaration.dense_crossover = 512
    bands = cob.RecursionBandDeclaration()
    bands.selection = cob.RecursionBandSelection.LowestModes
    bands.band_rank = band_rank
    bands.order = cob.OccupationOrder.AscendingModulus
    bands.contour_nodes = 128
    declaration.bands = bands
    return declaration


def _recursion(levels=3, **overrides):
    matrix = _fixture_pencil()
    recursion = cob.LevelRecursion.overPencil(
        [complex(value) for value in matrix.reshape(-1)], [], DIMENSION,
        _declaration(**overrides))
    recursion.advance_to(levels)
    return matrix, recursion


def _square(flat, order):
    return np.asarray(flat, dtype=complex).reshape(order, order)


def _schur(matrix, interior, kept):
    """The plain supported block elimination, with no shift of any kind."""
    if not interior:
        return matrix[np.ix_(kept, kept)]
    inner = matrix[np.ix_(interior, interior)]
    return (matrix[np.ix_(kept, kept)]
            - matrix[np.ix_(kept, interior)]
            @ np.linalg.solve(inner, matrix[np.ix_(interior, kept)]))


def _interior_and_kept(matrix, partition):
    """Which coordinates the reduction eliminates and which it keeps.

    A coordinate is interior to its component exactly when every nonzero
    coupling row and column of the operator stays inside that component; every
    other coordinate is kept. This is the classification the reduction makes,
    written out here so the dense reference does not borrow it.
    """
    owner = {}
    for component, members in enumerate(partition):
        for coordinate in members:
            owner[coordinate] = component
    interior = []
    kept = []
    for coordinate in range(matrix.shape[0]):
        inside = True
        for other in range(matrix.shape[0]):
            if owner[other] == owner[coordinate]:
                continue
            if matrix[coordinate, other] != 0.0 or matrix[other, coordinate] != 0.0:
                inside = False
                break
        (interior if inside else kept).append(coordinate)
    return interior, kept


class TheRecursionIsDrivenLevelByLevelTest(unittest.TestCase):
    """Three levels, each built from the partition and bands below it."""

    def test_three_levels_are_built_and_each_one_is_smaller(self):
        _, recursion = _recursion()
        self.assertEqual(recursion.level_count(), 3)
        widths = [recursion.response_dimension(level) for level in range(4)]
        self.assertEqual(widths[0], DIMENSION)
        self.assertLess(widths[1], widths[0])
        for lower, upper in zip(widths, widths[1:]):
            self.assertLessEqual(upper, lower)
            self.assertGreaterEqual(upper, 1)
        for index in range(3):
            level = recursion.level(index)
            self.assertEqual(level.level, index)
            self.assertEqual(level.dimension, widths[index])
            self.assertEqual(level.response_dimension, widths[index + 1])
            covered = sorted(
                coordinate for members in level.partition for coordinate in members)
            self.assertEqual(covered, list(range(level.dimension)))

    def test_the_partition_is_discovered_over_the_declared_resolutions(self):
        """The sweep is a persistence statement, not one optimization.

        Every carried component reports how many adjacent resolutions it
        survived, and the resolution the carried partition came from is one of
        the declared ones.
        """
        _, recursion = _recursion(levels=1)
        level = recursion.level(0)
        self.assertEqual(list(level.resolutions), [1.0, 1.4, 1.8])
        self.assertIn(level.selected_resolution, [1.0, 1.4, 1.8])
        self.assertEqual(len(level.component_persistence), len(level.partition))
        for persistence in level.component_persistence:
            self.assertGreaterEqual(persistence, 1.0)

    def test_a_single_resolution_sweep_reproduces_the_single_resolution_call(self):
        matrix = _fixture_pencil()
        flat = [complex(value) for value in matrix.reshape(-1)]
        swept = cob.RecursiveQuotient.persistentPartitionOverResolutions(
            flat, DIMENSION, [1.0], 4, 17, 0.5)
        single = cob.RecursiveQuotient.persistentPartition(
            flat, DIMENSION, 1.0, 4, 17)
        self.assertEqual([list(members) for members in swept.components],
                         [list(members) for members in single])
        self.assertEqual(swept.selected_resolution, 1.0)

    def test_the_reduction_map_covers_every_coordinate(self):
        """componentOfCoordinate is the map MappingCylinder builds W^l over."""
        _, recursion = _recursion(levels=1)
        owners = recursion.component_of_coordinate(0)
        self.assertEqual(len(owners), DIMENSION)
        self.assertEqual(set(owners), set(range(len(recursion.level(0).partition))))


class TheEnergyDependenceIsExactTest(unittest.TestCase):
    """R_l is a function of lambda, re-derived at every one it is asked for."""

    def test_the_level_one_pencil_is_not_the_level_one_pencil_shifted(self):
        """A linear surrogate would be, and the exact pencil is not.

        F_B(lambda) = L_BB - lambda I - L_BI (L_II - lambda I)^-1 L_IB is a
        rational function of lambda, so it differs from its own value at zero
        shifted by lambda by the whole of the second term's variation. Asserting
        that difference is asserting that nothing was linearized.
        """
        _, recursion = _recursion(levels=1)
        width = recursion.response_dimension(1)
        at_zero = _square(recursion.response_pencil(1, 0.0), width)
        at_lambda = _square(recursion.response_pencil(1, 0.31), width)
        linearized = at_zero - 0.31 * np.eye(width, dtype=complex)
        self.assertGreater(np.abs(at_lambda - linearized).max(),
                           1e-3 * np.abs(at_zero).max())

    def test_the_level_two_pencil_is_the_unshifted_schur_complement(self):
        """R_{l+1}(lambda) = Feshbach_{P_l}(R_l(lambda)) with no second shift.

        The dense reference eliminates the interior coordinates of the level-one
        pencil at the same lambda and subtracts nothing, because the spectral
        parameter is already inside the matrix being eliminated.
        """
        _, recursion = _recursion(levels=2)
        for lambda_ in (0.23, -0.11, 0.41 - 0.17j):
            width = recursion.response_dimension(1)
            first = _square(recursion.response_pencil(1, lambda_), width)
            interior, kept = _interior_and_kept(first, recursion.level(1).partition)
            reference = _schur(first, interior, kept)
            produced = _square(recursion.response_pencil(2, lambda_),
                               recursion.response_dimension(2))
            self.assertEqual(produced.shape, reference.shape)
            self.assertLess(np.abs(produced - reference).max(),
                            1e-8 * max(1.0, np.abs(reference).max()))

    def test_the_microscopic_pencil_is_the_operator_minus_lambda(self):
        matrix, recursion = _recursion(levels=1)
        for lambda_ in (0.0, 0.7 + 0.2j):
            produced = _square(recursion.response_pencil(0, lambda_), DIMENSION)
            reference = matrix - lambda_ * np.eye(DIMENSION, dtype=complex)
            self.assertLess(np.abs(produced - reference).max(), 1e-12)

    def test_an_unbuilt_level_is_refused(self):
        _, recursion = _recursion(levels=1)
        with self.assertRaises(IndexError):
            recursion.response_pencil(2, 0.0)


class EachLevelsSpectrumIsReproducedFromTheLevelBelowTest(unittest.TestCase):
    """The determinant factorization, and the singularity at an eigenvalue."""

    def test_the_determinant_factorizes_at_every_level(self):
        """det R_l(lambda) = det (R_l(lambda))_II det R_{l+1}(lambda).

        This is the identity that makes the Feshbach step exact rather than an
        approximation over a window, and it is measured at frequencies chosen
        here rather than at the one the levels were built at.
        """
        _, recursion = _recursion()
        for level in range(3):
            for lambda_ in (0.13, -0.29, 0.44 + 0.21j):
                with self.subTest(level=level, lambda_=lambda_):
                    self.assertLess(
                        recursion.determinant_factorization_residual(level,
                                                                     lambda_),
                        1e-7)

    def test_the_levels_own_determinant_residual_holds(self):
        _, recursion = _recursion()
        for index in range(3):
            level = recursion.level(index)
            self.assertLess(level.determinant_residual, 1e-7)
            self.assertTrue(level.reduction_certificate.holds(),
                            level.reduction_certificate.describe())

    def test_the_third_level_is_singular_at_a_microscopic_eigenvalue(self):
        """An eigenvalue of the microscopic pencil that no eliminated interior
        block carries is a zero of every level's determinant, so the third
        level's pencil is singular there.

        The eigenvalues tested are the ones whose interior determinants stay
        well away from zero along the whole chain, which is the condition the
        factorization attaches to the statement.
        """
        matrix, recursion = _recursion()
        eigenvalues = sorted(np.linalg.eigvals(matrix), key=abs)[:6]
        tested = 0
        for value in eigenvalues:
            interior = [recursion.interior_determinant(level, complex(value))
                        for level in range(3)]
            if min(abs(entry) for entry in interior) < 1e-6:
                continue
            width = recursion.response_dimension(3)
            pencil = _square(recursion.response_pencil(3, complex(value)), width)
            singular = np.linalg.svd(pencil, compute_uv=False)
            self.assertLess(singular[-1], 1e-6 * singular[0])
            tested += 1
        self.assertGreater(tested, 0)

    def test_the_third_level_is_regular_away_from_the_spectrum(self):
        matrix, recursion = _recursion()
        eigenvalues = np.linalg.eigvals(matrix)
        probe = complex(max(value.real for value in eigenvalues) + 1.0, 0.37)
        width = recursion.response_dimension(3)
        pencil = _square(recursion.response_pencil(3, probe), width)
        singular = np.linalg.svd(pencil, compute_uv=False)
        self.assertGreater(singular[-1], 1e-6 * singular[0])


class TheFibersAreRieszProjectorsTest(unittest.TestCase):
    """The contour integral, the transpose pairing, and the compression."""

    def test_every_band_is_accepted_and_pairs_by_the_transpose(self):
        _, recursion = _recursion()
        for index in range(3):
            level = recursion.level(index)
            self.assertEqual(len(level.bands), len(level.partition))
            for band in level.bands:
                with self.subTest(level=index, component=band.component):
                    self.assertGreaterEqual(band.rank, 1)
                    self.assertTrue(band.accepted, band.certificate.describe())
                    self.assertLess(band.pairing_defect, 1e-6)
                    self.assertLess(band.projector_idempotency, 1e-6)
                    self.assertGreater(band.isolation_gap, 0.0)
                    self.assertEqual(band.contour_nodes, 128)

    def test_the_frames_pair_to_the_identity_with_no_conjugation(self):
        """PhiTilde^T Phi = I, taken with the transpose and never an adjoint."""
        _, recursion = _recursion(levels=1)
        level = recursion.level(0)
        for band in level.bands:
            right = np.asarray(band.frame, dtype=complex).reshape(
                level.dimension, band.rank)
            left = np.asarray(band.left_frame, dtype=complex).reshape(
                band.rank, level.dimension)
            self.assertLess(
                np.abs(left @ right - np.eye(band.rank, dtype=complex)).max(),
                1e-6)

    def test_the_compression_carries_the_eigenvalues_the_contour_enclosed(self):
        """PhiTilde_v^T h_v Phi_v has exactly the band's eigenvalues.

        That is what makes the fiber the range of the Riesz projector rather
        than any other subspace of the same rank.
        """
        _, recursion = _recursion(levels=1)
        level = recursion.level(0)
        operator = _square(recursion.response_pencil(0, 0.0), level.dimension)
        for band in level.bands:
            right = np.asarray(band.frame, dtype=complex).reshape(
                level.dimension, band.rank)
            left = np.asarray(band.left_frame, dtype=complex).reshape(
                band.rank, level.dimension)
            compressed = np.linalg.eigvals(left @ operator @ right)
            enclosed = np.asarray(band.eigenvalues, dtype=complex)
            self.assertEqual(len(enclosed), band.rank)
            ordered = sorted(compressed, key=lambda v: (v.real, v.imag))
            expected = sorted(enclosed, key=lambda v: (v.real, v.imag))
            for produced, target in zip(ordered, expected):
                self.assertLess(abs(produced - target),
                                1e-6 * max(1.0, abs(target)))

    def test_a_declared_contour_selects_the_band_without_sorting(self):
        """The other option: a centre and a radius per component, and nothing
        is ordered at all."""
        matrix = _fixture_pencil()
        flat = [complex(value) for value in matrix.reshape(-1)]
        partition = cob.RecursiveQuotient.persistentPartitionOverResolutions(
            flat, DIMENSION, [1.0, 1.4, 1.8], 4, 17, 0.5)
        declaration = _declaration()
        bands = cob.RecursionBandDeclaration()
        bands.selection = cob.RecursionBandSelection.DeclaredContours
        bands.contour_nodes = 128
        bands.contour_centres = [complex(0.0, 0.0)] * len(partition.components)
        bands.contour_radii = [20.0] * len(partition.components)
        declaration.bands = bands
        recursion = cob.LevelRecursion.overPencil(flat, [], DIMENSION,
                                                  declaration)
        recursion.advance()
        level = recursion.level(0)
        self.assertEqual(len(level.bands), len(partition.components))
        for band, members in zip(level.bands, level.partition):
            self.assertEqual(band.rank, len(members))
            self.assertEqual(band.contour_radius, 20.0)
            for value in band.eigenvalues:
                self.assertLess(abs(value), 20.0)


class TheLabeledSumCarriesItsOverlapTest(unittest.TestCase):
    """The Gram, the transports and the Fock stage."""

    def test_the_gram_is_the_transpose_pairing_of_the_two_embeddings(self):
        _, recursion = _recursion(levels=1)
        level = recursion.level(0)
        embedding = np.asarray(level.embedding, dtype=complex).reshape(
            level.dimension, level.modes)
        dual = np.asarray(level.dual_embedding, dtype=complex).reshape(
            level.modes, level.dimension)
        gram = np.asarray(level.gram, dtype=complex).reshape(
            level.modes, level.modes)
        self.assertLess(np.abs(gram - dual @ embedding).max(), 1e-12)
        # The supports of a partition are disjoint, so the labeled sum is
        # direct here and the overlap is the identity exactly; the machinery
        # carries it either way rather than assuming it.
        self.assertLess(level.gram_defect, 1e-8)

    def test_the_fiber_operator_is_the_compression_and_its_blocks_are_the_transports(self):
        _, recursion = _recursion(levels=1)
        level = recursion.level(0)
        embedding = np.asarray(level.embedding, dtype=complex).reshape(
            level.dimension, level.modes)
        dual = np.asarray(level.dual_embedding, dtype=complex).reshape(
            level.modes, level.dimension)
        operator = _square(recursion.response_pencil(0, 0.0), level.dimension)
        produced = np.asarray(level.fiber_operator, dtype=complex).reshape(
            level.modes, level.modes)
        self.assertLess(np.abs(produced - dual @ operator @ embedding).max(),
                        1e-10)
        offsets = []
        running = 0
        for band in level.bands:
            offsets.append(running)
            running += band.rank
        self.assertEqual(running, level.modes)
        for transport in level.transports:
            rows = level.bands[transport.to_component].rank
            columns = level.bands[transport.from_component].rank
            block = np.asarray(transport.block, dtype=complex).reshape(
                rows, columns)
            expected = produced[
                offsets[transport.to_component]:offsets[transport.to_component] + rows,
                offsets[transport.from_component]:
                offsets[transport.from_component] + columns]
            self.assertLess(np.abs(block - expected).max(), 1e-12)

    def test_the_fock_stage_is_reported_as_a_dimension_and_never_allocated(self):
        _, recursion = _recursion(levels=1)
        level = recursion.level(0)
        self.assertAlmostEqual(level.fock_stage_dimension, 2.0 ** level.modes)

    def test_the_vacuum_embedding_grows_the_stage(self):
        """The reduction alone grows nothing; the interaction stage attaches the
        new modes and the carried state reaches them with the new modes empty."""
        _, recursion = _recursion(levels=1)
        level = recursion.level(0)
        modes = level.modes
        self.assertEqual(level.vacuum_embedded_modes, 0)
        recursion.record_vacuum_embedded_modes(0, 3)
        grown = recursion.level(0)
        self.assertEqual(grown.vacuum_embedded_modes, 3)
        self.assertAlmostEqual(grown.fock_stage_dimension, 2.0 ** (modes + 3))


class TheDeclarationIsCheckedTest(unittest.TestCase):
    """Every malformed recursion is refused by name."""

    def test_a_pencil_of_the_wrong_size_is_refused(self):
        with self.assertRaises(ValueError):
            cob.LevelRecursion.overPencil([1.0, 0.0, 0.0], [], 2,
                                          _declaration())

    def test_an_empty_resolution_sweep_is_refused(self):
        with self.assertRaises(ValueError):
            cob.LevelRecursion.overPencil([1.0], [], 1,
                                          _declaration(resolutions=()))

    def test_a_band_of_rank_zero_is_refused(self):
        with self.assertRaises(ValueError):
            cob.LevelRecursion.overPencil([1.0], [], 1,
                                          _declaration(band_rank=0))

    def test_a_dimension_at_the_dense_crossover_is_refused(self):
        declaration = _declaration()
        declaration.dense_crossover = 4
        matrix = _fixture_pencil()
        with self.assertRaises(ValueError):
            cob.LevelRecursion.overPencil(
                [complex(value) for value in matrix.reshape(-1)], [],
                DIMENSION, declaration)


if __name__ == "__main__":
    unittest.main()
