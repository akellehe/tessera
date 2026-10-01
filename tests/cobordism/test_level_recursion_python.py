# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""#1191 — the recursion driven level by level, energy dependence kept exact.

Sections 3, 5 and 15 of the whitepaper run one box at every scale: the partition
of the response network, the Riesz projector of each response vertex onto the
band its declared selection names, the exact energy-dependent Feshbach map to the
next level, the transport blocks read in the fibers, and the labeled sum with its
bilinear overlap. Five claims carry that here and each is asserted against a
dense reference built in this file.

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

THE FIBERS ARE EXACT RIESZ PROJECTORS AND THEIR FRAMES PAIR BY THE TRANSPOSE.
Each band's projector is the spectral projector onto the invariant subspace of
the eigenvalues its selection encloses, formed from the block's complex Schur
form with the selected eigenvalues reordered to the leading block and the
Sylvester equation solved for the invariant subspace. It equals V_B (V^-1)_B
built from an independent eigendecomposition, it is exact on a
non-diagonalizable block where no eigenvector matrix can be inverted, its two
frames satisfy PhiTilde^T Phi = I with no conjugation, and the compression
PhiTilde^T h_v Phi has exactly the eigenvalues the selection enclosed. A band
is read whatever its isolation: a selection that separates two eigenvalues
equal at the declared tolerance, a declared contour through an eigenvalue and
a declared contour around nothing each return their band, with ``accepted``
false and the measured gap beside it. One selection has no projector, the one
that takes an eigenvalue in and leaves an eigenvalue exactly equal to it out,
and the read says so by name.

THE LABELED SUM CARRIES ITS OVERLAP RATHER THAN ASSUMING IT AWAY. The Gram is the
transpose pairing of the two embeddings, the transports are its named blocks, and
the Fock stage is reported as a dimension the recursion never allocates.
"""

import math
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
        self.assertEqual(swept.selectedResolution, 1.0)

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


def _band_read(block, rank=1, order=cob.OccupationOrder.AscendingRealPart,
               tolerance=1e-15, selection=None, centre=None, radius=None):
    """`LevelRecursion.read_band` on one block as a level of one component."""
    block = np.asarray(block, dtype=complex)
    bands = cob.RecursionBandDeclaration()
    if selection is None:
        bands.selection = cob.RecursionBandSelection.LowestModes
        bands.band_rank = rank
        bands.order = order
    else:
        bands.selection = selection
        bands.contour_centres = [complex(centre)]
        bands.contour_radii = [float(radius)]
    return cob.LevelRecursion.read_band(
        [complex(value) for value in block.reshape(-1)], block.shape[0],
        bands, 0, tolerance)


def _frames(read, order):
    right = np.asarray(read.frame, dtype=complex).reshape(order, read.rank)
    left = np.asarray(read.left_frame, dtype=complex).reshape(read.rank, order)
    return right, left


class TheFibersAreRieszProjectorsTest(unittest.TestCase):
    """The exact projector, the transpose pairing, and the compression."""

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
                    self.assertLess(band.invariant_subspace_residual, 1e-6)
                    self.assertGreater(band.isolation_gap, 0.0)

    def test_the_projector_is_the_eigenprojector_of_the_selected_eigenvalues(self):
        """P_v = V_B (V^-1)_B over the selected eigenvalues.

        The reference is built from numpy's eigendecomposition of the block,
        matching each selected eigenvalue to a column of V; the produced
        projector is Phi_v PhiTilde_v^T restricted to the block's coordinates.
        Both are the same exact object, the spectral projector of the band,
        and differ by rounding alone: each carries about the condition number
        of V times the unit roundoff, below 1e-13 on these blocks.
        """
        _, recursion = _recursion(levels=1)
        level = recursion.level(0)
        operator = _square(recursion.response_pencil(0, 0.0), level.dimension)
        for band, members in zip(level.bands, level.partition):
            block = operator[np.ix_(members, members)]
            values, vectors = np.linalg.eig(block)
            inverse = np.linalg.inv(vectors)
            columns = []
            for selected in band.eigenvalues:
                distances = np.abs(values - selected)
                column = int(np.argmin(distances))
                self.assertLess(distances[column], 1e-10)
                columns.append(column)
            self.assertEqual(len(set(columns)), band.rank)
            reference = vectors[:, columns] @ inverse[columns, :]
            right, left = _frames(band, level.dimension)
            produced = (right @ left)[np.ix_(members, members)]
            self.assertLess(np.abs(produced - reference).max(), 1e-13)

    def test_a_non_diagonalizable_block_has_an_exact_projector(self):
        """The Schur route needs no eigenvector matrix.

        The block [[1, 1, 1/2], [0, 1, 1], [0, 0, 3]] has the eigenvalue 1 with
        algebraic multiplicity two and geometric multiplicity one, so no
        eigenvector matrix is invertible. The spectral projector onto the
        generalized eigenspace of 1 is I minus the projector onto the
        eigenvector (1/2, 1/2, 1) of 3 along its left eigenvector (0, 0, 1):
        [[1, 0, -1/2], [0, 1, -1/2], [0, 0, 0]]. Every step of the read on
        this upper-triangular block is exact in binary arithmetic (the Schur
        form is the block itself and the Sylvester solve divides by 2), so
        the projector is reproduced to the declared tolerance.
        """
        block = np.array([[1.0, 1.0, 0.5], [0.0, 1.0, 1.0], [0.0, 0.0, 3.0]])
        read = _band_read(block, rank=2)
        self.assertEqual(read.rank, 2)
        np.testing.assert_allclose(sorted(read.eigenvalues, key=abs),
                                   [1.0, 1.0], atol=1e-15)
        right, left = _frames(read, 3)
        expected = np.array([[1.0, 0.0, -0.5], [0.0, 1.0, -0.5],
                             [0.0, 0.0, 0.0]])
        self.assertLess(np.abs(right @ left - expected).max(), 1e-15)
        self.assertLess(read.projector_idempotency, 1e-15)
        self.assertLess(read.pairing_defect, 1e-15)
        self.assertLess(read.invariant_subspace_residual, 1e-15)
        self.assertEqual(read.isolation_gap, 2.0)
        self.assertTrue(read.accepted, read.certificate.describe())

    def test_a_band_rank_that_splits_an_exactly_multiple_eigenvalue_has_no_value(
            self):
        """diag(1, 1, 3) with band rank one: the two eigenvalues 1 are equal
        exactly, so the band would take one of them and leave the other out.
        The Sylvester operator of the projector, Y -> T_11 Y - Y T_22, has the
        difference 1 - 1 = 0 among its eigenvalues and is singular: a
        rank-one part of that eigenspace is no invariant subspace of its own,
        the projector has no value, and the read says so by name, with the
        eigenvalue."""
        with self.assertRaisesRegex(
                ValueError, r"takes the eigenvalue \(1\.000000, 0\.000000\) "
                "of its block into the band and leaves an eigenvalue exactly "
                "equal to it out"):
            _band_read(np.diag([1.0, 1.0, 3.0]), rank=1)

    def test_a_band_rank_through_a_near_degenerate_pair_is_read_and_not_accepted(
            self):
        """diag(1, 1 + 2^-50, 3) with band rank one at the tolerance 1e-15.

        The block's Frobenius norm is sqrt(1 + (1 + 2^-50)^2 + 9), about
        3.317, so two eigenvalues at most 3.317e-15 apart are equal at the
        tolerance; 1 and 1 + 2^-50 are 2^-50 = 8.9e-16 apart, so the band of
        rank one separates two eigenvalues equal at the tolerance. The read
        is made all the same. The order of the two is the order of their
        exact keys, so the band is the eigenvalue 1; the block is diagonal,
        so its Schur form is the block itself, the coupling T_12 of the
        Sylvester equation is zero, and the projector is diag(1, 0, 0) with
        every residual zero exactly. What the read reports is the isolation
        gap, 2^-50 exactly, and a fiber that is not accepted, with a
        certificate that does not hold; the recorded circle is centred on 1
        with the radius halfway to the excluded neighbour, 2^-51."""
        gap = 2.0 ** -50
        read = _band_read(np.diag([1.0, 1.0 + gap, 3.0]), rank=1)
        self.assertEqual(read.rank, 1)
        self.assertEqual(list(read.eigenvalues), [1.0])
        self.assertEqual(read.isolation_gap, gap)
        self.assertLess(read.isolation_gap, 1e-15 * math.sqrt(11.0))
        self.assertEqual(read.contour_centre, 1.0)
        self.assertEqual(read.contour_radius, gap / 2.0)
        self.assertEqual(read.contour_gap, math.inf)
        right, left = _frames(read, 3)
        self.assertTrue(np.array_equal(right @ left, np.diag([1.0, 0.0, 0.0])))
        self.assertEqual(read.projector_idempotency, 0.0)
        self.assertEqual(read.pairing_defect, 0.0)
        self.assertEqual(read.invariant_subspace_residual, 0.0)
        self.assertFalse(read.accepted)
        self.assertFalse(read.certificate.holds())
        self.assertEqual(read.certificate.residual, math.inf)
        # the same block with the pair 2^-40 = 9.1e-13 apart, far above the
        # 3.3e-15 resolution, is the same projector, accepted
        apart = _band_read(np.diag([1.0, 1.0 + 2.0 ** -40, 3.0]), rank=1)
        self.assertEqual(apart.isolation_gap, 2.0 ** -40)
        self.assertTrue(apart.accepted, apart.certificate.describe())

    def test_a_declared_contour_through_an_eigenvalue_is_read_and_not_accepted(
            self):
        """The circle about 1 of radius 2 passes through the eigenvalue 3 of
        diag(1, 3). Membership is the strict comparison of the distance from
        the centre with the radius: 1 is at distance 0 and is in the band, 3
        is at distance 2.0, which is not below 2.0, and is out. The band is
        therefore the eigenvalue 1, with the projector diag(1, 0) exactly
        (the block is diagonal) and the isolation gap 3 - 1 = 2. The
        distance from the circle to the nearest eigenvalue is
        |2.0 - 2.0| = 0, which is the report that an eigenvalue is on the
        circle, and the fiber is not accepted."""
        read = _band_read(np.diag([1.0, 3.0]),
                          selection=cob.RecursionBandSelection.DeclaredContours,
                          centre=1.0, radius=2.0)
        self.assertEqual(read.rank, 1)
        self.assertEqual(list(read.eigenvalues), [1.0])
        self.assertFalse(read.encloses_everything)
        self.assertEqual(read.isolation_gap, 2.0)
        self.assertEqual(read.contour_gap, 0.0)
        right, left = _frames(read, 2)
        self.assertTrue(np.array_equal(right @ left, np.diag([1.0, 0.0])))
        self.assertEqual(read.projector_idempotency, 0.0)
        self.assertFalse(read.accepted)
        self.assertFalse(read.certificate.holds())
        # the circle of radius 1.5 passes through no eigenvalue: its nearest
        # is 3, at |2 - 1.5| = 0.5 from the circle, and the same band is
        # accepted
        clear = _band_read(np.diag([1.0, 3.0]),
                           selection=cob.RecursionBandSelection.DeclaredContours,
                           centre=1.0, radius=1.5)
        self.assertEqual(list(clear.eigenvalues), [1.0])
        self.assertEqual(clear.contour_gap, 0.5)
        self.assertTrue(clear.accepted, clear.certificate.describe())

    def test_a_declared_contour_enclosing_nothing_is_the_band_of_rank_zero(self):
        """The circle about 10 of radius 1 encloses neither eigenvalue of
        diag(1, 3) (distances 9 and 7). The band has rank zero: no
        eigenvalue, the zero projector, frames with no column, and residuals
        that are zero exactly, because the zero projector is idempotent, the
        empty frames pair to the 0 x 0 identity and the zero subspace is
        invariant. Nothing is selected, so there is no selected eigenvalue to
        measure an isolation from and the gap is infinite; the circle's
        nearest eigenvalue is 3, at |7 - 1| = 6 from it. A band of rank zero
        is not a fiber, so it is not accepted."""
        read = _band_read(np.diag([1.0, 3.0]),
                          selection=cob.RecursionBandSelection.DeclaredContours,
                          centre=10.0, radius=1.0)
        self.assertEqual(read.rank, 0)
        self.assertEqual(list(read.eigenvalues), [])
        self.assertEqual(list(read.frame), [])
        self.assertEqual(list(read.left_frame), [])
        self.assertFalse(read.encloses_everything)
        self.assertEqual(read.isolation_gap, math.inf)
        self.assertEqual(read.contour_gap, 6.0)
        self.assertEqual(read.contour_centre, 10.0)
        self.assertEqual(read.contour_radius, 1.0)
        self.assertEqual(read.projector_idempotency, 0.0)
        self.assertEqual(read.pairing_defect, 0.0)
        self.assertEqual(read.invariant_subspace_residual, 0.0)
        self.assertFalse(read.accepted)
        self.assertFalse(read.certificate.holds())

    def test_a_band_of_the_whole_block_encloses_everything(self):
        """A band rank at or above the block's order selects every eigenvalue:
        the selection has no excluded eigenvalue to measure to, so its radius
        and the isolation gap are infinite, and the read says so. The
        invariant subspace is the whole coordinate space, so the projector is
        the identity and both frames are the canonical basis exactly, on a
        block that is neither diagonal nor normal as on any other, and the
        three residual certificates are zero exactly. The recorded centre is
        the mean of the eigenvalues, the trace 6 over the order 3."""
        block = np.array([[1.0, 2.0 + 1.0j, 0.5], [0.0, 3.0, -1.0j],
                          [0.25, 0.0, 2.0]])
        read = _band_read(block, rank=5)
        self.assertEqual(read.rank, 3)
        self.assertTrue(read.encloses_everything)
        self.assertEqual(read.contour_radius, math.inf)
        self.assertEqual(read.isolation_gap, math.inf)
        self.assertLess(abs(read.contour_centre - 2.0), 1e-14)
        np.testing.assert_allclose(
            sorted(read.eigenvalues, key=lambda v: (v.real, v.imag)),
            sorted(np.linalg.eigvals(block), key=lambda v: (v.real, v.imag)),
            atol=1e-13)
        right, left = _frames(read, 3)
        self.assertTrue(np.array_equal(right, np.eye(3)))
        self.assertTrue(np.array_equal(left, np.eye(3)))
        self.assertEqual(read.projector_idempotency, 0.0)
        self.assertEqual(read.pairing_defect, 0.0)
        self.assertEqual(read.invariant_subspace_residual, 0.0)
        self.assertTrue(read.accepted, read.certificate.describe())

    def test_the_modulus_order_breaks_ties_by_real_then_imaginary_part(self):
        """Under AscendingModulus the eigenvalues 1 and -1 of diag(1, -1, 2)
        tie in modulus; the real part orders -1 first, and a band of rank one
        is that eigenvalue, isolated from 1 by 2."""
        read = _band_read(np.diag([1.0, -1.0, 2.0]), rank=1,
                          order=cob.OccupationOrder.AscendingModulus)
        self.assertEqual(list(read.eigenvalues), [-1.0])
        self.assertEqual(read.isolation_gap, 2.0)
        self.assertEqual(read.contour_radius, 1.0)

    def test_the_selection_is_recorded_as_the_declared_rule_defines_it(self):
        """Under LowestModes the centre is the mean of the selected
        eigenvalues and the radius sits halfway between the farthest selected
        and the nearest excluded eigenvalue, measured from that centre."""
        _, recursion = _recursion(levels=1)
        level = recursion.level(0)
        operator = _square(recursion.response_pencil(0, 0.0), level.dimension)
        for band, members in zip(level.bands, level.partition):
            values = np.linalg.eigvals(operator[np.ix_(members, members)])
            selected = np.asarray(band.eigenvalues, dtype=complex)
            centre = selected.mean()
            self.assertLess(abs(band.contour_centre - centre), 1e-12)
            excluded = [value for value in values
                        if np.abs(selected - value).min() > 1e-10]
            self.assertEqual(len(excluded) + band.rank, len(members))
            inside = np.abs(selected - centre).max()
            outside = min(abs(value - centre) for value in excluded)
            self.assertLess(abs(band.contour_radius - 0.5 * (inside + outside)),
                            1e-12)
            self.assertFalse(band.encloses_everything)

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

    def test_the_compression_carries_the_eigenvalues_the_selection_enclosed(self):
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
            self.assertTrue(band.encloses_everything)
            self.assertEqual(band.isolation_gap, math.inf)
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


class TheRemainingReadsTest(unittest.TestCase):
    """The base dimension, the response determinant, the persistence
    overlaps, the fibers' contour centres and spectrum, and the construction
    over a spacetime."""

    def test_the_base_dimension_and_the_response_determinant(self):
        _, recursion = _recursion(levels=1)
        self.assertEqual(recursion.base_dimension(), DIMENSION)
        for lambda_ in (0.13, 0.44 + 0.21j):
            size = recursion.response_dimension(0)
            pencil = _square(recursion.response_pencil(0, lambda_), size)
            self.assertAlmostEqual(
                abs(recursion.response_determinant(0, lambda_)
                    - np.linalg.det(pencil)), 0.0,
                delta=1e-10 * max(1.0, abs(np.linalg.det(pencil))))

    def test_the_persistence_overlap_is_declared_and_its_worst_reported(self):
        self.assertEqual(cob.LevelRecursionDeclaration().persistence_overlap,
                         0.5)
        _, recursion = _recursion(levels=1)
        level = recursion.level(0)
        self.assertGreaterEqual(level.worst_persistence_overlap, 0.5)
        self.assertLessEqual(level.worst_persistence_overlap, 1.0)
        _, single = _recursion(levels=1, resolutions=(1.0,))
        self.assertTrue(math.isnan(single.level(0).worst_persistence_overlap))

    def test_each_band_is_enclosed_by_its_contour(self):
        _, recursion = _recursion(levels=1)
        for band in recursion.level(0).bands:
            for value in band.eigenvalues:
                self.assertLess(abs(value - band.contour_centre),
                                band.contour_radius)

    def test_the_fiber_spectrum_is_that_of_the_fiber_operator(self):
        _, recursion = _recursion(levels=1)
        level = recursion.level(0)
        operator = _square(level.fiber_operator, level.modes)
        expected = sorted(np.linalg.eigvals(operator),
                          key=lambda value: (value.real, value.imag))
        np.testing.assert_allclose(level.fiber_spectrum, expected, atol=1e-9)

    def _regular_tetrahedron(self, band_rank):
        spacetime = T.Spacetime.fromVertexTuples(3, [[0, 1, 2, 3]], 1.0, 0.0)
        for edge in spacetime.getEdgeList().toVector():
            edge.setLength(math.sqrt(8.0))
        declaration = _declaration(resolutions=(1.0,), band_rank=band_rank)
        return cob.LevelRecursion.overSpacetime(
            spacetime, 1, cob.HodgeMetricSource.WhitneyPencil, declaration)

    def test_the_recursion_over_a_spacetime_reads_its_edge_operator(self):
        """Over a spacetime the base is the degree-one operator of the
        declared metric source: one coordinate per edge. The band rank is the
        number of edges, so every component's band is its whole block, which
        is a spectral set whatever the partition is."""
        recursion = self._regular_tetrahedron(band_rank=6)
        self.assertEqual(recursion.base_dimension(), 6)
        recursion.advance()
        level = recursion.level(0)
        self.assertEqual(sorted(i for part in level.partition for i in part),
                         list(range(6)))
        for band, members in zip(level.bands, level.partition):
            self.assertEqual(band.rank, len(members))
            self.assertTrue(band.encloses_everything)

    def test_a_band_rank_through_a_multiple_eigenvalue_of_a_level_is_reported(
            self):
        """The edge operator of the regular tetrahedron has multiple
        eigenvalues by symmetry, and the block of its first component carries
        one of them. A band of rank one is a part of that eigenvalue's
        eigenspace, which is no invariant subspace of its own. What the turn
        does with it depends on how the Schur form returns the multiple
        eigenvalue, and both outcomes say what happened.

        When the computed copies differ by rounding, the selection separates
        two eigenvalues equal at the declared tolerance 1e-6: the turn is
        taken, the band is read from the order of the exact keys, its
        isolation gap is at or below the tolerance times the block's
        Frobenius norm, it is not accepted, and the level's certificate does
        not hold. When the computed copies are equal exactly, the Sylvester
        equation of the projector is singular: the turn has no value, the
        read says so by name, and no level is recorded."""
        recursion = self._regular_tetrahedron(band_rank=1)
        try:
            recursion.advance()
        except ValueError as error:
            self.assertIn("leaves an eigenvalue exactly equal to it out",
                          str(error))
            self.assertEqual(recursion.level_count(), 0)
            return
        self.assertEqual(recursion.level_count(), 1)
        level = recursion.level(0)
        operator = _square(recursion.response_pencil(0, 0.0), level.dimension)
        split = [
            band for band, members in zip(level.bands, level.partition)
            if band.isolation_gap <= 1e-6 * np.linalg.norm(
                operator[np.ix_(members, members)])]
        self.assertGreaterEqual(len(split), 1)
        for band in split:
            self.assertEqual(band.rank, 1)
            self.assertGreater(band.isolation_gap, 0.0)
            self.assertFalse(band.accepted)
            self.assertFalse(band.certificate.holds())
        self.assertFalse(level.certificate.holds())

    def test_a_level_carries_a_band_of_rank_zero(self):
        """One turn over two blocks, [[1, 1/2], [1/2, 1]] on coordinates 0
        and 1 and the same plus 4 on coordinates 2 and 3, coupled by 0.01
        between coordinates 1 and 2, with a declared circle per component.

        The partition is read first under `LowestModes`; whatever it is, the
        component that holds coordinate 0 is given the circle about 0 of
        radius 100, which encloses every eigenvalue of any block of this
        operator (they lie between 0.4 and 5.6), and every other component
        the circle about 1000 of radius 1, which encloses none. The first
        band is then its whole block and the others have rank zero.

        The turn is taken. A band of rank zero contributes no mode and no
        transport: the level's modes are those of the first band, the Gram
        is the identity of that order because the band of a whole block has
        the canonical basis as both frames, and the only transport is the
        first band's own block. The bands of rank zero are not accepted, so
        the level's certificate does not hold."""
        block = np.array([[1.0, 0.5], [0.5, 1.0]])
        matrix = np.zeros((4, 4), dtype=complex)
        matrix[:2, :2] = block
        matrix[2:, 2:] = block + 4.0 * np.eye(2)
        matrix[1, 2] = matrix[2, 1] = 0.01
        flat = [complex(value) for value in matrix.reshape(-1)]
        lowest = cob.LevelRecursion.overPencil(
            flat, [], 4, _declaration(resolutions=(1.0,), band_rank=1))
        lowest.advance()
        partition = [list(part) for part in lowest.level(0).partition]
        # the two blocks are separate components, so that a component other
        # than the first exists to carry the band of rank zero
        self.assertGreaterEqual(len(partition), 2, partition)
        first = next(index for index, part in enumerate(partition)
                     if 0 in part)
        declaration = _declaration(resolutions=(1.0,))
        bands = cob.RecursionBandDeclaration()
        bands.selection = cob.RecursionBandSelection.DeclaredContours
        bands.contour_centres = [0.0 if index == first else 1000.0
                                 for index in range(len(partition))]
        bands.contour_radii = [100.0 if index == first else 1.0
                               for index in range(len(partition))]
        declaration.bands = bands
        recursion = cob.LevelRecursion.overPencil(flat, [], 4, declaration)
        recursion.advance()
        self.assertEqual(recursion.level_count(), 1)
        level = recursion.level(0)
        self.assertEqual([list(part) for part in level.partition], partition)
        order = len(partition[first])
        for index, band in enumerate(level.bands):
            with self.subTest(component=index):
                if index == first:
                    self.assertEqual(band.rank, order)
                    self.assertTrue(band.encloses_everything)
                    self.assertTrue(band.accepted,
                                    band.certificate.describe())
                else:
                    self.assertEqual(band.rank, 0)
                    self.assertEqual(list(band.eigenvalues), [])
                    self.assertEqual(list(band.frame), [])
                    self.assertEqual(list(band.left_frame), [])
                    self.assertFalse(band.accepted)
        self.assertEqual(level.modes, order)
        np.testing.assert_array_equal(_square(level.gram, order),
                                      np.eye(order))
        self.assertEqual(level.gram_defect, 0.0)
        self.assertEqual([(t.to_component, t.from_component)
                          for t in level.transports], [(first, first)])
        self.assertEqual(len(level.fiber_spectrum), order)
        self.assertEqual(level.fock_stage_dimension, 2.0 ** order)
        self.assertFalse(level.certificate.holds())


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
