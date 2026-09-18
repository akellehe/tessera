"""Exact fixtures for the exterior-algebra / graded-tensor primitives
(issue #766): OccupationBitset, ExteriorAlgebra, GradedTensorComplex,
FockDirectSum, EdgeModeRegistry.

Acceptance coverage (ticket #766):

* exhaustive CAR and sign tests on all three-mode basis states (and every
  induced mode reordering);
* duplicate complete one-particle modes wedge to exactly zero;
* random vertex relabelings preserve all physical amplitudes after applying
  permutation parity;
* product-complex Hodge fixtures match the graded tensor construction, and
  second-quantized direct-sum/hopping fixtures match dense Fock references;
* pair creation changes occupation number by two and preserves total
  fermion parity.

Exactness bar: CAR, wedge signs, dimension identities and Gram/Pauli
determinants are integer/algebraic identities — integer-valued fixtures are
compared with exact equality, floating fixtures to double round-off.

The dense Fock references are INDEPENDENT numpy Jordan-Wigner constructions
(kron chains), not re-derivations through the bindings under test.

Skips cleanly when tessera was built without the quantum subsystem.
"""

from __future__ import annotations

import itertools
import unittest
from math import comb

import numpy as np

try:
    from tessera.quantum import (
        EdgeModeRegistry,
        ExteriorAlgebra,
        FockDirectSum,
        GradedTensorComplex,
        OccupationBitset,
    )
    HAVE_QUANTUM = True
except ImportError:
    HAVE_QUANTUM = False


# ─── helpers ───────────────────────────────────────────────────────────────

def dense(coo):
    """(rows, cols, values, n) COO tuple -> dense complex ndarray."""
    rows, cols, vals, n = coo
    out = np.zeros((n, n), dtype=complex)
    for r, c, v in zip(rows, cols, vals):
        out[r, c] += v
    return out


# Two-level factor basis {|0>, |1>}: annihilation |1> -> |0>.
_S_MINUS = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
_Z = np.diag([1.0, -1.0]).astype(complex)


def jw_annihilation(mode: int, n_modes: int) -> np.ndarray:
    """Independent dense Jordan-Wigner a_mode on the n(b) = sum b_i 2^i basis.

    Mode 0 is the least-significant bit, so the FIRST kron factor is mode
    n_modes-1; the Z string sits on the modes strictly below `mode`.
    """
    op = np.eye(1, dtype=complex)
    for m in range(n_modes - 1, -1, -1):
        if m > mode:
            factor = np.eye(2, dtype=complex)
        elif m == mode:
            factor = _S_MINUS
        else:
            factor = _Z
        op = np.kron(op, factor)
    return op


def anticommutator(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return x @ y + y @ x


def inversion_parity(seq) -> int:
    inv = sum(
        1
        for i in range(len(seq))
        for j in range(i + 1, len(seq))
        if seq[i] > seq[j]
    )
    return -1 if inv % 2 else +1


# Boundary of the triangulated circle with 3 vertices / 3 edges
# (e0: 0->1, e1: 1->2, e2: 2->0) and of the interval (one edge 0->1).
CIRCLE_D1 = np.array(
    [[-1.0, 0.0, 1.0], [1.0, -1.0, 0.0], [0.0, 1.0, -1.0]], dtype=complex
)
INTERVAL_D1 = np.array([[-1.0], [1.0]], dtype=complex)


# ─── OccupationBitset ──────────────────────────────────────────────────────

@unittest.skipUnless(HAVE_QUANTUM, "tessera built without the quantum subsystem")
class TestOccupationBitset(unittest.TestCase):
    """Chunked bitsets with the exact prefix-popcount sign rule."""

    def test_chunking_thresholds(self) -> None:
        self.assertEqual(OccupationBitset(0).chunkCount(), 0)
        self.assertEqual(OccupationBitset(1).chunkCount(), 1)
        self.assertEqual(OccupationBitset(64).chunkCount(), 1)
        self.assertEqual(OccupationBitset(65).chunkCount(), 2)
        self.assertEqual(OccupationBitset(200).chunkCount(), 4)

    def test_prefix_popcount_matches_reference_across_chunks(self) -> None:
        rng = np.random.default_rng(7)
        n_modes = 200
        occupied = sorted(rng.choice(n_modes, size=60, replace=False).tolist())
        b = OccupationBitset.fromOccupiedModes(n_modes, occupied)
        self.assertEqual(b.count(), 60)
        self.assertEqual(b.occupiedModes(), occupied)
        for probe in [0, 1, 63, 64, 65, 127, 128, 129, 191, 199, 200]:
            expected = sum(1 for m in occupied if m < probe)
            self.assertEqual(b.prefixPopcount(probe), expected)

    def test_creation_annihilation_signs_across_chunk_boundaries(self) -> None:
        n_modes = 200
        b = OccupationBitset.fromOccupiedModes(n_modes, [0, 63, 64, 128])
        # prefix below 65 = {0, 63, 64} -> odd -> sign -1.
        self.assertEqual(b.applyCreation(65), -1)
        self.assertTrue(b.test(65))
        # Pauli exclusion: 0 and state unchanged.
        chunks_before = b.chunks()
        self.assertEqual(b.applyCreation(65), 0)
        self.assertEqual(b.chunks(), chunks_before)
        # annihilate it again: same prefix -> same sign.
        self.assertEqual(b.applyAnnihilation(65), -1)
        self.assertFalse(b.test(65))
        # annihilating an empty mode is 0.
        self.assertEqual(b.applyAnnihilation(65), 0)
        # parity flips with each successful creation.
        self.assertEqual(b.parity(), +1)  # 4 occupied
        b.applyCreation(199)
        self.assertEqual(b.parity(), -1)

    def test_index_round_trip_and_validation(self) -> None:
        b = OccupationBitset.fromIndex(5, 0b10110)
        self.assertEqual(b.occupiedModes(), [1, 2, 4])
        self.assertEqual(b.toIndex(), 0b10110)
        with self.assertRaises(ValueError):
            OccupationBitset.fromIndex(3, 8)  # index >= 2^3
        with self.assertRaises(ValueError):
            OccupationBitset.fromIndex(65, 0)  # too many modes for an index
        with self.assertRaises(ValueError):
            OccupationBitset.fromOccupiedModes(4, [1, 1])  # duplicate
        with self.assertRaises(ValueError):
            OccupationBitset(4).test(4)  # out of range

    def test_permutation_parity_matches_inversion_count(self) -> None:
        rng = np.random.default_rng(11)
        n_modes = 12
        for _ in range(50):
            perm = rng.permutation(n_modes).tolist()
            occupied = sorted(
                rng.choice(
                    n_modes, size=int(rng.integers(0, n_modes + 1)), replace=False
                ).tolist()
            )
            b = OccupationBitset.fromOccupiedModes(n_modes, occupied)
            images = [perm[m] for m in occupied]
            self.assertEqual(b.permutationParity(perm), inversion_parity(images))
            self.assertEqual(sorted(b.permuted(perm).occupiedModes()),
                             sorted(images))
        with self.assertRaises(ValueError):
            OccupationBitset(3).permutationParity([0, 0, 1])  # not a bijection

    def test_correct_at_large_mode_count(self) -> None:
        """Data-structure correctness far above the machine-word threshold."""
        n_modes = 4096
        b = OccupationBitset(n_modes)
        self.assertEqual(b.chunkCount(), 64)
        occupied = list(range(0, n_modes, 97))
        for m in occupied:
            b.set(m)
        self.assertEqual(b.count(), len(occupied))
        for probe in (64, 970, 2048, 4095, 4096):
            self.assertEqual(
                b.prefixPopcount(probe), sum(1 for m in occupied if m < probe)
            )
        # creation sign at the top of the range: 42 occupied below 4090.
        below = sum(1 for m in occupied if m < 4090)
        self.assertEqual(b.applyCreation(4090), (-1) ** below)


# ─── CAR: exhaustive three-mode fixtures ───────────────────────────────────

@unittest.skipUnless(HAVE_QUANTUM, "tessera built without the quantum subsystem")
class TestCARThreeModesExhaustive(unittest.TestCase):
    """{a_i, a_j} = 0, {a_i+, a_j+} = 0, {a_i, a_j+} = delta_ij — exhaustive
    at M = 3, exact (integer identities), against independent JW references,
    on every basis state and under every induced mode reordering."""

    M = 3

    def setUp(self) -> None:
        self.alg = ExteriorAlgebra(self.M)
        self.a = [dense(self.alg.annihilationMatrixCOO(i)) for i in range(self.M)]
        self.adag = [dense(self.alg.creationMatrixCOO(i)) for i in range(self.M)]

    def test_matrices_match_independent_jordan_wigner(self) -> None:
        for i in range(self.M):
            ref = jw_annihilation(i, self.M)
            np.testing.assert_array_equal(self.a[i], ref)
            np.testing.assert_array_equal(self.adag[i], ref.conj().T)

    def test_car_exhaustive(self) -> None:
        eye = np.eye(2**self.M, dtype=complex)
        zero = np.zeros_like(eye)
        for i in range(self.M):
            for j in range(self.M):
                np.testing.assert_array_equal(
                    anticommutator(self.a[i], self.a[j]), zero
                )
                np.testing.assert_array_equal(
                    anticommutator(self.adag[i], self.adag[j]), zero
                )
                expected = eye if i == j else zero
                np.testing.assert_array_equal(
                    anticommutator(self.a[i], self.adag[j]), expected
                )

    def test_bit_level_signs_match_matrices_on_all_basis_states(self) -> None:
        for idx in range(2**self.M):
            for mode in range(self.M):
                for matrix, apply_name in (
                    (self.adag[mode], "applyCreation"),
                    (self.a[mode], "applyAnnihilation"),
                ):
                    bits = OccupationBitset.fromIndex(self.M, idx)
                    sign = getattr(bits, apply_name)(mode)
                    column = matrix[:, idx]
                    if sign == 0:
                        np.testing.assert_array_equal(column, 0)
                        self.assertEqual(bits.toIndex(), idx)  # unchanged
                    else:
                        expected = np.zeros(2**self.M, dtype=complex)
                        expected[bits.toIndex()] = sign
                        np.testing.assert_array_equal(column, expected)

    def test_car_under_every_induced_mode_reordering(self) -> None:
        eye = np.eye(2**self.M, dtype=complex)
        zero = np.zeros_like(eye)
        for perm in itertools.permutations(range(self.M)):
            u = dense(self.alg.modePermutationMatrixCOO(list(perm)))
            # U is a signed permutation unitary.
            np.testing.assert_array_equal(u.conj().T @ u, eye)
            # U a_i U+ = a_perm(i): the reordering is exactly intertwined.
            for i in range(self.M):
                np.testing.assert_array_equal(
                    u @ self.a[i] @ u.conj().T, self.a[perm[i]]
                )
            # And the reordered generators satisfy the CAR verbatim.
            b0 = u @ self.a[0] @ u.conj().T
            b1 = u @ self.adag[1] @ u.conj().T
            np.testing.assert_array_equal(anticommutator(b0, b1), zero)

    def test_number_parity_and_diagonal_operators(self) -> None:
        n_total = dense(self.alg.totalNumberMatrixCOO())
        parity = dense(self.alg.parityMatrixCOO())
        n_sum = sum(dense(self.alg.numberMatrixCOO(i)) for i in range(self.M))
        np.testing.assert_array_equal(n_total, n_sum)
        for idx in range(2**self.M):
            n = bin(idx).count("1")
            self.assertEqual(n_total[idx, idx], n)
            self.assertEqual(parity[idx, idx], (-1) ** n)


# ─── dimension, wedge, Gram determinant, contraction ───────────────────────

@unittest.skipUnless(HAVE_QUANTUM, "tessera built without the quantum subsystem")
class TestDimensionAndWedge(unittest.TestCase):
    def test_fock_dimension_is_two_to_the_m(self) -> None:
        for m in range(0, 9):
            alg = ExteriorAlgebra(m)
            self.assertEqual(alg.fockDimension(), 2**m)
            self.assertEqual(len(alg.vacuumState()), 2**m)

    def test_matrix_layer_mode_cap_fails_loudly(self) -> None:
        with self.assertRaises(ValueError):
            ExteriorAlgebra(25)

    def test_wedge_of_basis_modes_reproduces_bitset_states(self) -> None:
        m = 4
        alg = ExteriorAlgebra(m)
        basis = np.eye(m, dtype=complex)
        for occupied in itertools.chain.from_iterable(
            itertools.combinations(range(m), k) for k in range(m + 1)
        ):
            state = alg.wedge([basis[i] for i in occupied])
            expected = alg.basisState(
                OccupationBitset.fromOccupiedModes(m, list(occupied))
            )
            np.testing.assert_array_equal(state, expected)

    def test_wedge_antisymmetry_sign(self) -> None:
        m = 4
        alg = ExteriorAlgebra(m)
        basis = np.eye(m, dtype=complex)
        forward = alg.wedge([basis[0], basis[2]])
        backward = alg.wedge([basis[2], basis[0]])
        np.testing.assert_array_equal(forward, -backward)

    def test_gram_determinant_identity(self) -> None:
        """||v1 ^ ... ^ vn||^2 = det(<vi, vj>) to double round-off."""
        rng = np.random.default_rng(23)
        m = 6
        alg = ExteriorAlgebra(m)
        for n in range(1, 5):
            vectors = [
                rng.standard_normal(m) + 1j * rng.standard_normal(m)
                for _ in range(n)
            ]
            state = alg.wedge(vectors)
            norm_sq = float(np.vdot(state, state).real)
            gram = np.array(
                [[np.vdot(vi, vj) for vj in vectors] for vi in vectors]
            )
            det = np.linalg.det(gram)
            self.assertAlmostEqual(det.imag, 0.0, delta=1e-12 * abs(det))
            self.assertAlmostEqual(
                norm_sq, det.real, delta=1e-13 * max(1.0, abs(det.real))
            )

    def test_duplicate_complete_modes_wedge_to_exactly_zero(self) -> None:
        m = 5
        alg = ExteriorAlgebra(m)
        basis = np.eye(m, dtype=complex)
        # Repeated complete basis mode: exact zero at the bit level.
        np.testing.assert_array_equal(
            alg.wedge([basis[2], basis[2]]), np.zeros(2**m)
        )
        np.testing.assert_array_equal(
            alg.wedge([basis[1], basis[3], basis[1]]), np.zeros(2**m)
        )
        # A repeated GENERAL one-particle vector cancels to double round-off
        # (not bitwise: FMA contraction makes complex v_i*v_j and v_j*v_i
        # differ in the last bit of the imaginary part). The EXACT-zero
        # guarantee of the ticket is for duplicate complete modes above.
        rng = np.random.default_rng(3)
        v = rng.standard_normal(m) + 1j * rng.standard_normal(m)
        np.testing.assert_allclose(alg.wedge([v, v]), np.zeros(2**m),
                                   atol=1e-14)
        # For a repeated REAL vector the products are bitwise equal and the
        # cancellation is exact.
        vr = (rng.standard_normal(m) + 0j)
        np.testing.assert_array_equal(alg.wedge([vr, vr]), np.zeros(2**m))
        # More vectors than modes is identically zero as well (dim reason).
        vs = [rng.standard_normal(m) + 1j * rng.standard_normal(m)
              for _ in range(m + 1)]
        np.testing.assert_allclose(alg.wedge(vs), np.zeros(2**m), atol=1e-12)

    def test_smeared_car_and_contraction(self) -> None:
        rng = np.random.default_rng(5)
        m = 5
        alg = ExteriorAlgebra(m)
        v = rng.standard_normal(m) + 1j * rng.standard_normal(m)
        w = rng.standard_normal(m) + 1j * rng.standard_normal(m)
        a_w = dense(alg.annihilationOperatorCOO(w))
        adag_v = dense(alg.creationOperatorCOO(v))
        # {a(w), a+(v)} = <w, v> * I.
        np.testing.assert_allclose(
            anticommutator(a_w, adag_v),
            np.vdot(w, v) * np.eye(2**m),
            atol=1e-14,
        )
        # contract() is a(w) applied to the state.
        state = alg.wedge([v, rng.standard_normal(m) + 0j])
        np.testing.assert_allclose(
            alg.contract(w, state), a_w @ state, atol=1e-14
        )
        # Interior product is an odd antiderivation:
        # i_w(x ^ y) = (i_w x) ^ y + (-1)^deg(x) x ^ (i_w y) for 1-vectors.
        x = rng.standard_normal(m) + 1j * rng.standard_normal(m)
        y = rng.standard_normal(m) + 1j * rng.standard_normal(m)
        lhs = alg.contract(w, alg.wedge([x, y]))
        rhs = complex(np.vdot(w, x)) * alg.wedge([y]) - complex(
            np.vdot(w, y)
        ) * alg.wedge([x])
        np.testing.assert_allclose(lhs, rhs, atol=1e-13)


# ─── occupation-sector projectors ──────────────────────────────────────────

@unittest.skipUnless(HAVE_QUANTUM, "tessera built without the quantum subsystem")
class TestSectorProjectors(unittest.TestCase):
    def test_three_mode_subset_projectors_are_exact(self) -> None:
        m = 5
        subset = [1, 2, 4]
        alg = ExteriorAlgebra(m)
        projectors = [
            dense(alg.subsetSectorProjectorCOO(subset, n)) for n in range(4)
        ]
        eye = np.eye(2**m, dtype=complex)
        # Resolution of the identity and exact orthogonal idempotents.
        np.testing.assert_array_equal(sum(projectors), eye)
        for n, p in enumerate(projectors):
            np.testing.assert_array_equal(p @ p, p)
            for k in range(n + 1, 4):
                np.testing.assert_array_equal(p @ projectors[k], 0 * eye)
            # rank Lambda^n of a 3-mode factor = C(3, n) * 2^(M-3).
            self.assertEqual(int(np.trace(p).real), comb(3, n) * 2 ** (m - 3))
            # Subset number operator acts as n on the sector.
            n_subset = sum(dense(alg.numberMatrixCOO(i)) for i in subset)
            np.testing.assert_array_equal(n_subset @ p, n * p)

    def test_total_sector_projectors(self) -> None:
        m = 4
        alg = ExteriorAlgebra(m)
        eye = np.eye(2**m, dtype=complex)
        total = sum(dense(alg.sectorProjectorCOO(n)) for n in range(m + 1))
        np.testing.assert_array_equal(total, eye)
        # N = sum_n n * P_n exactly.
        n_from_sectors = sum(
            n * dense(alg.sectorProjectorCOO(n)) for n in range(m + 1)
        )
        np.testing.assert_array_equal(
            n_from_sectors, dense(alg.totalNumberMatrixCOO())
        )
        with self.assertRaises(ValueError):
            alg.subsetSectorProjectorCOO([0, 0, 1], 1)  # duplicate mode


# ─── graded swap: elementary parity combinations ───────────────────────────

@unittest.skipUnless(HAVE_QUANTUM, "tessera built without the quantum subsystem")
class TestGradedSwap(unittest.TestCase):
    def test_odd_odd_is_minus_one_all_others_plus_one(self) -> None:
        m_a, m_b = 2, 2
        f = FockDirectSum(m_a, m_b)
        s = dense(f.gradedSwapMatrixCOO())
        dim_a, dim_b = 2**m_a, 2**m_b
        for i_a in range(dim_a):
            for i_b in range(dim_b):
                col = i_a + dim_a * i_b       # |i_a> x |i_b> in F_A x F_B
                row = i_b + dim_b * i_a       # |i_b> x |i_a> in F_B x F_A
                p_a = bin(i_a).count("1") % 2
                p_b = bin(i_b).count("1") % 2
                expected = -1.0 if (p_a == 1 and p_b == 1) else +1.0
                column = np.zeros(dim_a * dim_b, dtype=complex)
                column[row] = expected
                np.testing.assert_array_equal(s[:, col], column)

    def test_swap_is_unitary_and_squares_to_identity(self) -> None:
        f_ab = FockDirectSum(1, 2)
        f_ba = FockDirectSum(2, 1)
        s_ab = dense(f_ab.gradedSwapMatrixCOO())
        s_ba = dense(f_ba.gradedSwapMatrixCOO())
        eye = np.eye(8, dtype=complex)
        np.testing.assert_array_equal(s_ab.conj().T @ s_ab, eye)
        np.testing.assert_array_equal(s_ba @ s_ab, eye)


# ─── graded tensor differential and product-complex Hodge fixtures ────────

@unittest.skipUnless(HAVE_QUANTUM, "tessera built without the quantum subsystem")
class TestGradedTensorComplex(unittest.TestCase):
    """The cubical torus (circle x circle) and cylinder (interval x circle)
    are ACTUAL product cell complexes whose chain complex equals the graded
    tensor construction on the nose; every comparison here is exact or to
    double round-off."""

    def torus(self) -> "GradedTensorComplex":
        return GradedTensorComplex(
            [3, 3], [CIRCLE_D1], [3, 3], [CIRCLE_D1], 0.0
        )

    def test_dimensions_and_blocks(self) -> None:
        prod = self.torus()
        self.assertEqual(prod.maxDegree(), 2)
        self.assertEqual(
            [prod.chainDimension(n) for n in range(3)], [9, 18, 9]
        )
        self.assertEqual(prod.blocks(1), [(0, 1), (1, 0)])
        self.assertEqual(prod.blocks(2), [(1, 1)])

    def test_differential_matches_hand_built_cubical_torus(self) -> None:
        prod = self.torus()
        eye3 = np.eye(3, dtype=complex)
        # Independent hand assembly of the cubical-torus boundary operators
        # in the documented block convention (blocks ascending p; within a
        # block, index = i_a * dimB + i_b):
        #   C_1 = (v x e) ++ (e x v);  d(v x e) = v x de,  d(e x v) = de x v
        #   d(e x e) = de x e - e x de   (Koszul sign (-1)^1 on the second).
        d1_hand = np.hstack([np.kron(eye3, CIRCLE_D1), np.kron(CIRCLE_D1, eye3)])
        d2_hand = np.vstack(
            [np.kron(CIRCLE_D1, eye3), -np.kron(eye3, CIRCLE_D1)]
        )
        np.testing.assert_array_equal(prod.differential(1), d1_hand)
        np.testing.assert_array_equal(prod.differential(2), d2_hand)

    def test_boundary_of_boundary_is_exactly_zero(self) -> None:
        prod = self.torus()
        np.testing.assert_array_equal(
            prod.differential(1) @ prod.differential(2), np.zeros((9, 9))
        )

    def test_graded_leibniz_rule_on_product_elements(self) -> None:
        """d(a x b) = da x b + (-1)^deg(a) a x db, blockwise exact."""
        rng = np.random.default_rng(17)
        prod = self.torus()
        a = rng.standard_normal(3) + 1j * rng.standard_normal(3)  # a in A_1
        b = rng.standard_normal(3) + 1j * rng.standard_normal(3)  # b in B_1
        chain = np.kron(a, b)  # the only degree-2 block is (1, 1)
        image = prod.differential(2) @ chain
        # Blocks of C_1 are ordered [(0, 1), (1, 0)]: block (0, 1) = A_0 x B_1
        # receives da x b; block (1, 0) = A_1 x B_0 receives (-1)^1 a x db.
        expected = np.concatenate(
            [np.kron(CIRCLE_D1 @ a, b), -np.kron(a, CIRCLE_D1 @ b)]
        )
        np.testing.assert_allclose(image, expected, atol=1e-14)

    def test_torus_betti_numbers_via_kunneth(self) -> None:
        prod = self.torus()
        d1 = prod.differential(1)
        d2 = prod.differential(2)
        r1 = np.linalg.matrix_rank(d1)
        r2 = np.linalg.matrix_rank(d2)
        b0 = 9 - r1
        b1 = 18 - r1 - r2
        b2 = 9 - r2
        self.assertEqual((b0, b1, b2), (1, 2, 1))

    def test_hodge_spectra_are_pairwise_sums(self) -> None:
        """spec Delta_n(A x B) = multiset union over p+q=n of
        { lambda_p^A + mu_q^B } — Kunneth at the Hodge level."""
        for prod, dims_a, dims_b in (
            (self.torus(), [3, 3], [3, 3]),
            (
                GradedTensorComplex([2, 1], [INTERVAL_D1], [3, 3], [CIRCLE_D1]),
                [2, 1],
                [3, 3],
            ),
        ):
            max_p = len(dims_a) - 1
            max_q = len(dims_b) - 1
            spec_a = [
                np.linalg.eigvalsh(prod.factorLaplacianA(p)) for p in range(max_p + 1)
            ]
            spec_b = [
                np.linalg.eigvalsh(prod.factorLaplacianB(q)) for q in range(max_q + 1)
            ]
            for n in range(prod.maxDegree() + 1):
                got = np.sort(np.linalg.eigvalsh(prod.laplacian(n)))
                expected = np.sort(
                    np.concatenate(
                        [
                            (spec_a[p][:, None] + spec_b[n - p][None, :]).ravel()
                            for p in range(max_p + 1)
                            if 0 <= n - p <= max_q
                        ]
                    )
                )
                np.testing.assert_allclose(got, expected, atol=1e-10)

    def test_cylinder_betti_numbers(self) -> None:
        prod = GradedTensorComplex([2, 1], [INTERVAL_D1], [3, 3], [CIRCLE_D1])
        d1 = prod.differential(1)
        d2 = prod.differential(2)
        b0 = prod.chainDimension(0) - np.linalg.matrix_rank(d1)
        b1 = (
            prod.chainDimension(1)
            - np.linalg.matrix_rank(d1)
            - np.linalg.matrix_rank(d2)
        )
        b2 = prod.chainDimension(2) - np.linalg.matrix_rank(d2)
        self.assertEqual((b0, b1, b2), (1, 1, 0))

    def test_constructor_validation(self) -> None:
        with self.assertRaises(ValueError):
            GradedTensorComplex([3, 2], [CIRCLE_D1], [3, 3], [CIRCLE_D1])
        bogus = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
        with self.assertRaises(ValueError):
            # d o d != 0: two identity "differentials".
            GradedTensorComplex([2, 2, 2], [bogus, bogus], [3, 3], [CIRCLE_D1])
        with self.assertRaises(ValueError):
            self.torus().differential(0)
        with self.assertRaises(ValueError):
            self.torus().differential(3)


# ─── Fock direct-sum functor, dGamma, hopping fixtures ─────────────────────

@unittest.skipUnless(HAVE_QUANTUM, "tessera built without the quantum subsystem")
class TestFockDirectSum(unittest.TestCase):
    M_A = 2
    M_B = 2

    def setUp(self) -> None:
        self.f = FockDirectSum(self.M_A, self.M_B)
        self.joint = self.f.jointAlgebra()
        self.left = self.f.leftAlgebra()
        self.right = self.f.rightAlgebra()

    def test_direct_sums_become_graded_tensor_products(self) -> None:
        """Joint CAR generators equal the graded lifts EXACTLY: left lifts
        are X x 1; right ODD lifts carry the (-1)^N_A Koszul twist."""
        for i in range(self.M_A):
            lifted = dense(
                self.f.liftLeftCOO(dense(self.left.creationMatrixCOO(i)))
            )
            np.testing.assert_array_equal(
                dense(self.joint.creationMatrixCOO(i)), lifted
            )
        for j in range(self.M_B):
            lifted = dense(
                self.f.liftRightCOO(
                    dense(self.right.creationMatrixCOO(j)), True
                )
            )
            np.testing.assert_array_equal(
                dense(self.joint.creationMatrixCOO(self.M_A + j)), lifted
            )
        # An even right operator lifts without the twist.
        n_b = dense(self.right.numberMatrixCOO(1))
        np.testing.assert_array_equal(
            dense(self.f.liftRightCOO(n_b, False)),
            dense(self.joint.numberMatrixCOO(self.M_A + 1)),
        )

    def test_wrong_koszul_twist_fails(self) -> None:
        """Negative control: lifting an ODD operator without the parity
        twist does NOT reproduce the joint generator."""
        a_b0 = dense(self.right.creationMatrixCOO(0))
        wrong = dense(self.f.liftRightCOO(a_b0, False))
        right_gen = dense(self.joint.creationMatrixCOO(self.M_A))
        self.assertTrue(np.any(wrong != right_gen))

    def test_block_diagonal_dgamma_is_sum_of_lifts_integer_exact(self) -> None:
        l_a = np.array([[1.0, 2.0], [2.0, -1.0]], dtype=complex)
        l_b = np.array([[3.0, 1.0], [1.0, 0.0]], dtype=complex)
        zero_c = np.zeros((self.M_A, self.M_B), dtype=complex)
        got = dense(self.f.dGammaBlockCOO(l_a, l_b, zero_c))
        expected = dense(
            self.f.liftLeftCOO(dense(self.left.dGammaCOO(l_a)))
        ) + dense(
            self.f.liftRightCOO(dense(self.right.dGammaCOO(l_b)), False)
        )
        np.testing.assert_array_equal(got, expected)

    def test_dgamma_matches_dense_fock_reference(self) -> None:
        """dGamma of a full block one-particle operator equals the
        independent dense JW Fock reference sum_ij L_ij a_i+ a_j."""
        rng = np.random.default_rng(31)
        m = self.M_A + self.M_B
        l_a = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
        l_a = l_a + l_a.conj().T
        l_b = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
        l_b = l_b + l_b.conj().T
        c = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
        l_full = np.asarray(self.f.assembleBlockOneParticle(l_a, l_b, c))
        # Assembly itself is exact.
        np.testing.assert_array_equal(l_full[:2, :2], l_a)
        np.testing.assert_array_equal(l_full[2:, 2:], l_b)
        np.testing.assert_array_equal(l_full[:2, 2:], c)
        np.testing.assert_array_equal(l_full[2:, :2], c.conj().T)

        got = dense(self.f.dGammaBlockCOO(l_a, l_b, c))
        a_ops = [jw_annihilation(i, m) for i in range(m)]
        reference = sum(
            l_full[i, j] * (a_ops[i].conj().T @ a_ops[j])
            for i in range(m)
            for j in range(m)
        )
        np.testing.assert_allclose(got, reference, atol=1e-13)
        # Hermitian L gives a Hermitian dGamma(L).
        np.testing.assert_allclose(got, got.conj().T, atol=1e-13)

    def test_coupling_blocks_become_hopping_terms(self) -> None:
        rng = np.random.default_rng(37)
        m = self.M_A + self.M_B
        c = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
        zero2 = np.zeros((2, 2), dtype=complex)
        hopping = dense(self.f.dGammaBlockCOO(zero2, zero2, c))
        a_ops = [jw_annihilation(i, m) for i in range(m)]
        reference = sum(
            c[i, j - self.M_A] * (a_ops[i].conj().T @ a_ops[j])
            + np.conj(c[i, j - self.M_A]) * (a_ops[j].conj().T @ a_ops[i])
            for i in range(self.M_A)
            for j in range(self.M_A, m)
        )
        np.testing.assert_allclose(hopping, reference, atol=1e-13)

    def test_dgamma_spectrum_is_occupation_subset_sums(self) -> None:
        rng = np.random.default_rng(41)
        m = 4
        alg = ExteriorAlgebra(m)
        l = rng.standard_normal((m, m)) + 1j * rng.standard_normal((m, m))
        l = l + l.conj().T
        one_particle = np.linalg.eigvalsh(l)
        many_body = np.sort(np.linalg.eigvalsh(dense(alg.dGammaCOO(l))))
        subset_sums = np.sort(
            [
                sum(one_particle[list(s)])
                for k in range(m + 1)
                for s in itertools.combinations(range(m), k)
            ]
        )
        np.testing.assert_allclose(many_body, subset_sums, atol=1e-10)

    def test_dgamma_commutes_with_number_operator(self) -> None:
        rng = np.random.default_rng(43)
        m = 4
        alg = ExteriorAlgebra(m)
        l = rng.standard_normal((m, m)) + 1j * rng.standard_normal((m, m))
        dg = dense(alg.dGammaCOO(l))
        n = dense(alg.totalNumberMatrixCOO())
        np.testing.assert_allclose(dg @ n - n @ dg, 0 * dg, atol=1e-13)


# ─── pair creation ─────────────────────────────────────────────────────────

@unittest.skipUnless(HAVE_QUANTUM, "tessera built without the quantum subsystem")
class TestPairCreation(unittest.TestCase):
    def test_pair_creation_raises_n_by_two_and_preserves_parity(self) -> None:
        m = 4
        alg = ExteriorAlgebra(m)
        adag = [dense(alg.creationMatrixCOO(i)) for i in range(m)]
        n = dense(alg.totalNumberMatrixCOO())
        p = dense(alg.parityMatrixCOO())
        for i, j in itertools.combinations(range(m), 2):
            q = adag[i] @ adag[j]
            # [N, Q] = 2 Q: occupation number changes by exactly two.
            np.testing.assert_array_equal(n @ q - q @ n, 2 * q)
            # [P, Q] = 0: total fermion parity is preserved.
            np.testing.assert_array_equal(p @ q - q @ p, 0 * q)
            # Nilpotent: creating the same pair twice is exactly zero.
            np.testing.assert_array_equal(q @ q, 0 * q)
            # On every basis state where it acts, N goes up by two.
            for idx in range(2**m):
                column = q[:, idx]
                nonzero = np.nonzero(column)[0]
                if len(nonzero) == 0:
                    continue
                self.assertEqual(len(nonzero), 1)
                target = int(nonzero[0])
                self.assertEqual(
                    bin(target).count("1"), bin(idx).count("1") + 2
                )
                self.assertIn(column[target], (1.0 + 0j, -1.0 + 0j))


# ─── edge-mode registry: convention + compilation order ────────────────────

@unittest.skipUnless(HAVE_QUANTUM, "tessera built without the quantum subsystem")
class TestEdgeModeRegistry(unittest.TestCase):
    def build_registry(self) -> "EdgeModeRegistry":
        reg = EdgeModeRegistry()
        # Two lineage components, mixed insertion order and stored
        # directions: a triangle {0,1,2} and a path 3-4-5.
        reg.addEdge(4, 3, +1, "root/1")
        reg.addEdge(0, 1, +1, "root/0")
        reg.addEdge(2, 0, -1, "root/0")
        reg.addEdge(5, 4, -1, "root/1")
        reg.addEdge(1, 2, +1, "root/0")
        reg.addEdge(3, 5, +1, "root/1")  # not a path edge; still fine
        return reg

    def test_canonical_order_is_lineage_then_vertex_pair(self) -> None:
        reg = self.build_registry()
        order = reg.canonicalModeOrder()
        keyed = [
            (
                reg.record(mode_id).lineageKey,
                min(reg.record(mode_id).vertexA, reg.record(mode_id).vertexB),
                max(reg.record(mode_id).vertexA, reg.record(mode_id).vertexB),
            )
            for mode_id in order
        ]
        self.assertEqual(keyed, sorted(keyed))
        # positions invert the order.
        positions = reg.compilationPositions()
        for pos, mode_id in enumerate(order):
            self.assertEqual(positions[mode_id], pos)

    def test_storage_reversal_changes_nothing_observable(self) -> None:
        reg = self.build_registry()
        order_before = reg.canonicalModeOrder()
        signs_before = [
            reg.canonicalOrientationSign(m) for m in range(reg.modeCount())
        ]
        for mode_id in (0, 2, 5):
            reg.reverseStoredDirection(mode_id)
        self.assertEqual(reg.canonicalModeOrder(), order_before)
        self.assertEqual(
            [reg.canonicalOrientationSign(m) for m in range(reg.modeCount())],
            signs_before,
        )
        # But the stored record did flip.
        rec = reg.record(0)
        self.assertEqual((rec.vertexA, rec.vertexB, rec.orientationSign),
                         (3, 4, -1))

    def test_flip_orientation_flips_the_physical_sign(self) -> None:
        reg = self.build_registry()
        before = reg.canonicalOrientationSign(1)
        reg.flipOrientation(1)
        self.assertEqual(reg.canonicalOrientationSign(1), -before)
        self.assertEqual(reg.canonicalModeOrder(),
                         self.build_registry().canonicalModeOrder())

    def test_registration_validation(self) -> None:
        reg = self.build_registry()
        with self.assertRaises(ValueError):
            reg.addEdge(1, 0, +1, "root/0")  # duplicate unordered pair
        with self.assertRaises(ValueError):
            reg.addEdge(7, 7, +1, "root/2")  # self-loop
        with self.assertRaises(ValueError):
            reg.addEdge(8, 9, 0, "root/2")  # invalid sign
        with self.assertRaises(ValueError):
            reg.record(99)

    def test_reversal_invariant_amplitudes_via_canonical_signs(self) -> None:
        """One-particle data read through canonicalOrientationSign is
        invariant under storage reversals and flips sign under a physical
        orientation flip — the documented reorientation convention."""
        rng = np.random.default_rng(53)
        reg = self.build_registry()
        m = reg.modeCount()
        alg = ExteriorAlgebra(m)
        coeffs = rng.standard_normal(m) + 1j * rng.standard_normal(m)

        def physical_vector(registry) -> np.ndarray:
            positions = registry.compilationPositions()
            v = np.zeros(m, dtype=complex)
            for mode_id in range(m):
                v[positions[mode_id]] = (
                    coeffs[mode_id]
                    * registry.canonicalOrientationSign(mode_id)
                )
            return v

        baseline_vector = physical_vector(reg)
        baseline_wedge = alg.wedge([baseline_vector])
        # Storage reversals change nothing: identical one-particle data,
        # identical amplitudes, exactly.
        for mode_id in (1, 3, 4):
            reg.reverseStoredDirection(mode_id)
        np.testing.assert_array_equal(physical_vector(reg), baseline_vector)
        np.testing.assert_array_equal(
            alg.wedge([physical_vector(reg)]), baseline_wedge
        )
        # A physical orientation flip multiplies exactly that mode's
        # one-particle component by -1 and leaves the others alone.
        reg.flipOrientation(2)
        flipped = physical_vector(reg)
        position = reg.compilationPositions()[2]
        expected = baseline_vector.copy()
        expected[position] = -expected[position]
        np.testing.assert_array_equal(flipped, expected)


# ─── the per-edge carrier the ontology names ───────────────────────────────

@unittest.skipUnless(HAVE_QUANTUM, "tessera built without the quantum subsystem")
class TestRegistryFromSpacetime(unittest.TestCase):
    """`EdgeModeRegistry.fromSpacetime` builds the ontology's carrier: one
    two-level occupation mode per EDGE, h_K = span{|e> : e in K1} (#804).
    Before it existed, every carrier in the tree was built over a band's
    degree-k cells, so at degree two the one-particle modes were triangles."""

    @staticmethod
    def _spacetime(cells):
        # `import tessera` then attribute access: the submodule is not
        # importable directly.
        import tessera
        return tessera.spacetime.Spacetime.fromVertexTuples(2, cells)

    def test_one_mode_per_edge(self):
        st = self._spacetime([[0, 1, 2], [1, 2, 3]])
        edges = st.getEdgeList().toVector()
        reg = EdgeModeRegistry.fromSpacetime(st)
        self.assertEqual(reg.modeCount(), len(edges))
        registered = {
            frozenset({reg.record(m).vertexA, reg.record(m).vertexB})
            for m in range(reg.modeCount())
        }
        self.assertEqual(
            registered,
            {frozenset({e.getSource().getId(), e.getTarget().getId()})
             for e in edges})

    def test_stored_direction_is_the_edge_orientation(self):
        # Registered on the edge's own source -> target direction, so the sign
        # against the canonical min -> max direction stays derivable.
        st = self._spacetime([[0, 1, 2]])
        reg = EdgeModeRegistry.fromSpacetime(st)
        for m in range(reg.modeCount()):
            rec = reg.record(m)
            self.assertEqual(rec.orientationSign, +1)
            expected = +1 if rec.vertexA < rec.vertexB else -1
            self.assertEqual(reg.canonicalOrientationSign(m), expected)

    def test_canonical_order_is_the_endpoint_sort_under_one_lineage(self):
        st = self._spacetime([[0, 1, 2], [1, 2, 3]])
        reg = EdgeModeRegistry.fromSpacetime(st)
        pairs = [
            (min(reg.record(m).vertexA, reg.record(m).vertexB),
             max(reg.record(m).vertexA, reg.record(m).vertexB))
            for m in reg.canonicalModeOrder()
        ]
        self.assertEqual(pairs, sorted(pairs))

    def test_the_lineage_key_is_honoured(self):
        st = self._spacetime([[0, 1, 2]])
        reg = EdgeModeRegistry.fromSpacetime(st, "component/7")
        for m in range(reg.modeCount()):
            self.assertEqual(reg.record(m).lineageKey, "component/7")

    def test_the_registry_ignores_the_geometry_entirely(self):
        # It stores incidence and lineage only -- never a length, never a
        # connection phase. Rewriting both must not move a single record.
        st = self._spacetime([[0, 1, 2], [1, 2, 3]])
        before = EdgeModeRegistry.fromSpacetime(st)
        snapshot = [(before.record(m).vertexA, before.record(m).vertexB,
                     before.record(m).orientationSign,
                     before.record(m).lineageKey)
                    for m in before.canonicalModeOrder()]
        for k, e in enumerate(st.getEdgeList().toVector()):
            e.setLength(complex(0.5 + k, -0.25 * k))
            e.setPhase(complex(0.3 * k, 1.1 - k))
        after = EdgeModeRegistry.fromSpacetime(st)
        self.assertEqual(
            [(after.record(m).vertexA, after.record(m).vertexB,
              after.record(m).orientationSign, after.record(m).lineageKey)
             for m in after.canonicalModeOrder()],
            snapshot)

    def test_the_carrier_dimension_is_two_to_the_edge_count(self):
        # The whole point: F(h_K) over the per-edge modes.
        st = self._spacetime([[0, 1, 2]])
        reg = EdgeModeRegistry.fromSpacetime(st)
        self.assertEqual(reg.modeCount(), 3)
        self.assertEqual(2 ** reg.modeCount(),
                         sum(comb(reg.modeCount(), n)
                             for n in range(reg.modeCount() + 1)))

    def test_an_empty_complex_gives_an_empty_registry(self):
        st = self._spacetime([])
        self.assertEqual(EdgeModeRegistry.fromSpacetime(st).modeCount(), 0)


# ─── relabeling invariance: the full parity pipeline ───────────────────────

@unittest.skipUnless(HAVE_QUANTUM, "tessera built without the quantum subsystem")
class TestRelabelingInvariance(unittest.TestCase):
    """Random vertex relabelings preserve all physical amplitudes after
    applying the induced permutation parity (ticket acceptance)."""

    def build_registry(self) -> "EdgeModeRegistry":
        reg = EdgeModeRegistry()
        edges = [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 5)]
        for a, b in edges:
            reg.addEdge(a, b, +1, "root/0")
        return reg

    def test_relabeling_preserves_amplitudes_with_parity(self) -> None:
        reg = self.build_registry()
        m = reg.modeCount()
        alg = ExteriorAlgebra(m)
        rng = np.random.default_rng(59)
        for trial in range(5):
            new_ids = rng.permutation(100)[:6]  # scrambled fresh vertex ids
            vertex_map = {old: int(new_ids[old]) for old in range(6)}
            relabeled = reg.relabeled(vertex_map)
            perm = EdgeModeRegistry.orderPermutation(reg, relabeled)

            # The permutation really is position_before -> position_after.
            pos_before = reg.compilationPositions()
            pos_after = relabeled.compilationPositions()
            for mode_id in range(m):
                self.assertEqual(perm[pos_before[mode_id]],
                                 pos_after[mode_id])

            u = dense(alg.modePermutationMatrixCOO(perm))
            eye = np.eye(2**m, dtype=complex)
            np.testing.assert_array_equal(u.conj().T @ u, eye)

            # Column structure: U|b> = parity(b) |perm(b)> for EVERY basis
            # state — amplitudes are preserved after applying the parity.
            for idx in range(2**m):
                bits = OccupationBitset.fromIndex(m, idx)
                target = bits.permuted(perm).toIndex()
                parity = bits.permutationParity(perm)
                expected = np.zeros(2**m, dtype=complex)
                expected[target] = parity
                np.testing.assert_array_equal(u[:, idx], expected)

            # U intertwines the CAR generators: U a_i+ U+ = a_perm(i)+.
            for i in range(m):
                np.testing.assert_array_equal(
                    u @ dense(alg.creationMatrixCOO(i)) @ u.conj().T,
                    dense(alg.creationMatrixCOO(perm[i])),
                )

            # Physical amplitudes: transported states and transported
            # observables give identical numbers.
            pi = np.zeros((m, m), dtype=complex)
            for i in range(m):
                pi[perm[i], i] = 1.0
            vs = [
                rng.standard_normal(m) + 1j * rng.standard_normal(m)
                for _ in range(3)
            ]
            psi = alg.wedge(vs)
            psi_t = alg.wedge([pi @ v for v in vs])
            np.testing.assert_allclose(psi_t, u @ psi, atol=1e-13)

            l = rng.standard_normal((m, m)) + 1j * rng.standard_normal((m, m))
            l = l + l.conj().T
            amp = np.vdot(psi, dense(alg.dGammaCOO(l)) @ psi)
            amp_t = np.vdot(
                psi_t, dense(alg.dGammaCOO(pi @ l @ pi.conj().T)) @ psi_t
            )
            self.assertAlmostEqual(amp.real, amp_t.real, delta=1e-11)
            self.assertAlmostEqual(amp.imag, amp_t.imag, delta=1e-11)

    def test_relabeling_validation(self) -> None:
        reg = self.build_registry()
        with self.assertRaises(ValueError):
            reg.relabeled({0: 10})  # missing vertices
        with self.assertRaises(ValueError):
            reg.relabeled({v: 7 for v in range(6)})  # not injective
        other = EdgeModeRegistry()
        other.addEdge(0, 1, +1, "x")
        with self.assertRaises(ValueError):
            EdgeModeRegistry.orderPermutation(reg, other)


if __name__ == "__main__":
    unittest.main()
