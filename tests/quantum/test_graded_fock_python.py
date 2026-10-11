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
        self.assertEqual(OccupationBitset(0).chunk_count(), 0)
        self.assertEqual(OccupationBitset(1).chunk_count(), 1)
        self.assertEqual(OccupationBitset(64).chunk_count(), 1)
        self.assertEqual(OccupationBitset(65).chunk_count(), 2)
        self.assertEqual(OccupationBitset(200).chunk_count(), 4)

    def test_prefix_popcount_matches_reference_across_chunks(self) -> None:
        rng = np.random.default_rng(7)
        n_modes = 200
        occupied = sorted(rng.choice(n_modes, size=60, replace=False).tolist())
        b = OccupationBitset.from_occupied_modes(n_modes, occupied)
        self.assertEqual(b.count(), 60)
        self.assertEqual(b.occupied_modes(), occupied)
        for probe in [0, 1, 63, 64, 65, 127, 128, 129, 191, 199, 200]:
            expected = sum(1 for m in occupied if m < probe)
            self.assertEqual(b.prefix_popcount(probe), expected)

    def test_creation_annihilation_signs_across_chunk_boundaries(self) -> None:
        n_modes = 200
        b = OccupationBitset.from_occupied_modes(n_modes, [0, 63, 64, 128])
        # prefix below 65 = {0, 63, 64} -> odd -> sign -1.
        self.assertEqual(b.apply_creation(65), -1)
        self.assertTrue(b.test(65))
        # Pauli exclusion: 0 and state unchanged.
        chunks_before = b.chunks()
        self.assertEqual(b.apply_creation(65), 0)
        self.assertEqual(b.chunks(), chunks_before)
        # annihilate it again: same prefix -> same sign.
        self.assertEqual(b.apply_annihilation(65), -1)
        self.assertFalse(b.test(65))
        # annihilating an empty mode is 0.
        self.assertEqual(b.apply_annihilation(65), 0)
        # parity flips with each successful creation.
        self.assertEqual(b.parity(), +1)  # 4 occupied
        b.apply_creation(199)
        self.assertEqual(b.parity(), -1)

    def test_index_round_trip_and_validation(self) -> None:
        b = OccupationBitset.from_index(5, 0b10110)
        self.assertEqual(b.occupied_modes(), [1, 2, 4])
        self.assertEqual(b.to_index(), 0b10110)
        with self.assertRaises(ValueError):
            OccupationBitset.from_index(3, 8)  # index >= 2^3
        with self.assertRaises(ValueError):
            OccupationBitset.from_index(65, 0)  # too many modes for an index
        with self.assertRaises(ValueError):
            OccupationBitset.from_occupied_modes(4, [1, 1])  # duplicate
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
            b = OccupationBitset.from_occupied_modes(n_modes, occupied)
            images = [perm[m] for m in occupied]
            self.assertEqual(b.permutation_parity(perm), inversion_parity(images))
            self.assertEqual(sorted(b.permuted(perm).occupied_modes()),
                             sorted(images))
        with self.assertRaises(ValueError):
            OccupationBitset(3).permutation_parity([0, 0, 1])  # not a bijection

    def test_correct_at_large_mode_count(self) -> None:
        """Data-structure correctness far above the machine-word threshold."""
        n_modes = 4096
        b = OccupationBitset(n_modes)
        self.assertEqual(b.chunk_count(), 64)
        occupied = list(range(0, n_modes, 97))
        for m in occupied:
            b.set(m)
        self.assertEqual(b.count(), len(occupied))
        for probe in (64, 970, 2048, 4095, 4096):
            self.assertEqual(
                b.prefix_popcount(probe), sum(1 for m in occupied if m < probe)
            )
        # creation sign at the top of the range: 42 occupied below 4090.
        below = sum(1 for m in occupied if m < 4090)
        self.assertEqual(b.apply_creation(4090), (-1) ** below)


# ─── CAR: exhaustive three-mode fixtures ───────────────────────────────────

@unittest.skipUnless(HAVE_QUANTUM, "tessera built without the quantum subsystem")
class TestCARThreeModesExhaustive(unittest.TestCase):
    """{a_i, a_j} = 0, {a_i+, a_j+} = 0, {a_i, a_j+} = delta_ij — exhaustive
    at M = 3, exact (integer identities), against independent JW references,
    on every basis state and under every induced mode reordering."""

    M = 3

    def setUp(self) -> None:
        self.alg = ExteriorAlgebra(self.M)
        self.a = [dense(self.alg.annihilation_matrix_coo(i)) for i in range(self.M)]
        self.adag = [dense(self.alg.creation_matrix_coo(i)) for i in range(self.M)]

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
                    (self.adag[mode], "apply_creation"),
                    (self.a[mode], "apply_annihilation"),
                ):
                    bits = OccupationBitset.from_index(self.M, idx)
                    sign = getattr(bits, apply_name)(mode)
                    column = matrix[:, idx]
                    if sign == 0:
                        np.testing.assert_array_equal(column, 0)
                        self.assertEqual(bits.to_index(), idx)  # unchanged
                    else:
                        expected = np.zeros(2**self.M, dtype=complex)
                        expected[bits.to_index()] = sign
                        np.testing.assert_array_equal(column, expected)

    def test_car_under_every_induced_mode_reordering(self) -> None:
        eye = np.eye(2**self.M, dtype=complex)
        zero = np.zeros_like(eye)
        for perm in itertools.permutations(range(self.M)):
            u = dense(self.alg.mode_permutation_matrix_coo(list(perm)))
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
        n_total = dense(self.alg.total_number_matrix_coo())
        parity = dense(self.alg.parity_matrix_coo())
        n_sum = sum(dense(self.alg.number_matrix_coo(i)) for i in range(self.M))
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
            self.assertEqual(alg.fock_dimension(), 2**m)
            self.assertEqual(len(alg.vacuum_state()), 2**m)

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
            expected = alg.basis_state(
                OccupationBitset.from_occupied_modes(m, list(occupied))
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
        a_w = dense(alg.annihilation_operator_coo(w))
        adag_v = dense(alg.creation_operator_coo(v))
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
            dense(alg.subset_sector_projector_coo(subset, n)) for n in range(4)
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
            n_subset = sum(dense(alg.number_matrix_coo(i)) for i in subset)
            np.testing.assert_array_equal(n_subset @ p, n * p)

    def test_total_sector_projectors(self) -> None:
        m = 4
        alg = ExteriorAlgebra(m)
        eye = np.eye(2**m, dtype=complex)
        total = sum(dense(alg.sector_projector_coo(n)) for n in range(m + 1))
        np.testing.assert_array_equal(total, eye)
        # N = sum_n n * P_n exactly.
        n_from_sectors = sum(
            n * dense(alg.sector_projector_coo(n)) for n in range(m + 1)
        )
        np.testing.assert_array_equal(
            n_from_sectors, dense(alg.total_number_matrix_coo())
        )
        with self.assertRaises(ValueError):
            alg.subset_sector_projector_coo([0, 0, 1], 1)  # duplicate mode


# ─── graded swap: elementary parity combinations ───────────────────────────

@unittest.skipUnless(HAVE_QUANTUM, "tessera built without the quantum subsystem")
class TestGradedSwap(unittest.TestCase):
    def test_odd_odd_is_minus_one_all_others_plus_one(self) -> None:
        m_a, m_b = 2, 2
        f = FockDirectSum(m_a, m_b)
        s = dense(f.graded_swap_matrix_coo())
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
        s_ab = dense(f_ab.graded_swap_matrix_coo())
        s_ba = dense(f_ba.graded_swap_matrix_coo())
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
        self.assertEqual(prod.max_degree(), 2)
        self.assertEqual(
            [prod.chain_dimension(n) for n in range(3)], [9, 18, 9]
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
                np.linalg.eigvalsh(prod.factor_laplacian_a(p)) for p in range(max_p + 1)
            ]
            spec_b = [
                np.linalg.eigvalsh(prod.factor_laplacian_b(q)) for q in range(max_q + 1)
            ]
            for n in range(prod.max_degree() + 1):
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
        b0 = prod.chain_dimension(0) - np.linalg.matrix_rank(d1)
        b1 = (
            prod.chain_dimension(1)
            - np.linalg.matrix_rank(d1)
            - np.linalg.matrix_rank(d2)
        )
        b2 = prod.chain_dimension(2) - np.linalg.matrix_rank(d2)
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
        self.joint = self.f.joint_algebra()
        self.left = self.f.left_algebra()
        self.right = self.f.right_algebra()

    def test_direct_sums_become_graded_tensor_products(self) -> None:
        """Joint CAR generators equal the graded lifts EXACTLY: left lifts
        are X x 1; right ODD lifts carry the (-1)^N_A Koszul twist."""
        for i in range(self.M_A):
            lifted = dense(
                self.f.lift_left_coo(dense(self.left.creation_matrix_coo(i)))
            )
            np.testing.assert_array_equal(
                dense(self.joint.creation_matrix_coo(i)), lifted
            )
        for j in range(self.M_B):
            lifted = dense(
                self.f.lift_right_coo(
                    dense(self.right.creation_matrix_coo(j)), True
                )
            )
            np.testing.assert_array_equal(
                dense(self.joint.creation_matrix_coo(self.M_A + j)), lifted
            )
        # An even right operator lifts without the twist.
        n_b = dense(self.right.number_matrix_coo(1))
        np.testing.assert_array_equal(
            dense(self.f.lift_right_coo(n_b, False)),
            dense(self.joint.number_matrix_coo(self.M_A + 1)),
        )

    def test_wrong_koszul_twist_fails(self) -> None:
        """Negative control: lifting an ODD operator without the parity
        twist does NOT reproduce the joint generator."""
        a_b0 = dense(self.right.creation_matrix_coo(0))
        wrong = dense(self.f.lift_right_coo(a_b0, False))
        right_gen = dense(self.joint.creation_matrix_coo(self.M_A))
        self.assertTrue(np.any(wrong != right_gen))

    def test_block_diagonal_dgamma_is_sum_of_lifts_integer_exact(self) -> None:
        l_a = np.array([[1.0, 2.0], [2.0, -1.0]], dtype=complex)
        l_b = np.array([[3.0, 1.0], [1.0, 0.0]], dtype=complex)
        zero_c = np.zeros((self.M_A, self.M_B), dtype=complex)
        got = dense(self.f.d_gamma_block_coo(l_a, l_b, zero_c))
        expected = dense(
            self.f.lift_left_coo(dense(self.left.d_gamma_coo(l_a)))
        ) + dense(
            self.f.lift_right_coo(dense(self.right.d_gamma_coo(l_b)), False)
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
        l_full = np.asarray(self.f.assemble_block_one_particle(l_a, l_b, c))
        # Assembly itself is exact.
        np.testing.assert_array_equal(l_full[:2, :2], l_a)
        np.testing.assert_array_equal(l_full[2:, 2:], l_b)
        np.testing.assert_array_equal(l_full[:2, 2:], c)
        np.testing.assert_array_equal(l_full[2:, :2], c.conj().T)

        got = dense(self.f.d_gamma_block_coo(l_a, l_b, c))
        a_ops = [jw_annihilation(i, m) for i in range(m)]
        reference = sum(
            l_full[i, j] * (a_ops[i].conj().T @ a_ops[j])
            for i in range(m)
            for j in range(m)
        )
        np.testing.assert_allclose(got, reference, atol=1e-13)
        # Hermitian L gives a Hermitian d_gamma(L).
        np.testing.assert_allclose(got, got.conj().T, atol=1e-13)

    def test_coupling_blocks_become_hopping_terms(self) -> None:
        rng = np.random.default_rng(37)
        m = self.M_A + self.M_B
        c = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
        zero2 = np.zeros((2, 2), dtype=complex)
        hopping = dense(self.f.d_gamma_block_coo(zero2, zero2, c))
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
        many_body = np.sort(np.linalg.eigvalsh(dense(alg.d_gamma_coo(l))))
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
        dg = dense(alg.d_gamma_coo(l))
        n = dense(alg.total_number_matrix_coo())
        np.testing.assert_allclose(dg @ n - n @ dg, 0 * dg, atol=1e-13)


@unittest.skipUnless(HAVE_QUANTUM, "tessera built without the quantum subsystem")
class TestDirectedCoupling(unittest.TestCase):
    """Whitepaper Section 6: d_gamma(C) = sum (C_AB)_ij eps_A,i iota^j_B +
    sum (C_BA)_ij eps_B,i iota^j_A, with NO Hermitian-conjugate relation
    between the two directed blocks assumed."""

    M_A = 2
    M_B = 3

    def setUp(self) -> None:
        self.f = FockDirectSum(self.M_A, self.M_B)
        self.m = self.M_A + self.M_B
        self.a_ops = [jw_annihilation(i, self.m) for i in range(self.m)]
        rng = np.random.default_rng(47)

        def cplx(*shape):
            return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)

        # Complex-symmetric diagonal blocks and two INDEPENDENT couplings.
        self.l_a = cplx(self.M_A, self.M_A)
        self.l_a = self.l_a + self.l_a.T
        self.l_b = cplx(self.M_B, self.M_B)
        self.l_b = self.l_b + self.l_b.T
        self.c_ab = cplx(self.M_A, self.M_B)
        self.c_ba = cplx(self.M_B, self.M_A)

    def test_assembly_places_both_directed_blocks_as_given(self) -> None:
        full = np.asarray(FockDirectSum.assemble_block_one_particle(
            self.l_a, self.l_b, self.c_ab, self.c_ba))
        a, b = self.M_A, self.M_B
        np.testing.assert_array_equal(full[:a, :a], self.l_a)
        np.testing.assert_array_equal(full[a:, a:], self.l_b)
        np.testing.assert_array_equal(full[:a, a:], self.c_ab)
        np.testing.assert_array_equal(full[a:, :a], self.c_ba)
        # Nothing forces C_BA = C_AB^dagger.
        self.assertGreater(np.abs(self.c_ba - self.c_ab.conj().T).max(), 0.1)

    def test_directed_dgamma_matches_the_dense_reference(self) -> None:
        got = dense(self.f.d_gamma_block_coo(
            self.l_a, self.l_b, self.c_ab, self.c_ba))
        a = self.M_A
        adag = [op.conj().T for op in self.a_ops]
        reference = sum(
            self.l_a[i, j] * adag[i] @ self.a_ops[j]
            for i in range(a) for j in range(a))
        reference = reference + sum(
            self.l_b[i, j] * adag[a + i] @ self.a_ops[a + j]
            for i in range(self.M_B) for j in range(self.M_B))
        # eps_A,i iota^j_B: hopping B -> A by C_AB ...
        reference = reference + sum(
            self.c_ab[i, j] * adag[i] @ self.a_ops[a + j]
            for i in range(a) for j in range(self.M_B))
        # ... and eps_B,i iota^j_A: hopping A -> B by C_BA, independently.
        reference = reference + sum(
            self.c_ba[i, j] * adag[a + i] @ self.a_ops[j]
            for i in range(self.M_B) for j in range(a))
        np.testing.assert_allclose(got, reference, rtol=0, atol=1e-12)
        # Complex-symmetric blocks with independent couplings: dGamma is
        # neither Hermitian nor forced to be.
        self.assertGreater(np.abs(got - got.conj().T).max(), 0.1)

    def test_one_directed_block_alone_hops_one_way(self) -> None:
        zero_a = np.zeros((self.M_A, self.M_A), dtype=complex)
        zero_b = np.zeros((self.M_B, self.M_B), dtype=complex)
        only_ab = dense(self.f.d_gamma_block_coo(
            zero_a, zero_b, self.c_ab,
            np.zeros((self.M_B, self.M_A), dtype=complex)))
        a = self.M_A
        adag = [op.conj().T for op in self.a_ops]
        reference = sum(
            self.c_ab[i, j] * adag[i] @ self.a_ops[a + j]
            for i in range(a) for j in range(self.M_B))
        np.testing.assert_allclose(only_ab, reference, rtol=0, atol=1e-13)
        # No hidden h.c.: every term raises N_A by exactly one, [N_A, X] = X,
        # so the A -> B amplitudes are exactly absent.
        n_a = sum(adag[i] @ self.a_ops[i] for i in range(a))
        commutator = n_a @ only_ab - only_ab @ n_a
        np.testing.assert_allclose(commutator, only_ab, rtol=0, atol=1e-13)

    def test_star_structure_form_is_the_special_case(self) -> None:
        c = self.c_ab
        special = dense(self.f.d_gamma_block_coo(self.l_a, self.l_b, c))
        general = dense(self.f.d_gamma_block_coo(
            self.l_a, self.l_b, c, c.conj().T))
        np.testing.assert_array_equal(special, general)
        np.testing.assert_array_equal(
            np.asarray(FockDirectSum.assemble_block_one_particle(
                self.l_a, self.l_b, c)),
            np.asarray(FockDirectSum.assemble_block_one_particle(
                self.l_a, self.l_b, c, c.conj().T)))

    def test_free_spectrum_is_the_subset_sums_of_the_complex_block(self) -> None:
        full = np.asarray(FockDirectSum.assemble_block_one_particle(
            self.l_a, self.l_b, self.c_ab, self.c_ba))
        one_particle = np.linalg.eigvals(full)
        many_body = np.linalg.eigvals(dense(self.f.d_gamma_block_coo(
            self.l_a, self.l_b, self.c_ab, self.c_ba)))
        subset_sums = [
            sum(one_particle[list(s)]) if s else 0j
            for k in range(self.m + 1)
            for s in itertools.combinations(range(self.m), k)
        ]
        # Match every many-body eigenvalue to a distinct subset sum.
        remaining = list(subset_sums)
        for z in many_body:
            k = int(np.argmin([abs(z - w) for w in remaining]))
            self.assertLess(abs(z - remaining[k]), 1e-9)
            remaining.pop(k)

    def test_directed_shapes_are_validated(self) -> None:
        with self.assertRaises(ValueError):
            FockDirectSum.assemble_block_one_particle(
                self.l_a, self.l_b, self.c_ab, self.c_ab)  # C_BA is M_B x M_A
        with self.assertRaises(ValueError):
            self.f.d_gamma_block_coo(self.l_a, self.l_b, self.c_ba, self.c_ba)


# ─── pair creation ─────────────────────────────────────────────────────────

@unittest.skipUnless(HAVE_QUANTUM, "tessera built without the quantum subsystem")
class TestPairCreation(unittest.TestCase):
    def test_pair_creation_raises_n_by_two_and_preserves_parity(self) -> None:
        m = 4
        alg = ExteriorAlgebra(m)
        adag = [dense(alg.creation_matrix_coo(i)) for i in range(m)]
        n = dense(alg.total_number_matrix_coo())
        p = dense(alg.parity_matrix_coo())
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
        reg.add_edge(4, 3, +1, "root/1")
        reg.add_edge(0, 1, +1, "root/0")
        reg.add_edge(2, 0, -1, "root/0")
        reg.add_edge(5, 4, -1, "root/1")
        reg.add_edge(1, 2, +1, "root/0")
        reg.add_edge(3, 5, +1, "root/1")  # not a path edge; still fine
        return reg

    def test_canonical_order_is_lineage_then_vertex_pair(self) -> None:
        reg = self.build_registry()
        order = reg.canonical_mode_order()
        keyed = [
            (
                reg.record(mode_id).lineage_key,
                min(reg.record(mode_id).vertex_a, reg.record(mode_id).vertex_b),
                max(reg.record(mode_id).vertex_a, reg.record(mode_id).vertex_b),
            )
            for mode_id in order
        ]
        self.assertEqual(keyed, sorted(keyed))
        # positions invert the order.
        positions = reg.compilation_positions()
        for pos, mode_id in enumerate(order):
            self.assertEqual(positions[mode_id], pos)

    def test_storage_reversal_changes_nothing_observable(self) -> None:
        reg = self.build_registry()
        order_before = reg.canonical_mode_order()
        signs_before = [
            reg.canonical_orientation_sign(m) for m in range(reg.mode_count())
        ]
        for mode_id in (0, 2, 5):
            reg.reverse_stored_direction(mode_id)
        self.assertEqual(reg.canonical_mode_order(), order_before)
        self.assertEqual(
            [reg.canonical_orientation_sign(m) for m in range(reg.mode_count())],
            signs_before,
        )
        # But the stored record did flip.
        rec = reg.record(0)
        self.assertEqual((rec.vertex_a, rec.vertex_b, rec.orientation_sign),
                         (3, 4, -1))

    def test_flip_orientation_flips_the_physical_sign(self) -> None:
        reg = self.build_registry()
        before = reg.canonical_orientation_sign(1)
        reg.flip_orientation(1)
        self.assertEqual(reg.canonical_orientation_sign(1), -before)
        self.assertEqual(reg.canonical_mode_order(),
                         self.build_registry().canonical_mode_order())

    def test_registration_validation(self) -> None:
        reg = self.build_registry()
        with self.assertRaises(ValueError):
            reg.add_edge(1, 0, +1, "root/0")  # duplicate unordered pair
        with self.assertRaises(ValueError):
            reg.add_edge(7, 7, +1, "root/2")  # self-loop
        with self.assertRaises(ValueError):
            reg.add_edge(8, 9, 0, "root/2")  # invalid sign
        with self.assertRaises(ValueError):
            reg.record(99)

    def test_reversal_invariant_amplitudes_via_canonical_signs(self) -> None:
        """One-particle data read through canonicalOrientationSign is
        invariant under storage reversals and flips sign under a physical
        orientation flip — the documented reorientation convention."""
        rng = np.random.default_rng(53)
        reg = self.build_registry()
        m = reg.mode_count()
        alg = ExteriorAlgebra(m)
        coeffs = rng.standard_normal(m) + 1j * rng.standard_normal(m)

        def physical_vector(registry) -> np.ndarray:
            positions = registry.compilation_positions()
            v = np.zeros(m, dtype=complex)
            for mode_id in range(m):
                v[positions[mode_id]] = (
                    coeffs[mode_id]
                    * registry.canonical_orientation_sign(mode_id)
                )
            return v

        baseline_vector = physical_vector(reg)
        baseline_wedge = alg.wedge([baseline_vector])
        # Storage reversals change nothing: identical one-particle data,
        # identical amplitudes, exactly.
        for mode_id in (1, 3, 4):
            reg.reverse_stored_direction(mode_id)
        np.testing.assert_array_equal(physical_vector(reg), baseline_vector)
        np.testing.assert_array_equal(
            alg.wedge([physical_vector(reg)]), baseline_wedge
        )
        # A physical orientation flip multiplies exactly that mode's
        # one-particle component by -1 and leaves the others alone.
        reg.flip_orientation(2)
        flipped = physical_vector(reg)
        position = reg.compilation_positions()[2]
        expected = baseline_vector.copy()
        expected[position] = -expected[position]
        np.testing.assert_array_equal(flipped, expected)


# ─── the per-edge carrier the ontology names ───────────────────────────────

@unittest.skipUnless(HAVE_QUANTUM, "tessera built without the quantum subsystem")
class TestRegistryFromSpacetime(unittest.TestCase):
    """`EdgeModeRegistry.from_spacetime` builds the ontology's carrier: one
    two-level occupation mode per EDGE, h_K = span{|e> : e in K1} (#804).
    Before it existed, every carrier in the tree was built over a band's
    degree-k cells, so at degree two the one-particle modes were triangles."""

    @staticmethod
    def _spacetime(cells):
        # `import tessera` then attribute access: the submodule is not
        # importable directly.
        import tessera
        return tessera.spacetime.Spacetime.from_vertex_tuples(2, cells)

    def test_one_mode_per_edge(self):
        st = self._spacetime([[0, 1, 2], [1, 2, 3]])
        edges = st.get_edge_list().to_vector()
        reg = EdgeModeRegistry.from_spacetime(st)
        self.assertEqual(reg.mode_count(), len(edges))
        registered = {
            frozenset({reg.record(m).vertex_a, reg.record(m).vertex_b})
            for m in range(reg.mode_count())
        }
        self.assertEqual(
            registered,
            {frozenset({e.get_source().get_id(), e.get_target().get_id()})
             for e in edges})

    def test_stored_direction_is_the_edge_orientation(self):
        # Registered on the edge's own source -> target direction, so the sign
        # against the canonical min -> max direction stays derivable.
        st = self._spacetime([[0, 1, 2]])
        reg = EdgeModeRegistry.from_spacetime(st)
        for m in range(reg.mode_count()):
            rec = reg.record(m)
            self.assertEqual(rec.orientation_sign, +1)
            expected = +1 if rec.vertex_a < rec.vertex_b else -1
            self.assertEqual(reg.canonical_orientation_sign(m), expected)

    def test_canonical_order_is_the_endpoint_sort_under_one_lineage(self):
        st = self._spacetime([[0, 1, 2], [1, 2, 3]])
        reg = EdgeModeRegistry.from_spacetime(st)
        pairs = [
            (min(reg.record(m).vertex_a, reg.record(m).vertex_b),
             max(reg.record(m).vertex_a, reg.record(m).vertex_b))
            for m in reg.canonical_mode_order()
        ]
        self.assertEqual(pairs, sorted(pairs))

    def test_the_lineage_key_is_honoured(self):
        st = self._spacetime([[0, 1, 2]])
        reg = EdgeModeRegistry.from_spacetime(st, "component/7")
        for m in range(reg.mode_count()):
            self.assertEqual(reg.record(m).lineage_key, "component/7")

    def test_the_registry_ignores_the_geometry_entirely(self):
        # It stores incidence and lineage only -- never a length, never a
        # connection phase. Rewriting both must not move a single record.
        st = self._spacetime([[0, 1, 2], [1, 2, 3]])
        before = EdgeModeRegistry.from_spacetime(st)
        snapshot = [(before.record(m).vertex_a, before.record(m).vertex_b,
                     before.record(m).orientation_sign,
                     before.record(m).lineage_key)
                    for m in before.canonical_mode_order()]
        for k, e in enumerate(st.get_edge_list().to_vector()):
            e.set_length(complex(0.5 + k, -0.25 * k))
            e.set_phase(complex(0.3 * k, 1.1 - k))
        after = EdgeModeRegistry.from_spacetime(st)
        self.assertEqual(
            [(after.record(m).vertex_a, after.record(m).vertex_b,
              after.record(m).orientation_sign, after.record(m).lineage_key)
             for m in after.canonical_mode_order()],
            snapshot)

    def test_the_carrier_dimension_is_two_to_the_edge_count(self):
        # The whole point: F(h_K) over the per-edge modes.
        st = self._spacetime([[0, 1, 2]])
        reg = EdgeModeRegistry.from_spacetime(st)
        self.assertEqual(reg.mode_count(), 3)
        self.assertEqual(2 ** reg.mode_count(),
                         sum(comb(reg.mode_count(), n)
                             for n in range(reg.mode_count() + 1)))

    def test_an_empty_complex_gives_an_empty_registry(self):
        st = self._spacetime([])
        self.assertEqual(EdgeModeRegistry.from_spacetime(st).mode_count(), 0)


# ─── relabeling invariance: the full parity pipeline ───────────────────────

@unittest.skipUnless(HAVE_QUANTUM, "tessera built without the quantum subsystem")
class TestRelabelingInvariance(unittest.TestCase):
    """Random vertex relabelings preserve all physical amplitudes after
    applying the induced permutation parity (ticket acceptance)."""

    def build_registry(self) -> "EdgeModeRegistry":
        reg = EdgeModeRegistry()
        edges = [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 5)]
        for a, b in edges:
            reg.add_edge(a, b, +1, "root/0")
        return reg

    def test_relabeling_preserves_amplitudes_with_parity(self) -> None:
        reg = self.build_registry()
        m = reg.mode_count()
        alg = ExteriorAlgebra(m)
        rng = np.random.default_rng(59)
        for trial in range(5):
            new_ids = rng.permutation(100)[:6]  # scrambled fresh vertex ids
            vertex_map = {old: int(new_ids[old]) for old in range(6)}
            relabeled = reg.relabeled(vertex_map)
            perm = EdgeModeRegistry.order_permutation(reg, relabeled)

            # The permutation really is position_before -> position_after.
            pos_before = reg.compilation_positions()
            pos_after = relabeled.compilation_positions()
            for mode_id in range(m):
                self.assertEqual(perm[pos_before[mode_id]],
                                 pos_after[mode_id])

            u = dense(alg.mode_permutation_matrix_coo(perm))
            eye = np.eye(2**m, dtype=complex)
            np.testing.assert_array_equal(u.conj().T @ u, eye)

            # Column structure: U|b> = parity(b) |perm(b)> for EVERY basis
            # state — amplitudes are preserved after applying the parity.
            for idx in range(2**m):
                bits = OccupationBitset.from_index(m, idx)
                target = bits.permuted(perm).to_index()
                parity = bits.permutation_parity(perm)
                expected = np.zeros(2**m, dtype=complex)
                expected[target] = parity
                np.testing.assert_array_equal(u[:, idx], expected)

            # U intertwines the CAR generators: U a_i+ U+ = a_perm(i)+.
            for i in range(m):
                np.testing.assert_array_equal(
                    u @ dense(alg.creation_matrix_coo(i)) @ u.conj().T,
                    dense(alg.creation_matrix_coo(perm[i])),
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
            amp = np.vdot(psi, dense(alg.d_gamma_coo(l)) @ psi)
            amp_t = np.vdot(
                psi_t, dense(alg.d_gamma_coo(pi @ l @ pi.conj().T)) @ psi_t
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
        other.add_edge(0, 1, +1, "x")
        with self.assertRaises(ValueError):
            EdgeModeRegistry.order_permutation(reg, other)


if __name__ == "__main__":
    unittest.main()
