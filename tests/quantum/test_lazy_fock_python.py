"""Acceptance and property tests for the lazy graded Fock oracle and
boundary carrier (issue #771): LazyFockEngine / LazyFockState.

Acceptance coverage (ticket #771; the #780 covariance-layer
cross-validation is deferred to post-merge integration and replaced here
by dense references built from the merged #766 machinery):

* dim Lambda* C^M = 2^M on enumerated small fixtures;
* a one-particle W/color state and a nonseparable proton-spin oracle are
  represented without product-state approximation (J^2 reads 3/4 for the
  mixed-symmetry doublet and 15/4 for the symmetric quartet);
* dGamma direct-sum/hopping fixtures and subset-sum spectra match dense
  Fock references (FockDirectSum.dGammaBlock, OccupationSpectra);
* the vacuum embedding preserves all preexisting amplitudes, and the
  spec-5.7 inductive compatibility read matches an independent dense
  computation;
* different parenthesizations agree after graded associators;
* local operations do not expand disconnected tensor branches (node
  identity of the untouched sibling is preserved), while a crossing
  operation MUST expand the partition;
* exact DAG evaluation matches a dense vector on every crossover fixture;
* truncation mode's observed amplitude error is bounded by its reported
  discarded norm;
* checkpoint/replay reproduces the expression and the amplitudes.

The dense references are INDEPENDENT numpy Jordan-Wigner constructions
(kron chains — the test_graded_fock_python.py pattern), including an
independent wedge-word oracle for graded tensors over interleaved mode
sets and a JW polynomial-extension oracle for local maps; they are never
re-derivations through the bindings under test.

Exactness bar: sign rules, dimension identities, Gram/Slater determinants
and the associator are algebraic identities — compared at double
round-off (1e-12..1e-15); truncation-mode claims are inequalities against
the reported bound.

Skips cleanly when tessera was built without the quantum subsystem.
"""

from __future__ import annotations

import itertools
import math
import unittest

import numpy as np

import tessera

cob = tessera.cobordism

try:
    from tessera.quantum import (
        EdgeModeRegistry,
        ExteriorAlgebra,
        FockDirectSum,
        LazyFockEngine,
        LazyNodeKind,
        LazySectorKind,
    )
    HAVE_QUANTUM = True
except ImportError:
    HAVE_QUANTUM = False


# ─── independent dense Jordan-Wigner reference layer ──────────────────────

_S_MINUS = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
_Z = np.diag([1.0, -1.0]).astype(complex)


def jw_annihilation(mode: int, n_modes: int) -> np.ndarray:
    """Dense JW a_mode on the n(b) = sum b_i 2^i basis (mode 0 = LSB)."""
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


def jw_creation(mode: int, n_modes: int) -> np.ndarray:
    return jw_annihilation(mode, n_modes).conj().T


def word_matrix(occupied, n_modes: int) -> np.ndarray:
    """The wedge word a_{i1}† a_{i2}† … a_{ik}† for ascending
    i1 < … < ik — matrix product with a_{i1}† leftmost (applied last)."""
    op = np.eye(2 ** n_modes, dtype=complex)
    for m in sorted(occupied):
        op = op @ jw_creation(m, n_modes)
    return op


def dense_state_from_terms(terms, n_modes: int) -> np.ndarray:
    """Independent wedge-word preparation: sum_b amp(b) word(b) |vac>."""
    vac = np.zeros(2 ** n_modes, dtype=complex)
    vac[0] = 1.0
    out = np.zeros_like(vac)
    for occupied, amp in terms:
        out += amp * (word_matrix(occupied, n_modes) @ vac)
    return out


def dense_tensor_oracle(terms_a, terms_b, n_modes: int) -> np.ndarray:
    """psi_A (x) psi_B := (sum amp_A word_A)(sum amp_B word_B)|vac> —
    the definitional wedge-word product, independent of the engine's
    Koszul-sign formula."""
    vac = np.zeros(2 ** n_modes, dtype=complex)
    vac[0] = 1.0
    op_a = sum(amp * word_matrix(occ, n_modes) for occ, amp in terms_a)
    op_b = sum(amp * word_matrix(occ, n_modes) for occ, amp in terms_b)
    return op_a @ (op_b @ vac)


def vacuum_projector(support, n_modes: int) -> np.ndarray:
    out = np.eye(2 ** n_modes, dtype=complex)
    for m in support:
        out = out @ (np.eye(2 ** n_modes, dtype=complex)
                     - jw_creation(m, n_modes) @ jw_annihilation(m, n_modes))
    return out


def local_op_dense(op_support: np.ndarray, support, n_modes: int) -> np.ndarray:
    """The JW polynomial extension of a support-Fock matrix: O =
    sum_{b',b} O[b',b] word(b') |vac_S><vac_S| word(b)†, built purely from
    dense JW matrices (independent oracle for applyLocalMapDense)."""
    support = sorted(support)
    s = len(support)
    dim = 2 ** n_modes
    proj = vacuum_projector(support, n_modes)
    out = np.zeros((dim, dim), dtype=complex)
    for bp in range(2 ** s):
        occ_bp = [support[k] for k in range(s) if (bp >> k) & 1]
        left = word_matrix(occ_bp, n_modes)
        for b in range(2 ** s):
            v = op_support[bp, b]
            if v == 0:
                continue
            occ_b = [support[k] for k in range(s) if (b >> k) & 1]
            right = word_matrix(occ_b, n_modes).conj().T
            out += v * (left @ proj @ right)
    return out


def dgamma_dense(one_particle: np.ndarray, support, n_modes: int) -> np.ndarray:
    support = sorted(support)
    dim = 2 ** n_modes
    out = np.zeros((dim, dim), dtype=complex)
    for i, gi in enumerate(support):
        for j, gj in enumerate(support):
            v = one_particle[i, j]
            if v == 0:
                continue
            out += v * (jw_creation(gi, n_modes) @ jw_annihilation(gj, n_modes))
    return out


def all_keys(n_modes: int):
    for idx in range(2 ** n_modes):
        yield [m for m in range(n_modes) if (idx >> m) & 1]


def random_terms(rng, modes, n_terms):
    keys = set()
    terms = []
    for _ in range(n_terms):
        k = tuple(sorted(rng.choice(modes, size=rng.integers(0, len(modes) + 1),
                                    replace=False).tolist()))
        if k in keys:
            continue
        keys.add(k)
        terms.append((list(k), complex(rng.normal(), rng.normal())))
    return terms


TOL = 1e-12


# ─── carrier and dimension ─────────────────────────────────────────────────

@unittest.skipUnless(HAVE_QUANTUM, "tessera built without the quantum subsystem")
class TestCarrierDimension(unittest.TestCase):
    """dim Lambda* C^M = 2^M on enumerated small fixtures."""

    def test_stage_dimension_identity(self):
        for m in range(13):
            self.assertEqual(LazyFockEngine.stage_dimension(m), 2 ** m)
        with self.assertRaises(ValueError):
            LazyFockEngine.stage_dimension(64)

    def test_dense_vector_dimension_and_vacuum(self):
        for m in (0, 1, 3, 6):
            eng = LazyFockEngine(m)
            vec = np.asarray(eng.dense_vector(eng.vacuum()))
            self.assertEqual(vec.shape[0], 2 ** m)
            expected = np.zeros(2 ** m, dtype=complex)
            expected[0] = 1.0
            np.testing.assert_allclose(vec, expected, rtol=0, atol=0)

    def test_full_basis_enumeration_matches_exterior_algebra(self):
        m = 3
        eng = LazyFockEngine(m)
        alg = ExteriorAlgebra(m)
        self.assertEqual(alg.fock_dimension(), 2 ** m)
        occupations = list(all_keys(m))
        amps = [complex(k + 1, -k) for k in range(len(occupations))]
        state = eng.occupation_state(list(range(m)), occupations, amps)
        vec = np.asarray(eng.dense_vector(state))
        # 2^M enumerated coordinates, one per exterior basis state.
        for idx, occ in enumerate(occupations):
            self.assertAlmostEqual(vec[idx], amps[idx], delta=0)
            read = eng.amplitude(state, occ)
            self.assertEqual(read.value, amps[idx])
            self.assertEqual(read.discarded_norm, 0.0)

    def test_dense_export_cap(self):
        eng = LazyFockEngine(30)  # beyond kMaxDenseModes = 24
        with self.assertRaises(ValueError):
            eng.dense_vector(eng.vacuum())

    def test_from_registry_mode_count(self):
        reg = EdgeModeRegistry()
        reg.add_edge(0, 1, +1, "root/a")
        reg.add_edge(1, 2, +1, "root/b")
        reg.add_edge(2, 3, +1, "root/a")
        eng = LazyFockEngine.from_registry(reg)
        self.assertEqual(eng.mode_count(), 3)


# ─── one-particle W/color state and nonseparable spin oracle ──────────────

@unittest.skipUnless(HAVE_QUANTUM, "tessera built without the quantum subsystem")
class TestOracleStates(unittest.TestCase):
    """W/color and proton-spin oracle states without product-state
    approximation."""

    def test_w_state_amplitudes_and_norm(self):
        eng = LazyFockEngine(3)
        w = eng.wedge_state([0, 1, 2], np.ones((3, 1), dtype=complex) / np.sqrt(3))
        for m in range(3):
            self.assertAlmostEqual(eng.amplitude(w, [m]).value,
                                   1 / np.sqrt(3), delta=TOL)
        # No two-particle or vacuum contamination.
        self.assertEqual(eng.amplitude(w, []).value, 0.0)
        self.assertEqual(eng.amplitude(w, [0, 1]).value, 0.0)
        self.assertAlmostEqual(eng.norm_squared(w).value, 1.0, delta=TOL)
        self.assertEqual(w.definite_occupation(), 1)
        self.assertEqual(w.definite_parity(), -1)

    def test_w_state_covariance_rank_one_projector(self):
        eng = LazyFockEngine(3)
        v = np.ones((3, 1), dtype=complex) / np.sqrt(3)
        w = eng.wedge_state([0, 1, 2], v)
        read = eng.covariance_matrix(w)
        gamma = np.asarray(read.matrix)
        np.testing.assert_allclose(gamma, v @ v.conj().T, atol=TOL)
        # Pure global state (Gamma^2 = Gamma) with MIXED per-edge
        # marginals (occupation 1/3 each): a derived marginal, not a
        # stored product state.
        np.testing.assert_allclose(gamma @ gamma, gamma, atol=TOL)
        np.testing.assert_allclose(np.diag(gamma), np.full(3, 1 / 3), atol=TOL)
        self.assertTrue(read.certificate.holds())

    def _j2_read(self, eng, chi, quark_modes):
        """<J^2> via dGamma spin operators: ||S_z chi||^2 +
        (||S_- chi||^2 + ||S_+ chi||^2)/2 on a normalized chi."""
        n = len(quark_modes)
        sz = np.zeros((2 * n, 2 * n), dtype=complex)
        sp = np.zeros((2 * n, 2 * n), dtype=complex)
        modes = [m for pair in quark_modes for m in pair]
        for q in range(n):
            up, down = 2 * q, 2 * q + 1  # positions within `modes`
            sz[up, up] = 0.5
            sz[down, down] = -0.5
            sp[up, down] = 1.0
        sm = sp.conj().T
        chi_z = eng.apply_d_gamma(chi, modes, sz)
        chi_p = eng.apply_d_gamma(chi, modes, sp)
        chi_m = eng.apply_d_gamma(chi, modes, sm)
        return (eng.norm_squared(chi_z).value.real
                + 0.5 * (eng.norm_squared(chi_m).value.real
                         + eng.norm_squared(chi_p).value.real))

    def test_proton_spin_oracle_j2_three_quarters(self):
        # Nonseparable mixed-symmetry doublet chi_MS =
        # (2|up up down> - |up down up> - |down up up>)/sqrt(6), stored as
        # ONE global sparse occupation block — no product-state
        # approximation anywhere.
        eng = LazyFockEngine(6)
        quark_modes = [(0, 1), (2, 3), (4, 5)]
        chi = eng.occupation_state(
            list(range(6)),
            [[0, 2, 5], [0, 3, 4], [1, 2, 4]],
            [2 / np.sqrt(6), -1 / np.sqrt(6), -1 / np.sqrt(6)])
        self.assertAlmostEqual(eng.norm_squared(chi).value.real, 1.0, delta=TOL)
        self.assertEqual(chi.kind(), LazyNodeKind.Occupation)
        self.assertFalse(chi.is_boundary_fixture())
        self.assertAlmostEqual(self._j2_read(eng, chi, quark_modes), 0.75,
                               delta=1e-10)

    def test_proton_spin_oracle_ma_doublet(self):
        eng = LazyFockEngine(6)
        quark_modes = [(0, 1), (2, 3), (4, 5)]
        chi = eng.occupation_state(
            list(range(6)), [[0, 3, 4], [1, 2, 4]],
            [1 / np.sqrt(2), -1 / np.sqrt(2)])
        self.assertAlmostEqual(self._j2_read(eng, chi, quark_modes), 0.75,
                               delta=1e-10)

    def test_delta_spin_j2_fifteen_quarters(self):
        eng = LazyFockEngine(6)
        quark_modes = [(0, 1), (2, 3), (4, 5)]
        chi = eng.occupation_state(list(range(6)), [[0, 2, 4]], [1.0])
        self.assertAlmostEqual(self._j2_read(eng, chi, quark_modes), 3.75,
                               delta=1e-10)
        sym = eng.occupation_state(
            list(range(6)), [[0, 2, 5], [0, 3, 4], [1, 2, 4]],
            [1 / np.sqrt(3)] * 3)
        self.assertAlmostEqual(self._j2_read(eng, sym, quark_modes), 3.75,
                               delta=1e-10)


# ─── builders / validation / boundary fixture ──────────────────────────────

@unittest.skipUnless(HAVE_QUANTUM, "tessera built without the quantum subsystem")
class TestBuilders(unittest.TestCase):
    def test_occupation_state_merges_duplicates_and_drops_zeros(self):
        eng = LazyFockEngine(3)
        s = eng.occupation_state([0, 1], [[0], [0], [1]], [1.0, 2.0, 0.0])
        self.assertEqual(eng.amplitude(s, [0]).value, 3.0)
        self.assertEqual(eng.amplitude(s, [1]).value, 0.0)
        self.assertAlmostEqual(eng.norm_squared(s).value.real, 9.0, delta=0)

    def test_occupation_term_outside_modes_raises(self):
        eng = LazyFockEngine(3)
        with self.assertRaises(ValueError):
            eng.occupation_state([0, 1], [[2]], [1.0])

    def test_non_finite_amplitude_raises(self):
        eng = LazyFockEngine(2)
        with self.assertRaises(ValueError):
            eng.occupation_state([0], [[0]], [complex(np.nan, 0.0)])

    def test_boundary_fixture_label_required_and_carried(self):
        eng = LazyFockEngine(3)
        with self.assertRaises(ValueError):
            eng.boundary_product_fixture([0, 1], [1.0, 1.0], [0.0, 0.0], "")
        fx = eng.boundary_product_fixture(
            [0, 2], [1 / np.sqrt(2), 1.0], [1 / np.sqrt(2), 0.0],
            "M0 product preparation")
        self.assertTrue(fx.is_boundary_fixture())
        self.assertEqual(fx.boundary_fixture_label(), "M0 product preparation")
        # Product amplitudes: (|0> + |1>)/sqrt(2) on mode 0, |0> on mode 2.
        self.assertAlmostEqual(fx.discarded_norm(), 0.0, delta=0)
        self.assertAlmostEqual(eng.amplitude(fx, []).value, 1 / np.sqrt(2),
                               delta=TOL)
        self.assertAlmostEqual(eng.amplitude(fx, [0]).value, 1 / np.sqrt(2),
                               delta=TOL)
        self.assertEqual(eng.amplitude(fx, [2]).value, 0.0)
        # A non-fixture state is never labeled: product storage is opt-in.
        self.assertFalse(eng.vacuum().is_boundary_fixture())

    def test_duplicate_wedge_orbital_is_exactly_zero(self):
        eng = LazyFockEngine(4)
        v = np.zeros((4, 2), dtype=complex)
        v[:, 0] = [1, 2, 3, 4]
        v[:, 1] = [1, 2, 3, 4]
        w = eng.wedge_state([0, 1, 2, 3], v)
        self.assertEqual(eng.norm_squared(w).value, 0.0)
        for occ in ([0, 1], [1, 3], [0, 3]):
            self.assertAlmostEqual(abs(eng.amplitude(w, occ).value), 0.0,
                                   delta=1e-14)

    def test_density_boundary_sector_carried_vectorized(self):
        # A density-operator boundary sector rides the SAME six node
        # kinds, vectorized on a doubled (ket + bra) mode register:
        # |rho>> = sum_ij rho_ij |i>_ket (x) |j>_bra. Trace and
        # occupation reads are inner products against the vectorized
        # identity — exact, with no dedicated density node (spec 14.2's
        # node list is authoritative; the arbitrary-mode-count carrier
        # makes the doubled register free).
        eng = LazyFockEngine(2)  # mode 0 = ket copy, mode 1 = bra copy
        rho = np.array([[0.7, 0.2 - 0.1j], [0.2 + 0.1j, 0.3]], dtype=complex)
        vec_rho = eng.occupation_state(
            [0, 1], [[], [1], [0], [0, 1]],
            [rho[0, 0], rho[0, 1], rho[1, 0], rho[1, 1]])
        vec_id = eng.occupation_state([0, 1], [[], [0, 1]], [1.0, 1.0])
        # tr rho = <<I|rho>> = 1.
        self.assertAlmostEqual(eng.inner_product(vec_id, vec_rho).value, 1.0,
                               delta=TOL)
        # tr(rho n) = <<I| (n_ket x 1) |rho>> = rho_11 (n is even: no
        # Koszul twist touches the bra register).
        n_op = np.diag([0.0, 1.0]).astype(complex)
        self.assertAlmostEqual(
            eng.inner_product(vec_id,
                             eng.apply_local_map_dense(vec_rho, [0], n_op)).value,
            rho[1, 1], delta=TOL)
        # Mixedness is visible in the carrier: tr(rho^2) = ||vec(rho)||^2
        # (Hilbert-Schmidt norm of a Hermitian rho) < 1.
        purity = float(np.trace(rho @ rho).real)
        self.assertAlmostEqual(eng.norm_squared(vec_rho).value.real, purity,
                               delta=TOL)
        self.assertLess(purity, 1.0)


# ─── local maps against the independent JW polynomial extension ───────────

@unittest.skipUnless(HAVE_QUANTUM, "tessera built without the quantum subsystem")
class TestLocalMapsAgainstJordanWigner(unittest.TestCase):
    """Exact identity: a support-Fock matrix acts as its JW polynomial
    extension — for even AND odd operators, on arbitrary (interleaved)
    supports."""

    def _random_state(self, eng, rng, m):
        terms = random_terms(rng, list(range(m)), 6)
        occ = [t[0] for t in terms]
        amp = [t[1] for t in terms]
        state = eng.occupation_state(list(range(m)), occ, amp)
        return state, dense_state_from_terms(terms, m)

    def test_creation_annihilation_match_jw(self):
        m = 5
        rng = np.random.default_rng(3)
        eng = LazyFockEngine(m)
        state, dense = self._random_state(eng, rng, m)
        for mode in range(m):
            got_c = np.asarray(eng.dense_vector(eng.apply_creation(state, mode)))
            np.testing.assert_allclose(
                got_c, jw_creation(mode, m) @ dense, atol=TOL)
            got_a = np.asarray(
                eng.dense_vector(eng.apply_annihilation(state, mode)))
            np.testing.assert_allclose(
                got_a, jw_annihilation(mode, m) @ dense, atol=TOL)

    def test_local_map_matches_jw_polynomial_extension(self):
        m = 5
        rng = np.random.default_rng(11)
        eng = LazyFockEngine(m)
        state, dense = self._random_state(eng, rng, m)
        support = [1, 3]  # interleaved with the rest
        op = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
        got = np.asarray(
            eng.dense_vector(eng.apply_local_map_dense(state, support, op)))
        expected = local_op_dense(op, support, m) @ dense
        np.testing.assert_allclose(got, expected, atol=TOL)

    def test_local_map_three_mode_support(self):
        m = 6
        rng = np.random.default_rng(12)
        eng = LazyFockEngine(m)
        state, dense = self._random_state(eng, rng, m)
        support = [0, 2, 5]
        op = rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8))
        got = np.asarray(
            eng.dense_vector(eng.apply_local_map_dense(state, support, op)))
        np.testing.assert_allclose(got, local_op_dense(op, support, m) @ dense,
                                   atol=TOL)

    def test_local_map_coo_equals_dense(self):
        m = 4
        rng = np.random.default_rng(5)
        eng = LazyFockEngine(m)
        state, _ = self._random_state(eng, rng, m)
        op = np.zeros((4, 4), dtype=complex)
        op[1, 2] = 1.5 - 0.5j
        op[3, 0] = -2.0j
        got_dense = eng.apply_local_map_dense(state, [1, 2], op)
        got_coo = eng.apply_local_map_coo(state, [1, 2], [1, 3], [2, 0],
                                       [1.5 - 0.5j, -2.0j])
        np.testing.assert_allclose(np.asarray(eng.dense_vector(got_coo)),
                                   np.asarray(eng.dense_vector(got_dense)),
                                   atol=0)

    def test_car_nilpotency_and_anticommutator(self):
        m = 4
        rng = np.random.default_rng(9)
        eng = LazyFockEngine(m)
        state, dense = self._random_state(eng, rng, m)
        # a_i† a_i† = 0 exactly.
        twice = eng.apply_creation(eng.apply_creation(state, 2), 2)
        self.assertEqual(eng.norm_squared(twice).value, 0.0)
        # {a_i, a_j†} = delta_ij through engine evaluation.
        for i in range(m):
            for j in range(m):
                x = eng.apply_annihilation(eng.apply_creation(state, j), i)
                y = eng.apply_creation(eng.apply_annihilation(state, i), j)
                lhs = (np.asarray(eng.dense_vector(x))
                       + np.asarray(eng.dense_vector(y)))
                expected = dense if i == j else np.zeros_like(dense)
                np.testing.assert_allclose(lhs, expected, atol=TOL)


# ─── graded tensor semantics ───────────────────────────────────────────────

@unittest.skipUnless(HAVE_QUANTUM, "tessera built without the quantum subsystem")
class TestGradedTensor(unittest.TestCase):
    def test_tensor_matches_wedge_word_oracle_interleaved(self):
        # Mode sets A = {0, 2}, B = {1, 3}: interleaved in the global
        # order — the Koszul sign rule against the definitional
        # wedge-word product.
        m = 4
        rng = np.random.default_rng(21)
        eng = LazyFockEngine(m)
        terms_a = random_terms(rng, [0, 2], 4)
        terms_b = random_terms(rng, [1, 3], 4)
        a = eng.occupation_state([0, 2], [t[0] for t in terms_a],
                                [t[1] for t in terms_a])
        b = eng.occupation_state([1, 3], [t[0] for t in terms_b],
                                [t[1] for t in terms_b])
        got = np.asarray(eng.dense_vector(eng.graded_tensor(a, b)))
        expected = dense_tensor_oracle(terms_a, terms_b, m)
        np.testing.assert_allclose(got, expected, atol=TOL)

    def test_parenthesizations_agree_after_graded_associators(self):
        m = 6
        rng = np.random.default_rng(23)
        eng = LazyFockEngine(m)
        parts = ([0, 3], [1, 5], [2, 4])
        states = []
        for modes in parts:
            terms = random_terms(rng, modes, 3)
            states.append(eng.occupation_state(
                modes, [t[0] for t in terms], [t[1] for t in terms]))
        a, b, c = states
        left = eng.graded_tensor(eng.graded_tensor(a, b), c)
        right = eng.graded_tensor(a, eng.graded_tensor(b, c))
        # Different DAGs...
        self.assertNotEqual(left.content_hash(), right.content_hash())
        # ...same amplitudes: the associator is an algebraic identity —
        # signs and structure exact, values to double round-off (the two
        # parenthesizations multiply the SAME factors in different
        # floating-point association orders).
        for occ in all_keys(m):
            lv = eng.amplitude(left, occ).value
            rv = eng.amplitude(right, occ).value
            self.assertLessEqual(abs(lv - rv), 1e-14 * max(1.0, abs(rv)))

    def test_parenthesizations_agree_exactly_on_integer_fixtures(self):
        # With ±1/±2 integer amplitudes every product is exact in binary
        # floating point, so the associator holds with EXACT equality —
        # including every Koszul sign.
        m = 6
        eng = LazyFockEngine(m)
        a = eng.occupation_state([0, 3], [[0], [3], [0, 3]], [1.0, -2.0, 1.0])
        b = eng.occupation_state([1, 5], [[1], []], [1.0, -1.0])
        c = eng.occupation_state([2, 4], [[2, 4], [4]], [2.0, 1.0])
        left = eng.graded_tensor(eng.graded_tensor(a, b), c)
        right = eng.graded_tensor(a, eng.graded_tensor(b, c))
        for occ in all_keys(m):
            self.assertEqual(eng.amplitude(left, occ).value,
                             eng.amplitude(right, occ).value)

    def test_tensor_swap_koszul_sign(self):
        eng = LazyFockEngine(4)
        # Odd (N=1) x odd (N=1): swap must flip every amplitude.
        a = eng.occupation_state([0, 2], [[0], [2]], [1.0, 2.0])
        b = eng.occupation_state([1, 3], [[1], [3]], [0.5, -1.0])
        ab = np.asarray(eng.dense_vector(eng.graded_tensor(a, b)))
        ba = np.asarray(eng.dense_vector(eng.graded_tensor(b, a)))
        np.testing.assert_allclose(ab, -ba, atol=0)
        # Even x odd: swap is the identity.
        e = eng.occupation_state([0, 2], [[0, 2], []], [1.0, 3.0])
        eb = np.asarray(eng.dense_vector(eng.graded_tensor(e, b)))
        be = np.asarray(eng.dense_vector(eng.graded_tensor(b, e)))
        np.testing.assert_allclose(eb, be, atol=0)

    def test_tensor_mode_overlap_raises(self):
        eng = LazyFockEngine(3)
        a = eng.occupation_state([0, 1], [[0]], [1.0])
        b = eng.occupation_state([1, 2], [[2]], [1.0])
        with self.assertRaises(ValueError):
            eng.graded_tensor(a, b)

    def test_vacuum_tensor_is_identity_on_amplitudes(self):
        m = 4
        rng = np.random.default_rng(29)
        eng = LazyFockEngine(m)
        terms = random_terms(rng, [1, 2], 3)
        s = eng.occupation_state([1, 2], [t[0] for t in terms],
                                [t[1] for t in terms])
        t = eng.graded_tensor(s, eng.vacuum_on([0, 3]))
        for occ, amp in terms:
            self.assertEqual(eng.amplitude(t, occ).value,
                             eng.amplitude(s, occ).value)


# ─── dGamma, direct sums, hopping, subset-sum spectra ─────────────────────

@unittest.skipUnless(HAVE_QUANTUM, "tessera built without the quantum subsystem")
class TestDGammaAndSpectra(unittest.TestCase):
    def _dense_coo(self, coo):
        rows, cols, vals, n = coo
        out = np.zeros((n, n), dtype=complex)
        for r, c, v in zip(rows, cols, vals):
            out[r, c] += v
        return out

    def _engine_operator_matrix(self, eng, m, apply_fn):
        """Assemble the dense matrix of an engine-applied operator by
        acting on every basis state."""
        dim = 2 ** m
        out = np.zeros((dim, dim), dtype=complex)
        for idx, occ in enumerate(all_keys(m)):
            basis = eng.occupation_state(list(range(m)), [occ], [1.0])
            out[:, idx] = np.asarray(eng.dense_vector(apply_fn(basis)))
        return out

    def test_dgamma_matches_independent_jw_dense(self):
        m = 5
        rng = np.random.default_rng(31)
        eng = LazyFockEngine(m)
        h = rng.normal(size=(m, m)) + 1j * rng.normal(size=(m, m))
        h = h + h.conj().T
        got = self._engine_operator_matrix(
            eng, m, lambda s: eng.apply_d_gamma(s, list(range(m)), h))
        np.testing.assert_allclose(got, dgamma_dense(h, range(m), m), atol=TOL)

    def test_dgamma_direct_sum_and_hopping_match_fock_direct_sum(self):
        # d_gamma([[L_A, C], [C†, L_B]]) against the merged #766
        # FockDirectSum.dGammaBlock — direct sums become graded tensor
        # products, coupling blocks become hopping terms.
        ma, mb = 2, 2
        m = ma + mb
        rng = np.random.default_rng(37)
        eng = LazyFockEngine(m)
        f = FockDirectSum(ma, mb)
        la = rng.normal(size=(ma, ma)) + 1j * rng.normal(size=(ma, ma))
        la = la + la.conj().T
        lb = rng.normal(size=(mb, mb)) + 1j * rng.normal(size=(mb, mb))
        lb = lb + lb.conj().T
        c = rng.normal(size=(ma, mb)) + 1j * rng.normal(size=(ma, mb))
        block = np.asarray(FockDirectSum.assemble_block_one_particle(la, lb, c))
        got = self._engine_operator_matrix(
            eng, m, lambda s: eng.apply_d_gamma(s, list(range(m)), block))
        expected = self._dense_coo(f.d_gamma_block_coo(la, lb, c))
        np.testing.assert_allclose(got, expected, atol=TOL)
        # Zero coupling: exact direct-sum identity.
        got0 = self._engine_operator_matrix(
            eng, m,
            lambda s: eng.apply_d_gamma(
                s, list(range(m)),
                np.asarray(FockDirectSum.assemble_block_one_particle(
                    la, lb, np.zeros((ma, mb), dtype=complex)))))
        expected0 = self._dense_coo(
            f.d_gamma_block_coo(la, lb, np.zeros((ma, mb), dtype=complex)))
        np.testing.assert_allclose(got0, expected0, atol=TOL)

    def test_dgamma_conserves_occupation_block_sparsity(self):
        eng = LazyFockEngine(4)
        h = np.eye(4, dtype=complex)
        s = eng.occupation_state([0, 1, 2, 3], [[0, 2]], [1.0])
        out = eng.apply_d_gamma(s, [0, 1, 2, 3], h)
        self.assertEqual(out.definite_occupation(), 2)
        self.assertEqual(out.definite_parity(), +1)
        # Out-of-sector reads short-circuit to exact zero.
        self.assertEqual(eng.amplitude(out, [0]).value, 0.0)
        self.assertEqual(eng.amplitude(out, [0, 1, 2]).value, 0.0)

    def test_free_spectrum_matches_subset_sums_and_dense_sectors(self):
        m = 5
        rng = np.random.default_rng(41)
        eng = LazyFockEngine(m)
        h = rng.normal(size=(m, m)) + 1j * rng.normal(size=(m, m))
        h = h + h.conj().T
        eigs = sorted(np.linalg.eigvalsh(h).tolist())
        dense = dgamma_dense(h, range(m), m)
        for particles in range(m + 1):
            values, cert = eng.free_spectrum(h, particles)
            expected = cob.OccupationSpectra.subset_sums(
                [complex(x) for x in eigs], particles)
            np.testing.assert_allclose(values, expected, atol=1e-9)
            self.assertTrue(cert.holds())
            # Dense sector eigenvalues (project dGamma onto popcount = N).
            idx = [i for i in range(2 ** m) if bin(i).count("1") == particles]
            sector = dense[np.ix_(idx, idx)]
            sector_eigs = sorted(np.linalg.eigvalsh(sector).tolist())
            np.testing.assert_allclose(
                sorted(v.real for v in values), sector_eigs, atol=1e-9)

    def test_free_spectrum_from_eigenvalues_delegates_exactly(self):
        eng = LazyFockEngine(1)
        spec = [complex(1, 0.5), complex(2, -1), complex(4, 0), complex(0.25, 2)]
        for particles in range(5):
            values, cert = eng.free_spectrum_from_eigenvalues(spec, particles)
            expected = cob.OccupationSpectra.subset_sums(spec, particles)
            np.testing.assert_allclose(values, expected, atol=0)
            self.assertTrue(cert.holds())

    def test_dgamma_on_slater_single_replacement_identity(self):
        # d_gamma(L)(v1 ^ v2) = (L v1) ^ v2 + v1 ^ (L v2) — exact.
        m = 4
        rng = np.random.default_rng(43)
        eng = LazyFockEngine(m)
        l = rng.normal(size=(m, m)) + 1j * rng.normal(size=(m, m))
        v = rng.normal(size=(m, 2)) + 1j * rng.normal(size=(m, 2))
        w = eng.wedge_state(list(range(m)), v)
        got = np.asarray(eng.dense_vector(eng.apply_d_gamma(w, list(range(m)), l)))
        r1 = np.column_stack([l @ v[:, 0], v[:, 1]])
        r2 = np.column_stack([v[:, 0], l @ v[:, 1]])
        expected = (np.asarray(eng.dense_vector(eng.wedge_state(list(range(m)), r1)))
                    + np.asarray(eng.dense_vector(eng.wedge_state(list(range(m)), r2))))
        np.testing.assert_allclose(got, expected, atol=TOL)


# ─── vacuum embedding and inductive compatibility ─────────────────────────

@unittest.skipUnless(HAVE_QUANTUM, "tessera built without the quantum subsystem")
class TestVacuumEmbedding(unittest.TestCase):
    def test_embedding_preserves_all_amplitudes(self):
        # New modes INTERLEAVED with the old ones: iota must still be the
        # identity on every preexisting amplitude (empty right word).
        m = 6
        rng = np.random.default_rng(47)
        eng = LazyFockEngine(m)
        terms = random_terms(rng, [0, 2, 4], 6)
        s = eng.occupation_state([0, 2, 4], [t[0] for t in terms],
                                [t[1] for t in terms])
        emb = eng.embed_in_vacuum(s, [1, 3, 5])
        for occ, _ in terms:
            self.assertEqual(eng.amplitude(emb, occ).value,
                             eng.amplitude(s, occ).value)
        # And nothing new appeared.
        self.assertAlmostEqual(eng.norm_squared(emb).value.real,
                               eng.norm_squared(s).value.real, delta=0)
        # Any key occupying a new mode is exactly zero.
        self.assertEqual(eng.amplitude(emb, [1]).value, 0.0)
        self.assertEqual(eng.amplitude(emb, [0, 3]).value, 0.0)

    def test_embedding_preserves_covariance_block(self):
        eng = LazyFockEngine(4)
        v = np.array([[1.0], [1.0j]], dtype=complex) / np.sqrt(2)
        w = eng.wedge_state([0, 2], v)
        emb = eng.embed_in_vacuum(w, [1, 3])
        g0 = np.asarray(eng.covariance_matrix(w).matrix)
        g1 = np.asarray(eng.covariance_matrix(emb).matrix)
        np.testing.assert_allclose(g0, g1, atol=TOL)
        np.testing.assert_allclose(g1[[1, 3], :], 0.0, atol=0)

    def test_embedding_mode_clash_raises(self):
        eng = LazyFockEngine(3)
        s = eng.occupation_state([0, 1], [[0]], [1.0])
        with self.assertRaises(ValueError):
            eng.embed_in_vacuum(s, [1])

    def test_inductive_compatibility_zero_for_consistent_extension(self):
        # U_{M+1} = the SAME support operator: iota U_M = U_{M+1} iota
        # exactly, so epsilon = 0 on any active subspace.
        eng = LazyFockEngine(3)
        rng = np.random.default_rng(53)
        u = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
        read = eng.inductive_compatibility(
            [0, 1], [0, 1, 2], [0, 1], u, [0, 1], u,
            [[], [0], [1], [0, 1]])
        self.assertEqual(read.active_dimension, 4)
        self.assertLess(read.epsilon, 1e-14)
        self.assertTrue(read.certificate.holds())

    def test_inductive_compatibility_detects_new_mode_coupling(self):
        # U_{M+1} hops into the fresh mode: epsilon must match the
        # independent dense computation of ||iota U_M - U_{M+1} iota||
        # restricted to the active columns.
        m1 = 3
        eng = LazyFockEngine(m1)
        rng = np.random.default_rng(59)
        u_small = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
        u_big = rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8))
        active = [[], [0], [1], [0, 1]]
        read = eng.inductive_compatibility(
            [0, 1], [0, 1, 2], [0, 1], u_small, [0, 1, 2], u_big, active)
        # Independent dense: JW polynomial extensions over 3 modes.
        big_small = local_op_dense(u_small, [0, 1], m1)
        big_big = local_op_dense(u_big, [0, 1, 2], m1)
        cols = [sum(1 << m for m in occ) for occ in active]
        defect = (big_small - big_big)[:, cols]
        expected = np.linalg.svd(defect, compute_uv=False)[0]
        self.assertAlmostEqual(read.epsilon, expected, delta=1e-10)
        self.assertGreater(read.epsilon, 0.1)  # a real defect was planted


# ─── laziness: crossing-only expansion, shared siblings, sector routing ───

@unittest.skipUnless(HAVE_QUANTUM, "tessera built without the quantum subsystem")
class TestLaziness(unittest.TestCase):
    def _two_branch(self, eng, rng):
        ta = random_terms(rng, [0, 1], 3)
        tb = random_terms(rng, [2, 3], 3)
        a = eng.occupation_state([0, 1], [t[0] for t in ta], [t[1] for t in ta])
        b = eng.occupation_state([2, 3], [t[0] for t in tb], [t[1] for t in tb])
        return eng.graded_tensor(a, b), ta, tb

    def test_left_branch_op_shares_right_sibling(self):
        eng = LazyFockEngine(4)
        rng = np.random.default_rng(61)
        t, _, _ = self._two_branch(eng, rng)
        before = eng.expansion_count()
        op = np.diag([1.0, 2.0, 3.0, 4.0]).astype(complex)
        t2 = eng.apply_local_map_dense(t, [0, 1], op)
        self.assertEqual(eng.expansion_count(), before)
        self.assertEqual(t2.kind(), LazyNodeKind.GradedTensor)
        # The untouched RIGHT sibling is the SAME node object.
        self.assertEqual(t2.child_node_ids()[1], t.child_node_ids()[1])
        self.assertNotEqual(t2.child_node_ids()[0], t.child_node_ids()[0])

    def test_right_branch_even_op_shares_left_sibling(self):
        eng = LazyFockEngine(4)
        rng = np.random.default_rng(67)
        t, _, _ = self._two_branch(eng, rng)
        before = eng.expansion_count()
        op = np.zeros((4, 4), dtype=complex)  # even: number-conserving
        op[0, 0] = 1.0
        op[1, 1] = -1.0
        op[2, 2] = 2.0
        op[3, 3] = 0.5
        t2 = eng.apply_local_map_dense(t, [2, 3], op)
        self.assertEqual(eng.expansion_count(), before)
        self.assertEqual(t2.child_node_ids()[0], t.child_node_ids()[0])

    def test_right_branch_odd_op_with_definite_parity_sibling_stays_lazy(self):
        eng = LazyFockEngine(4)
        # Left branch parity-definite (single N=1 term).
        a = eng.occupation_state([0, 1], [[0]], [1.0])
        b = eng.occupation_state([2, 3], [[2], []], [0.5, 0.5])
        t = eng.graded_tensor(a, b)
        before = eng.expansion_count()
        t2 = eng.apply_creation(t, 3)  # odd operator on the right branch
        self.assertEqual(eng.expansion_count(), before)
        self.assertEqual(t2.child_node_ids()[0], t.child_node_ids()[0])
        # Exactness of the Koszul twist against independent JW dense.
        expected = jw_creation(3, 4) @ dense_tensor_oracle(
            [([0], 1.0)], [([2], 0.5), ([], 0.5)], 4)
        np.testing.assert_allclose(np.asarray(eng.dense_vector(t2)), expected,
                                   atol=TOL)

    def test_crossing_op_expands_partition(self):
        eng = LazyFockEngine(4)
        rng = np.random.default_rng(71)
        t, ta, tb = self._two_branch(eng, rng)
        before = eng.expansion_count()
        # Hopping ACROSS the partition: modes 1 (left) and 2 (right).
        hop = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
        t2 = eng.apply_d_gamma(t, [1, 2], hop)
        self.assertEqual(eng.expansion_count(), before + 1)
        self.assertEqual(t2.kind(), LazyNodeKind.Occupation)
        # Exact against the independent dense oracle.
        expected = dgamma_dense(hop, [1, 2], 4) @ dense_tensor_oracle(ta, tb, 4)
        np.testing.assert_allclose(np.asarray(eng.dense_vector(t2)), expected,
                                   atol=TOL)

    def test_disconnected_third_branch_never_expands(self):
        eng = LazyFockEngine(6)
        rng = np.random.default_rng(73)
        ta = random_terms(rng, [0, 1], 3)
        tb = random_terms(rng, [2, 3], 3)
        tc = random_terms(rng, [4, 5], 3)
        a = eng.occupation_state([0, 1], [t[0] for t in ta], [t[1] for t in ta])
        b = eng.occupation_state([2, 3], [t[0] for t in tb], [t[1] for t in tb])
        c = eng.occupation_state([4, 5], [t[0] for t in tc], [t[1] for t in tc])
        t = eng.graded_tensor(eng.graded_tensor(a, b), c)
        before = eng.expansion_count()
        hop = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
        t2 = eng.apply_d_gamma(t, [1, 2], hop)  # crosses a|b, disconnected from c
        # The a|b partition expanded once; the c branch node is untouched.
        self.assertEqual(eng.expansion_count(), before + 1)
        self.assertEqual(t2.kind(), LazyNodeKind.GradedTensor)
        self.assertEqual(t2.child_node_ids()[1], t.child_node_ids()[1])

    def test_dgamma_inside_one_branch_stays_lazy(self):
        eng = LazyFockEngine(4)
        rng = np.random.default_rng(79)
        t, _, _ = self._two_branch(eng, rng)
        before = eng.expansion_count()
        l = np.array([[1.0, 2.0], [2.0, -1.0]], dtype=complex)
        t2 = eng.apply_d_gamma(t, [0, 1], l)
        self.assertEqual(eng.expansion_count(), before)
        self.assertEqual(t2.child_node_ids()[1], t.child_node_ids()[1])

    def test_sector_sum_routes_and_survives_even_maps(self):
        eng = LazyFockEngine(3)
        even = eng.occupation_state([0, 1, 2], [[], [0, 1]], [0.6, 0.8])
        odd = eng.occupation_state([0, 1, 2], [[2]], [1.0])
        s = eng.sector_sum([even, odd], LazySectorKind.Parity)
        self.assertEqual(s.kind(), LazyNodeKind.SectorSum)
        self.assertEqual(eng.amplitude(s, [2]).value, 1.0)
        self.assertEqual(eng.amplitude(s, [0, 1]).value, 0.8)
        # An even (number-conserving) map preserves the sector structure.
        l = np.eye(3, dtype=complex)
        s2 = eng.apply_d_gamma(s, [0, 1, 2], l)
        self.assertEqual(s2.kind(), LazyNodeKind.SectorSum)

    def test_sector_sum_duplicate_sector_raises(self):
        eng = LazyFockEngine(2)
        a = eng.occupation_state([0, 1], [[0]], [1.0])
        b = eng.occupation_state([0, 1], [[1]], [1.0])
        with self.assertRaises(ValueError):
            eng.sector_sum([a, b], LazySectorKind.Occupation)
        with self.assertRaises(ValueError):
            eng.sector_sum([a, b], LazySectorKind.Parity)


# ─── memoization ───────────────────────────────────────────────────────────

@unittest.skipUnless(HAVE_QUANTUM, "tessera built without the quantum subsystem")
class TestMemoization(unittest.TestCase):
    def test_memo_hits_and_cold_equality(self):
        m = 8
        rng = np.random.default_rng(83)
        eng = LazyFockEngine(m)
        terms_a = random_terms(rng, [0, 1, 2, 3], 8)
        terms_b = random_terms(rng, [4, 5, 6, 7], 8)
        a = eng.occupation_state([0, 1, 2, 3], [t[0] for t in terms_a],
                                [t[1] for t in terms_a])
        b = eng.occupation_state([4, 5, 6, 7], [t[0] for t in terms_b],
                                [t[1] for t in terms_b])
        t = eng.graded_tensor(a, b)
        eng.clear_memo()
        warm1 = eng.norm_squared(t).value
        misses_after_first = eng.memo_misses()
        hits_after_first = eng.memo_hits()
        warm2 = eng.norm_squared(t).value
        self.assertEqual(warm1, warm2)
        # Second evaluation was served from the memo (no new misses)...
        self.assertEqual(eng.memo_misses(), misses_after_first)
        self.assertGreaterEqual(eng.memo_hits(), hits_after_first)
        # ...and a cold recomputation gives the identical value.
        eng.clear_memo()
        cold = eng.norm_squared(t).value
        self.assertEqual(cold, warm1)

    def test_memo_shared_across_structurally_equal_nodes(self):
        eng = LazyFockEngine(4)
        a1 = eng.occupation_state([0, 1], [[0], [1]], [1.0, 2.0])
        a2 = eng.occupation_state([0, 1], [[0], [1]], [1.0, 2.0])
        self.assertEqual(a1.content_hash(), a2.content_hash())
        self.assertNotEqual(a1.root_node_id(), a2.root_node_id())
        eng.clear_memo()
        eng.materialize(a1)
        misses = eng.memo_misses()
        eng.materialize(a2)  # structurally equal: memo hit, no new miss
        self.assertEqual(eng.memo_misses(), misses)
        self.assertGreater(eng.memo_hits(), 0)


# ─── quasi-free / Slater reference layer ──────────────────────────────────

@unittest.skipUnless(HAVE_QUANTUM, "tessera built without the quantum subsystem")
class TestSlaterQuasiFree(unittest.TestCase):
    """Gamma_ef = P_ef from a spectral projector — the #780 covariance
    layer's dense/oracle reference, validated against dense #766-built
    references (direct #780 cross-validation deferred to post-merge
    integration)."""

    def _random_projector(self, rng, m, rank):
        h = rng.normal(size=(m, m)) + 1j * rng.normal(size=(m, m))
        h = h + h.conj().T
        _, vecs = np.linalg.eigh(h)
        v = vecs[:, :rank]
        return v @ v.conj().T

    def test_slater_from_projector_covariance_equals_p(self):
        m, rank = 6, 3
        rng = np.random.default_rng(89)
        p = self._random_projector(rng, m, rank)
        eng = LazyFockEngine(m)
        ref = eng.slater_from_projector(list(range(m)), p, 1e-10)
        self.assertEqual(ref.rank, rank)
        self.assertLess(ref.projector_residual, 1e-12)
        self.assertTrue(ref.certificate.holds())
        gamma = np.asarray(eng.covariance_matrix(ref.state).matrix)
        np.testing.assert_allclose(gamma, p, atol=1e-12)

    def test_slater_covariance_general_path_agrees_with_closed_form(self):
        m, rank = 5, 2
        rng = np.random.default_rng(97)
        p = self._random_projector(rng, m, rank)
        eng = LazyFockEngine(m)
        ref = eng.slater_from_projector(list(range(m)), p, 1e-10)
        closed = np.asarray(eng.covariance_matrix(ref.state).matrix)
        general = np.asarray(
            eng.covariance_matrix(eng.materialize(ref.state)).matrix)
        np.testing.assert_allclose(general, closed, atol=1e-12)
        np.testing.assert_allclose(general, p, atol=1e-12)

    def test_non_projector_input_fails_loudly(self):
        eng = LazyFockEngine(3)
        not_p = np.diag([0.5, 0.5, 0.0]).astype(complex)  # not idempotent
        with self.assertRaises(ValueError):
            eng.slater_from_projector([0, 1, 2], not_p, 1e-10)

    def test_wedge_norm_equals_gram_determinant(self):
        m = 5
        rng = np.random.default_rng(101)
        eng = LazyFockEngine(m)
        v = rng.normal(size=(m, 3)) + 1j * rng.normal(size=(m, 3))
        w = eng.wedge_state(list(range(m)), v)
        gram = v.conj().T @ v
        self.assertAlmostEqual(eng.norm_squared(w).value.real,
                               np.linalg.det(gram).real, delta=1e-9)

    def test_slater_amplitudes_are_slater_determinants(self):
        m = 5
        rng = np.random.default_rng(103)
        eng = LazyFockEngine(m)
        v = rng.normal(size=(m, 2)) + 1j * rng.normal(size=(m, 2))
        w = eng.wedge_state(list(range(m)), v)
        for occ in itertools.combinations(range(m), 2):
            minor = v[list(occ), :]
            self.assertAlmostEqual(eng.amplitude(w, list(occ)).value,
                                   np.linalg.det(minor), delta=1e-12)

    def test_wedge_matches_independent_jw_construction(self):
        m = 4
        rng = np.random.default_rng(107)
        eng = LazyFockEngine(m)
        v = rng.normal(size=(m, 2)) + 1j * rng.normal(size=(m, 2))
        got = np.asarray(eng.dense_vector(eng.wedge_state(list(range(m)), v)))
        vac = np.zeros(2 ** m, dtype=complex)
        vac[0] = 1.0
        adag1 = sum(v[i, 0] * jw_creation(i, m) for i in range(m))
        adag2 = sum(v[i, 1] * jw_creation(i, m) for i in range(m))
        np.testing.assert_allclose(got, adag1 @ (adag2 @ vac), atol=TOL)


# ─── exact certification vs truncation mode ────────────────────────────────

@unittest.skipUnless(HAVE_QUANTUM, "tessera built without the quantum subsystem")
class TestTruncation(unittest.TestCase):
    def test_exact_mode_reads_are_algebraically_exact(self):
        eng = LazyFockEngine(3)
        self.assertTrue(eng.exact_mode())
        s = eng.occupation_state([0, 1, 2], [[0], [1, 2]], [0.6, 0.8])
        read = eng.amplitude(s, [0])
        self.assertEqual(read.discarded_norm, 0.0)
        self.assertTrue(read.certificate.holds())
        self.assertEqual(read.certificate.grade,
                         cob.CertificateGrade.AlgebraicallyExact)

    def test_truncation_error_bounded_by_reported_discarded_norm(self):
        m = 4
        rng = np.random.default_rng(109)
        eng_exact = LazyFockEngine(m)
        occupations = list(all_keys(m))
        amps = [complex(rng.normal(), rng.normal()) for _ in occupations]
        # Plant several tiny components.
        for k in (1, 5, 9, 13):
            amps[k] = complex(1e-7 * rng.normal(), 1e-7 * rng.normal())
        exact = eng_exact.occupation_state(list(range(m)), occupations, amps)
        exact_vec = np.asarray(eng_exact.dense_vector(exact))
        for eta in (1e-9, 1e-6, 1e-3):
            eng = LazyFockEngine(m)
            eng.set_truncation_threshold(eta, 1.0)
            self.assertFalse(eng.exact_mode())
            s = eng.occupation_state(list(range(m)), occupations, amps)
            t = eng.materialize(s)
            reported = t.discarded_norm()
            for idx, occ in enumerate(occupations):
                read = eng.amplitude(t, occ)
                err = abs(read.value - exact_vec[idx])
                self.assertLessEqual(err, reported + 1e-15)
                self.assertEqual(read.discarded_norm, reported)

    def test_truncation_bound_survives_unitary_maps_and_tensors(self):
        m = 4
        rng = np.random.default_rng(113)
        # Exact pipeline.
        eng0 = LazyFockEngine(m)
        terms_a = [([0], 1.0), ([1], 1e-8)]
        terms_b = [([2], 0.8), ([3], 0.6)]
        a0 = eng0.occupation_state([0, 1], [t[0] for t in terms_a],
                                  [t[1] for t in terms_a])
        b0 = eng0.occupation_state([2, 3], [t[0] for t in terms_b],
                                  [t[1] for t in terms_b])
        x = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
        u, _ = np.linalg.qr(x)  # unitary on modes {0,1}
        exact = eng0.apply_local_map_dense(
            eng0.graded_tensor(a0, b0), [0, 1], u)
        exact_vec = np.asarray(eng0.dense_vector(exact))
        # Truncating pipeline: the tiny term is dropped at materialize.
        eng = LazyFockEngine(m)
        eng.set_truncation_threshold(1e-6, 1.0)
        a = eng.materialize(eng.occupation_state(
            [0, 1], [t[0] for t in terms_a], [t[1] for t in terms_a]))
        self.assertGreater(a.discarded_norm(), 0.0)
        b = eng.occupation_state([2, 3], [t[0] for t in terms_b],
                                [t[1] for t in terms_b])
        t = eng.apply_local_map_dense(eng.graded_tensor(a, b), [0, 1], u)
        bound = t.discarded_norm()
        self.assertGreater(bound, 0.0)
        got_vec = np.asarray(eng.dense_vector(t))
        self.assertLessEqual(np.linalg.norm(got_vec - exact_vec),
                             bound + 1e-15)
        for occ in all_keys(m):
            idx = sum(1 << mm for mm in occ)
            read = eng.amplitude(t, occ)
            self.assertLessEqual(abs(read.value - exact_vec[idx]),
                                 bound + 1e-15)

    def test_truncation_certificate_reports_bound_against_budget(self):
        eng = LazyFockEngine(2)
        eng.set_truncation_threshold(1e-3, 1e-2)
        s = eng.materialize(eng.occupation_state(
            [0, 1], [[0], [1]], [1.0, 1e-4]))
        read = eng.amplitude(s, [0])
        self.assertEqual(read.certificate.grade,
                         cob.CertificateGrade.CertifiedNumerical)
        self.assertAlmostEqual(read.certificate.residual, s.discarded_norm(),
                               delta=0)
        self.assertTrue(read.certificate.holds())      # 1e-4 <= 1e-2 budget
        eng2 = LazyFockEngine(2)
        eng2.set_truncation_threshold(1e-3, 1e-6)        # tighter budget
        s2 = eng2.materialize(eng2.occupation_state(
            [0, 1], [[0], [1]], [1.0, 1e-4]))
        self.assertFalse(eng2.amplitude(s2, [0]).certificate.holds())

    def test_truncation_threshold_validation(self):
        eng = LazyFockEngine(2)
        with self.assertRaises(ValueError):
            eng.set_truncation_threshold(0.0, 1.0)
        with self.assertRaises(ValueError):
            eng.set_truncation_threshold(-1e-3, 1.0)
        eng.set_truncation_threshold(1e-3, 1.0)
        eng.clear_truncation()
        self.assertTrue(eng.exact_mode())


# ─── serialization: content hashes, no flattening, checkpoint/replay ─────

@unittest.skipUnless(HAVE_QUANTUM, "tessera built without the quantum subsystem")
class TestSerialization(unittest.TestCase):
    def _pipeline_state(self, eng, rng):
        ta = random_terms(rng, [0, 1], 3)
        tb = random_terms(rng, [2, 3], 3)
        a = eng.occupation_state([0, 1], [t[0] for t in ta], [t[1] for t in ta])
        b = eng.occupation_state([2, 3], [t[0] for t in tb], [t[1] for t in tb])
        return eng.graded_tensor(a, b)

    def test_round_trip_hash_and_amplitudes_bit_exact(self):
        m = 4
        rng = np.random.default_rng(127)
        eng = LazyFockEngine(m)
        t = self._pipeline_state(eng, rng)
        blob = eng.serialize(t)
        t2 = eng.deserialize(blob)
        self.assertEqual(t2.content_hash(), t.content_hash())
        self.assertEqual(t2.kind(), t.kind())
        self.assertEqual(t2.node_count(), t.node_count())
        for occ in all_keys(m):
            self.assertEqual(eng.amplitude(t2, occ).value,
                             eng.amplitude(t, occ).value)

    def test_serialization_does_not_flatten_the_dag(self):
        # The checkpoint stores the EXPRESSION (tensor root + children +
        # a lazy LocalMap over a wedge), not one flattened amplitude
        # table.
        eng = LazyFockEngine(5)
        rng = np.random.default_rng(131)
        v = rng.normal(size=(3, 1)) + 1j * rng.normal(size=(3, 1))
        w = eng.wedge_state([0, 2, 4], v)
        op = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
        lazy = eng.apply_local_map_dense(w, [0, 2], op)  # LocalMap over Wedge
        rest = eng.occupation_state([1, 3], [[1], [3]], [0.5, 0.5])
        t = eng.graded_tensor(lazy, rest)
        blob = eng.serialize(t)
        t2 = eng.deserialize(blob)
        self.assertEqual(t2.kind(), LazyNodeKind.GradedTensor)
        self.assertEqual(t2.node_count(), t.node_count())
        self.assertEqual(blob.count('"kind":"wedge"'), 1)
        self.assertEqual(blob.count('"kind":"localMap"'), 1)
        # A materialized copy is a different, single-node expression.
        flat = eng.materialize(t)
        self.assertEqual(flat.node_count(), 1)
        self.assertNotEqual(flat.content_hash(), t2.content_hash())
        # Same amplitudes either way.
        for occ in ([0, 1], [2, 3], [4, 1]):
            self.assertAlmostEqual(eng.amplitude(t2, occ).value,
                                   eng.amplitude(flat, occ).value,
                                   delta=1e-12)

    def test_checkpoint_replay_continues_the_pipeline(self):
        m = 4
        rng = np.random.default_rng(137)
        eng = LazyFockEngine(m)
        t = self._pipeline_state(eng, rng)
        hop = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
        final_direct = eng.apply_d_gamma(t, [1, 2], hop)
        # Checkpoint mid-pipeline, replay in a FRESH engine, continue.
        blob = eng.serialize(t)
        eng2 = LazyFockEngine(m)
        t2 = eng2.deserialize(blob)
        final_replayed = eng2.apply_d_gamma(t2, [1, 2], hop)
        np.testing.assert_allclose(
            np.asarray(eng2.dense_vector(final_replayed)),
            np.asarray(eng.dense_vector(final_direct)), atol=0)

    def test_tampered_checkpoint_rejected(self):
        eng = LazyFockEngine(3)
        s = eng.occupation_state([0, 1], [[0]], [0.75])
        blob = eng.serialize(s)
        tampered = blob.replace("0.75", "0.76")
        with self.assertRaises(ValueError):
            eng.deserialize(tampered)

    def test_schema_and_universe_mismatches_rejected(self):
        eng = LazyFockEngine(3)
        s = eng.occupation_state([0, 1], [[0]], [1.0])
        blob = eng.serialize(s)
        with self.assertRaises(ValueError):
            eng.deserialize(blob.replace('"version":1', '"version":2'))
        with self.assertRaises(ValueError):
            eng.deserialize(blob.replace("tessera.lazyfock.dag", "other.schema"))
        other = LazyFockEngine(4)
        with self.assertRaises(ValueError):
            other.deserialize(blob)

    def test_fixture_label_and_discarded_norm_round_trip(self):
        eng = LazyFockEngine(2)
        eng.set_truncation_threshold(1e-3, 1.0)
        fx = eng.boundary_product_fixture([0, 1], [1.0, 1.0], [1e-5, 0.5],
                                        "boundary fixture: M0 \"edge\" prep")
        t = eng.materialize(fx)
        self.assertGreater(t.discarded_norm(), 0.0)
        blob = eng.serialize(t)
        t2 = eng.deserialize(blob)
        self.assertEqual(t2.boundary_fixture_label(), t.boundary_fixture_label())
        self.assertEqual(t2.discarded_norm(), t.discarded_norm())

    def test_awkward_doubles_round_trip_bit_exact(self):
        eng = LazyFockEngine(2)
        amps = [complex(1 / 3, -1 / 7), complex(1e-17, math.pi),
                complex(-0.0, 2 ** -52)]
        s = eng.occupation_state([0, 1], [[0], [1], [0, 1]], amps)
        t = eng.deserialize(eng.serialize(s))
        self.assertEqual(t.content_hash(), s.content_hash())
        for occ, amp in zip([[0], [1], [0, 1]], amps):
            self.assertEqual(eng.amplitude(t, occ).value, amp)


# ─── relabeling / mode-order invariance ────────────────────────────────────

@unittest.skipUnless(HAVE_QUANTUM, "tessera built without the quantum subsystem")
class TestRelabeling(unittest.TestCase):
    def _check_invariance(self, eng, state, perm, m):
        permuted = eng.permute_modes(state, perm)
        from tessera.quantum import OccupationBitset
        for occ in all_keys(m):
            key = OccupationBitset.from_occupied_modes(m, occ)
            parity = key.permutation_parity(perm)
            image = sorted(perm[mm] for mm in occ)
            lhs = eng.amplitude(permuted, image).value
            rhs = parity * eng.amplitude(state, occ).value
            self.assertAlmostEqual(lhs, rhs, delta=1e-12)

    def test_relabeling_preserves_amplitudes_occupation_and_tensor(self):
        m = 5
        rng = np.random.default_rng(139)
        eng = LazyFockEngine(m)
        ta = random_terms(rng, [0, 2], 3)
        tb = random_terms(rng, [1, 3, 4], 4)
        a = eng.occupation_state([0, 2], [t[0] for t in ta], [t[1] for t in ta])
        b = eng.occupation_state([1, 3, 4], [t[0] for t in tb],
                                [t[1] for t in tb])
        t = eng.graded_tensor(a, b)
        for _ in range(4):
            perm = rng.permutation(m).tolist()
            self._check_invariance(eng, t, perm, m)

    def test_relabeling_preserves_amplitudes_wedge_and_local_map(self):
        m = 5
        rng = np.random.default_rng(149)
        eng = LazyFockEngine(m)
        v = rng.normal(size=(4, 2)) + 1j * rng.normal(size=(4, 2))
        w = eng.wedge_state([0, 1, 3, 4], v)
        op = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
        lazy = eng.apply_local_map_dense(w, [1, 3], op)  # LocalMap over Wedge
        self.assertEqual(lazy.kind(), LazyNodeKind.LocalMap)
        for _ in range(3):
            perm = rng.permutation(m).tolist()
            self._check_invariance(eng, lazy, perm, m)

    def test_registry_relabeling_drives_the_permutation(self):
        reg = EdgeModeRegistry()
        reg.add_edge(10, 11, +1, "compA")
        reg.add_edge(11, 12, +1, "compB")
        reg.add_edge(12, 13, +1, "compA")
        eng = LazyFockEngine.from_registry(reg)
        m = eng.mode_count()
        state = eng.occupation_state(list(range(m)),
                                    [[0], [1], [0, 1, 2]],
                                    [0.5, 0.5j, 1.0])
        relabeled = reg.relabeled({10: 40, 11: 3, 12: 22, 13: 7})
        perm = EdgeModeRegistry.order_permutation(reg, relabeled)
        self._check_invariance(eng, state, perm, m)

    def test_relabeling_conjugates_covariance(self):
        m = 4
        rng = np.random.default_rng(151)
        eng = LazyFockEngine(m)
        v = rng.normal(size=(m, 2)) + 1j * rng.normal(size=(m, 2))
        w = eng.wedge_state(list(range(m)), v)
        perm = rng.permutation(m).tolist()
        g0 = np.asarray(eng.covariance_matrix(w).matrix)
        g1 = np.asarray(eng.covariance_matrix(eng.permute_modes(w, perm)).matrix)
        p = np.zeros((m, m))
        for i, pi in enumerate(perm):
            p[pi, i] = 1.0
        np.testing.assert_allclose(g1, p @ g0 @ p.T, atol=1e-12)


# ─── guards, arbitrary mode count, refusals ────────────────────────────────

@unittest.skipUnless(HAVE_QUANTUM, "tessera built without the quantum subsystem")
class TestGuardsAndScale(unittest.TestCase):
    def test_expansion_refusal_threshold(self):
        eng = LazyFockEngine(8)
        eng.set_max_expansion_terms(10)
        occ_a = [list(t) for k in range(5)
                 for t in itertools.combinations([0, 1, 2, 3], k)][:8]
        occ_b = [list(t) for k in range(5)
                 for t in itertools.combinations([4, 5, 6, 7], k)][:8]
        a = eng.occupation_state([0, 1, 2, 3], occ_a, [1.0] * len(occ_a))
        b = eng.occupation_state([4, 5, 6, 7], occ_b, [1.0] * len(occ_b))
        t = eng.graded_tensor(a, b)  # 64 terms > 10
        with self.assertRaises(Exception):
            eng.materialize(t)

    def test_arbitrary_mode_count_via_chunked_bitsets(self):
        # M = 300 modes: NO 2^M anywhere — sparse keys are chunked #766
        # bitsets. Creation across chunks, dGamma hopping between distant
        # modes, tensors across far mode groups, exact amplitudes.
        m = 300
        eng = LazyFockEngine(m)
        s = eng.occupation_state([10, 150, 290], [[10], [150]], [0.6, 0.8])
        s2 = eng.apply_creation(s, 290)
        # Signs: mode 290 creation crosses one occupied mode (10 or 150).
        self.assertAlmostEqual(eng.amplitude(s2, [10, 290]).value, -0.6,
                               delta=0)
        self.assertAlmostEqual(eng.amplitude(s2, [150, 290]).value, -0.8,
                               delta=0)
        hop = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
        s3 = eng.apply_d_gamma(s, [10, 290], hop)  # hop 10 <-> 290
        self.assertAlmostEqual(eng.amplitude(s3, [290]).value, 0.6, delta=0)
        far = eng.occupation_state([250], [[250]], [1.0])
        t = eng.graded_tensor(s, far)
        self.assertAlmostEqual(eng.amplitude(t, [150, 250]).value, 0.8,
                               delta=0)
        self.assertAlmostEqual(eng.norm_squared(t).value.real, 1.0, delta=TOL)
        # Wedge over distant modes: Slater determinant amplitudes intact.
        v = np.array([[1.0, 1.0], [1.0, -1.0], [0.5, 0.25]], dtype=complex)
        w = eng.wedge_state([5, 100, 299], v)
        self.assertAlmostEqual(eng.amplitude(w, [5, 100]).value,
                               np.linalg.det(v[[0, 1], :]), delta=1e-12)
        self.assertAlmostEqual(eng.amplitude(w, [100, 299]).value,
                               np.linalg.det(v[[1, 2], :]), delta=1e-12)

    def test_support_outside_coverage_raises_with_embed_hint(self):
        eng = LazyFockEngine(4)
        s = eng.occupation_state([0, 1], [[0]], [1.0])
        with self.assertRaises(ValueError) as ctx:
            eng.apply_creation(s, 3)
        self.assertIn("embedInVacuum", str(ctx.exception))
        # After embedding, the same operation is well-defined.
        s2 = eng.apply_creation(eng.embed_in_vacuum(s, [3]), 3)
        self.assertAlmostEqual(eng.amplitude(s2, [0, 3]).value, -1.0, delta=0)

    def test_amplitude_outside_state_modes_is_vacuum_zero(self):
        eng = LazyFockEngine(4)
        s = eng.occupation_state([0, 1], [[0]], [1.0])
        self.assertEqual(eng.amplitude(s, [3]).value, 0.0)
        self.assertEqual(eng.amplitude(s, [0, 3]).value, 0.0)

    def test_local_map_support_cap(self):
        eng = LazyFockEngine(2)
        s = eng.occupation_state([0, 1], [[0]], [1.0])
        with self.assertRaises(ValueError):
            eng.apply_local_map_dense(s, [0, 1], np.eye(3, dtype=complex))


if __name__ == "__main__":
    unittest.main()
