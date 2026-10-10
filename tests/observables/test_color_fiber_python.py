# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""Exact fixtures for the three-edge SU(3) color kernel (issue #767):
ColorFiber (constant algebra, normalizers, certificates, sector reads) and
ColorAnchor (the calibrated weighted oriented-triangle anchoring kernel).

Acceptance coverage (ticket #767):

* F3^dag F3 = I and |det F3| = 1;
* Gell-Mann matrices Hermitian/traceless with Tr(lambda_a lambda_b) =
  2 delta_ab;
* [E_ij, E_kl] = delta_jk E_il - delta_il E_kj on the 3x3 matrix units AND
  the full 8x8 Fock bilinears;
* det(gC) = det(C) for random CERTIFIED g in SU(3);
* the singlet wedge vanishes for duplicate color modes and reaches unit
  Gram determinant for an orthonormal triad;
* literal-triangle and extended-atlas anchor fixtures pass, an abstract
  unanchored rank-three band fails, the calibrated score never exceeds one
  (to round-off) and reaches one exactly on the concentrated oracle, and
  post-hoc weight selection is rejected;
* every read is invariant under oriented edge relabeling and in-band SU(3)
  frame changes.

Exactness bar: algebraic identities are compared at double round-off
(~1e-15); expected values are built from sqrt(3) and the cube root of
unity ALGEBRAICALLY, with floating representations compared only at the
final boundary.  Anything looser (matrix products, eigen-modulus paths) is
labeled with its honest tolerance.

The references are INDEPENDENT NumPy constructions (exp-based DFT, a
hardcoded Gell-Mann table, dense Jordan-Wigner kron chains as in
tests/quantum/test_graded_fock_python.py, and a standalone NumPy anchor
evaluator) — never re-derivations through the bindings under test.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

import tessera
from tessera.quantum import ExteriorAlgebra

ColorFiber = tessera.ColorFiber
ColorAnchor = tessera.ColorAnchor
OrientedTriangle = tessera.OrientedTriangle

SQRT3 = math.sqrt(3.0)
# Algebraic omega = (-1 + i sqrt(3)) / 2 — the same closed form the kernel
# documents; the EXP-based value below is the independent cross-check.
OMEGA_ALG = complex(-0.5, SQRT3 / 2.0)
OMEGA_EXP = np.exp(2j * np.pi / 3.0)


# ─── independent references ────────────────────────────────────────────────

def dense(coo):
    """(rows, cols, values, n) COO tuple -> dense complex ndarray."""
    rows, cols, vals, n = coo
    out = np.zeros((n, n), dtype=complex)
    for r, c, v in zip(rows, cols, vals):
        out[r, c] += v
    return out


_S_MINUS = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
_Z = np.diag([1.0, -1.0]).astype(complex)


def jw_annihilation(mode: int, n_modes: int) -> np.ndarray:
    """Independent dense Jordan-Wigner a_mode on the n(b) = sum b_i 2^i
    basis (same reference construction as test_graded_fock_python.py)."""
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


# The standard Gell-Mann table, hardcoded independently of the C++ path.
GELL_MANN_REF = {
    1: np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=complex),
    2: np.array([[0, -1j, 0], [1j, 0, 0], [0, 0, 0]], dtype=complex),
    3: np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]], dtype=complex),
    4: np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]], dtype=complex),
    5: np.array([[0, 0, -1j], [0, 0, 0], [1j, 0, 0]], dtype=complex),
    6: np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=complex),
    7: np.array([[0, 0, 0], [0, 0, -1j], [0, 1j, 0]], dtype=complex),
    8: np.diag([1, 1, -2]).astype(complex) / SQRT3,
}


def random_su3(rng) -> np.ndarray:
    """A Haar-ish random SU(3) element, CERTIFIED before use."""
    z = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    q, r = np.linalg.qr(z)
    q = q @ np.diag(r.diagonal() / np.abs(r.diagonal()))
    q = q / np.linalg.det(q) ** (1.0 / 3.0)
    # Certification is part of the fixture: reject a bad sample loudly.
    assert np.max(np.abs(q.conj().T @ q - np.eye(3))) <= 1e-12
    assert abs(np.linalg.det(q) - 1.0) <= 1e-12
    return q


def ref_anchor_terms(frame, edge_weights, triangles):
    """Standalone NumPy evaluation of |det(|W_tau|^{1/2} R_tau Phi)|^2 and
    det phases for a DIAGONAL edge-weight vector."""
    absw = np.abs(np.asarray(edge_weights, dtype=float))
    terms, phases = [], []
    for edges, signs in triangles:
        rows = np.array(
            [s * np.asarray(frame)[e, :] for e, s in zip(edges, signs)])
        a = np.diag(np.sqrt(absw[list(edges)])) @ rows
        d = np.linalg.det(a)
        terms.append(abs(d) ** 2)
        phases.append(np.angle(d) if d != 0 else np.nan)
    return np.array(terms), np.array(phases)


def ref_overlapping(triangles):
    """Which declared triangles SHARE A BOUNDARY EDGE with another declared
    triangle -- the overlap relation the determinant-phase coherence is
    recorded on (#808).  An OrientedTriangle names only its three edge rows,
    so shared-edge is the only sharing relation the atlas determines."""
    mask = np.zeros(len(triangles), dtype=bool)
    for i, (edges_i, _s) in enumerate(triangles):
        for j, (edges_j, _t) in enumerate(triangles):
            if i != j and set(edges_i) & set(edges_j):
                mask[i] = True
                break
    return mask


def ref_profile(frame, edge_weights, triangles, weights):
    """Standalone NumPy score / participation ratio / phase coherence.  The
    coherence runs only over triangles that genuinely overlap (#808)."""
    terms, phases = ref_anchor_terms(frame, edge_weights, triangles)
    w = np.asarray(weights, dtype=float)
    score = float(np.sum(w * terms))
    sum_t, sum_t2 = float(np.sum(terms)), float(np.sum(terms**2))
    pr = (sum_t * sum_t / sum_t2) if sum_t2 > 0 else 0.0
    u = w * terms
    mask = (terms > 0) & ref_overlapping(triangles)
    if np.sum(u[mask]) > 0:
        coherence = abs(np.sum(u[mask] * np.exp(1j * phases[mask]))) / float(
            np.sum(u[mask]))
    else:
        coherence = np.nan
    return score, terms, pr, phases, coherence


def orthonormal_band(rng, n_edges, edge_weights):
    """A random rank-three |W|-orthonormal band over n_edges edges."""
    frame = rng.normal(size=(n_edges, 3)) + 1j * rng.normal(size=(n_edges, 3))
    return ColorAnchor.orthonormalize_frame(frame, np.asarray(edge_weights))


# ─── the exact Fourier frame from omega ────────────────────────────────────

class TestFourierFrame(unittest.TestCase):
    """F3 built from the cube root of unity: unitarity, |det| = 1, and the
    identification of (1, omega, omega^2)/sqrt(3) with its cyclic triad."""

    def test_omega_is_the_algebraic_cube_root(self) -> None:
        w = ColorFiber.omega()
        self.assertEqual(w, OMEGA_ALG)
        # Cross-check against the independent exp-based value at the final
        # floating boundary.
        self.assertLessEqual(abs(w - OMEGA_EXP), 1e-15)

    def test_omega_cubic_identities(self) -> None:
        w = ColorFiber.omega()
        # 1 + omega + conj(omega) cancels EXACTLY with the algebraic
        # components (conj(omega) is the algebraic omega^2).
        self.assertEqual(1.0 + w + np.conj(w), 0.0)
        self.assertLessEqual(abs(w**3 - 1.0), 1e-15)
        self.assertLessEqual(abs(w**2 - np.conj(w)), 1e-15)

    def test_f3_dagger_f3_is_identity(self) -> None:
        f = ColorFiber.fourier_frame()
        self.assertLessEqual(
            np.max(np.abs(f.conj().T @ f - np.eye(3))), 1e-15)

    def test_f3_det_modulus_one(self) -> None:
        f = ColorFiber.fourier_frame()
        self.assertLessEqual(abs(abs(np.linalg.det(f)) - 1.0), 1e-15)

    def test_f3_matches_independent_exp_dft(self) -> None:
        f = ColorFiber.fourier_frame()
        j, k = np.meshgrid(np.arange(3), np.arange(3), indexing="ij")
        ref = OMEGA_EXP ** (j * k) / SQRT3
        self.assertLessEqual(np.max(np.abs(f - ref)), 1e-15)

    def test_omega_phase_state_is_one_basis_vector(self) -> None:
        v = ColorFiber.omega_phase_state()
        ref = np.array([1.0, OMEGA_ALG, np.conj(OMEGA_ALG)]) / SQRT3
        self.assertLessEqual(np.max(np.abs(v - ref)), 1e-15)
        f = ColorFiber.fourier_frame()
        self.assertTrue(np.array_equal(v, f[:, 1]))
        self.assertTrue(
            np.array_equal(v, ColorFiber.fourier_basis_vector(1)))

    def test_cyclic_triad_is_orthonormal(self) -> None:
        cols = [ColorFiber.fourier_basis_vector(k) for k in range(3)]
        for a in range(3):
            for b in range(3):
                inner = np.vdot(cols[a], cols[b])
                self.assertLessEqual(
                    abs(inner - (1.0 if a == b else 0.0)), 1e-15)

    def test_cyclic_triad_generated_by_pointwise_z3_powers(self) -> None:
        # v_k are Z3 characters: sqrt(3) * (v1 ∘ v1) = v2 and
        # sqrt(3) * (v1 ∘ v2) = v0 — the cyclic orbit of the omega pattern.
        v0 = ColorFiber.fourier_basis_vector(0)
        v1 = ColorFiber.fourier_basis_vector(1)
        v2 = ColorFiber.fourier_basis_vector(2)
        self.assertLessEqual(np.max(np.abs(SQRT3 * v1 * v1 - v2)), 1e-15)
        self.assertLessEqual(np.max(np.abs(SQRT3 * v1 * v2 - v0)), 1e-15)

    def test_fourier_basis_vector_range_errors(self) -> None:
        with self.assertRaises(ValueError):
            ColorFiber.fourier_basis_vector(3)
        with self.assertRaises(ValueError):
            ColorFiber.fourier_basis_vector(-1)

    def test_constants_regenerate_identically(self) -> None:
        # The constant algebra is a pure function: two generations agree
        # bitwise (no state, no cache to drift — cold recomputation IS the
        # production path).
        self.assertTrue(
            np.array_equal(ColorFiber.fourier_frame(),
                           ColorFiber.fourier_frame()))
        for a in range(1, 9):
            self.assertTrue(
                np.array_equal(ColorFiber.gell_mann(a),
                               ColorFiber.gell_mann(a)))


# ─── Gell-Mann generators on the one-occupation sector ─────────────────────

class TestGellMann(unittest.TestCase):
    def test_hermitian_exact(self) -> None:
        for a in range(1, 9):
            la = ColorFiber.gell_mann(a)
            self.assertEqual(np.max(np.abs(la - la.conj().T)), 0.0)

    def test_traceless_exact(self) -> None:
        for a in range(1, 9):
            self.assertEqual(np.trace(ColorFiber.gell_mann(a)), 0.0)

    def test_trace_orthonormalization_two_delta(self) -> None:
        for a in range(1, 9):
            for b in range(1, 9):
                tr = np.trace(ColorFiber.gell_mann(a) @ ColorFiber.gell_mann(b))
                expected = 2.0 if a == b else 0.0
                self.assertLessEqual(abs(tr - expected), 1e-15,
                                     msg=f"Tr(l{a} l{b})")

    def test_matches_independent_table(self) -> None:
        for a in range(1, 9):
            self.assertLessEqual(
                np.max(np.abs(ColorFiber.gell_mann(a) - GELL_MANN_REF[a])),
                1e-15, msg=f"lambda_{a}")

    def test_cartan_generators_h1_h2(self) -> None:
        u = ColorFiber.matrix_unit
        self.assertTrue(
            np.array_equal(ColorFiber.gell_mann(3), u(0, 0) - u(1, 1)))
        h2 = (u(0, 0) + u(1, 1) - 2.0 * u(2, 2)) / SQRT3
        self.assertLessEqual(
            np.max(np.abs(ColorFiber.gell_mann(8) - h2)), 1e-15)

    def test_index_errors(self) -> None:
        for bad in (0, 9, -1):
            with self.assertRaises(ValueError):
                ColorFiber.gell_mann(bad)


# ─── E_ij bilinears and the gl(3) commutator identity ──────────────────────

class TestBilinears(unittest.TestCase):
    def test_gl3_commutators_on_matrix_units_exact(self) -> None:
        u = ColorFiber.matrix_unit
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for L in range(3):
                        lhs = u(i, j) @ u(k, L) - u(k, L) @ u(i, j)
                        rhs = np.zeros((3, 3), dtype=complex)
                        if j == k:
                            rhs += u(i, L)
                        if i == L:
                            rhs -= u(k, j)
                        self.assertEqual(np.max(np.abs(lhs - rhs)), 0.0,
                                         msg=f"[E{i}{j}, E{k}{L}]")

    def test_gl3_commutators_on_fock_bilinears_exact(self) -> None:
        e = {(i, j): ColorFiber.hopping_matrix(i, j)
             for i in range(3) for j in range(3)}
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for L in range(3):
                        lhs = e[i, j] @ e[k, L] - e[k, L] @ e[i, j]
                        rhs = np.zeros((8, 8), dtype=complex)
                        if j == k:
                            rhs += e[i, L]
                        if i == L:
                            rhs -= e[k, j]
                        self.assertEqual(np.max(np.abs(lhs - rhs)), 0.0,
                                         msg=f"[E{i}{j}, E{k}{L}] (Fock)")

    def test_creation_annihilation_match_jordan_wigner(self) -> None:
        for i in range(3):
            self.assertEqual(
                np.max(np.abs(ColorFiber.creation_matrix(i) -
                              jw_creation(i, 3))), 0.0)
            self.assertEqual(
                np.max(np.abs(ColorFiber.annihilation_matrix(i) -
                              jw_annihilation(i, 3))), 0.0)

    def test_car_anticommutators(self) -> None:
        for i in range(3):
            for j in range(3):
                ai = ColorFiber.annihilation_matrix(i)
                cj = ColorFiber.creation_matrix(j)
                anti = ai @ cj + cj @ ai
                expected = np.eye(8) if i == j else np.zeros((8, 8))
                self.assertEqual(np.max(np.abs(anti - expected)), 0.0)

    def test_hopping_matches_jw_product(self) -> None:
        for i in range(3):
            for j in range(3):
                ref = jw_creation(i, 3) @ jw_annihilation(j, 3)
                self.assertEqual(
                    np.max(np.abs(ColorFiber.hopping_matrix(i, j) - ref)),
                    0.0)

    def test_triplet_basis_indices(self) -> None:
        self.assertEqual(tuple(ColorFiber.triplet_basis_indices()), (1, 2, 4))

    def test_restrict_hopping_is_matrix_unit(self) -> None:
        for i in range(3):
            for j in range(3):
                got = ColorFiber.restrict_to_triplet(
                    ColorFiber.hopping_matrix(i, j))
                self.assertEqual(
                    np.max(np.abs(got - ColorFiber.matrix_unit(i, j))), 0.0)

    def test_restrict_dgamma_is_identity_map(self) -> None:
        rng = np.random.default_rng(11)
        mats = [ColorFiber.gell_mann(a) for a in range(1, 9)]
        mats.append(rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3)))
        for m in mats:
            got = ColorFiber.restrict_to_triplet(ColorFiber.d_gamma(m))
            self.assertLessEqual(np.max(np.abs(got - m)), 1e-15)

    def test_dgamma_annihilates_vacuum_sector(self) -> None:
        rng = np.random.default_rng(12)
        m = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
        dg = ColorFiber.d_gamma(m)
        vac = np.zeros(8, dtype=complex)
        vac[0] = 1.0
        self.assertEqual(np.max(np.abs(dg @ vac)), 0.0)

    def test_shape_and_index_errors(self) -> None:
        with self.assertRaises(ValueError):
            ColorFiber.restrict_to_triplet(np.eye(3, dtype=complex))
        with self.assertRaises(ValueError):
            ColorFiber.matrix_unit(3, 0)
        with self.assertRaises(ValueError):
            ColorFiber.creation_matrix(3)


# ─── the N = 0,1,2,3 sector projectors ─────────────────────────────────────

class TestSectorProjectors(unittest.TestCase):
    def popcount_projector(self, n: int) -> np.ndarray:
        return np.diag([1.0 if bin(b).count("1") == n else 0.0
                        for b in range(8)]).astype(complex)

    def test_match_independent_popcount_masks(self) -> None:
        for n in range(4):
            self.assertEqual(
                np.max(np.abs(ColorFiber.sector_projector(n) -
                              self.popcount_projector(n))), 0.0)

    def test_idempotent_orthogonal_complete(self) -> None:
        projectors = [ColorFiber.sector_projector(n) for n in range(4)]
        total = np.zeros((8, 8), dtype=complex)
        for a, p in enumerate(projectors):
            self.assertEqual(np.max(np.abs(p @ p - p)), 0.0)
            for b in range(a):
                self.assertEqual(
                    np.max(np.abs(p @ projectors[b])), 0.0)
            total += p
        self.assertEqual(np.max(np.abs(total - np.eye(8))), 0.0)

    def test_sector_dimensions_1_3_3_1(self) -> None:
        for n, dim in zip(range(4), (1, 3, 3, 1)):
            self.assertEqual(np.trace(ColorFiber.sector_projector(n)),
                             complex(dim))

    def test_named_projectors_are_the_sectors(self) -> None:
        self.assertTrue(np.array_equal(ColorFiber.vacuum_projector(),
                                       ColorFiber.sector_projector(0)))
        self.assertTrue(np.array_equal(ColorFiber.triplet_projector(),
                                       ColorFiber.sector_projector(1)))
        self.assertTrue(np.array_equal(ColorFiber.anti_triplet_projector(),
                                       ColorFiber.sector_projector(2)))
        self.assertTrue(np.array_equal(ColorFiber.singlet_projector(),
                                       ColorFiber.sector_projector(3)))

    def test_sector_above_top_is_zero(self) -> None:
        self.assertEqual(np.max(np.abs(ColorFiber.sector_projector(4))), 0.0)

    def test_fermion_parity_pattern(self) -> None:
        # 1 ⊕ 3 ⊕ 3̄ ⊕ 1 parities: even, odd, even, odd.
        parity = np.diag([(-1.0) ** bin(b).count("1") for b in range(8)])
        for n, sign in zip(range(4), (+1.0, -1.0, +1.0, -1.0)):
            p = ColorFiber.sector_projector(n)
            self.assertEqual(np.max(np.abs(parity @ p - sign * p)), 0.0)

    def test_delegation_matches_quantum_primitive(self) -> None:
        # The color API layers interpretation over the #766 primitive — the
        # matrices must be the SAME object content.
        alg = ExteriorAlgebra(3)
        for n in range(4):
            self.assertEqual(
                np.max(np.abs(ColorFiber.sector_projector(n) -
                              dense(alg.sector_projector_coo(n)))), 0.0)


# ─── the traceless adjoint-octet projector ─────────────────────────────────

class TestAdjointOctet(unittest.TestCase):
    def test_matches_independent_construction(self) -> None:
        vec_i = np.eye(3, dtype=complex).reshape(9, order="F")
        ref = np.eye(9) - np.outer(vec_i, vec_i.conj()) / 3.0
        self.assertLessEqual(
            np.max(np.abs(ColorFiber.adjoint_octet_projector() - ref)), 1e-15)

    def test_projector_algebra(self) -> None:
        p8 = ColorFiber.adjoint_octet_projector()
        self.assertLessEqual(np.max(np.abs(p8 - p8.conj().T)), 1e-15)
        self.assertLessEqual(np.max(np.abs(p8 @ p8 - p8)), 1e-15)
        self.assertLessEqual(abs(np.trace(p8) - 8.0), 1e-15)

    def test_fixes_gellmann_kills_identity(self) -> None:
        p8 = ColorFiber.adjoint_octet_projector()
        for a in range(1, 9):
            v = ColorFiber.gell_mann(a).reshape(9, order="F")
            self.assertLessEqual(np.max(np.abs(p8 @ v - v)), 1e-15)
        vec_i = np.eye(3, dtype=complex).reshape(9, order="F")
        self.assertLessEqual(np.max(np.abs(p8 @ vec_i)), 1e-15)

    def test_vec_convention_is_column_major(self) -> None:
        rng = np.random.default_rng(21)
        m = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
        p8 = ColorFiber.adjoint_octet_projector()
        got = p8 @ m.reshape(9, order="F")
        want = ColorFiber.traceless_part(m).reshape(9, order="F")
        self.assertLessEqual(np.max(np.abs(got - want)), 1e-15)

    def test_octet_read_frobenius_split(self) -> None:
        rng = np.random.default_rng(22)
        m = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
        read = ColorFiber.octet_read(m)
        frob = np.linalg.norm(m, "fro") ** 2
        self.assertLessEqual(abs(read.octet + read.singlet - frob),
                             1e-13 * frob)
        self.assertLessEqual(
            abs(read.singlet - abs(np.trace(m)) ** 2 / 3.0), 1e-13 * frob)

    def test_octet_read_on_generators_and_identity(self) -> None:
        for a in range(1, 9):
            read = ColorFiber.octet_read(ColorFiber.gell_mann(a))
            self.assertLessEqual(abs(read.octet - 2.0), 1e-14)
            self.assertLessEqual(read.singlet, 1e-15)
        read = ColorFiber.octet_read(np.eye(3, dtype=complex))
        self.assertEqual(read.octet, 0.0)
        self.assertEqual(read.singlet, 3.0)


# ─── perimeter vs Hilbert normalization; the color vector ──────────────────

class TestNormalizers(unittest.TestCase):
    def test_color_vector_is_unit_and_parallel(self) -> None:
        rng = np.random.default_rng(31)
        z = rng.normal(size=3) + 1j * rng.normal(size=3)
        c = ColorFiber.color_vector(z)
        self.assertLessEqual(abs(np.vdot(c, c).real - 1.0), 1e-15)
        self.assertLessEqual(np.max(np.abs(c * np.linalg.norm(z) - z)),
                             1e-15 * np.linalg.norm(z))
        self.assertTrue(np.array_equal(c, ColorFiber.hilbert_normalized(z)))

    def test_omega_pattern_color_vector_is_fourier_basis_vector(self) -> None:
        # Unit-modulus squared lengths with the omega phases: the color
        # vector IS the identified Fourier basis vector.
        z = np.array([1.0, OMEGA_ALG, np.conj(OMEGA_ALG)])
        c = ColorFiber.color_vector(z)
        self.assertLessEqual(
            np.max(np.abs(c - ColorFiber.omega_phase_state())), 1e-15)

    def test_perimeter_independent_reference(self) -> None:
        rng = np.random.default_rng(32)
        z = rng.normal(size=3) + 1j * rng.normal(size=3)
        ref = float(np.sum(np.sqrt(np.abs(z))))
        self.assertLessEqual(abs(ColorFiber.perimeter(z) - ref), 1e-15 * ref)
        # Algebraic fixture: |z_i| = 1 each -> perimeter exactly 3.
        z3 = np.array([1.0, OMEGA_ALG, np.conj(OMEGA_ALG)])
        self.assertLessEqual(abs(ColorFiber.perimeter(z3) - 3.0), 1e-15)

    def test_perimeter_normalized_has_unit_perimeter(self) -> None:
        rng = np.random.default_rng(33)
        z = 3.0 * (rng.normal(size=3) + 1j * rng.normal(size=3))
        zn = ColorFiber.perimeter_normalized(z)
        self.assertLessEqual(abs(ColorFiber.perimeter(zn) - 1.0), 1e-14)

    def test_perimeter_and_hilbert_are_distinct_apis(self) -> None:
        # The L1 scale gauge is NOT the L2 state normalization: on a
        # generic triangle they disagree, and the perimeter-normalized
        # vector is not a unit Hilbert vector.
        z = np.array([2.0 + 0.5j, -0.25 + 1.0j, 0.75 - 0.3j])
        zp = ColorFiber.perimeter_normalized(z)
        zh = ColorFiber.hilbert_normalized(z)
        self.assertGreater(np.max(np.abs(zp - zh)), 1e-3)
        self.assertGreater(abs(np.linalg.norm(zp) - 1.0), 1e-3)
        self.assertLessEqual(abs(np.linalg.norm(zh) - 1.0), 1e-15)

    def test_scale_gauge_covariance(self) -> None:
        rng = np.random.default_rng(34)
        z = rng.normal(size=3) + 1j * rng.normal(size=3)
        s = 2.75
        zp1 = ColorFiber.perimeter_normalized(z)
        zp2 = ColorFiber.perimeter_normalized(s * s * z)
        self.assertLessEqual(np.max(np.abs(zp1 - zp2)), 1e-14)

    def test_zero_inputs_raise(self) -> None:
        zero = np.zeros(3, dtype=complex)
        with self.assertRaises(ValueError):
            ColorFiber.color_vector(zero)
        with self.assertRaises(ValueError):
            ColorFiber.hilbert_normalized(zero)
        with self.assertRaises(ValueError):
            ColorFiber.perimeter_normalized(zero)


# ─── det(C) / det(C†C) certificates ────────────────────────────────────────

class TestWedgeCertificates(unittest.TestCase):
    def test_orthonormal_triad_reaches_unit_gram(self) -> None:
        f = ColorFiber.fourier_frame()
        self.assertLessEqual(abs(ColorFiber.singlet_gram(f) - 1.0), 1e-15)
        rng = np.random.default_rng(41)
        for _ in range(5):
            g = random_su3(rng)
            self.assertLessEqual(abs(ColorFiber.singlet_gram(g) - 1.0),
                                 1e-13)

    def test_duplicate_color_modes_vanish(self) -> None:
        # Honest precision label: the det-based certificate cancels a
        # duplicate mode at double ROUND-OFF (Eigen's 3x3 determinant
        # expands along the first column, so the cancellation is not
        # bitwise); the EXACT zero is the exterior-algebra wedge below.
        rng = np.random.default_rng(42)
        a = rng.normal(size=3) + 1j * rng.normal(size=3)
        b = rng.normal(size=3) + 1j * rng.normal(size=3)
        scale = float(np.linalg.norm(a) ** 2 * np.linalg.norm(b))
        for cols in ((a, a, b), (a, b, a), (b, a, a)):
            self.assertLessEqual(
                abs(ColorFiber.color_wedge_columns(*cols)), 1e-15 * scale)
        c = np.column_stack([a, a, b])
        self.assertLessEqual(abs(ColorFiber.color_wedge(c)), 1e-15 * scale)
        self.assertLessEqual(ColorFiber.singlet_gram(c),
                             (1e-15 * scale) ** 2)

    def test_duplicate_color_modes_wedge_to_exact_zero_in_fock(self) -> None:
        # The #766 primitive carries the EXACT Pauli identity for duplicate
        # COMPLETE modes (its own documented exactness domain): the wedge is
        # bitwise zero, and every color-sector read of it is exactly zero.
        alg = ExteriorAlgebra(3)
        basis = np.eye(3, dtype=complex)
        psi = alg.wedge([basis[0], basis[0], basis[2]])
        self.assertEqual(np.max(np.abs(psi)), 0.0)
        w = ColorFiber.sector_weights(np.asarray(psi))
        self.assertEqual((w.vacuum, w.quark, w.anti_triplet, w.singlet),
                         (0.0, 0.0, 0.0, 0.0))
        # A repeated GENERAL complex color column cancels at double
        # round-off (documented in the #766 suite: FMA contraction), in
        # agreement with the det-based certificate above.
        rng = np.random.default_rng(42)
        a = rng.normal(size=3) + 1j * rng.normal(size=3)
        b = rng.normal(size=3) + 1j * rng.normal(size=3)
        psi2 = np.asarray(alg.wedge([a, a, b]))
        self.assertLessEqual(np.max(np.abs(psi2)), 1e-14)

    def test_wedge_antisymmetry_exact(self) -> None:
        rng = np.random.default_rng(43)
        a = rng.normal(size=3) + 1j * rng.normal(size=3)
        b = rng.normal(size=3) + 1j * rng.normal(size=3)
        c = rng.normal(size=3) + 1j * rng.normal(size=3)
        self.assertEqual(ColorFiber.color_wedge_columns(a, b, c),
                         -ColorFiber.color_wedge_columns(b, a, c))

    def test_det_gc_equals_det_c_for_certified_su3(self) -> None:
        rng = np.random.default_rng(44)
        c = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
        det_c = ColorFiber.color_wedge(c)
        for _ in range(20):
            g = random_su3(rng)
            self.assertTrue(ColorFiber.is_special_unitary(g, 1e-12))
            self.assertLessEqual(
                abs(ColorFiber.color_wedge(g @ c) - det_c),
                1e-12 * max(1.0, abs(det_c)))
            # And the Gram certificate is invariant too.
            self.assertLessEqual(
                abs(ColorFiber.singlet_gram(g @ c) -
                    ColorFiber.singlet_gram(c)),
                1e-12 * max(1.0, abs(det_c)) ** 2)

    def test_singlet_gram_is_abs_wedge_squared(self) -> None:
        rng = np.random.default_rng(45)
        c = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
        self.assertLessEqual(
            abs(ColorFiber.singlet_gram(c) - abs(ColorFiber.color_wedge(c))**2),
            1e-13 * max(1.0, abs(ColorFiber.color_wedge(c)) ** 2))

    def test_wedge_matches_exterior_algebra_top_sector(self) -> None:
        # Cross-representation: the SAME certificate through the Fock wedge
        # |psi> = a†(c1) a†(c2) a†(c3) |vac>:  ||psi||^2 = det(C†C) and psi
        # is a pure top-wedge (N=3) state.
        rng = np.random.default_rng(46)
        c = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
        creation = [sum(c[i, col] * ColorFiber.creation_matrix(i)
                        for i in range(3)) for col in range(3)]
        vac = np.zeros(8, dtype=complex)
        vac[0] = 1.0
        psi = creation[0] @ (creation[1] @ (creation[2] @ vac))
        norm2 = float(np.vdot(psi, psi).real)
        self.assertLessEqual(abs(norm2 - ColorFiber.singlet_gram(c)),
                             1e-13 * max(1.0, norm2))
        weights = ColorFiber.sector_weights(psi)
        self.assertEqual(weights.vacuum, 0.0)
        self.assertEqual(weights.quark, 0.0)
        self.assertEqual(weights.anti_triplet, 0.0)
        self.assertLessEqual(abs(weights.singlet - norm2), 1e-15 * norm2)

    def test_is_special_unitary_negative_controls(self) -> None:
        self.assertFalse(
            ColorFiber.is_special_unitary(2.0 * np.eye(3, dtype=complex)))
        # Unitary but det = omega != 1.
        u = np.diag([OMEGA_ALG, 1.0, 1.0])
        self.assertFalse(ColorFiber.is_special_unitary(u))
        self.assertTrue(ColorFiber.is_special_unitary(np.eye(3, dtype=complex)))


# ─── sector reads (weights only, no classification) ────────────────────────

class TestSectorReads(unittest.TestCase):
    def test_basis_state_weights_exact(self) -> None:
        for b in range(8):
            psi = np.zeros(8, dtype=complex)
            psi[b] = 1.0
            w = ColorFiber.sector_weights(psi)
            got = (w.vacuum, w.quark, w.anti_triplet, w.singlet)
            expected = [0.0, 0.0, 0.0, 0.0]
            expected[bin(b).count("1")] = 1.0
            self.assertEqual(got, tuple(expected))

    def test_random_state_matches_masks_and_sums(self) -> None:
        rng = np.random.default_rng(51)
        psi = rng.normal(size=8) + 1j * rng.normal(size=8)
        w = ColorFiber.sector_weights(psi)
        for n, field in zip(range(4),
                            (w.vacuum, w.quark, w.anti_triplet, w.singlet)):
            mask = [bin(b).count("1") == n for b in range(8)]
            self.assertLessEqual(
                abs(field - float(np.sum(np.abs(psi[mask]) ** 2))), 1e-15)
        total = w.vacuum + w.quark + w.anti_triplet + w.singlet
        self.assertLessEqual(abs(total - float(np.vdot(psi, psi).real)),
                             1e-13)

    def test_one_particle_state_reads_quark_sector(self) -> None:
        rng = np.random.default_rng(52)
        v = rng.normal(size=3) + 1j * rng.normal(size=3)
        vac = np.zeros(8, dtype=complex)
        vac[0] = 1.0
        psi = sum(v[i] * ColorFiber.creation_matrix(i)
                  for i in range(3)) @ vac
        w = ColorFiber.sector_weights(psi)
        self.assertEqual(w.vacuum, 0.0)
        self.assertEqual(w.anti_triplet, 0.0)
        self.assertEqual(w.singlet, 0.0)
        self.assertLessEqual(
            abs(w.quark - float(np.linalg.norm(v) ** 2)), 1e-14)

    def test_diquark_pair_reads_anti_triplet_sector(self) -> None:
        # a_0† a_1† |vac> is a pure two-occupation (anti-triplet) read.
        vac = np.zeros(8, dtype=complex)
        vac[0] = 1.0
        psi = (ColorFiber.creation_matrix(0) @
               (ColorFiber.creation_matrix(1) @ vac))
        w = ColorFiber.sector_weights(psi)
        self.assertEqual((w.vacuum, w.quark, w.anti_triplet, w.singlet),
                         (0.0, 0.0, 1.0, 0.0))

    def test_weights_invariant_under_mode_relabeling(self) -> None:
        # Oriented edge relabeling acts by the SIGNED permutation unitary of
        # the #766 primitives; every occupation read is invariant.
        rng = np.random.default_rng(53)
        psi = rng.normal(size=8) + 1j * rng.normal(size=8)
        alg = ExteriorAlgebra(3)
        for perm in ([1, 2, 0], [2, 1, 0], [0, 2, 1]):
            u = dense(alg.mode_permutation_matrix_coo(perm))
            w0 = ColorFiber.sector_weights(psi)
            w1 = ColorFiber.sector_weights(u @ psi)
            for f0, f1 in zip(
                    (w0.vacuum, w0.quark, w0.anti_triplet, w0.singlet),
                    (w1.vacuum, w1.quark, w1.anti_triplet, w1.singlet)):
                self.assertLessEqual(abs(f0 - f1), 1e-15)

    def test_size_error(self) -> None:
        with self.assertRaises(ValueError):
            ColorFiber.sector_weights(np.zeros(4, dtype=complex))


# ─── the calibrated anchor: oracle, atlas, calibration ─────────────────────

class TestAnchorOracle(unittest.TestCase):
    def test_literal_triangle_oracle_reaches_one(self) -> None:
        rng = np.random.default_rng(61)
        w = np.array([2.0, 0.5, 1.25])
        phi = orthonormal_band(rng, 3, w)
        anchor = ColorAnchor([OrientedTriangle([0, 1, 2], [1, 1, 1])])
        p = anchor.evaluate(phi, w)
        self.assertLessEqual(abs(p.score - 1.0), 1e-13)
        self.assertLessEqual(abs(p.max_term - 1.0), 1e-13)
        self.assertEqual(p.max_term_index, 0)
        self.assertLessEqual(abs(p.participation_ratio - 1.0), 1e-13)
        self.assertTrue(p.positive_regime)
        self.assertEqual(p.krein_signatures, [[3, 0, 0]])
        self.assertLessEqual(p.calibration_margin, 1e-12)
        self.assertLessEqual(p.frame_gram_residual, 1e-12)
        self.assertEqual(p.weighting_id, "uniform")
        self.assertEqual(p.weights, [1.0])
        # The attached #764 certificate (shared vocabulary, no bare read):
        # closed-form given the verified |W|-orthonormal premise on a
        # decoupled diagonal weight.
        cob = tessera.cobordism
        cert = p.certificate
        self.assertEqual(cert.grade, cob.CertificateGrade.StructureExact)
        self.assertEqual(cert.domain, cob.CertificateDomain.Static)
        self.assertEqual(cert.regime,
                         cob.CertificateRegime.PositiveSemidefinite)
        self.assertTrue(cert.holds())
        self.assertLessEqual(cert.residual, 1e-12)
        self.assertEqual(cert.tolerance, 1e-9)
        # Unmeasured quantities are NaN, never zero (#764 convention).
        self.assertTrue(math.isnan(cert.conditioning))
        self.assertTrue(math.isnan(cert.dense_reference_error))

    def test_oracle_exact_algebraic_fixture_f3(self) -> None:
        # Phi = F3 on three unit-weight edges: A_tau IS F3, |det A|^2 = 1.
        f = ColorFiber.fourier_frame()
        w = np.ones(3)
        tri = OrientedTriangle([0, 1, 2], [1, 1, 1])
        a_tau = ColorAnchor.anchor_matrix(f, w, tri)
        self.assertLessEqual(np.max(np.abs(a_tau - f)), 1e-15)
        anchor = ColorAnchor([tri])
        p = anchor.evaluate(f, w)
        # "Reaches one exactly" at the machine-precision bar: unit weights
        # make A_tau = F3 bitwise, so the score is 1 to double round-off.
        self.assertLessEqual(abs(p.score - 1.0), 1e-15)
        # The determinant phase equals arg det F3 (algebraic value -i for
        # this DFT: det F3 = (omega^2 - omega)(...)/3^{3/2} — compare to
        # the independently computed reference).
        ref_phase = float(np.angle(np.linalg.det(np.asarray(f))))
        self.assertLessEqual(abs(np.exp(1j * p.det_phases[0]) -
                                 np.exp(1j * ref_phase)), 1e-13)

    def test_calibrated_score_never_exceeds_one(self) -> None:
        # Property test over random in-domain fixtures: exact-arithmetic
        # bound score <= 1; floating evaluation may exceed it only by
        # round-off (final-boundary comparison at 1e-13).
        rng = np.random.default_rng(62)
        n_edges = 12
        for trial in range(30):
            w = rng.uniform(0.2, 3.0, size=n_edges)
            phi = orthonormal_band(rng, n_edges, w)
            tris = []
            for _ in range(rng.integers(1, 7)):
                edges = rng.choice(n_edges, size=3, replace=False)
                signs = rng.choice([-1, 1], size=3)
                tris.append(OrientedTriangle(
                    [int(e) for e in edges], [int(s) for s in signs]))
            anchor = ColorAnchor(tris)
            p = anchor.evaluate(phi, w)
            self.assertGreaterEqual(p.score, -1e-14, msg=f"trial {trial}")
            self.assertLessEqual(p.score, 1.0 + 1e-13, msg=f"trial {trial}")
            for t in p.terms:
                self.assertLessEqual(t, 1.0 + 1e-13)
                self.assertGreaterEqual(t, 0.0)
            self.assertLessEqual(p.calibration_margin, 1e-12,
                                 msg="diagonal weights are decoupled")

    def test_extended_atlas_matches_independent_reference(self) -> None:
        # The production case: an extended fiber anchored by an atlas of
        # overlapping oriented triangles.  Full profile against the
        # standalone NumPy evaluator.
        rng = np.random.default_rng(63)
        n_edges = 6
        w = rng.uniform(0.5, 2.0, size=n_edges)
        phi = orthonormal_band(rng, n_edges, w)
        tri_spec = [((0, 1, 2), (1, 1, 1)), ((2, 3, 4), (1, -1, 1)),
                    ((0, 3, 5), (-1, 1, 1)), ((1, 4, 5), (1, 1, -1))]
        tris = [OrientedTriangle(list(e), list(s)) for e, s in tri_spec]
        conv = [0.4, 0.3, 0.2, 0.1]
        anchor = ColorAnchor(tris, conv)
        p = anchor.evaluate(phi, w)

        score, terms, pr, phases, coherence = ref_profile(
            phi, w, tri_spec, conv)
        self.assertLessEqual(abs(p.score - score), 1e-12)
        self.assertLessEqual(np.max(np.abs(np.array(p.terms) - terms)),
                             1e-12)
        self.assertLessEqual(abs(p.participation_ratio - pr), 1e-10)
        self.assertLessEqual(abs(p.phase_coherence - coherence), 1e-10)
        self.assertLessEqual(
            abs(p.phase_dispersion - (1.0 - coherence)), 1e-10)
        for got, want, t in zip(p.det_phases, phases, terms):
            if t > 1e-20:
                self.assertLessEqual(
                    abs(np.exp(1j * got) - np.exp(1j * want)), 1e-9)
        self.assertLessEqual(abs(p.max_term - float(np.max(terms))), 1e-12)
        self.assertEqual(p.max_term_index, int(np.argmax(terms)))
        self.assertEqual(p.weighting_id, "declared")
        self.assertEqual(p.weights, conv)
        # A genuinely extended read: several triangles participate and the
        # score sits strictly inside the calibrated interval.
        self.assertGreater(p.participation_ratio, 1.0)
        self.assertGreater(p.score, 0.0)
        self.assertLess(p.score, 1.0)

    def test_anchor_matrix_matches_reference(self) -> None:
        rng = np.random.default_rng(64)
        n_edges = 5
        w = rng.uniform(0.1, 2.0, size=n_edges)
        phi = rng.normal(size=(n_edges, 3)) + 1j * rng.normal(
            size=(n_edges, 3))
        tri = OrientedTriangle([4, 0, 2], [-1, 1, -1])
        a_tau = ColorAnchor.anchor_matrix(phi, w, tri)
        rows = np.array([-1 * phi[4, :], +1 * phi[0, :], -1 * phi[2, :]])
        ref = np.diag(np.sqrt(w[[4, 0, 2]])) @ rows
        self.assertLessEqual(np.max(np.abs(a_tau - ref)), 1e-15)

    def test_orthonormalize_frame_enters_domain(self) -> None:
        rng = np.random.default_rng(65)
        w = rng.uniform(0.2, 4.0, size=7)
        phi = orthonormal_band(rng, 7, w)
        gram = phi.conj().T @ np.diag(np.abs(w)) @ phi
        self.assertLessEqual(np.max(np.abs(gram - np.eye(3))), 1e-13)

    def test_orthonormalize_rejects_rank_deficient(self) -> None:
        frame = np.zeros((4, 3), dtype=complex)
        frame[:, 0] = [1.0, 0.0, 0.0, 0.0]
        frame[:, 1] = [0.0, 1.0, 0.0, 0.0]
        frame[:, 2] = [1.0, 1.0, 0.0, 0.0]  # dependent
        with self.assertRaises(ValueError):
            ColorAnchor.orthonormalize_frame(frame, np.ones(4))


class TestAnchorNegativeControls(unittest.TestCase):
    def test_unanchored_band_off_the_faces_fails(self) -> None:
        # An abstract rank-three band supported AWAY from every declared
        # anchoring face: the anchor read is exactly zero — the band FAILS
        # the anchor.  The undefined phase datum is NaN, never zero.
        rng = np.random.default_rng(71)
        w = np.ones(9)
        frame = np.zeros((9, 3), dtype=complex)
        block = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
        frame[6:9, :] = block  # support only on edges {6, 7, 8}
        phi = ColorAnchor.orthonormalize_frame(frame, w)
        anchor = ColorAnchor([OrientedTriangle([0, 1, 2], [1, 1, 1]),
                              OrientedTriangle([3, 4, 5], [1, 1, 1])])
        p = anchor.evaluate(phi, w)
        self.assertEqual(p.score, 0.0)
        self.assertEqual(list(p.terms), [0.0, 0.0])
        self.assertEqual(p.participation_ratio, 0.0)
        self.assertEqual(p.max_term, 0.0)
        for phase in p.det_phases:
            self.assertTrue(math.isnan(phase))
        self.assertTrue(math.isnan(p.phase_coherence))
        self.assertTrue(math.isnan(p.phase_dispersion))

    def test_degenerate_band_on_the_faces_fails(self) -> None:
        # Support ON the faces but with no alternating volume (two equal
        # frame rows on the triangle): |det A_tau|^2 collapses.
        rng = np.random.default_rng(72)
        w = np.full(4, 0.7)
        frame = rng.normal(size=(4, 3)) + 1j * rng.normal(size=(4, 3))
        frame[1, :] = frame[0, :]  # duplicate one-chain rows on the face
        phi = ColorAnchor.orthonormalize_frame(frame, w)
        # Right-multiplication preserves the duplicated rows exactly.
        self.assertTrue(np.array_equal(phi[0, :], phi[1, :]))
        anchor = ColorAnchor([OrientedTriangle([0, 1, 2], [1, 1, 1])])
        p = anchor.evaluate(phi, w)
        self.assertLessEqual(p.score, 1e-28)

    def test_score_ordering_oracle_extended_unanchored(self) -> None:
        rng = np.random.default_rng(73)
        # Oracle.
        w3 = np.ones(3)
        oracle = ColorAnchor([OrientedTriangle([0, 1, 2], [1, 1, 1])])
        s_oracle = oracle.evaluate(orthonormal_band(rng, 3, w3), w3).score
        # Extended.
        w6 = np.ones(6)
        ext = ColorAnchor([OrientedTriangle([0, 1, 2], [1, 1, 1]),
                           OrientedTriangle([2, 3, 4], [1, 1, 1]),
                           OrientedTriangle([0, 4, 5], [1, 1, 1])])
        s_ext = ext.evaluate(orthonormal_band(rng, 6, w6), w6).score
        # Unanchored.
        w9 = np.ones(9)
        frame = np.zeros((9, 3), dtype=complex)
        frame[6:9, :] = np.eye(3)
        un = ColorAnchor([OrientedTriangle([0, 1, 2], [1, 1, 1])])
        s_un = un.evaluate(ColorAnchor.orthonormalize_frame(frame, w9),
                           w9).score
        self.assertLessEqual(abs(s_oracle - 1.0), 1e-13)
        self.assertGreater(s_oracle, s_ext)
        self.assertGreater(s_ext, s_un)
        self.assertEqual(s_un, 0.0)

    def test_unnormalized_frame_rejected(self) -> None:
        # The calibration identity is undefined outside the
        # |W|-orthonormal domain: a raw frame must be rejected, not scored.
        rng = np.random.default_rng(74)
        frame = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
        anchor = ColorAnchor([OrientedTriangle([0, 1, 2], [1, 1, 1])])
        with self.assertRaisesRegex(ValueError, "orthonormal"):
            anchor.evaluate(frame, np.ones(3))

    def test_empty_atlas_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ColorAnchor([])

    def test_bad_triangles_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ColorAnchor([OrientedTriangle([0, 0, 1], [1, 1, 1])])
        with self.assertRaises(ValueError):
            ColorAnchor([OrientedTriangle([0, 1, 2], [1, 2, 1])])

    def test_shape_and_range_errors(self) -> None:
        anchor = ColorAnchor([OrientedTriangle([0, 1, 5], [1, 1, 1])])
        phi = ColorAnchor.orthonormalize_frame(
            np.eye(3, dtype=complex), np.ones(3))
        with self.assertRaises(ValueError):  # edge 5 out of range
            anchor.evaluate(phi, np.ones(3))
        anchor2 = ColorAnchor([OrientedTriangle([0, 1, 2], [1, 1, 1])])
        with self.assertRaises(ValueError):  # weights length mismatch
            anchor2.evaluate(phi, np.ones(4))


class TestAnchorDeclaredWeighting(unittest.TestCase):
    def fixture(self):
        rng = np.random.default_rng(81)
        w = np.ones(6)
        phi = orthonormal_band(rng, 6, w)
        tris = [OrientedTriangle([0, 1, 2], [1, 1, 1]),
                OrientedTriangle([3, 4, 5], [1, 1, 1])]
        return phi, w, tris

    def test_post_hoc_weight_selection_rejected(self) -> None:
        phi, w, tris = self.fixture()
        anchor = ColorAnchor(tris)
        self.assertFalse(anchor.sealed())
        anchor.evaluate(phi, w)
        self.assertTrue(anchor.sealed())
        with self.assertRaisesRegex(RuntimeError, "post-hoc"):
            anchor.declare_weights([1.0, 0.0])

    def test_failed_evaluate_still_seals(self) -> None:
        # Even a REJECTED read has examined the data: the weighting seals.
        phi, w, tris = self.fixture()
        anchor = ColorAnchor(tris)
        with self.assertRaises(ValueError):
            anchor.evaluate(np.asarray(phi) * 2.0, w)  # not orthonormal
        self.assertTrue(anchor.sealed())
        with self.assertRaisesRegex(RuntimeError, "post-hoc"):
            anchor.declare_weights([1.0, 0.0])

    def test_declaration_before_data_is_allowed(self) -> None:
        phi, w, tris = self.fixture()
        anchor = ColorAnchor(tris)
        self.assertEqual(anchor.weighting_id(), "uniform")
        self.assertEqual(anchor.weights(), [0.5, 0.5])
        anchor.declare_weights([0.75, 0.25])
        self.assertEqual(anchor.weighting_id(), "declared")
        p = anchor.evaluate(phi, w)
        self.assertEqual(p.weighting_id, "declared")
        self.assertEqual(p.weights, [0.75, 0.25])
        self.assertLessEqual(
            abs(p.score - (0.75 * p.terms[0] + 0.25 * p.terms[1])), 1e-15)

    def test_non_convex_weightings_rejected(self) -> None:
        _, _, tris = self.fixture()
        with self.assertRaises(ValueError):
            ColorAnchor(tris, [0.5, 0.6])  # sum != 1
        with self.assertRaises(ValueError):
            ColorAnchor(tris, [1.5, -0.5])  # negative
        with self.assertRaises(ValueError):
            ColorAnchor(tris, [1.0])  # wrong length
        anchor = ColorAnchor(tris)
        with self.assertRaises(ValueError):
            anchor.declare_weights([0.2, 0.2])

    def test_uniform_weighting_is_the_default_declaration(self) -> None:
        phi, w, tris = self.fixture()
        anchor = ColorAnchor(tris)
        p = anchor.evaluate(phi, w)
        self.assertEqual(p.weighting_id, "uniform")
        self.assertEqual(p.weights, [0.5, 0.5])
        self.assertLessEqual(
            abs(p.score - 0.5 * (p.terms[0] + p.terms[1])), 1e-15)


class TestAnchorInvariances(unittest.TestCase):
    def fixture(self):
        rng = np.random.default_rng(91)
        n_edges = 6
        w = rng.uniform(0.5, 2.0, size=n_edges)
        phi = orthonormal_band(rng, n_edges, w)
        tri_spec = [((0, 1, 2), (1, 1, 1)), ((2, 3, 4), (1, -1, 1)),
                    ((0, 3, 5), (-1, 1, 1))]
        tris = [OrientedTriangle(list(e), list(s)) for e, s in tri_spec]
        return rng, np.asarray(phi), w, tris

    def profile(self, phi, w, tris, weights=None):
        anchor = ColorAnchor(tris) if weights is None else ColorAnchor(
            tris, weights)
        return anchor.evaluate(phi, w)

    def test_in_band_su3_frame_change_invariant(self) -> None:
        rng, phi, w, tris = self.fixture()
        p0 = self.profile(phi, w, tris)
        for _ in range(5):
            g = random_su3(rng)
            p1 = self.profile(phi @ g, w, tris)
            self.assertLessEqual(abs(p1.score - p0.score), 1e-12)
            self.assertLessEqual(
                np.max(np.abs(np.array(p1.terms) - np.array(p0.terms))),
                1e-12)
            self.assertLessEqual(
                abs(p1.participation_ratio - p0.participation_ratio), 1e-9)
            self.assertLessEqual(
                abs(p1.phase_coherence - p0.phase_coherence), 1e-9)
            # det g = 1: every determinant PHASE is itself invariant.
            for a, b in zip(p1.det_phases, p0.det_phases):
                self.assertLessEqual(
                    abs(np.exp(1j * a) - np.exp(1j * b)), 1e-9)

    def test_full_u3_change_shifts_all_phases_by_det(self) -> None:
        rng, phi, w, tris = self.fixture()
        p0 = self.profile(phi, w, tris)
        theta = 0.813
        g = np.exp(1j * theta / 3.0) * random_su3(rng)  # det g = e^{i theta}
        p1 = self.profile(phi @ g, w, tris)
        self.assertLessEqual(abs(p1.score - p0.score), 1e-12)
        for a, b in zip(p1.det_phases, p0.det_phases):
            self.assertLessEqual(
                abs(np.exp(1j * (a - b - theta)) - 1.0), 1e-9)
        self.assertLessEqual(abs(p1.phase_coherence - p0.phase_coherence),
                             1e-9)

    def test_oriented_edge_relabeling_exact_invariance(self) -> None:
        _, phi, w, tris = self.fixture()
        p0 = self.profile(phi, w, tris)
        perm = [3, 5, 0, 1, 4, 2]  # old edge e -> new row perm[e]
        n = len(perm)
        phi_p = np.zeros_like(phi)
        w_p = np.zeros_like(w)
        for e in range(n):
            phi_p[perm[e], :] = phi[e, :]
            w_p[perm[e]] = w[e]
        tris_p = [OrientedTriangle([perm[e] for e in t.edges],
                                   list(t.signs)) for t in tris]
        p1 = self.profile(phi_p, w_p, tris_p)
        # Pure reindexing: the same numbers flow through the same
        # operations — bitwise equality, not just tolerance.
        self.assertEqual(p1.score, p0.score)
        self.assertEqual(list(p1.terms), list(p0.terms))
        self.assertEqual(list(p1.det_phases), list(p0.det_phases))
        self.assertEqual(p1.phase_coherence, p0.phase_coherence)
        self.assertEqual(p1.krein_signatures, p0.krein_signatures)

    def test_stored_orientation_reversal_exact_invariance(self) -> None:
        # Reversing a stored edge orientation negates its frame row and
        # flips the incidence sign in every touching triangle descriptor:
        # the effective oriented boundary is unchanged.
        _, phi, w, tris = self.fixture()
        p0 = self.profile(phi, w, tris)
        for flip_edge in range(phi.shape[0]):
            phi_f = phi.copy()
            phi_f[flip_edge, :] = -phi_f[flip_edge, :]
            tris_f = []
            for t in tris:
                signs = list(t.signs)
                for k, e in enumerate(t.edges):
                    if e == flip_edge:
                        signs[k] = -signs[k]
                tris_f.append(OrientedTriangle(list(t.edges), signs))
            p1 = self.profile(phi_f, w, tris_f)
            self.assertEqual(p1.score, p0.score)
            self.assertEqual(list(p1.terms), list(p0.terms))
            self.assertEqual(list(p1.det_phases), list(p0.det_phases))

    def test_cyclic_triangle_rotation_is_even(self) -> None:
        # The orientation fixes the boundary ordering up to a CYCLIC (even)
        # permutation: det A_tau itself is invariant.
        _, phi, w, tris = self.fixture()
        t = tris[0]
        base = ColorAnchor.anchor_matrix(phi, w, t)
        rot = OrientedTriangle([t.edges[1], t.edges[2], t.edges[0]],
                               [t.signs[1], t.signs[2], t.signs[0]])
        rotated = ColorAnchor.anchor_matrix(phi, w, rot)
        d0, d1 = np.linalg.det(base), np.linalg.det(rotated)
        self.assertLessEqual(abs(d1 - d0), 1e-14 * max(1.0, abs(d0)))

    def test_odd_permutation_flips_the_determinant_only(self) -> None:
        # An odd reordering is the OPPOSITE orientation: det negates
        # (phase shifts by pi), |det|^2 unchanged.
        _, phi, w, tris = self.fixture()
        t = tris[0]
        base = ColorAnchor.anchor_matrix(phi, w, t)
        swap = OrientedTriangle([t.edges[1], t.edges[0], t.edges[2]],
                                [t.signs[1], t.signs[0], t.signs[2]])
        swapped = ColorAnchor.anchor_matrix(phi, w, swap)
        d0, d1 = np.linalg.det(base), np.linalg.det(swapped)
        self.assertLessEqual(abs(d1 + d0), 1e-14 * max(1.0, abs(d0)))
        self.assertLessEqual(abs(abs(d1) ** 2 - abs(d0) ** 2),
                             1e-13 * max(1.0, abs(d0) ** 2))


class TestAnchorSignedAndMatrixWeights(unittest.TestCase):
    def test_signed_sector_krein_reported_separately(self) -> None:
        # One timelike-signed edge weight: the score still restricts with
        # |W_tau|^{1/2} (identical to the |w| run), and the restricted
        # block's Krein signature is reported separately per triangle.
        rng = np.random.default_rng(101)
        w_signed = np.array([1.5, -0.8, 1.1, 0.9, 1.3, 0.6])
        phi = orthonormal_band(rng, 6, w_signed)
        tris = [OrientedTriangle([0, 1, 2], [1, 1, 1]),
                OrientedTriangle([3, 4, 5], [1, 1, 1])]
        p_signed = ColorAnchor(tris).evaluate(phi, w_signed)
        p_abs = ColorAnchor(tris).evaluate(phi, np.abs(w_signed))
        self.assertEqual(p_signed.score, p_abs.score)
        self.assertEqual(list(p_signed.terms), list(p_abs.terms))
        self.assertFalse(p_signed.positive_regime)
        self.assertTrue(p_abs.positive_regime)
        self.assertEqual(p_signed.krein_signatures, [[2, 0, 1], [3, 0, 0]])
        self.assertEqual(p_abs.krein_signatures, [[3, 0, 0], [3, 0, 0]])
        cob = tessera.cobordism
        self.assertEqual(p_signed.certificate.regime,
                         cob.CertificateRegime.HermitianIndefinite)
        self.assertEqual(p_abs.certificate.regime,
                         cob.CertificateRegime.PositiveSemidefinite)
        self.assertTrue(p_signed.certificate.holds())

    def test_zero_weight_reports_zero_mode(self) -> None:
        rng = np.random.default_rng(102)
        w = np.array([1.0, 0.0, 1.0, 1.0])
        frame = np.zeros((4, 3), dtype=complex)
        frame[[0, 2, 3], :] = rng.normal(size=(3, 3)) + 1j * rng.normal(
            size=(3, 3))
        phi = ColorAnchor.orthonormalize_frame(frame, w)
        p = ColorAnchor([OrientedTriangle([0, 1, 2], [1, 1, 1])]).evaluate(
            phi, w)
        self.assertEqual(p.krein_signatures, [[2, 1, 0]])
        self.assertFalse(p.positive_regime)

    def test_matrix_weight_diagonal_agrees_with_vector_path(self) -> None:
        rng = np.random.default_rng(103)
        w = rng.uniform(0.5, 2.0, size=5)
        phi = orthonormal_band(rng, 5, w)
        tris = [OrientedTriangle([0, 1, 2], [1, -1, 1]),
                OrientedTriangle([1, 3, 4], [-1, 1, 1])]
        p_vec = ColorAnchor(tris).evaluate(phi, w)
        p_mat = ColorAnchor(tris).evaluate_matrix(
            phi, np.diag(w).astype(complex))
        self.assertLessEqual(abs(p_vec.score - p_mat.score), 1e-12)
        self.assertLessEqual(
            np.max(np.abs(np.array(p_vec.terms) - np.array(p_mat.terms))),
            1e-12)
        self.assertEqual(p_vec.krein_signatures, p_mat.krein_signatures)
        # Grades name the claim class honestly: closed-form structure-exact
        # on the diagonal path, certified-numerical on the eigen-modulus
        # matrix path.
        cob = tessera.cobordism
        self.assertEqual(p_vec.certificate.grade,
                         cob.CertificateGrade.StructureExact)
        self.assertEqual(p_mat.certificate.grade,
                         cob.CertificateGrade.CertifiedNumerical)
        self.assertTrue(p_mat.certificate.holds())

    def test_matrix_weight_requires_hermitian(self) -> None:
        rng = np.random.default_rng(104)
        phi = orthonormal_band(rng, 4, np.ones(4))
        bad = np.eye(4, dtype=complex)
        bad[0, 1] = 1.0  # not Hermitian
        anchor = ColorAnchor([OrientedTriangle([0, 1, 2], [1, 1, 1])])
        with self.assertRaisesRegex(ValueError, "Hermitian"):
            anchor.evaluate_matrix(phi, bad)

    def test_coupled_matrix_weight_checked_not_assumed(self) -> None:
        # A coupled Hermitian weight: the profile is still an exact
        # evaluation of A_tau = |W_tau|^{1/2} R_tau phi (cross-checked in
        # NumPy), and the <= 1 calibration is CHECKED via the reported
        # margin rather than assumed.
        rng = np.random.default_rng(105)
        n = 5
        base = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
        weight = base @ base.conj().T + 0.5 * np.eye(n)  # Hermitian PD
        frame = rng.normal(size=(n, 3)) + 1j * rng.normal(size=(n, 3))
        phi = ColorAnchor.orthonormalize_frame_matrix(frame, weight)
        tri_spec = [((0, 1, 2), (1, 1, 1)), ((2, 3, 4), (1, -1, 1))]
        tris = [OrientedTriangle(list(e), list(s)) for e, s in tri_spec]
        p = ColorAnchor(tris).evaluate_matrix(phi, weight)

        # Independent NumPy reference for the matrix path.
        terms_ref = []
        for edges, signs in tri_spec:
            s = np.diag(signs).astype(complex)
            block = s @ weight[np.ix_(list(edges), list(edges))] @ s
            lam, u = np.linalg.eigh(block)
            sqrt_mod = u @ np.diag(np.sqrt(np.abs(lam))) @ u.conj().T
            rows = np.array(
                [sg * phi[e, :] for e, sg in zip(edges, signs)])
            terms_ref.append(abs(np.linalg.det(sqrt_mod @ rows)) ** 2)
        self.assertLessEqual(
            np.max(np.abs(np.array(p.terms) - np.array(terms_ref))), 1e-11)
        self.assertLessEqual(
            abs(p.score - 0.5 * float(np.sum(terms_ref))), 1e-11)
        self.assertTrue(np.isfinite(p.calibration_margin))
        self.assertTrue(p.positive_regime)
        # The frame was |W|-orthonormalized, so the domain certificate holds.
        self.assertLessEqual(p.frame_gram_residual, 1e-9)


class TestConstantAlgebraSelfCheck(unittest.TestCase):
    def test_verify_constant_algebra_at_round_off(self) -> None:
        # The startup-check contract (debug builds run this automatically;
        # every build can call it): the WHOLE constant algebra re-derives
        # within double round-off.
        self.assertLessEqual(ColorFiber.verify_constant_algebra(), 1e-12)

    def test_constant_algebra_certificate(self) -> None:
        # The same claim in the shared #764 vocabulary: AlgebraicallyExact,
        # holds, with the measured residual and the startup tolerance; the
        # re-derivation is deterministic so the residual matches the raw
        # call bitwise.
        cob = tessera.cobordism
        cert = ColorFiber.constant_algebra_certificate()
        self.assertEqual(cert.grade, cob.CertificateGrade.AlgebraicallyExact)
        self.assertEqual(cert.domain, cob.CertificateDomain.Static)
        self.assertTrue(cert.holds())
        self.assertEqual(cert.residual, ColorFiber.verify_constant_algebra())
        self.assertEqual(cert.tolerance, 1e-12)
        self.assertTrue(math.isnan(cert.conditioning))


# ─── #774: the singlet complement resolving 3 x 3bar = 1 + 8 ───────────────

class TestAdjointSingletProjector(unittest.TestCase):
    def test_matches_independent_construction(self) -> None:
        vec_i = np.eye(3, dtype=complex).reshape(9, order="F")
        ref = np.outer(vec_i, vec_i.conj()) / 3.0
        self.assertLessEqual(
            np.max(np.abs(ColorFiber.adjoint_singlet_projector() - ref)),
            1e-15)

    def test_bitwise_complement_of_the_octet_projector(self) -> None:
        # The implementation is LITERALLY I9 - P8: assert the delegation
        # bitwise, not just to tolerance.
        p1 = ColorFiber.adjoint_singlet_projector()
        p8 = ColorFiber.adjoint_octet_projector()
        self.assertTrue(np.array_equal(p1, np.eye(9, dtype=complex) - p8))

    def test_singlet_plus_octet_resolve_to_identity(self) -> None:
        # THE #774 acceptance identity: P1 + P8 = I9 on 3 x 3bar.
        p1 = ColorFiber.adjoint_singlet_projector()
        p8 = ColorFiber.adjoint_octet_projector()
        self.assertEqual(np.max(np.abs(p1 + p8 - np.eye(9))), 0.0)

    def test_projector_algebra(self) -> None:
        p1 = ColorFiber.adjoint_singlet_projector()
        self.assertLessEqual(np.max(np.abs(p1 - p1.conj().T)), 1e-15)
        self.assertLessEqual(np.max(np.abs(p1 @ p1 - p1)), 1e-15)
        self.assertLessEqual(abs(np.trace(p1) - 1.0), 1e-15)
        self.assertEqual(np.linalg.matrix_rank(p1), 1)

    def test_mutually_orthogonal_with_the_octet(self) -> None:
        p1 = ColorFiber.adjoint_singlet_projector()
        p8 = ColorFiber.adjoint_octet_projector()
        self.assertLessEqual(np.max(np.abs(p1 @ p8)), 1e-15)
        self.assertLessEqual(np.max(np.abs(p8 @ p1)), 1e-15)

    def test_extracts_the_trace_part(self) -> None:
        rng = np.random.default_rng(741)
        m = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
        p1 = ColorFiber.adjoint_singlet_projector()
        got = (p1 @ m.reshape(9, order="F")).reshape(3, 3, order="F")
        want = (np.trace(m) / 3.0) * np.eye(3)
        self.assertLessEqual(np.max(np.abs(got - want)), 1e-15)

    def test_octet_read_weights_are_the_projector_weights(self) -> None:
        # The 1 + 8 resolution and the Frobenius split are the SAME split:
        # octetRead weights equal ||P8 v||^2 and ||P1 v||^2.
        rng = np.random.default_rng(742)
        m = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
        v = m.reshape(9, order="F")
        read = ColorFiber.octet_read(m)
        p1 = ColorFiber.adjoint_singlet_projector()
        p8 = ColorFiber.adjoint_octet_projector()
        self.assertLessEqual(
            abs(read.octet - np.linalg.norm(p8 @ v) ** 2), 1e-13)
        self.assertLessEqual(
            abs(read.singlet - np.linalg.norm(p1 @ v) ** 2), 1e-13)


# ─── #774: the traceless even bilinears on Fock space ──────────────────────

class TestOctetBilinears(unittest.TestCase):
    def _jw_number(self) -> np.ndarray:
        return sum(jw_creation(k, 3) @ jw_annihilation(k, 3)
                   for k in range(3))

    def test_matches_independent_jordan_wigner_reference(self) -> None:
        number = self._jw_number()
        for i in range(3):
            for j in range(3):
                ref = jw_creation(i, 3) @ jw_annihilation(j, 3)
                if i == j:
                    ref = ref - number / 3.0
                self.assertLessEqual(
                    np.max(np.abs(ColorFiber.octet_bilinear(i, j) - ref)),
                    1e-15, msg=f"T_{i}{j}")

    def test_delegation_identity_is_bitwise(self) -> None:
        # T_ij == d_gamma(traceless_part(matrix_unit(i, j))) -- the literal
        # composition documented in the header.
        for i in range(3):
            for j in range(3):
                want = ColorFiber.d_gamma(ColorFiber.traceless_part(
                    ColorFiber.matrix_unit(i, j)))
                self.assertTrue(np.array_equal(
                    ColorFiber.octet_bilinear(i, j), want))

    def test_diagonal_family_sums_to_zero(self) -> None:
        # sum_i T_ii = N - 3 (N/3): zero up to the dGamma accumulation
        # rounding (double round-off; fl(1/3) sums in operator assembly).
        total = sum(ColorFiber.octet_bilinear(i, i) for i in range(3))
        self.assertLessEqual(np.max(np.abs(total)), 1e-15)

    def test_even_fermion_parity(self) -> None:
        # T_ij conserves N, hence commutes with (-1)^N: the traceless
        # bilinears are EVEN elements of the graded algebra (whitepaper
        # "Fock space as an inductive limit of interactions").
        parity = dense(ExteriorAlgebra(3).parity_matrix_coo())
        for i in range(3):
            for j in range(3):
                t = ColorFiber.octet_bilinear(i, j)
                self.assertEqual(np.max(np.abs(t @ parity - parity @ t)),
                                 0.0)

    def test_triplet_restriction_is_the_traceless_matrix_unit(self) -> None:
        for i in range(3):
            for j in range(3):
                got = ColorFiber.restrict_to_triplet(
                    ColorFiber.octet_bilinear(i, j))
                want = ColorFiber.traceless_part(ColorFiber.matrix_unit(i, j))
                self.assertLessEqual(np.max(np.abs(got - want)), 1e-15)

    def test_bilinears_span_the_octet(self) -> None:
        # The nine T_ij restrict to the nine traceless matrix units, whose
        # span is EXACTLY the 8-dimensional octet = range(P8).
        vecs = np.column_stack([
            ColorFiber.restrict_to_triplet(
                ColorFiber.octet_bilinear(i, j)).reshape(9, order="F")
            for i in range(3) for j in range(3)])
        self.assertEqual(np.linalg.matrix_rank(vecs), 8)
        p8 = ColorFiber.adjoint_octet_projector()
        self.assertLessEqual(np.max(np.abs(p8 @ vecs - vecs)), 1e-15)

    def test_gellmann_bilinears_are_octet_combinations(self) -> None:
        # d_gamma(lambda_a) is a combination of the T_ij (lambda_a is
        # traceless): d_gamma(lambda_a) = sum_ij (lambda_a)_ij T_ij.
        for a in range(1, 9):
            lam = ColorFiber.gell_mann(a)
            combo = sum(lam[i, j] * ColorFiber.octet_bilinear(i, j)
                        for i in range(3) for j in range(3))
            self.assertLessEqual(
                np.max(np.abs(combo - ColorFiber.d_gamma(lam))), 1e-15)

    def test_index_validation(self) -> None:
        with self.assertRaises(ValueError):
            ColorFiber.octet_bilinear(3, 0)
        with self.assertRaises(ValueError):
            ColorFiber.octet_bilinear(0, 5)


# ─── #774: the adjoint quadratic Casimir C = 3 P8 ──────────────────────────

class TestAdjointCasimir(unittest.TestCase):
    def test_casimir_matrix_is_three_times_the_octet_projector(self) -> None:
        # THE identity: sum_a ad(lambda_a/2)^2 = C2(adjoint) P8 = 3 P8.
        c = ColorFiber.adjoint_casimir_matrix()
        p8 = ColorFiber.adjoint_octet_projector()
        self.assertLessEqual(np.max(np.abs(c - 3.0 * p8)), 1e-14)

    def test_casimir_matrix_matches_independent_commutator_sum(self) -> None:
        # Independent NumPy construction from the hardcoded Gell-Mann
        # table: K_a vec(M) = vec([lambda_a/2, M]) via the Kronecker rule
        # vec(AMB) = (B^T kron A) vec(M).
        eye = np.eye(3, dtype=complex)
        ref = np.zeros((9, 9), dtype=complex)
        for a in range(1, 9):
            half = GELL_MANN_REF[a] / 2.0
            k = np.kron(eye.T, half) - np.kron(half.T, eye)
            ref = ref + k @ k
        self.assertLessEqual(
            np.max(np.abs(ColorFiber.adjoint_casimir_matrix() - ref)), 1e-14)

    def test_rayleigh_on_generators_identity_and_zero(self) -> None:
        for a in range(1, 9):
            self.assertLessEqual(
                abs(ColorFiber.adjoint_casimir(ColorFiber.gell_mann(a)) - 3.0),
                1e-14)
        self.assertLessEqual(
            abs(ColorFiber.adjoint_casimir(np.eye(3, dtype=complex))), 1e-14)
        self.assertTrue(np.isnan(
            ColorFiber.adjoint_casimir(np.zeros((3, 3), dtype=complex))))

    def test_rayleigh_is_three_times_the_octet_fraction(self) -> None:
        rng = np.random.default_rng(743)
        m = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
        read = ColorFiber.octet_read(m)
        fraction = read.octet / (read.octet + read.singlet)
        self.assertLessEqual(
            abs(ColorFiber.adjoint_casimir(m) - 3.0 * fraction), 1e-13)

    def test_su3_invariance(self) -> None:
        rng = np.random.default_rng(744)
        for _ in range(5):
            g = random_su3(rng)
            self.assertTrue(ColorFiber.is_special_unitary(g))
            m = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
            self.assertLessEqual(
                abs(ColorFiber.adjoint_casimir(g @ m @ g.conj().T) -
                    ColorFiber.adjoint_casimir(m)), 1e-12)


# ─── #774: octet generators transform in the adjoint under SU(3) ───────────

class TestAdjointTransformation(unittest.TestCase):
    def test_traceless_projection_commutes_with_conjugation(self) -> None:
        # The octet is an invariant subspace: traceless_part(g M g^dag) ==
        # g traceless_part(M) g^dag for every certified g in SU(3).
        rng = np.random.default_rng(745)
        for _ in range(5):
            g = random_su3(rng)
            m = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
            got = ColorFiber.traceless_part(g @ m @ g.conj().T)
            want = g @ ColorFiber.traceless_part(m) @ g.conj().T
            self.assertLessEqual(np.max(np.abs(got - want)), 1e-13)

    def test_generators_transform_by_a_real_orthogonal_adjoint(self) -> None:
        # g lambda_a g^dag = sum_b R_ba lambda_b with R_ab =
        # Tr(lambda_a g lambda_b g^dag)/2 REAL, orthogonal, det +1 -- the
        # adjoint (8-dimensional) representation of a random certified
        # SU(3) element (property test over several draws).
        rng = np.random.default_rng(746)
        lams = [ColorFiber.gell_mann(a) for a in range(1, 9)]
        for _ in range(5):
            g = random_su3(rng)
            self.assertTrue(ColorFiber.is_special_unitary(g))
            r = np.zeros((8, 8), dtype=complex)
            for a in range(8):
                for b in range(8):
                    r[a, b] = np.trace(
                        lams[a] @ g @ lams[b] @ g.conj().T) / 2.0
            self.assertLessEqual(np.max(np.abs(r.imag)), 1e-12)
            r = r.real
            self.assertLessEqual(np.max(np.abs(r @ r.T - np.eye(8))), 1e-12)
            self.assertLessEqual(abs(np.linalg.det(r) - 1.0), 1e-11)
            # the transformation law itself, generator by generator
            for b in range(8):
                got = g @ lams[b] @ g.conj().T
                want = sum(r[a, b] * lams[a] for a in range(8))
                self.assertLessEqual(np.max(np.abs(got - want)), 1e-12)

    def test_octet_projector_commutes_with_the_conjugation_action(self) -> None:
        # vec(g M g^dag) = (conj(g) kron g) vec(M): P8 commutes with the
        # conjugation operator (and so does P1) -- the resolution 1 + 8 is
        # SU(3)-equivariant.
        rng = np.random.default_rng(747)
        p8 = ColorFiber.adjoint_octet_projector()
        p1 = ColorFiber.adjoint_singlet_projector()
        for _ in range(3):
            g = random_su3(rng)
            k = np.kron(g.conj(), g)
            self.assertLessEqual(np.max(np.abs(p8 @ k - k @ p8)), 1e-13)
            self.assertLessEqual(np.max(np.abs(p1 @ k - k @ p1)), 1e-13)

    def test_consistent_with_fiber_connection_adjoint_image(self) -> None:
        # The #770 faithful PU(3) image consumes the SAME #767 projector:
        # Ad(g) (P8 vec M) == P8 vec(g M g^dag) -- cross-kernel agreement,
        # and Ad is center-blind (Ad(omega g) = Ad(g)).
        rng = np.random.default_rng(748)
        conn = tessera.observables.FiberConnection
        for _ in range(3):
            g = random_su3(rng)
            ad = conn.adjoint_representation(g)
            m = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
            p8 = ColorFiber.adjoint_octet_projector()
            got = ad @ (p8 @ m.reshape(9, order="F"))
            want = p8 @ (g @ m @ g.conj().T).reshape(9, order="F")
            self.assertLessEqual(np.max(np.abs(got - want)), 1e-12)
            omega = ColorFiber.omega()
            self.assertLessEqual(
                np.max(np.abs(conn.adjoint_representation(omega * g) - ad)),
                1e-12)


# ─── #808: the coherence is an OVERLAP datum ───────────────────────────────

class TestAnchorOverlapCoherence(unittest.TestCase):
    """The whitepaper and the header both say the determinant-phase
    coherence is recorded on OVERLAPPING oriented triangles.  The resultant
    is restricted to triangles that genuinely share a boundary edge with
    another declared triangle -- the only sharing relation an
    OrientedTriangle atlas determines."""

    def _band(self, seed, n_edges, w):
        rng = np.random.default_rng(seed)
        return orthonormal_band(rng, n_edges, w)

    def test_a_disjoint_atlas_reports_no_overlap_coherence(self):
        # Two triangles on DISJOINT edge rows, both with a nonzero
        # determinant: nothing overlaps, so there is no overlap datum and
        # the coherence is UNKNOWN (NaN), never a value with no overlap
        # content behind it.
        w = np.array([1.3, 0.7, 1.1, 0.9, 1.7, 0.5])
        phi = self._band(11, 6, w)
        anchor = ColorAnchor([OrientedTriangle([0, 1, 2], [1, 1, 1]),
                              OrientedTriangle([3, 4, 5], [1, 1, 1])])
        p = anchor.evaluate(phi, w)
        self.assertGreater(p.terms[0], 0.0)
        self.assertGreater(p.terms[1], 0.0)
        for phase in p.det_phases:
            self.assertFalse(math.isnan(phase))
        self.assertEqual(p.overlapping_triangles, 0)
        self.assertEqual(p.overlap_relation, "shared-edge")
        self.assertTrue(math.isnan(p.phase_coherence))
        self.assertTrue(math.isnan(p.phase_dispersion))
        self.assertEqual(anchor.overlapping_triangle_count(), 0)
        self.assertFalse(anchor.overlaps_another(0))
        self.assertFalse(anchor.overlaps_another(1))

    def test_a_single_triangle_atlas_has_nothing_to_overlap(self):
        w = np.array([1.3, 0.7, 1.1])
        phi = self._band(12, 3, w)
        p = ColorAnchor([OrientedTriangle([0, 1, 2], [1, 1, 1])]).evaluate(
            phi, w)
        self.assertAlmostEqual(p.score, 1.0, delta=1e-12)
        self.assertEqual(p.overlapping_triangles, 0)
        self.assertTrue(math.isnan(p.phase_coherence))

    def test_an_edge_sharing_atlas_carries_the_coherence(self):
        w = np.array([1.3, 0.7, 1.1, 0.9])
        phi = self._band(13, 4, w)
        tri_spec = [((0, 1, 2), (1, 1, 1)), ((0, 1, 3), (1, 1, 1))]
        anchor = ColorAnchor([OrientedTriangle(list(e), list(s))
                              for e, s in tri_spec])
        p = anchor.evaluate(phi, w)
        self.assertEqual(p.overlapping_triangles, 2)
        self.assertTrue(anchor.overlaps_another(0))
        self.assertTrue(anchor.overlaps_another(1))
        _score, _terms, _pr, _phases, coherence = ref_profile(
            phi, w, tri_spec, [0.5, 0.5])
        self.assertFalse(math.isnan(p.phase_coherence))
        self.assertLessEqual(abs(p.phase_coherence - coherence), 1e-10)

    def test_an_isolated_face_does_not_vote_on_the_coherence(self):
        # A mixed atlas: two edge-sharing triangles plus one on disjoint
        # rows.  The isolated face still reports its own term and phase, but
        # it has no overlap partner and therefore no say in the coherence.
        w = np.array([1.3, 0.7, 1.1, 0.9, 1.7, 0.5, 1.2])
        phi = self._band(14, 7, w)
        shared = [((0, 1, 2), (1, 1, 1)), ((0, 1, 3), (1, 1, 1))]
        mixed = shared + [((4, 5, 6), (1, 1, 1))]
        conv = [0.4, 0.4, 0.2]
        anchor = ColorAnchor([OrientedTriangle(list(e), list(s))
                              for e, s in mixed], conv)
        p = anchor.evaluate(phi, w)
        self.assertEqual(p.overlapping_triangles, 2)
        self.assertGreater(p.terms[2], 0.0)          # measured ...
        self.assertFalse(math.isnan(p.det_phases[2]))  # ... and reported
        # the coherence equals the two-triangle overlap resultant exactly
        _s, _t, _pr, _ph, restricted = ref_profile(phi, w, shared, conv[:2])
        self.assertLessEqual(abs(p.phase_coherence - restricted), 1e-10)
        # ... and NOT the global resultant over all three declared faces
        terms, phases = ref_anchor_terms(phi, w, mixed)
        u = np.asarray(conv) * terms
        global_resultant = abs(np.sum(u * np.exp(1j * phases))) / float(
            np.sum(u))
        self.assertGreater(abs(p.phase_coherence - global_resultant), 1e-6)

    def test_the_overlap_set_is_declared_before_any_datum_is_examined(self):
        # The relation is a property of the DECLARED atlas: it is available
        # before evaluate() and does not change when the data arrive.
        anchor = ColorAnchor([OrientedTriangle([0, 1, 2], [1, 1, 1]),
                              OrientedTriangle([2, 3, 4], [1, 1, 1]),
                              OrientedTriangle([5, 6, 7], [1, 1, 1])])
        self.assertEqual(anchor.overlapping_triangle_count(), 2)
        self.assertTrue(anchor.overlaps_another(0))
        self.assertTrue(anchor.overlaps_another(1))
        self.assertFalse(anchor.overlaps_another(2))
        w = np.full(8, 1.1)
        p = anchor.evaluate(self._band(15, 8, w), w)
        self.assertEqual(p.overlapping_triangles, 2)


if __name__ == "__main__":
    unittest.main()
