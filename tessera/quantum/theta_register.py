# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The theta quantization register: a boundary marking lattice quantized at level k.

A boundary torus carries its marking lattice ``H^1(M; Z)``, a modulus ``tau`` in
the upper half-plane, and -- between two markings -- an integer monodromy
``M``. This register treats the lattice as a WEIGHT lattice and takes the
Hilbert space to be its exponential: the level-``k`` theta functions on the
Jacobian ``C / (Z + tau Z)``,

    theta_j(z; tau) = sum_n exp( pi i k (n + j/k)^2 tau + 2 pi i k (n + j/k) z ),

``j`` in ``Z/k``. For ``g`` tori with a ``g x g`` period matrix ``Omega`` the
characteristics are ``j`` in ``(Z/k)^g``; for block-diagonal ``Omega`` the
lattice sum factorises exactly, so a DIRECT SUM of lattices is a TENSOR
PRODUCT of Hilbert spaces. That is the point of the register.

A symplectic integer matrix ``M = [[A, B], [C, D]]`` acts by
``Omega -> (A Omega + B)(C Omega + D)^-1`` and ``z -> (C Omega + D)^-T z`` and
carries the theta space to itself:

    theta_j(z'; Omega') = e(z) sum_l rho_k(M)_jl theta_l(z; Omega),
    e(z) = c_M exp( pi i k z^T (C Omega + D)^-1 C z ),

with ``rho_k(M)`` the Weil representation, independent of ``z`` and of
``Omega``. It is DETERMINED HERE BY A LEAST-SQUARES FIT of that law over
sampled ``z``, never typed in: the fit residual certifies that the space is
closed under the action, and the closed forms (Hadamard and the phase gate at
level 2) are checks on the result. An anti-symplectic ``M`` (``M^T J M = -J``;
the orientation-reversing collar swap) factors as ``M = M+ R`` with
``R = diag(I, -I)`` acting by ``Omega -> -conj(Omega)`` and complex
conjugation on the basis, so ``rho(M) = rho(M+) K`` is antiunitary.

Nothing here touches the engine: the register reads periods, moduli and
monodromies the geometry already produced.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np

__all__ = ["ThetaRegister", "WeilFit"]


@dataclass(frozen=True)
class WeilFit:
    """The Weil matrix of one symplectic or anti-symplectic integer matrix.

    ``matrix`` is unitary (normalised from the fit); the map on a state
    vector ``v`` in the theta basis is ``matrix @ v`` when ``antiunitary`` is
    false and ``matrix @ conj(v)`` when it is true. ``residual`` is the
    relative least-squares residual of the transformation law -- the
    certificate that the theta space is closed under the action --
    ``unitarity_defect`` is ``|rho^H rho / c - I|`` before normalisation,
    and ``scale`` the ``c`` divided out (``|det(C Omega + D)|`` and the
    constant of the automorphy factor, which the fit absorbs).
    """
    matrix: np.ndarray
    antiunitary: bool
    residual: float
    unitarity_defect: float
    scale: float

    def apply(self, vector):
        vector = np.asarray(vector, dtype=complex)
        return self.matrix @ (np.conj(vector) if self.antiunitary else vector)

    def compose(self, other):
        """``self`` after ``other`` as (anti)linear maps:
        ``(U1 K^a1)(U2 K^a2) = U1 conj^a1(U2) K^(a1 + a2)``."""
        inner = np.conj(other.matrix) if self.antiunitary else other.matrix
        return WeilFit(self.matrix @ inner, self.antiunitary != other.antiunitary,
                       max(self.residual, other.residual),
                       max(self.unitarity_defect, other.unitarity_defect), 1.0)


class ThetaRegister:
    """Level-``k`` quantization of marking lattices (see the module docstring).

    Every operation is a method here; the numerical parameters are the
    level, the tolerance a check is judged at, and the truncation epsilon of
    the lattice sums.
    """

    def __init__(self, level=2, tolerance=1e-12, truncation_epsilon=1e-18,
                 samples_per_dimension=8, sample_radius=0.25):
        if int(level) < 1:
            raise ValueError("level must be a positive integer")
        self.level = int(level)
        self.tolerance = float(tolerance)
        self.truncation_epsilon = float(truncation_epsilon)
        self.samples_per_dimension = int(samples_per_dimension)
        self.sample_radius = float(sample_radius)

    # ------------------------------------------------------------ lattices
    def characteristics(self, genus):
        """Every ``j`` in ``(Z/k)^g``, lexicographic with the FIRST torus
        slowest, so that the genus-2 basis is ``kron(first, second)``."""
        return np.array(list(itertools.product(range(self.level), repeat=int(genus))), dtype=float)

    def dimension(self, genus):
        return self.level ** int(genus)

    @staticmethod
    def as_period_matrix(omega):
        """A modulus or a period matrix as a ``g x g`` complex array in the
        Siegel upper half-space (symmetric, positive-definite imaginary part)."""
        omega = np.atleast_2d(np.asarray(omega, dtype=complex))
        if omega.shape[0] != omega.shape[1]:
            raise ValueError("a period matrix is square; got %s" % (omega.shape,))
        if np.linalg.norm(omega - omega.T) > 1e-12 * max(1.0, np.linalg.norm(omega)):
            raise ValueError("a period matrix is symmetric")
        if np.linalg.eigvalsh(omega.imag).min() <= 0.0:
            raise ValueError("Im Omega must be positive definite (the modulus in the upper half-plane)")
        return omega

    def truncation_radius(self, omega, imag_z_bound):
        """The lattice radius ``N`` past which every term is below the
        truncation epsilon: ``|term| <= exp(-pi k lam |m|^2 + 2 pi k |m| b)``
        with ``lam`` the smallest eigenvalue of ``Im Omega`` and ``b`` a bound
        on ``|Im z|``; the sum runs over ``|n_i| <= N``."""
        lam = float(np.linalg.eigvalsh(np.asarray(omega).imag).min())
        k = self.level
        radius = 1
        while True:
            m = max(radius - 1.0, 0.0)          # |m| >= |n| - |j/k| >= radius - 1 on the shell
            bound = math.exp(-math.pi * k * lam * m * m + 2.0 * math.pi * k * m * imag_z_bound)
            if bound <= self.truncation_epsilon and radius >= 2:
                return radius + 1
            radius += 1
            if radius > 200:
                raise ValueError("lattice sum does not converge to the truncation epsilon: "
                                 "Im Omega too small (lambda_min = %g)" % lam)

    def basis(self, z, omega):
        """All ``theta_j(z; Omega)``: an array ``(k^g, n_samples)`` for sample
        points ``z`` of shape ``(n_samples, g)``."""
        omega = self.as_period_matrix(omega)
        genus = omega.shape[0]
        z = np.atleast_2d(np.asarray(z, dtype=complex))
        if z.shape[1] != genus:
            raise ValueError("z has %d components for genus %d" % (z.shape[1], genus))
        radius = self.truncation_radius(omega, float(np.abs(z.imag).max()) if z.size else 0.0)
        grid = np.array(list(itertools.product(range(-radius, radius + 1), repeat=genus)), dtype=float)
        k = self.level
        out = np.empty((self.dimension(genus), z.shape[0]), dtype=complex)
        for row, j in enumerate(self.characteristics(genus)):
            m = grid + j / k                                    # (P, g)
            quad = np.einsum("pa,ab,pb->p", m, omega, m)        # (P,)
            lin = m @ z.T                                       # (P, S)
            out[row] = np.exp(1j * math.pi * k * quad[:, None] + 2j * math.pi * k * lin).sum(axis=0)
        return out

    # ------------------------------------------------------------ symplectic action
    @staticmethod
    def symplectic_form(genus):
        genus = int(genus)
        return np.block([[np.zeros((genus, genus)), np.eye(genus)],
                         [-np.eye(genus), np.zeros((genus, genus))]])

    @classmethod
    def symplectic_sign(cls, matrix):
        """``+1`` for ``M^T J M = J``, ``-1`` for ``M^T J M = -J``, else a
        ``ValueError``: only those act on the theta space."""
        matrix = np.asarray(matrix, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] % 2:
            raise ValueError("a symplectic matrix is 2g x 2g; got %s" % (matrix.shape,))
        if np.abs(matrix - np.rint(matrix)).max() > 1e-12:
            raise ValueError("the monodromy must be an integer matrix")
        form = cls.symplectic_form(matrix.shape[0] // 2)
        product = matrix.T @ form @ matrix
        if np.abs(product - form).max() < 1e-12:
            return +1
        if np.abs(product + form).max() < 1e-12:
            return -1
        raise ValueError("neither symplectic nor anti-symplectic: M^T J M is not +-J")

    @staticmethod
    def blocks(matrix):
        matrix = np.asarray(matrix, dtype=float)
        genus = matrix.shape[0] // 2
        return (matrix[:genus, :genus], matrix[:genus, genus:],
                matrix[genus:, :genus], matrix[genus:, genus:])

    def siegel(self, matrix, omega):
        """The action of a symplectic ``M`` on ``Omega``, the map on ``z``, and
        the ``z``-dependent automorphy factor (its constant is absorbed by the fit)."""
        omega = self.as_period_matrix(omega)
        a, b, c, d = self.blocks(matrix)
        denominator = c @ omega + d
        inverse = np.linalg.inv(denominator)
        omega_after = (a @ omega + b) @ inverse
        omega_after = 0.5 * (omega_after + omega_after.T)     # exact symmetry up to rounding

        def move(z):
            return np.asarray(z, dtype=complex) @ inverse          # rows z -> z (C Omega + D)^-T

        def factor(z):
            z = np.asarray(z, dtype=complex)
            return np.exp(1j * math.pi * self.level * np.einsum("sa,ab,sb->s", z, inverse @ c, z))
        return omega_after, move, factor

    def orientation_reversal(self, omega):
        """``R = diag(I, -I)``: ``Omega -> -conj(Omega)``, and on the basis
        ``theta_j(z; Omega) -> conj theta_j(z; Omega) = theta_j(-conj z; -conj Omega)``."""
        return -np.conj(self.as_period_matrix(omega))

    # ------------------------------------------------------------ the Weil matrix
    def samples(self, genus, rng):
        count = self.samples_per_dimension * self.dimension(genus)
        return self.sample_radius * (rng.uniform(-1, 1, (count, genus)) + 1j * rng.uniform(-1, 1, (count, genus)))

    def weil(self, matrix, omega, rng=None):
        """``rho_k(M)`` by least squares on the transformation law (see the
        module docstring), with its residual and unitarity defect."""
        rng = np.random.default_rng(0) if rng is None else rng
        omega = self.as_period_matrix(omega)
        genus = omega.shape[0]
        sign = self.symplectic_sign(matrix)
        matrix = np.asarray(matrix, dtype=float)
        z = self.samples(genus, rng)
        if sign == -1:
            reversal = np.diag(np.concatenate([np.ones(genus), -np.ones(genus)]))
            matrix = matrix @ reversal                       # M+ = M R  (R^-1 = R)
            source_omega = self.orientation_reversal(omega)
            source_z = -np.conj(z)
            source = self.basis(source_z, source_omega)       # = conj theta(z; Omega)
        else:
            source_omega, source_z, source = omega, z, self.basis(z, omega)
        omega_after, move, factor = self.siegel(matrix, source_omega)
        target = self.basis(move(source_z), omega_after)
        design = source * factor(source_z)[None, :]
        fitted = np.linalg.lstsq(design.T, target.T, rcond=None)[0].T
        residual = float(np.linalg.norm(target - fitted @ design) / np.linalg.norm(target))
        gram = fitted.conj().T @ fitted
        scale = float(np.trace(gram).real / gram.shape[0])
        defect = float(np.linalg.norm(gram / scale - np.eye(gram.shape[0])))
        return WeilFit(fitted / math.sqrt(scale), sign == -1, residual, defect, scale)

    # ------------------------------------------------------------ structure
    @staticmethod
    def projective_distance(left, right):
        """``min_phase |left - phase right| / |right|``."""
        left, right = np.asarray(left), np.asarray(right)
        overlap = np.vdot(right, left)
        phase = overlap / abs(overlap) if abs(overlap) > 0 else 1.0
        return float(np.linalg.norm(left - phase * right) / np.linalg.norm(right))

    def coherent_state(self, z, omega):
        """The geometric state at the holonomy point ``z``: the unit vector
        of ``Theta_k(Omega)`` with coefficients ``conj(theta_j(z; Omega))``
        in the theta basis -- the evaluation functional at ``z`` (the
        reproducing kernel), the state the boundary geometry defines once
        its holonomies are ``z``. In the basis that the identity monodromy
        identifies with the product of the torus registers, a diagonal
        ``Omega`` gives a product vector and an off-diagonal ``Omega``
        does not."""
        values = self.basis(np.atleast_2d(np.asarray(z, dtype=complex)), omega)[:, 0]
        vector = np.conj(values)
        norm = float(np.linalg.norm(vector))
        if not norm > 0.0:
            raise ValueError("the coherent state vanishes at z = %r" % (z,))
        return vector / norm

    @staticmethod
    def entanglement_entropy(vector, dims):
        """The Schmidt spectrum (probabilities, descending) and the
        entanglement entropy in bits of a unit vector of
        ``C^d1 (x) C^d2``, the first factor the row index of the
        ``d1 x d2`` coefficient matrix."""
        d1, d2 = int(dims[0]), int(dims[1])
        matrix = np.asarray(vector, dtype=complex).reshape(d1, d2)
        singular = np.linalg.svd(matrix, compute_uv=False)
        weights = singular ** 2
        total = float(weights.sum())
        if not total > 0.0:
            raise ValueError("the vector vanishes")
        probabilities = weights / total
        nonzero = probabilities[probabilities > 0.0]
        entropy = float(-(nonzero * np.log2(nonzero)).sum())
        return entropy, probabilities

    @staticmethod
    def schmidt_rank(operator, dims, tolerance=1e-12):
        """The operator Schmidt rank of a ``d1 d2 x d1 d2`` matrix across the
        ``(d1, d2)`` bipartition: 1 for a product ``A (x) B``, more when the
        operator entangles."""
        d1, d2 = int(dims[0]), int(dims[1])
        tensor = np.asarray(operator).reshape(d1, d2, d1, d2).transpose(0, 2, 1, 3).reshape(d1 * d1, d2 * d2)
        singular = np.linalg.svd(tensor, compute_uv=False)
        return int((singular > tolerance * singular[0]).sum())

    def level_two_forms(self):
        """The level-2 closed forms the fit is checked against: ``rho(S)`` is
        Hadamard, ``rho(T)`` the phase gate, ``rho(shear)`` controlled-Z."""
        if self.level != 2:
            raise ValueError("the closed forms here are the level-2 ones")
        hadamard = np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2.0)
        phase = np.diag([1.0, 1j])
        controlled_z = np.diag([1.0, 1.0, 1.0, -1.0]).astype(complex)
        return {"S": hadamard, "T": phase, "shear": controlled_z}

    @staticmethod
    def generators():
        """``S = [[0,-1],[1,0]]`` (``tau -> -1/tau``), ``T = [[1,1],[0,1]]``
        (``tau -> tau + 1``), the collar swap, and the genus-2 shear
        ``[[I, B], [0, I]]`` with ``B = [[0,1],[1,0]]`` (``Omega -> Omega + B``)."""
        shear = np.eye(4)
        shear[0, 3] = shear[1, 2] = 1.0
        return {"S": np.array([[0, -1], [1, 0]], dtype=float),
                "T": np.array([[1, 1], [0, 1]], dtype=float),
                "swap": np.array([[0, 1], [1, 0]], dtype=float),
                "shear": shear}

    @staticmethod
    def direct_sum(first, second):
        first, second = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
        g1, g2 = first.shape[0] // 2, second.shape[0] // 2
        a1, b1, c1, d1 = ThetaRegister.blocks(first)
        a2, b2, c2, d2 = ThetaRegister.blocks(second)
        zero12, zero21 = np.zeros((g1, g2)), np.zeros((g2, g1))
        return np.block([[a1, zero12, b1, zero12], [zero21, a2, zero21, b2],
                         [c1, zero12, d1, zero12], [zero21, c2, zero21, d2]])
