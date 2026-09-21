# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The finite-element Coulomb kernel, densities, and Hartree-Fock on the
covariance.

Conventions, fixed here and used throughout. The one-particle modes are the
M-orthonormal eigenvectors `z_n` of a pencil (nodal values), so creation and
annihilation operators of the modes obey the canonical anticommutation
relations although the nodal basis is not orthogonal. The density of a pair of
modes is a 0-chain, the load vector

    rho^{mn}_v = sum_ij conj(z_im) z_jn int phi_i phi_j phi_v ,

the integral of conj(psi_m) psi_n against the vertex function phi_v
(`WhitneyMass.vertexDensityContraction`). The Coulomb potential of a density
`rho` is the piecewise-linear solution of Poisson's equation,
`A0 v = 4 pi e^2 rho`, with `A0` the stiffness matrix, and the interaction
energy of two densities is `rho_a^dagger K rho_b` with the kernel
`K = 4 pi e^2 A0^+`. On a closed cell the constants are the kernel of `A0`:
the pseudo-inverse acts on the complement, which is the neutralizing uniform
background, and returns the potential of zero mean.

The many-body Hamiltonian in the mode basis is

    H = dGamma(h) + 1/2 sum_vw K_vw : rho_v rho_w : ,   rho_v = dGamma(T_v),

with `T_v` the matrix of pair densities at vertex `v`, and its mean-field
energy in a quasi-free state of covariance `Gamma_ij = <a_j^dagger a_i>` is
the direct and exchange Wick contraction

    E = tr(h Gamma) + 1/2 sum_vw K_vw [tr(T_v Gamma) tr(T_w Gamma) - tr(T_v Gamma T_w Gamma)].
"""
import itertools

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from tessera import chainhodge as ch
from tessera.drivers.bands import E2


class CoulombKernel:
    """`K = strength * A0^+` on the vertices of a complex, applied by one sparse
    factorization. `stiffness` and `mass` are the degree-zero pencil at the
    trivial connection; `strength` is 4 pi e^2 in the units of the run."""

    def __init__(self, stiffness, mass, strength=4.0 * np.pi * E2):
        self.stiffness = sp.csc_matrix(stiffness)
        self.strength = float(strength)
        n = self.stiffness.shape[0]
        # The integrals of the vertex functions: the constant potential paired
        # against a load vector, and the weight that defines the mean.
        self.weights = np.asarray(sp.csc_matrix(mass).sum(axis=1)).ravel().real
        self.volume = float(self.weights.sum())
        bordered = sp.bmat([[self.stiffness, sp.csc_matrix(self.weights.reshape(n, 1))],
                            [sp.csc_matrix(self.weights.reshape(1, n)), None]], format="csc")
        self._factor = spla.splu(bordered.astype(complex))
        self.size = n

    @classmethod
    def of_cell(cls, cell, strength=4.0 * np.pi * E2):
        return cls(cell.stiffness.dressed(), cell.mass.dressed(), strength)

    def potential(self, rho):
        """The zero-mean potential of the load vector(s) `rho` (columns), with
        the uniform background removed: `strength * A0^+ rho`."""
        rho = np.asarray(rho, dtype=complex)
        columns = rho.reshape(self.size, -1)
        neutral = columns - np.outer(self.weights, columns.sum(axis=0)) / self.volume
        rhs = np.vstack([neutral, np.zeros((1, neutral.shape[1]), dtype=complex)])
        solution = self._factor.solve(rhs)[: self.size]
        return (self.strength * solution).reshape(rho.shape)

    def energy(self, rho_a, rho_b=None):
        """`rho_a^dagger K rho_b` (without the factor one half)."""
        rho_b = rho_a if rho_b is None else rho_b
        return np.vdot(rho_a, self.potential(rho_b))


def pair_densities(complex_, squared_lengths, modes):
    """`T[:, m, n] = rho^{mn}`: the load vectors of conj(z_m) z_n for every pair
    of the columns of `modes`, shape (vertices, modes, modes).

    On a top simplex of volume |T| the three-factor integral is
    |T| mu_abc d! / (d + 3)! with mu_abc = 1 + delta_ab + delta_ac + delta_bc
    + 2 delta_ab delta_bc, so with x = conj(z_m) and y = z_n restricted to it,

        sum_ab mu_abc x_a y_b = S_x S_y + sum_a x_a y_a + x_c S_y + S_x y_c + 2 x_c y_c ,

    which is what is accumulated here for every pair at once. It is the
    vectorized form of `WhitneyMass.vertexDensityContraction`.
    """
    modes = np.asarray(modes, dtype=complex)
    vertex_index = {int(cell[0]): i for i, cell in enumerate(complex_.kSimplexVertices(0))}
    tops = np.array([[vertex_index[int(v)] for v in cell] for cell in complex_.orientedTopSimplices()])
    d = tops.shape[1] - 1
    volumes = np.array(ch.WhitneyMass.certificate(complex_, list(squared_lengths)).volumes)
    weight = volumes * float(np.prod(np.arange(1, d + 1))) / float(np.prod(np.arange(1, d + 4)))
    count = modes.shape[1]
    local = modes[tops]                                   # (tops, d + 1, modes)
    scatter = [sp.csr_matrix((np.ones(len(tops)), (tops[:, c], np.arange(len(tops)))),
                             shape=(modes.shape[0], len(tops))) for c in range(d + 1)]
    sum_y = local.sum(axis=1)                             # (tops, modes)
    T = np.empty((modes.shape[0], count, count), dtype=complex)
    for m in range(count):
        x = local[:, :, m].conj()                         # (tops, d + 1)
        sum_x = x.sum(axis=1)
        common = sum_x[:, None] * sum_y + np.einsum("ta,tan->tn", x, local)
        total = np.zeros((modes.shape[0], count), dtype=complex)
        for c in range(d + 1):
            term = common + x[:, c, None] * sum_y + sum_x[:, None] * local[:, c, :] \
                + 2.0 * x[:, c, None] * local[:, c, :]
            total += scatter[c] @ (weight[:, None] * term)
        T[:, m, :] = total
    return T


class ModeInteraction:
    """The Coulomb interaction in a basis of M-orthonormal modes, and
    Hartree-Fock on it.

    `energies` are the one-particle levels of the modes, `T` their pair
    densities, and `sheets` the number of identical spin copies (1 or 2): with
    two sheets the mode list is doubled, sheet by sheet, and the pair density
    is diagonal in the sheet.
    """

    def __init__(self, kernel, energies, T, sheets=1):
        self.kernel = kernel
        self.sheets = int(sheets)
        orbitals = len(energies)
        self.orbitals = orbitals
        self.size = orbitals * self.sheets
        self.h = np.kron(np.eye(self.sheets), np.diag(np.asarray(energies, dtype=complex)))
        self.T = np.asarray(T, dtype=complex)
        flat = self.T.reshape(self.T.shape[0], orbitals * orbitals)
        potentials = kernel.potential(flat)
        # (mn|pq) = rho^{nm dagger}... stored as W[mn, pq] = sum_v conj(rho^{nm}_v) (K rho^{pq})_v,
        # and conj(rho^{nm}) = rho^{mn}, so W[mn, pq] = sum_v rho^{mn}_v (K rho^{pq})_v.
        self.W = (flat.T @ potentials).reshape(orbitals, orbitals, orbitals, orbitals)

    # -- one-particle matrices of the density at a vertex, with the sheets

    def density_operator(self, vertex):
        return np.kron(np.eye(self.sheets), self.T[vertex])

    def _blocks(self, gamma):
        n = self.orbitals
        return [[gamma[a * n:(a + 1) * n, b * n:(b + 1) * n] for b in range(self.sheets)]
                for a in range(self.sheets)]

    def fock(self, gamma):
        """F(Gamma) = h + V_H[Gamma] + V_x[Gamma]."""
        blocks = self._blocks(np.asarray(gamma, dtype=complex))
        total = sum(blocks[a][a] for a in range(self.sheets))
        # tr(T_w Gamma) = sum_pq T_w[p, q] Gamma[q, p]
        hartree = np.einsum("mnpq,qp->mn", self.W, total)
        F = self.h.copy()
        n = self.orbitals
        for a in range(self.sheets):
            F[a * n:(a + 1) * n, a * n:(a + 1) * n] += hartree
            for b in range(self.sheets):
                # -(T_v Gamma T_w)[m, n] K_vw = -sum_pq W[mp, qn] Gamma_ab[p, q]
                F[a * n:(a + 1) * n, b * n:(b + 1) * n] -= np.einsum("mpqn,pq->mn", self.W, blocks[a][b])
        return F

    def energy(self, gamma):
        """The mean-field energy of the covariance `gamma` (direct and exchange)."""
        gamma = np.asarray(gamma, dtype=complex)
        F = self.fock(gamma)
        return 0.5 * np.trace((self.h + F) @ gamma)

    def interaction_energy(self, gamma):
        gamma = np.asarray(gamma, dtype=complex)
        return self.energy(gamma) - np.trace(self.h @ gamma)

    def roothaan(self, particles, gamma=None, tolerance=1e-12, max_iterations=500, mixing=0.5):
        """The Roothaan loop: build F(Gamma), fill its lowest `particles` modes,
        mix, repeat. Returns (Gamma, energy, iterations, residual), the residual
        being the commutator norm ||[F, Gamma]||, which vanishes at a fixed point."""
        if gamma is None:
            gamma = self._filled(self.h, particles)
        residual = np.inf
        for iteration in range(1, max_iterations + 1):
            F = self.fock(gamma)
            residual = np.linalg.norm(F @ gamma - gamma @ F)
            if residual < tolerance:
                break
            gamma = (1.0 - mixing) * gamma + mixing * self._filled(F, particles)
            # Mixing leaves the projector manifold; purify back onto it.
            gamma = self._filled(-gamma, particles)
        return gamma, self.energy(gamma).real, iteration, residual

    @staticmethod
    def _filled(F, particles):
        values, vectors = np.linalg.eigh(0.5 * (F + F.conj().T))
        occupied = vectors[:, :particles]
        return occupied @ occupied.conj().T

    # -- the explicit Fock-space Hamiltonian (small mode counts only)

    def fock_hamiltonian(self, algebra):
        """H = dGamma(h) + 1/2 sum_vw K_vw (rho_v rho_w - dGamma(T_v T_w)) as a
        sparse matrix on the exterior algebra `algebra` (an `ExteriorAlgebra`
        over `size` modes). The subtraction is the normal ordering."""
        def lifted(one_particle):
            rows, cols, values, n = algebra.dGammaCOO(np.asarray(one_particle, dtype=complex))
            return sp.csr_matrix((values, (rows, cols)), shape=(n, n))

        vertices = self.T.shape[0]
        K = np.column_stack([self.kernel.potential(np.eye(vertices)[:, v]) for v in range(vertices)])
        H = lifted(self.h)
        densities = [lifted(self.density_operator(v)) for v in range(vertices)]
        for v in range(vertices):
            for w in range(vertices):
                if K[v, w] == 0.0:
                    continue
                H = H + 0.5 * K[v, w] * (densities[v] @ densities[w]
                                         - lifted(self.density_operator(v) @ self.density_operator(w)))
        return H.tocsr()


# ---------------------------------------------------------------- uniform electron gas

# The potential of a unit point charge at its own site in a simple cubic lattice
# of side 1 with a neutralizing background is -MADELUNG_SC (in units of e^2 / L).
MADELUNG_SC = 2.837297479


def closed_shell_momenta(max_squared):
    """The integer vectors n with |n|^2 <= max_squared: the closed shells of a
    cubic cell sampled at the zone centre, momenta 2 pi n / L."""
    reach = int(np.sqrt(max_squared)) + 1
    return np.array([n for n in itertools.product(range(-reach, reach + 1), repeat=3)
                     if sum(x * x for x in n) <= max_squared], dtype=float)


def plane_wave_exchange(momenta, side, strength=4.0 * np.pi * E2, corrected=True):
    """The exchange energy of doubly filled plane waves in a cubic cell of side
    `side`: -sum_{i != j} strength / (V |k_i - k_j|^2), and with `corrected`
    the zero-momentum term that a cell sampled at one point leaves out. The
    Coulomb kernel is singular but integrable at zero momentum transfer; its
    integral over the missing cell of momentum space is the self-interaction of
    a point charge with its images and background, strength * MADELUNG_SC /
    (4 pi side) per orbital (Gygi & Baldereschi, Physical Review B 34, 4405 (1986))."""
    k = 2.0 * np.pi / side * np.asarray(momenta, dtype=float)
    q2 = ((k[:, None, :] - k[None, :, :]) ** 2).sum(axis=-1)
    volume = side ** 3
    pairs = np.where(q2 > 0.0, strength / (volume * np.where(q2 > 0.0, q2, 1.0)), 0.0)
    energy = -pairs.sum()
    if corrected:
        energy -= len(k) * strength * MADELUNG_SC / (4.0 * np.pi * side)
    return energy


def electron_gas_exchange(electrons, volume, strength=4.0 * np.pi * E2):
    """-(3/4) (3/pi)^{1/3} e^2 n^{1/3} per electron, times the electrons."""
    density = electrons / volume
    return -0.75 * (3.0 / np.pi) ** (1.0 / 3.0) * strength / (4.0 * np.pi) * density ** (1.0 / 3.0) * electrons
