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
        """The kernel of a periodic cell, inverted by Fourier transform."""
        return GridCoulombKernel(cell, strength)

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


class GridCoulombKernel:
    """`CoulombKernel` on a `CrystalCell`, without a factorization.

    Every vertex of the periodic Kuhn grid is equivalent, so the stiffness
    matrix commutes with the grid translations and is diagonal on the lattice
    plane waves: its symbol is the Fourier transform of one of its rows. The
    pseudo-inverse divides by the symbol away from zero momentum and sets the
    mean to zero, which is the same potential `CoulombKernel` returns (the
    vertex weights are all equal here), exactly and in O(n log n).
    """

    def __init__(self, cell, strength=4.0 * np.pi * E2):
        self.strength = float(strength)
        self.shape = tuple(cell.divisions)
        self.size = cell.size
        stiffness = cell.stiffness.dressed().real.tocsr()
        self.weights = np.asarray(cell.mass.dressed().real.sum(axis=1)).ravel()
        self.volume = float(self.weights.sum())
        if np.ptp(self.weights) > 1e-10 * self.weights.mean():
            raise ValueError("the grid's vertices are not equivalent; use CoulombKernel")
        row = stiffness.getrow(0)
        stencil = np.zeros(self.size)
        stencil[row.indices] = row.data
        # The row as unwrapped grid offsets and entries, for the symbol in closed form.
        offsets = cell.index[row.indices].astype(float)
        offsets = np.where(offsets > np.array(self.shape) / 2.0, offsets - np.array(self.shape), offsets)
        self._offsets, self._entries = offsets, row.data.astype(float)
        mass_row = sp.csr_matrix(cell.mass.dressed().real).getrow(0)
        lookup = dict(zip(mass_row.indices.tolist(), mass_row.data.tolist()))
        self._mass_entries = np.array([lookup.get(int(index), 0.0) for index in row.indices])
        self._reciprocal, self._volume, self._lattice = cell.reciprocal, cell.volume, cell.lattice
        symbol = np.fft.fftn(stencil.reshape(self.shape)).real
        # Translation invariance, checked on a second row rather than assumed.
        probe = cell.grid.vertexId(1, 2, 3)
        other = stiffness.getrow(probe)
        moved = np.zeros(self.size)
        moved[other.indices] = other.data
        shifted = np.roll(moved.reshape(self.shape), (-1, -2, -3), axis=(0, 1, 2))
        if np.abs(shifted - stencil.reshape(self.shape)).max() > 1e-10 * np.abs(stencil).max():
            raise ValueError("the stiffness matrix is not translation invariant; use CoulombKernel")
        self._inverse = np.where(symbol > 1e-12 * symbol.max(), 1.0 / np.where(symbol > 0.0, symbol, 1.0), 0.0)
        self._inverse.flat[0] = 0.0

    def symbol(self, wavevectors):
        """The symbol of the stiffness matrix at the wavevectors `p` (rows, in
        units of the grid's reciprocal steps, so integers are the wavevectors the
        cell supports): a(p) = sum_n A_{0 n} exp(2 pi i p . n / N), a closed form
        in the entries of one row. A lattice plane wave of wavevector p is an
        eigenvector of the stiffness matrix dressed by the momentum p - round(p),
        with this eigenvalue."""
        p = np.atleast_2d(np.asarray(wavevectors, dtype=float)) / np.array(self.shape)
        return np.real(np.exp(2j * np.pi * (p @ self._offsets.T)) @ self._entries)

    def zero_momentum_constant(self, refinements=(2, 3, 4, 6, 8), transfers=None):
        """The term of the Coulomb kernel that a cell sampled at its zone centre
        leaves out at zero momentum transfer, for a normalized charge: the
        auxiliary-function correction of Gygi and Baldereschi, with the kernel's
        own symbol as the auxiliary function,

            c = strength * [ <1 / a(p)>_p  -  (1 / n) sum_{G != 0} 1 / a(G) ] ,

        the average of the inverse symbol over every wavevector of the mesh (the
        momentum-space integral a denser and denser momentum set converges to)
        minus its sampling on the wavevectors the cell supports.

        The average is an integral of a function with the singularity
        (n / V) / k^2 at the origin. That part is taken analytically: the
        periodized function g(k) = (n / V) sum_K exp(-alpha |k + K|^2) / |k + K|^2
        over the reciprocal lattice of the vertex grid has the same singularity
        and the average 1 / (4 pi^(3/2) sqrt(alpha)). The remainder 1 / a - g is
        bounded, with a jump at the origin only, so its mean on grids
        `refinements` times finer with the origin left out errs by odd inverse
        powers of the refinement from the third on; five refinements remove the
        first four of them. For the continuum
        kernel strength / (V q^2) on a simple cubic cell this constant is
        strength * MADELUNG_SC / (4 pi L).

        `transfers` lists the momentum transfers of a momentum set (reciprocal
        coordinates of the cell, the zero transfer included, each with the
        weight 1 / len): the sampling then runs over G + q for every transfer,
        which is the sampling of the supercell the set is equivalent to."""
        n, shape = self.size, np.array(self.shape)
        step = self._reciprocal                                  # p is in units of the wavevectors the cell supports
        spacing = (self._volume / n) ** (1.0 / 3.0)
        alpha = 0.25 * spacing ** 2
        images = np.stack(np.meshgrid(*[np.arange(-2, 3)] * 3, indexing="ij"), axis=-1).reshape(-1, 3) * shape

        def remainder(p):
            symbol = self.symbol(p)
            singular = np.zeros(len(p))
            for image in images:
                k2 = (((p + image) @ step) ** 2).sum(axis=1)
                singular += np.where(k2 > 0.0, np.exp(-alpha * k2) / np.where(k2 > 0.0, k2, 1.0), 0.0)
            regular = symbol > 1e-12 * np.abs(self._entries).max()
            return np.where(regular, 1.0 / np.where(regular, symbol, 1.0) - (n / self._volume) * singular, 0.0)

        means = []
        for m in refinements:
            axes = [np.arange(N * m) / m for N in self.shape]
            p = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
            means.append(sum(remainder(chunk).sum() for chunk in np.array_split(p, max(1, len(p) // 200000)))
                         / (n * m ** 3))
        m = np.array(refinements, dtype=float)
        design = np.column_stack([np.ones_like(m)] + [1.0 / m ** power for power in (3, 5, 7, 9)][:len(m) - 1])
        average = np.linalg.lstsq(design, np.array(means), rcond=None)[0][0] + 1.0 / (4.0 * np.pi ** 1.5 * np.sqrt(alpha))
        supported = np.stack(np.meshgrid(*[np.arange(N) for N in self.shape], indexing="ij"), axis=-1).reshape(-1, 3)
        transfers = np.zeros((1, 3)) if transfers is None else np.atleast_2d(np.asarray(transfers, dtype=float))
        discrete = 0.0
        for transfer in transfers:
            coarse = self.symbol(supported + transfer)
            regular = coarse > 1e-12 * np.abs(self._entries).max()
            discrete += np.where(regular, 1.0 / np.where(regular, coarse, 1.0), 0.0).sum() / (n * len(transfers))
        return self.strength * (average - discrete)

    def inverse_symbol(self, kappa=None, zero_momentum=None):
        """1 / a(G + kappa) on the wavevectors G the cell supports, in the
        ordering of the discrete Fourier transform: the Coulomb kernel on
        charges of crystal momentum `kappa` (reciprocal coordinates of the
        cell), the inverse of the stiffness matrix dressed by that momentum.

        At `kappa = None` the entry at G = 0 is zero: the inverse on the
        complement of the constants. `zero_momentum` replaces the entry at
        G = 0, at any momentum, by the value that gives a normalized charge the
        energy `zero_momentum`; with `zero_momentum_constant()` this is the
        auxiliary-function treatment of that entry, which is continuous in the
        momentum."""
        if kappa is None:
            inverse = self._inverse.copy()
        else:
            grid = np.stack(np.meshgrid(*[np.arange(N) for N in self.shape], indexing="ij"), axis=-1).reshape(-1, 3)
            inverse = (1.0 / self.symbol(grid + np.asarray(kappa, dtype=float))).reshape(self.shape)
        if zero_momentum is not None:
            inverse.flat[0] = zero_momentum * self.size / self.strength
        return inverse

    def potential_derivative(self, rho, axis, kappa=None):
        """The derivative of `potential(rho, kappa, zero_momentum)` with respect
        to the Cartesian component `axis` of the crystal momentum: the Fourier
        multiplier -strength a'(G + kappa) / a(G + kappa)^2 with the gradient of
        the symbol in closed form, a'(k) = -sum_n A_0n dx_n sin(k . dx_n). At the
        zone centre (`kappa = None`) the entry at G = 0 is a constant of the
        momentum and has no derivative."""
        grid = np.stack(np.meshgrid(*[np.arange(N) for N in self.shape], indexing="ij"), axis=-1).reshape(-1, 3)
        if kappa is not None:
            grid = grid + np.asarray(kappa, dtype=float)
        angle = 2.0 * np.pi * ((grid / np.array(self.shape)) @ self._offsets.T)
        displacement = (self._offsets / np.array(self.shape)) @ self._lattice
        gradient = -(np.sin(angle) @ (self._entries * displacement[:, axis])).reshape(self.shape)
        inverse = self._inverse if kappa is None else self.inverse_symbol(kappa)
        multiplier = -gradient * inverse ** 2
        rho = np.asarray(rho, dtype=complex)
        columns = rho.reshape(self.size, -1)
        field = columns.T.reshape((-1,) + self.shape)
        solved = np.fft.ifftn(np.fft.fftn(field, axes=(1, 2, 3)) * multiplier, axes=(1, 2, 3))
        return (self.strength * solved.reshape(-1, self.size).T).reshape(rho.shape)

    def momentum_entry_limit(self, direction):
        """The limit of `momentum_entry(kappa)` times q^2 as the momentum tends
        to zero along the Cartesian `direction`: strength / (n q^ . H q^ / 2)
        with H = -sum_n A_0n dx_n dx_n^T the Hessian of the symbol. Linear
        functions are reproduced exactly by piecewise-linear elements, so
        H / 2 = (V / n) 1 and the limit is strength / V."""
        direction = np.asarray(direction, dtype=float)
        direction = direction / np.linalg.norm(direction)
        displacement = (self._offsets / np.array(self.shape)) @ self._lattice
        hessian = -(displacement * self._entries[:, None]).T @ displacement
        return self.strength / (self.size * 0.5 * float(direction @ hessian @ direction))

    def kinetic_modes(self, count, kappa=None):
        """The `count` lowest eigenpairs of the kinetic pencil dressed by the
        crystal momentum `kappa`, in closed form. The stiffness and the mass
        matrix commute with the grid translations, so the eigenvectors are the
        lattice plane waves exp(2 pi i G . n / N) / sqrt(n m(G + kappa)) with the
        eigenvalues a(G + kappa) / m(G + kappa), m being the symbol of the mass
        matrix. At the zone centre the constant (G = 0, eigenvalue zero) is left
        out. Returns (eigenvalues, the flat indices of the wavevectors in the
        ordering of the discrete Fourier transform, the mass symbol there);
        `mode_coefficients` projects loads on them."""
        supported = np.stack(np.meshgrid(*[np.arange(N) for N in self.shape], indexing="ij"), axis=-1).reshape(-1, 3)
        shift = np.zeros(3) if kappa is None else np.asarray(kappa, dtype=float)
        p = (supported + shift) / np.array(self.shape)
        phase = np.exp(2j * np.pi * (p @ self._offsets.T))
        stiffness, mass = np.real(phase @ self._entries), np.real(phase @ self._mass_entries)
        values = stiffness / mass
        order = np.argsort(values, kind="stable")
        if kappa is None or not np.any(shift):
            order = order[order != 0]
        order = order[:count]
        return values[order], order, mass[order]

    def mode_coefficients(self, loads, indices, mass):
        """B^dagger load for the plane-wave modes of `kinetic_modes` (rows: modes;
        columns: the columns of `loads`)."""
        loads = np.asarray(loads, dtype=complex).reshape(self.size, -1)
        field = loads.T.reshape((-1,) + self.shape)
        transformed = np.fft.fftn(field, axes=(1, 2, 3)).reshape(-1, self.size).T
        return transformed[indices] / np.sqrt(self.size * mass)[:, None]

    def auxiliary_function(self, kappa=None):
        """F(q) = (strength / n) sum_G 1 / a(G + q), the smooth periodic
        auxiliary function whose average over every momentum, minus this sum at
        the sampled momenta, is `zero_momentum_constant`. At the zone centre
        the entry at G = 0 is left out."""
        return self.strength * float(self.inverse_symbol(kappa).sum()) / self.size

    def momentum_entry(self, kappa):
        """The energy of a normalized charge of crystal momentum `kappa` in the
        G = 0 entry of the kernel, strength / (n a(kappa)); strength / (V q^2)
        in the continuum."""
        return self.strength / (self.size * float(self.symbol(np.asarray(kappa, dtype=float))[0]))

    def potential(self, rho, kappa=None, zero_momentum=None):
        """The potential of the load vectors `rho` (columns), at the crystal
        momentum `kappa` when given; see `inverse_symbol`."""
        rho = np.asarray(rho, dtype=complex)
        columns = rho.reshape(self.size, -1)
        grid = columns.T.reshape((-1,) + self.shape)
        inverse = self._inverse if kappa is None and zero_momentum is None else self.inverse_symbol(kappa, zero_momentum)
        solved = np.fft.ifftn(np.fft.fftn(grid, axes=(1, 2, 3)) * inverse, axes=(1, 2, 3))
        return (self.strength * solved.reshape(-1, self.size).T).reshape(rho.shape)

    def energy(self, rho_a, rho_b=None):
        rho_b = rho_a if rho_b is None else rho_b
        return np.vdot(rho_a, self.potential(rho_b))


def pair_loads(cell, x, Y, kappa_y=None, kappa_x=None):
    """The loads int phi_c x y of the product of a section x of the crystal
    momentum `kappa_x` with every column of Y, sections of `kappa_y`
    (`chainhodge.PairLoads` with the Bloch links of the two momenta; None is
    the zone centre). For conj(psi') psi between the momenta k' and k pass
    x = conj(z'), kappa_x = -k', kappa_y = k: the load carries k - k'. With
    kappa_x = None it is the dressed weighted mass matrix M_0^U[x] applied to Y.
    Real functions at the zone centre have real loads."""
    columns = np.asarray(Y).reshape(cell.size, -1)
    loads = cell.pair_loader.loads(cell.bloch_link_array(kappa_x), cell.bloch_link_array(kappa_y),
                                   np.asarray(x, dtype=complex), columns.astype(complex, copy=False))
    real = kappa_x is None and kappa_y is None and np.isrealobj(x) and np.isrealobj(columns)
    return loads.real if real else loads


def pair_loads_derivative(cell, x, Y, axis, kappa_y=None, kappa_x=None):
    """The derivative of `pair_loads(cell, x, Y, kappa_y, kappa_x)` with respect
    to the Cartesian component `axis` of the crystal momentum of Y
    (`chainhodge.PairLoads.loadsPhaseDerivativeAlong` with the displacements of
    the edges along that axis as weights)."""
    columns = np.asarray(Y).reshape(cell.size, -1)
    return cell.pair_loader.loadsPhaseDerivativeAlong(
        cell.bloch_link_array(kappa_x), cell.bloch_link_array(kappa_y), np.asarray(x, dtype=complex),
        columns.astype(complex, copy=False), np.ascontiguousarray(cell.edge_displacements[:, axis]))


def pair_densities(complex_, squared_lengths, modes):
    """`T[:, m, n] = rho^{mn}`: the load vectors of conj(z_m) z_n for every pair
    of the columns of `modes`, shape (vertices, modes, modes)
    (`chainhodge.PairLoads` at the trivial connection)."""
    modes = np.asarray(modes, dtype=complex)
    loader = ch.PairLoads(complex_, list(squared_lengths))
    trivial = np.ones(loader.numEdges, dtype=complex)
    count = modes.shape[1]
    T = np.empty((modes.shape[0], count, count), dtype=complex)
    for m in range(count):
        T[:, m, :] = loader.loads(trivial, trivial, modes[:, m].conj(), modes)
    return T


class ModeInteraction:
    """The Coulomb interaction in a basis of M-orthonormal modes, and
    Hartree-Fock on it.

    `energies` are the one-particle levels of the modes, `T` their pair
    densities, and `sheets` the number of identical spin copies (1 or 2): with
    two sheets the mode list is doubled, sheet by sheet, and the pair density
    is diagonal in the sheet. `zero_momentum` is the entry of the kernel at zero
    momentum transfer when the kernel itself has zero mean
    (`GridCoulombKernel.zero_momentum_constant`): the pair density of modes m
    and p has the charge <m|p> there, so the term is -c Gamma in exchange, and
    in the direct part it cancels against the ions of a neutral cell.
    """

    def __init__(self, kernel, energies, T, sheets=1, zero_momentum=0.0):
        self.kernel = kernel
        self.zero_momentum = float(zero_momentum)
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

    @classmethod
    def from_integrals(cls, kernel, one_particle, integrals, sheets=1, zero_momentum=0.0):
        """The interaction from the one-particle matrix of the modes and their
        Coulomb integrals `integrals[m, n, p, q] = (mn|pq)` already computed,
        without the pair densities (`density_operator` and `fock_hamiltonian`
        need those and are not available)."""
        self = cls.__new__(cls)
        self.kernel, self.zero_momentum, self.sheets = kernel, float(zero_momentum), int(sheets)
        self.orbitals = len(one_particle)
        self.size = self.orbitals * self.sheets
        self.h = np.kron(np.eye(self.sheets), np.asarray(one_particle, dtype=complex))
        self.T, self.W = None, np.asarray(integrals)
        return self

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
                F[a * n:(a + 1) * n, b * n:(b + 1) * n] -= np.einsum("mpqn,pq->mn", self.W, blocks[a][b]) \
                    + self.zero_momentum * blocks[a][b]
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
# The same for a face-centred cubic lattice whose conventional cube has side 1.
MADELUNG_FCC = 4.584862074


def probe_charge_constant(side, lattice="sc", e2=2.0):
    """e^2 MADELUNG / side: the integral of the Coulomb kernel over the cell of
    momentum space that a finite momentum set leaves out at zero momentum
    transfer, for the supercell the set is equivalent to (a simple cubic or a
    face-centred cubic lattice of conventional side `side`). `e2` is the squared
    charge in the units of the run (2 in rydberg atomic units)."""
    return e2 * {"sc": MADELUNG_SC, "fcc": MADELUNG_FCC}[lattice] / side


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
