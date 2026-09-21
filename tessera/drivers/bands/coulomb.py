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

    def zero_momentum_constant(self, refinements=(2, 4, 8)):
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
        powers of the refinement from the third on, which the refinements
        remove. For the continuum
        kernel strength / (V q^2) on a simple cubic cell this constant is
        strength * MADELUNG_SC / (4 pi L)."""
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
        design = np.column_stack([np.ones_like(m), 1.0 / m ** 3, 1.0 / m ** 5][:len(m)])
        average = np.linalg.lstsq(design, np.array(means), rcond=None)[0][0] + 1.0 / (4.0 * np.pi ** 1.5 * np.sqrt(alpha))
        coarse = self.symbol(np.stack(np.meshgrid(*[np.arange(N) for N in self.shape], indexing="ij"),
                                      axis=-1).reshape(-1, 3))
        regular = coarse > 1e-12 * np.abs(self._entries).max()
        discrete = np.where(regular, 1.0 / np.where(regular, coarse, 1.0), 0.0).sum() / n
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

    def potential_derivative(self, rho, axis):
        """The derivative of `potential(rho, kappa, zero_momentum)` with respect
        to the Cartesian component `axis` of the crystal momentum, at the zone
        centre: the Fourier multiplier -strength a'(G) / a(G)^2 with the gradient
        of the symbol in closed form, a'(k) = -sum_n A_0n dx_n sin(k . dx_n). The
        entry at G = 0 is a constant of the momentum and has no derivative."""
        grid = np.stack(np.meshgrid(*[np.arange(N) for N in self.shape], indexing="ij"), axis=-1).reshape(-1, 3)
        angle = 2.0 * np.pi * ((grid / np.array(self.shape)) @ self._offsets.T)
        displacement = (self._offsets / np.array(self.shape)) @ self._lattice
        gradient = -(np.sin(angle) @ (self._entries * displacement[:, axis])).reshape(self.shape)
        multiplier = -gradient * self._inverse ** 2
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


def bloch_twist(cell, tops, kappa):
    """exp(2 pi i kappa . d) for the grid displacement d of every vertex of
    every top simplex from the simplex's first vertex: the link phases of the
    crystal momentum `kappa`, per simplex, for `TripleIntegrals.loads`."""
    n = np.array(cell.divisions)
    d = (cell.index[tops] - cell.index[tops[:, :1]]) % n
    d = np.where(d == n - 1, -1, d)
    return np.exp(2j * np.pi * ((d / n) @ np.asarray(kappa, dtype=float)))


class TripleIntegrals:
    """The integrals of three piecewise-linear functions against the vertex
    basis, per top simplex, vectorized over a complex.

    On a top simplex of volume |T| the three-factor integral is
    |T| mu_abc d! / (d + 3)! with mu_abc = 1 + delta_ab + delta_ac + delta_bc
    + 2 delta_ab delta_bc, so for two functions x and y restricted to it,

        sum_ab mu_abc x_a y_b = S_x S_y + sum_a x_a y_a + x_c S_y + S_x y_c + 2 x_c y_c .

    `loads(x, Y)` returns, for every column y of Y, the load vector
    int phi_c x y of the product: the vectorized form of
    `WhitneyMass.vertexDensityContraction`, and equally of `M_0[x] y`. With a
    `twist` (`bloch_twist`) the columns of Y are the cell-periodic parts of
    sections of crystal momentum kappa and the result is `M_0^U[x] y`, the
    weighted mass matrix dressed by the link phases of that momentum: the load
    of a pair density that carries the momentum kappa.
    """

    def __init__(self, complex_, squared_lengths):
        vertex_index = {int(cell[0]): i for i, cell in enumerate(complex_.kSimplexVertices(0))}
        self.size = len(vertex_index)
        self.tops = np.array([[vertex_index[int(v)] for v in cell] for cell in complex_.orientedTopSimplices()])
        d = self.tops.shape[1] - 1
        volumes = np.array(ch.WhitneyMass.certificate(complex_, list(squared_lengths)).volumes)
        if np.abs(volumes.imag).max() == 0.0:
            volumes = volumes.real                          # a real geometry keeps real loads real
        self.weight = volumes * float(np.prod(np.arange(1, d + 1))) / float(np.prod(np.arange(1, d + 4)))
        count = len(self.tops)
        self.scatter = [sp.csr_matrix((np.ones(count), (self.tops[:, c], np.arange(count))),
                                      shape=(self.size, count)) for c in range(d + 1)]

    def loads(self, x, Y, twist=None, block=48):
        x = np.asarray(x)
        Y = np.asarray(Y).reshape(self.size, -1)
        if Y.shape[1] > block:                                # bound the per-simplex temporaries
            return np.hstack([self.loads(x, Y[:, start:start + block], twist, block)
                              for start in range(0, Y.shape[1], block)])
        local_x = x[self.tops]                                # (tops, d + 1)
        local_y = Y[self.tops]                                # (tops, d + 1, columns)
        if twist is not None:
            local_y = local_y * twist[:, :, None]             # carried to the first vertex of the simplex
        sum_x, sum_y = local_x.sum(axis=1), local_y.sum(axis=1)
        common = sum_x[:, None] * sum_y + np.einsum("ta,tan->tn", local_x, local_y)
        total = np.zeros((self.size, Y.shape[1]), dtype=np.result_type(x, local_y, self.weight))
        for c, scatter in enumerate(self.scatter):
            term = common + local_x[:, c, None] * sum_y + sum_x[:, None] * local_y[:, c, :] \
                + 2.0 * local_x[:, c, None] * local_y[:, c, :]
            if twist is not None:
                term = term * twist[:, c, None].conj()         # and back to the vertex the load belongs to
            total += scatter @ (self.weight[:, None] * term)
        return total


def pair_densities(complex_, squared_lengths, modes):
    """`T[:, m, n] = rho^{mn}`: the load vectors of conj(z_m) z_n for every pair
    of the columns of `modes`, shape (vertices, modes, modes)."""
    modes = np.asarray(modes, dtype=complex)
    integrals = TripleIntegrals(complex_, squared_lengths)
    count = modes.shape[1]
    T = np.empty((modes.shape[0], count, count), dtype=complex)
    for m in range(count):
        T[:, m, :] = integrals.loads(modes[:, m].conj(), modes)
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
