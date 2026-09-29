# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""A self-consistent crystal in the mean field of the Coulomb interaction with
norm-conserving pseudopotentials, twice: on the periodic mesh, and in plane
waves as the reference. Rydberg atomic units throughout (see `pseudopotential`).

Both calculations solve the same problem. The ionic local potential is split
as `V_loc = V_sr + V_lr`, with `V_lr` the potential of a Gaussian charge -Z of
width `width` at every ion. On the mesh that charge is a source of the same
Coulomb kernel the electrons interact through, which acts on the complement of
the constants. In plane waves it is the Fourier series of the continuum kernel,
whose divergent average cancels against the electrons and leaves the uniform
remainder 4 pi Z width^2 / Omega (`PlaneWaveCrystal.alignment`), a shift of
every level that the mesh does not carry.
The Hartree potential has zero mean. Exchange is the nonlocal operator of the
filled orbitals, compressed onto the computed bands. The nonlocal part of the pseudopotential is the
separable sum over ions and projectors.

On the mesh the potentials enter the pencil as weighted mass matrices of their
vertex values, the Hartree potential solves the finite-element Poisson equation
for the exact load vector of the density, and the separable part is a term
`P D P^T` of low rank, `P = M beta` with `beta` the vertex values of the
projector functions, applied inside the shift-invert solve by the Woodbury
identity. That the shift lies below the whole spectrum is certified by inertia:
with `B = A - sigma M` positive definite, the number of eigenvalues of
`B + P D P^T` below zero is the number of negative eigenvalues of the
capacitance `D^{-1} + P^T B^{-1} P` minus that of `D^{-1}`.

The conventional cubic cell of a face-centred crystal sampled at its zone
centre is the primitive cell sampled at the zone centre and the three X points,
which is how the plane-wave reference runs it.
"""
import itertools
from dataclasses import dataclass, field

import numpy as np
import scipy.linalg
import scipy.sparse as sp
from scipy.special import spherical_jn

from tessera import chainhodge as ch
from tessera.drivers.bands import coulomb
from tessera.drivers.bands.crystal import CrystalCell
from tessera.drivers.bands.pseudopotential import real_harmonics

COULOMB_STRENGTH = 8.0 * np.pi        # 4 pi e^2 with e^2 = 2 Ry bohr


@dataclass
class Crystal:
    """Lattice vectors (rows, bohr) and ions as (pseudopotential, fractional position)."""
    lattice: np.ndarray
    ions: list
    electrons: int = field(init=False)

    def __post_init__(self):
        self.lattice = np.asarray(self.lattice, dtype=float)
        self.electrons = int(round(sum(p.valence for p, _ in self.ions)))

    @property
    def volume(self):
        return abs(np.linalg.det(self.lattice))

    @classmethod
    def zinc_blende(cls, a, cation, anion, conventional=True):
        """The zinc-blende crystal of lattice constant `a` (bohr): the cation on
        the face-centred sites and the anion displaced by a (1, 1, 1) / 4."""
        if not conventional:
            lattice = 0.5 * a * np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
            return cls(lattice, [(cation, np.zeros(3)), (anion, np.full(3, 0.25))])
        sites = [np.array(s) for s in ((0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0))]
        ions = [(cation, s) for s in sites] + [(anion, s + 0.25) for s in sites]
        return cls(a * np.eye(3), ions)


class PulayMixer:
    """Direct inversion in the iterative subspace on the potential: the next
    input is the combination of the stored inputs whose combined residual
    (output minus input) is smallest, moved along that residual by `step`.
    Plain mixing diverges once the cell is large enough to hold a
    charge-sloshing mode; this does not.

    Reference: Pulay, Chemical Physics Letters 73, 393 (1980)."""

    def __init__(self, step=0.3, history=6):
        self.step, self.history = float(step), int(history)
        self.inputs, self.residuals = [], []

    def next(self, current, produced):
        self.inputs.append(np.array(current, copy=True))
        self.residuals.append(np.asarray(produced) - np.asarray(current))
        self.inputs, self.residuals = self.inputs[-self.history:], self.residuals[-self.history:]
        m = len(self.inputs)
        overlap = np.array([[np.vdot(a.ravel(), b.ravel()).real for b in self.residuals] for a in self.residuals])
        system = np.block([[overlap, np.ones((m, 1))], [np.ones((1, m)), np.zeros((1, 1))]])
        rhs = np.zeros(m + 1)
        rhs[-1] = 1.0
        try:
            weights = np.linalg.solve(system, rhs)[:m]
        except np.linalg.LinAlgError:
            weights = np.zeros(m)
            weights[-1] = 1.0
        return sum(w * (x + self.step * r) for w, x, r in zip(weights, self.inputs, self.residuals))


# ---------------------------------------------------------------- plane waves

def _radial_transform(pseudo, function, q, l=0, weight_r=2):
    """4 pi int r^weight_r f(r) j_l(q r) dr on the pseudopotential's radial mesh."""
    r = pseudo.r
    values = function * r ** weight_r
    return 4.0 * np.pi * np.trapezoid(values[None, :] * spherical_jn(l, np.outer(q, r)), r, axis=1)


class PlaneWaveCrystal:
    """The reference: plane waves up to the kinetic cutoff `cutoff` (Ry) at the
    crystal momenta `kpoints` (Cartesian, inverse bohr) with `weights`."""

    def __init__(self, crystal, kpoints, weights, cutoff, width=1.2):
        self.crystal, self.cutoff, self.width = crystal, float(cutoff), float(width)
        self.kpoints = [np.asarray(k, dtype=float) for k in kpoints]
        self.weights = np.asarray(weights, dtype=float) / np.sum(weights)
        A = crystal.lattice
        self.reciprocal = 2.0 * np.pi * np.linalg.inv(A).T
        reach = 2.0 * np.sqrt(cutoff) + max(np.linalg.norm(k) for k in self.kpoints)
        self.shape = tuple(int(2 * np.ceil(reach * np.linalg.norm(A[i]) / (2.0 * np.pi)) + 2) for i in range(3))
        indices = np.stack(np.meshgrid(*[np.fft.fftfreq(n, 1.0 / n) for n in self.shape], indexing="ij"), axis=-1)
        self.G = indices @ self.reciprocal
        self.G2 = (self.G ** 2).sum(axis=-1)
        self.integers = indices.astype(int)
        self._ion_tables()
        self.alignment = sum(4.0 * np.pi * pseudo.valence * self.width ** 2 / crystal.volume for pseudo, _ in crystal.ions)
        self.bases = [self._basis(k) for k in self.kpoints]

    def _ion_tables(self):
        crystal, omega = self.crystal, self.crystal.volume
        q = np.sqrt(self.G2).ravel()
        unique, inverse = np.unique(np.round(q, 10), return_inverse=True)
        self.ionic = np.zeros(q.shape, dtype=complex)
        self.atomic_density = np.zeros(q.shape, dtype=complex)
        phases = {}
        for pseudo, position in crystal.ions:
            tau = position @ crystal.lattice
            key = id(pseudo)
            if key not in phases:
                short = _radial_transform(pseudo, pseudo.short_range_at(pseudo.r, self.width), unique)
                nonzero = unique > 1e-12
                long = np.where(nonzero, -8.0 * np.pi * pseudo.valence * np.exp(-0.5 * self.width ** 2 * unique ** 2)
                                / np.where(nonzero, unique ** 2, 1.0), 4.0 * np.pi * pseudo.valence * self.width ** 2)
                density = np.trapezoid(pseudo.density[None, :] * spherical_jn(0, np.outer(unique, pseudo.r)),
                                       pseudo.r, axis=1)
                phases[key] = ((short + long)[inverse], density[inverse])
            structure = np.exp(-1j * (self.G.reshape(-1, 3) @ tau))
            self.ionic += structure * phases[key][0] / omega
            self.atomic_density += structure * phases[key][1] / omega
        self.ionic = self.ionic.reshape(self.shape)
        self.atomic_density = self.atomic_density.reshape(self.shape)

    def _basis(self, k):
        crystal, omega = self.crystal, self.crystal.volume
        q = self.G.reshape(-1, 3) + k
        keep = np.nonzero((q ** 2).sum(axis=1) <= self.cutoff)[0]
        q = q[keep]
        norm = np.linalg.norm(q, axis=1)
        columns, coefficients = [], []
        for pseudo, position in crystal.ions:
            tau = position @ crystal.lattice
            structure = np.exp(-1j * (q @ tau))
            for i, (l, r_beta) in enumerate(pseudo.projectors):
                radial = _radial_transform(pseudo, r_beta, norm, l, weight_r=1)
                for harmonic in real_harmonics(l, q):
                    columns.append((-1j) ** l * harmonic * radial * structure / np.sqrt(omega))
                    coefficients.append((id(pseudo), i, len(columns) - 1))
        P = np.array(columns).T if columns else np.zeros((len(keep), 0), dtype=complex)
        D = self._projector_coefficients(P.shape[1])
        return {"indices": self.integers.reshape(-1, 3)[keep], "kinetic": (q ** 2).sum(axis=1), "P": P, "D": D}

    def _projector_coefficients(self, rank):
        D = np.zeros((rank, rank))
        position = 0
        for pseudo, _ in self.crystal.ions:
            spans = []
            for i, (l, _) in enumerate(pseudo.projectors):
                spans.append((i, l, position))
                position += 2 * l + 1
            for i, l, start_i in spans:
                for j, l2, start_j in spans:
                    if l == l2 and pseudo.D[i, j] != 0.0:
                        for m in range(2 * l + 1):
                            D[start_i + m, start_j + m] = pseudo.D[i, j]
        return D

    def _hamiltonian(self, basis, potential_g):
        idx = basis["indices"]
        difference = idx[:, None, :] - idx[None, :, :]
        local = potential_g[difference[..., 0] % self.shape[0], difference[..., 1] % self.shape[1],
                            difference[..., 2] % self.shape[2]]
        return np.diag(basis["kinetic"]) + local + basis["P"] @ basis["D"] @ basis["P"].conj().T

    def _effective(self, density_r):
        density_g = np.fft.fftn(density_r) / density_r.size
        hartree = np.where(self.G2 > 1e-12, COULOMB_STRENGTH * density_g / np.where(self.G2 > 1e-12, self.G2, 1.0), 0.0)
        return self.ionic + hartree

    def run(self, bands, tolerance=1e-8, mixing=0.3, max_iterations=80):
        """The Hartree mean field, the direct Wick contraction alone, iterated to
        self-consistency: the starting point of `run_hartree_fock`. Returns a
        dict with the levels per crystal momentum (Ry), the density on the grid,
        and the history of the density residual."""
        crystal = self.crystal
        occupied = crystal.electrons // 2
        density = np.maximum(np.fft.ifftn(self.atomic_density).real * self.atomic_density.size, 1e-12)
        density *= crystal.electrons / (density.mean() * crystal.volume)
        history = []
        mixer = PulayMixer(mixing)
        for iteration in range(max_iterations):
            potential_g = self._effective(density)
            levels, new_density = [], np.zeros(self.shape)
            for basis, weight in zip(self.bases, self.weights):
                values, vectors = scipy.linalg.eigh(self._hamiltonian(basis, potential_g),
                                                    subset_by_index=[0, bands - 1])
                levels.append(values)
                for b in range(occupied):
                    grid = np.zeros(self.shape, dtype=complex)
                    idx = basis["indices"]
                    grid[idx[:, 0] % self.shape[0], idx[:, 1] % self.shape[1], idx[:, 2] % self.shape[2]] = vectors[:, b]
                    wave = np.fft.ifftn(grid) * grid.size / np.sqrt(crystal.volume)
                    new_density += 2.0 * weight * np.abs(wave) ** 2
            residual = np.sqrt(np.mean((new_density - density) ** 2)) * crystal.volume / crystal.electrons
            history.append(residual)
            if residual < tolerance:
                break
            density = np.maximum(mixer.next(density, new_density), 1e-12)
        return {"levels": levels, "density": density, "history": history, "converged": history[-1] < tolerance,
                "planes": [len(b["kinetic"]) for b in self.bases]}


    # -- Hartree-Fock

    def _waves(self, basis, vectors):
        """The cell-periodic parts u(r) = sum_G c_G e^{i G . r} of the columns of `vectors`."""
        idx = basis["indices"]
        out = []
        for b in range(vectors.shape[1]):
            grid = np.zeros(self.shape, dtype=complex)
            grid[idx[:, 0] % self.shape[0], idx[:, 1] % self.shape[1], idx[:, 2] % self.shape[2]] = vectors[:, b]
            out.append(np.fft.ifftn(grid) * grid.size)
        return out

    def _exchange(self, k, basis, targets, waves, occupied, madelung):
        """The exchange operator applied to the cell-periodic parts `targets`
        of sections of momentum `k`, as plane-wave coefficients on `basis`, with
        the filled bands `waves` of the momentum set. The kernel carries the
        momentum transfer; its divergent entry (no transfer within the set) is
        replaced by the probe-charge term, at the momenta of the set and
        continuously away from them."""
        idx = basis["indices"]
        omega = self.crystal.volume
        columns = []
        for u_i in targets:
            total = np.zeros(self.shape, dtype=complex)
            for k_other, weight in enumerate(self.weights):
                q = self.G + (k - self.kpoints[k_other])
                q2 = (q ** 2).sum(axis=-1)
                kernel = COULOMB_STRENGTH / np.maximum(q2, 1e-300)
                nearest = np.unravel_index(np.argmin(q2), q2.shape)
                if q2[nearest] < 1e-2 * (self.reciprocal ** 2).sum(axis=1).min():
                    kernel[nearest] = madelung * omega / weight
                for u_j in waves[k_other][:occupied]:
                    pair = np.fft.fftn(np.conj(u_j) * u_i) / (u_i.size * omega)
                    total -= weight * u_j * (np.fft.ifftn(kernel * pair) * u_i.size)
            coefficients = np.fft.fftn(total) / total.size
            columns.append(coefficients[idx[:, 0] % self.shape[0], idx[:, 1] % self.shape[1], idx[:, 2] % self.shape[2]])
        return np.array(columns).T

    def _probe_constant(self, supercell_side):
        return (coulomb.probe_charge_constant(supercell_side, "sc") if np.isscalar(supercell_side)
                else coulomb.probe_charge_constant(*supercell_side))

    def run_hartree_fock(self, bands, supercell_side, tolerance=1e-6, mixing=0.3, max_outer=40, max_inner=30,
                         log=None):
        """Hartree-Fock: the Hartree run supplies the starting orbitals, then
        the exchange operator is added. Exchange
        is compressed onto the computed bands (it is exact on them), and the
        zero-momentum term that the momentum set leaves out is restored by the
        probe-charge correction -2 MADELUNG / supercell_side on the filled bands,
        `supercell_side` being the side of the cubic supercell the momentum set
        is equivalent to, or a pair (side, "fcc") for a face-centred one (a
        uniform grid of n^3 momenta of a face-centred crystal is the
        face-centred supercell of conventional side n a).

        Two nested loops. The outer one rebuilds the exchange operator from the
        current orbitals; the inner one converges the Hartree potential at fixed
        exchange with Pulay mixing, whose premise (the output is a function of
        the mixed input) holds only there."""
        crystal = self.crystal
        occupied = crystal.electrons // 2
        density = self.run(bands)["density"]
        madelung = self._probe_constant(supercell_side)
        potential_g = self._effective(density)
        current = [scipy.linalg.eigh(self._hamiltonian(basis, potential_g), subset_by_index=[0, bands - 1])[1]
                   for basis in self.bases]

        def density_of(vectors):
            out = np.zeros(self.shape)
            for basis, weight, v in zip(self.bases, self.weights, vectors):
                for u in self._waves(basis, v[:, :occupied]):
                    out += 2.0 * weight * np.abs(u) ** 2 / crystal.volume
            return out

        def hartree_of(n):
            n_g = np.fft.fftn(n) / n.size
            return np.where(self.G2 > 1e-12, COULOMB_STRENGTH * n_g / np.where(self.G2 > 1e-12, self.G2, 1.0), 0.0)

        history, levels = [], None
        density = density_of(current)
        for outer in range(max_outer):
            waves = [self._waves(basis, v) for basis, v in zip(self.bases, current)]
            compressed = []
            for k_index, psi in enumerate(current):
                W = self._exchange(self.kpoints[k_index], self.bases[k_index], waves[k_index], waves, occupied,
                                   madelung)
                overlap = psi.conj().T @ W
                factor = np.linalg.cholesky(-0.5 * (overlap + overlap.conj().T))
                compressed.append(W @ np.linalg.inv(factor).conj().T)
            mixer, inner_density = PulayMixer(mixing), density
            inner_tolerance = max(0.3 * tolerance, 0.3 * (history[-1] if history else 1e-2))
            for inner in range(max_inner):
                local_g = self.ionic + hartree_of(inner_density)
                levels, produced = [], []
                for basis, xi in zip(self.bases, compressed):
                    values, v = scipy.linalg.eigh(self._hamiltonian(basis, local_g) - xi @ xi.conj().T,
                                                  subset_by_index=[0, bands - 1])
                    levels.append(values)
                    produced.append(v)
                out_density = density_of(produced)
                inner_change = np.sqrt(np.mean((out_density - inner_density) ** 2)) * crystal.volume / crystal.electrons
                if inner_change < inner_tolerance:
                    break
                inner_density = np.maximum(mixer.next(inner_density, out_density), 1e-12)
            change = np.sqrt(np.mean((out_density - density) ** 2)) * crystal.volume / crystal.electrons
            history.append(change)
            if log:
                log(f"  plane waves, exchange update {outer:2d}: density change {change:.2e} ({inner + 1} inner)")
            current, density = produced, out_density
            if change < tolerance:
                break
        return {"levels": levels, "vectors": current, "history": history, "converged": history[-1] < tolerance,
                "local": self.ionic + hartree_of(density), "madelung": madelung}

    def dielectric_constant(self, result, shift, tolerance=1e-8, max_iterations=20):
        """The independent-particle macroscopic dielectric constant of a
        converged Hartree-Fock run along the small momentum `shift`,

            1 + (8 pi / V q^2) sum_k w_k sum_ia 4 |<u_ik | u_a,k+q>|^2 / (e_a(k + q) - e_i(k)) ,

        with the Hartree-Fock sections at k + q solved for in the potential and
        with the filled bands of the run; the overlaps of cell-periodic parts
        are exact in plane waves. The twin of `MeshCrystal.bands_at` and
        `momentum_pairs` on any momentum set."""
        crystal = self.crystal
        occupied = crystal.electrons // 2
        shift = np.asarray(shift, dtype=float)
        waves = [self._waves(basis, v) for basis, v in zip(self.bases, result["vectors"])]
        total = 0.0
        for k, weight, basis, levels, vectors in zip(self.kpoints, self.weights, self.bases,
                                                     result["levels"], result["vectors"]):
            shifted = self._basis(k + shift)
            local = self._hamiltonian(shifted, result["local"])
            # The sections of the run, on the plane waves the two bases share, start the compression.
            position = {tuple(index): row for row, index in enumerate(basis["indices"])}
            rows = np.array([position.get(tuple(index), -1) for index in shifted["indices"]])
            current = np.where(rows[:, None] >= 0, vectors[np.maximum(rows, 0)], 0.0)
            previous = None
            for _ in range(max_iterations):
                W = self._exchange(k + shift, shifted, self._waves(shifted, current), waves, occupied,
                                   result["madelung"])
                overlap = current.conj().T @ W
                xi = W @ np.linalg.inv(np.linalg.cholesky(-0.5 * (overlap + overlap.conj().T))).conj().T
                values, current = scipy.linalg.eigh(local - xi @ xi.conj().T, subset_by_index=[0, vectors.shape[1] - 1])
                if previous is not None and np.abs(values - previous).max() < tolerance:
                    break
                previous = values
            padded = np.zeros((len(basis["indices"]), current.shape[1]), dtype=complex)
            padded[rows[rows >= 0]] = current[rows >= 0]
            overlaps = vectors[:, :occupied].conj().T @ padded[:, occupied:]              # filled x empty
            gaps = values[occupied:][None, :] - levels[:occupied][:, None]
            total += weight * np.sum(np.abs(overlaps) ** 2 / gaps)
        return 1.0 + COULOMB_STRENGTH / (crystal.volume * (shift ** 2).sum()) * 4.0 * total


# ---------------------------------------------------------------- the mesh

def solve_with_projectors(A, M, P, D, count, sigma, tolerance=1e-10):
    """The lowest `count` eigenpairs of (A + P D P^T, M) with the low-rank term
    in factored form, by the certified block solver
    (`SparsePencilSolver.lowestWithLowRank`): Woodbury inside the shift-invert
    solve, every copy of a degenerate level returned, and the shift certified
    below the whole spectrum by inertia. Returns (values, vectors, residual,
    below) with real vectors for a real pencil: the residual of the returned
    pairs and the inertia certificate are reported separately, and a caller
    holds each to its own threshold."""
    read = ch.SparsePencilSolver.lowestWithLowRank(
        sp.csc_matrix(A, dtype=complex), sp.csc_matrix(M, dtype=complex),
        np.asarray(P, dtype=complex), np.asarray(D, dtype=complex), count, sigma, tolerance=tolerance)
    values = np.array(read.eigenvalues.values).real
    vectors = np.asarray(read.vectors)
    if np.isrealobj(A.data if sp.issparse(A) else A) and np.isrealobj(P):
        vectors = _real_span(A, M, P, D, vectors)
    certificate = read.eigenvalues.certificate
    return values, vectors, certificate.residual, bool(read.shiftBelowSpectrum)


def _real_span(A, M, P, D, vectors):
    """Real M-orthonormal eigenvectors spanning the complex ones of a real
    pencil with a real low-rank term, without assembling the term."""
    stacked = np.hstack([vectors.real, vectors.imag])
    gram = stacked.T @ (M @ stacked)
    weights, basis = np.linalg.eigh(0.5 * (gram + gram.T))
    keep = weights > 1e-10 * weights.max()
    span = stacked @ (basis[:, keep] / np.sqrt(weights[keep]))
    projected = span.T @ (A @ span) + (span.T @ P) @ D @ (P.T @ span)
    _, rotation = np.linalg.eigh(0.5 * (projected + projected.T))
    return (span @ rotation)[:, :vectors.shape[1]]      # see screening.real_modes on a cut level


class MeshCrystal:
    """The same calculation on the periodic mesh, at the zone centre."""

    def __init__(self, crystal, divisions, width=1.2, approximations=None):
        from tessera.drivers.bands.settings import Approximations
        self.approximations = Approximations() if approximations is None else approximations
        self.crystal, self.width = crystal, float(width)
        self.cell = CrystalCell(crystal.lattice, divisions, kinetic_scale=1.0)
        cell = self.cell
        self.stiffness = cell.stiffness.dressed().real.tocsc()
        self.mass = cell.mass.dressed().real.tocsc()
        self.kernel = coulomb.GridCoulombKernel(cell, COULOMB_STRENGTH)
        # The zero-momentum term of the kernel that sampling the cell at its zone
        # centre leaves out, from the kernel's own symbol.
        self.zero_momentum = self.kernel.zero_momentum_constant(self.approximations.refinements)
        self.ionic = self._ionic_potential()
        self.P, self.D = self._projectors()

    def _displacements(self, position):
        """Vertex positions relative to the nearest periodic image of an ion."""
        offset = self.cell.fractional - np.asarray(position, dtype=float)
        offset -= np.rint(offset)
        return offset @ self.cell.lattice

    def _ionic_potential(self):
        cell, crystal = self.cell, self.crystal
        values = np.zeros(cell.size)
        # Short range: a direct sum over the nearest images (it decays like the Gaussian).
        for pseudo, position in crystal.ions:
            for image in itertools.product(self.approximations.images, repeat=3):
                offset = (cell.fractional - position - np.asarray(image)) @ cell.lattice
                radius = np.linalg.norm(offset, axis=1)
                near = radius < 6.0 * self.width + 2.0
                values[near] += pseudo.short_range_at(radius[near], self.width)
        # Long range: the Gaussian charge -Z of every ion is a source of the same Coulomb
        # kernel the electrons interact through, the inverse of the stiffness matrix on the
        # complement of the constants. The load of each ion carries its charge exactly.
        weights = self.kernel.weights
        charge = np.zeros(cell.size)
        for pseudo, position in crystal.ions:
            gaussian = np.zeros(cell.size)
            for image in itertools.product(self.approximations.images, repeat=3):
                offset = (cell.fractional - position - np.asarray(image)) @ cell.lattice
                gaussian += np.exp(-0.5 * (offset ** 2).sum(axis=1) / self.width ** 2)
            charge += pseudo.valence * gaussian / (weights @ gaussian)
        return values - self.kernel.potential(self.mass @ charge).real

    def _projectors(self):
        columns, blocks = [], []
        for pseudo, position in self.crystal.ions:
            offset = self._displacements(position)
            radius = np.linalg.norm(offset, axis=1)
            for i, (l, _) in enumerate(pseudo.projectors):
                radial = pseudo.projector_at(i, radius)
                for m, harmonic in enumerate(real_harmonics(l, offset)):
                    columns.append(radial * harmonic)
                    blocks.append((id(pseudo), len(columns) - 1))
        beta = np.array(columns).T
        self._beta = beta
        self._beta_ion = np.concatenate([np.full(sum(2 * l + 1 for l, _ in pseudo.projectors), index)
                                         for index, (pseudo, _) in enumerate(self.crystal.ions)]).astype(int)
        rank = beta.shape[1]
        D = np.zeros((rank, rank))
        position = 0
        for pseudo, _ in self.crystal.ions:
            spans = []
            for i, (l, _) in enumerate(pseudo.projectors):
                spans.append((i, l, position))
                position += 2 * l + 1
            for i, l, start_i in spans:
                for j, l2, start_j in spans:
                    if l == l2 and pseudo.D[i, j] != 0.0:
                        for m in range(2 * l + 1):
                            D[start_i + m, start_j + m] = pseudo.D[i, j]
        if self.approximations.projector_quadrature == 0:
            return self.mass @ beta, D
        self._loads = self._projector_loads()
        return np.hstack([local.assemble() for local in self._loads]), D

    def _projector_loads(self):
        """The loads int beta(r - tau) lambda_v(r) dr of every ion's projector
        functions by quadrature on the tetrahedra (`loads.SimplexQuadrature`), the
        radial tables read as `Pseudopotential.projector_at` reads them, summed
        over the images of the ion within the reach of each table."""
        from tessera.drivers.bands import loads
        rule = loads.SimplexQuadrature(self.cell, self.approximations.projector_quadrature)
        out = []
        for pseudo, position in self.crystal.ions:
            def functions(offsets, pseudo=pseudo):
                radius = np.linalg.norm(offsets, axis=1)
                return np.array([pseudo.projector_at(i, radius) * harmonic
                                 for i, (l, _) in enumerate(pseudo.projectors)
                                 for harmonic in real_harmonics(l, offsets)]).T
            if not pseudo.projectors:
                out.append(loads.LocalLoads(self.cell.size, np.zeros(0, dtype=int), np.zeros((0, 3)), np.zeros((0, 0))))
                continue
            reach = max(loads.radial_reach(pseudo.r[1:], r_beta[1:] / pseudo.r[1:]) for _, r_beta in pseudo.projectors)
            out.append(rule.loads(functions, position, reach, self.approximations.images))
        return out

    def _projector_derivative(self, axis, mass_derivative):
        """The derivative of the projector loads with respect to the Cartesian
        component `axis` of the crystal momentum, at the zone centre."""
        if self.approximations.projector_quadrature:
            return np.hstack([local.derivative(axis) for local in self._loads])
        shifted = np.zeros(self._beta.shape, dtype=complex)                     # P_q = M_q (beta exp(-i q . (x - tau)))
        for index, (_, position) in enumerate(self.crystal.ions):
            columns = self._beta_ion == index
            shifted[:, columns] = -1j * self._displacements(position)[:, axis, None] * self._beta[:, columns]
        return mass_derivative @ self._beta + self.mass @ shifted

    def _projectors_at(self, kappa, mass):
        """The projector loads at the crystal momentum `kappa`. A separable
        term sum_R |beta_R> D <beta_R| acts on the cell-periodic part of a
        section of momentum k through beta(r - tau) exp(-i k . (r - tau)) summed
        over images: by quadrature the load of every image carries the phase of
        its displacement to the vertex (`loads.LocalLoads.assemble`); the
        interpolant is loaded with the mass matrix dressed by the same momentum."""
        k = self.cell.momentum(kappa)
        if self.approximations.projector_quadrature:
            return np.hstack([local.assemble(k) for local in self._loads])
        dressed = self._beta.astype(complex)
        for index, (_, position) in enumerate(self.crystal.ions):
            phase = np.exp(-1j * (self._displacements(position) @ k))
            columns = self._beta_ion == index
            dressed[:, columns] *= phase[:, None]
        return mass @ dressed

    @property
    def triple(self):
        if not hasattr(self, "_triple"):
            self._triple = coulomb.TripleIntegrals(self.cell.complex, self.cell.squared_lengths)
        return self._triple

    def _exchange(self, filled, orbitals, kappa=None):
        """K applied to the columns of `orbitals`, sections of crystal momentum
        `kappa` (the zone centre when None), with the filled zone-centre
        orbitals `filled`: one Poisson solve per pair, at the momentum the pair
        density carries, the entry of the kernel at G = 0 being the
        auxiliary-function constant (`GridCoulombKernel.inverse_symbol`)."""
        loads = (lambda x, Y: self.triple.loads(x, Y)) if kappa is None \
            else (lambda x, Y: coulomb.pair_loads(self.cell, x, Y, kappa))
        W = np.zeros(orbitals.shape, dtype=float if kappa is None else complex)
        for j in range(filled.shape[1]):
            pair = self.kernel.potential(loads(filled[:, j], orbitals), kappa, self.zero_momentum)
            W -= loads(filled[:, j], pair if kappa is not None else pair.real)
        return W

    @staticmethod
    def _compress(W, orbitals):
        """xi with K = -xi xi^dagger on the span of `orbitals`, from W = K orbitals."""
        overlap = orbitals.conj().T @ W
        return W @ np.linalg.inv(np.linalg.cholesky(-0.5 * (overlap + overlap.conj().T))).conj().T

    def _compressed(self, W, orbitals, projectors):
        """The low-rank term of the pencil: the projectors of the ions and the
        exchange operator compressed onto `orbitals`, K = -xi xi^dagger."""
        xi = self._compress(W, orbitals)
        rank = projectors.shape[1]
        P = np.hstack([projectors, xi])
        D = np.zeros((P.shape[1], P.shape[1]))
        D[:rank, :rank] = self.D
        D[rank:, rank:] = -np.eye(xi.shape[1])
        return P, D

    def atomic_density(self):
        values = np.zeros(self.cell.size)
        for pseudo, position in self.crystal.ions:
            for image in itertools.product(self.approximations.images, repeat=3):
                offset = (self.cell.fractional - position - np.asarray(image)) @ self.cell.lattice
                values += pseudo.density_at(np.linalg.norm(offset, axis=1))
        weights = self.kernel.weights
        return values * self.crystal.electrons / (weights @ values)

    def effective_potential(self, nodal_density, load=None):
        """V_ion + V_H at the vertices. The Hartree potential is solved for the
        load vector of the density: the exact one of the occupied orbitals when
        given, the interpolated one otherwise."""
        load = self.mass @ nodal_density if load is None else load
        return self.ionic + self.kernel.potential(load).real

    def run(self, bands, tolerance=1e-7, mixing=0.3, max_iterations=60, log=None):
        """The Hartree mean field, the direct Wick contraction alone, iterated to
        self-consistency at the zone centre: the starting point of
        `run_hartree_fock`. Returns a dict with the levels (Ry), the
        certificates of the last solve, and the history of the density residual."""
        cell, crystal = self.cell, self.crystal
        occupied = crystal.electrons // 2
        weights = self.kernel.weights
        density = self.atomic_density()
        potential = self.effective_potential(density)
        history = []
        mixer = PulayMixer(mixing)
        for iteration in range(max_iterations):
            weighted = cell.weighted_mass(potential).dressed().real.tocsc()
            A = (self.stiffness + weighted).tocsc()
            # Below the spectrum, and close to it once a lowest level is known.
            sigma = float(potential.min()) - 2.0 if iteration == 0 else float(values[0]) - 0.5
            values, vectors, residual, below = solve_with_projectors(A, self.mass, self.P, self.D, bands, sigma)
            orbitals = vectors[:, :occupied]
            new_density = 2.0 * (orbitals ** 2).sum(axis=1)
            load = 2.0 * np.array(ch.WhitneyMass.vertexDensityContraction(
                cell.complex, cell.squared_lengths, orbitals.astype(complex), orbitals.astype(complex))).real
            change = np.sqrt(weights @ (new_density - density) ** 2 / crystal.volume) * crystal.volume / crystal.electrons
            history.append(change)
            if log:
                log(f"  iteration {iteration:2d} density change {change:.2e} gap "
                    f"{(values[occupied] - values[occupied - 1]) * 13.605693:.4f} eV")
            if change < tolerance:
                break
            # The potential is what is mixed; the density is kept to measure the change.
            density = new_density
            potential = mixer.next(potential, self.effective_potential(new_density, load))
        return {"levels": values, "vectors": vectors, "residual": residual, "shift_below_spectrum": below,
                "history": history, "converged": history[-1] < tolerance, "spacing": cell.spacing,
                "potential": potential,
                "certified": bool(below and residual < 1e-8 and history[-1] < tolerance)}

    # -- Hartree-Fock

    def run_hartree_fock(self, bands, tolerance=1e-5, mixing=0.3, max_outer=100, max_inner=30, start=None, log=None):
        """Hartree-Fock on the mesh at the zone centre, the mean field of the
        quartic Coulomb interaction: the Hartree potential of the density and the
        exchange operator

            (K psi)(r) = - sum_{j filled} psi_j(r) int psi_j(r') psi(r') v(r - r') dr' ,

        both through the finite-element Coulomb kernel. Exchange is applied to
        every computed band (one Poisson solve per pair with a filled band, the
        product taken as a triple integral of piecewise-linear functions) and
        compressed onto them, K = -xi xi^T, which is exact on the computed bands
        and joins the pseudopotential's projectors in the low-rank term of the
        pencil. The kernel has zero mean, which drops the zero-momentum term of
        exchange; it is restored on the filled bands by the auxiliary-function
        correction with the kernel's own symbol as the auxiliary function
        (`GridCoulombKernel.zero_momentum_constant`).
        The Hartree mean field supplies the starting orbitals when it
        converges, one diagonalization in the potential of the atomic density
        otherwise. The exchange operator is rebuilt in an outer loop and the Hartree potential converged
        at fixed exchange in an inner one, as in `PlaneWaveCrystal`.

        References: Lin, Journal of Chemical Theory and Computation 12, 2242
        (2016), for the compression; Gygi & Baldereschi, Physical Review B 34,
        4405 (1986), for the zero-momentum term."""
        cell, crystal = self.cell, self.crystal
        occupied = crystal.electrons // 2
        integrals = self.triple
        weights = self.kernel.weights
        # The Hartree mean field starts the loop when it has a self-consistent state. Without exchange a
        # semiconductor can be gapless (gallium arsenide is, to 0.03 eV), and the filling of a gapless spectrum
        # does not converge; one diagonalization in the potential of the atomic density starts the loop then.
        # Hartree-Fock has more than one stationary state, and which one a loop reaches depends on its start:
        # the electronic energy is returned so that states can be compared, the lowest being the mean field.
        # `start` (a dict with "vectors" and "levels": an earlier run, or `prolonged` from a coarser mesh)
        # replaces both.
        if start is None:
            start = self.run(bands, log=log)
            if not start["converged"]:
                start = self.run(bands, max_iterations=1, log=log)
        start = dict(start)
        start.setdefault("residual", 0.0)
        start.setdefault("shift_below_spectrum", True)
        orbitals, values = np.asarray(start["vectors"])[:, :bands], np.asarray(start["levels"])[:bands]

        def load_of(vectors):
            filled = vectors[:, :occupied]
            return 2.0 * sum(integrals.loads(filled[:, j], filled[:, j:j + 1])[:, 0] for j in range(occupied))

        nodal = lambda vectors: 2.0 * (vectors[:, :occupied] ** 2).sum(axis=1)
        norm = lambda difference: np.sqrt(weights @ difference ** 2 / crystal.volume) * crystal.volume / crystal.electrons
        history, density = [], nodal(orbitals)
        residual, below = start["residual"], start["shift_below_spectrum"]
        for outer in range(max_outer):
            # The exchange operator on every computed band, compressed: K = -xi xi^T.
            P, D = self._compressed(self._exchange(orbitals[:, :occupied], orbitals), orbitals, self.P)
            # The Hartree potential converged at this exchange.
            mixer = PulayMixer(mixing)
            hartree = self.kernel.potential(load_of(orbitals)).real
            inner_tolerance = max(0.3 * tolerance, 0.3 * (history[-1] if history else 1e-2))
            inner_density = density
            for inner in range(max_inner):
                A = (self.stiffness + cell.weighted_mass(self.ionic + hartree).dressed().real).tocsc()
                values, produced, residual, below = solve_with_projectors(A, self.mass, P, D, bands,
                                                                          float(values[0]) - 1.0)
                out_density = nodal(produced)
                inner_change = norm(out_density - inner_density)
                inner_density = out_density
                if inner_change < inner_tolerance:
                    break
                hartree = mixer.next(hartree, self.kernel.potential(load_of(produced)).real)
            change = norm(out_density - density)
            history.append(change)
            if log:
                log(f"  mesh, exchange update {outer:2d}: density change {change:.2e} ({inner + 1} inner) gap "
                    f"{(values[occupied] - values[occupied - 1]) * 13.605693:.4f} eV")
            orbitals, density = produced, out_density
            if change < tolerance:
                break
        # E = sum over the filled orbitals of (h_ii + e_i), h the kinetic and ionic part.
        filled = orbitals[:, :occupied]
        ionic = (self.stiffness + cell.weighted_mass(self.ionic).dressed().real).tocsc()
        overlap = filled.T @ self.P
        one_particle = np.einsum("vi,vi->i", filled, ionic @ filled) + np.einsum("ip,pq,iq->i", overlap, self.D, overlap)
        energy = float(np.sum(one_particle + values[:occupied]))
        return {"levels": values, "vectors": orbitals, "residual": residual, "shift_below_spectrum": below,
                "history": history, "converged": history[-1] < tolerance, "spacing": cell.spacing, "energy": energy,
                "certified": bool(below and residual < 1e-8 and history[-1] < tolerance)}

    # -- more bands and the quasiparticle equation on a momentum set

    def _shifted(self, vectors, shift):
        """The cell-periodic parts of the same sections written at the momentum
        kappa + shift, `shift` a reciprocal vector of the cell (integers): the
        lattice plane wave of `shift` moves from the link phases to the vertex
        values, exactly."""
        phase = np.exp(-2j * np.pi * (self.cell.index @ (np.asarray(shift, dtype=float) / np.array(self.cell.divisions))))
        return np.asarray(vectors) * phase[:, None]

    def _set_exchange(self, momenta, constant, filled, k, targets):
        """K_k applied to `targets` (sections of momenta[k]) with the filled
        sections `filled[k']` of every momentum of the set; see
        `run_hartree_fock_set`."""
        cell, count = self.cell, len(momenta)
        W = np.zeros(targets.shape, dtype=complex)
        for other in range(count):
            transfer = tuple(a - b for a, b in zip(momenta[k], momenta[other]))
            same = other == k
            minus = tuple(-v for v in momenta[other])
            for j in range(filled[other].shape[1]):
                z = filled[other][:, j]
                loads = coulomb.pair_loads(cell, z.conj(), targets, momenta[k], minus)
                potential = self.kernel.potential(loads, None if same else transfer, count * constant if same else None)
                W -= coulomb.pair_loads(cell, z, potential, transfer, momenta[other]) / count
        return W

    def extend_bands_set(self, run, bands, tolerance=1e-5, max_iterations=10, log=None):
        """Converge `bands` Hartree-Fock levels at every momentum of a converged
        `run_hartree_fock_set`, the filled sections (and with them the Hartree
        potential and the exchange operator) fixed, as `extend_bands` does at
        the zone centre."""
        cell = self.cell
        occupied = self.crystal.electrons // 2
        momenta, count = run["momenta"], len(run["momenta"])
        filled = [np.asarray(v)[:, :occupied] for v in run["vectors"]]
        load = np.zeros(cell.size)
        for k in range(count):
            minus = tuple(-v for v in momenta[k])
            for j in range(occupied):
                z = filled[k][:, j]
                load += 2.0 / count * coulomb.pair_loads(cell, z.conj(), z[:, None], momenta[k], minus)[:, 0].real
        local = self.ionic + self.kernel.potential(load).real
        levels, vectors, converged = [], [], True
        for k in range(count):
            A, M = cell.pencil(momenta[k], cell.weighted_mass(local))
            projectors = self._projectors_at(momenta[k], M)
            orbitals, values, previous = np.asarray(run["vectors"][k]).astype(complex), np.asarray(run["levels"][k]), None
            for iteration in range(max_iterations):
                W = self._set_exchange(momenta, run["zero_momentum"], filled, k, orbitals)
                P, D = self._compressed(W, orbitals, projectors)
                values, orbitals, _, _ = solve_with_projectors(A, M, P, D, bands, float(values[0]) - 1.0)
                orbitals = orbitals.astype(complex)
                change = np.inf if previous is None or len(previous) != len(values) else np.abs(values - previous).max()
                previous = values
                if log:
                    log(f"  momentum {momenta[k]}, bands {bands}, compression {iteration}: {change:.2e} Ry")
                if change < tolerance:
                    break
            converged = converged and change < tolerance
            levels.append(values)
            vectors.append(orbitals)
        return {"momenta": momenta, "levels": levels, "vectors": vectors, "occupied": occupied,
                "zero_momentum": run["zero_momentum"], "converged": bool(converged)}

    def quasiparticle_set(self, extended, states, bands=None):
        """The quasiparticle equation on a momentum set, for the states
        (momentum index, band) in `states`: the screened interaction of the
        random-phase approximation at every momentum transfer of the set, its
        pairs running from a filled section at k to an empty one at k + q for
        every k, and the self-energy of a state summed over the transfers,

            Sigma_c(n k; w) = sum_q sum_m sum_t |w^t_{nk, m k+q}|^2 / (w - e_m(k + q) -+ W_t(q)) .

        Orbitals are normalized in the cell, so every Coulomb integral carries
        1 / N_k, the normalization in the supercell the set is equivalent to. A
        sum k + q that leaves the first zone is brought back by a reciprocal
        vector, moved into the vertex values (`_shifted`), so that every pair
        of one transfer carries exactly that transfer. The entry of the kernel
        at zero transfer and G = 0 is left out here (the term `RandomPhase`
        restores with `set_head` at the zone centre).

        `bands` is the number of bands kept, one number or one per momentum.
        Returns {state: (mean-field level, quasiparticle level, renormalization)}."""
        cell = self.cell
        occupied = int(extended["occupied"])
        momenta, count = [np.asarray(kappa, dtype=float) for kappa in extended["momenta"]], len(extended["momenta"])
        bands = len(extended["levels"][0]) if bands is None else bands
        bands = [int(bands)] * count if np.isscalar(bands) else [int(b) for b in bands]      # per momentum

        def partner(k, c):
            """(index of momenta[k] + momenta[c] in the set, the reciprocal vector that brings it back)."""
            total = momenta[k] + momenta[c]
            for index, kappa in enumerate(momenta):
                shift = total - kappa
                if np.abs(shift - np.rint(shift)).max() < 1e-9:
                    return index, np.rint(shift)
            raise ValueError("the momentum set is not closed under addition")

        minus = lambda kappa: tuple(-v for v in kappa)
        classes = []
        for c in range(count):
            loads, gaps = [], []
            for k in range(count):
                k2, shift = partner(k, c)
                sections = self._shifted(np.asarray(extended["vectors"][k2])[:, occupied:bands[k2]], shift)
                for i in range(occupied):
                    z = np.asarray(extended["vectors"][k])[:, i]
                    loads.append(coulomb.pair_loads(cell, z.conj(), sections, tuple(momenta[k] + momenta[c]),
                                                    minus(momenta[k])))
                    gaps.append(np.asarray(extended["levels"][k2])[occupied:bands[k2]] - extended["levels"][k][i])
            loads, gaps = np.hstack(loads), np.concatenate(gaps)
            transfer = None if not np.any(momenta[c]) else tuple(momenta[c])
            potentials = self.kernel.potential(loads, transfer)
            coupling = loads.conj().T @ potentials / count
            root = np.sqrt(gaps)
            casida = np.diag(gaps ** 2) + 4.0 * root[:, None] * coupling * root[None, :]
            squared, Z = np.linalg.eigh(0.5 * (casida + casida.conj().T))
            omega = np.sqrt(squared)
            classes.append((omega, (root[:, None] * Z) / np.sqrt(omega)[None, :], potentials))
        out = {}
        for k, n in states:
            z = np.asarray(extended["vectors"][k])[:, n]
            level = float(extended["levels"][k][n])
            terms = []
            for c, (omega, modes, potentials) in enumerate(classes):
                k2, shift = partner(k, c)
                sections = self._shifted(np.asarray(extended["vectors"][k2])[:, :bands[k2]], shift)
                state_loads = coulomb.pair_loads(cell, z.conj(), sections, tuple(momenta[k] + momenta[c]),
                                                 minus(momenta[k]))
                weights = 2.0 * np.abs((state_loads.conj().T @ potentials / count) @ modes) ** 2      # bands x modes
                levels = np.asarray(extended["levels"][k2])[:bands[k2]]
                poles = np.where((np.arange(bands[k2]) < occupied)[:, None], levels[:, None] - omega[None, :],
                                 levels[:, None] + omega[None, :])
                terms.append((weights, poles))
            energy = level
            for _ in range(60):
                value = sum(np.sum(w / (energy - p)) for w, p in terms)
                slope = -sum(np.sum(w / (energy - p) ** 2) for w, p in terms)
                step = (level + value - energy) / (1.0 - slope)
                energy += step
                if abs(step) < 1e-11:
                    break
            out[(k, n)] = (level, float(energy), float(1.0 / (1.0 - slope)))
        return out

    def prolonged(self, coarse, run):
        """The start of `run_hartree_fock` on this mesh from a converged run on the
        coarser mesh `coarse` (a `MeshCrystal` of the same crystal): the orbitals
        as the piecewise-linear functions they are, evaluated at the vertices of
        this mesh. In a Kuhn cube the simplex of a point is read off the order
        of its fractional coordinates, and its barycentric weights are their
        successive differences. When the divisions are multiples the coarse
        functions lie in the fine space and nothing is lost."""
        ratio = np.array(coarse.cell.divisions) / np.array(self.cell.divisions)
        position = self.cell.index * ratio
        base = np.floor(position + 1e-12).astype(int)
        fraction = position - base
        order = np.argsort(-fraction, axis=1, kind="stable")
        sorted_fraction = np.take_along_axis(fraction, order, axis=1)
        weights = np.column_stack([1.0 - sorted_fraction[:, 0], sorted_fraction[:, 0] - sorted_fraction[:, 1],
                                   sorted_fraction[:, 1] - sorted_fraction[:, 2], sorted_fraction[:, 2]])
        n = np.array(coarse.cell.divisions)
        corner = base.copy()
        vectors = np.zeros((self.cell.size, np.asarray(run["vectors"]).shape[1]))
        for step in range(4):
            wrapped = corner % n
            ids = (wrapped[:, 0] * n[1] + wrapped[:, 1]) * n[2] + wrapped[:, 2]
            vectors += weights[:, step, None] * np.asarray(run["vectors"]).real[ids]
            if step < 3:
                corner[np.arange(len(corner)), order[:, step]] += 1
        return {"vectors": vectors, "levels": np.asarray(run["levels"])}

    # -- Hartree-Fock on a momentum set

    def run_hartree_fock_set(self, bands, momenta, tolerance=1e-5, mixing=0.3, max_outer=40, max_inner=30, log=None):
        """Hartree-Fock with the covariance sampled on the momentum set
        `momenta` (reciprocal coordinates of the cell): a uniform grid that
        contains the zone centre, so that the differences of its members are
        its members again. At each momentum the pencil is the one dressed by
        the flat connection of that momentum. The exchange operator carries the
        momentum transfer k - k' in its kernel,

            (K_k z)(r) = -(1 / N_k) sum_{k' j} psi_jk'(r) int v_{k-k'}(r - r') conj(psi_jk'(r')) psi(r') dr' ,

        with the pair densities loaded between the two momenta
        (`coulomb.pair_loads`, the library's `WhitneyMass.pairLoads`) and the kernel the inverse of
        the stiffness matrix dressed by the transfer; at zero transfer its entry
        at G = 0 is the auxiliary-function constant of the set
        (`GridCoulombKernel.zero_momentum_constant(transfers=momenta)`), which
        is the constant of the supercell the set is equivalent to. The loops
        are those of `run_hartree_fock`. Returns the levels and the
        cell-periodic parts per momentum."""
        cell, crystal = self.cell, self.crystal
        occupied = crystal.electrons // 2
        momenta = [tuple(float(x) for x in kappa) for kappa in momenta]
        count = len(momenta)
        triple, weights = self.triple, self.kernel.weights
        constant = self.kernel.zero_momentum_constant(self.approximations.refinements, transfers=momenta)
        masses = [cell.pencil(kappa)[1] for kappa in momenta]
        projectors = [self._projectors_at(kappa, M) for kappa, M in zip(momenta, masses)]
        start = self.run(bands, log=log)

        def pencil(k, hartree):
            return cell.pencil(momenta[k], cell.weighted_mass(self.ionic + hartree))[0]

        def load_of(orbitals):
            total = np.zeros(cell.size)
            for k in range(count):
                for j in range(occupied):
                    z = orbitals[k][:, j]
                    total += 2.0 / count * coulomb.pair_loads(cell, z.conj(), z[:, None], momenta[k],
                                                              tuple(-v for v in momenta[k]))[:, 0].real
            return total

        def exchange(k, orbitals):
            return self._set_exchange(momenta, constant, [v[:, :occupied] for v in orbitals], k, orbitals[k])

        nodal = lambda orbitals: 2.0 / count * sum((np.abs(v[:, :occupied]) ** 2).sum(axis=1) for v in orbitals)
        norm = lambda difference: np.sqrt(weights @ difference ** 2 / crystal.volume) * crystal.volume / crystal.electrons
        hartree = self.kernel.potential(self.mass @ self.atomic_density()).real
        orbitals, values = [], []
        for k in range(count):                                       # the Hartree mean field of the zone centre starts every momentum
            A = cell.pencil(momenta[k], cell.weighted_mass(start["potential"]))[0]
            v, z, _, _ = solve_with_projectors(A, masses[k], projectors[k], self.D, bands, float(start["levels"][0]) - 1.0)
            orbitals.append(z.astype(complex)); values.append(v)
        history, density = [], nodal(orbitals)
        residual, below = 0.0, True
        for outer in range(max_outer):
            terms = [self._compressed(exchange(k, orbitals), orbitals[k], projectors[k]) for k in range(count)]
            mixer = PulayMixer(mixing)
            hartree = self.kernel.potential(load_of(orbitals)).real
            inner_tolerance = max(0.3 * tolerance, 0.3 * (history[-1] if history else 1e-2))
            inner_density = density
            for inner in range(max_inner):
                produced, residual, below = [], 0.0, True
                for k in range(count):
                    values[k], z, r, b = solve_with_projectors(pencil(k, hartree), masses[k], *terms[k], bands,
                                                               float(values[k][0]) - 1.0)
                    produced.append(z.astype(complex)); residual = max(residual, r); below = below and b
                out_density = nodal(produced)
                inner_change = norm(out_density - inner_density)
                inner_density = out_density
                if inner_change < inner_tolerance:
                    break
                hartree = mixer.next(hartree, self.kernel.potential(load_of(produced)).real)
            change = norm(out_density - density)
            history.append(change)
            if log:
                log(f"  momentum set, exchange update {outer:2d}: density change {change:.2e} ({inner + 1} inner)")
            orbitals, density = produced, out_density
            if change < tolerance:
                break
        return {"momenta": momenta, "levels": values, "vectors": orbitals, "residual": residual,
                "shift_below_spectrum": below, "history": history, "converged": history[-1] < tolerance,
                "zero_momentum": constant, "spacing": cell.spacing,
                "certified": bool(below and residual < 1e-8 and history[-1] < tolerance)}

    # -- more bands, and the one-shot quasiparticle correction

    def extend_bands(self, mean_field, bands, tolerance=1e-5, max_iterations=10, log=None):
        """Converge `bands` Hartree-Fock levels on top of a converged run of
        `run_hartree_fock`. The filled orbitals, and with them the Hartree
        potential and the exchange operator, are fixed; what is iterated is the
        compression of exchange, which is exact only on the bands it was built
        from and so has to be rebuilt on the larger set until the empty levels
        stop moving."""
        cell, crystal = self.cell, self.crystal
        occupied = crystal.electrons // 2
        integrals = self.triple
        filled = mean_field["vectors"][:, :occupied]
        load = 2.0 * sum(integrals.loads(filled[:, j], filled[:, j:j + 1])[:, 0] for j in range(occupied))
        hartree = self.kernel.potential(load).real
        A = (self.stiffness + cell.weighted_mass(self.ionic + hartree).dressed().real).tocsc()
        orbitals, values = mean_field["vectors"], mean_field["levels"]
        previous = None
        for iteration in range(max_iterations):
            P, D = self._compressed(self._exchange(filled, orbitals), orbitals, self.P)
            values, orbitals, residual, below = solve_with_projectors(A, self.mass, P, D, bands,
                                                                       float(values[0]) - 1.0)
            change = np.inf if previous is None or len(previous) != len(values) else np.abs(values - previous).max()
            if log:
                log(f"  bands {bands}, compression {iteration}: largest level change {change:.2e} Ry")
            previous = values
            if change < tolerance:
                break
        return {"levels": values, "vectors": orbitals, "residual": residual, "shift_below_spectrum": below,
                "converged": bool(change < tolerance), "occupied": occupied,
                "local_potential": self.ionic + hartree}

    def covariance_certificate(self, extended, bands, step=0.05, steps=20):
        """The converged state as a `CovarianceState` on the lowest `bands`
        Hartree-Fock modes with two sheets, and the Fock operator rebuilt there
        as the Wick contraction of the Coulomb kernel (`ModeInteraction`), a
        second route to the one the pencil was solved with. Returns the purity
        defect and the particle number of the state, the largest entry of
        F(Gamma) - diag(levels), which holds the two routes to each other on
        the span of those modes, and the largest change of Gamma under
        `meanFieldEvolve`, which vanishes at a fixed point."""
        from tessera import quantum
        cell = self.cell
        occupied = int(extended["occupied"])
        modes = np.asarray(extended["vectors"])[:, :bands]
        levels = np.asarray(extended["levels"])[:bands]
        ionic = (self.stiffness + cell.weighted_mass(self.ionic).dressed().real).tocsc()
        overlap = modes.T @ self.P
        one_particle = modes.T @ (ionic @ modes) + overlap @ self.D @ overlap.T
        T = coulomb.pair_densities(cell.complex, cell.squared_lengths, modes)
        interaction = coulomb.ModeInteraction(self.kernel, levels, T, sheets=2, zero_momentum=self.zero_momentum)
        interaction.h = np.kron(np.eye(2), one_particle).astype(complex)
        frame = np.zeros((2 * bands, 2 * occupied), dtype=complex)
        for sheet in range(2):
            frame[sheet * bands + np.arange(occupied), sheet * occupied + np.arange(occupied)] = 1.0
        state = quantum.CovarianceState.fromSlaterFrame(frame)
        gamma = np.array(state.gamma())
        fock = interaction.fock(gamma)
        state.meanFieldEvolve(lambda g: interaction.fock(g), step, steps)
        return {"purity_defect": float(state.purityDefect()), "particles": float(state.particleNumber().real),
                "fock_defect": float(np.abs(fock - np.kron(np.eye(2), np.diag(levels))).max()),
                "stationarity_defect": float(np.abs(np.array(state.gamma()) - gamma).max()),
                "energy": float(interaction.energy(gamma).real)}

    def bands_at(self, extended, kappa, tolerance=1e-5, max_iterations=12, exchange=True, converge=None, log=None):
        """The Hartree-Fock levels and the cell-periodic parts of their sections
        at the crystal momentum `kappa` (reciprocal coordinates of the cell), as
        many as `extended` holds: the pencil dressed by the flat connection of
        that momentum, with the Hartree potential and the filled orbitals of the
        zone centre. The exchange operator carries the momentum transfer in its
        kernel and, as in `extend_bands`, its compression is rebuilt until the
        levels stop moving; `converge` limits that test to the lowest levels
        (the compression converges slowly on the highest of a set, which a
        caller then leaves out of the pairs). `exchange=False` is for the
        levels of `run`, the Hartree mean field."""
        cell = self.cell
        occupied = int(extended["occupied"])
        filled = np.asarray(extended["vectors"])[:, :occupied]
        bands = len(extended["levels"])
        A, M = cell.pencil(kappa, cell.weighted_mass(extended["local_potential"]))
        projectors = self._projectors_at(kappa, M)
        orbitals, values = np.asarray(extended["vectors"]).astype(complex), np.asarray(extended["levels"])
        previous = None
        for iteration in range(max_iterations):
            P, D = (self._compressed(self._exchange(filled, orbitals, kappa), orbitals, projectors) if exchange
                    else (projectors, self.D))
            values, orbitals, residual, below = solve_with_projectors(A, M, P, D, bands, float(values[0]) - 1.0)
            change = np.inf if previous is None else np.abs(values - previous)[:converge].max()
            if log:
                log(f"  momentum {tuple(kappa)}, compression {iteration}: largest level change {change:.2e} Ry")
            previous = values
            if change < tolerance:
                break
        return {"levels": values, "vectors": orbitals, "kappa": tuple(kappa), "residual": residual,
                "shift_below_spectrum": below, "converged": bool(change < tolerance), "occupied": occupied}

    def momentum_pairs(self, extended, at_momentum, bands=None):
        """The particle-hole pairs of momentum transfer q: a filled orbital i of
        the zone centre and an empty section a of `bands_at` (filled index
        slow). Their pair densities conj(psi_i) psi_{a, q} carry the momentum q
        and are loaded as piecewise-linear integrals with the link phases of
        that momentum; the Coulomb kernel they meet is the inverse of the
        stiffness matrix dressed by it.

        This is the finite-momentum route to the response at vanishing
        momentum, which `vanishing_momentum_pairs` gives in closed form; the two
        agree at second order in the momentum.

        Returns the argument of `RandomPhase.set_head` for this momentum: the
        level differences e_a(q) - e_i(0), the coupling (ia|jb) without the
        G = 0 entry of the kernel, the charges (the G = 0 components of the
        loads), the energy of a normalized charge in that entry, and the modes
        the pairs are made of."""
        occupied = int(extended["occupied"])
        bands = len(extended["levels"]) if bands is None else int(bands)
        kappa = at_momentum["kappa"]
        filled = np.asarray(extended["vectors"])[:, :occupied]
        empties = np.asarray(at_momentum["vectors"])[:, occupied:bands]
        loads = [coulomb.pair_loads(self.cell, filled[:, i], empties, kappa) for i in range(occupied)]
        potentials = [self.kernel.potential(load, kappa, 0.0) for load in loads]
        coupling = np.block([[loads[i].conj().T @ potentials[j] for j in range(occupied)] for i in range(occupied)])
        gaps = (np.asarray(at_momentum["levels"])[None, occupied:bands]
                - np.asarray(extended["levels"])[:occupied, None]).ravel()
        return {"gaps": gaps, "coupling": coupling, "charges": np.concatenate([load.sum(axis=0) for load in loads]),
                "entry": self.kernel.momentum_entry(kappa),
                "pairs": [(i, a) for i in range(occupied) for a in range(occupied, bands)]}

    def momentum_term(self, extended, kappa, states, bands=None, buffer=8, log=None):
        """Everything the self-energy integrand S_n(q; w) of the modes `states`
        needs at the finite momentum transfer `kappa`: the Hartree-Fock levels
        and sections there (`bands_at`), the particle-hole pairs between the
        zone centre and that momentum with the whole Coulomb kernel of that
        momentum (the entry at G = 0 included, which is finite), and per mode n
        the integrals (n m_q | pair) over every section m_q. `RandomPhase`
        evaluates S_n from it (`set_momentum_terms`). `buffer` extra bands are
        solved for and left out, because the compression of exchange converges
        slowly on the highest of a set."""
        occupied = int(extended["occupied"])
        bands = len(extended["levels"]) - buffer if bands is None else int(bands)
        top = min(bands + buffer, len(extended["levels"]))
        truncated = {"levels": np.asarray(extended["levels"])[:top], "vectors": np.asarray(extended["vectors"])[:, :top],
                     "occupied": occupied, "local_potential": extended["local_potential"]}
        at_momentum = self.bands_at(truncated, kappa, converge=bands, log=log)
        sections = np.asarray(at_momentum["vectors"])[:, :bands]
        modes = np.asarray(extended["vectors"])
        loads = [coulomb.pair_loads(self.cell, modes[:, i], sections[:, occupied:], kappa) for i in range(occupied)]
        potentials = np.hstack([self.kernel.potential(load, kappa) for load in loads])
        coupling = np.hstack(loads).conj().T @ potentials
        blocks = {n: coulomb.pair_loads(self.cell, modes[:, n], sections, kappa).conj().T @ potentials for n in states}
        levels = np.asarray(at_momentum["levels"])[:bands]
        gaps = (levels[None, occupied:] - np.asarray(extended["levels"])[:occupied, None]).ravel()
        return {"kappa": tuple(kappa), "levels": levels, "gaps": gaps, "coupling": coupling, "blocks": blocks,
                "x": 1.0 / self.kernel.momentum_entry(kappa), "auxiliary": self.kernel.auxiliary_function(kappa),
                "converged": at_momentum["converged"],
                "pairs": [(i, a) for i in range(occupied) for a in range(occupied, bands)]}

    def vertex(self, extended, bands=None, include=()):
        """The argument tuple of `RandomPhase.set_vertex` for
        `approximations.self_energy_order`: the `vertex_bands` modes nearest the
        gap (half filled, half empty, among the lowest `bands`; the modes in
        `include` first, which in a folded cell need not be the nearest) and
        their Coulomb integrals (pq|rs), with the entry of the kernel at G = 0
        as in exchange."""
        settings = self.approximations
        occupied = int(extended["occupied"])
        bands = len(extended["levels"]) if bands is None else int(bands)
        chosen = sorted(int(n) for n in include)
        nearest = sorted(range(bands), key=lambda m: (abs(m - occupied + 0.5), m))
        half = settings.vertex_bands // 2
        for m in nearest:                                             # fill up, keeping the two sides balanced
            side = [c for c in chosen if (c < occupied) == (m < occupied)]
            if m not in chosen and len(chosen) < settings.vertex_bands and len(side) < max(half, settings.vertex_bands - half):
                chosen.append(m)
        for m in nearest:                                             # a side that ran out leaves room for the other
            if m not in chosen and len(chosen) < settings.vertex_bands:
                chosen.append(m)
        chosen = sorted(chosen)
        modes = np.asarray(extended["vectors"])[:, chosen]
        count = len(chosen)
        loads = np.hstack([self.triple.loads(modes[:, p], modes) for p in range(count)])           # pairs (p, q)
        potentials = self.kernel.potential(loads, None, self.zero_momentum).real
        interaction = (loads.T @ potentials).reshape(count, count, count, count)
        return settings.self_energy_order, chosen, interaction, settings.vertex_poles

    def momentum_terms(self, extended, states, bands=None, log=None):
        """The argument tuple of `RandomPhase.set_momentum_terms` for the
        momentum transfers of `approximations.zero_momentum_order` (empty at
        order 1, which leaves the closed form at vanishing momentum in place)."""
        nodes = self.approximations.momentum_nodes
        terms = [self.momentum_term(extended, kappa, states, bands, log=log) for kappa, _ in nodes]
        return terms, [weight for _, weight in nodes], self.zero_momentum + self.kernel.auxiliary_function()

    def vanishing_momentum_pairs(self, extended, coupling, bands=None, directions=None):
        """The limit of `momentum_pairs` as the momentum tends to zero along each
        of `directions` (the three Cartesian axes by default), in closed form. The pairs become those of the
        zone centre (their level differences and their coupling `coupling`,
        as in `coulomb_integrals`), and the charge of a pair per unit momentum
        is the derivative of 1^T M_0^U[psi_i] z_a(q),

            d_ia = 1^T (d M_0^U[psi_i]) z_a + psi_i^T (dH - e_a dM) z_a / (e_a - e_i) ,

        the second term being first-order perturbation theory, which needs no
        linear solve because the filled orbitals are eigenvectors. dH is the
        derivative of the Hartree-Fock pencil with respect to a uniform change
        of the link phases: entrywise for the stiffness matrix, the weighted
        mass matrix of the local potential and the mass matrix
        (`CovariantChainHodge.sparsePencilPhaseDerivativeAlong` with the edge
        displacements as weights), the product rule on the projector
        loads, and for exchange

            dK = - sum_j [ dL_j G L_j + L_j dG L_j + L_j G dL_j ] ,   L_j = M_0^U[psi_j] ,

        with the derivative of the Coulomb kernel from the gradient of its
        symbol (`GridCoulombKernel.potential_derivative`). The charge along a
        direction is the contraction of the three Cartesian derivatives with
        it. Returns one argument of `RandomPhase.set_head` per direction."""
        cell = self.cell
        occupied = int(extended["occupied"])
        bands = len(extended["levels"]) if bands is None else int(bands)
        levels = np.asarray(extended["levels"])[:bands]
        filled = np.asarray(extended["vectors"])[:, :occupied]
        empties = np.asarray(extended["vectors"])[:, occupied:bands]
        local = cell.weighted_mass(extended["local_potential"])
        weighted = [cell.weighted_mass(filled[:, j]) for j in range(occupied)]
        plain = [w.dressed().real for w in weighted]
        loads = [L @ empties for L in plain]                                    # L_j z_a
        filled_loads = [L @ filled for L in plain]                              # L_j psi_i
        filled_potentials = [self.kernel.potential(u, None, self.zero_momentum) for u in filled_loads]
        gaps = (levels[None, occupied:] - levels[:occupied, None])
        overlap, overlap_empty = filled.T @ self.P, empties.T @ self.P
        covariant = cell.covariant()
        derivatives = []
        for axis in range(3):
            weights = [float(w) for w in cell.edge_displacements[:, axis]]
            pencil = covariant.sparsePencilPhaseDerivativeAlong(weights)
            dM = sp.csc_matrix(pencil.M)
            dA = cell.kinetic_scale * sp.csc_matrix(pencil.A) + sp.csc_matrix(
                covariant.dressedVertexPotentialPhaseDerivativeAlong(list(local.values), weights))
            current = filled.T @ (dA @ empties) - (filled.T @ (dM @ empties)) * levels[None, occupied:]
            dP = self._projector_derivative(axis, dM)
            current = current + (filled.T @ dP) @ self.D @ overlap_empty.T + overlap @ self.D @ (dP.conj().T @ empties)
            # Exchange.
            direct = np.zeros((occupied, bands - occupied), dtype=complex)
            for j in range(occupied):
                dL = sp.csc_matrix(covariant.dressedVertexPotentialPhaseDerivativeAlong(list(weighted[j].values), weights))
                d_filled = dL @ filled
                first = self.kernel.potential(d_filled, None, self.zero_momentum).conj().T @ loads[j]
                second = self.kernel.potential_derivative(filled_loads[j], axis).conj().T @ loads[j]
                third = filled_potentials[j].conj().T @ (dL @ empties)
                current = current - (first + second + third)
                direct[j] = np.asarray(dL.sum(axis=0)).ravel() @ empties       # 1^T dM_0^U[psi_j] z_a
            derivatives.append((direct + current / gaps).ravel())
        out = []
        for direction in (np.eye(3) if directions is None else np.atleast_2d(np.asarray(directions, dtype=float))):
            direction = direction / np.linalg.norm(direction)
            out.append({"gaps": gaps.ravel(), "coupling": np.asarray(coupling),
                        "charges": sum(direction[axis] * derivatives[axis] for axis in range(3)),
                        "entry": self.kernel.momentum_entry_limit(direction), "limit": True,
                        "pairs": [(i, a) for i in range(occupied) for a in range(occupied, bands)]})
        return out

    def kinetic_basis(self, count):
        """The `count` lowest eigenpairs of the kinetic pencil above the
        constant, as real M-orthonormal vectors: the basis in which the Coulomb
        kernel is diagonal, strength / lambda (`KineticBasisScreening`). The
        constant has the eigenvalue zero and carries no entry of the kernel."""
        from tessera.drivers.bands.screening import real_modes
        read = self.cell.solve(count=count + 1, sigma=-1.0)
        values, vectors = real_modes(self.stiffness, self.mass, read.vectors)
        return values[1:], vectors[:, 1:]

    def basis_coefficients(self, extended, basis, bands=None, states=()):
        """The coefficients B^T load in the basis `basis` of the pair densities
        of the particle-hole pairs (filled index slow), and per mode n in
        `states` those of psi_n psi_m for every mode m."""
        occupied = int(extended["occupied"])
        bands = len(extended["levels"]) if bands is None else int(bands)
        modes = np.asarray(extended["vectors"])[:, :bands]
        pairs = np.vstack([(basis.T @ self.triple.loads(modes[:, i], modes[:, occupied:])).T for i in range(occupied)])
        return pairs, {n: (basis.T @ self.triple.loads(modes[:, n], modes)).T for n in states}

    def quasiparticle_levels(self, extended, states, log=None):
        """The one-shot GW correction on the Hartree-Fock levels of
        `extend_bands` for the modes listed in `states`, with the screened
        interaction of the direct random-phase approximation in the basis of
        all particle-hole pairs of the computed bands.

        The Coulomb integrals are formed with the Fourier kernel of the grid,
        one Poisson solve per particle-hole pair. That kernel has zero mean,
        which drops the zero-momentum term of the screened interaction exactly
        as it drops that of exchange. The exchange part was restored in the mean
        field by the auxiliary-function constant on the filled bands. The
        correlation part is restored here by the same constant times the
        inverse dielectric function at vanishing momentum, from the charges of
        the pairs per unit momentum in closed form
        (`vanishing_momentum_pairs`, `RandomPhase.set_head`), averaged over the
        three Cartesian axes.

        Returns a dict: per state the Hartree-Fock level, the quasiparticle
        level without (`body`) and with (`quasiparticle`) the zero-momentum
        term, the renormalization factor and the residual of the quasiparticle
        equation, all in rydberg."""
        from tessera.drivers.bands.screening import RandomPhase
        self.approximations.require_implemented()
        cell = self.cell
        occupied = extended["occupied"]
        energies, orbitals = extended["levels"], extended["vectors"]
        empties = orbitals[:, occupied:]
        integrals = self.triple
        potentials, loads = [], []
        for i in range(occupied):
            pair_loads = integrals.loads(orbitals[:, i], empties)
            loads.append(pair_loads)
            potentials.append(self.kernel.potential(pair_loads).real)
        coupling = np.block([[loads[i].T @ potentials[j] for j in range(occupied)] for i in range(occupied)])
        blocks = {}
        vertex = self.vertex(extended)
        for n in sorted(set(states) | (set(vertex[1]) if vertex[0] > 1 else set())):
            state_loads = integrals.loads(orbitals[:, n], orbitals)
            blocks[n] = np.hstack([state_loads.T @ potentials[j] for j in range(occupied)])
        if log:
            log(f"  random phase: {coupling.shape[0]} particle-hole pairs")
        rpa = RandomPhase.from_pieces(energies, occupied, coupling, blocks)
        constant = self.zero_momentum
        out = {n: {"mean_field": float(energies[n]), "body": float(rpa.quasiparticle(n)[0])} for n in states}
        dielectric = rpa.set_head(constant, self.vanishing_momentum_pairs(extended, coupling))
        rpa.set_momentum_terms(*self.momentum_terms(extended, states, log=log))
        rpa.set_vertex(*vertex)
        for n in states:
            energy, weight = rpa.quasiparticle(n)
            defect = abs(energies[n] + rpa.correlation(n, energy)[0] - energy)
            out[n].update({"quasiparticle": float(energy), "renormalization": float(weight), "defect": float(defect)})
        return {"states": out, "correlation_energy": float(rpa.correlation_energy()), "head_constant": constant,
                "dielectric_constant": float(dielectric), "head_defect": rpa.head_defect,
                "independent_particle_dielectric_constant": float(rpa.independent_particle_dielectric_constant),
                "pairs": coupling.shape[0]}

    def coulomb_integrals(self, extended, bands=None):
        """The Coulomb integrals of the lowest `bands` Hartree-Fock modes that
        the random-phase and quasiparticle steps need, for every mode: the
        particle-hole coupling (ia|jb) and, per mode n, (nm|jb) over every mode
        m and every pair, with one Poisson solve per particle-hole pair. The
        lowest modes of a larger converged set are the same Hartree-Fock modes,
        so a band-count study truncates one set.
        Returns (levels, occupied, coupling, integrals)."""
        cell = self.cell
        occupied = int(extended["occupied"])
        bands = len(extended["levels"]) if bands is None else int(bands)
        truncated = {"levels": np.asarray(extended["levels"])[:bands], "vectors": np.asarray(extended["vectors"])[:, :bands],
                     "occupied": occupied, "local_potential": np.asarray(extended["local_potential"])}
        energies, orbitals = truncated["levels"], truncated["vectors"]
        empties = orbitals[:, occupied:]
        triple = self.triple
        potentials, loads = [], []
        for i in range(occupied):
            pair_loads = triple.loads(orbitals[:, i], empties)
            loads.append(pair_loads)
            potentials.append(self.kernel.potential(pair_loads).real)
        coupling = np.block([[loads[i].T @ potentials[j] for j in range(occupied)] for i in range(occupied)])
        stacked = np.hstack(potentials)                                   # vertices x pairs
        integrals = {n: triple.loads(orbitals[:, n], orbitals).T @ stacked for n in range(bands)}
        return energies, occupied, coupling, integrals
