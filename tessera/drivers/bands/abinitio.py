# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""A self-consistent crystal in the local density approximation with
norm-conserving pseudopotentials, twice: on the periodic mesh, and in plane
waves as the reference. Rydberg atomic units throughout (see `pseudopotential`).

Both calculations solve the same problem. The ionic local potential is split
as `V_loc = V_sr + V_lr`, with `V_lr` the potential of a Gaussian charge -Z of
width `width` at every ion, whose divergent average cancels against the
electrons and is dropped (its finite remainder 4 pi Z width^2 / Omega is kept).
The Hartree potential has zero mean. Exchange and correlation are evaluated
pointwise on the density. The nonlocal part of the pseudopotential is the
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
from tessera.drivers.bands.pseudopotential import lda_potential, real_harmonics

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
        exchange_correlation = np.fft.fftn(lda_potential(density_r)) / density_r.size
        return self.ionic + hartree + exchange_correlation

    def run(self, bands, tolerance=1e-8, mixing=0.3, max_iterations=80):
        """Iterate to self-consistency. Returns a dict with the levels per
        crystal momentum (Ry), the density on the grid, and the history of the
        density residual."""
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

    def _exchange(self, k_index, waves, occupied, madelung):
        """The exchange operator applied to every computed band at one momentum,
        as plane-wave coefficients: W_i = K psi_i, with the divergent term of the
        kernel dropped and replaced by the probe-charge term on the filled bands."""
        basis = self.bases[k_index]
        idx = basis["indices"]
        omega = self.crystal.volume
        columns = []
        for u_i in waves[k_index]:
            total = np.zeros(self.shape, dtype=complex)
            for k_other, weight in enumerate(self.weights):
                q = self.G + (self.kpoints[k_index] - self.kpoints[k_other])
                q2 = (q ** 2).sum(axis=-1)
                kernel = np.where(q2 > 1e-10, COULOMB_STRENGTH / np.where(q2 > 1e-10, q2, 1.0), 0.0)
                for u_j in waves[k_other][:occupied]:
                    pair = np.fft.fftn(np.conj(u_j) * u_i) / (u_i.size * omega)
                    total -= weight * u_j * (np.fft.ifftn(kernel * pair) * u_i.size)
            coefficients = np.fft.fftn(total) / total.size
            columns.append(coefficients[idx[:, 0] % self.shape[0], idx[:, 1] % self.shape[1], idx[:, 2] % self.shape[2]])
        return np.array(columns).T

    def run_hartree_fock(self, bands, supercell_side, tolerance=1e-6, mixing=0.3, max_outer=40, max_inner=30,
                         log=None):
        """Hartree-Fock: the local-density run supplies the starting orbitals,
        then the exchange operator replaces exchange and correlation. Exchange
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
        madelung = (coulomb.probe_charge_constant(supercell_side, "sc") if np.isscalar(supercell_side)
                    else coulomb.probe_charge_constant(*supercell_side))
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
                W = self._exchange(k_index, waves, occupied, madelung)
                W[:, :occupied] -= madelung * psi[:, :occupied]
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
        return {"levels": levels, "vectors": current, "history": history, "converged": history[-1] < tolerance}

    def dielectric_constant(self, result, step=1e-4):
        """The independent-particle macroscopic dielectric constant of a
        converged Hartree-Fock run, 1 + (8 pi / 3 V) sum_k w_k sum_ia 4 |r_ia|^2 /
        (e_a - e_i). The current vertex J = [H_0, r] is 2 (k + G) for the kinetic
        part plus the derivative of the separable projectors (central
        differences in k), and r follows from [H_0 + v, r] = J solved in the band
        basis with the local-density exchange-correlation potential as the local
        v, exactly as in `MeshCrystal.dipoles`, of which this is the plane-wave
        twin on any momentum set."""
        crystal = self.crystal
        occupied = crystal.electrons // 2
        density = np.zeros(self.shape)
        for basis, weight, v in zip(self.bases, self.weights, result["vectors"]):
            for u in self._waves(basis, v[:, :occupied]):
                density += 2.0 * weight * np.abs(u) ** 2 / crystal.volume
        auxiliary_g = self._effective(density)                  # ionic + Hartree + local exchange-correlation
        total = 0.0
        for k, weight, basis, levels, vectors in zip(self.kpoints, self.weights, self.bases,
                                                     result["levels"], result["vectors"]):
            keep = self._kept(k)
            q = self.G.reshape(-1, 3)[keep] + k
            auxiliary = vectors.conj().T @ (self._hamiltonian(basis, auxiliary_g) @ vectors)
            h, rotation = np.linalg.eigh(0.5 * (auxiliary + auxiliary.conj().T))
            difference = h[:, None] - h[None, :]
            safe = np.abs(difference) > 1e-6 * max(1.0, np.abs(h).max())
            gaps = levels[occupied:][None, :] - levels[:occupied][:, None]          # filled x empty
            for alpha in range(3):
                shift = np.zeros(3)
                shift[alpha] = step
                dP = (self._projectors_at(k + shift, keep) - self._projectors_at(k - shift, keep)) / (2.0 * step)
                P, D = basis["P"], basis["D"]
                current = vectors.conj().T @ ((2.0 * q[:, alpha])[:, None] * vectors)
                current += (vectors.conj().T @ dP) @ D @ (P.conj().T @ vectors) \
                    + (vectors.conj().T @ P) @ D @ (dP.conj().T @ vectors)
                rotated = rotation.conj().T @ current @ rotation
                inner = np.where(safe, rotated / np.where(safe, difference, 1.0), 0.0)
                inner[:occupied, :occupied] = 0.0
                inner[occupied:, occupied:] = 0.0
                position = rotation @ inner @ rotation.conj().T
                r = position[:occupied, occupied:]
                total += weight * np.sum(np.abs(r) ** 2 / gaps)
        return 1.0 + COULOMB_STRENGTH / (3.0 * crystal.volume) * 4.0 * total

    def _kept(self, k):
        q = self.G.reshape(-1, 3) + k
        return np.nonzero((q ** 2).sum(axis=1) <= self.cutoff)[0]

    def _projectors_at(self, k, keep):
        """The separable projectors on the plane waves `keep` of the basis of a
        nearby momentum, evaluated at the momentum `k`."""
        crystal, omega = self.crystal, self.crystal.volume
        q = self.G.reshape(-1, 3)[keep] + k
        norm = np.linalg.norm(q, axis=1)
        columns = []
        for pseudo, position in crystal.ions:
            structure = np.exp(-1j * (q @ (position @ crystal.lattice)))
            for l, r_beta in pseudo.projectors:
                radial = _radial_transform(pseudo, r_beta, norm, l, weight_r=1)
                for harmonic in real_harmonics(l, q):
                    columns.append((-1j) ** l * harmonic * radial * structure / np.sqrt(omega))
        return np.array(columns).T


# ---------------------------------------------------------------- the mesh

def solve_with_projectors(A, M, P, D, count, sigma, tolerance=1e-10):
    """The lowest `count` eigenpairs of (A + P D P^T, M) with the low-rank term
    in factored form, by the certified block solver
    (`SparsePencilSolver.lowestWithLowRank`): Woodbury inside the shift-invert
    solve, every copy of a degenerate level returned, and the shift certified
    below the whole spectrum by inertia. Returns (values, vectors, residual,
    below) with real vectors for a real pencil."""
    read = ch.SparsePencilSolver.lowestWithLowRank(
        sp.csc_matrix(A, dtype=complex), sp.csc_matrix(M, dtype=complex),
        np.asarray(P, dtype=complex), np.asarray(D, dtype=complex), count, sigma, tolerance=tolerance)
    values = np.array(read.eigenvalues.values).real
    vectors = np.asarray(read.vectors)
    if np.isrealobj(A.data if sp.issparse(A) else A) and np.isrealobj(P):
        vectors = _real_span(A, M, P, D, vectors)
    certificate = read.eigenvalues.certificate
    return values, vectors, certificate.residual, bool(read.shiftBelowSpectrum and certificate.holds())


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

    def __init__(self, crystal, divisions, width=1.2):
        self.crystal, self.width = crystal, float(width)
        self.cell = CrystalCell(crystal.lattice, divisions, kinetic_scale=1.0)
        cell = self.cell
        self.stiffness = cell.stiffness.dressed().real.tocsc()
        self.mass = cell.mass.dressed().real.tocsc()
        self.kernel = coulomb.GridCoulombKernel(cell, COULOMB_STRENGTH)
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
            for image in itertools.product((-1, 0, 1), repeat=3):
                offset = (cell.fractional - position - np.asarray(image)) @ cell.lattice
                radius = np.linalg.norm(offset, axis=1)
                near = radius < 6.0 * self.width + 2.0
                values[near] += pseudo.short_range_at(radius[near], self.width)
        # Long range: the Fourier series of the Gaussian charges, which the
        # vertices (a uniform grid) sample exactly through one inverse transform.
        shape = cell.divisions
        indices = np.stack(np.meshgrid(*[np.fft.fftfreq(n, 1.0 / n) for n in shape], indexing="ij"), axis=-1)
        G = indices @ cell.reciprocal
        G2 = (G ** 2).sum(axis=-1)
        series = np.zeros(shape, dtype=complex)
        for pseudo, position in crystal.ions:
            structure = np.exp(-1j * (G @ (position @ cell.lattice)))
            nonzero = G2 > 1e-12
            form = np.where(nonzero, -COULOMB_STRENGTH * pseudo.valence * np.exp(-0.5 * self.width ** 2 * G2)
                            / np.where(nonzero, G2, 1.0), 4.0 * np.pi * pseudo.valence * self.width ** 2)
            series += structure * form / crystal.volume
        return values + (np.fft.ifftn(series) * series.size).real.ravel()

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
        return self.mass @ beta, D

    def atomic_density(self):
        values = np.zeros(self.cell.size)
        for pseudo, position in self.crystal.ions:
            for image in itertools.product((-1, 0, 1), repeat=3):
                offset = (self.cell.fractional - position - np.asarray(image)) @ self.cell.lattice
                values += pseudo.density_at(np.linalg.norm(offset, axis=1))
        weights = self.kernel.weights
        return values * self.crystal.electrons / (weights @ values)

    def effective_potential(self, nodal_density, load=None):
        """V_ion + V_H + V_xc at the vertices. The Hartree potential is solved
        for the load vector of the density: the exact one of the occupied
        orbitals when given, the interpolated one otherwise."""
        load = self.mass @ nodal_density if load is None else load
        return self.ionic + self.kernel.potential(load).real + lda_potential(nodal_density)

    def run(self, bands, tolerance=1e-7, mixing=0.3, max_iterations=60, log=None):
        """Iterate to self-consistency at the zone centre. Returns a dict with
        the levels (Ry), the certificates of the last solve, and the history of
        the density residual."""
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
                "certified": bool(below and residual < 1e-8 and history[-1] < tolerance)}

    # -- Hartree-Fock

    def run_hartree_fock(self, bands, tolerance=1e-5, mixing=0.3, max_outer=40, max_inner=30, log=None):
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
        exchange; it is restored by the probe-charge correction
        -2 MADELUNG / L on the filled bands (L the side of the cubic cell).
        The local-density run supplies the starting orbitals. The exchange
        operator is rebuilt in an outer loop and the Hartree potential converged
        at fixed exchange in an inner one, as in `PlaneWaveCrystal`.

        References: Lin, Journal of Chemical Theory and Computation 12, 2242
        (2016), for the compression; Gygi & Baldereschi, Physical Review B 34,
        4405 (1986), for the zero-momentum term."""
        cell, crystal = self.cell, self.crystal
        occupied = crystal.electrons // 2
        side = float(np.linalg.norm(crystal.lattice[0]))
        if not np.allclose(crystal.lattice, side * np.eye(3)):
            raise NotImplementedError("the probe-charge correction is implemented for a cubic cell")
        madelung = 2.0 * coulomb.MADELUNG_SC / side
        integrals = coulomb.TripleIntegrals(cell.complex, cell.squared_lengths)
        weights = self.kernel.weights
        start = self.run(bands, log=log)
        orbitals, values = start["vectors"], start["levels"]
        rank = self.P.shape[1]

        def load_of(vectors):
            filled = vectors[:, :occupied]
            return 2.0 * sum(integrals.loads(filled[:, j], filled[:, j:j + 1])[:, 0] for j in range(occupied))

        nodal = lambda vectors: 2.0 * (vectors[:, :occupied] ** 2).sum(axis=1)
        norm = lambda difference: np.sqrt(weights @ difference ** 2 / crystal.volume) * crystal.volume / crystal.electrons
        history, density = [], nodal(orbitals)
        residual, below = start["residual"], start["shift_below_spectrum"]
        for outer in range(max_outer):
            # The exchange operator on every computed band, compressed: K = -xi xi^T.
            filled = orbitals[:, :occupied]
            W = np.zeros_like(orbitals)
            for j in range(occupied):
                pair_potential = self.kernel.potential(integrals.loads(filled[:, j], orbitals)).real
                W -= integrals.loads(filled[:, j], pair_potential)
            loaded = self.mass @ filled
            W -= madelung * loaded @ (loaded.T @ orbitals)
            overlap = orbitals.T @ W
            xi = W @ np.linalg.inv(np.linalg.cholesky(-0.5 * (overlap + overlap.T))).T
            P = np.hstack([self.P, xi])
            D = np.zeros((P.shape[1], P.shape[1]))
            D[:rank, :rank] = self.D
            D[rank:, rank:] = -np.eye(xi.shape[1])
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
        return {"levels": values, "vectors": orbitals, "residual": residual, "shift_below_spectrum": below,
                "history": history, "converged": history[-1] < tolerance, "spacing": cell.spacing,
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
        side = float(np.linalg.norm(crystal.lattice[0]))
        madelung = 2.0 * coulomb.MADELUNG_SC / side
        integrals = coulomb.TripleIntegrals(cell.complex, cell.squared_lengths)
        filled = mean_field["vectors"][:, :occupied]
        load = 2.0 * sum(integrals.loads(filled[:, j], filled[:, j:j + 1])[:, 0] for j in range(occupied))
        hartree = self.kernel.potential(load).real
        A = (self.stiffness + cell.weighted_mass(self.ionic + hartree).dressed().real).tocsc()
        loaded = self.mass @ filled
        rank = self.P.shape[1]
        orbitals, values = mean_field["vectors"], mean_field["levels"]
        previous = None
        for iteration in range(max_iterations):
            W = -madelung * loaded @ (loaded.T @ orbitals)
            for j in range(occupied):
                W -= integrals.loads(filled[:, j], self.kernel.potential(integrals.loads(filled[:, j], orbitals)).real)
            overlap = orbitals.T @ W
            xi = W @ np.linalg.inv(np.linalg.cholesky(-0.5 * (overlap + overlap.T))).T
            P = np.hstack([self.P, xi])
            D = np.zeros((P.shape[1], P.shape[1]))
            D[:rank, :rank] = self.D
            D[rank:, rank:] = -np.eye(xi.shape[1])
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

    def dipoles(self, extended):
        """r_ia for every particle-hole pair (filled index slow).

        The current vertex is the derivative of the one-particle pencil with
        respect to a uniform change of the link phases, phi_e -> phi_e + k . dx_e:
        it multiplies the entry (v, w) of the stiffness, the weighted mass and
        the mass matrix by i dx_vw, and the projector loads by -i (x - tau). In
        operator terms it is J = [H_0, r] with H_0 the operator without
        exchange, and because a local potential commutes with r,

            [H_0 + v, r] = J        for every local v.

        The exchange operator is nonlocal and does not commute with r, so
        dividing J by Hartree-Fock level differences would underestimate r by
        the ratio of the local to the Hartree-Fock gap. Instead the identity is
        solved in the basis of the computed bands with v chosen to keep H_0 + v
        well gapped (the local-density exchange-correlation potential of the
        Hartree-Fock density; any local v gives the same r in a complete basis):
        H_0 + v is diagonalized in the band basis, r'_pq = J'_pq / (h_p - h_q)
        there between its filled and its empty levels, and r is rotated back.
        Position elements inside the filled or inside the empty manifold are
        left out: between levels that a mesh splits by its own error they are
        arbitrarily large, and they reach a particle-hole pair only through the
        small mismatch of the two filled subspaces."""
        cell = self.cell
        occupied = int(extended["occupied"])
        energies, orbitals = np.asarray(extended["levels"]), np.asarray(extended["vectors"])
        bands = orbitals.shape[1]
        filled = orbitals[:, :occupied]
        # The exchange operator on the bands, as in `extend_bands`, for H_0 = F - K.
        side = float(np.linalg.norm(self.crystal.lattice[0]))
        triple = coulomb.TripleIntegrals(cell.complex, cell.squared_lengths)
        loaded = self.mass @ filled
        W = -coulomb.probe_charge_constant(side, "sc") * loaded @ (loaded.T @ orbitals)
        for j in range(occupied):
            W -= triple.loads(filled[:, j], self.kernel.potential(triple.loads(filled[:, j], orbitals)).real)
        exchange = orbitals.T @ W
        density = 2.0 * (filled ** 2).sum(axis=1)
        local = cell.weighted_mass(lda_potential(density)).dressed().real
        auxiliary = np.diag(energies) - 0.5 * (exchange + exchange.T) + orbitals.T @ (local @ orbitals)
        levels, rotation = np.linalg.eigh(0.5 * (auxiliary + auxiliary.T))
        difference = levels[:, None] - levels[None, :]
        safe = np.abs(difference) > 1e-6 * max(1.0, np.abs(levels).max())
        weighted = cell.weighted_mass(extended["local_potential"])
        out = np.zeros((occupied * (bands - occupied), 3))
        for alpha in range(3):
            def derivative(grid_matrix):
                displacement = (grid_matrix.step * np.array(cell.divisions)) @ (cell.lattice / np.array(cell.divisions)[:, None])
                values = grid_matrix.data.real * displacement[:, alpha]
                return sp.csr_matrix((values, (grid_matrix.row, grid_matrix.col)), shape=grid_matrix.shape)
            dA = derivative(cell.stiffness) + derivative(weighted)          # times i
            dM = derivative(cell.mass)                                      # times i
            # J_mn = z_m^T (dA - e dM) z_n; with the pencil's levels on the right the two
            # orderings differ by the antisymmetry that makes i J Hermitian, so symmetrize.
            first = orbitals.T @ (dA @ orbitals)
            second = orbitals.T @ (dM @ orbitals)
            current = first - 0.5 * (second * energies[None, :] + energies[:, None] * second)
            shifted = np.zeros_like(self.P)
            column = 0
            for pseudo, position in self.crystal.ions:
                offset = self._displacements(position)[:, alpha]
                width = sum(2 * l + 1 for l, _ in pseudo.projectors)
                shifted[:, column:column + width] = offset[:, None] * self.P[:, column:column + width]
                column += width
            overlap_shifted, overlap = orbitals.T @ shifted, orbitals.T @ self.P
            current += -overlap_shifted @ self.D @ overlap.T + overlap @ self.D @ overlap_shifted.T
            current = 0.5 * (current - current.T)                              # the coefficient of i is antisymmetric
            rotated = rotation.T @ current @ rotation
            position_rotated = np.where(safe, rotated / np.where(safe, difference, 1.0), 0.0)
            position_rotated[:occupied, :occupied] = 0.0
            position_rotated[occupied:, occupied:] = 0.0
            position_matrix = rotation @ position_rotated @ rotation.T
            out[:, alpha] = position_matrix[:occupied, occupied:].ravel()
        return out

    def quasiparticle_levels(self, extended, states, log=None):
        """The one-shot GW correction on the Hartree-Fock levels of
        `extend_bands` for the modes listed in `states`, with the screened
        interaction of the direct random-phase approximation in the basis of
        all particle-hole pairs of the computed bands.

        The Coulomb integrals are formed with the Fourier kernel of the grid,
        one Poisson solve per particle-hole pair. That kernel has zero mean,
        which drops the zero-momentum term of the screened interaction exactly
        as it drops that of exchange. The exchange part was restored in the mean
        field by the probe-charge constant c = 2 MADELUNG / L on the filled
        bands. The correlation part is restored here from the response at
        vanishing momentum, which the particle-hole pairs carry through the
        current operator (`dipoles`, `RandomPhase.set_head`); on the energy
        shell it is +c (1 - 1/eps) / 2 on a filled level and -c (1 - 1/eps) / 2
        on an empty one, with eps the macroscopic dielectric constant, which is
        returned.

        Returns a dict: per state the Hartree-Fock level, the quasiparticle
        level without (`body`) and with (`quasiparticle`) the zero-momentum
        term, the renormalization factor and the residual of the quasiparticle
        equation, all in rydberg."""
        from tessera.drivers.bands.screening import RandomPhase
        cell = self.cell
        occupied = extended["occupied"]
        energies, orbitals = extended["levels"], extended["vectors"]
        empties = orbitals[:, occupied:]
        integrals = coulomb.TripleIntegrals(cell.complex, cell.squared_lengths)
        potentials, loads = [], []
        for i in range(occupied):
            pair_loads = integrals.loads(orbitals[:, i], empties)
            loads.append(pair_loads)
            potentials.append(self.kernel.potential(pair_loads).real)
        coupling = np.block([[loads[i].T @ potentials[j] for j in range(occupied)] for i in range(occupied)])
        blocks = {}
        for n in states:
            state_loads = integrals.loads(orbitals[:, n], orbitals)
            blocks[n] = np.hstack([state_loads.T @ potentials[j] for j in range(occupied)])
        if log:
            log(f"  random phase: {coupling.shape[0]} particle-hole pairs")
        rpa = RandomPhase.from_pieces(energies, occupied, coupling, blocks)
        side = float(np.linalg.norm(self.crystal.lattice[0]))
        constant = 2.0 * coulomb.MADELUNG_SC / side
        out = {n: {"mean_field": float(energies[n]), "body": float(rpa.quasiparticle(n)[0])} for n in states}
        dielectric = rpa.set_head(constant, self.dipoles(extended), self.crystal.volume, COULOMB_STRENGTH)
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
        m and every pair, with one Poisson solve per particle-hole pair. Also
        the dipoles of the pairs. The lowest modes of a larger converged set are
        the same Hartree-Fock modes, so a band-count study truncates one set.
        Returns (levels, occupied, coupling, integrals, dipoles)."""
        cell = self.cell
        occupied = int(extended["occupied"])
        bands = len(extended["levels"]) if bands is None else int(bands)
        truncated = {"levels": np.asarray(extended["levels"])[:bands], "vectors": np.asarray(extended["vectors"])[:, :bands],
                     "occupied": occupied, "local_potential": np.asarray(extended["local_potential"])}
        energies, orbitals = truncated["levels"], truncated["vectors"]
        empties = orbitals[:, occupied:]
        triple = coulomb.TripleIntegrals(cell.complex, cell.squared_lengths)
        potentials, loads = [], []
        for i in range(occupied):
            pair_loads = triple.loads(orbitals[:, i], empties)
            loads.append(pair_loads)
            potentials.append(self.kernel.potential(pair_loads).real)
        coupling = np.block([[loads[i].T @ potentials[j] for j in range(occupied)] for i in range(occupied)])
        stacked = np.hstack(potentials)                                   # vertices x pairs
        integrals = {n: triple.loads(orbitals[:, n], orbitals).T @ stacked for n in range(bands)}
        return energies, occupied, coupling, integrals, self.dipoles(truncated)
