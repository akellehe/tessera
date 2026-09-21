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
        self.kernel = coulomb.CoulombKernel(self.stiffness, self.mass, COULOMB_STRENGTH)
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
