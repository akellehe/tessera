# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The Born-Oppenheimer energy of a crystal on the mesh, the forces on its ions
as the exact derivative of that energy, and the optical phonon at the zone
centre from the restoring force of a sublattice displacement. Rydberg atomic
units (see `pseudopotential`); forces in rydberg per bohr.

The energy. The ions of `abinitio.MeshCrystal` enter the pencil through three
vertex functions: the short-range part of the local potential, the Gaussian
charge -Z whose potential through the mesh Coulomb kernel is the long-range
part, and the projector functions of the separable part. The energy of the
ions among themselves is split the same way: the Gaussian charges of all ions
interact through the mesh kernel, which acts on the complement of the constants,
and what a point charge has beyond its Gaussian is taken analytically,

    E_ii = 1/2 Q^T G Q  +  1/2 sum' 2 Z_a Z_b erfc(|R_ab + L| / 2 w) / |R_ab + L|
           - sum_a Z_a^2 / (sqrt(pi) w) ,

with Q the load of the Gaussian charges of width w, G the kernel, the primed
sum over pairs of ions and lattice vectors L without an ion and itself, and the
last term the energy of a Gaussian with itself, which the first term contains.
The electrons and the Gaussians then meet in one quadratic form of the neutral
charge, 1/2 (rho - Q)^T G (rho - Q), so the constant that the kernel leaves out
multiplies the vanishing total charge. In the continuum limit the sum of E_ii
and the electronic energy is the total energy of the pseudopotential method:
the uniform term -4 pi w^2 Z_tot^2 / Omega of the Ewald sum with Gaussians of
this width, which E_ii does not carry, is the term N_e * `alignment` that the
mesh levels do not carry either (see `abinitio`).

The electronic energy is the Hartree-Fock functional of the filled orbitals,

    E_el = 2 sum_i z_i^T A z_i + rho . V_ion + 2 sum_i c_i^T D c_i
           + 1/2 rho^T G rho + sum_i z_i^T (K z_i) ,

rho the load of the density, c_i = P^T z_i the projector overlaps, K the
exchange operator with the zero-momentum term of the kernel. It is stationary
in the orbitals at a converged run, and neither the mass matrix nor the kernel
depends on where the ions are, so the derivative of the total energy with
respect to an ion position is the explicit one (Hellmann and Feynman):

    dE/dR_a = rho . dV_sr,a/dR_a - (V_H + V_lr) . M dq_a/dR_a
              + 4 sum_i (dc_i/dR_a)^T D c_i + dE_pairs/dR_a ,

with q_a the vertex values of the normalized Gaussian of ion a (its
normalization on the mesh moves with the ion and is differentiated too), V_H and
V_lr the potentials of the electrons and of all Gaussians. The vertex functions
are those of `abinitio.MeshCrystal`, the tabulated radial functions carried by
cubic splines in the radius, and are differentiated as they stand: the slope of
the spline, and the gradient of the real solid harmonics. An ion moves relative
to a fixed mesh, so the energy carries the dependence on the position of an ion
within a mesh cell, and the derivative carries it as well; the forces of a
crystal at rest do not sum to zero exactly, and the sum is a measure of that
dependence.

The phonon. At the zone centre the optical mode of a crystal of two sublattices
moves them against each other with the centre of mass at rest. The restoring
force on the relative coordinate u is sampled at five or more displacements,
Hartree-Fock converged again at each, and interpolated by the polynomial through
every sample, anharmonic terms included; the force constant is minus its slope.
The macroscopic field of a longitudinal mode has zero wavevector and is not
carried by a periodic cell, so this frequency is the transverse one. The field
adds 4 pi e^2 Z*^2 / (epsilon Omega) to the force constant, with Z* the Born
effective charge of the mode, the slope of the dipole of a pair of ions in the
displacement. The electronic part of the dipole is the phase of the many-body
expectation of exp(-i b . X) on the filled orbitals, b a reciprocal vector of
the cell (`LatticeEnergy.polarization_phases`); on the mesh the plane wave is
interpolated between the vertices, and the effective charge of a rigid
translation, which vanishes in the continuum, measures what that costs.
"""
import copy
import itertools

import numpy as np
from scipy.special import erfc

from tessera.drivers.bands import coulomb
from tessera.drivers.bands.abinitio import Crystal

RYDBERG_JOULE = 2.1798723611035e-18
BOHR_METRE = 5.29177210903e-11
ATOMIC_MASS_KILOGRAM = 1.66053906660e-27


# ---------------------------------------------------------------- the gradients of the vertex functions

def solid_harmonics(l, vectors):
    """r^l times the real harmonics of `pseudopotential.real_harmonics`, and
    their gradients: lists of arrays of shape (n,) and (n, 3)."""
    x, y, z = vectors.T
    zero, one = np.zeros_like(x), np.ones_like(x)
    if l == 0:
        return [np.full(len(x), 1.0 / np.sqrt(4.0 * np.pi))], [np.zeros((len(x), 3))]
    if l == 1:
        c = np.sqrt(3.0 / (4.0 * np.pi))
        return [c * x, c * y, c * z], [c * np.column_stack(g) for g in ((one, zero, zero), (zero, one, zero),
                                                                        (zero, zero, one))]
    if l == 2:
        a, b = np.sqrt(15.0 / (4.0 * np.pi)), np.sqrt(5.0 / (16.0 * np.pi))
        values = [a * x * y, a * y * z, a * z * x, b * (2.0 * z * z - x * x - y * y), 0.5 * a * (x * x - y * y)]
        gradients = [a * np.column_stack(g) for g in ((y, x, zero), (zero, z, y), (z, zero, x))]
        gradients += [b * np.column_stack((-2.0 * x, -2.0 * y, 4.0 * z)), a * np.column_stack((x, -y, zero))]
        return values, gradients
    raise NotImplementedError("solid harmonics are implemented up to l = 2")


def projector_gradients(pseudo, index, offset):
    """The gradient with respect to `offset` (rows, vertex minus ion) of
    beta(r) Y_lm, one array of shape (n, 3) per m: the function is
    (beta / r^l) times a solid harmonic, and beta / r^l is a constant inside
    the first interval of the radial table (`Pseudopotential.projector_at`), so
    at the ion itself only l = 1 has a gradient."""
    l, _ = pseudo.projectors[index]
    radius = np.linalg.norm(offset, axis=1)
    inside = radius < pseudo.r[1]
    safe = np.where(inside, 1.0, radius)
    value, slope = pseudo.projector_at(index, radius), pseudo.projector_slope(index, radius)
    scaled = np.where(inside, pseudo.projector_origin(index), value / safe ** l)
    scaled_slope = np.where(inside, 0.0, slope / safe ** l - l * value / safe ** (l + 1))
    solids, gradients = solid_harmonics(l, offset)
    unit = offset / safe[:, None]
    return [(scaled_slope * solid)[:, None] * unit + scaled[:, None] * gradient
            for solid, gradient in zip(solids, gradients)]


# ---------------------------------------------------------------- the energy and the forces

class LatticeEnergy:
    """The total energy of `mesh` (an `abinitio.MeshCrystal`) and the forces on
    its ions, for a converged Hartree-Fock run of that mesh."""

    def __init__(self, mesh):
        self.mesh = mesh
        cell, crystal, width = mesh.cell, mesh.crystal, mesh.width
        weights = mesh.kernel.weights
        images = [np.asarray(image) for image in itertools.product(mesh.approximations.images, repeat=3)]
        # The Gaussian charge of every ion as `MeshCrystal._ionic_potential` builds it, and its gradient.
        self.charges = np.zeros((len(crystal.ions), cell.size))
        self.charge_gradients = np.zeros((len(crystal.ions), 3, cell.size))
        self.short_range_gradients = np.zeros((len(crystal.ions), 3, cell.size))
        for a, (pseudo, position) in enumerate(crystal.ions):
            gaussian, gradient = np.zeros(cell.size), np.zeros((3, cell.size))
            for image in images:
                offset = (cell.fractional - position - image) @ cell.lattice
                radius = np.linalg.norm(offset, axis=1)
                term = np.exp(-0.5 * radius ** 2 / width ** 2)
                gaussian += term
                gradient += (offset / width ** 2).T * term               # d/dR of exp(-|r - R|^2 / 2 w^2)
                near = (radius < 6.0 * width + 2.0) & (radius > 1e-12)
                slope = pseudo.short_range_slope(radius[near], width)
                self.short_range_gradients[a][:, near] -= (slope / radius[near]) * offset[near].T
            norm = weights @ gaussian
            self.charges[a] = pseudo.valence * gaussian / norm
            self.charge_gradients[a] = pseudo.valence * (gradient / norm - np.outer(gradient @ weights, gaussian) / norm ** 2)
        self.load = mesh.mass @ self.charges.sum(axis=0)
        self.long_range = -mesh.kernel.potential(self.load).real         # of all Gaussians, as the electrons see it

    # -- the ions among themselves

    def _pairs(self):
        """(a, b, R_a - R_b - L) for every pair of ions and lattice vector without an ion and itself."""
        crystal, width = self.mesh.crystal, self.mesh.width
        heights = 1.0 / np.linalg.norm(np.linalg.inv(crystal.lattice), axis=0)      # between opposite faces
        reach = max(max(self.mesh.approximations.images), int(np.ceil(12.0 * width / heights.min())))
        for image in itertools.product(range(-reach, reach + 1), repeat=3):
            for a, (_, first) in enumerate(crystal.ions):
                for b, (_, second) in enumerate(crystal.ions):
                    if a != b or any(image):
                        yield a, b, (np.asarray(first) - np.asarray(second) - np.asarray(image)) @ crystal.lattice

    def ion_ion(self):
        crystal, width = self.mesh.crystal, self.mesh.width
        valence = [pseudo.valence for pseudo, _ in crystal.ions]
        energy = -0.5 * float(self.load @ self.long_range) - sum(z ** 2 for z in valence) / (np.sqrt(np.pi) * width)
        for a, b, separation in self._pairs():
            r = np.linalg.norm(separation)
            energy += valence[a] * valence[b] * erfc(0.5 * r / width) / r
        return energy

    def _pair_gradients(self):
        crystal, width = self.mesh.crystal, self.mesh.width
        valence = [pseudo.valence for pseudo, _ in crystal.ions]
        gradient = np.zeros((len(crystal.ions), 3))
        for a, b, separation in self._pairs():
            if a == b:
                continue
            r = np.linalg.norm(separation)
            slope = -erfc(0.5 * r / width) / r ** 2 - np.exp(-0.25 * r ** 2 / width ** 2) / (np.sqrt(np.pi) * width * r)
            gradient[a] += 2.0 * valence[a] * valence[b] * slope * separation / r      # the pair (b, a) supplies the half
        return gradient

    # -- the electrons

    def _filled(self, run):
        return np.asarray(run["vectors"])[:, :self.mesh.crystal.electrons // 2].real

    def density_load(self, filled):
        cell = self.mesh.cell
        return 2.0 * sum(coulomb.pair_loads(cell, filled[:, j], filled[:, j:j + 1])[:, 0] for j in range(filled.shape[1]))

    def electronic(self, run):
        """The Hartree-Fock functional of the filled orbitals of `run`. At a
        converged run it is the "energy" of `MeshCrystal.run_hartree_fock`, and
        unlike the sum of levels its error is of second order in that of the
        orbitals."""
        mesh, filled = self.mesh, self._filled(run)
        rho = self.density_load(filled)
        overlap = filled.T @ mesh.P
        one_particle = 2.0 * np.einsum("vi,vi->", filled, mesh.stiffness @ filled) + rho @ mesh.ionic \
            + 2.0 * np.einsum("ip,pq,iq->", overlap, mesh.D, overlap)
        hartree = 0.5 * float(rho @ mesh.kernel.potential(rho).real)
        exchange = float(np.einsum("vi,vi->", filled, mesh._exchange(filled, filled)))
        return float(one_particle + hartree + exchange)

    def total(self, run):
        return self.electronic(run) + self.ion_ion()

    def polarization_phases(self, run):
        """Im ln det S_a for the three reciprocal vectors b_a of the cell, with
        S_ij = sum_c exp(-i b_a . r_c) rho^{ij}_c the component of the pair
        densities of the filled orbitals at b_a: the phase of the many-body
        expectation of exp(-i b_a . X), whose change under a displacement of the
        ions is the change of the electronic polarization of a cell sampled at
        its zone centre (each phase is defined modulo 2 pi, and the dipole of
        the electrons of the cell is 2 sum_a (phase_a / 2 pi) a_a, two per
        orbital with the charge -1).

        References: Resta, Physical Review Letters 80, 1800 (1998);
        King-Smith & Vanderbilt, Physical Review B 47, 1651 (1993)."""
        mesh, filled = self.mesh, self._filled(run)
        pairs = np.stack([coulomb.pair_loads(mesh.cell, filled[:, i], filled) for i in range(filled.shape[1])], axis=1)
        phases = []
        for axis in range(3):
            wave = mesh._shifted(np.ones((mesh.cell.size, 1)), np.eye(3)[axis])[:, 0]
            phases.append(float(np.angle(np.linalg.det(np.einsum("c,cij->ij", wave, pairs)))))
        return np.array(phases)

    def forces(self, run):
        """-dE/dR for every ion (rows), Cartesian, rydberg per bohr."""
        mesh, filled = self.mesh, self._filled(run)
        rho = self.density_load(filled)
        electrostatic = mesh.mass @ (mesh.kernel.potential(rho).real + self.long_range)
        overlap = filled.T @ mesh.P                                     # c_i, (filled, rank)
        gradient = self._pair_gradients()
        for a, (pseudo, position) in enumerate(mesh.crystal.ions):
            gradient[a] += self.short_range_gradients[a] @ rho - self.charge_gradients[a] @ electrostatic
            columns = np.nonzero(mesh._beta_ion == a)[0]
            contracted = overlap @ mesh.D[:, columns]                    # (D c_i) on the projectors of this ion
            # d beta / dR = -grad beta, so dc_i/dR = -(load of grad beta)^T z_i, loaded as the projectors are.
            loaded = self.projector_gradient_loads(a)                    # (vertices, projector, axis)
            for column in range(len(columns)):
                gradient[a] -= 4.0 * (loaded[:, column, :].T @ filled) @ contracted[:, column]
        return -gradient

    def projector_gradient_loads(self, a):
        """The loads of the gradients of the projector functions of ion `a`,
        (vertices, projector, Cartesian axis), by the rule that loads the
        projectors themselves (`MeshCrystal._projectors`): the collapsed Gauss
        rule of `projector_quadrature` points per direction on every
        tetrahedron (`loads.SimplexQuadrature`), or with 0 the mass matrix on
        the vertex values of the gradient. The energy is then differentiated as
        the pencil holds it."""
        mesh = self.mesh
        pseudo, position = mesh.crystal.ions[a]
        points = getattr(mesh.approximations, "projector_quadrature", 0)
        if not pseudo.projectors:
            return np.zeros((mesh.cell.size, 0, 3))
        if not points:
            offset = mesh._displacements(position)
            functions = [g for i in range(len(pseudo.projectors)) for g in projector_gradients(pseudo, i, offset)]
            return np.stack([mesh.mass @ function for function in functions], axis=1)
        from tessera.drivers.bands import loads

        def gradients(offsets):
            functions = [g for i in range(len(pseudo.projectors)) for g in projector_gradients(pseudo, i, offsets)]
            return np.stack(functions, axis=1).reshape(len(offsets), -1)            # [row, (projector, axis)]
        if not hasattr(self, "_rule"):
            self._rule = loads.SimplexQuadrature(mesh.cell, points)
        reach = max(loads.radial_reach(pseudo.r[1:], r_beta[1:] / pseudo.r[1:]) for _, r_beta in pseudo.projectors)
        local = self._rule.loads(gradients, position, reach, mesh.approximations.images)
        return local.assemble().real.reshape(mesh.cell.size, -1, 3)


# ---------------------------------------------------------------- moving the ions

def displaced(crystal, displacements):
    """`crystal` with every ion moved by its row of `displacements` (Cartesian, bohr)."""
    shifts = np.asarray(displacements, dtype=float) @ np.linalg.inv(crystal.lattice)
    return Crystal(crystal.lattice, [(pseudo, np.asarray(position) + shift)
                                     for (pseudo, position), shift in zip(crystal.ions, shifts)])


def with_ions(mesh, crystal):
    """`mesh` with the ions of `crystal` (the same lattice and species): the
    cell, the kernel and its zero-momentum constant do not depend on the ions
    and are shared."""
    if not np.allclose(crystal.lattice, mesh.crystal.lattice) or len(crystal.ions) != len(mesh.crystal.ions):
        raise ValueError("the ions move within the same cell")
    moved = copy.copy(mesh)
    moved.crystal = crystal
    moved.ionic = moved._ionic_potential()
    moved.P, moved.D = moved._projectors()
    return moved


# ---------------------------------------------------------------- the optical phonon at the zone centre

def frequency_thz(force_constant, mass):
    """sqrt(k / m) / 2 pi in terahertz, k in rydberg per square bohr and m in
    atomic mass units; negative for an unstable mode (k < 0)."""
    rate = force_constant * RYDBERG_JOULE / BOHR_METRE ** 2 / (mass * ATOMIC_MASS_KILOGRAM)
    return float(np.sign(rate) * np.sqrt(abs(rate)) / (2.0 * np.pi) * 1e-12)


def frozen_phonon(mesh, run, sublattices, masses, direction, amplitude=0.04, count=5, bands=None, tolerance=1e-7,
                  translation=False, log=None):
    """The restoring force of the zone-centre optical mode along `direction`.

    `sublattices` are the two lists of ion indices and `masses` their atomic
    masses; the relative coordinate u moves the second against the first along
    the unit `direction` with the centre of mass at rest. Hartree-Fock is
    converged again at `count` >= 5 equally spaced values of u in
    [-amplitude, amplitude] (bohr), each from the orbitals of `run`, and the
    generalized force per pair of ions, f = -dE/du (a vector: the change of the
    relative coordinate in every Cartesian direction), is interpolated in u by
    the polynomial through all samples. Returns the samples, the polynomial
    coefficients of every component (highest power first), the energies, and
    the largest difference between the energy differences and the integral of
    the interpolated force along the path. The dipole of a pair (ions and
    electrons, `LatticeEnergy.polarization_phases`) is sampled along with the
    force, and its slope at u = 0 is the effective charge of the mode, the Born
    charge Z* (a vector: the polarization in every direction).

    With `translation` every ion moves by u instead, a displacement that costs
    nothing in the continuum: its force constant measures the dependence of the
    energy on where the ions sit within the mesh, which the optical mode carries
    at the same order."""
    if count < 5:
        raise ValueError("the force is sampled at five displacements or more")
    first, second = (list(s) for s in sublattices)
    if len(first) != len(second):
        raise ValueError("the two sublattices hold the same number of ions")
    direction = np.asarray(direction, dtype=float) / np.linalg.norm(direction)
    total_mass = masses[0] + masses[1]
    weights = np.zeros(len(mesh.crystal.ions))
    weights[first], weights[second] = -masses[1] / total_mass, masses[0] / total_mass
    if translation:
        weights[:] = 1.0
    pattern = np.outer(weights, direction)
    bands = len(run["levels"]) if bands is None else bands
    samples = np.linspace(-amplitude, amplitude, count)
    forces, energies, phases = [], [], []
    for u in samples:
        moved = with_ions(mesh, displaced(mesh.crystal, u * pattern))
        state = run if u == 0.0 else moved.run_hartree_fock(bands, tolerance=tolerance, start=run)
        if not state["converged"]:
            raise ValueError(f"Hartree-Fock did not converge at the displacement {u:+.4f} bohr")
        energy = LatticeEnergy(moved)
        forces.append(weights @ energy.forces(state) / len(first))
        energies.append(energy.total(state) / len(first))
        phases.append(energy.polarization_phases(state))
        if log:
            log(f"  u = {u:+.4f} bohr: force along the mode {forces[-1] @ direction:+.6e} Ry/bohr, "
                f"energy {energies[-1]:.8f} Ry per pair")
    forces, energies = np.array(forces), np.array(energies)
    # The dipole per pair relative to the first sample: the ions, and the electrons through their phases.
    valence = np.array([pseudo.valence for pseudo, _ in mesh.crystal.ions])
    winding = np.angle(np.exp(1j * (np.array(phases) - phases[0])))
    dipole = (np.outer(samples - samples[0], (valence * weights).sum() * direction)
              + 2.0 * (winding / (2.0 * np.pi)) @ mesh.cell.lattice) / len(first)
    born = np.array([np.polyval(np.polyder(np.polyfit(samples, dipole[:, axis], count - 1)), 0.0) for axis in range(3)])
    polynomials = [np.polyfit(samples, forces[:, axis], count - 1) for axis in range(3)]
    along = np.polyfit(samples, forces @ direction, count - 1)
    work = -(np.polyval(np.polyint(along), samples) - np.polyval(np.polyint(along), samples[0]))
    return {"direction": direction, "displacements": samples, "forces": forces, "energies": energies,
            "dipole": dipole, "effective_charge": born, "polynomials": polynomials, "energy_defect": float(np.abs(work - (energies - energies[0])).max())}


def longitudinal_force_constant(effective_charge, dielectric_constant, volume):
    """4 pi e^2 Z*^2 / (epsilon Omega), e^2 = 2: what the macroscopic field of a
    longitudinal optical mode adds to the force constant of the transverse
    one, for the effective charge Z* of the mode, the electronic dielectric
    constant and the volume per pair of ions. The relation is exact for a
    crystal of cubic symmetry.

    Reference: Born & Huang, Dynamical Theory of Crystal Lattices (1954), section 7."""
    return 8.0 * np.pi * effective_charge ** 2 / (dielectric_constant * volume)


def optical_phonon(mesh, run, sublattices, masses, directions=None, dielectric_constant=None, **options):
    """The optical phonon at the zone centre from `frozen_phonon` along each of
    `directions` (the three Cartesian axes by default). With three independent
    directions the force-constant matrix Phi = -df/du at u = 0 is complete and
    its eigenvalues give the three frequencies, which a mesh of lower symmetry
    than the crystal splits; with fewer, the frequency of each direction d is
    that of d . Phi d. The reduced mass carries the motion of the relative
    coordinate. Frequencies in terahertz. With the electronic
    `dielectric_constant` of the crystal the longitudinal frequency along each
    direction follows from the effective charge of the mode along it
    (`longitudinal_force_constant`)."""
    directions = np.eye(3) if directions is None else np.atleast_2d(np.asarray(directions, dtype=float))
    reduced = masses[0] * masses[1] / (masses[0] + masses[1])
    reads = [frozen_phonon(mesh, run, sublattices, masses, direction, **options) for direction in directions]
    # Column by column: -d f / du along each direction at u = 0.
    columns = [np.array([-np.polyval(np.polyder(p), 0.0) for p in read["polynomials"]]) for read in reads]
    result = {"reads": reads, "reduced_mass": reduced,
              "residual_force": [float(np.linalg.norm([np.polyval(p, 0.0) for p in read["polynomials"]]))
                                 for read in reads],
              "along": [frequency_thz(float(read["direction"] @ column), reduced)
                        for read, column in zip(reads, columns)],
              "effective_charge": [float(read["direction"] @ read["effective_charge"]) for read in reads]}
    if dielectric_constant is not None:
        volume = mesh.crystal.volume / len(list(sublattices[0]))
        result["longitudinal_along"] = [
            frequency_thz(float(read["direction"] @ column)
                          + longitudinal_force_constant(charge, dielectric_constant, volume), reduced)
            for read, column, charge in zip(reads, columns, result["effective_charge"])]
    if len(directions) == 3 and abs(np.linalg.det(np.array([read["direction"] for read in reads]))) > 1e-6:
        basis = np.array([read["direction"] for read in reads]).T        # Phi basis = columns
        matrix = np.column_stack(columns) @ np.linalg.inv(basis)
        result["force_constants"] = matrix
        result["asymmetry"] = float(np.abs(matrix - matrix.T).max())
        result["frequencies"] = [frequency_thz(k, reduced) for k in np.linalg.eigvalsh(0.5 * (matrix + matrix.T))]
    return result


# ---------------------------------------------------------------- gallium arsenide

def dielectric_constant(mesh, mean_field, screening_bands, log=None):
    """The electronic dielectric constant at vanishing momentum of the
    Hartree-Fock state, in the random-phase approximation on `screening_bands`
    bands (`RandomPhase.set_head` with the closed-form pair charges of
    `MeshCrystal.vanishing_momentum_pairs`), as the quasiparticle run uses it."""
    from tessera.drivers.bands import screening
    extended = mesh.extend_bands(mean_field, screening_bands + 24, log=log)
    levels, occupied, coupling, integrals = mesh.coulomb_integrals(extended, screening_bands)
    heads = mesh.vanishing_momentum_pairs(extended, coupling, screening_bands)
    rpa = screening.RandomPhase.from_pieces(levels, occupied, coupling, integrals)
    return float(rpa.set_head(mesh.zero_momentum, heads))


def gallium_arsenide_phonon(cation_upf, anion_upf, divisions, bands=24, amplitude=0.04, count=5, directions=None,
                            translation=True, screening_bands=200, approximations=None, a=None, tolerance=1e-7,
                            log=print):
    """The transverse optical phonon of gallium arsenide at the zone centre, with
    the norm-conserving pseudopotentials `cation_upf` and `anion_upf`, on the
    meshes `divisions` of the conventional cell: Hartree-Fock, the forces at
    rest, `optical_phonon`, and with `translation` the force constant and the
    effective charge of the rigid translation along the same directions, both
    of which vanish in the continuum. With `screening_bands` > 0 the electronic
    dielectric constant of the crystal at rest (`dielectric_constant`) gives the
    longitudinal frequency. Values are extrapolated over
    the meshes with `richardson` and reported with `richardson_amplification`
    and the measured frequencies (`reference.GALLIUM_ARSENIDE`)."""
    from tessera.drivers.bands import BOHR
    from tessera.drivers.bands.abinitio import MeshCrystal
    from tessera.drivers.bands.crystal import richardson, richardson_amplification
    from tessera.drivers.bands.pseudopotential import Pseudopotential
    from tessera.drivers.bands.reference import GALLIUM_ARSENIDE
    from tessera.drivers.bands.settings import Approximations
    approximations = Approximations() if approximations is None else approximations
    cation, anion = Pseudopotential.from_upf(cation_upf), Pseudopotential.from_upf(anion_upf)
    lattice_constant = (GALLIUM_ARSENIDE.lattice_constant if a is None else a) / BOHR
    crystal = Crystal.zinc_blende(lattice_constant, cation, anion, conventional=True)
    sublattices, masses = (range(4), range(4, 8)), (GALLIUM_ARSENIDE.cation_mass, GALLIUM_ARSENIDE.anion_mass)
    directions = np.eye(3) if directions is None else np.atleast_2d(np.asarray(directions, dtype=float))
    options = {"amplitude": amplitude, "count": count, "bands": bands, "tolerance": tolerance, "log": log}
    runs, spacings, previous = [], [], None
    for n in sorted(divisions):
        mesh = MeshCrystal(crystal, n, approximations=approximations)
        start = mesh.prolonged(*previous) if previous else None
        mean_field = mesh.run_hartree_fock(bands, tolerance=tolerance, start=start, log=log)
        previous = (mesh, mean_field)
        energy = LatticeEnergy(mesh)
        at_rest = energy.forces(mean_field)
        if not mean_field["converged"]:
            raise ValueError(f"Hartree-Fock did not converge on the mesh of {n} divisions")
        epsilon = dielectric_constant(mesh, mean_field, screening_bands, log) if screening_bands else None
        read = optical_phonon(mesh, mean_field, sublattices, masses, directions, epsilon, **options)
        row = {"divisions": n, "certified": bool(mean_field["certified"]), "energy": energy.total(mean_field),
               "dielectric_constant": epsilon, "effective_charge": read["effective_charge"],
               "longitudinal_along": read.get("longitudinal_along"),
               "largest_force_at_rest": float(np.abs(at_rest).max()), "net_force_at_rest": at_rest.sum(axis=0),
               "along": read["along"], "residual_force": read["residual_force"],
               "energy_defect": [sample["energy_defect"] for sample in read["reads"]]}
        for name in ("force_constants", "asymmetry", "frequencies"):
            if name in read:
                row[name] = read[name]
        if translation:
            moved = [frozen_phonon(mesh, mean_field, sublattices, masses, direction, translation=True, **options)
                     for direction in directions]
            row["translation_force_constant"] = [float(-np.polyval(np.polyder(np.polyfit(
                sample["displacements"], sample["forces"] @ sample["direction"], count - 1)), 0.0)) for sample in moved]
            row["translation_effective_charge"] = [float(sample["direction"] @ sample["effective_charge"])
                                                   for sample in moved]
        log(f"N={n}: transverse optical phonon along the directions {np.round(row['along'], 3)} THz "
            f"(measured {GALLIUM_ARSENIDE.optical_phonon_transverse} THz); effective charge "
            f"{np.round(row['effective_charge'], 3)}"
            + ("" if epsilon is None else f", dielectric constant {epsilon:.3f}, longitudinal "
               f"{np.round(row['longitudinal_along'], 3)} THz (measured "
               f"{GALLIUM_ARSENIDE.optical_phonon_longitudinal} THz)"))
        runs.append(row)
        spacings.append(mesh.cell.spacing)
    result = {"approximations": approximations.record(), "runs": runs, "directions": directions,
              "measured_transverse": GALLIUM_ARSENIDE.optical_phonon_transverse,
              "measured_longitudinal": GALLIUM_ARSENIDE.optical_phonon_longitudinal,
              "certified": all(row["certified"] for row in runs)}
    if len(runs) > 1:
        result["extrapolated"] = [float(v) for v in richardson(spacings, [row["along"] for row in runs])[0]]
        if screening_bands:
            result["extrapolated_longitudinal"] = [
                float(v) for v in richardson(spacings, [row["longitudinal_along"] for row in runs])[0]]
        result["amplification"] = richardson_amplification(spacings)
    return result


def main(argv=None):
    import argparse
    import json
    from tessera.drivers.bands.settings import Approximations
    parser = argparse.ArgumentParser(
        prog="python -m tessera.drivers.bands.gaas phonon",
        description="The transverse optical phonon of GaAs at the zone centre from the restoring force of the "
                    "gallium sublattice against the arsenic one: Hartree-Fock converged at every displacement, the "
                    "forces the exact derivative of the energy on the mesh, against the measured frequency. The ions "
                    "move relative to a fixed mesh. The longitudinal frequency adds the macroscopic field of the "
                    "mode, from its Born effective charge (the change of the polarization phase of the filled "
                    "orbitals with the displacement) and the electronic dielectric constant of the random-phase "
                    "approximation.")
    parser.add_argument("--cation", required=True, help="the gallium pseudopotential (Unified Pseudopotential Format)")
    parser.add_argument("--anion", required=True, help="the arsenic pseudopotential")
    parser.add_argument("--divisions", type=int, nargs="+", default=(8, 12, 16, 20, 24, 32),
                        help="mesh divisions of the conventional cell; multiples of four put every ion on a vertex")
    parser.add_argument("--bands", type=int, default=24, help="bands of the self-consistent loop")
    parser.add_argument("--amplitude", type=float, default=0.04, help="largest relative displacement, bohr")
    parser.add_argument("--displacements", type=int, default=5,
                        help="displacements per direction, five or more; the force is interpolated through all of them")
    parser.add_argument("--direction", type=float, nargs=3, action="append", default=None,
                        help="a direction of displacement (repeatable); the three Cartesian axes by default, which "
                             "give the whole force-constant matrix")
    parser.add_argument("--skip-translation", action="store_true",
                        help="leave out the force constant of the rigid translation, the measure of the dependence "
                             "of the energy on the position of the ions within the mesh")
    parser.add_argument("--screening-bands", type=int, default=200,
                        help="bands of the dielectric constant behind the longitudinal frequency; 0 leaves it out")
    parser.add_argument("--tolerance", type=float, default=1e-7, help="density change at which Hartree-Fock stops")
    parser.add_argument("--lattice-constant", type=float, default=None, help="angstrom; the measured one by default")
    parser.add_argument("--out", default=None, help="write the result as JSON")
    Approximations.add_arguments(parser)
    args = parser.parse_args(argv)
    result = gallium_arsenide_phonon(args.cation, args.anion, args.divisions, args.bands, args.amplitude,
                                     args.displacements, args.direction, not args.skip_translation,
                                     args.screening_bands, Approximations.from_arguments(args), args.lattice_constant, args.tolerance,
                                     log=lambda line: print(line, flush=True))
    if "extrapolated" in result:
        print("extrapolated (THz): transverse", np.round(result["extrapolated"], 3), "longitudinal",
              np.round(result.get("extrapolated_longitudinal", []), 3), "amplification",
              round(result["amplification"], 1), "measured", result["measured_transverse"],
              result["measured_longitudinal"])
    if args.out:
        with open(args.out, "w") as handle:
            json.dump(result, handle, indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else x)
    return result
