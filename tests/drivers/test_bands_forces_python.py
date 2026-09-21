# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The Born-Oppenheimer energy on the mesh, the forces on the ions and the
zone-centre optical phonon (#1169). Every fixture is synthetic: soft ions with
s, p and d projectors tabulated on a logarithmic radial mesh, as a
pseudopotential file tabulates them."""
import itertools

import numpy as np
import pytest
from scipy.special import erf, erfc

from tessera.drivers.bands import abinitio, forces
from tessera.drivers.bands import pseudopotential as pp
from tessera.drivers.bands.crystal import richardson


def soft_ion(valence=2.0, core=1.1, s=1.5, p=None, d=None, scale=0.7):
    r = np.exp(np.linspace(np.log(1e-6), np.log(12.0), 2400))
    local = -2.0 * valence * erf(r / core) / r
    projectors, strengths = [], []
    for l, strength in ((0, s), (1, p), (2, d)):
        if strength is None:
            continue
        beta = r ** l * np.exp(-(r / scale) ** 2)
        beta /= np.sqrt(np.trapezoid((r * beta) ** 2, r))
        projectors.append((l, r * beta))
        strengths.append(strength)
    density = valence * (1.0 / (np.pi * 1.3 ** 2)) ** 1.5 * np.exp(-(r / 1.3) ** 2)
    return pp.Pseudopotential("X", valence, r, local, projectors, np.diag(strengths), 4.0 * np.pi * r ** 2 * density)


def two_ions(second=(0.07, 0.13, 0.21)):
    """One ion on a vertex of the mesh of eight divisions, with s, p and d
    projectors, and one at `second`, with s and p projectors."""
    first = soft_ion(2.0, 1.1, 1.5, 0.8, 0.5)
    return abinitio.Crystal(7.0 * np.eye(3), [(first, np.full(3, 0.5)), (soft_ion(2.0, 0.9, 1.0, -0.6), np.array(second))])


@pytest.fixture(scope="module")
def converged():
    crystal = two_ions()
    mesh = abinitio.MeshCrystal(crystal, 8)
    run = mesh.run_hartree_fock(6, tolerance=1e-9)
    assert run["converged"]
    return crystal, mesh, run


def moved(mesh, crystal, ion, axis, step):
    displacement = np.zeros((len(crystal.ions), 3))
    displacement[ion, axis] = step
    return forces.with_ions(mesh, forces.displaced(crystal, displacement))


# ---------------------------------------------------------------- the vertex functions

def test_the_radial_slopes_are_the_slopes_of_the_splines():
    ion = soft_ion(2.0, 1.1, 1.5, 0.8, 0.5)
    radius = np.array([3e-3, 0.11, 0.7, 1.9, 4.2])
    step = 1e-7
    difference = lambda f: (f(radius + step) - f(radius - step)) / (2.0 * step)
    assert ion.short_range_slope(radius, 1.2) == pytest.approx(difference(lambda r: ion.short_range_at(r, 1.2)),
                                                                       rel=1e-6, abs=1e-8)
    for index in range(3):
        assert ion.projector_slope(index, radius) == pytest.approx(
            difference(lambda r: ion.projector_at(index, r)), rel=1e-6, abs=1e-8)
    # The series of d/dr erf(a r) / r below a r = 1e-2 meets the closed form there.
    width = 1.2
    edge = 1e-2 * np.sqrt(2.0) * width
    below, above = (ion.short_range_slope(np.array([edge * (1.0 + sign * 1e-9)]), width)[0] for sign in (-1, 1))
    assert below == pytest.approx(above, rel=1e-8)


def test_the_solid_harmonics_are_the_real_harmonics_and_their_gradients():
    rng = np.random.default_rng(7)
    vectors = rng.normal(size=(12, 3))
    radius = np.linalg.norm(vectors, axis=1)
    for l in (0, 1, 2):
        values, gradients = forces.solid_harmonics(l, vectors)
        for value, harmonic in zip(values, pp.real_harmonics(l, vectors)):
            assert value == pytest.approx(radius ** l * harmonic, abs=1e-12)
        for m in range(2 * l + 1):
            for axis in range(3):
                shift = np.zeros(3)
                shift[axis] = 1e-6
                numeric = (forces.solid_harmonics(l, vectors + shift)[0][m]
                           - forces.solid_harmonics(l, vectors - shift)[0][m]) / 2e-6
                assert gradients[m][:, axis] == pytest.approx(numeric, abs=1e-8)


def test_the_gradient_of_a_projector_at_its_own_ion():
    """At the ion only l = 1 has a gradient: the limit of beta / r times the
    gradient of the solid harmonic."""
    ion = soft_ion(2.0, 1.1, 1.5, 0.8, 0.5)
    small = 1e-3
    for index, (l, _) in enumerate(ion.projectors):
        at_ion = forces.projector_gradients(ion, index, np.zeros((1, 3)))
        for m in range(2 * l + 1):
            for axis in range(3):
                shift = np.zeros((1, 3))
                shift[0, axis] = small
                value = lambda offset: ion.projector_at(index, np.linalg.norm(offset, axis=1)) * pp.real_harmonics(l, offset)[m]
                numeric = (value(shift) - value(-shift))[0] / (2.0 * small)
                assert at_ion[m][0, axis] == pytest.approx(numeric, rel=1e-4, abs=1e-6)


# ---------------------------------------------------------------- the ions among themselves

def ewald(lattice, positions, charges, alpha, reach=6):
    """The energy (Ry, e^2 = 2) of point charges in a uniform neutralizing
    background, by the Ewald sum: the independent reference of the test."""
    lattice = np.asarray(lattice, dtype=float)
    volume = abs(np.linalg.det(lattice))
    reciprocal = 2.0 * np.pi * np.linalg.inv(lattice).T
    cartesian = np.asarray(positions) @ lattice
    images = np.array(list(itertools.product(range(-reach, reach + 1), repeat=3)))
    energy = 0.0
    for a, b in itertools.product(range(len(charges)), repeat=2):
        separation = cartesian[a] - cartesian[b] - images @ lattice
        r = np.linalg.norm(separation, axis=1)
        r = r[r > 1e-12]
        energy += charges[a] * charges[b] * np.sum(erfc(alpha * r) / r)
    G = images[np.any(images != 0, axis=1)] @ reciprocal
    G2 = (G ** 2).sum(axis=1)
    structure = np.abs(np.exp(1j * G @ cartesian.T) @ np.asarray(charges)) ** 2
    energy += (4.0 * np.pi / volume) * np.sum(np.exp(-G2 / (4.0 * alpha ** 2)) / G2 * structure)
    total = float(np.sum(charges))
    energy -= 2.0 * alpha / np.sqrt(np.pi) * float(np.sum(np.square(charges))) + np.pi * total ** 2 / (alpha ** 2 * volume)
    return energy


def test_the_energy_of_the_ions_tends_to_the_ewald_sum():
    """Second order in the mesh spacing, with both ions on vertices of every
    mesh so that the error is a series in the spacing. The uniform term of the Ewald sum
    with Gaussians of the mesh's width is left out by the kernel, which acts
    on the complement of the constants."""
    crystal = two_ions((0.25, 0.25, 0.5))
    positions = [position for _, position in crystal.ions]
    reference = ewald(crystal.lattice, positions, [2.0, 2.0], alpha=0.45)
    assert reference == pytest.approx(ewald(crystal.lattice, positions, [2.0, 2.0], alpha=0.7), abs=1e-9)
    spacings, values = [], []
    for n in (8, 12, 16):
        mesh = abinitio.MeshCrystal(crystal, n, approximations=abinitio_settings(refinement_terms=1))
        spacings.append(mesh.cell.spacing)
        values.append(forces.LatticeEnergy(mesh).ion_ion())
    target = reference + 4.0 * np.pi * 1.2 ** 2 * 4.0 ** 2 / crystal.volume
    errors = np.abs(np.array(values) - target)
    assert errors[0] > 1e-3 and errors[2] < 0.3 * errors[0]
    assert richardson(spacings, values)[0] == pytest.approx(target, abs=5e-4)


def abinitio_settings(**options):
    from tessera.drivers.bands.settings import Approximations
    return Approximations(**options)


# ---------------------------------------------------------------- the energy and the forces

def test_the_energy_functional_is_the_energy_of_a_converged_run(converged):
    _, mesh, run = converged
    energy = forces.LatticeEnergy(mesh)
    assert energy.electronic(run) == pytest.approx(run["energy"], abs=1e-8)
    # The Gaussian charges are those the ionic potential was built from: what is left is the short-range sum.
    assert mesh.kernel.weights @ energy.charges.T == pytest.approx([2.0, 2.0], abs=1e-12)
    short_range = np.zeros(mesh.cell.size)
    for pseudo, position in mesh.crystal.ions:
        for image in itertools.product((-2, -1, 0, 1, 2), repeat=3):
            radius = np.linalg.norm((mesh.cell.fractional - position - np.asarray(image)) @ mesh.cell.lattice, axis=1)
            short_range += np.where(radius < 6.0 * mesh.width + 2.0, pseudo.short_range_at(radius, mesh.width), 0.0)
    assert np.abs(mesh.ionic - energy.long_range - short_range).max() < 1e-12


def test_the_force_is_the_derivative_of_the_energy_at_fixed_orbitals(converged):
    """The explicit derivative, term by term exact: the ion on a vertex and the
    ion inside a mesh cell, against central differences small enough that no
    vertex crosses a knot of a radial table."""
    crystal, mesh, run = converged
    analytic = forces.LatticeEnergy(mesh).forces(run)
    step = 1e-5
    for ion, axis in itertools.product(range(2), range(3)):
        plus, minus = (forces.LatticeEnergy(moved(mesh, crystal, ion, axis, sign * step)).total(run) for sign in (1, -1))
        assert analytic[ion, axis] == pytest.approx(-(plus - minus) / (2.0 * step), abs=5e-7)
    assert np.abs(analytic).max() > 0.05


@pytest.mark.slow
def test_the_force_is_the_derivative_of_the_self_consistent_energy(converged):
    """Hellmann and Feynman: with Hartree-Fock converged again at the displaced
    ions the difference quotient is that at fixed orbitals."""
    crystal, mesh, run = converged
    analytic = forces.LatticeEnergy(mesh).forces(run)
    step = 1e-4
    for ion, axis in ((1, 0),):
        energies = []
        for sign in (1, -1):
            displaced_mesh = moved(mesh, crystal, ion, axis, sign * step)
            state = displaced_mesh.run_hartree_fock(6, tolerance=1e-9, start=run)
            assert state["converged"]
            energies.append(forces.LatticeEnergy(displaced_mesh).total(state))
        assert analytic[ion, axis] == pytest.approx(-(energies[0] - energies[1]) / (2.0 * step), abs=2e-5)


# ---------------------------------------------------------------- the optical phonon

def test_frequency_units():
    # 1 Ry / bohr^2 on one atomic mass unit.
    assert forces.frequency_thz(1.0, 1.0) == pytest.approx(108.97, rel=1e-3)
    assert forces.frequency_thz(-1.0, 1.0) == pytest.approx(-108.97, rel=1e-3)


@pytest.mark.slow
def test_the_restoring_force_of_the_optical_mode():
    """Two sublattices of one ion each, both on vertices (the caesium chloride
    arrangement). The interpolated force integrates to the energy differences
    and its slope is the second difference of the energy."""
    first, second = soft_ion(2.0, 1.1, 1.5, 0.8), soft_ion(2.0, 0.9, 1.0, -0.6)
    crystal = abinitio.Crystal(7.0 * np.eye(3), [(first, np.zeros(3)), (second, np.full(3, 0.5))])
    mesh = abinitio.MeshCrystal(crystal, 8)
    run = mesh.run_hartree_fock(6, tolerance=1e-9)
    assert run["converged"]
    with pytest.raises(ValueError, match="five"):
        forces.frozen_phonon(mesh, run, ([0], [1]), (20.0, 30.0), (1, 0, 0), count=3)
    read = forces.optical_phonon(mesh, run, ([0], [1]), (20.0, 30.0), directions=[(1.0, 0.0, 0.0)],
                                 amplitude=0.04, count=5, tolerance=1e-9)
    sample = read["reads"][0]
    assert sample["energy_defect"] < 1e-7
    u, energy = sample["displacements"], sample["energies"]
    # The second difference of fourth order, on the same five samples.
    second_difference = (-energy[4] + 16.0 * energy[3] - 30.0 * energy[2] + 16.0 * energy[1] - energy[0]) \
        / (12.0 * (u[3] - u[2]) ** 2)
    slope = -np.polyval(np.polyder(sample["polynomials"][0]), 0.0)
    assert slope == pytest.approx(second_difference, rel=5e-3)
    assert read["along"][0] == pytest.approx(forces.frequency_thz(slope, 12.0), rel=1e-12)
    assert read["residual_force"][0] < 1e-8                      # the site symmetry survives on the mesh


def test_the_phonon_run_is_reached_from_the_command_line(capsys):
    from tessera.drivers.bands import gaas
    with pytest.raises(SystemExit) as stop:
        gaas.main(["phonon", "--help"])
    assert stop.value.code == 0
    text = capsys.readouterr().out
    assert "--displacements" in text and "--zero-momentum-order" in text and "Born effective charge" in text


# ---------------------------------------------------------------- the polarization and the longitudinal mode

def test_the_polarization_phase_follows_an_orbital():
    """One filled orbital, a Gaussian: the phase along each reciprocal vector
    is minus 2 pi times the fractional position of its centre, and moving the
    centre winds the phase by the displacement."""
    crystal = abinitio.Crystal(7.0 * np.eye(3), [(soft_ion(), np.full(3, 0.5))])
    mesh = abinitio.MeshCrystal(crystal, 12, approximations=abinitio_settings(refinement_terms=1))
    energy = forces.LatticeEnergy(mesh)

    def phases(centre):
        offset = mesh.cell.fractional - centre
        offset -= np.rint(offset)
        orbital = np.exp(-0.5 * ((offset @ mesh.cell.lattice) ** 2).sum(axis=1) / 0.8 ** 2)
        return energy.polarization_phases({"vectors": (orbital / np.sqrt(orbital @ (mesh.mass @ orbital)))[:, None]})

    centre = np.array([0.31, 0.47, 0.12])
    assert phases(centre) == pytest.approx(-2.0 * np.pi * centre, abs=5e-3)
    winding = phases(centre + np.array([0.01, 0.0, 0.0])) - phases(centre)
    assert winding == pytest.approx([-2.0 * np.pi * 0.01, 0.0, 0.0], abs=1e-3)


def test_the_longitudinal_force_constant():
    # Z* = 2.2 and epsilon = 10.9 in the 304.6 cubic bohr of a pair of gallium arsenide: 8.02 THz becomes 8.74 THz.
    extra = forces.longitudinal_force_constant(2.2, 10.9, 304.6)
    assert extra == pytest.approx(8.0 * np.pi * 4.84 / (10.9 * 304.6))
    reduced = 69.723 * 74.921595 / (69.723 + 74.921595)
    transverse = (8.02 / forces.frequency_thz(1.0, reduced)) ** 2
    assert forces.frequency_thz(transverse + extra, reduced) == pytest.approx(8.74, abs=0.01)
