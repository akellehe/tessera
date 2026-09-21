# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The accelerator of the exchange update of Hartree-Fock (#1173): the mean
field of a span against the library's Wick contraction, its first and second
variations against differences of the energy, the minimization, and the loops
of `MeshCrystal` with and without it."""
import argparse

import numpy as np
import pytest
from scipy.linalg import expm

from tessera.drivers.bands import BOHR, abinitio, acceleration, coulomb
from tessera.drivers.bands.settings import Approximations
from tests.drivers.test_bands_abinitio_python import soft_atom


@pytest.fixture(scope="module")
def span():
    """Six modes of a crystal of one soft ion, and the mean field of their span."""
    crystal = abinitio.Crystal(6.0 * np.eye(3), [(soft_atom(), np.array([0.4, 0.45, 0.55]))])
    mesh = abinitio.MeshCrystal(crystal, 6)
    modes = mesh.run(6, max_iterations=1)["vectors"][:, :6]
    accelerator = acceleration.ExchangeAccelerator(mesh, 1, 0)
    accelerator.zone_centre(modes)
    return mesh, modes, accelerator.last[0]


def _moved(field, frame, kappa):
    basis = field._complete(frame)
    n = field.filled
    X = np.zeros((basis.shape[0],) * 2, dtype=kappa.dtype)
    X[n:, :n], X[:n, n:] = kappa, -kappa.conj().T
    return [(basis @ expm(X))[:, :n]]


def test_the_mean_field_of_a_span_is_the_wick_contraction_of_the_library(span):
    """The integrals kept per unordered pair are those of `ModeInteraction`, the
    closed-shell Fock operator and energy are its `fock` and `energy` on two
    sheets filled alike, and the exchange operator and the density load taken
    from the stored pair potentials are those of the mesh."""
    mesh, modes, field = span
    integrals = acceleration.SpanIntegrals(mesh.kernel, mesh.triple, modes)
    T = coulomb.pair_densities(mesh.cell.complex, mesh.cell.squared_lengths, modes)
    library = coulomb.ModeInteraction(mesh.kernel, np.zeros(6), T, sheets=2, zero_momentum=mesh.zero_momentum)
    assert np.abs(library.W - integrals.tensor).max() < 1e-14
    library.h = np.kron(np.eye(2), field.one_particle[0]).astype(complex)
    frame = np.linalg.qr(np.random.default_rng(0).normal(size=(6, 1)))[0]
    P = frame @ frame.T
    gamma = np.kron(np.eye(2), P)
    assert np.abs(library.fock(gamma)[:6, :6] - field.fock([P])[0]).max() < 1e-13
    assert field.energy([frame]) == pytest.approx(library.energy(gamma).real, abs=1e-13)
    rebuilt = coulomb.ModeInteraction.from_integrals(mesh.kernel, field.one_particle[0], integrals.tensor, sheets=2,
                                                     zero_momentum=mesh.zero_momentum)
    assert np.abs(rebuilt.fock(gamma) - library.fock(gamma)).max() < 1e-13
    filled = modes @ frame
    assert np.abs(mesh._exchange(filled, modes) - integrals.exchange(frame, mesh.zero_momentum)).max() < 1e-13
    assert np.abs(2.0 * mesh.triple.loads(filled[:, 0], filled)[:, 0] - integrals.density_load(frame)).max() < 1e-13


def test_the_first_and_second_variations_are_those_of_the_energy(span):
    _, _, field = span
    frame = np.linalg.qr(np.random.default_rng(1).normal(size=(6, 1)))[0]
    units = [unit.reshape(5, 1) for unit in np.eye(5)]
    h = 1e-4
    energy = lambda kappa: field.energy(_moved(field, frame, kappa))
    gradient = np.array([(energy(h * u) - energy(-h * u)) / (2.0 * h) for u in units])
    assert np.abs(gradient - field.gradient([field._complete(frame)[:, :1]])[0].ravel()).max() < 1e-7
    hessian = np.array([[(energy(h * (a + b)) - energy(h * (a - b)) - energy(h * (b - a)) + energy(-h * (a + b)))
                         / (4.0 * h * h) for b in units] for a in units])
    matrix = field.hessian([field._complete(frame)[:, :1]])
    assert np.abs(matrix - matrix.T).max() < 1e-13 and np.abs(hessian - matrix).max() < 1e-6


def test_the_minimization_reaches_a_stationary_pure_state_from_a_state_of_negative_curvature(span):
    """Started at the highest mode of the span, where the second variation is
    negative, the minimization reaches the energy the Roothaan loop of
    `ModeInteraction` reaches, at a pure covariance that commutes with its
    Fock operator and where the second variation is positive."""
    mesh, modes, field = span
    start = [np.eye(6)[:, 5:6]]
    assert field.lowest_curvature(start) < 0.0
    frames, read = field.minimize(start)
    assert read["energy"] < read["energy_before"] and read["gradient"] < 1e-10
    certificate = field.certificate(frames)
    assert certificate["purity_defect"] < 1e-12 and certificate["particles"] == pytest.approx(2.0, abs=1e-12)
    assert certificate["commutator"] < 1e-10 and field.lowest_curvature(frames) > 0.0
    integrals = acceleration.SpanIntegrals(mesh.kernel, mesh.triple, modes)
    library = coulomb.ModeInteraction.from_integrals(mesh.kernel, field.one_particle[0], integrals.tensor, sheets=2,
                                                     zero_momentum=mesh.zero_momentum)
    assert read["energy"] == pytest.approx(library.roothaan(2)[1], abs=1e-10)
    again, second = field.minimize(frames)                              # a stationary frame is returned unchanged
    assert second["steps"] == 0 and np.abs(again[0] - frames[0]).max() == 0.0


def test_the_earlier_frames_join_the_span_orthonormally():
    rng = np.random.default_rng(2)
    mass = np.diag(rng.uniform(0.5, 2.0, size=30))
    orthonormal = lambda block: block @ np.linalg.inv(np.linalg.cholesky(block.T @ mass @ block)).T
    orbitals = orthonormal(rng.normal(size=(30, 4)))
    inside, outside = orbitals[:, :2] @ rng.normal(size=(2, 2)), rng.normal(size=(30, 2))
    extended = acceleration.extended_span(mass, orbitals, [inside, outside])
    assert extended.shape == (30, 6) and np.abs(extended[:, :4] - orbitals).max() == 0.0
    assert np.abs(extended.T @ mass @ extended - np.eye(6)).max() < 1e-12
    assert acceleration.extended_span(mass, orbitals, []) is orbitals


def test_the_accelerated_loop_has_the_fixed_point_of_the_plain_loop():
    """One soft ion off the symmetric site, tight tolerance: the same levels and
    the same energy with and without the accelerator, in fewer exchange updates
    with it; the energy of the state each exchange operator is built from never
    rises, and the state reached is a minimum on its span. The span of the
    computed bands alone (no history) has the same fixed point."""
    crystal = abinitio.Crystal(6.0 * np.eye(3), [(soft_atom(), np.array([0.4, 0.45, 0.55]))])
    mesh = abinitio.MeshCrystal(crystal, 6)
    plain = mesh.run_hartree_fock(6, tolerance=1e-8, accelerate=False)
    accelerated = mesh.run_hartree_fock(6, tolerance=1e-8)
    assert plain["certified"] and accelerated["certified"]
    assert np.abs(plain["levels"] - accelerated["levels"]).max() < 1e-7
    assert accelerated["energy"] == pytest.approx(plain["energy"], abs=1e-8)
    assert accelerated["energies"][-1] == pytest.approx(plain["energy"], abs=1e-8)
    assert 2 * len(accelerated["history"]) <= len(plain["history"])
    assert np.all(np.diff(accelerated["energies"]) < 1e-12) and accelerated["lowest_curvature"] > 0.0
    without_history = abinitio.MeshCrystal(crystal, 6, approximations=Approximations(exchange_history=0))
    shorter = without_history.run_hartree_fock(6, tolerance=1e-8)
    assert shorter["energy"] == pytest.approx(plain["energy"], abs=1e-8)
    assert len(accelerated["history"]) <= len(shorter["history"]) <= len(plain["history"])


def test_a_start_interpolated_between_meshes_that_do_not_nest_is_orthonormalized():
    """From 6 divisions to 8 the prolonged orbitals are interpolants, no longer
    orthonormal in the finer mass matrix; the accelerator orthonormalizes the
    span it is given, and the run reaches the state of a run from scratch."""
    crystal = abinitio.Crystal(6.0 * np.eye(3), [(soft_atom(), np.array([0.4, 0.45, 0.55]))])
    coarse, fine = abinitio.MeshCrystal(crystal, 6), abinitio.MeshCrystal(crystal, 8)
    start = fine.prolonged(coarse, coarse.run_hartree_fock(5))
    gram = start["vectors"].T @ (fine.mass @ start["vectors"])
    assert np.abs(gram - np.eye(5)).max() > 1e-3
    assert np.abs(acceleration.orthonormal(fine.mass, start["vectors"]).T @ fine.mass
                  @ acceleration.orthonormal(fine.mass, start["vectors"]) - np.eye(5)).max() < 1e-12
    scratch, continued = fine.run_hartree_fock(5, tolerance=1e-7), fine.run_hartree_fock(5, tolerance=1e-7, start=start)
    assert continued["certified"] and continued["energy"] == pytest.approx(scratch["energy"], abs=1e-7)
    assert np.all(np.diff(continued["energies"]) < 1e-12)


def test_eight_soft_ions_converge():
    """The zinc-blende crystal of eight soft ions, on which the plain exchange
    update contracts by under five per cent a step and is still at a density
    change of 1e-3 after thirty updates: with the accelerator it converges to
    1e-6."""
    atom = soft_atom()
    crystal = abinitio.Crystal.zinc_blende(5.0 / BOHR, atom, atom, conventional=True)
    mesh = abinitio.MeshCrystal(crystal, 6)
    run = mesh.run_hartree_fock(12, tolerance=1e-6, max_outer=15)
    assert run["converged"] and run["certified"] and len(run["history"]) <= 12
    assert run["energy"] == pytest.approx(-0.612490, abs=2e-6) and run["lowest_curvature"] > 0.0
    assert np.all(np.diff(run["energies"]) < 1e-12)
    plain = mesh.run_hartree_fock(12, tolerance=1e-6, max_outer=30, accelerate=False)
    assert not plain["converged"] and plain["history"][-1] > 5e-4 and plain["energy"] > run["energy"]


def test_the_accelerator_on_a_momentum_set_is_that_of_the_supercell():
    """The span of a momentum set holds the modes of every momentum, and its
    energy is that of one cell: half the energy of the doubled cell the set
    {0, 1/2} is equivalent to. The levels are those of the plain loop."""
    atom = soft_atom()
    single = abinitio.Crystal(6.0 * np.eye(3), [(atom, np.full(3, 0.5))])
    double = abinitio.Crystal(np.diag([12.0, 6.0, 6.0]), [(atom, np.array([0.25, 0.5, 0.5])),
                                                           (atom, np.array([0.75, 0.5, 0.5]))])
    reference = abinitio.MeshCrystal(double, (12, 6, 6)).run_hartree_fock(8, tolerance=1e-8)
    mesh = abinitio.MeshCrystal(single, 6)
    momenta = [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0)]
    run = mesh.run_hartree_fock_set(4, momenta, tolerance=1e-8)
    plain = mesh.run_hartree_fock_set(4, momenta, tolerance=1e-8, accelerate=False)
    assert run["certified"] and plain["certified"] and 2 * len(run["history"]) <= len(plain["history"])
    assert max(np.abs(a - b).max() for a, b in zip(run["levels"], plain["levels"])) < 1e-6
    assert 2.0 * run["energies"][-1] == pytest.approx(reference["energy"], abs=1e-5)
    assert np.all(np.diff(run["energies"]) < 1e-12) and run["lowest_curvature"] > 0.0


def test_the_history_reaches_the_command_line():
    parser = argparse.ArgumentParser()
    Approximations.add_arguments(parser)
    assert Approximations.from_arguments(parser.parse_args([])).exchange_history == Approximations().exchange_history
    assert Approximations.from_arguments(parser.parse_args(["--exchange-history", "0"])).exchange_history == 0
    with pytest.raises(ValueError):
        Approximations(exchange_history=-1)
