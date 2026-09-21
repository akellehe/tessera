# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Stage 4 of the band-structure drivers (#1159): the finite-element Coulomb
kernel against the periodic Coulomb potential, Hartree-Fock on the covariance
against exact diagonalization and the Wick engine, and the exchange energy of
the uniform electron gas."""
import itertools

import numpy as np
import pytest
import scipy.linalg
import scipy.sparse as sp

import tessera.quantum as quantum
from tessera import chainhodge as ch
from tessera import cobordism as cob
from tessera.drivers.bands import coulomb
from tessera.drivers.bands.crystal import CrystalCell


def _tetrahedron_modes():
    K = cob.ChainComplex.fromTopCells([[0, 1, 2, 3]])
    s = [1.0, 1.3, 0.9, 1.1, 1.2, 1.05]
    pencil = ch.CovariantChainHodge(ch.ChainHodge(K, s), ch.Connection.trivial(K)).pencil(0)
    A, M = pencil.A.real, pencil.B.real
    levels, modes = scipy.linalg.eigh(A, M)
    return K, s, A, M, levels, modes


class TestKernel:
    def test_gaussian_charge_against_its_fourier_series(self):
        """A0 v = 4 pi rho on the torus with a neutralizing background, against
        v(r) = sum_{G != 0} 4 pi rho(G) e^{i G . r} / (V G^2), at second order."""
        width, side = 0.18, 1.0
        errors = []
        for n in (8, 12, 16):
            cell = CrystalCell.cubic(side, n, kinetic_scale=1.0)
            offset = cell.fractional - 0.5
            offset -= np.rint(offset)
            r2 = (offset ** 2).sum(axis=1) * side ** 2
            density = np.exp(-r2 / (2 * width ** 2)) / (2 * np.pi * width ** 2) ** 1.5
            kernel = coulomb.CoulombKernel.of_cell(cell, strength=4 * np.pi)
            potential = kernel.potential(cell.mass.dressed() @ density).real
            assert abs(kernel.weights @ potential) < 1e-10               # zero mean
            reference = np.zeros(cell.size)
            for g in itertools.product(range(-10, 11), repeat=3):
                g2 = (2 * np.pi / side) ** 2 * sum(x * x for x in g)
                if g2 > 0:
                    phase = np.cos(2 * np.pi * (offset @ np.array(g, dtype=float)))
                    reference += 4 * np.pi * np.exp(-0.5 * width ** 2 * g2) / (side ** 3 * g2) * phase
            errors.append(np.abs(potential - reference).max() / np.abs(reference).max())
        assert errors[2] < 0.03 and errors[1] < 0.7 * errors[0] and errors[2] < 0.7 * errors[1]

    def test_pair_densities_are_the_vertex_density_contraction(self):
        K, s, _, M, _, modes = _tetrahedron_modes()
        T = coulomb.pair_densities(K, s, modes)
        for m, n in ((0, 0), (1, 2), (3, 1)):
            reference = ch.WhitneyMass.vertexDensityContraction(K, s, modes[:, m:m + 1].conj(), modes[:, n:n + 1])
            assert T[:, m, n] == pytest.approx(np.array(reference), abs=1e-15)
        assert np.abs(T.sum(axis=0) - np.eye(4)).max() < 1e-13           # sum_v rho^{mn}_v = <m|n>


class TestHartreeFock:
    def test_tetrahedron_with_two_sheets_against_the_full_fock_space(self):
        K, s, A, M, levels, modes = _tetrahedron_modes()
        kernel = coulomb.CoulombKernel(A, M, strength=3.0)
        interaction = coulomb.ModeInteraction(kernel, levels, coulomb.pair_densities(K, s, modes), sheets=2)
        gamma, energy, _, residual = interaction.roothaan(3)
        assert residual < 1e-11
        state = quantum.CovarianceState.fromBandProjector(gamma)
        assert state.purityDefect() < 1e-10 and state.particleNumber().real == pytest.approx(3.0)

        algebra = quantum.ExteriorAlgebra(8)
        H = interaction.fock_hamiltonian(algebra)
        assert abs(H - H.conj().T).max() < 1e-12
        # The energy of the explicit determinant is the covariance energy.
        _, vectors = np.linalg.eigh(gamma)
        determinant = algebra.wedge([vectors[:, -1 - i] for i in range(3)])
        determinant /= np.linalg.norm(determinant)
        assert (determinant.conj() @ (H @ determinant)).real == pytest.approx(energy, abs=1e-11)
        # The exact ground state of the three-particle sector lies below it.
        rows, _, _, _ = algebra.sectorProjectorCOO(3)
        sector = np.unique(np.array(rows))
        exact = np.linalg.eigvalsh(H[sector][:, sector].toarray())[0]
        assert exact < energy and energy - exact < 0.1

    def test_the_interaction_energy_is_the_quartic_wick_moment(self):
        K, s, A, M, levels, modes = _tetrahedron_modes()
        kernel = coulomb.CoulombKernel(A, M, strength=3.0)
        interaction = coulomb.ModeInteraction(kernel, levels, coulomb.pair_densities(K, s, modes), sheets=2)
        gamma, _, _, _ = interaction.roothaan(3)
        state = quantum.CovarianceState.fromBandProjector(gamma)
        kernel_matrix = np.column_stack([kernel.potential(np.eye(4)[:, v]) for v in range(4)]).real
        total = 0.0
        for v in range(4):
            for w in range(4):
                Tv, Tw = interaction.density_operator(v), interaction.density_operator(w)
                moment = state.wickBilinearMoment([Tv, Tw])
                assert moment.certificate.holds()
                total += 0.5 * kernel_matrix[v, w] * (moment.value - np.trace(Tv @ Tw @ gamma))
        assert total.real == pytest.approx(interaction.interaction_energy(gamma).real, abs=1e-11)
        assert abs(total.imag) < 1e-11

    def test_the_fixed_point_is_stationary_under_mean_field_evolution(self):
        K, s, A, M, levels, modes = _tetrahedron_modes()
        interaction = coulomb.ModeInteraction(coulomb.CoulombKernel(A, M, strength=3.0), levels,
                                              coulomb.pair_densities(K, s, modes), sheets=2)
        gamma, _, _, _ = interaction.roothaan(3)
        state = quantum.CovarianceState.fromBandProjector(gamma)
        steps = state.meanFieldEvolve(lambda g: interaction.fock(g), 0.05, 40)
        assert len(steps) == 40
        assert np.abs(state.gamma() - gamma).max() < 1e-9
        # Away from the fixed point the same evolution moves the state.
        moved = quantum.CovarianceState.fromBandProjector(interaction._filled(interaction.h + 0.3 * np.ones((8, 8)), 3))
        before = moved.gamma().copy()
        moved.meanFieldEvolve(lambda g: interaction.fock(g), 0.05, 40)
        assert np.abs(moved.gamma() - before).max() > 1e-3


class TestElectronGas:
    def test_exchange_integrals_of_plane_waves_converge_to_the_coulomb_kernel(self):
        """(ij|ji) of two plane waves is 4 pi / (V |k_i - k_j|^2)."""
        errors = []
        for n in (6, 12):
            cell = CrystalCell.cubic(1.0, n, kinetic_scale=1.0)
            mass = cell.mass.dressed()
            momenta = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=float)
            modes = np.exp(2j * np.pi * (cell.fractional @ momenta.T))
            modes /= np.sqrt(np.einsum("ij,ij->j", modes.conj(), mass @ modes).real)
            interaction = coulomb.ModeInteraction(coulomb.CoulombKernel.of_cell(cell, strength=4 * np.pi),
                                                  np.zeros(4), coulomb.pair_densities(cell.complex,
                                                                                      cell.squared_lengths, modes))
            worst = 0.0
            for i, j in ((0, 1), (1, 2), (0, 3)):
                q2 = (2 * np.pi) ** 2 * ((momenta[i] - momenta[j]) ** 2).sum()
                worst = max(worst, abs(interaction.W[i, j, j, i] / (4 * np.pi / q2) - 1.0))
            assert abs(interaction.W[0, 0, 0, 0]) < 1e-12        # the background removes G = 0
            errors.append(worst)
        assert errors[1] < 0.35 * errors[0] and errors[1] < 0.12      # second order: a quarter

    def test_the_zero_momentum_correction_restores_the_electron_gas_value(self):
        ratios = {}
        for shell in (3, 9, 12):
            momenta = coulomb.closed_shell_momenta(shell)
            exact = coulomb.electron_gas_exchange(2 * len(momenta), 1.0, strength=4 * np.pi)
            ratios[shell] = (coulomb.plane_wave_exchange(momenta, 1.0, 4 * np.pi, corrected=False) / exact,
                             coulomb.plane_wave_exchange(momenta, 1.0, 4 * np.pi) / exact)
        assert len(coulomb.closed_shell_momenta(12)) == 179
        assert ratios[12][0] < 0.76                                 # a quarter of the exchange is missing
        assert abs(ratios[12][1] - 1.0) < 0.025 and abs(ratios[9][1] - 1.0) < abs(ratios[3][1] - 1.0)
