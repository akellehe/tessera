# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Stages 5 and 6b of the band-structure drivers (#1159): the gauge response of
the tetrahedron with the exact phase Hessian, the independent-particle
polarizability of the electron gas, and the random-phase and one-shot
quasiparticle machinery against its exact limits."""
import numpy as np
import pytest
import scipy.linalg

from tessera import chainhodge as ch
from tessera import cobordism as cob
from tessera.drivers.bands import coulomb, screening
from tessera.drivers.bands.crystal import CrystalCell


class TestGaugeQuanta:
    @pytest.fixture
    def response(self):
        K = cob.ChainComplex.fromTopCells([[0, 1, 2, 3]])
        # Squared length 8: the exact triplet of the edge operator sits at 5, the
        # coexact one at 10, so the particle-hole energy is 5.
        cov = ch.CovariantChainHodge(ch.ChainHodge(K, [8.0] * 6), ch.Connection.trivial(K))
        edges = K.kSimplexVertices(1)
        gradient = np.array([[1.0 if e[1] == v else -1.0 if e[0] == v else 0.0 for e in edges]
                             for v in range(4)])
        return screening.GaugeResponse(cov, 1, occupied=[0, 1, 2]), gradient

    def test_levels_and_the_ward_identity(self, response):
        gauge, gradient = response
        assert gauge.levels == pytest.approx([5, 5, 5, 10, 10, 10], abs=1e-10)
        induced = gauge.induced_stiffness(0.0)
        assert np.abs(induced.imag).max() < 1e-12 and np.abs(induced - induced.T).max() < 1e-12
        # Every pure-gauge direction costs nothing; the paramagnetic term alone would.
        assert np.abs(induced @ gradient.T).max() < 1e-8
        assert np.abs(gauge.paramagnetic(0.0) @ gradient.T).max() > 1.0
        assert np.abs(gauge.diamagnetic @ gradient.T).max() > 1.0

    def test_induced_stiffness_and_collective_modes_on_the_coexact_triplet(self, response):
        gauge, gradient = response
        coexact = scipy.linalg.null_space(gradient)
        assert coexact.shape == (6, 3)
        d = np.trace(coexact.T @ gauge.diamagnetic.real @ coexact) / 3
        pi = np.trace(coexact.T @ gauge.paramagnetic(0.0).real @ coexact) / 3
        assert d == pytest.approx(21.525, abs=1e-6) and pi == pytest.approx(1.875, abs=1e-6)
        # With a bare plaquette stiffness 4 beta the modes sit just below the
        # particle-hole energy for every beta: omega^2 = gap^2 (1 - pi / (4 beta + d)).
        for beta in (0.25, 1.0, 5.0):
            ratio = np.sqrt(1.0 - pi / (4.0 * beta + d))
            assert 0.95 < ratio < 0.98
        # The physical triplet is split by the base-vertex transport convention.
        stiffness = np.linalg.eigvalsh(coexact.T @ gauge.induced_stiffness(0.0).real @ coexact)
        assert 0.04 < np.ptp(stiffness) / stiffness.mean() < 0.08

    def test_the_hessian_of_the_occupied_energy(self):
        """D - Pi(0) against finite differences of the sum of the occupied levels."""
        K = cob.ChainComplex.fromTopCells([[0, 1, 2, 3]])
        base = ch.ChainHodge(K, [8.0] * 6)

        def occupied_energy(phases):
            links = [complex(np.exp(1j * p)) for p in phases]
            cov = ch.CovariantChainHodge(base, ch.Connection(K, links), 7, False)
            return np.sort(np.array(cov.spectrum(1).eigenvalues).real)[:3].sum()

        gauge = screening.GaugeResponse(ch.CovariantChainHodge(base, ch.Connection.trivial(K)), 1, [0, 1, 2])
        induced = gauge.induced_stiffness(0.0).real
        step = 1e-3
        for a, b in ((0, 0), (1, 4), (2, 5)):
            ea, eb = step * np.eye(6)[a], step * np.eye(6)[b]
            finite = (occupied_energy(ea + eb) - occupied_energy(ea - eb)
                      - occupied_energy(eb - ea) + occupied_energy(-ea - eb)) / (4 * step ** 2)
            assert induced[a, b] == pytest.approx(finite, abs=2e-5)


def _gas_modes(cell, max_squared):
    """Real plane-wave modes (the constant, cosines and sines) up to a shell,
    M-normalized, with their levels: exact eigenvectors of the grid pencil."""
    A, M = cell.pencil()
    columns = [np.ones(cell.size)]
    for n in coulomb.closed_shell_momenta(max_squared):
        if tuple(n) > (0.0, 0.0, 0.0):
            phase = 2 * np.pi * (cell.fractional @ n)
            columns += [np.cos(phase), np.sin(phase)]
    modes = np.array(columns).T
    modes /= np.sqrt(np.einsum("ij,ij->j", modes, M.real @ modes))
    levels = np.einsum("ij,ij->j", modes, A.real @ modes)
    order = np.argsort(levels, kind="stable")
    return levels[order], modes[:, order]


class TestPolarizability:
    def test_static_response_of_the_electron_gas_against_the_exact_sum(self):
        """Seven doubly filled plane waves probed at q = 2 pi (1, 0, 0) / L:
        chi0 = -(4 / V) sum_k 1 / (e(k + q) - e(k)) over the filled k whose
        partner k + q is empty."""
        occupied = coulomb.closed_shell_momenta(1)
        q = np.array([1.0, 0.0, 0.0])
        exact = 0.0
        for k in occupied:
            if ((k + q) ** 2).sum() > 1:
                exact -= 4.0 / ((2 * np.pi) ** 2 * (((k + q) ** 2).sum() - (k ** 2).sum()))
        values, spacings = [], []
        for n in (8, 12, 16):
            cell = CrystalCell.cubic(1.0, n, kinetic_scale=1.0)
            levels, modes = _gas_modes(cell, 4)
            T = coulomb.pair_densities(cell.complex, cell.squared_lengths, modes)
            probe = np.exp(2j * np.pi * (cell.fractional @ q))
            values.append(screening.static_polarizability(levels, T, 7, probe))
            spacings.append(cell.spacing)
        errors = np.abs(np.array(values) / exact - 1.0)
        assert errors[2] < errors[1] < errors[0] and errors[2] < 0.1
        from tessera.drivers.bands.crystal import richardson
        assert richardson(spacings, values)[0] == pytest.approx(exact, rel=0.02)


class TestRandomPhaseAndQuasiparticles:
    @pytest.fixture
    def system(self):
        """Two tetrahedra glued on a face: five modes, two doubly filled, on a
        Hartree-Fock starting point."""
        K = cob.ChainComplex.fromTopCells([[0, 1, 2, 3], [1, 2, 3, 4]])
        rng = np.random.default_rng(0)
        s = list(1.0 + 0.2 * rng.uniform(-1, 1, K.numSimplices(1)))
        pencil = ch.CovariantChainHodge(ch.ChainHodge(K, s), ch.Connection.trivial(K)).pencil(0)
        A, M = pencil.A.real, pencil.B.real
        levels, modes = scipy.linalg.eigh(A, M)

        def build(strength):
            kernel = coulomb.CoulombKernel(A, M, strength=strength)
            both = coulomb.ModeInteraction(kernel, levels, coulomb.pair_densities(K, s, modes), sheets=2)
            gamma, _, _, residual = both.roothaan(4)
            assert residual < 1e-10
            fock_levels, rotation = np.linalg.eigh(both.fock(gamma)[:5, :5].real)
            orbitals = coulomb.ModeInteraction(kernel, fock_levels, coulomb.pair_densities(K, s, modes @ rotation))
            return screening.RandomPhase(fock_levels, orbitals.W, 2)
        return build

    def test_two_routes_to_the_correlation_energy(self, system):
        rpa = system(2.0)
        plasmon = rpa.correlation_energy()
        assert plasmon < 0.0
        assert rpa.correlation_energy_by_frequency(400) == pytest.approx(plasmon, rel=1e-7)

    def test_the_self_energy_reduces_to_second_order_at_weak_coupling(self, system):
        ratios = []
        for strength in (2.0, 0.2, 0.02):
            rpa = system(strength)
            level = rpa.energies[2]
            ratios.append(rpa.correlation(2, level)[0] / rpa.second_order_correlation(2, level))
        assert abs(ratios[2] - 1.0) < 2e-3 and abs(ratios[1] - 1.0) < 2e-2 < abs(ratios[0] - 1.0)

    def test_the_quasiparticle_equation_is_solved(self, system):
        rpa = system(2.0)
        for n in (1, 2):                          # the highest filled and the lowest empty level
            energy, weight = rpa.quasiparticle(n)
            assert energy == pytest.approx(rpa.energies[n] + rpa.correlation(n, energy)[0], abs=1e-9)
            assert 0.9 < weight < 1.0
        # Correlation closes the Hartree-Fock gap... or opens it; here it moves both
        # levels by less than the gap, and the gap stays positive.
        gap = rpa.quasiparticle(2)[0] - rpa.quasiparticle(1)[0]
        assert 0.9 * (rpa.energies[2] - rpa.energies[1]) < gap < 1.1 * (rpa.energies[2] - rpa.energies[1])

    def test_complex_modes_are_refused(self, system):
        rpa = system(2.0)
        with pytest.raises(ValueError, match="real modes"):
            screening.RandomPhase(rpa.energies, rpa.W * (1.0 + 0.5j), 2)


def test_real_modes_of_a_degenerate_shell():
    cell = CrystalCell.cubic(1.0, 4, kinetic_scale=1.0)
    A, M = cell.pencil()
    read = cell.solve((0, 0, 0), 7)
    levels, modes = screening.real_modes(A, M, read.vectors)
    assert np.isrealobj(modes) and levels == pytest.approx(read.energies, abs=1e-8)
    assert np.abs(modes.T @ (M.real @ modes) - np.eye(7)).max() < 1e-10
    assert np.abs(A.real @ modes - (M.real @ modes) * levels).max() < 1e-7
