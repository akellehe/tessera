# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Stage 1 of the band-structure drivers (#1159): free electrons on the torus.
The premises of the Hermitian specialization at a crystal momentum, the
entrywise Bloch dressing against the C++ covariant operator, the folded
free-electron dispersion with its zone-boundary degeneracies, the agreement of
the degree-zero spectrum with the exact sector of degree one, gauge covariance,
and Richardson extrapolation in the mesh spacing."""
import numpy as np
import pytest

from tessera import chainhodge as ch
from tessera.drivers.bands.crystal import CrystalCell, GridMatrix, k_path, richardson

KAPPA = (0.3, -0.2, 0.1)
FCC = 0.5 * np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])


def free_levels(cell, kappa, count):
    span = range(-3, 4)
    k = np.array([cell.momentum(np.add(kappa, (i, j, l))) for i in span for j in span for l in span])
    return np.sort(cell.kinetic_scale * (k ** 2).sum(axis=1))[:count]


class TestCertificates:
    @pytest.mark.parametrize("lattice", [np.eye(3), FCC])
    @pytest.mark.parametrize("kappa", [(0.0, 0.0, 0.0), KAPPA, (0.5, 0.0, 0.0)])
    def test_every_premise_holds_and_the_fast_dressing_is_the_covariant_pencil(self, lattice, kappa):
        cell = CrystalCell(lattice, 4, kinetic_scale=1.0, crossover=512)
        certificate = cell.certify(kappa)
        assert certificate.holds(), certificate
        assert certificate.dressing_defect < 1e-14 and certificate.margin == pytest.approx(np.pi)
        read = cell.solve(kappa, 6)
        assert read.certified() and read.residual <= 1e-10

    def test_above_the_crossover_the_regime_is_read_from_the_sparse_defects(self):
        cell = CrystalCell.cubic(1.0, 4, kinetic_scale=1.0, crossover=8)
        certificate = cell.certify(KAPPA)
        assert certificate.holds() and certificate.notes

    @pytest.mark.parametrize("kappa", [(0.0, 0.0, 0.0), KAPPA])
    def test_the_declared_fields_on_a_spacetime_give_the_same_pencil(self, kappa):
        """Squared lengths through Edge.setLength and the momentum through
        Edge.setPhase, read back by WhitneyMass and Connection.fromSpacetime."""
        cell = CrystalCell(FCC, 3, kinetic_scale=2.5)
        A, M = cell.pencil(kappa)
        A_fields, M_fields = cell.pencil_from_spacetime(kappa)
        assert abs(A - A_fields).max() < 1e-13 * abs(A).max()
        assert abs(M - M_fields).max() < 1e-13 * abs(M).max()

    def test_a_matrix_off_the_grid_is_refused(self):
        cell = CrystalCell.cubic(1.0, 5, kinetic_scale=1.0)
        far = np.zeros((cell.size, cell.size))
        far[0, cell.grid.vertexId(2, 0, 0)] = 1.0
        with pytest.raises(ValueError, match="not neighbours"):
            GridMatrix(cell, far)


class TestFreeElectrons:
    def test_dispersion_along_a_path_with_zone_boundary_degeneracies(self):
        cell = CrystalCell.cubic(1.0, 8, kinetic_scale=1.0)
        path = k_path([(0, 0, 0), (0.5, 0, 0), (0.5, 0.5, 0), (0.5, 0.5, 0.5)], segments=2)
        assert len(path) == 7
        for read in cell.bands(path, count=4):
            assert read.certified()
            expected = free_levels(cell, read.kappa, 4)
            scale = max(expected.max(), 1.0)
            assert np.abs(read.energies - expected).max() < 0.12 * scale
        x, m, r = (cell.solve(k, 8).energies for k in ((0.5, 0, 0), (0.5, 0.5, 0), (0.5, 0.5, 0.5)))
        # |k| = |k - G| for 2, 4 and 8 reciprocal vectors at X, M and R. Inversion
        # pairs k with -k exactly; the rest of each star splits at the mesh error.
        assert np.ptp(x[:2]) < 1e-9 * x[0] and np.ptp(m[:2]) < 1e-9 * m[0] and np.ptp(r[:2]) < 1e-9 * r[0]
        assert np.ptp(m[:4]) < 0.08 * m[0] and np.ptp(r[:8]) < 0.15 * r[0]

    def test_fcc_cell_converges_to_its_own_free_levels(self):
        errors = []
        for n in (6, 12):
            cell = CrystalCell(FCC, n, kinetic_scale=1.0)
            read = cell.solve(KAPPA, 3)
            errors.append(np.abs(read.energies / free_levels(cell, KAPPA, 3) - 1.0).max())
        assert errors[1] < 0.3 * errors[0]

    def test_richardson_removes_the_leading_mesh_error(self):
        cells = [CrystalCell.cubic(1.0, n, kinetic_scale=1.0) for n in (6, 8, 12)]
        levels = np.array([cell.solve(KAPPA, 2).energies for cell in cells])
        exact = free_levels(cells[0], KAPPA, 2)
        extrapolated, _ = richardson([cell.spacing for cell in cells], levels)
        assert np.abs(levels[-1] / exact - 1.0).max() > 5e-3
        assert np.abs(extrapolated / exact - 1.0).max() < 2e-4
        with pytest.raises(ValueError):
            richardson([1.0, 0.5], [1.0, 2.0], orders=(2, 4))
        from tessera.drivers.bands.crystal import richardson_amplification
        # What the extrapolation multiplies anything outside its error model by.
        assert richardson_amplification([1 / 16, 1 / 24, 1 / 32]) == pytest.approx(5.6, abs=0.1)
        assert richardson_amplification([1 / n for n in (8, 12, 16, 20, 24, 32)]) == pytest.approx(26.6, abs=0.1)


class TestDegreesAndGauge:
    def test_degree_zero_is_the_exact_sector_of_degree_one(self):
        cell = CrystalCell.cubic(1.0, 3, kinetic_scale=1.0, crossover=512)
        links = cell.grid.blochLinks(cell.edges, list(KAPPA))
        cov = ch.CovariantChainHodge(cell.base, ch.Connection(cell.complex, links), 7, False)
        zero = np.sort(np.array(cov.spectrum(0).eigenvalues).real)
        one = np.sort(np.array(cov.spectrum(1).eigenvalues).real)
        assert zero[0] > 1.0                      # no harmonic function at this momentum
        for level in zero:
            assert np.abs(one - level).min() < 1e-9 * max(level, 1.0)

    def test_the_spectrum_is_gauge_invariant(self):
        cell = CrystalCell.cubic(1.0, 3, kinetic_scale=1.0, crossover=512)
        links = cell.grid.blochLinks(cell.edges, list(KAPPA))
        cov = ch.CovariantChainHodge(cell.base, ch.Connection(cell.complex, links), 7, False)
        rng = np.random.default_rng(3)
        gauge = {v: complex(np.exp(1j * rng.uniform(-np.pi, np.pi))) for v in range(cell.size)}
        moved = cov.gauged(gauge)
        assert not np.allclose(moved.connection().links(), cov.connection().links())
        a = np.sort(np.array(cov.spectrum(0).eigenvalues).real)
        b = np.sort(np.array(moved.spectrum(0).eigenvalues).real)
        assert np.abs(a - b).max() < 1e-10 * a.max()


class TestTracking:
    def test_bands_are_followed_through_a_crossing(self):
        """Free electrons from the zone centre toward X: the branch that starts
        in the first shell with G = (-1, 0, 0) falls while the other five rise,
        and sorted levels would read every crossing as avoided."""
        from tessera.drivers.bands.crystal import track_bands
        cell = CrystalCell.cubic(1.0, 6, kinetic_scale=1.0)
        path = [(0.02 + 0.06 * step, 0.013, 0.007) for step in range(8)]     # generic: no exact degeneracy
        reads = cell.bands(path, count=7)
        tracked, weights = track_bands(cell, reads)
        # Plane waves overlap by exactly 0 or 1. The five lowest branches stay in
        # the window; a rising branch leaves it for a falling one from the next
        # shell, and that hand-over is reported as a lost band, not followed.
        assert np.all(weights[:, :5] > 0.999) and np.any(weights[:, 5:] <= 0.0)
        frac = cell.fractional
        mass = cell.mass.dressed()
        for band in range(5):
            # Each tracked band keeps one reciprocal vector: its weight on that plane wave stays near 1.
            labels = []
            for p, read in enumerate(reads):
                index = int(np.argmin(np.abs(read.energies - tracked[p, band])))
                best = max(((abs(np.vdot(np.exp(2j * np.pi * (frac @ np.array(g, dtype=float))),
                                         mass @ read.vectors[:, index])) ** 2, g)
                            for g in [(0, 0, 0), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]),
                           key=lambda item: item[0])
                labels.append(best[1])
            assert len(set(labels)) == 1
        sorted_levels = np.array([read.energies for read in reads])
        assert not np.allclose(sorted_levels, tracked)             # the ordering did change along the path
