# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Stage 2a of the band-structure drivers (#1159): a static potential as the
weighted mass matrix. The separable cosine potential against the Mathieu
characteristic values, a Gaussian well against the radial equation, a nonlocal
projector as a certified low-rank update, and the plane-wave reference of the
empirical pseudopotential."""
import numpy as np
import pytest
import scipy.linalg

from tessera import cobordism as cob
from tessera.drivers.bands import potentials as pot
from tessera.drivers.bands.crystal import CrystalCell, richardson


class TestMathieu:
    AMPLITUDE = 40.0

    def _edges(self, kappa, antiperiodic):
        axis = pot.mathieu_levels(self.AMPLITUDE, 1.0, 1.0, antiperiodic, 3)
        reference = pot.separable_levels([axis] * 3, 4)
        spacings, levels = [], []
        for n in (8, 12, 16):
            cell = CrystalCell.cubic(1.0, n, kinetic_scale=1.0)
            read = cell.solve(kappa, 4, pot.cosine_potential(cell, self.AMPLITUDE))
            assert read.certified()
            spacings.append(cell.spacing)
            levels.append(read.energies)
        return reference, richardson(spacings, levels)[0], np.array(levels)

    @pytest.mark.parametrize("kappa,antiperiodic", [((0, 0, 0), False), ((0.5, 0.5, 0.5), True)])
    def test_band_edges_are_three_mathieu_values(self, kappa, antiperiodic):
        reference, extrapolated, levels = self._edges(kappa, antiperiodic)
        # The band edge to the plan's 1e-3 after extrapolation, from meshes that
        # are each off by several per cent.
        assert abs(levels[-1][0] / reference[0] - 1.0) > 0.03
        assert abs(extrapolated[0] / reference[0] - 1.0) < 1e-3
        # The next level is a cubic triplet, which the mesh splits into a doublet
        # and a singlet; its mean converges with the edge.
        assert abs(extrapolated[1:4].mean() / reference[1:4].mean() - 1.0) < 6e-3

    def test_the_gap_at_the_zone_edge_opens_with_the_amplitude(self):
        """Nearly free electrons: the first gap at X is |V_G| = amplitude / 2 * 2."""
        cell = CrystalCell.cubic(1.0, 12, kinetic_scale=1.0)
        weak = 0.5
        read = cell.solve((0.5, 0.0, 0.0), 2, pot.cosine_potential(cell, weak))
        assert read.energies[1] - read.energies[0] == pytest.approx(weak, rel=0.03)
        axis = pot.mathieu_levels(weak, 1.0, 1.0, True, 2)
        assert axis[1] - axis[0] == pytest.approx(weak, rel=1e-3)


class TestGaussianWell:
    def test_bound_state_converges_to_the_isolated_well(self):
        depth, width, side = 12.0, 1.2, 9.0
        reference = pot.radial_levels(lambda r: -depth * np.exp(-r * r / (2 * width ** 2)), 1.0)[0]
        spacings, levels = [], []
        for n in (10, 14, 20):
            cell = CrystalCell.cubic(side, n, kinetic_scale=1.0)
            read = cell.solve((0, 0, 0), 1, pot.gaussian_well(cell, depth, width))
            assert read.certified()
            spacings.append(cell.spacing)
            levels.append(read.energies[0])
        assert levels[0] > levels[1] > levels[2] > reference     # piecewise-linear levels lie above
        assert richardson(spacings, levels)[0] == pytest.approx(reference, rel=5e-3)

    def test_radial_solver_on_the_harmonic_oscillator(self):
        levels = pot.radial_levels(lambda r: r * r, 1.0, 0, 2, radius=12.0, points=3000)
        assert levels == pytest.approx([3.0, 7.0], rel=1e-7)
        assert pot.radial_levels(lambda r: r * r, 1.0, 1, 1, radius=12.0, points=3000)[0] == pytest.approx(5.0, rel=1e-6)


class TestNonlocalProjector:
    def test_low_rank_update_gives_the_solves_of_the_explicit_pencil(self):
        """A rank-two projector P D P^dagger added to A - sigma M through the
        Woodbury identity: the premise is verified, the solves agree with the
        dense pencil that contains the projector, and shift-invert iteration on
        them converges to its lowest level."""
        cell = CrystalCell.cubic(6.0, 4, kinetic_scale=1.0)
        A, M = (x.toarray() for x in cell.pencil((0, 0, 0), pot.gaussian_well(cell, 3.0, 1.5)))
        rng = np.random.default_rng(7)
        beta = M @ rng.normal(size=(cell.size, 2))
        D = np.diag([-2.0, 1.5])
        sigma = -6.0
        base, explicit = A - sigma * M, A - sigma * M + beta @ D @ beta.T
        n = cell.size
        update = cob.LowRankUpdate(list(base.ravel()), n)
        update.setUpdate(list((beta @ D).ravel()), list(beta.T.conj().ravel()), 2)
        assert update.updateRank == 2 and update.spansAffectedChange(list(explicit.ravel()))

        vector = M @ rng.normal(size=n)
        for _ in range(40):
            solved = update.solve(list(vector.astype(complex)))
            assert solved.certificate.holds()
            assert solved.certificate.grade == cob.CertificateGrade.StructureExact
            x = np.array(solved.values)
            assert np.abs(x - np.linalg.solve(explicit, vector)).max() < 1e-9 * np.abs(x).max()
            vector = M @ (x / np.sqrt((x.conj() @ M @ x).real))
        rayleigh = (x.conj() @ (A + beta @ D @ beta.T) @ x).real / (x.conj() @ M @ x).real
        reference = scipy.linalg.eigh(A + beta @ D @ beta.T, M, eigvals_only=True)[0]
        assert rayleigh == pytest.approx(reference, rel=1e-8)


class TestEmpiricalPseudopotential:
    def test_plane_wave_bands_of_gallium_arsenide(self):
        epm = pot.ZincBlendeEPM.gallium_arsenide()
        unit = 2 * np.pi / epm.a
        centre = epm.plane_wave_bands((0, 0, 0), 8)
        assert centre[1:4] == pytest.approx(centre[1], abs=1e-9)          # the valence triplet
        assert centre[4] - centre[3] == pytest.approx(1.4186, abs=2e-3)   # the direct gap
        assert centre[3] - centre[0] == pytest.approx(12.249, abs=5e-3)   # the valence width
        x = epm.plane_wave_bands(unit * np.array([1.0, 0, 0]), 5)
        l = epm.plane_wave_bands(unit * np.array([0.5, 0.5, 0.5]), 5)
        assert x[4] > centre[4] and l[4] > centre[4]                       # the gap is direct
        assert np.abs(epm.plane_wave_bands((0, 0, 0), 8, cutoff=90.0) - centre).max() < 1e-3

    def test_the_conventional_cell_folds_the_x_points_onto_the_centre(self):
        epm = pot.ZincBlendeEPM.gallium_arsenide()
        folded = epm.folded_bands((0, 0, 0), 16)
        unit = 2 * np.pi / epm.a
        parts = [epm.plane_wave_bands(q, 4) for q in ((0, 0, 0), (unit, 0, 0), (0, unit, 0), (0, 0, unit))]
        assert folded == pytest.approx(np.sort(np.concatenate(parts)))

    def test_the_potential_is_real_and_has_the_period_of_the_primitive_cell(self):
        epm = pot.ZincBlendeEPM.gallium_arsenide()
        cell = CrystalCell(epm.conventional_lattice(), 8)
        V = epm.potential(cell).reshape(8, 8, 8)
        assert np.abs(V - np.roll(V, (4, 4), axis=(1, 2))).max() < 1e-12   # a (0, 1, 1) / 2
        assert np.abs(V - np.roll(V, (4, 4), axis=(0, 1))).max() < 1e-12
        assert abs(V.mean()) < 1e-12                                        # V(G = 0) = 0
