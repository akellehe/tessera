# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The sparse Hermitian pencil eigensolver (#1157) against its references: the
dense spectrum below the crossover, the exact plane-wave eigenpairs of the
translation-invariant grid, and the free-particle levels |k + G|^2 it converges
to at second order in the mesh spacing."""
import numpy as np
import pytest
import scipy.sparse as sp

from tessera import chainhodge as ch
from tessera import cobordism as cob
from tests.chainhodge._periodic import covariant_torus, cubic_grid, free_levels

KAPPA = (0.3, -0.2, 0.1)


def _plane_wave_levels(grid, pencil, kappa):
    """The exact spectrum of the grid's pencil. The mesh is invariant under
    grid translations, so every lattice plane wave is an eigenvector of both
    matrices and its eigenvalue is its Rayleigh quotient."""
    n = grid.divisions()
    frac = np.array([grid.fractionalCoordinates(v) for v in range(grid.vertexCount())])
    levels, worst = [], 0.0
    for i in range(n[0]):
        for j in range(n[1]):
            for l in range(n[2]):
                z = np.exp(2j * np.pi * (frac @ np.array([i, j, l], dtype=float)))
                Az, Mz = pencil.A @ z, pencil.M @ z
                lam = (np.vdot(z, Az) / np.vdot(z, Mz)).real
                worst = max(worst, np.linalg.norm(Az - lam * Mz) / np.linalg.norm(Mz))
                levels.append(lam)
    assert worst < 1e-9 * max(levels), "plane waves are exact eigenvectors"
    return np.sort(levels)


class TestAgainstTheDenseReference:
    @pytest.mark.parametrize("kappa,sigma", [((0, 0, 0), -1.0), (KAPPA, 0.0), (KAPPA, -5.0)])
    def test_lowest_pairs_match_the_dense_spectrum(self, kappa, sigma):
        grid = cubic_grid(4)
        _, _, cov = covariant_torus(grid, kappa)
        pencil = cov.sparsePencil()
        dense = cov.pencil(0)
        assert abs(pencil.A.toarray() - dense.A).max() < 1e-13 * abs(dense.A).max()
        assert abs(pencil.M.toarray() - dense.B).max() < 1e-13 * abs(dense.B).max()

        count = 12
        read = ch.SparsePencilSolver.lowest(pencil.A, pencil.M, count, sigma)
        reference = np.sort(np.array(cov.spectrum(0).eigenvalues).real)[:count]
        values = np.array(read.eigenvalues.values)
        assert np.all(values.imag == 0.0)
        assert np.abs(values.real - reference).max() < 1e-10 * reference[-1]

        cert = read.eigenvalues.certificate
        assert read.converged and cert.holds()
        assert cert.grade == cob.CertificateGrade.CertifiedNumerical
        assert cert.regime == cob.CertificateRegime.PositiveSemidefinite
        assert cert.residual <= 1e-10 and max(read.residuals) == cert.residual
        assert 1.0 <= cert.conditioning < 1e8
        assert read.shiftBelowSpectrum
        assert read.hermitianDefectA < 1e-12 and read.hermitianDefectM < 1e-12
        Z = read.vectors
        assert abs(Z.conj().T @ (pencil.M @ Z) - np.eye(count)).max() < 1e-10
        assert read.orthonormalityDefect < 1e-10
        assert np.abs(pencil.A @ Z - (pencil.M @ Z) * values.real).max() < 1e-8

    def test_the_tetrahedron_is_solved_completely(self):
        K = cob.ChainComplex.fromTopCells([[0, 1, 2, 3]])
        hodge = ch.ChainHodge(K, [1.0] * 6)
        cov = ch.CovariantChainHodge(hodge, ch.Connection.trivial(K))
        pencil = cov.sparsePencil()
        read = ch.SparsePencilSolver.lowest(pencil.A, pencil.M, 4, -1.0)
        # The constant and the exact triplet at 40 / a^2.
        assert np.array(read.eigenvalues.values).real == pytest.approx([0, 40, 40, 40], abs=1e-10)
        assert read.eigenvalues.certificate.holds()


class TestOnTheGrid:
    def test_degenerate_levels_are_all_found(self):
        """At k = 0 the first shells have multiplicity 6 and 12, and every copy
        must be returned."""
        grid = cubic_grid(6)
        _, _, cov = covariant_torus(grid, crossover=8)  # the dense path would refuse here
        pencil = cov.sparsePencil()
        with pytest.raises(ValueError, match="crossover"):
            cov.pencil(0)
        exact = _plane_wave_levels(grid, pencil, (0, 0, 0))
        count = 27
        read = ch.SparsePencilSolver.lowest(pencil.A, pencil.M, count, -1.0)
        values = np.array(read.eigenvalues.values).real
        assert read.eigenvalues.certificate.holds()
        assert np.abs(values - exact[:count]).max() < 1e-9 * exact[count - 1]
        # The shells 1, 6, 12, 8 of (2 pi n)^2. The six axis waves are related by
        # symmetries of the mesh and stay exactly degenerate; the higher shells
        # split at the order of the mesh error.
        assert abs(values[0]) < 1e-9
        assert np.ptp(values[1:7]) < 1e-9 * values[1]
        assert values[1] == pytest.approx(free_levels(grid, (0, 0, 0), 2)[1], rel=0.1)

        # A block as wide as the largest shell, with a basis small enough to force
        # thick restarts, finds the same levels.
        narrow = ch.SparsePencilSolver.lowest(pencil.A, pencil.M, count, -1.0, block_size=14,
                                              max_basis_size=56)
        assert narrow.eigenvalues.certificate.holds() and narrow.restarts > 0
        assert np.abs(np.array(narrow.eigenvalues.values).real - exact[:count]).max() < 1e-9 * exact[count - 1]

    def test_free_particle_levels_converge_at_second_order(self):
        sizes = (6, 8, 12)
        errors = []
        for n in sizes:
            grid = cubic_grid(n)
            _, _, cov = covariant_torus(grid, KAPPA, crossover=8)
            pencil = cov.sparsePencil()
            read = ch.SparsePencilSolver.lowest(pencil.A, pencil.M, 4, 0.0)
            assert read.eigenvalues.certificate.holds()
            continuum = free_levels(grid, KAPPA, 4)
            values = np.array(read.eigenvalues.values).real
            errors.append(np.abs(values / continuum - 1.0).max())
        slope = np.polyfit(np.log(1.0 / np.array(sizes)), np.log(errors), 1)[0]
        assert 1.9 <= slope <= 2.1
        assert errors[-1] < 0.05

    def test_a_shift_inside_the_spectrum_returns_the_levels_above_it(self):
        grid = cubic_grid(5)
        _, _, cov = covariant_torus(grid, KAPPA, crossover=8)
        pencil = cov.sparsePencil()
        exact = _plane_wave_levels(grid, pencil, KAPPA)
        sigma = 0.5 * (exact[5] + exact[6])
        read = ch.SparsePencilSolver.lowest(pencil.A, pencil.M, 5, sigma)
        assert not read.shiftBelowSpectrum
        assert read.eigenvalues.certificate.regime == cob.CertificateRegime.HermitianIndefinite
        assert read.eigenvalues.certificate.holds()
        assert np.array(read.eigenvalues.values).real == pytest.approx(exact[6:11], rel=1e-9)


class TestRefusals:
    def _pencil(self):
        _, _, cov = covariant_torus(cubic_grid(3), KAPPA)
        return cov.sparsePencil()

    def test_a_pencil_that_is_not_hermitian_is_refused_by_name(self):
        p = self._pencil()
        skew = sp.csc_matrix(([1.0 + 0j], ([0], [1])), shape=p.A.shape)
        with pytest.raises(ValueError, match="A is not Hermitian"):
            ch.SparsePencilSolver.lowest(p.A + skew, p.M, 3, -1.0)
        with pytest.raises(ValueError, match="M is not Hermitian"):
            ch.SparsePencilSolver.lowest(p.A, p.M + skew, 3, -1.0)

    def test_an_indefinite_mass_matrix_is_refused(self):
        p = self._pencil()
        with pytest.raises(ValueError, match="not positive definite"):
            ch.SparsePencilSolver.lowest(p.A, -p.M, 3, -1.0)

    def test_count_and_shape_are_checked(self):
        p = self._pencil()
        with pytest.raises(ValueError, match="count"):
            ch.SparsePencilSolver.lowest(p.A, p.M, 0, -1.0)
        with pytest.raises(ValueError, match="count"):
            ch.SparsePencilSolver.lowest(p.A, p.M, p.A.shape[0] + 1, -1.0)
        with pytest.raises(ValueError, match="square"):
            ch.SparsePencilSolver.lowest(p.A[:, :-1], p.M, 3, -1.0)

    def test_a_shift_on_an_eigenvalue_is_refused(self):
        K = cob.ChainComplex.fromTopCells([[0, 1, 2, 3]])
        cov = ch.CovariantChainHodge(ch.ChainHodge(K, [1.0] * 6), ch.Connection.trivial(K))
        p = cov.sparsePencil()
        # The constant is an eigenvector at zero. A factorization that meets the
        # zero pivot refuses; one that rounds past it reports the conditioning.
        try:
            read = ch.SparsePencilSolver.lowest(p.A, p.M, 2, 0.0)
        except RuntimeError as refusal:
            assert "singular" in str(refusal)
        else:
            assert read.eigenvalues.certificate.conditioning > 1e12


class TestHermitianSpecialization:
    def test_every_measured_premise_holds_at_a_crystal_momentum(self):
        grid = cubic_grid(4)
        _, base, cov = covariant_torus(grid, KAPPA, measure_certificate=True)
        assert base.certificate().allowable and base.certificate().margin == pytest.approx(np.pi)
        assert not base.certificate().continuationAmbiguous
        assert cov.connection().isUnitary()
        regime = cov.regimeCertificate(0)
        assert regime.regime == cob.CertificateRegime.ComplexSymmetricPencil
        P = cov.pencil(0)
        assert np.linalg.norm(P.A - P.A.conj().T) < 1e-12 * np.linalg.norm(P.A)
        assert np.linalg.norm(P.B - P.B.conj().T) < 1e-12 * np.linalg.norm(P.B)
        np.linalg.cholesky(P.B)
        assert cov.certificate().covarianceMetric < 1e-12
