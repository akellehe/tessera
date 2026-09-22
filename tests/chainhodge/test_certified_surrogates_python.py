# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Certified Schur surrogates of the chain-level pencil (#1192, whitepaper v15
section 4, lines 674-683 and 707-714).

Two things are held here. First, the Feshbach complement at an interior
resonance: the declared supported generalized inverse replaces the inverse only
after the compatibility condition has been measured, the range and null
projectors of the interior block are recorded, the resonant interior modes are
retained as explicit fiber coordinates, and every null vector of the resonant
reduction lifts to a null vector of the pencil itself. Second, the
Craig-Bampton/AMLS surrogate: its spectrum is held to the exact Feshbach map by
an inequality that is measured on every pair it claims, on a complex symmetric
fixture (the trivial connection, where the transpose identities make the pencil
complex symmetric) and on a non-normal one (a connection whose links are not of
unit modulus, where they do not).

Every term here is the one the whitepaper uses. The *interface* B is the set of
retained coordinates and the *interior* I the eliminated ones. P(lambda) =
A - lambda M is the pencil at the shift. F_B(lambda) = P_BB - P_BI P_II^{-1}
P_IB is the Feshbach complement. An *interior resonance* is a shift at which
P_II is singular. The *window* of a surrogate is a closed disc in the complex
spectral plane, because the spectrum of a non-normal or complex symmetric
pencil is complex and an interval on the real axis would not enclose it.
"""
import numpy as np
import pytest

from tessera import chainhodge as ch
from tessera import cobordism as cob
from tests.chainhodge._fixtures import torus33, torus_cells

PS = ch.PencilSchur
KS = ch.Branch.KontsevichSegal


def _edges(K):
    return [tuple(int(v) for v in e) for e in K.kSimplexVertices(1)]


def _split_interface(K, N):
    """Interface = the edges with exactly one endpoint in the first row block,
    the cut that makes the two halves of the torus interact only through B."""
    left = {v for v in range(N * N) if v // N < N // 2}
    return [i for i, e in enumerate(_edges(K)) if (e[0] in left) != (e[1] in left)]


def _complex_symmetric_pencil(N=4, seed=11):
    """The undressed pencil at complex squared lengths: the transpose identities
    hold, so (A~, M) is a complex symmetric pair."""
    rng = np.random.default_rng(seed)
    cells, _ = torus_cells(N)
    K = cob.ChainComplex.fromTopCells(cells)
    s = [complex(1.0 + 0.3 * rng.normal(), 0.3 * rng.normal())
         for _ in range(K.numSimplices(1))]
    base = ch.ChainHodge(K, s, ch.Preset.L2, KS)
    P = base.pencil(1)
    return K, np.array(P.A), np.array(P.B)


def _non_normal_pencil(N=4, seed=23):
    """The pencil dressed by a connection whose links are not of unit modulus:
    the regime certificate calls this one non-normal."""
    rng = np.random.default_rng(seed)
    cells, _ = torus_cells(N)
    K = cob.ChainComplex.fromTopCells(cells)
    n1 = K.numSimplices(1)
    s = [complex(1.0 + 0.3 * rng.normal(), 0.3 * rng.normal()) for _ in range(n1)]
    links = [complex(rng.normal(), rng.normal()) for _ in range(n1)]
    base = ch.ChainHodge(K, s, ch.Preset.L2, KS)
    cov = ch.CovariantChainHodge(base, ch.Connection(K, links))
    P = cov.pencil(1)
    return K, np.array(P.A), np.array(P.B)


def _interior_spectrum(A, M, interface):
    interior = [i for i in range(A.shape[0]) if i not in set(interface)]
    idx = np.ix_(interior, interior)
    return np.linalg.eigvals(np.linalg.solve(M[idx], A[idx])), interior


class TestResonantFeshbach:
    """The interior resonance: the generalized inverse, its projectors, the
    compatibility condition, and the reduction's own certificate."""

    @pytest.mark.parametrize("fixture", ["complex-symmetric", "non-normal"])
    def test_projectors_are_complementary_and_idempotent(self, fixture):
        K, A, M = (_complex_symmetric_pencil() if fixture == "complex-symmetric"
                   else _non_normal_pencil())
        interface = _split_interface(K, 4)
        values, interior = _interior_spectrum(A, M, interface)
        F = PS.feshbach(A, M, complex(values[0]), interface, 1e-8)
        assert F.interiorSingular
        R, Nproj = np.array(F.rangeProjector), np.array(F.nullProjector)
        ni = len(interior)
        assert R.shape == (ni, ni) and Nproj.shape == (ni, ni)
        # Both are projectors, and the null projector's rank is the nullity.
        assert np.abs(R @ R - R).max() < 1e-8
        assert np.abs(Nproj @ Nproj - Nproj).max() < 1e-8
        q = F.interiorNullSpace.shape[1]
        assert q >= 1
        assert abs(np.trace(Nproj).real - q) < 1e-6
        assert abs(np.trace(R).real - (ni - q)) < 1e-6
        assert F.interiorRank == ni - q

    @pytest.mark.parametrize("fixture", ["complex-symmetric", "non-normal"])
    def test_null_spaces_are_the_kernels_they_claim(self, fixture):
        K, A, M = (_complex_symmetric_pencil() if fixture == "complex-symmetric"
                   else _non_normal_pencil())
        interface = _split_interface(K, 4)
        values, interior = _interior_spectrum(A, M, interface)
        lam = complex(values[1])
        F = PS.feshbach(A, M, lam, interface, 1e-8)
        P = A - lam * M
        PII = P[np.ix_(interior, interior)]
        scale = np.abs(PII).max()
        Nv, NL = np.array(F.interiorNullSpace), np.array(F.interiorLeftNullSpace)
        # P_II N = 0 and N_L^T P_II = 0: the right kernel and the left kernel in
        # the transpose pairing the theory uses, never the adjoint.
        assert np.abs(PII @ Nv).max() < 1e-7 * scale
        assert np.abs(NL.T @ PII).max() < 1e-7 * scale

    @pytest.mark.parametrize("fixture", ["complex-symmetric", "non-normal"])
    def test_resonant_reduction_lifts_to_the_pencil(self, fixture):
        """The certificate of the resonant reduction: every null vector of
        Fhat(lambda) lifts to a null vector of P(lambda). The residual is
        measured, not asserted."""
        K, A, M = (_complex_symmetric_pencil() if fixture == "complex-symmetric"
                   else _non_normal_pencil())
        interface = _split_interface(K, 4)
        values, interior = _interior_spectrum(A, M, interface)
        lam = complex(values[0])
        F = PS.feshbach(A, M, lam, interface, 1e-8)
        nb = len(interface)
        q = F.interiorNullSpace.shape[1]
        assert F.resonantResponse.shape == (nb + q, nb + q)
        # Take the reduction's own null space and lift it by hand, so that the
        # certificate is reproduced rather than read back.
        Fhat = np.array(F.resonantResponse)
        u, sv, vh = np.linalg.svd(Fhat)
        nullity = int(np.sum(sv <= 1e-8 * sv[0]))
        P = A - lam * M
        scaleP = np.linalg.norm(P)
        T, Z = np.array(F.constraintModes), np.array(F.resonantModes)
        for j in range(nullity):
            y = vh[Fhat.shape[1] - 1 - j].conj()
            x = T @ y[:nb] + Z @ y[nb:]
            assert np.linalg.norm(P @ x) <= 1e-7 * scaleP * np.linalg.norm(x)
        if nullity > 0:
            assert np.isfinite(F.liftResidual) and F.liftResidual < 1e-7

    @pytest.mark.parametrize("fixture", ["complex-symmetric", "non-normal"])
    def test_compatibility_and_independence_are_measured(self, fixture):
        """The whitepaper's two conditions on the block elimination: solvability
        (y^T P_IB x_B = 0 for every y in ker P_II^T) and independence of the
        chosen interior solution (P_BI ker P_II = 0). Both are reported as
        residuals, and both are reproduced here from the blocks."""
        K, A, M = (_complex_symmetric_pencil() if fixture == "complex-symmetric"
                   else _non_normal_pencil())
        interface = sorted(_split_interface(K, 4))
        values, interior = _interior_spectrum(A, M, interface)
        lam = complex(values[2])
        F = PS.feshbach(A, M, lam, interface, 1e-8)
        P = A - lam * M
        PIB = P[np.ix_(interior, interface)]
        PBI = P[np.ix_(interface, interior)]
        NL = np.array(F.interiorLeftNullSpace)
        Nv = np.array(F.interiorNullSpace)
        assert abs(np.linalg.norm(NL.T @ PIB) / np.linalg.norm(PIB)
                   - F.compatibilityResidual) < 1e-10
        assert abs(np.linalg.norm(PBI @ Nv) / np.linalg.norm(PBI)
                   - F.independenceResidual) < 1e-10
        # The interior solve residual is the compatibility residual: the
        # generalized inverse solves the interior equation exactly on the range
        # of P_II and nowhere else.
        assert abs(F.solveResidual - F.compatibilityResidual) < 1e-10

    def test_away_from_a_resonance_the_projectors_are_trivial(self):
        K, A, M = _complex_symmetric_pencil()
        interface = _split_interface(K, 4)
        F = PS.feshbach(A, M, complex(0.37, -0.11), interface)
        assert not F.interiorSingular
        ni = A.shape[0] - len(interface)
        assert np.abs(np.array(F.rangeProjector) - np.eye(ni)).max() == 0.0
        assert np.abs(np.array(F.nullProjector)).max() == 0.0
        assert F.interiorNullSpace.shape[1] == 0
        assert F.compatible and F.responseIndependent
        assert F.determinantResidual < 1e-8


class TestCongruenceCertificates:
    def test_congruence_of_a_symmetric_pair_is_symmetric(self):
        K, A, M = _complex_symmetric_pencil()
        interface = _split_interface(K, 4)
        F = PS.feshbach(A, M, 0.0 + 0.0j, interface)
        G = PS.craigBampton(A, M, F.constraintModes)
        assert G.symmetryDefect < 1e-10
        assert G.metricSymmetryDefect < 1e-10
        # The constraint modes have an identity block on the interface, so the
        # basis is far from rank deficient.
        assert G.basisConditionInverse > 1e-6


class TestCertifiedSurrogate:
    """The acceptance of #1192: surrogate spectra held to the exact Feshbach map
    within their certified residuals, on a non-normal and a complex symmetric
    fixture."""

    @staticmethod
    def _window(A, M, interface):
        """A disc around the interior spectrum's centre of mass, wide enough to
        hold several levels and narrow enough to discard others."""
        values, _ = _interior_spectrum(A, M, interface)
        centre = complex(np.mean(values))
        radii = np.sort(np.abs(values - centre))
        return centre, float(radii[max(2, len(radii) // 8)])

    @pytest.mark.parametrize("fixture", ["complex-symmetric", "non-normal"])
    def test_surrogate_spectrum_sits_on_the_exact_feshbach_map(self, fixture):
        K, A, M = (_complex_symmetric_pencil() if fixture == "complex-symmetric"
                   else _non_normal_pencil())
        interface = _split_interface(K, 4)
        centre, radius = self._window(A, M, interface)
        surrogate = PS.craigBamptonSurrogate(
            A, M, interface, centre, radius, 4.0 * radius,
            complex(0.0, 0.0), 1e-6, 1e-12)
        assert surrogate.retainedModes > 0
        assert len(surrogate.windowIndices) > 0
        # Every claimed pair's distance from the exact Feshbach map is within
        # the bound its own fine-space residual certifies.
        for i in range(len(surrogate.windowIndices)):
            assert surrogate.feshbachHolds[i]
            if not surrogate.resonantAtEigenvalue[i]:
                assert surrogate.feshbachDefects[i] <= surrogate.feshbachBounds[i] * 1.000001 + 1e-300

    @pytest.mark.parametrize("fixture", ["complex-symmetric", "non-normal"])
    def test_the_defect_is_the_exact_map_recomputed(self, fixture):
        """The reported defect is sigma of F_B(theta) applied to the claimed
        eigenvector's interface part, recomputed here from the dense complement
        so that the certificate is verified rather than trusted."""
        K, A, M = (_complex_symmetric_pencil() if fixture == "complex-symmetric"
                   else _non_normal_pencil())
        interface = sorted(_split_interface(K, 4))
        centre, radius = self._window(A, M, interface)
        surrogate = PS.craigBamptonSurrogate(
            A, M, interface, centre, radius, 4.0 * radius)
        V = np.array(surrogate.basis)
        for position, index in enumerate(surrogate.windowIndices):
            if surrogate.resonantAtEigenvalue[position]:
                continue
            theta = complex(surrogate.eigenvalues[index])
            x = V @ np.array(surrogate.vectors)[:, index]
            xB = x[interface]
            F = PS.feshbach(A, M, theta, interface, 1e-12)
            expected = (np.linalg.norm(np.array(F.response) @ xB)
                        / (np.linalg.norm(F.response) * np.linalg.norm(xB)))
            assert abs(expected - surrogate.feshbachDefects[position]) < 1e-8 * max(expected, 1.0)

    @pytest.mark.parametrize("fixture", ["complex-symmetric", "non-normal"])
    def test_a_wide_enough_retention_radius_certifies(self, fixture):
        """With every fixed-interface mode of the window retained and the
        tolerance declared at the residual scale the surrogate certifies; with a
        retention radius inside the window it refuses by name and still returns
        its numbers."""
        K, A, M = (_complex_symmetric_pencil() if fixture == "complex-symmetric"
                   else _non_normal_pencil())
        interface = _split_interface(K, 4)
        centre, radius = self._window(A, M, interface)
        wide = PS.craigBamptonSurrogate(
            A, M, interface, centre, radius, 6.0 * radius, complex(0.0, 0.0), 1e-4)
        assert wide.discardedModeSeparation > 0.0
        assert wide.certified, wide.refusal
        narrow = PS.craigBamptonSurrogate(
            A, M, interface, centre, radius, 0.5 * radius, complex(0.0, 0.0), 1e-4)
        assert not narrow.certified
        assert "window" in narrow.refusal
        assert narrow.discardedModeSeparation < 0.0
        assert len(narrow.eigenvalues) > 0

    def test_the_reduction_basis_is_the_constraint_modes_and_the_kept_modes(self):
        K, A, M = _complex_symmetric_pencil()
        interface = sorted(_split_interface(K, 4))
        centre, radius = self._window(A, M, interface)
        surrogate = PS.craigBamptonSurrogate(
            A, M, interface, centre, radius, 3.0 * radius)
        nb = len(interface)
        V = np.array(surrogate.basis)
        assert V.shape == (A.shape[0], nb + surrogate.retainedModes)
        F = PS.feshbach(A, M, complex(0.0, 0.0), interface)
        assert np.abs(V[:, :nb] - np.array(F.constraintModes)).max() < 1e-12
        # The retained columns are supported on the interior alone: a
        # fixed-interface mode holds the interface at zero, which is what
        # "fixed-interface" means.
        assert np.abs(V[interface, nb:]).max() < 1e-14

    def test_the_congruence_of_a_complex_symmetric_pair_stays_symmetric(self):
        K, A, M = _complex_symmetric_pencil()
        interface = _split_interface(K, 4)
        centre, radius = self._window(A, M, interface)
        surrogate = PS.craigBamptonSurrogate(
            A, M, interface, centre, radius, 3.0 * radius)
        assert surrogate.reduced.symmetryDefect < 1e-8
        assert surrogate.reduced.metricSymmetryDefect < 1e-8

    def test_a_negative_retention_radius_is_refused(self):
        K, A, M = _complex_symmetric_pencil()
        interface = _split_interface(K, 4)
        with pytest.raises(ValueError):
            PS.craigBamptonSurrogate(A, M, interface, complex(0.0, 0.0), 1.0, 0.5)


class TestRecursiveQuotientSurrogateRegimes:
    """`RecursiveQuotient::craigBampton` no longer refuses the non-normal and
    complex-symmetric-pencil regimes: the regime decides the pairing, not
    whether the surrogate exists."""

    @staticmethod
    def _quotient(A, M, dim):
        components = [list(range(0, dim // 2)), list(range(dim // 2, dim))]
        return cob.RecursiveQuotient.overPencil(
            [complex(z) for z in A.reshape(-1)],
            [complex(z) for z in M.reshape(-1)], dim, components)

    def test_complex_symmetric_pencil_level_builds_a_surrogate(self):
        K, A, M = _complex_symmetric_pencil()
        quotient = self._quotient(A, M, A.shape[0])
        assert quotient.regime == cob.CertificateRegime.ComplexSymmetricPencil
        values = np.linalg.eigvals(np.linalg.solve(M, A))
        upper = float(np.sort(values.real)[len(values) // 4])
        read = quotient.craigBampton(float(values.real.min()) - 1.0, upper,
                                     upper + 10.0, 1e-4)
        assert sum(read.retainedModes) > 0
        assert len(read.windowEigenvalues) > 0
        # The retained pairs are pairs of the fine pencil: their residuals are
        # what the certificate is read from, and they are finite and small.
        assert max(read.eigenResiduals) < 1e-4

    def test_non_normal_pencil_level_builds_a_surrogate(self):
        K, A, M = _non_normal_pencil()
        quotient = self._quotient(A, M, A.shape[0])
        assert quotient.regime == cob.CertificateRegime.NonNormal
        values = np.linalg.eigvals(np.linalg.solve(M, A))
        upper = float(np.sort(values.real)[len(values) // 4])
        read = quotient.craigBampton(float(values.real.min()) - 1.0, upper,
                                     upper + 10.0, 1e-3)
        assert sum(read.retainedModes) > 0
        assert max(read.eigenResiduals) < 1e-3
