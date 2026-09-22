# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Certified Schur surrogates of the chain-level pencil (#1192, whitepaper v15
section 4, lines 674-683 and 707-714).

Two things are held here. First, the Feshbach complement at an interior
resonance: the declared supported generalized inverse of the interior block is
the Drazin inverse built on its Riesz projector, the range and null projectors
recorded are the Riesz spectral projectors, the resonant interior modes retained
explicitly are a basis of the generalized eigenspace, the pencil applied to any
retained fiber is read off the reduction alone, and every null vector of the
reduction lifts to a null vector of the pencil. Second, the Craig-Bampton/AMLS
surrogate: its spectrum is held to the exact Feshbach map by an inequality that
is measured on every pair it claims, on a complex symmetric fixture (the trivial
connection, where the transpose identities make the pencil complex symmetric)
and on a non-normal one (a connection whose links are not of unit modulus, where
they do not), and its window is a disc in the complex plane.

Every term here is the one the whitepaper and the code use. The *interface* B
is the set of retained coordinates and the *interior* I the eliminated ones.
P(lambda) = A - lambda M is the pencil at the shift and P_II its interior block.
F_B(lambda) = P_BB - P_BI P_II^{-1} P_IB is the Feshbach complement. An
*interior resonance* is a shift at which P_II has an eigenvalue inside the
*resonance disc*, the closed disc about zero of radius resonance_radius times
the spectral radius of P_II. The *generalized eigenspace* of those eigenvalues
is the span of every eigenvector and every Jordan chain belonging to them; the
*Riesz projector* Pi0 is the spectral projector onto it along the invariant
subspace of the other eigenvalues, (1/2 pi i) times the contour integral of the
resolvent around the disc, which commutes with P_II and is oblique rather than
Hermitian-orthogonal. The *Drazin inverse* at zero is P_II^D = (P_II + Pi0)^{-1}
(I - Pi0): the inverse of P_II on the complementary invariant subspace and zero
on the generalized eigenspace, so P_II P_II^D = P_II^D P_II = I - Pi0. The
*window* of a surrogate is a closed disc |theta - c| <= rho in the complex
spectral plane, because the spectrum of a non-normal or complex symmetric
pencil is complex and an interval on the real axis would not enclose it.
"""
import numpy as np
import pytest

from tessera import chainhodge as ch
from tessera import cobordism as cob
from tests.chainhodge._fixtures import torus_cells

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


#: The resonance disc's radius, relative to the spectral radius of P_II, used
#: wherever a resonance is declared at an eigenvalue computed in floating point:
#: wide enough to enclose the computed scatter of a two-by-two Jordan block,
#: which is of the order of the square root of machine precision.
RADIUS = 1e-6


def _declared_pencil(seed, jordan, symmetric=False):
    """A pencil (A, M) of order 11 whose interior block at the shift lambda is
    P_II = S J S^{-1} with J carrying a double eigenvalue at zero -- a Jordan
    block of size two when `jordan`, two semisimple copies otherwise -- and
    five further eigenvalues well away from it. When `symmetric` the block is
    P_II = W D W^T with W complex orthogonal, so P_II is complex symmetric and
    the resonance is semisimple. Returns (A, M, interface, interior, lambda,
    P_II)."""
    rng = np.random.default_rng(seed)
    nb, ni = 4, 7
    n = nb + ni
    interface = list(range(nb))
    interior = list(range(nb, n))
    lam = complex(0.37, -0.21)
    others = np.array([1.3 + 0.4j, -0.8 + 0.9j, 2.1 - 0.3j, 0.9 + 1.1j, -1.6 - 0.5j])
    if symmetric:
        G = rng.normal(size=(ni, ni)) + 1j * rng.normal(size=(ni, ni))
        W = np.linalg.eig(G + G.T)[1]
        for j in range(ni):  # complex-orthogonalize in the bilinear form
            for k in range(j):
                W[:, j] -= (W[:, k] @ W[:, j]) * W[:, k]
            W[:, j] /= np.sqrt(W[:, j] @ W[:, j])
        PII = W @ np.diag(np.concatenate([[0.0, 0.0], others])) @ W.T
    else:
        J = np.diag(np.concatenate([[0.0, 0.0], others])).astype(complex)
        if jordan:
            J[0, 1] = 1.0
        S = rng.normal(size=(ni, ni)) + 1j * rng.normal(size=(ni, ni))
        PII = S @ J @ np.linalg.inv(S)
    M = rng.normal(size=(n, n)) + 0.2j * rng.normal(size=(n, n))
    M = M + M.T + 5.0 * np.eye(n)
    A = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    A = A + A.T
    if not symmetric:
        M = M + 0.3 * (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))
        A = A + 0.7 * (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))
    idx = np.ix_(interior, interior)
    A[idx] = PII + lam * M[idx]
    return A, M, interface, interior, lam, PII


def _rel(X, Y):
    return np.linalg.norm(X - Y) / max(np.linalg.norm(Y), 1e-300)


class TestResonantFeshbach:
    """The interior resonance: the Riesz projectors, the Drazin inverse, the
    compatibility condition, and the reduction's own certificates."""

    @pytest.mark.parametrize("fixture", ["complex-symmetric", "non-normal"])
    def test_projectors_are_the_riesz_projectors(self, fixture):
        """Both recorded projectors are idempotent, complementary, commute with
        P_II, and the null projector has the trace of the number of enclosed
        eigenvalues: the defining properties of a Riesz spectral projector."""
        K, A, M = (_complex_symmetric_pencil() if fixture == "complex-symmetric"
                   else _non_normal_pencil())
        interface = _split_interface(K, 4)
        values, interior = _interior_spectrum(A, M, interface)
        lam = complex(values[0])
        F = PS.feshbach(A, M, lam, interface, 1e-10, RADIUS)
        assert F.interiorSingular
        R, Pi0 = np.array(F.rangeProjector), np.array(F.nullProjector)
        ni = len(interior)
        PII = (A - lam * M)[np.ix_(interior, interior)]
        q = F.resonantSpace.shape[1]
        assert q >= 1
        assert np.abs(R @ R - R).max() < 1e-10
        assert np.abs(Pi0 @ Pi0 - Pi0).max() < 1e-10
        assert np.abs(R + Pi0 - np.eye(ni)).max() < 1e-12
        assert np.linalg.norm(Pi0 @ PII - PII @ Pi0) < 1e-10 * np.linalg.norm(PII)
        assert abs(np.trace(Pi0) - q) < 1e-8
        assert abs(F.projectorTrace - q) < 1e-8
        assert F.projectorIdempotency < 1e-10
        assert F.interiorRank == ni - q
        # The projector is spectral, not Hermitian-orthogonal: it need not be
        # Hermitian, and on the non-normal fixture it is not.
        if fixture == "non-normal":
            assert np.abs(Pi0 - Pi0.conj().T).max() > 1e-6

    @pytest.mark.parametrize("fixture", ["complex-symmetric", "non-normal"])
    def test_the_inverse_is_the_drazin_inverse(self, fixture):
        """P_II^D P_II = P_II P_II^D = I - Pi0, P_II^D Pi0 = 0, and for a
        semisimple resonance P_II P_II^D P_II = P_II. For a complex symmetric
        P_II the inverse is complex symmetric."""
        K, A, M = (_complex_symmetric_pencil() if fixture == "complex-symmetric"
                   else _non_normal_pencil())
        interface = _split_interface(K, 4)
        values, interior = _interior_spectrum(A, M, interface)
        lam = complex(values[1])
        F = PS.feshbach(A, M, lam, interface, 1e-10, RADIUS)
        PII = (A - lam * M)[np.ix_(interior, interior)]
        PD, Pi0 = np.array(F.interiorInverse), np.array(F.nullProjector)
        I = np.eye(len(interior))
        assert _rel(PD @ PII, I - Pi0) < 1e-10
        assert _rel(PII @ PD, I - Pi0) < 1e-10
        assert np.linalg.norm(PD @ Pi0) < 1e-10 * np.linalg.norm(PD)
        assert _rel(PD @ PII @ PD, PD) < 1e-10
        # A generic resonance of these fixtures is a single simple eigenvalue,
        # hence semisimple; the resonant eigenvalue is at the computed scatter
        # of the shift, so P_II^D is a {1}-inverse to that accuracy.
        assert _rel(PII @ PD @ PII, PII) < 1e-6
        if fixture == "complex-symmetric":
            assert _rel(PD.T, PD) < 1e-10
            assert _rel(Pi0.T, Pi0) < 1e-10

    def test_a_jordan_block_takes_the_drazin_not_the_group_or_moore_penrose_inverse(self):
        """On an interior block with a Jordan block of size two at the
        resonance, the group inverse does not exist and the Moore-Penrose
        inverse is not similarity-covariant; the Drazin inverse is what is
        used. It satisfies P^D P P^D = P^D and P^3 P^D = P^2 (index two), it
        is NOT a {1}-inverse (P P^D P differs from P by the nilpotent part),
        and it differs from the Moore-Penrose inverse by order one -- so the
        test proves which inverse is in use rather than assuming it."""
        A, M, interface, interior, lam, PII = _declared_pencil(4242, jordan=True)
        F = PS.feshbach(A, M, lam, interface, 1e-10, RADIUS)
        assert F.interiorSingular
        assert F.resonantSpace.shape[1] == 2
        PD, Pi0 = np.array(F.interiorInverse), np.array(F.nullProjector)
        I = np.eye(len(interior))
        assert _rel(PD @ PII, I - Pi0) < 1e-10
        assert _rel(PII @ PD, I - Pi0) < 1e-10
        assert _rel(PD @ PII @ PD, PD) < 1e-10
        assert _rel(PII @ PII @ PII @ PD, PII @ PII) < 1e-10
        assert _rel(PII @ PD @ PII, PII) > 1e-3
        pinv = np.linalg.pinv(PII, rcond=1e-8)
        assert _rel(pinv, PD) > 1e-3
        # The reduction's certificates hold at machine precision even so.
        assert F.reductionResidual < 1e-12
        assert F.projectorIdempotency < 1e-12
        assert abs(F.projectorTrace - 2.0) < 1e-10

    @pytest.mark.parametrize("fixture", ["complex-symmetric", "non-normal"])
    def test_resonant_bases_span_the_generalized_eigenspace(self, fixture):
        """N spans ran Pi0 and is invariant under P_II; N_L spans ran Pi0^T,
        is dual to N in the transpose pairing, and Pi0 = N N_L^T."""
        K, A, M = (_complex_symmetric_pencil() if fixture == "complex-symmetric"
                   else _non_normal_pencil())
        interface = _split_interface(K, 4)
        values, interior = _interior_spectrum(A, M, interface)
        lam = complex(values[1])
        F = PS.feshbach(A, M, lam, interface, 1e-10, RADIUS)
        PII = (A - lam * M)[np.ix_(interior, interior)]
        Nv, NL, Pi0 = np.array(F.resonantSpace), np.array(F.resonantLeftSpace), np.array(F.nullProjector)
        q = Nv.shape[1]
        assert _rel(NL.T @ Nv, np.eye(q)) < 1e-10
        assert _rel(Nv @ NL.T, Pi0) < 1e-10
        compressed = NL.T @ PII @ Nv
        assert np.linalg.norm(PII @ Nv - Nv @ compressed) < 1e-10 * np.linalg.norm(PII)
        assert np.linalg.norm(NL.T @ PII - compressed @ NL.T) < 1e-10 * np.linalg.norm(PII)
        assert np.linalg.norm(Pi0 @ Nv - Nv) < 1e-10
        assert np.linalg.norm(Pi0.T @ NL - NL) < 1e-10

    @pytest.mark.parametrize("fixture", ["complex-symmetric", "non-normal", "jordan"])
    def test_the_resonant_reduction_reproduces_the_pencil_on_every_fiber(self, fixture):
        """The reduction is exact on every retained coordinate, not only on its
        null space: with W = [T | Z] the retained fibers,

            P(lambda) W = E_B Fhat[top] + E_I N Fhat[bottom],

        which rests on P_II P_II^D = I - N N_L^T. The identity is recomputed
        here from the blocks, on a Jordan-block fixture too."""
        if fixture == "jordan":
            A, M, interface, interior, lam, _ = _declared_pencil(7, jordan=True)
        else:
            K, A, M = (_complex_symmetric_pencil() if fixture == "complex-symmetric"
                       else _non_normal_pencil())
            interface = sorted(_split_interface(K, 4))
            values, interior = _interior_spectrum(A, M, interface)
            lam = complex(values[0])
        F = PS.feshbach(A, M, lam, interface, 1e-10, RADIUS)
        nb = len(interface)
        q = F.resonantSpace.shape[1]
        assert F.resonantResponse.shape == (nb + q, nb + q)
        P = A - lam * M
        W = np.hstack([np.array(F.constraintModes), np.array(F.resonantModes)])
        Fhat = np.array(F.resonantResponse)
        predicted = np.zeros_like(W)
        predicted[interface, :] = Fhat[:nb, :]
        predicted[interior, :] = np.array(F.resonantSpace) @ Fhat[nb:, :]
        residual = (np.linalg.norm(P @ W - predicted)
                    / (np.linalg.norm(P) * np.linalg.norm(W)))
        assert residual < 1e-10
        assert abs(residual - F.reductionResidual) < 1e-12

    @pytest.mark.parametrize("fixture", ["complex-symmetric", "non-normal"])
    def test_resonant_reduction_lifts_its_null_vectors_to_the_pencil(self, fixture):
        """Every null vector of Fhat(lambda) lifts to a null vector of
        P(lambda). The residual is measured, not asserted."""
        K, A, M = (_complex_symmetric_pencil() if fixture == "complex-symmetric"
                   else _non_normal_pencil())
        interface = _split_interface(K, 4)
        values, interior = _interior_spectrum(A, M, interface)
        lam = complex(values[0])
        F = PS.feshbach(A, M, lam, interface, 1e-10, RADIUS)
        nb = len(interface)
        Fhat = np.array(F.resonantResponse)
        u, sv, vh = np.linalg.svd(Fhat)
        nullity = int(np.sum(sv <= 1e-10 * sv[0]))
        P = A - lam * M
        scaleP = np.linalg.norm(P)
        T, Z = np.array(F.constraintModes), np.array(F.resonantModes)
        for j in range(nullity):
            y = vh[Fhat.shape[1] - 1 - j].conj()
            x = T @ y[:nb] + Z @ y[nb:]
            assert np.linalg.norm(P @ x) <= 1e-7 * scaleP * np.linalg.norm(x)
        if nullity > 0:
            assert np.isfinite(F.liftResidual) and F.liftResidual < 1e-7
        else:
            # There is no null vector to lift, and the read says so rather than
            # reporting a zero it did not measure.
            assert np.isnan(F.liftResidual)

    @pytest.mark.parametrize("fixture", ["complex-symmetric", "non-normal"])
    def test_compatibility_and_independence_are_measured_on_the_projector(self, fixture):
        """The whitepaper's two conditions on the block elimination, in their
        spectral form: solvability, Pi0 P_IB x_B = 0 (the interface load has
        no component in the generalized eigenspace), and independence of the
        chosen interior solution, P_BI Pi0 = 0. Both are reported as residuals
        and both are reproduced here from the blocks and the projector."""
        K, A, M = (_complex_symmetric_pencil() if fixture == "complex-symmetric"
                   else _non_normal_pencil())
        interface = sorted(_split_interface(K, 4))
        values, interior = _interior_spectrum(A, M, interface)
        lam = complex(values[2])
        F = PS.feshbach(A, M, lam, interface, 1e-10, RADIUS)
        P = A - lam * M
        PIB = P[np.ix_(interior, interface)]
        PBI = P[np.ix_(interface, interior)]
        Pi0 = np.array(F.nullProjector)
        assert abs(np.linalg.norm(Pi0 @ PIB) / np.linalg.norm(PIB)
                   - F.compatibilityResidual) < 1e-10
        assert abs(np.linalg.norm(PBI @ Pi0) / np.linalg.norm(PBI)
                   - F.independenceResidual) < 1e-10
        # The interior solve residual is the compatibility residual: the
        # Drazin inverse solves the interior equation exactly on the
        # complementary invariant subspace and nowhere else.
        assert abs(F.solveResidual - F.compatibilityResidual) < 1e-10

    @pytest.mark.parametrize("fixture", ["semisimple", "jordan", "complex-symmetric"])
    def test_the_reduction_is_similarity_covariant(self, fixture):
        """The reduction commutes with every similarity of the interior
        coordinates, which is the reason the Drazin inverse is the declared
        one: for D = diag(I_B, S) with S a random non-unitary similarity of
        the interior, reducing D^{-1} P D and transforming back gives the
        untransformed reduction -- the response F_B unchanged, the projector
        and the inverse conjugated by S -- at machine precision. The
        Moore-Penrose inverse lacks this property: its complement of the
        transformed pencil differs from the untransformed one by order one,
        which the test shows on the same fixture."""
        A, M, interface, interior, lam, PII = _declared_pencil(
            11, jordan=(fixture == "jordan"), symmetric=(fixture == "complex-symmetric"))
        rng = np.random.default_rng(77)
        ni, n = len(interior), A.shape[0]
        S = rng.normal(size=(ni, ni)) + 1j * rng.normal(size=(ni, ni))
        assert _rel(S.conj().T @ S, np.eye(ni)) > 1e-1  # genuinely non-unitary
        D = np.eye(n, dtype=complex)
        D[np.ix_(interior, interior)] = S
        Dinv = np.linalg.inv(D)
        F = PS.feshbach(A, M, lam, interface, 1e-10, RADIUS)
        G = PS.feshbach(Dinv @ A @ D, Dinv @ M @ D, lam, interface, 1e-10, RADIUS)
        assert F.interiorSingular and G.interiorSingular
        Sinv = np.linalg.inv(S)
        assert _rel(np.array(G.response), np.array(F.response)) < 1e-9
        assert _rel(S @ np.array(G.nullProjector) @ Sinv, np.array(F.nullProjector)) < 1e-9
        assert _rel(S @ np.array(G.interiorInverse) @ Sinv, np.array(F.interiorInverse)) < 1e-9
        # The Moore-Penrose complement on the same two pencils is not covariant.
        P = A - lam * M
        Pt = Dinv @ P @ D
        B, Iidx = interface, interior
        mp = P[np.ix_(B, B)] - P[np.ix_(B, Iidx)] @ np.linalg.pinv(P[np.ix_(Iidx, Iidx)], rcond=1e-8) @ P[np.ix_(Iidx, B)]
        mpt = Pt[np.ix_(B, B)] - Pt[np.ix_(B, Iidx)] @ np.linalg.pinv(Pt[np.ix_(Iidx, Iidx)], rcond=1e-8) @ Pt[np.ix_(Iidx, B)]
        assert _rel(mpt, mp) > 1e-3

    def test_away_from_a_resonance_the_projectors_are_trivial(self):
        K, A, M = _complex_symmetric_pencil()
        interface = _split_interface(K, 4)
        F = PS.feshbach(A, M, complex(0.37, -0.11), interface)
        assert not F.interiorSingular
        ni = A.shape[0] - len(interface)
        assert np.abs(np.array(F.rangeProjector) - np.eye(ni)).max() == 0.0
        assert np.abs(np.array(F.nullProjector)).max() == 0.0
        assert F.resonantSpace.shape[1] == 0
        assert F.interiorInverse.size == 0
        assert F.compatible and F.responseIndependent
        assert F.determinantResidual < 1e-8
        assert F.resonanceSeparation > 1.0

    def test_the_resonance_disc_is_the_declaration(self):
        """The same shift is a resonance or not according to the declared disc:
        the enclosure and separation report the margin of the call."""
        K, A, M = _non_normal_pencil()
        interface = _split_interface(K, 4)
        values, interior = _interior_spectrum(A, M, interface)
        lam = complex(values[0])
        wide = PS.feshbach(A, M, lam, interface, 1e-10, 1e-6)
        assert wide.interiorSingular
        assert wide.resonanceEnclosure < 1.0 < wide.resonanceSeparation
        narrow = PS.feshbach(A, M, lam, interface, 1e-10, 1e-15)
        assert not narrow.interiorSingular
        with pytest.raises(ValueError):
            PS.feshbach(A, M, lam, interface, 1e-10, -1.0)


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
        # the bound its own fine-space residual certifies. The inequality is
        # exact mathematics, so it holds for a truncated surrogate too: what
        # truncation costs is the size of the bound, not its truth.
        for i in range(len(surrogate.windowIndices)):
            if surrogate.resonantAtEigenvalue[i]:
                continue
            assert surrogate.feshbachHolds[i]
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
    def test_retaining_every_mode_makes_the_surrogate_exact(self, fixture):
        """A retention radius that discards nothing leaves the reduction basis
        square and invertible, so the surrogate is the pencil itself: its
        claimed eigenvalues are eigenvalues of the fine pencil and it certifies
        at the tightest tolerance the arithmetic allows."""
        K, A, M = (_complex_symmetric_pencil() if fixture == "complex-symmetric"
                   else _non_normal_pencil())
        interface = _split_interface(K, 4)
        centre, radius = self._window(A, M, interface)
        full = PS.craigBamptonSurrogate(
            A, M, interface, centre, radius, 1e9, complex(0.0, 0.0), 1.0)
        assert full.retainedModes == A.shape[0] - len(interface)
        assert np.isinf(full.discardedModeSeparation)
        assert all(full.feshbachHolds)
        assert full.certified, full.refusal
        exact = np.linalg.eigvals(np.linalg.solve(M, A))
        assert len(full.windowIndices) > 0
        for index in full.windowIndices:
            theta = complex(full.eigenvalues[index])
            assert np.min(np.abs(exact - theta)) < 1e-6 * max(1.0, abs(theta))

    @pytest.mark.parametrize("fixture", ["complex-symmetric", "non-normal"])
    def test_a_retention_radius_inside_the_window_refuses(self, fixture):
        """A fixed-interface mode discarded from inside the declared window
        makes the surrogate uncertified, by name; it is still returned with all
        of its numbers rather than refused outright."""
        K, A, M = (_complex_symmetric_pencil() if fixture == "complex-symmetric"
                   else _non_normal_pencil())
        interface = _split_interface(K, 4)
        centre, radius = self._window(A, M, interface)
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
            PS.craigBamptonSurrogate(A, M, interface, complex(0.0, 0.0), 1.0, -0.5)

    def test_the_window_is_a_disc_in_the_complex_plane(self):
        """On a pencil with a declared complex interior spectrum, the disc
        window retains a mode with a small imaginary part -- a point that is
        not in the real interval at all -- and drops a strongly damped mode
        whose real part lies inside that interval, which a rule by real part
        would have kept. The certified defects of the disc surrogate are
        inside their bounds."""
        rng = np.random.default_rng(5)
        nb, ni = 3, 6
        n = nb + ni
        interface, interior = list(range(nb)), list(range(nb, n))
        centre, radius = complex(1.0, 0.0), 0.5
        lightly_damped = centre + 0.3j * radius        # inside the disc, not real
        strongly_damped = centre + 3.0j * radius       # Re inside [0.5, 1.5], outside the disc
        spectrum = np.array([lightly_damped, strongly_damped, 1.2, 3.0, -2.0, 4.5 + 0.2j])
        M = np.eye(n, dtype=complex)
        A = np.zeros((n, n), dtype=complex)
        C = rng.normal(size=(nb, nb)) + 1j * rng.normal(size=(nb, nb))
        A[:nb, :nb] = C + C.T
        coupling = 0.05 * (rng.normal(size=(nb, ni)) + 1j * rng.normal(size=(nb, ni)))
        A[np.ix_(interface, interior)] = coupling
        A[np.ix_(interior, interface)] = coupling.T
        A[np.ix_(interior, interior)] = np.diag(spectrum)
        surrogate = PS.craigBamptonSurrogate(A, M, interface, centre, radius, radius,
                                             complex(0.0, 0.0), 1e-2)
        # Retained modes are the interior unit vectors whose eigenvalue lies in
        # the disc: read them off the basis.
        V = np.array(surrogate.basis)
        retained = []
        for column in V[:, nb:].T:
            support = np.flatnonzero(np.abs(column) > 1e-8)
            assert len(support) == 1
            retained.append(complex(spectrum[support[0] - nb]))
        assert surrogate.retainedModes == 2
        assert any(abs(t - lightly_damped) < 1e-12 for t in retained)
        assert any(abs(t - 1.2) < 1e-12 for t in retained)
        assert all(abs(t - strongly_damped) > 1e-6 for t in retained)
        assert surrogate.discardedModeSeparation > 0.0
        for i in range(len(surrogate.windowIndices)):
            if not surrogate.resonantAtEigenvalue[i]:
                assert surrogate.feshbachHolds[i]
                assert surrogate.feshbachDefects[i] <= surrogate.feshbachBounds[i] * 1.000001 + 1e-300


class TestRecursiveQuotientSurrogateRegimes:
    """`RecursiveQuotient::craigBampton` no longer refuses the non-normal and
    complex-symmetric-pencil regimes: the regime decides the pairing, not
    whether the surrogate exists."""

    @staticmethod
    def _quotient(A, M):
        dim = A.shape[0]
        components = [list(range(0, dim // 2)), list(range(dim // 2, dim))]
        return cob.RecursiveQuotient.overPencil(
            [complex(z) for z in A.reshape(-1)],
            [complex(z) for z in M.reshape(-1)], dim, components)

    @pytest.mark.parametrize("fixture", ["complex-symmetric", "non-normal"])
    def test_a_bilinear_level_builds_a_surrogate(self, fixture):
        """Both bilinear regimes used to be refused outright. With every
        fixed-interface mode retained the surrogate is the level itself, so the
        pairs it returns are pairs of the fine pencil and their residuals say
        so."""
        K, A, M = (_complex_symmetric_pencil() if fixture == "complex-symmetric"
                   else _non_normal_pencil())
        quotient = self._quotient(A, M)
        assert quotient.regime in (cob.CertificateRegime.NonNormal,
                                   cob.CertificateRegime.ComplexSymmetricPencil)
        values = np.linalg.eigvals(np.linalg.solve(M, A))
        lower = float(values.real.min()) - 1.0
        upper = float(np.sort(values.real)[len(values) // 2])
        read = quotient.craigBampton(lower, upper, float(values.real.max()) + 10.0, 1e-4)
        assert sum(read.retainedModes) > 0
        assert len(read.windowEigenvalues) > 0
        assert max(read.eigenResiduals) < 1e-6
        # Every level the surrogate reports inside the window is a level of the
        # fine pencil.
        for value in read.windowEigenvalues:
            assert np.min(np.abs(values.real - value)) < 1e-6 * max(1.0, abs(value))
