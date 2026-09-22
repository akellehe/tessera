# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The gaps of the harmonic chains and the rank report across the dense
crossover (#1204).

Integration specification section 11 step 2 and Requirement 2, scaling
verification plan T2: the rank conditions (R1)-(R4) with their gap ratios
sigma_r / sigma_{r+1}, the trend of the smallest nonzero singular value with N,
and a read at epsilon = 0 only with its gap. Below the crossover (512 cells)
the kernel and the four products are decomposed by dense SVD; above it
`SparseRank` measures the same singular values without forming them. The
tests check the two paths against each other, the sparse path against dense
SVDs formed here above the crossover, and the trend with N on a Euclidean and
a Lorentzian curved torus."""
import math

import numpy as np
import pytest
from scipy.linalg import subspace_angles

from tessera import chainhodge as ch
from tessera import cobordism as cob
from tests.chainhodge._fixtures import edges, flat_torus, torus33, torus_cells

KS = ch.Branch.KontsevichSegal


def curved_torus(N, epsilon=0.0, lorentz=True, amp=0.3, jitter=0.15, seed=0, Lt=1.0, Lx=2.0):
    """The conformally flat torus of the scaling verification plan's section 7
    with period ratio 2: s_e = e^{2 phi(mid_e)} (dx^2 - e^{-2 i eps} dt^2)
    (Lorentzian) or e^{2 phi}(dx^2 + dt^2), phi = amp sin(2 pi t / Lt)
    cos(2 pi x / Lx). With unequal periods no lattice diagonal is near null.
    Returns the complex, the squared lengths, and the continuum harmonic edge
    integrals (dt, dx)."""
    rng = np.random.default_rng(seed)
    cells, vid = torus_cells(N)
    K = cob.ChainComplex.fromTopCells(cells)
    period = np.array([Lt, Lx])
    coords = {vid(i, j): np.array([(i + jitter * rng.uniform(-1, 1)) * Lt / N,
                                   (j + jitter * rng.uniform(-1, 1)) * Lx / N])
              for i in range(N) for j in range(N)}

    def phi(p):
        return amp * np.sin(2 * np.pi * p[0] / Lt) * np.cos(2 * np.pi * p[1] / Lx)

    rot = np.exp(-2j * epsilon)
    s, W, tau = [], [], []
    for (a, b) in edges(K):
        d = coords[b] - coords[a]
        d -= period * np.round(d / period)
        q = (d[1] ** 2 - rot * d[0] ** 2) if lorentz else (d[1] ** 2 + d[0] ** 2)
        weight = np.exp(2 * phi(coords[a] + 0.5 * d))
        s.append(complex(weight * q))
        tau.append(complex(-weight * d[0] ** 2) if lorentz else 0j)
        W.append(d.copy())
    curved_torus.timelike_parts = tau
    return K, s, np.array(W, dtype=complex)


def _sweep(K, s, tau, epsilons):
    """LorentzianFamily.sweep with the declaration the build expects: the
    timelike part of every squared length (#1207), or, before #1207, a
    declared causal type per edge (timelike where the edge has a timelike
    part and its squared length is negative)."""
    if hasattr(ch, "CausalType"):
        types = [ch.CausalType.Timelike if (t != 0 and v.real < 0) else ch.CausalType.Spacelike
                 for v, t in zip(s, tau)]
        return ch.LorentzianFamily.sweep(K, s, types, epsilons, 1, ch.Preset.L2, KS)
    return ch.LorentzianFamily.sweep(K, s, tau, epsilons, 1, ch.Preset.L2, KS)


def _stacked(hodge):
    B1 = hodge.boundary(1).toarray()
    B2 = hodge.boundary(2).toarray()
    return np.vstack([B2.T, B1 @ hodge.Minv(1).toarray()])


def _products(hodge, k=1):
    """The four products of (R1)-(R4) for the Whitney preset, formed densely."""
    M = [hodge.Minv(j).toarray() for j in range(3)]
    B1 = hodge.boundary(1).toarray()
    B2 = hodge.boundary(2).toarray()
    return [B2.T @ np.linalg.solve(M[1], B2), B1 @ M[1] @ B1.T,
            B1.T @ np.linalg.solve(M[0], B1), B2 @ M[2] @ B2.T]


def _instances():
    K6, s6, _ = flat_torus(6, 0.25, False, seed=4)
    rng = np.random.default_rng(29)
    s6 = [v + 0.01j * rng.normal() for v in s6]
    K, sL = torus33()
    _, sE = torus33(1.0, 1.0, 1.0)
    return {"T6 Lorentzian 3x3": (K, sL, KS), "T7 Euclidean 3x3": (K, sE, KS),
            "complex jittered 6x6": (K6, s6, ch.Branch.Continuation)}


class TestDenseAndSparsePathsAgree:
    """Below the crossover both paths run on the same instance."""

    @pytest.mark.parametrize("name", list(_instances()))
    def test_rank_conditions(self, name):
        K, s, branch = _instances()[name]
        hodge = ch.ChainHodge(K, s, ch.Preset.L2, branch)
        dense = hodge.rankConditions(1)
        sparse = hodge.rankConditions(1, 10.0, True)
        assert dense.dense and not sparse.dense
        assert list(dense.measured) == list(sparse.measured) == list(dense.expected)
        assert dense.kernelIsHarmonic and sparse.kernelIsHarmonic
        for d, sp_ in zip(dense.splits, sparse.splits):
            assert d.at == sp_.at
            assert sp_.largest == pytest.approx(d.largest, rel=1e-9)
            assert sp_.sigmaAt == pytest.approx(d.sigmaAt, rel=1e-8)
            assert sp_.sigmaNext < 1e-12 * sp_.largest and d.sigmaNext < 1e-12 * d.largest
            assert sp_.gap > 1e10 and d.gap > 1e10

    @pytest.mark.parametrize("name", list(_instances()))
    def test_harmonic_chains(self, name):
        K, s, branch = _instances()[name]
        hodge = ch.ChainHodge(K, s, ch.Preset.L2, branch)
        dense = hodge.harmonicChains(1)
        sparse = hodge.harmonicChains(1, 10.0, True)
        assert dense.nullity == sparse.nullity == 2
        assert np.max(np.degrees(subspace_angles(dense.images, sparse.images))) < 1e-8
        assert sparse.largestSingular == pytest.approx(dense.largestSingular, rel=1e-9)
        assert sparse.lastKept == pytest.approx(dense.lastKept, rel=1e-8)
        assert math.isfinite(sparse.gap) and sparse.gap > 1e10 and dense.gap > 1e10

    def test_grassmann_preset(self):
        K, s = torus33()
        hodge = ch.ChainHodge(K, s, ch.Preset.GRASSMANN_ALL)
        dense = hodge.rankConditions(1)
        sparse = hodge.rankConditions(1, 10.0, True)
        assert list(dense.measured) == list(sparse.measured) == list(dense.expected)
        for d, sp_ in zip(dense.splits, sparse.splits):
            assert sp_.sigmaAt == pytest.approx(d.sigmaAt, rel=1e-8)
        read = hodge.harmonicChains(1, 10.0, True)
        assert read.nullity == 2 and read.gap > 1e10

    def test_every_degree(self):
        """Degrees 0 and 2 carry the vacuous conditions (expected rank 0) as
        an empty split: nothing required, nothing beyond."""
        K, s = torus33()
        hodge = ch.ChainHodge(K, s, ch.Preset.L2, KS)
        for k in (0, 2):
            dense = hodge.rankConditions(k)
            sparse = hodge.rankConditions(k, 10.0, True)
            assert list(dense.measured) == list(sparse.measured) == list(dense.expected)
            for i, (d, sp_) in enumerate(zip(dense.splits, sparse.splits)):
                if dense.expected[i] == 0:
                    assert math.isinf(sp_.sigmaAt) and sp_.sigmaNext == 0.0 and math.isinf(sp_.gap)
                else:
                    assert sp_.sigmaAt == pytest.approx(d.sigmaAt, rel=1e-8)


@pytest.fixture(scope="module")
def instance():
    K, s, _ = curved_torus(16, 0.1)
    return ch.ChainHodge(K, s, ch.Preset.L2, KS)


class TestAboveTheCrossover:
    """N = 16: 768 edges, above the default crossover of 512 cells. The dense
    SVDs of the stacked matrix and the four products are formed here, in the
    test, as the oracle for the sparse path."""

    def test_rank_conditions_are_measured(self, instance):
        rep = instance.rankConditions(1)
        assert not rep.dense
        assert rep.kernelIsHarmonic and list(rep.measured) == list(rep.expected)
        for split, P, rho in zip(rep.splits, _products(instance), rep.expected):
            sv = np.linalg.svd(P, compute_uv=False)
            assert split.at == rho
            assert split.largest == pytest.approx(sv[0], rel=1e-9)
            assert split.sigmaAt == pytest.approx(sv[rho - 1], rel=1e-7)
            assert split.sigmaNext < 1e-11 * sv[0]
            assert split.gap > 1e8

    def test_harmonic_gap_is_measured(self, instance):
        read = instance.harmonicChains(1)
        assert not read.dense and read.nullity == 2
        sv = np.linalg.svd(_stacked(instance), compute_uv=False)
        assert read.largestSingular == pytest.approx(sv[0], rel=1e-9)
        assert read.lastKept == pytest.approx(sv[-3], rel=1e-8)
        assert read.firstDiscarded < 1e-13 * sv[0]
        assert math.isfinite(read.gap) and read.gap > 1e10

    def test_lorentzian_sweep_carries_the_gap_of_every_read(self):
        """The epsilon = 0 member of a family above the crossover carries its
        gap (Requirement 2); before #1204 it was NaN there."""
        K, s, _ = curved_torus(14, 0.0)                       # 588 edges
        reads = _sweep(K, s, curved_torus.timelike_parts, [0.0, 0.1])
        assert [r.epsilon for r in reads] == [0.0, 0.1]
        for r in reads:
            assert not r.harmonic.dense and r.harmonic.nullity == 2
            assert math.isfinite(r.harmonic.gap) and r.harmonic.gap > 1e8
            assert 0.0 < r.harmonic.lastKept < r.harmonic.largestSingular


def _trend(Ns, **kw):
    """Per N: the relative smallest nonzero singular value of S and of each
    product, the gaps, and whether the conditions hold."""
    rows = []
    for N in Ns:
        K, s, _ = curved_torus(N, **kw)
        hodge = ch.ChainHodge(K, s, ch.Preset.L2, KS)
        read = hodge.harmonicChains(1)
        rep = hodge.rankConditions(1)
        rows.append({"N": N, "dense": read.dense, "nullity": read.nullity,
                     "S": read.lastKept / read.largestSingular, "S_gap": read.gap,
                     "R": [sp_.sigmaAt / sp_.largest for sp_ in rep.splits],
                     "R_gap": [sp_.gap for sp_ in rep.splits], "holds": rep.kernelIsHarmonic})
    return rows


def _slope(Ns, values):
    return float(np.polyfit(np.log(Ns), np.log(values), 1)[0])


TREND_NS = (8, 12, 16, 24)


@pytest.fixture(scope="module")
def euclidean():
    return _trend(TREND_NS, lorentz=False)


@pytest.fixture(scope="module")
def rotated():
    return _trend(TREND_NS, epsilon=0.1)


@pytest.fixture(scope="module")
def real():
    return _trend(TREND_NS, epsilon=0.0)


class TestTrendWithN:
    """T2: (R1)-(R4) with their gaps and the smallest nonzero singular value
    against N = 8, 12 (dense) and 16, 24 (sparse). On the Euclidean torus and on
    the Lorentzian torus rotated to epsilon = 0.1 every condition holds with a
    Euclidean-like gap at every N and the margin of the stacked matrix falls
    like a power of N. On the real Lorentzian torus (epsilon = 0) the margin
    falls markedly faster: the rank-condition failure set of Proposition 2
    approached under refinement. It is (R1), rank(d_2^T M_1^{-1} d_2) =
    rank d_2, whose margin collapses: relative smallest required singular
    value 1.0e-4, 2.1e-6, 1.8e-6, 6.3e-8 at N = 8, 12, 16, 24 (gap 8.6e13 down
    to 7.4e8), against 2.2e-3 down to 1.9e-4 at epsilon = 0.1. The gaps of the
    sparse path (N = 16, 24) sit one to two decades below the dense ones
    because its first discarded singular value is measured as ||P N_C||, an
    upper bound at the rounding level."""

    Ns = TREND_NS

    @pytest.mark.parametrize("family", ["euclidean", "rotated", "real"])
    def test_every_size_reports_conditions_and_gaps(self, family, request):
        rows = request.getfixturevalue(family)
        assert [r["dense"] for r in rows] == [True, True, False, False]
        for r in rows:
            assert r["nullity"] == 2 and r["holds"]
            assert all(math.isfinite(g) and g > 1e8 for g in r["R_gap"])
            assert math.isfinite(r["S_gap"]) and r["S_gap"] > 1e10

    @pytest.mark.parametrize("family", ["euclidean", "rotated"])
    def test_allowable_margin_falls_like_a_power_of_n(self, family, request):
        rows = request.getfixturevalue(family)
        S = [r["S"] for r in rows]
        assert all(a > b for a, b in zip(S, S[1:]))
        assert -2.0 < _slope(self.Ns, S) < -0.5

    def test_real_lorentzian_margin_collapses_faster(self, rotated, real):
        s_rot = [r["S"] for r in rotated]
        s_real = [r["S"] for r in real]
        assert _slope(self.Ns, s_real) < _slope(self.Ns, s_rot) - 0.75
        assert s_real[-1] < 0.2 * s_rot[-1]
        r1_rot = [r["R"][0] for r in rotated]
        r1_real = [r["R"][0] for r in real]
        assert _slope(self.Ns, r1_real) < _slope(self.Ns, r1_rot) - 2.0
        assert r1_real[-1] < 1e-2 * r1_rot[-1]
        assert real[-1]["R_gap"][0] < 1e-2 * rotated[-1]["R_gap"][0]
