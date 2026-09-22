# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Lorentzian protocol (#908, #1207; integration specification §10,
Requirement 2): the family with the timelike part of every squared length
rotated by e^{-2 i epsilon}, epsilon on the certificate, reads at epsilon = 0
only inside a family with at least one epsilon > 0 and with their gap, the
labeled extrapolation, and the scaling verification plan's G5 rows on the
rotated curved torus, read through the library's family."""
import math

import numpy as np
import pytest
from scipy.linalg import subspace_angles

from tessera import chainhodge as ch
from tests.chainhodge._fixtures import (conformal_torus_rotated, conformal_torus_split,
                                        edges, torus33, torus33_timelike_parts)

LF = ch.LorentzianFamily
KS = ch.Branch.KontsevichSegal


def _hausdorff(a, b):
    a, b = np.asarray(a), np.asarray(b)
    d = np.abs(a[:, None] - b[None, :])
    return max(d.min(axis=1).max(), d.min(axis=0).max())


class TestRotate:
    def test_the_timelike_part_of_every_squared_length_rotates(self):
        """On the CDT-like torus a diagonal (one time step and one space step)
        keeps its spacelike part and has its timelike part rotated; only the
        purely spacelike horizontal edges do not move."""
        K, s = torus33()
        tau = torus33_timelike_parts(K)
        eps = 0.2
        phase = np.exp(-2j * eps)
        r = LF.rotate(s, tau, eps)
        kinds = {"h": 0, "v": 0, "d": 0}
        for (a, b), v, w in zip(edges(K), s, r):
            di, dj = (b // 3 - a // 3) % 3, (b % 3 - a % 3) % 3
            if di == 0:
                kinds["h"] += 1
                assert w == v
            elif dj == 0:
                kinds["v"] += 1
                assert w == pytest.approx(-0.5 * phase, abs=1e-15)
            else:
                kinds["d"] += 1
                assert w == pytest.approx(1.0 - 0.5 * phase, abs=1e-15)
        assert kinds == {"h": 9, "v": 9, "d": 9}

    def test_zero_rotation_returns_the_squared_lengths_exactly(self):
        K, s = torus33()
        assert LF.rotate(s, torus33_timelike_parts(K), 0.0) == s

    def test_a_null_edge_has_its_timelike_part_rotated(self):
        # dx^2 - dt^2 with dx = dt = 1 is null as a whole, and its timelike part
        # -dt^2 still rotates; a purely timelike edge rotates whole.
        s = [1.0, 0.0, -1.0]
        tau = [0.0, -1.0, -1.0]
        r = LF.rotate(s, tau, 0.3)
        assert r[0] == 1.0
        assert r[1] == pytest.approx(1.0 - np.exp(-0.6j), abs=1e-15)
        assert r[2] == pytest.approx(-np.exp(-0.6j), abs=1e-15)

    def test_the_split_must_cover_every_edge(self):
        K, s = torus33()
        with pytest.raises(ValueError):
            LF.rotate(s, [0.0] * 3, 0.1)

    @pytest.mark.parametrize("eps", [-0.1, math.inf, math.nan])
    def test_a_rotation_outside_the_family_is_refused(self, eps):
        K, s = torus33()
        with pytest.raises(ValueError):
            LF.rotate(s, torus33_timelike_parts(K), eps)

    def test_a_non_finite_timelike_part_is_refused(self):
        with pytest.raises(ValueError):
            LF.rotate([1.0, -1.0], [0.0, complex(math.nan, 0.0)], 0.1)


class TestInstanceAndSweep:
    def test_epsilon_on_certificate_and_allowability(self):
        K, s = torus33()
        tau = torus33_timelike_parts(K)
        for eps in (0.05, 0.1, 0.3):
            hodge = LF.instance(K, s, tau, eps, ch.Preset.L2, KS)
            cert = hodge.certificate()
            assert cert.epsilon == eps
            assert cert.allowable and cert.margin > 0.0
            assert not cert.continuationAmbiguous
        zero = LF.instance(K, s, tau, 0.0, ch.Preset.L2, KS)
        assert zero.certificate().epsilon == 0.0
        assert not zero.certificate().allowable
        assert math.isnan(ch.ChainHodge(K, s).certificate().epsilon)

    def test_sweep_reads_carry_gap_and_are_continuous(self):
        K, s = torus33()
        tau = torus33_timelike_parts(K)
        epsilons = [0.0, 0.01, 0.02, 0.04]
        reads = LF.sweep(K, s, tau, epsilons, 1, ch.Preset.L2, KS, 10.0, True)
        assert [r.epsilon for r in reads] == epsilons
        for r in reads:
            assert r.harmonic.nullity == 2
            assert math.isfinite(r.harmonic.gap) and r.harmonic.gap > 1e3
            assert len(r.eigenvalues) == 27
        assert not reads[0].allowable and all(r.allowable for r in reads[1:])
        ev0 = np.array(reads[0].eigenvalues)
        assert np.max(np.abs(ev0.imag)) < 3e-14 * np.max(np.abs(ev0))
        scale = np.max(np.abs(ev0))
        d = [_hausdorff(r.eigenvalues, ev0) / scale for r in reads[1:]]
        assert d[0] < d[1] < d[2]          # the spectrum moves continuously away from the boundary
        # Measured 0.149 at epsilon = 0.01 for the specified family, which also
        # rotates the timelike part of the diagonals (rotating only the vertical
        # edges, as before #1207, moved it 0.083).
        assert d[0] < 0.2

    def test_zero_epsilon_read_is_a_family_member_with_gap(self):
        K, s = torus33()
        tau = torus33_timelike_parts(K)
        reads = LF.sweep(K, s, tau, [0.0, 0.1], 1, ch.Preset.L2, KS)
        assert [r.epsilon for r in reads] == [0.0, 0.1]
        assert math.isfinite(reads[0].harmonic.gap)

    @pytest.mark.parametrize("epsilons", [[0.0], [0.0, 0.0], []])
    def test_a_lone_zero_epsilon_sweep_is_refused(self, epsilons):
        """Requirement 2: results at epsilon = 0 are never reported alone."""
        K, s = torus33()
        with pytest.raises(ValueError):
            LF.sweep(K, s, torus33_timelike_parts(K), epsilons, 1, ch.Preset.L2, KS)

    @pytest.mark.parametrize("epsilons", [[-0.1, 0.1], [0.1, math.nan]])
    def test_a_sweep_outside_the_family_is_refused(self, epsilons):
        K, s = torus33()
        with pytest.raises(ValueError):
            LF.sweep(K, s, torus33_timelike_parts(K), epsilons, 1, ch.Preset.L2, KS)


class TestExtrapolation:
    def test_recovers_polynomial_value_at_zero(self):
        eps = [0.1, 0.2, 0.3, 0.4]
        vals = [1.0 + 2.0 * e - 3.0 * e * e + 0.5j * e for e in eps]
        ex = LF.extrapolateToZero(eps, vals, 2)
        assert ex.extrapolated == pytest.approx(1.0, abs=1e-12)
        assert ex.residual < 1e-12 and ex.order == 2
        assert "extrapolation" in ex.label

    def test_rejects_zero_epsilon_and_short_input(self):
        with pytest.raises(ValueError):
            LF.extrapolateToZero([0.0, 0.1], [1.0, 1.0])
        with pytest.raises(ValueError):
            LF.extrapolateToZero([0.1], [1.0])
        with pytest.raises(ValueError):
            LF.extrapolateToZero([0.1, 0.2], [1.0])

    def test_extrapolated_eigenvalue_approaches_the_boundary_value(self):
        K, s = torus33()
        tau = torus33_timelike_parts(K)
        epsilons = [0.0025, 0.005, 0.0075, 0.01]
        reads = LF.sweep(K, s, tau, epsilons, 1, ch.Preset.L2, KS, 10.0, True)
        # The trace of the pencil operator is analytic in epsilon (individual
        # eigenvalues cross and split, so a single one is not an extrapolation
        # target); a quadratic through reads at epsilon <= 0.01 reaches the
        # boundary value to 4.4e-4 relative. The specified family rotates the
        # diagonals too and its trace bends faster: through reads at
        # epsilon <= 0.02 the quadratic misses by 3.3e-3.
        traces = [complex(np.sum(r.eigenvalues)) for r in reads]
        ex = LF.extrapolateToZero(epsilons, traces, 2)
        zero = LF.sweep(K, s, tau, [0.0] + epsilons, 1, ch.Preset.L2, KS, 10.0, True)[0]
        target = complex(np.sum(zero.eigenvalues))
        assert abs(ex.extrapolated - target) < 1e-3 * abs(target)
        assert abs(target.imag) < 1e-9 * abs(target)


class TestEvidenceRotatedCurvedTorus:
    """Scaling verification plan G5 (§10 finding), reproduced as labeled
    evidence through the library's family: with the timelike part of every
    squared length rotated by e^{-2 i epsilon} the curved Lorentzian torus is
    allowable and its harmonic images converge at second order; at epsilon = 0
    the angles are reported beside the family with the gap and carry no pass
    criterion."""

    @staticmethod
    def _reads(N, epsilons):
        K, s, tau, W = conformal_torus_split(N, 0.3, 0.15, seed=1)
        reads = LF.sweep(K, s, tau, epsilons, 1, ch.Preset.L2, KS, 10.0, False, 2048)
        angles = []
        for read in reads:
            assert read.harmonic.nullity == 2
            angles.append(float(np.max(np.degrees(subspace_angles(read.harmonic.images, W)))))
        return angles, reads

    def test_the_family_is_the_plans_coordinate_rotation(self):
        """rotate on the declared split is the plan's s_e = e^{2 phi}(dx^2 -
        e^{-2 i eps} dt^2), edge for edge."""
        K, s, tau, _ = conformal_torus_split(8, 0.3, 0.15, seed=1)
        for eps in (0.03, 0.1, 0.3, 0.6):
            _, reference, _ = conformal_torus_rotated(8, 0.3, 0.15, eps, seed=1)
            rotated = np.array(LF.rotate(s, tau, eps))
            assert np.max(np.abs(rotated - np.array(reference))) < 1e-15

    @pytest.mark.parametrize("eps", [0.1, 0.3, 0.6])
    def test_the_margins_are_the_specifications_family(self, eps):
        """Each local metric e^{2 phi}(dx^2 - e^{-2 i eps} dt^2) has eigenvalue
        arguments pi - 2 eps and 0, so the continuum margin is 2 eps; the mesh's
        minimum margin sits just below it. Rotating whole edges declared
        timelike instead leaves a different family, with a smaller margin on
        this mesh (0.08 against 0.19 at eps = 0.1)."""
        K, s, tau, _ = conformal_torus_split(8, 0.3, 0.15, seed=1)
        margin = LF.instance(K, s, tau, eps, ch.Preset.L2, KS, 2048).certificate().margin
        assert 0.9 * 2.0 * eps <= margin <= 2.0 * eps + 1e-12
        whole = [v * np.exp(-2j * eps) if v.real < 0.0 else v for v in s]
        other = ch.ChainHodge(K, whole, ch.Preset.L2, KS, 2048, eps).certificate().margin
        assert other < 0.5 * margin

    @pytest.mark.parametrize("eps", [0.1, 0.3, 0.6])
    def test_rotated_family_converges(self, eps):
        (a8,), (r8,) = self._reads(8, [eps])
        (a12,), (r12,) = self._reads(12, [eps])
        assert r8.allowable and r12.allowable and r8.epsilon == eps
        assert a8 < 5.0
        assert a8 / a12 >= 1.5 ** 1.5

    def test_zero_epsilon_is_reported_with_gap(self):
        (a8, _), (z8, _) = self._reads(8, [0.0, 0.1])
        (a12, _), (z12, _) = self._reads(12, [0.0, 0.1])
        assert not z8.allowable and z8.margin == pytest.approx(0.0, abs=1e-12)
        assert math.isfinite(a8) and math.isfinite(a12)
        assert math.isfinite(z8.harmonic.gap) and math.isfinite(z12.harmonic.gap)
