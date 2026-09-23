# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Acceptance tests for the Berry-cancelled exchange / rotation holonomy
(:class:`tessera.observables.ExchangeHolonomy`), ticket #772 / design spec
section 15 (Algorithm H) / whitepaper "Fermion statistics from simplicial
orientation".

Covers every ticket acceptance bullet:

* the RAW loop determinant carries an arbitrary common Berry phase (injected
  through a complex metric phase — visibly not +-1: the mandatory negative
  control), while the normalized one-exchange ratio and the structural
  permutation parity give exactly -1; two exchanges give +1;
* an odd cluster exchanging with an even composite gives +1 (the transported
  three-cycle realization, the identical even-composite swap, and the #766
  graded-swap regression);
* the normalized character and the structural parity are invariant under
  vertex relabeling, arbitrary in-band frame rotation, and simplex
  reorientation (the channel-separation gauges);
* duplicate complete one-particle modes wedge to exactly zero (#766,
  regression-tested here);
* a closed-gap fixture returns UNCERTIFIED, never a sign;
* the clean spin-1/2 2 pi ratio is exactly -1 on the constructed total-space
  spin holonomy cycle, the vector fixture exactly +1, and the doubly
  cancelled spin-statistics ratio chi(exchange) * chi(2 pi)^{-1} is +1;
* spin / non-spin Cech fixtures accept / reject the SO(d) -> Spin(d) lift
  with the w2 obstruction (the pillowcase nontrivial class);
* the existing total-J-squared oracle values 3/4 (proton eigenstate) and
  15/4 (Delta) remain exact.

Analytic fixture: two (or more) localized modes on a ring of cells, moved by
half-turn translation (exchange), full-turn translation (double exchange),
or held static (the matched reference).  All step overlaps are real positive
diagonal/permutation blocks times the injected metric phase, so every
expected character is exact in closed form.
"""
import cmath
import math
import unittest

import numpy as np

import tessera

obs = tessera.observables
quantum = tessera.quantum

EH = obs.ExchangeHolonomy

MACHINE = 1e-12   # closed-form fixtures evaluated in floating point
OMEGA = 0.7       # the injected common metric (Berry) phase per step

UP = np.array([1.0, 0.0], complex)
DN = np.array([0.0, 1.0], complex)


# --------------------------------------------------------------------------- #
# fixture builders
# --------------------------------------------------------------------------- #
def _mode(x, n):
    """A localized unit mode at ring position x: cos/sin interpolation
    between cells floor(x) and floor(x)+1 (unit norm, real nonnegative)."""
    k = int(math.floor(x)) % n
    f = x - math.floor(x)
    v = np.zeros(n, complex)
    v[k] += math.cos(f * math.pi / 2.0)
    v[(k + 1) % n] += math.sin(f * math.pi / 2.0)
    return v


def _translation_frames(positions, n, steps, distance):
    """Frames of len(positions) modes advancing `distance` cells over
    `steps` cyclic steps on a ring of n cells."""
    frames = []
    for t in range(steps):
        x = distance * t / steps
        cols = [_mode((p + x) % n, n) for p in positions]
        frames.append(np.stack(cols, axis=1))
    return frames


def _weights(n, omega=OMEGA):
    return np.exp(1j * omega) * np.ones(n, complex)


def _exchange_frames(steps=8, n=8):
    """One exchange: two modes at 0 and 4 advance half the ring."""
    return _translation_frames([0, 4], n, steps, 4)


def _double_exchange_frames(steps=16, n=8):
    """Two exchanges: the same two modes advance the whole ring."""
    return _translation_frames([0, 4], n, steps, 8)


def _three_cycle_frames(steps=8, n=12):
    """One odd cluster (mode at 0) exchanging with an even composite (modes
    at 4 and 8): the +4 translation realizes the block swap as the
    microscopic mode three-cycle."""
    return _translation_frames([0, 4, 8], n, steps, 4)


def _even_even_frames(steps=8, n=8):
    """Two IDENTICAL even composites (adjacent mode pairs at {0,1} and
    {4,5}) exchanging by the +4 translation."""
    return _translation_frames([0, 1, 4, 5], n, steps, 4)


def _reference_frames(frames, steps=None):
    """The matched non-exchanging reference: the base frame held for the
    same number of steps (same rank, same timing, same metric)."""
    count = len(frames) if steps is None else steps
    return [frames[0]] * count


def _loop(frames, n=None, omega=OMEGA, cfg=None):
    n = frames[0].shape[0] if n is None else n
    if cfg is None:
        return EH.loopHolonomy(frames, _weights(n, omega))
    return EH.loopHolonomy(frames, _weights(n, omega), cfg)


# ---- synthetic SpectralFiber construction (via the public fromRecord) ---- #
_CERT_UNMEASURED = float("nan")


def _fiber(frame, cells, weights=None, accepted=True, degree=0,
           eigenvalues=None):
    """A synthetic #769 SpectralFiber from an explicit frame over explicit
    cell tuples, built through the public record schema (the documented
    replay/serialization path)."""
    frame = np.asarray(frame, complex)
    if frame.ndim == 1:
        frame = frame[:, None]
    rows, rank = frame.shape
    if weights is None:
        weights = np.exp(1j * OMEGA) * np.ones(rows, complex)
    weights = np.asarray(weights, complex)
    eigenvalues = ([0.0] * rank if eigenvalues is None
                   else list(eigenvalues))
    # Psi^dagger W Phi = I for the biorthogonal-normalized left frame:
    # Psi = W^{-dagger} Phi G^{-dagger} with G = Phi^dagger W Phi.
    gram = frame.conj().T @ np.diag(weights) @ frame
    left = np.linalg.solve(np.diag(weights).conj(),
                           np.linalg.solve(gram, frame.conj().T
                                           @ np.diag(weights)).conj().T)
    flat = lambda m: [complex(m[i, j]) for i in range(m.shape[0])
                      for j in range(m.shape[1])]
    inner = {
        "grade": "certified-numerical" if accepted else "heuristic-discovery",
        "domain": "band-window",
        "regime": "non-normal",
        "residual": 0.0 if accepted else _CERT_UNMEASURED,
        "conditioning": 1.0,
        "dense_reference_error": _CERT_UNMEASURED,
        "tolerance": 1e-9,
    }
    certificate = {
        "degree": degree, "rank": rank,
        "lower_gap": 1.0, "upper_gap": 1.0,
        "localization": 1.0 / rows,
        "projector_residual": 0.0, "eigen_residual": 0.0,
        "left_residual": 0.0, "gram_defect": 0.0,
        "condition_number": 1.0,
        "positive_signature": rank, "negative_signature": 0,
        "frequency_lower": 0.0, "frequency_upper": 0.0,
        "self_adjoint": False, "accepted": bool(accepted),
        "certificate": inner,
    }
    record = {
        "schema_version": 1, "record_type": "spectral_fiber",
        "cells": [[int(v) for v in cell] for cell in cells],
        "rows": rows, "rank": rank,
        "eigenvalues_re": [float(np.real(e)) for e in eigenvalues],
        "eigenvalues_im": [float(np.imag(e)) for e in eigenvalues],
        "right_frame_re": [float(z.real) for z in flat(frame)],
        "right_frame_im": [float(z.imag) for z in flat(frame)],
        "left_frame_re": [float(z.real) for z in flat(left)],
        "left_frame_im": [float(z.imag) for z in flat(left)],
        "weights_re": [float(w.real) for w in weights],
        "weights_im": [float(w.imag) for w in weights],
        "certificate": certificate,
    }
    return obs.SpectralFiber.fromRecord(record)


def _ring_cells(n, offset=0):
    return [[offset + c] for c in range(n)]


def _block_steps(positions, n, steps, distance, ranks=None, omega=OMEGA):
    """T x B per-block fiber tracking of the translation fixture: block b
    holds `ranks[b]` adjacent modes starting at positions[b]."""
    ranks = [1] * len(positions) if ranks is None else ranks
    out = []
    for t in range(steps):
        x = distance * t / steps
        row = []
        for p, r in zip(positions, ranks):
            cols = [_mode((p + k + x) % n, n) for k in range(r)]
            row.append(_fiber(np.stack(cols, axis=1), _ring_cells(n),
                              weights=_weights(n, omega)))
        out.append(row)
    return out


def _static_block_steps(positions, n, steps, ranks=None, omega=OMEGA):
    return _block_steps(positions, n, steps, 0, ranks=ranks, omega=omega)


def _rot(theta):
    return np.array([[math.cos(theta), -math.sin(theta)],
                     [math.sin(theta), math.cos(theta)]])


def _rx(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[1.0, 0, 0], [0, c, -s], [0, s, c]])


def _ry(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, 0, s], [0, 1.0, 0], [-s, 0, c]])


def _rz(theta, d=3):
    c, s = math.cos(theta), math.sin(theta)
    out = np.eye(d)
    out[0, 0] = c
    out[1, 1] = c
    out[0, 1] = -s
    out[1, 0] = s
    return out


def _random_rotation(d, seed):
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.standard_normal((d, d)))
    q = q @ np.diag(np.sign(np.diag(r)))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def _kron(*factors):
    out = factors[0]
    for f in factors[1:]:
        out = np.kron(out, f)
    return out


# --------------------------------------------------------------------------- #
# transport primitive
# --------------------------------------------------------------------------- #
class TransportPrimitiveTest(unittest.TestCase):
    def test_polar_unitary_is_unitary_and_matches_svd(self):
        rng = np.random.default_rng(7)
        m = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
        p = EH.polarUnitary(m)
        self.assertLess(np.abs(p.conj().T @ p - np.eye(4)).max(), MACHINE)
        u, _s, vh = np.linalg.svd(m)
        self.assertLess(np.abs(p - u @ vh).max(), 1e-9)

    def test_polar_is_unitarily_equivariant(self):
        rng = np.random.default_rng(8)
        m = rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3))
        g0, _ = np.linalg.qr(rng.standard_normal((3, 3))
                             + 1j * rng.standard_normal((3, 3)))
        g1, _ = np.linalg.qr(rng.standard_normal((3, 3))
                             + 1j * rng.standard_normal((3, 3)))
        lhs = EH.polarUnitary(g1 @ m @ g0)
        rhs = g1 @ EH.polarUnitary(m) @ g0
        self.assertLess(np.abs(lhs - rhs).max(), 1e-9)

    def test_single_frame_loop_is_identity(self):
        frame = np.eye(4, 2, dtype=complex)
        read = _loop([frame], n=4, omega=0.0)
        self.assertLess(np.abs(read.holonomy - np.eye(2)).max(), MACHINE)
        self.assertTrue(read.certificate.holds())

    def test_loop_reports_unit_modulus_determinant(self):
        read = _loop(_exchange_frames())
        self.assertAlmostEqual(abs(read.determinant), 1.0, delta=MACHINE)
        self.assertLess(read.unitarityResidual, MACHINE)
        self.assertEqual(read.steps, 8)
        self.assertEqual(read.rank, 2)
        self.assertEqual(len(read.stepReads), 8)
        self.assertTrue(all(s.certified for s in read.stepReads))

    def test_shape_mismatch_throws(self):
        frames = _exchange_frames()
        frames[3] = frames[3][:, :1]
        with self.assertRaises(ValueError):
            _loop(frames)

    def test_weights_size_mismatch_throws(self):
        with self.assertRaises(ValueError):
            EH.loopHolonomy(_exchange_frames(), np.ones(5, complex))

    def test_per_step_weights_count_mismatch_throws(self):
        frames = _exchange_frames()
        with self.assertRaises(ValueError):
            EH.loopHolonomyPerStep(frames,
                                   [np.ones(8, complex)] * (len(frames) - 1))

    def test_empty_loop_throws(self):
        with self.assertRaises(ValueError):
            EH.loopHolonomy([], np.ones(8, complex))

    def test_per_step_weights_match_constant_weights(self):
        frames = _exchange_frames()
        a = EH.loopHolonomy(frames, _weights(8))
        b = EH.loopHolonomyPerStep(frames, [_weights(8)] * len(frames))
        self.assertLess(np.abs(a.holonomy - b.holonomy).max(), MACHINE)


# --------------------------------------------------------------------------- #
# the exchange fixtures (interferometric channel)
# --------------------------------------------------------------------------- #
class ExchangeCharacterTest(unittest.TestCase):
    def test_raw_determinant_is_visibly_not_a_sign(self):
        """The mandatory negative control: the raw loop determinant carries
        the injected common Berry (metric) phase and is NOT +-1."""
        steps = 8
        read = _loop(_exchange_frames(steps))
        expected = -cmath.exp(1j * 2 * OMEGA * steps)
        self.assertLess(abs(read.determinant - expected), MACHINE)
        self.assertGreater(abs(read.determinant - 1.0), 0.5)
        self.assertGreater(abs(read.determinant + 1.0), 0.5)

    def test_reference_determinant_carries_the_same_berry_phase(self):
        steps = 8
        frames = _exchange_frames(steps)
        ref = _loop(_reference_frames(frames))
        self.assertLess(
            abs(ref.determinant - cmath.exp(1j * 2 * OMEGA * steps)),
            MACHINE)

    def test_one_exchange_normalized_ratio_is_exactly_minus_one(self):
        frames = _exchange_frames()
        chi = EH.exchangeCharacter(_loop(frames),
                                   _loop(_reference_frames(frames)))
        self.assertLess(abs(chi.character + 1.0), MACHINE)
        self.assertEqual(chi.characterSign, -1)
        self.assertTrue(chi.certificate.holds())
        self.assertTrue(chi.timingMatched)
        self.assertTrue(chi.ranksMatched)
        self.assertEqual(chi.channel, obs.HolonomyChannel.ParticleExchange)

    def test_two_exchanges_give_plus_one(self):
        frames = _double_exchange_frames()
        chi = EH.exchangeCharacter(_loop(frames),
                                   _loop(_reference_frames(frames)))
        self.assertLess(abs(chi.character - 1.0), MACHINE)
        self.assertEqual(chi.characterSign, +1)
        self.assertTrue(chi.certificate.holds())

    def test_odd_cluster_with_even_composite_gives_plus_one(self):
        """The microscopic three-cycle realization of the odd/even block
        swap: mode parity (-1)^{1*2} = +1 shows up in the determinant."""
        frames = _three_cycle_frames()
        chi = EH.exchangeCharacter(_loop(frames, n=12),
                                   _loop(_reference_frames(frames), n=12))
        self.assertLess(abs(chi.character - 1.0), MACHINE)
        self.assertEqual(chi.characterSign, +1)

    def test_identical_even_composites_exchange_gives_plus_one(self):
        frames = _even_even_frames()
        chi = EH.exchangeCharacter(_loop(frames),
                                   _loop(_reference_frames(frames)))
        self.assertLess(abs(chi.character - 1.0), MACHINE)
        self.assertEqual(chi.characterSign, +1)

    def test_timing_mismatch_is_reported_and_uncertified(self):
        frames = _exchange_frames()
        chi = EH.exchangeCharacter(
            _loop(frames), _loop(_reference_frames(frames, steps=6)))
        self.assertFalse(chi.timingMatched)
        self.assertFalse(chi.certificate.holds())
        self.assertEqual(chi.characterSign, 0)

    def test_rank_mismatch_is_reported_and_uncertified(self):
        frames = _exchange_frames()
        ref = [frames[0][:, :1]] * len(frames)
        chi = EH.exchangeCharacter(_loop(frames), _loop(ref))
        self.assertFalse(chi.ranksMatched)
        self.assertFalse(chi.certificate.holds())
        self.assertEqual(chi.characterSign, 0)

    def test_leaking_transfer_is_rejected_before_a_sign(self):
        # An orthogonal jump: the tracked modes teleport by 2 cells, the
        # overlap singular values collapse, and no sign is emitted.
        frames = [np.stack([_mode(0, 8), _mode(4, 8)], axis=1),
                  np.stack([_mode(2, 8), _mode(6, 8)], axis=1)]
        loop = _loop(frames + frames)
        self.assertFalse(loop.certificate.holds())
        self.assertLess(loop.minStepSingularValue, 1e-9)
        chi = EH.exchangeCharacter(loop, _loop(_reference_frames(frames,
                                                                 steps=4)))
        self.assertEqual(chi.characterSign, 0)
        self.assertFalse(chi.certificate.holds())

    def test_ill_conditioned_transfer_is_rejected(self):
        cfg = obs.ExchangeHolonomyConfig()
        cfg.conditionCap = 1.0000001  # every real step conditions above 1
        frames = _exchange_frames()
        # Unequal per-column motion => step conditioning > cap.
        frames[1] = np.stack([_mode(0.9, 8), _mode(4.2, 8)], axis=1)
        read = _loop(frames, cfg=cfg)
        self.assertFalse(read.certificate.holds())


# --------------------------------------------------------------------------- #
# structural permutation channel (block tracking over #769 fibers)
# --------------------------------------------------------------------------- #
class BlockPermutationTest(unittest.TestCase):
    STEPS = 16

    def test_one_exchange_swap_and_minus_one_parities(self):
        steps = _block_steps([0, 4], 8, self.STEPS, 4)
        ref = _static_block_steps([0, 4], 8, self.STEPS)
        read = EH.blockPermutation(steps, ref)
        self.assertEqual(list(read.blockPermutation), [1, 0])
        self.assertEqual(read.blockParity, -1)
        self.assertEqual(read.occupationParity, -1)
        self.assertEqual(list(read.blockRanks), [1, 1])
        self.assertTrue(read.certificate.holds())
        self.assertLess(read.residualInBlockMotion, MACHINE)

    def test_double_exchange_identity_and_plus_one(self):
        steps = _block_steps([0, 4], 8, 2 * self.STEPS, 8)
        read = EH.blockPermutation(steps)
        self.assertEqual(list(read.blockPermutation), [0, 1])
        self.assertEqual(read.blockParity, +1)
        self.assertEqual(read.occupationParity, +1)

    def test_odd_even_three_cycle_statistic_plus_one(self):
        """Odd cluster {0} + even composite {4, 8}: the mode three-cycle has
        parity +1 — the graded sign (-1)^{1*2} of the block swap."""
        steps = _block_steps([0, 4, 8], 12, self.STEPS, 4)
        read = EH.blockPermutation(steps, composites=[[0], [1, 2]])
        self.assertEqual(list(read.blockPermutation), [1, 2, 0])
        self.assertEqual(read.occupationParity, +1)
        self.assertEqual(read.blockParity, +1)  # a 3-cycle of labels
        # The composite's constituents scatter between territories, so the
        # composite-level view is honestly absent (never guessed)...
        self.assertEqual(list(read.compositePermutation), [])
        self.assertEqual(read.compositeParity, 0)
        # ...while the block and mode channels stay certified.
        self.assertTrue(read.certificate.holds())

    def test_identical_even_composites_swap_parities(self):
        """Two rank-two blocks exchanged, each carrying two occupied modes:
        the statistic is the graded sign (-1)^{2*2} = +1 of the declared
        occupations, and the rank-parity cross-check agrees because here the
        occupations and the ranks coincide."""
        steps = _block_steps([0, 4], 8, self.STEPS, 4, ranks=[2, 2])
        two = obs.ClusterOccupancy(occupation=2, sheetCount=1)
        read = EH.blockPermutation(steps, composites=[[0], [1]],
                                   occupancies=[two, two])
        self.assertEqual(list(read.blockPermutation), [1, 0])
        self.assertEqual(list(read.blockRanks), [2, 2])
        self.assertEqual(list(read.blockOccupations), [2, 2])
        self.assertEqual(read.blockParity, -1)     # block labels transpose
        self.assertEqual(read.occupationParity, +1)  # (-1)^{2*2}: the statistic
        self.assertEqual(read.rankParity, +1)        # the cross-check agrees
        self.assertTrue(read.rankParityAgrees)
        self.assertFalse(read.rankParityRetired)
        self.assertEqual(list(read.compositePermutation), [1, 0])
        self.assertEqual(read.compositeParity, -1)
        self.assertTrue(read.certificate.holds())

    def test_missing_reference_reports_unmeasured_residual(self):
        read = EH.blockPermutation(_block_steps([0, 4], 8, self.STEPS, 4))
        self.assertTrue(math.isnan(read.residualInBlockMotion))
        self.assertEqual(read.occupationParity, -1)

    def test_exchanging_reference_is_rejected(self):
        steps = _block_steps([0, 4], 8, self.STEPS, 4)
        read = EH.blockPermutation(steps, steps)  # reference exchanges too
        self.assertFalse(read.certificate.holds())
        self.assertEqual(read.occupationParity, 0)

    def test_residual_detects_uncancelled_in_block_motion(self):
        # The moving loop crosses per-cell metric phases the static
        # reference never sees: the cancelled in-block motion is nonzero
        # and reported, while the parity stays exact.
        n, steps = 8, self.STEPS
        varying = np.exp(1j * np.linspace(0.0, 1.1, n))
        loop = []
        for t in range(steps):
            x = 4 * t / steps
            row = [_fiber(np.stack([_mode((p + x) % n, n)], axis=1),
                          _ring_cells(n), weights=varying)
                   for p in (0, 4)]
            loop.append(row)
        ref = [[_fiber(_mode(p, n), _ring_cells(n), weights=np.ones(n))
                for p in (0, 4)] for _ in range(steps)]
        read = EH.blockPermutation(loop, ref)
        self.assertTrue(read.certificate.holds())
        self.assertEqual(read.occupationParity, -1)
        self.assertGreater(read.residualInBlockMotion, 1e-3)

    def test_gap_closure_returns_uncertified_not_a_sign(self):
        steps = _block_steps([0, 4], 8, self.STEPS, 4)
        broken = _block_steps([0, 4], 8, self.STEPS, 4)
        # one band on the loop lost its isolation (accepted=False)
        t = self.STEPS // 2
        x = 4 * t / self.STEPS
        broken[t][0] = _fiber(_mode(x % 8, 8), _ring_cells(8),
                              accepted=False)
        read = EH.blockPermutation(broken)
        self.assertFalse(read.certificate.holds())
        self.assertEqual(read.occupationParity, 0)
        self.assertEqual(read.blockParity, 0)
        self.assertEqual(list(read.blockPermutation), [])
        # the clean tracking, for contrast, certifies
        self.assertTrue(EH.blockPermutation(steps).certificate.holds())

    def test_block_teleport_is_uncertified(self):
        # Blocks jump 2 cells per step: no certified continuation.
        steps = []
        for t in range(4):
            x = 2 * t
            steps.append([_fiber(_mode((p + x) % 8, 8), _ring_cells(8))
                          for p in (0, 4)])
        read = EH.blockPermutation(steps)
        self.assertFalse(read.certificate.holds())
        self.assertEqual(read.occupationParity, 0)

    def test_block_count_change_is_uncertified(self):
        steps = _block_steps([0, 4], 8, 4, 0)
        steps[2] = steps[2][:1]
        read = EH.blockPermutation(steps)
        self.assertFalse(read.certificate.holds())

    def test_bad_composites_partition_throws(self):
        steps = _static_block_steps([0, 4], 8, 4)
        with self.assertRaises(ValueError):
            EH.blockPermutation(steps, composites=[[0]])       # not covering
        with self.assertRaises(ValueError):
            EH.blockPermutation(steps, composites=[[0, 1], [1]])  # overlap

    def test_empty_tracking_throws(self):
        with self.assertRaises(ValueError):
            EH.blockPermutation([])


# --------------------------------------------------------------------------- #
# invariance: relabeling, in-band rotation, reorientation (channel gauges)
# --------------------------------------------------------------------------- #
class InvarianceTest(unittest.TestCase):
    def _chi(self, frames, ref_frames, n=8):
        return EH.exchangeCharacter(_loop(frames, n=n),
                                    _loop(ref_frames, n=n))

    def test_character_invariant_under_in_band_rotation(self):
        rng = np.random.default_rng(42)
        frames = _exchange_frames()
        ref = _reference_frames(frames)
        base = self._chi(frames, ref)
        rotated = []
        for f in frames:
            g, _ = np.linalg.qr(rng.standard_normal((2, 2))
                                + 1j * rng.standard_normal((2, 2)))
            rotated.append(f @ g)
        chi = self._chi(rotated, ref)
        self.assertLess(abs(chi.character - base.character), 1e-9)
        self.assertEqual(chi.characterSign, base.characterSign)

    def test_raw_determinant_invariant_under_in_band_rotation(self):
        rng = np.random.default_rng(43)
        frames = _exchange_frames()
        base = _loop(frames)
        rotated = []
        for f in frames:
            g, _ = np.linalg.qr(rng.standard_normal((2, 2))
                                + 1j * rng.standard_normal((2, 2)))
            rotated.append(f @ g)
        self.assertLess(abs(_loop(rotated).determinant - base.determinant),
                        1e-9)

    def test_character_invariant_under_vertex_relabeling(self):
        rng = np.random.default_rng(44)
        perm = list(rng.permutation(8))
        frames = _exchange_frames()
        ref = _reference_frames(frames)
        base = self._chi(frames, ref)
        relabeled = EH.permutedCellFrames(frames, perm)
        relabeled_ref = EH.permutedCellFrames(ref, perm)
        # uniform weights: the permuted metric equals the original
        chi = self._chi(relabeled, relabeled_ref)
        self.assertLess(abs(chi.character - base.character), MACHINE)

    def test_character_invariant_under_simplex_reorientation(self):
        signs = [1, -1, 1, -1, -1, 1, 1, -1]
        frames = _exchange_frames()
        ref = _reference_frames(frames)
        base = self._chi(frames, ref)
        chi = self._chi(EH.reorientedFrames(frames, signs),
                        EH.reorientedFrames(ref, signs))
        self.assertLess(abs(chi.character - base.character), MACHINE)

    def test_combined_gauges_never_enter_the_characters(self):
        rng = np.random.default_rng(45)
        signs = [int(s) for s in rng.choice([-1, 1], size=8)]
        perm = list(rng.permutation(8))
        frames = _exchange_frames()
        ref = _reference_frames(frames)
        base = self._chi(frames, ref)
        gauged = EH.permutedCellFrames(EH.reorientedFrames(frames, signs),
                                       perm)
        gauged_ref = EH.permutedCellFrames(EH.reorientedFrames(ref, signs),
                                           perm)
        rotated = []
        for f in gauged:
            g, _ = np.linalg.qr(rng.standard_normal((2, 2))
                                + 1j * rng.standard_normal((2, 2)))
            rotated.append(f @ g)
        chi = self._chi(rotated, gauged_ref)
        self.assertLess(abs(chi.character - base.character), 1e-9)
        self.assertEqual(chi.characterSign, -1)

    def test_structural_parity_invariant_under_relabeling(self):
        steps = _block_steps([0, 4], 8, 16, 4)
        base = EH.blockPermutation(steps)
        # rename every vertex id v -> 100 + 7*v (order-scrambling injective
        # map applied to the fiber cell tuples; frames untouched)
        relabeled = []
        for row in steps:
            new_row = []
            for f in row:
                cells = [[100 + 7 * v for v in cell]
                         for cell in f.cellVertices()]
                new_row.append(_fiber(f.rightFrame(), cells,
                                      weights=np.asarray(f.weightDiagonal())))
            relabeled.append(new_row)
        read = EH.blockPermutation(relabeled)
        self.assertEqual(list(read.blockPermutation),
                         list(base.blockPermutation))
        self.assertEqual(read.occupationParity, base.occupationParity)
        self.assertEqual(read.blockParity, base.blockParity)

    def test_structural_parity_invariant_under_in_band_rotation(self):
        rng = np.random.default_rng(46)
        steps = _block_steps([0, 4], 8, 16, 4, ranks=[2, 2])
        ref = _static_block_steps([0, 4], 8, 16, ranks=[2, 2])
        base = EH.blockPermutation(steps, ref)
        rotated = []
        for row in steps:
            new_row = []
            for f in row:
                g, _ = np.linalg.qr(rng.standard_normal((2, 2))
                                    + 1j * rng.standard_normal((2, 2)))
                new_row.append(_fiber(np.asarray(f.rightFrame()) @ g,
                                      f.cellVertices(),
                                      weights=np.asarray(f.weightDiagonal())))
            rotated.append(new_row)
        read = EH.blockPermutation(rotated, ref)
        self.assertEqual(list(read.blockPermutation),
                         list(base.blockPermutation))
        self.assertEqual(read.occupationParity, base.occupationParity)
        self.assertLess(abs(read.residualInBlockMotion
                            - base.residualInBlockMotion), 1e-9)


# --------------------------------------------------------------------------- #
# the constructed total-space spin holonomy cycle (rotation channel)
# --------------------------------------------------------------------------- #
class RotationCycleTest(unittest.TestCase):
    STEPS = 12

    def test_gamma_clifford_relations(self):
        for d in (3, 4):
            for a in range(d):
                for b in range(d):
                    ga, gb = EH.gamma(a, d), EH.gamma(b, d)
                    anti = ga @ gb + gb @ ga
                    expected = 2.0 * np.eye(EH.spinorDimension(d)) * (a == b)
                    self.assertLess(np.abs(anti - expected).max(), MACHINE)

    def test_spin_generator_eigenvalues_are_half_i(self):
        ev = np.linalg.eigvals(EH.spinGenerator(0, 1, 4))
        self.assertLess(np.abs(np.sort(ev.imag)
                               - np.array([-0.5, -0.5, 0.5, 0.5])).max(),
                        MACHINE)
        self.assertLess(np.abs(ev.real).max(), MACHINE)

    def test_two_pi_spinor_rotation_is_minus_identity(self):
        for d in (3, 4):
            dim = EH.spinorDimension(d)
            u = EH.spinorRotation(2 * math.pi, 0, 1, d)
            self.assertLess(np.abs(u + np.eye(dim)).max(), MACHINE)

    def test_spin_half_two_pi_ratio_is_exactly_minus_one(self):
        frame0 = EH.transverseSpinorFrame(0, 1, 4)
        rot = EH.rotationLoopFrames(frame0, 0, 1, 4, 1, self.STEPS)
        ref = EH.referenceLoopFrames(frame0, self.STEPS)
        chi = EH.rotationCharacter(_loop(rot, n=4), _loop(ref, n=4))
        self.assertLess(abs(chi.character + 1.0), MACHINE)
        self.assertEqual(chi.characterSign, -1)
        self.assertEqual(chi.channel, obs.HolonomyChannel.PhysicalRotation)
        self.assertTrue(chi.certificate.holds())

    def test_spin_half_raw_determinant_carries_berry_phase(self):
        frame0 = EH.transverseSpinorFrame(0, 1, 4)
        rot = EH.rotationLoopFrames(frame0, 0, 1, 4, 1, self.STEPS)
        raw = _loop(rot, n=4).determinant
        expected = -cmath.exp(1j * OMEGA * self.STEPS)
        self.assertLess(abs(raw - expected), MACHINE)
        self.assertGreater(min(abs(raw - 1.0), abs(raw + 1.0)), 0.5)

    def test_four_pi_ratio_is_plus_one(self):
        frame0 = EH.transverseSpinorFrame(0, 1, 4)
        rot = EH.rotationLoopFrames(frame0, 0, 1, 4, 2, 2 * self.STEPS)
        ref = EH.referenceLoopFrames(frame0, 2 * self.STEPS)
        chi = EH.rotationCharacter(_loop(rot, n=4), _loop(ref, n=4))
        self.assertLess(abs(chi.character - 1.0), MACHINE)
        self.assertEqual(chi.characterSign, +1)

    def test_vector_two_pi_ratio_is_exactly_plus_one(self):
        frame0 = np.zeros((4, 1), complex)
        frame0[0, 0] = 1.0
        rot = EH.vectorLoopFrames(frame0, 0, 1, 4, 1, self.STEPS)
        ref = EH.referenceLoopFrames(frame0, self.STEPS)
        chi = EH.rotationCharacter(_loop(rot, n=4), _loop(ref, n=4))
        self.assertLess(abs(chi.character - 1.0), MACHINE)
        self.assertEqual(chi.characterSign, +1)
        self.assertTrue(chi.certificate.holds())

    def test_axis_polarized_spinor_shows_no_relative_phase(self):
        # A Sigma_01 EIGENvector is stationary under the (0,1) rotation:
        # the loop is pure gauge and the ratio is +1 (the documented
        # transverse-frame requirement, demonstrated).
        sigma = np.asarray(EH.spinGenerator(0, 1, 4))
        w, v = np.linalg.eigh(1j * sigma)
        frame0 = v[:, [0]]
        rot = EH.rotationLoopFrames(frame0, 0, 1, 4, 1, self.STEPS)
        ref = EH.referenceLoopFrames(frame0, self.STEPS)
        chi = EH.rotationCharacter(_loop(rot, n=4), _loop(ref, n=4))
        self.assertLess(abs(chi.character - 1.0), MACHINE)

    def test_rotation_acts_on_the_whole_carried_frame(self):
        # TOTAL-SPACE action: a rank-2 transverse frame (both doublets)
        # rotates as one object; each transverse column contributes -1, so
        # the even-rank determinant ratio is +1 — the frame is never read
        # as per-column Bloch products.
        sigma = np.asarray(EH.spinGenerator(0, 1, 4))
        w, v = np.linalg.eigh(1j * sigma)
        # eigenvalues sorted ascending: two -1/2 then two +1/2
        t1 = (v[:, [0]] + v[:, [2]]) / math.sqrt(2)
        t2 = (v[:, [1]] + v[:, [3]]) / math.sqrt(2)
        frame0 = np.hstack([t1, t2])
        rot = EH.rotationLoopFrames(frame0, 0, 1, 4, 1, self.STEPS)
        ref = EH.referenceLoopFrames(frame0, self.STEPS)
        chi = EH.rotationCharacter(_loop(rot, n=4), _loop(ref, n=4))
        self.assertLess(abs(chi.character - 1.0), MACHINE)

    def test_rotation_loop_input_validation(self):
        frame0 = EH.transverseSpinorFrame(0, 1, 4)
        with self.assertRaises(ValueError):
            EH.rotationLoopFrames(frame0, 0, 1, 4, 1, 2)   # < 3 steps
        with self.assertRaises(ValueError):
            EH.rotationLoopFrames(np.ones((3, 1), complex), 0, 1, 4, 1, 8)
        with self.assertRaises(ValueError):
            EH.spinorRotation(1.0, 2, 2, 4)                # equal axes
        with self.assertRaises(ValueError):
            EH.gamma(0, 5)                                 # unsupported d


# --------------------------------------------------------------------------- #
# the total-space J^2 measuring stick (existing oracles stay exact)
# --------------------------------------------------------------------------- #
class TotalJSquaredTest(unittest.TestCase):
    def test_proton_eigenstate_is_exactly_three_quarters(self):
        proton = (2 * _kron(UP, UP, DN) - _kron(UP, DN, UP)
                  - _kron(DN, UP, UP))
        self.assertLess(abs(EH.totalJSquared(proton) - 0.75), 1e-14)

    def test_delta_is_exactly_fifteen_quarters(self):
        self.assertLess(abs(EH.totalJSquared(_kron(UP, UP, UP)) - 3.75),
                        1e-14)

    def test_product_uud_is_seven_quarters(self):
        self.assertLess(abs(EH.totalJSquared(_kron(UP, UP, DN)) - 1.75),
                        1e-14)

    def test_single_spin_is_three_quarters(self):
        self.assertLess(abs(EH.totalJSquared(UP) - 0.75), 1e-14)

    def test_two_spin_singlet_and_triplet(self):
        singlet = (_kron(UP, DN) - _kron(DN, UP)) / math.sqrt(2)
        self.assertLess(abs(EH.totalJSquared(singlet)), 1e-14)
        self.assertLess(abs(EH.totalJSquared(_kron(UP, UP)) - 2.0), 1e-14)

    def test_operator_matches_expectation_read(self):
        rng = np.random.default_rng(9)
        state = rng.standard_normal(8) + 1j * rng.standard_normal(8)
        j2 = np.asarray(EH.totalJSquaredOperator(3))
        expected = float(np.real(state.conj() @ j2 @ state
                                 / (state.conj() @ state)))
        self.assertLess(abs(EH.totalJSquared(state) - expected), 1e-12)
        self.assertLess(np.abs(j2 - j2.conj().T).max(), MACHINE)

    def test_input_validation(self):
        with self.assertRaises(ValueError):
            EH.totalJSquared(np.ones(6, complex))          # not 2^n
        with self.assertRaises(ValueError):
            EH.totalJSquared(np.zeros(8, complex))         # zero state
        with self.assertRaises(ValueError):
            EH.totalJSquaredOperator(0)
        with self.assertRaises(ValueError):
            EH.totalJSquaredOperator(11)


# --------------------------------------------------------------------------- #
# the doubly cancelled spin-statistics ratio
# --------------------------------------------------------------------------- #
class SpinStatisticsTest(unittest.TestCase):
    def _spin_half_pair(self):
        frames = _exchange_frames()
        exchange = EH.exchangeCharacter(_loop(frames),
                                        _loop(_reference_frames(frames)))
        frame0 = EH.transverseSpinorFrame(0, 1, 4)
        rot = EH.rotationLoopFrames(frame0, 0, 1, 4, 1, 12)
        ref = EH.referenceLoopFrames(frame0, 12)
        rotation = EH.rotationCharacter(_loop(rot, n=4), _loop(ref, n=4))
        return exchange, rotation

    def test_doubly_cancelled_ratio_is_plus_one_on_spin_half(self):
        exchange, rotation = self._spin_half_pair()
        self.assertLess(abs(exchange.character + 1.0), MACHINE)
        self.assertLess(abs(rotation.character + 1.0), MACHINE)
        ratio = EH.doublyCancelledRatio(exchange, rotation)
        self.assertLess(abs(ratio - 1.0), MACHINE)

    def test_channel_tags_are_enforced(self):
        exchange, rotation = self._spin_half_pair()
        with self.assertRaises(ValueError):
            EH.doublyCancelledRatio(rotation, exchange)
        with self.assertRaises(ValueError):
            EH.doublyCancelledRatio(exchange, exchange)


# --------------------------------------------------------------------------- #
# fiber-composed transport (composing #769; gap closure semantics)
# --------------------------------------------------------------------------- #
class FiberLoopTest(unittest.TestCase):
    def _fiber_loop(self, steps=16, accepted_mask=None, offset=0):
        frames = _exchange_frames(steps)
        accepted_mask = [True] * steps if accepted_mask is None else \
            accepted_mask
        return [_fiber(frames[t], _ring_cells(8, offset),
                       accepted=accepted_mask[t]) for t in range(steps)]

    def test_fiber_loop_matches_plain_frames(self):
        steps = 16
        frames = _exchange_frames(steps)
        plain = _loop(frames)
        fibers = self._fiber_loop(steps)
        read = EH.fiberLoopHolonomy(fibers)
        self.assertLess(np.abs(np.asarray(read.holonomy)
                               - np.asarray(plain.holonomy)).max(), MACHINE)
        self.assertTrue(read.certificate.holds())
        self.assertFalse(read.uncertifiedBand)

    def test_closed_gap_returns_uncertified_not_a_sign(self):
        mask = [True] * 16
        mask[7] = False
        loop = self._fiber_loop(16, accepted_mask=mask)
        read = EH.fiberLoopHolonomy(loop)
        self.assertTrue(read.uncertifiedBand)
        self.assertFalse(read.certificate.holds())
        ref = [_fiber(_exchange_frames(16)[0], _ring_cells(8))] * 16
        chi = EH.exchangeCharacter(read, EH.fiberLoopHolonomy(ref))
        self.assertEqual(chi.characterSign, 0)
        self.assertFalse(chi.certificate.holds())

    def test_rank_change_is_uncertified_never_thrown(self):
        loop = self._fiber_loop(8)
        frames = _exchange_frames(8)
        loop[3] = _fiber(frames[3][:, :1], _ring_cells(8))
        read = EH.fiberLoopHolonomy(loop)
        self.assertFalse(read.certificate.holds())
        self.assertEqual(np.asarray(read.holonomy).size, 0)

    def test_disjoint_supports_leak_is_uncertified(self):
        a = _fiber(_mode(0, 4), [[0], [1], [2], [3]])
        b = _fiber(_mode(0, 4), [[10], [11], [12], [13]])
        read = EH.fiberLoopHolonomy([a, b])
        self.assertFalse(read.certificate.holds())
        self.assertEqual(read.minStepSingularValue, 0.0)

    def test_fiber_loop_invariant_under_cell_relabeling(self):
        base = EH.fiberLoopHolonomy(self._fiber_loop(16))
        relabeled = EH.fiberLoopHolonomy(self._fiber_loop(16, offset=500))
        self.assertLess(abs(relabeled.determinant - base.determinant),
                        MACHINE)

    def test_shared_cell_restriction_composes_overlapping_windows(self):
        # Consecutive fibers see overlapping cell WINDOWS of the same
        # underlying mode: transport composes on the shared cells only.
        n = 8
        mode = _mode(2.5, n)
        full = _fiber(mode, _ring_cells(n))
        window = _fiber(mode[1:6][:, None] /
                        np.linalg.norm(mode[1:6]),
                        [[c] for c in range(1, 6)])
        read = EH.fiberLoopHolonomy([full, window])
        self.assertEqual(read.rank, 1)
        self.assertTrue(all(s.certified for s in read.stepReads))


# --------------------------------------------------------------------------- #
# #766 delegation regressions (wedge norm, graded swap)
# --------------------------------------------------------------------------- #
class GradedDelegationTest(unittest.TestCase):
    def test_duplicate_complete_modes_have_zero_wedge_norm(self):
        alg = quantum.ExteriorAlgebra(3)
        v = np.array([0.3 + 0.1j, -0.7, 0.64j], complex)
        w = np.array([1.0, 0.5j, -0.25], complex)
        wedge = np.asarray(alg.wedge([v, v]))
        self.assertEqual(float(np.linalg.norm(wedge)), 0.0)
        # nonzero for distinct modes, with the Gram-determinant norm
        wedge2 = np.asarray(alg.wedge([v, w]))
        gram = np.array([[v.conj() @ v, v.conj() @ w],
                         [w.conj() @ v, w.conj() @ w]])
        self.assertLess(abs(np.linalg.norm(wedge2) ** 2
                            - np.linalg.det(gram).real), 1e-12)

    def test_graded_swap_signs_match_the_parity_table(self):
        fock = quantum.FockDirectSum(1, 1)
        rows, cols, vals, n = fock.gradedSwapMatrixCOO()
        swap = np.zeros((n, n), complex)
        for r, c, v in zip(rows, cols, vals):
            swap[r, c] += v
        # |1> x |1> (odd/odd) picks up -1; every other elementary parity
        # combination is +1 (odd/even = +1 — the acceptance table).
        odd_odd = np.zeros(4, complex)
        odd_odd[3] = 1.0   # both modes occupied
        self.assertLess(abs((swap @ odd_odd)[3] + 1.0), MACHINE)
        even_odd = np.zeros(4, complex)
        even_odd[1] = 1.0  # left occupied only
        self.assertLess(abs(np.abs(swap @ even_odd).max() - 1.0), MACHINE)
        self.assertLess(abs((swap @ even_odd)[2] - 1.0), MACHINE)

    def test_mode_parity_matches_numpy_determinant_sign(self):
        rng = np.random.default_rng(11)
        for _ in range(5):
            perm = list(rng.permutation(6))
            matrix = np.zeros((6, 6))
            for i, p in enumerate(perm):
                matrix[p, i] = 1.0
            det = round(float(np.linalg.det(matrix)))
            bits = quantum.OccupationBitset.fromOccupiedModes(6,
                                                              list(range(6)))
            self.assertEqual(bits.permutationParity(perm), det)


# --------------------------------------------------------------------------- #
# SO(d) -> Spin(d) lift machinery
# --------------------------------------------------------------------------- #
class SpinLiftTest(unittest.TestCase):
    def test_rotation_log_roundtrip_random(self):
        from scipy.linalg import expm
        for d, seed in ((3, 1), (3, 2), (4, 3), (4, 4), (5, 5)):
            r = _random_rotation(d, seed)
            a = np.asarray(EH.rotationLog(r))
            self.assertLess(np.abs(a + a.T).max(), 1e-9)   # antisymmetric
            self.assertLess(np.abs(expm(a) - r).max(), 1e-9)

    def test_rotation_log_handles_angle_pi(self):
        from scipy.linalg import expm
        r = np.diag([1.0, -1.0, -1.0])
        a = np.asarray(EH.rotationLog(r))
        self.assertLess(np.abs(expm(a) - r).max(), 1e-9)

    def test_rotation_log_rejects_non_rotation(self):
        with self.assertRaises(ValueError):
            EH.rotationLog(np.diag([1.0, 1.0, 2.0]))
        with self.assertRaises(ValueError):
            EH.rotationLog(np.diag([1.0, 1.0, -1.0]))  # det = -1

    def test_lift_has_half_angle_eigenphases(self):
        theta = 0.9
        for d in (3, 4):
            s = np.asarray(EH.rotationToSpin(_rz(theta, d), d))
            phases = np.sort(np.angle(np.linalg.eigvals(s)))
            self.assertLess(abs(abs(phases).max() - theta / 2), 1e-9)

    def test_lift_conjugates_gammas_by_the_rotation(self):
        # The defining covering-homomorphism property:
        # S gamma(x) S^{-1} = gamma(R x).
        for d, seed in ((3, 21), (4, 22)):
            r = _random_rotation(d, seed)
            s = np.asarray(EH.rotationToSpin(r, d))
            rng = np.random.default_rng(seed + 100)
            x = rng.standard_normal(d)
            gx = sum(x[a] * np.asarray(EH.gamma(a, d)) for a in range(d))
            grx = sum((r @ x)[a] * np.asarray(EH.gamma(a, d))
                      for a in range(d))
            self.assertLess(np.abs(s @ gx @ s.conj().T - grx).max(), 1e-9)

    def test_lift_orientation_matches_the_documented_cycle(self):
        # The documented coherence relation between the two conventions:
        # rotationToSpin(R_ab(theta)) = spinorRotation(-theta, a, b, d).
        theta = 0.63
        for d in (3, 4):
            lifted = np.asarray(EH.rotationToSpin(_rz(theta, d), d))
            cycle = np.asarray(EH.spinorRotation(-theta, 0, 1, d))
            self.assertLess(np.abs(lifted - cycle).max(), 1e-9)

    def test_lift_is_a_projective_homomorphism(self):
        r1 = _random_rotation(3, 31)
        r2 = _random_rotation(3, 32)
        s12 = np.asarray(EH.rotationToSpin(r1 @ r2, 3))
        prod = np.asarray(EH.rotationToSpin(r1, 3)) @ np.asarray(
            EH.rotationToSpin(r2, 3))
        delta = min(np.abs(s12 - prod).max(), np.abs(s12 + prod).max())
        self.assertLess(delta, 1e-9)

    def test_loop_lift_two_pi_is_minus_one(self):
        for d in (3, 4):
            loop = [_rz(2 * math.pi * t / 12, d) for t in range(12)]
            read = EH.loopLiftCharacter(loop, d)
            self.assertEqual(read.character, -1)
            self.assertTrue(read.certificate.holds())
            self.assertLess(read.closureResidual, 1e-9)

    def test_loop_lift_four_pi_is_plus_one(self):
        loop = [_rz(4 * math.pi * t / 24) for t in range(24)]
        read = EH.loopLiftCharacter(loop, 3)
        self.assertEqual(read.character, +1)

    def test_loop_lift_contractible_wiggle_is_plus_one(self):
        loop = [_rz(0.4 * math.sin(2 * math.pi * t / 10)) for t in range(10)]
        read = EH.loopLiftCharacter(loop, 3)
        self.assertEqual(read.character, +1)
        self.assertTrue(read.certificate.holds())

    def test_loop_lift_pi_step_is_uncertified(self):
        loop = [np.eye(3), _rz(math.pi), np.eye(3), _rz(math.pi)]
        read = EH.loopLiftCharacter(loop, 3)
        self.assertEqual(read.character, 0)
        self.assertFalse(read.certificate.holds())
        self.assertGreater(read.maxStepAngle, math.pi - 1e-9)

    def _tetra_edges(self):
        return [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

    def _tetra_triangles(self):
        return [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]

    def test_spin_fixture_accepts_vertex_frame_data(self):
        frames = {v: _random_rotation(3, 50 + v) for v in range(4)}
        edges = self._tetra_edges()
        rotations = [frames[i] @ frames[j].T for i, j in edges]
        read = EH.spinLift(edges, rotations, self._tetra_triangles(), 3)
        self.assertTrue(read.liftExists)
        self.assertFalse(read.obstructed)
        self.assertTrue(read.certificate.holds())
        self.assertLess(read.maxCocycleResidual, 1e-9)
        self.assertEqual(len(read.edgeSigns), 6)
        # the returned signs satisfy w_t * prod_{e in t} s_e = +1
        for tri, w in zip(self._tetra_triangles(), read.triangleSigns):
            prod = 1
            for c in range(3):
                key = tuple(sorted((tri[c], tri[(c + 1) % 3])))
                prod *= read.edgeSigns[edges.index(key)]
            self.assertEqual(w * prod, +1)

    def test_spin_fixture_accepts_quaternion_tetrahedron(self):
        # i j k = -1 on (0,1,2), completed to a global SO(3) cocycle on the
        # tetrahedron: the total class is trivial (S^2 data from a genuine
        # bundle), so the lift exists even though triangle signs are
        # nontrivial.
        edges = [(0, 1), (1, 2), (2, 0), (3, 0), (1, 3), (2, 3)]
        rotations = [_rx(math.pi), _ry(math.pi), _rz(math.pi),
                     np.eye(3), _rx(math.pi), _rz(math.pi)]
        triangles = [[0, 1, 2], [0, 1, 3], [1, 2, 3], [0, 2, 3]]
        read = EH.spinLift(edges, rotations, triangles, 3)
        self.assertTrue(read.certificate.holds())
        self.assertLess(read.maxCocycleResidual, 1e-9)
        self.assertTrue(read.liftExists)
        self.assertEqual(sum(1 for w in read.triangleSigns if w < 0) % 2, 0)

    def test_non_spin_fixture_rejects_the_pillowcase_class(self):
        # The minimal closed surface with the nontrivial SO(3) bundle: two
        # triangles glued along all three pi-rotation edges.  The two
        # traversal orders i j k vs k j i differ by the center, so exactly
        # one triangle sign is -1: w2 evaluates 1 on the fundamental class
        # and the lift is REJECTED.
        edges = [(0, 1), (1, 2), (2, 0)]
        rotations = [_rx(math.pi), _ry(math.pi), _rz(math.pi)]
        triangles = [[0, 1, 2], [0, 2, 1]]
        read = EH.spinLift(edges, rotations, triangles, 3)
        self.assertTrue(read.certificate.holds())
        self.assertFalse(read.liftExists)
        self.assertTrue(read.obstructed)
        self.assertEqual(len(read.edgeSigns), 0)
        self.assertEqual(sum(1 for w in read.triangleSigns if w < 0) % 2, 1)

    def test_obstruction_class_survives_vertex_relabeling(self):
        edges = [(7, 5), (5, 9), (9, 7)]
        rotations = [_rx(math.pi), _ry(math.pi), _rz(math.pi)]
        triangles = [[7, 5, 9], [7, 9, 5]]
        read = EH.spinLift(edges, rotations, triangles, 3)
        self.assertTrue(read.obstructed)

    def test_broken_cocycle_is_uncertified_neither_verdict(self):
        edges = [(0, 1), (1, 2), (2, 0)]
        rotations = [_rz(0.3), _rz(0.4), _rz(0.2)]  # product != I
        read = EH.spinLift(edges, rotations, [[0, 1, 2]], 3)
        self.assertFalse(read.certificate.holds())
        self.assertFalse(read.liftExists)
        self.assertFalse(read.obstructed)

    def test_spin_lift_input_validation(self):
        with self.assertRaises(ValueError):
            EH.spinLift([(0, 1)], [], [[0, 1, 2]], 3)      # count mismatch
        with self.assertRaises(ValueError):
            EH.spinLift([(0, 1)], [_rz(0.1)], [[0, 1, 2]], 3)  # missing edge
        with self.assertRaises(ValueError):
            EH.spinLift([(0, 1), (1, 2), (2, 0)],
                        [_rz(0.0)] * 3, [[0, 1]], 3)       # not a triangle
        with self.assertRaises(ValueError):
            EH.spinLift([(0, 0)], [_rz(0.1)], [], 3)       # self-loop


if __name__ == "__main__":
    unittest.main()
