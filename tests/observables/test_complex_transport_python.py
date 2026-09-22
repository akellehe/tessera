# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""Exact fixtures for the complex fibre transport (issue #1199):
:class:`tessera.observables.ComplexTransport`.

Acceptance coverage (ticket #1199):

* the fibre map M_AB = Phi~_A^T T_AB Phi_B is retained as an element of
  GL(r, C): it equals the bilinear pairing computed independently with
  numpy, it is never a polar factor, and a transport whose determinant has
  modulus far from one is still accepted and reported with that modulus;
* the leakage certificate is the whitepaper's quantity
  ||(I - P_A) T_AB P_B||_2, measured on the transfer before the restriction
  to the bands, and it is zero exactly when the transfer carries the source
  band into the destination band;
* transport composes and reverses exactly: compose is the ordered product,
  a closed holonomy's power traces, characteristic polynomial and
  determinant are unchanged by a frame change at the base point, the
  reversal is the transpose rather than the conjugate adjoint, and the dual
  transport M^-T has determinant (det M)^-1 and is an involution;
* the Kato equation transports an isolated band: the direct rotation
  intertwines two orthogonal projectors exactly and is unitary, the
  unnormalized intertwiner does the same for oblique idempotents, the
  exponential step's intertwining residual is third order in the step and
  is reported rather than hidden, and a path that is sampled too coarsely
  is refused with a reason instead of being transported approximately;
* the exchange character chi_F = det(H_ex H_ref^-1) of a known exchange is
  reproduced from the unprojected general-linear holonomies -- exactly -1
  for a single exchange of two odd-rank bands and +1 for a double one --
  with its modulus reported and never required, so that a character of
  modulus other than one is certified whenever the paths did not leak.

Independent references: every expected value is built with numpy and scipy
from the definitions (an explicitly assembled projector, a matrix
exponential, numpy's determinant and its two-norm), never by calling a
second method of the kernel under test.

Exactness bar: structural identities (a transpose, an ordered product, a
determinant of a permutation block) are compared at double round-off
(~1e-14). Values that pass through a matrix inverse, a matrix square root
or a singular value decomposition are compared at 1e-11, which is the
honest tolerance of those paths on the well-conditioned fixtures below.
"""

from __future__ import annotations

import unittest

import numpy as np
from scipy.linalg import expm

import tessera

obs = tessera.observables

CT = obs.ComplexTransport
KatoScheme = obs.KatoScheme

ROUNDOFF = 1e-14
INVERSE = 1e-11


def rng():
    """The one seeded generator every random fixture below draws from."""
    return np.random.default_rng(20261199)


def random_complex(generator, *shape):
    return generator.normal(size=shape) + 1j * generator.normal(size=shape)


def invertible(generator, size, cap=50.0):
    """A well-conditioned random element of GL(size, C)."""
    while True:
        candidate = random_complex(generator, size, size)
        if np.linalg.cond(candidate) < cap:
            return candidate


def orthogonal_projector(generator, dimension, rank):
    """A Hermitian rank-`rank` projector on C^dimension."""
    columns, _ = np.linalg.qr(random_complex(generator, dimension, rank))
    return columns @ columns.conj().T


def oblique_idempotent(generator, dimension, rank):
    """A non-Hermitian idempotent: a similarity transform of a coordinate
    projector, which is what a Riesz projector of a non-normal operator is."""
    similarity = invertible(generator, dimension)
    diagonal = np.diag([1.0] * rank + [0.0] * (dimension - rank)).astype(complex)
    return similarity @ diagonal @ np.linalg.inv(similarity)


def two_norm(matrix):
    return float(np.linalg.norm(np.asarray(matrix), 2))


# ─── synthetic SpectralFiber construction (through the public record) ──────

_UNMEASURED = float("nan")


def fiber(frame, cells, weights=None, accepted=True, degree=0,
          eigenvalues=None, isolation=1.0):
    """A synthetic SpectralFiber over explicit cell tuples, built through the
    public record schema, which is the documented replay path.

    The left frame is the biorthogonal one, Psi^dagger W Phi = I, so that the
    fibre's transpose dual satisfies Phi~^T Phi = I and the fibre map below is
    the whitepaper's bilinear pairing.
    """
    frame = np.asarray(frame, complex)
    if frame.ndim == 1:
        frame = frame[:, None]
    rows, rank = frame.shape
    if weights is None:
        weights = np.ones(rows, complex)
    weights = np.asarray(weights, complex)
    eigenvalues = [0.0] * rank if eigenvalues is None else list(eigenvalues)
    gram = frame.conj().T @ np.diag(weights) @ frame
    left = np.linalg.solve(
        np.diag(weights).conj(),
        np.linalg.solve(gram, frame.conj().T @ np.diag(weights)).conj().T)

    def flat(matrix):
        return [complex(matrix[i, j])
                for i in range(matrix.shape[0])
                for j in range(matrix.shape[1])]

    inner = {
        "grade": "certified-numerical" if accepted else "heuristic-discovery",
        "domain": "band-window",
        "regime": "non-normal",
        "residual": 0.0 if accepted else _UNMEASURED,
        "conditioning": 1.0,
        "dense_reference_error": _UNMEASURED,
        "tolerance": 1e-9,
    }
    certificate = {
        "degree": degree, "rank": rank,
        "lower_gap": isolation, "upper_gap": isolation,
        "nearest_discarded_separation": isolation,
        "localization": 1.0 / rows,
        "projector_residual": 0.0, "eigen_residual": 1e-13,
        "left_residual": 2e-13, "gram_defect": 0.0,
        "projector_norm": 1.0,
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


def cells(count, offset=0):
    return [[offset + c] for c in range(count)]


def band_pair(generator, dimension=6, rank=2):
    """Two bands on one cell space whose ranges are orthogonal to each other,
    so that a transfer can be built with an exactly known escaping part.

    Returns the destination band, the source band and their two frames, which
    have orthonormal columns; a transfer Phi_A G Phi_B^dagger then has fibre
    map exactly G and leaks exactly nothing.
    """
    columns, _ = np.linalg.qr(random_complex(generator, dimension, 2 * rank))
    frame_from = columns[:, :rank]
    frame_to = columns[:, rank:]
    return (fiber(frame_to, cells(dimension)),
            fiber(frame_from, cells(dimension)), frame_to, frame_from)


def hermitian_direction(generator, dimension):
    """A Hermitian matrix of unit two-norm, so that a parameter step of size s
    moves a projector by an amount bounded by a small multiple of s."""
    matrix = random_complex(generator, dimension, dimension)
    matrix = matrix + matrix.conj().T
    return matrix / np.linalg.norm(matrix, 2)


# ─── the fibre map and its frame law ───────────────────────────────────────

class TestFiberMap(unittest.TestCase):
    """M_AB = Phi~_A^T T_AB Phi_B: the bilinear pairing, kept in GL(r, C)."""

    def test_fiber_map_is_the_bilinear_pairing(self) -> None:
        generator = rng()
        dual = random_complex(generator, 5, 2)
        transfer = random_complex(generator, 5, 4)
        right = random_complex(generator, 4, 2)
        expected = dual.T @ transfer @ right
        measured = np.asarray(CT.fiberMap(dual, transfer, right))
        self.assertLess(np.abs(measured - expected).max(), ROUNDOFF)

    def test_fiber_map_never_conjugates_the_dual_frame(self) -> None:
        """The pairing is bilinear, so a purely imaginary dual frame does NOT
        pick up the sign a sesquilinear pairing would give it."""
        dual = 1j * np.eye(2, dtype=complex)
        transfer = np.eye(2, dtype=complex)
        right = np.eye(2, dtype=complex)
        measured = np.asarray(CT.fiberMap(dual, transfer, right))
        self.assertLess(np.abs(measured - 1j * np.eye(2)).max(), ROUNDOFF)

    def test_shape_mismatch_is_refused(self) -> None:
        generator = rng()
        with self.assertRaises(ValueError):
            CT.fiberMap(random_complex(generator, 5, 2),
                        random_complex(generator, 4, 4),
                        random_complex(generator, 4, 2))
        with self.assertRaises(ValueError):
            CT.fiberMap(random_complex(generator, 5, 2),
                        random_complex(generator, 5, 4),
                        random_complex(generator, 3, 2))

    def test_frame_law_is_the_bifundamental_one(self) -> None:
        generator = rng()
        transport = invertible(generator, 3)
        frame_to = invertible(generator, 3)
        frame_from = invertible(generator, 3)
        expected = np.linalg.inv(frame_to) @ transport @ frame_from
        measured = np.asarray(
            CT.frameChanged(transport, frame_to, frame_from))
        self.assertLess(np.abs(measured - expected).max(), INVERSE)


# ─── the leakage certificate ───────────────────────────────────────────────

class TestLeakage(unittest.TestCase):
    """leak_AB = ||(I - P_A) T_AB P_B||_2, measured before the restriction."""

    def test_leakage_matches_the_definition(self) -> None:
        generator = rng()
        projector_to = orthogonal_projector(generator, 6, 2)
        projector_from = orthogonal_projector(generator, 6, 2)
        transfer = random_complex(generator, 6, 6)
        expected = two_norm(
            (np.eye(6) - projector_to) @ transfer @ projector_from)
        measured = CT.leakage(projector_to, transfer, projector_from)
        self.assertLess(abs(measured - expected), INVERSE)

    def test_a_transfer_into_the_destination_band_does_not_leak(self) -> None:
        generator = rng()
        projector_to = orthogonal_projector(generator, 6, 2)
        projector_from = orthogonal_projector(generator, 6, 2)
        # Post-composing with P_A sends everything into the destination band.
        transfer = projector_to @ random_complex(generator, 6, 6)
        self.assertLess(CT.leakage(projector_to, transfer, projector_from),
                        INVERSE)

    def test_an_orthogonal_jump_leaks_the_whole_transfer(self) -> None:
        """A transfer whose image is orthogonal to the destination band leaks
        its entire norm: nothing survives the restriction."""
        projector_to = np.diag([1.0, 1.0, 0.0, 0.0]).astype(complex)
        projector_from = np.diag([0.0, 0.0, 1.0, 1.0]).astype(complex)
        transfer = np.zeros((4, 4), complex)
        transfer[2, 2] = 3.0          # source band -> outside the destination
        self.assertLess(
            abs(CT.leakage(projector_to, transfer, projector_from) - 3.0),
            INVERSE)

    def test_shape_mismatch_is_refused(self) -> None:
        generator = rng()
        with self.assertRaises(ValueError):
            CT.leakage(orthogonal_projector(generator, 6, 2),
                       random_complex(generator, 5, 6),
                       orthogonal_projector(generator, 6, 2))


# ─── the complete transport read ───────────────────────────────────────────

class TestTransportRead(unittest.TestCase):
    """The complete transport A <- B: unprojected, certified on the leakage,
    and reported with everything the whitepaper asks for beside it."""

    # The declared fibre map of the fixtures below: a transfer
    # Phi_A G Phi_B^dagger has fibre map exactly G, and G is chosen far from
    # any unitary so that the absence of a polar reduction is visible.
    G = np.array([[3.0, 1.0], [0.0, 2.0]], complex)

    def test_the_map_is_the_unprojected_pairing(self) -> None:
        generator = rng()
        destination, source, frame_to, frame_from = band_pair(generator)
        transfer = frame_to @ self.G @ frame_from.conj().T
        read = CT.transport(destination, source, transfer)
        self.assertLess(np.abs(np.asarray(read.map) - self.G).max(), INVERSE)
        self.assertTrue(read.accepted)
        self.assertTrue(read.certificate.holds())
        # Unprojected: the map is not a unitary and no polar factor is taken.
        product = np.asarray(read.map).conj().T @ np.asarray(read.map)
        self.assertGreater(np.abs(product - np.eye(2)).max(), 1.0)

    def test_a_determinant_of_large_modulus_is_reported_not_corrected(self) -> None:
        generator = rng()
        destination, source, frame_to, frame_from = band_pair(generator)
        transfer = frame_to @ self.G @ frame_from.conj().T
        read = CT.transport(destination, source, transfer)
        # det G = 3 * 2 = 6, reported as it is rather than driven to modulus
        # one by a polar factor or a determinant root.
        self.assertLess(abs(read.determinant - 6.0), INVERSE)
        self.assertTrue(read.certificate.holds())

    def test_a_leaking_transfer_is_rejected_with_a_reason(self) -> None:
        generator = rng()
        destination, source, _, _ = band_pair(generator)
        transfer = random_complex(generator, 6, 6)
        read = CT.transport(destination, source, transfer)
        self.assertFalse(read.accepted)
        self.assertIn("leak", read.rejectionReason)
        self.assertFalse(read.certificate.holds())
        # The numbers survive the rejection.
        self.assertGreater(read.leakage, 0.0)
        self.assertEqual(np.asarray(read.map).shape, (2, 2))

    def test_the_reported_leakage_is_the_whitepaper_quantity(self) -> None:
        generator = rng()
        destination, source, _, _ = band_pair(generator)
        transfer = random_complex(generator, 6, 6)
        read = CT.transport(destination, source, transfer)
        expected = two_norm((np.eye(6) - np.asarray(destination.projector()))
                            @ transfer @ np.asarray(source.projector()))
        self.assertLess(abs(read.leakage - expected), INVERSE)
        scale = two_norm(transfer @ np.asarray(source.projector()))
        self.assertLess(abs(read.relativeLeakage - expected / scale), INVERSE)

    def test_a_transfer_into_the_destination_band_is_accepted(self) -> None:
        generator = rng()
        destination, source, frame_to, frame_from = band_pair(generator)
        transfer = frame_to @ self.G @ frame_from.conj().T
        read = CT.transport(destination, source, transfer)
        self.assertLess(read.leakage, INVERSE)
        self.assertLess(read.relativeLeakage, INVERSE)
        self.assertTrue(read.invertible)
        self.assertEqual(read.numericalRank, 2)

    def test_the_endpoint_certificates_are_carried_through(self) -> None:
        generator = rng()
        destination, source, frame_to, frame_from = band_pair(generator)
        read = CT.transport(destination, source,
                            frame_to @ self.G @ frame_from.conj().T)
        # The fixture declares isolation 1 and projector norm 1 on both ends.
        self.assertLess(abs(read.toIsolation - 1.0), ROUNDOFF)
        self.assertLess(abs(read.toResolventBound - 1.0), ROUNDOFF)
        self.assertLess(abs(read.fromResolventBound - 1.0), ROUNDOFF)
        self.assertLess(abs(read.toRightFrameResidual - 1e-13), 1e-18)
        self.assertLess(abs(read.toLeftFrameResidual - 2e-13), 1e-18)
        self.assertLess(abs(read.fromRightFrameResidual - 1e-13), 1e-18)
        self.assertLess(abs(read.fromLeftFrameResidual - 2e-13), 1e-18)

    def test_an_uncertified_endpoint_band_is_refused(self) -> None:
        generator = rng()
        columns, _ = np.linalg.qr(random_complex(generator, 6, 4))
        source = fiber(columns[:, :2], cells(6), accepted=False)
        destination = fiber(columns[:, 2:], cells(6))
        projector_to = np.asarray(destination.projector())
        read = CT.transport(destination, source,
                            projector_to @ random_complex(generator, 6, 6))
        self.assertFalse(read.accepted)
        self.assertIn("uncertified", read.rejectionReason)

    def test_a_rank_mismatch_is_refused(self) -> None:
        generator = rng()
        columns, _ = np.linalg.qr(random_complex(generator, 6, 3))
        source = fiber(columns[:, :1], cells(6))
        destination = fiber(columns[:, 1:], cells(6))
        read = CT.transport(destination, source,
                            random_complex(generator, 6, 6))
        self.assertFalse(read.accepted)
        self.assertIn("rank", read.rejectionReason)

    def test_a_transfer_of_the_wrong_shape_throws(self) -> None:
        generator = rng()
        destination, source, _, _ = band_pair(generator)
        with self.assertRaises(ValueError):
            CT.transport(destination, source, random_complex(generator, 5, 6))

    def test_describe_names_the_rank_and_the_verdict(self) -> None:
        generator = rng()
        destination, source, _, _ = band_pair(generator)
        read = CT.transport(destination, source,
                            random_complex(generator, 6, 6))
        summary = read.describe()
        self.assertIn("GL(2, C)", summary)
        self.assertIn("rejected", summary)


# ─── composition, reversal and the holonomy ────────────────────────────────

class TestCompositionAndReversal(unittest.TestCase):
    """Transport composes and reverses exactly on a fixture."""

    def test_compose_is_the_ordered_product_first_link_first(self) -> None:
        generator = rng()
        links = [invertible(generator, 3) for _ in range(4)]
        expected = links[3] @ links[2] @ links[1] @ links[0]
        measured = np.asarray(CT.compose(links))
        self.assertLess(np.abs(measured - expected).max(), ROUNDOFF)

    def test_an_empty_path_needs_a_declared_rank(self) -> None:
        self.assertEqual(np.asarray(CT.compose([], 3)).shape, (3, 3))
        with self.assertRaises(ValueError):
            CT.compose([])

    def test_reversal_is_the_transpose_not_the_adjoint(self) -> None:
        generator = rng()
        transfer = random_complex(generator, 4, 5)
        reversed_transfer = np.asarray(CT.reversedTransfer(transfer))
        self.assertLess(np.abs(reversed_transfer - transfer.T).max(),
                        ROUNDOFF)
        # A complex transfer distinguishes the two reversals.
        self.assertGreater(
            np.abs(reversed_transfer - transfer.conj().T).max(), 1.0)

    def test_the_dual_transport_inverts_the_determinant(self) -> None:
        generator = rng()
        transport = invertible(generator, 3)
        dual = np.asarray(CT.dualTransport(transport))
        self.assertLess(np.abs(dual - np.linalg.inv(transport).T).max(),
                        INVERSE)
        self.assertLess(
            abs(complex(np.linalg.det(dual))
                - 1.0 / complex(np.linalg.det(transport))),
            INVERSE)

    def test_the_dual_transport_is_an_involution(self) -> None:
        generator = rng()
        transport = invertible(generator, 3)
        twice = np.asarray(CT.dualTransport(CT.dualTransport(transport)))
        self.assertLess(np.abs(twice - transport).max(), INVERSE)

    def test_a_singular_map_has_no_dual_transport(self) -> None:
        with self.assertRaises(ValueError):
            CT.dualTransport(np.diag([1.0, 0.0, 1.0]).astype(complex))

    def test_the_reversed_transfer_undoes_the_forward_one(self) -> None:
        """On a groupoid fixture -- a complex-orthogonal transfer, where
        T^T = T^-1 -- the transport of the reversed transfer composes with the
        forward transport to the identity exactly."""
        generator = rng()
        antisymmetric = random_complex(generator, 6, 6)
        antisymmetric = antisymmetric - antisymmetric.T
        transfer = expm(antisymmetric)          # T^T T = I exactly
        self.assertLess(np.abs(transfer.T @ transfer - np.eye(6)).max(),
                        INVERSE)
        frame = invertible(generator, 6)
        band = fiber(frame, cells(6))
        forward = np.asarray(
            CT.fiberMap(band.dualFrame(), transfer, band.rightFrame()))
        backward = np.asarray(
            CT.fiberMap(band.dualFrame(),
                        CT.reversedTransfer(transfer), band.rightFrame()))
        self.assertLess(
            np.abs(np.asarray(CT.compose([forward, backward]))
                   - np.eye(6)).max(),
            INVERSE)

    def test_holonomy_invariants_survive_a_frame_change(self) -> None:
        generator = rng()
        links = [invertible(generator, 3) for _ in range(3)]
        base = CT.holonomy(links)
        frame = invertible(generator, 3)
        # A frame change at the base point conjugates the holonomy; inserting
        # it on every link leaves the product conjugated once.
        changed = [np.asarray(CT.frameChanged(link, frame, frame))
                   for link in links]
        moved = CT.holonomy(changed)
        for before, after in zip(base.power_traces, moved.power_traces):
            self.assertLess(abs(before - after), INVERSE)
        self.assertLess(abs(base.determinant - moved.determinant), INVERSE)
        for before, after in zip(base.characteristic_polynomial,
                                 moved.characteristic_polynomial):
            self.assertLess(abs(before - after), INVERSE)


# ─── Kato transport ────────────────────────────────────────────────────────

class TestKatoTransport(unittest.TestCase):
    """The Kato equation K' = [P', P] K and the steps that solve it."""

    def test_the_generator_is_the_commutator(self) -> None:
        generator = rng()
        projector = orthogonal_projector(generator, 5, 2)
        rate = random_complex(generator, 5, 5)
        expected = rate @ projector - projector @ rate
        measured = np.asarray(CT.katoGenerator(rate, projector))
        self.assertLess(np.abs(measured - expected).max(), ROUNDOFF)

    def test_the_direct_rotation_intertwines_exactly_and_is_unitary(self) -> None:
        generator = rng()
        start = orthogonal_projector(generator, 6, 2)
        unitary = expm(1j * 0.3 * hermitian_direction(generator, 6))
        end = unitary @ start @ unitary.conj().T
        step = np.asarray(CT.katoStep(start, end, KatoScheme.DirectRotation))
        self.assertLess(np.abs(step @ start - end @ step).max(), INVERSE)
        self.assertLess(np.abs(step.conj().T @ step - np.eye(6)).max(),
                        INVERSE)

    def test_the_intertwiner_is_exact_for_oblique_idempotents(self) -> None:
        generator = rng()
        start = oblique_idempotent(generator, 5, 2)
        end = oblique_idempotent(generator, 5, 2)
        step = np.asarray(CT.katoStep(start, end, KatoScheme.Intertwiner))
        expected = end @ start + (np.eye(5) - end) @ (np.eye(5) - start)
        self.assertLess(np.abs(step - expected).max(), ROUNDOFF)
        self.assertLess(np.abs(step @ start - end @ step).max(), INVERSE)

    def test_the_exponential_step_is_third_order_and_reported(self) -> None:
        """exp([P1, P0]) is the Kato equation's own exponential step. Its
        intertwining residual falls like the cube of the step, which is
        measured here rather than assumed away."""
        generator = rng()
        start = orthogonal_projector(generator, 6, 2)
        direction = hermitian_direction(generator, 6)
        residuals = []
        for scale in (0.2, 0.1, 0.05):
            unitary = expm(1j * scale * direction)
            end = unitary @ start @ unitary.conj().T
            step = np.asarray(
                CT.katoStep(start, end, KatoScheme.ExponentialGenerator))
            residuals.append(two_norm(step @ start - end @ step))
        for coarse, fine in zip(residuals, residuals[1:]):
            self.assertGreater(coarse / fine, 5.0)   # third order: about 8

    def test_a_closed_projector_loop_gives_a_band_holonomy(self) -> None:
        generator = rng()
        start = orthogonal_projector(generator, 6, 2)
        direction = hermitian_direction(generator, 6)
        angles = np.linspace(0.0, 2.0 * np.pi, 33)
        path = [expm(1j * 0.4 * np.sin(a) * direction) @ start
                @ expm(-1j * 0.4 * np.sin(a) * direction) for a in angles]
        read = CT.katoTransport(path, KatoScheme.DirectRotation)
        self.assertTrue(read.complete)
        self.assertEqual(read.steps, len(path) - 1)
        self.assertEqual(read.rank, 2)
        self.assertLess(read.intertwiningResidual, INVERSE)
        self.assertLess(read.composedIntertwiningResidual, INVERSE)
        self.assertLess(read.idempotencyResidual, INVERSE)
        self.assertTrue(read.certificate.holds())
        # A closed loop returns a transport of the band onto itself.
        transport = np.asarray(read.transport)
        self.assertLess(np.abs(transport @ path[0] - path[0] @ transport).max(),
                        INVERSE)

    def test_the_composed_transport_is_the_ordered_product_of_the_steps(self) -> None:
        generator = rng()
        start = orthogonal_projector(generator, 5, 2)
        direction = hermitian_direction(generator, 5)
        path = [expm(1j * s * direction) @ start @ expm(-1j * s * direction)
                for s in (0.0, 0.05, 0.11, 0.16)]
        read = CT.katoTransport(path, KatoScheme.Intertwiner)
        product = np.eye(5, dtype=complex)
        for step in read.stepTransports:
            product = np.asarray(step) @ product
        self.assertLess(np.abs(np.asarray(read.transport) - product).max(),
                        INVERSE)

    def test_a_coarsely_sampled_path_is_refused_not_approximated(self) -> None:
        """Two projectors at operator-norm distance one have no unique
        geodesic: the read is incomplete with the reason named, and no
        transport is invented."""
        start = np.diag([1.0, 0.0]).astype(complex)
        end = np.diag([0.0, 1.0]).astype(complex)
        read = CT.katoTransport([start, end], KatoScheme.DirectRotation)
        self.assertFalse(read.complete)
        self.assertIn("distance", read.invalidReason)
        self.assertEqual(np.asarray(read.transport).size, 0)
        self.assertFalse(read.certificate.holds())

    def test_the_rank_held_along_the_path_is_reported(self) -> None:
        generator = rng()
        start = orthogonal_projector(generator, 6, 3)
        direction = hermitian_direction(generator, 6)
        path = [expm(1j * s * direction) @ start @ expm(-1j * s * direction)
                for s in (0.0, 0.08, 0.16)]
        read = CT.katoTransport(path)
        self.assertEqual(read.rank, 3)
        self.assertLess(read.rankDefect, INVERSE)
        self.assertGreater(read.maxProjectorStep, 0.0)

    def test_fewer_than_two_projectors_throws(self) -> None:
        generator = rng()
        with self.assertRaises(ValueError):
            CT.katoTransport([orthogonal_projector(generator, 4, 1)])

    def test_a_ragged_projector_path_throws(self) -> None:
        generator = rng()
        with self.assertRaises(ValueError):
            CT.katoTransport([orthogonal_projector(generator, 4, 1),
                              orthogonal_projector(generator, 5, 1)])

    def test_the_fibre_path_reads_its_projectors_from_the_bands(self) -> None:
        generator = rng()
        columns, _ = np.linalg.qr(random_complex(generator, 6, 2))
        direction = hermitian_direction(generator, 6)
        loop = [fiber(expm(1j * s * direction) @ columns, cells(6))
                for s in (0.0, 0.05, 0.1)]
        read = CT.katoTransportOnFibers(loop, KatoScheme.Intertwiner)
        self.assertTrue(read.complete)
        self.assertEqual(read.rank, 2)
        self.assertLess(read.intertwiningResidual, INVERSE)

    def test_a_fibre_path_over_changing_cells_is_refused(self) -> None:
        generator = rng()
        columns, _ = np.linalg.qr(random_complex(generator, 6, 2))
        loop = [fiber(columns, cells(6)),
                fiber(columns, cells(6, offset=100))]
        with self.assertRaises(ValueError):
            CT.katoTransportOnFibers(loop)

    def test_the_band_transport_is_the_bilinear_pairing_of_the_kato_map(self) -> None:
        generator = rng()
        columns, _ = np.linalg.qr(random_complex(generator, 6, 2))
        direction = hermitian_direction(generator, 6)
        start = fiber(columns, cells(6))
        end = fiber(expm(1j * 0.1 * direction) @ columns, cells(6))
        read = CT.katoTransportOnFibers([start, end])
        link = np.asarray(CT.bandTransport(read.transport, end.dualFrame(),
                                           start.rightFrame()))
        expected = (np.asarray(end.dualFrame()).T
                    @ np.asarray(read.transport)
                    @ np.asarray(start.rightFrame()))
        self.assertLess(np.abs(link - expected).max(), INVERSE)
        self.assertEqual(link.shape, (2, 2))


# ─── the exchange character ────────────────────────────────────────────────

def block_swap(rank):
    """The interchange pi_AB of two complete rank-`rank` frames on
    E_A (+) E_B, whose determinant is (-1)^{rank * rank}."""
    size = 2 * rank
    permutation = np.zeros((size, size), complex)
    for i in range(rank):
        permutation[rank + i, i] = 1.0
        permutation[i, rank + i] = 1.0
    return permutation


class TestExchangeCharacter(unittest.TestCase):
    """chi_F = det(H_ex H_ref^-1) from the unprojected general-linear
    holonomies, with its modulus reported and never required."""

    def test_a_single_exchange_of_two_odd_bands_gives_minus_one(self) -> None:
        generator = rng()
        inner_a = invertible(generator, 3)
        inner_b = invertible(generator, 3)
        within = np.zeros((6, 6), complex)
        within[:3, :3] = inner_a
        within[3:, 3:] = inner_b
        exchange = block_swap(3) @ within
        read = CT.exchangeCharacter(exchange, within, 0.0)
        self.assertLess(abs(read.character + 1.0), INVERSE)
        self.assertLess(read.distanceToMinusOne, INVERSE)
        self.assertTrue(read.certificate.holds())
        # The independently computed determinant of the interchange.
        self.assertLess(
            abs(complex(np.linalg.det(block_swap(3))) + 1.0), ROUNDOFF)

    def test_a_double_exchange_gives_plus_one(self) -> None:
        generator = rng()
        inner_a = invertible(generator, 3)
        inner_b = invertible(generator, 3)
        within = np.zeros((6, 6), complex)
        within[:3, :3] = inner_a
        within[3:, 3:] = inner_b
        swap = block_swap(3)
        read = CT.exchangeCharacter(swap @ swap @ within, within, 0.0)
        self.assertLess(abs(read.character - 1.0), INVERSE)
        self.assertLess(read.distanceToPlusOne, INVERSE)
        self.assertTrue(read.certificate.holds())

    def test_an_even_rank_frame_exchange_gives_plus_one(self) -> None:
        """det pi_AB = (-1)^{r_A r_B} is +1 at even rank: exchanging whole
        frames of even rank supplies no sign at all."""
        generator = rng()
        within = np.zeros((4, 4), complex)
        within[:2, :2] = invertible(generator, 2)
        within[2:, 2:] = invertible(generator, 2)
        read = CT.exchangeCharacter(block_swap(2) @ within, within, 0.0)
        self.assertLess(abs(read.character - 1.0), INVERSE)

    def test_the_modulus_is_reported_and_never_required(self) -> None:
        """A realized motion whose within-block transports differ from the
        reference gives a character of modulus other than one. The number is
        reported with its modulus, and the certificate -- which grades the
        leakage premise -- still holds."""
        generator = rng()
        within = np.zeros((6, 6), complex)
        within[:3, :3] = invertible(generator, 3)
        within[3:, 3:] = invertible(generator, 3)
        reference = 0.5 * within
        read = CT.exchangeCharacter(block_swap(3) @ within, reference, 0.0)
        expected = -1.0 / (0.5 ** 6)
        self.assertLess(abs(read.character - expected), INVERSE)
        self.assertLess(abs(read.modulus - abs(expected)), INVERSE)
        self.assertGreater(read.modulus, 2.0)
        self.assertTrue(read.certificate.holds())

    def test_the_two_routes_to_the_character_agree(self) -> None:
        generator = rng()
        exchange = invertible(generator, 4)
        reference = invertible(generator, 4)
        read = CT.exchangeCharacter(exchange, reference, 0.0)
        self.assertLess(read.routeAgreementResidual, INVERSE)
        self.assertLess(
            abs(read.determinantRatio
                - complex(np.linalg.det(exchange))
                / complex(np.linalg.det(reference))),
            INVERSE)

    def test_a_leaking_path_is_uncertified_but_still_reported(self) -> None:
        generator = rng()
        within = np.zeros((6, 6), complex)
        within[:3, :3] = invertible(generator, 3)
        within[3:, 3:] = invertible(generator, 3)
        read = CT.exchangeCharacter(block_swap(3) @ within, within, 1e-3)
        self.assertLess(abs(read.character + 1.0), INVERSE)
        self.assertFalse(read.certificate.holds())

    def test_a_singular_reference_leaves_the_character_unmeasured(self) -> None:
        generator = rng()
        singular = np.zeros((3, 3), complex)
        singular[0, 0] = 1.0
        read = CT.exchangeCharacter(invertible(generator, 3), singular, 0.0)
        self.assertFalse(read.referenceInvertible)
        self.assertFalse(read.certificate.holds())
        self.assertTrue(np.isnan(read.character.real))

    def test_a_rank_mismatch_throws(self) -> None:
        generator = rng()
        with self.assertRaises(ValueError):
            CT.exchangeCharacter(invertible(generator, 3),
                                 invertible(generator, 4), 0.0)

    def test_the_character_of_two_declared_paths(self) -> None:
        generator = rng()
        within = np.zeros((6, 6), complex)
        within[:3, :3] = invertible(generator, 3)
        within[3:, 3:] = invertible(generator, 3)
        half = block_swap(3)
        read = CT.exchangeCharacterOfPaths([within, half], [within], 0.0)
        self.assertLess(abs(read.character + 1.0), INVERSE)
        self.assertEqual(read.rank, 6)

    def test_an_empty_path_throws(self) -> None:
        generator = rng()
        with self.assertRaises(ValueError):
            CT.exchangeCharacterOfPaths([], [invertible(generator, 3)], 0.0)


if __name__ == "__main__":
    unittest.main()
