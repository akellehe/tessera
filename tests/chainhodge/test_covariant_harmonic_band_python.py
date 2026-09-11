# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""`CovariantChainHodge.harmonicBand`: the lambda = 0 band without a contour.

The specification states both the identification and the computational path
(RSF Sec. 5, Prop. 2): under the rank conditions (R1)-(R4), `ker L_1 = H_1`
with no Jordan block at zero and the `lambda = 0` Riesz projector IS the
projector onto `H_1`; and `H_k = M_k ker S` with the sparse stacked matrix
`S = [d_{k+1}^T ; d_k M_k]`.

`ChainHodge::harmonicChains` already reads that null space for the undressed
operator. `CovariantChainHodge` -- the connection-dressed operator every
cobordism reading goes through -- had only the contour. These tests cover the
dressed reading `S^U = [(d_{k+1}^{U^-1})^T ; d_k^U M_k^U]`.

What they do NOT claim is that the two bands are interchangeable in the
cobordism layer. They span one subspace and return two bases of it, and
`wholeHarmonicResidualOn` reads a coefficient vector in the band's basis; see
`tests/cobordism/test_harmonic_band_is_a_null_space_python.py`.
"""
import numpy as np
import pytest

from tessera import chainhodge as ch
from tessera import cobordism as cob
from tests.chainhodge._fixtures import random_allowable, torus_cells


def sine_angle(A, B):
    """sigma_max of (I - Q_A Q_A^H) Q_B: zero iff span(B) lies in span(A)."""
    QA = np.linalg.qr(np.asarray(A))[0]
    QB = np.linalg.qr(np.asarray(B))[0]
    residual = QB - QA @ (QA.conj().T @ QB)
    return float(np.linalg.svd(residual, compute_uv=False)[0]) if residual.size else 0.0


def coincide(A, B):
    return max(sine_angle(A, B), sine_angle(B, A))


@pytest.fixture(scope="module")
def operator():
    """A 4x4 torus dressed by a random C* connection: b_1 = 2, so the degree-1
    harmonic space is two-dimensional, and the connection is neither trivial
    nor unitary, so the pencil is genuinely non-normal."""
    generator = np.random.default_rng(17)
    cells, _ = torus_cells(4)
    complex_ = cob.ChainComplex.fromTopCells(cells)
    lengths = random_allowable(complex_, generator, 0.3)
    base = ch.ChainHodge(complex_, lengths)
    # A PURE GAUGE connection: the links are nowhere trivial, but the
    # curvature vanishes, so the twisted incidences still compose to zero and
    # a harmonic space exists. A connection with curvature has none -- see
    # `test_a_curved_connection_has_no_harmonic_space` -- and is the wrong
    # regime for a band at zero.
    gauge = {int(v[0]): complex(generator.normal(), generator.normal()) + 0.3
             for v in complex_.kSimplexVertices(0)}
    return ch.CovariantChainHodge(base, ch.Connection.trivial(complex_).gauge(gauge))


def contour_band(operator):
    """The reference reading: a circle around zero excluding the first
    nonzero eigenvalue, at the spectral radius the spectrum reports."""
    eigenvalues = np.abs(np.asarray(operator.spectrum(1).eigenvalues))
    scale = eigenvalues.max()
    nonzero = eigenvalues[eigenvalues > 1e-9 * max(scale, 1.0)]
    radius = 0.5 * nonzero.min() if nonzero.size else 1.0
    return operator.band(1, ch.Contour.circle(0.0 + 0.0j, radius, 64))


def test_the_harmonic_band_spans_what_the_contour_band_spans(operator):
    contour = contour_band(operator)
    harmonic = operator.harmonicBand(1)
    assert harmonic.rank() == contour.rank()
    assert coincide(np.asarray(contour.images), np.asarray(harmonic.images)) < 1e-9


def test_the_harmonic_band_is_annihilated_by_the_pencil(operator):
    aux = np.asarray(operator.pencilAux(1))
    images = np.asarray(operator.harmonicBand(1).images)
    relative = np.linalg.norm(aux @ images) / (np.linalg.norm(aux) * np.linalg.norm(images))
    assert relative < 1e-12


def test_the_chains_and_the_images_are_related_by_the_chain_metric(operator):
    """H_k = M_k^U ker S^U, and G_k^U H_k is the kernel back again."""
    read = operator.harmonicChains(1)
    chains, images = np.asarray(read.chains), np.asarray(read.images)
    assert np.linalg.norm(np.asarray(operator.applyMinv(1, images)) - chains) \
        / np.linalg.norm(chains) < 1e-12
    assert np.linalg.norm(np.asarray(operator.applyG(1, chains)) - images) \
        / np.linalg.norm(images) < 1e-10


def test_the_kernel_vectors_are_orthonormal(operator):
    """The dense path takes them from the SVD's V, so they come out orthonormal."""
    images = np.asarray(operator.harmonicChains(1).images)
    gram = images.conj().T @ images
    assert np.linalg.norm(gram - np.eye(gram.shape[0])) < 1e-12


def test_the_sparse_path_finds_the_same_space(operator):
    """Below the crossover by SVD, above it by rank-revealing QR: one subspace."""
    dense = operator.harmonicChains(1)
    sparse = operator.harmonicChains(1, 10.0, True)
    assert sparse.nullity == dense.nullity
    assert coincide(np.asarray(dense.images), np.asarray(sparse.images)) < 1e-8


def test_the_band_says_it_has_no_contour(operator):
    """A zero node count, rather than a plausible-looking circle."""
    certificate = operator.harmonicBand(1).certificate
    assert certificate.nodeCount == 0
    assert "no contour" in certificate.contour
    # Idempotency and the resolvent maximum certify a contour quadrature and
    # have no meaning without one.
    assert np.isnan(certificate.idempotency)
    assert np.isnan(certificate.resolventMax)


def test_the_band_carries_the_kernel_margin(operator):
    """The singular gap of S^U is reported, so the margin is not assumed."""
    certificate = operator.harmonicBand(1).certificate
    assert certificate.rank > 0
    assert certificate.rankTolerance > 0.0
    assert certificate.singularGap > 1e6


def test_the_left_frame_agrees_with_the_contour_band(operator):
    """The shared tail: both readings reach the same isotropy verdict."""
    contour = contour_band(operator)
    harmonic = operator.harmonicBand(1)
    assert harmonic.certificate.leftFrameAvailable == contour.certificate.leftFrameAvailable
    if harmonic.certificate.leftFrameAvailable:
        # Phi~^T Phi = I is the defining property of the canonical left frame,
        # and it holds in whichever basis the band chose.
        product = np.asarray(harmonic.leftFrame).T @ np.asarray(harmonic.frame)
        assert np.linalg.norm(product - np.eye(product.shape[0])) < 1e-8


def test_a_curved_connection_has_no_harmonic_space(operator):
    """Where the reading correctly returns nothing.

    Without flatness the twisted incidences do not compose to zero
    (Prop. 5.1 (iv): `d_1^U d_2^U t = U_rp (F_t - 1)[r]`), so `S^U` has full
    column rank and its kernel is empty. The null-space reading reports a
    nullity of zero rather than a small-singular-value artifact, and the
    contour finds nothing to enclose either.
    """
    generator = np.random.default_rng(23)
    cells, _ = torus_cells(4)
    complex_ = cob.ChainComplex.fromTopCells(cells)
    base = ch.ChainHodge(complex_, random_allowable(complex_, generator, 0.3))
    links = [complex(generator.normal(), generator.normal())
             for _ in range(complex_.numSimplices(1))]
    curved = ch.CovariantChainHodge(base, ch.Connection(complex_, links))
    assert not curved.connection().isUnitary()
    assert curved.harmonicChains(1).nullity == 0
    assert curved.harmonicBand(1).rank() == 0
