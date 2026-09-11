# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The degree-1 harmonic band is a null space, not a spectral cluster (#1058).

The engine reads the band by a Riesz contour quadrature: a dense eigensolve to
pick the radius, then one bordered factorization per contour node applied to
the whole identity, for the connection and again for its inverse. That is the
general machinery for a band of a non-normal pencil, and it is correct for any
band anywhere in the plane.

The HARMONIC band is not any band. The pencil is `A~_1^U z = lambda M_1^U z`,
so the harmonic band at `lambda = 0` is exactly `ker A~_1^U`, and

    A~_1^U = M_1^U (d_1^{U^-1})^T (M_0^U)^-1 d_1^U M_1^U
           + d_2^U M_2^U (d_2^{U^-1})^T

splits into two terms whose separate kernels intersect in
`ker(d_1^U M_1^U) intersect ker((d_2^{U^-1})^T)`.

Neither identification is free. `ker A~_1^U` is the band only when the harmonic
eigenvalue is EXACTLY zero and semisimple -- a defective eigenvalue has an
invariant subspace larger than its kernel, and only the contour sees it. The
split form needs more still: `z^T A~ z = 0` forces both terms to vanish only
for a DEFINITE inner metric, and these squared lengths are complex and the
chain metric complex symmetric and indefinite, so the two terms are free to
cancel against each other.

These tests measure whether they do, on the geometry the engine actually
relaxes. They are the standing guard on a faster reading: if a geometry ever
leaves the regime where the three coincide, this is what says so, by name.

Subspaces are compared by the SINE-based principal angle, `sigma_max` of
`(I - Q_A Q_A^H) Q_B`, which is zero exactly when `span(B)` lies in `span(A)`.
The arccos of a singular value near one loses half the available digits and
reports 1e-6 degrees for a machine-precision agreement.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))),
    "examples", "cobordism"))

import emergence_animation as ea  # noqa: E402
from tessera._tessera.cobordism import PencilLayer  # noqa: E402

TAU_A, TAU_B, GRID = 0.3 + 1.1j, -0.2 + 0.8j, 3
# The band and the two null spaces are the same subspace to round-off; the
# bound is loose enough for the conditioning of a 90-edge complex and tight
# enough that a genuinely different subspace cannot pass.
COINCIDENT = 1e-9


def null_space(M):
    """An orthonormal basis of the null space and the singular-value gap."""
    M = np.asarray(M)
    _, s, Vh = np.linalg.svd(M, full_matrices=True)
    cut = (s[0] if s.size else 1.0) * max(M.shape) * np.finfo(float).eps * 10.0
    keep = (s.size - int((s > cut).sum())) + (Vh.shape[0] - s.size)
    basis = Vh[Vh.shape[0] - keep:, :].conj().T if keep else Vh[:0, :].conj().T
    gap = (s[s.size - keep - 1] / s[s.size - keep]) if 0 < keep < s.size else np.inf
    return basis, gap


def sine_angle(A, B):
    """sigma_max of (I - Q_A Q_A^H) Q_B: zero iff span(B) lies in span(A)."""
    QA = np.linalg.qr(np.asarray(A))[0]
    QB = np.linalg.qr(np.asarray(B))[0]
    residual = QB - QA @ (QA.conj().T @ QB)
    return float(np.linalg.svd(residual, compute_uv=False)[0]) if residual.size else 0.0


def coincide(A, B):
    """The symmetric agreement of two subspaces: both inclusions."""
    return max(sine_angle(A, B), sine_angle(B, A))


def read_three(node):
    """The band and the two null spaces of one geometry."""
    assembled = PencilLayer.assemble([node.spacetime()])
    op = assembled.op
    contour = PencilLayer.harmonic_contour(assembled, 1)
    band = op.band(1, contour)
    aux = np.asarray(op.pencilAux(1))
    # The split form, each block scaled to its own norm so that neither
    # dominates the other's numerical rank.
    closed = np.asarray((op.twistedBoundary(1) @ op.Minv(1)).todense())
    coclosed = np.asarray(op.twistedBoundaryDual(2).todense()).T
    stacked = np.vstack([closed / np.linalg.norm(closed),
                         coclosed / np.linalg.norm(coclosed)])
    return dict(node=node, assembled=assembled, op=op, band=band, aux=aux,
                pencil_null=null_space(aux), hodge_null=null_space(stacked),
                eigenvalues=np.asarray(op.spectrum(1).eigenvalues))


@pytest.fixture(scope="module")
def seeded():
    held = {}
    config = ea.build_config(steps=0, inputs=ea.InputMode.QUBIT, grid=GRID,
                             operator="xx", tau_a=TAU_A, tau_b=TAU_B,
                             input_weight=100.0, regge=False, pin_boundary=True,
                             readout="whole", tori=2, output_state="0.1+1.3j")
    ea.drive(config, progress=False, on_node=lambda node: held.update(node=node))
    return held["node"]


@pytest.fixture(scope="module")
def reading(seeded):
    return read_three(seeded)


def test_the_harmonic_eigenvalue_is_zero_and_not_merely_small(reading):
    """The band is at zero, so a null space can stand in for it at all."""
    magnitudes = np.sort(np.abs(reading["eigenvalues"]))
    rank = reading["band"].rank()
    scale = magnitudes[-1]
    assert magnitudes[rank - 1] < 1e-12 * scale, "the harmonic cluster is not at zero"
    assert magnitudes[rank] > 1e-3 * scale, "no gap above the harmonic cluster"


def test_the_null_space_of_the_pencil_has_the_band_rank(reading):
    basis, gap = reading["pencil_null"]
    assert basis.shape[1] == reading["band"].rank()
    assert gap > 1e6, f"the null space of A~ is not cleanly separated (gap {gap:.3e})"


def test_the_band_is_annihilated_by_the_pencil(reading):
    """A~ Z = 0 to round-off: the band IS in the kernel."""
    aux, images = reading["aux"], np.asarray(reading["band"].images)
    relative = np.linalg.norm(aux @ images) / (np.linalg.norm(aux) * np.linalg.norm(images))
    assert relative < 1e-13, f"||A~ Z|| / (||A~|| ||Z||) = {relative:.3e}"


def test_the_band_and_the_pencil_null_space_are_one_subspace(reading):
    """The contour buys nothing here: the two spans coincide."""
    agreement = coincide(np.asarray(reading["band"].images), reading["pencil_null"][0])
    assert agreement < COINCIDENT, f"sine principal angle {agreement:.3e}"


def test_the_split_form_agrees_with_the_pencil_null_space(reading):
    """The two terms of A~ do not cancel against each other on this geometry.

    This is the assertion with no theorem behind it: the chain metric is
    complex symmetric and indefinite, so a cancellation is permitted. It does
    not happen here, which is why the closed-and-co-closed form is a legitimate
    reading of the band and not only a Riemannian habit.
    """
    agreement = coincide(reading["pencil_null"][0], reading["hodge_null"][0])
    assert reading["hodge_null"][0].shape[1] == reading["pencil_null"][0].shape[1]
    assert agreement < COINCIDENT, f"sine principal angle {agreement:.3e}"


def test_the_band_rank_is_the_first_betti_number(reading):
    """dim ker L_1 = b_1, the Hodge count, here over a complex chain metric."""
    node = reading["node"]
    assert reading["band"].rank() == node.betti(node.spacetime())[1]


@pytest.mark.parametrize("amplitude", [0.02, 0.10, 0.30])
def test_the_three_still_coincide_off_the_seeded_lengths(seeded, amplitude):
    """The agreement is not a property of the seeded collar.

    Every squared length is moved by an independent complex multiplicative
    jitter, which is where a relaxed complex lives: a metric with no symmetry
    left and no relation to the one the identification was first measured on.
    The ranks and the agreement are unchanged, which is what a RANK argument
    predicts and a positivity argument would not. The two kernels being
    intersected have codimensions `rank d_1` and `rank d_2`, so their
    intersection has dimension at least `b_1` for any non-degenerate metric;
    only an accidental cancellation between the two terms of `A~` could
    enlarge `ker A~` beyond it, and that is a codimension condition a generic
    perturbation does not meet.
    """
    spacetime = seeded.spacetime()
    edges = list(spacetime.getEdgeList().toVector())
    original = [edge.getLength() for edge in edges]
    generator = np.random.default_rng(3)
    try:
        for edge, length in zip(edges, original):
            step = generator.standard_normal() + 1j * generator.standard_normal()
            edge.setLength(length * (1.0 + amplitude * step))
        jittered = read_three(seeded)
        rank = jittered["band"].rank()
        assert jittered["pencil_null"][0].shape[1] == rank
        assert jittered["hodge_null"][0].shape[1] == rank
        magnitudes = np.sort(np.abs(jittered["eigenvalues"]))
        assert magnitudes[rank - 1] < 1e-12 * magnitudes[-1]
        assert magnitudes[rank] > 1e-3 * magnitudes[-1]
        images = np.asarray(jittered["band"].images)
        assert coincide(images, jittered["pencil_null"][0]) < COINCIDENT
        assert coincide(jittered["pencil_null"][0], jittered["hodge_null"][0]) < COINCIDENT
    finally:
        for edge, length in zip(edges, original):
            edge.setLength(length)
