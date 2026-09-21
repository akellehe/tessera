# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The sparse pencil solver with a factored low-rank term (#1161): the pencil
(A + P D P^dagger, M) solved through the Woodbury identity, against the dense
pencil with the term added, with the inertia certificate of the shift."""
import numpy as np
import pytest
import scipy.linalg

from tessera import chainhodge as ch
from tests.chainhodge._periodic import covariant_torus, cubic_grid

KAPPA = (0.3, -0.2, 0.1)


def _problem(kappa=KAPPA, n=4, rank=5, seed=2):
    _, _, cov = covariant_torus(cubic_grid(n), kappa)
    pencil = cov.sparsePencil()
    rng = np.random.default_rng(seed)
    size = pencil.A.shape[0]
    functions = rng.normal(size=(size, rank)) + 1j * rng.normal(size=(size, rank))
    P = pencil.M @ functions
    D = np.diag([-30.0, -12.0, 8.0, 20.0, 45.0][:rank]).astype(complex)
    dense = pencil.A.toarray() + P @ D @ P.conj().T
    return pencil, P, D, dense


def test_levels_vectors_and_certificate_against_the_dense_pencil():
    pencil, P, D, dense = _problem()
    reference = scipy.linalg.eigh(dense, pencil.M.toarray(), eigvals_only=True)
    sigma = reference[0] - 1.0
    read = ch.SparsePencilSolver.lowestWithLowRank(pencil.A, pencil.M, P, D, 10, sigma)
    values = np.array(read.eigenvalues.values).real
    assert values == pytest.approx(reference[:10], abs=1e-8 * abs(reference).max())
    assert read.eigenvalues.certificate.holds() and read.shiftBelowSpectrum
    Z = read.vectors
    assert np.abs(Z.conj().T @ (pencil.M @ Z) - np.eye(10)).max() < 1e-9
    assert np.abs(dense @ Z - (pencil.M @ Z) * values).max() < 1e-7 * abs(reference).max()
    free = scipy.linalg.eigh(pencil.A.toarray(), pencil.M.toarray(), eigvals_only=True)
    assert np.abs(reference[:10] - free[:10]).max() > 1e-3           # the term did move the levels


def test_the_inertia_certificate_counts_levels_below_the_shift():
    pencil, P, D, dense = _problem()
    reference = scipy.linalg.eigh(dense, pencil.M.toarray(), eigvals_only=True)
    # Between the two lowest levels: A - sigma M alone may well be positive
    # definite, but the pencil with the term has one level below the shift.
    sigma = 0.5 * (reference[0] + reference[1])
    read = ch.SparsePencilSolver.lowestWithLowRank(pencil.A, pencil.M, P, D, 4, sigma)
    assert not read.shiftBelowSpectrum
    assert np.array(read.eigenvalues.values).real == pytest.approx(reference[1:5], abs=1e-8 * abs(reference).max())


def test_degenerate_levels_keep_every_copy():
    """At the zone centre with a term that leaves the first shell's sine triplet
    untouched, the solver still returns whole multiplets."""
    _, _, cov = covariant_torus(cubic_grid(5))
    pencil = cov.sparsePencil()
    size = pencil.A.shape[0]
    P = pencil.M @ np.ones((size, 1), dtype=complex)                  # couples to the constant only
    D = np.array([[0.7 + 0j]])
    dense = pencil.A.toarray() + P @ D @ P.conj().T
    reference = scipy.linalg.eigh(dense, pencil.M.toarray(), eigvals_only=True)[:7]
    read = ch.SparsePencilSolver.lowestWithLowRank(pencil.A, pencil.M, P, D, 7, -1.0)
    assert np.array(read.eigenvalues.values).real == pytest.approx(reference, abs=1e-8 * reference.max())
    assert np.ptp(reference[1:7]) < 1e-9 * reference[1]             # the six-fold shell, all copies found


def test_refusals():
    pencil, P, D, _ = _problem()
    with pytest.raises(ValueError, match="n x r"):
        ch.SparsePencilSolver.lowestWithLowRank(pencil.A, pencil.M, P[:-1], D, 3, -50.0)
    with pytest.raises(ValueError, match="not Hermitian"):
        ch.SparsePencilSolver.lowestWithLowRank(pencil.A, pencil.M, P, D + 1j * np.eye(5), 3, -50.0)
