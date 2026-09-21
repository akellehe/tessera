# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Sparse block pencils (#1157): two sheets as a direct sum double every level,
and a constant coupling between them splits every level by exactly +Delta and
-Delta."""
import numpy as np
import pytest
import scipy.sparse as sp

from tessera import chainhodge as ch
from tests.chainhodge._periodic import covariant_torus, cubic_grid

KAPPA = (0.3, -0.2, 0.1)


def _sheet(n=3):
    _, _, cov = covariant_torus(cubic_grid(n), KAPPA)
    return cov.sparsePencil()


def _levels(pencil, count, sigma=-10.0):
    read = ch.SparsePencilSolver.lowest(pencil.A, pencil.M, count, sigma)
    assert read.eigenvalues.certificate.holds()
    return np.array(read.eigenvalues.values).real


def test_direct_sum_doubles_every_level():
    sheet = _sheet()
    n = sheet.A.shape[0]
    both = ch.SparsePencilComposition.directSum(sheet, sheet)
    assert both.degree == 0 and both.A.shape == (2 * n, 2 * n) and both.M.shape == (2 * n, 2 * n)
    assert abs(both.A - sp.block_diag([sheet.A, sheet.A])).max() == 0.0
    assert abs(both.M - sp.block_diag([sheet.M, sheet.M])).max() == 0.0
    single = _levels(sheet, 5)
    assert _levels(both, 10) == pytest.approx(np.repeat(single, 2), rel=1e-10)


def test_constant_coupling_splits_every_level():
    sheet = _sheet()
    delta = 0.75
    coupled = ch.SparsePencilComposition.hoppingBlock(sheet, sheet, delta * sheet.M)
    assert abs(coupled.A - coupled.A.conj().T).max() < 1e-15
    single = _levels(sheet, 5)
    expected = np.sort(np.concatenate([single - delta, single + delta]))
    assert _levels(coupled, 10) == pytest.approx(expected, rel=1e-10)


def test_reverse_block_defaults_to_the_adjoint():
    sheet = _sheet()
    n = sheet.A.shape[0]
    C = sp.random(n, n, density=0.05, random_state=1, dtype=float).astype(complex) * (1 + 2j)
    R = sp.random(n, n, density=0.05, random_state=2, dtype=float).astype(complex)
    hermitian = ch.SparsePencilComposition.hoppingBlock(sheet, sheet, C).A.toarray()
    assert np.abs(hermitian[:n, n:] - C.toarray()).max() == 0.0
    assert np.abs(hermitian[n:, :n] - C.conj().T.toarray()).max() == 0.0
    general = ch.SparsePencilComposition.hoppingBlock(sheet, sheet, C, R).A.toarray()
    assert np.abs(general[n:, :n] - R.toarray()).max() == 0.0


def test_sheets_of_different_size_and_refusals():
    small, large = _sheet(3), _sheet(4)
    ns, nl = small.A.shape[0], large.A.shape[0]
    both = ch.SparsePencilComposition.directSum(small, large)
    assert both.A.shape == (ns + nl, ns + nl)
    merged = np.sort(np.concatenate([_levels(small, 4), _levels(large, 4)]))[:4]
    assert _levels(both, 4) == pytest.approx(merged, rel=1e-10)
    with pytest.raises(ValueError, match="n_a x n_b"):
        ch.SparsePencilComposition.hoppingBlock(small, large, sp.csc_matrix((ns, ns), dtype=complex))
    with pytest.raises(ValueError, match="n_b x n_a"):
        ch.SparsePencilComposition.hoppingBlock(small, large, sp.csc_matrix((ns, nl), dtype=complex),
                                                sp.csc_matrix((ns, nl), dtype=complex))
    broken = ch.SparsePencil(0, small.A, large.M)
    with pytest.raises(ValueError, match="square of one size"):
        ch.SparsePencilComposition.directSum(broken, small)
