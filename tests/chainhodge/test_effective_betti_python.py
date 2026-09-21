# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Effective Betti numbers from the sparse pencil (#1157): the rank of the
spectral band of the declared operator in [0, epsilon]. They are spectral, not
incidence data: a bottleneck raises the degree-zero count above the number of
connected components, and a connection with holonomy empties the harmonic
spaces of a torus whose incidence Betti numbers are (1, 3, 3, 1)."""
import numpy as np
import pytest

from tessera import chainhodge as ch
from tessera import cobordism as cob
from tests.chainhodge._periodic import covariant_torus, cubic_grid

KAPPA = (0.3, -0.2, 0.1)


def _read(cov, epsilon, **kwargs):
    pencil = cov.sparsePencil()
    return ch.SparsePencilSolver.effectiveBetti(pencil.A, pencil.M, epsilon, -1.0, **kwargs)


class TestBottleneck:
    def test_two_rings_joined_by_a_long_edge_are_two_effective_components(self):
        """One connected component by incidence; two at any scale between the
        bridge level and the first level of a ring."""
        ring = lambda first: [[first + i, first + (i + 1) % 6] for i in range(6)]
        K = cob.ChainComplex.fromTopCells(ring(0) + ring(6) + [[0, 6]])
        s = [1.0e6 if tuple(e) == (0, 6) else 1.0 for e in K.kSimplexVertices(1)]
        cov = ch.CovariantChainHodge(ch.ChainHodge(K, s), ch.Connection.trivial(K))
        assert list(K.bettiNumbers()) == [1, 2]

        read = _read(cov, 0.01)
        assert read.rank == 2 and read.complete and read.certified
        assert read.lastInside < 1e-4 and read.firstOutside > 0.2 and read.gap > 1e4
        dense = np.sort(np.array(cov.spectrum(0).eigenvalues).real)
        assert np.count_nonzero(dense <= 0.01) == 2
        # Below the bridge level only the constant is left; the incidence count is the limit.
        assert _read(cov, 1e-8).rank == 1


class TestCovariantTorus:
    def test_trivial_connection_counts_the_constant_then_the_first_shell(self):
        K, _, cov = covariant_torus(cubic_grid(4))
        assert list(K.bettiNumbers()) == [1, 3, 3, 1]
        kernel = _read(cov, 1.0)
        assert kernel.rank == 1 and kernel.certified and kernel.gap == np.inf
        assert abs(kernel.lastInside) < 1e-10
        shell = _read(cov, 60.0)
        assert shell.rank == 7 and shell.certified
        assert shell.gap == pytest.approx(2.0, rel=1e-9)  # 96 / 48: the next shell over this one
        assert cov.band(0, ch.Contour.circle(30.0, 31.0, 64)).rank() == 7

    def test_a_connection_with_holonomy_has_no_near_kernel(self):
        """The incidence of the complex is unchanged, but the covariant
        operator has no zero mode: its lowest level is |k|^2 on the mesh."""
        K, _, cov = covariant_torus(cubic_grid(4), KAPPA)
        assert list(K.bettiNumbers()) == [1, 3, 3, 1]
        empty = _read(cov, 1.0)
        assert empty.rank == 0 and empty.complete and empty.certified
        assert np.isnan(empty.lastInside) and empty.firstOutside > 5.0
        lowest = np.sort(np.array(cov.spectrum(0).eigenvalues).real)
        assert empty.firstOutside == pytest.approx(lowest[0], rel=1e-10)
        for epsilon in (20.0, 40.0):
            assert _read(cov, epsilon).rank == np.count_nonzero(lowest <= epsilon)

    def test_the_harmonic_spaces_follow_the_connection_not_the_incidence(self):
        K, _, trivial = covariant_torus(cubic_grid(3))
        _, _, twisted = covariant_torus(cubic_grid(3), KAPPA)
        assert list(K.bettiNumbers()) == [1, 3, 3, 1]
        assert [trivial.harmonicChains(k).nullity for k in (0, 1)] == [1, 3]
        assert [twisted.harmonicChains(k).nullity for k in (0, 1)] == [0, 0]

    def test_the_count_grows_until_it_leaves_the_window(self):
        _, _, cov = covariant_torus(cubic_grid(4))
        read = _read(cov, 60.0, initial_count=2)
        assert read.rank == 7 and read.complete
        assert len(read.band.eigenvalues.values) > 7


def test_the_window_must_lie_above_the_shift():
    _, _, cov = covariant_torus(cubic_grid(3))
    pencil = cov.sparsePencil()
    with pytest.raises(ValueError, match="above the shift"):
        ch.SparsePencilSolver.effectiveBetti(pencil.A, pencil.M, -2.0, -1.0)
