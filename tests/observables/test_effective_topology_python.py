# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Effective topology as an observable (#1157): what a declared operator sees
at a scale, read separately from the incidence of the complex it lives on, and
the named signatures (effective torus, sphere, components) that certify it."""
import numpy as np
import pytest

from tessera import chainhodge as ch
from tessera import cobordism as cob
from tessera import observables as obs
from tests.chainhodge._periodic import covariant_torus, cubic_grid

KAPPA = (0.3, -0.2, 0.1)
Method = obs.EffectiveBettiNumber.Method


class TestTorus:
    def test_the_actual_torus_is_an_effective_torus_at_the_trivial_connection(self):
        K, _, cov = covariant_torus(cubic_grid(3))
        assert list(K.bettiNumbers()) == [1, 3, 3, 1]
        topology = obs.EffectiveTopology.read(cov, 1.0)
        assert topology.betti() == [1, 3, 3, 1] and topology.certified()
        assert topology.dimension() == 3 and topology.epsilon() == 1.0
        assert all(d.method == Method.DenseSpectrum and d.reason == "" for d in topology.degrees())
        verdict = obs.EffectiveTorus(3).certify(topology)
        assert verdict.holds() and verdict.signature == "effective 3-torus"
        assert verdict.expected == [1, 3, 3, 1] == verdict.measured
        assert verdict.gap > 1e6  # the harmonic bands sit at rounding level
        assert not obs.EffectiveSphere(3).certify(topology).holds()

    def test_a_connection_with_holonomy_is_no_effective_torus_on_the_same_complex(self):
        K, _, cov = covariant_torus(cubic_grid(3), KAPPA)
        assert list(K.bettiNumbers()) == [1, 3, 3, 1]  # the incidence has not changed
        verdict = obs.EffectiveTorus(3).certify(cov, 1.0)
        assert verdict.measured == [0, 0, 0, 0]
        assert verdict.certified and not verdict.matches and not verdict.holds()
        assert obs.EffectiveComponents(0, 3).certify(cov, 1.0).holds()

    def test_a_coarser_scale_sees_more(self):
        _, _, cov = covariant_torus(cubic_grid(3))
        first = np.sort(np.abs(np.array(cov.spectrum(0).eigenvalues)))[1]
        topology = obs.EffectiveTopology.read(cov, 1.01 * first)
        assert topology.betti()[0] == 7  # the constant and the six axis waves
        assert not obs.EffectiveTorus(3).certify(topology).matches


class TestOtherSpaces:
    def test_the_boundary_of_a_tetrahedron_is_an_effective_sphere(self):
        K = cob.ChainComplex.fromTopCells([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
        cov = ch.CovariantChainHodge(ch.ChainHodge(K, [1.0] * 6), ch.Connection.trivial(K))
        verdict = obs.EffectiveSphere(2).certify(cov, 1e-6)
        assert verdict.holds() and verdict.measured == [1, 0, 1]
        assert not obs.EffectiveTorus(2).certify(cov, 1e-6).matches

    def test_a_bottleneck_is_two_effective_components_on_one_actual_component(self):
        ring = lambda first: [[first + i, first + (i + 1) % 6] for i in range(6)]
        K = cob.ChainComplex.fromTopCells(ring(0) + ring(6) + [[0, 6]])
        s = [1.0e6 if tuple(e) == (0, 6) else 1.0 for e in K.kSimplexVertices(1)]
        cov = ch.CovariantChainHodge(ch.ChainHodge(K, s), ch.Connection.trivial(K))
        assert list(K.bettiNumbers()) == [1, 2]
        assert obs.EffectiveComponents(2, 1).certify(cov, 0.01).holds()
        assert obs.EffectiveComponents(1, 1).certify(cov, 1e-8).holds()
        assert obs.EffectiveComponents(2, 1).name() == "2 effective components"


class TestAboveTheCrossover:
    def test_degree_zero_comes_from_the_sparse_pencil_and_the_rest_is_unmeasured(self):
        _, _, cov = covariant_torus(cubic_grid(4), crossover=8)
        topology = obs.EffectiveTopology.read(cov, 1.0)
        assert topology.betti() == [1, -1, -1, -1] and not topology.certified()
        zero, one = topology.degrees()[0], topology.degrees()[1]
        assert zero.method == Method.SparsePencil and zero.certified and zero.gap > 1e6
        assert one.method == Method.Unmeasured and "inverse metric" in one.reason
        # A signature is held only to the degrees it requires.
        assert obs.EffectiveComponents(1, 3).certify(topology).holds()
        torus = obs.EffectiveTorus(3).certify(topology)
        assert not torus.matches and not torus.holds()

    def test_holonomy_empties_the_sparse_read_too(self):
        _, _, cov = covariant_torus(cubic_grid(4), KAPPA, crossover=8)
        zero = obs.EffectiveTopology.read(cov, 1.0).degrees()[0]
        assert zero.rank == 0 and zero.certified and np.isnan(zero.lastInside)
        assert zero.firstOutside > 5.0


def test_refusals():
    _, _, cov = covariant_torus(cubic_grid(3))
    with pytest.raises(ValueError, match="positive"):
        obs.EffectiveTopology.read(cov, 0.0)
    with pytest.raises(ValueError, match="dimension"):
        obs.EffectiveTorus(2).certify(cov, 1.0)
    with pytest.raises(ValueError):
        obs.EffectiveTorus(0)
    with pytest.raises(ValueError):
        obs.EffectiveSphere(0)
    with pytest.raises(ValueError):
        obs.EffectiveComponents(-1, 3)
