# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Second derivatives of the Whitney pencil in the squared lengths (#1185).

The directional derivative of the spectral-entropy gradient of the default
(Whitney) operator needs D_v dM_k/ds_e of the Whitney mass matrices and
D_v dh_k/ds_e of the covariant operator. Both are analytic; they are checked
here against central differences of the analytic first derivatives along the
same direction (the lengths are holomorphic coordinates, so a real step t in
s + t v differentiates along v), against the symmetry of the Hessian, and the
directional first derivative against the sum of the per-edge derivatives."""
import numpy as np
import pytest

from tessera import chainhodge as ch
from tessera import cobordism as cob

WM = ch.WhitneyMass

FIXTURES = {
    "1-complex": [[0, 1], [0, 2], [1, 2], [2, 3]],
    "2-complex": [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3], [2, 3, 4]],
    "3-complex": [[0, 1, 2, 3], [1, 2, 3, 4]],
}
STEP = 1e-5


def _instance(name, seed):
    rng = np.random.default_rng(seed)
    K = cob.ChainComplex.fromTopCells(FIXTURES[name])
    n = K.numSimplices(1)
    s = np.array([complex(1.0 + 0.1 * rng.normal(), 0.05 * rng.normal()) for _ in range(n)])
    v = rng.normal(size=n) + 1j * rng.normal(size=n)
    return K, s, v, rng


def _dense(sparse):
    return sparse.toarray()


class TestMassSecondDerivatives:
    @pytest.mark.parametrize("name", sorted(FIXTURES))
    def test_directional_is_the_sum_of_the_edge_derivatives(self, name):
        K, s, v, _ = _instance(name, 3)
        for k in range(K.dimension() + 1):
            expected = sum(v[e] * _dense(WM.assembleDerivative(K, list(s), k, e)) for e in range(len(s)))
            got = _dense(WM.assembleDirectionalDerivative(K, list(s), k, list(v)))
            np.testing.assert_allclose(got, expected, atol=1e-12 * max(1.0, np.abs(expected).max()))

    @pytest.mark.parametrize("name", sorted(FIXTURES))
    def test_against_central_differences_of_the_first_derivative(self, name):
        K, s, v, _ = _instance(name, 5)
        for k in range(K.dimension() + 1):
            second = WM.assembleSecondDerivatives(K, list(s), k, list(v))
            assert len(second) == len(s)
            for e in range(len(s)):
                up = _dense(WM.assembleDerivative(K, list(s + STEP * v), k, e))
                down = _dense(WM.assembleDerivative(K, list(s - STEP * v), k, e))
                fd = (up - down) / (2 * STEP)
                np.testing.assert_allclose(_dense(second[e]), fd, atol=1e-6 * max(1.0, np.abs(fd).max()))

    @pytest.mark.parametrize("name", sorted(FIXTURES))
    def test_the_hessian_is_symmetric(self, name):
        K, s, _, _ = _instance(name, 7)
        n = len(s)
        for k in range(K.dimension() + 1):
            unit = [np.eye(n)[f] for f in range(n)]
            H = [WM.assembleSecondDerivatives(K, list(s), k, list(u)) for u in unit]
            for e in range(n):
                for f in range(e + 1, n):
                    a, b = _dense(H[f][e]), _dense(H[e][f])
                    np.testing.assert_allclose(a, b, atol=1e-12 * max(1.0, np.abs(a).max()))

    def test_scaling_identity_differentiated(self):
        """sum_e s_e dM_k/ds_e = (d/2 - k) M_k differentiated along v:
        sum_e s_e D_v dM_k/ds_e = (d/2 - k - 1) D_v M_k."""
        K, s, v, _ = _instance("3-complex", 11)
        d = K.dimension()
        for k in range(d + 1):
            second = WM.assembleSecondDerivatives(K, list(s), k, list(v))
            total = sum(s[e] * _dense(second[e]) for e in range(len(s)))
            directional = _dense(WM.assembleDirectionalDerivative(K, list(s), k, list(v)))
            np.testing.assert_allclose(total, (d / 2 - k - 1) * directional,
                                       atol=1e-11 * max(1.0, np.abs(directional).max()))

    def test_a_direction_off_a_simplex_leaves_its_block(self):
        """The second derivative along a direction supported away from every
        simplex of an edge's star is zero on that edge."""
        K = cob.ChainComplex.fromTopCells(FIXTURES["2-complex"])
        s = [1.0 + 0.1j * i for i in range(K.numSimplices(1))]
        edges = [tuple(int(x) for x in e) for e in K.kSimplexVertices(1)]
        v = [1.0 if e == (3, 4) else 0.0 for e in edges]
        second = WM.assembleSecondDerivatives(K, s, 1, v)
        assert np.abs(_dense(second[edges.index((0, 1))])).max() == 0.0
        assert np.abs(_dense(second[edges.index((2, 3))])).max() > 0.0

    def test_direction_length_is_checked(self):
        K = cob.ChainComplex.fromTopCells(FIXTURES["2-complex"])
        s = [1.0] * K.numSimplices(1)
        with pytest.raises(ValueError, match="one direction entry per edge"):
            WM.assembleSecondDerivatives(K, s, 1, [1.0])


class TestCovariantOperatorSecondDerivative:
    def _covariant(self, K, s, links):
        return ch.CovariantChainHodge(ch.ChainHodge(K, list(s)), ch.Connection(K, links))

    @pytest.mark.parametrize("name", sorted(FIXTURES))
    def test_against_central_differences(self, name):
        K, s, v, rng = _instance(name, 13)
        links = [complex(np.exp(1j * rng.normal())) for _ in range(len(s))]
        cov = self._covariant(K, s, links)
        up = self._covariant(K, s + STEP * v, links)
        down = self._covariant(K, s - STEP * v, links)
        for k in range(K.dimension() + 1):
            for e in range(len(s)):
                fd = (up.covariantOperatorDerivative(k, e) - down.covariantOperatorDerivative(k, e)) / (2 * STEP)
                got = cov.covariantOperatorSecondDerivative(k, e, list(v))
                np.testing.assert_allclose(got, fd, atol=1e-6 * max(1.0, np.abs(fd).max()))

    @pytest.mark.parametrize("name", sorted(FIXTURES))
    def test_directional_is_the_sum_of_the_edge_derivatives(self, name):
        K, s, v, rng = _instance(name, 17)
        links = [complex(np.exp(1j * rng.normal())) for _ in range(len(s))]
        cov = self._covariant(K, s, links)
        for k in range(K.dimension() + 1):
            expected = sum(v[e] * cov.covariantOperatorDerivative(k, e) for e in range(len(s)))
            got = cov.covariantOperatorDirectionalDerivative(k, list(v))
            np.testing.assert_allclose(got, expected, atol=1e-11 * max(1.0, np.abs(expected).max()))

    def test_the_hessian_is_symmetric(self):
        K, s, _, rng = _instance("2-complex", 19)
        n = len(s)
        links = [complex(np.exp(1j * rng.normal())) for _ in range(n)]
        cov = self._covariant(K, s, links)
        unit = np.eye(n)
        for e, f in [(0, 1), (2, 5), (3, 7)]:
            a = cov.covariantOperatorSecondDerivative(1, e, list(unit[f]))
            b = cov.covariantOperatorSecondDerivative(1, f, list(unit[e]))
            np.testing.assert_allclose(a, b, atol=1e-11 * max(1.0, np.abs(a).max()))
