# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Covariant chain operator (#909, specification §5): the dressed Whitney metric
and twisted incidences, Prop. 5.1 (i)-(vi) on random complex instances
(§14 T1-T4 tolerances), the spacetime phase adapter, and the Grassmann
preset's dressing."""
import cmath

import numpy as np
import pytest

import tessera
from tessera import chainhodge as ch
from tessera import cobordism as cob
from tests.chainhodge._fixtures import edges, random_allowable, torus_cells

TWO_COMPLEX = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3], [2, 3, 4]]


def _random_links(K, rng):
    return [complex(rng.normal(), rng.normal()) for _ in range(K.numSimplices(1))]


def _random_gauge(K, rng):
    return {int(v[0]): complex(rng.normal(), rng.normal()) + 0.3 for v in K.kSimplexVertices(0)}


def _random_complex_lengths(K, rng):
    return [complex(rng.normal(), rng.normal()) for _ in range(K.numSimplices(1))]


def _hausdorff(a, b):
    a, b = np.asarray(a), np.asarray(b)
    d = np.abs(a[:, None] - b[None, :])
    return max(d.min(axis=1).max(), d.min(axis=0).max())


class TestConnection:
    def test_links_inverse_gauge_curvature(self):
        rng = np.random.default_rng(1)
        K = cob.ChainComplex.fromTopCells(TWO_COMPLEX)
        links = _random_links(K, rng)
        U = ch.Connection(K, links)
        es = edges(K)
        for (x, y), u in zip(es, links):
            assert U.link(x, y) == u
            assert U.link(y, x) == pytest.approx(1.0 / u, rel=1e-15)
        assert U.link(2, 2) == 1.0
        Ui = U.inverse()
        assert all(a == pytest.approx(1.0 / b, rel=1e-15) for a, b in zip(Ui.links(), links))
        g = _random_gauge(K, rng)
        Ug = U.gauge(g)
        for (x, y), u, ug in zip(es, links, Ug.links()):
            assert ug == pytest.approx(u / g[x] * g[y], rel=1e-14)
        p, q, r = 0, 1, 2
        assert U.curvature(p, q, r) == pytest.approx(U.link(r, q) * U.link(q, p) * U.link(p, r), rel=1e-14)
        assert not U.isUnitary()
        assert ch.Connection.trivial(K).isUnitary()
        with pytest.raises(ValueError):
            ch.Connection(K, links[:-1])
        with pytest.raises(ValueError):
            ch.Connection(K, [0.0] + links[1:])

    def test_spacetime_phase_adapter(self):
        """The stored phase is the C* connection on the edge's source->target
        orientation: U_xy = e^{i phi} when the source is x < y, else e^{-i phi}."""
        st = tessera.Spacetime.fromVertexTuples(2, TWO_COMPLEX, 1.0, 0.0)
        K = ch.WhitneyMass.complexOf(st)
        phases = {}
        for i, e in enumerate(st.getEdgeList().toVector()):
            phi = 0.3 * i + 0.1j * (i % 3)
            e.setPhase(phi)
            phases[(e.getSource().getId(), e.getTarget().getId())] = phi
        U = ch.Connection.fromSpacetime(st, K)
        for (x, y), u in zip(edges(K), U.links()):
            if (x, y) in phases:
                assert u == pytest.approx(cmath.exp(1j * phases[(x, y)]), rel=1e-14)
            else:
                assert u == pytest.approx(cmath.exp(-1j * phases[(y, x)]), rel=1e-14)


PROBES = ("transposeMetric", "transposePencilProbe", "transposeOperatorProbe", "covarianceMetric",
          "covariancePencilProbe", "covarianceOperatorProbe", "curvature", "pureGaugeSimilarityProbe",
          "pairingInvariance")


def _assert_proposition_3(cov, K, trivial=False):
    """Proposition 3 (i)-(vi) as asserted on construction: every residual
    measured, finite, and within the certificate's tolerance 10 n eps cond."""
    cert = cov.certificate()
    assert cert.holds
    n = max(K.numSimplices(k) for k in range(K.dimension() + 1))
    assert cert.tolerance == pytest.approx(10 * n * np.finfo(float).eps * cert.conditionEstimate, rel=1e-12)
    for name in PROBES:
        value = getattr(cert, name)
        assert np.isfinite(value) and value <= cert.tolerance, (name, value, cert.tolerance)
    if trivial:
        assert cert.trivialReductionProbe <= cert.tolerance
    else:
        assert np.isnan(cert.trivialReductionProbe)


def _assert_harmonic_dimension(base):
    """Step 2 of the numerical program on the instance: dim H_1 = b_1 and (R1)-(R4)."""
    betti = base.betti()
    for k in range(base.dimension() + 1):
        assert base.harmonicChains(k).nullity == betti[k]
        assert base.rankConditions(k).kernelIsHarmonic


class TestProposition51:
    """T1-T4 (§14): random complex s and U on the torus N = 4, 6, 8
    (n_1 = 48, 108, 192) and on the small 2-complex, at the tolerances §14
    states: covariance <= 7.1e-14, transpose identity <= 2.4e-14, curvature
    formula (iv) to 5e-16, pure-gauge isospectrality to 2e-13 (F_B symmetry,
    <= 8.0e-12, is in test_pencil_schur_python.py)."""

    @pytest.mark.parametrize("N", [4, 6, 8])
    def test_exact_properties_random_complex(self, N):
        rng = np.random.default_rng(100 + N)
        cells, _ = torus_cells(N)
        K = cob.ChainComplex.fromTopCells(cells)
        assert K.numSimplices(1) == 3 * N * N
        s = _random_complex_lengths(K, rng)
        base = ch.ChainHodge(K, s, ch.Preset.L2, ch.Branch.KontsevichSegal)
        _assert_harmonic_dimension(base)
        U = ch.Connection(K, _random_links(K, rng))
        cov = ch.CovariantChainHodge(base, U)
        _assert_proposition_3(cov, K)
        cert = cov.certificate()
        assert cert.transposeMetric <= 2.4e-14
        assert cert.covarianceMetric <= 7.1e-14
        assert cert.curvature <= 5e-16  # (iv) at round-off on the scale of U_rp(F_t - 1)
        full = cov.verify(1)
        assert full.transposePencil <= 2.4e-14
        assert full.covariancePencil <= 7.1e-14
        assert full.pureGaugeIsospectrality <= 2e-13
        assert np.isnan(full.trivialReduction)  # U is not trivial here

    @pytest.mark.parametrize("cells", [TWO_COMPLEX, [[0, 1, 2, 3], [1, 2, 3, 4]], "torus33"])
    @pytest.mark.parametrize("seed", [1, 2])
    def test_every_instance_asserts_proposition_3(self, cells, seed):
        """Every construction measures and asserts (i)-(vi) at every degree;
        a trivial connection also reduces to L_k (i). An instance built
        without its certificate carries none."""
        rng = np.random.default_rng(seed)
        if cells == "torus33":
            from tests.chainhodge._fixtures import torus33
            K, s = torus33()
        else:
            K = cob.ChainComplex.fromTopCells(cells)
            s = _random_complex_lengths(K, rng)
        base = ch.ChainHodge(K, s, ch.Preset.L2, ch.Branch.KontsevichSegal)
        _assert_harmonic_dimension(base)
        _assert_proposition_3(ch.CovariantChainHodge(base, ch.Connection(K, _random_links(K, rng))), K)
        _assert_proposition_3(ch.CovariantChainHodge(base, ch.Connection.trivial(K)), K, trivial=True)
        grassmann = ch.ChainHodge(K, s, ch.Preset.GRASSMANN_ALL)
        _assert_proposition_3(ch.CovariantChainHodge(grassmann, ch.Connection(K, _random_links(K, rng))), K)
        bare = ch.CovariantChainHodge(base, ch.Connection(K, _random_links(K, rng)), 7, False)
        assert not bare.certificate().holds and np.isnan(bare.certificate().tolerance)

    def test_trivial_connection_reduces_to_l1(self):
        rng = np.random.default_rng(5)
        K = cob.ChainComplex.fromTopCells(TWO_COMPLEX)
        s = random_allowable(K, rng, 0.3)
        base = ch.ChainHodge(K, s)
        cov = ch.CovariantChainHodge(base, ch.Connection.trivial(K))
        full = cov.verify(1)
        assert full.trivialReduction <= 1e-13
        np.testing.assert_allclose(cov.pencil(1).A, base.pencil(1).A, atol=1e-13)
        np.testing.assert_allclose(cov.covariantOperator(1), base.hodgeOperator(1), atol=1e-12)

    def test_transpose_identity_on_h(self):
        """(ii) h_1(s,U)^T = G_1^{U^{-1}} h_1(s,U^{-1}) (G_1^{U^{-1}})^{-1}."""
        rng = np.random.default_rng(9)
        K = cob.ChainComplex.fromTopCells(TWO_COMPLEX)
        s = _random_complex_lengths(K, rng)
        base = ch.ChainHodge(K, s, ch.Preset.L2, ch.Branch.KontsevichSegal)
        U = ch.Connection(K, _random_links(K, rng))
        cov, dual = ch.CovariantChainHodge(base, U), ch.CovariantChainHodge(base, U.inverse())
        h = cov.covariantOperator(1)
        hd = dual.covariantOperator(1)
        Md = dual.Minv(1).toarray()
        rhs = np.linalg.solve(Md, hd @ Md)  # G^{U^-1} h^{U^-1} (G^{U^-1})^{-1} = M^{-1} h M
        np.testing.assert_allclose(h.T, rhs, atol=1e-11 * np.abs(h).max())
        np.testing.assert_allclose(cov.Minv(1).toarray().T, Md, atol=1e-15)

    def test_gauge_covariance_of_h_and_rho(self):
        """(iii) h_1(s,U^g) = rho_1(g) h_1(s,U) rho_1(g)^{-1}, rho_k(g) = diag(g_{b(sigma)}^{-1})."""
        rng = np.random.default_rng(11)
        K = cob.ChainComplex.fromTopCells(TWO_COMPLEX)
        s = _random_complex_lengths(K, rng)
        base = ch.ChainHodge(K, s, ch.Preset.L2, ch.Branch.KontsevichSegal)
        U = ch.Connection(K, _random_links(K, rng))
        g = _random_gauge(K, rng)
        cov = ch.CovariantChainHodge(base, U)
        rho = np.asarray(cov.rho(1, g)).ravel()
        expected = np.array([1.0 / g[min(e)] for e in edges(K)])
        np.testing.assert_allclose(rho, expected, rtol=1e-15)
        h = cov.covariantOperator(1)
        hg = cov.gauged(g).covariantOperator(1)
        np.testing.assert_allclose(hg, np.diag(rho) @ h @ np.diag(1.0 / rho), atol=1e-11 * np.abs(h).max())

    def test_curvature_and_no_flatness(self):
        """(iv) d_1^U d_2^U t = U_rp (F_t - 1)[r]; the twisted differential does not square to zero."""
        rng = np.random.default_rng(13)
        K = cob.ChainComplex.fromTopCells(TWO_COMPLEX)
        s = random_allowable(K, rng)
        U = ch.Connection(K, _random_links(K, rng))
        cov = ch.CovariantChainHodge(ch.ChainHodge(K, s), U)
        C = (cov.twistedBoundary(1) @ cov.twistedBoundary(2)).toarray()
        verts = [int(v[0]) for v in K.kSimplexVertices(0)]
        for t, tri in enumerate(K.kSimplexVertices(2)):
            p, q, r = (int(v) for v in tri)
            col = C[:, t].copy()
            expected = U.link(r, p) * (U.curvature(p, q, r) - 1.0)
            assert col[verts.index(r)] == pytest.approx(expected, abs=1e-14)
            col[verts.index(r)] = 0.0
            assert np.linalg.norm(col) < 1e-15
        assert np.linalg.norm(C) > 1e-3
        assert cov.certificate().curvature < 1e-14

    def test_non_unit_links_are_retained_and_operator_stays_non_normal(self):
        rng = np.random.default_rng(17)
        K = cob.ChainComplex.fromTopCells(TWO_COMPLEX)
        s = random_allowable(K, rng)
        links = [2.0 * u for u in _random_links(K, rng)]
        U = ch.Connection(K, links)
        cov = ch.CovariantChainHodge(ch.ChainHodge(K, s), U)
        assert list(cov.connection().links()) == links
        h = cov.covariantOperator(1)
        assert np.linalg.norm(h @ h.conj().T - h.conj().T @ h) > 1e-6 * np.linalg.norm(h) ** 2


class TestGrassmannDressing:
    def test_dressed_chain_metric_and_pencil_on_chains(self):
        rng = np.random.default_rng(19)
        K = cob.ChainComplex.fromTopCells(TWO_COMPLEX)
        s = _random_complex_lengths(K, rng)
        base = ch.ChainHodge(K, s, ch.Preset.GRASSMANN_ALL)
        U = ch.Connection(K, _random_links(K, rng))
        cov = ch.CovariantChainHodge(base, U)
        assert cov.certificate().transposeMetric <= 1e-14
        assert cov.certificate().covarianceMetric <= 1e-13
        P = cov.pencil(1)
        assert P.variable == ch.PencilVariable.Chain
        with pytest.raises(RuntimeError):
            cov.Minv(1)
        full = cov.verify(1)
        assert full.transposePencil <= 1e-12 and full.pureGaugeIsospectrality <= 1e-11


class TestBaseVertexConvention:
    def test_base_vertex_is_minimum_and_dressing_uses_single_links(self):
        rng = np.random.default_rng(23)
        K = cob.ChainComplex.fromTopCells(TWO_COMPLEX)
        s = random_allowable(K, rng)
        U = ch.Connection(K, _random_links(K, rng))
        base = ch.ChainHodge(K, s)
        cov = ch.CovariantChainHodge(base, U)
        M = base.Minv(1).toarray()
        MU = cov.Minv(1).toarray()
        es = edges(K)
        for i, e in enumerate(es):
            for j, f in enumerate(es):
                if M[i, j] != 0:
                    assert MU[i, j] == pytest.approx(M[i, j] * U.link(min(e), min(f)), rel=1e-14)
        B = base.boundary(1).toarray()
        BU = cov.twistedBoundary(1).toarray()
        verts = [int(v[0]) for v in K.kSimplexVertices(0)]
        for i, v in enumerate(verts):
            for j, e in enumerate(es):
                if B[i, j] != 0:
                    assert BU[i, j] == pytest.approx(B[i, j] * U.link(v, min(e)), rel=1e-14)
