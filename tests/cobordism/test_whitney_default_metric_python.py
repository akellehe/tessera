# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The Whitney pencil with its connection as the default metric of the
pipeline (#1185).

Sections 3.1 and 3.2 of the whitepaper fix the metric weights W_k = M_k^-1 of
the Whitney pencil and read the covariant operator h_k(z, U) throughout. Every
pipeline entry point -- HodgeLaplacian, SpectralFiberTracker, MultiCobordism
and RecursiveQuotient -- therefore reads that operator unless the diagonal
weights are named. The tests below change the connection U (the edge phases)
of one fixed geometry and see each entry point's default spectrum move, check
that a pure-gauge U leaves it where it was, and check that the diagonal
weights, named, stay blind to U. RecursiveQuotient.overCells pairs its operator
and its metric from one source."""
import cmath

import numpy as np
import pytest

import tessera
from tessera import chainhodge as ch
from tessera import cobordism as cob
from tessera import observables as obs

HL = cob.HodgeLaplacian
MC = cob.MultiCobordism
Whitney = cob.HodgeMetricSource.WhitneyPencil
Diagonal = cob.HodgeMetricSource.DiagonalWeights
PS = ch.PencilSchur

# The boundary of a tetrahedron, with a fifth vertex coned onto the edge (2, 3).
TWO_COMPLEX = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3], [2, 3, 4]]
VERTICES = [0, 1, 2, 3, 4]


def _spacetime():
    """One fixed Euclidean geometry with the trivial connection."""
    rng = np.random.default_rng(11)
    st = tessera.Spacetime.fromVertexTuples(2, TWO_COMPLEX, 1.0, 0.0)
    for e in st.getEdgeList().toVector():
        e.setLength(cmath.sqrt(1.0 + 0.3 * rng.random()))
        e.setPhase(0.0)
    return st


def _twist(st, seed=5):
    """A connection with nonzero flux through the triangles: random edge
    phases, so U = exp(i phi) is not a gauge transform of U = 1."""
    rng = np.random.default_rng(seed)
    for e in st.getEdgeList().toVector():
        e.setPhase(complex(0.8 * rng.normal(), 0.1 * rng.normal()))


def _pure_gauge(st, seed=9):
    """U_xy = exp(i (chi_y - chi_x)): a gauge transform of U = 1."""
    rng = np.random.default_rng(seed)
    chi = {v: rng.normal() for v in VERTICES}
    for e in st.getEdgeList().toVector():
        a, b = e.getSource().getId(), e.getTarget().getId()
        e.setPhase(chi[b] - chi[a])


def _spectrum(values):
    return np.sort_complex(np.asarray(values, dtype=complex))


def _moved(before, after):
    """Hausdorff distance between two spectra of the same size."""
    return max(np.min(np.abs(after - z)) for z in before)


class TestDefaults:
    def test_every_entry_point_defaults_to_the_whitney_pencil(self):
        assert HL.defaultMetricSource() == Whitney
        st = _spacetime()
        assert HL(st).metricSource() == Whitney
        assert HL(st, HL.defaultWeightConvention()).metricSource() == Whitney
        assert obs.SpectralFiberTracker(st).metricSource() == Whitney
        assert obs.SpectralFiberTracker(st, obs.SpectralFiberConfig(), HL.defaultWeightConvention()).metricSource() == Whitney
        assert MC(st, [], [], [1]).metricSource() == Whitney
        assert cob.EigenstateSynthesis(st, 1).metricSource() == Whitney
        cells = cob.ChainComplex.fromSpacetime(st).kSimplexVertices(1)
        q = cob.RecursiveQuotient.overCells(st, 1, [[list(c) for c in cells]])
        assert q.metricSource() == Whitney and q.isPencil()

    def test_the_diagonal_weights_stay_reachable_by_name(self):
        st = _spacetime()
        assert HL(st, HL.defaultWeightConvention(), Diagonal).metricSource() == Diagonal
        assert obs.SpectralFiberTracker(st, obs.SpectralFiberConfig(), Diagonal).metricSource() == Diagonal
        assert obs.SpectralFiberTracker(st, metric_source=Diagonal).metricSource() == Diagonal
        assert MC(st, [], [], [1], metric_source=Diagonal).metricSource() == Diagonal
        cells = cob.ChainComplex.fromSpacetime(st).kSimplexVertices(1)
        q = cob.RecursiveQuotient.overCells(st, 1, [[list(c) for c in cells]], metric_source=Diagonal)
        assert q.metricSource() == Diagonal and not q.isPencil()

    def test_the_process_default_is_read_at_call_time(self):
        st = _spacetime()
        previous = HL.defaultMetricSource()
        HL.setDefaultMetricSource(Diagonal)
        try:
            assert HL(st).metricSource() == Diagonal
            assert obs.SpectralFiberTracker(st).metricSource() == Diagonal
            assert MC(st, [], [], [1]).metricSource() == Diagonal
        finally:
            HL.setDefaultMetricSource(previous)
        assert HL(st).metricSource() == Whitney


class TestHodgeLaplacianMovesWithU:
    @pytest.mark.parametrize("k", [0, 1, 2])
    def test_changing_u_moves_the_default_spectrum(self, k):
        st = _spacetime()
        trivial = _spectrum(HL(st).eigenvalues(k))
        _twist(st)
        twisted = _spectrum(HL(st).eigenvalues(k))
        assert _moved(trivial, twisted) > 1e-3 * max(1.0, np.abs(trivial).max())

    @pytest.mark.parametrize("k", [0, 1, 2])
    def test_a_pure_gauge_u_leaves_it(self, k):
        st = _spacetime()
        trivial = _spectrum(HL(st).eigenvalues(k))
        _pure_gauge(st)
        gauged = _spectrum(HL(st).eigenvalues(k))
        assert _moved(trivial, gauged) < 1e-9 * max(1.0, np.abs(trivial).max())

    def test_the_named_diagonal_weights_do_not_see_u(self):
        st = _spacetime()
        trivial = _spectrum(HL(st, HL.defaultWeightConvention(), Diagonal).eigenvalues(1))
        _twist(st)
        twisted = _spectrum(HL(st, HL.defaultWeightConvention(), Diagonal).eigenvalues(1))
        np.testing.assert_allclose(twisted, trivial, atol=1e-12)

    def test_the_default_operator_is_the_dressed_pencil(self):
        """laplacian(k) = M^-1 A~ with (A~, M) = pencil(k), the covariant
        operator of the chainhodge objects at the spacetime's connection."""
        st = _spacetime()
        _twist(st)
        hl = HL(st)
        K = ch.WhitneyMass.complexOf(st)
        cov = ch.CovariantChainHodge(ch.ChainHodge(K, ch.WhitneyMass.squaredLengthsOf(st, K)),
                                     ch.Connection.fromSpacetime(st, K))
        n = K.numSimplices(1)
        A, M = (np.asarray(x, dtype=complex).reshape(n, n) for x in hl.pencil(1))
        L = np.asarray(hl.laplacian(1), dtype=complex).reshape(n, n)
        np.testing.assert_allclose(M @ L, A, atol=1e-11 * np.abs(A).max())
        np.testing.assert_allclose(_spectrum(np.linalg.eigvals(L)),
                                   _spectrum(np.linalg.eigvals(cov.covariantOperator(1))),
                                   atol=1e-9 * max(1.0, np.abs(L).max()))

    def test_the_pencil_is_the_whitney_sources(self):
        st = _spacetime()
        with pytest.raises(RuntimeError, match="DiagonalWeights"):
            HL(st, HL.defaultWeightConvention(), Diagonal).pencil(1)

    def test_the_entropy_gradient_is_taken_of_the_default_operator(self):
        """The default spectral-entropy gradient differentiates the Whitney
        operator the entropy is taken of: its real part is the directional
        derivative of the entropy along Re z, checked by central differences."""
        st = _spacetime()
        _twist(st)
        k = 1
        grad = np.asarray(HL(st).spectralEntropyGradient(k), dtype=complex)
        edges = st.getEdgeList().toVector()
        step = 1e-6
        for index in (0, 3, len(edges) - 1):
            e = edges[index]
            s0 = e.getLength() ** 2
            e.setLength(cmath.sqrt(s0 + step))
            up = HL(st).spectralEntropy(k)
            e.setLength(cmath.sqrt(s0 - step))
            down = HL(st).spectralEntropy(k)
            e.setLength(cmath.sqrt(s0))
            assert grad[index].real == pytest.approx((up - down) / (2 * step), rel=1e-4, abs=1e-8)

    @pytest.mark.parametrize("mode", [cob.HodgeEntropyPhaseMode.IncludeComplexPhase,
                                      cob.HodgeEntropyPhaseMode.IgnoreComplexPhase])
    def test_the_entropy_directional_derivative_is_taken_of_the_default_operator(self, mode):
        """d/dt of the default entropy gradient at z + t v (the Hessian-vector
        product the joint-stationarity direction descends along) against
        central differences of the gradient itself."""
        st = _spacetime()
        _twist(st)
        k = 1
        edges = st.getEdgeList().toVector()
        rng = np.random.default_rng(23)
        v = rng.normal(size=len(edges)) + 1j * rng.normal(size=len(edges))
        got = np.asarray(HL(st).spectralEntropyGradientDirectionalDerivative(k, list(v), mode), dtype=complex)
        s0 = [e.getLength() ** 2 for e in edges]
        step = 1e-6

        def gradient_at(t):
            for e, s, ve in zip(edges, s0, v):
                e.setLength(cmath.sqrt(s + t * ve))
            return np.asarray(HL(st).spectralEntropyGradient(k, mode), dtype=complex)

        fd = (gradient_at(step) - gradient_at(-step)) / (2 * step)
        gradient_at(0.0)
        np.testing.assert_allclose(got, fd, atol=1e-5 * max(1.0, np.abs(fd).max()))


class TestTrackerMovesWithU:
    def _band_values(self, st, **kwargs):
        cfg = obs.SpectralFiberConfig()
        cfg.degrees = [1]
        read = obs.SpectralFiberTracker(st, cfg, **kwargs).enumerateBands(VERTICES, 1)
        return read, _spectrum([z for f in read.fibers for z in f.eigenvalues()])

    def test_changing_u_moves_the_default_bands(self):
        st = _spacetime()
        read, trivial = self._band_values(st)
        assert read.solverPath == "pencil-riesz"
        assert read.regime == cob.CertificateRegime.ComplexSymmetricPencil
        _twist(st)
        read, twisted = self._band_values(st)
        assert read.solverPath == "pencil-riesz"
        assert len(twisted) == len(trivial)
        assert _moved(trivial, twisted) > 1e-3 * max(1.0, np.abs(trivial).max())

    def test_the_bands_are_those_of_the_covariant_operator(self):
        st = _spacetime()
        _twist(st)
        _, values = self._band_values(st)
        K = ch.WhitneyMass.complexOf(st)
        cov = ch.CovariantChainHodge(ch.ChainHodge(K, ch.WhitneyMass.squaredLengthsOf(st, K)),
                                     ch.Connection.fromSpacetime(st, K))
        expected = _spectrum(np.linalg.eigvals(cov.covariantOperator(1)))
        np.testing.assert_allclose(values, expected, atol=1e-8 * max(1.0, np.abs(expected).max()))

    def test_the_named_diagonal_bands_do_not_see_u(self):
        st = _spacetime()
        _, trivial = self._band_values(st, metric_source=Diagonal)
        _twist(st)
        _, twisted = self._band_values(st, metric_source=Diagonal)
        np.testing.assert_allclose(twisted, trivial, atol=1e-10)


class TestMultiCobordismMovesWithU:
    def test_changing_u_moves_the_nodes_hodge_entropy(self):
        st = _spacetime()
        node = MC(st, [], [], [1])
        node.set_hodge_degrees([1])
        trivial = node.hodge_entropy()
        _twist(st)
        twisted = node.hodge_entropy()
        assert abs(twisted - trivial) > 1e-6

    def test_the_analysis_pass_reads_the_whitney_bands(self):
        """runRecursiveAnalysis builds its tracker, quotient and transports on
        the node's metric source: its band windows move with U."""
        import json

        def windows(st):
            node = MC(st, [], [], [1])
            node.run_recursive_analysis()
            record = json.loads(node.checkpoint_json)
            text = json.dumps(record)
            return record, text

        st = _spacetime()
        record, text = windows(st)
        assert "complex-symmetric-pencil" in text or "ComplexSymmetricPencil" in text
        _twist(st)
        twisted_record, twisted_text = windows(st)
        assert twisted_text != text


class TestRecursiveQuotientPairsOneSource:
    def _components(self, st, k):
        cells = [list(c) for c in cob.ChainComplex.fromSpacetime(st).kSimplexVertices(k)]
        left = [c for c in cells if set(c) <= {0, 1, 2, 3}]
        right = [c for c in cells if set(c) <= {2, 3, 4}]
        return left, right

    @pytest.mark.parametrize("k", [0, 1])
    def test_default_level_is_the_whitney_pencil_of_the_same_operator(self, k):
        st = _spacetime()
        _twist(st)
        left, right = self._components(st, k)
        q = cob.RecursiveQuotient.overCells(st, k, [left, right])
        assert q.isPencil() and q.metricSource() == Whitney
        n = q.dimension
        A, M = (np.asarray(x, dtype=complex).reshape(n, n) for x in HL(st).pencil(k))
        np.testing.assert_allclose(np.asarray(q.pencilMetric(), dtype=complex).reshape(n, n), M, atol=1e-15)
        interface = list(q.interfaceIndices)
        lam = complex(0.7, 0.1)
        read = q.feshbach(lam, -1.0, 2.0)
        F = PS.feshbach(A, M, lam, interface)
        m = len(interface)
        np.testing.assert_allclose(np.asarray(read.response, dtype=complex).reshape(m, m), F.response,
                                   atol=1e-9 * max(1.0, np.abs(F.response).max()))

    def test_changing_u_moves_the_reduction(self):
        st = _spacetime()
        left, right = self._components(st, 1)
        before = np.asarray(cob.RecursiveQuotient.overCells(st, 1, [left, right]).staticReduction().effectiveOperator,
                            dtype=complex)
        _twist(st)
        after = np.asarray(cob.RecursiveQuotient.overCells(st, 1, [left, right]).staticReduction().effectiveOperator,
                           dtype=complex)
        assert before.shape == after.shape
        assert np.abs(after - before).max() > 1e-3 * max(1.0, np.abs(before).max())

    def test_diagonal_level_pairs_the_diagonal_operator_with_its_weights(self):
        st = _spacetime()
        left, right = self._components(st, 1)
        q = cob.RecursiveQuotient.overCells(st, 1, [left, right], metric_source=Diagonal)
        assert not q.isPencil() and q.pencilMetric() == []
        assert q.metricSource() == Diagonal
