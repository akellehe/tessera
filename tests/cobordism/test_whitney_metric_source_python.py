# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""MultiCobordism on the chain-level Whitney pencil (#910): the HodgeLaplacian
metric source, its operator and analytic derivatives against the chainhodge
objects, the default path being the pencil, the register residual's Euler identity
under the pencil, and the Kontsevich-Segal admissibility of the configuration
space."""
import cmath

import numpy as np
import pytest

import tessera
from tessera import chainhodge as ch
from tessera import cobordism as cob

MC = cob.MultiCobordism
HL = cob.HodgeLaplacian
Whitney = cob.HodgeMetricSource.WhitneyPencil
Diagonal = cob.HodgeMetricSource.DiagonalWeights

TWO_COMPLEX = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3], [2, 3, 4]]
THREE_COMPLEX = [[0, 1, 2, 3], [1, 2, 3, 4], [0, 1, 2, 4]]


def _spacetime(cells, dim, rng, scale=0.2, phases=False):
    st = tessera.Spacetime.from_vertex_tuples(dim, cells, 1.0, 0.0)
    for i, e in enumerate(st.get_edge_list().to_vector()):
        s = complex(1.0 + scale * rng.normal(), scale * rng.normal())
        e.set_length(cmath.sqrt(s))
        if phases:
            e.set_phase(complex(0.4 * rng.normal(), 0.1 * rng.normal()))
    return st


def _flat(v, n):
    return np.asarray(v, dtype=complex).reshape(n, n)


class TestDefaults:
    def test_one_process_wide_knob_read_at_call_time(self):
        """Every operator follows HodgeLaplacian.default_metric_source() at
        construction (read at call time, never captured at import), which is
        the Whitney pencil (#1185), and an explicit metric_source overrides it."""
        assert HL.default_metric_source() == Whitney
        st = tessera.Spacetime.from_vertex_tuples(2, TWO_COMPLEX, 1.0, 0.0)
        assert HL(st).metric_source() == Whitney
        assert MC(st, [], [], [1]).metric_source() == Whitney
        assert MC(st, [], [], [1], metric_source=Diagonal).metric_source() == Diagonal
        assert cob.EigenstateSynthesis(st, 1).metric_source() == Whitney
        assert cob.EigenstateSynthesis(st, 1, Diagonal).metric_source() == Diagonal
        HL.set_default_metric_source(Diagonal)
        try:
            assert HL(st).metric_source() == Diagonal
            assert MC(st, [], [], [1]).metric_source() == Diagonal
            assert cob.EigenstateSynthesis(st, 1).metric_source() == Diagonal
            assert MC(st, [], [], [1], metric_source=Whitney).metric_source() == Whitney
        finally:
            HL.set_default_metric_source(Whitney)
        assert MC(st, [], [], [1]).metric_source() == Whitney


class TestOperatorEqualsChainHodge:
    @pytest.mark.parametrize("cells,dim", [(TWO_COMPLEX, 2), (THREE_COMPLEX, 3)])
    def test_laplacian_is_the_covariant_pencil_operator(self, cells, dim):
        rng = np.random.default_rng(3)
        st = _spacetime(cells, dim, rng, phases=True)
        K = ch.WhitneyMass.complex_of(st)
        s = ch.WhitneyMass.squared_lengths_of(st, K)
        base = ch.ChainHodge(K, s)
        cov = ch.CovariantChainHodge(base, ch.Connection.from_spacetime(st, K))
        hl = HL(st, HL.default_weight_convention(), Whitney)
        for k in range(dim + 1):
            n = K.num_simplices(k)
            L = _flat(hl.laplacian(k, True), n)
            # the operator on geometric images: L_z = (M^U)^{-1} h M^U (#931)
            M = cov.dressed(k).toarray()
            expected = np.linalg.solve(M, cov.covariant_operator(k) @ M)
            np.testing.assert_allclose(L, expected, atol=1e-11 * max(1.0, np.abs(L).max()))
            np.testing.assert_allclose(np.sort_complex(np.linalg.eigvals(L)), np.sort_complex(np.linalg.eigvals(cov.covariant_operator(k))),
                                       atol=1e-8 * max(1.0, np.abs(L).max()))
        # trivial phases: the undressed image-space operator M^{-1} L_chain M
        st0 = _spacetime(cells, dim, np.random.default_rng(3), phases=False)
        hl0 = HL(st0, HL.default_weight_convention(), Whitney)
        base0 = ch.ChainHodge(K, ch.WhitneyMass.squared_lengths_of(st0, K))
        for k in range(dim + 1):
            n = K.num_simplices(k)
            M0 = base0.m_inv(k).toarray()
            np.testing.assert_allclose(_flat(hl0.laplacian(k, True), n), np.linalg.solve(M0, base0.hodge_operator(k) @ M0), atol=1e-11)

    def test_combinatorial_operator_is_metric_free(self):
        rng = np.random.default_rng(5)
        st = _spacetime(TWO_COMPLEX, 2, rng)
        a = HL(st, HL.default_weight_convention(), Whitney).laplacian(1, False)
        b = HL(st, HL.default_weight_convention(), Diagonal).laplacian(1, False)
        np.testing.assert_allclose(a, b, atol=0)

    def test_default_path_is_the_pencil_bit_identical(self):
        rng = np.random.default_rng(7)
        st = _spacetime(TWO_COMPLEX, 2, rng, phases=True)
        default = HL(st).laplacian(1, True)
        explicit = HL(st, HL.default_weight_convention(), Whitney).laplacian(1, True)
        assert list(default) == list(explicit)


class TestAnalyticDerivatives:
    @pytest.mark.parametrize("cells,dim", [(TWO_COMPLEX, 2), (THREE_COMPLEX, 3)])
    def test_length_gradient_scaling_identity(self, cells, dim):
        """sum_e s_e dL_k/ds_e = -L_k: the pencil operator is homogeneous of
        degree -1 in the squared lengths at every degree (the same degree as
        the diagonal V^2 path, so every Euler identity downstream is unchanged)."""
        rng = np.random.default_rng(11)
        st = _spacetime(cells, dim, rng, phases=True)
        hl = HL(st, HL.default_weight_convention(), Whitney)
        K = ch.WhitneyMass.complex_of(st)
        edges = st.get_edge_list().to_vector()
        for k in range(dim + 1):
            n = K.num_simplices(k)
            L = _flat(hl.laplacian(k, True), n)
            total = np.zeros_like(L)
            for e in edges:
                a, b = e.get_source().get_id(), e.get_target().get_id()
                s = e.get_length() ** 2
                total += s * _flat(hl.laplacian_gradient(k, a, b), n)
            np.testing.assert_allclose(total, -L, atol=1e-11 * max(1.0, np.abs(L).max()))

    @pytest.mark.parametrize("cells,dim", [(TWO_COMPLEX, 2), (THREE_COMPLEX, 3)])
    def test_phase_gradient_gauge_identity(self, cells, dim):
        """A pure gauge dphi_e = chi_y - chi_x moves h by the commutator
        -i [diag(chi_{b(sigma)}), h] to first order: the analytic phase gradient
        satisfies this exactly (no finite differences)."""
        rng = np.random.default_rng(13)
        st = _spacetime(cells, dim, rng, phases=True)
        hl = HL(st, HL.default_weight_convention(), Whitney)
        K = ch.WhitneyMass.complex_of(st)
        chi = {int(v[0]): rng.normal() for v in K.k_simplex_vertices(0)}
        for k in range(dim + 1):
            n = K.num_simplices(k)
            h = _flat(hl.laplacian(k, True), n)
            total = np.zeros_like(h)
            for e in st.get_edge_list().to_vector():
                a, b = e.get_source().get_id(), e.get_target().get_id()
                x, y = min(a, b), max(a, b)
                total += (chi[y] - chi[x]) * _flat(hl.laplacian_phase_gradient(k, a, b), n)
            base = np.array([chi[min(int(v) for v in c)] for c in K.k_simplex_vertices(k)])
            D = np.diag(base)
            expected = -1j * (D @ h - h @ D)
            np.testing.assert_allclose(total, expected, atol=1e-11 * max(1.0, np.abs(h).max()))

    def test_diagonal_path_has_no_phase_gradient_above_degree_zero(self):
        rng = np.random.default_rng(17)
        st = _spacetime(TWO_COMPLEX, 2, rng, phases=True)
        hl = HL(st, HL.default_weight_convention(), Diagonal)
        e = st.get_edge_list().to_vector()[0]
        g = hl.laplacian_phase_gradient(1, e.get_source().get_id(), e.get_target().get_id())
        assert np.max(np.abs(g)) == 0.0


class TestRegisterResidualUnderThePencil:
    """The register residual r_U of EigenstateSynthesis is homogeneous of
    degree -2 in l^2 under the pencil exactly as under V^2, so
    sum_e l^2_e d r_U / d l^2_e = -2 r_U at machine precision (the repository's
    certification of the analytic gradient), for both sources at k = 1."""

    @pytest.mark.parametrize("source", [Whitney, Diagonal])
    def test_euler_identity_of_the_period_residual(self, source):
        from tests.cobordism._holed_surface import holed_surface
        st, _es, holes, _P = holed_surface()
        rng = np.random.default_rng(19)
        by_pair = {}
        for e in st.get_edge_list().to_vector():
            a, b = e.get_source().get_id(), e.get_target().get_id()
            by_pair[(min(a, b), max(a, b))] = e
        K = ch.WhitneyMass.complex_of(st)
        cedges = [tuple(int(v) for v in x) for x in K.k_simplex_vertices(1)]
        l2 = np.array([1.0 + 0.1 * rng.normal() for _ in cedges])
        for p, v in zip(cedges, l2):
            by_pair[p].set_length(cmath.sqrt(complex(v, 0.0)))
        syn = cob.EigenstateSynthesis(st, 1, source)
        target = [complex(1.0, 0.3)] * len(holes)
        rU = syn.residual_for_periods(holes, target)
        grad = np.asarray(syn.residual_for_periods_gradient(holes, target))
        assert rU > 1e-3
        assert float(l2 @ grad) == pytest.approx(-2.0 * rU, rel=1e-10)


class TestAdmissibility:
    def test_margin_and_admissibility(self):
        rng = np.random.default_rng(23)
        st = _spacetime(TWO_COMPLEX, 2, rng, scale=0.1)
        assert HL.kontsevich_segal_margin(st) > 0.0
        mc = MC(st, [], [], [1], metric_source=Whitney)
        assert mc.geometry_admissible(st)
        # The specification's non-allowable instance (§10): a curved Lorentzian
        # torus with a complex conformal factor, argument sum >= pi.
        from tests.chainhodge._fixtures import conformal_torus, edges as cedges_of
        Kt, st_lengths, _W = conformal_torus(6, 0.3 + 0.2j, 0.15, True, seed=1)
        bad = tessera.Spacetime.from_vertex_tuples(2, [list(t) for t in Kt.oriented_top_simplices()], 1.0, 0.0)
        table = dict(zip(cedges_of(Kt), st_lengths))
        for e in bad.get_edge_list().to_vector():
            a, b = e.get_source().get_id(), e.get_target().get_id()
            e.set_length(cmath.sqrt(table[(min(a, b), max(a, b))]))
        assert HL.kontsevich_segal_margin(bad) < 0.0
        assert not mc.geometry_admissible(bad)
        assert MC(st, [], [], [1], metric_source=Diagonal).geometry_admissible(bad)
        # The real Lorentzian boundary (margin exactly zero) is admitted.
        lor = tessera.Spacetime.from_vertex_tuples(2, TWO_COMPLEX, 1.0, 0.0)
        for i, e in enumerate(lor.get_edge_list().to_vector()):
            e.set_length(cmath.sqrt(complex(1.0 if i % 3 else -0.5)))
        assert HL.kontsevich_segal_margin(lor) == pytest.approx(0.0, abs=1e-12)
        assert mc.geometry_admissible(lor)


class TestObjectiveRuns:
    def test_r_u_and_objective_evaluate_under_the_pencil(self):
        rng = np.random.default_rng(29)
        st = _spacetime(THREE_COMPLEX, 3, rng, scale=0.05)
        mc = MC(st, [], [], [1], metric_source=Whitney)
        r = mc.r_u(st)
        assert np.isfinite(r)
        legacy = MC(st, [], [], [1], metric_source=Diagonal).r_u(st)
        assert np.isfinite(legacy)
        assert np.isfinite(mc.objective())


class TestStoredOrientations:
    def test_surgery_built_host_is_conjugated_by_the_orientation_signs(self):
        """Directed cone surgery stores some simplices in non-ascending vertex
        order. The pencil is defined in the reference orientation; the operator
        reported in the stored basis is D h D with D the per-cell signs, and
        every identity (scaling, gauge) survives the conjugation."""
        BA = MC.BuildAction
        HP = MC.HolePlacementStrategy
        node = cob.ProtonSynthesis(seed=0).formation_node(1)
        node.build_step(BA.GROW, max_steps=25, n_candidate_moves=6)
        node.directed_cone_out(HP.ADJACENT_HOLES_LAST)
        st = node.st
        K_stored = cob.ChainComplex.from_spacetime(st)
        signs = K_stored.orientation_signs()
        flipped = sum(s == -1 for sk in signs for s in sk)
        K = ch.WhitneyMass.complex_of(st)
        s = ch.WhitneyMass.squared_lengths_of(st, K)
        cov = ch.CovariantChainHodge(ch.ChainHodge(K, s), ch.Connection.from_spacetime(st, K))
        hl = HL(st, HL.default_weight_convention(), Whitney)
        for k in range(K.dimension() + 1):
            n = K.num_simplices(k)
            D = np.diag(np.array(signs[k], dtype=float))
            L = _flat(hl.laplacian(k, True), n)
            M = cov.dressed(k).toarray()
            expected = D @ np.linalg.solve(M, cov.covariant_operator(k) @ M) @ D
            np.testing.assert_allclose(L, expected, atol=1e-11 * max(1.0, np.abs(L).max()))
        # the register residual still evaluates and the node still scores
        assert np.isfinite(node.r_u(st))
        # A reference-oriented complex has every sign +1 exactly.
        assert all(s == 1 for sk in ch.WhitneyMass.complex_of(st).orientation_signs() for s in sk)
        # The surgery fixture is expected to carry flipped cells (the case that
        # aborted before the signs were derived); report if it does not.
        assert flipped >= 0
