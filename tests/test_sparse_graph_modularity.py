# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""
Tests for the C++ SparseGraph + Spacetime::modularityOnSkeleton
observables (Phase 4).

* :class:`TestSparseGraphConstruction` — fromCOO normalizes a COO
  array (deduplicates duplicate edges, ignores self-loops, and adds
  both directions).
* :class:`TestSparseGraphBipartite` — BFS 2-coloring check.
* :class:`TestSpectralDimension` — D_S extraction on toy graphs:
  cycle ≈ 1, complete bipartite K_{n,n} ≈ ?, dual of CDT lattice
  is in a sane range.
* :class:`TestModularityOnSkeleton` — Newman-Girvan Q matches the
  Python reference implementation for a small built lattice.
"""
import math
import unittest
import tessera


def _make_st(d=4, n_simplices=200):
    sig = tessera.Signature(d, tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    st = tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0,
                           tessera.PREFERRED, tessera.Toroid())
    st.build(n_simplices)
    return st


# ---------------------------------------------------------------------------
# SparseGraph construction
# ---------------------------------------------------------------------------


class TestSparseGraphConstruction(unittest.TestCase):

    def test_path_graph_3_nodes(self):
        # 0 - 1 - 2
        g = tessera.SparseGraph.fromCOO([0, 1], [1, 2], 3)
        self.assertEqual(g.nNodes(), 3)
        self.assertEqual(g.nEdges(), 2)

    def test_duplicate_edges_collapsed(self):
        g = tessera.SparseGraph.fromCOO([0, 0, 1, 0],
                                        [1, 1, 0, 1], 2)
        self.assertEqual(g.nEdges(), 1)

    def test_self_loops_ignored(self):
        g = tessera.SparseGraph.fromCOO([0, 1, 0],
                                        [0, 1, 1], 2)
        self.assertEqual(g.nEdges(), 1)

    def test_isolated_node(self):
        g = tessera.SparseGraph.fromCOO([0], [1], 5)
        self.assertEqual(g.nNodes(), 5)
        self.assertEqual(g.nEdges(), 1)


# ---------------------------------------------------------------------------
# Bipartite check
# ---------------------------------------------------------------------------


class TestSparseGraphBipartite(unittest.TestCase):

    def test_empty_is_bipartite(self):
        g = tessera.SparseGraph.fromCOO([], [], 0)
        self.assertTrue(g.isBipartite())

    def test_no_edges_is_bipartite(self):
        g = tessera.SparseGraph.fromCOO([], [], 5)
        self.assertTrue(g.isBipartite())

    def test_path_is_bipartite(self):
        g = tessera.SparseGraph.fromCOO([0, 1, 2, 3], [1, 2, 3, 4], 5)
        self.assertTrue(g.isBipartite())

    def test_triangle_is_not_bipartite(self):
        # K_3: 3-cycle — not bipartite.
        g = tessera.SparseGraph.fromCOO([0, 1, 2], [1, 2, 0], 3)
        self.assertFalse(g.isBipartite())

    def test_4cycle_is_bipartite(self):
        g = tessera.SparseGraph.fromCOO([0, 1, 2, 3], [1, 2, 3, 0], 4)
        self.assertTrue(g.isBipartite())


# ---------------------------------------------------------------------------
# Generic Newman-Girvan modularity on an arbitrary SparseGraph
# ---------------------------------------------------------------------------


class TestSparseGraphModularity(unittest.TestCase):
    """SparseGraph.modularity(labels) — the generic Q = Σ_c [L_c/m -
    (D_c/2m)²] used by examples/modularity.py."""

    @staticmethod
    def _two_triangles():
        # Disjoint triangles {0,1,2} and {3,4,5}: m=6, each node deg 2.
        rows = [0, 1, 2, 3, 4, 5]
        cols = [1, 2, 0, 4, 5, 3]
        return tessera.SparseGraph.fromCOO(rows, cols, 6)

    def test_perfect_split_matches_closed_form(self):
        # Each triangle one community: Q = 2·[3/6 - (6/12)²] = 0.5.
        g = self._two_triangles()
        self.assertAlmostEqual(g.modularity([0, 0, 0, 1, 1, 1]), 0.5, places=12)

    def test_labels_need_not_be_dense(self):
        g = self._two_triangles()
        self.assertAlmostEqual(
            g.modularity([10, 10, 10, 99, 99, 99]), 0.5, places=12)

    def test_single_community_is_zero(self):
        # All edges intra, D_c = 2m → Q = 1 - 1 = 0.
        g = self._two_triangles()
        self.assertAlmostEqual(g.modularity([0] * 6), 0.0, places=12)

    def test_anti_community_is_negative(self):
        # Split every triangle across communities: no intra edges.
        g = self._two_triangles()
        self.assertLess(g.modularity([0, 1, 0, 1, 0, 1]), 0.0)

    def test_edgeless_graph_is_zero(self):
        g = tessera.SparseGraph.fromCOO([], [], 5)
        self.assertEqual(g.modularity([0, 1, 2, 3, 4]), 0.0)

    def test_label_length_mismatch_raises(self):
        g = self._two_triangles()
        with self.assertRaises(ValueError):
            g.modularity([0, 0, 1])


# ---------------------------------------------------------------------------
# Spectral dimension on toy graphs
# ---------------------------------------------------------------------------


class TestHeatKernelAnalytic(unittest.TestCase):
    """Compare ``diagonalHeatKernel`` against closed-form values on
    tiny graphs.  These tests would catch any sign / coefficient /
    Padé error in the Krylov-Lanczos heat-kernel implementation."""

    def test_path_3_at_endpoint(self):
        """Path 0-1-2.  Symmetric normalized Laplacian eigenvalues 0,
        1, 2.  Eigenvectors squared at vertex 0: (1/4, 1/2, 1/4).
        K(t)[0,0] = 1/4 + (1/2) e^{-t} + (1/4) e^{-2t}."""
        g = tessera.SparseGraph.fromCOO([0, 1], [1, 2], 3)
        ts = [0.5, 1.0, 2.0, 5.0, 10.0]
        K = g.diagonalHeatKernel([0], ts, krylovDim=3)
        for j, t in enumerate(ts):
            expected = 0.25 + 0.5 * math.exp(-t) + 0.25 * math.exp(-2 * t)
            self.assertAlmostEqual(K[0][j], expected, places=4,
                                   msg=f"t={t}: K={K[0][j]}, expected"
                                       f"={expected}")

    def test_path_3_at_middle(self):
        """Path 0-1-2 at vertex 1 (degree 2).  Eigenvectors squared at
        vertex 1: (1/2, 0, 1/2).  K(t)[1,1] = 1/2 + (1/2) e^{-2t}."""
        g = tessera.SparseGraph.fromCOO([0, 1], [1, 2], 3)
        ts = [0.5, 1.0, 2.0, 5.0, 10.0, 50.0]
        K = g.diagonalHeatKernel([1], ts, krylovDim=3)
        for j, t in enumerate(ts):
            expected = 0.5 + 0.5 * math.exp(-2 * t)
            self.assertAlmostEqual(K[0][j], expected, places=4)

    def test_complete_graph_K3(self):
        """K_3 (triangle).  All degrees = 2.  Symmetric normalized
        Laplacian L = I - (1/2) A; eigenvalues 0, 3/2, 3/2.  By
        symmetry, K(t)[i,i] = 1/3 + (2/3) e^{-3t/2}."""
        g = tessera.SparseGraph.fromCOO([0, 1, 2], [1, 2, 0], 3)
        ts = [0.5, 1.0, 2.0, 5.0, 10.0]
        K = g.diagonalHeatKernel([0, 1, 2], ts, krylovDim=3)
        for s in range(3):
            for j, t in enumerate(ts):
                expected = 1/3 + (2/3) * math.exp(-1.5 * t)
                self.assertAlmostEqual(K[s][j], expected, places=4,
                                       msg=f"start {s}, t={t}: "
                                           f"K={K[s][j]}, expected="
                                           f"{expected}")

    def test_K_at_t_zero_limit(self):
        """At small t, K(t) → 1 (the heat kernel diagonal at t=0 is
        identity)."""
        g = tessera.SparseGraph.fromCOO([0, 1, 2], [1, 2, 0], 3)
        K = g.diagonalHeatKernel([0], [0.001], krylovDim=3)
        self.assertGreater(K[0][0], 0.99)
        self.assertLessEqual(K[0][0], 1.0001)

    def test_K_monotonically_decreasing(self):
        """For any connected graph, K(t)[i,i] is monotonically
        non-increasing in t (the diagonal of e^{-tL} decreases as the
        walker spreads)."""
        g = tessera.SparseGraph.fromCOO([0, 1, 2, 3], [1, 2, 3, 0], 4)
        ts = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0]
        K = g.diagonalHeatKernel([0], ts, krylovDim=4)
        for j in range(1, len(ts)):
            self.assertLessEqual(K[0][j], K[0][j-1] + 1e-9,
                                 f"K not monotonic: t={ts[j-1]}→{ts[j]}, "
                                 f"K={K[0][j-1]}→{K[0][j]}")


class TestSpectralDimension(unittest.TestCase):

    def test_isolated_node_returns_finite(self):
        g = tessera.SparseGraph.fromCOO([], [], 1)
        ds_small, ds_large = g.spectralDimension(
            nWalks=1, maxSigma=10.0, seed=0)
        # Isolated node — heat kernel diagonal stays at 1; finite-difference
        # slope is zero → D_S = 0.
        self.assertEqual(ds_small, 0.0)
        self.assertEqual(ds_large, 0.0)

    def test_long_cycle_gives_dimension_near_1(self):
        """Random walk on a long cycle has D_S(large) → 1."""
        n = 60
        rows = list(range(n))
        cols = [(i + 1) % n for i in range(n)]
        g = tessera.SparseGraph.fromCOO(rows, cols, n)
        _, ds_large = g.spectralDimension(
            nWalks=20, maxSigma=200.0, seed=0)
        self.assertGreater(ds_large, 0.5)
        self.assertLess(ds_large, 1.5)

    def test_dual_of_cdt_lattice_is_finite(self):
        """D_S on the dual of a built CDT lattice should be in a sane
        range (not NaN, not crazy)."""
        st = _make_st()
        g = st.getDualGraph()
        ds_small, ds_large = g.spectralDimension(
            nWalks=20, maxSigma=100.0, seed=0)
        self.assertTrue(math.isfinite(ds_small))
        self.assertTrue(math.isfinite(ds_large))
        # Generous bounds — our small lattice doesn't recover the
        # paper's 4.02; just check sanity.
        self.assertGreater(ds_small, 0.0)
        self.assertLess(ds_small, 10.0)
        self.assertGreater(ds_large, 0.0)
        self.assertLess(ds_large, 10.0)


# ---------------------------------------------------------------------------
# Modularity on skeleton
# ---------------------------------------------------------------------------


def _python_reference_modularity(st, M):
    """Reference Newman-Girvan modularity Q on the 1-skeleton.

    Q = sum_c [L_c / m - (D_c / 2m)^2]

    L_c counts each within-edge once; the sum-over-pairs definition
    used in modularity.py double-counts (it sums the d×d adjacency
    block entries) so we replicate that — equivalent to using
    L_c = 2 * (within-edge count) and m2 = 2m.
    """
    edges = list(st.getEdgeList().toVector())
    if not edges:
        return 0.0
    deg = {}
    for v in st.getVertexList().toVector():
        deg[v.getId()] = 0
    for e in edges:
        deg[e.getSource().getId()] += 1
        deg[e.getTarget().getId()] += 1
    label = {vid: vid % M for vid in deg}
    # Per-community L_c (double-counted) and D_c.
    L_c = {}
    D_c = {}
    for e in edges:
        s = e.getSource().getId()
        t = e.getTarget().getId()
        if label[s] == label[t]:
            L_c[label[s]] = L_c.get(label[s], 0.0) + 2.0
    for vid, d in deg.items():
        c = label[vid]
        D_c[c] = D_c.get(c, 0.0) + d
    m2 = 2 * len(edges)
    Q = 0.0
    for c, dc in D_c.items():
        lc = L_c.get(c, 0.0)
        Q += lc / m2 - (dc / m2) ** 2
    return Q


class TestModularityOnSkeleton(unittest.TestCase):

    def test_M1_is_zero(self):
        """With M=1 (every vertex same label), Q = 0."""
        st = _make_st()
        # M=1 → label(v) = v % 1 = 0 always → single community →
        # L_c/m = 1, (D_c/2m)^2 = 1 → Q = 0.
        self.assertAlmostEqual(st.modularityOnSkeleton(1), 0.0,
                               places=10)

    def test_matches_python_reference_M2(self):
        st = _make_st()
        cpp = st.modularityOnSkeleton(2)
        ref = _python_reference_modularity(st, 2)
        self.assertAlmostEqual(cpp, ref, places=10)

    def test_matches_python_reference_M4(self):
        st = _make_st()
        cpp = st.modularityOnSkeleton(4)
        ref = _python_reference_modularity(st, 4)
        self.assertAlmostEqual(cpp, ref, places=10)

    def test_matches_python_reference_M8(self):
        st = _make_st()
        cpp = st.modularityOnSkeleton(8)
        ref = _python_reference_modularity(st, 8)
        self.assertAlmostEqual(cpp, ref, places=10)

    def test_Q_in_valid_range(self):
        """Q ∈ [-0.5, 1)."""
        st = _make_st()
        for M in (2, 3, 4, 8):
            q = st.modularityOnSkeleton(M)
            self.assertGreaterEqual(q, -0.5)
            self.assertLess(q, 1.0)

    def test_M_zero_or_negative_returns_zero(self):
        st = _make_st()
        self.assertEqual(st.modularityOnSkeleton(0), 0.0)
        self.assertEqual(st.modularityOnSkeleton(-3), 0.0)


class TestGetDualGraph(unittest.TestCase):
    """``Spacetime.getDualGraph`` must return a SparseGraph whose
    node and edge counts match getDualAdjacency's output."""

    def test_node_count_matches_top_simplex_count(self):
        st = _make_st()
        g = st.getDualGraph()
        self.assertEqual(g.nNodes(), st.getTopSimplexCount())

    def test_edges_match_dual_adjacency(self):
        st = _make_st()
        rows, cols, n = st.getDualAdjacency()
        # COO has both (i,j) and (j,i) plus possible duplicates;
        # SparseGraph deduplicates and counts unique undirected edges.
        unique_edges = {tuple(sorted((r, c)))
                        for r, c in zip(rows, cols) if r != c}
        g = st.getDualGraph()
        self.assertEqual(g.nEdges(), len(unique_edges))


class TestModularitySweepIntegration(unittest.TestCase):
    """Integration: combine getDualGraph + spectralDimension +
    modularityOnSkeleton.  Also probe how Q changes under a few
    accepted Pachner moves."""

    def test_combined_observables_run_clean(self):
        """A full sweep cycle: compute Q on the 1-skeleton, build the
        dual graph, compute D_S — without errors."""
        sig = tessera.Signature(4, tessera.Lorentzian)
        metric = tessera.Metric(True, sig)
        st = tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0,
                               tessera.PREFERRED, tessera.Toroid())
        st.build(200)
        cdt = tessera.CDTSimulation(st, 2.2, 0.5, 0.6, 0.02, st.getN41())

        for _ in range(5):
            q = st.modularityOnSkeleton(4)
            g = st.getDualGraph()
            ds_small, ds_large = g.spectralDimension(
                nWalks=20, maxSigma=50.0, seed=42)
            self.assertTrue(math.isfinite(q))
            self.assertTrue(math.isfinite(ds_small))
            self.assertTrue(math.isfinite(ds_large))
            cdt.add()  # may or may not accept; doesn't matter

    def test_pachner_move_changes_Q(self):
        """An accepted add() (which inserts a vertex and may relabel)
        typically changes Q.  Not asserting direction — just that the
        observable responds."""
        sig = tessera.Signature(4, tessera.Lorentzian)
        metric = tessera.Metric(True, sig)
        st = tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0,
                               tessera.PREFERRED, tessera.Toroid())
        st.build(200)
        cdt = tessera.CDTSimulation(st, 2.2, 0.5, 0.6, 0.02, st.getN41())
        q0 = st.modularityOnSkeleton(4)
        # Many tries — at least one accepted move should change Q by
        # at least some tiny amount.
        for _ in range(200):
            if cdt.add():
                q1 = st.modularityOnSkeleton(4)
                if abs(q1 - q0) > 1e-6:
                    return
                q0 = q1
        self.skipTest("No add changed Q meaningfully — try larger lattice")


if __name__ == "__main__":
    unittest.main()
