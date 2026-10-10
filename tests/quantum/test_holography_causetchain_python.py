"""§H6 acceptance: CausetChain integration into the holography pipeline.

Spec §H6 acceptance:

  On a trivial chain CausetChain the result agrees numerically with
  the regular-lattice path; on a branching causet it does not.

A "trivial chain" Spacetime has one vertex per time slice connected
by timelike edges to its neighbours. ``Causet.chain_from(st)`` extracts
the chain with ``hoppingPairs == [(0,1), (1,2), ..., (N-2, N-1)]``,
which is identical to the standard 1D nearest-neighbour pattern that
``SchwingerHamiltonian::mpo`` builds by default. So:

  1. Running the holography pipeline with ``cfg.tdvp.hopping_pairs``
     copied from the trivial-chain extraction must give numerically
     identical results to running with ``hoppingPairs = []`` (default).
  2. Building a Spacetime with a branching antichain (one vertex
     splits into two) gives ``hopping_pairs`` that differ from the
     default chain, so the resulting D_S(σ) profile differs.

Skips cleanly when tessera was built without TESSERA_QUANTUM=1.
"""

from __future__ import annotations

import unittest
import cmath

try:
    import tessera
    from tessera.quantum import TDVPConfig, SchwingerQuench, Causet
    from tessera.quantum.holography import (
        HolographyConfig,
        EmergentSpectralDimension,
        MutualInformationProfile,
    )
    HAVE_QUANTUM = True
except ImportError:
    HAVE_QUANTUM = False


def _trivial_chain_spacetime(n_sites: int) -> "tessera.Spacetime":
    """A 1D chain Spacetime: vertex k at time t=k, timelike edge
    between every adjacent pair (k, k+1)."""
    st = tessera.Spacetime()
    verts = [
        st.create_vertex(k, [float(k)])
        for k in range(n_sites)
    ]
    for k in range(n_sites - 1):
        st.create_edge(verts[k], verts[k + 1], cmath.sqrt(complex(-1.0)))  # timelike
    return st


def _branching_spacetime() -> "tessera.Spacetime":
    """A small branching causet: time slices have 1, 2, 1 vertices
    respectively, with timelike edges crossing one slice each.

    Layer 0:   v0
                |\\
    Layer 1:   v1 v2
                |/
    Layer 2:   v3
    """
    st = tessera.Spacetime()
    v0 = st.create_vertex(0, [0.0])
    v1 = st.create_vertex(1, [1.0])
    v2 = st.create_vertex(2, [1.0])
    v3 = st.create_vertex(3, [2.0])
    st.create_edge(v0, v1, cmath.sqrt(complex(-1.0)))
    st.create_edge(v0, v2, cmath.sqrt(complex(-1.0)))
    st.create_edge(v1, v3, cmath.sqrt(complex(-1.0)))
    st.create_edge(v2, v3, cmath.sqrt(complex(-1.0)))
    return st


def _baseline_config(N: int, hoppingPairs=None,
                      vertexIds=None) -> "HolographyConfig":
    cfg = HolographyConfig()
    cfg.tdvp = TDVPConfig()
    cfg.tdvp.N = N; cfg.tdvp.a = 1.0; cfg.tdvp.g = 1.0
    cfg.tdvp.m = 0.5; cfg.tdvp.L0 = 0.0
    cfg.tdvp.dmrg_max_bond_dim = 32; cfg.tdvp.dmrg_n_sweeps = 8
    cfg.tdvp.dmrg_krylov_dim = 4; cfg.tdvp.dmrg_cutoff = 1e-12
    cfg.tdvp.i0 = 1; cfg.tdvp.d = 3
    cfg.tdvp.dt = 0.2; cfg.tdvp.T = 0.4; cfg.tdvp.snapshot_every = 1
    cfg.tdvp.max_bond_dim = 40; cfg.tdvp.cutoff = 1e-10; cfg.tdvp.krylov_dim = 10
    cfg.tdvp.quiet = True; cfg.tdvp.conserve_qns = True
    if hoppingPairs is not None:
        cfg.tdvp.hopping_pairs = hoppingPairs
    cfg.sigma_min = 0.1; cfg.sigma_max = 100.0; cfg.sigma_count = 24
    cfg.epsilon_i = 1e-8; cfg.krylov_dim = 30
    cfg.include_temporal = False  # the structural test is about the
                                  # spatial hopping graph, not temporal MI
    if vertexIds is not None:
        cfg.vertex_ids = vertexIds
    return cfg


@unittest.skipUnless(HAVE_QUANTUM, "tessera built without TESSERA_QUANTUM=1")
class TestTrivialChainAgrees(unittest.TestCase):
    """A trivial chain extracted from Spacetime must give the same
    numerics as the default NN-chain pipeline."""

    def test_trivial_chain_matches_default(self) -> None:
        N = 6
        st = _trivial_chain_spacetime(N)
        chain = Causet.chain_from(st)
        expected_hops = [(k, k + 1) for k in range(N - 1)]
        self.assertEqual(sorted(chain.hopping_pairs), expected_hops)
        self.assertEqual(chain.n_sites, N)

        r0 = EmergentSpectralDimension(_baseline_config(N)).compute()
        chain_pairs = [tuple(p) for p in chain.hopping_pairs]
        r1 = EmergentSpectralDimension(
            _baseline_config(N,
                              hoppingPairs=chain_pairs,
                              vertexIds=list(chain.vertex_ids))
        ).compute()

        self.assertEqual(r0.graph_n_vertices, r1.graph_n_vertices)
        self.assertEqual(r0.graph_n_edges,    r1.graph_n_edges)
        for a, b in zip(r0.P, r1.P):
            self.assertAlmostEqual(a, b, places=10,
                msg=f"P(σ) differs: {a} vs {b}")
        for a, b in zip(r0.d_s, r1.d_s):
            self.assertAlmostEqual(a, b, places=10,
                msg=f"D_S differs: {a} vs {b}")


@unittest.skipUnless(HAVE_QUANTUM, "tessera built without TESSERA_QUANTUM=1")
class TestBranchingChainDiffers(unittest.TestCase):
    """A non-trivial causet (with branching antichains) produces a
    different hopping graph than the default chain."""

    def test_branching_changes_hopping_graph(self) -> None:
        st = _branching_spacetime()
        chain = Causet.chain_from(st)
        self.assertEqual(chain.n_sites, 4)
        canonical = [(0, 1), (1, 2), (2, 3)]
        chain_pairs = sorted(tuple(p) for p in chain.hopping_pairs)
        self.assertNotEqual(chain_pairs, canonical,
            msg=f"branching causet hopping {chain_pairs} should not equal "
                f"canonical chain")


@unittest.skipUnless(HAVE_QUANTUM, "tessera built without TESSERA_QUANTUM=1")
class TestVertexIdMapping(unittest.TestCase):
    """``MutualInformationProfile.vertex_id`` looks up the spacetime
    ID for each flat-site index when ``vertex_ids`` is supplied."""

    def test_vertex_ids_round_trip(self) -> None:
        N = 6
        st = _trivial_chain_spacetime(N)
        chain = Causet.chain_from(st)
        chain_pairs = [tuple(p) for p in chain.hopping_pairs]
        cfg = _baseline_config(N,
                                hoppingPairs=chain_pairs,
                                vertexIds=list(chain.vertex_ids))
        cfg.tdvp.record_mutual_information = True
        quench = SchwingerQuench(cfg.tdvp).evolve()
        profile = MutualInformationProfile(quench.snapshots, cfg)
        for k in range(N):
            self.assertEqual(profile.vertex_id(k), chain.vertex_ids[k])

    def test_default_vertex_ids_are_flat_indices(self) -> None:
        """Without vertexIds, vertex_id(k) = k."""
        cfg = _baseline_config(6)
        cfg.tdvp.record_mutual_information = True
        quench = SchwingerQuench(cfg.tdvp).evolve()
        profile = MutualInformationProfile(quench.snapshots, cfg)
        for k in range(6):
            self.assertEqual(profile.vertex_id(k), k)


if __name__ == "__main__":
    unittest.main()
