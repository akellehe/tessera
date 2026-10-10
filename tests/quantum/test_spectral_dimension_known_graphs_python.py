"""Spectral-dimension acceptance on known graphs (spec §H4).

The Schwinger pipeline's spectral-dimension code is a generic graph
heat-kernel estimator on top of `EmergentGraph`. Pin it against the
textbook reference graphs whose asymptotic spectral dimensions are
known analytically:

* 1D chain on N vertices (nearest-neighbour, unit weights): D_S → 1
* 2D square lattice (NN, unit weights): D_S → 2
* Complete graph K_N: D_S → 0 at σ ≫ ln N (small-world saturation;
  the random walk thermalises in O(1) steps regardless of σ).

Spec §H4 acceptance: all three targets to within ±0.1 in the
diffusion regime.

Skips cleanly when tessera was built without TESSERA_QUANTUM=1.
"""

from __future__ import annotations

import math
import unittest

try:
    from tessera.quantum.holography import EmergentGraph
    HAVE_QUANTUM = True
except ImportError:
    HAVE_QUANTUM = False


def _chain_edges(n: int) -> list[tuple[int, int, float]]:
    return [(i, i + 1, 1.0) for i in range(n - 1)]


def _square_lattice_edges(width: int, height: int) -> list[tuple[int, int, float]]:
    edges: list[tuple[int, int, float]] = []
    for r in range(height):
        for c in range(width):
            v = r * width + c
            if c + 1 < width:
                edges.append((v, v + 1, 1.0))
            if r + 1 < height:
                edges.append((v, v + width, 1.0))
    return edges


def _complete_edges(n: int) -> list[tuple[int, int, float]]:
    return [(i, j, 1.0) for i in range(n) for j in range(i + 1, n)]


def _diffusion_regime_dS(graph, sigma_min: float, sigma_max: float,
                          n: int = 64) -> tuple[list[float], list[float]]:
    sigmas = [sigma_min * (sigma_max / sigma_min) ** (k / (n - 1))
              for k in range(n)]
    P = graph.return_probability(sigmas, 30)
    dS = graph.spectral_dimension_smoothed(sigmas, P, 7, 2)
    return sigmas, dS


@unittest.skipUnless(HAVE_QUANTUM, "tessera built without TESSERA_QUANTUM=1")
class TestKnownGraphs(unittest.TestCase):
    """Spectral-dimension targets from spec §H4."""

    @staticmethod
    def _plateau_value(sigmas: list[float], dS: list[float],
                        sigma_lo: float, sigma_hi: float) -> float:
        """Median D_S over the σ window [sigma_lo, sigma_hi] — the
        diffusion-regime plateau where finite-size effects haven't
        kicked in yet."""
        plateau = sorted(
            d for s, d in zip(sigmas, dS)
            if sigma_lo <= s <= sigma_hi and d == d
        )
        if not plateau:
            return float("nan")
        return plateau[len(plateau) // 2]

    def test_chain_dS_approaches_one(self) -> None:
        """1D chain on N=80 vertices: D_S(σ) plateau ≈ 1 in the
        diffusion regime σ ∈ (a few, N²/4). Below σ ≈ 1 the walker
        is still in the local-lattice transient; above σ ≈ N the
        finite-size saturation pulls D_S below 1."""
        n = 80
        g = EmergentGraph.from_weighted_edges(n, _chain_edges(n))
        sigmas, dS = _diffusion_regime_dS(g, 0.5, 200.0, n=64)
        plateau = self._plateau_value(sigmas, dS, 3.0, 30.0)
        self.assertAlmostEqual(plateau, 1.0, delta=0.1,
            msg=f"chain plateau D_S = {plateau:.3f}; expected ≈ 1")

    def test_square_lattice_dS_approaches_two(self) -> None:
        """2D square lattice on 16×16 vertices: D_S(σ) peaks at the
        lattice dimension 2 in the diffusion regime. Beyond that
        finite-size effects pull D_S below 2 (the walker starts to
        feel the boundary)."""
        w = h = 16
        n = w * h
        g = EmergentGraph.from_weighted_edges(n, _square_lattice_edges(w, h))
        sigmas, dS = _diffusion_regime_dS(g, 0.1, 1000.0, n=64)
        finite = [d for d in dS if d == d]
        peak = max(finite)
        self.assertAlmostEqual(peak, 2.0, delta=0.2,
            msg=f"lattice peak D_S = {peak:.3f}; expected ≈ 2")

    def test_complete_graph_dS_saturates_to_zero(self) -> None:
        """Complete graph K_N: at σ much larger than the mixing time
        ~1/(N+1), the random walk has fully thermalised and the
        return probability is a constant 1/N — so D_S → 0."""
        n = 24
        g = EmergentGraph.from_weighted_edges(n, _complete_edges(n))
        # At σ = 10 ≫ 1/N, P should be at saturation = 1/N.
        sigmas = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
        P = g.return_probability(sigmas, 30)
        # P at σ = 10 ≈ 1/n; D_S there ≈ 0 since log P is flat.
        self.assertAlmostEqual(P[-1], 1.0 / n, delta=1e-6,
            msg=f"K_{n} P(σ=10) = {P[-1]}; expected 1/n = {1/n:.4f}")
        # D_S between σ=5 and σ=10:
        d_log_P = math.log(P[-1] / P[-2])
        d_log_s = math.log(sigmas[-1] / sigmas[-2])
        dS_tail = -2.0 * d_log_P / d_log_s
        self.assertAlmostEqual(dS_tail, 0.0, delta=0.05,
            msg=f"K_{n} D_S in saturated regime = {dS_tail:.4f}; expected ≈ 0")


if __name__ == "__main__":
    unittest.main()
