// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_OBSERVABLES_SPARSEGRAPH_H
#define TESSERA_OBSERVABLES_SPARSEGRAPH_H

#include "graph/SpectralGraph.hpp"

#include <cstdint>
#include <random>
#include <utility>
#include <vector>

// === tessera subsystem ns fwd-decls ===
namespace tessera::graph {}
namespace tessera::mesh {}
namespace tessera::quantum {}
namespace tessera::simulations {}
namespace tessera::spacetime {}
namespace tessera::observables {
using namespace ::tessera::mesh;
using namespace ::tessera::graph;
using namespace ::tessera::spacetime;
using namespace ::tessera::simulations;
using namespace ::tessera::quantum;

/// Undirected sparse graph in compressed sparse row (CSR) form.
///
/// References: Newman and Girvan, "Finding and evaluating community structure
/// in networks", arXiv:cond-mat/0308217 (modularity); Ambjorn, Jurkiewicz,
/// Loll, "Spectral Dimension of the Universe", arXiv:hep-th/0505113.
///
/// Built from a coordinate-format (COO) `(rows, cols)` array, typically from
/// :func:`Spacetime::getDualAdjacency` for the modularity and
/// spectral-dimension observables. All operations are read-only on the graph
/// data; the modularity sweep recomputes the graph from the spacetime at every
/// measurement rather than mutating it.
///
/// Derives from ``SpectralGraph``, inheriting the diagonal heat-kernel and
/// return-probability machinery. The override here installs the symmetric
/// normalised Laplacian \f$ L_{\mathrm{sym}} = I - D^{-1/2} A D^{-1/2} \f$.
///
/// Storage is plain ``std::vector`` with no Eigen dependency, so this header is
/// cheap to include anywhere in tessera_core.
class SparseGraph : public SpectralGraph {
public:
  /// Build from coordinate format. ``rows`` and ``cols`` may contain
  /// duplicates and need not be symmetric: both directions are added once each
  /// and duplicates collapsed, giving a binary adjacency.
  static SparseGraph fromCOO(const std::vector<std::uint32_t> &rows,
                             const std::vector<std::uint32_t> &cols,
                             std::uint32_t n);

  std::size_t nNodes() const noexcept { return nNodes_; }
  std::size_t nEdges() const noexcept { return nEdges_; }
  /// CSR index pointers, length nNodes() + 1.
  const std::vector<std::int64_t> &indptr() const noexcept { return indptr_; }
  /// CSR column indices, length 2 * nEdges() (each undirected edge
  /// stored twice).
  const std::vector<std::uint32_t> &indices() const noexcept { return indices_; }
  /// Per-node degree.  ``degree(i) == indptr_[i+1] - indptr_[i]``.
  std::uint32_t degree(std::uint32_t i) const noexcept {
    return static_cast<std::uint32_t>(indptr_[i + 1] - indptr_[i]);
  }

  // ── SpectralGraph overrides ─────────────────────────────────────────
  int nVertices() const override {
    return static_cast<int>(nNodes_);
  }

  /// \f$ y \leftarrow L_{\mathrm{sym}} x \f$ with
  /// \f$ L_{\mathrm{sym}} = I - D^{-1/2} A D^{-1/2} \f$. Isolated nodes
  /// (degree 0) get `invSqrtDeg = 0`, so the product reduces to
  /// \f$ y_i = x_i \f$: the Laplacian acts as the identity on them.
  void applyLaplacian(std::vector<double> const &x,
                        std::vector<double> &y) const override;

  /// True iff the graph is 2-colorable (no odd cycle).
  /// Breadth-first-search based; empty graphs are trivially bipartite.
  bool isBipartite() const;

  /// Newman-Girvan modularity \f$ Q \f$ of the partition given by per-node
  /// community ``labels`` (length `nNodes()`):
  ///
  /// \f[
  ///   Q = \sum_c \left[ \frac{L_c}{m} - \left(\frac{D_c}{2m}\right)^2
  ///   \right]
  /// \f]
  ///
  /// where \f$ L_c \f$ is the intra-community edge count, \f$ D_c \f$ the
  /// summed degree of community \f$ c \f$, and \f$ m \f$ the total edge count.
  /// Distinct label values are distinct communities; labels need not be dense or
  /// zero-based. Returns 0 for an empty or edgeless graph.
  /// @throws std::invalid_argument when ``labels.size() != nNodes()``.
  [[nodiscard]] double modularity(const std::vector<int> &labels) const;

  /// Diagonal of the heat kernel \f$ e^{-t L_{\mathrm{sym}}} \f$ for each
  /// (start, t) pair.
  ///
  /// Returns a flat row-major matrix of shape
  /// ``(starts.size(), times.size())``. ``out[w][j]`` approximates
  /// \f$ [e^{-t_j L_{\mathrm{sym}}}]_{s_w, s_w} \f$. An empty graph returns
  /// 1.0 by convention.
  ///
  /// The random-walk Laplacian \f$ L_{\mathrm{rw}} = I - D^{-1} A \f$ and the
  /// symmetric normalised \f$ L_{\mathrm{sym}} \f$ are related by
  /// \f$ L_{\mathrm{rw}} = D^{-1/2} L_{\mathrm{sym}} D^{1/2} \f$, so the
  /// diagonal entries of \f$ e^{-tL} \f$ agree. The spectral dimension can
  /// therefore be extracted from either, and \f$ L_{\mathrm{sym}} \f$ is
  /// symmetric, which permits Lanczos rather than Arnoldi.
  std::vector<double> diagonalHeatKernel(
      const std::vector<std::uint32_t> &starts,
      const std::vector<double> &times,
      int krylovDim = 30) const;

  /// Estimate the spectral dimension at small and large diffusion times by a
  /// centred finite difference of
  /// \f$ -2\, d\log K(t) / d\log t \f$.
  ///
  /// Picks ``nWalks`` random start nodes uniformly without replacement, capped
  /// at ``nNodes()``.
  ///
  /// Returns ``(D_S_small, D_S_large)``, or ``{NaN, NaN}`` if the graph is too
  /// small or has no valid \f$ \log K \f$ samples.
  ///
  /// This overload samples random walks and returns a (small, large) pair,
  /// unlike the inherited static ``SpectralGraph::spectralDimension(sigmas, P)``,
  /// a pure finite-difference helper on a precomputed \f$ P(\sigma) \f$ curve.
  std::pair<double, double> spectralDimension(
      int nWalks, double maxSigma, std::mt19937 *rng,
      double tailFraction = 0.2, int nTimes = 40,
      double tMin = 0.5, int krylovDim = 30) const;

private:
  std::size_t nNodes_ = 0;
  std::size_t nEdges_ = 0;
  std::vector<std::int64_t> indptr_;     // size nNodes + 1
  std::vector<std::uint32_t> indices_;   // size 2 * nEdges
  std::vector<double> invSqrtDeg_;        // size nNodes; 0.0 for isolated nodes
};

}  // namespace tessera

#endif  // TESSERA_OBSERVABLES_SPARSEGRAPH_H
