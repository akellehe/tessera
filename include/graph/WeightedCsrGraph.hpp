// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_GRAPH_WEIGHTEDCSRGRAPH_HPP
#define TESSERA_GRAPH_WEIGHTEDCSRGRAPH_HPP

#include "graph/SpectralGraph.hpp"

#include <cstddef>
#include <utility>
#include <vector>

namespace tessera::mesh {}
namespace tessera::observables {}
namespace tessera::quantum {}
namespace tessera::simulations {}
namespace tessera::spacetime {}
namespace tessera::graph {

/// A weighted graph in compressed sparse row form, with the unnormalised
/// Laplacian \f$ L = D - W \f$.
///
/// `indptr[v] .. indptr[v+1]` indexes into `indices` and `weights` for the
/// neighbours of \f$ v \f$. The adjacency is symmetric: each undirected edge
/// appears once per endpoint. `degrees[v]` is \f$ \sum_w W_{vw} \f$, held
/// separately rather than summed per call because the callers build it while
/// assembling the adjacency.
///
/// This carries the storage and the matrix-vector product; the Krylov sweep and
/// the spectral dimension come from `SpectralGraph`. Derive from it when a
/// graph needs more than the product, as `quantum::EmergentGraph` does, or use
/// it directly when it does not.
class WeightedCsrGraph : public SpectralGraph {
public:
    WeightedCsrGraph() = default;

    WeightedCsrGraph(int n,
                     std::vector<int> indptr,
                     std::vector<int> indices,
                     std::vector<double> weights,
                     std::vector<double> degrees)
      : n_(n),
        indptr_(std::move(indptr)),
        indices_(std::move(indices)),
        weights_(std::move(weights)),
        degrees_(std::move(degrees)) {}

    [[nodiscard]] int nVertices() const noexcept override { return n_; }

    /// \f$ y \leftarrow (D - W) x \f$: the diagonal term \f$ D_{vv} x_v \f$
    /// less the weighted sum over the row's neighbours.
    void applyLaplacian(std::vector<double> const& x,
                        std::vector<double>& y) const override {
        y.assign(static_cast<std::size_t>(n_), 0.0);
        for (int i = 0; i < n_; ++i) {
            const auto iu = static_cast<std::size_t>(i);
            double s = degrees_[iu] * x[iu];
            const int lo = indptr_[iu];
            const int hi = indptr_[iu + 1];
            for (int k = lo; k < hi; ++k) {
                const auto ku = static_cast<std::size_t>(k);
                s -= weights_[ku] *
                     x[static_cast<std::size_t>(indices_[ku])];
            }
            y[iu] = s;
        }
    }

protected:
    int n_{0};
    std::vector<int>    indptr_;
    std::vector<int>    indices_;
    std::vector<double> weights_;
    std::vector<double> degrees_;
};

} // namespace tessera::graph

#endif // TESSERA_GRAPH_WEIGHTEDCSRGRAPH_HPP
