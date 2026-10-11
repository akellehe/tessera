// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_GRAPH_WEIGHTEDCSRGRAPH_HPP
#define TESSERA_GRAPH_WEIGHTEDCSRGRAPH_HPP

#include "graph/CSRBuilder.hpp"
#include "graph/SpectralGraph.hpp"

#include <cstddef>
#include <tuple>
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
/// separately rather than summed per call because it is built with the
/// adjacency.
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

    /// The graph on vertices \f$ 0 \dots n-1 \f$ of an undirected edge list,
    /// each edge \f$ (u, v, w) \f$ listed once with \f$ u \ne v \f$: both
    /// directions go into the adjacency, \f$ u \to v \f$ before
    /// \f$ v \to u \f$, through buildFromCOO. The endpoints are not checked.
    [[nodiscard]] static WeightedCsrGraph
    fromWeightedEdges(int n,
                      std::vector<std::tuple<int, int, double>> const& edges) {
        std::vector<int>    rows;
        std::vector<int>    cols;
        std::vector<double> weights;
        rows.reserve(edges.size() * 2);
        cols.reserve(edges.size() * 2);
        weights.reserve(edges.size() * 2);
        for (auto const& [u, v, w] : edges) {
            rows.push_back(u); cols.push_back(v); weights.push_back(w);
            rows.push_back(v); cols.push_back(u); weights.push_back(w);
        }
        WeightedCsrGraph g;
        g.buildFromCOO(n, rows, cols, weights);
        return g;
    }

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
    /// Fill the graph from a symmetric COO, each undirected edge listed once
    /// per endpoint: the adjacency by graph::buildCSRFromCOO, and each degree
    /// as its row's weights summed in CSR order.
    void buildFromCOO(int n,
                      std::vector<int> const& rows,
                      std::vector<int> const& cols,
                      std::vector<double> const& weights) {
        n_ = n;
        buildCSRFromCOO<int, double, int>(
            static_cast<std::size_t>(n_), rows, cols, weights,
            indptr_, indices_, weights_);

        degrees_.assign(static_cast<std::size_t>(n_), 0.0);
        for (int v = 0; v < n_; ++v) {
            const int lo = indptr_[static_cast<std::size_t>(v)];
            const int hi = indptr_[static_cast<std::size_t>(v) + 1];
            double d = 0.0;
            for (int k = lo; k < hi; ++k) {
                d += weights_[static_cast<std::size_t>(k)];
            }
            degrees_[static_cast<std::size_t>(v)] = d;
        }
    }

    int n_{0};
    std::vector<int>    indptr_;
    std::vector<int>    indices_;
    std::vector<double> weights_;
    std::vector<double> degrees_;
};

} // namespace tessera::graph

#endif // TESSERA_GRAPH_WEIGHTEDCSRGRAPH_HPP
