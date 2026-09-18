// Copyright (c) 2026 Twin Vector Labs LLC. All rights reserved.
//
// Shared CSR construction from a symmetric COO edge list: one prefix-sum plus
// cursor pass covering both callers, SparseGraph::fromCOO (binary, uint32
// indices, int64 pointers) and EmergentGraph::buildFromCOO_ (weighted, int
// indices and pointers), via the (Idx, Ptr) / (Idx, W, Ptr) type pairs.
//
// Contract: input rows/cols must already be the symmetric form (each
// undirected edge listed twice). Out-of-range indices are undefined
// behaviour; callers validate before calling.

#pragma once

#include <cstddef>
#include <numeric>
#include <vector>

// === tessera subsystem ns fwd-decls ===
namespace tessera::mesh {}
namespace tessera::observables {}
namespace tessera::quantum {}
namespace tessera::simulations {}
namespace tessera::spacetime {}
namespace tessera::graph {
using namespace ::tessera::mesh;
using namespace ::tessera::spacetime;
using namespace ::tessera::observables;
using namespace ::tessera::simulations;
using namespace ::tessera::quantum;

// Build unweighted CSR (indptr, indices) from a symmetric COO.
//
// Outputs are written via reference so the helper composes with classes
// that own their own storage vectors. ``indptr`` has size ``n + 1``;
// ``indices`` has size ``rows.size()``.
template <typename Idx, typename Ptr>
void buildCSRFromCOO(std::size_t n,
                       std::vector<Idx> const& rows,
                       std::vector<Idx> const& cols,
                       std::vector<Ptr>& indptr,
                       std::vector<Idx>& indices) {
    indptr.assign(n + 1, Ptr{0});
    for (auto r : rows)
        ++indptr[static_cast<std::size_t>(r) + 1];
    std::partial_sum(indptr.begin(), indptr.end(), indptr.begin());

    indices.assign(rows.size(), Idx{0});
    std::vector<Ptr> cursor(n, Ptr{0});
    for (std::size_t k = 0; k < rows.size(); ++k) {
        const auto r   = static_cast<std::size_t>(rows[k]);
        const auto pos = static_cast<std::size_t>(indptr[r] + cursor[r]);
        indices[pos] = cols[k];
        ++cursor[r];
    }
}

// Weighted overload. Builds (indptr, indices, weightsOut) so the
// per-row neighbour list and its parallel weight array stay aligned.
template <typename Idx, typename W, typename Ptr>
void buildCSRFromCOO(std::size_t n,
                       std::vector<Idx> const& rows,
                       std::vector<Idx> const& cols,
                       std::vector<W>   const& weights,
                       std::vector<Ptr>& indptr,
                       std::vector<Idx>& indices,
                       std::vector<W>&   weightsOut) {
    indptr.assign(n + 1, Ptr{0});
    for (auto r : rows)
        ++indptr[static_cast<std::size_t>(r) + 1];
    std::partial_sum(indptr.begin(), indptr.end(), indptr.begin());

    indices.assign(rows.size(), Idx{0});
    weightsOut.assign(rows.size(), W{0});
    std::vector<Ptr> cursor(n, Ptr{0});
    for (std::size_t k = 0; k < rows.size(); ++k) {
        const auto r   = static_cast<std::size_t>(rows[k]);
        const auto pos = static_cast<std::size_t>(indptr[r] + cursor[r]);
        indices[pos]    = cols[k];
        weightsOut[pos] = weights[k];
        ++cursor[r];
    }
}

} // namespace tessera::graph
