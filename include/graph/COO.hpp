// Copyright (c) 2026 Twin Vector Labs LLC. All rights reserved.
//
// The weighted COO (coordinate-list) edge arrays (rows, cols, weights, n):
// what MutualInformationProfile::weightedAdjacency returns and EmergentGraph's
// constructor reads. The CSR builder in graph/CSRBuilder.hpp consumes the same
// field convention, and Spacetime::getDualAdjacency returns unweighted
// (rows, cols, n) arrays in that convention, as a tuple.
//
// Each undirected edge appears twice in the arrays (rows[k]=u, cols[k]=v and
// rows[k']=v, cols[k']=u) so the CSR builder can lay out the per-row neighbour
// lists in one pass.

#pragma once

#include <cstddef>
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

// Weighted COO. ``rows.size() == cols.size() == weights.size() == 2 * |E|``
// for undirected graphs (each edge listed both directions); ``weights[k]`` is
// the weight of the directed entry (rows[k] → cols[k]), and both directions of
// an undirected edge carry the same weight.
template <typename Idx = int, typename W = double>
struct WeightedCOO {
    std::vector<Idx> rows;
    std::vector<Idx> cols;
    std::vector<W>   weights;
    Idx              n{0};

    [[nodiscard]] std::size_t size()  const noexcept { return rows.size(); }
    [[nodiscard]] bool        empty() const noexcept { return rows.empty(); }
};

} // namespace tessera::graph
