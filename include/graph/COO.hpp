// Copyright (c) 2026 Twin Vector Labs LLC. All rights reserved.
//
// Shared COO (coordinate-list) sparse edge representations: the one canonical
// type for the (rows, cols, [weights], n) edge arrays emitted by
// Spacetime::getDualAdjacency (unweighted dual graph),
// MutualInformationProfile::weightedAdjacency (mutual-information graph) and
// EmergentGraph::fromWeightedEdges. The CSR builder in graph/CSRBuilder.hpp
// consumes the same field convention.
//
// Each undirected edge appears twice in the arrays (rows[k]=u, cols[k]=v and
// rows[k']=v, cols[k']=u) so the CSR builder can lay out the per-row neighbour
// lists in one pass.

#pragma once

#include <cstddef>
#include <cstdint>
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

// Unweighted COO. ``rows.size() == cols.size() == 2 * |E|`` for
// undirected graphs (each edge listed both directions).
template <typename Idx = std::uint32_t>
struct COO {
    std::vector<Idx> rows;
    std::vector<Idx> cols;
    Idx              n{0};

    [[nodiscard]] std::size_t size()  const noexcept { return rows.size(); }
    [[nodiscard]] bool        empty() const noexcept { return rows.empty(); }
};

// Weighted COO. Same convention; ``weights[k]`` is the weight of the
// directed entry (rows[k] → cols[k]). For an undirected weighted graph
// both directions carry the same weight.
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
