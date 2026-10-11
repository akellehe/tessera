// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_VERTEXIDPAIR_H
#define TESSERA_VERTEXIDPAIR_H

#include <algorithm>
#include <cstdint>
#include <utility>

// === tessera subsystem ns fwd-decls ===
namespace tessera::graph {}
namespace tessera::observables {}
namespace tessera::quantum {}
namespace tessera::simulations {}
namespace tessera::spacetime {}
namespace tessera::mesh {
using namespace ::tessera::graph;
using namespace ::tessera::spacetime;
using namespace ::tessera::observables;
using namespace ::tessera::simulations;
using namespace ::tessera::quantum;

/// Two vertex ids in ascending order: the key of an undirected edge, the same
/// whichever way round the edge is stored. Not ``EdgeKey``, which keeps the
/// stored source → target order and carries a fingerprint.
using VertexIdPair = std::pair<std::uint64_t, std::uint64_t>;

/// The ascending pair of two vertex ids, ``{min(a, b), max(a, b)}``.
[[nodiscard]] constexpr VertexIdPair ascendingIds(std::uint64_t a, std::uint64_t b) noexcept {
  return {std::min(a, b), std::max(a, b)};
}

} // namespace tessera::mesh

#endif // TESSERA_VERTEXIDPAIR_H
