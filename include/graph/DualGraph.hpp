// Copyright (c) 2026 Twin Vector Labs LLC. All rights reserved.
//
// Centralised definition of the simplicial-complex dual graph: every
// top-simplex is a vertex, and two top-simplices are adjacent when they share
// a facet. Both Spacetime::getDualAdjacency (which collects the edges into a
// COO array) and WilsonLoop::dualNeighbors (which traces dual-graph loops)
// go through this walk.

#pragma once

#include "mesh/Simplex.h"

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

// Top-simplex dual-graph neighbours of ``sigma``: every other simplex
// sharing one of sigma's facets. Boundary facets contribute no
// neighbours (their ``getCofaces()`` returns only sigma itself).
//
// Pointer comparison is used to skip sigma itself — tessera's mesh
// keeps one canonical Simplex per fingerprint, so pointer equality
// matches fingerprint equality. Callers that want to defensively also
// filter on dimension can post-process the returned vector.
[[nodiscard]] inline std::vector<SimplexPtr>
dualNeighbors(SimplexPtr const& sigma) {
    std::vector<SimplexPtr> nbrs;
    for (auto const& facet : sigma->getFacets()) {
        for (auto const& coface : facet->getCofaces()) {
            if (coface != sigma) nbrs.push_back(coface);
        }
    }
    return nbrs;
}

} // namespace tessera::graph
