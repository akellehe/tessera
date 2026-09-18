// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "spacetime/topologies/Topology.h"
#include <vector>
#include <memory>
#include <iostream>
#include <stdexcept>

#include "spacetime/Spacetime.h"
#include "mesh/Vertex.h"

// === tessera subsystem ns fwd-decls ===
namespace tessera::graph {}
namespace tessera::mesh {}
namespace tessera::observables {}
namespace tessera::quantum {}
namespace tessera::simulations {}
namespace tessera::spacetime {
using namespace ::tessera::mesh;
using namespace ::tessera::graph;
using namespace ::tessera::observables;
using namespace ::tessera::simulations;
using namespace ::tessera::quantum;
Topology::~Topology() = default;
void Topology::build(Spacetime *spacetime, int numSimplices) {
  std::cout << "Building Topology (base)" << std::endl;
}

int Topology::dimension() const {
  // Only the fixed-triangulation fixtures have an intrinsic manifold dimension;
  // the dimension-parametric CDT topologies (Toroid, Sphere, Cylinder) take
  // theirs from the spacetime's signature at build() time and so do not
  // override this.
  throw std::logic_error(
      "Topology::dimension(): this topology has no intrinsic manifold dimension "
      "(it is parametric in the spacetime signature). Build it with an explicit "
      "Signature(d, …) instead of deriving d from the topology.");
}

void Topology::buildExplicit(
    Spacetime *spacetime, std::size_t numVertices,
    const std::vector<std::vector<std::uint64_t>> &topSimplices) {
  std::vector<VertexPtr> verts;
  verts.reserve(numVertices);
  // Coordinate-free vertices: the triangulation is purely combinatorial here.
  // Allocate through the id-counter path, not the explicit-id overload, so the
  // counter advances past these ids.  On a fresh spacetime that still assigns
  // 0..numVertices-1, and it leaves the counter able to hand out collision-free
  // ids to anything that later grows the complex (e.g. a pre-geometric Pachner
  // add move).
  for (std::size_t i = 0; i < numVertices; ++i) {
    verts.push_back(spacetime->createVertex());
  }
  for (const auto &simplex : topSimplices) {
    VertexPtrs sv;
    sv.reserve(simplex.size());
    for (auto id : simplex) sv.push_back(verts.at(static_cast<std::size_t>(id)));
    spacetime->createSimplex(sv);
  }
}
} // namespace tessera::spacetime
