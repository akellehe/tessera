// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "spacetime/topologies/SolidSimplex.h"

#include <cmath>
#include <complex>
#include <memory>

#include "mesh/Edge.h"
#include "mesh/EdgeList.h"
#include "spacetime/Metric.h"
#include "spacetime/Spacetime.h"

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace tessera::spacetime {

void SolidSimplex::build(Spacetime *spacetime, int /*numSimplices*/) {
  if (n_ < 1) throw std::invalid_argument("SolidSimplex requires n >= 1");
  // Δ^n: a single top simplex on n+1 vertices.
  const std::size_t numV = static_cast<std::size_t>(n_) + 1;
  std::vector<std::uint64_t> s;
  s.reserve(numV);
  for (std::size_t v = 0; v < numV; ++v) s.push_back(static_cast<std::uint64_t>(v));
  buildExplicit(spacetime, numV, {std::move(s)});
}

std::shared_ptr<Spacetime> SolidSimplex::spacetime(int dimension, bool balancedEdges) {
  if (dimension < 1)
    throw std::invalid_argument("SolidSimplex::spacetime: dimension must be at least one");
  auto metric = std::make_shared<Metric>(true, Signature(dimension, SignatureType::Lorentzian));
  std::shared_ptr<Topology> topology = std::make_shared<SolidSimplex>(dimension);
  auto host = std::make_shared<Spacetime>(metric, SpacetimeType::CDT, 1.0, 1.0,
                                          Foliation::PREFERRED, topology);
  host->build();
  // The wiring mode is stamped before any growth, and the seed's own
  // uniform |l^2| = 1 edges honor it too (balanced: l = sqrt(1/2)*(1+i)).
  host->setBalancedEdgeWiring(balancedEdges);
  for (auto *edge : host->getEdgeList()->toVector())
    edge->setLength(balancedEdges ? Spacetime::balancedLength(1.0) : std::sqrt(std::complex<double>(1.0, 0.0)));
  return host;
}

} // namespace tessera::spacetime
