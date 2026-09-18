// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

//
// Created by andrew on 12/14/25.
//

#include <algorithm>
#include <limits>
#include <memory>
#include <vector>

#include "mesh/ForwardDeclarations.h"
#include "mesh/TemporalOrientation.h"
#include "mesh/Vertex.h"


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
TemporalOrientation::TemporalOrientation(uint8_t ti_, uint8_t tf_)
    // A k-simplex has k+1 vertices, so k = ti + tf - 1. Clamp at zero: (0, 0)
    // would otherwise wrap to 255 through uint8_t, and decTi()/decTf() reach
    // (0, 0) from (1, 0) and (0, 1). The default constructor already uses 0.
    : ti(ti_), tf(tf_),
      k(static_cast<uint8_t>(ti_ + tf_ > 0 ? ti_ + tf_ - 1 : 0)),
      fingerprint({ti_, tf_}) {
}

TemporalOrientation::TemporalOrientation()
    : ti(0), tf(0), k(0), fingerprint({0, 0}) {
}

[[nodiscard]] std::pair<uint8_t, uint8_t> TemporalOrientation::numeric() const {
  return {ti, tf};
}

[[nodiscard]] size_t TemporalOrientation::hash() const {
  return fingerprint.fingerprint();
}


[[nodiscard]] TemporalOrientation TemporalOrientation::flip() const {
  TemporalOrientation o{tf, ti};
  return o;
}

[[nodiscard]]
TemporalOrientation TemporalOrientation::decTi() const {
  if (ti == 0) return {0, tf};
  TemporalOrientation o{static_cast<uint8_t>(ti - 1), tf};
  return o;
}

[[nodiscard]]
TemporalOrientation TemporalOrientation::decTf() const {
  if (tf == 0) return {ti, 0};
  TemporalOrientation o{ti, static_cast<uint8_t>(tf - 1)};
  return o;
}

[[nodiscard]] std::string TemporalOrientation::toString() const noexcept {
  return "<TemporalOrientation: (" + std::to_string(ti) + ", " + std::to_string(tf) + ")>";
}

bool TemporalOrientation::operator==(const TemporalOrientation &other) const noexcept {
  return ti == other.ti && tf == other.tf;
}

[[nodiscard]] TimeOrientation TemporalOrientation::getOrientation() const {
  if (ti == tf) return TimeOrientation::UNKNOWN;
  if (ti > tf) return TimeOrientation::PRESENT;
  return TimeOrientation::FUTURE;
}

[[nodiscard]] std::vector<TemporalOrientation> TemporalOrientation::getFacialOrientations() const {
  if (ti + tf == 0) return {};
  if (ti == 0) return {decTf()};
  if (tf == 0) return {decTi()};
  std::vector<TemporalOrientation> orientations;
  orientations.reserve(2);
  orientations.push_back(decTi());
  orientations.push_back(decTf());
  return orientations;
}

/// A k-simplex has \f$ k+1 \f$ vertices.
[[nodiscard]] uint8_t TemporalOrientation::getK() const {
  return k;
}

TemporalOrientation TemporalOrientation::orientationOf(const VertexPtrs &vertices) {
  // Two passes: find the min/max times first, then count. One pass mis-counts
  // when the first vertex seen is on the final slice rather than the initial one.
  double tMin = std::numeric_limits<double>::max();
  double tMax = std::numeric_limits<double>::lowest();
  for (const auto &v : vertices) {
    double t = v->getTime();
    tMin = std::min(tMin, t);
    tMax = std::max(tMax, t);
  }
  uint8_t tiVertices = 0;
  uint8_t tfVertices = 0;
  for (const auto &v : vertices) {
    if (v->getTime() == tMin)
      tiVertices++;
    else
      tfVertices++;
  }
  return {tiVertices, tfVertices};
}
}
