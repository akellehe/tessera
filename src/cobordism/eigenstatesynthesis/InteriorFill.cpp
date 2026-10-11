// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

// EigenstateSynthesis (include/cobordism/EigenstateSynthesis.h): the fixed-
// boundary interior fill. One of the translation units that define the class's
// members by topic.

#include "cobordism/EigenstateSynthesis.h"
#include "quantum/ChoiJamiolkowski.h"

#include <numeric>
#include <limits>
#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "cobordism/ChainComplex.h"
#include "mesh/Edge.h"
#include "mesh/EdgeList.h"
#include "mesh/Fingerprint.h"
#include "mesh/Simplex.h"
#include "mesh/Vertex.h"
#include "mesh/VertexList.h"
#include "spacetime/Metric.h"
#include "spacetime/Signature.h"
#include "spacetime/Spacetime.h"
#include "spacetime/pachner/AddMove.h"

#ifdef TESSERA_CUDA
#include "cuda/eigenstate_cuda.h"
#endif

namespace tessera::cobordism {

using cd = std::complex<double>;

// === Fixed-boundary interior fill ===

std::vector<std::complex<double>> EigenstateSynthesis::interiorWeights() const {
  std::vector<std::complex<double>> w;
  w.reserve(interiorEdgeIdx_.size());
  for (const auto i : interiorEdgeIdx_)
    w.push_back((edges_[i]->getLength() * edges_[i]->getLength()));
  return w;
}

std::vector<std::complex<double>> EigenstateSynthesis::interiorPhases() const {
  std::vector<std::complex<double>> th;
  th.reserve(interiorEdgeIdx_.size());
  for (const auto i : interiorEdgeIdx_) th.push_back(edges_[i]->getPhase());
  return th;
}

void EigenstateSynthesis::setInteriorWeights(const std::vector<double> &w) {
  if (w.size() != interiorEdgeIdx_.size())
    throw std::runtime_error(
        "EigenstateSynthesis::setInteriorWeights: got " +
        std::to_string(w.size()) + " weights, expected " +
        std::to_string(interiorEdgeIdx_.size()));
  for (std::size_t k = 0; k < interiorEdgeIdx_.size(); ++k)
    edges_[interiorEdgeIdx_[k]]->setLength(std::sqrt(std::complex<double>{w[k], 0.0}));
}

void EigenstateSynthesis::setInteriorPhases(
    const std::vector<std::complex<double>> &theta) {
  if (theta.size() != interiorEdgeIdx_.size())
    throw std::runtime_error(
        "EigenstateSynthesis::setInteriorPhases: got " +
        std::to_string(theta.size()) + " phases, expected " +
        std::to_string(interiorEdgeIdx_.size()));
  for (std::size_t k = 0; k < interiorEdgeIdx_.size(); ++k)
    edges_[interiorEdgeIdx_[k]]->setPhase(theta[k]);
}

std::vector<::tessera::mesh::VertexIdPair>
EigenstateSynthesis::boundaryEdges() const {
  std::vector<::tessera::mesh::VertexIdPair> out;
  out.reserve(boundaryEdgeIdx_.size());
  for (const auto i : boundaryEdgeIdx_) {
    out.push_back(edges_[i]->ascendingIds());
  }
  return out;
}

std::vector<::tessera::mesh::VertexIdPair>
EigenstateSynthesis::interiorEdges() const {
  std::vector<::tessera::mesh::VertexIdPair> out;
  out.reserve(interiorEdgeIdx_.size());
  for (const auto i : interiorEdgeIdx_) {
    out.push_back(edges_[i]->ascendingIds());
  }
  return out;
}

bool EigenstateSynthesis::growInterior(std::uint64_t seed) {
  if (!st_) return false;
  // Cone a fresh interior vertex via the boundary-fixed pre-geometric Pachner
  // add: a 1→(d+1) stellar subdivision, always interior, so ∂W is left exactly
  // fixed (the move never touches a boundary face).
  ::tessera::spacetime::AddMove move(
      st_.get(), seed, /*relabelEnabled=*/false,
      ::tessera::spacetime::PachnerMode::PreGeometric, /*boundaryFixed=*/true);
  if (!move.propose()) return false;
  if (!move.apply()) return false;
  // The vertex set changed: rebuild the operator over the fresh vertex order
  // (the new apex has the largest id, so it appends last in sorted order and
  // the existing psi indices are preserved), then re-capture the tunable edges
  // and the interior/boundary partition.
  laplacian_ = HodgeLaplacian(st_, HodgeLaplacian::defaultWeightConvention(), metricSource_);
  capture();
  classifyBoundary();
  return true;
}

}  // namespace tessera::cobordism
