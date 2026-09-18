// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "observables/LiveComplex.h"

#include <algorithm>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <string>

#include "mesh/Edge.h"
#include "mesh/EdgeList.h"
#include "mesh/Vertex.h"
#include "mesh/VertexList.h"
#include "spacetime/Metric.h"
#include "spacetime/Signature.h"
#include "spacetime/Spacetime.h"

namespace tessera::observables {
using ::tessera::mesh::Edge;

std::shared_ptr<Spacetime> LiveComplex::load(
    const std::vector<std::vector<std::uint64_t>> &cells,
    const std::map<std::pair<std::uint64_t, std::uint64_t>,
                   std::complex<double>> &squaredLengths,
    const std::map<std::uint64_t, double> &vertexTimes, int dimensions) {
  if (cells.empty()) {
    throw std::invalid_argument("LiveComplex::load needs at least one top cell");
  }
  // fromVertexTuples lays down the top cells; `dimensions` is the recorded complex
  // dimension, passed through rather than inferred. The metric and vertex times
  // are then restored as recorded and the facet skeleton is completed.
  auto st = Spacetime::fromVertexTuples(dimensions, cells, 1.0, 0.0);
  if (!vertexTimes.empty()) {
    const auto &vertexList = st->getVertexList();
    for (const auto &kv : vertexTimes) {
      vertexList->get(kv.first)->setTime(kv.second);
    }
  }
  // The edges of a freshly loaded complex are mutable: restoring the recorded
  // squared lengths is what rehydration means, not a change to emergent state.
  for (Edge *e : st->getEdgeList()->toVector()) {
    const std::uint64_t a = e->getSource()->getId();
    const std::uint64_t b = e->getTarget()->getId();
    auto it = squaredLengths.find(a < b ? std::make_pair(a, b)
                                        : std::make_pair(b, a));
    if (it == squaredLengths.end()) {
      throw std::out_of_range(
          "LiveComplex::load: edge (" + std::to_string(std::min(a, b)) + ", " +
          std::to_string(std::max(a, b)) +
          ") of the loaded complex has no recorded squared length — a partial "
          "metric is never silently defaulted");
    }
    e->setLength(std::sqrt(it->second));
  }
  // Complete the facet/coface skeleton that the dual-volume and deficit reads
  // walk. fromVertexTuples leaves only the top cells; this builds the same skeleton a
  // ReggeSolver or ChainComplex pass would.
  st->materializeFacets();
  return st;
}

std::shared_ptr<Spacetime> LiveComplex::subcomplex(
    const std::vector<std::vector<std::uint64_t>> &cells, int dimensions) {
  if (cells.empty()) {
    throw std::invalid_argument(
        "LiveComplex::subcomplex needs at least one cell");
  }
  // The cells are selected by the caller from the ambient top cells, and
  // `dimensions` is the ambient complex's dimension. This re-instantiates that
  // selection with a uniform metric for the block-residual diagnostic.
  return Spacetime::fromVertexTuples(dimensions, cells, 1.0, 0.0);
}

LiveComplex::Relabeled LiveComplex::relabel(const Spacetime &spacetime,
                                            std::uint64_t seed) {
  // Read the recorded geometry off the live complex (const reads only). The
  // dimension comes from the metric signature, not from a cell-size guess.
  const int dimensions = spacetime.getMetric()->getSignature()->getDimensions();
  std::vector<std::vector<std::uint64_t>> cells;
  cells.reserve(spacetime.getTopSimplices().size());
  for (const auto *c : spacetime.getTopSimplices()) {
    std::vector<std::uint64_t> vids;
    vids.reserve(c->getVertices().size());
    for (const auto *v : c->getVertices()) vids.push_back(v->getId());
    cells.push_back(std::move(vids));
  }
  std::map<std::pair<std::uint64_t, std::uint64_t>, std::complex<double>> edges;
  for (const auto *e : spacetime.getEdgeList()->toVector()) {
    const std::uint64_t a = e->getSource()->getId();
    const std::uint64_t b = e->getTarget()->getId();
    edges[a < b ? std::make_pair(a, b) : std::make_pair(b, a)] =
        (e->getLength() * e->getLength());
  }
  std::map<std::uint64_t, double> times;
  const auto &vertexList = spacetime.getVertexList();
  for (const auto &cell : cells) {
    for (std::uint64_t v : cell) {
      if (!times.count(v)) times[v] = vertexList->get(v)->getTime();
    }
  }

  // A random vertex-id permutation and cell-order shuffle, both seed-determined.
  std::set<std::uint64_t> uniqueVertices;
  for (const auto &cell : cells) {
    for (std::uint64_t v : cell) uniqueVertices.insert(v);
  }
  std::vector<std::uint64_t> allVertices(uniqueVertices.begin(),
                                         uniqueVertices.end());
  std::vector<std::uint64_t> shuffled = allVertices;
  std::mt19937_64 rng(seed);
  std::shuffle(shuffled.begin(), shuffled.end(), rng);
  std::map<std::uint64_t, std::uint64_t> perm;
  for (std::size_t i = 0; i < allVertices.size(); ++i) {
    perm[allVertices[i]] = shuffled[i];
  }

  std::vector<std::size_t> order(cells.size());
  std::iota(order.begin(), order.end(), 0);
  std::shuffle(order.begin(), order.end(), rng);

  std::vector<std::vector<std::uint64_t>> permutedCells;
  permutedCells.reserve(cells.size());
  for (std::size_t idx : order) {
    std::vector<std::uint64_t> permuted;
    permuted.reserve(cells[idx].size());
    for (std::uint64_t v : cells[idx]) permuted.push_back(perm.at(v));
    permutedCells.push_back(std::move(permuted));
  }
  std::map<std::pair<std::uint64_t, std::uint64_t>, std::complex<double>>
      permutedEdges;
  for (const auto &kv : edges) {
    const std::uint64_t a = perm.at(kv.first.first);
    const std::uint64_t b = perm.at(kv.first.second);
    permutedEdges[a < b ? std::make_pair(a, b) : std::make_pair(b, a)] =
        kv.second;
  }
  std::map<std::uint64_t, double> permutedTimes;
  for (const auto &kv : times) permutedTimes[perm.at(kv.first)] = kv.second;

  Relabeled out;
  out.spacetime = load(permutedCells, permutedEdges, permutedTimes, dimensions);
  out.vertexMap = std::move(perm);
  return out;
}

}  // namespace tessera::observables
