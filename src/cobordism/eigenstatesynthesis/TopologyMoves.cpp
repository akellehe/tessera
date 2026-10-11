// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

// EigenstateSynthesis (include/cobordism/EigenstateSynthesis.h): the gated
// topology moves: the checked cut and the composed stellar move. One of the
// translation units that define the class's members by topic.

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

// === Gated topology moves: the checked cut and the composed stellar move ===

std::pair<bool, std::string> EigenstateSynthesis::removeInteriorCellChecked(
    const std::vector<std::uint64_t> &cell) {
  if (!removeInteriorCell(cell))
    return {false,
            "not an interior top cell (or the removal would touch dW)"};
  const auto verdict = dualComplexValid();  // {true, "ok"} when the dual holds
  if (!verdict.first) restoreLastRemoval();
  return verdict;
}

std::pair<bool, std::string> EigenstateSynthesis::stellarSubdivideInterior(
    const std::vector<std::uint64_t> &cell) {
  if (!st_) return {false, "no spacetime"};
  const int d = st_->getMetric()->getSignature()->getDimensions();
  const std::size_t topVerts = (d >= 0) ? static_cast<std::size_t>(d) + 1 : 0;
  if (topVerts < 2 || cell.size() != topVerts)
    return {false, "cell is not a top cell (" + std::to_string(cell.size()) +
                       " vertices, expected " + std::to_string(topVerts) + ")"};

  // The facet fan: `cell` with each vertex dropped in turn (its d+1 codim-one
  // facets). The attach cones the fresh vertex onto every facet, so the parent
  // cell's 1-skeleton survives its removal (each parent edge keeps a fan
  // coface) and the subdivision is exactly 1 -> (d+1).
  std::vector<std::uint64_t> want(cell.begin(), cell.end());
  std::sort(want.begin(), want.end());
  std::vector<std::vector<std::uint64_t>> fan;
  fan.reserve(want.size());
  for (std::size_t skip = 0; skip < want.size(); ++skip) {
    std::vector<std::uint64_t> facet;
    facet.reserve(want.size() - 1);
    for (std::size_t i = 0; i < want.size(); ++i)
      if (i != skip) facet.push_back(want[i]);
    fan.push_back(std::move(facet));
  }

  if (!attachInteriorVertex(fan))
    return {false, "attach rejected (invalid fan spec, or dW perturbed)"};
  if (!removeInteriorCell(want)) {
    detachLastInteriorVertex();
    return {false,
            "not an interior top cell (or the removal would touch dW)"};
  }
  const auto verdict = dualComplexValid();
  if (!verdict.first) {
    // LIFO rollback: the removal happened after the attach.
    restoreLastRemoval();
    detachLastInteriorVertex();
    return verdict;
  }

  // The uniform re-pin: the seeds are built with every edge at squared length 1
  // and phase 0 (the unit cochain metric), and the move holds that by
  // construction rather than by the time-rule coincidence on all-same-time
  // seeds.
  for (const auto e : st_->getEdgeList()->toVector()) {
    if (e == nullptr) continue;
    e->setLength({1.0, 0.0});  // spacelike unit length
    e->setPhase(0.0);
  }
  return verdict;  // {true, "ok"}
}

std::vector<std::vector<std::uint64_t>> EigenstateSynthesis::topCells() const {
  std::vector<std::vector<std::uint64_t>> cells;
  if (!st_) return cells;
  const int d = st_->getMetric()->getSignature()->getDimensions();
  const std::size_t topVerts = (d >= 0) ? static_cast<std::size_t>(d) + 1 : 0;
  if (topVerts == 0) return cells;
  for (const auto s : st_->getSimplices()) {
    if (s == nullptr) continue;
    if (s->size() != topVerts) continue;
    std::vector<std::uint64_t> ids;
    ids.reserve(topVerts);
    for (const auto v : s->getVertices())
      if (v != nullptr) ids.push_back(v->getId());
    if (ids.size() != topVerts) continue;
    std::sort(ids.begin(), ids.end());
    cells.push_back(std::move(ids));
  }
  std::sort(cells.begin(), cells.end());
  cells.erase(std::unique(cells.begin(), cells.end()), cells.end());
  return cells;
}

std::pair<bool, std::string> EigenstateSynthesis::dualComplexValid() const {
  const auto tops = topCells();
  if (tops.empty()) return {false, "no top cells"};
  const int dim = static_cast<int>(tops.front().size()) - 1;
  // The dangling-facet check needs the (n-1)-cell universe, which
  // cellSimplices() is when the synthesis degree sits one below the top
  // dimension (the register layers). At other degrees the facet universe is not
  // tracked here, so only the top-cell conditions are checked.
  const bool facetDegree = (k_ == dim - 1);
  return ChainComplex::dualComplexIsValid(
      tops, dim,
      facetDegree ? cellSimplices()
                  : std::vector<std::vector<std::uint64_t>>{});
}

}  // namespace tessera::cobordism
