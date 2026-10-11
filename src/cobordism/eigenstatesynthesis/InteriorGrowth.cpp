// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

// EigenstateSynthesis (include/cobordism/EigenstateSynthesis.h): free interior
// connectivity, the general growth primitive. One of the translation units that
// define the class's members by topic.

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

// === Free interior connectivity (general growth primitive) ===

void EigenstateSynthesis::rollbackAttachment(const Attachment &att) {
  // Remove in dependency order: simplices reference edges and vertices, edges
  // reference vertices. removeVertex also drops any edge still incident to the
  // vertex, so removing the created edges first (some, among the spec's own
  // vertices, are not incident to the new vertex) leaves only the vertex.
  for (auto *s : att.createdSimplices)
    if (s != nullptr) st_->removeSimplex(s);
  for (auto *e : att.createdEdges)
    if (e != nullptr) st_->removeEdge(e);
  if (att.vertex != nullptr) st_->removeVertex(att.vertex);
}

bool EigenstateSynthesis::attachInteriorVertex(
    const std::vector<std::vector<std::uint64_t>> &incidentSimplices) {
  if (!st_) return false;
  if (incidentSimplices.empty()) return false;  // would isolate the new vertex

  // The spec may reference only existing vertices. Build id -> Vertex* and the
  // current max id in one pass.
  std::unordered_map<std::uint64_t, ::tessera::mesh::Vertex *> idToVert;
  std::uint64_t maxId = 0;
  bool anyVert = false;
  for (const auto v : st_->getVertexList()->toVector()) {
    if (v == nullptr) continue;
    idToVert.emplace(v->getId(), v);
    maxId = anyVert ? std::max(maxId, v->getId()) : v->getId();
    anyVert = true;
  }
  if (!anyVert) return false;
  for (const auto &spec : incidentSimplices) {
    if (spec.empty()) return false;  // a face needs >= 1 existing vertex
    // spec ∪ {new vertex} is one simplex; the Fingerprint caps a simplex at kMax
    // vertices (createSimplex would otherwise throw mid-mutation).
    if (spec.size() + 1 > ::tessera::mesh::kMax) return false;
    std::unordered_set<std::uint64_t> seen;
    for (const std::uint64_t id : spec) {
      if (idToVert.find(id) == idToVert.end()) return false;  // dangling ref
      if (!seen.insert(id).second) return false;              // duplicate
    }
  }

  // Snapshot the pinned boundary (id-pair -> (complex w, theta)) for the
  // bit-exact check. The whole complex l2 is compared, not (Re, phase), so the
  // dW invariant catches Im-only corruption too.
  std::map<std::pair<std::uint64_t, std::uint64_t>,
           std::pair<std::complex<double>, std::complex<double>>>
      boundaryBefore;
  for (const auto i : boundaryEdgeIdx_) {
    const std::uint64_t a = edges_[i]->getSource()->getId();
    const std::uint64_t b = edges_[i]->getTarget()->getId();
    boundaryBefore[{std::min(a, b), std::max(a, b)}] = {
        edges_[i]->squaredLength(), edges_[i]->getPhase()};
  }

  // Fresh interior vertex with the largest id, so it sorts last and preserves
  // the boundary-support psi prefix. maxId+1 rather than the vertexIdCounter,
  // which can be stale relative to explicitly-id'd fixture vertices.
  Attachment att;
  ::tessera::mesh::Vertex *vnew = st_->createVertex(maxId + 1);
  att.vertex = vnew;

  // Create one simplex per spec; createSimplexTracked materializes the simplex's
  // full 1-skeleton (every pairwise edge) and reports the freshly inserted edges
  // so detach can undo exactly.
  for (const auto &spec : incidentSimplices) {
    ::tessera::mesh::VertexPtrs verts;
    verts.reserve(spec.size() + 1);
    for (const std::uint64_t id : spec) verts.push_back(idToVert[id]);
    verts.push_back(vnew);
    const auto res = st_->createSimplexTracked(verts);
    if (res.created && res.simplex != nullptr)
      att.createdSimplices.push_back(res.simplex);
    for (const auto e : res.newEdges)
      if (e != nullptr) att.createdEdges.push_back(e);
  }

  // Re-capture so the operator / partition track the grown complex.
  laplacian_ = HodgeLaplacian(st_, HodgeLaplacian::defaultWeightConvention(), metricSource_);
  capture();
  classifyBoundary();

  // Validate the only two invariants the experiment allows.
  // (a) Valid downward-closed complex: every pair within each new simplex carries
  //     an edge (createSimplexTracked guarantees this — assert it as a real gate).
  std::set<std::pair<std::uint64_t, std::uint64_t>> edgeKeys;
  for (const auto e : edges_) {
    const std::uint64_t a = e->getSource()->getId();
    const std::uint64_t b = e->getTarget()->getId();
    edgeKeys.insert({std::min(a, b), std::max(a, b)});
  }
  bool valid = true;
  for (const auto &spec : incidentSimplices) {
    std::vector<std::uint64_t> ids(spec.begin(), spec.end());
    ids.push_back(vnew->getId());
    for (std::size_t i = 0; valid && i + 1 < ids.size(); ++i)
      for (std::size_t j = i + 1; valid && j < ids.size(); ++j) {
        const std::pair<std::uint64_t, std::uint64_t> key{
            std::min(ids[i], ids[j]), std::max(ids[i], ids[j])};
        if (edgeKeys.find(key) == edgeKeys.end()) valid = false;
      }
  }
  // (b) Pinned boundary dW bit-exact: same edge set, same weights/phases.
  if (valid) {
    std::size_t matched = 0;
    for (const auto i : boundaryEdgeIdx_) {
      const std::uint64_t a = edges_[i]->getSource()->getId();
      const std::uint64_t b = edges_[i]->getTarget()->getId();
      const auto it =
          boundaryBefore.find({std::min(a, b), std::max(a, b)});
      if (it == boundaryBefore.end() ||
          it->second.first != edges_[i]->squaredLength() ||
          it->second.second != edges_[i]->getPhase()) {
        valid = false;
        break;
      }
      ++matched;
    }
    if (matched != boundaryBefore.size()) valid = false;  // a boundary edge vanished
  }

  if (!valid) {
    rollbackAttachment(att);
    laplacian_ = HodgeLaplacian(st_, HodgeLaplacian::defaultWeightConvention(), metricSource_);
    capture();
    classifyBoundary();
    return false;
  }
  attachments_.push_back(std::move(att));
  return true;
}

bool EigenstateSynthesis::detachLastInteriorVertex() {
  if (!st_ || attachments_.empty()) return false;
  const Attachment att = std::move(attachments_.back());
  attachments_.pop_back();
  rollbackAttachment(att);
  laplacian_ = HodgeLaplacian(st_, HodgeLaplacian::defaultWeightConvention(), metricSource_);
  capture();
  classifyBoundary();
  return true;
}

std::vector<std::uint64_t> EigenstateSynthesis::vertexIds() const {
  std::vector<std::uint64_t> ids;
  if (!st_) return ids;
  for (const auto v : st_->getVertexList()->toVector())
    if (v != nullptr) ids.push_back(v->getId());
  std::sort(ids.begin(), ids.end());
  ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
  return ids;
}

std::vector<std::uint64_t> EigenstateSynthesis::boundaryVertexIds() const {
  return boundaryVertexIdsSorted_;
}

}  // namespace tessera::cobordism
