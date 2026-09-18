// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "cobordism/EigenstateSynthesis.h"

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

EigenstateSynthesis::EigenstateSynthesis(std::shared_ptr<Spacetime> st, int k,
                                         HodgeLaplacian::MetricSource metricSource)
    : st_(st), k_(k), metricSource_(metricSource),
      laplacian_(std::move(st), HodgeLaplacian::defaultWeightConvention(), metricSource) {
  if (k_ < 0)
    throw std::runtime_error(
        "EigenstateSynthesis: degree k must be non-negative; got k=" +
        std::to_string(k_));
  if (!st_) return;
  capture();
  classifyBoundary();
}

void EigenstateSynthesis::capture() {
  order_ = 0;
  edges_.clear();
  cellOrdering_.clear();
  if (!st_) return;

  // The psi ordering: at k=0 the sorted-id vertex set (one component per vertex),
  // matching HodgeLaplacian's k=0 indexing; at k>=1 the canonical ChainComplex
  // k-cell column order, matching the metric L_k the operator assembles. order_
  // is the operator dimension N either way.
  std::unordered_set<std::uint64_t> idset;
  for (const auto v : st_->getVertexList()->toVector())
    if (v != nullptr) idset.insert(v->getId());

  if (k_ == 0) {
    std::vector<std::uint64_t> ids(idset.begin(), idset.end());
    std::sort(ids.begin(), ids.end());
    cellOrdering_.reserve(ids.size());
    for (const std::uint64_t id : ids) cellOrdering_.push_back({id});
  } else {
    cellOrdering_ = ChainComplex::fromSpacetime(*st_).kSimplexVertices(k_);
  }
  order_ = cellOrdering_.size();

  // Stable edge order: the tunable edges in EdgeList order — those carrying
  // weight in L = D - A (both endpoints present in the vertex set, not a
  // self-loop). This is HodgeLaplacian::assemble's edge filter, so the exposed
  // {w_ij, theta_ij} are the parameters the Laplacian reads.
  for (const auto e : st_->getEdgeList()->toVector()) {
    if (e == nullptr) continue;
    const auto s = e->getSource();
    const auto t = e->getTarget();
    if (s == nullptr || t == nullptr) continue;
    if (s->getId() == t->getId()) continue;
    if (idset.find(s->getId()) == idset.end() ||
        idset.find(t->getId()) == idset.end())
      continue;
    edges_.push_back(e);
  }
}

void EigenstateSynthesis::classifyBoundary() {
  interiorEdgeIdx_.clear();
  boundaryEdgeIdx_.clear();
  boundaryVertexIdsSorted_.clear();
  interiorVertexCount_ = 0;
  if (!st_) return;

  // Top cells have d+1 vertices (d = metric dimension), matching the
  // Spacetime's own top-simplex bookkeeping (getRandomTopSimplex) so this
  // partition agrees with the pre-geometric Pachner moves growInterior reuses.
  const int d = st_->getMetric()->getSignature()->getDimensions();
  const std::size_t topVerts = (d >= 0) ? static_cast<std::size_t>(d) + 1 : 0;

  // ∂W: codim-1 faces (a top cell with one vertex dropped) belonging to exactly
  // one top cell. An edge sits on the boundary only once codim-1 faces are at
  // least edges themselves (topVerts >= 3); below that there is no boundary and
  // every tunable edge is interior.
  std::set<std::pair<std::uint64_t, std::uint64_t>> boundaryEdgeKeys;
  std::unordered_set<std::uint64_t> boundaryVertexIds;
  if (topVerts >= 3) {
    std::map<std::vector<std::uint64_t>, int> facetCount;
    for (const auto s : st_->getSimplices()) {
      if (s == nullptr) continue;
      if (s->size() != topVerts) continue;
      std::vector<std::uint64_t> ids;
      ids.reserve(topVerts);
      for (const auto v : s->getVertices())
        if (v != nullptr) ids.push_back(v->getId());
      if (ids.size() != topVerts) continue;
      std::sort(ids.begin(), ids.end());
      for (std::size_t skip = 0; skip < ids.size(); ++skip) {
        std::vector<std::uint64_t> facet;
        facet.reserve(ids.size() - 1);
        for (std::size_t i = 0; i < ids.size(); ++i)
          if (i != skip) facet.push_back(ids[i]);
        ++facetCount[facet];
      }
    }
    for (const auto &[facet, count] : facetCount) {
      if (count != 1) continue;  // interior facet (shared by two top cells)
      for (const std::uint64_t id : facet) boundaryVertexIds.insert(id);
      // facet is sorted ascending, so (facet[i], facet[j]) for i<j is ordered.
      for (std::size_t i = 0; i + 1 < facet.size(); ++i)
        for (std::size_t j = i + 1; j < facet.size(); ++j)
          boundaryEdgeKeys.insert({facet[i], facet[j]});
    }
  }

  // An edge is on ∂W iff some boundary facet contains both endpoints; otherwise
  // it is interior (free).
  for (std::size_t e = 0; e < edges_.size(); ++e) {
    const std::uint64_t a = edges_[e]->getSource()->getId();
    const std::uint64_t b = edges_[e]->getTarget()->getId();
    const std::pair<std::uint64_t, std::uint64_t> key =
        a < b ? std::make_pair(a, b) : std::make_pair(b, a);
    if (boundaryEdgeKeys.find(key) != boundaryEdgeKeys.end())
      boundaryEdgeIdx_.push_back(e);
    else
      interiorEdgeIdx_.push_back(e);
  }

  // Interior vertices: those on no boundary face (the coned-in apexes).
  std::unordered_set<std::uint64_t> seen;
  for (const auto v : st_->getVertexList()->toVector()) {
    if (v == nullptr) continue;
    if (!seen.insert(v->getId()).second) continue;
    if (boundaryVertexIds.find(v->getId()) == boundaryVertexIds.end())
      ++interiorVertexCount_;
  }

  // Persist the boundary vertex set (sorted) for boundaryVertexIds() — the
  // candidate pool a "boundary-star" connectivity search wires into.
  boundaryVertexIdsSorted_.assign(boundaryVertexIds.begin(),
                                  boundaryVertexIds.end());
  std::sort(boundaryVertexIdsSorted_.begin(), boundaryVertexIdsSorted_.end());
}

std::vector<cd> EigenstateSynthesis::apply(const std::vector<cd> &psi) const {
  const std::size_t N = order_;
  if (psi.size() != N)
    throw std::runtime_error(
        "EigenstateSynthesis::apply: psi has length " +
        std::to_string(psi.size()) + ", expected " + std::to_string(N));
  std::vector<cd> out(N, cd(0.0, 0.0));
  if (N == 0) return out;
  // Reassembled from the live edges on each call (see readoutLaplacian): the
  // U(1) connection L = D - A at k = 0, the Hodge L_k at k >= 1.
  const std::vector<cd> L = readoutLaplacian();
  for (std::size_t i = 0; i < N; ++i) {
    cd acc(0.0, 0.0);
    for (std::size_t j = 0; j < N; ++j) acc += L[i * N + j] * psi[j];
    out[i] = acc;
  }
  return out;
}

double EigenstateSynthesis::residual(const std::vector<cd> &psi) const {
  const std::size_t N = order_;
  if (psi.size() != N)
    throw std::runtime_error(
        "EigenstateSynthesis::residual: psi has length " +
        std::to_string(psi.size()) + ", expected " + std::to_string(N));
  if (N == 0) return 0.0;

  // Normalize: r and the eigenvector condition are scale-invariant, and r is
  // defined for a unit target.
  double nrm2 = 0.0;
  for (const cd &c : psi) nrm2 += std::norm(c);
  if (nrm2 <= 0.0) return 0.0;
  const double inv = 1.0 / std::sqrt(nrm2);
  std::vector<cd> p(N);
  for (std::size_t i = 0; i < N; ++i) p[i] = psi[i] * inv;

  const std::vector<cd> Lp = apply(p);
  // lambda = p^dagger L p (real for Hermitian L; take the real part).
  cd lam(0.0, 0.0);
  for (std::size_t i = 0; i < N; ++i) lam += std::conj(p[i]) * Lp[i];
  const double lambda = lam.real();

  // r = || L p - lambda p ||^2 = ||(I - p p^dagger) L p||^2 (p unit).
  double r = 0.0;
  for (std::size_t i = 0; i < N; ++i) r += std::norm(Lp[i] - lambda * p[i]);
  return r;
}

double EigenstateSynthesis::rayleigh(const std::vector<cd> &psi) const {
  const std::size_t N = order_;
  if (psi.size() != N)
    throw std::runtime_error(
        "EigenstateSynthesis::rayleigh: psi has length " +
        std::to_string(psi.size()) + ", expected " + std::to_string(N));
  if (N == 0) return 0.0;

  const std::vector<cd> Lp = apply(psi);
  cd num(0.0, 0.0);
  double den = 0.0;
  for (std::size_t i = 0; i < N; ++i) {
    num += std::conj(psi[i]) * Lp[i];
    den += std::norm(psi[i]);
  }
  if (den <= 0.0) return 0.0;
  return num.real() / den;
}

std::vector<std::complex<double>> EigenstateSynthesis::weights() const {
  std::vector<std::complex<double>> w;
  w.reserve(edges_.size());
  for (const auto e : edges_) w.push_back((e->getLength() * e->getLength()));
  return w;
}

std::vector<std::complex<double>> EigenstateSynthesis::phases() const {
  std::vector<std::complex<double>> th;
  th.reserve(edges_.size());
  for (const auto e : edges_) th.push_back(e->getPhase());
  return th;
}

void EigenstateSynthesis::setWeights(const std::vector<double> &w) {
  if (w.size() != edges_.size())
    throw std::runtime_error(
        "EigenstateSynthesis::setWeights: got " + std::to_string(w.size()) +
        " weights, expected " + std::to_string(edges_.size()));
  for (std::size_t i = 0; i < edges_.size(); ++i)
    edges_[i]->setLength(std::sqrt(std::complex<double>{w[i], 0.0}));
}

void EigenstateSynthesis::setPhases(
    const std::vector<std::complex<double>> &theta) {
  if (theta.size() != edges_.size())
    throw std::runtime_error(
        "EigenstateSynthesis::setPhases: got " + std::to_string(theta.size()) +
        " phases, expected " + std::to_string(edges_.size()));
  for (std::size_t i = 0; i < edges_.size(); ++i)
    edges_[i]->setPhase(theta[i]);
}

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

std::vector<std::pair<std::uint64_t, std::uint64_t>>
EigenstateSynthesis::boundaryEdges() const {
  std::vector<std::pair<std::uint64_t, std::uint64_t>> out;
  out.reserve(boundaryEdgeIdx_.size());
  for (const auto i : boundaryEdgeIdx_) {
    const std::uint64_t a = edges_[i]->getSource()->getId();
    const std::uint64_t b = edges_[i]->getTarget()->getId();
    out.emplace_back(std::min(a, b), std::max(a, b));
  }
  return out;
}

std::vector<std::pair<std::uint64_t, std::uint64_t>>
EigenstateSynthesis::interiorEdges() const {
  std::vector<std::pair<std::uint64_t, std::uint64_t>> out;
  out.reserve(interiorEdgeIdx_.size());
  for (const auto i : interiorEdgeIdx_) {
    const std::uint64_t a = edges_[i]->getSource()->getId();
    const std::uint64_t b = edges_[i]->getTarget()->getId();
    out.emplace_back(std::min(a, b), std::max(a, b));
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
        (edges_[i]->getLength() * edges_[i]->getLength()), edges_[i]->getPhase()};
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
          it->second.first != (edges_[i]->getLength() * edges_[i]->getLength()) ||
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

// === Surgery: the topology-changing interior remove move ===

std::vector<std::vector<std::uint64_t>>
EigenstateSynthesis::interiorTopCells() const {
  std::vector<std::vector<std::uint64_t>> cells;
  if (!st_) return cells;
  const int d = st_->getMetric()->getSignature()->getDimensions();
  const std::size_t topVerts = (d >= 0) ? static_cast<std::size_t>(d) + 1 : 0;
  if (topVerts == 0) return cells;
  const std::unordered_set<std::uint64_t> bverts(
      boundaryVertexIdsSorted_.begin(), boundaryVertexIdsSorted_.end());
  for (const auto s : st_->getSimplices()) {
    if (s == nullptr || s->size() != topVerts) continue;
    std::vector<std::uint64_t> ids;
    ids.reserve(topVerts);
    bool allInterior = true;
    for (const auto v : s->getVertices()) {
      if (v == nullptr) { allInterior = false; break; }
      ids.push_back(v->getId());
      if (bverts.find(v->getId()) != bverts.end()) allInterior = false;
    }
    if (!allInterior || ids.size() != topVerts) continue;
    std::sort(ids.begin(), ids.end());
    cells.push_back(std::move(ids));
  }
  std::sort(cells.begin(), cells.end());
  cells.erase(std::unique(cells.begin(), cells.end()), cells.end());
  return cells;
}

bool EigenstateSynthesis::removeInteriorCell(
    const std::vector<std::uint64_t> &cell) {
  if (!st_) return false;
  const int d = st_->getMetric()->getSignature()->getDimensions();
  const std::size_t topVerts = (d >= 0) ? static_cast<std::size_t>(d) + 1 : 0;
  if (topVerts < 2 || cell.size() != topVerts) return false;
  std::vector<std::uint64_t> want(cell.begin(), cell.end());
  std::sort(want.begin(), want.end());

  // The cell must be interior: no boundary vertex (so no ∂W face is removed).
  const std::unordered_set<std::uint64_t> bverts(
      boundaryVertexIdsSorted_.begin(), boundaryVertexIdsSorted_.end());
  for (const std::uint64_t id : want)
    if (bverts.find(id) != bverts.end()) return false;

  // Locate the matching top simplex, and collect the other top cells' vertex
  // sets (to tell which of `want`'s edges remain covered after removal). Refuse
  // to remove the last top cell of the top dimension (it would drop the complex
  // dimension and promote orphan facets to top cells).
  ::tessera::mesh::Simplex *target = nullptr;
  std::vector<std::vector<std::uint64_t>> otherTop;
  for (const auto s : st_->getSimplices()) {
    if (s == nullptr || s->size() != topVerts) continue;
    std::vector<std::uint64_t> ids;
    ids.reserve(topVerts);
    for (const auto v : s->getVertices())
      if (v != nullptr) ids.push_back(v->getId());
    std::sort(ids.begin(), ids.end());
    if (target == nullptr && ids == want)
      target = s;
    else
      otherTop.push_back(std::move(ids));
  }
  if (target == nullptr || otherTop.empty()) return false;

  // An edge {u,v} of the cell is orphaned iff no other top cell contains both
  // endpoints. Map endpoint pairs -> Edge* for the orphaned ones.
  std::map<std::pair<std::uint64_t, std::uint64_t>, ::tessera::mesh::Edge *>
      edgeByPair;
  for (const auto e : edges_) {
    const std::uint64_t a = e->getSource()->getId();
    const std::uint64_t b = e->getTarget()->getId();
    edgeByPair[{std::min(a, b), std::max(a, b)}] = e;
  }
  const auto covered = [&](std::uint64_t u, std::uint64_t v) {
    for (const auto &c : otherTop) {
      const bool hu = std::find(c.begin(), c.end(), u) != c.end();
      const bool hv = std::find(c.begin(), c.end(), v) != c.end();
      if (hu && hv) return true;
    }
    return false;
  };

  Removal rem;
  rem.cell = want;
  std::vector<::tessera::mesh::Edge *> toRemove;
  for (std::size_t i = 0; i + 1 < want.size(); ++i)
    for (std::size_t j = i + 1; j < want.size(); ++j) {
      const std::uint64_t u = want[i];
      const std::uint64_t v = want[j];
      if (covered(u, v)) continue;  // edge survives in another top cell
      // edgeByPair is keyed {min, max}; canonicalize here too rather than
      // leaning on `want` having been sorted fifty lines up. A miss would
      // silently leave an orphaned edge in place and out of removedEdges,
      // so the rollback could not undo the move.
      const auto it = edgeByPair.find({std::min(u, v), std::max(u, v)});
      if (it == edgeByPair.end()) continue;  // already absent
      rem.removedEdges.emplace_back(u, v, it->second->getLength(),
                                    it->second->getPhase());
      toRemove.push_back(it->second);
    }

  // Snapshot ∂W (id-pair -> (complex w, theta)) for the bit-exact check. The
  // whole complex l2, not (Re, phase): the dW invariant covers Im corruption.
  std::map<std::pair<std::uint64_t, std::uint64_t>,
           std::pair<std::complex<double>, std::complex<double>>>
      boundaryBefore;
  for (const auto i : boundaryEdgeIdx_) {
    const std::uint64_t a = edges_[i]->getSource()->getId();
    const std::uint64_t b = edges_[i]->getTarget()->getId();
    boundaryBefore[{std::min(a, b), std::max(a, b)}] = {
        (edges_[i]->getLength() * edges_[i]->getLength()), edges_[i]->getPhase()};
  }

  // Mutate: drop the top cell, then its orphaned edges.
  st_->removeSimplex(target);
  for (auto *e : toRemove)
    if (e != nullptr) st_->removeEdge(e);

  laplacian_ = HodgeLaplacian(st_, HodgeLaplacian::defaultWeightConvention(), metricSource_);
  capture();
  classifyBoundary();

  // ∂W must be preserved bit-exactly: every previously-boundary edge still
  // present with the same weight/phase. Newly exposed boundary edges — the
  // opened hole — are allowed, so this is a subset check, not equality.
  bool valid = true;
  std::map<std::pair<std::uint64_t, std::uint64_t>,
           std::pair<std::complex<double>, std::complex<double>>>
      liveWeights;
  for (const auto e : edges_) {
    const std::uint64_t a = e->getSource()->getId();
    const std::uint64_t b = e->getTarget()->getId();
    const std::pair<std::uint64_t, std::uint64_t> key{std::min(a, b),
                                                      std::max(a, b)};
    liveWeights[key] = {(e->getLength() * e->getLength()), e->getPhase()};
  }
  for (const auto &[key, wp] : boundaryBefore) {
    const auto it = liveWeights.find(key);
    if (it == liveWeights.end() || it->second.first != wp.first ||
        it->second.second != wp.second) {
      valid = false;
      break;
    }
  }

  if (!valid) {
    applyRestore(rem);
    laplacian_ = HodgeLaplacian(st_, HodgeLaplacian::defaultWeightConvention(), metricSource_);
    capture();
    classifyBoundary();
    return false;
  }
  removals_.push_back(std::move(rem));
  return true;
}

bool EigenstateSynthesis::applyRestore(const Removal &rem) {
  if (!st_) return false;
  std::unordered_map<std::uint64_t, ::tessera::mesh::Vertex *> idToVert;
  for (const auto v : st_->getVertexList()->toVector())
    if (v != nullptr) idToVert.emplace(v->getId(), v);
  ::tessera::mesh::VertexPtrs verts;
  verts.reserve(rem.cell.size());
  for (const std::uint64_t id : rem.cell) {
    const auto it = idToVert.find(id);
    if (it == idToVert.end()) return false;  // a cell vertex vanished
    verts.push_back(it->second);
  }
  // Re-create the top cell; createSimplexTracked rebuilds exactly the missing
  // edges (the orphaned ones removed above), leaving surviving edges untouched.
  st_->createSimplexTracked(verts);
  // Restore the removed edges' weights/phases bit-exactly.
  std::map<std::pair<std::uint64_t, std::uint64_t>, ::tessera::mesh::Edge *>
      edgeByPair;
  for (const auto e : st_->getEdgeList()->toVector()) {
    if (e == nullptr || e->getSource() == nullptr || e->getTarget() == nullptr)
      continue;
    const std::uint64_t a = e->getSource()->getId();
    const std::uint64_t b = e->getTarget()->getId();
    edgeByPair[{std::min(a, b), std::max(a, b)}] = e;
  }
  for (const auto &[u, v, w, theta] : rem.removedEdges) {
    const auto it = edgeByPair.find({std::min(u, v), std::max(u, v)});
    if (it != edgeByPair.end()) {
      it->second->setLength(w);  // the recorded complex LENGTH, bit-exact
      it->second->setPhase(theta);
    }
  }
  return true;
}

bool EigenstateSynthesis::restoreLastRemoval() {
  if (!st_ || removals_.empty()) return false;
  const Removal rem = std::move(removals_.back());
  removals_.pop_back();
  applyRestore(rem);
  laplacian_ = HodgeLaplacian(st_, HodgeLaplacian::defaultWeightConvention(), metricSource_);
  capture();
  classifyBoundary();
  return true;
}

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

// === The discovered operator: ker L₁(W − ∂W) ===

std::vector<std::vector<std::uint64_t>>
EigenstateSynthesis::bulkMinusBoundaryCells() const {
  std::vector<std::vector<std::uint64_t>> out;
  if (!st_) return out;
  const std::unordered_set<std::uint64_t> bverts(
      boundaryVertexIdsSorted_.begin(), boundaryVertexIdsSorted_.end());
  for (auto &e : ChainComplex::fromSpacetime(*st_).kSimplexVertices(1)) {
    bool interior = true;
    for (const std::uint64_t v : e)
      if (bverts.count(v)) { interior = false; break; }
    if (interior) out.push_back(e);
  }
  return out;
}


std::vector<cd>
EigenstateSynthesis::bulkMinusBoundaryHarmonicMatrix(double tol,
                                                     bool metric) const {
  using Eigen::Index;
  using Eigen::MatrixXcd;
  using Eigen::VectorXcd;
  if (!st_)
    return {};

  // W − ∂W is the subcomplex induced on the interior vertices (those on no ∂W
  // face); a cell belongs to it iff all of its vertices are interior. Too few
  // interior vertices means no interior 1-cycle, so nothing is carried: three
  // interior vertices already carry nothing.
  const std::unordered_set<std::uint64_t> bverts(
      boundaryVertexIdsSorted_.begin(), boundaryVertexIdsSorted_.end());
  const auto interiorCell = [&](const std::vector<std::uint64_t> &c) {
    for (const std::uint64_t v : c)
      if (bverts.count(v))
        return false;
    return true;
  };

  const ChainComplex cc = ChainComplex::fromSpacetime(*st_);
  const auto v0 = cc.kSimplexVertices(0);
  const auto v1 = cc.kSimplexVertices(1);
  const auto v2 = cc.kSimplexVertices(2);
  const std::size_t n0 = v0.size(), n1 = v1.size(), n2 = v2.size();
  if (n1 == 0)
    return {};

  // The interior-cell index lists into the full C_k orderings.
  std::vector<Index> i0, i1, i2;
  for (std::size_t i = 0; i < n0; ++i)
    if (interiorCell(v0[i]))
      i0.push_back(static_cast<Index>(i));
  for (std::size_t i = 0; i < n1; ++i)
    if (interiorCell(v1[i]))
      i1.push_back(static_cast<Index>(i));
  for (std::size_t i = 0; i < n2; ++i)
    if (interiorCell(v2[i]))
      i2.push_back(static_cast<Index>(i));
  const Index m1 = static_cast<Index>(i1.size());
  if (m1 == 0)
    return {};

  // Restrict the boundary maps to the interior rows and columns once. The
  // combinatorial and metric modes differ only in the weights wrapped around
  // these same incidence matrices.
  const std::vector<long> &d1 = cc.boundaryMatrix(1);
  const std::vector<long> &d2 = cc.boundaryMatrix(2);
  MatrixXcd D1 = MatrixXcd::Zero(static_cast<Index>(i0.size()), m1);
  for (Index r = 0; r < static_cast<Index>(i0.size()); ++r)
    for (Index c = 0; c < m1; ++c)
      D1(r, c) = static_cast<double>(d1[static_cast<std::size_t>(i0[r]) * n1 +
                                        static_cast<std::size_t>(i1[c])]);
  MatrixXcd D2 = MatrixXcd::Zero(m1, static_cast<Index>(i2.size()));
  for (Index r = 0; r < m1; ++r)
    for (Index c = 0; c < static_cast<Index>(i2.size()); ++c)
      D2(r, c) = static_cast<double>(d2[static_cast<std::size_t>(i1[r]) * n2 +
                                        static_cast<std::size_t>(i2[c])]);

  MatrixXcd L = D1.transpose() * D1 + D2 * D2.transpose();
  if (metric) {
    const HodgeLaplacian hodge(st_, HodgeLaplacian::defaultWeightConvention(), metricSource_);
    const auto fullW0 = hodge.weights(0);
    const auto fullW1 = hodge.weights(1);
    const auto fullW2 = hodge.weights(2);
    if (fullW0.size() != n0 || fullW1.size() != n1 ||
        (!i2.empty() && fullW2.size() != n2))
      return {};

    const auto selectWeights = [](const std::vector<cd> &full,
                                  const std::vector<Index> &indices) {
      VectorXcd selected(static_cast<Index>(indices.size()));
      for (Index i = 0; i < selected.size(); ++i)
        selected[i] = full[static_cast<std::size_t>(
            indices[static_cast<std::size_t>(i)])];
      return selected;
    };
    const VectorXcd w0 = selectWeights(fullW0, i0);
    const VectorXcd w1 = selectWeights(fullW1, i1);
    const VectorXcd w2 = selectWeights(fullW2, i2);
    if (!w0.allFinite() || !w1.allFinite() || !w2.allFinite())
      return {};
    for (Index i = 0; i < w1.size(); ++i)
      if (std::abs(w1[i]) <= std::numeric_limits<double>::min())
        return {};
    for (Index i = 0; i < w2.size(); ++i)
      if (std::abs(w2[i]) <= std::numeric_limits<double>::min())
        return {};

    L = w1.cwiseInverse().asDiagonal() * D1.transpose() * w0.asDiagonal() * D1;
    if (!i2.empty())
      L += D2 * w2.cwiseInverse().asDiagonal() * D2.transpose() *
           w1.asDiagonal();
    if (!L.allFinite())
      return {};

    // The signed metric operator is generally non-normal. Its right nullspace
    // is the span of the right singular vectors with sigma < tol.
    Eigen::BDCSVD<MatrixXcd> svd(L, Eigen::ComputeFullV);
    const Eigen::VectorXd &sigma = svd.singularValues(); // descending
    const MatrixXcd &V = svd.matrixV();
    std::vector<cd> out;
    for (Index offset = 0; offset < sigma.size(); ++offset) {
      const Index j = sigma.size() - 1 - offset; // smallest first
      if (sigma[j] >= tol)
        continue;
      for (Index i = 0; i < m1; ++i)
        out.push_back(V(i, j));
    }
    return out;
  }

  // Combinatorial ker L₁ ≅ H₁ of the interior, in ascending eigenvalue order.
  Eigen::SelfAdjointEigenSolver<MatrixXcd> eig(L);
  const Eigen::VectorXd &lam = eig.eigenvalues();
  const MatrixXcd &V = eig.eigenvectors();
  std::vector<cd> out;
  for (Index j = 0; j < lam.size(); ++j) {
    if (std::abs(lam[j]) >= tol)
      continue;
    for (Index i = 0; i < m1; ++i)
      out.push_back(V(i, j));
  }
  return out;
}

std::vector<cd> EigenstateSynthesis::readoutLaplacian() const {
  // k = 0 scores the U(1) connection operator, k >= 1 the Hodge L_k. Read
  // through the captured operator: its vertex/cell ordering is the one
  // cellOrdering_ was built from, and neither entry point consults a spectral
  // cache (both reassemble from the live edges on every call), so repeated
  // perturb-then-query reflects every perturbation.
  return k_ == 0 ? laplacian_.connectionLaplacian()
                 : laplacian_.laplacian(k_, /*metric=*/true);
}

std::vector<cd> EigenstateSynthesis::readoutHarmonicMatrix() const {
  // k = 0 reads the U(1) connection operator, k >= 1 the Hodge L_k. The
  // degree-zero register's content is the U(1) flux carried around a hole:
  // ker L_0 = b_0 at any weights, so L_0's harmonics can carry no flux and a
  // degree-zero readout taken from them would be identically gauge-flat. The
  // connection operator is indexed over the full sorted vertex order, which is
  // exactly cellOrdering_ at k = 0.
  const HodgeLaplacian hodge(st_, HodgeLaplacian::defaultWeightConvention(), metricSource_);
  return k_ == 0 ? hodge.connectionHarmonicMatrix(1e-9)
                 : hodge.harmonicMatrix(k_, 1e-9, /*metric=*/true);
}

EigenstateSynthesis::RegisterReadout EigenstateSynthesis::assembleRegisterReadout(
    const std::vector<std::vector<std::uint64_t>> &holes) const {
  const auto joinIds = [](const std::vector<std::uint64_t> &c) {
    std::string out = "(";
    for (std::size_t i = 0; i < c.size(); ++i) {
      if (i) out += ",";
      out += std::to_string(c[i]);
    }
    return out + ")";
  };

  RegisterReadout out;
  const std::size_t n = order_;
  // Harmonics fresh from the live complex: surgery between calls moves them,
  // and the operator's own spectral cache is keyed to construction time.
  out.H = readoutHarmonicMatrix();
  if (n == 0) {
    if (!holes.empty())
      throw std::runtime_error(
          "EigenstateSynthesis::assembleRegisterReadout: the complex has no "
          "k-cells to carry periods");
    return out;
  }
  if (out.H.size() % n != 0)
    throw std::runtime_error(
        "EigenstateSynthesis::assembleRegisterReadout: the harmonic matrix "
        "width " + std::to_string(out.H.size()) +
        " is not a multiple of the captured operator dimension " +
        std::to_string(n) + " (the complex changed behind the synthesizer)");
  out.dim = out.H.size() / n;

  std::map<std::vector<std::uint64_t>, std::size_t> col;
  for (std::size_t i = 0; i < cellOrdering_.size(); ++i) col[cellOrdering_[i]] = i;

  const std::size_t m = holes.size();
  const std::size_t hv = static_cast<std::size_t>(k_) + 2;
  out.P.assign(out.dim * m, cd(0.0, 0.0));
  out.leakColumns.reserve(m);
  for (std::size_t q = 0; q < m; ++q) {
    std::vector<std::uint64_t> h = holes[q];
    std::sort(h.begin(), h.end());
    if (h.size() != hv)
      throw std::runtime_error(
          "EigenstateSynthesis::assembleRegisterReadout: hole " + joinIds(h) +
          " has " + std::to_string(h.size()) + " vertices; a degree-" +
          std::to_string(k_) + " period needs a removed (k+1)-cell of " +
          std::to_string(hv));
    // Facets are visited in the degree's walk order: the (a,b),(b,c),(a,c) edge
    // walk of a circle at k = 1, so the period accumulates bit for bit in the
    // register layers' summation order; the canonical drop-v_j order otherwise.
    // The hole's leak facet is the first of the walk — the (a,b) edge, or the
    // drop-v_0 facet — which carries boundary sign +1 in both conventions, so
    // adding the leak there moves that hole's period by exactly the leak.
    std::vector<std::size_t> walk(hv);
    for (std::size_t i = 0; i < hv; ++i) walk[i] = i;
    if (k_ == 1) std::rotate(walk.begin(), walk.end() - 1, walk.end());
    for (std::size_t w = 0; w < hv; ++w) {
      const std::size_t j = walk[w];
      std::vector<std::uint64_t> f;
      f.reserve(hv - 1);
      for (std::size_t i = 0; i < hv; ++i)
        if (i != j) f.push_back(h[i]);
      const auto it = col.find(f);
      if (it == col.end())
        throw std::runtime_error(
            "EigenstateSynthesis::assembleRegisterReadout: facet " +
            joinIds(f) + " of hole " + joinIds(h) +
            " is not a k-cell of the complex (not a boundary cycle here)");
      if (w == 0) out.leakColumns.push_back(it->second);
      // The induced-orientation boundary sign: facet j of the sorted hole
      // drops vertex v_j and carries (-1)^j.
      const double s = (j % 2 == 0) ? 1.0 : -1.0;
      for (std::size_t r = 0; r < out.dim; ++r)
        out.P[r * m + q] += s * out.H[r * n + it->second];
    }
  }
  return out;
}

std::vector<cd> EigenstateSynthesis::cyclePeriods(
    const std::vector<std::vector<std::uint64_t>> &holes) const {
  return assembleRegisterReadout(holes).P;
}

std::vector<cd> EigenstateSynthesis::carriedRepresentative(
    const std::vector<std::vector<std::uint64_t>> &holes,
    const std::vector<cd> &targetPeriods) const {
  if (targetPeriods.size() != holes.size())
    throw std::runtime_error(
        "EigenstateSynthesis::carriedRepresentative: " +
        std::to_string(targetPeriods.size()) + " target periods for " +
        std::to_string(holes.size()) + " holes");
  return carriedFromReadout(assembleRegisterReadout(holes), targetPeriods);
}

// === Charge sector: the E/B split of the field strength F ∈ Ω² ===

std::vector<cd> EigenstateSynthesis::curvatureFromConnection(
    const std::vector<cd> &A) const {
  if (k_ != 2)
    throw std::runtime_error(
        "EigenstateSynthesis::curvatureFromConnection: requires a degree-2 "
        "instance (the field strength F is a 2-cochain); this instance is "
        "degree " +
        std::to_string(k_));
  // The connection A is a degree-1 cochain in the canonical ChainComplex 1-cell
  // order; map each sorted edge (u,v) to its index so the per-plaquette signed
  // edge sum can read A by edge.
  const auto edges1 = ChainComplex::fromSpacetime(*st_).kSimplexVertices(1);
  if (A.size() != edges1.size())
    throw std::runtime_error(
        "EigenstateSynthesis::curvatureFromConnection: the connection has " +
        std::to_string(A.size()) + " components; the complex has " +
        std::to_string(edges1.size()) + " 1-cells (edges)");
  std::map<std::vector<std::uint64_t>, std::size_t> edgeIdx;
  for (std::size_t i = 0; i < edges1.size(); ++i) edgeIdx[edges1[i]] = i;

  // F = dA on each sorted 2-cell (a,b,c): the induced-orientation signed edge sum
  // +A(a,b) + A(b,c) - A(a,c) — facet (drop v_j) carries (-1)^j, the same boundary
  // convention cyclePeriods uses (drop a -> +(b,c), drop b -> -(a,c), drop c ->
  // +(a,b)).
  std::vector<cd> F(order_, cd(0.0, 0.0));
  for (std::size_t i = 0; i < cellOrdering_.size(); ++i) {
    const auto &cell = cellOrdering_[i];  // sorted (a,b,c)
    if (cell.size() != 3)
      throw std::runtime_error(
          "EigenstateSynthesis::curvatureFromConnection: a degree-2 cell has " +
          std::to_string(cell.size()) + " vertices (expected 3)");
    const std::vector<std::vector<std::uint64_t>> facets = {
        {cell[0], cell[1]}, {cell[1], cell[2]}, {cell[0], cell[2]}};
    const double sign[3] = {1.0, 1.0, -1.0};  // +(a,b) +(b,c) -(a,c)
    for (std::size_t j = 0; j < 3; ++j) {
      const auto it = edgeIdx.find(facets[j]);
      if (it == edgeIdx.end())
        throw std::runtime_error(
            "EigenstateSynthesis::curvatureFromConnection: edge (" +
            std::to_string(facets[j][0]) + "," + std::to_string(facets[j][1]) +
            ") of a 2-cell is not a 1-cell of the complex");
      F[i] += sign[j] * A[it->second];
    }
  }
  return F;
}

EigenstateSynthesis::FieldStrengthSplit EigenstateSynthesis::fieldStrengthSplit(
    const std::vector<cd> &F) const {
  if (k_ != 2)
    throw std::runtime_error(
        "EigenstateSynthesis::fieldStrengthSplit: requires a degree-2 instance "
        "(the field strength F is a 2-cochain); this instance is degree " +
        std::to_string(k_));
  if (F.size() != order_)
    throw std::runtime_error(
        "EigenstateSynthesis::fieldStrengthSplit: F has " +
        std::to_string(F.size()) + " components; the degree-2 operator has " +
        std::to_string(order_) + " cells");

  // Map each sorted edge (u,v) to its live Edge* so each plaquette's causal
  // type is read off Edge::isTimelike(), the sanctioned causal test:
  // arg(l^2) ~ +/-pi.
  //
  // The split is binary: a cell is electric if any edge is timelike and magnetic
  // otherwise, so it cannot express a cell whose edges are mixed (off every
  // definite argument). Such a cell lands in `magnetic` rather than being
  // reported as indefinite. Widening the split is out of scope here; the gap is
  // recorded so it is not mistaken for a measurement.
  std::map<std::pair<std::uint64_t, std::uint64_t>, ::tessera::mesh::Edge *> em;
  for (auto *e : edges_) {
    const std::uint64_t a = e->getSource()->getId();
    const std::uint64_t b = e->getTarget()->getId();
    em[{std::min(a, b), std::max(a, b)}] = e;
  }

  FieldStrengthSplit split;
  split.electric.assign(order_, cd(0.0, 0.0));
  split.magnetic.assign(order_, cd(0.0, 0.0));
  for (std::size_t i = 0; i < cellOrdering_.size(); ++i) {
    const auto &cell = cellOrdering_[i];  // sorted (a,b,c)
    const std::pair<std::uint64_t, std::uint64_t> facets[3] = {
        {cell[0], cell[1]}, {cell[1], cell[2]}, {cell[0], cell[2]}};
    // Electric iff any leg is timelike (one temporal index, the discrete F_{0i});
    // else purely-spacelike = magnetic (F_{ij}).
    bool electric = false;
    for (const auto &f : facets) {
      const auto it = em.find(f);
      if (it == em.end())
        throw std::runtime_error(
            "EigenstateSynthesis::fieldStrengthSplit: edge (" +
            std::to_string(f.first) + "," + std::to_string(f.second) +
            ") of a 2-cell is not an edge of the complex");
      if (it->second->isTimelike()) {
        electric = true;
        break;
      }
    }
    if (electric) {
      split.electric[i] = F[i];
      split.electricCells.push_back(i);
    } else {
      split.magnetic[i] = F[i];
      split.magneticCells.push_back(i);
    }
  }
  return split;
}

cd EigenstateSynthesis::gaussLawCharge(
    const std::vector<cd> &F, const std::vector<std::uint64_t> &enclosedVertices,
    bool electricOnly) const {
  if (k_ != 2)
    throw std::runtime_error(
        "EigenstateSynthesis::gaussLawCharge: requires a degree-2 instance (the "
        "field strength F is a 2-cochain); this instance is degree " +
        std::to_string(k_));
  if (F.size() != order_)
    throw std::runtime_error(
        "EigenstateSynthesis::gaussLawCharge: F has " +
        std::to_string(F.size()) + " components; the degree-2 operator has " +
        std::to_string(order_) + " cells");

  // The electric (timelike-leg) plaquettes, the same E/B causal split, so Q
  // lives on the same temporal sector. The magnetic-only Q — the full flux on an
  // all-spacelike complex — is recovered with electricOnly = false.
  const FieldStrengthSplit split = fieldStrengthSplit(F);
  const std::set<std::size_t> electric(split.electricCells.begin(),
                                       split.electricCells.end());

  // Sorted 2-cell tuple -> its component index, so a boundary face of an
  // enclosed tetrahedron maps back to its F entry.
  std::map<std::vector<std::uint64_t>, std::size_t> faceIdx;
  for (std::size_t i = 0; i < cellOrdering_.size(); ++i)
    faceIdx[cellOrdering_[i]] = i;

  const std::set<std::uint64_t> enclosed(enclosedVertices.begin(),
                                         enclosedVertices.end());

  // V = the closed star of the enclosed vertices (every tetrahedron touching
  // one). S = boundary 2-chain dV: accumulate each enclosed tetrahedron's four
  // (-1)^j-signed faces; faces interior to V (shared by two V-cells with
  // opposite induced orientation) cancel, leaving the enclosing surface.
  const auto tets = ChainComplex::fromSpacetime(*st_).kSimplexVertices(3);
  std::map<std::vector<std::uint64_t>, double> boundary;
  for (const auto &tet : tets) {
    if (tet.size() != 4) continue;
    bool touches = false;
    for (const std::uint64_t v : tet)
      if (enclosed.count(v)) {
        touches = true;
        break;
      }
    if (!touches) continue;
    for (int j = 0; j < 4; ++j) {  // drop vertex j carries (-1)^j; tet sorted
      std::vector<std::uint64_t> face;
      face.reserve(3);
      for (int q = 0; q < 4; ++q)
        if (q != j) face.push_back(tet[static_cast<std::size_t>(q)]);
      boundary[face] += (j % 2 == 0) ? 1.0 : -1.0;
    }
  }

  // Q = sum over the surviving surface plaquettes of the orientation-signed F,
  // restricted to the electric (temporal-sector) plaquettes when asked.
  cd Q(0.0, 0.0);
  for (const auto &[face, coeff] : boundary) {
    if (std::abs(coeff) < 1e-12) continue;  // interior face, cancelled
    const auto it = faceIdx.find(face);
    if (it == faceIdx.end()) continue;
    if (electricOnly && electric.find(it->second) == electric.end()) continue;
    Q += coeff * F[it->second];
  }
  return Q;
}

std::vector<cd> EigenstateSynthesis::carriedFromReadout(
    const RegisterReadout &ro, const std::vector<cd> &targetPeriods) const {
  const std::size_t n = order_;
  const std::size_t m = ro.leakColumns.size();
  if (n == 0) return {};
  if (targetPeriods.size() != m)
    throw std::runtime_error(
        "EigenstateSynthesis::carriedFromReadout: " +
        std::to_string(targetPeriods.size()) + " target periods for " +
        std::to_string(m) + " cycles");

  // The minimum-norm least-squares projection onto the carried period rows.
  const std::vector<cd> c = lstsqOverReadout(ro, targetPeriods);

  // The carried representative plus the minimal leak: psi = sum_r c_r h_r,
  // then each cycle's uncarried remainder lands on its leak column, so the
  // cochain's periods are exactly the targets.
  std::vector<cd> psi(n, cd(0.0, 0.0));
  for (std::size_t r = 0; r < ro.dim; ++r) {
    const cd cr = c[r];
    for (std::size_t i = 0; i < n; ++i) psi[i] += cr * ro.H[r * n + i];
  }
  for (std::size_t q = 0; q < m; ++q) {
    cd carried(0.0, 0.0);
    for (std::size_t r = 0; r < ro.dim; ++r) carried += c[r] * ro.P[r * m + q];
    psi[ro.leakColumns[q]] += targetPeriods[q] - carried;
  }
  return psi;
}

std::vector<cd> EigenstateSynthesis::lstsqOverReadout(
    const RegisterReadout &ro, const std::vector<cd> &targetPeriods) const {
  // c = (P^T)^+ target (minimum-norm least squares, matching numpy.linalg.lstsq):
  // the SVD projection of the targets onto the carried period rows. Shared by
  // carriedFromReadout (r_U's leaked state) and periodGapForLoops (r_psi's
  // period gap), so the two terms fit onto the same carried space.
  const std::size_t m = ro.leakColumns.size();
  if (ro.dim == 0) return {};
  Eigen::VectorXcd t(static_cast<Eigen::Index>(m));
  for (std::size_t q = 0; q < m; ++q)
    t[static_cast<Eigen::Index>(q)] = targetPeriods[q];
  Eigen::MatrixXcd Pt(static_cast<Eigen::Index>(m),
                      static_cast<Eigen::Index>(ro.dim));
  for (std::size_t q = 0; q < m; ++q)
    for (std::size_t r = 0; r < ro.dim; ++r)
      Pt(static_cast<Eigen::Index>(q), static_cast<Eigen::Index>(r)) =
          ro.P[r * m + q];
  const Eigen::VectorXcd c =
      Pt.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(t);
  return std::vector<cd>(c.data(), c.data() + c.size());
}

EigenstateSynthesis::RegisterReadout EigenstateSynthesis::assembleReadoutOverLoops(
    const std::vector<EdgeLoop> &loops) const {
  RegisterReadout out;
  const std::size_t n = order_;
  out.H = readoutHarmonicMatrix();
  if (n == 0) {
    if (!loops.empty())
      throw std::runtime_error(
          "EigenstateSynthesis::assembleReadoutOverLoops: the complex has no "
          "k-cells to carry periods");
    return out;
  }
  if (out.H.size() % n != 0)
    throw std::runtime_error(
        "EigenstateSynthesis::assembleReadoutOverLoops: the harmonic matrix "
        "width " + std::to_string(out.H.size()) +
        " is not a multiple of the captured operator dimension " +
        std::to_string(n));
  out.dim = out.H.size() / n;
  std::map<std::vector<std::uint64_t>, std::size_t> col;
  for (std::size_t i = 0; i < cellOrdering_.size(); ++i) col[cellOrdering_[i]] = i;
  const std::size_t m = loops.size();
  out.P.assign(out.dim * m, cd(0.0, 0.0));
  out.leakColumns.reserve(m);
  for (std::size_t q = 0; q < m; ++q) {
    if (loops[q].empty())
      throw std::runtime_error(
          "EigenstateSynthesis::assembleReadoutOverLoops: loop " +
          std::to_string(q) + " is empty");
    bool first = true;
    Edge::walkLoop(loops[q], [&](std::uint64_t a, std::uint64_t b, double s) {
      const std::vector<std::uint64_t> e = {std::min(a, b), std::max(a, b)};
      const auto it = col.find(e);
      if (it == col.end())
        throw std::runtime_error(
            "EigenstateSynthesis::assembleReadoutOverLoops: loop edge (" +
            std::to_string(a) + "," + std::to_string(b) +
            ") is not a k-cell of the complex");
      if (first) {
        out.leakColumns.push_back(it->second);
        first = false;
      }
      for (std::size_t r = 0; r < out.dim; ++r)
        out.P[r * m + q] += s * out.H[r * n + it->second];
    });
  }
  return out;
}

std::vector<cd> EigenstateSynthesis::cyclePeriodsOverLoops(
    const std::vector<EdgeLoop> &loops) const {
  return assembleReadoutOverLoops(loops).P;
}

double EigenstateSynthesis::residualForLoops(
    const std::vector<EdgeLoop> &loops,
    const std::vector<cd> &targetPeriods) const {
  const std::vector<cd> psi =
      carriedFromReadout(assembleReadoutOverLoops(loops), targetPeriods);
  if (psi.empty()) return 0.0;
  return residual(psi);
}

std::vector<cd> EigenstateSynthesis::carriedRepresentativeOverLoops(
    const std::vector<EdgeLoop> &loops, const std::vector<cd> &targetPeriods) const {
  return carriedFromReadout(assembleReadoutOverLoops(loops), targetPeriods);
}

std::vector<cd> EigenstateSynthesis::periodsOfCochainOverLoops(
    const std::vector<cd> &cochain, const std::vector<EdgeLoop> &loops) const {
  std::map<std::vector<std::uint64_t>, std::size_t> col;
  for (std::size_t i = 0; i < cellOrdering_.size(); ++i) col[cellOrdering_[i]] = i;
  std::vector<cd> out(loops.size(), cd(0.0, 0.0));
  for (std::size_t q = 0; q < loops.size(); ++q)
    Edge::walkLoop(loops[q], [&](std::uint64_t a, std::uint64_t b, double s) {
      const std::vector<std::uint64_t> e = {std::min(a, b), std::max(a, b)};
      const auto it = col.find(e);
      if (it == col.end())
        throw std::runtime_error(
            "EigenstateSynthesis::periodsOfCochainOverLoops: loop edge (" +
            std::to_string(a) + "," + std::to_string(b) +
            ") is not a k-cell of the complex");
      if (it->second < cochain.size())
        out[q] += cd(s, 0.0) * cochain[it->second];
    });
  return out;
}

double EigenstateSynthesis::residualForPeriods(
    const std::vector<std::vector<std::uint64_t>> &holes,
    const std::vector<cd> &targetPeriods) const {
  const std::vector<cd> psi = carriedRepresentative(holes, targetPeriods);
  if (psi.empty()) return 0.0;  // no k-cells to carry periods
  return residual(psi);
}

std::vector<EigenstateSynthesis::EdgeLoop> EigenstateSynthesis::holeLoops(
    const std::vector<std::vector<std::uint64_t>> &holes, const char *who) const {
  auto &vlist = *st_->getVertexList();
  auto edge = [&](std::uint64_t u, std::uint64_t v) {
    auto *vu = vlist.get(u), *vv = vlist.get(v);
    if (!vu || !vv)
      throw std::runtime_error(std::string(who) + ": hole vertex absent");
    return Edge(vu, vv, cd(1.0, 0.0));
  };
  std::vector<EdgeLoop> loops;
  loops.reserve(holes.size());
  for (const auto &h : holes) {
    if (h.size() != 3)
      throw std::runtime_error(std::string(who) + ": hole has " +
                               std::to_string(h.size()) +
                               " vertices, expected 3");
    std::vector<std::uint64_t> s(h);
    std::sort(s.begin(), s.end());
    loops.push_back({edge(s[0], s[1]), edge(s[1], s[2]), edge(s[2], s[0])});
  }
  return loops;
}

double EigenstateSynthesis::periodGapForLoops(
    const std::vector<EdgeLoop> &loops,
    const std::vector<cd> &targetPeriods) const {
  const std::size_t m = loops.size();
  if (targetPeriods.size() != m)
    throw std::runtime_error(
        "EigenstateSynthesis::periodGapForLoops: " +
        std::to_string(targetPeriods.size()) + " target periods for " +
        std::to_string(m) + " loops");
  if (m == 0) return 0.0;
  const RegisterReadout ro = assembleReadoutOverLoops(loops);
  // The carried object stays a pure harmonic (no leak): least-squares-fit the
  // target onto the carried period rows (lstsqOverReadout, the same projection
  // r_U's carriedFromReadout uses, so the two terms share the realizable zero
  // set) and return the squared norm of the remainder no harmonic can reach,
  // ||carried - target||^2 = ||P^T c - target||^2. When ro.dim == 0 nothing is
  // carried and the gap is the whole target (c is empty -> carried = 0).
  const std::vector<cd> c = lstsqOverReadout(ro, targetPeriods);
  double gap = 0.0;
  for (std::size_t q = 0; q < m; ++q) {
    cd carried(0.0, 0.0);
    for (std::size_t r = 0; r < ro.dim; ++r) carried += c[r] * ro.P[r * m + q];
    gap += std::norm(carried - targetPeriods[q]);
  }
  return gap;
}

double EigenstateSynthesis::periodGapForPeriods(
    const std::vector<std::vector<std::uint64_t>> &holes,
    const std::vector<cd> &targetPeriods) const {
  return periodGapForLoops(
      holeLoops(holes, "EigenstateSynthesis::periodGapForPeriods"),
      targetPeriods);
}

std::vector<double> EigenstateSynthesis::periodGradientOverLoops(
    const std::vector<EdgeLoop> &loops,
    const std::vector<cd> &targetPeriods) const {
  // The signed edge-loop machinery is degree-1 by construction: a loop period
  // reads an edge (1-cell) cochain, and everything below (M = L_1, the triangle
  // low-rank dM, the cell->index map over 2-vertex tuples) is the k = 1 layout.
  // Other degrees go through the hole entry points, which route by degree.
  if (k_ != 1)
    throw std::runtime_error(
        "EigenstateSynthesis::periodGradientOverLoops: the edge-loop core is "
        "degree-1 machinery (loops are closed walks of 1-cells); this "
        "synthesis is degree " + std::to_string(k_) +
        " — use residualForPeriodsGradient, which routes holes by degree.");
  using Eigen::Index;
  using Eigen::MatrixXcd;
  using Eigen::VectorXcd;
  const std::size_t n1 = order_;
  std::vector<double> grad(n1, 0.0);
  const std::size_t m = loops.size();
  if (n1 == 0 || m == 0) return grad;
  if (targetPeriods.size() != m)
    throw std::runtime_error(
        "EigenstateSynthesis::periodGradientOverLoops: " +
        std::to_string(targetPeriods.size()) + " target periods for " +
        std::to_string(m) + " loops");
  static constexpr double kNullTol = 1e-7;
  const Index N = static_cast<Index>(n1);

  // ---- chain complex, weights, and the metric Laplacian M = L1 ----
  const ChainComplex cc = ChainComplex::fromSpacetime(*st_);
  const std::vector<std::vector<std::uint64_t>> &cells1 = cellSimplices();
  const auto tris = cc.kSimplexVertices(2);
  const std::size_t n2 = tris.size();
  const std::vector<long> &d1flat = cc.boundaryMatrix(1);  // n0 x n1
  const std::vector<long> &d2flat = cc.boundaryMatrix(2);  // n1 x n2
  const std::size_t n0 = d1flat.size() / n1;
  const HodgeLaplacian hl(st_, HodgeLaplacian::defaultWeightConvention(), metricSource_);
  const std::vector<cd> W1v = hl.weights(1);  // n1, signed complex
  const std::vector<cd> W2v = hl.weights(2);  // n2, signed complex
  const std::vector<cd> Lflat = hl.laplacian(1, /*metric=*/true);

  MatrixXcd M(N, N);
  for (std::size_t i = 0; i < n1; ++i)
    for (std::size_t j = 0; j < n1; ++j)
      M(static_cast<Index>(i), static_cast<Index>(j)) = Lflat[i * n1 + j];
  VectorXcd W1(N);
  for (std::size_t i = 0; i < n1; ++i) W1[static_cast<Index>(i)] = W1v[i];
  MatrixXcd d1m(static_cast<Index>(n0), N);
  for (std::size_t v = 0; v < n0; ++v)
    for (std::size_t c = 0; c < n1; ++c)
      d1m(static_cast<Index>(v), static_cast<Index>(c)) =
          static_cast<double>(d1flat[v * n1 + c]);
  MatrixXcd d2m(N, static_cast<Index>(n2));
  for (std::size_t c = 0; c < n1; ++c)
    for (std::size_t t = 0; t < n2; ++t)
      d2m(static_cast<Index>(c), static_cast<Index>(t)) =
          static_cast<double>(d2flat[c * n2 + t]);
  const MatrixXcd K1 = d1m.transpose() * d1m;  // n1 x n1
  VectorXcd W2inv(static_cast<Index>(n2));
  for (std::size_t t = 0; t < n2; ++t) W2inv[static_cast<Index>(t)] = 1.0 / W2v[t];
  const MatrixXcd K2 = d2m * W2inv.asDiagonal() * d2m.transpose();  // n1 x n1

  // ---- index maps: cell -> index, edge -> l^2, edge -> incident triangles ----
  auto key = [](std::uint64_t a, std::uint64_t b) {
    return std::pair<std::uint64_t, std::uint64_t>(std::min(a, b), std::max(a, b));
  };
  std::map<std::pair<std::uint64_t, std::uint64_t>, std::size_t> cidx1;
  for (std::size_t i = 0; i < n1; ++i) cidx1[key(cells1[i][0], cells1[i][1])] = i;
  std::map<std::pair<std::uint64_t, std::uint64_t>, std::complex<double>> l2map;
  for (auto *e : edges_)
    l2map[key(e->getSource()->getId(), e->getTarget()->getId())] =
        (e->getLength() * e->getLength());
  std::map<std::pair<std::uint64_t, std::uint64_t>, std::vector<std::size_t>> trisOf;
  for (std::size_t ti = 0; ti < n2; ++ti)
    for (int i = 0; i < 3; ++i)
      for (int j = i + 1; j < 3; ++j)
        trisOf[key(tris[ti][i], tris[ti][j])].push_back(ti);
  auto L2 = [&](std::uint64_t a, std::uint64_t b) -> cd {
    if (a == b) return cd(0.0, 0.0);
    auto it = l2map.find(key(a, b));
    return it == l2map.end() ? cd(0.0, 0.0) : it->second;
  };

  // ---- Q (signed edge-loop covector) + each cycle's leak column ----
  // Any closed walk of oriented edges, not just a removed triangle's boundary:
  // Q(q, edge) += +1 along the stored orientation, -1 against; the leak is the
  // loop's first edge.
  MatrixXcd Q = MatrixXcd::Zero(static_cast<Index>(m), N);
  std::vector<std::size_t> leakCol(m);
  for (std::size_t q = 0; q < m; ++q) {
    const EdgeLoop &loop = loops[q];
    if (loop.empty())
      throw std::runtime_error(
          "EigenstateSynthesis::periodGradientOverLoops: loop " +
          std::to_string(q) + " is empty");
    const Edge &fe = loop.front();
    leakCol[q] = cidx1.at(key(fe.getSource()->getId(), fe.getTarget()->getId()));
    Edge::walkLoop(loop, [&](std::uint64_t a, std::uint64_t b, double s) {
      Q(static_cast<Index>(q),
        static_cast<Index>(cidx1.at(key(a, b)))) += s;
    });
  }

  // ---- eigendecomposition of M; harmonic (null) / non-null split ----
  // The signed operator is generally non-self-adjoint (real but non-symmetric
  // on the real-l^2 manifold), so a general eigensolver is required: a
  // self-adjoint one reads a single triangle and silently symmetrizes, which
  // makes this gradient come out identically zero.
  Eigen::ComplexEigenSolver<MatrixXcd> eig(M);
  const VectorXcd lam = eig.eigenvalues();
  const MatrixXcd U = eig.eigenvectors();
  std::vector<Index> nullIdx, nnIdx;
  for (Index i = 0; i < N; ++i) (std::abs(lam[i]) < kNullTol ? nullIdx : nnIdx).push_back(i);
  const Index nd = static_cast<Index>(nullIdx.size());
  const Index nnd = static_cast<Index>(nnIdx.size());
  if (nd == 0) return grad;  // no harmonics -> nothing carried
  MatrixXcd Un(N, nd), Unn(N, nnd);
  for (Index r = 0; r < nd; ++r) Un.col(r) = U.col(nullIdx[r]);
  for (Index r = 0; r < nnd; ++r) Unn.col(r) = U.col(nnIdx[r]);
  // Left (dual) basis for the non-self-adjoint perturbation: rows of U^-1 are
  // the covectors v_m with v_m . u_l = delta_ml, which first-order eigenvector
  // perturbation of a non-symmetric M needs in place of U^T.
  const MatrixXcd Uinv = U.inverse();
  MatrixXcd Vnn(nnd, N);
  for (Index r = 0; r < nnd; ++r) Vnn.row(r) = Uinv.row(nnIdx[r]);
  VectorXcd invlam(nnd);  // 1 / (0 - lambda_nn) for the eigenvector perturbation
  for (Index r = 0; r < nnd; ++r) invlam[r] = -1.0 / lam[nnIdx[r]];

  // ---- carried representative psi (via Un / Q / pseudo-inverse), p, rho, r_U ----
  VectorXcd target(static_cast<Index>(m));
  for (std::size_t q = 0; q < m; ++q) target[static_cast<Index>(q)] = targetPeriods[q];
  const MatrixXcd A = Q * Un;                            // m x nd
  const MatrixXcd AtAi = (A.transpose() * A).inverse();  // nd x nd
  const VectorXcd c = (AtAi * A.transpose()).cast<cd>() * target;  // nd
  VectorXcd psi = Un.cast<cd>() * c;                    // n1
  const VectorXcd carried = Q.cast<cd>() * psi;         // m
  for (std::size_t q = 0; q < m; ++q)
    psi[static_cast<Index>(leakCol[q])] +=
        target[static_cast<Index>(q)] - carried[static_cast<Index>(q)];
  const double nrm = psi.norm();
  if (nrm <= 0.0) return grad;
  const VectorXcd p = psi / nrm;
  const VectorXcd Mp = M.cast<cd>() * p;
  const double lamR = (p.dot(Mp)).real();
  const VectorXcd rho = Mp - lamR * p;
  const double rU = rho.squaredNorm();

  // ---- per-edge analytic gradient d r_U / d l^2 (low-rank dM + perturbation) ----
  for (std::size_t je = 0; je < n1; ++je) {
    const Index j = static_cast<Index>(je);
    const auto ek = key(cells1[je][0], cells1[je][1]);
    // dM for the signed operator M = W1^-1 K1 + K2 W1 (K2 = d2 W2^-1 d2^T),
    // under the V^2 weights (the HodgeLaplacian default): W1_j = l^2_j exactly,
    // so dW1_j/dl^2_j = 1 and, every piece rank one,
    //   dM = -(1/W1_j^2) e_j (K1 row j)                       [d(W1^-1) K1]
    //      + (K2 col j) e_j^T                                 [K2 W1 -> K2 dW1]
    //      + per triangle t on e:
    //        -(dW2_t/W2_t^2) (d2 col t)((W1 o d2 col t))^T    [d(W2^-1) term]
    // dM is generally non-symmetric, like M itself.
    VectorXcd dMp;
    MatrixXcd core;
    if (metricSource_ == HodgeLaplacian::MetricSource::WhitneyPencil) {
      // The pencil's exact dense dL_1/dl^2_e (HodgeLaplacian::laplacianGradient
      // under WhitneyPencil); the low-rank diagonal-weight form below does not
      // apply to a non-diagonal metric.
      const std::vector<cd> flat = laplacian_.laplacianGradient(1, cells1[je][0], cells1[je][1]);
      MatrixXcd dM = MatrixXcd::Zero(N, N);
      for (Index a = 0; a < N; ++a)
        for (Index b = 0; b < N; ++b)
          dM(a, b) = flat[static_cast<std::size_t>(a) * static_cast<std::size_t>(N) + static_cast<std::size_t>(b)];
      dMp = dM * p;
      core = Vnn * (dM * Un);
    } else {
      std::vector<VectorXcd> colsA, colsB;
      VectorXcd ev = VectorXcd::Zero(N);
      ev[j] = 1.0;
      colsA.push_back(ev);
      colsB.push_back((-1.0 / (W1[j] * W1[j])) * K1.row(j).transpose());
      colsA.push_back(K2.col(j));
      colsB.push_back(ev);
      for (std::size_t ti : trisOf[ek]) {
        const auto &t = tris[ti];
        Eigen::Matrix2cd G;
        for (int i = 0; i < 2; ++i)
          for (int jj = 0; jj < 2; ++jj)
            G(i, jj) = 0.5 * (L2(t[0], t[i + 1]) + L2(t[0], t[jj + 1]) - L2(t[i + 1], t[jj + 1]));
        const cd detG = G.determinant();
        const cd W2ti = W2v[ti];
        // W2 must be the V^2 weight detG/4 this derivation assumes.
        if (std::abs(detG / 4.0 - W2ti) > 1e-9 * std::max(1.0, std::abs(W2ti)) ||
            std::abs(detG) < 1e-12)
          continue;
        auto ind = [&](int pp, int qq) -> double {
          return (pp != qq && key(t[pp], t[qq]) == ek) ? 1.0 : 0.0;
        };
        Eigen::Matrix2cd dG;
        for (int i = 0; i < 2; ++i)
          for (int jj = 0; jj < 2; ++jj)
            dG(i, jj) = 0.5 * (ind(0, i + 1) + ind(0, jj + 1) - ind(i + 1, jj + 1));
        // W2 = detG/4 => dW2 = W2 * tr(G^-1 dG), by Jacobi's formula.
        const cd dW2ti = W2ti * (G.inverse() * dG).trace();
        const VectorXcd dcol = d2m.col(static_cast<Index>(ti));
        colsA.push_back(dcol);
        colsB.push_back((-dW2ti / (W2ti * W2ti)) * W1.cwiseProduct(dcol));
      }
      const Index r = static_cast<Index>(colsA.size());
      MatrixXcd fa(N, r), fb(N, r);
      for (Index k = 0; k < r; ++k) {
        fa.col(k) = colsA[static_cast<std::size_t>(k)];
        fb.col(k) = colsB[static_cast<std::size_t>(k)];
      }
      // dM p, the eigenvector perturbation dUn, the pseudo-inverse perturbation, dpsi.
      dMp = fa.cast<cd>() * (fb.transpose().cast<cd>() * p);
      core = (Vnn * fa) * (fb.transpose() * Un);  // nnd x nd
    }
    const MatrixXcd dUn = Unn * (invlam.asDiagonal() * core);               // n1 x nd
    const MatrixXcd dA = Q * dUn;                                           // m x nd
    const MatrixXcd dAplus =
        -AtAi * (dA.transpose() * A + A.transpose() * dA) * AtAi * A.transpose() +
        AtAi * dA.transpose();                                             // nd x m
    const VectorXcd dc = dAplus.cast<cd>() * target;                       // nd
    VectorXcd dpsi = dUn.cast<cd>() * c + Un.cast<cd>() * dc;              // n1
    const VectorXcd dcarried = Q.cast<cd>() * dpsi;                        // m
    for (std::size_t q = 0; q < m; ++q)
      dpsi[static_cast<Index>(leakCol[q])] += -dcarried[static_cast<Index>(q)];
    const VectorXcd Mdpsi = M.cast<cd>() * dpsi;
    grad[je] = 2.0 * (rho.dot(dMp)).real() +
               (2.0 / nrm) * (rho.dot(Mdpsi - lamR * dpsi)).real() -
               (2.0 * rU / nrm) * (p.dot(dpsi)).real();
  }
  return grad;
}

std::vector<double> EigenstateSynthesis::periodGradientGeneral(
    const std::vector<std::vector<std::uint64_t>> &holes,
    const std::vector<cd> &targetPeriods) const {
  // k = 0 reads a different operator, not a different weight: the U(1)
  // connection L^U(1) = D - A is complex Hermitian (full l^2 and U(1) phases),
  // so it gets its own complex core rather than the laplacian(k).real()
  // projection below.
  if (k_ == 0) return periodGradientDegreeZero(holes, targetPeriods);
  // Arbitrary-degree exact d r_U / d l^2 over the removed-(k+1)-cell holes.
  // M = L_k, with the per-edge dL_k/dl^2 (HodgeLaplacian::laplacianGradient, on
  // Simplex::volumeGradient) through first-order eigenvector perturbation, and
  // the period covector plus leak from each hole's facet boundary (the
  // assembleRegisterReadout convention). Equals the k = 1 loop core
  // (periodGradientOverLoops) on triangle holes; certified by the Euler identity
  // Sum_e l^2_e d r_U/d l^2_e = -r_U.
  using Eigen::Index;
  using Eigen::MatrixXcd;
  using Eigen::VectorXcd;
  const std::size_t nk = order_;                 // # k-cells (rows/cols of L_k)
  const ChainComplex cc = ChainComplex::fromSpacetime(*st_);
  const std::vector<std::vector<std::uint64_t>> edges1 = cc.kSimplexVertices(1);
  std::vector<double> grad(edges1.size(), 0.0);  // d r_U / d l^2_e, 1-cell order
  const std::size_t m = holes.size();
  if (nk == 0 || m == 0) return grad;
  if (targetPeriods.size() != m)
    throw std::runtime_error(
        "EigenstateSynthesis::periodGradientGeneral: " +
        std::to_string(targetPeriods.size()) + " target periods for " +
        std::to_string(m) + " holes");
  static constexpr double kNullTol = 1e-7;
  const Index N = static_cast<Index>(nk);

  // ---- M = L_k, the signed operator, kept complex: a .real() here would
  // project it, and value and gradient must see the same M ----
  const std::vector<cd> Lflat = HodgeLaplacian(st_, HodgeLaplacian::defaultWeightConvention(), metricSource_).laplacian(k_, /*metric=*/true);
  MatrixXcd M(N, N);
  for (std::size_t i = 0; i < nk; ++i)
    for (std::size_t j = 0; j < nk; ++j)
      M(static_cast<Index>(i), static_cast<Index>(j)) = Lflat[i * nk + j];

  // ---- Q (period covector, m x nk) + leak column, the assembleRegisterReadout
  // boundary convention: a hole is a removed (k+1)-cell; its facets (drop v_j,
  // sign (-1)^j) are k-cells; the leak is the first facet of the walk. ----
  std::map<std::vector<std::uint64_t>, std::size_t> col;
  for (std::size_t i = 0; i < cellOrdering_.size(); ++i) col[cellOrdering_[i]] = i;
  const std::size_t hv = static_cast<std::size_t>(k_) + 2;
  MatrixXcd Q = MatrixXcd::Zero(static_cast<Index>(m), N);
  std::vector<std::size_t> leakCol(m, 0);
  for (std::size_t q = 0; q < m; ++q) {
    std::vector<std::uint64_t> h = holes[q];
    std::sort(h.begin(), h.end());
    if (h.size() != hv)
      throw std::runtime_error(
          "EigenstateSynthesis::periodGradientGeneral: hole has " +
          std::to_string(h.size()) + " vertices, expected " + std::to_string(hv));
    std::vector<std::size_t> walk(hv);
    for (std::size_t i = 0; i < hv; ++i) walk[i] = i;
    if (k_ == 1) std::rotate(walk.begin(), walk.end() - 1, walk.end());
    for (std::size_t w = 0; w < hv; ++w) {
      const std::size_t j = walk[w];
      std::vector<std::uint64_t> f;
      f.reserve(hv - 1);
      for (std::size_t i = 0; i < hv; ++i)
        if (i != j) f.push_back(h[i]);
      const auto it = col.find(f);
      if (it == col.end())
        throw std::runtime_error(
            "EigenstateSynthesis::periodGradientGeneral: a hole facet is not a "
            "k-cell of the complex");
      if (w == 0) leakCol[q] = it->second;
      Q(static_cast<Index>(q), static_cast<Index>(it->second)) +=
          (j % 2 == 0) ? 1.0 : -1.0;
    }
  }

  // ---- harmonic (null) / non-null eigensplit of M ----
  // The signed operator is generally non-self-adjoint (real but non-symmetric
  // on the real-l^2 manifold), so a general eigensolver is required: a
  // self-adjoint one reads a single triangle and silently symmetrizes, which
  // makes this gradient come out identically zero.
  Eigen::ComplexEigenSolver<MatrixXcd> eig(M);
  const VectorXcd lam = eig.eigenvalues();
  const MatrixXcd U = eig.eigenvectors();
  std::vector<Index> nullIdx, nnIdx;
  for (Index i = 0; i < N; ++i)
    (std::abs(lam[i]) < kNullTol ? nullIdx : nnIdx).push_back(i);
  const Index nd = static_cast<Index>(nullIdx.size());
  const Index nnd = static_cast<Index>(nnIdx.size());
  if (nd == 0) return grad;  // no harmonics -> nothing carried
  MatrixXcd Un(N, nd), Unn(N, nnd);
  for (Index r = 0; r < nd; ++r) Un.col(r) = U.col(nullIdx[r]);
  for (Index r = 0; r < nnd; ++r) Unn.col(r) = U.col(nnIdx[r]);
  // Left (dual) basis for the non-self-adjoint perturbation: rows of U^-1 are
  // the covectors v_m with v_m . u_l = delta_ml, which first-order eigenvector
  // perturbation of a non-symmetric M needs in place of U^T.
  const MatrixXcd Uinv = U.inverse();
  MatrixXcd Vnn(nnd, N);
  for (Index r = 0; r < nnd; ++r) Vnn.row(r) = Uinv.row(nnIdx[r]);
  VectorXcd invlam(nnd);
  for (Index r = 0; r < nnd; ++r) invlam[r] = -1.0 / lam[nnIdx[r]];

  // ---- carried representative psi, p, rho, r_U (same as the loop core) ----
  VectorXcd target(static_cast<Index>(m));
  for (std::size_t q = 0; q < m; ++q) target[static_cast<Index>(q)] = targetPeriods[q];
  const MatrixXcd A = Q * Un;                            // m x nd
  const MatrixXcd AtAi = (A.transpose() * A).inverse();  // nd x nd
  const VectorXcd c = (AtAi * A.transpose()).cast<cd>() * target;
  VectorXcd psi = Un.cast<cd>() * c;
  const VectorXcd carried = Q.cast<cd>() * psi;
  for (std::size_t q = 0; q < m; ++q)
    psi[static_cast<Index>(leakCol[q])] +=
        target[static_cast<Index>(q)] - carried[static_cast<Index>(q)];
  const double nrm = psi.norm();
  if (nrm <= 0.0) return grad;
  const VectorXcd p = psi / nrm;
  const VectorXcd Mp = M.cast<cd>() * p;
  const double lamR = (p.dot(Mp)).real();
  const VectorXcd rho = Mp - lamR * p;
  const double rU = rho.squaredNorm();

  // ---- per-edge analytic gradient: dM_e = laplacianGradient, dense perturbation ----
  const HodgeLaplacian hl(st_, HodgeLaplacian::defaultWeightConvention(), metricSource_);
  for (std::size_t je = 0; je < edges1.size(); ++je) {
    const std::vector<cd> dMflat =
        hl.laplacianGradient(k_, edges1[je][0], edges1[je][1]);
    if (dMflat.empty()) continue;
    MatrixXcd dM(N, N);
    for (std::size_t i = 0; i < nk; ++i)
      for (std::size_t j = 0; j < nk; ++j)
        dM(static_cast<Index>(i), static_cast<Index>(j)) = dMflat[i * nk + j];
    const VectorXcd dMp = dM.cast<cd>() * p;
    // eigenvector perturbation of the harmonic block: dUn = Unn diag(invlam) Unn^T dM Un
    const MatrixXcd core = (Vnn * dM) * Un;                      // nnd x nd
    const MatrixXcd dUn = Unn * (invlam.asDiagonal() * core);    // N x nd
    const MatrixXcd dA = Q * dUn;                                // m x nd
    const MatrixXcd dAplus =
        -AtAi * (dA.transpose() * A + A.transpose() * dA) * AtAi * A.transpose() +
        AtAi * dA.transpose();                                  // nd x m
    const VectorXcd dc = dAplus.cast<cd>() * target;
    VectorXcd dpsi = dUn.cast<cd>() * c + Un.cast<cd>() * dc;
    const VectorXcd dcarried = Q.cast<cd>() * dpsi;
    for (std::size_t q = 0; q < m; ++q)
      dpsi[static_cast<Index>(leakCol[q])] += -dcarried[static_cast<Index>(q)];
    const VectorXcd Mdpsi = M.cast<cd>() * dpsi;
    grad[je] = 2.0 * (rho.dot(dMp)).real() +
               (2.0 / nrm) * (rho.dot(Mdpsi - lamR * dpsi)).real() -
               (2.0 * rU / nrm) * (p.dot(dpsi)).real();
  }
  return grad;
}

std::vector<double> EigenstateSynthesis::periodGradientDegreeZero(
    const std::vector<std::vector<std::uint64_t>> &holes,
    const std::vector<cd> &targetPeriods) const {
  // Exact d r_U / d l^2 at k = 0, against the operator residualForPeriods
  // scores: the complex Hermitian vertex operator L^U(1) = D - A
  // (HodgeLaplacian::connectionLaplacian, D_ii = sum_e |l^2_e|,
  // A_ij = l^2_e e^{i phase_e}). That is the U(1) connection Laplacian, not the
  // Hodge L_0 = d_1 W_1^-1 d_1^T: the degree-zero register carries U(1) flux,
  // and ker L_0 is always b_0, so an L_0 readout would be identically gauge-flat
  // and carry nothing. readoutHarmonicMatrix() picks the same operator, so value
  // and gradient agree.
  // The structure mirrors periodGradientGeneral in complex arithmetic; the
  // product rule d||rho||^2 = 2 Re(rho^dagger d rho) is complex-safe as-is. Two
  // differences from the k >= 1 core, both forced by the operator:
  //   * dL^U(1) per edge has exactly four entries — dL_ii = dL_jj = d|w|/dw
  //     evaluated along the real axis (Re w / |w|; the manifold is real
  //     signed l^2), dL_ij = -e^{i phase}, dL_ji = -e^{-i phase} — and no
  //     volume weights.
  //   * The least-squares fit uses the SVD pseudo-inverse and its constant-rank
  //     derivative (Golub-Pereyra): at k = 0 a globally gauge-flat harmonic has
  //     zero period on every hole, so A = Q U_n is generically
  //     column-rank-deficient and the k >= 1 cores' (A^dagger A)^{-1} would be
  //     singular. The SVD fit is what the functional's lstsqOverReadout
  //     applies, so the gradient differentiates the value returned.
  // Euler identity: L^U(1)(s l^2) = s L^U(1)(l^2) for s > 0 (degree +1), so
  // Sum_e l^2_e d r_U/d l^2_e = +2 r_U (the k >= 1 metric L_k is degree -1,
  // giving -r_U there).
  using Eigen::Index;
  using Eigen::MatrixXcd;
  using Eigen::VectorXcd;
  const std::size_t n0 = order_;  // # vertices (rows/cols of L^U(1))
  const ChainComplex cc = ChainComplex::fromSpacetime(*st_);
  const std::vector<std::vector<std::uint64_t>> edges1 = cc.kSimplexVertices(1);
  std::vector<double> grad(edges1.size(), 0.0);  // d r_U / d l^2_e, 1-cell order
  const std::size_t m = holes.size();
  if (n0 == 0 || m == 0) return grad;
  if (targetPeriods.size() != m)
    throw std::runtime_error(
        "EigenstateSynthesis::periodGradientDegreeZero: " +
        std::to_string(targetPeriods.size()) + " target periods for " +
        std::to_string(m) + " holes");
  static constexpr double kNullTol = 1e-7;
  const Index N = static_cast<Index>(n0);

  // ---- M = L^U(1) (Hermitian complex; the full l^2 and U(1) phases) ----
  const std::vector<cd> Lflat = readoutLaplacian();
  MatrixXcd M(N, N);
  for (std::size_t i = 0; i < n0; ++i)
    for (std::size_t j = 0; j < n0; ++j)
      M(static_cast<Index>(i), static_cast<Index>(j)) = Lflat[i * n0 + j];

  // ---- Q (period covector, m x n0) + leak column: a k = 0 hole is a removed
  // 1-cell (a vertex pair); its drop-v_j facets are the two vertices with sign
  // (-1)^j — the assembleRegisterReadout convention, leak on the first facet. ----
  std::map<std::vector<std::uint64_t>, std::size_t> col;
  for (std::size_t i = 0; i < cellOrdering_.size(); ++i) col[cellOrdering_[i]] = i;
  const std::size_t hv = 2;  // k + 2 vertices per hole at k = 0
  MatrixXcd Q = MatrixXcd::Zero(static_cast<Index>(m), N);
  std::vector<std::size_t> leakCol(m, 0);
  for (std::size_t q = 0; q < m; ++q) {
    std::vector<std::uint64_t> h = holes[q];
    std::sort(h.begin(), h.end());
    if (h.size() != hv)
      throw std::runtime_error(
          "EigenstateSynthesis::periodGradientDegreeZero: hole has " +
          std::to_string(h.size()) + " vertices, expected " +
          std::to_string(hv));
    for (std::size_t j = 0; j < hv; ++j) {
      std::vector<std::uint64_t> f;
      f.reserve(hv - 1);
      for (std::size_t i = 0; i < hv; ++i)
        if (i != j) f.push_back(h[i]);
      const auto it = col.find(f);
      if (it == col.end())
        throw std::runtime_error(
            "EigenstateSynthesis::periodGradientDegreeZero: a hole vertex is "
            "not a 0-cell of the complex");
      if (j == 0) leakCol[q] = it->second;
      Q(static_cast<Index>(q), static_cast<Index>(it->second)) +=
          (j % 2 == 0) ? 1.0 : -1.0;
    }
  }

  // ---- harmonic (null) / non-null eigensplit of M (Hermitian ⇒ real λ) ----
  Eigen::SelfAdjointEigenSolver<MatrixXcd> eig(M);
  const VectorXcd lam = eig.eigenvalues();
  const MatrixXcd U = eig.eigenvectors();
  std::vector<Index> nullIdx, nnIdx;
  for (Index i = 0; i < N; ++i)
    (std::abs(lam[i]) < kNullTol ? nullIdx : nnIdx).push_back(i);
  const Index nd = static_cast<Index>(nullIdx.size());
  const Index nnd = static_cast<Index>(nnIdx.size());
  if (nd == 0) return grad;  // no harmonics -> nothing carried
  MatrixXcd Un(N, nd), Unn(N, nnd);
  for (Index r = 0; r < nd; ++r) Un.col(r) = U.col(nullIdx[r]);
  for (Index r = 0; r < nnd; ++r) Unn.col(r) = U.col(nnIdx[r]);
  VectorXcd invlam(nnd);
  for (Index r = 0; r < nnd; ++r) invlam[r] = -1.0 / lam[nnIdx[r]];

  // ---- the SVD pseudo-inverse fit (min-norm, as lstsqOverReadout) ----
  VectorXcd target(static_cast<Index>(m));
  for (std::size_t q = 0; q < m; ++q) target[static_cast<Index>(q)] = targetPeriods[q];
  const MatrixXcd A = Q.cast<cd>() * Un;  // m x nd
  Eigen::JacobiSVD<MatrixXcd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
  const Index rank = svd.rank();
  MatrixXcd Aplus = MatrixXcd::Zero(nd, static_cast<Index>(m));
  if (rank > 0) {
    const MatrixXcd Ur = svd.matrixU().leftCols(rank);
    const MatrixXcd Vr = svd.matrixV().leftCols(rank);
    const VectorXcd sr = svd.singularValues().head(rank);
    Aplus = Vr * sr.cwiseInverse().asDiagonal() * Ur.adjoint();
  }
  const VectorXcd c = Aplus * target;

  // ---- carried representative psi, p, rho, r_U (as the k >= 1 cores) ----
  VectorXcd psi = Un * c;
  const VectorXcd carried = Q.cast<cd>() * psi;
  for (std::size_t q = 0; q < m; ++q)
    psi[static_cast<Index>(leakCol[q])] +=
        target[static_cast<Index>(q)] - carried[static_cast<Index>(q)];
  const double nrm = psi.norm();
  if (nrm <= 0.0) return grad;
  const VectorXcd p = psi / nrm;
  const VectorXcd Mp = M * p;
  const double lamR = (p.dot(Mp)).real();
  const VectorXcd rho = Mp - lamR * p;
  const double rU = rho.squaredNorm();

  // Constant-rank pseudo-inverse pieces: the range/row-space projectors and
  // the two Gram factors of the Golub–Pereyra derivative.
  const MatrixXcd Im_AAp =
      MatrixXcd::Identity(static_cast<Index>(m), static_cast<Index>(m)) - A * Aplus;
  const MatrixXcd Ind_ApA = MatrixXcd::Identity(nd, nd) - Aplus * A;
  const MatrixXcd ApApAdj = Aplus * Aplus.adjoint();   // nd x nd
  const MatrixXcd ApAdjAp = Aplus.adjoint() * Aplus;   // m x m

  // ---- per-edge index/value lookups ----
  std::map<std::pair<std::uint64_t, std::uint64_t>, const Edge *> edgeOf;
  for (const auto *e : edges_)
    edgeOf[{std::min(e->getSource()->getId(), e->getTarget()->getId()),
            std::max(e->getSource()->getId(), e->getTarget()->getId())}] = e;

  // ---- per-edge analytic gradient: the four-entry dL^U(1), dense perturbation ----
  for (std::size_t je = 0; je < edges1.size(); ++je) {
    const auto eIt = edgeOf.find({std::min(edges1[je][0], edges1[je][1]),
                                  std::max(edges1[je][0], edges1[je][1])});
    if (eIt == edgeOf.end()) continue;  // no live edge carries this 1-cell
    const Edge *edge = eIt->second;
    const auto srcIt = col.find({edge->getSource()->getId()});
    const auto tgtIt = col.find({edge->getTarget()->getId()});
    if (srcIt == col.end() || tgtIt == col.end()) continue;
    const Index is = static_cast<Index>(srcIt->second);
    const Index it = static_cast<Index>(tgtIt->second);
    const cd w = (edge->getLength() * edge->getLength());
    // d|w|/d(Re w): Re w / |w| — the real-axis directional derivative (sign w
    // for real w). At the |w| kink (w == 0) take the symmetric subgradient 0.
    const double dAbs = (std::abs(w) > 0.0) ? w.real() / std::abs(w) : 0.0;
    const cd zPhase = std::exp(cd(0.0, 1.0) * edge->getPhase());
    MatrixXcd dM = MatrixXcd::Zero(N, N);
    dM(is, is) += dAbs;                  // dD_ii
    dM(it, it) += dAbs;                  // dD_jj
    dM(is, it) -= zPhase;                // -dA_ij
    dM(it, is) -= std::conj(zPhase);     // -dA_ji (Hermitian)
    const VectorXcd dMp = dM * p;
    // Eigenvector perturbation of the harmonic block (complex Hermitian):
    // dUn = Unn diag(1/(0 - λ_nn)) Unn^dagger dM Un.
    const MatrixXcd core = (Unn.adjoint() * dM) * Un;         // nnd x nd
    const MatrixXcd dUn = Unn * (invlam.asDiagonal() * core); // N x nd
    const MatrixXcd dA = Q.cast<cd>() * dUn;                  // m x nd
    // Constant-rank derivative of the pseudo-inverse (Golub–Pereyra):
    // dA+ = -A+ dA A+ + (A+ A+^dagger) dA^dagger (I - A A+)
    //       + (I - A+ A) dA^dagger (A+^dagger A+).
    const MatrixXcd dAplus = -Aplus * dA * Aplus +
                             ApApAdj * dA.adjoint() * Im_AAp +
                             Ind_ApA * dA.adjoint() * ApAdjAp;
    const VectorXcd dc = dAplus * target;
    VectorXcd dpsi = dUn * c + Un * dc;
    const VectorXcd dcarried = Q.cast<cd>() * dpsi;
    for (std::size_t q = 0; q < m; ++q)
      dpsi[static_cast<Index>(leakCol[q])] += -dcarried[static_cast<Index>(q)];
    const VectorXcd Mdpsi = M * dpsi;
    grad[je] = 2.0 * (rho.dot(dMp)).real() +
               (2.0 / nrm) * (rho.dot(Mdpsi - lamR * dpsi)).real() -
               (2.0 * rU / nrm) * (p.dot(dpsi)).real();
  }
  return grad;
}

std::vector<double> EigenstateSynthesis::residualForPeriodsGradient(
    const std::vector<std::vector<std::uint64_t>> &holes,
    const std::vector<cd> &targetPeriods) const {
  // Arbitrary-degree exact d r_U / d l^2, in ChainComplex 1-cell (edge) order.
  // At k = 1 (triangle holes) this routes through the fast low-rank edge-loop
  // core (periodGradientOverLoops), which builds the chain complex once and uses
  // a per-edge low-rank dM, so a relaxation loop stays affordable; it is
  // value-identical to the general path (verified to 1.7e-15). For k >= 2 the
  // degree-generic periodGradientGeneral is used (M = L_k, the per-edge analytic
  // dL_k/dl^2). Both satisfy the exact Euler identity
  // Sum_e l^2_e d r_U/d l^2_e = -2 r_U: with the V^2 weights L_k is homogeneous
  // of degree -1 in l^2 and r_U = ||(L - lambda)p||^2 of degree -2, measured as
  // r_U(s*l^2) = r_U/s^2 exactly.
  if (k_ == 1 && metricSource_ == HodgeLaplacian::MetricSource::DiagonalWeights)
    return periodGradientOverLoops(
        holeLoops(holes, "EigenstateSynthesis::residualForPeriodsGradient"),
        targetPeriods);
  // Under the Whitney pencil the loop core's low-rank dM is replaced by the
  // dense analytic dL_1/dl^2 inside periodGradientOverLoops; the general core
  // reads the same derivative through HodgeLaplacian::laplacianGradient.
  return periodGradientGeneral(holes, targetPeriods);
}

std::vector<cd> EigenstateSynthesis::periodGapForLoopsGradient(
    const std::vector<EdgeLoop> &loops,
    const std::vector<cd> &targetPeriods) const {
  // Degree-1 machinery, exactly as periodGradientOverLoops (and as
  // the r_psi functional itself — periodGapForLoops reads loop periods of an
  // edge cochain, defined only at k = 1).
  if (k_ != 1)
    throw std::runtime_error(
        "EigenstateSynthesis::periodGapForLoopsGradient: the edge-loop core "
        "is degree-1 machinery (loops are closed walks of 1-cells); this "
        "synthesis is degree " + std::to_string(k_) + ".");
  // The hard-pin sibling of periodGradientOverLoops (r_U): same first-order
  // eigenvector-perturbation setup (M = L1, harmonic split Un/Unn, the per-edge
  // low-rank dM, dUn), but the score is the period gap r_psi = ||A c - t||^2
  // with A = Q Un and c the least-squares fit, rather than the leaked state's
  // non-harmonicity. Least-squares optimality A^T r = 0 (envelope theorem) drops
  // the dc term, so d r_psi / d l^2 = 2 Re( r^H (Q dUn) c ): no leak, no dpsi
  // chain.
  //
  // The M / eigensplit / per-edge-dM machinery below duplicates
  // periodGradientOverLoops so that the r_U gradient, which is finite-difference
  // and GPU-mirror verified, need not be touched. Keep the two copies in sync;
  // test_period_gap_python.py finite-difference-guards this one.
  using Eigen::Index;
  using Eigen::MatrixXcd;
  using Eigen::VectorXcd;
  const std::size_t n1 = order_;
  std::vector<cd> grad(n1, cd(0.0, 0.0));
  const std::size_t m = loops.size();
  if (n1 == 0 || m == 0) return grad;
  if (targetPeriods.size() != m)
    throw std::runtime_error(
        "EigenstateSynthesis::periodGapForLoopsGradient: " +
        std::to_string(targetPeriods.size()) + " target periods for " +
        std::to_string(m) + " loops");
  static constexpr double kNullTol = 1e-7;
  const Index N = static_cast<Index>(n1);

  // ---- chain complex, weights, and the metric Laplacian M = L1 ----
  const ChainComplex cc = ChainComplex::fromSpacetime(*st_);
  const std::vector<std::vector<std::uint64_t>> &cells1 = cellSimplices();
  const auto tris = cc.kSimplexVertices(2);
  const std::size_t n2 = tris.size();
  const std::vector<long> &d1flat = cc.boundaryMatrix(1);  // n0 x n1
  const std::vector<long> &d2flat = cc.boundaryMatrix(2);  // n1 x n2
  const std::size_t n0 = d1flat.size() / n1;
  const HodgeLaplacian hl(st_, HodgeLaplacian::defaultWeightConvention(), metricSource_);
  const std::vector<cd> W1v = hl.weights(1);  // n1, signed complex
  const std::vector<cd> W2v = hl.weights(2);  // n2, signed complex
  const std::vector<cd> Lflat = hl.laplacian(1, /*metric=*/true);

  MatrixXcd M(N, N);
  for (std::size_t i = 0; i < n1; ++i)
    for (std::size_t j = 0; j < n1; ++j)
      M(static_cast<Index>(i), static_cast<Index>(j)) = Lflat[i * n1 + j];
  VectorXcd W1(N);
  for (std::size_t i = 0; i < n1; ++i) W1[static_cast<Index>(i)] = W1v[i];
  MatrixXcd d1m(static_cast<Index>(n0), N);
  for (std::size_t v = 0; v < n0; ++v)
    for (std::size_t c = 0; c < n1; ++c)
      d1m(static_cast<Index>(v), static_cast<Index>(c)) =
          static_cast<double>(d1flat[v * n1 + c]);
  MatrixXcd d2m(N, static_cast<Index>(n2));
  for (std::size_t c = 0; c < n1; ++c)
    for (std::size_t t = 0; t < n2; ++t)
      d2m(static_cast<Index>(c), static_cast<Index>(t)) =
          static_cast<double>(d2flat[c * n2 + t]);
  const MatrixXcd K1 = d1m.transpose() * d1m;  // n1 x n1
  VectorXcd W2inv(static_cast<Index>(n2));
  for (std::size_t t = 0; t < n2; ++t) W2inv[static_cast<Index>(t)] = 1.0 / W2v[t];
  const MatrixXcd K2 = d2m * W2inv.asDiagonal() * d2m.transpose();  // n1 x n1

  // ---- index maps: cell -> index, edge -> l^2, edge -> incident triangles ----
  auto key = [](std::uint64_t a, std::uint64_t b) {
    return std::pair<std::uint64_t, std::uint64_t>(std::min(a, b), std::max(a, b));
  };
  std::map<std::pair<std::uint64_t, std::uint64_t>, std::size_t> cidx1;
  for (std::size_t i = 0; i < n1; ++i) cidx1[key(cells1[i][0], cells1[i][1])] = i;
  std::map<std::pair<std::uint64_t, std::uint64_t>, std::complex<double>> l2map;
  for (auto *e : edges_)
    l2map[key(e->getSource()->getId(), e->getTarget()->getId())] =
        (e->getLength() * e->getLength());
  std::map<std::pair<std::uint64_t, std::uint64_t>, std::vector<std::size_t>> trisOf;
  for (std::size_t ti = 0; ti < n2; ++ti)
    for (int i = 0; i < 3; ++i)
      for (int j = i + 1; j < 3; ++j)
        trisOf[key(tris[ti][i], tris[ti][j])].push_back(ti);
  auto L2 = [&](std::uint64_t a, std::uint64_t b) -> cd {
    if (a == b) return cd(0.0, 0.0);
    auto it = l2map.find(key(a, b));
    return it == l2map.end() ? cd(0.0, 0.0) : it->second;
  };

  // ---- Q (signed edge-loop covector); the gap needs no leak column ----
  MatrixXcd Q = MatrixXcd::Zero(static_cast<Index>(m), N);
  for (std::size_t q = 0; q < m; ++q) {
    if (loops[q].empty())
      throw std::runtime_error(
          "EigenstateSynthesis::periodGapForLoopsGradient: loop " +
          std::to_string(q) + " is empty");
    Edge::walkLoop(loops[q], [&](std::uint64_t a, std::uint64_t b, double s) {
      Q(static_cast<Index>(q), static_cast<Index>(cidx1.at(key(a, b)))) += s;
    });
  }

  // ---- eigendecomposition of M; harmonic (null) / non-null split ----
  // The signed operator is generally non-self-adjoint (real but non-symmetric
  // on the real-l^2 manifold), so a general eigensolver is required: a
  // self-adjoint one reads a single triangle and silently symmetrizes, which
  // makes this gradient come out identically zero.
  Eigen::ComplexEigenSolver<MatrixXcd> eig(M);
  const VectorXcd lam = eig.eigenvalues();
  const MatrixXcd U = eig.eigenvectors();
  std::vector<Index> nullIdx, nnIdx;
  for (Index i = 0; i < N; ++i) (std::abs(lam[i]) < kNullTol ? nullIdx : nnIdx).push_back(i);
  const Index nd = static_cast<Index>(nullIdx.size());
  const Index nnd = static_cast<Index>(nnIdx.size());
  if (nd == 0) return grad;  // no harmonics -> nothing carried, gap is constant
  MatrixXcd Un(N, nd), Unn(N, nnd);
  for (Index r = 0; r < nd; ++r) Un.col(r) = U.col(nullIdx[r]);
  for (Index r = 0; r < nnd; ++r) Unn.col(r) = U.col(nnIdx[r]);
  // Left (dual) basis for the non-self-adjoint perturbation: rows of U^-1 are
  // the covectors v_m with v_m . u_l = delta_ml, which first-order eigenvector
  // perturbation of a non-symmetric M needs in place of U^T.
  const MatrixXcd Uinv = U.inverse();
  MatrixXcd Vnn(nnd, N);
  for (Index r = 0; r < nnd; ++r) Vnn.row(r) = Uinv.row(nnIdx[r]);
  VectorXcd invlam(nnd);  // 1 / (0 - lambda_nn) for the eigenvector perturbation
  for (Index r = 0; r < nnd; ++r) invlam[r] = -1.0 / lam[nnIdx[r]];

  // ---- the least-squares fit c and the period-gap residual r = A c - target ----
  // Rank-robust min-norm fit (Jacobi SVD), matching periodGapForLoops's value so
  // the gradient stays consistent with the function it differentiates, and so a
  // column-rank-deficient A (carried harmonic count nd > read-out cycle count m,
  // or dependent/zero-period harmonics) yields the min-norm solution rather than
  // the NaN a singular (A^T A)^{-1} would give. A^T r = 0 holds for any
  // least-squares solution, so the dropped dc term is unaffected.
  VectorXcd target(static_cast<Index>(m));
  for (std::size_t q = 0; q < m; ++q) target[static_cast<Index>(q)] = targetPeriods[q];
  const MatrixXcd A = Q * Un;                            // m x nd
  const MatrixXcd Ac = A.cast<cd>();
  const VectorXcd c =
      Ac.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(target);  // nd
  const VectorXcd r = Ac * c - target;                 // m; A^T r = 0 (optimality)

  // ---- per-edge analytic gradient d r_psi / d l^2 (low-rank dM + perturbation) ----
  for (std::size_t je = 0; je < n1; ++je) {
    const Index j = static_cast<Index>(je);
    const auto ek = key(cells1[je][0], cells1[je][1]);
    // dM for the signed operator M = W1^-1 K1 + K2 W1 (K2 = d2 W2^-1 d2^T),
    // under the V^2 weights (the HodgeLaplacian default): W1_j = l^2_j exactly,
    // so dW1_j/dl^2_j = 1 and, every piece rank one,
    //   dM = -(1/W1_j^2) e_j (K1 row j)                       [d(W1^-1) K1]
    //      + (K2 col j) e_j^T                                 [K2 W1 -> K2 dW1]
    //      + per triangle t on e:
    //        -(dW2_t/W2_t^2) (d2 col t)((W1 o d2 col t))^T    [d(W2^-1) term]
    // dM is generally non-symmetric, like M itself.
    std::vector<VectorXcd> colsA, colsB;
    VectorXcd ev = VectorXcd::Zero(N);
    ev[j] = 1.0;
    colsA.push_back(ev);
    colsB.push_back((-1.0 / (W1[j] * W1[j])) * K1.row(j).transpose());
    colsA.push_back(K2.col(j));
    colsB.push_back(ev);
    for (std::size_t ti : trisOf[ek]) {
      const auto &t = tris[ti];
      Eigen::Matrix2cd G;
      for (int i = 0; i < 2; ++i)
        for (int jj = 0; jj < 2; ++jj)
          G(i, jj) = 0.5 * (L2(t[0], t[i + 1]) + L2(t[0], t[jj + 1]) - L2(t[i + 1], t[jj + 1]));
      const cd detG = G.determinant();
      const cd W2ti = W2v[ti];
      // W2 must be the V^2 weight detG/4 this derivation assumes.
      if (std::abs(detG / 4.0 - W2ti) > 1e-9 * std::max(1.0, std::abs(W2ti)) ||
          std::abs(detG) < 1e-12)
        continue;
      auto ind = [&](int pp, int qq) -> double {
        return (pp != qq && key(t[pp], t[qq]) == ek) ? 1.0 : 0.0;
      };
      Eigen::Matrix2cd dG;
      for (int i = 0; i < 2; ++i)
        for (int jj = 0; jj < 2; ++jj)
          dG(i, jj) = 0.5 * (ind(0, i + 1) + ind(0, jj + 1) - ind(i + 1, jj + 1));
      // W2 = detG/4 => dW2 = W2 * tr(G^-1 dG), by Jacobi's formula.
      const cd dW2ti = W2ti * (G.inverse() * dG).trace();
      const VectorXcd dcol = d2m.col(static_cast<Index>(ti));
      colsA.push_back(dcol);
      colsB.push_back((-dW2ti / (W2ti * W2ti)) * W1.cwiseProduct(dcol));
    }
    const Index rk = static_cast<Index>(colsA.size());
    MatrixXcd fa(N, rk), fb(N, rk);
    for (Index k = 0; k < rk; ++k) {
      fa.col(k) = colsA[static_cast<std::size_t>(k)];
      fb.col(k) = colsB[static_cast<std::size_t>(k)];
    }
    // The harmonic-subspace perturbation dUn, then dA = Q dUn; the envelope
    // theorem (A^T r = 0) leaves only 2 Re( r^H (dA c) ).
    const MatrixXcd core = (Vnn * fa) * (fb.transpose() * Un);  // nnd x nd
    const MatrixXcd dUn = Unn * (invlam.asDiagonal() * core);               // n1 x nd
    const MatrixXcd dA = Q * dUn;                                           // m x nd
    // Complex gradient: r is holomorphic in l^2, so no .real() projection
    // belongs here. The value is
    //   g = dr/d(Re l^2) - i dr/d(Im l^2),
    // whose real part is the real-locus derivative. Discarding the imaginary
    // half would leave the register term unable to move in the plane the
    // descent direction steps in.
    grad[je] = 2.0 * r.dot(dA.cast<cd>() * c);
  }
  return grad;
}

std::vector<cd> EigenstateSynthesis::periodGapForPeriodsGradient(
    const std::vector<std::vector<std::uint64_t>> &holes,
    const std::vector<cd> &targetPeriods) const {
  // Route by degree, exactly as residualForPeriodsGradient does: the fast
  // low-rank edge-loop core at k = 1, the degree-generic core at k >= 2. k = 0
  // reads a different operator — the complex Hermitian U(1) connection
  // L^U(1) = D - A, not a weight variant of L_k — so it has no period-gap core
  // at all, and that is stated here at the entry point rather than deeper in.
  if (k_ == 0)
    throw std::runtime_error(
        "EigenstateSynthesis::periodGapForPeriodsGradient: the period gap has "
        "no core at degree 0 — the U(1) connection L = D - A it reads is a "
        "different (complex Hermitian) operator, not a weight variant of "
        "L_k; this synthesis is degree " +
        std::to_string(k_) + ".");
  if (k_ == 1)
    return periodGapForLoopsGradient(
        holeLoops(holes, "EigenstateSynthesis::periodGapForPeriodsGradient"),
        targetPeriods);
  return periodGapGradientOverHoles(holes, targetPeriods);
}

std::vector<cd> EigenstateSynthesis::periodGapGradientOverHoles(
    const std::vector<std::vector<std::uint64_t>> &holes,
    const std::vector<cd> &targetPeriods) const {
  // Arbitrary-degree exact d r_psi / d l^2 for the period gap
  // r_psi = ||A c - t||^2, with A = Q U_n the periods of the harmonic basis and
  // c the least-squares fit. The setup mirrors periodGradientGeneral term for
  // term — same M = L_k, same period covector Q, same non-self-adjoint
  // eigensplit — but the score differs, and so does the derivative:
  //
  // least-squares optimality gives A^dagger r = 0, so by the envelope theorem the
  // dc term drops out entirely and
  //     d r_psi / d l^2_e = 2 Re( r^dagger (Q dU_n) c ),
  // where dU_n is the first-order perturbation of the harmonic block. That is
  // the same identity the k = 1 loop core uses; this is its degree-generic form.
  //
  // The fit uses the SVD pseudo-inverse rather than normal equations: A can be
  // rank-deficient (more harmonics than holes, or degenerate periods), where
  // (A^T A)^-1 is singular. It matches lstsqOverReadout, which is what the value
  // uses, so gradient and value fit the same way.
  //
  // Certified by the Euler identity for a degree-0 functional:
  //     Sum_e l^2_e d r_psi / d l^2_e = 0.
  // L_k is homogeneous of degree -1 in l^2, so l^2 -> s l^2 sends L -> L/s,
  // which leaves the kernel — and hence the normalized harmonic basis, A, c and
  // the gap — unchanged. By contrast r_U is of degree -2, with Euler sum -2 r_U.
  using Eigen::Index;
  using Eigen::MatrixXcd;
  using Eigen::VectorXcd;
  const std::size_t nk = order_;
  const ChainComplex cc = ChainComplex::fromSpacetime(*st_);
  const std::vector<std::vector<std::uint64_t>> edges1 = cc.kSimplexVertices(1);
  std::vector<cd> grad(edges1.size(), cd(0.0, 0.0));
  const std::size_t m = holes.size();
  if (nk == 0 || m == 0) return grad;
  if (targetPeriods.size() != m)
    throw std::runtime_error(
        "EigenstateSynthesis::periodGapGradientOverHoles: " +
        std::to_string(targetPeriods.size()) + " target periods for " +
        std::to_string(m) + " holes");
  if (k_ == 0)
    throw std::runtime_error(
        "EigenstateSynthesis::periodGapGradientOverHoles: no period-gap core at "
        "degree 0 — the U(1) connection L = D - A it reads is a different "
        "(complex Hermitian) operator, not a weight variant of L_k; this "
        "synthesis is degree " +
        std::to_string(k_) + ".");
  static constexpr double kNullTol = 1e-9;   // harmonicMatrix's tolerance: this
                                             // must differentiate the harmonic
                                             // set the value reads, not r_U's 1e-7
  const Index N = static_cast<Index>(nk);

  const std::vector<cd> Lflat = HodgeLaplacian(st_, HodgeLaplacian::defaultWeightConvention(), metricSource_).laplacian(k_, /*metric=*/true);
  MatrixXcd M(N, N);
  for (std::size_t i = 0; i < nk; ++i)
    for (std::size_t j = 0; j < nk; ++j)
      M(static_cast<Index>(i), static_cast<Index>(j)) = Lflat[i * nk + j];

  // Period covector Q: a hole is a removed (k+1)-cell, its drop-one facets are
  // k-cells with sign (-1)^j — the assembleRegisterReadout convention.
  std::map<std::vector<std::uint64_t>, std::size_t> col;
  for (std::size_t i = 0; i < cellOrdering_.size(); ++i) col[cellOrdering_[i]] = i;
  const std::size_t hv = static_cast<std::size_t>(k_) + 2;
  MatrixXcd Q = MatrixXcd::Zero(static_cast<Index>(m), N);
  for (std::size_t q = 0; q < m; ++q) {
    std::vector<std::uint64_t> h = holes[q];
    std::sort(h.begin(), h.end());
    if (h.size() != hv)
      throw std::runtime_error(
          "EigenstateSynthesis::periodGapGradientOverHoles: hole has " +
          std::to_string(h.size()) + " vertices, expected " + std::to_string(hv));
    for (std::size_t j = 0; j < hv; ++j) {
      std::vector<std::uint64_t> f;
      f.reserve(hv - 1);
      for (std::size_t i = 0; i < hv; ++i)
        if (i != j) f.push_back(h[i]);
      const auto it = col.find(f);
      if (it == col.end())
        throw std::runtime_error(
            "EigenstateSynthesis::periodGapGradientOverHoles: a hole facet is "
            "not a k-cell of the complex");
      Q(static_cast<Index>(q), static_cast<Index>(it->second)) +=
          (j % 2 == 0) ? 1.0 : -1.0;
    }
  }

  // Non-self-adjoint eigensplit: the signed operator is real but not symmetric,
  // so a self-adjoint solver would read one triangle and symmetrize.
  Eigen::ComplexEigenSolver<MatrixXcd> eig(M);
  const VectorXcd lam = eig.eigenvalues();
  const MatrixXcd U = eig.eigenvectors();
  std::vector<Index> nullIdx, nnIdx;
  for (Index i = 0; i < N; ++i)
    (std::abs(lam[i]) < kNullTol ? nullIdx : nnIdx).push_back(i);
  const Index nd = static_cast<Index>(nullIdx.size());
  const Index nnd = static_cast<Index>(nnIdx.size());
  if (nd == 0) return grad;      // no harmonics: nothing carried, gap constant
  MatrixXcd Un(N, nd), Unn(N, nnd);
  for (Index r = 0; r < nd; ++r) Un.col(r) = U.col(nullIdx[r]);
  for (Index r = 0; r < nnd; ++r) Unn.col(r) = U.col(nnIdx[r]);
  const MatrixXcd Uinv = U.inverse();
  MatrixXcd Vnn(nnd, N);
  for (Index r = 0; r < nnd; ++r) Vnn.row(r) = Uinv.row(nnIdx[r]);
  VectorXcd invlam(nnd);
  for (Index r = 0; r < nnd; ++r) invlam[r] = -1.0 / lam[nnIdx[r]];

  VectorXcd target(static_cast<Index>(m));
  for (std::size_t q = 0; q < m; ++q) target[static_cast<Index>(q)] = targetPeriods[q];
  const MatrixXcd A = Q * Un;                                   // m x nd
  Eigen::BDCSVD<MatrixXcd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
  const VectorXcd c = svd.solve(target);                        // min-norm fit
  const VectorXcd r = A * c - target;                           // the gap vector

  const HodgeLaplacian hl(st_, HodgeLaplacian::defaultWeightConvention(), metricSource_);
  for (std::size_t je = 0; je < edges1.size(); ++je) {
    const std::vector<cd> dMflat =
        hl.laplacianGradient(k_, edges1[je][0], edges1[je][1]);
    if (dMflat.empty()) continue;
    MatrixXcd dM(N, N);
    for (std::size_t i = 0; i < nk; ++i)
      for (std::size_t j = 0; j < nk; ++j)
        dM(static_cast<Index>(i), static_cast<Index>(j)) = dMflat[i * nk + j];
    const MatrixXcd core = (Vnn * dM) * Un;                     // nnd x nd
    const MatrixXcd dUn = Unn * (invlam.asDiagonal() * core);   // N x nd
    const MatrixXcd dA = Q * dUn;                               // m x nd
    // Envelope theorem, kept complex; see the k = 1 core.
    grad[je] = 2.0 * r.dot(dA * c);
  }
  return grad;
}


std::vector<double> EigenstateSynthesis::residualForLoopsGradient(
    const std::vector<EdgeLoop> &loops,
    const std::vector<cd> &targetPeriods) const {
  return periodGradientOverLoops(loops, targetPeriods);
}

}  // namespace tessera::cobordism
