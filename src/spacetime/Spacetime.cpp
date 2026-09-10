// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

//
// Created by andrew on 10/23/25.
//

// (was: #include <pybind11/pybind11.h> — removed; unreferenced.)
#include "Logger.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <map>
#include <memory>
#include <queue>
#include <set>
#include <stdexcept>
#include <string>
#include "spacetime/Spacetime.h"
#include <cstdio>
#include "graph/CSRBuilder.hpp"
#include "graph/DualGraph.hpp"
#include "graph/IndexByKey.hpp"
#include "graph/SpectralGraph.hpp"
#include "mesh/SimplexFilter.h"
#include "observables/MIUnits.hpp"
#include "observables/SparseGraph.h"
#include "mesh/TemporalOrientation.h"
#include "mesh/ForwardDeclarations.h"
#include "mesh/EdgeList.h"
#include "mesh/Edge.h"

#include <tuple>
#include <unordered_map>
#include "spacetime/topologies/Toroid.h"

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

// ========================================
// Constructors
// ========================================

Spacetime::Spacetime() {
  Signature signature(4, SignatureType::Lorentzian);
  metric = std::make_shared<Metric>(true, signature);
  spacetimeType = SpacetimeType::CDT;
  alpha = 1.;
  topology = std::make_shared<Toroid>();
  a = 1.;
  foliation = Foliation::PREFERRED;
}

Spacetime::Spacetime(
  std::shared_ptr<Metric> metric_,
  const SpacetimeType spacetimeType_,
  std::optional<double> alpha_,
  std::optional<double> a_,
  Foliation foliation_,
  std::optional<std::shared_ptr<Topology> > topology_) : metric(std::move(metric_)), spacetimeType(spacetimeType_), foliation(foliation_) {
  alpha = alpha_.value_or(1.);
  a = a_.value_or(1.);
  topology = topology_.has_value() ? std::move(*topology_) : std::make_shared<Toroid>();
}

// ========================================
// Creation Methods
// ========================================

std::pair<SimplexPtr, bool> Spacetime::createSimplex(
  const VertexPtrs &vertices,
  const Edges &edges
) {
  // A Simplex's identity is its Fingerprint, which stores at most kMax vertex
  // IDs (mesh/Fingerprint.h). Past that, addId() silently drops IDs, so two
  // distinct >kMax-vertex simplices can share a (truncated) fingerprint and the
  // second is silently treated as a duplicate — never registered, but returned
  // with created=true. Fail loudly instead of corrupting the complex (issue
  // #77). kMax = 8 supports simplices up to dimension 7, well beyond CDT (≤5
  // vertices) and the cobordism extension (≤6).
  if (vertices.size() > kMax) {
    throw std::invalid_argument(
        "Spacetime::createSimplex: " + std::to_string(vertices.size()) +
        "-vertex simplex exceeds the Fingerprint capacity kMax=" +
        std::to_string(kMax) + " (max simplex dimension " +
        std::to_string(kMax - 1) + "). See issue #77.");
  }
  // Compute hash directly without allocating a temporary Fingerprint.
  std::uint64_t hash = 0;
  for (const auto &v : vertices) {
    hash ^= Fingerprint::mix64(v->getId());
  }
  auto *found = simplexIndex_.find(hash);
  if (found) {
#ifdef TESSERA_ASSERTIONS
    CLOG(DEBUG_LEVEL, "You attempted to create a simplex that already exists: ", (*found)->toString());
#endif
    return {*found, false};
  }
  // Construct the Simplex in place in simplexStorage_ so its address is
  // stable for the lifetime of this Spacetime.  initialize() registers the
  // simplex pointer back on its vertices, so we must do that AFTER emplace
  // (which is when &simplexStorage_.back() becomes the canonical address).
  SimplexPtr simplex;
  if (!freeSimplexSlots_.empty()) {
    // Reuse a slot a removed simplex left behind. Its address is already
    // stable, and assigning over it keeps that address, so the deque never has
    // to grow. The generation carries across the assignment and is bumped, so a
    // pointer taken before the reuse is distinguishable from one taken after.
    const std::uint32_t slot = freeSimplexSlots_.back();
    freeSimplexSlots_.pop_back();
    const std::uint32_t generation = simplexStorage_[slot].generation_ + 1;
    simplexStorage_[slot] = Simplex(this, vertices, edges);
    simplex = &simplexStorage_[slot];
    simplex->generation_ = generation;
    simplex->poolSlot_ = slot;
  } else {
    simplexStorage_.emplace_back(this, vertices, edges);
    simplex = &simplexStorage_.back();
    simplex->poolSlot_ = static_cast<std::uint32_t>(simplexStorage_.size() - 1);
  }
  if (!simplex->initialized) {
    simplex->initialize(simplex);
  }
  registerSimplex(simplex, false);
  return {simplex, true};
}

std::pair<SimplexPtr, bool> Spacetime::createSimplex(
  const VertexPtrs &vertices
) {
  auto r = createSimplexTracked(vertices);
  return {r.simplex, r.created};
}

Spacetime::CreateSimplexResult Spacetime::createSimplexTracked(
  const VertexPtrs &vertices
) {
  CreateSimplexResult result;
  Edges edges_{};
  for (std::size_t i = 0; i < vertices.size() - 1; i++) {
    for (std::size_t j = i + 1; j < vertices.size(); j++) {
      // Auto-wired causal LENGTHS (#639): same-time => spacelike l = sqrt(a),
      // cross-slice => timelike l = i*sqrt(alpha*a) (l^2 = -alpha*a). Always
      // Lorentzian — the old signature guard made a non-Lorentzian metric wire
      // every edge spacelike, which was a Euclidean path (#641).
      const std::complex<double> len = autoWiredLength(
          vertices[i]->getTime() != vertices[j]->getTime());
      auto [edge, inserted] =
        edgeList->tryAdd(vertices[i], vertices[j], len);
      // Mirror createEdge: register the edge on the endpoints'
      // adjacency lists.  addOutEdge/addInEdge dedupe internally so
      // calling them on a pre-existing edge is a no-op.
      vertices[i]->addOutEdge(edge);
      vertices[j]->addInEdge(edge);
      edges_.push_back(edge);
      if (inserted) result.newEdges.push_back(edge);
    }
  }
  auto [simplex, created] = createSimplex(vertices, edges_);
  result.simplex = simplex;
  result.created = created;
  // If the simplex already existed, every edge we touched was also
  // already there — tryAdd returned inserted=false for each — so
  // newEdges is empty.  Belt-and-braces:
  if (!created) result.newEdges.clear();
  return result;
}

std::pair<SimplexPtr, bool> Spacetime::createSimplex(const std::tuple<uint8_t, uint8_t> &numericOrientation) {
  // The factory speaks LENGTHS (#639): spacelike ℓ = √a is real, timelike
  // ℓ = i·√(α·a) is imaginary (ℓ² = -α·a). There is no non-Lorentzian branch —
  // the old "Euclidean: all edges positive" fallback was a Euclidean path and
  // is gone (#641).
  const std::complex<double> spacelikeLength = autoWiredLength(false);
  const std::complex<double> timelikeLength = autoWiredLength(true);
  TemporalOrientation orientation = {
    std::get<0>(numericOrientation),
    std::get<1>(numericOrientation)
  };
  std::uint8_t k = orientation.getK();
  auto [ti, tf] = orientation.numeric();
  VertexPtrs vertices = {};
  vertices.reserve(k);
  Edges edges = {};
  edges.reserve(Simplex::computeNumberOfEdges(k));
  for (int i = 0; i < ti; i++) {
    // Create ti vertices at currentTime (initial time slice).
    VertexPtr newVertex = vertexList->add(vertexIdCounter++, {static_cast<double>(currentTime)});
    for (const auto &existingVertex : vertices) {
      EdgePtr edge = edgeList->
          add(existingVertex, newVertex, spacelikeLength);
      existingVertex->addOutEdge(edge);
      newVertex->addInEdge(edge);
      edges.push_back(edge);
    }
    vertices.push_back(newVertex);
  }
  for (int i = 0; i < tf; i++) {
    // Create ti Spacelike vertices
    // Use coning to construct the vertex edges. For each new vertex; draw an edge to each existing vertex.
    /// We can't just use the vertexList .size() here, because some vertices can be removed. We need to keep a
    /// counter:
    VertexPtr newVertex = vertexList->add(vertexIdCounter++, {static_cast<double>(currentTime + 1)});
    for (const auto &existingVertex : vertices) {
      EdgePtr edge;
      if (existingVertex->getTime() < newVertex->getTime()) {
        edge = edgeList->add(existingVertex, newVertex, timelikeLength);
      } else {
        edge = edgeList->add(existingVertex, newVertex, spacelikeLength);
      }
      existingVertex->addOutEdge(edge);
      newVertex->addInEdge(edge);
      edges.push_back(edge);
    }
    vertices.push_back(newVertex);
  }
  return createSimplex(vertices, edges);
}

double Spacetime::getAlpha() const noexcept {
  return alpha;
}

double Spacetime::getA() const noexcept {
  return a;
}

std::pair<SimplexPtr, bool> Spacetime::createSimplex(std::size_t k) {
  const std::complex<double> spacelikeLength = autoWiredLength(false);  // same-time class
  VertexPtrs vertices = {};
  vertices.reserve(k);
  Edges edges = {};
  edges.reserve(Simplex::computeNumberOfEdges(k));
  for (int i = 0; i < k; i++) {
    // Use coning to construct the vertex edges. For each new vertex; draw an edge to each existing vertex.
    VertexPtr newVertex = vertexList->add(vertexIdCounter++, {static_cast<double>(currentTime)});
    for (const auto &existingVertex : vertices) {
      EdgePtr edge = edgeList->add(existingVertex, newVertex, spacelikeLength);
      existingVertex->addOutEdge(edge);
      newVertex->addInEdge(edge);
      edges.push_back(edge);
    }
    vertices.push_back(newVertex);
  }
  return createSimplex(vertices, edges);
}

std::uint64_t Spacetime::nextFreeVertexId() noexcept {
  // Advance past any id already in use (explicit createVertex(id) or a topology
  // builder) so a no-arg / reserved id never collides with an existing vertex.
  // VertexList::add on a duplicate id returns the EXISTING vertex — a silent
  // alias that, when coned, makes a self-edge (#267).
  while (vertexList->contains(vertexIdCounter)) ++vertexIdCounter;
  return vertexIdCounter++;
}

VertexPtr Spacetime::createVertex() noexcept {
  return vertexList->add(nextFreeVertexId());
}

VertexPtr Spacetime::createVertex(const std::uint64_t id) const noexcept {
  return vertexList->add(id);
}

VertexPtr Spacetime::createVertex(const std::uint64_t id, const std::vector<double> &coords) const noexcept {
  return vertexList->add(id, coords);
}

VertexPtr Spacetime::createVertex(const std::vector<double> &coords) noexcept {
  return vertexList->add(nextFreeVertexId(), coords);
}

EdgePtr Spacetime::createEdge(
  const VertexPtr &src,
  const VertexPtr &tgt
) const noexcept {
  EdgePtr edge = edgeList->add(src, tgt);
  src->addOutEdge(edge);
  tgt->addInEdge(edge);
  ++structuralRevision_;                 // a 1-cell appeared without a simplex
  return edge;
}

EdgePtr Spacetime::createEdge(
  const VertexPtr &src,
  const VertexPtr &tgt,
  std::complex<double> length
) const noexcept {
#ifdef TESSERA_ASSERTIONS
  const std::complex<double> squaredLength = length * length;
  if (src->getTime() == tgt->getTime() &&
      !(squaredLength.imag() == 0.0 && squaredLength.real() > 0.0)) {
    CLOG(INFO_LEVEL, "You attempted to create a same-time (spacelike) edge with non-positive squared length");
    std::abort();
  }
#endif
  EdgePtr edge = edgeList->add(src, tgt, length);
  src->addOutEdge(edge);
  tgt->addInEdge(edge);
  ++structuralRevision_;                 // a 1-cell appeared without a simplex
  return edge;
}

// ========================================
// Complex Building Methods
// ========================================
void Spacetime::build(int numSimplices) {
  return topology->build(this, numSimplices);
}

std::shared_ptr<Spacetime> Spacetime::fromCells(
  int dimensions,
  const std::vector<std::vector<std::uint64_t>> &cells,
  double weight,
  std::complex<double> phase,
  const std::optional<std::vector<double>> &vertexTimes
) {
  auto metric = std::make_shared<Metric>(
    true, Signature(dimensions, SignatureType::Lorentzian));
  auto st = std::make_shared<Spacetime>(
    metric, SpacetimeType::CDT, 1.0, 1.0, Foliation::PREFERRED, std::nullopt);

  // One vertex per distinct id, created in ascending id order so the labels are
  // deterministic. Under the tracked-metric rule each vertex carries its single
  // time coordinate (arity one — never the length-2/3 vector getTime() rejects).
  std::set<std::uint64_t> ids;
  for (const auto &cell : cells)
    for (auto v : cell) ids.insert(v);
  std::unordered_map<std::uint64_t, VertexPtr> vmap;
  vmap.reserve(ids.size());
  for (auto id : ids) {
    if (vertexTimes) {
      if (id >= vertexTimes->size())
        throw std::out_of_range(
          "Spacetime::fromCells: vertexTimes too short to index vertex id "
          + std::to_string(id));
      vmap[id] = st->createVertex(id, {(*vertexTimes)[id]});
    } else {
      vmap[id] = st->createVertex(id);
    }
  }

  // One top simplex per cell; createSimplex auto-wires the edges, assigning
  // spacelike (+a) / timelike (-alpha*a) lengths from the vertex times under the
  // tracked rule, or all-spacelike when the vertices are coordinate-free.
  for (const auto &cell : cells) {
    std::vector<std::uint64_t> sortedIds(cell.begin(), cell.end());
    std::sort(sortedIds.begin(), sortedIds.end());
    VertexPtrs verts;
    verts.reserve(sortedIds.size());
    for (auto v : sortedIds) verts.push_back(vmap[v]);
    st->createSimplex(verts);
  }

  // Uniform Hermitian pin: overwrite every edge's geometry. Skipped under the
  // tracked-metric rule, where the auto-wired causal lengths ARE the geometry —
  // pinning there would overwrite every timelike edge with a spacelike unit
  // length and hand back a Euclidean complex in disguise (#644; the guard was
  // found commented out, contradicting both this comment and the causal-sign
  // tests).
  if (!vertexTimes) {
    for (const auto &edge : st->getEdgeList()->toVector()) {
      edge->setLength(std::sqrt(std::complex<double>{weight, 0.0}));
      edge->setPhase(phase);
    }
  }
  return st;
}

std::shared_ptr<Spacetime> Spacetime::subcomplexWithinVertexSet(
    const std::set<std::uint64_t> &vertexSet) const {
  std::vector<std::vector<std::uint64_t>> cellsInsideVertexSet;
  for (const auto &topSimplex : getTopSimplices()) {
    auto cellVertexIds = topSimplex->topTuple();
    bool cellIsInsideVertexSet = true;
    for (auto vertexId : cellVertexIds)
      if (!vertexSet.count(vertexId)) {
        cellIsInsideVertexSet = false;
        break;
      }
    if (cellIsInsideVertexSet)
      cellsInsideVertexSet.push_back(std::move(cellVertexIds));
  }
  if (cellsInsideVertexSet.empty()) return nullptr;
  return Spacetime::fromCells(getDimensions(), cellsInsideVertexSet, 1.0, 0.0);
}

int Spacetime::getDimensions() const noexcept {
  return metric->getSignature()->getDimensions();
}

std::vector<std::vector<std::uint64_t>> Spacetime::prismCells(
  const std::vector<std::vector<std::uint64_t>> &cells,
  int layers,
  const std::optional<std::unordered_map<std::uint64_t, std::uint64_t>> &twist
) {
  // Per-layer vertex stride: one past the largest base id. Each layer l offsets
  // its vertices by stride*l, so the layers occupy disjoint id ranges.
  std::uint64_t stride = 0;
  for (const auto &cell : cells)
    for (auto v : cell) stride = std::max(stride, v + 1);

  // The base permutation phi (identity unless a twist is supplied), indexed by
  // id. A missing twist key maps to itself.
  std::vector<std::uint64_t> phi1(stride);
  for (std::uint64_t v = 0; v < stride; ++v) {
    phi1[v] = v;
    if (twist) {
      auto it = twist->find(v);
      if (it != twist->end()) phi1[v] = it->second;
    }
  }

  // phi[ell] = phi1 composed with itself ell times (phi[0] = identity), so the
  // twist is applied cumulatively as the layers climb.
  std::vector<std::vector<std::uint64_t>> phi;
  phi.reserve(static_cast<std::size_t>(layers) + 1);
  std::vector<std::uint64_t> ident(stride);
  for (std::uint64_t v = 0; v < stride; ++v) ident[v] = v;
  phi.push_back(std::move(ident));
  for (int l = 0; l < layers; ++l) {
    const auto &prev = phi.back();
    std::vector<std::uint64_t> next(stride);
    for (std::uint64_t v = 0; v < stride; ++v) next[v] = phi1[prev[v]];
    phi.push_back(std::move(next));
  }

  std::set<std::vector<std::uint64_t>> out;
  for (int ell = 0; ell < layers; ++ell) {
    const auto &lo = phi[ell];
    const auto &hi = phi[ell + 1];
    const std::uint64_t loOff = stride * static_cast<std::uint64_t>(ell);
    const std::uint64_t hiOff = stride * static_cast<std::uint64_t>(ell + 1);
    for (const auto &cell : cells) {
      std::vector<std::uint64_t> base(cell.begin(), cell.end());
      std::sort(base.begin(), base.end());
      const std::size_t m = base.size();
      for (std::size_t j = 0; j < m; ++j) {
        std::vector<std::uint64_t> s;
        s.reserve(m + 1);
        for (std::size_t i = 0; i <= j; ++i) s.push_back(lo[base[i]] + loOff);
        for (std::size_t i = j; i < m; ++i) s.push_back(hi[base[i]] + hiOff);
        std::sort(s.begin(), s.end());
        out.insert(std::move(s));
      }
    }
  }
  return {out.begin(), out.end()};
}

std::vector<std::vector<std::uint64_t>> Spacetime::worldprismBoundaryFaces(
    const std::vector<std::uint64_t> &facet, std::uint64_t loOffset,
    std::uint64_t hiOffset) {
  // The staircase triangulation of the prism g x I over the (d-1)-simplex g
  // (facet, sorted, m vertices): the m d-simplices S_k = {lo[g_0..g_k]} u
  // {hi[g_k..g_{m-1}]}, k = 0..m-1. Its (d-1)-faces that appear in exactly ONE
  // staircase cell are the boundary of partial(g x I) (the caps g and g_top and
  // the side worldsheets); the shared (interior) face is the staircase diagonal.
  const std::size_t m = facet.size();
  std::vector<std::vector<std::uint64_t>> staircase;
  staircase.reserve(m);
  for (std::size_t k = 0; k < m; ++k) {
    std::vector<std::uint64_t> s;
    s.reserve(m + 1);
    for (std::size_t i = 0; i <= k; ++i) s.push_back(facet[i] + loOffset);
    for (std::size_t i = k; i < m; ++i) s.push_back(facet[i] + hiOffset);
    std::sort(s.begin(), s.end());
    staircase.push_back(std::move(s));
  }
  std::map<std::vector<std::uint64_t>, int> faceCount;
  for (const auto &s : staircase)
    for (std::size_t drop = 0; drop < s.size(); ++drop) {
      std::vector<std::uint64_t> f;
      f.reserve(s.size() - 1);
      for (std::size_t i = 0; i < s.size(); ++i)
        if (i != drop) f.push_back(s[i]);
      ++faceCount[f];  // f is already sorted (s is)
    }
  std::vector<std::vector<std::uint64_t>> boundary;
  for (auto &[f, n] : faceCount)
    if (n == 1) boundary.push_back(f);
  return boundary;
}

std::vector<std::vector<std::uint64_t>> Spacetime::symmetricStackCells(
    const std::vector<std::vector<std::uint64_t>> &baseCells, int nApexSlices) {
  // The symmetric APEX stacking (NOT a prism), dimension-generic via coface
  // mirroring: each top d-simplex t cones UP to a cell-apex f_t (up-cone t u {f_t})
  // and DOWN to the top copy (down-cone = the point reflection of the up-cone
  // through f_t). The gap over a (d-1)-facet g shared by two cofaces (apexes
  // f1, f2) is [f1,f2] * partial(g x I): the join of the canonical dual edge with
  // the boundary of the worldprism over g (worldprismBoundaryFaces). Its caps
  // reproduce the up/down reflection; its sides mirror across g's lower faces. In
  // d=2 the sides are worldlines, so this is EXACTLY the #413 octahedron split on
  // the dual edge (no diagonal); in d>=3 the side worldsheets take a globally
  // consistent staircase diagonal. nApexSlices reflect-and-cap layers stack into a
  // tall cobordism: primal layer ell holds v + ell*stride, apexes start at
  // (nApexSlices+1)*stride. nApexSlices = 1 is the single #413 reflection.
  if (nApexSlices < 1)
    throw std::runtime_error("symmetricStackCells: nApexSlices must be >= 1");

  std::uint64_t stride = 0;
  std::size_t cellSize = 0;
  for (const auto &c : baseCells) {
    cellSize = std::max(cellSize, c.size());
    for (auto v : c) stride = std::max(stride, v + 1);
  }
  if (cellSize < 3)
    throw std::runtime_error(
        "symmetricStackCells: base needs (d+1)-vertex cells with d >= 2");
  const int dim = static_cast<int>(cellSize) - 1;  // base manifold dimension
  (void)dim;  // documented; the per-facet loop derives the codimension structure

  // Dedup top d-simplices in first-appearance order (the apex indexing matches
  // the original #413 lexicographic-per-input order in d=2).
  std::vector<std::vector<std::uint64_t>> tops;
  std::map<std::vector<std::uint64_t>, std::size_t> topIndex;
  for (const auto &raw : baseCells) {
    if (raw.size() != cellSize) continue;  // skip lower-dimensional / malformed
    std::vector<std::uint64_t> t(raw.begin(), raw.end());
    std::sort(t.begin(), t.end());
    if (topIndex.emplace(t, tops.size()).second) tops.push_back(std::move(t));
  }
  const std::size_t nTops = tops.size();

  // Each (d-1)-facet -> the indices of the top simplices that share it.
  std::map<std::vector<std::uint64_t>, std::vector<std::size_t>> facetCofaces;
  for (std::size_t ti = 0; ti < nTops; ++ti) {
    const auto &t = tops[ti];
    for (std::size_t drop = 0; drop < t.size(); ++drop) {
      std::vector<std::uint64_t> g;
      g.reserve(t.size() - 1);
      for (std::size_t i = 0; i < t.size(); ++i)
        if (i != drop) g.push_back(t[i]);
      facetCofaces[g].push_back(ti);  // g already sorted
    }
  }

  const std::uint64_t apexBase =
      static_cast<std::uint64_t>(nApexSlices + 1) * stride;
  std::set<std::vector<std::uint64_t>> out;
  const auto add = [&out](std::vector<std::uint64_t> c) {
    std::sort(c.begin(), c.end());  // canonical storage key (carries no orientation)
    out.insert(std::move(c));
  };

  for (int j = 0; j < nApexSlices; ++j) {
    const std::uint64_t lo = static_cast<std::uint64_t>(j) * stride;
    const std::uint64_t hi = static_cast<std::uint64_t>(j + 1) * stride;
    const std::uint64_t apexSlice =
        apexBase + static_cast<std::uint64_t>(j) * nTops;

    for (std::size_t ti = 0; ti < nTops; ++ti) {
      const auto &t = tops[ti];
      const std::uint64_t f = apexSlice + ti;
      std::vector<std::uint64_t> up, dn;
      up.reserve(t.size() + 1);
      dn.reserve(t.size() + 1);
      for (auto v : t) { up.push_back(v + lo); dn.push_back(v + hi); }
      up.push_back(f);
      dn.push_back(f);
      add(std::move(up));   // up-cone (d,1)
      add(std::move(dn));   // down-cone (1,d) -- the point reflection through f
    }

    for (const auto &[g, cof] : facetCofaces) {
      if (cof.size() != 2) continue;  // hole-boundary facet: a tube wall
      const std::uint64_t f1 = apexSlice + cof[0], f2 = apexSlice + cof[1];
      for (auto s : worldprismBoundaryFaces(g, lo, hi)) {
        s.push_back(f1);
        s.push_back(f2);
        add(std::move(s));  // [f1,f2] * (a boundary face of the worldprism g x I)
      }
    }
  }
  return {out.begin(), out.end()};
}

// ========================================
// Query Methods
// ========================================
SpacetimeType Spacetime::getSpacetimeType() const noexcept {
  return spacetimeType;
}

double Spacetime::getCurrentTime() const noexcept {
  return static_cast<double>(currentTime);
}

std::complex<double> Spacetime::autoWiredLength(bool crossSlice) const noexcept {
  const double squaredMagnitude = crossSlice ? alpha * a : a;
  if (balancedEdgeWiring_) return balancedLength(squaredMagnitude);
  return crossSlice ? std::complex<double>{0.0, std::sqrt(squaredMagnitude)}
                    : std::complex<double>{std::sqrt(squaredMagnitude), 0.0};
}

std::uint64_t Spacetime::metricRevisionKey() const noexcept {
  std::uint64_t key = structuralRevision_;
  if (edgeList)
    for (const auto &edge : edgeList->toVector())
      key += edge->lengthRevision() + edge->phaseRevision();
  return key;
}

const std::shared_ptr<EdgeList> &Spacetime::getEdgeList() const noexcept {
  return edgeList;
}

Foliation Spacetime::getFoliation() const noexcept {
  return foliation;
}

const std::shared_ptr<Metric> &Spacetime::getMetric() const noexcept {
  return metric;
}

// ========================================
// Time-Slice & Spatial-Subgraph Queries
// ========================================

std::vector<int> Spacetime::getTimeSlices() const {
  std::set<int> times;
  for (auto *v : vertexList->liveVector())
    times.insert(static_cast<int>(v->getTime()));
  return {times.begin(), times.end()};
}

VertexPtrs Spacetime::getVerticesAtTime(int t) const {
  VertexPtrs result;
  for (auto *v : vertexList->liveVector())
    if (static_cast<int>(v->getTime()) == t)
      result.push_back(v);
  return result;
}

std::pair<VertexPtrs, Edges> Spacetime::getSpatialSubgraph(int t) const {
  auto verts = getVerticesAtTime(t);
  std::unordered_set<std::uint64_t> vidSet;
  for (auto *v : verts) vidSet.insert(v->getId());

  Edges spatial;
  std::unordered_set<std::uint64_t> seen;
  for (auto *v : verts) {
    for (const auto &e : v->getEdges()) {
      auto fp = e->fingerprint.fingerprint();
      if (seen.count(fp)) continue;
      seen.insert(fp);
      if (vidSet.count(e->getSource()->getId())
          && vidSet.count(e->getTarget()->getId())
          && e->isSpacelike())
        spatial.push_back(e);
    }
  }
  return {verts, spatial};
}

std::unordered_map<std::uint64_t, int>
Spacetime::bfsDistances(VertexPtr center, int maxDepth) const {
  // Build adjacency from spacelike edges at center's time slice
  int t = static_cast<int>(center->getTime());
  auto [verts, edges] = getSpatialSubgraph(t);

  std::unordered_map<std::uint64_t, std::vector<std::uint64_t>> adj;
  for (auto *v : verts) adj[v->getId()]; // ensure entry
  for (auto *e : edges) {
    auto s = e->getSource()->getId();
    auto tgt = e->getTarget()->getId();
    adj[s].push_back(tgt);
    adj[tgt].push_back(s);
  }

  std::unordered_map<std::uint64_t, int> dist;
  dist[center->getId()] = 0;
  std::queue<std::uint64_t> q;
  q.push(center->getId());
  while (!q.empty()) {
    auto vid = q.front(); q.pop();
    if (maxDepth >= 0 && dist[vid] >= maxDepth) continue;
    for (auto nbr : adj[vid]) {
      if (!dist.count(nbr)) {
        dist[nbr] = dist[vid] + 1;
        q.push(nbr);
      }
    }
  }
  return dist;
}

const std::shared_ptr<VertexList> &Spacetime::getVertexList() const noexcept {
  return vertexList;
}

std::size_t Spacetime::getTopVertexCount() const noexcept {
  // d+1, the vertex count of a top-dimensional simplex. Single source of truth
  // for "what is a top cell" (see registerSimplex / updateOrientationCounters /
  // getBoundary).
  return static_cast<std::size_t>(metric->getSignature()->getDimensions()) + 1;
}

std::vector<std::vector<std::uint64_t>> Spacetime::getBoundary() const {
  // Canonical boundary derivation: facet-counting from the top simplices.
  // A codimension-one face is on the boundary iff exactly one top simplex
  // contains it (an interior face is shared by two). This is computed purely
  // from the vertex sets, so it is side-effect-free and immune to the
  // lazy-facet problem that the coface-based ``getExternalSimplices`` has to
  // work around — no ``Simplex`` facet objects need to exist.

  // The top set is ``topSimplicesVec`` — the simplices ``registerSimplex``
  // keyed as top-dimensional, i.e. those whose vertex count equals
  // ``getTopVertexCount()`` (= signature d+1). This is exactly the set of top
  // cells *provided the signature dimension matches the triangulation's*
  // dimension; build fixtures with ``Signature(Topology::dimension(), …)`` so
  // it does. (For a complex of the wrong signature ``topSimplicesVec`` is
  // empty and the boundary comes back empty.)
  const std::size_t topVertexCount = getTopVertexCount();

  // Count, per codimension-one face, how many top simplices own it. Each top
  // simplex's codim-1 faces are obtained by dropping one vertex in turn from
  // its sorted vertex tuple (the remainder stays sorted, so the face is a
  // canonical key). The ``std::map`` keeps the result in sorted face order.
  std::map<std::vector<std::uint64_t>, int> incidence;
  std::vector<std::uint64_t> verts;
  for (const auto &simplex : topSimplicesVec) {
    verts.clear();
    verts.reserve(topVertexCount);
    for (const auto &v : simplex->getVertices()) verts.push_back(v->getId());
    std::sort(verts.begin(), verts.end());
    for (std::size_t omit = 0; omit < verts.size(); ++omit) {
      std::vector<std::uint64_t> face;
      face.reserve(verts.size() - 1);
      for (std::size_t i = 0; i < verts.size(); ++i)
        if (i != omit) face.push_back(verts[i]);
      ++incidence[face];
    }
  }

  std::vector<std::vector<std::uint64_t>> boundary;
  for (auto &[face, count] : incidence)
    if (count == 1) boundary.push_back(face);
  return boundary;
}

void Spacetime::materializeFacets() noexcept {
  // Facets are materialized lazily — ``Simplex::getFacets()`` creates them on
  // first access and registers each one back into ``simplicesVec`` (via
  // ``registerSimplex``). For CDT-built complexes this already happened during
  // gluing, so the loop below converges immediately. For complexes assembled
  // from scratch (e.g. a hand-built triangulation), the facets do not exist
  // yet, and materializing them *during* a range-for over ``simplicesVec``
  // would both invalidate the iterator (the vector grows) and read incomplete
  // coface counts (a shared facet looks like a boundary facet until its second
  // coface registers).
  //
  // ``getFacets()`` appends any newly-created facet simplices to
  // ``simplicesVec``; index iteration walks into them too, so a single pass
  // reaches a fixpoint (dimension strictly decreases, terminating at vertices
  // whose ``getFacets()`` is a no-op).
  //
  // Index iteration — re-reading ``simplicesVec[i]`` and ``size()`` each step —
  // is safe under the growth a range-for would not survive: reallocation moves
  // the buffer, but the next ``simplicesVec[i]`` reads the current buffer.
  // No copy of the vector is made. For CDT-built complexes the facets already
  // exist, so every ``getFacets()`` is a cached no-op and nothing is appended.
  for (std::size_t i = 0; i < simplicesVec.size(); ++i) {
    simplicesVec[i]->getFacets();
  }
}

void Spacetime::materializeFacets(const SimplexPtr &root) noexcept {
  // ``getFacets()`` hash-reuses a face that already exists (adding only the
  // missing coface link back to its parent) and creates-and-registers the
  // missing ones, so recursing from the root down to the vertices completes
  // the root's whole face lattice and its coface wiring. Shared sub-faces are
  // visited once per parent; every visit past the first is a cached no-op.
  if (root == nullptr || root->size() <= 1) return;
  for (const auto &facet : root->getFacets()) {
    materializeFacets(facet);
  }
}

SimplexSet Spacetime::getExternalSimplices() noexcept {
  // Boundary detection needs every facet's coface count to be complete: a facet
  // is on the boundary iff it has fewer than two cofaces. Force lazy facet
  // materialization to a fixpoint first so the coface counts are final.
  materializeFacets();

  // Cofaces are now fully linked; no further simplices will be created, so a
  // direct scan is safe.
  SimplexSet result{};
  for (const auto &simplex : simplicesVec) {
    if (simplex->hasBoundaryFacet()) result.insert(simplex);
  }
  return result;
}

std::vector<SimplexPtr> Spacetime::getSimplicesWithOrientation(std::tuple<uint8_t, uint8_t> orientation) const {
  TemporalOrientation o{std::get<0>(orientation), std::get<1>(orientation)};
  std::vector<SimplexPtr> result{};
  for (const auto &simplex : simplicesVec) {
    if (simplex->getOrientation() == o) result.push_back(simplex);
  }
  return result;
}

std::tuple<std::vector<std::uint32_t>,
           std::vector<std::uint32_t>,
           std::uint32_t>
Spacetime::getDualAdjacency() const {
  const std::uint32_t N = static_cast<std::uint32_t>(topSimplicesVec.size());

  // Map fingerprint → index in topSimplicesVec
  // (topSimplexVecIndex already exists but maps to pool slots; rebuild a clean one)
  auto fpToIdx = ::tessera::graph::indexByKey<std::uint32_t>(
      topSimplicesVec,
      [](auto const& s) { return s->fingerprint.fingerprint(); });

  std::vector<std::uint32_t> rows, cols;
  rows.reserve(N * 5);  // ~d+1 neighbours per simplex in d dimensions
  cols.reserve(N * 5);

  for (std::uint32_t i = 0; i < N; ++i) {
    auto const& simplex = topSimplicesVec[i];
    for (auto const& coface : ::tessera::graph::dualNeighbors(simplex)) {
      const auto fp = coface->fingerprint.fingerprint();
      auto it = fpToIdx.find(fp);
      if (it != fpToIdx.end()) {
        rows.push_back(it->second);
        cols.push_back(i);
      }
    }
  }

  return {std::move(rows), std::move(cols), N};
}

std::vector<VertexPtrs> Spacetime::getConnectedComponents() const {
  VertexPtrSet seen{};
  std::vector<VertexPtrs> components{};
  for (auto vertex : vertexList->toVector()) {
    if (seen.contains(vertex)) {
      continue;
    }
    VertexPtrs component{};
    VertexPtrs stack{vertex};
    while (!stack.empty()) {
      VertexPtr current = stack.back();
      stack.pop_back();
      if (seen.contains(current)) {
        continue;
      }
      seen.insert(current);
      component.push_back(current);
      for (const auto &edge : current->getOutEdges()) {
        VertexPtr neighbor = vertexList->get(edge->getTarget()->getId());
        if (neighbor != nullptr && !seen.contains(neighbor)) {
          stack.push_back(neighbor);
        }
      }
      for (const auto &edge : current->getInEdges()) {
        VertexPtr neighbor = vertexList->get(edge->getSource()->getId());
        if (neighbor != nullptr && !seen.contains(neighbor)) {
          stack.push_back(neighbor);
        }
      }
    }
    components.push_back(component);
  }
  return components;
}

SimplexPtr Spacetime::getSimplex(SimplexPtr simplex) const {
  if (simplex && simplex->vecIdx_ != UINT32_MAX) return simplex;
  return nullptr;
}

SimplexPtr Spacetime::getSimplex(std::uint64_t fingerprint) const {
  auto *s = simplexIndex_.find(fingerprint);
  if (!s) return nullptr;
  return *s;
}

// ========================================
// Manipulation & Helper Methods
// ========================================

double Spacetime::incrementTime() noexcept {
  currentTime++;
  return static_cast<double>(currentTime);
}

void Spacetime::swapVertexLabels(VertexPtr v1, VertexPtr v2) {
  if (v1 == v2 || v1->getId() == v2->getId()) return;

  auto id1 = v1->getId();
  auto id2 = v2->getId();

  // Collect simplices containing EXACTLY ONE of v1, v2 and record which
  // vertex they belong to BEFORE swapping IDs (hasVertex compares by ID).
  // Simplices containing both are unaffected (XOR fingerprint is symmetric).
  struct AffectedSimplex { SimplexPtr ptr; bool hadV1; };
  std::vector<AffectedSimplex> affected;

  for (const auto &s : v1->getSimplices()) {
    if (!s->hasVertex(v2))
      affected.push_back({s, true});
  }
  for (const auto &s : v2->getSimplices()) {
    if (!s->hasVertex(v1))
      affected.push_back({s, false});
  }

  // Edges incident to exactly one of v1, v2 change their vertex pair and so
  // change their lookup key. Collect and detach them BEFORE the ids move: the
  // key is derived from the endpoints, so it can only be computed while they
  // still hold the ids the edge was inserted under. Edges between v1 and v2
  // keep their pair and are left alone.
  //
  // Detaching every affected edge before reinserting any of them is what makes
  // a shared neighbour safe: two edges that exchange keys are both out of the
  // map before either goes back in.
  struct AffectedEdge { EdgePtr ptr; std::uint32_t slot; };
  std::vector<AffectedEdge> affectedEdges;
  for (const auto &e : v1->getEdges())
    if (!e->hasVertex(id2)) affectedEdges.push_back({e, UINT32_MAX});
  for (const auto &e : v2->getEdges())
    if (!e->hasVertex(id1)) affectedEdges.push_back({e, UINT32_MAX});

  for (auto &affectedEdge : affectedEdges)
    affectedEdge.slot = edgeList->detachEdge(EdgeList::keyOf(*affectedEdge.ptr));

  // Swap vertex IDs, rekey vertex list
  v1->setId(id2);
  v2->setId(id1);
  vertexList->swapKeys(id1, id2);

  // Rebuild each fingerprint from the endpoints rather than patching it by id.
  // A patched fingerprint drifts the moment one update is missed, and a drifted
  // fingerprint can never be found again.
  for (auto &affectedEdge : affectedEdges) {
    auto *e = affectedEdge.ptr;
    e->fingerprint.setIds({e->getSource()->getId(), e->getTarget()->getId()});
    e->fingerprint.refresh();
  }

  for (auto &affectedEdge : affectedEdges) {
    if (affectedEdge.slot == UINT32_MAX) continue;
    if (!edgeList->reattachEdge(affectedEdge.slot))
      throw std::runtime_error(
          "Spacetime::swapVertexLabels: two live edges claim the vertex pair (" +
          std::to_string(affectedEdge.ptr->getSource()->getId()) + ", " +
          std::to_string(affectedEdge.ptr->getTarget()->getId()) +
          ") after the swap.");
  }

  // Re-key simplexIndex_: erase old fingerprints, update, re-insert.
  // Only re-key simplices that are registered (vecIdx_ != UINT32_MAX).
  for (auto &[s, hadV1] : affected) {
    if (s->vecIdx_ != UINT32_MAX)
      simplexIndex_.erase(s->fingerprint.fingerprint());
  }
  for (auto &[s, hadV1] : affected) {
    if (hadV1) {
      s->fingerprint.removeId(id1);
      s->fingerprint.addId(id2);
    } else {
      s->fingerprint.removeId(id2);
      s->fingerprint.addId(id1);
    }
    s->fingerprint.refresh();
    if (s->vecIdx_ != UINT32_MAX)
      simplexIndex_.insert(s->fingerprint.fingerprint(), s);
  }
}

bool Spacetime::removeIfIsolated(const VertexPtr &vertex) noexcept {
  if (vertex->degree() == 0) {
    CLOG(DEBUG_LEVEL, "Removing vertex: ", vertex->toString());
    vertexList->remove(vertex);
    return true;
  }
  CLOG(DEBUG_LEVEL, "NOT Removing vertex: ", vertex->toString());
  return false;
}

void Spacetime::addObservable(const std::shared_ptr<Observable> &observable) {
  observables.push_back(observable);
}

// ========================================
// Internal Management
// ========================================

SimplexPtr Spacetime::registerSimplex(const SimplexPtr &simplex, bool internal) {
  auto fp = simplex->fingerprint.fingerprint();
  auto *existing = simplexIndex_.find(fp);
  if (existing) {
    return *existing;
  }

  // Storage ownership (poolSlot_) is established by the caller — either
  // createSimplex's emplace_back into simplexStorage_, or an external caller
  // who has already placed the Simplex in stable storage.  registerSimplex
  // is only responsible for the live-set bookkeeping below.

  simplex->vecIdx_ = static_cast<std::uint32_t>(simplicesVec.size());
  simplicesVec.push_back(simplex);
  simplexIndex_.insert(fp, simplex);
  ++structuralRevision_;                 // combinatorics changed (Betti cache keys on this)

  if (simplex->size() == getTopVertexCount()) {
    simplex->topVecIdx_ = static_cast<std::uint32_t>(topSimplicesVec.size());
    topSimplicesVec.push_back(simplex);
  }
  updateOrientationCounters(simplex, +1);
  // Mirror the simplex into each of its edges' simplex index so the
  // hot path in Vertex::removeOutEdge / removeInEdge can look up edge
  // cofaces directly instead of iterating every simplex incident to
  // the endpoint and filtering by hasVertex.
  for (auto const& e : simplex->getEdges()) {
    if (e != nullptr) e->registerSimplex(simplex);
  }
  return simplex;
}

void Spacetime::unregisterSimplex(const SimplexPtr &simplex) {
  if (simplex->isStale()) {
#ifdef TESSERA_ASSERTIONS
    CLOG(CRITICAL_LEVEL, "You attempted to unregister a simplex that does not exist!");
#endif
    return;
  }
  ++structuralRevision_;                 // combinatorics changed (Betti cache keys on this)
  auto vecIdx = simplex->vecIdx_;

  updateOrientationCounters(simplex, -1);

  // Drop this simplex from each of its edges' simplex index. Mirror of
  // the registerSimplex hook; must happen BEFORE the pool slot is freed
  // because we need to access ``simplex->getEdges()``.
  for (auto const& e : simplex->getEdges()) {
    if (e != nullptr) e->unregisterSimplex(simplex);
  }

  auto fp = simplex->fingerprint.fingerprint();

  // Swap-and-pop from simplicesVec (O(1) via stored index)
  if (vecIdx + 1 < static_cast<std::uint32_t>(simplicesVec.size())) {
    simplicesVec[vecIdx] = simplicesVec.back();
    simplicesVec[vecIdx]->vecIdx_ = vecIdx;
  }
  simplicesVec.pop_back();
  simplex->vecIdx_ = UINT32_MAX;
  simplexIndex_.erase(fp);

  // Swap-and-pop from topSimplicesVec (O(1) via stored index)
  auto topIdx = simplex->topVecIdx_;
  if (topIdx != UINT32_MAX) {
    if (topIdx + 1 < static_cast<std::uint32_t>(topSimplicesVec.size())) {
      topSimplicesVec[topIdx] = topSimplicesVec.back();
      topSimplicesVec[topIdx]->topVecIdx_ = topIdx;
    }
    topSimplicesVec.pop_back();
    simplex->topVecIdx_ = UINT32_MAX;
  }

  // The Simplex shell stays in simplexStorage_ at its stable address; releasing
  // the heap-allocated children reclaims the rest. vecIdx_ == UINT32_MAX is the
  // stale marker callers should already be checking against.
  simplex->releaseChildren();

  // Offer the slot back, but not yet: a Pachner move in flight captured
  // SimplexPtr in propose() and reads them in apply(), so handing this slot out
  // before the move finishes would let one of those pointers address a
  // different simplex. reclaimSimplexSlots, called at a sweep boundary, is
  // where it becomes available.
  // Externally-owned simplices have no slot here, and a slot already queued
  // must not be queued a second time: a removed simplex can be registered again
  // before the queue drains, and removed again after, which would otherwise
  // hand one slot to two simplices.
  if (simplex->poolSlot_ != UINT32_MAX && !simplex->pendingFree_) {
    simplex->pendingFree_ = true;
    pendingSimplexSlots_.push_back(simplex->poolSlot_);
  }
}

std::size_t Spacetime::reclaimSimplexSlots() noexcept {
  std::size_t released = 0;
  for (const std::uint32_t slot : pendingSimplexSlots_) {
    Simplex &simplex = simplexStorage_[slot];
    if (!simplex.pendingFree_) continue;
    simplex.pendingFree_ = false;
    // A simplex removed during the sweep may have been registered again since.
    // Its slot is live and must not be handed out.
    if (!simplex.isStale()) continue;
    freeSimplexSlots_.push_back(slot);
    ++released;
  }
  pendingSimplexSlots_.clear();
  return released;
}

void Spacetime::reserve(int nSimplices) {
  simplexIndex_.reserve(nSimplices);
  simplicesVec.reserve(nSimplices);
  topSimplicesVec.reserve(nSimplices);
  edgeList->reserve(nSimplices);
  vertexList->reserve(nSimplices);
}

// ========================================
// Counting & Access
// ========================================

std::size_t Spacetime::getSimplexCount() const noexcept {
  return n41Count + n32Count;
}

std::size_t Spacetime::getVertexCount() const noexcept {
  return vertexList->size();
}

std::size_t Spacetime::getN41() const noexcept {
  return n41Count;
}

std::size_t Spacetime::getN32() const noexcept {
  return n32Count;
}

const std::vector<SimplexPtr>& Spacetime::getSimplices() const noexcept {
  return simplicesVec;
}

const std::vector<SimplexPtr>& Spacetime::getTopSimplices() const noexcept {
  return topSimplicesVec;
}

VertexPtr Spacetime::getRandomVertex() { return getRandomVertex(rng); }

VertexPtr Spacetime::getRandomVertex(std::mt19937 &generator) {
  const auto &verts = vertexList->liveVector();
  if (verts.empty()) return nullptr;
  std::uniform_int_distribution<std::size_t> dist(0, verts.size() - 1);
  return verts[dist(generator)];
}

SimplexPtr Spacetime::getRandomSimplex() { return getRandomSimplex(rng); }

SimplexPtr Spacetime::getRandomSimplex(std::mt19937 &generator) {
  if (simplicesVec.empty()) return nullptr;
  std::uniform_int_distribution<std::size_t> dist(0, simplicesVec.size() - 1);
  return simplicesVec[dist(generator)];
}

SimplexPtr Spacetime::getRandomTopSimplex() { return getRandomTopSimplex(rng); }

SimplexPtr Spacetime::getRandomTopSimplex(std::mt19937 &generator) {
  if (topSimplicesVec.empty()) return nullptr;
  std::uniform_int_distribution<std::size_t> dist(0, topSimplicesVec.size() - 1);
  return topSimplicesVec[dist(generator)];
}

SimplexPtr Spacetime::findSimplexByVerts(
    const VertexPtrs &vertices) const noexcept {
  // Same hash formulation as createSimplex(verts, edges): commutative
  // XOR-mix over vertex IDs.  Order-independent, duplicate-safe.
  std::uint64_t hash = 0;
  for (const auto &v : vertices) {
    hash ^= Fingerprint::mix64(v->getId());
  }
  auto *found = simplexIndex_.find(hash);
  return found ? *found : nullptr;
}

SimplexPtr Spacetime::getRandomSimplexWithOrientation(uint8_t ti, uint8_t tf) {
  TemporalOrientation target{ti, tf};
  // Try random sampling first (fast if many match)
  for (int attempt = 0; attempt < 100; ++attempt) {
    auto s = getRandomSimplex();
    if (s && s->getOrientation() == target) return s;
  }
  // Fallback: linear scan and pick random from matches
  std::vector<SimplexPtr> matches;
  for (const auto &s : simplicesVec) {
    if (s->getOrientation() == target) matches.push_back(s);
  }
  if (matches.empty()) return nullptr;
  std::uniform_int_distribution<std::size_t> dist(0, matches.size() - 1);
  return matches[dist(rng)];
}

void Spacetime::updateOrientationCounters(const SimplexPtr &simplex, int delta) {
  auto d = metric->getSignature()->getDimensions();
  auto nVerts = simplex->size();
  // Only count top-dimensional simplices (d-simplices have d+1 vertices)
  if (nVerts != getTopVertexCount()) return;
  auto [ti, tf] = simplex->getOrientation().numeric();
  // (d, 1) or (1, d) type
  if ((ti == d && tf == 1) || (ti == 1 && tf == d)) {
    n41Count += delta;
  }
  // (d-1, 2) or (2, d-1) type
  else if ((ti == d - 1 && tf == 2) || (ti == 2 && tf == d - 1)) {
    n32Count += delta;
  }
}

::tessera::observables::SparseGraph Spacetime::getDualGraph() const {
  auto [rows, cols, n] = getDualAdjacency();
  return ::tessera::observables::SparseGraph::fromCOO(rows, cols, n);
}

namespace {

// Private SpectralGraph subclass used only by
// Spacetime::getSpectralDimensionOnSkeleton — holds the CSR of the
// weighted 1-skeleton of filtered top simplices and supplies the
// L = D - W matvec. Lives in an anonymous namespace because no other
// code paths currently consume the weighted-skeleton graph directly;
// promote to a public class when a second consumer appears.
class SkeletonSpectralView final : public SpectralGraph {
 public:
  SkeletonSpectralView(int n,
                       std::vector<int> indptr,
                       std::vector<int> indices,
                       std::vector<double> weights,
                       std::vector<double> degrees)
      : n_(n),
        indptr_(std::move(indptr)),
        indices_(std::move(indices)),
        weights_(std::move(weights)),
        degrees_(std::move(degrees)) {}

  int nVertices() const override { return n_; }

  void applyLaplacian(std::vector<double> const& x,
                        std::vector<double>& y) const override {
    y.assign(static_cast<std::size_t>(n_), 0.0);
    for (int i = 0; i < n_; ++i) {
      double s = degrees_[static_cast<std::size_t>(i)] *
                 x[static_cast<std::size_t>(i)];
      const int lo = indptr_[static_cast<std::size_t>(i)];
      const int hi = indptr_[static_cast<std::size_t>(i) + 1];
      for (int k = lo; k < hi; ++k) {
        s -= weights_[static_cast<std::size_t>(k)] *
             x[static_cast<std::size_t>(indices_[
                 static_cast<std::size_t>(k)])];
      }
      y[static_cast<std::size_t>(i)] = s;
    }
  }

 private:
  int n_;
  std::vector<int>    indptr_, indices_;
  std::vector<double> weights_, degrees_;
};

}  // anonymous namespace

std::vector<double>
Spacetime::getSpectralDimensionOnSkeleton(
    std::vector<double> const& sigmas,
    int krylovDim,
    SimplexFilter const& filter,
    int topK,
    int skeletonDim) const {
  if (skeletonDim != 1) {
    throw std::invalid_argument(
        "Spacetime::getSpectralDimensionOnSkeleton: skeletonDim != 1 "
        "is reserved for follow-up #36; only the 1-skeleton is "
        "supported in this build");
  }
  if (topK < 1) {
    throw std::invalid_argument(
        "Spacetime::getSpectralDimensionOnSkeleton: topK must be >= 1");
  }
  const std::uint64_t expectedVerts =
      static_cast<std::uint64_t>(topK) + 1;

  // Walk filtered top simplices; collect unique edges with MI-derived
  // weights. Same shape as the previous body of
  // InteractionSimulation::getSpectralDimension (pre-#31), now living
  // here so both pipelines share it.
  std::unordered_map<VertexPtr, int> idx;
  std::vector<std::tuple<int, int, double>> edgeList;
  std::set<std::pair<int, int>> seen;
  for (SimplexPtr s : simplicesVec) {
    if (s == nullptr) continue;
    if (s->size() != expectedVerts) continue;
    if (!filter.accept(s)) continue;
    for (EdgePtr e : s->getEdges()) {
      if (e == nullptr) continue;
      VertexPtr a = e->getSource();
      VertexPtr b = e->getTarget();
      if (a == nullptr || b == nullptr || a == b) continue;
      if (!idx.count(a)) idx[a] = static_cast<int>(idx.size());
      if (!idx.count(b)) idx[b] = static_cast<int>(idx.size());
      const int ia = idx.at(a);
      const int ib = idx.at(b);
      const auto key = std::minmax(ia, ib);
      if (!seen.insert({key.first, key.second}).second) continue;
      const double len = std::abs(e->getLength());  // |l| = sqrt(|l^2|)
      const double w   = ::tessera::observables::kIMax * std::exp(-len);
      edgeList.emplace_back(ia, ib, w);
    }
  }
  if (edgeList.empty()) {
    return std::vector<double>(sigmas.size(), 0.0);
  }

  // Build CSR (each undirected edge listed twice).
  const int n = static_cast<int>(idx.size());
  std::vector<int>    rows, cols;
  std::vector<double> ws;
  rows.reserve(edgeList.size() * 2);
  cols.reserve(edgeList.size() * 2);
  ws.reserve(edgeList.size() * 2);
  for (auto const& [u, v, w] : edgeList) {
    rows.push_back(u); cols.push_back(v); ws.push_back(w);
    rows.push_back(v); cols.push_back(u); ws.push_back(w);
  }
  std::vector<int>    indptr, indices;
  std::vector<double> weights;
  ::tessera::graph::buildCSRFromCOO<int, double, int>(
      static_cast<std::size_t>(n), rows, cols, ws,
      indptr, indices, weights);

  std::vector<double> degrees(static_cast<std::size_t>(n), 0.0);
  for (int v = 0; v < n; ++v) {
    const int lo = indptr[static_cast<std::size_t>(v)];
    const int hi = indptr[static_cast<std::size_t>(v) + 1];
    for (int k = lo; k < hi; ++k) {
      degrees[static_cast<std::size_t>(v)] +=
          weights[static_cast<std::size_t>(k)];
    }
  }

  SkeletonSpectralView view(n, std::move(indptr), std::move(indices),
                              std::move(weights), std::move(degrees));
  const std::vector<double> p = view.returnProbability(sigmas, krylovDim);
  return SpectralGraph::spectralDimension(sigmas, p);
}

double Spacetime::modularityOnSkeleton(int M) const {
  if (M < 1) return 0.0;
  std::size_t n = getVertexCount();
  if (n == 0) return 0.0;

  // Per-vertex degree and label.
  std::vector<std::uint32_t> deg(0);
  std::vector<int> label(0);
  // Build a vertex-id → contiguous-index map so we can index into
  // arrays of size n.  (Vertex IDs aren't necessarily dense.)
  std::unordered_map<std::uint64_t, std::uint32_t> idToIdx;
  idToIdx.reserve(n);
  std::vector<std::uint64_t> idsByIdx;
  idsByIdx.reserve(n);
  for (const auto &v : vertexList->toVector()) {
    auto id = v->getId();
    idToIdx.emplace(id, static_cast<std::uint32_t>(idsByIdx.size()));
    idsByIdx.push_back(id);
  }
  deg.assign(idsByIdx.size(), 0);
  label.assign(idsByIdx.size(), 0);
  for (std::size_t i = 0; i < idsByIdx.size(); ++i) {
    label[i] = static_cast<int>(idsByIdx[i] % static_cast<std::uint64_t>(M));
  }

  std::size_t m = edgeList->size();
  if (m == 0) return 0.0;

  // L_c (intra-community edge count) and D_c (sum of degrees) per
  // community label.
  std::unordered_map<int, double> L_c;
  std::unordered_map<int, double> D_c;
  for (const auto &e : edgeList->toVector()) {
    auto srcId = e->getSource()->getId();
    auto tgtId = e->getTarget()->getId();
    auto si = idToIdx[srcId];
    auto ti = idToIdx[tgtId];
    deg[si]++;
    deg[ti]++;
    if (label[si] == label[ti]) {
      // Each within-community edge contributes 2 to ``L_c`` in the
      // canonical formula (sum over ordered pairs).  We mirror
      // examples/modularity.py which uses
      // ``A[np.ix_(nodes, nodes)].sum()`` — that double-counts each
      // within-edge — so we add 2 here.
      L_c[label[si]] += 2.0;
    }
  }
  for (std::size_t i = 0; i < deg.size(); ++i) {
    D_c[label[i]] += deg[i];
  }

  double m2 = 2.0 * static_cast<double>(m);
  double Q = 0.0;
  for (const auto &[c, dc] : D_c) {
    double lc = 0.0;
    auto it = L_c.find(c);
    if (it != L_c.end()) lc = it->second;
    Q += lc / m2 - (dc / m2) * (dc / m2);
  }
  return Q;
}

void Spacetime::removeSimplex(const SimplexPtr &simplex) {
  // Defensive: double-remove (or remove of an already-stale simplex via a
  // dangling cache entry) is a silent no-op.  Without this, the getters
  // below would trip the stale-deref tripwire under TESSERA_ASSERTIONS,
  // whereas every caller's intent here is "if this simplex is already gone,
  // there's nothing to do."
  if (simplex->isStale()) return;
  // Remove from vertex simplex lists
  for (const auto &v : simplex->getVertices()) {
    v->removeSimplex(simplex);
  }
  // Remove coface references from facets
  if (simplex->hasFacets()) {
    for (const auto &facet : simplex->getFacets()) {
      facet->removeCoface(simplex);
    }
  }
  unregisterSimplex(simplex);
}

std::size_t Spacetime::pruneOrphanedSimplices() {
  const int topSize = getMetric()->getSignature()->getDimensions() + 1;
  // Snapshot first: removeSimplex mutates the simplex list (and the vertices'
  // incidence lists that hasTopCoface reads), so we must not iterate it live.
  std::vector<SimplexPtr> snapshot(getSimplices().begin(), getSimplices().end());
  std::size_t pruned = 0;
  for (const auto &s : snapshot) {
    if (s->isStale()) continue;
    if (static_cast<int>(s->size()) >= topSize) continue;  // keep top cells
    if (s->hasTopCoface()) continue;                       // genuine face
    removeSimplex(s);
    ++pruned;
  }
  return pruned;
}

std::size_t Spacetime::pruneOrphanedSimplices(
    const std::vector<std::uint64_t> &cellVertexIds) {
  // Enumerate the proper faces of the cell (every non-empty strict subset of
  // its vertex ids) and unregister each registered one that no current top
  // cell covers. Largest faces first, so removeSimplex cleans a face's coface
  // back-references while the face's own facets are still registered. The
  // subset lookup uses the same commutative id-hash as createSimplex /
  // findSimplexByVerts; the vertex-count check below keeps an (astronomically
  // unlikely) hash collision from unregistering an unrelated simplex.
  const std::size_t n = cellVertexIds.size();
  if (n < 2 || n > 16) return 0;
  std::size_t pruned = 0;
  for (std::size_t faceSize = n - 1; faceSize >= 1; --faceSize) {
    for (std::uint64_t mask = 1; mask < (std::uint64_t{1} << n); ++mask) {
      if (static_cast<std::size_t>(std::popcount(mask)) != faceSize) continue;
      std::uint64_t hash = 0;
      for (std::size_t i = 0; i < n; ++i) {
        if (mask & (std::uint64_t{1} << i)) {
          hash ^= Fingerprint::mix64(cellVertexIds[i]);
        }
      }
      auto *found = simplexIndex_.find(hash);
      if (!found) continue;
      SimplexPtr face = *found;
      if (face->isStale() || face->size() != faceSize) continue;
      if (face->hasTopCoface()) continue;  // covered by a surviving top cell
      removeSimplex(face);
      ++pruned;
    }
  }
  return pruned;
}

void Spacetime::removeEdge(const EdgePtr &edge) {
  if (edge == nullptr) return;
  if (auto src = edge->getSource()) src->removeOutEdge(edge);
  if (auto tgt = edge->getTarget()) tgt->removeInEdge(edge);
  absorbRemovedEdgeRevisions(edge);
  edgeList->remove(edge);
}

void Spacetime::removeVertex(const VertexPtr &vertex) {
  if (vertex == nullptr) return;
  // Snapshot the incidence list before mutating — removeEdge mutates it.
  Edges incident = vertex->getEdges();
  for (auto const &e : incident) removeEdge(e);
  vertexList->remove(vertex);
}

} // namespace tessera::spacetime
