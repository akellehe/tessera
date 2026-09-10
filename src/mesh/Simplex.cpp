// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "mesh/Vertex.h"
#include "mesh/Simplex.h"
#include "mesh/ForwardDeclarations.h"
#include "mesh/TemporalOrientation.h"
#include "spacetime/Spacetime.h"
#include "Logger.h"
#include "utils.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <numbers>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>

// === tessera subsystem ns fwd-decls ===
namespace tessera::graph {}
namespace tessera::observables {}
namespace tessera::quantum {}
namespace tessera::simulations {}
namespace tessera::spacetime {}
namespace tessera::mesh {
namespace {
/// Principal complex square root with a negative-zero imaginary part normalised
/// away first: std::sqrt lands on the far side of the branch cut for -0.0, and the
/// real-typed sign tests this replaces were immune to that where the complex form
/// is not (#638).
inline std::complex<double> principalSqrt(std::complex<double> z) {
    if (z.imag() == 0.0) z = {z.real(), 0.0};
    return std::sqrt(z);
}
}  // namespace
using namespace ::tessera::graph;
using namespace ::tessera::spacetime;
using namespace ::tessera::observables;
using namespace ::tessera::simulations;
using namespace ::tessera::quantum;

// Tripwire: catch dereferences of stale Simplex pointers.  Storage is
// stable, so reads from a logically-removed simplex won't fault — they'd
// just see empty child vectors and proceed silently.  This macro turns
// that silent failure mode into a loud abort under TESSERA_ASSERTIONS,
// while costing nothing in release builds.  Used at the top of the
// hot getters that callers might invoke through a cached SimplexPtr.
#ifdef TESSERA_ASSERTIONS
  #define TESSERA_TRIPWIRE_LIVE(method_name)                              \
    do {                                                                  \
      if (isStale()) {                                                    \
        CLOG(CRITICAL_LEVEL, "Stale Simplex* dereferenced via " method_name \
                             " — caller is holding a pointer to a "      \
                             "simplex that was already removed.");        \
        std::abort();                                                     \
      }                                                                   \
    } while (0)
#else
  #define TESSERA_TRIPWIRE_LIVE(method_name) ((void)0)
#endif


std::vector<std::uint64_t> Simplex::topTuple() const {
  std::vector<std::uint64_t> sortedVertexIdentifiers;
  for (const auto *vertex : getVertices())
    sortedVertexIdentifiers.push_back(vertex->getId());
  std::sort(sortedVertexIdentifiers.begin(), sortedVertexIdentifiers.end());
  return sortedVertexIdentifiers;
}

bool Simplex::hasFacets() const {
  TESSERA_TRIPWIRE_LIVE("hasFacets");
  return !facets.empty();
}

#ifdef TESSERA_ASSERTIONS
class SimplexCorruptionDetector : public CorruptionDetector<SimplexPtr, SimplexPtrHash, SimplexPtrEq> {
};
#endif

const std::vector<SimplexPtr> &Simplex::getFacets() {
  TESSERA_TRIPWIRE_LIVE("getFacets");
#if TESSERA_ASSERTIONS
  if (getVertices().empty()) throw std::runtime_error("Simplex is empty");
#endif
  if (getVertices().size() == 1) {
#if TESSERA_ASSERTIONS
    validate();
#endif
    return facets;
  }

  if (facets.empty()) {
    const auto &verts = vertices;  // Use member directly, avoid copy
    const std::size_t n = verts.size();
    const std::size_t facetSize = n - 1;

    facets.reserve(n);

    // CRITICAL OPTIMIZATION: Cache edges once before loop
    const auto &allEdges = getEdges();

    // Use this directly — no hash lookup needed.
    SimplexPtr coface = this;

    for (std::size_t skip = 0; skip < n; ++skip) {
      const auto skipVertexId = verts[skip]->getId();

      // Build faceVertices efficiently in one pass
      VertexPtrs faceVertices{};
      faceVertices.reserve(facetSize);
      for (std::size_t i = 0; i < n; ++i) {
        if (i != skip) faceVertices.push_back(verts[i]);
      }

      // Filter edges without the skipped vertex
      Edges faceEdges{};
      faceEdges.reserve(facetSize);  // Approximate size
      for (const auto &e : allEdges) {
        if (!e->hasVertex(skipVertexId)) faceEdges.push_back(e);
      }

      const auto &[facet, inserted] = spacetime->createSimplex(faceVertices, faceEdges); // Gets or creates!
      if (coface != nullptr && !facet->hasCoface(coface)) {
        facet->addCoface(coface);
      }
      facets.push_back(facet);
    }
  }
#if TESSERA_ASSERTIONS
  for (const auto &f : facets) {
    if (!isCofaceTo(f)) {
      CLOG(DEBUG_LEVEL, toString(), " is not a coface to ", f->toString());
      std::abort();
    }
  }
  validate();
#endif
  return facets;
}

///
/// @param vertices_
Simplex::Simplex(
  Spacetime *spacetime_,
  const VertexPtrs &vertices_,
  Edges edges_
) : spacetime(spacetime_), orientation(TemporalOrientation::orientationOf(vertices_)), vertices(vertices_),
    edges(std::move(edges_)),
    fingerprint({0}) {
#if TESSERA_ASSERTIONS
  if (vertices_.empty()) throw std::runtime_error("Simplex is empty");
#endif
}

Simplex::Simplex(
  Spacetime *spacetime_,
  const VertexPtrs &vertices_,
  Edges edges_,
  const TemporalOrientation &orientation_
) : spacetime(spacetime_), orientation(orientation_), vertices(vertices_), edges(std::move(edges_)),
    fingerprint() {
  for (const auto &v : vertices_) {
    fingerprint.addId(v->getId());
  }
  fingerprint.refresh();
#if TESSERA_ASSERTIONS
  if (vertices_.empty()) throw std::runtime_error("Simplex is empty");
#endif
}

Simplex* Simplex::create(Spacetime *spacetime_, const VertexPtrs &vertices_, const Edges &edges_) {
#if TESSERA_ASSERTIONS
  if (vertices_.empty()) throw std::runtime_error("Simplex is empty");
#endif
  Simplex* simplex = new Simplex(spacetime_, vertices_, edges_);
  if (!simplex->initialized) {
    simplex->initialize(simplex);
  }
  // registerToVertices() is called during initialize(); addSimplex() deduplicates.
  return simplex;
}

bool Simplex::isInitialized() const noexcept { return initialized; }

void Simplex::releaseChildren() noexcept {
#ifdef TESSERA_ASSERTIONS
  if (!isStale()) {
    CLOG(CRITICAL_LEVEL, "releaseChildren() called on a still-registered "
                         "simplex; caller must mark it stale first.");
    std::abort();
  }
#endif
  // swap-with-empty deallocates the underlying buffer (clear() alone would
  // keep capacity).  Order doesn't matter — none of these refer to each
  // other.
  VertexPtrs().swap(vertices);
  Edges().swap(edges);
  Simplices().swap(facets);
  Simplices().swap(cofaces);
  // The geometry cache is heap-backed too, and a removed simplex never refills
  // it, so it is dead weight for the rest of the run if it is only invalidated.
  geomCacheSlot_.release();
}

Simplex* Simplex::create(Spacetime *spacetime_,
                           const VertexPtrs &vertices_,
                           const Edges &edges_,
                           const TemporalOrientation &orientation_) {
#if TESSERA_ASSERTIONS
  if (vertices_.empty()) throw std::runtime_error("Simplex is empty");
#endif
  Simplex* simplex = new Simplex(spacetime_, vertices_, edges_, orientation_);
  simplex->initialize(simplex);
  return simplex;
}

void Simplex::initialize(Simplex* simplex) {
#ifdef TESSERA_ASSERTIONS
  if (simplex->initialized) {
    CLOG(DEBUG_LEVEL, "You attempted to re-initialize a simplex! Behavior is undefined.");
    std::abort();
  }
#endif
  std::vector<IdType> ids = {};
  ids.reserve(vertices.size());
  for (const auto &v : vertices) {
    ti = std::min(ti, v->getTime());
    tf = std::max(tf, v->getTime());
    ids.push_back(v->getId());
  }
  fingerprint.setIds(ids);
  _isSpatial = ti == tf;

  // We have to register AFTER the fingerprint is set:
  registerToVertices(simplex);
  initialized = true;
}

// getTi(), getTf() inlined in Simplex.h

void Simplex::registerToVertices(Simplex* simplex) {
  for (const auto &owner : simplex->getVertices()) {
    owner->addSimplex(simplex);
  }
}

#ifdef TESSERA_VERBOSE
std::string Simplex::toString() const noexcept {
  std::stringstream sigmaLabel;
  sigmaLabel << std::to_string(getOrientation().getK()) << "-";
  sigmaLabel << "\\sigma";

  std::stringstream orientationStr;
  orientationStr << "^{(" << std::to_string(std::get<0>(getOrientation().numeric())) << "/";
  orientationStr << std::to_string(std::get<1>(getOrientation().numeric())) << ")}";

  std::string fp = std::to_string(fingerprint.fingerprint());
  std::string fpShort = fp.size() >= 6
      ? fp.substr(0, 3) + fp.substr(fp.size() - 3)
      : fp;
  std::stringstream fpStr;
  fpStr << "_{" << fpShort << "}";

  std::stringstream vertexStr;
  std::vector<IdType> vids{};
  for (const auto &v : vertices) vids.push_back(v->getId());
  std::sort(vids.begin(), vids.end());
  for (const auto &v : vids) {
    vertexStr << std::to_string(v);
    if (v != vids[vids.size() - 1]) {
      vertexStr << "|";
    }
  }

  std::stringstream ss;
  ss << "<" << sigmaLabel.str() << orientationStr.str() << fpStr.str() << " " << vertexStr.str() << ">";
  return latexToUtf8(ss.str());
}
#endif

// getOrientation() inlined in Simplex.h

[[nodiscard]] const VertexPtrs &Simplex::getVertices() const noexcept {
  TESSERA_TRIPWIRE_LIVE("getVertices");
  return vertices;
}

// isSpatial() / isTimelike() inlined in Simplex.h (uses cached _isSpatial)

[[nodiscard]] std::size_t Simplex::computeNumberOfEdges(std::size_t k) {
  if (k == 4) return 6;
  if (k == 3) return 3;
  if (k == 2) return 1;
  if (k == 0 || k == 1) return 0;

  int n = 0;
  for (int i = 0; i < k; i++) {
    n = n + i;
  }
  return n;
}

template<typename T>
T Simplex::binomial(unsigned n, unsigned k) const {
  if (k > n) return 0;
  k = std::min(k, n - k);

  T result = 1;
  for (unsigned i = 1; i <= k; ++i) {
    result = result * (n - (k - i));
    result /= i;
  }

  return result;
}

std::size_t Simplex::getNumberOfFaces(std::size_t j) const {
  auto k = getOrientation().getK();
  return binomial<std::size_t>(k + 1, j + 1);
}

std::size_t Simplex::getNumberOfEdges() const {
  auto k = getOrientation().getK();
  return (k + 1) * k / 2;
}

void Simplex::addCoface(SimplexPtr coface) {
#if TESSERA_ASSERTIONS
  if (coface == nullptr) {
    CLOG(DEBUG_LEVEL, "Coface was null");
    std::abort();
  }
  if (!coface->isCofaceTo(this)) {
    CLOG(DEBUG_LEVEL, coface->toString(), " is not a coface of ", toString());
    throw std::runtime_error("You attempted to add a coface to a facet for which it is not a coface!");
  }
  if (hasCoface(coface)) {
    CLOG(DEBUG_LEVEL, "You attempted to add a duplicate coface: ", coface->toString(), " to simplex ", toString());
    std::abort();
  }
#endif
  // Cofaces are 0-2 elements; linear duplicate check is faster than hash table
  if (!hasCoface(coface)) {
    cofaces.push_back(coface);
  }
}

void Simplex::removeCoface(SimplexPtr coface) {
  auto fp = coface->fingerprint.fingerprint();
  for (auto it = cofaces.begin(); it != cofaces.end(); ++it) {
    if ((*it)->fingerprint.fingerprint() == fp) {
      // Swap-and-pop for O(1) removal
      *it = cofaces.back();
      cofaces.pop_back();
      return;
    }
  }
}

[[nodiscard]] bool Simplex::hasCoface(SimplexPtr coface) const {
  TESSERA_TRIPWIRE_LIVE("hasCoface");
  auto fp = coface->fingerprint.fingerprint();
  for (const auto &c : cofaces) {
    if (c->fingerprint.fingerprint() == fp) return true;
  }
  return false;
}

[[nodiscard]] bool Simplex::hasVertex(const VertexPtr &vertex) const {
  TESSERA_TRIPWIRE_LIVE("hasVertex");
  const auto id = vertex->getId();
  for (const auto &v : vertices)
    if (v->getId() == id) return true;
  return false;
}

[[nodiscard]] bool Simplex::hasEdgeContaining(const IdType vertexId) const {
  for (const auto &e : getEdges()) {
    if (e->getSource()->getId() == vertexId) return true;
    if (e->getTarget()->getId() == vertexId) return true;
  }
  return false;
}

void Simplex::validate() const {
#ifdef TESSERA_ASSERTIONS
  for (const auto &e : getEdges()) {
    if (!hasVertex(e->getSource())) {
      CLOG(ERROR_LEVEL, "Missing source for one of its edges: ", e->toString());
      throw std::runtime_error("Missing source for one of its edges.");
    }
    if (!hasVertex(e->getTarget())) {
      CLOG(ERROR_LEVEL, "Missing target for one of it's edges: ", e->toString());
      throw std::runtime_error("Missing target for one of its edges.");
    }
    if (getVertices().size() == 1) return; // A 0-simplex will have no edges.
    for (const auto &v : getVertices()) {
      if (!hasEdgeContaining(v->getId())) {
        CLOG(ERROR_LEVEL, "Missing an edge for vertex: ", v->toString(), " on simplex ", toString(), " with edges:");
        for (const auto &e2 : getEdges()) {
          CLOG(ERROR_LEVEL, "    - ", e2->toString());
        }
        throw std::runtime_error("Missing an edge for a vertex.");
      }
    }
  }
#endif
}

[[nodiscard]] const Edges &Simplex::getEdges() const {
  TESSERA_TRIPWIRE_LIVE("getEdges");
  return edges;
}

[[nodiscard]] bool Simplex::hasEdge(const EdgePtr &edge) const {
  if (!hasVertex(edge->getSource())) {
    return false;
  }
  if (!hasVertex(edge->getTarget())) {
    return false;
  }
  for (const auto &e : getEdges()) {
    if (e->getSource()->getId() == edge->getSource()->getId() && e->getTarget()->getId() == edge->getTarget()->
      getId()) {
      return true;
    }
  }
  return false;
}

[[nodiscard]] bool Simplex::hasEdge(const VertexPtr &vertexA, const VertexPtr &vertexB) const {
  if (!hasVertex(vertexA) || !hasVertex(vertexB)) return false;
  auto aId = vertexA->getId();
  auto bId = vertexB->getId();
  for (const auto &e : edges) {
    if ((e->getSource()->getId() == aId && e->getTarget()->getId() == bId) ||
        (e->getSource()->getId() == bId && e->getTarget()->getId() == aId))
      return true;
  }
  return false;
}

[[nodiscard]] const Simplices &
Simplex::getCofaces() const noexcept {
  TESSERA_TRIPWIRE_LIVE("getCofaces");
  return cofaces;
}

bool Simplex::isCofaceTo(const SimplexPtr &facet, bool shallow) const {
  if (shallow) {
    if (getOrientation().getK() != facet->getOrientation().getK() + 1) {
      return false;
    }
  }
  for (const auto &v : facet->getVertices()) {
    if (!hasVertex(v)) return false;
  }
  return true;
}

bool Simplex::operator==(const Simplex &other) const noexcept {
  return fingerprint.fingerprint() == other.fingerprint.fingerprint();
}

bool Simplex::operator==(const Simplex* other) const noexcept {
  return fingerprint.fingerprint() == other->fingerprint.fingerprint();
}

std::uint64_t Simplex::hash() const noexcept {
  return fingerprint.fingerprint();
}

bool Simplex::isBoundary() const noexcept {
  return cofaces.size() < 2;
}

bool Simplex::hasBoundaryFacet() {
  for (const auto &face : getFacets()) {
    if (face->isBoundary()) return true;
  }
  return false;
}

std::size_t Simplex::maxKPlusOneCofaces() const {
  return getNumberOfFaces(getOrientation().getK());
}

// size() inlined in Simplex.h

bool Simplex::replaceVertex(const VertexPtr &oldVertex, const VertexPtr &newVertex) {
  // TODO: Probably make this cascade, but we should just go to the Vertex for things to cascade to.
  if (hasVertex(newVertex)) {
#if TESSERA_ASSERTIONS
    validate();
#endif
    return false;
  }
  auto oldId = oldVertex->getId();
  std::size_t oldIndex = vertices.size(); // sentinel
  for (std::size_t i = 0; i < vertices.size(); ++i) {
    if (vertices[i]->getId() == oldId) { oldIndex = i; break; }
  }
  if (oldIndex == vertices.size()) return false;

  vertices[oldIndex] = newVertex;

  fingerprint.removeId(oldId);
  fingerprint.addId(newVertex->getId());

  // Clear cached facets/cofaces — they depend on the vertex set which just changed.
  facets.clear();
  cofaces.clear();

  return true;
}

VertexIdMap Simplex::getVertexIdLookup() const noexcept {
  VertexIdMap lookup{};
  for (const auto &v : vertices) {
    lookup.emplace(v->getId(), v);
  }
  return lookup;
}

bool Simplex::removeEdge(const EdgePtr &edge) {
  auto fp = edge->fingerprint.fingerprint();
  for (auto it = edges.begin(); it != edges.end(); ++it) {
    if ((*it)->fingerprint.fingerprint() == fp) {
      // Absorb the removed edge's revision (plus one) so the geometry
      // cache key — structural revision + Σ edge revisions — strictly
      // increases across the removal instead of falling back to a value
      // it held before (which could false-hit a stale cache section).
      structuralRevision_ += (*it)->lengthRevision() + 1;
      *it = edges.back();
      edges.pop_back();
      // Keep the Edge → Simplex index in sync. Without this the
      // edge's `simplices_` still claims this simplex as a member and
      // the next `Vertex::removeOutEdge` would dispatch into a
      // simplex that no longer contains the edge.
      edge->unregisterSimplex(this);
      return true;
    }
  }
  return false;
}

bool Simplex::addEdge(const EdgePtr &edge) {
  auto fp = edge->fingerprint.fingerprint();
  for (const auto &e : edges) {
    if (e->fingerprint.fingerprint() == fp) return false;
  }
  edges.push_back(edge);
  // The edge set changed, so every cached Gram/Cayley-Menger section is
  // stale; bumping the structural revision retires their keys.
  ++structuralRevision_;
  edge->registerSimplex(this);
  return true;
}

// updateVertexId / swapVertexIds are intentional no-ops (inlined in
// Simplex.h). The Simplex stores VertexPtrs and reads IDs through
// them; when Spacetime::swapVertexLabels rewrites a Vertex's ID, the
// Simplex sees the new ID automatically on its next ``getId()``.

bool Simplex::hasStoredFacet(const SimplexPtr &facet) const {
  if (facets.empty()) return false;
  for (const auto &f : facets) {
    if (f == facet) return true;
  }
  return false;
}

std::pair<SimplexPtr, Simplices> Simplex::cone(VertexPtr vertex) {
  auto signature = spacetime->getMetric()->getSignature();
  auto foliation = spacetime->getFoliation();
  if (signature->getSignatureType() == SignatureType::Lorentzian) {
    // We have to preserve causality. That means if we cone to e.g. a (1, 3) facet (one vertex at \f$ t \f$, 3 at
    // \f$ t+1 \f$) with a (1, 4) coface; then the new simplex has to be a (2, 3) simplex with (2, 3) - (1, 3) = (1, 0)
    // so we have to create a new vertex at time \f$ t \f$ rather than \f$ t+1 \f$ (which would have been the second
    // slot)
    // In general given a \f$ (n, m) \f$ simplex with a \f$ (n-1, m) \f$ or \f$ (n, m-1) \f$ facet; we have to match the
    // facet, but then what happens next depends on the foliation (preferred or not). If the foliation is preferred;
    // then we need a layer of timelike edges between every layer of spacelike edges. In order to ensure we only pair
    // compatible simplices; we just have to ensure the vertices stay balanced on either end of the spacelike sheet.
    //
    // If we have a e.g. a (3, 1) simplex with a (2, 1) facet, then we have (3, 1) - (2, 1) = (1, 0) = 1 extra vertex
    // at \f$ t \f$ . So we need to add the vertex with which we cone at \f$ t = t+1 \f$ to make the new coface a (2, 2)
    // simplex.
    if (foliation == Foliation::PREFERRED && !cofaces.empty()) {
      auto [facet_ti, facet_tf] = getOrientation().numeric();
      auto [coface_ti, coface_tf] = (*cofaces.begin())->getOrientation().numeric();
      if (coface_ti > facet_ti) {
        // Need an extra tf vertex.
        vertex->setTime(getTf());
      } else if (coface_tf > facet_tf) {
        // Need an extra ti vertex.
        vertex->setTime(getTi());
      }
    }
  }
  VertexPtrs kPlusOneVertices{vertices.begin(), vertices.end()};
  Edges newEdges{edges.begin(), edges.end()};
  for (auto &existing : kPlusOneVertices) {
    if (existing->getTime() == vertex->getTime()) {
      // Spacelike edge (same time slice): ℓ² = a
      newEdges.push_back(spacetime->createEdge(existing, vertex, std::sqrt(std::complex<double>(spacetime->getA()))));
    } else {
      // Timelike edge (different time slices): ℓ² = -α·a
      newEdges.push_back(spacetime->createEdge(existing, vertex, std::sqrt(std::complex<double>(-(spacetime->getAlpha() * spacetime->getA())))));
    }
  }
  kPlusOneVertices.push_back(vertex);
  auto [kSimplex, created] = spacetime->createSimplex(kPlusOneVertices, newEdges);
  Simplices newFacets{};
  auto myFingerprint = fingerprint.fingerprint();
  for (const auto &f : kSimplex->getFacets()) {
    if (f->fingerprint.fingerprint() != myFingerprint) {
      facets.push_back(f);
    }
  }
  return {kSimplex, facets};
}

// =====================================================================
// Geometry
// =====================================================================

std::complex<double> Simplex::determinant(const std::vector<std::complex<double>> &M, int n) {
    if (n == 1) return M[0];
    if (n == 2) return M[0] * M[3] - M[1] * M[2];
    std::vector<std::complex<double>> A(M);
    std::complex<double> det = 1.0;
    for (int col = 0; col < n; ++col) {
        int pivot = col;
        double maxVal = std::abs(A[col * n + col]);
        for (int row = col + 1; row < n; ++row) {
            double val = std::abs(A[row * n + col]);
            if (val > maxVal) { maxVal = val; pivot = row; }
        }
        if (maxVal < 1e-15) return 0.0;
        if (pivot != col) {
            for (int j = 0; j < n; ++j)
                std::swap(A[col * n + j], A[pivot * n + j]);
            det = -det;
        }
        det *= A[col * n + col];
        for (int row = col + 1; row < n; ++row) {
            std::complex<double> factor = A[row * n + col] / A[col * n + col];
            for (int j = col + 1; j < n; ++j)
                A[row * n + j] -= factor * A[col * n + j];
        }
    }
    return det;
}

std::vector<std::complex<double>> Simplex::cofactorMatrix(
    const std::vector<std::complex<double>> &M, int n) {
    std::vector<std::complex<double>> C(n * n, 0.0);
    if (n == 1) { C[0] = 1.0; return C; }
    std::vector<std::complex<double>> sub((n - 1) * (n - 1));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            int si = 0;
            for (int r = 0; r < n; ++r) {
                if (r == i) continue;
                int sj = 0;
                for (int c = 0; c < n; ++c) {
                    if (c == j) continue;
                    sub[si * (n - 1) + sj] = M[r * n + c];
                    sj++;
                }
                si++;
            }
            std::complex<double> sign = ((i + j) % 2 == 0) ? 1.0 : -1.0;
            C[i * n + j] = sign * determinant(sub, n - 1);
        }
    }
    return C;
}

std::vector<std::complex<double>> Simplex::localSquaredLengths(
    const VertexPtrs &ordering) const {
    // Flat (n x n) table of signed squared lengths by LOCAL index in
    // `ordering`: entry (i*n + j) is l^2 of the edge between ordering[i] and
    // ordering[j], 0 when the pair carries no edge — the same convention the
    // hashed per-entry lookups this replaces used (#672). One linear pass over
    // the edge list with direct id matching: no mix64 hashing, no
    // unordered_map, and immune to the (astronomically unlikely) XOR-pair
    // aliasing the hashed form admitted. Duplicate pairs keep the old
    // last-edge-wins order; a self-edge matches no (i, j) pair, exactly as a
    // zero-fingerprint entry was never read.
    const int n = static_cast<int>(ordering.size());
    std::vector<std::complex<double>> sq(static_cast<std::size_t>(n) * n,
                                         std::complex<double>{0.0, 0.0});
    for (const auto &e : edges) {
        const std::uint64_t sid = e->getSource()->getId();
        const std::uint64_t tid = e->getTarget()->getId();
        int si = -1, ti = -1;
        for (int k = 0; k < n; ++k) {
            const std::uint64_t vid = ordering[static_cast<std::size_t>(k)]->getId();
            if (vid == sid) si = k;
            else if (vid == tid) ti = k;
        }
        if (si >= 0 && ti >= 0) {
            const std::complex<double> l2 = e->getLength() * e->getLength();
            sq[static_cast<std::size_t>(si) * n + ti] = l2;
            sq[static_cast<std::size_t>(ti) * n + si] = l2;
        }
    }
    return sq;
}

std::vector<std::complex<double>> Simplex::gramMatrix() const {
    int dPlus1 = static_cast<int>(vertices.size());
    int d = dPlus1 - 1;
    if (d < 1) return {};

    // Squared-distance lookup on the honest signed l^2: a timelike edge keeps its
    // Lorentzian sign in G, so det(G) records the cell's metric signature. There is
    // no Wick-rotated (|l^2|) mode -- the Euclidean path is gone, not merely unused
    // (#641).
    const auto sq = localSquaredLengths(vertices);
    auto getSq = [&](int i, int j) -> std::complex<double> {
        return sq[static_cast<std::size_t>(i) * dPlus1 + j];
    };

    std::vector<std::complex<double>> G(d * d, std::complex<double>{0.0, 0.0});
    for (int i = 0; i < d; ++i)
        for (int j = 0; j < d; ++j)
            G[i * d + j] = 0.5 * (getSq(0, i + 1) + getSq(0, j + 1)
                                   - getSq(i + 1, j + 1));
    return G;
}

std::vector<std::complex<double>> Simplex::cayleyMengerMatrix() const {
    int dPlus1 = static_cast<int>(vertices.size());
    if (dPlus1 < 1) return {};

    const auto sq = localSquaredLengths(vertices);

    // Bordered matrix: zero corner, a border of ones, squared distances inside.
    int n = dPlus1 + 1;
    std::vector<std::complex<double>> B(n * n, 0.0);
    for (int k = 1; k < n; ++k) { B[k] = 1.0; B[k * n] = 1.0; }
    for (int i = 0; i < dPlus1; ++i)
        for (int j = 0; j < dPlus1; ++j)
            B[(i + 1) * n + (j + 1)] = sq[static_cast<std::size_t>(i) * dPlus1 + j];
    return B;
}

std::vector<std::complex<double>> Simplex::cayleyMengerCanonical(
    std::vector<std::pair<std::uint64_t, int>> &pos1) const {
    const int dPlus1 = static_cast<int>(vertices.size());
    pos1.clear();
    if (dPlus1 < 1) return {};

    // Canonical order: vertices sorted by ascending id (the reference orientation).
    std::vector<VertexPtr> sorted(vertices.begin(), vertices.end());
    std::sort(sorted.begin(), sorted.end(),
              [](const VertexPtr a, const VertexPtr b) {
                  return a->getId() < b->getId();
              });
    pos1.reserve(static_cast<std::size_t>(dPlus1));
    for (int i = 0; i < dPlus1; ++i)   // border-offset
        pos1.emplace_back(sorted[static_cast<std::size_t>(i)]->getId(), i + 1);

    const auto sq = localSquaredLengths(sorted);

    const int n = dPlus1 + 1;
    std::vector<std::complex<double>> B(static_cast<std::size_t>(n) * n, 0.0);
    for (int k = 1; k < n; ++k) { B[k] = 1.0; B[k * n] = 1.0; }
    for (int i = 0; i < dPlus1; ++i)
        for (int j = 0; j < dPlus1; ++j)
            B[(i + 1) * n + (j + 1)] = sq[static_cast<std::size_t>(i) * dPlus1 + j];
    return B;
}

std::uint64_t Simplex::geometryRevisionKey() const noexcept {
    std::uint64_t key = structuralRevision_;
    for (const auto &e : edges) key += e->lengthRevision();
    return key;
}

// Each section accessor is the same double-checked pattern: a lock-free hit
// when the section's published key equals the current geometry-revision key,
// else a mutex-serialized fill that publishes the key LAST (release), so a
// concurrent reader either sees the old key (and takes the mutex) or the new
// key with the payload already written. Lengths only mutate in the serial
// phases between parallel evaluations, so within a parallel region the key is
// constant and the returned reference stays valid.
const Simplex::GeomCache &Simplex::gramCache() const {
    const std::uint64_t key = geometryRevisionKey();
    if (geomCacheState_().gramKey.load(std::memory_order_acquire) != key) {
        std::lock_guard<std::mutex> lock(geomCacheState_().mutex);
        if (geomCacheState_().gramKey.load(std::memory_order_relaxed) != key)
            fillGramSection(key);
    }
    return geomCacheState_().cache;
}

const Simplex::GeomCache &Simplex::gramCofCache() const {
    const std::uint64_t key = geometryRevisionKey();
    if (geomCacheState_().gramCofKey.load(std::memory_order_acquire) != key) {
        std::lock_guard<std::mutex> lock(geomCacheState_().mutex);
        if (geomCacheState_().gramKey.load(std::memory_order_relaxed) != key)
            fillGramSection(key);
        if (geomCacheState_().gramCofKey.load(std::memory_order_relaxed) != key)
            fillGramCofSection(key);
    }
    return geomCacheState_().cache;
}

const Simplex::GeomCache &Simplex::cmCache() const {
    const std::uint64_t key = geometryRevisionKey();
    if (geomCacheState_().cmKey.load(std::memory_order_acquire) != key) {
        std::lock_guard<std::mutex> lock(geomCacheState_().mutex);
        if (geomCacheState_().cmKey.load(std::memory_order_relaxed) != key)
            fillCMSection(key);
    }
    return geomCacheState_().cache;
}

const Simplex::GeomCache &Simplex::cmCanonicalCache() const {
    const std::uint64_t key = geometryRevisionKey();
    if (geomCacheState_().cmCanonKey.load(std::memory_order_acquire) != key) {
        std::lock_guard<std::mutex> lock(geomCacheState_().mutex);
        if (geomCacheState_().cmCanonKey.load(std::memory_order_relaxed) != key)
            fillCMCanonSection(key);
    }
    return geomCacheState_().cache;
}

// The fills run the direct pipeline verbatim — same functions, same inputs —
// so cached values are bit-for-bit what an uncached call would produce.
void Simplex::fillGramSection(std::uint64_t key) const {
    const int d = static_cast<int>(vertices.size()) - 1;
    geomCacheState_().cache.gram = gramMatrix();
    geomCacheState_().cache.gramDet =
        (d >= 1 && static_cast<int>(geomCacheState_().cache.gram.size()) == d * d)
            ? determinant(geomCacheState_().cache.gram, d)
            : std::complex<double>{0.0, 0.0};
    geomCacheState_().gramKey.store(key, std::memory_order_release);
}

void Simplex::fillGramCofSection(std::uint64_t key) const {
    const int d = static_cast<int>(vertices.size()) - 1;
    geomCacheState_().cache.gramCof =
        (d >= 1 && static_cast<int>(geomCacheState_().cache.gram.size()) == d * d)
            ? cofactorMatrix(geomCacheState_().cache.gram, d)
            : std::vector<std::complex<double>>{};
    geomCacheState_().gramCofKey.store(key, std::memory_order_release);
}

void Simplex::fillCMSection(std::uint64_t key) const {
    const int n = static_cast<int>(vertices.size()) + 1;
    geomCacheState_().cache.cm = cayleyMengerMatrix();
    if (static_cast<int>(geomCacheState_().cache.cm.size()) == n * n) {
        geomCacheState_().cache.cmDet = determinant(geomCacheState_().cache.cm, n);
        geomCacheState_().cache.cmCof = cofactorMatrix(geomCacheState_().cache.cm, n);
    } else {
        geomCacheState_().cache.cmDet = {0.0, 0.0};
        geomCacheState_().cache.cmCof.clear();
    }
    geomCacheState_().cmKey.store(key, std::memory_order_release);
}

void Simplex::fillCMCanonSection(std::uint64_t key) const {
    const int n = static_cast<int>(vertices.size()) + 1;
    geomCacheState_().cache.cmCanon = cayleyMengerCanonical(geomCacheState_().cache.canonPos1);
    geomCacheState_().cache.cmCanonCof =
        (static_cast<int>(geomCacheState_().cache.cmCanon.size()) == n * n)
            ? cofactorMatrix(geomCacheState_().cache.cmCanon, n)
            : std::vector<std::complex<double>>{};
    geomCacheState_().cmCanonKey.store(key, std::memory_order_release);
}

std::complex<double> Simplex::dihedralAngle(SimplexPtr hinge) const {
    const int dPlus1 = static_cast<int>(vertices.size());
    // The two vertices of this simplex not in the hinge.
    const auto hingeVerts = hinge->getVertices();
    std::vector<int> opposite;
    for (int k = 0; k < dPlus1; ++k) {
        bool inHinge = false;
        for (const auto &hv : hingeVerts)
            if (hv->getId() == vertices[k]->getId()) { inHinge = true; break; }
        if (!inHinge) opposite.push_back(k);
    }
    if (opposite.size() != 2) return {0.0, 0.0};
    const int vi = opposite[0], vj = opposite[1];

    // Cayley-Menger cofactors -> the dihedral cosine ratio, UN-clamped:
    //
    //     cos(theta) = -C_ij / (sqrt(C_ii) * sqrt(C_jj))
    //
    // TWO separate principal square roots, never sqrt(C_ii * C_jj). For complex
    // a, b the two differ by a sign exactly when both sit on the negative real
    // axis -- with the unit tetrahedron's C_ii = C_jj = -3, sqrt(C_ii*C_jj) is
    // +3 while sqrt(C_ii)*sqrt(C_jj) is (i*r3)(i*r3) = -3. Folding the product
    // under one root is what used to force a hand-applied (-1)^d parity fix, a
    // three-way branch dispatch, and an i<->j anchoring swap; taking the roots
    // separately makes all three emerge from the branch structure instead (#638).
    //
    // Every causal configuration is this one expression. Same-sign cofactors put
    // the wedge on one side of the light cone: a real angle for |r| <= 1, a boost
    // (pure-imaginary acos) for |r| > 1. Opposite signs mean the wedge CROSSES the
    // cone -- the denominator turns pure-imaginary, r = i*y, and the principal
    // acos(i*y) = pi/2 - i*asinh(y) reproduces Sorkin's quarter turn (#581) with
    // no special case. Around a flat one-ray-per-quadrant vertex star the boosts
    // telescope to zero and four crossings sum to 2*pi, so 2*pi - sum = 0 holds.
    //
    // Evaluate in the canonical (sorted-by-id) frame so a cell a Pachner move
    // stored in causal order yields the same deficit as the same geometry built
    // sorted -- otherwise the action depends on build history.
    const int n = dPlus1 + 1;
    // Cached canonical frame (#668): the sorted-by-id Cayley-Menger matrix and
    // its cofactors are hinge-independent, so every hinge of this cell reads
    // the same fill instead of recomputing the O(n^5) cofactor pass per call.
    const GeomCache &cc = cmCanonicalCache();
    const auto &cof = cc.cmCanonCof;
    if (static_cast<int>(cof.size()) != n * n) return {0.0, 0.0};
    const int bi = canonicalPosition(cc.canonPos1, vertices[vi]->getId());
    const int bj = canonicalPosition(cc.canonPos1, vertices[vj]->getId());
    if (bi == 0 || bj == 0) return {0.0, 0.0};
    const std::complex<double> Cij = cof[static_cast<std::size_t>(bi) * n + bj];
    const std::complex<double> Cii = cof[static_cast<std::size_t>(bi) * n + bi];
    const std::complex<double> Cjj = cof[static_cast<std::size_t>(bj) * n + bj];
    const std::complex<double> denom = principalSqrt(Cii) * principalSqrt(Cjj);
    if (std::abs(denom) < 1e-15) return {0.0, 0.0};
    std::complex<double> r = -Cij / denom;
    // acos is cut on (-inf,-1] and [1,inf), so for a REAL ratio with |r| > 1 --
    // the same-sign (boost) wedge -- the sign of Im(theta) is decided by which
    // side of the cut the argument sits on, i.e. by the sign of its zero
    // imaginary part. Complex division leaves that to floating-point accident,
    // so it is pinned here instead: +0.0, the side the real-typed
    // acos(complex(r, 0.0)) took. The boost ORIENTATION is not determined by
    // edge lengths alone (a PT reflection flips it at identical l^2), so this is
    // a convention -- but it must be a stated one, not an emergent rounding.
    if (r.imag() == 0.0) r = {r.real(), 0.0};
    return std::acos(r);
}

std::complex<double> Simplex::deficitAngle() const {
    using cd = std::complex<double>;
    const cd twoPi(2.0 * std::numbers::pi, 0.0);
    if (!spacetime || vertices.empty()) return twoPi;
    const int topSize =
        spacetime->getMetric()->getSignature()->getDimensions() + 1;
    (void)topSize;
    cd sum(0.0, 0.0);
    for (auto *sigma : incidentTopCells())
        sum += sigma->dihedralAngle(const_cast<Simplex *>(this));
    return twoPi - sum;
}

std::map<std::pair<std::uint64_t, std::uint64_t>, std::complex<double>>
Simplex::deficitAngleGradient() const {
    using cd = std::complex<double>;
    std::map<std::pair<std::uint64_t, std::uint64_t>, cd> grad;
    if (!spacetime || vertices.empty()) return grad;
    const int topSize =
        spacetime->getMetric()->getSignature()->getDimensions() + 1;

    // The top cells containing this hinge -- the same set deficitAngle
    // sums over. d(eps)/dl^2 = -sum_tau d(theta_tau)/dl^2.
    (void)topSize;
    for (auto *tau : incidentTopCells()) {
        const auto &tv = tau->getVertices();
        const int m = static_cast<int>(tv.size());          // d + 1
        // local indices of the two vertices NOT in the hinge
        std::vector<int> opp;
        for (int k = 0; k < m; ++k) {
            bool inHinge = false;
            for (const auto &hv : vertices)
                if (hv->getId() == tv[k]->getId()) { inHinge = true; break; }
            if (!inHinge) opp.push_back(k);
        }
        if (opp.size() != 2) continue;
        const int bi = opp[0] + 1, bj = opp[1] + 1;          // CM border offset

        const int n = m + 1;                                 // CM is (d+2)x(d+2)
        // Cached raw-order Cayley-Menger pipeline (#668): shared by every
        // hinge of tau and by the Hessian below.
        const GeomCache &tc = tau->cmCache();
        const std::complex<double> detB = tc.cmDet;
        if (std::abs(detB) < 1e-300) continue;
        const std::vector<std::complex<double>> &C = tc.cmCof;
        if (static_cast<int>(C.size()) != n * n) continue;
        // B^-1 = adj(B)/det = cof^T/det ; B symmetric => Binv symmetric.
        std::vector<std::complex<double>> Binv(static_cast<std::size_t>(n) * n);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                Binv[i * n + j] = C[j * n + i] / detB;

        const cd Cij = C[bi * n + bj];
        const cd Cii = C[bi * n + bi];
        const cd Cjj = C[bj * n + bj];
        // Same unified branch as dihedralAngle (#638): two separate
        // principal roots, one expression for every causal regime. The sC/sP
        // sign flags and the crossing/non-crossing dispatch this replaces were
        // artifacts of folding the product under one root.
        const cd denom = principalSqrt(Cii) * principalSqrt(Cjj);
        if (std::abs(denom) < 1e-300) continue;
        cd r = -Cij / denom;
        // Pin the branch side exactly as the value does: for a real ratio with
        // |r| > 1 the sign of Im(theta) is decided by the sign of the zero
        // imaginary part, and the derivative must sit on the SAME sheet as the
        // value or it disagrees with a finite difference of it.
        if (r.imag() == 0.0) r = {r.real(), 0.0};
        const cd theta = std::acos(r);
        const cd sinTheta = std::sin(theta);
        if (std::abs(sinTheta) < 1e-300) continue;       // flat/folded: skip
        const cd dthetaDr = cd(-1.0, 0.0) / sinTheta;

        // dC_pq for the edge (a,b): dB is the indicator at (a+1,b+1)&(b+1,a+1);
        // dC = det[ tr(B^-1 dB) B^-1 - B^-1 dB B^-1 ], extracted entrywise.
        auto dCof = [&](int p, int q, int a, int b) -> cd {
            const cd bab = Binv[(a + 1) * n + (b + 1)];
            return detB * (2.0 * bab * Binv[p * n + q]
                           - (Binv[p * n + (a + 1)] * Binv[(b + 1) * n + q]
                              + Binv[p * n + (b + 1)] * Binv[(a + 1) * n + q]));
        };
        for (int a = 0; a < m; ++a) {
            for (int b = a + 1; b < m; ++b) {
                const cd dCij = dCof(bi, bj, a, b);
                const cd dCii = dCof(bi, bi, a, b);
                const cd dCjj = dCof(bj, bj, a, b);
                // d(sqrt(Cii)*sqrt(Cjj)) = denom * (dCii/Cii + dCjj/Cjj) / 2 --
                // branch-free, and valid on both sides of a light-cone crossing
                // without a sign flag.
                const cd ddenom = denom * 0.5 * (dCii / Cii + dCjj / Cjj);
                const cd dr = -(dCij * denom - Cij * ddenom) / (denom * denom);
                const std::uint64_t va = tv[a]->getId(), vb = tv[b]->getId();
                grad[{std::min(va, vb), std::max(va, vb)}] -= dthetaDr * dr;
            }
        }
    }
    return grad;
}

std::map<std::pair<std::pair<std::uint64_t, std::uint64_t>,
                   std::pair<std::uint64_t, std::uint64_t>>,
         std::complex<double>>
Simplex::deficitAngleHessian() const {
    using cd = std::complex<double>;
    using EK = std::pair<std::uint64_t, std::uint64_t>;
    std::map<std::pair<EK, EK>, cd> hess;
    if (!spacetime || vertices.empty()) return hess;
    const int topSize =
        spacetime->getMetric()->getSignature()->getDimensions() + 1;

    // d^2(eps)/dl^2_e dl^2_f = -sum_tau d^2(theta_tau). Same top-cell set and
    // cofactor machinery as deficitAngleGradient, carried one more
    // derivative: d^2 theta = (d2theta/dr^2) dr_e dr_f + (dtheta/dr) d2r.
    for (const auto &tau : vertices[0]->getSimplices()) {
        if (static_cast<int>(tau->size()) != topSize) continue;
        bool containsAll = true;
        for (std::size_t i = 1; i < vertices.size(); ++i)
            if (!tau->hasVertex(vertices[i])) { containsAll = false; break; }
        if (!containsAll) continue;

        const auto &tv = tau->getVertices();
        const int m = static_cast<int>(tv.size());
        std::vector<int> opp;
        for (int k = 0; k < m; ++k) {
            bool inHinge = false;
            for (const auto &hv : vertices)
                if (hv->getId() == tv[k]->getId()) { inHinge = true; break; }
            if (!inHinge) opp.push_back(k);
        }
        if (opp.size() != 2) continue;
        const int bi = opp[0] + 1, bj = opp[1] + 1;

        const int n = m + 1;
        // Same cached raw-order Cayley-Menger section as the gradient (#668).
        const GeomCache &tc = tau->cmCache();
        const std::complex<double> detB = tc.cmDet;
        if (std::abs(detB) < 1e-300) continue;
        const std::vector<std::complex<double>> &C = tc.cmCof;
        if (static_cast<int>(C.size()) != n * n) continue;
        std::vector<std::complex<double>> Binv(static_cast<std::size_t>(n) * n);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                Binv[i * n + j] = C[j * n + i] / detB;

        const cd Cij = C[bi * n + bj];
        const cd Cii = C[bi * n + bi];
        const cd Cjj = C[bj * n + bj];
        // One branch for every causal regime, as in the value and the gradient
        // (#638). theta = acos(r), r = -Cij/(sqrt(Cii)*sqrt(Cjj)), so
        // dtheta/dr = -1/sin(theta) and d2theta/dr2 = -r/sin^3(theta).
        const cd denom = principalSqrt(Cii) * principalSqrt(Cjj);
        if (std::abs(denom) < 1e-300) continue;
        cd r = -Cij / denom;
        // Pin the branch side exactly as the value does: for a real ratio with
        // |r| > 1 the sign of Im(theta) is decided by the sign of the zero
        // imaginary part, and the derivative must sit on the SAME sheet as the
        // value or it disagrees with a finite difference of it.
        if (r.imag() == 0.0) r = {r.real(), 0.0};
        const cd theta = std::acos(r);
        const cd sinT = std::sin(theta);
        if (std::abs(sinT) < 1e-300) continue;
        const cd dthetaDr = cd(-1.0, 0.0) / sinT;
        const cd d2thetaDr2 = -r / (sinT * sinT * sinT);

        auto bb = [&](int x, int y) -> cd { return Binv[x * n + y]; };
        // dC_pq/dl^2_(a,b): a,b local vertex indices (CM border = +1).
        auto dCof = [&](int p, int q, int a, int b) -> cd {
            const int A = a + 1, Bn = b + 1;
            return detB * (2.0 * bb(A, Bn) * bb(p, q)
                           - bb(p, A) * bb(Bn, q) - bb(p, Bn) * bb(A, q));
        };
        // dBinv_xy/dl^2_(c,d) = -(Binv_xC Binv_Dy + Binv_xD Binv_Cy).
        auto dBi = [&](int x, int y, int c, int d) -> cd {
            const int Cn = c + 1, Dn = d + 1;
            return -(bb(x, Cn) * bb(Dn, y) + bb(x, Dn) * bb(Cn, y));
        };
        // d^2 C_pq/dl^2_(a,b) dl^2_(c,d) = ddetB*T + detB*dT.
        auto d2Cof = [&](int p, int q, int a, int b, int c, int d) -> cd {
            const int A = a + 1, Bn = b + 1, Cn = c + 1, Dn = d + 1;
            const cd T = 2.0 * bb(A, Bn) * bb(p, q)
                             - bb(p, A) * bb(Bn, q) - bb(p, Bn) * bb(A, q);
            const cd ddetB = detB * 2.0 * bb(Cn, Dn);
            const cd dT =
                2.0 * (dBi(A, Bn, c, d) * bb(p, q) + bb(A, Bn) * dBi(p, q, c, d))
                - (dBi(p, A, c, d) * bb(Bn, q) + bb(p, A) * dBi(Bn, q, c, d))
                - (dBi(p, Bn, c, d) * bb(A, q) + bb(p, Bn) * dBi(A, q, c, d));
            return ddetB * T + detB * dT;
        };

        struct Loc { int a, b; std::uint64_t va, vb; cd dr; };
        std::vector<Loc> es;
        for (int a = 0; a < m; ++a)
            for (int b = a + 1; b < m; ++b) {
                const cd dCij = dCof(bi, bj, a, b);
                const cd dCii = dCof(bi, bi, a, b);
                const cd dCjj = dCof(bj, bj, a, b);
                const cd ddenom = denom * 0.5 * (dCii / Cii + dCjj / Cjj);
                const cd dr = -(dCij * denom - Cij * ddenom) / (denom * denom);
                es.push_back({a, b, tv[a]->getId(), tv[b]->getId(), dr});
            }

        for (const auto &e : es) {
            const cd dCij_e = dCof(bi, bj, e.a, e.b);
            const cd dCii_e = dCof(bi, bi, e.a, e.b);
            const cd dCjj_e = dCof(bj, bj, e.a, e.b);
            // denom = exp((ln Cii + ln Cjj)/2), so d denom = denom * L' and
            // d2 denom = denom * (L'_e L'_f + L''_ef) with
            // L' = (dCii/Cii + dCjj/Cjj)/2. Branch-free, no sign flags.
            const cd Le = 0.5 * (dCii_e / Cii + dCjj_e / Cjj);
            const cd ddenom_e = denom * Le;
            for (const auto &f : es) {
                const cd dCij_f = dCof(bi, bj, f.a, f.b);
                const cd dCii_f = dCof(bi, bi, f.a, f.b);
                const cd dCjj_f = dCof(bj, bj, f.a, f.b);
                const cd Lf = 0.5 * (dCii_f / Cii + dCjj_f / Cjj);
                const cd ddenom_f = denom * Lf;

                const cd d2Cij = d2Cof(bi, bj, e.a, e.b, f.a, f.b);
                const cd d2Cii = d2Cof(bi, bi, e.a, e.b, f.a, f.b);
                const cd d2Cjj = d2Cof(bj, bj, e.a, e.b, f.a, f.b);
                const cd Lef = 0.5 * (d2Cii / Cii - dCii_e * dCii_f / (Cii * Cii)
                                      + d2Cjj / Cjj - dCjj_e * dCjj_f / (Cjj * Cjj));
                const cd d2denom = denom * (Le * Lf + Lef);

                // Quotient rule, second order, on r = N/Den with N = -Cij.
                const cd N = -Cij, Ne = -dCij_e, Nf = -dCij_f, Nef = -d2Cij;
                const cd Den = denom, De = ddenom_e, Df = ddenom_f, Def = d2denom;
                const cd d2r =
                    ((Nef * Den + Ne * Df - Nf * De - N * Def) * Den
                     - 2.0 * (Ne * Den - N * De) * Df) / (Den * Den * Den);

                const cd d2theta = d2thetaDr2 * e.dr * f.dr + dthetaDr * d2r;
                const EK ke{std::min(e.va, e.vb), std::max(e.va, e.vb)};
                const EK kf{std::min(f.va, f.vb), std::max(f.va, f.vb)};
                hess[{ke, kf}] -= d2theta;      // eps = 2pi - sum theta
            }
        }
    }
    return hess;
}

std::complex<double> Simplex::area() const {
    if (edges.size() < 3) return {0.0, 0.0};
    auto sq = [&](std::size_t k) { return (edges[k]->getLength() * edges[k]->getLength()); };
    const std::complex<double> a2 = sq(0), b2 = sq(1), c2 = sq(2);
    const std::complex<double> val = 2.0 * (a2 * b2 + b2 * c2 + c2 * a2)
                                     - (a2 * a2 + b2 * b2 + c2 * c2);
    // Heron's radicand under a COMPLEX root. The old real path clamped a
    // non-positive radicand to 0, which silently reported zero area for every
    // timelike triangle (the mixed-causal hinge of a CDT (4,1) cell, among
    // others). Zero was never their area; it was what a double could represent.
    return std::sqrt(val) / 4.0;
}

std::complex<double> Simplex::volume() const {
    int d = static_cast<int>(vertices.size()) - 1;
    if (d < 1) return {0.0, 0.0};

    // Honest, signature-respecting Gram matrix: timelike edges keep l^2 < 0,
    // so det(G) can be negative for a Lorentzian cell. Cached (#668): volume()
    // is evaluated once per facet per dual-volume recursion step.
    const GeomCache &gc = gramCache();
    if (static_cast<int>(gc.gram.size()) != d * d) return {0.0, 0.0};

    const std::complex<double> detG = gc.gramDet;
    double factorial = 1.0;
    for (int i = 2; i <= d; ++i) factorial *= static_cast<double>(i);

    // V = sqrt(det G)/d!, principal branch. The old real path took
    // sqrt(|det G|) and hand-restored sign(det G) -- the same artifact as the
    // dihedral parity fix (#638): folding the magnitude under the root discards
    // a sign the complex root carries by itself. A Lorentzian cell with
    // det G < 0 therefore returns an IMAGINARY content, which is what its
    // d-content is, rather than the negative real a double could hold.
    return std::sqrt(detG) / factorial;
}

void Simplex::assertSpacelikeAdmissible(double tol) const {
    const int n = static_cast<int>(size());  // vertices = d + 1
    if (n < 2) return;                        // trivially admissible
    const int d = n - 1;

    // Skip simplices that contain any non-spacelike (null/timelike/worldline)
    // edge: their admissibility is Lorentzian, not the spacelike triangle
    // inequalities. Causal character is the canonical Edge classification
    // (Edge::isSpacelike, Im of the complex length), not a hand-rolled
    // sign-of-l^2 test (#581).
    for (const auto &e : edges)
        if (!e->isSpacelike()) return;

    // All edges spacelike: the Gram matrix must be positive-definite. Check via
    // Sylvester's criterion (every leading principal minor > 0) so the test
    // stays Eigen-free, reusing the existing determinant helper.
    const std::vector<std::complex<double>> g = gramMatrix();
    for (int k = 1; k <= d; ++k) {
        std::vector<std::complex<double>> sub(static_cast<std::size_t>(k) * k);
        for (int i = 0; i < k; ++i)
            for (int j = 0; j < k; ++j)
                sub[static_cast<std::size_t>(i) * k + j] =
                    g[static_cast<std::size_t>(i) * d + j];
        const std::complex<double> minor = determinant(sub, k);
        // A genuinely spacelike cell has real, positive leading minors. A
        // nonzero imaginary part means the cell is not spacelike at all, which
        // this assertion exists to catch, so it fails rather than projecting.
        if (!(minor.imag() == 0.0 && minor.real() > tol)) {
            throw std::runtime_error(
                "Simplex::assertSpacelikeAdmissible: inadmissible spacelike "
                "simplex — Gram matrix is not positive-definite (leading minor "
                + std::to_string(k) + " = " + std::to_string(minor.real()) +
                " + " + std::to_string(minor.imag()) + "i" +
                "); the spacelike triangle inequalities are violated. The metric "
                "is not silently repaired.");
        }
    }
}

namespace {

// The old signedSqrt = sign(x)*sqrt(|x|) is gone: it was not a branch choice but a
// real-valued convention that refused to go imaginary, mapping a timelike
// circumcentric height to a negative real instead of the imaginary value it is.
// principalSqrt (file scope, above) replaces it (#641).

// Circumcenter (barycentric) + signed R² from the Gram matrix G (flat d×d,
// relative to vertex 0) with its determinant and cofactors precomputed — the
// cached Gram sections (#668) enter here. Solves G β = ½·diag(G) Eigen-free
// via the adjugate (cofactorᵀ/det); λ_0 = 1−Σβ, λ_i = β_i; R² = Σ_i β_i·(½ G_ii).
void circumFromGramCore(const std::vector<std::complex<double>>& G,
                        const std::complex<double> detG,
                        const std::vector<std::complex<double>>& cof, int d,
                        std::vector<std::complex<double>>& bary,
                        std::complex<double>& r2) {
    using cd = std::complex<double>;
    bary.assign(static_cast<std::size_t>(d) + 1, cd{0.0, 0.0});
    if (d <= 0) { bary[0] = cd{1.0, 0.0}; r2 = cd{0.0, 0.0}; return; }
    std::vector<cd> halfDiag(d);
    for (int i = 0; i < d; ++i)
        halfDiag[i] = 0.5 * G[static_cast<std::size_t>(i) * d + i];
    // β_i = Σ_j (G⁻¹)_ij·halfDiag_j, with (G⁻¹)_ij = adj_ij/det = C_ji/det.
    std::vector<cd> beta(d, cd{0.0, 0.0});
    cd sum{0.0, 0.0};
    for (int i = 0; i < d; ++i) {
        cd acc{0.0, 0.0};
        for (int j = 0; j < d; ++j)
            acc += cof[static_cast<std::size_t>(j) * d + i] * halfDiag[j];
        beta[i] = (detG != cd{0.0, 0.0}) ? acc / detG : cd{0.0, 0.0};
        bary[static_cast<std::size_t>(i) + 1] = beta[i];
        sum += beta[i];
    }
    bary[0] = cd{1.0, 0.0} - sum;
    r2 = cd{0.0, 0.0};
    for (int i = 0; i < d; ++i) r2 += beta[i] * halfDiag[i];
}

// Sign (±1) of the circumcenter of coface `cf` at the vertex of `cf` not in its
// facet `s` — the side of the facet's hull on which c(cf) sits.
double oppositeVertexSign(const ::tessera::mesh::Simplex* cf,
                          const ::tessera::mesh::Simplex* s) {
    const auto& cfv = cf->getVertices();
    const auto& sv = s->getVertices();
    int oppIdx = -1;
    for (std::size_t i = 0; i < cfv.size(); ++i) {
        bool inS = false;
        for (const auto* w : sv)
            if (w->getId() == cfv[i]->getId()) { inS = true; break; }
        if (!inS) { oppIdx = static_cast<int>(i); break; }
    }
    if (oppIdx < 0) return 1.0;
    const std::vector<std::complex<double>> bary = cf->circumcenterBarycentric();
    // This +/-1 is GEOMETRIC, not a branch of a square root: it records which side
    // of the shared facet c(cf) fell on, and an obtuse cell genuinely needs the -1.
    // Orientation is not a function of edge lengths, so no complex root supplies
    // it -- deleting this would silently switch the signed dual-volume convention
    // to the unsigned overcount (#605 audits exactly this sign).
    //
    // Reading it off Re(bary) is bit-identical to the real-Lorentzian behaviour,
    // since bary is real there, and continues off-axis by continuity in Re. How it
    // should generalise for a genuinely off-axis geometry is the open design
    // question on #637; it is deliberately NOT settled here.
    return (bary[static_cast<std::size_t>(oppIdx)].real() < 0.0) ? -1.0 : 1.0;
}

// Recursive signed circumcentric dual content of `s` in an n-complex.
std::complex<double> dualVolRec(const ::tessera::mesh::Simplex* s, int n) {
    const int k = static_cast<int>(s->size()) - 1;
    if (k >= n) return {1.0, 0.0};  // top cell: dual is a point (content 1)
    const std::complex<double> rk2 = s->circumradiusSquared();
    std::complex<double> acc{0.0, 0.0};
    for (const auto& cf : s->getCofaces()) {
        const std::complex<double> h =
            oppositeVertexSign(cf, s) * principalSqrt(cf->circumradiusSquared() - rk2);
        acc += h * dualVolRec(cf, n);
    }
    return acc / static_cast<double>(n - k);
}

// Exact d(R^2)/d(l^2_e) for simplex `s` w.r.t. edge (ea,eb): R^2 = h^T G^-1 h
// (h = 1/2 diag G), so dR^2 = 2(dh)^T beta - beta^T (dG) beta, beta = G^-1 h.
// The Gram matrix is linear in l^2, so dG/dh are indicator matrices.
std::complex<double> dCircumR2(const ::tessera::mesh::Simplex* s,
                 std::uint64_t ea, std::uint64_t eb) {
    const int d = static_cast<int>(s->size()) - 1;
    if (d <= 0) return {0.0, 0.0};
    const auto& sv = s->getVertices();
    // Cached Gram pipeline (#668): this runs once per (cell, edge) pair in the
    // dual-volume gradient, all against the same cell geometry.
    const auto &gc = s->gramCofCache();
    const std::vector<std::complex<double>> &G = gc.gram;
    if (static_cast<int>(G.size()) != d * d) return {0.0, 0.0};
    const std::complex<double> detG = gc.gramDet;
    if (std::abs(detG) < 1e-300) return {0.0, 0.0};
    const std::vector<std::complex<double>> &cofG = gc.gramCof;
    std::vector<std::complex<double>> h(d), beta(d, std::complex<double>{0.0, 0.0});
    for (int i = 0; i < d; ++i) h[i] = 0.5 * G[i * d + i];
    for (int i = 0; i < d; ++i) {              // beta = G^-1 h, (G^-1)_ij=cof_ji/det
        std::complex<double> a{0.0, 0.0};
        for (int j = 0; j < d; ++j) a += cofG[j * d + i] * h[j];
        beta[i] = a / detG;
    }
    const std::uint64_t lo = std::min(ea, eb), hi = std::max(ea, eb);
    auto ind = [&](int p, int q) -> double {
        if (p == q) return 0.0;
        const std::uint64_t a = sv[p]->getId(), b = sv[q]->getId();
        return (std::min(a, b) == lo && std::max(a, b) == hi) ? 1.0 : 0.0;
    };
    std::vector<std::complex<double>> dG(static_cast<std::size_t>(d) * d), dh(d);
    for (int i = 0; i < d; ++i)
        for (int j = 0; j < d; ++j)
            dG[i * d + j] = 0.5 * (ind(0, i + 1) + ind(0, j + 1) - ind(i + 1, j + 1));
    for (int i = 0; i < d; ++i) dh[i] = 0.5 * dG[i * d + i];
    std::complex<double> r{0.0, 0.0};
    for (int i = 0; i < d; ++i) r += 2.0 * dh[i] * beta[i];
    for (int i = 0; i < d; ++i)
        for (int j = 0; j < d; ++j) r -= beta[i] * dG[i * d + j] * beta[j];
    return r;
}

// Exact d^2(R^2)/d(l^2_e)d(l^2_f). Since dR^2 = 2(dh)^T beta - beta^T dG beta and
// G is linear in l^2 (dG, dh constant indicators), the second derivative is
// 2(dh_e)^T (d_f beta) - 2 beta^T dG_e (d_f beta), with
// d_f beta = G^-1 (dh_f - dG_f beta). Symmetric in (e,f).
std::complex<double> d2CircumR2(const ::tessera::mesh::Simplex* s,
                  std::uint64_t ea, std::uint64_t eb,
                  std::uint64_t fa, std::uint64_t fb) {
    const int d = static_cast<int>(s->size()) - 1;
    if (d <= 0) return {0.0, 0.0};
    const auto& sv = s->getVertices();
    // Cached Gram pipeline (#668), as in dCircumR2: one fill serves every
    // (edge, edge) pair of this cell's Hessian block.
    const auto &gc = s->gramCofCache();
    const std::vector<std::complex<double>> &G = gc.gram;
    if (static_cast<int>(G.size()) != d * d) return {0.0, 0.0};
    const std::complex<double> detG = gc.gramDet;
    if (std::abs(detG) < 1e-300) return {0.0, 0.0};
    const std::vector<std::complex<double>> &cofG = gc.gramCof;
    std::vector<std::complex<double>> Ginv(static_cast<std::size_t>(d) * d);
    for (int i = 0; i < d; ++i)
        for (int j = 0; j < d; ++j)
            Ginv[i * d + j] = cofG[j * d + i] / detG;   // (G^-1)_ij = C_ji/det
    std::vector<std::complex<double>> h(d), beta(d, std::complex<double>{0.0, 0.0});
    for (int i = 0; i < d; ++i) h[i] = 0.5 * G[i * d + i];
    for (int i = 0; i < d; ++i) {
        std::complex<double> a{0.0, 0.0};
        for (int j = 0; j < d; ++j) a += Ginv[i * d + j] * h[j];
        beta[i] = a;
    }
    auto indMat = [&](std::uint64_t e0, std::uint64_t e1,
                      std::vector<std::complex<double>>& dG, std::vector<std::complex<double>>& dh) {
        const std::uint64_t lo = std::min(e0, e1), hi = std::max(e0, e1);
        auto ind = [&](int p, int q) -> double {
            if (p == q) return 0.0;
            const std::uint64_t a = sv[p]->getId(), b = sv[q]->getId();
            return (std::min(a, b) == lo && std::max(a, b) == hi) ? 1.0 : 0.0;
        };
        dG.assign(static_cast<std::size_t>(d) * d, 0.0);
        dh.assign(d, 0.0);
        for (int i = 0; i < d; ++i)
            for (int j = 0; j < d; ++j)
                dG[i * d + j] =
                    0.5 * (ind(0, i + 1) + ind(0, j + 1) - ind(i + 1, j + 1));
        for (int i = 0; i < d; ++i) dh[i] = 0.5 * dG[i * d + i];
    };
    std::vector<std::complex<double>> dG_e, dh_e, dG_f, dh_f;
    indMat(ea, eb, dG_e, dh_e);
    indMat(fa, fb, dG_f, dh_f);
    // d_f beta = G^-1 (dh_f - dG_f beta)
    std::vector<std::complex<double>> tmp(d, {0.0, 0.0}), dbeta_f(d, {0.0, 0.0});
    for (int i = 0; i < d; ++i) {
        std::complex<double> a = dh_f[i];
        for (int j = 0; j < d; ++j) a -= dG_f[i * d + j] * beta[j];
        tmp[i] = a;
    }
    for (int i = 0; i < d; ++i) {
        std::complex<double> a{0.0, 0.0};
        for (int j = 0; j < d; ++j) a += Ginv[i * d + j] * tmp[j];
        dbeta_f[i] = a;
    }
    std::complex<double> r{0.0, 0.0};
    for (int i = 0; i < d; ++i) r += 2.0 * dh_e[i] * dbeta_f[i];
    for (int i = 0; i < d; ++i)
        for (int j = 0; j < d; ++j)
            r -= 2.0 * beta[i] * dG_e[i * d + j] * dbeta_f[j];
    return r;
}

}  // namespace

std::vector<std::complex<double>> Simplex::circumcenterBarycentric() const {
    const int d = static_cast<int>(size()) - 1;
    std::vector<std::complex<double>> bary;
    std::complex<double> r2{0.0, 0.0};
    const GeomCache &gc = gramCofCache();       // (#668)
    circumFromGramCore(gc.gram, gc.gramDet, gc.gramCof, d, bary, r2);
    return bary;
}

std::complex<double> Simplex::circumradiusSquared() const {
    const int d = static_cast<int>(size()) - 1;
    std::vector<std::complex<double>> bary;
    std::complex<double> r2{0.0, 0.0};
    const GeomCache &gc = gramCofCache();       // (#668)
    circumFromGramCore(gc.gram, gc.gramDet, gc.gramCof, d, bary, r2);
    return r2;
}

std::vector<Simplex *> Simplex::incidentTopCells() const {
    std::vector<Simplex *> out;
    if (!spacetime || vertices.empty()) return out;
    const int topSize =
        spacetime->getMetric()->getSignature()->getDimensions() + 1;
    std::unordered_set<std::uint64_t> seen;
    for (const auto &anchor : vertices) {
        for (const auto &sigma : anchor->getSimplices()) {
            if (static_cast<int>(sigma->size()) != topSize) continue;
            bool containsAll = true;
            for (const auto &hv : vertices)
                if (!sigma->hasVertex(hv)) { containsAll = false; break; }
            if (!containsAll) continue;
            if (seen.insert(sigma->fingerprint.fingerprint()).second)
                out.push_back(sigma);
        }
    }
    return out;
}

bool Simplex::hasTopCoface() const {
    if (!spacetime || vertices.empty()) return false;
    const int topSize =
        spacetime->getMetric()->getSignature()->getDimensions() + 1;
    if (static_cast<int>(size()) >= topSize) return true;  // already top
    return !incidentTopCells().empty();
}

int Simplex::ambientTopDimension() const {
    // Prefer the metric dimension: it is the genuine ambient n and is immune to
    // orphan cofaces a move may have left dangling in this simplex's coface list.
    if (spacetime) return spacetime->getMetric()->getSignature()->getDimensions();
    // Coordinate-free fixture (no spacetime): fall back to the coface walk.
    const Simplex* top = this;
    while (!top->getCofaces().empty()) top = top->getCofaces()[0];
    return static_cast<int>(top->size()) - 1;
}

std::complex<double> Simplex::dualVolume() const {
    return dualVolRec(this, ambientTopDimension());
}

std::map<std::pair<std::uint64_t, std::uint64_t>, std::complex<double>>
Simplex::volumeGradient() const {
    // dV/dl^2_e = (V/2) tr(G^-1 dG_e), Jacobi's formula on the Gram determinant
    // (V = sgn sqrt(|det G|)/d!, G linear in l^2 so dG_e is an indicator matrix —
    // the same dG the #354 dCircumR2 uses). G^-1 via the adjugate (cofactor^T/det),
    // Eigen-free, matching circumFromGram / volume().
    std::map<std::pair<std::uint64_t, std::uint64_t>, std::complex<double>> grad;
    const int d = static_cast<int>(size()) - 1;
    if (d < 1) return grad;
    const GeomCache &gc = gramCofCache();       // (#668)
    const std::vector<std::complex<double>> &G = gc.gram;
    if (static_cast<int>(G.size()) != d * d) return grad;
    const std::complex<double> detG = gc.gramDet;
    if (std::abs(detG) < 1e-300) return grad;
    const std::vector<std::complex<double>> &cofG = gc.gramCof;  // cof[r*d+c] = C_rc
    const std::complex<double> V = volume();
    const auto &sv = vertices;
    for (std::size_t p = 0; p < sv.size(); ++p)
        for (std::size_t q = p + 1; q < sv.size(); ++q) {
            const std::uint64_t a = sv[p]->getId(), b = sv[q]->getId();
            const std::pair<std::uint64_t, std::uint64_t> ek{std::min(a, b),
                                                             std::max(a, b)};
            auto ind = [&](int i, int j) -> double {
                if (i == j) return 0.0;
                const std::uint64_t x = sv[static_cast<std::size_t>(i)]->getId();
                const std::uint64_t y = sv[static_cast<std::size_t>(j)]->getId();
                return (std::min(x, y) == ek.first && std::max(x, y) == ek.second)
                           ? 1.0 : 0.0;
            };
            // tr(G^-1 dG) = sum_ij (G^-1)_ij dG_ji; dG symmetric, (G^-1)_ij=cof_ji/det.
            std::complex<double> tr{0.0, 0.0};
            for (int i = 0; i < d; ++i)
                for (int j = 0; j < d; ++j) {
                    const std::complex<double> dGij =
                        0.5 * (ind(0, i + 1) + ind(0, j + 1) - ind(i + 1, j + 1));
                    const std::complex<double> GinvIJ =
                        cofG[static_cast<std::size_t>(j) * d + i] / detG;
                    tr += GinvIJ * dGij;
                }
            grad[ek] += 0.5 * V * tr;
        }
    return grad;
}

std::map<std::pair<std::uint64_t, std::uint64_t>, std::complex<double>>
Simplex::volumeGradientDirectionalDerivative(
    const std::map<std::pair<std::uint64_t, std::uint64_t>,
                   std::complex<double>> &direction) const {
    // Jacobi's formula differentiated a second time. G is LINEAR in l^2, so the
    // d^2G/dl^2 dl^2 term vanishes identically and the entire second derivative
    // is carried by the two first-order pieces assembled below.
    std::map<std::pair<std::uint64_t, std::uint64_t>, std::complex<double>> out;
    const int d = static_cast<int>(size()) - 1;
    if (d < 1) return out;
    const GeomCache &gc = gramCofCache();
    const std::vector<std::complex<double>> &G = gc.gram;
    if (static_cast<int>(G.size()) != d * d) return out;
    const std::complex<double> detG = gc.gramDet;
    if (std::abs(detG) < 1e-300) return out;
    const std::vector<std::complex<double>> &cofG = gc.gramCof;
    const std::complex<double> V = volume();
    const auto &sv = vertices;
    const std::size_t dim = static_cast<std::size_t>(d);
    const std::complex<double> zero{0.0, 0.0};

    // G^-1 via the adjugate, exactly as volumeGradient(): (G^-1)_ij = cof_ji/det.
    std::vector<std::complex<double>> Ginv(dim * dim);
    for (int i = 0; i < d; ++i)
        for (int j = 0; j < d; ++j)
            Ginv[static_cast<std::size_t>(i) * dim + j] =
                cofG[static_cast<std::size_t>(j) * dim + i] / detG;

    // The direction's entry for the edge between simplex-local vertices x, y:
    // zero on the diagonal and for any edge the caller left out.
    const auto directionAt = [&](int x, int y) -> std::complex<double> {
        if (x == y) return zero;
        const std::uint64_t a = sv[static_cast<std::size_t>(x)]->getId();
        const std::uint64_t b = sv[static_cast<std::size_t>(y)]->getId();
        const auto found = direction.find({std::min(a, b), std::max(a, b)});
        return found == direction.end() ? zero : found->second;
    };

    // Gdot = sum_f v_f dG_f, read straight off G's affine form
    // G_ij = (l^2_{0,i+1} + l^2_{0,j+1} - l^2_{i+1,j+1})/2.
    std::vector<std::complex<double>> Gdot(dim * dim);
    for (int i = 0; i < d; ++i)
        for (int j = 0; j < d; ++j)
            Gdot[static_cast<std::size_t>(i) * dim + j] =
                0.5 * (directionAt(0, i + 1) + directionAt(0, j + 1) -
                       directionAt(i + 1, j + 1));

    std::complex<double> tau{0.0, 0.0};   // tr(G^-1 Gdot)
    for (int i = 0; i < d; ++i)
        for (int j = 0; j < d; ++j)
            tau += Ginv[static_cast<std::size_t>(i) * dim + j] *
                   Gdot[static_cast<std::size_t>(j) * dim + i];
    const std::complex<double> Vdot = 0.5 * V * tau;

    // M = G^-1 Gdot G^-1 once per simplex; every edge is then one sparse
    // contraction tr(M dG_e) rather than a fresh triple product.
    std::vector<std::complex<double>> GinvGdot(dim * dim, zero);
    for (int i = 0; i < d; ++i)
        for (int k = 0; k < d; ++k) {
            const std::complex<double> left =
                Ginv[static_cast<std::size_t>(i) * dim + k];
            if (left == zero) continue;
            for (int j = 0; j < d; ++j)
                GinvGdot[static_cast<std::size_t>(i) * dim + j] +=
                    left * Gdot[static_cast<std::size_t>(k) * dim + j];
        }
    std::vector<std::complex<double>> M(dim * dim, zero);
    for (int i = 0; i < d; ++i)
        for (int k = 0; k < d; ++k) {
            const std::complex<double> left =
                GinvGdot[static_cast<std::size_t>(i) * dim + k];
            if (left == zero) continue;
            for (int j = 0; j < d; ++j)
                M[static_cast<std::size_t>(i) * dim + j] +=
                    left * Ginv[static_cast<std::size_t>(k) * dim + j];
        }

    for (std::size_t p = 0; p < sv.size(); ++p)
        for (std::size_t q = p + 1; q < sv.size(); ++q) {
            const std::uint64_t a = sv[p]->getId(), b = sv[q]->getId();
            const std::pair<std::uint64_t, std::uint64_t> ek{std::min(a, b),
                                                             std::max(a, b)};
            auto ind = [&](int i, int j) -> double {
                if (i == j) return 0.0;
                const std::uint64_t x = sv[static_cast<std::size_t>(i)]->getId();
                const std::uint64_t y = sv[static_cast<std::size_t>(j)]->getId();
                return (std::min(x, y) == ek.first && std::max(x, y) == ek.second)
                           ? 1.0 : 0.0;
            };
            std::complex<double> traceGinv{0.0, 0.0};  // t_e = tr(G^-1 dG_e)
            std::complex<double> traceM{0.0, 0.0};     // tr(M dG_e)
            for (int i = 0; i < d; ++i)
                for (int j = 0; j < d; ++j) {
                    const std::complex<double> dGij =
                        0.5 * (ind(0, i + 1) + ind(0, j + 1) - ind(i + 1, j + 1));
                    if (dGij == zero) continue;
                    traceGinv += Ginv[static_cast<std::size_t>(i) * dim + j] * dGij;
                    traceM += M[static_cast<std::size_t>(i) * dim + j] * dGij;
                }
            out[ek] += 0.5 * Vdot * traceGinv - 0.5 * V * traceM;
        }
    return out;
}

namespace {

/// Is this complex number exactly zero?
[[nodiscard]] inline bool isExactlyZero(std::complex<double> z) noexcept {
    return z.real() == 0.0 && z.imag() == 0.0;
}

/// The chain rule through a square root: d/dt sqrt(x(t)) = x'(t) / (2 sqrt(x)).
///
/// The circumcentric dual volume is built from roots of circumradius
/// differences, and such a difference vanishes exactly whenever a facet's
/// circumsphere coincides with its hinge's. sqrt has infinite slope at the
/// origin, so the quotient genuinely diverges there -- but only when the
/// radicand actually moves. It does not move along most directions: `x'` is the
/// change in a circumradius difference under one edge's squared length, and an
/// edge the facet does not carry leaves it exactly zero. Along such a direction
/// the radicand is pinned at zero, so the root is identically zero and so is its
/// derivative.
///
/// Evaluating the quotient as written turns that case into `(1/0) * 0`, which is
/// NaN, and one NaN contaminates every edge the hinge contributes to. Taking the
/// zero numerator first gives the value the limit actually has.
[[nodiscard]] inline std::complex<double> rootChainRule(
        std::complex<double> radicandDerivative,
        std::complex<double> root) noexcept {
    if (isExactlyZero(radicandDerivative)) return {0.0, 0.0};
    return radicandDerivative / (2.0 * root);
}

/// A product in which an exactly-zero factor wins.
///
/// Used where a root that has vanished multiplies a derivative that may itself
/// have diverged. The root being exactly zero pins the product to zero whenever
/// the other factor is finite, which is the case here; written plainly the
/// expression would be `0 * inf` and evaluate to NaN.
[[nodiscard]] inline std::complex<double> productWithZeroAbsorbing(
        std::complex<double> a, std::complex<double> b) noexcept {
    if (isExactlyZero(a) || isExactlyZero(b)) return {0.0, 0.0};
    return a * b;
}

} // namespace

bool Simplex::dualGeometryIsDegenerate() const {
    if (vertices.empty()) return false;
    const int n = ambientTopDimension();
    const int k = static_cast<int>(size()) - 1;
    if (k != n - 2) return false;          // the (n-2) hinge the dual is built on

    const std::complex<double> hingeRadius = circumradiusSquared();
    for (const auto& facet : getCofaces()) {
        const std::complex<double> facetRadius = facet->circumradiusSquared();
        if (isExactlyZero(facetRadius - hingeRadius)) return true;
        for (const auto& top : facet->getCofaces())
            if (isExactlyZero(top->circumradiusSquared() - facetRadius))
                return true;
    }
    return false;
}

std::map<std::pair<std::uint64_t, std::uint64_t>, std::complex<double>>
Simplex::dualVolumeGradient() const {
    std::map<std::pair<std::uint64_t, std::uint64_t>, std::complex<double>> grad;
    if (vertices.empty()) return grad;
    const int n = ambientTopDimension();
    const int k = static_cast<int>(size()) - 1;
    if (k != n - 2) return grad;          // the (n-2) hinge the Regge action needs

    const std::complex<double> Rh2 = circumradiusSquared();
    struct Facet {
        const Simplex* cf; double sgn;
        std::complex<double> R1; std::complex<double> inner;
        std::vector<std::pair<const Simplex*,
                              std::pair<double, std::complex<double>>>> tops;
    };
    std::vector<Facet> fs;
    std::set<std::pair<std::uint64_t, std::uint64_t>> edges;
    for (const auto& cf : getCofaces()) {
        Facet f;
        f.cf = cf; f.sgn = oppositeVertexSign(cf, this);
        f.R1 = cf->circumradiusSquared(); f.inner = {0.0, 0.0};
        for (const auto& tp : cf->getCofaces()) {
            const double sgn2 = oppositeVertexSign(tp, cf);
            const std::complex<double> R2 = tp->circumradiusSquared();
            f.inner += sgn2 * principalSqrt(R2 - f.R1);
            f.tops.push_back({tp, {sgn2, R2}});
            const auto& tv = tp->getVertices();
            for (std::size_t i = 0; i < tv.size(); ++i)
                for (std::size_t j = i + 1; j < tv.size(); ++j) {
                    const std::uint64_t a = tv[i]->getId(), b = tv[j]->getId();
                    edges.insert({std::min(a, b), std::max(a, b)});
                }
        }
        fs.push_back(std::move(f));
    }
    const double inv = 1.0 / (static_cast<double>(n - k) * (n - k - 1));
    for (const auto& e : edges) {
        const std::complex<double> dRh = dCircumR2(this, e.first, e.second);
        std::complex<double> dV{0.0, 0.0};
        for (const auto& f : fs) {
            const std::complex<double> dR1 = dCircumR2(f.cf, e.first, e.second);
            const std::complex<double> x1 = f.R1 - Rh2;
            const std::complex<double> ss1 = principalSqrt(x1);
            // d/dx sqrt(x) = 1/(2 sqrt(x)) on the principal branch. The old form
            // took 1/(2 sqrt(|x| + eps)), which is the derivative of signedSqrt
            // plus a regulator; neither is needed once the root is complex.
            std::complex<double> dinner{0.0, 0.0};
            for (const auto& t : f.tops) {
                const std::complex<double> R2 = t.second.second;
                const double sgn2 = t.second.first;
                const std::complex<double> dR2 = dCircumR2(t.first, e.first, e.second);
                const std::complex<double> x2 = R2 - f.R1;
                dinner += sgn2 * rootChainRule(dR2 - dR1, principalSqrt(x2));
            }
            dV += f.sgn * (productWithZeroAbsorbing(rootChainRule(dR1 - dRh, ss1),
                                                    f.inner)
                           + productWithZeroAbsorbing(ss1, dinner));
        }
        grad[e] = dV * inv;
    }
    return grad;
}

std::map<std::pair<std::pair<std::uint64_t, std::uint64_t>,
                   std::pair<std::uint64_t, std::uint64_t>>,
         std::complex<double>>
Simplex::dualVolumeHessian() const {
    using EK = std::pair<std::uint64_t, std::uint64_t>;
    std::map<std::pair<EK, EK>, std::complex<double>> hess;
    if (vertices.empty()) return hess;
    const int n = ambientTopDimension();
    const int k = static_cast<int>(size()) - 1;
    if (k != n - 2) return hess;

    const std::complex<double> Rh2 = circumradiusSquared();
    struct Facet {
        const Simplex* cf; double sgn; std::complex<double> R1;
        std::vector<std::pair<const Simplex*,
                              std::pair<double, std::complex<double>>>> tops;
    };
    std::vector<Facet> fs;
    std::set<EK> edges;
    for (const auto& cf : getCofaces()) {
        Facet f; f.cf = cf; f.sgn = oppositeVertexSign(cf, this);
        f.R1 = cf->circumradiusSquared();
        for (const auto& tp : cf->getCofaces()) {
            const double sgn2 = oppositeVertexSign(tp, cf);
            f.tops.push_back({tp, {sgn2, tp->circumradiusSquared()}});
            const auto& tv = tp->getVertices();
            for (std::size_t i = 0; i < tv.size(); ++i)
                for (std::size_t j = i + 1; j < tv.size(); ++j) {
                    const std::uint64_t a = tv[i]->getId(), b = tv[j]->getId();
                    edges.insert({std::min(a, b), std::max(a, b)});
                }
        }
        fs.push_back(std::move(f));
    }
    const double inv = 1.0 / (static_cast<double>(n - k) * (n - k - 1));
    // g(x) = sqrt(x) on the principal branch, so g'(x) = 1/(2 sqrt(x)) and
    // g''(x) = -1/(4 x sqrt(x)). Both blow up where the radicand vanishes, which
    // happens exactly when a facet's circumsphere coincides with its hinge's.
    // As in the gradient, the radicand is pinned at zero along every direction
    // whose edge the facet does not carry, so the chain rule's numerator is zero
    // there and the term is zero; taking the numerator first is what keeps a
    // vanishing radicand from turning the whole row into NaN.
    auto firstOrder = [](std::complex<double> x, std::complex<double> dx) {
        return rootChainRule(dx, principalSqrt(x));
    };
    auto secondOrder = [](std::complex<double> x, std::complex<double> dxe,
                          std::complex<double> dxf) {
        if (isExactlyZero(dxe) || isExactlyZero(dxf)) return std::complex<double>{0.0, 0.0};
        return -0.25 / (x * principalSqrt(x)) * dxe * dxf;
    };
    const std::vector<EK> ev(edges.begin(), edges.end());
    for (const auto& e : ev) {
        const std::complex<double> dRh_e = dCircumR2(this, e.first, e.second);
        for (const auto& f : ev) {
            const std::complex<double> dRh_f = dCircumR2(this, f.first, f.second);
            const std::complex<double> d2Rh =
                d2CircumR2(this, e.first, e.second, f.first, f.second);
            std::complex<double> dV2{0.0, 0.0};
            for (const auto& fac : fs) {
                const std::complex<double> dR1_e = dCircumR2(fac.cf, e.first, e.second);
                const std::complex<double> dR1_f = dCircumR2(fac.cf, f.first, f.second);
                const std::complex<double> d2R1 =
                    d2CircumR2(fac.cf, e.first, e.second, f.first, f.second);
                const std::complex<double> x1 = fac.R1 - Rh2;
                const std::complex<double> ss1 = principalSqrt(x1);
                const std::complex<double> dx1_e = dR1_e - dRh_e, dx1_f = dR1_f - dRh_f;
                const std::complex<double> dss1_e = firstOrder(x1, dx1_e),
                                           dss1_f = firstOrder(x1, dx1_f);
                const std::complex<double> d2ss1 =
                    secondOrder(x1, dx1_e, dx1_f) + firstOrder(x1, d2R1 - d2Rh);
                std::complex<double> S{0.0,0.0}, dS_e{0.0,0.0}, dS_f{0.0,0.0}, d2S{0.0,0.0};
                for (const auto& t : fac.tops) {
                    const double sgn2 = t.second.first;
                    const std::complex<double> R2 = t.second.second;
                    const std::complex<double> dR2_e = dCircumR2(t.first, e.first, e.second);
                    const std::complex<double> dR2_f = dCircumR2(t.first, f.first, f.second);
                    const std::complex<double> d2R2 =
                        d2CircumR2(t.first, e.first, e.second, f.first, f.second);
                    const std::complex<double> x2 = R2 - fac.R1;
                    const std::complex<double> dx2_e = dR2_e - dR1_e, dx2_f = dR2_f - dR1_f;
                    S += sgn2 * principalSqrt(x2);
                    dS_e += sgn2 * firstOrder(x2, dx2_e);
                    dS_f += sgn2 * firstOrder(x2, dx2_f);
                    d2S += sgn2 * (secondOrder(x2, dx2_e, dx2_f)
                                   + firstOrder(x2, d2R2 - d2R1));
                }
                dV2 += fac.sgn
                       * (productWithZeroAbsorbing(d2ss1, S)
                          + productWithZeroAbsorbing(dss1_e, dS_f)
                          + productWithZeroAbsorbing(dss1_f, dS_e)
                          + productWithZeroAbsorbing(ss1, d2S));
            }
            hess[{e, f}] = dV2 * inv;
        }
    }
    return hess;
}

std::complex<double> Simplex::hodgeStar() const {
    const std::complex<double> v = volume();
    if (v == std::complex<double>{0.0, 0.0}) {
        throw std::runtime_error(
            "Simplex::hodgeStar: primal volume is zero (degenerate simplex)");
    }
    return dualVolume() / v;
}

}
