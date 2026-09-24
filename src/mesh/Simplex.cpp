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
/// Principal complex square root, with a negative-zero imaginary part normalised away
/// first: std::sqrt lands on the far side of the branch cut for -0.0, which a real-typed
/// sign test would never see but the complex form does.
inline std::complex<double> principalSqrt(std::complex<double> z) {
    if (z.imag() == 0.0) z = {z.real(), 0.0};
    return std::sqrt(z);
}

/// The dihedral cosine and angle at a hinge, from a Cayley-Menger cofactor
/// matrix.
///
/// `cofactors` is the n x n cofactor matrix of the (d+2) x (d+2) Cayley-Menger
/// matrix, and `bi`, `bj` are the border offsets of the two vertices outside
/// the hinge. The angle is
///
///     theta = acos(r),  r = -C_ij / (sqrt(C_ii) sqrt(C_jj))
///
/// with two separate principal roots rather than one root of the product: that
/// single expression covers every causal regime, where folding the roots
/// together would need sign flags and a crossing dispatch. Opposite-signed
/// cofactors make the denominator pure imaginary and the principal
/// acos(i y) = pi/2 - i asinh(y) reproduces Sorkin's quarter turn with no
/// special case.
///
/// `ok` is false when the cofactors are unusable -- a degenerate denominator,
/// or a cofactor matrix of the wrong size -- and the caller skips the hinge.
///
/// The value reads this in the canonical sorted-by-id frame and the derivatives
/// in the raw stored order. The ratio is invariant under the permutation
/// relating them, which tests/mesh/test_dihedral_frame_invariance_python.py
/// asserts against a finite difference.
struct DihedralCosine {
    bool ok{false};
    std::complex<double> r{};      ///< the cosine, with its branch pinned
    std::complex<double> theta{};  ///< acos(r)
    /// The three cofactors and their combined root, which the derivative
    /// chains differentiate: d(denom) = denom (dCii/Cii + dCjj/Cjj) / 2.
    std::complex<double> Cij{};
    std::complex<double> Cii{};
    std::complex<double> Cjj{};
    std::complex<double> denom{};
};

inline DihedralCosine dihedralCosine(
    const std::vector<std::complex<double>> &cofactors, int n, int bi, int bj,
    const Simplex::DihedralSheet &sheet = {}) {
    if (static_cast<int>(cofactors.size()) != n * n) return {};
    const std::complex<double> Cij =
        cofactors[static_cast<std::size_t>(bi) * n + bj];
    const std::complex<double> Cii =
        cofactors[static_cast<std::size_t>(bi) * n + bi];
    const std::complex<double> Cjj =
        cofactors[static_cast<std::size_t>(bj) * n + bj];
    // The declared root product: the principal roots times the declared sign.
    // On the principal sheet the sign is +1 and this is the sheet-blind value.
    const std::complex<double> denom = static_cast<double>(sheet.rootSign) *
                                       principalSqrt(Cii) * principalSqrt(Cjj);
    if (std::abs(denom) < 1e-300) return {};
    std::complex<double> r = -Cij / denom;
    // acos is cut on (-inf,-1] and [1,inf), so for a real ratio with |r| > 1 --
    // the same-sign, boost wedge -- the sign of Im(theta) is decided by which
    // side of the cut the argument sits on, i.e. by the sign of its zero
    // imaginary part. Complex division would leave that to floating-point
    // accident, so it is pinned to +0.0, the side acos(complex(r, 0.0)) takes.
    // Boost orientation is not determined by edge lengths alone (a PT
    // reflection flips it at identical l^2), so this is a convention, and a
    // stated one rather than an emergent rounding. A derivative must sit on the
    // same sheet as the value or it disagrees with a finite difference of it.
    if (r.imag() == 0.0) r = {r.real(), 0.0};
    // The declared sheet (k, epsilon) of the inverse cosine,
    // theta = 2 pi k + epsilon Arccos(r); (0, +1) is the principal value.
    const std::complex<double> theta =
        2.0 * std::numbers::pi * static_cast<double>(sheet.branchIndex) +
        static_cast<double>(sheet.orientation) * std::acos(r);
    return {true, r, theta, Cij, Cii, Cjj, denom};
}

/// B^-1 = adj(B)/det = cof^T/det. B is symmetric, so B^-1 is too.
inline std::vector<std::complex<double>> inverseFromCofactors(
    const std::vector<std::complex<double>> &cofactors, int n,
    std::complex<double> det) {
    std::vector<std::complex<double>> inverse(static_cast<std::size_t>(n) * n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            inverse[i * n + j] = cofactors[j * n + i] / det;
    return inverse;
}
}  // namespace
using namespace ::tessera::graph;
using namespace ::tessera::spacetime;
using namespace ::tessera::observables;
using namespace ::tessera::simulations;
using namespace ::tessera::quantum;

// Tripwire for dereferences of stale Simplex pointers. Storage is stable, so a read
// from a logically removed simplex does not fault: it sees empty child vectors and
// proceeds silently. This macro turns that into an abort under TESSERA_ASSERTIONS and
// costs nothing in release builds. Used at the top of the hot getters a caller might
// reach through a cached SimplexPtr.
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

    // Cache the edges once, outside the loop.
    const auto &allEdges = getEdges();

    // Use this directly — no hash lookup needed.
    SimplexPtr coface = this;

    for (std::size_t skip = 0; skip < n; ++skip) {
      const auto skipVertexId = verts[skip]->getId();

      // Build faceVertices in one pass
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

  // Register after the fingerprint is set:
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
  // Edge identity is order-free -- Edge::operator== compares fingerprints, and
  // a fingerprint is an XOR over the endpoint ids -- so an edge stored
  // target-to-source is the same edge. Comparing source to source and target to
  // target would answer false for an edge EdgeList considers identical, and
  // disagree with the hasEdge(VertexPtr, VertexPtr) overload below.
  for (const auto &e : getEdges()) {
    if (*e == *edge) {
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
    // Causality has to be preserved. Coning a (1, 3) facet (one vertex at t, three at
    // t+1) into a (1, 4) coface needs a (2, 3) simplex, and (2, 3) - (1, 3) = (1, 0),
    // so the new vertex goes at t rather than t+1.
    //
    // In general an (n, m) simplex with an (n-1, m) or (n, m-1) facet has to match the
    // facet; what follows depends on the foliation. A preferred foliation needs a layer
    // of timelike edges between every layer of spacelike edges, so pairing only
    // compatible simplices amounts to keeping the vertices balanced on either side of
    // the spacelike sheet.
    //
    // A (3, 1) simplex with a (2, 1) facet leaves (3, 1) - (2, 1) = (1, 0), one extra
    // vertex at t, so the coning vertex goes at t+1 to make the new coface a (2, 2)
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
    // Flat (n x n) table of signed squared lengths by local index in
    // `ordering`: entry (i*n + j) is l^2 of the edge between ordering[i] and
    // ordering[j], 0 when the pair carries no edge. One linear pass over the edge list
    // with direct id matching: no mix64 hashing, no unordered_map, and immune to the
    // XOR-pair aliasing a hashed lookup admits. Duplicate pairs are last-edge-wins; a
    // self-edge matches no (i, j) pair.
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

    // Squared-distance lookup on the signed l^2: a timelike edge keeps its Lorentzian
    // sign in G, so det(G) records the cell's metric signature. There is no
    // Wick-rotated (|l^2|) mode.
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

// Each section accessor is the same double-checked pattern: a lock-free hit when the
// section's published key equals the current geometry-revision key, otherwise a
// mutex-serialized fill that publishes the key last (release), so a concurrent reader
// either sees the old key and takes the mutex, or sees the new key with the payload
// already written. Lengths mutate only in the serial phases between parallel
// evaluations, so within a parallel region the key is constant and the returned
// reference stays valid.
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

// The fills run the direct pipeline verbatim, same functions and same inputs, so a
// cached value is bit-for-bit what an uncached call produces.
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

Simplex::DihedralCofactors Simplex::dihedralCofactors(SimplexPtr hinge) const {
    // The front half of dihedralAngle, shared rather than copied so that a
    // sheet-carrying caller reads bit-for-bit the cofactors the angle is built
    // from. dihedralAngle below says why the canonical frame is the one to read.
    const int dPlus1 = static_cast<int>(vertices.size());
    const auto hingeVerts = hinge->getVertices();
    std::vector<int> opposite;
    for (int k = 0; k < dPlus1; ++k) {
        bool inHinge = false;
        for (const auto &hv : hingeVerts)
            if (hv->getId() == vertices[k]->getId()) { inHinge = true; break; }
        if (!inHinge) opposite.push_back(k);
    }
    if (opposite.size() != 2) return {};
    const int n = dPlus1 + 1;
    const GeomCache &cc = cmCanonicalCache();
    const auto &cof = cc.cmCanonCof;
    if (static_cast<int>(cof.size()) != n * n) return {};
    const int bi = canonicalPosition(cc.canonPos1, vertices[opposite[0]]->getId());
    const int bj = canonicalPosition(cc.canonPos1, vertices[opposite[1]]->getId());
    if (bi == 0 || bj == 0) return {};
    return {true,
            cof[static_cast<std::size_t>(bi) * n + bj],
            cof[static_cast<std::size_t>(bi) * n + bi],
            cof[static_cast<std::size_t>(bj) * n + bj]};
}

std::complex<double> Simplex::dihedralAngle(SimplexPtr hinge) const {
    return dihedralAngle(hinge, DihedralSheet{});
}

std::complex<double> Simplex::dihedralAngle(SimplexPtr hinge,
                                            const DihedralSheet &sheet) const {
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

    // Cayley-Menger cofactors -> the dihedral cosine ratio, unclamped:
    //
    //     cos(theta) = -C_ij / (sqrt(C_ii) * sqrt(C_jj))
    //
    // Two separate principal square roots, not sqrt(C_ii * C_jj). For complex a, b the
    // two differ by a sign exactly when both sit on the negative real axis: with the
    // unit tetrahedron's C_ii = C_jj = -3, sqrt(C_ii*C_jj) is +3 while
    // sqrt(C_ii)*sqrt(C_jj) is (i*r3)(i*r3) = -3. Folding the product under one root
    // forces a hand-applied (-1)^d parity fix, a three-way branch dispatch and an
    // i<->j anchoring swap; taking the roots separately makes all three emerge from the
    // branch structure.
    //
    // Every causal configuration is this one expression. Same-sign cofactors put the
    // wedge on one side of the light cone: a real angle for |r| <= 1, a boost
    // (pure-imaginary acos) for |r| > 1. Opposite signs mean the wedge crosses the cone:
    // the denominator turns pure-imaginary, r = i*y, and the principal
    // acos(i*y) = pi/2 - i*asinh(y) reproduces Sorkin's quarter turn with no special
    // case. Around a flat one-ray-per-quadrant vertex star the boosts telescope to zero
    // and four crossings sum to 2*pi, so 2*pi - sum = 0 holds.
    //
    // Evaluate in the canonical (sorted-by-id) frame so a cell a Pachner move stored in
    // causal order yields the same deficit as the same geometry built sorted; otherwise
    // the action depends on build history.
    const int n = dPlus1 + 1;
    // Cached canonical frame: the sorted-by-id Cayley-Menger matrix and its cofactors
    // are hinge-independent, so every hinge of this cell reads the same fill instead of
    // recomputing the O(n^5) cofactor pass per call.
    const GeomCache &cc = cmCanonicalCache();
    const auto &cof = cc.cmCanonCof;
    if (static_cast<int>(cof.size()) != n * n) return {0.0, 0.0};
    const int bi = canonicalPosition(cc.canonPos1, vertices[vi]->getId());
    const int bj = canonicalPosition(cc.canonPos1, vertices[vj]->getId());
    if (bi == 0 || bj == 0) return {0.0, 0.0};
    const DihedralCosine dihedral = dihedralCosine(cof, n, bi, bj, sheet);
    if (!dihedral.ok) return {0.0, 0.0};
    return dihedral.theta;
}

std::complex<double> Simplex::deficitAngle() const {
    return deficitAngle(DihedralSheets{});
}

std::complex<double> Simplex::deficitAngle(const DihedralSheets &sheets) const {
    using cd = std::complex<double>;
    const cd twoPi(2.0 * std::numbers::pi, 0.0);
    if (!spacetime || vertices.empty()) return twoPi;
    const int topSize =
        spacetime->getMetric()->getSignature()->getDimensions() + 1;
    (void)topSize;
    cd sum(0.0, 0.0);
    for (auto *sigma : incidentTopCells()) {
        const auto declared = sheets.find(sigma->topTuple());
        sum += sigma->dihedralAngle(
            const_cast<Simplex *>(this),
            declared == sheets.end() ? DihedralSheet{} : declared->second);
    }
    return twoPi - sum;
}

std::map<std::pair<std::uint64_t, std::uint64_t>, std::complex<double>>
Simplex::deficitAngleGradient() const {
    return deficitAngleGradient(DihedralSheets{});
}

std::map<std::pair<std::uint64_t, std::uint64_t>, std::complex<double>>
Simplex::deficitAngleGradient(const DihedralSheets &sheets) const {
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
        // local indices of the two vertices not in the hinge
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
        // Cached raw-order Cayley-Menger pipeline, shared by every hinge of tau and
        // by the Hessian below.
        const GeomCache &tc = tau->cmCache();
        const std::complex<double> detB = tc.cmDet;
        if (std::abs(detB) < 1e-300) continue;
        const std::vector<std::complex<double>> &C = tc.cmCof;
        const auto declared = sheets.find(tau->topTuple());
        const DihedralCosine dihedral = dihedralCosine(
            C, n, bi, bj,
            declared == sheets.end() ? DihedralSheet{} : declared->second);
        if (!dihedral.ok) continue;
        const std::vector<std::complex<double>> Binv =
            inverseFromCofactors(C, n, detB);
        const cd r = dihedral.r;
        const cd theta = dihedral.theta;
        const cd Cij = dihedral.Cij;
        const cd Cii = dihedral.Cii;
        const cd Cjj = dihedral.Cjj;
        const cd denom = dihedral.denom;
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
        // Same cached raw-order Cayley-Menger section as the gradient.
        const GeomCache &tc = tau->cmCache();
        const std::complex<double> detB = tc.cmDet;
        if (std::abs(detB) < 1e-300) continue;
        const std::vector<std::complex<double>> &C = tc.cmCof;
        const DihedralCosine dihedral = dihedralCosine(C, n, bi, bj);
        if (!dihedral.ok) continue;
        const std::vector<std::complex<double>> Binv =
            inverseFromCofactors(C, n, detB);
        // theta = acos(r) gives dtheta/dr = -1/sin(theta) and
        // d2theta/dr2 = -r/sin^3(theta).
        const cd r = dihedral.r;
        const cd theta = dihedral.theta;
        const cd Cij = dihedral.Cij;
        const cd Cii = dihedral.Cii;
        const cd Cjj = dihedral.Cjj;
        const cd denom = dihedral.denom;
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
    // Heron's radicand under a complex root. Clamping a non-positive radicand to 0
    // would report zero area for every timelike triangle (the mixed-causal hinge of a
    // causal dynamical triangulation (CDT) (4,1) cell, among others). Zero is not their
    // area; it is only what a double can represent.
    return std::sqrt(val) / 4.0;
}

std::complex<double> Simplex::volume() const {
    int d = static_cast<int>(vertices.size()) - 1;
    if (d < 1) return {0.0, 0.0};

    // Signature-respecting Gram matrix: timelike edges keep l^2 < 0, so det(G) can be
    // negative for a Lorentzian cell. Cached, because volume() is evaluated once per
    // facet per dual-volume recursion step.
    const GeomCache &gc = gramCache();
    if (static_cast<int>(gc.gram.size()) != d * d) return {0.0, 0.0};

    const std::complex<double> detG = gc.gramDet;
    double factorial = 1.0;
    for (int i = 2; i <= d; ++i) factorial *= static_cast<double>(i);

    // V = sqrt(det G)/d!, principal branch. Taking sqrt(|det G|) and hand-restoring
    // sign(det G) would be the same artifact as the dihedral parity fix: folding the
    // magnitude under the root discards a sign the complex root carries itself. A
    // Lorentzian cell with det G < 0 therefore returns an imaginary content, which is
    // its d-content, rather than the negative real a double can hold.
    return std::sqrt(detG) / factorial;
}

void Simplex::assertSpacelikeAdmissible(double tol) const {
    const int n = static_cast<int>(size());  // vertices = d + 1
    if (n < 2) return;                        // trivially admissible
    const int d = n - 1;

    // Skip simplices that contain any non-spacelike (null/timelike/worldline)
    // edge: their admissibility is Lorentzian, not the spacelike triangle
    // inequalities. Causal character comes from the canonical Edge classification
    // (Edge::isSpacelike), not from a hand-rolled sign-of-l^2 test.
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

// Roots here are principalSqrt (file scope, above), not sign(x)*sqrt(|x|): the latter
// is a real-valued convention that refuses to go imaginary, mapping a timelike
// circumcentric height to a negative real instead of the imaginary value it is.

// Circumcenter (barycentric) + signed R² from the Gram matrix G (flat d×d, relative to
// vertex 0) with its determinant and cofactors precomputed; the cached Gram sections
// enter here. Solves G β = ½·diag(G) Eigen-free via the adjugate (cofactorᵀ/det);
// λ_0 = 1−Σβ, λ_i = β_i; R² = Σ_i β_i·(½ G_ii).
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
    // This +/-1 is geometric, not a branch of a square root: it records which side of
    // the shared facet c(cf) fell on, and an obtuse cell needs the -1. Orientation is
    // not a function of edge lengths, so no complex root supplies it; dropping it would
    // silently switch the signed dual-volume convention to the unsigned overcount.
    //
    // Reading it off Re(bary) is bit-identical to the real-Lorentzian behaviour, since
    // bary is real there, and continues off-axis by continuity in Re. It is not settled
    // how this should generalise for a genuinely off-axis geometry.
    return (bary[static_cast<std::size_t>(oppIdx)].real() < 0.0) ? -1.0 : 1.0;
}

// The signed circumcentric height from c(s) to c(cf), s a facet of cf (defined with
// its derivatives below, at circumcentricHeight).
std::complex<double> circumcentricHeightValue(const ::tessera::mesh::Simplex* cf,
                                              const ::tessera::mesh::Simplex* s);

// Recursive signed circumcentric dual content of `s` in an n-complex.
std::complex<double> dualVolRec(const ::tessera::mesh::Simplex* s, int n) {
    const int k = static_cast<int>(s->size()) - 1;
    if (k >= n) return {1.0, 0.0};  // top cell: dual is a point (content 1)
    std::complex<double> acc{0.0, 0.0};
    for (const auto& cf : s->getCofaces())
        acc += circumcentricHeightValue(cf, s) * dualVolRec(cf, n);
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
    // Cached Gram pipeline: this runs once per (cell, edge) pair in the dual-volume
    // gradient, all against the same cell geometry.
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
    // Cached Gram pipeline, as in dCircumR2: one fill serves every (edge, edge) pair
    // of this cell's Hessian block.
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
    const GeomCache &gc = gramCofCache();
    circumFromGramCore(gc.gram, gc.gramDet, gc.gramCof, d, bary, r2);
    return bary;
}

std::complex<double> Simplex::circumradiusSquared() const {
    const int d = static_cast<int>(size()) - 1;
    std::vector<std::complex<double>> bary;
    std::complex<double> r2{0.0, 0.0};
    const GeomCache &gc = gramCofCache();
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
    // (V = sgn sqrt(|det G|)/d!, G linear in l^2 so dG_e is an indicator matrix, the
    // same dG dCircumR2 uses). G^-1 via the adjugate (cofactor^T/det),
    // Eigen-free, matching circumFromGram / volume().
    std::map<std::pair<std::uint64_t, std::uint64_t>, std::complex<double>> grad;
    const int d = static_cast<int>(size()) - 1;
    if (d < 1) return grad;
    const GeomCache &gc = gramCofCache();
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
    // Jacobi's formula differentiated a second time. G is linear in l^2, so the
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
/// Used only by the fallback of ``circumcentricHeight`` for a cell whose Gram
/// matrix is singular, where the circumcentre is undefined and the height is
/// read off the circumradius difference as before. An exactly zero numerator is
/// taken first so that a pinned radicand gives zero rather than `(1/0) * 0`.
[[nodiscard]] inline std::complex<double> rootChainRule(
        std::complex<double> radicandDerivative,
        std::complex<double> root) noexcept {
    if (isExactlyZero(radicandDerivative)) return {0.0, 0.0};
    return radicandDerivative / (2.0 * root);
}

/// A product in which an exactly-zero factor wins.
///
/// Used where a height that has vanished multiplies a derivative that may have
/// diverged in the singular-cell fallback: written plainly the product would be
/// `0 * inf` and evaluate to NaN. For finite factors it is the plain product.
[[nodiscard]] inline std::complex<double> productWithZeroAbsorbing(
        std::complex<double> a, std::complex<double> b) noexcept {
    if (isExactlyZero(a) || isExactlyZero(b)) return {0.0, 0.0};
    return a * b;
}

/// The signed circumcentric height from the circumcentre of a facet `s` to the
/// circumcentre of its coface `cf`, with its derivatives in the squared lengths
/// of the edges of `cf` (the height depends on no other edge).
///
/// The dual-volume recursion writes this height as
///     h = sgn(Re lambda_v) * sqrt(R^2_cf - R^2_s),
/// where v is the vertex of `cf` outside `s` and lambda_v the barycentric
/// coordinate of the circumcentre c(cf) at v. The radicand factors exactly. The
/// circumcentre c(cf) is equidistant from the vertices of `s`, so c(cf) - c(s)
/// is orthogonal to `s`; writing v = p + n with p in the affine hull of `s` and
/// n orthogonal to it gives c(cf) - c(s) = lambda_v n, hence
///     R^2_cf - R^2_s = lambda_v^2 <n, n>,   <n, n> = det G_cf / det G_s,
/// the second identity being the Schur complement of the Gram matrix of `cf` on
/// that of `s`. So h = sigma * lambda_v * q with q = sqrt(det G_cf / det G_s),
/// and sigma = +-1 the sign that makes this equal to the root above:
///     sigma = sgn(Re lambda_v) * (+1 if principalSqrt(y^2) = y, else -1),
///     y = lambda_v * q.
/// For real squared lengths of either signature sigma is identically +1.
///
/// This is the same function, not a different height. What changes is how it
/// is evaluated and differentiated near coincident circumcentres (lambda_v = 0:
/// a right angle opposite the face, as in every Kuhn tetrahedron of a cubic
/// lattice). There the difference R^2_cf - R^2_s cancels to rounding noise of
/// order 1e-16 R^2, whose root puts an error of order 1e-8 R into the height;
/// and differentiated through that root, the chain rule divides by
/// sqrt(R^2_cf - R^2_s), whose radicand is stationary to first order in every
/// direction, so the quotient is 0/0 and evaluates to NaN, inf or a spurious
/// zero, although the height is smooth there with derivative
/// sigma * q * d(lambda_v). The product form has neither defect: lambda_v is
/// rational in the squared lengths with denominator det G_cf, and q is the
/// root of a ratio of Gram determinants that stays away from zero on
/// nondegenerate cells. The dual volume, its gradient and its Hessian all read
/// the heights from here.
///
/// With P_e = G^-1 dG_e and beta = G_cf^-1 (diag G_cf / 2) (so lambda is beta
/// with lambda_0 = 1 - sum beta), and G linear in the squared lengths:
///     d beta_e     = G_cf^-1 dh_e - P_e beta,
///     d2 beta_ef   = -(P_e d beta_f + P_f d beta_e),
///     d log q_e    = (tr P_e[cf] - tr P_e[s]) / 2,
///     d2 log q_ef  = (-tr(P_f P_e)[cf] + tr(P_f P_e)[s]) / 2.
///
/// A singular Gram matrix (a cell of zero content) has no circumcentre and no
/// such factorization. There the height and its derivatives are taken, as
/// before, from the circumradius difference.
enum class HeightOrder { Value, Gradient, Hessian };

struct CircumcentricHeight {
    /// The edges of `cf`, as sorted vertex-id pairs (empty for HeightOrder::Value).
    std::vector<std::pair<std::uint64_t, std::uint64_t>> edges;
    /// The signed height.
    std::complex<double> value{0.0, 0.0};
    /// d h / d l^2, one entry per edge of `edges`.
    std::vector<std::complex<double>> gradient;
    /// d2 h / d l^2 d l^2, edges x edges row-major; filled only on request.
    std::vector<std::complex<double>> hessian;
};

[[nodiscard]] CircumcentricHeight circumcentricHeight(const Simplex *cf,
                                                      const Simplex *s,
                                                      HeightOrder order) {
    using cd = std::complex<double>;
    CircumcentricHeight out;
    const auto &cv = cf->getVertices();
    const auto &sv = s->getVertices();
    const int m = static_cast<int>(cv.size()) - 1;   // dimension of cf
    const int ms = static_cast<int>(sv.size()) - 1;  // dimension of s (m - 1)
    const bool withGradient = order != HeightOrder::Value;
    const bool withHessian = order == HeightOrder::Hessian;

    std::vector<std::pair<int, int>> local;          // cf-local vertex pairs
    if (withGradient)
        for (int p = 0; p <= m; ++p)
            for (int q = p + 1; q <= m; ++q) {
                local.emplace_back(p, q);
                const std::uint64_t a = cv[static_cast<std::size_t>(p)]->getId();
                const std::uint64_t b = cv[static_cast<std::size_t>(q)]->getId();
                out.edges.emplace_back(std::min(a, b), std::max(a, b));
            }
    const std::size_t nE = local.size();
    out.gradient.assign(nE, cd{0.0, 0.0});
    if (withHessian) out.hessian.assign(nE * nE, cd{0.0, 0.0});
    if (m < 1 || ms != m - 1) return out;

    // The vertex of cf outside s, and where each vertex of cf sits in s.
    int opp = -1;
    std::vector<int> inS(static_cast<std::size_t>(m) + 1, -1);
    for (int i = 0; i <= m; ++i) {
        for (int j = 0; j <= ms; ++j)
            if (sv[static_cast<std::size_t>(j)]->getId() ==
                cv[static_cast<std::size_t>(i)]->getId()) {
                inS[static_cast<std::size_t>(i)] = j;
                break;
            }
        if (inS[static_cast<std::size_t>(i)] < 0 && opp < 0) opp = i;
    }
    if (opp < 0) return out;

    const Simplex::GeomCache &gcf = cf->gramCofCache();
    const cd detCf = gcf.gramDet;
    const std::size_t mm = static_cast<std::size_t>(m);
    const std::size_t msz = static_cast<std::size_t>(ms);
    bool singular = gcf.gram.size() != mm * mm || gcf.gramCof.size() != mm * mm ||
                    std::abs(detCf) < 1e-300;
    // A vertex (ms = 0) has the empty Gram matrix, of determinant one.
    cd detS{1.0, 0.0};
    std::vector<cd> ginvS;
    if (!singular && ms >= 1) {
        const Simplex::GeomCache &gs = s->gramCofCache();
        detS = gs.gramDet;
        if (gs.gram.size() != msz * msz || gs.gramCof.size() != msz * msz ||
            std::abs(detS) < 1e-300) {
            singular = true;
        } else if (withGradient) {
            ginvS.resize(msz * msz);
            for (std::size_t i = 0; i < msz; ++i)
                for (std::size_t j = 0; j < msz; ++j)
                    ginvS[i * msz + j] = gs.gramCof[j * msz + i] / detS;
        }
    }

    if (singular) {
        // No circumcentre: the circumradius difference, as the recursion reads it.
        const double sgn = oppositeVertexSign(cf, s);
        const cd x = cf->circumradiusSquared() - s->circumradiusSquared();
        const cd root = principalSqrt(x);
        out.value = sgn * root;
        if (!withGradient) return out;
        std::vector<cd> dx(nE);
        for (std::size_t e = 0; e < nE; ++e) {
            const auto &[a, b] = out.edges[e];
            dx[e] = dCircumR2(cf, a, b) - dCircumR2(s, a, b);
            out.gradient[e] = sgn * rootChainRule(dx[e], root);
        }
        if (withHessian)
            for (std::size_t e = 0; e < nE; ++e)
                for (std::size_t f = 0; f < nE; ++f) {
                    const auto &[a, b] = out.edges[e];
                    const auto &[c, d] = out.edges[f];
                    const cd d2x = d2CircumR2(cf, a, b, c, d) - d2CircumR2(s, a, b, c, d);
                    cd second{0.0, 0.0};
                    if (!isExactlyZero(dx[e]) && !isExactlyZero(dx[f]))
                        second = -0.25 / (x * root) * dx[e] * dx[f];
                    out.hessian[e * nE + f] = sgn * (second + rootChainRule(d2x, root));
                }
        return out;
    }

    // beta = G_cf^-1 (diag G_cf / 2), with G_cf^-1 = cof^T / det, accumulated as
    // circumFromGramCore does so that the barycentric coordinates agree with
    // circumcenterBarycentric() bit for bit.
    std::vector<cd> halfDiag(mm), beta(mm, cd{0.0, 0.0});
    for (std::size_t i = 0; i < mm; ++i) halfDiag[i] = 0.5 * gcf.gram[i * mm + i];
    for (std::size_t i = 0; i < mm; ++i) {
        cd acc{0.0, 0.0};
        for (std::size_t j = 0; j < mm; ++j) acc += gcf.gramCof[j * mm + i] * halfDiag[j];
        beta[i] = acc / detCf;
    }
    const auto lambdaOf = [&](const std::vector<cd> &b) -> cd {
        if (opp > 0) return b[static_cast<std::size_t>(opp) - 1];
        cd sum{0.0, 0.0};
        for (const cd &bi : b) sum += bi;
        return cd{1.0, 0.0} - sum;
    };
    // For the derivative of the constant 1 in lambda_0, differentiate the sum only.
    const auto dLambdaOf = [&](const std::vector<cd> &db) -> cd {
        if (opp > 0) return db[static_cast<std::size_t>(opp) - 1];
        cd sum{0.0, 0.0};
        for (const cd &bi : db) sum -= bi;
        return sum;
    };
    const cd lambda = lambdaOf(beta);
    const cd q = principalSqrt(detCf / detS);
    const cd y = lambda * q;
    // sigma = sgn(Re lambda) * (the sign with principalSqrt(y^2) = sign * y), both
    // read off this lambda so that they agree bit for bit where lambda ~ 0.
    const double sgn = (lambda.real() < 0.0) ? -1.0 : 1.0;
    const double rootSign = (y.real() > 0.0 || (y.real() == 0.0 && y.imag() >= 0.0))
                                ? 1.0 : -1.0;
    const double sigma = sgn * rootSign;
    out.value = sigma * y;
    if (!withGradient) return out;

    std::vector<cd> ginv(mm * mm);
    for (std::size_t i = 0; i < mm; ++i)
        for (std::size_t j = 0; j < mm; ++j)
            ginv[i * mm + j] = gcf.gramCof[j * mm + i] / detCf;

    // Per edge: P_e on cf and on s, d beta_e, d lambda_e, d log q_e.
    std::vector<std::vector<cd>> Pcf(nE), Ps(nE), dBeta(nE);
    std::vector<cd> dLambda(nE), dLogQ(nE);
    const auto gramDerivative = [](int dim, int p, int q2, std::vector<cd> &dG) {
        // G_ij = (l^2_{0,i+1} + l^2_{0,j+1} - l^2_{i+1,j+1}) / 2 in local indices.
        const auto ind = [&](int x, int z) -> double {
            return ((x == p && z == q2) || (x == q2 && z == p)) ? 1.0 : 0.0;
        };
        const std::size_t n = static_cast<std::size_t>(dim);
        dG.assign(n * n, cd{0.0, 0.0});
        for (int i = 0; i < dim; ++i)
            for (int j = 0; j < dim; ++j)
                dG[static_cast<std::size_t>(i) * n + static_cast<std::size_t>(j)] =
                    0.5 * (ind(0, i + 1) + ind(0, j + 1) - ind(i + 1, j + 1));
    };
    const auto product = [](const std::vector<cd> &A, const std::vector<cd> &B,
                            std::size_t n) {
        std::vector<cd> C(n * n, cd{0.0, 0.0});
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t k = 0; k < n; ++k) {
                const cd a = A[i * n + k];
                if (isExactlyZero(a)) continue;
                for (std::size_t j = 0; j < n; ++j) C[i * n + j] += a * B[k * n + j];
            }
        return C;
    };
    const auto trace = [](const std::vector<cd> &A, std::size_t n) {
        cd t{0.0, 0.0};
        for (std::size_t i = 0; i < n; ++i) t += A[i * n + i];
        return t;
    };
    const auto traceOfProduct = [](const std::vector<cd> &A, const std::vector<cd> &B,
                                   std::size_t n) {
        cd t{0.0, 0.0};
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j) t += A[i * n + j] * B[j * n + i];
        return t;
    };
    std::vector<cd> dG;
    for (std::size_t e = 0; e < nE; ++e) {
        const auto [p, pq] = local[e];
        gramDerivative(m, p, pq, dG);
        Pcf[e] = product(ginv, dG, mm);
        std::vector<cd> db(mm, cd{0.0, 0.0});
        for (std::size_t i = 0; i < mm; ++i) {
            cd acc{0.0, 0.0};
            for (std::size_t j = 0; j < mm; ++j) {
                acc += ginv[i * mm + j] * (0.5 * dG[j * mm + j]);
                acc -= Pcf[e][i * mm + j] * beta[j];
            }
            db[i] = acc;
        }
        dBeta[e] = std::move(db);
        dLambda[e] = dLambdaOf(dBeta[e]);
        cd traceS{0.0, 0.0};
        const int ps = inS[static_cast<std::size_t>(p)];
        const int qs = inS[static_cast<std::size_t>(pq)];
        if (ms >= 1 && ps >= 0 && qs >= 0) {
            gramDerivative(ms, ps, qs, dG);
            Ps[e] = product(ginvS, dG, msz);
            traceS = trace(Ps[e], msz);
        }
        dLogQ[e] = 0.5 * (trace(Pcf[e], mm) - traceS);
        out.gradient[e] = sigma * (dLambda[e] * q + lambda * q * dLogQ[e]);
    }
    if (!withHessian) return out;

    for (std::size_t e = 0; e < nE; ++e)
        for (std::size_t f = e; f < nE; ++f) {
            // d2 beta_ef = -(P_e d beta_f + P_f d beta_e).
            std::vector<cd> d2b(mm, cd{0.0, 0.0});
            for (std::size_t i = 0; i < mm; ++i)
                for (std::size_t j = 0; j < mm; ++j)
                    d2b[i] -= Pcf[e][i * mm + j] * dBeta[f][j] +
                              Pcf[f][i * mm + j] * dBeta[e][j];
            const cd d2Lambda = dLambdaOf(d2b);
            cd d2LogQ = -0.5 * traceOfProduct(Pcf[f], Pcf[e], mm);
            if (!Ps[e].empty() && !Ps[f].empty())
                d2LogQ += 0.5 * traceOfProduct(Ps[f], Ps[e], msz);
            // q'' = q (d log q_e d log q_f + d2 log q_ef).
            const cd dQe = q * dLogQ[e], dQf = q * dLogQ[f];
            const cd d2Q = q * (dLogQ[e] * dLogQ[f] + d2LogQ);
            const cd value =
                sigma * (d2Lambda * q + dLambda[e] * dQf + dLambda[f] * dQe + lambda * d2Q);
            out.hessian[e * nE + f] = value;
            out.hessian[f * nE + e] = value;
        }
    return out;
}

std::complex<double> circumcentricHeightValue(const Simplex *cf, const Simplex *s) {
    return circumcentricHeight(cf, s, HeightOrder::Value).value;
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
    using cd = std::complex<double>;
    using EK = std::pair<std::uint64_t, std::uint64_t>;
    std::map<EK, cd> grad;
    if (vertices.empty()) return grad;
    const int n = ambientTopDimension();
    const int k = static_cast<int>(size()) - 1;
    if (k != n - 2) return grad;          // the (n-2) hinge the Regge action needs

    // |*h| = inv * sum_facets h1 * S,  S = sum_tops h2, with h1 the height from
    // c(hinge) to c(facet) and h2 from c(facet) to c(top) (circumcentricHeight):
    // d|*h| = inv * sum_facets (dh1 * S + h1 * dS).
    const double inv = 1.0 / (static_cast<double>(n - k) * (n - k - 1));
    std::map<EK, cd> dV;
    std::set<EK> edges;                   // the star: every edge of every top
    for (const auto& cf : getCofaces()) {
        const CircumcentricHeight h1 = circumcentricHeight(cf, this, HeightOrder::Gradient);
        cd S{0.0, 0.0};
        std::map<EK, cd> dS;
        for (const auto& tp : cf->getCofaces()) {
            const CircumcentricHeight h2 = circumcentricHeight(tp, cf, HeightOrder::Gradient);
            S += h2.value;
            for (std::size_t i = 0; i < h2.edges.size(); ++i) {
                dS[h2.edges[i]] += h2.gradient[i];
                edges.insert(h2.edges[i]);
            }
        }
        for (std::size_t i = 0; i < h1.edges.size(); ++i)
            dV[h1.edges[i]] += productWithZeroAbsorbing(h1.gradient[i], S);
        for (const auto& [e, d] : dS) dV[e] += productWithZeroAbsorbing(h1.value, d);
    }
    for (const auto& e : edges) {
        const auto it = dV.find(e);
        grad[e] = (it != dV.end() ? it->second : cd{0.0, 0.0}) * inv;
    }
    return grad;
}

std::map<std::pair<std::pair<std::uint64_t, std::uint64_t>,
                   std::pair<std::uint64_t, std::uint64_t>>,
         std::complex<double>>
Simplex::dualVolumeHessian() const {
    using cd = std::complex<double>;
    using EK = std::pair<std::uint64_t, std::uint64_t>;
    std::map<std::pair<EK, EK>, cd> hess;
    if (vertices.empty()) return hess;
    const int n = ambientTopDimension();
    const int k = static_cast<int>(size()) - 1;
    if (k != n - 2) return hess;

    // The star's edges, indexed, so the per-facet pieces assemble densely.
    std::set<EK> edgeSet;
    for (const auto& cf : getCofaces())
        for (const auto& tp : cf->getCofaces()) {
            const auto& tv = tp->getVertices();
            for (std::size_t i = 0; i < tv.size(); ++i)
                for (std::size_t j = i + 1; j < tv.size(); ++j) {
                    const std::uint64_t a = tv[i]->getId(), b = tv[j]->getId();
                    edgeSet.insert({std::min(a, b), std::max(a, b)});
                }
        }
    const std::vector<EK> ev(edgeSet.begin(), edgeSet.end());
    const std::size_t nE = ev.size();
    std::map<EK, std::size_t> index;
    for (std::size_t i = 0; i < nE; ++i) index.emplace(ev[i], i);
    const auto slots = [&](const CircumcentricHeight& h) {
        std::vector<std::ptrdiff_t> at(h.edges.size(), -1);
        for (std::size_t i = 0; i < h.edges.size(); ++i) {
            const auto it = index.find(h.edges[i]);
            if (it != index.end()) at[i] = static_cast<std::ptrdiff_t>(it->second);
        }
        return at;
    };

    // d2|*h|_ef = inv * sum_facets (d2h1_ef S + dh1_e dS_f + dh1_f dS_e + h1 d2S_ef),
    // each height smooth where circumcentres coincide (circumcentricHeight).
    const double inv = 1.0 / (static_cast<double>(n - k) * (n - k - 1));
    std::vector<cd> d2V(nE * nE, cd{0.0, 0.0});
    for (const auto& cf : getCofaces()) {
        const CircumcentricHeight h1 = circumcentricHeight(cf, this, HeightOrder::Hessian);
        cd S{0.0, 0.0};
        std::vector<cd> dS(nE, cd{0.0, 0.0}), d2S(nE * nE, cd{0.0, 0.0});
        for (const auto& tp : cf->getCofaces()) {
            const CircumcentricHeight h2 = circumcentricHeight(tp, cf, HeightOrder::Hessian);
            const auto at = slots(h2);
            const std::size_t m2 = h2.edges.size();
            S += h2.value;
            for (std::size_t i = 0; i < m2; ++i) {
                if (at[i] < 0) continue;
                const auto a = static_cast<std::size_t>(at[i]);
                dS[a] += h2.gradient[i];
                for (std::size_t j = 0; j < m2; ++j)
                    if (at[j] >= 0)
                        d2S[a * nE + static_cast<std::size_t>(at[j])] +=
                            h2.hessian[i * m2 + j];
            }
        }
        const auto at = slots(h1);
        const std::size_t m1 = h1.edges.size();
        std::vector<cd> g1(nE, cd{0.0, 0.0}), H1(nE * nE, cd{0.0, 0.0});
        for (std::size_t i = 0; i < m1; ++i) {
            if (at[i] < 0) continue;
            const auto a = static_cast<std::size_t>(at[i]);
            g1[a] = h1.gradient[i];
            for (std::size_t j = 0; j < m1; ++j)
                if (at[j] >= 0)
                    H1[a * nE + static_cast<std::size_t>(at[j])] = h1.hessian[i * m1 + j];
        }
        for (std::size_t e = 0; e < nE; ++e)
            for (std::size_t f = 0; f < nE; ++f)
                d2V[e * nE + f] += productWithZeroAbsorbing(H1[e * nE + f], S)
                                   + productWithZeroAbsorbing(g1[e], dS[f])
                                   + productWithZeroAbsorbing(g1[f], dS[e])
                                   + productWithZeroAbsorbing(h1.value, d2S[e * nE + f]);
    }
    for (std::size_t e = 0; e < nE; ++e)
        for (std::size_t f = 0; f < nE; ++f)
            hess[{ev[e], ev[f]}] = d2V[e * nE + f] * inv;
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
