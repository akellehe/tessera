// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_PACHNERMOVE_H
#define TESSERA_PACHNERMOVE_H

#include <algorithm>
#include <cstdint>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

#include "mesh/ForwardDeclarations.h"
#include "mesh/Simplex.h"
#include "mesh/TemporalOrientation.h"
#include "mesh/Vertex.h"
#include "spacetime/Spacetime.h"

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

namespace pachner_detail {

/// Check that a proposed simplex vertex set has a valid CDT
/// orientation — one of (d,1), (1,d), (d-1,2), (2,d-1) — and spans
/// exactly 2 time slices.  Mirrors the static helper in CDT.cpp; lives
/// here so the move classes can share it.
inline bool isValidCDTOrientation(const VertexPtrs &verts, int d) {
  std::unordered_set<std::uint64_t> times;
  for (const auto &v : verts) {
    times.insert(static_cast<std::uint64_t>(v->getTime()));
  }
  if (times.size() != 2) return false;
  auto orient = TemporalOrientation::orientationOf(verts);
  auto [ti, tf] = orient.numeric();
  if ((ti == d && tf == 1) || (ti == 1 && tf == d)) return true;
  if ((ti == d - 1 && tf == 2) || (ti == 2 && tf == d - 1)) return true;
  return false;
}

inline bool isN41Type(const SimplexPtr &s, int d) {
  auto [ti, tf] = s->getOrientation().numeric();
  return (ti == d && tf == 1) || (ti == 1 && tf == d);
}

inline bool isN32Type(const SimplexPtr &s, int d) {
  auto [ti, tf] = s->getOrientation().numeric();
  return (ti == d - 1 && tf == 2) || (ti == 2 && tf == d - 1);
}

inline bool isN41TypeVerts(const VertexPtrs &verts, int d) {
  auto [ti, tf] = TemporalOrientation::orientationOf(verts).numeric();
  return (ti == d && tf == 1) || (ti == 1 && tf == d);
}

inline bool isN32TypeVerts(const VertexPtrs &verts, int d) {
  auto [ti, tf] = TemporalOrientation::orientationOf(verts).numeric();
  return (ti == d - 1 && tf == 2) || (ti == 2 && tf == d - 1);
}

inline int spacetimeDim(const Spacetime &st) {
  return st.getMetric()->getSignature()->getDimensions();
}

/// Detach every edge in ``edges`` from its endpoints, remove it from
/// the spacetime's EdgeList, then clear the container. Used by all
/// five Pachner moves at the end of ``rollback()`` to undo edges that
/// ``apply()`` freshly inserted.
inline void removeAndClearEdges(Edges &edges, Spacetime *st) {
  for (const auto &e : edges) {
    e->getSource()->removeOutEdge(e);
    e->getTarget()->removeInEdge(e);
    st->absorbRemovedEdgeRevisions(e);   // #692: keep metricRevisionKey monotone
    st->getEdgeList()->remove(e);
  }
  edges.clear();
}

/// Order-preserving union of every simplex's vertex list,
/// de-duplicated by vertex ID. Used by FlipMove / IFlipMove /
/// ShiftMove during ``propose()`` to build the (d+2)-vertex span of
/// adjacent simplices before checking the orientation constraint.
/// Hash-set dedup makes this O(n) over the total vertex count instead
/// of the inlined O(n²) loop the move classes used to carry.
template <typename SimplexRange>
inline VertexPtrs unionVerticesAcross(SimplexRange const &simplices) {
  VertexPtrs out;
  std::unordered_set<std::uint64_t> seen;
  for (const auto &s : simplices) {
    for (const auto &v : s->getVertices()) {
      if (seen.insert(v->getId()).second) out.push_back(v);
    }
  }
  return out;
}

// ===================================================================
// Pre-geometric / boundary-fixed helpers.
//
// These read the incidence structure straight off the vertices'
// simplex lists, so they work on a *pre-geometric* complex (one built
// combinatorially via ``Topology::buildExplicit``, where facet/coface
// caches are not pre-materialised and the metric dimension may differ
// from the manifold dimension).  They are also coface-cache-independent
// in the CDT case, but the CDT move paths deliberately keep using the
// cached ``getCofaces`` walk so their behaviour is byte-identical.
// ===================================================================

/// Every top-dimensional simplex (``topVerts`` vertices) that contains
/// all of ``verts``.  Found by scanning the simplex list of ``verts``'
/// first vertex — no facet/coface materialisation required.
inline std::vector<SimplexPtr> topCofacesOf(const VertexPtrs &verts,
                                            int topVerts) {
  std::vector<SimplexPtr> out;
  if (verts.empty()) return out;
  for (const auto &s : verts.front()->getSimplices()) {
    if (static_cast<int>(s->size()) != topVerts) continue;
    bool all = true;
    for (std::size_t i = 1; i < verts.size(); ++i) {
      if (!s->hasVertex(verts[i])) { all = false; break; }
    }
    if (all) out.push_back(s);
  }
  return out;
}

/// Number of top simplices sharing the (codim-1) face ``facetVerts``.
inline int topCofaceCount(const VertexPtrs &facetVerts, int topVerts) {
  return static_cast<int>(topCofacesOf(facetVerts, topVerts).size());
}

/// A codimension-1 face is on the boundary ``∂W`` iff exactly one top
/// cell contains it; an interior face is shared by exactly two.
inline bool isBoundaryFacet(const VertexPtrs &facetVerts, int topVerts) {
  return topCofaceCount(facetVerts, topVerts) == 1;
}

/// Order a cell's vertices by ascending id.  ``cobordism::ChainComplex``
/// (and every coordinate-free fixture) takes the increasing-vertex-id
/// ordering as each simplex's reference orientation, so the simplicial
/// boundary signs (facet ``i`` ↦ ``(-1)^i``) glue consistently and
/// ``∂² = 0`` holds globally.  Pre-geometric moves build new cells from
/// mixed (shared/unique) vertex lists, so they must re-sort before
/// committing or the homology of the mutated complex is corrupted.
inline void sortByVertexId(VertexPtrs &verts) {
  std::sort(verts.begin(), verts.end(),
            [](const VertexPtr &a, const VertexPtr &b) {
              return a->getId() < b->getId();
            });
}

/// True iff vertices ``a`` and ``b`` already co-occur in some simplex
/// (i.e. the edge ``a–b`` already exists).  Used to reject a 2→(d+1)
/// flip whose new apex edge would collide with existing structure.
inline bool verticesAdjacent(const VertexPtr &a, const VertexPtr &b) {
  for (const auto &s : a->getSimplices()) {
    if (s->hasVertex(b)) return true;
  }
  return false;
}

/// True iff the edge ``a–b`` is interior: no boundary facet contains
/// both endpoints.  An interior-only (boundary-fixed) ``d→2`` flip
/// must not collapse an edge that lies on ``∂W``.
inline bool isInteriorEdge(const VertexPtr &a, const VertexPtr &b,
                           int topVerts) {
  for (const auto &s : a->getSimplices()) {
    if (static_cast<int>(s->size()) != topVerts) continue;
    if (!s->hasVertex(b)) continue;
    const auto &sv = s->getVertices();
    for (std::size_t skip = 0; skip < sv.size(); ++skip) {
      VertexPtrs facetVerts;
      facetVerts.reserve(sv.size() - 1);
      bool hasA = false, hasB = false;
      for (std::size_t i = 0; i < sv.size(); ++i) {
        if (i == skip) continue;
        facetVerts.push_back(sv[i]);
        if (sv[i]->getId() == a->getId()) hasA = true;
        if (sv[i]->getId() == b->getId()) hasB = true;
      }
      if (hasA && hasB && isBoundaryFacet(facetVerts, topVerts)) return false;
    }
  }
  return true;
}

/// True iff vertex ``v`` is interior: none of the codim-1 faces
/// incident to it lies on ``∂W``.  An interior-only (boundary-fixed)
/// ``(d+1)→1`` move must only delete interior vertices.
inline bool isInteriorVertex(const VertexPtr &v, int topVerts) {
  for (const auto &s : v->getSimplices()) {
    if (static_cast<int>(s->size()) != topVerts) continue;
    const auto &sv = s->getVertices();
    for (std::size_t skip = 0; skip < sv.size(); ++skip) {
      if (sv[skip]->getId() == v->getId()) continue;  // facet would drop v
      VertexPtrs facetVerts;
      facetVerts.reserve(sv.size() - 1);
      for (std::size_t i = 0; i < sv.size(); ++i) {
        if (i != skip) facetVerts.push_back(sv[i]);
      }
      if (isBoundaryFacet(facetVerts, topVerts)) return false;
    }
  }
  return true;
}

}  // namespace pachner_detail

/// Validity regime a :class:`PachnerMove` runs under.
///
/// * ``CDT`` — the original causal-dynamical-triangulations path: every
///   proposed cell must satisfy the time-sliced CDT orientation
///   constraint (``pachner_detail::isValidCDTOrientation``) and the
///   move dimension comes from the metric signature.  This path is left
///   byte-identical to the pre-#112 behaviour.
/// * ``PreGeometric`` — the CDT orientation/time-slice guards are
///   dropped so the bistellar moves run on a coordinate-free simplicial
///   complex (e.g. a ``SimplicialProduct`` fixture).  The move dimension
///   is read off the actual top cell rather than the metric, and a
///   manifold-preservation check stands in for the orientation guard.
enum class PachnerMode : std::uint8_t { CDT = 0, PreGeometric = 1 };


/// Transactional Pachner move: a propose-apply-rollback wrapper around
/// the geometric mutations of CDT::add / remove / flip / iflip / shift.
///
/// The interface separates a move's three life-cycle phases:
///
///   1. ``propose()`` — read-only.  Selects a target (random simplex /
///      facet / edge / vertex), validates that the move can be applied
///      to it, and records all the data needed to commit.  Does not
///      mutate the spacetime.  Returns ``false`` if no eligible target
///      can be found within the move's retry budget.
///
///   2. ``apply()`` — mutating.  Commits the proposed move to the
///      spacetime.  Builds an internal undo log (created simplices,
///      freshly-inserted edges, removed simplex vertex tuples, etc.)
///      that ``rollback()`` consumes.  Returns ``true`` on success.
///
///   3. ``rollback()`` — mutating.  Replays the undo log in reverse,
///      restoring the spacetime to the byte-identical state it was in
///      before ``apply()``.  Idempotent: a second call is a no-op.
///
/// The combinatorial Δ in (N0, N41, N32) is published by
/// ``dN0() / dN41() / dN32()`` after a successful ``propose()``, so the
/// caller (typically ``CDT::add()`` / etc.) can plug those into its
/// own action computation.  The base class deliberately *does not*
/// compute ΔS — the move is purely about geometry, not the action it
/// happens to be sampled against.
///
/// Locked-in characterization (see
/// docs/source/modularity-plan.md, "Discoveries from the safety-net pass"):
///
/// * Edges added by ``apply()`` are recorded by EdgePtr identity (not
///   by fingerprint hash, which is unstable under
///   ``swapVertexLabels``).
/// * ``apply()`` does not force facet/coface registration on newly
///   created simplices.  Coface registration is tessera's lazy
///   responsibility (triggered on the next ``getFacets`` walk).
/// * ``rollback()`` for moves that delete edges
///   (``RemoveMove``) must capture the deleted edges'
///   ``(sourceId, targetId, squaredLength)`` so it can reinsert them.
class PachnerMove {
public:
  virtual ~PachnerMove() = default;

  /// The validity regime this move runs under (CDT vs. pre-geometric).
  /// Defaults to :enumerator:`PachnerMode::CDT` so existing callers and
  /// the CDT Markov chain are unaffected.
  PachnerMode mode() const { return mode_; }

  /// True iff the move is restricted to the interior of the complex:
  /// it only fires when it leaves the boundary face-set ``∂W``
  /// (codim-1 faces in exactly one top cell) unchanged.
  bool boundaryFixed() const { return boundaryFixed_; }

  /// Pick a target and validate it.  No spacetime mutation.
  /// Returns ``true`` on success, ``false`` if no eligible target.
  virtual bool propose() = 0;

  /// Propose at a NAMED site instead of a drawn one, so a caller can walk the
  /// move space instead of sampling it.
  ///
  /// ``propose()`` draws its target, which makes the move space impossible to
  /// enumerate and, for the subclasses whose draw reads ``Spacetime::rng``
  /// (seeded from ``std::random_device``), impossible to reproduce from any
  /// seed.  ``proposeAt`` takes the same decision from its argument.  On
  /// success the object is in exactly the state a successful ``propose()``
  /// leaves it in, so ``apply()``, ``rollback()`` and the ``dN*`` counters
  /// behave identically; the two differ only in how the target was chosen.
  ///
  /// The site encoding is the subclass's own, documented on each override and
  /// produced by its ``sitesOn``.  Returns ``false`` when the site does not
  /// exist or is not eligible, exactly as ``propose()`` does when it finds no
  /// eligible target.
  ///
  /// Defaults to ``false``: a subclass that has not opted in is simply not
  /// enumerable, and says so rather than silently proposing something else.
  virtual bool proposeAt(const std::vector<std::uint64_t> & /*site*/) {
    return false;
  }

  /// The top simplex whose vertex ids are exactly \p site, or nullptr.
  ///
  /// A linear scan of the top cells rather than a lookup: every enumerable
  /// site is itself a top cell, so a walk of the move space is already
  /// O(cells) and this keeps each subclass from re-deriving the same match.
  /// Compares as a SET, because a site is named by which vertices it has and
  /// never by the order they happen to be stored in.
  static SimplexPtr topSimplexWithIds(const Spacetime &spacetime,
                                      const std::vector<std::uint64_t> &site) {
    if (site.empty()) return nullptr;
    std::vector<std::uint64_t> wanted(site);
    std::sort(wanted.begin(), wanted.end());
    for (const auto &topSimplex : spacetime.getTopSimplices()) {
      if (!topSimplex || topSimplex->size() != site.size()) continue;
      std::vector<std::uint64_t> have;
      have.reserve(topSimplex->size());
      for (const auto &vertex : topSimplex->getVertices())
        if (vertex) have.push_back(vertex->getId());
      std::sort(have.begin(), have.end());
      if (have == wanted) return topSimplex;
    }
    return nullptr;
  }

  /// Combinatorial change in vertex count if this move is applied.
  /// Valid only after a successful ``propose()``.
  virtual int dN0() const = 0;
  /// Combinatorial change in N41-type top-simplex count.
  virtual int dN41() const = 0;
  /// Combinatorial change in N32-type top-simplex count.
  virtual int dN32() const = 0;

  /// Log of the Metropolis combinatorial prefactor
  /// ``log(g(T'→T) · P_l(T') / [g(T→T') · P_l(T)])`` ([BGL] eq. 26).
  /// 0.0 for self-symmetric moves (shift) and the (2,d)/(d,2)
  /// flips; non-trivial for add/remove because of vertex selection.
  /// Valid only after a successful ``propose()``.
  virtual double metropolisLogPrefactor() const = 0;

  /// Commit the proposed move.  Builds the undo log.
  /// Must be called at most once per object.  Returns ``true`` on
  /// success.
  virtual bool apply() = 0;

  /// Replay the undo log in reverse.  After this returns,
  /// the spacetime is byte-identical to its state before ``apply()``.
  /// Idempotent.
  virtual void rollback() = 0;

  /// True iff ``apply()`` has been called and not yet rolled back.
  virtual bool isApplied() const = 0;

  /// IDs of the vertices whose neighborhood the move
  /// re-arranges, for informed proposal scoring (community-aware
  /// modularity sweeps).  Valid after a successful ``propose()``.
  virtual std::vector<std::uint64_t> touchedVertexIds() const = 0;

  /// Move-type tag for logging / acceptance-rate accounting.
  /// One of: ``"add"``, ``"remove"``, ``"flip"``, ``"iflip"``,
  /// ``"shift"``.
  virtual std::string moveType() const = 0;

protected:
  /// CDT-mode move (the default; preserves the pre-#112 behaviour).
  PachnerMove() = default;
  /// Move configured with an explicit validity regime / boundary policy.
  PachnerMove(PachnerMode mode, bool boundaryFixed)
      : mode_(mode), boundaryFixed_(boundaryFixed) {}

  PachnerMode mode_ = PachnerMode::CDT;
  bool boundaryFixed_ = false;
};

}  // namespace tessera

#endif  // TESSERA_PACHNERMOVE_H
