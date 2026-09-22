// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_TESSERA_SRC_EDGE_H_
#define TESSERA_TESSERA_SRC_EDGE_H_

#include "mesh/Fingerprint.h"
#include "mesh/ForwardDeclarations.h"
#include "mesh/EdgeKey.h"
// walkLoop calls Vertex::getId() non-dependently, so Vertex must be complete at
// its definition. Vertex.h only forward-declares Edge, so this include is acyclic.
#include "mesh/Vertex.h"

#include <complex>
#include <random>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <cstdint>


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
/// # Edge disposition
///
/// Two things fix an edge's disposition (spacelike, timelike, lightlike). The first is
/// the squared edge length: \f$ l^2 < 0 \f$ is timelike in a (-, +, +, +) signature and
/// spacelike in a (+, -, -, -) one; \f$ l^2 = 0 \f$ is lightlike in either.
///
/// The second is whether the endpoints sit at the same time (spacelike, within a spatial
/// slice) or at different times (timelike, crossing between slices). Causal dynamical
/// triangulation (CDT) has no lightlike edges.
///
/// Reference: Loll, "Quantum gravity from causal dynamical triangulations: a review", 2019.
enum class EdgeDisposition : uint8_t {
  Spacelike = 0,
  Timelike = 1,
  Lightlike = 2,
  /// A genuinely complex \f$ l^2 \f$ — an argument at none of the three definite
  /// values. The honest reading of an edge that has not acquired a causal
  /// character, and the common case for a randomly seeded complex.
  Mixed = 3,
  /// An absent edge, \f$ l = 0 \f$. Not a causal type.
  Degenerate = 4
};


/// # Edge
///
/// An edge linking two vertices in spacetime. Merging two vertices requires removing the
/// affected edges from their containers first: mutating an edge in place changes its
/// fingerprint and leaves it in a stale hash bucket, which is undefined behaviour. Edges
/// have far higher cardinality than vertices, so state is kept on the Vertex wherever
/// there is a choice.
///
/// ``source`` and ``target`` are the endpoints. Read as directed, the edge runs from
/// source to target; read as undirected, the order is arbitrary. ``length_`` is the
/// complex length under whichever spacetime metric is in use — real for spacelike,
/// imaginary for timelike. The squared length is derived by squaring and never stored.
///
class Edge {
  public:
    /// Construct from the (possibly complex) length \f$l\f$ — real for spacelike,
    /// imaginary for timelike, general complex off the real-Lorentzian locus. This is
    /// the edge's one degree of freedom; \f$l^2\f$ is derived by squaring, never stored.
    /// A caller holding an \f$l^2\f$ passes ``std::sqrt(l2)`` and so chooses the branch
    /// explicitly.
    Edge(
      const VertexPtr &source,
      const VertexPtr &target,
      std::complex<double> length
    );

    Edge(
      const VertexPtr &source,
      const VertexPtr &target
    );

    /// One endpoint of the edge. Edges are bidirectional; to traverse them as directed,
    /// follow `Vertex::outEdges` only and skip `Vertex::inEdges`.
    [[nodiscard]] const VertexPtr &getSource() const noexcept;

    /// The other endpoint. See `getSource`.
    [[nodiscard]] const VertexPtr &getTarget() const noexcept;

    /// The \f$\mathbb{C}^{*}\f$ connection phase \f$\varphi\f$ carried on this edge's stored
    /// source->target orientation. It is a second edge field, independent of the geometry:
    /// the link variable is \f$ U_{xy} = e^{i\varphi} \in \mathbb{C}^{*} \f$ with
    /// \f$ U_{yx} = U_{xy}^{-1} \f$ (the inverse, not the conjugate — the two coincide only
    /// for real \f$\varphi\f$), and a gauge transformation \f$ g \f$ acts by
    /// \f$ U_{xy} \mapsto g_x^{-1} U_{xy} g_y \f$, leaving `length_` and every metric weight
    /// built from it untouched.
    ///
    /// \f$\varphi\f$ is complex because the structure group is
    /// \f$ \mathbb{C}^{*} = U(1)\times\mathbb{R}^{+} \f$:
    /// \f$ e^{i\varphi} = e^{i\operatorname{Re}\varphi}\,e^{-\operatorname{Im}\varphi} \f$.
    /// `Re` is the compact U(1) angle in radians — the only part with winding, so the only
    /// part that quantizes and the only part a Wilson loop reads. `Im` is the non-compact
    /// \f$\mathbb{R}^{+}\f$ local scale and carries no quantum number.
    ///
    /// It is the link \f$ U = e^{i\varphi} \f$ of the connection: it twists the hopping term of
    /// the Aharonov-Bohm operator (`HodgeLaplacian::connectionLaplacian`) and dresses the Whitney
    /// pencil, so the default Hodge operator `laplacian(k)` is the covariant \f$ h_k(z,U) \f$ at
    /// every degree; only its spectrum's gauge invariance, never a rescaled metric weight,
    /// carries \f$\varphi\f$ into the geometry. The diagonal weights ignore it. The default
    /// (`phase = 0`) is the trivial link.
    ///
    /// @return The \f$\mathbb{C}^{*}\f$ connection phase; `Re` in radians, `Im` the log-scale.
    [[nodiscard]] std::complex<double> getPhase() const noexcept;

    /// The (possibly complex) edge length — the causal degree of freedom, distinct from the
    /// connection `phase` and from \f$l^2\f$. Real for spacelike, imaginary for timelike,
    /// general complex on the Picard-Lefschetz saddle. Causal character is read from
    /// \f$ \arg(l^2) \f$; see the predicates below.
    [[nodiscard]] std::complex<double> getLength() const noexcept;

    /// # Causal character from the argument of the squared length
    ///
    /// Classifying on \f$ \mathrm{Re}(l^2) \f$ alone would discard
    /// \f$ \mathrm{Im}(l^2) \f$, which is physical here. It fails exactly at the case of
    /// interest: at the lightlike point \f$ \mathrm{Im}(l^2) = 2x^2 \neq 0 \f$, so
    /// \f$ l^2 \f$ is purely imaginary and nonzero. A fully null \f$ l^2 = 0 \f$ does not
    /// occur non-trivially.
    ///
    /// Classifying on the argument accounts for both components. Writing
    /// \f$ l = |l| e^{i a} \f$,
    /// \f[ l^2 = |l|^2 e^{2ia}, \quad
    ///     \mathrm{Re}(l^2) = |l|^2 \cos 2a, \quad
    ///     \mathrm{Im}(l^2) = |l|^2 \sin 2a. \f]
    ///
    /// | \f$ a = \arg l \f$ | \f$ l^2 \f$ | disposition |
    /// |---|---|---|
    /// | \f$ 0 \f$ (mod \f$\pi\f$) | real positive | spacelike |
    /// | \f$ \pi/2 \f$ (mod \f$\pi\f$) | real negative | timelike |
    /// | \f$ \pi/4 \f$ (mod \f$\pi/2\f$) | purely imaginary | lightlike |
    /// | anything else | genuinely complex | mixed |
    /// | \f$ l = 0 \f$ | — | degenerate: an absent edge, not a causal type |
    ///
    /// A generic argument is reported as mixed, never assigned to the nearest of the
    /// three — that would invent a definiteness the geometry does not have. Under a
    /// uniformly drawn argument almost every edge is mixed, and the mixed fraction is a
    /// diagnostic: if relaxation imposes causal character, it should fall.
    ///
    /// `squaredArgument()` carries \f$ \arg(l^2) \f$ itself, so a consumer can see where
    /// an edge sits rather than only which bucket it fell in.
    ///
    /// `isDegenerate` is separate from `isNull`: a null edge is a lightlike ray, a
    /// degenerate one is absent. Exactly one of the five predicates holds for any edge.

    /// Absolute floor on the Euclidean modulus below which an edge is degenerate rather
    /// than any causal type. Dimensions of length. This is the one place the Euclidean
    /// modulus is the right norm — an edge with no extent is absent.
    static constexpr double kDegenerateEpsilon = 1e-12;

    /// Angular half-width, in radians, within which an argument counts as definite.
    ///
    /// Angular, so dimensionless and scale-free. An absolute floor cannot serve both
    /// candidates: \f$ \mathrm{Re}(l^2) \f$ has dimensions of length squared and
    /// \f$ |\mathrm{Im}(l)| \f$ dimensions of length, so one constant cannot cover both,
    /// and either choice silently reclassifies edges near the cone as a complex refines.
    ///
    /// In \f$ \arg(l^2) \f$ the three definite dispositions sit at \f$ 0 \f$,
    /// \f$ \pm\pi/2 \f$ and \f$ \pi \f$ — equivalently \f$ a = 0, \pi/4, \pi/2 \f$ — so one
    /// half-width applied at each treats them symmetrically. The buckets stay disjoint for
    /// any width below \f$ \pi/4 \f$.
    ///
    /// The value is a numerical-noise guard, not a bucket width: a deliberately
    /// constructed disposition lands on its argument to within a few ulp
    /// (\f$ \sim 10^{-16} \f$ rad), so \f$ 10^{-9} \f$ sits seven orders above float noise
    /// and eight below the \f$ \pi/4 \f$ collision bound. Widening it would absorb
    /// genuinely mixed edges into definite buckets.
    static constexpr double kCausalAngularEpsilon = 1e-9;

    /// \f$ \arg(l^2) \in (-\pi, \pi] \f$ — the measured quantity every predicate
    /// below classifies. Zero is spacelike, \f$ \pm\pi/2 \f$ lightlike, \f$ \pi \f$
    /// timelike, anything else mixed.
    [[nodiscard]] double squaredArgument() const noexcept;
    /// \f$ \mathrm{Re}(l^2) = x^2 - t^2 \f$, carried for consumers that want the
    /// interval itself. It does not decide the disposition on its own.
    [[nodiscard]] double lorentzianMagnitude() const noexcept;
    [[nodiscard]] bool isTimelike() const noexcept;
    [[nodiscard]] bool isSpacelike() const noexcept;
    [[nodiscard]] bool isNull() const noexcept;
    /// A genuinely complex \f$ l^2 \f$: no definite causal character.
    [[nodiscard]] bool isMixed() const noexcept;
    /// An absent edge (\f$ |l|_E \approx 0 \f$), which is not a causal type.
    [[nodiscard]] bool isDegenerate() const noexcept;
    [[nodiscard]] EdgeDisposition disposition() const noexcept;

#ifdef TESSERA_VERBOSE
    [[nodiscard]] std::string toString() const noexcept;
#else
    [[nodiscard]] std::string toString() const noexcept {
      return "";
    };
#endif

    /// Replace the source vertex in-place and update the fingerprint.
    ///
    /// The caller extracts this edge from its EdgeList before calling and reinserts it
    /// after. Modifying the fingerprint while the edge sits in a hash-keyed container
    /// leaves it in a stale bucket, which is undefined behaviour.
    /// Spacetime::swapVertexLabels shows the extract/update/reinsert pattern.
    void replaceSourceVertex(const VertexPtr &newSource);

    /// Replace the target vertex in-place and update the fingerprint.
    ///
    /// Same container-safety requirement as replaceSourceVertex.
    void replaceTargetVertex(const VertexPtr &newTarget);

    /// Whether a given vertex is an endpoint of this edge. The comparison is on
    /// source/target vertex ids, not on pointers.
    ///
    /// @param vertexId The id of the vertex to look for.
    /// @return true if the vertex is an endpoint of this edge
    bool hasVertex(std::uint64_t vertexId) const;
    bool hasVertex(const VertexPtr &vertex) const;

    bool operator==(const Edge &other) const;

    [[nodiscard]] std::uint64_t toHash() const;

    Fingerprint fingerprint{};

    /// The endpoint-id pair identifying this edge. Two edges with the same EdgeKey compare
    /// equal by value.
    ///
    /// @returns A tuple of {sourceId, targetId}.
    EdgeKey getKey() const noexcept;

    /// Set the complex edge length \f$l\f$ — the edge's one degree of freedom. Real for
    /// spacelike, imaginary for timelike, general complex off the real-Lorentzian locus.
    ///
    /// There is no squared-length setter. \f$l^2\f$ is not stored, so it cannot drift out
    /// of sync with \f$l\f$, and a caller holding an \f$l^2\f$ writes
    /// ``setLength(std::sqrt(l2))``, picking the branch explicitly. \f$l\f$ is the right
    /// primitive: \f$l \mapsto l^2\f$ is two-to-one, so \f$l^2\f$ cannot express which of
    /// \f$\pm l\f$ this edge is.
    ///
    /// Cost: a geometry specified by a squared value round-trips through
    /// \f$\sqrt{\cdot}\f$, so consumers see \f$l^2 \pm 1\f$ ulp rather than the exact
    /// value. That matters most in the ill-conditioned regime where the Cayley-Menger
    /// determinant approaches zero.
    void setLength(std::complex<double> l) noexcept {
      length_ = l;
      ++lengthRevision_;
    }

    /// Monotone per-edge write counter, bumped by every ``setLength``.
    /// ``Simplex``'s length-derived geometry cache keys on the sum of its
    /// edges' revisions, so an unchanged key proves no incident length changed
    /// since the cache was filled. ``setPhase`` does not bump it:
    /// the cache holds only length-derived data (Gram / Cayley-Menger), and
    /// the connection phase never enters those.
    [[nodiscard]] std::uint64_t lengthRevision() const noexcept {
      return lengthRevision_;
    }

    /// Set the \f$\mathbb{C}^{*}\f$ connection phase: `Re` the compact U(1) angle in radians,
    /// `Im` the non-compact log-scale. Used by the Aharonov-Bohm operator and its gauge
    /// transform to re-twist the edge without rebuilding the mesh. A real argument converts
    /// implicitly and reproduces the untwisted-geometry, real-angle case exactly.
    void setPhase(std::complex<double> p) noexcept {
      phase = p;
      ++phaseRevision_;
    }

    /// Monotone ``setPhase`` counter, the phase analogue of ``lengthRevision``.
    /// The Aharonov-Bohm operator reads phases, so the shared spectrum cache
    /// keys on both counters; the Simplex geometry cache (Gram/Cayley-Menger)
    /// keys on lengths alone and deliberately ignores this one.
    [[nodiscard]] std::uint64_t phaseRevision() const noexcept {
      return phaseRevision_;
    }

    /// Walk a closed loop of ordered directed steps (each Edge's
    /// getSource()->getTarget() is one traversal step). Invokes f(sourceId,
    /// targetId, sign) per step; sign = +1 if sourceId < targetId (canonical
    /// orientation) else -1.
    template <typename F>
    static void walkLoop(const std::vector<Edge> &loop, F &&f) {
      for (const Edge &step : loop) {
        const std::uint64_t u = step.getSource()->getId();
        const std::uint64_t v = step.getTarget()->getId();
        f(u, v, (u < v) ? 1.0 : -1.0);
      }
    }

    /// The Van Raamsdonk metric law: the spacelike length for a given mutual
    /// information ``I`` — the value to store via ``setLength`` on a
    /// same-time-slice edge. Returns −log(I/iMax), floored at −log(epsilon) (so
    /// the length stays finite) when I < epsilon·iMax, and when iMax ≤ 0 or
    /// I ≤ 0. Pass epsilon ≤ 0 to opt out of the floor, in which case a
    /// vanishing mutual information gives the divergent +∞ the law implies.
    /// Always real and ≥ 0, i.e. spacelike.
    [[nodiscard]] static double
    vanRaamsdonkLength(double I, double iMax,
                       double epsilon = 1e-10) noexcept;

    /// Time-aware Van Raamsdonk length for this edge, given the mutual information
    /// ``I`` between its endpoints (the one-forward-step convention): a worldline edge
    /// whose endpoints lie on different time slices (``Vertex::getTime``) is null and
    /// returns 0; a same-slice edge is spacelike and returns
    /// ``vanRaamsdonkLength(I, iMax, epsilon)``.
    [[nodiscard]] double
    vanRaamsdonkLengthFor(double I, double iMax,
                          double epsilon = 1e-10) const;

    /// Index into EdgeList::liveVec_ (maintained by EdgeList).
    std::uint32_t liveIdx_{UINT32_MAX};

    /// Simplices currently containing this edge — the edge's cofaces. Mirror of
    /// ``Vertex::simplices`` at edge granularity, used by ``Vertex::removeOutEdge`` /
    /// ``Vertex::removeInEdge`` to drop the edge from just the simplices that contain
    /// it, rather than iterating every simplex touching an endpoint and filtering by
    /// ``hasVertex``; that filtering scan dominated `thermalize` profiles.
    ///
    /// Maintained in lockstep with ``Simplex::edges``:
    ///   * Spacetime::registerSimplex registers the simplex on each of its edges
    ///   * Spacetime::unregisterSimplex removes it
    ///   * Simplex::addEdge / Simplex::removeEdge mirror the same callbacks at runtime
    ///
    /// A caller that mutates the index from inside an iteration loop (e.g.
    /// ``simplex->removeEdge(this)`` invalidates ``simplices_``) snapshots with
    /// ``simplicesCopy()`` first.
    void registerSimplex(SimplexPtr s);
    void unregisterSimplex(SimplexPtr s) noexcept;
    [[nodiscard]] Simplices const& simplices() const noexcept { return simplices_; }
    [[nodiscard]] Simplices simplicesCopy() const { return simplices_; }

  private:
    VertexPtr source = nullptr;
    VertexPtr target = nullptr;

    /// The complex edge length \f$l\f$ — the edge's one stored degree of freedom
    /// (distinct from the connection `phase`). Causal character is read from
    /// \f$ \arg(l^2) \f$. \f$l^2\f$ is derived by squaring at the point of use, never
    /// stored.
    std::complex<double> length_{};
    /// Monotone ``setLength`` counter read by ``lengthRevision()``; see there.
    std::uint64_t lengthRevision_{0};
    /// Monotone ``setPhase`` counter read by ``phaseRevision()``; see there.
    std::uint64_t phaseRevision_{0};
    std::complex<double> phase{0.0, 0.0};

    Simplices simplices_{};
};

}

#endif //TESSERA_TESSERA_SRC_EDGE_H_
