// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_SIMPLEX_H
#define TESSERA_SIMPLEX_H

#include "mesh/ForwardDeclarations.h"
#include <atomic>
#include <complex>
#include <cstdint>
#include <map>
#include <mutex>
#include <unordered_map>
#include <memory>
#include <utility>
#include <vector>
#include <functional>

#include "Logger.h"
#include "mesh/EdgeList.h"
#include "mesh/Fingerprint.h"
#include "mesh/TemporalOrientation.h"
#include "utils.h"

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

/// # Simplex Class
///
/// A simplex is a generalization of the concept of a triangle or tetrahedron to arbitrary dimensions. Each simplex
/// is defined by its vertices.
///
/// Each simplex has a volume \f$ V_s \f$, which can represent various physical properties depending on the context.
///
/// A k-simplex, \f$ \sigma^k \f$, within a simplicial complex, \f$ K \f$ is defined as a set of k+1 vertices.
/// Simplicial complex construction is a bit of a bottleneck in simulation of spacetime. At the moment; we declare some
/// vertices, then use coning to create a Simplex from those vertices. Those vertices are passed to the Simplex along
/// with the edges used to connect them as a performance optimization.
///
/// Most of the time building the simplicial complex is spent calculating facets from all subsets of Simplex Vertices. A
/// faster method for building the complex would be to avoid computing those vertices and edges; and just compute the
/// simplex as an abstraction with faces, cofaces, and an orientation. We'll leave this for a "Version 2 feature".
///
class Simplex {
  public:
    // ==================== Static Factory Methods ====================
    static Simplex* create(Spacetime *spacetime_, const VertexPtrs &vertices_, const Edges &edges_);
    static Simplex* create(Spacetime *spacetime_, const VertexPtrs &vertices_, const Edges &edges_, const TemporalOrientation &orientation_);
    [[nodiscard]] static std::size_t computeNumberOfEdges(std::size_t k);

    // ==================== Constructors & Initialization ====================
    /// @param vertices_
    explicit Simplex(Spacetime *spacetime_, const VertexPtrs &vertices_, Edges edges_);
    Simplex(Spacetime *spacetime_, const VertexPtrs &vertices_, Edges edges_ ,const TemporalOrientation &orientation_);

    std::uint64_t size() const noexcept { return vertices.size(); }

    /// The initialize step is necessary because the canonical owner of the Simplex object is the Spacetime, and ideally
    /// that canonical owner is the only one to permanently hang onto a std::shared_ptr. So when we initialize with this
    /// method we add the std::shared_ptr<Simplex> (aka SimplexPtr) to all the Vertex (es) that are members of the
    /// Simplex. Again; we define a Simplex abstractly as a set of vertices with a time orientation.
    /// When you construct a Spacetime which can abstractly be considered a Simplicial complex; having access to the
    /// Simplex by Vertex is pretty handy for bookkeeping.
    void initialize(Simplex* simplex);

    // Sorted vertex-id tuple of a top simplex.
    std::vector<std::uint64_t> topTuple() const;

    // ==================== String Representation ====================
#ifdef TESSERA_VERBOSE
    std::string toString() const noexcept;
#else
    std::string toString() const noexcept {
      return "";
    }
#endif

    // ==================== Basic Getters ====================
    /// Each simplex has an associated _orientation_ in the case you're preserving causality with your work. You can
    /// find specifics of the TemporalOrientation abstractly and concretely/computationally in the documentation for the
    /// TemporalOrientation
    [[nodiscard]] const TemporalOrientation &getOrientation() const noexcept { return orientation; }

    /// The earliest time assigned to a vertex in this Simplex.
    /// @returns ti for the Simplex.
    double getTi() const noexcept { return ti; }

    /// The latest time assigned to a vertex in this Simplex.
    /// @returns tf for the Simplex.
    double getTf() const noexcept { return tf; }

    // ==================== Vertex Queries ====================
    /// @return A list of Vertex (es) in traversal order. You can iterate these to walk the Face.
    [[nodiscard]] const VertexPtrs &getVertices() const noexcept;

    /// This method is self-explanatory. O(1) lookups for who has what.
    [[nodiscard]] bool hasVertex(const VertexPtr &vertex) const;

    /// This method produces a lookup table \f$ Id \rightarrow Vertex \f$. The only place it's used at the moment is for
    /// verifying state in our Python unit tests.
    [[nodiscard]] VertexIdMap getVertexIdLookup() const noexcept;


    // ==================== Edge Queries ====================
    /// @returns Edges in traversal order (the order of input vertices).
    [[nodiscard]] const Edges &getEdges() const;
    [[nodiscard]] std::size_t getNumberOfEdges() const;

    /// This method computes Edge (s) of the Simplex in traversal order. Note that the edges are effectively undirected
    /// since it can point either way as the direction relates to vertex order. So it's possible for e.g. vertices
    /// \f$ \{v_0, v_1, v_2\} \f$ to correspond to edges \f$ \{ e_{0 \rightarrow 1}, e_{2 \rightarrow 1}, e_{2 \rightarrow 0} \} \f$
    [[nodiscard]] bool hasEdge(const EdgePtr &edge) const;
    [[nodiscard]] bool hasEdge(const VertexPtr &vertexA, const VertexPtr &vertexB) const;
    [[nodiscard]] bool hasEdgeContaining(IdType vertexId) const;

    // ==================== Face & Facet Queries ====================
    ///
    /// A k-simplex is the convex hull of k + 1 affinely independent points. Each has faces of all dimensions from 0 up
    /// to k–1. A k-1 simplex is called a Facet.
    ///
    /// A j-face is a j-simplex incorporating a subset (of size j) of the k-simplex vertices.
    ///
    /// The number of j-faces ( \f$ \sigma^j \f$ ) of a k-simplex \f$ \sigma^k \f$ is given by
    ///
    /// \f[
    /// \binom{k+1}{j+1}
    /// \f]
    ///
    /// And the total number of faces of all dimensions is
    /// \f$ \sum_{j=0}^{k-1} \binom{k+1}{j+1} = 2^{k+1} - 2 \f$
    ///
    std::size_t getNumberOfFaces(std::size_t j) const;

    ///
    /// A Face, \f$ \sigma^{k-1} \subset \sigma^{k} \f$ of a k-simplex \f$ \sigma^k \f$ is any k-1 simplex contained by
    /// the k-simplex.
    ///
    /// To attach one Simplex \f$ \sigma_i^k \f$ to another \f$ \sigma_j^k \f$, we define the respective faces
    /// \f$ \sigma_i^{k-1} \f$ and \f$ \sigma_j^{k-1} \f$ at which they should be attached. The orientation is determined
    /// by the orientation of those respective `Simplex`es.
    ///
    /// The Facets are the \f$ \sigma^{k-1} \subset \sigma^{k} \f$ faces on which we'll most commonly join two simplices
    /// to form a simplicial complex \f$ K \f$.
    ///
    /// ## Canonical order and induced orientation
    ///
    /// Facets are returned in **canonical topological order**: facet \f$ i \f$ is
    /// this simplex with the \f$ i \f$-th vertex of ``getVertices()`` removed, the
    /// remaining vertices kept in their original relative order. That relative
    /// order is the facet's *induced orientation* from this simplex.
    ///
    /// Consequently the **orientation of facet \f$ i \f$ relative to this simplex
    /// is simply \f$ (-1)^i \f$** — its coefficient in the simplicial boundary
    /// \f$ \partial\sigma = \sum_i (-1)^i \,\sigma\!\setminus\! v_i \f$. The
    /// orientation is therefore already carried by the *return order*: a caller
    /// reads it from the facet's index, contextual to the simplex ``getFacets``
    /// was called on. There is deliberately no separate orientation flag or
    /// per-simplex orientation bit — that would re-implement what the ordering
    /// already encodes. (The vertex *sorting* used by ``Fingerprint`` is just a
    /// set-identity key and is unrelated to this orientation.)
    ///
    /// Note this is a per-simplex (contextual) orientation; a globally
    /// consistent boundary operator (\f$ \partial^2 = 0 \f$) additionally
    /// requires the complex's simplices to share a coherent vertex ordering,
    /// which is a property of how the complex was built, not of this method.
    ///
    /// @return all k-1 simplices contained within this k-simplex, in canonical
    ///   topological order (facet \f$ i \f$ omits vertex \f$ i \f$).
    const Simplices &getFacets();

    bool hasFacets() const;
    bool hasStoredFacet(const SimplexPtr &facet) const;

    // ==================== Coface Queries & Management ====================
    ///
    /// A simplex, \f$ \sigma \in K \f$ with vertices \f$ V_{\sigma} \f$  is a coface of \f$ \tau \in K \f$
    /// with vertices \f$ V_{\tau} \f$ iff \f$ V_{\tau} \subset V_{\sigma} \f$. For our purposes, however, we confine
    /// cofaces to those of dimensionality \f$ k+1 \f$ compared to the facet of dimension \f$ k \f$
    ///
    /// We define a _facet_ as a set of shared vertices. The facet of any given k-simplex \f$ \sigma^k \f$ is a k-1
    /// simplex, such that  \f$ \sigma_{k} \f$ is a coface of \f$ \sigma_{k-1} \f$.
    ///
    /// Register a \f$(k\!+\!1)\f$-simplex as a coface of this \f$ k \f$-simplex.
    /// The coface relation encodes the incidence structure of the simplicial complex:
    /// \f$ \sigma^{k+1} \f$ is a coface of \f$ \sigma^k \f$ iff
    /// \f$ \sigma^k \subset \sigma^{k+1} \f$ (the lower-dimensional simplex is a face
    /// of the higher-dimensional one).
    void addCoface(SimplexPtr simplex);

    /// Unregister a coface from this simplex. Called during simplex removal in
    /// Pachner moves to maintain consistent coface bookkeeping.
    void removeCoface(SimplexPtr simplex);

    /// Check whether this simplex is a coface of the given facet,
    /// i.e., whether all vertices of the facet are contained in this simplex.
    /// @param facet The candidate lower-dimensional simplex
    /// @param shallow If true, also require the dimension difference to be exactly 1
    bool isCofaceTo(const SimplexPtr &simplex, bool shallow=true) const;

    [[nodiscard]] bool hasCoface(SimplexPtr simplex) const;

    ///
    /// Co-faces are maintained as state rather than computed on the fly. This means any time a Simplex is attached to
    /// another Simplex; it must be added to the face at which it's attached as a co-face. If a Simplex, Edge, or Vertex
    /// within that Face is removed at any point; that effect should cascade up the ownership tree, which goes
    /// \f[
    /// Vertex \subset Edge \subset Simplex \subset Spacetime
    /// \f]
    ///
    /// @return The set of k-simplices that share this face.
    [[nodiscard]] const Simplices &getCofaces() const noexcept;

    /// This method computes the maximum number of k+1 co-faces that can be joined to this k-Simplex _in general_.
    /// To check whether a simplex is on the boundary (available for gluing), use `isBoundary()`.
    ///
    /// For a given k-simplex \f$ \sigma^k \f$, a co-face is defined as an m-simplex, \f$ \sigma^m \f$ such that \f$ m \gt k \f$
    /// and \f$ \sigma^k \subset \sigma^m \f$. The maximum number of co-faces that can be joined to a k-simplex is in
    /// general unbounded, but for our purposes we set it to the number of faces of the simplex, so we impose the
    /// constraint that the coface no be _generally_ \f$ m \gt k \f$, but exactly \f$ k + 1 \f$, so \f$ m = k + 1 \f$.
    ///
    /// This can be confusing because for the purpose of causally gluing simplices we look at a face, \f$ \sigma^k \f$
    /// of the (k+1)-simplex, \f$ \sigma^{k+1} \f$ where to that (k-) face we want to glue another (k+1)-simplex on one
    /// of it's k-faces. So the maximum number of co-faces that can be joined to a k-simplex is the number of faces of
    /// that simplex.
    ///
    /// @return
    std::size_t maxKPlusOneCofaces() const;

    // ==================== State Queries ====================
    /// True when every vertex lies on the same time slice (a purely spatial simplex).
    [[nodiscard]] bool isSpatial() const noexcept { return _isSpatial; }

    /// @deprecated Use isSpatial(). Historically misnamed: returns true for spatial
    /// simplices (all vertices at the same time), NOT for timelike ones.
    [[nodiscard]] bool isTimelike() const noexcept { return _isSpatial; }

    /// True when this simplex has fewer than 2 cofaces, i.e. it lies on
    /// the boundary of the complex and has a free face available for gluing.
    [[nodiscard]] bool isBoundary() const noexcept;

    /// True when any facet of this simplex is on the boundary (has < 2 cofaces).
    bool hasBoundaryFacet();
    std::uint64_t hash() const noexcept;

    // ==================== Geometry ====================
    //
    // Complexified geometry: every entry point below consumes the full complex
    // squared interval l^2. No Wick/magnitude path and no Re(l^2) projection are
    // applied; MultiCobordism::runStage2 optimizes these same complex coordinates.

    /// Gram matrix of this simplex from its edge lengths.
    /// Returns a flat (d x d) row-major matrix where d = size() - 1.
    /// Vertex 0 is the origin: G_ij = 1/2(s(v0,vi) + s(v0,vj) - s(vi,vj)),
    /// where s(.,.) is the squared edge length.
    ///
    /// By default the geometry is **signature-aware**: the signed l^2 is kept,
    /// so a timelike edge (l^2 < 0) carries its Lorentzian sign into G and
    /// det(G) records the metric signature of the cell.
    /// Always signature-aware: there is no Euclidean/Wick-rotated mode (#641).
    [[nodiscard]] std::vector<std::complex<double>> gramMatrix() const;

    /// Length-derived geometry cache. The Gram / Cayley-Menger pipeline
    /// (matrix, determinant, cofactors) is a pure function of this simplex's
    /// edge lengths, yet the action/gradient/Hessian paths historically
    /// recomputed it once per (hinge, coface) evaluation — the dominant engine
    /// cost in a perf sample of the live one-step drive. The four sections
    /// below memoize those products, each filled lazily by the SAME
    /// ``gramMatrix``/``cayleyMengerMatrix``/``cayleyMengerCanonical``/
    /// ``determinant``/``cofactorMatrix`` calls the direct path runs, so a
    /// cached value is bit-for-bit the recomputed one.
    ///
    /// Invalidation is by key comparison, never by callback: the key is the
    /// sum of the incident edges' monotone ``Edge::lengthRevision()`` counters
    /// plus this simplex's structural revision (bumped by ``addEdge``/
    /// ``removeEdge``, compensated so the key never repeats an old value).
    /// Any ``setLength`` on an incident edge therefore misses the cache and
    /// refills on next use — including perturb/restore probes, whose restore
    /// bumps the revision again (a conservative refill, never a stale hit).
    ///
    /// Concurrency: fills are serialized by a per-simplex mutex and published
    /// by a release-store of the section key; readers that observe the current
    /// key use the payload lock-free. Lengths are only mutated in the serial
    /// phases between parallel evaluations (stage-2 line search, move
    /// apply/rollback), so within a parallel region the key is constant and
    /// the returned references stay valid. Do not hold the references across
    /// a length mutation.
    struct GeomCache {
      std::vector<std::complex<double>> gram;       ///< flat d×d Gram
      std::complex<double> gramDet{};               ///< det(gram)
      std::vector<std::complex<double>> gramCof;    ///< cofactors of gram
      std::vector<std::complex<double>> cm;         ///< (d+2)² raw-order Cayley-Menger
      std::complex<double> cmDet{};                 ///< det(cm)
      std::vector<std::complex<double>> cmCof;      ///< cofactors of cm
      std::vector<std::complex<double>> cmCanon;    ///< canonical-frame Cayley-Menger
      std::vector<std::complex<double>> cmCanonCof; ///< cofactors of cmCanon
      /// vertex id → 1-based border index in the canonical order.
      ///
      /// A flat vector rather than a map: a cell carries d+1 vertices, so this
      /// holds at most six entries in four dimensions. An ``unordered_map``
      /// costs a bucket array and a node allocation per entry to answer a
      /// lookup that a linear scan over six contiguous pairs answers faster,
      /// and every simplex pays that whether or not it is ever asked.
      std::vector<std::pair<std::uint64_t, int>> canonPos1;
    };
    /// Cache with the ``gram``+``gramDet`` section current. O(#edges) key walk
    /// on a hit; fills via the direct pipeline on a miss.
    [[nodiscard]] const GeomCache &gramCache() const;
    /// Cache with ``gram``+``gramDet``+``gramCof`` current.
    [[nodiscard]] const GeomCache &gramCofCache() const;
    /// Cache with the raw-order ``cm``+``cmDet``+``cmCof`` section current.
    [[nodiscard]] const GeomCache &cmCache() const;
    /// Cache with the canonical ``cmCanon``+``cmCanonCof``+``canonPos1``
    /// section current (the frame ``dihedralAngle`` evaluates in).
    [[nodiscard]] const GeomCache &cmCanonicalCache() const;

    /// Cayley-Menger bordered matrix of this simplex: a flat (d+2) x (d+2)
    /// row-major matrix with a zero corner, a border of ones, and the squared
    /// edge-length matrix in the lower-right (d+1) x (d+1) block. Its cofactors
    /// give the dihedral angles (see ``dihedralAngle``).
    /// Always signature-aware: there is no Euclidean/Wick-rotated mode (#641).
    [[nodiscard]] std::vector<std::complex<double>> cayleyMengerMatrix() const;

    /// Lorentzian (Sorkin) dihedral angle at ``hinge`` within this top simplex,
    /// as a complex number, from the **signed** (non-Wick) Cayley-Menger
    /// cofactors — all three of the Sorkin/Asante-Dittrich m ∈ {0, 1, 2}
    /// regimes (#581), from a single expression (#638):
    ///
    /// \f[ \cos\theta = \frac{-C_{ij}}{\sqrt{C_{ii}}\,\sqrt{C_{jj}}} \f]
    ///
    /// **Two separate principal square roots, never** ``sqrt(C_ii*C_jj)``. For
    /// complex \f$a, b\f$ the two differ by a sign exactly when both sit on the
    /// negative real axis: with a unit tetrahedron's ``C_ii = C_jj = -3``,
    /// ``sqrt(C_ii*C_jj)`` is ``+3`` while ``sqrt(C_ii)*sqrt(C_jj)`` is
    /// ``(i*sqrt3)(i*sqrt3) = -3``. Folding the product under one root is what
    /// used to force a hand-applied ``(-1)^d`` parity fix, a three-way branch
    /// dispatch, and an i<->j anchoring swap; taking the roots separately makes
    /// all three *emerge* from the branch structure.
    ///
    /// The causal regimes are then cases of one formula, not code paths:
    ///
    /// * **Same-sign cofactors, |r| <= 1** — the wedge stays on one side of the
    ///   light cone and the angle is the ordinary real one (m = 0).
    /// * **Same-sign cofactors, |r| > 1** — the boost regime (m even):
    ///   ``std::acos`` returns a complex value whose imaginary part is the
    ///   rapidity and whose real part (0 or pi) counts crossed quadrants.
    /// * **Opposite-sign cofactors** — the m = 1 **light-cone crossing** (one
    ///   facet direction spacelike, one timelike). The denominator turns purely
    ///   imaginary, ``r = i*y``, and the principal
    ///   \f$\arccos(iy) = \pi/2 - i\,\mathrm{asinh}(y)\f$ is exactly Sorkin's
    ///   quarter turn plus a signed boost — with no special case. Around a flat
    ///   one-ray-per-quadrant Minkowski vertex star the four boosts telescope to
    ///   zero and four crossings sum to 2*pi, so ``2*pi - sum = 0`` still holds;
    ///   generic in CDT (every base-tet triangle of a (4,1) cell).
    ///
    /// This keeps the boost content, as it should be EVERYWHERE. Note
    /// the same-sign (m = 0 / boost) regimes' imaginary sign is the principal
    /// branch: the wedge's boost *orientation* is not determined by edge
    /// lengths alone (a PT reflection flips it at identical l^2), so only
    /// crossing wedges carry an intrinsically signed boost.
    /// Refs: Regge (1961); Sorkin, Lorentzian angles & trigonometry
    /// (arXiv:1908.10022); Asante-Dittrich-Padua-Arguelles, arXiv:2104.00485
    /// Eq. (10).
    [[nodiscard]] std::complex<double>
    dihedralAngle(SimplexPtr hinge) const;

    /// Complex Lorentzian deficit at this hinge: 2*pi minus the sum of
    /// ``dihedralAngle`` over the top simplices containing it. Real
    /// for an all-spacelike (Euclidean) neighbourhood (the ordinary angle
    /// defect); complex when timelike cells contribute boosts.
    [[nodiscard]] std::complex<double> deficitAngle() const;

    /// Exact analytic gradient of this hinge's ``deficitAngle`` with
    /// respect to the squared length of each surrounding edge:
    /// \f$ \partial \varepsilon / \partial \ell^2_e \f$. The deficit is
    /// \f$ 2\pi - \sum_\tau \theta_\tau \f$ over the top cells \f$ \tau \f$
    /// containing the hinge, with \f$ \theta = \arccos r \f$ and
    /// \f$ r = -C_{ij}/\pm\sqrt{|C_{ii}C_{jj}|} \f$ a ratio of cofactors of the
    /// (signed) Cayley-Menger matrix \f$ B \f$ (linear in \f$ \ell^2 \f$). Since
    /// \f$ C = \det(B)\,(B^{-1})^\top \f$ the cofactor derivatives are closed
    /// form; the same-sign boost branch uses \f$ d\theta/dr = -1/\sin\theta \f$
    /// so it matches ``std::acos`` exactly, and the m = 1 light-cone-crossing
    /// branch (\f$ C_{ii}C_{jj} < 0 \f$, #581) differentiates
    /// \f$ \theta = \pi/2 - i\,\mathrm{asinh}(y) \f$, \f$ y = C_{ij}/D \f$, via
    /// \f$ d\theta/dy = -i/\sqrt{1+y^2} \f$ (never singular). Keyed by sorted
    /// vertex-id edge; only the edges of the top cells touching the hinge
    /// appear. Complex (the boost part is carried, not truncated).
    [[nodiscard]] std::map<std::pair<std::uint64_t, std::uint64_t>,
                           std::complex<double>>
    deficitAngleGradient() const;

    /// Exact analytic Hessian of this hinge's deficit angle:
    /// \f$ \partial^2 \varepsilon / \partial \ell^2_e \partial \ell^2_f \f$.
    /// One derivative beyond ``deficitAngleGradient``: the same
    /// per-top-cell Cayley-Menger machinery carried to second order, with the
    /// cofactor second derivative
    /// \f$ \partial^2 C_{pq} = \partial(\det B\,T) \f$ (T the gradient's
    /// bracket), \f$ d\theta/dr = -1/\sin\theta \f$ and
    /// \f$ d^2\theta/dr^2 = -r/\sin^3\theta \f$ on same-sign wedges; on the
    /// m = 1 crossing (\f$ C_{ii}C_{jj} < 0 \f$, #581)
    /// \f$ d\theta/dy = -i/\sqrt{1+y^2} \f$ and
    /// \f$ d^2\theta/dy^2 = +i\,y/(1+y^2)^{3/2} \f$ with \f$ y = C_{ij}/D \f$.
    /// Keyed by the (sorted) edge pair; symmetric. Complex (the boost part is
    /// carried, not truncated).
    [[nodiscard]] std::map<std::pair<std::pair<std::uint64_t, std::uint64_t>,
                                     std::pair<std::uint64_t, std::uint64_t>>,
                           std::complex<double>>
    deficitAngleHessian() const;

    /// Area of this simplex interpreted as a triangular hinge (3 vertices).
    /// Uses Heron's formula on the three edge squared lengths;
    ///
    /// wickRotate is no longer supported. Everything should be fully Lorentzian. Passing wickRotate=true is an error.
    ///
    /// **Lorentzian note (#641):** the Heron radicand is carried under a complex
    /// square root, so a triangle with a negative radicand — every timelike
    /// (negative-content) triangle, e.g. the mixed-causal hinge of a CDT (4,1)
    /// cell — returns an **imaginary** area rather than the 0 the old real-typed
    /// clamp produced. Zero was never the area of those triangles; it was the
    /// value a real return type could represent.
    [[nodiscard]] std::complex<double> area() const;

    /// Signed d-content (volume) of this simplex on the honest,
    /// signature-respecting geometry: sign(det G) * sqrt(|det G|) / d!, with G
    /// the non-Wick-rotated ``gramMatrix`` and d = size() - 1. For a Euclidean
    /// (all-spacelike) simplex this is the ordinary positive volume; a
    /// Lorentzian cell whose tangent metric has a negative Gram determinant
    /// returns a negative content, recording the signature rather than
    /// discarding it the way |l^2| would.
    [[nodiscard]] std::complex<double> volume() const;

    /// Exact analytic gradient of this simplex's **signed `volume()`** with respect
    /// to the squared length of each of its edges:
    /// \f$ \partial V / \partial \ell^2_e \f$, returned as an edge-keyed map (sorted
    /// `(a,b)` ids). By Jacobi's formula on the Gram determinant
    /// (\f$ V = \mathrm{sgn}\,\sqrt{|\det G|}/d! \f$, \f$ G \f$ linear in \f$ \ell^2 \f$):
    /// \f$ \partial V/\partial\ell^2_e = \tfrac{V}{2}\,\mathrm{tr}(G^{-1}\,\partial_e G) \f$,
    /// the same machinery (`gramMatrix`/`determinant`/`cofactorMatrix`) the
    /// circumcentric `dualVolumeGradient` (#354) uses. This is the per-degree
    /// **Hodge inner-product weight** gradient (the weights \f$ W_k \f$ are signed
    /// simplex volumes), the keystone for an arbitrary-degree analytic
    /// \f$ \partial L_k/\partial\ell^2 \f$ and hence the general-k \f$ r_U \f$ gradient.
    [[nodiscard]] std::map<std::pair<std::uint64_t, std::uint64_t>, std::complex<double>>
    volumeGradient() const;

    /// Exact **directional second derivative** of the signed `volume()`:
    /// \f$ \sum_f v_f\,\partial^2 V/\partial\ell^2_e\partial\ell^2_f \f$, keyed by
    /// \f$ e \f$ over this simplex's edges, for the direction \f$ v \f$ supplied
    /// as the same edge-keyed map shape `volumeGradient()` returns (absent edges
    /// contribute zero). Differentiating Jacobi's formula once more, and using
    /// that \f$ G \f$ is **linear** in \f$ \ell^2 \f$ so \f$ \partial_f\partial_e G = 0 \f$:
    /// \f[ \frac{\partial^2 V}{\partial\ell^2_e\partial\ell^2_f}
    ///     = \frac{V}{4}\,t_e t_f
    ///       - \frac{V}{2}\,\mathrm{tr}\bigl(G^{-1}\partial_f G\,G^{-1}\partial_e G\bigr),
    ///     \qquad t_e=\mathrm{tr}(G^{-1}\partial_e G), \f]
    /// contracted against \f$ v \f$ this is
    /// \f$ \tfrac{1}{2}\dot V t_e-\tfrac{V}{2}\mathrm{tr}(M\,\partial_e G) \f$ with
    /// \f$ \dot G=\sum_f v_f\partial_f G \f$, \f$ \dot V=\tfrac{V}{2}\mathrm{tr}(G^{-1}\dot G) \f$
    /// and \f$ M=G^{-1}\dot G G^{-1} \f$ formed once per simplex. Exact to machine
    /// precision — no finite differences — and the Hessian half of the analytic
    /// \f$ \partial^2 L_k/\partial\ell^2\partial\ell^2 \f$ contraction the joint
    /// Regge--Hodge descent direction needs. Uses the same
    /// `gramMatrix`/`determinant`/`cofactorMatrix` cache as `volumeGradient()`.
    [[nodiscard]] std::map<std::pair<std::uint64_t, std::uint64_t>, std::complex<double>>
    volumeGradientDirectionalDerivative(
        const std::map<std::pair<std::uint64_t, std::uint64_t>,
                       std::complex<double>> &direction) const;

    /// Fail-loudly admissibility check for a purely-spacelike simplex.
    ///
    /// "Spacelike" means every edge has squared length > tol (the Edge
    /// convention: spacelike > 0, null = 0, timelike < 0). For such a cell the
    /// Gram matrix relative to vertex 0 must be positive-definite —
    /// equivalently the generalized triangle inequalities hold and the cell has
    /// real, nonzero d-content. If it does not, the simplex is inadmissible and
    /// this throws ``std::runtime_error`` rather than silently repairing it
    /// (positive-definiteness is checked Eigen-free via Sylvester's criterion
    /// on the leading principal minors). A simplex containing any null or
    /// timelike (worldline) edge is **skipped** (returns without checking): its
    /// admissibility is governed by the Lorentzian structure, not the spacelike
    /// triangle inequalities. Fewer than two vertices is trivially admissible.
    void assertSpacelikeAdmissible(double tol = 1e-12) const;

    /// Circumcenter of this simplex in **barycentric** coordinates
    /// (λ_0..λ_d, Σλ = 1), computed intrinsically from the signature-aware edge
    /// lengths (no embedding). λ_i is the weight on ``getVertices()[i]``. Solves
    /// G β = ½·diag(G) with G the Gram matrix relative to vertex 0, then
    /// λ_0 = 1 − Σβ, λ_i = β_i. Eigen-free (uses the determinant/cofactor
    /// helpers). A vertex falling outside the simplex has a negative λ.
    [[nodiscard]] std::vector<std::complex<double>> circumcenterBarycentric() const;

    /// Signed circumradius squared R² of this simplex (intrinsic, signature-
    /// aware): R² = ½·Σ_i β_i G_ii. Positive for a spacelike simplex; can be
    /// negative when the circumcenter–vertex displacement is timelike.
    [[nodiscard]] std::complex<double> circumradiusSquared() const;

    /// True iff this simplex is a genuine face of the current triangulation:
    /// some registered **top** cell (a (d+1)-vertex simplex, d the ambient
    /// spacetime dimension) contains all of this simplex's vertices. A Pachner
    /// move that removes a cell can leave a lazily-materialised sub-face (facet
    /// or (d-2)-hinge) registered with no surviving top coface — an *orphan*.
    /// Such an orphan is no longer part of the simplicial complex and must not
    /// contribute to the Regge action (it would carry a spurious bare-2π
    /// deficit). The hinge set the action sums over is exactly the (d-2)-faces
    /// for which this returns ``true``. Mirrors the top-cell scan in
    /// ``deficitAngle``; requires a non-null owning spacetime.
    [[nodiscard]] bool hasTopCoface() const;

    /// Signed **circumcentric dual cell volume** |★σ| of this k-simplex in the
    /// surrounding complex (the dual is (n−k)-dimensional, n = top dimension
    /// reached via cofaces). Built from circumcenters by the standard DEC
    /// recursion |★σ_k| = (1/(n−k)) Σ_{σ_{k+1}⊃σ_k} h·|★σ_{k+1}|, with a top
    /// cell's dual a point (volume 1) and h the signed circumcentric height
    /// between c(σ_k) and c(σ_{k+1}); signs follow the circumcenter's
    /// barycentric coordinate at the opposite vertex. Signature-aware: a
    /// timelike height contributes signed content (sign·√|h²|), matching
    /// ``volume()``. Negative content is meaningful, not an error.
    [[nodiscard]] std::complex<double> dualVolume() const;

    /// Exact analytic gradient of this hinge's ``dualVolume`` with respect to the
    /// squared length of each surrounding edge:
    /// \f$ \partial |\!\star\!\sigma| / \partial \ell^2_e \f$. Differentiates the
    /// DEC recursion through the circumradii, with
    /// \f$ R^2 = h^\top G^{-1} h \f$ (h = ½ diag G) so
    /// \f$ \partial R^2 = 2(\partial h)^\top\beta - \beta^\top(\partial G)\beta \f$,
    /// \f$ \beta = G^{-1}h \f$ (the Gram matrix is linear in \f$ \ell^2 \f$).
    /// Implemented for the \f$ (n-2) \f$-hinge case the Regge action needs (an
    /// edge in 3D), the dual being the two-level edge→facet→top recursion. Keyed
    /// by sorted vertex-id edge over the top cells touching the hinge. Returns an
    /// empty map for other codimensions.
    [[nodiscard]] std::map<std::pair<std::uint64_t, std::uint64_t>, std::complex<double>>
    dualVolumeGradient() const;

    /// Does this hinge's dual geometry sit on a point where its derivatives can
    /// fail to exist?
    ///
    /// The circumcentric dual is built from square roots of circumradius
    /// differences: \f$ \sqrt{R^2_{\text{facet}} - R^2_{\text{hinge}}} \f$ and
    /// \f$ \sqrt{R^2_{\text{top}} - R^2_{\text{facet}}} \f$, the distances
    /// between successive circumcentres. A difference vanishes when two
    /// circumcentres coincide, which in Lorentzian signature happens for real
    /// geometries -- two cells can share a null circumsphere.
    ///
    /// The dual content itself stays finite there, because it only multiplies by
    /// those roots. Its derivatives divide by them, and \f$ \sqrt{x} \f$ has
    /// infinite slope at the origin, so a derivative that moves the vanishing
    /// difference genuinely does not exist. ``dualVolumeGradient`` and
    /// ``dualVolumeHessian`` report non-finite entries in exactly that case, and
    /// this predicate is how a caller identifies which hinges are responsible
    /// without hunting for the NaN.
    ///
    /// True does not by itself mean a derivative diverged: a difference that
    /// vanishes and stays vanishing under a given edge contributes zero, which
    /// both derivative routines return. It means the hinge is a candidate.
    [[nodiscard]] bool dualGeometryIsDegenerate() const;

    /// Exact analytic Hessian of this hinge's ``dualVolume``:
    /// \f$ \partial^2 |\!\star\!\sigma| / \partial \ell^2_e \partial \ell^2_f \f$.
    /// One derivative beyond ``dualVolumeGradient``: the DEC facet→top recursion
    /// carried to second order through the circumradii (``d2CircumR2``, with
    /// \f$ \partial_f\beta = G^{-1}(\partial_f h - \partial_f G\,\beta) \f$) and the
    /// signed-sqrt heights (\f$ g''(x) = -\mathrm{sign}(x)/4|x|^{3/2} \f$). Keyed
    /// by the (sorted) edge pair; symmetric.
    [[nodiscard]] std::map<std::pair<std::pair<std::uint64_t, std::uint64_t>,
                                     std::pair<std::uint64_t, std::uint64_t>>,
                           std::complex<double>>
    dualVolumeHessian() const;

    /// Diagonal Hodge-star ratio ⋆ = |★σ| / |σ| (dual content over primal
    /// content) for this simplex — the bridge between the primal Laplacian
    /// weights and the dual Regge action.
    [[nodiscard]] std::complex<double> hodgeStar() const;

    /// Determinant of a square matrix (flat row-major, size n x n).
    [[nodiscard]] static std::complex<double> determinant(
        const std::vector<std::complex<double>> &M, int n);

    /// Cofactor matrix of a square matrix (flat row-major, size n x n).
    [[nodiscard]] static std::vector<std::complex<double>> cofactorMatrix(
        const std::vector<std::complex<double>> &M, int n);

    // ==================== Computational & Utility Methods ====================
    template<typename T> T binomial(unsigned n, unsigned k) const;


    // ==================== Modification Methods ====================
    bool addEdge(const EdgePtr &edge);
    bool removeEdge(const EdgePtr &edge);
    static void registerToVertices(Simplex* simplex);

    /// If you're working in a 3-complex (tetrahedrons), \f$ K \f$ this method should be appropriately called on a
    /// 2-simplex (a triangle), \f$ \sigma^2 \f$ or in general for a given k-complex, \f$ K \f$ you should just be
    /// calling this method on simplices of dimension k-1. It creates a new k-simplex by writing drawing edges from
    /// each vertex of this Facet to the new vertex. This creates a new k-simplex with a shared face (this Simplex!) in
    /// effectively O(1) time.
    ///
    /// @param vertex A new, standalone, orphaned vertex with no existing edges or associated simplices.
    /// @returns A pair of {simplex, facets}; The new k-simplex created by coning `vertex` to this facet and a vector of
    ///   new exterior facets resulting from the new simplex.
    std::pair<SimplexPtr, Simplices> cone(VertexPtr vertex);

    // ==================== Validation ====================
    void validate() const;

    // ==================== Operators ====================
    bool operator==(const Simplex &other) const noexcept;
    bool operator==(const Simplex* other) const noexcept;

    /// How many times this simplex's storage slot has been handed out.
    [[nodiscard]] std::uint32_t generation() const noexcept { return generation_; }

    // ==================== Public Data ====================
    Fingerprint fingerprint{};
    bool initialized{false};

    /// Indices maintained by Spacetime for O(1) swap-and-pop removal.
    /// UINT32_MAX means "not registered in that vector".
    std::uint32_t vecIdx_{UINT32_MAX};    // index in Spacetime::simplicesVec
    std::uint32_t poolSlot_{UINT32_MAX};  // index in Spacetime::simplexStorage_
    /// How many times this storage slot has been handed out.
    ///
    /// A slot is reused once the simplex that held it is removed, so a raw
    /// ``Simplex*`` kept across a removal can end up addressing a different
    /// simplex. A holder that also kept the generation it saw can tell the two
    /// apart; see ``SimplexRef``.
    std::uint32_t generation_{0};
    /// True while this slot is queued for reuse.
    ///
    /// A removed simplex can be registered again before the queue is drained,
    /// and a re-registered simplex can be removed again. Without this the slot
    /// would be queued twice and handed to two different simplices.
    bool pendingFree_{false};
    std::uint32_t topVecIdx_{UINT32_MAX}; // index in Spacetime::topSimplicesVec

#ifdef TESSERA_ASSERTIONS
    OwnershipManager<IdType, SimplexPtr, SimplexPtrHash, SimplexPtrEq> ownershipManager{};
#endif

    // ==================== Commented Out / Future Methods ====================
    /// Returns the hinges of the simplex. A hinge is a simplex contained within a higher dimensional simplex. The hinge
    /// is one dimension lower than the "parent" simplex.
    /// For a 4-simplex, \f$ \sigma = {v_0, ..., v_4} \f$ there are 10 edges and 10 triangular hinges.
    /// In this case a hinge is any triangle \f$ {v_i, v_j, v_k} \f$. There are \f$ \binom{5}{3} = 10 \f$ such
    /// triangles.
    ///
    /// The curvature at the hinge is the deficit angle.
    ///
    // const Simplices getHinges() const;

    /// Assuming the simplex is a hinge; returns the deficit angle associated with the hinge.
    ///
    /// The deficit angle is given by:
    ///
    /// \f[
    /// \epsilon = 2 \pi - \sum_{\sigma \supset h} \theta_h^{(\sigma)}
    /// \f]
    ///
    /// \f$ \theta_h^{(\sigma)} \f$ is the 4D dihedral angle between the two tetrahedral faces of simplex \f$ \sigma \f$
    /// meeting along triangle (hinge) \f$ h \f$.
    ///
    /// Or in english; the deficit angle is equal to \f$ 2 \pi \f$ minus the sum of the 4D dihedral angle of each
    /// simplex between the two tetrahedral faces meeting along triangle \f$ h \f$.
    ///
    /// When the hinge is exterior/on a boundary; the \f$ 2 \pi \f$ is replaced with \f$ \pi \f$.
    ///
    // const double getDeficitAngle() const;

    /// Compute dihedral angles from edge lengths.
    ///
    /// Let \f$ C \f$ be the cofactors of \f$ G \f$, \f$ C = cof(G) \f$ (a matrix of cofactors). Then the dihedral angle
    /// between the two tetrahedral faces opposite vertices \f$ i \f$ and \f$ j \f$ is given by:
    ///
    /// \f[
    /// cos(\theta_{ij}) = - \frac{C_{ij}}{\sqrt{C_{ii} C_{jj}}}, i \neq j, i, j \in {0, ..., n}
    /// \f]
    ///
    /// Map \f$ (i, j) \f$ to the hinge (triangle for a 4-simplex) opposite that pair.
    ///
    // const double computeDihedralAngles() const;
    // void computeEdges();

    /// This method replaces the vertex only, Edge (s) should be replaced by the Spacetime, because it maintains the
    /// global lookup for Edge (s). If the Edge source/target is replaced; it's not enough to update the Edge, since
    /// squaredLength data could be lost.
    ///
    /// WARNING: This Simplex must be removed from it's containers prior to calling this method. NOT removing it from it's
    ///   containers _first_ (and adding back in after) results in UNDEFINED BEHAVIOR!
    ///
    /// @param oldVertex The Vertex to replace
    /// @param newVertex The vertex with which to replace it.
    /// @return
    bool replaceVertex(const VertexPtr &oldVertex, const VertexPtr &newVertex);

    /// No-op. The Simplex stores VertexPtrs and reads IDs through
    /// them, so ``Spacetime::swapVertexLabels`` writing a new ID on
    /// a Vertex is visible to the Simplex on its next ``getId()``.
    /// Kept as an API hook for callers that may still invoke it.
    void updateVertexId(IdType oldId, IdType newId) { (void)oldId; (void)newId; }

    /// No-op — see updateVertexId.
    void swapVertexIds(IdType id1, IdType id2) { (void)id1; (void)id2; }

    bool isInitialized() const noexcept;

    /// True iff this Simplex has been logically removed from its Spacetime
    /// (i.e. ``Spacetime::unregisterSimplex`` has run on it).  Use this to
    /// validate cached ``Simplex*`` pointers before dereferencing — with
    /// stable-address storage the pointer itself is always safe to read,
    /// but a stale simplex has its child vectors cleared, so iterating
    /// them is a silent no-op rather than the live data the caller likely
    /// expected.  Equivalent to ``vecIdx_ == UINT32_MAX``.
    [[nodiscard]] bool isStale() const noexcept {
      return vecIdx_ == UINT32_MAX;
    }

    /// Release this Simplex's heap-allocated children (vertex/edge/facet/
    /// coface vectors), shrinking them to zero capacity.  Called by
    /// ``Spacetime::unregisterSimplex`` once the simplex has been removed
    /// from all live indices: the Simplex shell stays in
    /// ``Spacetime::simplexStorage_`` at its stable address (so cached
    /// ``Simplex*`` remain dereferenceable and read empty children), but the
    /// memory backing those children is returned to the allocator.  Callers
    /// MUST NOT invoke this while the simplex is still registered.
    void releaseChildren() noexcept;
  private:
    /// The current top (d+1)-cells that contain every vertex of this simplex,
    /// deduplicated by fingerprint. Scans the simplex lists of *all* this
    /// simplex's vertices (not just ``vertices[0]``): a Pachner remove∘rollback
    /// recreates a deleted vertex as a fresh object, so a sub-simplex created
    /// before the removal can keep a (now stale, empty-list) pointer to the old
    /// vertex while the genuine cofaces register on the new one. Anchoring the
    /// scan on a single stored vertex would then miss them — yielding a spurious
    /// bare-2π deficit. Membership is tested by vertex id, so the mixed-pointer
    /// case resolves correctly. The set is identical to a single-vertex scan
    /// whenever no vertex is stale.
    [[nodiscard]] std::vector<Simplex *> incidentTopCells() const;

    /// Bordered Cayley-Menger matrix built over this simplex's vertices sorted by
    /// ascending id -- the ChainComplex reference orientation. ``pos1`` is filled
    /// with each vertex id's 1-based bordered position in that canonical order.
    /// The signed (Lorentzian) dihedral-angle cofactor formula is sensitive to the
    /// order a cell's vertices happen to be stored in (a Pachner move stores them
    /// in causal, not sorted, order), which would make ``deficitAngle``
    /// -- and hence ``dualReggeAction`` -- depend on build history rather than on
    /// the geometry. Evaluating the standard formula in this fixed reference frame
    /// makes the deficit a true relabelling/order invariant. Identical to
    /// ``cayleyMengerMatrix`` when the cell is already stored sorted.
    /// Always signature-aware: there is no Euclidean/Wick-rotated mode (#641).
    [[nodiscard]] std::vector<std::complex<double>> cayleyMengerCanonical(
        std::vector<std::pair<std::uint64_t, int>> &pos1) const;

    /// The 1-based bordered position of \a vertexId in the canonical order, or
    /// 0 when the id is not one of this cell's vertices.
    [[nodiscard]] static int canonicalPosition(
        const std::vector<std::pair<std::uint64_t, int>> &pos1,
        std::uint64_t vertexId) noexcept {
      for (const auto &entry : pos1)
        if (entry.first == vertexId) return entry.second;
      return 0;
    }

    /// Ambient top dimension n for the circumcentric-dual recursion. When this
    /// simplex carries an owning spacetime, n is read straight off the metric
    /// signature — robust to stale/orphan cofaces a Pachner move may leave in
    /// this simplex's coface list (which would otherwise misdirect a
    /// ``getCofaces()[0]`` walk and corrupt ``dualVolume``). Falls back to the
    /// historical coface walk for coordinate-free fixtures with no spacetime.
    [[nodiscard]] int ambientTopDimension() const;

    /// The current geometry-revision key: structural revision plus the sum of
    /// the incident edges' ``lengthRevision()`` counters. Strictly increases
    /// under every incident ``setLength``/``addEdge``/``removeEdge``, so a
    /// section key equal to it proves the section's payload is current.
    [[nodiscard]] std::uint64_t geometryRevisionKey() const noexcept;
    /// Fill helpers for the four cache sections; each runs the direct pipeline
    /// under ``geomCacheMutex_`` and publishes by storing ``key`` last.
    void fillGramSection(std::uint64_t key) const;
    void fillGramCofSection(std::uint64_t key) const;
    void fillCMSection(std::uint64_t key) const;
    void fillCMCanonSection(std::uint64_t key) const;
    /// Flat (n × n) table of this simplex's signed squared edge lengths by
    /// LOCAL index in ``ordering`` (0 where the pair carries no edge), built
    /// in one linear pass over the edge list with direct id matching — the
    /// shared gather behind ``gramMatrix``/``cayleyMengerMatrix``/
    /// ``cayleyMengerCanonical``, replacing their per-entry hashed lookups.
    [[nodiscard]] std::vector<std::complex<double>> localSquaredLengths(
        const VertexPtrs &ordering) const;

    Spacetime *spacetime{nullptr};
    TemporalOrientation orientation{};

    VertexPtrs vertices{};

    Edges edges{};

    Simplices facets{};
    Simplices cofaces{};

    /// The cache payload plus its concurrency state, bundled so ``Simplex``
    /// stays copyable: atomics and mutexes are not, and simplices ARE copied
    /// by value elsewhere in the tree. A copy of a simplex has the same
    /// geometry but must refill on first use (fresh keys); an assignment
    /// additionally RESETS the target's keys — its stale payload belongs to
    /// its old geometry, and the copied edge revisions could otherwise
    /// reproduce a previously-published key and false-hit.
    struct GeomCacheState {
      GeomCache cache{};
      /// Per-section publication keys (0 = never filled; real keys are ≥ 1
      /// because ``structuralRevision_`` starts at 1).
      std::atomic<std::uint64_t> gramKey{0};
      std::atomic<std::uint64_t> gramCofKey{0};
      std::atomic<std::uint64_t> cmKey{0};
      std::atomic<std::uint64_t> cmCanonKey{0};
      std::mutex mutex;
      GeomCacheState() = default;
      GeomCacheState(const GeomCacheState &) noexcept {}
      GeomCacheState(GeomCacheState &&) noexcept {}
      GeomCacheState &operator=(const GeomCacheState &) noexcept {
        reset();
        return *this;
      }
      GeomCacheState &operator=(GeomCacheState &&) noexcept {
        reset();
        return *this;
      }
      void reset() noexcept {
        gramKey.store(0, std::memory_order_relaxed);
        gramCofKey.store(0, std::memory_order_relaxed);
        cmKey.store(0, std::memory_order_relaxed);
        cmCanonKey.store(0, std::memory_order_relaxed);
      }
      /// Invalidate the cache AND give its buffers back.
      ///
      /// ``reset`` deliberately keeps the buffers allocated: a live simplex
      /// whose geometry moved will refill them, and reusing the allocation is
      /// cheaper than making a new one. A simplex that has been removed never
      /// refills, and its shell is never freed -- ``Spacetime::simplexStorage_``
      /// holds every simplex ever created at a stable address so that cached
      /// ``Simplex*`` stay dereferenceable -- so its buffers would otherwise be
      /// held for the rest of the run.
      ///
      /// The keys go to zero before the payload is dropped, so a reader can
      /// never see a key that publishes freed data.
      void release() noexcept {
        std::lock_guard<std::mutex> guard(mutex);
        reset();
        GeomCache empty{};
        std::swap(cache, empty);
      }
    };
    /// Lazily allocated holder for ``GeomCacheState``.
    ///
    /// Held by value the cache costs every simplex about 270 bytes, roughly
    /// half the object, whether or not its geometry is ever evaluated. A Monte
    /// Carlo sweep evaluates none of it, and ``Spacetime::simplexStorage_``
    /// keeps every simplex ever created at a stable address, so by-value the
    /// cost is paid by every simplex the chain has ever touched for the rest of
    /// the run. Behind a pointer it costs eight bytes until something asks.
    ///
    /// The fill paths run under OpenMP, so the allocation is published with a
    /// compare-exchange: the loser of a race frees its own attempt and takes
    /// the winner's. Copy and move reset to empty, matching ``GeomCacheState``
    /// itself -- a copied simplex must refill rather than inherit a payload
    /// that belongs to the original's geometry.
    struct GeomCacheSlot {
      mutable std::atomic<GeomCacheState *> ptr{nullptr};
      GeomCacheSlot() = default;
      GeomCacheSlot(const GeomCacheSlot &) noexcept {}
      GeomCacheSlot(GeomCacheSlot &&) noexcept {}
      GeomCacheSlot &operator=(const GeomCacheSlot &) noexcept {
        release();
        return *this;
      }
      GeomCacheSlot &operator=(GeomCacheSlot &&) noexcept {
        release();
        return *this;
      }
      ~GeomCacheSlot() { release(); }
      void release() const noexcept {
        delete ptr.exchange(nullptr, std::memory_order_acq_rel);
      }
      [[nodiscard]] GeomCacheState &get() const {
        if (auto *live = ptr.load(std::memory_order_acquire)) return *live;
        auto *fresh = new GeomCacheState();
        GeomCacheState *expected = nullptr;
        if (ptr.compare_exchange_strong(expected, fresh,
                                        std::memory_order_acq_rel,
                                        std::memory_order_acquire))
          return *fresh;
        delete fresh;
        return *expected;
      }
    };
    mutable GeomCacheSlot geomCacheSlot_{};
    /// The geometry cache, allocated on first use.
    [[nodiscard]] GeomCacheState &geomCacheState_() const {
      return geomCacheSlot_.get();
    }
    /// Bumped by ``addEdge``; on ``removeEdge`` it absorbs the removed edge's
    /// revision plus one, so ``geometryRevisionKey`` never repeats a value it
    /// held before the removal.
    std::uint64_t structuralRevision_{1};

    bool _isSpatial;
    double ti{std::numeric_limits<double>::max()};
    double tf{-std::numeric_limits<double>::max()};
};

/// A ``Simplex*`` together with the generation its slot was on when taken.
///
/// Storage slots are reused once the simplex holding one is removed, so a bare
/// pointer kept across a removal can address a live simplex that is not the one
/// it was taken from. Pairing the pointer with a generation makes that case
/// answerable instead of silent: ``alive()`` is false once the slot has been
/// handed out again.
///
/// Use it for a reference held across a mutation. A pointer used and dropped
/// within one operation cannot go stale and does not need this.
struct SimplexRef {
    SimplexPtr simplex{nullptr};
    std::uint32_t generation{0};

    SimplexRef() = default;
    explicit SimplexRef(SimplexPtr s) noexcept
        : simplex(s), generation(s ? s->generation() : 0) {}

    /// True when the slot still holds the simplex this was taken from.
    [[nodiscard]] bool alive() const noexcept {
        return simplex != nullptr && simplex->generation() == generation;
    }

    /// The simplex, or nullptr once its slot has been reused.
    [[nodiscard]] SimplexPtr get() const noexcept {
        return alive() ? simplex : nullptr;
    }

    explicit operator bool() const noexcept { return alive(); }
};

}

#endif //TESSERA_SIMPLEX_H
