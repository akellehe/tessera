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

/// # Simplex
///
/// A simplex generalizes the triangle and the tetrahedron to arbitrary dimension and is
/// defined by its vertices: a k-simplex \f$ \sigma^k \f$ in a simplicial complex
/// \f$ K \f$ is a set of k+1 vertices. It carries a volume \f$ V_s \f$ computed from its
/// edge lengths.
///
/// Simplices are built by coning: vertices are declared, then a simplex is coned onto
/// them. The connecting edges are passed in alongside the vertices so the simplex need
/// not rediscover them. Most of the construction cost is in enumerating facets over
/// subsets of the vertices.
///
/// The geometry is Lorentzian throughout: squared edge lengths are signed (timelike
/// \f$ l^2 < 0 \f$, spacelike \f$ l^2 > 0 \f$), volumes come from the Cayley-Menger
/// determinant, and dihedral and deficit angles are complex.
///
/// Reference: Regge, "General relativity without coordinates", Nuovo Cimento 19, 558
/// (1961).
///
class Simplex {
  public:
    // ==================== Static Factory Methods ====================
    static Simplex* create(Spacetime *spacetime_, const VertexPtrs &vertices_, const Edges &edges_);
    static Simplex* create(Spacetime *spacetime_, const VertexPtrs &vertices_, const Edges &edges_, const TemporalOrientation &orientation_);
    [[nodiscard]] static std::size_t computeNumberOfEdges(std::size_t k);

    // ==================== Constructors & Initialization ====================
    /// Builds a simplex on \p vertices_ and \p edges_, deriving the temporal
    /// orientation from the vertex times.
    explicit Simplex(Spacetime *spacetime_, const VertexPtrs &vertices_, Edges edges_);
    /// Builds a simplex on \p vertices_ and \p edges_ with \p orientation_
    /// supplied rather than derived.
    Simplex(Spacetime *spacetime_, const VertexPtrs &vertices_, Edges edges_ ,const TemporalOrientation &orientation_);

    std::uint64_t size() const noexcept { return vertices.size(); }

    /// Registers this simplex on each of its member vertices. The Spacetime is the
    /// canonical owner of a Simplex; initialization only hands the vertices a pointer to
    /// it, so the complex can be walked from a vertex to the simplices containing it.
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
    /// The simplex's temporal orientation, used when causality is being preserved. See
    /// TemporalOrientation.
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

    /// O(1) membership test.
    [[nodiscard]] bool hasVertex(const VertexPtr &vertex) const;

    /// A lookup table \f$ Id \rightarrow Vertex \f$ over this simplex's vertices.
    [[nodiscard]] VertexIdMap getVertexIdLookup() const noexcept;


    // ==================== Edge Queries ====================
    /// @returns Edges in traversal order (the order of input vertices).
    [[nodiscard]] const Edges &getEdges() const;
    [[nodiscard]] std::size_t getNumberOfEdges() const;

    /// Edges of the simplex in traversal order. The edges are effectively undirected:
    /// the stored direction relates to vertex order only, so vertices
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
    /// Facets are returned in canonical topological order: facet \f$ i \f$ is this
    /// simplex with the \f$ i \f$-th vertex of ``getVertices()`` removed and the rest
    /// kept in their original relative order. That relative order is the facet's induced
    /// orientation from this simplex.
    ///
    /// The orientation of facet \f$ i \f$ relative to this simplex is therefore
    /// \f$ (-1)^i \f$, its coefficient in the simplicial boundary
    /// \f$ \partial\sigma = \sum_i (-1)^i \,\sigma\!\setminus\! v_i \f$. The return
    /// order carries the orientation, so a caller reads it off the facet's index; there
    /// is no separate orientation flag. The vertex sorting used by ``Fingerprint`` is a
    /// set-identity key and is unrelated to this orientation.
    ///
    /// This orientation is per-simplex. A globally consistent boundary operator
    /// (\f$ \partial^2 = 0 \f$) also requires the complex's simplices to share a
    /// coherent vertex ordering, which is a property of how the complex was built rather
    /// than of this method.
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

    /// Unregister a coface from this simplex. Called during simplex removal in Pachner
    /// moves, to keep the coface bookkeeping consistent.
    void removeCoface(SimplexPtr simplex);

    /// Whether this simplex is a coface of the given one, i.e. whether all of that
    /// simplex's vertices are contained in this simplex.
    /// @param simplex The candidate lower-dimensional simplex
    /// @param shallow If true, also require the dimension difference to be exactly 1
    bool isCofaceTo(const SimplexPtr &simplex, bool shallow=true) const;

    [[nodiscard]] bool hasCoface(SimplexPtr simplex) const;

    ///
    /// Cofaces are stored, not recomputed on demand: attaching one simplex to another
    /// registers it as a coface of the face it attaches at. Removing a Simplex, Edge or
    /// Vertex within that face cascades up the ownership chain
    /// \f[
    /// Vertex \subset Edge \subset Simplex \subset Spacetime
    /// \f]
    ///
    /// @return The set of k-simplices that share this face.
    [[nodiscard]] const Simplices &getCofaces() const noexcept;

    /// The maximum number of (k+1)-cofaces that can be joined to this k-simplex. To ask
    /// whether a simplex is on the boundary and so available for gluing, use
    /// `isBoundary()`.
    ///
    /// A coface of \f$ \sigma^k \f$ is an m-simplex \f$ \sigma^m \f$ with
    /// \f$ m \gt k \f$ and \f$ \sigma^k \subset \sigma^m \f$. That count is unbounded in
    /// general, so this restricts to \f$ m = k + 1 \f$: gluing happens on a k-face of a
    /// (k+1)-simplex, and the bound is then the number of faces of the simplex.
    ///
    /// @return The number of faces of this simplex.
    std::size_t maxKPlusOneCofaces() const;

    // ==================== State Queries ====================
    /// True when every vertex lies on the same time slice (a purely spatial simplex).
    [[nodiscard]] bool isSpatial() const noexcept { return _isSpatial; }

    /// @deprecated Use isSpatial(). The name is misleading: this returns true for
    /// spatial simplices (all vertices at the same time), not for timelike ones.
    [[nodiscard]] bool isTimelike() const noexcept { return _isSpatial; }

    /// True when this simplex has fewer than 2 cofaces, i.e. it lies on
    /// the boundary of the complex and has a free face available for gluing.
    [[nodiscard]] bool isBoundary() const noexcept;

    /// True when any facet of this simplex is on the boundary (has < 2 cofaces).
    bool hasBoundaryFacet();
    std::uint64_t hash() const noexcept;

    // ==================== Geometry ====================
    //
    // Complexified geometry: every entry point below consumes the full complex squared
    // interval l^2. There is no Wick/magnitude path and no Re(l^2) projection;
    // MultiCobordism::runStage2 optimizes these same complex coordinates.

    /// Gram matrix of this simplex from its edge lengths.
    /// Returns a flat (d x d) row-major matrix where d = size() - 1.
    /// Vertex 0 is the origin: G_ij = 1/2(s(v0,vi) + s(v0,vj) - s(vi,vj)),
    /// where s(.,.) is the squared edge length.
    ///
    /// The geometry is always signature-aware: the signed l^2 is kept, so a timelike
    /// edge (l^2 < 0) carries its Lorentzian sign into G and det(G) records the metric
    /// signature of the cell. There is no Euclidean or Wick-rotated mode.
    [[nodiscard]] std::vector<std::complex<double>> gramMatrix() const;

    /// Length-derived geometry cache. The Gram / Cayley-Menger pipeline (matrix,
    /// determinant, cofactors) is a pure function of this simplex's edge lengths, and
    /// recomputing it once per (hinge, coface) evaluation dominates the action,
    /// gradient and Hessian paths. The four sections below memoize those products, each
    /// filled lazily by the same
    /// ``gramMatrix``/``cayleyMengerMatrix``/``cayleyMengerCanonical``/
    /// ``determinant``/``cofactorMatrix`` calls the direct path runs, so a cached value
    /// is bit-for-bit the recomputed one.
    ///
    /// Invalidation is by key comparison, never by callback: the key is the
    /// sum of the incident edges' monotone ``Edge::lengthRevision()`` counters
    /// plus this simplex's structural revision (bumped by ``addEdge``/
    /// ``removeEdge``, compensated so the key never repeats an old value).
    /// Any ``setLength`` on an incident edge therefore misses the cache and refills on
    /// next use, including perturb/restore probes, whose restore bumps the revision
    /// again: a conservative refill, never a stale hit.
    ///
    /// Concurrency: fills are serialized by a per-simplex mutex and published
    /// by a release-store of the section key; readers that observe the current
    /// key use the payload lock-free. Lengths are mutated only in the serial phases
    /// between parallel evaluations (stage-2 line search, move apply/rollback), so
    /// within a parallel region the key is constant and the returned references stay
    /// valid. The references do not survive a length mutation.
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
      /// A flat vector rather than a map: a cell carries d+1 vertices, so this holds
      /// at most six entries in four dimensions. An ``unordered_map`` would cost a
      /// bucket array and a node allocation per entry, on every simplex, to answer a
      /// lookup that a linear scan over six contiguous pairs answers faster.
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
    /// give the dihedral angles (see ``dihedralAngle``). Always signature-aware: there
    /// is no Euclidean or Wick-rotated mode.
    [[nodiscard]] std::vector<std::complex<double>> cayleyMengerMatrix() const;

    /// Lorentzian (Sorkin) dihedral angle at ``hinge`` within this top simplex, as a
    /// complex number, from the signed (non-Wick) Cayley-Menger cofactors. One
    /// expression covers all three Sorkin / Asante-Dittrich regimes m ∈ {0, 1, 2}:
    ///
    /// \f[ \cos\theta = \frac{-C_{ij}}{\sqrt{C_{ii}}\,\sqrt{C_{jj}}} \f]
    ///
    /// Two separate principal square roots, not ``sqrt(C_ii*C_jj)``. For complex
    /// \f$a, b\f$ the two differ by a sign exactly when both sit on the negative real
    /// axis: with a unit tetrahedron's ``C_ii = C_jj = -3``, ``sqrt(C_ii*C_jj)`` is
    /// ``+3`` while ``sqrt(C_ii)*sqrt(C_jj)`` is ``(i*sqrt3)(i*sqrt3) = -3``. Folding
    /// the product under one root forces a hand-applied ``(-1)^d`` parity fix, a
    /// three-way branch dispatch and an i<->j anchoring swap; taking the roots
    /// separately makes all three emerge from the branch structure.
    ///
    /// The causal regimes are then cases of one formula rather than code paths:
    ///
    /// * Same-sign cofactors, |r| <= 1: the wedge stays on one side of the light cone
    ///   and the angle is the ordinary real one (m = 0).
    /// * Same-sign cofactors, |r| > 1: the boost regime (m even). ``std::acos`` returns
    ///   a complex value whose imaginary part is the rapidity and whose real part (0 or
    ///   pi) counts crossed quadrants.
    /// * Opposite-sign cofactors: the m = 1 light-cone crossing, one facet direction
    ///   spacelike and one timelike. The denominator turns purely imaginary,
    ///   ``r = i*y``, and the principal
    ///   \f$\arccos(iy) = \pi/2 - i\,\mathrm{asinh}(y)\f$ is Sorkin's quarter turn plus
    ///   a signed boost, with no special case. Around a flat one-ray-per-quadrant
    ///   Minkowski vertex star the four boosts telescope to zero and four crossings sum
    ///   to 2*pi, so ``2*pi - sum = 0`` still holds. This case is generic in causal
    ///   dynamical triangulation (CDT): every base-tetrahedron triangle of a (4,1) cell.
    ///
    /// The boost content is carried everywhere. In the same-sign regimes the imaginary
    /// sign is the principal branch: a wedge's boost orientation is not fixed by edge
    /// lengths alone (a PT reflection flips it at identical l^2), so only crossing
    /// wedges carry an intrinsically signed boost.
    ///
    /// References: Regge, "General relativity without coordinates", 1961;
    /// Sorkin, arXiv:1908.10022; Asante, Dittrich & Padua-Arguelles,
    /// arXiv:2104.00485.
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
    /// branch (\f$ C_{ii}C_{jj} < 0 \f$) differentiates
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
    /// m = 1 crossing (\f$ C_{ii}C_{jj} < 0 \f$)
    /// \f$ d\theta/dy = -i/\sqrt{1+y^2} \f$ and
    /// \f$ d^2\theta/dy^2 = +i\,y/(1+y^2)^{3/2} \f$ with \f$ y = C_{ij}/D \f$.
    /// Keyed by the (sorted) edge pair; symmetric. Complex (the boost part is
    /// carried, not truncated).
    [[nodiscard]] std::map<std::pair<std::pair<std::uint64_t, std::uint64_t>,
                                     std::pair<std::uint64_t, std::uint64_t>>,
                           std::complex<double>>
    deficitAngleHessian() const;

    /// Area of this simplex read as a triangular hinge (3 vertices), from Heron's
    /// formula on the three squared edge lengths. The geometry is fully Lorentzian.
    ///
    /// The Heron radicand is carried under a complex square root, so a triangle with a
    /// negative radicand — every timelike (negative-content) triangle, such as the
    /// mixed-causal hinge of a CDT (4,1) cell — returns an imaginary area. Zero is not
    /// the area of those triangles; it is only what a real return type can represent.
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
    /// circumcentric `dualVolumeGradient` uses. This is the per-degree Hodge
    /// inner-product weight gradient (the weights \f$ W_k \f$ are signed simplex
    /// volumes), from which the arbitrary-degree analytic
    /// \f$ \partial L_k/\partial\ell^2 \f$ and the general-k \f$ r_U \f$ gradient
    /// follow.
    [[nodiscard]] std::map<std::pair<std::uint64_t, std::uint64_t>, std::complex<double>>
    volumeGradient() const;

    /// Exact directional second derivative of the signed `volume()`:
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
    /// precision, with no finite differences: this is the Hessian half of the analytic
    /// \f$ \partial^2 L_k/\partial\ell^2\partial\ell^2 \f$ contraction that the joint
    /// Regge-Hodge descent direction needs. Uses the same
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
    /// timelike (worldline) edge is skipped, returning without checking: its
    /// admissibility is governed by the Lorentzian structure, not the spacelike
    /// triangle inequalities. Fewer than two vertices is trivially admissible.
    void assertSpacelikeAdmissible(double tol = 1e-12) const;

    /// Circumcenter of this simplex in barycentric coordinates
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

    /// True iff this simplex is a genuine face of the current triangulation: some
    /// registered top cell (a (d+1)-vertex simplex, d the ambient spacetime dimension)
    /// contains all of this simplex's vertices. A Pachner move that removes a cell can
    /// leave a lazily materialised sub-face (facet or (d-2)-hinge) registered with no
    /// surviving top coface — an orphan. An orphan is no longer part of the simplicial
    /// complex and does not contribute to the Regge action; it would carry a spurious
    /// bare-2π deficit. The hinge set the action sums over is exactly the (d-2)-faces
    /// for which this returns ``true``. Mirrors the top-cell scan in
    /// ``deficitAngle``; requires a non-null owning spacetime.
    [[nodiscard]] bool hasTopCoface() const;

    /// Signed circumcentric dual cell volume |★σ| of this k-simplex in the
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
    /// DEC recursion through its circumcentric heights. Each height, written in
    /// the recursion as \f$ \pm\sqrt{R^2_{\mathrm{coface}} - R^2_{\mathrm{face}}} \f$,
    /// is differentiated in the equal product form
    /// \f$ \lambda_v \sqrt{\det G_{\mathrm{coface}} / \det G_{\mathrm{face}}} \f$,
    /// with \f$ \lambda_v \f$ the barycentric coordinate of the coface's
    /// circumcentre at its vertex outside the face. That form stays smooth where
    /// the two circumcentres coincide (\f$ \lambda_v = 0 \f$, as at the right
    /// angles of a Kuhn triangulation), where the chain rule through the root of
    /// the difference divides zero by zero. Implemented for the
    /// \f$ (n-2) \f$-hinge case the Regge action needs (an edge in 3D), the dual
    /// being the two-level edge→facet→top recursion. Keyed by sorted vertex-id
    /// edge over the top cells touching the hinge. Returns an empty map for other
    /// codimensions.
    [[nodiscard]] std::map<std::pair<std::uint64_t, std::uint64_t>, std::complex<double>>
    dualVolumeGradient() const;

    /// Do two successive circumcentres of this hinge's dual coincide?
    ///
    /// The circumcentric dual is built from the heights between successive
    /// circumcentres, \f$ \sqrt{R^2_{\mathrm{facet}} - R^2_{\mathrm{hinge}}} \f$ and
    /// \f$ \sqrt{R^2_{\mathrm{top}} - R^2_{\mathrm{facet}}} \f$ up to sign, and this
    /// predicate is true when one of those differences is exactly zero: the
    /// coface's circumcentre lies in the face's hull. Every Kuhn triangulation
    /// of a cubic lattice has such hinges (the circumcentre of a right triangle
    /// is the midpoint of its hypotenuse).
    ///
    /// The heights are smooth there: each equals
    /// \f$ \lambda_v \sqrt{\det G_{\mathrm{coface}} / \det G_{\mathrm{face}}} \f$,
    /// which ``dualVolumeGradient`` and ``dualVolumeHessian`` differentiate, so
    /// both stay finite at such a hinge. Only a cell of zero content (a singular
    /// Gram matrix), which has no circumcentre, still has its heights
    /// differentiated through the root of the difference.
    [[nodiscard]] bool dualGeometryIsDegenerate() const;

    /// Exact analytic Hessian of this hinge's ``dualVolume``:
    /// \f$ \partial^2 |\!\star\!\sigma| / \partial \ell^2_e \partial \ell^2_f \f$.
    /// One derivative beyond ``dualVolumeGradient``: the DEC facet→top recursion
    /// carried to second order through the same product form of its heights,
    /// with \f$ \partial_f\beta = G^{-1}(\partial_f h - \partial_f G\,\beta) \f$,
    /// \f$ \partial_e\partial_f\beta = -G^{-1}(\partial_e G\,\partial_f\beta +
    /// \partial_f G\,\partial_e\beta) \f$ and the second derivative of the
    /// logarithm of a Gram determinant,
    /// \f$ -\mathrm{tr}(G^{-1}\partial_f G\,G^{-1}\partial_e G) \f$. Finite where
    /// circumcentres coincide. Keyed by the (sorted) edge pair; symmetric.
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

    /// Call this on a simplex of dimension k-1 within a k-complex \f$ K \f$ — on a
    /// triangle \f$ \sigma^2 \f$ in a complex of tetrahedra, for instance. It draws an
    /// edge from each vertex of this facet to the new vertex, producing a new k-simplex
    /// that shares this simplex as a face, in effectively O(1) time.
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

    /// Replaces the vertex only. Edges are replaced by the Spacetime, which owns the
    /// global edge lookup; updating an Edge's source or target in place is not enough,
    /// because the length data would be lost.
    ///
    /// Remove this Simplex from its containers before calling and add it back
    /// afterwards. Mutating it while it sits in a hash-keyed container is undefined
    /// behaviour.
    ///
    /// @param oldVertex The Vertex to replace
    /// @param newVertex The vertex with which to replace it.
    /// @return true if the vertex was replaced.
    bool replaceVertex(const VertexPtr &oldVertex, const VertexPtr &newVertex);

    /// No-op. The Simplex stores VertexPtrs and reads ids through them, so a new id
    /// written on a Vertex by ``Spacetime::swapVertexLabels`` is visible to the Simplex
    /// at its next ``getId()``. Present as an API hook for callers that invoke it.
    void updateVertexId(IdType oldId, IdType newId) { (void)oldId; (void)newId; }

    /// No-op — see updateVertexId.
    void swapVertexIds(IdType id1, IdType id2) { (void)id1; (void)id2; }

    bool isInitialized() const noexcept;

    /// True iff this Simplex has been logically removed from its Spacetime, i.e.
    /// ``Spacetime::unregisterSimplex`` has run on it. Validates a cached ``Simplex*``
    /// before dereferencing: with stable-address storage the pointer is always safe to
    /// read, but a stale simplex has its child vectors cleared, so iterating them is a
    /// silent no-op rather than live data. Equivalent to ``vecIdx_ == UINT32_MAX``.
    [[nodiscard]] bool isStale() const noexcept {
      return vecIdx_ == UINT32_MAX;
    }

    /// Release this Simplex's heap-allocated children (vertex, edge, facet and coface
    /// vectors), shrinking them to zero capacity. Called by
    /// ``Spacetime::unregisterSimplex`` once the simplex has been removed from all live
    /// indices: the Simplex shell stays in ``Spacetime::simplexStorage_`` at its stable
    /// address, so a cached ``Simplex*`` stays dereferenceable and reads empty children,
    /// while the memory backing those children returns to the allocator. Not to be
    /// called while the simplex is still registered.
    void releaseChildren() noexcept;
  private:
    /// The current top (d+1)-cells that contain every vertex of this simplex,
    /// deduplicated by fingerprint. Scans the simplex lists of all this simplex's
    /// vertices, not just ``vertices[0]``: a Pachner remove followed by rollback
    /// recreates a deleted vertex as a fresh object, so a sub-simplex created before the
    /// removal can hold a stale, empty-list pointer to the old vertex while the genuine
    /// cofaces register on the new one. Anchoring the scan on a single stored vertex
    /// would miss them and yield a spurious bare-2π deficit. Membership is tested by
    /// vertex id, so the mixed-pointer case resolves correctly. The result matches a
    /// single-vertex scan whenever no vertex is stale.
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
    /// ``cayleyMengerMatrix`` when the cell is already stored sorted. Always
    /// signature-aware: there is no Euclidean or Wick-rotated mode.
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
    /// ``getCofaces()[0]`` walk and corrupt ``dualVolume``). Falls back to a coface walk
    /// for coordinate-free fixtures with no spacetime.
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
    /// Flat (n × n) table of this simplex's signed squared edge lengths by local index
    /// in ``ordering`` (0 where the pair carries no edge), built in one linear pass over
    /// the edge list with direct id matching. This is the shared gather behind
    /// ``gramMatrix``/``cayleyMengerMatrix``/``cayleyMengerCanonical``, in place of a
    /// per-entry hashed lookup.
    [[nodiscard]] std::vector<std::complex<double>> localSquaredLengths(
        const VertexPtrs &ordering) const;

    Spacetime *spacetime{nullptr};
    TemporalOrientation orientation{};

    VertexPtrs vertices{};

    Edges edges{};

    Simplices facets{};
    Simplices cofaces{};

    /// The cache payload plus its concurrency state, bundled so ``Simplex`` stays
    /// copyable: atomics and mutexes are not copyable, and simplices are copied by value
    /// elsewhere in the tree. A copy of a simplex has the same geometry but refills on
    /// first use (fresh keys); an assignment also resets the target's keys, because its
    /// stale payload belongs to its old geometry and the copied edge revisions could
    /// otherwise reproduce a previously published key and hit falsely.
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
      /// Invalidate the cache and give its buffers back.
      ///
      /// ``reset`` keeps the buffers allocated: a live simplex whose geometry moved
      /// refills them, and reusing the allocation is cheaper than making a new one. A
      /// removed simplex never refills, and its shell is never freed, since
      /// ``Spacetime::simplexStorage_`` holds every simplex ever created at a stable
      /// address so cached ``Simplex*`` stay dereferenceable; its buffers would
      /// otherwise be held for the rest of the run.
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
    /// Held by value the cache costs every simplex about 270 bytes, roughly half the
    /// object, whether or not its geometry is ever evaluated. A Monte Carlo sweep
    /// evaluates none of it, and ``Spacetime::simplexStorage_`` keeps every simplex ever
    /// created at a stable address, so by value that cost is paid by every simplex the
    /// chain has ever touched, for the rest of the run. Behind a pointer it costs eight
    /// bytes until something asks.
    ///
    /// The fill paths run under OpenMP, so the allocation is published with a
    /// compare-exchange: the loser of a race frees its own attempt and takes the
    /// winner's. Copy and move reset to empty, matching ``GeomCacheState`` itself, so a
    /// copied simplex refills rather than inheriting a payload that belongs to the
    /// original's geometry.
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
