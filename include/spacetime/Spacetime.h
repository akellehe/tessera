// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_SPACETIME_H
#define TESSERA_SPACETIME_H

#include <cmath>
#include <map>
#include <memory>
#include <optional>
#include <deque>
#include <random>
#include <ranges>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <set>

#include "topologies/Topology.h"
#include "observables/Observable.h"
#include "mesh/EdgeList.h"
#include "Foliation.h"
#include "mesh/VertexList.h"
#include "spacetime/Metric.h"
#include "mesh/Simplex.h"
#include "mesh/FlatHashMap.h"

// === tessera subsystem ns fwd-decls ===
namespace tessera::graph {}
namespace tessera::mesh { class SimplexFilter; }
namespace tessera::observables { class SparseGraph; }
namespace tessera::quantum {}
namespace tessera::simulations {}
namespace tessera::spacetime { class PachnerMove; }
namespace tessera::spacetime {
using namespace ::tessera::mesh;
using namespace ::tessera::graph;
using namespace ::tessera::observables;
using namespace ::tessera::simulations;
using namespace ::tessera::quantum;
enum class SpacetimeType : uint8_t {
  CDT = 0,
  REGGE = 1,
  COSET = 2,
  REGGE_PACHNER = 3,
  GFT_SPIN_FOAM = 4,
  RICCI_FLOW_DISCRETIZATION = 5,
  // Variable complex edge weights (squaredLength * e^{i*phase}) for a
  // Hermitian-weighted complex, unlike CDT's fixed real edge lengths.
  HERMITIAN_WEIGHTED = 6
};

///
/// # Spacetime
///
/// The Spacetime class provides methods to create and manipulate the basic building blocks of a simplicial
/// complex \f$ \mathcal{K} \f$.
///
/// The Spacetime manages the simplicial complex structure, including vertices \f$ V \f$, edges \f$ E \f$, and
/// simplices \f$ \{\sigma^k_i\} \f$ of varying dimensions. It is responsible for constructing and maintaining
/// the topological relationships between these elements.
///
/// The Spacetime Topology is responsible for constructing Simplex(es) and the Topology (subclass) is responsible for
/// building the complex to match that topology.
///
/// Any assertions or state needed by the Topology to build the complex should be implemented in the Simplex.
///
class Spacetime {
  public:
    // ========================================
    // Constructors
    // ========================================

    /// Default constructor. Creates a 4D Lorentzian spacetime with CDT type and Toroid topology.
    Spacetime();
    ~Spacetime() = default;

    /// Parameterized constructor.
    /// @param metric_ The metric tensor defining the signature and dimension
    /// @param spacetimeType_ The type of quantum gravity formulation (CDT, Regge, etc.)
    /// @param alpha_ The fundamental length scale coefficient, the number of times the length of a timelike edge is the
    ///   length of a spacelike edge
    /// @param a_ The fundamental length unit for a spacelike edge.
    /// @param foliation_ The type of foliation for the spacetime. Preferred means spacelike slices are separated by
    ///   timelike slices. None means they can be interspersed.
    /// @param topology_ The topology of the spatial slices (default: Toroid)
    Spacetime(
      std::shared_ptr<Metric> metric_,
      SpacetimeType spacetimeType_,
      std::optional<double> alpha_,
      std::optional<double> a_,
      Foliation foliation_,
      std::optional<std::shared_ptr<Topology> > topology_);

    // ========================================
    // Creation Methods
    // ========================================

    std::pair<SimplexPtr, bool> createSimplex(const VertexPtrs &vertices);

    /// Result of a tracked simplex creation that records freshly-inserted
    /// edges, used by the transactional Pachner-move infrastructure to
    /// know what to undo on rollback.
    struct CreateSimplexResult {
      /// The simplex (existing or freshly created).
      SimplexPtr simplex;
      /// True if the simplex was newly created; false if it already existed.
      bool created;
      /// Edges that this call freshly inserted into the EdgeList (i.e.
      /// where ``EdgeList::tryAdd`` returned ``inserted=true``).
      /// Empty when ``created`` is false (no edges were touched).
      Edges newEdges;
    };

    ///
    /// This method computes energy for every Edge in the Spacetime that doesn't already have an energyDensity assigned
    /// to it.
    ///
    /// Timelike edges go from a time t to a time t + 1.
    /// Spacelike edges go from a point in space to another point in space.
    ///
    /// Energy is calculated as two terms. One is spacelike and is summed over edges (1-simplices). The second is
    /// timelike and is summed over triangles (2-simplices).
    ///
    /// The crazy thing is that the energy is a term of the Hamiltonian present in the time evolution operator,
    ///
    /// U(t, t_0) | n \ket = e^{-iE_n(t - t_0)/\hbar} | n \ket
    ///
    /// and it's the interaction of two systems that define both a triangle and a step forward in time. So for the
    /// operator that causes their interaction, we can just use time evolution with t - t_0 = 1 (one tick forward in
    /// time) and E_n is that which applies to the triangle formed by two systems and their resultant (mixed) state.
    ///
    /// We can calculate E_t as:
    ///
    /// E_t = \frac{1}{8\pi G}\left[\,\sum_{\ell \subset \Sigma_t} L_\ell\,\delta_\ell^{(3)} \;-\; \sum_{\Delta\,\text{bridging}} A_\Delta\,\psi_\Delta\,\right]
    /// \delta_\ell^{(3)} = 2\pi - \sum_{\tau \supset \ell} \theta_{\ell,\tau}
    ///
    /// \psi_\Delta = \sum_{\sigma \supset \Delta} \eta_{\Delta,\sigma}
    ///
    /// \begin{align*}
    /// E_t &= \text{discrete gravitational Hamiltonian (energy) on the spatial slice } \Sigma_t \\
    /// G &= \text{Newton's gravitational constant} \\
    /// \Sigma_t &= \text{3D spatial slice at time } t \text{ (triangulated by tetrahedra)} \\
    /// \ell &= \text{a 1-simplex (edge) lying entirely within } \Sigma_t \text{; "slice edge"} \\
    /// L_\ell &= \text{length of edge } \ell \\
    /// \delta_\ell^{(3)} &= \text{intrinsic 3D deficit angle around edge } \ell \\
    /// \tau &= \text{a 3-simplex (tetrahedron) of } \Sigma_t \\
    /// \theta_{\ell,\tau} &= \text{dihedral angle at edge } \ell \text{ inside tetrahedron } \tau \\
    /// \Delta &= \text{a 2-simplex (triangle) with vertices split between } \Sigma_t,\,\Sigma_{t+1}; \\
    /// &\quad \text{"bridging" or "timelike" triangle} \\
    /// A_\Delta &= \text{area of triangle } \Delta \\
    /// \psi_\Delta &= \text{extrinsic boost deficit around } \Delta \text{ (Lorentzian dihedral sum)} \\
    /// \sigma &= \text{a 4-simplex (pentatope) of the 4D triangulation} \\
    /// \eta_{\Delta,\sigma} &= \text{boost angle (rapidity) at hinge } \Delta \text{ inside 4-simplex } \sigma
    /// \end{align*}
    ///
    /// \begin{align*}
    /// \sum_{\ell \subset \Sigma_t} &: \text{sum over all edges } \ell \text{ contained in the slice } \Sigma_t \\
    /// \sum_{\tau \supset \ell} &: \text{sum over all tetrahedra } \tau \text{ containing edge } \ell \\
    /// \sum_{\Delta\,\text{bridging}} &: \text{sum over all bridging triangles between } \Sigma_t \text{ and } \Sigma_{t+1} \\
    /// \sum_{\sigma \supset \Delta} &: \text{sum over all 4-simplices } \sigma \text{ containing triangle } \Delta
    /// \end{align*}
    ///
    /// Now, in order to determine E_t at each triangle/hinge, we can just say that the temporal triangle carries across
    /// it the spatial energy of the edge at it's base. Concretely; given systems A and B \in \Sigma_t (a spatial slice)
    /// with state \rho_A and \rho_B there is an edge between them representing their mutual information. It should
    /// be assigned at the initial state we set up for the graph. Probably randomly or according to the constraint that
    /// each share some particular amount of mutual information with the other vertices in the slice.
    ///
    /// If we assume some value for the total energy of the system, E_{total}, then we can split that energy across
    /// every (spatial) edge in proportion to the edge length described by the van raamsdonk metric between those
    /// vertices. So each Edge in the initial (totally spatial) state should be assigned energy \frac{l}{E_{total}}.
    ///
    /// Now, given the energy on, E_{AB}, we can understand how time evolution moves them forward. If we have
    /// systems with \rho_A and \rho_B joined by edge E_{A \rightarrow B} then time evolution looks like
    ///
    /// \rho_AB = U(t, t_0) \rho_A \otimes \rho_B U^{\dagger}(t, t_0)
    ///
    /// where U(t, t_0) = e^{-i E_{A \rightarrow B} (t - t_0)}
    ///
    /// and since we're defining one interaction as one step forward in time; we can take (t - t_0) to be 1 and then
    /// we have
    ///
    /// U(t, t_0) = e^{-i E_{A \rightarrow B}}
    ///
    /// to use for mixing the systems. After they're mixed and we have the joint state \rho_{AB} then we can use KI
    /// decomposition to expand the \rho_{AB} (virtual) node into the three physical nodes A', \Sigma_{AB}, B' from
    /// which we actually draw our simplex.
    ///
    /// Once the simplex is drawn, we take that E_{A \rightarrow B} and distribute it across
    /// E_{A' \rightarrow \Sigma_{AB}} and E_{B' \rightarrow \Sigma_{AB}} in proportion to their edge lengths so there
    /// is a constant amount of energy per unit of length:
    ///
    /// \frac{E_{A \rightarrow B}}{d_{VR}(A' \rightarrow \Sigma_{AB}) + d_{VR}(B' \rightarrow \Sigma_{AB})}
    ///
    ///
    ///
    void labelEnergyDensity();


    /// Like ``createSimplex(vertices)`` but also reports the edges this
    /// call freshly inserted into the EdgeList — those that were
    /// auto-created from scratch rather than found existing.  Caller can
    /// use this to undo edge insertions on rollback.
    CreateSimplexResult createSimplexTracked(const VertexPtrs &vertices);

    /// Creates a simplex \f$ \sigma^k \f$ from explicit vertices and edges.
    /// @param vertices The vertices \f$ \{v_0, \ldots, v_k\} \f$ of the simplex
    /// @param edges The edges \f$ \{e_{ij}\} \f$ connecting the vertices
    /// @return {simplex, wasCreated} pair where wasCreated=true if newly created, false if already existed
    std::pair<SimplexPtr, bool> createSimplex(const VertexPtrs &vertices, const Edges &edges);

    /// Creates a simplex \f$ \sigma^{(t_i, t_f)} \f$ with the given causal orientation.
    /// @param numericOrientation Tuple (timelike_initial, timelike_final) defining the orientation
    /// @return {simplex, wasCreated} pair where wasCreated=true if newly created
    std::pair<SimplexPtr, bool> createSimplex(const std::tuple<uint8_t, uint8_t> &numericOrientation);

    /// Creates a k-simplex with randomly positioned vertices and edges of length \f$ \alpha \f$.
    /// @param k The dimension of the simplex (k+1 vertices)
    /// @return {simplex, wasCreated} pair where wasCreated=true if newly created
    std::pair<SimplexPtr, bool> createSimplex(std::size_t k);

    /// Creates a vertex \f$ v \f$ with the next available ID at the current time.
    /// @return Shared pointer to the created vertex
    [[nodiscard]] VertexPtr createVertex() noexcept;
    [[nodiscard]] VertexPtr createVertex(const std::vector<double> &coords) noexcept;


    /// Creates a vertex \f$ v \f$ with the given ID at the current time.
    /// @param id The unique identifier for the vertex
    /// @return Shared pointer to the created vertex
    [[nodiscard]] VertexPtr createVertex(std::uint64_t id) const noexcept;

    /// Creates a vertex \f$ v \f$ with the given ID and coordinates \f$ (t, x^1, \ldots, x^{n-1}) \f$.
    /// @param id The unique identifier for the vertex
    /// @param coords The spacetime coordinates of the vertex
    /// @return Shared pointer to the created vertex
    [[nodiscard]] VertexPtr createVertex(std::uint64_t id, const std::vector<double> &coords) const noexcept;

    /// Creates an edge \f$ e = (v_s, v_t) \f$ as a NULL edge: \f$ \ell^2 = 0 \f$,
    /// explicitly — no metric evaluation happens (#581; the doc previously
    /// claimed a metric-computed length). Callers that want a geometric length
    /// use the explicit-length overload or set it afterwards (``Edge::setLength``).
    /// @param src The source vertex \f$ v_s \f$
    /// @param tgt The target vertex \f$ v_t \f$
    /// @return Shared pointer to the created (null) edge
    [[nodiscard]] EdgePtr createEdge(const VertexPtr &src, const VertexPtr &tgt) const noexcept;

    /// Creates an edge \f$ e = (v_s, v_t) \f$ with an explicit complex LENGTH.
    /// A caller holding an \f$\ell^2\f$ passes ``std::sqrt(l2)`` and so chooses the
    /// branch explicitly (#639); this used to be a ``double`` funnel that could only
    /// express a real \f$\ell^2\f$.
    /// @param src The source vertex \f$ v_s \f$
    /// @param tgt The target vertex \f$ v_t \f$
    /// @param length The complex length \f$ \ell \f$ of the edge
    /// @return Shared pointer to the created edge
    [[nodiscard]] EdgePtr createEdge(const VertexPtr &src, const VertexPtr &tgt, std::complex<double> length) const noexcept;

    // ========================================
    // Complex Building Methods
    // ========================================

    /// Builds an n-dimensional (depending on your metric) triangulation/slice for t=0 with edge lengths equal to alpha
    /// matching the chosen topology. The default Topology is Toroid.
    ///
    /// Constructs an initial simplicial complex \f$ \mathcal{K}_0 \f$ by iteratively gluing \f$ n \f$ simplices.
    /// @param numSimplices The number of simplices to add to the initial complex
    void build(int numSimplices=3);

    /// Factory: build a pre-geometric simplicial complex from an explicit list
    /// of top \p cells (each a vertex-id tuple) — the cells-to-Spacetime
    /// builder the register/fill examples share. Creates a coordinate-free
    /// Lorentzian \f$ d \f$-dimensional CDT Spacetime, one vertex per distinct
    /// id, one top simplex per cell (auto-wiring the edges via
    /// :func:`createSimplex`), and sets the edge geometry by one of two
    /// explicit rules:
    ///
    ///   - **Uniform Hermitian pin** (when \p vertexTimes is absent): every
    ///     edge is pinned to squared length \p weight and phase \p phase. The
    ///     pre-geometric register/bulk surfaces use this.
    ///   - **Tracked metric** (when \p vertexTimes is present): each vertex
    ///     \f$ v \f$ is created carrying the single time coordinate
    ///     \f$ \text{vertexTimes}[v] \f$, so the tracked metric rule assigns
    ///     spacelike (equal-time) and timelike (differing-time) edges
    ///     automatically — the CDT-natural layered fill. \p weight and \p phase
    ///     are ignored.
    ///
    /// The time coordinate is always arity one: a vertex carries \f$ \{t\} \f$
    /// or no coordinate at all, never the length-2/3 vector that makes
    /// @ref Vertex::getTime throw (the coordinate-arity trap).
    ///
    /// @param dimensions The metric/signature dimension \f$ d \f$; cells should
    ///   be \f$ (d{+}1) \f$-vertex tuples to register as top simplices.
    /// @param cells Top cells as vertex-id tuples (order within a cell does not
    ///   matter; it is sorted).
    /// @param weight Uniform-pin squared length (ignored when \p vertexTimes is
    ///   given).
    /// @param phase Uniform-pin Hermitian phase (ignored when \p vertexTimes is
    ///   given).
    /// @param vertexTimes Optional per-vertex time, indexed by vertex id; its
    ///   presence selects the tracked-metric rule. Must be long enough to index
    ///   every vertex id appearing in \p cells.
    /// @return The freshly built Spacetime.
    [[nodiscard]] static std::shared_ptr<Spacetime> fromCells(
        int dimensions,
        const std::vector<std::vector<std::uint64_t>> &cells,
        double weight = 1.0,
        std::complex<double> phase = 0.0,
        const std::optional<std::vector<double>> &vertexTimes = std::nullopt);

    /// The dimension-generic staircase ("prism") triangulation of
    /// \f$ K \times [0, \text{layers}] \f$ from the top cells of a base
    /// complex \f$ K \f$. For each base cell \f$ (v_0 < \dots < v_{m-1}) \f$
    /// (an \f$ (m{-}1) \f$-simplex) and each layer, emits the \f$ m \f$ cells
    /// \f[
    ///   S_j = \{\mathrm{lo}[v_0], \dots, \mathrm{lo}[v_j]\} \cup
    ///         \{\mathrm{hi}[v_j], \dots, \mathrm{hi}[v_{m-1}]\},
    ///   \quad j = 0 \dots m-1,
    /// \f]
    /// where the lower copy in layer \f$ \ell \f$ is
    /// \f$ \mathrm{lo}[x] = \varphi^{\ell}(x) + s\,\ell \f$ and the upper copy
    /// is \f$ \mathrm{hi}[x] = \varphi^{\ell+1}(x) + s\,(\ell{+}1) \f$, with
    /// \f$ s \f$ the per-layer vertex stride (one past the largest base id).
    /// Adjacent prisms split shared walls by the same vertex-order rule, so the
    /// result is a consistent complex. This is the single source of the
    /// staircase the register fills carried as separate 3d and 4d copies; the
    /// rule is identical in every dimension (\f$ m = 3 \f$ gives tetrahedra
    /// over triangles, \f$ m = 4 \f$ gives 4-simplices over tetrahedra, …).
    ///
    /// @param cells Base top cells as vertex-id tuples.
    /// @param layers Number of product layers (\f$ \ge 1 \f$).
    /// @param twist Optional vertex permutation \f$ \varphi \f$ of the base,
    ///   applied cumulatively per layer (gluing the top end through a symmetry —
    ///   the mapping-torus-style twisted product). Identity when absent; keys
    ///   are base vertex ids and a missing id maps to itself.
    /// @return The prism's top cells as sorted vertex-id tuples, uniqued and
    ///   sorted.
    [[nodiscard]] static std::vector<std::vector<std::uint64_t>> prismCells(
        const std::vector<std::vector<std::uint64_t>> &cells,
        int layers = 1,
        const std::optional<std::unordered_map<std::uint64_t, std::uint64_t>>
            &twist = std::nullopt);

    /// The **symmetric apex stacking** of a triangulated \f$ d \f$-manifold into a
    /// cobordism \f$ (d{+}1) \f$-complex (any base dimension \f$ d \ge 2 \f$) --- a
    /// label-independent alternative to ::prismCells via **coface mirroring**. Each
    /// top \f$ d \f$-simplex \f$ t \f$ cones up to a single *cell-apex* \f$ f_t \f$
    /// (an up-cone \f$ t \cup \{f_t\} \f$) and down from \f$ f_t \f$ to the top copy
    /// (a down-cone --- the **point reflection** of the up-cone through \f$ f_t \f$).
    /// The gap over a \f$ (d{-}1) \f$-facet \f$ g \f$ shared by two cofaces (apexes
    /// \f$ f_1, f_2 \f$) is the join of the **canonical dual edge** \f$ f_1 f_2 \f$
    /// with the boundary of the worldprism \f$ g \times I \f$:
    /// \f$ [f_1,f_2] * \partial(g\times I) \f$. Its caps reproduce the up/down
    /// reflection; its sides mirror the connectivity across \f$ g \f$'s lower faces.
    /// In \f$ d=2 \f$ this is **exactly** the \#413 octahedron split on the dual edge
    /// (the worldprism sides are worldlines --- no diagonal); in \f$ d\ge 3 \f$ the
    /// side worldsheets take a globally-consistent (vertex-id-ordered) staircase
    /// diagonal, yielding a valid manifold on a tetrahedral \f$ S^3 \f$ base.
    ///
    /// The apex is a point reflection (a parity+time inversion), so the down-cone is
    /// the orientation-reverse of the up-cone: stacking \p nApexSlices reflect-and-cap
    /// layers gives an **alternating** \f$ (-1)^j \f$ per-slice chirality (a Dirac,
    /// non-chiral, twist --- both senses at once), never a single chiral screw.
    ///
    /// IDs: primal layer \f$ \ell \f$ (\f$ 0 \le \ell \le \texttt{nApexSlices} \f$)
    /// holds \f$ v + \ell\,\text{stride} \f$; apexes start at
    /// \f$ (\texttt{nApexSlices}{+}1)\,\text{stride} \f$ (one per top simplex per
    /// slice). \p nApexSlices \f$ = 1 \f$ reproduces the single-reflection \#413
    /// result bit-for-bit. A facet with a single incident top simplex (a hole
    /// boundary) is a tube wall, not gap-filled.
    /// @param baseCells the base manifold's top \f$ d \f$-simplices as vertex-id
    ///   tuples (uniform \f$ (d{+}1) \f$-vertex cells; \f$ d \f$ is inferred).
    /// @param nApexSlices the number of stacked apex (reflect-and-cap) layers
    ///   (\f$ \ge 1 \f$); 1 is the single \#413 reflection.
    /// @return the cobordism's \f$ (d{+}1) \f$-simplices as sorted vertex-id tuples,
    ///   uniqued.
    [[nodiscard]] static std::vector<std::vector<std::uint64_t>> symmetricStackCells(
        const std::vector<std::vector<std::uint64_t>> &baseCells,
        int nApexSlices = 1);

    // ========================================
    // Query Methods
    // ========================================

    /// @return Total number of top-dimensional (\f$ d \f$-) simplices, i.e.,
    ///   the four-volume \f$ N_4 = N_{41} + N_{32} \f$ in 4D CDT.
    [[nodiscard]] std::size_t getSimplexCount() const noexcept;

    /// @return Number of vertices \f$ N_0 \f$ in the triangulation.
    /// In the Regge action this appears as \f$ -(k_0 + 6\Delta)\, N_0 \f$.
    [[nodiscard]] std::size_t getVertexCount() const noexcept;

    /// @return Count of \f$(d,1) + (1,d)\f$-type simplices \f$ N_{41} \f$.
    /// These simplices have \f$ d \f$ vertices on one time slice and 1 on the adjacent slice.
    [[nodiscard]] std::size_t getN41() const noexcept;

    /// @return Count of \f$(d\!-\!1, 2) + (2, d\!-\!1)\f$-type simplices \f$ N_{32} \f$.
    /// These simplices have their vertices split \f$(d\!-\!1, 2)\f$ across adjacent slices.
    [[nodiscard]] std::size_t getN32() const noexcept;

    /// @return Const reference to the flat simplex vector \f$ \mathcal{K} \f$
    /// (all simplices of all dimensions registered in the complex).
    [[nodiscard]] const std::vector<SimplexPtr>& getSimplices() const noexcept;

    /// @return Const reference to the top-dimensional simplices only, in the
    /// same order used by @ref getDualAdjacency (i.e. element \a i is dual
    /// node \a i).  Lets callers attach per-node data to the dual COO without
    /// re-deriving the top-simplex indexing from Python.
    [[nodiscard]] const std::vector<SimplexPtr>& getTopSimplices() const noexcept;

    /// Deterministically seed the spacetime's internal RNG. The RNG drives
    /// the @ref getRandomVertex / @ref getRandomSimplex / @ref getRandomTopSimplex
    /// family used by Pachner moves' first-step sigma selection. Call this
    /// in tests that need byte-identical reproducibility across processes;
    /// otherwise the default seed comes from `std::random_device`.
    void setSeed(std::uint32_t s) noexcept { rng.seed(s); }

    /// Look up the live simplex with the given vertex tuple, if any.
    /// Returns ``nullptr`` if no simplex currently has exactly these
    /// vertices. The lookup is by fingerprint (commutative XOR-mix of
    /// vertex IDs), so order does not matter; duplicate vertices are
    /// folded.
    ///
    /// Intended for transactional-move rollback code that needs to
    /// resolve a simplex by its identifying verts at rollback time
    /// (the original ``SimplexPtr`` may be stale if another move ran
    /// in between).
    [[nodiscard]] SimplexPtr findSimplexByVerts(
        const VertexPtrs &vertices) const noexcept;

    /// Select a uniformly random vertex from the vertex list.
    /// Used by the (2d,2) delete move for blind-guessing vertex selection.
    /// @return A random vertex, or nullptr if none exist
    ///
    /// The no-argument form draws from the spacetime's own ``rng`` (seeded from
    /// ``std::random_device`` unless ``setSeed`` was called). The overload draws
    /// from a **caller-supplied** generator so a move with its own seeded engine
    /// (e.g. ``AddMove``) selects reproducibly from that seed — without this, a
    /// seeded move still made its random picks against the global ``rng`` and so
    /// was nondeterministic (issue #262).
    [[nodiscard]] VertexPtr getRandomVertex();
    [[nodiscard]] VertexPtr getRandomVertex(std::mt19937 &generator);

    /// Select a uniformly random simplex from the parallel access vector.
    /// @return A random simplex, or nullptr if the complex is empty
    [[nodiscard]] SimplexPtr getRandomSimplex();
    [[nodiscard]] SimplexPtr getRandomSimplex(std::mt19937 &generator);

    /// The vertex count of a top-dimensional simplex: \f$ d+1 \f$, where
    /// \f$ d \f$ is the metric signature's dimension. This is the **single
    /// source of truth** for "what counts as a top cell": ``registerSimplex``
    /// pushes a simplex onto ``topSimplicesVec`` exactly when its vertex count
    /// equals this, and ``getBoundary`` / ``getRandomTopSimplex`` read that set.
    /// A triangulation whose top cells are \f$ d' \f$-simplices is only seen as
    /// having top cells when the signature dimension matches it
    /// (\f$ d = d' \f$); see ``Topology::dimension``.
    [[nodiscard]] std::size_t getTopVertexCount() const noexcept;

    /// Select a uniformly random top-dimensional simplex.
    /// Used by the Metropolis algorithm to pick random move targets.
    /// @return A random d-simplex, or nullptr if none exist
    /// The overload draws from a caller-supplied generator (see
    /// ``getRandomVertex``; issue #262).
    [[nodiscard]] SimplexPtr getRandomTopSimplex();
    [[nodiscard]] SimplexPtr getRandomTopSimplex(std::mt19937 &generator);

    /// Select a uniformly random simplex with a specific causal orientation.
    /// Tries random sampling first (fast when many match), then falls back
    /// to a linear scan.
    /// @param ti Number of vertices on the initial time slice
    /// @param tf Number of vertices on the final time slice
    /// @return A matching simplex, or nullptr if none exist
    [[nodiscard]] SimplexPtr getRandomSimplexWithOrientation(uint8_t ti, uint8_t tf);

    /// Fully remove a \f$ d \f$-simplex from the complex: unregister it from the
    /// simplex set, remove it from each constituent vertex's simplex list, and
    /// clean up coface references in its facets. Used by CDT Pachner moves.
    /// @param simplex The simplex \f$ \sigma \f$ to remove
    void removeSimplex(const SimplexPtr &simplex);

    /// Monotone COMBINATORIAL revision of this complex: bumped by every simplex
    /// registration/unregistration and by standalone edge creation, never by an
    /// edge-length change. Purely topological invariants (Betti numbers) key
    /// their caches on it — an unchanged revision proves the complex's
    /// combinatorics did not change since the cache was filled.
    [[nodiscard]] std::uint64_t structuralRevision() const noexcept {
      return structuralRevision_;
    }
    /// The Betti numbers cached at the current structural revision, or nullptr
    /// when none were stored since the last combinatorial change. The
    /// COMPUTATION lives in the cobordism layer (ChainComplex::bettiNumbers);
    /// only the slot lives here, so the cache dies with the spacetime instead
    /// of dangling in a registry. Not synchronized: candidate spacetimes are
    /// thread-private clones, and the shared complex is scored serially.
    [[nodiscard]] const std::vector<int> *cachedBettiNumbers() const noexcept {
      return bettiCacheRevision_ == structuralRevision_ ? &bettiCache_ : nullptr;
    }
    /// Store Betti numbers computed against the CURRENT structural revision.
    void storeBettiNumbers(std::vector<int> numbers) const noexcept {
      bettiCache_ = std::move(numbers);
      bettiCacheRevision_ = structuralRevision_;
    }

    /// The Betti numbers of the sub-complex spanned by the vertex set that
    /// `vertexSetKey` names — the SUB-COMPLEX analogue of
    /// `cachedBettiNumbers`. `vertexSetKey` is the caller's hash of the
    /// region's vertex identifiers (see
    /// `MultiCobordism::blockVertexSetKey`).
    ///
    /// Consumers: the cobordism input-block residuals. Each block region is
    /// re-materialized as a FRESH spacetime on every objective evaluation, so
    /// its own `cachedBettiNumbers` slot is always empty and the Smith normal
    /// form recomputed every time (measured: 47.5% of the cycles in a live
    /// run). The parent outlives those temporaries and holds the answers for
    /// them.
    ///
    /// **When an entry stops being served.** Nothing marks entries dirty.
    /// Each entry records the `structuralRevision()` it was computed at and
    /// is returned only while that number still matches the parent's current
    /// one. `structuralRevision()` rises on every cell registration, cell
    /// removal, and edge creation, so one topology change retires every entry
    /// at once, whether or not it touched that particular region.
    ///
    /// **Why an entry cannot answer for the wrong sub-complex.** A
    /// sub-complex is determined by two things and no others: the parent's
    /// cells, and the vertex set (`subcomplexWithinVertexSet` takes the
    /// parent's top cells whose vertices all lie in the set). Serving an
    /// entry requires both to agree — the revision match fixes the parent's
    /// cells, and the key match fixes the vertex set — so a served answer was
    /// computed for a sub-complex identical to the one being asked about.
    /// Vertex identifiers are the right key because they ARE the region's
    /// definition; and although identifiers can be retired and reissued, any
    /// such change registers or removes cells and therefore moves the
    /// revision, retiring the entry before it could be mismatched.
    ///
    /// **Lifetime.** The store is a member, so it is created and destroyed
    /// with the spacetime and never outlives one; there is no registry and no
    /// clearing step between runs. A spacetime rebuilt from a snapshot is a
    /// new object that starts empty. A COPIED spacetime carries the entries
    /// and the revision they were stamped at together, and is combinatorially
    /// identical to its source at that moment, so they answer for it
    /// correctly.
    ///
    /// **Threading.** Not synchronized, on the same grounds as
    /// `cachedBettiNumbers`: the parallel candidate loop gives every thread
    /// its own rebuilt spacetime and reads and writes only that one, and the
    /// shared complex is scored serially.
    [[nodiscard]] const std::vector<int> *cachedSubcomplexBettiNumbers(
        std::uint64_t vertexSetKey) const noexcept {
      const auto entry = subBettiCache_.find(vertexSetKey);
      if (entry == subBettiCache_.end() ||
          entry->second.first != structuralRevision_)
        return nullptr;
      return &entry->second.second;
    }
    /// Store a sub-complex's Betti numbers against the CURRENT structural
    /// revision. An entry under the same key is overwritten; entries under
    /// other keys stay until a revision mismatch retires them, which costs
    /// only their memory — a complex carries a handful of block regions, so
    /// the store holds a handful of entries for as long as the spacetime
    /// lives.
    void storeSubcomplexBettiNumbers(std::uint64_t vertexSetKey,
                                     std::vector<int> numbers) const noexcept {
      subBettiCache_[vertexSetKey] = {structuralRevision_, std::move(numbers)};
    }

    /// Monotone METRIC revision: the structural revision plus the sum of every
    /// edge's length and phase revisions. Any combinatorial change, any
    /// ``setLength``, and any ``setPhase`` strictly increases it, so an
    /// unchanged value proves the operators built from this spacetime's
    /// geometry (the Hodge Laplacians and their spectra) are unchanged.
    /// O(#edges) walk per call — trivial beside the O(n³) work it gates.
    [[nodiscard]] std::uint64_t metricRevisionKey() const noexcept;

    /// The opaque spectral-cache slot stamped at the CURRENT metric revision,
    /// or nullptr when stale. The payload type is owned by the cobordism layer
    /// (HodgeLaplacian's shared spectrum map); this layer only provides the
    /// revision-stamped storage so the cache dies with the spacetime. The same
    /// non-synchronization contract as the Betti slot applies: candidate
    /// spacetimes are thread-private clones, and the shared complex is scored
    /// serially.
    [[nodiscard]] std::shared_ptr<void> cachedSpectralSlot() const noexcept {
      return spectralSlotRevision_ == metricRevisionKey() ? spectralSlot_
                                                          : nullptr;
    }
    /// Store the spectral payload computed against the CURRENT metric revision.
    void storeSpectralSlot(std::shared_ptr<void> payload) const noexcept {
      spectralSlot_ = std::move(payload);
      spectralSlotRevision_ = metricRevisionKey();
    }

    /// Balanced (causally undecided) edge-wiring mode (#690). OFF (default):
    /// new edges are wired on the causal axes — same-time ℓ = (√a, 0),
    /// cross-slice ℓ = (0, √(α·a)). ON: every auto-wired edge gets EQUAL real
    /// and imaginary components with the same per-class magnitude —
    /// ℓ = √(a/2)·(1+i) resp. √(α·a/2)·(1+i) — so ℓ² is purely imaginary
    /// (Re ℓ² = 0: born exactly on the null locus) and stage-2 relaxation
    /// must choose each edge's causal character. Creation-time convention
    /// only; existing edges are never rewritten by toggling this.
    void setBalancedEdgeWiring(bool balanced) noexcept {
      balancedEdgeWiring_ = balanced;
    }
    [[nodiscard]] bool balancedEdgeWiring() const noexcept {
      return balancedEdgeWiring_;
    }
    /// The balanced length of a given squared magnitude m: ℓ = √(m/2)·(1+i),
    /// so |ℓ|² = m and ℓ² = i·m.
    /// `timelikeBranch` selects the OTHER square root: `c(1 - i)` instead of
    /// `c(1 + i)`, i.e. `l^2 = -i|m|` instead of `+i|m|`. Both sit on the
    /// balanced convention (`Re l^2 = 0`, causally undecided at birth) and both
    /// carry the same magnitude, so the branch is what keeps a timelike
    /// disposition distinguishable from a spacelike one under balanced wiring
    /// — without it the two coincide and the disposition is silently erased
    /// (#741). `squaredMagnitude` is a MAGNITUDE: pass |m|, since `std::sqrt`
    /// of a negative double is NaN.
    [[nodiscard]] static std::complex<double> balancedLength(
        double squaredMagnitude, bool timelikeBranch = false) noexcept {
      const double c = std::sqrt(std::abs(squaredMagnitude) * 0.5);
      return {c, timelikeBranch ? -c : c};
    }
    /// Absorb a departing edge's revision counters into the structural
    /// revision, called immediately BEFORE removing it from the edge list.
    /// metricRevisionKey() sums live edges' counters, so an uncompensated
    /// removal would DECREASE the key and could later collide with a stamp
    /// made for an older geometry — a false spectral-cache hit (#692). The
    /// +1 covers the removal event itself; the mirror of the absorption
    /// Simplex::removeEdge already performs for its own geometry cache.
    void absorbRemovedEdgeRevisions(const EdgePtr &edge) noexcept {
      if (edge)
        structuralRevision_ += edge->lengthRevision() + edge->phaseRevision() + 1;
    }

    /// The auto-wiring rule for a new edge, honoring the wiring mode:
    /// `crossSlice` = the endpoints sit on different time slices (the
    /// timelike class, scale α·a); same-slice edges use scale a.
    [[nodiscard]] std::complex<double> autoWiredLength(bool crossSlice) const noexcept;

    /// Unregister every *orphaned* sub-simplex: a non-top simplex (size < d+1)
    /// that is no longer a face of any current top cell. Lazy facet/hinge
    /// materialisation (``Simplex::getFacets``) registers sub-simplices that a
    /// later Pachner move can strand once it removes the top cells above them.
    /// Such orphans are not part of the simplicial complex — they only persist
    /// as stale cache entries — yet they linger in ``getSimplices()``. The Regge
    /// action already excludes them (``Simplex::hasTopCoface`` filtering in
    /// ``ReggeSolver``), so this is not needed for action correctness; it is for
    /// callers (e.g. a move-driven optimiser, or a round-trip invariance check)
    /// that want the registered simplex set to stay exactly the closure of the
    /// top cells — bit-identical before and after an apply∘rollback. Edges and
    /// vertices are left untouched (the move classes own those). Returns the
    /// number of simplices pruned.
    std::size_t pruneOrphanedSimplices();

    /// Scoped variant: unregister the orphaned proper faces of **one** cell —
    /// every registered sub-simplex spanned by a proper subset of
    /// \p cellVertexIds that no current top cell covers. This is the same
    /// operation as the full sweep restricted to a single (typically
    /// just-removed) top cell's face lattice, so a move class can keep the
    /// "registered simplices = closure of the top cells" invariant at
    /// \f$ O(2^d) \f$ per move instead of an \f$ O(N) \f$ pass. Faces still
    /// covered by a surviving top cell are kept. Pruning is essential before
    /// :func:`removeEdge` on an orphaned edge: a registered face that outlives
    /// its edge keeps an empty edge set and silently reads
    /// \f$ \ell^2 = 0 \f$ in every Gram-matrix computation thereafter (#587).
    /// Returns the number of simplices pruned.
    /// @param cellVertexIds The vertex ids spanning the cell whose face
    ///   lattice is checked (need not itself be registered).
    std::size_t pruneOrphanedSimplices(
        const std::vector<std::uint64_t> &cellVertexIds);

    /// Fully remove an edge from the complex: drop it from its endpoints'
    /// in/out edge lists and from the EdgeList. The caller is responsible
    /// for first removing any simplices that contain the edge.
    /// @param edge The edge \f$ e \f$ to remove
    void removeEdge(const EdgePtr &edge);

    /// Fully remove a vertex from the complex: remove every edge incident
    /// to it (via :func:`removeEdge`) and drop the vertex from the
    /// VertexList. The caller is responsible for first removing any
    /// simplices that contain the vertex.
    /// @param vertex The vertex \f$ v \f$ to remove
    void removeVertex(const VertexPtr &vertex);

    /// @return The type of quantum gravity formulation (CDT, Regge, etc.)
    [[nodiscard]] SpacetimeType getSpacetimeType() const noexcept;

    /// @return The current time coordinate \f$ t \f$ for time-slicing
    [[nodiscard]] double getCurrentTime() const noexcept;

    /// @return The edge list \f$ E \f$ containing all edges in the complex
    [[nodiscard]] const std::shared_ptr<EdgeList> &getEdgeList() const noexcept;

    /// @return The metric tensor \f$ g_{\mu\nu} \f$ defining the geometry
    [[nodiscard]] const std::shared_ptr<Metric> &getMetric() const noexcept;

    /// @return The vertex list \f$ V \f$ containing all vertices in the complex
    [[nodiscard]] const std::shared_ptr<VertexList> &getVertexList() const noexcept;

    /// Reserves and returns the next unique vertex ID without
    /// allocating a Vertex object. Use this when constructing a
    /// Vertex subclass (e.g. ``quantum::QuantumVertex``) via the
    /// polymorphic ``VertexList::addAs<T>(id, id, ...)`` path —
    /// the typed-add API needs an id up front, and bypassing
    /// ``createVertex`` avoids allocating a temporary base Vertex
    /// only to immediately discard it.
    ///
    /// The returned id is guaranteed unique for the lifetime of
    /// this Spacetime; callers are responsible for actually
    /// inserting a vertex with this id (otherwise the id is
    /// simply unused).
    [[nodiscard]] std::uint64_t reserveVertexId() noexcept {
      return nextFreeVertexId();
    }

    /// The boundary surface of the complex: the codimension-one faces — one
    /// dimension below the top simplices — that belong to exactly one top
    /// simplex, returned as sorted vertex-id tuples. An interior face is shared
    /// by two top simplices and is excluded; a boundary face belongs to just
    /// one.
    ///
    /// This is the canonical, single-source boundary derivation. It is computed
    /// purely by facet-counting from the top-dimensional simplices (incidence
    /// \f$ == 1 \f$), so it is **side-effect-free** (``const``) and robust to
    /// lazily-materialized facets: "top" is the maximal vertex count actually
    /// present in the complex, and the codimension-one faces are enumerated
    /// combinatorially from the vertex sets rather than from materialized
    /// ``Simplex`` facet objects. A closed manifold returns an empty list.
    ///
    /// Contrast :func:`getExternalSimplices`, which returns the boundary
    /// *top cells* (whole d-simplices touching the boundary) and materializes
    /// facets as a side-effect.
    [[nodiscard]] std::vector<std::vector<std::uint64_t>> getBoundary() const;

    /// Force lazy facet materialization to a fixpoint. Facets are created on
    /// first access by ``Simplex::getFacets()``, which registers each new facet
    /// back into the complex; a single index pass over the (growing) simplex
    /// vector therefore materializes every face of every dimension down to the
    /// vertices, wiring up the coface incidence as it goes. After this call the
    /// complex's facet/coface structure is complete.
    ///
    /// This exposes — as an explicit, separately callable operation — the
    /// side-effect that :func:`getExternalSimplices` performs internally; call
    /// it directly when you want the materialization without the boundary scan.
    void materializeFacets() noexcept;

    /// Scoped variant: materialize the face lattice of **one** simplex —
    /// recursive ``Simplex::getFacets()`` from \p root down to its vertices,
    /// wiring the facet/coface incidence of every face on the way. Faces that
    /// already exist are reused (gaining only the missing coface link); the
    /// rest are created and registered. ``Simplex::dualVolume`` walks a hinge
    /// **up** through exactly these coface links, so a move class restoring a
    /// removed cell must restore this lattice too (#587) — this does that at
    /// \f$ O(2^d) \f$ instead of the full-complex fixpoint pass above.
    /// @param root The simplex whose face lattice to materialize.
    void materializeFacets(const SimplexPtr &root) noexcept;

    /// @return Simplices around the boundary of the simplicial complex. These simplices have at
    /// least one external face. They will tend to be in order of orientation (e.g. (4, 1) and (3, 2) for 4D CDT). Note
    /// that this method does not return 2-simplices as you might expect, but 5-simplices since those are the standard
    /// building blocks. You can get the 2-simplices by calling `getFacets()` on the 5-simplices and their facets until
    /// \f$ k=2 \f$.
    ///
    /// Materializes facets to a fixpoint (via :func:`materializeFacets`) as a
    /// side-effect, since the coface counts that flag a boundary facet are only
    /// complete once every facet exists. For the side-effect alone, call
    /// :func:`materializeFacets`; for the codimension-one boundary *faces*
    /// (rather than the top cells touching them), call :func:`getBoundary`.
    [[nodiscard]] SimplexSet getExternalSimplices() noexcept;

    /// Retrieves all simplices with a specific causal orientation.
    /// @param orientation The orientation tuple (timelike_initial, timelike_final)
    /// @return Set of simplices \f$ \{\sigma^{(t_i, t_f)}\} \f$ with the given orientation
    /// @note This method is for testing only and has poor runtime performance.
    std::vector<SimplexPtr> getSimplicesWithOrientation(std::tuple<uint8_t, uint8_t> orientation) const;

    /// Returns the foliation of the spacetime. Either NONE or PREFERRED. With PREFERRED foliation; each spatial slice
    /// lies between a time slice and vis versa. Otherwise they can be any-which-a-way.
    /// @return Foliation::PREFERRED or Foliation::NONE.
    [[nodiscard]] Foliation getFoliation() const noexcept;

    // ========================================
    // Time-Slice & Spatial-Subgraph Queries
    // ========================================

    /// Sorted list of integer time values present in the triangulation.
    [[nodiscard]] std::vector<int> getTimeSlices() const;

    /// All vertices at a given integer time slice.
    [[nodiscard]] VertexPtrs getVerticesAtTime(int t) const;

    /// Spatial subgraph at time \a t: vertices and spacelike edges
    /// (positive squared length) connecting them within the slice.
    [[nodiscard]] std::pair<VertexPtrs, Edges> getSpatialSubgraph(int t) const;

    /// BFS shortest-path distances from \a center through spacelike edges.
    /// If \a maxDepth >= 0, stops exploring beyond that depth.
    /// Returns a map from vertex ID to distance.
    [[nodiscard]] std::unordered_map<std::uint64_t, int>
    bfsDistances(VertexPtr center, int maxDepth = -1) const;

    /// Returns the dual graph of the top-dimensional triangulation as COO-format
    /// adjacency data.  Two top simplices are adjacent if they share a (d-1)-face.
    ///
    /// @return (rows, cols, N) where rows[k],cols[k] are adjacent simplex indices
    ///   (0-based into the top-simplex array) and N is the number of top simplices.
    [[nodiscard]] std::tuple<std::vector<std::uint32_t>,
                             std::vector<std::uint32_t>,
                             std::uint32_t>
    getDualAdjacency() const;

    /// Convenience: build a :class:`SparseGraph` of the dual
    /// triangulation in one call.  Equivalent to
    /// ``SparseGraph::fromCOO(*getDualAdjacency())`` but avoids the
    /// intermediate Python-side conversion.
    [[nodiscard]] ::tessera::observables::SparseGraph getDualGraph() const;

    /// Spectral dimension D_S(σ) on the weighted 1-skeleton of top
    /// simplices that pass ``filter``. Walks ``getSimplices()`` keeping
    /// simplices with ``size() == topK + 1`` for which
    /// ``filter.accept(s)`` is true, unions their edges (uniqued by
    /// endpoint pair), weights them
    /// ``w_uv = I_max · exp(-sqrt(|squaredLength_uv|))``, builds the
    /// unnormalised weighted Laplacian ``L = D - W``, and returns
    /// ``SpectralGraph::spectralDimension`` of the heat-kernel return
    /// probability.
    ///
    /// Sits next to :func:`modularityOnSkeleton`: Spacetime exposes
    /// graph-based observables as single methods that compose the
    /// inherited heat-kernel pipeline with a filtered top-simplex
    /// projection.
    ///
    /// ``skeletonDim`` reserves API space for higher-k skeletons; only
    /// ``skeletonDim == 1`` is currently supported.
    [[nodiscard]] std::vector<double>
    getSpectralDimensionOnSkeleton(
        std::vector<double> const& sigmas,
        int krylovDim,
        SimplexFilter const& filter,
        int topK = 4,
        int skeletonDim = 1) const;

    /// The sub-complex carried by a boundary block: a freshly-built `Spacetime` of
    /// exactly the top cells of `spacetime` all of whose vertices lie in `vertexSet`
    /// (the block's region). Returns `nullptr` when the region contains no full cell.
    /// This is where a block's vertex-set becomes an actual complex — the block itself
    /// only stores the vertex-set and target, never the cells.
    [[nodiscard]] std::shared_ptr<Spacetime> subcomplexWithinVertexSet(
      const std::set<std::uint64_t> &vertexSet) const;

    /// Newman-Girvan modularity Q on the vertex/edge 1-skeleton, with
    /// implicit labels ``label(v) = v.id() % M``.
    ///
    /// ``Q = sum_c [L_c / m - (D_c / 2m)^2]`` where ``L_c`` is the
    /// number of edges within community c, ``D_c`` the sum of degrees
    /// of nodes in c, and ``m = |E|``.
    ///
    /// Returns 0 if M < 2, the graph has no edges, or the spacetime
    /// has no vertices.  See ``examples/modularity.py:modularity``
    /// for the reference Python implementation.
    [[nodiscard]] double modularityOnSkeleton(int M) const;

    /// Computes the connected components of the vertex graph.
    ///
    /// Uses depth-first search to identify connected components in the graph
    /// \f$ G = (V, E) \f$ where vertices are connected by edges.
    ///
    /// @return Vector of connected components, each containing a list of vertices
    [[nodiscard]] std::vector<VertexPtrs> getConnectedComponents() const;

    /// Retrieves a simplex from the complex by pointer lookup.
    /// @param simplex The simplex to find
    /// @return The canonical simplex from the complex, or nullptr if not found
    [[nodiscard]] SimplexPtr getSimplex(SimplexPtr simplex) const;

    /// Retrieves a simplex from the complex by fingerprint.
    /// @param fingerprint The fingerprint hash \f$ h(\sigma) \f$ of the simplex
    /// @return The simplex with the given fingerprint, or nullptr if not found
    [[nodiscard]] SimplexPtr getSimplex(std::uint64_t fingerprint) const;

    // ========================================
    // Manipulation & Helper Methods
    // ========================================

    /// Increments the current time coordinate.
    /// @return The new current time \f$ t \leftarrow t + 1 \f$
    double incrementTime() noexcept;

    /// Swap the labels (IDs) of two vertices, updating all affected data structures.
    ///
    /// Implements the vertex relabeling step from Brunekreef Sec. 2.2.1/2.3.1:
    /// after inserting a new vertex, swap its label with a randomly chosen vertex
    /// to ensure uniform sampling over labelled triangulations.
    ///
    /// Updates: VertexList keys, Simplex fingerprints and vertex-ID maps,
    /// and re-registers affected simplices in the hash tables.
    ///
    /// @param v1 First vertex
    /// @param v2 Second vertex (may be the same as v1, in which case no-op)
    void swapVertexLabels(VertexPtr v1, VertexPtr v2);

    /// Removes a vertex if it has no incident edges.
    ///
    /// Checks if \f$ \deg(v) = 0 \f$ and removes \f$ v \f$ from the vertex list if so.
    ///
    /// @param vertex The vertex \f$ v \f$ to potentially remove
    /// @return true if the vertex was removed, false if it has incident edges
    [[nodiscard]] bool removeIfIsolated(const VertexPtr &vertex) noexcept;

    /// Adds an observable to track during evolution.
    ///
    /// Observables are measured after each update step to monitor the system's properties.
    ///
    /// @param observable The observable \f$ \mathcal{O} \f$ to track
    void addObservable(const std::shared_ptr<Observable> &observable);

    // ========================================
    // Internal Management
    // ========================================

    /// Registers a simplex in the spacetime's internal data structures.
    ///
    /// Adds the simplex to:
    /// - The main simplex set \f$ K \f$
    /// - The appropriate orientation-indexed sets (external or internal)
    ///
    /// @param simplex The simplex \f$ \sigma \f$ to register
    /// @param internal true if the simplex is fully internal (all faces glued), false otherwise
    /// @return The registered simplex
    SimplexPtr registerSimplex(const SimplexPtr &simplex, bool internal);


    void reserve(int nSimplices);

    /// Make the slots freed since the last call available for reuse.
    ///
    /// Call it where no Pachner move is in flight -- the end of a sweep. A move
    /// captures ``SimplexPtr`` in ``propose()`` and reads them in ``apply()``,
    /// so a slot freed during the move must not be handed back out until the
    /// move is over. Returns how many slots were released.
    std::size_t reclaimSimplexSlots() noexcept;

    /// Storage slots currently held for reuse, and slots waiting for the next
    /// ``reclaimSimplexSlots``. Reported so a caller can see reuse working
    /// rather than infer it from resident memory.
    [[nodiscard]] std::size_t freeSimplexSlotCount() const noexcept {
      return freeSimplexSlots_.size();
    }
    [[nodiscard]] std::size_t pendingSimplexSlotCount() const noexcept {
      return pendingSimplexSlots_.size();
    }
    /// Slots ever allocated, live or free -- the size of the storage deque.
    [[nodiscard]] std::size_t simplexStorageSize() const noexcept {
      return simplexStorage_.size();
    }

    /// Alpha is the coefficient that determines the ratio of timelike edge lengths to space like edge lengths. That
    /// relationship is
    ///
    /// \f[
    ///  l_s = +a
    /// \f]
    /// and
    /// \f[
    ///  l_t = - \alpha a
    /// \f]
    ///
    /// @return The coefficient representing the number of times the timelike length is compared to the spatial length.
    [[nodiscard]] double getAlpha() const noexcept;

    /// `a` is the coefficient that sets the fixed edge length for spacelike edges according to
    ///
    /// \f[
    ///  l_s = +a
    /// \f]
    /// for spacelike edges and
    /// \f[
    ///  l_t = - \alpha a
    /// \f]
    ///
    /// for timelike edges.
    ///
    /// @return The constant spacelike edge length.
    [[nodiscard]] double getA() const noexcept;

    [[nodiscard]] int getDimensions() const noexcept;

    /// Unregisters a simplex from the spacetime's internal data structures.
    ///
    /// Removes the simplex from all indexed sets. This should be called before
    /// modifying a simplex's fingerprint to avoid hash table corruption.
    ///
    /// @param simplex The simplex \f$ \sigma \f$ to unregister
    void unregisterSimplex(const SimplexPtr &simplex);

  private:
    // The boundary \f$ (d{-}1) \f$-faces of the worldprism \f$ g\times I \f$ used by
    // ::symmetricStackCells: the staircase triangulation of the prism over the
    // \f$ (d{-}1) \f$-facet \p facet (sorted, \f$ d \f$ vertices) between the
    // \p loOffset (bottom) and \p hiOffset (top) primal layers, keeping the faces
    // that lie on \f$ \partial(g\times I) \f$ (the caps \f$ g, g_\text{top} \f$ plus
    // the side worldsheets). Each returned face has \f$ d \f$ vertices; joined with
    // the dual edge it forms a \f$ (d{+}1) \f$-cell. The diagonal is the global
    // vertex-id staircase, so a worldsheet shared by several facets is split
    // consistently (a valid complex). In \f$ d=2 \f$ the sides are worldlines, so
    // the result is exactly the \#413 octahedron's boundary edges.
    [[nodiscard]] static std::vector<std::vector<std::uint64_t>> worldprismBoundaryFaces(
        const std::vector<std::uint64_t> &facet, std::uint64_t loOffset,
        std::uint64_t hiOffset);

    // The next vertex id not already in use: advances vertexIdCounter past any
    // explicitly-assigned ids so a no-arg/reserved id never aliases an existing
    // vertex (VertexList::add returns the existing vertex on a duplicate id;
    // coning that alias makes a self-edge — #267).
    [[nodiscard]] std::uint64_t nextFreeVertexId() noexcept;

    std::shared_ptr<EdgeList> edgeList = std::make_shared<EdgeList>();
    std::shared_ptr<VertexList> vertexList = std::make_shared<VertexList>();

    IdType vertexIdCounter = 0;
    SpacetimeType spacetimeType;
    double alpha = 1.;
    double a = 1.;
    Foliation foliation = Foliation::PREFERRED;
    std::shared_ptr<Metric> metric;
    std::shared_ptr<Topology> topology;
    std::uint64_t currentTime = 0;

    /// Storage for every Simplex this Spacetime has ever owned, live or
    /// logically removed.  std::deque is required: it gives stable element
    /// addresses across push_back, so raw Simplex* cached anywhere (vertex
    /// simplex lists, simplex facets/cofaces, edge simplex indices, Pachner-
    /// move snapshots, etc.) remain valid for the Spacetime's lifetime.
    ///
    /// Slots ARE recycled, one sweep behind. ``unregisterSimplex`` marks a slot
    /// stale via vecIdx_ == UINT32_MAX, clears the Simplex's heap-allocated
    /// children, and puts the slot on ``pendingSimplexSlots_``;
    /// ``reclaimSimplexSlots`` moves that list to ``freeSimplexSlots_``, from
    /// which ``createSimplexTracked`` allocates before growing the deque.
    ///
    /// Holding the slot back until the caller says it is safe is what makes the
    /// reuse sound. ``removeSimplex`` already clears every back-reference before
    /// unregistering -- the vertices' simplex lists, the facets' coface links
    /// and each edge's simplex index -- so no live structure points at a removed
    /// shell. What remains is a pointer captured across a mutation, and the
    /// Pachner moves do capture: ``RemoveMove::incident_`` and the
    /// ``oldSimplices_`` of the flip and shift moves are read in ``apply()``
    /// after being taken in ``propose()``. Deferring reuse past the end of the
    /// sweep means nothing a move is holding can be handed out underneath it.
    ///
    /// Reuse assigns over the slot, so a pointer kept from before still
    /// addresses a valid Simplex -- a different one. ``Simplex::generation()``
    /// counts how many times the slot has been handed out, and ``SimplexRef``
    /// pairs a pointer with the generation it was taken at so a holder can tell.
    std::deque<Simplex> simplexStorage_{};
    /// Slots freed this sweep, not yet safe to hand out.
    std::vector<std::uint32_t> pendingSimplexSlots_{};
    /// Slots freed in an earlier sweep and available now.
    std::vector<std::uint32_t> freeSimplexSlots_{};
    std::vector<SimplexPtr> simplicesVec{}; // flat array of live simplex pointers (swap-and-pop)
    FlatHashMap<SimplexPtr> simplexIndex_{}; // fingerprint → simplex ptr (dedup only)
    std::vector<SimplexPtr> topSimplicesVec{}; // top-dimensional simplices only
    std::size_t n41Count = 0; // (4,1) + (1,4) simplices
    std::size_t n32Count = 0; // (3,2) + (2,3) simplices
    std::mt19937 rng{std::random_device{}()};

    void updateOrientationCounters(const SimplexPtr &simplex, int delta);

    /// See structuralRevision(). Mutable because ``createEdge`` is const yet
    /// combinatorially mutates the complex (pre-existing API shape).
    mutable std::uint64_t structuralRevision_{1};
    /// Betti cache slot; valid iff bettiCacheRevision_ == structuralRevision_.
    mutable std::vector<int> bettiCache_{};
    mutable std::uint64_t bettiCacheRevision_{0};
    /// Sub-complex Betti slots: vertex-set key -> (revision stamped, numbers).
    mutable std::map<std::uint64_t, std::pair<std::uint64_t, std::vector<int>>>
        subBettiCache_{};
    /// Spectral-cache slot; valid iff spectralSlotRevision_ == metricRevisionKey().
    mutable std::shared_ptr<void> spectralSlot_{};
    mutable std::uint64_t spectralSlotRevision_{0};
    /// See setBalancedEdgeWiring().
    bool balancedEdgeWiring_{false};

    std::vector<std::shared_ptr<Observable> > observables{};
};
} // namespace tessera::spacetime

#endif //TESSERA_SPACETIME_H
