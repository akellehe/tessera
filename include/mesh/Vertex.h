// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

//
// Created by Andrew Kelleher on 10/19/25.
//

#ifndef TESSERA_TESSERA_SRC_VERTEX_H_
#define TESSERA_TESSERA_SRC_VERTEX_H_

#include <vector>
#include <memory>
#include <unordered_set>
#include "mesh/ForwardDeclarations.h"
#include "mesh/Fingerprint.h"

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

/// \brief A vertex in a causal set (causet) spacetime discretization.
///
/// Vertices are the nodes of a directed graph:
/// - edges carry a direction (source to target) representing a causal relation
/// - coordinates are optional and of arbitrary dimension (getTime() constrains which
///   dimensions are supported)
/// - each vertex keeps both incident edge lists, incoming and outgoing
/// - simplices register themselves on their constituent vertices, so topology queries
///   stay local to a vertex
///
/// Reference: Sorkin, arXiv:gr-qc/0309009
class Vertex {
    public:
        // ========================================
        // Constructors
        // ========================================

        /// Default constructor creating a vertex with ID 0
        Vertex() noexcept;

        /// Virtual destructor — Vertex is a polymorphic base
        /// (QuantumVertex lives in tessera_quantum) and VertexList
        /// owns instances through std::unique_ptr<Vertex>, so the
        /// destructor must be virtual for safe polymorphic deletion.
        virtual ~Vertex() = default;

        ///
        /// \brief Construct vertex with ID and spatial coordinates
        /// \param id_ Unique identifier for this vertex
        /// \param coords Position in spacetime (arbitrary dimension)
        ///
        /// Creates a vertex with specified coordinates. The dimensionality is determined by coords.size():
        /// - 1D: Time is \f$ |x_0| \f$
        /// - 4D+: Time is \f$ \sqrt{\sum_{i=0}^{N-1} x_i^2} \f$
        /// - 2D, 3D: Invalid - getTime() will throw std::out_of_range
        ///
        Vertex(const std::uint64_t id_, const std::vector<double> &coords) noexcept;

        ///
        /// \brief Construct coordinate-independent vertex with ID only
        /// \param id_ Unique identifier for this vertex
        ///
        /// Creates a vertex without coordinate information. Calling getCoordinates() on such
        /// a vertex will throw std::runtime_error.
        ///
        explicit Vertex(const std::uint64_t id_) noexcept;

        // ========================================
        // Core Properties
        // ========================================

        ///
        /// \brief Get the unique identifier of this vertex
        /// \return The vertex ID
        ///
        std::uint64_t getId() const noexcept;

        /// Set the vertex id. Used by Spacetime::swapVertexLabels for vertex
        /// relabeling. The caller updates every containing data structure
        /// (VertexList, Simplex fingerprints, and so on) afterwards.
        void setId(std::uint64_t newId) noexcept { id = newId; }

        /// \brief The temporal coordinate, by coordinate dimensionality:
        ///
        /// - 0 coordinates (empty): returns 0
        /// - 1 coordinate: \f$ t = |x_0| \f$
        /// - 4 or more: \f$ t = \sqrt{\sum_{i=0}^{N-1} x_i^2} \f$ (Euclidean norm)
        /// - 2 or 3: throws std::out_of_range
        ///
        /// Above four dimensions the convention takes the Euclidean magnitude across all
        /// temporal dimensions and leaves the spatial ones to the embedding geometry.
        ///
        /// \return The time coordinate
        /// \throws std::out_of_range if the coordinate vector has length 2 or 3
        ///
        [[nodiscard]] double getTime() const;

        void setTime(double time) noexcept;

        ///
        /// \brief Get the coordinate vector for this vertex
        ///
        /// \return Vector of coordinate values
        /// \throws std::runtime_error if the vertex is coordinate-independent (empty
        ///   coordinates)
        ///
        /// Vertices need not carry coordinates; some algorithms work purely with the
        /// combinatorial structure.
        ///
        const std::vector<double> &getCoordinates() const;

        ///
        /// \brief Set new coordinates for this vertex
        /// \param coords New coordinate vector
        ///
        /// Does not update cached values in edges or simplices, so edge lengths derived
        /// from coordinates go stale.
        ///
        void setCoordinates(const std::vector<double> &coords) noexcept;

        ///
        /// \brief Get the total degree (number of incident edges)
        /// \return Sum of in-degree and out-degree
        ///
        /// For a directed graph, this returns \f$ \deg(v) = \deg^-(v) + \deg^+(v) \f$
        ///
        std::size_t degree() const noexcept;

        // ========================================
        // Edge Management
        // ========================================

        ///
        /// \brief Find a specific edge incident to this vertex
        /// \param edge Edge to search for (compared by ID)
        /// \return Shared pointer to the edge if found, nullptr otherwise
        ///
        /// Searches both inEdges and outEdges, so the caller need not know the
        /// direction.
        ///
        EdgePtr getEdge(const EdgePtr &edge) const;

        ///
        /// \brief Get all incident edges (both incoming and outgoing)
        /// \return Set containing all edges where this vertex is source or target
        ///
        /// The returned vector is a copy. Complexity: O(|inEdges| + |outEdges|)
        ///
        Edges getEdges() const noexcept;

        ///
        /// \brief Get all edges targeting this vertex
        /// \return Set of incoming edges \f$ \{e \mid e.target = v\} \f$
        ///
        const Edges &getInEdges() const noexcept;

        ///
        /// \brief Get all edges originating from this vertex
        /// \return Set of outgoing edges \f$ \{e \mid e.source = v\} \f$
        ///
        const Edges &getOutEdges() const noexcept;

        ///
        /// \brief Add an incoming edge to this vertex
        /// \param edge Edge where this vertex is the target
        ///
        /// Does not verify that edge->getTarget() == this; the caller keeps the two
        /// consistent.
        ///
        void addInEdge(const EdgePtr &edge) noexcept;

        ///
        /// \brief Add an outgoing edge from this vertex
        /// \param edge Edge where this vertex is the source
        ///
        /// Does not verify that edge->getSource() == this; the caller keeps the two
        /// consistent.
        ///
        void addOutEdge(const EdgePtr &edge) noexcept;

        ///
        /// \brief Remove an incoming edge and update all affected simplices
        /// \param edge The edge to remove from inEdges
        ///
        /// Removes the edge from every simplex containing it (via Simplex::removeEdge)
        /// and then from this vertex's inEdges.
        ///
        /// With TESSERA_ASSERTIONS defined, aborts if the edge is null or is not in
        /// inEdges.
        ///
        void removeInEdge(const EdgePtr &edge) noexcept;

        ///
        /// \brief Remove an outgoing edge and update all affected simplices
        /// \param edge The edge to remove from outEdges
        ///
        /// Symmetric to removeInEdge(), on outEdges.
        ///
        /// With TESSERA_ASSERTIONS defined, aborts if the edge is null or is not in
        /// outEdges.
        ///
        void removeOutEdge(const EdgePtr &edge) noexcept;

        // ========================================
        // Simplex Management
        // ========================================

        ///
        /// \brief Get all simplices that contain this vertex
        /// \return Set of simplices where this vertex is a constituent
        ///
        /// The inverse of the simplex-to-vertices relation.
        ///
        const Simplices &getSimplices() const noexcept;

        ///
        /// \brief Register a simplex as containing this vertex
        /// \param simplex The simplex to add
        /// \return true if the simplex was newly added, false if already present
        ///
        /// The Simplex-to-Vertex relation is bidirectional: a simplex created with a set
        /// of vertices calls addSimplex() on each of them.
        ///
        /// With TESSERA_ASSERTIONS defined, this checks for a null simplex or null
        /// ``this``, runs checkDuplicates() before and after insertion, and aborts on a
        /// violation.
        ///
        bool addSimplex(const SimplexPtr &simplex);

        ///
        /// \brief Unregister a simplex from this vertex
        /// \param simplex The simplex to remove
        /// \return true if simplex was removed, false if not found
        ///
        /// Called during simplex destruction and vertex replacement.
        ///
        bool removeSimplex(const SimplexPtr &simplex);

        ///
        /// \brief Debug utility to detect duplicate simplices
        /// \param msg Error message to log/throw if duplicates found
        ///
        /// Only checks anything when TESSERA_ASSERTIONS is defined: scans the registered
        /// simplices for duplicate fingerprints, logs at CRITICAL_LEVEL and throws
        /// std::runtime_error if any exist.
        ///
        void checkDuplicates(const std::string &msg) const;

        // ========================================
        // Vertex Operations (Edge Migration)
        // ========================================

        ///
        /// \brief Move all edges (both in and out) to another vertex
        /// \param vertex Target vertex to receive edges
        /// \param spacetime The spacetime context (must be non-null)
        /// \return Pair of (old edges removed, new edges created)
        ///
        /// For each edge incident to this vertex: remove it from the endpoint edge lists
        /// and from the spacetime's edge registry, create the edge with ``vertex``
        /// substituted for ``this``, and insert that into the spacetime. Edge properties
        /// such as the length are preserved.
        ///
        /// With TESSERA_ASSERTIONS defined, throws std::runtime_error if spacetime is
        /// null.
        ///
        /// \see moveInEdgesTo(), moveOutEdgesTo()
        ///
        std::pair<EdgePtrSet, EdgePtrSet> moveEdgesTo(const VertexPtr &vertex, Spacetime *spacetime);

        ///
        /// \brief Move only incoming edges to another vertex
        /// \param vertex Target vertex to receive incoming edges
        /// \param spacetime The spacetime context (must be non-null)
        /// \return Pair of (old edges removed, new edges created)
        ///
        /// \f$ u \rightarrow \text{this} \f$ becomes
        /// \f$ u \rightarrow \text{vertex} \f$: the source vertices are unchanged and
        /// only the target is redirected.
        ///
        /// With TESSERA_ASSERTIONS defined, throws std::runtime_error if spacetime is
        /// null, and checks that sourceVertex != this.
        ///
        std::pair<EdgePtrSet, EdgePtrSet> moveInEdgesTo(const VertexPtr &vertex, Spacetime *spacetime);

        ///
        /// \brief Move only outgoing edges to another vertex
        /// \param vertex Target vertex to become new source
        /// \param spacetime The spacetime context (must be non-null)
        /// \return Pair of (old edges removed, new edges created)
        ///
        /// \f$ \text{this} \rightarrow u \f$ becomes
        /// \f$ \text{vertex} \rightarrow u \f$: the target vertices are unchanged and
        /// only the source is redirected.
        ///
        /// With TESSERA_ASSERTIONS defined, throws std::runtime_error if spacetime is
        /// null, and checks that targetVertex != this.
        ///
        std::pair<EdgePtrSet, EdgePtrSet> moveOutEdgesTo(const VertexPtr &vertex, Spacetime *spacetime);

        // ========================================
        // Operators and Utilities
        // ========================================

        ///
        /// \brief Equality comparison based on vertex ID
        /// \param vertex Vertex to compare against
        /// \return true if IDs match, false otherwise
        ///
        /// Two vertices are equal iff they have the same id, regardless of coordinates
        /// or topology, consistent with the hash function.
        ///
        bool operator==(const Vertex &vertex) const noexcept;

        ///
        /// \brief Human-readable representation
        /// \return LaTeX-formatted UTF-8 string describing the vertex
        ///
        /// With TESSERA_VERBOSE defined, shows the vertex id, in-degree, out-degree and
        /// time, e.g. \f$ V_{42}^{in=3} _{out=5}~(t=1.0) \f$. Otherwise returns the empty
        /// string.
        ///
#ifdef TESSERA_VERBOSE
        std::string toString() const noexcept;
#else
        std::string toString() const noexcept {
            return "";
        };
#endif

        // ========================================
        // Public Members
        // ========================================

        ///
        /// \brief Fingerprint for hashing and equality testing
        ///
        /// Computed from the vertex id and used as the hash-table key for SimplexPtrSet,
        /// VertexPtrSet and similar, giving O(1) lookup. Public so hot paths can read it
        /// directly.
        ///
        Fingerprint fingerprint;

        /// Index into VertexList::liveVec_ (maintained by VertexList).
        std::uint32_t liveIdx_{UINT32_MAX};

    private:
        // ========================================
        // Private Implementation
        // ========================================

        /// Edge direction for internal moveEdgesToImpl implementation
        enum class EdgeDirection { In, Out };

        ///
        /// \brief Internal implementation for moving edges
        /// \param recipient Target vertex
        /// \param spacetime Spacetime context
        /// \param direction Whether to move In or Out edges
        /// \return Pair of (old edges, new edges)
        ///
        std::pair<EdgePtrSet, EdgePtrSet>
        moveEdgesToImpl(const VertexPtr &recipient, Spacetime *spacetime, EdgeDirection direction);

        // ========================================
        // Private Members
        // ========================================

        Edges outEdges{};              ///< Edges where this vertex is the source
        Edges inEdges{};               ///< Edges where this vertex is the target
        Simplices simplices{};         ///< Simplices containing this vertex
        std::uint64_t id;              ///< Unique identifier
        std::vector<double> coordinates{};  ///< Spacetime position (may be empty)
};

}  // namespace tessera::mesh

// ========================================
// Standard Library Specializations
// ========================================

namespace std {

///
/// \brief Hash function specialization for tessera::mesh::Vertex
///
/// Lets Vertex objects be keys in std::unordered_set and std::unordered_map. The hash
/// is taken from the vertex id, so equal vertices hash equally. O(1): delegates to
/// std::hash<std::uint64_t>.
///
template<>
struct hash<tessera::mesh::Vertex> {
    size_t operator()(const tessera::mesh::Vertex &vertex) const noexcept {
        return std::hash<std::uint64_t>{}(vertex.getId());
    }
};

///
/// \brief Hash function specialization for tessera::mesh::Vertex*
///
/// Lets VertexPtr (Vertex*) be a hash-table key. Hashes the vertex id, not the pointer
/// address, so two distinct pointers to vertices with the same id hash equally.
///
template<>
struct hash<tessera::mesh::Vertex*> {
    size_t operator()(tessera::mesh::Vertex* const &vertex) const noexcept {
        return std::hash<std::uint64_t>{}(vertex->getId());
    }
};

///
/// \brief Equality comparison specialization for tessera::mesh::Vertex
///
/// Used by standard library containers to compare Vertex objects: two vertices are
/// equal iff they have the same id, consistent with the hash specialization above.
///
template<>
struct equal_to<tessera::mesh::Vertex> {
    size_t operator()(const tessera::mesh::Vertex &a, const tessera::mesh::Vertex &b) const noexcept {
        return a.getId() == b.getId();
    }
};

///
/// \brief Equality comparison specialization for tessera::mesh::Vertex*
///
/// Compares vertices by id, not by pointer address, consistent with the hash
/// specialization for Vertex*.
///
template<>
struct equal_to<tessera::mesh::Vertex*> {
    size_t operator()(tessera::mesh::Vertex* const &a, tessera::mesh::Vertex* const &b) const noexcept {
        return a->getId() == b->getId();
    }
};

}  // namespace std
#endif //TESSERA_TESSERA_SRC_VERTEX_H_

