// Copyright (c) 2026 Twin Vector Labs LLC. All rights reserved.
//
// Hasse-cover representation of a finite partial order over integer-indexed
// nodes. Storage uses tessera's mesh primitives (tessera::mesh::VertexList,
// tessera::mesh::EdgeList), so Poset instances interoperate with the rest of
// tessera's graph machinery: rendering, GraphML / dot export and the causal-set
// adapter all share the Vertex / Edge types used for the simplicial spacetime.
//
// ─── What this provides ───────────────────────────────────────────────────
//
// • Poset itself — the cover-edge container, with addCover / covers /
//   toDot / fromSpacetime.
// • OrderAgreement struct — pairwise statistics on two posets that share
//   a label set.
// • compareOrders() — Kendall tau, discordant fraction, Hasse edit
//   distance.
//
// All three are general-purpose; the quantum subsystem layers
// majorization-specific glue on top of them in
// include/quantum/Majorization.hpp.
//
// Reference for the causal-order reading: Sorkin, arXiv:gr-qc/0309009; and
// Bombelli, Lee, Meyer & Sorkin, "Space-time as a causal set", 1987.

#pragma once

#include "mesh/EdgeList.h"
#include "mesh/VertexList.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

// === cross-subsystem fwd-decls ===
namespace tessera::spacetime { class Spacetime; }

namespace tessera {

// Bring sibling subsystem names into scope for the root-level Poset class.
using namespace ::tessera::mesh;
using namespace ::tessera::spacetime;

/// Hasse-cover representation of a partial order on integer-indexed nodes.
///
/// Internally stores a VertexList (one Vertex per node, no coordinates) and an
/// EdgeList of cover edges. Cover edges are tessera::mesh::Edge objects; their
/// `squaredLength` and `disposition` fields are unused by the partial-order
/// semantics and stay at their default values.
class Poset {
public:
    /// Default-construct an empty Poset.
    Poset() = default;

    /// Construct with `getNodeCount` nodes, pre-populated as Vertex(id=0..n-1).
    explicit Poset(int getNodeCount);

    /// Value-style copy. The pool-based VertexList / EdgeList do not copy
    /// trivially, because of their back-pointer and fingerprint maps, so the
    /// copy is written out explicitly and callers can pass Poset by value.
    Poset(Poset const& other);
    Poset& operator=(Poset const& other);
    Poset(Poset&& other) noexcept = default;
    Poset& operator=(Poset&& other) noexcept = default;

    /// Number of nodes; valid ids are 0 .. getNodeCount()-1.
    int getNodeCount() const noexcept;

    /// Resize: add empty Vertex(id) entries for missing ids in [0, n). Existing
    /// nodes are not removed if n is smaller than the current count, and cover
    /// edges are preserved across resizes.
    void setNodeCount(int n);

    /// Number of cover edges currently registered.
    int getCoverCount() const noexcept;

    /// Add a strict cover edge a → b: a precedes b with no intermediate. The
    /// caller is responsible for transitivity and acyclicity; Poset does not
    /// validate either.
    ///
    /// Both endpoints must already exist, so setNodeCount() must have been
    /// called or the constructor given a sufficient n. Covers are not
    /// deduplicated: adding the same cover twice creates two parallel edges.
    /// Covers usually come from a transitive reduction, where duplicates
    /// cannot arise.
    void addCover(int a, int b);

    /// Replace the entire cover list with `new_covers`, in one pass. Equivalent
    /// to clearing the edges and calling addCover() for each pair.
    void setCovers(std::vector<std::pair<int, int>> const& new_covers);

    /// Materialize the cover edges as (from, to) integer pairs, in insertion
    /// order. Do not rely on that order: the underlying EdgeList may reorder
    /// entries when it reuses a free slot.
    std::vector<std::pair<int, int>> covers() const;

    /// The underlying mesh primitives, exposed for interoperation with the rest
    /// of tessera (visualization, GraphML export). Mutating through these
    /// handles bypasses Poset's invariants; treat them as read-only.
    VertexList const& vertices() const noexcept { return vertices_; }
    EdgeList const& edges() const noexcept { return edges_; }
    VertexList& vertices() noexcept { return vertices_; }
    EdgeList& edges() noexcept { return edges_; }

    // ─── Factories ─────────────────────────────────────────────────────

    /// Build the causal-set partial order on the vertices of a Spacetime: each
    /// timelike edge becomes a strict precedes-relation oriented from the
    /// earlier to the later vertex, and the resulting DAG is transitively
    /// reduced to cover edges. Node ids are a dense 0..n-1 remapping of the
    /// Spacetime vertex ids in ascending order.
    static Poset fromSpacetime(Spacetime const& st);

    // ─── Output ────────────────────────────────────────────────────────

    /// Graphviz DOT representation of the Hasse diagram, suitable for
    /// `dot -Tsvg`. Nodes are labelled by their integer id; a cover edge is
    /// drawn as "a -> b", meaning a strictly precedes b.
    std::string toDot() const;

private:
    VertexList vertices_;
    EdgeList   edges_;
};

// ─── Pairwise agreement between two posets on a shared label set ─────────

/// Pairwise agreement between two posets A and B on a shared label set.
///
/// Counted over unordered label pairs (i, j) with i < j:
///   - comparable in P: the transitive closure of P relates i to j.
///   - concordant: both posets relate the pair, in the same direction.
///   - discordant: both relate the pair, in opposite directions.
///   - only_a: A relates the pair, B does not.
///   - only_b: B relates the pair, A does not.
///
/// The five counts (concordant, discordant, only_a, only_b, neither) partition
/// the C(nLabels, 2) unordered pairs. With (A, B) = (majorization order,
/// Lieb-Robinson order), `only_a` counts the majorization-related pairs whose
/// endpoints lie outside the Lieb-Robinson cone, and `only_b` is symmetric.
struct OrderAgreement {
    double kendallTau{0.0};         ///< (concordant - discordant) / both, in [-1, 1]
    double discordantFraction{0.0}; ///< discordant / both, in [0, 1]
    double hasseEditDistance{0.0};  ///< |E_a △ E_b| / |E_a ∪ E_b|, in [0, 1]
    int    nConcordant{0};          ///< pairs related by both, same direction
    int    nDiscordant{0};          ///< pairs related by both, opposite directions
    int    nComparableBoth{0};      ///< pairs related by both posets
    int    nOnlyA{0};               ///< pairs related by a only
    int    nOnlyB{0};               ///< pairs related by b only
};

/// Pairwise agreement statistics between two posets on the same label set of
/// size nLabels.
///
/// Complexity: O(nLabels^3) for the Floyd-Warshall transitive closures, then
/// O(nLabels^2) to count pairs. Practical up to a few thousand labels.
OrderAgreement compareOrders(Poset const& a,
                              Poset const& b,
                              int nLabels);

} // namespace tessera
