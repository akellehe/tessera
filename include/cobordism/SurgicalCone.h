// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_COBORDISM_SURGICALCONE_H
#define TESSERA_COBORDISM_SURGICALCONE_H

#include <complex>
#include <cstdint>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace tessera::spacetime {
class Spacetime;
}  // namespace tessera::spacetime

namespace tessera::cobordism {
using namespace ::tessera::spacetime;

/// # SurgicalCone
///
/// The **topology-changing** surgical cone of the Emergent Color Topology epic
/// (#457, T3) — the genuine `b_k`-hole creator. Pachner moves and the stellar
/// refinement cone (T1/T2) are topology-**preserving**: no change in `b_k`
/// comes from them. A *surgical* cone does change the
/// topology, and is dangerous, so **every** move is gated on the full manifold
/// check — surgery is allowed *because* it is gated. Bypassing the gate is
/// exactly what broke the #353 weld; this class never bypasses it.
///
/// ## The two moves
/// * **cone-out** (`coneOut`) — the hole-creator. Remove a single top cell:
///   drop the \f$ d \f$-simplex, then every edge it had that no surviving top
///   cell still covers (the "decrement multiplicity, remove at zero" of the
///   ticket), then any vertex left with no incident edge. Removing one top cell
///   from a closed \f$ d \f$-manifold opens it to a manifold-with-boundary;
///   removing a cell disjoint from an existing hole raises \f$ b_{d-1} \f$ by 1
///   (on \f$ S^3 \f$, \f$ b_2 \f$ — the color register degree).
/// * **cone-in** (`coneIn`) — add a single top cell built on a **fresh** vertex
///   joined to \f$ d \f$ chosen existing vertices. Capping a hole's boundary
///   this way lowers \f$ b_{d-1} \f$ by 1.
///
/// ## The bridge
/// * **bridge** (`bridge`) — the bulk-drawer of the qubit cobordism
///   (historical requirement D1; implementation ticket #960). Create a top cell on
///   \f$ d+1 \f$ **existing** vertices, no fresh apex, auto-wiring every edge
///   the cell lacks with the engine's auto-wired length. It exists because the
///   bulk between two boundary surfaces is *drawn* on their own vertices (R4:
///   "choose a vertex on one of the boundary blocks, cone it into 4 vertices
///   on the other"): a cone-in mints a vertex and a cone-out removes a cell,
///   so neither can join two surfaces. The cell's faces that lie inside one
///   surface are that surface's own triangles, so the drawing never buries a
///   surface face and never creates a chord — that is a property of WHICH
///   vertices the caller chooses (`MultiCobordism` draws vertex splits across
///   two blocks), not of this move, which is deliberately NOT gated on
///   anything but the manifold check: a chord is not a topological defect the
///   gate could see.
///
/// ## The gate
/// After applying a move the candidate complex is accepted **only if**
/// `ChainComplex::dualComplexIsValid` holds over its top cells — a genuine
/// combinatorial **manifold-with-boundary** (facet coface counts in
/// \f$ \{1,2\} \f$, ridge links single paths/cycles, and the #429 recursive
/// \f$ n \geq 4 \f$ vertex-link validation). A move that would pinch the complex
/// or give a facet \f$ > 2 \f$ cofaces is rejected and rolled back, leaving the
/// complex bit-identical to its pre-move state.
///
/// ## Lifecycle and exact reversibility
/// Accepted moves are pushed on a stack; `rollback()` undoes the last one,
/// restoring the complex — every edge length and phase — bit-for-bit, so a
/// round trip leaves the dual Regge action (Re **and** Im) invariant. The
/// facet/coface bookkeeping is restored along with the values: a cone-out
/// prunes the removed cell's orphaned faces before dropping their edges (a
/// registered sub-simplex must never outlive its edges — one that does reads
/// \f$ \ell^2 = 0 \f$ in every later Gram-matrix computation), and the
/// rollback re-materializes the restored cell's face lattice, so the coface
/// walk behind ``Simplex::dualVolume`` retraces the pre-move circumcentric
/// dual volumes exactly (#587). The moves are first-class and composable
/// (cone-out two disjoint cells, then roll both back LIFO).
///
/// A bridge's undo is the one place the registered-simplex invariant differs:
/// the surfaces the bulk is drawn onto are registered \f$ (d-1) \f$-cells that
/// no top cell covers, and `Spacetime::pruneOrphanedSimplices` would delete
/// exactly those. So a bridge records which of its sub-faces already existed
/// and its undo unregisters only the faces the move itself introduced (and
/// the edges it alone inserted), leaving every pre-existing face — covered
/// or not — untouched.
class SurgicalCone {
 public:
  /// Bind the cone to a spacetime. Does not mutate it.
  explicit SurgicalCone(Spacetime *spacetime);
  ~SurgicalCone();

  SurgicalCone(const SurgicalCone &) = delete;
  SurgicalCone &operator=(const SurgicalCone &) = delete;

  /// Gated surgical **cone-out**: remove the top cell whose sorted vertex ids
  /// equal \p cell (plus its orphaned faces, its orphaned edges, and any
  /// vertex thereby isolated — the registered simplex set stays exactly the
  /// closure of the surviving top cells), then accept only if the result is a
  /// valid manifold-with-boundary. Returns
  /// `(true, "ok")` on acceptance; otherwise the complex is restored and the
  /// reason returned. Rejects removing the last top cell (it would drop the
  /// complex dimension).
  std::pair<bool, std::string> coneOut(const std::vector<std::uint64_t> &cell);

  /// Gated surgical **cone-in**: create a fresh vertex, join it to the \f$ d \f$
  /// vertices \p targetVerts to form a new top cell, then accept only if the
  /// result is a valid manifold-with-boundary. Returns `(true, "ok")` on
  /// acceptance; otherwise the additions are undone and the reason returned.
  /// \p timelike (#613) makes the edges joining the fresh apex to \p targetVerts
  /// **timelike** instead of spacelike — the CDT \f$ (4,1) \f$ split, \f$ d \f$
  /// vertices on one slice and the apex on the next. Only apex-incident edges are
  /// affected; pre-existing edges are never touched. Defaults to `false`, which is
  /// byte-identical to the behaviour before this parameter existed.
  ///
  /// The magnitude follows the CDT convention \f$ \ell_t^2 = -\alpha\,\ell_s^2 \f$
  /// with \f$ \alpha = 1 \f$ on unit spacelike edges.
  ///
  /// This seeds a disposition; it does not police one. Nothing prevents the
  /// geometric relaxation from later driving such an edge spacelike — that would be
  /// a runtime guard on the dynamics, which this project does not do.
  std::pair<bool, std::string> coneIn(
      const std::vector<std::uint64_t> &targetVerts, bool timelike = false);

  /// The timelike squared length a `timelike` cone-in writes, in the CDT
  /// convention \f$ \ell_t^2 = -\alpha\,\ell_s^2 \f$ with \f$ \alpha = 1 \f$.
  static constexpr double kTimelikeSquaredLength = -1.0;

  /// Gated surgical **bridge**: create the top cell on the \f$ d+1 \f$
  /// EXISTING, distinct vertices \p cellVertices (matched by vertex SET; the
  /// stored order is the order given), auto-wiring every edge the cell lacks
  /// with the engine's auto-wired length (`Spacetime::autoWiredLength`, the
  /// spacelike class on coordinate-free vertices), then accept only if the
  /// result is a valid manifold-with-boundary. Returns `(true, "ok")` on
  /// acceptance; otherwise the cell and every edge it alone introduced are
  /// removed bit-exactly and the reason returned. Refuses a vertex count other
  /// than \f$ d+1 \f$, a repeated or unknown vertex, and a cell that already
  /// exists. Nothing else is checked here: whether the cell's vertices split
  /// across two boundary blocks without a chord is the caller's draw
  /// (`MultiCobordism`), because the manifold gate is the ONLY gate.
  std::pair<bool, std::string> bridge(const std::vector<std::uint64_t> &cellVertices);

  /// Undo the last accepted move (LIFO), restoring the complex bit-for-bit —
  /// every edge length and phase, and the restored cell's facet/coface
  /// lattice (so circumcentric dual volumes retrace too). Returns `false` if
  /// nothing is applied.
  bool rollback();

  /// Roll every accepted move back, restoring the original complex. Returns the
  /// number of moves undone.
  std::size_t rollbackAll();

  /// Number of accepted, not-yet-rolled-back moves on the stack.
  [[nodiscard]] std::size_t depth() const;

  /// True iff at least one move is accepted and not yet rolled back.
  [[nodiscard]] bool isApplied() const;

  /// The Betti numbers \f$ b_0, \ldots, b_n \f$ (over \f$ \mathbb{Q} \f$) of the
  /// **current** complex (`ChainComplex::fromSpacetime(...).bettiNumbers()`).
  /// The read-out the `b_k`-delta tests assert a surgical move shifts by one.
  [[nodiscard]] std::vector<int> bettiNumbers() const;

  /// The manifold verdict on the **current** complex — the same gate `coneOut` /
  /// `coneIn` apply. `(true, "ok")` when it is a valid manifold-with-boundary;
  /// otherwise the first violation is named.
  [[nodiscard]] std::pair<bool, std::string> validate() const;

 private:
  /// One accepted surgical move, with everything needed to invert it exactly.
  struct Move {
    enum class Kind { ConeOut, ConeIn, Bridge };
    Kind kind;
    /// The d+1 vertex ids of the removed (cone-out) / added (cone-in, bridge)
    /// top cell.
    std::vector<std::uint64_t> cell;
    /// Edges touched: removed orphans (cone-out, to re-create) / freshly
    /// inserted edges (cone-in, to drop), each (u, v, l2, phase). The full
    /// COMPLEX l2 is recorded so the restore is bit-exact on analytically
    /// continued (Im l2 != 0) geometry too — a Re-only record silently
    /// projected every rejected probe onto the real axis (#581).
    std::vector<
        std::tuple<std::uint64_t, std::uint64_t, std::complex<double>, std::complex<double>>>
        edges;
    /// Vertices touched: isolated vertices removed (cone-out, to re-create) /
    /// the single fresh vertex (cone-in, to drop), each (id, coords). An empty
    /// coords vector marks a coordinate-free vertex.
    std::vector<std::pair<std::uint64_t, std::vector<double>>> verts;
    /// Whether the removed top cell carried a materialized facet lattice at
    /// move time (cone-out only). The undo then re-materializes the restored
    /// cell's face lattice — the coface links ``Simplex::dualVolume`` walks —
    /// instead of leaving the cell wired to vertices and edges alone. Left
    /// `false` for cone-in and for hosts that never materialized facets, so
    /// a rollback never creates bookkeeping the pre-move complex lacked.
    bool hadFacets{false};
    /// Bridge only: the proper sub-faces of the cell (sorted vertex-id tuples,
    /// every size from a single vertex up to a facet) that were REGISTERED
    /// before the move. The undo leaves these alone — among them are the
    /// boundary surfaces' own triangles, which no top cell covers and which a
    /// prune would therefore delete — and unregisters only the sub-faces the
    /// move's lifetime introduced that no surviving simplex still holds.
    std::vector<std::vector<std::uint64_t>> preexistingFaces;
  };

  /// Top cells of the bound spacetime as sorted vertex-id tuples (canonical
  /// C_d order), the input to the manifold check.
  [[nodiscard]] std::vector<std::vector<std::uint64_t>> topCells() const;

  /// Vertices-per-top-cell (d+1), or 0 if no dimension is set.
  [[nodiscard]] std::size_t topVerts() const;

  /// Re-create the exact cell of a cone-out Move (its isolated vertices, the top
  /// cell, and the removed edges' lengths/phases).
  void undoConeOut(const Move &m);
  /// Drop the exact cell of a cone-in Move (the top cell, its fresh edges, the
  /// fresh vertex).
  void undoConeIn(const Move &m);
  /// Drop the exact cell of a bridge Move: the top cell, then every sub-face
  /// the move introduced that no surviving simplex holds (largest first, so a
  /// face's coface links are cleaned before its own facets go), then the edges
  /// the move alone inserted. Pre-existing sub-faces, vertices and edges are
  /// never touched.
  void undoBridge(const Move &m);

  Spacetime *st_;
  std::vector<Move> moves_;
};

}  // namespace tessera::cobordism

#endif  // TESSERA_COBORDISM_SURGICALCONE_H
