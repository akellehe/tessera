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
/// The topology-changing surgical cone: the move set that creates and destroys
/// `b_k` holes. Every move is gated on the manifold check below.
///
/// References: Lickorish, "Simplicial moves on complexes and manifolds",
/// arXiv:math/9911256; for the causal dynamical triangulations (CDT) edge
/// convention used by a timelike cone-in, Ambjorn, Jurkiewicz and Loll,
/// arXiv:hep-th/0105267.
///
/// ## The moves
/// * **cone-out** (`coneOut`) — remove one top cell, plus the edges and
///   vertices it orphans. On a closed \f$ d \f$-manifold this opens a boundary;
///   removing a cell disjoint from an existing hole raises \f$ b_{d-1} \f$ by 1.
/// * **cone-in** (`coneIn`) — add one top cell on a fresh vertex joined to
///   \f$ d \f$ existing vertices. Capping a hole's boundary lowers
///   \f$ b_{d-1} \f$ by 1.
/// * **bridge** (`bridge`) — add one top cell on \f$ d+1 \f$ existing vertices,
///   with no fresh apex, auto-wiring the edges it lacks. It draws bulk between
///   two boundary surfaces on those surfaces' own vertices; whether that buries
///   a surface face or creates a chord is the caller's choice of vertices.
///
/// ## The gate
/// A move is accepted only if `ChainComplex::dualComplexIsValid` holds over the
/// candidate top cells. A rejected move is rolled back, leaving the complex
/// bit-identical to its pre-move state.
///
/// ## Lifecycle and exact reversibility
/// Accepted moves are pushed on a stack and unwind LIFO; `rollback()` undoes
/// the last one, restoring every edge length and phase bit-for-bit, so a round
/// trip leaves the dual Regge action invariant. The facet/coface bookkeeping is
/// restored with the values, so the coface walk behind ``Simplex::dualVolume``
/// retraces the pre-move dual volumes.
///
/// A bridge's undo differs: the surfaces the bulk is drawn onto are registered
/// \f$ (d-1) \f$-cells that no top cell covers, so a bridge records which of
/// its sub-faces already existed and unregisters only what it introduced.
class SurgicalCone {
 public:
  /// Bind the cone to a spacetime. Does not mutate it.
  explicit SurgicalCone(Spacetime *spacetime);
  ~SurgicalCone();

  SurgicalCone(const SurgicalCone &) = delete;
  SurgicalCone &operator=(const SurgicalCone &) = delete;

  /// Gated surgical **cone-out**: remove the top cell whose sorted vertex ids
  /// equal \p cell, plus its orphaned faces and edges and any vertex thereby
  /// isolated, so the simplex set stays the closure of the surviving top cells.
  /// Accepted only if the result is a valid manifold-with-boundary. Returns
  /// `(true, "ok")` on acceptance; otherwise the complex is restored and the
  /// reason returned. Rejects removing the last top cell.
  std::pair<bool, std::string> coneOut(const std::vector<std::uint64_t> &cell);

  /// Gated surgical **cone-in**: create a fresh vertex, join it to the \f$ d \f$
  /// vertices \p targetVerts to form a new top cell, then accept only if the
  /// result is a valid manifold-with-boundary. Returns `(true, "ok")` on
  /// acceptance; otherwise the additions are undone and the reason returned.
  ///
  /// \p timelike makes the edges joining the fresh apex to \p targetVerts
  /// timelike instead of spacelike: the CDT \f$ (4,1) \f$ split. Only
  /// apex-incident edges are affected. The magnitude follows the CDT convention
  /// \f$ \ell_t^2 = -\alpha\,\ell_s^2 \f$ with \f$ \alpha = 1 \f$ on unit
  /// spacelike edges. Nothing prevents the geometric relaxation from later
  /// driving such an edge spacelike.
  std::pair<bool, std::string> coneIn(
      const std::vector<std::uint64_t> &targetVerts, bool timelike = false);

  /// The timelike squared length a `timelike` cone-in writes, in the CDT
  /// convention \f$ \ell_t^2 = -\alpha\,\ell_s^2 \f$ with \f$ \alpha = 1 \f$.
  static constexpr double kTimelikeSquaredLength = -1.0;

  /// Gated surgical **bridge**: create the top cell on the \f$ d+1 \f$ existing,
  /// distinct vertices \p cellVertices (matched by vertex set; stored in the
  /// order given), auto-wiring every edge the cell lacks
  /// (`Spacetime::autoWiredLength`), then accept only if the result is a valid
  /// manifold-with-boundary. Returns `(true, "ok")` on acceptance; otherwise the
  /// cell and every edge it alone introduced are removed bit-exactly and the
  /// reason returned. Refuses a vertex count other than \f$ d+1 \f$, a repeated
  /// or unknown vertex, and a cell that already exists. The manifold check is
  /// the only further gate.
  std::pair<bool, std::string> bridge(const std::vector<std::uint64_t> &cellVertices);

  /// Undo the last accepted move (LIFO), restoring bit-for-bit every edge
  /// length and phase and the restored cell's facet/coface lattice. Returns
  /// `false` if nothing is applied.
  bool rollback();

  /// Roll every accepted move back, restoring the original complex. Returns the
  /// number of moves undone.
  std::size_t rollbackAll();

  /// Number of accepted, not-yet-rolled-back moves on the stack.
  [[nodiscard]] std::size_t depth() const;

  /// True iff at least one move is accepted and not yet rolled back.
  [[nodiscard]] bool isApplied() const;

  /// The Betti numbers \f$ b_0, \ldots, b_n \f$ over \f$ \mathbb{Q} \f$ of the
  /// current complex (`ChainComplex::fromSpacetime(...).bettiNumbers()`).
  [[nodiscard]] std::vector<int> bettiNumbers() const;

  /// The manifold verdict on the current complex, the gate every move applies.
  /// `(true, "ok")` when it is a valid manifold-with-boundary; otherwise the
  /// first violation is named.
  [[nodiscard]] std::pair<bool, std::string> validate() const;

 private:
  /// One accepted surgical move, with everything needed to invert it exactly.
  struct Move {
    enum class Kind { ConeOut, ConeIn, Bridge };
    Kind kind;
    /// The d+1 vertex ids of the removed (cone-out) / added (cone-in, bridge)
    /// top cell.
    std::vector<std::uint64_t> cell;
    /// Edges touched: removed orphans (cone-out, to re-create) or freshly
    /// inserted edges (cone-in, to drop), each (u, v, l2, phase). The whole
    /// complex l2 is recorded, so the restore is bit-exact on analytically
    /// continued (Im l2 != 0) geometry too.
    std::vector<
        std::tuple<std::uint64_t, std::uint64_t, std::complex<double>, std::complex<double>>>
        edges;
    /// Vertices touched: isolated vertices removed (cone-out, to re-create) /
    /// the single fresh vertex (cone-in, to drop), each (id, coords). An empty
    /// coords vector marks a coordinate-free vertex.
    std::vector<std::pair<std::uint64_t, std::vector<double>>> verts;
    /// Whether the removed top cell carried a materialized facet lattice at
    /// move time (cone-out only). The undo then re-materializes the restored
    /// cell's face lattice. `false` for cone-in and for hosts that never
    /// materialized facets.
    bool hadFacets{false};
    /// Bridge only: the proper sub-faces of the cell (sorted vertex-id tuples)
    /// registered before the move. The undo leaves these alone and unregisters
    /// only the sub-faces the move introduced that no surviving simplex holds.
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
  /// Drop the exact cell of a bridge Move: the top cell, then every sub-face the
  /// move introduced that no surviving simplex holds (largest first), then the
  /// edges the move alone inserted. Pre-existing faces, vertices and edges are
  /// untouched.
  void undoBridge(const Move &m);

  Spacetime *st_;
  std::vector<Move> moves_;
};

}  // namespace tessera::cobordism

#endif  // TESSERA_COBORDISM_SURGICALCONE_H
