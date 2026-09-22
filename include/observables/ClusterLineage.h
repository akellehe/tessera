// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_OBSERVABLES_CLUSTERLINEAGE_H
#define TESSERA_OBSERVABLES_CLUSTERLINEAGE_H

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "cobordism/ChainComplex.h"
#include "observables/Record.h"

namespace tessera::observables {

/// One level of an interaction history: the simplicial complex \f$ K_\ell \f$
/// that the interactions among level \f$ \ell \f$'s response vertices generate.
struct LevelComplex {
  /// The declared cells of \f$ K_\ell \f$ as vertex-id tuples in this level's
  /// own numbering \f$ 0,1,\dots,\texttt{vertices}-1 \f$ (the order of the ids
  /// within a cell is irrelevant). Declaring the top cells is enough: the face
  /// closure supplies every lower cell.
  std::vector<std::vector<std::uint64_t>> cells{};
  /// The number of vertices of \f$ K_\ell \f$. Every vertex id appearing in
  /// `cells` must be smaller than it. A level may carry vertices that no cell
  /// uses; they are still vertices of the cobordism and still carry a fiber
  /// edge.
  std::size_t vertices{0};
};

/// The interaction cobordism \f$ W \f$ of one or more ticks.
///
/// \f$ W \f$'s vertices are the levels' vertices laid end to end in level
/// order, so level \f$ \ell \f$'s own vertex \f$ v \f$ is \f$ W \f$'s vertex
/// \f$ \texttt{vertexOffsets}[\ell] + v \f$. Laying them out in that order
/// makes every incoming vertex id smaller than every outgoing one, so the
/// complex's reference orientation (ascending vertex id) already runs from the
/// incoming end towards the outgoing end and a fiber edge needs no sign of its
/// own.
struct InteractionCobordism {
  /// The chain complex of \f$ W \f$, built from `cells` by
  /// `cobordism::ChainComplex::fromCells`. It is not pure: the prisms over a
  /// level's top cells sit beside the next level's own top cells, which are one
  /// dimension lower.
  cobordism::ChainComplex complex{};
  /// Every declared cell of \f$ W \f$, as vertex-id tuples in \f$ W \f$'s
  /// numbering, in generation order (each level's own cells, then each step's
  /// prism cells).
  std::vector<std::vector<std::uint64_t>> cells{};
  /// `vertexOffsets[l]` is the id in \f$ W \f$ of level `l`'s vertex 0, for
  /// `l = 0 ... levels`; the last entry is \f$ W \f$'s total vertex count.
  std::vector<std::uint64_t> vertexOffsets{};
  /// The number of levels. The number of interaction steps is `levels - 1`.
  std::size_t levels{0};
  /// `levelOf[w]` is the level of \f$ W \f$'s vertex `w`.
  std::vector<std::size_t> levelOf{};
  /// `responseOf[w]` is the vertex of \f$ W \f$ at level `levelOf[w] + 1` that
  /// `w` reduces to, and the fiber edge of `w` is the pair
  /// `(w, responseOf[w])`. A vertex of the last level has no response vertex
  /// and carries the sentinel `w` itself.
  std::vector<std::uint64_t> responseOf{};
  /// The 1-simplices of \f$ W \f$ in canonical \f$ C_1(W) \f$ order, each as
  /// its ascending vertex pair. It is kept beside the complex so that an edge
  /// can be located by binary search on the lexicographic order rather than by
  /// rebuilding the cell list on every lookup.
  std::vector<std::vector<std::uint64_t>> edges{};

  /// The vertices of the incoming boundary \f$ \partial_{\mathrm{in}}W \f$:
  /// every vertex of level 0.
  [[nodiscard]] std::vector<std::uint64_t> incomingVertices() const;
  /// The vertices of the outgoing boundary \f$ \partial_{\mathrm{out}}W \f$:
  /// every vertex of the last level.
  [[nodiscard]] std::vector<std::uint64_t> outgoingVertices() const;
  /// The canonical \f$ C_1(W) \f$ index of the edge on vertices `a` and `b`,
  /// or -1 when the pair is not an edge of \f$ W \f$.
  [[nodiscard]] int edgeIndex(std::uint64_t a, std::uint64_t b) const;
  /// The canonical \f$ C_1(W) \f$ indices of the fiber edges, in ascending
  /// order of the edge's lower vertex. These are the only timelike edges.
  [[nodiscard]] std::vector<int> fiberEdges() const;
  [[nodiscard]] Record toRecord() const;
};

/// A cooriented separating cut \f$ \Sigma \f$.
///
/// The cut is carried by the integer 0-cochain \f$ u \f$ that is 0 on the
/// incoming side and 1 on the outgoing side. \f$ \Sigma \f$ is the
/// codimension-one cycle dual to \f$ \delta u \f$, and it is closed for free:
/// \f$ \delta\delta = 0 \f$ makes \f$ \delta u \f$ a cocycle, so no closedness
/// has to be imposed or checked on a candidate list of cells. It separates for
/// free as well: every edge of \f$ W \f$ with one endpoint on each side is a
/// crossing edge, so no path from the incoming boundary to the outgoing
/// boundary avoids the cut. Its coorientation is the direction from the
/// incoming side to the outgoing side, inherited from the cobordism's own
/// incoming/outgoing coorientation and from nothing else.
struct CoorientedCut {
  /// `side[w]` is 0 when \f$ W \f$'s vertex `w` lies on the incoming side of
  /// the cut and 1 when it lies on the outgoing side.
  std::vector<int> side{};
  /// The canonical \f$ C_1(W) \f$ indices of the edges the cut crosses: those
  /// whose two endpoints lie on opposite sides.
  std::vector<int> crossingEdges{};
  /// The coorientation sign of each entry of `crossingEdges`: \f$ +1 \f$ when
  /// the edge's canonical orientation (from its smaller to its larger vertex
  /// id) runs from the incoming side to the outgoing side, \f$ -1 \f$ when it
  /// runs the other way.
  std::vector<int> crossingSigns{};
  /// Whether the cut is a legitimate separating slice: it assigns a side to
  /// every vertex, puts the whole incoming boundary on the incoming side and
  /// the whole outgoing boundary on the outgoing side.
  bool separates{false};
  /// The named reasons `separates` is false; empty when it is true.
  std::vector<std::string> failedCertificates{};
  [[nodiscard]] Record toRecord() const;
};

/// An oriented cluster lineage: an integral one-chain
/// \f$ c_Q \in C_1(W,\partial W;\mathbb{Z}) \f$.
struct Lineage {
  /// The name the lineage is reported under. It carries no arithmetic.
  std::string clusterId{};
  /// One integer coefficient per 1-simplex of \f$ W \f$, in canonical
  /// \f$ C_1(W) \f$ order; the coefficient is read against the edge's
  /// canonical orientation (from its smaller to its larger vertex id).
  std::vector<int> coefficients{};
  /// \f$ n_Q \f$, the fermion number carried on the lineage: the number of
  /// occupied sheets of a sheeted cluster. It weights the lineage in the total
  /// \f$ N_q \f$ and never enters \f$ N_Q \f$ itself.
  int fermionNumber{1};
  [[nodiscard]] Record toRecord() const;
};

/// The reading of one lineage against one cut.
struct LineageNumberRead {
  /// The lineage's own name, copied through.
  std::string clusterId{};
  /// \f$ N_Q = c_Q\cdot\Sigma \in \mathbb{Z} \f$.
  int number{0};
  /// \f$ n_Q \f$, copied through from the lineage.
  int fermionNumber{1};
  /// Whether \f$ c_Q \f$ is a relative cycle, that is whether
  /// \f$ \partial c_Q \f$ is supported on \f$ \partial W \f$. Only a relative
  /// cycle carries a homology class, and only a homology class makes
  /// \f$ N_Q \f$ independent of the cut.
  bool relativeCycle{false};
  /// The vertices of \f$ W \f$ that are not on \f$ \partial W \f$ and at which
  /// \f$ \partial c_Q \f$ does not vanish: the lineage's sources in the
  /// interior. A source between two cuts is exactly what makes those two cuts
  /// disagree.
  std::vector<std::uint64_t> interiorSources{};
  /// Whether the cut used for the reading separates.
  bool cutSeparates{false};
  /// The named reasons the reading is not a certified intersection number;
  /// empty when it is one.
  std::vector<std::string> failedCertificates{};
  [[nodiscard]] Record toRecord() const;
};

/// The reading of a collection of lineages against one cut.
struct TotalLineageRead {
  /// \f$ N_q(\Sigma) = \sum_Q n_Q\,c_Q\cdot\Sigma \f$.
  int fermionNumber{0};
  /// \f$ B(\Sigma) = N_q(\Sigma)/3 \f$. The factor \f$ 1/3 \f$ is an explicit
  /// physical calibration — one occupied sheet per unit of baryon number on a
  /// three-sheeted cluster — and not a topological theorem.
  double baryonNumber{0.0};
  /// One reading per lineage, in the order the lineages were given.
  std::vector<LineageNumberRead> perLineage{};
  /// The named reasons the total is not certified: the union of the per-lineage
  /// reasons.
  std::vector<std::string> failedCertificates{};
  [[nodiscard]] Record toRecord() const;
};

/// # ClusterLineage
///
/// The oriented integer of a cluster's history: the lineage number
/// \f$ N_Q \f$, read as the simplicial intersection pairing of an integral
/// one-chain with a cooriented cut on the interaction cobordism.
///
/// ## The cobordism
///
/// One tick is the interaction cobordism \f$ W_\ell \f$: the mapping cylinder
/// of the reduction map from the level's complex \f$ K_\ell \f$ onto the
/// response vertices of the next level, with the cells of \f$ K_{\ell+1} \f$
/// attached on its outgoing end. It contains \f$ K_\ell \f$, the fiber edges
/// joining each vertex of \f$ K_\ell \f$ to its response vertex, and
/// \f$ K_{\ell+1} \f$, with
/// \f$ \partial W_\ell = K_\ell \sqcup K_{\ell+1} \f$. The geometric
/// realization is required: a purely algebraic reduction supplies the map but
/// no interior cells, and without interior cells a separating cut has nothing
/// to separate and a lineage has nothing to cross.
///
/// The mapping cylinder is triangulated by the staircase rule. For a cell
/// \f$ \sigma = (v_0 < v_1 < \dots < v_m) \f$ of \f$ K_\ell \f$ and the
/// reduction \f$ f \f$, the cells
/// \f[
///   S_j = \{v_0,\dots,v_j\}\cup\{f(v_j),\dots,f(v_m)\},\qquad j = 0,\dots,m,
/// \f]
/// are the prisms over \f$ \sigma \f$, the image part being a set. Where
/// \f$ f \f$ identifies two of \f$ \sigma \f$'s vertices the prism carries one
/// image vertex rather than two and is one dimension lower; that collapse is
/// what the reduction does, and the cell is kept rather than discarded. The two
/// parts of a prism draw on disjoint vertex-id ranges, so no prism can repeat a
/// vertex. Every fiber edge \f$ (v_j, f(v_j)) \f$ is a face of \f$ S_j \f$ and
/// the image \f$ f(\sigma) \f$ is a face of \f$ S_0 \f$, so declaring the top
/// cells of each level is enough. The fiber edges are the only timelike edges
/// of \f$ W \f$; the Lorentzian rotation of timelike squared lengths, where it
/// is applied at all, applies to them alone.
///
/// When a lineage spans several interaction steps the relevant cobordism is
/// the concatenation \f$ W = W_1\cup_\partial W_2\cup_\partial\cdots \f$ of
/// successive steps, with
/// \f$ \partial W = \partial_{\mathrm{in}}W_1 \sqcup
/// \partial_{\mathrm{out}}W_{\mathrm{last}} \f$. `history` builds the
/// concatenation directly, so an intermediate level's cells are shared between
/// the step that ends on it and the step that starts from it rather than
/// duplicated. Every statement below holds for one step and for a
/// concatenation alike.
///
/// ## The cut and the pairing
///
/// A separating slice \f$ \Sigma \f$ is a cooriented closed codimension-one
/// simplicial cut separating \f$ \partial_{\mathrm{in}}W \f$ from
/// \f$ \partial_{\mathrm{out}}W \f$. Because the boundary components are
/// closed, \f$ \Sigma \f$ is an absolute class in \f$ H_{n-1}(W) \f$,
/// homologous to both, and the intersection number with a lineage
/// \f$ c_Q \in H_1(W,\partial W) \f$ is the Lefschetz pairing
/// \f$ H_1(W,\partial W)\times H_{n-1}(W)\to\mathbb{Z} \f$, well defined
/// precisely because \f$ \Sigma \f$ is closed. This class represents the cut by
/// the integer 0-cochain \f$ u \f$ of `CoorientedCut::side`, and the pairing is
/// \f[
///   N_Q = c_Q\cdot\Sigma = \langle \delta u, c_Q\rangle
///        = \langle u, \partial_1 c_Q\rangle
///        = \sum_{e=(a<b)} (c_Q)_e\,\bigl(u(b) - u(a)\bigr),
/// \f]
/// computed with \f$ W \f$'s own boundary operator \f$ \partial_1 \f$. The sum
/// runs over the crossing edges alone, each contributing its coefficient times
/// its coorientation sign. No Lorentzian distance, real projection or level-set
/// ordering is required, and no sign is extracted from a spectral coordinate,
/// from the connection, from an eigenvalue or from a density: the coorientation
/// of the cobordism is the entire causal datum.
///
/// Three consequences follow exactly, with no tolerance and no estimate.
///
///   * **Reversal.** Reversing the lineage sends
///     \f$ c_Q\mapsto -c_Q \f$ and \f$ N_Q\mapsto -N_Q \f$, since the pairing
///     is linear in \f$ c_Q \f$.
///   * **Independence of the cut.** Two cuts \f$ u \f$ and \f$ u' \f$ that both
///     vanish on \f$ \partial_{\mathrm{in}}W \f$ and are both 1 on
///     \f$ \partial_{\mathrm{out}}W \f$ differ by a cochain \f$ u-u' \f$
///     supported in the interior, so
///     \f$ N_Q(u) - N_Q(u') = \langle u-u',\partial c_Q\rangle \f$. That
///     vanishes whenever \f$ \partial c_Q \f$ has no support where the two cuts
///     differ — whenever the lineage has no source in the slab between them.
///     For a relative cycle, whose boundary lies entirely on \f$ \partial W \f$,
///     it vanishes for every pair of cuts.
///   * **Pair creation.** A lineage that is the boundary of an oriented pair
///     surface, \f$ c = \partial_2 S \f$, has
///     \f$ N = \langle\delta u,\partial_2 S\rangle
///          = \langle\delta\delta u, S\rangle = 0 \f$, so the two lineages the
///     pair surface's boundary decomposes into carry \f$ +1 \f$ and \f$ -1 \f$
///     together.
///
/// ## The totals
///
/// For a collection of certified cluster and anti-cluster lineages,
/// \f$ N_q(\Sigma) = \sum_Q n_Q\,c_Q\cdot\Sigma \f$ and
/// \f$ B(\Sigma) = N_q(\Sigma)/3 \f$, where \f$ n_Q \f$ is the fermion number
/// carried on the lineage — the number of occupied sheets of a sheeted cluster
/// — so that a baryon may be one three-sheeted cluster carrying one occupied
/// mode per sheet, \f$ n_Q = 3 \f$, or three clusters carrying one each. The
/// factor \f$ 1/3 \f$ is one explicit physical calibration, not a topological
/// theorem.
///
/// The primary integer is the oriented intersection \f$ N_Q \f$. The relative
/// determinant winding \f$ \nu \f$ of the color transport is an independent
/// agreement test where a closed relative family exists; a disagreement is a
/// defect signal and is reported, never resolved here. Nothing in this
/// construction refers to an incidence cycle, a Betti number or a hole: a
/// cluster contained in a contractible region still has an oriented lineage
/// crossing the cobordism, because a path through a contractible region still
/// intersects a separating cut.
class ClusterLineage {
 public:
  static constexpr std::string_view kRecordKey = "cluster_lineage";
  static constexpr int kSchemaVersion = 1;

  // ---- the cobordism ----

  /// The interaction cobordism of a history of levels.
  ///
  /// @param levels The levels \f$ K_0,\dots,K_L \f$, in order; at least two.
  /// @param reductions One reduction per step, `reductions[l][v]` being the
  ///   index among level `l+1`'s vertices of the response vertex that level
  ///   `l`'s vertex `v` reduces to. There are `levels.size() - 1` of them and
  ///   `reductions[l]` has one entry per vertex of level `l`.
  /// @throws std::invalid_argument when fewer than two levels are given, when
  ///   the number of reductions does not match the number of steps, when a
  ///   reduction's length does not match its level's vertex count, when a
  ///   reduction names a vertex the next level does not have, when a level's
  ///   cell names a vertex that level does not have, or when a level has no
  ///   vertices.
  [[nodiscard]] static InteractionCobordism history(
      const std::vector<LevelComplex> &levels,
      const std::vector<std::vector<std::size_t>> &reductions);

  /// One interaction step: `history({incoming, outgoing}, {reduction})`.
  [[nodiscard]] static InteractionCobordism mappingCylinder(
      const LevelComplex &incoming, const std::vector<std::size_t> &reduction,
      const LevelComplex &outgoing);

  // ---- cuts ----

  /// The cut placed between level `afterLevel` and level `afterLevel + 1`:
  /// \f$ u(w) = 0 \f$ for a vertex of a level at or below `afterLevel` and
  /// \f$ u(w) = 1 \f$ above it. Every level boundary supplies one such cut and
  /// they are all homologous, so they all give the same \f$ N_Q \f$ for a
  /// lineage with no interior source.
  /// @throws std::invalid_argument when `afterLevel` is not below the last
  ///   level, which would leave one of the two boundaries on the wrong side.
  [[nodiscard]] static CoorientedCut levelCut(const InteractionCobordism &W,
                                              std::size_t afterLevel);

  /// The cut carried by a declared side per vertex, validated. Any assignment
  /// that is 0 on the whole incoming boundary and 1 on the whole outgoing
  /// boundary is a separating cut homologous to every `levelCut`, which is how
  /// a cut is moved past an individual vertex to test independence of the cut.
  /// A declaration that fails validation returns a cut with
  /// `separates = false` and the reasons named; it is not thrown on, because a
  /// failed cut is a reportable reading and not a programming error.
  /// @throws std::invalid_argument when `side` does not have one entry per
  ///   vertex of \f$ W \f$.
  [[nodiscard]] static CoorientedCut cutFromSides(const InteractionCobordism &W,
                                                  const std::vector<int> &side);

  // ---- lineages ----

  /// The lineage through the fiber edges out of one starting vertex: the chain
  /// \f$ \sum_\ell (w_\ell \to w_{\ell+1}) \f$, where \f$ w_0 \f$ is
  /// `startVertex` and \f$ w_{\ell+1} \f$ is \f$ w_\ell \f$'s response vertex.
  /// This is the lineage of a cluster tracked from the level of
  /// `startVertex` to the last level.
  /// @throws std::invalid_argument when `startVertex` is not a vertex of
  ///   \f$ W \f$.
  [[nodiscard]] static Lineage fromFiberPath(const InteractionCobordism &W,
                                             std::uint64_t startVertex,
                                             int fermionNumber = 1,
                                             const std::string &clusterId = {});

  /// The lineage of a cluster whose support is tracked across frames: one
  /// support per level, in level order starting at level `firstLevel`, each
  /// given in its own level's vertex numbering. The chain is the fiber path
  /// through each support's representative vertex, the smallest vertex id the
  /// support contains, which makes the construction deterministic. Which
  /// representative is chosen does not change \f$ N_Q \f$ against a level cut,
  /// because the pairing reads only the levels the chain's endpoints lie on.
  /// @throws std::invalid_argument when a support is empty, when a support
  ///   names a vertex its level does not have, when the supports do not cover
  ///   consecutive levels inside \f$ W \f$, or when a support's representative
  ///   does not reduce into the next support.
  [[nodiscard]] static Lineage fromTrackedSupports(
      const InteractionCobordism &W, std::size_t firstLevel,
      const std::vector<std::vector<std::uint64_t>> &supports, int fermionNumber = 1,
      const std::string &clusterId = {});

  /// The lineage along a declared vertex path of \f$ W \f$: consecutive
  /// vertices must be an edge of \f$ W \f$, and each step contributes
  /// \f$ +1 \f$ to that edge when it runs from the smaller to the larger vertex
  /// id and \f$ -1 \f$ when it runs the other way.
  /// @throws std::invalid_argument when the path has fewer than two vertices,
  ///   names a vertex \f$ W \f$ does not have, or steps along a pair that is
  ///   not an edge of \f$ W \f$.
  [[nodiscard]] static Lineage fromVertexPath(const InteractionCobordism &W,
                                              const std::vector<std::uint64_t> &path,
                                              int fermionNumber = 1,
                                              const std::string &clusterId = {});

  /// The lineage with every coefficient negated: the same cluster history
  /// traversed in the opposite direction, \f$ c_Q\mapsto -c_Q \f$.
  [[nodiscard]] static Lineage reversed(const Lineage &lineage);

  /// The boundary \f$ \partial_2 S \f$ of an oriented pair surface \f$ S \f$,
  /// as a lineage. \f$ S \f$ is an integral two-chain: one coefficient per
  /// 2-simplex of \f$ W \f$, in canonical \f$ C_2(W) \f$ order. This is how a
  /// pair event enters: its total lineage number is zero, so it creates
  /// \f$ +1 \f$ and \f$ -1 \f$ together.
  /// @throws std::invalid_argument when `surface` does not have one entry per
  ///   2-simplex of \f$ W \f$.
  [[nodiscard]] static Lineage pairSurfaceBoundary(const InteractionCobordism &W,
                                                   const std::vector<int> &surface,
                                                   int fermionNumber = 1,
                                                   const std::string &clusterId = {});

  // ---- the pairing ----

  /// \f$ \partial_1 c_Q \f$, one integer per vertex of \f$ W \f$, computed with
  /// the complex's own boundary operator.
  /// @throws std::invalid_argument when the lineage does not have one
  ///   coefficient per 1-simplex of \f$ W \f$.
  [[nodiscard]] static std::vector<int> relativeBoundary(const InteractionCobordism &W,
                                                         const Lineage &lineage);

  /// \f$ N_Q = c_Q\cdot\Sigma \f$, the bare integer, with no certificate
  /// attached. `read` is the certified route.
  /// @throws std::invalid_argument when the lineage does not have one
  ///   coefficient per 1-simplex of \f$ W \f$, or the cut one side per vertex.
  [[nodiscard]] static int intersectionNumber(const InteractionCobordism &W,
                                              const CoorientedCut &cut,
                                              const Lineage &lineage);

  /// \f$ N_Q \f$ with its certificates: whether the cut separates, whether the
  /// lineage is a relative cycle, and which interior vertices carry a source.
  [[nodiscard]] static LineageNumberRead read(const InteractionCobordism &W,
                                              const CoorientedCut &cut,
                                              const Lineage &lineage);

  /// \f$ N_q(\Sigma) = \sum_Q n_Q\,c_Q\cdot\Sigma \f$ and
  /// \f$ B(\Sigma) = N_q(\Sigma)/3 \f$ over a collection of lineages.
  [[nodiscard]] static TotalLineageRead totals(const InteractionCobordism &W,
                                               const CoorientedCut &cut,
                                               const std::vector<Lineage> &lineages);

  // ---- the compilation order ----

  /// The deterministic compilation-order key of one cluster's oriented
  /// lineage.
  ///
  /// Writing the exterior algebra as an ordered tensor product, or writing its
  /// creation operators in Jordan-Wigner form, needs a chosen order of the
  /// one-particle modes. The order is fixed by oriented component lineage: the
  /// modes a cluster carries are ordered together, and the clusters are
  /// ordered by the oriented integers of their histories. This function turns
  /// one such history into the string key that
  /// `quantum::EdgeModeRegistry::canonicalModeOrder` sorts on, so that the
  /// primary key of the mode order is a physical, relabelling-invariant
  /// integer rather than a vertex id.
  ///
  /// Sorting the keys lexicographically orders the clusters by ascending
  /// lineage number \f$ N_Q \f$, then by ascending fermion number
  /// \f$ n_Q \f$, then by cluster name. The two integers are written as
  /// fixed-width offset decimals, so that the lexicographic order of the
  /// strings is exactly the numeric order of the integers they encode, with
  /// negative lineage numbers — the anti-clusters — ordered before the
  /// positive ones.
  ///
  /// @throws std::invalid_argument when the reading carries a failed
  ///   certificate. A cut that does not separate, or a lineage that is not a
  ///   relative cycle, has no cut-independent lineage number, and ordering the
  ///   modes by a number that a different cut would change would make the
  ///   compilation order depend on the cut.
  [[nodiscard]] static std::string orderKey(const LineageNumberRead &read);
};

}  // namespace tessera::observables

#endif  // TESSERA_OBSERVABLES_CLUSTERLINEAGE_H
