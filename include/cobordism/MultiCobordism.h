// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_COBORDISM_MULTICOBORDISM_H
#define TESSERA_COBORDISM_MULTICOBORDISM_H

#include <complex>

#include <Eigen/Core>

#include "chainhodge/RieszBand.h"
#include "cobordism/CobordismObjective.h"
#include "cobordism/EigenstateSynthesis.h"
#include "cobordism/HodgeLaplacian.h"
#include "cobordism/PencilLayer.h"
#include "cobordism/SurfaceSeed.h"
#include "spacetime/pachner/AddMove.h"
#include "spacetime/pachner/FlipMove.h"
#include "spacetime/pachner/IFlipMove.h"
#include "spacetime/pachner/RemoveMove.h"
#include <cstdint>
#include <map>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace tessera::spacetime { class Spacetime; }
namespace tessera::mesh { class Edge; }
namespace tessera::observables { class SimplicialQubit; }

namespace tessera::cobordism {
using ::tessera::spacetime::Spacetime;

/// The cobordism frames the analysis overlay retains: one entry per completed
/// pass, oldest first, each holding that pass's components and their candidate
/// bands and anchors. Defined in `src/cobordism/RecursiveFiberSimulation.cpp`
/// and held here by pointer, so this header keeps its small include list.
struct AnalysisFrameHistory;

/// # MultiCobordism
///
/// Emergent-merge optimizer: the merge as an optimization with no prescribed
/// topology and no hand-placed register. From a bare host it grows the register
/// by gated surgical moves under the objective, and reads the register off
/// `getBoundary` at a caller-chosen degree \f$k\f$.
///
/// Reference: Regge and Williams, "Discrete structures in gravity",
/// arXiv:gr-qc/0012035. Baez and Dolan, "Higher-dimensional Algebra and
/// Topological Quantum Field Theory", arXiv:q-alg/9503002.
///
/// The scalar objective is selected explicitly. `Legacy` uses
/// \f$\|\nabla S_{\rm Regge}\|^2+\Gamma r_U\f$; `JointStationarity` uses
/// \f$\beta\|\nabla_zS_{\rm Regge}\|^2+
/// \eta\|\nabla_zS_{\rm Hodge}\|^2\f$; `MediatedCorrespondence` uses
/// \f$r_U+\beta|S_{\rm Regge}(W^*)|\f$.
///
/// Two stages:
///   * **Stage 1 (combinatorial):** greedy best-ΔF single random moves
///     `{add,remove,flip,iflip,cone_out,cone_in,cone_in_timelike,flip_disposition}`
///     — Pachner moves plus coning, the last two setting the causal disposition
///     (`shouldProposeDispositions`) — plus `bridge` on a node with surface
///     inputs while its bridge phase is incomplete (`kBridge`). Each move is
///     gated by `dualComplexValid` and by "no input vertex removed", and is
///     committed only if ΔF < 0. Target-conditioned modes may redraw a stalled
///     batch while the register is not carried; target-free
///     `JointStationarity` ends the stage when no improving sequence is found.
///   * **Stage 2 (geometric):** relax the squared edge coordinates
///     \f$z_e=\ell_e^2\f$ of the whole complex along the selected objective's
///     gradient, then map each accepted \f$z_e\f$ back to the continuous
///     square-root branch of the stored edge length \f$\ell_e\f$.
class MultiCobordism {
 public:
  /// A frame on a block's attached cells: the coordinates the two-body transfer
  /// is read in. The transfer is
  /// \f$ T_{AB} = (Z_A^\vee)^T (\tilde A)_{AB} Z_B \f$
  /// (`chainhodge::PencilSchur::transfer`), the matrix of the whole complex's
  /// pencil block in the frames' coordinates; a change of frame
  /// \f$ Z_A \mapsto Z_A g_A \f$, \f$ Z_B \mapsto Z_B g_B \f$ gives
  /// \f$ g_A^{-1} T g_B \f$. For two qubit tori the frames are their period
  /// frames (`observables::SimplicialQubit::periodFrame`) and \f$ T \f$ is
  /// \f$ 2 \times 2 \f$ in the \f$ |0\rangle, |1\rangle \f$ basis a two-body
  /// target is written in. The frame is never scored; the fiber it expresses
  /// is.
  struct BlockFrame {
    /// The block's attached fiber cells (sorted vertex tuples), one per row.
    std::vector<std::vector<std::uint64_t>> cells;
    /// \f$ Z \f$: one column per basis element on `cells`
    /// (\f$ |cells| \times r \f$, rows in the fiber's attachment order).
    Eigen::MatrixXcd images;
    /// \f$ Z^\vee \f$: the left partner under the transpose pairing, with
    /// \f$ (Z^\vee)^T M_k Z = I_r \f$ for \f$ M_k \f$ the inverse chain metric
    /// (the Whitney mass matrix) of the block's own pencil on `cells`. Built by
    /// `dualFrame`, or `inputFrameDual` on the block's own complex.
    Eigen::MatrixXcd dualImages;
    /// The frame's rank \f$ r \f$.
    [[nodiscard]] int rank() const noexcept { return static_cast<int>(images.cols()); }
  };

  /// A block's marking together with its input coefficients: the cycles
  /// \f$ A, B \f$ of the torus the block carries, as closed walks of directed
  /// steps in host vertex ids (`Marking`), and the coefficients \f$ (a, b) \f$
  /// of the state the block represents, one per cycle —
  /// \f$ (1, \tau_{in}) \f$ for an input torus.
  ///
  /// The engine uses a marking in two ways. It scores the block's own state
  /// against the coefficients: the block's live surface is read as the
  /// simplicial qubit over the marking (`blockQubit`), and the transported
  /// periods of its holomorphic form over \f$ (A, B) \f$ must equal the
  /// coefficients as a point of \f$ \mathbb{CP}^1 \f$ (`ownStateResidualOn`).
  /// It also derives the block's frame at every read (`deriveFrame`), in which
  /// the zero mode of the whole cobordism is reported as coefficients
  /// (`readInputState`), as is the two-body transfer.
  ///
  /// Unlike a frame stated at attachment (`setInputFrame`), a marking is
  /// combinatorial data that every engine move preserves, so the frame it
  /// normalizes stays the block's live kernel as its lengths move. The marking
  /// itself is never scored: it fixes the coordinates, and by
  /// \f$ A \cdot B = +1 \f$ the orientation, that the state is read in.
  struct BlockMarking {
    /// The cycles, each one closed walk starting at `baseVertex`.
    Marking cycles;
    /// \f$ (a, b, \dots) \f$: one coefficient per cycle, the state at the block.
    Eigen::VectorXcd coefficients;
    /// The common base point of the transported periods: the first vertex of
    /// the first cycle's walk that lies on every other cycle's walk (the
    /// `observables::SimplicialQubit::baseVertex` rule).
    std::uint64_t baseVertex{0};
    [[nodiscard]] int rank() const noexcept { return static_cast<int>(cycles.size()); }
  };

  /// A boundary block of the cobordism, either an input or an output. A block
  /// is not itself a complex: it stores the vertex set it occupies plus the
  /// target period vector its own \f$ L_k \f$ sub-complex must carry. The
  /// sub-complex is recovered on demand from `vertices` by
  /// `getRegion`, the ambient complex's top cells whose
  /// vertices all lie in the set.
  struct BoundaryBlock {
    /// Host vertex ids the block occupies.
    std::set<std::uint64_t> vertices;
    /// Target period vector the block's own \f$ L_k \f$ sub-complex must carry.
    std::vector<std::complex<double>> target;
    /// The fiber form of the target: a retained fiber on the block's degree-k
    /// cells with its Gram matrix, band eigenvalue, contour and certificate,
    /// alongside the period vector. Set on an input block by `setInputFiber`
    /// (pinned as boundary data by `pinInputFibers`); read on an output block
    /// by `readOutputFiber` after relaxation.
    std::optional<BoundaryFiber> fiber;
    /// True when the block is an input surface of the host: seeded by the
    /// region form of `seedInputs` on a `fromSurfaces` host, its
    /// \f$ (d-1) \f$-faces are cells of the host that no top cell need cover,
    /// and the bulk is drawn onto them by bridges. A block seeded from a
    /// simplex is never a surface.
    bool surface{false};
    /// A surface block's own faces (sorted vertex-id tuples): the input
    /// surface's \f$ (d-1) \f$-simplices, recorded when the block is seeded
    /// and carried by the block thereafter, so that the block's own complex
    /// (`blockSurface`, `blockSurfaceWithGeometry`) is the surface whatever the
    /// host currently registers. The host registers a surface face only while a
    /// top cell covers it or a read has materialized it — a cone-out dent
    /// uncovers a face and the host's orphan prune then drops it — whereas the
    /// surface's vertices and edges survive every stage-1 move. The faces are
    /// therefore the one part of the surface the vertex set alone cannot
    /// recover. Empty for an ordinary block, whose own complex is the host's
    /// top cells inside its vertex set. Neither pinned nor scored.
    std::vector<std::vector<std::uint64_t>> faces;
    /// The block's frame for the two-body transfer, set by `setInputFrame` on
    /// the cells of its attached fiber. The engine holds it constant; the
    /// block's own residual keeps it valid as the lengths move. Absent on an
    /// unframed block; when either input block lacks one, the transfer is read
    /// in identity frames on the cells. Ignored on a block that carries a
    /// `marking`, where the frame is derived live instead.
    std::optional<BlockFrame> frame;
    /// The block's marking with its input coefficients, set by
    /// `setInputMarking`. Absent on an unmarked block.
    std::optional<BlockMarking> marking;
  };

  /// An ordered, explicit period constraint, evaluated by
  /// `EigenstateSynthesis::residualForPeriods`. No component permutation is
  /// permitted: `target[i]` belongs to `holes[i]`.
  struct RegisterConstraint {
    /// Identity; re-declaring this name replaces the constraint.
    std::string name;
    /// Cochain degree the periods are read at.
    int degree{1};
    /// The cycles the periods are taken over, as vertex-id tuples.
    std::vector<std::vector<std::uint64_t>> holes;
    /// One target period per hole, in the same order.
    std::vector<std::complex<double>> target;
  };





  /// Whether a move may extend a boundary the node holds fixed. Default false.
  ///
  /// A cone-in over a boundary facet buries that facet and exposes the new
  /// cell's remaining facets; a cone-out removes a cell and exposes all of its
  /// facets. Either gives \f$ \partial W \f$ faces it did not have, changing
  /// which cobordism is being optimized. A refused candidate is simply not a
  /// member of the configuration space, as for `dualComplexValid`: nothing is
  /// clamped and nothing is rolled back.
  ///
  /// The gate applies only where a boundary is declared fixed
  /// (`hasFixedBoundary`) — a surface input or output block whose own triangles
  /// form a component of \f$ \partial W \f$. A node that declares none has its
  /// cones unrestricted whatever this is set to. A pinned region is not such a
  /// declaration: pinning constrains the geometry and does not veto a topology
  /// change (`declarePinnedRegion`).
  ///
  /// The gate protects both the boundary's facet set and its geometry, for
  /// every move kind rather than the cone kinds alone: a disposition flip on an
  /// edge of a boundary facet leaves the facet set untouched while changing the
  /// metric of \f$ \partial W \f$ underneath it. Since the boundary carries the
  /// input state, a trial that changes its geometry is not a cobordism of the
  /// declared states.
  void setBoundaryMayExtend(bool allowed) noexcept { boundaryMayExtend_ = allowed; }
  /// Whether a move may extend a boundary the node holds fixed.
  [[nodiscard]] bool boundaryMayExtend() const noexcept { return boundaryMayExtend_; }
  /// Whether this node declares a boundary it holds fixed: any surface input
  /// or output block, whose own triangles are a component of the boundary. A
  /// pinned region is not one (see `setBoundaryMayExtend`).
  [[nodiscard]] bool hasFixedBoundary() const noexcept;

  /// \param host              Bare host complex the register is grown from.
  /// \param inputTargets      One target period vector per input boundary block.
  /// \param outputTargets     One target period vector per output boundary
  ///   block, so that \f$ \partial W = \text{inputs} \sqcup \text{outputs} \f$:
  ///   a merge has one output, a 2→2 recombination has two. Each output, like
  ///   each input, is an emergent boundary sub-complex carrying its target.
  ///   Target-conditioned modes score it through `rU`; `JointStationarity`
  ///   keeps it only as readout metadata. The bulk routes which input
  ///   constituent reaches which output. An empty list pins nothing
  ///   downstream: `rU` then sums only the input blocks, and whatever the whole
  ///   complex comes to carry is read afterwards.
  /// \param degrees           Cochain degrees \f$ k \f$ the register is read at.
  /// \param gamma             Weight of the `rU` term in the `Legacy` objective.
  /// \param seed              RNG seed; the run is reproducible given it.
  /// \param precone           Number of gated cone-in moves used to pre-grow the
  ///   host before any optimization, giving surgery room to act. Each cone-in
  ///   adds one top cell on a fresh apex over a random facet and is accepted
  ///   only through the `dualComplexValid` gate (`preconeCells`). On the
  ///   single-\f$ \Delta^4 \f$ seed this enlarges the 4-ball. 0 leaves the host
  ///   untouched.
  /// \param shouldProposeDispositions  Whether stage 1 offers the causal
  ///   disposition moves.
  /// \param preconeTimelike   Draw every precone cone-in with the timelike
  ///   disposition (apex edges \f$ \ell^2 = -1 \f$).
  /// \param preconeAlternate  Alternate the precone cone-ins timelike and
  ///   spacelike, for balanced causal content at one uniform edge-length
  ///   magnitude. Takes precedence over \p preconeTimelike. With both false the
  ///   precone is all-spacelike.
  /// \param balancedEdgeWiring  Wire new edges to balance vertex degree.
  /// \param singularValueRatio  Replace the whole-complex term of `rU` — both
  ///   the single-output period residual and its `nearKernelResidual`
  ///   continuation — with the scale-invariant singular-value half-sum ratio
  ///   (`singularValueHalfSumRatio`). The input-block residuals still anchor
  ///   the input states.
  /// \param einsteinHilbert   Keep the Regge term selected by the objective
  ///   mode. False removes it: `JointStationarity` becomes Hodge entropy
  ///   stationarity alone, `MediatedCorrespondence` becomes `rU`, and `Legacy`
  ///   becomes `gamma * rU`. Stage 2 differentiates whatever scalar objective
  ///   remains.
  /// \param realSquaredLengthsOnly  Restrict stage 2 to the real
  ///   \f$ \ell^2 \f$ locus, for fixed-signature residual-only runs. False
  ///   keeps the complexified relaxation.
  /// \param metricSource      Metric source used to assemble the Hodge
  ///   Laplacian.
  MultiCobordism(
      std::shared_ptr<Spacetime> host,
      const std::vector<std::vector<std::complex<double>>> &inputTargets,
      const std::vector<std::vector<std::complex<double>>> &outputTargets,
      const std::vector<int> &degrees = {3}, double gamma = 1.0,
      std::uint64_t seed = 0, int precone = 0,
      bool shouldProposeDispositions = true, bool preconeTimelike = false,
      bool preconeAlternate = false,
                 bool balancedEdgeWiring = false,
                 bool singularValueRatio = false,
                 bool einsteinHilbert = true,
                 bool realSquaredLengthsOnly = false,
                 HodgeLaplacian::MetricSource metricSource =
                     HodgeLaplacian::defaultMetricSource());

  /// Where every Hodge operator this node scores, relaxes and reads takes its
  /// metric from, the operators of the post-hoc analysis pass
  /// (`runRecursiveAnalysis`) included. Defaults to the process-wide
  /// `HodgeLaplacian::defaultMetricSource()` read at construction, the
  /// chain-level Whitney pencil unless changed, so that the node, the static
  /// readouts, the observables and checkpoint replay agree. Under
  /// `WhitneyPencil` the operator is \f$ h_k(s,U) \f$ of the complex squared
  /// edge lengths and the edge-phase links, at every degree; `DiagonalWeights`
  /// is the per-simplex diagonal metric, selected by name.
  [[nodiscard]] HodgeLaplacian::MetricSource metricSource() const noexcept {
    return metricSource_;
  }

  /// Configuration-space admissibility of a geometry under the Whitney pencil:
  /// the closure of the Kontsevich–Segal allowable domain, i.e. margin
  /// \f$ \ge 0 \f$. The real Lorentzian boundary, margin exactly zero, is
  /// admitted and certified as the boundary. This is not a clamp, back-off or
  /// penalty: a proposal outside the domain is not a member of the
  /// configuration space, as a non-manifold proposal is not. Always true under
  /// `DiagonalWeights`.
  [[nodiscard]] bool geometryAdmissible(const std::shared_ptr<Spacetime> &spacetime) const;

  /// Whether the Regge term is included in the objective.
  [[nodiscard]] bool einsteinHilbertEnabled() const noexcept {
    return einsteinHilbert_;
  }
  /// Whether stage 2 is restricted to the real \f$ \ell^2 \f$ locus.
  [[nodiscard]] bool realSquaredLengthsOnly() const noexcept {
    return realSquaredLengthsOnly_;
  }

  /// Declare an ordered exact-period constraint, replacing a constraint with
  /// the same name. Every hole must contain degree + 2 distinct vertices and
  /// the hole count must equal the target width.
  /// @throws std::invalid_argument on malformed input.
  void declareRegisterConstraint(RegisterConstraint constraint);

  /// The declared register constraints, in declaration order.
  [[nodiscard]] const std::vector<RegisterConstraint> &registerConstraints()
      const noexcept {
    return registerConstraints_;
  }

  /// Remove every explicit register constraint. Emergent input/output targets
  /// are unaffected.
  void clearRegisterConstraints();

  /// `EigenstateSynthesis::relaxFixedBoundaryEigenstate` on the node's live
  /// spacetime and metric source. Mutates the live spacetime in place.
  [[nodiscard]] FixedBoundaryEigenstateResult relaxFixedBoundaryEigenstate(
      int degree,
      std::vector<std::vector<std::uint64_t>> supportCells,
      std::vector<std::complex<double>> target, double epsilon = 1e-10,
      int restarts = 64, int maxGrowth = 4, std::uint64_t seed = 0,
      int maxIterations = 200);

  /// `EigenstateSynthesis::relaxBoundaryStatePairs` on the node's live spacetime,
  /// metric source and declared pinned regions. Mutates the live spacetime in
  /// place.
  [[nodiscard]] BoundaryStateTransferResult relaxBoundaryStatePairs(
      int degree, std::string inputRegionName,
      std::vector<std::vector<std::uint64_t>> inputCells,
      std::vector<std::vector<std::complex<double>>> inputStates,
      std::string outputRegionName,
      std::vector<std::vector<std::uint64_t>> outputCells,
      std::vector<std::vector<std::complex<double>>> outputStates,
      bool commonEigenvalue = true, double epsilon = 1e-10,
      double boundaryEpsilon = 1e-10, int restarts = 64,
      int maxGrowth = 4, std::uint64_t seed = 0,
      int maxIterations = 200);

  /// `EigenstateSynthesis::relaxWholeComplexReadoutTargets` on the node's live
  /// spacetime, metric source and declared pinned regions. Mutates the live
  /// spacetime in place.
  [[nodiscard]] WholeComplexReadoutResult relaxWholeComplexReadoutTargets(
      int degree, std::string regionAName,
      std::vector<std::vector<std::uint64_t>> cellsA,
      std::vector<std::vector<std::complex<double>>> statesA,
      std::string regionBName,
      std::vector<std::vector<std::uint64_t>> cellsB,
      std::vector<std::vector<std::complex<double>>> statesB,
      std::vector<ReadoutChain> readouts,
      std::vector<std::vector<std::complex<double>>> targets,
      bool commonEigenvalue = true, double epsilon = 1e-10,
      double boundaryEpsilon = 1e-10, int restarts = 64,
      int maxGrowth = 4, std::uint64_t seed = 0,
      int maxIterations = 200);

  /// The space a two-body target is scored against. Each name is the space the
  /// reading takes the state from.
  ///
  /// `Transfer` is \f$(Z_A^\vee)^T \tilde A_1 Z_B\f$: the whole complex's
  /// degree-1 operator read as the coupling block between the two boundary
  /// frames. It is \f$2 \times 2\f$ here and factorizes across the boundaries,
  /// which is what lets it carry an entangled target.
  ///
  /// `Bulk` is \f$\ker L_1(W-\partial W)\f$, the Laplacian on interior cells
  /// with the boundary removed, read through a Choi frame
  /// (`geometricOperator`).
  ///
  /// `Whole` is \f$\ker L_1(W)\f$, the boundary included, read through the
  /// blocks' markings. Its rank is \f$b_1(W)\f$.
  ///
  /// `Operator` is the low-level transfer between two paired direct-sum frames.
  /// A target is meaningful only when expressed in those same direct-sum
  /// frames; equal dimensions alone do not identify it with a tensor-product
  /// two-qubit operator.
  enum class ReadoutMode { Transfer, Bulk, Whole, Operator };

  /// How the whole-complex reading pairs the harmonic columns against the input
  /// blocks.
  ///
  /// `Periods` integrates each harmonic column over the marked cycles
  /// (`Connection::transportedPeriod`). That is how a state is defined on a
  /// boundary torus, where the Hodge star is an endomorphism of \f$ H^1 \f$ and
  /// a modulus is the period ratio of the holomorphic line it selects. On the
  /// three-dimensional bulk the same reading is purely topological: the star
  /// there maps 1-forms to 2-forms, so no such line exists, changing the metric
  /// moves the harmonic representative by a coboundary, and a coboundary's
  /// period around a closed cycle telescopes to zero. A bulk run scored on
  /// periods therefore cannot improve.
  ///
  /// `Gram` contracts through the chain metric instead, \f$ f_c^T M_1 Z_a \f$
  /// with \f$ f_c \f$ the columns of the block's live frame on the host's
  /// edges: the transpose pairing that the harmonic Gram matrix
  /// \f$ Z^T M_1 Z \f$ already uses. The metric content is present in both; the
  /// period pairing is the contraction that annihilates it.
  ///
  /// The observation matrix has the same shape either way, so the fit, the
  /// period-frame normalization and the projective leak downstream are
  /// unchanged. `Periods` is the default.
  enum class WholePairing { Periods, Gram };
  /// Select how the whole-complex reading pairs harmonic columns.
  void setWholePairing(WholePairing pairing);
  /// The whole-complex pairing in force.
  [[nodiscard]] WholePairing wholePairing() const noexcept { return wholePairing_; }

  /// The readings summed into the two-body residual. A set rather than a single
  /// choice, so a run can be scored under more than one reading. Repeated
  /// values are normalized to their first occurrence.
  /// @throws std::invalid_argument if \p modes is empty.
  void setReadoutModes(std::vector<ReadoutMode> modes);
  /// The readings summed into the two-body residual.
  [[nodiscard]] const std::vector<ReadoutMode> &readoutModes() const noexcept {
    return readoutModes_;
  }
  /// Read a square operator from the target-free live bulk. \p frameCells is
  /// the ordered row-major Choi frame and must hold exactly
  /// \p stateDimension\f$^2\f$ interior edges; it may be empty only when the
  /// bulk itself has that many, in which case canonical bulk-cell order is
  /// used.
  ///
  /// Promotion succeeds only when the framed restriction of
  /// \f$\ker L_1(W-\partial W)\f$ has rank one. No target or constraint is
  /// read. The returned Choi vector is unit-normalized and phase-fixed; the
  /// operator uses the unitary scaling \f$\sqrt d\f$ and reports its unitarity
  /// error.
  [[nodiscard]] GeometricOperatorReadout geometricOperator(
      int stateDimension,
      std::vector<std::vector<std::uint64_t>> frameCells = {},
      double tol = 1e-9, bool metric = true) const;
  /// `geometricOperator` on the given complex rather than the live one. Stage 1
  /// prices each candidate on a complex rebuilt from a snapshot, so a live-only
  /// read would rank every move by the geometry it started from.
  [[nodiscard]] GeometricOperatorReadout geometricOperatorOn(
      const std::shared_ptr<Spacetime> &spacetime, int stateDimension,
      std::vector<std::vector<std::uint64_t>> frameCells = {},
      double tol = 1e-9, bool metric = true) const;
  /// Move-kind names. The four Pachner kinds alias the names their move classes
  /// own (`AddMove::kMoveType` and siblings), so there is one definition per
  /// kind rather than one per dispatch site.
  static constexpr const char *kAddMove = ::tessera::spacetime::AddMove::kMoveType;
  static constexpr const char *kRemoveMove =
      ::tessera::spacetime::RemoveMove::kMoveType;
  static constexpr const char *kFlipMove =
      ::tessera::spacetime::FlipMove::kMoveType;
  static constexpr const char *kIFlipMove =
      ::tessera::spacetime::IFlipMove::kMoveType;
  /// Surgical kinds: `SurgicalCone` operations reached only through this draw.
  static constexpr const char *kConeOut = "cone_out";
  static constexpr const char *kConeIn = "cone_in";
  static constexpr const char *kNoop = "noop";
  /// The two causal-disposition moves.
  static constexpr const char *kConeInTimelike = "cone_in_timelike";
  static constexpr const char *kFlipDisposition = "flip_disposition";
  /// One candidate move: its kind, and the payload naming where or how to act.
  /// Returned by `enumerateMoveSpecifications`.
  using MoveSpec = std::pair<std::string, std::vector<std::uint64_t>>;

  /// The four Pachner kinds addressed by site: the same moves, but the payload
  /// names where to act instead of seeding a draw. They carry distinct names
  /// because a bare vertex id and a bare seed are both one integer.
  static constexpr const char *kAddAt = "add_at";
  static constexpr const char *kRemoveAt = "remove_at";
  static constexpr const char *kFlipAt = "flip_at";
  static constexpr const char *kIFlipAt = "iflip_at";

  /// The depth ladder one stage-1 update walks, in the order it walks it:
  /// ascending 1..`maxLookahead` by default, or descending
  /// `combinatorialBreadth`..1 when a breadth is named. Returned as one list so
  /// the schedule can be read without driving a complex.
  [[nodiscard]] static std::vector<int> depthSchedule(int maxLookahead,
                                                      int combinatorialBreadth);

  /// Every candidate move on \p spacetime, rather than a sample of them.
  /// Unlike `drawRandomMoveSpecification`, which picks a kind uniformly and
  /// only then a site — and draws three of the four Pachner sites from
  /// `Spacetime::rng`, which no seed controls — this walk is complete and
  /// reproducible.
  ///
  /// The Pachner kinds come back as the `*_at` kinds above; the cone and
  /// disposition kinds already name their sites and come back unchanged. Each
  /// returned spec is a candidate to score, not a promise that it applies.
  /// \p withSurgery false leaves out the cone kinds (`shouldProposeSurgery`).
  [[nodiscard]] static std::vector<MoveSpec> enumerateMoveSpecifications(
      const std::shared_ptr<Spacetime> &spacetime, bool withDispositions = false,
      bool withSurgery = true);

  /// A `kFlipDisposition` payload names one edge by its two endpoint vertex ids.
  static constexpr std::size_t kEdgeEndpointCount = 2;

  /// True when \p payload names an edge, i.e. holds exactly two endpoint vertex
  /// ids.
  [[nodiscard]] static bool payloadNamesAnEdge(
      const std::vector<std::uint64_t> &payload) {
    return payload.size() == kEdgeEndpointCount;
  }

  /// Whether the stage-1 move draw also proposes causal dispositions: a
  /// timelike cone-in, and a disposition flip on an existing edge. Both are
  /// ordinary candidate moves — drawn at random, scored by `deltaF`, committed
  /// only when they lower \f$F\f$ — so nothing prescribes causal structure.
  /// Defaults to true, because on the single-\f$\Delta^4\f$ seed the timelike
  /// cone-ins are the only moves that lower \f$F\f$ and the only ones giving
  /// \f$\operatorname{Im} S \neq 0\f$. False recovers the six-move draw; stage
  /// 2 can still explore complex squared intervals.
  ///
  /// @note With these moves in the draw, \f$\|\nabla S\|^2\f$ can grow enormous
  ///   while the action itself stays finite. The cause is exact degeneracy of
  ///   tetrahedral facets. The circumcentre solves
  ///   \f$G\beta = \tfrac12\operatorname{diag} G\f$ and is undefined when
  ///   \f$\det G = 0\f$; the Lorentzian \f$G\f$ is indefinite, so a facet whose
  ///   span is tangent to the light cone — a null 3-face of zero 3-volume with
  ///   every edge at \f$|\ell^2| = 1\f$ — is reachable, and quantizing every
  ///   \f$\ell^2\f$ to \f$\pm1\f$ lands on that locus exactly. Of the
  ///   \f$2^6\f$ sign patterns of a tetrahedron's edges, the 12 degenerate ones
  ///   all have three timelike edges; triangles and pentatopes are never
  ///   degenerate at \f$\pm1\f$. Degenerate facets poison the discrete exterior
  ///   calculus dual recursion, since `Simplex::circumFromGram` divides by
  ///   \f$\det G\f$ under an exact-zero guard and a rounding-level residue
  ///   gives \f$\beta \sim 1/\det G\f$. Generic edge lengths essentially never
  ///   hit \f$\det G = 0\f$.
  [[nodiscard]] bool shouldProposeDispositions() const {
    return shouldProposeDispositions_;
  }

  /// Whether the stage-1 move draw and the exhaustive enumeration offer the
  /// surgical kinds: cone-out, cone-in and the timelike cone-in. Off, stage 1
  /// walks the four Pachner kinds alone (and the disposition flip, when
  /// dispositions are proposed): the moves that keep the complex the manifold
  /// it is, with the boundary it has. Defaults to true.
  [[nodiscard]] bool shouldProposeSurgery() const {
    return shouldProposeSurgery_;
  }
  void setShouldProposeSurgery(bool shouldProposeSurgery) {
    shouldProposeSurgery_ = shouldProposeSurgery;
  }

  // ---- module-level helpers (static) ----



  // ---- objective ----
  /// The per-block register residual summed over `registerDegrees_`: the sum of
  /// the block residual over every input and output block, plus the near-kernel
  /// residual per degree (`nearKernelResidual`).
  ///
  /// The period residual alone is a step function in the topology: before the
  /// first register opens it sits at its zero-filled-leak floor, so \f$F\f$
  /// carries no register-seeking gradient until a hole exists. The near-kernel
  /// term continues it below that threshold. Once \f$b_k\f$ reaches the
  /// expected register count the smallest singular values are exactly zero, the
  /// near-kernel term saturates at 0, and the period residual takes over
  /// scoring what the registers carry.
  [[nodiscard]] double rU(const std::shared_ptr<Spacetime> &st) const;

  /// The pre-topological register signal at one degree: the sum of the
  /// \p expectedRegisterCount smallest squared singular values of the metric
  /// \f$L_k\f$ (the signed operator under the process weight convention),
  /// normalized scale-free.
  ///
  /// Being metric, the term feels the continuously-valued edge lengths and
  /// descends along two channels: stage-1 surgery, where a hole zeroes the
  /// corresponding singular values exactly, and stage-2 tuning of the causal
  /// structure toward null directions, which opens near-kernels with no holes.
  ///
  /// Singular values are used rather than eigenvalues because the signed
  /// operator is non-normal: they are the eigenvalue magnitudes of the
  /// Hermitian \f$L^\dagger L\f$, share its kernel, and are smooth where
  /// \f$|\lambda|\f$ is not.
  ///
  /// Normalization is \f$n\,(\sum_{m\ \text{smallest}} \sigma^2) /
  /// (\sum \sigma^2)\f$, so a generic mode contributes about 1 and the range is
  /// \f$[0, m]\f$. The ratio is homogeneous of degree 0 in \f$\ell^2\f$, which
  /// closes the conformal-inflation descent channel a raw spectral sum would
  /// open. \f$m\f$ is `expectedRegisterCount`, making the term a soft
  /// relaxation of \f$b_k \ge m\f$; missing dimensions (\f$n < m\f$) count 1
  /// each.
  [[nodiscard]] static double nearKernelResidual(
      const std::shared_ptr<Spacetime> &st, int registerDegree,
      std::size_t expectedRegisterCount,
      HodgeLaplacian::MetricSource metricSource = HodgeLaplacian::defaultMetricSource());

  /// Exact complex gradient of `nearKernelResidual` with respect to each edge's
  /// \f$\ell^2\f$, in `ChainComplex` 1-cell order. It is the only part of `rU`
  /// with a derivative before a register exists: the period-gap terms sit at
  /// their constant full leak until holes open.
  ///
  /// The convention is \f$g = \partial r/\partial(\operatorname{Re}\ell^2)
  /// - i\,\partial r/\partial(\operatorname{Im}\ell^2)\f$, so the directional
  /// derivatives are \f$\operatorname{Re}(g)\f$ and
  /// \f$-\operatorname{Im}(g)\f$, and \f$\bar g\f$ is the steepest-ascent
  /// direction.
  ///
  /// With \f$H = L_k^\dagger L_k\f$, whose eigenvalues are the \f$\sigma^2\f$,
  /// \f$\partial\sigma_i^2 = w_i^H(\partial L^\dagger L + L^\dagger\partial L)w_i\f$
  /// for the normalized eigenvector \f$w_i\f$ and
  /// \f$\partial(\sum\sigma^2) = \partial\operatorname{tr}(H)\f$, combined by
  /// the quotient rule and scaled by \f$n\f$. Scale invariance gives the Euler
  /// identity \f$\sum \ell^2 g = 0\f$ in both parts.
  ///
  /// Exactly zero where the value is constant: no \f$k\f$-cells, or an
  /// identically-zero operator. Where two singular values coincide at the
  /// \f$m\f$-th place the selection is discontinuous and the functional
  /// non-smooth; this is the derivative away from such a tie, and the line
  /// search arbitrates at one.
  [[nodiscard]] static std::vector<std::complex<double>> nearKernelResidualGradient(
      const std::shared_ptr<Spacetime> &st, int registerDegree,
      std::size_t expectedRegisterCount,
      HodgeLaplacian::MetricSource metricSource = HodgeLaplacian::defaultMetricSource());

  /// The scale-invariant spectral-shape term that the `singularValueRatio` mode
  /// uses as the whole-complex contribution to `rU`, in place of both the
  /// single-output period residual and `nearKernelResidual`: the ratio of the
  /// sum of the lower half of the singular values of the metric \f$L_k\f$ to
  /// the sum of the upper half. With \f$n\f$ values descending and
  /// \f$h = \lfloor n/2 \rfloor\f$ it is
  /// \f$(\sigma_{n-h+1} + \dots + \sigma_n)/(\sigma_1 + \dots + \sigma_h)\f$;
  /// an odd \f$n\f$ leaves the median out of both halves.
  ///
  /// Each lower-half value is bounded by its upper-half counterpart, so the
  /// ratio lies in \f$[0, 1]\f$, and a uniform rescale of the geometry cancels,
  /// leaving no conformal-inflation channel. The term rewards a collapsing
  /// lower half of the spectrum and prescribes no target.
  ///
  /// Returns 1 when there are no \f$k\f$-cells, so an empty complex does not
  /// score as a collapsed spectrum, and 0 for \f$n = 1\f$ or an
  /// identically-zero \f$L_k\f$.
  [[nodiscard]] static double singularValueHalfSumRatio(
      const std::shared_ptr<Spacetime> &st, int registerDegree,
      HodgeLaplacian::MetricSource metricSource = HodgeLaplacian::defaultMetricSource());

  /// The number of registers the targets ask for: the largest component count
  /// over every input and output target vector (each component is carried by
  /// one register/hole, so a `[1, omega, omega^2]` target needs three).
  [[nodiscard]] std::size_t expectedRegisterCount() const;
  /// The injected objective's scalar, using the configured Regge and
  /// Hodge-entropy weights.
  [[nodiscard]] double objective() const;
  /// Sum of the normalized positive-operator Hodge entropies over the declared
  /// Hodge degrees — the degrees the entropy is taken at, not the register
  /// degrees. This is an observation: the joint objective minimizes the
  /// entropy's gradient norm, not its value.
  ///
  /// Reported unweighted. The per-degree weights balance stationarity residuals
  /// against each other in the objective; applying them to entropy values would
  /// report a number that is not any degree's entropy.
  [[nodiscard]] double hodgeEntropy() const;
  /// \f$\sum_k w_k\|\nabla_zS_{{\rm Hodge},k}\|^2\f$ over the declared Hodge
  /// degrees and their weights: the entropy half of the joint objective.
  ///
  /// Reads the same degrees and weights as the objective and accumulates in the
  /// term's order, so `hodgeEntropyWeight() * this` reproduces
  /// `ObjectiveTerms::hodgeStationarity` exactly.
  [[nodiscard]] double hodgeEntropyStationarity() const;
  /// Inject the functional this node descends. The engine calls through it and
  /// knows nothing about which objective it holds; an objective reads only
  /// `ObjectiveContext`. `objectiveName()` reports what is being optimized.
  /// @throws std::invalid_argument on a null objective, or on one whose
  ///   `minimumRegisterDegree()` exceeds a configured register degree.
  void setObjective(std::shared_ptr<CobordismObjective> objective);
  /// The injected functional. Never null: construction installs a default.
  [[nodiscard]] const std::shared_ptr<CobordismObjective> &objectiveSpec()
      const noexcept {
    return objectiveSpec_;
  }
  /// Inject an additional objective that holds a pinned region, alongside the
  /// bulk objective this node descends. With none supplied, the pinned region's
  /// objective is the bulk objective itself — the same instance, not a copy —
  /// and the run is bit-identical to a single-objective one.
  ///
  /// The region is not named here. The objective declares its own scope through
  /// `ObjectiveScope`, whose `RegionHandle` can only be obtained from
  /// `regionHandle`, so a mis-spelled region name cannot compile.
  ///
  /// Scoring is additive and the bulk objective keeps scoring everything,
  /// including the pinned interior, so boundary-interior edges contribute to
  /// both.
  /// @throws std::invalid_argument on a null objective.
  void setPinnedObjective(std::shared_ptr<CobordismObjective> objective);
  /// The additional pinned-region objective, or null where none is supplied.
  [[nodiscard]] const std::shared_ptr<CobordismObjective> &pinnedObjective()
      const noexcept {
    return pinnedObjectiveSpec_;
  }
  /// Drop the pinned-region objective, returning the node to a single
  /// objective scoring the whole cobordism.
  void clearPinnedObjective() noexcept { pinnedObjectiveSpec_.reset(); }

  /// One objective's decomposition, labelled by the objective that produced it
  /// and the region it was scored over. The record carries a contribution per
  /// objective rather than one summed record, so a reader can tell whether
  /// descent came from the bulk or from the pinned region.
  struct ObjectiveContribution {
    /// The objective's stable identifier.
    std::string objectiveName;
    /// The declared region, or empty for the whole cobordism.
    std::string regionName;
    /// That objective's terms over its own scope.
    ObjectiveTerms terms;
  };

  /// Every objective's contribution, in evaluation order: the bulk objective
  /// first, then the pinned-region objective where one is supplied. Summing the
  /// terms reproduces `objectiveTerms()` exactly.
  [[nodiscard]] std::vector<ObjectiveContribution> objectiveContributionsFor(
      const std::shared_ptr<Spacetime> &spacetime) const;
  /// `objectiveContributionsFor` on this node's own complex.
  [[nodiscard]] std::vector<ObjectiveContribution> objectiveContributions()
      const;

  /// The injected objective's stable identifier, as stamped on records.
  [[nodiscard]] std::string objectiveName() const;
  /// Whether the injected objective's value depends on prescribed boundary
  /// targets rather than on the geometry alone. A search policy that must stay
  /// unforced consults this rather than testing for an objective by name.
  [[nodiscard]] bool objectiveIsTargetConditioned() const;
  /// Select how the Hodge entropy treats the complex phase.
  void setHodgeEntropyPhaseMode(HodgeLaplacian::EntropyPhaseMode mode) noexcept {
    hodgeEntropyPhaseMode_ = mode;
  }
  /// The entropy phase mode in force.
  [[nodiscard]] HodgeLaplacian::EntropyPhaseMode hodgeEntropyPhaseMode() const
      noexcept {
    return hodgeEntropyPhaseMode_;
  }
  /// Declare the weight on the Hodge entropy stationarity term.
  void setHodgeEntropyWeight(double weight);
  /// The Hodge entropy weight.
  [[nodiscard]] double hodgeEntropyWeight() const noexcept {
    return hodgeEntropyWeight_;
  }

  /// Declare the weight on the connection-entropy stationarity term, the only
  /// term with a gradient in the connection phase. Zero by default, so a node
  /// acquires phase dynamics only on request: the Hodge-entropy term sees
  /// \f$ \varphi \f$ through \f$ h_k(z,U) \f$ but is differentiated in
  /// \f$ z \f$ alone, so at zero weight the phase is a declared field that no
  /// update moves.
  void setConnectionEntropyWeight(double weight);
  /// The connection-entropy stationarity weight.
  [[nodiscard]] double connectionEntropyWeight() const noexcept {
    return connectionEntropyWeight_;
  }

  /// Declare the Laplacian degrees the Hodge entropy term is summed over, and
  /// optionally a weight per degree.
  ///
  /// These are the degrees \f$k\f$ whose \f$L_k\f$ the entropy is taken of.
  /// They are configured here and nowhere else; the register degrees, which say
  /// where a register is constructed, never supply them, not even as a
  /// fallback. The default is \f$\{0\}\f$.
  ///
  /// An empty \p weights means uniform \f$1\f$. A non-empty one must match
  /// \p degrees in length.
  ///
  /// @throws std::invalid_argument on an empty degree list, a negative degree,
  ///   a repeated degree, or a weight list whose length differs from the
  ///   degree list's.
  void setHodgeDegrees(std::vector<int> degrees,
                       std::vector<double> weights = {});
  /// The declared Hodge degrees, in declaration order.
  [[nodiscard]] const std::vector<int> &hodgeDegrees() const noexcept {
    return hodgeDegrees_;
  }
  /// The declared per-degree weights, or empty for uniform.
  [[nodiscard]] const std::vector<double> &hodgeDegreeWeights() const noexcept {
    return hodgeDegreeWeights_;
  }

  /// The Hodge stationarity term broken down by declared degree, so a reader
  /// can tell which degree the descent came from. Empty for an objective with
  /// no Hodge term.
  [[nodiscard]] std::vector<HodgeDegreeContribution>
  hodgeDegreeContributionsFor(
      const std::shared_ptr<Spacetime> &spacetime) const;
  /// `hodgeDegreeContributionsFor` on this node's own complex.
  [[nodiscard]] std::vector<HodgeDegreeContribution> hodgeDegreeContributions()
      const;

  /// Declare the weight on the Regge term.
  void setReggeWeight(double weight);
  /// The Regge weight.
  [[nodiscard]] double reggeWeight() const noexcept { return reggeWeight_; }
  /// Weight on each input block's residual in `rU`; the output and whole-complex
  /// terms keep weight 1. Raising it makes the optimizer prioritize keeping the
  /// input states represented rather than only driving the whole to the output.
  /// Default 1.
  void setInputResidualWeight(double weight) { inputResidualWeight_ = weight; }

  // ---- the two stages + boundary-block construction ----
  /// Seed one input block per seed vertex; the region is the seed's cell
  /// neighbourhood, tagged with its target. The block is not grown here —
  /// `runStage1`'s `growBlockRegions` grows it under the objective.
  void seedInputs(const std::vector<std::uint64_t> &seeds);
  /// Seed one input block per explicit vertex region: the surface inputs of a
  /// `fromSurfaces` host. A boundary surface has no top cell for a seed
  /// vertex's cell neighbourhood to find, so its vertex set is the block, given
  /// as `SurfaceSeed::vertexIds` supplies it. Marks each block
  /// `BoundaryBlock::surface`. The block's target is the constructor's
  /// `inputTargets[i]`, as for the seed-vertex form; the state itself arrives
  /// as a fiber later.
  /// @throws std::invalid_argument on an empty region or a vertex absent from the host.
  void seedInputs(const std::vector<std::vector<std::uint64_t>> &regions);
  /// Seed one output block per seed vertex (see `seedInputs`).
  void seedOutputs(const std::vector<std::uint64_t> &seeds);

  // ---- fiber-form boundary targets ----

  /// Score blocks that carry a fiber-form target by the fiber residual instead
  /// of the period residual: the block's own sub-complex is read on the
  /// chain-level Whitney pencil, the band of the fiber's contour (or, when the
  /// fiber names none, the lowest band above the flat zero mode) is restricted
  /// to the fiber's cells, and the target images are least-squares fitted in
  /// that band's images, \f$ \min_C\|Z_T C-\Psi\|_F^2/\|\Psi\|_F^2 \f$. The
  /// coefficients of the read come only from the block's lengths and connection
  /// values; nothing is pinned. Folded into `rU`, so stage 1 scores candidate
  /// moves by it and stage 2 descends it through the same numerical
  /// register-residual path the period targets use. Off by default.
  void useFiberResiduals(bool enabled) { useFiberResiduals_ = enabled; }
  /// Whether fiber residuals are scored in place of period residuals.
  [[nodiscard]] bool usesFiberResiduals() const noexcept { return useFiberResiduals_; }
  /// Also score a marked block by the leak of its input state in the zero mode
  /// of the whole cobordism (`inputStateResidualOn`), added to the block's own
  /// residual under `useFiberResiduals` and carried at the same
  /// `inputResidualWeight`.
  ///
  /// The two residuals answer different questions and can move in opposite
  /// directions. `ownStateResidualOn` builds the Laplacian of one torus in
  /// isolation and asks whether that torus, judged alone, still represents the
  /// state it was handed. `inputStateResidualOn` builds the Laplacian of the
  /// entire cobordism, takes its zero mode, and asks how much of the input
  /// coefficients fails to lie in it. Scoring only the first leaves the
  /// relationship to the rest of the complex unconstrained, and the leak can
  /// rise as the two-body residual falls.
  ///
  /// A held boundary forces the issue: a block whose edges cannot move cannot
  /// change shape, so its own residual is identically zero and the input weight
  /// multiplies nothing. The leak is a term the relaxation trades against, not
  /// a constraint; every coordinate still varies. Off by default.
  void scoreWholeComplexLeak(bool enabled) { scoreWholeComplexLeak_ = enabled; }
  /// Whether the whole-complex leak is scored.
  [[nodiscard]] bool scoresWholeComplexLeak() const noexcept { return scoreWholeComplexLeak_; }
  /// Whether stage 2 also descends the degree-0 link phases through the
  /// analytic fiber gradient. Off by default: flux lifts the flat zero mode and
  /// moves the band a default contour selects, so a caller enables it together
  /// with a fixed contour on its fiber targets.
  void setFiberPhaseDescent(bool enabled) { fiberPhaseDescent_ = enabled; }
  /// Whether stage 2 descends the degree-0 link phases.
  [[nodiscard]] bool fiberPhaseDescent() const noexcept { return fiberPhaseDescent_; }

  /// The band a block's fiber is read in: `ZeroMode` for a surface block
  /// (`BoundaryBlock::surface`), `AsStored` for an ordinary block.
  ///
  /// A surface block does not use the contour stored on its fiber, because that
  /// contour is a circle drawn on the spectrum of the complex the fiber was
  /// read from, which is not the spectrum of the block's own pencil. The state
  /// a surface block represents is the zero mode of its own Laplacian, so its
  /// contour is the harmonic contour of its own assembled pencil, recomputed at
  /// every read as its lengths move. In particular it is not band 1, the
  /// default of the whole-complex read, which sits above the zero mode and is
  /// not a state.
  [[nodiscard]] static FiberBand fiberBandFor(const BoundaryBlock &block) noexcept {
    return block.surface ? FiberBand::ZeroMode : FiberBand::AsStored;
  }
  /// The fiber residual of one block on \p spacetime (see `useFiberResiduals`):
  /// read on the block's own complex (`blockComplexWithGeometry` — a surface
  /// block's own surface, an ordinary block's sub-complex) in the band
  /// `fiberBandFor` names. Full leak (1) when the block has no own complex,
  /// when a fiber cell is outside it, or when the band is empty.
  /// @throws std::logic_error when the block carries no fiber target or the
  ///   node's metric source is not the Whitney pencil.
  [[nodiscard]] double fiberResidualForBoundaryBlock(
      const BoundaryBlock &boundaryBlock, const std::shared_ptr<Spacetime> &spacetime) const;
  /// The fiber residual of input block \p index on the live complex.
  [[nodiscard]] double fiberResidualForInputBlock(std::size_t index) const;
  /// A fiber-form target carried by the whole complex: the node's own pencil is
  /// read on the fiber's contour (by default the lowest band above the flat
  /// zero mode), restricted to the fiber's cells, and the target images are
  /// least-squares fitted there. This is how an input node represents a state:
  /// its whole complex, grown from a single simplex, carries the fiber on the
  /// seed's cells. Scored inside `rU` under `useFiberResiduals`.
  void setWholeComplexFiberTarget(BoundaryFiber fiber);
  /// The whole-complex fiber target, or empty when none is set.
  [[nodiscard]] const std::optional<BoundaryFiber> &wholeComplexFiberTarget() const noexcept {
    return wholeFiberTarget_;
  }
  /// The whole-complex fiber residual on the live complex.
  /// @throws std::logic_error without a whole-complex fiber target.
  [[nodiscard]] double wholeComplexFiberResidual() const;
  /// Read the fiber the whole complex carries on the target's cells and contour
  /// (the band of \p contour when given). This is what a downstream node is fed.
  /// @throws std::logic_error without a whole-complex fiber target.
  [[nodiscard]] BoundaryFiber readWholeComplexFiber(const chainhodge::Contour *contour = nullptr,
                                                    double kappa = 10.0) const;


  /// A block's own surface: its \f$ (d-1) \f$-faces and its edges inside its
  /// vertex set, as sorted vertex-id tuples. A face is inside the block when
  /// the block carries it (`BoundaryBlock::faces`), when the host registers it,
  /// or when it is a facet of a top cell with all \f$ d \f$ of its vertices in
  /// the block. All three are enumerated from vertex tuples rather than from
  /// the lazily materialized facet links, so the answer does not depend on what
  /// an earlier read materialized. For a surface block this is the input
  /// surface itself, as long as no chord — an edge or face inside the block
  /// that is not part of its surface — has been created.
  struct BlockSurface {
    /// The block's \f$ (d-1) \f$-faces, as sorted vertex-id tuples.
    std::vector<std::vector<std::uint64_t>> faces;
    /// The block's edges, as sorted vertex-id pairs.
    std::vector<std::vector<std::uint64_t>> edges;
  };
  /// The own surface of \p block within \p spacetime.
  [[nodiscard]] static BlockSurface blockSurface(const BoundaryBlock &block,
                                                const Spacetime &spacetime);
  /// A surface block's own complex with the host's geometry: `blockSurface`'s
  /// faces as the top cells of a fresh \f$ (d-1) \f$-dimensional `Spacetime`
  /// (`Spacetime::fromVertexTuples`, keeping the host's vertex ids), every edge
  /// carrying the host's current length and phase, matched by vertex pair.
  ///
  /// Its Laplacian is the block's own Laplacian: the torus's own triangles with
  /// the live lengths and nothing of the bulk, so its degree-1 zero mode is the
  /// torus's harmonic space. That is the band the block's fiber residual is
  /// read in (`fiberResidualForBoundaryBlock`) and the complex its gradient is
  /// taken on (`fiberModeAscent`). The host is read, never changed, and nothing
  /// is pinned. It differs from `blockSubcomplexWithGeometry`, which takes the
  /// host's top cells inside the vertex set — a surface block of a
  /// \f$ d \f$-complex contains none of those.
  ///
  /// Null when the block has no face inside its vertex set, or when the host no
  /// longer holds an edge of one of its faces. The block then scores the full
  /// leak, so no move that tears a surface survives the ΔF acceptance.
  [[nodiscard]] static std::shared_ptr<Spacetime> blockSurfaceWithGeometry(
      const BoundaryBlock &block, const std::shared_ptr<Spacetime> &spacetime);

  // ---- the derived frame and the state at a block ----

  /// The frame derived for a marked block from its live own complex.
  ///
  /// `frame.cells` are the block's own edges, in the own pencil's canonical
  /// order. `frame.images` is \f$ F = Z\,\Pi^{-1} \f$, with \f$ Z \f$ the
  /// images of the zero mode of the block's own covariant Whitney pencil
  /// (`PencilLayer::harmonicContour` on `blockSurfaceWithGeometry`) and
  /// \f$ \Pi_{ca} = \oint_c z_a \f$ its transported periods over the marking's
  /// cycles from the common base point
  /// (`chainhodge::Connection::transportedPeriod`), so column \f$ b \f$ has
  /// period \f$ \delta_{cb} \f$ over cycle \f$ c \f$. On the collar seed this
  /// is `observables::SimplicialQubit::periodFrame` of the input torus through
  /// the id map.
  ///
  /// `frame.dualImages` is
  /// \f$ F^\vee = \tilde Z\,(\tilde Z^T M_1^U F)^{-T} \f$, with
  /// \f$ \tilde Z \f$ the dual kernel's images (the same zero mode under the
  /// inverse links, `AssembledPencil::dual`) and \f$ M_1^U \f$ the dressed
  /// Whitney mass matrix of the block's own pencil, giving the `BlockFrame`
  /// normalization \f$ (F^\vee)^T M_1^U F = I \f$ (`PencilLayer::dualFrame`). That pairing
  /// is the gauge-covariant one; at zero phases the dual kernel is the kernel.
  ///
  /// An `obstruction` is reported when the block's own kernel does not have the
  /// marking's rank (a torn surface, or phases that are not a pure gauge), a
  /// marking edge is absent from the own complex, the periods are singular
  /// (cycles that do not span the block's homology), or the pairing is
  /// isotropic. Such a block reads the full leak.
  struct DerivedFrame {
    /// The derived frame on the block's own edges.
    BlockFrame frame;
    /// \f$ \Pi \f$, the transported periods over the marking's cycles
    /// (rank × rank).
    Eigen::MatrixXcd periods;
    /// Rank of the block's own zero mode.
    int kernelRank{0};
    /// Empty when the frame was derived; otherwise why it could not be.
    std::string obstruction;
    /// True when the frame was derived without obstruction.
    [[nodiscard]] bool derived() const noexcept { return obstruction.empty(); }
  };
  /// The state at a marked block: the coefficients of the whole complex's zero
  /// mode in the block's live frame, alongside the block's input coefficients
  /// and its residual.
  ///
  /// `coefficients` are the transported periods, over the marking's cycles from
  /// `baseVertex`, of the least-squares combination of the whole's zero mode
  /// that fits the block's target edge values — the input coefficients written
  /// through the live frame — on the block's edges. That combination is a
  /// kernel vector of the whole, so its periods are its coordinates in the
  /// frame. `residual` is the fit's leak, i.e. the output state read at the
  /// block; `rU` instead scores a marked block by `ownStateResidualOn` at
  /// `weight`. Under a pure gauge the target's base-point factor cancels the
  /// transport's, so the pair is gauge invariant.
  ///
  /// `harmonicRank` is the whole's zero-mode rank and `frameRank` the own
  /// kernel's. An obstructed read — no frame, no zero mode, or a target edge
  /// absent from the whole — names the obstruction and reads the full leak 1.0.
  struct InputStateRead {
    /// Index of the input block.
    std::size_t block{0};
    /// The block's declared input coefficients.
    Eigen::VectorXcd input;
    /// The whole complex's zero mode in the block's live frame.
    Eigen::VectorXcd coefficients;
    /// Leak of the fit; 1.0 when obstructed.
    double residual{std::numeric_limits<double>::quiet_NaN()};
    /// Weight this block's residual carries in `rU`.
    double weight{1.0};
    /// Rank of the whole complex's zero mode.
    int harmonicRank{0};
    /// Rank of the block's own kernel.
    int frameRank{0};
    /// Base point the periods were transported from.
    std::uint64_t baseVertex{0};
    /// Empty when the read succeeded; otherwise why it could not be made.
    std::string obstruction;
  };

  // ---- two-body cobordism map ----

  /// The two-body target of the interaction node: \f$ \chi \f$ on the pair of
  /// input frames (\f$ r_A\times r_B \f$). `choiDecomposed` selects what the
  /// node reports: the state \f$ \mathrm{vec}(T_{AB}) \f$ on the pair space
  /// when true, the operator \f$ T_{AB} \f$ when false. The fit residual is the
  /// projective Frobenius leak in either reading — vec is linear, so the two
  /// coincide — and the readings differ only in what is returned and certified.
  struct TwoBodyTarget {
    /// The target \f$ \chi \f$ on the pair of input frames.
    Eigen::MatrixXcd chi{};
    /// Report the Choi-decomposed state rather than the operator.
    bool choiDecomposed{true};
    /// Target for `ReadoutMode::Operator`, expressed in the paired direct-sum
    /// frame returned by `pairedFrameTransferOn`. This is not a
    /// tensor-product two-state vector: equal dimensions do not identify
    /// \f$V_A\oplus V_{A^*}\f$ with \f$V_A\otimes V_{A^*}\f$. Empty means no
    /// paired-frame target was declared, so the operator reading returns the
    /// full leak.
    Eigen::MatrixXcd twoStateVector{};
    /// Optional input coefficients for this read, flattened in input-block
    /// order and then cycle order. Empty uses the coefficients carried by the
    /// live markings. Cases set this so that changing the boundary metric and
    /// the input state is one atomic read of the shared bulk.
    Eigen::VectorXcd inputCoefficients{};
  };
  /// One input pair and the output the gate should produce for it: a boundary
  /// metric, a target, and optionally the input coefficients read by
  /// `ReadoutMode::Whole`.
  ///
  /// A bulk fitted to a single pair reproduces that pair and nothing else —
  /// one \f$2\times2\f$ transfer against one pair is six real constraints
  /// against some eighty free bulk coordinates. Several cases constrain one
  /// shared bulk, so a geometry satisfying all of them under the transfer
  /// reading is an operator. Every case shares one triangulation, one gluing
  /// and one bulk; only the boundary metric and the declared state differ, so
  /// moving between cases involves no re-gluing.
  struct TwoBodyCase {
    /// Squared lengths on the boundary edges, by endpoint pair. Only the
    /// boundary is listed: the bulk is what is being solved for and is shared.
    std::vector<std::pair<std::pair<std::uint64_t, std::uint64_t>,
                          std::complex<double>>> boundary;
    /// The target \f$ \chi \f$ for this case.
    Eigen::MatrixXcd chi{};
    /// Report the Choi-decomposed state rather than the operator.
    bool choiDecomposed{true};
    /// Target for `ReadoutMode::Operator`, expressed in the paired direct-sum
    /// frame. It is not a tensor-product two-state vector; empty means this
    /// case declares no paired-frame target.
    Eigen::MatrixXcd twoStateVector{};
    /// Optional input coefficients for this case, flattened in input-block
    /// order and then cycle order. Empty reads the live markings'
    /// coefficients.
    Eigen::VectorXcd inputCoefficients{};
  };
  /// The reading of the bulk between the two attached input frames.
  struct TwoBodyRead {
    bool choiDecomposed{true};
    /// \f$ T_{AB} = (Z_A^\vee)^T(\tilde A^U)_{AB}Z_B \f$: with the full frames on
    /// the two attached cell sets (the coupling block of the whole between
    /// them, \f$ |A| \times |B| \f$) when the blocks carry no frame, or in the
    /// blocks' frames (\f$ r_A \times r_B \f$, `BlockFrame`) when both do.
    Eigen::MatrixXcd transfer{};
    /// True when `transfer` was read in the blocks' frames (a derived frame
    /// from `setInputMarking` or a supplied one from `setInputFrame`, on
    /// both), false when in identity frames on the cells.
    bool inFrames{false};
    /// True when both frames were derived live from the blocks' markings;
    /// false with supplied or identity frames.
    bool derivedFrames{false};
    /// \f$ \mathrm{vec}(T_{AB}) \f$, column-major, the Choi-decomposed state.
    Eigen::VectorXcd choiState{};
    /// Singular values of \f$ T_{AB} \f$: the Schmidt spectrum of the state.
    std::vector<double> singularValues{};
    /// Numerical Schmidt rank (singular values above \f$ 10^{-10}\sigma_{\max} \f$);
    /// one is the product (quasi-free) case, two the XY flip-flop's.
    int schmidtRank{0};
    /// The reversal identity residual of the transfer.
    double reversalResidual{std::numeric_limits<double>::quiet_NaN()};
    /// The projective Frobenius leak of the target in the reading.
    double residual{std::numeric_limits<double>::quiet_NaN()};
    /// The fiber residual of each input block carrying an attached fiber: the
    /// leak of its state fiber in its own kernel. On a marked block this is a
    /// diagnostic; the scored residual is in `inputStates`.
    std::vector<double> inputFiberResiduals{};
    /// The state at every marked input block (`readInputState`, block order).
    std::vector<InputStateRead> inputStates{};
    /// The cells the transfer's rows and columns are placed on: the frames'
    /// cells when framed, the attached fiber cells otherwise.
    std::vector<std::vector<std::uint64_t>> cellsA{};
    std::vector<std::vector<std::uint64_t>> cellsB{};
  };
  /// Attach a piped input fiber to this complex's cells. \p cells are degree-k
  /// cells of the live complex, one per fiber row, in the attachment order; that
  /// order is the attachment permutation. The fiber's own cell ids are upstream
  /// ids and are replaced, and the block's region grows to contain the attached
  /// cells.
  /// @throws std::invalid_argument on a count mismatch, a
  ///   cell absent from the live complex, or an overlap with another attached
  ///   input fiber's cells.
  void attachInputFiber(std::size_t index, BoundaryFiber fiber,
                        std::vector<std::vector<std::uint64_t>> cells);
  /// Set the frame the two-body transfer is read in on input block \p index:
  /// \p images and \p dualImages on \p cells, which must be the block's
  /// attached fiber cells in the attachment order, since the rows of the
  /// transfer's operands are those cells. The caller states the frame at
  /// attachment — a torus's period frame carried through the collar's id map,
  /// with its dual from `inputFrameDual` — and the engine holds it constant
  /// thereafter.
  ///
  /// When both input blocks carry a frame, `readTwoBody`, `twoBodyResidual` and
  /// `twoBodyResidualGradientOn` read the transfer in the frames
  /// (\f$ r_A \times r_B \f$); the gradient differentiates the pencil operator
  /// only, the frames being constants. Re-attaching the block's fiber clears
  /// its frame, since the rows referred to the previous attachment.
  /// @throws std::out_of_range on the index
  /// @throws std::logic_error when the block
  ///   carries no attached fiber; std::invalid_argument, by name, when the
  ///   cells are not the fiber's attached cells in their order, when the
  ///   images or the dual images have another row count, when the two have
  ///   different column counts or none, or when an entry is not finite.
  void setInputFrame(std::size_t index, std::vector<std::vector<std::uint64_t>> cells,
                     Eigen::MatrixXcd images, Eigen::MatrixXcd dualImages);
  /// The frame of input block \p index, or none.
  /// @throws std::out_of_range
  [[nodiscard]] const std::optional<BlockFrame> &inputFrame(std::size_t index) const;
  /// Drop the frame of input block \p index: the transfer returns to identity
  /// frames on the cells.
  /// @throws std::out_of_range
  void clearInputFrame(std::size_t index);
  /// Set the marking of input block \p index together with the block's input
  /// coefficients: \p cycles in host vertex ids (`Marking`, the convention of
  /// `monodromy`), one coefficient per cycle. Each cycle is ordered into one
  /// closed walk and rotated to start at the common base point. The block must
  /// carry an attached degree-1 fiber (`attachInputFiber`).
  ///
  /// From then on the block's residual in `rU`, under `useFiberResiduals`, is
  /// `ownStateResidualOn` in place of `fiberResidualForBoundaryBlock`, its
  /// stage-2 direction is `ownStateResidualGradientOn`, the whole's zero mode
  /// is reported in the derived frame as the output state (`readInputState`),
  /// and the two-body transfer is read in the derived frame, so a supplied
  /// `frame` on the block is never read.
  /// @throws std::out_of_range on the index;
  /// @throws std::logic_error without an
  ///   attached fiber; std::invalid_argument by name, when the fiber is not
  ///   at degree 1, when there is no cycle, when the coefficient count is not
  ///   the cycle count, when a coefficient is not finite or all are zero,
  ///   when a step is a self-loop or not an edge of the live complex inside
  ///   the block's vertex set, when a cycle's steps do not form one closed
  ///   walk, or when the cycles share no vertex.
  void setInputMarking(std::size_t index, Marking cycles, std::vector<std::complex<double>> coefficients);
  /// The marking of input block \p index, or none.
  /// @throws std::out_of_range
  [[nodiscard]] const std::optional<BlockMarking> &inputMarking(std::size_t index) const;
  /// Drop the marking of input block \p index: its scoring and transfer
  /// return to what they were before `setInputMarking`.
  /// @throws std::out_of_range
  void clearInputMarking(std::size_t index);
  /// The live frame of a marked \p block on \p spacetime (`DerivedFrame`):
  /// read-only on the geometry.
  /// @throws std::logic_error when the block carries no marking.
  [[nodiscard]] static DerivedFrame deriveFrame(const BoundaryBlock &block,
                                               const std::shared_ptr<Spacetime> &spacetime);
  /// `deriveFrame` of input block \p index on the live complex.
  /// @throws std::out_of_range on the index; std::logic_error without a marking.
  [[nodiscard]] DerivedFrame deriveInputFrame(std::size_t index) const;
  /// The block residual of a marked block: the target edge values
  /// \f$ t = F\,(a, b)^T \f$ of the block's input coefficients through its live
  /// frame on \p spacetime, and the leak of \f$ t \f$ in the zero mode of the
  /// entire cobordism — bulk and boundary edges in one pencil, at the whole's
  /// harmonic contour — restricted to the block's edges (`fiberResidualOn`,
  /// one target per block). Full leak 1.0 when the frame cannot be derived or
  /// the whole refuses the read.
  /// @throws std::logic_error without a marking or off the Whitney pencil.
  [[nodiscard]] double inputStateResidualOn(const BoundaryBlock &block,
                                            const std::shared_ptr<Spacetime> &spacetime) const;
  /// `inputStateResidualOn` of input block \p index on the live complex.
  [[nodiscard]] double inputStateResidual(std::size_t index) const;
  /// The state at input block \p index: the coefficients of the whole's zero
  /// mode in the block's live frame, its input coefficients, and its residual.
  /// @throws std::out_of_range
  /// @throws std::logic_error without a marking or off the Whitney pencil.
  [[nodiscard]] InputStateRead readInputState(std::size_t index) const;
  /// The block's live surface read as a simplicial qubit over the block's
  /// marking: the surface of `blockSurfaceWithGeometry`, the marking's cycles
  /// as (edge index, sign) steps in the surface's edge order, and the
  /// orientation fixed by the marking. A `Spacetime` stores no orientation, so
  /// the read is taken in the one with \f$ A \cdot B = +1 \f$
  /// (`observables::SimplicialQubit::intersectionNumber`), under which the flat
  /// torus that seeded the block reads \f$ \tau_{in} \f$. The whole qubit is
  /// read rather than a kernel, because the state a torus represents on its own
  /// is its holomorphic form, which the kernel alone does not determine.
  /// @throws std::logic_error without a marking; std::invalid_argument when
  ///   the marking does not have two cycles or the surface refuses the
  ///   qubit's validation; std::runtime_error when the block has no surface,
  ///   a marking edge is not an edge of the live surface, or the qubit's
  ///   construction is refused (a branch that cannot be continued, a
  ///   degenerate complex structure).
  [[nodiscard]] static observables::SimplicialQubit blockQubit(const BoundaryBlock &block,
                                                                const std::shared_ptr<Spacetime> &spacetime);
  /// `blockQubit` of input block \p index on the live complex.
  /// @throws std::out_of_range
  [[nodiscard]] observables::SimplicialQubit blockQubit(std::size_t index) const;
  /// The block residual on \p spacetime: with
  /// \f$ (P_A, P_B) \f$ the transported periods of the holomorphic form of
  /// the block's own Laplacian on its live surface (`blockQubit`) over the
  /// marking's cycles and \f$ (a, b) \f$ the block's input coefficients,
  /// \f$ r = 1 - |\langle (a,b) | (P_A, P_B)\rangle|^2 / (\|(a,b)\|^2
  /// \|(P_A,P_B)\|^2) = 1 - |\langle\psi(\tau_{in})|\psi(\hat\tau)\rangle|^2 \f$
  /// for \f$ \psi(\tau) = (1,\tau)/\sqrt{1+|\tau|^2} \f$. It is zero exactly
  /// when the block's own Laplacian represents the input state, and invariant
  /// under a pure gauge, since both periods carry the base point's factor.
  /// Full leak 1.0 when the block has no surface or the read is refused, as for
  /// a torn surface or a degenerate geometry. This is what `rU` scores at
  /// `inputResidualWeight` for a marked block: the torus keeps representing its
  /// input state on its own, while the whole's zero mode is the output state.
  /// @throws std::logic_error without a marking.
  [[nodiscard]] double ownStateResidualOn(const BoundaryBlock &block,
                                          const std::shared_ptr<Spacetime> &spacetime) const;
  /// `ownStateResidualOn` of input block \p index on the live complex.
  [[nodiscard]] double ownStateResidual(std::size_t index) const;
  /// The block leak from the periods and the coefficients alone:
  /// \f$ 1 - |\langle c | p\rangle|^2/(\|c\|^2\|p\|^2) \f$ with
  /// \f$ p = (P_A, P_B) \f$ the raw periods over the given marking and
  /// \f$ c = (a, b) \f$. Returns 1.0 when \f$ p \f$ vanishes.
  [[nodiscard]] static double ownStateLeakOf(std::complex<double> periodA, std::complex<double> periodB,
                                             const Eigen::VectorXcd &coefficients);
  /// `PencilLayer::dualFrame` of \p images on input block \p index's own pencil — its own
  /// complex with the live lengths (`blockComplexWithGeometry`: a surface
  /// block's surface, an ordinary block's sub-complex) on its attached fiber
  /// cells at the fiber's degree. Call it at attachment, when the block's
  /// lengths are the surface's own.
  /// @throws std::out_of_range on the index;
  /// @throws std::logic_error without an attached fiber or an own complex; and
  ///   what `dualFrame` throws.
  [[nodiscard]] Eigen::MatrixXcd inputFrameDual(std::size_t index, const Eigen::MatrixXcd &images) const;
  /// Set the two-body target; scored inside `rU` under `useFiberResiduals`
  /// once two input fibers are attached.
  /// @throws std::invalid_argument on an empty target, or on a shape that
  ///   disagrees with the transfer's when two input fibers are attached: the
  ///   frames' ranks \f$ r_A \times r_B \f$ when both blocks carry a frame,
  ///   the cell counts otherwise. With one frame set the shape is settled at
  ///   read time.
  void setTwoBodyTarget(Eigen::MatrixXcd chi, bool choiDecomposed = true);
  /// Fit one bulk to several input pairs at once. The sum lives in the
  /// objective rather than in the drive, so every move is scored against every
  /// state before it can be accepted and neither stage needs to know that there
  /// is more than one state.
  ///
  /// Evaluation writes each case's boundary metric in turn, scores every
  /// selected reading with that case's target fields where applicable, and
  /// restores the geometry. `ReadoutMode::Whole` consumes the case's input
  /// coefficients. The boundary is expected to be pinned; with an unpinned
  /// boundary the cases compete over coordinates the relaxation is free to
  /// move.
  ///
  /// An empty list restores the single-target behaviour.
  void setTwoBodyCases(std::vector<TwoBodyCase> cases);
  /// The declared two-body cases, in the order they were set.
  [[nodiscard]] const std::vector<TwoBodyCase> &twoBodyCases() const noexcept {
    return twoBodyCases_;
  }
  /// The two-body residual summed over the cases on \p spacetime, or the single
  /// target's residual when no cases are set. The complex is explicit because
  /// stage 1 scores every candidate on a complex rebuilt from a snapshot rather
  /// than on the live one.
  [[nodiscard]] double twoBodyResidualOverCasesOn(
      const std::shared_ptr<Spacetime> &spacetime) const;
  /// One residual per case on \p spacetime, in the order the cases were set;
  /// empty when none are set. The sum is what the drive minimizes, but it
  /// cannot show a step that improves one state at another's expense.
  [[nodiscard]] std::vector<double> twoBodyResidualsPerCaseOn(
      const std::shared_ptr<Spacetime> &spacetime) const;
  /// `twoBodyResidualsPerCaseOn` on the live complex.
  [[nodiscard]] std::vector<double> twoBodyResidualsPerCase() const {
    return twoBodyResidualsPerCaseOn(spacetime_);
  }
  /// `twoBodyResidualOverCasesOn` on the live complex.
  [[nodiscard]] double twoBodyResidualOverCases() const {
    return twoBodyResidualOverCasesOn(spacetime_);
  }
  /// The declared two-body target, or empty when none is set.
  [[nodiscard]] const std::optional<TwoBodyTarget> &twoBodyTarget() const noexcept {
    return twoBodyTarget_;
  }
  /// The two-body residual on the live complex.
  /// @throws std::logic_error without a target or without two attached input fibers.
  [[nodiscard]] double twoBodyResidual() const;
  /// The state the whole complex's harmonic form is meant to be
  /// (`ReadoutMode::Whole`). Its dimension must equal the harmonic rank; the
  /// reading refuses any other size rather than fitting it. Unset, the
  /// 4-dimensional two-body target is used.
  void setOutputStateTarget(Eigen::VectorXcd state);
  /// The declared output-state target, or empty when none is set.
  [[nodiscard]] const std::optional<Eigen::VectorXcd> &outputStateTarget() const noexcept {
    return outputStateTarget_;
  }
  /// The projective leak of a case target against the paired direct-sum frame
  /// transfer — the reading `ReadoutMode::Operator` selects. The caller is
  /// responsible for expressing the target in that frame; it is not a
  /// tensor-product identification. `1.0`, the full leak, when the target
  /// carries none.
  [[nodiscard]] double operatorResidualOn(
      const std::shared_ptr<Spacetime> &spacetime,
      const TwoBodyTarget &target) const;
  /// The projective leak of \p target against the operator promoted from
  /// \f$\ker L_1(W-\partial W)\f$ through a Choi frame — the reading
  /// `ReadoutMode::Bulk` selects. Returns the full leak 1.0 when the framed
  /// kernel is not rank one, the same convention a refused geometry takes under
  /// the transfer reading, so a selected readout set may sum them.
  [[nodiscard]] double bulkOperatorResidualOn(
      const std::shared_ptr<Spacetime> &spacetime,
      const TwoBodyTarget &target) const;
  /// The projective leak of \p target against the whole cobordism's degree-1
  /// harmonic form — the reading `ReadoutMode::Whole` selects.
  ///
  /// The harmonic space is a space, not a state; the input blocks pick the form
  /// out of it. Its columns' transported periods over every block's marking
  /// give \f$\Pi\f$, the input coefficients give \f$p\f$, and the coefficient
  /// vector \f$c\f$ minimizing \f$\|\Pi c - p\|\f$ is the form the inputs
  /// determine. A nonsingular square period block \f$B\f$ fixes the live period
  /// frame, and \f$Bc\f$ is the output state compared to the target
  /// projectively. All nonsingular marking groups must induce that same frame,
  /// and the frame is taken only from the live markings, as in `deriveFrame`.
  ///
  /// A harmonic space whose rank differs from the target's dimension scores the
  /// full leak 1.0, with the reason in `wholeHarmonicObstruction`. Two boundary
  /// tori give rank \f$b_1 = 2\f$ against a 4-dimensional target; four tori
  /// give rank 4.
  [[nodiscard]] double wholeHarmonicResidualOn(
      const std::shared_ptr<Spacetime> &spacetime,
      const TwoBodyTarget &target) const;
  /// Why the last `wholeHarmonicResidualOn` could not read a state, or empty
  /// when it could.
  [[nodiscard]] const std::string &wholeHarmonicObstruction() const noexcept {
    return wholeHarmonicObstruction_;
  }
  /// Read the bulk between the attached frames on the live complex, with
  /// certificates. Two inputs use the ordinary frame transfer; four inputs
  /// use the paired transfer with the first two blocks on side A and the last
  /// two on side B. A four-input read is a diagnostic, so its `residual` is
  /// NaN; no tensor-product target is inferred from the paired direct-sum
  /// matrix.
  /// @throws std::logic_error for any other input count.
  [[nodiscard]] TwoBodyRead readTwoBody() const;

  // ---- analytic gradients of the fiber-mode residuals ----
  /// A gradient over the live complex's edges in `EdgeList` order. Each entry
  /// packs \f$ (\partial/\partial\operatorname{Re}, \partial/\partial\operatorname{Im}) \f$
  /// of the real residual as a complex number, the convention `runStage2`
  /// descends.
  struct ResidualGradient {
    /// Derivative with respect to each edge's squared length.
    Eigen::VectorXcd lengths;
    /// Derivative with respect to each link phase; empty unless the residual
    /// varies with the links.
    Eigen::VectorXcd phases;
  };
  /// The analytic gradient of `fiberResidualOn(spacetime, fiber, band)` through
  /// the band's Riesz projector (`chainhodge::BandDerivative`), holomorphic in
  /// the squared lengths and the phases, over \p spacetime's edges. The contour
  /// is the choice the residual makes (`FiberBand`) and is held fixed.
  [[nodiscard]] ResidualGradient fiberResidualGradientOn(const std::shared_ptr<Spacetime> &spacetime,
                                                         const BoundaryFiber &fiber,
                                                         FiberBand band = FiberBand::AsStored) const;
  /// The analytic gradient of `inputStateResidualOn(block, spacetime)` over
  /// \p spacetime's edges, in the `runStage2` convention: the derivative of the
  /// output-state read's leak. The stage-2 direction of a marked block is
  /// `ownStateResidualGradientOn` instead.
  ///
  /// The leak is \f$ r = \|u\|^2/\|t\|^2 \f$, \f$ u = t - Z_T c \f$ with
  /// \f$ c \f$ the least-squares fit, differentiated with a moving target —
  /// \f$ dr = 2\,\mathrm{Re}\,dF \f$ with
  /// \f$ dF = \big[u^H(dt - dZ_T\,c) - r\,t^H dt\big]/\|t\|^2 \f$ — where
  /// \f$ dZ_T \f$ is the whole's band derivative on the block's edges
  /// (`chainhodge::BandDerivative`, as `fiberResidualGradientOn`) and
  /// \f$ dt = dF\,(a,b)^T \f$ the frame's derivative,
  /// \f$ dF = dZ\,\Pi^{-1} - Z\,\Pi^{-1}\,d\Pi\,\Pi^{-1} \f$, with \f$ dZ \f$
  /// the band derivative of the block's own zero mode on its own pencil and
  /// \f$ d\Pi \f$ the transported periods of \f$ dZ \f$ over the cycles.
  /// \f$ dt \f$ is supported on the block's own edges, mapped to the parent's
  /// by vertex pair, and vanishes on bulk edges. The contours are held fixed
  /// and no finite difference is taken. `phases` is empty at degree 1, and the
  /// gradient is zero when the frame cannot be derived.
  [[nodiscard]] ResidualGradient inputStateResidualGradientOn(const std::shared_ptr<Spacetime> &spacetime,
                                                              const BoundaryBlock &block) const;
  /// `inputStateResidualGradientOn` for several marked \p blocks at once, with
  /// the whole's band derivative computed once and shared. `fiberModeAscent`
  /// uses this for every marked block. Returns one gradient per block, in the
  /// given order; a block whose frame cannot be derived gets the zero gradient.
  /// @throws std::logic_error when a block carries no
  ///   marking or the metric source is not the Whitney pencil.
  [[nodiscard]] std::vector<ResidualGradient> inputStateResidualGradientsOn(
      const std::shared_ptr<Spacetime> &spacetime, const std::vector<const BoundaryBlock *> &blocks) const;
  /// The analytic gradient of `ownStateResidualOn(block, spacetime)` over
  /// \p spacetime's edges (`ResidualGradient`, the `runStage2` convention):
  /// \f$ r \f$ is a real function of \f$ \hat\tau \f$ alone, so
  /// \f$ dr = 2\,\mathrm{Re}\big(\partial_\tau r\, d\hat\tau\big) \f$ with
  /// the Wirtinger derivative \f$ \partial_\tau r = -[\bar b\,\bar u\,D -
  /// |u|^2\bar\tau]/(N D^2) \f$, \f$ u = \bar a + \bar b\tau \f$,
  /// \f$ N = |a|^2 + |b|^2 \f$, \f$ D = 1 + |\tau|^2 \f$ (the chart
  /// \f$ \sigma = 1/\tau \f$ when the marking is swapped), and
  /// \f$ d\hat\tau/dz_e \f$ the qubit's own analytic derivative
  /// (`observables::SimplicialQubit::tauDerivative`, holomorphic in the squared
  /// lengths). It is supported on the block's own edges, mapped to
  /// \p spacetime's by vertex pair, and zero on bulk edges. `phases` is empty,
  /// since \f$ \hat\tau \f$ is invariant under the pure gauge the surface
  /// carries. Zero when the read is refused; no finite difference is taken.
  /// @throws std::logic_error without a marking;
  /// @throws std::invalid_argument on a null spacetime.
  [[nodiscard]] ResidualGradient ownStateResidualGradientOn(const std::shared_ptr<Spacetime> &spacetime,
                                                            const BoundaryBlock &block) const;
  /// `ownStateResidualGradientOn` of input block \p index on the live complex.
  [[nodiscard]] ResidualGradient ownStateResidualGradient(std::size_t index) const;
  /// The analytic gradient of the frame-transfer two-body residual,
  /// \f$ d\tilde A^U = dM^U h + M^U dh \f$ on the attached blocks. This entry
  /// point is transfer-specific whatever readout is selected;
  /// `fiberModeAscent` dispatches the objective's selected readings.
  [[nodiscard]] ResidualGradient twoBodyResidualGradientOn(const std::shared_ptr<Spacetime> &spacetime,
                                                           const TwoBodyTarget &target) const;
  /// The analytic gradient of `wholeHarmonicResidualOn`, along the chain
  /// \f$Z \to \Pi \to c \to Bc \to r\f$: the whole complex's degree-1 harmonic
  /// images, their transported periods over the blocks' marked cycles, the
  /// least-squares coefficient vector, the live period-frame state, and its
  /// projective leak against the target.
  /// `BandDerivative::imagesLengthDerivative` supplies \f$dZ/ds_e\f$; the chain
  /// rule includes both \f$B\,dc\f$ and \f$(dB)c\f$, and the leak's derivative
  /// is the expression `twoBodyResidualGradientOn` takes with \f$T\f$ replaced
  /// by \f$Bc\f$.
  ///
  /// Zero where the residual itself refuses. `phases` is empty: this is a
  /// degree-1 length gradient.
  [[nodiscard]] ResidualGradient wholeHarmonicResidualGradientOn(
      const std::shared_ptr<Spacetime> &spacetime,
      const TwoBodyTarget &target) const;
  /// The ascent of every fiber-mode term of `rU` on the live complex: the
  /// whole-complex fiber target, each input block's fiber, and the two-body
  /// target. A block's fiber gradient is taken on the block's own complex — a
  /// surface block's own surface at its zero mode, an ordinary block's
  /// sub-complex at the fiber's contour — and mapped to the parent's edges by
  /// vertex pair.
  /// @throws std::logic_error when a selected two-body reading has no analytic gradient.
  [[nodiscard]] ResidualGradient fiberModeAscent() const;
  /// Whether every selected reading has an analytic gradient, so that
  /// `fiberModeAscent` is the direction of the objective being minimized.
  /// Transfer and whole-harmonic readings do; bulk and paired-frame operator
  /// readings do not, so stage 2 falls back to the numerical ascent of `rU`
  /// whenever either is selected.
  [[nodiscard]] bool readoutsHaveAnalyticGradient() const noexcept;

  /// Attach the fiber form of an input block's target (a prior cobordism's
  /// output fiber piped downstream).
  /// @throws std::out_of_range on the index.
  void setInputFiber(std::size_t index, BoundaryFiber fiber);
  /// Attach the fiber form of an output block's target.
  void setOutputFiber(std::size_t index, BoundaryFiber fiber);
  /// The fiber of input block \p index, or empty when none is attached.
  [[nodiscard]] const std::optional<BoundaryFiber> &inputFiber(std::size_t index) const;
  /// The fiber of output block \p index, or empty when none is attached.
  [[nodiscard]] const std::optional<BoundaryFiber> &outputFiber(std::size_t index) const;

  /// Pin the two input blocks' fibers as boundary data and relax the bulk so
  /// the whole complex carries them: the two fibers' images are concatenated
  /// on the union of their cells (the labeled sum on disjoint supports) and
  /// handed to `relaxFixedBoundaryEigenstate`, which holds the geometric
  /// boundary fixed and varies only interior weights. No interior simplex is
  /// identified by hand. The fixed-boundary fit takes one pinned state, so the
  /// fibers must be rank one; a joint multi-column fit is refused rather than
  /// approximated column by column.
  /// @throws std::invalid_argument unless exactly two input blocks carry
  ///   rank-one fibers at the requested degree on disjoint cells.
  [[nodiscard]] FixedBoundaryEigenstateResult pinInputFibers(
      int degree, double epsilon = 1e-10, int restarts = 64, int maxGrowth = 4,
      std::uint64_t seed = 0, int maxIterations = 200);

  /// Read the fiber form of an output block's target from the live complex:
  /// the Riesz band of \p contour on the whole complex's pencil (the harmonic
  /// contour of `PencilLayer::harmonicContour` when \p contour is null)
  /// restricted to the block's degree-\p degree cells. Stores it on the block
  /// and returns it. Requires the Whitney pencil metric source.
  [[nodiscard]] BoundaryFiber readOutputFiber(std::size_t index, int degree,
                                              const chainhodge::Contour *contour = nullptr,
                                              double kappa = 10.0);
  /// Stage 1 (combinatorial): greedy best-ΔF surgical moves on the complex.
  ///
  /// \param maxSteps             Maximum stage-1 updates.
  /// \param nCandidateMoves      Candidate moves drawn per batch.
  /// \param growBoundaries       Initialization pass. While true the boundary
  ///   regions grow to track the bulk until they carry their states
  ///   (`growBlockRegions`). Run the bulk evolution with it false, so that
  ///   \f$ \partial W \f$ stays frozen.
  /// \param maxLookahead         When a batch of single moves finds no
  ///   improvement, the search deepens iteratively — 2-move sequences, then 3,
  ///   up to this many moves — committing an F-lowering sequence as a whole.
  ///   A deepened batch builds and scores many more candidates per iteration,
  ///   so 1 (single moves) is the default. Every depth scores the same way,
  ///   unrelaxed.
  /// \param combinatorialBreadth When non-zero, the depth ladder runs the other
  ///   way round: sequences of exactly that many moves are searched first, and
  ///   the search backs off one move at a time only when nothing at the current
  ///   breadth lowers \f$ F \f$. This can find a pair, or a longer composition,
  ///   that improves the residual on a complex where every single move makes it
  ///   worse. Zero leaves the ascending \p maxLookahead schedule in place.
  std::vector<double> runStage1(int maxSteps = 200, int nCandidateMoves = 12,
                                bool growBoundaries = false,
                                int maxLookahead = 1,
                                int combinatorialBreadth = 0);
  /// Stage 2 (geometric): relax every squared edge coordinate
  /// \f$z_e=\ell_e^2\f$ of the whole complex toward a stationary point of the
  /// selected scalar objective.
  ///
  /// Derivatives are taken with respect to \f$z\f$ and subtracted from
  /// \f$z\f$ itself. `Edge` stores \f$\ell\f$, so a trial is written with the
  /// square root closest to the resident branch. Neither the imaginary
  /// component nor the complex phase is projected away.
  ///
  /// The real line-search scale is backed off until a trial lowers the selected
  /// objective by at least the absolute \p tolerance; otherwise the original
  /// lengths are restored verbatim and the call reports stationarity. An
  /// evaluation error also restores and propagates. `JointStationarity` and
  /// `MediatedCorrespondence` differentiate every scalar term; `Legacy` uses a
  /// Regge search direction with exact full-objective acceptance.
  std::vector<double> runStage2(double beta = 1.0, int maxIters = 200,
                                  double alpha0 = 0.05, double tolerance = 1e-12);
  /// The combined drive. Each iteration takes one combinatorial stage-1 update
  /// — a best-ΔF move, deepening to \p maxLookahead-move sequences on a stall —
  /// and then relaxes the geometry fully: stage-2 updates repeat until the
  /// absolute improvement test at \p tolerance reports diminishing returns, so
  /// every move is proposed from, and leaves behind, relaxed geometry.
  ///
  /// Target-conditioned modes can exit once the register is carried with the
  /// geometry stationary. Every mode can also exit once combinatorial moves
  /// have had no effect — nothing committed at any lookahead depth and nothing
  /// left to relax — for a few consecutive iterations. The last geometric
  /// relaxation before exit runs at a 1e-12 tolerance; if it still finds
  /// descent the loop continues on the freshly relaxed geometry.
  /// \p maxIters is the hard budget cap.
  ///
  /// \p nCandidateMoves, \p growBoundaries and \p maxLookahead parameterize the
  /// combinatorial half as in `runStage1`; \p beta, \p alpha0 and \p tolerance
  /// the geometric half as in `runStage2`. \p beta is stored as the node's
  /// Regge weight before either half runs, so both stages, `objective()` and
  /// the shared trace score one functional. `lastStage2Stationary()` reports
  /// the last geometric update's outcome. \p relaxBudgetPerMove caps the
  /// stage-2 updates following each committed move and the tight exit
  /// re-check, bounding slow descent tails.
  ///
  /// \returns the combined \f$ F \f$ trace.
  std::vector<double> run(int maxIters = 200, int nCandidateMoves = 12,
                          bool growBoundaries = false,
                          double beta = 1.0, double alpha0 = 0.05,
                          double tolerance = 10e-9, int maxLookahead = 1,
                          int relaxBudgetPerMove = 10,
                          int combinatorialBreadth = 0);

  /// One solve action on this node: the unit a search policy composes, so that
  /// the solve is driven through the engine rather than re-implemented by each
  /// consumer.
  enum class BuildAction { Grow, Evolve, Relax, ConeOut, ConeIn };

  /// Secondary sort for the directed cone-out probe; both orders are
  /// interior-first. `AdjacentHolesLast` sends cells sharing vertices with the
  /// existing holes to the back, so new holes come out separated;
  /// `AdjacentHolesFirst` brings them to the front, so the register clusters.
  /// For the first hole the orders coincide.
  enum class HolePlacementStrategy { AdjacentHolesFirst, AdjacentHolesLast };

  /// Apply one `BuildAction` to this node, in place. `Grow` and `Evolve` are
  /// `runStage1` with `growBoundaries` true and false; `Relax` is `runStage2`;
  /// `ConeOut` and `ConeIn` are the directed probes below. Parameters
  /// irrelevant to the chosen action are ignored.
  void buildStep(BuildAction action, int maxSteps = 30, int nCandidateMoves = 8,
                 double stage2Beta = 1.0, int stage2MaxIters = 10,
                 double stage2Alpha0 = 0.05,
                 HolePlacementStrategy holePlacementStrategy = HolePlacementStrategy::AdjacentHolesLast);

  /// Directed, gated cone-out: remove top cells deliberately. Candidate top
  /// cells are enumerated interior-first and ordered by \p strategy, each is
  /// tried with a gated `SurgicalCone::coneOut` and rolled back, and the
  /// hole-opener that most lowers this node's `rU` is kept. Repeats up to
  /// \p maxOpen times, stopping when no opener lowers `rU`.
  ///
  /// The manifold check inside `SurgicalCone::coneOut` is the only gate: a
  /// cone-out that removes a pinned vertex is accepted when the result is a
  /// valid manifold in its own right.
  ///
  /// \returns the number of holes opened.
  [[nodiscard]] int directedConeOut(HolePlacementStrategy strategy = HolePlacementStrategy::AdjacentHolesLast,
                                    int maxOpen = 6);

  /// Directed, gated cone-IN: select the register. Enumerates the boundary facets of the
  /// current emergent holes (capping one closes that hole), tries each with a gated
  /// `SurgicalCone::coneIn`, which builds on a fresh vertex and so removes
  /// nothing, and keeps the cap that most lowers `rU`. Repeats up to
  /// \p maxClose times and stops when no cap lowers `rU`.
  /// \returns the number of holes capped.
  [[nodiscard]] int directedConeIn(int maxClose = 6);
  /// Cone out \p count top cells chosen uniformly at random among those valid
  /// to remove: a deliberate perturbation rather than an improvement. Nothing
  /// here is priced, unlike `directedConeOut`, whose greedy rule is what walks
  /// a run into a local minimum. The objective is generally higher afterwards.
  ///
  /// A cell is valid when the manifold gate inside `SurgicalCone::coneOut`
  /// accepts the removal and the cell touches no declared pinned region.
  /// `directedConeOut` accepts removing a pinned vertex whenever the result is
  /// a manifold in its own right; a random backstep leaves held geometry alone.
  ///
  /// Draws from this node's seeded generator, so a backstep is reproducible
  /// from the run's seed.
  ///
  /// \returns how many cells were removed, which is less than \p count when
  ///   the valid candidates run out.
  [[nodiscard]] int randomConeOut(int count);


  // ==================================================================
  // Surface inputs and the bridge move
  // ==================================================================
  //
  // Two boundary surfaces sit in one host. The bulk between them starts as the
  // collar (`collar`), the minimal manifold connecting them; from there
  // stage 1 and stage 2 are emergent. `bridge` is one of stage 1's gated move
  // kinds: a candidate top cell on existing vertices, k from one surface block
  // and d+1-k from the other (1+3, 2+2, 3+1 for tetrahedra), applied through
  // `SurgicalCone::bridge`, gated by the manifold check and scored by ΔF. It is
  // offered while a face of a surface block is uncovered. Each part of the
  // split lies in one surface's own simplices, which makes a chord impossible
  // by construction — the manifold check could not see one.

  /// The stage-1 bridge move; the payload is the cell's \f$ d+1 \f$ vertex ids.
  /// Drawn only on a node with surface inputs whose bridge phase is incomplete.
  /// The draw is one uniformly chosen candidate of `bridgeCandidatesOn`, i.e. a
  /// top cell adjacent to the current frontier.
  static constexpr const char *kBridge = "bridge";

  /// Whether this node has surface inputs: at least two input blocks marked
  /// `BoundaryBlock::surface`. Only then does the stage-1 draw offer the bridge
  /// move.
  [[nodiscard]] bool hasSurfaceInputs() const;

  /// The faces of the surface input blocks that no top cell covers (sorted
  /// vertex-id tuples): empty on a collar seed, non-empty once a surgery has
  /// uncovered a surface face.
  [[nodiscard]] std::vector<std::vector<std::uint64_t>> uncoveredInputFaces() const;

  /// Completion of the bridge phase: every face of every surface input block
  /// has exactly one top cell on it and `Spacetime::getBoundary()` is exactly
  /// the union of the surface faces, so \f$ \partial W = T_A \sqcup T_B \f$.
  /// True by construction on a collar seed; false on a node without surface
  /// inputs.
  [[nodiscard]] bool bridgePhaseComplete() const;

  // ==================================================================
  // Pinning — a geometric constraint
  // ==================================================================
  //
  // Pinning constrains geometry and says nothing about what a cell carries. It
  // is declared by the caller and never derived from boundary blocks or their
  // targets, so a pinned set means the same thing whether or not a target is
  // present. That is what lets a boundary be declared without
  // target-conditioning the bulk geometry.
  //
  // Pinning and manifold validity act on different axes:
  //
  //   * a pinned edge — one whose endpoints are both pinned — is held at its
  //     resident squared length: stage 2 zeroes its descent component, so
  //     relaxation moves the rest of the complex around it.
  //
  //   * `dualComplexValid` gates the topology, and decides only whether the
  //     result is a valid manifold-with-boundary. A surgery that removes a
  //     pinned vertex is accepted when the result is a valid manifold in its
  //     own right; refusing it would foreclose a legitimate topology change,
  //     and surgery is the only topology-changing mechanism the engine has,
  //     since Pachner moves are bistellar and preserve Betti numbers.



  /// Declare a pinned region, replacing any existing region of the same name.
  void declarePinnedRegion(PinnedRegion region);

  /// Every declared pinned region, in declaration order.
  [[nodiscard]] const std::vector<PinnedRegion> &pinnedRegions() const noexcept {
    return pinnedRegions_;
  }

  /// Drop every declared region, leaving the whole complex free to relax.
  void clearPinnedRegions();

  /// Mint a `RegionHandle` for a declared region. This is the only way to
  /// obtain a non-empty handle, so an objective cannot reference a region that
  /// was never declared: a mis-spelled name throws here instead of compiling
  /// into a scope that matches nothing.
  /// @throws std::invalid_argument if no region of that name is declared.
  [[nodiscard]] RegionHandle regionHandle(const std::string &name) const;

  /// The union of every region's vertices — the flat view, for callers that need
  /// membership rather than provenance.
  [[nodiscard]] std::set<std::uint64_t> pinnedVertices() const;

  /// Whether the edge between \p a and \p b is held fixed: true when a single
  /// region contains both endpoints. Two regions that each contain one endpoint
  /// do not pin the edge between them; that edge spans the gap between two
  /// independently declared regions and is part of the bulk.
  [[nodiscard]] bool edgeIsPinned(std::uint64_t a, std::uint64_t b) const;

  // ==================================================================
  // Modes, the enumerable objective, and the analysis overlay
  // ==================================================================
  //
  // The no-feedback emergence firewall is enforced structurally:
  //
  //   * `objectiveOf` is a static function of `ObjectiveTerms`, a record with
  //     five named scalar members. `objectiveFor` can therefore consult only
  //     what `objectiveTermsFor` puts into that record, having no `this`
  //     through which to reach an analysis member. `objectiveTermNames`
  //     enumerates the list so a test can assert it.
  //   * `refinementDecisionOf` is likewise a static function of
  //     `RefinementIndicators`, five particle-independent geometric and
  //     numerical quantities, and `refinementIndicatorNames` enumerates them.
  //   * the only channel from the carried quantum state to the geometry is the
  //     `carriedStateEnergy` term, which is identically zero unless the run
  //     declares the `CertificatesBlindMeanField` sub-mode.
  //   * every recursive and certificate read produced by `runRecursiveAnalysis`
  //     lands in the checkpoint document and nowhere else: no member the
  //     objective or the refinement decision reads is written by that pass.

  /// The three top-level simulation modes.
  enum class SimulationMode {
    /// The production scientific mode: only the base geometric objective (plus
    /// the one permitted state-energy term) drives optimization; every particle
    /// and gauge quantity is a post-hoc observable.
    Emergence,
    /// A pinned carrier / spectral sector, to establish existence, measure a
    /// residual floor, or build an oracle fixture. Never counted as emergence.
    Synthesis,
    /// Recompute every derived hierarchy and certificate from a checkpoint and
    /// verify that no cached or serialized choice changed the result.
    Replay
  };

  /// The two labeled, Gaussian-closed emergence sub-modes. Recorded in
  /// provenance on every checkpoint; both carry a covariance purity
  /// certificate.
  enum class EmergenceSubmode {
    /// The carried state does not act back on the geometry.
    Strict,
    /// The carried state's energy density may enter the joint stationarity
    /// objective through \f$ h = h(\Gamma, g) \f$, while no component, fiber,
    /// transport, amplitude, color, particle, charge, flavor, exchange or spin
    /// certificate may influence a geometry move.
    CertificatesBlindMeanField
  };

  /// The enumerable term list the scalar objective is the sum of. `objectiveOf`
  /// is static over this record, so the objective reads nothing else. Every
  /// member is a geometric or target quantity except the last, which is the one
  /// permitted state channel. Declared at namespace scope alongside
  /// `CobordismObjective`, so an objective can be written without depending on
  /// this class.
  using ObjectiveTerms = ::tessera::cobordism::ObjectiveTerms;

  /// The names of `ObjectiveTerms`' members, in declaration order.
  [[nodiscard]] static std::vector<std::string> objectiveTermNames();

  /// The scalar objective: the sum of the declared terms. Static, so it cannot
  /// reach any analysis state.
  [[nodiscard]] static double objectiveOf(const ObjectiveTerms &terms);

  /// Decompose the objective on `spacetime` into its declared terms.
  [[nodiscard]] ObjectiveTerms objectiveTermsFor(
      const std::shared_ptr<Spacetime> &spacetime) const;
  /// `objectiveTermsFor` on this node's own complex.
  [[nodiscard]] ObjectiveTerms objectiveTerms() const;

  /// Select the simulation mode and (for emergence) its labeled sub-mode.
  /// Selecting anything other than `CertificatesBlindMeanField` sets the
  /// carried-state energy term to exactly zero.
  void setSimulationMode(SimulationMode mode,
                         EmergenceSubmode submode = EmergenceSubmode::Strict);
  /// The selected simulation mode.
  [[nodiscard]] SimulationMode simulationMode() const noexcept {
    return simulationMode_;
  }
  /// The selected emergence sub-mode.
  [[nodiscard]] EmergenceSubmode emergenceSubmode() const noexcept {
    return emergenceSubmode_;
  }
  /// `"emergence"` / `"synthesis"` / `"replay"`, as stamped on a checkpoint.
  [[nodiscard]] static std::string modeName(SimulationMode mode);
  /// `"strict"` / `"certificates_blind_mean_field"`, as stamped on a
  /// checkpoint.
  [[nodiscard]] static std::string submodeName(EmergenceSubmode submode);

  // ---- the carried quasi-free state ----

  /// Adopt the carried state: the covariance \f$ \Gamma \f$ (flat row-major,
  /// \f$ m \times m \f$) over \f$ m \f$ one-particle modes, each named by the
  /// \p degree-cell it occupies (`modeCells[i]`, matched by vertex set). No
  /// certificate is read; the state is numbers plus the cells they live on.
  /// @throws std::invalid_argument on a non-square covariance, a size mismatch
  ///   against \p modeCells, or a degree below one, which is the carried
  ///   state's declared domain rather than a capability limit.
  void setCarriedState(
      const std::vector<std::vector<std::uint64_t>> &modeCells, int degree,
      const std::vector<std::complex<double>> &covariance);
  /// Drop the carried state; the energy term becomes exactly zero.
  void clearCarriedState();
  /// Whether a carried state is adopted.
  [[nodiscard]] bool hasCarriedState() const noexcept {
    return !carriedModeCells_.empty();
  }
  /// The degree of the cells the carried modes occupy.
  [[nodiscard]] int carriedStateDegree() const noexcept {
    return carriedStateDegree_;
  }
  /// The cells naming the carried one-particle modes.
  [[nodiscard]] const std::vector<std::vector<std::uint64_t>> &
  carriedStateModeCells() const noexcept {
    return carriedModeCells_;
  }
  /// The carried covariance \f$ \Gamma \f$, flat row-major.
  [[nodiscard]] const std::vector<std::complex<double>> &
  carriedStateCovariance() const noexcept {
    return carriedCovariance_;
  }

  /// Declare the spectral-moment stiffness of the geometric action about the
  /// current geometry, the carrier (`HodgeLaplacian::spectralMomentStiffness`):
  /// the local spectral moments of orders \f$ 1,\dots,m \f$ at each of
  /// `degrees` are recorded now as the reference, and every objective gains
  /// \f$ \beta_M\sum_k\operatorname{Re}S_{M,k} \f$. `weight` 0 removes it.
  /// @throws std::invalid_argument for a negative or non-finite weight, a
  ///   negative coefficient, or no coefficients with a nonzero weight.
  void setMomentStiffness(double weight, const std::vector<int> &degrees,
                          const std::vector<double> &coefficients);
  [[nodiscard]] double momentStiffnessWeight() const noexcept { return momentStiffnessWeight_; }
  [[nodiscard]] const std::vector<int> &momentStiffnessDegrees() const noexcept {
    return momentStiffnessDegrees_;
  }
  [[nodiscard]] const std::vector<double> &momentStiffnessCoefficients() const noexcept {
    return momentStiffnessCoefficients_;
  }

  /// The mean-field coefficient \f$ \beta_E \f$, checkpointed. Default 0.
  void setCarriedStateEnergyWeight(double weight);
  /// The mean-field coefficient \f$ \beta_E \f$.
  [[nodiscard]] double carriedStateEnergyWeight() const noexcept {
    return carriedStateEnergyWeight_;
  }

  /// The carried-state energy density
  /// \f$ E_{\rm carried}(\Gamma,g)=\operatorname{Re}\operatorname{tr}
  ///     \bigl(\Gamma_S\,h_S(g)\bigr) \f$,
  /// the exact quasi-free expectation \f$ \langle d\Gamma(h)\rangle \f$ of the
  /// one-particle generator \f$ h_S(g)=\tfrac12(L_k+L_k^\dagger)\big|_S \f$ —
  /// the Hermitian part of the metric Hodge operator at the carried degree,
  /// restricted to the carried modes' cells. `S` is the set of carried mode
  /// cells that still exist in `spacetime`: a mode whose cell a move removed
  /// contributes nothing, and a cell the move created carries no occupation,
  /// so the term survives combinatorial change without a repair step.
  ///
  /// Depends only on \f$ \Gamma \f$ and the classical geometry. Exactly 0
  /// outside the `CertificatesBlindMeanField` sub-mode, with no carried state,
  /// or at weight zero.
  ///
  /// This is a real scalar the engine adds to its scalar objective, built from
  /// the Hermitian part of the operator so that the covariance evolution of
  /// `advanceCarriedState` stays unitary and therefore Gaussian-closed. The
  /// complex bilinear density \f$ \operatorname{tr}(\Gamma\,h(z,U)) \f$ of
  /// Section 7, with no adjoint and no real projection, together with its
  /// complex force on both edge fields and the self-consistent
  /// \f$ (z^{*},\Gamma^{*}) \f$ iteration that re-occupies the modes of
  /// \f$ h \f$, are `cobordism::JointAction` and
  /// `cobordism::SelfConsistentMeanField`.
  [[nodiscard]] double carriedStateEnergy(
      const std::shared_ptr<Spacetime> &spacetime) const;

  /// Exact analytic gradient of `carriedStateEnergy` with respect to each
  /// edge's \f$ \ell^2 \f$, in `getEdgeList()` order:
  /// \f$ \partial E/\partial z_e=\operatorname{Re}\operatorname{tr}
  ///     (\Gamma_S\,[\partial L_k/\partial z_e]_S) \f$ from
  /// `HodgeLaplacian::laplacianGradient` — no finite differences. The
  /// returned component is the real-plane ascent displacement
  /// \f$ \partial E/\partial(\operatorname{Re}z)+
  ///     i\,\partial E/\partial(\operatorname{Im}z) \f$, matching the
  /// convention stage 2 subtracts.
  [[nodiscard]] std::vector<std::complex<double>> carriedStateEnergyGradient(
      const std::shared_ptr<Spacetime> &spacetime) const;

  /// The purity defect \f$ \lVert\Gamma^2-\Gamma\rVert_F \f$ of the carried
  /// covariance, the Gaussianity certificate both emergence sub-modes report.
  /// NaN with no carried state.
  [[nodiscard]] double carriedStatePurityDefect() const;
  /// Whether the purity certificate holds at \p tolerance.
  [[nodiscard]] bool carriedStatePurityHolds(double tolerance = 1e-9) const;

  /// The checkpointed mean-field update schedule: \p dt per step, \p steps
  /// steps per `advanceCarriedState` call.
  void setMeanFieldSchedule(double dt, int steps);
  /// The mean-field step size.
  [[nodiscard]] double meanFieldStepSize() const noexcept {
    return meanFieldStepSize_;
  }
  /// Mean-field steps per `advanceCarriedState` call.
  [[nodiscard]] int meanFieldSteps() const noexcept { return meanFieldSteps_; }

  /// Advance the carried covariance through `CovarianceState::meanFieldEvolve`
  /// under the same generator the energy term uses,
  /// \f$ h(\Gamma,g)=\tfrac12(L_k+L_k^\dagger)|_S \f$, with the classical
  /// geometry closed over. The loop is Gaussian-closed by construction and the
  /// certificate measures that closure.
  ///
  /// The covariance is transported by this generator and is not re-occupied: a
  /// state carried along a moving geometry keeps the modes it was given, which
  /// is a different question from which modes of \f$ h(z) \f$ are filled at the
  /// geometry it arrives on. The occupation of the current operator's modes,
  /// and the fixed point of the two together, are
  /// `cobordism::SelfConsistentMeanField`.
  /// \returns the worst purity defect across the steps; NaN with no carried
  ///   state.
  double advanceCarriedState();

  // ---- particle-independent refinement ----

  /// The geometric and numerical indicators emergence-mode refinement may
  /// consult. Every member is a quantity of the base problem: none is a
  /// coarse-response residual, band gap, modularity, transport leakage, Wilson
  /// or center read, exchange read, anchor score, amplitude Gram defect, or
  /// particle score.
  struct RefinementIndicators {
    /// \f$ \lVert\nabla_z S_{\rm Regge}\rVert^2 \f$ on the current complex.
    double reggeStationarityResidual = 0.0;
    /// \f$ \sum_k\lVert\nabla_z S_{\rm Hodge,k}\rVert^2 \f$.
    double hodgeStationarityResidual = 0.0;
    /// Curvature concentration: \f$ \max_h|\varepsilon_h| \f$ over the mean
    /// \f$ |\varepsilon_h| \f$ across the \f$(d-2)\f$-hinges (1 = flat spread,
    /// large = curvature piling onto one hinge). 0 when no hinge carries any.
    double curvatureConcentration = 0.0;
    /// Mesh quality: \f$ \min_\sigma|{\rm vol}\,\sigma| /
    /// \max_\sigma|{\rm vol}\,\sigma| \f$ over the top cells, in `[0,1]`;
    /// 0 means a degenerate cell. 1 with no cells.
    double meshQuality = 1.0;
    /// Solver discretization error: the magnitude of the last accepted stage-2
    /// objective improvement; 0 at a stationary point.
    double solverError = 0.0;
  };

  /// The names of `RefinementIndicators`' members, in declaration order.
  [[nodiscard]] static std::vector<std::string> refinementIndicatorNames();

  /// Measure the indicators on this node's current complex.
  [[nodiscard]] RefinementIndicators refinementIndicators() const;

  /// The thresholds `refinementDecision` compares against. The defaults never
  /// fire: `reggeStationarityResidual`, `hodgeStationarityResidual`,
  /// `curvatureConcentration` and `solverError` are upper bounds crossed from
  /// below (infinity means never), while `meshQuality` is a lower bound crossed
  /// from above (0 means never).
  void setRefinementThresholds(const RefinementIndicators &thresholds);
  /// The configured refinement thresholds.
  [[nodiscard]] const RefinementIndicators &refinementThresholds()
      const noexcept {
    return refinementThresholds_;
  }

  /// Whether to refine, and which indicator asked.
  struct RefinementDecision {
    /// True when some indicator crossed its threshold.
    bool refine = false;
    /// The indicator that asked: one of `refinementIndicatorNames()`, or empty.
    std::string trigger{};
    /// The measured indicators the decision was taken on.
    RefinementIndicators indicators{};
  };

  /// The refinement rule. Static over the indicator record, so it cannot reach
  /// a certificate, a fiber, a transport or a particle read.
  [[nodiscard]] static RefinementDecision refinementDecisionOf(
      const RefinementIndicators &indicators,
      const RefinementIndicators &thresholds);
  /// `refinementDecisionOf` on this node's measured indicators.
  [[nodiscard]] RefinementDecision refinementDecision() const;

  /// Apply geometry refinement when, and only when, `refinementDecision()` asks
  /// for it, through the gated cone-in surgery `applyMoveSpecification` uses —
  /// the same `dualComplexValid` gate as stage 1 and `preconeCells`. No cell is
  /// inserted without passing it.
  /// \returns the number of refinement cells committed.
  int refineGeometry(int maxCells = 1);

  // ---- the post-hoc analysis overlay ----

  /// Analysis-overlay configuration. Disabled by default: with `enabled` false
  /// no part of the overlay runs.
  struct AnalysisConfig {
    /// Master switch. False means the overlay never runs.
    bool enabled = false;
    /// Run one pass after every \p cadence accepted combinatorial moves.
    int cadence = 1;
    /// Hodge degrees the spectral bands are enumerated at.
    std::vector<int> degrees{1};
    /// The modularity resolution sequence scanned per pass.
    std::vector<double> resolutions{1.0};
    /// How many cobordism frames the overlay retains. One analysis pass is one
    /// frame, so the retained frames are the last \p frameHistory passes, each
    /// keeping its components, its candidate bands and their anchors. The
    /// candidate's lifetime, its smallest adjacent-frame overlap, its
    /// per-frame band and anchor families and its lifetime transports are
    /// measured across them; with `frameHistory` 1 there is no history, the
    /// pass sees a single frame, and the lifetime certificates fail by name
    /// rather than passing vacuously. The default 4 clears the classifier's
    /// two-frame stability floor with room to lose a frame.
    /// @throws std::invalid_argument (from `setAnalysisConfig`) below 1.
    int frameHistory = 4;
    /// How the determinant line of a lifetime transport family is closed.
    ///
    ///  - `"none"` (the default): the family is an open cobordism segment with
    ///    no declared closure. Its phase is reported and the winding is left
    ///    unknown — an open path has no integer winding, and inventing one
    ///    would be a measurement nobody made.
    ///  - `"closed-family"`: the caller declares the candidate's world tube
    ///    closed, so the family is read cyclically (the closing step returns to
    ///    the first sample) and the closed-family determinant winding is the
    ///    winding reported. It is a declaration about the run, like a causal
    ///    type, and it is recorded on every read it produces. The overlay
    ///    consumes that winding; nothing here computes it.
    ///
    /// @throws std::invalid_argument (from `setAnalysisConfig`) on any other
    ///   value: an unrecognized closure is refused, never silently ignored.
    std::string lifetimeWindingClosure{"none"};
    /// Build the lazy Fock expression. For oracle and explicit non-Gaussian
    /// boundary data only, never the quasi-free production representation.
    bool fockOracle = false;
    /// Serve nothing from the `AnalyticCache`, so the cold path can be compared
    /// against the incremental one.
    bool coldCaches = false;
  };

  /// Configure the analysis overlay.
  void setAnalysisConfig(const AnalysisConfig &config);
  /// The analysis-overlay configuration in force.
  [[nodiscard]] const AnalysisConfig &analysisConfig() const noexcept {
    return analysisConfig_;
  }

  /// Deterministic provenance stamped on every checkpoint.
  void setProvenance(const std::string &configHash, const std::string &commit);
  /// The provenance configuration hash.
  [[nodiscard]] const std::string &provenanceConfigHash() const noexcept {
    return provenanceConfigHash_;
  }
  /// The provenance commit identifier.
  [[nodiscard]] const std::string &provenanceCommit() const noexcept {
    return provenanceCommit_;
  }

  /// Committed combinatorial moves since construction (the cadence counter).
  [[nodiscard]] std::uint64_t acceptedMoveCount() const noexcept {
    return acceptedMoveCount_;
  }
  /// Completed analysis passes since construction.
  [[nodiscard]] std::uint64_t analysisPassCount() const noexcept {
    return analysisPassCount_;
  }

  /// Run one post-hoc analysis pass over the current accepted geometry, in the
  /// firewall order: publish the accepted move's touched star to the
  /// `AnalyticCache`; update the component hierarchy and its invalidated
  /// ancestry; update the spectral projectors and the labeled retained-fiber
  /// sum; update the transports and the Wilson and center reads; update the
  /// quasi-free covariance and its Wick reads; build the lazy Fock expression
  /// when the oracle is selected; evaluate the particle reads; and record the
  /// checkpoint.
  ///
  /// Read-only on the geometry: this pass cannot accept, reject or prioritize a
  /// move, and writes nothing the objective or the refinement decision reads.
  void runRecursiveAnalysis();

  /// The versioned checkpoint document of the last pass, as JSON. Empty before
  /// the first pass. Unknown or uncertified values serialize as `null`, never
  /// as zero.
  [[nodiscard]] const std::string &checkpointJson() const noexcept {
    return checkpointJson_;
  }
  /// The checkpoint schema version this build writes and accepts.
  ///
  /// Version 5 carries the per-edge connection phase in `raw_complex.edges`
  /// alongside the length. In it, `particles.bound_supercomponents` holds the
  /// bound-supercomponent search records and `particles.baryons` holds the
  /// three-cluster verdict, one baryon read per binding of exactly three
  /// certified constituents.
  ///
  /// Documents at earlier versions are rejected on read rather than
  /// reinterpreted: their `baryons` entries mean something different, and
  /// silence about the phase is not evidence that the phase was zero.
  [[nodiscard]] static int checkpointSchemaVersion() noexcept { return 5; }

  /// The phase an older document would replay with, were one accepted: exactly
  /// zero. That is faithful to what such a document records, since it stores no
  /// phase, but not to the run that wrote it, whose edges may have carried any
  /// phase. That gap is why older documents are rejected rather than silently
  /// defaulted.
  [[nodiscard]] static std::complex<double> replayPhaseDefault() noexcept {
    return {0.0, 0.0};
  }

  /// Replay mode: rebuild the raw complex recorded in `checkpoint`, disable
  /// every cache, recompute every derived hierarchy and certificate, and
  /// return the freshly written checkpoint — stamped `"replay"`. The verdicts
  /// must equal the incremental run's.
  /// @throws std::invalid_argument on malformed JSON or an unknown
  ///   `schema_version`.
  [[nodiscard]] static std::string replayCheckpoint(
      const std::string &checkpoint);

  /// The `schema_version` recorded in `checkpoint`.
  /// @throws std::invalid_argument on malformed JSON or a missing version.
  [[nodiscard]] static int checkpointVersionOf(const std::string &checkpoint);

  /// The node's live complex.
  [[nodiscard]] std::shared_ptr<Spacetime> spacetime() const { return spacetime_; }
  /// The input boundary blocks, in seed order.
  [[nodiscard]] const std::vector<BoundaryBlock> &inputs() const {
    return inputBlocks_;
  }
  /// The output boundary blocks, in seed order.
  [[nodiscard]] const std::vector<BoundaryBlock> &outputs() const {
    return outputBlocks_;
  }
  /// Whether the last `runStage2` ended at a stationary point — no complex-z
  /// line-search trial lowered \f$ F \f$ by the absolute tolerance — rather
  /// than on the iteration budget. False before the first `runStage2` or `run`.
  /// After `run` it reports the last geometric update's outcome; each update
  /// resets the flag, so an earlier stationary point that a later topology
  /// change reopened does not latch.
  [[nodiscard]] bool lastStage2Stationary() const { return lastStage2Stationary_; }

  /// The lookahead depth of the last stage-1 update's committed sequence: 1 for
  /// an ordinary single move, greater than 1 when the single-move batch stalled
  /// and an F-lowering multi-move sequence was found at that depth, and 0 when
  /// no F-lowering sequence was found at any depth up to the update's
  /// `maxLookahead`. 0 before the first update.
  [[nodiscard]] int lastStage1Lookahead() const { return lastStage1LookaheadDepth_; }

 private:
  /// The fixed boundary as the gate reads it: `Spacetime::getBoundary` together
  /// with the `Spacetime::EdgeGeometry` of every edge lying inside one of those facets.
  /// Both halves come from one pass, because the gate takes this record twice
  /// per candidate — once before the move and once after — and reading the
  /// boundary is the whole cost of arming it.
  struct BoundaryRecord {
    std::set<std::vector<std::uint64_t>> facets;
    std::map<std::pair<std::uint64_t, std::uint64_t>, Spacetime::EdgeGeometry> edges;
  };
  [[nodiscard]] static BoundaryRecord boundaryRecordOf(const Spacetime &spacetime);

  /// One boundary block's `r_U` term: the sum over the register degrees of
  /// `EigenstateSynthesis::targetStateResidual` evaluated on the block's own
  /// sub-complex (`Spacetime::getRegion`) against the block's target. When the
  /// block has no full sub-complex yet, the full leak summed over the degrees.
  [[nodiscard]] double residualForBoundaryBlock(
      const BoundaryBlock &boundaryBlock,
      const std::shared_ptr<Spacetime> &spacetime) const;
  /// The same block term, sharing one `claimedMatchings` set with the rest of the
  /// `r_U` evaluation so this block's register degrees cannot re-use a relabeling
  /// another register already won (see
  /// `EigenstateSynthesis::targetStateResidual`).
  [[nodiscard]] double residualForBoundaryBlockWithDistinctMatchings(
      const BoundaryBlock &boundaryBlock,
      const std::shared_ptr<Spacetime> &spacetime,
      std::set<std::vector<int>> &claimedMatchings) const;
  // Seed one boundary block per (seed, target) — region = the seed's cell-neighbourhood
  // — appended to `destinationBlocks` (shared by seedInputs/seedOutputs). The blocks are
  // grown later by growBlockRegions, not here.
  void seedBlocks(const std::vector<std::uint64_t> &seeds,
                  const std::vector<std::vector<std::complex<double>>> &targets,
                  std::vector<BoundaryBlock> &destinationBlocks);
  /// The shared tail of `seedBlocks` and the region form of `seedInputs`: one
  /// block per (region, target), appended to `destinationBlocks`.
  void seedBlockRegions(const std::vector<std::set<std::uint64_t>> &regions,
                        const std::vector<std::vector<std::complex<double>>> &targets,
                        std::vector<BoundaryBlock> &destinationBlocks, bool surface);

  /// What the bridge phase reads off a complex: the surface input blocks with
  /// their faces and the simplices of those faces, the top cells, and the facet
  /// incidence (top cells per codimension-one face). All are sorted vertex-id
  /// tuples, computed from vertex tuples alone.
  struct SurfaceInventory {
    /// Indices of the surface input blocks.
    std::vector<std::size_t> blocks;
    /// Each block's own faces.
    std::vector<std::set<std::vector<std::uint64_t>>> faces;
    /// Each block's simplices of every dimension.
    std::vector<std::set<std::vector<std::uint64_t>>> simplices;
    /// The complex's top cells.
    std::set<std::vector<std::uint64_t>> topCells;
    /// Top cells per codimension-one face.
    std::map<std::vector<std::uint64_t>, int> facetIncidence;
  };
  /// The surface inventory of \p spacetime.
  [[nodiscard]] SurfaceInventory surfaceInventoryOf(const Spacetime &spacetime) const;
  /// `bridgePhaseComplete` on \p spacetime.
  [[nodiscard]] bool bridgePhaseCompleteOn(const Spacetime &spacetime) const;
  /// `uncoveredInputFaces` on \p spacetime.
  [[nodiscard]] std::vector<std::vector<std::uint64_t>> uncoveredInputFacesOn(
      const Spacetime &spacetime) const;
  /// Every bridge candidate adjacent to the frontier of \p spacetime: for each
  /// frontier face — a boundary facet of the top cells that is not a surface
  /// face, or an uncovered surface face — every completion by one more vertex
  /// such that the cell's part in each of the two blocks is a simplex of that
  /// block's surface, a full face only when it is uncovered. Sorted vertex
  /// tuples, distinct, none already a top cell; empty when the phase is
  /// complete or the node has no surface inputs.
  [[nodiscard]] std::vector<std::vector<std::uint64_t>> bridgeCandidatesOn(
      const Spacetime &spacetime) const;

  /// Record \p spacetime as a `Spacetime::Record`: its top cells as sorted
  /// vertex tuples, followed on a node with surface inputs by the uncovered
  /// surface faces, and every edge's length and phase. Both fields of an edge
  /// are carried, because boundary tori carry pure-gauge link phases and,
  /// under the Whitney pencil, the operator depends on them at every degree.
  [[nodiscard]] Spacetime::Record snapshotOf(const Spacetime &spacetime) const;
  [[nodiscard]] Spacetime::Record snapshot() const;
  /// The complex \p complexSnapshot records (`Spacetime::fromRecord`), carrying
  /// the node's edge-wiring mode. An edge the record does not hold, one a move
  /// created, keeps `Spacetime::fromVertexTuples`'s length and zero phase.
  /// This is the rebuild behind stage-1 candidates, the committed step, the
  /// precone and refinement cone-ins, and checkpoint replay.
  /// so combinatorial moves scored on the result wire their new edges under
  /// the same convention.
  [[nodiscard]] std::shared_ptr<Spacetime> build(
      const Spacetime::Record &complexSnapshot) const;

  /// Draw one random stage-1 move specification on `spacetime`: a `{kind, payload}`
  /// pair where `kind` is one of `add`/`remove`/`flip`/`iflip` (payload = a seed for
  /// the Pachner move), `cone_out`/`cone_in` (payload = the cell/face to cone), or,
  /// on a node with surface inputs whose bridge phase is incomplete, `bridge`
  /// (payload = the cell's vertex ids). The move is only described here, not
  /// applied — see `applyMoveSpecification`.
  [[nodiscard]] MoveSpec drawRandomMoveSpecification(const Spacetime &spacetime);
  /// Apply a move specification from `drawRandomMoveSpecification` to
  /// \p spacetime in place. Returns true when the move was applied and the
  /// result passes the `dualComplexValid` gate at `dualComplexGateDegree_`;
  /// otherwise the caller discards the candidate. Manifold validity is the
  /// whole gate: a move that removes a pinned vertex is accepted when what it
  /// leaves is a valid manifold.
  [[nodiscard]] bool applyMoveSpecification(
      const std::shared_ptr<Spacetime> &spacetime,
      const MoveSpec &moveSpecification);
  [[nodiscard]] double
  deltaF(const std::shared_ptr<Spacetime> &candidateSpacetime,
         double baseObjective, double baseResidualU,
         const std::set<std::vector<std::uint64_t>> &baseCellSet) const;
  /// One best-ΔF batch: \p nCandidateMoves candidates, each a sequence of
  /// \p lookaheadDepth gated random moves applied successively, each drawn
  /// against the evolving candidate, and committed as a whole only when the
  /// best sequence lowers \f$ F \f$.
  ///
  /// Every depth scores by the same localized, unrelaxed `deltaF`. The
  /// combinatorial moves exist to leave a local minimum and the geometric
  /// update to descend within the region the complex then occupies, so scoring
  /// a candidate through a relaxation would ask where a move lands after stage
  /// 2 rather than whether the move improves the state. A committed candidate
  /// is relaxed afterwards, bounded by the caller's `relaxBudgetPerMove`.
  /// Depth 1 pre-draws its batch and scores it in parallel; deeper searches
  /// stay serial, since each draw is made against the evolving candidate.
  ///
  /// \returns the committed ΔF, or 0.
  double step(int nCandidateMoves, int lookaheadDepth, double baseObjective);
  /// Every gated composition of `remainingMoves` combinatorial moves out of
  /// the complex `fromSnapshot` records, scored — as a whole, at the leaf
  /// only — by the same localized `deltaF` a single move is scored by.
  /// Returns the best (ΔF, snapshot) reached, with ΔF = +infinity when no
  /// composition of that length applies. This is the exhaustive counterpart
  /// of the sampled deep path in `step`: each level enumerates against the
  /// complex the previous level left, so the search walks a tree of the
  /// actual move space rather than a product of the base one.
  [[nodiscard]] std::pair<double, Spacetime::Record> bestComposition(
      const Spacetime::Record &fromSnapshot, int remainingMoves, double baseObjective,
      double baseResidualU,
      const std::set<std::vector<std::uint64_t>> &baseCellSet);
  /// One iteration of `runStage1`'s loop: optional boundary growth plus one
  /// best-ΔF candidate-move step, booked into \p objectiveTrace. A batch with
  /// no improving move is not a stall, since the batch is a random sample and
  /// the next iteration redraws.
  /// \returns whether the caller should keep iterating. Target-conditioned
  ///   modes continue until the register is carried; target-free
  ///   `JointStationarity` stops after the stalled batch.
  bool stage1Update(int nCandidateMoves, bool growBoundaries,
                    std::vector<double> &objectiveTrace, int maxLookahead = 1,
                    int combinatorialBreadth = 0);
  /// One iteration of `runStage2`: assemble the selected objective's complex-z
  /// ascent direction, subtract it from z, and run the backtracking line search.
  /// Appends an accepted objective and adapts `stepScale`; otherwise restores the
  /// original length branches and reports stationarity.
  bool stage2Update(double beta, double tolerance,
                    std::vector<double> &objectiveTrace, double &stepScale);
  /// Expand each not-yet-carrying block's scoring region by one shell — the
  /// vertices of every top cell touching it — so the window a block's residual
  /// is read over gains room for the holes that carry its state. Applies to
  /// every input block and every localized output block; a single output reads
  /// off the whole complex and has no block here.
  ///
  /// Two conditions bound the growth. A shell is kept only when it strictly
  /// lowers that block's residual, and growth happens only before the first
  /// committed combinatorial move, since once the bulk is being linked the
  /// states' read windows are settled. Without both, growth has no stopping
  /// point: a block that is not carrying scores the same constant full leak at
  /// any region size, so every shell is an exact tie and the regions grow until
  /// they cover the whole complex and all blocks read one identical
  /// sub-complex.
  ///
  /// Creates no cells, edges or vertices, and never moves the cobordism's
  /// boundary: the only write is each block's vertex set.
  void growBlockRegions();
  /// Pre-grow the seed by \p count gated cone-in moves before any optimization;
  /// the constructor calls this once when `precone > 0`. Each move cones a
  /// fresh apex onto a random codimension-one facet of a random top cell and is
  /// committed only through `applyMoveSpecification`'s `dualComplexValid` gate,
  /// the same gate stage 1 uses. It enlarges the complex so that surgery has
  /// room to act.
  ///
  /// \p count <= 0 is a no-op and leaves the RNG untouched. A draw onto an
  /// already-saturated facet is rejected by the gate and retried; if no valid
  /// cone-in is found for a cell, the pass stops early. \p timelike draws every
  /// cone timelike, \p alternate interleaves timelike and spacelike and takes
  /// precedence over \p timelike, and with neither set the precone is
  /// all-spacelike.
  void preconeCells(int count, bool timelike = false, bool alternate = false);

  std::shared_ptr<Spacetime> spacetime_;
  std::vector<std::vector<std::complex<double>>> inputTargets_;
  std::vector<std::vector<std::complex<double>>> outputTargets_;
  /// The register degrees \f$ k \f$ the objective scores at once; every `rU`
  /// term is summed over these, and a \f$ b_k \f$ register is forced to emerge
  /// for each.
  std::vector<int> registerDegrees_;
  /// The single degree at which the `dualComplexValid` move gate runs: the
  /// maximum register degree, since the degree-free validity check needs only
  /// the coarsest one.
  int dualComplexGateDegree_;
  double gamma_;
  /// The caller-declared pinned regions (`declarePinnedRegion`). Empty by
  /// default and never derived from boundary blocks or their targets: each
  /// region is a combinatorial declaration that constrains the geometry rather
  /// than gating any move.
  std::vector<PinnedRegion> pinnedRegions_;
  /// Ordered exact-period state constraints: explicit fixtures over a
  /// caller-supplied topology, separate from emergent block matching and from
  /// geometric pinning.
  std::vector<RegisterConstraint> registerConstraints_;
  /// Propagated to every spacetime this node constructs: the host before the
  /// precone, and each candidate snapshot rebuild.
  bool balancedEdgeWiring_{false};
  /// `rU`'s whole-complex term is `singularValueHalfSumRatio` instead of the
  /// period residual and `nearKernelResidual` pair.
  bool singularValueRatio_{false};
  /// False drops \f$ \|\nabla S_{\rm Regge}\|^2 \f$ from every objective site.
  bool einsteinHilbert_{true};
  /// Restrict stage-2 updates to real squared lengths. False preserves the
  /// general complexified geometry.
  bool realSquaredLengthsOnly_{false};
  /// The metric source of every Hodge operator this node builds (see `metricSource()`).
  HodgeLaplacian::MetricSource metricSource_{HodgeLaplacian::MetricSource::WhitneyPencil};
  /// The injected functional, and the only record of what this node descends.
  /// Never null: the constructor installs `LegacyObjective`.
  std::shared_ptr<CobordismObjective> objectiveSpec_;
  /// The optional additional objective holding a pinned region. Null means the
  /// pinned region's objective is the bulk objective, i.e. a single-objective
  /// run.
  std::shared_ptr<CobordismObjective> pinnedObjectiveSpec_;
  /// Assemble the firewalled input an objective reads. Private because the
  /// bound evaluators close over this node; an objective receives the
  /// assembled context and can reach nothing beyond it.
  ///
  /// The objective is passed so its declared scope can be resolved into the
  /// context's region and edge list. The engine reads the declaration rather
  /// than inferring a scope from the objective's role.
  [[nodiscard]] ObjectiveContext objectiveContextFor(
      const std::shared_ptr<Spacetime> &spacetime,
      const std::shared_ptr<CobordismObjective> &objective) const;
  /// The bulk objective's context, which is the whole cobordism.
  [[nodiscard]] ObjectiveContext objectiveContextFor(
      const std::shared_ptr<Spacetime> &spacetime) const;
  /// Whether the scalar this node reports admits a localized exact delta. This
  /// is not simply the bulk objective's declaration: with a pinned objective in
  /// force the reported scalar is a sum over two scopes, and differencing the
  /// bulk alone would score a surrogate that is not the objective.
  [[nodiscard]] bool compositeSupportsLocalizedDelta() const;
  /// Throw unless this node can honour everything the objective declares: its
  /// register-degree domain, and a scope naming a region this node has
  /// declared. A handle cannot be mis-spelled, but a region can be cleared
  /// after one was minted, and an objective pointing at a region that no longer
  /// exists must fail loudly rather than silently score nothing. Shared by the
  /// bulk and pinned injection points so neither can drift into accepting what
  /// the other refuses.
  void requireObjectiveAcceptable(
      const std::shared_ptr<CobordismObjective> &objective) const;
  /// The edge indices a region-scoped objective's sums run over.
  ///
  /// An edge is interior to the region when both endpoints lie in it, and
  /// straddling when exactly one does. Interior edges are always included;
  /// straddling edges only where the objective declared them so. The border is
  /// the one the node already defines, since `edgeIsPinned` holds exactly when
  /// a single region contains both endpoints.
  [[nodiscard]] std::vector<std::size_t> scopedEdgeIndices(
      const std::shared_ptr<Spacetime> &spacetime,
      const std::set<std::uint64_t> &region,
      bool includesStraddlingEdges) const;
  HodgeLaplacian::EntropyPhaseMode hodgeEntropyPhaseMode_{
      HodgeLaplacian::EntropyPhaseMode::IncludeComplexPhase};
  /// The declared Hodge degrees. Defaults to the degree-zero Laplacian alone
  /// and is never populated from `registerDegrees_`.
  std::vector<int> hodgeDegrees_{0};
  /// Per-degree weights, empty for uniform.
  std::vector<double> hodgeDegreeWeights_;
  double hodgeEntropyWeight_{1.0};
  double connectionEntropyWeight_{0.0};
  double reggeWeight_{1.0};
  /// Latched by the first committed combinatorial move. Block regions grow only
  /// before the bulk is connected, so once a move has linked the complex up the
  /// boundary states' read windows are settled.
  bool bulkConnected_{false};
  /// The scalar objective on \p spacetime. Evaluated in one place, so that
  /// `objective`, the stage-2 acceptance test and `deltaF` cannot disagree
  /// about what \f$ F \f$ is.
  [[nodiscard]] double objectiveFor(
      const std::shared_ptr<Spacetime> &spacetime) const;
  /// Weight on the input-block residual terms in `rU`
  /// (`setInputResidualWeight`).
  double inputResidualWeight_ = 1.0;
  /// Whether blocks with a fiber target are scored by the fiber residual.
  bool useFiberResiduals_{false};
  /// Whether the whole-complex leak is added to a marked block's residual.
  bool scoreWholeComplexLeak_{false};
  /// Whether stage 2 descends the degree-0 link phases.
  bool fiberPhaseDescent_{false};
  /// The whole-complex fiber target, when one is set.
  std::optional<BoundaryFiber> wholeFiberTarget_;
  /// The single two-body target, when one is set.
  std::optional<TwoBodyTarget> twoBodyTarget_;
  /// The declared two-body cases, in the order they were set.
  std::vector<TwoBodyCase> twoBodyCases_;
  /// Write \p boundaryCase's boundary metric onto \p spacetime, returning what
  /// was there so the caller can put it back. Const because the complex is held
  /// by pointer and the cases read it under several boundary conditions rather
  /// than change it: every writer restores.
  [[nodiscard]] std::vector<std::pair<std::pair<std::uint64_t, std::uint64_t>,
                                      std::complex<double>>>
  writeCaseBoundary(const TwoBodyCase &boundaryCase,
                    const std::shared_ptr<Spacetime> &spacetime) const;
  /// Restore exact complex edge lengths returned by `writeCaseBoundary`.
  /// Storing lengths rather than squared lengths preserves the square-root
  /// branch the live geometry carried before the case read.
  void restoreCaseBoundary(
      const std::vector<std::pair<std::pair<std::uint64_t, std::uint64_t>,
                                  std::complex<double>>> &lengths,
      const std::shared_ptr<Spacetime> &spacetime) const;
  /// `setBoundaryMayExtend`; false refuses a cone that grows a declared
  /// boundary. Nodes without one (`hasFixedBoundary`) are unaffected either way.
  bool boundaryMayExtend_{false};
  /// The transfer between the two attached input blocks on \p spacetime: in
  /// the blocks' frames when both carry one (`transferOperand`: derived live
  /// from a marking, or the supplied `BlockFrame`), in the full (identity)
  /// frames on the fibers' cells when neither does; refused geometries throw
  /// std::runtime_error.
  /// @throws std::logic_error when only one block carries a frame.
  [[nodiscard]] chainhodge::TransferResult frameTransferOn(
      const std::shared_ptr<Spacetime> &spacetime, const BoundaryBlock &A,
      const BoundaryBlock &B) const;
  /// One operand of the transfer on \p spacetime: the block's frame in effect —
  /// derived live from its marking when it carries one (`deriveFrame`; an
  /// obstructed derivation throws `std::runtime_error`, which the residual
  /// scores as the full leak), the supplied `BlockFrame` otherwise, or none —
  /// together with the cells the operand is placed on, which are the frame's
  /// cells when framed and the attached fiber's otherwise.
  struct TransferOperand {
    /// The cells the operand is placed on.
    std::vector<std::vector<std::uint64_t>> cells;
    /// The frame in effect, or empty when the operand is unframed.
    std::optional<BlockFrame> frame;
    /// True when the frame was derived live from the block's marking.
    bool derived{false};
  };
  /// The transfer operand of \p block on \p spacetime.
  [[nodiscard]] TransferOperand transferOperand(const BoundaryBlock &block,
                                                const std::shared_ptr<Spacetime> &spacetime) const;
  /// The target edge values of a marked block through a frame: a degree-1
  /// fiber on the frame's cells with images \f$ F\,(a, b)^T \f$.
  [[nodiscard]] static BoundaryFiber stateTargetOf(const BlockMarking &marking, const BlockFrame &frame);
  /// The transfer between two paired frames: each side's blocks stacked into one
  /// frame, with cells concatenated and images block-diagonal.
  ///
  /// With a conjugate pair per side each frame has rank 4, so the transfer is
  /// \f$4 \times 4\f$. Its coordinates transform as a direct sum rather than as
  /// the tensor product of two qubits; the two must not be identified from
  /// dimension alone.
  [[nodiscard]] chainhodge::TransferResult pairedFrameTransferOn(
      const std::shared_ptr<Spacetime> &spacetime,
      const std::vector<const BoundaryBlock *> &sideA,
      const std::vector<const BoundaryBlock *> &sideB) const;
  /// The projective leak of \p target against the frame transfer
  /// \f$T_{AB}\f$ — the reading `ReadoutMode::Transfer` selects. Returns the
  /// full leak 1.0 when the geometry is refused.
  [[nodiscard]] double transferResidualOn(
      const std::shared_ptr<Spacetime> &spacetime,
      const TwoBodyTarget &target) const;
  /// The gradient of the selected two-body residual, summed over the readings
  /// that implement one. Used by `fiberModeAscent`; bulk and paired-operator
  /// readings are refused so stage 2 can choose its numerical fallback.
  [[nodiscard]] ResidualGradient selectedTwoBodyResidualGradientOn(
      const std::shared_ptr<Spacetime> &spacetime,
      const TwoBodyTarget &target) const;
  /// The two-body residual of \p target on \p spacetime, summed over the
  /// selected readings.
  [[nodiscard]] double twoBodyResidualOn(const std::shared_ptr<Spacetime> &spacetime,
                                         const TwoBodyTarget &target) const;
  /// The two input blocks carrying an attached fiber, in block order.
  /// @throws std::logic_error unless exactly two do.
  [[nodiscard]] std::pair<const BoundaryBlock *, const BoundaryBlock *> attachedInputBlocks() const;
  /// The shape of the transfer between two attached input blocks: the frames'
  /// ranks when both carry one, the cell counts when neither does. Empty when
  /// only one does, in which case the shape is settled at read time, or when
  /// fewer than two fibers are attached.
  struct TransferShape {
    /// Row count of the transfer.
    Eigen::Index rows{0};
    /// Column count of the transfer.
    Eigen::Index cols{0};
    /// True when both blocks carry a frame.
    bool framed{false};
  };
  /// The transfer shape on the live complex.
  [[nodiscard]] std::optional<TransferShape> transferShape() const;
  /// The direct-sum transfer shape after splitting an even number of attached
  /// blocks into two sides. Empty when the split is unavailable, or when only
  /// some blocks carry frames, which the read itself refuses.
  [[nodiscard]] std::optional<TransferShape> pairedTransferShape() const;

  /// The fiber residual of \p fiber read on \p spacetime in \p band (see
  /// `useFiberResiduals`, `FiberBand`).
  [[nodiscard]] double fiberResidualOn(const std::shared_ptr<Spacetime> &spacetime,
                                       const BoundaryFiber &fiber,
                                       FiberBand band = FiberBand::AsStored) const;
  /// The block's region of \p spacetime (`Spacetime::getRegion`), which carries
  /// the parent's edge lengths and phases, with its facets materialized for the
  /// fiber reads. Null when the region holds no top cell.
  [[nodiscard]] static std::shared_ptr<Spacetime> blockSubcomplexWithGeometry(
      const BoundaryBlock &block, const std::shared_ptr<Spacetime> &spacetime);
  /// The complex a block's fiber is read on: a surface block's own surface
  /// (`blockSurfaceWithGeometry`), an ordinary block's sub-complex
  /// (`blockSubcomplexWithGeometry`). Null when there is nothing to read.
  [[nodiscard]] static std::shared_ptr<Spacetime> blockComplexWithGeometry(
      const BoundaryBlock &block, const std::shared_ptr<Spacetime> &spacetime);
  /// An input region stops growing once its residual drops below this, i.e.
  /// once it carries its state.
  double inputCarriedTolerance_ = 1e-12;
  /// The move and restart random source driving stage 1 and block construction.
  std::mt19937_64 randomNumberGenerator_;
  /// Whether the move draw offers the causal-disposition moves.
  bool shouldProposeDispositions_{true};
  /// Whether the move draw and the enumeration offer the surgical kinds.
  bool shouldProposeSurgery_{true};
  /// What the two-body target is scored against (`setReadoutModes`).
  std::vector<ReadoutMode> readoutModes_{ReadoutMode::Transfer};
  /// The state `ReadoutMode::Whole` scores against (`setOutputStateTarget`).
  std::optional<Eigen::VectorXcd> outputStateTarget_;
  /// Why the last harmonic readout could not name a state (empty when it
  /// could). Mutable because the readout is a const measurement that still has
  /// to be able to say why it refused.
  mutable std::string wholeHarmonicObstruction_;
  double convergenceTolerance_ = 1e-9;
  /// Set by `runStage2`: true when its last call stopped on the
  /// absolute-tolerance stationarity test, false when it hit the iteration
  /// budget.
  bool lastStage2Stationary_ = false;
  /// Set by `stage1Update`: the committed sequence's lookahead depth. 0 means
  /// the update committed nothing.
  int lastStage1LookaheadDepth_ = 0;
  std::vector<BoundaryBlock> inputBlocks_;
  WholePairing wholePairing_{WholePairing::Periods};
  std::vector<BoundaryBlock> outputBlocks_;

  // ---- analysis-overlay state ----
  //
  // None of the analysis members below is read by `objectiveFor`,
  // `objectiveTermsFor`, `deltaF`, `rU`, `step`, `stage1Update` or
  // `stage2Update`. The objective touches only the carried state
  // (`carriedModeCells_`, `carriedCovariance_`, `carriedStateEnergyWeight_`),
  // and only through the `ObjectiveTerms::carriedStateEnergy` scalar, and only
  // while the run declares `EmergenceSubmode::CertificatesBlindMeanField`.
  SimulationMode simulationMode_{SimulationMode::Emergence};
  EmergenceSubmode emergenceSubmode_{EmergenceSubmode::Strict};
  /// The carried modes' `carriedStateDegree_`-cells, by vertex tuple.
  std::vector<std::vector<std::uint64_t>> carriedModeCells_{};
  /// \f$ \Gamma \f$, flat row-major over the carried modes.
  std::vector<std::complex<double>> carriedCovariance_{};
  int carriedStateDegree_{1};
  double carriedStateEnergyWeight_{0.0};
  // The spectral-moment stiffness: its weight, degrees, order weights, and the
  // carrier's local moments recorded when it was declared.
  double momentStiffnessWeight_{0.0};
  std::vector<int> momentStiffnessDegrees_;
  std::vector<double> momentStiffnessCoefficients_;
  std::vector<std::vector<std::complex<double>>> momentStiffnessReference_;
  double meanFieldStepSize_{0.0};
  int meanFieldSteps_{0};
  /// Thresholds for `refinementDecisionOf`, all zero. The indicator struct's
  /// own defaults describe a healthy complex (`meshQuality` 1), which as a
  /// lower bound would fire on every real mesh; zero means "never" in both
  /// senses, so an unconfigured node never refines.
  RefinementIndicators refinementThresholds_{0.0, 0.0, 0.0, 0.0, 0.0};
  /// |ΔF| of the last accepted stage-2 update: the solver-error indicator.
  double lastStage2Improvement_{0.0};
  AnalysisConfig analysisConfig_{};
  std::string provenanceConfigHash_{};
  std::string provenanceCommit_{};
  std::uint64_t seed_{0};
  std::uint64_t acceptedMoveCount_{0};
  std::uint64_t analysisPassCount_{0};
  /// The cell set of the complex the last analysis pass saw, differenced
  /// against the current one to publish the accepted move's touched star.
  std::set<std::vector<std::uint64_t>> analysisCellSet_{};
  /// The edge lengths the last analysis pass saw, so a pure metric change
  /// publishes only the edges that moved and disjoint siblings stay served from
  /// cache.
  std::map<std::pair<std::uint64_t, std::uint64_t>, std::complex<double>>
      analysisEdgeLengths_{};
  /// Set true once `analysisCellSet_` holds a real observation.
  bool analysisCellSetValid_{false};
  /// The analytic cache the overlay reuses across passes, so a local change
  /// invalidates only the entries whose component meets the published star and
  /// disjoint siblings stay served. Held as `void` to keep this header's
  /// include list small, and rebound whenever `spacetime_` becomes a different
  /// object, which every committed combinatorial move causes.
  std::shared_ptr<void> analysisCache_{};
  /// The spacetime `analysisCache_` is bound to (identity comparison only).
  std::weak_ptr<Spacetime> analysisCacheBinding_{};
  /// The retained cobordism frames of the overlay (`AnalysisConfig::
  /// frameHistory`): what makes a lifetime, an adjacent-frame overlap and a
  /// lifetime transport family measurable rather than assumed. Null until the
  /// first pass.
  std::shared_ptr<AnalysisFrameHistory> analysisFrames_{};
  /// The last pass's checkpoint document.
  std::string checkpointJson_{};

  /// Run the overlay when the configuration asks for it. Called only after a
  /// move has been committed, so it cannot influence that move.
  void noteAcceptedMove();
  /// The overlay pass body, implemented in
  /// `src/cobordism/RecursiveFiberSimulation.cpp`.
  void runRecursiveAnalysisOn(const std::shared_ptr<Spacetime> &spacetime);
  /// Serialize the current raw complex and edge data for the checkpoint.
  [[nodiscard]] std::string rawComplexJson(
      const std::shared_ptr<Spacetime> &spacetime) const;
  /// The one-particle generator \f$ h_S(g) \f$ of the carried modes on
  /// \p spacetime, flat row-major over the carried modes. Entries for absent
  /// cells are 0.
  [[nodiscard]] std::vector<std::complex<double>> carriedStateGenerator(
      const std::shared_ptr<Spacetime> &spacetime) const;
};

}  // namespace tessera::cobordism

#endif  // TESSERA_COBORDISM_MULTICOBORDISM_H
