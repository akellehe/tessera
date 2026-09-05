// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_COBORDISM_MULTICOBORDISM_H
#define TESSERA_COBORDISM_MULTICOBORDISM_H

#include <complex>

#include <Eigen/Core>

#include "cobordism/CobordismObjective.h"
#include "cobordism/HodgeLaplacian.h"
#include "cobordism/PencilLayer.h"
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

namespace tessera::cobordism {
using ::tessera::spacetime::Spacetime;

/// # MultiCobordism
///
/// The C++ source-of-truth emergent-merge optimizer
/// (epic #457 / T5, #491): the merge as a **fully emergent** optimization — no
/// prescribed topology, no hand-placed register. From a bare host it grows the
/// register by **gated surgical moves** under the objective and reads the register
/// **dynamically** off `getBoundary` at a **user-defined degree k**.
///
/// The scalar objective is selected explicitly. `Legacy` preserves
/// \f$\|\nabla S_{\rm Regge}\|^2+\Gamma r_U\f$; `JointStationarity` uses
/// \f$\beta\|\nabla_zS_{\rm Regge}\|^2+
/// \eta\|\nabla_zS_{\rm Hodge}\|^2\f$; and `MediatedCorrespondence` reproduces
/// the documented \f$r_U+\beta|S_{\rm Regge}(W^*)|\f$ experiment.
///
/// Two stages, exactly as the reference:
///   * **Stage 1 (combinatorial):** greedy best-ΔF single random moves
///     `{add,remove,flip,iflip,cone_out,cone_in,cone_in_timelike,flip_disposition}`
///     (the last two are the causal dispositions — see `shouldProposeDispositions`),
///     each gated by `dualComplexValid` and "no input vertex removed", committed
///     only if ΔF < 0. Target-conditioned modes may redraw a stalled batch while
///     the register is not carried; target-free `JointStationarity` stops that
///     stage when no improving sequence is found.
///   * **Stage 2 (geometric):** relax the full complex squared edge coordinates
///     \f$z_e=\ell_e^2\f$ along the selected objective's gradient, then map each
///     accepted \f$z_e\f$ back to the continuous square-root branch of the stored
///     edge length \f$\ell_e\f$.
class MultiCobordism {
 public:
  /// An emergent boundary block of the cobordism — an input OR an output. A block is
  /// NOT itself a complex: it stores the vertex SET it occupies plus the target period
  /// vector its own `L_k` sub-complex must carry. The sub-complex is recovered on
  /// demand from `vertices` by `subcomplexWithinVertexSet` (the ambient complex's top
  /// cells whose vertices all lie in the set), so the vertex set — together with the
  /// ambient triangulation — determines the block's complex.
  struct BoundaryBlock {
    std::set<std::uint64_t> vertices;
    std::vector<std::complex<double>> target;
    /// The fiber form of the target (#916): a retained fiber on the block's
    /// degree-k cells with its Gram, band eigenvalue, contour, and certificate,
    /// beside the period vector. Set on an input block by `setInputFiber`
    /// (pinned as boundary data by `pinInputFibers`); read on an output block
    /// by `readOutputFiber` after relaxation.
    std::optional<BoundaryFiber> fiber;
  };

  /// An ordered, explicit period constraint evaluated by the existing
  /// EigenstateSynthesis::residualForPeriods residual. Unlike emergent target
  /// matching, no component permutation is permitted: target[i] belongs to
  /// holes[i]. This is the fixed input/output state pin used by operator
  /// experiments on a caller-supplied topology.
  struct RegisterConstraint {
    std::string name;
    int degree{1};
    std::vector<std::vector<std::uint64_t>> holes;
    std::vector<std::complex<double>> target;
  };

  /// Result of the historical fixed-boundary spectral relaxation. This is a
  /// direct inverse-eigenvector synthesis: selected cochain components have the
  /// fixed relative amplitudes `target`, all other amplitudes are free, and only
  /// interior edge geometry varies. The assembled cochain is globally
  /// normalized before evaluation, so the selected block represents a ray, not
  /// an absolute norm. This is distinct from the later period-based `rU`
  /// objective.
  struct FixedBoundaryEigenstateResult {
    bool converged{false};
    double residual{0.0};
    double eigenvalue{0.0};
    int degree{0};
    int growthSteps{0};
    std::size_t interiorVertexCount{0};
    std::size_t interiorEdgeCount{0};
    std::size_t auxiliaryCellCount{0};
    std::vector<std::vector<std::uint64_t>> supportCells;
    std::vector<std::complex<double>> target;
    std::vector<std::complex<double>> state;
  };

  /// Result of a boundary-value spectral transfer solve. Each witness carries
  /// one independently prepared input/output pair on two distinct components
  /// of the boundary of W. Boundary amplitudes and geometry are fixed; only
  /// bulk geometry and interior cochain amplitudes vary.
  struct BoundaryStateTransferResult {
    bool converged{false};
    bool commonEigenvalue{true};
    double residual{0.0};
    double eigenvalue{0.0};
    int degree{0};
    int growthSteps{0};
    std::size_t freeEdgeCount{0};
    std::size_t auxiliaryCellCount{0};
    std::string inputRegion;
    std::string outputRegion;
    std::vector<std::vector<std::uint64_t>> inputCells;
    std::vector<std::vector<std::uint64_t>> outputCells;
    std::vector<std::vector<std::complex<double>>> inputStates;
    std::vector<std::vector<std::complex<double>>> outputStates;
    std::vector<std::vector<std::complex<double>>> states;
    std::vector<double> stateResiduals;
    std::vector<double> stateEigenvalues;
    std::vector<double> inputBoundaryResiduals;
    std::vector<double> outputBoundaryResiduals;
    /// Best coupled residual after each relaxation/growth pass.
    std::vector<double> residualTrace;
  };

  /// A formal complex-coefficient degree-\f$k\f$ chain on the live complex.
  /// Its readout of a \f$k\f$-cochain \f$\psi\f$ is the chain–cochain pairing
  /// \f$\langle c,\psi\rangle=\sum_\sigma c_\sigma\,\psi(\sigma)\f$ over the
  /// listed cells (vertex sets; the value is taken on the ascending-order
  /// cell). A unit chain reads one cell, a \f$\pm1\f$ chain over a cycle reads
  /// a period, and a dense chain pairs with a reference state.
  using ReadoutChain =
      std::vector<std::pair<std::vector<std::uint64_t>, std::complex<double>>>;

  /// Result of a whole-complex readout relaxation: a spanning set of coupled
  /// eigenstate witnesses, each with both boundary components' amplitudes
  /// fixed as inputs and its whole-complex readouts fixed to the algebraic
  /// output. The readout constraints hold exactly on every witness; the
  /// residual measures only whether the whole complex carries the witnesses
  /// as eigenstates at the common eigenvalue.
  struct WholeComplexReadoutResult {
    bool converged{false};
    bool commonEigenvalue{true};
    double residual{0.0};
    double eigenvalue{0.0};
    int degree{0};
    int growthSteps{0};
    std::size_t freeEdgeCount{0};
    /// Free amplitude coordinates per witness after the readout constraints
    /// are eliminated (interior cells minus the readout rank).
    std::size_t auxiliaryCellCount{0};
    std::size_t readoutRank{0};
    std::string regionA;
    std::string regionB;
    std::vector<std::vector<std::uint64_t>> cellsA;
    std::vector<std::vector<std::uint64_t>> cellsB;
    /// The fixed boundary amplitudes after the joint per-witness
    /// normalization (component A, then component B).
    std::vector<std::vector<std::complex<double>>> statesA;
    std::vector<std::vector<std::complex<double>>> statesB;
    /// The readout targets scaled by the same per-witness factor.
    std::vector<std::vector<std::complex<double>>> targets;
    /// The readouts of the returned witnesses; `readoutDeviation` is their
    /// largest absolute difference from `targets` (round-off only).
    std::vector<std::vector<std::complex<double>>> readouts;
    double readoutDeviation{0.0};
    std::vector<std::vector<std::complex<double>>> states;
    std::vector<double> stateResiduals;
    std::vector<double> stateEigenvalues;
    std::vector<double> boundaryResidualsA;
    std::vector<double> boundaryResidualsB;
    /// Best coupled residual after each relaxation/growth pass.
    std::vector<double> residualTrace;
  };

  /// Target-free Choi promotion of the live metric
  /// \f$\ker L_1(W-\partial W)\f$ restricted to an ordered \f$d^2\f$ frame.
  /// identifiable is true only when that restriction has rank one. A
  /// multidimensional restriction is an operator family, not a Choi state, and
  /// is returned as an explicit obstruction rather than by reshaping an
  /// arbitrary kernel basis vector.
  struct GeometricOperatorReadout {
    bool identifiable{false};
    std::string obstruction;
    int stateDimension{0};
    bool metric{true};
    std::size_t bulkCellCount{0};
    std::size_t kernelDimension{0};
    std::size_t frameRank{0};
    double unitarityError{0.0};
    std::vector<double> frameSingularValues;
    std::vector<std::vector<std::uint64_t>> bulkCells;
    std::vector<std::vector<std::uint64_t>> frameCells;
    /// Unit-norm, phase-fixed Choi state. Empty unless identifiable.
    std::vector<std::complex<double>> choiState;
    /// \f$\sqrt d\,\operatorname{unvec}(|J\rangle)\f$, row-major. This is the
    /// unitary Choi normalization; unitarityError reports whether the inferred
    /// ray actually satisfies that assumption. Empty unless identifiable.
    std::vector<std::complex<double>> operatorMatrix;
  };

  /// `outputTargets` is a LIST of output boundary blocks (the full cobordism
  /// `∂W = inputs ⊔ outputs`, #491): a merge has one, a 2→2 recombination has two
  /// (diquark ⊔ antidiquark). Each output — like each input — is an emergent
  /// boundary sub-complex carrying its target. Target-conditioned modes score it
  /// through `r_U`; JointStationarity retains it only as readout metadata. The
  /// bulk routes the connectivity (which input constituent reaches which output).
  ///
  /// An **empty** `outputTargets` is a supported shape (#555): nothing is pinned
  /// downstream. In target-conditioned modes `rU` then sums only the input
  /// blocks; JointStationarity contains no `rU` term. Whatever the whole comes
  /// to carry is read after the fact—the emergent arm `ProtonIngredients` builds
  /// on this.
  ///
  /// `precone` (default 0) pre-grows the host by that many **gated cone-in moves**
  /// before any optimization — the emergent way to give surgery room to act, in
  /// place of a prebuilt host refinement. Each cone-in adds one top cell on a fresh
  /// apex over a random facet and is accepted only through the `dualComplexValid`
  /// gate (see `preconeCells`); on the single-Δ⁴ seed (a 4-ball) this enlarges the
  /// 4-ball. Reproducible given `seed`; `precone = 0` leaves the host untouched.
  /// `preconeTimelike` draws every precone cone-in as the TIMELIKE disposition
  /// (#613, apex edges ℓ² = −1); `preconeAlternate` instead ALTERNATES the
  /// cone-ins timelike/spacelike for balanced causal content at one uniform
  /// edge-length magnitude (it wins when both are set). Defaults keep the
  /// all-spacelike precone.
  /// `einsteinHilbert` (default true) keeps the Regge term selected by the
  /// objective mode. False removes that term: JointStationarity becomes Hodge
  /// entropy stationarity alone, MediatedCorrespondence becomes `rU`, and Legacy
  /// becomes `gamma*rU`. Stage 2 differentiates the remaining scalar objective.
  /// realSquaredLengthsOnly (default false) restricts stage 2 to the real
  /// \f$\ell^2\f$ locus. It is intended for fixed-signature residual-only
  /// experiments; the default retains the complexified relaxation used by the
  /// general simulator.
  ///
  /// `singularValueRatio` swaps the WHOLE-COMPLEX term of `rU` — both regimes:
  /// the single-output period residual and its `nearKernelResidual`
  /// continuation — for the scale-invariant singular-value half-sum ratio
  /// (`singularValueHalfSumRatio`). The input-block residuals keep anchoring
  /// the input states; nothing prescribes WHAT the whole comes to carry.
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

  /// Where every Hodge operator this node scores, relaxes, and reads takes its
  /// metric from. Defaults to the process-wide `HodgeLaplacian::defaultMetricSource()`
  /// read at construction, so a node, the static readouts, the observables, the
  /// capstone, and checkpoint replay always agree: an experiment on the
  /// chain-level Whitney pencil flips that default ONCE at startup
  /// (`HodgeLaplacian::setDefaultMetricSource(WhitneyPencil)`), exactly as the
  /// weight convention is flipped. Under `WhitneyPencil` the operator is
  /// \f$ h_k(s,U) \f$ of the complex squared edge lengths and the edge-phase
  /// links at EVERY degree; `DiagonalWeights` is the historical per-simplex
  /// diagonal metric, unchanged.
  [[nodiscard]] HodgeLaplacian::MetricSource metricSource() const noexcept {
    return metricSource_;
  }

  /// Configuration-space admissibility of a geometry under the Whitney pencil:
  /// the closure of the Kontsevich–Segal allowable domain, i.e. margin
  /// \f$ \ge 0 \f$ (the real Lorentzian boundary, margin exactly zero, is
  /// admitted and certified as the boundary). Not a clamp, back-off, or
  /// penalty: a proposal outside it is not a member of the configuration
  /// space, exactly as a non-manifold proposal is not. Always true under
  /// `DiagonalWeights`.
  [[nodiscard]] bool geometryAdmissible(const std::shared_ptr<Spacetime> &spacetime) const;

  [[nodiscard]] bool einsteinHilbertEnabled() const noexcept {
    return einsteinHilbert_;
  }
  [[nodiscard]] bool realSquaredLengthsOnly() const noexcept {
    return realSquaredLengthsOnly_;
  }

  /// Declare an ordered exact-period constraint, replacing a constraint with
  /// the same name. Every hole must contain degree + 2 distinct vertices and
  /// the hole count must equal the target width.
  /// @throws std::invalid_argument on malformed input.
  void declareRegisterConstraint(RegisterConstraint constraint);

  [[nodiscard]] const std::vector<RegisterConstraint> &registerConstraints()
      const noexcept {
    return registerConstraints_;
  }

  /// Remove every explicit register constraint. Emergent input/output targets
  /// are unaffected.
  void clearRegisterConstraints();

  /// Run the fixed-boundary inverse-eigenvector relaxation used by the
  /// pre-paper realizability report. Before global state normalization,
  /// `supportCells[i]` is pinned to `target[i]`; every other cochain component
  /// is an optimized auxiliary amplitude. Thus the normalized witness restricts
  /// to the target ray, while its support norm may be smaller than one. The
  /// target block is normalized once before optimization.
  ///
  /// Only `EigenstateSynthesis` interior weights and, at degree zero, interior
  /// U(1) phases are varied. The full geometric boundary is therefore held
  /// fixed. If the Rayleigh residual
  /// \f$\|L\psi-\langle\psi,L\psi\rangle\psi\|^2\f$ does not fall below
  /// `epsilon`, a boundary-preserving stellar subdivision is attempted and the
  /// relaxation repeats, up to `maxGrowth` times. No Regge term, period
  /// residual, charge constraint, or harmonic condition enters this mode.
  ///
  /// This method mutates the node's live spacetime in place. The ordinary
  /// `run`/`buildStep` implementations and their objectives are unchanged.
  [[nodiscard]] FixedBoundaryEigenstateResult relaxFixedBoundaryEigenstate(
      int degree,
      std::vector<std::vector<std::uint64_t>> supportCells,
      std::vector<std::complex<double>> target, double epsilon = 1e-10,
      int restarts = 64, int maxGrowth = 4, std::uint64_t seed = 0,
      int maxIterations = 200);

  /// Fit one shared bulk geometry to independently prepared boundary-state
  /// pairs. `inputRegionName` and `outputRegionName` must name two declared
  /// pinned regions whose vertex sets are exactly the two connected components
  /// of \f$\partial W\f$. `inputCells` and `outputCells` must enumerate every
  /// degree-`degree` cell of the corresponding component; each row of
  /// `inputStates` is normalized once and its paired output is scaled by the
  /// same factor, preserving the relative amplitudes of the supplied linear
  /// map. Both restrictions are then fixed on their ordered cell frames.
  ///
  /// Before fitting, every supplied state is required to have isolated-boundary
  /// eigenresidual below `boundaryEpsilon`. During fitting those boundary
  /// amplitudes remain exact in the returned, unnormalized witness cochains.
  /// Every other cochain component is an independent auxiliary amplitude for
  /// that pair. Edges held by any declared pinned region remain bit-identical;
  /// all other edge weights and, at degree zero, connection phases may vary.
  ///
  /// With `commonEigenvalue=true` (the operator-transfer default), the objective
  /// is
  /// \f[
  ///   R=\sum_j\|L_W\widehat\psi_j-\bar\lambda\widehat\psi_j\|^2,
  ///   \qquad
  ///   \bar\lambda=\frac1m\sum_j
  ///     \langle\widehat\psi_j,L_W\widehat\psi_j\rangle .
  /// \f]
  /// A converged witness span is therefore closed under linear combinations:
  /// attaching a new input in the fitted input span produces the same linear
  /// combination of the fitted outputs. When false, each pair uses its own
  /// Rayleigh quotient. No Regge, period, harmonic-eigenvalue, or charge term is
  /// added. Boundary-preserving stellar growth is retried up to `maxGrowth`.
  ///
  /// This method mutates the node's live spacetime in place. Existing relaxation
  /// modes and constructor defaults are unchanged.
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

  /// Fit one shared bulk geometry so the whole complex carries a spanning set
  /// of eigenstates whose boundary restrictions are prepared INPUT pairs and
  /// whose whole-complex readouts are prescribed outputs. `regionAName` and
  /// `regionBName` must name two declared pinned regions whose vertex sets are
  /// exactly the two connected components of \f$\partial W\f$; `cellsA` and
  /// `cellsB` must enumerate every degree-`degree` cell of the corresponding
  /// component. Witness `j` has boundary data `statesA[j]` on A and
  /// `statesB[j]` on B; the joint boundary vector is normalized once and
  /// `targets[j]` is scaled by the same factor. Each nonzero component
  /// restriction must be an isolated-boundary eigenstate (residual below
  /// `boundaryEpsilon`); an exactly zero restriction is admitted as the zero
  /// input on that component.
  ///
  /// `readouts[r]` is a formal chain on the live complex; the readout
  /// constraint \f$\langle c_r,\psi_j\rangle=\text{targets}[j][r]\f$ is imposed
  /// EXACTLY: the free amplitudes of witness `j` are parametrized on the
  /// affine solution set of its readout system (particular solution plus the
  /// readout null space), so no penalty weight enters. A readout system that
  /// the fixed boundary amplitudes make inconsistent is refused by name. The
  /// residual is the common-eigenvalue Rayleigh residual of
  /// `relaxBoundaryStatePairs`; bulk edge weights and, at degree zero,
  /// connection phases vary; every edge held by a declared pinned region is
  /// bit-identical. Boundary-preserving stellar growth is retried up to
  /// `maxGrowth`; existing cells persist under it, so readout chains survive.
  /// After the first pass every pass descends first from the previous pass's
  /// witnesses and live geometry (cells created by growth start at zero) and
  /// then from `restarts - 1` fresh random draws, so growth never discards
  /// progress; the first pass draws `restarts` random starts as
  /// `relaxBoundaryStatePairs` does on every pass.
  ///
  /// This method mutates the node's live spacetime in place.
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

  /// Read a square operator from the target-free live bulk. frameCells is the
  /// ordered row-major Choi frame and must contain exactly stateDimension^2
  /// interior edges. It may be empty only when the bulk itself has exactly that
  /// many edges, in which case canonical bulk-cell order is used.
  ///
  /// Promotion succeeds only when the framed restriction of
  /// \f$\ker L_1(W-\partial W)\f$ has rank one. No target or constraint is read.
  /// The returned Choi vector is unit-normalized and phase-fixed; the operator
  /// uses conventional unitary scaling \f$\sqrt d\f$ and reports its unitarity
  /// error rather than assuming it passed.
  [[nodiscard]] GeometricOperatorReadout geometricOperator(
      int stateDimension,
      std::vector<std::vector<std::uint64_t>> frameCells = {},
      double tol = 1e-9, bool metric = true) const;
  /// Move-kind names. Named rather than spelled as string literals at each site:
  /// every kind is written in the draw and compared in the apply, and a typo in
  /// either place would not fail to compile — it would silently misroute or
  /// disable the move.
  /// The four Pachner kinds are NOT redefined here — each move class owns its
  /// name (`AddMove::kMoveType` and siblings), and these alias those so there is
  /// exactly one definition per kind rather than one per dispatch site.
  static constexpr const char *kAddMove = ::tessera::spacetime::AddMove::kMoveType;
  static constexpr const char *kRemoveMove =
      ::tessera::spacetime::RemoveMove::kMoveType;
  static constexpr const char *kFlipMove =
      ::tessera::spacetime::FlipMove::kMoveType;
  static constexpr const char *kIFlipMove =
      ::tessera::spacetime::IFlipMove::kMoveType;
  /// Surgical kinds, owned here: they are `SurgicalCone` operations reached only
  /// through this draw, with no other definition to alias.
  static constexpr const char *kConeOut = "cone_out";
  static constexpr const char *kConeIn = "cone_in";
  static constexpr const char *kNoop = "noop";
  /// The two disposition moves (#613).
  static constexpr const char *kConeInTimelike = "cone_in_timelike";
  static constexpr const char *kFlipDisposition = "flip_disposition";

  /// A `kFlipDisposition` payload names one edge by its two endpoint vertex ids.
  static constexpr std::size_t kEdgeEndpointCount = 2;

  /// True when \p payload names an edge — exactly two endpoint vertex ids. Reads
  /// as the question being asked, where a bare `size() == 2` does not.
  [[nodiscard]] static bool payloadNamesAnEdge(
      const std::vector<std::uint64_t> &payload) {
    return payload.size() == kEdgeEndpointCount;
  }

  /// Whether the stage-1 move draw also proposes CAUSAL DISPOSITIONS (#613): a
  /// timelike cone-in, and a disposition flip on an existing edge. Both are
  /// ordinary candidate moves — drawn at random, scored by `deltaF`, committed
  /// only when they lower `F`. Nothing prescribes causal structure; the objective
  /// decides whether it wants any.
  ///
  /// They remain useful discrete proposals across causal sectors. The complex-z
  /// Stage 2 can also rotate continuously around `z=0`; it does not project the
  /// imaginary component away.
  ///
  /// Default **`true`** (#632): the causal moves are the seed's ONLY descent
  /// directions, so a draw without them is not a neutral default — it hides the
  /// physics. Measured on the single-Δ⁴ seed by enumerating EVERY move that adds a
  /// vertex (each of the C(5,4) facets coned in, both apex dispositions, plus the
  /// Pachner insertion): each of the five spacelike cone-ins RAISES `F` by `+0.777`
  /// and the Pachner add by `+2.58`, while each of the five timelike cone-ins LOWERS
  /// `F`, `‖∇S‖²` and `Re S` (`ΔF = -0.208`, all five equal by the seed's S₅
  /// symmetry), and they are the only moves giving `Im S ≠ 0`. With the six-move draw
  /// stage 1 finds nothing that lowers `F`, reports a stall it does not actually
  /// have, and left the seed through the since-removed trap-door escape — building
  /// an all-spacelike complex whose `Im S` is identically zero.
  ///
  /// KNOWN, UNRESOLVED (#632): with the moves in the draw, `‖∇S‖²` runs to `1.1e+15`
  /// on a 13-cell host and `1.03e+298` on a `Proton` node, while the ACTION itself
  /// stays finite (`S = -24.45 - 10.15i`). Value finite, gradient astronomical. The
  /// causal structure that emerges is real; the objective scoring it is not yet
  /// trustworthy on these hosts.
  ///
  /// The cause is NOT mixed disposition as such — it is exact degeneracy of the
  /// **tetrahedral facets**. The circumcentre solves `G β = ½ diag G`, so it is
  /// undefined exactly when `det G = 0`. The metric is Lorentzian throughout, so `G`
  /// is indefinite and `det G = 0` is a configuration the complex can actually reach:
  /// a facet whose span is tangent to the light cone — a NULL 3-face, zero 3-volume
  /// even though every one of its edges has `|ℓ²| = 1`. Quantising every `ℓ²` to
  /// exactly `±1` then lands on that locus **exactly**, rather than with measure zero
  /// as generic lengths would. Enumerating all `2⁶` sign patterns of a tetrahedron's
  /// edges:
  ///
  ///   * `0,1,2,4,5,6` timelike edges — never degenerate (`|det G| ∈ {0.5,1.0,1.5}`)
  ///   * **`3/6` timelike — 12 of those 20 patterns give `det G = 0` exactly**
  ///
  /// so `12/64` of all patterns are degenerate. Triangles and pentatopes never are at
  /// `±1` (`min|det G|` = `0.75` and `0.3125`); it is only the facets, and they poison
  /// the DEC dual recursion, which evaluates `circumradiusSquared` on the hinge's
  /// cofaces. `Simplex::circumFromGram` divides by `detG` under an EXACT-zero guard,
  /// so a rounding-level residue instead of a clean `0` gives `β ~ 1/detG`.
  ///
  /// The lever is therefore the uniform `±1` initialisation sitting on the degenerate
  /// locus, not a clamp in the dynamics: generic edge lengths essentially never hit
  /// `det G = 0`.
  ///
  /// Pass `false` to recover the six-move draw without explicit causal-disposition
  /// proposals. Stage 2 can still explore complex squared intervals.
  [[nodiscard]] bool shouldProposeDispositions() const {
    return shouldProposeDispositions_;
  }

  // ---- module-level helpers (static) ----
  /// Betti numbers (combinatorial, geometry-free).
  [[nodiscard]] static std::vector<int> betti(const Spacetime &st);
  /// The emergent k-register, read off `getBoundary`: the `(k+2)`-vertex tuples
  /// all of whose drop-one facets are boundary facets. Nothing placed.
  [[nodiscard]] static std::vector<std::vector<std::uint64_t>> emergentHoles(
      const Spacetime &st, int k);
  /// `Σ_e |actionGradientExact_e|²` — the full-complex Regge extremization term.
  [[nodiscard]] static double reggeActionGradient(const std::shared_ptr<Spacetime> &st);
  /// The relabeling-invariant, zero-filled residual of `targetState` against the
  /// `L_k` harmonic of `spacetime` over its emergent holes (`r_state` in the
  /// reference, the Python-binding name). For each register degree `k` it reads the
  /// emergent holes' cycle periods, least-squares-fits the target against them up to
  /// a relabeling of the target's components, and returns the smallest residual
  /// `\f$\lVert P c - t\rVert^2\f$`; with no emerged register it is the full leak
  /// `\f$\lVert t\rVert^2\f$`. Elemental: `residualForBoundaryBlock` sums this over
  /// the register degrees.
  [[nodiscard]] static double residualOfTargetStateAgainstHarmonic(
      const std::shared_ptr<Spacetime> &spacetime, int registerDegree,
      const std::vector<std::complex<double>> &targetState,
      HodgeLaplacian::MetricSource metricSource = HodgeLaplacian::defaultMetricSource());
  /// The same residual, with the winning relabeling RECORDED so no two registers in
  /// one `r_U` evaluation are scored against the same one. Every register scored
  /// independently picks its own argmin relabeling, and nothing stops a second
  /// register from picking that same matching — which reads both registers as
  /// carrying the same target component and rewards a complex whose registers all
  /// carry equal weights. `claimedMatchings` holds the relabelings the registers
  /// before this one already won: they are skipped here, and this register's argmin
  /// is inserted on the way out. Once every relabeling is claimed (more registers
  /// than the `d!` the target admits) the set is cleared and the exclusion restarts,
  /// so the residual is never the empty minimum.
  [[nodiscard]] static double residualOfTargetStateAgainstHarmonicWithDistinctMatching(
      const std::shared_ptr<Spacetime> &spacetime, int registerDegree,
      const std::vector<std::complex<double>> &targetState,
      std::set<std::vector<int>> &claimedMatchings,
      HodgeLaplacian::MetricSource metricSource = HodgeLaplacian::defaultMetricSource());

  // ---- objective ----
  /// The per-block register residual summed over `registerDegrees_`: `Σ r_U(boundary
  /// block)` over EVERY input and output block (the symmetric cobordism objective),
  /// PLUS the near-kernel residual per degree (see `nearKernelResidual`) — the
  /// pre-topological register signal. The period residual alone is a STEP function
  /// in the topology: before the first register opens it sits exactly at its
  /// zero-filled-leak floor (measured: `gamma * r_U = 50.000` for the seed and for
  /// every candidate cone-in), so F carries no register-seeking gradient at all
  /// until a hole exists. The near-kernel term is its analytic continuation below
  /// the topological threshold: on the near-kernel the period residual is a
  /// target-weighted sum of the smallest `|lambda|^2`, and this term is the same
  /// functional evaluated BEFORE the modes reach the kernel. The two meet at the
  /// opening: once `b_k` reaches the expected register count the smallest singular
  /// values are exactly zero, the near-kernel term saturates at 0, and the period
  /// residual takes over scoring WHAT the registers carry.
  [[nodiscard]] double rU(const std::shared_ptr<Spacetime> &st) const;

  /// The pre-topological register signal at one degree: the sum of the
  /// `expectedRegisterCount` smallest squared SINGULAR values of the METRIC
  /// `L_k` (the signed operator under the process weight convention),
  /// normalized scale-free.
  ///
  /// * **Metric, deliberately**: the term feels the continuously-valued edge
  ///   lengths, so it descends along TWO channels — stage-1 surgery (a genuine
  ///   hole zeroes the corresponding singular values exactly), and stage-2
  ///   tuning of the CAUSAL STRUCTURE toward null directions, which opens
  ///   near-kernels with no holes at all. The second channel is the point,
  ///   not a loophole: whether such causal near-kernels can carry a register
  ///   is the next level of exploration (readout semantics not implemented
  ///   here).
  /// * **Singular values, not eigenvalues**: the signed operator is
  ///   non-normal; singular values are the eigenvalue magnitudes of the
  ///   Hermitian `L^dagger L`, share the kernel exactly, and are smooth where
  ///   `|lambda|` is not — at a defective point the m smallest `|lambda|`
  ///   double-count a one-dimensional kernel, where the sigma count dimensions
  ///   honestly.
  /// * **Normalization**: `n * (sum of the m smallest sigma^2) / (sum of all
  ///   sigma^2)` — a generic mode contributes ~1, the range is [0, m]. `L_k`
  ///   is homogeneous of degree −1 in `l^2`, so a RAW spectral sum would hand
  ///   stage 2 a pure conformal-inflation descent channel (scale the geometry
  ///   up, every sigma shrinks); the ratio is degree 0 and closes exactly that
  ///   channel while leaving the causal-tuning channel open.
  /// * **Count**: `m` = the expected register count, read off the TARGETS
  ///   (`expectedRegisterCount`), never a constant — the term is the soft
  ///   relaxation of `b_k >= m`. Missing dimensions (`n < m`) count 1 each,
  ///   the worst case on the normalized scale.
  [[nodiscard]] static double nearKernelResidual(
      const std::shared_ptr<Spacetime> &st, int registerDegree,
      std::size_t expectedRegisterCount,
      HodgeLaplacian::MetricSource metricSource = HodgeLaplacian::defaultMetricSource());

  /// Exact COMPLEX gradient of `nearKernelResidual` with respect to each edge's
  /// `ℓ²`, in ChainComplex 1-cell order — the register residual's derivative,
  /// and the only part of `r_U` that has one before a register exists (the
  /// period-gap terms sit at their constant full leak until holes open, and a
  /// constant cannot move the geometry).
  ///
  /// Returns the #746 convention, `g = ∂r/∂(Re ℓ²) − i·∂r/∂(Im ℓ²)`, so the
  /// directional derivatives are `Re(g)` and `−Im(g)` and `conj(g)` is the
  /// steepest-ascent direction.
  ///
  /// With `H = L_k† L_k`, whose eigenvalues are the `σ²`:
  ///   `∂σᵢ² = wᵢᴴ(∂L† L + L† ∂L)wᵢ` for the normalized eigenvector `wᵢ`, and
  ///   `∂(Σσ²) = ∂ tr(H)`, combined by the quotient rule and scaled by `n`.
  ///
  /// Certified by the Euler identity `Σ ℓ²·g = 0` in both parts — the term is
  /// deliberately scale-invariant (degree 0), which is why it is a ratio rather
  /// than a raw spectral sum — verified on and off the real locus.
  ///
  /// Exactly zero where the value is constant (no `k`-cells, or an
  /// identically-zero operator). Where two singular values coincide at the
  /// `m`-th place the SELECTION is discontinuous and the functional genuinely
  /// non-smooth; this is the true derivative away from such a tie, and the
  /// line search arbitrates at one.
  [[nodiscard]] static std::vector<std::complex<double>> nearKernelResidualGradient(
      const std::shared_ptr<Spacetime> &st, int registerDegree,
      std::size_t expectedRegisterCount,
      HodgeLaplacian::MetricSource metricSource = HodgeLaplacian::defaultMetricSource());

  /// The scale-invariant spectral-shape term the `singularValueRatio` mode uses
  /// as the whole-complex contribution to `rU`, in place of BOTH the
  /// single-output period residual and `nearKernelResidual`: the ratio of the
  /// sum of the lower half of the singular values of the METRIC `L_k` (the same
  /// signed operator `nearKernelResidual` reads) to the sum of the upper half.
  /// With `n` values descending and `h = n/2` (integer division), it is
  /// `(σ_{n−h+1} + … + σ_n) / (σ_1 + … + σ_h)`; an odd `n` leaves the median out
  /// of both halves. Each lower-half value is bounded by its upper-half
  /// counterpart, so the ratio lives in `[0, 1]`, and `L_k` is homogeneous of
  /// degree −1 in `l^2`, so a uniform rescale of the geometry scales every
  /// `σ` alike and cancels — degree 0, no conformal-inflation channel, no
  /// prescribed target: the term rewards a collapsing lower half of the
  /// spectrum, and WHAT the register comes to carry is read out afterwards.
  /// No `k`-cells at all returns 1 (the worst case — an empty complex must not
  /// score as a collapsed spectrum); `n = 1` and an identically-zero `L_k`
  /// return 0 (no pair of halves to compare / every mode already kernel).
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
  /// HODGE degrees — the degrees the entropy is taken at, not the register
  /// degrees. This is an observation; the joint objective minimizes its
  /// gradient norm, not the entropy value itself.
  ///
  /// Reported UNWEIGHTED. The per-degree weights balance stationarity
  /// residuals against each other in the objective; applying them to entropy
  /// values would report a number that is not any degree's entropy.
  [[nodiscard]] double hodgeEntropy() const;
  /// \f$\sum_k w_k\|\nabla_zS_{{\rm Hodge},k}\|^2\f$ over the declared Hodge
  /// degrees and their weights: the entropy half of the joint objective.
  ///
  /// Reads the same degrees and weights the objective does, so an observation
  /// cannot disagree silently with the quantity being descended, and
  /// accumulates in the term's order, so `hodgeEntropyWeight() * this`
  /// reproduces `ObjectiveTerms::hodgeStationarity` exactly.
  [[nodiscard]] double hodgeEntropyStationarity() const;
  /// Inject the functional this node descends. The engine calls through it and
  /// knows nothing about which objective it holds; an objective reads only
  /// `ObjectiveContext`, which is the no-feedback firewall restated as an
  /// input type. `objectiveName()` is the single answer to what is being
  /// optimized — there is no second, enumerated answer that could disagree
  /// with it.
  /// @throws std::invalid_argument on a null objective, or on one whose
  ///   `minimumRegisterDegree()` exceeds a configured register degree.
  void setObjective(std::shared_ptr<CobordismObjective> objective);
  /// The injected functional. Never null: construction installs a default.
  [[nodiscard]] const std::shared_ptr<CobordismObjective> &objectiveSpec()
      const noexcept {
    return objectiveSpec_;
  }
  /// Inject an ADDITIONAL objective that holds a pinned region, alongside the
  /// bulk objective this node descends. Optional: with none supplied the pinned
  /// region's objective IS the bulk objective — one instance, not a copy with
  /// different defaults — and the run is bit-identical to a single-objective
  /// one.
  ///
  /// The region is not named here. The objective declares its own scope through
  /// `ObjectiveScope`, whose `RegionHandle` can only be obtained from
  /// `regionHandle`, so attaching an objective to a region never requires a
  /// caller to spell the same string twice and a mis-spelling cannot compile.
  ///
  /// Scoring is ADDITIVE and the bulk objective keeps scoring everything,
  /// including the pinned interior: the bulk sees one coherent cobordism and
  /// this is an additional hold on part of it. Boundary-interior edges
  /// therefore contribute to both, by design.
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
  void setHodgeEntropyPhaseMode(HodgeLaplacian::EntropyPhaseMode mode) noexcept {
    hodgeEntropyPhaseMode_ = mode;
  }
  [[nodiscard]] HodgeLaplacian::EntropyPhaseMode hodgeEntropyPhaseMode() const
      noexcept {
    return hodgeEntropyPhaseMode_;
  }
  void setHodgeEntropyWeight(double weight);
  [[nodiscard]] double hodgeEntropyWeight() const noexcept {
    return hodgeEntropyWeight_;
  }

  /// Declare the weight on the connection-entropy stationarity term — the ONLY
  /// term with a gradient in the connection phase. Zero by default, so a node
  /// acquires phase dynamics only when a caller asks for it; every \f$ L_k \f$
  /// is blind to \f$ \varphi \f$, so with this at zero the phase is a declared
  /// field that no geometric update can move.
  void setConnectionEntropyWeight(double weight);
  [[nodiscard]] double connectionEntropyWeight() const noexcept {
    return connectionEntropyWeight_;
  }

  /// Declare the Laplacian degrees the Hodge entropy term is summed over, and
  /// optionally a weight per degree.
  ///
  /// These are the degrees \f$k\f$ whose \f$L_k\f$ the entropy is taken of.
  /// They are configured HERE and read from nowhere else — in particular the
  /// register degrees, which answer the unrelated question of where a register
  /// is constructed, never supply them, not even as a fallback. The default is
  /// \f$\{0\}\f$.
  ///
  /// An empty `weights` means uniform \f$1\f$. A non-empty one must match
  /// `degrees` in length, so a caller cannot silently leave a degree
  /// unweighted.
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
  /// can tell WHICH degree the descent came from rather than only the total.
  /// Empty for an objective with no Hodge term.
  [[nodiscard]] std::vector<HodgeDegreeContribution>
  hodgeDegreeContributionsFor(
      const std::shared_ptr<Spacetime> &spacetime) const;
  /// `hodgeDegreeContributionsFor` on this node's own complex.
  [[nodiscard]] std::vector<HodgeDegreeContribution> hodgeDegreeContributions()
      const;

  void setReggeWeight(double weight);
  [[nodiscard]] double reggeWeight() const noexcept { return reggeWeight_; }
  /// Weight on each INPUT block's residual in `rU` (the output/whole term keeps
  /// weight 1). Raising it makes the optimizer prioritize keeping the input states
  /// represented, rather than only driving the whole to the output. Default 1.
  void setInputResidualWeight(double weight) { inputResidualWeight_ = weight; }

  // ---- the two stages + boundary-block construction ----
  /// Seed one INPUT block per seed vertex (region = the seed's cell-neighbourhood,
  /// tagged with its target). NOT grown here — runStage1's growBlockRegions grows
  /// it emergently under the objective.
  void seedInputs(const std::vector<std::uint64_t> &seeds);
  /// Seed one OUTPUT block per seed vertex (see seedInputs).
  void seedOutputs(const std::vector<std::uint64_t> &seeds);

  // ---- fiber-form boundary targets (#916) ----

  /// Score blocks that carry a fiber-form target by the FIBER residual (#940)
  /// instead of the period residual: the block's own sub-complex is read on the
  /// chain-level Whitney pencil, the band of the fiber's contour (or, when the
  /// fiber names none, the lowest band above the flat zero mode) is restricted
  /// to the fiber's cells, and the target images are least-squares fitted in
  /// that band's images, \f$ \min_C\|Z_T C-\Psi\|_F^2/\|\Psi\|_F^2 \f$. The
  /// coefficients of the read come from the block's lengths and connection
  /// values only; nothing is pinned. Folded into `rU`, so stage 1 scores
  /// candidate moves by it and stage 2 descends it through the same numerical
  /// register-residual path the period targets use. Off by default: every
  /// existing path is unchanged.
  void useFiberResiduals(bool enabled) { useFiberResiduals_ = enabled; }
  [[nodiscard]] bool usesFiberResiduals() const noexcept { return useFiberResiduals_; }
  /// Whether stage 2 also descends the degree-0 link phases through the
  /// analytic fiber gradient (#947). Off by default: flux lifts the flat zero
  /// mode and moves the band a default contour selects, so a caller enables
  /// it with a fixed contour on its fiber targets.
  void setFiberPhaseDescent(bool enabled) { fiberPhaseDescent_ = enabled; }
  [[nodiscard]] bool fiberPhaseDescent() const noexcept { return fiberPhaseDescent_; }
  /// The fiber residual of one block on \p spacetime (see `useFiberResiduals`).
  /// Full leak (1) when the block has no sub-complex, when a fiber cell is
  /// outside it, or when the band is empty.
  /// @throws std::logic_error when the block carries no fiber target or the
  ///   node's metric source is not the Whitney pencil.
  [[nodiscard]] double fiberResidualForBoundaryBlock(
      const BoundaryBlock &boundaryBlock, const std::shared_ptr<Spacetime> &spacetime) const;
  /// The fiber residual of input block \p index on the live complex.
  [[nodiscard]] double fiberResidualForInputBlock(std::size_t index) const;
  /// A fiber-form target carried by the WHOLE complex (#940): the node's own
  /// pencil is read on the fiber's contour (default: the lowest band above the
  /// flat zero mode), restricted to the fiber's cells, and the target images
  /// are least-squares fitted there. This is how an input node represents a
  /// state: its whole complex, grown from a single simplex, carries the fiber
  /// on the seed's cells. Scored inside `rU` under `useFiberResiduals`.
  void setWholeComplexFiberTarget(BoundaryFiber fiber);
  [[nodiscard]] const std::optional<BoundaryFiber> &wholeComplexFiberTarget() const noexcept {
    return wholeFiberTarget_;
  }
  /// The whole-complex fiber residual on the live complex.
  /// @throws std::logic_error without a whole-complex fiber target.
  [[nodiscard]] double wholeComplexFiberResidual() const;
  /// Read the fiber the whole complex carries on the target's cells and
  /// contour (the band of \p contour when given): what a downstream node is
  /// piped. @throws std::logic_error without a whole-complex fiber target.
  [[nodiscard]] BoundaryFiber readWholeComplexFiber(const chainhodge::Contour *contour = nullptr,
                                                    double kappa = 10.0) const;

  /// A single \p dimension-simplex host with a uniform metric
  /// (\f$ |\ell^2| = 1 \f$; balanced wiring gives \f$ \ell=\sqrt{1/2}(1+i) \f$),
  /// Lorentzian signature, the CDT type and preferred foliation: the canonical
  /// seed from which every host grows (`Proton::buildMinimalSeed` is this at
  /// dimension 4). @throws std::invalid_argument for dimension below 1.
  [[nodiscard]] static std::shared_ptr<Spacetime> seedSimplex(int dimension,
                                                              bool balancedEdges = false);

  // ---- two-body cobordism map (#941) ----

  /// The two-body target of the interaction node: \f$ \chi \f$ on the pair of
  /// input frames (\f$ r_A\times r_B \f$). `choiDecomposed` selects the READING
  /// the node reports: the state \f$ \mathrm{vec}(T_{AB}) \f$ on the pair space
  /// when true, the operator \f$ T_{AB} \f$ when false. The fit residual is the
  /// projective Frobenius leak in either reading (vec is linear, so the two
  /// coincide there); the readings differ in what is returned and certified.
  struct TwoBodyTarget {
    Eigen::MatrixXcd chi{};
    bool choiDecomposed{true};
  };
  /// The reading of the bulk between the two attached input frames.
  struct TwoBodyRead {
    bool choiDecomposed{true};
    /// \f$ T_{AB} = (Z_A^\vee)^T(\tilde A^U)_{AB}Z_B \f$ with the full frames on
    /// the two attached cell sets (the coupling block of the whole between them).
    Eigen::MatrixXcd transfer{};
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
    /// The fiber residual of each input block carrying an attached fiber.
    std::vector<double> inputFiberResiduals{};
    std::vector<std::vector<std::uint64_t>> cellsA{};
    std::vector<std::vector<std::uint64_t>> cellsB{};
  };
  /// Set an input block's region explicitly (#953): the vertex set whose
  /// sub-complex reads the block's fiber and whose internal edges a caller may
  /// pin. Emergent growth (`growBlockRegions`) can otherwise swallow the bulk,
  /// after which pinning the blocks pins everything. The attached fiber's cells
  /// must lie inside it. @throws std::out_of_range on the index;
  /// std::invalid_argument when an attached cell falls outside.
  void setInputBlockRegion(std::size_t index, std::set<std::uint64_t> vertices);
  /// Attach a piped input fiber to THIS complex's cells: `cells` (degree-k
  /// cells of the live complex, one per fiber row, in the attachment order — the
  /// attachment permutation is this order). The fiber's own cell ids are
  /// upstream ids and are replaced, and the block's region grows to contain
  /// the attached cells. @throws std::invalid_argument on a count mismatch, a
  /// cell absent from the live complex, or an overlap with another attached
  /// input fiber's cells.
  void attachInputFiber(std::size_t index, BoundaryFiber fiber,
                        std::vector<std::vector<std::uint64_t>> cells);
  /// Set the two-body target; scored inside `rU` under `useFiberResiduals`
  /// once two input fibers are attached. @throws std::invalid_argument on an
  /// empty target.
  void setTwoBodyTarget(Eigen::MatrixXcd chi, bool choiDecomposed = true);
  [[nodiscard]] const std::optional<TwoBodyTarget> &twoBodyTarget() const noexcept {
    return twoBodyTarget_;
  }
  /// The two-body residual on the live complex. @throws std::logic_error
  /// without a target or without two attached input fibers.
  [[nodiscard]] double twoBodyResidual() const;
  /// Read the bulk between the two attached frames on the live complex, with
  /// certificates. @throws std::logic_error without two attached input fibers.
  [[nodiscard]] TwoBodyRead readTwoBody() const;

  // ---- analytic gradients of the fiber-mode residuals (#947) ----
  /// A gradient over the live complex's edges in `EdgeList` order: each entry
  /// packs \f$ (\partial/\partial\operatorname{Re}, \partial/\partial\operatorname{Im}) \f$
  /// of the real residual as a complex number, the convention `runStage2`
  /// descends; `phases` is empty unless the residual varies with the links.
  struct ResidualGradient {
    Eigen::VectorXcd lengths;
    Eigen::VectorXcd phases;
  };
  /// The analytic gradient of `fiberResidualOn(spacetime, fiber)` through the
  /// band's Riesz projector (`chainhodge::BandDerivative`), holomorphic in
  /// the squared lengths and the phases, over \p spacetime's edges.
  [[nodiscard]] ResidualGradient fiberResidualGradientOn(const std::shared_ptr<Spacetime> &spacetime,
                                                         const BoundaryFiber &fiber) const;
  /// The analytic gradient of `twoBodyResidualOn` through the frame transfer
  /// (\f$ d\tilde A^U = dM^U h + M^U dh \f$ on the attached blocks).
  [[nodiscard]] ResidualGradient twoBodyResidualGradientOn(const std::shared_ptr<Spacetime> &spacetime,
                                                           const TwoBodyTarget &target) const;
  /// The ascent of every fiber-mode term of `rU` on the live complex: the
  /// whole-complex fiber target, each input block's fiber (its sub-complex
  /// gradient mapped to the parent's edges), and the two-body target.
  [[nodiscard]] ResidualGradient fiberModeAscent() const;

  /// Attach the fiber form of an input block's target (a prior cobordism's
  /// output fiber piped downstream). @throws std::out_of_range on the index.
  void setInputFiber(std::size_t index, BoundaryFiber fiber);
  /// Attach the fiber form of an output block's target.
  void setOutputFiber(std::size_t index, BoundaryFiber fiber);
  [[nodiscard]] const std::optional<BoundaryFiber> &inputFiber(std::size_t index) const;
  [[nodiscard]] const std::optional<BoundaryFiber> &outputFiber(std::size_t index) const;

  /// Pin the two input blocks' fibers as boundary data and relax the bulk so
  /// the whole complex carries them: the two fibers' images are concatenated
  /// on the union of their cells (the labeled sum on disjoint supports) and
  /// handed to `relaxFixedBoundaryEigenstate`, which holds the full geometric
  /// boundary fixed and varies only interior weights. The output emerges;
  /// nothing hand-identifies interior simplices. The fixed-boundary fit takes
  /// one pinned state, so the fibers must be rank one; a joint multi-column
  /// fit is refused by name rather than approximated column by column.
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
  /// `growBoundaries` is the INITIALIZATION pass: while true the boundary regions
  /// grow to track the bulk until they carry their states (growBlockRegions);
  /// run the bulk EVOLUTION with it false, so ∂W stays frozen.
  /// `maxLookahead`: when a batch of single moves finds no improvement, the
  /// search deepens iteratively — 2-move sequences, then 3, up to this many
  /// moves — committing an F-lowering sequence as a whole. DEFAULT 1 (single
  /// moves): a deepened batch builds and scores many more candidates per
  /// iteration, so deepening is a caller's choice rather than a default — the
  /// proton animation passes its `--max-lookahead-depth`. Every depth scores
  /// the same way, unrelaxed (#714).
  std::vector<double> runStage1(int maxSteps = 200, int nCandidateMoves = 12,
                                bool growBoundaries = false,
                                int maxLookahead = 1);
  /// Stage 2 (geometric): relax every full complex squared edge coordinate
  /// \f$z_e=\ell_e^2\f$ toward a stationary point/minimum of the selected scalar
  /// objective. Derivatives are taken with respect to \f$z\f$ and subtracted from
  /// \f$z\f$ itself; `Edge` stores \f$\ell\f$, so a trial is written with the square
  /// root closest to the resident branch. Neither the imaginary component nor the
  /// complex phase is projected away. The real line-search scale is backed off until
  /// a trial lowers the exact selected objective by at least `tolerance` (an absolute
  /// threshold); otherwise the original lengths are restored verbatim and the call
  /// reports stationarity. A genuine evaluation error also restores and propagates.
  /// `JointStationarity` and `MediatedCorrespondence` differentiate every scalar
  /// term. Mixed-term `Legacy` preserves its historical Regge search direction
  /// (with exact full-objective acceptance) for compatibility/performance.
  /// Default `tolerance` 1e-12: `runStage2` is the FINAL, precise relaxation of a
  /// drive (the combined `run` iterates its in-loop relaxations at the looser
  /// 10e-9 diminishing-returns cut and applies the same 1e-12 on its exit path).
  std::vector<double> runStage2(double beta = 1.0, int maxIters = 200,
                                  double alpha0 = 0.05, double tolerance = 1e-12);
  /// The combined drive. Each iteration takes ONE combinatorial stage-1 update —
  /// a best-ΔF move, deepening to `maxLookahead`-move sequences on a stall — and
  /// then relaxes the geometry FULLY: stage-2 updates repeat until the absolute
  /// improvement test at `tolerance` (default 10e-9) reports diminishing returns,
  /// so every move is proposed from, and leaves behind, relaxed geometry.
  ///
  /// Exit protocol: target-conditioned modes can exit once the register is carried
  /// with the geometry stationary. Every mode can also exit once combinatorial
  /// moves have had no effect (nothing committed at any lookahead depth AND
  /// nothing left to relax) for a few consecutive iterations (one stalled batch
  /// is draw noise, not proof).
  /// The LAST geometric relaxation before exit then runs at the tight 1e-12: if
  /// it still finds descent, the exit was premature and the loop continues on
  /// the freshly relaxed geometry; only a state stationary at 1e-12 exits.
  /// `maxIters` remains the hard budget cap.
  ///
  /// `nCandidateMoves`/`growBoundaries`/`maxLookahead` parameterize the
  /// combinatorial half exactly as in `runStage1`; `beta`/`alpha0`/`tolerance` the
  /// geometric half exactly as in `runStage2`. `run` stores `beta` as the node's
  /// Regge weight before either half runs, so stage 1, stage 2, `objective()`, and
  /// the shared trace all score one coherent functional.
  /// `lastStage2Stationary()` reports the LAST geometric update's outcome.
  /// `relaxBudgetPerMove` caps the stage-2 updates that follow each committed
  /// move (and the tight exit re-check): the stationarity test is the real
  /// terminator, the cap only bounds slow descent tails where the line search
  /// accepts a near-unbounded run of threshold-sized micro-steps.
  /// Returns the combined `F` trace.
  std::vector<double> run(int maxIters = 200, int nCandidateMoves = 12,
                          bool growBoundaries = false,
                          double beta = 1.0, double alpha0 = 0.05,
                          double tolerance = 10e-9, int maxLookahead = 1,
                          int relaxBudgetPerMove = 10);

  /// One canonical solve action on THIS node, the unit a search policy (Proton's build
  /// restart loop, a greedy driver, or the RL agent) composes — so the solve is driven
  /// through the engine, not re-implemented by each consumer.
  enum class BuildAction { Grow, Evolve, Relax, ConeOut, ConeIn };

  /// Candidate ordering for the directed cone-out probe's *secondary* sort (both orders are
  /// interior-first): `AdjacentHolesLast` sends cells that share vertices with the existing
  /// holes to the back, so new holes come out separated; `AdjacentHolesFirst` brings them to
  /// the front, so the register clusters. (For the first hole the orders coincide.)
  enum class HolePlacementStrategy { AdjacentHolesFirst, AdjacentHolesLast };

  /// Apply one `BuildAction` to this node (in place). Grow/Evolve = `runStage1` with
  /// `growBoundaries` true/false; Relax = `runStage2`; ConeOut/ConeIn = the directed probes
  /// below. Irrelevant params for a given action are ignored.
  void buildStep(BuildAction action, int maxSteps = 30, int nCandidateMoves = 8,
                 double stage2Beta = 1.0, int stage2MaxIters = 10,
                 double stage2Alpha0 = 0.05,
                 HolePlacementStrategy holePlacementStrategy = HolePlacementStrategy::AdjacentHolesLast);

  /// Directed, gated cone-OUT: remove top cells deliberately. Enumerates candidate top
  /// cells interior-first; `AdjacentHolesLast` then sends cells sharing vertices with the
  /// existing holes to the back (new holes separated), `AdjacentHolesFirst` to the front
  /// (holes cluster). Tries each with a gated `SurgicalCone::coneOut` (rolled back) and
  /// keeps the hole-opener that most lowers this node's `rU` (its realizability residual —
  /// which absorbs the output `r_state`, so this drives the register toward carrying the
  /// target on BOTH the 2→1 and 2→2 steps). Repeats up to `maxOpen`; stops when no opener
  /// lowers `rU`. Returns #holes opened.
  ///
  /// The manifold check inside `SurgicalCone::coneOut` is the ONLY gate: a cone-out that
  /// removes a pinned vertex is accepted when the result is a valid manifold in its own
  /// right (see the pinning section below).
  [[nodiscard]] int directedConeOut(HolePlacementStrategy strategy = HolePlacementStrategy::AdjacentHolesLast,
                                    int maxOpen = 6);

  /// Directed, gated cone-IN: select the register. Enumerates the boundary facets of the
  /// current emergent holes (capping one closes that hole), tries each with a gated
  /// `SurgicalCone::coneIn` (which builds on a fresh vertex, so it removes nothing), and
  /// keeps the cap that most lowers `rU` — i.e. drops the hole that hurts the carry. Repeats
  /// up to `maxClose`; stops when no cap lowers `rU`. Returns #holes capped.
  [[nodiscard]] int directedConeIn(int maxClose = 6);

  // ==================================================================
  // Pinning — a plain geometric constraint
  // ==================================================================
  //
  // Pinning says "do not change the geometry of these" and says NOTHING about
  // what information they carry. It is declared by the caller and never derived
  // from boundary blocks or their targets, so a pinned set means the same thing
  // whether or not any target is present.
  //
  // Pinning and manifold validity are NOT substitutes; they act on different
  // axes:
  //
  //   * pinning CONSTRAINS the geometry. A pinned edge — one whose endpoints are
  //     both pinned — is held at its resident squared length: stage 2 zeroes its
  //     descent component, so relaxation moves the rest of the complex around it.
  //     That is the "fixed boundary, relaxed bulk" structure of a cobordism,
  //     expressed without reference to any target.
  //
  //   * `dualComplexValid` GATES the topology. Whether a move may be applied is
  //     decided by whether the result is a valid manifold-with-boundary, and by
  //     nothing else. A surgery that removes a pinned vertex is ACCEPTED when the
  //     result is a valid manifold in its own right: refusing it would foreclose
  //     a legitimate topology change for a bookkeeping reason rather than a
  //     geometric one, and surgery is the only topology-changing mechanism the
  //     engine has (Pachner moves are bistellar and preserve Betti numbers).
  //
  // A pinned region is a DECLARED THING WITH AN IDENTITY, not a set computed on
  // demand. The separation it maintains is between WHICH cells are held and WHAT
  // they are held to: this class answers only the first. Keeping the second out
  // is what lets a boundary be declared without target-conditioning the bulk
  // geometry, which is the division the cobordism relaxation rule describes —
  // boundary fixed, bulk relaxed, the result read off the emergent bulk.

  /// A caller-declared pinned region: a named set of vertices held fixed. The name
  /// is its identity, so a region can be re-declared, referred to, and reported on.
  ///
  /// The region carries no target, no state and no objective. It is the "which".
  struct PinnedRegion {
    /// Identity. Re-declaring a region with an existing name replaces it.
    std::string name;
    /// The vertices held fixed. An edge relaxes unless BOTH endpoints are pinned.
    std::set<std::uint64_t> vertices;
  };

  /// Declare a pinned region, replacing any existing region of the same name.
  void declarePinnedRegion(PinnedRegion region);

  /// Every declared pinned region, in declaration order.
  [[nodiscard]] const std::vector<PinnedRegion> &pinnedRegions() const noexcept {
    return pinnedRegions_;
  }

  /// Drop every declared region, leaving the whole complex free to relax.
  void clearPinnedRegions();

  /// Mint a `RegionHandle` for a DECLARED region. This is the only way to
  /// obtain a non-empty handle, so an objective cannot reference a region that
  /// was never declared: a mis-spelling throws here, BY NAME, instead of
  /// compiling into a scope that silently matches nothing.
  /// @throws std::invalid_argument if no region of that name is declared.
  [[nodiscard]] RegionHandle regionHandle(const std::string &name) const;

  /// The union of every region's vertices — the flat view, for callers that need
  /// membership rather than provenance.
  [[nodiscard]] std::set<std::uint64_t> pinnedVertices() const;

  /// Whether the edge between `a` and `b` is held fixed: true iff some ONE region
  /// contains both endpoints. Two regions that each contain one endpoint do not
  /// pin the edge between them — that edge spans the gap between two independently
  /// declared regions and is part of the bulk.
  [[nodiscard]] bool edgeIsPinned(std::uint64_t a, std::uint64_t b) const;

  // ==================================================================
  // #776 — modes, the enumerable objective, and the analysis overlay
  // ==================================================================
  //
  // The no-feedback emergence firewall (whitepaper figure "No-feedback
  // emergence protocol") is enforced STRUCTURALLY here, not by
  // convention:
  //
  //   * `objectiveOf` is a **static** function of `ObjectiveTerms`, a record
  //     with exactly five named scalar members. `objectiveFor` may therefore
  //     consult only what `objectiveTermsFor` puts into that record — it has
  //     no `this` to reach an analysis member through. `objectiveTermNames`
  //     enumerates the list so a test can assert it, rather than trusting a
  //     comment.
  //   * `refinementDecisionOf` is likewise a **static** function of
  //     `RefinementIndicators` — five particle-independent geometric /
  //     numerical quantities — and `refinementIndicatorNames` enumerates them.
  //   * the ONLY channel from the carried quantum state to the geometry is the
  //     single `carriedStateEnergy` term, which is identically zero unless the
  //     run declares the `CertificatesBlindMeanField` sub-mode.
  //   * every recursive/certificate read produced by `runRecursiveAnalysis`
  //     lands in the checkpoint document and NOWHERE else: no member the
  //     objective or the refinement decision reads is written by that pass.

  /// Design spec §4 — the three top-level simulation modes.
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

  /// Design spec §4.1 — the two labeled, Gaussian-closed emergence sub-modes.
  /// Recorded in provenance on every checkpoint; both carry a covariance
  /// purity certificate.
  enum class EmergenceSubmode {
    /// The carried state does not act back on the geometry at all.
    Strict,
    /// The carried state's ENERGY DENSITY may enter the joint stationarity
    /// objective through `h = h(Γ, g)`, while no component, fiber, transport,
    /// amplitude, color, particle, charge, flavor, exchange, or spin
    /// certificate may influence a geometry move.
    CertificatesBlindMeanField
  };

  /// The COMPLETE, enumerable term list the scalar objective is the sum of.
  /// `objectiveOf` is static over this record, so the objective provably reads
  /// nothing else (see the firewall note above). Every member is a geometric
  /// or target quantity; the last is the one permitted state channel.
  /// Declared at namespace scope alongside `CobordismObjective` so an
  /// objective can be written without depending on this class; the alias keeps
  /// every existing use of `MultiCobordism::ObjectiveTerms` unchanged.
  using ObjectiveTerms = ::tessera::cobordism::ObjectiveTerms;

  /// The names of `ObjectiveTerms`' members, in declaration order — the
  /// firewall list a structural test asserts against.
  [[nodiscard]] static std::vector<std::string> objectiveTermNames();

  /// The scalar objective: the plain sum of the declared terms. STATIC by
  /// design — it cannot reach any analysis state.
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
  [[nodiscard]] SimulationMode simulationMode() const noexcept {
    return simulationMode_;
  }
  [[nodiscard]] EmergenceSubmode emergenceSubmode() const noexcept {
    return emergenceSubmode_;
  }
  /// `"emergence"` / `"synthesis"` / `"replay"`, as stamped on a checkpoint.
  [[nodiscard]] static std::string modeName(SimulationMode mode);
  /// `"strict"` / `"certificates_blind_mean_field"`, as stamped on a
  /// checkpoint.
  [[nodiscard]] static std::string submodeName(EmergenceSubmode submode);

  // ---- the carried quasi-free state ----

  /// Adopt the carried state: the covariance `Γ` (flat row-major, `m × m`)
  /// over `m` one-particle modes, each NAMED by the `degree`-cell it occupies
  /// (`modeCells[i]`, matched by vertex SET). Nothing here reads a
  /// certificate; the state is plain numbers plus the cells they live on.
  /// @throws std::invalid_argument on a non-square covariance, a size
  ///   mismatch against `modeCells`, or a degree below one (the carried
  ///   state's DECLARED domain, not a capability limit: \f$ L_0 \f$ and its
  ///   exact gradient exist too).
  void setCarriedState(
      const std::vector<std::vector<std::uint64_t>> &modeCells, int degree,
      const std::vector<std::complex<double>> &covariance);
  /// Drop the carried state (the energy term becomes exactly zero).
  void clearCarriedState();
  [[nodiscard]] bool hasCarriedState() const noexcept {
    return !carriedModeCells_.empty();
  }
  [[nodiscard]] int carriedStateDegree() const noexcept {
    return carriedStateDegree_;
  }
  [[nodiscard]] const std::vector<std::vector<std::uint64_t>> &
  carriedStateModeCells() const noexcept {
    return carriedModeCells_;
  }
  [[nodiscard]] const std::vector<std::complex<double>> &
  carriedStateCovariance() const noexcept {
    return carriedCovariance_;
  }

  /// The mean-field coefficient `β_E` (checkpointed). Default 0.
  void setCarriedStateEnergyWeight(double weight);
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
  /// Depends on `Γ` and the classical geometry ONLY. Exactly 0 outside the
  /// `CertificatesBlindMeanField` sub-mode, with no carried state, or at
  /// weight zero.
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

  /// The #780 purity defect \f$ \lVert\Gamma^2-\Gamma\rVert_F \f$ of the
  /// carried covariance (NaN with no carried state) — the Gaussianity
  /// certificate both emergence sub-modes must report.
  [[nodiscard]] double carriedStatePurityDefect() const;
  /// Whether the #780 purity certificate HOLDS at `tolerance`.
  [[nodiscard]] bool carriedStatePurityHolds(double tolerance = 1e-9) const;

  /// The checkpointed mean-field update schedule: `dt` per step, `steps`
  /// steps per `advanceCarriedState` call.
  void setMeanFieldSchedule(double dt, int steps);
  [[nodiscard]] double meanFieldStepSize() const noexcept {
    return meanFieldStepSize_;
  }
  [[nodiscard]] int meanFieldSteps() const noexcept { return meanFieldSteps_; }

  /// Advance the carried covariance through #780's
  /// `CovarianceState::meanFieldEvolve` under the SAME generator the energy
  /// term uses, \f$ h(\Gamma,g)=\tfrac12(L_k+L_k^\dagger)|_S \f$ with the
  /// classical geometry closed over. Returns the worst purity defect measured
  /// across the steps (NaN with no carried state); the loop is Gaussian-closed
  /// by construction and the certificate MEASURES that closure.
  double advanceCarriedState();

  // ---- particle-independent refinement ----

  /// The base geometric/numerical indicators emergence-mode refinement is
  /// allowed to consult. Every member is a quantity of the BASE problem: not
  /// one is a coarse-response residual, band gap, modularity, transport
  /// leakage, Wilson/center read, exchange read, anchor score, amplitude Gram
  /// defect, or particle score.
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
    /// Solver discretization error: the magnitude of the LAST accepted
    /// stage-2 objective improvement (0 at a stationary point).
    double solverError = 0.0;
  };

  /// The names of `RefinementIndicators`' members, in declaration order — the
  /// firewall list a structural test asserts against.
  [[nodiscard]] static std::vector<std::string> refinementIndicatorNames();

  /// Measure the indicators on this node's current complex.
  [[nodiscard]] RefinementIndicators refinementIndicators() const;

  /// The thresholds `refinementDecision` compares against. Defaults never
  /// fire: `reggeStationarityResidual`/`hodgeStationarityResidual`/
  /// `curvatureConcentration`/`solverError` are UPPER bounds crossed from
  /// below (infinity = never), `meshQuality` is a LOWER bound crossed from
  /// above (0 = never).
  void setRefinementThresholds(const RefinementIndicators &thresholds);
  [[nodiscard]] const RefinementIndicators &refinementThresholds()
      const noexcept {
    return refinementThresholds_;
  }

  /// Whether to refine, and which indicator asked. `trigger` is one of
  /// `refinementIndicatorNames()` or empty.
  struct RefinementDecision {
    bool refine = false;
    std::string trigger{};
    RefinementIndicators indicators{};
  };

  /// The refinement rule. STATIC over the indicator record by design: it
  /// cannot reach a certificate, a fiber, a transport, or a particle read.
  [[nodiscard]] static RefinementDecision refinementDecisionOf(
      const RefinementIndicators &indicators,
      const RefinementIndicators &thresholds);
  /// `refinementDecisionOf` on this node's measured indicators.
  [[nodiscard]] RefinementDecision refinementDecision() const;

  /// Apply geometry refinement when — and only when — `refinementDecision()`
  /// asks for it, through the EXISTING gated cone-in surgery
  /// (`applyMoveSpecification`'s `dualComplexValid` gate, the same primitive
  /// stage 1 and `preconeCells` use). Nothing is reimplemented and nothing is
  /// inserted by fiat. Returns the number of refinement cells committed.
  int refineGeometry(int maxCells = 1);

  // ---- the post-hoc analysis overlay ----

  /// Analysis-overlay configuration. DISABLED by default: with `enabled`
  /// false not one line of the overlay runs and the engine is bit-identical
  /// to the pre-#776 build.
  struct AnalysisConfig {
    /// Master switch. False = the overlay never runs (the default).
    bool enabled = false;
    /// Run one pass after every `cadence` ACCEPTED combinatorial moves.
    int cadence = 1;
    /// Hodge degrees the spectral bands are enumerated at.
    std::vector<int> degrees{1};
    /// The #765 modularity resolution sequence scanned per pass.
    std::vector<double> resolutions{1.0};
    /// Build the #771 lazy Fock expression (oracle / explicit non-Gaussian
    /// boundary data only — never the quasi-free production representation).
    bool fockOracle = false;
    /// Replay switch: serve NOTHING from the #764 `AnalyticCache`, so the
    /// cold path can be compared against the incremental one.
    bool coldCaches = false;
  };

  void setAnalysisConfig(const AnalysisConfig &config);
  [[nodiscard]] const AnalysisConfig &analysisConfig() const noexcept {
    return analysisConfig_;
  }

  /// Deterministic provenance stamped on every checkpoint.
  void setProvenance(const std::string &configHash, const std::string &commit);
  [[nodiscard]] const std::string &provenanceConfigHash() const noexcept {
    return provenanceConfigHash_;
  }
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

  /// Run ONE post-hoc analysis pass over the CURRENT accepted geometry, in the
  /// firewall order: publish the accepted move's touched star to the
  /// #764 `AnalyticCache`; update the #765 component hierarchy and its
  /// invalidated ancestry; update the #769 spectral projectors and the #768
  /// labeled retained-fiber sum; update the #770 transports and Wilson/center
  /// reads; update the #780 quasi-free covariance and its Wick reads; build
  /// the #771 lazy Fock expression only when the oracle is selected; evaluate
  /// the #773/#774/#775 particle reads; and record the checkpoint.
  ///
  /// Read-only on the geometry: this pass cannot accept, reject, or prioritize
  /// a move, and writes nothing the objective or the refinement decision
  /// reads.
  void runRecursiveAnalysis();

  /// The versioned checkpoint document of the last pass (schema version 5),
  /// as JSON. Empty before the first pass. Unknown /
  /// uncertified values serialize as `null`, never as zero.
  [[nodiscard]] const std::string &checkpointJson() const noexcept {
    return checkpointJson_;
  }
  /// The schema version this build writes and accepts. Version 5 adds the
  /// per-edge connection phase to `raw_complex.edges`: an edge carries TWO
  /// fields, and a version-4 document recorded only the length, so replaying
  /// one silently rebuilt every edge with a zero phase and dropped a live
  /// field. Version 4 also split the former `particles.baryons` block in two:
  /// the bound-supercomponent SEARCH records moved to
  /// `particles.bound_supercomponents` and `particles.baryons` now carries the
  /// three-cluster VERDICT itself, one baryon read per binding of exactly
  /// three certified constituents.
  ///
  /// A version-3 document is REJECTED on read rather than reinterpreted — its
  /// `baryons` entries mean a different thing. A version-4 document is also
  /// rejected, because its silence about the phase is not evidence the phase
  /// was zero; `replayPhaseDefault` documents the only reading under which one
  /// could be accepted.
  [[nodiscard]] static int checkpointSchemaVersion() noexcept { return 5; }

  /// The phase a pre-version-5 document would replay with, were such a document
  /// accepted: exactly zero. This is FAITHFUL to what version 4 recorded — it
  /// stored no phase at all, so zero is what its `raw_complex` describes — but
  /// it is NOT faithful to the run that wrote it, whose edges may have carried
  /// any phase. That gap is why version 4 is rejected on read rather than
  /// silently defaulted: a replay that quietly zeroes a written field is worse
  /// than one that refuses.
  [[nodiscard]] static std::complex<double> replayPhaseDefault() noexcept {
    return {0.0, 0.0};
  }

  /// Replay mode: rebuild the raw complex recorded in `checkpoint`, disable
  /// every cache, recompute every derived hierarchy and certificate, and
  /// return the freshly written checkpoint — stamped `"replay"`. The verdicts
  /// must equal the incremental run's.
  /// @throws std::invalid_argument on malformed JSON or an UNKNOWN
  ///   `schema_version` (a reader never guesses at a version it does not
  ///   know).
  [[nodiscard]] static std::string replayCheckpoint(
      const std::string &checkpoint);

  /// The `schema_version` recorded in `checkpoint`.
  /// @throws std::invalid_argument on malformed JSON or a missing version.
  [[nodiscard]] static int checkpointVersionOf(const std::string &checkpoint);

  [[nodiscard]] std::shared_ptr<Spacetime> spacetime() const { return spacetime_; }
  [[nodiscard]] const std::vector<BoundaryBlock> &inputs() const {
    return inputBlocks_;
  }
  [[nodiscard]] const std::vector<BoundaryBlock> &outputs() const {
    return outputBlocks_;
  }
  /// Whether the last `runStage2` ended because no complex-z line-search trial
  /// lowered `F` by the absolute `tolerance` threshold (`true`) versus hitting
  /// `maxIters` (`false`).
  /// Lets a caller report "stopped: stationary" vs "stopped: budget". `false` before
  /// the first `runStage2`/`run`; after `run`, reports the LAST geometric update's
  /// outcome (each update resets the flag, so an earlier stationary point that a
  /// later topology change reopened does not latch).
  [[nodiscard]] bool lastStage2Stationary() const { return lastStage2Stationary_; }

  /// The lookahead depth of the LAST stage-1 update's committed sequence: 1 = an
  /// ordinary single move, >1 = the search had to deepen (the single-move batch
  /// stalled and an F-lowering multi-move sequence was found at this depth), 0 =
  /// no F-lowering sequence found at ANY depth up to the update's `maxLookahead`
  /// (a stage-1 stall). 0 before the first update. Lets a driver/animation show
  /// WHEN the optimizer is looking more than one move into the future.
  [[nodiscard]] int lastStage1Lookahead() const { return lastStage1LookaheadDepth_; }

 private:
  using Snapshot =
      std::pair<std::vector<std::vector<std::uint64_t>>,
                std::map<std::pair<std::uint64_t, std::uint64_t>,
                         std::complex<double>>>;
  using MoveSpec = std::pair<std::string, std::vector<std::uint64_t>>;

  // ---- the pieces of residualOfTargetStateAgainstHarmonic ----
  /// The target state as a dense complex vector — the `t` the harmonic is fitted to,
  /// and (as its squared norm) the full leak when no holes have emerged to read over.
  [[nodiscard]] static Eigen::VectorXcd targetStateVector(
      const std::vector<std::complex<double>> &targetState);
  /// The emergent holes that can carry `targetDimension` components: `emergentHoles`
  /// at this degree, truncated to at most one hole per target component. Empty when
  /// no holes have emerged.
  [[nodiscard]] static std::vector<std::vector<std::uint64_t>> holesCarryingTheTarget(
      const Spacetime &spacetime, int registerDegree, std::size_t targetDimension);
  /// The period matrix \f$ P^{\top} \f$ of the degree's harmonics over `cycleHoles`:
  /// `(targetDimension, b_k)`, row = hole, column = harmonic, zero-filled past the
  /// holes that emerged (a component with no hole to sit in leaks in full).
  [[nodiscard]] static Eigen::MatrixXcd holePeriodMatrix(
      const std::shared_ptr<Spacetime> &spacetime, int registerDegree,
      int degreeBettiNumber,
      const std::vector<std::vector<std::uint64_t>> &cycleHoles,
      std::size_t targetDimension,
      HodgeLaplacian::MetricSource metricSource = HodgeLaplacian::defaultMetricSource());
  /// The target's components reordered onto the holes by `relabeling`: component
  /// `relabeling[q]` sits in hole `q`. A relabeling is a bijection, so each hole
  /// takes exactly one component.
  [[nodiscard]] static Eigen::VectorXcd relabeledTargetVector(
      const Eigen::VectorXcd &targetVector, const std::vector<int> &relabeling);
  /// One register's winning relabeling: which target component each of its holes
  /// carries, and the least-squares residual \f$ \min_c \lVert P^{\top} c - t \rVert^2 \f$
  /// that matching leaves. `scored` is false when every relabeling was skipped as
  /// already claimed, so nothing was evaluated.
  struct RelabelingMatch {
    double residual = 0.0;
    std::vector<int> relabeling;
    bool scored = false;
  };
  /// The argmin over the `d!` relabelings of the target components onto the holes.
  /// With `skipClaimed` the relabelings in `claimedMatchings` — the ones registers
  /// scored earlier already won — are passed over, so this register is read against
  /// a matching of its own.
  [[nodiscard]] static RelabelingMatch bestRelabelingOfTarget(
      const Eigen::MatrixXcd &periodMatrixTransposed,
      const Eigen::VectorXcd &targetVector,
      const std::set<std::vector<int>> &claimedMatchings, bool skipClaimed);

  /// One boundary block's `r_U` term: the sum over the register degrees of
  /// `residualOfTargetStateAgainstHarmonic` evaluated on the block's own
  /// sub-complex (`subcomplexWithinVertexSet`) against the block's target. When the
  /// block has no full sub-complex yet, the full leak summed over the degrees.
  [[nodiscard]] double residualForBoundaryBlock(
      const BoundaryBlock &boundaryBlock,
      const std::shared_ptr<Spacetime> &spacetime) const;
  /// The same block term, sharing one `claimedMatchings` set with the rest of the
  /// `r_U` evaluation so this block's register degrees cannot re-use a relabeling
  /// another register already won (see
  /// `residualOfTargetStateAgainstHarmonicWithDistinctMatching`).
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

  [[nodiscard]] Snapshot snapshotOf(const Spacetime &spacetime) const;
  [[nodiscard]] Snapshot snapshot() const;
  [[nodiscard]] std::shared_ptr<Spacetime> build(
      const Snapshot &complexSnapshot) const;

  /// Draw one random stage-1 move specification on `spacetime`: a `{kind, payload}`
  /// pair where `kind` is one of `add`/`remove`/`flip`/`iflip` (payload = a seed for
  /// the Pachner move) or `cone_out`/`cone_in` (payload = the cell/face to cone). The
  /// move is only described here, not applied — see `applyMoveSpecification`.
  [[nodiscard]] MoveSpec drawRandomMoveSpecification(const Spacetime &spacetime);
  /// Apply a move specification from `drawRandomMoveSpecification` to `spacetime`
  /// in place. Returns true iff the move was applied AND the result passes the
  /// `dualComplexValid` gate at `dualComplexGateDegree_`; otherwise the caller
  /// discards the candidate. Manifold validity is the whole gate — a move that
  /// removes a pinned vertex is accepted when what it leaves is a valid manifold.
  [[nodiscard]] bool applyMoveSpecification(
      const std::shared_ptr<Spacetime> &spacetime,
      const MoveSpec &moveSpecification);
  [[nodiscard]] double
  deltaF(const std::shared_ptr<Spacetime> &candidateSpacetime,
         double baseObjective, double baseResidualU,
         const std::set<std::vector<std::uint64_t>> &baseCellSet) const;
  /// One best-ΔF batch: `nCandidateMoves` candidates, each a sequence of
  /// `lookaheadDepth` gated random moves applied successively (each drawn against
  /// the evolving candidate), committed as a whole iff the best sequence lowers
  /// F. EVERY depth scores by the same localized, UNRELAXED `deltaF` (#714):
  /// the combinatorial moves exist to leave a local minimum and the geometric
  /// update to descend within the region the complex then occupies, so scoring
  /// a candidate through a relaxation would mix the two — it would ask where a
  /// move lands after stage 2 rather than whether the move improves the state.
  /// A committed candidate is relaxed afterwards, bounded by the caller's
  /// `relaxBudgetPerMove`. Depth 1 pre-draws its batch and scores it in
  /// parallel; deeper searches stay serial, since each draw is made against the
  /// evolving candidate. Returns the committed ΔF, or 0.
  double step(int nCandidateMoves, int lookaheadDepth, double baseObjective);
  /// One iteration of `runStage1`'s loop: optional boundary growth plus one
  /// best-ΔF candidate-move step, booked into `objectiveTrace`. A batch with no
  /// improving move is NOT a stall — the batch is a random sample, so the next
  /// iteration simply redraws. Returns whether the caller should keep iterating:
  /// target-conditioned modes continue until the register is carried, while
  /// target-free `JointStationarity` stops after the stalled batch.
  bool stage1Update(int nCandidateMoves, bool growBoundaries,
                    std::vector<double> &objectiveTrace, int maxLookahead = 1);
  /// One iteration of `runStage2`: assemble the selected objective's complex-z
  /// ascent direction, subtract it from z, and run the backtracking line search.
  /// Appends an accepted objective and adapts `stepScale`; otherwise restores the
  /// original length branches and reports stationarity.
  bool stage2Update(double beta, double tolerance,
                    std::vector<double> &objectiveTrace, double &stepScale);
  /// Grow each localized boundary block's region to track the bulk's growth: expand
  /// its vertex set by one shell (every top cell touching the current region), so the
  /// block gets room to develop the holes that carry its state — instead of staying
  /// frozen at the (too-small) construct-time region. Applies to every INPUT block
  /// and every localized OUTPUT block (a multi-output recombination); a single output
  /// reads off the whole and has no block here. Bounded: a block already carrying
  /// (residual < inputCarriedTolerance_) is left alone, so it stops growing once it
  /// represents its state. GATED per block: a shell that would RAISE the block's
  /// own r_U term is reverted (Δ <= 0 passes — the full-leak plateau of a region
  /// with no full cell yet is Δ == 0), so region growth can never raise F.
  /// Expand each not-yet-carrying block's SCORING REGION by one shell (the
  /// vertices of every top cell touching it), so the window a block's residual
  /// is read over gains room for the holes that carry its state.
  ///
  /// Two conditions bound it (#737). A shell is kept only when it STRICTLY
  /// lowers that block's residual, and growth happens only BEFORE the first
  /// committed combinatorial move — once the bulk is being linked, the states'
  /// read windows are settled. Without both, growth had no stopping point: a
  /// block that is not carrying scores the same constant full leak at any
  /// region size, so every shell was an exact tie, ties were kept, and the
  /// regions grew until they covered the whole complex and all blocks read one
  /// identical sub-complex.
  ///
  /// Creates no cells, edges, or vertices and never moves the cobordism's
  /// boundary: the only write is each block's vertex set.
  void growBlockRegions();
  /// Pre-grow the seed by `count` **gated cone-in moves** before any optimization
  /// (the constructor calls this once when `precone > 0`): each cones a fresh apex
  /// onto a random codim-1 facet of a random top cell and is committed only through
  /// `applyMoveSpecification`'s `dualComplexValid` gate — the same gate stage 1
  /// uses, so nothing is inserted by fiat. It enlarges the complex so
  /// surgery has room to act — the emergent analogue of a prebuilt host refinement.
  /// `count <= 0` is a no-op (RNG untouched). Best-effort: a draw onto an already-
  /// saturated facet is rejected by the gate and retried; if no valid cone-in is
  /// found for a cell, it stops early.
  /// `timelike` draws every cone timelike; `alternate` interleaves
  /// timelike/spacelike (and wins over `timelike`); default all-spacelike.
  void preconeCells(int count, bool timelike = false, bool alternate = false);

  std::shared_ptr<Spacetime> spacetime_;
  std::vector<std::vector<std::complex<double>>> inputTargets_;
  std::vector<std::vector<std::complex<double>>> outputTargets_;
  /// The register degrees `k` the objective scores at once (every `r_U` term is
  /// summed over these); a `b_k` register is forced to emerge for each.
  std::vector<int> registerDegrees_;
  /// The single degree at which the `dualComplexValid` move gate runs — the maximum
  /// register degree (the degree-free validity check needs only the coarsest one).
  int dualComplexGateDegree_;
  double gamma_;
  /// The caller-declared pinned regions (see `declarePinnedRegion`). Empty by
  /// default. Never derived from boundary blocks or their targets: each region is
  /// a plain combinatorial declaration that constrains the geometry rather than
  /// gating any move.
  std::vector<PinnedRegion> pinnedRegions_;
  /// Ordered exact-period state constraints. These are explicit fixtures over a
  /// caller-supplied topology, separate from emergent block matching and from
  /// geometric pinning.
  std::vector<RegisterConstraint> registerConstraints_;
  /// #690: propagated to every spacetime this node constructs
  /// (host before precone, and each candidate snapshot rebuild).
  bool balancedEdgeWiring_{false};
  /// #697: `rU`'s whole-complex term is `singularValueHalfSumRatio` instead of
  /// the period residual + `nearKernelResidual` pair (see the constructor).
  bool singularValueRatio_{false};
  /// #724: false drops `‖∇S_Regge‖²` from every objective site (see the ctor).
  bool einsteinHilbert_{true};
  /// Restrict stage-2 updates to real squared lengths. False preserves the
  /// general complexified geometry.
  bool realSquaredLengthsOnly_{false};
  /// The metric source of every Hodge operator this node builds (see `metricSource()`).
  HodgeLaplacian::MetricSource metricSource_{HodgeLaplacian::MetricSource::DiagonalWeights};
  /// The injected functional, and the only record of what this node descends.
  /// Never null: the constructor installs `LegacyObjective`, so a caller that
  /// never injects one gets exactly the objective it got before this became
  /// injectable.
  std::shared_ptr<CobordismObjective> objectiveSpec_;
  /// The optional additional objective holding a pinned region. Null means the
  /// pinned region's objective IS the bulk objective, which is the
  /// single-objective run.
  std::shared_ptr<CobordismObjective> pinnedObjectiveSpec_;
  /// Assemble the firewalled input an objective reads. Private because the
  /// bound evaluators close over this node; an objective receives the
  /// assembled context and can reach nothing beyond it.
  ///
  /// The objective is passed so its DECLARED scope can be resolved into the
  /// context's region and edge list. The engine reads the declaration; it does
  /// not infer a scope from the objective's role or decide one by convention.
  [[nodiscard]] ObjectiveContext objectiveContextFor(
      const std::shared_ptr<Spacetime> &spacetime,
      const std::shared_ptr<CobordismObjective> &objective) const;
  /// The bulk objective's context, which is the whole cobordism.
  [[nodiscard]] ObjectiveContext objectiveContextFor(
      const std::shared_ptr<Spacetime> &spacetime) const;
  /// Whether the SCALAR THIS NODE REPORTS admits a localized exact delta. Not
  /// simply the bulk objective's declaration: with a pinned objective in force
  /// the reported scalar is a sum over two scopes, and differencing the bulk
  /// alone would score a surrogate that is not the objective.
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
  /// An edge is INTERIOR to the region when both endpoints lie in it and
  /// STRADDLING when exactly one does. Interior edges are always in; straddling
  /// edges are in only where the objective declared them so. The border is the
  /// one the node already defines — `edgeIsPinned` holds exactly when a single
  /// region contains both endpoints — rather than a second notion of adjacency
  /// invented here.
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
  /// #737: latched by the first committed combinatorial move. Block regions
  /// grow only BEFORE the bulk is connected, so once a move has linked the
  /// complex up the boundary states' read windows are settled.
  bool bulkConnected_{false};
  /// The Einstein-Hilbert term of the objective, or 0 when it is switched off.
  /// One place, so `objective`, the stage-2 acceptance test, and `deltaF`
  /// cannot come to disagree about what F is.
  [[nodiscard]] double objectiveFor(
      const std::shared_ptr<Spacetime> &spacetime) const;
  /// Weight on the input-block residual terms in `rU` (see setInputResidualWeight).
  double inputResidualWeight_ = 1.0;
  bool useFiberResiduals_{false};
  bool fiberPhaseDescent_{false};
  std::optional<BoundaryFiber> wholeFiberTarget_;
  std::optional<TwoBodyTarget> twoBodyTarget_;
  /// The transfer between the full frames on the two attached input fibers'
  /// cells, on \p spacetime; refused geometries throw std::runtime_error.
  [[nodiscard]] chainhodge::TransferResult frameTransferOn(
      const std::shared_ptr<Spacetime> &spacetime, const BoundaryFiber &A,
      const BoundaryFiber &B) const;
  [[nodiscard]] double twoBodyResidualOn(const std::shared_ptr<Spacetime> &spacetime,
                                         const TwoBodyTarget &target) const;
  [[nodiscard]] std::pair<const BoundaryFiber *, const BoundaryFiber *> attachedInputFibers() const;

  /// The fiber residual of \p fiber read on \p spacetime (see `useFiberResiduals`).
  [[nodiscard]] double fiberResidualOn(const std::shared_ptr<Spacetime> &spacetime,
                                       const BoundaryFiber &fiber) const;
  /// The block's sub-complex WITH the parent's geometry: `subcomplexWithinVertexSet`
  /// builds it at unit lengths (its period residual is combinatorial), so the
  /// fiber reads copy every edge's length and phase from \p spacetime by vertex
  /// pair. Null when the region holds no top cell.
  [[nodiscard]] static std::shared_ptr<Spacetime> blockSubcomplexWithGeometry(
      const BoundaryBlock &block, const std::shared_ptr<Spacetime> &spacetime);
  /// An input region stops growing (growInputRegions) once its residual drops below
  /// this — i.e. once it carries its state.
  double inputCarriedTolerance_ = 1e-12;
  /// The move/restart random source driving stage 1 and block construction.
  std::mt19937_64 randomNumberGenerator_;
  /// #613: whether the move draw offers the disposition moves. See the accessor.
  bool shouldProposeDispositions_{true};
  double convergenceTolerance_ = 1e-9;
  /// Set by `runStage2`: `true` iff its last call stopped on the absolute-tolerance
  /// stationarity test, `false` iff it hit the `maxIters` budget. See lastStage2Stationary.
  bool lastStage2Stationary_ = false;
  /// Set by `stage1Update`: the committed sequence's lookahead depth (see
  /// `lastStage1Lookahead`). 0 = the update committed nothing.
  int lastStage1LookaheadDepth_ = 0;
  std::vector<BoundaryBlock> inputBlocks_;
  std::vector<BoundaryBlock> outputBlocks_;

  // ---- #776 state ----
  //
  // NONE of the analysis members below is read by `objectiveFor`,
  // `objectiveTermsFor`, `deltaF`, `rU`, `step`, `stage1Update`, or
  // `stage2Update`. The only member on this list the objective touches is the
  // carried state (`carriedModeCells_` / `carriedCovariance_` /
  // `carriedStateEnergyWeight_`), and only through the single
  // `ObjectiveTerms::carriedStateEnergy` scalar, and only while the run
  // declares `EmergenceSubmode::CertificatesBlindMeanField`.
  SimulationMode simulationMode_{SimulationMode::Emergence};
  EmergenceSubmode emergenceSubmode_{EmergenceSubmode::Strict};
  /// The carried modes' `carriedStateDegree_`-cells, by vertex tuple.
  std::vector<std::vector<std::uint64_t>> carriedModeCells_{};
  /// Γ, flat row-major over the carried modes.
  std::vector<std::complex<double>> carriedCovariance_{};
  int carriedStateDegree_{1};
  double carriedStateEnergyWeight_{0.0};
  double meanFieldStepSize_{0.0};
  int meanFieldSteps_{0};
  /// Thresholds for `refinementDecisionOf`. Explicitly ALL ZERO — the
  /// indicator struct's own defaults describe a healthy complex
  /// (`meshQuality` 1), which as a LOWER bound would fire on every real mesh.
  /// Zero is "never" for both senses, so an unconfigured node never refines.
  RefinementIndicators refinementThresholds_{0.0, 0.0, 0.0, 0.0, 0.0};
  /// |ΔF| of the last ACCEPTED stage-2 update (the solver-error indicator).
  double lastStage2Improvement_{0.0};
  AnalysisConfig analysisConfig_{};
  std::string provenanceConfigHash_{};
  std::string provenanceCommit_{};
  std::uint64_t seed_{0};
  std::uint64_t acceptedMoveCount_{0};
  std::uint64_t analysisPassCount_{0};
  /// The cell set of the complex the LAST analysis pass saw — differenced
  /// against the current one to publish the accepted move's touched star.
  std::set<std::vector<std::uint64_t>> analysisCellSet_{};
  /// The edge lengths the LAST analysis pass saw, so a pure metric change
  /// publishes only the edges that actually moved and disjoint siblings stay
  /// served from cache.
  std::map<std::pair<std::uint64_t, std::uint64_t>, std::complex<double>>
      analysisEdgeLengths_{};
  /// Set true once `analysisCellSet_` holds a real observation.
  bool analysisCellSetValid_{false};
  /// The #764 analytic cache the overlay reuses ACROSS passes, so a local
  /// change invalidates only the entries whose component meets the published
  /// star and disjoint siblings stay served. Held as `void` so this header
  /// keeps its include list; rebound whenever `spacetime_` becomes a
  /// different object (every committed combinatorial move rebuilds it).
  std::shared_ptr<void> analysisCache_{};
  /// The spacetime `analysisCache_` is bound to (identity comparison only).
  std::weak_ptr<Spacetime> analysisCacheBinding_{};
  /// The last pass's checkpoint document.
  std::string checkpointJson_{};

  /// Run the overlay when the config asks for it — called ONLY after a move
  /// has already been committed, so it cannot influence that move.
  void noteAcceptedMove();
  /// The overlay pass body (implemented beside the rest of the recursive
  /// integration, in `src/cobordism/RecursiveFiberSimulation.cpp`).
  void runRecursiveAnalysisOn(const std::shared_ptr<Spacetime> &spacetime);
  /// Serialize the current raw complex + edge data for the checkpoint.
  [[nodiscard]] std::string rawComplexJson(
      const std::shared_ptr<Spacetime> &spacetime) const;
  /// The one-particle generator `h_S(g)` of the carried modes on `spacetime`,
  /// flat row-major over the carried modes; entries for absent cells are 0.
  [[nodiscard]] std::vector<std::complex<double>> carriedStateGenerator(
      const std::shared_ptr<Spacetime> &spacetime) const;
};

}  // namespace tessera::cobordism

#endif  // TESSERA_COBORDISM_MULTICOBORDISM_H
