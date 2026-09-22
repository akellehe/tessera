// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_COBORDISM_COBORDISMOBJECTIVE_H
#define TESSERA_COBORDISM_COBORDISMOBJECTIVE_H

#include <complex>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "cobordism/HodgeLaplacian.h"

namespace tessera::spacetime { class Spacetime; }

namespace tessera::cobordism {
using ::tessera::spacetime::Spacetime;

/// # ObjectiveTerms
///
/// The complete term list a scalar cobordism objective is the sum of.
/// `CobordismObjective::total` is static over this record, so the collapse to
/// the number the optimizer compares reads nothing else. Every member is a
/// geometric or target quantity except the last, the one permitted state
/// channel.
///
/// At namespace scope, so an objective can be written without depending on
/// `MultiCobordism`, which aliases it as `MultiCobordism::ObjectiveTerms`.
struct ObjectiveTerms {
  /// \f$\beta_R\|\nabla_zS_{\rm Regge}\|^2\f$ — 0 when the Einstein-Hilbert
  /// term is deselected.
  double reggeStationarity = 0.0;
  /// \f$\eta_H\sum_k\|\nabla_zS_{{\rm Hodge},k}\|^2\f$ — joint stationarity.
  double hodgeStationarity = 0.0;
  /// \f$\eta_C\|\nabla_\varphi S_{\mathbb{C}^{*}}\|^2\f$ — stationarity of the
  /// connection operator's entropy in the connection phase. The only term with
  /// a \f$\varphi\f$ gradient: every \f$ L_k \f$ is blind to \f$\varphi\f$, so
  /// without it \f$\varphi\f$ is a declared field that no update moves.
  double connectionStationarity = 0.0;
  /// \f$\gamma r_U\f$ — the target-conditioned register residual.
  double registerResidual = 0.0;
  /// \f$r_U+\beta|S_{\rm Regge}(W^*)|\f$'s action magnitude.
  double actionMagnitude = 0.0;
  /// \f$\beta_E E_{\rm carried}(\Gamma,g)\f$ — the one permitted state
  /// channel, exactly 0.0 outside the certificates-blind mean-field sub-mode.
  double carriedStateEnergy = 0.0;
};

/// # ObjectiveContext
///
/// The no-feedback firewall for an injected objective, on the input side.
///
/// Plain data: geometry, a region, that region's declared target states, and
/// scalar configuration. No `MultiCobordism` reference, no pointer to one, and
/// no `std::function`, since a bound callable would capture the node. An
/// objective therefore cannot consult a component, fiber, transport, amplitude,
/// colour, charge, flavour, exchange, spin certificate or verdict.
/// `inputNames()` enumerates every field, so a test can assert the list.
///
/// Reading the complex is intended; reaching the analysis products of that
/// geometry is not, and those live in the checkpoint document, where no context
/// can see them.
struct ObjectiveContext {
  /// The complex being scored.
  std::shared_ptr<Spacetime> spacetime;

  /// The region of `spacetime` this objective is scored over, as a vertex set.
  /// Empty means the whole complex. Scoring is always against a region rather
  /// than implicitly against "the node", so several objectives — one per pinned
  /// region, say — can coexist on one complex.
  std::set<std::uint64_t> region;

  /// The edge coordinates this objective's sums run over, as indices into the
  /// complex's edge list.
  ///
  /// Absent means every edge: the whole-cobordism scope. A present but empty
  /// list means score nothing, which is different — a region whose interior
  /// contains no edge, with the straddling edges declared out, scores no
  /// coordinate at all. Collapsing the two would silently promote such a region
  /// to the entire complex.
  ///
  /// The engine resolves this from the objective's declared `ObjectiveScope`, so
  /// an objective never recomputes edge membership from `region`.
  std::optional<std::vector<std::size_t>> scoredEdges;

  /// The target states the region is scored against, for a target-conditioned
  /// objective. Empty for a purely geometric one.
  std::vector<std::vector<std::complex<double>>> regionTargets;

  /// The register degrees the objective is declared over.
  std::vector<int> registerDegrees;

  /// The Laplacian degrees \f$k\f$ the Hodge entropy term is summed over. Each
  /// entry selects which \f$L_k\f$ the entropy
  /// \f$S_k=-\operatorname{Tr}(\rho_k\log\rho_k)\f$ is taken of.
  ///
  /// This is not a register concept. `registerDegrees` answers an unrelated
  /// question — the degrees at which a register is constructed, where a target
  /// component with no cycle to occupy scores as full leakage — and this list
  /// never reads it, neither as a default nor as a fallback.
  ///
  /// The default is \f$\{0\}\f$, the degree-zero Laplacian alone.
  std::vector<int> hodgeDegrees{0};

  /// The weight on each entry of `hodgeDegrees`, positionally and of the same
  /// length. An empty list means uniform, which is the default.
  ///
  /// The per-degree gradient norms differ by more than an order of magnitude on
  /// a real complex, so a uniform sum lets the lowest degree dominate the
  /// descent direction while the higher ones barely move the geometry. That is a
  /// property of the operator, not something this class rebalances on a caller's
  /// behalf; a caller that wants the degrees comparable declares it here.
  ///
  /// Multiplying by \f$1\f$ is exact in binary floating point, so a
  /// single-degree run with uniform weights is bit-identical to the unweighted
  /// sum.
  std::vector<double> hodgeDegreeWeights;

  /// \f$\beta_R\f$, the Regge stationarity weight.
  double reggeWeight = 1.0;
  /// \f$\eta_H\f$, the Hodge-entropy stationarity weight.
  double hodgeEntropyWeight = 0.0;
  /// \f$\eta_C\f$, the connection-entropy stationarity weight — the term that
  /// makes \f$\varphi\f$ dynamical. Zero by default, so an objective acquires a
  /// \f$\varphi\f$ gradient only when a caller declares one.
  double connectionEntropyWeight = 0.0;
  /// \f$\gamma\f$, the register-residual weight.
  double gamma = 1.0;
  /// \f$\beta_E\f$, the carried-state energy weight. Exactly zero outside the
  /// certificates-blind mean-field sub-mode.
  double carriedStateEnergyWeight = 0.0;
  /// Whether the Einstein-Hilbert term is selected.
  bool einsteinHilbert = true;
  /// Whether the node scores its blocks by fiber residuals
  /// (`MultiCobordism::useFiberResiduals`), whose stage-2 direction is the
  /// analytic band-derivative ascent rather than a numerical difference of
  /// \f$ r_U \f$ over every edge coordinate. Under fiber residuals the residual
  /// is descended alongside the bulk term instead of only gating the line
  /// search. False by default.
  bool fiberResiduals = false;
  /// Which entropy the Hodge term reads: the complex operator or its
  /// phase-blind entrywise ablation.
  HodgeLaplacian::EntropyPhaseMode hodgeEntropyPhaseMode =
      HodgeLaplacian::EntropyPhaseMode::IncludeComplexPhase;

  /// \f$r_U\f$ on this region, computed by the engine and passed as a number
  /// rather than as a callable, so no node is reachable from here. Computed only
  /// when the objective declares `needsRegisterResidual`; NaN otherwise, never a
  /// silent zero.
  double registerResidual = std::numeric_limits<double>::quiet_NaN();
  /// \f$E_{\rm carried}(\Gamma,g)\f$, likewise a precomputed number. Exactly
  /// zero where the weight is zero.
  double carriedStateEnergy = 0.0;

  /// The names of every field above, in declaration order — the firewall list a
  /// structural test asserts against, as `objectiveTermNames` does for the
  /// output side.
  [[nodiscard]] static std::vector<std::string> inputNames();
};

/// # HodgeDegreeContribution
///
/// One degree's share of the Hodge stationarity term, so a reader can tell
/// which degree the descent came from rather than only the total — the same
/// discipline `MultiCobordism::ObjectiveContribution` applies to bulk versus
/// pinned-region objectives.
struct HodgeDegreeContribution {
  /// The Laplacian degree \f$k\f$.
  int degree = 0;
  /// The declared weight on this degree.
  double weight = 1.0;
  /// \f$\|\nabla_zS_k\|^2\f$ over the edges in scope, unweighted, so the raw
  /// spread across degrees is visible rather than folded into the weighting.
  double gradientNormSquared = 0.0;
  /// This degree's share of `ObjectiveTerms::hodgeStationarity`: the entropy
  /// weight times the degree weight times the norm above.
  ///
  /// Summing this member over the contributions reproduces the term to double
  /// round-off, not to the bit: the term applies the entropy weight once to the
  /// accumulated weighted norms, whereas each share here carries its own
  /// multiply.
  double contribution = 0.0;
};

/// # ObjectiveDirection
///
/// A stage-2 search direction together with the exact objective value at the
/// point it was taken from. `baselineComputed` is false when the objective did
/// not assemble its scalar while building the direction, in which case the
/// engine evaluates the scalar itself rather than trusting an accumulated
/// stage-1 trace.
struct ObjectiveDirection {
  /// The ascent displacement. Stage 2 subtracts a scaled multiple of it.
  Eigen::VectorXcd ascent;
  /// The ascent displacement in the connection phase \f$\varphi\f$, in the same
  /// edge order. Empty when the objective has no \f$\varphi\f$ dependence,
  /// which is the case for every functional of \f$ L_k \f$ alone. Stage 2
  /// subtracts a scaled multiple of it from the stored phases under the same
  /// line search and step scale that moves \f$ z \f$ — one search over both
  /// fields.
  Eigen::VectorXcd phaseAscent;
  /// The exact objective at the current point, when the direction's assembly
  /// already produced it.
  double baseline = 0.0;
  /// Whether `baseline` is meaningful.
  bool baselineComputed = false;
};

/// # ObjectiveDirectionContext
///
/// `ObjectiveContext` plus the extra data a stage-2 direction needs. Plain
/// data for the same reason, so the direction path cannot reach a node either.
struct ObjectiveDirectionContext {
  /// The scalar inputs, unchanged.
  ObjectiveContext scalar;
  /// The number of edge coordinates the direction is taken over.
  std::size_t edgeCount = 0;
  /// \f$\partial E_{\rm carried}/\partial z\f$, exact and analytic, computed by
  /// the engine. Empty where the carried-state weight is zero.
  std::vector<std::complex<double>> carriedStateEnergyGradient;
};

/// # RegionHandle
///
/// A reference to a declared pinned region. The only non-empty handle comes
/// from `MultiCobordism::regionHandle`, which looks the name up among the
/// declared regions and throws by name if it is not there. A bare
/// `std::string region` would let `"boundary"` and `"boundry"` both compile,
/// one of them silently scoring nothing; a handle cannot be spelled, only
/// obtained.
class RegionHandle {
 public:
  /// The whole cobordism — the default, and what declaring nothing produces.
  RegionHandle() = default;

  /// Whether this handle references the whole cobordism rather than a region.
  [[nodiscard]] bool isWholeCobordism() const noexcept { return name_.empty(); }
  /// The declared region's name, for reporting. Empty for the whole cobordism.
  [[nodiscard]] const std::string &name() const noexcept { return name_; }

  [[nodiscard]] bool operator==(const RegionHandle &other) const noexcept {
    return name_ == other.name_;
  }

 private:
  /// Only the engine mints a handle, and only for a region it has verified is
  /// declared.
  friend class MultiCobordism;
  explicit RegionHandle(std::string name) : name_(std::move(name)) {}
  std::string name_;
};

/// # ObjectiveScope
///
/// What an objective declares that it references: a pinned region, or — by
/// declaring nothing, the default — the whole cobordism. The engine honours the
/// declaration rather than inferring one from the objective's role.
///
/// Scope is independent of whether the referenced region's coordinates are
/// frozen. Pinning does two unrelated jobs: it names a region so an objective
/// can reference it, and it constrains relaxation by zeroing a pinned edge's
/// descent component before the line search. A pinned edge does not vary and is
/// still scored, and the bulk objective scores the pinned interior too.
struct ObjectiveScope {
  /// The region referenced. Default-constructed means the whole cobordism.
  /// Obtainable only from `MultiCobordism::regionHandle`, so it cannot name a
  /// region that was never declared.
  RegionHandle region;

  /// Whether edges with a single endpoint in `region` — the straddling edges —
  /// enter this objective's score. Part of the same scope declaration rather
  /// than a separate mechanism, and meaningless for a whole-cobordism scope,
  /// which has no border to straddle.
  ///
  /// A region-scoped objective will normally declare `false`, so the edges tying
  /// its region to the bulk are scored by the bulk's objective and not twice.
  /// `MultiCobordism::edgeIsPinned` holds exactly when a single region contains
  /// both endpoints, so a straddling edge is one with a single endpoint in the
  /// region.
  bool includesStraddlingEdges = true;

  /// Whether this scope is the whole cobordism, i.e. nothing was declared.
  [[nodiscard]] bool isWholeCobordism() const {
    return region.isWholeCobordism();
  }
};

/// # ObjectiveName
///
/// The identifiers objectives are known by, as named constants. A typo in a
/// repeated string literal would not fail to compile; it would fail to match.
class ObjectiveName {
 public:
  static constexpr const char *kJointStationarity = "joint_stationarity";
  static constexpr const char *kLegacy = "legacy";
  static constexpr const char *kMediatedCorrespondence =
      "mediated_correspondence";
};

/// # ObjectiveTermName
///
/// The declared term slots, likewise named. `CobordismObjective::
/// declaredTermNames` is assembled from these, so the list and the constants
/// cannot drift apart.
class ObjectiveTermName {
 public:
  static constexpr const char *kReggeStationarity = "regge_stationarity";
  static constexpr const char *kHodgeStationarity = "hodge_stationarity";
  static constexpr const char *kConnectionStationarity =
      "connection_stationarity";
  static constexpr const char *kRegisterResidual = "register_residual";
  static constexpr const char *kActionMagnitude = "action_magnitude";
  static constexpr const char *kCarriedStateEnergy = "carried_state_energy";
};

/// # CobordismObjective
///
/// The functional `MultiCobordism` descends, as an injected specification
/// rather than a value of a closed enum. An implementation declares the terms
/// it is the sum of, decomposes itself over a region of a complex, and supplies
/// a stage-2 search direction. It knows nothing about the engine that drives it,
/// and the engine knows nothing about which objective it holds.
///
/// Scoring is against a region, never implicitly against a whole node, so more
/// than one objective may coexist on a single complex.
class CobordismObjective {
 public:
  virtual ~CobordismObjective() = default;

  /// A stable identifier, stamped on records so a run says what it descended.
  [[nodiscard]] virtual std::string name() const = 0;

  /// The complete, enumerable term list this objective is the sum of. The
  /// names are the record's keys.
  [[nodiscard]] virtual std::vector<std::string> termNames() const = 0;

  /// Decompose this objective over the context's region.
  [[nodiscard]] virtual ObjectiveTerms terms(
      const ObjectiveContext &context) const = 0;

  /// The stage-2 search direction over the context's region.
  [[nodiscard]] virtual ObjectiveDirection direction(
      const ObjectiveDirectionContext &context) const = 0;

  /// This objective's Hodge stationarity term broken down by degree, or an
  /// empty list for an objective that has no such term.
  ///
  /// Reported separately from `terms` because `ObjectiveTerms` is a fixed record
  /// of scalars that `total` is static over: a per-degree breakdown decomposes
  /// one of those scalars rather than adding a term.
  [[nodiscard]] virtual std::vector<HodgeDegreeContribution>
  hodgeDegreeContributions(const ObjectiveContext &) const {
    return {};
  }

  /// Whether this objective's value depends on prescribed target states rather
  /// than on the geometry alone. A search policy that must stay unforced
  /// consults this rather than testing for an objective by name.
  [[nodiscard]] virtual bool isTargetConditioned() const = 0;

  /// What this objective references: a named pinned region, or — by declaring
  /// nothing — the whole cobordism. The engine honours the declaration rather
  /// than inferring one from the objective's role.
  ///
  /// Scope is a property of the instance, not of the class, so an existing
  /// objective can be pointed at a region without writing a new type. An
  /// implementation may still override this where its scope is intrinsic.
  [[nodiscard]] virtual ObjectiveScope scope() const { return scope_; }

  /// Declare what this objective references. Default-constructed means the
  /// whole cobordism, which is what an objective that never calls this
  /// declares.
  void setScope(ObjectiveScope scope) { scope_ = std::move(scope); }

  /// Whether this objective reads \f$r_U\f$. The engine computes that residual
  /// only when an objective asks for it, so a purely geometric objective never
  /// pays for a target-conditioned quantity it does not use.
  [[nodiscard]] virtual bool needsRegisterResidual() const { return false; }

  /// Whether a candidate move's objective change may be scored by a localized
  /// exact delta instead of by re-evaluating the whole functional.
  ///
  /// An objective built from global spectra or action magnitudes changes
  /// everywhere when one cell changes, so only its true scalar difference is
  /// correct and the engine pays for a full evaluation per candidate. An
  /// objective assembled from per-cell contributions can instead be differenced
  /// exactly over the cells a move touches. `false` is always correct and merely
  /// more expensive, hence the default.
  [[nodiscard]] virtual bool supportsLocalizedDelta() const { return false; }

  /// The lowest register degree over which this objective is declared. The
  /// engine refuses to install it on a node carrying a lower degree, so the
  /// restriction travels with the objective rather than living in the engine as
  /// a special case. Zero — no restriction — is the default.
  [[nodiscard]] virtual int minimumRegisterDegree() const { return 0; }

  /// The weight this objective puts on a numerically differentiated
  /// register-residual direction, given its configuration; zero for an objective
  /// that supplies an analytic direction for every term it has.
  ///
  /// Returned as a weight rather than applied here: handing an objective a
  /// callable that could difference a scalar over edge coordinates would mean
  /// handing it a closure over the node. The engine applies this weight to its
  /// own differentiation of \f$r_U\f$.
  [[nodiscard]] virtual double numericalRegisterResidualWeight(
      const ObjectiveContext &) const {
    return 0.0;
  }

  /// The scalar: the plain sum of the declared terms. Static, so the collapse to
  /// the optimizer's number has no `this` and cannot reach any state.
  [[nodiscard]] static double total(const ObjectiveTerms &terms);

  /// The declaration order of `ObjectiveTerms`' members. Every objective records
  /// into the same slots, so records stay comparable across objectives and a
  /// structural test can assert the list.
  [[nodiscard]] static std::vector<std::string> declaredTermNames();

 private:
  /// The declared scope. Default-constructed is the whole cobordism.
  ObjectiveScope scope_;
};

/// # JointStationarityObjective
///
/// Reference: Regge and Williams, "Discrete structures in gravity",
/// arXiv:gr-qc/0012035
/// Reference: De Domenico and Biamonte, "Spectral entropies as
/// information-theoretic tools for complex network comparison",
/// arXiv:1609.01214
///
/// \f$\beta_R\|\nabla_zS_{\rm Regge}\|^2+\eta_H\sum_k\|\nabla_zS_{{\rm
/// Hodge},k}\|^2\f$ — the Regge action and the Hodge spectral entropy
/// stationary at the same metric. The only one of the three built-ins that is
/// not target-conditioned.
///
/// This is a scalar diagnostic that a descent minimizes: every term is the
/// squared norm of the gradient of a real functional, so its minimum records
/// how nearly two real functionals are simultaneously stationary and its value
/// is a residual rather than an action. The holomorphic stationarity equations
/// of the joint action \f$S(z,U,\Gamma)\f$ themselves — the complex equations
/// \f$\partial S/\partial z=0\f$ and \f$U\,\partial S/\partial U=0\f$, with no
/// real projection selected and no norm minimized in their place — are
/// `JointAction` and are solved by `HolomorphicRelaxation`. The two answer
/// different questions and neither stands in for the other.
///
/// The Hodge sum runs over `ObjectiveContext::hodgeDegrees`, resolved
/// independently of the register degrees. Scoring more degrees shows more of the
/// spectrum, not more of the topology: exact zero modes are omitted from the
/// entropy and from its derivative, so each degree's term sits on the fixed-rank
/// stratum the current topology selects and is blind to a change in its own
/// kernel dimension — one blind spot per degree, one per Betti number. Only
/// stage-1 move acceptance sees a topology change, by re-evaluating the
/// functional after the move.
class JointStationarityObjective final : public CobordismObjective {
 public:
  [[nodiscard]] std::string name() const override;
  [[nodiscard]] std::vector<std::string> termNames() const override;
  [[nodiscard]] ObjectiveTerms terms(
      const ObjectiveContext &context) const override;
  [[nodiscard]] ObjectiveDirection direction(
      const ObjectiveDirectionContext &context) const override;
  [[nodiscard]] std::vector<HodgeDegreeContribution> hodgeDegreeContributions(
      const ObjectiveContext &context) const override;
  [[nodiscard]] bool isTargetConditioned() const override { return false; }
  /// A declared domain restriction, not a capability limit: the degree-zero
  /// \f$L_0=d_1W_1^{-1}d_1^{\mathsf T}\f$ is holomorphic in \f$z\f$ and its
  /// entropy gradient is exact, so the gradient exists at degree zero too.
  [[nodiscard]] int minimumRegisterDegree() const override { return 1; }
};

/// # LegacyObjective
///
/// \f$\beta_R\|\nabla_zS_{\rm Regge}\|^2+\gamma r_U\f$ — the compatibility
/// objective. Target-conditioned through \f$r_U\f$.
class LegacyObjective final : public CobordismObjective {
 public:
  [[nodiscard]] std::string name() const override;
  [[nodiscard]] std::vector<std::string> termNames() const override;
  [[nodiscard]] ObjectiveTerms terms(
      const ObjectiveContext &context) const override;
  [[nodiscard]] ObjectiveDirection direction(
      const ObjectiveDirectionContext &context) const override;
  [[nodiscard]] bool isTargetConditioned() const override { return true; }
  [[nodiscard]] bool needsRegisterResidual() const override { return true; }
  /// The Regge half has an exact per-cell delta and the register half is an
  /// exact \f$\gamma\,\Delta r_U\f$, so a candidate move is scored by
  /// differencing over the cells it touches rather than by re-evaluating the
  /// whole functional.
  [[nodiscard]] bool supportsLocalizedDelta() const override { return true; }
  [[nodiscard]] double numericalRegisterResidualWeight(
      const ObjectiveContext &context) const override;
};

/// # MediatedCorrespondenceObjective
///
/// \f$r_U+\beta|S_{\rm Regge}(W^*)|\f$ — the operator-cobordism experiment.
/// Target-conditioned through \f$r_U\f$.
class MediatedCorrespondenceObjective final : public CobordismObjective {
 public:
  [[nodiscard]] std::string name() const override;
  [[nodiscard]] std::vector<std::string> termNames() const override;
  [[nodiscard]] ObjectiveTerms terms(
      const ObjectiveContext &context) const override;
  [[nodiscard]] ObjectiveDirection direction(
      const ObjectiveDirectionContext &context) const override;
  [[nodiscard]] bool isTargetConditioned() const override { return true; }
  [[nodiscard]] bool needsRegisterResidual() const override { return true; }
  [[nodiscard]] double numericalRegisterResidualWeight(
      const ObjectiveContext &context) const override;
};

}  // namespace tessera::cobordism

#endif  // TESSERA_COBORDISM_COBORDISMOBJECTIVE_H
