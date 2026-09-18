// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_OBSERVABLES_PERSISTENTMODULARITY_H
#define TESSERA_OBSERVABLES_PERSISTENTMODULARITY_H

#include <array>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include "observables/Record.h"

// === tessera subsystem ns fwd-decls ===
namespace tessera::graph {}
namespace tessera::mesh {}
namespace tessera::quantum {}
namespace tessera::simulations {}
namespace tessera::spacetime {
  class Spacetime;
}
namespace tessera::observables {
using namespace ::tessera::mesh;
using namespace ::tessera::graph;
using namespace ::tessera::spacetime;
using namespace ::tessera::simulations;
using namespace ::tessera::quantum;

/// Stable, label-free identity of a discovered component.
///
/// The hash derives from the oriented incidence structure of the component's
/// children and its parent lineage — child hashes feed the parent hash — and
/// never from raw vertex or cell numbers. It is used for persistence matching
/// and deterministic tie-breaking, never as a physical observable. Two
/// structurally identical (automorphic) components share a hash by
/// construction; bookkeeping that must tell such twins apart, such as cache
/// invalidation, is positional (see :class:`InvalidationRead`).
///
/// ``level`` is the multilevel-aggregation depth at which the component was
/// formed: level 0 is an input cell, level ``k`` a community formed at the
/// ``k``-th aggregation round.
class ComponentId {
public:
  ComponentId() = default;
  ComponentId(std::string hash, std::size_t level)
      : hash_(std::move(hash)), level_(level) {}

  /// The canonical structural hash (32 lowercase hex characters).
  std::string canonicalHash() const { return hash_; }
  /// Multilevel-aggregation depth at which this component was formed.
  std::size_t level() const { return level_; }

  bool operator==(const ComponentId &o) const noexcept {
    return level_ == o.level_ && hash_ == o.hash_;
  }
  bool operator!=(const ComponentId &o) const noexcept {
    return !(*this == o);
  }
  /// Deterministic label-free ordering: (level, hash).
  bool operator<(const ComponentId &o) const noexcept {
    if (level_ != o.level_) return level_ < o.level_;
    return hash_ < o.hash_;
  }

private:
  std::string hash_;
  std::size_t level_ = 0;
};

/// Which search proposes the communities. Both score the same exact
/// \f$ Q_\gamma \f$ closed form, so their results are comparable on one
/// scale; they differ only in how a partition is searched for.
///
/// Reference: Newman, "Modularity and community structure in networks",
/// arXiv:physics/0602124; Blondel, Guillaume, Lambiotte, Lefebvre, "Fast
/// unfolding of communities in large networks", arXiv:0803.0476.
enum class DiscoveryStrategy {
  /// Multilevel aggregation from a fixed restart seed sequence, keeping the
  /// best exact score and reporting the restart spread.
  MultilevelAggregation,
  /// Leading-eigenvector bisection of the modularity matrix
  /// \f$ B_\gamma = A - \gamma k k^{T} / (2m) \f$, recursed until no group
  /// has a positive leading eigenvalue. The community count is fixed by the
  /// spectrum rather than by a caller-supplied parameter, and the search
  /// carries no seed.
  LeadingEigenvector,
};

/// Which real functional of the complex \f$ Q \f$ the search maximizes. Both
/// are always reported; this chooses only what is pursued.
enum class ModularityObjective {
  /// Maximize \f$ Q \f$ itself. Available only where \f$ Q \f$ is real, i.e.
  /// on a real adjacency, and the default there. It finds community structure;
  /// an anti-community scores below the one-community partition and is passed
  /// over.
  Score,
  /// Maximize \f$ |Q| \f$. Always available, and the only ordering once
  /// \f$ A \f$ is genuinely complex. It finds both community and
  /// anti-community structure, with \f$ \arg Q \f$ saying which was found.
  /// Selected automatically on a complex graph, where `Score` is not an
  /// ordering.
  Magnitude,
};

/// Configuration for the label-free multiscale component discovery.
struct PersistentModularityConfig {
  /// Which search proposes the communities.
  DiscoveryStrategy strategy = DiscoveryStrategy::MultilevelAggregation;
  /// Which real functional of the complex \f$ Q \f$ the search maximizes. On a
  /// complex graph ``Score`` is not an ordering, so ``Magnitude`` is selected
  /// regardless, with ``ResolutionSlice::objective`` reporting which was used.
  ///
  /// Setting ``Magnitude`` on a real graph is how anti-community structure is
  /// pursued: an anti-community has \f$ Q < 0 \f$, so maximizing \f$ Q \f$
  /// passes it over in favour of the one-community partition while maximizing
  /// \f$ |Q| \f$ finds it. Both readings are always reported.
  ModularityObjective objective = ModularityObjective::Score;
  /// LeadingEigenvector only: a group is indivisible when its leading
  /// eigenvalue does not exceed this. \f$ B_\gamma \f$ restricted to a group
  /// always annihilates the all-ones vector, so zero is always in the spectrum
  /// and the leading eigenvalue is never negative; "no positive eigenvalue"
  /// therefore means "at or below this tolerance".
  double leadingEigenvalueTolerance = 1e-9;
  /// LeadingEigenvector only: the minimum leading-to-second eigenvalue gap for
  /// a bisection to count as well determined. Below it the split is refused and
  /// reported unresolved with a named reason, rather than taken on an
  /// ill-conditioned eigenvector.
  double minEigenvalueGap = 1e-8;
  /// LeadingEigenvector only: groups of at most this many cells get an exact
  /// dense symmetric eigendecomposition of \f$ B_\gamma \f$ restricted to the
  /// group; larger groups fall back to shifted power iteration.
  ///
  /// The dense path exists because power iteration separates the leading pair
  /// at a rate set by their ratio, so a near-degenerate pair — exactly the case
  /// the gap certificate must adjudicate — is where iteration is slowest and
  /// least trustworthy. Deciding whether a pair is degenerate by an iteration
  /// that converges only when it is not would be circular.
  std::size_t denseEigenSolveMaxGroup = 1024;
  /// LeadingEigenvector only: hard cap on power-iteration steps per
  /// eigenpair on groups above ``denseEigenSolveMaxGroup``.  Non-convergence
  /// is reported, never silently accepted.
  int maxPowerIterations = 4096;
  /// LeadingEigenvector only: relative convergence tolerance of the power
  /// iteration's Rayleigh quotient.
  double powerIterationTolerance = 1e-12;
  /// LeadingEigenvector only: run a Kernighan-Lin style local refinement after
  /// each sign bisection. Without it the method scores measurably lower
  /// \f$ Q_\gamma \f$ than multilevel aggregation.
  bool kernighanLinRefinement = true;
  /// Resolution parameters \f$ \gamma \f$ for the scan, in scan order.
  /// Adjacent entries are matched into persistence tracks.
  std::vector<double> resolutions{1.0};
  /// Base of the fixed restart seed sequence: restart ``t`` uses seed
  /// ``splitmix64(baseSeed + t)``.  Deterministic by construction.
  std::uint64_t baseSeed = 0;
  /// Number of deterministic multilevel restarts per resolution. The best exact
  /// score is retained and the spread across restarts is reported; modularity
  /// maximization is NP-hard, so no claim of a global optimum is made.
  int restarts = 4;
  /// Hard cap on local-move sweeps per aggregation level.
  int maxSweepsPerLevel = 64;
  /// Minimum support overlap (Jaccard) for a persistence track to continue
  /// across adjacent resolutions.
  double overlapThreshold = 0.5;
};

/// One discovered component: canonical id, level-0 cell support, cached
/// sufficient statistics, and the exact per-component scores derived from
/// them.
struct ComponentRead {
  ComponentId id;
  /// Level-0 member cell ids. A set, listed ascending for reporting only: the
  /// ordering carries no convention and the identity never derives from these
  /// numbers.
  std::vector<std::uint64_t> support;
  /// \f$ \Sigma_{\mathrm{in}} \f$: total internal adjacency weight
  /// \f$ A(C,C) \f$ counting both ordered directions, including the self-loop
  /// convention of aggregated levels. Complex-valued; its argument is the
  /// causal character of this community's cohesion.
  std::complex<double> internalWeight{0.0, 0.0};
  /// \f$ S_C \f$: summed degree (strength) of the members; complex like the
  /// degrees.
  std::complex<double> strength{0.0, 0.0};
  /// Weighted conductance
  /// \f$ \mathrm{cut}(C) / \min(\mathrm{vol}\,C, \mathrm{vol}\,(V
  /// \setminus C)) \f$; 0 by convention when the denominator vanishes, i.e.
  /// for the whole-graph or empty community.
  ///
  /// NaN on a signed graph, where it is undefined: conductance is a ratio of
  /// volumes, and a signed community's strength is a difference, so there is
  /// nothing for the cut to be a fraction of. Left unmeasured rather than
  /// computed by a formula that does not apply.
  double conductance = 0.0;
  /// This community's exact additive term of \f$ Q_\gamma \f$,
  /// \f$ (\Sigma_{\mathrm{in}} - \gamma S_C^2 / S_A) / T \f$. These sum over
  /// a level's communities to that level's exact \f$ Q_\gamma \f$, so they are
  /// complex whenever it is.
  std::complex<double> modularityContribution{0.0, 0.0};
};

/// Named outcomes of one attempted leading-eigenvector bisection.
///
/// Named constants rather than literals at each site: a misspelling would
/// compile but produce a reason no consumer matches. Bound to Python so a
/// caller references the constant instead of retyping the string.
struct SplitReason {
  /// The group was bisected and the split raised \f$ Q_\gamma \f$.
  static constexpr const char *kSplitAccepted = "split-accepted";
  /// The leading eigenvalue is at or below the tolerance: no bisection of this
  /// group raises \f$ Q_\gamma \f$. This is the stopping rule of the
  /// leading-eigenvector method and an ordinary outcome; the group is
  /// indivisible, not defective.
  static constexpr const char *kNoPositiveEigenvalue = "no-positive-eigenvalue";
  /// The leading and second eigenvalues are separated by less than the declared
  /// minimum gap, so the leading eigenvector — and the sign pattern the
  /// bisection would use — is not well determined. The split is refused rather
  /// than taken on an ill-conditioned vector.
  static constexpr const char *kDegenerateLeadingPair =
      "degenerate-leading-pair";
  /// Fewer than two cells: nothing to bisect.
  static constexpr const char *kGroupTooSmall = "group-too-small";
  /// The eigenvector's sign pattern put every cell on one side, so the proposed
  /// bisection is not a bisection.
  static constexpr const char *kEmptySide = "empty-side";
  /// The bisection was well determined but did not raise \f$ Q_\gamma \f$.
  static constexpr const char *kSplitLowersModularity =
      "split-lowers-modularity";
  /// The power iteration hit ``maxPowerIterations`` without meeting
  /// ``powerIterationTolerance``. Reported, never silently accepted.
  static constexpr const char *kPowerIterationNotConverged =
      "power-iteration-not-converged";
};

/// One attempted bisection of the leading-eigenvector search: the spectrum
/// that decided it, and what was decided.
///
/// This is the strategy's certificate. The leading-to-second eigenvalue gap
/// measures how well determined the bisection is, in the same way the rest of
/// this layer certifies spectral isolation: separation measured in the
/// spectrum, never to a sort-order neighbour. The multilevel strategy's
/// corresponding measure is ``ResolutionSlice::restartSpread``.
///
/// Unmeasured quantities are NaN, never zero. A group that was never bisected
/// because it was too small has no eigenvalues, and says so.
struct SplitRead {
  /// Number of level-0 cells in the group that was examined.
  std::size_t groupSize = 0;
  /// Most positive eigenvalue of \f$ B_\gamma \f$ restricted to the group,
  /// over the complement of the all-ones vector. NaN when not computed.
  double leadingEigenvalue = std::numeric_limits<double>::quiet_NaN();
  /// Second most positive eigenvalue, by deflation.  NaN when not computed.
  double secondEigenvalue = std::numeric_limits<double>::quiet_NaN();
  /// ``leadingEigenvalue - secondEigenvalue``: how well determined the
  /// bisection is. NaN when either eigenvalue is unmeasured.
  double eigenvalueGap = std::numeric_limits<double>::quiet_NaN();
  /// Exact change in total \f$ Q_\gamma \f$ this split would produce, from
  /// this class's closed form. NaN when no split was evaluated.
  double deltaQ = std::numeric_limits<double>::quiet_NaN();
  /// Whether the group was actually bisected.
  bool accepted = false;
  /// Whether the spectrum determined the outcome. False means the split was
  /// refused because the answer was not well determined — a degenerate pair or
  /// a non-converged iteration. This differs from a determined "do not split",
  /// which is resolved but not accepted.
  bool resolved = true;
  /// One of :class:`SplitReason`.
  std::string reason;
  /// Cell counts of the two sides when accepted; 0 otherwise.
  std::size_t sizeA = 0;
  std::size_t sizeB = 0;
};

/// One deterministic restart: its seed and exact best score.
struct RestartRead {
  std::uint64_t seed = 0;
  /// Exact \f$ Q_\gamma \f$ of this restart's final partition, recomputed
  /// from scratch; complex and unreduced.
  std::complex<double> q{0.0, 0.0};
  /// The real scalar this restart was ranked on, derived from ``q`` by the
  /// slice's objective.
  double objectiveValue = 0.0;
  std::size_t communities = 0;
};

/// The discovery result at one resolution \f$ \gamma \f$.
struct ResolutionSlice {
  double gamma = 1.0;
  /// Which search produced this slice.
  DiscoveryStrategy strategy = DiscoveryStrategy::MultilevelAggregation;
  /// LeadingEigenvector only: one entry per attempted bisection, in the order
  /// attempted; the strategy's spectral certificate. Empty for
  /// MultilevelAggregation, which performs no bisections.
  std::vector<SplitRead> splits;
  /// Exact \f$ Q_\gamma \f$ of the winning partition, recomputed from scratch
  /// from the final labels: the best score across the deterministic restarts.
  /// A heuristic proposal, not a global optimum.
  ///
  /// Complex and unreduced. \f$ |q| \f$ is how much structure the partition
  /// has and \f$ \arg q \f$ is what kind: 0 a community, \f$ \pi \f$ an
  /// anti-community, \f$ \pm\pi/2 \f$ lightlike cohesion, anything else
  /// mixed. Exactly real on a real graph. Reported folded rather than split
  /// into two differently-typed halves, which would invite treating one as
  /// primary.
  std::complex<double> q{0.0, 0.0};
  /// The winning restart's incrementally accumulated score:
  /// \f$ Q_0 \f$ plus the sum of accepted exact \f$ \Delta Q \f$, by
  /// compensated summation. Agrees with ``q`` to double round-off.
  std::complex<double> qIncremental{0.0, 0.0};
  /// The real scalar the search maximized, derived from ``q``: ``q.real()``
  /// under ``Score`` and ``abs(q)`` under ``Magnitude``. Restarts are compared
  /// on this, and ``restartSpread`` measures it.
  double objectiveValue = 0.0;
  /// Which functional that was. Reported rather than assumed, because a complex
  /// graph selects ``Magnitude`` whatever the configuration asked for; ``Score``
  /// is not an ordering there.
  ModularityObjective objective = ModularityObjective::Score;
  /// Number of aggregation levels in the winning run's hierarchy.
  std::size_t levels = 0;
  /// Final-level components of the winning partition, ordered by canonical
  /// hash.
  std::vector<ComponentRead> components;
  /// The full multilevel hierarchy of the winning run: ``hierarchy[k]`` are
  /// the communities formed at aggregation level ``k + 1``, each ordered by
  /// canonical hash.  ``hierarchy.back() == components``.
  std::vector<std::vector<ComponentRead>> hierarchy;
  /// Every restart's exact score, in seed-sequence order. Empty under
  /// LeadingEigenvector, which carries no seed and does not restart.
  std::vector<RestartRead> restarts;
  /// Range (max minus min) of the restart scores: the restart uncertainty of
  /// the heuristic search. NaN under LeadingEigenvector, which has no restart
  /// spread to report.
  double restartSpread = 0.0;
};

/// A matched component pair across adjacent resolutions or across cobordism
/// time (two reports over a common cell-id universe).
struct ComponentMatch {
  ComponentId from;
  ComponentId to;
  /// Indices of the matched components in their source containers (the
  /// positional disambiguation for automorphic twins that share a hash).
  std::size_t fromIndex = 0;
  std::size_t toIndex = 0;
  /// Jaccard overlap of the level-0 cell supports.
  double supportOverlap = 0.0;
  /// Spectral-projector overlap. Populated only when a projector-overlap hook
  /// has been installed via
  /// :func:`PersistentModularity::setProjectorOverlapHook`. Absent (`nullopt`)
  /// means unknown, never zero.
  std::optional<double> projectorOverlap;
};

/// A component track across cobordism frames: the same emergent support
/// followed through consecutive frames by maximum support overlap.
///
/// Distinct from :class:`PersistenceTrack`, which follows a component across
/// the modularity resolution slices of a single frame. The fiber-acceptance
/// conjunct "lifetime across multiple cobordism frames" is ``frames()`` here. A
/// resolution-slice count says how stable a modularity proposal is under the
/// resolution parameter, and nothing about how long anything lived.
struct FrameTrack {
  /// One member per covered frame, consecutive from ``firstFrame``.
  std::vector<ComponentId> members;
  /// Positional index of each member within its frame's component list.
  std::vector<std::size_t> memberIndices;
  std::size_t firstFrame = 0;
  std::size_t lastFrame = 0;
  /// Smallest adjacent-frame support overlap along the track (1.0 for a
  /// single-frame track, which has no adjacent pair).
  double minAdjacentOverlap = 1.0;
  /// Number of consecutive cobordism frames covered: the fiber lifetime.
  [[nodiscard]] std::size_t frames() const noexcept { return members.size(); }
};

/// A component track across the resolution scan: the same emergent support
/// followed through consecutive slices by maximum support overlap.
struct PersistenceTrack {
  /// One member per covered slice, consecutive from ``firstSlice``.
  std::vector<ComponentId> members;
  /// Positional index of each member within its slice's final components.
  std::vector<std::size_t> memberIndices;
  std::size_t firstSlice = 0;
  std::size_t lastSlice = 0;
  double gammaFirst = 0.0;
  double gammaLast = 0.0;
  /// Smallest adjacent-slice support overlap along the track (1.0 for a
  /// single-slice track).
  double minAdjacentOverlap = 1.0;
  /// Mean weighted conductance of the members.
  double meanConductance = 0.0;
  /// Downstream weight-aware gap, localization and persistence status. Null
  /// means unknown; unknown is never encoded as zero. The lifetime and overlap
  /// here are proposal diagnostics and neither accept nor veto a fiber.
  Record weightAwareStatus;
};

/// The full resolution-scan report.
struct ScanReport {
  std::vector<ResolutionSlice> slices;
  /// Adjacent-slice best matches (slice r -> r + 1), all r.
  std::vector<ComponentMatch> matches;
  std::vector<PersistenceTrack> tracks;
};

/// Components and tracks invalidated by a local change (see
/// :func:`PersistentModularity::invalidatedAncestry`).  Positions
/// disambiguate automorphic twins that share a canonical hash.
struct InvalidationRead {
  /// Unique invalidated component ids.
  std::vector<ComponentId> components;
  /// (slice, hierarchy level index, index in level) of every invalidated
  /// component.  Hierarchy level index k refers to aggregation level k + 1.
  std::vector<std::array<std::size_t, 3>> positions;
  /// Indices into ``ScanReport::tracks`` of the affected tracks.
  std::vector<std::size_t> tracks;
};

/// Label-free discovery of connected modular components that persist across
/// resolution and cobordism time.
///
/// References: Newman, "Modularity and community structure in networks",
/// arXiv:physics/0602124; Blondel, Guillaume, Lambiotte, Lefebvre, "Fast
/// unfolding of communities in large networks", arXiv:0803.0476.
///
/// ## Domain and exact identities
///
/// The input is a finite complex-weighted undirected graph: the complex's
/// one-skeleton under a weight map (see :class:`WeightMap`). \f$ A \f$ is
/// complex symmetric, \f$ A_{ij} = A_{ji} \f$, because a weight is a property
/// of the edge and its magnitude and argument do not depend on which end it is
/// read from.
///
/// With complex degrees \f$ k_i = \sum_j A_{ij} \f$, their total
/// \f$ S_A = \sum_i k_i \f$, and the real positive scale
/// \f$ T = \sum_{ij} |A_{ij}| \f$:
///
/// \f[
///   Q_\gamma(P) = \frac{1}{T} \sum_{ij}
///     \left( A_{ij} - \gamma \frac{k_i k_j}{S_A} \right) [c_i = c_j],
/// \f]
///
/// computed from the per-community sufficient statistics
/// \f$ Q = \sum_c (\Sigma_{\mathrm{in}}(c) - \gamma S_c^2 / S_A) / T \f$
/// under the aggregated self-loop convention
/// \f$ A_{CC} = \Sigma_{\mathrm{in}}(C) \f$. Every cached local-move gain is
/// the exact closed form
///
/// \f[
///   \Delta Q(v: a \to b) = \frac{1}{T} \left[ 2 (w_{vb} - w_{va})
///     - 2\gamma \frac{k_v (k_v + S_b - S_a)}{S_A} \right],
/// \f]
///
/// evaluated in \f$ O(\deg v) \f$ from the cached community totals, so one
/// complete local-move sweep is near \f$ O(|E|) \f$ up to revisits.
///
/// Three properties fix that arrangement of \f$ S_A \f$ and \f$ T \f$:
///
/// * \f$ k_i \f$ is the row sum of \f$ A \f$ itself, so the null model is the
///   ordinary configuration model, with expected weight
///   \f$ k_i k_j / S_A \f$, rather than a magnitude surrogate. Since
///   \f$ \sum_{ij} k_i k_j / S_A = S_A = \sum_{ij} A_{ij} \f$, the null model
///   carries the same total weight the graph does.
/// * Consequently \f$ Q_1 \f$ of the one-community partition is exactly 0 for
///   any complex \f$ A \f$ (measured at 2.4e-16 over 500 random complex
///   graphs). That anchor is what makes \f$ |Q| \f$ mean "how much structure":
///   zero is the no-structure reading, not an arbitrary offset.
/// * \f$ T \f$ is real and strictly positive whenever any edge exists, so the
///   outer scale never vanishes. \f$ S_A \f$ still can; see the degenerate
///   cases below.
///
/// On a nonnegative real graph \f$ S_A = T = 2m \f$ and the formula reduces to
/// ordinary generalized modularity. That path evaluates the real expressions
/// verbatim, so such a graph scores bit-identically rather than merely agreeing
/// to round-off: the two forms differ in floating-point association even where
/// they agree in the reals. The branch is chosen by the graph, never by a flag.
///
/// These identities are exact in double arithmetic. Incremental accumulations
/// use compensated summation and are tested against recomputation from scratch
/// at the 1e-15 to 1e-14 double round-off standard.
///
/// ## Q is complex, and both parts are read
///
/// \f$ |Q| \f$ says how much structure a partition has and is what the search
/// maximizes; \f$ \arg Q \f$ says what kind, and is carried rather than
/// discarded. It is the edge causal classification lifted to a community:
/// \f$ \arg Q = 0 \f$ is spacelike cohesion (a community),
/// \f$ \arg Q = \pi \f$ is timelike cohesion (an anti-community),
/// \f$ \arg Q = \pm\pi/2 \f$ is lightlike, anything else mixed. Collapsing
/// the two into one real number would answer both questions badly, the same
/// error as reporting a winding without its modulus.
///
/// Anti-community structure is a target, not a failure mode: \f$ Q < 0 \f$
/// means a community bound by dissimilarity, and finding it is as much the
/// point as finding ordinary communities. Which one the search pursues is
/// :class:`ModularityObjective`; both are always reported.
///
/// Nothing is refused for being indefinite. A complex weight requires no
/// classification, so an edge whose \f$ \arg(\ell^2) \f$ is generic
/// contributes what it is; under a random initialization essentially every edge
/// is like that. The refusal vocabulary is reserved for genuine absences: no
/// edges, or a degenerate edge with \f$ \ell = 0 \f$, which has no weight to
/// carry (see :func:`causalWeightAvailability`).
///
/// ## Why complex symmetric rather than Hermitian
///
/// A Hermitian \f$ A \f$ — an antisymmetric phase on some chosen edge
/// direction — yields a real \f$ Q \f$, ordered without any
/// magnitude/argument split, which would be the more convenient shape. It was
/// built and measured, and it fails on the score itself.
///
/// The direction convention comes first, and only one of the two conventions
/// has a problem. On an 8-cell complex with 19 edges and every argument
/// generic, giving the worst absolute change in \f$ Q \f$ for a fixed
/// partition and the worst absolute eigenvalue change under one relabeling:
///
/// | convention | change in Q | change in spectrum | discovered partition |
/// |---|---|---|---|
/// | index order (i < j), re-derived from the labels | 0 | 3.6e-1 | different |
/// | stored source-to-target, intrinsic and carried through | 0 | 2.2e-15 | same |
///
/// So a Hermitian operator built on the stored direction is
/// relabeling-invariant, and label-freedom alone does not decide the question.
/// The null model does. A Hermitian \f$ B \f$ needs a Hermitian null term, and
/// both candidates fail:
///
/// * with \f$ |A| \f$ degrees, \f$ Q \f$ is direction-invariant but the
///   one-community \f$ Q_1 \f$ is -1.2 on the matter fixture rather than 0. The
///   anchor is gone, so the sign of \f$ Q \f$ no longer means "better or worse
///   than no structure", which is the reading a negative \f$ Q \f$ is wanted
///   for;
/// * with \f$ P_{ij} = k_i \overline{k_j} / S_A \f$ the anchor holds (3.5e-16)
///   and \f$ Q \f$ is real and signed, but \f$ Q \f$ then depends on the stored
///   edge direction. Flipping which end of one edge is called the source moves
///   it by 4.7e-1 on a \f$ Q \f$ of magnitude 0.65. A global reversal is
///   \f$ A \to A^{T} \f$, a symmetry; the single flip is not.
///
/// The symmetric operator has neither problem: there is no direction for it to
/// depend on, and its anchor holds at 2.5e-16.
///
/// The signed-real reduction of Gomez, Jensen and Arenas, with one null model
/// per sign, agrees with the complex operator where every edge is spacelike or
/// timelike, but scores a lightlike-cohesion community identically to a
/// spacelike one (+0.5 for both), because taking \f$ \mathrm{Re}(A) \f$
/// deletes exactly the edges whose character it was meant to read.
///
/// ## Heuristic status
///
/// Global modularity maximization is NP-hard. The discovery is a deterministic
/// multilevel aggregation from a fixed seed sequence that retains the best
/// exact score and reports the restart spread. It claims no global optimum,
/// whichever weight map feeds it: seeing the causal structure makes the
/// proposal better informed, not certified. Nothing in this class may enter the
/// emergence objective, and a modularity read may not veto an otherwise
/// certified fiber; fiber acceptance rests on the independent weight-aware gap,
/// localization, leakage, persistence and anchor certificates, reported here as
/// unknown rather than zero. Communities are proposals and carry no
/// connectivity guarantee.
///
/// ## Label-freedom
///
/// Visit order and tie-breaking derive from a canonical structural ranking
/// (iterated weighted color refinement with individualization by breadth-first
/// distance), and component identity from oriented incidence and lineage, never
/// from raw vertex numbers. The discovery is a pure function of the labeled
/// graph: edge input order never changes the result. Within a refinement class
/// whose members are structurally indistinguishable, individualization picks an
/// arbitrary representative (minimum cell id); when such classes are
/// automorphism orbits, the discovered hierarchy under a relabeling is the
/// automorphic image, with identical scores, identical per-level canonical-hash
/// multisets, and supports mapped up to graph automorphism.
///
/// ## Read-only
///
/// A pure observable: it never calls a solver and never mutates the spacetime
/// it reads. Instances are immutable after construction; the canonical ranking
/// is a lazily computed per-instance cache, and the sufficient-statistics
/// caches live inside each discovery run.
class PersistentModularity {
public:
  /// The map from edge geometry to similarity weight. Nothing else enters the
  /// metric.
  enum class WeightMap {
    /// \f$ w = 1 \f$ per edge: the combinatorial one-skeleton, the same graph
    /// `SparseGraph::modularity` and `Spacetime::modularityOnSkeleton` score.
    /// Causally blind by construction; it reads no geometry at all.
    Unit,
    /// \f$ w = e^{-|\ell|} \f$ with \f$ \ell \f$ the complex edge length:
    /// monotone decreasing in the edge magnitude (under the
    /// mutual-information convention \f$ \ell = -\log I \f$), with values in
    /// (0, 1]. Causally blind: it reads only the Euclidean modulus, so a
    /// timelike and a spacelike edge of equal magnitude get the same weight.
    ExpNegAbsLength,
    /// \f$ w = e^{-|\ell|} e^{i\arg(\ell^2)} \f$: the same similarity
    /// magnitude as `ExpNegAbsLength`, carrying the edge's causal character as
    /// its argument rather than collapsing it. Spacelike edges land on the
    /// positive real axis, timelike on the negative real axis, lightlike on
    /// \f$ \pm i \f$, and a generic argument stays where it is. No
    /// classification happens, so there is no indefinite case to refuse.
    /// \f$ \arg(\ell^2) \f$ is what `Edge::squaredArgument()` reports.
    CausalPhaseExpNegAbsLength,
  };


  /// Whether a causal weight map is available on a complex, with the
  /// disposition census that decides it. `reason` is empty exactly when
  /// `available` is true.
  struct CausalWeightRead {
    std::size_t spacelike = 0;
    std::size_t timelike = 0;
    std::size_t lightlike = 0;
    /// Edges with no definite causal character, i.e. generic
    /// \f$ \arg(\ell^2) \f$. These are ordinary edges for the complex weight
    /// map, which carries their argument as it stands; the count is a
    /// diagnostic, not a gate. Under a random initialization it is close to
    /// every edge.
    std::size_t mixed = 0;
    /// Absent edges, with Euclidean \f$ |\ell| \f$ below
    /// `Edge::kDegenerateEpsilon`. Not a causal type and not scored either way.
    /// These are a genuine absence and do make the map unavailable: an edge
    /// with no extent has no argument.
    std::size_t degenerate = 0;
    bool available = false;
    /// One of :class:`CausalWeightReason` when unavailable; empty otherwise.
    std::string reason;
  };

  /// Named reasons a causal weight map is unavailable. Named constants rather
  /// than literals: the reason is produced here and compared elsewhere, and a
  /// typo in either place would still compile.
  struct CausalWeightReason {
    /// At least one edge has \f$ |\ell| \f$ below `Edge::kDegenerateEpsilon`.
    /// Such an edge is absent rather than indefinite: it has no argument to
    /// carry, and \f$ \arg 0 \f$ reads nothing.
    static constexpr const char *kDegenerateEdgeLength = "degenerate-edge-length";
    /// The complex carries no scorable edge at all.
    static constexpr const char *kNoScorableEdges = "no-scorable-edges";
  };

  /// The disposition census of `st`'s one-skeleton and whether
  /// `CausalPhaseExpNegAbsLength` can be applied to it. Read-only; a caller may
  /// ask before constructing, and :func:`fromSpacetime` throws with the same
  /// named reason when it cannot.
  ///
  /// A nonzero mixed count does not make the map unavailable: the complex
  /// weight carries a generic argument as readily as a definite one. Only a
  /// genuine absence does. The mixed fraction is a useful diagnostic; it should
  /// fall if relaxation is imposing causal character.
  static CausalWeightRead causalWeightAvailability(const Spacetime &st);

  /// Build from an explicit real weighted edge list, signed or not. Cells are
  /// identified by arbitrary 64-bit ids; the node set is the union of the
  /// endpoint ids and ``isolatedCells``. Parallel edges are consolidated by
  /// weight summation. Self-loops and zero-weight edges are ignored at level 0,
  /// including a pair whose weights cancel to zero, which is a measured absence
  /// of net similarity rather than a dropped edge.
  /// @throws std::invalid_argument on non-finite weights or mismatched
  ///   lengths.
  static PersistentModularity fromWeightedEdges(
      const std::vector<std::uint64_t> &src,
      const std::vector<std::uint64_t> &tgt,
      const std::vector<double> &weight,
      const std::vector<std::uint64_t> &isolatedCells = {});

  /// The same, for a complex weighted edge list. Consolidation, self-loops and
  /// the cancel-to-zero convention are as above, and both components must be
  /// finite. A list that happens to be real takes the real path and scores
  /// exactly as the overload above.
  static PersistentModularity fromComplexWeightedEdges(
      const std::vector<std::uint64_t> &src,
      const std::vector<std::uint64_t> &tgt,
      const std::vector<std::complex<double>> &weight,
      const std::vector<std::uint64_t> &isolatedCells = {});

  /// Build the similarity graph from the spacetime's one-skeleton (vertices
  /// and edges) under the given weight map.  Read-only on the spacetime.
  static PersistentModularity fromSpacetime(
      const Spacetime &st, WeightMap map = WeightMap::ExpNegAbsLength);

  std::size_t nCells() const noexcept { return nNodes_; }
  std::size_t nEdges() const noexcept { return nEdges_; }
  /// True when some edge weight has a nonzero imaginary part, so \f$ Q \f$ is
  /// genuinely complex and ``Score`` is not an ordering. A property of the
  /// graph, not a setting: no caller selects the branch.
  bool isComplex() const noexcept { return complex_; }
  /// True when some edge weight is negative or non-real, i.e. when the graph
  /// leaves the nonnegative regime.
  bool isSigned() const noexcept { return signed_; }
  /// \f$ T = \sum_{ij} |A_{ij}| \f$, the real positive scale the score divides
  /// by. Equal to \f$ 2m = \sum_{ij} A_{ij} \f$ on a nonnegative graph.
  double totalWeight2() const noexcept { return twoM_; }
  /// \f$ S_A = \sum_{ij} A_{ij} \f$, the complex total the configuration null
  /// model redistributes. Equal to ``totalWeight2()`` on a nonnegative graph.
  std::complex<double> totalWeightSum() const noexcept { return sumA_; }
  /// The cell ids in internal storage order, i.e. input first-appearance order.
  /// Carries no convention.
  const std::vector<std::uint64_t> &cellIds() const noexcept {
    return cellIds_;
  }

  /// Exact generalized modularity \f$ Q_\gamma \f$ of a fixed partition:
  /// ``labels[i]`` labels cell ``cellIds()[i]``, and distinct values are
  /// distinct communities. At \f$ \gamma = 1 \f$ on a `Unit`-weight graph this
  /// is the Newman-Girvan score. Community terms are combined in ascending
  /// label order.
  ///
  /// Complex in general: \f$ |Q| \f$ is how much structure the partition has
  /// and \f$ \arg Q \f$ is what kind. Exactly real on a real graph, where the
  /// imaginary part is zero rather than small.
  /// @throws std::invalid_argument when ``labels.size() != nCells()``.
  std::complex<double> modularityGamma(const std::vector<int> &labels,
                                       double gamma) const;

  /// Deterministic label-free discovery at one resolution under
  /// ``cfg.strategy``. ``cfg.resolutions`` is ignored here.
  ///
  /// ``MultilevelAggregation`` runs multilevel aggregation over ``cfg.restarts``
  /// seeds from the fixed sequence, keeping the best exact score, with ties
  /// broken by the sorted component hash lists.
  ///
  /// ``LeadingEigenvector`` recursively bisects on the sign pattern of the
  /// leading eigenvector of \f$ B_\gamma \f$ restricted to each group,
  /// stopping where no group has a positive leading eigenvalue. The community
  /// count is therefore a reading of the spectrum, not a parameter. There is no
  /// seed: the search is a pure function of the labeled graph and
  /// \f$ \gamma \f$, and every attempted bisection is reported in
  /// ``ResolutionSlice::splits`` with the eigenvalues that decided it.
  ///
  /// Both strategies score the same exact \f$ Q_\gamma \f$ closed form
  /// (:func:`modularityGamma`), so their slices are directly comparable.
  /// Neither claims a global optimum.
  ResolutionSlice discover(double gamma,
                           const PersistentModularityConfig &cfg) const;

  /// The configurable resolution-sequence scan: one slice per entry of
  /// ``cfg.resolutions``, adjacent slices matched by support overlap into
  /// persistence tracks.
  ScanReport scanResolutions(const PersistentModularityConfig &cfg) const;

  /// Match components across resolution or cobordism time by simplex-support
  /// overlap: the Jaccard index on level-0 cell ids, with both sides
  /// referencing a common cell-id universe such as the same evolving complex.
  /// For each component of ``a`` the best-overlap partner in ``b`` is emitted
  /// when the overlap is positive, with ties broken by canonical hash then
  /// position. When a projector-overlap hook is installed its value is reported
  /// per match; the matching decisions themselves are support-based.
  std::vector<ComponentMatch> matchComponents(
      const std::vector<ComponentRead> &a,
      const std::vector<ComponentRead> &b) const;

  /// Follow components across cobordism frames: ``frames[t]`` is the component
  /// list read from cobordism frame ``t`` over a common cell-id universe.
  /// Consecutive frames are matched with :func:`matchComponents` and chained
  /// into tracks by best overlap at or above ``overlapThreshold``, the same rule
  /// :func:`scanResolutions` uses for resolution slices, applied along a
  /// different axis. This supplies the fiber lifetime across cobordism frames.
  /// A component appearing in only one frame gets a one-frame track: a lifetime
  /// of one is a measured fact about the candidate, not an artifact of reading
  /// a single resolution.
  ///
  /// One track is emitted per emergent support, in first-appearance order.
  /// An empty frame list yields no tracks.
  std::vector<FrameTrack> trackAcrossFrames(
      const std::vector<std::vector<ComponentRead>> &frames,
      double overlapThreshold = 0.5) const;

  /// Interface hook for spectral-projector overlap. The callback receives the
  /// two component ids and returns their projector overlap in [0, 1]. No
  /// projector is implemented here; without a hook the field stays absent,
  /// meaning unknown.
  using ProjectorOverlapHook =
      std::function<double(const ComponentId &, const ComponentId &)>;
  void setProjectorOverlapHook(ProjectorOverlapHook hook) {
    projectorHook_ = std::move(hook);
  }

  /// Components and tracks whose ancestry a local change touches: every
  /// component, at every hierarchy level of every slice, whose support
  /// intersects ``touchedCells``, plus the tracks containing one. Siblings with
  /// disjoint support remain valid. Bookkeeping over the report only; no
  /// recomputation is triggered.
  static InvalidationRead invalidatedAncestry(
      const ScanReport &report,
      const std::vector<std::uint64_t> &touchedCells);

private:
  PersistentModularity() = default;

  // Compressed sparse row (CSR) similarity graph; undirected, with each edge
  // stored in both directions.
  std::size_t nNodes_ = 0;
  std::size_t nEdges_ = 0;
  std::vector<std::int64_t> indptr_;
  std::vector<std::uint32_t> indices_;
  std::vector<std::complex<double>> weights_;
  std::vector<std::complex<double>> strength_;  // k_i = sum_j A_ij
  double twoM_ = 0.0;                     // T = sum_ij |A_ij|  (real, > 0)
  std::complex<double> sumA_{0.0, 0.0};   // SA = sum_i k_i
  // Branch selectors, both decided by the graph and never by a caller flag.
  // `signed_` marks departure from the nonnegative regime; `complex_` marks a
  // nonzero imaginary part, which makes Q complex and Score unavailable.
  bool signed_ = false;
  bool complex_ = false;
  std::vector<std::uint64_t> cellIds_;    // internal index -> cell id
  // Consolidated edges with their stored (input) orientation: the oriented
  // incidence used for level-1 identity hashing and exact cold recomputes.
  std::vector<std::uint32_t> orientedSrc_;
  std::vector<std::uint32_t> orientedTgt_;
  std::vector<std::complex<double>> orientedW_;

  // Lazily computed canonical structure (invariant colors + visit ranks).
  mutable bool canonicalReady_ = false;
  mutable std::vector<std::uint64_t> stableColor_;  // pre-individualization
  mutable std::vector<std::uint32_t> rank_;          // canonical visit rank

  ProjectorOverlapHook projectorHook_;

  void ensureCanonical() const;

  struct LevelGraph;   // aggregated weighted graph with self-loops
  struct RunResult;    // one restart's full multilevel outcome
  RunResult runOnce(double gamma, std::uint64_t seed,
                    const PersistentModularityConfig &cfg) const;

  /// The leading-eigenvector search: recursive spectral bisection producing a
  /// single level-0 partition, plus one `SplitRead` per attempted bisection
  /// appended to `splits`.
  RunResult runLeadingEigenvector(double gamma,
                                  const PersistentModularityConfig &cfg,
                                  std::vector<SplitRead> *splits) const;

  /// Group total of the null-model degrees,
  /// \f$ S_g = \sum_{i \in g} k_i \f$; complex like the degrees themselves.
  struct GroupStrength {
    std::complex<double> total{0.0, 0.0};
  };

  /// Which real symmetric part of the complex modularity matrix a spectral
  /// step operates on.
  ///
  /// \f$ B \f$ is complex symmetric, so it has no real spectrum to take a
  /// leading eigenvalue of. \f$ \mathrm{Re}(B) \f$ and \f$ \mathrm{Im}(B) \f$
  /// are each real symmetric, each annihilates the all-ones vector on a group
  /// for the same row-sum reason \f$ B \f$ does, and each therefore admits the
  /// dense solver and the shifted iteration unchanged. Bisecting on one of them
  /// proposes a split; acceptance is by exact \f$ Q \f$ either way. A real
  /// graph has \f$ \mathrm{Im}(B) = 0 \f$ and only the real part is
  /// consulted.
  enum class ModularityPart { Real, Imaginary };

  /// The configuration null model \f$ P \f$ of the generalized modularity
  /// matrix \f$ B_\gamma = A - \gamma P \f$, with
  /// \f$ P_{ij} = k_i k_j / S_A \f$.
  ///
  /// The degrees are the row sums of \f$ A \f$ itself, so
  /// \f$ \sum_{ij} P_{ij} = S_A \f$: the null model carries exactly the total
  /// weight the graph does, which is what makes the one-community \f$ Q_1 \f$
  /// vanish. A magnitude surrogate would break that identity and with it the
  /// meaning of \f$ |Q| \f$.
  ///
  /// The reciprocal of \f$ S_A \f$ is bound once at construction so contraction
  /// stays division-free inside the power-iteration inner loop.
  class NullModel {
   public:
    explicit NullModel(const PersistentModularity &owner)
        : strength_(&owner.strength_),
          invSumA_(owner.sumA_ == std::complex<double>(0.0, 0.0)
                       ? std::complex<double>(0.0, 0.0)
                       : std::complex<double>(1.0, 0.0) / owner.sumA_),
          real_(!owner.complex_),
          inv2m_(owner.twoM_ > 0.0 ? 1.0 / owner.twoM_ : 0.0) {}

    /// \f$ \gamma (Px)_i \f$ for the operand contraction
    /// \f$ \mathrm{dot} = \sum_j k_j x_j \f$ over whatever index set the
    /// caller contracted. A real nonnegative graph has
    /// \f$ S_A = T = 2m \f$ real and takes the real product.
    std::complex<double> coupling(double gamma, std::uint32_t i,
                                  std::complex<double> dot) const {
      if (real_) {
        return gamma * (*strength_)[i].real() * dot.real() * inv2m_;
      }
      return gamma * (*strength_)[i] * dot * invSumA_;
    }

    /// \f$ \gamma |k_i| |\mathrm{dot}| / |S_A| \f$: a bound on the coupling's
    /// magnitude for a Gershgorin radius, which may not rely on cancellation.
    double couplingBound(double gamma, std::uint32_t i, double dotAbs) const {
      return gamma * std::abs((*strength_)[i]) * dotAbs * std::abs(invSumA_);
    }

    bool isReal() const { return real_; }

   private:
    const std::vector<std::complex<double>> *strength_;
    std::complex<double> invSumA_{0.0, 0.0};
    bool real_ = true;
    double inv2m_ = 0.0;
  };

  /// The action of \f$ B_\gamma \f$ restricted to `group`, on a vector indexed
  /// by position within the group:
  ///
  /// \f[
  ///   (B^g x)_i = \sum_{j \in g} A_{ij} x_j - \gamma (Px)_i
  ///     - x_i \left( k^g_i - \gamma (P \mathbf{1}_g)_i \right)
  /// \f]
  ///
  /// the generalized modularity matrix the recursive subdivision step
  /// requires, with \f$ P \f$ the null model of :class:`NullModel`: rank one on
  /// a nonnegative graph, rank two once weights carry sign. The trailing
  /// diagonal term makes \f$ B^g \f$ annihilate the all-ones vector on the
  /// group; it is a row-sum correction and so generalizes across the rank of
  /// \f$ P \f$ unchanged. `groupDegree` is \f$ k^g \f$ and `groupStrength`
  /// carries \f$ S_g \f$ and its two signed channels, all precomputed.
  void applyGroupModularity(const std::vector<std::uint32_t> &group,
                            const std::vector<std::uint32_t> &positionOf,
                            const std::vector<std::complex<double>> &groupDegree,
                            const GroupStrength &groupStrength, double gamma,
                            ModularityPart part, const std::vector<double> &x,
                            std::vector<double> *out) const;

  /// The two most positive eigenvalues of \f$ B_\gamma \f$ restricted to
  /// `group`, over the complement of the all-ones vector, by dense symmetric
  /// eigendecomposition. Used when the group is small enough that the exact
  /// answer is cheap, which is where the gap certificate must be trusted.
  /// Returns false when the group is too small to have two eigenvalues in that
  /// complement.
  bool denseLeadingPair(const std::vector<std::uint32_t> &group,
                        const std::vector<std::uint32_t> &positionOf,
                        const std::vector<std::complex<double>> &groupDegree,
                        const GroupStrength &groupStrength, double gamma,
                        ModularityPart part, double *first, double *second,
                        std::vector<double> *firstVector,
                        double *last, std::vector<double> *lastVector) const;

  /// Most positive eigenpair of \f$ B_\gamma \f$ restricted to `group`, over
  /// the complement of the all-ones vector (which \f$ B^g \f$ always
  /// annihilates), by shifted power iteration from a canonical-rank start
  /// vector; no seed is used. `deflate` is orthogonalized against in addition
  /// to the all-ones vector, which yields the second eigenvalue on a second
  /// call. Returns false when the iteration did not converge within
  /// `maxPowerIterations`.
  bool leadingEigenpair(const std::vector<std::uint32_t> &group,
                        const std::vector<std::uint32_t> &positionOf,
                        const std::vector<std::complex<double>> &groupDegree,
                        const GroupStrength &groupStrength, double gamma,
                        ModularityPart part,
                        const PersistentModularityConfig &cfg,
                        const std::vector<double> *deflate,
                        double *eigenvalue,
                        std::vector<double> *eigenvector) const;

  /// Kernighan-Lin style local refinement of a sign bisection: repeatedly flip
  /// the single unflipped cell whose flip most improves the split's exact
  /// \f$ \Delta Q \f$, then rewind to the best cumulative point. Each cell
  /// moves at most once per pass, so a pass cannot cycle.
  ///
  /// `maximize` says which way "improves" runs and must match the direction the
  /// candidate was selected for. A split proposed by the most negative
  /// eigenvector is the anti-community one; refining it upward would walk it
  /// back toward the community split it was chosen not to be.
  void refineBisection(const std::vector<std::uint32_t> &group,
                       const std::vector<std::uint32_t> &positionOf,
                       const std::vector<std::complex<double>> &groupDegree,
                       const GroupStrength &groupStrength, double gamma,
                       ModularityPart part, bool maximize,
                       std::vector<double> *signs) const;

  /// Canonical hashes and the compact slot map for one partition of the level-0
  /// cells: the shared tail of community canonicalization, used by both
  /// discovery strategies so identity has one implementation.
  void canonicalizeCommunities(
      const LevelGraph &g, const std::vector<std::uint32_t> &comm,
      const std::vector<std::vector<std::uint32_t>> &membersOf,
      std::vector<std::vector<std::uint64_t>> &tokens, std::size_t levelNumber,
      std::vector<std::uint32_t> *slotToCompact,
      std::vector<std::string> *hashes) const;
  ResolutionSlice buildSlice(double gamma, const RunResult &winner,
                             std::vector<RestartRead> restarts) const;

  /// One chained track over consecutive component lists, shared by the
  /// resolution-slice and cobordism-frame trackers: members, their positions,
  /// the covered index window, and the worst adjacent overlap.
  struct Chain {
    std::vector<ComponentId> members;
    std::vector<std::size_t> memberIndices;
    std::size_t first = 0;
    std::size_t last = 0;
    double minAdjacentOverlap = 1.0;
  };
  /// Chain `steps` into tracks by best support overlap at or above
  /// `overlapThreshold`, appending every emitted match to `matchesOut` when
  /// supplied. The axis, resolution or cobordism time, is the caller's; the
  /// chaining rule is the same for both.
  std::vector<Chain> chainTracks(
      const std::vector<const std::vector<ComponentRead> *> &steps,
      double overlapThreshold,
      std::vector<ComponentMatch> *matchesOut) const;
};

}  // namespace tessera

#endif  // TESSERA_OBSERVABLES_PERSISTENTMODULARITY_H
