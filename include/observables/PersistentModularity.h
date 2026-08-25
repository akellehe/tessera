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
/// The hash is derived from the oriented incidence structure of the
/// component's children and its parent lineage (the child hashes feed the
/// parent hash), never from raw vertex/cell numbers.  It is used for
/// persistence matching and deterministic tie-breaking, never as a physical
/// observable.  Two structurally identical (automorphic) components share a
/// hash by construction; bookkeeping that must tell such twins apart (for
/// example cache invalidation) is positional, see
/// :class:`InvalidationRead`.
///
/// ``level`` is the multilevel-aggregation depth at which the component was
/// formed: level 0 is an input cell, level ``k`` a community formed at the
/// ``k``-th aggregation round of the discovery run.
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

/// Which search proposes the communities.  Both score the SAME exact
/// ``Q_gamma`` closed form, so their results are directly comparable on one
/// scale; they differ only in how a partition is searched for.
///
/// An enum rather than a name string: a mis-spelling is a compile error
/// instead of a value that compiles and silently selects a default.
enum class DiscoveryStrategy {
  /// Multilevel aggregation from a fixed restart seed sequence, keeping the
  /// best exact score and reporting the restart spread (the incumbent).
  MultilevelAggregation,
  /// Newman's leading-eigenvector bisection of the modularity matrix
  /// ``B_gamma = A - gamma k k^T / (2m)``, recursed until no group has a
  /// positive leading eigenvalue.  The community COUNT is fixed by the
  /// spectrum rather than by a caller-supplied parameter, and the search
  /// carries no seed.
  LeadingEigenvector,
};

/// Which real functional of the complex `Q` the search maximizes.  Both are
/// always REPORTED; this chooses only what is pursued.
///
/// An enum rather than a name string: a mis-spelling is a compile error
/// rather than a value that compiles and silently selects a default.
enum class ModularityObjective {
  /// Maximize `Q` itself.  Available only where `Q` is real (a real
  /// adjacency), which is the incumbent's regime and where this is the
  /// default so existing behaviour does not move.  Finds community
  /// structure; an anti-community scores below the one-community partition
  /// and is therefore passed over.
  Score,
  /// Maximize `|Q|`.  Always available, and the only ordered choice once
  /// `A` is genuinely complex.  Finds community AND anti-community
  /// structure, with `arg(Q)` saying which was found.  Selected
  /// automatically on a complex graph, where `Score` is not an ordering.
  Magnitude,
};

/// Configuration for the label-free multiscale component discovery.
struct PersistentModularityConfig {
  /// Which search proposes the communities.  Default keeps the incumbent, so
  /// an existing caller sees no change.
  DiscoveryStrategy strategy = DiscoveryStrategy::MultilevelAggregation;
  /// Which real functional of the complex ``Q`` the search maximizes.  The
  /// default keeps the incumbent's behaviour on a real graph; on a COMPLEX
  /// graph ``Score`` is not an ordering and ``Magnitude`` is selected
  /// regardless, with ``ResolutionSlice::objective`` reporting which was used.
  ///
  /// Setting ``Magnitude`` on a real graph is how anti-community structure is
  /// pursued: an anti-community has ``Q < 0``, so maximizing ``Q`` passes it
  /// over in favour of the one-community partition while maximizing ``|Q|``
  /// finds it.  Both readings are always reported.
  ModularityObjective objective = ModularityObjective::Score;
  /// LeadingEigenvector only: a group is indivisible when its leading
  /// eigenvalue does not exceed this.  ``B_gamma`` restricted to a group
  /// always annihilates the all-ones vector, so zero is always in the
  /// spectrum and the leading eigenvalue is never negative; "no positive
  /// eigenvalue" therefore means "at or below this tolerance".
  double leadingEigenvalueTolerance = 1e-9;
  /// LeadingEigenvector only: the minimum leading-to-second eigenvalue gap
  /// for a bisection to be considered well determined.  Below it the split
  /// is REFUSED and reported unresolved with a named reason rather than
  /// taken on an ill-conditioned eigenvector.
  double minEigenvalueGap = 1e-8;
  /// LeadingEigenvector only: groups of at most this many cells get an EXACT
  /// dense symmetric eigendecomposition of ``B_gamma`` restricted to the
  /// group; larger groups fall back to shifted power iteration.
  ///
  /// The dense path exists because power iteration separates the leading
  /// pair at a rate set by their ratio, so a near-degenerate pair — exactly
  /// the case the gap certificate must adjudicate — is where iteration is
  /// slowest and least trustworthy.  Deciding "is this pair degenerate?" by
  /// an iteration that converges only when it is not would be circular.  At
  /// this default the dense solve is well under a millisecond and covers
  /// every group the shipped complexes produce.
  std::size_t denseEigenSolveMaxGroup = 1024;
  /// LeadingEigenvector only: hard cap on power-iteration steps per
  /// eigenpair on groups above ``denseEigenSolveMaxGroup``.  Non-convergence
  /// is reported, never silently accepted.
  int maxPowerIterations = 4096;
  /// LeadingEigenvector only: relative convergence tolerance of the power
  /// iteration's Rayleigh quotient.
  double powerIterationTolerance = 1e-12;
  /// LeadingEigenvector only: run a Kernighan-Lin style local refinement
  /// after each sign bisection.  Without it the method scores measurably
  /// lower ``Q_gamma`` than multilevel aggregation; the flag exists so that
  /// cost can be measured rather than asserted.
  bool kernighanLinRefinement = true;
  /// Resolution parameters gamma for the scan, in scan order.  Adjacent
  /// entries are matched into persistence tracks.
  std::vector<double> resolutions{1.0};
  /// Base of the fixed restart seed sequence: restart ``t`` uses seed
  /// ``splitmix64(baseSeed + t)``.  Deterministic by construction.
  std::uint64_t baseSeed = 0;
  /// Number of deterministic multilevel restarts per resolution.  The best
  /// exact score is retained; the spread across restarts is reported
  /// honestly (no claim of the NP-hard global optimum is ever made).
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
  /// Level-0 member cell ids (a set; listed ascending for reporting only —
  /// the ordering carries no convention and the identity never derives from
  /// these numbers).
  std::vector<std::uint64_t> support;
  /// Sigma_in: total internal adjacency weight A(C,C) counting both ordered
  /// directions (self-loop convention of aggregated levels included).
  /// COMPLEX: its argument is the causal character of this community's
  /// cohesion, and dropping it would be dropping exactly that reading.
  std::complex<double> internalWeight{0.0, 0.0};
  /// S_C: summed degree (strength) of the members; complex like the degrees.
  std::complex<double> strength{0.0, 0.0};
  /// Weighted conductance cut(C) / min(vol C, vol V\C); 0 by convention when
  /// the denominator vanishes (whole-graph or empty community).
  ///
  /// NaN on a signed graph, where it is not defined: conductance is a ratio
  /// of VOLUMES, and a signed community's strength is a difference, so there
  /// is nothing for the cut to be a fraction OF.  Left unmeasured rather
  /// than computed by a formula that does not apply.
  double conductance = 0.0;
  /// This community's exact additive term of Q_gamma,
  /// ``(Sigma_in - gamma S_C^2 / SA) / T``.  These sum over a level's
  /// communities to that level's exact Q_gamma, so they are complex whenever
  /// it is.
  std::complex<double> modularityContribution{0.0, 0.0};
};

/// The named outcomes of one attempted leading-eigenvector bisection.
///
/// Constants rather than literals at each site: a mis-spelling here would
/// not fail to compile, it would silently produce a reason no consumer
/// matches.  Bound to Python so a caller references the constant instead of
/// retyping the string.
struct SplitReason {
  /// The group was bisected and the split raised ``Q_gamma``.
  static constexpr const char *kSplitAccepted = "split-accepted";
  /// The leading eigenvalue is at or below the tolerance: no bisection of
  /// this group raises ``Q_gamma``.  This is Newman's stopping rule and an
  /// ordinary, expected outcome — the group is indivisible, not defective.
  static constexpr const char *kNoPositiveEigenvalue = "no-positive-eigenvalue";
  /// The leading and second eigenvalues are separated by less than the
  /// declared minimum gap, so the leading eigenvector — and therefore the
  /// sign pattern the bisection would use — is not well determined.  The
  /// split is REFUSED rather than taken on an ill-conditioned vector.
  static constexpr const char *kDegenerateLeadingPair =
      "degenerate-leading-pair";
  /// Fewer than two cells: nothing to bisect.
  static constexpr const char *kGroupTooSmall = "group-too-small";
  /// The eigenvector's sign pattern put every cell on one side, so the
  /// proposed bisection is not a bisection.
  static constexpr const char *kEmptySide = "empty-side";
  /// The bisection was well determined but did not raise ``Q_gamma``.
  static constexpr const char *kSplitLowersModularity =
      "split-lowers-modularity";
  /// The power iteration hit ``maxPowerIterations`` without meeting
  /// ``powerIterationTolerance``.  Reported, never silently accepted.
  static constexpr const char *kPowerIterationNotConverged =
      "power-iteration-not-converged";
};

/// One attempted bisection of the leading-eigenvector search: the spectrum
/// that decided it, and what was decided.
///
/// This is the strategy's certificate.  The incumbent's honesty measure is
/// ``ResolutionSlice::restartSpread`` — an empirical proxy for whether the
/// search found anything good.  Here the leading-to-second eigenvalue gap
/// measures directly how well determined the bisection is, in the same way
/// the rest of this layer certifies spectral isolation: separation measured
/// in the spectrum, never to a sort-order neighbour.
///
/// Unmeasured quantities are NaN, never zero.  A group that was never
/// bisected because it was too small has no eigenvalues, and says so.
struct SplitRead {
  /// Number of level-0 cells in the group that was examined.
  std::size_t groupSize = 0;
  /// Most positive eigenvalue of ``B_gamma`` restricted to the group, over
  /// the complement of the all-ones vector.  NaN when not computed.
  double leadingEigenvalue = std::numeric_limits<double>::quiet_NaN();
  /// Second most positive eigenvalue, by deflation.  NaN when not computed.
  double secondEigenvalue = std::numeric_limits<double>::quiet_NaN();
  /// ``leadingEigenvalue - secondEigenvalue``: how well determined the
  /// bisection is.  NaN when either eigenvalue is unmeasured.
  double eigenvalueGap = std::numeric_limits<double>::quiet_NaN();
  /// Exact change in total ``Q_gamma`` this split would produce, from the
  /// class's own closed form.  NaN when no split was evaluated.
  double deltaQ = std::numeric_limits<double>::quiet_NaN();
  /// Whether the group was actually bisected.
  bool accepted = false;
  /// Whether the spectrum determined the outcome.  False means the split was
  /// refused because the answer was not well determined (a degenerate pair,
  /// or a non-converged iteration) — distinct from a determined "do not
  /// split", which is resolved and not accepted.
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
  /// Exact Q_gamma of this restart's final partition (cold recompute),
  /// complex and unreduced.
  std::complex<double> q{0.0, 0.0};
  /// The real scalar this restart was ranked on, derived from ``q`` by the
  /// slice's objective.
  double objectiveValue = 0.0;
  std::size_t communities = 0;
};

/// The discovery result at one resolution gamma.
struct ResolutionSlice {
  double gamma = 1.0;
  /// Which search produced this slice.
  DiscoveryStrategy strategy = DiscoveryStrategy::MultilevelAggregation;
  /// LeadingEigenvector only: one entry per attempted bisection, in the
  /// order attempted — the strategy's spectral certificate.  Empty for
  /// MultilevelAggregation, which performs no bisections.
  std::vector<SplitRead> splits;
  /// Exact Q_gamma of the winning partition, recomputed cold from the final
  /// labels.  The best score across the deterministic restarts — a heuristic
  /// proposal, not the NP-hard global optimum.
  ///
  /// COMPLEX and UNREDUCED: ``abs(q)`` is how much structure the partition
  /// has and ``arg(q)`` is what kind (0 a community, pi an ANTI-community,
  /// ±pi/2 lightlike cohesion, anything else mixed).  Exactly real on a real
  /// graph.  Reported folded rather than split into pieces, because two
  /// differently-typed halves invite treating one as primary.
  std::complex<double> q{0.0, 0.0};
  /// The winning restart's incrementally accumulated score
  /// (Q_0 + sum of accepted exact delta-Q, compensated summation).  Must
  /// agree with ``q`` to double round-off; tested against it.
  std::complex<double> qIncremental{0.0, 0.0};
  /// The real scalar the search actually maximized, derived from ``q``:
  /// ``q.real()`` under ``Score`` and ``abs(q)`` under ``Magnitude``.  This
  /// is what restarts are compared on and what ``restartSpread`` measures.
  double objectiveValue = 0.0;
  /// Which functional that was.  Reported rather than assumed, because a
  /// complex graph selects ``Magnitude`` whatever the config asked for —
  /// ``Score`` is not an ordering there.
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
  /// Every restart's exact score, in seed-sequence order.  Empty under
  /// LeadingEigenvector, which carries no seed and does not restart.
  std::vector<RestartRead> restarts;
  /// max - min of the restart scores: the honestly reported restart
  /// uncertainty of the heuristic search.  NaN under LeadingEigenvector —
  /// that search has no restart spread to report, and unmeasured is never
  /// encoded as zero.
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
  /// Spectral-projector overlap — the documented interface hook.  Populated
  /// only when a projector-overlap hook has been installed via
  /// :func:`PersistentModularity::setProjectorOverlapHook`; a later ticket
  /// supplies the projectors.  Absent (nullopt) means unknown, never zero.
  std::optional<double> projectorOverlap;
};

/// A component track across COBORDISM FRAMES: the same emergent support
/// followed through consecutive frames by maximum support overlap.
///
/// DISTINCT from :class:`PersistenceTrack`, which follows a component
/// across the MODULARITY RESOLUTION SLICES of a single frame.  The two are
/// different quantities and are never interchanged: the whitepaper's fiber
/// acceptance conjunct is "lifetime across multiple cobordism frames", and
/// that lifetime is ``frames()`` here.  A resolution-slice count says how
/// stable a modularity proposal is under the resolution parameter; it says
/// nothing about how long anything lived.
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
  /// Number of consecutive cobordism frames covered — the lifetime the
  /// whitepaper names.
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
  /// Downstream weight-aware gap/localization/persistence status.  Populated
  /// by the later weight-aware certificate tickets; Null means unknown —
  /// unknown is never encoded as zero.  Lifetime/overlap here are proposal
  /// diagnostics only and neither accept nor veto a fiber.
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
/// resolution and cobordism time (ticket #765).
///
/// **Domain and exact identities.**  The input is a finite COMPLEX weighted
/// undirected graph (the complex's one-skeleton under a documented weight
/// map; see :class:`WeightMap`).  `A` is complex SYMMETRIC — `A_ij = A_ji` —
/// because a weight is a property of the EDGE, and an edge's magnitude and
/// argument do not depend on which end you read it from.
///
/// With complex degrees `k_i = sum_j A_ij`, their total `SA = sum_i k_i`, and
/// the real positive scale `T = sum_ij |A_ij|`:
///
///   Q_gamma(P) = (1/T) sum_ij ( A_ij - gamma k_i k_j / SA ) [c_i = c_j],
///
/// via the per-community sufficient statistics
/// Q = sum_c ( Sigma_in(c) - gamma S_c^2 / SA ) / T, with the aggregated
/// self-loop convention A_CC = Sigma_in(C).  Every cached local move gain is
/// the exact closed form
///
///   dQ(v: a -> b) = [ 2 (w_vb - w_va) - 2 gamma k_v (k_v + S_b - S_a)/SA ]/T,
///
/// evaluated in O(deg v) from the cached community totals, so one complete
/// local-move sweep is near O(|E|) (up to revisits).
///
/// Three properties fix that particular arrangement of `SA` and `T`:
///
/// * `k_i` is the row sum of `A` itself, so the null model is the ordinary
///   configuration model — expected weight `k_i k_j / SA` — and NOT a
///   magnitude surrogate.  `sum_ij k_i k_j / SA = SA = sum_ij A_ij`, so the
///   null model carries the same total weight the graph does.
/// * consequently `Q_1(one community) = 0` EXACTLY, for any complex `A`.
///   That anchor is what makes `|Q|` mean "how much structure": zero is the
///   no-structure reading, not an arbitrary offset.  (Measured at 2.4e-16
///   over 500 random complex graphs.)
/// * `T` is real and strictly positive whenever any edge exists, so the outer
///   scale never vanishes.  `SA` still can — see **Degenerate cases**.
///
/// **Reduction.**  On a nonnegative real graph `SA = T = 2m` and the formula
/// is ordinary generalized modularity.  That path evaluates the incumbent's
/// expressions VERBATIM, so such a graph scores BIT-IDENTICALLY rather than
/// merely agreeing to round-off: the two forms differ in floating-point
/// association even where they agree in the reals (`frac*frac` with
/// `frac = s/2m` is not `s*s/(2m*2m)`), and the guarantee worth having is
/// the exact one.  The branch is chosen by the GRAPH, never by a flag.
///
/// These identities are exact in double arithmetic; incremental
/// accumulations use compensated summation and are tested against cold
/// recomputation at the ~1e-15..1e-14 double round-off standard.
///
/// **Q is complex, and both parts are read.**  `|Q|` says how much structure
/// a partition has and is what the search maximizes; `arg(Q)` says what KIND,
/// and is carried rather than discarded.  This is the classification #870
/// settled for causal type, lifted from an edge to a community: `arg(Q) = 0`
/// is spacelike cohesion (a community), `arg(Q) = pi` is timelike cohesion
/// (an ANTI-community), `arg(Q) = ±pi/2` is lightlike, and anything else is
/// mixed.  Collapsing the two into one real number would answer both
/// questions badly, which is exactly the error of reporting a winding without
/// its modulus.
///
/// Anti-community structure is a TARGET, not a failure mode: `Q < 0` means a
/// community bound by dissimilarity, and finding it is as much the point as
/// finding ordinary communities.  Which one the search pursues is
/// :class:`ModularityObjective`; both are always reported.
///
/// **Nothing is refused for being indefinite.**  A complex weight requires no
/// classification, so an edge whose `arg(l^2)` is generic simply contributes
/// what it is.  Under a random initialization essentially every edge is like
/// that, and it is the ordinary case rather than the exceptional one.  The
/// refusal vocabulary is kept for genuine ABSENCES — no edges, or a
/// degenerate edge with `l = 0`, which has no weight to carry rather than an
/// awkward one (see :func:`causalWeightAvailability`).
///
/// **Why complex symmetric and not Hermitian.**  A Hermitian `A` (an
/// antisymmetric phase on some chosen edge direction) yields a REAL `Q`,
/// which is ordered without a magnitude/argument split and is therefore the
/// tempting choice.  It was measured and rejected on two independent grounds:
///
/// * `Q_Hermitian = Re(Q_symmetric)` EXACTLY — the pairing of `(i,j)` with
///   `(j,i) = conj` in every community term makes it so.  The Hermitian
///   operator is thus the symmetric one with its imaginary part discarded,
///   and a component may not be discarded by construction; if it does not
///   matter it must cancel emergently.  On a lightlike-cohesion fixture the
///   symmetric reading gives `Q = 0.3 + 0.2i` where the Hermitian one gives
///   `0.3`, losing precisely the statement that the cohesion is lightlike.
/// * an antisymmetric phase needs a direction convention, and an index-order
///   convention is not label-free.  MEASURED: relabeling the cells and
///   re-deriving the convention moves the Hermitian modularity spectrum by
///   `|dlambda| = 2.0e-1`, and flipping a single edge's stored direction —
///   a representation choice with no physical content — moves it by
///   `3.0e-1`.  The symmetric operator needs no convention and so has
///   nothing to be sensitive to.
///
/// The signed-real reduction (Gomez-Jensen-Arenas, one null model per sign)
/// was likewise measured and dropped: it agrees with the complex operator
/// where every edge is spacelike or timelike, but it scores a
/// lightlike-cohesion community identically to a spacelike one (`+0.5` for
/// both), because taking `Re(A)` deletes exactly the edges whose character it
/// was meant to read.
///
/// **Heuristic status (mandatory reading).**  Global modularity maximization
/// is NP-hard; the discovery is a deterministic multilevel aggregation from
/// a fixed seed sequence that retains the best exact score and reports the
/// restart spread honestly.  Nothing here claims the global optimum, and
/// that is unchanged by which weight map feeds it: seeing the causal
/// structure makes the PROPOSAL better informed, not certified.  Modularity
/// is a heuristic proposal generator only.  Nothing in
/// this class may enter the emergence objective, and a modularity read may
/// never veto an otherwise certified fiber — fiber acceptance rests solely
/// on the independent weight-aware gap/localization/leakage/persistence/
/// anchor certificates (supplied by later tickets; reported here as unknown,
/// never zero).  Communities are proposals and carry no connectivity
/// guarantee.
///
/// **Label-freedom.**  Visit order and tie-breaking derive from a canonical
/// structural ranking (iterated weighted color refinement with
/// individualization by breadth-first distance), and component identity from
/// oriented incidence and lineage — never from raw vertex numbers.  The
/// discovery is a pure function of the labeled graph: edge input order never
/// changes the result.  Within a refinement class whose members are
/// structurally indistinguishable the individualization picks an arbitrary
/// representative (minimum cell id); when such classes are automorphism
/// orbits (all shipped fixtures) the discovered hierarchy under a relabeling
/// is the automorphic image — identical scores, identical per-level
/// canonical-hash multisets, supports mapped up to graph automorphism.
///
/// **Read-only.**  A pure observable: never calls a solver and never mutates
/// the spacetime it reads.  Instances are immutable after construction; the
/// canonical ranking is a lazily computed per-instance cache (the
/// sufficient-statistics caches live inside each discovery run).
class PersistentModularity {
public:
  /// The documented map from edge geometry to similarity weight.  Nothing
  /// else is silently mixed into the metric.
  enum class WeightMap {
    /// w = 1 per edge: the combinatorial one-skeleton, exactly the graph the
    /// legacy Newman-Girvan reads (SparseGraph::modularity,
    /// Spacetime::modularityOnSkeleton) score.  Causally blind by
    /// construction, and honestly so: it reads no geometry at all.
    Unit,
    /// w = exp(-|l|), l the complex edge length: monotone decreasing in the
    /// edge magnitude (the mutual-information convention l = -log I), values
    /// in (0, 1].  CAUSALLY BLIND — it reads only the Euclidean modulus, so a
    /// timelike and a spacelike edge of equal magnitude receive the identical
    /// weight.
    ExpNegAbsLength,
    /// w = exp(-|l|) * exp(i arg(l^2)): the same similarity MAGNITUDE as
    /// `ExpNegAbsLength`, carrying the edge's causal character as its ARGUMENT
    /// rather than collapsing it.  Spacelike edges land on the positive real
    /// axis, timelike on the negative real axis, lightlike on ±i, and a
    /// generic argument stays where it is — no classification happens, so
    /// there is no indefinite case to refuse.  `arg(l^2)` is the measured
    /// quantity `Edge::squaredArgument()` reports.
    CausalPhaseExpNegAbsLength,
  };


  /// Why a causal weight map is or is not available on a complex, with the
  /// disposition census that decides it.  Counts are measured; `reason` is
  /// empty exactly when `available` is true.
  struct CausalWeightRead {
    std::size_t spacelike = 0;
    std::size_t timelike = 0;
    std::size_t lightlike = 0;
    /// Edges with no definite causal character (arg(l^2) generic).  These
    /// are ORDINARY edges for the complex weight map, which carries their
    /// argument as it stands; the count is a diagnostic, not a gate.  Under a
    /// random initialization it is close to every edge.
    std::size_t mixed = 0;
    /// Absent edges (|l|_E below `Edge::kDegenerateEpsilon`); not a causal
    /// type and not scored either way.  These are a genuine absence and DO
    /// make the map unavailable: an edge with no extent has no argument.
    std::size_t degenerate = 0;
    bool available = false;
    /// One of :class:`CausalWeightReason` when unavailable; empty otherwise.
    std::string reason;
  };

  /// Named reasons a causal weight map is unavailable.  Constants rather than
  /// literals: the reason is produced here and compared elsewhere, and a typo
  /// in either place would not fail to compile.
  struct CausalWeightReason {
    /// At least one edge has |l| below `Edge::kDegenerateEpsilon`.  Such an
    /// edge is ABSENT rather than indefinite: it has no argument to carry,
    /// and `arg(0)` is not a reading of anything.
    static constexpr const char *kDegenerateEdgeLength = "degenerate-edge-length";
    /// The complex carries no scorable edge at all.
    static constexpr const char *kNoScorableEdges = "no-scorable-edges";
  };

  /// The disposition census of `st`'s one-skeleton and whether
  /// `CausalPhaseExpNegAbsLength` can be applied to it.  Read-only; a caller
  /// may ask before constructing, and :func:`fromSpacetime` throws with the
  /// same named reason when it cannot.
  ///
  /// A MIXED count does not make the map unavailable — the complex weight
  /// carries a generic argument as readily as a definite one.  Only a genuine
  /// absence does.  The census is worth reading anyway: the mixed FRACTION is
  /// the diagnostic #870 named, and it should FALL if relaxation is imposing
  /// causal character.
  static CausalWeightRead causalWeightAvailability(const Spacetime &st);

  /// Build from an explicit REAL weighted edge list, signed or not.  Cells
  /// are identified by arbitrary 64-bit ids; the node set is the union of the
  /// endpoint ids and ``isolatedCells``.  Parallel edges are consolidated by
  /// weight summation; self-loops and zero-weight edges are ignored at
  /// level 0 — including a pair whose weights CANCEL to zero, which is a
  /// measured absence of net similarity rather than a dropped edge.  Throws
  /// std::invalid_argument on non-finite weights or mismatched lengths.
  ///
  /// A wholly nonnegative list is bit-identical to what it has always
  /// produced.
  static PersistentModularity fromWeightedEdges(
      const std::vector<std::uint64_t> &src,
      const std::vector<std::uint64_t> &tgt,
      const std::vector<double> &weight,
      const std::vector<std::uint64_t> &isolatedCells = {});

  /// The same, for a COMPLEX weighted edge list.  Consolidation, self-loops
  /// and the cancel-to-zero convention are unchanged; both components must be
  /// finite.  A list that happens to be real takes the real path and scores
  /// exactly as the overload above would.
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
  /// True when some edge weight has a nonzero imaginary part, so ``Q`` is
  /// genuinely complex and ``Score`` is not an ordering.  A property of the
  /// GRAPH, not a setting: no caller selects the branch.
  bool isComplex() const noexcept { return complex_; }
  /// True when some edge weight is negative or non-real, i.e. when the graph
  /// leaves the nonnegative regime the incumbent formula was written for.
  bool isSigned() const noexcept { return signed_; }
  /// ``T = sum_ij |A_ij|``, the real positive scale the score divides by.
  /// Equal to ``2m = sum_ij A_ij`` on a nonnegative graph, which is what it
  /// has always returned there.
  double totalWeight2() const noexcept { return twoM_; }
  /// ``SA = sum_ij A_ij``, the complex total the configuration null model
  /// redistributes.  Equal to ``totalWeight2()`` on a nonnegative graph.
  std::complex<double> totalWeightSum() const noexcept { return sumA_; }
  /// The cell ids in internal storage order (input first-appearance order;
  /// carries no convention).
  const std::vector<std::uint64_t> &cellIds() const noexcept {
    return cellIds_;
  }

  /// Exact generalized modularity Q_gamma of a fixed partition
  /// (``labels[i]`` labels cell ``cellIds()[i]``; distinct values are
  /// distinct communities).  The fixed-partition entry point: at gamma = 1
  /// on a Unit-weight graph this is exactly the Newman-Girvan score.
  /// Community terms are combined in canonical-hash-free ascending label
  /// order.  Throws std::invalid_argument when labels.size() != nCells().
  ///
  /// COMPLEX in general: ``|Q|`` is how much structure the partition has and
  /// ``arg(Q)`` is what kind (see the class documentation).  Exactly real on
  /// a real graph, where its imaginary part is zero rather than small.
  std::complex<double> modularityGamma(const std::vector<int> &labels,
                                       double gamma) const;

  /// Deterministic label-free discovery at one resolution under
  /// ``cfg.strategy``.  ``cfg.resolutions`` is ignored here.
  ///
  /// ``MultilevelAggregation`` runs multilevel aggregation over
  /// ``cfg.restarts`` seeds from the fixed sequence, keeping the best exact
  /// score (ties broken by the sorted component hash lists).
  ///
  /// ``LeadingEigenvector`` recursively bisects on the sign pattern of the
  /// leading eigenvector of ``B_gamma`` restricted to each group, stopping
  /// where no group has a positive leading eigenvalue.  The community count
  /// is therefore a reading of the spectrum, not a parameter.  There is no
  /// seed: the search is a pure function of the labeled graph and gamma, and
  /// every attempted bisection is reported in ``ResolutionSlice::splits``
  /// with the eigenvalues that decided it.
  ///
  /// BOTH strategies score the same exact ``Q_gamma`` closed form
  /// (:func:`modularityGamma`), so their slices are comparable directly.
  /// Neither claims the NP-hard global optimum; see the heuristic-status
  /// note on this class, which applies unchanged to both.
  ResolutionSlice discover(double gamma,
                           const PersistentModularityConfig &cfg) const;

  /// The configurable resolution-sequence scan: one slice per entry of
  /// ``cfg.resolutions``, adjacent slices matched by support overlap into
  /// persistence tracks.
  ScanReport scanResolutions(const PersistentModularityConfig &cfg) const;

  /// Match components across resolution or cobordism time by simplex-support
  /// overlap (Jaccard on level-0 cell ids; both sides must reference a
  /// common cell-id universe, e.g. the same evolving complex).  For each
  /// component of ``a`` the best-overlap partner in ``b`` is emitted
  /// (overlap > 0), ties broken by canonical hash then position.  When a
  /// projector-overlap hook is installed its value is reported per match;
  /// matching decisions remain support-based until a later ticket supplies
  /// the projectors.
  std::vector<ComponentMatch> matchComponents(
      const std::vector<ComponentRead> &a,
      const std::vector<ComponentRead> &b) const;

  /// Follow components across COBORDISM FRAMES: ``frames[t]`` is the
  /// component list read from cobordism frame ``t`` over a common cell-id
  /// universe (the same evolving complex).  Consecutive frames are matched
  /// with :func:`matchComponents` and chained into tracks by best overlap
  /// at or above ``overlapThreshold``, exactly the rule
  /// :func:`scanResolutions` chains its resolution slices with — this is
  /// the SAME chaining over a DIFFERENT axis, and it is the supplier of
  /// the whitepaper's "lifetime across multiple cobordism frames".  A
  /// component that appears in only one frame gets a one-frame track: a
  /// lifetime of one is a measured fact about the candidate, never a
  /// structural artifact of reading a single resolution.
  ///
  /// One track is emitted per emergent support, in first-appearance order.
  /// An empty frame list yields no tracks.
  std::vector<FrameTrack> trackAcrossFrames(
      const std::vector<std::vector<ComponentRead>> &frames,
      double overlapThreshold = 0.5) const;

  /// The documented interface hook for spectral-projector overlap.  The
  /// callback receives the two component ids and returns their projector
  /// overlap in [0, 1].  This ticket only plumbs the hook: no projector is
  /// implemented here, and without a hook the field stays absent (unknown).
  using ProjectorOverlapHook =
      std::function<double(const ComponentId &, const ComponentId &)>;
  void setProjectorOverlapHook(ProjectorOverlapHook hook) {
    projectorHook_ = std::move(hook);
  }

  /// Components and tracks whose ancestry a local change touches: every
  /// component (at every hierarchy level of every slice) whose support
  /// intersects ``touchedCells``, plus the tracks containing one.  Siblings
  /// with disjoint support remain valid.  Pure bookkeeping over the report;
  /// no recomputation is triggered.
  static InvalidationRead invalidatedAncestry(
      const ScanReport &report,
      const std::vector<std::uint64_t> &touchedCells);

private:
  PersistentModularity() = default;

  // CSR similarity graph (undirected; each edge stored in both directions).
  std::size_t nNodes_ = 0;
  std::size_t nEdges_ = 0;
  std::vector<std::int64_t> indptr_;
  std::vector<std::uint32_t> indices_;
  std::vector<std::complex<double>> weights_;
  std::vector<std::complex<double>> strength_;  // k_i = sum_j A_ij
  double twoM_ = 0.0;                     // T = sum_ij |A_ij|  (real, > 0)
  std::complex<double> sumA_{0.0, 0.0};   // SA = sum_i k_i
  // Branch selectors, both decided by the GRAPH and never by a caller flag.
  // `signed_` marks departure from the nonnegative regime the incumbent
  // formula was written for; `complex_` marks a nonzero imaginary part, which
  // is what makes Q complex and Score unavailable.
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

  /// The leading-eigenvector search: recursive spectral bisection producing
  /// a single level-0 partition, plus one `SplitRead` per attempted
  /// bisection appended to `splits`.
  RunResult runLeadingEigenvector(double gamma,
                                  const PersistentModularityConfig &cfg,
                                  std::vector<SplitRead> *splits) const;

  /// Group total of the null-model degrees: ``S_g = sum_{i in g} k_i``,
  /// complex like the degrees themselves.
  struct GroupStrength {
    std::complex<double> total{0.0, 0.0};
  };

  /// Which real symmetric part of the complex modularity matrix a spectral
  /// step operates on.
  ///
  /// `B` is complex SYMMETRIC, so it has no real spectrum to take a leading
  /// eigenvalue of.  `Re(B)` and `Im(B)` are each real symmetric, each
  /// annihilates the all-ones vector on a group for the same row-sum reason
  /// `B` does, and each therefore admits the exact dense solver and the
  /// shifted iteration unchanged.  Bisecting on one of them PROPOSES a
  /// split; acceptance is by exact `Q` either way, so the proposal being
  /// heuristic costs nothing but a candidate.  A real graph has
  /// `Im(B) = 0` and only the real part is ever consulted.
  enum class ModularityPart { Real, Imaginary };

  /// The configuration null model `P` of the generalized modularity matrix
  /// `B_gamma = A - gamma P`, with `P_ij = k_i k_j / SA`.
  ///
  /// The degrees are the row sums of `A` itself, so `sum_ij P_ij = SA`: the
  /// null model carries exactly the total weight the graph does, which is
  /// what makes `Q_1(one community)` vanish.  Nothing here is a magnitude
  /// surrogate — a surrogate would break that identity and with it the
  /// meaning of `|Q|`.
  ///
  /// The reciprocal of `SA` is bound once at construction so contraction
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

    /// `gamma * (P x)_i` for the operand contraction `dot = sum_j k_j x_j`
    /// over whatever index set the caller contracted.
    ///
    /// A real nonnegative graph has `SA = T = 2m` real, and this evaluates
    /// the identical product the incumbent did — the bit pattern of such a
    /// graph's score does not move.
    std::complex<double> coupling(double gamma, std::uint32_t i,
                                  std::complex<double> dot) const {
      if (real_) {
        return gamma * (*strength_)[i].real() * dot.real() * inv2m_;
      }
      return gamma * (*strength_)[i] * dot * invSumA_;
    }

    /// `gamma * |k_i| |dot| / |SA|`: a bound on the coupling's magnitude for
    /// a Gershgorin radius, which may not rely on cancellation.
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

  /// The action of `B_gamma` restricted to `group` on a vector indexed by
  /// position within `group`:
  ///   (B^g x)_i = sum_{j in g} A_ij x_j
  ///               - gamma (P x)_i
  ///               - x_i (k^g_i - gamma (P 1_g)_i)
  /// the generalized modularity matrix Newman's subdivision step requires,
  /// with `P` the null model of :class:`NullModel` — rank one on a
  /// nonnegative graph, rank two once weights carry sign.  The trailing
  /// diagonal term is what makes `B^g` annihilate the all-ones vector on the
  /// group; it is a row-sum correction, so it generalizes across the rank of
  /// `P` unchanged.  `groupDegree` is k^g and `groupStrength` carries S_g and
  /// its two signed channels, all precomputed.
  void applyGroupModularity(const std::vector<std::uint32_t> &group,
                            const std::vector<std::uint32_t> &positionOf,
                            const std::vector<std::complex<double>> &groupDegree,
                            const GroupStrength &groupStrength, double gamma,
                            ModularityPart part, const std::vector<double> &x,
                            std::vector<double> *out) const;

  /// The two most positive eigenvalues of `B_gamma` restricted to `group`,
  /// over the complement of the all-ones vector, by EXACT dense symmetric
  /// eigendecomposition.  Used when the group is small enough that the exact
  /// answer is cheap, which is where the gap certificate must be trusted.
  /// Returns false when the group is too small to have two eigenvalues in
  /// that complement.
  bool denseLeadingPair(const std::vector<std::uint32_t> &group,
                        const std::vector<std::uint32_t> &positionOf,
                        const std::vector<std::complex<double>> &groupDegree,
                        const GroupStrength &groupStrength, double gamma,
                        ModularityPart part, double *first, double *second,
                        std::vector<double> *firstVector,
                        double *last, std::vector<double> *lastVector) const;

  /// Most positive eigenpair of `B_gamma` restricted to `group`, over the
  /// complement of the all-ones vector (which `B^g` always annihilates), by
  /// shifted power iteration from a canonical-rank start vector — no seed.
  /// `deflate` is orthogonalized against in addition to the all-ones vector,
  /// which yields the second eigenvalue on a second call.  Returns false
  /// when the iteration did not converge within `maxPowerIterations`.
  bool leadingEigenpair(const std::vector<std::uint32_t> &group,
                        const std::vector<std::uint32_t> &positionOf,
                        const std::vector<std::complex<double>> &groupDegree,
                        const GroupStrength &groupStrength, double gamma,
                        ModularityPart part,
                        const PersistentModularityConfig &cfg,
                        const std::vector<double> *deflate,
                        double *eigenvalue,
                        std::vector<double> *eigenvector) const;

  /// Kernighan-Lin style local refinement of a sign bisection: repeatedly
  /// flip the single unflipped cell whose flip most raises the split's
  /// exact delta-Q, then rewind to the best cumulative point.  Each cell
  /// moves at most once per pass, so a pass cannot cycle.
  void refineBisection(const std::vector<std::uint32_t> &group,
                       const std::vector<std::uint32_t> &positionOf,
                       const std::vector<std::complex<double>> &groupDegree,
                       const GroupStrength &groupStrength, double gamma,
                       ModularityPart part,
                       std::vector<double> *signs) const;

  /// Canonical hashes and the compact slot map for one partition of the
  /// level-0 cells: the shared tail of community canonicalization, used by
  /// both discovery strategies so there is one implementation of identity.
  void canonicalizeCommunities(
      const LevelGraph &g, const std::vector<std::uint32_t> &comm,
      const std::vector<std::vector<std::uint32_t>> &membersOf,
      std::vector<std::vector<std::uint64_t>> &tokens, std::size_t levelNumber,
      std::vector<std::uint32_t> *slotToCompact,
      std::vector<std::string> *hashes) const;
  ResolutionSlice buildSlice(double gamma, const RunResult &winner,
                             std::vector<RestartRead> restarts) const;

  /// One chained track over consecutive component lists (the shared core of
  /// the resolution-slice and cobordism-frame trackers): members, their
  /// positions, the covered index window, and the worst adjacent overlap.
  struct Chain {
    std::vector<ComponentId> members;
    std::vector<std::size_t> memberIndices;
    std::size_t first = 0;
    std::size_t last = 0;
    double minAdjacentOverlap = 1.0;
  };
  /// Chain `steps` into tracks by best support overlap at or above
  /// `overlapThreshold`; every emitted match is appended to `matchesOut`
  /// when supplied.  The axis (resolution or cobordism time) is the
  /// caller's; the chaining rule is identical and lives here once.
  std::vector<Chain> chainTracks(
      const std::vector<const std::vector<ComponentRead> *> &steps,
      double overlapThreshold,
      std::vector<ComponentMatch> *matchesOut) const;
};

}  // namespace tessera

#endif  // TESSERA_OBSERVABLES_PERSISTENTMODULARITY_H
