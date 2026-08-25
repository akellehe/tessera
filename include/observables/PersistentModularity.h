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

/// Configuration for the label-free multiscale component discovery.
struct PersistentModularityConfig {
  /// Which search proposes the communities.  Default keeps the incumbent, so
  /// an existing caller sees no change.
  DiscoveryStrategy strategy = DiscoveryStrategy::MultilevelAggregation;
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
  double internalWeight = 0.0;
  /// S_C: summed weighted degree (strength) of the members.
  double strength = 0.0;
  /// Weighted conductance cut(C) / min(vol C, vol V\C); 0 by convention when
  /// the denominator vanishes (whole-graph or empty community).
  double conductance = 0.0;
  /// This community's exact additive term of Q_gamma:
  /// Sigma_in/(2m) - gamma (S_C/(2m))^2.
  double modularityContribution = 0.0;
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
  /// Exact Q_gamma of this restart's final partition (cold recompute).
  double q = 0.0;
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
  double q = 0.0;
  /// The winning restart's incrementally accumulated score
  /// (Q_0 + sum of accepted exact delta-Q, compensated summation).  Must
  /// agree with ``q`` to double round-off; tested against it.
  double qIncremental = 0.0;
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
/// **Domain and exact identities.**  The input is a finite REAL weighted
/// undirected graph (the complex's one-skeleton under a documented weight
/// map; see :class:`WeightMap`).  Weights may be signed.  Two scoring
/// branches are selected by the graph itself, never by a flag:
///
/// *Unsigned branch* — every weight nonnegative.  Generalized modularity
///
///   Q_gamma(P) = (1/2m) sum_ij (A_ij - gamma k_i k_j / (2m)) [c_i = c_j],
///
/// via the per-community sufficient statistics
/// Q = sum_c [ Sigma_in(c)/(2m) - gamma (S_c/(2m))^2 ], with the aggregated
/// self-loop convention A_CC = Sigma_in(C).  Every cached local move gain is
/// the exact closed form
///
///   dQ(v: a -> b) = (w_vb - w_va)/m - gamma k_v (k_v + S_b - S_a) / (2 m^2),
///
/// evaluated in O(deg v) from the cached community totals, so one complete
/// local-move sweep is near O(|E|) (up to revisits).
///
/// *Signed branch* — at least one negative weight.  The configuration null
/// model is undefined on a signed graph: `2m` can vanish or go negative, and
/// the model is derived from a degree-preserving rewiring of a NONNEGATIVE
/// graph.  This class therefore uses the standard signed generalization of
/// Gomez, Jensen and Arenas (2009), which carries a separate null model per
/// sign.  With `A = A+ - A-`, `2m± = sum_ij A±_ij`, `T = 2m+ + 2m-`:
///
///   Q_gamma(P) = (1/T) sum_ij [ A_ij
///                  - gamma (k_i+ k_j+ / 2m+ - k_i- k_j- / 2m-) ] [c_i = c_j],
///
/// with the vanishing-null-model convention that a sign contributing no
/// weight contributes no null term (`2m± = 0` makes its whole term zero
/// rather than 0/0).  Its move gain is the same closed form with the null
/// contribution split across the two channels:
///
///   dQ(v: a -> b) = 2 (w_vb - w_va) / T
///                 - (2 gamma / T) [ k_v+ (k_v+ + S_b+ - S_a+) / 2m+
///                                 - k_v- (k_v- + S_b- - S_a-) / 2m- ].
///
/// **The unsigned branch is not the signed one specialized.**  It runs the
/// identical arithmetic it always has, so a nonnegative graph scores
/// BIT-IDENTICALLY to every previous release — the two differ in floating
/// point association even where they agree algebraically, and the guarantee
/// worth having is the exact one.  The signed formula does reduce to the
/// unsigned formula algebraically when `A- = 0`, which is what makes this a
/// generalization rather than a second method.
///
/// These identities are exact in double arithmetic; incremental
/// accumulations use compensated summation and are tested against cold
/// recomputation at the ~1e-15..1e-14 double round-off standard.
///
/// **Complex weights are refused, not reduced.**  A complex `A` makes `Q`
/// complex, and a complex `Q` has no ordering, so "maximize Q" is not a
/// statement.  Every real-valued reduction of it either discards a component
/// — which the geometry's imaginary part forbids, being physics rather than
/// residue — or folds the two into a magnitude, and `|Q|` rewards
/// anti-community structure exactly as much as community structure because
/// `Q < 0` means "worse than the null model".  Neither is a modularity.  So
/// the causal weight map is available only where every edge has a DEFINITE
/// disposition, and refuses BY NAME otherwise
/// (:func:`causalWeightAvailability`).  Note what this buys: in the definite
/// regime `Im(l^2)` is zero to within the declared angular tolerance, so
/// taking the causal SIGN there discards nothing measurable.  The refusal
/// falls exactly where discarding would begin.
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
    /// w = s * exp(-|l|), s the edge's causal sign: +1 spacelike, -1 timelike,
    /// 0 lightlike.  The same similarity magnitude as `ExpNegAbsLength`,
    /// signed by the causal character `Edge::disposition()` reads from
    /// arg(l^2).  Available only where every edge has a DEFINITE disposition;
    /// see :func:`causalWeightAvailability`.
    CausalExpNegAbsLength,
  };

  /// Why a causal weight map is or is not available on a complex, with the
  /// disposition census that decides it.  Counts are measured; `reason` is
  /// empty exactly when `available` is true.
  struct CausalWeightRead {
    std::size_t spacelike = 0;
    std::size_t timelike = 0;
    std::size_t lightlike = 0;
    /// Edges with no definite causal character (arg(l^2) generic).  Any one
    /// of these makes the causal map unavailable.
    std::size_t mixed = 0;
    /// Absent edges (|l|_E below `Edge::kDegenerateEpsilon`); not a causal
    /// type and not scored either way.
    std::size_t degenerate = 0;
    bool available = false;
    /// One of :class:`CausalWeightReason` when unavailable; empty otherwise.
    std::string reason;
  };

  /// Named reasons a causal weight map is unavailable.  Constants rather than
  /// literals: the reason is produced here and compared elsewhere, and a typo
  /// in either place would not fail to compile.
  struct CausalWeightReason {
    /// At least one edge has a generic arg(l^2).  Assigning it a causal sign
    /// would invent a definiteness the geometry does not have, which is
    /// exactly what the disposition classification refuses to do.
    static constexpr const char *kMixedCausalCharacter =
        "mixed-causal-character";
    /// The complex carries no scorable edge at all.
    static constexpr const char *kNoScorableEdges = "no-scorable-edges";
  };

  /// The disposition census of `st`'s one-skeleton and whether
  /// `CausalExpNegAbsLength` can be applied to it.  Read-only; a caller may
  /// ask before constructing, and :func:`fromSpacetime` throws with the same
  /// named reason when it cannot.
  static CausalWeightRead causalWeightAvailability(const Spacetime &st);

  /// Build from an explicit REAL weighted edge list, signed or not.  Cells
  /// are identified by arbitrary 64-bit ids; the node set is the union of the
  /// endpoint ids and ``isolatedCells``.  Parallel edges are consolidated by
  /// weight summation; self-loops and zero-weight edges are ignored at
  /// level 0 — including a pair whose weights CANCEL to zero, which is a
  /// measured absence of net similarity rather than a dropped edge.  Throws
  /// std::invalid_argument on non-finite weights or mismatched lengths.
  ///
  /// Negative weights select the signed scoring described on this class; a
  /// wholly nonnegative list takes the unsigned path and is bit-identical to
  /// what it has always produced.
  static PersistentModularity fromWeightedEdges(
      const std::vector<std::uint64_t> &src,
      const std::vector<std::uint64_t> &tgt,
      const std::vector<double> &weight,
      const std::vector<std::uint64_t> &isolatedCells = {});

  /// Build the similarity graph from the spacetime's one-skeleton (vertices
  /// and edges) under the given weight map.  Read-only on the spacetime.
  static PersistentModularity fromSpacetime(
      const Spacetime &st, WeightMap map = WeightMap::ExpNegAbsLength);

  std::size_t nCells() const noexcept { return nNodes_; }
  std::size_t nEdges() const noexcept { return nEdges_; }
  /// True when some edge weight is negative, so scoring uses the signed
  /// (rank-two) null model.  A property of the graph, not a setting: no
  /// caller selects the branch.
  bool isSigned() const noexcept { return signed_; }
  /// Total adjacency weight: 2m = sum_ij A_ij on a nonnegative graph, and
  /// T = 2m+ + 2m- (the total ABSOLUTE weight) when ``isSigned()``, that
  /// being the normalizer the signed score divides by.
  double totalWeight2() const noexcept { return twoM_; }
  /// 2m+, the positive channel's total weight (0 unless ``isSigned()``).
  double totalWeight2Positive() const noexcept { return twoMPos_; }
  /// 2m-, the negative channel's total magnitude, a NONNEGATIVE number
  /// (0 unless ``isSigned()``).
  double totalWeight2Negative() const noexcept { return twoMNeg_; }
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
  double modularityGamma(const std::vector<int> &labels, double gamma) const;

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
  std::vector<double> weights_;
  std::vector<double> strength_;          // k_i (signed sum)
  double twoM_ = 0.0;                     // 2m (signed sum; T in the signed
                                          // branch, where it is 2m+ + 2m-)
  // Per-sign channels.  `signed_` is the branch selector and is decided by
  // the graph, not by a caller flag: it is true exactly when some edge weight
  // is negative.  The strength vectors are always sized so every index is
  // valid; the unsigned branch leaves them zero and never reads them.
  bool signed_ = false;
  std::vector<double> strengthPos_;       // k_i+
  std::vector<double> strengthNeg_;       // k_i-  (nonnegative magnitudes)
  double twoMPos_ = 0.0;                  // 2m+
  double twoMNeg_ = 0.0;                  // 2m-
  std::vector<std::uint64_t> cellIds_;    // internal index -> cell id
  // Consolidated edges with their stored (input) orientation: the oriented
  // incidence used for level-1 identity hashing and exact cold recomputes.
  std::vector<std::uint32_t> orientedSrc_;
  std::vector<std::uint32_t> orientedTgt_;
  std::vector<double> orientedW_;

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

  /// Group totals of the null-model strength channels, gathered once per
  /// group of the leading-eigenvector recursion: `total` is S_g = sum of k_i
  /// over the group, `pos` and `neg` the same sums over k_i+ and k_i-.
  struct GroupStrength {
    double total = 0.0;
    double pos = 0.0;
    double neg = 0.0;
  };

  /// The configuration null model `P` of the generalized modularity matrix
  /// `B_gamma = A - gamma P`.
  ///
  /// On a nonnegative graph `P_ij = k_i k_j / 2m` — rank one, the degree
  /// -preserving rewiring model.  Once weights carry sign that model is not
  /// defined (`2m` may vanish or turn negative), and `P` becomes the
  /// Gomez-Jensen-Arenas rank-two model
  /// `P_ij = k_i+ k_j+ / 2m+ - k_i- k_j- / 2m-`, whose channels normalize
  /// separately.  A channel carrying no weight has reciprocal zero, so its
  /// whole term vanishes rather than dividing by zero — which is exactly
  /// what makes the rank-two model collapse onto the rank-one one when the
  /// graph happens to be nonnegative.
  ///
  /// The reciprocals are bound once at construction so contraction stays
  /// division-free inside the power-iteration inner loop.
  class NullModel {
   public:
    explicit NullModel(const PersistentModularity &owner)
        : strength_(&owner.strength_),
          strengthPos_(&owner.strengthPos_),
          strengthNeg_(&owner.strengthNeg_),
          inv2m_(owner.twoM_ > 0.0 ? 1.0 / owner.twoM_ : 0.0),
          invPos_(owner.twoMPos_ > 0.0 ? 1.0 / owner.twoMPos_ : 0.0),
          invNeg_(owner.twoMNeg_ > 0.0 ? 1.0 / owner.twoMNeg_ : 0.0),
          signed_(owner.signed_) {}

    /// `gamma * (P x)_i` for the operand contractions `dot = sum_j k_j x_j`,
    /// `dotPos = sum_j k_j+ x_j`, `dotNeg = sum_j k_j- x_j` over whatever
    /// index set the caller contracted.  The unsigned branch reads only
    /// `dot`, the signed branch only the other two, and the unsigned branch
    /// evaluates the identical product it did before the model gained a
    /// second channel — the bit pattern of a nonnegative graph's score does
    /// not move.
    double coupling(double gamma, std::uint32_t i, double dot, double dotPos,
                    double dotNeg) const {
      if (!signed_) return gamma * (*strength_)[i] * dot * inv2m_;
      return gamma * ((*strengthPos_)[i] * dotPos * invPos_ -
                      (*strengthNeg_)[i] * dotNeg * invNeg_);
    }

    /// The bound `gamma * |P|_i . |x|` used for a Gershgorin radius: the
    /// two signed channels are added rather than differenced, since a bound
    /// on the row's absolute sum may not rely on cancellation between them.
    double couplingBound(double gamma, std::uint32_t i, double dot,
                         double dotPos, double dotNeg) const {
      if (!signed_) return gamma * (*strength_)[i] * dot * inv2m_;
      return gamma * ((*strengthPos_)[i] * dotPos * invPos_ +
                      (*strengthNeg_)[i] * dotNeg * invNeg_);
    }

    bool isSigned() const { return signed_; }

   private:
    const std::vector<double> *strength_;
    const std::vector<double> *strengthPos_;
    const std::vector<double> *strengthNeg_;
    double inv2m_ = 0.0;
    double invPos_ = 0.0;
    double invNeg_ = 0.0;
    bool signed_ = false;
  };

  /// The leading-eigenvector search: recursive spectral bisection producing
  /// a single level-0 partition, plus one `SplitRead` per attempted
  /// bisection appended to `splits`.
  RunResult runLeadingEigenvector(double gamma,
                                  const PersistentModularityConfig &cfg,
                                  std::vector<SplitRead> *splits) const;

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
                            const std::vector<double> &groupDegree,
                            const GroupStrength &groupStrength, double gamma,
                            const std::vector<double> &x,
                            std::vector<double> *out) const;

  /// The two most positive eigenvalues of `B_gamma` restricted to `group`,
  /// over the complement of the all-ones vector, by EXACT dense symmetric
  /// eigendecomposition.  Used when the group is small enough that the exact
  /// answer is cheap, which is where the gap certificate must be trusted.
  /// Returns false when the group is too small to have two eigenvalues in
  /// that complement.
  bool denseLeadingPair(const std::vector<std::uint32_t> &group,
                        const std::vector<std::uint32_t> &positionOf,
                        const std::vector<double> &groupDegree,
                        const GroupStrength &groupStrength, double gamma,
                        double *first,
                        double *second,
                        std::vector<double> *firstVector) const;

  /// Most positive eigenpair of `B_gamma` restricted to `group`, over the
  /// complement of the all-ones vector (which `B^g` always annihilates), by
  /// shifted power iteration from a canonical-rank start vector — no seed.
  /// `deflate` is orthogonalized against in addition to the all-ones vector,
  /// which yields the second eigenvalue on a second call.  Returns false
  /// when the iteration did not converge within `maxPowerIterations`.
  bool leadingEigenpair(const std::vector<std::uint32_t> &group,
                        const std::vector<std::uint32_t> &positionOf,
                        const std::vector<double> &groupDegree,
                        const GroupStrength &groupStrength, double gamma,
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
                       const std::vector<double> &groupDegree,
                       const GroupStrength &groupStrength, double gamma,
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
