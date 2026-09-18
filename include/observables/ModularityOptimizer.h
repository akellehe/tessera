// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_OBSERVABLES_MODULARITYOPTIMIZER_H
#define TESSERA_OBSERVABLES_MODULARITYOPTIMIZER_H

#include <cstdint>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "observables/PersistentModularity.h"

// === tessera subsystem ns fwd-decls ===
namespace tessera::graph {}
namespace tessera::mesh {}
namespace tessera::quantum {}
namespace tessera::simulations {}
namespace tessera::spacetime {}
// === cross-subsystem fwd-decls ===
namespace tessera::simulations {
  class CDT;
}
namespace tessera::spacetime {
  class PachnerMove;
  class Spacetime;
}
namespace tessera::observables {
using namespace ::tessera::mesh;
using namespace ::tessera::graph;
using namespace ::tessera::spacetime;
using namespace ::tessera::simulations;
using namespace ::tessera::quantum;


/// One recorded measurement on the (modularity, spectral dimension)
/// trajectory.
struct ModularityMeasurement {
  double Q;            ///< Newman-Girvan modularity at the time of measurement.
  double dsSmall;      ///< Spectral dimension at small diffusion times.
  double dsLarge;      ///< Spectral dimension at large diffusion times.
  std::size_t nVertices;
  std::size_t nEdges;
  std::size_t nSimplices;
  int iter;            ///< Sweep iteration index.
  std::string direction;  ///< "up" | "down"
};

/// Configuration for :class:`ModularityOptimizer`.
struct ModularityOptimizerConfig {
  double targetDq = 0.05;          ///< Q increment between measurements.
  int maxIterations = 400;         ///< Hard cap per sweep direction.
  int nDiffusionWalks = 80;        ///< Diffusion start nodes per measurement.
  double maxSigma = 200.0;         ///< Upper bound of log-spaced t grid.
  int negativeRetryMax = 10;       ///< Max retries when D_S comes back negative.
  double epsilonQMax = 0.01;       ///< Up-sweep early-exit tolerance.
  int krylovDim = 30;              ///< Krylov subspace dim for heat kernel.
  int targetNModules = 4;          ///< Number of modules in the fixed partition.
};

/// Modularity sweep on a causal dynamical triangulation (CDT) spacetime, driven
/// by transactional Pachner moves accepted on the sign of the modularity change.
///
/// References: Newman, "Modularity and community structure in networks",
/// arXiv:physics/0602124; Ambjorn, Jurkiewicz, Loll, "Spectral Dimension of the
/// Universe", arXiv:hep-th/0505113.
///
/// # Score domain
///
/// Every modularity number this class produces — the sweep's \f$ Q \f$
/// trajectory (via ``Spacetime::modularityOnSkeleton``) and the label-free
/// discovery in :func:`discoverComponents` — is a Newman-Girvan or
/// generalized-modularity score on a combinatorial, nonnegative one-skeleton.
/// It is blind to signed and complex Hodge weights, so it is a heuristic
/// proposal generator: it may propose candidate component supports, it never
/// enters the emergence objective, and it may not veto an otherwise certified
/// fiber. Fiber acceptance rests on the independent weight-aware gap,
/// localization, leakage, persistence and anchor certificates.
///
/// Algorithm, per iteration:
///   1. Pick a random move type from {add, remove, flip, iflip, shift}.
///   2. ``cdt.proposeXxx()`` selects a target read-only. If no target is
///      eligible, try another move type (one fallback per iteration).
///   3. Snapshot \f$ Q \f$ on the spacetime 1-skeleton.
///   4. ``move.apply()`` commits the move.
///   5. Recompute \f$ Q \f$. Keep the move if it changed in the requested
///      direction, otherwise ``move.rollback()``.
///   6. If \f$ Q \f$ crossed the next ``targetDq`` threshold, build the dual
///      graph, measure the spectral dimension, and record a measurement.
///
/// The sweep's fixed-partition read (community = vertex id modulo
/// ``targetNModules``) remains available; label-free discovery is
/// :class:`PersistentModularity`.
///
/// The move-type hook in ``selectMoveType`` can bias the distribution toward
/// move types most likely to shift \f$ Q \f$ in the target direction; the
/// default is uniform over all five.
class ModularityOptimizer {
public:
  /// Progress callback signature: (iter, maxIter, currentQ, n_meas).
  using ProgressCallback =
      std::function<void(int, int, double, std::size_t)>;

  ModularityOptimizer(ModularityOptimizerConfig cfg, std::uint64_t seed)
      : cfg_(cfg), rng_(seed) {}

  /// Drive the spacetime via ``cdt`` Pachner moves to walk \f$ Q \f$ in the
  /// given ``direction`` ("up" or "down"). Records (modularity, spectral
  /// dimension) at every ``targetDq`` threshold crossing. Mutates ``cdt``'s
  /// spacetime in place and resets the per-sweep counters before running.
  std::vector<ModularityMeasurement> sweep(
      CDT &cdt,
      const std::string &direction,
      ProgressCallback progress = nullptr);

  /// Label-free discovery of persistent modular components on the current
  /// spacetime one-skeleton: builds the nonnegative similarity graph under
  /// ``map`` and delegates to :func:`PersistentModularity::scanResolutions`.
  /// Read-only: it never mutates the spacetime and never proposes or applies
  /// moves. The result is a heuristic proposal and must not feed the emergence
  /// objective. Deterministic for a fixed ``cfg`` seed sequence; the optimizer's
  /// own random number generator is untouched.
  ScanReport discoverComponents(
      const Spacetime &st, const PersistentModularityConfig &cfg,
      PersistentModularity::WeightMap map =
          PersistentModularity::WeightMap::ExpNegAbsLength) const;

  // Per-sweep counters, reset at the top of each ``sweep()`` call.
  /// Moves applied and kept, i.e. \f$ Q \f$ moved in the desired direction.
  std::int64_t getNAccepted() const noexcept { return nAccepted_; }
  /// Moves applied then rolled back, i.e. \f$ Q \f$ moved the wrong way.
  std::int64_t getNRolledBack() const noexcept { return nRolledBack_; }
  /// Iterations with no eligible Pachner-move proposal.
  std::int64_t getNNoMove() const noexcept { return nNoMove_; }
  /// Spectral-dimension measurements taken; equals the sweep result length.
  std::int64_t getNMeasurements() const noexcept { return nMeasurements_; }

private:
  ModularityOptimizerConfig cfg_;
  std::mt19937 rng_;
  std::int64_t nAccepted_ = 0;
  std::int64_t nRolledBack_ = 0;
  std::int64_t nNoMove_ = 0;
  std::int64_t nMeasurements_ = 0;

  // Try each move type in random order until one validates via propose().
  // Returns nullptr if all five fail.
  std::unique_ptr<PachnerMove> proposeAny(CDT &cdt);

  // Measure the spectral dimension on the dual graph and recompute modularity
  // on the 1-skeleton, retrying when the spectral dimension comes back
  // negative.
  ModularityMeasurement measure(
      CDT &cdt, int iter, const std::string &direction);
};

}  // namespace tessera

#endif  // TESSERA_OBSERVABLES_MODULARITYOPTIMIZER_H
