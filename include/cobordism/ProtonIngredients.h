// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_COBORDISM_PROTON_INGREDIENTS_H
#define TESSERA_COBORDISM_PROTON_INGREDIENTS_H

#include <complex>
#include <cstdint>
#include <memory>
#include <vector>

#include "cobordism/ProtonSynthesis.h"

namespace tessera::spacetime { class Spacetime; }

namespace tessera::cobordism {
using ::tessera::spacetime::Spacetime;

class MultiCobordism;  // returned (seeded, not run) by the node factories below

/// # ProtonIngredients
///
/// The ingredients arm of the proton experiment, whose final state is not
/// pinned. `ProtonSynthesis` is composed here unchanged and supplies the same
/// ingredients through the same two-step drive, except that the final state is
/// never pinned: step B's `outputTargets` is empty, so the objective is
///
///   `F = ‖∇S_Regge‖² + Γ·Σᵢ r_U(inputᵢ)`
///
/// and whatever the whole cobordism comes to carry is read afterwards rather
/// than driven. Exactly one variable differs from `ProtonSynthesis::build()` —
/// the singlet output target — so the two classes form an A/B experiment.
///
///   * Step A — recombination: `ProtonSynthesis::recombinationNode`, delegated
///     to a composed `ProtonSynthesis` configured identically: two neutral q-q̄
///     pairs `{1,-1,0}` ⊔ `{1,0,-1}` → a diquark `{1,ω}` ⊔ antidiquark
///     `{1,ω²}`. It pins those two outputs, so it is a controlled-synthesis
///     node: it runs in `SimulationMode::Synthesis` through
///     `ProtonSynthesis::driveNode`.
///   * Step B — formation with no output pinned: the same ideal diquark
///     `{1,ω}` plus third quark `{ω²}` inputs on the same single-Δ⁴ seed as
///     `ProtonSynthesis::formationNode`, but with an empty output-target list,
///     so no singlet enters the drive. It runs in the node's default mode,
///     `SimulationMode::Emergence` with the strict sub-mode, through the same
///     schedule (`ProtonSynthesis::NodeDrive::run`).
///
/// The seed stays uniform and all-spacelike (`ℓ² = +1`) by design: at
/// initialization no time has passed, and causal structure — which marks
/// sequences of events in the causal-set sense — may only emerge from the
/// optimization, never be initialized in.
///
/// Convergence carries no answer-shaped gate: no color tolerance, no minimum
/// hole count. An attempt converges iff it is
///
///   * stationary — step B's `runStage2` stopped on its relative-tolerance
///     stationarity test rather than on its iteration budget, and
///   * persistent — one further evolution pass (`runStage1`, ∂W frozen) plus
///     relaxation leaves the answer-agnostic summary stable: the emergent hole
///     count and `b_k` unchanged, and `F` within a relative tolerance.
///
/// Everything physical is a post-hoc observable: `emergentHoles()`, the final
/// objective, the inputs-only residual, and `singletResidual()` — the singlet
/// `r_state` of `ProtonSynthesis::singlet()` against the whole, reported as a
/// diagnostic so the unpinned result is comparable to the synthesis's carried
/// level.
class ProtonIngredients {
 public:
  /// Configure an ingredients-arm build. The knobs and their defaults are
  /// `ProtonSynthesis`'s, so the two arms differ only in what is pinned; see
  /// `ProtonSynthesis::ProtonSynthesis` for their meaning.
  explicit ProtonIngredients(std::uint64_t seed = 0, int registerDegree = 3,
                             double gamma = 50.0, double inputWeight = 20.0,
                             int precone = 0, bool shouldUseDirectedSurgery = false);

  /// Build the ingredients arm: run step A then step B with
  /// `ProtonSynthesis::build()`'s drive (initialization pass with
  /// `grow_boundaries=true`, evolution pass
  /// with ∂W frozen, optional directed cone probes, then `runStage2`),
  /// restarting across seeds until an attempt is stationary and persistent.
  /// When `maxRestarts` is exhausted the lowest-final-`F` attempt is kept and
  /// `converged()` is false. The persistence pass reuses
  /// `evolveSteps`/`stage2MaxIters`; `persistRelTol` is the relative
  /// `F`-stability tolerance. No color tolerance, no minimum hole count.
  /// Idempotent.
  void build(int maxRestarts = 16, int initSteps = 180, int evolveSteps = 60,
             int stage1CandidateMoves = 8,
             double stage2Beta = 1.0, int stage2MaxIters = 10,
             double persistRelTol = 0.05);

  /// Step A verbatim: `ProtonSynthesis::recombinationNode` on the composed
  /// `ProtonSynthesis` — the same seeded, not-yet-run 2→2 node, in
  /// `SimulationMode::Synthesis`, that `ProtonSynthesis::build()` drives.
  [[nodiscard]] std::shared_ptr<MultiCobordism> recombinationNode(std::uint64_t seed) const;
  /// Step B with nothing pinned: the same single-Δ⁴ seed and ideal diquark
  /// `{1,ω}` plus third quark `{ω²}` inputs as `ProtonSynthesis::formationNode`,
  /// but `outputTargets = {}`, so the final state is not driven and is read off
  /// the whole afterwards.
  [[nodiscard]] std::shared_ptr<MultiCobordism> formationNode(std::uint64_t seed) const;
  /// The joint inputs-only node: one `MultiCobordism` whose inputs are the
  /// three Z₃-symmetric neutral q-q̄ pairs `{1,−1,0} ⊔ {0,1,−1} ⊔ {−1,0,1}`
  /// (each summing to 0 — the only prepared content, held representable through
  /// their `r_U` terms for the whole build) and whose `outputTargets` is empty.
  /// No diquark, bare quark or intermediate is imposed anywhere: the objective
  /// is `‖∇S‖² + Γ·Σᵢ w·r_U(inputᵢ)`, and whatever the whole cobordism comes to
  /// carry — a baryon with a conjugate partner is the pre-registered
  /// expectation — is read afterwards, the singlet and conjugate-singlet
  /// residuals serving as diagnostics that never drive. Inputs are seeded at
  /// v0/v1/v2 of the single Δ⁴ seed. Not run; the caller drives it.
  [[nodiscard]] std::shared_ptr<MultiCobordism> jointNode(std::uint64_t seed) const;

  /// True iff the kept attempt was stationary and persistent. Never a statement
  /// about the singlet or the hole count. Triggers `build()`.
  [[nodiscard]] bool converged();
  /// Whether the kept attempt's final `runStage2` stopped on stationarity.
  /// Triggers `build()`.
  [[nodiscard]] bool stationary();
  /// Whether the kept attempt survived the continued evolution and relaxation
  /// pass with holes, `b_k` and `F` stable. Triggers `build()`.
  [[nodiscard]] bool persistent();
  /// The base seed of the kept attempt. Triggers `build()`.
  [[nodiscard]] std::uint64_t seed();
  /// The full relaxed step-B complex. Triggers `build()`.
  [[nodiscard]] std::shared_ptr<Spacetime> spacetime();
  /// The object read off is the whole step-B cobordism; provided for API parity
  /// with `ProtonSynthesis::block()`. Triggers `build()`.
  [[nodiscard]] std::shared_ptr<Spacetime> block();
  /// The emergent `(k+2)`-vertex holes on the whole — a topological observable,
  /// not a gate; any count, including zero. Triggers `build()`.
  [[nodiscard]] std::vector<std::vector<std::uint64_t>> emergentHoles();
  /// Diagnostic only: the relabeling-invariant singlet `r_state` of
  /// `ProtonSynthesis::singlet()` against the whole's `L_k` harmonic, reported
  /// so the unpinned result is comparable to the synthesis's carried level
  /// (`≈0` there). It never steers or gates this build. Triggers `build()`.
  [[nodiscard]] double singletResidual();
  /// Step B's inputs-only realizability residual `r_U` — the whole matter term
  /// of the ingredients arm's objective. Triggers `build()`.
  [[nodiscard]] double inputResidual();
  /// The kept attempt's final objective `F`. Triggers `build()`.
  [[nodiscard]] double finalObjective();
  /// Step A's `r_U`, reported exactly as `ProtonSynthesis` reports it. Triggers
  /// `build()`.
  [[nodiscard]] double diquarkResidual();

 private:
  /// Lazily run `build()` with default parameters on first accessor use.
  void ensureBuilt();
  /// The same minimal seed as `ProtonSynthesis`: a single Δ⁴ simplex with the
  /// uniform all-spacelike metric (`ℓ² = +1`; see the class note on why no
  /// causal structure is initialized). Mirrors
  /// `ProtonSynthesis::buildMinimalSeed`, which is private there.
  [[nodiscard]] static std::shared_ptr<Spacetime> buildMinimalSeed();

  // ---- configuration ----
  /// The synthesis arm, composed unchanged: supplies step A's node verbatim and
  /// the configuration contract (seed, degree, Γ, input weight, precone,
  /// directed surgery).
  ProtonSynthesis proton_;
  std::uint64_t baseSeed_;
  int registerDegree_;
  double gamma_;
  double inputResidualWeight_;
  int precone_;
  bool shouldUseDirectedSurgery_;

  // ---- build state (populated by build()) ----
  bool attempted_ = false;
  bool converged_ = false;
  bool stationary_ = false;
  bool persistent_ = false;
  std::uint64_t keptSeed_ = 0;
  std::shared_ptr<Spacetime> spacetime_;
  std::vector<std::vector<std::uint64_t>> emergentHoles_;
  double singletResidual_ = 0.0;
  double inputResidual_ = 0.0;
  double finalObjective_ = 0.0;
  double diquarkResidual_ = 0.0;
};

}  // namespace tessera::cobordism

#endif  // TESSERA_COBORDISM_PROTON_INGREDIENTS_H
