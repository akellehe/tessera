// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.
//
// A Gym-style reinforcement-learning environment over the `MultiCobordism` objective search.
//
// This is a harness, not a builder: MultiCobordism and Proton are the source of truth for
// proton construction, and this environment only drives them. Every macro-action is one
// `MultiCobordism::buildStep`, plus the `directedConeOut`/`directedConeIn` probe when
// `directedGrow` is set; the observation and reward only read published engine quantities
// (r_U, reggeActionGradient, emergentHoles, r_state, Betti numbers). The fixed
// `Proton.build()` schedule (init -> evolve -> relax) becomes the learning problem: the
// agent chooses which macro-action to take, and with what parameters.

#ifndef TESSERA_RL_COBORDISM_OBJECTIVE_ENV_H
#define TESSERA_RL_COBORDISM_OBJECTIVE_ENV_H

#include <array>
#include <complex>
#include <cstdint>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

namespace tessera::cobordism { class MultiCobordism; }

namespace tessera::rl {

/// The discrete macro-moves, drawn from the policy's categorical head. Grow and Evolve are
/// stage-1 surgery passes (boundary-growing and frozen-boundary respectively); Relax is a
/// stage-2 relaxation.
enum class Move { Grow = 0, Evolve = 1, Relax = 2 };
inline constexpr int kNumMoves = 3;
/// Continuous parameters per action, from the Gaussian head: [intensity, knob], each
/// clipped to [0, 1].
inline constexpr int kParamDim = 2;
inline constexpr int kBettiSlots = 5;
/// Observation layout: 4 signed-log scalars, 5 Betti slots, the hole count, 3 size counts,
/// the budget fraction, and a `kNumMoves`-wide one-hot of the previous move.
inline constexpr int kObsDim = 4 + kBettiSlots + 1 + 3 + 1 + kNumMoves;  // 17

/// `seed -> node` factory, for either the recombination or the formation setup.
using NodeFactory =
    std::function<std::shared_ptr<cobordism::MultiCobordism>(std::uint64_t)>;

/// Environment knobs: the engine setup, the per-move parameter ranges that the continuous
/// action interpolates within, the reward weights, and the termination rule.
struct EnvConfig {
  int registerDegree = 3;
  double gamma = 50.0;
  int maxActions = 8;
  std::pair<int, int> growSteps = {2, 8};
  std::pair<int, int> evolveSteps = {2, 8};
  std::pair<int, int> relaxIters = {1, 4};
  std::pair<double, double> betaRange = {0.25, 2.0};
  std::pair<double, double> alphaRange = {0.02, 0.2};
  int nCandidateMoves = 6;
  double carryTol = 0.5;
  int targetHoles = 3;
  double carryBonus = 3.0;
  double rewardScale = 1.0;
  double holeRewardWeight = 0.0;
  double rstateRewardWeight = 0.0;
  bool terminateOnCarry = true;
  bool directedGrow = false;
};

/// One environment transition: the `(obs, reward, done, info)` tuple, flattened.
struct StepResult {
  std::vector<float> obs;  // kObsDim
  double reward = 0.0;
  bool done = false;
  // --- info ---
  int move = -1;
  double F = 0.0;
  double deltaF = 0.0;
  double rU = 0.0;
  double rstate = 0.0;
  int holes = 0;
  bool carried = false;
  bool terminated = false;
  bool truncated = false;
  bool engineError = false;
};

/// A Gym-style reinforcement-learning environment over one `MultiCobordism` node's objective
/// search: a 17-dimensional observation, a hybrid action (one move plus two continuous
/// parameters), and a reward made of the signed-log ΔF drop, optional hole and r_state
/// shaping, a carry bonus and an error penalty. Deterministic in the reset seed.
class CobordismObjectiveEnv {
 public:
  /// An empty `target` selects recombination, where success is \f$ r_U \to 0 \f$. A
  /// non-empty `target` is a whole-cobordism target color state (for example the proton
  /// singlet); success is that state being carried over at least `targetHoles` holes.
  CobordismObjectiveEnv(NodeFactory nodeFactory,
                        std::vector<std::complex<double>> target, EnvConfig config);

  /// Seed a fresh node on a single Δ⁴ simplex and return the initial observation.
  std::vector<float> reset(std::uint64_t seed);
  /// Apply one macro-action (a move plus parameters clipped to [0, 1]) and return the
  /// resulting transition.
  StepResult step(Move move, std::array<float, kParamDim> params);

  [[nodiscard]] int obsDim() const { return kObsDim; }
  [[nodiscard]] int numMoves() const { return kNumMoves; }
  [[nodiscard]] int paramDim() const { return kParamDim; }
  [[nodiscard]] double currentF() const { return currentF_; }
  [[nodiscard]] const std::shared_ptr<cobordism::MultiCobordism> &node() const {
    return node_;
  }

 private:
  struct Metrics {
    double F = 0.0, gradN2 = 0.0, rU = 0.0, rstate = 0.0;
    int holes = 0;
    std::vector<int> betti;
    int nVertices = 0, nEdges = 0, nTopCells = 0;
  };
  [[nodiscard]] Metrics computeMetrics() const;
  [[nodiscard]] bool isCarried(const Metrics &m) const;
  [[nodiscard]] std::vector<float> observation(const Metrics &m) const;

  NodeFactory nodeFactory_;
  std::vector<std::complex<double>> target_;  // empty = no whole-cobordism target
  EnvConfig config_;

  std::shared_ptr<cobordism::MultiCobordism> node_;
  std::uint64_t seed_ = 0;
  int stepsTaken_ = 0;
  double currentF_ = 0.0;
  int lastMove_ = -1;
  bool carried_ = false;
  Metrics lastMetrics_;
};

/// `seed -> node` factories for the two setups: formation (2→1, the proton singlet carried by
/// the whole cobordism) and recombination (2→2, a colored diquark ⊔ antidiquark, with no
/// whole-cobordism target). Both build the node through `Proton`, so the agent drives the
/// same setup `Proton.build()` does.
[[nodiscard]] NodeFactory formationNodeFactory(int registerDegree = 3, double gamma = 50.0,
                                               double inputWeight = 20.0);
[[nodiscard]] NodeFactory recombinationNodeFactory(int registerDegree = 3, double gamma = 50.0,
                                                   double inputWeight = 20.0);

/// Environment builders for the two setups. Formation carries the proton singlet on the whole
/// cobordism; recombination has no whole-cobordism target, and succeeds as
/// \f$ r_U \to 0 \f$. Both wire up the `Proton`-backed factory and target internally.
[[nodiscard]] CobordismObjectiveEnv makeFormationEnv(EnvConfig config, double inputWeight = 20.0);
[[nodiscard]] CobordismObjectiveEnv makeRecombinationEnv(EnvConfig config,
                                                         double inputWeight = 20.0);

}  // namespace tessera::rl

#endif  // TESSERA_RL_COBORDISM_OBJECTIVE_ENV_H
