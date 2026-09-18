// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.
//
// Training loop, benchmark and checkpointing for the libtorch reinforcement learning. Trains
// the proximal-policy-optimization (PPO) policy over `CobordismObjectiveEnv` and compares it
// against random and grow-only baselines on the proton carry criterion. Nothing here touches
// proton construction; it only orchestrates the environment and the agent.

#ifndef TESSERA_RL_TRAINER_H
#define TESSERA_RL_TRAINER_H

#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "rl/CobordismObjectiveEnv.h"
#include "rl/PpoAgent.h"

namespace tessera::rl {

/// Training hyperparameters. The defaults are the proton-carry profile.
struct TrainConfig {
  int iterations = 8;
  int episodesPerIter = 3;
  int evalSeeds = 6;
  int hidden = 64;
  double lr = 7e-4;
  int updateEpochs = 8;
  double entropyCoef = 0.03;
  double entropyCoefFinal = 0.005;  // negative disables annealing
  std::uint64_t agentSeed = 0;
  bool evalDeterministic = true;
};

/// Evaluation statistics aggregated over held-out seeds; the carry rate is the proton
/// criterion.
struct EvalSummary {
  double carryRate = 0.0;
  double meanHoles = 0.0;
  double meanRstate = 0.0;
  double meanFinalF = 0.0;
  double meanReward = 0.0;
};

/// Diagnostics for one training iteration.
struct IterStat {
  int iteration = 0;
  double meanReturn = 0.0;
  double policyLoss = 0.0;
  double valueLoss = 0.0;
  double entropy = 0.0;
  double entropyCoef = 0.0;
};

/// The full benchmark: the learned policy against the random and grow-only baselines.
struct BenchmarkResult {
  std::vector<IterStat> history;
  EvalSummary rl;
  EvalSummary randomBaseline;
  EvalSummary growOnly;
  double trainTimeS = 0.0;
};

/// A macro-level policy: observation -> (move, parameters).
using PolicyFn = std::function<std::pair<Move, std::array<float, kParamDim>>(
    const std::vector<float> &)>;

/// The environment and training configurations for the proton-carry profile.
[[nodiscard]] EnvConfig carryProfileEnv();
[[nodiscard]] TrainConfig carryProfileTrain();

/// Evaluation statistics over `seeds` under `policy`; each seed starts from a fresh node.
[[nodiscard]] EvalSummary evaluate(CobordismObjectiveEnv &env, const PolicyFn &policy,
                                   const std::vector<std::uint64_t> &seeds);

/// Train PPO on the chosen target (formation or recombination) and evaluate it against the
/// random and grow-only baselines. If `checkpointPath` is non-empty, the trained policy is
/// written there with `torch::save`.
[[nodiscard]] BenchmarkResult benchmark(EnvConfig envConfig, TrainConfig trainConfig,
                                        bool formation, const std::string &checkpointPath = "");

/// Load a policy checkpoint into a fresh `HybridActorCritic`, for evaluation.
[[nodiscard]] HybridActorCritic loadPolicy(const std::string &checkpointPath, int obsDim,
                                           int nMoves, int paramDim, int hidden = 64);

}  // namespace tessera::rl

#endif  // TESSERA_RL_TRAINER_H
