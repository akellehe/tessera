// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.
//
// A libtorch actor-critic trained by proximal policy optimization (PPO) for the cobordism
// objective-search policy. Engine-agnostic: it sees only observation vectors and the
// `(obs, reward, done, info)` contract of `CobordismObjectiveEnv`. libtorch has no
// `torch::distributions`, so the categorical and Gaussian log-probability, entropy and
// sampling are computed explicitly here.
//
// Reference: Schulman, Wolski, Dhariwal, Radford & Klimov, arXiv:1707.06347.

#ifndef TESSERA_RL_PPO_AGENT_H
#define TESSERA_RL_PPO_AGENT_H

#include <array>
#include <cstdint>
#include <random>
#include <tuple>
#include <vector>

#include <torch/torch.h>

#include "rl/CobordismObjectiveEnv.h"

namespace tessera::rl {

/// Seed torch (network initialization and action sampling). The environment's engine RNG is
/// seeded separately, by `reset(seed)`.
void setSeed(std::uint64_t seed);

/// One sampled action: the categorical move and the Gaussian parameters, with their joint
/// log-probability and the value-baseline estimate. Produced on the rollout path, under
/// `NoGradGuard`.
struct ActOutput {
  int move = 0;
  std::array<float, kParamDim> params{};
  double logp = 0.0;
  double value = 0.0;
};

/// Shared-trunk actor-critic: a tanh multilayer-perceptron trunk feeding a categorical move
/// head, a diagonal-Gaussian parameter head (state-independent \f$ \log\sigma \f$,
/// initialized to -0.5) and a value head.
struct HybridActorCriticImpl : torch::nn::Module {
  HybridActorCriticImpl(int obsDim, int nMoves, int paramDim, int hidden = 64);

  /// (moveLogits, paramMean, paramStd, value) for a batch of observations.
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> forward(
      torch::Tensor obs);
  /// Sample one action for a single [1, obsDim] observation; `deterministic` takes the mode.
  ActOutput act(torch::Tensor obs, bool deterministic = false);
  /// The value baseline for a single [1, obsDim] observation.
  double valueOf(torch::Tensor obs);
  /// Joint log-probability, joint entropy and value for a batch; used by the PPO update.
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> evaluateActions(
      torch::Tensor obs, torch::Tensor moves, torch::Tensor params);

  int nMoves_ = 0;
  int paramDim_ = 0;
  torch::nn::Linear trunk1{nullptr}, trunk2{nullptr}, moveHead{nullptr},
      paramMeanHead{nullptr}, valueHead{nullptr};
  torch::Tensor paramLogStd;
};
TORCH_MODULE(HybridActorCritic);

/// One rollout transition. The generalized-advantage-estimation (GAE) advantage and the
/// return are filled in once the episode ends.
struct Transition {
  std::vector<float> obs;
  int move = 0;
  std::array<float, kParamDim> params{};
  double logp = 0.0;
  double value = 0.0;
  double reward = 0.0;
  double advantage = 0.0;
  double ret = 0.0;
};

/// Diagnostics from the last minibatch of a PPO update.
struct UpdateStats {
  double policyLoss = 0.0;
  double valueLoss = 0.0;
  double entropy = 0.0;
};

/// Proximal policy optimization for the hybrid policy: GAE advantages, a clipped surrogate
/// objective, value and entropy-bonus terms, and minibatch stochastic gradient descent.
class PPO {
 public:
  PPO(int obsDim, int nMoves, int paramDim, int hidden = 64, double lr = 3e-4,
      double gamma = 0.99, double lam = 0.95, double clip = 0.2, double valueCoef = 0.5,
      double entropyCoef = 0.01, int updateEpochs = 6, int minibatchSize = 64,
      double maxGradNorm = 0.5);

  /// Run one episode under the current policy and fill in the GAE advantages and returns.
  /// Returns the transitions; `finalInfo` receives the last `StepResult`, which supplies the
  /// benchmark metrics and the terminal bootstrap value.
  std::vector<Transition> collectEpisode(CobordismObjectiveEnv &env, std::uint64_t seed,
                                         StepResult &finalInfo);
  /// One PPO update over a batch of transitions: `updateEpochs` passes of minibatch SGD.
  UpdateStats update(std::vector<Transition> &transitions);
  /// The greedy (or, if not deterministic, sampled) action, for evaluation.
  ActOutput selectAction(const std::vector<float> &obs, bool deterministic = true);

  HybridActorCritic policy;
  double entropyCoef;  // public: the training loop anneals it over a run

 private:
  void finishGae(std::vector<Transition> &transitions, double lastValue) const;

  double gamma_, lam_, clip_, valueCoef_, maxGradNorm_;
  int updateEpochs_, minibatchSize_;
  torch::optim::Adam optimizer_;
  std::mt19937 shuffleRng_;
};

/// The greedy or sampled action for a bare (possibly checkpoint-loaded) policy given a raw
/// observation vector. Needs no `PPO` instance, so it serves the evaluation path.
[[nodiscard]] ActOutput selectPolicyAction(HybridActorCritic policy,
                                           const std::vector<float> &obs,
                                           bool deterministic = true);

}  // namespace tessera::rl

#endif  // TESSERA_RL_PPO_AGENT_H
