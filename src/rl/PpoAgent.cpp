// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "rl/PpoAgent.h"

#include <algorithm>
#include <numeric>
#include <vector>

namespace tessera::rl {

namespace {
constexpr double kLog2Pi = 1.8378770664093453;  // log(2*pi)

torch::Tensor obsToTensor(const std::vector<float> &obs) {
  return torch::from_blob(const_cast<float *>(obs.data()),
                          {static_cast<long>(obs.size())}, torch::kFloat)
      .clone();
}

// --- Categorical(logits) --------------------------------------------------------------
torch::Tensor categoricalLogProb(const torch::Tensor &logits, const torch::Tensor &actions) {
  auto logp = torch::log_softmax(logits, -1);
  return logp.gather(-1, actions.to(torch::kLong).unsqueeze(-1)).squeeze(-1);
}
torch::Tensor categoricalEntropy(const torch::Tensor &logits) {
  auto logp = torch::log_softmax(logits, -1);
  auto p = torch::softmax(logits, -1);
  return -(p * logp).sum(-1);
}
torch::Tensor categoricalSample(const torch::Tensor &logits) {
  return torch::multinomial(torch::softmax(logits, -1), 1).squeeze(-1);
}

// --- Normal(mean, std) (diagonal) -----------------------------------------------------
torch::Tensor normalLogProb(const torch::Tensor &x, const torch::Tensor &mean,
                            const torch::Tensor &std) {
  auto z = (x - mean) / std;
  return -0.5 * z.pow(2) - torch::log(std) - 0.5 * kLog2Pi;
}
torch::Tensor normalEntropy(const torch::Tensor &std) {
  return torch::log(std) + (0.5 + 0.5 * kLog2Pi);  // per-dim
}
torch::Tensor normalSample(const torch::Tensor &mean, const torch::Tensor &std) {
  return mean + std * torch::randn_like(mean);
}
}  // namespace

void setSeed(std::uint64_t seed) { torch::manual_seed(seed); }

// ---- HybridActorCritic ----------------------------------------------------------------
HybridActorCriticImpl::HybridActorCriticImpl(int obsDim, int nMoves, int paramDim, int hidden)
    : nMoves_(nMoves), paramDim_(paramDim) {
  trunk1 = register_module("trunk1", torch::nn::Linear(obsDim, hidden));
  trunk2 = register_module("trunk2", torch::nn::Linear(hidden, hidden));
  moveHead = register_module("move", torch::nn::Linear(hidden, nMoves));
  paramMeanHead = register_module("param_mean", torch::nn::Linear(hidden, paramDim));
  valueHead = register_module("value", torch::nn::Linear(hidden, 1));
  // State-independent log-σ for the continuous parameters (the usual PPO parameterization).
  paramLogStd = register_parameter("param_log_std", torch::full({paramDim}, -0.5, torch::kFloat));
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
HybridActorCriticImpl::forward(torch::Tensor obs) {
  auto h = torch::tanh(trunk2->forward(torch::tanh(trunk1->forward(obs))));
  auto moveLogits = moveHead->forward(h);
  auto paramMean = paramMeanHead->forward(h);
  auto paramStd = torch::exp(paramLogStd).expand_as(paramMean);
  auto value = valueHead->forward(h).squeeze(-1);
  return {moveLogits, paramMean, paramStd, value};
}

ActOutput HybridActorCriticImpl::act(torch::Tensor obs, bool deterministic) {
  torch::NoGradGuard noGrad;
  auto [moveLogits, paramMean, paramStd, value] = forward(obs);
  torch::Tensor move = deterministic ? torch::argmax(moveLogits, -1)
                                     : categoricalSample(moveLogits);
  torch::Tensor params = deterministic ? paramMean : normalSample(paramMean, paramStd);
  auto logp = categoricalLogProb(moveLogits, move) +
              normalLogProb(params, paramMean, paramStd).sum(-1);
  ActOutput out;
  out.move = static_cast<int>(move.reshape({-1})[0].item<int64_t>());
  auto flat = params.reshape({-1});
  for (int i = 0; i < paramDim_ && i < kParamDim; ++i)
    out.params[i] = flat[i].item<float>();
  out.logp = logp.reshape({-1})[0].item<double>();
  out.value = value.reshape({-1})[0].item<double>();
  return out;
}

double HybridActorCriticImpl::valueOf(torch::Tensor obs) {
  torch::NoGradGuard noGrad;
  auto value = std::get<3>(forward(obs));
  return value.reshape({-1})[0].item<double>();
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
HybridActorCriticImpl::evaluateActions(torch::Tensor obs, torch::Tensor moves,
                                       torch::Tensor params) {
  auto [moveLogits, paramMean, paramStd, value] = forward(obs);
  auto logp = categoricalLogProb(moveLogits, moves) +
              normalLogProb(params, paramMean, paramStd).sum(-1);
  auto entropy = categoricalEntropy(moveLogits) + normalEntropy(paramStd).sum(-1);
  return {logp, entropy, value};
}

// ---- PPO ------------------------------------------------------------------------------
PPO::PPO(int obsDim, int nMoves, int paramDim, int hidden, double lr, double gamma,
         double lam, double clip, double valueCoef, double entropyCoefIn, int updateEpochs,
         int minibatchSize, double maxGradNorm)
    : policy(obsDim, nMoves, paramDim, hidden),
      entropyCoef(entropyCoefIn),
      gamma_(gamma),
      lam_(lam),
      clip_(clip),
      valueCoef_(valueCoef),
      maxGradNorm_(maxGradNorm),
      updateEpochs_(updateEpochs),
      minibatchSize_(minibatchSize),
      optimizer_(policy->parameters(), torch::optim::AdamOptions(lr)),
      shuffleRng_(0) {}

std::vector<Transition> PPO::collectEpisode(CobordismObjectiveEnv &env, std::uint64_t seed,
                                            StepResult &finalInfo) {
  auto obs = env.reset(seed);
  std::vector<Transition> transitions;
  bool done = false;
  while (!done) {
    auto action = policy->act(obsToTensor(obs).unsqueeze(0));
    auto res = env.step(static_cast<Move>(action.move), action.params);
    Transition t;
    t.obs = obs;
    t.move = action.move;
    t.params = action.params;
    t.logp = action.logp;
    t.value = action.value;
    t.reward = res.reward;
    transitions.push_back(std::move(t));
    obs = res.obs;
    done = res.done;
    finalInfo = res;
  }
  // Bootstrap value: zero on a true termination (the target was carried), otherwise
  // V(final), which removes the bias a time-limit truncation would otherwise introduce.
  const double lastValue =
      finalInfo.terminated ? 0.0 : policy->valueOf(obsToTensor(obs).unsqueeze(0));
  finishGae(transitions, lastValue);
  return transitions;
}

void PPO::finishGae(std::vector<Transition> &transitions, double lastValue) const {
  double gae = 0.0;
  double nextValue = lastValue;
  for (auto it = transitions.rbegin(); it != transitions.rend(); ++it) {
    const double delta = it->reward + gamma_ * nextValue - it->value;
    gae = delta + gamma_ * lam_ * gae;
    it->advantage = gae;
    it->ret = gae + it->value;
    nextValue = it->value;
  }
}

UpdateStats PPO::update(std::vector<Transition> &transitions) {
  const int n = static_cast<int>(transitions.size());
  std::vector<float> obsFlat(static_cast<std::size_t>(n) * kObsDim);
  std::vector<float> paramsFlat(static_cast<std::size_t>(n) * kParamDim);
  std::vector<float> oldLogpV(n), returnsV(n), advV(n);
  std::vector<int64_t> movesV(n);
  for (int i = 0; i < n; ++i) {
    const auto &t = transitions[i];
    for (int j = 0; j < kObsDim; ++j) obsFlat[i * kObsDim + j] = t.obs[j];
    for (int j = 0; j < kParamDim; ++j) paramsFlat[i * kParamDim + j] = t.params[j];
    movesV[i] = t.move;
    oldLogpV[i] = static_cast<float>(t.logp);
    returnsV[i] = static_cast<float>(t.ret);
    advV[i] = static_cast<float>(t.advantage);
  }
  auto obs = torch::from_blob(obsFlat.data(), {n, kObsDim}, torch::kFloat).clone();
  auto params = torch::from_blob(paramsFlat.data(), {n, kParamDim}, torch::kFloat).clone();
  auto moves = torch::tensor(movesV);  // kLong
  auto oldLogp = torch::from_blob(oldLogpV.data(), {n}, torch::kFloat).clone();
  auto returns = torch::from_blob(returnsV.data(), {n}, torch::kFloat).clone();
  auto advantages = torch::from_blob(advV.data(), {n}, torch::kFloat).clone();
  advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8);

  std::vector<int> order(n);
  std::iota(order.begin(), order.end(), 0);
  UpdateStats stats;
  for (int epoch = 0; epoch < updateEpochs_; ++epoch) {
    std::shuffle(order.begin(), order.end(), shuffleRng_);
    for (int start = 0; start < n; start += minibatchSize_) {
      const int end = std::min(start + minibatchSize_, n);
      std::vector<int64_t> mbv(order.begin() + start, order.begin() + end);
      auto mb = torch::tensor(mbv);  // kLong minibatch indices
      auto logpEntropyValue = policy->evaluateActions(
          obs.index_select(0, mb), moves.index_select(0, mb), params.index_select(0, mb));
      auto logp = std::get<0>(logpEntropyValue);
      auto entropy = std::get<1>(logpEntropyValue);
      auto value = std::get<2>(logpEntropyValue);
      auto adv = advantages.index_select(0, mb);
      auto ratio = torch::exp(logp - oldLogp.index_select(0, mb));
      auto unclipped = ratio * adv;
      auto clipped = torch::clamp(ratio, 1.0 - clip_, 1.0 + clip_) * adv;
      auto policyLoss = -torch::min(unclipped, clipped).mean();
      auto valueLoss = (returns.index_select(0, mb) - value).pow(2).mean();
      auto entropyLoss = -entropy.mean();
      auto loss = policyLoss + valueCoef_ * valueLoss + entropyCoef * entropyLoss;
      optimizer_.zero_grad();
      loss.backward();
      torch::nn::utils::clip_grad_norm_(policy->parameters(), maxGradNorm_);
      optimizer_.step();
      stats.policyLoss = policyLoss.item<double>();
      stats.valueLoss = valueLoss.item<double>();
      stats.entropy = entropy.mean().item<double>();
    }
  }
  return stats;
}

ActOutput PPO::selectAction(const std::vector<float> &obs, bool deterministic) {
  return policy->act(obsToTensor(obs).unsqueeze(0), deterministic);
}

ActOutput selectPolicyAction(HybridActorCritic policy, const std::vector<float> &obs,
                             bool deterministic) {
  return policy->act(obsToTensor(obs).unsqueeze(0), deterministic);
}

}  // namespace tessera::rl
