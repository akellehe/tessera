// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "rl/Trainer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <utility>

#include <torch/torch.h>

namespace tessera::rl {

EnvConfig carryProfileEnv() {
  // The proton-carry profile: one Grow move gets a large stage-1 budget (the register
  // carries in roughly 50-120 engine steps), hole and r_state shaping are dense, the terminal
  // carry bonus is large, the action horizon is short, and the directed cone probes are on.
  EnvConfig c;
  c.maxActions = 3;
  c.growSteps = {50, 130};
  c.evolveSteps = {10, 40};
  c.relaxIters = {3, 8};
  c.nCandidateMoves = 8;
  c.holeRewardWeight = 2.0;
  c.rstateRewardWeight = 1.0;
  c.carryBonus = 10.0;
  c.terminateOnCarry = false;
  c.directedGrow = true;
  return c;
}

TrainConfig carryProfileTrain() {
  return TrainConfig{};  // the defaults are already the proton-carry profile
}

namespace {
struct EpisodeStat {
  double finalF = 0.0, reward = 0.0, rstate = 0.0;
  int holes = 0;
  bool carried = false;
};

EpisodeStat runEpisode(CobordismObjectiveEnv &env, const PolicyFn &policy,
                       std::uint64_t seed) {
  auto obs = env.reset(seed);
  double totalReward = 0.0;
  bool done = false;
  StepResult info;
  while (!done) {
    auto [move, params] = policy(obs);
    info = env.step(move, params);
    totalReward += info.reward;
    obs = info.obs;
    done = info.done;
  }
  return {info.F, totalReward, info.rstate, info.holes, info.carried};
}

std::vector<IterStat> trainLoop(PPO &ppo, CobordismObjectiveEnv &env, int iterations,
                                int episodesPerIter,
                                const std::vector<std::uint64_t> &trainSeeds,
                                double entropyCoefFinal) {
  std::vector<IterStat> history;
  const double entropyStart = ppo.entropyCoef;
  std::size_t seedIdx = 0;
  for (int it = 0; it < iterations; ++it) {
    // Linearly anneal the entropy bonus: explore the move mix early, commit late.
    if (entropyCoefFinal >= 0.0 && iterations > 1) {
      const double frac = static_cast<double>(it) / (iterations - 1);
      ppo.entropyCoef = entropyStart + frac * (entropyCoefFinal - entropyStart);
    }
    std::vector<Transition> batch;
    std::vector<double> returns;
    for (int e = 0; e < episodesPerIter; ++e) {
      StepResult info;
      auto trans = ppo.collectEpisode(
          env, trainSeeds[seedIdx % trainSeeds.size()], info);
      ++seedIdx;
      double epReturn = 0.0;
      for (const auto &t : trans) epReturn += t.reward;
      returns.push_back(epReturn);
      for (auto &t : trans) batch.push_back(std::move(t));
    }
    const auto stats = ppo.update(batch);
    IterStat rec;
    rec.iteration = it;
    rec.meanReturn = returns.empty()
                         ? 0.0
                         : std::accumulate(returns.begin(), returns.end(), 0.0) /
                               static_cast<double>(returns.size());
    rec.policyLoss = stats.policyLoss;
    rec.valueLoss = stats.valueLoss;
    rec.entropy = stats.entropy;
    rec.entropyCoef = ppo.entropyCoef;
    history.push_back(rec);
  }
  return history;
}
}  // namespace

EvalSummary evaluate(CobordismObjectiveEnv &env, const PolicyFn &policy,
                     const std::vector<std::uint64_t> &seeds) {
  EvalSummary s;
  int carried = 0, rstateCount = 0;
  double holes = 0.0, finalF = 0.0, reward = 0.0, rstateSum = 0.0;
  for (auto seed : seeds) {
    const auto e = runEpisode(env, policy, seed);
    carried += e.carried ? 1 : 0;
    holes += e.holes;
    finalF += e.finalF;
    reward += e.reward;
    if (std::isfinite(e.rstate)) {
      rstateSum += e.rstate;
      ++rstateCount;
    }
  }
  const double n = seeds.empty() ? 1.0 : static_cast<double>(seeds.size());
  s.carryRate = static_cast<double>(carried) / n;
  s.meanHoles = holes / n;
  s.meanFinalF = finalF / n;
  s.meanReward = reward / n;
  s.meanRstate = rstateCount ? rstateSum / rstateCount
                             : std::numeric_limits<double>::quiet_NaN();
  return s;
}

BenchmarkResult benchmark(EnvConfig envConfig, TrainConfig trainConfig, bool formation,
                          const std::string &checkpointPath) {
  setSeed(trainConfig.agentSeed);
  auto env = formation ? makeFormationEnv(envConfig) : makeRecombinationEnv(envConfig);
  PPO ppo(env.obsDim(), env.numMoves(), env.paramDim(), trainConfig.hidden, trainConfig.lr,
          /*gamma=*/0.99, /*lam=*/0.95, /*clip=*/0.2, /*valueCoef=*/0.5,
          trainConfig.entropyCoef, trainConfig.updateEpochs, /*minibatch=*/64,
          /*maxGradNorm=*/0.5);

  // Train and evaluate on disjoint seed sets, so the benchmark measures generalization.
  std::vector<std::uint64_t> trainSeeds;
  const int nTrainSeeds = std::max(8, trainConfig.episodesPerIter * 2);
  for (int i = 0; i < nTrainSeeds; ++i) trainSeeds.push_back(100 + i);
  std::vector<std::uint64_t> heldOut;
  for (int i = 0; i < trainConfig.evalSeeds; ++i) heldOut.push_back(i);

  const auto t0 = std::chrono::steady_clock::now();
  auto history = trainLoop(ppo, env, trainConfig.iterations, trainConfig.episodesPerIter,
                           trainSeeds, trainConfig.entropyCoefFinal);
  const auto t1 = std::chrono::steady_clock::now();

  const bool det = trainConfig.evalDeterministic;
  PolicyFn rlPolicy = [&ppo, det](const std::vector<float> &obs) {
    const auto a = ppo.selectAction(obs, det);
    return std::make_pair(static_cast<Move>(a.move), a.params);
  };
  const int nMoves = env.numMoves();
  PolicyFn randomPolicy = [nMoves](const std::vector<float> &) {
    const int move = static_cast<int>(torch::randint(0, nMoves, {1}).item<int64_t>());
    const auto p = torch::rand({kParamDim});
    std::array<float, kParamDim> params{};
    for (int i = 0; i < kParamDim; ++i) params[i] = p[i].item<float>();
    return std::make_pair(static_cast<Move>(move), params);
  };
  PolicyFn growOnly = [](const std::vector<float> &) {
    return std::make_pair(Move::Grow, std::array<float, kParamDim>{1.0f, 0.5f});
  };

  BenchmarkResult result;
  result.history = std::move(history);
  result.rl = evaluate(env, rlPolicy, heldOut);
  setSeed(trainConfig.agentSeed + 1);  // independent randomness for the baselines
  result.randomBaseline = evaluate(env, randomPolicy, heldOut);
  result.growOnly = evaluate(env, growOnly, heldOut);
  result.trainTimeS = std::chrono::duration<double>(t1 - t0).count();

  if (!checkpointPath.empty()) torch::save(ppo.policy, checkpointPath);
  return result;
}

HybridActorCritic loadPolicy(const std::string &checkpointPath, int obsDim, int nMoves,
                             int paramDim, int hidden) {
  HybridActorCritic policy(obsDim, nMoves, paramDim, hidden);
  torch::load(policy, checkpointPath);
  return policy;
}

}  // namespace tessera::rl
