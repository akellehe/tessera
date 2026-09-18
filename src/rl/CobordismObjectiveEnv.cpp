// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "rl/CobordismObjectiveEnv.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

#include "cobordism/MultiCobordism.h"
#include "cobordism/Proton.h"
#include "mesh/EdgeList.h"
#include "mesh/VertexList.h"
#include "spacetime/Spacetime.h"

namespace tessera::rl {

namespace cob = ::tessera::cobordism;
using cob::MultiCobordism;

namespace {
double lerp(double lo, double hi, double t) {
  t = std::clamp(t, 0.0, 1.0);
  return lo + (hi - lo) * t;
}
// Signed log: sign(x)·log1p(|x|), which compresses the wide range of F and ||grad S||^2.
double slog(double x) { return std::copysign(std::log1p(std::abs(x)), x); }
}  // namespace

CobordismObjectiveEnv::CobordismObjectiveEnv(NodeFactory nodeFactory,
                                             std::vector<std::complex<double>> target,
                                             EnvConfig config)
    : nodeFactory_(std::move(nodeFactory)),
      target_(std::move(target)),
      config_(config) {}

CobordismObjectiveEnv::Metrics CobordismObjectiveEnv::computeMetrics() const {
  // Read the engine's published quantities rather than recomputing the construction. F is
  // assembled from its two components, giving the value objective() returns without a third
  // eigensolve. r_state is only defined when a whole-cobordism target is set.
  const auto st = node_->spacetime();
  Metrics m;
  m.gradN2 = MultiCobordism::reggeActionGradient(st);
  m.rU = node_->rU(st);
  m.F = m.gradN2 + config_.gamma * m.rU;
  m.betti = MultiCobordism::betti(*st);
  m.holes = static_cast<int>(
      MultiCobordism::emergentHoles(*st, config_.registerDegree).size());
  m.rstate = target_.empty()
                 ? std::numeric_limits<double>::quiet_NaN()
                 : MultiCobordism::residualOfTargetStateAgainstHarmonic(
                       st, config_.registerDegree, target_);
  m.nVertices = static_cast<int>(st->getVertexList()->toVector().size());
  m.nEdges = static_cast<int>(st->getEdgeList()->toVector().size());
  m.nTopCells = static_cast<int>(st->getTopSimplices().size());
  return m;
}

bool CobordismObjectiveEnv::isCarried(const Metrics &m) const {
  // Carried: the target color state is an L_k harmonic over at least targetHoles emergent
  // holes. With no whole-cobordism target (recombination), success is realizability r_U ~ 0.
  if (target_.empty()) return m.rU < 1e-3;
  return m.holes >= config_.targetHoles && m.rstate < config_.carryTol;
}

std::vector<float> CobordismObjectiveEnv::observation(const Metrics &m) const {
  std::vector<float> obs;
  obs.reserve(kObsDim);
  obs.push_back(static_cast<float>(slog(m.F)));
  obs.push_back(static_cast<float>(slog(m.gradN2)));
  obs.push_back(static_cast<float>(slog(m.rU)));
  obs.push_back(static_cast<float>(slog(std::isnan(m.rstate) ? 0.0 : m.rstate)));
  for (int i = 0; i < kBettiSlots; ++i)
    obs.push_back(i < static_cast<int>(m.betti.size())
                      ? static_cast<float>(m.betti[i])
                      : 0.0f);
  obs.push_back(static_cast<float>(m.holes));
  obs.push_back(static_cast<float>(m.nVertices) / 50.0f);
  obs.push_back(static_cast<float>(m.nEdges) / 100.0f);
  obs.push_back(static_cast<float>(m.nTopCells) / 50.0f);
  obs.push_back(static_cast<float>(stepsTaken_) /
                static_cast<float>(std::max(1, config_.maxActions)));
  for (int mv = 0; mv < kNumMoves; ++mv)
    obs.push_back(lastMove_ == mv ? 1.0f : 0.0f);
  return obs;
}

std::vector<float> CobordismObjectiveEnv::reset(std::uint64_t seed) {
  seed_ = seed;
  node_ = nodeFactory_(seed);
  stepsTaken_ = 0;
  lastMove_ = -1;
  carried_ = false;
  lastMetrics_ = computeMetrics();
  currentF_ = lastMetrics_.F;
  return observation(lastMetrics_);
}

StepResult CobordismObjectiveEnv::step(Move move, std::array<float, kParamDim> params) {
  const double intensity = std::clamp(static_cast<double>(params[0]), 0.0, 1.0);
  const double knob = std::clamp(static_cast<double>(params[1]), 0.0, 1.0);

  const Metrics prev = lastMetrics_;  // state before this macro-action
  const double fBefore = currentF_;
  bool engineError = false;
  try {
    if (move == Move::Grow) {
      const int maxSteps = static_cast<int>(std::lround(
          lerp(config_.growSteps.first, config_.growSteps.second, intensity)));
      node_->buildStep(MultiCobordism::BuildAction::Grow, std::max(1, maxSteps),
                       config_.nCandidateMoves);
      if (config_.directedGrow)  // finish a register the random draws left short
        (void)node_->directedConeOut();
    } else if (move == Move::Evolve) {
      const int maxSteps = static_cast<int>(std::lround(
          lerp(config_.evolveSteps.first, config_.evolveSteps.second, intensity)));
      node_->buildStep(MultiCobordism::BuildAction::Evolve, std::max(1, maxSteps),
                       config_.nCandidateMoves);
      if (config_.directedGrow)  // select the register (drop a hole that hurts the carry)
        (void)node_->directedConeIn();
    } else {  // Move::Relax
      const int maxIters = static_cast<int>(std::lround(
          lerp(config_.relaxIters.first, config_.relaxIters.second, intensity)));
      const double beta = lerp(config_.betaRange.first, config_.betaRange.second, knob);
      const double alpha0 =
          lerp(config_.alphaRange.first, config_.alphaRange.second, intensity);
      node_->buildStep(MultiCobordism::BuildAction::Relax, /*maxSteps=*/30,
                       config_.nCandidateMoves, beta,
                       std::max(1, maxIters), alpha0);
    }
  } catch (...) {
    engineError = true;  // a failed engine stage makes this action a no-op; penalized below
  }

  lastMove_ = static_cast<int>(move);
  ++stepsTaken_;
  const Metrics m = computeMetrics();
  lastMetrics_ = m;
  currentF_ = m.F;

  // Reward: a dense, telescoping signed-log −ΔF term, optional proton shaping, a one-time
  // carry bonus, and a small penalty for a faulting engine move. With both shaping weights
  // at zero the reward is exactly the −ΔF drop.
  const double dFterm = config_.rewardScale * (slog(fBefore) - slog(m.F));
  double holeTerm = 0.0;
  if (config_.holeRewardWeight != 0.0) {
    const int prevH = std::min(prev.holes, config_.targetHoles);
    const int curH = std::min(m.holes, config_.targetHoles);
    holeTerm = config_.holeRewardWeight * static_cast<double>(curH - prevH);
  }
  double rstateTerm = 0.0;
  if (config_.rstateRewardWeight != 0.0 && !target_.empty() &&
      std::isfinite(prev.rstate) && std::isfinite(m.rstate)) {
    rstateTerm = config_.rstateRewardWeight * (slog(prev.rstate) - slog(m.rstate));
  }
  const double errorTerm = engineError ? -0.1 : 0.0;
  const bool carriedNow = isCarried(m);
  const double carryTerm = (carriedNow && !carried_) ? config_.carryBonus : 0.0;

  StepResult r;
  r.reward = dFterm + holeTerm + rstateTerm + errorTerm + carryTerm;
  r.terminated = carriedNow && config_.terminateOnCarry;
  carried_ = carriedNow;
  r.truncated = stepsTaken_ >= config_.maxActions;
  r.done = r.terminated || r.truncated;
  r.obs = observation(m);
  r.move = static_cast<int>(move);
  r.F = m.F;
  r.deltaF = m.F - fBefore;
  r.rU = m.rU;
  r.rstate = m.rstate;
  r.holes = m.holes;
  r.carried = carriedNow;
  r.engineError = engineError;
  return r;
}

NodeFactory formationNodeFactory(int registerDegree, double gamma, double inputWeight) {
  return [registerDegree, gamma, inputWeight](std::uint64_t seed)
             -> std::shared_ptr<MultiCobordism> {
    cob::Proton proton(seed, registerDegree, gamma, inputWeight);
    return proton.formationNode(seed);
  };
}

NodeFactory recombinationNodeFactory(int registerDegree, double gamma, double inputWeight) {
  return [registerDegree, gamma, inputWeight](std::uint64_t seed)
             -> std::shared_ptr<MultiCobordism> {
    cob::Proton proton(seed, registerDegree, gamma, inputWeight);
    return proton.recombinationNode(seed);
  };
}

CobordismObjectiveEnv makeFormationEnv(EnvConfig config, double inputWeight) {
  return CobordismObjectiveEnv(
      formationNodeFactory(config.registerDegree, config.gamma, inputWeight),
      cob::Proton::singlet(), config);
}

CobordismObjectiveEnv makeRecombinationEnv(EnvConfig config, double inputWeight) {
  return CobordismObjectiveEnv(
      recombinationNodeFactory(config.registerDegree, config.gamma, inputWeight),
      /*target=*/{}, config);
}

}  // namespace tessera::rl
