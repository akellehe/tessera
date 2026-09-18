// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "cobordism/ProtonIngredients.h"

#include <cmath>
#include <limits>
#include <utility>

#include "cobordism/MultiCobordism.h"
#include "mesh/Edge.h"
#include "mesh/EdgeList.h"
#include "mesh/Vertex.h"
#include "mesh/VertexList.h"
#include "spacetime/Foliation.h"
#include "spacetime/Metric.h"
#include "spacetime/Signature.h"
#include "spacetime/Spacetime.h"
#include "spacetime/topologies/SolidSimplex.h"
#include "spacetime/topologies/Topology.h"

namespace tessera::cobordism {

using complexd = std::complex<double>;

namespace {
constexpr int kDim = 4;  // framework dimension; the seed is a single Δ⁴ simplex
// How many settle-and-verify evolution+relaxation passes an attempt may take
// before it is judged non-persistent. The last pass must leave holes, b_k and F
// stable; earlier passes are allowed to finish settling a not-quite-stationary
// complex rather than throwing the attempt away.
constexpr int kMaxPersistencePasses = 3;
}  // namespace

ProtonIngredients::ProtonIngredients(std::uint64_t seed, int registerDegree,
                                     double gamma, double inputWeight, int precone,
                                     bool shouldUseDirectedSurgery)
    : proton_(seed, registerDegree, gamma, inputWeight, precone,
              shouldUseDirectedSurgery),
      baseSeed_(seed),
      registerDegree_(registerDegree),
      gamma_(gamma),
      inputResidualWeight_(inputWeight),
      precone_(precone),
      shouldUseDirectedSurgery_(shouldUseDirectedSurgery) {}

std::shared_ptr<Spacetime> ProtonIngredients::buildMinimalSeed() {
  // One pentatope with uniform |l^2| = 1, from the dimension-generic builder
  // the canonical arm also uses. The metric is all-spacelike by design: at
  // initialization no time has passed, so no causal structure is put in by
  // hand and any causal content must emerge.
  //
  // Unbalanced wiring, which is what this arm has always built. Unlike
  // Proton::buildMinimalSeed there is no parameter to select the balanced
  // variant, because nothing composes this arm with balanced edges.
  return MultiCobordism::seedSimplex(kDim, /*balancedEdges=*/false);
}

std::shared_ptr<MultiCobordism> ProtonIngredients::recombinationNode(
    std::uint64_t seed) const {
  // Step A is the canonical arm's node, verbatim: the composed Proton defines
  // the recombination setup.
  return proton_.recombinationNode(seed);
}

std::shared_ptr<MultiCobordism> ProtonIngredients::formationNode(
    std::uint64_t seed) const {
  // Step B with nothing pinned: the same seed complex and the same ideal diquark
  // {1,ω} plus third quark {ω²} inputs as Proton::formationNode, but with an
  // empty output-target list, so the objective's matter term is the inputs'
  // residuals alone and the whole's final state emerges. MultiCobordism supports
  // the empty list: with no output targets, rU sums only the input blocks.
  const complexd w = Proton::omega();
  const std::vector<complexd> diquark = {complexd(1.0, 0.0), w};
  const std::vector<complexd> thirdQuark = {w * w};
  auto host = buildMinimalSeed();
  // Capture the seed vertex ids before constructing the node (see
  // Proton::recombinationNode): precone_ > 0 regrows the complex in the
  // constructor, but the seed ids persist.
  std::vector<std::uint64_t> seedVertexIds;
  for (const auto *vertex : host->getVertexList()->toVector())
    seedVertexIds.push_back(vertex->getId());
  auto node = std::make_shared<MultiCobordism>(
      host, std::vector<std::vector<complexd>>{diquark, thirdQuark},
      std::vector<std::vector<complexd>>{},  // nothing pinned — the final state emerges
      std::vector<int>{registerDegree_}, gamma_, seed, precone_);
  node->setInputResidualWeight(inputResidualWeight_);
  node->seedInputs({seedVertexIds[0], seedVertexIds[1]});
  return node;
}

std::shared_ptr<MultiCobordism> ProtonIngredients::jointNode(
    std::uint64_t seed) const {
  // The joint inputs-only node; ProtonIngredients::jointNode states the
  // semantics. Inputs are the three Z₃-symmetric neutral pairs, the only
  // prepared content; outputs = {}. No diquark, bare quark or intermediate is
  // imposed anywhere — the pre-registered expectation, a baryon with a conjugate
  // partner, is read off the relaxed whole afterwards, never driven.
  const std::vector<std::vector<complexd>> pairs = {
      {complexd(1.0, 0.0), complexd(-1.0, 0.0), complexd(0.0, 0.0)},
      {complexd(0.0, 0.0), complexd(1.0, 0.0), complexd(-1.0, 0.0)},
      {complexd(-1.0, 0.0), complexd(0.0, 0.0), complexd(1.0, 0.0)}};
  auto host = buildMinimalSeed();
  // Capture the seed vertex ids before constructing the node (see
  // Proton::recombinationNode): precone_ > 0 regrows the complex in the
  // constructor, but the seed ids persist.
  std::vector<std::uint64_t> seedVertexIds;
  for (const auto *vertex : host->getVertexList()->toVector())
    seedVertexIds.push_back(vertex->getId());
  auto node = std::make_shared<MultiCobordism>(
      host, pairs,
      std::vector<std::vector<complexd>>{},  // nothing pinned — the final state emerges
      std::vector<int>{registerDegree_}, gamma_, seed, precone_);
  node->setInputResidualWeight(inputResidualWeight_);
  node->seedInputs({seedVertexIds[0], seedVertexIds[1], seedVertexIds[2]});
  return node;
}

void ProtonIngredients::build(int maxRestarts, int initSteps, int evolveSteps,
                              int stage1CandidateMoves,
                              double stage2Beta, int stage2MaxIters,
                              double persistRelTol) {
  if (attempted_) return;
  attempted_ = true;

  // The canonical arm's schedule, run through the same function it uses.
  const Proton::NodeDrive schedule{initSteps, evolveSteps,
                                   stage1CandidateMoves, stage2Beta,
                                   stage2MaxIters, shouldUseDirectedSurgery_};
  const auto runNode = [&](MultiCobordism &node) {
    Proton::driveNode(node, schedule);
  };

  // The answer-agnostic summary the persistence check compares: the emergent
  // hole count, b_k, and the objective. Not the singlet residual — no
  // answer-shaped quantity may steer or gate this build.
  const auto holeCount = [&](const std::shared_ptr<Spacetime> &whole) {
    return static_cast<int>(
        MultiCobordism::emergentHoles(*whole, registerDegree_).size());
  };
  const auto bettiAtRegisterDegree = [&](const std::shared_ptr<Spacetime> &whole) {
    const auto betti = MultiCobordism::betti(*whole);
    return registerDegree_ < static_cast<int>(betti.size()) ? betti[registerDegree_]
                                                            : 0;
  };

  double bestObjective = std::numeric_limits<double>::infinity();

  for (int attempt = 0; attempt < maxRestarts; ++attempt) {
    const std::uint64_t seedA = baseSeed_ + 2ULL * static_cast<std::uint64_t>(attempt);
    const std::uint64_t seedB = seedA + 1ULL;

    // ---- Step A — recombination: best-effort; its r_U is reported, not gated. ----
    auto stepA = recombinationNode(seedA);
    runNode(*stepA);
    const double diquarkR = stepA->rU(stepA->spacetime());

    // ---- Step B — formation with nothing pinned ----
    auto stepB = formationNode(seedB);
    runNode(*stepB);

    // Persistence: continued evolution (∂W frozen) plus relaxation must leave
    // the answer-agnostic summary stable. Up to kMaxPersistencePasses passes may
    // settle a not-quite-stationary complex; the last pass has to be the stable
    // one.
    bool persisted = false;
    for (int pass = 0; pass < kMaxPersistencePasses && !persisted; ++pass) {
      const auto whole = stepB->spacetime();
      const int holesBefore = holeCount(whole);
      const int bettiBefore = bettiAtRegisterDegree(whole);
      const double objectiveBefore = stepB->objective();
      stepB->runStage1(evolveSteps, stage1CandidateMoves,
                       /*growBoundaries=*/false);
      stepB->runStage2(stage2Beta, stage2MaxIters);
      const auto settled = stepB->spacetime();
      const double objectiveAfter = stepB->objective();
      persisted = holeCount(settled) == holesBefore &&
                  bettiAtRegisterDegree(settled) == bettiBefore &&
                  std::abs(objectiveAfter - objectiveBefore) <=
                      persistRelTol * std::max(std::abs(objectiveBefore), 1.0);
    }
    const bool isStationary = stepB->lastStage2Stationary();
    const bool ok = isStationary && persisted;

    const auto whole = stepB->spacetime();
    const double finalObjective = stepB->objective();

    // Keep the converged attempt, or the lowest-objective one so far. The
    // objective is the only ranking, never a target state.
    if (ok || finalObjective < bestObjective) {
      bestObjective = finalObjective;
      converged_ = ok;
      stationary_ = isStationary;
      persistent_ = persisted;
      keptSeed_ = seedA;
      spacetime_ = whole;
      emergentHoles_ = MultiCobordism::emergentHoles(*whole, registerDegree_);
      singletResidual_ = MultiCobordism::residualOfTargetStateAgainstHarmonic(
          whole, registerDegree_, Proton::singlet());  // diagnostic read, after the fact
      inputResidual_ = stepB->rU(whole);
      finalObjective_ = finalObjective;
      diquarkResidual_ = diquarkR;
    }
    if (ok) return;  // a stationary, persistent structure emerged — stop restarting
  }
}

void ProtonIngredients::ensureBuilt() {
  if (!attempted_) build();
}

bool ProtonIngredients::converged() {
  ensureBuilt();
  return converged_;
}

bool ProtonIngredients::stationary() {
  ensureBuilt();
  return stationary_;
}

bool ProtonIngredients::persistent() {
  ensureBuilt();
  return persistent_;
}

std::uint64_t ProtonIngredients::seed() {
  ensureBuilt();
  return keptSeed_;
}

std::shared_ptr<Spacetime> ProtonIngredients::spacetime() {
  ensureBuilt();
  return spacetime_;
}

std::shared_ptr<Spacetime> ProtonIngredients::block() {
  ensureBuilt();
  return spacetime_;  // the emergent object IS the whole (parity with Proton::block)
}

std::vector<std::vector<std::uint64_t>> ProtonIngredients::emergentHoles() {
  ensureBuilt();
  return emergentHoles_;
}

double ProtonIngredients::singletResidual() {
  ensureBuilt();
  return singletResidual_;
}

double ProtonIngredients::inputResidual() {
  ensureBuilt();
  return inputResidual_;
}

double ProtonIngredients::finalObjective() {
  ensureBuilt();
  return finalObjective_;
}

double ProtonIngredients::diquarkResidual() {
  ensureBuilt();
  return diquarkResidual_;
}

}  // namespace tessera::cobordism
