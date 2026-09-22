// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "cobordism/ProtonSynthesis.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#include "cobordism/MultiCobordism.h"
#include "observables/ColorFiber.h"
#include "mesh/Edge.h"
#include "mesh/EdgeList.h"
#include "mesh/Simplex.h"
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
}  // namespace

std::complex<double> ProtonSynthesis::omega() {
  // The one cube root of unity in the codebase, from ColorFiber: the algebraic
  // value (−1 + i√3)/2, never exp(2πi/3). Only the algebraic components make
  // 1 + ω + ω̄ cancel exactly; exp(2πi/3) leaves 4.4e-16, and that residual
  // lands in the colour-singlet targets this feeds.
  return observables::ColorFiber::omega();
}

std::vector<std::complex<double>> ProtonSynthesis::singlet() {
  const complexd w = omega();
  return {complexd(1.0, 0.0), w, w * w};
}

ProtonSynthesis::ProtonSynthesis(std::uint64_t seed, int registerDegree,
                                 double gamma, double inputWeight, int precone,
                                 bool shouldUseDirectedSurgery,
                                 bool preconeTimelike, bool preconeAlternate,
                                 bool balancedEdges, bool singularValueRatio,
                                 bool einsteinHilbert)
    : baseSeed_(seed),
      registerDegree_(registerDegree),
      gamma_(gamma),
      inputResidualWeight_(inputWeight),
      precone_(precone),
      shouldUseDirectedSurgery_(shouldUseDirectedSurgery),
      preconeTimelike_(preconeTimelike),
      preconeAlternate_(preconeAlternate) {
  balancedEdges_ = balancedEdges;
  singularValueRatio_ = singularValueRatio;
  einsteinHilbert_ = einsteinHilbert;
}

void ProtonSynthesis::NodeDrive::run(MultiCobordism &node) const {
  node.runStage1(initSteps, stage1CandidateMoves, /*growBoundaries=*/true);
  if (directedSurgery) (void)node.directedConeOut();
  node.runStage1(evolveSteps, stage1CandidateMoves, /*growBoundaries=*/false);
  if (directedSurgery) (void)node.directedConeIn();
  node.runStage2(stage2Beta, stage2MaxIters);
}

void ProtonSynthesis::requireSynthesisMode(const MultiCobordism &node) {
  // The synthesis pins colour targets. Targets are permitted only in the
  // labelled controlled-synthesis mode; the emergence protocol pins none, so a
  // targeted node may not run under the emergence label.
  const auto mode = node.simulationMode();
  if (mode == MultiCobordism::SimulationMode::Synthesis) return;
  std::string label = MultiCobordism::modeName(mode);
  if (mode == MultiCobordism::SimulationMode::Emergence)
    label += " (" + MultiCobordism::submodeName(node.emergenceSubmode()) + ")";
  throw std::invalid_argument(
      "ProtonSynthesis: the proton synthesis pins colour targets (the diquark "
      "pair and the {1, omega, omega^2} singlet) and runs only in the labelled "
      "controlled-synthesis mode (SimulationMode::Synthesis); this node is in " +
      label + " mode");
}

void ProtonSynthesis::driveNode(MultiCobordism &node, const NodeDrive &schedule) {
  requireSynthesisMode(node);  // before anything runs
  schedule.run(node);
}

std::shared_ptr<Spacetime> ProtonSynthesis::buildMinimalSeed(bool balancedEdges) {
  // A single Δ⁴ simplex (one pentatope, 5 vertices). Nothing is pre-built: the
  // proton's whole topology is grown from here, and the metric is uniform
  // (ℓ² = 1) so the geometry comes from the relaxation. Only the seed simplex
  // and the target colour states are imposed. The dimension-generic builder is
  // `MultiCobordism::seedSimplex`.
  return MultiCobordism::seedSimplex(kDim, balancedEdges);
}

std::shared_ptr<MultiCobordism> ProtonSynthesis::recombinationNode(std::uint64_t seed) const {
  // Step A inputs: two neutral q-q̄ pairs (Σ = 0). Outputs: a coloured diquark
  // {1,ω} ⊔ antidiquark {1,ω²} — 2-vectors, not the singlet. Seeded on a fresh
  // single-Δ⁴ seed, inputs at v0,v1 and outputs at v2,v3. Not run; the caller
  // drives it.
  const complexd w = omega();
  const std::vector<std::vector<complexd>> pairs = {
      {complexd(1.0, 0.0), complexd(-1.0, 0.0), complexd(0.0, 0.0)},
      {complexd(1.0, 0.0), complexd(0.0, 0.0), complexd(-1.0, 0.0)}};
  const std::vector<complexd> diquark = {complexd(1.0, 0.0), w};
  const std::vector<complexd> antidiquark = {complexd(1.0, 0.0), w * w};
  auto host = buildMinimalSeed(balancedEdges_);
  // Capture the seed vertex ids (not Vertex*) before constructing the node:
  // with precone_ > 0 the constructor regrows spacetime_ into a fresh complex,
  // destroying the original host's Vertex objects. The seed ids persist through
  // the rebuilds (build() preserves vertex ids), so the input and output anchors
  // stay valid.
  std::vector<std::uint64_t> seedVertexIds;
  for (const auto *vertex : host->getVertexList()->toVector())
    seedVertexIds.push_back(vertex->getId());
  auto node = std::make_shared<MultiCobordism>(
      host, pairs, std::vector<std::vector<complexd>>{diquark, antidiquark},
      std::vector<int>{registerDegree_}, gamma_, seed, precone_,
      /*shouldProposeDispositions=*/true, preconeTimelike_, preconeAlternate_,
      balancedEdges_, singularValueRatio_, einsteinHilbert_);
  // The label: this node pins targets, so it runs as controlled synthesis.
  node->setSimulationMode(MultiCobordism::SimulationMode::Synthesis);
  node->setInputResidualWeight(inputResidualWeight_);
  node->seedInputs({seedVertexIds[0], seedVertexIds[1]});
  node->seedOutputs({seedVertexIds[2], seedVertexIds[3]});
  return node;
}

std::shared_ptr<MultiCobordism> ProtonSynthesis::formationNode(std::uint64_t seed) const {
  // Step B inputs: the diquark {1,ω} plus the third quark {ω²}. Output: the
  // proton singlet, read off the whole cobordism (no seedOutputs). Seeded on a
  // fresh single-Δ⁴ seed, inputs at v0,v1. Not run; the caller drives it.
  const complexd w = omega();
  const std::vector<complexd> diquark = {complexd(1.0, 0.0), w};
  const std::vector<complexd> thirdQuark = {w * w};
  auto host = buildMinimalSeed(balancedEdges_);
  // Capture the seed vertex ids before constructing the node (see
  // recombinationNode): precone_ > 0 regrows the complex in the constructor, but
  // the seed ids persist.
  std::vector<std::uint64_t> seedVertexIds;
  for (const auto *vertex : host->getVertexList()->toVector())
    seedVertexIds.push_back(vertex->getId());
  auto node = std::make_shared<MultiCobordism>(
      host, std::vector<std::vector<complexd>>{diquark, thirdQuark},
      std::vector<std::vector<complexd>>{singlet()},
      std::vector<int>{registerDegree_}, gamma_, seed, precone_,
      /*shouldProposeDispositions=*/true, preconeTimelike_, preconeAlternate_,
      balancedEdges_, singularValueRatio_, einsteinHilbert_);
  // The label: this node pins targets, so it runs as controlled synthesis.
  node->setSimulationMode(MultiCobordism::SimulationMode::Synthesis);
  node->setInputResidualWeight(inputResidualWeight_);
  node->seedInputs({seedVertexIds[0], seedVertexIds[1]});
  return node;
}

std::shared_ptr<MultiCobordism> ProtonSynthesis::directNode(std::uint64_t seed) const {
  // One-step inputs: the three bare quarks {1}, {ω}, {ω²} and their three
  // anti-quarks, the elementwise conjugates {1}, {ω̄}, {ω̄²} (conjugation is the
  // antiparticle convention here: the antidiquark {1, ω²} is exactly the
  // conjugate of the diquark {1, ω}), so the prepared content is three q-q̄
  // pairs, not three quarks from nothing. Output: the proton singlet, read off
  // the whole cobordism (no seedOutputs, as in formationNode); the anti-baryon
  // partner is not pinned. Seeded on a fresh single-Δ⁴ seed. Not
  // run; the caller drives it.
  const complexd w = omega();
  const std::vector<std::vector<complexd>> quarksAndAntiquarks = {
      {complexd(1.0, 0.0)}, {w}, {w * w},
      {complexd(1.0, 0.0)}, {std::conj(w)}, {std::conj(w * w)}};
  auto host = buildMinimalSeed(balancedEdges_);
  // Capture the seed vertex ids before constructing the node (see
  // recombinationNode): precone_ > 0 regrows the complex in the constructor, but
  // the seed ids persist.
  std::vector<std::uint64_t> seedVertexIds;
  for (const auto *vertex : host->getVertexList()->toVector())
    seedVertexIds.push_back(vertex->getId());
  auto node = std::make_shared<MultiCobordism>(
      host, quarksAndAntiquarks, std::vector<std::vector<complexd>>{singlet()},
      std::vector<int>{registerDegree_}, gamma_, seed, precone_,
      /*shouldProposeDispositions=*/true, preconeTimelike_, preconeAlternate_,
      balancedEdges_, singularValueRatio_, einsteinHilbert_);
  // The label: this node pins targets, so it runs as controlled synthesis.
  node->setSimulationMode(MultiCobordism::SimulationMode::Synthesis);
  node->setInputResidualWeight(inputResidualWeight_);
  // Six blocks on a 5-vertex Δ⁴ seed, so the anchors cycle. On the bare seed
  // every block's region is the seed's full cell-neighbourhood anyway — the
  // anchor only distinguishes one block from another — and the blocks
  // differentiate as the gated growth takes each region where its own residual
  // wants it.
  std::vector<std::uint64_t> inputSeedVertexIds;
  for (std::size_t blockIndex = 0; blockIndex < quarksAndAntiquarks.size();
       ++blockIndex)
    inputSeedVertexIds.push_back(
        seedVertexIds[blockIndex % seedVertexIds.size()]);
  node->seedInputs(inputSeedVertexIds);
  return node;
}

void ProtonSynthesis::buildDirect(int maxRestarts, int initSteps,
                                  int evolveSteps, int stage1CandidateMoves,
                                  double stage2Beta, double colorTolerance,
                                  int minEmergentHoles) {
  if (attempted_) return;
  attempted_ = true;

  const std::vector<complexd> protonSinglet = singlet();
  double bestColorResidual = std::numeric_limits<double>::infinity();

  for (int attempt = 0; attempt < maxRestarts; ++attempt) {
    const std::uint64_t seed = baseSeed_ + static_cast<std::uint64_t>(attempt);
    auto node = directNode(seed);
    requireSynthesisMode(*node);  // `run` below bypasses driveNode
    // The combined drive: every `run` iteration interleaves the stage-1 surgery
    // update with the stage-2 geometric relaxation, so the optimizer takes
    // whichever kind of progress helps at each point — an initialization pass
    // growing the input regions until they carry, then an evolution pass with ∂W
    // frozen. The relaxation is folded into every iteration rather than run as a
    // separate pass.
    node->run(initSteps, stage1CandidateMoves, /*growBoundaries=*/true, stage2Beta);
    if (shouldUseDirectedSurgery_)  // directed surgery: remove cells / cap facets
      (void)node->directedConeOut();
    node->run(evolveSteps, stage1CandidateMoves, /*growBoundaries=*/false, stage2Beta);
    if (shouldUseDirectedSurgery_)  // select the best register (drop holes that hurt)
      (void)node->directedConeIn();

    auto whole = node->spacetime();
    const double colorR = MultiCobordism::residualOfTargetStateAgainstHarmonic(
        whole, registerDegree_, protonSinglet, node->metricSource());
    auto holes = MultiCobordism::emergentHoles(*whole, registerDegree_);
    const bool ok = colorR < colorTolerance &&
                    static_cast<int>(holes.size()) >= minEmergentHoles;

    // Keep the converged attempt, or the lowest-residual one so far otherwise.
    if (ok || colorR < bestColorResidual) {
      bestColorResidual = colorR;
      converged_ = ok;
      convergedSeed_ = seed;
      spacetime_ = whole;
      block_ = whole;  // the proton is the whole cobordism (read off the whole)
      emergentHoles_ = std::move(holes);
      colorResidual_ = colorR;
      diquarkResidual_ = 0.0;  // no step A in the one-step build
    }
    if (ok) return;  // the synthesis converged — stop restarting
  }
}

void ProtonSynthesis::build(int maxRestarts, int initSteps, int evolveSteps,
                            int stage1CandidateMoves, double stage2Beta,
                            int stage2MaxIters, double colorTolerance,
                            int minEmergentHoles) {
  if (attempted_) return;
  attempted_ = true;

  const std::vector<complexd> protonSinglet = singlet();

  // Drive one already-seeded node: an initialization pass that grows the
  // boundary regions until they carry (grow_boundaries=true), an evolution pass
  // with ∂W frozen (grow_boundaries=false), then the geometric relaxation. Node
  // setup — seed, targets, seeding, input weight — lives in
  // recombinationNode/formationNode, which stamp each node
  // SimulationMode::Synthesis; driveNode refuses any other mode.
  const NodeDrive schedule{initSteps, evolveSteps, stage1CandidateMoves,
                           stage2Beta, stage2MaxIters,
                           shouldUseDirectedSurgery_};
  const auto runNode = [&](MultiCobordism &node) {
    driveNode(node, schedule);
  };

  double bestColorResidual = std::numeric_limits<double>::infinity();

  for (int attempt = 0; attempt < maxRestarts; ++attempt) {
    const std::uint64_t seedA = baseSeed_ + 2ULL * static_cast<std::uint64_t>(attempt);
    const std::uint64_t seedB = seedA + 1ULL;

    // ---- Step A — recombination: best-effort; its r_U is reported, not gated. ----
    auto stepA = recombinationNode(seedA);
    runNode(*stepA);
    const double diquarkR = stepA->rU(stepA->spacetime());

    // ---- Step B — formation: the proton, read off the whole cobordism ----
    auto stepB = formationNode(seedB);
    runNode(*stepB);
    auto whole = stepB->spacetime();
    const double colorR = MultiCobordism::residualOfTargetStateAgainstHarmonic(
        whole, registerDegree_, protonSinglet, stepB->metricSource());
    auto holes = MultiCobordism::emergentHoles(*whole, registerDegree_);
    const bool ok = colorR < colorTolerance &&
                    static_cast<int>(holes.size()) >= minEmergentHoles;

    // Keep the converged attempt, or the lowest-residual one so far otherwise.
    if (ok || colorR < bestColorResidual) {
      bestColorResidual = colorR;
      converged_ = ok;
      convergedSeed_ = seedA;
      spacetime_ = whole;
      block_ = whole;  // the proton is the whole cobordism (read off the whole)
      emergentHoles_ = std::move(holes);
      colorResidual_ = colorR;
      diquarkResidual_ = diquarkR;
    }
    if (ok) return;  // the synthesis converged — stop restarting
  }
}

void ProtonSynthesis::ensureBuilt() {
  if (!attempted_) build();
}

bool ProtonSynthesis::converged() {
  ensureBuilt();
  return converged_;
}

std::uint64_t ProtonSynthesis::seed() {
  ensureBuilt();
  return convergedSeed_;
}

std::shared_ptr<Spacetime> ProtonSynthesis::spacetime() {
  ensureBuilt();
  return spacetime_;
}

std::shared_ptr<Spacetime> ProtonSynthesis::block() {
  ensureBuilt();
  return block_;
}

std::vector<std::vector<std::uint64_t>> ProtonSynthesis::emergentHoles() {
  ensureBuilt();
  return emergentHoles_;
}

double ProtonSynthesis::colorResidual() {
  ensureBuilt();
  return colorResidual_;
}

double ProtonSynthesis::diquarkResidual() {
  ensureBuilt();
  return diquarkResidual_;
}

}  // namespace tessera::cobordism
