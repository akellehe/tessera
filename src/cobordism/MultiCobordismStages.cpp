// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

// MultiCobordism (include/cobordism/MultiCobordism.h): the two stages and
// the cone moves: block-region growth, stage 1 and stage 2, run, and the
// directed and random cones. One of the translation units that define the
// class's members by responsibility
// (https://github.com/akellehe/tessera/issues/1481).

#include "MultiCobordismInternal.h"

namespace tessera::cobordism {

namespace {

// The boundary facets of `spacetime`, as sorted vertex-id tuples, for membership tests.
std::set<std::vector<std::uint64_t>> boundaryFacetSet(const Spacetime &spacetime) {
  std::set<std::vector<std::uint64_t>> facets;
  for (auto facet : spacetime.getBoundary()) {  // getBoundary() returns a fresh copy
    std::sort(facet.begin(), facet.end());
    facets.insert(facet);
  }
  return facets;
}

}  // namespace

void MultiCobordism::growBlockRegions() {
  // Growth is a setup step: it runs only before the bulk is connected, and only
  // when a shell strictly lowers the block's residual. Both conditions are
  // needed to give growth a stopping point, since a block that is not carrying
  // sits at the constant full-leak residual for any region size — so a
  // "keep unless the residual rises" gate scores every shell as an exact tie
  // and the regions grow until they cover the whole complex.
  if (bulkConnected_) return;   // the bulk is linked; the states stay as they are
  // Expand one block's read window by a shell — the vertices of every top cell
  // touching it — so it gets room to open the holes that carry it. A block
  // already carrying (residual < tolerance) is left alone.
  //
  // This grows a scoring region, never the cobordism's boundary: the only write
  // is to `block.vertices`, and every `spacetime_` access below is a read. A
  // shell is kept only when it strictly lowers the block's r_U term, so region
  // growth can never raise F and a permanently-leaking block cannot grow
  // forever.
  const auto growOneShell = [this](BoundaryBlock &block) {
    const double residualBefore = residualForBoundaryBlock(block, spacetime_);
    if (residualBefore < inputCarriedTolerance_) return;
    std::set<std::uint64_t> expanded = block.vertices;
    for (const auto &topSimplex : spacetime_->getTopSimplices()) {
      auto cellVertexIds = topSimplex->topTuple();
      bool touchesRegion = false;
      for (auto vertexId : cellVertexIds)
        if (block.vertices.count(vertexId)) {
          touchesRegion = true;
          break;
        }
      if (touchesRegion)
        expanded.insert(cellVertexIds.begin(), cellVertexIds.end());
    }
    std::set<std::uint64_t> original = std::move(block.vertices);
    block.vertices = std::move(expanded);
    // Strict: a shell is kept only if it actually improves the carry. A shell
    // that leaves the residual unchanged buys nothing and is what let the
    // regions sprawl, so it is reverted like a harmful one.
    if (residualForBoundaryBlock(block, spacetime_) >= residualBefore)
      block.vertices = std::move(original);
  };
  for (auto &inputBlock : inputBlocks_) growOneShell(inputBlock);
  // Localized output blocks (a 2→2 recombination's diquark ⊔ antidiquark) grow the
  // same way; a single output reads off the whole and has no block here, so this is
  // a no-op for the formation node.
  for (auto &outputBlock : outputBlocks_) growOneShell(outputBlock);
}

std::vector<int> MultiCobordism::depthSchedule(int maxLookahead,
                                               int combinatorialBreadth) {
  std::vector<int> schedule;
  if (combinatorialBreadth > 0) {
    // Backing off: sequences of exactly `combinatorialBreadth` moves are
    // searched first, and the search shortens by one move each time nothing at
    // the current breadth lowers F, down to single moves. The breadth is
    // searched on its own, not on top of the shorter ones: the question the
    // schedule asks is whether a composition of that length improves a complex
    // no shorter composition improves, and answering it means looking there
    // first rather than only on a plateau.
    schedule.reserve(static_cast<std::size_t>(combinatorialBreadth));
    for (int depth = combinatorialBreadth; depth >= 1; --depth)
      schedule.push_back(depth);
    return schedule;
  }
  // Iterative deepening (the default): single moves first — the cheap, common
  // case — deepening only on a stall.
  const int deepest = std::max(1, maxLookahead);
  schedule.reserve(static_cast<std::size_t>(deepest));
  for (int depth = 1; depth <= deepest; ++depth) schedule.push_back(depth);
  return schedule;
}

std::vector<double> MultiCobordism::runStage1(int maxSteps, int nCandidateMoves,
                                                 bool growBoundaries,
                                                 int maxLookahead,
                                                 int combinatorialBreadth) {
  std::vector<double> objectiveTrace = {objective()};
  for (int stepIndex = 0; stepIndex < maxSteps; ++stepIndex)
    if (!stage1Update(nCandidateMoves, growBoundaries, objectiveTrace,
                      maxLookahead, combinatorialBreadth))
      break;
  return objectiveTrace;
}

bool MultiCobordism::stage1Update(int nCandidateMoves, bool growBoundaries,
                                  std::vector<double> &objectiveTrace,
                                  int maxLookahead,
                                  int combinatorialBreadth) {
  // In target-conditioned modes the register is "carried" once summed r_U is
  // essentially zero. JointStationarity never consults this target diagnostic.
  constexpr double kRegisterCarriedTolerance = 1e-3;
  // Initialization only: while establishing the boundary states, let each
  // not-yet-carrying block expand its scoring region by a shell so it can
  // develop the holes that carry its state. Off during the bulk evolution.
  // This never moves ∂W (see growBlockRegions).
  //
  // Growing a region changes F, so it is booked into the trace; with the
  // per-block gate in `growBlockRegions` the booked delta is always <= 0.
  // `growBlockRegions` mutates only the blocks' vertex sets and never touches
  // `spacetime_`, so `reggeActionGradient` is unchanged and the whole objective
  // change is exactly `gamma_ * Δr_U`. Leaving it unbooked would let the
  // accumulated trace drift away from `objective()`, and that same accumulated
  // quantity gates acceptance.
  if (growBoundaries && objectiveSpec_->needsRegisterResidual()) {
    const double objectiveBeforeGrowth = objective();
    growBlockRegions();
    const double growthObjectiveDelta = objective() - objectiveBeforeGrowth;
    if (growthObjectiveDelta != 0.0)
      objectiveTrace.push_back(objectiveTrace.back() + growthObjectiveDelta);
  }
  // The depth ladder, in whichever order `depthSchedule` gives it: iterative
  // deepening by default (single moves first — the cheap, common case —
  // deepening to 2-move sequences, then 3, up to `maxLookahead`, only when the
  // shorter search finds nothing), or descending from `combinatorialBreadth`
  // and backing off when a breadth is named. Either way a sequence is scored
  // and committed as a whole, so an F-lowering pair whose first move alone
  // raises F is still reached by descent.
  lastStage1LookaheadDepth_ = 0;  // report: nothing committed until proven otherwise
  // A stalled search is allowed to go wide as well as deep: depth 1 keeps the
  // caller's fast batch (the common, cheap case), while each deepened batch
  // scans on the order of a hundred candidate sequences — deep sequences die on
  // the gate chain far more often and the move space grows with depth, so a
  // handful of draws would badly under-sample it. The budget is only spent
  // when depth 1 already failed, i.e. exactly when it is worth it.
  constexpr int kDeepLookaheadCandidates = 128;
  // Candidates without a localized delta use an exact global objective. Score
  // the unchanged base geometry once for the entire iterative-deepening pass
  // instead of once per candidate endpoint (up to 524 redundant evaluations at
  // depth five).
  const double baseObjective =
      compositeSupportsLocalizedDelta() ? 0.0 : objectiveFor(spacetime_);
  for (const int lookaheadDepth :
       depthSchedule(maxLookahead, combinatorialBreadth)) {
    // A non-positive `nCandidateMoves` is the exhaustive sentinel, and it
    // survives the deepening: taking `max` against the deep batch size would
    // turn "every candidate" into "128 of them" the moment the search left
    // depth 1, so a run asked to be exhaustive would quietly stop being so
    // exactly where the move space is largest.
    const int batchSize =
        nCandidateMoves <= 0 ? nCandidateMoves
        : lookaheadDepth == 1
            ? nCandidateMoves
            : std::max(nCandidateMoves, kDeepLookaheadCandidates);
    double objectiveDelta = step(batchSize, lookaheadDepth, baseObjective);
    // Final check: the draws found nothing, so the step is about to
    // report that it cannot descend. That claim is about the whole move set,
    // which a sample cannot support -- so price every available move before
    // making it, rather than calling a missed draw a local minimum.
    //
    // Only here, and only at depth 1: pricing one move costs a complex
    // rebuild, a validity check and two solver constructions, so paying for
    // the whole set on every step would make a long run intractable. This is a
    // rescue from an apparent dead end, not a second optimization pass.
    if (lookaheadDepth == 1 && batchSize > 0 &&
        objectiveDelta >= -convergenceTolerance_)
      objectiveDelta = step(0, lookaheadDepth, baseObjective);
    if (objectiveDelta < -convergenceTolerance_) {
      // An F-lowering surgery sequence: progress.
      objectiveTrace.push_back(objectiveTrace.back() + objectiveDelta);
      lastStage1LookaheadDepth_ = lookaheadDepth;
      return true;
    }
  }
  // A target-free objective is done when no improving sequence is found: there
  // is nothing else it was trying to reach. A target-conditioned one halts when
  // the register is carried; otherwise it keeps drawing, because a random miss
  // is not proof that no target-improving sequence exists. `maxSteps` bounds
  // those retries.
  if (!objectiveSpec_->isTargetConditioned()) return false;
  return rU(spacetime_) >= kRegisterCarriedTolerance;
}

void MultiCobordism::seedInputs(const std::vector<std::uint64_t> &seeds) {
  seedBlocks(seeds, inputTargets_, inputBlocks_);
}

void MultiCobordism::seedOutputs(const std::vector<std::uint64_t> &seeds) {
  seedBlocks(seeds, outputTargets_, outputBlocks_);
}

void MultiCobordism::seedBlocks(
    const std::vector<std::uint64_t> &seeds,
    const std::vector<std::vector<complexd>> &targets,
    std::vector<BoundaryBlock> &destinationBlocks) {
  // Seed one boundary block per (seed vertex, target): its initial region is the seed
  // vertex's cell-neighbourhood. The block is not pre-grown here — runStage1's
  // growBlockRegions grows it under the objective, so the carrying topology is fully
  // emergent. The seed vertex is the only anchor (it distinguishes one input/output
  // from another); everything else emerges.
  std::vector<std::set<std::uint64_t>> regions;
  for (std::size_t blockIndex = 0;
       blockIndex < targets.size() && blockIndex < seeds.size(); ++blockIndex) {
    const std::uint64_t seedVertexId = seeds[blockIndex];
    std::set<std::uint64_t> regionVertexIds;
    for (const auto &topSimplex : spacetime_->getTopSimplices()) {
      auto cellVertexIds = topSimplex->topTuple();
      if (std::find(cellVertexIds.begin(), cellVertexIds.end(), seedVertexId) !=
          cellVertexIds.end())
        regionVertexIds.insert(cellVertexIds.begin(), cellVertexIds.end());
    }
    regions.push_back(std::move(regionVertexIds));
  }
  seedBlockRegions(regions, targets, destinationBlocks, /*surface=*/false);
}

void MultiCobordism::seedBlockRegions(
    const std::vector<std::set<std::uint64_t>> &regions,
    const std::vector<std::vector<complexd>> &targets,
    std::vector<BoundaryBlock> &destinationBlocks, bool surface) {
  for (std::size_t blockIndex = 0;
       blockIndex < targets.size() && blockIndex < regions.size(); ++blockIndex) {
    BoundaryBlock block{regions[blockIndex], targets[blockIndex]};
    block.surface = surface;
    // A surface block carries its own faces from here on (see
    // `BoundaryBlock::faces`): the surface's simplices inside its vertex set
    // as the host holds them at seeding — registered on a bare-surface host,
    // facets of the collar's cells on a collar.
    if (surface) block.faces = blockSurface(block, *spacetime_).faces;
    destinationBlocks.push_back(std::move(block));
  }
}

void MultiCobordism::seedInputs(const std::vector<std::vector<std::uint64_t>> &regions) {
  std::set<std::uint64_t> live;
  for (const auto *vertex : spacetime_->getVertexList()->toVector())
    if (vertex != nullptr) live.insert(vertex->getId());
  std::vector<std::set<std::uint64_t>> regionSets;
  for (const auto &region : regions) {
    if (region.empty())
      throw std::invalid_argument("MultiCobordism::seedInputs: an input region is empty");
    std::set<std::uint64_t> vertices(region.begin(), region.end());
    for (const std::uint64_t v : vertices)
      if (!live.count(v))
        throw std::invalid_argument("MultiCobordism::seedInputs: region vertex " + std::to_string(v) +
                                    " is not a vertex of the host");
    regionSets.push_back(std::move(vertices));
  }
  seedBlockRegions(regionSets, inputTargets_, inputBlocks_, /*surface=*/true);
}

std::vector<double> MultiCobordism::runStage2(double beta, int maxIters,
                                                 double alpha0, double tolerance) {
  setReggeWeight(beta);
  std::vector<double> objectiveTrace = {objective()};
  double stepScale = alpha0;
  lastStage2Stationary_ = false;  // for maxIters == 0; each update reports its own
  for (int iterationIndex = 0; iterationIndex < maxIters; ++iterationIndex)
    if (!stage2Update(beta, tolerance, objectiveTrace, stepScale)) break;
  return objectiveTrace;
}

std::vector<double> MultiCobordism::run(int maxIters, int nCandidateMoves,
                                        bool growBoundaries, double beta,
                                        double alpha0, double tolerance,
                                        int maxLookahead,
                                        int relaxBudgetPerMove,
                                        int combinatorialBreadth) {
  setReggeWeight(beta);
  std::vector<double> objectiveTrace = {objective()};
  double stepScale = alpha0;
  lastStage2Stationary_ = false;  // for maxIters == 0; each update reports its own
  // A single stalled stage-1 batch is a random-draw miss, not proof the moves
  // have no effect (measured on a timelike-preconed drive: committed moves landed
  // several stalled batches apart), so exhaustion is only concluded after this
  // many consecutive no-effect iterations.
  constexpr int kConsecutiveNoEffectLimit = 3;
  int consecutiveNoEffect = 0;
  for (int iterationIndex = 0; iterationIndex < maxIters; ++iterationIndex) {
    // One combinatorial move (or lookahead sequence), then a full geometric
    // relaxation: stage-2 updates repeat until the absolute-improvement test
    // reports diminishing returns. Every committed move is therefore scored
    // from — and leaves behind — relaxed geometry (stage2Update re-reads the
    // edge list each call, picking up whatever the move just created).
    const bool stage1WantsAnotherIteration = stage1Update(
        nCandidateMoves, growBoundaries, objectiveTrace, maxLookahead,
        combinatorialBreadth);
    const bool moveCommitted = lastStage1LookaheadDepth_ > 0;
    // "Full" relaxation still needs a safety budget (as runStage2's maxIters):
    // Near a slow descent tail the line search can accept a near-unbounded
    // number of threshold-sized micro-steps, so the stationarity test alone
    // does not bound the loop in practice. Caller-tunable; the
    // stationarity test remains the real terminator.
    bool geometryRelaxed = false;
    for (int relaxIndex = 0; relaxIndex < relaxBudgetPerMove; ++relaxIndex) {
      if (!stage2Update(beta, tolerance, objectiveTrace, stepScale)) break;
      geometryRelaxed = true;
    }
    // "The combinatorial moves have no effect": nothing committed at any
    // lookahead depth and nothing left to relax — but only after enough
    // consecutive misses to rule out draw noise.
    if (!moveCommitted && !geometryRelaxed)
      ++consecutiveNoEffect;
    else
      consecutiveNoEffect = 0;
    const bool wantsExit =
        (!stage1WantsAnotherIteration && !geometryRelaxed) ||
        consecutiveNoEffect >= kConsecutiveNoEffectLimit;
    if (wantsExit) {
      // The last geometric relaxation before exit runs at a much tighter
      // tolerance than the in-loop diminishing-returns cut. If the tighter pass
      // still finds descent the state was not truly stationary — the exit was
      // premature — so keep looping on the freshly relaxed geometry (which may
      // also enable new moves). Exit only once stationary at 1e-12 too.
      constexpr double kExitRelTol = 1e-12;
      bool tighterPassFoundDescent = false;
      for (int relaxIndex = 0; relaxIndex < relaxBudgetPerMove; ++relaxIndex) {
        if (!stage2Update(beta, kExitRelTol, objectiveTrace, stepScale)) break;
        tighterPassFoundDescent = true;
      }
      if (!tighterPassFoundDescent) break;
      consecutiveNoEffect = 0;  // it moved: not done after all
    }
  }
  return objectiveTrace;
}

bool MultiCobordism::stage2Update(double beta, double tolerance,
                                  std::vector<double> &objectiveTrace,
                                  double &stepScale) {
  // Reset-then-set: the flag reports this call's outcome, so in the combined
  // drive (`run`) it reflects the most recent geometric update instead of
  // latching true after a stationary point a later topology change reopened.
  // (`runStage2` is unaffected: there a true flag breaks its loop immediately.)
  lastStage2Stationary_ = false;
  // Within one `runStage2` call the topology is fixed (only edge lengths move), so
  // re-reading the edge list here is free; in the combined drive (`run`) it is what
  // picks up the edges a stage-1 move just created or removed.
  const auto &edges = spacetime_->getEdgeList()->toVector();
  const std::size_t edgeCount = edges.size();
  Eigen::VectorXcd lengths(edgeCount);
  Eigen::VectorXcd squaredLengths(edgeCount);
  for (std::size_t edgeIndex = 0; edgeIndex < edgeCount; ++edgeIndex) {
    lengths(edgeIndex) = edges[edgeIndex]->getLength();
    squaredLengths(edgeIndex) = lengths(edgeIndex) * lengths(edgeIndex);
  }
  // The connection phase is the node's other edge field, and it relaxes on its
  // own coordinate: phi is not derived from l, so there is no square-root
  // branch to track and the trial is written directly.
  Eigen::VectorXcd phases(edgeCount);
  for (std::size_t edgeIndex = 0; edgeIndex < edgeCount; ++edgeIndex)
    phases(edgeIndex) = edges[edgeIndex]->getPhase();
  const auto restoreEdgeLengths = [&]() {
    for (std::size_t edgeIndex = 0; edgeIndex < edgeCount; ++edgeIndex) {
      edges[edgeIndex]->setLength(lengths(edgeIndex));
      edges[edgeIndex]->setPhase(phases(edgeIndex));
    }
  };
  const auto setSquaredLengths = [&](const Eigen::VectorXcd &trialSquared) {
    for (std::size_t edgeIndex = 0; edgeIndex < edgeCount; ++edgeIndex)
      edges[edgeIndex]->setLength(continuousSquareRoot(
          trialSquared(edgeIndex), lengths(edgeIndex)));
  };
  const auto setPhases = [&](const Eigen::VectorXcd &trialPhases) {
    for (std::size_t edgeIndex = 0; edgeIndex < edgeCount; ++edgeIndex)
      edges[edgeIndex]->setPhase(trialPhases(edgeIndex));
  };
  auto fullObjective = [&]() { return objectiveFor(spacetime_); };

  // Return the steepest-ascent displacement in the complex z plane for a real
  // scalar. Stage 2 subtracts it. This finite-difference path is reserved for
  // r_U, whose target/block composition has no one closed-form derivative yet.
  const auto scalarAscentDirection = [&](const auto &functional) {
    Eigen::VectorXcd ascent = Eigen::VectorXcd::Zero(edgeCount);
    const double relativeStep =
        std::cbrt(std::numeric_limits<double>::epsilon());
    for (std::size_t edgeIndex = 0; edgeIndex < edgeCount; ++edgeIndex) {
      const double coordinateStep =
          relativeStep * std::max(std::abs(squaredLengths(edgeIndex)), 1.0);
      const auto evaluateAt = [&](complexd value) {
        edges[edgeIndex]->setLength(continuousSquareRoot(
            value, lengths(edgeIndex)));
        return functional();
      };
      const double realPlus =
          evaluateAt(squaredLengths(edgeIndex) + coordinateStep);
      const double realMinus =
          evaluateAt(squaredLengths(edgeIndex) - coordinateStep);
      double imaginaryDerivative = 0.0;
      if (!realSquaredLengthsOnly_) {
        const double imaginaryPlus =
            evaluateAt(squaredLengths(edgeIndex) +
                       complexd{0.0, coordinateStep});
        const double imaginaryMinus =
            evaluateAt(squaredLengths(edgeIndex) -
                       complexd{0.0, coordinateStep});
        imaginaryDerivative =
            (imaginaryPlus - imaginaryMinus) / (2.0 * coordinateStep);
      }
      edges[edgeIndex]->setLength(lengths(edgeIndex));
      ascent(edgeIndex) = complexd{
          (realPlus - realMinus) / (2.0 * coordinateStep),
          imaginaryDerivative};
    }
    return ascent;
  };

  // Explicit fixed-hole constraints already have an exact analytic r_U
  // gradient. When they are the whole residual and the run stays on real l^2,
  // use it instead of evaluating r_U twice per edge. Mixed or complex-locus
  // objectives retain the general numerical path.
  const bool explicitConstraintsAreWholeResidual =
      !registerConstraints_.empty() && inputTargets_.empty() &&
      outputTargets_.empty() && inputBlocks_.empty() && outputBlocks_.empty();
  const auto explicitConstraintAscentDirection = [&]() {
    Eigen::VectorXcd ascent = Eigen::VectorXcd::Zero(edgeCount);
    std::map<std::pair<std::uint64_t, std::uint64_t>, std::size_t> edgeIndices;
    for (std::size_t edgeIndex = 0; edgeIndex < edgeCount; ++edgeIndex)
      edgeIndices[edgeKey(edges[edgeIndex])] = edgeIndex;
    const auto oneCells =
        ChainComplex::fromSpacetime(*spacetime_).kSimplexVertices(1);
    for (const auto &constraint : registerConstraints_) {
      EigenstateSynthesis synthesis(spacetime_, constraint.degree, metricSource_);
      const std::vector<double> gradient = synthesis.residualForPeriodsGradient(
          constraint.holes, constraint.target);
      if (gradient.size() != oneCells.size())
        throw std::runtime_error(
            "MultiCobordism: explicit r_U gradient has wrong edge count");
      for (std::size_t i = 0; i < oneCells.size(); ++i) {
        const auto &cell = oneCells[i];
        if (cell.size() != 2)
          continue;
        const auto found = edgeIndices.find(
            {std::min(cell[0], cell[1]), std::max(cell[0], cell[1])});
        if (found != edgeIndices.end())
          ascent[static_cast<Eigen::Index>(found->second)] +=
              complexd{gradient[i], 0.0};
      }
    }
    return ascent;
  };

  Eigen::VectorXcd descentDirection = Eigen::VectorXcd::Zero(edgeCount);
  // The phase's own descent direction, empty unless the injected objective
  // declares a phi dependence. Kept separate from `descentDirection` because
  // the two are displacements in different coordinates — z and phi are distinct
  // fields, and mixing them is the error the two-field split exists to prevent.
  Eigen::VectorXcd phaseDescentDirection = Eigen::VectorXcd::Zero(edgeCount);
  // Exact acceptance baseline at the current state rather than
  // objectiveTrace.back(). Joint mode assembles it from the exact gradients
  // already needed for its direction; the other modes recompute their scalar.
  // In the combined drive (`run`) the trace is accumulated from stage-1 deltas
  // and can drift from the true objective, so it is not a safe line-search
  // gate.

  double currentObjective = 0.0;
  double trialStepScale = stepScale;
  bool objectiveImproved = false;
  try {
    // The engine does not know which functional it is scoring. It assembles
    // the firewalled context; the injected objective supplies its own analytic
    // direction and, where it has assembled its scalar along the way, the exact
    // baseline the line search gates on. Any numerically differentiated
    // register-residual term is applied here by weight rather than inside the
    // objective: differencing a scalar over edge coordinates is engine
    // machinery, and handing an objective a callable that did it would mean
    // handing it a closure over this node.
    ObjectiveDirectionContext directionContext;
    directionContext.scalar = objectiveContextFor(spacetime_);
    directionContext.edgeCount = edgeCount;
    if (carriedStateEnergyWeight_ != 0.0)
      directionContext.carriedStateEnergyGradient =
          carriedStateEnergyGradient(spacetime_);
    const auto objectiveDirection = objectiveSpec_->direction(directionContext);
    descentDirection += objectiveDirection.ascent;
    if (objectiveDirection.phaseAscent.size() ==
        static_cast<Eigen::Index>(edgeCount))
      phaseDescentDirection += objectiveDirection.phaseAscent;
    if (objectiveDirection.baselineComputed)
      currentObjective = objectiveDirection.baseline;
    const double numericalResidualWeight =
        objectiveSpec_->numericalRegisterResidualWeight(
            directionContext.scalar);
    if (numericalResidualWeight != 0.0) {
      if (useFiberResiduals_ && readoutsHaveAnalyticGradient()) {
        // Every fiber-mode term of rU has an analytic gradient through
        // the band's Riesz projector and the frame transfer; the numerical
        // path is not used for them. Only while every selected reading has its
        // own analytic gradient, though; otherwise this would be the
        // direction of a different objective.
        const ResidualGradient analytic = fiberModeAscent();
        descentDirection += numericalResidualWeight * analytic.lengths;
        if (fiberPhaseDescent_ && analytic.phases.size() == static_cast<Eigen::Index>(edgeCount))
          phaseDescentDirection += numericalResidualWeight * analytic.phases;
      } else if (explicitConstraintsAreWholeResidual && realSquaredLengthsOnly_)
        descentDirection += numericalResidualWeight *
                            explicitConstraintAscentDirection();
      else
        descentDirection += numericalResidualWeight *
                            scalarAscentDirection(
                                [&]() { return rU(spacetime_); });
    }

    // The pinned-region objective's direction adds on top of the bulk's, the
    // same way its terms add on top of the bulk's terms. Its own declared scope
    // decides which coordinates it moves; the bulk keeps moving all of them.
    if (pinnedObjectiveSpec_) {
      ObjectiveDirectionContext pinnedContext;
      pinnedContext.scalar =
          objectiveContextFor(spacetime_, pinnedObjectiveSpec_);
      pinnedContext.edgeCount = edgeCount;
      if (carriedStateEnergyWeight_ != 0.0)
        pinnedContext.carriedStateEnergyGradient =
            directionContext.carriedStateEnergyGradient;
      const auto pinnedDirection =
          pinnedObjectiveSpec_->direction(pinnedContext);
      descentDirection += pinnedDirection.ascent;
      // The line search gates on the exact composite scalar, so a baseline
      // assembled from the bulk alone would understate it. Only the sum of both
      // objectives is the number being descended.
      if (objectiveDirection.baselineComputed &&
          pinnedDirection.baselineComputed)
        currentObjective = objectiveDirection.baseline + pinnedDirection.baseline;
      else if (objectiveDirection.baselineComputed)
        currentObjective = objectiveFor(spacetime_);
      const double pinnedNumericalWeight =
          pinnedObjectiveSpec_->numericalRegisterResidualWeight(
              pinnedContext.scalar);
      if (pinnedNumericalWeight != 0.0) {
        if (explicitConstraintsAreWholeResidual && realSquaredLengthsOnly_)
          descentDirection += pinnedNumericalWeight *
                              explicitConstraintAscentDirection();
        else
          descentDirection += pinnedNumericalWeight *
                              scalarAscentDirection(
                                  [&]() { return rU(spacetime_); });
      }
    }

    // A real-locus run varies Re(l^2) only. Analytic objectives encode both
    // real derivatives in the complex direction, so remove the imaginary
    // coordinate after every contribution has been assembled.
    if (realSquaredLengthsOnly_)
      for (Eigen::Index i = 0; i < descentDirection.size(); ++i)
        descentDirection[i] = complexd{descentDirection[i].real(), 0.0};

    restoreEdgeLengths();
    // Pinning enters here and only here: a pinned edge keeps its resident squared
    // length while the rest of the complex relaxes around it. Zeroing the descent
    // component is the whole mechanism — no clamp, no projection after the fact,
    // no special case in the line search, which then simply has no reason to move
    // the coordinate. With no region declared this loop does nothing.
    if (!pinnedRegions_.empty())
      for (std::size_t edgeIndex = 0; edgeIndex < edgeCount; ++edgeIndex) {
        const auto key = edges[edgeIndex]->getKey();
        if (edgeIsPinned(key.first, key.second)) {
          descentDirection(edgeIndex) = complexd{0.0, 0.0};
          // A pinned edge is held in both its fields. "Do not change these"
          // that froze the length while the phase drifted would be a pin in
          // name only.
          phaseDescentDirection(edgeIndex) = complexd{0.0, 0.0};
        }
      }
    // No coordinate can move, so every backtracking trial would evaluate the
    // unchanged global objective and fail the strict-improvement gate. This is
    // common at exact stationary points, when every selected term is disabled,
    // and when every edge that could move is pinned.
    if (descentDirection.squaredNorm() == 0.0 &&
        phaseDescentDirection.squaredNorm() == 0.0) {
      lastStage2Stationary_ = true;
      return false;
    }
    // An objective whose direction assembly already produced its scalar handed
    // it back as the direction's baseline, taken above. Only evaluate the
    // functional again when it did not, rather than re-deriving the same
    // gradients at the unchanged base geometry.
    if (!objectiveDirection.baselineComputed)
      currentObjective = fullObjective();
    // Absolute improvement threshold: the same tolerance has the same meaning
    // at every objective scale.
    const double improvementThreshold = tolerance;
    for (int lineSearchIndex = 0; lineSearchIndex < 24; ++lineSearchIndex) {
      // This is the key coordinate correction: derivatives are with respect to
      // z=l^2, so subtract the direction from z, then map back to Edge's stored
      // l on the continuous square-root branch. No component of z is projected.
      setSquaredLengths(squaredLengths -
                        trialStepScale * descentDirection);
      // One line search over both fields: the same step scale moves z and phi
      // together and the same strict-improvement gate accepts or rejects the
      // pair. Two searches would let one field buy an improvement the other
      // paid for, and the accepted state would not be a descent of the whole
      // objective.
      setPhases(phases - trialStepScale * phaseDescentDirection);
      if (!geometryAdmissible(spacetime_)) {
        // Outside the closure of the allowable domain: not a configuration,
        // so it is not scored; the step is shortened exactly as a non-improving
        // trial is.
        CLOG(INFO_LEVEL, "Trial geometry not Kontsevich-Segal admissible; shortening the step.");
        trialStepScale *= 0.5;
        continue;
      }
      const double trialObjective = fullObjective();
      CLOG(INFO_LEVEL, "-----------------------------------");
      CLOG(INFO_LEVEL, "Trial objective: ", trialObjective);
      CLOG(INFO_LEVEL, "Current objective: ", currentObjective);
      CLOG(INFO_LEVEL, "Improvement threshold: ", improvementThreshold);
      CLOG(INFO_LEVEL, "Improvement: ", currentObjective - trialObjective);
      CLOG(INFO_LEVEL, "-----------------------------------");
      if ((currentObjective - trialObjective) >= improvementThreshold) {
        CLOG(INFO_LEVEL, (currentObjective - trialObjective), "<=", improvementThreshold);
        CLOG(INFO_LEVEL, "Improved.");
        objectiveTrace.push_back(trialObjective);
        stepScale = std::min(stepScale * 1.3, 1.0);
        objectiveImproved = true;
        // Solver-error indicator: the magnitude of the improvement this
        // geometric update actually banked (0 once the relaxation is
        // stationary). A base numerical quantity — nothing derived.
        lastStage2Improvement_ = std::abs(currentObjective - trialObjective);
        currentObjective = trialObjective;
        break;
      }
      CLOG(INFO_LEVEL, (currentObjective - trialObjective), ">", improvementThreshold);
      CLOG(INFO_LEVEL, "Did not improve.");
      trialStepScale *= 0.5;
    }
  } catch (...) {
    // The error still propagates; it just does not take the geometry with it.
    // The throw comes from a trial the line search had not accepted, so the
    // complex the caller still holds must be the one it had on entry, not a
    // half-applied step everything downstream would then read.
    restoreEdgeLengths();
    throw;
  }
  if (!objectiveImproved) {
    restoreEdgeLengths();
    lastStage2Stationary_ = true;
    lastStage2Improvement_ = 0.0;  // stationary means zero solver error
    return false;
  }
  (void)beta;  // run/runStage2 synchronize this with reggeWeight_ before entry.
  return true;
}

int MultiCobordism::directedConeOut(HolePlacementStrategy strategy, int maxOpen) {
  if (registerDegrees_.empty()) return 0;
  constexpr int kMaxCandidates = 40;  // bound the scan; interior-first surfaces openers early
  constexpr int kProbeOpeners = 3;    // stop once a few openers are in hand
  const int registerDegree = registerDegrees_.front();
  auto spacetime = spacetime_;
  int opened = 0;
  for (int iteration = 0; iteration < maxOpen; ++iteration) {
    const auto holesBefore = emergentHoles(*spacetime, registerDegree);
    const std::size_t holeCountBefore = holesBefore.size();
    std::set<std::uint64_t> holeVertices;
    for (const auto &hole : holesBefore) holeVertices.insert(hole.begin(), hole.end());
    const auto boundary = boundaryFacetSet(*spacetime);

    std::vector<std::vector<std::uint64_t>> cells;
    for (const auto *simplex : spacetime->getTopSimplices())
      cells.push_back(simplex->topTuple());
    // Order interior-first (fewest boundary facets → hole-creators first); the secondary key
    // then places cells sharing vertices with the existing holes last (AdjacentHolesLast, a
    // separated register) or first (AdjacentHolesFirst, a clustered one).
    const auto orderKey = [&](const std::vector<std::uint64_t> &cell) {
      int boundaryFacets = 0;
      for (std::size_t i = 0; i < cell.size(); ++i) {
        std::vector<std::uint64_t> facet;
        for (std::size_t j = 0; j < cell.size(); ++j)
          if (j != i) facet.push_back(cell[j]);
        if (boundary.count(facet)) ++boundaryFacets;
      }
      int shared = 0;
      for (auto vertexId : cell)
        if (holeVertices.count(vertexId)) ++shared;
      return std::pair<int, int>(
          boundaryFacets,
          strategy == HolePlacementStrategy::AdjacentHolesFirst ? -shared : shared);
    };
    std::sort(cells.begin(), cells.end(),
              [&](const std::vector<std::uint64_t> &a,
                  const std::vector<std::uint64_t> &b) { return orderKey(a) < orderKey(b); });

    // Scored by the injected objective, so topology changes when the functional
    // in force wants it and not otherwise. Surgery is the only topology-changing
    // mechanism the engine has — Pachner moves are bistellar and preserve the PL
    // homeomorphism type, hence the Betti numbers, and geometric relaxation
    // changes no topology at all — so this probe is where a higher b_k becomes
    // reachable. It is never required: an objective indifferent to topology
    // finds no candidate that lowers it and nothing is committed.
    const double baseObjective = objectiveFor(spacetime);
    double bestObjective = baseObjective;
    std::vector<std::uint64_t> bestCell;
    int candidatesScanned = 0;
    int openersScanned = 0;
    SurgicalCone cone(spacetime.get());
    for (const auto &cell : cells) {
      if (candidatesScanned++ >= kMaxCandidates) break;
      // `coneOut` is itself gated on the full manifold check, so a candidate that
      // survives it leaves a valid manifold-with-boundary — including when it
      // removed a pinned vertex, which is a legitimate topology change.
      if (!cone.coneOut(cell).first) continue;  // gate rejected; nothing applied
      const bool opensHole =
          emergentHoles(*spacetime, registerDegree).size() > holeCountBefore;
      if (opensHole) {
        const double candidateObjective = objectiveFor(spacetime);
        if (candidateObjective < bestObjective) {
          bestObjective = candidateObjective;
          bestCell = cell;
        }
        ++openersScanned;
      }
      cone.rollback();
      if (opensHole && openersScanned >= kProbeOpeners) break;
    }
    if (bestCell.empty()) break;  // no opener lowers the objective
    if (!cone.coneOut(bestCell).first) break;
    ++opened;
  }
  return opened;
}

int MultiCobordism::randomConeOut(int count) {
  if (count <= 0 || !spacetime_) return 0;
  // Every vertex any declared region holds. A cell touching one is not a
  // candidate: `directedConeOut` may remove a pinned vertex when the result is
  // still a manifold, but a perturbation that eats the boundary a run asked to
  // hold is not the run that was asked for.
  std::set<std::uint64_t> held;
  for (const auto &region : pinnedRegions_)
    held.insert(region.vertices.begin(), region.vertices.end());

  std::vector<std::vector<std::uint64_t>> cells;
  for (const auto *simplex : spacetime_->getTopSimplices()) {
    auto cell = simplex->topTuple();
    bool touchesHeld = false;
    for (const auto vertexId : cell)
      if (held.count(vertexId)) { touchesHeld = true; break; }
    if (!touchesHeld) cells.push_back(std::move(cell));
  }
  // Uniform: shuffled whole and taken in that order, so no property of a cell
  // -- its position, its adjacency, what it would do to rU -- makes it more or
  // less likely to be chosen. Shuffling once and walking it is the same
  // distribution as drawing without replacement, and needs one pass.
  std::shuffle(cells.begin(), cells.end(), randomNumberGenerator_);

  SurgicalCone cone(spacetime_.get());
  int removed = 0;
  for (const auto &cell : cells) {
    if (removed >= count) break;
    // Gated exactly as every other cone-out is. A rejection is not an error
    // here: the candidate simply was not removable, and the next is tried.
    if (!cone.coneOut(cell).first) continue;
    ++removed;
  }
  return removed;
}

int MultiCobordism::directedConeIn(int maxClose) {
  if (registerDegrees_.empty()) return 0;
  constexpr int kMaxCandidates = 40;
  const int registerDegree = registerDegrees_.front();
  auto spacetime = spacetime_;
  int closed = 0;
  for (int iteration = 0; iteration < maxClose; ++iteration) {
    const auto holesBefore = emergentHoles(*spacetime, registerDegree);
    const std::size_t holeCountBefore = holesBefore.size();
    if (holeCountBefore == 0) break;
    const auto boundary = boundaryFacetSet(*spacetime);

    // Cap facets: drop-one facets of the current holes that lie on the boundary — capping
    // one (a cone-in over it) closes that hole.
    std::set<std::vector<std::uint64_t>> seen;
    std::vector<std::vector<std::uint64_t>> capFacets;
    for (const auto &hole : holesBefore) {
      for (std::size_t i = 0; i < hole.size(); ++i) {
        std::vector<std::uint64_t> facet;
        for (std::size_t j = 0; j < hole.size(); ++j)
          if (j != i) facet.push_back(hole[j]);
        std::sort(facet.begin(), facet.end());
        if (boundary.count(facet) && seen.insert(facet).second) capFacets.push_back(facet);
      }
    }

    // Scored by the injected objective, exactly as the cone-out probe is: a
    // hole closes when the functional in force is lowered by closing it.
    const double baseObjective = objectiveFor(spacetime);
    double bestObjective = baseObjective;
    std::vector<std::uint64_t> bestFacet;
    int candidatesScanned = 0;
    SurgicalCone cone(spacetime.get());
    for (const auto &facet : capFacets) {
      if (candidatesScanned++ >= kMaxCandidates) break;
      if (!cone.coneIn(facet).first) continue;
      if (emergentHoles(*spacetime, registerDegree).size() < holeCountBefore) {
        const double candidateObjective = objectiveFor(spacetime);
        if (candidateObjective < bestObjective) {
          bestObjective = candidateObjective;
          bestFacet = facet;
        }
      }
      cone.rollback();
    }
    if (bestFacet.empty()) break;  // no cap lowers the objective
    if (!cone.coneIn(bestFacet).first) break;
    ++closed;
  }
  return closed;
}

void MultiCobordism::buildStep(BuildAction action, int maxSteps, int nCandidateMoves,
                               double stage2Beta, int stage2MaxIters,
                               double stage2Alpha0,
                               HolePlacementStrategy holePlacementStrategy) {
  switch (action) {
    case BuildAction::Grow:
      runStage1(maxSteps, nCandidateMoves, /*growBoundaries=*/true);
      break;
    case BuildAction::Evolve:
      runStage1(maxSteps, nCandidateMoves, /*growBoundaries=*/false);
      break;
    case BuildAction::Relax:
      runStage2(stage2Beta, stage2MaxIters, stage2Alpha0);
      break;
    case BuildAction::ConeOut:
      (void)directedConeOut(holePlacementStrategy);
      break;
    case BuildAction::ConeIn:
      (void)directedConeIn();
      break;
  }
}

}  // namespace tessera::cobordism
