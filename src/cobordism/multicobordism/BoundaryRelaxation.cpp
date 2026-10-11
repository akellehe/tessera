// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

// MultiCobordism (include/cobordism/MultiCobordism.h): the boundary
// relaxations: the fixed-boundary eigenstate, boundary-state pairs and
// whole-complex readout targets. One of the translation units that define
// the class's members by responsibility
// (https://github.com/akellehe/tessera/issues/1481).

#include "Internal.h"

namespace tessera::cobordism {

FixedBoundaryEigenstateResult MultiCobordism::relaxFixedBoundaryEigenstate(
    int degree, std::vector<std::vector<std::uint64_t>> supportCells,
    std::vector<complexd> target, double epsilon, int restarts, int maxGrowth,
    std::uint64_t seed, int maxIterations) {
  return EigenstateSynthesis::relaxFixedBoundaryEigenstate(
      spacetime_, metricSource_, degree, std::move(supportCells), std::move(target), epsilon,
      restarts, maxGrowth, seed, maxIterations);
}

BoundaryStateTransferResult MultiCobordism::relaxBoundaryStatePairs(
    int degree, std::string inputRegionName,
    std::vector<std::vector<std::uint64_t>> inputCells,
    std::vector<std::vector<complexd>> inputStates,
    std::string outputRegionName,
    std::vector<std::vector<std::uint64_t>> outputCells,
    std::vector<std::vector<complexd>> outputStates, bool commonEigenvalue,
    double epsilon, double boundaryEpsilon, int restarts, int maxGrowth,
    std::uint64_t seed, int maxIterations) {
  return EigenstateSynthesis::relaxBoundaryStatePairs(
      spacetime_, metricSource_, pinnedRegions_, degree, std::move(inputRegionName),
      std::move(inputCells), std::move(inputStates), std::move(outputRegionName),
      std::move(outputCells), std::move(outputStates), commonEigenvalue, epsilon,
      boundaryEpsilon, restarts, maxGrowth, seed, maxIterations);
}

WholeComplexReadoutResult MultiCobordism::relaxWholeComplexReadoutTargets(
    int degree, std::string regionAName,
    std::vector<std::vector<std::uint64_t>> cellsA,
    std::vector<std::vector<complexd>> statesA, std::string regionBName,
    std::vector<std::vector<std::uint64_t>> cellsB,
    std::vector<std::vector<complexd>> statesB,
    std::vector<ReadoutChain> readouts,
    std::vector<std::vector<complexd>> targets, bool commonEigenvalue,
    double epsilon, double boundaryEpsilon, int restarts, int maxGrowth,
    std::uint64_t seed, int maxIterations) {
  return EigenstateSynthesis::relaxWholeComplexReadoutTargets(
      spacetime_, metricSource_, pinnedRegions_, degree, std::move(regionAName),
      std::move(cellsA), std::move(statesA), std::move(regionBName), std::move(cellsB),
      std::move(statesB), std::move(readouts), std::move(targets), commonEigenvalue,
      epsilon, boundaryEpsilon, restarts, maxGrowth, seed, maxIterations);
}

}  // namespace tessera::cobordism
