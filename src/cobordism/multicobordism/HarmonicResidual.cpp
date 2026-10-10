// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

// MultiCobordism (include/cobordism/MultiCobordism.h): the harmonic target
// residuals: Betti numbers, emergent holes, hole period matrices, target
// relabeling and the boundary-block residuals. One of the translation units
// that define the class's members by responsibility
// (https://github.com/akellehe/tessera/issues/1481).

#include "Internal.h"

namespace tessera::cobordism {

std::vector<int> MultiCobordism::betti(const Spacetime &spacetime) {
  // Betti numbers are purely combinatorial, and the residual path calls this on
  // every objective evaluation while only edge lengths move, so Smith normal
  // form dominates the profile. The spacetime's structural revision shows when
  // the last computation is still exact.
  if (const auto *cached = spacetime.cachedBettiNumbers()) return *cached;
  auto numbers = ChainComplex::fromSpacetime(spacetime).bettiNumbers();
  spacetime.storeBettiNumbers(numbers);
  return numbers;
}

std::vector<std::vector<std::uint64_t>> MultiCobordism::emergentHoles(
    const Spacetime &spacetime, int registerDegree) {
  // The (k+2)-vertex tuples all of whose drop-one facets are boundary facets.
  std::set<std::vector<std::uint64_t>> boundaryFacets;
  for (auto boundaryFacet : spacetime.getBoundary()) {
    std::sort(boundaryFacet.begin(), boundaryFacet.end());
    boundaryFacets.insert(std::move(boundaryFacet));
  }
  std::vector<std::vector<std::uint64_t>> emergentHoleTuples;
  if (boundaryFacets.empty() ||
      static_cast<int>(boundaryFacets.begin()->size()) !=
          registerDegree + 1)  // facets must be k-cells
    return emergentHoleTuples;
  std::set<std::uint64_t> boundaryVertexIds;
  for (const auto &boundaryFacet : boundaryFacets)
    for (auto vertexId : boundaryFacet) boundaryVertexIds.insert(vertexId);
  std::set<std::vector<std::uint64_t>> emergentHoleSet;
  for (const auto &boundaryFacet : boundaryFacets) {
    for (auto candidateVertexId : boundaryVertexIds) {
      if (std::find(boundaryFacet.begin(), boundaryFacet.end(),
                    candidateVertexId) != boundaryFacet.end())
        continue;
      std::vector<std::uint64_t> candidateHole = boundaryFacet;
      candidateHole.push_back(candidateVertexId);
      std::sort(candidateHole.begin(), candidateHole.end());
      bool allFacetsAreBoundary = true;
      for (std::size_t droppedIndex = 0; droppedIndex < candidateHole.size();
           ++droppedIndex) {
        std::vector<std::uint64_t> dropOneFacet;
        for (std::size_t copyIndex = 0; copyIndex < candidateHole.size();
             ++copyIndex)
          if (copyIndex != droppedIndex)
            dropOneFacet.push_back(candidateHole[copyIndex]);
        if (!boundaryFacets.count(dropOneFacet)) {
          allFacetsAreBoundary = false;
          break;
        }
      }
      if (allFacetsAreBoundary) emergentHoleSet.insert(candidateHole);
    }
  }
  emergentHoleTuples.assign(emergentHoleSet.begin(), emergentHoleSet.end());
  return emergentHoleTuples;
}

double MultiCobordism::reggeActionGradient(
    const std::shared_ptr<Spacetime> &spacetime) {
  ReggeSolver reggeSolver(spacetime, MatterConfiguration());
  double squaredGradientNorm = 0.0;
  for (const auto &gradientComponent : reggeSolver.actionGradientExact())
    squaredGradientNorm += std::norm(gradientComponent);
  return squaredGradientNorm;
}

Eigen::VectorXcd MultiCobordism::targetStateVector(
    const std::vector<complexd> &targetState) {
  Eigen::VectorXcd targetVector(targetState.size());
  for (std::size_t componentIndex = 0; componentIndex < targetState.size();
       ++componentIndex)
    targetVector(componentIndex) = targetState[componentIndex];
  return targetVector;
}

std::vector<std::vector<std::uint64_t>> MultiCobordism::holesCarryingTheTarget(
    const Spacetime &spacetime, int registerDegree, std::size_t targetDimension) {
  auto emergentHoleTuples = emergentHoles(spacetime, registerDegree);
  // One target component per hole: holes beyond the target's width have no component
  // to carry and take no part in the fit.
  if (emergentHoleTuples.size() > targetDimension)
    emergentHoleTuples.resize(targetDimension);
  return emergentHoleTuples;
}

Eigen::MatrixXcd MultiCobordism::holePeriodMatrix(
    const std::shared_ptr<Spacetime> &spacetime, int registerDegree,
    int degreeBettiNumber,
    const std::vector<std::vector<std::uint64_t>> &cycleHoles,
    std::size_t targetDimension, HodgeLaplacian::MetricSource metricSource) {
  EigenstateSynthesis eigenstateSynthesis(spacetime, registerDegree, metricSource);
  std::vector<complexd> flattenedCyclePeriods;  // rank x m, row-major
  try {
    flattenedCyclePeriods = eigenstateSynthesis.cyclePeriods(cycleHoles);
  } catch (const std::exception &) {
    // An operator this geometry cannot assemble carries no usable harmonic:
    // the zero-column matrix below, which the caller reads as the full leak.
    return Eigen::MatrixXcd::Zero(static_cast<int>(targetDimension), 0);
  }
  const std::size_t holeCount = cycleHoles.size();
  // The row count of the flattened periods is the numeric harmonic-kernel
  // dimension the synthesizer computed (HodgeLaplacian::harmonicMatrix at its
  // metric-dependent rank threshold), which is not necessarily the integer
  // Betti number: on geometrically extreme complexes, such as deep-lookahead
  // candidates near the null-face locus, the numeric rank can fall below the
  // topological one, and indexing by the Betti count would read past the end of
  // the vector. Bound every index by the data's own shape instead; fewer usable
  // harmonics means a larger residual, and a zero-column matrix reads as the
  // full leak in the caller.
  const std::size_t periodRowCount =
      holeCount == 0 ? 0 : flattenedCyclePeriods.size() / holeCount;
  const int harmonicRank =
      std::min(degreeBettiNumber, static_cast<int>(periodRowCount));
  Eigen::MatrixXcd periodMatrixTransposed = Eigen::MatrixXcd::Zero(
      static_cast<int>(targetDimension), std::max(harmonicRank, 0));
  for (int harmonicIndex = 0; harmonicIndex < harmonicRank; ++harmonicIndex)
    for (std::size_t holeIndex = 0; holeIndex < holeCount; ++holeIndex)
      periodMatrixTransposed(static_cast<int>(holeIndex), harmonicIndex) =
          flattenedCyclePeriods[static_cast<std::size_t>(harmonicIndex) * holeCount +
                                holeIndex];
  return periodMatrixTransposed;
}

Eigen::VectorXcd MultiCobordism::relabeledTargetVector(
    const Eigen::VectorXcd &targetVector, const std::vector<int> &relabeling) {
  Eigen::VectorXcd relabeled(targetVector.size());
  for (std::size_t holeIndex = 0; holeIndex < relabeling.size(); ++holeIndex)
    relabeled(holeIndex) = targetVector(relabeling[holeIndex]);
  return relabeled;
}

MultiCobordism::RelabelingMatch MultiCobordism::bestRelabelingOfTarget(
    const Eigen::MatrixXcd &periodMatrixTransposed,
    const Eigen::VectorXcd &targetVector,
    const std::set<std::vector<int>> &claimedMatchings, bool skipClaimed) {
  // Minimum over the relabelings of the target components of
  // ||pdT c - ts||^2, with c the least-squares solution. Total over every
  // configuration: a non-finite period matrix — an unbounded stage-2 trial can
  // overflow the polynomial cell weights, taking the harmonic periods out of
  // double range — scores +inf, an infinitely bad configuration the line search
  // rejects, instead of handing non-finite input to BDCSVD, whose compute and
  // solve are undefined behavior with asserts compiled out.
  if (!periodMatrixTransposed.allFinite()) {
    std::vector<int> identityRelabeling(
        static_cast<std::size_t>(targetVector.size()));
    std::iota(identityRelabeling.begin(), identityRelabeling.end(), 0);
    return {std::numeric_limits<double>::infinity(), identityRelabeling, true};
  }
  Eigen::BDCSVD<Eigen::MatrixXcd> periodSvd(
      periodMatrixTransposed, Eigen::ComputeThinU | Eigen::ComputeThinV);
  RelabelingMatch bestMatch;
  std::vector<int> relabeling(static_cast<std::size_t>(targetVector.size()));
  std::iota(relabeling.begin(), relabeling.end(), 0);
  do {
    if (skipClaimed && claimedMatchings.count(relabeling)) continue;
    const Eigen::VectorXcd relabeledTarget =
        relabeledTargetVector(targetVector, relabeling);
    const Eigen::VectorXcd leastSquaresCoefficients =
        periodSvd.solve(relabeledTarget);
    const double residual =
        (periodMatrixTransposed * leastSquaresCoefficients - relabeledTarget)
            .squaredNorm();
    if (!bestMatch.scored || residual < bestMatch.residual)
      bestMatch = {residual, relabeling, true};
  } while (std::next_permutation(relabeling.begin(), relabeling.end()));
  return bestMatch;
}

double MultiCobordism::residualOfTargetStateAgainstHarmonic(
    const std::shared_ptr<Spacetime> &spacetime, int registerDegree,
    const std::vector<complexd> &targetState,
    HodgeLaplacian::MetricSource metricSource) {
  // No other register to collide with: an empty claim set excludes nothing, so
  // this is the unconstrained minimum over the relabelings (`r_state`).
  std::set<std::vector<int>> claimedMatchings;
  return residualOfTargetStateAgainstHarmonicWithDistinctMatching(
      spacetime, registerDegree, targetState, claimedMatchings, metricSource);
}

double MultiCobordism::residualOfTargetStateAgainstHarmonicWithDistinctMatching(
    const std::shared_ptr<Spacetime> &spacetime, int registerDegree,
    const std::vector<complexd> &targetState,
    std::set<std::vector<int>> &claimedMatchings,
    HodgeLaplacian::MetricSource metricSource) {
  const Eigen::VectorXcd targetVector = targetStateVector(targetState);
  const double fullLeakResidual = targetVector.squaredNorm();  // zero-filled leak

  const auto bettiNumbers = betti(*spacetime);
  if (registerDegree < 0) //||  # TODO: why would we exit if registerDegree is higher than betti numbers?
    return fullLeakResidual;
  if (registerDegree >= static_cast<int>(bettiNumbers.size())) {
    CLOG(WARN_LEVEL, "register degree was higher than bettiNumbers!");
    return fullLeakResidual;
  }
  const int degreeBettiNumber = bettiNumbers[registerDegree];
  if (degreeBettiNumber == 0) return fullLeakResidual;

  const auto cycleHoles =
      holesCarryingTheTarget(*spacetime, registerDegree, targetState.size());
  if (cycleHoles.empty()) return fullLeakResidual;
  const Eigen::MatrixXcd periodMatrixTransposed =
      holePeriodMatrix(spacetime, registerDegree, degreeBettiNumber, cycleHoles,
                       targetState.size(), metricSource);
  // The matrix is bounded by the numeric harmonic rank (see holePeriodMatrix):
  // Zero usable harmonics on a geometrically extreme candidate means the
  // register carries nothing, the full leak, rather than an SVD of a 0-column
  // matrix.
  if (periodMatrixTransposed.cols() == 0) return fullLeakResidual;

  // The relabeling this register wins is withheld from the registers scored
  // after it, so no two are read against the same matching of components onto
  // holes.
  RelabelingMatch match = bestRelabelingOfTarget(
      periodMatrixTransposed, targetVector, claimedMatchings, /*skipClaimed=*/true);
  if (!match.scored) {
    // Every relabeling is already claimed: more registers than the d! this
    // target admits. Restart the exclusion rather than return an empty minimum.
    claimedMatchings.clear();
    match = bestRelabelingOfTarget(periodMatrixTransposed, targetVector,
                                   claimedMatchings, /*skipClaimed=*/false);
  }
  claimedMatchings.insert(match.relabeling);
  return match.residual;
}

double MultiCobordism::residualForBoundaryBlock(
    const BoundaryBlock &boundaryBlock,
    const std::shared_ptr<Spacetime> &spacetime) const {
  std::set<std::vector<int>> claimedMatchings;
  return residualForBoundaryBlockWithDistinctMatchings(boundaryBlock, spacetime,
                                                       claimedMatchings);
}

double MultiCobordism::residualForBoundaryBlockWithDistinctMatchings(
    const BoundaryBlock &boundaryBlock,
    const std::shared_ptr<Spacetime> &spacetime,
    std::set<std::vector<int>> &claimedMatchings) const {
  // A marked block (setInputMarking) is scored by the residual of its own
  // state: the leak of its input state in the holomorphic form of its own
  // Laplacian on its live surface. The zero mode of the entire cobordism is the
  // output state, reported by readInputState in the block's live frame.
  //
  // Under scoreWholeComplexLeak the whole's leak is added to that own residual
  // rather than replacing it, so the block's own-Laplacian residual stays in
  // the objective beside the bulk terms. The own term is identically zero on a
  // block whose edges cannot move, which is exactly when the whole's leak is
  // the only thing left to say how the block sits in the cobordism.
  if (useFiberResiduals_ && boundaryBlock.marking) {
    const double own = ownStateResidualOn(boundaryBlock, spacetime);
    if (!scoreWholeComplexLeak_) return own;
    return own + inputStateResidualOn(boundaryBlock, spacetime);
  }
  if (useFiberResiduals_ && boundaryBlock.fiber && boundaryBlock.fiber->images.cols() > 0)
    return fiberResidualForBoundaryBlock(boundaryBlock, spacetime);
  auto blockSubcomplex = spacetime->subcomplexWithinVertexSet(
    boundaryBlock.vertices);
  double residual = 0.0;
  if (!blockSubcomplex)  // no complex to read: the target leaks in full, per degree
    return static_cast<double>(registerDegrees_.size()) *
           targetStateVector(boundaryBlock.target).squaredNorm();
  // The sub-complex is a fresh spacetime whose per-instance Betti slot is
  // empty, so without help every evaluation would re-run the Smith normal form
  // once per block, per line-search trial, per candidate. The block's topology
  // is a pure function of the parent's cells and the vertex set, so the parent
  // caches the numbers per (structural revision, region fingerprint): on a hit,
  // pre-seed the child's slot so betti() below never computes; on a miss, store
  // the child's freshly computed numbers back on the parent.
  //
  // The region is named by `Fingerprint::fingerprintOf` over its vertex ids,
  // called as a static because a `Fingerprint` instance holds only `kMax`
  // identifiers and drops the rest, while a block region grows across the
  // complex.
  const std::uint64_t vertexSetKey =
      ::tessera::mesh::Fingerprint::fingerprintOf(boundaryBlock.vertices);
  if (const auto *cached =
          spacetime->cachedSubcomplexBettiNumbers(vertexSetKey))
    blockSubcomplex->storeBettiNumbers(*cached);
  for (int registerDegree : registerDegrees_)
    residual += residualOfTargetStateAgainstHarmonicWithDistinctMatching(
        blockSubcomplex, registerDegree, boundaryBlock.target, claimedMatchings);
  if (const auto *computed = blockSubcomplex->cachedBettiNumbers())
    spacetime->storeSubcomplexBettiNumbers(vertexSetKey, *computed);
  return residual;
}

}  // namespace tessera::cobordism
