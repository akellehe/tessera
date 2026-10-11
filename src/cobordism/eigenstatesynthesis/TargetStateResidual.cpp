// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

// EigenstateSynthesis (include/cobordism/EigenstateSynthesis.h): the target-
// state residual against the harmonic over the holes. One of the translation
// units that define the class's members by topic.

#include "cobordism/EigenstateSynthesis.h"
#include "quantum/ChoiJamiolkowski.h"

#include <numeric>
#include <limits>
#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "cobordism/ChainComplex.h"
#include "mesh/Edge.h"
#include "mesh/EdgeList.h"
#include "mesh/Fingerprint.h"
#include "mesh/Simplex.h"
#include "mesh/Vertex.h"
#include "mesh/VertexList.h"
#include "spacetime/Metric.h"
#include "spacetime/Signature.h"
#include "spacetime/Spacetime.h"
#include "spacetime/pachner/AddMove.h"

#ifdef TESSERA_CUDA
#include "cuda/eigenstate_cuda.h"
#endif

namespace tessera::cobordism {

using cd = std::complex<double>;

namespace {

using complexd = std::complex<double>;

// One register's winning relabeling: which target component each of its holes
// carries, and the least-squares residual min_c |P^T c - t|^2 that matching
// leaves. `scored` is false when every relabeling was skipped as already
// claimed, so nothing was evaluated.
struct RelabelingMatch {
  double residual = 0.0;
  std::vector<int> relabeling;
  bool scored = false;
};

Eigen::VectorXcd targetStateVector(
    const std::vector<complexd> &targetState) {
  Eigen::VectorXcd targetVector(targetState.size());
  for (std::size_t componentIndex = 0; componentIndex < targetState.size();
       ++componentIndex)
    targetVector(componentIndex) = targetState[componentIndex];
  return targetVector;
}

std::vector<std::vector<std::uint64_t>> holesCarryingTheTarget(
    const Spacetime &spacetime, int degree, std::size_t targetDimension) {
  auto emergentHoleTuples = spacetime.getHoles(degree);
  // One target component per hole: holes beyond the target's width have no component
  // to carry and take no part in the fit.
  if (emergentHoleTuples.size() > targetDimension)
    emergentHoleTuples.resize(targetDimension);
  return emergentHoleTuples;
}

Eigen::MatrixXcd holePeriodMatrix(
    const std::shared_ptr<Spacetime> &spacetime, int degree,
    int degreeBettiNumber,
    const std::vector<std::vector<std::uint64_t>> &cycleHoles,
    std::size_t targetDimension, HodgeLaplacian::MetricSource metricSource) {
  EigenstateSynthesis eigenstateSynthesis(spacetime, degree, metricSource);
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

Eigen::VectorXcd relabeledTargetVector(
    const Eigen::VectorXcd &targetVector, const std::vector<int> &relabeling) {
  Eigen::VectorXcd relabeled(targetVector.size());
  for (std::size_t holeIndex = 0; holeIndex < relabeling.size(); ++holeIndex)
    relabeled(holeIndex) = targetVector(relabeling[holeIndex]);
  return relabeled;
}

RelabelingMatch bestRelabelingOfTarget(
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

}  // namespace

double EigenstateSynthesis::targetStateResidual(
    const std::shared_ptr<Spacetime> &spacetime, int degree,
    const std::vector<complexd> &targetState,
    HodgeLaplacian::MetricSource metricSource) {
  // No other register to collide with: an empty claim set excludes nothing, so
  // this is the unconstrained minimum over the relabelings (`r_state`).
  std::set<std::vector<int>> claimedMatchings;
  return targetStateResidual(
      spacetime, degree, targetState, claimedMatchings, metricSource);
}

double EigenstateSynthesis::targetStateResidual(
    const std::shared_ptr<Spacetime> &spacetime, int degree,
    const std::vector<complexd> &targetState,
    std::set<std::vector<int>> &claimedMatchings,
    HodgeLaplacian::MetricSource metricSource) {
  const Eigen::VectorXcd targetVector = targetStateVector(targetState);
  const double fullLeakResidual = targetVector.squaredNorm();  // zero-filled leak

  const auto bettiNumbers = spacetime->getBettiNumbers();
  if (degree < 0) //||  # TODO: why would we exit if degree is higher than betti numbers?
    return fullLeakResidual;
  if (degree >= static_cast<int>(bettiNumbers.size())) {
    CLOG(WARN_LEVEL, "register degree was higher than bettiNumbers!");
    return fullLeakResidual;
  }
  const int degreeBettiNumber = bettiNumbers[degree];
  if (degreeBettiNumber == 0) return fullLeakResidual;

  const auto cycleHoles =
      holesCarryingTheTarget(*spacetime, degree, targetState.size());
  if (cycleHoles.empty()) return fullLeakResidual;
  const Eigen::MatrixXcd periodMatrixTransposed =
      holePeriodMatrix(spacetime, degree, degreeBettiNumber, cycleHoles,
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

}  // namespace tessera::cobordism
