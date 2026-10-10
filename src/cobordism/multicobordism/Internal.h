// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

// Private to the implementation of MultiCobordism in this directory, whose
// translation units define the members declared in
// include/cobordism/MultiCobordism.h by responsibility, one unit each. This
// header carries the includes and using-declarations those units share and
// the helpers more than one of them uses; a helper used by one unit lives in
// that unit's anonymous namespace
// (https://github.com/akellehe/tessera/issues/1481).

#pragma once

#include "cobordism/MultiCobordism.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <exception>
#include <functional>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <random>
#include <set>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <Eigen/Dense>

#include "Logger.h"
#include "chainhodge/BandDerivative.h"
#include "cobordism/ChainComplex.h"
#include "cobordism/EigenstateSynthesis.h"
#include "cobordism/HodgeLaplacian.h"
#include "cobordism/LevenbergMarquardt.h"
#include "cobordism/PencilLayer.h"
#include "cobordism/SurgicalCone.h"
#include "matter/MatterConfiguration.h"
#include "observables/SimplicialQubit.h"
#include "mesh/Edge.h"
#include "mesh/EdgeKey.h"
#include "mesh/EdgeList.h"
#include "mesh/Fingerprint.h"
#include "mesh/Simplex.h"
#include "mesh/Vertex.h"
#include "mesh/VertexList.h"
#include "quantum/ChoiJamiolkowski.h"
#include "simulations/ReggeSolver.h"
#include "spacetime/Spacetime.h"
#include "spacetime/topologies/SolidSimplex.h"
#include "spacetime/pachner/AddMove.h"
#include "spacetime/pachner/FlipMove.h"
#include "spacetime/pachner/IFlipMove.h"
#include "spacetime/pachner/RemoveMove.h"

namespace tessera::cobordism {

using ::tessera::MatterConfiguration;
using ::tessera::simulations::ReggeSolver;
using complexd = std::complex<double>;

namespace {

// How far the monodromy between two markings may sit from the identity before
// they are held to disagree about the frame of the whole. This is round-off
// room rather than a policy knob: a marking pair that genuinely frames the zero
// mode differently does so by an SL(2, Z) element, a whole integer away.
constexpr double kPeriodFrameMonodromyTolerance = 1e-8;

// A marking group frames the harmonic space when its period matrix is
// invertible. At four tori a cross pair has condition 1.8 and a conjugate pair
// 1e16, so this floor sits in an empty decade between them.
constexpr double kPeriodFrameConditionFloor = 1e-10;

// Frame agreement is checked over every group, which is a subset enumeration.
// There is one marking per boundary torus, so this bounds the number of tori
// rather than silently truncating the check.
constexpr std::size_t kPeriodFrameMaxMarkings = 16;

struct PeriodFrameSelection {
  Eigen::MatrixXcd block{};
  std::vector<Eigen::Index> rows{};
  std::string obstruction{};
};

// Select the period frame once for both the value and its derivative. A
// candidate is any group of complete markings whose cycle count equals the
// harmonic rank and whose period block is invertible. Every such group must
// induce the same frame; otherwise the whole read refuses rather than choosing
// an ordering-dependent answer.
inline PeriodFrameSelection selectPeriodFrame(
    const Eigen::MatrixXcd &periods,
    const std::vector<Eigen::Index> &markingRanks,
    bool requireAgreement = true) {
  PeriodFrameSelection selection;
  const Eigen::Index rank = periods.cols();
  if (markingRanks.size() > kPeriodFrameMaxMarkings) {
    selection.obstruction =
        "the whole carries " + std::to_string(markingRanks.size()) +
        " markings; frame agreement is only verified up to " +
        std::to_string(kPeriodFrameMaxMarkings);
    return selection;
  }
  std::vector<Eigen::Index> offsets(markingRanks.size());
  Eigen::Index row = 0;
  for (std::size_t m = 0; m < markingRanks.size(); ++m) {
    offsets[m] = row;
    row += markingRanks[m];
  }
  struct Candidate {
    Eigen::MatrixXcd block;
    std::vector<Eigen::Index> rows;
  };
  std::vector<Candidate> candidates;
  for (std::uint32_t mask = 1; mask < (1u << markingRanks.size()); ++mask) {
    Eigen::Index total = 0;
    for (std::size_t m = 0; m < markingRanks.size(); ++m)
      if ((mask >> m) & 1u) total += markingRanks[m];
    if (total != rank) continue;
    Candidate candidate{Eigen::MatrixXcd(rank, rank), {}};
    candidate.rows.reserve(static_cast<std::size_t>(rank));
    Eigen::Index at = 0;
    for (std::size_t m = 0; m < markingRanks.size(); ++m) {
      if (!((mask >> m) & 1u)) continue;
      candidate.block.middleRows(at, markingRanks[m]) =
          periods.middleRows(offsets[m], markingRanks[m]);
      for (Eigen::Index c = 0; c < markingRanks[m]; ++c)
        candidate.rows.push_back(offsets[m] + c);
      at += markingRanks[m];
    }
    const Eigen::VectorXd singularValues =
        candidate.block.jacobiSvd().singularValues();
    if (singularValues(singularValues.size() - 1) >
        kPeriodFrameConditionFloor * singularValues(0))
      candidates.push_back(std::move(candidate));
  }
  if (candidates.empty()) {
    selection.obstruction =
        "no group of markings frames the whole's degree-1 harmonic space: "
        "none has as many cycles as its rank (" +
        std::to_string(rank) + ") with an invertible period matrix";
    return selection;
  }
  // Agreement is a statement about periods: every marking group induces the
  // same frame on the whole's zero mode exactly when the integer matrix
  // relating them is the identity, which is what makes the frame the whole's
  // rather than a chosen block's.
  //
  // Under the Gram pairing these rows are not periods. The groups are then not
  // related by a topological monodromy and do not agree, so the caller reads in
  // the first group's frame instead of demanding agreement. That disagreement
  // is the metric content the pairing exists to expose.
  const Eigen::MatrixXcd firstInverse = candidates.front().block.inverse();
  for (std::size_t b = 1; requireAgreement && b < candidates.size(); ++b) {
    const double defect =
        (candidates[b].block * firstInverse -
         Eigen::MatrixXcd::Identity(rank, rank))
            .norm();
    if (defect > kPeriodFrameMonodromyTolerance) {
      selection.obstruction =
          "the marking groups disagree about the frame of the whole: the "
          "monodromy between group 0 and group " +
          std::to_string(b) + " differs from the identity by " +
          std::to_string(defect);
      return selection;
    }
  }
  selection.block = std::move(candidates.front().block);
  selection.rows = std::move(candidates.front().rows);
  return selection;
}

inline std::pair<std::uint64_t, std::uint64_t> edgeKey(
    const ::tessera::mesh::Edge *edge) {
  const auto sourceVertexId = edge->getSource()->getId();
  const auto targetVertexId = edge->getTarget()->getId();
  return {std::min(sourceVertexId, targetVertexId),
          std::max(sourceVertexId, targetVertexId)};
}

// z=l^2 is the optimization coordinate, while Edge stores l. Both square roots
// represent the same z; retain the root nearest the resident l so an accepted
// update cannot introduce an unrelated l -> -l branch jump.
inline complexd continuousSquareRoot(complexd z, complexd referenceLength) {
  const complexd principal = std::sqrt(z);
  return std::abs(principal - referenceLength) <=
                 std::abs(-principal - referenceLength)
             ? principal
             : -principal;
}

}  // namespace

}  // namespace tessera::cobordism
