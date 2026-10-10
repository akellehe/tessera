// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

// MultiCobordism (include/cobordism/MultiCobordism.h): the geometric
// operator readout. One of the translation units that define the class's
// members by responsibility
// (https://github.com/akellehe/tessera/issues/1481).

#include "Internal.h"

namespace tessera::cobordism {

MultiCobordism::GeometricOperatorReadout MultiCobordism::geometricOperator(
    int stateDimension, std::vector<std::vector<std::uint64_t>> frameCells,
    double tol, bool metric) const {
  return geometricOperatorOn(spacetime_, stateDimension, std::move(frameCells), tol, metric);
}

MultiCobordism::GeometricOperatorReadout MultiCobordism::geometricOperatorOn(
    const std::shared_ptr<Spacetime> &spacetime, int stateDimension,
    std::vector<std::vector<std::uint64_t>> frameCells, double tol,
    bool metric) const {
  GeometricOperatorReadout result;
  result.stateDimension = stateDimension;
  result.metric = metric;
  const auto obstruct = [&](std::string reason) {
    result.identifiable = false;
    result.obstruction = std::move(reason);
    result.choiState.clear();
    result.operatorMatrix.clear();
    return result;
  };
  if (!spacetime)
    return obstruct("the cobordism has no spacetime");
  if (stateDimension <= 0)
    return obstruct("state dimension must be positive");
  if (!(tol > 0.0) || !std::isfinite(tol))
    return obstruct("kernel tolerance must be finite and positive");

  const std::size_t d = static_cast<std::size_t>(stateDimension);
  if (d > std::numeric_limits<std::size_t>::max() / d)
    return obstruct("state dimension overflows the Choi width");
  const std::size_t choiWidth = d * d;
  EigenstateSynthesis synthesis(spacetime, 1, metricSource_);
  result.bulkCells = synthesis.bulkMinusBoundaryCells();
  result.bulkCellCount = result.bulkCells.size();
  if (result.bulkCells.empty())
    return obstruct("bulk-minus-boundary has no interior 1-cells");

  if (frameCells.empty()) {
    if (result.bulkCells.size() != choiWidth)
      return obstruct(
          "an ordered d^2 Choi frame is required when the bulk width differs");
    frameCells = result.bulkCells;
  }
  if (frameCells.size() != choiWidth)
    return obstruct("the Choi frame must contain exactly d^2 interior 1-cells");
  std::map<std::vector<std::uint64_t>, std::size_t> bulkIndex;
  for (std::size_t i = 0; i < result.bulkCells.size(); ++i)
    bulkIndex[result.bulkCells[i]] = i;
  std::set<std::vector<std::uint64_t>> usedFrameCells;
  std::vector<std::size_t> frameColumns;
  frameColumns.reserve(frameCells.size());
  for (auto &cell : frameCells) {
    std::sort(cell.begin(), cell.end());
    if (cell.size() != 2 || cell[0] == cell[1])
      return obstruct("a Choi frame entry is not an edge");
    const auto found = bulkIndex.find(cell);
    if (found == bulkIndex.end())
      return obstruct("a Choi frame edge is not in the bulk-minus-boundary");
    if (!usedFrameCells.insert(cell).second)
      return obstruct("the Choi frame repeats an interior edge");
    frameColumns.push_back(found->second);
  }
  result.frameCells = frameCells;

  const std::vector<complexd> flat =
      synthesis.bulkMinusBoundaryHarmonicMatrix(tol, metric);
  if (flat.size() % result.bulkCells.size() != 0)
    return obstruct("bulk harmonic matrix has an inconsistent shape");
  result.kernelDimension = flat.size() / result.bulkCells.size();
  if (result.kernelDimension == 0)
    return obstruct("bulk-minus-boundary kernel is empty");

  Eigen::MatrixXcd restricted(static_cast<Eigen::Index>(result.kernelDimension),
                              static_cast<Eigen::Index>(choiWidth));
  for (std::size_t row = 0; row < result.kernelDimension; ++row)
    for (std::size_t column = 0; column < choiWidth; ++column)
      restricted(static_cast<Eigen::Index>(row),
                 static_cast<Eigen::Index>(column)) =
          flat[row * result.bulkCells.size() + frameColumns[column]];
  if (!restricted.allFinite())
    return obstruct("the framed bulk kernel is non-finite");

  Eigen::JacobiSVD<Eigen::MatrixXcd> svd(restricted, Eigen::ComputeThinU |
                                                         Eigen::ComputeThinV);
  const Eigen::VectorXd &singularValues = svd.singularValues();
  result.frameSingularValues.reserve(
      static_cast<std::size_t>(singularValues.size()));
  for (Eigen::Index i = 0; i < singularValues.size(); ++i)
    result.frameSingularValues.push_back(singularValues[i]);
  const double leading = singularValues.size() == 0 ? 0.0 : singularValues[0];
  const double cutoff = tol * std::max(1.0, leading);
  for (Eigen::Index i = 0; i < singularValues.size(); ++i)
    if (singularValues[i] > cutoff)
      ++result.frameRank;
  if (result.frameRank == 0)
    return obstruct("the framed bulk kernel carries no Choi component");
  if (result.frameRank != 1)
    return obstruct("the framed bulk kernel has rank " +
                    std::to_string(result.frameRank) +
                    "; no unique Choi ray exists");

  // A = U Sigma V^H: the direct row-state spanning A is conj(V.col(0)). This
  // remains invariant under a change of basis among the bulk kernel rows.
  Eigen::VectorXcd choi = svd.matrixV().col(0).conjugate();
  for (Eigen::Index i = 0; i < choi.size(); ++i)
    if (std::abs(choi[i]) > cutoff) {
      choi *= std::conj(choi[i]) / std::abs(choi[i]);
      break;
    }
  result.choiState.assign(choi.data(), choi.data() + choi.size());

  result.operatorMatrix =
      ::tessera::quantum::ChoiJamiolkowski::operatorFromChoiState(
          result.choiState, stateDimension);
  Eigen::MatrixXcd U(stateDimension, stateDimension);
  for (int row = 0; row < stateDimension; ++row)
    for (int column = 0; column < stateDimension; ++column)
      U(row, column) = result.operatorMatrix[static_cast<std::size_t>(row) * d +
                                             static_cast<std::size_t>(column)];
  result.unitarityError =
      (U.adjoint() * U -
       Eigen::MatrixXcd::Identity(stateDimension, stateDimension))
          .norm();
  result.identifiable = true;
  result.obstruction.clear();
  return result;
}

}  // namespace tessera::cobordism
