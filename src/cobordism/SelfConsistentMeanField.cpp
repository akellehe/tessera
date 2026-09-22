// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "cobordism/SelfConsistentMeanField.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

#include <Eigen/Dense>

namespace tessera::cobordism {

using complexd = std::complex<double>;

namespace {

/// The Frobenius norm of the difference of two flat matrices of equal length.
double frobeniusDistance(const std::vector<complexd> &left,
                         const std::vector<complexd> &right) {
  if (left.size() != right.size())
    return std::numeric_limits<double>::quiet_NaN();
  double squared = 0.0;
  for (std::size_t index = 0; index < left.size(); ++index)
    squared += std::norm(left[index] - right[index]);
  return std::sqrt(squared);
}

/// A flat row-major square matrix lifted into Eigen.
Eigen::MatrixXcd toMatrix(const std::vector<complexd> &flat) {
  const auto order =
      static_cast<Eigen::Index>(std::llround(std::sqrt(
          static_cast<double>(flat.size()))));
  Eigen::MatrixXcd matrix(order, order);
  for (Eigen::Index i = 0; i < order; ++i)
    for (Eigen::Index j = 0; j < order; ++j)
      matrix(i, j) = flat[static_cast<std::size_t>(i * order + j)];
  return matrix;
}

/// \f$ \lVert\Gamma^2-\Gamma\rVert_F \f$, the Gaussianity certificate both
/// emergence sub-modes report. Exactly zero for a spectral projector, up to the
/// rounding of the eigendecomposition it was formed from.
double purityDefectOf(const std::vector<complexd> &covariance) {
  if (covariance.empty()) return std::numeric_limits<double>::quiet_NaN();
  const Eigen::MatrixXcd gamma = toMatrix(covariance);
  return (gamma * gamma - gamma).norm();
}

/// \f$ \operatorname{tr}(\Gamma h) \f$, the occupied band's energy.
complexd occupiedEnergyOf(const std::vector<complexd> &covariance,
                          const std::vector<complexd> &carrier) {
  if (covariance.empty() || covariance.size() != carrier.size())
    return complexd{0.0, 0.0};
  const Eigen::MatrixXcd gamma = toMatrix(covariance);
  const Eigen::MatrixXcd operatorMatrix = toMatrix(carrier);
  complexd trace{0.0, 0.0};
  for (Eigen::Index i = 0; i < gamma.rows(); ++i)
    for (Eigen::Index j = 0; j < gamma.cols(); ++j)
      trace += gamma(i, j) * operatorMatrix(j, i);
  return trace;
}

/// The Euclidean norm of the joint stationarity force in the two geometric
/// fields: the length equations and the link equations, without the moment
/// equations, which are constraints rather than forces.
double forceNormOf(const JointAction &action) {
  double squared = 0.0;
  for (const complexd &component : action.lengthStationarity())
    squared += std::norm(component);
  for (const complexd &component : action.linkStationarity())
    squared += std::norm(component);
  return std::sqrt(squared);
}

}  // namespace

SelfConsistentMeanField::SelfConsistentMeanField(
    JointAction action, SelfConsistentMeanFieldDeclaration declaration)
    : action_(std::move(action)), declaration_(std::move(declaration)) {
  if (!(declaration_.mixing > 0.0) || declaration_.mixing > 1.0)
    throw std::invalid_argument(
        "SelfConsistentMeanField: the mixing fraction lies in (0, 1]; a value "
        "outside it either freezes the covariance or overshoots the projector");
  if (declaration_.occupiedModes == 0)
    throw std::invalid_argument(
        "SelfConsistentMeanField: no mode is declared occupied, so the carried "
        "state's bilinear density is identically zero and there is no "
        "backreaction to make self-consistent");
}

SelfConsistentMeanFieldReport SelfConsistentMeanField::solve() {
  const bool ascending =
      declaration_.occupationOrder == OccupationOrder::AscendingRealPart;
  SelfConsistentMeanFieldReport report;

  // The uniform-seeded start: with no covariance declared, the solve begins
  // from the occupied modes of the operator at the geometry it was handed.
  if (action_.declaration().covariance.empty()) {
    JointActionDeclaration seeded = action_.declaration();
    seeded.covariance =
        action_.occupationProjector(declaration_.occupiedModes, ascending);
    action_ = JointAction(action_.spacetime(), std::move(seeded));
  }

  for (std::size_t iteration = 0; iteration < declaration_.maximumIterations;
       ++iteration) {
    SelfConsistentMeanFieldStep step;
    step.iteration = iteration;

    // (1) The geometry is made stationary against the action at the current
    // covariance. The inner solve advances the multipliers too, so the action
    // that comes back out of it is the one the re-occupation reads.
    HolomorphicRelaxation relaxation(action_, declaration_.geometry);
    const HolomorphicRelaxationReport geometryReport = relaxation.solve();
    action_ = relaxation.action();
    step.geometryConverged = geometryReport.converged;
    step.geometryResidualNorm = geometryReport.residualNorm;

    // (2) The covariance is re-occupied from the modes of h at the relaxed
    // geometry. This is the half of the fixed point the engine's own carried
    // state never took: a covariance that is only transported forgets which
    // modes of the current operator it is supposed to fill.
    const auto projector =
        action_.occupationProjector(declaration_.occupiedModes, ascending);
    const auto previous = action_.declaration().covariance;
    step.covarianceChange = frobeniusDistance(projector, previous);

    std::vector<complexd> mixed(projector.size(), complexd{0.0, 0.0});
    for (std::size_t index = 0; index < projector.size(); ++index)
      mixed[index] = (1.0 - declaration_.mixing) * previous[index] +
                     declaration_.mixing * projector[index];
    JointActionDeclaration updated = action_.declaration();
    updated.covariance = mixed;
    action_ = JointAction(action_.spacetime(), std::move(updated));

    // (3) The fixed-point measurements, taken with the covariance this step
    // produced: a geometry stationary for the previous covariance is not a
    // fixed point of the pair.
    step.forceNorm = forceNormOf(action_);
    step.purityDefect = purityDefectOf(mixed);
    step.action = action_.value();
    step.occupiedEnergy = occupiedEnergyOf(mixed, action_.carrierOperator());

    const auto ordered = action_.orderedCarrierEigenvalues(ascending);
    step.occupiedEigenvalues.assign(
        ordered.begin(),
        ordered.begin() +
            static_cast<std::ptrdiff_t>(
                std::min(declaration_.occupiedModes, ordered.size())));
    step.spectralGap =
        declaration_.occupiedModes > 0 &&
                declaration_.occupiedModes < ordered.size()
            ? std::abs(ordered[declaration_.occupiedModes] -
                       ordered[declaration_.occupiedModes - 1])
            : std::numeric_limits<double>::quiet_NaN();

    report.steps.push_back(step);
    report.forceNorm = step.forceNorm;
    report.covarianceChange = step.covarianceChange;
    report.purityDefect = step.purityDefect;
    report.covariance = mixed;
    report.occupiedEigenvalues = step.occupiedEigenvalues;
    report.occupiedEnergy = step.occupiedEnergy;
    report.spectralGap = step.spectralGap;
    report.action = step.action;

    if (step.forceNorm <= declaration_.tolerance &&
        step.covarianceChange <= declaration_.tolerance) {
      report.converged = true;
      break;
    }
  }
  return report;
}

}  // namespace tessera::cobordism
