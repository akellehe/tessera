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

/// The Euclidean norm of the joint stationarity force in the geometric fields
/// the inner relaxation declares variable.
///
/// The moment equations are left out because they are constraints rather than
/// forces, and a field the relaxation holds fixed is left out because its
/// equation is not one the solve is asked to satisfy: requiring stationarity of
/// a frozen coordinate would make every run report a fixed point it was never
/// looking for, exactly as including a frozen coordinate's equation in the
/// Newton system would overdetermine it.
double forceNormOf(const JointAction &action,
                   const HolomorphicRelaxationDeclaration &geometry) {
  double squared = 0.0;
  if (geometry.relaxLengths)
    for (const complexd &component : action.lengthStationarity())
      squared += std::norm(component);
  if (geometry.relaxLinks)
    for (const complexd &component : action.linkStationarity())
      squared += std::norm(component);
  return std::sqrt(squared);
}

/// The covariance the declared rule builds from the action's current carrier
/// operator, and the band ranks it grouped the spectrum into (empty under the
/// projector rule).
///
/// Under the band rule each band's projector is the difference of two prefix
/// projectors that end on band boundaries, so it does not depend on how the
/// eigensolver ordered the modes inside a degenerate band.
std::pair<std::vector<complexd>, std::vector<std::size_t>> covarianceOf(
    const JointAction &action,
    const SelfConsistentMeanFieldDeclaration &declaration) {
  const bool ascending =
      declaration.occupationOrder == OccupationOrder::AscendingRealPart;
  if (declaration.covarianceRule == CovarianceRule::OccupiedProjector)
    return {action.occupationProjector(declaration.occupiedModes, ascending),
            {}};

  const auto ordered = action.orderedCarrierEigenvalues(ascending);
  std::vector<std::size_t> starts;
  for (std::size_t index = 0; index < ordered.size(); ++index) {
    if (index == 0) {
      starts.push_back(0);
      continue;
    }
    const double scale = std::max(1.0, std::abs(ordered[index - 1]));
    if (std::abs(ordered[index] - ordered[index - 1]) >
        declaration.bandTolerance * scale)
      starts.push_back(index);
  }
  std::vector<std::size_t> ranks;
  for (std::size_t band = 0; band < starts.size(); ++band) {
    const std::size_t end =
        band + 1 < starts.size() ? starts[band + 1] : ordered.size();
    ranks.push_back(end - starts[band]);
  }
  if (declaration.bandOccupations.size() > ranks.size())
    throw std::invalid_argument(
        "SelfConsistentMeanField: " +
        std::to_string(declaration.bandOccupations.size()) +
        " band occupations were declared but the spectrum groups into only " +
        std::to_string(ranks.size()) + " bands");

  const std::size_t order = ordered.size();
  std::vector<complexd> covariance(order * order, complexd{0.0, 0.0});
  std::vector<complexd> lower(order * order, complexd{0.0, 0.0});
  for (std::size_t band = 0; band < declaration.bandOccupations.size();
       ++band) {
    const double occupation = declaration.bandOccupations[band];
    const double rank = static_cast<double>(ranks[band]);
    if (occupation > rank)
      throw std::invalid_argument(
          "SelfConsistentMeanField: band " + std::to_string(band) +
          " has rank " + std::to_string(ranks[band]) +
          " and cannot hold the declared occupation " +
          std::to_string(occupation));
    const std::size_t end = starts[band] + ranks[band];
    const auto upper = action.occupationProjector(end, ascending);
    if (occupation != 0.0)
      for (std::size_t index = 0; index < covariance.size(); ++index)
        covariance[index] += (occupation / rank) * (upper[index] - lower[index]);
    lower = upper;
  }
  return {covariance, ranks};
}

}  // namespace

SelfConsistentMeanField::SelfConsistentMeanField(
    JointAction action, SelfConsistentMeanFieldDeclaration declaration)
    : action_(std::move(action)), declaration_(std::move(declaration)) {
  if (!(declaration_.mixing > 0.0) || declaration_.mixing > 1.0)
    throw std::invalid_argument(
        "SelfConsistentMeanField: the mixing fraction lies in (0, 1]; a value "
        "outside it either freezes the covariance or overshoots the projector");
  if (declaration_.covarianceRule == CovarianceRule::BandFilling) {
    double total = 0.0;
    for (const double occupation : declaration_.bandOccupations) {
      if (!(occupation >= 0.0))
        throw std::invalid_argument(
            "SelfConsistentMeanField: a band occupation is a number of "
            "particles and cannot be negative");
      total += occupation;
    }
    if (!(total > 0.0))
      throw std::invalid_argument(
          "SelfConsistentMeanField: the band occupations are empty or sum to "
          "zero, so the carried state's bilinear density is identically zero "
          "and there is no backreaction to make self-consistent");
    if (!(declaration_.bandTolerance >= 0.0))
      throw std::invalid_argument(
          "SelfConsistentMeanField: the band tolerance cannot be negative");
  } else if (declaration_.occupiedModes == 0)
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
    seeded.covariance = covarianceOf(action_, declaration_).first;
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
    const auto [projector, ranks] = covarianceOf(action_, declaration_);
    step.bandRanks = ranks;
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
    step.forceNorm = forceNormOf(action_, declaration_.geometry);
    step.purityDefect = purityDefectOf(mixed);
    step.action = action_.value();
    step.occupiedEnergy = occupiedEnergyOf(mixed, action_.carrierOperator());

    const auto ordered = action_.orderedCarrierEigenvalues(ascending);
    // The occupied span: the declared mode count, or under the band rule every
    // mode up to the end of the last band with a nonzero occupation.
    std::size_t occupiedSpan = declaration_.occupiedModes;
    if (declaration_.covarianceRule == CovarianceRule::BandFilling) {
      occupiedSpan = 0;
      std::size_t end = 0;
      for (std::size_t band = 0; band < declaration_.bandOccupations.size() &&
                                 band < ranks.size();
           ++band) {
        end += ranks[band];
        if (declaration_.bandOccupations[band] != 0.0) occupiedSpan = end;
      }
    }
    step.occupiedEigenvalues.assign(
        ordered.begin(),
        ordered.begin() + static_cast<std::ptrdiff_t>(
                              std::min(occupiedSpan, ordered.size())));
    step.spectralGap =
        occupiedSpan > 0 && occupiedSpan < ordered.size()
            ? std::abs(ordered[occupiedSpan] - ordered[occupiedSpan - 1])
            : std::numeric_limits<double>::quiet_NaN();

    report.steps.push_back(step);
    report.forceNorm = step.forceNorm;
    report.covarianceChange = step.covarianceChange;
    report.purityDefect = step.purityDefect;
    report.covariance = mixed;
    report.occupiedEigenvalues = step.occupiedEigenvalues;
    report.occupiedEnergy = step.occupiedEnergy;
    report.spectralGap = step.spectralGap;
    report.bandRanks = step.bandRanks;
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
