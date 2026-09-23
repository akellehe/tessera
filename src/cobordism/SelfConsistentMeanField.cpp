// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "cobordism/SelfConsistentMeanField.h"

#include <algorithm>
#include <numeric>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

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

/// What one application of the declared covariance rule produced.
struct CovarianceBuild {
  /// Gamma, flat row-major.
  std::vector<complexd> covariance;
  /// The band ranks under the band rule; empty under the projector rule.
  std::vector<std::size_t> ranks;
  /// The eigenvalues of the operator the bands were read on, in the declared
  /// order: h itself, or its group average under a declared band symmetry.
  std::vector<complexd> ordered;
};

/// The operator the band rule reads its bands on: the carrier h, or, when a
/// band symmetry is declared, its group average
/// \f$ \bar h=|G|^{-1}\sum_g D(g)^{-1}hD(g) \f$.
Eigen::MatrixXcd bandOperator(const JointAction &action,
                              const SelfConsistentMeanFieldDeclaration &declaration) {
  const Eigen::MatrixXcd carrier = toMatrix(action.carrierOperator());
  if (declaration.bandSymmetry.empty()) return carrier;
  Eigen::MatrixXcd average = Eigen::MatrixXcd::Zero(carrier.rows(),
                                                    carrier.cols());
  for (const auto &flat : declaration.bandSymmetry) {
    if (flat.size() != static_cast<std::size_t>(carrier.size()))
      throw std::invalid_argument(
          "SelfConsistentMeanField: a band symmetry operator must be square "
          "over the carrier's cells");
    const Eigen::MatrixXcd action_g = toMatrix(flat);
    Eigen::FullPivLU<Eigen::MatrixXcd> lu(action_g);
    if (!lu.isInvertible())
      throw std::invalid_argument(
          "SelfConsistentMeanField: a band symmetry operator is singular");
    average += lu.inverse() * carrier * action_g;
  }
  return average / static_cast<double>(declaration.bandSymmetry.size());
}

/// The covariance the declared rule builds from the action's current carrier
/// operator.
///
/// Under the band rule the operator's eigendecomposition
/// \f$ V\Lambda V^{-1} \f$ is ordered by the declared rule, consecutive
/// eigenvalues within the band tolerance are grouped into bands, and
/// \f$ \Gamma=V\,\mathrm{diag}(w)\,V^{-1} \f$ with \f$ w=n_b/r_b \f$ on band
/// \f$ b \f$. Because \f$ w \f$ is constant on each band, \f$ \Gamma \f$ is a
/// combination of band projectors and does not depend on the basis the
/// eigensolver chose inside a degenerate band.
CovarianceBuild covarianceOf(
    const JointAction &action,
    const SelfConsistentMeanFieldDeclaration &declaration) {
  const bool ascending =
      declaration.occupationOrder == OccupationOrder::AscendingRealPart;
  CovarianceBuild build;
  if (declaration.covarianceRule == CovarianceRule::OccupiedProjector) {
    build.covariance =
        action.occupationProjector(declaration.occupiedModes, ascending);
    build.ordered = action.orderedCarrierEigenvalues(ascending);
    return build;
  }

  const Eigen::MatrixXcd target = bandOperator(action, declaration);
  Eigen::ComplexEigenSolver<Eigen::MatrixXcd> solver(target);
  if (solver.info() != Eigen::Success)
    throw std::runtime_error(
        "SelfConsistentMeanField: the band operator has no eigendecomposition");
  const Eigen::VectorXcd values = solver.eigenvalues();
  const Eigen::MatrixXcd vectors = solver.eigenvectors();
  std::vector<Eigen::Index> order(static_cast<std::size_t>(values.size()));
  std::iota(order.begin(), order.end(), Eigen::Index{0});
  std::stable_sort(order.begin(), order.end(),
                   [&](Eigen::Index left, Eigen::Index right) {
                     return ascending
                                ? values(left).real() < values(right).real()
                                : std::abs(values(left)) <
                                      std::abs(values(right));
                   });
  for (const auto index : order) build.ordered.push_back(values(index));

  std::vector<std::size_t> starts;
  for (std::size_t index = 0; index < build.ordered.size(); ++index) {
    if (index == 0) {
      starts.push_back(0);
      continue;
    }
    const double scale = std::max(1.0, std::abs(build.ordered[index - 1]));
    if (std::abs(build.ordered[index] - build.ordered[index - 1]) >
        declaration.bandTolerance * scale)
      starts.push_back(index);
  }
  for (std::size_t band = 0; band < starts.size(); ++band) {
    const std::size_t end =
        band + 1 < starts.size() ? starts[band + 1] : build.ordered.size();
    build.ranks.push_back(end - starts[band]);
  }
  if (declaration.bandOccupations.size() > build.ranks.size())
    throw std::invalid_argument(
        "SelfConsistentMeanField: " +
        std::to_string(declaration.bandOccupations.size()) +
        " band occupations were declared but the spectrum groups into only " +
        std::to_string(build.ranks.size()) + " bands");

  Eigen::VectorXcd weights = Eigen::VectorXcd::Zero(values.size());
  for (std::size_t band = 0; band < declaration.bandOccupations.size();
       ++band) {
    const double occupation = declaration.bandOccupations[band];
    const double rank = static_cast<double>(build.ranks[band]);
    if (occupation > rank)
      throw std::invalid_argument(
          "SelfConsistentMeanField: band " + std::to_string(band) +
          " has rank " + std::to_string(build.ranks[band]) +
          " and cannot hold the declared occupation " +
          std::to_string(occupation));
    for (std::size_t k = starts[band]; k < starts[band] + build.ranks[band];
         ++k)
      weights(order[k]) = complexd{occupation / rank, 0.0};
  }
  Eigen::FullPivLU<Eigen::MatrixXcd> lu(vectors);
  if (!lu.isInvertible())
    throw std::invalid_argument(
        "SelfConsistentMeanField: the band operator is defective, so no "
        "band projector is available from its eigenbasis");
  const Eigen::MatrixXcd gamma =
      vectors * weights.asDiagonal() * lu.inverse();
  const auto n = static_cast<std::size_t>(gamma.rows());
  build.covariance.assign(n * n, complexd{0.0, 0.0});
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j)
      build.covariance[i * n + j] =
          gamma(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j));
  return build;
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
  SelfConsistentMeanFieldReport report;

  // The uniform-seeded start: with no covariance declared, the solve begins
  // from the occupied modes of the operator at the geometry it was handed.
  if (action_.declaration().covariance.empty()) {
    JointActionDeclaration seeded = action_.declaration();
    seeded.covariance = covarianceOf(action_, declaration_).covariance;
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
    const CovarianceBuild build = covarianceOf(action_, declaration_);
    const auto &projector = build.covariance;
    const auto &ranks = build.ranks;
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

    // The eigenvalues of the operator the covariance was read on, taken at the
    // geometry this step ended on.
    const auto ordered = covarianceOf(action_, declaration_).ordered;
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
