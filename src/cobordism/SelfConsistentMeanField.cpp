// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "cobordism/SelfConsistentMeanField.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <utility>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/SVD>

#include "cobordism/HodgeLaplacian.h"
#include "mesh/Edge.h"
#include "mesh/EdgeList.h"
#include "spacetime/Spacetime.h"

namespace tessera::cobordism {

using complexd = std::complex<double>;

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

/// A number as text with three significant digits (printf's "%.3g"), for the
/// stop details.
std::string threeDigits(double value) {
  char buffer[32];
  std::snprintf(buffer, sizeof buffer, "%.3g", value);
  return buffer;
}

/// The Frobenius norm of the difference of two flat matrices of equal length.
double frobeniusDistance(const std::vector<complexd> &left,
                         const std::vector<complexd> &right) {
  if (left.size() != right.size()) return kNaN;
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

/// An Eigen matrix flattened row-major.
std::vector<complexd> toFlat(const Eigen::MatrixXcd &matrix) {
  const auto rows = static_cast<std::size_t>(matrix.rows());
  const auto columns = static_cast<std::size_t>(matrix.cols());
  std::vector<complexd> flat(rows * columns);
  for (std::size_t i = 0; i < rows; ++i)
    for (std::size_t j = 0; j < columns; ++j)
      flat[i * columns + j] =
          matrix(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j));
  return flat;
}

/// \f$ \lVert\Gamma^2-\Gamma\rVert_F \f$, the Gaussianity certificate both
/// emergence sub-modes report. Exactly zero for a spectral projector, up to the
/// rounding of the eigendecomposition it was formed from.
double purityDefectOf(const std::vector<complexd> &covariance) {
  if (covariance.empty()) return kNaN;
  const Eigen::MatrixXcd gamma = toMatrix(covariance);
  return (gamma * gamma - gamma).norm();
}

/// \f$ \operatorname{tr}(\Gamma h) \f$, the occupied energy.
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

/// Per-edge values summed over the declared edge classes: the value of a
/// shared coordinate is the sum of its edges' values, a link value taken on
/// the class's orientation when \p oriented. Without classes the values are
/// returned as they are.
std::vector<complexd> classSum(const std::vector<complexd> &values,
                               const HolomorphicRelaxationDeclaration &geometry,
                               bool oriented) {
  if (geometry.edgeClasses.empty()) return values;
  std::size_t count = 0;
  for (const std::size_t index : geometry.edgeClasses)
    count = std::max(count, index + 1);
  std::vector<complexd> summed(count, complexd{0.0, 0.0});
  for (std::size_t edge = 0;
       edge < values.size() && edge < geometry.edgeClasses.size(); ++edge) {
    const int orientation =
        oriented && !geometry.edgeClassOrientations.empty()
            ? geometry.edgeClassOrientations[edge]
            : 1;
    summed[geometry.edgeClasses[edge]] +=
        orientation > 0 ? values[edge] : -values[edge];
  }
  return summed;
}

/// The Euclidean norm of the joint stationarity force in the geometric fields
/// the geometry declaration relaxes.
///
/// The moment equations are left out because they are constraints rather than
/// forces, and a field the relaxation holds fixed is left out because its
/// equation is not one the solve is asked to satisfy: requiring stationarity of
/// a frozen coordinate would make every run report a fixed point it was never
/// looking for, exactly as including a frozen coordinate's equation in the
/// Newton system would overdetermine it.
///
/// Under declared edge classes the geometric fields are the shared
/// coordinates, and the force on each is the sum of the forces on the edges
/// that carry it (a link force taken on the class's orientation): the force
/// the Newton solve drives to zero.
double forceNormOf(const JointAction &action,
                   const HolomorphicRelaxationDeclaration &geometry) {
  double squared = 0.0;
  if (geometry.relaxLengths)
    for (const complexd &component :
         classSum(action.lengthStationarity(), geometry, false))
      squared += std::norm(component);
  if (geometry.relaxLinks)
    for (const complexd &component :
         classSum(action.linkStationarity(), geometry, true))
      squared += std::norm(component);
  return std::sqrt(squared);
}

/// The Euclidean norm of the pinned constraints' residuals
/// \f$ p_j(h_{\mathcal C})-p_j^{\star} \f$; zero when none is declared.
double momentResidualNormOf(const JointAction &action) {
  double squared = 0.0;
  for (const complexd &residual : action.momentResiduals())
    squared += std::norm(residual);
  return std::sqrt(squared);
}

/// The operator the declared rule reads its modes on, flat row-major: the
/// carrier \f$ h \f$, or under the band rule with a declared band symmetry
/// its group average \f$ \bar h=|G|^{-1}\sum_g D(g)^{-1}hD(g) \f$.
std::vector<complexd> bandOperatorFlat(
    const JointAction &action,
    const SelfConsistentMeanFieldDeclaration &declaration) {
  std::vector<complexd> carrier = action.carrierOperator();
  if (declaration.covarianceRule != CovarianceRule::BandFilling ||
      declaration.bandSymmetry.empty())
    return carrier;
  const Eigen::MatrixXcd h = toMatrix(carrier);
  Eigen::MatrixXcd average = Eigen::MatrixXcd::Zero(h.rows(), h.cols());
  for (const auto &flat : declaration.bandSymmetry) {
    if (flat.size() != carrier.size())
      throw std::invalid_argument(
          "SelfConsistentMeanField: a band symmetry operator must be square "
          "over the carrier's cells");
    const Eigen::MatrixXcd operatorOfG = toMatrix(flat);
    Eigen::FullPivLU<Eigen::MatrixXcd> lu(operatorOfG);
    if (!lu.isInvertible())
      throw std::invalid_argument(
          "SelfConsistentMeanField: a band symmetry operator is singular");
    average += lu.inverse() * h * operatorOfG;
  }
  return toFlat(average /
                static_cast<double>(declaration.bandSymmetry.size()));
}

/// The Riesz projector of the occupied fiber, the sum of the occupied bands'
/// projectors, flat row-major.
std::vector<complexd> fiberProjectorOf(const BandRead &read) {
  if (read.bands.empty()) return {};
  std::vector<complexd> projector(read.bands.front().projector.size(),
                                  complexd{0.0, 0.0});
  for (const auto &band : read.bands)
    for (std::size_t index = 0;
         index < projector.size() && index < band.projector.size(); ++index)
      projector[index] += band.projector[index];
  return projector;
}

/// The state a rebuild sets at a point from the band read there: the
/// covariance and, when fiber constraints are pinned, the fiber's projector
/// and, under `FiberConstraintForm::BandEigenvalues`, the projector of each
/// pinned band.
RebuiltCarrierState rebuiltStateOf(
    const BandRead &read,
    const SelfConsistentMeanFieldDeclaration &declaration) {
  RebuiltCarrierState state;
  state.covariance = read.covariance;
  // What the analytic Jacobian differentiates the state with: the
  // eigendecomposition the bands were read from, each band's modes and its
  // weight in the covariance, the tolerance the bands were grouped at, and
  // the symmetry the operator was averaged under.
  state.eigenvalues = read.eigenvalues;
  state.eigenvectors = read.eigenvectors;
  state.leftEigenvectors = read.leftEigenvectors;
  if (declaration.covarianceRule == CovarianceRule::BandFilling)
    state.bandSymmetry = declaration.bandSymmetry;
  const bool fiber = declaration.fiberMoments > 0;
  for (const auto &band : read.bands) {
    RebuiltBand entry;
    entry.modes = band.modes;
    entry.covarianceWeight =
        band.rank > 0 ? band.occupation / static_cast<double>(band.rank) : 0.0;
    entry.inFiber = fiber;
    state.bands.push_back(std::move(entry));
  }
  if (!fiber) return state;
  state.momentProjector = fiberProjectorOf(read);
  if (declaration.fiberConstraintForm == FiberConstraintForm::BandEigenvalues)
    for (std::size_t band = 0;
         band < read.bands.size() && band < declaration.fiberMoments; ++band) {
      state.bandProjectors.push_back(read.bands[band].projector);
      state.bandProjectorBands.push_back(band);
    }
  return state;
}

/// The power of the unit \f$ s \f$ a constraint's value carries: \f$ j \f$
/// for a power sum, one for a band mean.
int unitPower(const SpectralMomentConstraint &constraint) {
  return constraint.form == SpectralConstraintForm::PowerSum ? constraint.order
                                                             : 1;
}

/// The largest \f$ |z_e| \f$ over the edges of the complex.
double largestSquaredLengthOf(const JointAction &action) {
  const auto &spacetime = action.spacetime();
  if (!spacetime || !spacetime->getEdgeList()) return 0.0;
  double largest = 0.0;
  for (const auto *edge : spacetime->getEdgeList()->toVector())
    if (edge != nullptr) {
      const complexd length = edge->getLength();
      largest = std::max(largest, std::abs(length * length));
    }
  return largest;
}

/// The declared constraints of \p action with the pinned fiber constraints
/// installed: \f$ p_j(h_{\mathcal C})=p_j^{\star} \f$ for
/// \f$ j=1,\ldots,m_{\rm c} \f$ on the fiber of \p read, or, under
/// `FiberConstraintForm::BandEigenvalues`, \f$ \lambda_b=\lambda_b^{\star} \f$
/// for the first \f$ m_{\rm c} \f$ occupied bands, the targets
/// declared or else the fiber's values at the action's point. The multipliers
/// start at the least-squares estimate there, the \f$ \xi \f$ that best
/// balances the stationarity force on the relaxed coordinates,
/// \f$ \min_\xi\lVert F+G^{\mathsf T}\xi\rVert \f$ with \f$ G \f$ the
/// constraints' gradients (the first-order multiplier estimate of
/// constrained root finding), in the minimum-norm sense when pinned power
/// sums depend on one another. It is where the root find starts and changes
/// no equation. The action must carry the covariance it will be solved with.
/// Returns the fiber's rank.
/// @throws std::invalid_argument when more moments are pinned than the fiber's
///   rank, when targets are declared for a different number of moments, or
///   when the action already declares constraints of its own.
std::size_t installFiberMoments(
    JointAction &action, const SelfConsistentMeanFieldDeclaration &declaration,
    const BandRead &read) {
  std::size_t rank = 0;
  for (const auto &band : read.bands) rank += band.rank;
  if (declaration.fiberMoments == 0) return rank;
  const bool eigenvalues =
      declaration.fiberConstraintForm == FiberConstraintForm::BandEigenvalues;
  if (!eigenvalues && declaration.fiberMoments > rank)
    throw std::invalid_argument(
        "SelfConsistentMeanField: " +
        std::to_string(declaration.fiberMoments) +
        " power sums of the occupied fiber are declared pinned, but the fiber "
        "has rank " + std::to_string(rank) +
        "; its power sums j = 1 .. " + std::to_string(rank) +
        " are the ones WP v17 §3.4 pins");
  if (eigenvalues && declaration.fiberMoments > read.bands.size())
    throw std::invalid_argument(
        "SelfConsistentMeanField: " +
        std::to_string(declaration.fiberMoments) +
        " band eigenvalues are declared pinned, but the content occupies " +
        std::to_string(read.bands.size()) + " bands");
  if (!declaration.fiberMomentTargets.empty() &&
      declaration.fiberMomentTargets.size() != declaration.fiberMoments)
    throw std::invalid_argument(
        "SelfConsistentMeanField: " +
        std::to_string(declaration.fiberMomentTargets.size()) +
        " fiber moment targets were declared for " +
        std::to_string(declaration.fiberMoments) + " pinned moments");
  if (!action.declaration().momentConstraints.empty())
    throw std::invalid_argument(
        "SelfConsistentMeanField: the action already declares spectral "
        "constraints of its own, so the occupied fiber's moments cannot be "
        "pinned beside them");
  JointActionDeclaration pinned = action.declaration();
  pinned.momentProjector = fiberProjectorOf(read);
  // The unit: declared, or the fiber's spectral radius here.
  double scale = declaration.fiberMomentScale;
  if (!(scale > 0.0)) {
    scale = 0.0;
    for (const auto &band : read.bands)
      for (const complexd &value : band.eigenvalues)
        scale = std::max(scale, std::abs(value));
    if (!(scale > 0.0)) scale = 1.0;
  }
  pinned.momentScale = scale;
  for (std::size_t index = 0; index < declaration.fiberMoments; ++index) {
    SpectralMomentConstraint constraint;
    if (eigenvalues) {
      constraint.form = SpectralConstraintForm::BandMean;
      constraint.band = index;
      pinned.momentBandProjectors.push_back(read.bands[index].projector);
    } else {
      constraint.order = static_cast<int>(index + 1);
    }
    pinned.momentConstraints.push_back(constraint);
  }
  // The targets: declared (in the operator's own unit, so divided by s^j, or
  // by s for a band eigenvalue), or the fiber's own values at this point, in
  // the unit.
  JointAction measured(action.spacetime(), pinned);
  const std::vector<complexd> values = measured.constraintValues();
  for (std::size_t index = 0; index < pinned.momentConstraints.size();
       ++index)
    pinned.momentConstraints[index].target =
        declaration.fiberMomentTargets.empty()
            ? values[index]
            : declaration.fiberMomentTargets[index] /
                  std::pow(scale, static_cast<double>(unitPower(
                                      pinned.momentConstraints[index])));
  action = JointAction(action.spacetime(), std::move(pinned));

  // The least-squares multipliers: the force with every multiplier zero and
  // the constraints' gradients, both on the relaxed coordinates.
  const HolomorphicRelaxationDeclaration &geometry = declaration.geometry;
  std::vector<complexd> force;
  if (geometry.relaxLengths)
    for (const complexd &value :
         classSum(action.lengthStationarity(), geometry, false))
      force.push_back(value);
  if (geometry.relaxLinks)
    for (const complexd &value :
         classSum(action.linkStationarity(), geometry, true))
      force.push_back(value);
  const auto rows = static_cast<Eigen::Index>(force.size());
  const auto columns =
      static_cast<Eigen::Index>(action.declaration().momentConstraints.size());
  if (rows == 0 || columns == 0) return rank;
  const std::size_t edges = action.edgeCount();
  Eigen::MatrixXcd gradients(rows, columns);
  for (Eigen::Index column = 0; column < columns; ++column) {
    const std::vector<complexd> gradient =
        action.momentGradient(static_cast<std::size_t>(column));
    std::vector<complexd> entries;
    if (geometry.relaxLengths)
      for (const complexd &value : classSum(
               std::vector<complexd>(
                   gradient.begin(),
                   gradient.begin() + static_cast<std::ptrdiff_t>(edges)),
               geometry, false))
        entries.push_back(value);
    if (geometry.relaxLinks)
      for (const complexd &value : classSum(
               std::vector<complexd>(
                   gradient.begin() + static_cast<std::ptrdiff_t>(edges),
                   gradient.end()),
               geometry, true))
        entries.push_back(value);
    for (Eigen::Index row = 0; row < rows; ++row)
      gradients(row, column) = entries[static_cast<std::size_t>(row)];
  }
  Eigen::VectorXcd target(rows);
  for (Eigen::Index row = 0; row < rows; ++row)
    target(row) = -force[static_cast<std::size_t>(row)];
  const Eigen::JacobiSVD<Eigen::MatrixXcd> svd(
      gradients, Eigen::ComputeThinU | Eigen::ComputeThinV);
  const Eigen::VectorXd &singular = svd.singularValues();
  const double largest = singular.size() > 0 ? singular(0) : 0.0;
  Eigen::Index kept = 0;
  while (kept < singular.size() &&
         singular(kept) > geometry.rankTolerance * largest)
    ++kept;
  Eigen::VectorXcd estimate = Eigen::VectorXcd::Zero(columns);
  if (kept > 0) {
    Eigen::VectorXcd coefficients =
        svd.matrixU().leftCols(kept).adjoint() * target;
    for (Eigen::Index k = 0; k < kept; ++k) coefficients(k) /= singular(k);
    estimate = svd.matrixV().leftCols(kept) * coefficients;
  }
  std::vector<complexd> multipliers(static_cast<std::size_t>(columns));
  for (Eigen::Index k = 0; k < columns; ++k)
    multipliers[static_cast<std::size_t>(k)] = estimate(k);
  action.setMultipliers(multipliers);
  return rank;
}

/// The multipliers of the pinned constraints in the operator's own unit: the
/// multiplier of \f$ p_j(h/s) \f$ times \f$ s^{-j} \f$.
std::vector<complexd> ownUnitMultipliers(const JointAction &action) {
  std::vector<complexd> values = action.multipliers();
  const double scale = action.declaration().momentScale;
  const auto &constraints = action.declaration().momentConstraints;
  for (std::size_t index = 0; index < values.size(); ++index)
    values[index] /= std::pow(scale, unitPower(constraints[index]));
  return values;
}

/// One iterate's measurements: the force, the covariance and its bands, the
/// occupied energy, the pinned moments' multipliers and residuals, and the
/// action, all at the action's current geometry with the covariance
/// \p covariance, which the action must carry.
SelfConsistentMeanFieldStep measure(
    std::size_t index, const JointAction &action, const BandRead &read,
    const std::vector<complexd> &covariance,
    const std::vector<complexd> &previous,
    const HolomorphicRelaxationDeclaration &geometry) {
  SelfConsistentMeanFieldStep step;
  step.iteration = index;
  step.forceNorm = forceNormOf(action, geometry);
  step.covarianceChange =
      previous.empty() ? 0.0 : frobeniusDistance(covariance, previous);
  step.purityDefect = purityDefectOf(covariance);
  const ReportedActionValue value = action.reportedValue();
  step.action = value.value;
  step.actionAvailable = value.available;
  step.actionUnavailable = value.unavailable;
  step.occupiedEnergy = occupiedEnergyOf(covariance, action.carrierOperator());
  step.occupiedEigenvalues = read.occupiedEigenvalues;
  step.spectralGap = read.spectralGap;
  step.bandRanks = read.ranks;
  step.bands = read.bands;
  step.bandIsolation = read.bandIsolation;
  step.bandCrossing = read.crossing;
  step.multipliers = ownUnitMultipliers(action);
  step.momentResidualNorm = momentResidualNormOf(action);
  if (geometry.recordTerms) step.terms = actionTermRecords(action, geometry);
  return step;
}

/// The moment-constrained action's Hessian on the range of the
/// Hellmann-Feynman force (`SelfConsistentMeanFieldReport::forceHessian`),
/// from the joint Jacobian \p jacobian of the relaxation \p probe at the
/// action's point, whose geometric block is the Hessian of the geometric
/// action, the occupied energy and the multiplier terms. The tangent space of
/// the pinned constraints is the kernel of their analytic gradients
/// (`JointAction::momentGradient`), exact to rounding, rather than of the
/// Jacobian's difference-rule constraint rows: on a sheeted host only as many
/// power sums of the fiber are independent as it has distinct eigenvalues,
/// and difference rows of the dependent ones differ from exact dependence by
/// the rule's truncation, which would cut spurious directions from the
/// tangent space.
complexd forceHessianOf(const JointAction &action,
                        const HolomorphicRelaxationDeclaration &probe,
                        const Eigen::MatrixXcd &jacobian,
                        double &forceNorm, double &hessianScale) {
  std::vector<complexd> force;
  std::vector<complexd> scale;
  if (probe.relaxLengths)
    for (const complexd &value :
         classSum(action.hellmannFeynmanLengthForce(), probe, false)) {
      force.push_back(value);
      scale.emplace_back(1.0, 0.0);
    }
  if (probe.relaxLinks)
    for (const complexd &value :
         classSum(action.hellmannFeynmanLinkForce(), probe, true)) {
      force.push_back(value);
      // delta = i theta: d/dtheta = i d/ddelta
      scale.emplace_back(0.0, 1.0);
    }
  const auto geometric = static_cast<Eigen::Index>(force.size());
  Eigen::VectorXcd f(geometric);
  Eigen::VectorXcd t(geometric);
  for (Eigen::Index k = 0; k < geometric; ++k) {
    f(k) = force[static_cast<std::size_t>(k)];
    t(k) = scale[static_cast<std::size_t>(k)];
  }
  forceNorm = f.norm();
  if (geometric == 0 || jacobian.rows() < geometric) return {kNaN, kNaN};
  // In the coordinates (z, theta): H -> T H T and f -> T f, T = diag(t).
  const Eigen::MatrixXcd hessian =
      t.asDiagonal() * jacobian.topLeftCorner(geometric, geometric) *
      t.asDiagonal();
  hessianScale = hessian.norm();
  Eigen::VectorXcd direction = t.asDiagonal() * f;
  const auto constraints =
      static_cast<Eigen::Index>(action.constraintCount());
  if (constraints > 0) {
    // The tangent space of the pinned constraints: the kernel of their
    // analytic gradients in the same coordinates, by the pseudo-inverse with
    // the relaxation's declared rank tolerance.
    const std::size_t edges = action.edgeCount();
    Eigen::MatrixXcd gradients(constraints, geometric);
    for (Eigen::Index row = 0; row < constraints; ++row) {
      const std::vector<complexd> gradient =
          action.momentGradient(static_cast<std::size_t>(row));
      std::vector<complexd> entries;
      if (probe.relaxLengths)
        for (const complexd &value : classSum(
                 std::vector<complexd>(gradient.begin(),
                                       gradient.begin() +
                                           static_cast<std::ptrdiff_t>(edges)),
                 probe, false))
          entries.push_back(value);
      if (probe.relaxLinks)
        for (const complexd &value : classSum(
                 std::vector<complexd>(gradient.begin() +
                                           static_cast<std::ptrdiff_t>(edges),
                                       gradient.end()),
                 probe, true))
          entries.push_back(value);
      for (Eigen::Index column = 0; column < geometric; ++column)
        gradients(row, column) =
            entries[static_cast<std::size_t>(column)] * t(column);
    }
    Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXcd> cod;
    cod.setThreshold(probe.rankTolerance);
    cod.compute(gradients);
    direction -= cod.pseudoInverse() * (gradients * direction);
  }
  const complexd norm = direction.transpose() * direction;
  if (std::abs(norm) == 0.0) return {kNaN, kNaN};
  const complexd quotient = direction.transpose() * hessian * direction;
  return quotient / norm;
}

/// The joint Jacobian at the action's current point, the covariance (and the
/// fiber) rebuilt by \p follower at every node, its rank decision and the
/// Hessian along the Hellmann-Feynman force; written into the report.
void readJointJacobian(const JointAction &action,
                       const SelfConsistentMeanFieldDeclaration &declaration,
                       const BandFollower &follower,
                       SelfConsistentMeanFieldReport &report) {
  HolomorphicRelaxationDeclaration probe = declaration.geometry;
  // The Jacobian does not read the held sectors, and a probe need not be
  // refused where the solve already stopped.
  probe.heldSectors.clear();
  const bool fiber = !action.declaration().momentConstraints.empty();
  if (fiber) probe.relaxMultipliers = true;
  if (!probe.relaxLengths && !probe.relaxLinks && !probe.relaxMultipliers)
    return;
  CovarianceRebuild rebuild;
  rebuild.at = [&follower, &declaration](const JointAction &point) {
    return rebuiltStateOf(follower.read(bandOperatorFlat(point, declaration)),
                          declaration);
  };
  const HolomorphicRelaxation relaxation(action, probe, rebuild);
  const std::size_t size = relaxation.variableCount();
  report.jacobianSize = size;
  if (size == 0) return;
  const std::vector<complexd> flat = relaxation.jacobian();
  const Eigen::MatrixXcd jacobian = toMatrix(flat);
  const Eigen::JacobiSVD<Eigen::MatrixXcd> svd(jacobian);
  const Eigen::VectorXd &singular = svd.singularValues();
  const double largest = singular.size() > 0 ? singular(0) : 0.0;
  Eigen::Index rank = 0;
  while (rank < singular.size() &&
         singular(rank) > declaration.geometry.rankTolerance * largest)
    ++rank;
  report.jacobianRank = static_cast<std::size_t>(rank);
  report.largestSingularValue = largest;
  report.smallestRetainedSingularValue = rank > 0 ? singular(rank - 1) : kNaN;
  report.largestDiscardedSingularValue =
      rank < singular.size() ? singular(rank) : 0.0;
  report.rankGap = rank < singular.size() && rank > 0
                       ? singular(rank - 1) / singular(rank)
                       : (rank == 0 ? kNaN
                                    : std::numeric_limits<double>::infinity());
  report.forceHessian =
      forceHessianOf(action, probe, jacobian, report.hellmannFeynmanForceNorm,
                     report.forceHessianScale);
}

/// The report's summary of the point the solve stopped at, from its last
/// iterate, and the measurements taken once there.
void finishReport(SelfConsistentMeanFieldReport &report,
                  std::vector<SelfConsistentMeanFieldStep> steps,
                  const JointAction &action,
                  const SelfConsistentMeanFieldDeclaration &declaration,
                  const BandFollower &follower, double startScale) {
  report.steps = std::move(steps);
  const SelfConsistentMeanFieldStep &last = report.steps.back();
  report.forceNorm = last.forceNorm;
  report.covarianceChange = last.covarianceChange;
  report.purityDefect = last.purityDefect;
  report.covariance = action.declaration().covariance;
  report.occupiedEigenvalues = last.occupiedEigenvalues;
  report.occupiedEnergy = last.occupiedEnergy;
  report.spectralGap = last.spectralGap;
  report.bandRanks = last.bandRanks;
  report.bands = last.bands;
  report.bandIsolation = last.bandIsolation;
  report.action = last.action;
  report.actionAvailable = last.actionAvailable;
  report.actionUnavailable = last.actionUnavailable;
  report.bandCrossingIterates = 0;
  report.lowestBandOverlap = 1.0;
  for (const auto &step : report.steps) {
    if (step.bandCrossing) ++report.bandCrossingIterates;
    for (const auto &band : step.bands)
      report.lowestBandOverlap =
          std::min(report.lowestBandOverlap, std::abs(band.overlap));
  }
  // The pinned constraints in the operator's own unit: targets and residuals
  // times s^j, multipliers times s^-j.
  const double scale = action.declaration().momentScale;
  report.momentScale = scale;
  const auto &constraints = action.declaration().momentConstraints;
  const std::vector<complexd> residuals = action.momentResiduals();
  for (std::size_t index = 0; index < constraints.size(); ++index) {
    const double unit = std::pow(scale, unitPower(constraints[index]));
    report.momentTargets.push_back(constraints[index].target * unit);
    report.momentResiduals.push_back(residuals[index] * unit);
  }
  report.multipliers = ownUnitMultipliers(action);
  report.kontsevichSegalMargin =
      HodgeLaplacian::kontsevichSegalMargin(*action.spacetime());
  report.largestLengthRatio =
      startScale > 0.0 ? largestSquaredLengthOf(action) / startScale : 1.0;
  // A point at which the joint Jacobian has no value leaves it unread; its
  // rank fields and the Hessian along the force then read zero and NaN.
  auto unread = [&report]() {
    report.jacobianRank = 0;
    report.largestSingularValue = kNaN;
    report.smallestRetainedSingularValue = kNaN;
    report.largestDiscardedSingularValue = kNaN;
    report.rankGap = kNaN;
    report.forceHessian = {kNaN, kNaN};
  };
  try {
    readJointJacobian(action, declaration, follower, report);
  } catch (const std::invalid_argument &) {
    unread();
  } catch (const std::domain_error &) {
    unread();
  } catch (const std::runtime_error &) {
    unread();
  }
}

}  // namespace

// ------------------------------------------------------------ BandFollower

BandFollower::BandFollower(
    const SelfConsistentMeanFieldDeclaration &declaration)
    : rule_(declaration.covarianceRule),
      occupiedModes_(declaration.occupiedModes),
      occupations_(declaration.bandOccupations),
      tolerance_(declaration.bandTolerance),
      order_(declaration.occupationOrder),
      selection_(declaration.bandSelection) {
  if (rule_ == CovarianceRule::BandFilling) {
    double total = 0.0;
    for (const double occupation : occupations_) {
      if (!(occupation >= 0.0))
        throw std::invalid_argument(
            "BandFollower: a band occupation is a number of particles and "
            "cannot be negative");
      total += occupation;
    }
    if (!(total > 0.0))
      throw std::invalid_argument(
          "BandFollower: the band occupations are empty or sum to zero, so "
          "the carried state's bilinear density is identically zero and there "
          "is no backreaction to make self-consistent");
    if (!(tolerance_ >= 0.0))
      throw std::invalid_argument(
          "BandFollower: the band tolerance cannot be negative");
  } else if (occupiedModes_ == 0) {
    throw std::invalid_argument(
        "BandFollower: no mode is declared occupied, so the carried state's "
        "bilinear density is identically zero and there is no backreaction to "
        "make self-consistent");
  }
}

BandRead BandFollower::read(const std::vector<complexd> &operatorMatrix) const {
  BandRead out;
  const Eigen::MatrixXcd target = toMatrix(operatorMatrix);
  const Eigen::Index n = target.rows();
  const bool ascending = order_ == OccupationOrder::AscendingRealPart;

  Eigen::VectorXcd values;
  Eigen::MatrixXcd vectors;
  if (n > 0) {
    const Eigen::ComplexEigenSolver<Eigen::MatrixXcd> solver(target);
    if (solver.info() != Eigen::Success)
      throw std::runtime_error(
          "BandFollower: the operator's eigendecomposition did not converge");
    values = solver.eigenvalues();
    vectors = solver.eigenvectors();
  }
  // The declared order, stable so that equal keys keep the solver's order.
  std::vector<Eigen::Index> order(static_cast<std::size_t>(n));
  std::iota(order.begin(), order.end(), Eigen::Index{0});
  std::stable_sort(order.begin(), order.end(),
                   [&](Eigen::Index left, Eigen::Index right) {
                     return ascending
                                ? values(left).real() < values(right).real()
                                : std::abs(values(left)) <
                                      std::abs(values(right));
                   });
  std::vector<std::size_t> placeOf(static_cast<std::size_t>(n));
  for (std::size_t place = 0; place < order.size(); ++place) {
    out.ordered.push_back(values(order[place]));
    placeOf[static_cast<std::size_t>(order[place])] = place;
  }
  // The degenerate groups in the declared order: consecutive eigenvalues
  // within the band tolerance, relative to their size.
  std::vector<std::size_t> starts;
  std::vector<std::size_t> groupOfPlace(order.size(), 0);
  for (std::size_t place = 0; place < out.ordered.size(); ++place) {
    if (place == 0 ||
        std::abs(out.ordered[place] - out.ordered[place - 1]) >
            tolerance_ * std::max(1.0, std::abs(out.ordered[place - 1])))
      starts.push_back(place);
    groupOfPlace[place] = starts.size() - 1;
  }
  std::vector<std::size_t> groupRanks;
  for (std::size_t group = 0; group < starts.size(); ++group)
    groupRanks.push_back(
        (group + 1 < starts.size() ? starts[group + 1] : out.ordered.size()) -
        starts[group]);
  if (rule_ == CovarianceRule::BandFilling) out.ranks = groupRanks;

  // The bands the covariance fills: their modes (eigenvector indices), in the
  // declared order of where they were chosen.
  struct Selected {
    std::size_t declaredIndex = 0;
    double occupation = 0.0;
    std::vector<Eigen::Index> modes;
    std::vector<std::size_t> declaredPositions;
  };
  std::vector<Selected> selected;
  const bool choose =
      reference_.empty() || selection_ == BandSelection::SortEveryIterate;
  if (choose) {
    if (rule_ == CovarianceRule::BandFilling) {
      if (occupations_.size() > groupRanks.size())
        throw std::invalid_argument(
            "BandFollower: " + std::to_string(occupations_.size()) +
            " band occupations were declared but the spectrum groups into "
            "only " + std::to_string(groupRanks.size()) + " bands");
      for (std::size_t band = 0; band < occupations_.size(); ++band) {
        const double occupation = occupations_[band];
        if (occupation > static_cast<double>(groupRanks[band]))
          throw std::invalid_argument(
              "BandFollower: band " + std::to_string(band) + " has rank " +
              std::to_string(groupRanks[band]) +
              " and cannot hold the declared occupation " +
              std::to_string(occupation));
        if (occupation == 0.0) continue;
        Selected band_;
        band_.declaredIndex = band;
        band_.occupation = occupation;
        for (std::size_t place = starts[band];
             place < starts[band] + groupRanks[band]; ++place) {
          band_.modes.push_back(order[place]);
          band_.declaredPositions.push_back(place);
        }
        selected.push_back(std::move(band_));
      }
    } else {
      if (occupiedModes_ > order.size())
        throw std::invalid_argument(
            "BandFollower: " + std::to_string(occupiedModes_) +
            " occupied modes were asked for on an operator of " +
            std::to_string(order.size()) + " modes");
      Selected band_;
      band_.occupation = static_cast<double>(occupiedModes_);
      for (std::size_t place = 0; place < occupiedModes_; ++place) {
        band_.modes.push_back(order[place]);
        band_.declaredPositions.push_back(place);
      }
      selected.push_back(std::move(band_));
    }
    // Where a reference is held, the bands keep the places they were chosen
    // at, so a re-sorted band whose places moved is not read as chosen anew.
    if (!reference_.empty() && reference_.size() == selected.size())
      for (std::size_t band = 0; band < selected.size(); ++band)
        selected[band].declaredPositions = reference_[band].declaredPositions;
  }

  Eigen::MatrixXcd inverse;
  if (n > 0) {
    const Eigen::FullPivLU<Eigen::MatrixXcd> lu(vectors);
    if (!lu.isInvertible())
      throw std::invalid_argument(
          "BandFollower: the operator is defective, so no band projector is "
          "available from its eigenbasis");
    inverse = lu.inverse();
    out.eigenvalues.assign(values.data(), values.data() + n);
    out.eigenvectors = toFlat(vectors);
    out.leftEigenvectors = toFlat(inverse);
  }

  if (!choose) {
    // Continuation: each band takes, in the declared order, the eigenvectors
    // of largest weight in its previous projector that no earlier band took.
    std::vector<bool> taken(static_cast<std::size_t>(n), false);
    for (const Reference &reference : reference_) {
      const Eigen::MatrixXcd previous = toMatrix(reference.projector);
      const Eigen::MatrixXcd weight = inverse * previous * vectors;
      std::vector<Eigen::Index> candidates;
      for (Eigen::Index k = 0; k < n; ++k)
        if (!taken[static_cast<std::size_t>(k)]) candidates.push_back(k);
      std::stable_sort(candidates.begin(), candidates.end(),
                       [&](Eigen::Index left, Eigen::Index right) {
                         return weight(left, left).real() >
                                weight(right, right).real();
                       });
      Selected band_;
      band_.declaredIndex = reference.declaredIndex;
      band_.occupation = reference.occupation;
      band_.declaredPositions = reference.declaredPositions;
      for (std::size_t k = 0; k < reference.rank && k < candidates.size();
           ++k) {
        band_.modes.push_back(candidates[k]);
        taken[static_cast<std::size_t>(candidates[k])] = true;
      }
      selected.push_back(std::move(band_));
    }
  }

  // The covariance, each band's projector and its report.
  Eigen::VectorXcd weights = Eigen::VectorXcd::Zero(n);
  std::vector<bool> occupied(static_cast<std::size_t>(n), false);
  for (std::size_t index = 0; index < selected.size(); ++index) {
    Selected &band = selected[index];
    const auto rank = band.modes.size();
    OccupiedBand record;
    record.declaredIndex = band.declaredIndex;
    record.occupation = band.occupation;
    record.rank = rank;
    record.declaredPositions = band.declaredPositions;
    for (const Eigen::Index mode : band.modes) {
      weights(mode) =
          complexd{band.occupation / static_cast<double>(rank), 0.0};
      occupied[static_cast<std::size_t>(mode)] = true;
      record.positions.push_back(placeOf[static_cast<std::size_t>(mode)]);
      record.modes.push_back(static_cast<std::size_t>(mode));
    }
    std::sort(record.positions.begin(), record.positions.end());
    for (const std::size_t place : record.positions)
      record.eigenvalues.push_back(out.ordered[place]);
    Eigen::MatrixXcd projector = Eigen::MatrixXcd::Zero(n, n);
    for (const Eigen::Index mode : band.modes)
      projector += vectors.col(mode) * inverse.row(mode);
    record.projector = toFlat(projector);
    if (index < reference_.size() && rank > 0) {
      const Eigen::MatrixXcd previous = toMatrix(reference_[index].projector);
      record.overlap =
          (projector * previous).trace() / static_cast<double>(rank);
    }
    record.crossed = record.positions != record.declaredPositions;
    out.bands.push_back(std::move(record));
  }
  // A band splits a degenerate group when one of its modes shares the group
  // with a mode outside it.
  for (auto &record : out.bands) {
    std::vector<bool> inBand(order.size(), false);
    for (const std::size_t place : record.positions) inBand[place] = true;
    for (const std::size_t place : record.positions) {
      const std::size_t group = groupOfPlace[place];
      const std::size_t end =
          group + 1 < starts.size() ? starts[group + 1] : order.size();
      for (std::size_t other = starts[group]; other < end; ++other)
        if (!inBand[other]) record.ambiguous = true;
    }
  }
  if (n > 0) {
    const Eigen::MatrixXcd gamma =
        vectors * weights.asDiagonal() * inverse;
    out.covariance = toFlat(gamma);
  }

  // The occupied span in the declared order: the declared mode count, or
  // every mode up to the end of the last group with a nonzero occupation.
  std::size_t span = occupiedModes_;
  if (rule_ == CovarianceRule::BandFilling) {
    span = 0;
    std::size_t end = 0;
    for (std::size_t band = 0;
         band < occupations_.size() && band < groupRanks.size(); ++band) {
      end += groupRanks[band];
      if (occupations_[band] != 0.0) span = end;
    }
  }
  out.occupiedEigenvalues.assign(
      out.ordered.begin(),
      out.ordered.begin() +
          static_cast<std::ptrdiff_t>(std::min(span, out.ordered.size())));
  out.spectralGap = span > 0 && span < out.ordered.size()
                        ? std::abs(out.ordered[span] - out.ordered[span - 1])
                        : kNaN;

  // The isolation of the occupied bands from the rest of the spectrum.
  double isolation = std::numeric_limits<double>::infinity();
  bool inside = false;
  bool outside = false;
  for (Eigen::Index k = 0; k < n; ++k) {
    if (!occupied[static_cast<std::size_t>(k)]) {
      outside = true;
      continue;
    }
    inside = true;
    for (Eigen::Index j = 0; j < n; ++j)
      if (!occupied[static_cast<std::size_t>(j)])
        isolation = std::min(isolation, std::abs(values(k) - values(j)));
  }
  out.bandIsolation = inside && outside ? isolation : kNaN;
  for (const auto &record : out.bands) {
    out.crossing = out.crossing || record.crossed;
    out.lowestOverlap = std::min(out.lowestOverlap, std::abs(record.overlap));
  }
  return out;
}

void BandFollower::follow(const BandRead &read) {
  std::vector<Reference> reference;
  reference.reserve(read.bands.size());
  for (const auto &band : read.bands) {
    Reference entry;
    entry.occupation = band.occupation;
    entry.rank = band.rank;
    entry.declaredIndex = band.declaredIndex;
    entry.declaredPositions = band.declaredPositions;
    entry.projector = band.projector;
    reference.push_back(std::move(entry));
  }
  reference_ = std::move(reference);
}

// ------------------------------------------------- SelfConsistentMeanField

SelfConsistentMeanField::SelfConsistentMeanField(
    JointAction action, SelfConsistentMeanFieldDeclaration declaration)
    : action_(std::move(action)), declaration_(std::move(declaration)) {
  // The follower checks the declared rule: occupations that are not numbers
  // of particles, no occupation at all, or a negative band tolerance.
  const BandFollower check(declaration_);
  (void)check;
}

SelfConsistentMeanFieldReport SelfConsistentMeanField::solve() {
  return solveJointNewton();
}

HolomorphicRelaxation SelfConsistentMeanField::jointSystem() const {
  auto follower = std::make_shared<BandFollower>(declaration_);
  JointAction action = action_;
  const BandRead start = follower->read(bandOperatorFlat(action, declaration_));
  follower->follow(start);
  action.setCovariance(start.covariance);
  (void)installFiberMoments(action, declaration_, start);
  HolomorphicRelaxationDeclaration newton = declaration_.geometry;
  newton.tolerance = std::min(declaration_.geometry.tolerance,
                              declaration_.tolerance);
  if (declaration_.fiberMoments > 0) newton.relaxMultipliers = true;
  CovarianceRebuild rebuild;
  rebuild.at = [follower, declaration = declaration_](const JointAction &point) {
    return rebuiltStateOf(
        follower->read(bandOperatorFlat(point, declaration)), declaration);
  };
  return HolomorphicRelaxation(action, newton, rebuild);
}

SelfConsistentMeanFieldReport SelfConsistentMeanField::solveJointNewton() {
  SelfConsistentMeanFieldReport report;
  report.bandSelection = declaration_.bandSelection;
  BandFollower follower(declaration_);
  const double startScale = largestSquaredLengthOf(action_);
  const std::vector<complexd> declared = action_.declaration().covariance;

  // Iterate zero: the bands are chosen at the starting point by the declared
  // order, and the covariance is their density there; the declared power sums
  // of the fiber they make up are pinned there.
  const BandRead start = follower.read(bandOperatorFlat(action_, declaration_));
  follower.follow(start);
  action_.setCovariance(start.covariance);
  report.fiberRank = installFiberMoments(action_, declaration_, start);
  report.fiberConstraintForm = declaration_.fiberConstraintForm;
  const bool fiber = declaration_.fiberMoments > 0;
  std::vector<SelfConsistentMeanFieldStep> steps;
  steps.push_back(measure(0, action_, start, start.covariance, declared,
                          declaration_.geometry));
  std::vector<complexd> previous = start.covariance;

  // Newton's method on the joint system: every residual, Jacobian column and
  // trial point reads the covariance rebuilt there from the bands followed
  // from the last accepted iterate, and every accepted iterate moves the
  // reference to its own bands.
  HolomorphicRelaxationDeclaration newton = declaration_.geometry;
  newton.tolerance = std::min(declaration_.geometry.tolerance,
                              declaration_.tolerance);
  // the pinned moments' multipliers are unknowns beside the geometry
  if (fiber) newton.relaxMultipliers = true;
  CovarianceRebuild rebuild;
  rebuild.at = [&follower, this](const JointAction &point) {
    return rebuiltStateOf(
        follower.read(bandOperatorFlat(point, declaration_)), declaration_);
  };
  rebuild.accepted = [&](const JointAction &point) {
    const BandRead read = follower.read(bandOperatorFlat(point, declaration_));
    follower.follow(read);
    steps.push_back(measure(steps.size(), point, read, read.covariance,
                            previous, declaration_.geometry));
    previous = read.covariance;
  };
  HolomorphicRelaxation relaxation(action_, newton, rebuild);
  const HolomorphicRelaxationReport joint = relaxation.solve();
  action_ = relaxation.action();

  // The Newton iteration taken from each iterate, and the joint residual
  // there.
  for (std::size_t k = 0; k < steps.size(); ++k) {
    SelfConsistentMeanFieldStep &step = steps[k];
    if (k < joint.steps.size()) {
      step.newtonIterated = true;
      step.newton = joint.steps[k];
      step.geometryResidualNorm = joint.steps[k].residualNorm;
    } else {
      step.geometryResidualNorm = joint.residualNorm;
    }
    step.geometryConverged = step.geometryResidualNorm <= newton.tolerance;
    if (k + 1 < steps.size()) {
      step.geometryStopReason = RelaxationStop::Continued;
    } else {
      step.geometryStopReason = joint.stopReason;
      step.geometryStopDetail = joint.stopDetail;
    }
  }
  report.iterations = steps.size() - 1;
  const double momentResidual = steps.back().momentResidualNorm;
  finishReport(report, std::move(steps), action_, declaration_, follower,
               startScale);
  report.converged = report.forceNorm <= declaration_.tolerance &&
                     momentResidual <= declaration_.tolerance;
  if (report.converged) {
    report.stopReason = RelaxationStop::Converged;
    report.stopDetail =
        "the self-consistent force " + threeDigits(report.forceNorm) +
        (fiber ? " and the pinned moments' residual " +
                     threeDigits(momentResidual) + " are"
               : std::string{" is"}) +
        " at or below the tolerance " + threeDigits(declaration_.tolerance) +
        " after " + std::to_string(report.iterations) + " Newton steps";
    if (joint.stopReason != RelaxationStop::Converged)
      report.stopDetail += "; the Newton solve then stopped: " +
                           relaxationStopName(joint.stopReason) + ", " +
                           joint.stopDetail;
  } else {
    report.stopReason = joint.stopReason;
    report.stopDetail = joint.stopDetail;
  }
  return report;
}

}  // namespace tessera::cobordism
