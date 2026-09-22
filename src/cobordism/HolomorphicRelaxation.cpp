// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "cobordism/HolomorphicRelaxation.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

#include <Eigen/Dense>

#include "mesh/Edge.h"
#include "mesh/EdgeList.h"
#include "spacetime/Spacetime.h"

namespace tessera::cobordism {

using complexd = std::complex<double>;

namespace {

/// \f$ 2\pi \f$, the full turn the contour nodes are spread over.
constexpr double kTwoPi = 6.28318530717958647692528676655900577;

/// Which blocks of the stationarity system are in scope, and where each starts
/// in the reduced residual and variable vectors.
///
/// Equations and variables are in bijection block by block — one length
/// equation per relaxed length, one link equation per relaxed link, one moment
/// equation per relaxed multiplier — so the reduced system is square and the
/// two layouts are one object.
struct Layout {
  std::size_t edges = 0;
  std::size_t constraints = 0;
  bool lengths = false;
  bool links = false;
  bool multipliers = false;
  std::size_t lengthOffset = 0;
  std::size_t linkOffset = 0;
  std::size_t multiplierOffset = 0;
  std::size_t count = 0;

  Layout(std::size_t edgeCount, std::size_t constraintCount,
         const HolomorphicRelaxationDeclaration &declaration)
      : edges(edgeCount),
        constraints(constraintCount),
        lengths(declaration.relaxLengths),
        links(declaration.relaxLinks),
        multipliers(declaration.relaxMultipliers && constraintCount > 0) {
    lengthOffset = count;
    if (lengths) count += edges;
    linkOffset = count;
    if (links) count += edges;
    multiplierOffset = count;
    if (multipliers) count += constraints;
  }
};

/// The state of every variable of the system, taken so that a perturbation for
/// one Jacobian column can be undone exactly rather than recomputed.
struct StateSnapshot {
  std::vector<complexd> lengths;
  std::vector<complexd> phases;
  std::vector<complexd> multipliers;
};

StateSnapshot takeSnapshot(const JointAction &action) {
  StateSnapshot snapshot;
  const auto &spacetime = action.spacetime();
  if (spacetime && spacetime->getEdgeList()) {
    for (const auto *edge : spacetime->getEdgeList()->toVector()) {
      snapshot.lengths.push_back(edge != nullptr ? edge->getLength()
                                                 : complexd{0.0, 0.0});
      snapshot.phases.push_back(edge != nullptr ? edge->getPhase()
                                                : complexd{0.0, 0.0});
    }
  }
  snapshot.multipliers = action.multipliers();
  return snapshot;
}

void restoreSnapshot(JointAction &action, const StateSnapshot &snapshot) {
  const auto &spacetime = action.spacetime();
  if (spacetime && spacetime->getEdgeList()) {
    const auto edges = spacetime->getEdgeList()->toVector();
    for (std::size_t index = 0;
         index < edges.size() && index < snapshot.lengths.size(); ++index) {
      if (edges[index] == nullptr) continue;
      edges[index]->setLength(snapshot.lengths[index]);
      edges[index]->setPhase(snapshot.phases[index]);
    }
  }
  action.setMultipliers(snapshot.multipliers);
}

/// The square root of a new squared length taken by continuation from the
/// current one.
///
/// The map \f$ \ell\mapsto\ell^2 \f$ is two to one, so a squared length does not
/// say which of \f$ \pm\ell \f$ an edge is. Keeping the root on the same side as
/// the edge's present length means a relaxation path stays on one sheet between
/// consecutive iterations instead of flipping whenever the principal branch cut
/// is crossed.
complexd continuedRoot(complexd squared, complexd currentLength) {
  const complexd root = std::sqrt(squared);
  if ((std::conj(currentLength) * root).real() < 0.0) return -root;
  return root;
}

/// The residual of exactly the equations in scope, in the layout's block order.
std::vector<complexd> reducedResidual(const JointAction &action,
                                      const Layout &layout) {
  std::vector<complexd> residual;
  residual.reserve(layout.count);
  if (layout.lengths) {
    const auto block = action.lengthStationarity();
    residual.insert(residual.end(), block.begin(), block.end());
  }
  if (layout.links) {
    const auto block = action.linkStationarity();
    residual.insert(residual.end(), block.begin(), block.end());
  }
  if (layout.multipliers) {
    const auto block = action.momentResiduals();
    residual.insert(residual.end(), block.begin(), block.end());
  }
  residual.resize(layout.count, complexd{0.0, 0.0});
  return residual;
}

double euclideanNorm(const std::vector<complexd> &vector) {
  double squared = 0.0;
  for (const complexd &component : vector) squared += std::norm(component);
  return std::sqrt(squared);
}

/// Move the length coordinate of one edge to \p squared, in place.
void writeSquaredLength(const JointAction &action, std::size_t edgeIndex,
                        complexd squared) {
  const auto &spacetime = action.spacetime();
  if (!spacetime || !spacetime->getEdgeList()) return;
  const auto edges = spacetime->getEdgeList()->toVector();
  if (edgeIndex >= edges.size() || edges[edgeIndex] == nullptr) return;
  edges[edgeIndex]->setLength(
      continuedRoot(squared, edges[edgeIndex]->getLength()));
}

/// Multiply the link of one edge by \f$ e^{\delta} \f$, in place.
///
/// The stored coordinate is the phase \f$ \varphi \f$ of \f$ U=e^{i\varphi} \f$,
/// and \f$ U e^{\delta}=e^{i(\varphi-i\delta)} \f$, so the multiplicative step
/// is the increment \f$ \varphi\mapsto\varphi-i\delta \f$ applied to the stored
/// value. Nothing here forms a logarithm of a link or asks for its argument.
void multiplyLink(const JointAction &action, std::size_t edgeIndex,
                  complexd increment) {
  const auto &spacetime = action.spacetime();
  if (!spacetime || !spacetime->getEdgeList()) return;
  const auto edges = spacetime->getEdgeList()->toVector();
  if (edgeIndex >= edges.size() || edges[edgeIndex] == nullptr) return;
  edges[edgeIndex]->setPhase(edges[edgeIndex]->getPhase() -
                             complexd{0.0, 1.0} * increment);
}

/// The contour nodes \f$ \rho\,\omega^{n} \f$ and the weights
/// \f$ \omega^{-n}/(m\rho) \f$ of the Cauchy derivative rule at radius
/// \p radius, or the two-node central difference when that mode is declared.
std::vector<std::pair<complexd, complexd>> derivativeRule(
    const HolomorphicRelaxationDeclaration &declaration, double radius) {
  std::vector<std::pair<complexd, complexd>> rule;
  const std::size_t nodes =
      declaration.jacobianMode == HolomorphicJacobianMode::CentralDifference
          ? 2
          : declaration.contourNodes;
  rule.reserve(nodes);
  for (std::size_t node = 0; node < nodes; ++node) {
    const double angle =
        kTwoPi * static_cast<double>(node) / static_cast<double>(nodes);
    const complexd root{std::cos(angle), std::sin(angle)};
    const complexd offset = radius * root;
    const complexd weight =
        std::conj(root) / (static_cast<double>(nodes) * radius);
    rule.emplace_back(offset, weight);
  }
  return rule;
}

}  // namespace

HolomorphicRelaxation::HolomorphicRelaxation(
    JointAction action, HolomorphicRelaxationDeclaration declaration)
    : action_(std::move(action)), declaration_(std::move(declaration)) {
  if (declaration_.contourNodes < 5)
    throw std::invalid_argument(
        "HolomorphicRelaxation: the contour rule keeps at least five nodes; "
        "got " +
        std::to_string(declaration_.contourNodes));
  if (!(declaration_.contourRadius > 0.0))
    throw std::invalid_argument(
        "HolomorphicRelaxation: the contour radius must be positive");
  if (!declaration_.relaxLengths && !declaration_.relaxLinks &&
      !declaration_.relaxMultipliers)
    throw std::invalid_argument(
        "HolomorphicRelaxation: no field is declared relaxable, so the solve "
        "has no variables");
}

std::size_t HolomorphicRelaxation::equationCount() const {
  return Layout(action_.edgeCount(), action_.constraintCount(), declaration_)
      .count;
}

std::size_t HolomorphicRelaxation::variableCount() const {
  return equationCount();
}

std::vector<complexd> HolomorphicRelaxation::jacobian() const {
  const Layout layout(action_.edgeCount(), action_.constraintCount(),
                      declaration_);
  // The solve writes the geometry, so the Jacobian is assembled on a mutable
  // copy of the action; the complex itself is restored exactly afterwards.
  JointAction working = action_;
  const StateSnapshot snapshot = takeSnapshot(working);
  Eigen::MatrixXcd matrix = Eigen::MatrixXcd::Zero(
      static_cast<Eigen::Index>(layout.count),
      static_cast<Eigen::Index>(layout.count));

  const auto &spacetime = working.spacetime();
  const auto edges = spacetime && spacetime->getEdgeList()
                         ? spacetime->getEdgeList()->toVector()
                         : std::vector<::tessera::mesh::Edge *>{};

  // One column per relaxed coordinate. Each node of the contour moves that one
  // coordinate, evaluates the whole reduced residual, and is weighted back into
  // the column; the geometry is restored exactly after every node, so the
  // columns are independent of the order they are taken in.
  if (layout.lengths) {
    for (std::size_t edgeIndex = 0; edgeIndex < layout.edges; ++edgeIndex) {
      if (edgeIndex >= edges.size() || edges[edgeIndex] == nullptr) continue;
      const complexd length = snapshot.lengths[edgeIndex];
      const complexd squared = length * length;
      const double radius =
          declaration_.contourRadius * std::max(1.0, std::abs(squared));
      const auto column =
          static_cast<Eigen::Index>(layout.lengthOffset + edgeIndex);
      for (const auto &node : derivativeRule(declaration_, radius)) {
        writeSquaredLength(working, edgeIndex, squared + node.first);
        const auto residual = reducedResidual(working, layout);
        for (std::size_t row = 0; row < residual.size(); ++row)
          matrix(static_cast<Eigen::Index>(row), column) +=
              node.second * residual[row];
        restoreSnapshot(working, snapshot);
      }
    }
  }

  if (layout.links) {
    for (std::size_t edgeIndex = 0; edgeIndex < layout.edges; ++edgeIndex) {
      if (edgeIndex >= edges.size() || edges[edgeIndex] == nullptr) continue;
      const auto column =
          static_cast<Eigen::Index>(layout.linkOffset + edgeIndex);
      for (const auto &node :
           derivativeRule(declaration_, declaration_.contourRadius)) {
        multiplyLink(working, edgeIndex, node.first);
        const auto residual = reducedResidual(working, layout);
        for (std::size_t row = 0; row < residual.size(); ++row)
          matrix(static_cast<Eigen::Index>(row), column) +=
              node.second * residual[row];
        restoreSnapshot(working, snapshot);
      }
    }
  }

  // The multiplier columns are exact and analytic: the action is linear in
  // every xi_j, so the column is the moment's own gradient and the moment rows
  // of it are zero.
  if (layout.multipliers) {
    for (std::size_t index = 0; index < layout.constraints; ++index) {
      const auto gradient = working.momentGradient(index);
      const auto column =
          static_cast<Eigen::Index>(layout.multiplierOffset + index);
      if (layout.lengths)
        for (std::size_t edgeIndex = 0; edgeIndex < layout.edges; ++edgeIndex)
          matrix(static_cast<Eigen::Index>(layout.lengthOffset + edgeIndex),
                 column) = gradient[edgeIndex];
      if (layout.links)
        for (std::size_t edgeIndex = 0; edgeIndex < layout.edges; ++edgeIndex)
          matrix(static_cast<Eigen::Index>(layout.linkOffset + edgeIndex),
                 column) = gradient[layout.edges + edgeIndex];
    }
  }

  restoreSnapshot(working, snapshot);

  std::vector<complexd> flat(layout.count * layout.count, complexd{0.0, 0.0});
  for (std::size_t row = 0; row < layout.count; ++row)
    for (std::size_t column = 0; column < layout.count; ++column)
      flat[row * layout.count + column] =
          matrix(static_cast<Eigen::Index>(row),
                 static_cast<Eigen::Index>(column));
  return flat;
}

HolomorphicRelaxationReport HolomorphicRelaxation::solve() {
  const Layout layout(action_.edgeCount(), action_.constraintCount(),
                      declaration_);
  HolomorphicRelaxationReport report;
  report.initialResidualNorm =
      euclideanNorm(reducedResidual(action_, layout));
  report.residualNorm = report.initialResidualNorm;
  report.action = action_.value();
  report.multipliers = action_.multipliers();
  report.momentResiduals = action_.momentResiduals();
  if (layout.count == 0) {
    report.converged = report.residualNorm <= declaration_.tolerance;
    return report;
  }

  for (std::size_t iteration = 0; iteration < declaration_.maximumIterations;
       ++iteration) {
    const auto residual = reducedResidual(action_, layout);
    const double residualNorm = euclideanNorm(residual);
    report.residualNorm = residualNorm;
    if (residualNorm <= declaration_.tolerance) {
      report.converged = true;
      break;
    }

    const auto flatJacobian = jacobian();
    Eigen::MatrixXcd matrix(static_cast<Eigen::Index>(layout.count),
                            static_cast<Eigen::Index>(layout.count));
    for (std::size_t row = 0; row < layout.count; ++row)
      for (std::size_t column = 0; column < layout.count; ++column)
        matrix(static_cast<Eigen::Index>(row),
               static_cast<Eigen::Index>(column)) =
            flatJacobian[row * layout.count + column];
    Eigen::VectorXcd target(static_cast<Eigen::Index>(layout.count));
    for (std::size_t row = 0; row < layout.count; ++row)
      target(static_cast<Eigen::Index>(row)) = -residual[row];

    // The minimum-norm least-squares step. The connection block is singular
    // along every pure-gauge direction because the action is gauge invariant,
    // so this decomposition is what makes the step well defined: it is the one
    // solution orthogonal to the gauge orbit.
    Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXcd> decomposition;
    decomposition.setThreshold(declaration_.rankTolerance);
    decomposition.compute(matrix);
    const Eigen::VectorXcd step = decomposition.solve(target);

    HolomorphicStep record;
    record.iteration = iteration;
    record.residualNorm = residualNorm;
    record.jacobianRank = static_cast<std::size_t>(decomposition.rank());
    record.action = action_.value();

    const StateSnapshot snapshot = takeSnapshot(action_);
    double damping = 1.0;
    bool accepted = false;
    for (std::size_t attempt = 0; attempt <= declaration_.maximumDampings;
         ++attempt) {
      restoreSnapshot(action_, snapshot);
      if (layout.lengths)
        for (std::size_t edgeIndex = 0; edgeIndex < layout.edges; ++edgeIndex) {
          const complexd length = snapshot.lengths[edgeIndex];
          writeSquaredLength(
              action_, edgeIndex,
              length * length +
                  damping * step(static_cast<Eigen::Index>(
                                layout.lengthOffset + edgeIndex)));
        }
      if (layout.links)
        for (std::size_t edgeIndex = 0; edgeIndex < layout.edges; ++edgeIndex)
          multiplyLink(action_, edgeIndex,
                       damping * step(static_cast<Eigen::Index>(
                                     layout.linkOffset + edgeIndex)));
      if (layout.multipliers) {
        auto multipliers = snapshot.multipliers;
        for (std::size_t index = 0; index < layout.constraints; ++index)
          multipliers[index] +=
              damping * step(static_cast<Eigen::Index>(
                            layout.multiplierOffset + index));
        action_.setMultipliers(multipliers);
      }
      const double trialNorm = euclideanNorm(reducedResidual(action_, layout));
      if (trialNorm < residualNorm) {
        accepted = true;
        record.damping = damping;
        record.stepNorm = damping * step.norm();
        report.residualNorm = trialNorm;
        break;
      }
      damping *= 0.5;
    }
    if (!accepted) {
      restoreSnapshot(action_, snapshot);
      report.steps.push_back(record);
      break;
    }
    report.steps.push_back(record);
  }

  report.converged = report.residualNorm <= declaration_.tolerance;
  report.action = action_.value();
  report.multipliers = action_.multipliers();
  report.momentResiduals = action_.momentResiduals();
  return report;
}

}  // namespace tessera::cobordism
