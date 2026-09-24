// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "cobordism/HolomorphicRelaxation.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <string>
#include <stdexcept>
#include <utility>

#include <Eigen/Dense>
#include <Eigen/SVD>

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
  /// The number of length coordinates, which is also the number of link
  /// coordinates: one per edge, or one per declared edge class.
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

/// The coordinates the lengths and links are carried in: for each, the edges
/// that carry it and the orientation of each such edge's stored link relative
/// to the coordinate's link.
struct EdgeClasses {
  std::size_t edgeCount = 0;
  std::vector<std::vector<std::pair<std::size_t, int>>> members;
  std::size_t count() const { return members.size(); }
};

EdgeClasses edgeClassesOf(std::size_t edgeCount,
                          const HolomorphicRelaxationDeclaration &declaration) {
  EdgeClasses classes;
  classes.edgeCount = edgeCount;
  if (declaration.edgeClasses.empty()) {
    classes.members.resize(edgeCount);
    for (std::size_t edge = 0; edge < edgeCount; ++edge)
      classes.members[edge].emplace_back(edge, 1);
    return classes;
  }
  if (declaration.edgeClasses.size() != edgeCount)
    throw std::invalid_argument(
        "HolomorphicRelaxation: " +
        std::to_string(declaration.edgeClasses.size()) +
        " edge classes were declared for " + std::to_string(edgeCount) +
        " edges");
  if (!declaration.edgeClassOrientations.empty() &&
      declaration.edgeClassOrientations.size() != edgeCount)
    throw std::invalid_argument(
        "HolomorphicRelaxation: " +
        std::to_string(declaration.edgeClassOrientations.size()) +
        " edge class orientations were declared for " +
        std::to_string(edgeCount) + " edges");
  std::size_t count = 0;
  for (const std::size_t index : declaration.edgeClasses)
    count = std::max(count, index + 1);
  classes.members.resize(count);
  for (std::size_t edge = 0; edge < edgeCount; ++edge) {
    const int orientation = declaration.edgeClassOrientations.empty()
                                ? 1
                                : declaration.edgeClassOrientations[edge];
    if (orientation != 1 && orientation != -1)
      throw std::invalid_argument(
          "HolomorphicRelaxation: the orientation of edge " +
          std::to_string(edge) +
          " relative to its class must be plus or minus one; got " +
          std::to_string(orientation));
    classes.members[declaration.edgeClasses[edge]].emplace_back(edge,
                                                                orientation);
  }
  for (std::size_t index = 0; index < count; ++index)
    if (classes.members[index].empty())
      throw std::invalid_argument(
          "HolomorphicRelaxation: edge class " + std::to_string(index) +
          " has no edge; the classes must be numbered 0 to K - 1 without a "
          "gap");
  return classes;
}

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

/// Put one edge's two fields back to the values the snapshot holds.
///
/// A Jacobian column moves exactly one coordinate, so only that coordinate has
/// to be undone. Writing every edge back instead would bump every edge's
/// revision counter and discard the operator caches keyed on them, which costs
/// a full reassembly per contour node rather than per column.
void restoreEdge(const JointAction &action, const StateSnapshot &snapshot,
                 std::size_t edgeIndex) {
  const auto &spacetime = action.spacetime();
  if (!spacetime || !spacetime->getEdgeList()) return;
  const auto edges = spacetime->getEdgeList()->toVector();
  if (edgeIndex >= edges.size() || edges[edgeIndex] == nullptr) return;
  if (edgeIndex < snapshot.lengths.size())
    edges[edgeIndex]->setLength(snapshot.lengths[edgeIndex]);
  if (edgeIndex < snapshot.phases.size())
    edges[edgeIndex]->setPhase(snapshot.phases[edgeIndex]);
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
///
/// The equation of a length or link coordinate is the sum of the equations of
/// the edges that carry it, each link equation taken on the coordinate's
/// orientation: the derivative of the action along the shared coordinate.
std::vector<complexd> reducedResidual(const JointAction &action,
                                      const Layout &layout,
                                      const EdgeClasses &classes) {
  std::vector<complexd> residual;
  residual.reserve(layout.count);
  if (layout.lengths) {
    const auto block = action.lengthStationarity();
    for (const auto &members : classes.members) {
      complexd sum{0.0, 0.0};
      for (const auto &[edge, orientation] : members)
        if (edge < block.size()) sum += block[edge];
      residual.push_back(sum);
    }
  }
  if (layout.links) {
    const auto block = action.linkStationarity();
    for (const auto &members : classes.members) {
      complexd sum{0.0, 0.0};
      for (const auto &[edge, orientation] : members)
        if (edge < block.size())
          sum += orientation > 0 ? block[edge] : -block[edge];
      residual.push_back(sum);
    }
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
  if (declaration.jacobianMode == HolomorphicJacobianMode::RealAxisDifference) {
    // Both nodes as exact real numbers. Writing them as rho * exp(i pi n)
    // instead would give the second one an imaginary part of about
    // 1e-16 * rho, which is enough to land it on the far side of a cut that
    // runs along the real axis and to read a different branch there.
    rule.emplace_back(complexd{radius, 0.0},
                      complexd{1.0 / (2.0 * radius), 0.0});
    rule.emplace_back(complexd{-radius, 0.0},
                      complexd{-1.0 / (2.0 * radius), 0.0});
    return rule;
  }
  const std::size_t nodes = declaration.contourNodes;
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

/// The held monopole sectors, resolved against the mesh: for every held face
/// the stored edges it runs along with their orientation signs, the real
/// coboundary rows of the held faces over the link increments, and the
/// projector onto their row space in the solve's link coordinates.
struct SectorGeometry {
  /// Per sector, per face, the (edge index, sign) of its three sides: the face
  /// holonomy is the product of \f$ U_e^{\rm sign} \f$.
  std::vector<std::vector<std::array<std::pair<std::size_t, int>, 3>>> faces;
  /// Projector onto the row space of the held faces' coboundary, pulled back
  /// to the solve's link coordinates (coordinates by coordinates, real): the
  /// part of a link increment's real component that changes a held modulus.
  Eigen::MatrixXd modulusProjector;
  bool empty = true;
};

SectorGeometry resolveSectors(const JointAction &action,
                              const std::vector<HeldMonopoleSector> &sectors,
                              const EdgeClasses &classes) {
  SectorGeometry geometry;
  if (sectors.empty()) return geometry;
  const auto &spacetime = action.spacetime();
  if (!spacetime || !spacetime->getEdgeList())
    throw std::invalid_argument(
        "HolomorphicRelaxation: held monopole sectors need a mesh");
  const auto edges = spacetime->getEdgeList()->toVector();
  std::map<std::pair<std::uint64_t, std::uint64_t>, std::pair<std::size_t, int>> lookup;
  for (std::size_t index = 0; index < edges.size(); ++index) {
    if (edges[index] == nullptr) continue;
    const auto a = edges[index]->getSource()->getId();
    const auto b = edges[index]->getTarget()->getId();
    lookup[{a, b}] = {index, 1};
    lookup[{b, a}] = {index, -1};
  }
  std::vector<std::array<std::pair<std::size_t, int>, 3>> rows;
  for (const auto &sector : sectors) {
    geometry.faces.emplace_back();
    for (const auto &face : sector.faces) {
      std::array<std::pair<std::size_t, int>, 3> sides{};
      for (int k = 0; k < 3; ++k) {
        const auto found = lookup.find({face[k], face[(k + 1) % 3]});
        if (found == lookup.end())
          throw std::invalid_argument(
              "HolomorphicRelaxation: a held face runs along an edge the mesh "
              "does not have (" + std::to_string(face[k]) + ", " +
              std::to_string(face[(k + 1) % 3]) + ")");
        sides[k] = found->second;
      }
      geometry.faces.back().push_back(sides);
      rows.push_back(sides);
    }
  }
  Eigen::MatrixXd coboundary = Eigen::MatrixXd::Zero(
      static_cast<Eigen::Index>(rows.size()),
      static_cast<Eigen::Index>(edges.size()));
  for (std::size_t r = 0; r < rows.size(); ++r)
    for (const auto &side : rows[r])
      coboundary(static_cast<Eigen::Index>(r),
                 static_cast<Eigen::Index>(side.first)) += side.second;
  // A link coordinate moves every edge of its class, each on its own
  // orientation, so the rows are pulled back through that expansion.
  Eigen::MatrixXd expansion = Eigen::MatrixXd::Zero(
      static_cast<Eigen::Index>(edges.size()),
      static_cast<Eigen::Index>(classes.count()));
  for (std::size_t index = 0; index < classes.count(); ++index)
    for (const auto &[edge, orientation] : classes.members[index])
      if (edge < edges.size())
        expansion(static_cast<Eigen::Index>(edge),
                  static_cast<Eigen::Index>(index)) = orientation;
  coboundary = coboundary * expansion;
  const Eigen::JacobiSVD<Eigen::MatrixXd> svd(coboundary, Eigen::ComputeThinV);
  const auto &values = svd.singularValues();
  const double cut = values.size() > 0 ? 1e-10 * values(0) : 0.0;
  Eigen::Index rank = 0;
  while (rank < values.size() && values(rank) > cut) ++rank;
  const Eigen::MatrixXd basis = svd.matrixV().leftCols(rank);
  geometry.modulusProjector = basis * basis.transpose();
  geometry.empty = false;
  return geometry;
}

complexd faceHolonomy(const std::vector<::tessera::mesh::Edge *> &edges,
                      const std::array<std::pair<std::size_t, int>, 3> &sides) {
  complexd product{1.0, 0.0};
  for (const auto &side : sides) {
    const complexd link =
        std::exp(complexd{0.0, 1.0} * edges[side.first]->getPhase());
    product *= side.second > 0 ? link : 1.0 / link;
  }
  return product;
}

/// The monopole number of every held sector: the sum of the principal
/// arguments of its outward face holonomies over \f$ 2\pi \f$.
std::vector<int> sectorNumbers(const JointAction &action,
                               const SectorGeometry &geometry) {
  std::vector<int> numbers;
  if (geometry.empty) return numbers;
  const auto edges = action.spacetime()->getEdgeList()->toVector();
  for (const auto &sector : geometry.faces) {
    double total = 0.0;
    for (const auto &sides : sector) total += std::arg(faceHolonomy(edges, sides));
    numbers.push_back(static_cast<int>(std::lround(total / kTwoPi)));
  }
  return numbers;
}

std::vector<double> heldLogModuli(const JointAction &action,
                                  const SectorGeometry &geometry) {
  std::vector<double> moduli;
  if (geometry.empty) return moduli;
  const auto edges = action.spacetime()->getEdgeList()->toVector();
  for (const auto &sector : geometry.faces)
    for (const auto &sides : sector)
      moduli.push_back(std::log(std::abs(faceHolonomy(edges, sides))));
  return moduli;
}

/// The edges of a class put back to the values the snapshot holds.
void restoreClass(const JointAction &action, const StateSnapshot &snapshot,
                  const std::vector<std::pair<std::size_t, int>> &members) {
  for (const auto &member : members) restoreEdge(action, snapshot, member.first);
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
  if (!(declaration_.holonomyZeroMargin > 0.0))
    throw std::invalid_argument(
        "HolomorphicRelaxation: the holonomy zero margin must be positive");
  if (!declaration_.relaxLengths && !declaration_.relaxLinks &&
      !declaration_.relaxMultipliers)
    throw std::invalid_argument(
        "HolomorphicRelaxation: no field is declared relaxable, so the solve "
        "has no variables");
  const EdgeClasses classes =
      edgeClassesOf(action_.edgeCount(), declaration_);
  if (!declaration_.edgeClasses.empty()) {
    // The members of a class are one coordinate, so they must start equal:
    // the solve writes one value to all of them and would otherwise carry the
    // starting difference along unseen.
    const StateSnapshot snapshot = takeSnapshot(action_);
    for (std::size_t index = 0; index < classes.count(); ++index) {
      const auto &members = classes.members[index];
      const auto &[first, firstOrientation] = members.front();
      if (first >= snapshot.lengths.size()) continue;
      for (const auto &[edge, orientation] : members) {
        if (edge >= snapshot.lengths.size()) continue;
        const bool sameLength = snapshot.lengths[edge] * snapshot.lengths[edge] ==
                                snapshot.lengths[first] * snapshot.lengths[first];
        const complexd phase =
            orientation == firstOrientation ? snapshot.phases[first]
                                            : -snapshot.phases[first];
        if (!sameLength || snapshot.phases[edge] != phase)
          throw std::invalid_argument(
              "HolomorphicRelaxation: edges " + std::to_string(first) +
              " and " + std::to_string(edge) + " share class " +
              std::to_string(index) +
              " but do not carry equal squared lengths and equal links");
      }
    }
  }
  if (!declaration_.heldSectors.empty()) {
    const SectorGeometry geometry =
        resolveSectors(action_, declaration_.heldSectors, classes);
    const auto numbers = sectorNumbers(action_, geometry);
    for (std::size_t index = 0; index < numbers.size(); ++index)
      if (numbers[index] != declaration_.heldSectors[index].monopoleNumber)
        throw std::invalid_argument(
            "HolomorphicRelaxation: held sector " + std::to_string(index) +
            " is declared with monopole number " +
            std::to_string(declaration_.heldSectors[index].monopoleNumber) +
            " but the starting configuration carries " +
            std::to_string(numbers[index]));
  }
}

std::size_t HolomorphicRelaxation::equationCount() const {
  return Layout(edgeClassesOf(action_.edgeCount(), declaration_).count(),
                action_.constraintCount(), declaration_)
      .count;
}

std::size_t HolomorphicRelaxation::variableCount() const {
  return equationCount();
}

std::vector<complexd> HolomorphicRelaxation::jacobian() const {
  const EdgeClasses classes =
      edgeClassesOf(action_.edgeCount(), declaration_);
  const Layout layout(classes.count(), action_.constraintCount(),
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
  // coordinate (every edge of its class), evaluates the whole reduced
  // residual, and is weighted back into the column; the geometry is restored
  // exactly after every node, so the columns are independent of the order
  // they are taken in.
  if (layout.lengths) {
    for (std::size_t index = 0; index < layout.edges; ++index) {
      const auto &members = classes.members[index];
      const std::size_t first = members.front().first;
      if (first >= edges.size() || edges[first] == nullptr) continue;
      const complexd length = snapshot.lengths[first];
      const complexd squared = length * length;
      const double radius =
          declaration_.contourRadius * std::max(1.0, std::abs(squared));
      const auto column = static_cast<Eigen::Index>(layout.lengthOffset + index);
      for (const auto &node : derivativeRule(declaration_, radius)) {
        for (const auto &member : members)
          writeSquaredLength(working, member.first, squared + node.first);
        const auto residual = reducedResidual(working, layout, classes);
        for (std::size_t row = 0; row < residual.size(); ++row)
          matrix(static_cast<Eigen::Index>(row), column) +=
              node.second * residual[row];
        restoreClass(working, snapshot, members);
      }
    }
  }

  if (layout.links) {
    for (std::size_t index = 0; index < layout.edges; ++index) {
      const auto &members = classes.members[index];
      const std::size_t first = members.front().first;
      if (first >= edges.size() || edges[first] == nullptr) continue;
      const auto column = static_cast<Eigen::Index>(layout.linkOffset + index);
      for (const auto &node :
           derivativeRule(declaration_, declaration_.contourRadius)) {
        for (const auto &[edge, orientation] : members)
          multiplyLink(working, edge,
                       orientation > 0 ? node.first : -node.first);
        const auto residual = reducedResidual(working, layout, classes);
        for (std::size_t row = 0; row < residual.size(); ++row)
          matrix(static_cast<Eigen::Index>(row), column) +=
              node.second * residual[row];
        restoreClass(working, snapshot, members);
      }
    }
  }

  // The multiplier columns are exact and analytic: the action is linear in
  // every xi_j, so the column is the moment's own gradient, summed over each
  // class, and the moment rows of it are zero.
  if (layout.multipliers) {
    const std::size_t edgeCount = classes.edgeCount;
    for (std::size_t constraint = 0; constraint < layout.constraints;
         ++constraint) {
      const auto gradient = working.momentGradient(constraint);
      const auto column =
          static_cast<Eigen::Index>(layout.multiplierOffset + constraint);
      for (std::size_t index = 0; index < layout.edges; ++index) {
        complexd lengthPart{0.0, 0.0};
        complexd linkPart{0.0, 0.0};
        for (const auto &[edge, orientation] : classes.members[index]) {
          lengthPart += gradient[edge];
          linkPart += orientation > 0 ? gradient[edgeCount + edge]
                                      : -gradient[edgeCount + edge];
        }
        if (layout.lengths)
          matrix(static_cast<Eigen::Index>(layout.lengthOffset + index),
                 column) = lengthPart;
        if (layout.links)
          matrix(static_cast<Eigen::Index>(layout.linkOffset + index),
                 column) = linkPart;
      }
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
  const EdgeClasses classes =
      edgeClassesOf(action_.edgeCount(), declaration_);
  const Layout layout(classes.count(), action_.constraintCount(),
                      declaration_);
  HolomorphicRelaxationReport report;
  const SectorGeometry sectors =
      resolveSectors(action_, declaration_.heldSectors, classes);
  const std::vector<double> startModuli = heldLogModuli(action_, sectors);
  std::vector<int> declaredNumbers;
  for (const auto &sector : declaration_.heldSectors)
    declaredNumbers.push_back(sector.monopoleNumber);
  report.reggeHingeCount = action_.reggeHingeCount();
  report.reggeStructurallyZero = action_.reggeStructurallyZero();
  report.initialResidualNorm =
      euclideanNorm(reducedResidual(action_, layout, classes));
  report.residualNorm = report.initialResidualNorm;
  report.action = action_.value();
  report.multipliers = action_.multipliers();
  report.momentResiduals = action_.momentResiduals();
  report.reggeOffPrincipalAngles = action_.reggeOffPrincipalAngles();
  if (layout.count == 0) {
    report.converged = report.residualNorm <= declaration_.tolerance;
    report.sectorMonopoleNumbers = sectorNumbers(action_, sectors);
    return report;
  }

  for (std::size_t iteration = 0; iteration < declaration_.maximumIterations;
       ++iteration) {
    const auto residual = reducedResidual(action_, layout, classes);
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

    // The minimum-norm least-squares step from the singular value
    // decomposition. The connection block is singular along every pure-gauge
    // direction because the action is gauge invariant, so this is what makes
    // the step well defined: it is the one solution orthogonal to the
    // numerical null space. The rank is decided on the singular values,
    // relative to the largest, at the declared tolerance.
    // Two-sided Jacobi: every singular value accurate to rounding relative to
    // the largest, which the rank decision reads. The divide-and-conquer SVD
    // was measured to report a spurious singular value of 6e-10 (true value
    // below 1e-12) on a 72-coordinate level Jacobian, above a 1e-10 threshold.
    const Eigen::JacobiSVD<Eigen::MatrixXcd> svd(
        matrix, Eigen::ComputeThinU | Eigen::ComputeThinV);
    const Eigen::VectorXd &singular = svd.singularValues();
    const double largest = singular.size() > 0 ? singular(0) : 0.0;
    const double cut = declaration_.rankTolerance * largest;
    Eigen::Index rank = 0;
    while (rank < singular.size() && singular(rank) > cut) ++rank;
    Eigen::VectorXcd step =
        Eigen::VectorXcd::Zero(static_cast<Eigen::Index>(layout.count));
    if (rank > 0) {
      Eigen::VectorXcd coefficients =
          svd.matrixU().leftCols(rank).adjoint() * target;
      for (Eigen::Index k = 0; k < rank; ++k) coefficients(k) /= singular(k);
      step = svd.matrixV().leftCols(rank) * coefficients;
    }

    // Held sectors: remove from the link step the part of its real (modulus)
    // component that would change the modulus of a held face holonomy.
    if (layout.links && !sectors.empty) {
      Eigen::VectorXd modulus(static_cast<Eigen::Index>(layout.edges));
      for (std::size_t index = 0; index < layout.edges; ++index)
        modulus(static_cast<Eigen::Index>(index)) =
            step(static_cast<Eigen::Index>(layout.linkOffset + index)).real();
      const Eigen::VectorXd removed = sectors.modulusProjector * modulus;
      for (std::size_t index = 0; index < layout.edges; ++index)
        step(static_cast<Eigen::Index>(layout.linkOffset + index)) -=
            complexd{removed(static_cast<Eigen::Index>(index)), 0.0};
    }

    HolomorphicStep record;
    record.iteration = iteration;
    record.residualNorm = residualNorm;
    record.jacobianRank = static_cast<std::size_t>(rank);
    record.rankTolerance = declaration_.rankTolerance;
    record.largestSingularValue = largest;
    record.smallestRetainedSingularValue =
        rank > 0 ? singular(rank - 1) : std::numeric_limits<double>::quiet_NaN();
    record.largestDiscardedSingularValue =
        rank < singular.size() ? singular(rank) : 0.0;
    record.rankGap = rank < singular.size()
                         ? record.smallestRetainedSingularValue /
                               record.largestDiscardedSingularValue
                         : std::numeric_limits<double>::infinity();
    record.action = action_.value();
    record.holonomyZeroDistance = action_.holonomyZeroDistance();

    // The link part of the step on every edge, for the holonomy zero guard.
    std::vector<complexd> linkStep;
    if (layout.links) {
      linkStep.assign(classes.edgeCount, complexd{0.0, 0.0});
      for (std::size_t index = 0; index < layout.edges; ++index)
        for (const auto &[edge, orientation] : classes.members[index]) {
          const complexd value =
              step(static_cast<Eigen::Index>(layout.linkOffset + index));
          linkStep[edge] = orientation > 0 ? value : -value;
        }
    }
    const bool guarded =
        layout.links && std::isfinite(record.holonomyZeroDistance);

    const StateSnapshot snapshot = takeSnapshot(action_);
    double damping = 1.0;
    bool accepted = false;
    for (std::size_t attempt = 0; attempt <= declaration_.maximumDampings;
         ++attempt) {
      restoreSnapshot(action_, snapshot);
      if (guarded) {
        std::vector<complexd> increments(linkStep.size());
        for (std::size_t edgeIndex = 0; edgeIndex < linkStep.size();
             ++edgeIndex)
          increments[edgeIndex] = damping * linkStep[edgeIndex];
        const double clearance = action_.holonomyZeroClearance(
            increments, 0.25 * declaration_.holonomyZeroMargin);
        if (clearance < declaration_.holonomyZeroMargin) {
          ++record.zeroGuardDampings;
          damping *= 0.5;
          continue;
        }
      }
      // One value per coordinate, written to every edge of its class: the
      // members of a class stay equal exactly.
      if (layout.lengths)
        for (std::size_t index = 0; index < layout.edges; ++index) {
          const auto &members = classes.members[index];
          const complexd length = snapshot.lengths[members.front().first];
          const complexd squared =
              length * length +
              damping * step(static_cast<Eigen::Index>(layout.lengthOffset +
                                                       index));
          for (const auto &member : members)
            writeSquaredLength(action_, member.first, squared);
        }
      if (layout.links)
        for (std::size_t index = 0; index < layout.edges; ++index) {
          const complexd increment =
              damping *
              step(static_cast<Eigen::Index>(layout.linkOffset + index));
          for (const auto &[edge, orientation] : classes.members[index])
            multiplyLink(action_, edge,
                         orientation > 0 ? increment : -increment);
        }
      if (layout.multipliers) {
        auto multipliers = snapshot.multipliers;
        for (std::size_t index = 0; index < layout.constraints; ++index)
          multipliers[index] +=
              damping * step(static_cast<Eigen::Index>(
                            layout.multiplierOffset + index));
        action_.setMultipliers(multipliers);
      }
      if (!sectors.empty && sectorNumbers(action_, sectors) != declaredNumbers) {
        ++record.sectorGuardDampings;
        damping *= 0.5;
        continue;
      }
      // A trial point the action refuses to evaluate (a link driven to zero
      // or to infinity by an overflowing multiplicative step, a face holonomy
      // outside the domain of the holonomy term) has left the domain the
      // equations are posed on; it is halved like a step that does not
      // reduce the residual. The equations are unchanged.
      double trialNorm = 0.0;
      try {
        trialNorm = euclideanNorm(reducedResidual(action_, layout, classes));
      } catch (const std::invalid_argument &) {
        ++record.domainGuardDampings;
        damping *= 0.5;
        continue;
      }
      if (!std::isfinite(trialNorm)) {
        ++record.domainGuardDampings;
        damping *= 0.5;
        continue;
      }
      if (trialNorm < residualNorm) {
        accepted = true;
        record.damping = damping;
        record.stepNorm = damping * step.norm();
        report.residualNorm = trialNorm;
        break;
      }
      damping *= 0.5;
    }
    if (record.zeroGuardDampings > 0) ++report.zeroGuardDampedSteps;
    if (record.sectorGuardDampings > 0) ++report.sectorGuardDampedSteps;
    if (!accepted) {
      restoreSnapshot(action_, snapshot);
      report.steps.push_back(record);
      break;
    }
    report.steps.push_back(record);
  }

  report.converged = report.residualNorm <= declaration_.tolerance;
  report.sectorMonopoleNumbers = sectorNumbers(action_, sectors);
  const std::vector<double> endModuli = heldLogModuli(action_, sectors);
  for (std::size_t index = 0; index < endModuli.size(); ++index)
    report.heldModulusDrift = std::max(
        report.heldModulusDrift, std::abs(endModuli[index] - startModuli[index]));
  report.action = action_.value();
  report.multipliers = action_.multipliers();
  report.momentResiduals = action_.momentResiduals();
  report.reggeOffPrincipalAngles = action_.reggeOffPrincipalAngles();
  return report;
}

}  // namespace tessera::cobordism
