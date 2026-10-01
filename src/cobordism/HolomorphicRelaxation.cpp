// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "cobordism/HolomorphicRelaxation.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
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

/// \f$ 2\pi \f$, the full turn a monopole number is read in units of.
constexpr double kTwoPi = 6.28318530717958647692528676655900577;

/// A flat row-major matrix lifted into Eigen.
Eigen::MatrixXcd toMatrix(const std::vector<complexd> &flat, std::size_t order) {
  Eigen::MatrixXcd matrix(static_cast<Eigen::Index>(order),
                          static_cast<Eigen::Index>(order));
  for (std::size_t i = 0; i < order; ++i)
    for (std::size_t j = 0; j < order; ++j)
      matrix(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) =
          flat[i * order + j];
  return matrix;
}

/// An Eigen matrix flattened back to the row-major layout.
std::vector<complexd> toFlat(const Eigen::MatrixXcd &matrix) {
  const auto rows = static_cast<std::size_t>(matrix.rows());
  const auto columns = static_cast<std::size_t>(matrix.cols());
  std::vector<complexd> flat(rows * columns, complexd{0.0, 0.0});
  for (std::size_t i = 0; i < rows; ++i)
    for (std::size_t j = 0; j < columns; ++j)
      flat[i * columns + j] = matrix(static_cast<Eigen::Index>(i),
                                     static_cast<Eigen::Index>(j));
  return flat;
}

/// A number as text with three significant digits (printf's "%.3g"), for the
/// stop details.
std::string threeDigits(double value) {
  char buffer[32];
  std::snprintf(buffer, sizeof buffer, "%.3g", value);
  return buffer;
}

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

/// The state of every variable of the system, taken so that a refused trial
/// step can be undone exactly rather than recomputed.
struct StateSnapshot {
  std::vector<complexd> lengths;
  std::vector<complexd> phases;
  std::vector<complexd> multipliers;
  /// The carried covariance and the constrained fiber's projector, which a
  /// self-consistent solve rebuilds at every point and so must restore with
  /// the geometry.
  std::vector<complexd> covariance;
  std::vector<complexd> momentProjector;
  std::vector<std::vector<complexd>> momentBandProjectors;
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
  snapshot.covariance = action.declaration().covariance;
  snapshot.momentProjector = action.declaration().momentProjector;
  snapshot.momentBandProjectors = action.declaration().momentBandProjectors;
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
  if (action.declaration().covariance != snapshot.covariance)
    action.setCovariance(snapshot.covariance);
  if (action.declaration().momentProjector != snapshot.momentProjector)
    action.setMomentProjector(snapshot.momentProjector);
  if (action.declaration().momentBandProjectors !=
      snapshot.momentBandProjectors)
    action.setMomentBandProjectors(snapshot.momentBandProjectors);
}

/// Set on \p target the rebuilt \p state: the covariance and, when the
/// rebuild constrains a fiber, its projector and the projectors of the bands
/// whose means are pinned.
void applyState(const RebuiltCarrierState &state, JointAction &target) {
  target.setCovariance(state.covariance);
  if (!state.momentProjector.empty())
    target.setMomentProjector(state.momentProjector);
  if (!state.bandProjectors.empty())
    target.setMomentBandProjectors(state.bandProjectors);
}

/// Set on \p target the state a declared rebuild gives at its current point.
void applyRebuild(const CovarianceRebuild &rebuild, JointAction &target) {
  if (!rebuild.at) return;
  applyState(rebuild.at(target), target);
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

/// The held monopole sectors, resolved against the mesh: for every held face
/// the stored edges it runs along with their orientation signs, and the
/// link-modulus directions that the held faces' real coboundary rows leave
/// free, in the solve's link coordinates.
struct SectorGeometry {
  /// Per sector, per face, the (edge index, sign) of its three sides: the face
  /// holonomy is the product of \f$ U_e^{\rm sign} \f$.
  std::vector<std::vector<std::array<std::pair<std::size_t, int>, 3>>> faces;
  /// An orthonormal basis (one column each) of the kernel of the held faces'
  /// coboundary, pulled back to the solve's link coordinates: the real parts
  /// of a link increment that change no held modulus.
  Eigen::MatrixXd freeModuli;
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
  const Eigen::JacobiSVD<Eigen::MatrixXd> svd(coboundary, Eigen::ComputeFullV);
  const auto &values = svd.singularValues();
  const double cut = values.size() > 0 ? 1e-10 * values(0) : 0.0;
  Eigen::Index rank = 0;
  while (rank < values.size() && values(rank) > cut) ++rank;
  geometry.freeModuli =
      svd.matrixV().rightCols(svd.matrixV().cols() - rank);
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

/// The largest \f$ |z_e| \f$ over the length coordinates, each read on the
/// first edge of its class.
double largestSquaredLength(const JointAction &action,
                            const EdgeClasses &classes) {
  const auto &spacetime = action.spacetime();
  if (!spacetime || !spacetime->getEdgeList()) return 0.0;
  const auto edges = spacetime->getEdgeList()->toVector();
  double largest = 0.0;
  for (const auto &members : classes.members) {
    const std::size_t first = members.front().first;
    if (first >= edges.size() || edges[first] == nullptr) continue;
    const complexd length = edges[first]->getLength();
    largest = std::max(largest, std::abs(length * length));
  }
  return largest;
}

/// The Newton step on the tangent space of the held set, and the rank
/// decision it was solved with.
struct ConstrainedStep {
  Eigen::VectorXcd step;
  std::size_t rank = 0;
  double gap = std::numeric_limits<double>::infinity();
};

/// The minimum-norm least-squares solution of \f$ Jd=\text{target} \f$ over
/// the steps that keep the held moduli.
///
/// A step \f$ d \f$ keeps them exactly when the real part of its link block
/// lies in the kernel of the held faces' coboundary, so it is parametrized by
/// real unknowns: the real and imaginary parts of every length and multiplier
/// step, the imaginary part (the phase) of every link step, and the
/// coefficients of the link step's real part on the orthonormal basis
/// \f$ K \f$ of that kernel. The map from the unknowns to \f$ d \f$ is an
/// isometry of the real inner product, so the minimum-norm unknowns give the
/// minimum-norm step. The complex equations are split into their real and
/// imaginary parts, and the real system is solved from its singular value
/// decomposition in the minimum-norm sense.
///
/// Its rank is decided at the Jacobian's own rank boundary \p cut. For a
/// unit \f$ y \f$, \f$ \lVert JBy\rVert\ge\sigma_{\min}(J) \f$, so every
/// singular value of the real system below the Jacobian's smallest retained
/// singular value comes from a direction of the held set that lies in the
/// Jacobian's numerical null space, and there are at most twice its nullity
/// of them. The gauge directions are such directions: they change no face
/// holonomy, so they lie in the held set, and the Jacobian is singular along
/// them. The Jacobian's null vector along a gauge direction is exact to
/// rounding, so along that direction in the held set the real system reads a
/// singular value at rounding level, far below the Jacobian's physical
/// spectrum. Cutting at the Jacobian's rank boundary counts it as zero, as
/// the Jacobian's own rank decision does.
ConstrainedStep constrainedStep(const Eigen::MatrixXcd &jacobian,
                                const Eigen::VectorXcd &target,
                                const Layout &layout,
                                const Eigen::MatrixXd &freeModuli,
                                double cut) {
  const Eigen::Index n = jacobian.rows();
  const complexd one{1.0, 0.0};
  const complexd imaginary{0.0, 1.0};
  std::vector<Eigen::VectorXcd> basis;
  auto unit = [&](std::size_t index, complexd value) {
    Eigen::VectorXcd column = Eigen::VectorXcd::Zero(n);
    column(static_cast<Eigen::Index>(index)) = value;
    return column;
  };
  if (layout.lengths)
    for (std::size_t index = 0; index < layout.edges; ++index) {
      basis.push_back(unit(layout.lengthOffset + index, one));
      basis.push_back(unit(layout.lengthOffset + index, imaginary));
    }
  if (layout.links) {
    for (std::size_t index = 0; index < layout.edges; ++index)
      basis.push_back(unit(layout.linkOffset + index, imaginary));
    for (Eigen::Index k = 0; k < freeModuli.cols(); ++k) {
      Eigen::VectorXcd column = Eigen::VectorXcd::Zero(n);
      for (std::size_t index = 0; index < layout.edges; ++index)
        column(static_cast<Eigen::Index>(layout.linkOffset + index)) =
            complexd{freeModuli(static_cast<Eigen::Index>(index), k), 0.0};
      basis.push_back(column);
    }
  }
  if (layout.multipliers)
    for (std::size_t index = 0; index < layout.constraints; ++index) {
      basis.push_back(unit(layout.multiplierOffset + index, one));
      basis.push_back(unit(layout.multiplierOffset + index, imaginary));
    }

  const auto unknowns = static_cast<Eigen::Index>(basis.size());
  Eigen::MatrixXcd parametrization(n, unknowns);
  for (Eigen::Index k = 0; k < unknowns; ++k)
    parametrization.col(k) = basis[static_cast<std::size_t>(k)];
  const Eigen::MatrixXcd image = jacobian * parametrization;
  Eigen::MatrixXd system(2 * n, unknowns);
  system.topRows(n) = image.real();
  system.bottomRows(n) = image.imag();
  Eigen::VectorXd right(2 * n);
  right.head(n) = target.real();
  right.tail(n) = target.imag();

  const Eigen::JacobiSVD<Eigen::MatrixXd> svd(
      system, Eigen::ComputeThinU | Eigen::ComputeThinV);
  const Eigen::VectorXd &singular = svd.singularValues();
  Eigen::Index rank = 0;
  while (rank < singular.size() && singular(rank) > cut) ++rank;
  Eigen::VectorXd unknown = Eigen::VectorXd::Zero(unknowns);
  if (rank > 0) {
    Eigen::VectorXd coefficients =
        svd.matrixU().leftCols(rank).transpose() * right;
    for (Eigen::Index k = 0; k < rank; ++k) coefficients(k) /= singular(k);
    unknown = svd.matrixV().leftCols(rank) * coefficients;
  }
  ConstrainedStep out;
  out.step = parametrization * unknown.cast<complexd>();
  out.rank = static_cast<std::size_t>(rank);
  if (rank > 0 && rank < singular.size())
    out.gap = singular(rank - 1) / singular(rank);
  else if (rank == 0)
    out.gap = std::numeric_limits<double>::quiet_NaN();
  return out;
}

/// What refused a trial step.
enum class Refusal { None, SectorGuard, DomainGuard, ResidualTest };

}  // namespace

std::string relaxationStopName(RelaxationStop reason) {
  switch (reason) {
    case RelaxationStop::Converged:
      return "converged";
    case RelaxationStop::IterationBudget:
      return "the declared iterations ran out";
    case RelaxationStop::NoDescent:
      return "no damped step reduced the residual";
    case RelaxationStop::SectorBoundary:
      return "no stationary point in the declared monopole sector";
    case RelaxationStop::DomainBoundary:
      return "every damped step left the domain of the action";
    case RelaxationStop::HeldFloor:
      return "the residual is at its floor on the held set";
    case RelaxationStop::LengthRunaway:
      return "the squared lengths overflowed the double";
    case RelaxationStop::Continued:
      return "continued";
  }
  return "unknown";
}

HolomorphicRelaxation::HolomorphicRelaxation(
    JointAction action, HolomorphicRelaxationDeclaration declaration,
    CovarianceRebuild rebuild)
    : action_(std::move(action)),
      declaration_(std::move(declaration)),
      rebuild_(std::move(rebuild)) {
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

namespace {

/// The group average \f$ |G|^{-1}\sum_gD(g)^{-1}XD(g) \f$ of a flat matrix
/// under the declared band symmetry.
std::vector<complexd> symmetryAverage(
    const std::vector<complexd> &flat, std::size_t order,
    const std::vector<std::vector<complexd>> &symmetry) {
  const Eigen::MatrixXcd matrix = toMatrix(flat, order);
  Eigen::MatrixXcd average = Eigen::MatrixXcd::Zero(matrix.rows(), matrix.cols());
  for (const auto &operatorFlat : symmetry) {
    if (operatorFlat.size() != order * order)
      throw std::invalid_argument(
          "HolomorphicRelaxation::jacobian: a band symmetry operator must be "
          "square over the carrier's cells");
    const Eigen::MatrixXcd operatorOfG = toMatrix(operatorFlat, order);
    const Eigen::FullPivLU<Eigen::MatrixXcd> lu(operatorOfG);
    if (!lu.isInvertible())
      throw std::invalid_argument(
          "HolomorphicRelaxation::jacobian: a band symmetry operator is "
          "singular");
    average += lu.inverse() * matrix * operatorOfG;
  }
  return toFlat(average / static_cast<double>(symmetry.size()));
}

/// Refuse by name a rebuilt band that is not isolated at the declared
/// tolerance: an eigenvalue outside it within the tolerance, relative to
/// \f$ \max(1,|\lambda_k|) \f$, of one of its eigenvalues.
void requireIsolated(const RebuiltCarrierState &state) {
  const std::size_t n = state.eigenvalues.size();
  for (std::size_t index = 0; index < state.bands.size(); ++index) {
    const auto &band = state.bands[index];
    std::vector<bool> inside(n, false);
    for (const std::size_t mode : band.modes) {
      if (mode >= n)
        throw std::invalid_argument(
            "HolomorphicRelaxation::jacobian: band " + std::to_string(index) +
            " names mode " + std::to_string(mode) + " of an operator with " +
            std::to_string(n) + " modes");
      inside[mode] = true;
    }
    for (const std::size_t k : band.modes)
      for (std::size_t j = 0; j < n; ++j) {
        if (inside[j]) continue;
        const double distance =
            std::abs(state.eigenvalues[k] - state.eigenvalues[j]);
        const double scale = std::max(1.0, std::abs(state.eigenvalues[k]));
        if (distance <= state.bandTolerance * scale)
          throw std::invalid_argument(
              "HolomorphicRelaxation::jacobian: band " + std::to_string(index) +
              " is not isolated at the declared band tolerance " +
              threeDigits(state.bandTolerance) + ": its eigenvalue " +
              threeDigits(state.eigenvalues[k].real()) + " + " +
              threeDigits(state.eigenvalues[k].imag()) +
              "i and the eigenvalue " +
              threeDigits(state.eigenvalues[j].real()) + " + " +
              threeDigits(state.eigenvalues[j].imag()) +
              "i outside it are " + threeDigits(distance) +
              " apart, so the first-order perturbation of its Riesz "
              "projector, which divides by their difference, is not defined");
      }
  }
}

/// Add to \p matrix the columns' self-consistent part: for every geometric
/// coordinate, the response of the equations to the variation of the
/// rebuilt state along it. The operator's derivative along a class
/// coordinate is the sum over the class's edges (each link member on its
/// orientation), the variation of every band's Riesz projector follows from
/// it by first-order perturbation, the covariance's variation is the bands'
/// weighted sum and the pinned fiber's the sum over its bands, and the
/// equations' response is `JointAction::stationarityStateVariation`,
/// reduced onto the classes as the residual is.
void addSelfConsistentColumns(const JointAction &working,
                              const RebuiltCarrierState &state,
                              const Layout &layout,
                              const EdgeClasses &classes,
                              Eigen::MatrixXcd &matrix) {
  const auto &declaration = working.declaration();
  const bool matter =
      declaration.matterWeight != 0.0 && !declaration.covariance.empty();
  bool constrained = false;
  for (const auto &constraint : declaration.momentConstraints)
    if (constraint.multiplier != complexd{0.0, 0.0}) constrained = true;
  // The state enters no equation in scope.
  if (!matter && !constrained) return;
  const std::size_t n = state.eigenvalues.size();
  if (n == 0 || state.bands.empty())
    throw std::invalid_argument(
        "HolomorphicRelaxation::jacobian: the declared rebuild supplies no "
        "band data (no eigendecomposition and no bands), so the derivative "
        "of the rebuilt covariance along the coordinates cannot be formed");
  const std::size_t cells = n * n;
  if (state.eigenvectors.size() != cells ||
      state.leftEigenvectors.size() != cells)
    throw std::invalid_argument(
        "HolomorphicRelaxation::jacobian: the rebuild's eigenvectors are "
        "square matrices over the " + std::to_string(n) + " modes; got " +
        std::to_string(state.eigenvectors.size()) + " and " +
        std::to_string(state.leftEigenvectors.size()) + " entries");
  for (const std::size_t band : state.bandProjectorBands)
    if (band >= state.bands.size())
      throw std::invalid_argument(
          "HolomorphicRelaxation::jacobian: a band projector refers to band " +
          std::to_string(band) + " of " + std::to_string(state.bands.size()));
  requireIsolated(state);

  const CarrierDerivatives derivatives = working.carrierDerivatives();
  const std::size_t edges = classes.edgeCount;
  auto column = [&](std::size_t index, bool link) {
    // The operator's derivative along the class coordinate.
    std::vector<complexd> dh(cells, complexd{0.0, 0.0});
    for (const auto &[edge, orientation] : classes.members[index]) {
      if (edge >= edges) continue;
      const auto &part = link ? derivatives.links[edge]
                              : derivatives.lengths[edge];
      if (part.size() != cells)
        throw std::invalid_argument(
            "HolomorphicRelaxation::jacobian: the carrier's derivative on "
            "edge " + std::to_string(edge) + " has " +
            std::to_string(part.size()) + " entries, but the rebuild read " +
            std::to_string(n) + " modes");
      const double sign = link ? static_cast<double>(orientation) : 1.0;
      for (std::size_t i = 0; i < cells; ++i) dh[i] += sign * part[i];
    }
    if (!state.bandSymmetry.empty())
      dh = symmetryAverage(dh, n, state.bandSymmetry);
    // The variation of every band's projector, and of the state from them.
    CarriedStateVariation variation;
    variation.covariance.assign(cells, complexd{0.0, 0.0});
    if (!state.momentProjector.empty())
      variation.momentProjector.assign(cells, complexd{0.0, 0.0});
    variation.bandProjectors.resize(declaration.momentBandProjectors.size());
    std::vector<std::vector<complexd>> projectorVariations;
    projectorVariations.reserve(state.bands.size());
    for (const auto &band : state.bands) {
      projectorVariations.push_back(rieszProjectorDerivative(
          state.eigenvalues, state.eigenvectors, state.leftEigenvectors,
          band.modes, dh));
      const auto &dP = projectorVariations.back();
      for (std::size_t i = 0; i < cells; ++i) {
        variation.covariance[i] += band.covarianceWeight * dP[i];
        if (band.inFiber && !variation.momentProjector.empty())
          variation.momentProjector[i] += dP[i];
      }
    }
    for (std::size_t i = 0; i < state.bandProjectorBands.size() &&
                            i < variation.bandProjectors.size();
         ++i)
      variation.bandProjectors[i] =
          projectorVariations[state.bandProjectorBands[i]];
    const std::vector<complexd> response =
        working.stationarityStateVariation(variation);
    const auto col = static_cast<Eigen::Index>(
        link ? layout.linkOffset + index : layout.lengthOffset + index);
    for (std::size_t row = 0; row < classes.count(); ++row) {
      complexd lengthSum{0.0, 0.0};
      complexd linkSum{0.0, 0.0};
      for (const auto &[edge, orientation] : classes.members[row]) {
        if (edge >= edges) continue;
        lengthSum += response[edge];
        linkSum += static_cast<double>(orientation) * response[edges + edge];
      }
      if (layout.lengths)
        matrix(static_cast<Eigen::Index>(layout.lengthOffset + row), col) +=
            lengthSum;
      if (layout.links)
        matrix(static_cast<Eigen::Index>(layout.linkOffset + row), col) +=
            linkSum;
    }
  };
  for (std::size_t index = 0; index < classes.count(); ++index) {
    if (layout.lengths) column(index, false);
    if (layout.links) column(index, true);
  }
}

}  // namespace

std::vector<complexd> rieszProjectorDerivative(
    const std::vector<complexd> &eigenvalues,
    const std::vector<complexd> &eigenvectors,
    const std::vector<complexd> &leftEigenvectors,
    const std::vector<std::size_t> &modes,
    const std::vector<complexd> &operatorVariation) {
  const std::size_t n = eigenvalues.size();
  const std::size_t cells = n * n;
  if (eigenvectors.size() != cells || leftEigenvectors.size() != cells ||
      operatorVariation.size() != cells)
    throw std::invalid_argument(
        "rieszProjectorDerivative: the eigenvectors, their inverse and the "
        "operator's variation are square matrices over the " +
        std::to_string(n) + " modes; got " +
        std::to_string(eigenvectors.size()) + ", " +
        std::to_string(leftEigenvectors.size()) + " and " +
        std::to_string(operatorVariation.size()) + " entries");
  std::vector<bool> inside(n, false);
  for (const std::size_t mode : modes) {
    if (mode >= n)
      throw std::invalid_argument(
          "rieszProjectorDerivative: mode " + std::to_string(mode) +
          " is out of range for " + std::to_string(n) + " modes");
    inside[mode] = true;
  }
  const Eigen::MatrixXcd right = toMatrix(eigenvectors, n);
  const Eigen::MatrixXcd left = toMatrix(leftEigenvectors, n);
  // In the eigenbasis the variation is G = W dh V, and the projector's
  // variation has the entries G_kj / (lambda_k - lambda_j) between the band
  // and its complement, in both orders, and nothing elsewhere.
  const Eigen::MatrixXcd inBasis = left * toMatrix(operatorVariation, n) * right;
  Eigen::MatrixXcd variation = Eigen::MatrixXcd::Zero(
      static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(n));
  for (const std::size_t k : modes)
    for (std::size_t j = 0; j < n; ++j) {
      if (inside[j]) continue;
      const complexd gap = eigenvalues[k] - eigenvalues[j];
      if (gap == complexd{0.0, 0.0})
        throw std::invalid_argument(
            "rieszProjectorDerivative: the band is not isolated: mode " +
            std::to_string(k) + " inside it and mode " + std::to_string(j) +
            " outside it carry the same eigenvalue, so the quotient by "
            "their difference is not defined");
      const auto kk = static_cast<Eigen::Index>(k);
      const auto jj = static_cast<Eigen::Index>(j);
      variation(kk, jj) = inBasis(kk, jj) / gap;
      variation(jj, kk) = inBasis(jj, kk) / gap;
    }
  return toFlat(right * variation * left);
}

std::vector<complexd> HolomorphicRelaxation::jacobian() const {
  const EdgeClasses classes =
      edgeClassesOf(action_.edgeCount(), declaration_);
  const Layout layout(classes.count(), action_.constraintCount(),
                      declaration_);
  // The Jacobian is assembled on a copy of the action carrying the state a
  // declared rebuild gives at the point; the complex itself is read and not
  // written.
  JointAction working = action_;
  RebuiltCarrierState state;
  if (rebuild_.at) {
    state = rebuild_.at(working);
    applyState(state, working);
  }
  const std::size_t edges = classes.edgeCount;
  Eigen::MatrixXcd matrix = Eigen::MatrixXcd::Zero(
      static_cast<Eigen::Index>(layout.count),
      static_cast<Eigen::Index>(layout.count));

  // The geometric blocks: the Hessian of the action at fixed carried state,
  // reduced onto the classes. The equation of a class is the sum of its
  // edges' equations and its coordinate moves every edge of the class, so an
  // entry is the double sum over the two classes' members, each link member
  // on its orientation.
  if (layout.lengths || layout.links) {
    const std::vector<complexd> hessian = working.actionHessian();
    const std::size_t size = 2 * edges;
    auto entry = [&](std::size_t row, std::size_t column) {
      return hessian[row * size + column];
    };
    for (std::size_t row = 0; row < classes.count(); ++row)
      for (std::size_t column = 0; column < classes.count(); ++column) {
        complexd lengthLength{0.0, 0.0};
        complexd lengthLink{0.0, 0.0};
        complexd linkLength{0.0, 0.0};
        complexd linkLink{0.0, 0.0};
        for (const auto &[e, oe] : classes.members[row])
          for (const auto &[f, of] : classes.members[column]) {
            if (e >= edges || f >= edges) continue;
            const double se = static_cast<double>(oe);
            const double sf = static_cast<double>(of);
            lengthLength += entry(e, f);
            lengthLink += sf * entry(e, edges + f);
            linkLength += se * entry(edges + e, f);
            linkLink += se * sf * entry(edges + e, edges + f);
          }
        if (layout.lengths) {
          matrix(static_cast<Eigen::Index>(layout.lengthOffset + row),
                 static_cast<Eigen::Index>(layout.lengthOffset + column)) =
              lengthLength;
          if (layout.links)
            matrix(static_cast<Eigen::Index>(layout.lengthOffset + row),
                   static_cast<Eigen::Index>(layout.linkOffset + column)) =
                lengthLink;
        }
        if (layout.links) {
          if (layout.lengths)
            matrix(static_cast<Eigen::Index>(layout.linkOffset + row),
                   static_cast<Eigen::Index>(layout.lengthOffset + column)) =
                linkLength;
          matrix(static_cast<Eigen::Index>(layout.linkOffset + row),
                 static_cast<Eigen::Index>(layout.linkOffset + column)) =
              linkLink;
        }
      }
    // The self-consistent part of every geometric column.
    if (rebuild_.at)
      addSelfConsistentColumns(working, state, layout, classes, matrix);
  }

  // The multiplier columns are the constraints' gradients: the action is
  // linear in every xi_j, so the column is the moment's own gradient, summed
  // over each class, and the moment rows of it are zero. The constraint rows'
  // length and link entries are the derivatives of the constraints' values,
  // the same gradients: at a spectral projector the projector's own variation
  // is off-diagonal between its range and its kernel and contributes no
  // trace, so the gradient at fixed projector is the whole derivative. Taken
  // from one evaluation, the rows keep a pinned power sum that depends on
  // others (as on a sheeted fiber, whose degenerate eigenvalues leave only
  // as many independent power sums as distinct eigenvalues) exactly
  // dependent, which the rank decision reads.
  if (layout.multipliers) {
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
          linkPart += orientation > 0 ? gradient[edges + edge]
                                      : -gradient[edges + edge];
        }
        const auto row = column;
        if (layout.lengths) {
          matrix(static_cast<Eigen::Index>(layout.lengthOffset + index),
                 column) = lengthPart;
          matrix(row, static_cast<Eigen::Index>(layout.lengthOffset + index)) =
              lengthPart;
        }
        if (layout.links) {
          matrix(static_cast<Eigen::Index>(layout.linkOffset + index),
                 column) = linkPart;
          matrix(row, static_cast<Eigen::Index>(layout.linkOffset + index)) =
              linkPart;
        }
      }
    }
  }

  std::vector<complexd> flat(layout.count * layout.count, complexd{0.0, 0.0});
  for (std::size_t row = 0; row < layout.count; ++row)
    for (std::size_t column = 0; column < layout.count; ++column)
      flat[row * layout.count + column] =
          matrix(static_cast<Eigen::Index>(row),
                 static_cast<Eigen::Index>(column));
  return flat;
}

std::vector<complexd> HolomorphicRelaxation::residual() const {
  const EdgeClasses classes =
      edgeClassesOf(action_.edgeCount(), declaration_);
  const Layout layout(classes.count(), action_.constraintCount(),
                      declaration_);
  JointAction working = action_;
  applyRebuild(rebuild_, working);
  return reducedResidual(working, layout, classes);
}

std::vector<ActionTermRecord> actionTermRecords(
    const JointAction &action,
    const HolomorphicRelaxationDeclaration &declaration) {
  const EdgeClasses classes = edgeClassesOf(action.edgeCount(), declaration);
  const auto reducedNorm = [&](const std::vector<complexd> &lengths,
                               const std::vector<complexd> &links) {
    double squared = 0.0;
    if (declaration.relaxLengths)
      for (const auto &members : classes.members) {
        complexd sum{0.0, 0.0};
        for (const auto &[edge, orientation] : members)
          if (edge < lengths.size()) sum += lengths[edge];
        squared += std::norm(sum);
      }
    if (declaration.relaxLinks)
      for (const auto &members : classes.members) {
        complexd sum{0.0, 0.0};
        for (const auto &[edge, orientation] : members)
          if (edge < links.size())
            sum += orientation > 0 ? links[edge] : -links[edge];
        squared += std::norm(sum);
      }
    return std::sqrt(squared);
  };
  const std::size_t edges = action.edgeCount();
  std::vector<ActionTermRecord> records;
  ActionTermRecord constraints;
  constraints.name = "constraints";
  constraints.label = "sum_j xi_j (c_j - c_j*)";
  constraints.factored = false;
  std::vector<complexd> constraintLengths(edges, complexd{0.0, 0.0});
  std::vector<complexd> constraintLinks(edges, complexd{0.0, 0.0});
  ActionTermRecord action_;
  action_.name = "action";
  action_.label = "S";
  action_.factored = false;
  std::vector<complexd> totalLengths(edges, complexd{0.0, 0.0});
  std::vector<complexd> totalLinks(edges, complexd{0.0, 0.0});
  for (const ActionTermGradient &term : action.termGradients()) {
    ActionTermRecord record;
    record.name = term.name;
    record.label = term.label;
    record.weight = term.weight;
    record.bare = term.bare;
    record.factored = term.factored;
    record.value = term.value;
    record.gradientNorm =
        reducedNorm(term.lengthStationarity, term.linkStationarity);
    records.push_back(std::move(record));
    const bool constraint = term.name.rfind("constraint ", 0) == 0;
    for (std::size_t edge = 0; edge < edges; ++edge) {
      const complexd length = edge < term.lengthStationarity.size()
                                  ? term.lengthStationarity[edge]
                                  : complexd{0.0, 0.0};
      const complexd link = edge < term.linkStationarity.size()
                                ? term.linkStationarity[edge]
                                : complexd{0.0, 0.0};
      totalLengths[edge] += length;
      totalLinks[edge] += link;
      if (constraint) {
        constraintLengths[edge] += length;
        constraintLinks[edge] += link;
      }
    }
    action_.value += term.value;
    if (constraint) constraints.value += term.value;
  }
  constraints.bare = constraints.value;
  constraints.gradientNorm = reducedNorm(constraintLengths, constraintLinks);
  records.push_back(std::move(constraints));
  action_.bare = action_.value;
  action_.gradientNorm = reducedNorm(totalLengths, totalLinks);
  records.push_back(std::move(action_));
  return records;
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
  // The residual of the equations in scope at the action's current point,
  // with Gamma rebuilt there first when a rebuild is declared.
  auto evaluate = [&](JointAction &target) {
    applyRebuild(rebuild_, target);
    return reducedResidual(target, layout, classes);
  };
  auto recordAction = [](const JointAction &target, std::complex<double> &value,
                         bool &available, std::string &unavailable) {
    const ReportedActionValue reported = target.reportedValue();
    value = reported.value;
    available = reported.available;
    unavailable = reported.unavailable;
  };
  const double startScale =
      layout.lengths ? largestSquaredLength(action_, classes) : 0.0;
  // The only bound on the lengths is the datatype's: a squared length that
  // overflowed the double stops the solve; short of that the equations decide.
  const bool runawayDeclared = layout.lengths;
  report.reggeHingeCount = action_.reggeHingeCount();
  report.reggeStructurallyZero = action_.reggeStructurallyZero();
  report.initialResidualNorm = euclideanNorm(evaluate(action_));
  if (declaration_.recordTerms)
    report.initialTerms = actionTermRecords(action_, declaration_);
  report.residualNorm = report.initialResidualNorm;
  recordAction(action_, report.action, report.actionAvailable,
               report.actionUnavailable);
  report.multipliers = action_.multipliers();
  report.momentResiduals = action_.momentResiduals();
  report.reggeOffPrincipalAngles = action_.reggeOffPrincipalAngles();
  if (layout.count == 0) {
    report.converged = report.residualNorm <= declaration_.tolerance;
    report.stopReason = report.converged ? RelaxationStop::Converged
                                         : RelaxationStop::IterationBudget;
    report.stopDetail = "the solve has no variables; the residual norm is " +
                        threeDigits(report.residualNorm);
    report.sectorMonopoleNumbers = sectorNumbers(action_, sectors);
    return report;
  }

  bool stopped = false;
  for (std::size_t iteration = 0; iteration < declaration_.maximumIterations;
       ++iteration) {
    const auto residual = evaluate(action_);
    const double residualNorm = euclideanNorm(residual);
    report.residualNorm = residualNorm;
    if (residualNorm <= declaration_.tolerance) {
      report.converged = true;
      break;
    }

    // A Jacobian the action cannot form at this point (a refused derivative,
    // a band that is not isolated) leaves no Newton step to take from it;
    // the solve stops there and says why.
    std::vector<complexd> flatJacobian;
    std::string jacobianRefusal;
    try {
      flatJacobian = jacobian();
    } catch (const std::invalid_argument &refusal) {
      jacobianRefusal = refusal.what();
    } catch (const std::domain_error &refusal) {
      jacobianRefusal = refusal.what();
    } catch (const std::runtime_error &refusal) {
      jacobianRefusal = refusal.what();
    }
    if (!jacobianRefusal.empty()) {
      report.stopReason = RelaxationStop::DomainBoundary;
      report.stopDetail =
          "the Jacobian could not be formed at the point reached after " +
          std::to_string(iteration) + " accepted steps (residual norm " +
          threeDigits(residualNorm) + "): " + jacobianRefusal;
      stopped = true;
      break;
    }
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
    record.constrainedRankGap = std::numeric_limits<double>::quiet_NaN();

    // Held sectors: the step is the constrained Newton step, the minimum-norm
    // least-squares solution of the linearized equations over the tangent
    // space of the held set, so it keeps every held modulus exactly.
    if (layout.links && !sectors.empty) {
      // The Jacobian's rank boundary: the geometric mean of its smallest
      // retained singular value and the larger of its largest discarded one
      // and its cut, the midpoint of the gap its rank decision reads.
      const double below =
          std::max(rank < singular.size() ? singular(rank) : 0.0, cut);
      const double boundary =
          rank > 0 ? std::sqrt(singular(rank - 1) * below) : cut;
      const ConstrainedStep held = constrainedStep(
          matrix, target, layout, sectors.freeModuli, boundary);
      step = held.step;
      record.constrainedStep = true;
      record.constrainedRank = held.rank;
      record.constrainedRankGap = held.gap;
    }
    const double residualScale = target.norm();
    const double floor = (matrix * step - target).norm();
    record.linearResidual =
        residualScale > 0.0 ? floor / residualScale : 0.0;
    // A constrained step that cannot reduce the residual norm by more than
    // the tolerance the solve is asked to meet leaves the solve at the floor
    // of the held set: taking it would only move by rounding.
    if (record.constrainedStep &&
        residualNorm - floor <= declaration_.tolerance) {
      record.accepted = false;
      record.damping = 0.0;
      record.stepNorm = 0.0;
      recordAction(action_, record.action, record.actionAvailable,
                   record.actionUnavailable);
      if (declaration_.recordTerms)
        record.terms = actionTermRecords(action_, declaration_);
      report.steps.push_back(record);
      report.stopReason = RelaxationStop::HeldFloor;
      report.stopDetail =
          "the residual norm " + threeDigits(residualNorm) + " after " +
          std::to_string(iteration) +
          " accepted steps is at its floor on the held set: the constrained "
          "Newton step would leave " + threeDigits(floor) +
          " of it, a reduction no larger than the tolerance " +
          threeDigits(declaration_.tolerance);
      stopped = true;
      break;
    }
    recordAction(action_, record.action, record.actionAvailable,
                 record.actionUnavailable);

    const StateSnapshot snapshot = takeSnapshot(action_);
    double damping = 1.0;
    bool accepted = false;
    std::size_t trials = 0;
    Refusal lastRefusal = Refusal::None;
    std::string lastDomainRefusal;
    for (std::size_t attempt = 0; attempt <= declaration_.maximumDampings;
         ++attempt) {
      ++trials;
      restoreSnapshot(action_, snapshot);
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
        lastRefusal = Refusal::SectorGuard;
        damping *= 0.5;
        continue;
      }
      // A trial point the action refuses to evaluate (a link driven to zero
      // or to infinity by an overflowing multiplicative step, a face holonomy
      // outside the domain of the holonomy term, a singular dressed metric,
      // an operator a rebuild cannot read bands from) has left the domain the
      // equations are posed on; it is halved like a step that does not reduce
      // the residual. The equations are unchanged.
      double trialNorm = 0.0;
      try {
        trialNorm = euclideanNorm(evaluate(action_));
      } catch (const std::invalid_argument &refusal) {
        ++record.domainGuardDampings;
        lastRefusal = Refusal::DomainGuard;
        lastDomainRefusal = refusal.what();
        damping *= 0.5;
        continue;
      } catch (const std::domain_error &refusal) {
        ++record.domainGuardDampings;
        lastRefusal = Refusal::DomainGuard;
        lastDomainRefusal = refusal.what();
        damping *= 0.5;
        continue;
      } catch (const std::runtime_error &refusal) {
        ++record.domainGuardDampings;
        lastRefusal = Refusal::DomainGuard;
        lastDomainRefusal = refusal.what();
        damping *= 0.5;
        continue;
      }
      if (!std::isfinite(trialNorm)) {
        ++record.domainGuardDampings;
        lastRefusal = Refusal::DomainGuard;
        damping *= 0.5;
        continue;
      }
      if (trialNorm < residualNorm) {
        accepted = true;
        record.accepted = true;
        record.damping = damping;
        record.stepNorm = damping * step.norm();
        report.residualNorm = trialNorm;
        break;
      }
      ++record.residualTestDampings;
      lastRefusal = Refusal::ResidualTest;
      damping *= 0.5;
    }
    if (record.sectorGuardDampings > 0) ++report.sectorGuardDampedSteps;
    if (!accepted) {
      restoreSnapshot(action_, snapshot);
      record.damping = 0.0;
      record.stepNorm = 0.0;
      if (declaration_.recordTerms)
        record.terms = actionTermRecords(action_, declaration_);
      report.steps.push_back(record);
      report.residualNorm = residualNorm;
      // The smallest trial step says what blocks the Newton direction: an
      // arbitrarily short step along it is refused for that reason.
      const std::string smallest =
          "the smallest trial step, 2^-" +
          std::to_string(trials > 0 ? trials - 1 : 0) +
          " of the Newton step";
      const std::string counts =
          " (of the " + std::to_string(trials) + " trial steps, the residual "
          "test refused " + std::to_string(record.residualTestDampings) +
          ", the sector guard " + std::to_string(record.sectorGuardDampings) +
          " and the domain guard " +
          std::to_string(record.domainGuardDampings) +
          "; the residual norm is " + threeDigits(residualNorm) +
          " after " + std::to_string(iteration) + " accepted steps)";
      switch (lastRefusal) {
        case Refusal::SectorGuard:
          report.stopReason = RelaxationStop::SectorBoundary;
          report.stopDetail =
              "no stationary point in the declared monopole sector: " +
              smallest + " changed a held monopole number, so the Newton "
              "direction drives a held face holonomy across -1" + counts;
          break;
        case Refusal::DomainGuard:
          report.stopReason = RelaxationStop::DomainBoundary;
          report.stopDetail =
              smallest + " reached a point at which the action refuses to "
              "evaluate or its residual is not finite" +
              (lastDomainRefusal.empty() ? std::string{}
                                         : " (" + lastDomainRefusal + ")") +
              counts;
          break;
        case Refusal::ResidualTest:
        case Refusal::None:
          report.stopReason = RelaxationStop::NoDescent;
          report.stopDetail = "no damped step reduced the residual norm: " +
                              smallest + " did not reduce it" + counts;
          break;
      }
      stopped = true;
      break;
    }
    if (declaration_.recordTerms)
      record.terms = actionTermRecords(action_, declaration_);
    report.steps.push_back(record);
    if (rebuild_.accepted) rebuild_.accepted(action_);
    if (runawayDeclared) {
      const double largest = largestSquaredLength(action_, classes);
      if (!std::isfinite(largest)) {
        report.stopReason = RelaxationStop::LengthRunaway;
        report.stopDetail =
            "the squared lengths overflowed the double: after " +
            std::to_string(iteration + 1) +
            " accepted steps a squared length is beyond the largest finite "
            "value " +
            threeDigits(std::numeric_limits<double>::max()) +
            " (it was " + threeDigits(startScale) +
            " at the start); the residual norm is " +
            threeDigits(report.residualNorm);
        stopped = true;
        break;
      }
    }
  }

  report.converged = report.residualNorm <= declaration_.tolerance;
  if (report.converged) {
    report.stopReason = RelaxationStop::Converged;
    report.stopDetail = "the residual norm " +
                        threeDigits(report.residualNorm) +
                        " is at or below the tolerance " +
                        threeDigits(declaration_.tolerance) + " after " +
                        std::to_string(report.steps.size()) +
                        " accepted steps";
  } else if (!stopped) {
    report.stopReason = RelaxationStop::IterationBudget;
    report.stopDetail = "the declared " +
                        std::to_string(declaration_.maximumIterations) +
                        " iterations ran out at residual norm " +
                        threeDigits(report.residualNorm);
  }
  if (layout.lengths && startScale > 0.0)
    report.largestLengthRatio =
        largestSquaredLength(action_, classes) / startScale;
  report.sectorMonopoleNumbers = sectorNumbers(action_, sectors);
  const std::vector<double> endModuli = heldLogModuli(action_, sectors);
  for (std::size_t index = 0; index < endModuli.size(); ++index)
    report.heldModulusDrift = std::max(
        report.heldModulusDrift, std::abs(endModuli[index] - startModuli[index]));
  recordAction(action_, report.action, report.actionAvailable,
               report.actionUnavailable);
  report.multipliers = action_.multipliers();
  report.momentResiduals = action_.momentResiduals();
  report.reggeOffPrincipalAngles = action_.reggeOffPrincipalAngles();
  return report;
}

}  // namespace tessera::cobordism
