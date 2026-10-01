// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "cobordism/HolomorphicRelaxation.h"

#include <algorithm>
#include <chrono>
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
                              const EdgeClasses &classes,
                              double rankTolerance) {
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
  const double cut = values.size() > 0 ? rankTolerance * values(0) : 0.0;
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

std::vector<double> logModuliOf(const JointAction &action,
                                const SectorGeometry &geometry) {
  std::vector<double> moduli;
  if (geometry.empty) return moduli;
  const auto edges = action.spacetime()->getEdgeList()->toVector();
  for (const auto &sector : geometry.faces)
    for (const auto &sides : sector)
      moduli.push_back(std::log(std::abs(faceHolonomy(edges, sides))));
  return moduli;
}

/// The map from the real unknowns that parametrize the steps keeping the held
/// moduli to a step, one column per unknown.
///
/// A step \f$ d \f$ keeps the held moduli exactly when the real part of its
/// link block lies in the kernel of the held faces' coboundary, so it is
/// parametrized by real unknowns: the real and imaginary parts of every
/// length and multiplier step, the imaginary part (the phase) of every link
/// step, and the coefficients of the link step's real part on the columns of
/// \p freeModuli, an orthonormal basis of that kernel. The map is an isometry
/// of the real inner product, so the minimum-norm unknowns give the
/// minimum-norm step.
Eigen::MatrixXcd heldParametrization(const Layout &layout,
                                     const Eigen::MatrixXd &freeModuli,
                                     Eigen::Index n) {
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
  return parametrization;
}

}  // namespace

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
        resolveSectors(action_, declaration_.heldSectors, classes,
                       declaration_.rankTolerance);
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

/// Check that every rebuilt band names modes of the operator. A band that is
/// close to another eigenvalue is not refused: the perturbation of its Riesz
/// projector divides by the difference, the Jacobian carries what that gives,
/// and the isolation of every band is reported by the solve that rebuilt it.
void requireModesInRange(const RebuiltCarrierState &state) {
  const std::size_t n = state.eigenvalues.size();
  for (std::size_t index = 0; index < state.bands.size(); ++index)
    for (const std::size_t mode : state.bands[index].modes)
      if (mode >= n)
        throw std::invalid_argument(
            "HolomorphicRelaxation::jacobian: band " + std::to_string(index) +
            " names mode " + std::to_string(mode) + " of an operator with " +
            std::to_string(n) + " modes");
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
  requireModesInRange(state);

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
            "rieszProjectorDerivative: mode " + std::to_string(k) +
            " inside the band and mode " + std::to_string(j) +
            " outside it carry the same eigenvalue exactly, so the quotient "
            "by their difference has no value");
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
    const std::vector<complexd> hessian =
        working.actionHessian(layout.lengths, layout.links);
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

/// What a linearization keeps: the Jacobian, its singular value
/// decomposition, the parametrization of the held set and the decomposition
/// of the real system over it, and the Newton step's record.
struct HolomorphicLinearization::Implementation {
  std::vector<complexd> residual;
  Eigen::MatrixXcd jacobian;
  Eigen::MatrixXcd left;
  Eigen::MatrixXcd right;
  Eigen::VectorXd singular;
  Eigen::Index rank = 0;
  bool constrained = false;
  /// The map from the real unknowns of the held set to a step.
  Eigen::MatrixXcd parametrization;
  Eigen::MatrixXd constrainedLeft;
  Eigen::MatrixXd constrainedRight;
  Eigen::VectorXd constrainedSingular;
  Eigen::Index constrainedRank = 0;
  HolomorphicNewtonStep newton;

  [[nodiscard]] Eigen::VectorXcd solve(const Eigen::VectorXcd &target) const {
    const Eigen::Index size = jacobian.cols();
    if (!constrained) {
      Eigen::VectorXcd step = Eigen::VectorXcd::Zero(size);
      if (rank > 0) {
        Eigen::VectorXcd coefficients = left.leftCols(rank).adjoint() * target;
        for (Eigen::Index k = 0; k < rank; ++k) coefficients(k) /= singular(k);
        step = right.leftCols(rank) * coefficients;
      }
      return step;
    }
    const Eigen::Index n = jacobian.rows();
    Eigen::VectorXd stacked(2 * n);
    stacked.head(n) = target.real();
    stacked.tail(n) = target.imag();
    Eigen::VectorXd unknown = Eigen::VectorXd::Zero(parametrization.cols());
    if (constrainedRank > 0) {
      Eigen::VectorXd coefficients =
          constrainedLeft.leftCols(constrainedRank).transpose() * stacked;
      for (Eigen::Index k = 0; k < constrainedRank; ++k)
        coefficients(k) /= constrainedSingular(k);
      unknown = constrainedRight.leftCols(constrainedRank) * coefficients;
    }
    return parametrization * unknown.cast<complexd>();
  }
};

HolomorphicLinearization::HolomorphicLinearization(
    std::shared_ptr<const Implementation> implementation)
    : implementation_(std::move(implementation)) {}

std::vector<complexd> HolomorphicLinearization::solve(
    const std::vector<complexd> &rightHandSide) const {
  const Eigen::Index rows = implementation_->jacobian.rows();
  if (static_cast<Eigen::Index>(rightHandSide.size()) != rows)
    throw std::invalid_argument(
        "HolomorphicLinearization::solve: the right-hand side has " +
        std::to_string(rightHandSide.size()) + " entries for " +
        std::to_string(rows) + " equations");
  Eigen::VectorXcd target(rows);
  for (Eigen::Index row = 0; row < rows; ++row)
    target(row) = rightHandSide[static_cast<std::size_t>(row)];
  const Eigen::VectorXcd step = implementation_->solve(target);
  std::vector<complexd> out(static_cast<std::size_t>(step.size()));
  for (Eigen::Index index = 0; index < step.size(); ++index)
    out[static_cast<std::size_t>(index)] = step(index);
  return out;
}

const HolomorphicNewtonStep &HolomorphicLinearization::newtonStep()
    const noexcept {
  return implementation_->newton;
}

const std::vector<complexd> &HolomorphicLinearization::residual()
    const noexcept {
  return implementation_->residual;
}

HolomorphicLinearization HolomorphicRelaxation::linearization(
    const std::vector<complexd> &addedResidual,
    const std::vector<complexd> &addedJacobian) const {
  const EdgeClasses classes =
      edgeClassesOf(action_.edgeCount(), declaration_);
  const Layout layout(classes.count(), action_.constraintCount(),
                      declaration_);
  if (!addedResidual.empty() && addedResidual.size() != layout.count)
    throw std::invalid_argument(
        "HolomorphicRelaxation::linearization: the added residual has " +
        std::to_string(addedResidual.size()) + " entries for " +
        std::to_string(layout.count) + " equations");
  if (!addedJacobian.empty() &&
      addedJacobian.size() != layout.count * layout.count)
    throw std::invalid_argument(
        "HolomorphicRelaxation::linearization: the added Jacobian has " +
        std::to_string(addedJacobian.size()) + " entries for a system of " +
        std::to_string(layout.count) + " equations");
  auto kept = std::make_shared<HolomorphicLinearization::Implementation>();
  HolomorphicNewtonStep &out = kept->newton;
  out.rankTolerance = declaration_.rankTolerance;
  kept->residual = this->residual();
  for (std::size_t row = 0; row < addedResidual.size(); ++row)
    kept->residual[row] += addedResidual[row];
  out.residualNorm = euclideanNorm(kept->residual);
  out.step.assign(layout.count, complexd{0.0, 0.0});
  out.smallestRetainedSingularValue =
      std::numeric_limits<double>::quiet_NaN();
  out.rankGap = std::numeric_limits<double>::quiet_NaN();
  out.constrainedRankGap = std::numeric_limits<double>::quiet_NaN();
  const auto size = static_cast<Eigen::Index>(layout.count);
  kept->jacobian = Eigen::MatrixXcd::Zero(size, size);
  if (layout.count == 0) return HolomorphicLinearization(kept);

  const std::vector<complexd> flat = jacobian();
  for (Eigen::Index row = 0; row < size; ++row)
    for (Eigen::Index column = 0; column < size; ++column) {
      const std::size_t entry = static_cast<std::size_t>(row) * layout.count +
                                static_cast<std::size_t>(column);
      kept->jacobian(row, column) =
          flat[entry] +
          (addedJacobian.empty() ? complexd{0.0, 0.0} : addedJacobian[entry]);
    }
  Eigen::VectorXcd target(size);
  for (Eigen::Index row = 0; row < size; ++row)
    target(row) = -kept->residual[static_cast<std::size_t>(row)];

  // Two-sided Jacobi: every singular value accurate to rounding relative to
  // the largest, which the rank decision reads.
  const Eigen::JacobiSVD<Eigen::MatrixXcd> svd(
      kept->jacobian, Eigen::ComputeThinU | Eigen::ComputeThinV);
  kept->left = svd.matrixU();
  kept->right = svd.matrixV();
  kept->singular = svd.singularValues();
  const Eigen::VectorXd &singular = kept->singular;
  const double largest = singular.size() > 0 ? singular(0) : 0.0;
  const double cut = declaration_.rankTolerance * largest;
  Eigen::Index rank = 0;
  while (rank < singular.size() && singular(rank) > cut) ++rank;
  kept->rank = rank;
  out.jacobianRank = static_cast<std::size_t>(rank);
  out.largestSingularValue = largest;
  if (rank > 0) out.smallestRetainedSingularValue = singular(rank - 1);
  out.largestDiscardedSingularValue =
      rank < singular.size() ? singular(rank) : 0.0;
  if (rank > 0)
    out.rankGap = rank < singular.size()
                      ? singular(rank - 1) / singular(rank)
                      : std::numeric_limits<double>::infinity();

  if (layout.links && !declaration_.heldSectors.empty()) {
    // The steps that keep the held moduli, parametrized by real unknowns
    // (`heldParametrization`), and the real system over them.
    const SectorGeometry sectors =
        resolveSectors(action_, declaration_.heldSectors, classes,
                       declaration_.rankTolerance);
    kept->parametrization =
        heldParametrization(layout, sectors.freeModuli, size);
    const Eigen::MatrixXcd image = kept->jacobian * kept->parametrization;
    Eigen::MatrixXd system(2 * size, image.cols());
    system.topRows(size) = image.real();
    system.bottomRows(size) = image.imag();
    const Eigen::JacobiSVD<Eigen::MatrixXd> held(
        system, Eigen::ComputeThinU | Eigen::ComputeThinV);
    kept->constrainedLeft = held.matrixU();
    kept->constrainedRight = held.matrixV();
    kept->constrainedSingular = held.singularValues();
    const Eigen::VectorXd &values = kept->constrainedSingular;
    const double heldCut =
        values.size() > 0 ? declaration_.rankTolerance * values(0) : 0.0;
    Eigen::Index heldRank = 0;
    while (heldRank < values.size() && values(heldRank) > heldCut) ++heldRank;
    kept->constrainedRank = heldRank;
    kept->constrained = true;
    out.constrained = true;
    out.constrainedRank = static_cast<std::size_t>(heldRank);
    if (heldRank > 0)
      out.constrainedRankGap =
          heldRank < values.size()
              ? values(heldRank - 1) / values(heldRank)
              : std::numeric_limits<double>::infinity();
  }
  const Eigen::VectorXcd step = kept->solve(target);
  const double scale = target.norm();
  out.linearResidual =
      scale > 0.0 ? (kept->jacobian * step - target).norm() / scale : 0.0;
  for (Eigen::Index index = 0; index < size; ++index)
    out.step[static_cast<std::size_t>(index)] = step(index);
  return HolomorphicLinearization(kept);
}

HolomorphicNewtonStep HolomorphicRelaxation::newtonStep() const {
  return linearization().newtonStep();
}

std::vector<int> HolomorphicRelaxation::sectorMonopoleNumbers() const {
  if (declaration_.heldSectors.empty()) return {};
  const EdgeClasses classes =
      edgeClassesOf(action_.edgeCount(), declaration_);
  return sectorNumbers(
      action_, resolveSectors(action_, declaration_.heldSectors, classes,
                              declaration_.rankTolerance));
}

std::vector<double> HolomorphicRelaxation::heldLogModuli() const {
  if (declaration_.heldSectors.empty()) return {};
  const EdgeClasses classes =
      edgeClassesOf(action_.edgeCount(), declaration_);
  return logModuliOf(
      action_, resolveSectors(action_, declaration_.heldSectors, classes,
                              declaration_.rankTolerance));
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

}  // namespace tessera::cobordism
