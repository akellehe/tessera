// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "cobordism/WardFlux.h"

#include <algorithm>
#include <cmath>
#include <deque>
#include <map>
#include <set>
#include <stdexcept>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include "cobordism/ChainComplex.h"
#include "cobordism/HodgeLaplacian.h"
#include "spacetime/Spacetime.h"

namespace tessera::cobordism {

using complexd = std::complex<double>;

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kTwoPi = 6.283185307179586476925286766559;

/// Append \p name to \p failures unless it is already there, so a certificate
/// that fails for two reasons at once is still named once.
void nameFailure(std::vector<std::string> &failures, const std::string &name) {
  if (std::find(failures.begin(), failures.end(), name) == failures.end())
    failures.push_back(name);
}

/// The vertex identifiers of the complex, in canonical degree-zero cell order.
std::vector<std::uint64_t> vertexIds(const ChainComplex &complex) {
  std::vector<std::uint64_t> ids;
  for (const auto &cell : complex.kSimplexVertices(0))
    ids.push_back(cell.empty() ? 0 : cell[0]);
  return ids;
}

/// Whether the induced subgraph of the 1-skeleton on \p side is connected. An
/// empty side is not connected, since a cut with an empty side separates
/// nothing.
bool sideIsConnected(const std::vector<std::vector<std::uint64_t>> &edges,
                     const std::set<std::uint64_t> &side) {
  if (side.empty()) return false;
  std::map<std::uint64_t, std::vector<std::uint64_t>> adjacency;
  for (const auto &edge : edges) {
    if (edge.size() != 2) continue;
    if (side.count(edge[0]) == 0 || side.count(edge[1]) == 0) continue;
    adjacency[edge[0]].push_back(edge[1]);
    adjacency[edge[1]].push_back(edge[0]);
  }
  std::set<std::uint64_t> seen;
  std::deque<std::uint64_t> queue{*side.begin()};
  seen.insert(*side.begin());
  while (!queue.empty()) {
    const std::uint64_t vertex = queue.front();
    queue.pop_front();
    const auto found = adjacency.find(vertex);
    if (found == adjacency.end()) continue;
    for (const std::uint64_t neighbour : found->second) {
      if (seen.insert(neighbour).second) queue.push_back(neighbour);
    }
  }
  return seen.size() == side.size();
}

/// The integrality and reality certificates of a flux, written onto \p read.
/// The flux is a complex number and its imaginary part is never discarded: an
/// integer quark number is claimed only when the imaginary part is within
/// tolerance and the real part is within tolerance of an integer.
void readIntegerFlux(WardFluxRead &read, const WardFluxConfig &cfg) {
  const double nearest = std::round(read.flux.real());
  read.quarkNumberDefect = std::abs(read.flux - complexd{nearest, 0.0});
  const bool realEnough = std::abs(read.flux.imag()) <= cfg.imaginaryTolerance;
  const bool integral =
      std::abs(read.flux.real() - nearest) <= cfg.integralityTolerance;
  if (!realEnough) nameFailure(read.failedCertificates, "complex-flux");
  if (!integral) nameFailure(read.failedCertificates, "nonintegral-flux");
  if (realEnough && integral) {
    read.quarkNumber = static_cast<long long>(nearest);
    read.baryonNumber = nearest / 3.0;
  }
}

/// A flat row-major matrix lifted into Eigen.
Eigen::MatrixXcd toMatrix(const std::vector<complexd> &flat,
                          std::size_t order) {
  Eigen::MatrixXcd matrix(static_cast<Eigen::Index>(order),
                          static_cast<Eigen::Index>(order));
  for (std::size_t i = 0; i < order; ++i)
    for (std::size_t j = 0; j < order; ++j)
      matrix(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) =
          flat[i * order + j];
  return matrix;
}

/// The cut's cells and their coorientation, shared by both reads.
struct CutGeometry {
  std::vector<int> cells{};
  std::vector<int> coorientation{};
  std::set<std::uint64_t> inside{};
  std::set<std::uint64_t> outside{};
  std::set<std::uint64_t> touched{};
  std::vector<std::vector<std::uint64_t>> edges{};
};

CutGeometry cutGeometry(const ChainComplex &complex,
                        const CooorientedCut &cut) {
  CutGeometry geometry;
  geometry.edges = complex.kSimplexVertices(1);
  const auto ids = vertexIds(complex);
  const std::set<std::uint64_t> present(ids.begin(), ids.end());
  for (const std::uint64_t vertex : cut.incomingSide)
    if (present.count(vertex) != 0) geometry.inside.insert(vertex);
  for (const std::uint64_t vertex : present)
    if (geometry.inside.count(vertex) == 0) geometry.outside.insert(vertex);

  for (std::size_t index = 0; index < geometry.edges.size(); ++index) {
    const auto &edge = geometry.edges[index];
    if (edge.size() != 2) continue;
    const bool lowerInside = geometry.inside.count(edge[0]) != 0;
    const bool upperInside = geometry.inside.count(edge[1]) != 0;
    if (lowerInside == upperInside) continue;
    geometry.cells.push_back(static_cast<int>(index));
    geometry.coorientation.push_back(lowerInside ? +1 : -1);
    geometry.touched.insert(edge[0]);
    geometry.touched.insert(edge[1]);
  }
  return geometry;
}

}  // namespace

WardFluxRead WardFlux::flux(const JointAction &action,
                            const CooorientedCut &cut,
                            const WardFluxConfig &cfg) {
  if (!action.spacetime())
    throw std::invalid_argument(
        "WardFlux::flux: the action carries no spacetime");

  const ChainComplex complex = ChainComplex::fromSpacetime(*action.spacetime());
  const CutGeometry geometry = cutGeometry(complex, cut);
  const std::vector<complexd> current = action.canonicalWardCurrent();
  const std::vector<complexd> divergence = action.wardCurrentDivergence();
  const auto ids = vertexIds(complex);

  WardFluxRead read;
  read.label = cut.label;
  read.cutCells = geometry.cells;
  read.coorientation = geometry.coorientation;

  // 1. the flux itself: the cooriented sum of the current over the cut.
  complexd flux{0.0, 0.0};
  for (std::size_t index = 0; index < geometry.cells.size(); ++index) {
    const auto cell = static_cast<std::size_t>(geometry.cells[index]);
    const complexd value =
        cell < current.size() ? current[cell] : complexd{0.0, 0.0};
    read.cutCurrent.push_back(value);
    flux += static_cast<double>(geometry.coorientation[index]) * value;
  }
  read.flux = flux;

  // 2. the divergence theorem. With the boundary convention
  //    d[x<y] = [y] - [x] the enclosed divergence is minus the outward flux,
  //    identically; the residual says so by measurement.
  complexd enclosed{0.0, 0.0};
  double bulkMax = kNaN;
  std::optional<std::uint64_t> bulkVertex;
  std::size_t bulkVertices = 0;
  for (std::size_t index = 0; index < ids.size(); ++index) {
    if (geometry.inside.count(ids[index]) == 0) continue;
    const complexd value =
        index < divergence.size() ? divergence[index] : complexd{0.0, 0.0};
    enclosed += value;
    if (geometry.touched.count(ids[index]) != 0) continue;
    ++bulkVertices;
    const double magnitude = std::abs(value);
    if (std::isnan(bulkMax) || magnitude > bulkMax) {
      bulkMax = magnitude;
      bulkVertex = ids[index];
    }
  }
  read.enclosedDivergence = enclosed;
  read.divergenceTheoremResidual = std::abs(flux + enclosed);
  read.bulkDivergenceMax = bulkMax;
  read.bulkVertices = bulkVertices;
  read.bulkDivergenceVertex = bulkVertex;

  // 3. the number Section 13.4 states the flux equals, measured from the
  //    declared covariance rather than from the current.
  const auto occupations = action.occupationNumbers();
  if (!occupations.empty()) {
    const int degree = action.declaration().carrierDegree;
    const auto cells = complex.kSimplexVertices(degree);
    complexd enclosedFermions{0.0, 0.0};
    std::size_t enclosedCells = 0;
    for (std::size_t index = 0;
         index < cells.size() && index < occupations.size(); ++index) {
      bool wholly = !cells[index].empty();
      for (const std::uint64_t vertex : cells[index])
        wholly = wholly && geometry.inside.count(vertex) != 0;
      if (!wholly) continue;
      ++enclosedCells;
      enclosedFermions += occupations[index];
    }
    read.enclosedFermionNumber = enclosedFermions;
    read.enclosedCells = enclosedCells;
    read.fermionNumberResidual = std::abs(flux - enclosedFermions);
  }

  // 4. the certificates.
  if (geometry.inside.empty())
    nameFailure(read.failedCertificates, "empty-incoming-side");
  if (geometry.outside.empty())
    nameFailure(read.failedCertificates, "empty-outgoing-side");
  if (geometry.cells.empty()) nameFailure(read.failedCertificates, "empty-cut");
  const bool insideConnected = sideIsConnected(geometry.edges, geometry.inside);
  const bool outsideConnected =
      sideIsConnected(geometry.edges, geometry.outside);
  if (!geometry.inside.empty() && !insideConnected)
    nameFailure(read.failedCertificates, "disconnected-incoming-side");
  if (!geometry.outside.empty() && !outsideConnected)
    nameFailure(read.failedCertificates, "disconnected-outgoing-side");
  read.separating = insideConnected && outsideConnected &&
                    !geometry.cells.empty();
  if (bulkVertices > 0 && std::isfinite(bulkMax) &&
      bulkMax > cfg.divergenceTolerance)
    nameFailure(read.failedCertificates, "bulk-source");
  readIntegerFlux(read, cfg);
  return read;
}

WardHomologyRead WardFlux::homologousFluxes(
    const JointAction &action, const std::vector<CooorientedCut> &cuts,
    const WardFluxConfig &cfg) {
  if (!action.spacetime())
    throw std::invalid_argument(
        "WardFlux::homologousFluxes: the action carries no spacetime");

  WardHomologyRead read;
  read.cuts.reserve(cuts.size());
  for (const auto &cut : cuts) read.cuts.push_back(flux(action, cut, cfg));
  if (read.cuts.size() < 2) return read;

  const ChainComplex complex = ChainComplex::fromSpacetime(*action.spacetime());
  const auto ids = vertexIds(complex);
  const std::vector<complexd> divergence = action.wardCurrentDivergence();
  std::vector<std::set<std::uint64_t>> sides;
  sides.reserve(cuts.size());
  for (const auto &cut : cuts)
    sides.push_back(cutGeometry(complex, cut).inside);

  double worstDeviation = 0.0;
  double worstSlab = 0.0;
  for (std::size_t left = 0; left < read.cuts.size(); ++left) {
    for (std::size_t right = left + 1; right < read.cuts.size(); ++right) {
      worstDeviation =
          std::max(worstDeviation,
                   std::abs(read.cuts[left].flux - read.cuts[right].flux));
      double slab = 0.0;
      for (std::size_t index = 0; index < ids.size(); ++index) {
        const bool inLeft = sides[left].count(ids[index]) != 0;
        const bool inRight = sides[right].count(ids[index]) != 0;
        if (inLeft == inRight) continue;
        if (index < divergence.size()) slab += std::abs(divergence[index]);
      }
      worstSlab = std::max(worstSlab, slab);
    }
  }
  read.maxFluxDeviation = worstDeviation;
  read.maxSlabDivergence = worstSlab;
  read.invariant = worstDeviation <= cfg.divergenceTolerance;
  return read;
}

WardFluxRead WardFlux::difference(const WardFluxRead &state,
                                  const WardFluxRead &matched,
                                  const WardFluxConfig &cfg) {
  if (state.cutCells != matched.cutCells ||
      state.coorientation != matched.coorientation)
    throw std::invalid_argument(
        "WardFlux::difference: the state and the matched reference were read "
        "on different cuts, and a difference between different cuts is not a "
        "background removal");

  WardFluxRead read = state;
  read.flux = state.flux - matched.flux;
  read.enclosedDivergence =
      state.enclosedDivergence - matched.enclosedDivergence;
  read.divergenceTheoremResidual =
      std::abs(read.flux + read.enclosedDivergence);
  read.cutCurrent.clear();
  read.cutCurrent.reserve(state.cutCurrent.size());
  for (std::size_t index = 0; index < state.cutCurrent.size(); ++index)
    read.cutCurrent.push_back(state.cutCurrent[index] -
                              (index < matched.cutCurrent.size()
                                   ? matched.cutCurrent[index]
                                   : complexd{0.0, 0.0}));
  if (state.enclosedFermionNumber.has_value() &&
      matched.enclosedFermionNumber.has_value()) {
    read.enclosedFermionNumber =
        *state.enclosedFermionNumber - *matched.enclosedFermionNumber;
    read.fermionNumberResidual =
        std::abs(read.flux - *read.enclosedFermionNumber);
  } else {
    read.enclosedFermionNumber.reset();
    read.fermionNumberResidual = kNaN;
  }
  read.quarkNumber.reset();
  read.baryonNumber.reset();
  read.failedCertificates.clear();
  for (const std::string &failure : state.failedCertificates)
    if (failure != "nonintegral-flux" && failure != "complex-flux")
      nameFailure(read.failedCertificates, failure);
  readIntegerFlux(read, cfg);
  return read;
}

IntrinsicResponseRead WardFlux::intrinsicResponse(
    const JointAction &action, const CooorientedCut &cut,
    const std::vector<complexd> &samples, const IntrinsicResponseConfig &cfg) {
  if (!action.spacetime())
    throw std::invalid_argument(
        "WardFlux::intrinsicResponse: the action carries no spacetime");

  const ChainComplex complex = ChainComplex::fromSpacetime(*action.spacetime());
  const CutGeometry geometry = cutGeometry(complex, cut);
  const std::vector<complexd> current = action.canonicalWardCurrent();

  IntrinsicResponseRead read;
  read.label = cut.label;
  read.sliceCells = geometry.cells;
  const auto order = static_cast<Eigen::Index>(geometry.cells.size());
  if (order == 0) {
    nameFailure(read.failedCertificates, "empty-cut");
    return read;
  }

  // The two restrictions of the current to the cut, each carrying the cut's
  // coorientation so that the sum of either is the flux it belongs to.
  Eigen::VectorXcd rhoRight(order);
  Eigen::VectorXcd rhoLeft(order);
  for (Eigen::Index index = 0; index < order; ++index) {
    const auto position = static_cast<std::size_t>(index);
    const auto cell = static_cast<std::size_t>(geometry.cells[position]);
    const double sign = static_cast<double>(
        geometry.coorientation[static_cast<std::size_t>(index)]);
    rhoRight(index) =
        sign * (cell < current.size() ? current[cell] : complexd{0.0, 0.0});
    const complexd left = cfg.leftCurrent.empty()
                              ? (cell < current.size() ? current[cell]
                                                       : complexd{0.0, 0.0})
                              : (cell < cfg.leftCurrent.size()
                                     ? cfg.leftCurrent[cell]
                                     : complexd{0.0, 0.0});
    rhoLeft(index) = sign * left;
  }
  read.rhoRight.assign(rhoRight.data(), rhoRight.data() + order);
  read.rhoLeft.assign(rhoLeft.data(), rhoLeft.data() + order);

  // The slice operator: the degree-one metric Hodge operator of the complex,
  // restricted to the cut's cells. It is read from the framework's own
  // operator under the action's declared metric source, so the slice inherits
  // exactly the operator the action is built on.
  HodgeLaplacian hodge(action.spacetime(),
                       HodgeLaplacian::defaultWeightConvention(),
                       action.declaration().metricSource);
  const std::size_t edgeCount = complex.numSimplices(1);
  const auto flat = hodge.laplacian(1, /*metric=*/true);
  if (flat.size() != edgeCount * edgeCount) {
    nameFailure(read.failedCertificates, "singular-slice-operator");
    return read;
  }
  const Eigen::MatrixXcd full = toMatrix(flat, edgeCount);
  Eigen::MatrixXcd slice(order, order);
  for (Eigen::Index row = 0; row < order; ++row)
    for (Eigen::Index column = 0; column < order; ++column)
      slice(row, column) =
          full(geometry.cells[static_cast<std::size_t>(row)],
               geometry.cells[static_cast<std::size_t>(column)]);
  read.sliceOperator.reserve(static_cast<std::size_t>(order * order));
  for (Eigen::Index row = 0; row < order; ++row)
    for (Eigen::Index column = 0; column < order; ++column)
      read.sliceOperator.push_back(slice(row, column));

  // The poles: the eigenvalues of the slice operator, grouped into degenerate
  // bands. The Schur form is used rather than an eigenbasis, so a defective
  // slice operator still supplies its spectrum.
  Eigen::ComplexSchur<Eigen::MatrixXcd> schur(slice);
  if (schur.info() != Eigen::Success) {
    nameFailure(read.failedCertificates, "singular-slice-operator");
    return read;
  }
  std::vector<complexd> eigenvalues;
  eigenvalues.reserve(static_cast<std::size_t>(order));
  for (Eigen::Index index = 0; index < order; ++index)
    eigenvalues.push_back(schur.matrixT()(index, index));

  std::vector<bool> grouped(eigenvalues.size(), false);
  std::vector<std::vector<std::size_t>> bands;
  for (std::size_t index = 0; index < eigenvalues.size(); ++index) {
    if (grouped[index]) continue;
    std::vector<std::size_t> band{index};
    grouped[index] = true;
    for (std::size_t scan = 0; scan < band.size(); ++scan) {
      for (std::size_t other = 0; other < eigenvalues.size(); ++other) {
        if (grouped[other]) continue;
        if (std::abs(eigenvalues[band[scan]] - eigenvalues[other]) >
            cfg.degeneracyTolerance)
          continue;
        grouped[other] = true;
        band.push_back(other);
      }
    }
    bands.push_back(band);
  }

  for (const auto &band : bands) {
    complexd centre{0.0, 0.0};
    for (const std::size_t index : band) centre += eigenvalues[index];
    centre /= static_cast<double>(band.size());
    read.poles.push_back(centre);
    read.poleMultiplicity.push_back(band.size());
  }

  // The residues, taken on Riesz contours rather than on ordered eigenvectors:
  // the residue of Upsilon_Q at a band is minus the contour average of
  // rho~^T (s I - L)^{-1} rho around it, and the contour encloses the whole
  // band, so a degeneracy needs no eigenvector and no ordering.
  read.residues.reserve(read.poles.size());
  for (std::size_t band = 0; band < read.poles.size(); ++band) {
    double spread = 0.0;
    for (const std::size_t index : bands[band])
      spread =
          std::max(spread, std::abs(eigenvalues[index] - read.poles[band]));
    double nearest = std::numeric_limits<double>::infinity();
    for (std::size_t other = 0; other < read.poles.size(); ++other)
      if (other != band)
        nearest =
            std::min(nearest, std::abs(read.poles[other] - read.poles[band]));
    double radius = 0.0;
    if (std::isfinite(nearest)) {
      radius = spread + 0.25 * (nearest - spread);
    } else {
      double scale = 1.0;
      for (const complexd &value : eigenvalues)
        scale = std::max(scale, std::abs(value));
      radius = scale;
    }
    if (!(radius > 0.0)) radius = 1.0;

    const int nodes = 64;
    complexd integral{0.0, 0.0};
    bool solved = true;
    for (int node = 0; node < nodes; ++node) {
      const double angle = kTwoPi * (static_cast<double>(node) + 0.5) /
                           static_cast<double>(nodes);
      const complexd offset =
          radius * complexd{std::cos(angle), std::sin(angle)};
      const complexd point = read.poles[band] + offset;
      Eigen::MatrixXcd shifted =
          point * Eigen::MatrixXcd::Identity(order, order) - slice;
      Eigen::PartialPivLU<Eigen::MatrixXcd> lu(shifted);
      const Eigen::VectorXcd solved_vector = lu.solve(rhoRight);
      if (!solved_vector.allFinite()) {
        solved = false;
        break;
      }
      // (1/2 pi i) * integral of f(s) ds with ds = i * offset * dtheta and
      // dtheta = 2 pi / nodes reduces to the average of f(s) * offset.
      integral += (rhoLeft.transpose() * solved_vector)(0, 0) * offset;
    }
    if (!solved) {
      nameFailure(read.failedCertificates, "singular-slice-operator");
      read.residues.push_back(complexd{kNaN, kNaN});
      continue;
    }
    const complexd residue = integral / static_cast<double>(nodes);
    read.residues.push_back(-residue);
  }

  // The response and its analytic slope at the declared samples.
  read.samples = samples;
  for (const complexd &point : samples) {
    bool onPole = false;
    for (const complexd &pole : read.poles)
      onPole = onPole || std::abs(point - pole) <= cfg.poleTolerance;
    if (onPole) {
      nameFailure(read.failedCertificates, "sample-on-a-pole");
      read.response.push_back(complexd{kNaN, kNaN});
      read.slope.push_back(complexd{kNaN, kNaN});
      continue;
    }
    Eigen::MatrixXcd shifted =
        slice - point * Eigen::MatrixXcd::Identity(order, order);
    Eigen::PartialPivLU<Eigen::MatrixXcd> lu(shifted);
    const Eigen::VectorXcd first = lu.solve(rhoRight);
    const Eigen::VectorXcd second = lu.solve(first);
    if (!first.allFinite() || !second.allFinite()) {
      nameFailure(read.failedCertificates, "singular-slice-operator");
      read.response.push_back(complexd{kNaN, kNaN});
      read.slope.push_back(complexd{kNaN, kNaN});
      continue;
    }
    read.response.push_back((rhoLeft.transpose() * first)(0, 0));
    read.slope.push_back((rhoLeft.transpose() * second)(0, 0));
  }
  return read;
}

}  // namespace tessera::cobordism
