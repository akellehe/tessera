// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "cobordism/WardFlux.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

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

/// The agreement of the flux with the incoming boundary charge, written onto
/// \p read. Named as a failure when the two differ by more than the declared
/// tolerance; nothing is named when no covariance is declared, since then
/// there is no incoming charge to compare with.
void readChargeAgreement(WardFluxRead &read, const WardFluxConfig &cfg) {
  if (!read.incomingBoundaryCharge.has_value()) {
    read.boundaryChargeResidual = kNaN;
    return;
  }
  read.boundaryChargeResidual =
      std::abs(read.flux - *read.incomingBoundaryCharge);
  if (!(read.boundaryChargeResidual <= cfg.chargeTolerance))
    nameFailure(read.failedCertificates, "flux-is-not-the-incoming-charge");
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

/// The action's chain complex, checked to be a triangulation of \p W: the
/// same vertices and the same edges in the same canonical order, so that a
/// cut's crossing-edge indices and the current's indices name the same edges.
/// @throws std::invalid_argument when the action carries no spacetime, when
///   the two complexes differ, or when the cut does not assign one side per
///   vertex of \p W.
ChainComplex complexOverCobordism(const char *caller, const JointAction &action,
                                  const observables::InteractionCobordism &W,
                                  const observables::CoorientedCut *cut) {
  if (!action.spacetime())
    throw std::invalid_argument(std::string(caller) +
                                ": the action carries no spacetime");
  ChainComplex complex = ChainComplex::fromSpacetime(*action.spacetime());
  if (complex.kSimplexVertices(0) != W.complex.kSimplexVertices(0) ||
      complex.kSimplexVertices(1) != W.edges)
    throw std::invalid_argument(
        std::string(caller) +
        ": the action is not declared over the cobordism: its complex does "
        "not carry exactly the vertices and edges of W");
  if (cut != nullptr && cut->side.size() != W.levelOf.size())
    throw std::invalid_argument(
        std::string(caller) +
        ": the cut does not assign one side to every vertex of W");
  return complex;
}

/// Whether a vertex of \p W lies on its incoming boundary, its outgoing
/// boundary, or in its interior.
enum class Stratum { Incoming, Interior, Outgoing };

Stratum stratumOf(const observables::InteractionCobordism &W,
                  std::uint64_t vertex) {
  const std::size_t level = W.levelOf[static_cast<std::size_t>(vertex)];
  if (level == 0) return Stratum::Incoming;
  if (level + 1 >= W.levels) return Stratum::Outgoing;
  return Stratum::Interior;
}

}  // namespace

WardFluxRead WardFlux::flux(const JointAction &action,
                            const observables::InteractionCobordism &cobordism,
                            const observables::CoorientedCut &cut,
                            const WardFluxConfig &cfg) {
  const ChainComplex complex =
      complexOverCobordism("WardFlux::flux", action, cobordism, &cut);
  const std::vector<complexd> current = action.canonicalWardCurrent();
  const std::vector<complexd> divergence = action.wardCurrentDivergence();
  const auto ids = vertexIds(complex);

  WardFluxRead read;
  read.crossingEdges = cut.crossingEdges;
  read.crossingSigns = cut.crossingSigns;
  read.cutSeparates = cut.separates;

  // 1. the flux itself: the pairing of the current with delta u, which is the
  //    cooriented sum of the current over the crossing edges.
  complexd flux{0.0, 0.0};
  for (std::size_t index = 0; index < cut.crossingEdges.size(); ++index) {
    const auto cell = static_cast<std::size_t>(cut.crossingEdges[index]);
    const complexd value =
        cell < current.size() ? current[cell] : complexd{0.0, 0.0};
    read.crossingCurrent.push_back(value);
    flux += static_cast<double>(cut.crossingSigns[index]) * value;
  }
  read.flux = flux;

  // 2. the divergence theorem, the charge the current brings in through the
  //    incoming boundary, and the Ward identity at the interior vertices.
  complexd incomingSide{0.0, 0.0};
  complexd incomingBoundary{0.0, 0.0};
  double bulkMax = kNaN;
  std::optional<std::uint64_t> bulkVertex;
  std::size_t bulkVertices = 0;
  for (std::size_t index = 0; index < ids.size(); ++index) {
    const std::uint64_t vertex = ids[index];
    const complexd value =
        index < divergence.size() ? divergence[index] : complexd{0.0, 0.0};
    if (cut.side[static_cast<std::size_t>(vertex)] == 0) incomingSide += value;
    const Stratum stratum = stratumOf(cobordism, vertex);
    if (stratum == Stratum::Incoming) incomingBoundary += value;
    if (stratum != Stratum::Interior) continue;
    ++bulkVertices;
    const double magnitude = std::abs(value);
    if (std::isnan(bulkMax) || magnitude > bulkMax) {
      bulkMax = magnitude;
      bulkVertex = vertex;
    }
  }
  read.incomingSideDivergence = incomingSide;
  read.divergenceTheoremResidual = std::abs(flux + incomingSide);
  read.incomingBoundaryDivergence = -incomingBoundary;
  read.bulkDivergenceMax = bulkMax;
  read.bulkVertices = bulkVertices;
  read.bulkDivergenceVertex = bulkVertex;

  // 3. the incoming and outgoing states' charges, read from the declared
  //    covariance on the carrier cells lying wholly in each boundary, and not
  //    from the current.
  const auto occupations = action.occupationNumbers();
  if (!occupations.empty()) {
    const int degree = action.declaration().carrierDegree;
    const auto cells = complex.kSimplexVertices(degree);
    complexd incomingCharge{0.0, 0.0};
    complexd outgoingCharge{0.0, 0.0};
    for (std::size_t index = 0;
         index < cells.size() && index < occupations.size(); ++index) {
      if (cells[index].empty()) continue;
      bool incoming = true;
      bool outgoing = true;
      for (const std::uint64_t vertex : cells[index]) {
        const Stratum stratum = stratumOf(cobordism, vertex);
        incoming = incoming && stratum == Stratum::Incoming;
        outgoing = outgoing && stratum == Stratum::Outgoing;
      }
      if (incoming) {
        ++read.incomingBoundaryCells;
        incomingCharge += occupations[index];
      }
      if (outgoing) {
        ++read.outgoingBoundaryCells;
        outgoingCharge += occupations[index];
      }
    }
    read.incomingBoundaryCharge = incomingCharge;
    read.outgoingBoundaryCharge = outgoingCharge;
  }

  // 4. the certificates.
  if (!cut.separates)
    nameFailure(read.failedCertificates, "cut-does-not-separate");
  if (cut.crossingEdges.empty())
    nameFailure(read.failedCertificates, "empty-cut");
  if (bulkVertices == 0)
    nameFailure(read.failedCertificates, "no-interior-vertex");
  else if (!(bulkMax <= cfg.divergenceTolerance))
    nameFailure(read.failedCertificates, "bulk-source");
  readChargeAgreement(read, cfg);
  readIntegerFlux(read, cfg);
  return read;
}

WardHomologyRead WardFlux::homologousFluxes(
    const JointAction &action,
    const observables::InteractionCobordism &cobordism,
    const std::vector<observables::CoorientedCut> &cuts,
    const WardFluxConfig &cfg) {
  const ChainComplex complex = complexOverCobordism(
      "WardFlux::homologousFluxes", action, cobordism, nullptr);

  WardHomologyRead read;
  read.cuts.reserve(cuts.size());
  for (const auto &cut : cuts)
    read.cuts.push_back(flux(action, cobordism, cut, cfg));
  if (read.cuts.size() < 2) return read;

  const auto ids = vertexIds(complex);
  const std::vector<complexd> divergence = action.wardCurrentDivergence();

  double worstDeviation = 0.0;
  double worstSlab = 0.0;
  for (std::size_t left = 0; left < cuts.size(); ++left) {
    for (std::size_t right = left + 1; right < cuts.size(); ++right) {
      WardSlabRead slab;
      slab.first = left;
      slab.second = right;
      slab.fluxDifference = read.cuts[left].flux - read.cuts[right].flux;
      double slabMagnitude = 0.0;
      for (std::size_t index = 0; index < ids.size(); ++index) {
        const auto vertex = static_cast<std::size_t>(ids[index]);
        const int weight = cuts[left].side[vertex] - cuts[right].side[vertex];
        if (weight == 0) continue;
        slab.slabVertices.push_back(ids[index]);
        const complexd value =
            index < divergence.size() ? divergence[index] : complexd{0.0, 0.0};
        slab.slabDivergence += static_cast<double>(weight) * value;
        slabMagnitude += std::abs(value);
      }
      slab.slabIdentityResidual =
          std::abs(slab.fluxDifference - slab.slabDivergence);
      worstDeviation = std::max(worstDeviation, std::abs(slab.fluxDifference));
      worstSlab = std::max(worstSlab, slabMagnitude);
      read.slabs.push_back(std::move(slab));
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
  if (state.crossingEdges != matched.crossingEdges ||
      state.crossingSigns != matched.crossingSigns)
    throw std::invalid_argument(
        "WardFlux::difference: the state and the matched reference were read "
        "on different cuts, and a difference between different cuts is not a "
        "background removal");

  WardFluxRead read = state;
  read.flux = state.flux - matched.flux;
  read.incomingSideDivergence =
      state.incomingSideDivergence - matched.incomingSideDivergence;
  read.divergenceTheoremResidual =
      std::abs(read.flux + read.incomingSideDivergence);
  read.incomingBoundaryDivergence =
      state.incomingBoundaryDivergence - matched.incomingBoundaryDivergence;
  read.crossingCurrent.clear();
  read.crossingCurrent.reserve(state.crossingCurrent.size());
  for (std::size_t index = 0; index < state.crossingCurrent.size(); ++index)
    read.crossingCurrent.push_back(state.crossingCurrent[index] -
                                   (index < matched.crossingCurrent.size()
                                        ? matched.crossingCurrent[index]
                                        : complexd{0.0, 0.0}));
  const auto subtract = [](const std::optional<complexd> &a,
                           const std::optional<complexd> &b) {
    return a.has_value() && b.has_value() ? std::optional<complexd>(*a - *b)
                                          : std::optional<complexd>{};
  };
  read.incomingBoundaryCharge =
      subtract(state.incomingBoundaryCharge, matched.incomingBoundaryCharge);
  read.outgoingBoundaryCharge =
      subtract(state.outgoingBoundaryCharge, matched.outgoingBoundaryCharge);
  read.quarkNumber.reset();
  read.baryonNumber.reset();
  read.failedCertificates.clear();
  for (const std::string &failure : state.failedCertificates)
    if (failure != "nonintegral-flux" && failure != "complex-flux" &&
        failure != "flux-is-not-the-incoming-charge")
      nameFailure(read.failedCertificates, failure);
  readChargeAgreement(read, cfg);
  readIntegerFlux(read, cfg);
  return read;
}

IntrinsicResponseRead WardFlux::intrinsicResponse(
    const JointAction &action,
    const observables::InteractionCobordism &cobordism,
    const observables::CoorientedCut &cut, const std::vector<complexd> &samples,
    const IntrinsicResponseConfig &cfg) {
  const ChainComplex complex = complexOverCobordism(
      "WardFlux::intrinsicResponse", action, cobordism, &cut);
  const std::vector<complexd> current = action.canonicalWardCurrent();

  IntrinsicResponseRead read;
  read.sliceCells = cut.crossingEdges;
  const auto order = static_cast<Eigen::Index>(cut.crossingEdges.size());
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
    const auto cell = static_cast<std::size_t>(cut.crossingEdges[position]);
    const double sign = static_cast<double>(
        cut.crossingSigns[static_cast<std::size_t>(index)]);
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
          full(cut.crossingEdges[static_cast<std::size_t>(row)],
               cut.crossingEdges[static_cast<std::size_t>(column)]);
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
    // The contour sits between the band's own extent and the nearest other
    // band, so it encloses this band whole and no part of any other. A band
    // whose extent reaches the next band cannot be enclosed alone, and its
    // residue is refused rather than read on a contour that straddles two.
    if (!(nearest > spread)) {
      nameFailure(read.failedCertificates, "bands-not-separated");
      read.residues.push_back(complexd{kNaN, kNaN});
      continue;
    }
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

    complexd integral{0.0, 0.0};
    bool resolved = true;
    for (int node = 0; node < cfg.contourNodes; ++node) {
      const double angle = kTwoPi * (static_cast<double>(node) + 0.5) /
                           static_cast<double>(cfg.contourNodes);
      const complexd offset =
          radius * complexd{std::cos(angle), std::sin(angle)};
      const complexd point = read.poles[band] + offset;
      const Eigen::MatrixXcd shifted =
          point * Eigen::MatrixXcd::Identity(order, order) - slice;
      const Eigen::PartialPivLU<Eigen::MatrixXcd> lu(shifted);
      const Eigen::VectorXcd column = lu.solve(rhoRight);
      if (!column.allFinite()) {
        resolved = false;
        break;
      }
      // (1/2 pi i) * integral of f(s) ds with ds = i * offset * dtheta and
      // dtheta = 2 pi / nodes reduces to the average of f(s) * offset.
      integral += (rhoLeft.transpose() * column)(0, 0) * offset;
    }
    if (!resolved) {
      nameFailure(read.failedCertificates, "singular-slice-operator");
      read.residues.push_back(complexd{kNaN, kNaN});
      continue;
    }
    const complexd residue =
        integral / static_cast<double>(cfg.contourNodes);
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
