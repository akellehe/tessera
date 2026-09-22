// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "cobordism/BoundStatePole.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

namespace tessera::cobordism {

using complexd = std::complex<double>;

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kTwoPi = 6.283185307179586476925286766559;

void nameFailure(std::vector<std::string> &failures, const std::string &name) {
  if (std::find(failures.begin(), failures.end(), name) == failures.end())
    failures.push_back(name);
}

/// The interface/interior partition of a pencil, in the convention
/// `chainhodge::PencilSchur::feshbach` uses: the interface ascending and
/// deduplicated, the interior its complement, also ascending.
struct Partition {
  std::vector<int> interface{};
  std::vector<int> interior{};
};

Partition partitionOf(int order, const std::vector<int> &interface,
                      const char *who) {
  Partition partition;
  std::set<int> kept;
  for (const int index : interface) {
    if (index < 0 || index >= order)
      throw std::invalid_argument(std::string(who) +
                                  ": interface index out of range");
    kept.insert(index);
  }
  partition.interface.assign(kept.begin(), kept.end());
  for (int index = 0; index < order; ++index)
    if (kept.count(index) == 0) partition.interior.push_back(index);
  return partition;
}

/// One shift's worth of the response and its derivative, formed from a single
/// factorization of the interior block.
struct ResponseAtShift {
  Eigen::MatrixXcd response{};
  Eigen::MatrixXcd derivative{};
  bool interiorSingular{false};
};

/// A submatrix of \p source on the row set \p rows and the column set
/// \p columns.
Eigen::MatrixXcd block(const Eigen::MatrixXcd &source,
                       const std::vector<int> &rows,
                       const std::vector<int> &columns) {
  Eigen::MatrixXcd out(static_cast<Eigen::Index>(rows.size()),
                       static_cast<Eigen::Index>(columns.size()));
  for (std::size_t row = 0; row < rows.size(); ++row)
    for (std::size_t column = 0; column < columns.size(); ++column)
      out(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(column)) =
          source(rows[row], columns[column]);
  return out;
}

/// \f$ F_C(s) \f$ and \f$ F_C'(s) \f$ together.
///
/// The derivative is the closed form the header states, which follows from
/// \f$ dP/ds = -M \f$ alone; no finite difference and no step size enters it.
ResponseAtShift responseAtShift(const Eigen::MatrixXcd &A,
                                const Eigen::MatrixXcd &M,
                                const Partition &partition, complexd s,
                                double rankTolerance) {
  ResponseAtShift out;
  const Eigen::MatrixXcd P = A - s * M;
  const auto &B = partition.interface;
  const auto &I = partition.interior;
  const Eigen::MatrixXcd PBB = block(P, B, B);
  const Eigen::MatrixXcd MBB = block(M, B, B);
  if (I.empty()) {
    out.response = PBB;
    out.derivative = -MBB;
    return out;
  }
  const Eigen::MatrixXcd PBI = block(P, B, I);
  const Eigen::MatrixXcd PIB = block(P, I, B);
  const Eigen::MatrixXcd PII = block(P, I, I);
  const Eigen::MatrixXcd MBI = block(M, B, I);
  const Eigen::MatrixXcd MIB = block(M, I, B);
  const Eigen::MatrixXcd MII = block(M, I, I);

  Eigen::FullPivLU<Eigen::MatrixXcd> lu(PII);
  lu.setThreshold(rankTolerance);
  if (!lu.isInvertible()) {
    out.interiorSingular = true;
    return out;
  }
  const Eigen::MatrixXcd X = lu.solve(PIB);            // P_II^{-1} P_IB
  Eigen::FullPivLU<Eigen::MatrixXcd> transposed(PII.transpose());
  transposed.setThreshold(rankTolerance);
  const Eigen::MatrixXcd Z =
      transposed.solve(PBI.transpose()).transpose();   // P_BI P_II^{-1}
  out.response = PBB - PBI * X;
  out.derivative = -MBB + MBI * X + Z * MIB - Z * MII * X;
  return out;
}

/// \f$ \operatorname{tr}(F^{-1}F') \f$, formed by solving rather than by
/// inverting.
complexd logarithmicDerivativeOf(const ResponseAtShift &shift) {
  if (shift.interiorSingular || shift.response.rows() == 0)
    return complexd{kNaN, kNaN};
  Eigen::FullPivLU<Eigen::MatrixXcd> lu(shift.response);
  if (!lu.isInvertible()) return complexd{kNaN, kNaN};
  const Eigen::MatrixXcd solved = lu.solve(shift.derivative);
  return solved.trace();
}

/// The quadrature nodes of a circle and the factor each contributes: with
/// \f$ s=c+re^{i\theta} \f$ and \f$ ds=i(s-c)\,d\theta \f$, the contour
/// average \f$ (2\pi i)^{-1}\oint f\,ds \f$ is the mean of
/// \f$ f(s)\,(s-c) \f$ over equally spaced nodes.
struct ContourNode {
  complexd point{0.0, 0.0};
  complexd offset{0.0, 0.0};
};

std::vector<ContourNode> circle(complexd centre, double radius, int nodes) {
  std::vector<ContourNode> out;
  out.reserve(static_cast<std::size_t>(std::max(nodes, 0)));
  for (int node = 0; node < nodes; ++node) {
    const double angle =
        kTwoPi * (static_cast<double>(node) + 0.5) / static_cast<double>(nodes);
    ContourNode entry;
    entry.offset = radius * complexd{std::cos(angle), std::sin(angle)};
    entry.point = centre + entry.offset;
    out.push_back(entry);
  }
  return out;
}

/// The moments of the logarithmic derivative on one circle, for
/// \f$ p = 0 \dots \f$ \p count, together with whether any node met a singular
/// interior block.
///
/// The moments are taken in the scaled shifted variable
/// \f$ u=(s-c)/r \f$, so \f$ m_p=\sum_i u_i^{\,p} \f$ with every
/// \f$ |u_i|<1 \f$ inside the contour. That keeps every power bounded however
/// far the contour sits from the origin, and the zeros are mapped back by
/// \f$ s_i=c+r\,u_i \f$. \f$ m_0 \f$ is unchanged by the scaling and is the
/// enclosed count.
struct MomentRead {
  std::vector<complexd> moments{};
  bool interiorSingular{false};
  bool usable{true};
};

MomentRead momentsOn(const Eigen::MatrixXcd &A, const Eigen::MatrixXcd &M,
                     const Partition &partition, complexd centre, double radius,
                     int nodes, std::size_t count, double rankTolerance) {
  MomentRead read;
  read.moments.assign(count + 1, complexd{0.0, 0.0});
  const auto contour = circle(centre, radius, nodes);
  for (const ContourNode &node : contour) {
    const ResponseAtShift shift =
        responseAtShift(A, M, partition, node.point, rankTolerance);
    if (shift.interiorSingular) {
      read.interiorSingular = true;
      read.usable = false;
      return read;
    }
    const complexd value = logarithmicDerivativeOf(shift);
    if (!std::isfinite(value.real()) || !std::isfinite(value.imag())) {
      read.usable = false;
      return read;
    }
    const complexd scaled = node.offset / radius;
    complexd power{1.0, 0.0};
    for (std::size_t p = 0; p <= count; ++p) {
      read.moments[p] += power * value * node.offset;
      power *= scaled;
    }
  }
  for (complexd &moment : read.moments)
    moment /= static_cast<double>(nodes);
  return read;
}

/// The distinct zeros carried by the moments \f$ m_0\dots m_{2N-1} \f$ of
/// \f$ N \f$ enclosed zeros, through the Hankel pencil. The values returned
/// are in the scaled shifted variable \f$ u \f$ the moments were taken in.
/// Empty when the pencil is not solvable at the declared rank tolerance.
std::vector<complexd> zerosFromMoments(const std::vector<complexd> &moments,
                                       std::size_t enclosed,
                                       double rankTolerance) {
  if (enclosed == 0 || moments.size() < 2 * enclosed) return {};
  const auto n = static_cast<Eigen::Index>(enclosed);
  Eigen::MatrixXcd hankel(n, n);
  Eigen::MatrixXcd shifted(n, n);
  for (Eigen::Index row = 0; row < n; ++row)
    for (Eigen::Index column = 0; column < n; ++column) {
      hankel(row, column) =
          moments[static_cast<std::size_t>(row + column)];
      shifted(row, column) =
          moments[static_cast<std::size_t>(row + column + 1)];
    }

  // The rank of the Hankel matrix is the number of distinct zeros; a repeated
  // zero contributes one independent power-sum direction, not two.
  Eigen::JacobiSVD<Eigen::MatrixXcd> svd(hankel);
  const Eigen::VectorXd values = svd.singularValues();
  if (values.size() == 0 || values(0) <= 0.0) return {};
  Eigen::Index rank = 0;
  while (rank < values.size() && values(rank) > rankTolerance * values(0))
    ++rank;
  if (rank == 0) return {};

  const Eigen::MatrixXcd leading = hankel.topLeftCorner(rank, rank);
  const Eigen::MatrixXcd leadingShifted = shifted.topLeftCorner(rank, rank);
  Eigen::FullPivLU<Eigen::MatrixXcd> lu(leading);
  lu.setThreshold(rankTolerance);
  if (!lu.isInvertible()) return {};
  const Eigen::MatrixXcd companion = lu.solve(leadingShifted);
  Eigen::ComplexEigenSolver<Eigen::MatrixXcd> solver(companion,
                                                     /*computeEigenvectors=*/false);
  if (solver.info() != Eigen::Success) return {};
  std::vector<complexd> roots;
  roots.reserve(static_cast<std::size_t>(rank));
  for (Eigen::Index index = 0; index < rank; ++index)
    roots.push_back(solver.eigenvalues()(index));
  return roots;
}

}  // namespace

chainhodge::FeshbachResult BoundStatePole::response(
    const Eigen::MatrixXcd &A, const Eigen::MatrixXcd &M,
    const std::vector<int> &interface, complexd s, double rankTolerance) {
  return chainhodge::PencilSchur::feshbach(A, M, s, interface, rankTolerance);
}

complexd BoundStatePole::determinant(const Eigen::MatrixXcd &A,
                                     const Eigen::MatrixXcd &M,
                                     const std::vector<int> &interface,
                                     complexd s) {
  const chainhodge::FeshbachResult result =
      chainhodge::PencilSchur::feshbach(A, M, s, interface);
  if (result.interiorSingular) return complexd{kNaN, kNaN};
  return result.responseDeterminant;
}

Eigen::MatrixXcd BoundStatePole::responseDerivative(
    const Eigen::MatrixXcd &A, const Eigen::MatrixXcd &M,
    const std::vector<int> &interface, complexd s) {
  const auto order = static_cast<int>(A.rows());
  if (A.cols() != A.rows() || M.rows() != A.rows() || M.cols() != A.rows())
    throw std::invalid_argument(
        "BoundStatePole::responseDerivative: A and M must be square of the "
        "same size");
  const Partition partition =
      partitionOf(order, interface, "BoundStatePole::responseDerivative");
  return responseAtShift(A, M, partition, s, 1e-12).derivative;
}

complexd BoundStatePole::logarithmicDerivative(const Eigen::MatrixXcd &A,
                                               const Eigen::MatrixXcd &M,
                                               const std::vector<int> &interface,
                                               complexd s) {
  const auto order = static_cast<int>(A.rows());
  if (A.cols() != A.rows() || M.rows() != A.rows() || M.cols() != A.rows())
    throw std::invalid_argument(
        "BoundStatePole::logarithmicDerivative: A and M must be square of the "
        "same size");
  const Partition partition =
      partitionOf(order, interface, "BoundStatePole::logarithmicDerivative");
  return logarithmicDerivativeOf(
      responseAtShift(A, M, partition, s, 1e-12));
}

BoundStatePoleRead BoundStatePole::poles(const Eigen::MatrixXcd &A,
                                         const Eigen::MatrixXcd &M,
                                         const std::vector<int> &interface,
                                         complexd centre, double radius,
                                         const BoundStatePoleConfig &cfg) {
  if (A.cols() != A.rows() || M.rows() != A.rows() || M.cols() != A.rows())
    throw std::invalid_argument(
        "BoundStatePole::poles: A and M must be square of the same size");
  if (!(radius > 0.0))
    throw std::invalid_argument(
        "BoundStatePole::poles: the contour radius must be positive");
  const auto order = static_cast<int>(A.rows());
  const Partition partition =
      partitionOf(order, interface, "BoundStatePole::poles");

  BoundStatePoleRead read;
  read.centre = centre;
  read.radius = radius;
  read.nodes = cfg.contourNodes;
  if (partition.interface.empty()) {
    nameFailure(read.failedCertificates, "empty-interface");
    return read;
  }

  // 1. the argument principle on the declared contour. The zero count is read
  //    before any root is claimed, so a contour that does not separate its
  //    zeros refuses instead of answering.
  const std::size_t momentCount = 2 * cfg.maxZeros + 1;
  MomentRead counted =
      momentsOn(A, M, partition, centre, radius, cfg.contourNodes, momentCount,
                cfg.rankTolerance);
  read.interiorResonance = counted.interiorSingular;
  if (counted.interiorSingular)
    nameFailure(read.failedCertificates, "interior-resonance");
  if (!counted.usable) {
    read.zeroCount = complexd{kNaN, kNaN};
    read.zeroCountDefect = kNaN;
    return read;
  }
  read.zeroCount = counted.moments[0];
  const double rounded = std::round(read.zeroCount.real());
  read.zeroCountDefect =
      std::abs(read.zeroCount - complexd{rounded, 0.0});
  if (read.zeroCountDefect > cfg.zeroCountTolerance) {
    nameFailure(read.failedCertificates, "nonintegral-zero-count");
    return read;
  }
  if (rounded <= 0.0) {
    read.zeros = 0;
    nameFailure(read.failedCertificates, "no-zero-enclosed");
    return read;
  }
  if (rounded > static_cast<double>(cfg.maxZeros)) {
    nameFailure(read.failedCertificates, "too-many-zeros");
    return read;
  }
  read.zeros = static_cast<std::size_t>(rounded);

  // 2. the distinct zeros from the moments, mapped back out of the scaled
  //    shifted variable the moments were taken in.
  std::vector<complexd> roots =
      zerosFromMoments(counted.moments, read.zeros, cfg.rankTolerance);
  if (roots.empty()) {
    nameFailure(read.failedCertificates, "roots-not-separated");
    return read;
  }
  for (complexd &root : roots) root = centre + radius * root;

  // 3. each root's multiplicity, on a small contour that encloses it alone.
  auto localRadiusFor = [&](std::size_t index) {
    double nearest = std::numeric_limits<double>::infinity();
    for (std::size_t other = 0; other < roots.size(); ++other)
      if (other != index)
        nearest = std::min(nearest, std::abs(roots[other] - roots[index]));
    const double reference = std::isfinite(nearest) ? nearest : radius;
    return cfg.localRadiusFraction * reference;
  };

  std::vector<std::size_t> multiplicity(roots.size(), 1);
  for (std::size_t index = 0; index < roots.size(); ++index) {
    const double local = localRadiusFor(index);
    if (!(local > 0.0)) continue;
    const MomentRead around = momentsOn(A, M, partition, roots[index], local,
                                        cfg.contourNodes, 0, cfg.rankTolerance);
    if (!around.usable) continue;
    const double count = std::round(around.moments[0].real());
    if (count >= 1.0) multiplicity[index] = static_cast<std::size_t>(count);
  }

  // 4. Newton on D_C through the logarithmic derivative, with the step scaled
  //    by the multiplicity so a multiple root converges like a simple one.
  std::vector<double> lastStep(roots.size(), kNaN);
  for (std::size_t index = 0; index < roots.size(); ++index) {
    complexd point = roots[index];
    for (int step = 0; step < cfg.maxNewtonSteps; ++step) {
      const ResponseAtShift shift =
          responseAtShift(A, M, partition, point, cfg.rankTolerance);
      const complexd slope = logarithmicDerivativeOf(shift);
      if (!std::isfinite(slope.real()) || !std::isfinite(slope.imag())) break;
      if (std::abs(slope) == 0.0) break;
      const complexd correction =
          static_cast<double>(multiplicity[index]) / slope;
      point -= correction;
      lastStep[index] = std::abs(correction);
      if (lastStep[index] <= cfg.newtonTolerance * radius) break;
    }
    roots[index] = point;
  }

  // 5. the reported quantities at each refined root.
  read.poles = roots;
  read.multiplicity = multiplicity;
  read.newtonStep = lastStep;
  for (std::size_t index = 0; index < roots.size(); ++index) {
    const chainhodge::FeshbachResult at = chainhodge::PencilSchur::feshbach(
        A, M, roots[index], partition.interface, cfg.rankTolerance);
    read.determinantAtPole.push_back(
        at.interiorSingular ? complexd{kNaN, kNaN} : at.responseDeterminant);

    // D' = (D'/D) * D, from the logarithmic derivative and the determinant,
    // so the derivative is the analytic one and not a difference quotient.
    const ResponseAtShift shift =
        responseAtShift(A, M, partition, roots[index], cfg.rankTolerance);
    const complexd slope = logarithmicDerivativeOf(shift);
    complexd derivative{kNaN, kNaN};
    if (!at.interiorSingular && std::isfinite(slope.real()) &&
        std::isfinite(slope.imag()))
      derivative = slope * at.responseDeterminant;
    read.derivativeAtPole.push_back(derivative);
    read.simple.push_back(multiplicity[index] == 1 &&
                          std::isfinite(derivative.real()) &&
                          std::isfinite(derivative.imag()) &&
                          std::abs(derivative) > 0.0);

    double nearest = std::numeric_limits<double>::infinity();
    for (std::size_t other = 0; other < roots.size(); ++other)
      if (other != index)
        nearest = std::min(nearest, std::abs(roots[other] - roots[index]));
    read.separation.push_back(nearest);

    if (cfg.freeThreshold.has_value())
      read.bindingShift.push_back(roots[index] - *cfg.freeThreshold);
  }

  // 6. the residue of the supported resolvent, as a contour average of
  //    F_C(s)^{-1} on a circle enclosing the root alone. A multiple root keeps
  //    its whole residue matrix; nothing is reduced to a single number.
  const auto interfaceOrder =
      static_cast<Eigen::Index>(partition.interface.size());
  for (std::size_t index = 0; index < roots.size(); ++index) {
    const double local = localRadiusFor(index);
    Eigen::MatrixXcd accumulated =
        Eigen::MatrixXcd::Zero(interfaceOrder, interfaceOrder);
    bool usable = local > 0.0;
    if (usable) {
      const auto contour = circle(roots[index], local, cfg.contourNodes);
      for (const ContourNode &node : contour) {
        const ResponseAtShift shift =
            responseAtShift(A, M, partition, node.point, cfg.rankTolerance);
        if (shift.interiorSingular) {
          usable = false;
          break;
        }
        Eigen::FullPivLU<Eigen::MatrixXcd> lu(shift.response);
        lu.setThreshold(cfg.rankTolerance);
        if (!lu.isInvertible()) {
          usable = false;
          break;
        }
        accumulated += lu.solve(Eigen::MatrixXcd::Identity(
                            interfaceOrder, interfaceOrder)) *
                       node.offset;
      }
      accumulated /= static_cast<double>(cfg.contourNodes);
    }
    std::vector<complexd> flat;
    if (usable) {
      flat.reserve(static_cast<std::size_t>(interfaceOrder * interfaceOrder));
      for (Eigen::Index row = 0; row < interfaceOrder; ++row)
        for (Eigen::Index column = 0; column < interfaceOrder; ++column)
          flat.push_back(accumulated(row, column));
      read.residueNorm.push_back(accumulated.norm());
      Eigen::JacobiSVD<Eigen::MatrixXcd> svd(accumulated);
      const Eigen::VectorXd values = svd.singularValues();
      std::size_t rank = 0;
      if (values.size() > 0 && values(0) > 0.0)
        while (rank < static_cast<std::size_t>(values.size()) &&
               values(static_cast<Eigen::Index>(rank)) >
                   cfg.rankTolerance * values(0))
          ++rank;
      read.residueRank.push_back(rank);
    } else {
      read.residueNorm.push_back(kNaN);
      read.residueRank.push_back(0);
    }
    read.residue.push_back(std::move(flat));
  }

  // 7. the refinement continuation: the same search at a second node count,
  //    with each root matched to its nearest partner.
  if (cfg.refinementNodes > 0 && cfg.refinementNodes != cfg.contourNodes) {
    const MomentRead refined =
        momentsOn(A, M, partition, centre, radius, cfg.refinementNodes,
                  momentCount, cfg.rankTolerance);
    std::vector<complexd> refinedRoots;
    if (refined.usable)
      refinedRoots =
          zerosFromMoments(refined.moments, read.zeros, cfg.rankTolerance);
    for (complexd &root : refinedRoots) root = centre + radius * root;
    for (std::size_t index = 0; index < roots.size(); ++index) {
      complexd best{kNaN, kNaN};
      double bestDistance = std::numeric_limits<double>::infinity();
      for (const complexd &candidate : refinedRoots) {
        const double distance = std::abs(candidate - roots[index]);
        if (distance < bestDistance) {
          bestDistance = distance;
          best = candidate;
        }
      }
      read.continuedPole.push_back(best);
      read.continuationMovement.push_back(
          std::isfinite(bestDistance) ? bestDistance : kNaN);
    }
  }
  return read;
}

BoundStatePoleRead BoundStatePole::clusterPoles(
    const AssembledPencil &assembled, int k,
    const std::vector<int> &clusterCells, complexd centre, double radius,
    const BoundStatePoleConfig &cfg) {
  const chainhodge::Pencil pencil = PencilLayer::pencil(assembled, k);
  return poles(pencil.A, pencil.B, clusterCells, centre, radius, cfg);
}

}  // namespace tessera::cobordism
