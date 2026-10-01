// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "cobordism/BoundStatePole.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <set>
#include <stdexcept>
#include <utility>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include "chainhodge/RieszProjector.h"

namespace tessera::cobordism {

using complexd = std::complex<double>;

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

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

void requireSquarePair(const Eigen::MatrixXcd &A, const Eigen::MatrixXcd &M,
                       const char *who) {
  if (A.cols() != A.rows() || M.rows() != A.rows() || M.cols() != A.rows())
    throw std::invalid_argument(std::string(who) +
                                ": A and M must be square of the same size");
}

/// The number of singular values of \p X above \p threshold.
std::size_t rankAbove(const Eigen::MatrixXcd &X, double threshold) {
  if (X.rows() == 0 || X.cols() == 0) return 0;
  const Eigen::VectorXd values = Eigen::JacobiSVD<Eigen::MatrixXcd>(X)
                                     .singularValues();
  std::size_t rank = 0;
  for (Eigen::Index index = 0; index < values.size(); ++index)
    if (values(index) > threshold) ++rank;
  return rank;
}

/// The largest singular value of \p X, zero for an empty matrix.
double largestSingularValue(const Eigen::MatrixXcd &X) {
  if (X.rows() == 0 || X.cols() == 0) return 0.0;
  return Eigen::JacobiSVD<Eigen::MatrixXcd>(X).singularValues()(0);
}

/// \f$ M^{-1}A \f$ with \f$ M \f$ required invertible at the declared
/// relative tolerance of its pivots; the read refuses by name otherwise,
/// because the spectrum of a pencil with a singular metric is not the
/// spectrum of a matrix.
struct MetricInverse {
  Eigen::MatrixXcd inverse{};
  Eigen::MatrixXcd product{};
};

MetricInverse metricInverse(const Eigen::MatrixXcd &A,
                            const Eigen::MatrixXcd &M, double rankTolerance,
                            const char *what) {
  MetricInverse out;
  if (M.rows() == 0) {
    out.inverse = Eigen::MatrixXcd(0, 0);
    out.product = Eigen::MatrixXcd(0, 0);
    return out;
  }
  Eigen::FullPivLU<Eigen::MatrixXcd> lu(M);
  lu.setThreshold(rankTolerance);
  if (!lu.isInvertible())
    throw std::invalid_argument(
        std::string("BoundStatePole::poles: ") + what +
        " is singular at the declared rank tolerance, so the pencil's "
        "spectrum is not the spectrum of a matrix and its poles are not "
        "read");
  out.inverse = lu.inverse();
  out.product = lu.solve(A);
  return out;
}

/// The eigenvalues of \p X, as the diagonal of its complex Schur form, and the
/// clusters they form at the distance \p threshold: a cluster is the
/// transitive closure of "at distance at or below the threshold", so two
/// eigenvalues in different clusters are farther apart than the threshold.
struct ClusteredSpectrum {
  Eigen::ComplexSchur<Eigen::MatrixXcd> schur{};
  std::vector<complexd> eigenvalues{};
  std::vector<std::vector<std::size_t>> clusters{};
};

ClusteredSpectrum clusteredSpectrum(const Eigen::MatrixXcd &X,
                                    double threshold, bool withVectors,
                                    const char *what) {
  ClusteredSpectrum out;
  if (X.rows() == 0) return out;
  out.schur.compute(X, withVectors);
  if (out.schur.info() != Eigen::Success)
    throw std::runtime_error(std::string("BoundStatePole::poles: the complex "
                                         "Schur form of ") +
                             what + " did not converge");
  const Eigen::Index order = X.rows();
  out.eigenvalues.reserve(static_cast<std::size_t>(order));
  for (Eigen::Index index = 0; index < order; ++index)
    out.eigenvalues.push_back(out.schur.matrixT()(index, index));
  std::vector<bool> grouped(out.eigenvalues.size(), false);
  for (std::size_t index = 0; index < out.eigenvalues.size(); ++index) {
    if (grouped[index]) continue;
    std::vector<std::size_t> cluster{index};
    grouped[index] = true;
    for (std::size_t scan = 0; scan < cluster.size(); ++scan)
      for (std::size_t other = 0; other < out.eigenvalues.size(); ++other) {
        if (grouped[other]) continue;
        if (std::abs(out.eigenvalues[cluster[scan]] -
                     out.eigenvalues[other]) > threshold)
          continue;
        grouped[other] = true;
        cluster.push_back(other);
      }
    out.clusters.push_back(std::move(cluster));
  }
  return out;
}

/// The mean of the eigenvalues of one cluster, which is the trace of the
/// cluster's Schur block over its size.
complexd clusterValue(const ClusteredSpectrum &spectrum,
                      const std::vector<std::size_t> &cluster) {
  complexd total{0.0, 0.0};
  for (const std::size_t index : cluster) total += spectrum.eigenvalues[index];
  return total / static_cast<double>(cluster.size());
}

/// The diameter of one cluster: the largest distance between two of its
/// eigenvalues.
double clusterDiameter(const ClusteredSpectrum &spectrum,
                       const std::vector<std::size_t> &cluster) {
  double diameter = 0.0;
  for (const std::size_t left : cluster)
    for (const std::size_t right : cluster)
      diameter = std::max(diameter, std::abs(spectrum.eigenvalues[left] -
                                             spectrum.eigenvalues[right]));
  return diameter;
}

bool ascendingByParts(complexd left, complexd right) {
  if (left.real() != right.real()) return left.real() < right.real();
  return left.imag() < right.imag();
}

/// \f$ F_C(s) \f$ and \f$ F_C'(s) \f$ together, from one factorization of the
/// interior block. The derivative is the closed form the header states, which
/// follows from \f$ dP/ds = -M \f$ alone.
struct ResponseAtShift {
  Eigen::MatrixXcd response{};
  Eigen::MatrixXcd derivative{};
  bool interiorSingular{false};
};

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
  requireSquarePair(A, M, "BoundStatePole::responseDerivative");
  const Partition partition = partitionOf(
      static_cast<int>(A.rows()), interface,
      "BoundStatePole::responseDerivative");
  return responseAtShift(A, M, partition, s, 1e-12).derivative;
}

BoundStatePoleRead BoundStatePole::poles(const Eigen::MatrixXcd &A,
                                         const Eigen::MatrixXcd &M,
                                         const std::vector<int> &interface,
                                         const BoundStatePoleConfig &cfg) {
  requireSquarePair(A, M, "BoundStatePole::poles");
  const auto order = static_cast<int>(A.rows());
  const Partition partition =
      partitionOf(order, interface, "BoundStatePole::poles");

  BoundStatePoleRead read;
  if (partition.interface.empty()) {
    nameFailure(read.failedCertificates, "empty-interface");
    return read;
  }

  // 1. The block whose spectrum is the spectrum of the pencil, T = M^{-1} A,
  //    and the scale every decision of the read is measured against.
  const MetricInverse metric =
      metricInverse(A, M, cfg.rankTolerance, "the metric block");
  const Eigen::MatrixXcd &T = metric.product;
  read.scale = largestSingularValue(T);
  const double threshold = cfg.rankTolerance * read.scale;

  // 2. The eigenvalues of the pencil, clustered at the declared tolerance, and
  //    those of the interior pencil, which are the poles of the response and
  //    the points its domain excludes.
  const ClusteredSpectrum full =
      clusteredSpectrum(T, threshold, true, "the block");
  const Eigen::MatrixXcd AII = block(A, partition.interior, partition.interior);
  const Eigen::MatrixXcd MII = block(M, partition.interior, partition.interior);
  const MetricInverse interiorMetric = metricInverse(
      AII, MII, cfg.rankTolerance, "the interior block of the metric");
  const ClusteredSpectrum interior = clusteredSpectrum(
      interiorMetric.product, threshold, false, "the interior block");
  {
    std::vector<std::pair<complexd, std::size_t>> listed;
    for (const auto &cluster : interior.clusters)
      listed.emplace_back(clusterValue(interior, cluster), cluster.size());
    std::sort(listed.begin(), listed.end(), [](const auto &a, const auto &b) {
      return ascendingByParts(a.first, b.first);
    });
    for (const auto &[value, count] : listed) {
      read.interiorPoles.push_back(value);
      read.interiorMultiplicity.push_back(count);
    }
  }

  // 3. The clusters that are zeros of D_C: those no interior eigenvalue meets.
  //    A cluster an interior eigenvalue meets lies outside the domain the
  //    response is continued on and is named rather than reported.
  struct Cluster {
    complexd value{0.0, 0.0};
    std::vector<std::size_t> members{};
  };
  std::vector<Cluster> reported;
  for (const auto &members : full.clusters) {
    bool meetsInterior = false;
    for (const std::size_t member : members)
      for (const complexd &pole : interior.eigenvalues)
        if (std::abs(full.eigenvalues[member] - pole) <= threshold)
          meetsInterior = true;
    if (meetsInterior) {
      nameFailure(read.failedCertificates, "eigenvalue-at-interior-pole");
      continue;
    }
    reported.push_back(Cluster{clusterValue(full, members), members});
  }
  std::sort(reported.begin(), reported.end(),
            [](const Cluster &a, const Cluster &b) {
              return ascendingByParts(a.value, b.value);
            });

  // 4. Each pole: its spectral projector from the reordered Schur form, the
  //    Jordan structure of its Schur block, the residual of its invariant
  //    subspace and the residue of the supported resolvent.
  const auto interfaceOrder =
      static_cast<Eigen::Index>(partition.interface.size());
  for (const Cluster &cluster : reported) {
    const auto size = static_cast<Eigen::Index>(cluster.members.size());
    std::vector<bool> flagged(full.eigenvalues.size(), false);
    for (const std::size_t member : cluster.members) flagged[member] = true;
    const chainhodge::RieszProjectorRead riesz =
        chainhodge::rieszProjector(full.schur, flagged);
    const Eigen::MatrixXcd &V = riesz.right;
    const Eigen::MatrixXcd U11 = V.adjoint() * T * V;

    read.poles.push_back(cluster.value);
    read.multiplicity.push_back(cluster.members.size());
    read.clusterSpread.push_back(clusterDiameter(full, cluster.members));
    read.simple.push_back(cluster.members.size() == 1);
    read.subspaceResidual.push_back((T * V - V * U11).norm());

    // The nilpotent part N = U11 - mu I, scaled by the block's largest
    // singular value so that its powers are compared with the declared
    // tolerance on one footing; a zero block has a zero nilpotent part and
    // nothing to scale. With r_j the rank of N^j at the tolerance (r_0 the
    // size of the cluster), the number of Jordan blocks of size at least j
    // is r_{j-1} - r_j, so the number of blocks of size exactly j is
    // (r_{j-1} - r_j) - (r_j - r_{j+1}). The ranks of the powers of a
    // nilpotent matrix reach zero within its size and their differences do
    // not increase; a rank sequence read at the tolerance that fails either
    // is not the rank sequence of a nilpotent matrix, and the block sizes are
    // then unmeasured and named rather than reported as a list that does not
    // sum to the multiplicity.
    Eigen::MatrixXcd nilpotent =
        U11 - cluster.value * Eigen::MatrixXcd::Identity(size, size);
    if (read.scale > 0.0) nilpotent /= read.scale;
    std::vector<std::size_t> ranks{static_cast<std::size_t>(size)};
    Eigen::MatrixXcd power = Eigen::MatrixXcd::Identity(size, size);
    for (Eigen::Index exponent = 1; exponent <= size; ++exponent) {
      power = (power * nilpotent).eval();
      ranks.push_back(rankAbove(power, cfg.rankTolerance));
      if (ranks.back() == 0) break;
    }
    bool nilpotentSequence = ranks.back() == 0;
    for (std::size_t j = 1; j + 1 < ranks.size(); ++j)
      if (ranks[j] > ranks[j - 1] ||
          ranks[j - 1] - ranks[j] < ranks[j] - ranks[j + 1])
        nilpotentSequence = false;
    std::vector<std::size_t> blocks;
    if (nilpotentSequence) {
      for (std::size_t j = 1; j < ranks.size(); ++j) {
        const std::size_t atLeastJ = ranks[j - 1] - ranks[j];
        const std::size_t atLeastNext =
            j + 1 < ranks.size() ? ranks[j] - ranks[j + 1] : 0;
        for (std::size_t count = atLeastNext; count < atLeastJ; ++count)
          blocks.push_back(j);
      }
      std::sort(blocks.begin(), blocks.end(), std::greater<>());
    } else {
      nameFailure(read.failedCertificates, "jordan-structure-unresolved");
    }
    read.geometricMultiplicity.push_back(ranks[0] - ranks[1]);
    read.jordanBlocks.push_back(std::move(blocks));

    // The residue of F_C^{-1}: minus the interface block of Pi M^{-1}.
    const Eigen::MatrixXcd product = riesz.projector * metric.inverse;
    Eigen::MatrixXcd residue(interfaceOrder, interfaceOrder);
    for (Eigen::Index row = 0; row < interfaceOrder; ++row)
      for (Eigen::Index column = 0; column < interfaceOrder; ++column)
        residue(row, column) =
            -product(partition.interface[static_cast<std::size_t>(row)],
                     partition.interface[static_cast<std::size_t>(column)]);
    std::vector<complexd> flat;
    flat.reserve(static_cast<std::size_t>(interfaceOrder * interfaceOrder));
    for (Eigen::Index row = 0; row < interfaceOrder; ++row)
      for (Eigen::Index column = 0; column < interfaceOrder; ++column)
        flat.push_back(residue(row, column));
    read.residue.push_back(std::move(flat));
    read.residueNorm.push_back(residue.norm());
    read.residueRank.push_back(
        rankAbove(residue, cfg.rankTolerance * largestSingularValue(residue)));

    if (cfg.freeThreshold.has_value())
      read.bindingShift.push_back(cluster.value - *cfg.freeThreshold);
  }

  // 5. The separation of each pole from the nearest other reported pole.
  for (std::size_t index = 0; index < read.poles.size(); ++index) {
    double nearest = std::numeric_limits<double>::infinity();
    for (std::size_t other = 0; other < read.poles.size(); ++other)
      if (other != index)
        nearest = std::min(nearest,
                           std::abs(read.poles[other] - read.poles[index]));
    read.separation.push_back(nearest);
  }
  return read;
}

BoundStatePoleRead BoundStatePole::clusterPoles(
    const AssembledPencil &assembled, int k,
    const std::vector<int> &clusterCells, const BoundStatePoleConfig &cfg) {
  const chainhodge::Pencil pencil = PencilLayer::pencil(assembled, k);
  return poles(pencil.A, pencil.B, clusterCells, cfg);
}

}  // namespace tessera::cobordism
