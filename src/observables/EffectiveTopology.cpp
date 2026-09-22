// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "observables/EffectiveTopology.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/Jacobi>

namespace tessera::observables {

using chainhodge::Complex;
using chainhodge::CovariantChainHodge;
using chainhodge::EffectiveBettiRead;
using chainhodge::Pencil;
using chainhodge::Preset;
using chainhodge::SparseMatrix;
using chainhodge::SparsePencil;
using chainhodge::SparsePencilOptions;
using chainhodge::SparsePencilSolver;

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kInfinity = std::numeric_limits<double>::infinity();
constexpr double kMachine = std::numeric_limits<double>::epsilon();
// The rank-tolerance multiplier of the chainhodge rank decisions.
constexpr double kKappa = 10.0;

void checkScale(double epsilon, double minimumGap, const char *who) {
  if (!(epsilon > 0.0)) throw std::invalid_argument(std::string(who) + ": the scale epsilon must be positive");
  if (!(minimumGap >= 1.0))
    throw std::invalid_argument(std::string(who) +
                                ": the minimum gap is a ratio of moduli outside over inside the window and "
                                "must be at least 1");
}

void checkDegree(const CovariantChainHodge &cov, int k, const char *who) {
  if (k < 0 || k > cov.dimension())
    throw std::invalid_argument(std::string(who) + ": degree " + std::to_string(k) + " is outside [0, " +
                                std::to_string(cov.dimension()) + "]");
}

double gapOf(int rank, double lastInside, double firstOutside) {
  if (std::isnan(firstOutside)) return kNaN;
  return (rank > 0 && lastInside > 0.0) ? firstOutside / lastInside : kInfinity;
}

// Fill `separated` and `certified` from `converged` and the gap.
void judge(EffectiveBettiNumber &out, double minimumGap) {
  out.minimumGap = minimumGap;
  out.separated = out.gap >= minimumGap;  // NaN compares false
  out.certified = out.converged && out.separated;
}

// One degree's dense pencil in complex Schur form: C = B^{-1} A = U T U^*,
// with the moduli of the diagonal sorted and the residual of the whole form.
struct DenseSchur {
  Pencil pencil;
  Eigen::MatrixXcd T;
  Eigen::MatrixXcd U;
  std::vector<double> moduli;
  double residual{0.0};
};

DenseSchur denseSchur(const CovariantChainHodge &cov, int k) {
  DenseSchur out;
  out.pencil = cov.pencil(k);
  const Eigen::Index n = out.pencil.A.rows();
  if (n == 0) return out;
  // B^{-1} A by the instance's sparse solve: M_k^U under L2, whose pencil is
  // written on images, and G_k^U under GRASSMANN_ALL, written on chains.
  const Eigen::MatrixXcd C = cov.preset() == Preset::L2 ? cov.applyG(k, out.pencil.A)
                                                        : cov.applyMinv(k, out.pencil.A);
  Eigen::ComplexSchur<Eigen::MatrixXcd> schur(C, true);
  if (schur.info() != Eigen::Success)
    throw std::runtime_error("EffectiveTopology: the Schur decomposition of degree " + std::to_string(k) +
                             " did not converge");
  out.T = schur.matrixT();
  out.U = schur.matrixU();
  out.moduli.reserve(static_cast<std::size_t>(n));
  for (Eigen::Index i = 0; i < n; ++i) out.moduli.push_back(std::abs(out.T(i, i)));
  std::sort(out.moduli.begin(), out.moduli.end());
  const double normA = out.pencil.A.norm();
  const double defect = (out.pencil.A * out.U - out.pencil.B * (out.U * out.T)).norm();
  out.residual = normA > 0.0 ? defect / normA : defect;
  return out;
}

EffectiveBettiNumber countFromModuli(const DenseSchur &schur, int k, double epsilon, double tolerance,
                                     double minimumGap) {
  EffectiveBettiNumber out;
  out.degree = k;
  out.method = EffectiveBettiNumber::Method::DenseSpectrum;
  const std::vector<double> &moduli = schur.moduli;
  out.rank = static_cast<int>(std::upper_bound(moduli.begin(), moduli.end(), epsilon) - moduli.begin());
  if (out.rank > 0) out.lastInside = moduli[static_cast<std::size_t>(out.rank) - 1];
  if (out.rank < static_cast<int>(moduli.size())) out.firstOutside = moduli[static_cast<std::size_t>(out.rank)];
  // A complex whose degree has no cells has an empty band and nothing outside it.
  out.gap = moduli.empty() ? kInfinity : gapOf(out.rank, out.lastInside, out.firstOutside);
  out.converged = schur.residual <= tolerance;
  judge(out, minimumGap);
  return out;
}

// The degree-zero read from the sparse pencil, with the band's images.
struct SparseBand {
  EffectiveBettiNumber count;
  Eigen::MatrixXcd images;
  double residual{kNaN};
};

SparseBand fromSparsePencil(const CovariantChainHodge &cov, double epsilon, double tolerance, double minimumGap) {
  SparseBand out;
  out.count.degree = 0;
  try {
    const SparsePencil pencil = cov.sparsePencil(0);
    SparsePencilOptions options;
    options.tolerance = tolerance;
    // The shift at -epsilon: its Cholesky factorization certifies that no
    // eigenvalue lies below the window, so the count above it is |lambda| <= epsilon.
    const EffectiveBettiRead read = SparsePencilSolver::effectiveBetti(pencil.A, pencil.M, epsilon, -epsilon, options);
    out.count.method = EffectiveBettiNumber::Method::SparsePencil;
    out.count.rank = read.rank;
    out.count.lastInside = std::abs(read.lastInside);
    out.count.firstOutside = read.firstOutside;
    out.count.gap = gapOf(out.count.rank, out.count.lastInside, out.count.firstOutside);
    out.count.converged = read.certified;
    judge(out.count, minimumGap);
    out.images = read.band.vectors.leftCols(read.rank);
    out.residual = 0.0;
    for (int i = 0; i < read.rank; ++i)
      out.residual = std::max(out.residual, read.band.residuals[static_cast<std::size_t>(i)]);
  } catch (const std::logic_error &refusal) {
    // std::invalid_argument (a pencil that is not Hermitian positive definite)
    // and the Grassmann preset's refusal both derive from std::logic_error.
    out.count.method = EffectiveBettiNumber::Method::Unmeasured;
    out.count.reason = refusal.what();
  }
  return out;
}

EffectiveBettiNumber unmeasuredAboveCrossover(int k) {
  EffectiveBettiNumber out;
  out.degree = k;
  out.reason = "degree " + std::to_string(k) +
               " is at or above the dense crossover and its pencil contains an inverse metric, so it has no "
               "sparse form";
  return out;
}

// Exchange the adjacent diagonal entries k and k+1 of the complex Schur form,
// keeping C = U T U^*: the Givens rotation whose first column is the
// eigenvector of the 2x2 block for its second eigenvalue (LAPACK ztrexc).
void swapAdjacent(Eigen::MatrixXcd &T, Eigen::MatrixXcd &U, Eigen::Index k) {
  const Complex first = T(k, k);
  const Complex second = T(k + 1, k + 1);
  Eigen::JacobiRotation<Complex> G;
  G.makeGivens(T(k, k + 1), second - first);
  T.applyOnTheLeft(k, k + 1, G.adjoint());
  T.applyOnTheRight(k, k + 1, G);
  U.applyOnTheRight(k, k + 1, G);
  T(k + 1, k) = Complex(0.0, 0.0);
  T(k, k) = second;
  T(k + 1, k + 1) = first;
}

// The invariant subspace of the eigenvalues with |lambda| <= epsilon: the
// leading Schur vectors after the selected diagonal entries are moved to the
// top, an orthonormal basis in the pencil's variable, with the residual of
// the pencil on it.
struct BandFrame {
  Eigen::MatrixXcd basis;
  double residual{kNaN};
};

BandFrame bandFrame(const DenseSchur &schur, double epsilon) {
  Eigen::MatrixXcd T = schur.T;
  Eigen::MatrixXcd U = schur.U;
  const Eigen::Index n = T.rows();
  Eigen::Index next = 0;
  for (Eigen::Index j = 0; j < n; ++j) {
    if (!(std::abs(T(j, j)) <= epsilon)) continue;
    for (Eigen::Index k = j - 1; k >= next; --k) swapAdjacent(T, U, k);
    ++next;
  }
  BandFrame out;
  out.basis = U.leftCols(next);
  if (next == 0) {
    out.residual = 0.0;
    return out;
  }
  const Eigen::MatrixXcd T11 = T.topLeftCorner(next, next);
  const double normA = schur.pencil.A.norm();
  const double defect = (schur.pencil.A * out.basis - schur.pencil.B * (out.basis * T11)).norm();
  out.residual = normA > 0.0 ? defect / normA : defect;
  return out;
}

// An orthonormal basis of the columns of a full-column-rank matrix.
Eigen::MatrixXcd orthonormal(const Eigen::MatrixXcd &X) {
  if (X.cols() == 0) return Eigen::MatrixXcd(X.rows(), 0);
  Eigen::HouseholderQR<Eigen::MatrixXcd> qr(X);
  return qr.householderQ() * Eigen::MatrixXcd::Identity(X.rows(), X.cols());
}

// The chains of a band given in the pencil's variable.
Eigen::MatrixXcd chainsOf(const CovariantChainHodge &cov, int k, const Eigen::MatrixXcd &pencilVectors) {
  return cov.preset() == Preset::L2 ? cov.applyMinv(k, pencilVectors) : pencilVectors;
}

// The geometric images of a band given in the pencil's variable.
Eigen::MatrixXcd imagesOf(const CovariantChainHodge &cov, int k, const Eigen::MatrixXcd &pencilVectors) {
  return cov.preset() == Preset::L2 ? pencilVectors : cov.applyG(k, pencilVectors);
}

// sqrt(||X||_1 ||X||_inf), a bound on the 2-norm of a sparse matrix.
double normBound(const SparseMatrix &X) {
  if (X.nonZeros() == 0) return 0.0;
  Eigen::VectorXd columns = Eigen::VectorXd::Zero(X.cols());
  Eigen::VectorXd rows = Eigen::VectorXd::Zero(X.rows());
  for (int j = 0; j < X.outerSize(); ++j)
    for (SparseMatrix::InnerIterator it(X, j); it; ++it) {
      columns(it.col()) += std::abs(it.value());
      rows(it.row()) += std::abs(it.value());
    }
  return std::sqrt(columns.maxCoeff() * rows.maxCoeff());
}

double largestSingularValue(const Eigen::MatrixXcd &X) {
  if (X.size() == 0) return 0.0;
  Eigen::JacobiSVD<Eigen::MatrixXcd> svd(X);
  return svd.singularValues()(0);
}

// Split the band with orthonormal chain basis Q at degree k >= 1 into its
// coexact part (the null space of the twisted boundary on the band) and its
// exact part (the projection along ker d_k onto im d_k^#, which for a band is
// d_k^# of the degree k-1 band outside the kernel).
void splitBand(const CovariantChainHodge &cov, int k, const Eigen::MatrixXcd &Q, EffectiveHodgeSplit &out) {
  const auto r = Q.cols();
  const SparseMatrix &boundary = cov.twistedBoundary(k);          // d_k^U, n_{k-1} x n_k
  const SparseMatrix &boundaryDual = cov.twistedBoundaryDual(k);  // d_k^{U^{-1}}
  out.rankTolerance = kKappa * static_cast<double>(std::max<Eigen::Index>(boundary.rows(), r)) * kMachine *
                      normBound(boundary);
  if (r == 0) {
    out.exact = out.coexact = 0;
    out.exactFrame = out.coexactFrame = Eigen::MatrixXcd(Q.rows(), 0);
    out.splitGap = kInfinity;
    out.closure = 0.0;
    return;
  }
  const Eigen::MatrixXcd D = boundary * Q;
  Eigen::JacobiSVD<Eigen::MatrixXcd> svd(D, Eigen::ComputeFullV | Eigen::ComputeThinU);
  const Eigen::VectorXd sigma = svd.singularValues();
  out.boundarySingularValues.assign(sigma.data(), sigma.data() + sigma.size());
  Eigen::Index e = 0;
  while (e < sigma.size() && sigma(e) > out.rankTolerance) ++e;
  out.exact = static_cast<int>(e);
  out.coexact = static_cast<int>(r - e);
  // With fewer rows than band directions the missing singular values are zero.
  out.splitGap = (e == 0 || e == r || e == sigma.size()) ? kInfinity : sigma(e - 1) / sigma(e);
  const Eigen::MatrixXcd V = svd.matrixV();
  out.coexactFrame = Q * V.rightCols(r - e);
  if (e == 0) {
    out.exactFrame = Eigen::MatrixXcd(Q.rows(), 0);
    out.closure = 0.0;
    return;
  }
  // x_e = d^# y with d d^# y = d x: the projection of x along ker d onto
  // im d^#, d^# y = M_k^U (d_k^{U^{-1}})^T (M_{k-1}^U)^{-1} y.
  const auto sharp = [&](const Eigen::MatrixXcd &y) {
    return cov.applyMinv(k, Eigen::MatrixXcd(SparseMatrix(boundaryDual.transpose()) * cov.applyG(k - 1, y)));
  };
  const Eigen::MatrixXcd upper =
      boundary * sharp(Eigen::MatrixXcd::Identity(boundary.rows(), boundary.rows()));  // d d^# on C_{k-1}
  const Eigen::MatrixXcd y = upper.completeOrthogonalDecomposition().solve(Eigen::MatrixXcd(D * V.leftCols(e)));
  out.exactFrame = orthonormal(sharp(y));
  out.closure = largestSingularValue(out.exactFrame - Q * (Q.adjoint() * out.exactFrame));
}

}  // namespace

std::vector<EffectiveTopology> EffectiveTopology::readScales(const CovariantChainHodge &cov,
                                                             const std::vector<double> &epsilons,
                                                             double tolerance, double minimumGap) {
  for (double epsilon : epsilons) checkScale(epsilon, minimumGap, "EffectiveTopology");
  const int d = cov.dimension();
  const int crossover = cov.base().crossoverDimension();
  // Each dense degree's Schur form is computed once and read at every scale.
  std::vector<DenseSchur> dense(static_cast<std::size_t>(d) + 1);
  for (int k = 0; k <= d; ++k)
    if (cov.base().size(k) < crossover) dense[static_cast<std::size_t>(k)] = denseSchur(cov, k);
  std::vector<EffectiveTopology> out;
  out.reserve(epsilons.size());
  for (double epsilon : epsilons) {
    EffectiveTopology topology;
    topology.epsilon_ = epsilon;
    for (int k = 0; k <= d; ++k) {
      if (cov.base().size(k) < crossover)
        topology.degrees_.push_back(
            countFromModuli(dense[static_cast<std::size_t>(k)], k, epsilon, tolerance, minimumGap));
      else if (k == 0)
        topology.degrees_.push_back(fromSparsePencil(cov, epsilon, tolerance, minimumGap).count);
      else
        topology.degrees_.push_back(unmeasuredAboveCrossover(k));
    }
    out.push_back(std::move(topology));
  }
  return out;
}

EffectiveTopology EffectiveTopology::read(const CovariantChainHodge &cov, double epsilon, double tolerance,
                                          double minimumGap) {
  return readScales(cov, {epsilon}, tolerance, minimumGap).front();
}

EffectiveHodgeSplit EffectiveTopology::split(const CovariantChainHodge &cov, int degree, double epsilon,
                                             double tolerance, double minimumGap) {
  checkScale(epsilon, minimumGap, "EffectiveTopology::split");
  checkDegree(cov, degree, "EffectiveTopology::split");
  EffectiveHodgeSplit out;
  Eigen::MatrixXcd Q;
  if (cov.base().size(degree) >= cov.base().crossoverDimension()) {
    if (degree != 0) {
      out.band = unmeasuredAboveCrossover(degree);
      out.reason = out.band.reason;
      return out;
    }
    SparseBand sparse = fromSparsePencil(cov, epsilon, tolerance, minimumGap);
    out.band = sparse.count;
    if (out.band.method == EffectiveBettiNumber::Method::Unmeasured) {
      out.reason = out.band.reason;
      return out;
    }
    out.frameResidual = sparse.residual;
    Q = orthonormal(cov.applyMinv(0, sparse.images));
  } else {
    const DenseSchur schur = denseSchur(cov, degree);
    out.band = countFromModuli(schur, degree, epsilon, tolerance, minimumGap);
    const BandFrame frame = bandFrame(schur, epsilon);
    out.frameResidual = frame.residual;
    Q = orthonormal(chainsOf(cov, degree, frame.basis));
  }
  if (degree == 0) {
    // Degree zero has no boundary: its whole band is coexact.
    out.exact = 0;
    out.coexact = static_cast<int>(Q.cols());
    out.exactFrame = Eigen::MatrixXcd(Q.rows(), 0);
    out.coexactFrame = Q;
    out.splitGap = kInfinity;
    out.closure = 0.0;
  } else {
    splitBand(cov, degree, Q, out);
  }
  out.certified = out.band.certified && out.frameResidual <= tolerance && out.splitGap >= minimumGap &&
                  out.closure <= tolerance;
  return out;
}

EffectiveHodgeSplit EffectiveTopology::voids(const CovariantChainHodge &cov, double epsilon, double tolerance,
                                             double minimumGap) {
  if (cov.dimension() != 3)
    throw std::invalid_argument("EffectiveTopology::voids: the effective voids are the near-cycles of L_2 on a "
                                "complex of dimension three, and this complex has dimension " +
                                std::to_string(cov.dimension()));
  return split(cov, 2, epsilon, tolerance, minimumGap);
}

EffectiveComponentPartition EffectiveTopology::components(const CovariantChainHodge &cov, double epsilon,
                                                          double tolerance, double minimumGap) {
  checkScale(epsilon, minimumGap, "EffectiveTopology::components");
  EffectiveComponentPartition out;
  Eigen::MatrixXcd images;
  if (cov.base().size(0) >= cov.base().crossoverDimension()) {
    SparseBand sparse = fromSparsePencil(cov, epsilon, tolerance, minimumGap);
    out.band = sparse.count;
    if (out.band.method == EffectiveBettiNumber::Method::Unmeasured) {
      out.reason = out.band.reason;
      return out;
    }
    images = sparse.images;
  } else {
    const DenseSchur schur = denseSchur(cov, 0);
    out.band = countFromModuli(schur, 0, epsilon, tolerance, minimumGap);
    images = imagesOf(cov, 0, bandFrame(schur, epsilon).basis);
  }
  const auto r = images.cols();
  if (r == 0) {
    out.certified = out.band.certified;
    return out;
  }
  const Eigen::MatrixXcd Q = orthonormal(images);
  const auto n = Q.rows();
  // The first r pivots of the column-pivoted QR of Q^T: the vertices whose rows
  // of the band are most independent.
  Eigen::ColPivHouseholderQR<Eigen::MatrixXcd> pivoted(Q.transpose());
  const auto &permutation = pivoted.colsPermutation().indices();
  std::vector<Eigen::Index> pivots(static_cast<std::size_t>(r));
  Eigen::MatrixXcd P(r, r);
  for (Eigen::Index j = 0; j < r; ++j) {
    pivots[static_cast<std::size_t>(j)] = permutation(j);
    P.row(j) = Q.row(permutation(j));
  }
  Eigen::JacobiSVD<Eigen::MatrixXcd> pivotSvd(P);
  const Eigen::VectorXd pivotSigma = pivotSvd.singularValues();
  out.pivotConditioning = pivotSigma(r - 1) > 0.0 ? pivotSigma(0) / pivotSigma(r - 1) : kInfinity;
  if (!(out.pivotConditioning < 1.0 / (static_cast<double>(n) * kMachine))) {
    out.reason = "the band's rows at the pivots are singular, so no committor basis exists";
    return out;
  }
  // C = Q P^{-1}: one at its own pivot, zero at the others.
  const Eigen::MatrixXcd committors = P.transpose().fullPivLu().solve(Eigen::MatrixXcd(Q.transpose())).transpose();
  const auto vertices = cov.base().complex().kSimplexVertices(0);
  out.components.resize(static_cast<std::size_t>(r));
  for (Eigen::Index j = 0; j < r; ++j) {
    auto &component = out.components[static_cast<std::size_t>(j)];
    component.pivot = vertices[static_cast<std::size_t>(pivots[static_cast<std::size_t>(j)])].front();
    component.committor = committors.col(j);
  }
  out.partitionDefect = 0.0;
  for (Eigen::Index v = 0; v < n; ++v) {
    Eigen::Index owner = 0;
    committors.row(v).cwiseAbs().maxCoeff(&owner);
    out.components[static_cast<std::size_t>(owner)].support.push_back(vertices[static_cast<std::size_t>(v)].front());
    out.partitionDefect = std::max(out.partitionDefect, std::abs(committors.row(v).sum() - Complex(1.0, 0.0)));
  }
  for (auto &component : out.components) std::sort(component.support.begin(), component.support.end());
  out.certified = out.band.certified;
  return out;
}

std::vector<int> EffectiveTopology::betti() const {
  std::vector<int> out;
  out.reserve(degrees_.size());
  for (const auto &degree : degrees_) out.push_back(degree.rank);
  return out;
}

bool EffectiveTopology::certified() const noexcept {
  return std::all_of(degrees_.begin(), degrees_.end(), [](const EffectiveBettiNumber &degree) {
    return degree.method != EffectiveBettiNumber::Method::Unmeasured && degree.certified;
  });
}

EffectivePersistence EffectivePersistence::sweep(const CovariantChainHodge &cov, const std::vector<double> &epsilons,
                                                 double tolerance, double minimumGap) {
  if (epsilons.empty()) throw std::invalid_argument("EffectivePersistence: the sweep has no scales");
  return EffectivePersistence(EffectiveTopology::readScales(cov, epsilons, tolerance, minimumGap));
}

EffectivePersistence::EffectivePersistence(std::vector<EffectiveTopology> reads) : reads_(std::move(reads)) {
  if (reads_.empty()) throw std::invalid_argument("EffectivePersistence: the sequence has no reads");
  for (const auto &read : reads_)
    if (read.dimension() != reads_.front().dimension())
      throw std::invalid_argument("EffectivePersistence: the reads have different dimensions");
}

std::vector<EffectivePlateau> EffectivePersistence::plateaus(int k) const {
  if (k < 0 || k > dimension())
    throw std::invalid_argument("EffectivePersistence: degree " + std::to_string(k) + " is outside [0, " +
                                std::to_string(dimension()) + "]");
  std::vector<EffectivePlateau> out;
  bool open = false;
  for (std::size_t i = 0; i < reads_.size(); ++i) {
    const EffectiveBettiNumber &degree = reads_[i].degrees()[static_cast<std::size_t>(k)];
    const bool certified = degree.method != EffectiveBettiNumber::Method::Unmeasured && degree.certified;
    if (!certified) {
      open = false;
      continue;
    }
    if (open && out.back().rank == degree.rank) {
      out.back().last = i;
      out.back().lastEpsilon = reads_[i].epsilon();
      out.back().gap = std::min(out.back().gap, degree.gap);
      continue;
    }
    EffectivePlateau plateau;
    plateau.degree = k;
    plateau.rank = degree.rank;
    plateau.first = plateau.last = i;
    plateau.firstEpsilon = plateau.lastEpsilon = reads_[i].epsilon();
    plateau.gap = degree.gap;
    out.push_back(plateau);
    open = true;
  }
  return out;
}

int EffectivePersistence::persistentRank(int k) const {
  const std::vector<EffectivePlateau> runs = plateaus(k);
  return (runs.size() == 1 && runs.front().length() == reads_.size()) ? runs.front().rank : -1;
}

std::vector<int> EffectivePersistence::betti() const {
  std::vector<int> out;
  for (int k = 0; k <= dimension(); ++k) out.push_back(persistentRank(k));
  return out;
}

bool EffectivePersistence::certified() const {
  for (int k = 0; k <= dimension(); ++k)
    if (persistentRank(k) < 0) return false;
  return true;
}

namespace {

EffectiveSignatureCertificate prepare(const EffectiveSignature &signature, int dimension) {
  EffectiveSignatureCertificate out;
  out.signature = signature.name();
  out.expected = signature.betti();
  if (static_cast<int>(out.expected.size()) != dimension + 1)
    throw std::invalid_argument("EffectiveSignature: the " + signature.name() + " has dimension " +
                                std::to_string(static_cast<int>(out.expected.size()) - 1) +
                                " and the read has dimension " + std::to_string(dimension));
  out.matches = true;
  out.certified = true;
  out.gap = kInfinity;
  return out;
}

}  // namespace

EffectiveSignatureCertificate EffectiveSignature::certify(const EffectiveTopology &topology) const {
  EffectiveSignatureCertificate out = prepare(*this, topology.dimension());
  out.epsilon = topology.epsilon();
  out.scales = {topology.epsilon()};
  out.measured = topology.betti();
  for (std::size_t k = 0; k < out.expected.size(); ++k) {
    if (out.expected[k] < 0) continue;
    const EffectiveBettiNumber &degree = topology.degrees()[k];
    out.matches = out.matches && degree.rank == out.expected[k];
    out.certified = out.certified && degree.certified;
    // NaN (nothing outside the window) compares false and leaves the minimum alone.
    if (degree.gap < out.gap) out.gap = degree.gap;
  }
  return out;
}

EffectiveSignatureCertificate EffectiveSignature::certify(const CovariantChainHodge &cov, double epsilon,
                                                          double tolerance, double minimumGap) const {
  return certify(EffectiveTopology::read(cov, epsilon, tolerance, minimumGap));
}

EffectiveSignatureCertificate EffectiveSignature::certify(const EffectivePersistence &persistence) const {
  EffectiveSignatureCertificate out = prepare(*this, persistence.dimension());
  for (const auto &read : persistence.reads()) out.scales.push_back(read.epsilon());
  out.epsilon = out.scales.front();
  out.measured = persistence.betti();
  for (std::size_t k = 0; k < out.expected.size(); ++k) {
    if (out.expected[k] < 0) continue;
    out.matches = out.matches && out.measured[k] == out.expected[k];
    out.certified = out.certified && out.measured[k] >= 0;
    for (const auto &read : persistence.reads())
      if (read.degrees()[k].gap < out.gap) out.gap = read.degrees()[k].gap;
  }
  return out;
}

EffectiveTorus::EffectiveTorus(int dimension) : dimension_(dimension) {
  if (dimension < 1) throw std::invalid_argument("EffectiveTorus: the dimension must be at least 1");
}
std::string EffectiveTorus::name() const { return "effective " + std::to_string(dimension_) + "-torus"; }
std::vector<int> EffectiveTorus::betti() const {
  std::vector<int> row(static_cast<std::size_t>(dimension_) + 1, 1);
  for (int k = 1; k < dimension_; ++k)
    row[static_cast<std::size_t>(k)] = row[static_cast<std::size_t>(k) - 1] * (dimension_ - k + 1) / k;
  return row;
}

EffectiveSphere::EffectiveSphere(int dimension) : dimension_(dimension) {
  if (dimension < 1) throw std::invalid_argument("EffectiveSphere: the dimension must be at least 1");
}
std::string EffectiveSphere::name() const { return "effective " + std::to_string(dimension_) + "-sphere"; }
std::vector<int> EffectiveSphere::betti() const {
  std::vector<int> out(static_cast<std::size_t>(dimension_) + 1, 0);
  out.front() = 1;
  out.back() = 1;
  return out;
}

EffectiveComponents::EffectiveComponents(int count, int dimension) : count_(count), dimension_(dimension) {
  if (count < 0 || dimension < 0)
    throw std::invalid_argument("EffectiveComponents: the count and the dimension must be non-negative");
}
std::string EffectiveComponents::name() const {
  return std::to_string(count_) + " effective component" + (count_ == 1 ? "" : "s");
}
std::vector<int> EffectiveComponents::betti() const {
  std::vector<int> out(static_cast<std::size_t>(dimension_) + 1, -1);
  out.front() = count_;
  return out;
}

}  // namespace tessera::observables
