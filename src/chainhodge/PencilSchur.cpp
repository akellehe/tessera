// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "chainhodge/PencilSchur.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <set>
#include <stdexcept>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/SparseLU>

namespace tessera::chainhodge {

namespace {

constexpr double kTiny = 1e-300;

/// The rank of \p X at the relative threshold \p relative (a singular value at
/// or below \p relative times the largest one is zero), with the left and right
/// singular vectors, the threshold that decided it, and the singular gap
/// \f$ \varsigma_r/\varsigma_{r+1} \f$.
struct RankRead {
  int rank{0};
  double threshold{0.0};
  double gap{std::numeric_limits<double>::infinity()};
  Eigen::MatrixXcd U{};
  Eigen::MatrixXcd V{};
  Eigen::VectorXd values{};
};

RankRead rankRead(const Eigen::MatrixXcd &X, double relative) {
  RankRead out;
  if (X.rows() == 0 || X.cols() == 0) return out;
  Eigen::JacobiSVD<Eigen::MatrixXcd> svd(X, Eigen::ComputeFullU | Eigen::ComputeFullV);
  out.values = svd.singularValues();
  out.U = svd.matrixU();
  out.V = svd.matrixV();
  out.threshold = relative * out.values(0);
  for (int i = 0; i < out.values.size(); ++i)
    if (out.values(i) > out.threshold) ++out.rank;
  if (out.rank >= 1 && out.rank < out.values.size() && out.values(out.rank) > 0.0)
    out.gap = out.values(out.rank - 1) / out.values(out.rank);
  return out;
}

/// The Moore–Penrose pseudoinverse implied by a `RankRead`:
/// \f$ X^{\#} = V_r\Sigma_r^{-1}U_r^H \f$ over the kept singular triplets. This
/// is the declared supported generalized inverse of the whitepaper's block
/// elimination; it is reflexive (\f$ X^{\#}XX^{\#} = X^{\#} \f$) and satisfies
/// \f$ XX^{\#}X = X \f$, which is all the elimination uses.
Eigen::MatrixXcd pseudoInverse(const RankRead &read) {
  const int r = read.rank;
  Eigen::MatrixXcd out = Eigen::MatrixXcd::Zero(read.V.rows(), read.U.rows());
  for (int i = 0; i < r; ++i)
    out += (1.0 / read.values(i)) * read.V.col(i) * read.U.col(i).adjoint();
  return out;
}

/// Ascending order of a complex spectrum by \f$ (\mathrm{Re},\mathrm{Im}) \f$,
/// the order every spectrum in this subsystem is reported in.
std::vector<int> spectralOrder(const Eigen::VectorXcd &values) {
  std::vector<int> order(static_cast<std::size_t>(values.size()));
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](int a, int b) {
    if (values(a).real() != values(b).real()) return values(a).real() < values(b).real();
    return values(a).imag() < values(b).imag();
  });
  return order;
}

/// The Schur determinant identity on the log scale, filled in once the three
/// log determinants are known: the real part of
/// \f$ \log\det P - \log\det P_{II} - \log\det F_B \f$ and the distance of
/// its imaginary part to the nearest multiple of \f$ 2\pi \f$.
void finishLogResiduals(FeshbachResult &out) {
  constexpr double kTwoPi = 6.28318530717958647692;
  out.responseLogDeterminant = PencilSchur::logDeterminant(out.response);
  const Complex defect =
      out.pencilLogDeterminant - out.interiorLogDeterminant - out.responseLogDeterminant;
  out.logModulusResidual = std::abs(defect.real());
  out.logPhaseResidual = std::abs(std::remainder(defect.imag(), kTwoPi));
}

}  // namespace

Complex PencilSchur::logDeterminant(const Eigen::MatrixXcd &A) {
  if (A.rows() != A.cols())
    throw std::invalid_argument("PencilSchur::logDeterminant: the matrix must be square");
  if (A.rows() == 0) return Complex(0.0, 0.0);
  constexpr double kPi = 3.14159265358979323846;
  const Eigen::PartialPivLU<Eigen::MatrixXcd> lu(A);
  const Eigen::MatrixXcd &LU = lu.matrixLU();
  double modulus = 0.0, phase = 0.0;
  for (Eigen::Index i = 0; i < LU.rows(); ++i) {
    const Complex z = LU(i, i);
    if (z == Complex(0.0, 0.0)) return Complex(-std::numeric_limits<double>::infinity(), 0.0);
    modulus += std::log(std::abs(z));
    phase += std::arg(z);
  }
  if (lu.permutationP().determinant() < 0) phase += kPi;
  phase = std::remainder(phase, 2.0 * kPi);
  if (phase <= -kPi) phase += 2.0 * kPi;
  return Complex(modulus, phase);
}

FeshbachResult PencilSchur::feshbach(const Eigen::MatrixXcd &A, const Eigen::MatrixXcd &M,
                                     Complex lambda, const std::vector<int> &interface,
                                     double rankTolerance) {
  const int n = static_cast<int>(A.rows());
  if (A.cols() != n || M.rows() != n || M.cols() != n)
    throw std::invalid_argument("PencilSchur::feshbach: A and M must be square of the same size");
  FeshbachResult out;
  out.lambda = lambda;
  std::set<int> kept;
  for (const int b : interface) {
    if (b < 0 || b >= n) throw std::invalid_argument("PencilSchur::feshbach: interface index out of range");
    kept.insert(b);
  }
  out.interface.assign(kept.begin(), kept.end());
  for (int i = 0; i < n; ++i)
    if (kept.find(i) == kept.end()) out.interior.push_back(i);
  const Eigen::MatrixXcd P = A - lambda * M;
  const int nb = static_cast<int>(out.interface.size());
  const int ni = static_cast<int>(out.interior.size());
  Eigen::MatrixXcd PBB(nb, nb), PBI(nb, ni), PIB(ni, nb), PII(ni, ni);
  for (int i = 0; i < nb; ++i)
    for (int j = 0; j < nb; ++j) PBB(i, j) = P(out.interface[static_cast<std::size_t>(i)], out.interface[static_cast<std::size_t>(j)]);
  for (int i = 0; i < nb; ++i)
    for (int j = 0; j < ni; ++j) {
      PBI(i, j) = P(out.interface[static_cast<std::size_t>(i)], out.interior[static_cast<std::size_t>(j)]);
      PIB(j, i) = P(out.interior[static_cast<std::size_t>(j)], out.interface[static_cast<std::size_t>(i)]);
    }
  for (int i = 0; i < ni; ++i)
    for (int j = 0; j < ni; ++j) PII(i, j) = P(out.interior[static_cast<std::size_t>(i)], out.interior[static_cast<std::size_t>(j)]);
  out.pencilDeterminant = P.fullPivLu().determinant();
  out.pencilLogDeterminant = logDeterminant(P);
  out.interiorRank = ni;
  // The embedding of an interior-coordinate block into the full coordinates,
  // used for the constraint modes and for the retained resonant modes alike.
  const auto embedInterior = [&](const Eigen::MatrixXcd &block) {
    Eigen::MatrixXcd full = Eigen::MatrixXcd::Zero(n, block.cols());
    for (int j = 0; j < static_cast<int>(block.cols()); ++j)
      for (int i = 0; i < ni; ++i)
        full(out.interior[static_cast<std::size_t>(i)], j) = block(i, j);
    return full;
  };
  if (ni == 0) {
    out.response = PBB;
    out.interiorDeterminant = Complex(1.0, 0.0);
    out.interiorLogDeterminant = Complex(0.0, 0.0);
    out.responseDeterminant = PBB.fullPivLu().determinant();
    out.constraintModes = Eigen::MatrixXcd::Identity(n, nb);
    out.solveResidual = 0.0;
    out.rangeProjector = Eigen::MatrixXcd::Zero(0, 0);
    out.nullProjector = Eigen::MatrixXcd::Zero(0, 0);
  } else {
    Eigen::FullPivLU<Eigen::MatrixXcd> lu(PII);
    lu.setThreshold(rankTolerance);
    out.interiorDeterminant = lu.determinant();
    out.interiorLogDeterminant = logDeterminant(PII);
    if (lu.isInvertible()) {
      const Eigen::MatrixXcd X = lu.solve(PIB);  // P_II^{-1} P_IB
      out.solveResidual = (PII * X - PIB).norm() / std::max(PIB.norm(), kTiny);
      out.response = PBB - PBI * X;
      out.responseDeterminant = out.response.fullPivLu().determinant();
      out.constraintModes = embedInterior(Eigen::MatrixXcd(-X));
      for (int j = 0; j < nb; ++j)
        out.constraintModes(out.interface[static_cast<std::size_t>(j)], j) = Complex(1.0, 0.0);
      // An invertible interior block has full range and no kernel: the
      // projectors are the identity and zero, and no singular value decomposition
      // is taken to say so.
      out.rangeProjector = Eigen::MatrixXcd::Identity(ni, ni);
      out.nullProjector = Eigen::MatrixXcd::Zero(ni, ni);
      out.interiorNullSpace = Eigen::MatrixXcd(ni, 0);
      out.interiorLeftNullSpace = Eigen::MatrixXcd(ni, 0);
      out.resonantModes = Eigen::MatrixXcd(n, 0);
      const Complex product = out.interiorDeterminant * out.responseDeterminant;
      out.determinantResidual = std::abs(out.pencilDeterminant - product) /
                                std::max(std::abs(out.pencilDeterminant), kTiny);
      finishLogResiduals(out);
      return out;
    }
    // --- interior resonance: the generalized inverse and its projectors ---
    out.interiorSingular = true;
    const RankRead read = rankRead(PII, rankTolerance);
    const int r = read.rank;
    const int q = ni - r;
    out.interiorRank = r;
    out.interiorRankThreshold = read.threshold;
    out.interiorSingularGap = read.gap;
    const Eigen::MatrixXcd pinv = pseudoInverse(read);
    // ker P_II is spanned by the trailing right singular vectors; the left null
    // space in the transpose pairing is the conjugate of the trailing left
    // singular vectors, because y^T P_II = 0 iff P_II^H \bar y = 0.
    out.interiorNullSpace = read.V.rightCols(q);
    out.interiorLeftNullSpace = read.U.rightCols(q).conjugate();
    out.rangeProjector = PII * pinv;
    out.nullProjector = Eigen::MatrixXcd::Identity(ni, ni) - pinv * PII;
    const double scaleIB = std::max(PIB.norm(), kTiny);
    const double scaleBI = std::max(PBI.norm(), kTiny);
    const Eigen::MatrixXcd leftTest = out.interiorLeftNullSpace.transpose() * PIB;  // N_L^T P_IB
    out.compatibilityResidual = leftTest.norm() / scaleIB;
    out.compatible = out.compatibilityResidual <= rankTolerance;
    const Eigen::MatrixXcd coupling = PBI * out.interiorNullSpace;  // P_BI N
    out.independenceResidual = coupling.norm() / scaleBI;
    out.responseIndependent = out.independenceResidual <= rankTolerance;
    const Eigen::MatrixXcd X = pinv * PIB;  // P_II^# P_IB
    // P_II X = Pi_R P_IB, so this residual is the compatibility residual: the
    // interior equation is solvable exactly on the range of P_II.
    out.solveResidual = (PII * X - PIB).norm() / scaleIB;
    out.response = PBB - PBI * X;
    out.responseDeterminant = out.response.fullPivLu().determinant();
    out.constraintModes = embedInterior(Eigen::MatrixXcd(-X));
    for (int j = 0; j < nb; ++j)
      out.constraintModes(out.interface[static_cast<std::size_t>(j)], j) = Complex(1.0, 0.0);
    out.resonantModes = embedInterior(out.interiorNullSpace);
    // The resonant reduction over (x_B, c): the eliminated interface equation
    // above the compatibility constraint.
    out.resonantResponse = Eigen::MatrixXcd::Zero(nb + q, nb + q);
    out.resonantResponse.topLeftCorner(nb, nb) = out.response;
    out.resonantResponse.topRightCorner(nb, q) = coupling;
    out.resonantResponse.bottomLeftCorner(q, nb) = leftTest;
    // The determinant factorization det P = det P_II det F_B has no content
    // here: det P_II is zero at the resonance.
    out.determinantResidual = std::numeric_limits<double>::quiet_NaN();
    // The certificate of the reduction as a whole: the pencil applied to the
    // retained fibers is read off the reduction alone,
    //   P W = E_B Fhat[top] + E_I conj(N_L) Fhat[bottom],
    // which holds column by column and not only on the null space.
    {
      Eigen::MatrixXcd W(n, nb + q);
      W.leftCols(nb) = out.constraintModes;
      W.rightCols(q) = out.resonantModes;
      const Eigen::MatrixXcd applied = P * W;
      const Eigen::MatrixXcd interiorPart =
          out.interiorLeftNullSpace.conjugate() * out.resonantResponse.bottomRows(q);
      Eigen::MatrixXcd predicted = Eigen::MatrixXcd::Zero(n, nb + q);
      for (int j = 0; j < nb; ++j)
        predicted.row(out.interface[static_cast<std::size_t>(j)]) =
            out.resonantResponse.row(j);
      for (int i = 0; i < ni; ++i)
        predicted.row(out.interior[static_cast<std::size_t>(i)]) = interiorPart.row(i);
      out.reductionResidual =
          (applied - predicted).norm() / std::max(P.norm() * W.norm(), kTiny);
    }
    // The certificate of its null space: lift every null vector of the resonant
    // reduction back to the fine coordinates and measure how far it is from a
    // null vector of the pencil itself.
    const RankRead hat = rankRead(out.resonantResponse, rankTolerance);
    const int nullity = static_cast<int>(out.resonantResponse.cols()) - hat.rank;
    if (nullity > 0) {
      const double scaleP = std::max(P.norm(), kTiny);
      double worst = 0.0;
      for (int j = 0; j < nullity; ++j) {
        const Eigen::VectorXcd y = hat.V.col(hat.rank + j);
        const Eigen::VectorXcd x =
            out.constraintModes * y.head(nb) + out.resonantModes * y.tail(q);
        const double norm = std::max(x.norm(), kTiny);
        worst = std::max(worst, (P * x).norm() / (scaleP * norm));
      }
      out.liftResidual = worst;
    }
    return out;
  }
  const Complex product = out.interiorDeterminant * out.responseDeterminant;
  out.determinantResidual = std::abs(out.pencilDeterminant - product) /
                            std::max(std::abs(out.pencilDeterminant), kTiny);
  finishLogResiduals(out);
  return out;
}

FeshbachResult PencilSchur::sparseFeshbach(const SparseMatrix &A, const SparseMatrix &M,
                                           Complex lambda, const std::vector<int> &interface,
                                           double solveTolerance, SparseCostReport *report) {
  const int n = static_cast<int>(A.rows());
  if (A.cols() != n || M.rows() != n || M.cols() != n)
    throw std::invalid_argument(
        "PencilSchur::sparseFeshbach: A and M must be square of the same size");
  FeshbachResult out;
  out.lambda = lambda;
  std::set<int> kept;
  for (const int b : interface) {
    if (b < 0 || b >= n)
      throw std::invalid_argument("PencilSchur::sparseFeshbach: interface index out of range");
    kept.insert(b);
  }
  out.interface.assign(kept.begin(), kept.end());
  std::vector<int> position(static_cast<std::size_t>(n), -1);
  for (int i = 0; i < n; ++i)
    if (kept.find(i) == kept.end()) {
      position[static_cast<std::size_t>(i)] = static_cast<int>(out.interior.size());
      out.interior.push_back(i);
    }
  const int nb = static_cast<int>(out.interface.size());
  const int ni = static_cast<int>(out.interior.size());
  std::vector<int> interfacePosition(static_cast<std::size_t>(n), -1);
  for (int j = 0; j < nb; ++j)
    interfacePosition[static_cast<std::size_t>(out.interface[static_cast<std::size_t>(j)])] = j;
  const SparseMatrix P = SparseMatrix(A - lambda * M);
  // The four blocks, gathered from the stored entries alone: nothing of size
  // n x n is ever formed.
  std::vector<Eigen::Triplet<Complex>> interiorTrip;
  Eigen::MatrixXcd PBB = Eigen::MatrixXcd::Zero(nb, nb);
  Eigen::MatrixXcd PBI = Eigen::MatrixXcd::Zero(nb, ni);
  Eigen::MatrixXcd PIB = Eigen::MatrixXcd::Zero(ni, nb);
  for (int col = 0; col < P.outerSize(); ++col)
    for (SparseMatrix::InnerIterator it(P, col); it; ++it) {
      const int r = static_cast<int>(it.row());
      const int c = static_cast<int>(it.col());
      const int rb = interfacePosition[static_cast<std::size_t>(r)];
      const int cb = interfacePosition[static_cast<std::size_t>(c)];
      const int ri = position[static_cast<std::size_t>(r)];
      const int ci = position[static_cast<std::size_t>(c)];
      if (rb >= 0 && cb >= 0) PBB(rb, cb) = it.value();
      else if (rb >= 0) PBI(rb, ci) = it.value();
      else if (cb >= 0) PIB(ri, cb) = it.value();
      else interiorTrip.emplace_back(ri, ci, it.value());
    }
  out.interiorRank = ni;
  out.rangeProjector = Eigen::MatrixXcd::Identity(ni, ni);
  out.nullProjector = Eigen::MatrixXcd::Zero(ni, ni);
  out.interiorNullSpace = Eigen::MatrixXcd(ni, 0);
  out.interiorLeftNullSpace = Eigen::MatrixXcd(ni, 0);
  out.resonantModes = Eigen::MatrixXcd(n, 0);
  if (ni == 0) {
    out.response = PBB;
    out.interiorDeterminant = Complex(1.0, 0.0);
    out.constraintModes = Eigen::MatrixXcd::Identity(n, nb);
    out.solveResidual = 0.0;
    return out;
  }
  SparseMatrix PII(ni, ni);
  PII.setFromTriplets(interiorTrip.begin(), interiorTrip.end());
  PII.makeCompressed();
  const SparseCostMeter meter("bordered-lu", 0, n);
  Eigen::SparseLU<SparseMatrix> lu(PII);
  if (lu.info() != Eigen::Success)
    throw std::runtime_error(
        "PencilSchur::sparseFeshbach: the sparse factorization of the interior block failed at "
        "this shift, which is an interior resonance; the generalized inverse, its projectors and "
        "the resonant reduction rest on a singular value decomposition, so read the resonance "
        "with the dense PencilSchur::feshbach");
  out.interiorDeterminant = lu.determinant();
  const Eigen::MatrixXcd X = lu.solve(PIB);  // P_II^{-1} P_IB
  if (report)
    *report = meter.finish(static_cast<long long>(PII.rows()),
                           static_cast<long long>(PII.nonZeros()),
                           static_cast<long long>(lu.nnzL() + lu.nnzU()),
                           static_cast<long long>(nb));
  out.solveResidual = (PII * X - PIB).norm() / std::max(PIB.norm(), kTiny);
  // A sparse LU reveals no rank, and a determinant is no measure of singularity
  // at this size: the scale-free quantity the factorization does offer is the
  // residual of the solve it was asked for, and a block that cannot solve its
  // own interface load is a resonance.
  if (!(out.solveResidual <= solveTolerance))
    throw std::runtime_error(
        "PencilSchur::sparseFeshbach: the interior solve at this shift has relative residual " +
        std::to_string(out.solveResidual) + ", above the declared " +
        std::to_string(solveTolerance) +
        ": the interior block is numerically singular here, an interior resonance. Read it with "
        "the dense PencilSchur::feshbach, which carries the generalized inverse, its range and "
        "null projectors, and the resonant reduction");
  out.response = PBB - PBI * X;
  out.responseDeterminant = out.response.fullPivLu().determinant();
  out.constraintModes = Eigen::MatrixXcd::Zero(n, nb);
  for (int j = 0; j < nb; ++j) {
    out.constraintModes(out.interface[static_cast<std::size_t>(j)], j) = Complex(1.0, 0.0);
    for (int i = 0; i < ni; ++i)
      out.constraintModes(out.interior[static_cast<std::size_t>(i)], j) = -X(i, j);
  }
  // det P is the product of the two factors; it is not formed from a dense LU
  // of P, so the factorization residual is not measured on this path.
  out.pencilDeterminant = out.interiorDeterminant * out.responseDeterminant;
  out.determinantResidual = std::numeric_limits<double>::quiet_NaN();
  // The log-scale identity is not measured on the sparse path either: it needs
  // the determinant of P itself, which is never factorized here.
  return out;
}

CongruenceResult PencilSchur::craigBampton(const Eigen::MatrixXcd &A, const Eigen::MatrixXcd &M,
                                           const Eigen::MatrixXcd &T) {
  if (A.rows() != T.rows() || M.rows() != T.rows() || A.cols() != A.rows() || M.cols() != M.rows())
    throw std::invalid_argument("PencilSchur::craigBampton: T must have as many rows as A and M");
  CongruenceResult out;
  out.A = T.transpose() * A * T;
  out.M = T.transpose() * M * T;
  if (out.A.size() > 0)
    out.symmetryDefect = (out.A.transpose() - out.A).norm() / std::max(out.A.norm(), kTiny);
  if (out.M.size() > 0)
    out.metricSymmetryDefect = (out.M.transpose() - out.M).norm() / std::max(out.M.norm(), kTiny);
  if (T.size() > 0) {
    Eigen::JacobiSVD<Eigen::MatrixXcd> svd(T);
    const Eigen::VectorXd sv = svd.singularValues();
    out.basisConditionInverse = (sv(0) > 0.0) ? sv(sv.size() - 1) / sv(0) : 0.0;
  }
  return out;
}

SurrogateResult PencilSchur::craigBampton(const Eigen::MatrixXcd &A, const Eigen::MatrixXcd &M,
                                          const std::vector<int> &interface, Complex windowCentre,
                                          double windowRadius, double retentionRadius,
                                          Complex shift, double tolerance, double rankTolerance) {
  const int n = static_cast<int>(A.rows());
  if (A.cols() != n || M.rows() != n || M.cols() != n)
    throw std::invalid_argument("PencilSchur::craigBampton: A and M must be square of the same size");
  if (!(windowRadius >= 0.0))
    throw std::invalid_argument("PencilSchur::craigBampton: the window radius must be non-negative");
  if (!(retentionRadius >= 0.0))
    throw std::invalid_argument(
        "PencilSchur::craigBampton: the retention radius must be non-negative");
  SurrogateResult out;
  out.shift = shift;
  out.windowCentre = windowCentre;
  out.windowRadius = windowRadius;
  out.retentionRadius = retentionRadius;
  out.tolerance = tolerance;

  // The interface constraint modes at the declared shift, through the exact
  // Feshbach reduction: T = [I_B; -P_II(shift)^{-1} P_IB(shift)], or its
  // generalized-inverse form when the shift is an interior resonance.
  const FeshbachResult constraint = feshbach(A, M, shift, interface, rankTolerance);
  out.interface = constraint.interface;
  out.interior = constraint.interior;
  const int nb = static_cast<int>(out.interface.size());
  const int ni = static_cast<int>(out.interior.size());

  // The fixed-interface modes: the spectrum of the interior pencil (A_II, M_II).
  Eigen::MatrixXcd AII(ni, ni), MII(ni, ni);
  for (int i = 0; i < ni; ++i)
    for (int j = 0; j < ni; ++j) {
      AII(i, j) = A(out.interior[static_cast<std::size_t>(i)], out.interior[static_cast<std::size_t>(j)]);
      MII(i, j) = M(out.interior[static_cast<std::size_t>(i)], out.interior[static_cast<std::size_t>(j)]);
    }
  Eigen::MatrixXcd retained(ni, 0);
  if (ni > 0) {
    Eigen::FullPivLU<Eigen::MatrixXcd> metricLu(MII);
    metricLu.setThreshold(rankTolerance);
    if (!metricLu.isInvertible())
      throw std::runtime_error(
          "PencilSchur::craigBampton: the interior chain metric M_II is singular, so the "
          "fixed-interface pencil (A_II, M_II) has no spectrum to retain modes from");
    Eigen::ComplexEigenSolver<Eigen::MatrixXcd> interiorSolver(metricLu.solve(AII), true);
    if (interiorSolver.info() != Eigen::Success)
      throw std::runtime_error(
          "PencilSchur::craigBampton: the fixed-interface eigensolve did not converge");
    const Eigen::VectorXcd values = interiorSolver.eigenvalues();
    const std::vector<int> order = spectralOrder(values);
    std::vector<int> keep;
    for (const int index : order) {
      out.interiorEigenvalues.push_back(values(index));
      const double distance = std::abs(values(index) - windowCentre);
      if (distance <= retentionRadius)
        keep.push_back(index);
      else
        out.discardedModeSeparation = std::min(out.discardedModeSeparation, distance - windowRadius);
    }
    out.retainedModes = static_cast<int>(keep.size());
    retained = Eigen::MatrixXcd(ni, out.retainedModes);
    for (int t = 0; t < out.retainedModes; ++t) {
      Eigen::VectorXcd v = interiorSolver.eigenvectors().col(keep[static_cast<std::size_t>(t)]);
      v /= std::max(v.norm(), kTiny);
      retained.col(t) = v;
    }
  }

  // V = [T | [0; U]] and the congruence (V^T A V, V^T M V).
  out.basis = Eigen::MatrixXcd::Zero(n, nb + out.retainedModes);
  out.basis.leftCols(nb) = constraint.constraintModes;
  for (int t = 0; t < out.retainedModes; ++t)
    for (int i = 0; i < ni; ++i)
      out.basis(out.interior[static_cast<std::size_t>(i)], nb + t) = retained(i, t);
  out.reduced = craigBampton(A, M, out.basis);

  const int reducedDim = static_cast<int>(out.basis.cols());
  if (reducedDim > 0) {
    Eigen::FullPivLU<Eigen::MatrixXcd> reducedLu(out.reduced.M);
    reducedLu.setThreshold(rankTolerance);
    if (!reducedLu.isInvertible())
      throw std::runtime_error(
          "PencilSchur::craigBampton: the reduced chain metric V^T M V is singular, so the "
          "reduction basis is degenerate and its spectrum is not the surrogate spectrum; widen "
          "the retention radius or move the shift off the interior spectrum");
    Eigen::ComplexEigenSolver<Eigen::MatrixXcd> reducedSolver(reducedLu.solve(out.reduced.A), true);
    if (reducedSolver.info() != Eigen::Success)
      throw std::runtime_error("PencilSchur::craigBampton: the reduced eigensolve did not converge");
    const Eigen::VectorXcd values = reducedSolver.eigenvalues();
    const std::vector<int> order = spectralOrder(values);
    out.vectors = Eigen::MatrixXcd(reducedDim, reducedDim);
    for (int i = 0; i < reducedDim; ++i) {
      const int index = order[static_cast<std::size_t>(i)];
      out.eigenvalues.push_back(values(index));
      Eigen::VectorXcd y = reducedSolver.eigenvectors().col(index);
      y /= std::max(y.norm(), kTiny);
      out.vectors.col(i) = y;
    }
  }

  // Hold every claimed eigenvalue to the exact Feshbach map.
  const double scaleA = std::max(A.norm(), kTiny);
  double worstBound = 0.0;
  bool everyPairHolds = true;
  for (int i = 0; i < static_cast<int>(out.eigenvalues.size()); ++i) {
    const Complex theta = out.eigenvalues[static_cast<std::size_t>(i)];
    if (std::abs(theta - windowCentre) > windowRadius) continue;
    out.windowIndices.push_back(i);
    const Eigen::VectorXcd x = out.basis * out.vectors.col(i);
    const Eigen::MatrixXcd P = A - theta * M;
    const Eigen::VectorXcd residualVector = P * x;
    out.residuals.push_back(residualVector.norm() / (scaleA * std::max(x.norm(), kTiny)));
    Eigen::VectorXcd xB(nb);
    for (int j = 0; j < nb; ++j) xB(j) = x(out.interface[static_cast<std::size_t>(j)]);
    const FeshbachResult exact = feshbach(A, M, theta, interface, rankTolerance);
    out.resonantAtEigenvalue.push_back(exact.interiorSingular);
    const double scaleB = std::max(xB.norm(), kTiny);
    if (!exact.interiorSingular) {
      const double scaleF = std::max(exact.response.norm(), kTiny);
      const double defect = (exact.response * xB).norm() / (scaleF * scaleB);
      // ||P_BI P_II^{-1}|| from the transposed solve, so that the interior
      // block is factorized once per column block and never inverted.
      Eigen::MatrixXcd PII(ni, ni), PBI(nb, ni);
      for (int a = 0; a < ni; ++a)
        for (int b = 0; b < ni; ++b)
          PII(a, b) = P(out.interior[static_cast<std::size_t>(a)], out.interior[static_cast<std::size_t>(b)]);
      for (int a = 0; a < nb; ++a)
        for (int b = 0; b < ni; ++b)
          PBI(a, b) = P(out.interface[static_cast<std::size_t>(a)], out.interior[static_cast<std::size_t>(b)]);
      double amplification = 0.0;
      if (ni > 0) {
        const Eigen::MatrixXcd W = PII.transpose().fullPivLu().solve(PBI.transpose());
        Eigen::JacobiSVD<Eigen::MatrixXcd> wsvd(W);
        amplification = wsvd.singularValues()(0);
      }
      const double bound = (1.0 + amplification) * residualVector.norm() / (scaleF * scaleB);
      out.feshbachDefects.push_back(defect);
      out.feshbachBounds.push_back(bound);
      // The inequality is exact; the comparison allows for round-off in the
      // norms that enter both sides.
      const bool holds = defect <= bound * (1.0 + 1e-6) + kTiny;
      out.feshbachHolds.push_back(holds);
      everyPairHolds = everyPairHolds && holds;
      worstBound = std::max(worstBound, bound);
    } else {
      // At an interior resonance F_B is replaced by the resonant reduction, in
      // whose coordinates the claimed pair is (x_B, N^H x_I).
      Eigen::VectorXcd xI(ni);
      for (int j = 0; j < ni; ++j) xI(j) = x(out.interior[static_cast<std::size_t>(j)]);
      Eigen::VectorXcd coordinates(exact.resonantResponse.cols());
      coordinates.head(nb) = xB;
      if (exact.resonantResponse.cols() > nb)
        coordinates.tail(exact.resonantResponse.cols() - nb) =
            exact.interiorNullSpace.adjoint() * xI;
      const double scaleF = std::max(exact.resonantResponse.norm(), kTiny);
      const double defect = (exact.resonantResponse * coordinates).norm() /
                            (scaleF * std::max(coordinates.norm(), kTiny));
      out.feshbachDefects.push_back(defect);
      out.feshbachBounds.push_back(std::numeric_limits<double>::quiet_NaN());
      out.feshbachHolds.push_back(defect <= tolerance);
      everyPairHolds = everyPairHolds && (defect <= tolerance);
    }
  }
  const bool separated = out.discardedModeSeparation > 0.0;
  const bool bounded = worstBound <= tolerance;
  out.certified = everyPairHolds && separated && bounded;
  if (!out.certified) {
    if (!separated)
      out.refusal =
          "a discarded fixed-interface mode lies inside the declared window (separation " +
          std::to_string(out.discardedModeSeparation) +
          "); widen the retention radius until every mode of the window is retained";
    else if (!bounded)
      out.refusal = "the certified Feshbach bound " + std::to_string(worstBound) +
                    " exceeds the declared tolerance " + std::to_string(tolerance) +
                    "; the surrogate spectrum is not held to the exact map this closely";
    else
      out.refusal =
          "a claimed eigenvalue's Feshbach defect exceeds the bound its own residual certifies; "
          "the reduced eigenpair and the fine residual disagree";
  }
  return out;
}

FiberRestriction PencilSchur::restrictToFibers(const Eigen::MatrixXcd &A, const Eigen::MatrixXcd &M,
                                               const Eigen::MatrixXcd &Z) {
  return restrictToFibers(A, M, std::vector<Eigen::MatrixXcd>{Z});
}

FiberRestriction PencilSchur::restrictToFibers(const Eigen::MatrixXcd &A, const Eigen::MatrixXcd &M,
                                               const std::vector<Eigen::MatrixXcd> &fibers) {
  int total = 0;
  FiberRestriction out;
  for (const auto &Z : fibers) {
    if (Z.rows() != A.rows()) throw std::invalid_argument("PencilSchur::restrictToFibers: a fiber has the wrong length");
    out.blockOffsets.push_back(total);
    out.blockRanks.push_back(static_cast<int>(Z.cols()));
    total += static_cast<int>(Z.cols());
  }
  Eigen::MatrixXcd J(A.rows(), total);
  int offset = 0;
  for (const auto &Z : fibers) {
    J.middleCols(offset, static_cast<int>(Z.cols())) = Z;
    offset += static_cast<int>(Z.cols());
  }
  out.A = J.transpose() * A * J;
  out.gram = J.transpose() * M * J;
  return out;
}

Eigen::MatrixXcd PencilSchur::gramBlock(const Eigen::MatrixXcd &M, const Eigen::MatrixXcd &ZA,
                                        const Eigen::MatrixXcd &ZB) {
  return ZA.transpose() * M * ZB;
}

std::vector<int> PencilSchur::support(const Eigen::MatrixXcd &Z, double threshold) {
  const double scale = Z.size() ? Z.cwiseAbs().maxCoeff() : 0.0;
  std::vector<int> out;
  for (int i = 0; i < Z.rows(); ++i)
    if (Z.row(i).cwiseAbs().maxCoeff() > threshold * scale) out.push_back(i);
  return out;
}

bool PencilSchur::supportsShareTopSimplex(const cobordism::ChainComplex &K, int k,
                                          const std::vector<int> &supportA,
                                          const std::vector<int> &supportB) {
  const auto cells = K.kSimplexVertices(k);
  const std::set<int> a(supportA.begin(), supportA.end());
  const std::set<int> b(supportB.begin(), supportB.end());
  for (const auto &T : K.orientedTopSimplices()) {
    const std::set<std::uint64_t> top(T.begin(), T.end());
    bool hasA = false, hasB = false;
    for (const int i : a) {
      const auto &c = cells[static_cast<std::size_t>(i)];
      if (std::all_of(c.begin(), c.end(), [&](std::uint64_t v) { return top.count(v) > 0; })) { hasA = true; break; }
    }
    if (!hasA) continue;
    for (const int i : b) {
      const auto &c = cells[static_cast<std::size_t>(i)];
      if (std::all_of(c.begin(), c.end(), [&](std::uint64_t v) { return top.count(v) > 0; })) { hasB = true; break; }
    }
    if (hasB) return true;
  }
  return false;
}

TransferResult PencilSchur::transfer(const Eigen::MatrixXcd &AtildeU, const Eigen::MatrixXcd &AtildeUinv,
                                     const Eigen::MatrixXcd &ZA, const Eigen::MatrixXcd &ZAdual,
                                     const Eigen::MatrixXcd &ZB, const Eigen::MatrixXcd &ZBdual,
                                     double tolerance) {
  const int n = static_cast<int>(AtildeU.rows());
  if (AtildeUinv.rows() != n || ZA.rows() != n || ZAdual.rows() != n || ZB.rows() != n || ZBdual.rows() != n)
    throw std::invalid_argument("PencilSchur::transfer: every operand must live on the same coordinates");
  if (ZA.cols() != ZAdual.cols() || ZB.cols() != ZBdual.cols())
    throw std::invalid_argument("PencilSchur::transfer: a fiber and its dual must have the same rank");
  TransferResult out;
  out.tolerance = tolerance;
  // T_AB(U) = (Z_A^vee)^T A~^U Z_B;  T_BA(U^{-1}) = (Z_B^vee(U^{-1}))^T A~^{U^{-1}} Z_A(U^{-1}),
  // and the band of the dual connection has Z^vee(U^{-1}) = Z(U): the reverse
  // transfer is Z_B^T A~^{U^{-1}} Z_A^vee.
  out.forward = ZAdual.transpose() * AtildeU * ZB;
  out.reverse = ZB.transpose() * AtildeUinv * ZAdual;
  const double scale = std::max(out.forward.norm(), 1e-300);
  out.reversalResidual = (out.reverse - out.forward.transpose()).norm() / scale;
  if (!(out.reversalResidual <= tolerance))
    throw std::runtime_error(
        "PencilSchur::transfer: the reversal identity T_BA(U^-1) = T_AB(U)^T fails: relative residual " +
        std::to_string(out.reversalResidual) + " exceeds " + std::to_string(tolerance) +
        " (the transposes of the dressed pencil do not match: check (A~^U)^T = A~^{U^-1})");
  if (out.forward.rows() == out.forward.cols() && out.forward.rows() > 0) {
    const Eigen::MatrixXcd product = out.reverse * out.forward;
    const int r = static_cast<int>(out.forward.rows());
    out.groupoidResidual = (product - Eigen::MatrixXcd::Identity(r, r)).norm() / std::sqrt(static_cast<double>(r));
    out.groupoidHolds = out.groupoidResidual <= tolerance;
    if (out.groupoidHolds) out.dualTransfer = out.forward.transpose().inverse();
  }
  return out;
}

}  // namespace tessera::chainhodge
