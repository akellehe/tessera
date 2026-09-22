// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "chainhodge/SparseRank.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/OrderingMethods>
#include <Eigen/SVD>
#include <Eigen/SparseLU>
#include <Eigen/SparseQR>

namespace tessera::chainhodge {

namespace {

constexpr double kEps = std::numeric_limits<double>::epsilon();
constexpr double kInf = std::numeric_limits<double>::infinity();

using Mat = Eigen::MatrixXcd;
using Operator = std::function<Mat(const Mat &)>;

// Iteration controls. The largest singular value only sets a tolerance and a
// scale; the smallest is the reported margin and is iterated to near rounding.
constexpr int kPowerBlock = 6;
constexpr int kPowerIterations = 300;
constexpr double kPowerTolerance = 1e-12;
constexpr int kInverseBlock = 6;
constexpr int kInverseIterations = 500;
constexpr double kInverseTolerance = 1e-13;
constexpr int kNoiseIterations = 30;
constexpr std::uint64_t kSeed = 0x5eed1204ULL;

Mat randomBlock(Eigen::Index n, Eigen::Index b, std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::normal_distribution<double> g(0.0, 1.0);
  Mat X(n, b);
  for (Eigen::Index j = 0; j < b; ++j)
    for (Eigen::Index i = 0; i < n; ++i) X(i, j) = Complex(g(rng), g(rng));
  return X;
}

Mat orthonormalize(const Mat &X) {
  Eigen::HouseholderQR<Mat> qr(X);
  return qr.householderQ() * Mat::Identity(X.rows(), X.cols());
}

// Largest singular value of the operator A (n columns) by block power
// iteration on A^H A; Ritz values sigma(A X) for orthonormal X rise to it.
double largestSingular(const Operator &A, const Operator &AH, Eigen::Index n, std::uint64_t seed,
                       int iterations = kPowerIterations, double rtol = kPowerTolerance) {
  if (n == 0) return 0.0;
  const Eigen::Index b = std::min<Eigen::Index>(kPowerBlock, n);
  Mat X = orthonormalize(randomBlock(n, b, seed));
  double prev = -1.0;
  double value = 0.0;
  for (int it = 0; it < iterations; ++it) {
    const Mat Y = A(X);
    Eigen::JacobiSVD<Mat> svd(Y);
    value = svd.singularValues()(0);
    if (value == 0.0) return 0.0;
    if (std::abs(value - prev) <= rtol * value) break;
    prev = value;
    X = orthonormalize(AH(Y));
  }
  return value;
}

// The b smallest singular values of a full-row-rank operator W (r rows) by
// inverse subspace iteration: U <- (W W^H)^{-1} U, with Ritz values
// sigma(W^H U) for orthonormal U, which fall to the smallest singular values.
Eigen::VectorXd smallestSingular(const Operator &WH, const Operator &gramSolve, Eigen::Index r,
                                 std::uint64_t seed) {
  if (r == 0) return Eigen::VectorXd();
  const Eigen::Index b = std::min<Eigen::Index>(kInverseBlock, r);
  Mat U = orthonormalize(randomBlock(r, b, seed));
  Eigen::VectorXd theta = Eigen::VectorXd::Constant(b, kInf);
  for (int it = 0; it < kInverseIterations; ++it) {
    U = orthonormalize(gramSolve(U));
    const Mat Y = WH(U);
    Eigen::JacobiSVD<Mat> svd(Y, Eigen::ComputeThinV);
    // Rotate the block to its Ritz vectors, smallest last as in the SVD order.
    U = U * svd.matrixV();
    const Eigen::VectorXd sv = svd.singularValues();
    Eigen::VectorXd next(b);
    for (Eigen::Index i = 0; i < b; ++i) next(i) = sv(b - 1 - i);  // ascending
    const double change = std::abs(next(0) - theta(0));
    theta = next;
    if (change <= kInverseTolerance * std::max(theta(0), std::numeric_limits<double>::min())) break;
  }
  return theta;
}

double spectralNorm(const Mat &A) {
  if (A.rows() == 0 || A.cols() == 0) return 0.0;
  if (A.cols() <= 256) {
    Eigen::JacobiSVD<Mat> svd(A);
    return svd.singularValues()(0);
  }
  return largestSingular([&](const Mat &X) { return Mat(A * X); },
                         [&](const Mat &Y) { return Mat(A.adjoint() * Y); }, A.cols(), kSeed + 7);
}

// Rows of A that hold a nonzero, in order. Eigen's SparseQR refuses a matrix
// with an empty row; an empty row carries no singular value and no constraint.
std::vector<int> occupiedRows(const SparseMatrix &A) {
  std::vector<char> hit(static_cast<std::size_t>(A.rows()), 0);
  for (int c = 0; c < A.outerSize(); ++c)
    for (SparseMatrix::InnerIterator it(A, c); it; ++it)
      if (it.value() != Complex(0.0, 0.0)) hit[static_cast<std::size_t>(it.row())] = 1;
  std::vector<int> rows;
  for (int i = 0; i < static_cast<int>(A.rows()); ++i)
    if (hit[static_cast<std::size_t>(i)]) rows.push_back(i);
  return rows;
}

template <typename Scalar>
Eigen::SparseMatrix<Scalar> selectRows(const Eigen::SparseMatrix<Scalar> &A, const std::vector<int> &rows) {
  std::vector<int> map(static_cast<std::size_t>(A.rows()), -1);
  for (std::size_t i = 0; i < rows.size(); ++i) map[static_cast<std::size_t>(rows[i])] = static_cast<int>(i);
  std::vector<Eigen::Triplet<Scalar>> trip;
  trip.reserve(static_cast<std::size_t>(A.nonZeros()));
  for (int c = 0; c < A.outerSize(); ++c)
    for (typename Eigen::SparseMatrix<Scalar>::InnerIterator it(A, c); it; ++it) {
      const int r = map[static_cast<std::size_t>(it.row())];
      if (r >= 0 && it.value() != Scalar(0)) trip.emplace_back(r, static_cast<int>(it.col()), it.value());
    }
  Eigen::SparseMatrix<Scalar> B(static_cast<Eigen::Index>(rows.size()), A.cols());
  B.setFromTriplets(trip.begin(), trip.end());
  B.makeCompressed();
  return B;
}

}  // namespace

SingularSplit SparseRank::fromSingularValues(const Eigen::VectorXd &sv, Eigen::Index rows,
                                             Eigen::Index cols, double kappa, int at) {
  SingularSplit out;
  out.dense = true;
  const Eigen::Index count = sv.size();
  out.largest = count > 0 ? sv(0) : 0.0;
  out.tolerance = kappa * static_cast<double>(std::max(rows, cols)) * kEps * out.largest;
  int r = 0;
  for (Eigen::Index i = 0; i < count; ++i)
    if (sv(i) > out.tolerance) ++r;
  out.rank = r;
  out.at = at < 0 ? r : at;
  out.sigmaAt = out.at >= 1 ? (out.at <= count ? sv(out.at - 1) : 0.0) : kInf;
  out.sigmaNext = out.at < count ? sv(out.at) : 0.0;
  out.gap = out.sigmaNext > 0.0 ? out.sigmaAt / out.sigmaNext : kInf;
  return out;
}

SparseKernel SparseRank::kernel(const SparseMatrix &Ain, double kappa) {
  const Eigen::Index m = Ain.rows();
  const Eigen::Index n = Ain.cols();
  SparseKernel out;
  out.split.dense = false;
  if (n == 0) {
    out.basis = Mat(0, 0);
    return out;
  }
  const std::vector<int> rows = occupiedRows(Ain);
  if (rows.empty()) {
    out.basis = Mat::Identity(n, n);
    return out;
  }
  const SparseMatrix A = selectRows(Ain, rows);
  const Eigen::Index mA = A.rows();

  const double largest = largestSingular([&](const Mat &X) { return Mat(A * X); },
                                         [&](const Mat &Y) { return Mat(A.adjoint() * Y); }, n, kSeed);
  const double tol = kappa * static_cast<double>(std::max(m, n)) * kEps * largest;
  out.split.largest = largest;
  out.split.tolerance = tol;

  Eigen::SparseQR<SparseMatrix, Eigen::COLAMDOrdering<int>> qr;
  qr.setPivotThreshold(tol);
  qr.compute(A);
  if (qr.info() != Eigen::Success)
    throw std::runtime_error("SparseRank::kernel: sparse QR failed: " + qr.lastErrorMessage());
  const Eigen::Index r = qr.rank();
  out.split.rank = static_cast<int>(r);
  out.split.at = static_cast<int>(r);

  // A P = Q [R11 R12; 0 0]: the kernel is P [-R11^{-1} R12; I].
  const SparseMatrix Rtop = qr.matrixR().topRows(r);
  const SparseMatrix R11 = Rtop.leftCols(r);
  const SparseMatrix R11h = R11.adjoint();
  const SparseMatrix R12 = Rtop.rightCols(n - r);
  const Mat Cm = (r > 0 && n > r) ? Mat(R11.triangularView<Eigen::Upper>().solve(Mat(R12)))
                                  : Mat::Zero(r, n - r);
  Mat Y(n, n - r);
  if (n > r) {
    Y.topRows(r) = -Cm;
    Y.bottomRows(n - r).setIdentity();
    out.basis = orthonormalize(qr.colsPermutation() * Y);
  } else {
    out.basis = Mat(n, 0);
  }

  // The first discarded singular value, bounded by ||A N|| (Courant-Fischer);
  // none exists when the rank is the full min(m, n).
  if (r < std::min(mA, n)) out.split.sigmaNext = spectralNorm(Mat(A * out.basis));

  // The last kept one: sigma_min(W), W = [R11 R12], by inverse iteration on
  // W W^H = R11 (I + C C^H) R11^H with C = R11^{-1} R12 (Woodbury).
  if (r > 0) {
    Eigen::LLT<Mat> small;
    if (n > r) small.compute(Mat::Identity(n - r, n - r) + Cm.adjoint() * Cm);
    const Operator gramSolve = [&](const Mat &X) {
      Mat Z = R11.triangularView<Eigen::Upper>().solve(X);
      if (n > r) Z -= Cm * small.solve(Cm.adjoint() * Z);
      return Mat(R11h.triangularView<Eigen::Lower>().solve(Z));
    };
    const Operator WH = [&](const Mat &U) {
      Mat V(n, U.cols());
      V.topRows(r) = R11h * U;
      if (n > r) V.bottomRows(n - r) = R12.adjoint() * U;
      return V;
    };
    out.split.sigmaAt = smallestSingular(WH, gramSolve, r, kSeed + 1)(0);
  }
  out.split.gap = out.split.sigmaNext > 0.0 ? out.split.sigmaAt / out.split.sigmaNext : kInf;
  return out;
}

SingularSplit SparseRank::congruence(const SparseMatrix &C, const SparseMatrix &X, bool inverse,
                                     int structuralRank, double kappa) {
  const Eigen::Index p = C.rows();
  const Eigen::Index q = C.cols();
  if (X.rows() != p || X.cols() != p)
    throw std::invalid_argument("SparseRank::congruence: X is " + std::to_string(X.rows()) + "x" +
                                std::to_string(X.cols()) + ", C has " + std::to_string(p) + " rows");
  const int rho = structuralRank;
  if (rho < 0 || rho > std::min(p, q))
    throw std::invalid_argument("SparseRank::congruence: structural rank " + std::to_string(rho) +
                                " outside [0, min(p, q)]");
  SingularSplit out;
  out.dense = false;
  out.at = rho;
  if (q == 0) return out;
  if (rho == 0) {
    for (int c = 0; c < C.outerSize(); ++c)
      for (SparseMatrix::InnerIterator it(C, c); it; ++it)
        if (it.value() != Complex(0.0, 0.0))
          throw std::logic_error("SparseRank::congruence: structural rank 0 with a nonzero C");
    return out;  // P = 0: nothing required, nothing beyond
  }

  // QR of the integer matrix C^T (real), without its zero rows (zero columns
  // of C, which P annihilates exactly).
  const SparseMatrix Ctc = C.transpose();
  const std::vector<int> kept = occupiedRows(Ctc);
  Eigen::SparseMatrix<double> Ct = selectRows(Ctc, kept).real();
  Ct.makeCompressed();
  const Eigen::Index qk = Ct.rows();
  Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> qc;
  qc.compute(Ct);
  if (qc.info() != Eigen::Success)
    throw std::runtime_error("SparseRank::congruence: sparse QR of C^T failed: " + qc.lastErrorMessage());
  if (qc.rank() != rho)
    throw std::logic_error("SparseRank::congruence: the sparse QR of C^T finds rank " +
                           std::to_string(qc.rank()) + ", the exact rank is " + std::to_string(rho));
  const SparseMatrix W = SparseMatrix(qc.matrixR().topRows(rho).cast<Complex>());
  const SparseMatrix Wt = W.transpose();
  const Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int> Pi = qc.colsPermutation();
  const SparseMatrix Xp = SparseMatrix(Pi.transpose() * X * Pi);

  // T = W Xp^{+-1} W^T (rho x rho, complex symmetric) and its inverse.
  bool singular = false;
  SparseMatrix T;
  Eigen::SparseLU<SparseMatrix, Eigen::COLAMDOrdering<int>> luT, luX, luK;
  Operator apply, solve;
  if (!inverse) {
    T = SparseMatrix(W * Xp * Wt);
    T.makeCompressed();
    luT.compute(T);
    singular = luT.info() != Eigen::Success;
    apply = [&](const Mat &Y) { return Mat(T * Y); };
    solve = [&](const Mat &Y) { return Mat(luT.solve(Y)); };
  } else {
    luX.compute(Xp);
    if (luX.info() != Eigen::Success)
      throw std::invalid_argument("SparseRank::congruence: X is singular; X^{-1} is undefined");
    std::vector<Eigen::Triplet<Complex>> trip;
    trip.reserve(static_cast<std::size_t>(Xp.nonZeros() + 2 * W.nonZeros()));
    for (int c = 0; c < Xp.outerSize(); ++c)
      for (SparseMatrix::InnerIterator it(Xp, c); it; ++it)
        trip.emplace_back(static_cast<int>(it.row()), static_cast<int>(it.col()), it.value());
    for (int c = 0; c < W.outerSize(); ++c)
      for (SparseMatrix::InnerIterator it(W, c); it; ++it) {
        trip.emplace_back(static_cast<int>(p + it.row()), static_cast<int>(it.col()), it.value());
        trip.emplace_back(static_cast<int>(it.col()), static_cast<int>(p + it.row()), it.value());
      }
    SparseMatrix K(p + rho, p + rho);
    K.setFromTriplets(trip.begin(), trip.end());
    K.makeCompressed();
    luK.compute(K);
    singular = luK.info() != Eigen::Success;
    apply = [&](const Mat &Y) { return Mat(W * Mat(luX.solve(Mat(Wt * Y)))); };
    // [Xp W^T; W 0][u; z] = [0; -y]  =>  W Xp^{-1} W^T z = y.
    solve = [&](const Mat &Y) {
      Mat rhs = Mat::Zero(p + rho, Y.cols());
      rhs.bottomRows(rho) = -Y;
      return Mat(Mat(luK.solve(rhs)).bottomRows(rho));
    };
  }
  // T is complex symmetric: T^H y = conj(T conj y), and likewise its inverse.
  const Operator applyH = [&](const Mat &Y) { return Mat(apply(Y.conjugate()).conjugate()); };
  const Operator solveH = [&](const Mat &Y) { return Mat(solve(Y.conjugate()).conjugate()); };

  out.largest = largestSingular(apply, applyH, rho, kSeed + 2);
  out.tolerance = kappa * static_cast<double>(q) * kEps * out.largest;
  if (singular) {
    out.sigmaAt = 0.0;
    out.rank = rho - 1;
  } else {
    const Eigen::VectorXd theta = smallestSingular(applyH, [&](const Mat &U) { return solveH(solve(U)); },
                                                   rho, kSeed + 3);
    int below = 0;
    for (Eigen::Index i = 0; i < theta.size(); ++i)
      if (theta(i) <= out.tolerance) ++below;
    out.rank = rho - below;
    out.sigmaAt = theta(0);
  }

  // sigma_{rho+1}: ||P N_C|| on the orthonormal basis N_C of ker C, the last
  // qk - rho columns of the QR's Q (never formed) with the zero columns of C
  // as exact kernel vectors on which P vanishes identically.
  const Eigen::Index nullC = qk - rho;
  if (nullC > 0) {
    std::vector<int> map(static_cast<std::size_t>(q), -1);
    for (std::size_t i = 0; i < kept.size(); ++i) map[static_cast<std::size_t>(kept[i])] = static_cast<int>(i);
    const auto qApply = [&](const Eigen::MatrixXd &V) { return Eigen::MatrixXd(qc.matrixQ() * V); };
    const auto qtApply = [&](const Eigen::MatrixXd &V) { return Eigen::MatrixXd(qc.matrixQ().transpose() * V); };
    const Operator N = [&](const Mat &V) {  // q x nullC -> embedded kernel vectors
      Mat pad = Mat::Zero(qk, V.cols());
      pad.bottomRows(nullC) = V;
      const Mat inner = qApply(pad.real()).cast<Complex>() + Complex(0.0, 1.0) * qApply(pad.imag()).cast<Complex>();
      Mat full = Mat::Zero(q, V.cols());
      for (Eigen::Index i = 0; i < qk; ++i) full.row(kept[static_cast<std::size_t>(i)]) = inner.row(i);
      return full;
    };
    const Operator NH = [&](const Mat &Y) {
      Mat gathered(qk, Y.cols());
      for (Eigen::Index i = 0; i < qk; ++i) gathered.row(i) = Y.row(kept[static_cast<std::size_t>(i)]);
      const Mat inner = qtApply(gathered.real()).cast<Complex>() + Complex(0.0, 1.0) * qtApply(gathered.imag()).cast<Complex>();
      return Mat(inner.bottomRows(nullC));
    };
    // P y = C^T X^{+-1} C y, with X^{-1} through the permuted factorization.
    const Operator P = [&](const Mat &Y) {
      const Mat c = C * Y;
      Mat x;
      if (inverse) x = Pi * Mat(luX.solve(Mat(Pi.transpose() * c)));
      else x = X * c;
      return Mat(C.transpose() * x);
    };
    const Operator PH = [&](const Mat &Y) { return Mat(P(Y.conjugate()).conjugate()); };
    out.sigmaNext = largestSingular([&](const Mat &V) { return P(N(V)); },
                                    [&](const Mat &Y) { return NH(PH(Y)); }, nullC, kSeed + 4,
                                    kNoiseIterations, 1e-3);
  }
  out.gap = out.sigmaNext > 0.0 ? out.sigmaAt / out.sigmaNext : kInf;
  return out;
}

}  // namespace tessera::chainhodge
