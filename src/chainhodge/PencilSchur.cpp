// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "chainhodge/PencilSchur.h"

#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>

#include <Eigen/Dense>

namespace tessera::chainhodge {

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
  if (ni == 0) {
    out.response = PBB;
    out.interiorDeterminant = Complex(1.0, 0.0);
    out.interiorLogDeterminant = Complex(0.0, 0.0);
    out.responseDeterminant = PBB.fullPivLu().determinant();
    out.constraintModes = Eigen::MatrixXcd::Identity(n, nb);
    out.solveResidual = 0.0;
  } else {
    Eigen::FullPivLU<Eigen::MatrixXcd> lu(PII);
    lu.setThreshold(rankTolerance);
    out.interiorDeterminant = lu.determinant();
    out.interiorLogDeterminant = logDeterminant(PII);
    if (!lu.isInvertible()) {
      out.interiorSingular = true;
      out.determinantResidual = std::numeric_limits<double>::quiet_NaN();
      return out;
    }
    const Eigen::MatrixXcd X = lu.solve(PIB);  // P_II^{-1} P_IB
    out.solveResidual = (PII * X - PIB).norm() / std::max(PIB.norm(), 1e-300);
    out.response = PBB - PBI * X;
    out.responseDeterminant = out.response.fullPivLu().determinant();
    out.constraintModes = Eigen::MatrixXcd::Zero(n, nb);
    for (int j = 0; j < nb; ++j) {
      out.constraintModes(out.interface[static_cast<std::size_t>(j)], j) = Complex(1.0, 0.0);
      for (int i = 0; i < ni; ++i) out.constraintModes(out.interior[static_cast<std::size_t>(i)], j) = -X(i, j);
    }
  }
  const Complex product = out.interiorDeterminant * out.responseDeterminant;
  out.determinantResidual = std::abs(out.pencilDeterminant - product) /
                            std::max(std::abs(out.pencilDeterminant), 1e-300);
  out.responseLogDeterminant = logDeterminant(out.response);
  const Complex defect = out.pencilLogDeterminant - out.interiorLogDeterminant - out.responseLogDeterminant;
  constexpr double kTwoPi = 6.28318530717958647692;
  out.logModulusResidual = std::abs(defect.real());
  out.logPhaseResidual = std::abs(std::remainder(defect.imag(), kTwoPi));
  return out;
}

CongruenceResult PencilSchur::craigBampton(const Eigen::MatrixXcd &A, const Eigen::MatrixXcd &M,
                                           const Eigen::MatrixXcd &T) {
  if (A.rows() != T.rows() || M.rows() != T.rows() || A.cols() != A.rows() || M.cols() != M.rows())
    throw std::invalid_argument("PencilSchur::craigBampton: T must have as many rows as A and M");
  CongruenceResult out;
  out.A = T.transpose() * A * T;
  out.M = T.transpose() * M * T;
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
