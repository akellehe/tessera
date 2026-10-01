// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "chainhodge/RieszProjector.h"

#include <complex>

#include <Eigen/Dense>

namespace tessera::chainhodge {

namespace {

using Complex = std::complex<double>;

/// One adjacent swap of a complex Schur form: the unitary rotation \f$ G \f$
/// on positions \f$ k, k+1 \f$ that moves the eigenvalue at \f$ k+1 \f$ to
/// \f$ k \f$, applied as \f$ T \leftarrow G^H T G \f$ and \f$ Q \leftarrow QG \f$,
/// so that \f$ Q T Q^H \f$ is unchanged and the triangular form is kept. The
/// first column of \f$ G \f$ is the eigenvector of the \f$ 2\times 2 \f$ block
/// for the eigenvalue being moved up.
void swapAdjacent(Eigen::MatrixXcd &T, Eigen::MatrixXcd &Q, int k) {
  const Complex a = T(k, k), b = T(k, k + 1), c = T(k + 1, k + 1);
  Eigen::Matrix2cd G;
  if (b == Complex(0.0, 0.0)) {
    G << Complex(0.0, 0.0), Complex(1.0, 0.0), Complex(1.0, 0.0),
        Complex(0.0, 0.0);
  } else {
    Eigen::Vector2cd v;
    v << b, c - a;
    const double scale = v.norm();
    Eigen::Vector2cd w;
    w << -std::conj(c - a), std::conj(b);
    G.col(0) = v / scale;
    G.col(1) = w / scale;
  }
  T.middleCols(k, 2) = (T.middleCols(k, 2) * G).eval();
  T.middleRows(k, 2) = (G.adjoint() * T.middleRows(k, 2)).eval();
  Q.middleCols(k, 2) = (Q.middleCols(k, 2) * G).eval();
  T(k + 1, k) = Complex(0.0, 0.0);
}

}  // namespace

RieszProjectorRead rieszProjector(
    const Eigen::ComplexSchur<Eigen::MatrixXcd> &schur,
    const std::vector<bool> &enclosed) {
  Eigen::MatrixXcd T = schur.matrixT();
  Eigen::MatrixXcd Q = schur.matrixU();
  const int n = static_cast<int>(T.rows());
  // Positions are scanned upward; every flagged eigenvalue met so far has
  // already been moved into the leading block, so the entries between the
  // block and the current position are all unflagged and the flags of the
  // positions still to be scanned are untouched by the swaps.
  int lead = 0;
  for (int p = 0; p < n; ++p) {
    if (!enclosed[static_cast<std::size_t>(p)]) continue;
    for (int k = p; k > lead; --k) swapAdjacent(T, Q, k - 1);
    ++lead;
  }
  const int q = lead;
  const int m = n - q;
  const Eigen::MatrixXcd T11 = T.topLeftCorner(q, q);
  const Eigen::MatrixXcd T12 = T.topRightCorner(q, m);
  const Eigen::MatrixXcd T22 = T.bottomRightCorner(m, m);
  Eigen::MatrixXcd Y(q, m);
  for (int j = 0; j < m; ++j) {
    Eigen::VectorXcd rhs = T12.col(j);
    for (int i = 0; i < j; ++i) rhs += T22(i, j) * Y.col(i);
    const Eigen::MatrixXcd shifted =
        T11 - T22(j, j) * Eigen::MatrixXcd::Identity(q, q);
    Y.col(j) = shifted.triangularView<Eigen::Upper>().solve(rhs);
  }
  RieszProjectorRead out;
  out.right = Q.leftCols(q);
  Eigen::MatrixXcd leftRows(q, n);  // [I Y] Q^H
  leftRows.leftCols(q) = Eigen::MatrixXcd::Identity(q, q);
  leftRows.rightCols(m) = Y;
  leftRows = (leftRows * Q.adjoint()).eval();
  out.left = leftRows.transpose();
  out.projector = out.right * leftRows;
  return out;
}

}  // namespace tessera::chainhodge
