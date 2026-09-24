// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "chainhodge/GrownCellRule.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#include <Eigen/Dense>

namespace tessera::chainhodge {

namespace {

using Complex = std::complex<double>;

std::vector<std::pair<int, int>> localEdges(int d) {
  std::vector<std::pair<int, int>> edges;
  for (int i = 0; i <= d; ++i)
    for (int j = i + 1; j <= d; ++j) edges.emplace_back(i, j);
  return edges;
}

/// Index of the unknown \f$ (C\Gamma)_{ab} \f$, \f$ a\le b \f$, in row-major
/// upper-triangular order over \f$ d+1 \f$ vertices.
int unknownIndex(int a, int b, int n) {
  if (a > b) std::swap(a, b);
  return a * n - a * (a - 1) / 2 + (b - a);
}

double delta(int a, int b) { return a == b ? 1.0 : 0.0; }

/// The real coefficient matrix of the Whitney form: row per upper-triangular
/// block entry \f$ ([ij],[kl]) \f$, column per unknown \f$ (C\Gamma)_{ab} \f$.
Eigen::MatrixXd whitneyDesign(int d) {
  const auto edges = localEdges(d);
  const int n = d + 1;
  const int ne = static_cast<int>(edges.size());
  const int unknowns = n * (n + 1) / 2;
  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(ne * (ne + 1) / 2, unknowns);
  int row = 0;
  for (int p = 0; p < ne; ++p)
    for (int q = p; q < ne; ++q, ++row) {
      const auto [i, j] = edges[p];
      const auto [k, l] = edges[q];
      A(row, unknownIndex(j, l, n)) += 1.0 + delta(i, k);
      A(row, unknownIndex(j, k, n)) -= 1.0 + delta(i, l);
      A(row, unknownIndex(i, l, n)) -= 1.0 + delta(j, k);
      A(row, unknownIndex(i, k, n)) += 1.0 + delta(j, l);
    }
  return A;
}

int dimensionOfEdgeCount(Eigen::Index ne) {
  for (int d = 1; d <= 16; ++d)
    if (d * (d + 1) / 2 == ne) return d;
  return -1;
}

std::vector<Complex> lengthsFromMetric(const Eigen::MatrixXcd &g) {
  const int d = static_cast<int>(g.rows());
  std::vector<Complex> z;
  for (int i = 0; i <= d; ++i)
    for (int j = i + 1; j <= d; ++j) {
      if (i == 0)
        z.push_back(g(j - 1, j - 1));
      else
        z.push_back(g(i - 1, i - 1) + g(j - 1, j - 1) - 2.0 * g(i - 1, j - 1));
    }
  return z;
}

}  // namespace

Eigen::MatrixXcd GrownCellRule::whitneyBlock(const Eigen::MatrixXcd &scaledGradientGram) {
  const int n = static_cast<int>(scaledGradientGram.rows());
  if (n < 2 || scaledGradientGram.cols() != n)
    throw std::invalid_argument(
        "GrownCellRule::whitneyBlock: C*Gamma must be square of size d+1 >= 2");
  const int d = n - 1;
  const auto edges = localEdges(d);
  const int ne = static_cast<int>(edges.size());
  const Eigen::MatrixXcd G = 0.5 * (scaledGradientGram + scaledGradientGram.transpose());
  Eigen::MatrixXcd M(ne, ne);
  for (int p = 0; p < ne; ++p)
    for (int q = 0; q < ne; ++q) {
      const auto [i, j] = edges[p];
      const auto [k, l] = edges[q];
      M(p, q) = (1.0 + delta(i, k)) * G(j, l) - (1.0 + delta(i, l)) * G(j, k) -
                (1.0 + delta(j, k)) * G(i, l) + (1.0 + delta(j, l)) * G(i, k);
    }
  return M;
}

WhitneyLengthInversion GrownCellRule::invertWhitneyBlock(const Eigen::MatrixXcd &block) {
  if (block.rows() != block.cols())
    throw std::invalid_argument("GrownCellRule::invertWhitneyBlock: the block must be square");
  const int d = dimensionOfEdgeCount(block.rows());
  if (d != 2 && d != 3)
    throw std::invalid_argument(
        "GrownCellRule::invertWhitneyBlock: the block has " + std::to_string(block.rows()) +
        " rows; the rule inverts the 3-row block of a triangle (scale undetermined) or the "
        "6-row block of a tetrahedron (C = 14400/det(g/C)); at d >= 4 the scale is a "
        "(d-2)-th root, which the rule does not supply");
  const int n = d + 1;
  const int ne = static_cast<int>(block.rows());

  WhitneyLengthInversion out;
  out.dimension = d;
  const double normB = block.norm();
  out.asymmetry = normB > 0.0 ? (block - block.transpose()).norm() / normB : 0.0;

  const Eigen::MatrixXd A = whitneyDesign(d);
  Eigen::VectorXcd b(A.rows());
  int row = 0;
  for (int p = 0; p < ne; ++p)
    for (int q = p; q < ne; ++q) b(row++) = block(p, q);
  const Eigen::MatrixXcd Ac = A.cast<Complex>();
  const Eigen::VectorXcd x = Ac.colPivHouseholderQr().solve(b);
  out.residual = (Ac * x - b).norm();
  out.relativeResidual = b.norm() > 0.0 ? out.residual / b.norm() : 0.0;

  out.scaledGradientGram = Eigen::MatrixXcd::Zero(n, n);
  for (int a = 0; a < n; ++a)
    for (int c = a; c < n; ++c) {
      out.scaledGradientGram(a, c) = x(unknownIndex(a, c, n));
      out.scaledGradientGram(c, a) = out.scaledGradientGram(a, c);
    }

  const Eigen::MatrixXcd reduced = out.scaledGradientGram.bottomRightCorner(d, d);
  Eigen::FullPivLU<Eigen::MatrixXcd> lu(reduced);
  if (!lu.isInvertible())
    throw std::invalid_argument(
        "GrownCellRule::invertWhitneyBlock: the block of C*Gamma on v_1..v_d is singular");
  out.scaledMetric = lu.inverse();
  out.scaledSquaredLengths = lengthsFromMetric(out.scaledMetric);

  const double nan = std::numeric_limits<double>::quiet_NaN();
  if (d == 3) {
    const double factor = 6.0 * 4.0 * 5.0;  // d!(d+1)(d+2) at d = 3
    out.scale = factor * factor / out.scaledMetric.determinant();
    out.scaleDetermined = true;
    out.volume = 20.0 * out.scale;
    out.squaredLengths = lengthsFromMetric(out.scale * out.scaledMetric);
  } else {
    out.scale = Complex(nan, nan);
    out.volume = Complex(nan, nan);
    out.scaleDetermined = false;
  }
  return out;
}

std::complex<double> GrownCellRule::transportConnection(const Eigen::MatrixXcd &transport) {
  if (transport.rows() == 0 || transport.rows() != transport.cols())
    throw std::invalid_argument(
        "GrownCellRule::transportConnection: the transport must be square and nonempty; only "
        "common-rank links carry a connection");
  return transport.determinant();
}

}  // namespace tessera::chainhodge
