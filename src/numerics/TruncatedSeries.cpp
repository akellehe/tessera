// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "numerics/TruncatedSeries.h"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

#include <Eigen/Dense>

namespace tessera::numerics {

namespace {

using Complex = TruncatedSeries::Complex;
using Coefficients = std::vector<Complex>;

const Complex kZero{0.0, 0.0};
const Complex kOne{1.0, 0.0};

/// Throws `std::invalid_argument` unless \p order is an order a series
/// declares, 0 to `TruncatedSeries::kMaximumOrder`. \p what names the caller.
void requireOrder(int order, const std::string &what) {
  if (order < 0 || order > TruncatedSeries::kMaximumOrder)
    throw std::invalid_argument(
        what + ": the order " + std::to_string(order) +
        " is outside the orders a truncated series declares, 0 to " +
        std::to_string(TruncatedSeries::kMaximumOrder));
}

/// Throws `std::invalid_argument` unless the two operands of \p operation
/// have one order.
void requireSameOrder(int left, int right, const std::string &operation) {
  if (left != right)
    throw std::invalid_argument(
        "TruncatedSeries " + operation + ": the operands have orders " +
        std::to_string(left) + " and " + std::to_string(right) +
        "; a binary operation takes two series of one order");
}

/// Throws the `std::domain_error` of an operation that has no value at its
/// argument, naming the operation.
[[noreturn]] void noValue(const std::string &operation,
                          const std::string &reason) {
  throw std::domain_error("TruncatedSeries " + operation + ": " + reason);
}

/// The first \p count coefficients of the Cauchy product of \p f and \p g,
/// \f$ \sum_{j=0}^{k} f_j g_{k-j} \f$; both hold at least \p count
/// coefficients.
Coefficients product(const Coefficients &f, const Coefficients &g,
                     std::size_t count) {
  Coefficients out(count, kZero);
  for (std::size_t k = 0; k < count; ++k) {
    Complex sum = kZero;
    for (std::size_t j = 0; j <= k; ++j) sum += f[j] * g[k - j];
    out[k] = sum;
  }
  return out;
}

/// The first \p count coefficients of the quotient \f$ h = f / g \f$, from
/// \f$ g_0 h_k = f_k - \sum_{j=1}^{k} g_j h_{k-j} \f$; both operands hold at
/// least \p count coefficients and \f$ g_0 \f$ is nonzero.
Coefficients quotient(const Coefficients &f, const Coefficients &g,
                      std::size_t count) {
  Coefficients out(count, kZero);
  for (std::size_t k = 0; k < count; ++k) {
    Complex sum = f[k];
    for (std::size_t j = 1; j <= k; ++j) sum -= g[j] * out[k - j];
    out[k] = sum / g[0];
  }
  return out;
}

/// The coefficients of the square root \f$ g \f$ of \p f whose order-0
/// coefficient is \p root, from
/// \f$ 2 g_0 g_k = f_k - \sum_{j=1}^{k-1} g_j g_{k-j} \f$; \p root is
/// nonzero.
Coefficients rootCoefficients(const Coefficients &f, Complex root) {
  Coefficients out(f.size(), kZero);
  out[0] = root;
  for (std::size_t k = 1; k < f.size(); ++k) {
    Complex sum = f[k];
    for (std::size_t j = 1; j < k; ++j) sum -= out[j] * out[k - j];
    out[k] = sum / (2.0 * root);
  }
  return out;
}

/// The coefficients of the logarithm \f$ g \f$ of \p f whose order-0
/// coefficient is \p value, from
/// \f$ k f_0 g_k = k f_k - \sum_{j=1}^{k-1} j g_j f_{k-j} \f$; \f$ f_0 \f$ is
/// nonzero.
Coefficients logCoefficients(const Coefficients &f, Complex value) {
  Coefficients out(f.size(), kZero);
  out[0] = value;
  for (std::size_t k = 1; k < f.size(); ++k) {
    Complex sum = static_cast<double>(k) * f[k];
    for (std::size_t j = 1; j < k; ++j)
      sum -= static_cast<double>(j) * out[j] * f[k - j];
    out[k] = sum / (static_cast<double>(k) * f[0]);
  }
  return out;
}

/// The solution \f$ X \f$ of the Sylvester equation
/// \f$ T_1 X - X T_2 = C \f$ with \f$ T_1 \f$ and \f$ T_2 \f$ upper
/// triangular and without a common diagonal entry. Column \f$ j \f$ of the
/// equation reads
/// \f$ (T_1 - (T_2)_{jj} I) x_j = c_j + \sum_{i<j} (T_2)_{ij} x_i \f$, a
/// triangular system for \f$ x_j \f$ once the earlier columns are known.
Eigen::MatrixXcd solveTriangularSylvester(const Eigen::MatrixXcd &T1,
                                          const Eigen::MatrixXcd &T2,
                                          const Eigen::MatrixXcd &C) {
  const Eigen::Index rows = C.rows();
  const Eigen::Index cols = C.cols();
  Eigen::MatrixXcd X(rows, cols);
  for (Eigen::Index j = 0; j < cols; ++j) {
    Eigen::VectorXcd rhs = C.col(j);
    for (Eigen::Index i = 0; i < j; ++i) rhs += T2(i, j) * X.col(i);
    const Eigen::MatrixXcd shifted =
        T1 - T2(j, j) * Eigen::MatrixXcd::Identity(rows, rows);
    X.col(j) = shifted.triangularView<Eigen::Upper>().solve(rhs);
  }
  return X;
}

/// One adjacent swap of a complex Schur form \f$ Q T Q^H \f$: the unitary
/// rotation \f$ G \f$ on positions \f$ k, k+1 \f$ that exchanges the two
/// diagonal entries, applied as \f$ T \leftarrow G^H T G \f$ and
/// \f$ Q \leftarrow Q G \f$, so that \f$ Q T Q^H \f$ is unchanged and
/// \f$ T \f$ stays upper triangular. The first column of \f$ G \f$ is the
/// eigenvector of the \f$ 2 \times 2 \f$ block for the eigenvalue that moves
/// up. The two diagonal entries are written back as the exchanged values
/// themselves, so the diagonal of \f$ T \f$ is permuted exactly. The two
/// entries differ (the caller exchanges an eigenvalue of the group with one
/// outside it).
void swapAdjacent(Eigen::MatrixXcd &T, Eigen::MatrixXcd &Q, Eigen::Index k) {
  const Complex a = T(k, k), b = T(k, k + 1), c = T(k + 1, k + 1);
  Eigen::Matrix2cd G;
  if (b == kZero) {
    G << kZero, kOne, kOne, kZero;
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
  T(k + 1, k) = kZero;
  T(k, k) = c;
  T(k + 1, k + 1) = a;
}

/// Renders a complex number for an error message.
std::string describe(Complex value) {
  return "(" + std::to_string(value.real()) + ", " +
         std::to_string(value.imag()) + ")";
}

}  // namespace

// --------------------------------------------------------------------------
// TruncatedSeries
// --------------------------------------------------------------------------

TruncatedSeries::TruncatedSeries(int order) {
  requireOrder(order, "TruncatedSeries");
  coefficients_.assign(static_cast<std::size_t>(order) + 1, kZero);
}

TruncatedSeries::TruncatedSeries(std::vector<Complex> coefficients)
    : coefficients_(std::move(coefficients)) {
  if (coefficients_.empty())
    throw std::invalid_argument(
        "TruncatedSeries: the coefficient list is empty; a series carries at "
        "least its order-0 coefficient");
  if (coefficients_.size() >
      static_cast<std::size_t>(TruncatedSeries::kMaximumOrder) + 1)
    throw std::invalid_argument(
        "TruncatedSeries: " + std::to_string(coefficients_.size()) +
        " coefficients declare the order " +
        std::to_string(coefficients_.size() - 1) +
        ", which is above the largest order a truncated series declares, " +
        std::to_string(TruncatedSeries::kMaximumOrder));
}

TruncatedSeries TruncatedSeries::constant(int order, Complex value) {
  TruncatedSeries out(order);
  out.coefficients_[0] = value;
  return out;
}

TruncatedSeries TruncatedSeries::variable(int order, Complex point,
                                          Complex direction) {
  TruncatedSeries out(order);
  out.coefficients_[0] = point;
  if (order >= 1) out.coefficients_[1] = direction;
  return out;
}

TruncatedSeries::Complex TruncatedSeries::coefficient(int k) const {
  if (k < 0 || k > order())
    throw std::out_of_range(
        "TruncatedSeries::coefficient: the index " + std::to_string(k) +
        " is outside the coefficients 0 to " + std::to_string(order()) +
        " of this series");
  return coefficients_[static_cast<std::size_t>(k)];
}

TruncatedSeries::Complex TruncatedSeries::evaluate(Complex t) const noexcept {
  Complex value = coefficients_.back();
  for (std::size_t k = coefficients_.size() - 1; k-- > 0;)
    value = value * t + coefficients_[k];
  return value;
}

TruncatedSeries TruncatedSeries::truncated(int order) const {
  if (order < 0 || order > this->order())
    throw std::invalid_argument(
        "TruncatedSeries::truncated: the order " + std::to_string(order) +
        " is outside 0 to " + std::to_string(this->order()) +
        ", the orders a series of order " + std::to_string(this->order()) +
        " can be cut to");
  return TruncatedSeries(Coefficients(
      coefficients_.begin(),
      coefficients_.begin() + static_cast<std::ptrdiff_t>(order) + 1));
}

TruncatedSeries TruncatedSeries::derivative() const {
  if (order() == 0)
    noValue("derivative",
            "a series of order 0 determines no coefficient of its derivative");
  Coefficients out(coefficients_.size() - 1, kZero);
  for (std::size_t k = 0; k < out.size(); ++k)
    out[k] = static_cast<double>(k + 1) * coefficients_[k + 1];
  return TruncatedSeries(std::move(out));
}

TruncatedSeries TruncatedSeries::integral(Complex constant) const {
  const std::size_t count =
      order() < kMaximumOrder ? coefficients_.size() + 1 : coefficients_.size();
  Coefficients out(count, kZero);
  out[0] = constant;
  for (std::size_t k = 1; k < count; ++k)
    out[k] = coefficients_[k - 1] / static_cast<double>(k);
  return TruncatedSeries(std::move(out));
}

TruncatedSeries TruncatedSeries::operator-() const {
  TruncatedSeries out(*this);
  for (Complex &c : out.coefficients_) c = -c;
  return out;
}

TruncatedSeries &TruncatedSeries::operator+=(const TruncatedSeries &other) {
  requireSameOrder(order(), other.order(), "sum");
  for (std::size_t k = 0; k < coefficients_.size(); ++k)
    coefficients_[k] += other.coefficients_[k];
  return *this;
}

TruncatedSeries &TruncatedSeries::operator-=(const TruncatedSeries &other) {
  requireSameOrder(order(), other.order(), "difference");
  for (std::size_t k = 0; k < coefficients_.size(); ++k)
    coefficients_[k] -= other.coefficients_[k];
  return *this;
}

TruncatedSeries &TruncatedSeries::operator*=(const TruncatedSeries &other) {
  requireSameOrder(order(), other.order(), "product");
  coefficients_ =
      product(coefficients_, other.coefficients_, coefficients_.size());
  return *this;
}

TruncatedSeries &TruncatedSeries::operator/=(const TruncatedSeries &other) {
  requireSameOrder(order(), other.order(), "division");
  if (other.coefficients_[0] == kZero)
    noValue("division",
            "the order-0 coefficient of the divisor is exactly zero, so the "
            "quotient has no power series");
  coefficients_ =
      quotient(coefficients_, other.coefficients_, coefficients_.size());
  return *this;
}

TruncatedSeries &TruncatedSeries::operator+=(Complex value) {
  coefficients_[0] += value;
  return *this;
}

TruncatedSeries &TruncatedSeries::operator-=(Complex value) {
  coefficients_[0] -= value;
  return *this;
}

TruncatedSeries &TruncatedSeries::operator*=(Complex value) {
  for (Complex &c : coefficients_) c *= value;
  return *this;
}

TruncatedSeries &TruncatedSeries::operator/=(Complex value) {
  if (value == kZero)
    noValue("division", "the divisor is the number zero exactly");
  for (Complex &c : coefficients_) c /= value;
  return *this;
}

TruncatedSeries operator+(TruncatedSeries f, const TruncatedSeries &g) {
  f += g;
  return f;
}

TruncatedSeries operator-(TruncatedSeries f, const TruncatedSeries &g) {
  f -= g;
  return f;
}

TruncatedSeries operator*(TruncatedSeries f, const TruncatedSeries &g) {
  f *= g;
  return f;
}

TruncatedSeries operator/(TruncatedSeries f, const TruncatedSeries &g) {
  f /= g;
  return f;
}

TruncatedSeries operator+(TruncatedSeries f, Complex value) {
  f += value;
  return f;
}

TruncatedSeries operator-(TruncatedSeries f, Complex value) {
  f -= value;
  return f;
}

TruncatedSeries operator*(TruncatedSeries f, Complex value) {
  f *= value;
  return f;
}

TruncatedSeries operator/(TruncatedSeries f, Complex value) {
  f /= value;
  return f;
}

TruncatedSeries operator+(Complex value, TruncatedSeries f) {
  f += value;
  return f;
}

TruncatedSeries operator-(Complex value, const TruncatedSeries &f) {
  TruncatedSeries out = -f;
  out += value;
  return out;
}

TruncatedSeries operator*(Complex value, TruncatedSeries f) {
  f *= value;
  return f;
}

TruncatedSeries operator/(Complex value, const TruncatedSeries &f) {
  TruncatedSeries out = TruncatedSeries::constant(f.order(), value);
  out /= f;
  return out;
}

// --------------------------------------------------------------------------
// Elementary functions
// --------------------------------------------------------------------------

TruncatedSeries pow(const TruncatedSeries &f, int exponent) {
  TruncatedSeries base = f;
  long long remaining = exponent;
  if (remaining < 0) {
    if (f.coefficients().front() == kZero)
      noValue("pow",
              "a negative power of a series whose order-0 coefficient is "
              "exactly zero has no power series");
    base = TruncatedSeries::constant(f.order(), kOne) / f;
    remaining = -remaining;
  }
  TruncatedSeries out = TruncatedSeries::constant(f.order(), kOne);
  while (remaining > 0) {
    if ((remaining & 1) != 0) out *= base;
    remaining >>= 1;
    if (remaining > 0) base *= base;
  }
  return out;
}

TruncatedSeries sqrt(const TruncatedSeries &f, std::optional<Complex> branch) {
  const Coefficients &c = f.coefficients();
  if (c[0] == kZero)
    noValue("sqrt",
            "the order-0 coefficient is exactly zero, the branch point of the "
            "square root, where it has no power series");
  const Complex root = branch ? *branch : std::sqrt(c[0]);
  if (root == kZero)
    noValue("sqrt",
            "the given branch value is exactly zero, which is the root of no "
            "nonzero order-0 coefficient");
  return TruncatedSeries(rootCoefficients(c, root));
}

TruncatedSeries exp(const TruncatedSeries &f) {
  const Coefficients &c = f.coefficients();
  Coefficients out(c.size(), kZero);
  out[0] = std::exp(c[0]);
  for (std::size_t k = 1; k < c.size(); ++k) {
    Complex sum = kZero;
    for (std::size_t j = 1; j <= k; ++j)
      sum += static_cast<double>(j) * c[j] * out[k - j];
    out[k] = sum / static_cast<double>(k);
  }
  return TruncatedSeries(std::move(out));
}

TruncatedSeries log(const TruncatedSeries &f, std::optional<Complex> branch) {
  const Coefficients &c = f.coefficients();
  if (c[0] == kZero)
    noValue("log",
            "the order-0 coefficient is exactly zero, the branch point of the "
            "logarithm, where it has no power series");
  return TruncatedSeries(logCoefficients(c, branch ? *branch : std::log(c[0])));
}

std::pair<TruncatedSeries, TruncatedSeries> sinCos(const TruncatedSeries &f) {
  const Coefficients &c = f.coefficients();
  Coefficients sine(c.size(), kZero), cosine(c.size(), kZero);
  sine[0] = std::sin(c[0]);
  cosine[0] = std::cos(c[0]);
  for (std::size_t k = 1; k < c.size(); ++k) {
    Complex sineSum = kZero, cosineSum = kZero;
    for (std::size_t j = 1; j <= k; ++j) {
      const Complex weighted = static_cast<double>(j) * c[j];
      sineSum += weighted * cosine[k - j];
      cosineSum -= weighted * sine[k - j];
    }
    sine[k] = sineSum / static_cast<double>(k);
    cosine[k] = cosineSum / static_cast<double>(k);
  }
  return {TruncatedSeries(std::move(sine)), TruncatedSeries(std::move(cosine))};
}

TruncatedSeries sin(const TruncatedSeries &f) { return sinCos(f).first; }

TruncatedSeries cos(const TruncatedSeries &f) { return sinCos(f).second; }

TruncatedSeries atan2(const TruncatedSeries &y, const TruncatedSeries &x,
                      std::optional<Complex> branch) {
  requireSameOrder(y.order(), x.order(), "atan2");
  const Coefficients &yc = y.coefficients();
  const Coefficients &xc = x.coefficients();
  // x + i y and x - i y, formed componentwise so that each is exact up to
  // the one rounding of its two sums.
  Coefficients plus(xc.size()), minus(xc.size());
  for (std::size_t k = 0; k < xc.size(); ++k) {
    plus[k] = Complex(xc[k].real() - yc[k].imag(), xc[k].imag() + yc[k].real());
    minus[k] =
        Complex(xc[k].real() + yc[k].imag(), xc[k].imag() - yc[k].real());
  }
  if (plus[0] == kZero || minus[0] == kZero)
    noValue("atan2",
            "x + i y or x - i y is exactly zero at order 0 (for real "
            "arguments, the point is the origin), where the angle has no "
            "power series");
  Complex value;
  if (branch) {
    value = *branch;
  } else if (xc[0].imag() == 0.0 && yc[0].imag() == 0.0) {
    value = Complex(std::atan2(yc[0].real(), xc[0].real()), 0.0);
  } else {
    const Complex difference = std::log(plus[0]) - std::log(minus[0]);
    value = Complex(0.5 * difference.imag(), -0.5 * difference.real());
  }
  const Coefficients u = logCoefficients(plus, kZero);
  const Coefficients v = logCoefficients(minus, kZero);
  Coefficients out(xc.size(), kZero);
  out[0] = value;
  for (std::size_t k = 1; k < xc.size(); ++k) {
    // (u_k - v_k) / (2i)
    const Complex difference = u[k] - v[k];
    out[k] = Complex(0.5 * difference.imag(), -0.5 * difference.real());
  }
  return TruncatedSeries(std::move(out));
}

TruncatedSeries acos(const TruncatedSeries &f, std::optional<Complex> branch) {
  const Coefficients &c = f.coefficients();
  if (c[0] == kOne || c[0] == -kOne)
    noValue("acos",
            "the order-0 coefficient is exactly +1 or -1, a branch point of "
            "the inverse cosine, where it has no power series");
  const Complex angle = branch ? *branch : std::acos(c[0]);
  // The sine s of the angle is a square root of 1 - f^2; the angle picks the
  // root of the order-0 coefficient (1 - f_0)(1 + f_0).
  Coefficients radicand = product(c, c, c.size());
  for (Complex &r : radicand) r = -r;
  radicand[0] = (kOne - c[0]) * (kOne + c[0]);
  Complex root = std::sqrt(radicand[0]);
  if ((root * std::conj(std::sin(angle))).real() < 0.0) root = -root;
  const Coefficients sine = rootCoefficients(radicand, root);
  // theta' = -f' / s, to the order p - 1 that f' is known to.
  const std::size_t count = c.size() - 1;
  Coefficients numerator(count, kZero);
  for (std::size_t k = 0; k < count; ++k)
    numerator[k] = -static_cast<double>(k + 1) * c[k + 1];
  const Coefficients rate = quotient(numerator, sine, count);
  Coefficients out(c.size(), kZero);
  out[0] = angle;
  for (std::size_t k = 1; k < c.size(); ++k)
    out[k] = rate[k - 1] / static_cast<double>(k);
  return TruncatedSeries(std::move(out));
}

TruncatedSeries composePolynomial(const std::vector<Complex> &polynomial,
                                  const TruncatedSeries &f) {
  TruncatedSeries out(f.order());
  for (std::size_t j = polynomial.size(); j-- > 0;) {
    out *= f;
    out += polynomial[j];
  }
  return out;
}

// --------------------------------------------------------------------------
// TruncatedSeriesMatrix
// --------------------------------------------------------------------------

namespace {

/// Throws `std::invalid_argument` unless \p matrix is square; \p operation
/// names the caller.
void requireSquare(const TruncatedSeriesMatrix &matrix,
                   const std::string &operation) {
  if (matrix.rows() != matrix.cols())
    throw std::invalid_argument(
        "TruncatedSeriesMatrix::" + operation + ": the matrix is " +
        std::to_string(matrix.rows()) + " by " + std::to_string(matrix.cols()) +
        "; the operation is defined for a square matrix");
}

/// Throws `std::out_of_range` unless (\p row, \p col) is an entry of
/// \p matrix.
void requireEntry(const TruncatedSeriesMatrix &matrix, Eigen::Index row,
                  Eigen::Index col, const std::string &operation) {
  if (row < 0 || row >= matrix.rows() || col < 0 || col >= matrix.cols())
    throw std::out_of_range(
        "TruncatedSeriesMatrix::" + operation + ": the entry (" +
        std::to_string(row) + ", " + std::to_string(col) +
        ") is outside the " + std::to_string(matrix.rows()) + " by " +
        std::to_string(matrix.cols()) + " matrix");
}

/// Throws `std::invalid_argument` unless the two operands of \p operation
/// have one order.
void requireSameMatrixOrder(const TruncatedSeriesMatrix &left,
                            const TruncatedSeriesMatrix &right,
                            const std::string &operation) {
  if (left.order() != right.order())
    throw std::invalid_argument(
        "TruncatedSeriesMatrix " + operation + ": the operands have orders " +
        std::to_string(left.order()) + " and " + std::to_string(right.order()) +
        "; a binary operation takes two matrix series of one order");
}

/// The Cauchy product with matrix coefficients,
/// \f$ \sum_{j=0}^{k} A_j B_{k-j} \f$ for every order \f$ k \f$ of the
/// operands.
std::vector<Eigen::MatrixXcd> matrixProduct(
    const std::vector<Eigen::MatrixXcd> &A,
    const std::vector<Eigen::MatrixXcd> &B) {
  std::vector<Eigen::MatrixXcd> out;
  out.reserve(A.size());
  for (std::size_t k = 0; k < A.size(); ++k) {
    Eigen::MatrixXcd sum =
        Eigen::MatrixXcd::Zero(A.front().rows(), B.front().cols());
    for (std::size_t j = 0; j <= k; ++j) sum.noalias() += A[j] * B[k - j];
    out.push_back(std::move(sum));
  }
  return out;
}

}  // namespace

TruncatedSeriesMatrix::TruncatedSeriesMatrix(int order, Eigen::Index rows,
                                             Eigen::Index cols) {
  requireOrder(order, "TruncatedSeriesMatrix");
  if (rows < 0 || cols < 0)
    throw std::invalid_argument(
        "TruncatedSeriesMatrix: the shape " + std::to_string(rows) + " by " +
        std::to_string(cols) + " has a negative dimension");
  coefficients_.assign(static_cast<std::size_t>(order) + 1,
                       Eigen::MatrixXcd::Zero(rows, cols));
}

TruncatedSeriesMatrix::TruncatedSeriesMatrix(
    std::vector<Eigen::MatrixXcd> coefficients)
    : coefficients_(std::move(coefficients)) {
  if (coefficients_.empty())
    throw std::invalid_argument(
        "TruncatedSeriesMatrix: the coefficient list is empty; a matrix "
        "series carries at least its order-0 coefficient");
  if (coefficients_.size() >
      static_cast<std::size_t>(TruncatedSeries::kMaximumOrder) + 1)
    throw std::invalid_argument(
        "TruncatedSeriesMatrix: " + std::to_string(coefficients_.size()) +
        " coefficients declare the order " +
        std::to_string(coefficients_.size() - 1) +
        ", which is above the largest order a truncated series declares, " +
        std::to_string(TruncatedSeries::kMaximumOrder));
  for (std::size_t k = 1; k < coefficients_.size(); ++k)
    if (coefficients_[k].rows() != coefficients_.front().rows() ||
        coefficients_[k].cols() != coefficients_.front().cols())
      throw std::invalid_argument(
          "TruncatedSeriesMatrix: the coefficient of order " +
          std::to_string(k) + " is " + std::to_string(coefficients_[k].rows()) +
          " by " + std::to_string(coefficients_[k].cols()) +
          " and the coefficient of order 0 is " +
          std::to_string(coefficients_.front().rows()) + " by " +
          std::to_string(coefficients_.front().cols()) +
          "; the coefficients of a matrix series have one shape");
}

TruncatedSeriesMatrix TruncatedSeriesMatrix::identity(int order,
                                                      Eigen::Index size) {
  TruncatedSeriesMatrix out(order, size, size);
  out.coefficients_.front().setIdentity();
  return out;
}

TruncatedSeries TruncatedSeriesMatrix::entry(Eigen::Index row,
                                             Eigen::Index col) const {
  requireEntry(*this, row, col, "entry");
  std::vector<Complex> out;
  out.reserve(coefficients_.size());
  for (const Eigen::MatrixXcd &A : coefficients_) out.push_back(A(row, col));
  return TruncatedSeries(std::move(out));
}

void TruncatedSeriesMatrix::setEntry(Eigen::Index row, Eigen::Index col,
                                     const TruncatedSeries &series) {
  requireEntry(*this, row, col, "setEntry");
  if (series.order() != order())
    throw std::invalid_argument(
        "TruncatedSeriesMatrix::setEntry: the series has order " +
        std::to_string(series.order()) + " and the matrix has order " +
        std::to_string(order()) +
        "; an entry carries the order of its matrix");
  for (std::size_t k = 0; k < coefficients_.size(); ++k)
    coefficients_[k](row, col) = series.coefficients()[k];
}

Eigen::MatrixXcd TruncatedSeriesMatrix::evaluate(Complex t) const {
  Eigen::MatrixXcd value = coefficients_.back();
  for (std::size_t k = coefficients_.size() - 1; k-- > 0;)
    value = value * t + coefficients_[k];
  return value;
}

TruncatedSeriesMatrix TruncatedSeriesMatrix::operator+(
    const TruncatedSeriesMatrix &other) const {
  requireSameMatrixOrder(*this, other, "sum");
  if (rows() != other.rows() || cols() != other.cols())
    throw std::invalid_argument(
        "TruncatedSeriesMatrix sum: the operands are " +
        std::to_string(rows()) + " by " + std::to_string(cols()) + " and " +
        std::to_string(other.rows()) + " by " + std::to_string(other.cols()) +
        "; a sum takes two matrices of one shape");
  TruncatedSeriesMatrix out(*this);
  for (std::size_t k = 0; k < coefficients_.size(); ++k)
    out.coefficients_[k] += other.coefficients_[k];
  return out;
}

TruncatedSeriesMatrix TruncatedSeriesMatrix::operator-(
    const TruncatedSeriesMatrix &other) const {
  requireSameMatrixOrder(*this, other, "difference");
  if (rows() != other.rows() || cols() != other.cols())
    throw std::invalid_argument(
        "TruncatedSeriesMatrix difference: the operands are " +
        std::to_string(rows()) + " by " + std::to_string(cols()) + " and " +
        std::to_string(other.rows()) + " by " + std::to_string(other.cols()) +
        "; a difference takes two matrices of one shape");
  TruncatedSeriesMatrix out(*this);
  for (std::size_t k = 0; k < coefficients_.size(); ++k)
    out.coefficients_[k] -= other.coefficients_[k];
  return out;
}

TruncatedSeriesMatrix TruncatedSeriesMatrix::operator*(
    const TruncatedSeriesMatrix &other) const {
  requireSameMatrixOrder(*this, other, "product");
  if (cols() != other.rows())
    throw std::invalid_argument(
        "TruncatedSeriesMatrix product: the operands are " +
        std::to_string(rows()) + " by " + std::to_string(cols()) + " and " +
        std::to_string(other.rows()) + " by " + std::to_string(other.cols()) +
        "; a product takes a left operand with as many columns as the right "
        "operand has rows");
  return TruncatedSeriesMatrix(
      matrixProduct(coefficients_, other.coefficients_));
}

TruncatedSeriesMatrix TruncatedSeriesMatrix::inverse() const {
  requireSquare(*this, "inverse");
  const Eigen::Index n = rows();
  if (n == 0) return *this;
  const Eigen::PartialPivLU<Eigen::MatrixXcd> lu(coefficients_.front());
  for (Eigen::Index i = 0; i < n; ++i)
    if (lu.matrixLU()(i, i) == kZero)
      throw std::domain_error(
          "TruncatedSeriesMatrix::inverse: the elimination of the order-0 "
          "coefficient meets a pivot that is exactly zero (pivot " +
          std::to_string(i) +
          "), so the order-0 coefficient is singular and the inverse has no "
          "power series");
  std::vector<Eigen::MatrixXcd> out;
  out.reserve(coefficients_.size());
  out.push_back(lu.solve(Eigen::MatrixXcd::Identity(n, n)));
  for (std::size_t k = 1; k < coefficients_.size(); ++k) {
    Eigen::MatrixXcd rhs = Eigen::MatrixXcd::Zero(n, n);
    for (std::size_t j = 1; j <= k; ++j)
      rhs.noalias() -= coefficients_[j] * out[k - j];
    out.push_back(lu.solve(rhs));
  }
  return TruncatedSeriesMatrix(std::move(out));
}

TruncatedSeries TruncatedSeriesMatrix::determinant() const {
  requireSquare(*this, "determinant");
  const Eigen::Index n = rows();
  if (n == 0) return TruncatedSeries::constant(order(), kOne);
  std::vector<Eigen::MatrixXcd> X = coefficients_;
  for (Eigen::Index pass = 1; pass < n; ++pass) {
    // mu is linear, so it acts on each coefficient matrix separately.
    std::vector<Eigen::MatrixXcd> mu;
    mu.reserve(X.size());
    for (const Eigen::MatrixXcd &Xk : X) {
      Eigen::MatrixXcd M = Xk.triangularView<Eigen::StrictlyUpper>();
      Complex tail = kZero;
      for (Eigen::Index i = n; i-- > 0;) {
        M(i, i) = kZero - tail;
        tail += Xk(i, i);
      }
      mu.push_back(std::move(M));
    }
    X = matrixProduct(mu, coefficients_);
  }
  const double sign = (n - 1) % 2 == 0 ? 1.0 : -1.0;
  std::vector<Complex> out;
  out.reserve(X.size());
  for (const Eigen::MatrixXcd &Xk : X) out.push_back(sign * Xk(0, 0));
  return TruncatedSeries(std::move(out));
}

TruncatedSeriesMatrix TruncatedSeriesMatrix::rieszProjector(
    const std::function<bool(Complex)> &inGroup) const {
  requireSquare(*this, "rieszProjector");
  const Eigen::Index n = rows();
  const int p = order();
  if (n == 0) return *this;

  const Eigen::ComplexSchur<Eigen::MatrixXcd> schur(coefficients_.front(),
                                                    true);
  if (schur.info() != Eigen::Success)
    throw std::runtime_error(
        "TruncatedSeriesMatrix::rieszProjector: the QR iteration of the "
        "complex Schur form of the order-0 coefficient did not converge");
  Eigen::MatrixXcd T = schur.matrixT();
  Eigen::MatrixXcd Q = schur.matrixU();

  std::vector<bool> selected(static_cast<std::size_t>(n));
  Eigen::Index q = 0;
  for (Eigen::Index i = 0; i < n; ++i) {
    selected[static_cast<std::size_t>(i)] = inGroup(T(i, i));
    if (selected[static_cast<std::size_t>(i)]) ++q;
  }
  if (q == 0) return TruncatedSeriesMatrix(p, n, n);
  if (q == n) return identity(p, n);
  for (Eigen::Index a = 0; a < n; ++a) {
    if (!selected[static_cast<std::size_t>(a)]) continue;
    for (Eigen::Index b = 0; b < n; ++b)
      if (!selected[static_cast<std::size_t>(b)] && T(a, a) == T(b, b))
        throw std::domain_error(
            "TruncatedSeriesMatrix::rieszProjector: the eigenvalue " +
            describe(T(a, a)) +
            " of the order-0 coefficient is both inside the group and "
            "outside it, exactly; the difference of the two has no quotient "
            "and the projector of a part of a multiple eigenvalue is not "
            "defined");
  }

  // Reorder the Schur form so that the group leads. Positions are scanned
  // upward; every eigenvalue of the group met so far already sits in the
  // leading block, so the positions between that block and the current one
  // all hold eigenvalues outside the group, and the flags of the positions
  // still to be scanned are untouched by the swaps.
  Eigen::Index lead = 0;
  for (Eigen::Index position = 0; position < n; ++position) {
    if (!selected[static_cast<std::size_t>(position)]) continue;
    for (Eigen::Index k = position; k > lead; --k) swapAdjacent(T, Q, k - 1);
    ++lead;
  }
  const Eigen::Index m = n - q;
  const Eigen::MatrixXcd T11 = T.topLeftCorner(q, q);
  const Eigen::MatrixXcd T12 = T.topRightCorner(q, m);
  const Eigen::MatrixXcd T22 = T.bottomRightCorner(m, m);

  // The adapted basis V = Q [[I, -Y], [0, I]] with T11 Y - Y T22 = T12
  // block-diagonalizes the order-0 coefficient: V^{-1} A_0 V = diag(T11, T22),
  // and V^{-1} = [[I, Y], [0, I]] Q^H.
  const Eigen::MatrixXcd Y = solveTriangularSylvester(T11, T22, T12);
  Eigen::MatrixXcd V = Q;
  V.rightCols(m).noalias() -= Q.leftCols(q) * Y;
  Eigen::MatrixXcd Vinverse = Q.adjoint();
  Vinverse.topRows(q).noalias() += Y * Q.adjoint().bottomRows(m);

  // The coefficients of A of order 1 and above in the adapted basis; the
  // order-0 coefficient there is diag(T11, T22) and enters through the
  // Sylvester equations only.
  std::vector<Eigen::MatrixXcd> adapted(coefficients_.size());
  for (std::size_t j = 1; j < coefficients_.size(); ++j)
    adapted[j] = Vinverse * coefficients_[j] * V;

  std::vector<Eigen::MatrixXcd> projector(coefficients_.size(),
                                          Eigen::MatrixXcd::Zero(n, n));
  projector[0].topLeftCorner(q, q).setIdentity();
  for (std::size_t k = 1; k < coefficients_.size(); ++k) {
    Eigen::MatrixXcd S = Eigen::MatrixXcd::Zero(n, n);
    for (std::size_t j = 1; j < k; ++j)
      S.noalias() += projector[j] * projector[k - j];
    Eigen::MatrixXcd R = Eigen::MatrixXcd::Zero(n, n);
    for (std::size_t j = 1; j <= k; ++j) {
      R.noalias() -= adapted[j] * projector[k - j];
      R.noalias() += projector[k - j] * adapted[j];
    }
    projector[k].topLeftCorner(q, q) = -S.topLeftCorner(q, q);
    projector[k].bottomRightCorner(m, m) = S.bottomRightCorner(m, m);
    projector[k].topRightCorner(q, m) =
        solveTriangularSylvester(T11, T22, R.topRightCorner(q, m));
    projector[k].bottomLeftCorner(m, q) =
        solveTriangularSylvester(T22, T11, R.bottomLeftCorner(m, q));
  }
  for (Eigen::MatrixXcd &Pk : projector) Pk = (V * Pk * Vinverse).eval();
  return TruncatedSeriesMatrix(std::move(projector));
}

// --------------------------------------------------------------------------
// Reversion
// --------------------------------------------------------------------------

Eigen::VectorXcd SeriesReversion::evaluate(std::complex<double> t) const {
  if (coefficients.empty()) return Eigen::VectorXcd::Zero(0);
  Eigen::VectorXcd value = coefficients.back();
  for (std::size_t k = coefficients.size() - 1; k-- > 0;)
    value = value * t + coefficients[k];
  return value;
}

Eigen::VectorXcd SeriesReversion::step() const {
  if (coefficients.empty()) return Eigen::VectorXcd::Zero(0);
  Eigen::VectorXcd value = Eigen::VectorXcd::Zero(coefficients.front().size());
  for (std::size_t k = 1; k < coefficients.size(); ++k)
    value += coefficients[k];
  return value;
}

Eigen::MatrixXcd jacobian(const SeriesMap &map, const Eigen::VectorXcd &point) {
  const Eigen::Index n = point.size();
  Eigen::MatrixXcd out;
  for (Eigen::Index i = 0; i < n; ++i) {
    SeriesVector x;
    x.reserve(static_cast<std::size_t>(n));
    for (Eigen::Index j = 0; j < n; ++j)
      x.push_back(TruncatedSeries::variable(1, point(j),
                                            i == j ? kOne : kZero));
    const SeriesVector value = map(x);
    const auto rows = static_cast<Eigen::Index>(value.size());
    if (i == 0) out = Eigen::MatrixXcd::Zero(rows, n);
    if (rows != out.rows())
      throw std::invalid_argument(
          "jacobian: the map returned " + std::to_string(rows) +
          " components along direction " + std::to_string(i) + " and " +
          std::to_string(out.rows()) +
          " along direction 0; a map has one number of components");
    for (Eigen::Index r = 0; r < rows; ++r) {
      const TruncatedSeries &component = value[static_cast<std::size_t>(r)];
      if (component.order() != 1)
        throw std::invalid_argument(
            "jacobian: the map returned a series of order " +
            std::to_string(component.order()) +
            " for arguments of order 1; a map returns series of the order of "
            "its arguments");
      out(r, i) = component.coefficients()[1];
    }
  }
  return out;
}

SeriesReversion revert(const SeriesMap &map, const Eigen::VectorXcd &point,
                       int order, const LinearSolve &solve) {
  requireOrder(order, "revert");
  const Eigen::Index n = point.size();
  SeriesReversion out;
  out.coefficients.assign(static_cast<std::size_t>(order) + 1,
                          Eigen::VectorXcd::Zero(n));

  // The series of the map on the path known so far, cut after order k: the
  // components of x_0 + s_1 t + ... + s_{k-1} t^{k-1}, as series of order k.
  const auto evaluateAt = [&](int k) {
    SeriesVector x;
    x.reserve(static_cast<std::size_t>(n));
    for (Eigen::Index i = 0; i < n; ++i) {
      std::vector<Complex> c(static_cast<std::size_t>(k) + 1, kZero);
      c[0] = point(i);
      for (int j = 1; j < k; ++j)
        c[static_cast<std::size_t>(j)] =
            out.coefficients[static_cast<std::size_t>(j)](i);
      x.emplace_back(std::move(c));
    }
    SeriesVector value = map(x);
    for (const TruncatedSeries &component : value)
      if (component.order() != k)
        throw std::invalid_argument(
            "revert: the map returned a series of order " +
            std::to_string(component.order()) + " for arguments of order " +
            std::to_string(k) +
            "; a map returns series of the order of its arguments");
    return value;
  };

  if (order == 0) {
    const SeriesVector value = evaluateAt(0);
    out.residual.resize(static_cast<Eigen::Index>(value.size()));
    for (std::size_t r = 0; r < value.size(); ++r)
      out.residual(static_cast<Eigen::Index>(r)) = value[r].coefficients()[0];
    return out;
  }

  for (int k = 1; k <= order; ++k) {
    const SeriesVector value = evaluateAt(k);
    const auto rows = static_cast<Eigen::Index>(value.size());
    if (k == 1) {
      out.residual.resize(rows);
      for (Eigen::Index r = 0; r < rows; ++r)
        out.residual(r) = value[static_cast<std::size_t>(r)].coefficients()[0];
    } else if (rows != out.residual.size()) {
      throw std::invalid_argument(
          "revert: the map returned " + std::to_string(rows) +
          " components at order " + std::to_string(k) + " and " +
          std::to_string(out.residual.size()) +
          " at order 1; a map has one number of components");
    }
    // J s_k = -N_k, less the residual at order 1: the coefficient of t^k of
    // (1 - t) F(x_0) is -F(x_0) at k = 1 and zero above.
    Eigen::VectorXcd rhs(rows);
    for (Eigen::Index r = 0; r < rows; ++r)
      rhs(r) = -value[static_cast<std::size_t>(r)]
                    .coefficients()[static_cast<std::size_t>(k)];
    if (k == 1) rhs -= out.residual;
    Eigen::VectorXcd coefficient = solve(rhs);
    if (coefficient.size() != n)
      throw std::invalid_argument(
          "revert: the linear solve returned a vector of length " +
          std::to_string(coefficient.size()) + " at order " +
          std::to_string(k) + " for a point of length " + std::to_string(n) +
          "; a coefficient of the path has one entry per unknown");
    out.coefficients[static_cast<std::size_t>(k)] = std::move(coefficient);
  }
  return out;
}

}  // namespace tessera::numerics
