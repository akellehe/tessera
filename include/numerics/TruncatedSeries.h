// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_NUMERICS_TRUNCATEDSERIES_H
#define TESSERA_NUMERICS_TRUNCATEDSERIES_H

#include <complex>
#include <functional>
#include <optional>
#include <utility>
#include <vector>

#include <Eigen/Core>

namespace tessera::numerics {

/// # TruncatedSeries
///
/// The Taylor coefficients \f$ c_0, \dots, c_p \f$ of a function
/// \f$ f(t) = \sum_{k=0}^{p} c_k t^k + O(t^{p+1}) \f$ of one real or complex
/// parameter \f$ t \f$, to a declared order \f$ p \f$ between 0 and 10.
///
/// Evaluating a formula on series instead of numbers propagates the Taylor
/// coefficients of its arguments to those of its value exactly: every
/// operation below computes the coefficients of its result from the
/// coefficients of its operands by a finite recurrence that is an identity
/// between power series, so no difference quotient and no contour integral
/// enters. The coefficient of \f$ t^k \f$ of the result is
/// \f$ \frac{1}{k!} \frac{d^k}{dt^k} \f$ of the composed function at
/// \f$ t = 0 \f$, to rounding.
///
/// The order \f$ p \f$ is the only truncation: a result is the exact series
/// of the operation with every power above \f$ t^p \f$ left out, and
/// coefficient \f$ k \le p \f$ of a result depends only on coefficients
/// \f$ 0, \dots, k \f$ of the operands. Two series enter a binary operation
/// only when they have the same order; a number (`Complex`) enters as the
/// constant series of that value.
///
/// Notation used by the recurrences: \f$ f_k \f$ is coefficient \f$ k \f$ of
/// the series \f$ f \f$, and \f$ f' \f$ is the derivative with respect to
/// \f$ t \f$, whose coefficient \f$ k \f$ is \f$ (k+1) f_{k+1} \f$.
class TruncatedSeries {
 public:
  using Complex = std::complex<double>;

  /// The largest order a series can declare.
  static constexpr int kMaximumOrder = 10;

  /// The zero series of order \p order. Throws `std::invalid_argument` when
  /// \p order is negative or above `kMaximumOrder`.
  explicit TruncatedSeries(int order);

  /// The series with the given coefficients \f$ c_0, \dots, c_p \f$; its
  /// order is the number of coefficients less one. Throws
  /// `std::invalid_argument` when the list is empty or holds more than
  /// `kMaximumOrder + 1` coefficients.
  explicit TruncatedSeries(std::vector<Complex> coefficients);

  /// The constant series of value \p value and order \p order.
  [[nodiscard]] static TruncatedSeries constant(int order, Complex value);

  /// The series \f$ f(t) = x_0 + d\,t \f$ of order \p order, with
  /// \f$ x_0 \f$ the value of \p point and \f$ d \f$ that of \p direction:
  /// an independent variable at \f$ x_0 \f$ moving along \f$ d \f$. At
  /// order 0 only the constant term is carried.
  [[nodiscard]] static TruncatedSeries variable(int order, Complex point,
                                                Complex direction = {1.0, 0.0});

  /// The declared order \f$ p \f$.
  [[nodiscard]] int order() const noexcept {
    return static_cast<int>(coefficients_.size()) - 1;
  }

  /// The coefficients \f$ c_0, \dots, c_p \f$.
  [[nodiscard]] const std::vector<Complex> &coefficients() const noexcept {
    return coefficients_;
  }

  /// The coefficient \f$ c_k \f$. Throws `std::out_of_range` when \p k is
  /// negative or above the order.
  [[nodiscard]] Complex coefficient(int k) const;

  /// The value \f$ \sum_{k=0}^{p} c_k t^k \f$ of the truncated series at the
  /// parameter value \p t, by Horner's rule.
  [[nodiscard]] Complex evaluate(Complex t) const noexcept;

  /// The same series with the coefficients above \p order left out. Throws
  /// `std::invalid_argument` when \p order is negative or above the order of
  /// this series.
  [[nodiscard]] TruncatedSeries truncated(int order) const;

  /// The derivative \f$ f' \f$, coefficient \f$ k \f$ of which is
  /// \f$ (k+1) c_{k+1} \f$. A series of order \f$ p \f$ determines its
  /// derivative to order \f$ p - 1 \f$, which is the order of the result.
  /// Throws `std::domain_error` for a series of order 0, which determines no
  /// coefficient of its derivative.
  [[nodiscard]] TruncatedSeries derivative() const;

  /// The integral \f$ a + \int_0^t f \f$ with \f$ a \f$ the value of
  /// \p constant; its coefficient of order 0 is \f$ a \f$ and its coefficient
  /// \f$ k \ge 1 \f$ is \f$ c_{k-1} / k \f$. A series of order \f$ p \f$
  /// determines its integral to order \f$ p + 1 \f$, which is the order of
  /// the result when \f$ p < 10 \f$; the integral of a series of order 10 is
  /// carried to order 10, the largest order a series declares.
  [[nodiscard]] TruncatedSeries integral(Complex constant = {0.0, 0.0}) const;

  /// The negated series.
  [[nodiscard]] TruncatedSeries operator-() const;

  /// Coefficientwise sum. Throws `std::invalid_argument` when the orders
  /// differ, as every binary operation between two series does.
  TruncatedSeries &operator+=(const TruncatedSeries &other);
  /// Coefficientwise difference.
  TruncatedSeries &operator-=(const TruncatedSeries &other);
  /// The Cauchy product: \f$ (fg)_k = \sum_{j=0}^{k} f_j g_{k-j} \f$, the
  /// coefficient of \f$ t^k \f$ when the two sums are multiplied out.
  TruncatedSeries &operator*=(const TruncatedSeries &other);
  /// The quotient \f$ h = f / g \f$. Comparing coefficients of \f$ t^k \f$ in
  /// \f$ f = h g \f$ gives
  /// \f$ g_0 h_k = f_k - \sum_{j=1}^{k} g_j h_{k-j} \f$, which is solved for
  /// \f$ h_k \f$ in increasing \f$ k \f$. Throws `std::domain_error` when
  /// \f$ g_0 \f$ is exactly zero: the quotient then has no power series. A
  /// nonzero \f$ g_0 \f$ of any magnitude is divided by as it stands.
  TruncatedSeries &operator/=(const TruncatedSeries &other);

  /// Adds \p value to the constant term.
  TruncatedSeries &operator+=(Complex value);
  /// Subtracts \p value from the constant term.
  TruncatedSeries &operator-=(Complex value);
  /// Multiplies every coefficient by \p value.
  TruncatedSeries &operator*=(Complex value);
  /// Divides every coefficient by \p value. Throws `std::domain_error` when
  /// \p value is exactly zero.
  TruncatedSeries &operator/=(Complex value);

 private:
  std::vector<Complex> coefficients_;
};

/// The sum of two series of one order, coefficient by coefficient.
[[nodiscard]] TruncatedSeries operator+(TruncatedSeries f,
                                        const TruncatedSeries &g);
/// The difference of two series of one order, coefficient by coefficient.
[[nodiscard]] TruncatedSeries operator-(TruncatedSeries f,
                                        const TruncatedSeries &g);
/// The Cauchy product of two series of one order; see
/// `TruncatedSeries::operator*=`.
[[nodiscard]] TruncatedSeries operator*(TruncatedSeries f,
                                        const TruncatedSeries &g);
/// The quotient of two series of one order by the quotient recurrence; see
/// `TruncatedSeries::operator/=`. Throws `std::domain_error` when
/// \f$ g_0 \f$ is exactly zero.
[[nodiscard]] TruncatedSeries operator/(TruncatedSeries f,
                                        const TruncatedSeries &g);
/// The series \p f with \p value added to its constant term.
[[nodiscard]] TruncatedSeries operator+(TruncatedSeries f,
                                        TruncatedSeries::Complex value);
/// The series \p f with \p value subtracted from its constant term.
[[nodiscard]] TruncatedSeries operator-(TruncatedSeries f,
                                        TruncatedSeries::Complex value);
/// The series \p f with every coefficient multiplied by \p value.
[[nodiscard]] TruncatedSeries operator*(TruncatedSeries f,
                                        TruncatedSeries::Complex value);
/// The series \p f with every coefficient divided by \p value. Throws
/// `std::domain_error` when \p value is exactly zero.
[[nodiscard]] TruncatedSeries operator/(TruncatedSeries f,
                                        TruncatedSeries::Complex value);
/// The series \p f with \p value added to its constant term.
[[nodiscard]] TruncatedSeries operator+(TruncatedSeries::Complex value,
                                        TruncatedSeries f);
/// The constant series of value \p value less the series \p f.
[[nodiscard]] TruncatedSeries operator-(TruncatedSeries::Complex value,
                                        const TruncatedSeries &f);
/// The series \p f with every coefficient multiplied by \p value.
[[nodiscard]] TruncatedSeries operator*(TruncatedSeries::Complex value,
                                        TruncatedSeries f);
/// The quotient of the constant series of value \p value by \p f, by the
/// quotient recurrence. Throws `std::domain_error` when \f$ f_0 \f$ is
/// exactly zero.
[[nodiscard]] TruncatedSeries operator/(TruncatedSeries::Complex value,
                                        const TruncatedSeries &f);

/// The integer power \f$ f^n \f$. For \f$ n \ge 0 \f$ it is formed by
/// repeated squaring, that is by Cauchy products alone, so it has a value for
/// every series, and \f$ f^0 \f$ is the constant series 1. For \f$ n < 0 \f$
/// it is \f$ (1/f)^{|n|} \f$, the reciprocal by the quotient recurrence;
/// `std::domain_error` is thrown when \f$ f_0 \f$ is exactly zero.
[[nodiscard]] TruncatedSeries pow(const TruncatedSeries &f, int exponent);

/// The square root \f$ g = \sqrt{f} \f$ on a declared branch. Comparing
/// coefficients of \f$ t^k \f$ in \f$ g^2 = f \f$ gives
/// \f$ 2 g_0 g_k = f_k - \sum_{j=1}^{k-1} g_j g_{k-j} \f$ for \f$ k \ge 1 \f$.
///
/// The branch is the value of \f$ g_0 \f$: \p branch when it is given (the
/// root of \f$ f_0 \f$ the caller continues from, taken as it is given), and
/// the principal root of \f$ f_0 \f$ otherwise. Every higher coefficient
/// follows from \f$ g_0 \f$. Throws `std::domain_error` when \f$ f_0 \f$, or
/// the given \p branch, is exactly zero: zero is the branch point of the
/// root, where it has no power series.
[[nodiscard]] TruncatedSeries sqrt(
    const TruncatedSeries &f,
    std::optional<TruncatedSeries::Complex> branch = std::nullopt);

/// The exponential \f$ g = e^{f} \f$. Comparing coefficients of
/// \f$ t^{k-1} \f$ in \f$ g' = f' g \f$ gives
/// \f$ k g_k = \sum_{j=1}^{k} j f_j g_{k-j} \f$, with
/// \f$ g_0 = e^{f_0} \f$.
[[nodiscard]] TruncatedSeries exp(const TruncatedSeries &f);

/// The logarithm \f$ g = \log f \f$ on a continued branch. Comparing
/// coefficients of \f$ t^{k-1} \f$ in \f$ f g' = f' \f$ gives
/// \f$ k f_0 g_k = k f_k - \sum_{j=1}^{k-1} j g_j f_{k-j} \f$ for
/// \f$ k \ge 1 \f$.
///
/// The branch is the value of \f$ g_0 \f$: \p branch when it is given (the
/// logarithm of \f$ f_0 \f$ the caller continues from), and the principal
/// logarithm of \f$ f_0 \f$ otherwise. The higher coefficients are the same
/// on every branch. Throws `std::domain_error` when \f$ f_0 \f$ is exactly
/// zero, the branch point of the logarithm.
[[nodiscard]] TruncatedSeries log(
    const TruncatedSeries &f,
    std::optional<TruncatedSeries::Complex> branch = std::nullopt);

/// The sine and the cosine of \p f, in that order. With \f$ s = \sin f \f$
/// and \f$ c = \cos f \f$, comparing coefficients of \f$ t^{k-1} \f$ in
/// \f$ s' = f' c \f$ and \f$ c' = -f' s \f$ gives the coupled recurrence
/// \f$ k s_k = \sum_{j=1}^{k} j f_j c_{k-j} \f$,
/// \f$ k c_k = -\sum_{j=1}^{k} j f_j s_{k-j} \f$, with
/// \f$ s_0 = \sin f_0 \f$ and \f$ c_0 = \cos f_0 \f$.
[[nodiscard]] std::pair<TruncatedSeries, TruncatedSeries> sinCos(
    const TruncatedSeries &f);

/// The sine of \p f: the first series of `sinCos`.
[[nodiscard]] TruncatedSeries sin(const TruncatedSeries &f);

/// The cosine of \p f: the second series of `sinCos`.
[[nodiscard]] TruncatedSeries cos(const TruncatedSeries &f);

/// The angle \f$ \theta = \operatorname{atan2}(y, x) \f$ of the point
/// \f$ (x, y) \f$: the function with \f$ \cos\theta = x / \sqrt{x^2 + y^2} \f$
/// and \f$ \sin\theta = y / \sqrt{x^2 + y^2} \f$, that is
/// \f$ \theta = \frac{1}{2i} \left( \log(x + iy) - \log(x - iy) \right) \f$.
/// Its coefficients of order \f$ k \ge 1 \f$ are therefore
/// \f$ \theta_k = \frac{1}{2i} (u_k - v_k) \f$ with \f$ u = \log(x + iy) \f$
/// and \f$ v = \log(x - iy) \f$, each by the recurrence of `log`; this is the
/// derivative \f$ \theta' = (x y' - y x') / (x^2 + y^2) \f$ in partial
/// fractions.
///
/// The order-0 value is \p branch when it is given (the angle the caller
/// continues from). Otherwise it is `std::atan2` of the real parts when
/// \f$ x_0 \f$ and \f$ y_0 \f$ are both real, the angle in
/// \f$ [-\pi, \pi] \f$, and
/// \f$ \frac{1}{2i} (\operatorname{Log}(x_0 + i y_0) -
/// \operatorname{Log}(x_0 - i y_0)) \f$ with principal logarithms when either
/// is not real. Throws `std::domain_error` when \f$ x_0 + i y_0 \f$ or
/// \f$ x_0 - i y_0 \f$ is exactly zero (for real arguments: at the origin),
/// where the angle has no power series.
[[nodiscard]] TruncatedSeries atan2(
    const TruncatedSeries &y, const TruncatedSeries &x,
    std::optional<TruncatedSeries::Complex> branch = std::nullopt);

/// The inverse cosine \f$ \theta = \arccos f \f$ on a declared sheet, through
/// its derivative \f$ \theta' = -f' / s \f$ with \f$ s = \sin\theta \f$ a
/// square root of \f$ 1 - f^2 \f$: \f$ s \f$ is formed by the recurrence of
/// `sqrt` from its order-0 value, the quotient \f$ q = -f' / s \f$ by the
/// quotient recurrence, and \f$ \theta_k = q_{k-1} / k \f$ for
/// \f$ k \ge 1 \f$.
///
/// The order-0 value \f$ \theta_0 \f$ is \p branch when it is given (the
/// angle the caller continues from, with \f$ \cos\theta_0 = f_0 \f$), and the
/// principal inverse cosine of \f$ f_0 \f$ otherwise. It fixes the sign of
/// \f$ s \f$: \f$ s_0 \f$ is the one of the two roots
/// \f$ \pm\sqrt{(1 - f_0)(1 + f_0)} \f$ that equals \f$ \sin\theta_0 \f$,
/// selected as the root whose product with the complex conjugate of
/// \f$ \sin\theta_0 \f$ has the larger real part. Throws `std::domain_error`
/// when \f$ f_0 \f$ is exactly \f$ +1 \f$ or \f$ -1 \f$, the branch points
/// of the inverse cosine.
[[nodiscard]] TruncatedSeries acos(
    const TruncatedSeries &f,
    std::optional<TruncatedSeries::Complex> branch = std::nullopt);

/// The composition \f$ P(f) = \sum_{j=0}^{n} a_j f^j \f$ of the polynomial
/// with coefficients \p polynomial \f$ = (a_0, \dots, a_n) \f$ with the
/// series \p f, by Horner's rule in series arithmetic:
/// \f$ P(f) = a_0 + f (a_1 + f (a_2 + \dots)) \f$, each step one Cauchy
/// product and one sum. A polynomial is a finite sum, so the composition has
/// a value for every \f$ f \f$; the empty polynomial is zero.
[[nodiscard]] TruncatedSeries composePolynomial(
    const std::vector<TruncatedSeries::Complex> &polynomial,
    const TruncatedSeries &f);

/// # TruncatedSeriesMatrix
///
/// A dense matrix of series: the coefficient matrices
/// \f$ A_0, \dots, A_p \f$ of \f$ A(t) = \sum_{k=0}^{p} A_k t^k + O(t^{p+1}) \f$,
/// all of one shape, to a declared order \f$ p \f$ between 0 and 10. Entry
/// \f$ (i, j) \f$ of \f$ A(t) \f$ is the `TruncatedSeries` with coefficients
/// \f$ (A_0)_{ij}, \dots, (A_p)_{ij} \f$.
class TruncatedSeriesMatrix {
 public:
  using Complex = std::complex<double>;

  /// The zero matrix of the given shape and order. Throws
  /// `std::invalid_argument` when \p order is negative or above
  /// `TruncatedSeries::kMaximumOrder`, or a dimension is negative.
  TruncatedSeriesMatrix(int order, Eigen::Index rows, Eigen::Index cols);

  /// The matrix series with the given coefficient matrices
  /// \f$ A_0, \dots, A_p \f$. Throws `std::invalid_argument` when the list is
  /// empty, holds more than `TruncatedSeries::kMaximumOrder + 1` matrices, or
  /// holds matrices of different shapes.
  explicit TruncatedSeriesMatrix(std::vector<Eigen::MatrixXcd> coefficients);

  /// The constant series of the \p size by \p size identity matrix.
  [[nodiscard]] static TruncatedSeriesMatrix identity(int order,
                                                      Eigen::Index size);

  /// The declared order \f$ p \f$.
  [[nodiscard]] int order() const noexcept {
    return static_cast<int>(coefficients_.size()) - 1;
  }
  /// The number of rows.
  [[nodiscard]] Eigen::Index rows() const noexcept {
    return coefficients_.front().rows();
  }
  /// The number of columns.
  [[nodiscard]] Eigen::Index cols() const noexcept {
    return coefficients_.front().cols();
  }

  /// The coefficient matrices \f$ A_0, \dots, A_p \f$.
  [[nodiscard]] const std::vector<Eigen::MatrixXcd> &coefficients()
      const noexcept {
    return coefficients_;
  }

  /// The series of entry (\p row, \p col). Throws `std::out_of_range` for an
  /// index outside the matrix.
  [[nodiscard]] TruncatedSeries entry(Eigen::Index row, Eigen::Index col) const;

  /// Replaces entry (\p row, \p col) by \p series. Throws `std::out_of_range`
  /// for an index outside the matrix and `std::invalid_argument` when the
  /// order of \p series differs from the order of the matrix.
  void setEntry(Eigen::Index row, Eigen::Index col,
                const TruncatedSeries &series);

  /// The matrix \f$ \sum_{k=0}^{p} A_k t^k \f$ at the parameter value \p t,
  /// by Horner's rule.
  [[nodiscard]] Eigen::MatrixXcd evaluate(Complex t) const;

  /// Coefficientwise sum. Throws `std::invalid_argument` when the orders or
  /// the shapes differ.
  [[nodiscard]] TruncatedSeriesMatrix operator+(
      const TruncatedSeriesMatrix &other) const;
  /// Coefficientwise difference. Throws `std::invalid_argument` when the
  /// orders or the shapes differ.
  [[nodiscard]] TruncatedSeriesMatrix operator-(
      const TruncatedSeriesMatrix &other) const;
  /// The matrix product, the Cauchy product with matrix coefficients:
  /// \f$ (AB)_k = \sum_{j=0}^{k} A_j B_{k-j} \f$. Throws
  /// `std::invalid_argument` when the orders differ or the inner dimensions
  /// do not agree.
  [[nodiscard]] TruncatedSeriesMatrix operator*(
      const TruncatedSeriesMatrix &other) const;

  /// The inverse \f$ B = A^{-1} \f$ of a square matrix series. Comparing
  /// coefficients of \f$ t^k \f$ in \f$ A B = I \f$ gives
  /// \f$ A_0 B_0 = I \f$ and
  /// \f$ A_0 B_k = -\sum_{j=1}^{k} A_j B_{k-j} \f$ for \f$ k \ge 1 \f$; every
  /// order is solved with the one factorization of \f$ A_0 \f$ (Gaussian
  /// elimination with partial pivoting).
  ///
  /// Throws `std::domain_error` when the elimination of \f$ A_0 \f$ meets a
  /// pivot that is exactly zero: \f$ A_0 \f$ is then singular and the inverse
  /// has no power series. A nonzero pivot of any magnitude is divided by as
  /// it stands. Throws `std::invalid_argument` when the matrix is not square.
  [[nodiscard]] TruncatedSeriesMatrix inverse() const;

  /// The determinant \f$ \det A(t) \f$ of a square matrix series, as a
  /// series, by Bird's division-free algorithm carried out in series
  /// arithmetic. For an \f$ n \times n \f$ matrix \f$ X \f$ let
  /// \f$ \mu(X) \f$ be the upper triangular matrix that keeps the entries of
  /// \f$ X \f$ above the diagonal and has the diagonal entries
  /// \f$ \mu(X)_{ii} = -\sum_{j > i} X_{jj} \f$. Starting from
  /// \f$ X^{(1)} = A \f$ and iterating \f$ X^{(m+1)} = \mu(X^{(m)}) A \f$,
  /// the determinant is \f$ (-1)^{n-1} \f$ times entry \f$ (1, 1) \f$ of
  /// \f$ X^{(n)} \f$ (R. S. Bird, Information Processing Letters 111 (2011)
  /// 1072).
  ///
  /// The algorithm uses sums and products only, so it is an identity in the
  /// ring of truncated series: the determinant has a value for every matrix
  /// series, a singular \f$ A_0 \f$ included, and no pivot is divided by. It
  /// costs \f$ n - 1 \f$ products of matrix series. The sums it forms are
  /// those of the expanded determinant and are not pivoted, so for a large
  /// dense matrix the cancellation between them costs digits that an
  /// elimination keeps; for a matrix with an invertible \f$ A_0 \f$ the
  /// logarithmic derivative
  /// \f$ (\det A)' / \det A = \operatorname{tr}(A^{-1} A') \f$ is available
  /// from `inverse`. Throws `std::invalid_argument` when the matrix is not
  /// square. The determinant of the 0 by 0 matrix is the constant series 1.
  [[nodiscard]] TruncatedSeries determinant() const;

  /// The series \f$ P(t) = \sum_{k=0}^{p} P_k t^k \f$ of the Riesz (spectral)
  /// projector of a group of eigenvalues of the square matrix series
  /// \f$ A(t) \f$.
  ///
  /// The group is declared at order 0: it consists of the eigenvalues
  /// \f$ \lambda \f$ of \f$ A_0 \f$ for which \p inGroup returns true, and
  /// \f$ P(t) \f$ is the projector onto the generalized eigenspace of the
  /// eigenvalues of \f$ A(t) \f$ that continue them. \f$ P_0 \f$ is the
  /// projector of \f$ A_0 \f$ that commutes with \f$ A_0 \f$, has the
  /// generalized eigenspace of the group as its range and that of the
  /// remaining eigenvalues as its kernel.
  ///
  /// The higher coefficients follow, with no contour integral, from the two
  /// identities that define a spectral projector, \f$ P(t)^2 = P(t) \f$ and
  /// \f$ A(t) P(t) = P(t) A(t) \f$, order by order. In a basis adapted to
  /// \f$ P_0 \f$, in which \f$ A_0 = \operatorname{diag}(T_{11}, T_{22}) \f$
  /// with \f$ T_{11} \f$ carrying the group and \f$ T_{22} \f$ the rest,
  /// write \f$ S_k = \sum_{j=1}^{k-1} P_j P_{k-j} \f$ and
  /// \f$ R_k = -\sum_{j=1}^{k} [A_j, P_{k-j}] \f$. The coefficient of
  /// \f$ t^k \f$ of \f$ P^2 = P \f$ fixes the diagonal blocks,
  /// \f$ (P_k)_{11} = -(S_k)_{11} \f$ and \f$ (P_k)_{22} = (S_k)_{22} \f$,
  /// and that of \f$ [A, P] = 0 \f$ fixes the blocks between the group and
  /// its complement as the solutions of the Sylvester equations
  /// \f$ T_{11} X - X T_{22} = (R_k)_{12} \f$ for \f$ X = (P_k)_{12} \f$ and
  /// \f$ T_{22} Y - Y T_{11} = (R_k)_{21} \f$ for \f$ Y = (P_k)_{21} \f$.
  ///
  /// The adapted basis comes from the complex Schur form of \f$ A_0 \f$,
  /// reordered so that the group leads, so \f$ T_{11} \f$ and \f$ T_{22} \f$
  /// are upper triangular and each Sylvester equation is solved by back
  /// substitution, dividing by the differences
  /// \f$ \lambda - \mu \f$ of an eigenvalue \f$ \lambda \f$ of the group and
  /// an eigenvalue \f$ \mu \f$ outside it. For a diagonalizable \f$ A_0 \f$
  /// in its eigenbasis the first order reduces to the entries
  /// \f$ (A_1)_{ab} / (\lambda_a - \lambda_b) \f$ of first-order perturbation
  /// theory. No eigenvector matrix is inverted, so a Jordan block of
  /// \f$ A_0 \f$ inside the group or outside it is handled as it stands, and
  /// \f$ P(t) \f$ is a power series even where the eigenvalues inside the
  /// group split with a square-root branch point.
  ///
  /// A group that is empty gives the zero series and a group that holds
  /// every eigenvalue gives the constant identity. Throws
  /// `std::domain_error` when an eigenvalue of the group is exactly equal to
  /// an eigenvalue outside it, for then the difference has no quotient and
  /// the projector is not defined; eigenvalues that differ by any nonzero
  /// amount are divided by as they stand. Throws `std::invalid_argument` when
  /// the matrix is not square and `std::runtime_error` when the QR iteration
  /// of the Schur form of \f$ A_0 \f$ reports that it did not converge.
  [[nodiscard]] TruncatedSeriesMatrix rieszProjector(
      const std::function<bool(Complex)> &inGroup) const;

 private:
  std::vector<Eigen::MatrixXcd> coefficients_;
};

/// A point or a value with one series per component.
using SeriesVector = std::vector<TruncatedSeries>;

/// A map \f$ F \f$ evaluated in series arithmetic: it receives the series of
/// the components of a point \f$ x(t) \f$ and returns the series of the
/// components of \f$ F(x(t)) \f$, all of the order of its arguments.
using SeriesMap = std::function<SeriesVector(const SeriesVector &)>;

/// The linear solve of a reversion: it receives a right-hand side \f$ r \f$
/// and returns the solution \f$ s \f$ of \f$ J s = r \f$ with
/// \f$ J = F'(x_0) \f$ the Jacobian at the base point.
using LinearSolve = std::function<Eigen::VectorXcd(const Eigen::VectorXcd &)>;

/// # SeriesReversion
///
/// The result of `revert`: the path \f$ s(t) = \sum_{k=1}^{p} s_k t^k \f$
/// with \f$ F(x_0 + s(t)) = (1 - t) F(x_0) + O(t^{p+1}) \f$.
struct SeriesReversion {
  /// The residual \f$ F(x_0) \f$ at the base point.
  Eigen::VectorXcd residual{};
  /// The coefficients \f$ s_0 = 0, s_1, \dots, s_p \f$ of the path, one
  /// vector per order.
  std::vector<Eigen::VectorXcd> coefficients{};

  /// The path at the parameter value \p t,
  /// \f$ s(t) = \sum_{k=1}^{p} s_k t^k \f$, by Horner's rule.
  [[nodiscard]] Eigen::VectorXcd evaluate(std::complex<double> t) const;

  /// The order-\f$ p \f$ step \f$ s(1) = \sum_{k=1}^{p} s_k \f$.
  [[nodiscard]] Eigen::VectorXcd step() const;
};

/// The Jacobian \f$ J = F'(x_0) \f$ of a map evaluated in series arithmetic,
/// at the point \f$ x_0 \f$ given by \p point:
/// column \f$ i \f$ is the coefficient of \f$ t \f$ of
/// \f$ F(x_0 + t e_i) \f$, with \f$ e_i \f$ the \f$ i \f$-th coordinate
/// direction, read from one evaluation of \p map on series of order 1 per
/// column. It is the exact derivative, to rounding, and has as many rows as
/// \p map returns components.
[[nodiscard]] Eigen::MatrixXcd jacobian(const SeriesMap &map,
                                        const Eigen::VectorXcd &point);

/// The reversion of the equations \f$ F(x) = 0 \f$ about the point
/// \f$ x_0 \f$ given by \p point, to the order \f$ p \f$ given by
/// \p order: the series \f$ s(t) = \sum_{k=1}^{p} s_k t^k \f$ with
/// \f$ F(x_0 + s(t)) = (1 - t) F(x_0) + O(t^{p+1}) \f$, along which the
/// residual falls linearly to zero at \f$ t = 1 \f$.
///
/// With \f$ J = F'(x_0) \f$, the coefficient of \f$ t^k \f$ of
/// \f$ F(x_0 + s(t)) \f$ is \f$ J s_k + N_k \f$, where \f$ N_k \f$ is the
/// coefficient of \f$ t^k \f$ of \f$ F(x_0 + \sum_{j=1}^{k-1} s_j t^j) \f$
/// and holds every term in which \f$ s_k \f$ does not appear (\f$ s_k \f$
/// enters order \f$ k \f$ only through the linear term). Order by order the
/// path therefore solves \f$ J s_1 = -F(x_0) \f$ and
/// \f$ J s_k = -N_k \f$ for \f$ k \ge 2 \f$. \f$ N_k \f$ is read from one
/// evaluation of \p map on series of order \f$ k \f$, and each linear system
/// is handed to \p solve, so the caller decides how a rank-deficient
/// \f$ J \f$ is treated.
///
/// The order-\f$ p \f$ step is `SeriesReversion::step`,
/// \f$ s(1) = \sum_{k=1}^{p} s_k \f$. At order 1 it is Newton's step
/// \f$ -J^{-1} F(x_0) \f$. The series \f$ s(t) \f$ is the Taylor series of
/// the solution path of \f$ F(x) = (1 - t) F(x_0) \f$ through \f$ x_0 \f$;
/// its sum at \f$ t = 1 \f$ approaches a root as the order grows when
/// \f$ t = 1 \f$ lies inside its disc of convergence.
///
/// At order 0 the path has no coefficient beyond \f$ s_0 = 0 \f$ and only
/// the residual is evaluated. Throws `std::invalid_argument` when \p order is
/// negative or above `TruncatedSeries::kMaximumOrder`, when \p map returns a
/// series whose order differs from that of its arguments, or when \p solve
/// returns a vector whose length differs from that of \p point.
[[nodiscard]] SeriesReversion revert(const SeriesMap &map,
                                     const Eigen::VectorXcd &point, int order,
                                     const LinearSolve &solve);

}  // namespace tessera::numerics

#endif  // TESSERA_NUMERICS_TRUNCATEDSERIES_H
