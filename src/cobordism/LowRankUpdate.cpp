// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "cobordism/LowRankUpdate.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace tessera::cobordism {

namespace {

using cd = std::complex<double>;

Eigen::MatrixXcd toMatrix(const std::vector<cd> &flat, int rows, int cols,
                          const char *name) {
  if (rows < 0 || cols < 0 ||
      flat.size() != static_cast<std::size_t>(rows) *
                         static_cast<std::size_t>(cols))
    throw std::invalid_argument(std::string(name) +
                                ": flat size does not match dimensions");
  Eigen::MatrixXcd matrix(rows, cols);
  for (int i = 0; i < rows; ++i)
    for (int j = 0; j < cols; ++j)
      matrix(i, j) = flat[static_cast<std::size_t>(i) * cols + j];
  return matrix;
}

std::vector<cd> toFlat(const Eigen::MatrixXcd &matrix) {
  std::vector<cd> flat(static_cast<std::size_t>(matrix.rows()) *
                       static_cast<std::size_t>(matrix.cols()));
  for (Eigen::Index i = 0; i < matrix.rows(); ++i)
    for (Eigen::Index j = 0; j < matrix.cols(); ++j)
      flat[static_cast<std::size_t>(i) * matrix.cols() + j] = matrix(i, j);
  return flat;
}

double inverseRcond(double rcond) {
  // Eigen's rcond() can return 0 for a singular factor; report +inf then.
  return rcond > 0.0 ? 1.0 / rcond : std::numeric_limits<double>::infinity();
}

} // namespace

LowRankUpdate::LowRankUpdate(const std::vector<cd> &base, int dim) {
  refactor(base, dim);
}

void LowRankUpdate::refactor(const std::vector<cd> &base, int dim) {
  base_ = toMatrix(base, dim, dim, "LowRankUpdate: base");
  baseFactorization_.compute(base_);
  // Condition estimate of the factorization, reported by every solve.
  baseConditioning_ = inverseRcond(baseFactorization_.rcond());
  dim_ = dim;
  clearUpdate();
}

void LowRankUpdate::setUpdate(const std::vector<cd> &left,
                              const std::vector<cd> &right, int rank) {
  if (rank < 0)
    throw std::invalid_argument("setUpdate: negative rank");
  left_ = toMatrix(left, dim_, rank, "setUpdate: left");
  right_ = toMatrix(right, rank, dim_, "setUpdate: right");
  rank_ = rank;
  // Z = A^{-1} U (factor solve, multi-rhs) and the LU of the capacitance
  // I_r + W Z depend only on the factors, so they are computed once per update.
  if (rank_ > 0) {
    capacitanceSolvedLeft_ = baseFactorization_.solve(left_);
    capacitanceFactorization_.compute(Eigen::MatrixXcd::Identity(rank_, rank_) +
                                      right_ * capacitanceSolvedLeft_);
    capacitanceConditioning_ = inverseRcond(capacitanceFactorization_.rcond());
  } else {
    clearUpdate();
  }
}

void LowRankUpdate::clearUpdate() noexcept {
  left_.resize(dim_, 0);
  right_.resize(0, dim_);
  capacitanceSolvedLeft_.resize(dim_, 0);
  capacitanceConditioning_ = 1.0;
  rank_ = 0;
}

Eigen::MatrixXcd LowRankUpdate::deltaMatrix() const {
  if (rank_ == 0)
    return Eigen::MatrixXcd::Zero(dim_, dim_);
  return left_ * right_;
}

CertifiedVector LowRankUpdate::solve(const std::vector<cd> &rhs,
                                     double tolerance) const {
  if (rhs.size() != static_cast<std::size_t>(dim_))
    throw std::invalid_argument("solve: rhs size does not match dimension");
  Eigen::VectorXcd b(dim_);
  for (int i = 0; i < dim_; ++i)
    b(i) = rhs[static_cast<std::size_t>(i)];

  const Eigen::VectorXcd baseSolution = baseFactorization_.solve(b);
  Eigen::VectorXcd solution = baseSolution;
  double conditioning = baseConditioning_;

  if (rank_ > 0) {
    const Eigen::VectorXcd correction =
        capacitanceFactorization_.solve(right_ * baseSolution);
    solution.noalias() -= capacitanceSolvedLeft_ * correction;
    conditioning = std::max(conditioning, capacitanceConditioning_);
  }

  // Measured residual of the updated system.
  Eigen::VectorXcd residualVector = base_ * solution - b;
  if (rank_ > 0)
    residualVector.noalias() += left_ * (right_ * solution);
  const double scale = b.norm();
  const double residual =
      scale > 0.0 ? residualVector.norm() / scale : residualVector.norm();

  CertifiedVector result;
  result.values.assign(solution.data(), solution.data() + solution.size());
  result.certificate = Certificate::structureExact(
      CertificateDomain::Static, CertificateRegime::NonNormal, residual,
      conditioning, tolerance);
  return result;
}

std::vector<cd> LowRankUpdate::apply(const std::vector<cd> &x) const {
  if (x.size() != static_cast<std::size_t>(dim_))
    throw std::invalid_argument("apply: vector size does not match dimension");
  Eigen::VectorXcd v(dim_);
  for (int i = 0; i < dim_; ++i)
    v(i) = x[static_cast<std::size_t>(i)];
  Eigen::VectorXcd y = base_ * v;
  if (rank_ > 0)
    y.noalias() += left_ * (right_ * v);
  return {y.data(), y.data() + y.size()};
}

bool LowRankUpdate::spansAffectedChange(const std::vector<cd> &updated,
                                        double tolerance) const {
  const Eigen::MatrixXcd updatedMatrix =
      toMatrix(updated, dim_, dim_, "spansAffectedChange: updated");
  const double scale = updatedMatrix.norm();
  const double defect = (updatedMatrix - base_ - deltaMatrix()).norm();
  return defect <= tolerance * std::max(scale, 1.0);
}

LowRankUpdate::TouchedFactors LowRankUpdate::factorsFromTouched(
    const std::vector<cd> &base, const std::vector<cd> &updated, int dim,
    const std::vector<int> &touched) {
  const Eigen::MatrixXcd baseMatrix =
      toMatrix(base, dim, dim, "factorsFromTouched: base");
  const Eigen::MatrixXcd updatedMatrix =
      toMatrix(updated, dim, dim, "factorsFromTouched: updated");
  std::vector<bool> isTouched(static_cast<std::size_t>(dim), false);
  for (const int index : touched) {
    if (index < 0 || index >= dim)
      throw std::invalid_argument("factorsFromTouched: index out of range");
    isTouched[static_cast<std::size_t>(index)] = true;
  }

  const Eigen::MatrixXcd delta = updatedMatrix - baseMatrix;

  TouchedFactors factors;
  // Support check: a nonzero outside the touched rows and columns means the
  // declared star does not span the change, so the factors are not exact.
  factors.spansChange = true;
  for (int i = 0; i < dim && factors.spansChange; ++i) {
    if (isTouched[static_cast<std::size_t>(i)])
      continue;
    for (int j = 0; j < dim; ++j) {
      if (isTouched[static_cast<std::size_t>(j)])
        continue;
      if (delta(i, j) != cd{0.0, 0.0}) {
        factors.spansChange = false;
        break;
      }
    }
  }
  if (!factors.spansChange)
    return factors;

  // Touched rows carry their whole delta rows (selector column e_i, right row =
  // delta row). Remaining support sits in touched columns on untouched rows,
  // carried as (residual column, selector row e_j). All-zero rows and columns
  // are trimmed, so the rank stays at most 2 * |active touched|.
  std::vector<Eigen::VectorXcd> leftColumns;
  std::vector<Eigen::RowVectorXcd> rightRows;
  for (int i = 0; i < dim; ++i) {
    if (!isTouched[static_cast<std::size_t>(i)])
      continue;
    if (delta.row(i).norm() == 0.0)
      continue;
    Eigen::VectorXcd selector = Eigen::VectorXcd::Zero(dim);
    selector(i) = cd{1.0, 0.0};
    leftColumns.push_back(std::move(selector));
    rightRows.emplace_back(delta.row(i));
  }
  for (int j = 0; j < dim; ++j) {
    if (!isTouched[static_cast<std::size_t>(j)])
      continue;
    Eigen::VectorXcd column = delta.col(j);
    for (int i = 0; i < dim; ++i)
      if (isTouched[static_cast<std::size_t>(i)])
        column(i) = cd{0.0, 0.0}; // rows already carried above
    if (column.norm() == 0.0)
      continue;
    Eigen::RowVectorXcd selector = Eigen::RowVectorXcd::Zero(dim);
    selector(j) = cd{1.0, 0.0};
    leftColumns.push_back(std::move(column));
    rightRows.push_back(std::move(selector));
  }

  factors.rank = static_cast<int>(leftColumns.size());
  Eigen::MatrixXcd left(dim, factors.rank);
  Eigen::MatrixXcd right(factors.rank, dim);
  for (int r = 0; r < factors.rank; ++r) {
    left.col(r) = leftColumns[static_cast<std::size_t>(r)];
    right.row(r) = rightRows[static_cast<std::size_t>(r)];
  }
  factors.left = toFlat(left);
  factors.right = toFlat(right);
  return factors;
}

CertifiedVector LowRankUpdate::rankOneEigenvalues(
    const std::vector<double> &eigenvalues, const std::vector<cd> &z,
    double rho, double tolerance) {
  const std::size_t n = eigenvalues.size();
  if (z.size() != n)
    throw std::invalid_argument(
        "rankOneEigenvalues: eigenvalue and vector sizes differ");
  for (std::size_t i = 1; i < n; ++i)
    if (eigenvalues[i] < eigenvalues[i - 1])
      throw std::invalid_argument(
          "rankOneEigenvalues: eigenvalues must be ascending (Hermitian "
          "domain; use a general eigensolve for the non-normal regime)");

  CertifiedVector result;
  if (n == 0) {
    result.certificate = Certificate::certifiedNumerical(
        CertificateDomain::Static, CertificateRegime::HermitianIndefinite, 0.0,
        1.0, tolerance);
    return result;
  }

  std::vector<double> weights(n);
  double totalWeight = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    weights[i] = std::norm(z[i]);
    totalWeight += weights[i];
  }
  const double spread =
      std::max({std::abs(eigenvalues.front()), std::abs(eigenvalues.back()),
                std::abs(rho) * totalWeight, 1.0});

  std::vector<double> updated;
  updated.reserve(n);
  double residual = 0.0;
  double conditioning = 1.0;

  if (rho == 0.0 || totalWeight == 0.0) {
    updated = eigenvalues;
  } else {
    // Deflation: modes with negligible weight keep their eigenvalue and the
    // neglected shift is charged to the residual. Coincident poles merge their
    // weights, each extra copy keeping its eigenvalue.
    constexpr double eps = std::numeric_limits<double>::epsilon();
    const double weightFloor = eps * totalWeight;
    const double mergeGap = eps * spread;
    std::vector<double> poles;
    std::vector<double> poleWeights;
    for (std::size_t i = 0; i < n; ++i) {
      if (weights[i] <= weightFloor) {
        updated.push_back(eigenvalues[i]);
        residual = std::max(residual, std::abs(rho) * weights[i] / spread);
        continue;
      }
      if (!poles.empty() && eigenvalues[i] - poles.back() <= mergeGap) {
        poleWeights.back() += weights[i];
        updated.push_back(eigenvalues[i]); // the multiplicity-preserved copy
        continue;
      }
      poles.push_back(eigenvalues[i]);
      poleWeights.push_back(weights[i]);
    }

    const auto m = poles.size();
    const double activeWeight =
        std::accumulate(poleWeights.begin(), poleWeights.end(), 0.0);
    // Sensitivity scale of the secular roots: the spread over the smallest
    // active pole gap.
    for (std::size_t i = 1; i < m; ++i)
      conditioning = std::max(
          conditioning, spread / std::max(poles[i] - poles[i - 1], mergeGap));
    const auto secular = [&](double lambda) {
      double value = 1.0;
      for (std::size_t i = 0; i < m; ++i)
        value += rho * poleWeights[i] / (poles[i] - lambda);
      return value;
    };
    // One root per interlacing interval: (d_k, d_{k+1}) plus the outer interval
    // on the sign(rho) side, bounded by |rho| * total weight. On each interval f
    // is monotone with known limit signs at the pole endpoints, and the finite
    // outer endpoint satisfies sign(f) = sign(rho). Bisection probes only strict
    // midpoints, so it never evaluates at a pole; a midpoint that rounds onto an
    // endpoint means the bracket has reached representability and the loop stops.
    for (std::size_t k = 0; k < m; ++k) {
      double lo;
      double hi;
      if (rho > 0.0) {
        lo = poles[k];
        hi = k + 1 < m ? poles[k + 1]
                       : poles[m - 1] + rho * activeWeight * (1.0 + eps) +
                             mergeGap;
      } else {
        lo = k == 0 ? poles[0] + rho * activeWeight * (1.0 + eps) - mergeGap
                    : poles[k - 1];
        hi = poles[k];
      }
      double a = lo;
      double b = hi;
      for (int iteration = 0; iteration < 200 && (b - a) > eps * spread;
           ++iteration) {
        const double mid = 0.5 * (a + b);
        if (mid <= a || mid >= b)
          break; // no representable interior point left
        const double fm = secular(mid);
        if (fm == 0.0) {
          a = mid;
          b = mid;
          break;
        }
        // Root is where f crosses zero rising (rho > 0) or falling
        // (rho < 0): keep the half whose endpoint signs still straddle.
        if ((fm < 0.0) == (rho > 0.0))
          a = mid;
        else
          b = mid;
      }
      updated.push_back(0.5 * (a + b));
      residual = std::max(residual, (b - a) / spread);
    }
    std::sort(updated.begin(), updated.end());
  }

  // Exact trace identity: sum(lambda') = sum(d) + rho * ||z||^2.
  double traceUpdated = 0.0;
  double traceBase = 0.0;
  for (const double value : updated)
    traceUpdated += value;
  for (const double value : eigenvalues)
    traceBase += value;
  residual = std::max(
      residual, std::abs(traceUpdated - traceBase - rho * totalWeight) /
                    (static_cast<double>(n) * spread));

  result.values.reserve(updated.size());
  for (const double value : updated)
    result.values.emplace_back(value, 0.0);
  result.certificate = Certificate::certifiedNumerical(
      CertificateDomain::Static, CertificateRegime::HermitianIndefinite,
      residual, conditioning, tolerance);
  return result;
}

} // namespace tessera::cobordism
