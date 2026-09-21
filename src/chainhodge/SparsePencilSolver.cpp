// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "chainhodge/SparsePencilSolver.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseLU>

namespace tessera::chainhodge {

namespace {

using Dense = Eigen::MatrixXcd;
using Vector = Eigen::VectorXcd;

double hermitianDefect(const SparseMatrix &X) {
  const double norm = X.norm();
  const double defect = SparseMatrix(X - SparseMatrix(X.adjoint())).norm();
  return norm > 0.0 ? defect / norm : defect;
}

double oneNorm(const SparseMatrix &X) {
  double worst = 0.0;
  for (int c = 0; c < X.outerSize(); ++c) {
    double sum = 0.0;
    for (SparseMatrix::InnerIterator it(X, c); it; ++it) sum += std::abs(it.value());
    worst = std::max(worst, sum);
  }
  return worst;
}

// The factorized B = A - sigma M: Cholesky when B is positive definite (which
// is the statement that sigma lies below the spectrum), LU otherwise.
struct ShiftedFactorization {
  Eigen::SimplicialLLT<SparseMatrix> cholesky;
  Eigen::SparseLU<SparseMatrix> lu;
  bool positiveDefinite{false};

  void compute(const SparseMatrix &B) {
    cholesky.compute(B);
    positiveDefinite = (cholesky.info() == Eigen::Success);
    if (positiveDefinite) return;
    lu.compute(B);
    if (lu.info() != Eigen::Success)
      throw std::runtime_error(
          "SparsePencilSolver: A - sigma M is singular; the shift is an eigenvalue of the pencil");
  }

  [[nodiscard]] Dense solve(const Dense &rhs) const {
    if (positiveDefinite) return cholesky.solve(rhs);
    return lu.solve(rhs);
  }
};

// Hager's estimate of ||B^{-1}||_1 for a Hermitian B, whose adjoint solves are
// its own solves. Higham, "FORTRAN codes for estimating the one-norm of a real
// or complex matrix", ACM TOMS 14, 1988.
double inverseOneNormEstimate(const ShiftedFactorization &factor, Eigen::Index n, int &solves) {
  Vector x = Vector::Constant(n, Complex(1.0 / static_cast<double>(n), 0.0));
  double estimate = 0.0;
  for (int iteration = 0; iteration < 5; ++iteration) {
    const Vector y = factor.solve(x);
    ++solves;
    estimate = y.lpNorm<1>();
    Vector sign(n);
    for (Eigen::Index i = 0; i < n; ++i) {
      const double modulus = std::abs(y(i));
      sign(i) = modulus > 0.0 ? y(i) / modulus : Complex(1.0, 0.0);
    }
    const Vector z = factor.solve(sign);
    ++solves;
    Eigen::Index largest = 0;
    const double zMax = z.cwiseAbs().maxCoeff(&largest);
    if (iteration > 0 && zMax <= std::abs(z.dot(x))) break;
    x.setZero();
    x(largest) = Complex(1.0, 0.0);
  }
  return estimate;
}

class RandomColumns {
 public:
  explicit RandomColumns(std::uint64_t seed) : engine_(seed) {}
  void fill(Eigen::Ref<Dense> block) {
    for (Eigen::Index c = 0; c < block.cols(); ++c)
      for (Eigen::Index r = 0; r < block.rows(); ++r)
        block(r, c) = Complex(normal_(engine_), normal_(engine_));
  }

 private:
  std::mt19937_64 engine_;
  std::normal_distribution<double> normal_{0.0, 1.0};
};

double massNorm(const SparseMatrix &M, const Vector &w) {
  return std::sqrt(std::max(0.0, w.dot(M * w).real()));
}

// Append the columns of `candidates` to the M-orthonormal basis held in the
// first `size` columns of V. Each is projected off everything before it until
// a pass no longer shortens it (Kahan and Parlett's "twice is enough"): a
// column accepted after a pass that removed most of it would be rounding noise
// normalized into the basis, and the projected matrix would stop being the
// Rayleigh-Ritz matrix. A column that vanishes spans nothing new and is
// replaced by a random one.
void appendOrthonormal(const SparseMatrix &M, Dense &V, Eigen::Index size, const Dense &candidates,
                       RandomColumns &random) {
  for (Eigen::Index j = 0; j < candidates.cols(); ++j) {
    Vector w = candidates.col(j);
    const Eigen::Index before = size + j;
    for (int attempt = 0;; ++attempt) {
      const double original = massNorm(M, w);
      double norm = original;
      bool accepted = original > 0.0 && before == 0;
      for (int pass = 0; pass < 6 && !accepted && norm > 1e-14 * original && norm > 0.0; ++pass) {
        const Vector coefficients = V.leftCols(before).adjoint() * (M * w);
        w.noalias() -= V.leftCols(before) * coefficients;
        const double shortened = massNorm(M, w);
        accepted = shortened > 0.5 * norm;
        norm = shortened;
      }
      if (accepted) {
        V.col(before) = w / norm;
        break;
      }
      if (attempt == 4)
        throw std::runtime_error("SparsePencilSolver: could not extend the Lanczos basis");
      Dense fresh(V.rows(), 1);
      random.fill(fresh);
      w = fresh.col(0);
    }
  }
}

struct RitzSelection {
  std::vector<double> lambda;  // ascending, above the shift
  Dense coefficients;          // basis coefficients of the selected Ritz vectors
};

// The `count` Ritz values nearest above the shift (largest positive theta).
RitzSelection selectAboveShift(const Eigen::SelfAdjointEigenSolver<Dense> &ritz, double sigma, int count) {
  const auto &theta = ritz.eigenvalues();  // ascending
  RitzSelection out;
  std::vector<Eigen::Index> chosen;
  for (Eigen::Index i = theta.size() - 1; i >= 0 && static_cast<int>(chosen.size()) < count; --i)
    if (theta(i) > 0.0) chosen.push_back(i);
  out.coefficients.resize(theta.size(), static_cast<Eigen::Index>(chosen.size()));
  for (std::size_t c = 0; c < chosen.size(); ++c) {
    out.lambda.push_back(sigma + 1.0 / theta(chosen[c]));
    out.coefficients.col(static_cast<Eigen::Index>(c)) = ritz.eigenvectors().col(chosen[c]);
  }
  return out;
}

std::vector<double> relativeResiduals(const Dense &AZ, const Dense &MZ, const std::vector<double> &lambda,
                                      double sigma) {
  std::vector<double> out(lambda.size());
  for (std::size_t i = 0; i < lambda.size(); ++i) {
    const auto c = static_cast<Eigen::Index>(i);
    const double residual = (AZ.col(c) - lambda[i] * MZ.col(c)).norm();
    const double scale = (AZ.col(c) - sigma * MZ.col(c)).norm();
    out[i] = scale > 0.0 ? residual / scale : residual;
  }
  return out;
}

// Everything about a pencil and a shift that does not depend on how many
// pairs are asked for: the measured premises and the factorization.
struct Prepared {
  const SparseMatrix &A;
  const SparseMatrix &M;
  double sigma;
  SparseMatrix B;
  ShiftedFactorization factor;
  double hermitianDefectA{0.0};
  double hermitianDefectM{0.0};
  double conditioning{0.0};
  int conditioningSolves{0};
  // The low-rank term P D P^dagger of the left-hand matrix, and what the
  // Woodbury identity needs of it: B^{-1} P and (I + D P^dagger B^{-1} P)^{-1} D.
  Dense P, D, solvedP, core;
  bool shiftBelowSpectrum{false};

  [[nodiscard]] Dense applyA(const Dense &Z) const {
    Dense out = A * Z;
    if (P.cols() > 0) out.noalias() += P * (D * (P.adjoint() * Z));
    return out;
  }

  [[nodiscard]] Dense solve(const Dense &rhs) const {
    Dense y = factor.solve(rhs);
    if (P.cols() > 0) y.noalias() -= solvedP * (core * (P.adjoint() * y));
    return y;
  }

  Prepared(const SparseMatrix &A_, const SparseMatrix &M_, double sigma_,
           const SparsePencilOptions &options, const LowRankTerm &term = {})
      : A(A_), M(M_), sigma(sigma_), P(term.left), D(term.core) {
    const Eigen::Index n = A.rows();
    if (A.cols() != n || M.rows() != n || M.cols() != n)
      throw std::invalid_argument("SparsePencilSolver: A and M must be square of one size");
    hermitianDefectA = hermitianDefect(A);
    hermitianDefectM = hermitianDefect(M);
    if (hermitianDefectA > options.hermitianTolerance)
      throw std::invalid_argument("SparsePencilSolver: A is not Hermitian (relative defect " +
                                  std::to_string(hermitianDefectA) + ")");
    if (hermitianDefectM > options.hermitianTolerance)
      throw std::invalid_argument("SparsePencilSolver: M is not Hermitian (relative defect " +
                                  std::to_string(hermitianDefectM) + ")");
    {
      const Eigen::SimplicialLLT<SparseMatrix> massCholesky(M);
      if (massCholesky.info() != Eigen::Success)
        throw std::invalid_argument("SparsePencilSolver: M is not positive definite");
    }
    if (P.cols() > 0) {
      if (P.rows() != n || D.rows() != P.cols() || D.cols() != P.cols())
        throw std::invalid_argument("SparsePencilSolver: the low-rank term must be n x r with an r x r core");
      const double coreNorm = D.norm();
      if ((D - D.adjoint()).norm() > options.hermitianTolerance * std::max(coreNorm, 1e-300))
        throw std::invalid_argument("SparsePencilSolver: the core of the low-rank term is not Hermitian");
    }
    B = A - Complex(sigma, 0.0) * M;
    B.makeCompressed();
    factor.compute(B);
    conditioning = oneNorm(B) * inverseOneNormEstimate(factor, n, conditioningSolves);
    shiftBelowSpectrum = factor.positiveDefinite;
    if (P.cols() > 0) {
      const Eigen::Index r = P.cols();
      solvedP = factor.solve(P);
      const Dense gram = P.adjoint() * solvedP;                       // P^dagger B^{-1} P
      const Eigen::FullPivLU<Dense> inner(Dense::Identity(r, r) + D * gram);
      if (!inner.isInvertible())
        throw std::runtime_error(
            "SparsePencilSolver: A + P D P^dagger - sigma M is singular; the shift is an eigenvalue");
      core = inner.solve(D);
      // Inertia. With B positive definite, the block matrix [[B, P], [P^dagger, -D^{-1}]]
      // has the inertia of B plus that of -(D^{-1} + P^dagger B^{-1} P), and also that
      // of B + P D P^dagger plus that of -D^{-1}; so B + P D P^dagger is positive
      // definite exactly when D^{-1} + P^dagger B^{-1} P has as many negative
      // eigenvalues as D^{-1}. An invertible core is needed for the count.
      if (shiftBelowSpectrum) {
        const Eigen::FullPivLU<Dense> coreLU(D);
        if (!coreLU.isInvertible()) {
          shiftBelowSpectrum = false;
        } else {
          const Dense inverse = coreLU.inverse();
          auto negatives = [](const Dense &X) {
            const Eigen::SelfAdjointEigenSolver<Dense> es(0.5 * (X + X.adjoint()), Eigen::EigenvaluesOnly);
            return (es.eigenvalues().array() < 0.0).count();
          };
          shiftBelowSpectrum = negatives(inverse + gram) == negatives(inverse);
        }
      }
    }
  }
};

SparsePencilRead solvePrepared(const Prepared &prepared, int count, const SparsePencilOptions &options) {
  const SparseMatrix &A = prepared.A;
  const SparseMatrix &M = prepared.M;
  const double sigma = prepared.sigma;
  const Eigen::Index n = A.rows();
  if (count < 1 || count > n)
    throw std::invalid_argument("SparsePencilSolver: count must lie in [1, n]");

  SparsePencilRead read;
  read.hermitianDefectA = prepared.hermitianDefectA;
  read.hermitianDefectM = prepared.hermitianDefectM;
  read.shiftBelowSpectrum = prepared.shiftBelowSpectrum;
  read.solves = prepared.conditioningSolves;

  int block = options.blockSize > 0 ? options.blockSize : count + std::max(2, count / 4);
  block = static_cast<int>(std::min<Eigen::Index>(block, n));
  Eigen::Index capacity = options.maxBasisSize > 0 ? options.maxBasisSize : 10 * block;
  // Room for the vectors a restart keeps and one more block.
  capacity = std::min<Eigen::Index>(
      std::max<Eigen::Index>(capacity, std::max(count, block) + block), n);
  read.blockSize = block;

  RandomColumns random(options.seed);
  // One spare block beyond the capacity holds the next block across a restart.
  Dense V(n, std::min<Eigen::Index>(capacity + block, n));
  Dense T = Dense::Zero(capacity, capacity);
  Dense start(n, block);
  random.fill(start);
  appendOrthonormal(M, V, 0, start, random);
  Eigen::Index size = block;         // columns of V in use
  Eigen::Index lastStart = 0;        // the block not yet expanded
  Eigen::Index lastSize = block;

  RitzSelection selection;
  Dense Z, AZ, MZ;
  std::vector<double> residuals;
  for (; read.iterations < options.maxIterations;) {
    // Expand the newest block and complete the projected matrix with it.
    const Dense W = prepared.solve(M * V.middleCols(lastStart, lastSize));
    read.solves += static_cast<int>(lastSize);
    ++read.iterations;
    const Dense H = V.leftCols(size).adjoint() * (M * W);
    T.block(0, lastStart, size, lastSize) = H;
    T.block(lastStart, 0, lastSize, size) = H.adjoint();
    const Dense corner = H.middleRows(lastStart, lastSize);
    T.block(lastStart, lastStart, lastSize, lastSize) = 0.5 * (corner + corner.adjoint());

    const Eigen::SelfAdjointEigenSolver<Dense> ritz(T.topLeftCorner(size, size));
    if (ritz.info() != Eigen::Success)
      throw std::runtime_error("SparsePencilSolver: the projected eigenproblem did not converge");
    selection = selectAboveShift(ritz, sigma, count);
    Z = V.leftCols(size) * selection.coefficients;
    AZ = prepared.applyA(Z);
    MZ = M * Z;
    residuals = relativeResiduals(AZ, MZ, selection.lambda, sigma);
    const bool enough = static_cast<int>(selection.lambda.size()) == count;
    const bool met = std::all_of(residuals.begin(), residuals.end(),
                                 [&](double r) { return r <= options.tolerance; });
    if ((enough && met) || size == n) break;

    // The next block is what the expansion left outside the basis, topped up
    // with random columns when the expanded block was narrower.
    const Eigen::Index next = std::min<Eigen::Index>(block, n - size);
    Dense candidates(n, next);
    random.fill(candidates);
    const Eigen::Index carried = std::min<Eigen::Index>(next, W.cols());
    candidates.leftCols(carried) = W.leftCols(carried);
    appendOrthonormal(M, V, size, candidates, random);
    if (size + next > capacity) {
      // Thick restart. The next block was orthonormalized against the whole
      // old basis, so it carries the Krylov remainder and not directions the
      // compression discards; the basis is now compressed to the Ritz vectors
      // nearest above the shift, whose projected matrix is diagonal, and the
      // block moves up behind them.
      const auto &theta = ritz.eigenvalues();
      const Eigen::Index keep = std::min<Eigen::Index>(size, std::max<Eigen::Index>(count, block));
      const Dense kept = V.leftCols(size) * ritz.eigenvectors().rightCols(keep);
      const Dense moved = V.middleCols(size, next);
      V.leftCols(keep) = kept;
      V.middleCols(keep, next) = moved;
      T.setZero();
      for (Eigen::Index i = 0; i < keep; ++i) T(i, i) = theta(size - keep + i);
      size = keep;
      ++read.restarts;
    }
    lastStart = size;
    lastSize = next;
    size += next;
  }

  // Rayleigh-Ritz on the original pencil within the selected vectors: the
  // eigenvalues come from (A, M) themselves and a degenerate cluster is
  // resolved in one step.
  const auto found = static_cast<Eigen::Index>(selection.lambda.size());
  if (found > 0) {
    Dense K = Z.adjoint() * AZ;
    Dense G = Z.adjoint() * MZ;
    K = 0.5 * (K + K.adjoint()).eval();
    G = 0.5 * (G + G.adjoint()).eval();
    const Eigen::GeneralizedSelfAdjointEigenSolver<Dense> refine(K, G);
    if (refine.info() == Eigen::Success) {
      const Dense Y = refine.eigenvectors();
      Z = (Z * Y).eval();
      AZ = (AZ * Y).eval();
      MZ = (MZ * Y).eval();
      for (Eigen::Index i = 0; i < found; ++i)
        selection.lambda[static_cast<std::size_t>(i)] = refine.eigenvalues()(i);
      residuals = relativeResiduals(AZ, MZ, selection.lambda, sigma);
    }
  }

  read.vectors = Z;
  read.residuals = residuals;
  read.converged = static_cast<int>(found) == count &&
                   std::all_of(residuals.begin(), residuals.end(),
                               [&](double r) { return r <= options.tolerance; });
  if (found > 0) {
    const Dense gram = Z.adjoint() * MZ;
    read.orthonormalityDefect = (gram - Dense::Identity(found, found)).cwiseAbs().maxCoeff();
  }

  const double conditioning = prepared.conditioning;
  double worst = residuals.empty() ? std::numeric_limits<double>::infinity()
                                   : *std::max_element(residuals.begin(), residuals.end());
  if (static_cast<int>(found) != count) worst = std::numeric_limits<double>::infinity();
  for (double lambda : selection.lambda) read.eigenvalues.values.emplace_back(lambda, 0.0);
  const double scale = selection.lambda.empty() ? 1.0 : std::max(1.0, std::abs(selection.lambda.back()));
  const bool nonnegative = read.shiftBelowSpectrum && !selection.lambda.empty() &&
                           selection.lambda.front() >= -options.tolerance * scale;
  read.eigenvalues.certificate = cobordism::Certificate::certifiedNumerical(
      cobordism::CertificateDomain::BandWindow,
      nonnegative ? cobordism::CertificateRegime::PositiveSemidefinite
                  : cobordism::CertificateRegime::HermitianIndefinite,
      worst, conditioning, options.tolerance);
  return read;
}

}  // namespace

SparsePencilRead SparsePencilSolver::lowest(const SparsePencil &pencil, int count, double sigma,
                                            const SparsePencilOptions &options) {
  return lowest(pencil.A, pencil.M, count, sigma, options);
}

SparsePencilRead SparsePencilSolver::lowest(const SparseMatrix &A, const SparseMatrix &M, int count,
                                            double sigma, const SparsePencilOptions &options) {
  const Prepared prepared(A, M, sigma, options);
  return solvePrepared(prepared, count, options);
}

SparsePencilRead SparsePencilSolver::lowest(const SparseMatrix &A, const SparseMatrix &M,
                                            const LowRankTerm &term, int count, double sigma,
                                            const SparsePencilOptions &options) {
  const Prepared prepared(A, M, sigma, options, term);
  return solvePrepared(prepared, count, options);
}

EffectiveBettiRead SparsePencilSolver::effectiveBetti(const SparseMatrix &A, const SparseMatrix &M,
                                                      double epsilon, double sigma,
                                                      const SparsePencilOptions &options,
                                                      int initialCount) {
  if (!(epsilon > sigma))
    throw std::invalid_argument("SparsePencilSolver::effectiveBetti: epsilon must lie above the shift");
  const Prepared prepared(A, M, sigma, options);
  const auto n = static_cast<int>(A.rows());
  EffectiveBettiRead out;
  out.epsilon = epsilon;
  // Ask for more pairs until one lies outside the window: only then is the
  // count inside it complete.
  for (int count = std::min(n, std::max(1, initialCount));; count = std::min(n, 2 * count)) {
    out.band = solvePrepared(prepared, count, options);
    const auto &values = out.band.eigenvalues.values;
    out.rank = 0;
    while (out.rank < static_cast<int>(values.size()) &&
           values[static_cast<std::size_t>(out.rank)].real() <= epsilon)
      ++out.rank;
    out.complete = out.rank < static_cast<int>(values.size()) || static_cast<int>(values.size()) == n;
    if (out.complete || !out.band.converged || count == n) break;
  }
  const auto &values = out.band.eigenvalues.values;
  if (out.rank > 0) out.lastInside = values[static_cast<std::size_t>(out.rank) - 1].real();
  if (out.rank < static_cast<int>(values.size()))
    out.firstOutside = values[static_cast<std::size_t>(out.rank)].real();
  if (out.rank < static_cast<int>(values.size()))
    out.gap = (out.rank > 0 && out.lastInside > 0.0) ? out.firstOutside / out.lastInside
                                                      : std::numeric_limits<double>::infinity();
  out.certified = out.complete && out.band.shiftBelowSpectrum && out.band.eigenvalues.certificate.holds();
  return out;
}

}  // namespace tessera::chainhodge
