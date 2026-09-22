// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "chainhodge/ChainHodge.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/OrderingMethods>
#include <Eigen/SVD>
#include <Eigen/SparseLU>
#include <Eigen/SparseQR>

#if defined(__linux__)
#include <unistd.h>
#endif

namespace tessera::chainhodge {

namespace {

constexpr double kEps = std::numeric_limits<double>::epsilon();

// One stored entry of a sparse factor: the complex scalar plus its index.
constexpr double kBytesPerStoredEntry =
    static_cast<double>(sizeof(Complex) + sizeof(SparseMatrix::StorageIndex));
constexpr double kBytesPerMegabyte = 1024.0 * 1024.0;

SparseMatrix sparseBoundary(const cobordism::ChainComplex &K, int k) {
  const int rows = (k >= 1) ? static_cast<int>(K.numSimplices(k - 1)) : 0;
  const int cols = static_cast<int>(K.numSimplices(k));
  SparseMatrix B(rows, cols);
  if (k >= 1) {
    const auto &entries = K.boundaryEntries(k);
    std::vector<Eigen::Triplet<Complex>> trip;
    trip.reserve(entries.size());
    for (const auto &e : entries)
      trip.emplace_back(e.row, e.column, Complex(static_cast<double>(e.value), 0.0));
    B.setFromTriplets(trip.begin(), trip.end());
  }
  B.makeCompressed();
  return B;
}

// Numerical rank at tolerance kappa * max(m,n) * eps * sigma_max, and the
// singular values (descending).
std::pair<int, Eigen::VectorXd> numericalRank(const Eigen::MatrixXcd &A, double kappa,
                                              double *toleranceOut = nullptr) {
  if (A.rows() == 0 || A.cols() == 0) {
    if (toleranceOut) *toleranceOut = 0.0;
    return {0, Eigen::VectorXd()};
  }
  Eigen::JacobiSVD<Eigen::MatrixXcd> svd(A);
  const Eigen::VectorXd sv = svd.singularValues();
  const double tol = kappa * static_cast<double>(std::max(A.rows(), A.cols())) * kEps * sv(0);
  if (toleranceOut) *toleranceOut = tol;
  int r = 0;
  for (int i = 0; i < sv.size(); ++i)
    if (sv(i) > tol) ++r;
  return {r, sv};
}

}  // namespace

SparseCostMeter::SparseCostMeter(std::string operation, int degree, int dimension)
    : operation_(std::move(operation)), degree_(degree), dimension_(dimension),
      startResident_(residentMegabytes()), start_(std::chrono::steady_clock::now()) {}

double SparseCostMeter::residentMegabytes() {
#if defined(__linux__)
  std::ifstream statm("/proc/self/statm");
  long long pages = 0;
  long long resident = 0;
  if (statm >> pages >> resident)
    return static_cast<double>(resident) * static_cast<double>(::sysconf(_SC_PAGESIZE)) /
           kBytesPerMegabyte;
#endif
  return std::numeric_limits<double>::quiet_NaN();
}

SparseCostReport SparseCostMeter::finish(long long systemRows, long long systemNonZeros,
                                         long long factorNonZeros,
                                         long long rightHandSides) const {
  SparseCostReport report;
  report.operation = operation_;
  report.degree = degree_;
  report.dimension = dimension_;
  report.systemRows = systemRows;
  report.systemNonZeros = systemNonZeros;
  report.factorNonZeros = factorNonZeros;
  report.rightHandSides = rightHandSides;
  report.wallSeconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start_).count();
  const double endResident = residentMegabytes();
  report.residentMegabytes = endResident - startResident_;
  if (factorNonZeros > 0) {
    report.factorMegabytes =
        static_cast<double>(factorNonZeros) * kBytesPerStoredEntry / kBytesPerMegabyte;
    if (systemNonZeros > 0)
      report.fillIn = static_cast<double>(factorNonZeros) / static_cast<double>(systemNonZeros);
  }
  return report;
}

SparseMatrix stackSparse(const SparseMatrix &top, const SparseMatrix &bottom, int columns) {
  if (top.rows() > 0 && static_cast<int>(top.cols()) != columns)
    throw std::invalid_argument("stackSparse: the top block has " + std::to_string(top.cols()) +
                                " columns, expected " + std::to_string(columns));
  if (bottom.rows() > 0 && static_cast<int>(bottom.cols()) != columns)
    throw std::invalid_argument("stackSparse: the bottom block has " +
                                std::to_string(bottom.cols()) + " columns, expected " +
                                std::to_string(columns));
  const int rows = static_cast<int>(top.rows() + bottom.rows());
  std::vector<Eigen::Triplet<Complex>> trip;
  trip.reserve(static_cast<std::size_t>(top.nonZeros() + bottom.nonZeros()));
  const auto scatter = [&](const SparseMatrix &A, int rowOffset) {
    for (int col = 0; col < A.outerSize(); ++col)
      for (SparseMatrix::InnerIterator it(A, col); it; ++it)
        trip.emplace_back(rowOffset + static_cast<int>(it.row()), static_cast<int>(it.col()),
                          it.value());
  };
  scatter(top, 0);
  scatter(bottom, static_cast<int>(top.rows()));
  SparseMatrix S(rows, columns);
  S.setFromTriplets(trip.begin(), trip.end());
  S.makeCompressed();
  return S;
}

struct ChainHodge::Factorization {
  Eigen::SparseLU<SparseMatrix> lu;
  bool ok{false};
};

ChainHodge::ChainHodge(cobordism::ChainComplex K, SquaredLengths s, Preset preset, Branch branch,
                       int crossoverDimension, double epsilon)
    : K_(std::move(K)), s_(std::move(s)), preset_(preset), branch_(branch),
      crossover_(crossoverDimension) {
  if (crossover_ < 1) throw std::invalid_argument("ChainHodge: crossover must be >= 1");
  const int d = K_.dimension();
  if (d < 0) throw std::invalid_argument("ChainHodge: empty complex");
  cert_ = WhitneyMass::certificate(K_, s_, branch_);
  cert_.epsilon = epsilon;
  sparse_.reserve(static_cast<std::size_t>(d) + 1);
  boundary_.reserve(static_cast<std::size_t>(d) + 1);
  for (int k = 0; k <= d; ++k) {
    sparse_.push_back(WhitneyMass::assemble(K_, s_, k, preset_, branch_));
    boundary_.push_back(sparseBoundary(K_, k));
  }
  factor_.assign(static_cast<std::size_t>(d) + 1, nullptr);
}

void ChainHodge::checkDegree(int k) const {
  if (k < 0 || k > K_.dimension())
    throw std::invalid_argument("ChainHodge: degree " + std::to_string(k) + " outside [0," +
                                std::to_string(K_.dimension()) + "]");
}

void ChainHodge::requireDense(int n, const char *what) const {
  if (n >= crossover_)
    throw std::length_error(std::string("ChainHodge::") + what + ": dimension " +
                            std::to_string(n) + " is at or above the dense crossover " +
                            std::to_string(crossover_));
}

int ChainHodge::size(int k) const {
  checkDegree(k);
  return static_cast<int>(K_.numSimplices(k));
}

const SparseMatrix &ChainHodge::Minv(int k) const {
  checkDegree(k);
  if (preset_ != Preset::L2)
    throw std::logic_error(
        "ChainHodge::Minv: the GRASSMANN_ALL preset's sparse object is the chain metric "
        "G_k (chainMetricSparse); its inverse is dense and is applied by applyMinv");
  return sparse_[static_cast<std::size_t>(k)];
}

const SparseMatrix &ChainHodge::chainMetricSparse(int k) const {
  checkDegree(k);
  if (preset_ != Preset::GRASSMANN_ALL)
    throw std::logic_error(
        "ChainHodge::chainMetricSparse: the L2 preset's chain metric G_k = M_k^{-1} is dense "
        "and is applied by applyG; its sparse object is Minv");
  return sparse_[static_cast<std::size_t>(k)];
}

const SparseMatrix &ChainHodge::boundary(int k) const {
  checkDegree(k);
  return boundary_[static_cast<std::size_t>(k)];
}

const ChainHodge::Factorization &ChainHodge::factorization(int k) const {
  auto &slot = factor_[static_cast<std::size_t>(k)];
  if (!slot) {
    auto f = std::make_shared<Factorization>();
    f->lu.compute(sparse_[static_cast<std::size_t>(k)]);
    f->ok = (f->lu.info() == Eigen::Success);
    slot = std::move(f);
  }
  if (!slot->ok)
    throw std::runtime_error("ChainHodge: the sparse metric at degree " + std::to_string(k) +
                             " is singular; no solve is defined (rank conditions fail here)");
  return *slot;
}

Eigen::MatrixXcd ChainHodge::solveSparse(int k, const Eigen::MatrixXcd &rhs) const {
  if (rhs.rows() != static_cast<Eigen::Index>(K_.numSimplices(k)))
    throw std::invalid_argument("ChainHodge: right-hand side has " + std::to_string(rhs.rows()) +
                                " rows, expected " + std::to_string(K_.numSimplices(k)));
  if (rhs.cols() == 0) return Eigen::MatrixXcd(rhs.rows(), 0);
  return factorization(k).lu.solve(rhs);
}

Eigen::MatrixXcd ChainHodge::applyG(int k, const Eigen::MatrixXcd &c) const {
  checkDegree(k);
  if (preset_ == Preset::L2) return solveSparse(k, c);
  return sparse_[static_cast<std::size_t>(k)] * c;
}

Eigen::MatrixXcd ChainHodge::applyMinv(int k, const Eigen::MatrixXcd &c) const {
  checkDegree(k);
  if (preset_ == Preset::L2) return sparse_[static_cast<std::size_t>(k)] * c;
  return solveSparse(k, c);
}

Pencil ChainHodge::pencil(int k) const {
  checkDegree(k);
  const int n = size(k);
  requireDense(n, "pencil");
  const int d = K_.dimension();
  Pencil P;
  P.degree = k;
  P.B = Eigen::MatrixXcd(sparse_[static_cast<std::size_t>(k)]);
  P.A = Eigen::MatrixXcd::Zero(n, n);
  const SparseMatrix &Mk = sparse_[static_cast<std::size_t>(k)];
  if (preset_ == Preset::L2) {
    P.variable = PencilVariable::GeometricImage;
    if (k >= 1) {
      // (∂_k M_k)^T M_{k-1}^{-1} (∂_k M_k)
      const Eigen::MatrixXcd X = Eigen::MatrixXcd(boundary_[static_cast<std::size_t>(k)] * Mk);
      const Eigen::MatrixXcd Y = solveSparse(k - 1, X);
      P.A += X.transpose() * Y;
    }
    if (k < d) {
      const SparseMatrix &Bk1 = boundary_[static_cast<std::size_t>(k) + 1];
      const SparseMatrix &Mk1 = sparse_[static_cast<std::size_t>(k) + 1];
      P.A += Eigen::MatrixXcd(Bk1 * Mk1 * SparseMatrix(Bk1.transpose()));
    }
  } else {
    P.variable = PencilVariable::Chain;
    if (k >= 1) {
      const SparseMatrix &Bk = boundary_[static_cast<std::size_t>(k)];
      const SparseMatrix &Gk0 = sparse_[static_cast<std::size_t>(k) - 1];
      P.A += Eigen::MatrixXcd(SparseMatrix(Bk.transpose()) * Gk0 * Bk);
    }
    if (k < d) {
      const SparseMatrix &Bk1 = boundary_[static_cast<std::size_t>(k) + 1];
      // G_k ∂_{k+1} G_{k+1}^{-1} ∂_{k+1}^T G_k
      const Eigen::MatrixXcd X = Eigen::MatrixXcd(SparseMatrix(Bk1.transpose()) * Mk);
      const Eigen::MatrixXcd Y = solveSparse(k + 1, X);
      P.A += X.transpose() * Y;
    }
  }
  return P;
}

Eigen::MatrixXcd ChainHodge::pencilAux(int k) const {
  if (preset_ != Preset::L2)
    throw std::logic_error("ChainHodge::pencilAux: the auxiliary form M_k A_k M_k is the Whitney "
                           "preset's; the GRASSMANN_ALL pencil is pencil(k) on chains");
  return pencil(k).A;
}

Eigen::MatrixXcd ChainHodge::hodgeOperator(int k) const {
  checkDegree(k);
  const int n = size(k);
  requireDense(n, "hodgeOperator");
  const int d = K_.dimension();
  const SparseMatrix &Mk = sparse_[static_cast<std::size_t>(k)];
  Eigen::MatrixXcd L = Eigen::MatrixXcd::Zero(n, n);
  if (preset_ == Preset::L2) {
    if (k >= 1) {
      const SparseMatrix &Bk = boundary_[static_cast<std::size_t>(k)];
      const Eigen::MatrixXcd Y = solveSparse(k - 1, Eigen::MatrixXcd(Bk));
      L += Mk * (SparseMatrix(Bk.transpose()) * Y);
    }
    if (k < d) {
      const SparseMatrix &Bk1 = boundary_[static_cast<std::size_t>(k) + 1];
      const SparseMatrix &Mk1 = sparse_[static_cast<std::size_t>(k) + 1];
      const Eigen::MatrixXcd W = solveSparse(k, Eigen::MatrixXcd::Identity(n, n));
      L += Eigen::MatrixXcd(Bk1 * Mk1 * SparseMatrix(Bk1.transpose())) * W;
    }
    return L;
  }
  return solveSparse(k, pencil(k).A);
}

SparseMatrix ChainHodge::stackedMatrix(int k) const {
  checkDegree(k);
  const int d = K_.dimension();
  const int n = size(k);
  const SparseMatrix &Mk = sparse_[static_cast<std::size_t>(k)];
  SparseMatrix top, bottom;
  if (preset_ == Preset::L2) {
    // S = [∂_{k+1}^T ; ∂_k M_k]
    if (k < d) top = SparseMatrix(boundary_[static_cast<std::size_t>(k) + 1].transpose());
    if (k >= 1) bottom = SparseMatrix(boundary_[static_cast<std::size_t>(k)] * Mk);
  } else {
    // S = [∂_k ; ∂_{k+1}^T G_k]
    if (k >= 1) top = boundary_[static_cast<std::size_t>(k)];
    if (k < d)
      bottom = SparseMatrix(SparseMatrix(boundary_[static_cast<std::size_t>(k) + 1].transpose()) * Mk);
  }
  return stackSparse(top, bottom, n);
}

SparseKernelRead ChainHodge::sparseNullSpace(const SparseMatrix &S, double kappa,
                                             SparseCostReport *report) {
  SparseKernelRead read;
  const int n = static_cast<int>(S.cols());
  // A matrix carries no degree, so the report says so rather than naming one.
  const SparseCostMeter meter("stacked-qr", -1, n);
  SparseMatrix ST = SparseMatrix(S.adjoint());  // ker S = range(S^H)^perp
  ST.makeCompressed();
  // The column norms of S come from its stored entries: S is never densified.
  double colNorm = 0.0;
  for (int c = 0; c < S.outerSize(); ++c) {
    double sum = 0.0;
    for (SparseMatrix::InnerIterator it(S, c); it; ++it) sum += std::norm(it.value());
    colNorm = std::max(colNorm, std::sqrt(sum));
  }
  const double tol = kappa * static_cast<double>(std::max(S.rows(), S.cols())) * kEps * colNorm;
  Eigen::SparseQR<SparseMatrix, Eigen::COLAMDOrdering<int>> qr;
  qr.setPivotThreshold(tol);
  qr.compute(ST);
  if (qr.info() != Eigen::Success)
    throw std::runtime_error(
        "ChainHodge::sparseNullSpace: the rank-revealing sparse QR of S^H failed; the stacked "
        "cochain matrix has no usable factorization at this geometry");
  const int r = static_cast<int>(qr.rank());
  read.rank = r;
  read.tolerance = tol;
  if (n > r) {
    // The kernel is Q applied to the trailing unit vectors, which is
    // n x (n - r): the n x n orthogonal factor is never formed.
    Eigen::MatrixXcd trailing = Eigen::MatrixXcd::Zero(n, n - r);
    trailing.bottomRows(n - r).setIdentity();
    read.kernel = qr.matrixQ() * trailing;
  } else {
    read.kernel = Eigen::MatrixXcd(n, 0);
  }
  read.cost = meter.finish(static_cast<long long>(ST.rows()),
                           static_cast<long long>(ST.nonZeros()),
                           static_cast<long long>(qr.matrixR().nonZeros()), n - r);
  if (report) *report = read.cost;
  return read;
}

Eigen::MatrixXcd ChainHodge::applyPencilOperator(int k, const Eigen::MatrixXcd &Z) const {
  checkDegree(k);
  const int n = size(k);
  if (Z.rows() != n)
    throw std::invalid_argument("ChainHodge::applyPencilOperator: the operand has " +
                                std::to_string(Z.rows()) + " rows, expected " + std::to_string(n));
  const int d = K_.dimension();
  const SparseMatrix &Mk = sparse_[static_cast<std::size_t>(k)];
  Eigen::MatrixXcd out = Eigen::MatrixXcd::Zero(n, Z.cols());
  if (Z.cols() == 0) return out;
  if (preset_ == Preset::L2) {
    // Ã_k = M_k ∂_k^T M_{k-1}^{-1} ∂_k M_k + ∂_{k+1} M_{k+1} ∂_{k+1}^T
    if (k >= 1) {
      const SparseMatrix &Bk = boundary_[static_cast<std::size_t>(k)];
      const Eigen::MatrixXcd W = Bk * Eigen::MatrixXcd(Mk * Z);
      const Eigen::MatrixXcd Y = solveSparse(k - 1, W);
      out += Mk * Eigen::MatrixXcd(SparseMatrix(Bk.transpose()) * Y);
    }
    if (k < d) {
      const SparseMatrix &Bk1 = boundary_[static_cast<std::size_t>(k) + 1];
      const SparseMatrix &Mk1 = sparse_[static_cast<std::size_t>(k) + 1];
      const Eigen::MatrixXcd W = SparseMatrix(Bk1.transpose()) * Z;
      out += Bk1 * Eigen::MatrixXcd(Mk1 * W);
    }
  } else {
    // A_k = ∂_k^T G_{k-1} ∂_k + G_k ∂_{k+1} G_{k+1}^{-1} ∂_{k+1}^T G_k
    if (k >= 1) {
      const SparseMatrix &Bk = boundary_[static_cast<std::size_t>(k)];
      const Eigen::MatrixXcd W = sparse_[static_cast<std::size_t>(k) - 1] * Eigen::MatrixXcd(Bk * Z);
      out += SparseMatrix(Bk.transpose()) * W;
    }
    if (k < d) {
      const SparseMatrix &Bk1 = boundary_[static_cast<std::size_t>(k) + 1];
      const Eigen::MatrixXcd W = SparseMatrix(Bk1.transpose()) * Eigen::MatrixXcd(Mk * Z);
      const Eigen::MatrixXcd Y = solveSparse(k + 1, W);
      out += Mk * Eigen::MatrixXcd(Bk1 * Y);
    }
  }
  return out;
}

HarmonicRead ChainHodge::harmonicChains(int k, double kappa, bool forceSparse) const {
  checkDegree(k);
  const int n = size(k);
  HarmonicRead read;
  read.degree = k;
  const bool dense = !forceSparse && n < crossover_;
  read.dense = dense;
  const SparseMatrix S = stackedMatrix(k);
  Eigen::MatrixXcd kernel;
  if (S.rows() == 0) {
    kernel = Eigen::MatrixXcd::Identity(n, n);
    read.rank = 0;
    read.tolerance = 0.0;
    read.gap = std::numeric_limits<double>::infinity();
  } else if (dense) {
    const Eigen::MatrixXcd Sd(S);
    Eigen::JacobiSVD<Eigen::MatrixXcd> svd(Sd, Eigen::ComputeFullV);
    const Eigen::VectorXd sv = svd.singularValues();
    const double tol = kappa * static_cast<double>(std::max(Sd.rows(), Sd.cols())) * kEps * sv(0);
    int r = 0;
    for (int i = 0; i < sv.size(); ++i)
      if (sv(i) > tol) ++r;
    read.rank = r;
    read.tolerance = tol;
    kernel = svd.matrixV().rightCols(n - r);
    if (r < sv.size() && sv(r) > 0.0)
      read.gap = (r >= 1) ? sv(r - 1) / sv(r) : std::numeric_limits<double>::infinity();
    else
      read.gap = std::numeric_limits<double>::infinity();
  } else {
    const SparseKernelRead sparse = sparseNullSpace(S, kappa);
    read.rank = sparse.rank;
    read.tolerance = sparse.tolerance;
    kernel = sparse.kernel;
    read.gap = std::numeric_limits<double>::quiet_NaN();
  }
  read.nullity = static_cast<int>(kernel.cols());
  if (preset_ == Preset::L2) {
    read.images = kernel;                                      // z = ker S
    read.chains = sparse_[static_cast<std::size_t>(k)] * kernel;  // h = M_k z
  } else {
    read.chains = kernel;                                       // h = ker S
    read.images = sparse_[static_cast<std::size_t>(k)] * kernel;  // G_k h
  }
  return read;
}

Eigen::MatrixXcd ChainHodge::geometricImage(int k, const Eigen::MatrixXcd &H) const {
  return applyG(k, H);
}

Eigen::MatrixXcd ChainHodge::harmonicGram(const HarmonicRead &read) const {
  // Whitney: H^T Z = Z^T M_k Z; Grassmann: H^T (G_k H). The transpose pairing.
  return read.chains.transpose() * read.images;
}

RankReport ChainHodge::rankConditions(int k, double kappa) const {
  checkDegree(k);
  const int d = K_.dimension();
  requireDense(size(k), "rankConditions");
  RankReport rep;
  rep.degree = k;
  rep.kappa = kappa;
  const int rankLower = (k >= 1) ? K_.rankOfBoundary(k) : 0;
  const int rankUpper = (k < d) ? K_.rankOfBoundary(k + 1) : 0;
  rep.expected = {{rankUpper, rankLower, rankLower, rankUpper}};
  const SparseMatrix &Mk = sparse_[static_cast<std::size_t>(k)];
  std::array<int, 4> m{{0, 0, 0, 0}};
  if (preset_ == Preset::L2) {
    if (k < d) {
      const SparseMatrix &Bk1 = boundary_[static_cast<std::size_t>(k) + 1];
      const SparseMatrix &Mk1 = sparse_[static_cast<std::size_t>(k) + 1];
      const Eigen::MatrixXcd Z = solveSparse(k, Eigen::MatrixXcd(Bk1));
      m[0] = numericalRank(Eigen::MatrixXcd(SparseMatrix(Bk1.transpose())) * Z, kappa).first;
      m[3] = numericalRank(Eigen::MatrixXcd(Bk1 * Mk1 * SparseMatrix(Bk1.transpose())), kappa).first;
    }
    if (k >= 1) {
      const SparseMatrix &Bk = boundary_[static_cast<std::size_t>(k)];
      m[1] = numericalRank(Eigen::MatrixXcd(Bk * Mk * SparseMatrix(Bk.transpose())), kappa).first;
      const Eigen::MatrixXcd Y = solveSparse(k - 1, Eigen::MatrixXcd(Bk));
      m[2] = numericalRank(Eigen::MatrixXcd(SparseMatrix(Bk.transpose())) * Y, kappa).first;
    }
  } else {
    if (k < d) {
      const SparseMatrix &Bk1 = boundary_[static_cast<std::size_t>(k) + 1];
      m[0] = numericalRank(Eigen::MatrixXcd(SparseMatrix(Bk1.transpose()) * Mk * Bk1), kappa).first;
      const Eigen::MatrixXcd Y = solveSparse(k + 1, Eigen::MatrixXcd(SparseMatrix(Bk1.transpose())));
      m[3] = numericalRank(Eigen::MatrixXcd(Bk1) * Y, kappa).first;
    }
    if (k >= 1) {
      const SparseMatrix &Bk = boundary_[static_cast<std::size_t>(k)];
      const SparseMatrix &Gk0 = sparse_[static_cast<std::size_t>(k) - 1];
      const Eigen::MatrixXcd Y = solveSparse(k, Eigen::MatrixXcd(SparseMatrix(Bk.transpose())));
      m[1] = numericalRank(Eigen::MatrixXcd(Bk) * Y, kappa).first;
      m[2] = numericalRank(Eigen::MatrixXcd(SparseMatrix(Bk.transpose()) * Gk0 * Bk), kappa).first;
    }
  }
  rep.measured = m;
  for (int i = 0; i < 4; ++i) rep.holds[static_cast<std::size_t>(i)] = (m[static_cast<std::size_t>(i)] == rep.expected[static_cast<std::size_t>(i)]);
  rep.decompositionHolds = rep.holds[0] && rep.holds[1];
  rep.kernelIsHarmonic = rep.decompositionHolds && rep.holds[2] && rep.holds[3];
  return rep;
}

SpectrumRead ChainHodge::spectrum(int k) const {
  const Pencil P = pencil(k);
  const int n = static_cast<int>(P.A.rows());
  SpectrumRead read;
  read.degree = k;
  if (n == 0) return read;
  const Eigen::MatrixXcd C = solveSparse(k, P.A);  // B^{-1} A
  Eigen::ComplexEigenSolver<Eigen::MatrixXcd> es(C, true);
  if (es.info() != Eigen::Success)
    throw std::runtime_error("ChainHodge::spectrum: eigensolver did not converge");
  std::vector<int> order(static_cast<std::size_t>(n));
  std::iota(order.begin(), order.end(), 0);
  const auto &ev = es.eigenvalues();
  std::sort(order.begin(), order.end(), [&](int a, int b) {
    if (ev(a).real() != ev(b).real()) return ev(a).real() < ev(b).real();
    return ev(a).imag() < ev(b).imag();
  });
  read.eigenvalues.reserve(static_cast<std::size_t>(n));
  read.vectors.resize(n, n);
  const double normA = P.A.norm();
  double worst = 0.0;
  for (int i = 0; i < n; ++i) {
    const int j = order[static_cast<std::size_t>(i)];
    read.eigenvalues.push_back(ev(j));
    Eigen::VectorXcd x = es.eigenvectors().col(j);
    x /= x.norm();
    read.vectors.col(i) = x;
    const double res = (P.A * x - ev(j) * (P.B * x)).norm();
    worst = std::max(worst, normA > 0.0 ? res / normA : res);
  }
  read.residual = worst;
  return read;
}

}  // namespace tessera::chainhodge
