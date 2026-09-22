// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "cobordism/RecursiveQuotient.h"

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/LU>
#include <Eigen/QR>
#include <Eigen/SVD>
#include <Eigen/SparseLU>

#include <algorithm>
#include <cmath>
#include <limits>
#include <cstring>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "cobordism/AnalyticCache.h"
#include "cobordism/ChainComplex.h"
#include "cobordism/HodgeLaplacian.h"
#include "cobordism/IntegerLinalg.h"
#include "cobordism/OccupationSpectra.h"
#include "observables/PersistentModularity.h"
#include "mesh/Vertex.h"
#include "mesh/VertexList.h"
#include "spacetime/Spacetime.h"

namespace tessera::cobordism {

namespace {

using cd = std::complex<double>;
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kTwoPi = 6.283185307179586476925286766559;

Eigen::MatrixXcd toMatrix(const std::vector<cd> &flat, int rows, int cols,
                          const char *name) {
  if (rows < 0 || cols < 0 ||
      flat.size() !=
          static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols))
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

/// Spectral norm below a size guard, Frobenius norm above, to keep the SVD
/// cheap.
double matrixNorm2(const Eigen::MatrixXcd &m, int denseCrossover) {
  if (m.rows() == 0 || m.cols() == 0) return 0.0;
  if (m.rows() < denseCrossover && m.cols() < denseCrossover)
    return Eigen::JacobiSVD<Eigen::MatrixXcd>(m).singularValues()(0);
  return m.norm();
}

std::string cellName(const std::vector<std::uint64_t> &vertices) {
  std::ostringstream out;
  out << "cell(";
  for (std::size_t i = 0; i < vertices.size(); ++i) {
    if (i) out << ",";
    out << vertices[i];
  }
  out << ")";
  return out.str();
}

}  // namespace

RecursiveQuotient::Options::Options() = default;

/// Per-component interior factorization/solve payload, cached as an opaque
/// pointer in the AnalyticCache and keyed by the component's cell vertex-id
/// set, so a published TouchedStar invalidates exactly the touched
/// components.
struct RecursiveQuotient::ComponentSolve {
  cd lambda{0.0, 0.0};
  int interiorDim{0};
  std::vector<int> adjacentKept{};   // kept POSITIONS coupled to this interior
  Eigen::MatrixXcd X{};              // interiorDim x |adjacentKept|
  Eigen::MatrixXcd contribution{};   // |adjacentKept|^2: L_KI X
  Eigen::MatrixXcd rightKernel{};    // interiorDim x nullity, orthonormal
  Eigen::MatrixXcd leftKernel{};     // interiorDim x leftNullity
  std::vector<std::vector<long>> integerBasis{};
  Eigen::MatrixXcd modeRow{};        // nullity x |adjacentKept|
  Eigen::MatrixXcd modeCol{};        // |adjacentKept| x nullity
  Eigen::MatrixXcd modeBlock{};      // nullity x nullity
  double solveResidual{0.0};
  double compatibilityResidual{0.0};
  double conditioning{kNaN};
  cd interiorDet{kNaN, 0.0};
  bool detValid{false};
  bool eliminationCertified{true};
  // Whether the exact integer topological kernel was computed. False on the
  // matrix path and on integer-kernel overflow, where an empty integerBasis
  // means "not measured" rather than "measured zero".
  bool integerKernelMeasured{false};
  std::string note{};
};

// --------------------------------------------------------------------------
// Construction
// --------------------------------------------------------------------------

void RecursiveQuotient::initMatrix(const std::vector<cd> &op, int dim,
                                   const std::vector<cd> &weights,
                                   const std::vector<std::vector<int>> &components,
                                   const Options &options) {
  if (dim < 0) throw std::invalid_argument("RecursiveQuotient: negative dim");
  const Eigen::MatrixXcd dense = toMatrix(op, dim, dim, "RecursiveQuotient op");
  op_ = dense.sparseView();
  op_.makeCompressed();
  opNorm_ = dense.norm();
  dim_ = dim;
  options_ = options;
  if (weights.empty()) {
    weights_ = Eigen::VectorXcd::Ones(dim);
  } else {
    if (static_cast<int>(weights.size()) != dim)
      throw std::invalid_argument(
          "RecursiveQuotient: weights length must equal dim");
    weights_ = Eigen::VectorXcd(dim);
    for (int i = 0; i < dim; ++i) weights_(i) = weights[static_cast<std::size_t>(i)];
  }
  if (components.empty() && dim > 0)
    throw std::invalid_argument("RecursiveQuotient: no components given");
  components_.clear();
  std::vector<bool> covered(static_cast<std::size_t>(dim), false);
  for (const auto &component : components) {
    std::set<int> unique;
    for (int index : component) {
      if (index < 0 || index >= dim)
        throw std::invalid_argument(
            "RecursiveQuotient: component index out of range");
      unique.insert(index);
      covered[static_cast<std::size_t>(index)] = true;
    }
    components_.emplace_back(unique.begin(), unique.end());
  }
  for (int i = 0; i < dim; ++i)
    if (!covered[static_cast<std::size_t>(i)])
      throw std::invalid_argument(
          "RecursiveQuotient: component union does not cover every cell");
  if (provenance_.size() != static_cast<std::size_t>(dim)) {
    provenance_.resize(static_cast<std::size_t>(dim));
    for (int i = 0; i < dim; ++i)
      provenance_[static_cast<std::size_t>(i)] =
          "coord(" + std::to_string(i) + ")";
  }
  classify();
  detectRegime();
}

RecursiveQuotient RecursiveQuotient::overMatrix(
    const std::vector<cd> &op, int dim, const std::vector<cd> &weights,
    const std::vector<std::vector<int>> &components, const Options &options) {
  RecursiveQuotient quotient;
  quotient.initMatrix(op, dim, weights, components, options);
  return quotient;
}

RecursiveQuotient RecursiveQuotient::overCells(
    std::shared_ptr<Spacetime> st, int degree,
    const std::vector<std::vector<std::vector<std::uint64_t>>> &componentCells,
    const Options &options, std::shared_ptr<AnalyticCache> cache) {
  if (!st) throw std::invalid_argument("RecursiveQuotient: null spacetime");
  if (degree < 0) throw std::invalid_argument("RecursiveQuotient: degree < 0");

  RecursiveQuotient quotient;
  quotient.st_ = st;
  quotient.cache_ = std::move(cache);
  quotient.degree_ = degree;

  const ChainComplex cc = ChainComplex::fromSpacetime(*st);
  HodgeLaplacian hodge(st);

  // Canonical cell order: the ChainComplex column order (sorted vertex-id
  // tuples), which L_k is indexed over. A vertex carried by no simplex is not
  // a 0-cell and so is not a coordinate.
  const std::vector<std::vector<std::uint64_t>> cells =
      cc.kSimplexVertices(degree);
  const int dim = static_cast<int>(cells.size());

  std::map<std::vector<std::uint64_t>, int> cellIndex;
  for (int i = 0; i < dim; ++i) cellIndex[cells[static_cast<std::size_t>(i)]] = i;

  std::vector<std::vector<int>> indexComponents;
  for (const auto &component : componentCells) {
    std::vector<int> indices;
    for (const auto &cell : component) {
      std::vector<std::uint64_t> key(cell);
      std::sort(key.begin(), key.end());
      const auto found = cellIndex.find(key);
      if (found == cellIndex.end())
        throw std::invalid_argument("RecursiveQuotient: unknown cell " +
                                    cellName(key));
      indices.push_back(found->second);
    }
    indexComponents.push_back(std::move(indices));
  }

  Options resolved = options;
  for (const auto &cell : options.selectedInteriorCells) {
    std::vector<std::uint64_t> key(cell);
    std::sort(key.begin(), key.end());
    const auto found = cellIndex.find(key);
    if (found == cellIndex.end())
      throw std::invalid_argument(
          "RecursiveQuotient: unknown selected cell " + cellName(key));
    resolved.selectedInteriorIndices.push_back(found->second);
  }

  quotient.cellVertices_ = cells;
  quotient.provenance_.resize(static_cast<std::size_t>(dim));
  for (int i = 0; i < dim; ++i)
    quotient.provenance_[static_cast<std::size_t>(i)] =
        cellName(cells[static_cast<std::size_t>(i)]);

  quotient.hasBoundary_ = true;
  if (degree >= 1) {
    quotient.boundaryK_ = cc.boundaryMatrix(degree);
    quotient.boundaryKRows_ = static_cast<int>(cc.numSimplices(degree - 1));
  } else {
    // At k = 0 there are no (-1)-chains, so the boundary block is empty and
    // the interior zero-mode condition is the coboundary one alone.
    quotient.boundaryK_.clear();
    quotient.boundaryKRows_ = 0;
  }
  quotient.boundaryK1_ = cc.boundaryMatrix(degree + 1);
  quotient.boundaryK1Cols_ = static_cast<int>(cc.numSimplices(degree + 1));

  // W_k for every degree, degree zero included (there W_0 = I).
  quotient.initMatrix(hodge.laplacian(degree), dim, hodge.weights(degree),
                      indexComponents, resolved);
  return quotient;
}

RecursiveQuotient RecursiveQuotient::overVertexSupports(
    std::shared_ptr<Spacetime> st, int degree,
    const std::vector<std::vector<std::uint64_t>> &componentVertexSupports,
    const Options &options, std::shared_ptr<AnalyticCache> cache) {
  if (!st) throw std::invalid_argument("RecursiveQuotient: null spacetime");
  const ChainComplex cc = ChainComplex::fromSpacetime(*st);
  // The same canonical ChainComplex column order overCells uses.
  const std::vector<std::vector<std::uint64_t>> cells =
      cc.kSimplexVertices(degree);

  std::vector<std::set<std::uint64_t>> supports;
  for (const auto &support : componentVertexSupports)
    supports.emplace_back(support.begin(), support.end());

  std::vector<std::vector<std::vector<std::uint64_t>>> componentCells(
      supports.size());
  std::vector<std::vector<std::uint64_t>> residual;
  for (const auto &cell : cells) {
    bool claimed = false;
    for (std::size_t s = 0; s < supports.size(); ++s) {
      bool inside = true;
      for (const std::uint64_t vertex : cell)
        if (supports[s].find(vertex) == supports[s].end()) {
          inside = false;
          break;
        }
      if (inside) {
        componentCells[s].push_back(cell);
        claimed = true;
      }
    }
    if (!claimed) residual.push_back(cell);
  }
  if (!residual.empty()) componentCells.push_back(std::move(residual));
  return overCells(std::move(st), degree, componentCells, options,
                   std::move(cache));
}

// --------------------------------------------------------------------------
// Classification and regime
// --------------------------------------------------------------------------

void RecursiveQuotient::classify() {
  claimants_.assign(static_cast<std::size_t>(dim_), {});
  for (std::size_t v = 0; v < components_.size(); ++v)
    for (const int index : components_[v])
      claimants_[static_cast<std::size_t>(index)].push_back(
          static_cast<int>(v));

  // Coupling pattern (either orientation counts).
  std::vector<std::set<int>> neighbors(static_cast<std::size_t>(dim_));
  for (int outer = 0; outer < op_.outerSize(); ++outer)
    for (Eigen::SparseMatrix<cd>::InnerIterator it(op_, outer); it; ++it) {
      if (it.row() == it.col() || it.value() == cd(0.0, 0.0)) continue;
      neighbors[static_cast<std::size_t>(it.row())].insert(
          static_cast<int>(it.col()));
      neighbors[static_cast<std::size_t>(it.col())].insert(
          static_cast<int>(it.row()));
    }
  // A pencil level couples coordinates through M as well: a cell whose
  // metric row leaves its component is an interface cell of the pencil.
  if (pencil_)
    for (int outer = 0; outer < pencilMetric_.outerSize(); ++outer)
      for (Eigen::SparseMatrix<cd>::InnerIterator it(pencilMetric_, outer); it; ++it) {
        if (it.row() == it.col() || it.value() == cd(0.0, 0.0)) continue;
        neighbors[static_cast<std::size_t>(it.row())].insert(
            static_cast<int>(it.col()));
        neighbors[static_cast<std::size_t>(it.col())].insert(
            static_cast<int>(it.row()));
      }

  std::vector<std::set<int>> memberSets;
  memberSets.reserve(components_.size());
  for (const auto &component : components_)
    memberSets.emplace_back(component.begin(), component.end());

  const std::set<int> selected(options_.selectedInteriorIndices.begin(),
                               options_.selectedInteriorIndices.end());
  for (const int index : selected)
    if (index < 0 || index >= dim_)
      throw std::invalid_argument(
          "RecursiveQuotient: selected index out of range");

  interior_.assign(components_.size(), {});
  interfaceIndices_.clear();
  keptKinds_.clear();
  keptOwner_.clear();
  interfacePosition_.assign(static_cast<std::size_t>(dim_), -1);

  for (int i = 0; i < dim_; ++i) {
    const auto &owners = claimants_[static_cast<std::size_t>(i)];
    bool interiorCell = owners.size() == 1;
    if (interiorCell) {
      const std::set<int> &cells = memberSets[static_cast<std::size_t>(owners[0])];
      for (const int j : neighbors[static_cast<std::size_t>(i)])
        if (cells.find(j) == cells.end()) {
          interiorCell = false;
          break;
        }
    }
    const bool keepSelected = selected.find(i) != selected.end();
    if (interiorCell && !keepSelected) {
      interior_[static_cast<std::size_t>(owners[0])].push_back(i);
    } else {
      interfacePosition_[static_cast<std::size_t>(i)] =
          static_cast<int>(interfaceIndices_.size());
      interfaceIndices_.push_back(i);
      keptKinds_.push_back(interiorCell && keepSelected
                               ? RetainedCoordinateKind::Selected
                               : RetainedCoordinateKind::Interface);
      keptOwner_.push_back(owners.empty() ? 0 : owners[0]);
    }
  }

  // Cache-kind qualifier: two instances may share a component vertex set (the
  // AnalyticCache key) yet classify its cells differently under a different
  // partition, degree or rank tolerance. The fingerprint folds the
  // classification in (splitmix64-style mixing).
  auto mix = [](std::uint64_t h, std::uint64_t v) {
    h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    h *= 0xbf58476d1ce4e5b9ULL;
    return h ^ (h >> 31);
  };
  std::uint64_t fingerprint = mix(0x243f6a8885a308d3ULL,
                                  static_cast<std::uint64_t>(dim_));
  fingerprint = mix(fingerprint, static_cast<std::uint64_t>(degree_ + 2));
  std::uint64_t bits = 0;
  static_assert(sizeof(bits) == sizeof(options_.rankTolerance));
  std::memcpy(&bits, &options_.rankTolerance, sizeof(bits));
  fingerprint = mix(fingerprint, bits);
  std::memcpy(&bits, &options_.tolerance, sizeof(bits));
  fingerprint = mix(fingerprint, bits);
  fingerprint = mix(fingerprint,
                    static_cast<std::uint64_t>(options_.denseCrossover));
  for (const int kept : interfaceIndices_)
    fingerprint = mix(fingerprint, static_cast<std::uint64_t>(kept) + 1);
  for (const auto &interior : interior_) {
    fingerprint = mix(fingerprint, interior.size() + 0x51ULL);
    for (const int index : interior)
      fingerprint = mix(fingerprint, static_cast<std::uint64_t>(index) + 3);
  }
  partitionFingerprint_ = fingerprint;
}

void RecursiveQuotient::detectRegime() {
  // Every branch below measures the operator. On a Lorentzian complex
  // L_0 = d_1 W_1^-1 d_1^T is routinely indefinite.
  if (pencil_) {
    // A pencil level (Ã, M) is regime-classified by complex symmetry of both
    // matrices (M L = (M L)^T with M complex symmetric) rather than by
    // Hermiticity. A broken symmetry is the non-normal regime.
    const double tolP = std::max(options_.tolerance, 1e-12);
    const Eigen::SparseMatrix<cd> opT = Eigen::SparseMatrix<cd>(op_.transpose());
    const Eigen::SparseMatrix<cd> mT = Eigen::SparseMatrix<cd>(pencilMetric_.transpose());
    const double opDefect = Eigen::SparseMatrix<cd>(op_ - opT).norm() /
                            std::max(op_.norm(), 1e-300);
    const double mDefect = Eigen::SparseMatrix<cd>(pencilMetric_ - mT).norm() /
                           std::max(pencilMetric_.norm(), 1e-300);
    regime_ = (opDefect <= tolP && mDefect <= tolP)
                  ? CertificateRegime::ComplexSymmetricPencil
                  : CertificateRegime::NonNormal;
    return;
  }
  // Hermiticity against the carried metric: WL vs (WL)^dagger.
  const Eigen::SparseMatrix<cd> weighted = weights_.asDiagonal() * op_;
  const Eigen::SparseMatrix<cd> adjoint =
      Eigen::SparseMatrix<cd>(weighted.adjoint());
  const double scale = std::max(weighted.norm(), 1e-300);
  const double hermiticity =
      Eigen::SparseMatrix<cd>(weighted - adjoint).norm() / scale;
  const double tol = std::max(options_.tolerance, 1e-12);
  if (hermiticity > tol) {
    regime_ = CertificateRegime::NonNormal;
    return;
  }
  bool positiveMetric = true;
  for (int i = 0; i < dim_; ++i)
    if (!(weights_(i).real() > 0.0) || std::abs(weights_(i).imag()) > tol) {
      positiveMetric = false;
      break;
    }
  if (positiveMetric && dim_ < options_.denseCrossover) {
    // Pivoted LDLT decides semidefiniteness far cheaper than an eigensolve.
    const Eigen::MatrixXcd dense = Eigen::MatrixXcd(weighted);
    const Eigen::MatrixXcd hermitian = 0.5 * (dense + dense.adjoint());
    Eigen::LDLT<Eigen::MatrixXcd> ldlt(hermitian);
    if (ldlt.info() == Eigen::Success &&
        (dim_ == 0 || ldlt.vectorD().real().minCoeff() >=
                          -tol * std::max(scale, 1.0))) {
      regime_ = CertificateRegime::PositiveSemidefinite;
      return;
    }
  }
  // Hermitian but signed metric, verified-indefinite, or too large to verify:
  // the weaker stationarity regime.
  regime_ = CertificateRegime::HermitianIndefinite;
}

// --------------------------------------------------------------------------
// Interior solves
// --------------------------------------------------------------------------

const std::vector<int> &RecursiveQuotient::interiorIndices(int component) const {
  if (component < 0 || component >= componentCount())
    throw std::out_of_range("RecursiveQuotient: component out of range");
  return interior_[static_cast<std::size_t>(component)];
}

std::vector<std::uint64_t> RecursiveQuotient::componentVertexIds(
    int component) const {
  std::set<std::uint64_t> ids;
  for (const int index : components_[static_cast<std::size_t>(component)])
    if (index >= 0 &&
        static_cast<std::size_t>(index) < cellVertices_.size())
      for (const std::uint64_t id : cellVertices_[static_cast<std::size_t>(index)])
        ids.insert(id);
  return {ids.begin(), ids.end()};
}

std::vector<long> RecursiveQuotient::integerKernelStack(int component,
                                                        int *rows) const {
  // Stacked integer conditions for a combinatorial interior zero mode:
  // boundary rows d_k[:, I] and coboundary rows d_{k+1}[I, :]^T.
  const auto &interior = interior_[static_cast<std::size_t>(component)];
  const int m = static_cast<int>(interior.size());
  const int rowsK = boundaryKRows_;
  const int rowsK1 = boundaryK1Cols_;
  std::vector<long> stack(static_cast<std::size_t>(rowsK + rowsK1) *
                              static_cast<std::size_t>(m),
                          0);
  for (int r = 0; r < rowsK; ++r)
    for (int c = 0; c < m; ++c)
      stack[static_cast<std::size_t>(r) * m + c] =
          boundaryK_[static_cast<std::size_t>(r) * dim_ +
                     interior[static_cast<std::size_t>(c)]];
  // d_{k+1} is dim_ x |C_{k+1}|; its transpose rows are the columns.
  for (int r = 0; r < rowsK1; ++r)
    for (int c = 0; c < m; ++c)
      stack[static_cast<std::size_t>(rowsK + r) * m + c] =
          boundaryK1_[static_cast<std::size_t>(
                          interior[static_cast<std::size_t>(c)]) *
                          rowsK1 +
                      r];
  *rows = rowsK + rowsK1;
  return stack;
}

std::shared_ptr<RecursiveQuotient::ComponentSolve>
RecursiveQuotient::computeSolve(int component, cd lambda) const {
  auto solve = std::make_shared<ComponentSolve>();
  solve->lambda = lambda;
  const auto &interior = interior_[static_cast<std::size_t>(component)];
  const int m = static_cast<int>(interior.size());
  solve->interiorDim = m;

  // Adjacent kept positions (columns of L_IK with any coupling).
  std::vector<int> interiorPos(static_cast<std::size_t>(dim_), -1);
  for (int i = 0; i < m; ++i)
    interiorPos[static_cast<std::size_t>(interior[static_cast<std::size_t>(i)])] = i;
  std::set<int> adjacent;
  for (const int fine : interior)
    for (Eigen::SparseMatrix<cd>::InnerIterator it(op_, fine); it; ++it) {
      // column `fine`: rows coupled into the interior cell
      const int other = static_cast<int>(it.row());
      const int kept = interfacePosition_[static_cast<std::size_t>(other)];
      if (kept >= 0 && it.value() != cd(0.0, 0.0)) adjacent.insert(kept);
    }
  // and the transposed couplings (rows `fine`): iterate all columns once.
  for (int outer = 0; outer < op_.outerSize(); ++outer)
    for (Eigen::SparseMatrix<cd>::InnerIterator it(op_, outer); it; ++it) {
      if (it.value() == cd(0.0, 0.0)) continue;
      const int row = static_cast<int>(it.row());
      const int col = static_cast<int>(it.col());
      if (interiorPos[static_cast<std::size_t>(row)] >= 0) {
        const int kept = interfacePosition_[static_cast<std::size_t>(col)];
        if (kept >= 0) adjacent.insert(kept);
      }
    }
  if (pencil_ && lambda != cd(0.0, 0.0))
    for (int outer = 0; outer < pencilMetric_.outerSize(); ++outer)
      for (Eigen::SparseMatrix<cd>::InnerIterator it(pencilMetric_, outer); it; ++it) {
        if (it.value() == cd(0.0, 0.0)) continue;
        const int row = static_cast<int>(it.row());
        const int col = static_cast<int>(it.col());
        if (interiorPos[static_cast<std::size_t>(row)] >= 0) {
          const int kept = interfacePosition_[static_cast<std::size_t>(col)];
          if (kept >= 0) adjacent.insert(kept);
        }
        if (interiorPos[static_cast<std::size_t>(col)] >= 0) {
          const int kept = interfacePosition_[static_cast<std::size_t>(row)];
          if (kept >= 0) adjacent.insert(kept);
        }
      }
  solve->adjacentKept.assign(adjacent.begin(), adjacent.end());
  const int a = static_cast<int>(solve->adjacentKept.size());

  if (m == 0) {
    solve->X = Eigen::MatrixXcd::Zero(0, a);
    solve->contribution = Eigen::MatrixXcd::Zero(a, a);
    solve->interiorDet = cd(1.0, 0.0);
    solve->detValid = true;
    return solve;
  }

  // Interior/coupling blocks in local order: sparse triplets for the interior
  // block (dense staging only below the crossover), dense skinny couplings
  // (m x a and a x m).
  std::vector<Eigen::Triplet<cd>> interiorTriplets;
  Eigen::MatrixXcd loadBlock = Eigen::MatrixXcd::Zero(m, a);      // L_IK
  Eigen::MatrixXcd keptBlock = Eigen::MatrixXcd::Zero(a, m);      // L_KI
  std::vector<int> keptFine(static_cast<std::size_t>(a));
  for (int j = 0; j < a; ++j)
    keptFine[static_cast<std::size_t>(j)] =
        interfaceIndices_[static_cast<std::size_t>(
            solve->adjacentKept[static_cast<std::size_t>(j)])];
  std::vector<int> keptLocal(static_cast<std::size_t>(dim_), -1);
  for (int j = 0; j < a; ++j)
    keptLocal[static_cast<std::size_t>(keptFine[static_cast<std::size_t>(j)])] = j;

  for (int outer = 0; outer < op_.outerSize(); ++outer)
    for (Eigen::SparseMatrix<cd>::InnerIterator it(op_, outer); it; ++it) {
      const int row = static_cast<int>(it.row());
      const int col = static_cast<int>(it.col());
      const int rowInterior = interiorPos[static_cast<std::size_t>(row)];
      const int colInterior = interiorPos[static_cast<std::size_t>(col)];
      if (rowInterior >= 0 && colInterior >= 0)
        interiorTriplets.emplace_back(rowInterior, colInterior, it.value());
      else if (rowInterior >= 0 && keptLocal[static_cast<std::size_t>(col)] >= 0)
        loadBlock(rowInterior, keptLocal[static_cast<std::size_t>(col)]) =
            it.value();
      else if (colInterior >= 0 && keptLocal[static_cast<std::size_t>(row)] >= 0)
        keptBlock(keptLocal[static_cast<std::size_t>(row)], colInterior) =
            it.value();
    }
  if (pencil_) {
    // P(lambda) = A - lambda M on every block: interior, load (I x K), and kept (K x I).
    for (int outer = 0; outer < pencilMetric_.outerSize(); ++outer)
      for (Eigen::SparseMatrix<cd>::InnerIterator it(pencilMetric_, outer); it; ++it) {
        const cd shift = -lambda * it.value();
        if (shift == cd(0.0, 0.0)) continue;
        const int row = static_cast<int>(it.row());
        const int col = static_cast<int>(it.col());
        const int rowInterior = interiorPos[static_cast<std::size_t>(row)];
        const int colInterior = interiorPos[static_cast<std::size_t>(col)];
        if (rowInterior >= 0 && colInterior >= 0)
          interiorTriplets.emplace_back(rowInterior, colInterior, shift);
        else if (rowInterior >= 0 && keptLocal[static_cast<std::size_t>(col)] >= 0)
          loadBlock(rowInterior, keptLocal[static_cast<std::size_t>(col)]) += shift;
        else if (colInterior >= 0 && keptLocal[static_cast<std::size_t>(row)] >= 0)
          keptBlock(keptLocal[static_cast<std::size_t>(row)], colInterior) += shift;
      }
  } else {
    for (int i = 0; i < m; ++i)
      interiorTriplets.emplace_back(i, i, -lambda);
  }
  Eigen::SparseMatrix<cd> sparseShifted(m, m);
  sparseShifted.setFromTriplets(interiorTriplets.begin(),
                                interiorTriplets.end());
  sparseShifted.prune([](int, int, const cd &value) {
    return value != cd(0.0, 0.0);
  });
  sparseShifted.makeCompressed();
  const double loadScale = std::max(loadBlock.norm(), 1e-300);

  // Exact integer topological zero modes (unshifted, spacetime path), via
  // IntegerLinalg.
  if (hasBoundary_ && lambda == cd(0.0, 0.0)) {
    int rows = 0;
    const std::vector<long> stack = integerKernelStack(component, &rows);
    try {
      solve->integerBasis = integerNullspace(stack, rows, m);
      solve->integerKernelMeasured = true;
    } catch (const std::overflow_error &) {
      solve->integerBasis.clear();
      solve->integerKernelMeasured = false;
      solve->note = "integer kernel overflow";
    }
  }

  // Numerical kernel and factor solve. Below the crossover an LU factor solve
  // runs first; the rank-revealing SVD path is an escalation taken when the
  // pivots flag rank deficiency or the measured residual fails.
  if (m < options_.denseCrossover) {
    const Eigen::MatrixXcd shifted = Eigen::MatrixXcd(sparseShifted);
    {
      Eigen::PartialPivLU<Eigen::MatrixXcd> fastLU(shifted);
      const Eigen::MatrixXcd &luMatrix = fastLU.matrixLU();
      double diagMin = kInf;
      double diagMax = 0.0;
      for (int i = 0; i < m; ++i) {
        diagMin = std::min(diagMin, std::abs(luMatrix(i, i)));
        diagMax = std::max(diagMax, std::abs(luMatrix(i, i)));
      }
      if (m > 0 && diagMin > options_.rankTolerance * std::max(diagMax, 1e-300)) {
        const Eigen::MatrixXcd candidate = fastLU.solve(loadBlock);
        const double residual =
            (shifted * candidate - loadBlock).norm() / loadScale;
        if (residual <= std::max(options_.tolerance, 1e-12)) {
          solve->X = candidate;
          solve->conditioning = diagMax / diagMin;
          solve->interiorDet = fastLU.determinant();
          solve->detValid = true;
          solve->solveResidual = residual;
          solve->contribution = keptBlock * solve->X;
          return solve;
        }
      }
    }
    Eigen::JacobiSVD<Eigen::MatrixXcd> svd(
        shifted, Eigen::ComputeThinU | Eigen::ComputeThinV);
    const auto &sigma = svd.singularValues();
    const double sigmaMax = sigma.size() ? sigma(0) : 0.0;
    const double cut = options_.rankTolerance * std::max(sigmaMax, 1e-300);
    int rank = 0;
    for (Eigen::Index i = 0; i < sigma.size(); ++i)
      if (sigma(i) > cut) ++rank;
    const int nullity = m - rank;
    solve->conditioning =
        rank > 0 ? sigmaMax / sigma(rank - 1) : kNaN;
    if (nullity > 0) {
      solve->rightKernel = svd.matrixV().rightCols(nullity);
      solve->leftKernel = svd.matrixU().rightCols(nullity);
    }
    if (nullity == 0) {
      Eigen::PartialPivLU<Eigen::MatrixXcd> lu(shifted);
      solve->X = lu.solve(loadBlock);
      solve->interiorDet = lu.determinant();
      solve->detValid = true;
    } else {
      // Supported elimination: retain the kernel, eliminate the Euclidean
      // orthocomplement Q with adjoint test space — the Moore-Penrose response
      // in the Hermitian regimes. No diagonal regularizer.
      const Eigen::MatrixXcd &kernel = solve->rightKernel;
      Eigen::HouseholderQR<Eigen::MatrixXcd> qr(kernel);
      const Eigen::MatrixXcd full =
          qr.householderQ() * Eigen::MatrixXcd::Identity(m, m);
      const Eigen::MatrixXcd complement = full.rightCols(m - nullity);
      const Eigen::MatrixXcd eliminated =
          complement.adjoint() * shifted * complement;
      Eigen::PartialPivLU<Eigen::MatrixXcd> lu(eliminated);
      // A defective shifted block (Jordan structure meeting the kernel) makes
      // the complement block singular; refuse rather than regularize.
      const Eigen::MatrixXcd &luMatrix = lu.matrixLU();
      double diagMin = kInf;
      double diagMax = 0.0;
      for (int i = 0; i < m - nullity; ++i) {
        diagMin = std::min(diagMin, std::abs(luMatrix(i, i)));
        diagMax = std::max(diagMax, std::abs(luMatrix(i, i)));
      }
      if (m - nullity > 0 &&
          diagMin <= options_.rankTolerance * std::max(diagMax, 1e-300)) {
        solve->eliminationCertified = false;
        solve->note = "defective interior block at the shift";
        solve->X = Eigen::MatrixXcd::Zero(m, a);
        solve->contribution = Eigen::MatrixXcd::Zero(a, a);
        solve->modeRow = Eigen::MatrixXcd::Zero(nullity, a);
        solve->modeCol = Eigen::MatrixXcd::Zero(a, nullity);
        solve->modeBlock = Eigen::MatrixXcd::Zero(nullity, nullity);
        solve->interiorDet = cd(0.0, 0.0);
        solve->detValid = true;
        solve->solveResidual = kInf;
        return solve;
      }
      const Eigen::MatrixXcd solvedLoad =
          lu.solve(complement.adjoint() * loadBlock);
      solve->X = complement * solvedLoad;
      solve->modeRow =
          kernel.adjoint() * loadBlock -
          (kernel.adjoint() * shifted * complement) * solvedLoad;
      const Eigen::MatrixXcd solvedKernelSide =
          lu.solve(complement.adjoint() * shifted * kernel);
      solve->modeCol =
          keptBlock * kernel - (keptBlock * complement) * solvedKernelSide;
      solve->modeBlock =
          kernel.adjoint() * shifted * kernel -
          (kernel.adjoint() * shifted * complement) * solvedKernelSide;
      solve->interiorDet = cd(0.0, 0.0);
      solve->detValid = true;
      // Left-kernel compatibility, the exact solvability condition; ~0
      // automatically in the positive regime.
      solve->compatibilityResidual =
          (solve->leftKernel.adjoint() * loadBlock).norm() / loadScale;
      // Residual on the eliminated subspace only; kernel rows are couplings,
      // not residuals.
      solve->solveResidual =
          (complement.adjoint() * (shifted * solve->X - loadBlock)).norm() /
          loadScale;
      solve->contribution = keptBlock * solve->X;
      return solve;
    }
  } else {
    // Sparse factor solve at scale. SparseLU pivoting reveals a singular
    // block, which needs the dense rank-revealing path; refused above the
    // crossover rather than approximated.
    Eigen::SparseLU<Eigen::SparseMatrix<cd>> lu;
    lu.compute(sparseShifted);
    if (lu.info() != Eigen::Success) {
      solve->eliminationCertified = false;
      solve->note =
          "singular interior block at/above the dense crossover: the "
          "rank-revealing kernel path refuses at scale";
      solve->X = Eigen::MatrixXcd::Zero(m, a);
      solve->contribution = Eigen::MatrixXcd::Zero(a, a);
      solve->solveResidual = kInf;
      return solve;
    }
    solve->X = lu.solve(loadBlock);
    solve->interiorDet = lu.determinant();
    solve->detValid = true;
  }

  solve->solveResidual =
      (sparseShifted * solve->X - loadBlock).norm() / loadScale;
  solve->contribution = keptBlock * solve->X;
  return solve;
}

std::shared_ptr<RecursiveQuotient::ComponentSolve>
RecursiveQuotient::componentSolve(int component) const {
  if (component < 0 || component >= componentCount())
    throw std::out_of_range("RecursiveQuotient: component out of range");
  if (solves_.empty())
    solves_.assign(components_.size(), nullptr);
  auto &slot = solves_[static_cast<std::size_t>(component)];
  if (slot) return slot;

  const std::string kind =
      "recursive-quotient-static#" + std::to_string(partitionFingerprint_);
  if (cache_ && st_) {
    const auto key = componentVertexIds(component);
    if (auto cached = cache_->fetch(key, kind, degree_)) {
      slot = std::static_pointer_cast<ComponentSolve>(cached);
      return slot;
    }
    slot = computeSolve(component, cd(0.0, 0.0));
    const double residual =
        std::max(slot->solveResidual, slot->compatibilityResidual);
    cache_->store(key, kind, degree_, slot,
                  Certificate::structureExact(
                      CertificateDomain::Static, regime_, residual,
                      slot->conditioning, options_.tolerance));
    return slot;
  }
  slot = computeSolve(component, cd(0.0, 0.0));
  return slot;
}

const std::vector<std::shared_ptr<RecursiveQuotient::ComponentSolve>> &
RecursiveQuotient::shiftedSolves(cd lambda) const {
  const std::pair<double, double> key{lambda.real(), lambda.imag()};
  auto found = shifted_.find(key);
  if (found != shifted_.end()) return found->second;
  std::vector<std::shared_ptr<ComponentSolve>> solves;
  solves.reserve(components_.size());
  for (int component = 0; component < componentCount(); ++component)
    solves.push_back(computeSolve(component, lambda));
  return shifted_.emplace(key, std::move(solves)).first->second;
}

// --------------------------------------------------------------------------
// Reads
// --------------------------------------------------------------------------

RecursiveQuotient::InteriorNullspaceRead RecursiveQuotient::interiorNullspace(
    int component) const {
  const auto solve = componentSolve(component);
  InteriorNullspaceRead read;
  read.component = component;
  read.nullity = static_cast<std::size_t>(solve->rightKernel.cols());
  read.integerNullity = solve->integerBasis.size();
  read.integerBasis = solve->integerBasis;
  read.integerNullityMeasured = solve->integerKernelMeasured;
  // The numerical kernel of the weighted interior block and the exact integer
  // topological nullity are different quantities, so the discrepancy is
  // recorded; it stays NaN when no integer nullity was measured.
  read.nullityDiscrepancy =
      solve->integerKernelMeasured
          ? static_cast<double>(static_cast<long long>(read.nullity) -
                                static_cast<long long>(read.integerNullity))
          : std::numeric_limits<double>::quiet_NaN();
  read.kernelBasis = toFlat(solve->rightKernel);
  read.leftKernelBasis = toFlat(solve->leftKernel);

  // Measured residual of the returned numerical basis.
  double residual = 0.0;
  const auto &interior = interior_[static_cast<std::size_t>(component)];
  const int m = static_cast<int>(interior.size());
  if (m > 0 && solve->rightKernel.cols() > 0) {
    Eigen::MatrixXcd block = Eigen::MatrixXcd::Zero(m, m);
    std::vector<int> interiorPos(static_cast<std::size_t>(dim_), -1);
    for (int i = 0; i < m; ++i)
      interiorPos[static_cast<std::size_t>(
          interior[static_cast<std::size_t>(i)])] = i;
    for (int outer = 0; outer < op_.outerSize(); ++outer)
      for (Eigen::SparseMatrix<cd>::InnerIterator it(op_, outer); it; ++it) {
        const int row = interiorPos[static_cast<std::size_t>(it.row())];
        const int col = interiorPos[static_cast<std::size_t>(it.col())];
        if (row >= 0 && col >= 0) block(row, col) = it.value();
      }
    residual = (block * solve->rightKernel).norm() /
               std::max(block.norm(), 1e-300);
  }
  read.certificate = Certificate::structureExact(
      CertificateDomain::Static, regime_, residual, solve->conditioning,
      options_.tolerance);
  return read;
}

const RecursiveQuotient::StaticReductionRead &
RecursiveQuotient::staticReduction() const {
  if (static_) return *static_;

  const int kept = static_cast<int>(interfaceIndices_.size());
  std::vector<std::shared_ptr<ComponentSolve>> solves;
  solves.reserve(components_.size());
  int modeCount = 0;
  for (int component = 0; component < componentCount(); ++component) {
    solves.push_back(componentSolve(component));
    modeCount += static_cast<int>(solves.back()->rightKernel.cols());
  }
  const int reduced = kept + modeCount;

  StaticReductionRead read;
  read.interfaceIndices = interfaceIndices_;
  Eigen::MatrixXcd effective = Eigen::MatrixXcd::Zero(reduced, reduced);

  // Kept block of the fine operator.
  for (int outer = 0; outer < op_.outerSize(); ++outer)
    for (Eigen::SparseMatrix<cd>::InnerIterator it(op_, outer); it; ++it) {
      const int row = interfacePosition_[static_cast<std::size_t>(it.row())];
      const int col = interfacePosition_[static_cast<std::size_t>(it.col())];
      if (row >= 0 && col >= 0) effective(row, col) += it.value();
    }

  double solveResidual = 0.0;
  double compatibilityResidual = 0.0;
  bool certified = true;
  int modeOffset = kept;
  for (int component = 0; component < componentCount(); ++component) {
    const ComponentSolve &solve = *solves[static_cast<std::size_t>(component)];
    const int a = static_cast<int>(solve.adjacentKept.size());
    for (int i = 0; i < a; ++i)
      for (int j = 0; j < a; ++j)
        effective(solve.adjacentKept[static_cast<std::size_t>(i)],
                  solve.adjacentKept[static_cast<std::size_t>(j)]) -=
            solve.contribution(i, j);
    const int modes = static_cast<int>(solve.rightKernel.cols());
    for (int t = 0; t < modes; ++t) {
      for (int j = 0; j < a; ++j) {
        effective(modeOffset + t,
                  solve.adjacentKept[static_cast<std::size_t>(j)]) =
            solve.modeRow(t, j);
        effective(solve.adjacentKept[static_cast<std::size_t>(j)],
                  modeOffset + t) = solve.modeCol(j, t);
      }
      for (int u = 0; u < modes; ++u)
        effective(modeOffset + t, modeOffset + u) = solve.modeBlock(t, u);
    }
    modeOffset += modes;
    solveResidual = std::max(solveResidual, solve.solveResidual);
    compatibilityResidual =
        std::max(compatibilityResidual, solve.compatibilityResidual);
    certified = certified && solve.eliminationCertified;
  }

  // Coordinates with provenance.
  read.coordinates.reserve(static_cast<std::size_t>(reduced));
  for (int position = 0; position < kept; ++position) {
    RetainedCoordinate coordinate;
    coordinate.kind = keptKinds_[static_cast<std::size_t>(position)];
    coordinate.component = keptOwner_[static_cast<std::size_t>(position)];
    coordinate.fineIndex = interfaceIndices_[static_cast<std::size_t>(position)];
    coordinate.embedding.assign(static_cast<std::size_t>(dim_), cd(0.0, 0.0));
    coordinate.embedding[static_cast<std::size_t>(coordinate.fineIndex)] =
        cd(1.0, 0.0);
    coordinate.provenance =
        provenance_[static_cast<std::size_t>(coordinate.fineIndex)];
    read.coordinates.push_back(std::move(coordinate));
  }
  for (int component = 0; component < componentCount(); ++component) {
    const ComponentSolve &solve = *solves[static_cast<std::size_t>(component)];
    const auto &interior = interior_[static_cast<std::size_t>(component)];
    for (int t = 0; t < solve.rightKernel.cols(); ++t) {
      RetainedCoordinate coordinate;
      coordinate.kind = RetainedCoordinateKind::Harmonic;
      coordinate.component = component;
      coordinate.fineIndex = -1;
      coordinate.embedding.assign(static_cast<std::size_t>(dim_), cd(0.0, 0.0));
      for (int i = 0; i < solve.interiorDim; ++i)
        coordinate.embedding[static_cast<std::size_t>(
            interior[static_cast<std::size_t>(i)])] = solve.rightKernel(i, t);
      coordinate.provenance = "harmonic[c" + std::to_string(component) + "#" +
                              std::to_string(t) + "]";
      read.coordinates.push_back(std::move(coordinate));
    }
  }

  read.effectiveOperator = toFlat(effective);
  read.solveResidual = solveResidual;
  read.compatibilityResidual = compatibilityResidual;
  const double residual =
      certified ? std::max(solveResidual, compatibilityResidual) : kInf;
  read.certificate = Certificate::structureExact(
      CertificateDomain::Static, regime_, residual, kNaN, options_.tolerance);
  static_ = std::move(read);
  return *static_;
}

Certificate RecursiveQuotient::staticProbeCertificate(
    const std::vector<cd> &probe) const {
  const int kept = static_cast<int>(interfaceIndices_.size());
  if (static_cast<int>(probe.size()) != kept)
    throw std::invalid_argument(
        "RecursiveQuotient: probe length must equal interfaceIndices().size()");
  const StaticReductionRead &reduction = staticReduction();
  const int reduced = static_cast<int>(reduction.coordinates.size());
  const Eigen::MatrixXcd effective =
      toMatrix(reduction.effectiveOperator, reduced, reduced, "effective");

  // Assemble the fine vector x = [b on kept; x_I* on interiors].
  Eigen::VectorXcd fine = Eigen::VectorXcd::Zero(dim_);
  Eigen::VectorXcd keptProbe(kept);
  for (int position = 0; position < kept; ++position) {
    keptProbe(position) = probe[static_cast<std::size_t>(position)];
    fine(interfaceIndices_[static_cast<std::size_t>(position)]) =
        keptProbe(position);
  }
  double compatibility = 0.0;
  for (int component = 0; component < componentCount(); ++component) {
    const auto solve = componentSolve(component);
    const auto &interior = interior_[static_cast<std::size_t>(component)];
    const int a = static_cast<int>(solve->adjacentKept.size());
    if (solve->interiorDim == 0) continue;
    Eigen::VectorXcd local(a);
    for (int j = 0; j < a; ++j)
      local(j) = keptProbe(solve->adjacentKept[static_cast<std::size_t>(j)]);
    const Eigen::VectorXcd interiorResponse = -(solve->X * local);
    for (int i = 0; i < solve->interiorDim; ++i)
      fine(interior[static_cast<std::size_t>(i)]) = interiorResponse(i);
    if (solve->leftKernel.cols() > 0) {
      // Compatibility of this load: L_IB b must be orthogonal to the left
      // kernel. Automatic in the positive regime.
      Eigen::VectorXcd loadVector = Eigen::VectorXcd::Zero(solve->interiorDim);
      std::vector<int> interiorPos(static_cast<std::size_t>(dim_), -1);
      for (int i = 0; i < solve->interiorDim; ++i)
        interiorPos[static_cast<std::size_t>(
            interior[static_cast<std::size_t>(i)])] = i;
      for (int outer = 0; outer < op_.outerSize(); ++outer)
        for (Eigen::SparseMatrix<cd>::InnerIterator it(op_, outer); it; ++it) {
          const int row = interiorPos[static_cast<std::size_t>(it.row())];
          const int keptCol =
              interfacePosition_[static_cast<std::size_t>(it.col())];
          if (row >= 0 && keptCol >= 0)
            loadVector(row) += it.value() * keptProbe(keptCol);
        }
      const double scale = std::max(loadVector.norm(), 1e-300);
      compatibility = std::max(
          compatibility,
          (solve->leftKernel.adjoint() * loadVector).norm() / scale);
    }
  }

  const Eigen::VectorXcd weighted = weights_.asDiagonal() * (op_ * fine);
  const double probeScale = std::max(keptProbe.squaredNorm(), 1e-300);
  double residual = 0.0;

  if (regime_ == CertificateRegime::NonNormal ||
      regime_ == CertificateRegime::ComplexSymmetricPencil) {
    // The complex-symmetric pencil takes this path too: the pairing is
    // bilinear, so there is no energy minimum to claim. Certified block
    // elimination: eliminated interior rows of L x vanish (retained kernel
    // directions are couplings, not residuals) and the left-kernel
    // compatibility condition holds.
    const Eigen::VectorXcd applied = op_ * fine;
    double eliminated = 0.0;
    for (int component = 0; component < componentCount(); ++component) {
      const auto solve = componentSolve(component);
      const auto &interior = interior_[static_cast<std::size_t>(component)];
      if (solve->interiorDim == 0) continue;
      Eigen::VectorXcd rows(solve->interiorDim);
      for (int i = 0; i < solve->interiorDim; ++i)
        rows(i) = applied(interior[static_cast<std::size_t>(i)]);
      if (solve->rightKernel.cols() > 0)
        rows -= solve->rightKernel * (solve->rightKernel.adjoint() * rows);
      eliminated += rows.squaredNorm();
    }
    residual = std::sqrt(eliminated) /
               (std::max(opNorm_, 1e-300) * std::sqrt(probeScale));
    residual = std::max(residual, compatibility);
  } else {
    // Hermitian regimes: fine energy x^dag W L x against the coarse quadratic
    // b^dag (W L_eff) b restricted to the kept block. The positive regime
    // certifies the interior minimum (stationarity + convexity), the
    // indefinite regime stationarity alone.
    const cd fineEnergy = fine.dot(weighted);
    Eigen::VectorXcd keptWeights(kept);
    for (int position = 0; position < kept; ++position)
      keptWeights(position) = weights_(
          interfaceIndices_[static_cast<std::size_t>(position)]);
    const Eigen::MatrixXcd effectiveKept = effective.topLeftCorner(kept, kept);
    const Eigen::VectorXcd coarseApplied =
        keptWeights.asDiagonal() * (effectiveKept * keptProbe);
    const cd coarseEnergy = keptProbe.dot(coarseApplied);
    residual = std::abs(fineEnergy - coarseEnergy) / probeScale;
    // Stationarity of the interior response.
    const Eigen::VectorXcd applied = op_ * fine;
    double stationarity = 0.0;
    for (int component = 0; component < componentCount(); ++component) {
      const auto solve = componentSolve(component);
      const auto &interior = interior_[static_cast<std::size_t>(component)];
      if (solve->interiorDim == 0) continue;
      Eigen::VectorXcd rows(solve->interiorDim);
      for (int i = 0; i < solve->interiorDim; ++i)
        rows(i) = applied(interior[static_cast<std::size_t>(i)]);
      if (solve->rightKernel.cols() > 0)
        rows -= solve->rightKernel * (solve->rightKernel.adjoint() * rows);
      stationarity += rows.squaredNorm();
    }
    residual = std::max(residual,
                        std::sqrt(stationarity) /
                            (std::max(opNorm_, 1e-300) * std::sqrt(probeScale)));
    residual = std::max(residual, compatibility);
  }
  return Certificate::structureExact(CertificateDomain::Static, regime_,
                                     residual, kNaN, options_.tolerance);
}

Certificate RecursiveQuotient::verifyStatic() const {
  const int kept = static_cast<int>(interfaceIndices_.size());
  Certificate worst = Certificate::structureExact(
      CertificateDomain::Static, regime_, 0.0, kNaN, options_.tolerance);
  double worstResidual = -1.0;
  std::vector<std::vector<cd>> probes;
  for (int position = 0; position < kept; ++position) {
    std::vector<cd> probe(static_cast<std::size_t>(kept), cd(0.0, 0.0));
    probe[static_cast<std::size_t>(position)] = cd(1.0, 0.0);
    probes.push_back(std::move(probe));
  }
  if (kept > 0)
    probes.emplace_back(static_cast<std::size_t>(kept), cd(1.0, 0.0));
  for (const auto &probe : probes) {
    const Certificate certificate = staticProbeCertificate(probe);
    const double residual = certificate.residual();
    if (std::isnan(worstResidual) || residual > worstResidual ||
        std::isnan(residual)) {
      worstResidual = residual;
      worst = certificate;
    }
  }
  return worst;
}

RecursiveQuotient::FeshbachRead RecursiveQuotient::feshbach(
    cd lambda, double windowLower, double windowUpper) const {
  if (windowLower > windowUpper)
    throw std::invalid_argument("RecursiveQuotient: windowLower > windowUpper");
  const auto &solves = shiftedSolves(lambda);
  const int kept = static_cast<int>(interfaceIndices_.size());

  FeshbachRead read;
  read.lambda = lambda;
  read.windowLower = windowLower;
  read.windowUpper = windowUpper;

  int modeCount = 0;
  bool certified = true;
  for (const auto &solve : solves) {
    modeCount += static_cast<int>(solve->rightKernel.cols());
    certified = certified && solve->eliminationCertified;
  }
  read.resonant = modeCount > 0;
  const int reduced = kept + modeCount;
  Eigen::MatrixXcd response = Eigen::MatrixXcd::Zero(reduced, reduced);
  for (int outer = 0; outer < op_.outerSize(); ++outer)
    for (Eigen::SparseMatrix<cd>::InnerIterator it(op_, outer); it; ++it) {
      const int row = interfacePosition_[static_cast<std::size_t>(it.row())];
      const int col = interfacePosition_[static_cast<std::size_t>(it.col())];
      if (row >= 0 && col >= 0) response(row, col) += it.value();
    }
  if (pencil_) {
    for (int outer = 0; outer < pencilMetric_.outerSize(); ++outer)
      for (Eigen::SparseMatrix<cd>::InnerIterator it(pencilMetric_, outer); it; ++it) {
        const int row = interfacePosition_[static_cast<std::size_t>(it.row())];
        const int col = interfacePosition_[static_cast<std::size_t>(it.col())];
        if (row >= 0 && col >= 0) response(row, col) -= lambda * it.value();
      }
  } else {
    for (int position = 0; position < kept; ++position)
      response(position, position) -= lambda;
  }

  double solveResidual = 0.0;
  double compatibilityResidual = 0.0;
  int modeOffset = kept;
  for (std::size_t component = 0; component < solves.size(); ++component) {
    const ComponentSolve &solve = *solves[component];
    const int a = static_cast<int>(solve.adjacentKept.size());
    for (int i = 0; i < a; ++i)
      for (int j = 0; j < a; ++j)
        response(solve.adjacentKept[static_cast<std::size_t>(i)],
                 solve.adjacentKept[static_cast<std::size_t>(j)]) -=
            solve.contribution(i, j);
    const int modes = static_cast<int>(solve.rightKernel.cols());
    for (int t = 0; t < modes; ++t) {
      for (int j = 0; j < a; ++j) {
        response(modeOffset + t,
                 solve.adjacentKept[static_cast<std::size_t>(j)]) =
            solve.modeRow(t, j);
        response(solve.adjacentKept[static_cast<std::size_t>(j)],
                 modeOffset + t) = solve.modeCol(j, t);
      }
      for (int u = 0; u < modes; ++u)
        response(modeOffset + t, modeOffset + u) = solve.modeBlock(t, u);
    }
    modeOffset += modes;
    solveResidual = std::max(solveResidual, solve.solveResidual);
    compatibilityResidual =
        std::max(compatibilityResidual, solve.compatibilityResidual);
  }

  // Coordinates: kept cells then resonant modes.
  for (int position = 0; position < kept; ++position) {
    RetainedCoordinate coordinate;
    coordinate.kind = keptKinds_[static_cast<std::size_t>(position)];
    coordinate.component = keptOwner_[static_cast<std::size_t>(position)];
    coordinate.fineIndex = interfaceIndices_[static_cast<std::size_t>(position)];
    coordinate.embedding.assign(static_cast<std::size_t>(dim_), cd(0.0, 0.0));
    coordinate.embedding[static_cast<std::size_t>(coordinate.fineIndex)] =
        cd(1.0, 0.0);
    coordinate.provenance =
        provenance_[static_cast<std::size_t>(coordinate.fineIndex)];
    read.coordinates.push_back(std::move(coordinate));
  }
  for (std::size_t component = 0; component < solves.size(); ++component) {
    const ComponentSolve &solve = *solves[component];
    const auto &interior = interior_[component];
    for (int t = 0; t < solve.rightKernel.cols(); ++t) {
      RetainedCoordinate coordinate;
      coordinate.kind = RetainedCoordinateKind::Resonant;
      coordinate.component = static_cast<int>(component);
      coordinate.fineIndex = -1;
      coordinate.embedding.assign(static_cast<std::size_t>(dim_), cd(0.0, 0.0));
      for (int i = 0; i < solve.interiorDim; ++i)
        coordinate.embedding[static_cast<std::size_t>(
            interior[static_cast<std::size_t>(i)])] = solve.rightKernel(i, t);
      std::ostringstream provenance;
      provenance << "resonant[c" << component << "#" << t << "@(" << lambda.real()
                 << "," << lambda.imag() << ")]";
      coordinate.provenance = provenance.str();
      read.coordinates.push_back(std::move(coordinate));
    }
  }

  // Exact determinant factorization check below the dense crossover.
  read.determinantResidual = kNaN;
  if (!read.resonant && dim_ < options_.denseCrossover && certified) {
    Eigen::MatrixXcd fullShifted = Eigen::MatrixXcd(op_);
    for (int i = 0; i < dim_; ++i) fullShifted(i, i) -= lambda;
    const cd fullDet = Eigen::PartialPivLU<Eigen::MatrixXcd>(fullShifted)
                           .determinant();
    cd interiorDet = cd(1.0, 0.0);
    bool interiorDetValid = true;
    for (const auto &solve : solves) {
      if (!solve->detValid) interiorDetValid = false;
      interiorDet *= solve->interiorDet;
    }
    if (interiorDetValid && reduced > 0) {
      const cd responseDet =
          Eigen::PartialPivLU<Eigen::MatrixXcd>(response).determinant();
      const cd rhs = interiorDet * responseDet;
      const double scale = std::abs(fullDet) + std::abs(rhs);
      read.determinantResidual =
          scale > 0.0 ? std::abs(fullDet - rhs) / scale : 0.0;
    }
  }

  read.response = toFlat(response);
  read.solveResidual = solveResidual;
  read.compatibilityResidual = compatibilityResidual;
  double residual = certified
                        ? std::max(solveResidual, compatibilityResidual)
                        : kInf;
  if (!std::isnan(read.determinantResidual))
    residual = std::max(residual, read.determinantResidual);
  read.certificate = Certificate::structureExact(
      CertificateDomain::BandWindow, regime_, residual, kNaN,
      options_.tolerance);
  return read;
}

std::vector<cd> RecursiveQuotient::contourDeterminants(
    cd lambda, double radius, int nodes, std::vector<cd> &interiorDets) const {
  const int kept = static_cast<int>(interfaceIndices_.size());
  std::vector<cd> responseDets(static_cast<std::size_t>(nodes));
  interiorDets.assign(static_cast<std::size_t>(nodes), cd(1.0, 0.0));
  for (int node = 0; node < nodes; ++node) {
    const double angle = kTwoPi * node / nodes;
    const cd z = lambda + radius * cd(std::cos(angle), std::sin(angle));
    Eigen::MatrixXcd response = Eigen::MatrixXcd::Zero(kept, kept);
    for (int outer = 0; outer < op_.outerSize(); ++outer)
      for (Eigen::SparseMatrix<cd>::InnerIterator it(op_, outer); it; ++it) {
        const int row = interfacePosition_[static_cast<std::size_t>(it.row())];
        const int col = interfacePosition_[static_cast<std::size_t>(it.col())];
        if (row >= 0 && col >= 0) response(row, col) += it.value();
      }
    for (int position = 0; position < kept; ++position)
      response(position, position) -= z;
    cd interiorDet = cd(1.0, 0.0);
    for (int component = 0; component < componentCount(); ++component) {
      const auto solve = computeSolve(component, z);
      if (!solve->detValid || solve->rightKernel.cols() > 0 ||
          !solve->eliminationCertified)
        throw std::domain_error(
            "RecursiveQuotient: contour meets the interior spectrum; choose "
            "a different radius");
      interiorDet *= solve->interiorDet;
      const int a = static_cast<int>(solve->adjacentKept.size());
      for (int i = 0; i < a; ++i)
        for (int j = 0; j < a; ++j)
          response(solve->adjacentKept[static_cast<std::size_t>(i)],
                   solve->adjacentKept[static_cast<std::size_t>(j)]) -=
              solve->contribution(i, j);
    }
    interiorDets[static_cast<std::size_t>(node)] = interiorDet;
    responseDets[static_cast<std::size_t>(node)] =
        kept > 0 ? Eigen::PartialPivLU<Eigen::MatrixXcd>(response).determinant()
                 : cd(1.0, 0.0);
    if (responseDets[static_cast<std::size_t>(node)] == cd(0.0, 0.0))
      throw std::domain_error(
          "RecursiveQuotient: det F_B vanishes on the contour; choose a "
          "different radius");
  }
  return responseDets;
}

int RecursiveQuotient::windingFromPhases(const std::vector<cd> &values,
                                         double *maxStep) {
  double total = 0.0;
  double largest = 0.0;
  const std::size_t count = values.size();
  for (std::size_t node = 0; node < count; ++node) {
    const cd ratio = values[(node + 1) % count] / values[node];
    const double step = std::arg(ratio);
    largest = std::max(largest, std::abs(step));
    total += step;
  }
  if (maxStep) *maxStep = largest / (kTwoPi / 2.0);
  return static_cast<int>(std::llround(total / kTwoPi));
}

RecursiveQuotient::MultiplicityRead RecursiveQuotient::multiplicity(
    cd lambda, double radius, int nodes) const {
  if (radius <= 0.0)
    throw std::invalid_argument("RecursiveQuotient: radius must be positive");
  if (nodes < 8)
    throw std::invalid_argument("RecursiveQuotient: need at least 8 nodes");

  MultiplicityRead read;
  read.lambda = lambda;
  read.contourRadius = radius;

  std::vector<cd> interiorCoarse;
  const std::vector<cd> responseCoarse =
      contourDeterminants(lambda, radius, nodes, interiorCoarse);
  double coarseStep = 0.0;
  double coarseInteriorStep = 0.0;
  const int coarseResponse = windingFromPhases(responseCoarse, &coarseStep);
  const int coarseInterior =
      windingFromPhases(interiorCoarse, &coarseInteriorStep);

  std::vector<cd> interiorFine;
  const std::vector<cd> responseFine =
      contourDeterminants(lambda, radius, 2 * nodes, interiorFine);
  double fineStep = 0.0;
  double fineInteriorStep = 0.0;
  const int fineResponse = windingFromPhases(responseFine, &fineStep);
  const int fineInterior = windingFromPhases(interiorFine, &fineInteriorStep);

  const bool stable =
      coarseResponse == fineResponse && coarseInterior == fineInterior;
  read.nodes = 2 * nodes;
  read.responseWinding = fineResponse;
  read.interiorWinding = fineInterior;
  read.algebraic = fineResponse + fineInterior;
  read.phaseStepMargin =
      std::max(std::max(fineStep, fineInteriorStep), 0.0);

  // Geometric multiplicity: dim ker F_B(lambda) at the rank tolerance.
  const FeshbachRead pencil = feshbach(lambda, lambda.real(), lambda.real());
  const int kept = static_cast<int>(interfaceIndices_.size());
  const int reduced =
      static_cast<int>(pencil.coordinates.size());
  const Eigen::MatrixXcd response =
      toMatrix(pencil.response, reduced, reduced, "feshbach response");
  if (kept > 0 && !pencil.resonant) {
    Eigen::JacobiSVD<Eigen::MatrixXcd> svd(response);
    const auto &sigma = svd.singularValues();
    const double cut =
        options_.rankTolerance * std::max(sigma.size() ? sigma(0) : 0.0, 1e-300);
    int rank = 0;
    for (Eigen::Index i = 0; i < sigma.size(); ++i)
      if (sigma(i) > cut) ++rank;
    read.geometric = reduced - rank;
  } else if (pencil.resonant) {
    // lambda resonates with the interior block: geometric multiplicity of
    // the full operator still reads from the retained-response kernel.
    Eigen::JacobiSVD<Eigen::MatrixXcd> svd(response);
    const auto &sigma = svd.singularValues();
    const double cut =
        options_.rankTolerance * std::max(sigma.size() ? sigma(0) : 0.0, 1e-300);
    int rank = 0;
    for (Eigen::Index i = 0; i < sigma.size(); ++i)
      if (sigma(i) > cut) ++rank;
    read.geometric = reduced - rank;
  }
  read.semisimple = read.algebraic == read.geometric;

  const double residual = stable ? read.phaseStepMargin : kInf;
  read.certificate = Certificate::certifiedNumerical(
      CertificateDomain::BandWindow, regime_, residual, kNaN,
      /*tolerance=*/0.5);
  return read;
}

RecursiveQuotient::CraigBamptonRead RecursiveQuotient::craigBampton(
    double windowLower, double windowUpper, double modeCutoff,
    double residualTolerance) const {
  if (windowLower > windowUpper)
    throw std::invalid_argument("RecursiveQuotient: windowLower > windowUpper");
  if (modeCutoff < windowUpper)
    throw std::invalid_argument(
        "RecursiveQuotient: modeCutoff must cover the window upper edge");
  // The regime decides the pairing, not whether the surrogate exists. In the
  // two Hermitian regimes the pairing is the adjoint against the positive
  // diagonal chain metric W, which is what a self-adjoint eigensolver needs; in
  // the non-normal and complex-symmetric-pencil regimes it is the transpose
  // against the level's own metric -- the carried Gram on a pencil level, the
  // diagonal weights otherwise -- and the eigenproblems are solved by a general
  // complex eigensolver, which assumes nothing about the spectrum.
  const bool bilinear = regime_ == CertificateRegime::NonNormal ||
                        regime_ == CertificateRegime::ComplexSymmetricPencil;
  if (!bilinear)
    for (int i = 0; i < dim_; ++i)
      if (!(weights_(i).real() > 0.0) ||
          std::abs(weights_(i).imag()) > options_.tolerance)
        throw std::invalid_argument(
            "RecursiveQuotient: Craig-Bampton in a Hermitian regime needs a positive chain "
            "metric; a signed or complex metric is the non-normal regime, whose surrogate uses "
            "the transpose pairing");
  // The metric the reduced pencil is written against: the carried Gram on a
  // pencil level, the diagonal weights otherwise, held sparsely so that no
  // dimension-square object is built to hold a diagonal.
  const Eigen::SparseMatrix<cd> metric = [&] {
    if (pencil_) return pencilMetric_;
    Eigen::SparseMatrix<cd> diagonal(dim_, dim_);
    diagonal.reserve(Eigen::VectorXi::Constant(dim_, 1));
    for (int i = 0; i < dim_; ++i) diagonal.insert(i, i) = weights_(i);
    diagonal.makeCompressed();
    return diagonal;
  }();

  const int kept = static_cast<int>(interfaceIndices_.size());
  CraigBamptonRead read;
  read.windowLower = windowLower;
  read.windowUpper = windowUpper;
  read.modeCutoff = modeCutoff;
  read.discardedModeGap = kInf;

  struct ComponentModes {
    Eigen::MatrixXcd vectors;  // interiorDim x retained (W-orthonormal)
    int retained{0};
  };
  std::vector<ComponentModes> componentModes(components_.size());
  int totalModes = 0;
  for (int component = 0; component < componentCount(); ++component) {
    const auto &interior = interior_[static_cast<std::size_t>(component)];
    const int m = static_cast<int>(interior.size());
    if (m >= options_.denseCrossover)
      throw std::length_error(
          "RecursiveQuotient: fixed-interface eigensolve refuses at/above "
          "the dense crossover");
    if (m == 0) continue;
    Eigen::MatrixXcd block = Eigen::MatrixXcd::Zero(m, m);
    std::vector<int> interiorPos(static_cast<std::size_t>(dim_), -1);
    for (int i = 0; i < m; ++i)
      interiorPos[static_cast<std::size_t>(
          interior[static_cast<std::size_t>(i)])] = i;
    for (int outer = 0; outer < op_.outerSize(); ++outer)
      for (Eigen::SparseMatrix<cd>::InnerIterator it(op_, outer); it; ++it) {
        const int row = interiorPos[static_cast<std::size_t>(it.row())];
        const int col = interiorPos[static_cast<std::size_t>(it.col())];
        if (row >= 0 && col >= 0) block(row, col) = it.value();
      }
    ComponentModes &modes = componentModes[static_cast<std::size_t>(component)];
    if (bilinear) {
      // The interior pencil (L_II, W_II) solved by a general complex
      // eigensolver: no adjoint is formed and no order is assumed. The
      // frequency of a complex level is its real part, the convention every
      // band window in this subsystem is stated in (CertifiedBand's
      // [min Re, max Re]).
      Eigen::MatrixXcd metricBlock = Eigen::MatrixXcd::Zero(m, m);
      for (int outer = 0; outer < metric.outerSize(); ++outer)
        for (Eigen::SparseMatrix<cd>::InnerIterator it(metric, outer); it; ++it) {
          const int row = interiorPos[static_cast<std::size_t>(it.row())];
          const int col = interiorPos[static_cast<std::size_t>(it.col())];
          if (row >= 0 && col >= 0) metricBlock(row, col) = it.value();
        }
      Eigen::FullPivLU<Eigen::MatrixXcd> metricLu(metricBlock);
      metricLu.setThreshold(options_.rankTolerance);
      if (!metricLu.isInvertible())
        throw std::invalid_argument(
            "RecursiveQuotient: Craig-Bampton found the interior chain metric singular on "
            "component " + std::to_string(component) +
            ", so its fixed-interface pencil has no spectrum to retain modes from");
      Eigen::ComplexEigenSolver<Eigen::MatrixXcd> eigensolver(metricLu.solve(block), true);
      if (eigensolver.info() != Eigen::Success)
        throw std::runtime_error(
            "RecursiveQuotient: fixed-interface eigensolve failed");
      const Eigen::VectorXcd values = eigensolver.eigenvalues();
      std::vector<int> order(static_cast<std::size_t>(values.size()));
      std::iota(order.begin(), order.end(), 0);
      std::sort(order.begin(), order.end(), [&](int a, int b) {
        if (values(a).real() != values(b).real())
          return values(a).real() < values(b).real();
        return values(a).imag() < values(b).imag();
      });
      std::vector<int> keep;
      for (const int index : order) {
        if (values(index).real() <= modeCutoff)
          keep.push_back(index);
        else
          read.discardedModeGap =
              std::min(read.discardedModeGap, values(index).real() - windowUpper);
      }
      modes.retained = static_cast<int>(keep.size());
      modes.vectors = Eigen::MatrixXcd(m, modes.retained);
      for (int t = 0; t < modes.retained; ++t) {
        Eigen::VectorXcd v = eigensolver.eigenvectors().col(keep[static_cast<std::size_t>(t)]);
        v /= std::max(v.norm(), 1e-300);
        modes.vectors.col(t) = v;
      }
    } else {
      // W-similarity: W^{1/2} L W^{-1/2} is Hermitian for a W-self-adjoint L.
      Eigen::VectorXd sqrtW(m);
      for (int i = 0; i < m; ++i)
        sqrtW(i) = std::sqrt(
            weights_(interior[static_cast<std::size_t>(i)]).real());
      Eigen::MatrixXcd symmetric = block;
      for (int i = 0; i < m; ++i)
        for (int j = 0; j < m; ++j)
          symmetric(i, j) *= sqrtW(i) / sqrtW(j);
      const Eigen::MatrixXcd hermitized =
          0.5 * (symmetric + symmetric.adjoint());
      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> eigensolver(hermitized);
      if (eigensolver.info() != Eigen::Success)
        throw std::runtime_error(
            "RecursiveQuotient: fixed-interface eigensolve failed");
      const auto &values = eigensolver.eigenvalues();
      int retained = 0;
      for (Eigen::Index i = 0; i < values.size(); ++i)
        if (values(i) <= modeCutoff) ++retained;
      if (retained < m)
        read.discardedModeGap =
            std::min(read.discardedModeGap, values(retained) - windowUpper);
      modes.retained = retained;
      modes.vectors = Eigen::MatrixXcd(m, retained);
      for (int t = 0; t < retained; ++t)
        for (int i = 0; i < m; ++i)
          modes.vectors(i, t) = eigensolver.eigenvectors()(i, t) / sqrtW(i);
    }
    totalModes += modes.retained;
    read.retainedModes.push_back(modes.retained);
  }

  const int reducedDim = kept + totalModes;
  Eigen::MatrixXcd basis = Eigen::MatrixXcd::Zero(dim_, reducedDim);
  for (int position = 0; position < kept; ++position)
    basis(interfaceIndices_[static_cast<std::size_t>(position)], position) =
        cd(1.0, 0.0);
  // Constraint modes psi = -L_II^{+} L_IK per component, reusing the static
  // component solves verbatim.
  for (int component = 0; component < componentCount(); ++component) {
    const auto solve = componentSolve(component);
    const auto &interior = interior_[static_cast<std::size_t>(component)];
    for (int j = 0; j < static_cast<int>(solve->adjacentKept.size()); ++j) {
      const int position = solve->adjacentKept[static_cast<std::size_t>(j)];
      for (int i = 0; i < solve->interiorDim; ++i)
        basis(interior[static_cast<std::size_t>(i)], position) =
            -solve->X(i, j);
    }
  }
  int column = kept;
  for (int component = 0; component < componentCount(); ++component) {
    const auto &modes = componentModes[static_cast<std::size_t>(component)];
    const auto &interior = interior_[static_cast<std::size_t>(component)];
    for (int t = 0; t < modes.retained; ++t, ++column)
      for (int i = 0; i < static_cast<int>(interior.size()); ++i)
        basis(interior[static_cast<std::size_t>(i)], column) =
            modes.vectors(i, t);
  }

  // The congruence: the transpose pairing in the bilinear regimes, the adjoint
  // against the positive metric in the Hermitian ones.
  const Eigen::MatrixXcd weightedBasis = metric * basis;
  const Eigen::MatrixXcd operatorBasis =
      bilinear ? Eigen::MatrixXcd(op_ * basis)
               : Eigen::MatrixXcd(weights_.asDiagonal() * (op_ * basis));
  const Eigen::MatrixXcd stiffness =
      bilinear ? Eigen::MatrixXcd(basis.transpose() * operatorBasis)
               : Eigen::MatrixXcd(basis.adjoint() * operatorBasis);
  const Eigen::MatrixXcd mass = bilinear
                                    ? Eigen::MatrixXcd(basis.transpose() * weightedBasis)
                                    : Eigen::MatrixXcd(basis.adjoint() * weightedBasis);
  read.basis = toFlat(basis);
  read.reducedStiffness = toFlat(stiffness);
  read.reducedMass = toFlat(mass);

  double worstResidual = 0.0;
  if (reducedDim > 0 && bilinear) {
    Eigen::FullPivLU<Eigen::MatrixXcd> massLu(mass);
    massLu.setThreshold(options_.rankTolerance);
    if (!massLu.isInvertible())
      throw std::invalid_argument(
          "RecursiveQuotient: Craig-Bampton found the reduced chain metric V^T W V singular, so "
          "the reduction basis is degenerate and its spectrum is not the surrogate spectrum");
    Eigen::ComplexEigenSolver<Eigen::MatrixXcd> reduced(massLu.solve(stiffness), true);
    if (reduced.info() != Eigen::Success)
      throw std::runtime_error("RecursiveQuotient: reduced eigensolve failed");
    const double scale = std::max(opNorm_, 1e-300);
    for (Eigen::Index i = 0; i < reduced.eigenvalues().size(); ++i) {
      const cd value = reduced.eigenvalues()(i);
      if (value.real() < windowLower || value.real() > windowUpper) continue;
      read.windowEigenvalues.push_back(value.real());
      const Eigen::VectorXcd fine = basis * reduced.eigenvectors().col(i);
      const double norm = std::max(fine.norm(), 1e-300);
      const double residual =
          (Eigen::VectorXcd(op_ * fine) - value * Eigen::VectorXcd(metric * fine)).norm() /
          (scale * norm);
      read.eigenResiduals.push_back(residual);
      worstResidual = std::max(worstResidual, residual);
    }
  } else if (reducedDim > 0) {
    const Eigen::MatrixXcd stiffnessHermitized =
        0.5 * (stiffness + stiffness.adjoint());
    const Eigen::MatrixXcd massHermitized = 0.5 * (mass + mass.adjoint());
    Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXcd> pencil(
        stiffnessHermitized, massHermitized);
    if (pencil.info() != Eigen::Success)
      throw std::runtime_error("RecursiveQuotient: reduced eigensolve failed");
    const double scale = std::max(opNorm_, 1e-300);
    for (Eigen::Index i = 0; i < pencil.eigenvalues().size(); ++i) {
      const double value = pencil.eigenvalues()(i);
      if (value < windowLower || value > windowUpper) continue;
      read.windowEigenvalues.push_back(value);
      const Eigen::VectorXcd fine = basis * pencil.eigenvectors().col(i);
      const double norm = std::max(fine.norm(), 1e-300);
      const double residual =
          (weights_.asDiagonal() * (op_ * fine) -
           value * (weights_.asDiagonal() * fine))
              .norm() /
          (scale * norm);
      read.eigenResiduals.push_back(residual);
      worstResidual = std::max(worstResidual, residual);
    }
  }
  const double declared =
      residualTolerance < 0.0 ? options_.tolerance : residualTolerance;
  read.certificate = Certificate::certifiedNumerical(
      CertificateDomain::BandWindow, regime_, worstResidual, kNaN, declared);
  return read;
}

RecursiveQuotient::LabeledFiberSumRead RecursiveQuotient::labeledFiberSum()
    const {
  const StaticReductionRead &reduction = staticReduction();
  const int kept = static_cast<int>(interfaceIndices_.size());

  LabeledFiberSumRead read;
  read.policy = options_.embeddingPolicy;

  // Fiber E_v: every kept cell claimed by v — a shared interface cell appears
  // in every claiming fiber — plus the retained modes owned by v.
  std::vector<Eigen::VectorXcd> columns;
  for (int component = 0; component < componentCount(); ++component) {
    int rank = 0;
    for (int position = 0; position < kept; ++position) {
      const int fine = interfaceIndices_[static_cast<std::size_t>(position)];
      const auto &owners = claimants_[static_cast<std::size_t>(fine)];
      if (std::find(owners.begin(), owners.end(), component) == owners.end())
        continue;
      Eigen::VectorXcd columnVector = Eigen::VectorXcd::Zero(dim_);
      columnVector(fine) = cd(1.0, 0.0);
      columns.push_back(columnVector);
      ++rank;
    }
    for (std::size_t coordinate = static_cast<std::size_t>(kept);
         coordinate < reduction.coordinates.size(); ++coordinate) {
      const RetainedCoordinate &retained = reduction.coordinates[coordinate];
      if (retained.component != component) continue;
      Eigen::VectorXcd columnVector(dim_);
      for (int i = 0; i < dim_; ++i)
        columnVector(i) = retained.embedding[static_cast<std::size_t>(i)];
      columns.push_back(columnVector);
      ++rank;
    }
    if (rank > 0) {
      read.summandComponents.push_back(component);
      read.summandRanks.push_back(rank);
    }
  }

  LabeledFiberSumRead summary = summarizeFiberSum(columns);
  summary.summandComponents = std::move(read.summandComponents);
  summary.summandRanks = std::move(read.summandRanks);
  return summary;
}

RecursiveQuotient::LabeledFiberSumRead RecursiveQuotient::summarizeFiberSum(
    const std::vector<Eigen::VectorXcd> &columns) const {
  LabeledFiberSumRead read;
  read.policy = options_.embeddingPolicy;

  const int total = static_cast<int>(columns.size());
  // An empty labeled sum is legitimate: a single component covering every cell
  // keeps no interface cell, and an interior block with no kernel retains no
  // mode. It is trivially an exact isometry, reported rather than computed
  // because the spectral norm and the SVD below are undefined at size zero (a
  // zero-size JacobiSVD faults in a Release build).
  if (total == 0) {
    read.nominalRank = 0;
    read.effectiveRank = 0;
    read.gramDefect = 0.0;
    read.quotientNullity = 0;
    read.certificate = Certificate::algebraicallyExact(
        CertificateDomain::Static, regime_, 0.0, options_.tolerance);
    return read;
  }
  Eigen::MatrixXcd embedding(dim_, total);
  for (int j = 0; j < total; ++j) {
    Eigen::VectorXcd columnVector = columns[static_cast<std::size_t>(j)];
    // |W|-unit normalization keeps the Gram scale-free; a W-null column is
    // left raw and its Gram diagonal reports the null norm.
    cd wNorm = cd(0.0, 0.0);
    if (pencil_) {
      // The complex bilinear pairing c^T M c: no conjugation.
      wNorm = (columnVector.transpose() * (pencilMetric_ * columnVector))(0, 0);
    } else {
      for (int i = 0; i < dim_; ++i)
        wNorm += std::conj(columnVector(i)) * weights_(i) * columnVector(i);
    }
    const double magnitude = std::sqrt(std::abs(wNorm));
    if (magnitude > 1e-300) columnVector /= magnitude;
    embedding.col(j) = columnVector;
  }
  const Eigen::MatrixXcd gram =
      pencil_ ? Eigen::MatrixXcd(embedding.transpose() * (pencilMetric_ * embedding))
              : Eigen::MatrixXcd(embedding.adjoint() * (weights_.asDiagonal() * embedding));
  read.embedding = toFlat(embedding);
  read.gram = toFlat(gram);
  read.nominalRank = static_cast<std::size_t>(total);

  const Eigen::MatrixXcd defect =
      gram - Eigen::MatrixXcd::Identity(total, total);
  read.gramDefect = matrixNorm2(defect, options_.denseCrossover);

  Eigen::JacobiSVD<Eigen::MatrixXcd> svd(
      gram, Eigen::ComputeThinU | Eigen::ComputeThinV);
  const auto &sigma = svd.singularValues();
  const double cut =
      options_.rankTolerance * std::max(sigma.size() ? sigma(0) : 0.0, 1e-300);
  int rank = 0;
  for (Eigen::Index i = 0; i < sigma.size(); ++i)
    if (sigma(i) > cut) ++rank;
  read.quotientNullity = static_cast<std::size_t>(total - rank);

  switch (options_.embeddingPolicy) {
    case FiberEmbeddingPolicy::CarryGramExactly:
      read.effectiveRank = read.nominalRank;
      read.certificate = Certificate::algebraicallyExact(
          CertificateDomain::Static, regime_, 0.0, options_.tolerance);
      break;
    case FiberEmbeddingPolicy::CertifiedNearIsometry:
      read.effectiveRank = read.nominalRank;
      read.certificate = Certificate::algebraicallyExact(
          CertificateDomain::Static, regime_, read.gramDefect,
          options_.nearIsometryEpsilon);
      break;
    case FiberEmbeddingPolicy::QuotientKernel: {
      read.effectiveRank = static_cast<std::size_t>(rank);
      read.quotientBasis = toFlat(svd.matrixV().leftCols(rank));
      const double discarded =
          total - rank > 0 ? sigma(rank) / std::max(sigma(0), 1e-300) : 0.0;
      read.certificate = Certificate::certifiedNumerical(
          CertificateDomain::Static, regime_, discarded, kNaN,
          options_.rankTolerance);
      break;
    }
  }
  return read;
}

RecursiveQuotient::LabeledFiberSumRead RecursiveQuotient::certifiedFiberSum(
    const std::vector<CertifiedBand> &bands) const {
  std::vector<Eigen::VectorXcd> columns;
  std::vector<int> summandComponents;
  std::vector<int> summandRanks;
  std::vector<CertifiedFiberSummand> summandCertificates;
  // The worst isolation gap over the summed bands. An unknown side (NaN) does
  // not participate; all sides unknown leaves the field NaN.
  double worstGap = kInf;
  bool sawGap = false;
  bool allAccepted = true;

  for (const CertifiedBand &band : bands) {
    if (band.component < 0 || band.component >= componentCount())
      throw std::invalid_argument(
          "certifiedFiberSum: band names an unknown component");
    const std::size_t expected =
        static_cast<std::size_t>(dim_) * band.rank;
    if (band.frame.size() != expected)
      throw std::invalid_argument(
          "certifiedFiberSum: frame size does not match dimension x rank");

    const int rank = static_cast<int>(band.rank);
    if (rank > 0) {
      const Eigen::MatrixXcd frame =
          toMatrix(band.frame, dim_, rank, "certified band frame");
      for (int j = 0; j < rank; ++j) columns.push_back(frame.col(j));
    }

    // Every supplied band is reported, rank-zero or uncertified included, so
    // the summand lists stay 1:1 with the input.
    summandComponents.push_back(band.component);
    summandRanks.push_back(rank);
    CertifiedFiberSummand summand;
    summand.component = band.component;
    summand.rank = band.rank;
    summand.lowerGap = band.lowerGap;
    summand.upperGap = band.upperGap;
    summand.frequencyLower = band.frequencyLower;
    summand.frequencyUpper = band.frequencyUpper;
    summand.accepted = band.accepted;
    summand.certificate = band.certificate;
    summandCertificates.push_back(summand);

    allAccepted = allAccepted && band.accepted;
    for (const double gap : {band.lowerGap, band.upperGap}) {
      if (std::isnan(gap)) continue;
      worstGap = std::min(worstGap, gap);
      sawGap = true;
    }
  }

  LabeledFiberSumRead read = summarizeFiberSum(columns);
  read.summandComponents = std::move(summandComponents);
  read.summandRanks = std::move(summandRanks);
  read.summandCertificates = std::move(summandCertificates);
  read.fromCertifiedBands = true;
  read.allBandsAccepted = allAccepted && !bands.empty();
  read.worstIsolationGap = sawGap ? worstGap : kNaN;
  // The Gram treatment cannot turn an uncertified summand into a certified
  // sum: the isometry claim may hold exactly while the certified
  // isolated-subspace claim does not, so the sum carries a marker that never
  // holds.
  if (!read.allBandsAccepted)
    read.certificate =
        Certificate::heuristicDiscovery(CertificateDomain::BandWindow, regime_);
  return read;
}

RecursiveQuotient::FockStageRead RecursiveQuotient::fockStage(
    const LabeledFiberSumRead &sum, std::size_t maxTerms) const {
  FockStageRead read;
  read.policy = sum.policy;
  read.gramDefect = sum.gramDefect;

  const int total = static_cast<int>(sum.nominalRank);
  if (total == 0) {
    // Fock of the zero space is the one-dimensional vacuum line, and its
    // free many-body spectrum is the single value 0.
    read.modes = 0;
    read.fockDimension = 1.0;
    read.spectrumMaterialized = true;
    read.fockSpectrum = {cd(0.0, 0.0)};
    read.certificate = Certificate::algebraicallyExact(
        CertificateDomain::Static, regime_, 0.0, options_.tolerance);
    return read;
  }
  if (sum.embedding.size() !=
      static_cast<std::size_t>(dim_) * static_cast<std::size_t>(total))
    throw std::invalid_argument(
        "fockStage: labeled-sum embedding does not match this level");

  Eigen::MatrixXcd embedding =
      toMatrix(sum.embedding, dim_, total, "labeled sum embedding");
  Eigen::MatrixXcd gram = toMatrix(sum.gram, total, total, "labeled sum gram");
  // h = J^dagger W L J: the one-particle operator compressed onto the labeled
  // sum in the W-pairing L is self-adjoint against.
  const Eigen::MatrixXcd dense = Eigen::MatrixXcd(op_);
  Eigen::MatrixXcd oneParticle =
      embedding.adjoint() * (weights_.asDiagonal() * (dense * embedding));

  // Under `QuotientKernel` the stage is built on the quotient by ker G, so the
  // overcounted directions leave the basis.
  bool quotiented = false;
  if (sum.policy == FiberEmbeddingPolicy::QuotientKernel &&
      sum.effectiveRank < sum.nominalRank) {
    const int effective = static_cast<int>(sum.effectiveRank);
    const Eigen::MatrixXcd basis =
        toMatrix(sum.quotientBasis, total, effective, "quotient basis");
    oneParticle = basis.adjoint() * oneParticle * basis;
    gram = basis.adjoint() * gram * basis;
    quotiented = true;
  }

  const int modes = static_cast<int>(oneParticle.rows());
  read.modes = static_cast<std::size_t>(modes);
  read.oneParticle = toFlat(oneParticle);
  read.gram = toFlat(gram);
  read.fockDimension = std::ldexp(1.0, modes);

  // The one-particle spectrum is that of the pencil (h, G): the labeled-sum
  // basis is not orthonormal in general, and the eigenvalues of h alone would
  // assume G = I. A singular G without `QuotientKernel` means an unremoved
  // overcount, so there is no spectrum to report and this refuses.
  const bool singular =
      !quotiented && sum.quotientNullity > 0 &&
      sum.policy != FiberEmbeddingPolicy::QuotientKernel;
  if (singular) {
    read.spectrumMaterialized = false;
    read.certificate =
        Certificate::heuristicDiscovery(CertificateDomain::Static, regime_);
    return read;
  }

  const Eigen::MatrixXcd pencil = gram.partialPivLu().solve(oneParticle);
  Eigen::ComplexEigenSolver<Eigen::MatrixXcd> solver(pencil,
                                                     /*computeEigenvectors=*/false);
  std::vector<cd> spectrum(static_cast<std::size_t>(modes));
  for (int i = 0; i < modes; ++i)
    spectrum[static_cast<std::size_t>(i)] = solver.eigenvalues()(i);
  std::sort(spectrum.begin(), spectrum.end(), [](const cd &a, const cd &b) {
    if (a.real() != b.real()) return a.real() < b.real();
    return a.imag() < b.imag();
  });
  read.oneParticleSpectrum = spectrum;

  // The free many-body spectrum of dGamma(h) is the exact set of occupation
  // subset sums; the enumeration refuses past the declared budget rather than
  // allocating 2^M entries.
  try {
    read.fockSpectrum = OccupationSpectra::fockSums(spectrum, maxTerms);
    read.spectrumMaterialized = true;
  } catch (const std::length_error &) {
    read.spectrumMaterialized = false;
  }

  read.certificate =
      sum.policy == FiberEmbeddingPolicy::CertifiedNearIsometry
          ? Certificate::algebraicallyExact(CertificateDomain::Static, regime_,
                                            sum.gramDefect,
                                            options_.nearIsometryEpsilon)
          : Certificate::algebraicallyExact(CertificateDomain::Static, regime_,
                                            0.0, options_.tolerance);
  return read;
}

std::vector<std::vector<int>> RecursiveQuotient::persistentPartition(
    const std::vector<cd> &op, int dim, double gamma, int restarts,
    std::uint64_t baseSeed) {
  if (dim < 0 || op.size() != static_cast<std::size_t>(dim) *
                                 static_cast<std::size_t>(dim))
    throw std::invalid_argument(
        "persistentPartition: flat size does not match dimension");
  if (restarts <= 0)
    throw std::invalid_argument("persistentPartition: restarts must be > 0");
  if (dim == 0) return {};

  // The similarity graph of a response network: the symmetrized off-diagonal
  // magnitude w_ij = |R_ij| + |R_ji|. The diagonal never enters; the magnitude
  // is taken because modularity needs a nonnegative weight and the operator is
  // complex.
  std::vector<std::uint64_t> src;
  std::vector<std::uint64_t> tgt;
  std::vector<double> weight;
  for (int i = 0; i < dim; ++i) {
    for (int j = i + 1; j < dim; ++j) {
      const double w =
          std::abs(op[static_cast<std::size_t>(i) * dim + j]) +
          std::abs(op[static_cast<std::size_t>(j) * dim + i]);
      if (w <= 0.0) continue;
      src.push_back(static_cast<std::uint64_t>(i));
      tgt.push_back(static_cast<std::uint64_t>(j));
      weight.push_back(w);
    }
  }
  // Every coordinate is declared a node, so an uncoupled coordinate still
  // appears in the partition.
  std::vector<std::uint64_t> isolated(static_cast<std::size_t>(dim));
  for (int i = 0; i < dim; ++i)
    isolated[static_cast<std::size_t>(i)] = static_cast<std::uint64_t>(i);

  const observables::PersistentModularity modularity =
      observables::PersistentModularity::fromWeightedEdges(src, tgt, weight,
                                                           isolated);
  observables::PersistentModularityConfig config;
  config.restarts = restarts;
  config.baseSeed = baseSeed;
  const observables::ResolutionSlice slice = modularity.discover(gamma, config);

  std::vector<std::vector<int>> partition;
  std::vector<bool> claimed(static_cast<std::size_t>(dim), false);
  for (const observables::ComponentRead &component : slice.components) {
    std::vector<int> members;
    for (const std::uint64_t cell : component.support) {
      const int index = static_cast<int>(cell);
      if (index < 0 || index >= dim) continue;
      if (claimed[static_cast<std::size_t>(index)]) continue;
      claimed[static_cast<std::size_t>(index)] = true;
      members.push_back(index);
    }
    if (!members.empty()) {
      std::sort(members.begin(), members.end());
      partition.push_back(std::move(members));
    }
  }
  // A coordinate no discovered community claimed becomes its own component;
  // the partition handed to `nextLevel` must cover every index.
  for (int i = 0; i < dim; ++i)
    if (!claimed[static_cast<std::size_t>(i)]) partition.push_back({i});
  return partition;
}

std::vector<std::vector<int>> RecursiveQuotient::childPersistentPartition(
    double gamma, int restarts, std::uint64_t baseSeed) const {
  const StaticReductionRead &reduction = staticReduction();
  return persistentPartition(reduction.effectiveOperator,
                             static_cast<int>(reduction.coordinates.size()),
                             gamma, restarts, baseSeed);
}

RecursiveQuotient::ResponseNetworkRead RecursiveQuotient::responseNetwork()
    const {
  const StaticReductionRead &reduction = staticReduction();
  const int kept = static_cast<int>(interfaceIndices_.size());
  const int reduced = static_cast<int>(reduction.coordinates.size());
  const Eigen::MatrixXcd effective =
      toMatrix(reduction.effectiveOperator, reduced, reduced, "effective");

  ResponseNetworkRead read;
  read.stalkCoordinates.assign(components_.size(), {});

  // An empty reduction is legitimate, as for `labeledFiberSum`, and leaves a
  // 0 x 0 reduced operator. `Eigen::maxCoeff` is undefined at size zero, so the
  // empty network is reported rather than computed: one empty stalk per
  // component, no edges, nothing uncovered.
  if (reduced == 0) {
    read.stalkDimensions.assign(components_.size(), 0);
    read.vertexBlocks.assign(components_.size(), {});
    read.coverageResidual = 0.0;
    read.certificate = Certificate::algebraicallyExact(
        CertificateDomain::Static, regime_, 0.0, options_.tolerance);
    return read;
  }

  for (int position = 0; position < kept; ++position) {
    const int fine = interfaceIndices_[static_cast<std::size_t>(position)];
    for (const int component : claimants_[static_cast<std::size_t>(fine)])
      read.stalkCoordinates[static_cast<std::size_t>(component)].push_back(
          position);
  }
  for (int coordinate = kept; coordinate < reduced; ++coordinate)
    read.stalkCoordinates[static_cast<std::size_t>(
                              reduction.coordinates[static_cast<std::size_t>(
                                                        coordinate)]
                                  .component)]
        .push_back(coordinate);
  for (const auto &stalk : read.stalkCoordinates)
    read.stalkDimensions.push_back(static_cast<int>(stalk.size()));

  const double scale = std::max(effective.cwiseAbs().maxCoeff(), 1e-300);
  for (int component = 0; component < componentCount(); ++component) {
    const auto &stalk = read.stalkCoordinates[static_cast<std::size_t>(component)];
    const int stalkDim = static_cast<int>(stalk.size());
    Eigen::MatrixXcd block(stalkDim, stalkDim);
    for (int i = 0; i < stalkDim; ++i)
      for (int j = 0; j < stalkDim; ++j)
        block(i, j) = effective(stalk[static_cast<std::size_t>(i)],
                                stalk[static_cast<std::size_t>(j)]);
    read.vertexBlocks.push_back(toFlat(block));
  }

  std::vector<std::vector<bool>> hasEdge(
      components_.size(), std::vector<bool>(components_.size(), false));
  for (int from = 0; from < componentCount(); ++from)
    for (int to = 0; to < componentCount(); ++to) {
      if (from == to) continue;
      const auto &stalkFrom =
          read.stalkCoordinates[static_cast<std::size_t>(from)];
      const auto &stalkTo = read.stalkCoordinates[static_cast<std::size_t>(to)];
      Eigen::MatrixXcd block(static_cast<int>(stalkFrom.size()),
                             static_cast<int>(stalkTo.size()));
      bool shared = false;
      for (std::size_t i = 0; i < stalkFrom.size(); ++i)
        for (std::size_t j = 0; j < stalkTo.size(); ++j) {
          block(static_cast<int>(i), static_cast<int>(j)) =
              effective(stalkFrom[i], stalkTo[j]);
          if (stalkFrom[i] == stalkTo[j]) shared = true;
        }
      const double magnitude =
          block.size() > 0 ? block.cwiseAbs().maxCoeff() : 0.0;
      if (magnitude > options_.tolerance * scale || shared) {
        hasEdge[static_cast<std::size_t>(from)][static_cast<std::size_t>(to)] =
            true;
        ResponseEdge edge;
        edge.from = from;
        edge.to = to;
        edge.block = toFlat(block);
        read.edges.push_back(std::move(edge));
      }
    }

  // Coverage: reduced entries outside every vertex/edge block support.
  double coverage = 0.0;
  std::vector<std::vector<int>> owners(static_cast<std::size_t>(reduced));
  for (int component = 0; component < componentCount(); ++component)
    for (const int coordinate :
         read.stalkCoordinates[static_cast<std::size_t>(component)])
      owners[static_cast<std::size_t>(coordinate)].push_back(component);
  for (int i = 0; i < reduced; ++i)
    for (int j = 0; j < reduced; ++j) {
      const double magnitude = std::abs(effective(i, j));
      if (magnitude == 0.0) continue;
      bool covered = false;
      for (const int u : owners[static_cast<std::size_t>(i)]) {
        for (const int v : owners[static_cast<std::size_t>(j)])
          if (u == v || hasEdge[static_cast<std::size_t>(u)]
                               [static_cast<std::size_t>(v)]) {
            covered = true;
            break;
          }
        if (covered) break;
      }
      if (!covered) coverage = std::max(coverage, magnitude);
    }
  read.coverageResidual = coverage;
  read.certificate = Certificate::algebraicallyExact(
      CertificateDomain::Static, regime_, coverage / scale, options_.tolerance);
  return read;
}

RecursiveQuotient::SheafRealizationRead RecursiveQuotient::sheafRealization()
    const {
  SheafRealizationRead read;
  if (regime_ == CertificateRegime::NonNormal ||
      regime_ == CertificateRegime::ComplexSymmetricPencil) {
    // A cellular sheaf Laplacian is self-adjoint, so a non-normal response
    // network has no such realization and is retained as is. The
    // complex-symmetric pencil is refused for the same reason.
    read.certificate = Certificate::certifiedNumerical(
        CertificateDomain::Static, regime_, kInf, kNaN, options_.tolerance);
    return read;
  }
  const ResponseNetworkRead network = responseNetwork();
  const int componentTotal = componentCount();

  // Collect undirected edges (u < v) with their forward blocks.
  struct Undirected {
    int u{0};
    int v{0};
    Eigen::MatrixXcd forward;  // block(u, v)
  };
  std::vector<Undirected> undirected;
  double hermiticity = 0.0;
  double scale = 1e-300;
  std::map<std::pair<int, int>, Eigen::MatrixXcd> directed;
  for (const auto &edge : network.edges) {
    const int rows = network.stalkDimensions[static_cast<std::size_t>(edge.from)];
    const int cols = network.stalkDimensions[static_cast<std::size_t>(edge.to)];
    directed[{edge.from, edge.to}] =
        toMatrix(edge.block, rows, cols, "edge block");
  }
  for (const auto &[key, forward] : directed) {
    if (key.first >= key.second) continue;
    const auto reverse = directed.find({key.second, key.first});
    scale = std::max(scale, forward.norm());
    if (reverse != directed.end())
      hermiticity = std::max(
          hermiticity, (forward - reverse->second.adjoint()).norm());
    else
      hermiticity = std::max(hermiticity, forward.norm());
    undirected.push_back({key.first, key.second, forward});
  }
  if (hermiticity > options_.tolerance * scale) {
    read.certificate = Certificate::certifiedNumerical(
        CertificateDomain::Static, regime_, hermiticity / scale, kNaN,
        options_.tolerance);
    return read;
  }

  // Factor each off-diagonal block: -L_uv = rho_u^dagger rho_v via SVD.
  std::vector<Eigen::MatrixXcd> diagonal(static_cast<std::size_t>(componentTotal));
  for (int component = 0; component < componentTotal; ++component) {
    const int stalkDim =
        network.stalkDimensions[static_cast<std::size_t>(component)];
    diagonal[static_cast<std::size_t>(component)] =
        Eigen::MatrixXcd::Zero(stalkDim, stalkDim);
  }
  std::vector<int> edgeDims;
  std::vector<std::vector<cd>> maps;
  for (const auto &edge : undirected) {
    const Eigen::MatrixXcd negative = -edge.forward;
    Eigen::JacobiSVD<Eigen::MatrixXcd> svd(
        negative, Eigen::ComputeThinU | Eigen::ComputeThinV);
    const auto &sigma = svd.singularValues();
    const double cut =
        options_.rankTolerance * std::max(sigma.size() ? sigma(0) : 0.0, 1e-300);
    int rank = 0;
    for (Eigen::Index i = 0; i < sigma.size(); ++i)
      if (sigma(i) > cut) ++rank;
    Eigen::MatrixXcd rhoU(rank, negative.rows());
    Eigen::MatrixXcd rhoV(rank, negative.cols());
    for (int r = 0; r < rank; ++r) {
      const double root = std::sqrt(sigma(r));
      rhoU.row(r) = root * svd.matrixU().col(r).adjoint();
      rhoV.row(r) = root * svd.matrixV().col(r).adjoint();
    }
    diagonal[static_cast<std::size_t>(edge.u)] += rhoU.adjoint() * rhoU;
    diagonal[static_cast<std::size_t>(edge.v)] += rhoV.adjoint() * rhoV;
    edgeDims.push_back(rank);
    maps.push_back(toFlat(rhoU));
    maps.push_back(toFlat(rhoV));
  }

  // The realization stands only if the sheaf Laplacian also reproduces the
  // vertex blocks.
  double residual = 0.0;
  for (int component = 0; component < componentTotal; ++component) {
    const int stalkDim =
        network.stalkDimensions[static_cast<std::size_t>(component)];
    const Eigen::MatrixXcd vertexBlock = toMatrix(
        network.vertexBlocks[static_cast<std::size_t>(component)], stalkDim,
        stalkDim, "vertex block");
    const double blockScale = std::max(vertexBlock.norm(), scale);
    residual = std::max(
        residual,
        (vertexBlock - diagonal[static_cast<std::size_t>(component)]).norm() /
            blockScale);
  }
  read.reconstructionResidual = residual;
  read.certificate = Certificate::certifiedNumerical(
      CertificateDomain::Static, regime_, residual, kNaN, options_.tolerance);
  if (read.certificate.holds()) {
    read.emitted = true;
    read.edgeStalkDimensions = edgeDims;
    read.restrictionMaps = maps;
    read.simplicial = true;
    for (const int stalkDim : network.stalkDimensions)
      if (stalkDim > 1) read.simplicial = false;
    for (const int edgeDim : edgeDims)
      if (edgeDim > 1) read.simplicial = false;
  }
  return read;
}

RecursiveQuotient RecursiveQuotient::childOver(
    const std::vector<cd> &op,
    const std::vector<RetainedCoordinate> &coordinates,
    const std::vector<std::vector<int>> &components,
    const Options &options) const {
  const int reduced = static_cast<int>(coordinates.size());
  // Child chain metric: the reduced coordinates' W-norms — kept cells their
  // weights, retained modes their indefinite norms.
  std::vector<cd> childWeights(static_cast<std::size_t>(reduced));
  for (int coordinate = 0; coordinate < reduced; ++coordinate) {
    const RetainedCoordinate &retained =
        coordinates[static_cast<std::size_t>(coordinate)];
    cd wNorm = cd(0.0, 0.0);
    for (int i = 0; i < dim_; ++i)
      wNorm += std::conj(retained.embedding[static_cast<std::size_t>(i)]) *
               weights_(i) * retained.embedding[static_cast<std::size_t>(i)];
    childWeights[static_cast<std::size_t>(coordinate)] = wNorm;
  }
  RecursiveQuotient child;
  child.level_ = level_ + 1;
  child.provenance_.resize(static_cast<std::size_t>(reduced));
  for (int coordinate = 0; coordinate < reduced; ++coordinate)
    child.provenance_[static_cast<std::size_t>(coordinate)] =
        "L" + std::to_string(level_) + ":" +
        coordinates[static_cast<std::size_t>(coordinate)].provenance;
  child.initMatrix(op, reduced, childWeights, components, options);
  return child;
}

RecursiveQuotient RecursiveQuotient::overPencil(
    const std::vector<cd> &A, const std::vector<cd> &M, int dim,
    const std::vector<std::vector<int>> &components, const Options &options) {
  RecursiveQuotient quotient;
  quotient.pencil_ = true;
  const Eigen::MatrixXcd denseM = toMatrix(M, dim, dim, "RecursiveQuotient pencil metric");
  quotient.pencilMetric_ = denseM.sparseView();
  quotient.pencilMetric_.makeCompressed();
  quotient.initMatrix(A, dim, {}, components, options);
  return quotient;
}

std::vector<cd> RecursiveQuotient::pencilMetric() const {
  if (!pencil_) return {};
  return toFlat(Eigen::MatrixXcd(pencilMetric_));
}

Eigen::MatrixXcd RecursiveQuotient::pencilConstraintModes(
    const std::vector<RetainedCoordinate> &coordinates,
    const std::vector<std::shared_ptr<ComponentSolve>> &solves) const {
  const int reduced = static_cast<int>(coordinates.size());
  Eigen::MatrixXcd T = Eigen::MatrixXcd::Zero(dim_, reduced);
  for (int p = 0; p < reduced; ++p) {
    const RetainedCoordinate &coordinate = coordinates[static_cast<std::size_t>(p)];
    if (coordinate.fineIndex < 0) {
      // A retained resonant mode embeds as the kernel vector in fine
      // coordinates.
      for (int i = 0; i < dim_; ++i)
        T(i, p) = coordinate.embedding[static_cast<std::size_t>(i)];
      continue;
    }
    T(coordinate.fineIndex, p) = cd(1.0, 0.0);
    const int position = interfacePosition_[static_cast<std::size_t>(coordinate.fineIndex)];
    if (position < 0) continue;
    for (std::size_t component = 0; component < solves.size(); ++component) {
      const ComponentSolve &solve = *solves[component];
      const auto it = std::find(solve.adjacentKept.begin(), solve.adjacentKept.end(), position);
      if (it == solve.adjacentKept.end()) continue;
      const int j = static_cast<int>(it - solve.adjacentKept.begin());
      const auto &interior = interior_[component];
      for (int i = 0; i < solve.interiorDim && i < static_cast<int>(interior.size()); ++i)
        T(interior[static_cast<std::size_t>(i)], p) -= solve.X(i, j);
    }
  }
  return T;
}

RecursiveQuotient RecursiveQuotient::pencilChildOver(
    const std::vector<cd> &op,
    const std::vector<RetainedCoordinate> &coordinates,
    const std::vector<std::vector<int>> &components,
    const Options &options,
    const std::vector<std::shared_ptr<ComponentSolve>> &solves) const {
  const int reduced = static_cast<int>(coordinates.size());
  const Eigen::MatrixXcd T = pencilConstraintModes(coordinates, solves);
  const Eigen::MatrixXcd gram = T.transpose() * (pencilMetric_ * T);
  RecursiveQuotient child;
  child.pencil_ = true;
  child.pencilMetric_ = gram.sparseView();
  child.pencilMetric_.makeCompressed();
  child.level_ = level_ + 1;
  child.provenance_.resize(static_cast<std::size_t>(reduced));
  for (int coordinate = 0; coordinate < reduced; ++coordinate)
    child.provenance_[static_cast<std::size_t>(coordinate)] =
        "L" + std::to_string(level_) + ":" +
        coordinates[static_cast<std::size_t>(coordinate)].provenance;
  Options childOptions = options;
  childOptions.embeddingPolicy = FiberEmbeddingPolicy::CarryGramExactly;
  child.initMatrix(op, reduced, {}, components, childOptions);
  return child;
}

RecursiveQuotient RecursiveQuotient::nextLevel(
    const std::vector<std::vector<int>> &components,
    const Options &options) const {
  const StaticReductionRead &reduction = staticReduction();
  RecursiveQuotient child;
  if (pencil_) {
    std::vector<std::shared_ptr<ComponentSolve>> solves;
    for (int component = 0; component < componentCount(); ++component)
      solves.push_back(componentSolve(component));
    child = pencilChildOver(reduction.effectiveOperator, reduction.coordinates,
                            components, options, solves);
  } else {
    child = childOver(reduction.effectiveOperator, reduction.coordinates,
                      components, options);
  }
  child.levelProvenance_.origin = LevelOrigin::StaticResponse;
  // A static level carries no window: lambda = 0 is a point, not a band, so
  // the window fields stay NaN.
  child.levelProvenance_.solveResidual = reduction.solveResidual;
  child.levelProvenance_.compatibilityResidual = reduction.compatibilityResidual;
  child.levelProvenance_.certificate = reduction.certificate;
  return child;
}

RecursiveQuotient RecursiveQuotient::nextLevel(
    const std::vector<std::vector<int>> &components) const {
  return nextLevel(components, options_);
}

RecursiveQuotient RecursiveQuotient::nextLevelAtLambda(
    const std::vector<std::vector<int>> &components, cd lambda,
    double windowLower, double windowUpper, const Options &options) const {
  // R_{l+1}(lambda) = Feshbach_{P_l}(R_l(lambda)): the child's operator is the
  // exact energy-dependent response at the declared lambda, not the static
  // complement, which does not preserve the nonzero spectrum.
  const FeshbachRead response = feshbach(lambda, windowLower, windowUpper);
  RecursiveQuotient child =
      pencil_ ? pencilChildOver(response.response, response.coordinates,
                                components, options, shiftedSolves(lambda))
              : childOver(response.response, response.coordinates, components,
                          options);
  child.levelProvenance_.origin = LevelOrigin::BandPencil;
  child.levelProvenance_.lambda = lambda;
  child.levelProvenance_.windowLower = windowLower;
  child.levelProvenance_.windowUpper = windowUpper;
  child.levelProvenance_.solveResidual = response.solveResidual;
  child.levelProvenance_.compatibilityResidual = response.compatibilityResidual;
  child.levelProvenance_.resonant = response.resonant;
  child.levelProvenance_.certificate = response.certificate;
  return child;
}

RecursiveQuotient RecursiveQuotient::nextLevelAtLambda(
    const std::vector<std::vector<int>> &components, cd lambda,
    double windowLower, double windowUpper) const {
  return nextLevelAtLambda(components, lambda, windowLower, windowUpper,
                           options_);
}

RecursiveQuotient RecursiveQuotient::nextLevelFromSurrogate(
    const std::vector<std::vector<int>> &components, double windowLower,
    double windowUpper, double modeCutoff, double residualTolerance,
    const Options &options) const {
  const CraigBamptonRead surrogate =
      craigBampton(windowLower, windowUpper, modeCutoff, residualTolerance);
  const std::size_t basisColumns =
      surrogate.basis.empty()
          ? 0
          : surrogate.basis.size() / static_cast<std::size_t>(dim_);
  const int columns = static_cast<int>(basisColumns);
  const Eigen::MatrixXcd basis =
      toMatrix(surrogate.basis, dim_, columns, "surrogate basis");
  const Eigen::MatrixXcd stiffness =
      toMatrix(surrogate.reducedStiffness, columns, columns,
               "surrogate reduced stiffness");
  const Eigen::MatrixXcd mass = toMatrix(surrogate.reducedMass, columns,
                                         columns, "surrogate reduced mass");

  // A level carries a diagonal chain metric while the surrogate's reduced
  // pencil (K, M) is generalized. On the M-orthonormalized basis V M^{-1/2} the
  // child metric is the identity and its operator M^{-1/2} K M^{-1/2}; that
  // congruence preserves the generalized eigenvalues of (K, M) exactly, leaving
  // the approximation entirely in the truncation.
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> massSolver(mass);
  if (massSolver.info() != Eigen::Success)
    throw std::invalid_argument(
        "nextLevelFromSurrogate: reduced mass eigendecomposition failed");
  const Eigen::VectorXd massValues = massSolver.eigenvalues();
  const double largest = massValues.size() ? massValues(massValues.size() - 1)
                                           : 0.0;
  const double floorValue = options_.rankTolerance * std::max(largest, 1e-300);
  Eigen::VectorXcd inverseRootValues(massValues.size());
  for (Eigen::Index i = 0; i < massValues.size(); ++i) {
    if (massValues(i) <= floorValue)
      throw std::invalid_argument(
          "nextLevelFromSurrogate: reduced mass is singular — the surrogate "
          "basis is rank deficient and no M-orthonormalization exists");
    inverseRootValues(i) = cd(1.0 / std::sqrt(massValues(i)), 0.0);
  }
  const Eigen::MatrixXcd inverseRoot =
      massSolver.eigenvectors() * inverseRootValues.asDiagonal() *
      massSolver.eigenvectors().adjoint();
  const Eigen::MatrixXcd childOperator = inverseRoot * stiffness * inverseRoot;
  const Eigen::MatrixXcd childBasis = basis * inverseRoot;

  // The child's coordinates are the M-orthonormalized surrogate modes:
  // retained interior coordinates, each a stalk coordinate of the surrogate
  // rather than a fine cell of this level.
  std::vector<RetainedCoordinate> coordinates(
      static_cast<std::size_t>(columns));
  for (int j = 0; j < columns; ++j) {
    RetainedCoordinate &coordinate = coordinates[static_cast<std::size_t>(j)];
    coordinate.kind = RetainedCoordinateKind::Selected;
    coordinate.component = 0;
    coordinate.fineIndex = -1;
    coordinate.embedding.resize(static_cast<std::size_t>(dim_));
    for (int i = 0; i < dim_; ++i)
      coordinate.embedding[static_cast<std::size_t>(i)] = childBasis(i, j);
    coordinate.provenance = "amls#" + std::to_string(j);
  }

  RecursiveQuotient child;
  if (pencil_) {
    // A pencil level carries the congruence (V^T A V, V^T M V); the M^{-1/2}
    // orthonormalization above is Hermitian-only and is not applied to a
    // complex symmetric M.
    const Eigen::MatrixXcd A = Eigen::MatrixXcd(op_);
    const Eigen::MatrixXcd congruentA = basis.transpose() * A * basis;
    const Eigen::MatrixXcd congruentM =
        basis.transpose() * (pencilMetric_ * basis);
    for (int j = 0; j < columns; ++j)
      for (int i = 0; i < dim_; ++i)
        coordinates[static_cast<std::size_t>(j)]
            .embedding[static_cast<std::size_t>(i)] = basis(i, j);
    child.pencil_ = true;
    child.pencilMetric_ = congruentM.sparseView();
    child.pencilMetric_.makeCompressed();
    child.level_ = level_ + 1;
    child.provenance_.resize(static_cast<std::size_t>(columns));
    for (int j = 0; j < columns; ++j)
      child.provenance_[static_cast<std::size_t>(j)] =
          "L" + std::to_string(level_) + ":" +
          coordinates[static_cast<std::size_t>(j)].provenance;
    Options childOptions = options;
    childOptions.embeddingPolicy = FiberEmbeddingPolicy::CarryGramExactly;
    child.initMatrix(toFlat(congruentA), columns, {}, components, childOptions);
  } else {
    child = childOver(toFlat(childOperator), coordinates, components, options);
  }
  child.levelProvenance_.origin = LevelOrigin::Surrogate;
  child.levelProvenance_.windowLower = surrogate.windowLower;
  child.levelProvenance_.windowUpper = surrogate.windowUpper;
  child.levelProvenance_.discardedModeGap = surrogate.discardedModeGap;
  child.levelProvenance_.surrogateResidual =
      surrogate.eigenResiduals.empty()
          ? kNaN
          : *std::max_element(surrogate.eigenResiduals.begin(),
                              surrogate.eigenResiduals.end());
  child.levelProvenance_.certificate = surrogate.certificate;
  return child;
}

RecursiveQuotient RecursiveQuotient::nextLevelFromSurrogate(
    const std::vector<std::vector<int>> &components, double windowLower,
    double windowUpper, double modeCutoff, double residualTolerance) const {
  return nextLevelFromSurrogate(components, windowLower, windowUpper,
                                modeCutoff, residualTolerance, options_);
}

void RecursiveQuotient::invalidate() {
  static_.reset();
  solves_.clear();
  shifted_.clear();
  if (st_) {
    HodgeLaplacian hodge(st_);
    const std::vector<cd> flat = hodge.laplacian(degree_);
    const Eigen::MatrixXcd dense = toMatrix(flat, dim_, dim_, "laplacian");
    op_ = dense.sparseView();
    op_.makeCompressed();
    opNorm_ = dense.norm();
    const std::vector<cd> weights = hodge.weights(degree_);
    for (int i = 0; i < dim_ && i < static_cast<int>(weights.size()); ++i)
      weights_(i) = weights[static_cast<std::size_t>(i)];
    detectRegime();
  }
}

}  // namespace tessera::cobordism
