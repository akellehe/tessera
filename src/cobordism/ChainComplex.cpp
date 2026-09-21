// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "cobordism/ChainComplex.h"

#include <Eigen/Dense>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "cobordism/IntegerLinalg.h"
#include "mesh/Simplex.h"
#include "mesh/Vertex.h"
#include "spacetime/Spacetime.h"

namespace tessera::cobordism {

using Face = std::vector<std::uint64_t>;  // sorted vertex ids

namespace {
// Sorted vertex ids of a simplex — the homological reference ordering.
Face sortedIds(const SimplexPtr &s) {
  Face ids;
  for (const auto &v : s->getVertices()) ids.push_back(v->getId());
  std::sort(ids.begin(), ids.end());
  return ids;
}
}  // namespace

ChainComplex ChainComplex::fromSpacetime(const Spacetime &K) {
  ChainComplex cc;

  // Face lattice via Simplex::getFacets: a BFS down from the top cells,
  // de-duplicated by fingerprint and bucketed by dimension.
  std::map<int, std::vector<SimplexPtr>> byDim;
  std::unordered_set<std::uint64_t> seen;
  // Seed from the top-dimensional cells only: the mesh keeps orphaned
  // sub-simplices belonging to no current top cell, which would add spurious
  // cycles and negative Betti numbers.
  std::size_t topSize = 0;
  for (const auto &s : K.getSimplices())
    if (s != nullptr) topSize = std::max(topSize, static_cast<std::size_t>(s->size()));
  std::vector<SimplexPtr> stack;
  for (const auto &s : K.getSimplices())
    if (s != nullptr && s->size() == topSize) stack.push_back(s);
  while (!stack.empty()) {
    SimplexPtr s = stack.back();
    stack.pop_back();
    if (s == nullptr) continue;
    if (!seen.insert(s->fingerprint.fingerprint()).second) continue;
    const int k = static_cast<int>(s->size()) - 1;
    byDim[k].push_back(s);
    if (k >= 1)
      for (const auto &f : s->getFacets()) stack.push_back(f);
  }

  if (byDim.empty()) return cc;  // empty complex
  const int n = byDim.rbegin()->first;
  cc.dimension_ = n;

  // Order each dimension deterministically (by sorted vertex ids) and index
  // the simplices by fingerprint for boundary lookups.
  std::vector<std::vector<SimplexPtr>> faces(n + 1);
  std::vector<std::unordered_map<std::uint64_t, int>> index(n + 1);
  cc.counts_.assign(n + 1, 0);
  cc.faceVerts_.assign(n + 1, {});
  for (int k = 0; k <= n; ++k) {
    auto &vec = byDim[k];
    std::sort(vec.begin(), vec.end(), [](const SimplexPtr &a, const SimplexPtr &b) {
      return sortedIds(a) < sortedIds(b);
    });
    faces[k] = vec;
    cc.counts_[k] = vec.size();
    cc.faceVerts_[k].reserve(vec.size());
    for (int j = 0; j < static_cast<int>(vec.size()); ++j) {
      index[k][vec[j]->fingerprint.fingerprint()] = j;
      cc.faceVerts_[k].push_back(sortedIds(vec[j]));
    }
  }

  // Boundary ∂_k (rows = |C_{k-1}|, cols = |C_k|): each column is a k-simplex
  // and its nonzero rows are its facets. getFacets() is in canonical order —
  // facet i drops vertex i — so the coefficient is (-1)^i.
  cc.entries_.assign(n + 1, {});
  for (int k = 1; k <= n; ++k) {
    const int cols = static_cast<int>(cc.counts_[k]);
    auto &entries = cc.entries_[k];
    entries.reserve(static_cast<std::size_t>(cols) * (k + 1));
    for (int j = 0; j < cols; ++j) {
      const auto &facets = faces[k][j]->getFacets();
      for (int i = 0; i < static_cast<int>(facets.size()); ++i) {
        const int r = index[k - 1].at(facets[i]->fingerprint.fingerprint());
        entries.push_back({r, j, (i % 2 == 0) ? 1 : -1});
      }
    }
  }
  return cc;
}

ChainComplex ChainComplex::fromTopCells(
    const std::vector<std::vector<std::uint64_t>> &topCells) {
  ChainComplex cc;
  if (topCells.empty()) return cc;
  std::size_t nv = 0;
  for (const auto &raw : topCells) nv = std::max(nv, raw.size());
  if (nv == 0) return cc;
  const int n = static_cast<int>(nv) - 1;
  cc.dimension_ = n;

  // Face closure: every subset of every (sorted) top cell, bucketed by
  // dimension in a set so the order is lexicographic on sorted tuples.
  std::vector<std::set<Face>> faces(static_cast<std::size_t>(n) + 1);
  for (const auto &raw : topCells) {
    if (raw.size() != nv)
      throw std::invalid_argument(
          "ChainComplex::fromTopCells: every top cell must have the same number "
          "of vertices (pure complex)");
    Face cell(raw);
    std::sort(cell.begin(), cell.end());
    if (std::adjacent_find(cell.begin(), cell.end()) != cell.end())
      throw std::invalid_argument("ChainComplex::fromTopCells: a cell repeats a vertex");
    // Enumerate subsets by bitmask over the (n+1) vertices.
    const unsigned full = 1u << nv;
    for (unsigned mask = 1; mask < full; ++mask) {
      Face f;
      for (std::size_t i = 0; i < nv; ++i)
        if (mask & (1u << i)) f.push_back(cell[i]);
      faces[f.size() - 1].insert(std::move(f));
    }
  }

  cc.counts_.assign(static_cast<std::size_t>(n) + 1, 0);
  cc.faceVerts_.assign(static_cast<std::size_t>(n) + 1, {});
  std::vector<std::map<Face, int>> index(static_cast<std::size_t>(n) + 1);
  for (int k = 0; k <= n; ++k) {
    auto &fk = cc.faceVerts_[static_cast<std::size_t>(k)];
    fk.assign(faces[static_cast<std::size_t>(k)].begin(), faces[static_cast<std::size_t>(k)].end());
    cc.counts_[static_cast<std::size_t>(k)] = fk.size();
    for (int j = 0; j < static_cast<int>(fk.size()); ++j)
      index[static_cast<std::size_t>(k)][fk[static_cast<std::size_t>(j)]] = j;
  }

  cc.entries_.assign(static_cast<std::size_t>(n) + 1, {});
  for (int k = 1; k <= n; ++k) {
    const int cols = static_cast<int>(cc.counts_[static_cast<std::size_t>(k)]);
    const auto &fk = cc.faceVerts_[static_cast<std::size_t>(k)];
    auto &entries = cc.entries_[static_cast<std::size_t>(k)];
    entries.reserve(static_cast<std::size_t>(cols) * (static_cast<std::size_t>(k) + 1));
    for (int j = 0; j < cols; ++j) {
      const Face &cell = fk[static_cast<std::size_t>(j)];
      for (int i = 0; i <= k; ++i) {
        Face facet;
        facet.reserve(cell.size() - 1);
        for (int p = 0; p <= k; ++p)
          if (p != i) facet.push_back(cell[static_cast<std::size_t>(p)]);
        const int r = index[static_cast<std::size_t>(k) - 1].at(facet);
        entries.push_back({r, j, (i % 2 == 0) ? 1 : -1});
      }
    }
  }
  return cc;
}

std::vector<std::vector<int>> ChainComplex::orientationSigns() const {
  std::vector<std::vector<int>> signs(static_cast<std::size_t>(std::max(dimension_ + 1, 0)));
  if (dimension_ < 0) return signs;
  signs[0].assign(counts_[0], 1);
  for (int k = 1; k <= dimension_; ++k) {
    const std::size_t rows = counts_[static_cast<std::size_t>(k) - 1];
    const std::size_t cols = counts_[static_cast<std::size_t>(k)];
    const auto &flat = boundaryMatrix(k);
    const auto &cells = faceVerts_[static_cast<std::size_t>(k)];
    std::map<Face, std::size_t> facetIndex;
    for (std::size_t r = 0; r < rows; ++r) facetIndex[faceVerts_[static_cast<std::size_t>(k) - 1][r]] = r;
    signs[static_cast<std::size_t>(k)].assign(cols, 1);
    for (std::size_t j = 0; j < cols; ++j) {
      // The facet dropping the smallest vertex: reference coefficient (-1)^0 = +1.
      Face facet(cells[j].begin() + 1, cells[j].end());
      const auto it = facetIndex.find(facet);
      if (it == facetIndex.end())
        throw std::runtime_error("ChainComplex::orientationSigns: a facet is missing from C_" +
                                 std::to_string(k - 1));
      const long entry = flat[it->second * cols + j];
      if (entry != 1 && entry != -1)
        throw std::runtime_error("ChainComplex::orientationSigns: stored boundary entry of modulus != 1");
      signs[static_cast<std::size_t>(k)][j] =
          static_cast<int>(entry) * signs[static_cast<std::size_t>(k) - 1][it->second];
    }
    // Verify the whole map: stored = D_{k-1} ref D_k, entry by entry.
    for (std::size_t j = 0; j < cols; ++j)
      for (std::size_t i = 0; i <= static_cast<std::size_t>(k); ++i) {
        Face facet;
        for (std::size_t p = 0; p <= static_cast<std::size_t>(k); ++p)
          if (p != i) facet.push_back(cells[j][p]);
        const std::size_t r = facetIndex.at(facet);
        const long ref = (i % 2 == 0) ? 1 : -1;
        const long expected = ref * signs[static_cast<std::size_t>(k) - 1][r] * signs[static_cast<std::size_t>(k)][j];
        if (flat[r * cols + j] != expected)
          throw std::runtime_error("ChainComplex::orientationSigns: the stored boundary map at degree " +
                                   std::to_string(k) + " is not a signed-permutation image of the reference map");
      }
  }
  return signs;
}

std::size_t ChainComplex::numSimplices(int k) const noexcept {
  if (k < 0 || k > dimension_) return 0;
  return counts_[static_cast<std::size_t>(k)];
}

int ChainComplex::eulerCharacteristic() const noexcept {
  int chi = 0;
  for (int k = 0; k <= dimension_; ++k)
    chi += (k % 2 == 0 ? 1 : -1) * static_cast<int>(counts_[static_cast<std::size_t>(k)]);
  return chi;
}

const std::vector<ChainComplex::BoundaryEntry> &ChainComplex::boundaryEntries(int k) const {
  static const std::vector<BoundaryEntry> kEmpty{};
  if (k < 0 || k > dimension_) return kEmpty;
  return entries_[static_cast<std::size_t>(k)];
}

const std::vector<long> &ChainComplex::boundaryMatrix(int k) const {
  static const std::vector<long> kEmpty{};
  if (k < 0 || k > dimension_) return kEmpty;
  const std::lock_guard<std::mutex> lock(dense_->mutex);
  // Sized once, before any reference into it exists; every copy of a complex
  // has the same dimension, so a shared cache is sized the same by all.
  if (dense_->flat.empty()) {
    dense_->flat.assign(static_cast<std::size_t>(dimension_) + 1, {});
    dense_->built.assign(static_cast<std::size_t>(dimension_) + 1, 0);
  }
  auto &flat = dense_->flat[static_cast<std::size_t>(k)];
  if (!dense_->built[static_cast<std::size_t>(k)]) {
    if (k >= 1) {
      const std::size_t rows = counts_[static_cast<std::size_t>(k) - 1];
      const std::size_t cols = counts_[static_cast<std::size_t>(k)];
      flat.assign(rows * cols, 0);
      for (const auto &e : entries_[static_cast<std::size_t>(k)])
        flat[static_cast<std::size_t>(e.row) * cols + static_cast<std::size_t>(e.column)] = e.value;
    }
    dense_->built[static_cast<std::size_t>(k)] = 1;
  }
  return flat;
}

bool ChainComplex::boundaryComposesToZero() const {
  // ∂_{k-1} ∘ ∂_k = 0, column by column: the column of ∂_k for a k-simplex
  // lists its facets, and each facet's column of ∂_{k-1} lists the ridges, so
  // the composite's column is a signed sum over a handful of ridges.
  for (int k = 2; k <= dimension_; ++k) {
    const auto &outer = entries_[static_cast<std::size_t>(k)];
    const auto &inner = entries_[static_cast<std::size_t>(k) - 1];
    // Entries are grouped by ascending column: start[c] .. start[c+1] is column c.
    const std::size_t innerCols = counts_[static_cast<std::size_t>(k) - 1];
    std::vector<std::size_t> start(innerCols + 1, 0);
    for (const auto &e : inner) ++start[static_cast<std::size_t>(e.column) + 1];
    for (std::size_t c = 0; c < innerCols; ++c) start[c + 1] += start[c];
    std::map<int, long> column;
    for (std::size_t begin = 0; begin < outer.size();) {
      std::size_t end = begin;
      while (end < outer.size() && outer[end].column == outer[begin].column) ++end;
      column.clear();
      for (std::size_t m = begin; m < end; ++m) {
        const auto facet = static_cast<std::size_t>(outer[m].row);
        for (std::size_t q = start[facet]; q < start[facet + 1]; ++q)
          column[inner[q].row] += static_cast<long>(outer[m].value) * inner[q].value;
      }
      for (const auto &entry : column)
        if (entry.second != 0) return false;
      begin = end;
    }
  }
  return true;
}

int ChainComplex::rankOfBoundary(int k) const {
  if (k < 1 || k > dimension_) return 0;
  return integerRank(boundaryMatrix(k),
                     static_cast<int>(counts_[k - 1]), static_cast<int>(counts_[k]));
}

int ChainComplex::gf2RankOfBoundary(int k) const {
  if (k < 1 || k > dimension_) return 0;
  const auto &M = boundaryMatrix(k);
  std::vector<int> bits(M.size());
  for (std::size_t i = 0; i < M.size(); ++i) bits[i] = static_cast<int>(M[i] & 1);
  return gf2Rank(std::move(bits), static_cast<int>(counts_[k - 1]),
                 static_cast<int>(counts_[k]));
}

std::vector<int> ChainComplex::bettiNumbers() const {
  std::vector<int> b;
  if (dimension_ < 0) return b;
  b.assign(dimension_ + 1, 0);
  for (int k = 0; k <= dimension_; ++k)
    b[k] = static_cast<int>(counts_[k]) - rankOfBoundary(k) - rankOfBoundary(k + 1);
  return b;
}

std::vector<int> ChainComplex::bettiNumbersGF2() const {
  std::vector<int> b;
  if (dimension_ < 0) return b;
  b.assign(dimension_ + 1, 0);
  for (int k = 0; k <= dimension_; ++k)
    b[k] = static_cast<int>(counts_[k]) - gf2RankOfBoundary(k) - gf2RankOfBoundary(k + 1);
  return b;
}

std::vector<std::vector<std::uint64_t>> ChainComplex::kSimplexVertices(int k) const {
  if (k < 0 || k > dimension_) return {};  // out of range: no such simplices
  return faceVerts_[static_cast<std::size_t>(k)];
}

std::vector<std::vector<std::uint64_t>> ChainComplex::orientedTopSimplices() const {
  return kSimplexVertices(dimension_);  // empty complex (d < 0) yields {}
}

std::vector<int> ChainComplex::fundamentalClass() const {
  // [W] ∈ H_d is the ±1 generator of ker ∂_d, relative to the
  // increasing-vertex reference orientation of each top simplex.
  const int d = dimension_;
  if (d < 1)
    throw std::runtime_error(
        "ChainComplex::fundamentalClass: a closed oriented manifold of "
        "dimension >= 1 is required");
  const int rows = static_cast<int>(counts_[static_cast<std::size_t>(d - 1)]);
  const int cols = static_cast<int>(counts_[static_cast<std::size_t>(d)]);
  Eigen::MatrixXd topBoundary(rows, cols);
  const auto &flat = boundaryMatrix(d);
  for (int r = 0; r < rows; ++r)
    for (int c = 0; c < cols; ++c)
      topBoundary(r, c) =
          static_cast<double>(flat[static_cast<std::size_t>(r) * cols + c]);

  // dimensionOfKernel() = cols − rank is the true dim ker ∂_d. Eigen's
  // FullPivLU::kernel() returns one all-zero column for a 0-dimensional kernel,
  // so kernel().cols() cannot distinguish b_d = 0 from b_d = 1.
  const Eigen::FullPivLU<Eigen::MatrixXd> decomposition(topBoundary);
  if (decomposition.dimensionOfKernel() != 1)
    throw std::runtime_error(
        "ChainComplex::fundamentalClass: a closed connected oriented " +
        std::to_string(d) + "-manifold is required (dim ker ∂_" +
        std::to_string(d) + " = b_" + std::to_string(d) +
        " must be 1, so the fundamental class is unique up to sign)");
  const Eigen::MatrixXd kernel = decomposition.kernel();

  // All entries share one magnitude, so scaling by the first nonzero entry
  // makes them exactly ±1 and fixes the overall sign.
  Eigen::VectorXd generator = kernel.col(0);
  const double scale = generator.cwiseAbs().maxCoeff();
  const double threshold = 1e-9 * (scale > 0.0 ? scale : 1.0);
  int firstNonzero = 0;
  while (firstNonzero < generator.size() &&
         std::abs(generator[firstNonzero]) <= threshold)
    ++firstNonzero;
  if (firstNonzero == generator.size())
    throw std::runtime_error(
        "ChainComplex::fundamentalClass: the generator of ker ∂_" +
        std::to_string(d) + " is numerically zero (no fundamental class)");
  generator /= generator[firstNonzero];

  std::vector<int> epsilon(static_cast<std::size_t>(cols), 0);
  for (int i = 0; i < generator.size(); ++i)
    epsilon[static_cast<std::size_t>(i)] =
        static_cast<int>(std::lround(generator[i]));
  return epsilon;
}

// Intersection form, computed without geometry:
//
//   1. A basis of H^2 as triangle cochains: closed but not exact, one per
//      two-dimensional hole.
//   2. Pair them with the Alexander-Whitney cup product: on a four-simplex
//      v0<v1<v2<v3<v4 the first cochain is evaluated on the front triangle
//      (v0,v1,v2) and the second on the back triangle (v2,v3,v4).
//   3. Sum over the manifold, each four-simplex weighted by its ±1 orientation
//      from the fundamental class.
std::vector<double> ChainComplex::intersectionForm() const {
  if (dimension_ != 4) return {};
  const int numTwoDimensionalHoles = bettiNumbers()[2];  // rank of H_2
  if (numTwoDimensionalHoles == 0) return {};

  const int numEdges = static_cast<int>(counts_[1]);
  const int numTriangles = static_cast<int>(counts_[2]);
  const int numTetrahedra = static_cast<int>(counts_[3]);
  const int numFourSimplices = static_cast<int>(counts_[4]);

  auto boundaryMatrixAsEigen = [&](int k, int rows, int cols) {
    Eigen::MatrixXd matrix(rows, cols);
    const auto &flat = boundaryMatrix(k);
    for (int row = 0; row < rows; ++row)
      for (int col = 0; col < cols; ++col)
        matrix(row, col) =
            static_cast<double>(flat[static_cast<std::size_t>(row) * cols + col]);
    return matrix;
  };
  // Boundary maps: each sends a cell to the (signed) sum of its faces.
  const Eigen::MatrixXd triangleBoundaries =
      boundaryMatrixAsEigen(2, numEdges, numTriangles);          // triangles -> edges
  const Eigen::MatrixXd tetrahedronBoundaries =
      boundaryMatrixAsEigen(3, numTriangles, numTetrahedra);     // tetrahedra -> triangles

  // The ±1 generator of ker ∂_4; throws unless the complex is a closed
  // orientable 4-manifold.
  const std::vector<int> orientationPerFourSimplex = fundamentalClass();

  // Closed triangle cochains are the null space of the transposed tetrahedron
  // boundary map; exact ones are the columns of the transposed triangle
  // boundary map. The basis is the closed cochains independent of the exact.
  const Eigen::MatrixXd closedTriangleCochains =
      Eigen::FullPivLU<Eigen::MatrixXd>(tetrahedronBoundaries.transpose()).kernel();
  const Eigen::MatrixXd exactTriangleCochains = triangleBoundaries.transpose();

  const double zeroTolerance = 1e-9;
  auto numericalRank = [&](const Eigen::MatrixXd &matrix) {
    if (matrix.cols() == 0) return 0;
    Eigen::FullPivLU<Eigen::MatrixXd> decomposition(matrix);
    decomposition.setThreshold(zeroTolerance);
    return static_cast<int>(decomposition.rank());
  };
  Eigen::MatrixXd spannedSoFar = exactTriangleCochains;
  int spannedRank = numericalRank(spannedSoFar);
  std::vector<Eigen::VectorXd> cohomologyBasis;
  for (int j = 0; j < closedTriangleCochains.cols() &&
                  static_cast<int>(cohomologyBasis.size()) < numTwoDimensionalHoles;
       ++j) {
    Eigen::MatrixXd augmented(numTriangles, spannedSoFar.cols() + 1);
    if (spannedSoFar.cols() > 0) augmented.leftCols(spannedSoFar.cols()) = spannedSoFar;
    augmented.col(spannedSoFar.cols()) = closedTriangleCochains.col(j);
    if (numericalRank(augmented) > spannedRank) {  // genuinely new cohomology class
      cohomologyBasis.push_back(closedTriangleCochains.col(j));
      spannedSoFar = augmented;
      ++spannedRank;
    }
  }

  // Triangle index from its sorted vertices, for the cup-product faces below.
  std::map<std::array<std::uint64_t, 3>, int> triangleIndexByVertices;
  for (int j = 0; j < numTriangles; ++j) {
    const auto &vertices = faceVerts_[2][static_cast<std::size_t>(j)];
    triangleIndexByVertices[{vertices[0], vertices[1], vertices[2]}] = j;
  }

  // Cup product summed over the oriented manifold (step 2 + 3 above).
  const int numClasses = static_cast<int>(cohomologyBasis.size());
  std::vector<double> intersectionMatrix(
      static_cast<std::size_t>(numClasses) * numClasses, 0.0);
  for (int s = 0; s < numFourSimplices; ++s) {
    const auto &vertices = faceVerts_[4][static_cast<std::size_t>(s)];
    const int frontTriangle =
        triangleIndexByVertices.at({vertices[0], vertices[1], vertices[2]});
    const int backTriangle =
        triangleIndexByVertices.at({vertices[2], vertices[3], vertices[4]});
    const double orientation =
        static_cast<double>(orientationPerFourSimplex[static_cast<std::size_t>(s)]);
    for (int a = 0; a < numClasses; ++a)
      for (int b = 0; b < numClasses; ++b)
        intersectionMatrix[static_cast<std::size_t>(a) * numClasses + b] +=
            orientation * cohomologyBasis[a][frontTriangle] *
            cohomologyBasis[b][backTriangle];
  }
  // The crossing pairing is symmetric; average away any numerical asymmetry.
  for (int a = 0; a < numClasses; ++a)
    for (int b = a + 1; b < numClasses; ++b) {
      const double mean =
          0.5 * (intersectionMatrix[static_cast<std::size_t>(a) * numClasses + b] +
                 intersectionMatrix[static_cast<std::size_t>(b) * numClasses + a]);
      intersectionMatrix[static_cast<std::size_t>(a) * numClasses + b] = mean;
      intersectionMatrix[static_cast<std::size_t>(b) * numClasses + a] = mean;
    }
  return intersectionMatrix;
}

int ChainComplex::signature() const {
  const std::vector<double> intersectionMatrix = intersectionForm();
  if (intersectionMatrix.empty()) return 0;
  const int size = static_cast<int>(
      std::lround(std::sqrt(static_cast<double>(intersectionMatrix.size()))));
  Eigen::MatrixXd form(size, size);
  for (int row = 0; row < size; ++row)
    for (int col = 0; col < size; ++col)
      form(row, col) = intersectionMatrix[static_cast<std::size_t>(row) * size + col];
  // Signature = (number of positive eigenvalues) - (number of negative ones).
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(form, Eigen::EigenvaluesOnly);
  double largestMagnitude = 0.0;
  for (int i = 0; i < size; ++i)
    largestMagnitude = std::max(largestMagnitude, std::abs(solver.eigenvalues()[i]));
  // Relative threshold for "nonzero"; the form is unimodular on a closed
  // 4-manifold, so its eigenvalues sit well away from zero.
  const double zeroTolerance = 1e-7 * (largestMagnitude > 0 ? largestMagnitude : 1.0);
  int numPositive = 0, numNegative = 0;
  for (int i = 0; i < size; ++i) {
    const double eigenvalue = solver.eigenvalues()[i];
    if (eigenvalue > zeroTolerance) ++numPositive;
    else if (eigenvalue < -zeroTolerance) ++numNegative;
  }
  return numPositive - numNegative;
}

std::vector<long> ChainComplex::torsion(int k) const {
  std::vector<long> out;
  if (k < 0 || k + 1 > dimension_) return out;  // torsion of H_k comes from ∂_{k+1}
  const int kk = k + 1;
  auto snf = smithNormalForm(boundaryMatrix(kk),
                             static_cast<int>(counts_[kk - 1]),
                             static_cast<int>(counts_[kk]));
  for (long d : snf.invariantFactors)
    if (d > 1) out.push_back(d);
  return out;
}

// ===========================================================================
// Stiefel–Whitney numbers (mod-2 characteristic numbers)
// ===========================================================================
//
// Read off the mod-2 cohomology ring:
//
//   1. H^k(K; Z/2): cocycles modulo coboundaries, the coboundary being the
//      transpose of the mod-2 boundary operator.
//   2. Cup product on cochains via Alexander–Whitney (front/back faces).
//   3. Wu classes v_k, defined by <v_k ∪ x, [K]> = <Sq^k x, [K]> for every
//      x in H^{n-k}; the defining system's matrix is the Poincaré-duality
//      pairing.
//   4. w = Sq(v); each Stiefel–Whitney number is a degree-n monomial in the
//      w_i evaluated on [K] (the mod-2 sum of all top simplices).
//
// Only the Steenrod squares expressible through the ordinary cup product are
// implemented (Sq^k on a degree-k class is the cup square; Sq^k on a lower
// degree class is zero). A class that needs a higher cup-i product raises.
namespace {

using Gf2Vector = std::vector<std::uint8_t>;  // dense vector over GF(2), entries 0/1
using Gf2Matrix = std::vector<Gf2Vector>;     // a list of rows, each the same length

// Reduce `rows` to reduced row-echelon form in place. Returns the pivot column
// of each surviving (nonzero) row; its size is the rank.
std::vector<int> gf2ReduceRows(Gf2Matrix &rows, int numColumns) {
  std::vector<int> pivotColumns;
  int pivotRow = 0;
  for (int column = 0; column < numColumns && pivotRow < static_cast<int>(rows.size());
       ++column) {
    int found = -1;
    for (int r = pivotRow; r < static_cast<int>(rows.size()); ++r)
      if (rows[r][column]) { found = r; break; }
    if (found < 0) continue;
    std::swap(rows[pivotRow], rows[found]);
    for (int r = 0; r < static_cast<int>(rows.size()); ++r)
      if (r != pivotRow && rows[r][column])
        for (int c = 0; c < numColumns; ++c) rows[r][c] ^= rows[pivotRow][c];
    pivotColumns.push_back(column);
    ++pivotRow;
  }
  return pivotColumns;
}

// Basis of the null space {x : matrix·x = 0} of a GF(2) matrix with `numColumns`
// columns. With no rows, every standard basis vector is a kernel vector.
Gf2Matrix gf2Kernel(Gf2Matrix matrix, int numColumns) {
  const std::vector<int> pivotColumns = gf2ReduceRows(matrix, numColumns);
  std::vector<char> isPivot(numColumns, 0);
  for (int p : pivotColumns) isPivot[p] = 1;
  Gf2Matrix basis;
  for (int freeColumn = 0; freeColumn < numColumns; ++freeColumn) {
    if (isPivot[freeColumn]) continue;
    Gf2Vector x(numColumns, 0);
    x[freeColumn] = 1;
    for (int t = 0; t < static_cast<int>(pivotColumns.size()); ++t)
      x[pivotColumns[t]] = matrix[t][freeColumn];
    basis.push_back(std::move(x));
  }
  return basis;
}

// Incrementally maintained spanning set, kept in echelon form. add() records
// `candidate` and returns true iff it is independent of everything added so far.
struct Gf2Span {
  Gf2Matrix echelonRows;
  std::vector<int> leadingColumn;
  int numColumns;
  explicit Gf2Span(int columns) : numColumns(columns) {}
  bool add(Gf2Vector candidate) {
    for (int t = 0; t < static_cast<int>(echelonRows.size()); ++t)
      if (candidate[leadingColumn[t]])
        for (int c = 0; c < numColumns; ++c) candidate[c] ^= echelonRows[t][c];
    int lead = -1;
    for (int c = 0; c < numColumns; ++c)
      if (candidate[c]) { lead = c; break; }
    if (lead < 0) return false;  // already in the span
    echelonRows.push_back(std::move(candidate));
    leadingColumn.push_back(lead);
    return true;
  }
};

// Solve matrix·x = rhs over GF(2) for a square, invertible matrix. Throws if
// the system is not uniquely solvable.
Gf2Vector gf2Solve(const Gf2Matrix &matrix, const Gf2Vector &rhs) {
  const int n = static_cast<int>(rhs.size());
  Gf2Matrix augmented(matrix.size(), Gf2Vector(n + 1, 0));
  for (int r = 0; r < static_cast<int>(matrix.size()); ++r) {
    for (int c = 0; c < n; ++c) augmented[r][c] = matrix[r][c];
    augmented[r][n] = rhs[r];
  }
  const std::vector<int> pivotColumns = gf2ReduceRows(augmented, n + 1);
  if (static_cast<int>(pivotColumns.size()) != n || pivotColumns.back() == n)
    throw std::runtime_error(
        "ChainComplex::stiefelWhitneyNumbers: the Poincaré-duality pairing is "
        "not invertible (the complex is not a closed manifold)");
  Gf2Vector x(n, 0);
  for (int t = 0; t < n; ++t) x[pivotColumns[t]] = augmented[t][n];
  return x;
}

bool isZeroVector(const Gf2Vector &v) {
  for (std::uint8_t e : v)
    if (e) return false;
  return true;
}

}  // namespace

std::map<std::string, int> ChainComplex::stiefelWhitneyNumbers() const {
  std::map<std::string, int> numbers;
  const int n = dimension_;
  if (n < 0) return numbers;  // empty complex: no characteristic numbers

  const auto countAt = [&](int k) {
    return (k < 0 || k > n) ? 0 : static_cast<int>(counts_[static_cast<std::size_t>(k)]);
  };

  // Mod-2 boundary entry ∂_k[row][col] (k-simplex col -> its (k-1)-faces).
  // The dense maps, fetched once: boundaryMatrix() takes the cache lock, which
  // has no place inside the bit loops below.
  std::vector<const std::vector<long> *> denseBoundary(static_cast<std::size_t>(n) + 1);
  for (int k = 0; k <= n; ++k) denseBoundary[static_cast<std::size_t>(k)] = &boundaryMatrix(k);
  const auto boundaryBit = [&](int k, int row, int col) -> std::uint8_t {
    const auto &flat = *denseBoundary[static_cast<std::size_t>(k)];
    const int cols = countAt(k);
    return static_cast<std::uint8_t>(
        std::abs(flat[static_cast<std::size_t>(row) * cols + col]) & 1);
  };

  // Index of a k-simplex from its sorted vertex ids (for cup-product faces).
  std::vector<std::map<Face, int>> indexOfFace(n + 1);
  for (int k = 0; k <= n; ++k)
    for (int i = 0; i < countAt(k); ++i)
      indexOfFace[k][faceVerts_[k][static_cast<std::size_t>(i)]] = i;

  // ---- mod-2 cohomology bases, one cocycle per class ----
  // H^k = ker(δ^k) / im(δ^{k-1}); δ^k = (∂_{k+1})^T, coboundaries = rows of ∂_k.
  std::vector<Gf2Matrix> cohomology(n + 1);  // cohomology[k] = basis cochains in C^k
  for (int k = 0; k <= n; ++k) {
    const int dim = countAt(k);
    // Coboundary operator δ^k as a matrix on length-`dim` cochains, with one
    // row per (k+1)-simplex: (δ^k α)(τ) = α(∂τ).
    Gf2Matrix coboundaryOperator;
    if (k + 1 <= n) {
      const int higher = countAt(k + 1);
      coboundaryOperator.assign(higher, Gf2Vector(dim, 0));
      for (int tau = 0; tau < higher; ++tau)
        for (int face = 0; face < dim; ++face)
          coboundaryOperator[tau][face] = boundaryBit(k + 1, face, tau);
    }
    const Gf2Matrix cocycles = gf2Kernel(std::move(coboundaryOperator), dim);

    // Span seeded with the coboundaries (rows of ∂_k); cocycles independent of
    // them are the cohomology generators.
    Gf2Span span(dim);
    for (int row = 0; row < countAt(k - 1); ++row) {
      Gf2Vector coboundary(dim, 0);
      for (int col = 0; col < dim; ++col) coboundary[col] = boundaryBit(k, row, col);
      span.add(std::move(coboundary));
    }
    for (const Gf2Vector &cocycle : cocycles)
      if (span.add(cocycle)) cohomology[k].push_back(cocycle);
  }

  // ---- Alexander–Whitney cup product on cochains (mod 2) ----
  // (α ∪ β) on a (p+q)-simplex [v0..v_{p+q}] = α(v0..vp) · β(vp..v_{p+q}).
  const auto cup = [&](const Gf2Vector &alpha, int p, const Gf2Vector &beta,
                       int q) -> Gf2Vector {
    const int degree = p + q;
    const int dim = countAt(degree);
    Gf2Vector product(dim, 0);
    if (degree > n) return product;
    for (int s = 0; s < dim; ++s) {
      const Face &vertices = faceVerts_[degree][static_cast<std::size_t>(s)];
      const Face front(vertices.begin(), vertices.begin() + (p + 1));
      const Face back(vertices.begin() + p, vertices.end());
      product[s] = static_cast<std::uint8_t>(
          alpha[indexOfFace[p].at(front)] & beta[indexOfFace[q].at(back)]);
    }
    return product;
  };

  // Evaluate a top-degree cochain on the fundamental class [K]: the mod-2 sum
  // over all top simplices.
  const auto evaluateOnFundamentalClass = [&](const Gf2Vector &topCochain) -> int {
    int total = 0;
    for (std::uint8_t e : topCochain) total ^= e;
    return total;
  };

  // ---- Wu classes v_k (1 ≤ k ≤ n/2; the rest vanish for degree reasons) ----
  // Sq^k on a degree-(n-k) class is the cup square when k = n-k, zero when
  // k > n-k, and a higher cup-i product when k < n-k.
  std::vector<Gf2Vector> wuClass(n + 1);
  for (int k = 0; k <= n; ++k) wuClass[k].assign(countAt(k), 0);
  for (int k = 1; 2 * k <= n; ++k) {
    const int complement = n - k;
    const auto &basisK = cohomology[k];
    const auto &basisComplement = cohomology[complement];
    if (basisK.empty()) continue;  // H^k = 0 ⇒ v_k = 0
    if (basisK.size() != basisComplement.size())
      throw std::runtime_error(
          "ChainComplex::stiefelWhitneyNumbers: mod-2 Poincaré duality fails "
          "(dim H^" + std::to_string(k) + " != dim H^" + std::to_string(complement) +
          "); the complex is not a closed manifold");

    // Pairing matrix P[j][i] = <e_i ∪ x_j, [K]> and right-hand side
    // r[j] = <Sq^k x_j, [K]>.
    Gf2Matrix pairing(basisComplement.size(), Gf2Vector(basisK.size(), 0));
    Gf2Vector rightHandSide(basisComplement.size(), 0);
    for (int j = 0; j < static_cast<int>(basisComplement.size()); ++j) {
      for (int i = 0; i < static_cast<int>(basisK.size()); ++i)
        pairing[j][i] = static_cast<std::uint8_t>(
            evaluateOnFundamentalClass(cup(basisK[i], k, basisComplement[j], complement)));
      if (k == complement)  // Sq^k on a degree-k class is the cup square
        rightHandSide[j] = static_cast<std::uint8_t>(evaluateOnFundamentalClass(
            cup(basisComplement[j], complement, basisComplement[j], complement)));
      // k < complement would need a higher cup-i product.
      else
        throw std::runtime_error(
            "ChainComplex::stiefelWhitneyNumbers: Wu class v_" + std::to_string(k) +
            " requires a higher Steenrod cup-i product (i>0), which is deferred "
            "");
    }
    const Gf2Vector coefficients = gf2Solve(pairing, rightHandSide);
    Gf2Vector v(countAt(k), 0);
    for (int i = 0; i < static_cast<int>(coefficients.size()); ++i)
      if (coefficients[i])
        for (int c = 0; c < countAt(k); ++c) v[c] ^= basisK[i][c];
    wuClass[k] = std::move(v);
  }

  // ---- Stiefel–Whitney classes w_i = Σ_j Sq^j(v_{i-j}) ----
  // Sq^0(v_i) = v_i; Sq^j(v_{i-j}) for j ≥ 1 is the cup square when 2j = i and
  // zero when the Wu class vanishes; anything else needs a higher square.
  // Evaluated lazily.
  std::vector<bool> haveW(n + 1, false);
  std::vector<Gf2Vector> wClass(n + 1);
  const auto stiefelWhitney = [&](int i) -> const Gf2Vector & {
    if (haveW[i]) return wClass[i];
    Gf2Vector w(countAt(i), 0);
    if (i <= n - i)  // Sq^0(v_i): v_i itself (zero unless 1 ≤ i ≤ n/2)
      for (int c = 0; c < countAt(i); ++c) w[c] ^= wuClass[i][c];
    for (int j = 1; j <= i; ++j) {
      const int m = i - j;  // degree of the Wu class being squared
      if (isZeroVector(wuClass[m])) continue;  // Sq^j(0) = 0
      if (j > m) continue;                      // Sq^j on degree m < j is 0
      if (j == m) {                             // Sq^j on degree j is the square
        const Gf2Vector square = cup(wuClass[m], m, wuClass[m], m);
        for (int c = 0; c < countAt(i); ++c) w[c] ^= square[c];
        continue;
      }
      throw std::runtime_error(
          "ChainComplex::stiefelWhitneyNumbers: Stiefel–Whitney class w_" +
          std::to_string(i) + " requires a higher Steenrod cup-i product (i>0), "
          "which is deferred");
    }
    haveW[i] = true;
    wClass[i] = std::move(w);
    return wClass[i];
  };

  // ---- Stiefel–Whitney numbers: every partition of n into positive parts ----
  // The monomial w_{i_1}···w_{i_r} cupped together and evaluated on [K]. Parts
  // ascend so a zero factor short-circuits before an unimplemented
  // higher-square factor is requested.
  std::function<void(int, int, std::vector<int> &)> forEachPartition =
      [&](int remaining, int minimumPart, std::vector<int> &parts) {
        if (remaining == 0) {
          // Build a readable monomial key, e.g. {1,1,2} -> "w1^2w2".
          std::string key;
          for (int p = 0; p < static_cast<int>(parts.size());) {
            int q = p;
            while (q < static_cast<int>(parts.size()) && parts[q] == parts[p]) ++q;
            const int multiplicity = q - p;
            key += "w" + std::to_string(parts[p]);
            if (multiplicity > 1) key += "^" + std::to_string(multiplicity);
            p = q;
          }
          // Evaluate the cup-product monomial, short-circuiting on a zero factor.
          Gf2Vector accumulator;
          int accumulatorDegree = 0;
          bool zero = false;
          for (int idx = 0; idx < static_cast<int>(parts.size()); ++idx) {
            const Gf2Vector &factor = stiefelWhitney(parts[idx]);
            if (idx == 0) {
              accumulator = factor;
              accumulatorDegree = parts[idx];
            } else {
              accumulator = cup(accumulator, accumulatorDegree, factor, parts[idx]);
              accumulatorDegree += parts[idx];
            }
            if (isZeroVector(accumulator)) { zero = true; break; }
          }
          numbers[key] = zero ? 0 : evaluateOnFundamentalClass(accumulator);
          return;
        }
        for (int part = minimumPart; part <= remaining; ++part) {
          parts.push_back(part);
          forEachPartition(remaining - part, part, parts);
          parts.pop_back();
        }
      };
  std::vector<int> parts;
  forEachPartition(n, 1, parts);
  return numbers;
}

std::pair<bool, std::string> ChainComplex::dualComplexIsValid(
    const std::vector<std::vector<std::uint64_t>> &topCells, int dim,
    const std::vector<std::vector<std::uint64_t>> &facetCells) {
  using Cell = std::vector<std::uint64_t>;
  const auto joinIds = [](const Cell &c) {
    std::string out = "(";
    for (std::size_t i = 0; i < c.size(); ++i) {
      if (i) out += ",";
      out += std::to_string(c[i]);
    }
    return out + ")";
  };
  const auto connectedFrom = [](int start, int count,
                                const std::map<int, std::vector<int>> &adj) {
    std::vector<int> stack{start};
    std::map<int, bool> seen{{start, true}};
    while (!stack.empty()) {
      const int x = stack.back();
      stack.pop_back();
      const auto it = adj.find(x);
      if (it == adj.end()) continue;
      for (const int y : it->second)
        if (!seen.count(y)) {
          seen[y] = true;
          stack.push_back(y);
        }
    }
    return static_cast<int>(seen.size()) == count;
  };

  if (topCells.empty()) return {false, "no top cells"};
  const std::size_t nv = static_cast<std::size_t>(dim) + 1;
  std::vector<Cell> cells;
  cells.reserve(topCells.size());
  for (const auto &raw : topCells) {
    if (raw.size() != nv) return {false, "mixed top-cell dimension"};
    Cell c = raw;
    std::sort(c.begin(), c.end());
    cells.push_back(std::move(c));
  }
  {
    auto dedup = cells;
    std::sort(dedup.begin(), dedup.end());
    if (std::adjacent_find(dedup.begin(), dedup.end()) != dedup.end())
      return {false, "duplicate top cell"};
  }

  // Facet coface counts in {1, 2}; the dangling-facet check against the
  // supplied (n-1)-cell universe.
  std::map<Cell, int> cofaces;
  for (const auto &c : cells)
    for (std::size_t j = 0; j < nv; ++j) {
      Cell f;
      f.reserve(nv - 1);
      for (std::size_t i = 0; i < nv; ++i)
        if (i != j) f.push_back(c[i]);
      ++cofaces[f];
    }
  for (const auto &[f, count] : cofaces)
    if (count > 2)
      return {false,
              "facet " + joinIds(f) + " has " + std::to_string(count) + " cofaces"};
  for (const auto &raw : facetCells) {
    Cell f = raw;
    std::sort(f.begin(), f.end());
    if (!cofaces.count(f))
      return {false, "dangling facet " + joinIds(f) + " (0 cofaces)"};
  }

  // Ridge links: the top cells around each (n-2)-simplex, glued along the
  // facets containing it, must form a single path or cycle (no pinches).
  std::map<Cell, std::vector<int>> atRidge;
  for (std::size_t ci = 0; ci < cells.size(); ++ci) {
    const Cell &c = cells[ci];
    for (std::size_t a = 0; a < nv; ++a)
      for (std::size_t b = a + 1; b < nv; ++b) {
        Cell r;
        r.reserve(nv - 2);
        for (std::size_t i = 0; i < nv; ++i)
          if (i != a && i != b) r.push_back(c[i]);
        atRidge[r].push_back(static_cast<int>(ci));
      }
  }
  for (const auto &[r, cis] : atRidge) {
    if (cis.size() < 2) continue;
    std::map<Cell, std::vector<int>> byFacet;
    for (const int ci : cis)
      for (const std::uint64_t v : cells[static_cast<std::size_t>(ci)])
        if (!std::binary_search(r.begin(), r.end(), v)) {
          Cell f = r;
          f.insert(std::upper_bound(f.begin(), f.end(), v), v);
          byFacet[f].push_back(ci);
        }
    std::map<int, std::vector<int>> adj;
    for (const auto &[f, fc] : byFacet)
      if (fc.size() == 2) {
        adj[fc[0]].push_back(fc[1]);
        adj[fc[1]].push_back(fc[0]);
      }
    if (!connectedFrom(cis.front(), static_cast<int>(cis.size()), adj))
      return {false, "ridge " + joinIds(r) + ": link is disconnected (pinch)"};
  }
  if (dim == 2) return {true, "ok"};

  // n >= 4: a PL manifold has every facet in <= 2 cofaces (checked above) and
  // every vertex link a valid (n-1)-manifold. Recurse on the links (link top
  // cells = each cell minus the vertex), bottoming out at the n == 3 check.
  if (dim >= 4) {
    std::map<std::uint64_t, std::vector<Cell>> linkTops;
    for (const auto &c : cells)
      for (const std::uint64_t v : c) {
        Cell lf;
        lf.reserve(nv - 1);
        for (const std::uint64_t u : c)
          if (u != v) lf.push_back(u);
        linkTops[v].push_back(std::move(lf));
      }
    for (const auto &[v, lt] : linkTops) {
      const auto verdict = dualComplexIsValid(lt, dim - 1, {});
      if (!verdict.first)
        return {false,
                "vertex " + std::to_string(v) + " link: " + verdict.second};
    }
    return {true, "ok"};
  }

  // n == 3: vertex links must be 2-spheres (interior) or disks (boundary).
  std::map<std::uint64_t, std::vector<Cell>> atVertex;
  for (const auto &c : cells)
    for (const std::uint64_t v : c) {
      Cell lf;
      lf.reserve(nv - 1);
      for (const std::uint64_t u : c)
        if (u != v) lf.push_back(u);
      atVertex[v].push_back(lf);
    }
  for (const auto &[v, linkFaces] : atVertex) {
    std::map<std::pair<std::uint64_t, std::uint64_t>, std::vector<int>> edgeFaces;
    for (std::size_t i = 0; i < linkFaces.size(); ++i) {
      const Cell &lf = linkFaces[i];
      for (std::size_t a = 0; a < lf.size(); ++a)
        for (std::size_t b = a + 1; b < lf.size(); ++b)
          edgeFaces[{lf[a], lf[b]}].push_back(static_cast<int>(i));
    }
    std::map<int, std::vector<int>> adj;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> boundaryEdges;
    for (const auto &[le, fs] : edgeFaces) {
      if (fs.size() == 2) {
        adj[fs[0]].push_back(fs[1]);
        adj[fs[1]].push_back(fs[0]);
      } else if (fs.size() == 1) {
        boundaryEdges.push_back(le);
      } else {
        return {false, "vertex " + std::to_string(v) + ": link edge in " +
                           std::to_string(fs.size()) + " faces"};
      }
    }
    if (!connectedFrom(0, static_cast<int>(linkFaces.size()), adj))
      return {false,
              "vertex " + std::to_string(v) + ": link is disconnected (pinch)"};
    std::map<std::uint64_t, int> linkVertexDegree;
    for (const auto &lf : linkFaces)
      for (const std::uint64_t u : lf) linkVertexDegree[u] = 0;
    const int chi = static_cast<int>(linkVertexDegree.size()) -
                    static_cast<int>(edgeFaces.size()) +
                    static_cast<int>(linkFaces.size());
    if (boundaryEdges.empty()) {
      if (chi != 2)
        return {false, "vertex " + std::to_string(v) + ": closed link has chi=" +
                           std::to_string(chi) + ", not S^2"};
    } else {
      if (chi != 1)
        return {false, "vertex " + std::to_string(v) + ": bounded link has chi=" +
                           std::to_string(chi) + ", not a disk"};
      std::map<std::uint64_t, std::vector<std::uint64_t>> badj;
      for (const auto &[a, b] : boundaryEdges) {
        badj[a].push_back(b);
        badj[b].push_back(a);
      }
      for (const auto &[u, nbrs] : badj)
        if (nbrs.size() != 2)
          return {false, "vertex " + std::to_string(v) +
                             ": link boundary is not a 1-manifold"};
      std::vector<std::uint64_t> stack{boundaryEdges.front().first};
      std::map<std::uint64_t, bool> seen{{boundaryEdges.front().first, true}};
      while (!stack.empty()) {
        const std::uint64_t x = stack.back();
        stack.pop_back();
        for (const std::uint64_t y : badj[x])
          if (!seen.count(y)) {
            seen[y] = true;
            stack.push_back(y);
          }
      }
      if (seen.size() != badj.size())
        return {false, "vertex " + std::to_string(v) +
                           ": link boundary has several circles"};
    }
  }
  return {true, "ok"};
}

namespace {

using OrientCell = std::vector<std::uint64_t>;

/// Parenthesised vertex-id tuple, for error messages.
std::string joinCellIds(const OrientCell &c) {
  std::string out = "(";
  for (std::size_t i = 0; i < c.size(); ++i) {
    if (i) out += ",";
    out += std::to_string(c[i]);
  }
  return out + ")";
}

/// Sorted-unique cells of uniform size `nv`, in lexicographic order.
///
/// The order makes the component roots of the propagation below deterministic
/// and independent of the order the caller supplies cells in.
void collectOrientCells(const std::vector<std::vector<std::uint64_t>> &raws,
                        std::size_t nv, const char *context,
                        std::set<OrientCell> &uniq) {
  for (const auto &raw : raws) {
    if (raw.size() != nv)
      throw std::runtime_error(
          std::string("ChainComplex::") + context + ": cell " +
          joinCellIds(raw) + " has " + std::to_string(raw.size()) +
          " vertices, expected " + std::to_string(nv) +
          " (one dimension throughout)");
    OrientCell c = raw;
    std::sort(c.begin(), c.end());
    uniq.insert(std::move(c));
  }
}

/// The orientation covector of a pseudomanifold, one entry per cell.
///
/// Facet j of a sorted cell drops vertex j and carries the boundary sign
/// (-1)^j. Across an interior facet the two induced signs must cancel, so
/// eps_b = -eps_a * s_a * s_b; a boundary facet imposes nothing. Each connected
/// component is rooted at its lex-smallest cell with eps = +1.
///
/// Throws if a facet has more than two cofaces (not a pseudomanifold), or if
/// propagation reaches a cell twice with opposite signs (not orientable).
std::vector<int> propagateOrientation(const std::vector<OrientCell> &cells,
                                      std::size_t nv, const char *context,
                                      const char *nonOrientableNote) {
  std::map<OrientCell, std::vector<std::pair<std::size_t, int>>> cofaces;
  for (std::size_t ci = 0; ci < cells.size(); ++ci)
    for (std::size_t j = 0; j < nv; ++j) {
      OrientCell f;
      f.reserve(nv - 1);
      for (std::size_t i = 0; i < nv; ++i)
        if (i != j) f.push_back(cells[ci][i]);
      cofaces[f].emplace_back(ci, (j % 2 == 0) ? 1 : -1);
    }
  for (const auto &[f, at] : cofaces)
    if (at.size() > 2)
      throw std::runtime_error(
          std::string("ChainComplex::") + context + ": facet " +
          joinCellIds(f) + " has " + std::to_string(at.size()) +
          " cofaces (not a pseudomanifold)");

  std::vector<int> eps(cells.size(), 0);
  for (std::size_t root = 0; root < cells.size(); ++root) {
    if (eps[root] != 0) continue;
    eps[root] = 1;
    std::vector<std::size_t> stack{root};
    while (!stack.empty()) {
      const std::size_t a = stack.back();
      stack.pop_back();
      for (std::size_t j = 0; j < nv; ++j) {
        OrientCell f;
        f.reserve(nv - 1);
        for (std::size_t i = 0; i < nv; ++i)
          if (i != j) f.push_back(cells[a][i]);
        const int sa = (j % 2 == 0) ? 1 : -1;
        for (const auto &[b, sb] : cofaces.at(f)) {
          if (b == a) continue;
          const int want = -eps[a] * sa * sb;
          if (eps[b] == 0) {
            eps[b] = want;
            stack.push_back(b);
          } else if (eps[b] != want) {
            throw std::runtime_error(
                std::string("ChainComplex::") + context +
                ": orientation propagation contradicts itself at facet " +
                joinCellIds(f) + " (" + nonOrientableNote + ")");
          }
        }
      }
    }
  }
  return eps;
}

} // namespace

std::vector<int> ChainComplex::endSignCovector(
    const std::vector<std::vector<std::uint64_t>> &surfaceCells,
    const std::vector<std::vector<std::uint64_t>> &holes) {
  if (holes.empty()) return {};

  // The oriented complex is the union surface u holes.
  std::set<OrientCell> uniq;
  const std::size_t nv = holes.front().size();
  collectOrientCells(holes, nv, "endSignCovector", uniq);
  collectOrientCells(surfaceCells, nv, "endSignCovector", uniq);
  const std::vector<OrientCell> cells(uniq.begin(), uniq.end());

  const std::vector<int> eps = propagateOrientation(
      cells, nv, "endSignCovector", "the end surface is non-orientable");

  // Project back onto the hole cells, in the order the caller gave them.
  std::vector<int> sigma;
  sigma.reserve(holes.size());
  for (const auto &raw : holes) {
    OrientCell h = raw;
    std::sort(h.begin(), h.end());
    const auto it = std::lower_bound(cells.begin(), cells.end(), h);
    sigma.push_back(eps[static_cast<std::size_t>(it - cells.begin())]);
  }
  return sigma;
}

std::vector<int> ChainComplex::orientationCovector(
    const std::vector<std::vector<std::uint64_t>> &topCells) {
  if (topCells.empty()) return {};

  // Sorted-unique cells: the canonical C_d column order, so the covector aligns
  // with orientedTopSimplices() and ignores the order topCells arrives in.
  std::set<OrientCell> uniq;
  const std::size_t nv = topCells.front().size();
  collectOrientCells(topCells, nv, "orientationCovector", uniq);
  const std::vector<OrientCell> cells(uniq.begin(), uniq.end());

  return propagateOrientation(cells, nv, "orientationCovector",
                              "the complex is non-orientable");
}

}  // namespace tessera::cobordism
