// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "cobordism/SpacetimeComposition.h"

#include <cmath>
#include <stdexcept>
#include <string>

#include "spacetime/Spacetime.h"

namespace tessera::cobordism {

namespace {

using cd = std::complex<double>;

void requireSquare(const std::vector<cd> &matrix, int dim, const char *name) {
  if (dim < 0 || matrix.size() != static_cast<std::size_t>(dim) *
                                      static_cast<std::size_t>(dim))
    throw std::invalid_argument(std::string(name) +
                                ": flat size does not match dimension");
}

/// L_k of one factor, with the dimension the caller needs to index it.
std::pair<std::vector<cd>, int> laplacianOf(
    const std::shared_ptr<Spacetime> &factor, int degree, bool metric,
    const char *name) {
  if (factor == nullptr)
    throw std::invalid_argument(std::string(name) + " is null");
  if (degree < 0)
    throw std::invalid_argument(std::string(name) +
                                ": degree must not be negative");
  const HodgeLaplacian hodge(factor);
  std::vector<cd> flat = hodge.laplacian(degree, metric);
  // laplacian() returns an empty matrix above the top dimension, where the
  // factor has no k-cells; the composition of an empty operator is empty.
  const auto dim = static_cast<int>(
      flat.empty() ? 0 : static_cast<int>(std::llround(std::sqrt(
                             static_cast<double>(flat.size())))));
  return {std::move(flat), dim};
}

} // namespace

// ---- matrix level -----------------------------------------------------------

std::vector<cd> SpacetimeComposition::kroneckerSum(
    const std::vector<cd> &operatorA, int dimA,
    const std::vector<cd> &operatorB, int dimB) {
  requireSquare(operatorA, dimA, "kroneckerSum: operatorA");
  requireSquare(operatorB, dimB, "kroneckerSum: operatorB");
  const std::size_t dim =
      static_cast<std::size_t>(dimA) * static_cast<std::size_t>(dimB);
  std::vector<cd> result(dim * dim, cd{0.0, 0.0});
  // A ⊗ I: block (iA, jA) is operatorA[iA,jA] * I_{dimB}.
  for (int iA = 0; iA < dimA; ++iA)
    for (int jA = 0; jA < dimA; ++jA) {
      const cd value = operatorA[static_cast<std::size_t>(iA) * dimA + jA];
      if (value == cd{0.0, 0.0}) continue;
      for (int iB = 0; iB < dimB; ++iB) {
        const std::size_t row = static_cast<std::size_t>(iA) * dimB + iB;
        const std::size_t col = static_cast<std::size_t>(jA) * dimB + iB;
        result[row * dim + col] += value;
      }
    }
  // I ⊗ B: block (iA, iA) receives operatorB.
  for (int iA = 0; iA < dimA; ++iA)
    for (int iB = 0; iB < dimB; ++iB)
      for (int jB = 0; jB < dimB; ++jB) {
        const cd value = operatorB[static_cast<std::size_t>(iB) * dimB + jB];
        if (value == cd{0.0, 0.0}) continue;
        const std::size_t row = static_cast<std::size_t>(iA) * dimB + iB;
        const std::size_t col = static_cast<std::size_t>(iA) * dimB + jB;
        result[row * dim + col] += value;
      }
  return result;
}

std::vector<cd> SpacetimeComposition::kroneckerProduct(
    const std::vector<cd> &operatorA, int dimA,
    const std::vector<cd> &operatorB, int dimB) {
  requireSquare(operatorA, dimA, "kroneckerProduct: operatorA");
  requireSquare(operatorB, dimB, "kroneckerProduct: operatorB");
  const std::size_t dim =
      static_cast<std::size_t>(dimA) * static_cast<std::size_t>(dimB);
  std::vector<cd> result(dim * dim, cd{0.0, 0.0});
  // Entry ((iA,iB),(jA,jB)) is A[iA,jA] * B[iB,jB]. A zero in A kills its whole
  // block, so skipping there saves the inner pair outright.
  for (int iA = 0; iA < dimA; ++iA)
    for (int jA = 0; jA < dimA; ++jA) {
      const cd a = operatorA[static_cast<std::size_t>(iA) * dimA + jA];
      if (a == cd{0.0, 0.0}) continue;
      for (int iB = 0; iB < dimB; ++iB)
        for (int jB = 0; jB < dimB; ++jB) {
          const cd b = operatorB[static_cast<std::size_t>(iB) * dimB + jB];
          if (b == cd{0.0, 0.0}) continue;
          const std::size_t row = static_cast<std::size_t>(iA) * dimB + iB;
          const std::size_t col = static_cast<std::size_t>(jA) * dimB + jB;
          result[row * dim + col] = a * b;
        }
    }
  return result;
}

std::vector<cd> SpacetimeComposition::directSum(
    const std::vector<cd> &operatorA, int dimA,
    const std::vector<cd> &operatorB, int dimB) {
  requireSquare(operatorA, dimA, "directSum: operatorA");
  requireSquare(operatorB, dimB, "directSum: operatorB");
  const std::size_t dim =
      static_cast<std::size_t>(dimA) + static_cast<std::size_t>(dimB);
  std::vector<cd> result(dim * dim, cd{0.0, 0.0});
  for (int i = 0; i < dimA; ++i)
    for (int j = 0; j < dimA; ++j)
      result[static_cast<std::size_t>(i) * dim + j] =
          operatorA[static_cast<std::size_t>(i) * dimA + j];
  // B occupies the trailing block, offset by dimA in both directions.
  for (int i = 0; i < dimB; ++i)
    for (int j = 0; j < dimB; ++j)
      result[(static_cast<std::size_t>(i) + dimA) * dim +
             (static_cast<std::size_t>(j) + dimA)] =
          operatorB[static_cast<std::size_t>(i) * dimB + j];
  return result;
}

// ---- space level ------------------------------------------------------------

std::vector<cd> SpacetimeComposition::cartesianProduct(
    const std::shared_ptr<Spacetime> &factorA,
    const std::shared_ptr<Spacetime> &factorB, int degree, bool metric) {
  const auto [a, dimA] = laplacianOf(factorA, degree, metric,
                                     "cartesianProduct: factorA");
  const auto [b, dimB] = laplacianOf(factorB, degree, metric,
                                     "cartesianProduct: factorB");
  return kroneckerSum(a, dimA, b, dimB);
}

std::vector<cd> SpacetimeComposition::tensorProduct(
    const std::shared_ptr<Spacetime> &factorA,
    const std::shared_ptr<Spacetime> &factorB, int degree, bool metric) {
  const auto [a, dimA] = laplacianOf(factorA, degree, metric,
                                     "tensorProduct: factorA");
  const auto [b, dimB] = laplacianOf(factorB, degree, metric,
                                     "tensorProduct: factorB");
  return kroneckerProduct(a, dimA, b, dimB);
}

std::vector<cd> SpacetimeComposition::directSum(
    const std::shared_ptr<Spacetime> &factorA,
    const std::shared_ptr<Spacetime> &factorB, int degree, bool metric) {
  const auto [a, dimA] = laplacianOf(factorA, degree, metric,
                                     "directSum: factorA");
  const auto [b, dimB] = laplacianOf(factorB, degree, metric,
                                     "directSum: factorB");
  return directSum(a, dimA, b, dimB);
}

}  // namespace tessera::cobordism
