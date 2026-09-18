// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "cobordism/OccupationSpectra.h"

#include "cobordism/SpacetimeComposition.h"

#include <algorithm>
#include <stdexcept>

namespace tessera::cobordism {

namespace {

using cd = std::complex<double>;

bool ascendingReIm(const cd &x, const cd &y) {
  if (x.real() != y.real())
    return x.real() < y.real();
  return x.imag() < y.imag();
}

// C(n, k) saturating at limit+1 so callers can compare against a cap without
// overflow.
std::size_t binomialCapped(std::size_t n, std::size_t k, std::size_t limit) {
  if (k > n)
    return 0;
  k = std::min(k, n - k);
  std::size_t result = 1;
  for (std::size_t i = 1; i <= k; ++i) {
    // result *= (n - k + i) / i, exactly (numerator divisible stepwise).
    const std::size_t numerator = n - k + i;
    if (result > (limit + 1) / numerator * i)
      return limit + 1;
    result = result * numerator / i;
    if (result > limit)
      return limit + 1;
  }
  return result;
}

void requireSquare(const std::vector<cd> &matrix, int dim, const char *name) {
  if (dim < 0 || matrix.size() != static_cast<std::size_t>(dim) *
                                      static_cast<std::size_t>(dim))
    throw std::invalid_argument(std::string(name) +
                                ": flat size does not match dimension");
}

void sortSpectrum(std::vector<cd> &values) {
  std::sort(values.begin(), values.end(), ascendingReIm);
}

} // namespace

std::vector<cd> OccupationSpectra::subsetSums(const std::vector<cd> &oneParticle,
                                              int particles,
                                              std::size_t maxTerms) {
  if (particles < 0)
    throw std::invalid_argument("subsetSums: negative particle number");
  const std::size_t n = oneParticle.size();
  const auto want = static_cast<std::size_t>(particles);
  if (want > n)
    return {}; // Pauli exclusion: no N-particle sector beyond n modes.
  const std::size_t count = binomialCapped(n, want, maxTerms);
  if (count > maxTerms)
    throw std::length_error(
        "subsetSums: output exceeds maxTerms; unmaterializable");

  std::vector<cd> sums;
  sums.reserve(count);
  if (want == 0) {
    sums.push_back(cd{0.0, 0.0}); // the vacuum
    sortSpectrum(sums);
    return sums;
  }
  // Lexicographic enumeration of index combinations; each sum is
  // accumulated fresh (no cancellation drift across subsets).
  std::vector<std::size_t> combo(want);
  for (std::size_t i = 0; i < want; ++i)
    combo[i] = i;
  while (true) {
    cd sum{0.0, 0.0};
    for (const std::size_t index : combo)
      sum += oneParticle[index];
    sums.push_back(sum);
    // Advance to the next combination.
    std::size_t position = want;
    while (position > 0 && combo[position - 1] == n - want + (position - 1))
      --position;
    if (position == 0)
      break;
    ++combo[position - 1];
    for (std::size_t i = position; i < want; ++i)
      combo[i] = combo[i - 1] + 1;
  }
  sortSpectrum(sums);
  return sums;
}

std::vector<cd> OccupationSpectra::fockSums(const std::vector<cd> &oneParticle,
                                            std::size_t maxTerms) {
  const std::size_t n = oneParticle.size();
  if (n >= 63 || (std::size_t{1} << n) > maxTerms)
    throw std::length_error(
        "fockSums: 2^n output exceeds maxTerms; unmaterializable");
  std::vector<cd> sums;
  sums.reserve(std::size_t{1} << n);
  sums.push_back(cd{0.0, 0.0}); // vacuum
  // Incremental doubling: sums over subsets of the first m modes.
  for (std::size_t mode = 0; mode < n; ++mode) {
    const std::size_t existing = sums.size();
    for (std::size_t i = 0; i < existing; ++i)
      sums.push_back(sums[i] + oneParticle[mode]);
  }
  sortSpectrum(sums);
  return sums;
}

std::vector<cd> OccupationSpectra::directSumSubsetSums(
    const std::vector<cd> &factorA, const std::vector<cd> &factorB,
    int particles, std::size_t maxTerms) {
  if (particles < 0)
    throw std::invalid_argument("directSumSubsetSums: negative particle number");
  const auto want = static_cast<std::size_t>(particles);
  if (want > factorA.size() + factorB.size())
    return {};
  // Cap check on the total output before enumerating any split.
  const std::size_t total =
      binomialCapped(factorA.size() + factorB.size(), want, maxTerms);
  if (total > maxTerms)
    throw std::length_error(
        "directSumSubsetSums: output exceeds maxTerms; unmaterializable");

  std::vector<cd> merged;
  merged.reserve(total);
  for (std::size_t inA = 0; inA <= want; ++inA) {
    const std::size_t inB = want - inA;
    if (inA > factorA.size() || inB > factorB.size())
      continue;
    const std::vector<cd> sumsA =
        subsetSums(factorA, static_cast<int>(inA), maxTerms);
    const std::vector<cd> sumsB =
        subsetSums(factorB, static_cast<int>(inB), maxTerms);
    for (const cd &a : sumsA)
      for (const cd &b : sumsB)
        merged.push_back(a + b);
  }
  sortSpectrum(merged);
  return merged;
}

std::vector<cd> OccupationSpectra::directSum(const std::vector<cd> &blockA,
                                             int dimA,
                                             const std::vector<cd> &blockB,
                                             int dimB) {
  // One block-diagonal assembly in the subsystem. hoppingBlock remains for the
  // coupled case; this was only its zero-coupling specialization, and that is
  // the same matrix SpacetimeComposition builds.
  return SpacetimeComposition::directSum(blockA, dimA, blockB, dimB);
}

std::vector<cd> OccupationSpectra::hoppingBlock(
    const std::vector<cd> &blockA, int dimA, const std::vector<cd> &blockB,
    int dimB, const std::vector<cd> &coupling,
    const std::vector<cd> &couplingReverse) {
  requireSquare(blockA, dimA, "hoppingBlock: blockA");
  requireSquare(blockB, dimB, "hoppingBlock: blockB");
  const auto nA = static_cast<std::size_t>(dimA);
  const auto nB = static_cast<std::size_t>(dimB);
  if (coupling.size() != nA * nB)
    throw std::invalid_argument(
        "hoppingBlock: coupling must be dimA x dimB flat row-major");
  const bool hermitianReverse = couplingReverse.empty();
  if (!hermitianReverse && couplingReverse.size() != nB * nA)
    throw std::invalid_argument(
        "hoppingBlock: couplingReverse must be dimB x dimA flat row-major");

  const std::size_t dim = nA + nB;
  std::vector<cd> result(dim * dim, cd{0.0, 0.0});
  for (std::size_t i = 0; i < nA; ++i)
    for (std::size_t j = 0; j < nA; ++j)
      result[i * dim + j] = blockA[i * nA + j];
  for (std::size_t i = 0; i < nB; ++i)
    for (std::size_t j = 0; j < nB; ++j)
      result[(nA + i) * dim + (nA + j)] = blockB[i * nB + j];
  for (std::size_t i = 0; i < nA; ++i)
    for (std::size_t j = 0; j < nB; ++j)
      result[i * dim + (nA + j)] = coupling[i * nB + j];
  for (std::size_t i = 0; i < nB; ++i)
    for (std::size_t j = 0; j < nA; ++j)
      result[(nA + i) * dim + j] = hermitianReverse
                                       ? std::conj(coupling[j * nB + i])
                                       : couplingReverse[i * nA + j];
  return result;
}

} // namespace tessera::cobordism
