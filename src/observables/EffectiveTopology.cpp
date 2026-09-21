// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "observables/EffectiveTopology.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace tessera::observables {

using chainhodge::Complex;
using chainhodge::CovariantChainHodge;
using chainhodge::EffectiveBettiRead;
using chainhodge::SparsePencil;
using chainhodge::SparsePencilOptions;
using chainhodge::SparsePencilSolver;
using chainhodge::SpectrumRead;

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kInfinity = std::numeric_limits<double>::infinity();

double gapOf(int rank, double lastInside, double firstOutside) {
  if (std::isnan(firstOutside)) return kNaN;
  return (rank > 0 && lastInside > 0.0) ? firstOutside / lastInside : kInfinity;
}

EffectiveBettiNumber fromDenseSpectrum(const CovariantChainHodge &cov, int k, double epsilon,
                                       double tolerance) {
  EffectiveBettiNumber out;
  out.degree = k;
  out.method = EffectiveBettiNumber::Method::DenseSpectrum;
  const SpectrumRead spectrum = cov.spectrum(k);
  std::vector<double> moduli;
  moduli.reserve(spectrum.eigenvalues.size());
  for (const Complex &lambda : spectrum.eigenvalues) moduli.push_back(std::abs(lambda));
  std::sort(moduli.begin(), moduli.end());
  out.rank = static_cast<int>(std::upper_bound(moduli.begin(), moduli.end(), epsilon) - moduli.begin());
  if (out.rank > 0) out.lastInside = moduli[static_cast<std::size_t>(out.rank) - 1];
  if (out.rank < static_cast<int>(moduli.size())) out.firstOutside = moduli[static_cast<std::size_t>(out.rank)];
  // A complex whose degree has no cells has an empty band and nothing outside it.
  out.gap = moduli.empty() ? kInfinity : gapOf(out.rank, out.lastInside, out.firstOutside);
  out.certified = spectrum.residual <= tolerance;
  return out;
}

EffectiveBettiNumber fromSparsePencil(const CovariantChainHodge &cov, double epsilon, double tolerance) {
  EffectiveBettiNumber out;
  out.degree = 0;
  try {
    const SparsePencil pencil = cov.sparsePencil(0);
    SparsePencilOptions options;
    options.tolerance = tolerance;
    // The shift at -epsilon: its Cholesky factorization certifies that no
    // eigenvalue lies below the window, so the count above it is |lambda| <= epsilon.
    const EffectiveBettiRead read =
        SparsePencilSolver::effectiveBetti(pencil.A, pencil.M, epsilon, -epsilon, options);
    out.method = EffectiveBettiNumber::Method::SparsePencil;
    out.rank = read.rank;
    out.lastInside = std::abs(read.lastInside);
    out.firstOutside = read.firstOutside;
    out.gap = gapOf(out.rank, out.lastInside, out.firstOutside);
    out.certified = read.certified;
  } catch (const std::logic_error &refusal) {
    // std::invalid_argument (a pencil that is not Hermitian positive definite)
    // and the Grassmann preset's refusal both derive from std::logic_error.
    out.method = EffectiveBettiNumber::Method::Unmeasured;
    out.reason = refusal.what();
  }
  return out;
}

std::vector<int> binomials(int d) {
  std::vector<int> row(static_cast<std::size_t>(d) + 1, 1);
  for (int k = 1; k < d; ++k)
    row[static_cast<std::size_t>(k)] = row[static_cast<std::size_t>(k) - 1] * (d - k + 1) / k;
  return row;
}

}  // namespace

EffectiveTopology EffectiveTopology::read(const CovariantChainHodge &cov, double epsilon, double tolerance) {
  if (!(epsilon > 0.0)) throw std::invalid_argument("EffectiveTopology: the scale epsilon must be positive");
  EffectiveTopology out;
  out.epsilon_ = epsilon;
  const int d = cov.dimension();
  const int crossover = cov.base().crossoverDimension();
  for (int k = 0; k <= d; ++k) {
    if (cov.base().size(k) < crossover) {
      out.degrees_.push_back(fromDenseSpectrum(cov, k, epsilon, tolerance));
    } else if (k == 0) {
      out.degrees_.push_back(fromSparsePencil(cov, epsilon, tolerance));
    } else {
      EffectiveBettiNumber unmeasured;
      unmeasured.degree = k;
      unmeasured.reason = "degree " + std::to_string(k) +
                          " is at or above the dense crossover and its pencil contains an inverse "
                          "metric, so it has no sparse form";
      out.degrees_.push_back(std::move(unmeasured));
    }
  }
  return out;
}

std::vector<int> EffectiveTopology::betti() const {
  std::vector<int> out;
  out.reserve(degrees_.size());
  for (const auto &degree : degrees_) out.push_back(degree.rank);
  return out;
}

bool EffectiveTopology::certified() const noexcept {
  return std::all_of(degrees_.begin(), degrees_.end(), [](const EffectiveBettiNumber &degree) {
    return degree.method != EffectiveBettiNumber::Method::Unmeasured && degree.certified;
  });
}

EffectiveSignatureCertificate EffectiveSignature::certify(const EffectiveTopology &topology) const {
  EffectiveSignatureCertificate out;
  out.signature = name();
  out.epsilon = topology.epsilon();
  out.expected = betti();
  out.measured = topology.betti();
  if (out.expected.size() != out.measured.size())
    throw std::invalid_argument("EffectiveSignature: the " + name() + " has dimension " +
                                std::to_string(static_cast<int>(out.expected.size()) - 1) +
                                " and the read has dimension " + std::to_string(topology.dimension()));
  out.matches = true;
  out.certified = true;
  out.gap = kInfinity;
  for (std::size_t k = 0; k < out.expected.size(); ++k) {
    if (out.expected[k] < 0) continue;
    const EffectiveBettiNumber &degree = topology.degrees()[k];
    out.matches = out.matches && degree.rank == out.expected[k];
    out.certified = out.certified && degree.certified;
    // NaN (nothing outside the window) compares false and leaves the minimum alone.
    if (degree.gap < out.gap) out.gap = degree.gap;
  }
  return out;
}

EffectiveSignatureCertificate EffectiveSignature::certify(const CovariantChainHodge &cov, double epsilon,
                                                          double tolerance) const {
  return certify(EffectiveTopology::read(cov, epsilon, tolerance));
}

EffectiveTorus::EffectiveTorus(int dimension) : dimension_(dimension) {
  if (dimension < 1) throw std::invalid_argument("EffectiveTorus: the dimension must be at least 1");
}
std::string EffectiveTorus::name() const { return "effective " + std::to_string(dimension_) + "-torus"; }
std::vector<int> EffectiveTorus::betti() const { return binomials(dimension_); }

EffectiveSphere::EffectiveSphere(int dimension) : dimension_(dimension) {
  if (dimension < 1) throw std::invalid_argument("EffectiveSphere: the dimension must be at least 1");
}
std::string EffectiveSphere::name() const { return "effective " + std::to_string(dimension_) + "-sphere"; }
std::vector<int> EffectiveSphere::betti() const {
  std::vector<int> out(static_cast<std::size_t>(dimension_) + 1, 0);
  out.front() = 1;
  out.back() = 1;
  return out;
}

EffectiveComponents::EffectiveComponents(int count, int dimension) : count_(count), dimension_(dimension) {
  if (count < 0 || dimension < 0)
    throw std::invalid_argument("EffectiveComponents: the count and the dimension must be non-negative");
}
std::string EffectiveComponents::name() const {
  return std::to_string(count_) + " effective component" + (count_ == 1 ? "" : "s");
}
std::vector<int> EffectiveComponents::betti() const {
  std::vector<int> out(static_cast<std::size_t>(dimension_) + 1, -1);
  out.front() = count_;
  return out;
}

}  // namespace tessera::observables
