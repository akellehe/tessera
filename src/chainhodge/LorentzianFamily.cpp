// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "chainhodge/LorentzianFamily.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <Eigen/Dense>

namespace tessera::chainhodge {

SquaredLengths LorentzianFamily::rotate(const SquaredLengths &s,
                                        const SquaredLengths &timelikeParts, double epsilon) {
  if (timelikeParts.size() != s.size())
    throw std::invalid_argument("LorentzianFamily::rotate: one timelike part per edge is required (" +
                                std::to_string(s.size()) + " edges, " +
                                std::to_string(timelikeParts.size()) + " timelike parts)");
  if (!std::isfinite(epsilon) || epsilon < 0.0)
    throw std::invalid_argument("LorentzianFamily::rotate: the rotation must be a finite epsilon >= 0 "
                                "(got " + std::to_string(epsilon) + ")");
  // s_e(eps) = (s_e - tau_e) + e^{-2 i eps} tau_e, written as s_e + (e^{-2 i eps} - 1) tau_e so
  // that eps = 0 returns s exactly.
  const Complex shift = std::exp(Complex(0.0, -2.0 * epsilon)) - Complex(1.0, 0.0);
  SquaredLengths out(s);
  for (std::size_t e = 0; e < s.size(); ++e) {
    const Complex tau = timelikeParts[e];
    if (!std::isfinite(tau.real()) || !std::isfinite(tau.imag()))
      throw std::invalid_argument("LorentzianFamily::rotate: the timelike part of edge " +
                                  std::to_string(e) + " is not finite");
    out[e] = s[e] + shift * tau;
  }
  return out;
}

ChainHodge LorentzianFamily::instance(const cobordism::ChainComplex &K, const SquaredLengths &s,
                                      const SquaredLengths &timelikeParts, double epsilon,
                                      Preset preset, Branch branch, int crossoverDimension) {
  return ChainHodge(K, rotate(s, timelikeParts, epsilon), preset, branch, crossoverDimension,
                    epsilon);
}

std::vector<LorentzianRead> LorentzianFamily::sweep(const cobordism::ChainComplex &K,
                                                    const SquaredLengths &s,
                                                    const SquaredLengths &timelikeParts,
                                                    const std::vector<double> &epsilons,
                                                    int degree, Preset preset, Branch branch,
                                                    double kappa, bool withSpectrum,
                                                    int crossoverDimension) {
  // Requirement 2: a family is read at one or more epsilon > 0, and a read at epsilon = 0 is
  // reported only beside them, never alone.
  bool positive = false;
  for (const double eps : epsilons) {
    if (!std::isfinite(eps) || eps < 0.0)
      throw std::invalid_argument("LorentzianFamily::sweep: every rotation must be a finite "
                                  "epsilon >= 0 (got " + std::to_string(eps) + ")");
    positive = positive || eps > 0.0;
  }
  if (!positive)
    throw std::invalid_argument("LorentzianFamily::sweep: the family needs at least one epsilon > 0; "
                                "a read at epsilon = 0 is never reported alone");
  std::vector<LorentzianRead> out;
  out.reserve(epsilons.size());
  for (const double eps : epsilons) {
    const ChainHodge hodge =
        instance(K, s, timelikeParts, eps, preset, branch, crossoverDimension);
    LorentzianRead read;
    read.epsilon = eps;
    read.allowable = hodge.certificate().allowable;
    read.margin = hodge.certificate().margin;
    read.degree = degree;
    read.harmonic = hodge.harmonicChains(degree, kappa);
    if (withSpectrum && hodge.size(degree) < crossoverDimension)
      read.eigenvalues = hodge.spectrum(degree).eigenvalues;
    out.push_back(std::move(read));
  }
  return out;
}

LorentzianExtrapolation LorentzianFamily::extrapolateToZero(const std::vector<double> &epsilons,
                                                            const std::vector<Complex> &values,
                                                            int order) {
  const int n = static_cast<int>(epsilons.size());
  if (n < 2 || values.size() != epsilons.size())
    throw std::invalid_argument("LorentzianFamily::extrapolateToZero: at least two reads with one "
                                "value each are required");
  for (const double e : epsilons)
    if (!(e > 0.0))
      throw std::invalid_argument("LorentzianFamily::extrapolateToZero: every read must be at "
                                  "epsilon > 0; an instance value at zero is not an extrapolation input");
  if (order < 0) throw std::invalid_argument("LorentzianFamily::extrapolateToZero: order must be >= 0");
  const int deg = std::min(order, n - 1);
  Eigen::MatrixXcd V(n, deg + 1);
  Eigen::VectorXcd y(n);
  for (int i = 0; i < n; ++i) {
    double p = 1.0;
    for (int j = 0; j <= deg; ++j) {
      V(i, j) = p;
      p *= epsilons[static_cast<std::size_t>(i)];
    }
    y(i) = values[static_cast<std::size_t>(i)];
  }
  const Eigen::VectorXcd coef = V.colPivHouseholderQr().solve(y);
  LorentzianExtrapolation ex;
  ex.epsilons = epsilons;
  ex.values = values;
  ex.order = deg;
  ex.extrapolated = coef(0);
  ex.residual = std::sqrt((V * coef - y).squaredNorm() / static_cast<double>(n));
  return ex;
}

}  // namespace tessera::chainhodge
