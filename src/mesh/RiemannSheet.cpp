// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "mesh/RiemannSheet.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>

namespace tessera::mesh {
namespace {

constexpr double kTwoPi = 2.0 * std::numbers::pi;

/// Fold an angle difference into \f$ (-\pi, \pi] \f$: the turn a single step of
/// a path makes, on the assumption that it makes less than half a turn.
inline double wrapToPi(double a) noexcept {
  a = std::fmod(a + std::numbers::pi, kTwoPi);
  if (a <= 0.0) a += kTwoPi;
  return a - std::numbers::pi;
}

/// Normalise a negative-zero imaginary part away. std::arg and std::sqrt read
/// the sign of a zero imaginary part, so -0.0 lands on the far side of the cut
/// from +0.0 -- invisible to a real-typed sign test, visible here.
inline std::complex<double> normaliseZeroImaginary(std::complex<double> z) noexcept {
  if (z.imag() == 0.0) return {z.real(), 0.0};
  return z;
}

}  // namespace

double principalArgument(std::complex<double> z) noexcept {
  return std::arg(normaliseZeroImaginary(z));
}

std::complex<double> principalSquareRoot(std::complex<double> z) noexcept {
  return std::sqrt(normaliseZeroImaginary(z));
}

std::complex<double> principalArcCosine(std::complex<double> r) noexcept {
  return std::acos(normaliseZeroImaginary(r));
}

// ---------------------------------------------------------------------------
// SheetedSqrt
// ---------------------------------------------------------------------------

SheetedSqrt::SheetedSqrt(std::complex<double> radicand) noexcept
    : radicand_(normaliseZeroImaginary(radicand)) {}

SheetedSqrt::SheetedSqrt(std::complex<double> radicand, int winding) noexcept
    : radicand_(normaliseZeroImaginary(radicand)), winding_(winding) {}

void SheetedSqrt::advance(std::complex<double> radicand) noexcept {
  const std::complex<double> next = normaliseZeroImaginary(radicand);
  if (std::abs(next) == 0.0 || std::abs(radicand_) == 0.0) {
    // The branch point itself carries no argument, so there is no turn to
    // accumulate: hold the declared sheet and record that the path met it. A
    // path through the branch point does not determine a continuation at all,
    // and inventing one here would hide that from the caller.
    touchedBranchPoint_ = true;
    lastStep_ = 0.0;
    radicand_ = next;
    return;
  }
  const double from = principalArgument(radicand_);
  const double to = principalArgument(next);
  const double step = wrapToPi(to - from);
  lastStep_ = step;
  // The continued argument of the new radicand is (Arg z_old + 2 pi w) + step,
  // and its winding is how many turns that sits above its own principal value.
  // The quotient is an integer up to rounding, so llround recovers it exactly.
  const double continued = from + kTwoPi * static_cast<double>(winding_) + step;
  winding_ = static_cast<int>(std::llround((continued - to) / kTwoPi));
  radicand_ = next;
}

std::complex<double> SheetedSqrt::value() const noexcept {
  const std::complex<double> root = principalSquareRoot(radicand_);
  return (sheet() == 0) ? root : -root;
}

double SheetedSqrt::declaredArgument() const noexcept {
  return principalArgument(radicand_) + kTwoPi * static_cast<double>(winding_);
}

int SheetedSqrt::sheet() const noexcept {
  const int parity = winding_ % 2;
  return (parity < 0) ? parity + 2 : parity;
}

// ---------------------------------------------------------------------------
// SheetedAcos
// ---------------------------------------------------------------------------

SheetedAcos::SheetedAcos(std::complex<double> cosine) noexcept
    : cosine_(normaliseZeroImaginary(cosine)) {}

SheetedAcos::SheetedAcos(std::complex<double> cosine, int branchIndex,
                         int orientation) noexcept
    : cosine_(normaliseZeroImaginary(cosine)),
      branchIndex_(branchIndex),
      orientation_(orientation >= 0 ? 1 : -1) {}

std::complex<double> SheetedAcos::value() const noexcept {
  const std::complex<double> principal = principalArcCosine(cosine_);
  return std::complex<double>(kTwoPi * static_cast<double>(branchIndex_), 0.0) +
         static_cast<double>(orientation_) * principal;
}

void SheetedAcos::advance(std::complex<double> cosine) noexcept {
  const std::complex<double> previous = value();
  const std::complex<double> next = normaliseZeroImaginary(cosine);
  const std::complex<double> principal = principalArcCosine(next);

  double best = std::numeric_limits<double>::infinity();
  int bestIndex = branchIndex_;
  int bestOrientation = orientation_;
  for (const int epsilon : {1, -1}) {
    const std::complex<double> base = static_cast<double>(epsilon) * principal;
    // For a fixed epsilon the candidates differ by 2 pi in the real part, so the
    // nearest k is the rounded offset; its two neighbours cover the ties and the
    // cases where the imaginary parts decide.
    const long long centre =
        std::llround((previous.real() - base.real()) / kTwoPi);
    for (const long long k : {centre - 1, centre, centre + 1}) {
      const std::complex<double> candidate =
          base + std::complex<double>(kTwoPi * static_cast<double>(k), 0.0);
      const double distance = std::abs(candidate - previous);
      if (distance < best) {
        best = distance;
        bestIndex = static_cast<int>(k);
        bestOrientation = epsilon;
      }
    }
  }
  cosine_ = next;
  branchIndex_ = bestIndex;
  orientation_ = bestOrientation;
  lastStep_ = best;
}

}  // namespace tessera::mesh
