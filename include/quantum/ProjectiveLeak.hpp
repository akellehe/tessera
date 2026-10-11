// The projective leak of one ray against another, and its first-order
// change: how much of a wanted state lies outside the ray a reading spans.
//
// Both arguments are Eigen vectors or matrices of one shape. Vectors pair by
// <r, w> = r^dagger w and matrices by the Frobenius product tr(r^dagger w),
// so an operator and its Choi vector leak alike.
//
// Each function is forced inline: a body inlined late is contracted into
// fused multiply-adds differently from the same expression written at the
// call site, which moves the last bit of the result under -march=native.

#pragma once

#include <Eigen/Dense>

#include <algorithm>
#include <complex>

namespace tessera::graph {}
namespace tessera::mesh {}
namespace tessera::observables {}
namespace tessera::simulations {}
namespace tessera::spacetime {}
namespace tessera::quantum {

namespace detail {
/// \f$ \langle a,b\rangle = \sum_i \bar a_i b_i \f$. Vectors take Eigen's
/// `dot`, which is defined only for vectors, and matrices the
/// conjugate-product sum; the two round differently under FMA contraction,
/// so neither stands in for the other.
template <typename A, typename B>
[[nodiscard, gnu::always_inline]] inline std::complex<double>
rayProduct(const Eigen::MatrixBase<A> &a, const Eigen::MatrixBase<B> &b) {
  if constexpr (A::IsVectorAtCompileTime && B::IsVectorAtCompileTime)
    return a.dot(b);
  else
    return (a.conjugate().cwiseProduct(b)).sum();
}
} // namespace detail

/// The projective leak of `wanted` against the ray of `reading`:
/// \f$ \min_c \|c\,r - w\|^2/\|w\|^2 = 1 - |\langle r,w\rangle|^2/(\|r\|^2\|w\|^2) \f$,
/// zero when the two are parallel and one when they are orthogonal. Taken as
/// \f$ (\|w\|^2 - |\langle r,w\rangle|^2/\|r\|^2)/\|w\|^2 \f$ and clamped at
/// zero against round-off. A zero reading spans no ray and leaks in full
/// (1.0). `wanted` must be nonzero.
template <typename Reading, typename Wanted>
[[nodiscard, gnu::always_inline]] inline double
projectiveLeak(const Eigen::MatrixBase<Reading> &reading, const Eigen::MatrixBase<Wanted> &wanted) {
  const double readingSquaredNorm = reading.squaredNorm();
  if (!(readingSquaredNorm > 0.0)) return 1.0;
  const std::complex<double> overlap = detail::rayProduct(reading, wanted);  // <r, w>
  const double leak = wanted.squaredNorm() - std::norm(overlap) / readingSquaredNorm;
  return std::max(0.0, leak / wanted.squaredNorm());
}

/// The sensitivity of `projectiveLeak(reading, wanted)` to a move `dReading`
/// of the reading, `wanted` held fixed: the leak changes by \f$ 2\,\mathrm{Re}\,dF \f$
/// to first order, with
/// \f$ dF = -(\langle r,w\rangle\langle w,dr\rangle\|r\|^2 - |\langle r,w\rangle|^2\langle r,dr\rangle)/(\|r\|^4\|w\|^2) \f$.
/// When the reading is holomorphic in a coordinate \f$ z \f$ and `dReading` is
/// \f$ \partial r/\partial z \f$, \f$ dF \f$ is the Wirtinger derivative of the
/// leak in \f$ z \f$. `overlap`, `readingSquaredNorm` and `wantedSquaredNorm` are
/// \f$ \langle r,w\rangle \f$, \f$ \|r\|^2 \f$ and \f$ \|w\|^2 \f$, passed in
/// because a sweep over many moves computes them once.
template <typename Reading, typename Wanted, typename Move>
[[nodiscard, gnu::always_inline]] inline std::complex<double>
projectiveLeakSensitivity(const Eigen::MatrixBase<Reading> &reading, const Eigen::MatrixBase<Wanted> &wanted,
                          const Eigen::MatrixBase<Move> &dReading, std::complex<double> overlap,
                          double readingSquaredNorm, double wantedSquaredNorm) {
  const std::complex<double> wantedAlong = detail::rayProduct(wanted, dReading);    // <w, dr>
  const std::complex<double> readingAlong = detail::rayProduct(reading, dReading);  // <r, dr>
  return -(overlap * wantedAlong * readingSquaredNorm - std::norm(overlap) * readingAlong) /
         (readingSquaredNorm * readingSquaredNorm * wantedSquaredNorm);
}

} // namespace tessera::quantum
