// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_COBORDISM_DENSEREFERENCE_H
#define TESSERA_COBORDISM_DENSEREFERENCE_H

#include <complex>
#include <cstddef>
#include <vector>

#include "cobordism/Certificate.h"

namespace tessera::cobordism {

/// # DenseReference
///
/// Dense reference kernels, usable only below a configurable dimension
/// crossover. They supply the independent answer a structured path is compared
/// against (a `Certificate`'s `denseReferenceError`); at or above the crossover
/// they throw.
///
/// Kernels:
///  - `solve` — dense partial-pivot LU factor solve; the inverse is never
///    formed;
///  - `spectrum` — dense eigenvalues. The self-adjoint solver is applied only
///    after Hermiticity is verified
///    (\f$ \|A - A^\dagger\| \le \text{tol}\cdot\|A\| \f$); otherwise the
///    general non-normal solver runs and the certificate records that;
///  - `fockSpectrum` — a dense one-particle eigensolve, then explicit
///    occupation subset-sum enumeration.
///
/// Every result carries a `Certificate` with the measured residual and
/// conditioning.
class DenseReference {
  public:
    /// Dense kernels are for fixtures, not production scale.
    static constexpr int kDefaultCrossoverDimension = 512;

    /// @param crossoverDimension The dimension at and above which every
    ///   dense kernel refuses. @throws std::invalid_argument if < 1.
    explicit DenseReference(int crossoverDimension = kDefaultCrossoverDimension);

    /// The dimension at and above which dense kernels refuse.
    [[nodiscard]] int crossoverDimension() const noexcept { return crossover_; }
    /// Reconfigure the crossover. @throws std::invalid_argument if < 1.
    void setCrossoverDimension(int crossoverDimension);
    /// Whether a `dim`-dimensional dense computation is permitted.
    [[nodiscard]] bool belowCrossover(int dim) const noexcept {
      return dim < crossover_;
    }

    /// Dense LU solve of \f$ Ax = b \f$ (flat row-major `dim` x `dim`).
    /// Certificate: structure-exact factor solve with measured relative
    /// residual \f$ \|Ax-b\|/\|b\| \f$ and the LU condition estimate.
    /// @throws std::invalid_argument on size mismatch; std::length_error at
    ///   or above the crossover.
    [[nodiscard]] CertifiedVector solve(const std::vector<std::complex<double>> &matrix,
                                        int dim,
                                        const std::vector<std::complex<double>> &rhs,
                                        double tolerance = 1e-12) const;

    /// Dense eigenvalues of a `dim` x `dim` operator, sorted ascending by
    /// \f$ (\mathrm{Re}, \mathrm{Im}) \f$. `selfAdjoint = true` requests the
    /// self-adjoint solver; the request is honoured only when
    /// \f$ \|A-A^\dagger\| \le \text{tol}\cdot\|A\| \f$ is verified, else the
    /// general solver runs and the certificate's regime reports `NonNormal`.
    /// The residual is \f$ \max_i \|Av_i-\lambda_iv_i\| / \|A\| \f$ over the
    /// computed pairs; the conditioning is the eigenvector matrix condition
    /// estimate (1 on the verified self-adjoint path).
    /// @throws std::invalid_argument on size mismatch; std::length_error at
    ///   or above the crossover.
    [[nodiscard]] CertifiedVector spectrum(const std::vector<std::complex<double>> &matrix,
                                           int dim, bool selfAdjoint,
                                           double tolerance = 1e-10) const;

    /// Eigenvalues of the one-particle operator (as `spectrum`), then the exact
    /// \f$ \binom{n}{N} \f$ occupation subset sums for the `particles` sector.
    /// The certificate inherits the eigensolve's residual, regime and
    /// conditioning; the enumeration adds only rounding.
    /// @throws as `spectrum`, plus std::length_error when the sector itself
    ///   is unmaterializable.
    [[nodiscard]] CertifiedVector fockSpectrum(
        const std::vector<std::complex<double>> &oneParticle, int dim,
        int particles, bool selfAdjoint, double tolerance = 1e-10) const;

  private:
    void requireBelowCrossover(int dim, const char *kernel) const;

    int crossover_{kDefaultCrossoverDimension};
};

}  // namespace tessera::cobordism

#endif  // TESSERA_COBORDISM_DENSEREFERENCE_H
