// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_COBORDISM_LOWRANKUPDATE_H
#define TESSERA_COBORDISM_LOWRANKUPDATE_H

#include <Eigen/Core>
#include <Eigen/LU>

#include <complex>
#include <vector>

#include "cobordism/Certificate.h"

namespace tessera::cobordism {

/// # LowRankUpdate
///
/// Structure-exact Woodbury and secular update helpers for low-rank local
/// operator changes.
///
/// **Woodbury identity (solves).** The base operator \f$ A \f$ is factored once
/// by partial-pivot LU (general complex square; no positive-definite or
/// Hermitian assumption). For a registered change \f$ \Delta = UW \f$ with
/// \f$ U: n\times r \f$ and \f$ W: r\times n \f$:
/// \f[ (A+UW)^{-1}b = A^{-1}b - A^{-1}U\,(I_r + WA^{-1}U)^{-1}\,W A^{-1} b. \f]
/// The capacitance \f$ I_r + WA^{-1}U \f$ is itself LU-factored; no explicit
/// inverse is formed. The result is exact when \f$ UW \f$ spans the whole
/// affected change, which `factorsFromTouched` and `spansAffectedChange`
/// verify and `refactor` falls back from.
///
/// **Secular identity (eigenvalues).** For a Hermitian operator with known
/// eigenvalues \f$ d_1\le\dots\le d_n \f$ and a rank-one Hermitian update
/// \f$ \rho\,zz^\dagger \f$ expressed in the eigenbasis, the updated eigenvalues
/// are the roots of the secular equation:
/// \f[ f(\lambda) = 1 + \rho\sum_i \frac{|z_i|^2}{d_i-\lambda} = 0. \f]
/// Each interlacing interval contributes one root, found by bisection to
/// machine bracket width. Indefinite Hermitian operators are allowed; the
/// non-normal Lorentzian regime is refused, since interlacing fails there.
class LowRankUpdate {
  public:
    /// The factors of a touched-star operator change:
    /// \f$ \Delta = \text{left}\cdot\text{right} \f$ with rank at most twice the
    /// touched-index count. `spansChange` false means the delta has support
    /// outside the declared touched rows and columns, so the factors are not
    /// exact and the caller must cold-recompute.
    struct TouchedFactors {
      bool spansChange{false};  ///< exactness verdict (false = cold-recompute)
      int rank{0};              ///< number of factor columns/rows
      std::vector<std::complex<double>> left{};   ///< dim x rank, row-major
      std::vector<std::complex<double>> right{};  ///< rank x dim, row-major
    };

    /// Factor the base operator (flat row-major `dim` x `dim`) with
    /// partial-pivot LU. @throws std::invalid_argument on a size mismatch.
    LowRankUpdate(const std::vector<std::complex<double>> &base, int dim);

    /// The operator dimension \f$ n \f$.
    [[nodiscard]] int dimension() const noexcept { return dim_; }
    /// Rank of the registered change (0 = none).
    [[nodiscard]] int updateRank() const noexcept { return rank_; }

    /// Register the change \f$ \Delta = UW \f$ (`left`: dim x rank,
    /// `right`: rank x dim, both flat row-major), replacing any previous one.
    /// @throws std::invalid_argument on size mismatch.
    void setUpdate(const std::vector<std::complex<double>> &left,
                   const std::vector<std::complex<double>> &right, int rank);
    /// Drop the registered change (back to the bare base operator).
    void clearUpdate() noexcept;

    /// Solve \f$ (A + UW)x = b \f$ by the Woodbury identity above. The
    /// certificate carries the measured relative residual, the conditioning
    /// \f$ \max(\kappa_{LU}(A), \kappa_{LU}(I+WA^{-1}U)) \f$, and the given
    /// tolerance. @throws std::invalid_argument on rhs size.
    [[nodiscard]] CertifiedVector solve(const std::vector<std::complex<double>> &rhs,
                                        double tolerance = 1e-12) const;

    /// Apply the updated operator: \f$ y = (A + UW)x \f$.
    [[nodiscard]] std::vector<std::complex<double>> apply(
        const std::vector<std::complex<double>> &x) const;

    /// Exactness check: whether the registered \f$ UW \f$ spans the whole
    /// change to `updated`, i.e. \f$ \|(\text{updated}-A) - UW\|_F \le
    /// \text{tolerance}\cdot\|\text{updated}\|_F \f$. A false return means the
    /// low-rank path is not exact; cold-recompute instead.
    [[nodiscard]] bool spansAffectedChange(
        const std::vector<std::complex<double>> &updated,
        double tolerance = 1e-12) const;

    /// Build exact factors for the change `base` -> `updated` from the declared
    /// touched row/column index set: rows in `touched` are captured by
    /// identity-selector left factors, the remaining touched columns by the
    /// delta's columns. \f$ UW \f$ equals the delta exactly when the delta's
    /// support lies in the touched rows and columns; `spansChange` reports
    /// whether it does. All-zero rows and columns are trimmed, so the rank is at
    /// most twice the number of active touched indices.
    /// @throws std::invalid_argument on size mismatch or out-of-range index.
    [[nodiscard]] static TouchedFactors factorsFromTouched(
        const std::vector<std::complex<double>> &base,
        const std::vector<std::complex<double>> &updated, int dim,
        const std::vector<int> &touched);

    /// Cold-recompute fallback: refactor `base` as the new base operator and
    /// clear any registered update.
    void refactor(const std::vector<std::complex<double>> &base, int dim);

    /// Secular rank-one Hermitian eigenvalue update: the ascending eigenvalues
    /// of \f$ \mathrm{diag}(d) + \rho zz^\dagger \f$ with `z` in the eigenbasis
    /// of the base operator. The certificate's residual is the largest of the
    /// final relative bisection bracket widths, the relative trace-identity
    /// defect \f$ |\sum\lambda' - \sum d - \rho\|z\|^2| \f$, and the
    /// deflated-weight bound.
    /// @throws std::invalid_argument when `eigenvalues` is not real ascending or
    ///   sizes mismatch.
    [[nodiscard]] static CertifiedVector rankOneEigenvalues(
        const std::vector<double> &eigenvalues,
        const std::vector<std::complex<double>> &z, double rho,
        double tolerance = 1e-10);

  private:
    [[nodiscard]] Eigen::MatrixXcd deltaMatrix() const;

    int dim_{0};
    int rank_{0};
    Eigen::MatrixXcd base_{};
    Eigen::PartialPivLU<Eigen::MatrixXcd> baseFactorization_{};
    /// 1/rcond of the base LU, computed once per (re)factorization.
    double baseConditioning_{1.0};
    Eigen::MatrixXcd left_{};
    Eigen::MatrixXcd right_{};
    /// \f$ Z = A^{-1}U \f$ and the LU of the capacitance \f$ I_r + WZ \f$,
    /// computed once per `setUpdate`.
    Eigen::MatrixXcd capacitanceSolvedLeft_{};
    Eigen::PartialPivLU<Eigen::MatrixXcd> capacitanceFactorization_{};
    double capacitanceConditioning_{1.0};
};

}  // namespace tessera::cobordism

#endif  // TESSERA_COBORDISM_LOWRANKUPDATE_H
