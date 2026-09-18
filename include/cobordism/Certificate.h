// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_COBORDISM_CERTIFICATE_H
#define TESSERA_COBORDISM_CERTIFICATE_H

#include <complex>
#include <limits>
#include <string>
#include <vector>

namespace tessera::cobordism {

/// How a result was obtained, in decreasing order of a-priori strength: the
/// class of statement the producer makes.
enum class CertificateGrade {
  /// A closed-form identity evaluated in floating point, such as the
  /// Kronecker-sum spectrum rule; the only error source is rounding, so the
  /// residual sits at machine precision.
  AlgebraicallyExact,
  /// Exact given a verified structural premise: a Woodbury solve is exact
  /// provided the registered low-rank factors span the whole operator change.
  /// The premise check and the arithmetic residual are both reported.
  StructureExact,
  /// An iterative or truncated computation carrying an explicit residual, a
  /// condition number and, on crossover fixtures, a dense-reference error.
  CertifiedNumerical,
  /// An uncertified proposal (search heuristics, discovery scores). Never
  /// `holds()`.
  HeuristicDiscovery,
};

/// The spectral domain a certificate speaks for. `Static` is the
/// zero-frequency, whole-operator statement; `BandWindow` restricts the claim
/// to an explicit frequency window \f$ \Omega \f$.
enum class CertificateDomain { Static, BandWindow };

/// The metric regime the producing kernel verified. A self-adjoint solver is
/// applied only under `PositiveSemidefinite` or `HermitianIndefinite`;
/// `NonNormal` results carry general-eigensolver conditioning instead.
///
/// `ComplexSymmetricPencil` is the chain-level Whitney pencil's regime: the
/// operator is symmetric for a complex symmetric chain metric \f$ M \f$,
/// \f$ M L = (M L)^T \f$ (for a dressed connection,
/// \f$ (\tilde A^U)^T = \tilde A^{U^{-1}} \f$). Its pairings are bilinear, so a
/// band carries `det B_C`, `cond B_C` and an isotropy certificate but no
/// inertia.
enum class CertificateRegime {
  PositiveSemidefinite,
  HermitianIndefinite,
  NonNormal,
  ComplexSymmetricPencil,
};

/// The serialized spelling of a grade, domain or regime, and its inverse.
///
/// These are the tokens `toRecord`/`fromRecord` write and read across the
/// cobordism and observables subsystems. They live here because the readers
/// throw on an unrecognised token, so a second spelling anywhere produces
/// records nothing can parse.
[[nodiscard]] const char *gradeName(CertificateGrade grade) noexcept;
[[nodiscard]] const char *domainName(CertificateDomain domain) noexcept;
[[nodiscard]] const char *regimeName(CertificateRegime regime) noexcept;

/// Throws std::invalid_argument on a token none of the above produces.
[[nodiscard]] CertificateGrade gradeFromName(const std::string &name);
[[nodiscard]] CertificateDomain domainFromName(const std::string &name);
[[nodiscard]] CertificateRegime regimeFromName(const std::string &name);

/// # Certificate
///
/// The certification record attached to a kernel result: the claim grade, its
/// domain and metric regime, and the measured numbers below. Quantities are
/// relative unless the producer documents otherwise; an unmeasured one is a
/// quiet NaN, not zero.
class Certificate {
  public:
    /// Not-measured marker for the optional fields.
    static constexpr double kUnmeasured = std::numeric_limits<double>::quiet_NaN();

    /// Default: an uncertified `HeuristicDiscovery` with nothing measured.
    Certificate() = default;

    /// An `AlgebraicallyExact` claim with its measured rounding residual.
    [[nodiscard]] static Certificate algebraicallyExact(CertificateDomain domain,
                                                        CertificateRegime regime,
                                                        double residual,
                                                        double tolerance);

    /// A `StructureExact` claim: exact given the verified structural premise;
    /// `conditioning` is the condition estimate of the linear algebra that
    /// evaluated it (e.g. the Woodbury capacitance).
    [[nodiscard]] static Certificate structureExact(CertificateDomain domain,
                                                    CertificateRegime regime,
                                                    double residual,
                                                    double conditioning,
                                                    double tolerance);

    /// A `CertifiedNumerical` claim with residual and conditioning.
    [[nodiscard]] static Certificate certifiedNumerical(CertificateDomain domain,
                                                        CertificateRegime regime,
                                                        double residual,
                                                        double conditioning,
                                                        double tolerance);

    /// An uncertified `HeuristicDiscovery` marker (never `holds()`).
    [[nodiscard]] static Certificate heuristicDiscovery(CertificateDomain domain,
                                                        CertificateRegime regime);

    /// The claim class (see `CertificateGrade`).
    [[nodiscard]] CertificateGrade grade() const noexcept { return grade_; }
    /// The spectral domain the claim speaks for.
    [[nodiscard]] CertificateDomain domain() const noexcept { return domain_; }
    /// The metric regime the producing kernel verified.
    [[nodiscard]] CertificateRegime regime() const noexcept { return regime_; }

    /// Measured relative residual of the produced result (NaN = not measured).
    [[nodiscard]] double residual() const noexcept { return residual_; }

    /// Condition estimate of the computation (>= 1; NaN = not measured).
    [[nodiscard]] double conditioning() const noexcept { return conditioning_; }

    /// Relative error against the dense reference kernel, when the result was
    /// cross-checked on a crossover fixture (NaN = not cross-checked).
    [[nodiscard]] double denseReferenceError() const noexcept {
      return denseReferenceError_;
    }
    /// Record the dense-reference error measured on a crossover fixture.
    void setDenseReferenceError(double error) noexcept {
      denseReferenceError_ = error;
    }

    /// The tolerance the producer declared for `holds()`.
    [[nodiscard]] double tolerance() const noexcept { return tolerance_; }

    /// Whether the certificate stands: a certified grade whose measured
    /// residual met the declared tolerance. `HeuristicDiscovery` never holds;
    /// an unmeasured (NaN) residual never holds.
    [[nodiscard]] bool holds() const noexcept {
      return grade_ != CertificateGrade::HeuristicDiscovery &&
             residual_ <= tolerance_;
    }

    /// One-line human-readable summary (grade/domain/regime + numbers).
    [[nodiscard]] std::string describe() const;

  private:
    Certificate(CertificateGrade grade, CertificateDomain domain,
                CertificateRegime regime, double residual, double conditioning,
                double tolerance) noexcept
        : grade_(grade), domain_(domain), regime_(regime), residual_(residual),
          conditioning_(conditioning), tolerance_(tolerance) {}

    CertificateGrade grade_{CertificateGrade::HeuristicDiscovery};
    CertificateDomain domain_{CertificateDomain::Static};
    CertificateRegime regime_{CertificateRegime::NonNormal};
    double residual_{kUnmeasured};
    double conditioning_{kUnmeasured};
    double denseReferenceError_{kUnmeasured};
    double tolerance_{0.0};
};

/// A vector-valued kernel result (a solution, an eigenvalue list, a spectrum)
/// together with the `Certificate` that grades it.
struct CertifiedVector {
  /// The result values (a solution vector or a sorted eigenvalue list).
  std::vector<std::complex<double>> values{};
  /// The certification record grading `values`.
  Certificate certificate{};
};

}  // namespace tessera::cobordism

#endif  // TESSERA_COBORDISM_CERTIFICATE_H
