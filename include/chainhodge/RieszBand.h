// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_CHAINHODGE_RIESZBAND_H
#define TESSERA_CHAINHODGE_RIESZBAND_H

#include <complex>
#include <cstddef>
#include <limits>
#include <string>
#include <vector>

#include <Eigen/Core>

namespace tessera::chainhodge {

using Complex = std::complex<double>;

/// # Contour
///
/// A closed, positively oriented contour \f$ \Gamma_C \f$ in the complex
/// spectral plane, as quadrature nodes \f$ \zeta_j \f$ with weights
/// \f$ w_j \f$ such that \f$ \frac{1}{2\pi i}\oint_{\Gamma_C} f(\zeta)\,d\zeta
/// \approx \sum_j w_j f(\zeta_j) \f$ — the trapezoidal rule on the pencil
/// resolvent. `circle` builds the
/// \f$ N \f$-node trapezoidal rule on \f$ \zeta(\theta) = c + r e^{i\theta} \f$,
/// for which \f$ w_j = (r/N) e^{i\theta_j} \f$; an explicit rule may be
/// supplied directly. The node count is a certificate, not a policy: a
/// caller compares two node counts to bound the quadrature error.
struct Contour {
  std::vector<Complex> nodes{};
  std::vector<Complex> weights{};
  /// Human-readable description recorded on the band ("circle c=…, r=…, N=…").
  std::string description{};

  [[nodiscard]] static Contour circle(Complex center, double radius, int nodeCount = 32);
  [[nodiscard]] std::size_t nodeCount() const noexcept { return nodes.size(); }
};

/// The certificates of one Riesz band: contour and node count, projector
/// idempotency, rank by SVD under the instance tolerance policy,
/// the maximal resolvent norm on the contour, \f$ \det B_C \f$ and
/// \f$ \operatorname{cond} B_C \f$, and the left/right residuals of the reduced
/// operator when the left frame exists. Quiet NaN means unmeasured. No sign
/// or inertia is extracted from \f$ B_C \f$.
struct BandCertificate {
  std::string contour{};
  int nodeCount{0};
  /// \f$ \|P^2 - P\|_F / \|P\|_F \f$.
  double idempotency{std::numeric_limits<double>::quiet_NaN()};
  /// Numerical rank of \f$ P \f$ (SVD, tolerance \f$ \kappa\,n\,\epsilon_m\,\sigma_{\max} \f$).
  int rank{0};
  double rankTolerance{0.0};
  /// \f$ \sigma_r / \sigma_{r+1} \f$ of \f$ P \f$ (\f$ +\infty \f$ when nothing is discarded).
  double singularGap{std::numeric_limits<double>::infinity()};
  /// \f$ \max_j \|(\zeta_j I - h)^{-1}\|_2 \f$ over the contour nodes.
  double resolventMax{std::numeric_limits<double>::quiet_NaN()};
  /// \f$ \det B_C \f$ and \f$ \operatorname{cond}_2 B_C \f$ of the pairing matrix.
  Complex detB{0.0, 0.0};
  double condB{std::numeric_limits<double>::quiet_NaN()};
  /// \f$ \sigma_{\min}(B_C) / \|G^U\Phi\|_2 \f$ with orthonormal frames: the
  /// normalized pairing, zero exactly when a direction of the band is
  /// self-orthogonal (rank one: \f$ |u^{\vee T} G u| / \|Gu\| \f$). The isotropy
  /// test compares this to the declared tolerance; a condition number cannot
  /// see rank-one isotropy.
  double pairingScale{std::numeric_limits<double>::quiet_NaN()};
  /// Whether the canonical left frame exists (\f$ \det B_C \ne 0 \f$ at the
  /// declared isotropy tolerance) and, if not, why.
  bool leftFrameAvailable{false};
  std::string leftFrameRefusal{};
  /// \f$ \|h\Phi - \Phi J\| \f$ and \f$ \|\tilde\Phi^T h - J\tilde\Phi^T\| \f$ with
  /// \f$ J = \tilde\Phi^T h \Phi \f$, each relative to the band's own scale
  /// \f$ \max(\|h\Phi\|, \rho\|\Phi\|) \f$ with \f$ \rho \f$ the largest contour node
  /// modulus (so a zero band is not 0/0); NaN without a left frame.
  double rightResidual{std::numeric_limits<double>::quiet_NaN()};
  double leftResidual{std::numeric_limits<double>::quiet_NaN()};
};

/// # Band
///
/// One Riesz band of the covariant pencil operator \f$ h_k(s,U) \f$: the
/// projector \f$ P_C(U) \f$ on chains, a right frame
/// \f$ \Phi_C \f$ spanning \f$ \operatorname{Ran}P_C(U) \f$, the same contour's
/// band for the dual connection \f$ \Phi_C^\vee = \Phi_C(U^{-1}) \f$, the
/// geometric images \f$ Z = G_k^U\Phi_C \f$, the pairing matrix
/// \f$ B_C(U) = (\Phi_C^\vee)^T G_k^U \Phi_C \f$, the canonical left frame
/// \f$ \tilde\Phi_C = G_k^{U^{-1}}\Phi_C^\vee B_C^{-T} \f$ (so that
/// \f$ \tilde\Phi^T\Phi = I \f$; empty when the band is isotropic), the
/// reduced operator \f$ J = \tilde\Phi^T h \Phi \f$ (Jordan structure retained),
/// and the covariance \f$ \Gamma = \Phi\tilde\Phi^T \f$ with occupations
/// \f$ n_e = \Gamma_{ee} \f$ — at \f$ U = 1 \f$ this is \f$ \Phi\Phi^T G_1 \f$,
/// the \f$ G_1 \f$-orthogonal projector onto the fiber.
///
/// Reference: Kato, "Perturbation Theory for Linear Operators", 1966, for the
/// Riesz projector of a spectral band.
struct Band {
  int degree{0};
  Contour contour{};
  Eigen::MatrixXcd projector{};
  Eigen::MatrixXcd frame{};        // Phi (chains), n x r
  Eigen::MatrixXcd dualFrame{};    // Phi^vee (chains), n x r
  Eigen::MatrixXcd images{};       // Z = G^U Phi
  Eigen::MatrixXcd pairing{};      // B_C
  Eigen::MatrixXcd leftFrame{};    // Phi~ (empty when refused)
  Eigen::MatrixXcd reduced{};      // J
  Eigen::MatrixXcd covariance{};   // Gamma = Phi Phi~^T
  BandCertificate certificate{};
  [[nodiscard]] int rank() const noexcept { return static_cast<int>(frame.cols()); }
  /// \f$ n_e = \Gamma_{ee} \f$ (empty without a left frame).
  [[nodiscard]] Eigen::VectorXcd occupations() const { return covariance.diagonal(); }
};

}  // namespace tessera::chainhodge

#endif  // TESSERA_CHAINHODGE_RIESZBAND_H
