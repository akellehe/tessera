// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_OBSERVABLES_EFFECTIVETOPOLOGY_H
#define TESSERA_OBSERVABLES_EFFECTIVETOPOLOGY_H

#include <limits>
#include <string>
#include <vector>

#include "chainhodge/CovariantChainHodge.h"
#include "chainhodge/SparsePencilSolver.h"

namespace tessera::observables {

/// The effective Betti number of one degree of a declared operator at a scale.
struct EffectiveBettiNumber {
  /// How the count was obtained.
  enum class Method {
    /// The dense spectrum of the degree's pencil (below the crossover).
    DenseSpectrum,
    /// `chainhodge::SparsePencilSolver::effectiveBetti` on the sparse degree-zero pencil.
    SparsePencil,
    /// Not measured: the degree has no sparse pencil and is at or above the
    /// crossover, or the sparse pencil is not Hermitian. `reason` says which.
    Unmeasured,
  };

  int degree{0};
  /// \f$ \beta_k^{\mathrm{eff}}(\epsilon) \f$: the number of eigenvalues with
  /// \f$ |\lambda| \le \epsilon \f$, with multiplicity; \f$ -1 \f$ when
  /// unmeasured.
  int rank{-1};
  /// The largest modulus inside the window; quiet NaN when the band is empty.
  double lastInside{std::numeric_limits<double>::quiet_NaN()};
  /// The smallest modulus outside the window; quiet NaN when there is none.
  double firstOutside{std::numeric_limits<double>::quiet_NaN()};
  /// `firstOutside / lastInside`: \f$ +\infty \f$ for an empty band or a band
  /// at zero, quiet NaN when nothing lies outside the window.
  double gap{std::numeric_limits<double>::quiet_NaN()};
  Method method{Method::Unmeasured};
  /// Whether the count is certified: the dense pencil residual, or the sparse
  /// read's own certificate, met the tolerance.
  bool certified{false};
  /// Why the degree is unmeasured; empty otherwise.
  std::string reason{};
};

/// # EffectiveTopology
///
/// What a declared operator sees at a scale, as opposed to what the complex is.
///
/// A complex has an actual topology, fixed by its incidence: the Betti numbers
/// of `cobordism::ChainComplex::bettiNumbers`, which the `spacetime::Topology`
/// classes construct by design (`Toroid`, `Sphere`, `PeriodicKuhnGrid`) and
/// which change only when cells are attached. An operator on that complex has
/// an effective topology: the effective Betti number
/// \f[
///   \beta_k^{\mathrm{eff}}(\epsilon) = \mathrm{rank}\, P_{[0,\epsilon]}\bigl(h_k(s, U)\bigr),
/// \f]
/// the rank of the spectral band of the covariant operator of degree
/// \f$ k \f$ near zero. It depends on the squared lengths and on the
/// connection, it can emerge and disappear under relaxation, and it is the
/// count the dynamics sees. The two agree only in a limit: at the trivial
/// connection, under the rank conditions of `chainhodge::ChainHodge`, and as
/// \f$ \epsilon \to 0 \f$. Away from it they part in both directions. Two
/// regions joined through a bottleneck are one component by incidence and two
/// effective components at any scale above the bridge level; a torus carrying
/// a connection with nontrivial holonomy keeps its incidence Betti numbers
/// \f$ (1, 3, 3, 1) \f$ while every effective Betti number of the covariant
/// operator is zero.
///
/// This class is the effective read and nothing else: it never consults the
/// incidence ranks, so a caller cannot obtain one kind of number where the
/// other was meant. Each degree is counted from the dense spectrum of its
/// pencil below the crossover. At or above the crossover only degree zero is
/// available, from the sparse pencil by `chainhodge::SparsePencilSolver::effectiveBetti`
/// with the shift at \f$ -\epsilon \f$ (whose Cholesky factorization certifies
/// that nothing lies below the window); the pencils of higher degrees contain
/// an inverse metric and are reported as unmeasured rather than estimated.
class EffectiveTopology {
 public:
  /// Read every degree of \p cov at scale \p epsilon.
  /// @param tolerance The dense pencil residual, or the sparse certificate's
  ///   residual, a degree must meet to be certified.
  /// @throws std::invalid_argument when \p epsilon is not positive.
  [[nodiscard]] static EffectiveTopology read(const chainhodge::CovariantChainHodge &cov, double epsilon,
                                              double tolerance = 1e-10);

  [[nodiscard]] double epsilon() const noexcept { return epsilon_; }
  [[nodiscard]] int dimension() const noexcept { return static_cast<int>(degrees_.size()) - 1; }
  /// One read per degree, \f$ k = 0 \ldots d \f$.
  [[nodiscard]] const std::vector<EffectiveBettiNumber> &degrees() const noexcept { return degrees_; }
  /// \f$ (\beta_0^{\mathrm{eff}}, \ldots, \beta_d^{\mathrm{eff}}) \f$ with
  /// \f$ -1 \f$ for an unmeasured degree.
  [[nodiscard]] std::vector<int> betti() const;
  /// True when every degree was measured and certified.
  [[nodiscard]] bool certified() const noexcept;

 private:
  double epsilon_{0.0};
  std::vector<EffectiveBettiNumber> degrees_{};
};

/// The verdict of an `EffectiveSignature` on an `EffectiveTopology`.
struct EffectiveSignatureCertificate {
  /// The signature's name, such as "effective 3-torus".
  std::string signature{};
  double epsilon{0.0};
  /// The Betti numbers the signature requires, \f$ -1 \f$ where it requires
  /// nothing.
  std::vector<int> expected{};
  /// The effective Betti numbers that were read, \f$ -1 \f$ where unmeasured.
  std::vector<int> measured{};
  /// Every required degree was measured and equals its required value.
  bool matches{false};
  /// Every required degree's read was certified.
  bool certified{false};
  /// The smallest gap among the required degrees: how far the recognized
  /// bands stand from the rest of the spectrum.
  double gap{std::numeric_limits<double>::quiet_NaN()};

  [[nodiscard]] bool holds() const noexcept { return matches && certified; }
};

/// # EffectiveSignature
///
/// A named pattern of effective Betti numbers, recognized in an operator
/// rather than built into a complex: the effective counterpart of a
/// `spacetime::Topology`. `spacetime::Toroid` constructs a complex whose
/// incidence is a torus; `EffectiveTorus` certifies that an operator, on
/// whatever complex, has the near-kernel of one.
///
/// The claim is one of rank. Betti numbers do not determine a space up to
/// homeomorphism, and an effective signature says nothing about incidence, so
/// a certificate that holds means that the operator's spectral bands near zero
/// have the ranks of the named space at the stated scale, with the stated gap,
/// and nothing more.
class EffectiveSignature {
 public:
  virtual ~EffectiveSignature() = default;
  [[nodiscard]] virtual std::string name() const = 0;
  /// The required \f$ (\beta_0, \ldots, \beta_d) \f$, \f$ -1 \f$ where nothing
  /// is required.
  [[nodiscard]] virtual std::vector<int> betti() const = 0;

  /// Compare an effective read with this signature.
  /// @throws std::invalid_argument when the read has a different dimension.
  [[nodiscard]] EffectiveSignatureCertificate certify(const EffectiveTopology &topology) const;
  /// `certify(EffectiveTopology::read(cov, epsilon, tolerance))`.
  [[nodiscard]] EffectiveSignatureCertificate certify(const chainhodge::CovariantChainHodge &cov, double epsilon,
                                                      double tolerance = 1e-10) const;
};

/// The effective \f$ d \f$-torus: \f$ \beta_k^{\mathrm{eff}} = \binom{d}{k} \f$.
class EffectiveTorus : public EffectiveSignature {
 public:
  /// @throws std::invalid_argument for \f$ d < 1 \f$.
  explicit EffectiveTorus(int dimension);
  [[nodiscard]] std::string name() const override;
  [[nodiscard]] std::vector<int> betti() const override;

 private:
  int dimension_;
};

/// The effective \f$ d \f$-sphere: \f$ \beta_0^{\mathrm{eff}} =
/// \beta_d^{\mathrm{eff}} = 1 \f$ and zero between.
class EffectiveSphere : public EffectiveSignature {
 public:
  /// @throws std::invalid_argument for \f$ d < 1 \f$.
  explicit EffectiveSphere(int dimension);
  [[nodiscard]] std::string name() const override;
  [[nodiscard]] std::vector<int> betti() const override;

 private:
  int dimension_;
};

/// \f$ n \f$ effective components: \f$ \beta_0^{\mathrm{eff}} = n \f$ on a
/// complex of dimension \f$ d \f$, with no requirement on the higher degrees.
/// This is the signature a mesh above the dense crossover can be held to,
/// since degree zero is the one read from the sparse pencil.
class EffectiveComponents : public EffectiveSignature {
 public:
  /// @throws std::invalid_argument for \f$ n < 0 \f$ or \f$ d < 0 \f$.
  EffectiveComponents(int count, int dimension);
  [[nodiscard]] std::string name() const override;
  [[nodiscard]] std::vector<int> betti() const override;

 private:
  int count_;
  int dimension_;
};

}  // namespace tessera::observables

#endif  // TESSERA_OBSERVABLES_EFFECTIVETOPOLOGY_H
