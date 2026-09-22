// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_CHAINHODGE_LORENTZIANFAMILY_H
#define TESSERA_CHAINHODGE_LORENTZIANFAMILY_H

#include <complex>
#include <limits>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "chainhodge/ChainHodge.h"
#include "chainhodge/WhitneyMass.h"
#include "cobordism/ChainComplex.h"

namespace tessera::chainhodge {

/// One member of the \f$ \varepsilon \f$ family: the read of a Lorentzian
/// instance at a reported rotation, carrying its allowability,
/// margin, and the harmonic kernel's gap certificate. A read at
/// \f$ \varepsilon = 0 \f$ exists only as a member of a family and always
/// carries its gap.
struct LorentzianRead {
  double epsilon{0.0};
  bool allowable{false};
  double margin{0.0};
  int degree{1};
  HarmonicRead harmonic{};
  /// Dense spectrum at the degree when it was requested and the instance was
  /// below the crossover; empty otherwise.
  std::vector<Complex> eigenvalues{};
};

/// The labeled extrapolation of a family of reads to \f$ \varepsilon \to 0 \f$:
/// a least-squares polynomial in \f$ \varepsilon \f$ through reads at
/// \f$ \varepsilon > 0 \f$ evaluated at zero. It is a separate, labeled step,
/// never a computed value of the instance at \f$ \varepsilon = 0 \f$.
struct LorentzianExtrapolation {
  std::vector<double> epsilons{};
  std::vector<Complex> values{};
  int order{0};
  Complex extrapolated{0.0, 0.0};
  /// Root-mean-square misfit of the polynomial on the supplied reads.
  double residual{0.0};
  std::string label{"extrapolation to epsilon -> 0 from reads at epsilon > 0; not an instance value"};
};

/// # LorentzianFamily
///
/// A Lorentzian instance is computed as the family \f$ s_e(\varepsilon) \f$ with
/// the timelike part of every squared length rotated by
/// \f$ e^{-2i\varepsilon} \f$ — complex lengths on the allowable side of the
/// Kontsevich–Segal boundary — at one or more reported
/// \f$ \varepsilon > 0 \f$ (integration specification, Requirement 2). Complex
/// lengths are the \f$ i\varepsilon \f$, and spectral bands are selected on the
/// complex plane. Results at \f$ \varepsilon = 0 \f$ are reported only alongside
/// their gap certificate and never alone; extrapolation to
/// \f$ \varepsilon \to 0 \f$ is a separate, labeled step. The instance
/// certificate carries \f$ \varepsilon \f$.
///
/// Every squared length is declared together with its timelike part
/// \f$ \tau_e \f$, the contribution of the edge's temporal displacement: for a
/// metric \f$ e^{2\varphi}(-dt^2 + dx^2) \f$ an edge with displacement
/// \f$ (\Delta t_e, \Delta x_e) \f$ has
/// \f$ s_e = e^{2\varphi}(\Delta x_e^2 - \Delta t_e^2) \f$ and
/// \f$ \tau_e = -e^{2\varphi}\Delta t_e^2 \f$. The family rotates that part
/// and keeps the spacelike remainder \f$ s_e - \tau_e \f$:
/// \f$ s_e(\varepsilon) = (s_e - \tau_e) + e^{-2i\varepsilon}\tau_e \f$. An
/// edge that is purely spacelike has \f$ \tau_e = 0 \f$ and does not move; a
/// diagonal of a causal lattice, spacelike or null as a whole, still has its
/// timelike part rotated. The split is an input: nothing here infers it from a
/// squared length or classifies an edge.
///
/// Reference: Kontsevich & Segal, "Wick rotation and the positivity of energy
/// in quantum field theory", arXiv:2105.10161.
class LorentzianFamily {
 public:
  /// \f$ s_e(\varepsilon) = s_e + (e^{-2i\varepsilon} - 1)\,\tau_e \f$: the
  /// timelike part \f$ \tau_e \f$ of every squared length rotated by
  /// \f$ e^{-2i\varepsilon} \f$, the spacelike part \f$ s_e - \tau_e \f$
  /// unchanged. At \f$ \varepsilon = 0 \f$ it returns \p s exactly.
  /// @throws std::invalid_argument when \p timelikeParts and \p s differ in
  ///   length, when a timelike part is not finite, or when \p epsilon is
  ///   negative or not finite (the family lies at \f$ \varepsilon \ge 0 \f$).
  [[nodiscard]] static SquaredLengths rotate(const SquaredLengths &s,
                                             const SquaredLengths &timelikeParts,
                                             double epsilon);

  /// The instance at \f$ \varepsilon \f$: a `ChainHodge` over
  /// `rotate(s, timelikeParts, epsilon)` whose certificate records
  /// \f$ \varepsilon \f$.
  [[nodiscard]] static ChainHodge instance(const cobordism::ChainComplex &K,
                                           const SquaredLengths &s,
                                           const SquaredLengths &timelikeParts,
                                           double epsilon, Preset preset = Preset::L2,
                                           Branch branch = Branch::Continuation,
                                           int crossoverDimension = ChainHodge::kDefaultCrossoverDimension);

  /// The family of reads at the given rotations, at one degree: allowability,
  /// margin, and the harmonic kernel with its gap for every member (the
  /// \f$ \varepsilon = 0 \f$ member therefore carries its gap), plus the dense
  /// spectrum when \p withSpectrum is set and the instance is below the
  /// crossover.
  /// @throws std::invalid_argument when \p epsilons holds no
  ///   \f$ \varepsilon > 0 \f$ (a read at \f$ \varepsilon = 0 \f$ is never
  ///   reported alone), or any negative or non-finite rotation.
  [[nodiscard]] static std::vector<LorentzianRead> sweep(
      const cobordism::ChainComplex &K, const SquaredLengths &s,
      const SquaredLengths &timelikeParts, const std::vector<double> &epsilons, int degree,
      Preset preset = Preset::L2, Branch branch = Branch::Continuation, double kappa = 10.0,
      bool withSpectrum = false,
      int crossoverDimension = ChainHodge::kDefaultCrossoverDimension);

  /// Least-squares polynomial of degree \f$ \min(\text{order}, n-1) \f$ in
  /// \f$ \varepsilon \f$ through reads at \f$ \varepsilon > 0 \f$, evaluated at
  /// zero. Labeled as an extrapolation.
  /// @throws std::invalid_argument on fewer than two reads, mismatched
  ///   lengths, or any \f$ \varepsilon \le 0 \f$ (an instance value at zero is
  ///   not an extrapolation input).
  [[nodiscard]] static LorentzianExtrapolation extrapolateToZero(
      const std::vector<double> &epsilons, const std::vector<Complex> &values, int order = 2);
};

}  // namespace tessera::chainhodge

#endif  // TESSERA_CHAINHODGE_LORENTZIANFAMILY_H
