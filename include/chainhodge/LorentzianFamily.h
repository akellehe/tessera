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

/// The declared causal type of an edge. This is an INPUT: it is never
/// inferred from the real part, imaginary part, argument, or modulus of a
/// squared length.
enum class CausalType { Spacelike, Timelike, Null };
/// One declared causal type per edge, in the canonical edge order.
using CausalTypes = std::vector<CausalType>;

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
/// \f$ \varepsilon > 0 \f$. Complex lengths are the \f$ i\varepsilon \f$, and
/// spectral bands are selected on the complex plane. Results at
/// \f$ \varepsilon = 0 \f$ are reported only alongside their gap certificate and
/// never alone; extrapolation to \f$ \varepsilon \to 0 \f$ is a separate,
/// labeled step. The instance certificate carries \f$ \varepsilon \f$.
///
/// Every squared length carries a declared `CausalType`; `rotate` multiplies
/// the timelike ones by \f$ e^{-2i\varepsilon} \f$ and leaves the others
/// untouched. Null edges are not rotated. Nothing here classifies an edge.
///
/// Reference: Kontsevich & Segal, "Wick rotation and the positivity of energy
/// in quantum field theory", arXiv:2105.10161.
class LorentzianFamily {
 public:
  /// \f$ s_e(\varepsilon) \f$: timelike entries multiplied by
  /// \f$ e^{-2i\varepsilon} \f$, spacelike and null entries unchanged.
  /// @throws std::invalid_argument when \p types and \p s differ in length.
  [[nodiscard]] static SquaredLengths rotate(const SquaredLengths &s, const CausalTypes &types,
                                             double epsilon);

  /// The instance at \f$ \varepsilon \f$: a `ChainHodge` over `rotate(s, types, epsilon)`
  /// whose certificate records \f$ \varepsilon \f$.
  [[nodiscard]] static ChainHodge instance(const cobordism::ChainComplex &K,
                                           const SquaredLengths &s, const CausalTypes &types,
                                           double epsilon, Preset preset = Preset::L2,
                                           Branch branch = Branch::Continuation,
                                           int crossoverDimension = ChainHodge::kDefaultCrossoverDimension);

  /// The family of reads at the given rotations, at one degree: allowability,
  /// margin, and the harmonic kernel with its gap for every member (the
  /// \f$ \varepsilon = 0 \f$ member therefore carries its gap), plus the dense
  /// spectrum when \p withSpectrum is set and the instance is below the
  /// crossover.
  [[nodiscard]] static std::vector<LorentzianRead> sweep(
      const cobordism::ChainComplex &K, const SquaredLengths &s, const CausalTypes &types,
      const std::vector<double> &epsilons, int degree, Preset preset = Preset::L2,
      Branch branch = Branch::Continuation, double kappa = 10.0, bool withSpectrum = false,
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
