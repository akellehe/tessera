// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_CHAINHODGE_PENCILSCHUR_H
#define TESSERA_CHAINHODGE_PENCILSCHUR_H

#include <complex>
#include <cstddef>
#include <limits>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "chainhodge/WhitneyMass.h"
#include "cobordism/ChainComplex.h"

namespace tessera::chainhodge {

/// One Feshbach complement of a symmetric pencil.
struct FeshbachResult {
  Complex lambda{0.0, 0.0};
  /// Interface (kept) and interior (eliminated) coordinates, ascending.
  std::vector<int> interface{};
  std::vector<int> interior{};
  /// \f$ F_B(\lambda) = P_{BB} - P_{BI} P_{II}^{-1} P_{IB} \f$, \f$ P = A - \lambda M \f$.
  Eigen::MatrixXcd response{};
  /// The constraint modes \f$ T = [I_B;\ -P_{II}^{-1} P_{IB}] \f$ in the full
  /// coordinate order (\f$ n \times |B| \f$): the fibers the kept coordinates
  /// carry, whose congruence \f$ T^T M T \f$ is the inherited chain metric.
  Eigen::MatrixXcd constraintModes{};
  Complex interiorDeterminant{0.0, 0.0};
  Complex responseDeterminant{0.0, 0.0};
  Complex pencilDeterminant{0.0, 0.0};
  /// \f$ |\det P - \det P_{II}\det F_B| / \max(|\det P|, \epsilon) \f$.
  double determinantResidual{std::numeric_limits<double>::quiet_NaN()};
  /// Relative residual of the interior solve.
  double solveResidual{std::numeric_limits<double>::quiet_NaN()};
  /// True when \f$ P_{II} \f$ was singular at the shift (an interior
  /// resonance): the complement is not defined and `response` is empty.
  bool interiorSingular{false};
};

/// A congruence \f$ (T^T A T,\ T^T M T) \f$.
struct CongruenceResult {
  Eigen::MatrixXcd A{};
  Eigen::MatrixXcd M{};
};

/// The coarse pencil and chain metric restricted to retained fibers,
/// \f$ (\hat A, \mathcal G) = (Z^T \tilde A Z,\ Z^T M Z) \f$, with the block
/// offsets of the fibers that were concatenated.
struct FiberRestriction {
  Eigen::MatrixXcd A{};
  Eigen::MatrixXcd gram{};
  std::vector<int> blockOffsets{};
  std::vector<int> blockRanks{};
};

/// A transfer between two fibers with its reversal certificate.
struct TransferResult {
  /// \f$ T_{AB}(U) = (Z_A^\vee)^T (\tilde A^U)_{AB} Z_B \f$.
  Eigen::MatrixXcd forward{};
  /// \f$ T_{BA}(U^{-1}) = (Z_B^\vee(U^{-1}))^T (\tilde A^{U^{-1}})_{BA} Z_A(U^{-1})
  ///   = Z_B^T \tilde A^{U^{-1}} Z_A^\vee \f$.
  Eigen::MatrixXcd reverse{};
  /// \f$ \|T_{BA}(U^{-1}) - T_{AB}(U)^T\| / \|T_{AB}\| \f$: the reversal identity.
  double reversalResidual{std::numeric_limits<double>::quiet_NaN()};
  double tolerance{0.0};
  /// The groupoid hypothesis \f$ T_{BA} = T_{AB}^{-1} \f$, measured as
  /// \f$ \|T_{BA} T_{AB} - I\| \f$ (square, same rank); false otherwise.
  bool groupoidHolds{false};
  double groupoidResidual{std::numeric_limits<double>::quiet_NaN()};
  /// \f$ T_{AB}^{-T} \f$, the dual transfer \f$ M^\vee = M^{-T} \f$, emitted only
  /// when the groupoid hypothesis holds; empty otherwise.
  Eigen::MatrixXcd dualTransfer{};
};

/// # PencilSchur
///
/// The recursion on the symmetric pencil \f$ \mathcal P(\lambda) = \tilde A -
/// \lambda M \f$ on geometric images, dense below the crossover. Partition
/// coordinates into interface \f$ B \f$ and interior \f$ I \f$.
///
/// * (a) `feshbach`: \f$ F_B(\lambda) = \mathcal P_{BB} - \mathcal P_{BI}\mathcal P_{II}^{-1}
///   \mathcal P_{IB} \f$, \f$ \det\mathcal P = \det\mathcal P_{II}\det F_B \f$;
///   \f$ F_B(\lambda;U)^T = F_B(\lambda;U^{-1}) \f$, symmetric at \f$ U = 1 \f$.
/// * (b) `craigBampton`: the congruence \f$ (T^T\tilde A T,\ T^T M T) \f$.
/// * (c) `restrictToFibers`: retained fibers with images \f$ Z \f$ give
///   \f$ \mathcal G_{\ell+1} = Z^T M Z \f$ and \f$ \hat A_{\ell+1} = Z^T\tilde A Z \f$;
///   off-diagonal blocks \f$ Z_A^T M Z_B \f$ vanish unless the supports of
///   the two images share a top simplex (`supportsShareTopSimplex`).
/// * (d) `transfer`: \f$ T_{AB}(U) = (Z_A^\vee)^T(\tilde A^U)_{AB}Z_B \f$ with
///   \f$ T_{BA}(U^{-1}) = T_{AB}(U)^T \f$ enforced as a runtime assertion, and
///   \f$ M^\vee = M^{-T} \f$ only under the certified groupoid hypothesis.
///
/// Every pairing is the transpose; no conjugation enters.
class PencilSchur {
 public:
  [[nodiscard]] static FeshbachResult feshbach(const Eigen::MatrixXcd &A,
                                               const Eigen::MatrixXcd &M, Complex lambda,
                                               const std::vector<int> &interface,
                                               double rankTolerance = 1e-12);
  [[nodiscard]] static CongruenceResult craigBampton(const Eigen::MatrixXcd &A,
                                                     const Eigen::MatrixXcd &M,
                                                     const Eigen::MatrixXcd &T);
  /// Concatenated fibers \p Z (columns) as one block.
  [[nodiscard]] static FiberRestriction restrictToFibers(const Eigen::MatrixXcd &A,
                                                         const Eigen::MatrixXcd &M,
                                                         const Eigen::MatrixXcd &Z);
  /// Several fibers, each a block of columns, with block offsets recorded.
  [[nodiscard]] static FiberRestriction restrictToFibers(
      const Eigen::MatrixXcd &A, const Eigen::MatrixXcd &M,
      const std::vector<Eigen::MatrixXcd> &fibers);
  /// \f$ Z_A^T M Z_B \f$, one off-diagonal Gram block.
  [[nodiscard]] static Eigen::MatrixXcd gramBlock(const Eigen::MatrixXcd &M,
                                                  const Eigen::MatrixXcd &ZA,
                                                  const Eigen::MatrixXcd &ZB);
  /// Whether two image supports (canonical degree-\p k cell indices) share a
  /// top simplex of \p K — the locality condition under which their Gram
  /// block may be nonzero.
  [[nodiscard]] static bool supportsShareTopSimplex(const cobordism::ChainComplex &K, int k,
                                                    const std::vector<int> &supportA,
                                                    const std::vector<int> &supportB);
  /// The support of an image: the indices of its nonzero rows above
  /// \p threshold times its largest modulus.
  [[nodiscard]] static std::vector<int> support(const Eigen::MatrixXcd &Z,
                                                double threshold = 1e-12);
  /// The transfer between fibers with the reversal identity asserted.
  /// @param AtildeU the dressed pencil operator for \f$ U \f$.
  /// @param AtildeUinv the dressed pencil operator for \f$ U^{-1} \f$ (the dual
  ///   instance).
  /// @param ZA the band's images on \f$ A \f$ for \f$ U \f$.
  /// @param ZAdual the band's images on \f$ A \f$ for \f$ U^{-1} \f$.
  /// @param ZB the band's images on \f$ B \f$ for \f$ U \f$.
  /// @param ZBdual the band's images on \f$ B \f$ for \f$ U^{-1} \f$.
  /// @param tolerance relative tolerance on the reversal identity.
  /// @throws std::runtime_error, by name with the measured residual, when
  ///   \f$ \|T_{BA}(U^{-1}) - T_{AB}(U)^T\| \f$ exceeds \p tolerance times
  ///   \f$ \|T_{AB}\| \f$.
  [[nodiscard]] static TransferResult transfer(const Eigen::MatrixXcd &AtildeU,
                                               const Eigen::MatrixXcd &AtildeUinv,
                                               const Eigen::MatrixXcd &ZA,
                                               const Eigen::MatrixXcd &ZAdual,
                                               const Eigen::MatrixXcd &ZB,
                                               const Eigen::MatrixXcd &ZBdual,
                                               double tolerance = 1e-8);
};

}  // namespace tessera::chainhodge

#endif  // TESSERA_CHAINHODGE_PENCILSCHUR_H
