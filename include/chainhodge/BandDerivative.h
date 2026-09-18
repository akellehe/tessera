// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_CHAINHODGE_BANDDERIVATIVE_H
#define TESSERA_CHAINHODGE_BANDDERIVATIVE_H

#include <complex>
#include <cstddef>
#include <vector>

#include <Eigen/Core>

#include "chainhodge/CovariantChainHodge.h"
#include "chainhodge/RieszBand.h"

namespace tessera::chainhodge {

/// # BandDerivative
///
/// Analytic derivatives of a Riesz band's geometric images with respect to
/// one squared edge length or one link phase. The band's projector is
/// the contour quadrature \f$ P=\sum_j w_j R(\zeta_j) \f$ of the pencil
/// resolvent \f$ R(\zeta)=(\zeta I-h)^{-1} \f$ on chains, its right frame
/// \f$ \Phi \f$ spans \f$ \operatorname{Ran}P \f$, and its images are
/// \f$ Z=G^U\Phi \f$. Holding the spanning frame fixed at the evaluation point
/// (\f$ P\Phi=\Phi \f$ there), the first-order motion of the image span is
/// \f[
///   dZ = G^U\bigl(-\,dM^U\,Z + dP\,\Phi\bigr),\qquad
///   dP\,\Phi=\sum_j w_j\,R(\zeta_j)\,dh\,R(\zeta_j)\,\Phi ,
/// \f]
/// with \f$ dM^U \f$ the dressed sparse metric derivative and \f$ dh \f$ the
/// covariant operator derivative (`CovariantChainHodge`). Everything is
/// holomorphic in the squared lengths and in the phases (no conjugation
/// enters), so a real objective's partial derivatives along the real and
/// imaginary parts of a coordinate are \f$ 2\operatorname{Re} \f$ and
/// \f$ -2\operatorname{Im} \f$ of the same holomorphic sensitivity. The
/// resolvent frames \f$ R(\zeta_j)\Phi \f$ are computed once per band and
/// reused across coordinates. Dense, below the crossover.
class BandDerivative {
 public:
  /// \f$ R(\zeta_j)\Phi \f$ for every contour node, computed once.
  struct ResolventFrames {
    int degree{0};
    Contour contour{};
    Eigen::MatrixXcd frame{};              // Phi (n x r)
    std::vector<Eigen::MatrixXcd> applied{};  // R(zeta_j) Phi, one per node
  };

  [[nodiscard]] static ResolventFrames resolventFrames(const CovariantChainHodge &cov, int k,
                                                       const Contour &contour,
                                                       const Eigen::MatrixXcd &frame);

  /// \f$ dP\,\Phi = \sum_j w_j R(\zeta_j)\,dh\,R(\zeta_j)\Phi \f$ for a given
  /// operator derivative \p dh (dense \f$ n\times n \f$).
  [[nodiscard]] static Eigen::MatrixXcd projectorDerivativeApplied(const CovariantChainHodge &cov,
                                                                   const ResolventFrames &frames,
                                                                   const Eigen::MatrixXcd &dh);

  /// \f$ dZ = G^U(-\,dM\,Z + dP\,\Phi) \f$ for a given pair (\p dM sparse, \p dh dense).
  [[nodiscard]] static Eigen::MatrixXcd imagesDerivative(const CovariantChainHodge &cov,
                                                         const ResolventFrames &frames,
                                                         const Eigen::MatrixXcd &images,
                                                         const SparseMatrix &dM,
                                                         const Eigen::MatrixXcd &dh);

  /// \f$ dZ/ds_e \f$ for the edge at canonical index \p edgeIndex.
  [[nodiscard]] static Eigen::MatrixXcd imagesLengthDerivative(const CovariantChainHodge &cov,
                                                               const ResolventFrames &frames,
                                                               const Eigen::MatrixXcd &images,
                                                               std::size_t edgeIndex);
  /// \f$ dZ/d\varphi_e \f$ for the multiplicative link variation
  /// \f$ U_e=e^{i\varphi_e} \f$ of the edge at canonical index \p edgeIndex.
  [[nodiscard]] static Eigen::MatrixXcd imagesPhaseDerivative(const CovariantChainHodge &cov,
                                                              const ResolventFrames &frames,
                                                              const Eigen::MatrixXcd &images,
                                                              std::size_t edgeIndex);

  /// \f$ d\tilde A^U = dh\,M^U + h\,dM^U \f$ for the pencil operator
  /// \f$ \tilde A^U = h\,M^U \f$ (the covariant operator is
  /// \f$ h=\tilde A^U (M^U)^{-1} \f$), dense, for one edge's squared length.
  [[nodiscard]] static Eigen::MatrixXcd pencilOperatorLengthDerivative(const CovariantChainHodge &cov, int k,
                                                                       std::size_t edgeIndex);
  /// The same for the link phase of one edge.
  [[nodiscard]] static Eigen::MatrixXcd pencilOperatorPhaseDerivative(const CovariantChainHodge &cov, int k,
                                                                      std::size_t edgeIndex);
};

}  // namespace tessera::chainhodge

#endif  // TESSERA_CHAINHODGE_BANDDERIVATIVE_H
