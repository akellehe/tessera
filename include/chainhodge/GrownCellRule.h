// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_CHAINHODGE_GROWNCELLRULE_H
#define TESSERA_CHAINHODGE_GROWNCELLRULE_H

#include <complex>
#include <vector>

#include <Eigen/Core>

namespace tessera::chainhodge {

/// The result of inverting one degree-1 Whitney mass block of a
/// \f$ d \f$-simplex \f$ T = [v_0,\dots,v_d] \f$.
///
/// Local orders: vertices \f$ 0,\dots,d \f$; edges \f$ (0,1),(0,2),\dots,(d-1,d) \f$
/// (lexicographic), the order `WhitneyMass::topSimplexBlocks` uses for its
/// degree-1 rows.
struct WhitneyLengthInversion {
  /// The simplex dimension \f$ d \f$, read from the block size
  /// \f$ d(d+1)/2 \f$.
  int dimension{0};
  /// The least-squares solution \f$ C\Gamma \f$, a symmetric
  /// \f$ (d+1)\times(d+1) \f$ matrix: the gradient Gram matrix
  /// \f$ \Gamma_{ab} = \langle d\lambda_a, d\lambda_b\rangle \f$ times
  /// \f$ C = |T|/((d+1)(d+2)) \f$.
  Eigen::MatrixXcd scaledGradientGram{};
  /// \f$ g/C \f$: the inverse of the block of \f$ C\Gamma \f$ on the vertices
  /// \f$ 1,\dots,d \f$, the metric in the edge basis \f$ e_i = v_i - v_0 \f$
  /// divided by the scale \f$ C \f$.
  Eigen::MatrixXcd scaledMetric{};
  /// True when the scale is fixed by the block, which is the case at
  /// \f$ d = 3 \f$ only. At \f$ d = 2 \f$ the exponent \f$ d-2 \f$ vanishes and
  /// the scale is undetermined (the conformal invariance of 1-form
  /// \f$ L^2 \f$ norms in two dimensions).
  bool scaleDetermined{false};
  /// \f$ C = 14400/\det(g/C) \f$ at \f$ d = 3 \f$; quiet NaN otherwise.
  std::complex<double> scale{};
  /// \f$ |T| = (d+1)(d+2)\,C \f$; quiet NaN when the scale is undetermined.
  std::complex<double> volume{};
  /// The squared edge lengths \f$ z_{0i} = g_{ii} \f$,
  /// \f$ z_{ij} = g_{ii} + g_{jj} - 2g_{ij} \f$ in local edge order; empty when
  /// the scale is undetermined.
  std::vector<std::complex<double>> squaredLengths{};
  /// The same formulas applied to \f$ g/C \f$: the squared lengths divided by
  /// \f$ C \f$, defined at every dimension.
  std::vector<std::complex<double>> scaledSquaredLengths{};
  /// \f$ \| A\,x - b \|_2 \f$ of the least-squares fit of the
  /// \f$ d(d+1)(d^2+d+2)/8 \f$ upper-triangular block entries \f$ b \f$ by the
  /// Whitney form \f$ A\,x \f$ in the unknowns \f$ x = C\Gamma \f$: zero for a
  /// block of Whitney form, and otherwise the distance of the block from the
  /// family of Whitney blocks.
  double residual{0.0};
  /// `residual` divided by \f$ \|b\|_2 \f$.
  double relativeResidual{0.0};
  /// \f$ \|B - B^{\mathsf T}\|_F / \|B\|_F \f$ of the input block \f$ B \f$,
  /// which the fit (over the upper triangle) does not see.
  double asymmetry{0.0};
};

/// # GrownCellRule
///
/// The squared lengths and the connection of a cell grown among response
/// vertices (whitepaper v17, Section 3 and the Section 15 recursion box).
///
/// **Lengths.** On one top simplex \f$ T \f$ of dimension \f$ d \f$, with
/// barycentric coordinates \f$ \lambda_a \f$, the degree-1 Whitney mass block
/// of the Whitney forms \f$ w_{[ij]} = \lambda_i d\lambda_j - \lambda_j d\lambda_i \f$
/// is linear in the gradient Gram matrix \f$ \Gamma \f$:
/// \f[
///   (M_1^{(T)})_{[ij],[kl]} = C\bigl[(1+\delta_{ik})\Gamma_{jl}
///     - (1+\delta_{il})\Gamma_{jk} - (1+\delta_{jk})\Gamma_{il}
///     + (1+\delta_{jl})\Gamma_{ik}\bigr],\qquad C = \frac{|T|}{(d+1)(d+2)} .
/// \f]
/// `invertWhitneyBlock` (i) solves this linear system for the
/// \f$ (d+1)(d+2)/2 \f$ entries of \f$ C\Gamma \f$ by least squares over the
/// upper-triangular block entries, (ii) inverts the block of \f$ C\Gamma \f$ on
/// the vertices other than \f$ v_0 \f$ to obtain \f$ g/C \f$, (iii) fixes the
/// scale from \f$ |T| = \sqrt{\det g}/d! \f$, which gives
/// \f$ C^{d-2} = (d!(d+1)(d+2))^2/\det(g/C) \f$ and in three dimensions
/// \f$ C = 14400/\det(g/C) \f$ with no square root and no branch, and (iv) reads
/// the squared lengths off the metric. On a block of Whitney form the scale
/// returned is the one the block carries, whatever branch of \f$ |T| \f$ it was
/// assembled on.
///
/// **Phase.** The connection on a grown edge between response vertices
/// \f$ v \f$ and \f$ w \f$ is the full complex determinant of the transport
/// between their fibers, \f$ U_{vw} = \det M_{vw} \f$: the connection of the
/// fibers' determinant line. For rank-one fibers with transport the
/// multiplication by \f$ U_e \f$ it returns \f$ U_e \f$.
class GrownCellRule {
 public:
  /// The degree-1 Whitney block of a \f$ d \f$-simplex from a symmetric
  /// \f$ (d+1)\times(d+1) \f$ matrix \f$ C\Gamma \f$ (the forward map of the
  /// length inversion).
  [[nodiscard]] static Eigen::MatrixXcd whitneyBlock(const Eigen::MatrixXcd &scaledGradientGram);

  /// Inverts a \f$ d(d+1)/2 \f$-square degree-1 block for the squared lengths.
  /// Throws `std::invalid_argument` for a block that is not the size of a
  /// \f$ d \f$-simplex's edge set with \f$ d = 2 \f$ or \f$ d = 3 \f$ (for
  /// \f$ d\ge4 \f$ the scale is a \f$ (d-2) \f$-th root, which the rule does not
  /// supply), and when the block of \f$ C\Gamma \f$ on \f$ v_1,\dots,v_d \f$ is
  /// singular.
  [[nodiscard]] static WhitneyLengthInversion invertWhitneyBlock(const Eigen::MatrixXcd &block);

  /// \f$ U_{vw} = \det M_{vw} \f$ for a square transport block. Throws
  /// `std::invalid_argument` for a non-square or empty block: only common-rank
  /// links carry a connection.
  [[nodiscard]] static std::complex<double> transportConnection(const Eigen::MatrixXcd &transport);
};

}  // namespace tessera::chainhodge

#endif  // TESSERA_CHAINHODGE_GROWNCELLRULE_H
