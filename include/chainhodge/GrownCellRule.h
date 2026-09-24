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
  /// The squared edge lengths, the quadratic form of \f$ g \f$ on the edge
  /// vectors: \f$ z_{0i} = g_{ii} \f$,
  /// \f$ z_{ij} = (e_j-e_i)^{\mathsf T}g(e_j-e_i) = g_{ii} + g_{jj} - g_{ij} - g_{ji} \f$,
  /// in local edge order; empty when the scale is undetermined.
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

/// The squared lengths of a grown \f$ d \f$-simplex read from the inherited
/// pairing \f$ \mathfrak g \f$ between its \f$ d+1 \f$ response vertices,
/// with the pairing in the role of \f$ |T|\,\Gamma \f$, so that
/// \f$ C\Gamma = \mathfrak g/((d+1)(d+2)) \f$.
struct GrownCellInversion {
  /// The simplex dimension \f$ d \f$ (the pairing is \f$ (d+1)\times(d+1) \f$).
  int dimension{0};
  /// \f$ C\Gamma = \mathfrak g/((d+1)(d+2)) \f$.
  Eigen::MatrixXcd scaledGradientGram{};
  /// \f$ g/C \f$, the inverse of the block of \f$ C\Gamma \f$ on
  /// \f$ v_1,\dots,v_d \f$.
  Eigen::MatrixXcd scaledMetric{};
  /// True at \f$ d = 3 \f$; false at \f$ d = 2 \f$, where the scale is
  /// undetermined.
  bool scaleDetermined{false};
  /// \f$ C = 14400/\det(g/C) \f$ at \f$ d = 3 \f$; quiet NaN otherwise.
  std::complex<double> scale{};
  /// \f$ |T| = (d+1)(d+2)\,C \f$; quiet NaN when the scale is undetermined.
  std::complex<double> volume{};
  /// The squared lengths in local edge order \f$ (0,1),(0,2),\dots \f$, as in
  /// `WhitneyLengthInversion::squaredLengths`; empty when the scale is
  /// undetermined.
  std::vector<std::complex<double>> squaredLengths{};
  /// The squared lengths divided by \f$ C \f$, defined at every dimension.
  std::vector<std::complex<double>> scaledSquaredLengths{};
  /// \f$ \lVert\mathfrak g\,\mathbf 1\rVert_2/\lVert\mathfrak g\rVert_F \f$:
  /// barycentric gradients sum to zero, so this vanishes for the pairing of a
  /// simplex and measures how far the grown cell is from a simplicial geometry.
  double rowSumDefect{0.0};
  /// \f$ \lVert\mathfrak g - \mathfrak g^{\mathsf T}\rVert_F/\lVert\mathfrak g\rVert_F \f$.
  double asymmetry{0.0};
  /// \f$ \mathfrak g_{vw}^2/(\mathfrak g_{vv}\mathfrak g_{ww}) \f$: the part
  /// of the pairing that a change of fiber frames, which multiplies
  /// \f$ \mathfrak g_{vw} \f$ by \f$ \det g_v\det g_w \f$, does not move.
  Eigen::MatrixXcd frameInvariantRatios{};
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

  /// The grown-cell identification: each response vertex's fiber plays the
  /// role of the barycentric gradient \f$ d\lambda_v \f$, so the inherited
  /// pairing \p pairing between the \f$ d+1 \f$ vertices of a grown simplex is
  /// read as \f$ |T|\,\Gamma \f$ and steps (ii)-(iv) of the length rule follow
  /// with no fit. At level zero the pairing of the exact chains
  /// \f$ \delta_0 e_v \f$ on one simplex is exactly \f$ |T|\,\Gamma \f$ and the
  /// squared lengths are returned. Throws `std::invalid_argument` unless the
  /// pairing is \f$ 3\times3 \f$ or \f$ 4\times4 \f$, and when the block on
  /// \f$ v_1,\dots,v_d \f$ is singular.
  [[nodiscard]] static GrownCellInversion invertVertexPairing(const Eigen::MatrixXcd &pairing);

  /// The inherited pairing on the determinant line,
  /// \f$ \mathfrak g_{vw} = \det\bigl(Y_v^{\mathsf T}\,Z_w\bigr) \f$, where
  /// \p frames holds the fiber frames \f$ Y_v \f$ (chains, \f$ n\times r \f$)
  /// and \p images the matching \f$ Z_w = G_1^U Y_w \f$. Entries between
  /// fibers of different rank are undefined and returned as quiet NaN.
  [[nodiscard]] static Eigen::MatrixXcd determinantPairing(
      const std::vector<Eigen::MatrixXcd> &frames, const std::vector<Eigen::MatrixXcd> &images);

  /// The gauge-invariant inherited pairing of the grown-cell rule, the
  /// face-anchor pattern of WP §10: the dual-connection frame of \f$ v \f$
  /// paired with the image of the frame of \f$ w \f$ on the determinant line,
  /// with the transport divided out,
  /// \f[
  ///   \hat{\mathfrak g}_{vv} = \det\bigl((Y^\vee_v)^{\mathsf T}G_1^UY_v\bigr),\qquad
  ///   \hat{\mathfrak g}_{vw} = \frac{\det\bigl((Y^\vee_v)^{\mathsf T}G_1^UY_w\bigr)}
  ///                                   {U_{vw}}\quad(v\ne w),
  /// \f]
  /// \p dualFrames holding \f$ Y^\vee_v \f$, \p images holding
  /// \f$ G_1^U Y_w \f$ and \p connection the matrix of
  /// \f$ U_{vw} = \det M_{vw} \f$ (its diagonal is not read). A microscopic
  /// gauge transformation multiplies the numerator and \f$ U_{vw} \f$ by the
  /// same factor. Entries between fibers of different rank are quiet NaN.
  [[nodiscard]] static Eigen::MatrixXcd gaugeInvariantPairing(
      const std::vector<Eigen::MatrixXcd> &dualFrames,
      const std::vector<Eigen::MatrixXcd> &images, const Eigen::MatrixXcd &connection);

  /// The dual-connection frame rescaled so that
  /// \f$ \det((Y^\vee)^{\mathsf T}Y) = 1 \f$: the fiber's frame normalized to
  /// determinant one against its dual-connection partner. The first column is
  /// divided by the determinant, so no root is taken. With it, a change of
  /// frame \f$ Y\mapsto Yg \f$ forces \f$ Y^\vee\mapsto Y^\vee h \f$ with
  /// \f$ \det h\,\det g = 1 \f$, under which `gaugeInvariantPairing` is
  /// invariant. Throws `std::invalid_argument` when the pairing is singular or
  /// the shapes differ.
  [[nodiscard]] static Eigen::MatrixXcd normalizeDualFrame(const Eigen::MatrixXcd &frame,
                                                           const Eigen::MatrixXcd &dualFrame);

  /// \f$ U_{vw} = \det M_{vw} \f$ for a square transport block. Throws
  /// `std::invalid_argument` for a non-square or empty block: only common-rank
  /// links carry a connection.
  [[nodiscard]] static std::complex<double> transportConnection(const Eigen::MatrixXcd &transport);
};

}  // namespace tessera::chainhodge

#endif  // TESSERA_CHAINHODGE_GROWNCELLRULE_H
