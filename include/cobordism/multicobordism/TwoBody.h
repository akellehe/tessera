// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

// The two-body cobordism map: its targets, cases and reads.

#ifndef TESSERA_COBORDISM_MULTICOBORDISM_TWO_BODY_H
#define TESSERA_COBORDISM_MULTICOBORDISM_TWO_BODY_H

#include <complex>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "cobordism/multicobordism/InputState.h"

namespace tessera::cobordism::multicobordism {

/// The space a two-body target is scored against. Each name is the space the
/// reading takes the state from.
///
/// `Transfer` is \f$(Z_A^\vee)^T \tilde A_1 Z_B\f$: the whole complex's
/// degree-1 operator read as the coupling block between the two boundary
/// frames. It is \f$2 \times 2\f$ here and factorizes across the boundaries,
/// which is what lets it carry an entangled target.
///
/// `Bulk` is \f$\ker L_1(W-\partial W)\f$, the Laplacian on interior cells
/// with the boundary removed, read through a Choi frame
/// (`geometricOperator`).
///
/// `Whole` is \f$\ker L_1(W)\f$, the boundary included, read through the
/// blocks' markings. Its rank is \f$b_1(W)\f$.
///
/// `Operator` is the low-level transfer between two paired direct-sum frames.
/// A target is meaningful only when expressed in those same direct-sum
/// frames; equal dimensions alone do not identify it with a tensor-product
/// two-qubit operator.
enum class ReadoutMode { Transfer, Bulk, Whole, Operator };

/// How the whole-complex reading pairs the harmonic columns against the input
/// blocks.
///
/// `Periods` integrates each harmonic column over the marked cycles
/// (`Connection::transportedPeriod`). That is how a state is defined on a
/// boundary torus, where the Hodge star is an endomorphism of \f$ H^1 \f$ and
/// a modulus is the period ratio of the holomorphic line it selects. On the
/// three-dimensional bulk the same reading is purely topological: the star
/// there maps 1-forms to 2-forms, so no such line exists, changing the metric
/// moves the harmonic representative by a coboundary, and a coboundary's
/// period around a closed cycle telescopes to zero. A bulk run scored on
/// periods therefore cannot improve.
///
/// `Gram` contracts through the chain metric instead, \f$ f_c^T M_1 Z_a \f$
/// with \f$ f_c \f$ the columns of the block's live frame on the host's
/// edges: the transpose pairing that the harmonic Gram matrix
/// \f$ Z^T M_1 Z \f$ already uses. The metric content is present in both; the
/// period pairing is the contraction that annihilates it.
///
/// The observation matrix has the same shape either way, so the fit, the
/// period-frame normalization and the projective leak downstream are
/// unchanged. `Periods` is the default.
enum class WholePairing { Periods, Gram };

/// The two-body target of the interaction node: \f$ \chi \f$ on the pair of
/// input frames (\f$ r_A\times r_B \f$). `choiDecomposed` selects what the
/// node reports: the state \f$ \mathrm{vec}(T_{AB}) \f$ on the pair space
/// when true, the operator \f$ T_{AB} \f$ when false. The fit residual is the
/// projective Frobenius leak in either reading — vec is linear, so the two
/// coincide — and the readings differ only in what is returned and certified.
struct TwoBodyTarget {
  /// The target \f$ \chi \f$ on the pair of input frames.
  Eigen::MatrixXcd chi{};
  /// Report the Choi-decomposed state rather than the operator.
  bool choiDecomposed{true};
  /// Target for `ReadoutMode::Operator`, expressed in the paired direct-sum
  /// frame returned by `pairedFrameTransferOn`. This is not a
  /// tensor-product two-state vector: equal dimensions do not identify
  /// \f$V_A\oplus V_{A^*}\f$ with \f$V_A\otimes V_{A^*}\f$. Empty means no
  /// paired-frame target was declared, so the operator reading returns the
  /// full leak.
  Eigen::MatrixXcd twoStateVector{};
  /// Optional input coefficients for this read, flattened in input-block
  /// order and then cycle order. Empty uses the coefficients carried by the
  /// live markings. Cases set this so that changing the boundary metric and
  /// the input state is one atomic read of the shared bulk.
  Eigen::VectorXcd inputCoefficients{};
};

/// One input pair and the output the gate should produce for it: a boundary
/// metric, a target, and optionally the input coefficients read by
/// `ReadoutMode::Whole`.
///
/// A bulk fitted to a single pair reproduces that pair and nothing else —
/// one \f$2\times2\f$ transfer against one pair is six real constraints
/// against some eighty free bulk coordinates. Several cases constrain one
/// shared bulk, so a geometry satisfying all of them under the transfer
/// reading is an operator. Every case shares one triangulation, one gluing
/// and one bulk; only the boundary metric and the declared state differ, so
/// moving between cases involves no re-gluing.
struct TwoBodyCase {
  /// Squared lengths on the boundary edges, by endpoint pair. Only the
  /// boundary is listed: the bulk is what is being solved for and is shared.
  std::vector<std::pair<std::pair<std::uint64_t, std::uint64_t>,
                        std::complex<double>>> boundary;
  /// The target \f$ \chi \f$ for this case.
  Eigen::MatrixXcd chi{};
  /// Report the Choi-decomposed state rather than the operator.
  bool choiDecomposed{true};
  /// Target for `ReadoutMode::Operator`, expressed in the paired direct-sum
  /// frame. It is not a tensor-product two-state vector; empty means this
  /// case declares no paired-frame target.
  Eigen::MatrixXcd twoStateVector{};
  /// Optional input coefficients for this case, flattened in input-block
  /// order and then cycle order. Empty reads the live markings'
  /// coefficients.
  Eigen::VectorXcd inputCoefficients{};
};

/// The reading of the bulk between the two attached input frames.
struct TwoBodyRead {
  bool choiDecomposed{true};
  /// \f$ T_{AB} = (Z_A^\vee)^T(\tilde A^U)_{AB}Z_B \f$: with the full frames on
  /// the two attached cell sets (the coupling block of the whole between
  /// them, \f$ |A| \times |B| \f$) when the blocks carry no frame, or in the
  /// blocks' frames (\f$ r_A \times r_B \f$, `BlockFrame`) when both do.
  Eigen::MatrixXcd transfer{};
  /// True when `transfer` was read in the blocks' frames (a derived frame
  /// from `setInputMarking` or a supplied one from `setInputFrame`, on
  /// both), false when in identity frames on the cells.
  bool inFrames{false};
  /// True when both frames were derived live from the blocks' markings;
  /// false with supplied or identity frames.
  bool derivedFrames{false};
  /// \f$ \mathrm{vec}(T_{AB}) \f$, column-major, the Choi-decomposed state.
  Eigen::VectorXcd choiState{};
  /// Singular values of \f$ T_{AB} \f$: the Schmidt spectrum of the state.
  std::vector<double> singularValues{};
  /// Numerical Schmidt rank (singular values above \f$ 10^{-10}\sigma_{\max} \f$);
  /// one is the product (quasi-free) case, two the XY flip-flop's.
  int schmidtRank{0};
  /// The reversal identity residual of the transfer.
  double reversalResidual{std::numeric_limits<double>::quiet_NaN()};
  /// The projective Frobenius leak of the target in the reading.
  double residual{std::numeric_limits<double>::quiet_NaN()};
  /// The fiber residual of each input block carrying an attached fiber: the
  /// leak of its state fiber in its own kernel. On a marked block this is a
  /// diagnostic; the scored residual is in `inputStates`.
  std::vector<double> inputFiberResiduals{};
  /// The state at every marked input block (`readInputState`, block order).
  std::vector<InputStateRead> inputStates{};
  /// The cells the transfer's rows and columns are placed on: the frames'
  /// cells when framed, the attached fiber cells otherwise.
  std::vector<std::vector<std::uint64_t>> cellsA{};
  std::vector<std::vector<std::uint64_t>> cellsB{};
};

}  // namespace tessera::cobordism::multicobordism

#endif  // TESSERA_COBORDISM_MULTICOBORDISM_TWO_BODY_H
