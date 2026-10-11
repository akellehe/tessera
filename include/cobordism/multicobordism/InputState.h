// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

// The state read at an input block.

#ifndef TESSERA_COBORDISM_MULTICOBORDISM_INPUT_STATE_H
#define TESSERA_COBORDISM_MULTICOBORDISM_INPUT_STATE_H

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>

#include <Eigen/Core>

namespace tessera::cobordism::multicobordism {

/// The state at a marked block: the coefficients of the whole complex's zero
/// mode in the block's live frame, alongside the block's input coefficients
/// and its residual.
///
/// `coefficients` are the transported periods, over the marking's cycles from
/// `baseVertex`, of the least-squares combination of the whole's zero mode
/// that fits the block's target edge values — the input coefficients written
/// through the live frame — on the block's edges. That combination is a
/// kernel vector of the whole, so its periods are its coordinates in the
/// frame. `residual` is the fit's leak, i.e. the output state read at the
/// block; `rU` instead scores a marked block by `ownStateResidualOn` at
/// `weight`. Under a pure gauge the target's base-point factor cancels the
/// transport's, so the pair is gauge invariant.
///
/// `harmonicRank` is the whole's zero-mode rank and `frameRank` the own
/// kernel's. An obstructed read — no frame, no zero mode, or a target edge
/// absent from the whole — names the obstruction and reads the full leak 1.0.
struct InputStateRead {
  /// Index of the input block.
  std::size_t block{0};
  /// The block's declared input coefficients.
  Eigen::VectorXcd input;
  /// The whole complex's zero mode in the block's live frame.
  Eigen::VectorXcd coefficients;
  /// Leak of the fit; 1.0 when obstructed.
  double residual{std::numeric_limits<double>::quiet_NaN()};
  /// Weight this block's residual carries in `rU`.
  double weight{1.0};
  /// Rank of the whole complex's zero mode.
  int harmonicRank{0};
  /// Rank of the block's own kernel.
  int frameRank{0};
  /// Base point the periods were transported from.
  std::uint64_t baseVertex{0};
  /// Empty when the read succeeded; otherwise why it could not be made.
  std::string obstruction;
};

}  // namespace tessera::cobordism::multicobordism

#endif  // TESSERA_COBORDISM_MULTICOBORDISM_INPUT_STATE_H
