// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

// Analytic gradients of the fiber-mode residuals.

#ifndef TESSERA_COBORDISM_MULTICOBORDISM_GRADIENTS_H
#define TESSERA_COBORDISM_MULTICOBORDISM_GRADIENTS_H


#include <Eigen/Core>

namespace tessera::cobordism::multicobordism {

/// A gradient over the live complex's edges in `EdgeList` order. Each entry
/// packs \f$ (\partial/\partial\operatorname{Re}, \partial/\partial\operatorname{Im}) \f$
/// of the real residual as a complex number, the convention `runStage2`
/// descends.
struct ResidualGradient {
  /// Derivative with respect to each edge's squared length.
  Eigen::VectorXcd lengths;
  /// Derivative with respect to each link phase; empty unless the residual
  /// varies with the links.
  Eigen::VectorXcd phases;
};

}  // namespace tessera::cobordism::multicobordism

#endif  // TESSERA_COBORDISM_MULTICOBORDISM_GRADIENTS_H
