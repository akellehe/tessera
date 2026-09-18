// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_COBORDISM_COMBINATORIALDIMENSION_H
#define TESSERA_COBORDISM_COMBINATORIALDIMENSION_H

#include <memory>

#include "observables/Observable.h"

// === tessera subsystem ns fwd-decls ===
namespace tessera::spacetime { class Spacetime; }
namespace tessera::cobordism {
using namespace ::tessera::observables;
using namespace ::tessera::spacetime;

/// # CombinatorialDimension
///
/// Observable: the combinatorial dimension of a triangulation — the largest
/// \f$ k \f$ for which a \f$ k \f$-simplex is present (the maximum simplex
/// vertex count minus one), or \f$ -1 \f$ for the empty complex.
///
/// An exact integer fixed by the simplex set alone, equal to \f$ n \f$ for a
/// piecewise-linear \f$ n \f$-manifold triangulation. It is not the spectral
/// dimension, the Hausdorff dimension, or the metric's declared dimension: a
/// hand-assembled \f$ S^2 \f$ can live inside a 4D-signature ``Spacetime``.
/// Dimension-dependent logic keys off this integer.
///
/// Returns the dimension as a double, per the :class:`Observable` interface.
class CombinatorialDimension : public Observable {
  public:
    double compute(const std::shared_ptr<Spacetime> &spacetime) override;
};

}  // namespace tessera::cobordism

#endif  // TESSERA_COBORDISM_COMBINATORIALDIMENSION_H
