// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_OBSERVABLES_SPECTRAL_H
#define TESSERA_OBSERVABLES_SPECTRAL_H

#include <memory>

#include "observables/Observable.h"

// === tessera subsystem ns fwd-decls ===
namespace tessera::spacetime { class Spacetime; }
namespace tessera::observables {
using namespace ::tessera::spacetime;

/// # SpectralGap
///
/// Reference: Lim, "Hodge Laplacians on graphs", arXiv:1507.05379.
///
/// First spectral gap \f$ \lambda_1 - \lambda_0 \f$ of the U(1) connection graph
/// Laplacian (`cobordism::HodgeLaplacian::connectionEigenvalues`) on a
/// triangulation: a scalar wrapper over its ascending eigenvalues. The gap is a
/// gauge-invariant interference signature; on the triangle it falls from
/// \f$ 3 \f$ at zero flux to \f$ 0 \f$ at half a flux quantum
/// (\f$ \Phi = \pi \f$), where the two lowest modes become degenerate. Returns 0
/// when there are fewer than two vertices (no gap).
///
/// This is not the gap of the Hodge Laplacian \f$ L_0 \f$
/// (`HodgeLaplacian::eigenvalues(0)`): that operator is built from
/// \f$ \partial_1 \f$ and the weight alone, so its gap is a different number and
/// carries no flux dependence.
///
/// The U(1) connection lives on each `Edge`, so a `Spacetime` carries everything
/// the measurement needs.
class SpectralGap : public Observable {
  public:
    double compute(const std::shared_ptr<Spacetime> &spacetime) override;
};

/// # HarmonicDimension
///
/// \f$ \dim \ker L^{U(1)} \f$: the number of harmonic zero-modes of the U(1)
/// connection graph Laplacian (the count of
/// `cobordism::HodgeLaplacian::connectionHarmonics`). At zero flux this equals
/// the number of connected components \f$ b_0 \f$. A nonzero U(1) flux lifts the
/// zero-mode (magnetic frustration), so the count drops below the
/// flux-independent topological count from `ChainComplex`. Returns 0 for the
/// empty complex.
///
/// This is not \f$ \dim \ker L_0 \f$ of the Hodge Laplacian: the row sums of
/// \f$ L_0 = \partial_1 W_1^{-1}\partial_1^{\dagger} \f$ vanish identically, so
/// the constant function is harmonic at any weights and
/// \f$ \dim \ker L_0 = b_0 \f$ always, duplicating
/// `ChainComplex::bettiNumbers` and carrying no flux content.
class HarmonicDimension : public Observable {
  public:
    double compute(const std::shared_ptr<Spacetime> &spacetime) override;
};

}  // namespace tessera::observables

#endif  // TESSERA_OBSERVABLES_SPECTRAL_H
