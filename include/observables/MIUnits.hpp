// Copyright (c) 2026 Twin Vector Labs LLC. All rights reserved.
//
// Shared mutual-information normalisation constant, used by every consumer of
// the mutual-information-to-length map ℓ = −log(I / I_max):
//
//   • InteractionSimulation (Pachner cells, Regge action)
//   • WeightedSparseGraph::fromSpacetimeSkeleton (heat-kernel weights)
//   • MutualInformationProfile::buildSpacetime (edge lengths)

#pragma once

#include <cmath>

// === tessera subsystem ns fwd-decls ===
namespace tessera::graph {}
namespace tessera::mesh {}
namespace tessera::quantum {}
namespace tessera::simulations {}
namespace tessera::spacetime {}
namespace tessera::observables {
using namespace ::tessera::mesh;
using namespace ::tessera::graph;
using namespace ::tessera::spacetime;
using namespace ::tessera::simulations;
using namespace ::tessera::quantum;

/// Maximum mutual information of a maximally-entangled qubit pair,
/// \f$ I_{\max} = 2\log 2 \f$. Used as the normalisation in
/// ℓ = −log(I / kIMax). For a qudit basis of dimension \f$ d \f$ the
/// corresponding value is \f$ 2\log d \f$.
inline const double kIMax = 2.0 * std::log(2.0);

} // namespace tessera::observables
