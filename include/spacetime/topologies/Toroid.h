// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_TOROID_H
#define TESSERA_TOROID_H

#include "Topology.h"

// === tessera subsystem ns fwd-decls ===
namespace tessera::graph {}
namespace tessera::mesh {}
namespace tessera::observables {}
namespace tessera::quantum {}
namespace tessera::simulations {}
namespace tessera::spacetime {
using namespace ::tessera::mesh;
using namespace ::tessera::graph;
using namespace ::tessera::observables;
using namespace ::tessera::simulations;
using namespace ::tessera::quantum;
class Spacetime;

/// # Toroidal topology
///
/// Spatial slices are \f$(d\!-\!1)\f$-tori \f$ T^{d-1} \f$, giving a spacetime
/// manifold \f$ \mathcal{M} \cong T^{d-1} \times S^1 \f$ with periodic boundary
/// conditions in both space and time. This is the default topology for causal
/// dynamical triangulation (CDT) simulations and the most common one in the
/// literature.
///
/// Periodic time means the last time slice \f$ t = T \f$ is identified with the
/// first \f$ t = 0 \f$, so the triangulation has no temporal boundaries.
///
/// The build stacks time slabs, each the staircase triangulation of
/// \f$ S^{d-1} \times [t, t+1] \f$ over a layer of \f$ d+1 \f$ vertices.
///
/// Reference: Ambjorn, Goerlich, Jurkiewicz & Loll, arXiv:1203.3591.
///
class Toroid : public Topology {
  public:
    /// Build a toroidal triangulation with multiple time layers, one unit of
    /// coordinate time each.
    ///
    /// @param spacetime The spacetime to populate
    /// @param numSimplices Target number of top-dimensional simplices
    void build(Spacetime *spacetime, int numSimplices) override;
};

} // namespace tessera::spacetime

#endif //TESSERA_TOROID_H
