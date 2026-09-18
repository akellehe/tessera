// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_CYLINDER_H
#define TESSERA_CYLINDER_H

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

/// # Cylindrical topology
///
/// Spatial slices have a closed topology \f$ \Sigma \f$ but time is not
/// periodic: the manifold is \f$ \Sigma \times [0, T] \f$, with open temporal
/// boundaries at \f$ t = 0 \f$ and \f$ t = T \f$. Used for fixed initial and
/// final spatial slices, e.g. the transition amplitude between two spatial
/// geometries.
///
/// The build creates layers by coning in the forward time direction only, so
/// time increases monotonically.
///
class Cylinder : public Topology {
  public:
    /// Build a cylindrical triangulation with open time boundaries.
    ///
    /// Each layer grows by coning exterior facets forward in time only.
    /// The first and last time slices are boundary slices.
    ///
    /// @param spacetime The spacetime to populate
    /// @param numSimplices Target number of top-dimensional simplices
    void build(Spacetime *spacetime, int numSimplices) override;
};

} // namespace tessera::spacetime

#endif //TESSERA_CYLINDER_H
