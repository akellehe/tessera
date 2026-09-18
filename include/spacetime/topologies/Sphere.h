// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_SPHERE_H
#define TESSERA_SPHERE_H

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

/// # Spherical topology
///
/// Spatial slices are \f$(d\!-\!1)\f$-spheres \f$ S^{d-1} \f$, giving a
/// spacetime manifold \f$ \mathcal{M} \cong S^{d-1} \times S^1 \f$. This is
/// the topology of most 4D causal dynamical triangulation (CDT) simulations,
/// whose spatial slices are three-spheres \f$ S^3 \f$.
///
/// The Euclidean de Sitter solution (the round four-sphere \f$ S^4 \f$)
/// decomposes into \f$ S^3 \f$ spatial slices, so this is the topology used to
/// study de Sitter quantum gravity. The volume profile of each slice follows
///
/// \f[
///   V_3(t) \propto \cos^3\!\left(\frac{\pi\, t}{T}\right)
/// \f]
///
/// for the continuum \f$ S^4 \f$ geometry.
///
/// The build alternates coning direction between layers to close the manifold.
///
/// Reference: Ambjorn, Jurkiewicz & Loll, "Reconstructing the Universe",
/// arXiv:hep-th/0505154.
///
class Sphere : public Topology {
  public:
    /// Build a spherical triangulation by coning in alternating \f$ \pm t \f$ directions.
    ///
    /// @param spacetime The spacetime to populate
    /// @param numSimplices Target number of top-dimensional simplices
    void build(Spacetime *spacetime, int numSimplices) override;
};

} // namespace tessera::spacetime

#endif //TESSERA_SPHERE_H
