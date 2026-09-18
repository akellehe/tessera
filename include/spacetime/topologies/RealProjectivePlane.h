// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_REALPROJECTIVEPLANE_H
#define TESSERA_REALPROJECTIVEPLANE_H

#include "Topology.h"

// === tessera subsystem ns fwd-decls ===
namespace tessera::spacetime {
class Spacetime;

/// # Real projective plane (minimal 6-vertex triangulation)
///
/// The unique minimal triangulation of the real projective plane
/// \f$ \mathbb{RP}^2 \f$ — the hemi-icosahedron, i.e. the 10 triangles of the
/// icosahedron modulo the antipodal map, on the complete graph
/// \f$ K_6 \f$. f-vector (6, 15, 10), \f$ \chi = 1 \f$, non-orientable
/// (\f$ w_1^2[\mathbb{RP}^2] = 1 \f$).
///
/// Exact, fixed, pre-geometric (coordinate-free); ``build()`` ignores
/// ``numSimplices``.
class RealProjectivePlane : public Topology {
  public:
    RealProjectivePlane() = default;

    /// \f$ \mathbb{RP}^2 \f$ is a closed surface; its top cells are triangles.
    [[nodiscard]] int dimension() const override { return 2; }

    /// Build the 6-vertex \f$ \mathbb{RP}^2 \f$. ``numSimplices`` is ignored.
    void build(Spacetime *spacetime, int numSimplices) override;
};

} // namespace tessera::spacetime

#endif // TESSERA_REALPROJECTIVEPLANE_H
