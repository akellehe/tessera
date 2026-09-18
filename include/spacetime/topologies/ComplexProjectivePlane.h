// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_COMPLEXPROJECTIVEPLANE_H
#define TESSERA_COMPLEXPROJECTIVEPLANE_H

#include "Topology.h"

// === tessera subsystem ns fwd-decls ===
namespace tessera::spacetime {
class Spacetime;

/// # Complex projective plane (minimal 9-vertex triangulation)
///
/// Kühnel's 9-vertex triangulation of the complex projective plane
/// \f$ \mathbb{CP}^2 \f$ — the unique minimal triangulation of any manifold
/// that is neither a sphere nor the boundary of a simplex. A closed,
/// orientable, smooth 4-manifold with
/// f-vector \f$ (9, 36, 84, 90, 36) \f$, Euler characteristic
/// \f$ \chi = 3 \f$, Betti numbers \f$ (1, 0, 1, 0, 1) \f$, and a definite
/// intersection form of rank one — so the signature has absolute value one,
/// \f$ |\sigma| = 1 \f$. Its symmetry group has order 54.
///
/// ## Construction
///
/// The 36 four-simplices are the orbits of twelve base simplices under the
/// order-three permutation \f$ S = (1\,4\,7)(2\,5\,8)(3\,6\,9) \f$ of the nine
/// vertices (each orbit has three members, \f$ 12 \times 3 = 36 \f$). The base
/// list is the one tabulated by Kühnel and Banchoff and reproduced by Schwartz,
/// "Trisecting the 9-vertex complex projective plane". Vertices are relabeled
/// from the literature's \f$ 1\ldots 9 \f$ to tessera's \f$ 0\ldots 8 \f$, under
/// which \f$ S \f$ becomes \f$ (0\,3\,6)(1\,4\,7)(2\,5\,8) \f$.
///
/// ## Orientation
///
/// \f$ \mathbb{CP}^2 \f$ and its orientation reversal
/// \f$ \overline{\mathbb{CP}}^2 \f$ are the same simplicial complex; they
/// differ only in the choice of fundamental class (which generator of
/// \f$ \ker \partial_4 \f$). The orientation-independent invariant is the
/// signature's magnitude \f$ |\sigma| = 1 \f$; its sign follows from how the
/// fundamental class is picked.
///
/// Exact, fixed, pre-geometric (coordinate-free); ``build()`` ignores
/// ``numSimplices``.
class ComplexProjectivePlane : public Topology {
  public:
    ComplexProjectivePlane() = default;

    /// \f$ \mathbb{CP}^2 \f$ is a closed 4-manifold; its top cells are 4-simplices.
    [[nodiscard]] int dimension() const override { return 4; }

    /// Build the 9-vertex \f$ \mathbb{CP}^2 \f$. ``numSimplices`` is ignored.
    void build(Spacetime *spacetime, int numSimplices) override;
};

} // namespace tessera::spacetime

#endif // TESSERA_COMPLEXPROJECTIVEPLANE_H
