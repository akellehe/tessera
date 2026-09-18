// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_SIMPLEXBOUNDARYSPHERE_H
#define TESSERA_SIMPLEXBOUNDARYSPHERE_H

#include "Topology.h"

// === tessera subsystem ns fwd-decls ===
namespace tessera::spacetime {
class Spacetime;

/// # Simplex-boundary sphere
///
/// The minimal triangulation of the n-sphere, \f$ S^n = \partial\Delta^{n+1}
/// \f$: \f$ n+2 \f$ vertices with every \f$ (n+1) \f$-subset a top n-simplex.
/// f-vector \f$ \binom{n+2}{k+1} \f$ for \f$ k = 0..n \f$
/// (e.g. \f$ S^4 \f$: (6, 15, 20, 15, 6)); \f$ \chi = 1 + (-1)^n \f$.
///
/// The exact, fixed minimal triangulation used by the cobordism fixtures,
/// distinct from :class:`Sphere`, which grows a CDT-style
/// \f$ S^{d-1}\times S^1 \f$ initial condition of a requested size. Its
/// ``build()`` ignores ``numSimplices`` and is pre-geometric (coordinate-free
/// vertices; see ``Topology::buildExplicit``).
class SimplexBoundarySphere : public Topology {
  public:
    /// @param n Dimension of the sphere (n >= 1).
    explicit SimplexBoundarySphere(int n) : n_(n) {}

    [[nodiscard]] int n() const noexcept { return n_; }

    /// \f$ S^n \f$ is an n-manifold; its top cells are n-simplices.
    [[nodiscard]] int dimension() const override { return n_; }

    /// Build \f$ S^n = \partial\Delta^{n+1} \f$. ``numSimplices`` is ignored
    /// (the triangulation is fixed).
    void build(Spacetime *spacetime, int numSimplices) override;

  private:
    int n_;
};

} // namespace tessera::spacetime

#endif // TESSERA_SIMPLEXBOUNDARYSPHERE_H
