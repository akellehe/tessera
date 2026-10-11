// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_SOLIDSIMPLEX_H
#define TESSERA_SOLIDSIMPLEX_H

#include <memory>
#include "Topology.h"

// === tessera subsystem ns fwd-decls ===
namespace tessera::spacetime {
class Spacetime;

/// # Solid simplex (closed n-ball)
///
/// The solid simplex \f$ \Delta^n \f$: a single top n-simplex on
/// \f$ n+1 \f$ vertices with all its faces — a triangulated closed n-ball
/// whose boundary is \f$ S^{n-1} = \partial\Delta^n \f$.
/// f-vector \f$ \binom{n+1}{k+1} \f$ for \f$ k = 0..n \f$; \f$ \chi = 1 \f$
/// (contractible). \f$ \Delta^4 \f$ is the smallest cobordism filling
/// \f$ S^3 \to \emptyset \f$.
///
/// Exact, fixed, pre-geometric (coordinate-free); ``build()`` ignores
/// ``numSimplices``.
class SolidSimplex : public Topology {
  public:
    /// @param n Dimension of the simplex (n >= 1).
    explicit SolidSimplex(int n) : n_(n) {}

    /// A built spacetime of the solid \p dimension-simplex: a single-simplex
    /// host with a uniform metric (\f$ |\ell^2| = 1 \f$; balanced wiring gives
    /// \f$ \ell=\sqrt{1/2}(1+i) \f$),
    /// Lorentzian signature, and the causal dynamical triangulations (CDT) type
    /// and preferred foliation. This is the seed every host grows from;
    /// `ProtonSynthesis::buildMinimalSeed` is this at dimension 4.
    ///
    /// Reference: Ambjorn, Jurkiewicz and Loll, "Dynamically Triangulating
    /// Lorentzian Quantum Gravity", arXiv:hep-th/0105267.
    /// @throws std::invalid_argument for dimension below 1.
    [[nodiscard]] static std::shared_ptr<Spacetime> spacetime(int dimension,
                                                               bool balancedEdges = false);

    [[nodiscard]] int n() const noexcept { return n_; }

    /// \f$ \Delta^n \f$ is an n-ball (n-manifold with boundary); its single top
    /// cell is an n-simplex.
    [[nodiscard]] int dimension() const override { return n_; }

    /// Build the solid n-simplex. ``numSimplices`` is ignored.
    void build(Spacetime *spacetime, int numSimplices) override;

  private:
    int n_;
};

} // namespace tessera::spacetime

#endif // TESSERA_SOLIDSIMPLEX_H
