// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_TOPOLOGY_H
#define TESSERA_TOPOLOGY_H

#include <cstddef>
#include <cstdint>
#include <vector>

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

/// # Spatial topology
///
/// Abstract base class for the topology of spatial slices in a foliated
/// spacetime. In causal dynamical triangulation (CDT) the spacetime manifold
/// \f$ \mathcal{M} \f$ has the product structure
///
/// \f[
///   \mathcal{M} \cong \Sigma \times I
/// \f]
///
/// where \f$ \Sigma \f$ is a compact spatial manifold and \f$ I \f$ is either
/// an interval \f$[0, T]\f$ (cylinder) or a circle \f$ S^1 \f$ (periodic time).
/// The topology of \f$ \Sigma \f$ is fixed throughout the simulation and
/// determines the boundary conditions and initial triangulation.
///
/// Subclasses implement `build()` to construct an initial triangulation with
/// their spatial topology. The CDT topologies grow one by coning — a seed
/// \f$ d \f$-simplex per time layer whose exterior facets are iteratively coned
/// to new vertices — or, for :class:`Toroid`, by stacking staircase-triangulated
/// time slabs. The fixed-triangulation fixtures instead list their top simplices
/// explicitly (``buildExplicit``).
///
/// Reference: Ambjorn, Jurkiewicz & Loll, arXiv:hep-th/0105267.
///
class Topology {
  public:
    virtual ~Topology();

    /// Build an initial triangulation with the given spatial topology.
    ///
    /// Creates \f$ d \f$-simplices across multiple time layers using iterated
    /// coning from seed simplices. The resulting simplicial complex
    /// \f$ \mathcal{K} \f$ has the combinatorial structure of the chosen topology.
    ///
    /// @param spacetime The spacetime in which to build the triangulation
    /// @param numSimplices Target number of top-dimensional simplices to create
    virtual void build(Spacetime *spacetime, int numSimplices) = 0;

    /// The intrinsic dimension \f$ d \f$ of the manifold this topology
    /// triangulates: its top cells are \f$ d \f$-simplices on \f$ d+1 \f$
    /// vertices. A caller uses it to pick the matching ``Signature(d, …)`` so
    /// the complex's top cells register as top-dimensional
    /// (``Spacetime::getTopVertexCount`` == d+1). Without that match
    /// ``topSimplicesVec`` stays empty and boundary / random-top queries return
    /// nothing.
    ///
    /// The fixed-triangulation fixtures (``SimplexBoundarySphere``,
    /// ``SolidSimplex``, ``RealProjectivePlane``, ``RealProjectiveSpace``,
    /// ``SphereCircleProduct``, ``ComplexProjectivePlane``, ``SimplicialProduct``,
    /// ``StellarSubdivision``) each report their manifold dimension. The
    /// dimension-parametric CDT topologies (``Toroid``, ``Sphere``,
    /// ``Cylinder``) have no intrinsic dimension — theirs is read from the
    /// spacetime's signature at ``build()`` time — so they inherit this base
    /// implementation, which throws.
    [[nodiscard]] virtual int dimension() const;

  protected:
    /// Build an explicit, pre-geometric triangulation from a combinatorial
    /// description: create `numVertices` coordinate-free vertices
    /// (ids 0..numVertices-1) and one top simplex per vertex-id tuple in
    /// `topSimplices`.
    ///
    /// No coordinates are assigned. The cobordism capabilities (homology,
    /// characteristic numbers, cobordism existence) are purely combinatorial;
    /// geometry (vertex coordinates / edge lengths) is layered on only when a
    /// geometric capability such as reconstruction or the Regge action needs
    /// it. Edges are still materialized by ``createSimplex`` so the incidence
    /// structure is complete; any squared length they carry is an unused
    /// placeholder, not meaningful geometry.
    ///
    /// Shared by the exact, fixed-triangulation topologies
    /// (``SimplexBoundarySphere``, ``SolidSimplex``, ``RealProjectivePlane``,
    /// …) whose ``build()`` ignores ``numSimplices``.
    static void buildExplicit(
        Spacetime *spacetime, std::size_t numVertices,
        const std::vector<std::vector<std::uint64_t>> &topSimplices);
};

} // namespace tessera::spacetime

#endif //TESSERA_TOPOLOGY_H
