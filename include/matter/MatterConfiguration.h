// Copyright (c) 2026 Twin Vector Labs LLC. All rights reserved.
#pragma once

#include "mesh/ForwardDeclarations.h"
#include <functional>
#include <unordered_map>
#include <vector>

namespace tessera::spacetime { class Spacetime; }

namespace tessera {
using namespace ::tessera::mesh;
using namespace ::tessera::spacetime;
}  // namespace tessera

namespace tessera::matter {

/// Whether a hinge's vertices all lie on one time slice or span two.
enum class HingeType { SPATIAL, TIMELIKE };

/// A worldline: an ordered sequence of vertices connected by timelike edges,
/// representing the trajectory of a point particle through the foliation.
struct Worldline {
    std::vector<VertexPtr> vertices;  ///< ordered by time slice
    double mass;                      ///< particle mass (geometrized units)
};

/// Intrinsic (coordinate-free) specification of stress-energy on a triangulation.
///
/// For point particles, the matter action is the proper-time action:
/// \f$S_{\text{matter}} = -M \int d\tau = -M \sum_{e \in \text{worldline}} \sqrt{-\ell^2_e}\f$
///
/// The Regge equations \f$\partial S/\partial \ell^2_e = 0\f$ then give the
/// discrete Einstein equations sourced by the particle.
class MatterConfiguration {
  public:
    /// Assign a static point mass along its worldline through all time slices.
    ///
    /// Traces a worldline from \a center through the foliation by following
    /// timelike edges, then stores the worldline for the proper-time action.
    void setWorldlineMass(VertexPtr center, double mass, const Spacetime &st);

    /// Assign energy density \f$\rho\f$ uniformly to a top-simplex.
    void setEnergyDensity(SimplexPtr simplex, double rho);

    /// Assign energy density as a function of geodesic distance from a
    /// center vertex.
    void setRadialProfile(VertexPtr center,
                          std::function<double(double)> rhoOfR);

    // ==================== Accessors ====================

    /// Get all registered worldlines.
    [[nodiscard]] const std::vector<Worldline>& getWorldlines() const noexcept {
        return worldlines_;
    }

    // ==================== Utilities ====================

    /// Trace a worldline through all time slices, passing through the spatial
    /// center of each slice.  Returns one vertex per slice, ordered by time.
    ///
    /// On each slice, the center vertex is the one with the most spacelike
    /// neighbors.  Ties are broken by BFS (depth <= 5) sum-of-distances.
    /// The depth limit avoids wrapping around compact topologies.
    static std::vector<VertexPtr> buildWorldline(VertexPtr center,
                                                  const Spacetime &st);

    /// Classify a hinge as spatial (all vertices at one time) or timelike.
    static HingeType classifyHinge(SimplexPtr hinge);

  private:
    std::vector<Worldline> worldlines_;
    std::unordered_map<std::uint64_t, double> simplexRho_;    // simplex fp → ρ

    struct RadialProfile {
        VertexPtr center;
        std::function<double(double)> rho;
    };
    std::vector<RadialProfile> radialProfiles_;
};

} // namespace tessera::matter
