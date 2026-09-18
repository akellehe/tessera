// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_SIMULATION_H
#define TESSERA_SIMULATION_H

#include <functional>

// === tessera subsystem ns fwd-decls ===
namespace tessera::graph {}
namespace tessera::mesh {}
namespace tessera::observables {}
namespace tessera::quantum {}
namespace tessera::spacetime {}
namespace tessera::simulations {
using namespace ::tessera::mesh;
using namespace ::tessera::graph;
using namespace ::tessera::spacetime;
using namespace ::tessera::observables;
using namespace ::tessera::quantum;

/// # Simulation Base Class
///
/// Abstract interface for Monte Carlo simulations of discrete spacetime
/// geometries. Each subclass implements one approach to quantum gravity:
///
///   - Causal dynamical triangulation (CDT): Metropolis sampling over causal
///     triangulations weighted by the Regge action. Preserves a global time
///     foliation.
///   - Regge calculus: varies edge lengths on a fixed triangulation, computing
///     the Einstein-Hilbert action from deficit angles at hinges.
///   - Causal set theory: Poisson sprinklings of points in a Lorentzian
///     manifold, with order relations encoding causal structure.
///
/// The simulation lifecycle has two phases:
///
///   1. Tuning: adjust bare coupling constants (e.g. the cosmological constant)
///      so that macroscopic observables (e.g. total spacetime volume) reach
///      their target values. In CDT this drives \f$ N_4 \to \bar{N}_4 \f$.
///
///   2. Thermalization: evolve the system under the Monte Carlo dynamics until
///      it reaches thermal equilibrium, after which configurations are drawn
///      from the stationary distribution
///      \f$ P(\mathcal{T}) \propto e^{-S[\mathcal{T}]} \f$.
///
class Simulation {
  public:
    virtual ~Simulation() = default;

    /// Tune bare coupling constants toward physically meaningful values.
    ///
    /// In CDT this adjusts \f$ k_4 \f$ so the four-volume fluctuates around the
    /// target. In Regge calculus it constructs an initial triangulation. In
    /// causal set theory it sets the sprinkling density.
    virtual void tune(std::function<void(int,int)> progress = nullptr);

    /// Evolve the system to thermal equilibrium.
    /// Evolve the system to thermal equilibrium.
    ///
    /// In CDT this runs Monte Carlo sweeps until the action stabilizes. In
    /// causal set theory this applies a Poisson sprinkling to break
    /// Lorentz-invariance-violating lattice artifacts. In Regge calculus it
    /// applies random edge-length perturbations.
    virtual void thermalize();
};

} // namespace tessera::simulations

#endif //TESSERA_SIMULATION_H
