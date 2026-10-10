// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

// MultiCobordism (include/cobordism/MultiCobordism.h): construction. One of
// the translation units that define the class's members by responsibility
// (https://github.com/akellehe/tessera/issues/1481).

#include "MultiCobordismInternal.h"

namespace tessera::cobordism {

MultiCobordism::MultiCobordism(
    std::shared_ptr<Spacetime> host,
    const std::vector<std::vector<complexd>> &inputTargets,
    const std::vector<std::vector<complexd>> &outputTargets,
    const std::vector<int> &degrees, double gamma, std::uint64_t seed,
    int precone, bool shouldProposeDispositions, bool preconeTimelike,
    bool preconeAlternate, bool balancedEdgeWiring, bool singularValueRatio,
    bool einsteinHilbert, bool realSquaredLengthsOnly,
    HodgeLaplacian::MetricSource metricSource)
    : spacetime_(std::move(host)),
      inputTargets_(inputTargets),
      outputTargets_(outputTargets),
      registerDegrees_(degrees),
      dualComplexGateDegree_(
          registerDegrees_.empty()
              ? 0
              : *std::max_element(registerDegrees_.begin(),
                                  registerDegrees_.end())),
      gamma_(gamma),
      balancedEdgeWiring_(balancedEdgeWiring),
      singularValueRatio_(singularValueRatio),
      einsteinHilbert_(einsteinHilbert),
      realSquaredLengthsOnly_(realSquaredLengthsOnly),
      metricSource_(metricSource),
      randomNumberGenerator_(seed) {
  // The wiring mode must reach the host before any precone growth below wires
  // its first edge.
  if (spacetime_) spacetime_->setBalancedEdgeWiring(balancedEdgeWiring_);
  // Assigned in the body rather than the member init list: the member is
  // declared last and C++ initializes in declaration order, so an init-list
  // entry here would trigger a reorder warning. It is a bool with an in-class
  // default, so nothing depends on it being set earlier.
  shouldProposeDispositions_ = shouldProposeDispositions;
  // The deterministic provenance stamp of every checkpoint this node writes.
  // Assigned in the body for the same declaration-order reason.
  seed_ = seed;
  // Install the built-in matching the default mode, so `objectiveSpec_` is
  // never null.
  objectiveSpec_ = std::make_shared<LegacyObjective>();
  // Pre-grow the seed by `precone` gated cone-ins before any optimization, so
  // that the stage-1 search starts from a larger complex. No input or output
  // block is seeded yet, so nothing is pinned and the gate is the only
  // constraint. `precone <= 0` leaves the host and RNG untouched.
  // `preconeTimelike` draws every cone-in with the timelike disposition;
  // `preconeAlternate` instead alternates timelike and spacelike for balanced
  // causal content, and takes precedence. The default is all-spacelike.
  if (precone > 0) preconeCells(precone, preconeTimelike, preconeAlternate);
}

}  // namespace tessera::cobordism
