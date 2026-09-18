// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "observables/SpacetimeVolume.h"
#include "spacetime/Spacetime.h"

// === tessera subsystem ns fwd-decls ===
namespace tessera::graph {}
namespace tessera::mesh {}
namespace tessera::quantum {}
namespace tessera::simulations {}
namespace tessera::spacetime {}
namespace tessera::observables {
using namespace ::tessera::mesh;
using namespace ::tessera::graph;
using namespace ::tessera::spacetime;
using namespace ::tessera::simulations;
using namespace ::tessera::quantum;

double SpacetimeVolume::compute(const std::shared_ptr<Spacetime> &spacetime) {
  return static_cast<double>(spacetime->getTopSimplexCount());
}

double SpacetimeVolume::update(const std::shared_ptr<Spacetime> &spacetime) {
  return compute(spacetime);
}

} // namespace tessera::observables
