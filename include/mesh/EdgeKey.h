// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

//
// Created by andrew on 12/12/25.
//

#ifndef TESSERA_EDGEKEY_H
#define TESSERA_EDGEKEY_H

#include "mesh/ForwardDeclarations.h"
#include <unordered_set>
#include <string>
#include <vector>
#include <cstdint>
#include <stdexcept>

// === tessera subsystem ns fwd-decls ===
namespace tessera::graph {}
namespace tessera::observables {}
namespace tessera::quantum {}
namespace tessera::simulations {}
namespace tessera::spacetime {}
namespace tessera::mesh {
using namespace ::tessera::graph;
using namespace ::tessera::spacetime;
using namespace ::tessera::observables;
using namespace ::tessera::simulations;
using namespace ::tessera::quantum;

/// The pair of endpoint vertex ids identifying an edge. Two edges with equal keys are
/// the same edge; ``fingerprint`` is the hashed form used as a container key.
class EdgeKey {
  public:
    IdType first{0};
    IdType second{0};

    EdgeKey(IdType sourceId_, IdType targetId_);

    bool operator==(const EdgeKey &other) const;

    [[nodiscard]] std::uint64_t hash() const;
#ifdef TESSERA_VERBOSE
    std::string toString() const noexcept;
#else
    std::string toString() const noexcept;
#endif


    Fingerprint fingerprint;
};
} // namespace tessera::mesh

#endif //TESSERA_EDGEKEY_H
