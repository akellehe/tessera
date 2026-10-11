// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

// The candidate moves the combinatorial stage proposes.

#ifndef TESSERA_COBORDISM_MULTICOBORDISM_MOVES_H
#define TESSERA_COBORDISM_MULTICOBORDISM_MOVES_H

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace tessera::cobordism::multicobordism {

/// One candidate move: its kind, and the payload naming where or how to act.
/// Returned by `enumerateMoveSpecifications`.
using MoveSpec = std::pair<std::string, std::vector<std::uint64_t>>;

}  // namespace tessera::cobordism::multicobordism

#endif  // TESSERA_COBORDISM_MULTICOBORDISM_MOVES_H
