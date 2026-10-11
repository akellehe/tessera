// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

// The build actions and hole placements of the two stages.

#ifndef TESSERA_COBORDISM_MULTICOBORDISM_STAGES_H
#define TESSERA_COBORDISM_MULTICOBORDISM_STAGES_H


namespace tessera::cobordism::multicobordism {

/// One solve action on this node: the unit a search policy composes, so that
/// the solve is driven through the engine rather than re-implemented by each
/// consumer.
enum class BuildAction { Grow, Evolve, Relax, ConeOut, ConeIn };

/// Secondary sort for the directed cone-out probe; both orders are
/// interior-first. `AdjacentHolesLast` sends cells sharing vertices with the
/// existing holes to the back, so new holes come out separated;
/// `AdjacentHolesFirst` brings them to the front, so the register clusters.
/// For the first hole the orders coincide.
enum class HolePlacementStrategy { AdjacentHolesFirst, AdjacentHolesLast };

}  // namespace tessera::cobordism::multicobordism

#endif  // TESSERA_COBORDISM_MULTICOBORDISM_STAGES_H
