// Spacetime → causal-set-chain adapter for the quantum subsystem.
//
// The Schwinger MPO (SchwingerModel.hpp) lives on a regular 1D lattice
// with N sites and nearest-neighbour hopping pairs (n, n+1). This
// adapter replaces that lattice with a "chain of antichains" sourced
// from a tessera::spacetime::Spacetime: each antichain is the set of
// vertices at a fixed integer time slice, and hopping follows the
// timelike causal-set edges connecting adjacent slices.
//
// This header only extracts the data; it does not rebuild an MPO. When
// every antichain holds exactly one vertex the chain of antichains
// coincides with the 1D lattice, so `SchwingerHamiltonian::mpoChain`
// can run directly with `params.N = chain.nSites` and
// `chain.hoppingPairs` as the hopping graph.
//
// Provided here:
//   • CausetChain — flattened (lattice site → spacetime vertex ID)
//     mapping, the hopping pairs, and the inherited Hasse-cover Poset.
//   • Causet — adapter façade (static methods only).
//
// References:
//   Sorkin, "Causal Sets: Discrete Gravity", arXiv:gr-qc/0309009.
//   Bombelli, Lee, Meyer & Sorkin, "Space-time as a causal set",
//     Phys. Rev. Lett. 59, 521 (1987).

#pragma once

#include "Poset.h"

#include <cstdint>
#include <utility>
#include <vector>

namespace tessera::spacetime { class Spacetime; }

// === tessera subsystem ns fwd-decls ===
namespace tessera::graph {}
namespace tessera::mesh {}
namespace tessera::observables {}
namespace tessera::simulations {}
namespace tessera::spacetime {}
namespace tessera::quantum {
using namespace ::tessera::mesh;
using namespace ::tessera::graph;
using namespace ::tessera::spacetime;
using namespace ::tessera::observables;
using namespace ::tessera::simulations;

// Spacetime → 1D lattice adapter (data class).
//
// `antichains[s]` is the sorted list of Spacetime vertex IDs at
// `times[s]`, where `times` is ascending-sorted. The flat lattice
// site index of a (slice, position-in-antichain) pair is the
// concatenation:
//
//   flat_idx = (Σ_{r<s} |antichains[r]|) + position
//
// `vertexIds[flat_idx]` is the inverse map: lattice site → spacetime
// vertex ID. `nSites = sum(|antichains[s]|) = vertexIds.size()`.
//
// `hoppingPairs` lists the (i, j) flat-lattice-site pairs coupled by
// adjacent-time-slice timelike edges; they replace the
// "Σ_n (X_n X_{n+1} + Y_n Y_{n+1})" sum in H_hop. Pairs are stored once
// with i < j; the MPO builder applies σ⁺σ⁻ + σ⁻σ⁺ symmetrically per
// pair. Edges spanning non-adjacent slices are skipped — they are
// transitively reduced out by Poset::fromSpacetime and carry no
// physical hopping term.
//
// `partialOrder` is the Hasse-cover Poset on flat-lattice-site IDs,
// inherited from Spacetime via Poset::fromSpacetime. It is the "≼_cs"
// entry of the causal-order comparison.
struct CausetChain {
    int nSites{0};
    std::vector<int> times;                                 // ascending
    std::vector<std::vector<std::uint64_t>> antichains;     // [tIdx][pos]
    std::vector<std::uint64_t> vertexIds;                  // [flat_idx]
    std::vector<std::pair<int, int>> hoppingPairs;         // [k] = (i, j) i<j
    tessera::Poset partialOrder;
};

// Façade for tessera::spacetime::Spacetime → causal-set adapters.
// Stateless; not instantiable.
class Causet {
public:
    Causet() = delete;
    Causet(Causet const&) = delete;
    Causet& operator=(Causet const&) = delete;

    // Walk the Spacetime's vertex list, group by integer time slice
    // (Vertex::getTime() truncated to int), and extract:
    //
    //   • the antichain layering (antichains, times),
    //   • the flat lattice ↔ spacetime ID mapping (vertexIds),
    //   • the adjacent-slice timelike-edge hopping pairs,
    //   • the Hasse cover Poset on flat lattice IDs.
    //
    // All four outputs share the same flat-index labelling, so a
    // caller can feed them interchangeably to Majorization::agreement,
    // an MPO builder, or a visualisation backend.
    //
    // Antichain ordering inside a slice is by ascending Spacetime
    // vertex ID, so the result is deterministic. Edges with
    // squaredLength ≥ 0 (spacelike or null) are ignored, as are
    // timelike edges with src.time == tgt.time.
    [[nodiscard]] static CausetChain
    chainFrom(tessera::spacetime::Spacetime const& st);
};

} // namespace tessera::quantum
