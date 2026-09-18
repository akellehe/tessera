// q-qbar quench: prepare a flux-tube initial state from a vacuum-like
// Schwinger-model matrix product state (MPS) by flipping two spins
// separated by `d` sites.
//
// This follows the string-breaking construction of Buyens et al.
// (2014).
//
// ─── What this does ───────────────────────────────────────────────────────
//
// In the Kogut-Susskind (staggered) + Jordan-Wigner + Gauss-eliminated
// Schwinger formulation, the electric field on link n is
//
//     L_n  =  L_0  +  Σ_{k=1..n} [ (1 - σ^z_k)/2  -  (1 - (-1)^k)/2 ]
//          =  c_n  -  ½ Σ_{k=1..n} σ^z_k
//
// (see include/quantum/SchwingerModel.hpp). Flipping σ^z at two sites
// (k1, k2) with k1 < k2 in opposite directions produces a flux tube:
// L_n shifts by +1 (or −1) on links in [k1, k2 − 1] and returns to the
// vacuum value elsewhere.
//
// To create a +1-flux tube on links [i0, i0+d-1], we need to flip σ^z by
// −2 at site i0 (Up → Dn) and by +2 at site i0+d (Dn → Up):
//
//     U_qq̄(i0, d)  =  σ⁻_{i0}  ·  σ⁺_{i0+d}
//
// ─── Parity constraint ────────────────────────────────────────────────────
//
// For the heavy-quark vacuum |↑↓↑↓ … ⟩ (Up at odd 1-based, Dn at even):
//
//   * σ⁻_{i0} acts non-trivially  ⇔  i0 is odd  (Up sublattice)
//   * σ⁺_{i0+d} acts non-trivially ⇔ i0+d is even (Dn sublattice)
//
// So d must be odd; the heavy-quark acceptance test uses d = 5.
//
// ─── Sz preservation ──────────────────────────────────────────────────────
//
// σ⁻_{i0} · σ⁺_{i0+d} commutes with total Sz (one raise plus one lower
// is net zero charge), so the result stays in the input's U(1) charge
// sector. An MPS built with ConserveQNs=true is therefore valid input.

#pragma once

#include <itensor/all.h>

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

// The q-qbar quench operator. One instance binds the operator's
// parameters (location i0, separation d, parity check); `apply` takes a
// state and returns the quenched state.
class QqbarQuench {
public:
    // Construct a quench operator with σ⁻ acting at site `i0` and σ⁺ at
    // site `i0 + d` (both 1-based). The heavy-quark Néel parity
    // constraint (i0 odd, d odd) is enforced by default; pass
    // `enforceParity = false` to override, for example in the
    // broader-mass regime where the ground-state pattern shifts.
    QqbarQuench(int i0, int d, bool enforceParity = true) noexcept;

    [[nodiscard]] int  i0() const noexcept             { return i0_; }
    [[nodiscard]] int  d()  const noexcept             { return d_;  }
    [[nodiscard]] bool enforceParity() const noexcept  { return enforceParity_; }

    // Apply σ⁻_{i0} σ⁺_{i0+d} to a normalized MPS, return a fresh
    // normalized MPS in the same SiteSet. Throws std::invalid_argument
    // when (i0, d) are out of range or violate the parity constraint
    // (only when enforceParity is true).
    [[nodiscard]] itensor::MPS
    apply(itensor::MPS const& psi, itensor::SpinHalf const& sites) const;

private:
    int  i0_;
    int  d_;
    bool enforceParity_;
};

} // namespace tessera::quantum
