// Schwinger-model Hamiltonian on a Jordan-Wigner'd Kogut-Susskind
// (staggered) fermion chain with Gauss's law eliminated. Builds an
// ITensor matrix product operator (MPO) via AutoMPO, plus a dense Eigen
// reference matrix for small-N cross-checks.
//
// ─── Physics ──────────────────────────────────────────────────────
//
// The dimensional spin Hamiltonian below (1-based site indexing
// n = 1..N) is unitarily equivalent — via a global spin flip
// σ^z → −σ^z combined with an index relabeling — to the Bañuls et al.
// Hamiltonian at L₀ = 0:
//
//     H = H_hop + H_m + H_E
//
//     H_hop = (1/(4a)) Σ_{n=1..N-1}  (X_n X_{n+1} + Y_n Y_{n+1})
//           = (1/(2a)) Σ_{n=1..N-1}  (σ⁺_n σ⁻_{n+1} + σ⁻_n σ⁺_{n+1})
//
//     H_m   = (m/2) Σ_{n=1..N}       (-1)^n σ^z_n
//
//     H_E   = (g²a/2) Σ_{n=1..N-1}   L_n²
//
//     L_n   = L₀ + Σ_{k=1..n} [(1 - σ^z_k)/2  -  (1 - (-1)^k)/2]
//           = c_n - (1/2) Σ_{k=1..n} σ^z_k
//
//     c_n   = L₀ + ((-1)^n - 1)/4
//
// Bañuls' dimensionless parameters are x = 1/(g²a²) and μ = 2m/(g²a).
// The dimensional energy E_dim relates to their dimensionless eigenvalue
// E_W as E_W = (2/(ag²)) E_dim.
//
// ─── What this header exposes ──────────────────────────────────────
//
// • SchwingerParams — dimensional inputs (data class).
// • SchwingerMPO    — ITensor MPO + the SiteSet plus the c-number constant
//                     (data class).
// • SchwingerDense  — same Hamiltonian as a 2^N×2^N real-symmetric matrix
//                     (data class).
// • SchwingerHamiltonian — builder bundling a SchwingerParams with the
//                     Hamiltonian representations downstream code consumes
//                     (MPO for DMRG / TDVP, dense matrix for small-N
//                     cross-checks).
//
// Why `constant` is split off: AutoMPO encodes only operator-valued terms,
// while the pure-identity part of L_n² is a c-number (E_const, derived in
// src/quantum/SchwingerModel.cpp). SchwingerMPO and SchwingerDense expose
// it as a member so callers can recover the full physical energy
// (⟨H⟩ + constant) without rebuilding.
//
// References:
//   Schwinger, "Gauge invariance and mass. II", Phys. Rev. 128, 2425
//     (1962).
//   Kogut & Susskind, "Hamiltonian formulation of Wilson's lattice gauge
//     theories", Phys. Rev. D 11, 395 (1975) — staggered fermions.
//   Jordan & Wigner, "Ueber das Paulische Aequivalenzverbot" (1928).
//   Bañuls, Cichy, Jansen & Cirac, "The mass spectrum of the Schwinger
//     model with Matrix Product States", arXiv:1305.3765.

#pragma once

#include <itensor/all.h>

#include <Eigen/Dense>

#include <cstddef>
#include <utility>
#include <vector>

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

// Dimensional inputs to the Schwinger Hamiltonian, in lattice units. For
// Bañuls' dimensionless x, μ: set a = 1, g = 1/√x and m = (μ/2) g² a.
struct SchwingerParams {
    int    N{0};   // staggered sites, 1-based; must be ≥ 2 (≥ 4 for nontrivial H_E)
    double a{1.0}; // lattice spacing  (must be > 0)
    double m{0.0}; // bare fermion mass
    double g{1.0}; // gauge coupling   (g = 0 is the free-Dirac limit; allowed)
    double L0{0.0};// background electric field on the link to the left of site 1
};

// Operator-valued Schwinger Hamiltonian as an ITensor MPO.
//
// Total physical energy on a state |ψ⟩ is ⟨ψ|H|ψ⟩ + constant.
struct SchwingerMPO {
    SchwingerParams params;
    itensor::SpinHalf sites;  // SpinHalf SiteSet; carries QN structure if enabled
    itensor::MPO H;           // operator-valued part of the Hamiltonian
    double constant{0.0};     // c-number shift from expanding L_n² (see .cpp)
};

// Dense 2^N × 2^N reference Hamiltonian. No symmetry reduction; bit n of
// the row index corresponds to spin n, in the convention documented in
// src/quantum/SchwingerModel.cpp. Capped at N = 16; practical use is
// N ≤ 12.
struct SchwingerDense {
    SchwingerParams params;
    Eigen::MatrixXd H;     // 2^N × 2^N, real symmetric
    double constant{0.0};  // same c-number shift as in SchwingerMPO
};

// Schwinger-Hamiltonian builder.
//
// One instance binds a SchwingerParams; the methods build the Hamiltonian
// representations downstream code consumes (MPO for DMRG / TDVP, dense
// matrix for small-N cross-checks). Stateless beyond its parameters —
// every method returns a freshly assembled representation.
//
// `conserveQns = true` (the default on `mpo()` / `mpoChain()`) makes the
// bond indices carry total Sz, which after the Jordan-Wigner
// transformation is total electric charge. That blocks the MPO and MPS
// into U(1) sectors, speeds up DMRG convergence in the charge-neutral
// sector, and lets a Néel initial state pin the ground-state sector. Pass
// false when measuring operators that do not preserve Sz, such as σ^x,
// the building block of the lattice charge-conjugation operator.
class SchwingerHamiltonian {
public:
    explicit SchwingerHamiltonian(SchwingerParams params) noexcept;

    [[nodiscard]] SchwingerParams const& params() const noexcept { return params_; }

    // Build the operator-valued MPO via ITensor's AutoMPO, with the
    // standard 1D nearest-neighbour hopping graph.
    [[nodiscard]] SchwingerMPO mpo(bool conserveQns = true) const;

    // Causal-set-chain variant: the hopping graph for H_hop is supplied
    // as a list of (site_i, site_j) pairs in 0-based flat-lattice
    // indexing. The mass and electric-field terms keep the standard 1D
    // formulas, which are well defined whenever the lattice has a linear
    // ordering — guaranteed when the underlying causal set is a chain
    // (one vertex per time slice).
    //
    // For a chain causal set, `hoppingPairs` is exactly
    // `[(0,1), (1,2), …, (N-2, N-1)]` and the resulting MPO is identical
    // to `mpo(conserveQns)`.
    //
    // Sites in `hoppingPairs` are 0-based flat indices in [0, p.N - 1].
    // ITensor indexes sites from 1, so 1 is added when feeding AutoMPO.
    [[nodiscard]] SchwingerMPO mpoChain(
        std::vector<std::pair<int, int>> const& hoppingPairs,
        bool conserveQns = true) const;

    // Dense 2^N × 2^N Hamiltonian, no symmetry reduction. Throws
    // std::invalid_argument when N < 2 or N > 16.
    [[nodiscard]] SchwingerDense denseMatrix() const;

    // The c-number part of L_n² accumulated over n = 1..N-1, times the
    // (g²a/2) prefactor of H_E. Returned alongside the MPO / dense H so
    // callers can recover the full physical energy by adding it to ⟨H⟩.
    //
    //   constant = (g²a/2) Σ_{n=1..N-1} (c_n² + n/4)
    //
    // Derived at the top of src/quantum/SchwingerModel.cpp.
    [[nodiscard]] double constant() const;

private:
    SchwingerParams params_;
};

} // namespace tessera::quantum
