// SchwingerQuench — q-qbar quench plus real-time evolution under the
// time-dependent variational principle (TDVP) for the Schwinger model.
//
// A self-contained driver from the density-matrix renormalization group
// (DMRG) through the quench to TDVP: it takes a flat config struct and
// returns per-snapshot scalar diagnostics. The same class also runs the
// causal-order comparison.
//
// ─── End-to-end pipeline (SchwingerQuench::evolve) ────────────────────
//
//   1. Build the Schwinger MPO and run DMRG to the ground state.
//   2. Apply the σ⁻_{i0} · σ⁺_{i0+d} quench, flipping the spins at the
//      ends of the q-qbar pair (Quench.hpp).
//   3. Record the post-quench observables (snapshot at t = 0).
//   4. Step TDVP forward by Δt for n_steps = T/Δt steps. Every
//      `snapshotEvery` steps, record ⟨L_n⟩(t), ⟨σ^z_n⟩(t), the bond
//      dimension, and the energy ⟨ψ(t)|H|ψ(t)⟩.
//   5. Optionally compute the Schmidt spectra and the majorization poset
//      per snapshot. This costs O(N²) SVDs per snapshot and is off by
//      default.
//
// ─── Causal-order comparison (SchwingerQuench::compareCausalOrders) ──────
//
// Build three partial orders on the (cut, time) labels produced by
// `evolve()` (with recordSpectra forced on) and report pairwise
// agreement statistics. CausalCompare.hpp defines the three orders.
//
// Reference: Haegeman, Lubich, Oseledets, Vandereycken & Verstraete,
// "Unifying time evolution and optimization with matrix product
// states", arXiv:1408.5056.

#pragma once

#include "quantum/CausalCompare.hpp"   // CausalComparisonReport
#include "quantum/DMRGRunner.hpp"      // GroundStateResult
#include "quantum/Majorization.hpp"     // MajorizationPredicate, Poset
#include "quantum/Schmidt.hpp"          // SchmidtSpectra

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

// Configuration for a complete DMRG-then-TDVP run. Hamiltonian fields
// mirror QuantumConfig; the rest configure the quench, the TDVP loop,
// and per-step observable recording.
struct TDVPConfig {
    // ─── Hamiltonian (passed to SchwingerParams) ────────────────────────
    int    N{0};
    double a{1.0};
    double m{0.0};
    double g{1.0};
    double L0{0.0};

    // ─── DMRG ground-state setup ────────────────────────────────────────
    int    dmrgMaxBondDim{100};
    int    dmrgNSweeps{12};
    int    dmrgKrylovDim{4};
    double dmrgCutoff{1e-12};

    // ─── q-qbar quench: σ⁻_{i0} · σ⁺_{i0+d} ─────────────────────────────
    int  i0{0};                // first site of the pair, 1-based
    int  d{0};                 // separation, must be odd for the heavy-
                               // quark Néel parity to align (Quench.hpp)
    bool quenchEnforceParity{true};

    // ─── TDVP loop ─────────────────────────────────────────────────────
    double dt{0.05};           // real-time step
    double T{1.0};             // total evolution time (sets n_steps = T/dt)
    int    maxBondDim{200};   // bond-dimension cap during TDVP sweeps
    int    krylovDim{12};     // Krylov / Lanczos dimension per local solve
    double cutoff{1e-10};      // SVD truncation per local solve
    int    snapshotEvery{1};  // record observables every k steps (≥ 1)
    bool   quiet{true};
    bool   conserveQns{true};

    // ─── Observable recording ──────────────────────────────────────────
    // recordSpectra must be true for snapshots to carry any spectra;
    // recordPoset additionally builds the majorization poset on them and
    // implies recordSpectra.
    //
    // recordMutualInformation stores the full N×N all-pairs site-site
    // mutual-information matrix per snapshot, which
    // tessera.quantum.holography turns into the (site, time) graph for
    // the spectral-dimension measurement. Cost: O(N² · χ³) per snapshot.
    bool recordSpectra{false};
    bool recordPoset{false};
    bool recordMutualInformation{false};
    // recordBondMutualInformation stores the full (N-1) × (N-1)
    // tripartite-information matrix on bond cuts per snapshot, used by
    // the dual-lattice spectral-dimension pipeline. Each entry is the
    // van Raamsdonk-style tripartite information S(A) + S(C) − S(B) for
    // outer regions A, C separated by the middle region B between cuts.
    // Cost: O(N³ · χ⁵) per snapshot, heavier than site-site MI by
    // roughly N · χ² because of the contiguous-interval entropy sweep.
    bool recordBondMutualInformation{false};

    // Optional explicit hopping graph for the Schwinger Hamiltonian.
    // Empty (the default) uses the standard 1D nearest-neighbour chain
    // (sites n, n+1 for n = 0..N−2). Non-empty instead selects
    // SchwingerHamiltonian::mpoChain(hoppingPairs, conserveQns), which
    // routes a tessera.Spacetime → CausetChain hopping pattern into the
    // TDVP evolution. Each pair is (i, j) in 0-based flat-lattice
    // indices.
    std::vector<std::pair<int, int>> hoppingPairs;
};

// Per-step diagnostics. Schmidt spectra, poset, and mutual-information
// fields are populated only when the corresponding TDVPConfig flag is set.
struct TDVPSnapshot {
    double time{0.0};
    double energy{0.0};               // ⟨ψ|H|ψ⟩ + sm.constant
    int    bondDim{0};               // maxLinkDim(ψ) at this time
    std::vector<double> zProfile;    // ⟨σ^z_n⟩ for n = 1..N
    std::vector<double> lProfile;    // ⟨L_n⟩  for n = 1..N-1
    SchmidtSpectra      spectra;      // populated if cfg.recordSpectra
    Poset               poset;        // populated if cfg.recordPoset
    // Symmetric N×N matrix of site-site mutual information in nats,
    // stored row-major in a flat vector (length N·N). Zero diagonal.
    // Populated iff cfg.recordMutualInformation. Empty otherwise.
    std::vector<double> mutualInformation;
    // Symmetric (N-1) × (N-1) matrix of bond-cut tripartite information
    // in nats, stored row-major in a flat vector. Zero diagonal.
    // Populated iff cfg.recordBondMutualInformation. Empty otherwise.
    std::vector<double> bondMutualInformation;
};

// Result bundle from SchwingerQuench::evolve: ground-state diagnostics,
// then the snapshots, starting at t = 0 (post-quench, before any TDVP
// step).
struct QuenchResult {
    GroundStateResult         groundState;
    std::vector<TDVPSnapshot> snapshots;
};

// Schwinger-model quench and dynamics pipeline.
//
// One instance binds a TDVPConfig; the methods run the full DMRG →
// quench → TDVP loop (`evolve`) or that loop followed by the
// causal-order comparison (`compareCausalOrders`). Stateless beyond its
// config — every method runs the underlying pipeline from scratch.
class SchwingerQuench {
public:
    explicit SchwingerQuench(TDVPConfig config) noexcept;

    [[nodiscard]] TDVPConfig const& config() const noexcept { return config_; }

    // Run the full DMRG → quench → TDVP pipeline. Throws
    // std::invalid_argument on bad config (out-of-range i0, parity
    // mismatch with quenchEnforceParity, etc.).
    [[nodiscard]] QuenchResult evolve() const;

    // End-to-end causal-comparison pipeline: evolve(), then build three
    // partial orders on the (cut, time) labels and compare. Forces
    // `cfg.recordSpectra = true` regardless of the input.
    //
    // `vLr` is the Lieb-Robinson velocity in lattice units (sites per
    // unit time). The default 1.0 is the free-fermion group velocity for
    // this hopping coefficient.
    //
    // `predicate` selects the majorization variant for ≼_maj. nullptr
    // means classical majorization (StandardMajorization{1e-12}).
    [[nodiscard]] CausalComparisonReport compareCausalOrders(
        double vLr = 1.0,
        MajorizationPredicate const* predicate = nullptr) const;

private:
    TDVPConfig config_;
};

} // namespace tessera::quantum
