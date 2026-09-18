// SchwingerModel — ground-state pipeline for the Schwinger model on a
// Jordan-Wigner spin chain, driven by the density-matrix
// renormalization group (DMRG).
//
// A self-contained ground-state driver: it takes a flat config struct
// and returns plain scalars, so no matrix product state (MPS), MPO or
// ITensor type crosses the Python boundary.
//
// What this header exposes:
//
// • QuantumConfig                   — Hamiltonian + DMRG sweep settings (data class).
// • GroundStateResult               — DMRG diagnostics (data class).
// • GroundStateMajorizationResult   — DMRG + Schmidt + majorization-poset
//                                     bundle (data class).
// • SchwingerModel                  — façade holding a QuantumConfig; its
//                                     methods run DMRG and, optionally,
//                                     the Schmidt + majorization extension.
//
// What SchwingerModel methods do internally:
//   1. Build the SchwingerMPO from the dimensional parameters.
//   2. Set up a Néel |↑↓↑↓…⟩ initial MPS in the Sz = 0 sector.
//   3. Run ITensor's two-site dmrg() with a sweep schedule built from
//      the QuantumConfig (bond-dimension cap, sweep count, cutoff,
//      Krylov dimension).
//   4. Read back the final operator energy ⟨H⟩, the c-number constant
//      from the L_n² expansion, the achieved bond dimension, and the
//      truncation error of the last sweep.
//   5. (solveWithMajorization only) compute every contiguous-cut Schmidt
//      spectrum of the optimized MPS and build the strict-majorization
//      Hasse poset on those spectra.
//
// References:
//   Schollwoeck, "The density-matrix renormalization group in the age
//     of matrix product states", arXiv:1008.3477.
//   White, "Density matrix formulation for quantum renormalization
//     groups", Phys. Rev. Lett. 69, 2863 (1992).

#pragma once

#include "quantum/Majorization.hpp"     // Poset
#include "quantum/Schmidt.hpp"          // SchmidtSpectra
#include "quantum/SchwingerModel.hpp"  // SchwingerHamiltonian, SchwingerParams

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

// Flat, Python-friendly configuration: Hamiltonian parameters plus DMRG
// sweep settings. The defaults suit the small-to-moderate N used in
// tests; raise maxBondDim and nSweeps for tighter convergence.
struct QuantumConfig {
    // ─── Hamiltonian (passed straight to SchwingerParams) ───────────────
    int    N{0};       // staggered sites, 1-based; must be ≥ 2
    double a{1.0};     // lattice spacing
    double m{0.0};     // bare fermion mass
    double g{1.0};     // gauge coupling
    double L0{0.0};    // background electric field on the link left of site 1

    // ─── DMRG ──────────────────────────────────────────────────────────
    int    maxBondDim{100};   // cap on the MPS bond dimension during sweeps
    int    nSweeps{12};        // total sweep count
    double cutoff{1e-12};       // SVD truncation threshold per local solve
    int    krylovDim{4};       // Lanczos / Krylov dimension per local solve
    bool   quiet{true};         // suppress ITensor's per-sweep diagnostics
    bool   conserveQns{true};  // U(1) total-Sz conservation on the SiteSet

    // ─── Time-dependent variational principle (TDVP) / quench
    // parameters. Unused by SchwingerModel; carried here so one config
    // struct covers the whole Python-facing API.
    double dt{0.01};   // real-time step size
    double T{1.0};     // total evolution time
};

// What SchwingerModel::solve() returns. `constant` and `operatorEnergy`
// are exposed alongside `energy` because a caller comparing against
// published numerics needs to know whether a value is the operator-only
// part or the full physical energy.
struct GroundStateResult {
    double energy{0.0};          // ⟨H⟩ + constant — the full physical energy
    double operatorEnergy{0.0}; // ⟨H⟩ alone — what ITensor's dmrg() returned
    double constant{0.0};        // c-number shift from L_n² expansion
    int    bondDim{0};          // largest bond dimension of the optimized MPS
    double truncationErr{0.0};  // largest truncation error in the final sweep
};

// What SchwingerModel::solveWithMajorization() returns. Bundles the same
// scalar diagnostics that solve() returns with the Schmidt spectra and
// the majorization poset of those spectra.
struct GroundStateMajorizationResult {
    GroundStateResult groundState;  // energy, bondDim, truncationErr …
    SchmidtSpectra    spectra;       // labelled contiguous-cut spectra
    Poset             poset;         // Hasse cover edges of strict majorization
};

// Schwinger-model ground-state pipeline.
//
// One instance binds a QuantumConfig; the methods run either the bare
// DMRG ground state or the DMRG + Schmidt + majorization-poset
// pipeline. Stateless beyond its config — every method runs the
// underlying ITensor pipeline from scratch.
class SchwingerModel {
public:
    explicit SchwingerModel(QuantumConfig config) noexcept;

    [[nodiscard]] QuantumConfig const& config() const noexcept { return config_; }

    // Run DMRG to the ground state. Throws std::invalid_argument when the
    // Hamiltonian parameters are invalid (forwarded from
    // SchwingerHamiltonian::mpo's validation: N ≥ 2, a > 0).
    [[nodiscard]] GroundStateResult solve() const;

    // Run DMRG, then extract every contiguous-cut Schmidt spectrum of the
    // optimized MPS and build the strict-majorization Hasse poset on those
    // spectra. `tol` is the numerical slack on the majorization
    // comparisons.
    [[nodiscard]] GroundStateMajorizationResult
    solveWithMajorization(double tol = 1e-12) const;

private:
    QuantumConfig config_;
};

} // namespace tessera::quantum
