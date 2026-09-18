// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.
//
// Two-site, real-time integrator for the time-dependent variational
// principle (TDVP) on matrix product states (MPS). It is built entirely
// on Apache-2.0 ITensor *core* primitives (applyExp / LocalMPO /
// MPS::svdBond / sweepnext), so the distributed binary carries no code
// of unstated license. See THIRD_PARTY_NOTICES.md.
//
// The algorithm is the standard two-site TDVP sweep: traverse the chain
// forming each two-site block, evolve it forward a half time-step with a
// local Krylov exponential, SVD-truncate it back onto the chain, then
// evolve the resulting one-site bond tensor backward a half time-step.
// Only the two-site (NumCenter = 2) path is implemented: it is the only
// configuration tessera uses, and a two-site sweep grows the bond
// dimension on its own, so the one-site subspace-expansion
// (basisExtension) path is unnecessary.
//
// References:
//   Haegeman, Cirac, Osborne, Pizorn, Verschelde & Verstraete,
//     "Time-dependent variational principle for quantum lattices",
//     arXiv:1103.0936.
//   Haegeman, Lubich, Oseledets, Vandereycken & Verstraete, "Unifying
//     time evolution and optimization with matrix product states",
//     arXiv:1408.5056 — the two-site integrator and the Lie-Trotter
//     sweep splitting used here.

#ifndef TESSERA_QUANTUM_TDVP_INTEGRATOR_HPP
#define TESSERA_QUANTUM_TDVP_INTEGRATOR_HPP

#include <itensor/all.h>

namespace tessera::quantum {

class TDVPIntegrator {
public:
    // Evolve the MPS `psi` under the MPO Hamiltonian `H` by the (complex)
    // time `t`, applying the two-site TDVP sweep(s) prescribed by `sweeps`.
    //
    // Real-time evolution e^{-iHΔt} corresponds to t = -i·Δt. Returns
    // the variational energy ⟨ψ|H|ψ⟩ measured at the final bond.
    //
    // Recognised Args: "NumCenter" (must be 2), "Truncate" (default true),
    // "DoNormalize" (default true), "Silent", "Quiet",
    // "RespectDegenerate", plus the per-sweep Cutoff / MinDim / MaxDim /
    // MaxIter taken from `sweeps`, and "ErrGoal" forwarded to the Krylov
    // exponentiation.
    static itensor::Real evolve(itensor::MPS& psi,
                                itensor::MPO const& H,
                                itensor::Cplx t,
                                itensor::Sweeps const& sweeps,
                                itensor::Args args = itensor::Args::global());
};

}  // namespace tessera::quantum

#endif  // TESSERA_QUANTUM_TDVP_INTEGRATOR_HPP
