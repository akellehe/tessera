// Choi-state propagator for the Schwinger Hamiltonian — the temporal
// half of the holography pipeline.
//
// For a unitary U on N qubits the Choi state on a doubled register is
//
//     |U⟩  =  (1/√d) Σ_x  |x⟩_in ⊗ U|x⟩_out  =  (U_out ⊗ I_in) |Φ+⟩^{⊗N},
//
// where |Φ+⟩^{⊗N} is the tensor product of Bell pairs between matching
// (in, out) sites. Temporal mutual information between (site i, time s)
// and (site j, time t) is then the 2-site reduced-density-matrix
// mutual information on (in_i, out_j) of the Choi state of U_{s→t}.
//
// For a TIME-INDEPENDENT Hamiltonian (the Schwinger H is one) the
// propagator U_{s→t} depends only on the duration (t − s), so this
// class indexes Choi states by *stride* rather than absolute (s, t)
// pairs.
//
// Site ordering on the doubled chain is INTERLEAVED:
//
//     site 1 = in_1,  site 2 = out_1,  site 3 = in_2,  site 4 = out_2, …
//
// so Bell pairs are nearest-neighbour (bond dimension 2 within pairs,
// 1 between pairs) and bond-dimension growth under the time-dependent
// variational principle (TDVP) stays local.
//
// The Hamiltonian on the doubled chain acts as the identity on every
// in-site and as the Schwinger H on the out-sites, re-indexed onto the
// even-numbered sites. AutoMPO handles the resulting longer-range
// σ⁺σ⁻ + σ⁻σ⁺ hopping and the L_n² expansion.
//
// References:
//   Choi, "Completely positive linear maps on complex matrices",
//     Linear Algebra Appl. 10, 285 (1975) — the correspondence between
//     channels and bipartite states.
//   Jamiołkowski, "Linear transformations which preserve trace and
//     positive semidefiniteness of operators", Rep. Math. Phys. 3, 275
//     (1972) — the equivalent map.
//   Pollock, Rodríguez-Rosario, Frauenheim, Paternostro & Modi,
//     "Operational Markov condition for quantum processes",
//     arXiv:1801.09811 — the process-tensor / Choi-state view of
//     multi-time correlations.

#pragma once

#include "quantum/SchwingerModel.hpp"

#include <Eigen/Dense>
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

// Stateless utility class — not instantiable, all methods static. It
// holds no config; it exposes the building blocks that
// MutualInformationProfile composes.
class ChoiPropagator {
public:
    ChoiPropagator() = delete;
    ChoiPropagator(ChoiPropagator const&) = delete;
    ChoiPropagator& operator=(ChoiPropagator const&) = delete;

    // Sweep settings for the TDVP evolution on the doubled chain,
    // grouped so callers need not thread the scalars individually
    // through the public API.
    struct TDVPSettings {
        double dt{0.05};
        int    maxBondDim{200};
        int    krylovDim{12};
        double cutoff{1e-10};
        bool   quiet{true};
    };

    // Build the doubled-chain SiteSet. Total length 2N, ConserveQNs
    // is forced OFF because the initial Bell state superposes the
    // Sz = +1 and Sz = -1 sectors of each (in, out) pair.
    [[nodiscard]] static itensor::SpinHalf
    doubledSites(int N);

    // The product-Bell-pair initial state |Φ+⟩^{⊗N} as a 2N-site matrix
    // product state, with bond dimension 2 within each (in_k, out_k)
    // pair and 1 between pairs. This is the Choi state of the identity
    // channel.
    [[nodiscard]] static itensor::MPS
    bellChainMPS(itensor::SpinHalf const& doubled);

    // The Schwinger Hamiltonian on the doubled chain, acting only on
    // the out-register (even-numbered sites) and as the identity on
    // the in-register (odd-numbered sites).
    [[nodiscard]] static itensor::MPO
    outputHamiltonianMPO(itensor::SpinHalf const& doubled,
                          SchwingerParams const& p);

    // Run TDVP on `psi` for the requested duration under H_out.
    // `psi` must live on the 2N-site doubled SiteSet returned by
    // doubledSites(); `H` must be the MPO returned by
    // outputHamiltonianMPO() (or a 2N-site MPO defined on the same
    // SiteSet).
    [[nodiscard]] static itensor::MPS
    evolve(itensor::MPS psi,
           itensor::MPO const& H,
           double duration,
           TDVPSettings const& settings);

    // Build the Choi state of U_{0→duration} from scratch: Bell-chain
    // initial state, then TDVP evolution under outputHamiltonianMPO.
    [[nodiscard]] static itensor::MPS
    choiState(SchwingerParams const& p,
              double duration,
              TDVPSettings const& settings);

    // Temporal mutual-information matrix from a Choi-state MPS.
    //
    // For a Choi state on the 2N-site doubled chain (interleaved order
    // in_k = site 2k-1, out_k = site 2k), the entry (i-1, j-1) of the
    // returned N×N matrix is I({in_i} : {out_j}) in nats — the
    // temporal MI between site i at the input time and site j at the
    // output time. Diagonal entries quantify "this site's state at
    // time t is still correlated with what it was at time 0".
    //
    // For the identity channel: returns 2·ln(2) on the diagonal and
    // zero off the diagonal (each Bell pair contributes I = 2 ln 2).
    [[nodiscard]] static Eigen::MatrixXd
    temporalMutualInformation(itensor::MPS const& choi, int N);
};

} // namespace tessera::quantum
