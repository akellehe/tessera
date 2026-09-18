// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.
//
// Two-site real-time integrator for the time-dependent variational
// principle. See include/quantum/TDVPIntegrator.hpp for the algorithm
// and the licensing rationale.

#include "quantum/TDVPIntegrator.hpp"

#include <itensor/all.h>

#include <cmath>
#include <stdexcept>

namespace tessera::quantum {

itensor::Real
TDVPIntegrator::evolve(itensor::MPS& psi,
                       itensor::MPO const& H,
                       itensor::Cplx t,
                       itensor::Sweeps const& sweeps,
                       itensor::Args args) {
    using namespace itensor;

    // Local effective Hamiltonian: the MPO projected onto the current
    // bond, built from ITensor core.
    LocalMPO PH(H, args);

    // Degenerate singular values are kept together during svdBond unless
    // the caller says otherwise.
    args.add("RespectDegenerate", args.getBool("RespectDegenerate", true));

    const bool silent = args.getBool("Silent", false);
    if (silent) {
        args.add("Quiet", true);
        args.add("PrintEigs", false);
        args.add("NoMeasure", true);
        args.add("DebugLevel", -1);
    }
    const bool quiet = args.getBool("Quiet", false);
    const int debug_level = args.getInt("DebugLevel", (quiet ? -1 : 0));

    const int numCenter = args.getInt("NumCenter", 2);
    if (numCenter != 2) {
        throw std::invalid_argument(
            "TDVPIntegrator implements only the two-site scheme (NumCenter=2); "
            "the one-site / basisExtension path is intentionally unsupported.");
    }
    args.add("Truncate", args.getBool("Truncate", true));

    const int N = length(psi);
    Real energy = NAN;

    // Put the state in canonical form with the orthogonality centre at site 1.
    if ((!isOrtho(psi)) || (psi.leftLim() != 0)) {
        psi.position(1);
    }

    args.add("DebugLevel", debug_level);

    for (int sw = 1; sw <= sweeps.nsweep(); ++sw) {
        args.add("Sweep", sw);
        args.add("NSweep", sweeps.nsweep());
        args.add("Cutoff", sweeps.cutoff(sw));
        args.add("MinDim", sweeps.mindim(sw));
        args.add("MaxDim", sweeps.maxdim(sw));
        args.add("MaxIter", sweeps.niter(sw));

        ITensor phi0, phi1;
        Spectrum spec;

        // One full TDVP sweep is a forward half-sweep (ha == 1, bonds
        // 1..N-1) then a backward half-sweep (ha == 2, bonds N-1..1),
        // driven by sweepnext.
        for (int b = 1, ha = 1; ha <= 2; sweepnext(b, ha, N, {"NumCenter=", numCenter})) {
            PH.numCenter(numCenter);
            PH.position(b, psi);

            // (1) Forward half-step on the two-site block: phi1 <- e^{(t/2) H_eff} phi1.
            phi1 = psi(b) * psi(b + 1);
            applyExp(PH, phi1, t / 2, args);
            if (args.getBool("DoNormalize", true)) {
                phi1 /= norm(phi1);
            }

            // (2) SVD-truncate the block back onto the chain, moving the centre.
            spec = psi.svdBond(b, phi1, (ha == 1 ? Fromleft : Fromright), PH, args);

            // (3) Backward half-step on the one-site bond tensor left behind,
            //     unless we are at the turning point of the sweep.
            if ((ha == 1 && b + numCenter - 1 != N) || (ha == 2 && b != 1)) {
                const auto b1 = (ha == 1 ? b + 1 : b);
                phi0 = psi(b1);

                PH.numCenter(numCenter - 1);
                PH.position(b1, psi);

                applyExp(PH, phi0, -t / 2, args);
                if (args.getBool("DoNormalize", true)) {
                    phi0 /= norm(phi0);
                }
                psi.ref(b1) = phi0;

                ITensor H_phi0;
                PH.product(phi0, H_phi0);
                energy = real(eltC(dag(phi0) * H_phi0));
            } else {
                ITensor H_phi1;
                PH.product(phi1, H_phi1);
                energy = real(eltC(dag(phi1) * H_phi1));
            }
        }
    }

    psi.rightLim(psi.leftLim() + 2);
    if (args.getBool("DoNormalize", true)) {
        psi.normalize();
    }

    return energy;
}

}  // namespace tessera::quantum
