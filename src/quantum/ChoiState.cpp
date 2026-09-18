// Implementation of ChoiPropagator. See include/quantum/ChoiState.hpp
// for the construction and the site-ordering convention.

#include "quantum/ChoiState.hpp"

#include "quantum/MutualInformation.hpp"
#include "quantum/SchwingerModel.hpp"

#include "quantum/TDVPIntegrator.hpp"  // two-site TDVP on ITensor core primitives

#include <itensor/all.h>
#include <itensor/mps/autompo.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

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

namespace {

// Closed form of the Schwinger c-number coefficients, duplicated from
// SchwingerModel.cpp so this translation unit does not depend on the
// internal helpers there. The constant does not enter the temporal
// mutual information — it is a c-number shift on energy, not on the
// dynamics — but keeping it makes out-register expectation values
// consistent with the standard SchwingerMPO build.
inline double c_n(int n, double L0) {
    return L0 + ((n % 2 == 0) ? 0.0 : -0.5);
}

inline double tailSumC(int k, int N, double L0) {
    double s = 0.0;
    for (int n = k; n <= N - 1; ++n) s += c_n(n, L0);
    return s;
}

} // namespace

itensor::SpinHalf
ChoiPropagator::doubledSites(int N) {
    if (N < 2) {
        throw std::invalid_argument(
            "ChoiPropagator::doubledSites: N must be >= 2");
    }
    return itensor::SpinHalf(2 * N, {"ConserveQNs=", false});
}

itensor::MPS
ChoiPropagator::bellChainMPS(itensor::SpinHalf const& sites) {
    using namespace itensor;
    const int twoN = length(sites);
    if (twoN < 2 || (twoN % 2) != 0) {
        throw std::invalid_argument(
            "ChoiPropagator::bellChainMPS: doubled site set must have "
            "even length >= 2");
    }
    const int N = twoN / 2;

    // Start from |↑↑…↑⟩ and apply U = CNOT · (H ⊗ I) to each
    // (in_k, out_k) pair: U|↑↑⟩ = (|↑↑⟩ + |↓↓⟩) / √2 = |Φ+⟩.
    //
    // Each pair's gate is a 4×4 unitary applied by the standard two-site
    // ITensor pattern (contract neighbouring site tensors, multiply by
    // the gate, SVD back), so the resulting MPS has bond dimension 2
    // within each pair and 1 between pairs.

    auto state = InitState(sites);
    for (int i = 1; i <= twoN; ++i) state.set(i, "Up");
    MPS psi(state);

    const double s2inv = 1.0 / std::sqrt(2.0);

    for (int k = 1; k <= N; ++k) {
        const int i = 2 * k - 1;  // in_k
        const int j = 2 * k;       // out_k
        psi.position(i);

        auto si = sites(i);
        auto sj = sites(j);

        // 4×4 unitary in the basis (|↑↑⟩, |↑↓⟩, |↓↑⟩, |↓↓⟩); ITensor
        // SpinHalf indexes Up = 1, Dn = 2.
        //
        //   U·|↑↑⟩ = (|↑↑⟩ + |↓↓⟩)/√2   <- the Bell-prep column
        //
        // The remaining columns keep U unitary (their output columns are
        // orthonormal), so the following SVD accumulates no noise on the
        // unused subspace:
        //   U·|↑↓⟩ = (|↑↓⟩ + |↓↑⟩)/√2
        //   U·|↓↑⟩ = (|↑↑⟩ − |↓↓⟩)/√2
        //   U·|↓↓⟩ = (|↑↓⟩ − |↓↑⟩)/√2
        ITensor G(prime(si), prime(sj), si, sj);
        // Column |↑↑⟩:
        G.set(prime(si) = 1, prime(sj) = 1, si = 1, sj = 1, s2inv);
        G.set(prime(si) = 2, prime(sj) = 2, si = 1, sj = 1, s2inv);
        // Column |↑↓⟩:
        G.set(prime(si) = 1, prime(sj) = 2, si = 1, sj = 2, s2inv);
        G.set(prime(si) = 2, prime(sj) = 1, si = 1, sj = 2, s2inv);
        // Column |↓↑⟩:
        G.set(prime(si) = 1, prime(sj) = 1, si = 2, sj = 1,  s2inv);
        G.set(prime(si) = 2, prime(sj) = 2, si = 2, sj = 1, -s2inv);
        // Column |↓↓⟩:
        G.set(prime(si) = 1, prime(sj) = 2, si = 2, sj = 2,  s2inv);
        G.set(prime(si) = 2, prime(sj) = 1, si = 2, sj = 2, -s2inv);

        // wf = G * (psi(i) * psi(i+1)), then SVD back into MPS.
        ITensor wf = psi(i) * psi(j);
        wf *= G;
        wf.noPrime();
        // svdBond at bond i places the orthogonality center at i+1.
        psi.svdBond(i, wf, Fromleft);
    }
    return psi;
}

itensor::MPO
ChoiPropagator::outputHamiltonianMPO(itensor::SpinHalf const& sites,
                                       SchwingerParams const& p) {
    using namespace itensor;
    const int twoN = length(sites);
    if (twoN != 2 * p.N) {
        throw std::invalid_argument(
            "ChoiPropagator::outputHamiltonianMPO: doubled site set "
            "length must be 2 * params.N");
    }
    if (p.a <= 0.0) {
        throw std::invalid_argument(
            "ChoiPropagator::outputHamiltonianMPO: params.a must be positive");
    }

    auto ampo = AutoMPO(sites);

    // Map: physical out-site n  →  doubled-chain site index 2n.
    auto out = [](int n) { return 2 * n; };

    // ── Hopping: (1/(2a)) Σ_{n=1..N-1} (σ⁺_n σ⁻_{n+1} + σ⁻_n σ⁺_{n+1})
    // on the out-register. On the doubled chain the pair (out_n, out_{n+1})
    // = (site 2n, site 2n+2) is range-2.
    {
        const double t = 0.5 / p.a;
        for (int n = 1; n <= p.N - 1; ++n) {
            ampo += t, "S+", out(n), "S-", out(n + 1);
            ampo += t, "S-", out(n), "S+", out(n + 1);
        }
    }

    // ── Mass: m Σ_n (-1)^n "Sz"_{out_n}
    if (p.m != 0.0) {
        for (int n = 1; n <= p.N; ++n) {
            const double sign = (n % 2 == 0) ? +1.0 : -1.0;
            ampo += p.m * sign, "Sz", out(n);
        }
    }

    // ── Electric-field operator part: the same L_n² expansion as the
    // standard Schwinger MPO, with out_k = site 2k:
    //   H_E_op = − Σ_{k=1..N-1} (g² a A_k) "Sz"_{out_k}
    //          + Σ_{1 ≤ j < k ≤ N-1} (g² a (N−k)) "Sz"_{out_j} "Sz"_{out_k}.
    {
        const double Eg = p.g * p.g * p.a;

        for (int k = 1; k <= p.N - 1; ++k) {
            const double Ak = tailSumC(k, p.N, p.L0);
            if (Ak != 0.0) ampo += -Eg * Ak, "Sz", out(k);
        }

        for (int k = 2; k <= p.N - 1; ++k) {
            const double w = Eg * static_cast<double>(p.N - k);
            if (w == 0.0) continue;
            for (int j = 1; j < k; ++j) {
                ampo += w, "Sz", out(j), "Sz", out(k);
            }
        }
    }

    return toMPO(ampo);
}

itensor::MPS
ChoiPropagator::evolve(itensor::MPS psi,
                        itensor::MPO const& H,
                        double duration,
                        TDVPSettings const& s) {
    using namespace itensor;
    if (duration < 0.0) {
        throw std::invalid_argument(
            "ChoiPropagator::evolve: duration must be non-negative");
    }
    if (s.dt <= 0.0) {
        throw std::invalid_argument(
            "ChoiPropagator::evolve: dt must be positive");
    }

    if (duration == 0.0) return psi;

    auto sweepsTdvp = Sweeps(1);
    sweepsTdvp.maxdim() = s.maxBondDim;
    sweepsTdvp.cutoff() = s.cutoff;
    sweepsTdvp.niter()  = s.krylovDim;

    const Cplx tStep{0.0, -s.dt};
    const auto args = Args(
        "Truncate",     true,
        "DoNormalize",  true,
        "Silent",       s.quiet,
        "NumCenter",    2,
        "ErrGoal",      1e-7);

    const int nSteps = static_cast<int>(std::round(duration / s.dt));
    for (int step = 0; step < nSteps; ++step) {
        TDVPIntegrator::evolve(psi, H, tStep, sweepsTdvp, args);
    }
    return psi;
}

itensor::MPS
ChoiPropagator::choiState(SchwingerParams const& p,
                           double duration,
                           TDVPSettings const& settings) {
    auto sites = doubledSites(p.N);
    auto psi   = bellChainMPS(sites);
    auto H     = outputHamiltonianMPO(sites, p);
    return evolve(std::move(psi), H, duration, settings);
}

Eigen::MatrixXd
ChoiPropagator::temporalMutualInformation(itensor::MPS const& choi, int N) {
    using namespace itensor;
    const int twoN = length(choi);
    if (twoN != 2 * N) {
        throw std::invalid_argument(
            "ChoiPropagator::temporalMutualInformation: choi length "
            "must be 2 * N");
    }

    Eigen::MatrixXd out = Eigen::MatrixXd::Zero(N, N);
    if (N < 1) return out;

    // Cache single-site entropies of every doubled-chain site so each
    // pair only triggers one extra (2-site) RDM call.
    std::vector<double> Sin(static_cast<std::size_t>(N + 1), 0.0);   // 1-based: Sin[i] = S(ρ_{in_i})
    std::vector<double> Sout(static_cast<std::size_t>(N + 1), 0.0);  //          Sout[j] = S(ρ_{out_j})
    for (int i = 1; i <= N; ++i) {
        const int siteIn  = 2 * i - 1;
        const int siteOut = 2 * i;
        Sin[static_cast<std::size_t>(i)] =
            MutualInformation::vonNeumannEntropy(
                MutualInformation::oneSiteReducedDensity(choi, siteIn));
        Sout[static_cast<std::size_t>(i)] =
            MutualInformation::vonNeumannEntropy(
                MutualInformation::oneSiteReducedDensity(choi, siteOut));
    }

    // I({in_i} : {out_j}) = S(ρ_{in_i}) + S(ρ_{out_j}) − S(ρ_{in_i, out_j}).
    // (in_i, out_j) on the doubled chain is sites (2i-1, 2j); always
    // i_site < j_site for i ≤ j.
    for (int i = 1; i <= N; ++i) {
        for (int j = 1; j <= N; ++j) {
            const int siteIn  = 2 * i - 1;
            const int siteOut = 2 * j;
            // twoSiteReducedDensity requires its first site index to
            // be the smaller one. siteIn = 2i-1 and siteOut = 2j, so
            // siteIn < siteOut iff i ≤ j; for i > j the two are swapped,
            // which is harmless because ρ_{in_i, out_j} and
            // ρ_{out_j, in_i} carry the same information.
            int lo = std::min(siteIn, siteOut);
            int hi = std::max(siteIn, siteOut);
            const auto rho =
                MutualInformation::twoSiteReducedDensity(choi, lo, hi);
            const double Sij = MutualInformation::vonNeumannEntropy(rho);
            const double I   = Sin[static_cast<std::size_t>(i)] +
                                Sout[static_cast<std::size_t>(j)] - Sij;
            out(i - 1, j - 1) = I;
        }
    }
    return out;
}

} // namespace tessera::quantum
