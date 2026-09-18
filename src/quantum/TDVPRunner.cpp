// Implementation of SchwingerQuench — the q-qbar quench + TDVP pipeline
// and the causal-order comparison. See include/quantum/TDVPRunner.hpp
// for the pipeline description.

#include "quantum/TDVPRunner.hpp"

#include "quantum/CausalCompare.hpp"
#include "quantum/MutualInformation.hpp"
#include "quantum/Quench.hpp"
#include "quantum/SchwingerModel.hpp"

#include "quantum/TDVPIntegrator.hpp"  // two-site TDVP on ITensor core primitives

#include <itensor/all.h>

#include <algorithm>
#include <cmath>

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

// ─── Observable helpers ───────────────────────────────────────────────────

// ⟨σ^z_n⟩ for every site n = 1..N: orthogonalise at site n, contract
// bra × Sz × ket, and multiply by 2 to convert ITensor's "Sz" (= ½ σ^z)
// to bare σ^z.
std::vector<double> sigmaZProfile(itensor::MPS const& psi_in,
                                    itensor::SpinHalf const& sites) {
    using namespace itensor;
    auto psi = psi_in;
    const int N = length(psi);
    std::vector<double> out(static_cast<std::size_t>(N), 0.0);
    for (int n = 1; n <= N; ++n) {
        psi.position(n);
        auto Sz  = op(sites, "Sz", n);
        auto bra = dag(psi(n));
        bra.prime("Site");
        const auto val = eltC(bra * Sz * psi(n));
        out[static_cast<std::size_t>(n - 1)] = 2.0 * val.real();
    }
    return out;
}

// ⟨L_n⟩ for every link n = 1..N-1 from the σ^z profile:
//   ⟨L_n⟩  =  c_n  -  ½ Σ_{k=1..n} ⟨σ^z_k⟩
// with c_n = L0 + ((-1)^n - 1)/4, matching SchwingerModel.cpp.
std::vector<double> lProfileFromSz(std::vector<double> const& sz, double L0) {
    const int N = static_cast<int>(sz.size());
    std::vector<double> L(static_cast<std::size_t>(std::max(N - 1, 0)), 0.0);
    double cum = 0.0;
    for (int n = 1; n <= N - 1; ++n) {
        cum += sz[static_cast<std::size_t>(n - 1)];
        const double c_n = L0 + ((n % 2 == 0) ? 0.0 : -0.5);
        L[static_cast<std::size_t>(n - 1)] = c_n - 0.5 * cum;
    }
    return L;
}

double computeEnergy(itensor::MPS const& psi, itensor::MPO const& H,
                     double constantShift) {
    return std::real(itensor::innerC(psi, H, psi)) + constantShift;
}

TDVPSnapshot makeSnapshot(double t,
                           itensor::MPS const& psi,
                           SchwingerMPO const& sm,
                           SchwingerParams const& p,
                           bool recordSpectra,
                           bool recordPoset,
                           bool recordMutualInformation,
                           bool recordBondMutualInformation) {
    TDVPSnapshot snap;
    snap.time      = t;
    snap.energy    = computeEnergy(psi, sm.H, sm.constant);
    snap.bondDim  = itensor::maxLinkDim(psi);
    snap.zProfile = sigmaZProfile(psi, sm.sites);
    snap.lProfile = lProfileFromSz(snap.zProfile, p.L0);
    if (recordSpectra || recordPoset) {
        snap.spectra = Schmidt::allOf(psi);
        if (recordPoset) {
            snap.poset = Majorization::posetOf(snap.spectra.spectra);
        }
    }
    if (recordMutualInformation) {
        // All-pairs site-site mutual information, flattened row-major
        // for the Python side to rebuild.
        auto mi = MutualInformation::allPairs(psi);
        const int N = static_cast<int>(mi.rows());
        snap.mutualInformation.assign(static_cast<std::size_t>(N) * N, 0.0);
        for (int a = 0; a < N; ++a) {
            for (int b = 0; b < N; ++b) {
                snap.mutualInformation[static_cast<std::size_t>(a * N + b)] =
                    mi(a, b);
            }
        }
    }
    if (recordBondMutualInformation) {
        // All-pairs bond-cut tripartite info. Flatten (N-1) × (N-1)
        // matrix to row-major.
        auto bm = MutualInformation::allBondPairs(psi);
        const int B = static_cast<int>(bm.rows());
        snap.bondMutualInformation.assign(
            static_cast<std::size_t>(B) * B, 0.0);
        for (int a = 0; a < B; ++a) {
            for (int b = 0; b < B; ++b) {
                snap.bondMutualInformation[static_cast<std::size_t>(a * B + b)] =
                    bm(a, b);
            }
        }
    }
    return snap;
}

itensor::Sweeps makeDmrgSweeps(int maxBond, int nSweeps,
                                int krylov, double cutoff) {
    auto sweeps = itensor::Sweeps(nSweeps);
    sweeps.maxdim() = std::min(20, maxBond),
                      std::min(40, maxBond),
                      std::min(80, maxBond),
                      maxBond, maxBond;
    sweeps.cutoff() = cutoff;
    sweeps.niter()  = krylov;
    sweeps.noise()  = 1e-7, 1e-8, 0.0;
    return sweeps;
}

itensor::Sweeps makeTdvpSweeps(int maxBond, int krylov, double cutoff) {
    auto sweeps = itensor::Sweeps(1);
    sweeps.maxdim() = maxBond;
    sweeps.cutoff() = cutoff;
    sweeps.niter()  = krylov;
    return sweeps;
}

itensor::MPS neelInit(itensor::SpinHalf const& sites, int N) {
    auto state = itensor::InitState(sites);
    for (int i = 1; i <= N; ++i) {
        state.set(i, (i % 2 == 1) ? "Up" : "Dn");
    }
    return itensor::MPS(state);
}

} // namespace

SchwingerQuench::SchwingerQuench(TDVPConfig config) noexcept
    : config_(config) {}

QuenchResult SchwingerQuench::evolve() const {
    auto const& cfg = config_;

    // (1) Build the Schwinger MPO and run DMRG to the GS.
    SchwingerParams p;
    p.N = cfg.N; p.a = cfg.a; p.m = cfg.m; p.g = cfg.g; p.L0 = cfg.L0;

    auto sm = cfg.hoppingPairs.empty()
        ? SchwingerHamiltonian{p}.mpo(cfg.conserveQns)
        : SchwingerHamiltonian{p}.mpoChain(cfg.hoppingPairs, cfg.conserveQns);
    auto psi0 = neelInit(sm.sites, cfg.N);
    auto sweepsDmrg = makeDmrgSweeps(
        cfg.dmrgMaxBondDim, cfg.dmrgNSweeps,
        cfg.dmrgKrylovDim, cfg.dmrgCutoff);
    auto [E_gs, psiGs] = itensor::dmrg(
        sm.H, psi0, sweepsDmrg,
        itensor::Args("Silent", cfg.quiet));

    QuenchResult result;
    result.groundState.operatorEnergy = E_gs;
    result.groundState.constant        = sm.constant;
    result.groundState.energy          = E_gs + sm.constant;
    result.groundState.bondDim        = itensor::maxLinkDim(psiGs);
    result.groundState.truncationErr  = cfg.dmrgCutoff;

    // (2) Apply the q-qbar quench, producing the string state.
    auto psi = QqbarQuench{cfg.i0, cfg.d, cfg.quenchEnforceParity}
                   .apply(psiGs, sm.sites);

    // (3) Initial snapshot. t = 0 is immediately after the quench,
    // before any TDVP step; its energy sits above the ground-state
    // energy by the quench excitation cost.
    result.snapshots.push_back(makeSnapshot(
        /*t=*/0.0, psi, sm, p,
        cfg.recordSpectra, cfg.recordPoset,
        cfg.recordMutualInformation,
        cfg.recordBondMutualInformation));

    // (4) TDVP loop. Real-time evolution e^{-i H Δt} corresponds to
    // TDVPIntegrator::evolve with the time argument t = -i Δt.
    auto sweepsTdvp = makeTdvpSweeps(
        cfg.maxBondDim, cfg.krylovDim, cfg.cutoff);
    const itensor::Cplx tStep{0.0, -cfg.dt};
    const auto tdvpArgs = itensor::Args(
        "Truncate",     true,
        "DoNormalize",  true,
        "Silent",       cfg.quiet,
        "NumCenter",    2,
        "ErrGoal",      1e-7);

    const int nSteps = static_cast<int>(std::round(cfg.T / cfg.dt));
    for (int step = 1; step <= nSteps; ++step) {
        TDVPIntegrator::evolve(psi, sm.H, tStep, sweepsTdvp, tdvpArgs);
        if (step % cfg.snapshotEvery == 0 || step == nSteps) {
            const double currentT = step * cfg.dt;
            result.snapshots.push_back(makeSnapshot(
                currentT, psi, sm, p,
                cfg.recordSpectra, cfg.recordPoset,
                cfg.recordMutualInformation,
                cfg.recordBondMutualInformation));
        }
    }

    return result;
}

CausalComparisonReport SchwingerQuench::compareCausalOrders(
    double vLr,
    MajorizationPredicate const* predicate) const
{
    // The cross-time majorization poset needs the spectra, so force
    // recordSpectra on. recordPoset stays off: the per-(time, cut)
    // posets are built below instead.
    TDVPConfig cfg = config_;
    cfg.recordSpectra = true;
    cfg.recordPoset   = false;

    const auto quench = SchwingerQuench{cfg}.evolve();
    auto orders = CausalOrders::fromSnapshots(
        quench.snapshots, vLr, predicate);

    StandardMajorization defaultPredicate{1e-12};
    MajorizationPredicate const& effectivePredicate =
        predicate ? *predicate : defaultPredicate;

    CausalComparisonReport report;
    report.nLabels    = static_cast<int>(orders.labels.size());
    report.nSnapshots = static_cast<int>(quench.snapshots.size());
    report.vLr        = vLr;
    report.majKind    = effectivePredicate.name();
    report.majVsLr    = Majorization::agreement(orders.maj, orders.lr, report.nLabels);
    report.majVsCs    = Majorization::agreement(orders.maj, orders.cs, report.nLabels);
    report.lrVsCs     = Majorization::agreement(orders.lr,  orders.cs, report.nLabels);
    return report;
}

} // namespace tessera::quantum
