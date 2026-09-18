// Implementation of SchwingerModel — see include/quantum/DMRGRunner.hpp
// for the design.

#include "quantum/DMRGRunner.hpp"

#include <itensor/all.h>

#include <algorithm>

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

// Néel |↑↓↑↓…⟩ initial state. For even N the total Sz is 0, the
// charge-neutral sector. With ConserveQNs=true on the SiteSet, DMRG
// stays in that sector.
itensor::MPS neelInit(itensor::SpinHalf const& sites, int N) {
    auto state = itensor::InitState(sites);
    for (int i = 1; i <= N; ++i) {
        state.set(i, (i % 2 == 1) ? "Up" : "Dn");
    }
    return itensor::MPS(state);
}

// Sweep schedule for ITensor::dmrg. The bond-dimension cap ramps
// 20 → 40 → 80 → max over the first sweeps, so early iterations do not
// commit truncation errors that later sweeps must undo. Noise is on for
// the first two sweeps to escape local minima, then off so the later
// sweeps converge cleanly.
itensor::Sweeps makeSweeps(QuantumConfig const& cfg) {
    auto sweeps = itensor::Sweeps(cfg.nSweeps);
    const int b = cfg.maxBondDim;
    sweeps.maxdim() = std::min(20, b),
                      std::min(40, b),
                      std::min(80, b),
                      b, b;
    sweeps.cutoff() = cfg.cutoff;
    sweeps.niter()  = cfg.krylovDim;
    sweeps.noise()  = 1e-7, 1e-8, 0.0;
    return sweeps;
}

// Run DMRG and package the diagnostics. The optimized MPS is returned
// alongside so callers that want Schmidt spectra can reuse it.
struct DmrgRun {
    GroundStateResult result;
    itensor::MPS      psi;
    SchwingerMPO      mpo;
};

DmrgRun runDmrg(QuantumConfig const& cfg) {
    SchwingerParams p;
    p.N = cfg.N; p.a = cfg.a; p.m = cfg.m; p.g = cfg.g; p.L0 = cfg.L0;

    auto sm = SchwingerHamiltonian{p}.mpo(cfg.conserveQns);
    auto psi0 = neelInit(sm.sites, p.N);

    auto sweeps = makeSweeps(cfg);
    auto [energy, psi] = itensor::dmrg(
        sm.H, psi0, sweeps,
        itensor::Args("Silent", cfg.quiet));

    GroundStateResult r;
    r.operatorEnergy = energy;
    r.constant        = sm.constant;
    r.energy          = energy + sm.constant;
    r.bondDim        = itensor::maxLinkDim(psi);
    r.truncationErr  = cfg.cutoff;

    return {r, std::move(psi), std::move(sm)};
}

} // namespace

SchwingerModel::SchwingerModel(QuantumConfig config) noexcept
    : config_(config) {}

GroundStateResult SchwingerModel::solve() const {
    return runDmrg(config_).result;
}

GroundStateMajorizationResult
SchwingerModel::solveWithMajorization(double tol) const {
    auto run = runDmrg(config_);

    GroundStateMajorizationResult out;
    out.groundState = run.result;

    // All contiguous-cut Schmidt spectra. At N ≤ 20 this adds well under
    // a second to the DMRG run.
    out.spectra = Schmidt::allOf(run.psi);

    // Majorization poset (Hasse cover edges only).
    out.poset = Majorization::posetOf(out.spectra.spectra, tol);

    return out;
}

} // namespace tessera::quantum
