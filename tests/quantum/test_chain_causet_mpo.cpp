// Chain-causet acceptance — `SchwingerHamiltonian::mpoChain` reproduces
// `SchwingerHamiltonian::mpo` exactly on a chain causet (one vertex
// per time slice).
//
// The test does two equivalence checks plus a robustness probe:
//
//   (1) Synthetic chain-NN hopping list, by hand: pairs (0,1), (1,2), …
//       The chain MPO ground-state energy should equal the standard
//       MPO's to ~1e-12 (DMRG-side numerical noise).
//
//   (2) Spacetime-derived chain causet via Causet::chainFrom. We build
//       a 1+1D toy spacetime where each time slice has a single vertex
//       linked to the next slice's vertex by a timelike edge.
//       Causet::chainFrom returns hoppingPairs = [(0,1), (1,2), …]
//       and the MPO from those pairs should ALSO equal the standard
//       MPO.
//
//   (3) Dense-Hamiltonian agreement at small N (N=6): the chain-causet
//       MPO's lowest eigenvalue (in the Sz=0 sector) matches the dense
//       reference's lowest Sz=0 eigenvalue to ~1e-10. This is the
//       same correctness criterion that `test_schwinger_spectrum.cpp`
//       applies to the standard MPO; we just route through the new
//       API instead.
//
// If any of these fail, the chain-causet path is no longer a sanity
// reproduction of the regular Schwinger model and the multi-vertex-
// antichain extension shouldn't be trusted on top of it.

#include "Poset.h"
#include "quantum/CausetChain.hpp"
#include "quantum/SchwingerModel.hpp"
#include "spacetime/Spacetime.h"

#include <itensor/all.h>
#include <Eigen/Dense>

#include <complex>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <utility>
#include <vector>

using namespace tessera::quantum;
using itensor::dmrg;
using itensor::Sweeps;
using itensor::MPS;
using itensor::InitState;

namespace {

// Spacetime::createEdge takes the complex edge length l, not the squared
// length: Edge::isTimelike() reads the argument of l^2, so a timelike edge
// of unit magnitude is l = sqrt(-1) = i and a spacelike one is l = 1.
constexpr std::complex<double> kTimelikeLength{0.0, 1.0};
constexpr std::complex<double> kSpacelikeLength{1.0, 0.0};

// Réel Néel initial state for the Sz=0 sector.
MPS neel_init(itensor::SpinHalf const& sites, int N) {
    auto state = InitState(sites);
    for (int i = 1; i <= N; ++i)
        state.set(i, (i % 2 == 1) ? "Up" : "Dn");
    return MPS(state);
}

double dmrg_gs_energy(SchwingerMPO const& sm,
                      int max_bond = 64,
                      int nSweeps = 10) {
    auto psi0 = neel_init(sm.sites, sm.params.N);
    auto sweeps = Sweeps(nSweeps);
    sweeps.maxdim() = 20, 40, 80, max_bond, max_bond;
    sweeps.cutoff() = 1e-12;
    sweeps.niter()  = 4;
    sweeps.noise()  = 1e-7, 1e-8, 0.0;
    auto [E, psi] = dmrg(sm.H, psi0, sweeps, {"Silent=", true});
    return E;
}

double lowest_eigenvalue_sz0(Eigen::MatrixXd const& H, int N) {
    const Eigen::Index dim = H.rows();
    const int half = N / 2;
    std::vector<Eigen::Index> idx;
    for (Eigen::Index s = 0; s < dim; ++s) {
        if (std::popcount(static_cast<std::uint64_t>(s))
            == static_cast<unsigned>(half)) {
            idx.push_back(s);
        }
    }
    const Eigen::Index k = static_cast<Eigen::Index>(idx.size());
    Eigen::MatrixXd Hsub(k, k);
    for (Eigen::Index i = 0; i < k; ++i)
        for (Eigen::Index j = 0; j < k; ++j)
            Hsub(i, j) = H(idx[i], idx[j]);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Hsub);
    return es.eigenvalues()(0);
}

bool acceptance_synthetic_chain_pairs() {
    std::cout << "Acceptance #1 — synthetic NN hopping list\n";

    SchwingerParams p;
    p.N = 8; p.a = 1.0; p.g = 1.0; p.m = 0.5; p.L0 = 0.0;

    SchwingerHamiltonian H{p};
    auto std_mpo = H.mpo();
    std::vector<std::pair<int, int>> nn_pairs;
    for (int n = 0; n < p.N - 1; ++n) nn_pairs.emplace_back(n, n + 1);
    auto chain_mpo = H.mpoChain(nn_pairs);

    const double E_std   = dmrg_gs_energy(std_mpo);
    const double E_chain = dmrg_gs_energy(chain_mpo);
    const double diff    = std::abs(E_std - E_chain);
    const bool ok = diff < 1e-9;

    std::cout << "  N=" << p.N << " m/g=" << p.m / p.g
              << " L0=" << p.L0
              << "  E_std=" << E_std
              << "  E_chain=" << E_chain
              << "  |Δ|=" << diff
              << "  " << (ok ? "PASS" : "FAIL") << "\n";
    return ok;
}

bool acceptance_spacetime_chain_extraction() {
    std::cout << "Acceptance #2 — Spacetime-derived chain causet\n";

    // Build a 1D chain spacetime: each time slice has one vertex,
    // linked to the next by a timelike edge. Mimics what a properly-
    // built `tessera::CausalSet` chain would produce. Use 8 slices for
    // N=8 lattice sites total.
    constexpr int N = 8;
    tessera::spacetime::Spacetime st;
    std::vector<tessera::mesh::VertexPtr> verts;
    verts.reserve(static_cast<std::size_t>(N));
    for (int t = 0; t < N; ++t) {
        verts.push_back(st.createVertex(static_cast<std::uint64_t>(t),
            std::vector<double>{static_cast<double>(t)}));
    }
    for (int t = 0; t + 1 < N; ++t) {
        st.createEdge(verts[static_cast<std::size_t>(t)],
                      verts[static_cast<std::size_t>(t + 1)],
                      kTimelikeLength);
    }

    auto chain = tessera::quantum::Causet::chainFrom(st);
    if (chain.nSites != N) {
        std::cout << "  FAIL: chain.nSites=" << chain.nSites
                  << " expected " << N << "\n";
        return false;
    }

    SchwingerParams p;
    p.N = chain.nSites; p.a = 1.0; p.g = 1.0; p.m = 0.5; p.L0 = 0.0;

    SchwingerHamiltonian H{p};
    auto std_mpo   = H.mpo();
    auto chain_mpo = H.mpoChain(chain.hoppingPairs);

    const double E_std   = dmrg_gs_energy(std_mpo);
    const double E_chain = dmrg_gs_energy(chain_mpo);
    const double diff    = std::abs(E_std - E_chain);
    const bool ok = diff < 1e-9;

    std::cout << "  N=" << p.N
              << "  E_std=" << E_std
              << "  E_chain=" << E_chain
              << "  |Δ|=" << diff
              << "  " << (ok ? "PASS" : "FAIL") << "\n";
    return ok;
}

bool acceptance_dense_agreement() {
    std::cout << "Acceptance #3 — chain MPO matches dense Sz=0 eigenvalue\n";

    SchwingerParams p;
    p.N = 6; p.a = 1.0; p.g = 1.0; p.m = 0.25; p.L0 = 0.0;

    std::vector<std::pair<int, int>> nn_pairs;
    for (int n = 0; n < p.N - 1; ++n) nn_pairs.emplace_back(n, n + 1);
    SchwingerHamiltonian H{p};
    auto chain_mpo = H.mpoChain(nn_pairs);
    auto dense     = H.denseMatrix();

    const double E_chain_op = dmrg_gs_energy(chain_mpo);
    const double E_chain    = E_chain_op + chain_mpo.constant;
    const double E_dense    = lowest_eigenvalue_sz0(dense.H, p.N) + dense.constant;
    const double diff       = std::abs(E_chain - E_dense);
    const bool ok = diff < 1e-8;

    std::cout << "  N=" << p.N
              << "  E_chain=" << E_chain
              << "  E_dense=" << E_dense
              << "  |Δ|=" << diff
              << "  " << (ok ? "PASS" : "FAIL") << "\n";
    return ok;
}

bool acceptance_rejects_invalid_pairs() {
    std::cout << "Acceptance #4 — invalid hopping pairs rejected\n";

    SchwingerParams p;
    p.N = 4; p.a = 1.0; p.g = 1.0; p.m = 0.5; p.L0 = 0.0;

    bool ok = true;

    SchwingerHamiltonian H{p};

    // Out-of-range index.
    try {
        std::vector<std::pair<int, int>> bad{{0, 1}, {1, 99}};
        [[maybe_unused]] auto _ = H.mpoChain(bad);
        std::cout << "  FAIL: out-of-range index NOT rejected\n";
        ok = false;
    } catch (std::invalid_argument const&) { /* expected */ }

    // Self-loop.
    try {
        std::vector<std::pair<int, int>> bad{{0, 1}, {2, 2}};
        [[maybe_unused]] auto _ = H.mpoChain(bad);
        std::cout << "  FAIL: self-loop NOT rejected\n";
        ok = false;
    } catch (std::invalid_argument const&) { /* expected */ }

    std::cout << "  " << (ok ? "PASS" : "FAIL") << "\n";
    return ok;
}

} // namespace

int main() {
    bool ok = true;
    ok &= acceptance_synthetic_chain_pairs();
    ok &= acceptance_spacetime_chain_extraction();
    ok &= acceptance_dense_agreement();
    ok &= acceptance_rejects_invalid_pairs();
    std::cout << (ok ? "\nALL PASS\n" : "\nFAILURES\n");
    return ok ? 0 : 1;
}
