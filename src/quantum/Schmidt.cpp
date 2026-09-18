// Schmidt-spectrum extraction for contiguous-interval bipartitions of a
// matrix product state. See include/quantum/Schmidt.hpp for what is
// computed and why.

#include "quantum/Schmidt.hpp"

#include <itensor/all.h>
#include <itensor/decomp.h>

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

// Extract the non-increasing list of squared singular values from the
// diagonal "S" tensor returned by ITensor's three-argument svd().
// Squaring turns Schmidt values σ into density-matrix eigenvalues
// λ = σ², the convention Majorization::posetOf expects.
std::vector<double> diag_squared(itensor::ITensor const& S) {
    using namespace itensor;
    auto inds = S.inds();
    if (inds.size() != 2) {
        throw std::runtime_error("schmidt: expected S to have rank 2");
    }
    const int rank = std::min(inds(1).dim(), inds(2).dim());
    std::vector<double> out;
    out.reserve(static_cast<std::size_t>(rank));
    for (int k = 1; k <= rank; ++k) {
        const double sigma = elt(S, k, k);
        out.push_back(sigma * sigma);
    }
    std::sort(out.begin(), out.end(), std::greater<double>{});
    return out;
}

} // namespace

std::vector<double> Schmidt::of(itensor::MPS const& psi_in, int i, int j) {
    using namespace itensor;

    const int N = length(psi_in);
    if (i < 1 || j > N || i > j) {
        throw std::invalid_argument(
            "Schmidt::of: interval [i, j] must satisfy 1 ≤ i ≤ j ≤ N");
    }
    if (i == 1 && j == N) {
        // Trivial bipartition (whole chain | empty); the reduced density
        // matrix on the empty side has rank 1 with eigenvalue 1 (assuming
        // the input MPS is normalized).
        return {1.0};
    }

    // Bring the orthogonality center inside [i, j], at site i. Sites
    // 1..i-1 are then left-canonical and sites i+1..N right-canonical, so
    // contractions outside [i, j] collapse to identities and the reduced
    // density matrix on A depends only on the contracted T_i...T_j
    // tensor (see include/quantum/Schmidt.hpp).
    MPS psi = psi_in;
    psi.position(i);

    // Contract sites i..j into a single tensor M whose indices are
    //   (left bond α, site_i, site_{i+1}, …, site_j, right bond β)
    ITensor M = psi(i);
    for (int k = i + 1; k <= j; ++k) {
        M *= psi(k);
    }

    // Collect the site indices of [i, j] for the U-side of the SVD. The
    // V side then carries the bond indices (whatever's left).
    std::vector<Index> site_inds;
    site_inds.reserve(static_cast<std::size_t>(j - i + 1));
    for (int k = i; k <= j; ++k) {
        site_inds.push_back(siteIndex(psi, k));
    }

    // Cutoff truncation is disabled so the full spectrum is read,
    // including small but nonzero values the default SVD would drop.
    // MaxDim is generous; the actual rank is bounded by min(2^|A|, D²).
    auto args = Args("Cutoff", 0.0, "MaxDim", 1 << 24);
    auto [U, S, V] = svd(M, IndexSet(site_inds), args);

    return diag_squared(S);
}

SchmidtSpectra Schmidt::allOf(itensor::MPS const& psi) {
    const int N = itensor::length(psi);
    SchmidtSpectra out;
    out.N = N;

    // Reserve N(N+1)/2 - 1 slots: every contiguous interval [i, j] with
    // 1 ≤ i ≤ j ≤ N, minus the trivial full-chain cut.
    const int M = N * (N + 1) / 2 - 1;
    out.intervals.reserve(static_cast<std::size_t>(std::max(M, 0)));
    out.spectra.reserve(static_cast<std::size_t>(std::max(M, 0)));

    for (int i = 1; i <= N; ++i) {
        for (int j = i; j <= N; ++j) {
            if (i == 1 && j == N) continue;
            out.intervals.push_back({i, j});
            out.spectra.push_back(Schmidt::of(psi, i, j));
        }
    }
    return out;
}

} // namespace tessera::quantum
