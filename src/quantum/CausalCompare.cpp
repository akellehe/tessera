// Implementation of CausalOrders::fromSnapshots — the cross-time poset
// construction. See include/quantum/CausalCompare.hpp for the
// definitions of the three orders.

#include "quantum/CausalCompare.hpp"
#include "quantum/TDVPRunner.hpp"   // full definition of TDVPSnapshot

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

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

// Closest-pair distance between contiguous intervals A = [i1, j1] and
// B = [i2, j2]. Returns 0 if they overlap, otherwise the integer gap
// between the disjoint ranges.
int intervalDistance(int i1, int j1, int i2, int j2) {
    if (j1 < i2) return i2 - j1;
    if (j2 < i1) return i1 - j2;
    return 0;
}

// Build a Poset from a directed boolean adjacency matrix `strict[i][j]`
// of strict-precedes relations. Applies transitive reduction so the
// returned Poset has Hasse cover edges only.
Poset posetFromStrict(std::vector<std::vector<char>> const& strict) {
    const int n = static_cast<int>(strict.size());
    Poset p(n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (!strict[static_cast<std::size_t>(i)]
                       [static_cast<std::size_t>(j)]) continue;
            bool hasIntermediate = false;
            for (int k = 0; k < n; ++k) {
                if (k == i || k == j) continue;
                if (strict[static_cast<std::size_t>(i)]
                          [static_cast<std::size_t>(k)] &&
                    strict[static_cast<std::size_t>(k)]
                          [static_cast<std::size_t>(j)]) {
                    hasIntermediate = true;
                    break;
                }
            }
            if (!hasIntermediate) p.addCover(i, j);
        }
    }
    return p;
}

// Aggregate every (cut, time) into a flat label list and collect the
// corresponding spectra into a flat vector-of-vectors, so
// Majorization::posetOf is called once.
struct Flattened {
    std::vector<LabelSpacetime>          labels;
    std::vector<std::vector<double>>     spectra;
};

Flattened flattenSnapshots(std::vector<TDVPSnapshot> const& snapshots) {
    Flattened out;
    for (int tIdx = 0;
         tIdx < static_cast<int>(snapshots.size()); ++tIdx) {
        auto const& snap = snapshots[static_cast<std::size_t>(tIdx)];
        if (snap.spectra.spectra.empty()) {
            throw std::runtime_error(
                "CausalOrders::fromSnapshots: snapshots must have "
                "recordSpectra=true");
        }
        for (int k = 0;
             k < static_cast<int>(snap.spectra.spectra.size()); ++k) {
            out.labels.push_back({k, tIdx,
                snap.spectra.intervals[static_cast<std::size_t>(k)].i,
                snap.spectra.intervals[static_cast<std::size_t>(k)].j,
                snap.time});
            out.spectra.push_back(
                snap.spectra.spectra[static_cast<std::size_t>(k)]);
        }
    }
    return out;
}

// Lieb-Robinson cone Hasse poset:
//   (a, b) is a strict-LR edge iff
//       labels[a].time < labels[b].time AND
//       intervalDistance(A, B)  ≤  vLr · (labels[b].time - labels[a].time)
Poset buildLrPoset(std::vector<LabelSpacetime> const& labels, double vLr) {
    const int n = static_cast<int>(labels.size());
    std::vector<std::vector<char>> strict(static_cast<std::size_t>(n),
                                          std::vector<char>(
                                              static_cast<std::size_t>(n), 0));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j) continue;
            const auto& A = labels[static_cast<std::size_t>(i)];
            const auto& B = labels[static_cast<std::size_t>(j)];
            if (A.time >= B.time) continue;
            const double dt = B.time - A.time;
            const int    d  = intervalDistance(A.intervalI, A.intervalJ,
                                                B.intervalI, B.intervalJ);
            if (static_cast<double>(d) <= vLr * dt) {
                strict[static_cast<std::size_t>(i)]
                      [static_cast<std::size_t>(j)] = 1;
            }
        }
    }
    return posetFromStrict(strict);
}

// Regular-chain causal-set Hasse poset:
//   (a, b) is strict iff labels[a].tIdx < labels[b].tIdx.
Poset buildCsPosetRegularChain(std::vector<LabelSpacetime> const& labels) {
    const int n = static_cast<int>(labels.size());
    std::vector<std::vector<char>> strict(static_cast<std::size_t>(n),
                                          std::vector<char>(
                                              static_cast<std::size_t>(n), 0));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (labels[static_cast<std::size_t>(i)].tIdx
              < labels[static_cast<std::size_t>(j)].tIdx) {
                strict[static_cast<std::size_t>(i)]
                      [static_cast<std::size_t>(j)] = 1;
            }
        }
    }
    return posetFromStrict(strict);
}

} // namespace

CausalOrders CausalOrders::fromSnapshots(
    std::vector<TDVPSnapshot> const& snapshots,
    double vLr,
    MajorizationPredicate const* predicate)
{
    auto flat = flattenSnapshots(snapshots);

    // Materialize a default StandardMajorization{1e-12} when no
    // predicate is supplied. Its lifetime ends with this function, which
    // suffices: Majorization::posetOf only needs it during the call.
    StandardMajorization defaultPredicate{1e-12};
    MajorizationPredicate const& effectivePredicate =
        predicate ? *predicate : defaultPredicate;

    CausalOrders out;
    out.labels = std::move(flat.labels);
    out.maj = Majorization::posetOf(flat.spectra, effectivePredicate);
    out.lr  = buildLrPoset(out.labels, vLr);
    out.cs  = buildCsPosetRegularChain(out.labels);
    return out;
}

} // namespace tessera::quantum
