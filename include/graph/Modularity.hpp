// Copyright (c) 2026 Twin Vector Labs LLC. All rights reserved.
//
// The one summation order for modularity, shared by
// Spacetime::modularityOnSkeleton, SparseGraph::modularity and
// PersistentModularity::modularityGamma (on a nonnegative real graph), so the
// three give the same bits on the same graph and partition. Each caller
// accumulates two totals per community, its internal weight Sigma_in(c) and
// its strength S_c, and hands them over in ascending community label; the
// terms Sigma_in(c)/2m - gamma (S_c/2m)^2 are then added with Kahan
// compensation. At unit weights every total is an integer, exact in double
// whatever order it accumulates in, so the order of the community terms is
// the only rounding left, and it is fixed here.
//
// Reference: Newman and Girvan, "Finding and evaluating community structure in
// networks", arXiv:cond-mat/0308217

#pragma once

#include <algorithm>
#include <unordered_map>
#include <utility>
#include <vector>

// === tessera subsystem ns fwd-decls ===
namespace tessera::mesh {}
namespace tessera::observables {}
namespace tessera::quantum {}
namespace tessera::simulations {}
namespace tessera::spacetime {}
namespace tessera::graph {
using namespace ::tessera::mesh;
using namespace ::tessera::spacetime;
using namespace ::tessera::observables;
using namespace ::tessera::simulations;
using namespace ::tessera::quantum;

/// Kahan-compensated running sum: the rounding error of each addition is
/// carried into the next, so the total does not drift with the term count.
class Kahan {
public:
    void add(double x) noexcept {
        const double y = x - c_;
        const double t = s_ + y;
        c_ = (t - s_) - y;
        s_ = t;
    }
    [[nodiscard]] double value() const noexcept { return s_; }

private:
    double s_ = 0.0;
    double c_ = 0.0;
};

/// One community's modularity totals.
struct CommunityTotals {
    /// \f$ \Sigma_{in}(c) \f$: the adjacency weight inside the community,
    /// each internal edge counted from both ends (2 per unit edge).
    double internal = 0.0;
    /// \f$ S_c \f$: the summed strength (weighted degree) of its nodes.
    double strength = 0.0;
};

/// ``(label, totals)`` per community; ``modularityInLabelOrder`` takes them in
/// ascending label order.
using LabelledCommunities = std::vector<std::pair<int, CommunityTotals>>;

/// Sort communities into ascending label order for ``modularityInLabelOrder``.
/// Labels are distinct, so the order is total.
inline void sortByLabel(LabelledCommunities &communities) {
    std::sort(communities.begin(), communities.end(),
              [](const auto &a, const auto &b) { return a.first < b.first; });
}

/// The communities of a hashed per-label table, in ascending label order.
[[nodiscard]] inline LabelledCommunities
sortedByLabel(const std::unordered_map<int, CommunityTotals> &byLabel) {
    LabelledCommunities out(byLabel.begin(), byLabel.end());
    sortByLabel(out);
    return out;
}

/// Generalized modularity
/// \f$ Q_\gamma = \sum_c \left[ \Sigma_{in}(c)/2m - \gamma (S_c/2m)^2 \right] \f$
/// from per-community totals, with \f$ 2m \f$ = ``twoM`` the total strength.
///
/// ``communities`` iterates ``(label, CommunityTotals)`` pairs, a
/// ``LabelledCommunities`` or a ``std::map<int, CommunityTotals>``, holding
/// every label that some node carries, in ascending label order. The terms
/// are added in that order with Kahan compensation; a different order gives
/// a value that can differ in the last bits.
template <typename Communities>
[[nodiscard]] double modularityInLabelOrder(const Communities &communities,
                                            double twoM,
                                            double gamma = 1.0) noexcept {
    Kahan q;
    for (const auto &entry : communities) {
        const CommunityTotals &c = entry.second;
        const double frac = c.strength / twoM;
        q.add(c.internal / twoM - gamma * frac * frac);
    }
    return q.value();
}

} // namespace tessera::graph
