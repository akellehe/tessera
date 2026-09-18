// Implementation of tessera::Poset / OrderAgreement / compareOrders. See
// include/Poset.h for the design.

#include "Poset.h"

#include "graph/IndexByKey.hpp"
#include "mesh/Edge.h"
#include "mesh/EdgeList.h"
#include "mesh/Vertex.h"
#include "mesh/VertexList.h"
#include "spacetime/Spacetime.h"

#include <algorithm>
#include <cstdint>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tessera {

namespace {

// Floyd-Warshall transitive closure on the cover-edge DAG. Returns
// adj[i][j] = true iff there's a directed path i → … → j in the strict
// precedes-cover direction.
std::vector<std::vector<char>>
transitive_closure(Poset const& p, int nLabels) {
    const int n = nLabels;
    std::vector<std::vector<char>> M(static_cast<std::size_t>(n),
                                     std::vector<char>(static_cast<std::size_t>(n),
                                                       0));
    for (auto const& [a, b] : p.covers()) {
        if (a < n && b < n && a >= 0 && b >= 0) {
            M[static_cast<std::size_t>(a)][static_cast<std::size_t>(b)] = 1;
        }
    }
    for (int k = 0; k < n; ++k) {
        for (int i = 0; i < n; ++i) {
            if (!M[static_cast<std::size_t>(i)][static_cast<std::size_t>(k)]) continue;
            for (int j = 0; j < n; ++j) {
                if (M[static_cast<std::size_t>(k)][static_cast<std::size_t>(j)]) {
                    M[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = 1;
                }
            }
        }
    }
    return M;
}

} // namespace

// ─── Poset ────────────────────────────────────────────────────────────────

Poset::Poset(int n_nodes_) {
    setNodeCount(n_nodes_);
}

Poset::Poset(Poset const& other) {
    setNodeCount(other.getNodeCount());
    setCovers(other.covers());
}

Poset& Poset::operator=(Poset const& other) {
    if (this != &other) {
        vertices_ = VertexList{};
        edges_ = EdgeList{};
        setNodeCount(other.getNodeCount());
        setCovers(other.covers());
    }
    return *this;
}

int Poset::getNodeCount() const noexcept {
    return static_cast<int>(vertices_.size());
}

void Poset::setNodeCount(int n) {
    if (n < 0) return;
    const int current = getNodeCount();
    for (int i = current; i < n; ++i) {
        vertices_.add(static_cast<std::uint64_t>(i));
    }
}

int Poset::getCoverCount() const noexcept {
    return static_cast<int>(edges_.size());
}

void Poset::addCover(int a, int b) {
    if (a < 0 || b < 0) {
        throw std::invalid_argument("Poset::addCover: node indices must be ≥ 0");
    }
    // Auto-resize so callers don't have to pre-call setNodeCount — the
    // causal-comparison code computes covers and node counts in one pass.
    const int needed = std::max(a, b) + 1;
    if (needed > getNodeCount()) setNodeCount(needed);
    auto* src = vertices_[static_cast<std::uint64_t>(a)];
    auto* dst = vertices_[static_cast<std::uint64_t>(b)];
    if (src == nullptr || dst == nullptr) {
        throw std::runtime_error("Poset::addCover: vertex lookup failed");
    }
    edges_.add(src, dst);
}

void Poset::setCovers(std::vector<std::pair<int, int>> const& new_covers) {
    edges_ = EdgeList{};
    for (auto const& [a, b] : new_covers) {
        addCover(a, b);
    }
}

std::vector<std::pair<int, int>> Poset::covers() const {
    std::vector<std::pair<int, int>> out;
    out.reserve(edges_.size());
    for (auto const* e : edges_.toVector()) {
        if (e == nullptr) continue;
        const auto src_id = e->getSource()->getId();
        const auto dst_id = e->getTarget()->getId();
        out.emplace_back(static_cast<int>(src_id), static_cast<int>(dst_id));
    }
    return out;
}

std::string Poset::toDot() const {
    std::ostringstream os;
    os << "digraph poset {\n";
    os << "  rankdir=BT;\n";
    os << "  node [shape=circle, style=filled, fillcolor=\"#eef\"];\n";
    for (int i = 0; i < getNodeCount(); ++i) os << "  " << i << ";\n";
    for (auto const& [a, b] : covers()) {
        os << "  " << a << " -> " << b << ";\n";
    }
    os << "}\n";
    return os.str();
}

Poset Poset::fromSpacetime(Spacetime const& st) {
    // Causet-adapter inheritance —
    // inherit a partial order
    // from a tessera::spacetime::Spacetime by treating each timelike edge as a strict
    // precedes-relation oriented earliest-time → latest-time, then
    // transitively reducing the resulting DAG to its Hasse covers.
    //
    // Vertex-ID convention: the Poset's integer node IDs are a dense
    // 0..n-1 remapping of the Spacetime's uint64_t vertex IDs in
    // ascending order. We don't preserve the raw uint64_t IDs because
    // (a) Poset's API is dense int-based and (b) Spacetime IDs may be
    // sparse after vertex deletions. The remapping is monotonic, so a
    // caller who sorts Spacetime vertex IDs ascending recovers the
    // Poset node order trivially.
    auto const& vlist = st.getVertexList();
    if (!vlist) return Poset{};

    auto const& live = vlist->liveVector();
    const int n = static_cast<int>(live.size());
    if (n == 0) return Poset{};

    // Build the dense remapping (spacetime uint64_t id) → (Poset int idx).
    // Sorting by raw ID gives a stable, deterministic node order.
    std::vector<std::uint64_t> id_in_order;
    id_in_order.reserve(static_cast<std::size_t>(n));
    for (auto* v : live) {
        if (v != nullptr) id_in_order.push_back(v->getId());
    }
    std::sort(id_in_order.begin(), id_in_order.end());
    auto id_to_idx = ::tessera::graph::indexByKey(
        id_in_order, [](auto id) { return id; });
    const int n_dense = static_cast<int>(id_in_order.size());

    // Collect strict-precedes pairs (earliest → latest) from timelike
    // edges. "Timelike" is the canonical Edge::isTimelike() classifier
    // (see include/mesh/Edge.h §EdgeDisposition; the superseded
    // sign-of-Re-squaredLength test is NOT used) AND we require a
    // strict time ordering between endpoints, since a timelike edge
    // with zero time difference would be a metric inconsistency we
    // don't want to silently propagate as a partial-order relation.
    auto const& elist = st.getEdgeList();
    std::vector<std::pair<int, int>> strict_pairs;
    if (elist) {
        auto const& edges = elist->toVector();
        strict_pairs.reserve(edges.size());
        for (auto const* e : edges) {
            if (e == nullptr) continue;
            auto const& src = e->getSource();
            auto const& dst = e->getTarget();
            if (!src || !dst) continue;
            if (!e->isTimelike()) continue;  // spacelike or null

            const double t_s = src->getTime();
            const double t_d = dst->getTime();
            if (t_s == t_d) continue;  // metric-inconsistent; skip defensively

            const std::uint64_t earlier_id = (t_s < t_d) ? src->getId() : dst->getId();
            const std::uint64_t later_id   = (t_s < t_d) ? dst->getId() : src->getId();
            auto it_e = id_to_idx.find(earlier_id);
            auto it_l = id_to_idx.find(later_id);
            if (it_e == id_to_idx.end() || it_l == id_to_idx.end()) continue;
            strict_pairs.emplace_back(it_e->second, it_l->second);
        }
    }

    // Floyd-Warshall transitive closure on the strict-precedes DAG.
    // The graph is acyclic by construction (every edge increases time
    // strictly), so the closure is well-defined.
    using Row = std::vector<char>;
    std::vector<Row> reach(static_cast<std::size_t>(n_dense),
                           Row(static_cast<std::size_t>(n_dense), 0));
    for (auto const& [a, b] : strict_pairs) {
        reach[static_cast<std::size_t>(a)][static_cast<std::size_t>(b)] = 1;
    }
    for (int k = 0; k < n_dense; ++k) {
        for (int i = 0; i < n_dense; ++i) {
            if (!reach[static_cast<std::size_t>(i)][static_cast<std::size_t>(k)]) continue;
            for (int j = 0; j < n_dense; ++j) {
                if (reach[static_cast<std::size_t>(k)][static_cast<std::size_t>(j)]) {
                    reach[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = 1;
                }
            }
        }
    }

    // Transitive reduction: (a, b) is a Hasse cover iff a < b in the
    // closure AND there is no intermediate w with a < w < b. We dedupe
    // the candidate edge set first (a CDT mesh can produce the same
    // (i, j) ordered pair from multiple parallel timelike edges).
    std::set<std::pair<int, int>> uniq(strict_pairs.begin(),
                                       strict_pairs.end());
    Poset out(n_dense);
    for (auto const& [a, b] : uniq) {
        if (a == b) continue;  // self-loop guard, shouldn't happen
        bool dominated = false;
        for (int w = 0; w < n_dense && !dominated; ++w) {
            if (w == a || w == b) continue;
            if (reach[static_cast<std::size_t>(a)][static_cast<std::size_t>(w)] &&
                reach[static_cast<std::size_t>(w)][static_cast<std::size_t>(b)]) {
                dominated = true;
            }
        }
        if (!dominated) out.addCover(a, b);
    }
    return out;
}

// ─── compareOrders ───────────────────────────────────────────────────────

OrderAgreement compareOrders(Poset const& a, Poset const& b, int nLabels) {
    auto a_clos = transitive_closure(a, nLabels);
    auto b_clos = transitive_closure(b, nLabels);

    int concordant = 0, discordant = 0, comparable_both = 0;
    int only_a = 0, only_b = 0;
    for (int i = 0; i < nLabels; ++i) {
        for (int j = i + 1; j < nLabels; ++j) {
            const bool a_lt = a_clos[static_cast<std::size_t>(i)]
                                    [static_cast<std::size_t>(j)];
            const bool a_gt = a_clos[static_cast<std::size_t>(j)]
                                    [static_cast<std::size_t>(i)];
            const bool b_lt = b_clos[static_cast<std::size_t>(i)]
                                    [static_cast<std::size_t>(j)];
            const bool b_gt = b_clos[static_cast<std::size_t>(j)]
                                    [static_cast<std::size_t>(i)];
            const bool a_comparable = a_lt || a_gt;
            const bool b_comparable = b_lt || b_gt;
            if (a_comparable && b_comparable) {
                ++comparable_both;
                if ((a_lt && b_lt) || (a_gt && b_gt)) ++concordant;
                else                                  ++discordant;
            } else if (a_comparable) {
                ++only_a;
            } else if (b_comparable) {
                ++only_b;
            }
        }
    }

    OrderAgreement out;
    out.nConcordant      = concordant;
    out.nDiscordant      = discordant;
    out.nComparableBoth = comparable_both;
    out.nOnlyA          = only_a;
    out.nOnlyB          = only_b;
    out.kendallTau       = (comparable_both > 0)
        ? static_cast<double>(concordant - discordant) / comparable_both
        : 0.0;
    out.discordantFraction = (comparable_both > 0)
        ? static_cast<double>(discordant) / comparable_both
        : 0.0;

    auto a_covers = a.covers();
    auto b_covers = b.covers();
    std::set<std::pair<int, int>> edges_a(a_covers.begin(), a_covers.end());
    std::set<std::pair<int, int>> edges_b(b_covers.begin(), b_covers.end());
    int sym_diff = 0;
    for (auto const& e : edges_a) if (!edges_b.count(e)) ++sym_diff;
    for (auto const& e : edges_b) if (!edges_a.count(e)) ++sym_diff;
    const int union_card = (static_cast<int>(edges_a.size()) +
                            static_cast<int>(edges_b.size()) +
                            sym_diff) / 2;
    out.hasseEditDistance = (union_card > 0)
        ? static_cast<double>(sym_diff) / union_card
        : 0.0;
    return out;
}

} // namespace tessera
