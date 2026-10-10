// Causet-adapter acceptance tests (docs/source/quantum-plan.md §6):
//
// tessera::Poset::fromSpacetime(Spacetime const&) inherits a partial order
// from a tessera::spacetime::Spacetime by treating timelike edges (Edge::getSquaredLength
// < 0) as strict precedes-relations oriented earliest-time → latest-time,
// then transitively reducing the resulting DAG to its Hasse covers.
//
// These tests build small hand-crafted Spacetimes and verify:
//
//   (1) Two-slice ladder — every cross-slice timelike edge is a cover.
//   (2) Three-slice chain with a "skipping" edge — transitive reduction
//       collapses the skipping edge so that only adjacent-slice covers
//       remain.
//   (3) Empty Spacetime — empty Poset.
//   (4) Vertices with only spacelike edges — no covers (nodes still all
//       present).
//   (5) Self-comparison under compareOrders — Kendall-τ = 1, edit
//       distance = 0.
//
// We bypass the topology-driven `Spacetime::build()` path and use the
// public `createVertex(id, coords)` / `createEdge(src, tgt, length)` API
// directly. That keeps the tests free of any CDT / Toroid topology
// build requirements while still exercising the real VertexList /
// EdgeList machinery.

#include "Poset.h"
#include "spacetime/Spacetime.h"

#include <complex>
#include <algorithm>
#include <iostream>
#include <set>
#include <utility>
#include <vector>

namespace {

// Spacetime::createEdge takes the complex edge length l, not the squared
// length: Edge::isTimelike() reads the argument of l^2, so a timelike edge
// of unit magnitude is l = sqrt(-1) = i and a spacelike one is l = 1.
constexpr std::complex<double> kTimelikeLength{0.0, 1.0};
constexpr std::complex<double> kSpacelikeLength{1.0, 0.0};

// Convenience: build a vertex at integer time `t`, ID `id`, with 1D
// coords {static_cast<double>(t)}. Vertex::getTime() returns |x_0| for
// 1D coords, so this gives integer-valued times that match the
// time-slice / antichain pattern the causet adapter needs.
tessera::mesh::VertexPtr make_vertex(tessera::spacetime::Spacetime& st, std::uint64_t id, int t) {
    return st.createVertex(id, std::vector<double>{static_cast<double>(t)});
}

bool covers_equal(std::vector<std::pair<int, int>> got,
                  std::vector<std::pair<int, int>> want) {
    std::sort(got.begin(), got.end());
    std::sort(want.begin(), want.end());
    return got == want;
}

bool acceptance_two_slice_ladder() {
    std::cout << "Acceptance #1 — 2-slice ladder, every cross-slice edge is a cover\n";

    tessera::spacetime::Spacetime st;
    auto v0 = make_vertex(st, 0, 0);
    auto v1 = make_vertex(st, 1, 0);
    auto v2 = make_vertex(st, 2, 1);
    auto v3 = make_vertex(st, 3, 1);
    // Four cross-slice timelike edges (l = i, so l^2 = -1). No same-time
    // edges, so spacelike at the slice level is implicit.
    st.createEdge(v0, v2, kTimelikeLength);
    st.createEdge(v0, v3, kTimelikeLength);
    st.createEdge(v1, v2, kTimelikeLength);
    st.createEdge(v1, v3, kTimelikeLength);

    auto p = tessera::Poset::fromSpacetime(st);
    const std::vector<std::pair<int, int>> want{
        {0, 2}, {0, 3}, {1, 2}, {1, 3}
    };
    const auto got = p.covers();
    const bool ok = (p.getNodeCount() == 4) && covers_equal(got, want);

    std::cout << "  getNodeCount=" << p.getNodeCount()
              << " (want 4)  covers=" << got.size()
              << " (want " << want.size() << ")  "
              << (ok ? "PASS" : "FAIL") << "\n";
    return ok;
}

bool acceptance_three_slice_with_skip() {
    std::cout << "Acceptance #2 — 3-slice chain, skipping edge is reduced\n";

    tessera::spacetime::Spacetime st;
    auto v0 = make_vertex(st, 0, 0);
    auto v1 = make_vertex(st, 1, 1);
    auto v2 = make_vertex(st, 2, 2);
    // Adjacent-slice edges + a "skip" edge from t=0 directly to t=2.
    // The strict precedes-DAG has 0→1, 1→2, 0→2. Transitive reduction
    // should drop 0→2 because 0→1→2 already provides that order.
    st.createEdge(v0, v1, kTimelikeLength);
    st.createEdge(v1, v2, kTimelikeLength);
    st.createEdge(v0, v2, kTimelikeLength);

    auto p = tessera::Poset::fromSpacetime(st);
    const std::vector<std::pair<int, int>> want{{0, 1}, {1, 2}};
    const auto got = p.covers();
    const bool ok = (p.getNodeCount() == 3) && covers_equal(got, want);

    std::cout << "  getNodeCount=" << p.getNodeCount()
              << "  covers={ ";
    for (auto [a, b] : got) std::cout << "(" << a << "," << b << ") ";
    std::cout << "}  " << (ok ? "PASS" : "FAIL") << "\n";
    return ok;
}

bool acceptance_empty_spacetime() {
    std::cout << "Acceptance #3 — empty Spacetime → empty Poset\n";
    tessera::spacetime::Spacetime st;
    auto p = tessera::Poset::fromSpacetime(st);
    const bool ok = (p.getNodeCount() == 0) && p.covers().empty();
    std::cout << "  getNodeCount=" << p.getNodeCount()
              << "  covers=" << p.covers().size()
              << "  " << (ok ? "PASS" : "FAIL") << "\n";
    return ok;
}

bool acceptance_only_spacelike_edges() {
    std::cout << "Acceptance #4 — same-slice spacelike-only graph → no covers\n";

    tessera::spacetime::Spacetime st;
    auto v0 = make_vertex(st, 0, 0);
    auto v1 = make_vertex(st, 1, 0);
    auto v2 = make_vertex(st, 2, 0);
    // Spacelike edges only — squaredLength > 0, all at the same time
    // slice. Should produce a Poset with 3 nodes and 0 covers.
    st.createEdge(v0, v1, kSpacelikeLength);
    st.createEdge(v1, v2, kSpacelikeLength);

    auto p = tessera::Poset::fromSpacetime(st);
    const bool ok = (p.getNodeCount() == 3) && p.covers().empty();
    std::cout << "  getNodeCount=" << p.getNodeCount()
              << "  covers=" << p.covers().size()
              << "  " << (ok ? "PASS" : "FAIL") << "\n";
    return ok;
}

bool acceptance_self_comparison() {
    std::cout << "Acceptance #5 — compareOrders(p, p) = perfect agreement\n";

    tessera::spacetime::Spacetime st;
    auto v0 = make_vertex(st, 0, 0);
    auto v1 = make_vertex(st, 1, 1);
    auto v2 = make_vertex(st, 2, 2);
    auto v3 = make_vertex(st, 3, 1);
    st.createEdge(v0, v1, kTimelikeLength);
    st.createEdge(v0, v3, kTimelikeLength);
    st.createEdge(v1, v2, kTimelikeLength);
    st.createEdge(v3, v2, kTimelikeLength);

    auto p = tessera::Poset::fromSpacetime(st);
    auto agree = tessera::compareOrders(p, p, p.getNodeCount());

    const bool ok = (agree.kendallTau == 1.0) &&
                    (agree.discordantFraction == 0.0) &&
                    (agree.hasseEditDistance == 0.0) &&
                    (agree.nDiscordant == 0);
    std::cout << "  τ=" << agree.kendallTau
              << "  discordant_frac=" << agree.discordantFraction
              << "  edit_dist=" << agree.hasseEditDistance
              << "  " << (ok ? "PASS" : "FAIL") << "\n";
    return ok;
}

bool acceptance_dense_remap() {
    std::cout << "Acceptance #6 — sparse Spacetime IDs are densely remapped\n";

    // Use non-contiguous spacetime vertex IDs (0, 5, 11, 13). The Poset
    // should remap them densely to 0..3 in ascending-ID order, so
    // ID-monotonic edges produce ID-monotonic covers in Poset space.
    tessera::spacetime::Spacetime st;
    auto v_lo = make_vertex(st, 0,  0);
    auto v_md = make_vertex(st, 5,  1);
    auto v_hi = make_vertex(st, 11, 2);
    auto v_xx = make_vertex(st, 13, 2);
    st.createEdge(v_lo, v_md, kTimelikeLength);
    st.createEdge(v_md, v_hi, kTimelikeLength);
    st.createEdge(v_md, v_xx, kTimelikeLength);

    auto p = tessera::Poset::fromSpacetime(st);
    // Ascending sort of {0, 5, 11, 13} → indices 0, 1, 2, 3. Covers
    // (0, 5)→(0, 1), (5, 11)→(1, 2), (5, 13)→(1, 3).
    const std::vector<std::pair<int, int>> want{{0, 1}, {1, 2}, {1, 3}};
    const auto got = p.covers();
    const bool ok = (p.getNodeCount() == 4) && covers_equal(got, want);

    std::cout << "  getNodeCount=" << p.getNodeCount()
              << "  covers={ ";
    for (auto [a, b] : got) std::cout << "(" << a << "," << b << ") ";
    std::cout << "}  " << (ok ? "PASS" : "FAIL") << "\n";
    return ok;
}

bool acceptance_to_dot_format() {
    std::cout << "Acceptance #7 — toDot() emits the cover edges\n";
    tessera::spacetime::Spacetime st;
    auto v0 = make_vertex(st, 0, 0);
    auto v1 = make_vertex(st, 1, 1);
    st.createEdge(v0, v1, kTimelikeLength);

    auto p = tessera::Poset::fromSpacetime(st);
    const auto dot = p.toDot();
    const bool ok = (dot.find("digraph poset")  != std::string::npos) &&
                    (dot.find("0 -> 1")          != std::string::npos);
    std::cout << "  contains digraph header=" << (dot.find("digraph poset")  != std::string::npos)
              << " contains '0 -> 1'=" << (dot.find("0 -> 1") != std::string::npos)
              << "  " << (ok ? "PASS" : "FAIL") << "\n";
    return ok;
}

} // namespace

int main() {
    bool ok = true;
    ok &= acceptance_two_slice_ladder();
    ok &= acceptance_three_slice_with_skip();
    ok &= acceptance_empty_spacetime();
    ok &= acceptance_only_spacelike_edges();
    ok &= acceptance_self_comparison();
    ok &= acceptance_dense_remap();
    ok &= acceptance_to_dot_format();
    std::cout << (ok ? "\nALL PASS\n" : "\nFAILURES\n");
    return ok ? 0 : 1;
}
