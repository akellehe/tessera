// Acceptance tests for tessera::quantum::Causet::chainFrom
// (docs/source/quantum-plan.md §6 — tessera-embedded Schwinger lattice
// extraction).
//
// These tests exercise:
//
//   (1) Trivial chain — N time slices each with a single vertex; the
//       extractor collapses to a regular 1D lattice with hopping pairs
//       (0,1), (1,2), …, (N-2, N-1). This is the case the existing
//       Schwinger MPO can run on without any modification — it's the
//       baseline we need before exploring branching causets.
//
//   (2) Branching antichain — a single t=0 vertex linked to two t=1
//       vertices. The flat lattice has 3 sites; hopping pairs are
//       (0, 1) and (0, 2). This is the configuration that will
//       eventually need a tree-tensor-network MPO; here we just verify
//       the data extraction is correct.
//
//   (3) Empty Spacetime — empty CausetChain.
//
//   (4) Sparse vertex IDs — non-contiguous Spacetime IDs are densely
//       remapped to flat lattice indices in a way consistent with
//       Poset::fromSpacetime, so the inherited partialOrder's node
//       IDs match the flat lattice indices used in hoppingPairs.
//
//   (5) Skipping edges — a timelike edge that spans two slices (t=0
//       to t=2 with t=1 also populated) is NOT included in
//       hoppingPairs (would not be a Hasse cover).

#include "Poset.h"
#include "quantum/CausetChain.hpp"
#include "spacetime/Spacetime.h"

#include <complex>
#include <algorithm>
#include <iostream>
#include <utility>
#include <vector>

namespace {

// Spacetime::createEdge takes the complex edge length l, not the squared
// length: Edge::isTimelike() reads the argument of l^2, so a timelike edge
// of unit magnitude is l = sqrt(-1) = i and a spacelike one is l = 1.
constexpr std::complex<double> kTimelikeLength{0.0, 1.0};
constexpr std::complex<double> kSpacelikeLength{1.0, 0.0};

tessera::mesh::VertexPtr make_vertex(tessera::spacetime::Spacetime& st, std::uint64_t id, int t) {
    return st.createVertex(id, std::vector<double>{static_cast<double>(t)});
}

bool pairs_equal(std::vector<std::pair<int, int>> got,
                 std::vector<std::pair<int, int>> want) {
    std::sort(got.begin(), got.end());
    std::sort(want.begin(), want.end());
    return got == want;
}

bool acceptance_trivial_chain() {
    std::cout << "Acceptance #1 — N=4 trivial chain → uniform 1D lattice\n";

    tessera::spacetime::Spacetime st;
    auto v0 = make_vertex(st, 0, 0);
    auto v1 = make_vertex(st, 1, 1);
    auto v2 = make_vertex(st, 2, 2);
    auto v3 = make_vertex(st, 3, 3);
    st.createEdge(v0, v1, kTimelikeLength);
    st.createEdge(v1, v2, kTimelikeLength);
    st.createEdge(v2, v3, kTimelikeLength);

    auto chain = tessera::quantum::Causet::chainFrom(st);

    const std::vector<std::pair<int, int>> want_hops{
        {0, 1}, {1, 2}, {2, 3}
    };
    bool ok = true;
    ok &= (chain.nSites == 4);
    ok &= (chain.times.size() == 4);
    ok &= (chain.antichains.size() == 4);
    ok &= pairs_equal(chain.hoppingPairs, want_hops);
    ok &= (chain.partialOrder.getNodeCount() == 4);
    ok &= (chain.partialOrder.covers().size() == 3);

    std::cout << "  nSites=" << chain.nSites
              << "  hops=" << chain.hoppingPairs.size()
              << "  poset_covers=" << chain.partialOrder.covers().size()
              << "  " << (ok ? "PASS" : "FAIL") << "\n";
    return ok;
}

bool acceptance_branching_antichain() {
    std::cout << "Acceptance #2 — branching causet (1 root, 2 leaves)\n";

    tessera::spacetime::Spacetime st;
    auto v0 = make_vertex(st, 0, 0);
    auto v1 = make_vertex(st, 1, 1);
    auto v2 = make_vertex(st, 2, 1);
    st.createEdge(v0, v1, kTimelikeLength);
    st.createEdge(v0, v2, kTimelikeLength);

    auto chain = tessera::quantum::Causet::chainFrom(st);

    bool ok = true;
    ok &= (chain.nSites == 3);
    ok &= (chain.times.size() == 2);  // t=0, t=1
    ok &= (chain.antichains.size() == 2);
    ok &= (chain.antichains[0].size() == 1);  // {0}
    ok &= (chain.antichains[1].size() == 2);  // {1, 2}
    // Flat layout: site 0 = vid 0, site 1 = vid 1, site 2 = vid 2.
    // Both timelike edges connect site 0 to site 1 / 2 → two hops.
    const std::vector<std::pair<int, int>> want_hops{{0, 1}, {0, 2}};
    ok &= pairs_equal(chain.hoppingPairs, want_hops);
    // partialOrder should also see two covers: 0→1 and 0→2.
    const std::vector<std::pair<int, int>> want_covers{{0, 1}, {0, 2}};
    auto got_covers = chain.partialOrder.covers();
    ok &= pairs_equal(got_covers, want_covers);

    std::cout << "  nSites=" << chain.nSites
              << "  antichain_sizes=[" << chain.antichains[0].size()
              << "," << chain.antichains[1].size() << "]"
              << "  hops=" << chain.hoppingPairs.size()
              << "  " << (ok ? "PASS" : "FAIL") << "\n";
    return ok;
}

bool acceptance_empty_spacetime() {
    std::cout << "Acceptance #3 — empty Spacetime → empty CausetChain\n";

    tessera::spacetime::Spacetime st;
    auto chain = tessera::quantum::Causet::chainFrom(st);

    bool ok = (chain.nSites == 0)
            && chain.times.empty()
            && chain.antichains.empty()
            && chain.vertexIds.empty()
            && chain.hoppingPairs.empty()
            && (chain.partialOrder.getNodeCount() == 0);

    std::cout << "  nSites=" << chain.nSites
              << "  times=" << chain.times.size()
              << "  " << (ok ? "PASS" : "FAIL") << "\n";
    return ok;
}

bool acceptance_sparse_ids() {
    std::cout << "Acceptance #4 — sparse Spacetime IDs → dense flat indices\n";

    tessera::spacetime::Spacetime st;
    auto va = make_vertex(st, 7,  0);
    auto vb = make_vertex(st, 11, 1);
    auto vc = make_vertex(st, 19, 2);
    st.createEdge(va, vb, kTimelikeLength);
    st.createEdge(vb, vc, kTimelikeLength);

    auto chain = tessera::quantum::Causet::chainFrom(st);

    // Sorted ascending IDs: 7 → site 0, 11 → site 1, 19 → site 2.
    bool ok = true;
    ok &= (chain.nSites == 3);
    ok &= (chain.vertexIds.size() == 3);
    ok &= (chain.vertexIds[0] == 7);
    ok &= (chain.vertexIds[1] == 11);
    ok &= (chain.vertexIds[2] == 19);
    const std::vector<std::pair<int, int>> want{{0, 1}, {1, 2}};
    ok &= pairs_equal(chain.hoppingPairs, want);

    std::cout << "  vertexIds=[" << chain.vertexIds[0]
              << "," << chain.vertexIds[1]
              << "," << chain.vertexIds[2] << "]"
              << "  hops_match=" << pairs_equal(chain.hoppingPairs, want)
              << "  " << (ok ? "PASS" : "FAIL") << "\n";
    return ok;
}

bool acceptance_skipping_edge_dropped() {
    std::cout << "Acceptance #5 — skipping (t=0 → t=2) timelike edge dropped\n";

    tessera::spacetime::Spacetime st;
    auto v0 = make_vertex(st, 0, 0);
    auto v1 = make_vertex(st, 1, 1);
    auto v2 = make_vertex(st, 2, 2);
    st.createEdge(v0, v1, kTimelikeLength);
    st.createEdge(v1, v2, kTimelikeLength);
    st.createEdge(v0, v2, kTimelikeLength);  // skips t=1 — should NOT be a hop

    auto chain = tessera::quantum::Causet::chainFrom(st);

    const std::vector<std::pair<int, int>> want_hops{{0, 1}, {1, 2}};
    bool ok = pairs_equal(chain.hoppingPairs, want_hops);
    // partialOrder should also reduce 0→2 out as a non-cover.
    const std::vector<std::pair<int, int>> want_covers{{0, 1}, {1, 2}};
    ok &= pairs_equal(chain.partialOrder.covers(), want_covers);

    std::cout << "  hops=" << chain.hoppingPairs.size()
              << " (want 2)  poset_covers=" << chain.partialOrder.covers().size()
              << " (want 2)  " << (ok ? "PASS" : "FAIL") << "\n";
    return ok;
}

} // namespace

int main() {
    bool ok = true;
    ok &= acceptance_trivial_chain();
    ok &= acceptance_branching_antichain();
    ok &= acceptance_empty_spacetime();
    ok &= acceptance_sparse_ids();
    ok &= acceptance_skipping_edge_dropped();
    std::cout << (ok ? "\nALL PASS\n" : "\nFAILURES\n");
    return ok ? 0 : 1;
}
