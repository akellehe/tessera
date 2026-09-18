// Implementation of the Causet adapter — see
// include/quantum/CausetChain.hpp for the chain layout and conventions.

#include "quantum/CausetChain.hpp"

#include "Poset.h"
#include "mesh/Edge.h"
#include "mesh/EdgeList.h"
#include "mesh/Vertex.h"
#include "mesh/VertexList.h"
#include "spacetime/Spacetime.h"

#include <algorithm>
#include <cstdint>
#include <map>
#include <unordered_map>

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

CausetChain Causet::chainFrom(tessera::spacetime::Spacetime const& st) {
    CausetChain out;

    auto const& vlist = st.getVertexList();
    if (!vlist) return out;

    // ── Group live vertices by integer time slice. std::map keeps the
    // slice index ascending, which is what the chain layout needs:
    // earliest slice → site 0, next slice → the next block of sites.
    std::map<int, std::vector<std::uint64_t>> byTime;
    for (auto* v : vlist->liveVector()) {
        if (v == nullptr) continue;
        const int t = static_cast<int>(v->getTime());
        byTime[t].push_back(v->getId());
    }
    if (byTime.empty()) return out;

    // ── Flatten into the chain layout: ascending-ID inside each
    // antichain, ascending-time across antichains. Build the inverse
    // map (spacetime ID → flat lattice index) along the way.
    out.times.reserve(byTime.size());
    out.antichains.reserve(byTime.size());
    std::unordered_map<std::uint64_t, int> idToFlat;
    int flatIdx = 0;
    for (auto& [t, ids] : byTime) {
        std::sort(ids.begin(), ids.end());
        out.times.push_back(t);
        out.antichains.push_back(ids);
        for (auto id : ids) {
            idToFlat.emplace(id, flatIdx);
            out.vertexIds.push_back(id);
            ++flatIdx;
        }
    }
    out.nSites = flatIdx;

    // ── For each timelike edge whose endpoints land on adjacent
    // antichains, record a hopping pair in the flat-index space.
    std::unordered_map<int, int> timeToAidx;
    timeToAidx.reserve(out.times.size());
    for (int i = 0; i < static_cast<int>(out.times.size()); ++i) {
        timeToAidx.emplace(out.times[static_cast<std::size_t>(i)], i);
    }

    auto const& elist = st.getEdgeList();
    if (elist) {
        for (auto const* e : elist->toVector()) {
            if (e == nullptr) continue;
            if (!e->isTimelike()) continue;  // spacelike/null
            auto const& src = e->getSource();
            auto const& dst = e->getTarget();
            if (!src || !dst) continue;
            const int tS = static_cast<int>(src->getTime());
            const int tD = static_cast<int>(dst->getTime());
            if (tS == tD) continue;

            auto itS = timeToAidx.find(tS);
            auto itD = timeToAidx.find(tD);
            if (itS == timeToAidx.end() || itD == timeToAidx.end())
                continue;
            const int aidxS = itS->second;
            const int aidxD = itD->second;
            if (std::abs(aidxS - aidxD) != 1) continue;  // not adjacent

            auto itFs = idToFlat.find(src->getId());
            auto itFd = idToFlat.find(dst->getId());
            if (itFs == idToFlat.end() || itFd == idToFlat.end())
                continue;
            int i = itFs->second;
            int j = itFd->second;
            if (i == j) continue;
            if (i > j) std::swap(i, j);
            out.hoppingPairs.emplace_back(i, j);
        }
    }
    // Deduplicate: a causal dynamical triangulation (CDT) mesh can
    // produce parallel edges between the same vertex pair through
    // different simplices.
    std::sort(out.hoppingPairs.begin(), out.hoppingPairs.end());
    out.hoppingPairs.erase(
        std::unique(out.hoppingPairs.begin(), out.hoppingPairs.end()),
        out.hoppingPairs.end());

    // ── Inherit the Hasse cover Poset on the flat-lattice label set.
    // Poset::fromSpacetime uses the same ascending-ID remapping we
    // applied above, so its node IDs coincide with our flat-lattice
    // indices.
    out.partialOrder = tessera::Poset::fromSpacetime(st);
    return out;
}

} // namespace tessera::quantum
