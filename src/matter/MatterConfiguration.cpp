// Copyright (c) 2026 Twin Vector Labs LLC. All rights reserved.
#include "matter/MatterConfiguration.h"
#include "spacetime/Spacetime.h"
#include "mesh/Simplex.h"
#include "mesh/Vertex.h"
#include "mesh/Edge.h"
#include "mesh/Fingerprint.h"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <queue>
#include <set>
#include <unordered_map>
#include <unordered_set>

namespace tessera::matter {

void MatterConfiguration::setWorldlineMass(VertexPtr center, double mass,
                                            const Spacetime &st) {
    worldlines_.push_back({buildWorldline(center, st), mass});
}

void MatterConfiguration::setEnergyDensity(SimplexPtr simplex, double rho) {
    simplexRho_[simplex->fingerprint.fingerprint()] = rho;
}

// =====================================================================
// Worldline construction by following timelike edges
// =====================================================================

/// Find the spatial center of a slice: the vertex with the most spacelike
/// neighbors.  Ties are broken by BFS (depth <= 5) sum-of-distances —
/// the tied vertex with the smallest local total wins.
static VertexPtr sliceCenter(const std::vector<VertexPtr> &sliceVerts) {
    if (sliceVerts.empty()) return nullptr;
    if (sliceVerts.size() == 1) return sliceVerts[0];

    // Build spacelike adjacency within the slice
    std::unordered_set<std::uint64_t> sliceIds;
    for (auto *v : sliceVerts) sliceIds.insert(v->getId());

    std::unordered_map<std::uint64_t, std::vector<std::uint64_t>> adj;
    for (auto *v : sliceVerts) {
        for (const auto &e : v->getEdges()) {
            if (!e->isSpacelike()) continue;  // skip timelike/null
            auto *other = (e->getSource()->getId() == v->getId())
                          ? e->getTarget() : e->getSource();
            if (sliceIds.count(other->getId()))
                adj[v->getId()].push_back(other->getId());
        }
    }

    // Primary criterion: most spacelike neighbors
    int maxDegree = 0;
    for (auto *v : sliceVerts) {
        int deg = static_cast<int>(adj[v->getId()].size());
        if (deg > maxDegree) maxDegree = deg;
    }

    // Collect candidates with max degree
    std::vector<VertexPtr> candidates;
    for (auto *v : sliceVerts) {
        if (static_cast<int>(adj[v->getId()].size()) == maxDegree)
            candidates.push_back(v);
    }
    if (candidates.size() == 1) return candidates[0];

    // Tiebreaker: BFS depth <= 5, pick smallest total distance
    constexpr int maxDepth = 5;
    VertexPtr best = candidates[0];
    int bestTotal = std::numeric_limits<int>::max();

    for (auto *v : candidates) {
        std::unordered_map<std::uint64_t, int> dist;
        dist[v->getId()] = 0;
        std::queue<std::uint64_t> q;
        q.push(v->getId());
        int total = 0;
        while (!q.empty()) {
            auto uid = q.front(); q.pop();
            if (dist[uid] >= maxDepth) continue;
            for (auto nid : adj[uid]) {
                if (!dist.count(nid)) {
                    dist[nid] = dist[uid] + 1;
                    total += dist[nid];
                    q.push(nid);
                }
            }
        }
        if (total < bestTotal) {
            bestTotal = total;
            best = v;
        }
    }
    return best;
}

std::vector<VertexPtr> MatterConfiguration::buildWorldline(
    VertexPtr center, const Spacetime &st) {

    // Group all vertices by time slice
    std::unordered_map<int, std::vector<VertexPtr>> sliceMap;
    for (auto *v : st.getVertexList()->liveVector())
        sliceMap[static_cast<int>(v->getTime())].push_back(v);

    std::set<int> timeSet;
    for (const auto &[t, _] : sliceMap) timeSet.insert(t);
    std::vector<int> times(timeSet.begin(), timeSet.end());

    int centerTime = static_cast<int>(center->getTime());

    // Find center's index in sorted times
    auto centerIt = std::find(times.begin(), times.end(), centerTime);
    int centerIdx = static_cast<int>(centerIt - times.begin());

    // For each time slice, find the spatial median vertex. The worldline
    // passes through the center of each slice so the point mass sits at
    // the spatial origin of the effective spacetime.
    std::unordered_map<int, VertexPtr> medianCache;
    medianCache[centerTime] = center;

    auto getMedian = [&](int t) -> VertexPtr {
        auto it = medianCache.find(t);
        if (it != medianCache.end()) return it->second;
        auto sliceIt = sliceMap.find(t);
        if (sliceIt == sliceMap.end()) return nullptr;
        auto *m = sliceCenter(sliceIt->second);
        medianCache[t] = m;
        return m;
    };

    // Trace forward (increasing time)
    std::vector<VertexPtr> forward;
    for (int i = centerIdx + 1; i < static_cast<int>(times.size()); ++i) {
        VertexPtr next = getMedian(times[i]);
        if (!next) break;
        forward.push_back(next);
    }

    // Trace backward (decreasing time)
    std::vector<VertexPtr> backward;
    for (int i = centerIdx - 1; i >= 0; --i) {
        VertexPtr prev = getMedian(times[i]);
        if (!prev) break;
        backward.push_back(prev);
    }

    // Combine: backward (reversed) + center + forward
    std::reverse(backward.begin(), backward.end());
    std::vector<VertexPtr> worldline;
    worldline.reserve(backward.size() + 1 + forward.size());
    worldline.insert(worldline.end(), backward.begin(), backward.end());
    worldline.push_back(center);
    worldline.insert(worldline.end(), forward.begin(), forward.end());

    return worldline;
}

// =====================================================================
// Hinge classification
// =====================================================================

HingeType MatterConfiguration::classifyHinge(SimplexPtr hinge) {
    auto verts = hinge->getVertices();
    if (verts.empty()) return HingeType::SPATIAL;

    double t0 = verts[0]->getTime();
    for (std::size_t i = 1; i < verts.size(); ++i) {
        if (std::abs(verts[i]->getTime() - t0) > 0.5)
            return HingeType::TIMELIKE;
    }
    return HingeType::SPATIAL;
}

void MatterConfiguration::setRadialProfile(
    VertexPtr center, std::function<double(double)> rhoOfR) {
    radialProfiles_.push_back({center, std::move(rhoOfR)});
}

}  // namespace tessera::matter
