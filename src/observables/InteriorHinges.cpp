// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "observables/InteriorHinges.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "mesh/Simplex.h"
#include "mesh/Vertex.h"
#include "spacetime/Spacetime.h"

namespace tessera::observables {
using ::tessera::mesh::Simplex;

namespace {

std::vector<std::uint64_t> sortedVids(const Simplex &s) {
  std::vector<std::uint64_t> vids;
  vids.reserve(s.getVertices().size());
  for (const auto &v : s.getVertices()) vids.push_back(v->getId());
  std::sort(vids.begin(), vids.end());
  return vids;
}

// Multi-source breadth-first search (BFS) shell distance from `seeds` over the
// one-skeleton of the current top cells. The walk graph is built from the top
// cells rather than the raw edge list, so an orphan edge stranded by a Pachner
// move cannot shortcut the walk. Returns vertex id -> shell; unreachable
// vertices are absent.
std::unordered_map<std::uint64_t, int> bfsShells(
    const std::vector<std::vector<std::uint64_t>> &tops,
    const std::vector<std::uint64_t> &seeds) {
  std::unordered_map<std::uint64_t, std::set<std::uint64_t>> adjacency;
  for (const auto &top : tops) {
    for (std::size_t i = 0; i < top.size(); ++i) {
      for (std::size_t j = i + 1; j < top.size(); ++j) {
        adjacency[top[i]].insert(top[j]);
        adjacency[top[j]].insert(top[i]);
      }
    }
  }
  std::unordered_map<std::uint64_t, int> dist;
  std::queue<std::uint64_t> frontier;
  std::set<std::uint64_t> uniqueSeeds(seeds.begin(), seeds.end());
  for (std::uint64_t s : uniqueSeeds) {
    if (adjacency.count(s)) {
      dist[s] = 0;
      frontier.push(s);
    }
  }
  while (!frontier.empty()) {
    std::uint64_t u = frontier.front();
    frontier.pop();
    for (std::uint64_t v : adjacency[u]) {
      if (!dist.count(v)) {
        dist[v] = dist[u] + 1;
        frontier.push(v);
      }
    }
  }
  return dist;
}

double mean(const std::vector<double> &xs) {
  double s = 0.0;
  for (double x : xs) s += x;
  return s / static_cast<double>(xs.size());
}

// Population standard deviation (ddof = 0).
double populationStd(const std::vector<double> &xs) {
  const double m = mean(xs);
  double acc = 0.0;
  for (double x : xs) acc += (x - m) * (x - m);
  return std::sqrt(acc / static_cast<double>(xs.size()));
}

}  // namespace

InteriorHinges::InteriorHinges(std::shared_ptr<const Spacetime> spacetime,
                               std::vector<std::vector<std::uint64_t>> holes)
    : spacetime_(std::move(spacetime)), holes_(std::move(holes)) {
  // Current top cells. This reader is 4D-specific: the hinge dimension
  // (triangles) and the radius root (1/4) both track d = 4, so a non-4D complex
  // is refused rather than read.
  const auto &topSimplices = spacetime_->getTopSimplices();
  std::vector<std::vector<std::uint64_t>> tops;
  tops.reserve(topSimplices.size());
  for (const auto *t : topSimplices) {
    if (t->getVertices().size() != 5) {
      throw std::invalid_argument(
          "InteriorHinges: a top cell has " +
          std::to_string(t->getVertices().size()) +
          " vertices; this reader is for 4-complexes (5-vertex top cells) — on "
          "a d-complex the hinges are the (d-2)-simplices and the radius root "
          "is 1/d, so use the reader that matches your dimension");
    }
    tops.push_back(sortedVids(*t));
  }
  if (tops.empty()) {
    throw std::invalid_argument("InteriorHinges: spacetime has no top cells");
  }

  // Boundary tetrahedra are the codimension-1 faces owned by exactly one top
  // cell (Spacetime::getBoundary). A triangle is a boundary hinge iff it is a
  // face of some boundary tetrahedron, i.e. its coface fan contains a
  // once-shared tetrahedron; every other triangle of the current top cells is
  // interior (closed fan).
  census_.boundaryTets = spacetime_->getBoundary();
  census_.nBoundaryTets = static_cast<int>(census_.boundaryTets.size());
  std::set<std::vector<std::uint64_t>> boundaryTriangles;
  for (const auto &tet : census_.boundaryTets) {  // 4 vertices, sorted
    for (std::size_t omit = 0; omit < tet.size(); ++omit) {
      std::vector<std::uint64_t> tri;
      tri.reserve(tet.size() - 1);
      for (std::size_t i = 0; i < tet.size(); ++i) {
        if (i != omit) tri.push_back(tet[i]);
      }
      boundaryTriangles.insert(std::move(tri));
    }
  }

  // The BFS shell seeds are the selected holes' vertices.
  std::vector<std::uint64_t> holeVertices;
  for (const auto &hole : holes_) {
    for (std::uint64_t v : hole) holeVertices.push_back(v);
  }
  std::sort(holeVertices.begin(), holeVertices.end());
  holeVertices.erase(std::unique(holeVertices.begin(), holeVertices.end()),
                     holeVertices.end());
  const auto shellOf = bfsShells(tops, holeVertices);

  // The registered triangle and tetrahedron Simplex objects live in
  // getSimplices(); hasTopCoface() keeps exactly those contained in a current
  // top cell, which are the objects the deficit and dual-volume reads need. One
  // pass over getSimplices() collects the triangle census, the interior hinges'
  // curvature and the tetrahedron count.
  int nTets = 0;
  int nHingesTotal = 0;
  int nHingesBoundary = 0;
  std::vector<Hinge> interior;
  for (const auto *s : spacetime_->getSimplices()) {
    const std::size_t vc = s->getVertices().size();
    if (vc == 4) {
      if (s->hasTopCoface()) ++nTets;
      continue;
    }
    if (vc != 3 || !s->hasTopCoface()) continue;
    ++nHingesTotal;
    std::vector<std::uint64_t> tri = sortedVids(*s);
    if (boundaryTriangles.count(tri)) {
      ++nHingesBoundary;
      continue;
    }
    const std::complex<double> deficit = s->deficitAngle();
    Hinge hinge;
    hinge.re = deficit.real();
    hinge.im = deficit.imag();
    hinge.dv = s->dualVolume().real();
    // shell = the minimum BFS distance over the triangle's vertices reachable
    // from a hole; empty when no vertex is reachable or no holes were given.
    std::optional<int> shell;
    for (std::uint64_t v : tri) {
      auto it = shellOf.find(v);
      if (it != shellOf.end()) {
        shell = shell ? std::min(*shell, it->second) : it->second;
      }
    }
    hinge.shell = shell;
    hinge.vids = std::move(tri);
    interior.push_back(std::move(hinge));
  }
  // Deterministic hinge order (sorted by vertex tuple) so every downstream sum
  // is bit-stable.
  std::sort(interior.begin(), interior.end(),
            [](const Hinge &a, const Hinge &b) { return a.vids < b.vids; });
  hinges_ = std::move(interior);

  census_.nTops = static_cast<int>(tops.size());
  census_.nTets = nTets;
  census_.nHingesTotal = nHingesTotal;
  census_.nHingesInterior = static_cast<int>(hinges_.size());
  census_.nHingesBoundary = nHingesBoundary;
  census_.nHoleVertices = static_cast<int>(holeVertices.size());
}

InteriorHinges::Masses InteriorHinges::masses() const {
  Masses out;
  if (hinges_.empty()) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    out.mShell = out.mSum = out.mAction = out.maxAbsIm = nan;
    out.nImNonzero = 0;
    out.empty = true;
    return out;
  }
  out.empty = false;
  // Per-shell bins of the real part of the deficit angle, shell-ascending with
  // the unshelled bin (nullopt) last.
  std::map<int, std::vector<double>> shelled;
  std::vector<double> unshelled;
  double mSum = 0.0;
  double mAction = 0.0;
  double maxAbsIm = 0.0;
  int nImNonzero = 0;
  for (const auto &h : hinges_) {
    if (h.shell) {
      shelled[*h.shell].push_back(h.re);
    } else {
      unshelled.push_back(h.re);
    }
    mSum += h.re;
    mAction += h.re * std::fabs(h.dv);
    maxAbsIm = std::max(maxAbsIm, std::fabs(h.im));
    if (std::fabs(h.im) > IM_TOL) ++nImNonzero;
  }
  double mShell = 0.0;
  for (const auto &kv : shelled) {
    const double sm = mean(kv.second);
    out.shellMeans.emplace_back(kv.first, sm);
    mShell += sm;
  }
  if (!unshelled.empty()) {
    const double sm = mean(unshelled);
    out.shellMeans.emplace_back(std::nullopt, sm);
    mShell += sm;
  }
  out.mShell = mShell;
  out.mSum = mSum;
  out.mAction = mAction;
  out.maxAbsIm = maxAbsIm;
  out.nImNonzero = nImNonzero;
  return out;
}

InteriorHinges::Radii InteriorHinges::radii() const {
  Radii out;
  std::unordered_set<std::uint64_t> boundaryVertexIds;
  for (const auto &tet : census_.boundaryTets) {
    for (std::uint64_t v : tet) boundaryVertexIds.insert(v);
  }
  // V_dual: the sum of dual-volume magnitudes over the strictly interior
  // vertices (those on no boundary tetrahedron), i.e. the signature-aware
  // circumcentric dual 4-volume. The 0-simplices are walked in id order,
  // skipping orphans (no top coface) and boundary vertices.
  std::vector<const Simplex *> vertexSimplices;
  for (const auto *s : spacetime_->getSimplices()) {
    if (s->getVertices().size() == 1) vertexSimplices.push_back(s);
  }
  std::sort(vertexSimplices.begin(), vertexSimplices.end(),
            [](const Simplex *a, const Simplex *b) {
              return a->getVertices()[0]->getId() <
                     b->getVertices()[0]->getId();
            });
  double vDual = 0.0;
  int nInterior = 0;
  for (const auto *s : vertexSimplices) {
    if (!s->hasTopCoface()) continue;  // orphan 0-simplex stranded by a move
    const std::uint64_t vid = s->getVertices()[0]->getId();
    if (boundaryVertexIds.count(vid)) continue;
    vDual += std::abs(s->dualVolume());
    ++nInterior;
  }
  double vPrimal = 0.0;
  for (const auto *t : spacetime_->getTopSimplices()) {
    vPrimal += std::fabs(t->volume());
  }
  const double nan = std::numeric_limits<double>::quiet_NaN();
  out.vDual = vDual;
  out.vPrimal = vPrimal;
  out.nInteriorVertices = nInterior;
  out.rDual = vDual > 0.0 ? std::pow(vDual, 0.25) : nan;
  out.rPrimal = vPrimal > 0.0 ? std::pow(vPrimal, 0.25) : nan;
  return out;
}

InteriorHinges::Localization InteriorHinges::localization() const {
  Localization out;
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();
  if (hinges_.empty()) {
    out.pr = out.concentration = out.meanRe = out.stdRe = out.stdOverMean = nan;
    out.rmsShellRadius = out.fracWithinShell1 = nan;
    out.empty = true;
    return out;
  }
  out.empty = false;
  std::vector<double> re;
  std::vector<double> weight;
  re.reserve(hinges_.size());
  weight.reserve(hinges_.size());
  for (const auto &h : hinges_) {
    re.push_back(h.re);
    weight.push_back(std::fabs(h.re * std::fabs(h.dv)));
  }
  double totalWeight = 0.0;
  double sumWeightSq = 0.0;
  for (double w : weight) {
    totalWeight += w;
    sumWeightSq += w * w;
  }
  out.pr = totalWeight > 0.0
               ? (totalWeight * totalWeight /
                  (static_cast<double>(weight.size()) * sumWeightSq))
               : nan;
  out.concentration = (out.pr == out.pr && out.pr > 0.0) ? 1.0 / out.pr : nan;

  const bool allShelled =
      std::all_of(hinges_.begin(), hinges_.end(),
                  [](const Hinge &h) { return h.shell.has_value(); });
  out.rmsShellRadius = nan;
  out.fracWithinShell1 = nan;
  if (allShelled && totalWeight > 0.0) {
    std::map<int, std::vector<std::size_t>> byShell;
    for (std::size_t i = 0; i < hinges_.size(); ++i) {
      byShell[*hinges_[i].shell].push_back(i);
    }
    for (const auto &kv : byShell) {
      ShellProfile p;
      p.n = static_cast<int>(kv.second.size());
      double reSum = 0.0;
      double wSum = 0.0;
      for (std::size_t i : kv.second) {
        reSum += re[i];
        wSum += weight[i];
      }
      p.meanRe = reSum / static_cast<double>(kv.second.size());
      p.weightShare = wSum / totalWeight;
      out.shellProfile.emplace_back(kv.first, p);
    }
    double rms = 0.0;
    double near = 0.0;
    for (std::size_t i = 0; i < hinges_.size(); ++i) {
      const double sh = static_cast<double>(*hinges_[i].shell);
      rms += weight[i] * sh * sh;
      if (*hinges_[i].shell <= 1) near += weight[i];
    }
    out.rmsShellRadius = std::sqrt(rms / totalWeight);
    out.fracWithinShell1 = near / totalWeight;
  }

  out.meanRe = mean(re);
  out.stdRe = populationStd(re);
  out.stdOverMean = out.meanRe != 0.0 ? out.stdRe / std::fabs(out.meanRe) : inf;
  return out;
}

InteriorHinges::RmTable InteriorHinges::rmTable(const Masses &mass,
                                                const Radii &rad) const {
  RmTable out;
  out.physical = PHYSICAL_RM;
  // 3 mass definitions x 2 radius definitions, mass outer and radius inner.
  const std::pair<const char *, double> masses[] = {
      {"m_shell", mass.mShell}, {"m_sum", mass.mSum}, {"m_action", mass.mAction}};
  const std::pair<const char *, double> radii[] = {{"r_dual", rad.rDual},
                                                   {"r_primal", rad.rPrimal}};
  double spreadMin = std::numeric_limits<double>::infinity();
  double spreadMax = -std::numeric_limits<double>::infinity();
  bool anyFinite = false;
  for (const auto &m : masses) {
    for (const auto &r : radii) {
      const double value = r.second * m.second;  // rad * mass
      out.combos.emplace_back(std::string(r.first) + " x " + m.first, value);
      if (std::isfinite(value)) {
        spreadMin = std::min(spreadMin, value);
        spreadMax = std::max(spreadMax, value);
        anyFinite = true;
      }
    }
  }
  const double nan = std::numeric_limits<double>::quiet_NaN();
  out.spreadMin = anyFinite ? spreadMin : nan;
  out.spreadMax = anyFinite ? spreadMax : nan;
  return out;
}

}  // namespace tessera::observables
