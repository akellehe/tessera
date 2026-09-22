// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "observables/ClusterLineage.h"

#include <algorithm>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>

namespace tessera::observables {

namespace {

Record::List intList(const std::vector<int> &values) {
  Record::List out;
  out.reserve(values.size());
  for (const int value : values) out.emplace_back(value);
  return out;
}

Record::List vertexList(const std::vector<std::uint64_t> &values) {
  Record::List out;
  out.reserve(values.size());
  for (const std::uint64_t value : values) out.emplace_back(static_cast<std::int64_t>(value));
  return out;
}

Record::List stringList(const std::vector<std::string> &values) {
  Record::List out;
  out.reserve(values.size());
  for (const auto &value : values) out.emplace_back(value);
  return out;
}

/// Append `name` to `into` unless it is already there, so a union of named
/// reasons never repeats one.
void addReason(std::vector<std::string> &into, const std::string &name) {
  if (std::find(into.begin(), into.end(), name) == into.end()) into.push_back(name);
}

/// One signed integer written as a fixed-width offset decimal, so that the
/// lexicographic order of the strings is the numeric order of the integers
/// they encode. The offset is \f$ 2^{31} \f$, which brings every `int` into
/// \f$ [0, 2^{32}-1] \f$ and so into ten decimal digits.
std::string offsetDecimal(int value) {
  const std::int64_t shifted =
      static_cast<std::int64_t>(value) + static_cast<std::int64_t>(2147483648LL);
  std::string digits = std::to_string(shifted);
  return std::string(10 - digits.size(), '0') + digits;
}

}  // namespace

// --------------------------------------------------------------------------
// InteractionCobordism
// --------------------------------------------------------------------------

std::vector<std::uint64_t> InteractionCobordism::incomingVertices() const {
  std::vector<std::uint64_t> out;
  if (levels == 0) return out;
  for (std::uint64_t w = vertexOffsets[0]; w < vertexOffsets[1]; ++w) out.push_back(w);
  return out;
}

std::vector<std::uint64_t> InteractionCobordism::outgoingVertices() const {
  std::vector<std::uint64_t> out;
  if (levels == 0) return out;
  for (std::uint64_t w = vertexOffsets[levels - 1]; w < vertexOffsets[levels]; ++w) out.push_back(w);
  return out;
}

int InteractionCobordism::edgeIndex(std::uint64_t a, std::uint64_t b) const {
  if (a == b) return -1;
  const std::vector<std::uint64_t> key{std::min(a, b), std::max(a, b)};
  const auto it = std::lower_bound(edges.begin(), edges.end(), key);
  if (it == edges.end() || *it != key) return -1;
  return static_cast<int>(it - edges.begin());
}

std::vector<int> InteractionCobordism::fiberEdges() const {
  std::vector<int> out;
  for (std::size_t w = 0; w < levelOf.size(); ++w) {
    if (levelOf[w] + 1 >= levels) continue;
    const int j = edgeIndex(static_cast<std::uint64_t>(w), responseOf[w]);
    if (j >= 0) out.push_back(j);
  }
  return out;
}

Record InteractionCobordism::toRecord() const {
  Record::Map map;
  map["levels"] = Record(static_cast<std::int64_t>(levels));
  map["vertex_offsets"] = Record(vertexList(vertexOffsets));
  map["vertices"] = Record(static_cast<std::int64_t>(levelOf.size()));
  map["edges"] = Record(static_cast<std::int64_t>(edges.size()));
  map["declared_cells"] = Record(static_cast<std::int64_t>(cells.size()));
  map["dimension"] = Record(complex.dimension());
  Record::List f;
  for (const std::size_t count : complex.fVector()) f.emplace_back(static_cast<std::int64_t>(count));
  map["f_vector"] = Record(std::move(f));
  map["fiber_edges"] = Record(static_cast<std::int64_t>(fiberEdges().size()));
  return Record(std::move(map));
}

// --------------------------------------------------------------------------
// The reads
// --------------------------------------------------------------------------

Record CoorientedCut::toRecord() const {
  Record::Map map;
  map["side"] = Record(intList(side));
  map["crossing_edges"] = Record(intList(crossingEdges));
  map["crossing_signs"] = Record(intList(crossingSigns));
  map["separates"] = Record(separates);
  map["failed_certificates"] = Record(stringList(failedCertificates));
  return Record(std::move(map));
}

Record Lineage::toRecord() const {
  Record::Map map;
  map["cluster_id"] = Record(clusterId);
  map["coefficients"] = Record(intList(coefficients));
  map["fermion_number"] = Record(fermionNumber);
  return Record(std::move(map));
}

Record LineageNumberRead::toRecord() const {
  Record::Map map;
  map["cluster_id"] = Record(clusterId);
  map["number"] = Record(number);
  map["fermion_number"] = Record(fermionNumber);
  map["relative_cycle"] = Record(relativeCycle);
  map["interior_sources"] = Record(vertexList(interiorSources));
  map["cut_separates"] = Record(cutSeparates);
  map["failed_certificates"] = Record(stringList(failedCertificates));
  return Record(std::move(map));
}

Record TotalLineageRead::toRecord() const {
  Record::Map map;
  map["schema_version"] = Record(ClusterLineage::kSchemaVersion);
  map["fermion_number"] = Record(fermionNumber);
  map["baryon_number"] = Record(baryonNumber);
  Record::List per;
  per.reserve(perLineage.size());
  for (const auto &read : perLineage) per.emplace_back(read.toRecord());
  map["per_lineage"] = Record(std::move(per));
  map["failed_certificates"] = Record(stringList(failedCertificates));
  return Record(std::move(map));
}

// --------------------------------------------------------------------------
// The cobordism
// --------------------------------------------------------------------------

InteractionCobordism ClusterLineage::history(
    const std::vector<LevelComplex> &levels,
    const std::vector<std::vector<std::size_t>> &reductions) {
  if (levels.size() < 2)
    throw std::invalid_argument(
        "ClusterLineage::history: a cobordism needs at least two levels (an incoming and an "
        "outgoing one)");
  if (reductions.size() + 1 != levels.size())
    throw std::invalid_argument(
        "ClusterLineage::history: one reduction is required per interaction step, that is one "
        "fewer than the number of levels");
  for (std::size_t l = 0; l < levels.size(); ++l) {
    if (levels[l].vertices == 0)
      throw std::invalid_argument("ClusterLineage::history: level " + std::to_string(l) +
                                  " has no vertices");
    for (const auto &cell : levels[l].cells) {
      if (cell.empty())
        throw std::invalid_argument("ClusterLineage::history: level " + std::to_string(l) +
                                    " declares a cell with no vertices");
      for (const std::uint64_t v : cell)
        if (v >= levels[l].vertices)
          throw std::invalid_argument("ClusterLineage::history: level " + std::to_string(l) +
                                      " declares a cell on vertex " + std::to_string(v) +
                                      ", which the level does not have");
    }
  }
  for (std::size_t l = 0; l + 1 < levels.size(); ++l) {
    if (reductions[l].size() != levels[l].vertices)
      throw std::invalid_argument(
          "ClusterLineage::history: the reduction of step " + std::to_string(l) +
          " must name one response vertex per vertex of level " + std::to_string(l));
    for (const std::size_t r : reductions[l])
      if (r >= levels[l + 1].vertices)
        throw std::invalid_argument(
            "ClusterLineage::history: the reduction of step " + std::to_string(l) +
            " names response vertex " + std::to_string(r) + ", which level " +
            std::to_string(l + 1) + " does not have");
  }

  InteractionCobordism W;
  W.levels = levels.size();
  W.vertexOffsets.assign(levels.size() + 1, 0);
  std::uint64_t offset = 0;
  for (std::size_t l = 0; l < levels.size(); ++l) {
    W.vertexOffsets[l] = offset;
    offset += static_cast<std::uint64_t>(levels[l].vertices);
  }
  W.vertexOffsets[levels.size()] = offset;

  W.levelOf.assign(offset, 0);
  W.responseOf.assign(offset, 0);
  for (std::size_t l = 0; l < levels.size(); ++l)
    for (std::uint64_t v = 0; v < levels[l].vertices; ++v) {
      const std::uint64_t w = W.vertexOffsets[l] + v;
      W.levelOf[w] = l;
      W.responseOf[w] = (l + 1 < levels.size())
                            ? W.vertexOffsets[l + 1] + static_cast<std::uint64_t>(reductions[l][v])
                            : w;
    }

  // Each level's own cells, and each level's vertices as singleton cells so
  // that a vertex no cell uses is still a vertex of W and still receives a
  // fiber edge below.
  for (std::size_t l = 0; l < levels.size(); ++l) {
    for (const auto &cell : levels[l].cells) {
      std::vector<std::uint64_t> shifted;
      shifted.reserve(cell.size());
      for (const std::uint64_t v : cell) shifted.push_back(W.vertexOffsets[l] + v);
      std::sort(shifted.begin(), shifted.end());
      W.cells.push_back(std::move(shifted));
    }
    for (std::uint64_t v = 0; v < levels[l].vertices; ++v)
      W.cells.push_back({W.vertexOffsets[l] + v});
  }

  // The prisms of each step: for a cell (v_0 < ... < v_m) of the incoming
  // level, the staircase cells S_j = {v_0,...,v_j} u {f(v_j),...,f(v_m)}, the
  // image part being a set. Where the reduction identifies two of the cell's
  // vertices the prism carries one image vertex rather than two and is one
  // dimension lower; that collapse is what the reduction does and the cell is
  // kept, not discarded. The two parts draw on disjoint vertex-id ranges, so no
  // staircase cell can repeat a vertex.
  for (std::size_t l = 0; l + 1 < levels.size(); ++l) {
    const auto staircase = [&](std::vector<std::uint64_t> local) {
      std::sort(local.begin(), local.end());
      const std::size_t m = local.size();
      for (std::size_t j = 0; j < m; ++j) {
        std::vector<std::uint64_t> image;
        image.reserve(m - j);
        for (std::size_t i = j; i < m; ++i)
          image.push_back(W.vertexOffsets[l + 1] +
                          static_cast<std::uint64_t>(reductions[l][local[i]]));
        std::sort(image.begin(), image.end());
        image.erase(std::unique(image.begin(), image.end()), image.end());
        std::vector<std::uint64_t> cell;
        cell.reserve(j + 1 + image.size());
        for (std::size_t i = 0; i <= j; ++i) cell.push_back(W.vertexOffsets[l] + local[i]);
        cell.insert(cell.end(), image.begin(), image.end());
        W.cells.push_back(std::move(cell));
      }
    };
    for (const auto &cell : levels[l].cells) staircase(cell);
    for (std::uint64_t v = 0; v < levels[l].vertices; ++v) staircase({v});
  }

  W.complex = cobordism::ChainComplex::fromCells(W.cells);
  if (W.complex.numSimplices(0) != offset)
    throw std::runtime_error(
        "ClusterLineage::history: the cobordism's vertex count does not match the levels' total, "
        "so the canonical C_0 index and the vertex id have parted company");
  W.edges = W.complex.kSimplexVertices(1);
  return W;
}

InteractionCobordism ClusterLineage::mappingCylinder(const LevelComplex &incoming,
                                                     const std::vector<std::size_t> &reduction,
                                                     const LevelComplex &outgoing) {
  return history({incoming, outgoing}, {reduction});
}

// --------------------------------------------------------------------------
// Cuts
// --------------------------------------------------------------------------

CoorientedCut ClusterLineage::levelCut(const InteractionCobordism &W, std::size_t afterLevel) {
  if (W.levels < 2 || afterLevel + 1 >= W.levels)
    throw std::invalid_argument(
        "ClusterLineage::levelCut: the cut must be placed below the last level, so that the "
        "incoming and outgoing boundaries end up on opposite sides");
  std::vector<int> side(W.levelOf.size(), 0);
  for (std::size_t w = 0; w < W.levelOf.size(); ++w) side[w] = W.levelOf[w] > afterLevel ? 1 : 0;
  return cutFromSides(W, side);
}

CoorientedCut ClusterLineage::cutFromSides(const InteractionCobordism &W,
                                           const std::vector<int> &side) {
  if (side.size() != W.complex.numSimplices(0))
    throw std::invalid_argument(
        "ClusterLineage::cutFromSides: the declared sides must have one entry per vertex of the "
        "cobordism");
  CoorientedCut cut;
  cut.side = side;
  for (const int s : side)
    if (s != 0 && s != 1) {
      addReason(cut.failedCertificates, "cut-side-not-binary");
      break;
    }
  for (const std::uint64_t w : W.incomingVertices())
    if (side[w] != 0) {
      addReason(cut.failedCertificates, "incoming-boundary-not-on-the-incoming-side");
      break;
    }
  for (const std::uint64_t w : W.outgoingVertices())
    if (side[w] != 1) {
      addReason(cut.failedCertificates, "outgoing-boundary-not-on-the-outgoing-side");
      break;
    }
  for (std::size_t j = 0; j < W.edges.size(); ++j) {
    const int a = side[W.edges[j][0]];
    const int b = side[W.edges[j][1]];
    if (a == b) continue;
    cut.crossingEdges.push_back(static_cast<int>(j));
    cut.crossingSigns.push_back(b - a);
  }
  cut.separates = cut.failedCertificates.empty();
  return cut;
}

// --------------------------------------------------------------------------
// Lineages
// --------------------------------------------------------------------------

Lineage ClusterLineage::fromFiberPath(const InteractionCobordism &W, std::uint64_t startVertex,
                                      int fermionNumber, const std::string &clusterId) {
  if (startVertex >= W.levelOf.size())
    throw std::invalid_argument("ClusterLineage::fromFiberPath: the start vertex is not a vertex "
                                "of the cobordism");
  std::vector<std::uint64_t> path{startVertex};
  while (W.levelOf[path.back()] + 1 < W.levels) path.push_back(W.responseOf[path.back()]);
  if (path.size() < 2) {
    Lineage lineage;
    lineage.clusterId = clusterId;
    lineage.fermionNumber = fermionNumber;
    lineage.coefficients.assign(W.complex.numSimplices(1), 0);
    return lineage;
  }
  return fromVertexPath(W, path, fermionNumber, clusterId);
}

Lineage ClusterLineage::fromTrackedSupports(const InteractionCobordism &W, std::size_t firstLevel,
                                            const std::vector<std::vector<std::uint64_t>> &supports,
                                            int fermionNumber, const std::string &clusterId) {
  if (supports.size() < 2)
    throw std::invalid_argument(
        "ClusterLineage::fromTrackedSupports: a lineage needs the cluster's support on at least "
        "two consecutive levels");
  if (firstLevel + supports.size() > W.levels)
    throw std::invalid_argument(
        "ClusterLineage::fromTrackedSupports: the supports run past the cobordism's last level");
  std::vector<std::uint64_t> path;
  for (std::size_t i = 0; i < supports.size(); ++i) {
    const std::size_t level = firstLevel + i;
    if (supports[i].empty())
      throw std::invalid_argument("ClusterLineage::fromTrackedSupports: the support on level " +
                                  std::to_string(level) + " is empty");
    std::set<std::uint64_t> global;
    for (const std::uint64_t v : supports[i]) {
      const std::uint64_t w = W.vertexOffsets[level] + v;
      if (w >= W.vertexOffsets[level + 1])
        throw std::invalid_argument("ClusterLineage::fromTrackedSupports: the support on level " +
                                    std::to_string(level) + " names vertex " + std::to_string(v) +
                                    ", which the level does not have");
      global.insert(w);
    }
    if (i == 0) {
      path.push_back(*global.begin());
      continue;
    }
    const std::uint64_t arrived = W.responseOf[path.back()];
    if (global.find(arrived) == global.end())
      throw std::invalid_argument(
          "ClusterLineage::fromTrackedSupports: the representative of the support on level " +
          std::to_string(level - 1) + " reduces to vertex " + std::to_string(arrived) +
          ", which the support on level " + std::to_string(level) + " does not contain");
    path.push_back(arrived);
  }
  return fromVertexPath(W, path, fermionNumber, clusterId);
}

Lineage ClusterLineage::fromVertexPath(const InteractionCobordism &W,
                                       const std::vector<std::uint64_t> &path, int fermionNumber,
                                       const std::string &clusterId) {
  if (path.size() < 2)
    throw std::invalid_argument(
        "ClusterLineage::fromVertexPath: a lineage needs a path of at least two vertices");
  Lineage lineage;
  lineage.clusterId = clusterId;
  lineage.fermionNumber = fermionNumber;
  lineage.coefficients.assign(W.complex.numSimplices(1), 0);
  for (const std::uint64_t w : path)
    if (w >= W.levelOf.size())
      throw std::invalid_argument("ClusterLineage::fromVertexPath: vertex " + std::to_string(w) +
                                  " is not a vertex of the cobordism");
  for (std::size_t i = 0; i + 1 < path.size(); ++i) {
    const int j = W.edgeIndex(path[i], path[i + 1]);
    if (j < 0)
      throw std::invalid_argument("ClusterLineage::fromVertexPath: the pair (" +
                                  std::to_string(path[i]) + ", " + std::to_string(path[i + 1]) +
                                  ") is not an edge of the cobordism");
    lineage.coefficients[static_cast<std::size_t>(j)] += path[i] < path[i + 1] ? 1 : -1;
  }
  return lineage;
}

Lineage ClusterLineage::reversed(const Lineage &lineage) {
  Lineage out = lineage;
  for (int &coefficient : out.coefficients) coefficient = -coefficient;
  return out;
}

Lineage ClusterLineage::pairSurfaceBoundary(const InteractionCobordism &W,
                                            const std::vector<int> &surface, int fermionNumber,
                                            const std::string &clusterId) {
  if (surface.size() != W.complex.numSimplices(2))
    throw std::invalid_argument(
        "ClusterLineage::pairSurfaceBoundary: the pair surface must have one coefficient per "
        "2-simplex of the cobordism");
  Lineage lineage;
  lineage.clusterId = clusterId;
  lineage.fermionNumber = fermionNumber;
  lineage.coefficients.assign(W.complex.numSimplices(1), 0);
  for (const auto &entry : W.complex.boundaryEntries(2))
    lineage.coefficients[static_cast<std::size_t>(entry.row)] +=
        entry.value * surface[static_cast<std::size_t>(entry.column)];
  return lineage;
}

// --------------------------------------------------------------------------
// The pairing
// --------------------------------------------------------------------------

std::vector<int> ClusterLineage::relativeBoundary(const InteractionCobordism &W,
                                                  const Lineage &lineage) {
  if (lineage.coefficients.size() != W.complex.numSimplices(1))
    throw std::invalid_argument(
        "ClusterLineage::relativeBoundary: the lineage must have one coefficient per 1-simplex of "
        "the cobordism");
  std::vector<int> out(W.complex.numSimplices(0), 0);
  for (const auto &entry : W.complex.boundaryEntries(1))
    out[static_cast<std::size_t>(entry.row)] +=
        entry.value * lineage.coefficients[static_cast<std::size_t>(entry.column)];
  return out;
}

int ClusterLineage::intersectionNumber(const InteractionCobordism &W, const CoorientedCut &cut,
                                       const Lineage &lineage) {
  if (lineage.coefficients.size() != W.complex.numSimplices(1))
    throw std::invalid_argument(
        "ClusterLineage::intersectionNumber: the lineage must have one coefficient per 1-simplex "
        "of the cobordism");
  if (cut.side.size() != W.complex.numSimplices(0))
    throw std::invalid_argument(
        "ClusterLineage::intersectionNumber: the cut must have one side per vertex of the "
        "cobordism");
  int number = 0;
  for (const auto &entry : W.complex.boundaryEntries(1))
    number += cut.side[static_cast<std::size_t>(entry.row)] * entry.value *
              lineage.coefficients[static_cast<std::size_t>(entry.column)];
  return number;
}

LineageNumberRead ClusterLineage::read(const InteractionCobordism &W, const CoorientedCut &cut,
                                       const Lineage &lineage) {
  LineageNumberRead out;
  out.clusterId = lineage.clusterId;
  out.fermionNumber = lineage.fermionNumber;
  out.cutSeparates = cut.separates;
  out.number = intersectionNumber(W, cut, lineage);
  const std::vector<int> boundary = relativeBoundary(W, lineage);
  for (std::size_t w = 0; w < boundary.size(); ++w) {
    const std::size_t level = W.levelOf[w];
    if (level == 0 || level + 1 == W.levels) continue;
    if (boundary[w] != 0) out.interiorSources.push_back(static_cast<std::uint64_t>(w));
  }
  out.relativeCycle = out.interiorSources.empty();
  if (!out.cutSeparates) addReason(out.failedCertificates, "cut-does-not-separate");
  if (!out.relativeCycle) addReason(out.failedCertificates, "lineage-not-a-relative-cycle");
  return out;
}

TotalLineageRead ClusterLineage::totals(const InteractionCobordism &W, const CoorientedCut &cut,
                                        const std::vector<Lineage> &lineages) {
  TotalLineageRead out;
  out.perLineage.reserve(lineages.size());
  for (const auto &lineage : lineages) {
    LineageNumberRead one = read(W, cut, lineage);
    out.fermionNumber += one.fermionNumber * one.number;
    for (const auto &reason : one.failedCertificates) addReason(out.failedCertificates, reason);
    out.perLineage.push_back(std::move(one));
  }
  out.baryonNumber = static_cast<double>(out.fermionNumber) / 3.0;
  return out;
}

std::string ClusterLineage::orderKey(const LineageNumberRead &read) {
  if (!read.failedCertificates.empty()) {
    std::string reasons;
    for (const auto &reason : read.failedCertificates) {
      if (!reasons.empty()) reasons += ", ";
      reasons += reason;
    }
    throw std::invalid_argument(
        "ClusterLineage::orderKey: the reading of lineage '" + read.clusterId +
        "' is not certified (" + reasons +
        "), so its lineage number is not independent of the cut and cannot fix "
        "a compilation order that every cut agrees on.");
  }
  return "lineage:" + offsetDecimal(read.number) +
         ":fermion:" + offsetDecimal(read.fermionNumber) + ":" + read.clusterId;
}

}  // namespace tessera::observables
