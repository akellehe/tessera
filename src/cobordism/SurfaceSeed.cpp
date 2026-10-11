// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "cobordism/SurfaceSeed.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>

#include <Eigen/Dense>

#include "cobordism/ChainComplex.h"
#include "mesh/Edge.h"
#include "mesh/EdgeList.h"
#include "mesh/Simplex.h"
#include "mesh/Vertex.h"
#include "mesh/VertexList.h"
#include "spacetime/Spacetime.h"

namespace tessera::cobordism {

using complexd = std::complex<double>;

namespace {

// An edge's endpoints as an ascending vertex-id pair.
std::pair<std::uint64_t, std::uint64_t> edgeKey(const ::tessera::mesh::Edge *edge) {
  const auto a = edge->getSource()->getId();
  const auto b = edge->getTarget()->getId();
  return {std::min(a, b), std::max(a, b)};
}

}  // namespace

SurfaceSeed SurfaceSeed::fromSurfaces(
    const std::vector<std::shared_ptr<Spacetime>> &surfaces) {
  if (surfaces.empty())
    throw std::invalid_argument("SurfaceSeed::fromSurfaces: no surfaces");
  int surfaceDimension = -1;
  for (const auto &surface : surfaces) {
    if (!surface) throw std::invalid_argument("SurfaceSeed::fromSurfaces: null surface");
    if (surfaceDimension < 0) surfaceDimension = surface->getDimensions();
    if (surface->getDimensions() != surfaceDimension)
      throw std::invalid_argument("SurfaceSeed::fromSurfaces: the surfaces differ in dimension");
    if (surface->getTopSimplices().empty())
      throw std::invalid_argument("SurfaceSeed::fromSurfaces: a surface has no top cells");
  }
  // Disjoint id ranges: surface s's vertices, in ascending id order, take the
  // next block of host ids. The host is one d-dimensional complex whose only
  // cells are the surfaces' own (d-1)-simplices; fromVertexTuples registers each as
  // the non-top simplex it is and auto-wires its edges, which are then set to
  // the surface's lengths (verbatim) with zero phases.
  SurfaceSeed seed;
  std::vector<std::vector<std::uint64_t>> cells;
  std::map<std::pair<std::uint64_t, std::uint64_t>, complexd> lengths;
  std::uint64_t nextId = 0;
  for (const auto &surface : surfaces) {
    std::vector<std::uint64_t> ids;
    for (const auto *vertex : surface->getVertexList()->toVector())
      if (vertex != nullptr) ids.push_back(vertex->getId());
    std::sort(ids.begin(), ids.end());
    std::map<std::uint64_t, std::uint64_t> hostId;
    for (const std::uint64_t id : ids) hostId[id] = nextId++;
    for (const auto &topSimplex : surface->getTopSimplices()) {
      std::vector<std::uint64_t> cell;
      for (const auto *vertex : topSimplex->getVertices()) cell.push_back(hostId.at(vertex->getId()));
      cells.push_back(std::move(cell));
    }
    for (const auto *edge : surface->getEdgeList()->toVector()) {
      if (edge == nullptr || edge->getSource() == nullptr || edge->getTarget() == nullptr) continue;
      const std::uint64_t u = hostId.at(edge->getSource()->getId());
      const std::uint64_t v = hostId.at(edge->getTarget()->getId());
      lengths[{std::min(u, v), std::max(u, v)}] = edge->getLength();
    }
    seed.vertexIds.push_back(std::move(hostId));
  }
  seed.host = Spacetime::fromVertexTuples(surfaceDimension + 1, cells, 1.0, complexd(0.0, 0.0));
  for (auto *edge : seed.host->getEdgeList()->toVector()) {
    const auto found = lengths.find(edgeKey(edge));
    if (found == lengths.end())
      throw std::logic_error("SurfaceSeed::fromSurfaces: a host edge belongs to no surface");
    edge->setLength(found->second);
    edge->setPhase(complexd(0.0, 0.0));
  }
  return seed;
}

SurfaceSeed SurfaceSeed::joinedCollars(
    const std::vector<std::shared_ptr<Spacetime>> &surfaces, int layers,
    const std::vector<std::uint64_t> &twist) {
  const std::string prefix = "SurfaceSeed::joinedCollars: ";
  if (surfaces.size() % 2 != 0 || surfaces.empty())
    throw std::invalid_argument(prefix + "an even, non-zero number of surfaces is required: each is collared with its partner");
  // Three layers, not one. A prism cell spans two adjacent layers, so the
  // all-interior cell the join removes exists only with two interior layers.
  if (layers < 3)
    throw std::invalid_argument(prefix + "layers must be at least three: with fewer, every cell touches a surface and there is none to remove");
  for (const auto &surface : surfaces)
    if (!surface) throw std::invalid_argument(prefix + "null surface");
  // Every collar is the same prism over the shared face set, so it is built
  // once and shifted. SeedCollar has already refused surfaces that do not
  // present one face set, and the pairs are checked through it below.
  std::vector<SurfaceSeed> collars;
  for (std::size_t pair = 0; pair < surfaces.size(); pair += 2)
    collars.push_back(collar(surfaces[pair], surfaces[pair + 1], layers, twist));
  const auto cellsOf = [](const Spacetime &spacetime) {
    std::vector<std::vector<std::uint64_t>> cells;
    for (const auto &topSimplex : spacetime.getTopSimplices()) {
      auto tuple = topSimplex->topTuple();
      std::sort(tuple.begin(), tuple.end());
      cells.push_back(std::move(tuple));
    }
    return cells;
  };
  // A cell no facet of which is on the boundary: the one that can be removed
  // without touching a surface.
  const auto interiorCell = [](const std::vector<std::vector<std::uint64_t>> &cells) {
    std::map<std::vector<std::uint64_t>, int> facetCount;
    for (const auto &cell : cells)
      for (std::size_t skip = 0; skip < cell.size(); ++skip) {
        std::vector<std::uint64_t> facet;
        for (std::size_t i = 0; i < cell.size(); ++i)
          if (i != skip) facet.push_back(cell[i]);
        ++facetCount[facet];
      }
    std::set<std::uint64_t> onBoundary;
    for (const auto &[facet, count] : facetCount)
      if (count == 1) onBoundary.insert(facet.begin(), facet.end());
    for (const auto &cell : cells) {
      bool interior = true;
      for (const auto vertex : cell)
        if (onBoundary.count(vertex) != 0) { interior = false; break; }
      if (interior) return cell;
    }
    return std::vector<std::uint64_t>{};
  };
  std::vector<std::vector<std::uint64_t>> joined;
  std::vector<std::map<std::uint64_t, std::uint64_t>> vertexIds;
  std::vector<std::uint64_t> sphere;   // the first collar's removed cell
  std::uint64_t offset = 0;
  for (std::size_t index = 0; index < collars.size(); ++index) {
    auto cells = cellsOf(*collars[index].host);
    std::uint64_t span = 0;
    for (const auto &cell : cells)
      for (const auto vertex : cell) span = std::max(span, vertex + 1);
    for (auto &cell : cells)
      for (auto &vertex : cell) vertex += offset;
    for (auto &ids : collars[index].vertexIds) {
      for (auto &[surfaceId, hostId] : ids) hostId += offset;
      vertexIds.push_back(std::move(ids));
    }
    auto removed = interiorCell(cells);
    if (removed.empty())
      throw std::invalid_argument(prefix + "a collar has no all-interior cell to remove; more layers are needed");
    // Dropped before the relabel below, not after: the relabel rewrites this
    // cell's own vertices, so a comparison made afterwards would not recognize
    // it and the cell would survive -- leaving its facets with three cofaces.
    cells.erase(std::remove_if(cells.begin(), cells.end(),
                               [&](const std::vector<std::uint64_t> &cell) {
                                 auto sorted = cell;
                                 std::sort(sorted.begin(), sorted.end());
                                 return sorted == removed;
                               }),
                cells.end());
    // The first collar's removed cell is the sphere every other collar is
    // glued onto: identifying the boundaries of two removed tetrahedra is a
    // connected sum along S^2, which adds no first homology.
    if (index == 0) {
      sphere = removed;
    } else {
      std::map<std::uint64_t, std::uint64_t> identify;
      for (std::size_t i = 0; i < removed.size(); ++i) identify[removed[i]] = sphere[i];
      for (auto &cell : cells)
        for (auto &vertex : cell) {
          const auto found = identify.find(vertex);
          if (found != identify.end()) vertex = found->second;
        }
      for (auto &ids : vertexIds)
        for (auto &[surfaceId, hostId] : ids) {
          const auto found = identify.find(hostId);
          if (found != identify.end()) hostId = found->second;
        }
    }
    for (auto &cell : cells) joined.push_back(std::move(cell));
    offset += span;
  }
  // One gate on the whole, as collar takes on its prism.
  const auto verdict = ChainComplex::dualComplexIsValid(joined, surfaces.front()->getDimensions() + 1);
  if (!verdict.first)
    throw std::invalid_argument(prefix + "the joined collars are not a manifold-with-boundary: " + verdict.second);
  SurfaceSeed seed;
  seed.host = Spacetime::fromVertexTuples(surfaces.front()->getDimensions() + 1, joined, 1.0, complexd(0.0, 0.0));
  // Each surface's own lengths verbatim on its edges, the auto-wired length
  // everywhere else, zero phases throughout -- collar's convention.
  std::map<std::pair<std::uint64_t, std::uint64_t>, complexd> lengths;
  for (std::size_t index = 0; index < surfaces.size(); ++index)
    for (const auto *edge : surfaces[index]->getEdgeList()->toVector()) {
      if (edge == nullptr || edge->getSource() == nullptr || edge->getTarget() == nullptr) continue;
      const auto &ids = vertexIds[index];
      const auto source = ids.find(edge->getSource()->getId());
      const auto target = ids.find(edge->getTarget()->getId());
      if (source == ids.end() || target == ids.end()) continue;
      lengths[{std::min(source->second, target->second), std::max(source->second, target->second)}] =
          edge->getLength();
    }
  const complexd interior = seed.host->autoWiredLength(/*crossSlice=*/false);
  for (auto *edge : seed.host->getEdgeList()->toVector()) {
    const auto found = lengths.find(edgeKey(edge));
    edge->setLength(found != lengths.end() ? found->second : interior);
    edge->setPhase(complexd(0.0, 0.0));
  }
  seed.vertexIds = std::move(vertexIds);
  return seed;
}

SurfaceSeed SurfaceSeed::tubedCollars(
    const std::vector<std::shared_ptr<Spacetime>> &surfaces, int layers,
    const std::vector<std::uint64_t> &twist, const TubeSpec &tube) {
  const std::string prefix = "SurfaceSeed::tubedCollars: ";
  if (surfaces.size() != 4)
    throw std::invalid_argument(prefix + "exactly four surfaces are required: two collars, each with its far surface");
  for (const auto &surface : surfaces)
    if (!surface) throw std::invalid_argument(prefix + "null surface");
  if (tube.layers < 1) throw std::invalid_argument(prefix + "the tube needs at least one layer");
  if (!(tube.length > 0.0)) throw std::invalid_argument(prefix + "the tube length must be positive");
  if (!(tube.waist > 0.0)) throw std::invalid_argument(prefix + "the tube waist must be positive");
  if (tube.faceA.size() != 3 || tube.faceB.size() != 3)
    throw std::invalid_argument(prefix + "the attachment faces must be triangles given in the far surfaces' own vertex ids");
  const auto isFaceOf = [](const Spacetime &surface, const std::vector<std::uint64_t> &face) {
    std::vector<std::uint64_t> wanted = face;
    std::sort(wanted.begin(), wanted.end());
    for (const auto &topSimplex : surface.getTopSimplices()) {
      auto tuple = topSimplex->topTuple();
      std::sort(tuple.begin(), tuple.end());
      if (tuple == wanted) return true;
    }
    return false;
  };
  if (!isFaceOf(*surfaces[1], tube.faceA))
    throw std::invalid_argument(prefix + "faceA is no face of the first collar's far surface");
  if (!isFaceOf(*surfaces[3], tube.faceB))
    throw std::invalid_argument(prefix + "faceB is no face of the second collar's far surface");
  // The two collars, each as collar builds it (the twist on each far
  // surface), on disjoint host id ranges.
  std::vector<SurfaceSeed> collars = {collar(surfaces[0], surfaces[1], layers, twist),
                                      collar(surfaces[2], surfaces[3], layers, twist)};
  std::vector<std::vector<std::uint64_t>> joined;
  std::vector<std::map<std::uint64_t, std::uint64_t>> vertexIds;
  std::uint64_t offset = 0;
  for (auto &collar : collars) {
    std::uint64_t span = 0;
    for (const auto &topSimplex : collar.host->getTopSimplices()) {
      auto tuple = topSimplex->topTuple();
      for (auto &vertex : tuple) {
        span = std::max(span, vertex + 1);
        vertex += offset;
      }
      std::sort(tuple.begin(), tuple.end());
      joined.push_back(std::move(tuple));
    }
    for (auto &ids : collar.vertexIds) {
      for (auto &[surfaceId, hostId] : ids) hostId += offset;
      vertexIds.push_back(std::move(ids));
    }
    offset += span;
  }
  // The attachment faces in host ids, matched vertex by vertex in the given
  // order, the last two of faceB reversed when reflecting.
  std::array<std::uint64_t, 3> endA{}, endB{};
  for (int i = 0; i < 3; ++i) {
    endA[static_cast<std::size_t>(i)] = vertexIds[1].at(tube.faceA[static_cast<std::size_t>(i)]);
    const int j = tube.reflect ? (i == 0 ? 0 : 3 - i) : i;
    endB[static_cast<std::size_t>(i)] = vertexIds[3].at(tube.faceB[static_cast<std::size_t>(j)]);
  }
  // The prism over the abstract triangle {0, 1, 2}: layer l's vertex i is
  // 3 l + i (prismCells' stride is 3). Layer 0 is endA, layer `tube.layers`
  // is endB, the layers between are fresh host ids.
  const auto prism = Spacetime::prismCells({{0, 1, 2}}, tube.layers);
  std::vector<std::vector<std::uint64_t>> rings(static_cast<std::size_t>(tube.layers) + 1,
                                                std::vector<std::uint64_t>(3));
  for (int layer = 0; layer <= tube.layers; ++layer)
    for (int i = 0; i < 3; ++i) {
      std::uint64_t id;
      if (layer == 0) id = endA[static_cast<std::size_t>(i)];
      else if (layer == tube.layers) id = endB[static_cast<std::size_t>(i)];
      else id = offset + static_cast<std::uint64_t>(3 * (layer - 1) + i);
      rings[static_cast<std::size_t>(layer)][static_cast<std::size_t>(i)] = id;
    }
  const auto hostIdOf = [&](std::uint64_t prismVertex) {
    return rings[static_cast<std::size_t>(prismVertex / 3)][static_cast<std::size_t>(prismVertex % 3)];
  };
  std::set<std::pair<std::uint64_t, std::uint64_t>> tubeEdges;
  for (const auto &cell : prism) {
    std::vector<std::uint64_t> mapped;
    for (const auto vertex : cell) mapped.push_back(hostIdOf(vertex));
    for (std::size_t a = 0; a < mapped.size(); ++a)
      for (std::size_t b = a + 1; b < mapped.size(); ++b)
        tubeEdges.insert({std::min(mapped[a], mapped[b]), std::max(mapped[a], mapped[b])});
    std::sort(mapped.begin(), mapped.end());
    joined.push_back(std::move(mapped));
  }
  // One gate on the whole.
  const int dimension = surfaces.front()->getDimensions() + 1;
  const auto verdict = ChainComplex::dualComplexIsValid(joined, dimension);
  if (!verdict.first)
    throw std::invalid_argument(prefix + "the tubed collars are not a manifold-with-boundary: " + verdict.second);
  SurfaceSeed seed;
  seed.host = Spacetime::fromVertexTuples(dimension, joined, 1.0, complexd(0.0, 0.0));
  // Each surface's own lengths verbatim on its edges, the auto-wired length
  // everywhere else, zero phases throughout -- collar's convention.
  std::map<std::pair<std::uint64_t, std::uint64_t>, complexd> lengths;
  for (std::size_t index = 0; index < surfaces.size(); ++index)
    for (const auto *edge : surfaces[index]->getEdgeList()->toVector()) {
      if (edge == nullptr || edge->getSource() == nullptr || edge->getTarget() == nullptr) continue;
      const auto &ids = vertexIds[index];
      const auto source = ids.find(edge->getSource()->getId());
      const auto target = ids.find(edge->getTarget()->getId());
      if (source == ids.end() || target == ids.end()) continue;
      lengths[{std::min(source->second, target->second), std::max(source->second, target->second)}] =
          edge->getLength();
    }
  // The tube's Euclidean geometry: the two attachment faces laid out in the
  // plane from their own lengths (vertex 0 at the origin, vertex 1 on the
  // x-axis, vertex 2 above), ring l the affine interpolation at s = l/layers
  // scaled about its centroid by the waist (the end rings by 1), at height
  // l * length; every edge with an interior-ring endpoint gets the distance
  // of its endpoints. The end rings' own edges are the surfaces', already
  // set above.
  const auto layoutOf = [&](const std::array<std::uint64_t, 3> &end) {
    const auto lengthOf = [&](std::uint64_t u, std::uint64_t v) {
      const auto found = lengths.find({std::min(u, v), std::max(u, v)});
      if (found == lengths.end())
        throw std::logic_error("SurfaceSeed::tubedCollars: an attachment face edge has no surface length");
      return found->second.real();
    };
    const double c = lengthOf(end[0], end[1]);  // 0-1
    const double a = lengthOf(end[1], end[2]);  // 1-2
    const double b = lengthOf(end[2], end[0]);  // 2-0
    const double cosine = std::clamp((b * b + c * c - a * a) / (2.0 * b * c), -1.0, 1.0);
    const double alpha = std::acos(cosine);
    return std::array<Eigen::Vector2d, 3>{Eigen::Vector2d(0.0, 0.0), Eigen::Vector2d(c, 0.0),
                                          Eigen::Vector2d(b * std::cos(alpha), b * std::sin(alpha))};
  };
  const auto layoutA = layoutOf(endA);
  const auto layoutB = layoutOf(endB);
  std::vector<std::array<Eigen::Vector3d, 3>> positions(static_cast<std::size_t>(tube.layers) + 1);
  for (int layer = 0; layer <= tube.layers; ++layer) {
    const double s = static_cast<double>(layer) / static_cast<double>(tube.layers);
    const double scale = (layer == 0 || layer == tube.layers) ? 1.0 : tube.waist;
    std::array<Eigen::Vector2d, 3> ring;
    Eigen::Vector2d centroid(0.0, 0.0);
    for (int i = 0; i < 3; ++i) {
      ring[static_cast<std::size_t>(i)] = (1.0 - s) * layoutA[static_cast<std::size_t>(i)] + s * layoutB[static_cast<std::size_t>(i)];
      centroid += ring[static_cast<std::size_t>(i)] / 3.0;
    }
    for (int i = 0; i < 3; ++i) {
      const Eigen::Vector2d q = centroid + scale * (ring[static_cast<std::size_t>(i)] - centroid);
      positions[static_cast<std::size_t>(layer)][static_cast<std::size_t>(i)] =
          Eigen::Vector3d(q(0), q(1), static_cast<double>(layer) * tube.length);
    }
  }
  std::map<std::uint64_t, std::pair<int, int>> ringSlot;
  for (int layer = 0; layer <= tube.layers; ++layer)
    for (int i = 0; i < 3; ++i) ringSlot[rings[static_cast<std::size_t>(layer)][static_cast<std::size_t>(i)]] = {layer, i};
  for (const auto &[u, v] : tubeEdges) {
    const auto [lu, iu] = ringSlot.at(u);
    const auto [lv, iv] = ringSlot.at(v);
    if (lu == lv && (lu == 0 || lu == tube.layers)) continue;  // a surface edge, set verbatim above
    const Eigen::Vector3d pu = positions[static_cast<std::size_t>(lu)][static_cast<std::size_t>(iu)];
    const Eigen::Vector3d pv = positions[static_cast<std::size_t>(lv)][static_cast<std::size_t>(iv)];
    lengths[{u, v}] = complexd((pu - pv).norm(), 0.0);
  }
  const complexd interior = seed.host->autoWiredLength(/*crossSlice=*/false);
  for (auto *edge : seed.host->getEdgeList()->toVector()) {
    const auto found = lengths.find(edgeKey(edge));
    edge->setLength(found != lengths.end() ? found->second : interior);
    edge->setPhase(complexd(0.0, 0.0));
  }
  seed.vertexIds = std::move(vertexIds);
  seed.tubeRings = std::move(rings);
  return seed;
}

SurfaceSeed SurfaceSeed::collar(const std::shared_ptr<Spacetime> &surfaceA,
                                                       const std::shared_ptr<Spacetime> &surfaceB,
                                                       int layers,
                                                       const std::vector<std::uint64_t> &twist) {
  if (!surfaceA || !surfaceB) throw std::invalid_argument("SurfaceSeed::collar: null surface");
  if (layers < 1) throw std::invalid_argument("SurfaceSeed::collar: layers must be at least one");
  if (surfaceA->getDimensions() != surfaceB->getDimensions())
    throw std::invalid_argument("SurfaceSeed::collar: the surfaces differ in dimension");
  if (surfaceA->getTopSimplices().empty() || surfaceB->getTopSimplices().empty())
    throw std::invalid_argument("SurfaceSeed::collar: a surface has no top cells");
  // A surface's vertices in ascending id order are its base indices; the two
  // surfaces must present one and the same face set under those indices, the
  // shared triangulation the collar is the product of.
  const auto indexOf = [](const Spacetime &surface) {
    std::vector<std::uint64_t> ids;
    for (const auto *vertex : surface.getVertexList()->toVector())
      if (vertex != nullptr) ids.push_back(vertex->getId());
    std::sort(ids.begin(), ids.end());
    std::map<std::uint64_t, std::uint64_t> index;
    for (std::size_t n = 0; n < ids.size(); ++n) index[ids[n]] = n;
    return index;
  };
  const auto facesOf = [](const Spacetime &surface, const std::map<std::uint64_t, std::uint64_t> &index) {
    std::set<std::vector<std::uint64_t>> faces;
    for (const auto &topSimplex : surface.getTopSimplices()) {
      std::vector<std::uint64_t> face;
      for (const auto *vertex : topSimplex->getVertices()) face.push_back(index.at(vertex->getId()));
      std::sort(face.begin(), face.end());
      faces.insert(std::move(face));
    }
    return faces;
  };
  const auto indexA = indexOf(*surfaceA);
  auto indexB = indexOf(*surfaceB);
  // The twist: surface B's base indices relabelled before the identification,
  // so that base index k of A meets base index twist[k] of B. Empty is the
  // identity and the product collar. Whether the relabelling is a simplicial
  // automorphism is settled by the face-set comparison below, which is the
  // check this function already makes -- there is no separate gate.
  if (!twist.empty()) {
    if (twist.size() != indexB.size())
      throw std::invalid_argument("SurfaceSeed::collar: the twist has " +
                                  std::to_string(twist.size()) + " entries for " +
                                  std::to_string(indexB.size()) + " vertices");
    std::vector<bool> seen(twist.size(), false);
    for (const std::uint64_t to : twist) {
      if (to >= twist.size() || seen[to])
        throw std::invalid_argument("SurfaceSeed::collar: the twist is not a permutation of the "
                                    "surface's base indices");
      seen[to] = true;
    }
    for (auto &[id, index] : indexB) index = twist[index];
  }
  if (indexA.size() != indexB.size())
    throw std::invalid_argument("SurfaceSeed::collar: the surfaces differ in combinatorics: surface A has " +
                                std::to_string(indexA.size()) + " vertices, surface B " +
                                std::to_string(indexB.size()));
  const auto facesA = facesOf(*surfaceA, indexA);
  const auto facesB = facesOf(*surfaceB, indexB);
  const auto nameFace = [](const std::vector<std::uint64_t> &face) {
    std::string text = "(";
    for (std::size_t i = 0; i < face.size(); ++i) text += (i ? "," : "") + std::to_string(face[i]);
    return text + ")";
  };
  for (const auto &face : facesA)
    if (!facesB.count(face))
      throw std::invalid_argument("SurfaceSeed::collar: the surfaces differ in combinatorics: face " +
                                  nameFace(face) + " of surface A (base indices) is no face of surface B");
  for (const auto &face : facesB)
    if (!facesA.count(face))
      throw std::invalid_argument("SurfaceSeed::collar: the surfaces differ in combinatorics: face " +
                                  nameFace(face) + " of surface B (base indices) is no face of surface A");
  // The staircase prism over the shared faces: layer l is base index + n*l
  // (prismCells' stride is one past the largest base index, n here), so layer
  // 0 is surface A, the last layer surface B, and the layers between are
  // fresh interior vertices.
  const std::uint64_t n = static_cast<std::uint64_t>(indexA.size());
  const std::vector<std::vector<std::uint64_t>> base(facesA.begin(), facesA.end());
  const auto cells = Spacetime::prismCells(base, layers);
  const int dimension = surfaceA->getDimensions() + 1;
  // One gate on the whole: the manifold check on the result, refused by name.
  const auto verdict = ChainComplex::dualComplexIsValid(cells, dimension);
  if (!verdict.first)
    throw std::invalid_argument("SurfaceSeed::collar: the collar is not a manifold-with-boundary: " +
                                verdict.second);
  SurfaceSeed seed;
  seed.host = Spacetime::fromVertexTuples(dimension, cells, 1.0, complexd(0.0, 0.0));
  std::map<std::uint64_t, std::uint64_t> hostIdA;
  std::map<std::uint64_t, std::uint64_t> hostIdB;
  const std::uint64_t offsetB = n * static_cast<std::uint64_t>(layers);
  for (const auto &[id, index] : indexA) hostIdA[id] = index;
  for (const auto &[id, index] : indexB) hostIdB[id] = offsetB + index;
  // The surfaces' own lengths verbatim on their edges, the auto-wired length
  // on every other edge, zero phases throughout.
  std::map<std::pair<std::uint64_t, std::uint64_t>, complexd> lengths;
  const auto collect = [&](const Spacetime &surface, const std::map<std::uint64_t, std::uint64_t> &hostId) {
    for (const auto *edge : surface.getEdgeList()->toVector()) {
      if (edge == nullptr || edge->getSource() == nullptr || edge->getTarget() == nullptr) continue;
      const std::uint64_t u = hostId.at(edge->getSource()->getId());
      const std::uint64_t v = hostId.at(edge->getTarget()->getId());
      lengths[{std::min(u, v), std::max(u, v)}] = edge->getLength();
    }
  };
  collect(*surfaceA, hostIdA);
  collect(*surfaceB, hostIdB);
  const complexd interior = seed.host->autoWiredLength(/*crossSlice=*/false);
  for (auto *edge : seed.host->getEdgeList()->toVector()) {
    const auto found = lengths.find(edgeKey(edge));
    edge->setLength(found != lengths.end() ? found->second : interior);
    edge->setPhase(complexd(0.0, 0.0));
  }
  seed.vertexIds = {std::move(hostIdA), std::move(hostIdB)};
  return seed;
}

}  // namespace tessera::cobordism
