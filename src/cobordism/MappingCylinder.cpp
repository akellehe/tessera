// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "cobordism/MappingCylinder.h"

#include <algorithm>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>

namespace tessera::cobordism {
namespace {

using Cell = std::vector<std::uint64_t>;

Cell sortedUnique(Cell cell) {
  std::sort(cell.begin(), cell.end());
  cell.erase(std::unique(cell.begin(), cell.end()), cell.end());
  return cell;
}

bool contains(const Cell &whole, const Cell &part) {
  return std::includes(whole.begin(), whole.end(), part.begin(), part.end());
}

}  // namespace

MappingCylinder::MappingCylinder(MappingCylinderDeclaration declaration)
    : declaration_(std::move(declaration)) {
  if (declaration_.incomingTopCells.empty())
    throw std::invalid_argument(
        "MappingCylinder: the incoming complex has no top cell, so there is no "
        "cylinder to build over it");

  std::vector<Cell> incoming;
  incoming.reserve(declaration_.incomingTopCells.size());
  std::size_t width = 0;
  for (const auto &cell : declaration_.incomingTopCells) {
    Cell sorted = sortedUnique(cell);
    if (sorted.size() != cell.size())
      throw std::invalid_argument(
          "MappingCylinder: an incoming top cell repeats a vertex, so it is "
          "degenerate and names no simplex");
    if (width == 0)
      width = sorted.size();
    else if (sorted.size() != width)
      throw std::invalid_argument(
          "MappingCylinder: the incoming top cells are not pure; one carries " +
          std::to_string(sorted.size()) + " vertices and another " +
          std::to_string(width));
    incoming.push_back(std::move(sorted));
  }
  std::sort(incoming.begin(), incoming.end());
  incoming.erase(std::unique(incoming.begin(), incoming.end()), incoming.end());
  read_.incomingDimension = static_cast<int>(width) - 1;
  read_.cylinderDimension = read_.incomingDimension + 1;

  std::set<std::uint64_t> incomingVertexSet;
  for (const Cell &cell : incoming)
    incomingVertexSet.insert(cell.begin(), cell.end());
  read_.incomingVertices.assign(incomingVertexSet.begin(),
                                incomingVertexSet.end());

  std::set<std::uint64_t> responseVertexSet;
  for (const std::uint64_t vertex : read_.incomingVertices) {
    const auto image = declaration_.reductionMap.find(vertex);
    if (image == declaration_.reductionMap.end())
      throw std::invalid_argument(
          "MappingCylinder: the reduction map carries no image for the "
          "incoming vertex " +
          std::to_string(vertex) + ", so it is not a map on the whole complex");
    if (incomingVertexSet.count(image->second) != 0)
      throw std::invalid_argument(
          "MappingCylinder: the response vertex " +
          std::to_string(image->second) +
          " is also an incoming vertex, and the cylinder holds both ends at "
          "once, so their identifiers must be disjoint");
    responseVertexSet.insert(image->second);
  }
  read_.responseVertices.assign(responseVertexSet.begin(),
                                responseVertexSet.end());

  std::vector<Cell> outgoing;
  std::size_t outgoingWidth = 0;
  for (const auto &cell : declaration_.outgoingTopCells) {
    Cell sorted = sortedUnique(cell);
    if (sorted.size() != cell.size())
      throw std::invalid_argument(
          "MappingCylinder: an outgoing top cell repeats a vertex, so it is "
          "degenerate and names no simplex");
    if (outgoingWidth == 0)
      outgoingWidth = sorted.size();
    else if (sorted.size() != outgoingWidth)
      throw std::invalid_argument(
          "MappingCylinder: the outgoing top cells are not pure; one carries " +
          std::to_string(sorted.size()) + " vertices and another " +
          std::to_string(outgoingWidth));
    for (const std::uint64_t vertex : sorted)
      if (responseVertexSet.count(vertex) == 0)
        throw std::invalid_argument(
            "MappingCylinder: the outgoing top cell names the vertex " +
            std::to_string(vertex) +
            ", which no response vertex of the reduction map carries; the "
            "interactions attach cells among response vertices and nowhere "
            "else");
    outgoing.push_back(std::move(sorted));
  }
  std::sort(outgoing.begin(), outgoing.end());
  outgoing.erase(std::unique(outgoing.begin(), outgoing.end()), outgoing.end());
  read_.outgoingDimension =
      outgoingWidth == 0 ? -1 : static_cast<int>(outgoingWidth) - 1;

  // The staircase triangulation of each prism, with the degenerate simplices
  // dropped.
  std::set<Cell> cylinder;
  std::set<Cell> imageCandidates;
  for (const Cell &cell : incoming) {
    Cell images;
    images.reserve(cell.size());
    for (const std::uint64_t vertex : cell)
      images.push_back(declaration_.reductionMap.at(vertex));
    imageCandidates.insert(sortedUnique(images));
    for (std::size_t step = 0; step < cell.size(); ++step) {
      Cell simplex(cell.begin(),
                   cell.begin() + static_cast<std::ptrdiff_t>(step) + 1);
      bool degenerate = false;
      std::set<std::uint64_t> seen;
      for (std::size_t later = step; later < cell.size(); ++later) {
        if (!seen.insert(images[later]).second) {
          degenerate = true;
          break;
        }
        simplex.push_back(images[later]);
      }
      if (degenerate) continue;
      std::sort(simplex.begin(), simplex.end());
      cylinder.insert(std::move(simplex));
    }
  }
  read_.cylinderTopCells.assign(cylinder.begin(), cylinder.end());

  // The maximal image cells: the images of the incoming top cells, with any
  // cell contained in another removed.
  for (const Cell &candidate : imageCandidates) {
    bool maximal = true;
    for (const Cell &other : imageCandidates)
      if (other != candidate && contains(other, candidate)) {
        maximal = false;
        break;
      }
    if (maximal) read_.imageTopCells.push_back(candidate);
  }

  for (const std::uint64_t vertex : read_.incomingVertices)
    read_.fiberEdges.emplace_back(vertex,
                                  declaration_.reductionMap.at(vertex));

  std::set<std::pair<std::uint64_t, std::uint64_t>> cross;
  for (const Cell &simplex : read_.cylinderTopCells)
    for (const std::uint64_t first : simplex)
      for (const std::uint64_t second : simplex) {
        if (incomingVertexSet.count(first) == 0) continue;
        if (responseVertexSet.count(second) == 0) continue;
        cross.emplace(first, second);
      }
  read_.crossEdges.assign(cross.begin(), cross.end());

  // The facets of the cylinder carried by exactly one top cell, split by end.
  std::map<Cell, int> cofaces;
  for (const Cell &simplex : read_.cylinderTopCells)
    for (std::size_t omitted = 0; omitted < simplex.size(); ++omitted) {
      Cell facet;
      facet.reserve(simplex.size() - 1);
      for (std::size_t position = 0; position < simplex.size(); ++position)
        if (position != omitted) facet.push_back(simplex[position]);
      ++cofaces[facet];
    }
  for (const auto &entry : cofaces) {
    if (entry.second != 1) continue;
    std::size_t incomingCount = 0;
    for (const std::uint64_t vertex : entry.first)
      if (incomingVertexSet.count(vertex) != 0) ++incomingCount;
    if (incomingCount == entry.first.size())
      read_.incomingFreeFacets.push_back(entry.first);
    else if (incomingCount == 0)
      read_.outgoingFreeFacets.push_back(entry.first);
    else
      read_.sideFreeFacets.push_back(entry.first);
  }

  std::size_t checked = 0;
  std::size_t failed = 0;

  std::set<Cell> incomingSet(incoming.begin(), incoming.end());
  std::set<Cell> incomingFree(read_.incomingFreeFacets.begin(),
                              read_.incomingFreeFacets.end());
  read_.incomingBoundaryIsTheIncomingComplex = incomingFree == incomingSet;
  checked += incomingSet.size() + incomingFree.size();
  for (const Cell &cell : incomingSet)
    if (incomingFree.count(cell) == 0) ++failed;
  for (const Cell &cell : incomingFree)
    if (incomingSet.count(cell) == 0) ++failed;

  read_.outgoingComplexContainsTheImage = !read_.imageTopCells.empty();
  for (const Cell &image : read_.imageTopCells) {
    ++checked;
    bool carried = false;
    for (const Cell &cell : outgoing)
      if (contains(cell, image)) {
        carried = true;
        break;
      }
    if (!carried) {
      read_.outgoingComplexContainsTheImage = false;
      ++failed;
    }
  }

  read_.hasNoSideWall = read_.sideFreeFacets.empty();
  checked += read_.sideFreeFacets.size();
  failed += read_.sideFreeFacets.size();

  read_.boundaryIsTheDisjointUnion =
      read_.incomingBoundaryIsTheIncomingComplex &&
      read_.outgoingComplexContainsTheImage && read_.hasNoSideWall;
  read_.boundaryResidual =
      checked == 0 ? 1.0
                   : static_cast<double>(failed) / static_cast<double>(checked);
  read_.certificate = Certificate::algebraicallyExact(
      CertificateDomain::Static, CertificateRegime::PositiveSemidefinite,
      read_.boundaryResidual, 0.0);
}

}  // namespace tessera::cobordism
