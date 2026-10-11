// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

// MultiCobordism (include/cobordism/MultiCobordism.h): surface inputs and the
// bridge move: block surfaces, the fixed boundary, surface inventories,
// uncovered input faces and bridge candidates. One of the translation units
// that define the class's members by responsibility
// (https://github.com/akellehe/tessera/issues/1481).

#include "Internal.h"

namespace tessera::cobordism {

bool MultiCobordism::hasFixedBoundary() const noexcept {
  // A surface block only. Its own triangles are a component of the boundary,
  // declared by the caller, which is what makes dW a stated fact rather than
  // whatever the search exposed. A pinned region is deliberately not one:
  // "pinning constrains the geometry, it does not veto a topology change" is
  // the settled reading of `declarePinnedRegion` (the acceptance in
  // applyMoveSpecification and the ManifoldValidityIsTheOnlyGate tests), and
  // making it imply a topological gate would quietly reverse it.
  for (const auto &block : inputBlocks_)
    if (block.surface) return true;
  for (const auto &block : outputBlocks_)
    if (block.surface) return true;
  return false;
}

MultiCobordism::SurfaceInventory MultiCobordism::surfaceInventoryOf(
    const Spacetime &spacetime) const {
  SurfaceInventory inventory;
  for (const auto &topSimplex : spacetime.getTopSimplices()) {
    auto cell = topSimplex->topTuple();
    for (std::size_t omit = 0; omit < cell.size(); ++omit) {
      std::vector<std::uint64_t> facet;
      for (std::size_t i = 0; i < cell.size(); ++i)
        if (i != omit) facet.push_back(cell[i]);
      ++inventory.facetIncidence[facet];
    }
    inventory.topCells.insert(std::move(cell));
  }
  for (std::size_t index = 0; index < inputBlocks_.size(); ++index) {
    const auto &block = inputBlocks_[index];
    if (!block.surface) continue;
    const BlockSurface surface = block.ownSurface(spacetime);
    std::set<std::vector<std::uint64_t>> faces(surface.faces.begin(), surface.faces.end());
    // Every non-empty subset of a face is a simplex of the surface: the parts
    // a bridge may take from this block.
    std::set<std::vector<std::uint64_t>> simplices;
    for (const auto &face : faces)
      for (std::uint64_t mask = 1; mask < (std::uint64_t{1} << face.size()); ++mask) {
        std::vector<std::uint64_t> part;
        for (std::size_t i = 0; i < face.size(); ++i)
          if (mask & (std::uint64_t{1} << i)) part.push_back(face[i]);
        simplices.insert(std::move(part));
      }
    inventory.blocks.push_back(index);
    inventory.faces.push_back(std::move(faces));
    inventory.simplices.push_back(std::move(simplices));
  }
  return inventory;
}

bool MultiCobordism::hasSurfaceInputs() const {
  std::size_t count = 0;
  for (const auto &block : inputBlocks_)
    if (block.surface) ++count;
  return count >= 2;
}

std::vector<std::vector<std::uint64_t>> MultiCobordism::uncoveredInputFacesOn(
    const Spacetime &spacetime) const {
  std::vector<std::vector<std::uint64_t>> uncovered;
  bool anySurface = false;
  for (const auto &block : inputBlocks_) anySurface = anySurface || block.surface;
  if (!anySurface) return uncovered;
  const SurfaceInventory inventory = surfaceInventoryOf(spacetime);
  for (const auto &faces : inventory.faces)
    for (const auto &face : faces)
      if (!inventory.facetIncidence.count(face)) uncovered.push_back(face);
  return uncovered;
}

std::vector<std::vector<std::uint64_t>> MultiCobordism::uncoveredInputFaces() const {
  return uncoveredInputFacesOn(*spacetime_);
}

bool MultiCobordism::bridgePhaseCompleteOn(const Spacetime &spacetime) const {
  const SurfaceInventory inventory = surfaceInventoryOf(spacetime);
  if (inventory.blocks.empty()) return false;
  // Every surface face has exactly one top cell on it, and the boundary of
  // the top cells is exactly the union of the surface faces. Both read off
  // one facet-incidence count, so the two conditions cannot disagree about
  // what a boundary facet is.
  std::set<std::vector<std::uint64_t>> surfaceFaces;
  for (const auto &faces : inventory.faces) surfaceFaces.insert(faces.begin(), faces.end());
  if (surfaceFaces.empty()) return false;
  for (const auto &face : surfaceFaces) {
    const auto found = inventory.facetIncidence.find(face);
    if (found == inventory.facetIncidence.end() || found->second != 1) return false;
  }
  for (const auto &[facet, count] : inventory.facetIncidence)
    if (count == 1 && !surfaceFaces.count(facet)) return false;
  return true;
}

bool MultiCobordism::bridgePhaseComplete() const { return bridgePhaseCompleteOn(*spacetime_); }

std::vector<std::vector<std::uint64_t>> MultiCobordism::bridgeCandidatesOn(
    const Spacetime &spacetime) const {
  std::vector<std::vector<std::uint64_t>> candidates;
  const SurfaceInventory inventory = surfaceInventoryOf(spacetime);
  if (inventory.blocks.size() < 2) return candidates;
  const std::size_t faceSize = static_cast<std::size_t>(std::max(0, spacetime.getDimensions()));
  // Which surface block (by position in the inventory) each vertex belongs to.
  std::map<std::uint64_t, std::size_t> blockOfVertex;
  std::vector<std::vector<std::uint64_t>> blockVertices(inventory.blocks.size());
  for (std::size_t b = 0; b < inventory.blocks.size(); ++b)
    for (const std::uint64_t v : inputBlocks_[inventory.blocks[b]].vertices) {
      blockOfVertex.emplace(v, b);
      blockVertices[b].push_back(v);
    }
  // The frontier: boundary facets of the top cells that are not surface faces
  // (their other side is still open), and surface faces no top cell covers.
  std::set<std::vector<std::uint64_t>> surfaceFaces;
  for (const auto &faces : inventory.faces) surfaceFaces.insert(faces.begin(), faces.end());
  std::vector<std::vector<std::uint64_t>> frontier;
  for (const auto &[facet, count] : inventory.facetIncidence)
    if (count == 1 && !surfaceFaces.count(facet)) frontier.push_back(facet);
  for (const auto &face : surfaceFaces)
    if (!inventory.facetIncidence.count(face)) frontier.push_back(face);
  // A part of a candidate cell inside block b is admissible when it is a
  // simplex of b's surface — and, when it is a whole face, an uncovered one:
  // a covered surface face already has the one top cell it will ever have.
  const auto partAdmissible = [&](std::size_t b, std::vector<std::uint64_t> part) {
    std::sort(part.begin(), part.end());
    if (!inventory.simplices[b].count(part)) return false;
    if (part.size() == faceSize && inventory.facetIncidence.count(part)) return false;
    return true;
  };
  std::set<std::vector<std::uint64_t>> distinct;
  for (const auto &face : frontier) {
    // The face's vertices by block; a face touching a vertex outside every
    // surface block (a bulk vertex a later cone-in minted) is not bridged.
    std::map<std::size_t, std::vector<std::uint64_t>> parts;
    bool bridgeable = true;
    for (const std::uint64_t v : face) {
      const auto found = blockOfVertex.find(v);
      if (found == blockOfVertex.end()) {
        bridgeable = false;
        break;
      }
      parts[found->second].push_back(v);
    }
    if (!bridgeable || parts.empty() || parts.size() > 2) continue;
    // The blocks a completing vertex may come from: the face's own blocks,
    // or — when the face lies in one block — any other surface block.
    std::vector<std::size_t> sources;
    for (const auto &[b, part] : parts) sources.push_back(b);
    if (parts.size() == 1)
      for (std::size_t b = 0; b < inventory.blocks.size(); ++b)
        if (!parts.count(b)) sources.push_back(b);
    for (const std::size_t b : sources)
      for (const std::uint64_t v : blockVertices[b]) {
        if (std::find(face.begin(), face.end(), v) != face.end()) continue;
        std::map<std::size_t, std::vector<std::uint64_t>> cellParts = parts;
        cellParts[b].push_back(v);
        if (cellParts.size() != 2) continue;  // the split is across exactly two blocks
        bool admissible = true;
        for (const auto &[block, part] : cellParts)
          if (!partAdmissible(block, part)) {
            admissible = false;
            break;
          }
        if (!admissible) continue;
        std::vector<std::uint64_t> cell(face.begin(), face.end());
        cell.push_back(v);
        std::sort(cell.begin(), cell.end());
        if (inventory.topCells.count(cell)) continue;
        if (distinct.insert(cell).second) candidates.push_back(std::move(cell));
      }
  }
  return candidates;
}

}  // namespace tessera::cobordism
