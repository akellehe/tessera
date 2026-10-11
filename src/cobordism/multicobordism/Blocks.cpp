// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

// The boundary blocks' own reads (include/cobordism/multicobordism/Blocks.h):
// a block's surface, its region and surface complexes with the host's
// geometry, its derived frame and its qubit, and a marking's state target.

#include "Internal.h"

namespace tessera::cobordism::multicobordism {

BlockSurface BoundaryBlock::ownSurface(const Spacetime &spacetime) const {
  const BoundaryBlock &block = *this;
  const std::size_t faceSize = static_cast<std::size_t>(std::max(0, spacetime.getDimensions()));
  const auto inside = [&](const std::vector<std::uint64_t> &ids) {
    for (const std::uint64_t v : ids)
      if (!block.vertices.count(v)) return false;
    return true;
  };
  // The faces the block carries (a surface block's own triangles as seeded),
  // sorted so a tuple compares with the host's canonical ones.
  std::set<std::vector<std::uint64_t>> faces;
  for (auto face : block.faces) {
    std::sort(face.begin(), face.end());
    if (face.size() == faceSize && inside(face)) faces.insert(std::move(face));
  }
  // The host's registered (d-1)-simplices inside the block: the surface's own
  // triangles, whether or not a top cell has reached them yet.
  for (const auto &simplex : spacetime.getSimplices()) {
    if (simplex == nullptr || simplex->isStale() || simplex->size() != faceSize) continue;
    auto ids = simplex->topTuple();
    if (inside(ids)) faces.insert(std::move(ids));
  }
  // The facets of the top cells inside the block: a covered surface face that
  // no read has materialized yet is still a face of the surface.
  for (const auto &topSimplex : spacetime.getTopSimplices()) {
    const auto cell = topSimplex->topTuple();
    for (std::size_t omit = 0; omit < cell.size(); ++omit) {
      std::vector<std::uint64_t> facet;
      for (std::size_t i = 0; i < cell.size(); ++i)
        if (i != omit) facet.push_back(cell[i]);
      if (facet.size() == faceSize && inside(facet)) faces.insert(std::move(facet));
    }
  }
  std::set<std::vector<std::uint64_t>> edges;
  for (const auto *edge : spacetime.getEdgeList()->toVector()) {
    if (edge == nullptr || edge->getSource() == nullptr || edge->getTarget() == nullptr) continue;
    const auto key = edge->ascendingIds();
    if (block.vertices.count(key.first) && block.vertices.count(key.second))
      edges.insert({key.first, key.second});
  }
  return {std::vector<std::vector<std::uint64_t>>(faces.begin(), faces.end()),
          std::vector<std::vector<std::uint64_t>>(edges.begin(), edges.end())};
}

std::shared_ptr<Spacetime> BoundaryBlock::ownSurfaceComplex(const std::shared_ptr<Spacetime> &spacetime) const {
  const BoundaryBlock &block = *this;
  if (!spacetime) return nullptr;
  const BlockSurface surface = block.ownSurface(*spacetime);
  if (surface.faces.empty()) return nullptr;
  // The surface's own (d-1)-simplices as the top cells of a (d-1)-dimensional
  // complex on the host's vertex ids; fromVertexTuples auto-wires their edges, which
  // then take the host's live lengths and phases by vertex pair. The bulk's
  // cells never enter: this is the block's own Laplacian, not a restriction
  // of the whole's.
  auto own = Spacetime::fromVertexTuples(spacetime->getDimensions() - 1, surface.faces, 1.0, complexd(0.0, 0.0));
  if (!own->copyEdgeGeometryFrom(*spacetime)) return nullptr;  // a torn surface carries nothing
  own->materializeFacets();
  return own;
}

std::shared_ptr<Spacetime> BoundaryBlock::region(const std::shared_ptr<Spacetime> &spacetime) const {
  const BoundaryBlock &block = *this;
  auto sub = spacetime->getRegion(block.vertices);
  if (!sub) return nullptr;
  sub->materializeFacets();
  return sub;
}

std::shared_ptr<Spacetime> BoundaryBlock::fiberComplex(const std::shared_ptr<Spacetime> &spacetime) const {
  const BoundaryBlock &block = *this;
  return block.surface ? block.ownSurfaceComplex(spacetime)
                       : block.region(spacetime);
}

DerivedFrame BoundaryBlock::deriveFrame(const std::shared_ptr<Spacetime> &spacetime) const {
  const BoundaryBlock &block = *this;
  if (!block.marking) throw std::logic_error("BoundaryBlock::deriveFrame: the block carries no marking");
  const BlockMarking &marking = *block.marking;
  DerivedFrame out;
  const auto own = block.fiberComplex(spacetime);
  if (!own) {
    out.obstruction = "the block has no own complex (a torn surface carries no frame)";
    return out;
  }
  try {
    const AssembledPencil assembled = PencilLayer::assemble({own});
    if (assembled.dimension() < 1) {
      out.obstruction = "the block's own complex has no edges";
      return out;
    }
    // The zero mode of the block's own covariant pencil: its harmonic contour,
    // recomputed here as the lengths move, with the phases entering through
    // the dressed pencil.
    const chainhodge::Contour contour = PencilLayer::harmonicContour(assembled, 1);
    const chainhodge::Band band = assembled.op->band(1, contour);
    out.kernelRank = band.rank();
    if (band.rank() != marking.rank()) {
      out.obstruction = "the block's own kernel has rank " + std::to_string(band.rank()) + " for a marking of " +
                        std::to_string(marking.rank()) +
                        " cycles (a torn surface, or link phases that are not a pure gauge)";
      return out;
    }
    const auto rank = static_cast<Eigen::Index>(marking.rank());
    // Pi_{ca}: the transported period of zero-mode column a over cycle c from
    // the common base point (a marking step absent from the own complex is
    // refused by name by the connection, and read as an obstruction here).
    Eigen::MatrixXcd periods(rank, rank);
    for (Eigen::Index c = 0; c < rank; ++c)
      for (Eigen::Index a = 0; a < rank; ++a)
        periods(c, a) = assembled.op->connection().transportedPeriod(
            band.images.col(a), marking.cycles[static_cast<std::size_t>(c)]);
    Eigen::FullPivLU<Eigen::MatrixXcd> lu(periods);
    if (!lu.isInvertible()) {
      out.obstruction = "the periods of the block's own kernel over the marking are singular: the cycles do not "
                        "span the block's homology";
      return out;
    }
    // F = Z Pi^{-1}: column b has period delta_{cb} over cycle c.
    const Eigen::MatrixXcd F = band.images * lu.inverse();
    // F^vee = Z~ (Z~^T M_1^U F)^{-T} with Z~ the dual kernel's images (the
    // zero mode under the inverse links) and M_1^U the dressed Whitney mass
    // matrix: (F^vee)^T M_1^U F = I, the BlockFrame contract paired between
    // the kernel and the dual kernel.
    const Eigen::MatrixXcd dualImages = assembled.dual->applyG(1, band.dualFrame);
    const Eigen::MatrixXcd M(assembled.op->Minv(1));
    const Eigen::MatrixXcd pairing = dualImages.transpose() * M * F;
    Eigen::FullPivLU<Eigen::MatrixXcd> pairingLu(pairing);
    if (!pairingLu.isInvertible()) {
      out.obstruction = "the pairing of the block's dual kernel against its frame is singular (an isotropic frame)";
      return out;
    }
    out.frame.cells = assembled.complex().kSimplexVertices(1);
    out.frame.images = F;
    out.frame.dualImages = pairingLu.solve(dualImages.transpose()).transpose();
    out.periods = periods;
  } catch (const std::runtime_error &e) {
    out.obstruction = std::string("the block's own pencil refused the read: ") + e.what();
  } catch (const std::invalid_argument &e) {
    out.obstruction = std::string("the block's own pencil refused the read: ") + e.what();
  }
  return out;
}

observables::SimplicialQubit BoundaryBlock::qubit(const std::shared_ptr<Spacetime> &spacetime) const {
  const BoundaryBlock &block = *this;
  if (!block.marking) throw std::logic_error("BoundaryBlock::qubit: the block carries no marking (setInputMarking)");
  if (block.marking->rank() != 2)
    throw std::invalid_argument("BoundaryBlock::qubit: a qubit torus is marked by two cycles; this marking has " +
                                std::to_string(block.marking->rank()));
  if (!spacetime) throw std::invalid_argument("BoundaryBlock::qubit: null spacetime");
  const std::shared_ptr<Spacetime> surface = block.ownSurfaceComplex(spacetime);
  if (!surface)
    throw std::runtime_error("BoundaryBlock::qubit: the block has no surface (a face of the torus lost an "
                             "edge, so it carries no state)");
  // The surface as SimplicialQubit's Spacetime constructor indexes it:
  // Vertices by ascending id, edges by ascending index pair. The marking's
  // steps (host ids, u -> v) become (edge index, sign) with the sign of the
  // step against the edge's stored orientation (min -> max).
  std::vector<std::uint64_t> ids;
  for (const auto *vertex : surface->getVertexList()->liveVector()) ids.push_back(vertex->getId());
  std::sort(ids.begin(), ids.end());
  std::map<std::uint64_t, std::uint64_t> indexOfId;
  for (std::size_t n = 0; n < ids.size(); ++n) indexOfId[ids[n]] = n;
  std::vector<VertexIdPair> pairs;
  for (const mesh::Edge *edge : surface->getEdgeList()->toVector()) {
    const std::uint64_t a = indexOfId.at(edge->getSource()->getId());
    const std::uint64_t b = indexOfId.at(edge->getTarget()->getId());
    pairs.push_back(ascendingIds(a, b));
  }
  std::sort(pairs.begin(), pairs.end());
  std::map<VertexIdPair, std::size_t> indexOfPair;
  for (std::size_t n = 0; n < pairs.size(); ++n) indexOfPair[pairs[n]] = n;
  auto cycleOf = [&](const std::vector<std::pair<std::uint64_t, std::uint64_t>> &steps, const char *name) {
    observables::SimplicialQubit::Cycle cycle;
    for (const auto &[u, v] : steps) {
      const auto iu = indexOfId.find(u), iv = indexOfId.find(v);
      if (iu == indexOfId.end() || iv == indexOfId.end())
        throw std::runtime_error(std::string("BoundaryBlock::qubit: a step of cycle ") + name + " (" +
                                 std::to_string(u) + " -> " + std::to_string(v) +
                                 ") leaves the block's live surface");
      const auto found = indexOfPair.find(ascendingIds(iu->second, iv->second));
      if (found == indexOfPair.end())
        throw std::runtime_error(std::string("BoundaryBlock::qubit: the marking edge (") + std::to_string(u) +
                                 ", " + std::to_string(v) + ") of cycle " + name +
                                 " is not an edge of the block's live surface");
      cycle.emplace_back(found->second, iu->second < iv->second ? 1 : -1);
    }
    return cycle;
  };
  const observables::SimplicialQubit::Cycle cycleA = cycleOf(block.marking->cycles[0], "A");
  const observables::SimplicialQubit::Cycle cycleB = cycleOf(block.marking->cycles[1], "B");
  // The orientation the marking fixes, A . B = +1. The
  // container stores none; the fundamental class picks one of the two, and
  // the intersection number says whether it is the marking's.
  observables::SimplicialQubit read(surface, cycleA, cycleB, /*reversed=*/false);
  if (read.intersectionNumber() < 0.0) return observables::SimplicialQubit(surface, cycleA, cycleB, /*reversed=*/true);
  return read;
}

BoundaryFiber BlockMarking::stateTarget(const BlockFrame &frame) const {
  const BlockMarking &marking = *this;
  BoundaryFiber target;
  target.degree = 1;
  target.cells = frame.cells;
  target.images = frame.images * marking.coefficients;
  return target;
}

}  // namespace tessera::cobordism::multicobordism
