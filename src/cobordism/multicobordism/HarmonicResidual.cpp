// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

// MultiCobordism (include/cobordism/MultiCobordism.h): the harmonic target
// residuals: Betti numbers, emergent holes, hole period matrices, target
// relabeling and the boundary-block residuals. One of the translation units
// that define the class's members by responsibility
// (https://github.com/akellehe/tessera/issues/1481).

#include "Internal.h"

namespace tessera::cobordism {

double MultiCobordism::residualForBoundaryBlock(
    const BoundaryBlock &boundaryBlock,
    const std::shared_ptr<Spacetime> &spacetime) const {
  std::set<std::vector<int>> claimedMatchings;
  return residualForBoundaryBlockWithDistinctMatchings(boundaryBlock, spacetime,
                                                       claimedMatchings);
}

double MultiCobordism::residualForBoundaryBlockWithDistinctMatchings(
    const BoundaryBlock &boundaryBlock,
    const std::shared_ptr<Spacetime> &spacetime,
    std::set<std::vector<int>> &claimedMatchings) const {
  // A marked block (setInputMarking) is scored by the residual of its own
  // state: the leak of its input state in the holomorphic form of its own
  // Laplacian on its live surface. The zero mode of the entire cobordism is the
  // output state, reported by readInputState in the block's live frame.
  //
  // Under scoreWholeComplexLeak the whole's leak is added to that own residual
  // rather than replacing it, so the block's own-Laplacian residual stays in
  // the objective beside the bulk terms. The own term is identically zero on a
  // block whose edges cannot move, which is exactly when the whole's leak is
  // the only thing left to say how the block sits in the cobordism.
  if (useFiberResiduals_ && boundaryBlock.marking) {
    const double own = ownStateResidualOn(boundaryBlock, spacetime);
    if (!scoreWholeComplexLeak_) return own;
    return own + inputStateResidualOn(boundaryBlock, spacetime);
  }
  if (useFiberResiduals_ && boundaryBlock.fiber && boundaryBlock.fiber->images.cols() > 0)
    return fiberResidualForBoundaryBlock(boundaryBlock, spacetime);
  auto blockSubcomplex = spacetime->getRegion(
    boundaryBlock.vertices);
  double residual = 0.0;
  if (!blockSubcomplex)  // no complex to read: the target leaks in full, per degree
    return static_cast<double>(registerDegrees_.size()) *
           Eigen::Map<const Eigen::VectorXcd>(
               boundaryBlock.target.data(),
               static_cast<Eigen::Index>(boundaryBlock.target.size()))
               .squaredNorm();
  // The sub-complex is a fresh spacetime whose per-instance Betti slot is
  // empty, so without help every evaluation would re-run the Smith normal form
  // once per block, per line-search trial, per candidate. The block's topology
  // is a pure function of the parent's cells and the vertex set, so the parent
  // caches the numbers per (structural revision, region fingerprint): on a hit,
  // pre-seed the child's slot so betti() below never computes; on a miss, store
  // the child's freshly computed numbers back on the parent.
  //
  // The region is named by `Fingerprint::fingerprintOf` over its vertex ids,
  // called as a static because a `Fingerprint` instance holds only `kMax`
  // identifiers and drops the rest, while a block region grows across the
  // complex.
  const std::uint64_t vertexSetKey =
      ::tessera::mesh::Fingerprint::fingerprintOf(boundaryBlock.vertices);
  if (const auto *cached =
          spacetime->cachedSubcomplexBettiNumbers(vertexSetKey))
    blockSubcomplex->storeBettiNumbers(*cached);
  for (int registerDegree : registerDegrees_)
    residual += EigenstateSynthesis::targetStateResidual(
        blockSubcomplex, registerDegree, boundaryBlock.target, claimedMatchings);
  if (const auto *computed = blockSubcomplex->cachedBettiNumbers())
    spacetime->storeSubcomplexBettiNumbers(vertexSetKey, *computed);
  return residual;
}

}  // namespace tessera::cobordism
