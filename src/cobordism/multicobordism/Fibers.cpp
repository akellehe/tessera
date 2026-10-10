// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

// MultiCobordism (include/cobordism/MultiCobordism.h): fiber residuals,
// block geometry, input frames and the fiber setters. One of the translation
// units that define the class's members by responsibility
// (https://github.com/akellehe/tessera/issues/1481).

#include "Internal.h"

namespace tessera::cobordism {

// ---- fiber-form boundary targets ----

double MultiCobordism::fiberResidualOn(const std::shared_ptr<Spacetime> &spacetime,
                                       const BoundaryFiber &target, FiberBand band) const {
  if (target.images.cols() == 0)
    throw std::logic_error("MultiCobordism::fiberResidualOn: the fiber target has no images");
  if (metricSource_ != HodgeLaplacian::MetricSource::WhitneyPencil)
    throw std::logic_error("MultiCobordism::fiberResidualOn: the fiber residual is read on the chain-level "
                           "Whitney pencil; this node uses the diagonal-weight metric");
  const double targetNorm = target.images.squaredNorm();
  if (!(targetNorm > 0.0))
    throw std::logic_error("MultiCobordism::fiberResidualOn: the fiber target is zero");
  if (!spacetime) return 1.0;  // no complex to read: the target leaks in full
  // A candidate geometry the pencil refuses (a singular dressed metric, a
  // branch or allowability failure, a cell outside the complex) cannot carry
  // the state: it scores as the full leak, exactly as a block with no emerged
  // register does under the period residual. Contract errors still propagate.
  BoundaryFiber read;
  try {
    const AssembledPencil assembled = PencilLayer::assemble({spacetime});
    if (assembled.dimension() < target.degree) return 1.0;
    const std::vector<int> idx = PencilLayer::indicesOf(assembled, target.degree, target.cells);
    if (idx.size() != target.cells.size()) return 1.0;
    const chainhodge::Contour contour = fiberContourOn(assembled, target, band);
    read = PencilLayer::readBoundaryFiber(assembled, target.degree, contour, target.cells);
  } catch (const std::runtime_error &) {
    return 1.0;
  } catch (const std::invalid_argument &) {
    return 1.0;
  }
  if (read.images.cols() == 0 || read.images.rows() != target.images.rows()) return 1.0;
  // Least-squares fit of the target images in the band's images on the cells.
  const Eigen::MatrixXcd coefficients = read.images.colPivHouseholderQr().solve(target.images);
  const double leak = (read.images * coefficients - target.images).squaredNorm();
  return leak / targetNorm;
}

chainhodge::Contour MultiCobordism::fiberContourOn(const AssembledPencil &assembled,
                                                   const BoundaryFiber &target, FiberBand band) {
  switch (band) {
    case FiberBand::ZeroMode:
      // The zero mode of this pencil: the block's own Laplacian's harmonic
      // space, whatever contour the fiber stores (see `fiberBandFor`).
      return PencilLayer::harmonicContour(assembled, target.degree);
    case FiberBand::AsStored:
      break;
  }
  return target.contour.nodes.empty() ? PencilLayer::bandContour(assembled, target.degree, 1)
                                      : target.contour;
}

bool MultiCobordism::adoptParentEdgeGeometry(Spacetime &child, const Spacetime &parent) {
  std::map<std::pair<std::uint64_t, std::uint64_t>, ::tessera::mesh::Edge *> parentEdges;
  for (auto *edge : parent.getEdgeList()->toVector()) parentEdges.emplace(edgeKey(edge), edge);
  for (auto *edge : child.getEdgeList()->toVector()) {
    const auto found = parentEdges.find(edgeKey(edge));
    if (found == parentEdges.end()) return false;
    // Through the canonical record, so a parent edge stored the other way
    // round from the child's hands over the phase on the child's orientation.
    restoreEdgeGeometry(*edge, edgeGeometryOf(*found->second));
  }
  return true;
}

std::shared_ptr<Spacetime> MultiCobordism::blockSubcomplexWithGeometry(
    const BoundaryBlock &block, const std::shared_ptr<Spacetime> &spacetime) {
  auto sub = spacetime->subcomplexWithinVertexSet(block.vertices);
  if (!sub) return nullptr;
  if (!adoptParentEdgeGeometry(*sub, *spacetime))
    throw std::logic_error("MultiCobordism::blockSubcomplexWithGeometry: a block edge is absent from the parent");
  sub->materializeFacets();
  return sub;
}

std::shared_ptr<Spacetime> MultiCobordism::blockSurfaceWithGeometry(
    const BoundaryBlock &block, const std::shared_ptr<Spacetime> &spacetime) {
  if (!spacetime) return nullptr;
  const BlockSurface surface = blockSurface(block, *spacetime);
  if (surface.faces.empty()) return nullptr;
  // The surface's own (d-1)-simplices as the top cells of a (d-1)-dimensional
  // complex on the host's vertex ids; fromVertexTuples auto-wires their edges, which
  // then take the host's live lengths and phases by vertex pair. The bulk's
  // cells never enter: this is the block's own Laplacian, not a restriction
  // of the whole's.
  auto own = Spacetime::fromVertexTuples(spacetime->getDimensions() - 1, surface.faces, 1.0, complexd(0.0, 0.0));
  if (!adoptParentEdgeGeometry(*own, *spacetime)) return nullptr;  // a torn surface carries nothing
  own->materializeFacets();
  return own;
}

std::shared_ptr<Spacetime> MultiCobordism::blockComplexWithGeometry(
    const BoundaryBlock &block, const std::shared_ptr<Spacetime> &spacetime) {
  return block.surface ? blockSurfaceWithGeometry(block, spacetime)
                       : blockSubcomplexWithGeometry(block, spacetime);
}

double MultiCobordism::fiberResidualForBoundaryBlock(
    const BoundaryBlock &boundaryBlock, const std::shared_ptr<Spacetime> &spacetime) const {
  if (!boundaryBlock.fiber || boundaryBlock.fiber->images.cols() == 0)
    throw std::logic_error("MultiCobordism::fiberResidualForBoundaryBlock: the block carries no fiber target");
  return fiberResidualOn(blockComplexWithGeometry(boundaryBlock, spacetime), *boundaryBlock.fiber,
                         fiberBandFor(boundaryBlock));
}

// ---- two-body cobordism map ----

void MultiCobordism::attachInputFiber(std::size_t index, BoundaryFiber fiber,
                                      std::vector<std::vector<std::uint64_t>> cells) {
  if (index >= inputBlocks_.size())
    throw std::out_of_range("MultiCobordism::attachInputFiber: input block index out of range");
  if (fiber.images.cols() == 0)
    throw std::invalid_argument("MultiCobordism::attachInputFiber: the fiber has no images");
  if (cells.size() != static_cast<std::size_t>(fiber.images.rows()))
    throw std::invalid_argument("MultiCobordism::attachInputFiber: one attachment cell per fiber row");
  const ChainComplex K = ChainComplex::fromSpacetime(*spacetime_);
  std::set<std::vector<std::uint64_t>> live;
  for (auto c : K.kSimplexVertices(fiber.degree)) {
    std::sort(c.begin(), c.end());
    live.insert(c);
  }
  std::set<std::vector<std::uint64_t>> seen;
  for (auto &c : cells) {
    std::sort(c.begin(), c.end());
    if (!live.count(c))
      throw std::invalid_argument("MultiCobordism::attachInputFiber: an attachment cell is absent from the "
                                  "live complex at degree " + std::to_string(fiber.degree));
    if (!seen.insert(c).second)
      throw std::invalid_argument("MultiCobordism::attachInputFiber: an attachment cell is repeated");
  }
  for (std::size_t other = 0; other < inputBlocks_.size(); ++other) {
    if (other == index || !inputBlocks_[other].fiber) continue;
    for (const auto &c : inputBlocks_[other].fiber->cells)
      if (seen.count(c))
        throw std::invalid_argument("MultiCobordism::attachInputFiber: attachment cell overlaps input fiber " +
                                    std::to_string(other));
  }
  // Attaching a fiber to cells makes those cells the block's: the block's
  // region grows to contain them, so the block's own sub-complex reads them
  // (a cell outside the region would score as the full leak forever).
  for (const auto &c : cells)
    for (const std::uint64_t v : c) inputBlocks_[index].vertices.insert(v);
  fiber.cells = std::move(cells);
  inputBlocks_[index].fiber = std::move(fiber);
  // A frame's rows are the cells of the attachment they were stated for.
  inputBlocks_[index].frame.reset();
}

void MultiCobordism::setInputFrame(std::size_t index, std::vector<std::vector<std::uint64_t>> cells,
                                   Eigen::MatrixXcd images, Eigen::MatrixXcd dualImages) {
  if (index >= inputBlocks_.size())
    throw std::out_of_range("MultiCobordism::setInputFrame: input block index out of range");
  const auto &fiber = inputBlocks_[index].fiber;
  if (!fiber || fiber->images.cols() == 0)
    throw std::logic_error("MultiCobordism::setInputFrame: input block " + std::to_string(index) +
                           " carries no attached fiber; attach the fiber first (attachInputFiber)");
  if (cells.size() != fiber->cells.size())
    throw std::invalid_argument("MultiCobordism::setInputFrame: the frame names " + std::to_string(cells.size()) +
                                " cells but input block " + std::to_string(index) + "'s fiber is attached to " +
                                std::to_string(fiber->cells.size()));
  for (std::size_t i = 0; i < cells.size(); ++i) {
    std::sort(cells[i].begin(), cells[i].end());
    if (cells[i] != fiber->cells[i])
      throw std::invalid_argument("MultiCobordism::setInputFrame: frame row " + std::to_string(i) +
                                  " is not input block " + std::to_string(index) +
                                  "'s attached fiber cell at that row (same cells in the attachment order)");
  }
  const auto rows = static_cast<Eigen::Index>(cells.size());
  if (images.rows() != rows || dualImages.rows() != rows)
    throw std::invalid_argument("MultiCobordism::setInputFrame: the images (" + std::to_string(images.rows()) +
                                " rows) and the dual images (" + std::to_string(dualImages.rows()) +
                                " rows) must have one row per cell (" + std::to_string(rows) + ")");
  if (images.cols() == 0)
    throw std::invalid_argument("MultiCobordism::setInputFrame: the frame has no columns");
  if (dualImages.cols() != images.cols())
    throw std::invalid_argument("MultiCobordism::setInputFrame: the dual images have " +
                                std::to_string(dualImages.cols()) + " columns but the images have " +
                                std::to_string(images.cols()) + " (a frame and its dual share the rank)");
  if (!images.allFinite() || !dualImages.allFinite())
    throw std::invalid_argument("MultiCobordism::setInputFrame: a frame entry is not finite");
  inputBlocks_[index].frame = BlockFrame{std::move(cells), std::move(images), std::move(dualImages)};
}

const std::optional<MultiCobordism::BlockFrame> &MultiCobordism::inputFrame(std::size_t index) const {
  if (index >= inputBlocks_.size())
    throw std::out_of_range("MultiCobordism::inputFrame: input block index out of range");
  return inputBlocks_[index].frame;
}

void MultiCobordism::clearInputFrame(std::size_t index) {
  if (index >= inputBlocks_.size())
    throw std::out_of_range("MultiCobordism::clearInputFrame: input block index out of range");
  inputBlocks_[index].frame.reset();
}

Eigen::MatrixXcd MultiCobordism::dualFrame(const std::shared_ptr<Spacetime> &complex, int degree,
                                           const std::vector<std::vector<std::uint64_t>> &cells,
                                           const Eigen::MatrixXcd &images) {
  if (!complex) throw std::invalid_argument("MultiCobordism::dualFrame: null complex");
  if (images.cols() == 0) throw std::invalid_argument("MultiCobordism::dualFrame: the frame has no columns");
  if (static_cast<std::size_t>(images.rows()) != cells.size())
    throw std::invalid_argument("MultiCobordism::dualFrame: one image row per cell (" +
                                std::to_string(images.rows()) + " rows, " + std::to_string(cells.size()) + " cells)");
  const AssembledPencil assembled = PencilLayer::assemble({complex});
  if (degree < 0 || degree > assembled.dimension())
    throw std::invalid_argument("MultiCobordism::dualFrame: degree " + std::to_string(degree) +
                                " is outside the complex's dimension " + std::to_string(assembled.dimension()));
  std::vector<std::vector<std::uint64_t>> sorted = cells;
  for (auto &c : sorted) std::sort(c.begin(), c.end());
  // A cell absent from the complex at this degree is refused by name here.
  const std::vector<int> idx = PencilLayer::indicesOf(assembled, degree, sorted);
  // B = Z^T M_k Z on the cells: the frame's pairing under the transpose pairing
  // of the complex's own pencil (M_k the Whitney mass matrix, the chain
  // metric's inverse, as PencilLayer::readBoundaryFiber's Gram uses it).
  const Eigen::MatrixXcd M(assembled.op->Minv(degree));
  const auto n = static_cast<Eigen::Index>(idx.size());
  Eigen::MatrixXcd MBB(n, n);
  for (Eigen::Index i = 0; i < n; ++i)
    for (Eigen::Index j = 0; j < n; ++j)
      MBB(i, j) = M(idx[static_cast<std::size_t>(i)], idx[static_cast<std::size_t>(j)]);
  const Eigen::MatrixXcd pairing = images.transpose() * MBB * images;
  Eigen::FullPivLU<Eigen::MatrixXcd> lu(pairing);
  if (!lu.isInvertible())
    throw std::runtime_error("MultiCobordism::dualFrame: the frame's pairing Z^T M Z is singular (an isotropic "
                             "frame); no dual under the transpose pairing exists");
  // Z^vee = Z B^{-T}, i.e. (Z^vee)^T = B^{-1} Z^T, so that (Z^vee)^T M Z = B^{-1} B = I.
  return lu.solve(images.transpose()).transpose();
}

Eigen::MatrixXcd MultiCobordism::inputFrameDual(std::size_t index, const Eigen::MatrixXcd &images) const {
  if (index >= inputBlocks_.size())
    throw std::out_of_range("MultiCobordism::inputFrameDual: input block index out of range");
  const BoundaryBlock &block = inputBlocks_[index];
  if (!block.fiber || block.fiber->images.cols() == 0)
    throw std::logic_error("MultiCobordism::inputFrameDual: input block " + std::to_string(index) +
                           " carries no attached fiber");
  const auto own = blockComplexWithGeometry(block, spacetime_);
  if (!own)
    throw std::logic_error("MultiCobordism::inputFrameDual: input block " + std::to_string(index) +
                           " has no own complex to pair on");
  return dualFrame(own, block.fiber->degree, block.fiber->cells, images);
}

void MultiCobordism::setWholeComplexFiberTarget(BoundaryFiber fiber) {
  if (fiber.images.cols() == 0)
    throw std::invalid_argument("MultiCobordism::setWholeComplexFiberTarget: the fiber has no images");
  if (fiber.cells.size() != static_cast<std::size_t>(fiber.images.rows()))
    throw std::invalid_argument("MultiCobordism::setWholeComplexFiberTarget: one image row per cell");
  for (auto &c : fiber.cells) std::sort(c.begin(), c.end());
  wholeFiberTarget_ = std::move(fiber);
}

double MultiCobordism::wholeComplexFiberResidual() const {
  if (!wholeFiberTarget_)
    throw std::logic_error("MultiCobordism::wholeComplexFiberResidual: no whole-complex fiber target");
  return fiberResidualOn(spacetime_, *wholeFiberTarget_);
}

BoundaryFiber MultiCobordism::readWholeComplexFiber(const chainhodge::Contour *contour,
                                                    double kappa) const {
  if (!wholeFiberTarget_)
    throw std::logic_error("MultiCobordism::readWholeComplexFiber: no whole-complex fiber target");
  if (metricSource_ != HodgeLaplacian::MetricSource::WhitneyPencil)
    throw std::logic_error("MultiCobordism::readWholeComplexFiber: read on the chain-level Whitney pencil; "
                           "this node uses the diagonal-weight metric");
  const AssembledPencil assembled = PencilLayer::assemble({spacetime_});
  const BoundaryFiber &target = *wholeFiberTarget_;
  const chainhodge::Contour chosen =
      contour ? *contour
              : (target.contour.nodes.empty() ? PencilLayer::bandContour(assembled, target.degree, 1)
                                              : target.contour);
  return PencilLayer::readBoundaryFiber(assembled, target.degree, chosen, target.cells, kappa);
}

double MultiCobordism::fiberResidualForInputBlock(std::size_t index) const {
  if (index >= inputBlocks_.size())
    throw std::out_of_range("MultiCobordism::fiberResidualForInputBlock: input block index out of range");
  return fiberResidualForBoundaryBlock(inputBlocks_[index], spacetime_);
}

void MultiCobordism::setInputFiber(std::size_t index, BoundaryFiber fiber) {
  if (index >= inputBlocks_.size())
    throw std::out_of_range("MultiCobordism::setInputFiber: input block index out of range");
  inputBlocks_[index].fiber = std::move(fiber);
}

void MultiCobordism::setOutputFiber(std::size_t index, BoundaryFiber fiber) {
  if (index >= outputBlocks_.size())
    throw std::out_of_range("MultiCobordism::setOutputFiber: output block index out of range");
  outputBlocks_[index].fiber = std::move(fiber);
}

const std::optional<BoundaryFiber> &MultiCobordism::inputFiber(std::size_t index) const {
  if (index >= inputBlocks_.size())
    throw std::out_of_range("MultiCobordism::inputFiber: input block index out of range");
  return inputBlocks_[index].fiber;
}

const std::optional<BoundaryFiber> &MultiCobordism::outputFiber(std::size_t index) const {
  if (index >= outputBlocks_.size())
    throw std::out_of_range("MultiCobordism::outputFiber: output block index out of range");
  return outputBlocks_[index].fiber;
}

MultiCobordism::FixedBoundaryEigenstateResult MultiCobordism::pinInputFibers(
    int degree, double epsilon, int restarts, int maxGrowth, std::uint64_t seed, int maxIterations) {
  if (inputBlocks_.size() != 2 || !inputBlocks_[0].fiber || !inputBlocks_[1].fiber)
    throw std::invalid_argument(
        "MultiCobordism::pinInputFibers: exactly two input blocks carrying fibers are required");
  const BoundaryFiber &A = *inputBlocks_[0].fiber;
  const BoundaryFiber &B = *inputBlocks_[1].fiber;
  if (A.degree != degree || B.degree != degree)
    throw std::invalid_argument("MultiCobordism::pinInputFibers: the input fibers are not at degree " +
                                std::to_string(degree));
  if (A.rank() != 1 || B.rank() != 1)
    throw std::invalid_argument("MultiCobordism::pinInputFibers: the input fibers have ranks " +
                                std::to_string(A.rank()) + " and " + std::to_string(B.rank()) +
                                "; the fixed-boundary fit pins one state, so only rank-one fibers are "
                                "pinned (a joint multi-column fit is not approximated column by column)");
  std::vector<std::vector<std::uint64_t>> support;
  std::vector<complexd> target;
  std::set<std::vector<std::uint64_t>> seen;
  auto append = [&](const BoundaryFiber &f, const char *name) {
    for (Eigen::Index r = 0; r < f.images.rows(); ++r) {
      std::vector<std::uint64_t> cell;
      for (const auto v : f.cells[static_cast<std::size_t>(r)]) cell.push_back(static_cast<std::uint64_t>(v));
      std::sort(cell.begin(), cell.end());
      if (!seen.insert(cell).second)
        throw std::invalid_argument(std::string("MultiCobordism::pinInputFibers: ") + name +
                                    " overlaps the other input fiber on a boundary cell; the two inputs "
                                    "are the disjoint components of the boundary");
      support.push_back(std::move(cell));
      target.push_back(f.images(r, 0));
    }
  };
  append(A, "input fiber 0");
  append(B, "input fiber 1");
  return relaxFixedBoundaryEigenstate(degree, std::move(support), std::move(target), epsilon, restarts,
                                      maxGrowth, seed, maxIterations);
}

BoundaryFiber MultiCobordism::readOutputFiber(std::size_t index, int degree,
                                              const chainhodge::Contour *contour, double kappa) {
  if (index >= outputBlocks_.size())
    throw std::out_of_range("MultiCobordism::readOutputFiber: output block index out of range");
  if (metricSource_ != HodgeLaplacian::MetricSource::WhitneyPencil)
    throw std::logic_error("MultiCobordism::readOutputFiber: the fiber form of a target is read on the "
                           "chain-level Whitney pencil; this node uses the diagonal-weight metric");
  const AssembledPencil assembled = PencilLayer::assemble({spacetime_});
  const std::vector<std::uint64_t> region(outputBlocks_[index].vertices.begin(),
                                          outputBlocks_[index].vertices.end());
  const std::vector<int> idx = PencilLayer::cellsWithin(assembled, degree, region);
  if (idx.empty())
    throw std::invalid_argument("MultiCobordism::readOutputFiber: the output block carries no degree-" +
                                std::to_string(degree) + " cell");
  const auto cells = assembled.complex().kSimplexVertices(degree);
  std::vector<std::vector<std::uint64_t>> blockCells;
  for (const int j : idx) blockCells.push_back(cells[static_cast<std::size_t>(j)]);
  const chainhodge::Contour chosen =
      contour ? *contour : PencilLayer::harmonicContour(assembled, degree);
  BoundaryFiber fiber = PencilLayer::readBoundaryFiber(assembled, degree, chosen, blockCells, kappa);
  outputBlocks_[index].fiber = fiber;
  return fiber;
}

}  // namespace tessera::cobordism
