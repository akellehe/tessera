// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "observables/BlockResiduals.h"

#include <set>
#include <string>

#include "cobordism/MultiCobordism.h"
#include "mesh/Simplex.h"
#include "mesh/Vertex.h"
#include "observables/LiveComplex.h"
#include "spacetime/Spacetime.h"

namespace tessera::observables {
using ::tessera::cobordism::MultiCobordism;

namespace {

// The ambient top cells (intrinsic vertex order) whose vertices all lie in the
// region. This reads the live complex; it builds nothing.
std::vector<std::vector<std::uint64_t>> cellsInRegion(
    const RegisterContext &ctx, const std::set<std::uint64_t> &region) {
  std::vector<std::vector<std::uint64_t>> inside;
  for (const auto *cell : ctx.spacetime()->getTopSimplices()) {
    std::vector<std::uint64_t> vids;
    vids.reserve(cell->getVertices().size());
    bool allInside = true;
    for (const auto *v : cell->getVertices()) {
      const std::uint64_t id = v->getId();
      if (!region.count(id)) {
        allInside = false;
        break;
      }
      vids.push_back(id);
    }
    if (allInside) inside.push_back(std::move(vids));
  }
  return inside;
}

}  // namespace

double BlockResiduals::blockResidual(const RegisterContext &ctx,
                                     const Block &block, int &nCellsInRegion,
                                     double &targetNorm2) {
  const std::set<std::uint64_t> region(block.vertices.begin(),
                                       block.vertices.end());
  const auto cellsInside = cellsInRegion(ctx, region);
  nCellsInRegion = static_cast<int>(cellsInside.size());
  targetNorm2 = 0.0;
  for (const auto &t : block.target) targetNorm2 += std::norm(t);
  if (cellsInside.empty()) {
    return targetNorm2;  // full leak: no cell carries the target
  }
  // The block's sub-complex is a selection of existing cells re-instantiated
  // with a uniform metric by the loader.
  auto sub = LiveComplex::subcomplex(cellsInside, ctx.dimensions());
  return MultiCobordism::residualOfTargetStateAgainstHarmonic(
      sub, ctx.degree(), block.target);
}

Record BlockResiduals::recordForBlocks(const RegisterContext &ctx,
                                       const std::vector<Block> &blocks) const {
  Record::List rows;
  for (std::size_t index = 0; index < blocks.size(); ++index) {
    const Block &block = blocks[index];
    int nCellsInRegion = 0;
    double targetNorm2 = 0.0;
    const double residual =
        blockResidual(ctx, block, nCellsInRegion, targetNorm2);
    const std::set<std::uint64_t> region(block.vertices.begin(),
                                         block.vertices.end());

    Record::Map row;
    row["label"] = block.label.empty() ? ("block" + std::to_string(index))
                                       : block.label;
    row["n_region_vertices"] = static_cast<int>(region.size());
    row["n_cells_in_region"] = nCellsInRegion;
    row["full_leak"] = (nCellsInRegion == 0);
    row["residual"] = residual;
    row["target_norm2"] = targetNorm2;
    Record::splitComplex(row, "target", block.target);
    rows.emplace_back(std::move(row));
  }
  Record::Map m;
  m["n_blocks"] = static_cast<int>(rows.size());
  m["blocks"] = std::move(rows);
  return Record(std::move(m));
}

Record BlockResiduals::record(const RegisterContext &ctx) const {
  return recordForBlocks(ctx, blocks_);
}

Record BlockResiduals::recordRelabeled(
    const RegisterContext &ctx,
    const std::map<std::uint64_t, std::uint64_t> &perm) const {
  // Block regions are vertex-id sets; map them through the relabel permutation
  // (targets and labels carry no ids). A region can reference vertices no longer
  // in any top cell (surgical moves orphan them); those ids are inert in the
  // residual and the permutation maps the live set onto itself, so leaving an
  // unmapped id unchanged preserves the region size.
  std::vector<Block> mapped;
  mapped.reserve(blocks_.size());
  for (const Block &block : blocks_) {
    Block b = block;
    for (std::uint64_t &v : b.vertices) {
      auto it = perm.find(v);
      if (it != perm.end()) v = it->second;
    }
    mapped.push_back(std::move(b));
  }
  return recordForBlocks(ctx, mapped);
}

double BlockResiduals::computeHeadline(const RegisterContext &ctx) const {
  double total = 0.0;
  for (const Block &block : blocks_) {
    int nCellsInRegion = 0;
    double targetNorm2 = 0.0;
    total += blockResidual(ctx, block, nCellsInRegion, targetNorm2);
  }
  return total;
}

}  // namespace tessera::observables
