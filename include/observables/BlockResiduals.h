// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_OBSERVABLES_BLOCK_RESIDUALS_H
#define TESSERA_OBSERVABLES_BLOCK_RESIDUALS_H

#include <complex>
#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "observables/RegisterObservable.h"

namespace tessera::observables {

/// # BlockResiduals
///
/// Per-output-block carry residuals. Each provenance block (a vertex region plus
/// a target) is scored against its own sub-complex, as
/// `ProtonIngredients::outputBlockResidual` scores it: the ambient top cells all
/// of whose vertices lie in the region form the block's sub-complex, carrying a
/// uniform metric, and are scored with
/// `MultiCobordism::residualOfTargetStateAgainstHarmonic`. An empty region
/// reports the full leak \f$ \|\mathrm{target}\|^2 \f$.
///
/// Blocks are supplied to the constructor from recorded build history, never
/// inferred. The sub-complex is loaded by `LiveComplex` — a strict selection of
/// existing cells re-instantiated through `Spacetime::fromVertexTuples` — and never
/// built inside this reader. Block regions carry vertex ids, so
/// `recordRelabeled` maps them through the RELABEL permutation.
class BlockResiduals : public RegisterObservable {
  public:
    /// One provenance block: a label, its emergent vertex region, and its
    /// register target.
    struct Block {
      std::string label;
      std::vector<std::uint64_t> vertices;
      std::vector<std::complex<double>> target;
    };

    explicit BlockResiduals(std::vector<Block> blocks)
        : blocks_(std::move(blocks)) {}

    [[nodiscard]] std::string recordKey() const override {
      return std::string(kRecordKey);
    }
    [[nodiscard]] bool needsProvenance() const override { return true; }
    [[nodiscard]] bool hasProvenance() const override {
      return !blocks_.empty();
    }
    [[nodiscard]] Record record(const RegisterContext &ctx) const override;
    [[nodiscard]] Record recordRelabeled(
        const RegisterContext &ctx,
        const std::map<std::uint64_t, std::uint64_t> &perm) const override;

    static constexpr std::string_view kRecordKey = "block_residuals";

  protected:
    /// The headline is the total carry leak — the sum of the block residuals.
    [[nodiscard]] double computeHeadline(
        const RegisterContext &ctx) const override;

  private:
    /// The record over an explicit block list (shared by `record` and
    /// `recordRelabeled`).
    [[nodiscard]] Record recordForBlocks(
        const RegisterContext &ctx,
        const std::vector<Block> &blocks) const;
    /// One block's carry residual (the full leak when the region has no cell).
    [[nodiscard]] static double blockResidual(const RegisterContext &ctx,
                                              const Block &block,
                                              int &nCellsInRegion,
                                              double &targetNorm2);

    std::vector<Block> blocks_;
};

}  // namespace tessera::observables

#endif  // TESSERA_OBSERVABLES_BLOCK_RESIDUALS_H
