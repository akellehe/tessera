// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_OBSERVABLES_LIVE_COMPLEX_H
#define TESSERA_OBSERVABLES_LIVE_COMPLEX_H

#include <complex>
#include <cstdint>
#include <map>
#include <memory>
#include <utility>
#include <vector>

// === tessera subsystem ns fwd-decls ===
namespace tessera::spacetime {
  class Spacetime;
}
namespace tessera::observables {
using namespace ::tessera::spacetime;

/// # LiveComplex
///
/// Loader and transform layer sitting outside the read-only observables. It
/// loads a saved combinatorial and metric description back into a live,
/// skeleton-complete `Spacetime`, and produces a relabeled copy for the RELABEL
/// gate. It never builds a spacetime of its own and never re-runs the emergent
/// dynamics; those live in Proton, ProtonIngredients and MultiCobordism. A
/// recorded geometry is read back only through `Spacetime::fromCells`:
///
///   * `Spacetime::fromCells` materializes only the top cells (\f$ \partial
///     \Delta^5 \f$ comes back as its 6 pentatopes and nothing else). The
///     facet/coface skeleton that `dualVolume()` and `deficitAngle()` walk is
///     then completed by `Spacetime::materializeFacets()`, which reproduces the
///     `ReggeSolver` plus `ChainComplex::fromSpacetime` skeleton bit for bit
///     (interior hinge census, `V_dual` and `m_sum` all agree).
///   * The metric (complex squared lengths) and per-vertex times are loaded back
///     exactly as recorded. A missing edge length throws; a partial metric is
///     never silently defaulted.
///
/// `RegisterContext` and every Observable then only read the resulting live
/// complex; they never see a dump and never build anything.
class LiveComplex {
  public:
    /// A relabeled rebuild: the live relabeled complex plus the vertex-id
    /// permutation that produced it (original id → relabeled id), so
    /// vertex-id-bearing observable configuration (e.g. provenance block
    /// regions) can be mapped through it.
    struct Relabeled {
      std::shared_ptr<Spacetime> spacetime;
      std::map<std::uint64_t, std::uint64_t> vertexMap;
    };

    /// Load a live, skeleton-complete complex from explicit top cells and
    /// per-edge complex squared lengths. Shared by geometry-dump rehydration and
    /// the RELABEL rebuild. The geometry is supplied wholesale and read back
    /// through `Spacetime::fromCells`; nothing is constructed.
    ///
    /// `cells` keep their intrinsic vertex order — the stored order carries the
    /// orientation, so it is never sorted. `squaredLengths` maps each
    /// `(min id, max id)` vertex pair to its complex squared length.
    /// `vertexTimes` (may be empty) maps vertex id to recorded time and is
    /// applied before the lengths. `dimensions` is the recorded complex
    /// dimension, passed straight through to `fromCells` and never inferred from
    /// a cell. The facet skeleton is completed with `materializeFacets()` so the
    /// result is immediately readable.
    /// @throws std::invalid_argument if `cells` is empty.
    /// @throws std::out_of_range if a built edge has no recorded squared
    ///   length — a partial metric is never silently defaulted.
    [[nodiscard]] static std::shared_ptr<Spacetime> load(
        const std::vector<std::vector<std::uint64_t>> &cells,
        const std::map<std::pair<std::uint64_t, std::uint64_t>,
                       std::complex<double>> &squaredLengths,
        const std::map<std::uint64_t, double> &vertexTimes, int dimensions);

    /// Load the block-residual sub-complex. `cells` are ambient top cells
    /// already selected by the caller — the strict subset whose vertices all lie
    /// in a provenance region, with no new topology, surgery or dynamics —
    /// re-instantiated through `Spacetime::fromCells` with a uniform metric
    /// (weight 1.0). The uniform metric is part of the definition of the carry
    /// diagnostic: it matches how the drive's `r_U` scored the block, which is
    /// metric-independent by design. The result is identical to
    /// `MultiCobordism::subcomplexWithinVertexSet`, duplicated here so a reader
    /// can score a block without constructing the build driver. The skeleton is
    /// not materialized; the `r_state` read this feeds builds only what it needs.
    /// `dimensions` is the ambient complex's canonical dimension (from
    /// `RegisterContext::dimensions()`), passed straight through to `fromCells`
    /// and never inferred from a cell.
    /// @throws std::invalid_argument if `cells` is empty.
    [[nodiscard]] static std::shared_ptr<Spacetime> subcomplex(
        const std::vector<std::vector<std::uint64_t>> &cells, int dimensions);

    /// A relabeled rebuild of `spacetime` under a random vertex-id permutation,
    /// deterministic given `seed`. The cell enumeration order is shuffled as
    /// well, which catches enumeration-order dependence; the metric and vertex
    /// times carry across. The recorded geometry is read off `spacetime` and
    /// re-loaded under the permutation via `load`. Any genuine relabeling lets
    /// the RELABEL gate compare like with like — the specific permutation is not
    /// part of any reported value — so a `std::mt19937_64` stream suffices.
    [[nodiscard]] static Relabeled relabel(const Spacetime &spacetime,
                                           std::uint64_t seed);
};

}  // namespace tessera::observables

#endif  // TESSERA_OBSERVABLES_LIVE_COMPLEX_H
