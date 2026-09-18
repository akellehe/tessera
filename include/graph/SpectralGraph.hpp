// Copyright (c) 2026 Twin Vector Labs LLC. All rights reserved.
//
// Abstract spectral-graph interface. The inherited Krylov-Lanczos machinery
// is implemented in src/graph/SpectralGraph.cpp.

#pragma once

#include <cstdint>
#include <vector>

// === tessera subsystem ns fwd-decls ===
namespace tessera::mesh {}
namespace tessera::observables {}
namespace tessera::quantum {}
namespace tessera::simulations {}
namespace tessera::spacetime {}
namespace tessera::graph {
using namespace ::tessera::mesh;
using namespace ::tessera::spacetime;
using namespace ::tessera::observables;
using namespace ::tessera::simulations;
using namespace ::tessera::quantum;

/// # SpectralGraph
///
/// A vertex count plus a Laplacian matvec is enough to compute the diagonal of
/// the heat kernel, the return probability
/// \f$ P(\sigma) = (1/|V|)\,\mathrm{Tr}\,e^{-\sigma L} \f$, and the spectral
/// dimension \f$ D_S(\sigma) = -2\,d\log P / d\log\sigma \f$.
///
/// Two derivations share the Krylov-Lanczos sweep, the Padé-13 dense
/// tridiagonal matrix exponential, and the finite-difference \f$ D_S \f$ logic
/// declared below:
///
/// * `tessera::observables::SparseGraph` (`include/observables/SparseGraph.h`)
///   uses the symmetric normalised Laplacian
///   \f$ L_\mathrm{sym} = I - D^{-1/2} A D^{-1/2} \f$.
/// * `tessera::quantum::EmergentGraph` (`include/quantum/Holography.hpp`) uses
///   the weighted Laplacian \f$ L = D - W \f$ on a CSR adjacency.
///
/// Reference: Ambjorn, Jurkiewicz & Loll, arXiv:hep-th/0505113.
class SpectralGraph {
public:
    virtual ~SpectralGraph() = default;

    /// Vertex count. Must match the dimension `applyLaplacian` expects.
    virtual int nVertices() const = 0;

    /// \f$ y \leftarrow L x \f$. `x` must have length `nVertices()`; `y` need
    /// not be sized on entry and is correctly sized on exit.
    virtual void applyLaplacian(std::vector<double> const& x,
                                  std::vector<double>& y) const = 0;

    // ── Concrete, inherited ───────────────────────────────────────────

    /// Diagonal of \f$ e^{-\sigma L} \f$ for each (start vertex, \f$ \sigma \f$)
    /// pair, via Krylov-Lanczos with full Gram-Schmidt re-orthogonalisation.
    /// The projected tridiagonal \f$ T \f$ at each starting vector is
    /// exponentiated with Padé-13 scaling-and-squaring, so that
    /// \f$ [e^{-\sigma T}]_{0,0} \f$ is exactly
    /// \f$ \langle e_v | e^{-\sigma L} | e_v \rangle \f$ to Krylov order.
    ///
    /// Returns a flat row-major matrix of shape
    /// (`starts.size()` × `sigmas.size()`): `out[s * nSigmas + j]` is the
    /// diagonal entry at vertex `starts[s]` for \f$ \sigma_j \f$.
    std::vector<double> diagonalHeatKernel(
        std::vector<int> const& starts,
        std::vector<double> const& sigmas,
        int krylovDim = 30) const;

    /// \f$ P(\sigma) = (1/|V|)\,\mathrm{Tr}\,e^{-\sigma L} \f$, estimated as the
    /// mean diagonal heat-kernel entry over a random `m`-subset of vertices
    /// (Hutchinson-style unbiased estimator on the trace diagonal). With
    /// `m >= n` every vertex is used and the result is exact; the subset is
    /// sampled without replacement. `m <= 0` requests the default
    /// `min(n, 3000)`, which cuts the heat-kernel cost from O(n²) to O(n·m)
    /// at a variance penalty the Savitzky-Golay smoother absorbs.
    ///
    /// `seed` controls the subset-sampling RNG for reproducibility.
    std::vector<double> returnProbability(
        std::vector<double> const& sigmas,
        int krylovDim = 30,
        int m = 0,
        std::uint64_t seed = 0) const;

    /// \f$ D_S(\sigma) = -2\,d\log P / d\log\sigma \f$ via centered finite
    /// differences (one-sided at endpoints). Pure function; the graph instance
    /// is unused. Returns a vector aligned with `sigmas`; entries where
    /// \f$ P \leq 0 \f$ or non-finite are NaN.
    static std::vector<double> spectralDimension(
        std::vector<double> const& sigmas,
        std::vector<double> const& P);

    /// Savitzky-Golay smoothed \f$ D_S(\sigma) \f$: at each interior
    /// \f$ \sigma \f$, fit a local polynomial of order `polyOrder` over a
    /// centered window of `windowSize` \f$ (\log\sigma, \log P) \f$ samples and
    /// read off the slope. Endpoints use a one-sided window.
    /// @throws std::invalid_argument unless `windowSize` is odd and at least
    ///   3, `polyOrder` is at least 1, and `polyOrder + 1 <= windowSize`.
    static std::vector<double> spectralDimensionSmoothed(
        std::vector<double> const& sigmas,
        std::vector<double> const& P,
        int windowSize = 5,
        int polyOrder = 2);
};

} // namespace tessera::graph
