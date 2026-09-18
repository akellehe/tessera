// Copyright (c) 2026 Twin Vector Labs LLC. All rights reserved.
#pragma once

#include <vector>
#include <utility>

namespace tessera {

/// Fruchterman-Reingold spring-electrical graph layout: spring attraction along
/// edges plus Coulomb repulsion between node pairs, with multiplicative step
/// cooling and a node cap on the O(n^2) repulsion for large graphs. All members
/// are static.
class ForceLayout {
public:
    /// Spring-electrical force-directed layout in 3D. Returns a flat row-major
    /// vector of size \a n * 3 holding the (x, y, z) positions.
    ///
    /// @param n           Number of nodes
    /// @param edges       Index pairs (i, j) defining graph edges
    /// @param centerIdx   If >= 0, pin this node at the origin
    /// @param initPos     Initial positions (flat, n*3).  Random if empty.
    /// @param restLengths Per-edge rest lengths (unit if empty)
    /// @param springK     Spring constant for edge attraction
    /// @param repulsionK  Coulomb constant for node repulsion
    /// @param iters       Number of iterations
    /// @param cooling     Multiplicative step-size decay per iteration
    /// @param repulsionCap Max nodes for all-pairs repulsion (limits O(n^2))
    /// @param seed        Random seed for reproducible layouts
    /// @return Flat row-major vector of size n*3 with (x, y, z) positions
    [[nodiscard]] static std::vector<double> layout3D(
        int n,
        const std::vector<std::pair<int, int>> &edges,
        int centerIdx = -1,
        const std::vector<double> &initPos = {},
        const std::vector<double> &restLengths = {},
        double springK = 0.01,
        double repulsionK = 0.5,
        int iters = 300,
        double cooling = 0.995,
        int repulsionCap = 200,
        int seed = 42);

    /// Spring-electrical force-directed layout in 2D.
    ///
    /// Backs both the per-time-slice spacetime render layout and the radial
    /// curvature-slice layout via two optional constraints:
    ///
    ///   * \a targetRadii (length n) — radius-constrained mode.  Each node's
    ///     distance from the origin is pinned to \a targetRadii[i] and only
    ///     the angular coordinate is solved: forces are projected tangentially
    ///     and radii re-snapped every iteration, so nodes slide along circles.
    ///   * \a groups (length n) — group-scoped repulsion.  Coulomb repulsion
    ///     runs only between nodes sharing a group id (e.g. one time slice);
    ///     when empty, repulsion is global (capped).
    ///
    /// The two are independent and may be combined. Returns a flat row-major
    /// vector of size \a n * 2 with the (x, y) positions.
    ///
    /// @param n           Number of nodes
    /// @param edges       Index pairs (i, j) defining graph edges
    /// @param targetRadii Per-node pinned radius (length n enables radial mode)
    /// @param groups      Per-node group id (length n scopes repulsion)
    /// @param centerIdx   If >= 0, pin this node at the origin
    /// @param initPos     Initial positions (flat, n*2).  Random if empty.
    /// @param restLengths Per-edge rest lengths (unit if empty)
    /// @param springK     Spring constant for edge attraction
    /// @param repulsionK  Coulomb constant for node repulsion
    /// @param iters       Number of iterations
    /// @param cooling     Multiplicative step-size decay per iteration
    /// @param repulsionCap Max nodes per repulsion group (limits O(n^2))
    /// @param initialStep Initial per-iteration max displacement
    /// @param seed        Random seed for reproducible layouts
    /// @return Flat row-major vector of size n*2 with (x, y) positions
    [[nodiscard]] static std::vector<double> layout2D(
        int n,
        const std::vector<std::pair<int, int>> &edges,
        const std::vector<double> &targetRadii = {},
        const std::vector<int> &groups = {},
        int centerIdx = -1,
        const std::vector<double> &initPos = {},
        const std::vector<double> &restLengths = {},
        double springK = 0.02,
        double repulsionK = 0.3,
        int iters = 200,
        double cooling = 0.995,
        int repulsionCap = 200,
        double initialStep = 0.5,
        int seed = 42);

private:
    /// Accumulate the pairwise Coulomb repulsion between 2D nodes a and b.
    static void repel2D(std::vector<double> &pos,
                        std::vector<double> &forces,
                        int a, int b, double repulsionK, double eps);
};

} // namespace tessera
