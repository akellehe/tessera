// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_PERIODICKUHNGRID_H
#define TESSERA_PERIODICKUHNGRID_H

#include <array>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "Topology.h"

// === tessera subsystem ns fwd-decls ===
namespace tessera::spacetime {
class Spacetime;

/// # Periodic Kuhn grid
///
/// The Kuhn (staircase) triangulation of the three-torus \f$ T^3 \f$ on an
/// \f$ N_1 \times N_2 \times N_3 \f$ grid, with the squared edge lengths of a
/// flat metric: the periodic analogue of `SimplicialProduct`, and the mesh of
/// a crystal's periodic cell.
///
/// The vertex with grid index \f$ (i, j, l) \f$, \f$ 0 \le i < N_1 \f$,
/// \f$ 0 \le j < N_2 \f$, \f$ 0 \le l < N_3 \f$, has id
/// \f$ i N_2 N_3 + j N_3 + l \f$. Each grid cube is cut into six tetrahedra
/// along its main diagonal, one per permutation \f$ \pi \f$ of the three axes:
/// the tetrahedron on \f$ p_0 = (i,j,l) \f$, \f$ p_1 = p_0 + e_{\pi(1)} \f$,
/// \f$ p_2 = p_1 + e_{\pi(2)} \f$, \f$ p_3 = p_2 + e_{\pi(3)} \f$, every index
/// reduced modulo its \f$ N_a \f$. Adjacent cubes cut their shared face along
/// the same diagonal, so the result is a consistent complex with
/// \f$ 6 N_1 N_2 N_3 \f$ tetrahedra, \f$ 7 N_1 N_2 N_3 \f$ edges, every vertex
/// of degree 14, and incidence Betti numbers \f$ (1, 3, 3, 1) \f$. Those
/// certify the mesh. They are not the harmonic content of an operator on it:
/// that is the effective Betti number, the rank of the operator's spectral
/// band near zero (`chainhodge::SparsePencilSolver::effectiveBetti`), and the
/// covariant operator of a crystal momentum with nontrivial holonomy has none.
///
/// Every edge is an integer vector \f$ n \f$ of the grid: a positive Kuhn edge
/// has \f$ n \in \{0,1\}^3 \setminus \{0\} \f$ (the three axis steps
/// \f$ e_a \f$, the three face diagonals \f$ e_a + e_b \f$, and the body
/// diagonal \f$ e_1 + e_2 + e_3 \f$), and the same edge read from its other
/// end is \f$ -n \f$. \f$ N_a \ge 3 \f$ is required so that two vertices are
/// joined by at most one edge and the unwrapped displacement of an edge is
/// determined by its two vertex ids.
///
/// The geometry is a constant metric. With lattice vectors
/// \f$ a_1, a_2, a_3 \f$ spanning the periodic cell and the Gram matrix
/// \f$ A_{ab} = a_a \cdot a_b \f$, a grid step along axis \f$ a \f$ is
/// \f$ a_a / N_a \f$, the step metric is \f$ g_{ab} = A_{ab} / (N_a N_b) \f$,
/// and the squared length of the edge \f$ n \f$ is \f$ s_e = n^T g\, n \f$.
/// Only the Gram matrix enters; no coordinates are stored.
///
/// A crystal momentum is a flat U(1) connection on this complex. Written in
/// reciprocal coordinates \f$ \kappa = (\kappa_1, \kappa_2, \kappa_3) \f$
/// (\f$ k = \sum_a \kappa_a b_a \f$ with \f$ b_a \cdot a_c = 2\pi\delta_{ac} \f$),
/// the phase of the directed edge \f$ x \to y \f$ with unwrapped displacement
/// \f$ n \f$ is
/// \f[
///   \varphi_{x \to y} = k \cdot \Delta x = 2\pi \sum_a \kappa_a\, n_a / N_a ,
/// \f]
/// which sums to zero around every triangle (the connection is flat) and to
/// \f$ 2\pi\kappa_a = k \cdot a_a \f$ around the fundamental cycle along axis
/// \f$ a \f$. `blochLinks` returns \f$ e^{i\varphi} \f$ per edge in the order
/// the edges are given, which for `cobordism::ChainComplex::kSimplexVertices(1)`
/// is the link list `chainhodge::Connection` takes.
///
/// Reference: Kuhn, "Some combinatorial lemmas in topology", IBM Journal of
/// Research and Development 4, 1960, for the triangulation of the cube.
class PeriodicKuhnGrid : public Topology {
  public:
    /// A cell as its vertex ids.
    using Cell = std::vector<std::uint64_t>;
    /// A closed walk as directed steps \f$ (u \to v) \f$, the convention of
    /// `chainhodge::Connection::holonomy`.
    using Walk = std::vector<std::pair<std::uint64_t, std::uint64_t>>;
    /// A real symmetric \f$ 3 \times 3 \f$ matrix, row by row.
    using Gram = std::array<std::array<double, 3>, 3>;

    /// @param n1,n2,n3 Grid divisions per axis, each \f$ \ge 3 \f$.
    /// @param latticeGram \f$ A_{ab} = a_a \cdot a_b \f$ of the three lattice
    ///   vectors of the periodic cell: symmetric and positive definite.
    /// @throws std::invalid_argument for a division below 3 or a Gram matrix
    ///   that is not symmetric positive definite.
    PeriodicKuhnGrid(int n1, int n2, int n3, const Gram &latticeGram);

    /// The cubic cell of side \p a with \p n divisions per axis:
    /// \f$ A = a^2 I \f$.
    [[nodiscard]] static PeriodicKuhnGrid cubic(int n, double a = 1.0);

    [[nodiscard]] int dimension() const override { return 3; }

    /// Build the grid into \p spacetime and set every edge's length to
    /// \f$ \sqrt{s_e} \f$ (real and positive). `numSimplices` is ignored.
    void build(Spacetime *spacetime, int numSimplices) override;

    /// \f$ (N_1, N_2, N_3) \f$.
    [[nodiscard]] const std::array<int, 3> &divisions() const noexcept { return n_; }
    /// \f$ A_{ab} = a_a \cdot a_b \f$, as given.
    [[nodiscard]] const Gram &latticeGram() const noexcept { return gram_; }
    /// \f$ g_{ab} = A_{ab} / (N_a N_b) \f$.
    [[nodiscard]] Gram stepMetric() const;
    /// \f$ N_1 N_2 N_3 \f$.
    [[nodiscard]] std::size_t vertexCount() const noexcept;

    /// The id of grid index \f$ (i, j, l) \f$, each index reduced modulo its
    /// division (negative indices wrap).
    [[nodiscard]] std::uint64_t vertexId(int i, int j, int l) const;
    /// The grid index \f$ (i, j, l) \f$ of a vertex id.
    /// @throws std::out_of_range for an id at or above `vertexCount()`.
    [[nodiscard]] std::array<int, 3> gridIndex(std::uint64_t id) const;
    /// \f$ (i/N_1, j/N_2, l/N_3) \f$: the vertex position in units of the
    /// lattice vectors.
    [[nodiscard]] std::array<double, 3> fractionalCoordinates(std::uint64_t id) const;

    /// The \f$ 6 N_1 N_2 N_3 \f$ tetrahedra as sorted vertex-id tuples, sorted:
    /// the input of `Spacetime::fromVertexTuples` and
    /// `cobordism::ChainComplex::fromTopCells`.
    [[nodiscard]] std::vector<Cell> cells() const;

    /// The unwrapped integer displacement \f$ n \f$ of the directed edge
    /// \f$ x \to y \f$: in \f$ \{0,1\}^3 \setminus \{0\} \f$ along the Kuhn
    /// orientation and its negative against it.
    /// @throws std::invalid_argument when \f$ (x, y) \f$ is not an edge of
    ///   the grid.
    [[nodiscard]] std::array<int, 3> displacement(std::uint64_t x, std::uint64_t y) const;
    /// \f$ s_e = n^T g\, n \f$ for the edge between \p x and \p y.
    /// @throws std::invalid_argument as `displacement`.
    [[nodiscard]] double squaredLength(std::uint64_t x, std::uint64_t y) const;
    /// `squaredLength` for every edge of \p edges (each a vertex-id pair), in
    /// the given order: with `ChainComplex::kSimplexVertices(1)` these are the
    /// squared lengths in canonical edge order.
    [[nodiscard]] std::vector<std::complex<double>> squaredLengths(
        const std::vector<Cell> &edges) const;

    /// \f$ \varphi_{x \to y} = 2\pi \sum_a \kappa_a n_a / N_a \f$ for the
    /// directed edge \f$ x \to y \f$; antisymmetric under \f$ x \leftrightarrow y \f$.
    /// @throws std::invalid_argument as `displacement`.
    [[nodiscard]] double blochPhase(std::uint64_t x, std::uint64_t y,
                                    const std::array<double, 3> &kappa) const;
    /// \f$ e^{i\varphi_{x \to y}} \f$ for every edge \f$ (x, y) \f$ of
    /// \p edges, read from its first vertex to its second, in the given order.
    [[nodiscard]] std::vector<std::complex<double>> blochLinks(
        const std::vector<Cell> &edges, const std::array<double, 3> &kappa) const;

    /// The closed walk of \f$ N_a \f$ axis steps from \p base along axis
    /// \p axis (0, 1 or 2) and back to \p base around the torus. The holonomy
    /// of `blochLinks(kappa)` over it is \f$ e^{2\pi i \kappa_a} \f$.
    /// @throws std::invalid_argument for an axis outside \f$ \{0,1,2\} \f$;
    ///   std::out_of_range for a base at or above `vertexCount()`.
    [[nodiscard]] Walk fundamentalCycle(int axis, std::uint64_t base = 0) const;

  private:
    std::array<int, 3> n_;
    Gram gram_;
};

} // namespace tessera::spacetime

#endif // TESSERA_PERIODICKUHNGRID_H
