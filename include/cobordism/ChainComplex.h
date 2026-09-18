// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_COBORDISM_CHAINCOMPLEX_H
#define TESSERA_COBORDISM_CHAINCOMPLEX_H

#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

// === tessera subsystem ns fwd-decls ===
namespace tessera::spacetime { class Spacetime; }
namespace tessera::cobordism {
using namespace ::tessera::spacetime;

/// # ChainComplex
///
/// The simplicial chain complex of a triangulation: the boundary operators
/// ∂_k : C_k → C_{k-1} over ℤ, the homology invariants derived from them
/// (Betti numbers over ℤ and ℤ/2, torsion coefficients), and the ∂²=0 check.
/// Built from vertex sets alone, independent of geometry.
///
/// Simplices are the face closure of the complex's simplices. Each k-simplex is
/// identified by its sorted vertex-id tuple; its reference orientation is the
/// increasing-vertex-id ordering, so
/// ∂[v_0 < … < v_k] = Σ_i (−1)^i [v_0,…,v̂_i,…,v_k].
///
/// Reference: Horak and Jost, arXiv:1105.2712
class ChainComplex {
  public:
    /// Build the chain complex from a triangulation. Reads vertex sets only; no
    /// geometry required.
    [[nodiscard]] static ChainComplex fromSpacetime(const Spacetime &K);

    /// Build the chain complex of a pure simplicial complex from its top cells
    /// alone, as vertex-id tuples (order within a cell is irrelevant). The face
    /// closure is every subset of every top cell, oriented by ascending vertex
    /// id, so
    /// \f$ \partial[v_0<\dots<v_k] = \sum_i (-1)^i [v_0<\dots\widehat{v_i}\dots<v_k] \f$.
    /// Cells come out in the canonical lexicographic order.
    /// @throws std::invalid_argument on an impure cell list or a degenerate
    ///   cell (a repeated vertex, or a size differing from the other cells).
    [[nodiscard]] static ChainComplex fromTopCells(
        const std::vector<std::vector<std::uint64_t>> &topCells);

    /// Top dimension n (largest k with a k-simplex), or -1 if empty.
    [[nodiscard]] int dimension() const noexcept { return dimension_; }

    /// |C_k|, the number of k-simplices (0 if k out of range).
    [[nodiscard]] std::size_t numSimplices(int k) const noexcept;

    /// f-vector (|C_0|, …, |C_n|).
    [[nodiscard]] const std::vector<std::size_t> &fVector() const noexcept { return counts_; }

    /// Euler characteristic χ = Σ_k (−1)^k |C_k|.
    [[nodiscard]] int eulerCharacteristic() const noexcept;

    /// The boundary matrix ∂_k (rows = |C_{k-1}|, cols = |C_k|), flat row-major.
    /// Entries in {−1, 0, +1}. ∂_0 is empty. Out-of-range k returns an empty matrix.
    [[nodiscard]] const std::vector<long> &boundaryMatrix(int k) const;

    /// Check ∂_{k-1} ∘ ∂_k = 0 for all k (the chain-complex axiom).
    [[nodiscard]] bool boundaryComposesToZero() const;

    /// Whether the Poincaré/Lefschetz dual block decomposition of a pure
    /// \f$ n \f$-complex is a valid cell complex, equivalently whether the
    /// primal is a combinatorial manifold-with-boundary. Cells are sorted
    /// vertex-id tuples. Checks facet coface counts in \f$ \{1, 2\} \f$, that
    /// every entry of a non-empty `facetCells` is carried by some top cell,
    /// ridge links that are single paths or cycles, and at \f$ n = 3 \f$ vertex
    /// links that are 2-spheres (interior) or disks (boundary). Returns
    /// (ok, reason) naming the first violation.
    [[nodiscard]] static std::pair<bool, std::string> dualComplexIsValid(
        const std::vector<std::vector<std::uint64_t>> &topCells, int dim,
        const std::vector<std::vector<std::uint64_t>> &facetCells = {});

    /// Betti numbers b_0..b_n over ℚ (free ranks of H_k):
    /// b_k = |C_k| − rank ∂_k − rank ∂_{k+1}.
    [[nodiscard]] std::vector<int> bettiNumbers() const;

    /// Betti numbers over GF(2): b_k = |C_k| − rank₂ ∂_k − rank₂ ∂_{k+1}.
    [[nodiscard]] std::vector<int> bettiNumbersGF2() const;

    /// Torsion coefficients of \f$ H_k \f$: the invariant factors \f$ > 1 \f$
    /// of \f$ \partial_{k+1} \f$.
    [[nodiscard]] std::vector<long> torsion(int k) const;

    /// The k-simplices as sorted vertex-id tuples, in the canonical order of
    /// \f$ C_k \f$ — the column order of \f$ \partial_{k+1} \f$ and the row
    /// order of \f$ \partial_k \f$. For \f$ k = \f$ dimension() this is
    /// orientedTopSimplices(). Empty when \f$ k \f$ is out of range.
    [[nodiscard]] std::vector<std::vector<std::uint64_t>> kSimplexVertices(int k) const;

    /// The top simplices as sorted vertex-id tuples, in the canonical column
    /// order of \f$ \partial_d \f$ (\f$ d = \f$ dimension()), the ordering
    /// fundamentalClass()'s signs refer to. Empty for the empty complex.
    [[nodiscard]] std::vector<std::vector<std::uint64_t>> orientedTopSimplices() const;

    /// The fundamental class \f$ [W] \in H_d \f$ of a closed oriented
    /// \f$ d \f$-manifold: the per top-simplex orientation signs
    /// \f$ \varepsilon_t = \pm 1 \f$ (indexed as orientedTopSimplices()) making
    /// the top chain \f$ \sum_t \varepsilon_t\, t \f$ a cycle: the
    /// \f$ \pm 1 \f$ generator of \f$ \ker \partial_d \f$, with the overall sign
    /// fixed by making the first nonzero entry \f$ +1 \f$.
    /// @throws std::runtime_error if no such class exists (\f$ \dim \ker
    ///   \partial_d \neq 1 \f$) or \f$ d < 1 \f$.
    [[nodiscard]] std::vector<int> fundamentalClass() const;

    /// The end sign covector: the induced-orientation charge pattern
    /// \f$ \sigma \in \{\pm 1\}^{|\text{holes}|} \f$ of an end surface.
    /// `surfaceCells` are the end's top cells (sorted vertex-id tuples, all of
    /// one dimension); \p holes are the removed cells whose boundary cycles
    /// carry the periods. Their union is oriented as in orientationCovector(),
    /// and \f$ \sigma_k \f$ is the coefficient assigned to \p holes entry
    /// \f$ k \f$, so every closed form's signed periods obey
    /// \f$ \sum_k \sigma_k p_k = 0 \f$ end by end.
    /// @throws std::runtime_error if the cells are not all of one dimension,
    ///   a facet has more than two cofaces (not a pseudomanifold), or the
    ///   orientation propagation contradicts itself (non-orientable).
    [[nodiscard]] static std::vector<int> endSignCovector(
        const std::vector<std::vector<std::uint64_t>> &surfaceCells,
        const std::vector<std::vector<std::uint64_t>> &holes);

    /// The induced-orientation covector of a whole top-cell complex: the
    /// per-cell sign \f$ \varepsilon_t \in \{\pm 1\} \f$ from facet-sharing
    /// propagation. Each `topCells` entry is a sorted vertex-id tuple, all of
    /// one dimension; the returned vector follows the sorted-unique
    /// (lexicographic) order of those cells, the canonical \f$ C_d \f$ column
    /// order of orientedTopSimplices(). Each connected component's
    /// lexicographically smallest cell carries \f$ +1 \f$; facet \f$ j \f$ of a
    /// sorted cell carries boundary sign \f$ (-1)^j \f$ and across an interior
    /// facet \f$ \varepsilon_b = -\varepsilon_a s_a s_b \f$. Boundary facets
    /// impose no constraint, so the complex need not be closed. Empty for the
    /// empty complex.
    /// @throws std::runtime_error if the cells are not all of one dimension, a
    ///   facet has more than two cofaces (not a pseudomanifold), or the
    ///   orientation propagation contradicts itself (non-orientable).
    [[nodiscard]] static std::vector<int> orientationCovector(
        const std::vector<std::vector<std::uint64_t>> &topCells);

    /// The symmetric intersection form \f$ Q_{ij} = \langle \alpha_i \cup
    /// \alpha_j, [K] \rangle \f$ on a basis \f$ \{\alpha_i\} \f$ of the free
    /// part of \f$ H^2 \f$, as a flat row-major \f$ b_2 \times b_2 \f$ matrix.
    /// Defined for a closed oriented 4-manifold; the cup product is the
    /// Alexander–Whitney product evaluated on the fundamental class
    /// \f$ [K] \f$. Empty when \f$ n \neq 4 \f$ or \f$ b_2 = 0 \f$.
    /// @throws std::runtime_error if \f$ n = 4 \f$, \f$ b_2 > 0 \f$, but the
    ///   complex is not closed-orientable (no fundamental class).
    [[nodiscard]] std::vector<double> intersectionForm() const;

    /// Signature \f$ \sigma = b_+ - b_- \f$ of the intersection form
    /// (Sylvester inertia). 0 when \f$ n \neq 4 \f$ or \f$ b_2 = 0 \f$.
    [[nodiscard]] int signature() const;

    /// Mod-2 Stiefel–Whitney numbers of a closed PL \f$ n \f$-manifold:
    /// \f$ \langle w_{i_1}\cdots w_{i_r}, [K] \rangle \in \mathbb{Z}/2 \f$ for
    /// every partition \f$ (i_1,\dots,i_r) \f$ of \f$ n \f$ into positive parts,
    /// keyed by the monomial (e.g. ``"w2^2"``, ``"w1w3"``). Empty when the
    /// complex is empty.
    ///
    /// Computed from \f$ H^*(K;\mathbb{Z}/2) \f$, the Alexander–Whitney cup
    /// product, and the Wu classes \f$ v_k \f$ — defined by
    /// \f$ \langle v_k \cup x, [K]\rangle = \langle \mathrm{Sq}^k x, [K]\rangle \f$
    /// for all \f$ x \in H^{n-k} \f$ — giving \f$ w = \mathrm{Sq}(v) \f$. Covers
    /// the Steenrod squares expressible through the ordinary
    /// (\f$ \cup_0 \f$) product, enough for surfaces and closed orientable
    /// 4-manifolds with \f$ b_1 = 0 \f$.
    /// @throws std::runtime_error if a required Wu or Stiefel–Whitney class
    ///   needs a higher cup-\f$ i \f$ product (\f$ i>0 \f$).
    [[nodiscard]] std::map<std::string, int> stiefelWhitneyNumbers() const;

    /// Rank of \f$ \partial_k \f$ over \f$ \mathbb{Q} \f$, exact (0 if \f$ k \f$ is out of range).
    [[nodiscard]] int rankOfBoundary(int k) const;

    /// Per degree, the sign \f$ D_k[j] = \pm1 \f$ relating each stored
    /// \f$ k \f$-simplex orientation to the reference orientation (ascending
    /// vertex id): the stored boundary operators satisfy
    /// \f$ \partial_k^{\rm stored} = D_{k-1}\,\partial_k^{\rm ref}\,D_k \f$
    /// exactly. A complex built by `fromTopCells` has every sign \f$ +1 \f$.
    /// @throws std::runtime_error if the stored maps are not a signed-permutation
    ///   image of the reference maps (a facet entry missing or of modulus ≠ 1).
    [[nodiscard]] std::vector<std::vector<int>> orientationSigns() const;

  private:
    int dimension_{-1};
    std::vector<std::size_t> counts_{};                 // |C_k|
    std::vector<std::vector<long>> boundary_{};         // boundary_[k] = ∂_k
    // faceVerts_[k][j] = sorted vertex ids of the j-th k-simplex (column j of
    // ∂_{k+1} / row j of ∂_k).
    std::vector<std::vector<std::vector<std::uint64_t>>> faceVerts_{};
    [[nodiscard]] int gf2RankOfBoundary(int k) const;   // rank ∂_k over GF(2)
};

}  // namespace tessera::cobordism

#endif  // TESSERA_COBORDISM_CHAINCOMPLEX_H
