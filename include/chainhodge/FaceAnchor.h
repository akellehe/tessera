// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_CHAINHODGE_FACEANCHOR_H
#define TESSERA_CHAINHODGE_FACEANCHOR_H

#include <complex>
#include <cstddef>
#include <vector>

#include <Eigen/Core>

#include "chainhodge/CovariantChainHodge.h"
#include "chainhodge/WhitneyMass.h"
#include "cobordism/ChainComplex.h"

namespace tessera::chainhodge {

/// One triangle's face block: the \f$ 3\times3 \f$ matrix over the triangle's
/// three edges (canonical \f$ C_1 \f$ indices in `edgeIndices`, in the local
/// order \f$ (v_0v_1), (v_0v_2), (v_1v_2) \f$), its numerical rank, and the
/// preset it was assembled under.
struct FaceBlock {
  /// Index of the triangle in `ChainComplex::kSimplexVertices(2)`.
  std::size_t faceIndex{0};
  std::vector<int> edgeIndices{};
  Eigen::MatrixXcd block{};
  /// Numerical rank at tolerance \f$ \kappa\,3\,\epsilon_m\,\sigma_{\max} \f$
  /// with \f$ \kappa = 10 \f$.
  int rank{0};
  Preset preset{Preset::L2};
};

/// One tetrahedron's block at degree 0: the \f$ 4\times4 \f$ matrix over the
/// tetrahedron's four vertices (canonical \f$ C_0 \f$ indices in
/// `vertexIndices`, in ascending vertex order), its numerical rank, and the
/// preset it was assembled under.
struct TetrahedronBlock {
  /// Index of the tetrahedron in `ChainComplex::kSimplexVertices(3)`.
  std::size_t tetrahedronIndex{0};
  std::vector<int> vertexIndices{};
  Eigen::MatrixXcd block{};
  /// Numerical rank at tolerance \f$ \kappa\,4\,\epsilon_m\,\sigma_{\max} \f$
  /// with \f$ \kappa = 10 \f$.
  int rank{0};
  Preset preset{Preset::L2};
};

/// # FaceAnchor
///
/// The face anchor, realized with the Whitney metric.
///
/// For a nondegenerate triangle \f$ t \f$ the Whitney block
/// \f$ M_1^{(t)}\in\mathbb{C}^{3\times3} \f$ — the contribution of \f$ t \f$ to
/// \f$ M_1 \f$, restricted to its three edges — is nonsingular: the three
/// Whitney 1-forms of \f$ t \f$ are linearly independent functions on
/// \f$ t \f$ and \f$ M_1^{(t)} \f$ is their Gram matrix in a nondegenerate
/// metric. At \f$ d = 2 \f$ the triangle is a top simplex and the block is its
/// local block. At \f$ d\ge3 \f$ the block is the sum of the local \f$ M_1 \f$
/// blocks of every top simplex containing \f$ t \f$, restricted to \f$ t \f$'s
/// three edges: the part of \f$ M_1 \f$'s assembly that \f$ t \f$'s edges
/// receive from the top simplices through \f$ t \f$.
///
/// With the connection, the connection-dressed face endomorphism on chains is
/// \f[
///   \Pi_\tau(U) = G_1^U\,M_1^{(\tau)U}\,G_1^U,\qquad
///   (M_1^{(\tau)U})_{ee'} = (M_1^{(\tau)})_{ee'}\,U_{b(e)b(e')},
/// \f]
/// covariant by property (iii) of `CovariantChainHodge`, applied by solves and
/// never formed densely; and the invariant anchor coordinate of a fiber
/// \f$ Q \f$ with right frame \f$ \Phi_Q \f$ (chains) and dual frame
/// \f$ \Phi_Q^\vee \f$ (the same band for \f$ U^{-1} \f$) is
/// \f[
///   \alpha_\tau = \det\bigl((\Phi_Q^\vee)^T\,\Pi_\tau(U)\,\Phi_Q\bigr)
///               = \det\bigl((Z_Q^\vee)^T\,M_1^{(\tau)U}\,Z_Q\bigr),\qquad
///   Z_Q = G_1^U\Phi_Q,\ \ Z_Q^\vee = G_1^{U^{-1}}\Phi_Q^\vee ,
/// \f]
/// the face block paired through the geometric images (the two forms agree
/// exactly because \f$ (G_1^U)^T = G_1^{U^{-1}} \f$). It is a well-defined
/// complex number, invariant under the paired frame gauge
/// \f$ \Phi\mapsto\Phi g \f$, \f$ \Phi^\vee\mapsto\Phi^\vee g^{-T} \f$ (at
/// \f$ U = 1 \f$ the \f$ O(r,\mathbb{C}) \f$ gauge of one frame) and under a
/// vertex gauge \f$ g \f$ through \f$ \rho_1(g) \f$, and generically nonzero
/// for a rank-three fiber. Under the Grassmann preset the per-face blade
/// block has rank two, so the same determinant vanishes identically for any
/// rank-three fiber. No conjugation appears anywhere; the pairing is the
/// transpose.
///
/// **The tetrahedral anchor at degree 0.** The same construction one degree
/// down and one dimension up: for a nondegenerate tetrahedron \f$ T \f$
/// the Whitney block \f$ M_0^{(T)}\in\mathbb{C}^{4\times4} \f$ over its four
/// vertices is \f$ \mathrm{vol}(T)\,(I+\mathbf 1\mathbf 1^T)/20 \f$ at
/// \f$ d = 3 \f$ (the Gram of the four barycentric functions), rank four, and
/// for \f$ d\ge4 \f$ the sum of the local degree-0 blocks of the top simplices
/// containing \f$ T \f$ restricted to its vertices (the same reading as the
/// face block). With the connection,
/// \f[
///   \Pi_T(U) = G_0^U\,M_0^{(T)U}\,G_0^U,\qquad
///   (M_0^{(T)U})_{vw} = (M_0^{(T)})_{vw}\,U_{vw},
/// \f]
/// since \f$ b(v) = v \f$ for a vertex, and the anchor coordinate of a rank-four
/// fiber is \f$ \alpha_T = \det((Z_Q^\vee)^T M_0^{(T)U} Z_Q) \f$. Because
/// \f$ (G_0^U)^T = G_0^{U^{-1}} \f$ (property (ii) of `CovariantChainHodge`),
/// the left factor of \f$ \Pi_T \f$ is already the transposed dual-side
/// metric: the form is
/// covariant under the transpose pairing and no dagger enters. The Grassmann
/// degree-0 block pairs scalars and has rank one, so the tetrahedral anchor is
/// defined for the Whitney metric only and refuses the Grassmann preset by
/// name. The degree-0 harmonic band at trivial holonomy is the vertex-volume
/// chain \f$ M_0\mathbf 1 \f$, whose geometric image is the constant vector;
/// `flatZeroModeOverlap` reports how much of that mode a band contains (a
/// certificate: the one case in which a fiber component carries no spectral
/// content). Nothing is excluded by it.
class FaceAnchor {
 public:
  // ---- tetrahedral anchor at degree 0 ----

  /// \f$ M_0^{(T)} \f$ of the tetrahedron at canonical index
  /// \p tetrahedronIndex under the Whitney metric (sum over the top simplices
  /// containing \f$ T \f$ of their local degree-0 blocks, restricted to
  /// \f$ T \f$'s vertices; at \f$ d = 3 \f$ exactly its own block).
  /// @throws std::invalid_argument on a bad index, a complex without
  ///   tetrahedra, or a squared-length vector of the wrong size.
  [[nodiscard]] static TetrahedronBlock whitneyTetrahedronBlock(
      const cobordism::ChainComplex &K, const SquaredLengths &s, std::size_t tetrahedronIndex,
      Branch branch = Branch::Continuation);
  /// Every tetrahedron's Whitney block, in canonical tetrahedron order.
  [[nodiscard]] static std::vector<TetrahedronBlock> whitneyTetrahedronBlocks(
      const cobordism::ChainComplex &K, const SquaredLengths &s,
      Branch branch = Branch::Continuation);
  /// The tetrahedron block of the instance's own preset: Whitney for
  /// `Preset::L2`; the Grassmann preset is refused by name (its degree-0 block
  /// has rank one).
  [[nodiscard]] static TetrahedronBlock tetrahedronBlock(const ChainHodge &hodge,
                                                         std::size_t tetrahedronIndex);
  /// \f$ M_0^{(T)U} \f$: the block dressed entrywise by \f$ U_{vw} \f$.
  [[nodiscard]] static Eigen::MatrixXcd dressedTetrahedronBlock(const TetrahedronBlock &block,
                                                                const cobordism::ChainComplex &K,
                                                                const Connection &U);
  /// \f$ \Pi_T(U)\,c = G_0^U\,M_0^{(T)U}\,G_0^U\,c \f$ for chains \p c
  /// (\f$ n_0\times m \f$), by two solves and one \f$ 4\times4 \f$ product.
  [[nodiscard]] static Eigen::MatrixXcd applyTetrahedronEndomorphism(
      const CovariantChainHodge &cov, std::size_t tetrahedronIndex, const Eigen::MatrixXcd &c);
  /// \f$ \alpha_T = \det((Z_Q^\vee)^T M_0^{(T)U} Z_Q) \f$ from the fiber's
  /// degree-0 geometric images (\f$ n_0\times r \f$ each).
  [[nodiscard]] static Complex tetrahedronAnchorCoordinate(const CovariantChainHodge &cov,
                                                           std::size_t tetrahedronIndex,
                                                           const Eigen::MatrixXcd &Zdual,
                                                           const Eigen::MatrixXcd &Z);
  /// The same from the chain frames (images by solves with
  /// \f$ M_0^{U^{-1}} \f$ and \f$ M_0^U \f$).
  [[nodiscard]] static Complex tetrahedronAnchorCoordinateFromChains(
      const CovariantChainHodge &cov, std::size_t tetrahedronIndex,
      const Eigen::MatrixXcd &PhiDual, const Eigen::MatrixXcd &Phi);
  /// \f$ \alpha_T \f$ for every tetrahedron, canonical order.
  [[nodiscard]] static std::vector<Complex> tetrahedronAnchorCoordinates(
      const CovariantChainHodge &cov, const Eigen::MatrixXcd &Zdual, const Eigen::MatrixXcd &Z);
  /// Certificate: how much of the degree-0 flat zero mode a band contains.
  /// The vertex-volume chain \f$ M_0\mathbf 1 \f$ has the constant geometric
  /// image \f$ z_0 = \mathbf 1 \f$; its Riesz projection onto the band with images
  /// \p Z and dual images \p Zdual is \f$ Pz_0 = Z\,B_C^{-1}(Z^\vee)^T M_0^U z_0 \f$
  /// with \f$ B_C = (Z^\vee)^T M_0^U Z \f$, and the overlap is
  /// \f$ |(Pz_0)^T M_0^U (Pz_0)| / |z_0^T M_0^U z_0| \f$: 1 when the band is the
  /// harmonic band at trivial holonomy, exactly 0 for a band of other
  /// eigenvalues (transpose biorthogonality), in between when a lifted zero
  /// mode is mixed in. Transpose pairing throughout; invariant under the
  /// paired frame gauge.
  /// @throws std::runtime_error for an isotropic band (\f$ B_C \f$ singular).
  [[nodiscard]] static double flatZeroModeOverlap(const CovariantChainHodge &cov,
                                                  const Eigen::MatrixXcd &Zdual,
                                                  const Eigen::MatrixXcd &Z);

  /// \f$ M_1^{(t)} \f$ of the triangle at canonical index \p faceIndex under the
  /// Whitney metric (sum over the top simplices containing \f$ t \f$ of their
  /// local blocks, restricted to \f$ t \f$'s edges).
  /// @throws std::invalid_argument on a bad index, a complex without
  ///   triangles, or a squared-length vector of the wrong size.
  [[nodiscard]] static FaceBlock whitneyFaceBlock(const cobordism::ChainComplex &K,
                                                  const SquaredLengths &s, std::size_t faceIndex,
                                                  Branch branch = Branch::Continuation);
  /// Every triangle's Whitney block, in canonical triangle order.
  [[nodiscard]] static std::vector<FaceBlock> whitneyFaceBlocks(const cobordism::ChainComplex &K,
                                                                const SquaredLengths &s,
                                                                Branch branch = Branch::Continuation);
  /// The Grassmann per-face blade block: \f$ \langle u_e, u_{e'}\rangle \f$ of the
  /// triangle's three edge vectors by the polarization identity; rank two for a
  /// nondegenerate triangle.
  [[nodiscard]] static FaceBlock grassmannFaceBlock(const cobordism::ChainComplex &K,
                                                    const SquaredLengths &s, std::size_t faceIndex);
  /// The face block of the instance's own preset: Whitney for `Preset::L2`,
  /// Grassmann for `Preset::GRASSMANN_ALL`.
  [[nodiscard]] static FaceBlock faceBlock(const ChainHodge &hodge, std::size_t faceIndex);

  /// \f$ M_1^{(\tau)U} \f$: the block dressed entrywise by \f$ U_{b(e)b(e')} \f$,
  /// \f$ b(e) = \min e \f$.
  [[nodiscard]] static Eigen::MatrixXcd dressedFaceBlock(const FaceBlock &block,
                                                         const cobordism::ChainComplex &K,
                                                         const Connection &U);

  /// \f$ \Pi_\tau(U)\,c = G_1^U\,M_1^{(\tau)U}\,G_1^U\,c \f$ for chains \p c
  /// (\f$ n_1\times m \f$), by two solves and one \f$ 3\times3 \f$ product; the
  /// block acts on the triangle's three edge coordinates and annihilates the
  /// rest.
  [[nodiscard]] static Eigen::MatrixXcd applyFaceEndomorphism(const CovariantChainHodge &cov,
                                                              std::size_t faceIndex,
                                                              const Eigen::MatrixXcd &c);

  /// \f$ \alpha_\tau = \det\bigl((Z_Q^\vee)^T M_1^{(\tau)U} Z_Q\bigr) \f$ from the
  /// fiber's geometric images \p Zdual (\f$ Z_Q^\vee = G_1^{U^{-1}}\Phi_Q^\vee \f$)
  /// and \p Z (\f$ Z_Q = G_1^U\Phi_Q \f$), each \f$ n_1\times r \f$.
  [[nodiscard]] static Complex anchorCoordinate(const CovariantChainHodge &cov, std::size_t faceIndex,
                                                const Eigen::MatrixXcd &Zdual,
                                                const Eigen::MatrixXcd &Z);
  /// The same from the chain frames \p PhiDual and \p Phi:
  /// \f$ \alpha_\tau = \det((\Phi_Q^\vee)^T\Pi_\tau(U)\Phi_Q) \f$, the images
  /// formed by solves with \f$ M_1^{U^{-1}} \f$ and \f$ M_1^U \f$.
  [[nodiscard]] static Complex anchorCoordinateFromChains(const CovariantChainHodge &cov,
                                                          std::size_t faceIndex,
                                                          const Eigen::MatrixXcd &PhiDual,
                                                          const Eigen::MatrixXcd &Phi);
  /// \f$ \alpha_\tau \f$ for every triangle, canonical order.
  [[nodiscard]] static std::vector<Complex> anchorCoordinates(const CovariantChainHodge &cov,
                                                              const Eigen::MatrixXcd &Zdual,
                                                              const Eigen::MatrixXcd &Z);

  /// Numerical rank of a small dense block at tolerance
  /// \f$ \kappa\,\max(m,n)\,\epsilon_m\,\sigma_{\max} \f$.
  [[nodiscard]] static int numericalRank(const Eigen::MatrixXcd &A, double kappa = 10.0);

 private:
  [[nodiscard]] static std::vector<int> triangleEdgeIndices(const cobordism::ChainComplex &K,
                                                            std::size_t faceIndex);
  [[nodiscard]] static std::vector<int> tetrahedronVertexIndices(const cobordism::ChainComplex &K,
                                                                 std::size_t tetrahedronIndex);
};

}  // namespace tessera::chainhodge

#endif  // TESSERA_CHAINHODGE_FACEANCHOR_H
