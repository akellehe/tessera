// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_CHAINHODGE_SPARSEPENCIL_H
#define TESSERA_CHAINHODGE_SPARSEPENCIL_H

#include "chainhodge/WhitneyMass.h"

namespace tessera::chainhodge {

/// A pencil \f$ A - \lambda M \f$ with both matrices sparse: the generalized
/// eigenproblem \f$ A z = \lambda M z \f$ at a size where a dense `Pencil`
/// cannot be formed.
///
/// The degree-zero Whitney pencil is the case that is sparse as it stands,
/// \f$ \tilde A_0^U = \partial_1^U M_1^U (\partial_1^{U^{-1}})^T \f$ and
/// \f$ M_0^U \f$ (`CovariantChainHodge::sparsePencil`): at \f$ U = 1 \f$ these
/// are the stiffness and mass matrices of piecewise-linear finite elements.
struct SparsePencil {
  int degree{0};
  SparseMatrix A{};
  SparseMatrix M{};
};

/// # SparsePencilComposition
///
/// Block assembly of two sparse pencils: the sparse counterpart of
/// `cobordism::OccupationSpectra::directSum` and `hoppingBlock`, acting on both
/// matrices of the pencil at once.
///
/// Two copies of a complex that share one geometry (two sheets, such as the
/// two spin components of an electron) carry the block-diagonal pencil
/// \f[
///   A = \begin{pmatrix} A_a & 0 \\ 0 & A_b \end{pmatrix},\qquad
///   M = \begin{pmatrix} M_a & 0 \\ 0 & M_b \end{pmatrix},
/// \f]
/// whose spectrum is the union of the two spectra. A coupling between the
/// sheets enters the left-hand matrix only,
/// \f[
///   A = \begin{pmatrix} A_a & C \\ C' & A_b \end{pmatrix},
/// \f]
/// with \f$ M \f$ unchanged. The coupling block is written in the pencil's own
/// variable, as the matrix of a bilinear form between the two sheets: an
/// operator that multiplies by a function \f$ c(x) \f$ enters as its weighted
/// mass matrix \f$ M_0[c] \f$ (`WhitneyMass::assembleVertexPotential`), so a
/// constant coupling \f$ \Delta \f$ between two identical sheets is
/// \f$ C = \Delta\, M \f$ and splits every level \f$ \lambda \f$ into
/// \f$ \lambda \pm \Delta \f$ exactly.
///
/// Sheet \f$ a \f$ occupies the leading indices. Every assembly is exact:
/// entries are copied, nothing is solved or decomposed.
class SparsePencilComposition {
 public:
  /// \f$ (A_a \oplus A_b,\ M_a \oplus M_b) \f$. The degree of the result is
  /// the degree of \p a.
  /// @throws std::invalid_argument when a pencil's two matrices are not square
  ///   of one size.
  [[nodiscard]] static SparsePencil directSum(const SparsePencil &a, const SparsePencil &b);

  /// The coupled pencil with \f$ C = \f$ \p coupling (\f$ n_a \times n_b \f$)
  /// and \f$ C' = \f$ \p couplingReverse (\f$ n_b \times n_a \f$). An empty
  /// (\f$ 0 \times 0 \f$) \p couplingReverse selects \f$ C' = C^\dagger \f$,
  /// the Hermitian coupling; for a pencil that is not Hermitian the reverse
  /// block is independent data and must be passed explicitly.
  /// @throws std::invalid_argument on a dimension mismatch.
  [[nodiscard]] static SparsePencil hoppingBlock(const SparsePencil &a, const SparsePencil &b,
                                                 const SparseMatrix &coupling,
                                                 const SparseMatrix &couplingReverse = SparseMatrix());
};

}  // namespace tessera::chainhodge

#endif  // TESSERA_CHAINHODGE_SPARSEPENCIL_H
