// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_CHAINHODGE_RIESZPROJECTOR_H
#define TESSERA_CHAINHODGE_RIESZPROJECTOR_H

#include <vector>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

namespace tessera::chainhodge {

/// # RieszProjectorRead
///
/// The Riesz (spectral) projector \f$ \Pi \f$ of a square matrix onto the
/// generalized eigenspace of a set of its eigenvalues, with an orthonormal
/// basis of its range and the dual left basis, as `rieszProjector` produces
/// them. The generalized eigenspace of a set of eigenvalues is the span of
/// every eigenvector and every Jordan chain belonging to them; the spectral
/// projector onto it is the projector that commutes with the matrix, has that
/// space as its range and has the invariant subspace of the remaining
/// eigenvalues as its kernel.
struct RieszProjectorRead {
  /// \f$ \Pi \f$, \f$ n \times n \f$.
  Eigen::MatrixXcd projector{};
  /// \f$ N \f$, \f$ n \times q \f$ with orthonormal columns spanning
  /// \f$ \operatorname{ran}\Pi \f$: the leading Schur vectors after the
  /// reordering.
  Eigen::MatrixXcd right{};
  /// \f$ N_L \f$, \f$ n \times q \f$, the basis of \f$ \operatorname{ran}\Pi^T \f$
  /// dual to \f$ N \f$ in the transpose pairing, \f$ N_L^T N = I_q \f$, so that
  /// \f$ \Pi = N N_L^T \f$.
  Eigen::MatrixXcd left{};
};

/// The Riesz projector of a square matrix \f$ X = QTQ^H \f$ onto the
/// generalized eigenspace of the eigenvalues flagged in \p enclosed (one flag
/// per diagonal position of \f$ T \f$), from its complex Schur form.
///
/// The form is reordered by unitary swaps of adjacent diagonal entries so that
/// the flagged eigenvalues lead,
/// \f$ T = \begin{pmatrix} T_{11} & T_{12} \\ 0 & T_{22} \end{pmatrix} \f$,
/// the Sylvester equation \f$ T_{11}Y - YT_{22} = T_{12} \f$ is solved by back
/// substitution (its two operands are triangular and share no eigenvalue when
/// the flagged set is a union of whole clusters of equal eigenvalues), and
/// \f$ \Pi = Q\begin{pmatrix} I & Y \\ 0 & 0 \end{pmatrix}Q^H \f$. No
/// eigenvector matrix is inverted, so a Jordan block among the flagged
/// eigenvalues costs nothing in accuracy.
///
/// The caller is responsible for flagging a set that is separated from its
/// complement: a flagged eigenvalue equal to an unflagged one makes the
/// Sylvester operator singular and the projector undefined.
[[nodiscard]] RieszProjectorRead rieszProjector(
    const Eigen::ComplexSchur<Eigen::MatrixXcd> &schur,
    const std::vector<bool> &enclosed);

}  // namespace tessera::chainhodge

#endif  // TESSERA_CHAINHODGE_RIESZPROJECTOR_H
