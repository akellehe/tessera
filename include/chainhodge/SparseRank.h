// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_CHAINHODGE_SPARSERANK_H
#define TESSERA_CHAINHODGE_SPARSERANK_H

#include <limits>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include "chainhodge/WhitneyMass.h"

namespace tessera::chainhodge {

/// The singular values on either side of one split of a matrix's singular
/// spectrum \f$ \varsigma_1 \ge \varsigma_2 \ge \dots \f$: the numerical rank
/// decision and the margin it was taken with.
///
/// The split index `at` is the numerical rank for a kernel read and the exact
/// rank a condition requires for a rank condition; when the condition holds
/// the two coincide. By convention \f$ \varsigma_0 = +\infty \f$ and a
/// singular value past the last one (or one that is structurally zero) is
/// \f$ 0 \f$, so `gap` is \f$ +\infty \f$ when nothing lies beyond the split.
struct SingularSplit {
  /// The split index: `sigmaAt` is \f$ \varsigma_{\mathrm{at}} \f$ and
  /// `sigmaNext` is \f$ \varsigma_{\mathrm{at}+1} \f$.
  int at{0};
  /// The numerical rank: the number of singular values above `tolerance`.
  int rank{0};
  /// \f$ \kappa\,\max(m,n)\,\epsilon_m\,\varsigma_1 \f$, the threshold that decided `rank`.
  double tolerance{0.0};
  /// \f$ \varsigma_1 \f$.
  double largest{0.0};
  /// \f$ \varsigma_{\mathrm{at}} \f$ (\f$ +\infty \f$ at `at` = 0).
  double sigmaAt{std::numeric_limits<double>::infinity()};
  /// \f$ \varsigma_{\mathrm{at}+1} \f$ (\f$ 0 \f$ past the last singular value).
  double sigmaNext{0.0};
  /// \f$ \varsigma_{\mathrm{at}}/\varsigma_{\mathrm{at}+1} \f$ (\f$ +\infty \f$ when
  /// `sigmaNext` is zero).
  double gap{std::numeric_limits<double>::infinity()};
  /// Whether a dense SVD (true) or the sparse factorizations (false) measured it.
  bool dense{true};
};

/// The kernel of a sparse matrix with the certificate of its rank decision.
struct SparseKernel {
  /// Split at the numerical rank: `sigmaAt` is the last kept and `sigmaNext`
  /// the first discarded singular value.
  SingularSplit split{};
  /// An orthonormal basis of the numerical kernel (\f$ n \times (n - r) \f$).
  Eigen::MatrixXcd basis{};
};

/// # SparseRank
///
/// Numerical ranks of sparse matrices with the singular values on either side
/// of the decision, measured without forming a dense matrix of the size of the
/// problem.
///
/// `kernel` reads the kernel of a sparse \f$ A \f$ (\f$ m \times n \f$) from a
/// thresholded sparse QR, \f$ A P = Q \begin{pmatrix} R_{11} & R_{12} \\ 0 & R_{22}
/// \end{pmatrix} \f$ with \f$ R_{22} \f$ below the threshold and dropped. The
/// kernel is \f$ P \begin{pmatrix} -R_{11}^{-1}R_{12} \\ I \end{pmatrix} \f$,
/// orthonormalized, and \f$ Q \f$ is never formed. The nonzero singular values
/// of \f$ A \f$ are those of \f$ W = (R_{11}\ R_{12}) \f$ up to
/// \f$ \|R_{22}\| \f$; the last kept one, \f$ \varsigma_r = \sigma_{\min}(W) \f$,
/// comes from inverse subspace iteration on \f$ W W^\dagger =
/// R_{11}(I + C C^\dagger)R_{11}^\dagger \f$ with \f$ C = R_{11}^{-1}R_{12} \f$
/// (triangular solves and one small Hermitian solve), each Ritz value evaluated
/// as \f$ \|W^\dagger u\| \f$ so that nothing is squared. The first discarded
/// one is bounded by \f$ \varsigma_{r+1} \le \|A N\|_2 \f$ for the orthonormal
/// kernel basis \f$ N \f$ (Courant–Fischer), which is the measured value.
/// The threshold is the dense policy's, \f$ \kappa\,\max(m,n)\,\epsilon_m\,\varsigma_1 \f$,
/// with \f$ \varsigma_1 \f$ from block power iteration. The QR is thresholded on
/// the norm of what remains of each column, not strongly rank-revealing: a
/// singular value within a small factor of the tolerance can be counted on the
/// other side than a dense SVD would count it. The split reports it either way
/// (a kept value near the tolerance shows as a small gap).
///
/// `congruence` reads a matrix of the form \f$ P = C^T X C \f$ or
/// \f$ P = C^T X^{-1} C \f$ with \f$ C \f$ an integer matrix of known exact
/// rank \f$ \rho \f$ and \f$ X \f$ sparse, complex symmetric and nonsingular:
/// the products whose ranks the rank conditions (R1)–(R4) compare with the
/// ranks of the boundary maps. Its kernel contains \f$ \ker C \f$ exactly, so
/// its nonzero singular values are those of the \f$ \rho \times \rho \f$ matrix
/// \f$ T = W \tilde X^{\pm1} W^T \f$, where \f$ C^T\Pi = Q\begin{pmatrix} W \\ 0
/// \end{pmatrix} \f$ is a sparse QR of \f$ C^T \f$ (\f$ Q \f$ real orthogonal)
/// and \f$ \tilde X = \Pi^T X \Pi \f$. \f$ T^{-1} \f$ is applied through a sparse
/// LU of \f$ T \f$ (for \f$ X \f$) or of the bordered matrix
/// \f$ \begin{pmatrix} \tilde X & W^T \\ W & 0 \end{pmatrix} \f$ (for
/// \f$ X^{-1} \f$), and the smallest singular values of \f$ T \f$ come from
/// inverse subspace iteration. The split is at \f$ \rho \f$:
/// \f$ \varsigma_\rho \f$ is the smallest singular value the condition requires
/// to be nonzero, and \f$ \varsigma_{\rho+1} \f$, zero in exact arithmetic, is
/// measured as \f$ \|P N_C\|_2 \f$ on the orthonormal basis \f$ N_C \f$ of
/// \f$ \ker C \f$ from the same QR (the rounding level of the product, as a
/// dense SVD of the formed \f$ P \f$ would see it).
///
/// `fromSingularValues` states a dense SVD's singular values in the same form.
///
/// Reference: L. V. Foster and T. A. Davis, "Algorithm 933: Reliable
/// calculation of numerical rank, null space bases, pseudoinverse solutions,
/// and basic solutions using SuiteSparseQR", ACM Trans. Math. Softw. 40 (2013),
/// for singular-value certificates of a sparse rank-revealing QR.
class SparseRank {
 public:
  /// The kernel of \p A with the singular values on either side of its
  /// numerical rank at tolerance \f$ \kappa\,\max(m,n)\,\epsilon_m\,\varsigma_1 \f$.
  /// @throws std::runtime_error when the sparse QR fails.
  [[nodiscard]] static SparseKernel kernel(const SparseMatrix &A, double kappa = 10.0);

  /// The rank of \f$ P = C^T X C \f$ (\p inverse false) or
  /// \f$ P = C^T X^{-1} C \f$ (\p inverse true), \f$ q \times q \f$ for
  /// \f$ C \f$ of size \f$ p \times q \f$, split at the exact rank
  /// \p structuralRank of the integer matrix \f$ C \f$, with numerical rank at
  /// tolerance \f$ \kappa\,q\,\epsilon_m\,\varsigma_1(P) \f$. When a
  /// factorization meets an exactly singular \f$ T \f$, `sigmaAt` is zero and
  /// `rank` is \f$ \rho - 1 \f$ (at most).
  /// @throws std::invalid_argument on mismatched sizes;
  ///   std::logic_error when the sparse QR of the integer matrix does not find
  ///   \p structuralRank.
  [[nodiscard]] static SingularSplit congruence(const SparseMatrix &C, const SparseMatrix &X,
                                                bool inverse, int structuralRank,
                                                double kappa = 10.0);

  /// A dense SVD's singular values \p sv (descending) of an \f$ m \times n \f$
  /// matrix as a split at \p at (the numerical rank when \p at is negative),
  /// with the dense tolerance policy.
  [[nodiscard]] static SingularSplit fromSingularValues(const Eigen::VectorXd &sv, Eigen::Index rows,
                                                        Eigen::Index cols, double kappa, int at = -1);
};

}  // namespace tessera::chainhodge

#endif  // TESSERA_CHAINHODGE_SPARSERANK_H
