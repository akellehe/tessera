// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_CHAINHODGE_SPARSEPENCILSOLVER_H
#define TESSERA_CHAINHODGE_SPARSEPENCILSOLVER_H

#include <cstdint>
#include <limits>
#include <vector>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include "chainhodge/SparsePencil.h"
#include "chainhodge/WhitneyMass.h"
#include "cobordism/Certificate.h"

namespace tessera::chainhodge {

/// Controls of `SparsePencilSolver::lowest`. Zero selects the documented
/// automatic value.
struct SparsePencilOptions {
  /// The tolerance the certificate declares for the relative residual.
  double tolerance{1e-10};
  /// Columns per Lanczos block; automatic is `count + max(2, count / 4)`,
  /// capped at the matrix size. A block at least as wide as the largest
  /// multiplicity among the requested eigenvalues captures every copy of a
  /// degenerate one; the automatic width exceeds `count`, which always
  /// suffices. A narrower block can return a level with copies missing.
  int blockSize{0};
  /// Columns the basis may hold before it is compressed to its best Ritz
  /// vectors (a thick restart); automatic is ten blocks, capped at the matrix
  /// size. Storage is one dense \f$ n \times \f$ `maxBasisSize` matrix.
  int maxBasisSize{0};
  /// Block expansions (one factorized solve per block column each) before the
  /// solver returns unconverged.
  int maxIterations{200};
  /// Relative defect \f$ \|X - X^\dagger\|_F / \|X\|_F \f$ above which a
  /// matrix is refused as not Hermitian.
  double hermitianTolerance{1e-10};
  /// Seed of the random start block.
  std::uint64_t seed{20260920};
};

/// The result of `SparsePencilSolver::lowest`.
struct SparsePencilRead {
  /// The eigenvalues, ascending, with their certificate: the residual is
  /// \f$ \max_i \|A z_i - \lambda_i M z_i\|_2 / \|(A - \sigma M) z_i\|_2 \f$
  /// (which is \f$ \|A z_i - \lambda_i M z_i\| / \|A z_i\| \f$ at
  /// \f$ \sigma = 0 \f$ and stays defined for a zero eigenvalue), and the
  /// conditioning is the 1-norm condition estimate of the factorized matrix
  /// \f$ A - \sigma M \f$.
  cobordism::CertifiedVector eigenvalues{};
  /// The eigenvectors \f$ z_i \f$ as columns, in `eigenvalues` order,
  /// \f$ M \f$-orthonormal: \f$ z_i^\dagger M z_j = \delta_{ij} \f$.
  Eigen::MatrixXcd vectors{};
  /// The relative residual of each pair, in `eigenvalues` order.
  std::vector<double> residuals{};
  /// \f$ \max_{ij} |(Z^\dagger M Z - I)_{ij}| \f$.
  double orthonormalityDefect{std::numeric_limits<double>::quiet_NaN()};
  /// \f$ \|A - A^\dagger\|_F / \|A\|_F \f$ and the same for \f$ M \f$.
  double hermitianDefectA{std::numeric_limits<double>::quiet_NaN()};
  double hermitianDefectM{std::numeric_limits<double>::quiet_NaN()};
  /// True when \f$ A - \sigma M \f$ admitted a Cholesky factorization, i.e.
  /// \f$ \sigma \f$ lies below every eigenvalue of the pencil. The returned
  /// eigenvalues are then the lowest of the whole spectrum; otherwise they
  /// are the lowest of those above \f$ \sigma \f$.
  bool shiftBelowSpectrum{false};
  /// Whether every requested pair met the tolerance.
  bool converged{false};
  int blockSize{0};
  int iterations{0};
  int restarts{0};
  /// Solves with the factorized \f$ A - \sigma M \f$ (one per block column
  /// per expansion, plus the condition estimate's).
  int solves{0};
};

/// The effective Betti number of a sparse pencil at a scale
/// (`SparsePencilSolver::effectiveBetti`).
struct EffectiveBettiRead {
  /// \f$ \beta^{\mathrm{eff}}(\epsilon) \f$: the number of eigenvalues in
  /// \f$ (\sigma, \epsilon] \f$, counted with multiplicity.
  int rank{0};
  double epsilon{0.0};
  /// The largest eigenvalue inside the window and the smallest outside it;
  /// quiet NaN when there is none.
  double lastInside{std::numeric_limits<double>::quiet_NaN()};
  double firstOutside{std::numeric_limits<double>::quiet_NaN()};
  /// `firstOutside / lastInside`, the separation of the enclosed band from
  /// the rest of the spectrum: \f$ +\infty \f$ when the band is empty or lies
  /// at zero, quiet NaN when nothing was found outside the window.
  double gap{std::numeric_limits<double>::quiet_NaN()};
  /// True when an eigenvalue outside the window was found (or the whole
  /// spectrum was computed), so that nothing inside it can have been missed.
  bool complete{false};
  /// `complete`, with the shift certified below the spectrum and the
  /// eigenpairs' own certificate holding.
  bool certified{false};
  /// The eigenpairs the count was read from: the enclosed band followed by
  /// what was computed beyond it.
  SparsePencilRead band{};
};

/// # SparsePencilSolver
///
/// The lowest eigenpairs of a sparse Hermitian positive-definite pencil,
/// \f$ A z = \lambda M z \f$ with \f$ A = A^\dagger \f$ and
/// \f$ M = M^\dagger \succ 0 \f$, by block shift-invert Lanczos.
///
/// This is the Hermitian specialization of the Whitney pencil: real positive
/// squared lengths and a unimodular connection, for which the transpose
/// identities \f$ (\tilde A^U)^T = \tilde A^{U^{-1}} \f$ and
/// \f$ (M^U)^T = M^{U^{-1}} \f$ make both matrices Hermitian. Both properties
/// and the positivity of \f$ M \f$ are measured on the input and refused by
/// name when they fail; nothing is symmetrized or regularized.
///
/// With one sparse factorization of \f$ B = A - \sigma M \f$ the operator
/// \f$ B^{-1} M \f$ is self-adjoint in the \f$ M \f$ inner product and has
/// the eigenvalues \f$ \theta = 1/(\lambda - \sigma) \f$, so the eigenvalues
/// nearest above the shift are the largest \f$ \theta \f$ and converge first.
/// The basis is grown a block at a time and kept \f$ M \f$-orthonormal by
/// full reorthogonalization; the projected matrix is formed by explicit
/// projection, so it is the Rayleigh–Ritz matrix of the basis whatever
/// happened to the Krylov structure (a deflated column replaced by a random
/// one, a thick restart). Convergence is decided on the true residuals of
/// the original pencil, and a last Rayleigh–Ritz step on the converged
/// vectors takes the eigenvalues from \f$ (A, M) \f$ themselves.
///
/// \f$ B \f$ is factorized by sparse Cholesky when it is positive definite,
/// which certifies that \f$ \sigma \f$ lies below the spectrum, and by sparse
/// LU otherwise. Below the dense crossover `ChainHodge::spectrum` and
/// `CovariantChainHodge::spectrum` are the reference.
///
/// Reference: Ericsson & Ruhe, "The spectral transformation Lanczos method
/// for the numerical solution of large sparse generalized symmetric eigenvalue
/// problems", Mathematics of Computation 35, 1980.
class SparsePencilSolver {
 public:
  /// The \p count lowest eigenpairs above the shift \p sigma.
  /// @throws std::invalid_argument when the matrices are not square of one
  ///   size, \p count is outside \f$ [1, n] \f$, a matrix is not Hermitian to
  ///   `hermitianTolerance`, or \f$ M \f$ is not positive definite;
  ///   std::runtime_error when the factorization finds \f$ A - \sigma M \f$
  ///   singular (\f$ \sigma \f$ is an eigenvalue). A shift that misses an
  ///   eigenvalue only by rounding is not refused; the certificate's
  ///   conditioning reports it.
  [[nodiscard]] static SparsePencilRead lowest(const SparseMatrix &A, const SparseMatrix &M,
                                               int count, double sigma,
                                               const SparsePencilOptions &options = {});
  /// The effective Betti number at scale \p epsilon: the rank of the spectral
  /// band of the pencil in \f$ [0, \epsilon] \f$,
  /// \f$ \beta^{\mathrm{eff}}(\epsilon) = \mathrm{rank}\, P_{[0,\epsilon]} \f$,
  /// read at a size where the dense Riesz band of `CovariantChainHodge::band`
  /// cannot be formed. It is a property of the declared operator and not of
  /// the incidence of the complex: on a torus with incidence Betti number
  /// \f$ \beta_0 = 1 \f$ the covariant degree-zero operator of a connection
  /// with nontrivial holonomy has no near-kernel, and
  /// \f$ \beta_0^{\mathrm{eff}}(\epsilon) = 0 \f$ below its lowest level; at
  /// the trivial connection \f$ \beta_0^{\mathrm{eff}} \f$ counts the
  /// effective components, the pieces joined only through bottlenecks of
  /// conductance below the scale.
  ///
  /// Pairs are computed upward from the shift \p sigma, which must lie below
  /// the spectrum for the count to be the whole band (the read says whether it
  /// does), starting from \p initialCount and doubling until an eigenvalue
  /// outside the window appears. The band is certified by that eigenvalue and
  /// by the gap it leaves.
  /// @throws as `lowest`; std::invalid_argument when \p epsilon is not above
  ///   \p sigma.
  [[nodiscard]] static EffectiveBettiRead effectiveBetti(const SparseMatrix &A, const SparseMatrix &M,
                                                         double epsilon, double sigma,
                                                         const SparsePencilOptions &options = {},
                                                         int initialCount = 8);
  /// `lowest` on a `SparsePencil`.
  [[nodiscard]] static SparsePencilRead lowest(const SparsePencil &pencil, int count, double sigma,
                                               const SparsePencilOptions &options = {});
};

}  // namespace tessera::chainhodge

#endif  // TESSERA_CHAINHODGE_SPARSEPENCILSOLVER_H
