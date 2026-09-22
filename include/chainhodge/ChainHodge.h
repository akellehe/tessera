// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_CHAINHODGE_CHAINHODGE_H
#define TESSERA_CHAINHODGE_CHAINHODGE_H

#include <array>
#include <chrono>
#include <complex>
#include <cstddef>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include "chainhodge/WhitneyMass.h"
#include "cobordism/ChainComplex.h"

namespace tessera::chainhodge {

/// Which vector a pencil's eigenproblem \f$ A x = \lambda B x \f$ is written in.
enum class PencilVariable {
  /// Geometric images \f$ z = G_k h \f$ (the Whitney preset: \f$ (\tilde A_k, M_k) \f$).
  GeometricImage,
  /// Chains \f$ h \f$ (the Grassmann preset: \f$ (A_k, G_k) \f$).
  Chain,
};

/// A complex symmetric pencil \f$ A - \lambda B \f$ at one degree, dense.
struct Pencil {
  int degree{0};
  PencilVariable variable{PencilVariable::GeometricImage};
  Eigen::MatrixXcd A{};
  Eigen::MatrixXcd B{};
};

/// The harmonic space at one degree: the kernel of the stacked matrix
/// \f$ S \f$, the harmonic chains, and the rank certificate of the kernel
/// computation.
struct HarmonicRead {
  int degree{0};
  /// Harmonic chains \f$ H_k \f$ (\f$ n_k \times \text{nullity} \f$).
  Eigen::MatrixXcd chains{};
  /// Their geometric images \f$ G_k H_k \f$ — for the Whitney preset these are
  /// the kernel vectors of \f$ S \f$ themselves (orthonormal columns).
  Eigen::MatrixXcd images{};
  int nullity{0};
  /// Numerical rank of \f$ S \f$ and the tolerance that decided it.
  int rank{0};
  double tolerance{0.0};
  /// \f$ \varsigma_r/\varsigma_{r+1} \f$: last kept over first discarded singular
  /// value of \f$ S \f$ (\f$ +\infty \f$ when nothing was discarded; quiet NaN
  /// when the sparse path measured no singular values).
  double gap{std::numeric_limits<double>::infinity()};
  /// Whether the dense SVD (true) or the sparse rank-revealing QR (false)
  /// computed the kernel.
  bool dense{true};
};

/// # SparseCostReport
///
/// What one operation of the sparse production path cost, as the scaling
/// verification plan's P series asks for it: wall time, memory and fill-in
/// against the number of cells. Every field is measured on the operation that
/// produced the report; nothing is estimated from a model.
///
/// *Fill-in* is the ratio of the stored entries of the factorization to the
/// stored entries of the matrix that was factorized. It is one when the factors
/// are as sparse as the matrix and grows with the ordering's failure to avoid
/// new nonzeros; it is the quantity that decides whether a nominally sparse
/// path is a sparse path at all.
///
/// *Memory* is reported twice, for two different questions.
/// `factorMegabytes` is the memory the factors themselves occupy, computed
/// exactly from their stored entries: one complex scalar and one storage index
/// per entry. It is deterministic and comparable across machines.
/// `residentMegabytes` is the change in the process's resident set size across
/// the operation, read from the operating system; it includes the allocator's
/// own behaviour and every temporary the operation made, and it is quiet NaN
/// where the operating system does not publish it.
struct SparseCostReport {
  /// What was measured: "bordered-lu", "stacked-qr", "contour-band" or
  /// "pencil-apply".
  std::string operation{};
  int degree{0};
  /// \f$ n_k \f$, the number of cells of the degree the operation ran at.
  int dimension{0};
  /// Rows and stored entries of the matrix the factorization was taken of.
  long long systemRows{0};
  long long systemNonZeros{0};
  /// Stored entries of the factors: \f$ \mathrm{nnz}(L) + \mathrm{nnz}(U) \f$
  /// for an LU, \f$ \mathrm{nnz}(R) \f$ for a QR. Zero when the operation took
  /// no factorization.
  long long factorNonZeros{0};
  /// `factorNonZeros / systemNonZeros`; quiet NaN when nothing was factorized.
  double fillIn{std::numeric_limits<double>::quiet_NaN()};
  /// Wall-clock seconds of the whole operation, factorization and solves.
  double wallSeconds{0.0};
  /// Memory of the factors, from `factorNonZeros`; quiet NaN when nothing was
  /// factorized.
  double factorMegabytes{std::numeric_limits<double>::quiet_NaN()};
  /// Change in the process's resident set size across the operation; quiet NaN
  /// where the operating system does not publish it.
  double residentMegabytes{std::numeric_limits<double>::quiet_NaN()};
  /// Right-hand sides the factorization was applied to.
  long long rightHandSides{0};
};

/// # SparseCostMeter
///
/// A running measurement of one operation of the sparse production path.
/// Construct it immediately before the work, call `finish` immediately after
/// with the sizes the work involved, and the returned `SparseCostReport`
/// carries the elapsed wall time, the change in resident memory, and the
/// fill-in and factor memory derived from those sizes. It measures; it never
/// decides anything.
class SparseCostMeter {
 public:
  /// Start the clock for \p operation at degree \p degree and dimension
  /// \f$ n_k \f$ = \p dimension.
  SparseCostMeter(std::string operation, int degree, int dimension);
  /// Stop the clock and return the report.
  /// @param systemRows rows of the matrix that was factorized.
  /// @param systemNonZeros its stored entries.
  /// @param factorNonZeros stored entries of the factors; zero when the
  ///   operation took no factorization, in which case the fill-in and factor
  ///   memory are quiet NaN.
  /// @param rightHandSides right-hand sides the factorization was applied to.
  [[nodiscard]] SparseCostReport finish(long long systemRows, long long systemNonZeros,
                                        long long factorNonZeros,
                                        long long rightHandSides) const;
  /// The process's resident set size in megabytes, or quiet NaN where the
  /// operating system does not publish it.
  [[nodiscard]] static double residentMegabytes();

 private:
  std::string operation_;
  int degree_;
  int dimension_;
  double startResident_;
  std::chrono::steady_clock::time_point start_;
};

/// Two sparse blocks stacked one above the other into a single sparse matrix
/// of \p columns columns, without either block being densified. An empty block
/// (zero rows) contributes nothing.
/// @throws std::invalid_argument when a non-empty block does not have
///   \p columns columns.
[[nodiscard]] SparseMatrix stackSparse(const SparseMatrix &top, const SparseMatrix &bottom,
                                       int columns);

/// The null space of a sparse matrix by rank-revealing sparse QR, with the rank
/// and the threshold that decided it, and what the computation cost.
struct SparseKernelRead {
  /// An orthonormal basis of \f$ \ker S \f$ (\f$ n \times (n - r) \f$).
  Eigen::MatrixXcd kernel{};
  /// Numerical rank \f$ r \f$ of \f$ S \f$ and the pivot threshold that decided
  /// it.
  int rank{0};
  double tolerance{0.0};
  SparseCostReport cost{};
};

/// The rank conditions (R1)–(R4) at one degree, measured numerically against
/// the exact integer ranks of the boundary maps.
struct RankReport {
  int degree{0};
  /// Measured ranks of the four products, in the order R1, R2, R3, R4.
  std::array<int, 4> measured{{0, 0, 0, 0}};
  /// The exact ranks they must equal: \f$ \mathrm{rank}\,\partial_{k+1} \f$
  /// (R1, R4) and \f$ \mathrm{rank}\,\partial_k \f$ (R2, R3).
  std::array<int, 4> expected{{0, 0, 0, 0}};
  std::array<bool, 4> holds{{false, false, false, false}};
  /// (R1)–(R2): \f$ C_k = \mathrm{im}\,\partial_k^* \oplus \mathrm{im}\,\partial_{k+1}
  /// \oplus H_k \f$ and \f$ \dim H_k = b_k \f$.
  bool decompositionHolds{false};
  /// (R1)–(R4): \f$ \ker L_k = H_k \f$ with no Jordan block at zero.
  bool kernelIsHarmonic{false};
  double kappa{10.0};
};

/// The dense spectrum of one degree's pencil.
struct SpectrumRead {
  int degree{0};
  /// Eigenvalues sorted by \f$ (\mathrm{Re}, \mathrm{Im}) \f$.
  std::vector<Complex> eigenvalues{};
  /// \f$ \max_i \|A x_i - \lambda_i B x_i\| / \|A\|_F \f$ over the computed pairs.
  double residual{0.0};
  /// The right eigenvectors, columns in `eigenvalues` order, in the pencil's
  /// variable.
  Eigen::MatrixXcd vectors{};
};

/// # ChainHodge
///
/// The chain-level Hodge pencil of a complexified simplicial complex, with the
/// chain metric \f$ G_k = M_k^{-1} \f$ applied only by solves and never formed.
///
/// With the sparse inverse chain metrics \f$ M_k \f$ of `WhitneyMass`,
/// \f[
///   \partial_k^* = M_k \partial_k^T M_{k-1}^{-1},\qquad
///   L_k = \partial_k^*\partial_k + \partial_{k+1}\partial_{k+1}^*
///       = M_k\partial_k^T M_{k-1}^{-1}\partial_k + \partial_{k+1}M_{k+1}\partial_{k+1}^T M_k^{-1},
/// \f]
/// \f[
///   H_k = \{h:\ \partial_k h = 0,\ \partial_{k+1}^T M_k^{-1} h = 0\},\qquad
///   A_k = G_k L_k .
/// \f]
/// Auxiliary solve: with the geometric image \f$ z = G_k h \f$, so
/// \f$ h = M_k z \f$,
/// \f[
///   h \in H_k \iff \partial_{k+1}^T z = 0 \text{ and } \partial_k M_k z = 0,\qquad
///   A_k x = \lambda G_k x \iff \tilde A_k z = \lambda M_k z,\ z = G_k x,
/// \f]
/// \f[
///   \tilde A_k = M_k A_k M_k = M_k\partial_k^T M_{k-1}^{-1}\partial_k M_k
///              + \partial_{k+1}M_{k+1}\partial_{k+1}^T = \tilde A_k^T .
/// \f]
/// \f$ z \f$ is a function of the lengths and the chain, not a new variable;
/// on a boundary circle it is the vector of signed lengths. Computationally it
/// is the variable to solve for: \f$ H_k = M_k \ker S \f$ with the sparse
/// stacked matrix \f$ S = [\partial_{k+1}^T;\ \partial_k M_k] \f$, and only
/// \f$ M_{k-1}^{-1} \f$ is applied, by sparse factorization.
///
/// Rank conditions, with \f$ Z \f$ solving \f$ M_k Z = \partial_{k+1} \f$:
/// (R1) \f$ \mathrm{rank}(\partial_{k+1}^T Z) = \mathrm{rank}\,\partial_{k+1} \f$,
/// (R2) \f$ \mathrm{rank}(\partial_k M_k \partial_k^T) = \mathrm{rank}\,\partial_k \f$,
/// (R3) \f$ \mathrm{rank}(\partial_k^T M_{k-1}^{-1}\partial_k) = \mathrm{rank}\,\partial_k \f$,
/// (R4) \f$ \mathrm{rank}(\partial_{k+1}M_{k+1}\partial_{k+1}^T) = \mathrm{rank}\,\partial_{k+1} \f$.
/// Under (R1)–(R2), \f$ C_k = \mathrm{im}\,\partial_k^* \oplus \mathrm{im}\,\partial_{k+1}
/// \oplus H_k \f$ and \f$ \dim H_k = b_k \f$; under (R1)–(R4), \f$ \ker L_k = H_k \f$
/// with no Jordan block at 0 and the \f$ \lambda = 0 \f$ Riesz projector is the
/// projector onto \f$ H_k \f$. Nothing is regularized: the report states which
/// conditions hold.
///
/// The `GRASSMANN_ALL` preset keeps its sparse object on the other side: its
/// sparse matrix is the chain metric \f$ G_k \f$ itself, so its pencil is
/// written on chains, \f$ (A_k, G_k) \f$ with \f$ A_k = \partial_k^T G_{k-1}\partial_k
/// + G_k\partial_{k+1}G_{k+1}^{-1}\partial_{k+1}^T G_k \f$, and its harmonic
/// chains are \f$ \ker[\partial_k;\ \partial_{k+1}^T G_k] \f$. `Pencil::variable`
/// names which vector an eigenproblem is written in.
///
/// Dense objects (pencils, spectra, the dense SVD kernel) are formed only below
/// the crossover dimension; above it the kernel is computed by sparse
/// rank-revealing QR and dense requests refuse with `std::length_error`.
/// The adjoint is the transpose throughout; no conjugation enters any operator.
///
/// Reference: Eckmann, "Harmonische Funktionen und Randwertaufgaben in einem
/// Komplex", 1944, for the combinatorial Hodge Laplacian.
class ChainHodge {
 public:
  /// Default crossover, mirroring `cobordism::DenseReference`.
  static constexpr int kDefaultCrossoverDimension = 512;

  /// Build over a reference-oriented complex (`ChainComplex::fromTopCells`) at
  /// squared lengths \p s (canonical edge order), assembling every degree's
  /// sparse metric on the declared branch and the instance certificate.
  /// @throws std::invalid_argument as `WhitneyMass::assemble`.
  /// \p epsilon, when given, is the Lorentzian-protocol rotation the squared
  /// lengths were computed at; it is recorded on the certificate.
  ChainHodge(cobordism::ChainComplex K, SquaredLengths s, Preset preset = Preset::L2,
             Branch branch = Branch::Continuation,
             int crossoverDimension = kDefaultCrossoverDimension,
             double epsilon = std::numeric_limits<double>::quiet_NaN());

  [[nodiscard]] const cobordism::ChainComplex &complex() const noexcept { return K_; }
  [[nodiscard]] const SquaredLengths &squaredLengths() const noexcept { return s_; }
  [[nodiscard]] int dimension() const noexcept { return K_.dimension(); }
  [[nodiscard]] Preset preset() const noexcept { return preset_; }
  [[nodiscard]] Branch branch() const noexcept { return branch_; }
  [[nodiscard]] int crossoverDimension() const noexcept { return crossover_; }
  /// The instance certificate of the assembled geometry.
  [[nodiscard]] const InstanceCertificate &certificate() const noexcept { return cert_; }
  /// Number of \f$ k \f$-cells.
  [[nodiscard]] int size(int k) const;

  /// The sparse inverse chain metric \f$ M_k \f$ (Whitney preset).
  /// @throws std::logic_error under `GRASSMANN_ALL`, whose sparse object is
  ///   the chain metric (`chainMetricSparse`), by name.
  [[nodiscard]] const SparseMatrix &Minv(int k) const;
  /// The sparse chain metric \f$ G_k \f$ (Grassmann preset).
  /// @throws std::logic_error under `L2`, whose chain metric is dense and is
  ///   applied by `applyG`, by name.
  [[nodiscard]] const SparseMatrix &chainMetricSparse(int k) const;
  /// The sparse boundary map \f$ \partial_k \f$ (\f$ n_{k-1}\times n_k \f$),
  /// complex-typed for products; empty (\f$ 0\times n_0 \f$) at \f$ k = 0 \f$.
  [[nodiscard]] const SparseMatrix &boundary(int k) const;

  /// \f$ G_k c \f$: the geometric image, by sparse solve \f$ M_k u = c \f$
  /// (Whitney) or by the sparse product (Grassmann).
  [[nodiscard]] Eigen::MatrixXcd applyG(int k, const Eigen::MatrixXcd &c) const;
  /// \f$ M_k c = G_k^{-1} c \f$: sparse product (Whitney) or solve (Grassmann).
  [[nodiscard]] Eigen::MatrixXcd applyMinv(int k, const Eigen::MatrixXcd &c) const;

  /// The pencil operator applied to the columns of \p Z: \f$ \tilde A_k Z \f$
  /// (Whitney) or \f$ A_k Z \f$ (Grassmann), by sparse products and sparse
  /// solves with \f$ M_{k-1} \f$ (Whitney) or \f$ G_{k+1} \f$ (Grassmann). The
  /// dense operator is never formed, so this is the production path's pencil
  /// and it is defined at every size, at and above the crossover included.
  /// @throws std::invalid_argument when \p Z does not have \f$ n_k \f$ rows.
  [[nodiscard]] Eigen::MatrixXcd applyPencilOperator(int k, const Eigen::MatrixXcd &Z) const;
  /// The sparse stacked cochain matrix of degree \p k whose kernel is the
  /// harmonic space: \f$ S = [\partial_{k+1}^T;\ \partial_k M_k] \f$ (Whitney)
  /// or \f$ [\partial_k;\ \partial_{k+1}^T G_k] \f$ (Grassmann). It is
  /// assembled from the sparse boundary maps and the sparse metric alone and is
  /// never densified on the production path.
  [[nodiscard]] SparseMatrix stackedMatrix(int k) const;
  /// The null space of a sparse matrix by rank-revealing sparse QR of
  /// \f$ S^H \f$ (so that \f$ \ker S = (\operatorname{ran} S^H)^{\perp} \f$),
  /// at the pivot threshold
  /// \f$ \kappa\,\max(m,n)\,\epsilon_m\,\max_c\|S_{\cdot c}\| \f$. Neither
  /// \f$ S \f$ nor the orthogonal factor \f$ Q \f$ is formed densely: the
  /// kernel is \f$ Q \f$ applied to the trailing unit vectors, which is
  /// \f$ n \times (n - r) \f$ and no larger.
  /// @param report when non-null, receives the operation's cost.
  /// @throws std::runtime_error when the sparse QR fails, by name.
  [[nodiscard]] static SparseKernelRead sparseNullSpace(const SparseMatrix &S, double kappa,
                                                        SparseCostReport *report = nullptr);
  /// The dense pencil at degree \p k: \f$ (\tilde A_k, M_k) \f$ on images
  /// (Whitney) or \f$ (A_k, G_k) \f$ on chains (Grassmann), formed by solves.
  /// The dense form is a reference for the sparse production path, not the
  /// production path itself; `applyPencilOperator` is.
  /// @throws std::length_error at or above the crossover.
  [[nodiscard]] Pencil pencil(int k) const;
  /// \f$ \tilde A_k = M_k A_k M_k \f$ (Whitney), dense, formed by solves with
  /// \f$ M_{k-1} \f$.
  /// @throws std::logic_error under `GRASSMANN_ALL`; std::length_error at or
  ///   above the crossover.
  [[nodiscard]] Eigen::MatrixXcd pencilAux(int k) const;
  /// The dense Hodge operator \f$ L_k \f$ on chains.
  /// @throws std::length_error at or above the crossover.
  [[nodiscard]] Eigen::MatrixXcd hodgeOperator(int k) const;

  /// Harmonic chains \f$ H_k = M_k \ker S \f$, \f$ S = [\partial_{k+1}^T;\ \partial_k M_k] \f$
  /// (Whitney) or \f$ \ker[\partial_k;\ \partial_{k+1}^T G_k] \f$ (Grassmann).
  /// Below the crossover the kernel is the dense SVD's with tolerance
  /// \f$ \kappa\,\max(m,n)\,\epsilon_m\,\varsigma_{\max} \f$; at or above it, or
  /// when \p forceSparse is set, a sparse rank-revealing QR of \f$ S^T \f$
  /// with the same threshold supplies it (gap unmeasured).
  [[nodiscard]] HarmonicRead harmonicChains(int k, double kappa = 10.0,
                                            bool forceSparse = false) const;
  /// \f$ G_k H \f$ (equals `applyG`).
  [[nodiscard]] Eigen::MatrixXcd geometricImage(int k, const Eigen::MatrixXcd &H) const;
  /// The harmonic Gram \f$ \Phi^T G_k \Phi = Z^T M_k Z \f$ of a harmonic read
  /// (the complex bilinear restriction; its determinant is the isotropy
  /// indicator).
  [[nodiscard]] Eigen::MatrixXcd harmonicGram(const HarmonicRead &read) const;

  /// The rank conditions (R1)–(R4) at degree \p k with numerical ranks at
  /// tolerance \f$ \kappa\,\max(m,n)\,\epsilon_m\,\varsigma_{\max} \f$.
  /// @throws std::length_error at or above the crossover.
  [[nodiscard]] RankReport rankConditions(int k, double kappa = 10.0) const;

  /// Betti numbers over \f$ \mathbb{Q} \f$, exact from the integer incidence
  /// maps, independent of \f$ s \f$.
  [[nodiscard]] std::vector<int> betti() const { return K_.bettiNumbers(); }

  /// The dense spectrum of the degree-\p k pencil (the eigenvalues of
  /// \f$ L_k \f$), sorted by \f$ (\mathrm{Re}, \mathrm{Im}) \f$, with the pencil
  /// residual.
  /// @throws std::length_error at or above the crossover.
  [[nodiscard]] SpectrumRead spectrum(int k) const;

 private:
  cobordism::ChainComplex K_;
  SquaredLengths s_;
  Preset preset_;
  Branch branch_;
  int crossover_;
  InstanceCertificate cert_;
  std::vector<SparseMatrix> sparse_;    // M_k (L2) or G_k (Grassmann), k = 0..d
  std::vector<SparseMatrix> boundary_;  // ∂_k, k = 0..d

  struct Factorization;
  mutable std::vector<std::shared_ptr<Factorization>> factor_;

  void checkDegree(int k) const;
  void requireDense(int n, const char *what) const;
  [[nodiscard]] const Factorization &factorization(int k) const;
  [[nodiscard]] Eigen::MatrixXcd solveSparse(int k, const Eigen::MatrixXcd &rhs) const;
};

}  // namespace tessera::chainhodge

#endif  // TESSERA_CHAINHODGE_CHAINHODGE_H
