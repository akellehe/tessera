// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_CHAINHODGE_CHAINHODGE_H
#define TESSERA_CHAINHODGE_CHAINHODGE_H

#include <array>
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

  /// The dense pencil at degree \p k: \f$ (\tilde A_k, M_k) \f$ on images
  /// (Whitney) or \f$ (A_k, G_k) \f$ on chains (Grassmann), formed by solves.
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
  [[nodiscard]] Eigen::MatrixXcd stackedMatrix(int k) const;
};

}  // namespace tessera::chainhodge

#endif  // TESSERA_CHAINHODGE_CHAINHODGE_H
