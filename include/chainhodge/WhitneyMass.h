// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_CHAINHODGE_WHITNEYMASS_H
#define TESSERA_CHAINHODGE_WHITNEYMASS_H

#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include <Eigen/Core>
#include <Eigen/SparseCore>

// === tessera subsystem ns fwd-decls ===
namespace tessera::spacetime { class Spacetime; }
namespace tessera::cobordism { class ChainComplex; }

namespace tessera::chainhodge {

using Complex = std::complex<double>;
using SparseMatrix = Eigen::SparseMatrix<Complex>;
/// Complex squared edge lengths \f$ s_e \f$ in the canonical edge order of the
/// `cobordism::ChainComplex` they belong to (the row order of \f$ \partial_1 \f$'s
/// transpose, i.e. `ChainComplex::kSimplexVertices(1)`).
using SquaredLengths = std::vector<Complex>;

/// Which chain metric family an instance uses.
enum class Preset {
  /// The Whitney (Galerkin) metric: \f$ W_k := G_k = M_k^{-1} \f$ on chains,
  /// with the sparse inverse chain metric \f$ M_k \f$ assembled per top
  /// simplex. The default.
  L2,
  /// The Grassmann projection metric: the sparse *chain* metric
  /// \f$ G_k = \text{multiplicity} \circ \Gamma_k \f$ of blade pairings,
  /// polynomial in \f$ s \f$ and branch-free. A named alternative with a known
  /// deviation: its harmonic representatives depend on the triangulation at
  /// \f$ O(1) \f$ and its per-face block has rank two.
  GRASSMANN_ALL,
};

/// How the one root per top simplex, \f$ \sqrt{\det g_T} \f$, is fixed.
enum class Branch {
  /// Continuation from the unit Euclidean reference simplex along the straight
  /// segment \f$ g_T(t) = (1-t)\,g_{\rm ref} + t\,g_T \f$: the argument of the
  /// degree-\f$ d \f$ polynomial \f$ \det g_T(t) \f$ is tracked exactly through
  /// its roots. A root on the segment itself makes the continuation ambiguous;
  /// the value then falls back to `KontsevichSegal` and the certificate
  /// records the top simplex.
  Continuation,
  /// The Kontsevich–Segal rule: \f$ \sqrt{\det g_T} = \prod_i \sqrt{\lambda_i} \f$
  /// over the eigenvalues of \f$ g_T \f$ with each root principal and the cut
  /// at \f$ \arg\lambda = \pi \f$ resolved to \f$ +i \f$ (the
  /// \f$ e^{-2i\varepsilon} \f$ side). Single-valued on the allowable domain and
  /// uniformly \f$ i\sqrt{|\det g_T|} \f$ on real Lorentzian data.
  KontsevichSegal,
};

/// The certificate carried by every instance: Kontsevich–Segal allowability of
/// every top simplex, the minimal margin
/// \f$ \pi - \sum_i |\arg\lambda_i(g_T)| \f$, and, once the Lorentzian protocol
/// has set it, the rotation \f$ \varepsilon \f$.
struct InstanceCertificate {
  Branch branch{Branch::Continuation};
  /// True when every top simplex has strictly positive margin.
  bool allowable{false};
  /// \f$ \min_T \bigl(\pi - \sum_i |\arg\lambda_i(g_T)|\bigr) \f$; \f$ +\infty \f$
  /// for a complex without top simplices.
  double margin{std::numeric_limits<double>::infinity()};
  /// The margin of each top simplex, in `ChainComplex::orientedTopSimplices()`
  /// order.
  std::vector<double> margins{};
  /// \f$ |T| = \sqrt{\det g_T}/d! \f$ on the declared branch, per top simplex.
  std::vector<Complex> volumes{};
  /// \f$ \det g_T \f$ per top simplex.
  std::vector<Complex> gramDeterminants{};
  /// True when `Branch::Continuation` met a root of \f$ \det g_T(t) \f$ on the
  /// reference segment for some top simplex (listed in
  /// `ambiguousTopSimplices`); those volumes carry the Kontsevich–Segal value.
  bool continuationAmbiguous{false};
  std::vector<std::size_t> ambiguousTopSimplices{};
  /// The Lorentzian-protocol rotation the instance was computed at; quiet NaN
  /// until the protocol sets it.
  double epsilon{std::numeric_limits<double>::quiet_NaN()};
};

/// One top simplex's contribution to \f$ M_k \f$: the dense local block over
/// the \f$ k \f$-faces of \f$ T \f$ and, when requested, its derivative with
/// respect to each of \f$ T \f$'s own squared edge lengths.
struct TopSimplexBlock {
  /// Index of \f$ T \f$ in `ChainComplex::orientedTopSimplices()`.
  std::size_t topIndex{0};
  /// Canonical \f$ C_k \f$ indices of the \f$ k \f$-faces of \f$ T \f$, in the
  /// local (lexicographic on local vertex numbers) order the block rows use.
  std::vector<int> cellIndices{};
  /// Canonical \f$ C_1 \f$ indices of \f$ T \f$'s edges, in local order
  /// \f$ (0,1),(0,2),\dots,(d-1,d) \f$; `derivative[m]` differentiates with
  /// respect to `edgeIndices[m]`.
  std::vector<int> edgeIndices{};
  Eigen::MatrixXcd block{};
  std::vector<Eigen::MatrixXcd> derivative{};
};

/// # WhitneyMass
///
/// The sparse complex symmetric inverse chain metrics \f$ M_k \f$ of the
/// chain-level Whitney Hodge pencil, assembled from the complex squared edge
/// lengths alone.
///
/// For a top simplex \f$ T = [v_0 < \dots < v_d] \f$ with Gram matrix
/// \f$ (g_T)_{ij} = \tfrac12(s_{v_0v_i} + s_{v_0v_j} - s_{v_iv_j}) \f$,
/// \f$ i,j = 1..d \f$,
/// \f[
///   |T| = \frac{\sqrt{\det g_T}}{d!},\qquad
///   \Gamma_{ij} = (g_T^{-1})_{ij}\ (i,j\ge1),\quad
///   \Gamma_{0j} = -\sum_{i\ge1}\Gamma_{ij},\quad
///   \Gamma_{00} = \sum_{i,j\ge1}\Gamma_{ij},\qquad
///   \int_T \lambda_a\lambda_b\,d\mathrm{vol} = |T|\,\frac{1+\delta_{ab}}{(d+1)(d+2)} .
/// \f]
/// With the Whitney forms \f$ w_\sigma = k!\sum_i (-1)^i \lambda_{a_i}
/// d\lambda_{a_0}\wedge\cdots\widehat{d\lambda_{a_i}}\cdots\wedge d\lambda_{a_k} \f$
/// for \f$ \sigma = [a_0 < \dots < a_k] \f$ (so \f$ w_{[v_iv_j]} =
/// \lambda_i d\lambda_j - \lambda_j d\lambda_i \f$ and \f$ w_{[v_iv_jv_k]} =
/// 2(\lambda_i d\lambda_j\wedge d\lambda_k + \lambda_j d\lambda_k\wedge d\lambda_i
/// + \lambda_k d\lambda_i\wedge d\lambda_j) \f$), the mass matrix is the Gram
/// matrix of the Whitney forms in the piecewise-flat metric,
/// \f[
///   (M_k)_{\sigma\tau} = \sum_{T\supset\sigma,\tau} (k!)^2 \sum_{i,j}
///     (-1)^{i+j}\, I^T_{a_ib_j}\,
///     \det\bigl[\Gamma_{c_pe_q}\bigr]_{p,q=1..k},
/// \f]
/// \f$ c = \sigma\setminus a_i \f$, \f$ e = \tau\setminus b_j \f$, which reproduces
/// the closed forms
/// \f$ (M_0)_{vv'} = \sum_T |T|(1+\delta_{vv'})/((d+1)(d+2)) \f$,
/// \f$ (M_1)_{ee'} = \sum_T \tfrac{|T|}{(d+1)(d+2)}[(1+\delta_{ik})\Gamma_{jl}
/// - (1+\delta_{il})\Gamma_{jk} - (1+\delta_{jk})\Gamma_{il} + (1+\delta_{jl})\Gamma_{ik}] \f$
/// and \f$ (M_2)_{tt} = 1/|t| \f$ at \f$ d = 2 \f$, and supplies \f$ M_2 \f$ for
/// \f$ d \ge 3 \f$ by the expansion
/// \f$ \langle d\lambda_a\wedge d\lambda_b, d\lambda_c\wedge d\lambda_e\rangle =
/// \Gamma_{ac}\Gamma_{be} - \Gamma_{ae}\Gamma_{bc} \f$. Each \f$ M_k \f$ is
/// complex symmetric and sparse: \f$ (M_k)_{\sigma\tau} \ne 0 \f$ only if
/// \f$ \sigma\cup\tau \f$ lies in a top simplex. The reference orientation is
/// ascending vertex id (`cobordism::ChainComplex::fromTopCells`).
///
/// The only root is \f$ \sqrt{\det g_T} \f$, one per top simplex, fixed by the
/// declared `Branch`. The inputs are the squared lengths alone; the Whitney
/// polynomials are a construction rule internal to the definition, never
/// stored, evolved, or supplied. No conjugation appears anywhere.
///
/// The derivative \f$ \partial M_k/\partial s_e \f$ is supplied in closed form
/// per top simplex on the same sparsity pattern (from \f$ \partial g_T \f$,
/// \f$ \partial\Gamma = -\Gamma\,\partial g_T\,\Gamma \f$ on the
/// \f$ d\times d \f$ block, and \f$ \partial|T| = \tfrac12|T|\,
/// \mathrm{tr}(g_T^{-1}\partial g_T) \f$); it satisfies the scaling identity
/// \f$ \sum_e s_e\,\partial M_k/\partial s_e = (d/2 - k)\,M_k \f$, which is the
/// validation the tests use instead of finite differences.
///
/// Reference: Whitney, "Geometric Integration Theory", 1957, for the Whitney
/// forms.
class WhitneyMass {
 public:
  /// The oriented complex of a triangulation in the reference orientation
  /// (ascending vertex id): `ChainComplex::fromTopCells` over the sorted
  /// vertex-id tuples of the spacetime's top simplices.
  [[nodiscard]] static cobordism::ChainComplex complexOf(const spacetime::Spacetime &st);

  /// The complex squared edge lengths \f$ s_e = \ell_e^2 \f$ of a spacetime in
  /// the canonical edge order of \p K. Every edge of \p K must exist in the
  /// spacetime.
  /// @throws std::invalid_argument when an edge of \p K has no spacetime edge.
  [[nodiscard]] static SquaredLengths squaredLengthsOf(
      const spacetime::Spacetime &st, const cobordism::ChainComplex &K);

  /// Assemble the sparse inverse chain metric \f$ M_k \f$ (Whitney, `Preset::L2`)
  /// of \p K at squared lengths \p s on the declared branch. Output is
  /// \f$ n_k\times n_k \f$ in the canonical \f$ C_k \f$ order.
  /// @throws std::invalid_argument when \p s does not have one entry per edge
  ///   of \p K or \p k is outside \f$ [0, d] \f$.
  [[nodiscard]] static SparseMatrix assemble(const cobordism::ChainComplex &K,
                                             const SquaredLengths &s, int k,
                                             Branch branch = Branch::Continuation);

  /// Preset dispatch: `Preset::L2` returns `assemble` (the inverse chain metric
  /// \f$ M_k \f$); `Preset::GRASSMANN_ALL` returns `assembleGrassmann` (the
  /// sparse *chain* metric \f$ G_k \f$ of that preset — the two presets expose
  /// their sparse object, which is the inverse metric for Whitney and the
  /// metric itself for Grassmann).
  [[nodiscard]] static SparseMatrix assemble(const cobordism::ChainComplex &K,
                                             const SquaredLengths &s, int k,
                                             Preset preset, Branch branch);

  /// The Grassmann projection chain metric (`Preset::GRASSMANN_ALL`):
  /// \f$ (G_k)_{\sigma\tau} = m_{\sigma\tau}\,\langle\vec\sigma,\vec\tau\rangle \f$
  /// with \f$ m_{\sigma\tau} \f$ the number of simplices of the complex (of every
  /// dimension \f$ \ge k \f$) containing both \f$ \sigma \f$ and \f$ \tau \f$,
  /// and \f$ \langle\vec\sigma,\vec\tau\rangle =
  /// \det[\langle u_{\sigma_0\sigma_i}, u_{\tau_0\tau_j}\rangle]_{i,j=1..k}/(k!)^2 \f$
  /// the blade pairing from the polarization identity
  /// \f$ \langle u_{ab}, u_{cd}\rangle = \tfrac12(s_{bc} + s_{ad} - s_{bd} - s_{ac}) \f$.
  /// Polynomial in \f$ s \f$; no branch is involved.
  [[nodiscard]] static SparseMatrix assembleGrassmann(
      const cobordism::ChainComplex &K, const SquaredLengths &s, int k);

  /// \f$ \min_T \bigl(\pi - \sum_i |\arg\lambda_i(g_T)|\bigr) \f$ over the top
  /// simplices: positive iff every top simplex is Kontsevich–Segal allowable.
  [[nodiscard]] static double allowabilityMargin(const cobordism::ChainComplex &K,
                                                 const SquaredLengths &s);

  /// The full instance certificate: allowability, margins, volumes on
  /// the declared branch, Gram determinants, and continuation ambiguity.
  [[nodiscard]] static InstanceCertificate certificate(
      const cobordism::ChainComplex &K, const SquaredLengths &s,
      Branch branch = Branch::Continuation);

  /// \f$ \sqrt{\det g}/d! \f$ for one Gram matrix on the declared branch.
  /// \p ambiguous (optional) is set when `Branch::Continuation` met a root on
  /// the reference segment and the Kontsevich–Segal value was used instead.
  [[nodiscard]] static Complex volumeOnBranch(const Eigen::MatrixXcd &gram,
                                              Branch branch,
                                              bool *ambiguous = nullptr);

  /// \f$ \pi - \sum_i |\arg\lambda_i(g)| \f$ for one Gram matrix.
  [[nodiscard]] static double marginOf(const Eigen::MatrixXcd &gram);

  /// The per-top-simplex local blocks of \f$ M_k \f$, with derivatives when
  /// \p withDerivative is set. `assemble` is the sum of these blocks scattered
  /// to the canonical indices.
  [[nodiscard]] static std::vector<TopSimplexBlock> topSimplexBlocks(
      const cobordism::ChainComplex &K, const SquaredLengths &s, int k,
      Branch branch = Branch::Continuation, bool withDerivative = false);

  /// \f$ \partial M_k/\partial s_e \f$ for the edge at canonical index
  /// \p edgeIndex, as a sparse matrix on \f$ M_k \f$'s pattern.
  [[nodiscard]] static SparseMatrix assembleDerivative(
      const cobordism::ChainComplex &K, const SquaredLengths &s, int k,
      std::size_t edgeIndex, Branch branch = Branch::Continuation);

  /// The per-edge contractions \f$ c_e = \mathrm{tr}\bigl(X^T\,
  /// (\partial M_k/\partial s_e)\,Y\bigr) \f$ for every edge, from the local
  /// blocks without forming any derivative matrix — the quantity a gradient of
  /// a pencil eigenvalue or residual needs (\f$ z^T\,\partial M\,z \f$ for
  /// \f$ X = Y = z \f$). \p X and \p Y are \f$ n_k\times m \f$ with the same
  /// \f$ m \f$. The pairing is the transpose, never the conjugate.
  [[nodiscard]] static std::vector<Complex> derivativeContraction(
      const cobordism::ChainComplex &K, const SquaredLengths &s, int k,
      const Eigen::MatrixXcd &X, const Eigen::MatrixXcd &Y,
      Branch branch = Branch::Continuation);
};

}  // namespace tessera::chainhodge

#endif  // TESSERA_CHAINHODGE_WHITNEYMASS_H
