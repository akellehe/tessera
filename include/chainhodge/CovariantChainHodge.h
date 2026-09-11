// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_CHAINHODGE_COVARIANTCHAINHODGE_H
#define TESSERA_CHAINHODGE_COVARIANTCHAINHODGE_H

#include <complex>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include "chainhodge/ChainHodge.h"
#include "chainhodge/RieszBand.h"
#include "chainhodge/WhitneyMass.h"
#include "cobordism/Certificate.h"
#include "cobordism/ChainComplex.h"

// === tessera subsystem ns fwd-decls ===
namespace tessera::spacetime { class Spacetime; }

namespace tessera::chainhodge {

/// # Connection
///
/// A \f$ \mathbb{C}^* \f$ connection on the edges of a reference-oriented
/// complex: one link \f$ U_{xy} \in \mathbb{C}^* \f$ per canonical edge
/// \f$ x < y \f$, with \f$ U_{yx} = U_{xy}^{-1} \f$ derived exactly and
/// \f$ U_{xx} = 1 \f$ (specification Def. 5.1). A gauge transformation acts by
/// \f$ U_{xy} \mapsto g_x^{-1} U_{xy} g_y \f$, and the curvature of a triangle
/// \f$ t = [p < q < r] \f$ is the ordered product \f$ \mathcal F_t = U_{rq}U_{qp}U_{pr} \f$.
/// Links are never normalized to \f$ U/|U| \f$ and never conjugated.
class Connection {
 public:
  /// Links on the canonical edges of \p K, in canonical edge order.
  /// @throws std::invalid_argument on a length mismatch or a zero link.
  Connection(const cobordism::ChainComplex &K, std::vector<Complex> links);
  /// The trivial connection \f$ U \equiv 1 \f$.
  [[nodiscard]] static Connection trivial(const cobordism::ChainComplex &K);
  /// Adapter from a spacetime's stored edge phases: the stored phase
  /// \f$ \varphi \f$ is the \f$ \mathbb{C}^* \f$ connection on the edge's
  /// source-to-target orientation, so the canonical link is
  /// \f$ U_{xy} = e^{i\varphi} \f$ when the source is \f$ x < y \f$ and
  /// \f$ e^{-i\varphi} \f$ otherwise. The phase's real part is the compact
  /// angle and its imaginary part the log-scale; both are kept.
  [[nodiscard]] static Connection fromSpacetime(const spacetime::Spacetime &st,
                                                const cobordism::ChainComplex &K);

  [[nodiscard]] const std::vector<Complex> &links() const noexcept { return links_; }
  [[nodiscard]] std::size_t edgeCount() const noexcept { return links_.size(); }
  /// \f$ U_{xy} \f$ for any two vertices of a common simplex.
  /// @throws std::invalid_argument when \f$ (x,y) \f$ is not an edge (and \f$ x \ne y \f$).
  [[nodiscard]] Complex link(std::uint64_t x, std::uint64_t y) const;
  /// \f$ U^{-1} \f$: every link inverted.
  [[nodiscard]] Connection inverse() const;
  /// \f$ U^g \f$: \f$ U_{xy} \mapsto g_x^{-1} U_{xy} g_y \f$ for a vertex map
  /// \p g (every vertex of the complex must be present, nonzero).
  [[nodiscard]] Connection gauge(const std::map<std::uint64_t, Complex> &g) const;
  /// \f$ \mathcal F_t = U_{rq}U_{qp}U_{pr} \f$ for the triangle \f$ [p<q<r] \f$.
  [[nodiscard]] Complex curvature(std::uint64_t p, std::uint64_t q, std::uint64_t r) const;

  /// A closed walk on the 1-skeleton as directed steps \f$ (u \to v) \f$:
  /// consecutive steps chain (the target of one step is the source of the
  /// next) and the last target is the first source, the walk's BASE POINT.
  /// The engine's loop convention (`mesh::Edge::walkLoop`, the cycles of a
  /// `cobordism::MultiCobordism::Marking`): a step \f$ u \to v \f$ traverses
  /// the canonical edge \f$ [\min(u,v) < \max(u,v)] \f$ with sign \f$ +1 \f$
  /// when \f$ u < v \f$ (along the reference orientation) and \f$ -1 \f$
  /// otherwise.
  using Walk = std::vector<std::pair<std::uint64_t, std::uint64_t>>;
  /// The holonomy of a closed walk: the ordered product of the links read
  /// along its steps, \f$ U_{v_0v_1}U_{v_1v_2}\cdots U_{v_{n-1}v_0} \f$ — for a
  /// U(1) connection \f$ e^{i\sum_k \pm\varphi_k} \f$, the Wilson loop in the
  /// walk's direction (`observables::WilsonMode::U1_CONNECTION`). It is
  /// exactly 1 on every closed walk iff the connection is a pure gauge
  /// \f$ U = 1^g \f$.
  /// @throws std::invalid_argument when the walk is empty, does not chain or
  ///   close, or steps across a pair that is not an edge.
  [[nodiscard]] Complex holonomy(const Walk &walk) const;
  /// The TRANSPORTED PERIOD of a 1-cochain over a closed walk.
  ///
  /// WHAT: \p cochain is indexed like `links()` (the canonical edge order), and
  /// its value on an edge \f$ \sigma \f$ is expressed in the frame at the
  /// edge's base vertex \f$ b(\sigma) = \min\sigma \f$ — the convention of
  /// the twisted incidences \f$ (\partial_k^U)_{\tau\sigma} =
  /// (\partial_k)_{\tau\sigma}\,U_{b(\tau)b(\sigma)} \f$ (`CovariantChainHodge`),
  /// under which a link \f$ U_{xy} \f$ carries a value at \f$ y \f$ to
  /// \f$ x \f$. For the walk \f$ v_0 \to v_1 \to \cdots \to v_n = v_0 \f$
  /// with steps across the edges \f$ \sigma_k \f$ and signs \f$ s_k \f$,
  /// \f[
  ///   P = \sum_k s_k\,\bigl(U_{v_0v_1}U_{v_1v_2}\cdots U_{v_{k-1}v_k}\bigr)\,
  ///       U_{v_k\,b(\sigma_k)}\,\omega_{\sigma_k},
  /// \f]
  /// the value in the frame at the base point \f$ v_0 \f$: each step's value
  /// is moved from the edge's base vertex to the step's source
  /// (\f$ U_{v_k b(\sigma_k)} \f$: the identity along the edge's orientation,
  /// the link \f$ U_{v_kv_{k+1}} \f$ against it) and then back along the walk
  /// to \f$ v_0 \f$.
  ///
  /// WHY: with the trivial connection this is the plain signed sum
  /// \f$ \sum_k s_k\,\omega_{\sigma_k} \f$ of
  /// `cobordism::EigenstateSynthesis::periodsOfCochainOverLoops` and
  /// `cobordism::MultiCobordism::monodromy`. Under a pure gauge
  /// \f$ U = 1^g \f$, \f$ U_{xy} = g_x^{-1}g_y \f$, on a cochain of the twisted
  /// kernel (\f$ \omega^U = \rho_1\omega \f$, \f$ \omega^U_\sigma =
  /// g_{b(\sigma)}^{-1}\omega_\sigma \f$) every term collapses to
  /// \f$ g_{v_0}^{-1} s_k\omega_{\sigma_k} \f$: the transported period is
  /// \f$ g_{v_0}^{-1} \f$ times the plain period of the untwisted cochain —
  /// the base-point gauge factor only — so the ratio of the periods over two
  /// walks with a COMMON base point (\f$ \tau = P_B/P_A \f$ of a qubit torus,
  /// the coefficient pair of a state in a period frame) is gauge invariant.
  /// This is the one place the engine takes a period with parallel transport
  /// (`observables::SimplicialQubit`, the qubit spec §16; the period frames
  /// and the monodromy read of the cobordism spec §2, D3).
  /// @throws std::invalid_argument when the cochain has the wrong length or
  ///   the walk is empty, does not chain or close, or steps across a non-edge.
  [[nodiscard]] Complex transportedPeriod(const Eigen::VectorXcd &cochain, const Walk &walk) const;
  /// The steps of a closed walk, given in ANY order, as one closed walk:
  /// Hierholzer's circuit over the multigraph of the directed steps, started
  /// at the first step's source and taking each vertex's outgoing steps in
  /// the given order, so steps that already chain as given come back as
  /// given (the circuit of a simple cycle is its own step list). Empty when
  /// the steps do not form one closed walk: a vertex whose in- and
  /// out-degrees differ, or steps in more than one connected component.
  ///
  /// WHY: a marking's cycles are stated as edge steps whose order is a
  /// convenience (`observables::SimplicialQubit`'s cycles, the qubit spec
  /// §16; `cobordism::MultiCobordism::Marking`), while `transportedPeriod`
  /// walks the cycle in the order it is traversed, so every reader orders
  /// the steps by this one rule and two readers of one marking walk it the
  /// same way.
  [[nodiscard]] static Walk closedWalkOf(const Walk &steps);
  /// The common base point of several closed walks (the qubit spec §16): the
  /// first vertex of `walks[0]` that lies on every other walk, every walk
  /// rotated in place to start there — so that the transported periods of a
  /// twisted section over all of them carry one and the same base-point
  /// gauge factor and their ratios are gauge invariant. None when no vertex
  /// of the first walk lies on all the others (the walks are left as given).
  [[nodiscard]] static std::optional<std::uint64_t> commonBasePoint(std::vector<Walk> &walks);
  /// True when every link has unit modulus (a U(1) connection).
  [[nodiscard]] bool isUnitary(double tolerance = 1e-12) const;

 private:
  std::vector<Complex> links_;
  // The canonical edges (x < y) by index; the connection owns everything it
  // needs and never refers back to the complex it was built over.
  std::map<std::pair<std::uint64_t, std::uint64_t>, int> index_;
  std::vector<std::pair<std::uint64_t, std::uint64_t>> edges_;
  Connection() = default;
  // Checks that a walk is non-empty, chains, closes, and steps along edges;
  // returns the canonical index and sign of every step.
  [[nodiscard]] std::vector<std::pair<int, double>> checkWalk(const Walk &walk, const char *who) const;
};

/// The residuals of the exact properties of specification Prop. 5.1, measured
/// on an instance. Sparse identities are measured on construction; the dense
/// ones ((i) and (v)) on demand below the crossover. Each residual is relative
/// to the norm of the object it tests; quiet NaN means unmeasured.
struct CovarianceCertificate {
  /// (ii) \f$ \|(M_k^U)^T - M_k^{U^{-1}}\| / \|M_k^U\| \f$, max over degrees.
  double transposeMetric{std::numeric_limits<double>::quiet_NaN()};
  /// (ii) \f$ \|(\tilde A_k^U)^T - \tilde A_k^{U^{-1}}\| / \|\tilde A_k^U\| \f$ at the
  /// checked degree (dense; NaN above the crossover).
  double transposePencil{std::numeric_limits<double>::quiet_NaN()};
  /// (iii) covariance of \f$ M_k^U \f$ under a random gauge, max over degrees.
  double covarianceMetric{std::numeric_limits<double>::quiet_NaN()};
  /// (iii) covariance of \f$ \tilde A_k^U \f$ under the same gauge (dense).
  double covariancePencil{std::numeric_limits<double>::quiet_NaN()};
  /// (iv) \f$ \max_t \|\partial_1^U\partial_2^U t - U_{rp}(\mathcal F_t - 1)[r]\| \f$
  /// relative to \f$ \max_t |U_{rp}(\mathcal F_t-1)| + 1 \f$.
  double curvature{std::numeric_limits<double>::quiet_NaN()};
  /// (vi) invariance of \f$ \tilde c^T G_k^U c \f$ under
  /// \f$ \tilde c \mapsto \rho_k^{-1}\tilde c \f$, \f$ c \mapsto \rho_k c \f$ (random vectors).
  double pairingInvariance{std::numeric_limits<double>::quiet_NaN()};
  /// (i) \f$ \|h_k(s,1) - L_k\| / \|L_k\| \f$ when \f$ U = 1 \f$ (dense; NaN otherwise).
  double trivialReduction{std::numeric_limits<double>::quiet_NaN()};
  /// (v) pure-gauge isospectrality: Hausdorff distance between the spectra of
  /// \f$ h_k(s, 1^g) \f$ and \f$ L_k \f$ relative to the spectral radius (dense).
  double pureGaugeIsospectrality{std::numeric_limits<double>::quiet_NaN()};
  std::uint64_t gaugeSeed{0};
  int checkedDegree{1};
};

/// The chain-level pencil's own metric regime, measured (never assumed): the
/// operator is symmetric for the COMPLEX SYMMETRIC chain metric,
/// \f$ M L = (M L)^T \f$, which for a dressed connection is the transpose
/// identity \f$ (\tilde A^U)^T = \tilde A^{U^{-1}} \f$ and
/// \f$ (M^U)^T = M^{U^{-1}} \f$ of Prop. 5.1(ii). When both defects sit within
/// the tolerance the regime is `CertificateRegime::ComplexSymmetricPencil`;
/// otherwise it is `NonNormal`, and the defects say by how much.
struct PencilRegimeCertificate {
  cobordism::CertificateRegime regime{cobordism::CertificateRegime::NonNormal};
  /// \f$ \|(\tilde A^U)^T - \tilde A^{U^{-1}}\| / \|\tilde A^U\| \f$ (the symmetry
  /// defect of \f$ \tilde A \f$ at \f$ U = 1 \f$).
  double symmetryDefect{std::numeric_limits<double>::quiet_NaN()};
  /// \f$ \|(M^U)^T - M^{U^{-1}}\| / \|M^U\| \f$ (the symmetry defect of \f$ M \f$ at
  /// \f$ U = 1 \f$).
  double metricSymmetryDefect{std::numeric_limits<double>::quiet_NaN()};
  bool trivialConnection{false};
  double tolerance{1e-10};
};

/// # CovariantChainHodge
///
/// The covariant one-particle operator of specification §5: the sparse
/// inverse chain metrics dressed by the connection and the incidences twisted
/// by it, with single connection variables and no path,
/// \f[
///   (\partial_k^U)_{\tau\sigma} = (\partial_k)_{\tau\sigma}\,U_{b(\tau)b(\sigma)},\qquad
///   (M_k^U)_{\sigma\tau} = (M_k)_{\sigma\tau}\,U_{b(\sigma)b(\tau)},\qquad
///   b(\sigma) = \min\sigma,
/// \f]
/// the dressed chain metric \f$ G_k^U := (M_k^U)^{-1} \f$ (applied by solves),
/// and
/// \f[
///   h_1(s,U) = M_1^U(\partial_1^{U^{-1}})^T (M_0^U)^{-1}\partial_1^U
///            + \partial_2^U M_2^U(\partial_2^{U^{-1}})^T (M_1^U)^{-1},\qquad
///   \tilde A_1^U = M_1^U A_1^U M_1^U
///     = M_1^U(\partial_1^{U^{-1}})^T (M_0^U)^{-1}\partial_1^U M_1^U
///     + \partial_2^U M_2^U(\partial_2^{U^{-1}})^T ,
/// \f]
/// at every degree by the same rule; \f$ A_1^U x = \lambda G_1^U x \f$ is solved
/// as \f$ \tilde A_1^U z = \lambda M_1^U z \f$, \f$ z = G_1^U x \f$. With
/// \f$ \rho_k(g) = \mathrm{diag}(g_{b(\sigma)}^{-1}) \f$ the exact properties are
/// (i) \f$ h_1(s,1) = L_1 \f$; (ii) \f$ (M_k^U)^T = M_k^{U^{-1}} \f$,
/// \f$ (\tilde A_1^U)^T = \tilde A_1^{U^{-1}} \f$,
/// \f$ h_1(s,U)^T = G_1^{U^{-1}} h_1(s,U^{-1})(G_1^{U^{-1}})^{-1} \f$;
/// (iii) \f$ h_1(s,U^g) = \rho_1(g)h_1(s,U)\rho_1(g)^{-1} \f$ and likewise for
/// \f$ \tilde A_1^U \f$, \f$ M_1^U \f$; (iv) no flatness is required:
/// \f$ \partial_1^U\partial_2^U t = U_{rp}(\mathcal F_t - 1)[r] \f$ for
/// \f$ t = [p<q<r] \f$; (v) pure gauge \f$ U = 1^g \f$ is isospectral to
/// \f$ L_1 \f$; (vi) \f$ \tilde c^T G_k^U c \f$ is invariant under
/// \f$ \tilde c \mapsto \rho_k^{-1}\tilde c \f$, \f$ c \mapsto \rho_k c \f$.
/// The transpose carries \f$ U \f$; a dagger would carry \f$ \bar U \f$, which
/// is not covariant for \f$ g \in \mathbb{C}^* \f$. The sparse identities are
/// measured on every instance (`certificate`); the dense ones on request.
///
/// Under the `GRASSMANN_ALL` preset the sparse chain metric is dressed by the
/// same rule and the pencil is written on chains, as in `ChainHodge`.
class CovariantChainHodge {
 public:
  /// Dress \p base by \p U. The certificate's sparse residuals are measured
  /// here with a deterministic random gauge from \p gaugeSeed.
  CovariantChainHodge(const ChainHodge &base, Connection U, std::uint64_t gaugeSeed = 7,
                      bool measureCertificate = true);

  [[nodiscard]] const ChainHodge &base() const noexcept { return *base_; }
  [[nodiscard]] const Connection &connection() const noexcept { return U_; }
  [[nodiscard]] int dimension() const noexcept { return base_->dimension(); }
  [[nodiscard]] Preset preset() const noexcept { return base_->preset(); }
  [[nodiscard]] const CovarianceCertificate &certificate() const noexcept { return cert_; }

  /// \f$ M_k^U \f$ (Whitney) — the dressed sparse inverse chain metric.
  /// @throws std::logic_error under `GRASSMANN_ALL` (see `ChainHodge::Minv`).
  [[nodiscard]] const SparseMatrix &Minv(int k) const;
  /// The dressed sparse object of the preset, whichever side it is on.
  [[nodiscard]] const SparseMatrix &dressed(int k) const;
  /// \f$ \partial_k^U \f$.
  [[nodiscard]] const SparseMatrix &twistedBoundary(int k) const;
  /// \f$ \partial_k^{U^{-1}} \f$.
  [[nodiscard]] const SparseMatrix &twistedBoundaryDual(int k) const;
  /// \f$ \rho_k(g) \f$ as its diagonal, \f$ g_{b(\sigma)}^{-1} \f$ per \f$ k \f$-cell.
  [[nodiscard]] Eigen::VectorXcd rho(int k, const std::map<std::uint64_t, Complex> &g) const;

  /// \f$ G_k^U c = (M_k^U)^{-1} c \f$ by sparse solve (Whitney) or the dressed
  /// sparse product (Grassmann).
  [[nodiscard]] Eigen::MatrixXcd applyG(int k, const Eigen::MatrixXcd &c) const;
  /// \f$ M_k^U c \f$.
  [[nodiscard]] Eigen::MatrixXcd applyMinv(int k, const Eigen::MatrixXcd &c) const;
  /// \f$ h_k(s,U)\,c \f$ via solves; never formed densely here.
  [[nodiscard]] Eigen::MatrixXcd applyH(int k, const Eigen::MatrixXcd &c) const;
  /// The dense \f$ h_k(s,U) \f$ (below the crossover).
  [[nodiscard]] Eigen::MatrixXcd covariantOperator(int k) const;
  /// \f$ \partial h_k(s,U)/\partial s_e \f$ for the edge at canonical index
  /// \p edgeIndex, dense: the product rule over the dressed metrics (the
  /// dressing is independent of \f$ s \f$, so \f$ \partial M_k^U = \mathrm{dress}(\partial M_k) \f$),
  /// with \f$ \partial (M^U)^{-1} = -(M^U)^{-1}\,\partial M^U\,(M^U)^{-1} \f$ applied by
  /// solves. Below the crossover.
  [[nodiscard]] Eigen::MatrixXcd covariantOperatorDerivative(int k, std::size_t edgeIndex) const;
  /// \f$ \partial h_k(s,U)/\partial\varphi_e \f$ for the multiplicative variation
  /// \f$ U_e = e^{i\varphi_e} \f$ of one link (so \f$ \partial U_e = iU_e \f$ and
  /// \f$ \partial U_e^{-1} = -iU_e^{-1} \f$): every dressed entry whose base-vertex
  /// pair is that edge moves by \f$ \pm i \f$ times itself, in the metrics and in
  /// the twisted incidences alike. Below the crossover.
  [[nodiscard]] Eigen::MatrixXcd covariantOperatorPhaseDerivative(int k, std::size_t edgeIndex) const;
  /// Populate the lazy derivative caches at degree \p k (the sparse metric
  /// factorizations and the dense derivative workspace) so that later `const`
  /// calls only READ them.
  ///
  /// `solveDressed` and `derivativeWorkspace` fill `mutable` slots on first
  /// use, which makes two threads calling any of the derivative entry points
  /// concurrently a data race on the shared instance — the same hazard
  /// `MultiCobordism::step` fixed for the facet lattice with
  /// `materializeFacets`. A per-edge gradient loop calls this ONCE, serially,
  /// before going parallel; after it every slot is non-null and the shared
  /// `Eigen::SparseLU` is only ever solved against, which is thread-safe (the
  /// factors are read-only and the supernodal solve keeps its work buffer on
  /// the stack). Idempotent, and a no-op for a preset or a degree that has no
  /// dense derivative.
  void warmDerivatives(int k) const;
  /// \f$ \partial M_k^U/\partial s_e \f$: the dressed sparse metric derivative
  /// (the dressing is independent of \f$ s \f$).
  [[nodiscard]] SparseMatrix dressedDerivative(int k, std::size_t edgeIndex) const;
  /// \f$ \partial M_k^U/\partial\varphi_e \f$ for \f$ U_e = e^{i\varphi_e} \f$: every
  /// dressed entry whose base-vertex pair is that edge times \f$ \pm i \f$.
  [[nodiscard]] SparseMatrix dressedPhaseDerivative(int k, std::size_t edgeIndex) const;
  /// The dense dressed pencil \f$ (\tilde A_k^U, M_k^U) \f$ on images (Whitney)
  /// or \f$ (A_k^U, G_k^U) \f$ on chains (Grassmann).
  [[nodiscard]] Pencil pencil(int k) const;
  /// \f$ \tilde A_k^U \f$ (Whitney).
  [[nodiscard]] Eigen::MatrixXcd pencilAux(int k) const;
  /// The dense spectrum of the dressed pencil.
  [[nodiscard]] SpectrumRead spectrum(int k) const;
  /// The same base dressed by \f$ U^{-1} \f$.
  [[nodiscard]] CovariantChainHodge dual() const;
  /// The same base dressed by \f$ U^g \f$.
  [[nodiscard]] CovariantChainHodge gauged(const std::map<std::uint64_t, Complex> &g) const;

  /// The pencil resolvent applied to chains, \f$ (\zeta I - h_k)^{-1} c =
  /// M_k^U(\zeta M_k^U - \tilde A_k^U)^{-1} c \f$, through ONE sparse
  /// factorization of the bordered system
  /// \f$ \begin{pmatrix} \zeta M_k^U - \partial_{k+1}^U M_{k+1}^U(\partial_{k+1}^{U^{-1}})^T &
  /// -M_k^U(\partial_k^{U^{-1}})^T \\ -\partial_k^U M_k^U & M_{k-1}^U \end{pmatrix} \f$,
  /// whose Schur complement is \f$ \zeta M_k^U - \tilde A_k^U \f$ (the dense
  /// \f$ \tilde A_k^U \f$ is never formed). Whitney preset only.
  /// @throws std::logic_error under `GRASSMANN_ALL`; std::runtime_error when
  ///   \f$ \zeta \f$ is an eigenvalue (singular bordered system).
  [[nodiscard]] Eigen::MatrixXcd resolvent(int k, Complex zeta, const Eigen::MatrixXcd &c) const;

  /// The Riesz band of the contour (specification §6): \f$ P_C(U) =
  /// \sum_j w_j (\zeta_j I - h_k)^{-1} \f$ by the contour's quadrature rule, one
  /// sparse factorization per node, applied to the identity; the right frame
  /// from the SVD of \f$ P \f$ at the CH tolerance \f$ \kappa\,n\,\epsilon_m\,
  /// \sigma_{\max} \f$; the dual band \f$ \Phi^\vee \f$ from `dual()` on the SAME
  /// contour; \f$ B_C \f$, the left frame (refused by name when
  /// \f$ \sigma_{\min}(B_C) \le \text{isotropyTolerance}\cdot\sigma_{\max}(B_C) \f$,
  /// the isotropic band / exceptional-point indicator), \f$ J \f$, \f$ \Gamma \f$,
  /// and the certificates. Dense in \f$ n_k \f$: below the crossover only.
  [[nodiscard]] Band band(int k, const Contour &contour, double kappa = 10.0,
                          double isotropyTolerance = 1e-10) const;

  /// # The harmonic band without a contour
  ///
  /// The \f$ \lambda = 0 \f$ band of the pencil, read as the null space the
  /// specification prescribes rather than as a contour integral around it.
  ///
  /// The specification (RSF §5, Prop. 2) states both the identification and
  /// the computational path: under the rank conditions (R1)–(R4)
  /// \f$ \ker L_1 = H_1 \f$ with no Jordan block at zero and the
  /// \f$ \lambda = 0 \f$ Riesz projector IS the projector onto \f$ H_1 \f$;
  /// and \f$ H_k = M_k^U \ker S^U \f$ with the sparse stacked matrix
  /// \f[
  ///   S^U = \begin{pmatrix} (\partial_{k+1}^{U^{-1}})^T \\ \partial_k^U M_k^U \end{pmatrix},
  /// \f]
  /// only \f$ (M_0^U)^{-1} \f$ applied, by sparse factorization. This is the
  /// dressed form of `ChainHodge::harmonicChains`, which reads the same null
  /// space for the undressed operator.
  ///
  /// WHY it is not the same work as `band`: a contour reading factorizes the
  /// bordered system once per node and applies it to the whole identity, for
  /// \f$ U \f$ and again for \f$ U^{-1} \f$, then takes the SVD of an
  /// \f$ n\times n \f$ projector. That is the general machinery for a band
  /// of a non-normal pencil ANYWHERE in the plane. The harmonic band is not
  /// anywhere: it is at zero, where the invariant subspace is a kernel.
  ///
  /// The conditions are not free. (R1)–(R4) hold automatically only for
  /// Euclidean data; for complex data they fail on a complex-codimension-one
  /// set, and off that set the two readings part company. The certificate
  /// carries the numerical rank of \f$ S^U \f$, the tolerance that decided
  /// it, and the singular gap \f$ \varsigma_r/\varsigma_{r+1} \f$, so the
  /// margin is reported rather than assumed. A nullity that differs between
  /// \f$ U \f$ and \f$ U^{-1} \f$ is refused by name, exactly as `band`
  /// refuses a dual band of a different rank on the same contour.
  ///
  /// \p forceSparse takes the sparse rank-revealing QR below the crossover
  /// too (the gap goes unmeasured there, as in `ChainHodge::harmonicChains`).
  [[nodiscard]] HarmonicRead harmonicChains(int k, double kappa = 10.0,
                                            bool forceSparse = false) const;
  /// The \f$ \lambda = 0 \f$ band from `harmonicChains`, carrying everything
  /// `band` carries: \f$ \Phi \f$ on chains, \f$ \Phi^\vee \f$ from the
  /// dual connection's own null space, \f$ Z = G_k^U\Phi \f$ (the kernel
  /// vectors themselves), \f$ B_C \f$, the left frame under the same
  /// isotropy refusal, \f$ J \f$, \f$ \Gamma \f$, and the certificates.
  /// The contour on the returned band is empty: the circle has degenerated to
  /// the origin, and `nodeCount` is zero to say so.
  [[nodiscard]] Band harmonicBand(int k, double kappa = 10.0,
                                  double isotropyTolerance = 1e-10,
                                  bool forceSparse = false) const;

  /// The canonical left frame \f$ \tilde\Phi = G_k^{U^{-1}}\Phi^\vee B_C^{-T} \f$ of
  /// a band, recomputed from its dual frame and pairing with the dual
  /// instance's metric (`dual()` of the instance that produced the band).
  /// @throws std::runtime_error for an isotropic band (\f$ \det B_C = 0 \f$).
  [[nodiscard]] static Eigen::MatrixXcd leftFrame(const Band &band,
                                                  const CovariantChainHodge &dualInstance,
                                                  double isotropyTolerance = 1e-10);

  /// The pencil's metric regime at degree \p k (see `PencilRegimeCertificate`):
  /// `ComplexSymmetricPencil` when the transpose identities hold to
  /// \p tolerance, else `NonNormal`. Dense at the degree (below the crossover).
  [[nodiscard]] PencilRegimeCertificate regimeCertificate(int k, double tolerance = 1e-10) const;

  /// Measure the dense identities (i), (ii) on \f$ \tilde A_k^U \f$, (iii) on
  /// \f$ \tilde A_k^U \f$, and (v) at degree \p k and fold them into a copy of
  /// the certificate. Requires the degree below the crossover.
  [[nodiscard]] CovarianceCertificate verify(int k = 1) const;

 private:
  std::shared_ptr<const ChainHodge> base_;
  Connection U_;
  Connection Uinv_;
  std::vector<SparseMatrix> dressed_;        // M_k^U or G_k^U
  std::vector<SparseMatrix> dressedDual_;    // M_k^{U^{-1}}
  std::vector<SparseMatrix> twisted_;        // ∂_k^U
  std::vector<SparseMatrix> twistedDual_;    // ∂_k^{U^{-1}}
  std::vector<std::vector<std::uint64_t>> base_vertex_;  // b(σ) per degree, canonical order
  CovarianceCertificate cert_;

  struct Factorization;
  mutable std::vector<std::shared_ptr<Factorization>> factor_;
  [[nodiscard]] Eigen::MatrixXcd solveDressed(int k, const Eigen::MatrixXcd &rhs) const;
  void measureSparseIdentities(std::uint64_t seed);
  [[nodiscard]] static SparseMatrix dress(const SparseMatrix &M,
                                          const std::vector<std::uint64_t> &baseRow,
                                          const std::vector<std::uint64_t> &baseCol,
                                          const Connection &U);
  // Entries of a dressed matrix whose base-vertex pair is the edge (x,y),
  // times +i when the pair reads (x,y) and -i when it reads (y,x); `dual`
  // flips both signs (the U^{-1} dressing). Zero elsewhere.
  [[nodiscard]] static SparseMatrix phaseDerivative(const SparseMatrix &dressedM,
                                                    const std::vector<std::uint64_t> &baseRow,
                                                    const std::vector<std::uint64_t> &baseCol,
                                                    std::uint64_t x, std::uint64_t y, bool dual);
  struct DerivativeWorkspace;
  mutable std::vector<std::shared_ptr<DerivativeWorkspace>> workspace_;
  // The projector, right frame, and projector certificates of one instance on
  // a contour (no dual, no pairing): what `band` computes for U and for U^{-1}.
  struct ProjectorRead {
    Eigen::MatrixXcd projector;
    Eigen::MatrixXcd frame;
    BandCertificate certificate;
  };
  [[nodiscard]] ProjectorRead projectorOnContour(int k, const Contour &contour, double kappa) const;
  /// \f$ S^U = [(\partial_{k+1}^{U^{-1}})^T;\ \partial_k^U M_k^U] \f$, the
  /// dressed stacked matrix of RSF §5 whose kernel is \f$ G_k^U H_k \f$.
  [[nodiscard]] SparseMatrix stackedMatrix(int k) const;
  /// Everything a band carries beyond its two frames: \f$ B_C \f$, the
  /// isotropy verdict, the left frame, \f$ J \f$, \f$ \Gamma \f$ and the
  /// residual certificates. Shared by `band` and `harmonicBand` so the two
  /// readings cannot drift apart in what they report. \p spectralScale is the
  /// scale the residuals are taken relative to (a contour's largest node
  /// modulus; zero for the harmonic band, whose contour is the origin).
  void completeBand(Band &band, const CovariantChainHodge &dualInstance,
                    double spectralScale, double isotropyTolerance) const;
  [[nodiscard]] const DerivativeWorkspace &derivativeWorkspace(int k) const;
  [[nodiscard]] Eigen::MatrixXcd assembleDerivative(
      int k, const SparseMatrix *dMkm1, const SparseMatrix *dMk, const SparseMatrix *dMkp1,
      const SparseMatrix *dBk, const SparseMatrix *dBkDual, const SparseMatrix *dBkp1,
      const SparseMatrix *dBkp1Dual) const;
};

}  // namespace tessera::chainhodge

#endif  // TESSERA_CHAINHODGE_COVARIANTCHAINHODGE_H
